# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Simple visloc script
# --------------------------------------------------------
import argparse
import json
import math
import os
import random
import sys
import warnings

import numpy as np
import torch
from accelerate.utils import set_seed
from fvcore.nn import parameter_count
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from unicorrn.inference.cycle_inference_engine_2d2d import cycle_uniform_grid_inference
from unicorrn.model import build_model
from unicorrn.utils import cartesian_img_coord, safe_load_weights
from unicorrn.utils.config import read_yaml_config

from .visloc import (
    VislocInLoc,
    aggregate_stats,
    colmap_to_opencv_intrinsics,
    export_results,
    get_pose_error,
    opencv_to_colmap_intrinsics,
    rescale_points3d,
    run_pnp,
)

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="visloc dataset to eval"
    )
    parser.add_argument(
        "--model_config", type=str, required=True, help="UniCorrn model config path"
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path of UniCorrn checkpoint"
    )
    parser.add_argument(
        "--exp_name", type=str, required=True, help="Name of the experiment"
    )
    parser.add_argument(
        "--viz_plot",
        action="store_true",
        help="Use this flag to visualize correspondence results.",
    )
    parser.add_argument(
        "--matching_radius_px",
        type=float,
        default=1.0,
        help="matching radius for cycle consistency check",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=str,
        default="1.001",
        help="confidence values lower than threshold are invalid",
    )

    parser.add_argument(
        "--max_image_size",
        type=int,
        default=None,
        help="max image size for the fine resolution",
    )

    parser.add_argument(
        "--coarse_coverage",
        type=float,
        default=0.9,
        help="coarse coverage in coarse_to_fine estimating",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="coarse coverage in coarse_to_fine estimating",
    )
    parser.add_argument(
        "--max_keypoints",
        type=int,
        default=5_000,
        help="coarse coverage in coarse_to_fine estimating",
    )
    parser.add_argument(
        "--subscene_id",
        type=int,
        default=-1,
        help="Subscene ID of InLoc for debugging and visualization.",
    )
    parser.add_argument(
        "--map_id",
        type=int,
        default=-1,
        help="Map ID of InLoc for debugging and visualization.",
    )
    parser.add_argument("--run_id", type=int, default=1, help="pnp run value.")
    parser.add_argument("--grid_size", type=int, default=1, help="pnp run value.")
    parser.add_argument(
        "--resume_index", type=int, default=0, help="resume scene index."
    )
    parser.add_argument("--stop_index", type=int, default=356, help="stop scene index.")
    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument(
        "--pnp_mode",
        type=str,
        default="poselib",
        choices=["cv2", "poselib", "pycolmap"],
        help="pnp lib to use",
    )

    parser_reproj = parser.add_mutually_exclusive_group()
    parser_reproj.add_argument(
        "--reprojection_error", type=float, default=5.0, help="pnp reprojection error"
    )
    parser_reproj.add_argument(
        "--reprojection_error_diag_ratio",
        type=float,
        default=None,
        help="pnp reprojection error as a ratio of the diagonal of the image",
    )

    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=48,
        help="max batch size for inference on crops when using coarse to fine",
    )
    parser.add_argument(
        "--pnp_max_points",
        type=int,
        default=100_000,
        help="pnp maximum number of points kept",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/2d2d/InLoc", help="output path"
    )
    return parser


def plot_keypoints(img, kpts, name):
    kpts = np.round(kpts).astype(np.int32)

    plt.imshow(img)
    plt.scatter(kpts[:, 0], kpts[:, 1], color="green", s=0.5, alpha=0.6)

    plt.savefig(f"{name}_viz_kpts.png", dpi=120, bbox_inches="tight")
    plt.close()


def plot_matches(img1, img2, kpts1, kpts2, name):
    W1, H1 = img1.size
    W2, H2 = img2.size

    viz_img1 = np.array(img1)
    viz_img2 = np.array(img2)
    img1 = np.pad(
        viz_img1, ((0, max(H2 - H1, 0)), (0, 0), (0, 0)), "constant", constant_values=0
    )
    img2 = np.pad(
        viz_img2, ((0, max(H1 - H2, 0)), (0, 0), (0, 0)), "constant", constant_values=0
    )

    img = np.concatenate((img1, img2), axis=1)
    plt.figure()
    plt.imshow(img)
    cmap = plt.get_cmap("jet")

    num_matches = kpts2.shape[0]

    n_viz = num_matches
    viz_matches_query = kpts1.round().astype(np.int32)
    viz_matches_map = kpts2.round().astype(np.int32)

    for i in range(n_viz):
        (x1, y1), (x2, y2) = viz_matches_query[i], viz_matches_map[i]
        plt.plot(
            [x1, x2 + W1],
            [y1, y2],
            "-+",
            linewidth=0.25,
            markersize=2.5,
            alpha=0.25,
            color=cmap(i / n_viz),
            scalex=False,
            scaley=False,
        )

    plt.savefig(f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def resize_image_to_max(max_image_size, rgb, K):
    W, H = rgb.size
    if max_image_size and max(W, H) > max_image_size:
        islandscape = W >= H
        if islandscape:
            WMax = max_image_size
            HMax = int(H * (WMax / W))
        else:
            HMax = max_image_size
            WMax = int(W * (HMax / H))
        # resize_op = tvf.Compose([tvf.Resize(size=[HMax, WMax])])
        resized_rgb = rgb.resize((WMax, HMax), Image.BILINEAR)
        to_orig_max = np.array([[W / WMax, 0, 0], [0, H / HMax, 0], [0, 0, 1]])
        to_resize_max = np.array([[WMax / W, 0, 0], [0, HMax / H, 0], [0, 0, 1]])

        # Generate new camera parameters
        new_K = opencv_to_colmap_intrinsics(K)
        new_K[0, :] *= WMax / W
        new_K[1, :] *= HMax / H
        new_K = colmap_to_opencv_intrinsics(new_K)
    else:
        # rgb_tensor = ImgNorm(rgb).permute(1, 2, 0)
        resized_rgb = rgb.resize((WMax, HMax), Image.BILINEAR)
        to_orig_max = np.eye(3)
        to_resize_max = np.eye(3)
        HMax, WMax = H, W
        new_K = K
    return resized_rgb, new_K, to_orig_max, to_resize_max, (HMax, WMax)


def remove_boundary_keypoints(keypoints, H, W, boundary_distance=5):
    """
    Remove keypoints that are within a specified distance from the image boundary.

    Parameters:
    -----------
    keypoints : numpy.ndarray
        Array of shape (N, 2) containing keypoint coordinates (x, y)
    H : int
        Height of image
    W : int
        Width of image
    boundary_distance : int
        Minimum distance from boundary (default: 5 pixels)

    Returns:
    --------
    numpy.ndarray
        Filtered keypoints array with shape (M, 2) where M <= N
    """
    # If keypoints is empty, return it as is
    if len(keypoints) == 0:
        return keypoints

    # Extract x and y coordinates
    x_coords = keypoints[:, 0]
    y_coords = keypoints[:, 1]

    # Create boolean mask for valid keypoints
    # Keypoints are valid if they are at least boundary_distance pixels away from edges
    valid_mask = (
        (x_coords >= boundary_distance)  # Left boundary
        & (x_coords < W - boundary_distance)  # Right boundary
        & (y_coords >= boundary_distance)  # Top boundary
        & (y_coords < H - boundary_distance)  # Bottom boundary
    )

    # Filter keypoints using the mask
    filtered_keypoints = keypoints[valid_mask]

    return filtered_keypoints


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    conf_thr = (
        float(args.confidence_threshold)
        if args.confidence_threshold != "ninf"
        else float("-inf")
    )
    device = args.device
    pnp_mode = args.pnp_mode

    reprojection_error = args.reprojection_error
    reprojection_error_diag_ratio = args.reprojection_error_diag_ratio
    pnp_max_points = args.pnp_max_points

    cycle_consistency = f"cc_radius{args.matching_radius_px}px"

    args.exp_name = (
        f"{args.exp_name}_grid_size{args.grid_size}_{cycle_consistency}_conf{conf_thr}"
    )

    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    print(f"Evaluation directory path: {args.output_dir}")

    cfg = read_yaml_config(args.model_config)

    # set_seed(0)
    model = build_model(cfg.NAME, cfg=cfg)
    print(f"Model params: {parameter_count(model)[''] / 1e6}M")

    weights = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    safe_load_weights(model, weights["model"])
    print(f"Loaded ckpt from: {args.ckpt_path}\n")
    print("-" * 80)

    dataset = eval(args.dataset)
    dataset.set_resolution(model)

    query_names = []
    poses_pred = []
    pose_errors = []
    angular_errors = []

    print(f"Resume from {args.resume_index}, Stop at {args.stop_index}.")

    if args.viz_plot:
        assert (
            -1 < args.subscene_id < len(dataset)
        ), f"Invalid subscene id {args.subscene_id} for visualization. Expected value in range (0,{len(dataset)-1})."
        assert (
            -1 < args.map_id < 40
        ), f"Invalid map id {args.map_id} for visualization. Expected value in range (0,40)"

    stats = {"DUC1": [], "DUC2": []}

    for idx in tqdm(range(len(dataset))):
        if args.viz_plot and idx != args.subscene_id:
            continue

        if idx < args.resume_index:
            print(f"Skipping {idx}. Resume from {args.resume_index}")
            continue

        if idx == args.stop_index:
            print(f"Stopping at {args.stop_index}")
            break

        views = dataset[(idx)]
        query_view = views[0]  # 0 is the query
        map_views = views[1:]  # rest are mapped images
        query_names.append(query_view["image_name"])

        (
            query_resized_rgb,
            query_K,
            query_to_orig_max,
            query_to_resize_max,
            (HQ, WQ),
        ) = resize_image_to_max(
            args.max_image_size, query_view["rgb"], query_view["intrinsics"]
        )

        query_pts2d = []
        query_pts3d = []

        for i, map_view in enumerate(map_views):
            # debug condition
            if args.viz_plot and i != args.map_id:
                continue

            (map_resized_rgb, map_K, map_to_orig_max, map_to_resize_max, (HM, WM)) = (
                resize_image_to_max(
                    args.max_image_size, map_view["rgb"], map_view["intrinsics"]
                )
            )

            cache_file = None
            if args.output_dir is not None:
                cache_file = os.path.join(
                    args.output_dir,
                    "matches",
                    query_view["image_name"],
                    map_view["image_name"] + ".npz",
                )

            if cache_file is not None and os.path.isfile(cache_file):
                matches = np.load(cache_file)
                valid_pts3d = matches["valid_pts3d"]
                matches_query_img = matches["matches_query_img"]
                matches_map_img = matches["matches_map_img"]
                confidence = matches["confidence"]
                print(
                    f"loaded from cache query {idx} -> map {i}, pts3d: {valid_pts3d.shape}, kpts_query_img: {matches_query_img.shape}, kpts_map_img: {matches_map_img.shape}, confidence: {confidence.shape}"
                )
            else:
                map_img = np.array(map_resized_rgb)

                # rescale pts3d
                valid_all = map_view["valid"]
                pts3d = map_view["pts3d"]

                WM_full, HM_full = map_view["rgb"].size
                if WM_full != WM or HM_full != HM:
                    y_full, x_full = torch.where(valid_all)
                    pos2d_cv2 = (
                        torch.stack([x_full, y_full], dim=-1)
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    sparse_pts3d = pts3d[y_full, x_full].cpu().numpy()
                    _, _, pts3d, valid_all = rescale_points3d(
                        pos2d_cv2, sparse_pts3d, map_to_resize_max, HM, WM
                    )

                map_data = {
                    "pts3d": pts3d,
                    "valid_all": valid_all,
                    "pts3d_rescaled": map_view["pts3d_rescaled"].numpy(),
                    "valid_rescaled": map_view["valid_rescaled"].numpy(),
                    "map_K": map_K,
                    "query_K": query_K,
                }

                # find correspondences
                matches_map_img, matches_query_img, confidence, valid_pts3d = (
                    cycle_uniform_grid_inference(
                        np.array(map_resized_rgb),
                        np.array(query_resized_rgb),
                        model,
                        map_data,
                        grid_size=args.grid_size,
                        matching_radius_px=args.matching_radius_px,
                        unified_model=True,
                    )
                )

                # rescale to map img kpts to original resolution
                W, H = map_view["rgb"].size
                scale_x = W / WM
                scale_y = H / HM
                matches_map_img[..., 0] = matches_map_img[..., 0] * scale_x
                matches_map_img[..., 1] = matches_map_img[..., 1] * scale_y

                # rescale to query img kpts to original resolution
                W, H = query_view["rgb"].size
                scale_x = W / WQ
                scale_y = H / HQ
                matches_query_img[..., 0] = matches_query_img[..., 0] * scale_x
                matches_query_img[..., 1] = matches_query_img[..., 1] * scale_y

                if cache_file is not None:
                    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                    np.savez(
                        cache_file,
                        valid_pts3d=valid_pts3d,
                        matches_query_img=matches_query_img,
                        matches_map_img=matches_map_img,
                        confidence=confidence,
                    )

                # print(f"    map: rgb {map_view['rgb'].size}, resized: {map_resized_rgb.size}, rescaled: {map_view['rgb_rescaled'].shape}, pt3d: {map_view['pts3d_rescaled'].shape}")
                # print(f"    map: rgb {map_view['rgb'].size}, resized: {map_resized_rgb.size}, kpts: {map_kpts.shape} map_pt3d: {map_view['pts3d_rescaled'].shape}, valid_pt3d: {valid_pts3d.shape}, matches query {matches_query_img.shape}, map {matches_map_img.shape}, conf {confidence.shape}")

            # apply confidence
            if len(confidence) > 0 and conf_thr != float("-inf"):
                mask = confidence >= conf_thr
                mask = mask.reshape(-1)
                valid_pts3d = valid_pts3d[mask]
                matches_query_img = matches_query_img[mask]
                matches_map_img = matches_map_img[mask]
                confidence = confidence[mask]

            if len(valid_pts3d) == 0:
                pass
            else:
                query_pts3d.append(valid_pts3d)
                query_pts2d.append(matches_query_img)

                if args.viz_plot:
                    plot_path = f"{args.output_dir}/subscene{str(idx)}"
                    os.makedirs(plot_path, exist_ok=True)
                    plot_path += f"/map{i}_conf_thresh_{conf_thr}"
                    plot_matches(
                        map_view["rgb"],
                        query_view["rgb"],
                        matches_map_img,
                        matches_query_img,
                        plot_path,
                    )

        torch.cuda.empty_cache()
        success = False
        pr_querycam_to_world = None
        total_kpts = 0
        if len(query_pts2d) == 0:
            print(
                f"{idx} Number of matches for query view {query_view['image_name']}: 0"
            )

        if len(query_pts2d) > 0:
            query_pts2d = np.concatenate(query_pts2d, axis=0).astype(np.float32)
            query_pts3d = np.concatenate(query_pts3d, axis=0)
            total_kpts = query_pts2d.shape[0]

            if len(query_pts2d) > pnp_max_points:
                idxs = random.sample(range(len(query_pts2d)), pnp_max_points)
                query_pts3d = query_pts3d[idxs]
                query_pts2d = query_pts2d[idxs]

            W, H = query_view["rgb"].size
            reprojection_error_img = reprojection_error
            if reprojection_error_diag_ratio is not None:
                reprojection_error_img = reprojection_error_diag_ratio * math.sqrt(
                    W**2 + H**2
                )

            success, pr_querycam_to_world = run_pnp(
                query_pts2d,
                query_pts3d,
                query_view["intrinsics"],
                query_view["distortion"],
                pnp_mode,
                reprojection_error_img,
                img_size=[W, H],
            )

        abs_transl_error = float("inf")
        abs_angular_error = float("inf")
        if success:
            abs_transl_error, abs_angular_error = get_pose_error(
                pr_querycam_to_world, query_view["cam_to_world"]
            )

        pose_errors.append(abs_transl_error)
        angular_errors.append(abs_angular_error)
        poses_pred.append(pr_querycam_to_world)

        subscene_id = f"{idx}_{query_view['image_name']}"
        subscene_stat = {
            "correspondences": total_kpts,
            "translation_error": (
                abs_transl_error.tolist() if not np.isinf(abs_transl_error) else "inf"
            ),
            "rotation_error": (
                abs_angular_error.tolist() if not np.isinf(abs_angular_error) else "inf"
            ),
        }
        floor_scene = map_views[0]["image_name"][:4]
        stats[f"{floor_scene}"].append({subscene_id: subscene_stat})

    if args.viz_plot:
        print(f"Completed visualization!")
        sys.exit(0)

    if args.resume_index > 0:
        print(f"Resume completed from {args.resume_index} to {args.stop_index}")
        sys.exit(0)

    exp_label = f"{args.exp_name}_pnp_{pnp_mode}_run{args.run_id}"
    export_results(args.output_dir, exp_label, query_names, poses_pred)
    out_string = aggregate_stats(f"{args.dataset}", pose_errors, angular_errors)
    print(out_string)

    stats["translation_error"] = {
        "min": np.min(pose_errors).tolist(),
        "max": np.max(pose_errors).tolist(),
        "mean": np.mean(pose_errors).tolist(),
        "median": np.median(pose_errors).tolist(),
    }

    stats["rotation_error"] = {
        "min": np.min(angular_errors).tolist(),
        "max": np.max(angular_errors).tolist(),
        "mean": np.mean(angular_errors).tolist(),
        "median": np.median(angular_errors).tolist(),
    }

    stats_path = f"{args.output_dir}/floor_stats_conf{conf_thr}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    print("Completed.")
