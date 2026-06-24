import argparse
import datetime
import json
import os
import time
import warnings
from typing import List

import numpy as np
import torch

warnings.filterwarnings("ignore")

from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from unicorrn.datasets import (
    ImageToPointRegistrationCollateFn,
    RGBDScenes2D3DHardPairDataset,
)
from unicorrn.model import build_model
from unicorrn.utils import safe_load_weights
from unicorrn.utils.config import read_yaml_config
from unicorrn.utils.vision3d.array_ops import (
    apply_transform,
    denormalize_points_meta,
    evaluate_correspondences,
    registration_rmse,
)
from unicorrn.utils.vision3d.ops import back_project
from unicorrn.utils.vision3d.utils.opencv import registration_with_pnp_ransac


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate UniCorrn on RGBD Scenes v2.")
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
        "--dataset_dir", type=str, default="/projects/vig/Datasets/RGBDScenesV2"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="device to be used"
    )
    parser.add_argument(
        "--pcd2img_eval",
        action="store_true",
        help="Use this flag to evaluate 3D to 2D correspondences.",
    )

    args = parser.parse_args()
    return args


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def denormalize(image_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # Convert mean and std to tensors
    mean_tensor = torch.tensor(mean).unsqueeze(1).unsqueeze(2)
    std_tensor = torch.tensor(std).unsqueeze(1).unsqueeze(2)

    # Denormalize the image
    denormalized_image = image_tensor * std_tensor + mean_tensor

    # Clip pixel values to be in the range [0, 1]
    denormalized_image = torch.clamp(denormalized_image, 0, 1)

    return denormalized_image


def denormalize_min_max(scaled_values, min_val, max_val):
    denormalized_values = scaled_values * (max_val - min_val) + min_val
    return denormalized_values


def get_HW_resolution(maxdim=512):
    if maxdim == 512:
        return (384, 512)
    else:
        raise NotImplementedError


def evaluate_unified_img2pcd(
    model, split="val", dataset_dir="/projects/vig/Datasets/RGBDScenesV2"
):
    matching_radius_2d = 8.0
    matching_radius_3d = 0.0375
    overlap_threshold = None
    batch_size = 1
    num_workers = 8
    scene_name = None

    maxdim = max(model.patch_embed.img_size)
    H, W = get_HW_resolution(maxdim=maxdim)

    val_dataset = RGBDScenes2D3DHardPairDataset(
        dataset_dir=dataset_dir,
        subset=split,
        grid_size=0.02,
        max_points=None,
        max_queries=None,
        return_corr_indices=True,
        matching_radius_2d=matching_radius_2d,
        matching_radius_3d=matching_radius_3d,
        scene_name=scene_name,
        overlap_threshold=overlap_threshold,
        new_resolution=(H, W),
        use_augmentation=False,
        normalize_points=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=ImageToPointRegistrationCollateFn(),
    )

    print(f"Evaluating {split} split, dataset size: {len(val_dataset)}")

    acceptance_radius = 0.05
    inlier_ratio_threshold = 0.1
    ransac_num_iterations = 50_000
    ransac_distance_tolerance = 8.0
    rmse_threshold = 0.1

    scenes = {}
    times = []

    for data_dict in tqdm(val_loader):
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = v.cuda()

            if isinstance(v, List):
                new_list = []
                for item in v:
                    if isinstance(item, torch.Tensor):
                        new_list.append(item.cuda())

                data_dict[k] = new_list

        scene_name = data_dict["scene_name"]
        img = data_dict["image"]
        queries = data_dict["queries"]
        targets_gt = data_dict["targets"].cpu()
        transform = data_dict["transform"].cpu().numpy()
        intrinsics = data_dict["intrinsics"].cpu().numpy()
        norm_meta = data_dict["points_norm_meta"]

        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            preds = model.forward_img_to_pcd(
                src_img=img, sample=data_dict, query_pos_2d=queries
            )["img2pcd"]

        torch.cuda.synchronize()
        times.append(time.time() - start_time)

        pcd_points = data_dict["points"].cpu()
        raw_points = denormalize_points_meta(pcd_points, norm_meta).numpy()

        target_preds = preds["corr_predictions"].cpu().squeeze(0)
        target_preds = denormalize_points_meta(target_preds, norm_meta).numpy()

        targets_gt = apply_transform(targets_gt.squeeze(0).numpy(), transform)

        # 1. evaluate fine correspondences
        num_correspondences = target_preds.shape[0]

        if num_correspondences > 4:
            fine_matching_result_dict = evaluate_correspondences(
                target_preds, targets_gt, transform, positive_radius=acceptance_radius
            )
        else:
            fine_matching_result_dict = {
                "inlier_ratio": 0.0,
                "overlap": 0.0,
                "distance": 0.0,
            }

        inlier_ratio = fine_matching_result_dict["inlier_ratio"]
        corr_dist = fine_matching_result_dict["distance"]
        recall = float(inlier_ratio >= inlier_ratio_threshold)

        # 2. evaluate registration
        queries = queries.squeeze().cpu().numpy()

        if num_correspondences >= 4:
            estimated_transform = registration_with_pnp_ransac(
                target_preds,
                queries,
                intrinsics,
                num_iterations=ransac_num_iterations,
                distance_tolerance=ransac_distance_tolerance,
                transposed=False,  # pixel coordinates in (w,h)
            )
            rmse = registration_rmse(raw_points, transform, estimated_transform)
            registration_recall = float(rmse < rmse_threshold)
        else:
            estimated_transform = None
            registration_recall = 0.0

        if scene_name in scenes.keys():
            scenes[scene_name]["IR"].append(inlier_ratio)
            scenes[scene_name]["FMR"].append(recall)
            scenes[scene_name]["RR"].append(registration_recall)
            scenes[scene_name]["EPE"].append(corr_dist)
        else:
            scenes[scene_name] = {
                "IR": [inlier_ratio],
                "FMR": [recall],
                "RR": [registration_recall],
                "EPE": [corr_dist],
            }

    IR, FMR, RR, EPE = [], [], [], []
    total_samples = 0
    results = {}
    for name, scene in scenes.items():
        scene_ir = np.mean(scene["IR"])
        scene_fmr = np.mean(scene["FMR"])
        scene_rr = np.mean(scene["RR"])
        scene_epe = np.mean(scene["EPE"])
        IR.append(scene_ir)
        FMR.append(scene_fmr)
        RR.append(scene_rr)
        EPE.append(scene_epe)
        samples = len(scene["IR"])
        total_samples += samples
        results[name] = {
            "samples": samples,
            "IR": round(scene_ir * 100, 2),
            "FMR": round(scene_fmr * 100, 2),
            "RR": round(scene_rr * 100, 2),
            "EPE": round(scene_epe, 3),
        }
        print(
            f"{name}, samples: {samples}, IR: {scene_ir * 100:.2f}, fmr: {scene_fmr * 100:.2f}, rr: {scene_rr * 100:.2f}, epe: {scene_epe:.2f}"
        )

    results["overall"] = {
        "mean_ir": round(np.mean(IR) * 100, 1),
        "mean_fmr": round(np.mean(FMR) * 100, 1),
        "mean_rr": round(np.mean(RR) * 100, 1),
        "mean_epe": round(np.mean(EPE), 3),
        "avg_inference_time": round(np.mean(times), 4),
        "total_inference_time": str(datetime.timedelta(seconds=int(sum(times)))),
    }

    print(f"Evaluation results RGBD ScenesV2 - {split} split")
    for key, value in results["overall"].items():
        print(f"{key}: {value}")
    print("-" * 10)

    return results


def evaluate_unified_pcd2img(
    model, split="val", dataset_dir="/projects/vig/Datasets/RGBDScenesV2"
):
    matching_radius_2d = 8.0
    matching_radius_3d = 0.0375
    overlap_threshold = None
    batch_size = 1
    num_workers = 8
    scene_name = None

    val_dataset = RGBDScenes2D3DHardPairDataset(
        dataset_dir=dataset_dir,
        subset=split,
        grid_size=0.02,
        max_points=None,
        max_queries=None,
        return_corr_indices=True,
        matching_radius_2d=matching_radius_2d,
        matching_radius_3d=matching_radius_3d,
        scene_name=scene_name,
        overlap_threshold=overlap_threshold,
        new_resolution=(384, 512),
        use_augmentation=False,
        normalize_points=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=ImageToPointRegistrationCollateFn(),
    )

    print(f"Evaluating {split} split, dataset size: {len(val_dataset)}")

    acceptance_radius = 0.05
    inlier_ratio_threshold = 0.1
    ransac_num_iterations = 50_000
    ransac_distance_tolerance = 8.0
    rmse_threshold = 0.1

    scenes = {}
    times = []

    for data_dict in tqdm(val_loader):
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = v.cuda()

            if isinstance(v, List):
                new_list = []
                for item in v:
                    if isinstance(item, torch.Tensor):
                        new_list.append(item.cuda())

                data_dict[k] = new_list

        scene_name = data_dict["scene_name"]
        img = data_dict["image"]
        depth = data_dict["depth"]
        queries = data_dict["norm_targets"]
        targets_gt = data_dict["queries"].cpu()
        transform = data_dict["transform"].cpu().numpy()
        intrinsics = data_dict["intrinsics"]
        norm_meta = data_dict["points_norm_meta"]

        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            preds = model.forward_pcd_to_img(
                sample=data_dict, tgt_img=img, query_pos_3d=queries
            )["pcd2img"]

        torch.cuda.synchronize()
        times.append(time.time() - start_time)

        pcd_points = data_dict["points"].cpu()
        raw_points = denormalize_points_meta(pcd_points, norm_meta).numpy()

        target_preds = preds["corr_predictions"].cpu().squeeze(0)

        b, c, h, w = img.shape
        target_preds[:, 0] = torch.clamp(target_preds[:, 0] * w, min=0, max=w - 1)
        target_preds[:, 1] = torch.clamp(target_preds[:, 1] * h, min=0, max=h - 1)

        img_points = back_project(
            depth[:, :, :, 0],
            intrinsics.unsqueeze(0),
            depth_limit=6.0,
            transposed=True,
            return_mask=False,
        ).squeeze(0)

        target_preds_3d = (
            img_points[target_preds[:, 1].long(), target_preds[:, 0].long()]
            .cpu()
            .numpy()
        )
        queries_3d = data_dict["targets"].cpu().squeeze(0).numpy()

        # 1. evaluate fine correspondences
        num_correspondences = target_preds.shape[0]

        if num_correspondences > 4:
            fine_matching_result_dict = evaluate_correspondences(
                queries_3d,
                target_preds_3d,
                transform,
                positive_radius=acceptance_radius,
            )
        else:
            fine_matching_result_dict = {
                "inlier_ratio": 0.0,
                "overlap": 0.0,
                "distance": 0.0,
            }

        inlier_ratio = fine_matching_result_dict["inlier_ratio"]
        corr_dist = fine_matching_result_dict["distance"]
        recall = float(inlier_ratio >= inlier_ratio_threshold)

        # 2. evaluate registration
        intrinsics = intrinsics.cpu().numpy()
        target_preds = target_preds.cpu().numpy()

        if num_correspondences >= 4:
            estimated_transform = registration_with_pnp_ransac(
                queries_3d,  # 3d queries
                target_preds,  # 2d preds
                intrinsics,
                num_iterations=ransac_num_iterations,
                distance_tolerance=ransac_distance_tolerance,
                transposed=False,  # pixel coordinates in (w,h)
            )
            rmse = registration_rmse(raw_points, transform, estimated_transform)
            registration_recall = float(rmse < rmse_threshold)
        else:
            estimated_transform = None
            registration_recall = 0.0

        if scene_name in scenes.keys():
            scenes[scene_name]["IR"].append(inlier_ratio)
            scenes[scene_name]["FMR"].append(recall)
            scenes[scene_name]["RR"].append(registration_recall)
            scenes[scene_name]["EPE"].append(corr_dist)
        else:
            scenes[scene_name] = {
                "IR": [inlier_ratio],
                "FMR": [recall],
                "RR": [registration_recall],
                "EPE": [corr_dist],
            }

    IR, FMR, RR, EPE = [], [], [], []
    total_samples = 0
    results = {}
    for name, scene in scenes.items():
        scene_ir = np.mean(scene["IR"])
        scene_fmr = np.mean(scene["FMR"])
        scene_rr = np.mean(scene["RR"])
        scene_epe = np.mean(scene["EPE"])
        IR.append(scene_ir)
        FMR.append(scene_fmr)
        RR.append(scene_rr)
        EPE.append(scene_epe)
        samples = len(scene["IR"])
        total_samples += samples
        results[name] = {
            "samples": samples,
            "IR": round(scene_ir * 100, 2),
            "FMR": round(scene_fmr * 100, 2),
            "RR": round(scene_rr * 100, 2),
            "EPE": round(scene_epe, 3),
        }
        print(
            f"{name}, samples: {samples}, IR: {scene_ir * 100:.2f}, fmr: {scene_fmr * 100:.2f}, rr: {scene_rr * 100:.2f}, epe: {scene_epe:.2f}"
        )

    results["overall"] = {
        "mean_ir": round(np.mean(IR) * 100, 1),
        "mean_fmr": round(np.mean(FMR) * 100, 1),
        "mean_rr": round(np.mean(RR) * 100, 1),
        "mean_epe": round(np.mean(EPE), 3),
        "avg_inference_time": round(np.mean(times), 4),
        "total_inference_time": str(datetime.timedelta(seconds=int(sum(times)))),
    }

    print(f"Evaluation results 7Scenes - {split} split")
    for key, value in results["overall"].items():
        print(f"{key}: {value}")
    print("-" * 10)

    return results


if __name__ == "__main__":
    args = parse_args()
    set_seed(0)

    cfg = read_yaml_config(args.model_config)
    model = build_model(cfg.NAME, cfg=cfg)
    print(f"Model params: {parameter_count(model) / 1e6}M")

    weights = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    safe_load_weights(model, weights["model"])
    print(f"Loaded ckpt from: {args.ckpt_path}")

    model.eval()
    model = model.cuda()

    if not args.pcd2img_eval:
        result_dict = evaluate_unified_img2pcd(
            model,
            split=args.split,
            dataset_dir=args.dataset_dir,
        )
    elif args.pcd2img_eval:
        result_dict = evaluate_unified_pcd2img(
            model,
            split=args.split,
            dataset_dir=args.dataset_dir,
        )
    else:
        raise NotImplementedError

    # Define the directory and file path
    directory = "results/2d3d/rgbdscenesv2"
    file_path = f"{directory}/{args.split}_{args.exp_name}.json"

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Dump the JSON data to the file
    with open(file_path, "w") as file:
        json.dump(result_dict, file)
