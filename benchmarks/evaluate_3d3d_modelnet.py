import argparse
import json
import os

import numpy as np
import torch
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from unicorrn.datasets import ModelNetDataset, PointCloudRegistrationCollateFn
from unicorrn.inference.cycle_inference_engine_3d3d import cycle_inference
from unicorrn.model import build_model
from unicorrn.utils import safe_load_weights
from unicorrn.utils.config import read_yaml_config
from unicorrn.utils.vision3d.array_ops import (
    apply_transform,
    compose_transforms,
    denormalize_points_meta,
    inverse_transform,
)
from unicorrn.utils.vision3d.array_ops.metrics import (
    relative_rotation_error,
    relative_translation_error,
)
from unicorrn.utils.vision3d.utils.open3d import (
    registration_with_ransac_from_correspondences,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate UniCorrn on ModelNet.")
    parser.add_argument(
        "--model_config", type=str, required=True, help="UniCorrn model config path"
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path of UniCorrn checkpoint"
    )
    parser.add_argument(
        "--exp_name", type=str, required=True, help="Name of the experiment"
    )
    parser.add_argument("--benchmark_config", type=str, help="Benchmark config path")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="device to be used"
    )

    args = parser.parse_args()
    return args


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_modelnet_benchmark(ds_cfg):
    modelnet_ds = ModelNetDataset(
        ROOT=ds_cfg.DATASET.ModelNet.ROOT,
        meta_data=ds_cfg.DATASET.ModelNet.TEST_MODELNET_META_DATA,
        category_file=ds_cfg.DATASET.ModelNet.TEST_CATEGORIES,
        max_points=None,
        max_queries=None,
        grid_size=ds_cfg.DATASET.ModelNet.GRID_SIZE,
        downsample_voxel_size=ds_cfg.DATASET.ModelNet.DOWNSAMPLE_VOXEL_SIZE,
        matching_radius_3d=ds_cfg.DATASET.ModelNet.MATCHING_RADIUS_3D,
        use_augmentation=False,
        augmentation_noise=0.0,
        normalize_points=ds_cfg.NORMALIZE_POINTS,
        bidirectional=False,
        keep_ratio=0.7,
        return_raw=True,
        deterministic=True,
    )
    modellonet_ds = ModelNetDataset(
        ROOT=ds_cfg.DATASET.ModelNet.ROOT,
        meta_data=ds_cfg.DATASET.ModelNet.TEST_MODELNET_META_DATA,
        category_file=ds_cfg.DATASET.ModelNet.TEST_CATEGORIES,
        max_points=None,
        max_queries=None,
        grid_size=ds_cfg.DATASET.ModelNet.GRID_SIZE,
        downsample_voxel_size=ds_cfg.DATASET.ModelNet.DOWNSAMPLE_VOXEL_SIZE,
        matching_radius_3d=ds_cfg.DATASET.ModelNet.MATCHING_RADIUS_3D,
        use_augmentation=False,
        augmentation_noise=0.0,
        normalize_points=ds_cfg.NORMALIZE_POINTS,
        bidirectional=False,
        keep_ratio=0.5,
        return_raw=True,
        deterministic=True,
    )

    return modelnet_ds, modellonet_ds


def chamfer_distance(src_pcd, tgt_pcd, raw_pcd, tgt2src_transform, pred_transform):
    def square_distance(src, dst):
        return np.sum((src[:, None, :] - dst[None, :, :]) ** 2, axis=-1)

    src_pcd_aligned = apply_transform(
        src_pcd,
        compose_transforms(inverse_transform(tgt2src_transform), pred_transform),
    )
    tgt_pcd_aligned = apply_transform(tgt_pcd, pred_transform)
    dist_src = np.min(square_distance(src_pcd_aligned, raw_pcd), axis=-1)
    dist_tgt = np.min(square_distance(tgt_pcd_aligned, raw_pcd), axis=-1)
    cd = dist_src.mean() + dist_tgt.mean()

    return cd


def evaluate(model, dataloader, device):
    ransac_num_iterations = 50_000
    ransac_distance_threshold = 0.05
    ransac_n = 3
    min_overlap_ratio = 0.2

    chamfer_dist = []
    relative_re = []
    relative_te = []

    cuda_keys = [
        "src_pcd",
        "tgt_pcd",
        "src_grid_coord",
        "tgt_grid_coord",
        "src_length",
        "tgt_length",
    ]
    for sample in tqdm(dataloader):
        sample_ = {}
        for k, v in sample.items():
            if k in cuda_keys:
                sample_[k] = v.to(device)

        tgt2src_transform = sample["tgt2src_transform"].squeeze(0).numpy()

        corr_predictions_raw, cycle_matched = cycle_inference(
            sample_, model, sample["src_norm_meta"], sample["tgt_norm_meta"]
        )
        targets_pred_raw = corr_predictions_raw[cycle_matched]
        queries_raw = denormalize_points_meta(
            sample["src_pcd"][cycle_matched], sample["src_norm_meta"]
        ).numpy()
        num_correspondences = np.sum(cycle_matched)

        if num_correspondences >= queries_raw.shape[0] * min_overlap_ratio:
            # transform from src_pcd to tgt_pcd
            estimated_transform = registration_with_ransac_from_correspondences(
                targets_pred_raw,
                queries_raw,
                distance_threshold=ransac_distance_threshold,
                ransac_n=ransac_n,
                num_iterations=ransac_num_iterations,
            )
            relative_re.append(
                relative_rotation_error(
                    tgt2src_transform[:3, :3], estimated_transform[:3, :3]
                )
            )
            relative_te.append(
                relative_translation_error(
                    tgt2src_transform[:3, 3], estimated_transform[:3, 3]
                )
            )
            src_pcd_raw = denormalize_points_meta(
                sample["src_pcd"].squeeze(0), sample["src_norm_meta"]
            ).numpy()
            tgt_pcd_raw = denormalize_points_meta(
                sample["tgt_pcd"].squeeze(0), sample["tgt_norm_meta"]
            ).numpy()

            chamfer_dist.append(
                chamfer_distance(
                    src_pcd_raw,
                    tgt_pcd_raw,
                    sample["raw_pcd"].squeeze(0).numpy(),
                    tgt2src_transform,
                    estimated_transform,
                )
            )

    relative_re = np.array(relative_re).mean()
    relative_te = np.array(relative_te).mean()
    chamfer_dist = np.array(chamfer_dist).mean()

    return {"RRE": relative_re, "RTE": relative_te, "CD": chamfer_dist}


if __name__ == "__main__":
    args = parse_args()
    set_seed(0)

    model_cfg = read_yaml_config(args.model_config)
    model_cfg.BIDIRECTIONAL = False
    model = build_model(model_cfg.NAME, cfg=model_cfg)
    print(f"Model params: {parameter_count(model) / 1e6}M")

    weights = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    safe_load_weights(model, weights["model"])
    print(f"Loaded ckpt from: {args.ckpt_path}")

    model.eval()
    model.to(args.device)

    benchmark_cfg = read_yaml_config(args.benchmark_config)
    modelnet_benchmark, modellonet_benchmark = get_modelnet_benchmark(benchmark_cfg)
    collate_fn = PointCloudRegistrationCollateFn(
        ("tgt2src_transform", "queries", "norm_queries", "targets", "norm_targets")
    )

    json_dict = {}
    modelnet_dataloader = DataLoader(
        modelnet_benchmark,
        batch_size=1,
        num_workers=8,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=4,
    )
    modellonet_dataloader = DataLoader(
        modellonet_benchmark,
        batch_size=1,
        num_workers=8,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=4,
    )
    print(f"Starting evaluation ModelNet")
    result_dict = evaluate(model, modelnet_dataloader, args.device)
    rre, rte, cd = result_dict["RRE"], result_dict["RTE"], result_dict["CD"]
    print(f"RRE: {rre}; RTE: {rte}; CD: {cd}")
    json_dict["ModelNet"] = result_dict

    print(f"Starting evaluation ModelLoNet")
    result_dict = evaluate(model, modellonet_dataloader, args.device)
    rre, rte, cd = result_dict["RRE"], result_dict["RTE"], result_dict["CD"]
    print(f"RRE: {rre}; RTE: {rte}; CD: {cd}")
    json_dict["ModelLoNet"] = result_dict

    directory = "results/3d3d/modelnet/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = f"{directory}/{args.exp_name}.json"

    with open(file_path, "w") as f:
        json.dump(json_dict, f, indent=2)
