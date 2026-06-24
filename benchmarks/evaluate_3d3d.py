import argparse
import json
import os

import numpy as np
import torch
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from unicorrn.datasets import (
    ModelNetDataset,
    PointCloudRegistrationCollateFn,
    ThreeDMatchDataset,
)
from unicorrn.model import build_model
from unicorrn.utils import endpointerror, safe_load_weights
from unicorrn.utils.config import read_yaml_config
from unicorrn.utils.vision3d.array_ops import (
    denormalize_points_meta,
    evaluate_correspondences,
    registration_rmse,
)
from unicorrn.utils.vision3d.array_ops.metrics import (
    relative_rotation_error,
    relative_translation_error,
)
from unicorrn.utils.vision3d.utils.open3d import (
    registration_with_ransac_from_correspondences,
)

DATASETS = {"ModelNet": ModelNetDataset, "3DMatch": ThreeDMatchDataset}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate UniCorrn on 3DMatch")
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
    parser.add_argument("--split", type=str, default="val", help="Dataset split")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="device to be used"
    )

    args = parser.parse_args()
    return args


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_ds(ds_cfg, split):
    ds = {}
    for key in ds_cfg.DATASET.keys():
        if f"{split.upper()}_META_DATA" in ds_cfg.DATASET[key].keys():
            data_cls = DATASETS[key]
            ds[key] = data_cls(
                ROOT=ds_cfg.DATASET[key].ROOT,
                meta_data=ds_cfg.DATASET[key][f"{split.upper()}_META_DATA"],
                max_points=None,
                max_queries=None,
                grid_size=ds_cfg.DATASET[key].GRID_SIZE,
                downsample_voxel_size=ds_cfg.DATASET[key].DOWNSAMPLE_VOXEL_SIZE,
                matching_radius_3d=ds_cfg.DATASET[key].MATCHING_RADIUS_3D,
                use_augmentation=False,
                normalize_points=ds_cfg.NORMALIZE_POINTS,
                bidirectional=False,
            )

    return ds


def evaluate(model, dataloader, device):
    registration_recall = []
    relative_re = []
    relative_te = []
    inlier_ratio = []
    feature_matching_ratio = []
    distance = []
    epe = []

    acceptance_radius = 0.1
    inlier_ratio_threshold = 0.05
    ransac_num_iterations = 50_000
    ransac_distance_threshold = 0.05
    ransac_n = 3
    rmse_threshold = 0.2

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

        queries = sample["queries"].to(device)
        tgt2src_transform = sample["tgt2src_transform"].squeeze(0).numpy()

        if len(queries.shape) == 2:
            queries = queries[None]

        with torch.no_grad():
            preds = model(task="pcd2pcd", sample=sample_, query_pos=queries)

        epe.append(
            endpointerror(
                preds["corr_predictions"], sample["norm_targets"].to(device), dim=-1
            ).item()
        )

        targets_pred_raw = denormalize_points_meta(
            preds["corr_predictions"].squeeze(0).cpu(), sample["tgt_norm_meta"]
        ).numpy()
        targets_gt_raw = denormalize_points_meta(
            sample["targets"].squeeze(0), sample["tgt_norm_meta"]
        ).numpy()
        num_correspondences = targets_pred_raw.shape[0]
        if num_correspondences > 0:
            matching_result_dict = evaluate_correspondences(
                targets_pred_raw,
                targets_gt_raw,
                np.eye(4),
                positive_radius=acceptance_radius,
            )
            inlier_ratio.append(matching_result_dict["inlier_ratio"])
            feature_matching_ratio.append(
                float(matching_result_dict["inlier_ratio"] >= inlier_ratio_threshold)
            )
            distance.append(matching_result_dict["distance"])

        queries_raw = denormalize_points_meta(
            queries.squeeze(0).cpu(), sample["src_norm_meta"]
        ).numpy()
        if num_correspondences >= 4:
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
            tgt_pcd_raw = denormalize_points_meta(
                sample["tgt_pcd"].squeeze(0), sample["tgt_norm_meta"]
            ).numpy()
            rmse = registration_rmse(
                tgt_pcd_raw, tgt2src_transform, estimated_transform
            )
            success_registration = rmse < rmse_threshold
            registration_recall.append(float(success_registration))

    registration_recall = np.array(registration_recall).mean().item()
    relative_re = np.array(relative_re[bool(success_registration)]).mean().item()
    relative_te = np.array(relative_te[bool(success_registration)]).mean().item()
    inlier_ratio = np.array(inlier_ratio).mean().item()
    feature_matching_ratio = np.array(feature_matching_ratio).mean().item()
    distance = np.array(distance).mean().item()
    epe = np.array(epe).mean().item()

    return {
        "IR": inlier_ratio * 100,
        "FMR": feature_matching_ratio * 100,
        "RR": registration_recall * 100,
        "RRE": relative_re,
        "RTE": relative_te,
        "distance": distance,
        "EPE": epe,
    }


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
    val_datasets = get_ds(benchmark_cfg, args.split)
    collate_fn = PointCloudRegistrationCollateFn(
        ("tgt2src_transform", "queries", "norm_queries", "targets", "norm_targets")
    )

    json_dict = {}
    for name, dataset in val_datasets.items():
        val_dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=8,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            prefetch_factor=4,
        )
        print(f"Starting evaluation {args.split} split on {name}")
        result_dict = evaluate(model, val_dataloader, args.device)
        ir, fmr, rr = result_dict["IR"], result_dict["FMR"], result_dict["RR"]
        rre, rte = result_dict["RRE"], result_dict["RTE"]
        print(f"IR: {ir}; FMR: {fmr}; RR: {rr}; RRE: {rre}; RTE: {rte}")
        json_dict[name] = result_dict

    directory = "results/3d3d/3dmatch/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = f"{directory}/{args.split}_{args.exp_name}.json"

    with open(file_path, "w") as f:
        json.dump(json_dict, f, indent=2)
