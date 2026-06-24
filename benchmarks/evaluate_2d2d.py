import argparse
import json
import os

import torch
from accelerate.utils import set_seed
from fvcore.nn import parameter_count

from unicorrn.model import build_model
from unicorrn.utils import safe_load_weights
from unicorrn.utils.config import read_yaml_config

from . import MegaDepthPoseEstimationBenchmark, ScanNetBenchmark


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate UniCorrn on 2D2D benchmarks Megadepth1500 and ScanNet1500."
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
        "--benchmark", type=str, default="megadepth_1500", help="Name of the experiment"
    )
    parser.add_argument(
        "--query_points_path",
        type=str,
        default="./megadepth1500_query_points.json",
        help="Path of mega1500 keypoints extracted from RoMa",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=1.0001,
        help="confidence threshold for keypoint selection",
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
        "--select_layer", type=int, default=-1, help="select decoder layer"
    )
    parser.add_argument(
        "--unified_model",
        action="store_true",
        help="Whether to evaluate the unified model",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    args = parser.parse_args()
    return args


def test_scannet1500(args, model, query_points):
    scannet_benchmark = ScanNetBenchmark(
        data_root="/projects/vig/Datasets/ScanNet/", unified_model=args.unified_model
    )

    results = scannet_benchmark.benchmark(model, query_points)

    # Define the directory and file path
    name = args.exp_name
    directory = "results/2d2d/scannet"
    file_path = f"{directory}/scannet1500_{name}_cc_{args.coarse_coverage}_ov_{args.overlap}_coarse.json"

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Dump the JSON data to the file
    with open(file_path, "w") as file:
        json.dump(results, file)


def test_mega1500(args, model, query_points, name):
    scene_names = [
        "0015_0.1_0.3.npz",
        "0015_0.3_0.5.npz",
        "0022_0.1_0.3.npz",
        "0022_0.3_0.5.npz",
        "0022_0.5_0.7.npz",
    ]

    mega1500_benchmark = MegaDepthPoseEstimationBenchmark(
        data_root="/projects/vig/Datasets/MegaDepth/megadepth_indices/scene_info_val_1500",
        scene_names=scene_names,
        unified_model=args.unified_model,
    )
    mega1500_results = mega1500_benchmark.benchmark(
        model,
        query_points,
        model_name=name,
        batch_size=500,
        coarse_coverage=args.coarse_coverage,
        overlap=args.overlap,
        select_layer=args.select_layer,
        roma_kpts=True,
    )

    # Define the directory and file path
    directory = "results/2d2d/megadepth"
    file_path = (
        f"{directory}/mega1500_{name}_cc_{args.coarse_coverage}_ov_{args.overlap}.json"
    )

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Dump the JSON data to the file
    with open(file_path, "w") as file:
        json.dump(mega1500_results, file)


if __name__ == "__main__":
    args = parse_args()

    model_cfg = read_yaml_config(args.model_config)

    set_seed(args.seed)
    model = build_model(model_cfg.NAME, cfg=model_cfg)
    print(f"Model params: {parameter_count(model)[''] / 1e6}M")

    weights = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    safe_load_weights(model, weights["model"])
    print(f"Loaded ckpt from: {args.ckpt_path}")

    if args.benchmark == "megadepth_1500":
        with open(args.query_points_path, "r") as file:
            query_points = json.load(file)
        test_mega1500(args, model, query_points, name=args.exp_name)

    elif args.benchmark == "scannet_1500":
        with open(args.query_points_path, "r") as file:
            query_points = json.load(file)
        test_scannet1500(args, model, query_points)

    else:
        raise NotImplementedError(
            "Benchmark not supported. Currently supported megadepth1500 and scannet1500."
        )
