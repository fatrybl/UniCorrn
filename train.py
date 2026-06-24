"""
    Train 2D-2D task with multiple datasets similar to mast3r.
"""

import argparse
import logging
import os
import time
import warnings
from datetime import timedelta

import numpy as np
import torch
import torchvision.transforms as tvf
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    set_seed,
)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from unicorrn.datasets import (  # 2D2D; 2D3D; 3D3D; Common
    ARKitScenes_UnifiedDataset,
    BlendedMVS_UnifiedDataset,
    Co3d_UnifiedDataset,
    ImageToPointRegistrationCollateFn,
    MegaDepth_UnifiedDataset,
    ModelNetDataset,
    MultiTaskBatchDataLoader,
    PointCloudRegistrationCollateFn,
    RGBDScenes2D3DHardPairDataset,
    ScanNetpp_UnifiedDataset,
    SevenScenes2D3DHardPairDataset,
    StaticThings3D_UnifiedDataset,
    ThreeDMatchDataset,
    Waymo_UnifiedDataset,
    WildRGBD_UnifiedDataset,
)
from unicorrn.model import build_model
from unicorrn.trainer import TRAINER_REGISTRY, loss_functions, optimizers
from unicorrn.utils import (
    MetricMeter,
    adjust_learning_rate,
    on_main_process,
    safe_load_weights,
)
from unicorrn.utils.config import CfgNode, read_yaml_config

warnings.filterwarnings("ignore", message=".*antialias parameter.*")

os.environ["PYKEOPS_VERBOSE"] = "0"

logger = get_logger(__name__, log_level="INFO")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

LOG_STEPS = 100

checkpoint_tracker = []

DATASETS = {
    "ARKitScenes": ARKitScenes_UnifiedDataset,
    "BlendedMVS": BlendedMVS_UnifiedDataset,
    "Co3d": Co3d_UnifiedDataset,
    "MegaDepth": MegaDepth_UnifiedDataset,
    "ScanNetpp": ScanNetpp_UnifiedDataset,
    "StaticThings3D": StaticThings3D_UnifiedDataset,
    "Waymo": Waymo_UnifiedDataset,
    "WildRGBD": WildRGBD_UnifiedDataset,
    "7Scenes": SevenScenes2D3DHardPairDataset,
    "RGBDScenesV2": RGBDScenes2D3DHardPairDataset,
    "3DMatch": ThreeDMatchDataset,
    "ModelNet": ModelNetDataset,
}

IGNORE_KEYS = set(
    [
        "dataset",
        "depth",
        "depth1",
        "depth2",
        "camera_pose1",
        "camera_pose2",
        "K1",
        "K2",
        "view_name",
        "points_norm_meta",
        "intrinsics",
        "transform",
        "src",
        "tgt",
        "src_info",
        "tgt_info",
        "min_grid_coord",
        "tgt2src_transform",
        "batch_size",
        "query_indices",
        "target_indices",
        "src_norm_meta",
        "tgt_norm_meta",
        "min_src_grid_coord",
        "min_tgt_grid_coord",
        "scene_name",
        "image_file",
        "depth_file",
        "cloud_file",
        "overlap",
        "image_id",
        "cloud_id",
        "image_h",
        "image_w",
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train UniCorrn.")

    parser.add_argument(
        "--model_config",
        type=str,
        default=None,
        required=True,
        help="UniCorrn model config path",
    )
    parser.add_argument(
        "--trainer_config",
        type=str,
        default=None,
        required=True,
        help="Path of output directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Path of output directory",
    )
    parser.add_argument(
        "--training_stage", type=str, default="stage_1", help="UniCorrn training stage"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Total batch size for all GPUs"
    )
    parser.add_argument(
        "--accum_iter", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--pretrained_croco_weights",
        type=str,
        default=None,
        help="Path of pretrained croco weights",
    )
    parser.add_argument(
        "--pretrained_croco_decoder_weights",
        type=str,
        default=None,
        help="Path of pretrained croco weights",
    )
    parser.add_argument(
        "--resume_ckpt_path",
        type=str,
        default=None,
        help="Path of pretrained croco weights",
    )
    parser.add_argument("--resume_step", type=int, default=0, help="Batch size per GPU")
    parser.add_argument(
        "--prev_stage_epochs",
        type=int,
        default=0,
        help="The epoch number of previous stage",
    )
    parser.add_argument(
        "--restore_training_state",
        action="store_true",
        help="Whether to restore training from a previous checkpoint",
    )
    parser.add_argument(
        "--freeze_encoder", action="store_true", help="Whether to freeze encoder"
    )
    parser.add_argument(
        "--freeze_decoder", action="store_true", help="Whether to freeze decoder"
    )
    parser.add_argument(
        "--joint_finetune_trainable",
        action="store_true",
        help="Whether to only train joint finetune weights",
    )
    parser.add_argument(
        "--find_unused_parameters",
        action="store_true",
        help="Whether to check for unused parameters during backpropagation.",
    )
    parser.add_argument(
        "--set_static_graph",
        action="store_true",
        help="Whether to set model computation graph as static.",
    )
    parser.add_argument(
        "--gradient_as_bucket_view",
        action="store_true",
        help="Whether to use gradient as bucket view.",
    )
    parser.add_argument(
        "--lr_backbone",
        type=float,
        default=0.0,
        help="Whether to restore training from a previous checkpoint",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    args = parser.parse_args()
    return args


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def safe_load_croco_weights(model, weights, decoder_weights=None):
    safe_load_weights(model.img_encoder, weights["model"])
    if decoder_weights is not None:
        safe_load_weights(model.feat_encoder, weights["model"])


def get_train_ds_img2img(ds_cfg):
    transform = tvf.Compose(
        [
            tvf.ColorJitter(0.5, 0.5, 0.5, 0.1),
            tvf.ToTensor(),
            tvf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_ds = None

    for key in ds_cfg.TRAIN_DATA.IMG2IMG.keys():
        num_samples = ds_cfg.TRAIN_DATA.IMG2IMG[key]["SAMPLE_COUNT"]
        path = ds_cfg.TRAIN_DATA.IMG2IMG[key]["PATH"]

        params = {}
        if "PARAMS" in ds_cfg.TRAIN_DATA.IMG2IMG[key].keys():
            params = ds_cfg.TRAIN_DATA.IMG2IMG[key]["PARAMS"]

        split = "train" if key not in ("StaticThings3D", "Waymo") else None

        dataset_cls = DATASETS[key]

        init_ds = num_samples @ dataset_cls(
            ROOT=path,
            split=split,
            resolution=tuple(ds_cfg.TRAIN_DATA_RES),
            aug_crop="auto",
            aug_monocular=0.005,
            transform=transform,
            n_corres=8192,
            nneg=0.5,
            bidirectional=ds_cfg.BIDIRECTIONAL,
            max_queries=ds_cfg.MAX_QUERIES,
            **params,
        )

        if train_ds is None:
            train_ds = init_ds
        else:
            train_ds += init_ds

        logger.info(
            f"Training dataset img2img: {key}, split {split}, size {len(init_ds)}",
            main_process_only=True,
        )

    return train_ds


def get_val_ds_img2img(ds_cfg):
    transform = tvf.Compose(
        [tvf.ToTensor(), tvf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )

    val_ds = None

    for key in ds_cfg.VAL_DATA.IMG2IMG.keys():
        num_samples = ds_cfg.VAL_DATA.IMG2IMG[key]["SAMPLE_COUNT"]
        path = ds_cfg.VAL_DATA.IMG2IMG[key]["PATH"]

        split = "val" if key in ("BlendedMVS", "MegaDepth") else "test"
        dataset_cls = DATASETS[key]

        init_ds = num_samples @ dataset_cls(
            ROOT=path,
            split=split,
            resolution=tuple(ds_cfg.VAL_DATA_RES),
            seed=777,
            transform=transform,
            n_corres=1024,
            bidirectional=False,
            max_queries=1024,
            load_img2img=True,
            load_img2pcd=False,
            load_pcd2pcd=False,
        )
        if val_ds is None:
            val_ds = init_ds
        else:
            val_ds += init_ds

        logger.info(
            f"Validation dataset img2img: {key}, split {split}, size {len(init_ds)}",
            main_process_only=True,
        )

    return val_ds


def get_train_ds_img2pcd(ds_cfg):
    train_ds = None
    for key in ds_cfg.TRAIN_DATA.IMG2PCD.keys():
        data_cls = DATASETS[key]
        path = ds_cfg.TRAIN_DATA.IMG2PCD[key]["PATH"]

        params = {}
        if "PARAMS" in ds_cfg.TRAIN_DATA.IMG2PCD[key].keys():
            params = ds_cfg.TRAIN_DATA.IMG2PCD[key]["PARAMS"]

        IMG_W, IMG_H = tuple(ds_cfg.TRAIN_DATA_RES)

        if key in ("7Scenes", "RGBDScenesV2"):
            init_ds = data_cls(
                dataset_dir=path,
                subset="train",
                new_resolution=(IMG_H, IMG_W),
                max_queries=ds_cfg.MAX_QUERIES,
                scene_name=None,
                return_corr_indices=True,
                **params,
            )
        else:
            transform = tvf.Compose(
                [
                    tvf.ColorJitter(0.5, 0.5, 0.5, 0.1),
                    tvf.ToTensor(),
                    tvf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            num_samples = ds_cfg.TRAIN_DATA.IMG2PCD[key]["SAMPLE_COUNT"]
            split = "train" if key not in ("StaticThings3D", "Waymo") else None
            init_ds = num_samples @ data_cls(
                ROOT=path,
                split=split,
                resolution=tuple(ds_cfg.TRAIN_DATA_RES),
                aug_crop="auto",
                aug_monocular=0.005,
                transform=transform,
                n_corres=8192,
                nneg=0.5,
                bidirectional=ds_cfg.BIDIRECTIONAL,
                max_queries=ds_cfg.MAX_QUERIES,
                **params,
            )

        if train_ds is None:
            train_ds = init_ds
        else:
            train_ds += init_ds

        logger.info(
            f"Training dataset img2pcd: {key}, split train, size {len(init_ds)}",
            main_process_only=True,
        )

    return train_ds


def get_val_ds_img2pcd(ds_cfg):
    val_ds = None
    for key in ds_cfg.VAL_DATA.IMG2PCD.keys():
        data_cls = DATASETS[key]
        path = ds_cfg.VAL_DATA.IMG2PCD[key]["PATH"]

        params = {}
        if "PARAMS" in ds_cfg.VAL_DATA.IMG2PCD[key].keys():
            params = ds_cfg.VAL_DATA.IMG2PCD[key]["PARAMS"]

        IMG_W, IMG_H = tuple(ds_cfg.TRAIN_DATA_RES)

        init_ds = data_cls(
            dataset_dir=path,
            subset="val",
            new_resolution=(IMG_H, IMG_W),
            scene_name=None,
            return_corr_indices=True,
            max_points=None,
            max_queries=None,
            **params,
        )
        if val_ds is None:
            val_ds = init_ds
        else:
            val_ds += init_ds

        logger.info(
            f"Validation dataset img2pcd: {key}, split val, size {len(init_ds)}",
            main_process_only=True,
        )

    return val_ds


def get_train_ds_pcd2pcd(ds_cfg):
    train_ds = None
    for key in ds_cfg.TRAIN_DATA.PCD2PCD.keys():
        data_cls = DATASETS[key]
        path = ds_cfg.TRAIN_DATA.PCD2PCD[key]["PATH"]

        params = {}
        if "PARAMS" in ds_cfg.TRAIN_DATA.PCD2PCD[key].keys():
            params = ds_cfg.TRAIN_DATA.PCD2PCD[key]["PARAMS"]

        if key in ("3DMatch", "ModelNet"):
            init_ds = data_cls(
                ROOT=path,
                max_queries=ds_cfg.MAX_QUERIES,
                bidirectional=ds_cfg.BIDIRECTIONAL,
                downsample_voxel_size=None,
                **params,
            )
        else:
            transform = tvf.Compose(
                [
                    tvf.ColorJitter(0.5, 0.5, 0.5, 0.1),
                    tvf.ToTensor(),
                    tvf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            num_samples = ds_cfg.TRAIN_DATA.PCD2PCD[key]["SAMPLE_COUNT"]
            split = "train" if key not in ("StaticThings3D", "Waymo") else None
            init_ds = num_samples @ data_cls(
                ROOT=path,
                split=split,
                resolution=tuple(ds_cfg.TRAIN_DATA_RES),
                aug_crop="auto",
                aug_monocular=0.005,
                transform=transform,
                n_corres=8192,
                nneg=0.5,
                bidirectional=ds_cfg.BIDIRECTIONAL,
                max_queries=ds_cfg.MAX_QUERIES,
                **params,
            )

        if train_ds is None:
            train_ds = init_ds
        else:
            train_ds += init_ds

        logger.info(
            f"Training dataset pcd2pcd: {key}, split train, size {len(init_ds)}",
            main_process_only=True,
        )

    return train_ds


def get_val_ds_pcd2pcd(ds_cfg):
    val_ds = None
    for key in ds_cfg.VAL_DATA.PCD2PCD.keys():
        data_cls = DATASETS[key]
        path = ds_cfg.VAL_DATA.PCD2PCD[key]["PATH"]

        params = {}
        if "PARAMS" in ds_cfg.VAL_DATA.PCD2PCD[key].keys():
            params = ds_cfg.VAL_DATA.PCD2PCD[key]["PARAMS"]

        init_ds = data_cls(
            ROOT=path,
            max_points=None,
            max_queries=None,
            bidirectional=False,
            downsample_voxel_size=None,
            **params,
        )

        if val_ds is None:
            val_ds = init_ds
        else:
            val_ds += init_ds

        logger.info(
            f"Validation dataset pcd2pcd: {key}, split val, size {len(init_ds)}",
            main_process_only=True,
        )

    return val_ds


def move_to_device(batch, device):
    tasks = batch.keys()
    for task in tasks:
        for key in batch[task].keys():
            if key in IGNORE_KEYS:
                continue

            batch[task][key] = (
                batch[task][key].contiguous().to(device, non_blocking=True)
            )

    return batch


def get_loss(loss_cfg):
    loss_params = loss_cfg.PARAMS.to_dict() if isinstance(loss_cfg, CfgNode) else {}
    loss_type = loss_cfg.NAME if isinstance(loss_cfg, CfgNode) else loss_cfg
    loss_fn = loss_functions.get(loss_type)(**loss_params)
    return loss_fn


def train(cfg):
    set_seed(cfg.SEED)
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=cfg.find_unused_parameters,
        gradient_as_bucket_view=cfg.gradient_as_bucket_view,
    )
    process_kwargs = InitProcessGroupKwargs(
        backend="nccl", timeout=timedelta(minutes=120)
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=cfg.MIXED_PRECISION,
        kwargs_handlers=[ddp_kwargs, process_kwargs],
    )

    logger.info(cfg.dump(), main_process_only=True)

    # Model
    model = build_model(cfg.MODEL.NAME, cfg=cfg.MODEL)
    logger.info(
        f"{cfg.MODEL.NAME} Model params: {parameter_count(model) / 1e6}M",
        main_process_only=True,
    )

    if cfg.PRETRAINED_CROCO_PATH and not cfg.RESUME.CKPT_PATH:
        weights = torch.load(
            cfg.PRETRAINED_CROCO_WEIGHTS, map_location="cpu", weights_only=False
        )
        decoder_weights = (
            torch.load(
                cfg.PRETRAINED_CROCO_DECODER_WEIGHTS,
                map_location="cpu",
                weights_only=False,
            )
            if cfg.PRETRAINED_CROCO_DECODER_WEIGHTS is not None
            else None
        )
        safe_load_croco_weights(model, weights, decoder_weights=decoder_weights)
        logger.info(f"Loaded croco weights from: {cfg.PRETRAINED_CROCO_PATH}")

    if cfg.RESUME.CKPT_PATH:
        weights = torch.load(
            cfg.RESUME.CKPT_PATH, map_location="cpu", weights_only=False
        )
        safe_load_weights(model, weights["model"])
        logger.info(f"Loaded ckpt from: {cfg.RESUME.CKPT_PATH}")

    if cfg.FREEZE_ENCODER:
        model.freeze_croco_encoder()
        logger.info("Training with frozen croco encoder.", main_process_only=True)

    if cfg.FREEZE_DECODER:
        model.freeze_croco_decoder()
        logger.info("Training with frozen croco decoder.", main_process_only=True)

    if cfg.JOINT_FINETUNE_TRAINABLE:
        model.joint_finetune_trainable()
        logger.info(
            "Training with only joint finetune weights.", main_process_only=True
        )

    # Datasets: image to image
    train_dataset_img2img = get_train_ds_img2img(cfg.DATASET)
    val_dataset_img2img = get_val_ds_img2img(cfg.DATASET)

    train_sampler_img2img = train_dataset_img2img.make_sampler(
        cfg.BATCH_SIZE,
        shuffle=True,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        drop_last=True,
    )

    train_dataloader_img2img = DataLoader(
        train_dataset_img2img,
        sampler=train_sampler_img2img,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    train_dataloader_img2img.dataset.set_epoch(0)
    train_dataloader_img2img.sampler.set_epoch(0)

    val_dataloader_img2img = DataLoader(
        val_dataset_img2img,
        sampler=SequentialSampler(val_dataset_img2img),
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader_img2img.dataset.set_epoch(0)

    # Datasets: image to point cloud
    train_dataset_img2pcd = get_train_ds_img2pcd(cfg.DATASET)
    val_dataset_img2pcd = get_val_ds_img2pcd(cfg.DATASET)

    img2pcd_num_samples = cfg.DATASET["2D3D_SAMPLES"]
    if img2pcd_num_samples != 0:
        with_replacement = (
            True if img2pcd_num_samples > len(train_dataset_img2pcd) else False
        )
        train_sampler_img2pcd = RandomSampler(
            train_dataset_img2pcd,
            replacement=with_replacement,
            num_samples=img2pcd_num_samples,
        )
    else:
        train_sampler_img2pcd = train_dataset_img2pcd.make_sampler(
            cfg.BATCH_SIZE,
            shuffle=True,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            drop_last=True,
        )

    train_dataloader_img2pcd = DataLoader(
        train_dataset_img2pcd,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=ImageToPointRegistrationCollateFn(),
        sampler=train_sampler_img2pcd,
        drop_last=True,
        pin_memory=True,
    )
    train_dataloader_img2pcd.dataset.set_epoch(0)
    if hasattr(train_dataloader_img2pcd.sampler, "set_epoch"):
        train_dataloader_img2pcd.sampler.set_epoch(0)

    val_dataloader_img2pcd = DataLoader(
        val_dataset_img2pcd,
        batch_size=1,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=ImageToPointRegistrationCollateFn(),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    # Datasets: point cloud to point cloud
    train_dataset_pcd2pcd = get_train_ds_pcd2pcd(cfg.DATASET)
    val_dataset_pcd2pcd = get_val_ds_pcd2pcd(cfg.DATASET)

    collate_fn = PointCloudRegistrationCollateFn(
        (
            "tgt2src_transform",
            "queries",
            "norm_queries",
            "query_indices",
            "targets",
            "norm_targets",
            "target_indices",
        )
    )

    pcd2pcd_num_samples = cfg.DATASET["3D3D_SAMPLES"]
    if pcd2pcd_num_samples != 0:
        with_replacement = (
            True if pcd2pcd_num_samples > len(train_dataset_pcd2pcd) else False
        )
        train_sampler_pcd2pcd = RandomSampler(
            train_dataset_pcd2pcd,
            replacement=with_replacement,
            num_samples=pcd2pcd_num_samples,
        )
    else:
        train_sampler_pcd2pcd = train_dataset_pcd2pcd.make_sampler(
            cfg.BATCH_SIZE,
            shuffle=True,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            drop_last=True,
        )
    train_dataloader_pcd2pcd = DataLoader(
        train_dataset_pcd2pcd,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        sampler=train_sampler_pcd2pcd,
        drop_last=True,
        pin_memory=True,
    )
    train_dataloader_pcd2pcd.dataset.set_epoch(0)
    if hasattr(train_dataloader_pcd2pcd.sampler, "set_epoch"):
        train_dataloader_pcd2pcd.sampler.set_epoch(0)

    val_dataloader_pcd2pcd = DataLoader(
        val_dataset_pcd2pcd,
        batch_size=1,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    # optimizer
    optim_params = cfg.OPTIMIZER.PARAMS.to_dict()
    logger.info(
        f"Number of trainable parameters: {parameter_count(model) / 1e6} M",
        main_process_only=True,
    )
    optimizer = optimizers.get(cfg.OPTIMIZER.NAME)(
        model.parameters(), lr=cfg.OPTIMIZER.LR, **optim_params
    )
    lr = cfg.OPTIMIZER.LR

    # scheduler
    if cfg.SCHEDULER.USE:
        cfg.SCHEDULER.LR = cfg.OPTIMIZER.LR
        cfg.SCHEDULER.EPOCHS = cfg.EPOCHS

    # trainer state
    epochs = cfg.EPOCHS
    start_epoch = 0
    global_steps = 0
    loss_meter = MetricMeter()
    best_val_epe = {
        "img2img": float("inf"),
        "img2pcd": float("inf"),
        "pcd2pcd": float("inf"),
        "combined_average": float("inf"),
    }

    if cfg.RESUME.RESTORE_STATE:
        weights = torch.load(cfg.RESUME.CKPT_PATH, map_location="cpu")
        # optimizer.load_state_dict(weights["optimizer"])
        global_steps = weights["step"]
        start_epoch = weights["epoch"] + 1
        if "best_val_epe" in weights.keys():
            best_val_epe = weights["best_val_epe"]

        logger.info(
            f"Restored trainer state. Resuming training from epoch: {start_epoch}",
            main_process_only=True,
        )

        if accelerator.scaler is not None and weights["scaler_state"] is not None:
            accelerator.scaler.load_state_dict(weights["scaler_state"])
            logger.info(
                f"Restored gradiet scaler weights for mixed precision training.",
                main_process_only=True,
            )

    # criterion
    matching_loss_params = (
        cfg.MATCHING_LOSS.PARAMS.to_dict()
        if isinstance(cfg.MATCHING_LOSS, CfgNode)
        else {}
    )
    matching_loss_fn = get_loss(cfg.MATCHING_LOSS)
    feature_loss_params = (
        cfg.FEATURE_LOSS.PARAMS.to_dict()
        if isinstance(cfg.FEATURE_LOSS, CfgNode)
        else {}
    )
    feature_loss_fn = get_loss(cfg.FEATURE_LOSS)
    frustum_loss_cfg = getattr(cfg, "FRUSTUM_LOSS", None)
    frustum_loss_fn = get_loss(frustum_loss_cfg) if frustum_loss_cfg is not None else None

    # Prepare everything with `accelerator`.
    model, optimizer = accelerator.prepare(model, optimizer)
    # maybe unsafe?
    if accelerator.num_processes > 1 and cfg.set_static_graph:
        model._set_static_graph()

    trainer_params = cfg.TRAINER.PARAMS.to_dict()
    if cfg.TRAINER.NAME == "unified_trainer_gcd":
        trainer_params["param_names"] = [name for name, _ in model.named_parameters()]
    trainer = TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(**trainer_params)

    # task dataloader wrapper
    train_dataloader = MultiTaskBatchDataLoader(
        {
            "img2img": train_dataloader_img2img,
            "img2pcd": train_dataloader_img2pcd,
            "pcd2pcd": train_dataloader_pcd2pcd,
        },
        **trainer_params,
    )

    val_dataloader = {
        "img2img": val_dataloader_img2img,
        "img2pcd": val_dataloader_img2pcd,
        "pcd2pcd": val_dataloader_pcd2pcd,
    }

    tb_writer = None
    if accelerator.is_main_process:
        tb_writer = setup_output_dir(cfg)

    total_batch_size = (
        cfg.BATCH_SIZE * cfg.GRADIENT_ACCUMULATION_STEPS * accelerator.num_processes
    )

    logger.info("-" * 80, main_process_only=True)
    logger.info(
        f"Total training pairs: {len(train_dataloader) * cfg.BATCH_SIZE * accelerator.num_processes}"
    )
    logger.info(
        f"Total training pairs per GPU: {len(train_dataloader) * cfg.BATCH_SIZE}"
    )
    logger.info(
        f"Total validation pairs:  {len(val_dataloader_img2img) * cfg.BATCH_SIZE + len(val_dataloader_img2pcd) + len(val_dataloader_pcd2pcd)}"
    )
    logger.info("-" * 80, main_process_only=True)
    logger.info(
        f"Training pairs per task per GPU: 2D2D {len(train_dataloader_img2img) * cfg.BATCH_SIZE} 2D3D {len(train_dataloader_img2pcd) * cfg.BATCH_SIZE} 3D3D {len(train_dataloader_pcd2pcd) * cfg.BATCH_SIZE}",
        main_process_only=True,
    )
    logger.info(
        f"Validation pairs per task      : 2D2D {len(val_dataloader_img2img) * cfg.BATCH_SIZE} 2D3D {len(val_dataloader_img2pcd)} 3D3D {len(val_dataloader_pcd2pcd)}",
        main_process_only=True,
    )
    logger.info("-" * 80, main_process_only=True)
    logger.info(f"Number of GPUs: {accelerator.num_processes}")
    logger.info(f"Effective batch size {total_batch_size}", main_process_only=True)
    logger.info("Training started.", main_process_only=True)
    logger.info("-" * 80, main_process_only=True)

    prev_stage_epochs = cfg.prev_stage_epochs

    # with accelerator.autocast():
    for epoch in range(start_epoch, epochs):
        start_time = time.time()

        train_dataloader.loaders["img2img"].dataset.set_epoch(epoch + prev_stage_epochs)
        train_dataloader.loaders["img2img"].sampler.set_epoch(epoch + prev_stage_epochs)
        train_dataloader.loaders["img2pcd"].dataset.set_epoch(epoch + prev_stage_epochs)
        if hasattr(train_dataloader.loaders["img2pcd"].sampler, "set_epoch"):
            train_dataloader.loaders["img2pcd"].sampler.set_epoch(
                epoch + prev_stage_epochs
            )
        train_dataloader.loaders["pcd2pcd"].dataset.set_epoch(epoch + prev_stage_epochs)
        # train_dataloader.loaders["pcd2pcd"].sampler.set_epoch(epoch + prev_stage_epochs)
        if hasattr(train_dataloader.loaders["pcd2pcd"].sampler, "set_epoch"):
            train_dataloader.loaders["pcd2pcd"].sampler.set_epoch(
                epoch + prev_stage_epochs
            )

        model.train()
        optimizer.zero_grad()
        loss_meter.reset()

        for idx, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                batch = move_to_device(batch, accelerator.device)
                loss, loss_details = trainer.run_step(
                    model, batch, matching_loss_fn, feature_loss_fn, frustum_loss_fn
                )
                # loss, loss_details = trainer.run_step(model, batch, loss_fn)

                if np.isnan(loss.data.item()):
                    logger.info("loss is nan.", main_process_only=True)
                else:
                    accelerator.backward(loss)
                    loss_meter.update(**loss_details)

                if accelerator.sync_gradients:
                    global_steps += 1
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

                if cfg.SCHEDULER.USE and idx % cfg.GRADIENT_ACCUMULATION_STEPS == 0:
                    epoch_f = epoch + idx / len(train_dataloader)
                    lr = adjust_learning_rate(optimizer, epoch_f, cfg.SCHEDULER)

                optimizer.step()
                optimizer.zero_grad()

                if (
                    idx % cfg.GRADIENT_ACCUMULATION_STEPS == 0
                    and accelerator.is_main_process
                ):
                    log_step(cfg, loss_meter, tb_writer, lr, epoch, global_steps)

            del batch, loss, loss_details

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_checkpoint(
                cfg,
                accelerator,
                accelerator.unwrap_model(model),
                accelerator.unwrap_model(optimizer),
                global_steps,
                lr,
                epoch,
                best_val_epe,
                last=True,
            )
            best_val_epe = validate(
                cfg,
                accelerator,
                trainer,
                accelerator.unwrap_model(model),
                accelerator.unwrap_model(optimizer),
                val_dataloader,
                best_val_epe,
                lr,
                tb_writer,
                epoch,
                global_steps,
            )
            elapsed_time = time.time() - start_time
            estimated_time_left = elapsed_time * (epochs - (epoch + 1))

            elapsed_time = str(timedelta(seconds=int(elapsed_time)))
            estimated_time_left = str(timedelta(seconds=int(estimated_time_left)))

            logger.info(
                f"Training time epoch {epoch}: {elapsed_time}, estimated time left: {estimated_time_left}"
            )

        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    logger.info("Training completed.", main_process_only=True)


@torch.no_grad
def validate(
    cfg,
    accelerator,
    trainer,
    model,
    optimizer,
    val_dataloaders,
    best_val_epe,
    lr,
    tb_writer,
    epoch,
    global_steps,
):
    if tb_writer is None:
        return best_val_epe

    model.eval()
    val_meter = MetricMeter()

    logger.info(
        f"Validating {len(val_dataloaders['img2img']) * cfg.BATCH_SIZE} image pairs.",
        main_process_only=True,
    )
    logger.info(
        f"Validating {len(val_dataloaders['img2pcd'])} image-point pairs.",
        main_process_only=True,
    )
    logger.info(
        f"Validating {len(val_dataloaders['pcd2pcd'])} point pairs.",
        main_process_only=True,
    )

    val_dataloaders["img2img"].dataset.set_epoch(0)

    for task_key, dataloader in val_dataloaders.items():
        if getattr(trainer, f"{task_key}_wt") == 0:
            continue
        for batch in dataloader:
            with torch.no_grad():
                batch = move_to_device({task_key: batch}, accelerator.device)

                metrics = trainer.evaluate(model, batch)

                for key, val in metrics.items():
                    if np.isnan(val):
                        logger.info(f"epe is nan for {key}.", main_process_only=True)
                    else:
                        metric = {key: val}
                        val_meter.update(**metric)

                del batch

    log_step(cfg, val_meter, tb_writer, lr, epoch, global_steps, stage="val")

    for task_key, val in best_val_epe.items():
        if not task_key in val_meter.meters.keys():
            continue
        if val_meter.meters[task_key].avg < best_val_epe[task_key]:
            best_val_epe[task_key] = val_meter.meters[task_key].avg
            save_checkpoint(
                cfg,
                accelerator,
                model,
                optimizer,
                global_steps,
                lr,
                epoch,
                best_val_epe,
                best=True,
                task=task_key,
            )
            logger.info(
                f"VALIDATION: Best val {task_key} {best_val_epe[task_key]:.6f}",
                main_process_only=True,
            )
            logger.info(
                f"VALIDATION: Saved new best {task_key} model at step {epoch}",
                main_process_only=True,
            )

    logger.info("Completed validation", main_process_only=True)
    logger.info("-" * 80, main_process_only=True)

    return best_val_epe


@on_main_process
def log_step(cfg, loss_meter, tb_writer, lr, epoch, global_steps, stage="train"):
    if stage == "train" and (global_steps % cfg.LOG_INTERVAL != 0 or tb_writer is None):
        return

    if stage == "train":
        log_message = f"epoch: {epoch}, step: {global_steps}, lr: {lr:.6f}, "
        tb_writer.add_scalar("train/lr", lr, global_steps)

        for key, meter in loss_meter.meters.items():
            if key == "total_loss":
                tb_writer.add_scalar(f"train/{key}", meter.val, global_steps)

            tb_writer.add_scalar(f"{stage}/{key}_avg", meter.avg, global_steps)
            log_message += f"{key}: {meter.avg:.5f}, "
    else:
        log_message = f"VALIDATION: epoch: {epoch},  "
        for key, meter in loss_meter.meters.items():
            tb_writer.add_scalar(f"{stage}/epe_{key}_avg", meter.avg, global_steps)
            log_message += f"{key}: {meter.avg:.5f}, "

    logger.info(log_message, main_process_only=True)


@on_main_process
def save_checkpoint(
    cfg,
    accelerator,
    model,
    optimizer,
    global_steps,
    current_lr,
    epoch,
    best_val_epe,
    best=False,
    last=False,
    task="",
):
    if global_steps % cfg.CKPT_INTERVAL != 0 and not best and not last:
        return

    ckpt_path = os.path.join(cfg.OUTPUT_DIR, "ckpts")
    consolidated_state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler_state": (
            accelerator.scaler.state_dict() if accelerator.scaler is not None else None
        ),
        "step": global_steps,
        "epoch": epoch,
        "current_lr": current_lr,
        "best_val_epe": best_val_epe,
    }

    version = global_steps
    if best:
        version = f"best_{task}"

    if last:
        version = "last"

    ckpt_path = os.path.join(ckpt_path, f"model_{version}.pth")
    torch.save(consolidated_state_dict, ckpt_path)


@on_main_process
def setup_output_dir(cfg):
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, "ckpts"), exist_ok=True)

    summary_writter = SummaryWriter(cfg.OUTPUT_DIR)

    save_config_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    with open(save_config_path, "w+") as file:
        cfg.dump(stream=file)

    return summary_writter


if __name__ == "__main__":
    args = parse_args()

    cfg = read_yaml_config(args.trainer_config)
    cfg.MODEL = read_yaml_config(args.model_config)
    cfg.SUFFIX = args.training_stage
    cfg.OUTPUT_DIR = os.path.join(args.output_dir, cfg.SUFFIX)
    cfg.PRETRAINED_CROCO_WEIGHTS = args.pretrained_croco_weights
    cfg.PRETRAINED_CROCO_DECODER_WEIGHTS = args.pretrained_croco_decoder_weights
    cfg.MODEL.BIDIRECTIONAL = cfg.DATASET.BIDIRECTIONAL
    cfg.BATCH_SIZE = args.batch_size
    cfg.GRADIENT_ACCUMULATION_STEPS = args.accum_iter
    cfg.RESUME.CKPT_PATH = args.resume_ckpt_path
    cfg.RESUME.STEP = args.resume_step
    cfg.RESUME.RESTORE_STATE = args.restore_training_state
    cfg.OPTIMIZER.LR_BACKBONE = args.lr_backbone
    cfg.MIXED_PRECISION = args.mixed_precision
    cfg.gradient_as_bucket_view = args.gradient_as_bucket_view
    cfg.find_unused_parameters = args.find_unused_parameters
    cfg.set_static_graph = args.set_static_graph
    cfg.FREEZE_ENCODER = args.freeze_encoder
    cfg.FREEZE_DECODER = args.freeze_decoder
    cfg.JOINT_FINETUNE_TRAINABLE = args.joint_finetune_trainable
    cfg.prev_stage_epochs = args.prev_stage_epochs

    train(cfg)
