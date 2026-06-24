import os.path as osp
import random
from typing import Optional, Union, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as transforms

from torch.utils.data import Dataset

from ...utils.vision3d.array_ops import (
    GridSample,
    GridSample_Pointcept,
    apply_transform,
    back_project,
    compose_transforms,
    get_2d3d_correspondences_mutual,
    get_2d3d_correspondences_radius,
    get_transform_from_rotation_translation,
    inverse_transform,
    random_sample_small_transform,
    min_max_norm,
    center_shift,
    normalize_coord,
    normalize_coord_corr_points,
    center_shift_corr_points
)
from .data_io import load_pickle, read_depth_image, read_image, get_transform
from ..img2img.dust3r.datasets.base import EasyDataset

class RGBDScenes2D3DHardPairDataset(EasyDataset):
    def __init__(
        self,
        dataset_dir: str,
        subset: str,
        gray_scale: bool = False,
        grid_size: Optional[float] = 0.02,
        max_points: Optional[int] = None,
        max_queries: Optional[int] = None,
        return_corr_indices: bool = False,
        matching_method: str = "mutual_nearest",
        matching_radius_2d: float = 8.0,
        matching_radius_3d: float = 0.0375,
        scene_name: Optional[str] = None,
        overlap_threshold: Optional[float] = None,
        use_augmentation: bool = False,
        augmentation_noise: float = 0.005,
        new_resolution: Optional[Union[List[int], Tuple[int, int], int]] = None,
        normalize_points: bool = False,
    ):
        super().__init__()

        assert subset in ["train", "val", "test"], f"Bad subset name: {subset}."
        assert matching_method in ["mutual_nearest", "radius"], f"Bad matching method: {matching_method}"

        self.dataset_dir = dataset_dir
        self.data_dir = osp.join(self.dataset_dir, "data")
        self.metadata_dir = osp.join(self.dataset_dir, "metadata")
        self.subset = subset
        self.metadata_list = load_pickle(osp.join(self.metadata_dir, f"{self.subset}.pkl"))
        self.grid_sample = GridSample(grid_size)
        self.gray_scale = gray_scale

        if scene_name is not None and scene_name != "None":
            self.metadata_list = [x for x in self.metadata_list if x["scene_name"] == scene_name]

        if overlap_threshold is not None and overlap_threshold != "None":
            self.metadata_list = [x for x in self.metadata_list if x["overlap"] >= overlap_threshold]

        self.max_points = max_points
        self.max_queries = max_queries
        self.return_corr_indices = return_corr_indices
        self.matching_method = matching_method
        self.matching_radius_2d = matching_radius_2d
        self.matching_radius_3d = matching_radius_3d
        self.overlap_threshold = overlap_threshold
        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.new_resolution = (new_resolution, new_resolution) if isinstance(new_resolution,
                                                                             int) else new_resolution
        self.normalize_points = normalize_points

    def __len__(self):
        return len(self.metadata_list)

    def _trim_corrs(self, img_corr_pixels, img_corr_indices, pcd_corr_points, pcd_corr_indices):
        if not self.max_queries:
            return img_corr_pixels, img_corr_indices, pcd_corr_points, pcd_corr_indices

        length = img_corr_pixels.shape[0]
        if length >= self.max_queries:
            mask = np.random.choice(length, self.max_queries)
            return ( 
                img_corr_pixels[mask], 
                img_corr_indices[mask], 
                pcd_corr_points[mask], 
                pcd_corr_indices[mask]
            )

        mask = np.random.choice(length, self.max_queries - length)
        return (
            np.concatenate([img_corr_pixels, img_corr_pixels[mask]], axis=0),
            np.concatenate([img_corr_indices, img_corr_indices[mask]], axis=0),  
            np.concatenate([pcd_corr_points, pcd_corr_points[mask]], axis=0),
            np.concatenate([pcd_corr_indices, pcd_corr_indices[mask]], axis=0)
        )

    def __getitem__(self, index: int):
        if isinstance(index, tuple):
            index = index[0]

        data_dict = {}

        metadata: dict = self.metadata_list[index]
        data_dict["scene_name"] = metadata["scene_name"]
        data_dict["image_file"] = metadata["image_file"]
        data_dict["depth_file"] = metadata["depth_file"]
        data_dict["cloud_file"] = metadata["cloud_file"]
        data_dict["overlap"] = metadata["overlap"]
        data_dict["image_id"] = osp.basename(metadata["image_file"]).split(".")[0].split("_")[1]
        data_dict["cloud_id"] = osp.basename(metadata["cloud_file"]).split(".")[0].split("_")[1]

        intrinsics_file = osp.join(self.data_dir, metadata["scene_name"], "camera-intrinsics.txt")
        intrinsics = np.loadtxt(intrinsics_file)
        transform = metadata["cloud_to_image"]

        # read image
        depth = read_depth_image(osp.join(self.data_dir, metadata["depth_file"])).astype(np.float32)
        image = read_image(osp.join(self.data_dir, metadata["image_file"]), as_gray=self.gray_scale)

        data_dict["image_h"] = image.shape[0]
        data_dict["image_w"] = image.shape[1]
        image = torch.from_numpy(image).permute(2, 0, 1)

        # read points
        points = np.load(osp.join(self.data_dir, metadata["cloud_file"]))
        if self.max_points is not None and points.shape[0] > self.max_points:
            sel_indices = np.random.permutation(points.shape[0])[: self.max_points]
            points = points[sel_indices]

        if self.use_augmentation:
            # augment point cloud
            aug_transform = random_sample_small_transform()
            center = points.mean(axis=0)
            subtract_center = get_transform_from_rotation_translation(None, -center)
            add_center = get_transform_from_rotation_translation(None, center)
            aug_transform = compose_transforms(subtract_center, aug_transform, add_center)
            points = apply_transform(points, aug_transform)
            inv_aug_transform = inverse_transform(aug_transform)
            transform = compose_transforms(inv_aug_transform, transform)
            points += (np.random.rand(points.shape[0], 3) - 0.5) * self.aug_noise

        if self.new_resolution is not None and self.new_resolution != (image.shape[1], image.shape[2]):
            raw_image_h = image.shape[1]
            raw_image_w = image.shape[2]
            scale = (self.new_resolution[0] / raw_image_h, self.new_resolution[1] / raw_image_w)
            img_resize = transforms.Resize(self.new_resolution, interpolation=transforms.InterpolationMode.BILINEAR)
            image = img_resize(image)
            depth_resize = transforms.Resize(self.new_resolution, interpolation=transforms.InterpolationMode.NEAREST)
            depth = depth_resize(torch.from_numpy(depth[None])).numpy().squeeze()
            intrinsics[0, 0] = intrinsics[0, 0] * scale[0]
            intrinsics[1, 1] = intrinsics[1, 1] * scale[1]
            intrinsics[0, 2] = intrinsics[0, 2] * scale[0]
            intrinsics[1, 2] = intrinsics[1, 2] * scale[1]

        normalize = transforms.Compose([transforms.ToDtype(torch.float32, scale=True),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        # add normalized rgb information
        depth = np.concatenate([depth[..., None], normalize(image).permute(1, 2, 0).numpy()], axis=-1)
        image = get_transform(self.use_augmentation)(image).numpy()

        # build correspondences
        if self.return_corr_indices:
            if self.matching_method == "mutual_nearest":
                img_corr_pixels, pcd_corr_indices, pcd_corr_points = get_2d3d_correspondences_mutual(
                    depth, points, intrinsics, transform, self.matching_radius_2d, self.matching_radius_3d
                )
            else:
                img_corr_pixels, pcd_corr_indices, pcd_corr_points = get_2d3d_correspondences_radius(
                    depth, points, intrinsics, transform, self.matching_radius_2d, self.matching_radius_3d
                )
            img_corr_indices = img_corr_pixels[:, 0] * image.shape[1] + img_corr_pixels[:, 1]

            img_corr_pixels, img_corr_indices, pcd_corr_points, pcd_corr_indices = self._trim_corrs(
                                                                                        img_corr_pixels, 
                                                                                        img_corr_indices, 
                                                                                        pcd_corr_points, 
                                                                                        pcd_corr_indices
                                                                                    )

            H, W = self.new_resolution

            queries = np.ascontiguousarray(img_corr_pixels[:,[1,0]])  # H, W -> W, H
            norm_queries = queries / np.array([W, H])
            targets = np.ascontiguousarray(pcd_corr_points)  # X, Y, Z
            data_dict['queries'] = queries
            data_dict['norm_queries'] = norm_queries.astype(np.float32) 
            data_dict['targets'] = targets

        grid_sample = self.grid_sample(points)

        if self.normalize_points:
            norm_targets = normalize_coord_corr_points(targets, points)
            points, points_norm_meta = normalize_coord(points, return_meta=True)
            data_dict['norm_targets'] = norm_targets
            data_dict['points_norm_meta'] = points_norm_meta


        # build data dict
        data_dict['intrinsics'] = intrinsics.astype(np.float32)
        data_dict['transform'] = transform.astype(np.float32)
        data_dict['image'] = image.astype(np.float32)
        data_dict['depth'] = depth.astype(np.float32)
        data_dict['points'] = points.astype(np.float32)
        data_dict['grid_coord'] = grid_sample['grid_coord']
        data_dict['min_grid_coord'] = grid_sample['min_coord']
        data_dict['dataset'] = "RGBDScenesV2"

        return data_dict
