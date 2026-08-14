import os.path as osp
from typing import Optional, Union, List, Tuple

import torch
import torchvision.transforms.v2 as transforms
import numpy as np
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
    center_shift,
    normalize_coord,
    normalize_coord_corr_points,
    center_shift_corr_points
)
from .data_io import load_pickle, read_depth_image, read_image, get_transform
from .frustum_labels import build_frustum_queries
from ..img2img.dust3r.datasets.base import EasyDataset

def _get_frame_name(filename):
    _, seq_name, frame_name = filename.split('.')[0].split('/')
    seq_id = seq_name.split('-')[-1]
    frame_id = frame_name.split('_')[-1]
    output_name = f'{seq_id}-{frame_id}'
    return output_name


class SevenScenes2D3DHardPairDataset(EasyDataset):
    def __init__(
            self,
            dataset_dir: str,
            subset: str,
            max_points: Optional[int] = None,
            max_queries: Optional[int] = None,
            grid_size: Optional[float] = 0.02,
            return_corr_indices: bool = False,
            matching_method: str = 'mutual_nearest',
            matching_radius_2d: float = 8.0,
            matching_radius_3d: float = 0.0375,
            scene_name: Optional[str] = None,
            overlap_threshold: Optional[float] = None,
            use_augmentation: bool = False,
            augmentation_noise: float = 0.005,
            new_resolution: Optional[Union[List[int], Tuple[int, int], int]] = None,
            return_overlap_indices: bool = False,
            normalize_points: bool = True,
            return_frustum: bool = False,
            num_frustum_queries: int = 1024,
    ):
        super().__init__()

        assert subset in ['trainval', 'train', 'val', 'test']
        assert matching_method in ['mutual_nearest', 'radius'], f'Bad matching method: {matching_method}'

        self.grid_sample = GridSample(grid_size)

        self.dataset_dir = dataset_dir
        self.data_dir = osp.join(self.dataset_dir, 'data')
        self.metadata_dir = osp.join(self.dataset_dir, 'metadata')
        self.subset = subset
        self.metadata_list = load_pickle(osp.join(self.metadata_dir, f'{self.subset}-full.pkl'))

        if scene_name is not None and scene_name != "None":
            self.metadata_list = [x for x in self.metadata_list if x['scene_name'] == scene_name]

        # self.metadata_list = [x for x in self.metadata_list if 'seq-11/color_019.png' in x['image_file']]

        if overlap_threshold is not None and overlap_threshold != "None":
            self.metadata_list = [x for x in self.metadata_list if x['overlap'] >= overlap_threshold]

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
        self.return_overlap_indices = return_overlap_indices
        self.normalize_points = normalize_points
        self.return_frustum = return_frustum
        self.num_frustum_queries = num_frustum_queries

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
        """
        Data_dict keys:
            scene_name
            image_file
            depth_file
            cloud_file
            overlap
            image_id
            cloud_id
            image_h
            image_w
            grid_coord - grid coord with min_coord at (0, 0, 0)
            min_grid_coord - original min_grid_coord applied as offset
            queries - H, W, image pixel coordinates (query points)
            norm_queries
            img_corr_indices
            targets - X, Y, Z, point cloud coordinates (ground truth) (1-to-1 mapping)
            norm_targets
            pcd_corr_indices
            intrinsics
            transform - point cloud to image transform
            depth - depth map (H, W, 1 + C); C - image channels append to depth values
            feats - X, Y, Z, R, G, B
        """
        if isinstance(index, tuple):
            index = index[0]

        data_dict = {}

        metadata: dict = self.metadata_list[index]
        data_dict['scene_name'] = metadata['scene_name']
        data_dict['image_file'] = metadata['image_file']
        data_dict['depth_file'] = metadata['depth_file']
        data_dict['cloud_file'] = metadata['cloud_file']
        data_dict['overlap'] = metadata['overlap']
        data_dict['image_id'] = _get_frame_name(metadata['image_file'])
        data_dict['cloud_id'] = _get_frame_name(metadata['cloud_file'])

        intrinsics_file = osp.join(self.data_dir, metadata['scene_name'], 'camera-intrinsics.txt')
        intrinsics = np.loadtxt(intrinsics_file)
        transform = metadata['cloud_to_image']

        # read image
        image = read_image(osp.join(self.data_dir, metadata['image_file']), as_gray=False)
        depth = read_depth_image(osp.join(self.data_dir, metadata['depth_file'])).astype(np.float32)

        data_dict['image_h'] = image.shape[0]
        data_dict['image_w'] = image.shape[1]
        image = torch.from_numpy(image).permute(2, 0, 1)

        # read points
        points = np.load(osp.join(self.data_dir, metadata['cloud_file']))
        if self.max_points is not None and points.shape[0] > self.max_points:
            sel_indices = np.random.permutation(points.shape[0])[: self.max_points]
            points = points[sel_indices]

        # if self.center_shift:
        #     points = center_shift(points, apply_z=True)

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
            if self.matching_method == 'mutual_nearest':
                img_corr_pixels, pcd_corr_indices, pcd_corr_points = get_2d3d_correspondences_mutual(
                    depth, points, intrinsics, transform, self.matching_radius_2d, self.matching_radius_3d
                )
            else:
                img_corr_pixels, pcd_corr_indices, pcd_corr_points = get_2d3d_correspondences_radius(
                    depth, points, intrinsics, transform, self.matching_radius_2d, self.matching_radius_3d
                )
            img_corr_indices = img_corr_pixels[:, 0] * image.shape[2] + img_corr_pixels[:, 1]

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
            # data_dict['img_corr_indices'] = img_corr_indices
            data_dict['targets'] = targets
            # data_dict['pcd_corr_indices'] = pcd_corr_indices

        if self.return_overlap_indices:
            img_corr_pixels, pcd_corr_indices, pcd_corr_points = get_2d3d_correspondences_radius(
                depth, points, intrinsics, transform, self.matching_radius_2d, self.matching_radius_3d
            )
            img_corr_indices = img_corr_pixels[:, 0] * image.shape[2] + img_corr_pixels[:, 1]
            img_overlap_indices = np.unique(img_corr_indices)
            pcd_overlap_indices = np.unique(pcd_corr_indices)
            img_overlap_h_pixels = img_overlap_indices // image.shape[2]
            img_overlap_w_pixels = img_overlap_indices % image.shape[2]
            img_overlap_pixels = np.stack([img_overlap_h_pixels, img_overlap_w_pixels], axis=1)
            data_dict['img_overlap_pixels'] = img_overlap_pixels
            data_dict['img_overlap_indices'] = img_overlap_indices
            data_dict['pcd_overlap_indices'] = pcd_overlap_indices


        grid_sample = self.grid_sample(points)

        if self.normalize_points:
            norm_targets = normalize_coord_corr_points(targets, points)
            if self.return_frustum:
                if self.new_resolution is not None:
                    fr_h, fr_w = self.new_resolution
                else:
                    fr_h, fr_w = data_dict['image_h'], data_dict['image_w']
                frustum_queries, frustum_labels, frustum_distances = build_frustum_queries(
                    points, transform, intrinsics, fr_h, fr_w, self.num_frustum_queries
                )
                data_dict['frustum_queries'] = frustum_queries
                data_dict['frustum_labels'] = frustum_labels
                data_dict['frustum_distances'] = frustum_distances
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
        data_dict['dataset'] = "7Scenes"

        return data_dict





def run_test():
    import numpy as np
    import pyvista as pv

    from array_ops import apply_transform, back_project
    from utils.visualization import draw_correspondences, draw_straight_correspondences

    train_loader = SevenScenes2D3DHardPairDataset('../../7Scenes', 'train', gray_scale=False,
                                                  matching_method='mutual_nearest',
                                                  max_queries=100,
                                                  return_corr_indices=True, new_resolution=(240, 320))
    # print(neighbor_limits)
    data_dict = train_loader[1]
    # data_dict = tensor_to_array(data_dict)

    depth = data_dict["depth"]
    intrinsic = data_dict["intrinsics"]
    img_points, img_pixels = back_project(depth, intrinsic, depth_limit=6.0, return_pixels=True)

    pcd_points = data_dict["points"]
    transform = data_dict["transform"]

    img_corr_pixels = data_dict["queries"]

    img_indices = np.full_like(depth[..., 0], fill_value=-1, dtype=np.int32)
    img_indices[img_pixels[:, 0], img_pixels[:, 1]] = np.arange(img_pixels.shape[0])
    img_corr_indices = img_indices[img_corr_pixels[:, 0], img_corr_pixels[:, 1]]
    print(img_corr_indices.shape)
    print(data_dict['img_corr_indices'].shape)

    # print(img_corr_pixels)
    pcd_corr_points = data_dict['targets']
    print(data_dict["norm_queries"].shape)
    img_corr_pixels = data_dict["norm_queries"]

    draw_correspondences(img_points[..., :3], pcd_points, img_corr_indices, data_dict['pcd_corr_indices'],
                         offsets=(0., 1., 0.))
    img_plane = np.concatenate([img_corr_pixels,
                                np.ones(img_corr_pixels.shape[0])[..., None]], axis=-1)

    # pcd = pv.PolyData(pcd_points)
    # grid = pv.PolyData(data_dict['grid_coord'])
    # plotter = pv.Plotter(window_size=(1920, 1080))
    # plotter.set_viewup([0, -1, 0])
    # plotter.add_mesh(pcd, color='green')
    # plotter.add_mesh(grid, color='red', point_size=5)
    # plotter.show()


if __name__ == '__main__':
    run_test()
