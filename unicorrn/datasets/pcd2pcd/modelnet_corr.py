import os
from typing import Optional, Union

import numpy as np

import h5py

from .generic_3d3d_registration_dataset import Generic3D3DRegistrationDataset
from ...utils.vision3d.array_ops import inverse_transform, apply_transform
from ...utils.vision3d.array_ops.point_cloud_utils import (
    random_sphere_crop,
    random_sample_transform
)


class ModelNetDataset(Generic3D3DRegistrationDataset):
    def __init__(self,
                 ROOT: str,
                 meta_data: Union[str, None],
                 category_file: Union[str, None],
                 max_points: Optional[int] = None,
                 max_queries: Optional[int] = None,
                 grid_size: Optional[float] = 0.02,
                 downsample_voxel_size: Optional[float] = None,
                 matching_radius_3d: Optional[float] = 0.0375,
                 use_augmentation: bool = True,
                 augmentation_noise: float = 0.005,
                 normalize_points: bool = False,
                 bidirectional: bool = False,
                 keep_ratio: float = 0.7,
                 return_raw: bool = False,
                 deterministic: bool = False,
                 **kwargs):
        with open(os.path.join(ROOT, 'shape_names.txt'), 'r') as f:
            self._classes = f.readlines()
        self._category2idx = {e[1]: e[0] for e in enumerate(self._classes)}
        self._idx2category = self._classes
        self.categories = None
        if category_file is not None:
            with open(category_file, 'r') as f:
                category_labels = sorted(f.readlines())
            self.categories = [self._category2idx[l] for l in category_labels]

        super().__init__(root=ROOT,
                         meta_data=meta_data,
                         max_points=max_points,
                         max_queries=max_queries,
                         grid_size=grid_size,
                         downsample_voxel_size=downsample_voxel_size,
                         matching_radius_3d=matching_radius_3d,
                         use_augmentation=use_augmentation,
                         augmentation_noise=augmentation_noise,
                         normalize_points=normalize_points,
                         bidirectional=bidirectional)
        self.keep_ratio = keep_ratio
        self.return_raw = return_raw
        self.deterministic = deterministic

    def parse_meta_data(self, filepath) -> None:
        with open(filepath, 'r') as f:
            files = f.readlines()

        pcd_data = []
        for file in files:
            pcd, labels = self.load_pcd(os.path.join(self.root, file.rstrip()))
            valid_categories = np.isin(
                labels,
                self.categories
            ) if self.categories is not None else np.ones_like(labels, dtype=bool)
            pcd_data.append(pcd[valid_categories])
            # pcd_data.append(self.load_pcd(os.path.join(self.root, file.rstrip())))
        self.data_cache['pcd'] = [sample for sample in np.concatenate(pcd_data, axis=0)]

    def load_pcd(self, filepath) -> [np.ndarray, np.ndarray]:
        with h5py.File(filepath, 'r') as h5:
            pcd = np.asarray(h5['data'], dtype=np.float32)
            labels = np.asarray(h5['label']).squeeze(-1).astype(int)
        return pcd, labels

    def _apply_augmentation(self, pcd):
        aug_transform = random_sample_transform(45, 0.5)
        pcd = apply_transform(pcd, aug_transform)
        pcd += (np.random.rand(pcd.shape[0], 3) - 0.5) * self.aug_noise

        return pcd, aug_transform

    def __len__(self):
        return len(self.data_cache['pcd'])

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index = index[0]
        if self.deterministic:
            np.random.seed(index)

        src_pcd = self.data_cache['pcd'][index].copy()
        tgt_pcd = src_pcd.copy()
        if self.keep_ratio is not None:
            src_pcd = random_sphere_crop(src_pcd, keep_ratio=self.keep_ratio)
            tgt_pcd = random_sphere_crop(tgt_pcd, keep_ratio=self.keep_ratio)

        tgt_pcd, transform = self._apply_augmentation(tgt_pcd)
        tgt2src_transform = inverse_transform(transform)

        data_dict = self.construct_data_dict(src_pcd, tgt_pcd, tgt2src_transform)

        if self.return_raw:
            data_dict['raw_pcd'] = self.data_cache['pcd'][index]

        data_dict['dataset'] = "ModelNet"
        return data_dict


if __name__ == '__main__':
    from unicorrn.utils.vision3d.array_ops import apply_transform, denormalize_points_meta
    from unicorrn.utils.vision3d.utils.visualization import draw_straight_correspondences

    modelnet_demo = ModelNetDataset('./sample_data/modelnet/', None, max_points=None, max_queries=None,
                                    normalize_points=True)
    src_pcd = modelnet_demo.load_pcd(os.path.join(modelnet_demo.root, 'ply_data_train0.h5'))[10]
    tgt_pcd = src_pcd.copy()

    tgt_pcd, transform = modelnet_demo._apply_small_augmentation(tgt_pcd, scale=1.0)
    tgt2src_transform = inverse_transform(transform)

    data_dict = modelnet_demo.construct_data_dict(src_pcd, tgt_pcd, tgt2src_transform)
    src_norm_meta = data_dict['src_norm_meta']
    tgt_norm_meta = data_dict['tgt_norm_meta']
    src_pcd = denormalize_points_meta(data_dict['src_pcd'], src_norm_meta)
    tgt_pcd = denormalize_points_meta(data_dict['tgt_pcd'], tgt_norm_meta)
    queries = denormalize_points_meta(data_dict['queries'], src_norm_meta)
    targets = denormalize_points_meta(data_dict['targets'], tgt_norm_meta)

    # tgt_pcd = apply_transform(tgt_pcd, tgt2src_transform)
    # targets = apply_transform(targets, tgt2src_transform)
    draw_straight_correspondences(src_pcd, tgt_pcd, queries, targets, offsets=(0., 2., 0.))
