import os
from typing import Optional, Union

import numpy as np
import torch

from .generic_3d3d_registration_dataset import Generic3D3DRegistrationDataset


class ThreeDMatchDataset(Generic3D3DRegistrationDataset):
    def __init__(self,
                 ROOT: str,
                 meta_data: Union[str, None],
                 max_points: Optional[int] = None,
                 max_queries: Optional[int] = None,
                 grid_size: Optional[float] = 0.02,
                 downsample_voxel_size: Optional[float] = None,
                 matching_radius_3d: Optional[float] = 0.0375,
                 use_augmentation: bool = True,
                 augmentation_noise: float = 0.005,
                 normalize_points: bool = False,
                 bidirectional: bool = False,
                 **kwargs):
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

    def load_pcd(self, filepath) -> np.ndarray:
        pcd = torch.load(filepath, weights_only=False)
        return pcd

    def load_pose(self, filepath) -> np.ndarray:
        with open(filepath, 'r') as f:
            data = [line.rstrip() for line in f.readlines()[1:]]
        pose = np.fromstring(' '.join(data), sep=' ').reshape(4, 4)
        return pose[:3, :]

    def get_datapath(self, index):
        pcd_data = {'src_pcd': os.path.join(self.root, self.meta_data_list['src'][index]),
                    'tgt_pcd': os.path.join(self.root, self.meta_data_list['tgt'][index]),
                    'src_info': self.meta_data_list['src'][index].split('.')[0] + '.info.txt',
                    'tgt_info': self.meta_data_list['tgt'][index].split('.')[0] + '.info.txt'}
        return pcd_data

    def __len__(self):
        return len(self.meta_data_list['src'])

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index = index[0]
        
        src_pcd = self.load_pcd(os.path.join(self.root, self.meta_data_list['src'][index]))
        tgt_pcd = self.load_pcd(os.path.join(self.root, self.meta_data_list['tgt'][index]))
        src_info = self.meta_data_list['src'][index].split('.')[0] + '.info.txt'
        src_pose = self.load_pose(os.path.join(self.root, src_info))
        tgt_info = self.meta_data_list['tgt'][index].split('.')[0] + '.info.txt'
        tgt_pose = self.load_pose(os.path.join(self.root, tgt_info))
        tgt2src_transform = self.get_relative_pose(src_pose, tgt_pose)

        data_dict = self.construct_data_dict(src_pcd, tgt_pcd, tgt2src_transform)
        # data_dict['src'] = self.meta_data_list['src'][index]
        # data_dict['tgt'] = self.meta_data_list['tgt'][index]
        data_dict['dataset'] = "3DMatch"
        return data_dict


if __name__ == '__main__':
    from unicorrn.utils.vision3d.utils.visualization import draw_straight_correspondences
    from unicorrn.utils.vision3d.array_ops import apply_transform, denormalize_points_meta

    # with open('../datasets/3dmatch/train_info.pkl', 'rb') as f:
    #     data = pickle.load(f)
    threedmatch_demo = ThreeDMatchDataset('./sample_data/3dmatch/', None, max_points=5000, max_queries=500,
                                          use_augmentation=False, normalize_points=True)
    src_pcd = threedmatch_demo.load_pcd('./sample_data/3dmatch/rgbd/cloud_bin_8.pth')
    src_pose = threedmatch_demo.load_pose('./sample_data/3dmatch/rgbd/cloud_bin_8.info.txt')
    tgt_pcd = threedmatch_demo.load_pcd('./sample_data/3dmatch/rgbd/cloud_bin_49.pth')
    tgt_pose = threedmatch_demo.load_pose('./sample_data/3dmatch/rgbd/cloud_bin_49.info.txt')

    tgt2src_transform = threedmatch_demo.get_relative_pose(src_pose, tgt_pose)
    data_dict = threedmatch_demo.construct_data_dict(src_pcd, tgt_pcd, tgt2src_transform)
    tgt2src_transform = data_dict['tgt2src_transform']

    src_norm_meta = data_dict['src_norm_meta']
    tgt_norm_meta = data_dict['tgt_norm_meta']

    draw_straight_correspondences(data_dict['src_pcd'],
                                  data_dict['tgt_pcd'],
                                  data_dict['queries'],
                                  data_dict['targets'],
                                  offsets=(0., 2., 0.))
    # src_pcd_raw = denormalize_points_meta(data_dict['src_pcd'], src_norm_meta)
    # tgt_pcd_raw = denormalize_points_meta(data_dict['tgt_pcd'], tgt_norm_meta)
    # queries_raw = denormalize_points_meta(data_dict['queries'], src_norm_meta)
    # targets_raw = denormalize_points_meta(data_dict['targets'], tgt_norm_meta)
    #
    # tgt_pcd_a = apply_transform(tgt_pcd_raw, tgt2src_transform)
    # targets_a = apply_transform(targets_raw, tgt2src_transform)
    # draw_straight_correspondences(src_pcd_raw,
    #                               tgt_pcd_a,
    #                               queries_raw,
    #                               targets_a,
    #                               offsets=(0., 2., 0.))
