"""
Code borrowed from Loftr
https://github.com/zju3dv/LoFTR/blob/master/src/datasets/scannet.py
"""


import io

import cv2
import numpy as np
import h5py
import torch

import torch.utils as utils

from numpy.linalg import inv
from os import path as osp


from .base.base_stereo_view_dataset import BaseStereoViewDataset

def read_scannet_img(path):
    """

    Returns:
        image (torch.tensor): (1, h, w)
      
    """
    # read and resize image
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640,480), interpolation=cv2.INTER_LINEAR)
    return image

def read_scannet_depth(path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    depth = depth / 1000
    depth[~np.isfinite(depth)] = 0  # invalid
    return depth

def read_scannet_pose(path):
    """ Read ScanNet's Camera2World pose and transform it to World2Camera.
    
    Returns:
        pose_c2w (np.ndarray): (4, 4)
    """
    cam2world = np.loadtxt(path, delimiter=' ')
    return cam2world

def read_scannet_intrinsic(path):
    """ Read ScanNet's intrinsic matrix and return the 3x3 matrix.
    """
    intrinsic = np.loadtxt(path, delimiter=' ')
    return intrinsic[:-1, :-1]

class ScanNetDataset(BaseStereoViewDataset):
    def __init__(self,
                 *args,
                 ROOT,
                 split='train',
                 **kwargs):
        """Manage one scene of ScanNet Dataset.
        Args:
            root_dir (str): ScanNet root directory that contains scene folders.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            intrinsic_path (str): path to depth-camera intrinsic file.
            mode (str): options are ['train', 'val', 'test'].
            augment_fn (callable, optional): augments images with pre-defined visual effects.
            pose_dir (str): ScanNet root directory that contains all poses.
                (we use a separate (optional) pose_dir since we store images and poses separately.)
        """
        super().__init__(*args, **kwargs)
        assert split == "train", f"{split} not supported."
        
        self.ROOT = ROOT
        self.split = split

        scene_data_path = osp.join(ROOT, "scannet_indices", "scene_data")
        intrinsic_path = osp.join(ROOT, "scannet_indices", "intrinsics.npz")
        scene_list_path = osp.join(scene_data_path, "train_list", "scannet_all.txt")
        npz_path  = osp.join(scene_data_path, split)

        self.ROOT = osp.join(ROOT, "scans_uncomp")
        
        with open(scene_list_path, 'r') as f:
            npz_names = [name.split()[0] for name in f.readlines()]

        self.data_names = []
        min_overlap_score=0.4
        
        for name in npz_names:
            # prepare data_names, intrinsics and extrinsics(T)
            with np.load(osp.join(npz_path, name)) as data:
                scene_data_names = data['name']

                # Only sample 10s
                valid = (scene_data_names[:,-2:] % 10).sum(axis=-1) == 0
                scene_data_names = scene_data_names[valid]
                
                if 'score' in data.keys() and split not in ['val' or 'test']:
                    kept_mask = data['score'][valid] > min_overlap_score
                    scene_data_names = scene_data_names[kept_mask]

                if len(scene_data_names) > 10000:
                    pairinds = np.random.choice(np.arange(0,len(scene_data_names)),10000,replace=False)
                    scene_data_names = scene_data_names[pairinds]
                    
                self.data_names.append(scene_data_names)
                    
        self.data_names = np.vstack(self.data_names)
        self.intrinsics = dict(np.load(intrinsic_path))


    def __len__(self):
        return len(self.data_names)

    def _read_abs_pose(self, scene_name, name):
        pth = osp.join(self.ROOT,
                       scene_name,
                       'pose', f'frame-{name:06d}.pose.txt') 
        return read_scannet_pose(pth)

    def _compute_rel_pose(self, scene_name, name0, name1):
        pose0 = self._read_abs_pose(scene_name, name0)
        pose1 = self._read_abs_pose(scene_name, name1)
        
        return np.matmul(pose1, inv(pose0))  # (4, 4)

    def _get_views(self, idx, resolution, rng):
        data_name = self.data_names[idx]
        scene_name, scene_sub_name, stem_name_0, stem_name_1 = data_name
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'

        # read the grayscale image which will be resized to (1, 480, 640)
        img_name0 = osp.join(self.ROOT, scene_name, 'color', f"frame-{stem_name_0:06d}.color.jpg")
        img_name1 = osp.join(self.ROOT, scene_name, 'color', f"frame-{stem_name_1:06d}.color.jpg")

        depth_name0 = osp.join(self.ROOT, scene_name, 'depth', f'{stem_name_0}.png')
        depth_name1 = osp.join(self.ROOT, scene_name, 'depth', f'{stem_name_1}.png')

        if not osp.exists(img_name0) or not osp.exists(img_name1) or not osp.exists(depth_name0) or not osp.exists(depth_name1):
            if idx + 1 < len(self):
                return self._get_views(idx + 1, resolution, rng)
            else:
                return self._get_views(0, resolution, rng)

        image0 = read_scannet_img(img_name0)
        image1 = read_scannet_img(img_name1)


        # read the depthmap which is stored as (480, 640)
        if self.split in ['train', 'val']:
            depth0 = read_scannet_depth(osp.join(self.ROOT, scene_name, 'depth', f'{stem_name_0}.png'))
            depth1 = read_scannet_depth(osp.join(self.ROOT, scene_name, 'depth', f'{stem_name_1}.png'))
        else:
            depth0 = depth1 = torch.tensor([])

        # read the intrinsic of depthmap
        K_0 = K_1 = self.intrinsics[scene_name].copy().reshape(3, 3)

        # read camera to world
        camera_pose0 = self._read_abs_pose(scene_name, stem_name_0)
        camera_pose1 = self._read_abs_pose(scene_name, stem_name_1)

        views = []
        
        view_idx = 0
        image0, depth0, K_0 = self._crop_resize_if_necessary(image0, depth0, K_0, resolution, rng=rng, info=view_idx)

        views.append(dict(
            img=image0,
            depthmap=depth0.astype(np.float32),
            camera_pose=camera_pose0.astype(np.float32),
            camera_intrinsics=K_0.astype(np.float32),
            dataset='ScanNet',
            label=scene_name + '_' + f'{stem_name_0}.jpg',
            instance=f'{str(idx)}_{str(view_idx)}',            
            )
        )
        
        view_idx = 1
        image1, depth1, K_1 = self._crop_resize_if_necessary(image1, depth1, K_1, resolution, rng=rng, info=view_idx)

        views.append(dict(
            img=image1,
            depthmap=depth1.astype(np.float32),
            camera_pose=camera_pose1.astype(np.float32),
            camera_intrinsics=K_1.astype(np.float32),
            dataset='ScanNet',
            label=scene_name + '_' + f'{stem_name_1}.jpg',
            instance=f'{str(idx)}_{str(view_idx)}',            
            )
        )
        
        return views