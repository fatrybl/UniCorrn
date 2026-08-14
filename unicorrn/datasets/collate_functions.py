import torch
from torch.utils.data.dataloader import default_collate
from typing import Sequence, Mapping, List, Callable, Optional
import numpy as np
from itertools import chain
from ..utils.vision3d.array_ops import build_grid_and_radius_graph_pyramid_pack_mode


def array_to_tensor(x):
    """Convert all numpy arrays to pytorch tensors."""
    if isinstance(x, list):
        x = [array_to_tensor(item) for item in x]
    elif isinstance(x, tuple):
        x = tuple([array_to_tensor(item) for item in x])
    elif isinstance(x, dict):
        x = {key: array_to_tensor(value) for key, value in x.items()}
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x


def concat_point_feats(feats_list, points_list):
    """Concatenate per-sample [xyz, extra-channel] feature arrays.

    Samples without extra channels (None entries) are padded with zeros so that
    mixed batches stay well-defined: a zero extra channel carries no signal and,
    with zero-initialized new input weights, reproduces the xyz-only computation.
    """
    feat_dim = next(f.shape[1] for f in feats_list if f is not None)
    feats_list = [
        f if f is not None else np.concatenate(
            [p, np.zeros((p.shape[0], feat_dim - p.shape[1]), dtype=p.dtype)], axis=1
        )
        for f, p in zip(feats_list, points_list)
    ]
    return np.concatenate(feats_list, axis=0)


def collate_dict(data_dicts: List[dict]) -> dict:
    """Collate a batch of dict.

    The collated dict contains all keys from the batch, with each key mapped to a list of data. If a certain key is
    missing in one dict, `None` is used for padding so that all lists have the same length (the batch size).

    Args:
        data_dicts (List[dict]): A batch of data dicts.

    Returns:
        A dict with all data collated.
    """
    keys = set(chain(*[list(data_dict.keys()) for data_dict in data_dicts]))
    collated_dict = {key: [data_dict.get(key) for data_dict in data_dicts] for key in keys}
    return collated_dict


class PointCloudRegistrationCollateFn(Callable):
    def __init__(self,
                 batch_keys: Optional[Sequence[str]] = None):
        self.batch_keys = batch_keys

    def __call__(self,
                 data_dicts: List[dict]):
        batch_size = len(data_dicts)

        # 1. collate dict
        # if "pcd2pcd" in data_dicts[0].keys():
        #     batch = collate_dict(data_dicts)
        #     collated_dict = collate_dict(batch.pop("pcd2pcd"))
        # else:
        collated_dict = collate_dict(data_dicts)

        if batch_size == 1:
            collated_dict = {key: value[0] for key, value in collated_dict.items()}
            collated_dict["src_length"] = np.asarray([collated_dict["src_pcd"].shape[0]])
            collated_dict["tgt_length"] = np.asarray([collated_dict["tgt_pcd"].shape[0]])
        else:
            src_points_list = collated_dict.pop("src_pcd")
            tgt_points_list = collated_dict.pop("tgt_pcd")
            collated_dict["src_pcd"] = np.concatenate(src_points_list, axis=0)
            collated_dict["tgt_pcd"] = np.concatenate(tgt_points_list, axis=0)
            collated_dict["src_length"] = np.asarray([points.shape[0] for points in src_points_list])
            collated_dict["tgt_length"] = np.asarray([points.shape[0] for points in tgt_points_list])
            if "src_feats" in collated_dict:
                collated_dict["src_feats"] = concat_point_feats(collated_dict.pop("src_feats"), src_points_list)
            if "tgt_feats" in collated_dict:
                collated_dict["tgt_feats"] = concat_point_feats(collated_dict.pop("tgt_feats"), tgt_points_list)

            # additional attributes
            collated_dict["src_grid_coord"] = np.concatenate(collated_dict.pop("src_grid_coord"), axis=0)
            collated_dict["tgt_grid_coord"] = np.concatenate(collated_dict.pop("tgt_grid_coord"), axis=0)

        collated_dict["batch_size"] = batch_size
        if self.batch_keys is not None:
            for key in self.batch_keys:
                if batch_size > 1:
                    collated_dict[key] = np.stack(collated_dict.pop(key), axis=0)
                else:
                    collated_dict[key] = collated_dict.pop(key)[None]

        # 4. array to tensor
        collated_dict = array_to_tensor(collated_dict)

        return collated_dict


class GraphPyramid2D3DRegistrationCollateFn(Callable):
    def __init__(self, num_stages: int, voxel_size: float, search_radius: float, neighbor_limits: List[int]):
        self.num_stages = num_stages
        self.voxel_size = voxel_size
        self.search_radius = search_radius
        self.neighbor_limits = neighbor_limits

    def __call__(self, data_dicts: List[dict]):
        batch_size = len(data_dicts)

        # 1. collate dict
        collated_dict = collate_dict(data_dicts)

        # 2. handle batch size
        image = np.stack(collated_dict.pop("image"), axis=0)  # (B, *, H, W)
        depth = np.stack(collated_dict.pop("depth"), axis=0)  # (B, H, W)

        # additional attributes
        queries =  np.stack(collated_dict.pop("queries"), axis=0)
        targets = np.stack(collated_dict.pop("targets"), axis=0)
        norm_queries = np.stack(collated_dict.pop("norm_queries"), axis=0)
        norm_targets = np.stack(collated_dict.pop("norm_targets"), axis=0)


        if batch_size == 1:
            collated_dict = {key: value[0] for key, value in collated_dict.items()}
            collated_dict["lengths"] = np.asarray([collated_dict["points"].shape[0]])
        else:
            points_list = collated_dict.pop("points")
            collated_dict["points"] = np.concatenate(points_list, axis=0)
            collated_dict["lengths"] = np.asarray([points.shape[0] for points in points_list])
            collated_dict["feats"] = np.concatenate(collated_dict.pop("feats"), axis=0)
            collated_dict["intrinsics"] = np.stack(collated_dict.pop("intrinsics"), axis=0)  # (B, 3, 3)

            # additional attributes
            collated_dict["grid_coord"] = np.concatenate(collated_dict.pop("grid_coord"), axis=0)
            
        collated_dict["image"] = image
        collated_dict["depth"] = depth

        collated_dict["batch_size"] = batch_size

        # additional attributes
        collated_dict["queries"] = queries
        collated_dict["targets"] = targets
        collated_dict["norm_queries"] = norm_queries
        collated_dict["norm_targets"] = norm_targets

        # 3. build graph pyramid
        points = collated_dict.pop("points")
        lengths = collated_dict.pop("lengths")
        graph_pyramid_dict = build_grid_and_radius_graph_pyramid_pack_mode(
            points, lengths, self.num_stages, self.voxel_size, self.search_radius, self.neighbor_limits
        )
        collated_dict.update(graph_pyramid_dict)

        # 4. array to tensor
        collated_dict = array_to_tensor(collated_dict)

        return collated_dict


class ImageToPointRegistrationCollateFn(Callable):
    def __call__(self, data_dicts: List[dict]):
        batch_size = len(data_dicts)

        # 1. collate dict
        # if "img2pcd" in data_dicts[0].keys():
        #     batch = collate_dict(data_dicts)
        #     print(batch.keys())
        #     collated_dict = collate_dict(batch.pop("img2pcd"))
        # else:
        collated_dict = collate_dict(data_dicts)

        # 2. handle batch size
        image = np.stack(collated_dict.pop("image"), axis=0)  # (B, *, H, W)
        depth = np.stack(collated_dict.pop("depth"), axis=0)  # (B, H, W)

        # additional attributes
        queries =  np.stack(collated_dict.pop("queries"), axis=0)
        targets = np.stack(collated_dict.pop("targets"), axis=0)
        norm_queries = np.stack(collated_dict.pop("norm_queries"), axis=0)
        norm_targets = np.stack(collated_dict.pop("norm_targets"), axis=0)


        if batch_size == 1:
            collated_dict = {key: value[0] for key, value in collated_dict.items()}
            collated_dict["lengths"] = np.asarray([collated_dict["points"].shape[0]])
        else:
            points_list = collated_dict.pop("points")
            collated_dict["points"] = np.concatenate(points_list, axis=0)
            collated_dict["lengths"] = np.asarray([points.shape[0] for points in points_list])
            if "feats" in collated_dict:
                collated_dict["feats"] = concat_point_feats(collated_dict.pop("feats"), points_list)
            collated_dict["intrinsics"] = np.stack(collated_dict.pop("intrinsics"), axis=0)  # (B, 3, 3)

            # additional attributes
            collated_dict["grid_coord"] = np.concatenate(collated_dict.pop("grid_coord"), axis=0)

        collated_dict["image"] = image
        collated_dict["depth"] = depth

        collated_dict["batch_size"] = batch_size

        # additional attributes
        collated_dict["queries"] = queries
        collated_dict["targets"] = targets
        collated_dict["norm_queries"] = norm_queries
        collated_dict["norm_targets"] = norm_targets
        # collated_dict['norm_length'] = np.asarray(collated_dict.pop('norm_length'))

        if data_dicts[0].get("frustum_queries") is not None:
            for key in ("frustum_queries", "frustum_labels", "frustum_distances"):
                val = collated_dict[key]
                collated_dict[key] = val[None] if isinstance(val, np.ndarray) else np.stack(val, axis=0)

        # 3. array to tensor
        collated_dict = array_to_tensor(collated_dict)

        return collated_dict


class JointTrainingCollateFn(Callable):
    def __init__(self, enable_img2img=True, enable_img2pcd=True, enable_pcd2pcd=True):
        self.enable_img2img = enable_img2img
        self.enable_img2pcd = enable_img2pcd
        self.enable_pcd2pcd = enable_pcd2pcd

    def _prepare_img2img(self, data_dicts: List[dict]):
        batch_size = len(data_dicts)

        # 1. collate dict
        batch = collate_dict(data_dicts)

        if not self.enable_img2img:
            return

        if "img2img" not in batch.keys():
            collated_dict = batch
        else:
            collated_dict = collate_dict(batch.pop("img2img"))

        # 2. handle batch size
        img1 = torch.stack(collated_dict.pop("img1"), dim=0)  # (B, *, H, W)
        img2 = torch.stack(collated_dict.pop("img2"), dim=0)  # (B, *, H, W)
        depth1 = torch.stack(collated_dict.pop("depth1"), dim=0)  # (B, H, W)
        depth2 = torch.stack(collated_dict.pop("depth2"), dim=0)  # (B, H, W)

        # additional attributes
        queries =  torch.stack(collated_dict.pop("queries"), dim=0)
        targets = torch.stack(collated_dict.pop("targets"), dim=0)
        norm_queries = torch.stack(collated_dict.pop("norm_queries"), dim=0)
        norm_targets = torch.stack(collated_dict.pop("norm_targets"), dim=0)

        if batch_size == 1:
            collated_dict = {key: value[0] for key, value in collated_dict.items()}
        else:
            collated_dict["K1"] = np.stack(collated_dict.pop("K1"), axis=0)  # (B, 3, 3)
            collated_dict["K2"] = np.stack(collated_dict.pop("K2"), axis=0)  # (B, 3, 3)
            collated_dict["camera_pose1"] = np.stack(collated_dict.pop("camera_pose1"), axis=0)  # (B, 4, 4)
            collated_dict["camera_pose2"] = np.stack(collated_dict.pop("camera_pose2"), axis=0)  # (B, 4, 4)

            
        collated_dict["img1"] = img1
        collated_dict["img2"] = img2
        collated_dict["depth1"] = depth1
        collated_dict["depth2"] = depth2

        # additional attributes
        collated_dict["queries"] = queries
        collated_dict["targets"] = targets
        collated_dict["norm_queries"] = norm_queries
        collated_dict["norm_targets"] = norm_targets

        return collated_dict

    def _prepare_img2pcd(self, data_dicts: List[dict]):
        batch_size = len(data_dicts)

        # 1. collate dict
        batch = collate_dict(data_dicts)

        if not self.enable_img2pcd:
            return

        if "img2pcd" not in batch.keys():
            collated_dict = batch
        else:
            collated_dict = collate_dict(batch.pop("img2pcd"))

        # 2. handle batch size
        image = np.stack(collated_dict.pop("image"), axis=0)  # (B, *, H, W)
        depth = np.stack(collated_dict.pop("depth"), axis=0)

        # additional attributes
        queries =  np.stack(collated_dict.pop("queries"), axis=0)
        targets = np.stack(collated_dict.pop("targets"), axis=0)
        norm_queries = np.stack(collated_dict.pop("norm_queries"), axis=0)
        norm_targets = np.stack(collated_dict.pop("norm_targets"), axis=0)

        if batch_size == 1:
            collated_dict = {key: value[0] for key, value in collated_dict.items()}
            collated_dict["lengths"] = np.asarray([collated_dict["points"].shape[0]])
        else:
            points_list = collated_dict.pop("points")
            collated_dict["points"] = np.concatenate(points_list, axis=0)
            collated_dict["lengths"] = np.asarray([points.shape[0] for points in points_list])
            if "feats" in collated_dict:
                collated_dict["feats"] = concat_point_feats(collated_dict.pop("feats"), points_list)
            collated_dict["intrinsics"] = np.stack(collated_dict.pop("intrinsics"), axis=0)  # (B, 3, 3)

            # additional attributes
            collated_dict["grid_coord"] = np.concatenate(collated_dict.pop("grid_coord"), axis=0)

        collated_dict["image"] = image
        collated_dict["depth"] = depth
        collated_dict["batch_size"] = batch_size

        # additional attributes
        collated_dict["queries"] = queries
        collated_dict["targets"] = targets
        collated_dict["norm_queries"] = norm_queries
        collated_dict["norm_targets"] = norm_targets

        # 3. array to tensor
        collated_dict = array_to_tensor(collated_dict)
        return collated_dict

    def _prepare_pcd2pcd(self, data_dicts: List[dict]):
        batch_size = len(data_dicts)

        # 1. collate dict
        batch = collate_dict(data_dicts)

        if not self.enable_pcd2pcd:
            return

        if "pcd2pcd" not in batch.keys():
            collated_dict = batch
        else:
            collated_dict = collate_dict(batch.pop("pcd2pcd"))     

        # additional attributes
        queries =  np.stack(collated_dict.pop("queries"), axis=0)
        targets = np.stack(collated_dict.pop("targets"), axis=0)
        norm_queries = np.stack(collated_dict.pop("norm_queries"), axis=0)
        norm_targets = np.stack(collated_dict.pop("norm_targets"), axis=0)
        
        if batch_size == 1:
            collated_dict = {key: value[0] for key, value in collated_dict.items()}
            collated_dict["src_length"] = np.asarray([collated_dict["src_pcd"].shape[0]])
            collated_dict["tgt_length"] = np.asarray([collated_dict["tgt_pcd"].shape[0]])
        else:
            src_points_list = collated_dict.pop("src_pcd")
            tgt_points_list = collated_dict.pop("tgt_pcd")
            collated_dict["src_pcd"] = np.concatenate(src_points_list, axis=0)
            collated_dict["tgt_pcd"] = np.concatenate(tgt_points_list, axis=0)
            collated_dict["src_length"] = np.asarray([points.shape[0] for points in src_points_list])
            collated_dict["tgt_length"] = np.asarray([points.shape[0] for points in tgt_points_list])
            if "src_feats" in collated_dict:
                collated_dict["src_feats"] = concat_point_feats(collated_dict.pop("src_feats"), src_points_list)
            if "tgt_feats" in collated_dict:
                collated_dict["tgt_feats"] = concat_point_feats(collated_dict.pop("tgt_feats"), tgt_points_list)

            # additional attributes
            collated_dict["src_grid_coord"] = np.concatenate(collated_dict.pop("src_grid_coord"), axis=0)
            collated_dict["tgt_grid_coord"] = np.concatenate(collated_dict.pop("tgt_grid_coord"), axis=0)

        collated_dict["batch_size"] = batch_size

        # additional attributes
        collated_dict["queries"] = queries
        collated_dict["targets"] = targets
        collated_dict["norm_queries"] = norm_queries
        collated_dict["norm_targets"] = norm_targets

        # 2. array to tensor
        collated_dict = array_to_tensor(collated_dict)
        return collated_dict

    def __call__(self, data_dicts: List[dict]):
        batch = {}
        batch_img2img = self._prepare_img2img(data_dicts)
        batch_img2pcd = self._prepare_img2pcd(data_dicts)
        batch_pcd2pcd = self._prepare_pcd2pcd(data_dicts)

        if batch_img2img is not None:
            batch["img2img"] = batch_img2img
        if batch_img2pcd is not None:
            batch["img2pcd"] = batch_img2pcd
        if batch_pcd2pcd is not None:
            batch["pcd2pcd"] = batch_pcd2pcd

        return batch