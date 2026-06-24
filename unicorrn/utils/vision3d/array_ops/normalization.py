import numpy as np
from typing import Optional, Union, Tuple, List


def min_max_norm(
        a: np.ndarray,
        *,
        axis: int = 0,
        min_val: Optional[Union[np.ndarray, Tuple, List]] = None,
        max_val: Optional[Union[np.ndarray, Tuple, List]] = None,
        eps: Optional[float] = 1e-8
):
    a = a.astype(np.float32)
    min_val = np.min(a, axis=axis, keepdims=True) if min_val is None else np.asarray(min_val)
    max_val = np.max(a, axis=axis, keepdims=True) if max_val is None else np.asarray(max_val)

    norm = (a - min_val) / (max_val - min_val + eps)
    return norm


def center_shift(points, apply_z=True):
    """
        https://github.com/Pointcept/Pointcept/blob/main/pointcept/datasets/transform.py
    """
    x_min, y_min, z_min = points.min(axis=0)
    x_max, y_max, _ = points.max(axis=0)

    if apply_z:
        shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
    else:
        shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]

    points -= shift
    return points


def center_shift_corr_points(corr_points, points, apply_z=False):
    """
        https://github.com/Pointcept/Pointcept/blob/main/pointcept/datasets/transform.py
    """
    x_min, y_min, z_min = points.min(axis=0)
    x_max, y_max, _ = points.max(axis=0)

    if apply_z:
        shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
    else:
        shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]

    corr_points -= shift
    return corr_points


def normalize_coord(points, return_meta=False, eps=1e-8):
    """
        https://github.com/Pointcept/Pointcept/blob/main/pointcept/datasets/transform.py

        unit sphere normalization
    """
    # modified from pointnet2
    centroid = np.mean(points, axis=0, dtype=np.float32)
    points_center = points - centroid
    m = np.max(np.sqrt(np.sum(points_center ** 2, axis=1), dtype=np.float32))
    norm = points_center / (m + eps)
    if return_meta:
        norm_meta = {'centroid': centroid, 'length': m}
        return norm, norm_meta
    return norm


def normalize_coord_corr_points(corr_points, points, eps=1e-8):
    centroid = np.mean(points, axis=0)
    m = np.max(np.sqrt(np.sum((points - centroid) ** 2, axis=1)))

    norm_center = corr_points - centroid
    norm = norm_center / (m + eps)
    return norm
   

def denormalize_coord_corr_points(corr_points, points):
    centroid = np.mean(points, axis=0)
    m = np.max(np.sqrt(np.sum((points - centroid) ** 2, axis=1)))

    corr_points = corr_points * m
    corr_points += centroid
    return corr_points


def denormalize_points_meta(norm_points, norm_meta):
    points_center = norm_points * norm_meta['length']
    denorm_points = points_center + norm_meta['centroid']
    return denorm_points
