"""Geometric frustum labels for 3D point queries.

Given a point cloud and the ground-truth camera (intrinsics K and cloud->image
extrinsic T), a point P is labelled in-frustum iff it projects in front of the
camera and inside the image extent:

    P_cam = R @ P + t,        z = P_cam_z
    u = fx * x / z + cx,      v = fy * y / z + cy
    in_frustum = (z > 0) and (0 <= u < W) and (0 <= v < H)

This is the geometric (cone) definition: it ignores occlusion and texture by
design. T and K are used here for *labelling only*; the model never receives
them as input, so the learned classifier stays pose-independent.
"""

import numpy as np
from numpy import ndarray

from ...utils.vision3d.array_ops import apply_transform, normalize_coord_corr_points


def _resample(idx: ndarray, n: int) -> ndarray:
    """Resample indices to a fixed count, with replacement when scarce."""
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    if len(idx) == 0:
        return np.zeros(n, dtype=np.int64)
    return np.random.choice(idx, n, replace=len(idx) < n)


def build_frustum_queries(
    points: ndarray,
    transform: ndarray,
    intrinsics: ndarray,
    image_h: int,
    image_w: int,
    num_queries: int,
    eps: float = 1e-6,
) -> tuple[ndarray, ndarray]:
    """Sample a class-balanced set of point queries with in/out-frustum labels.

    Returns normalised 3D query coordinates (matching the model's point
    normalisation) and float {0, 1} membership labels.
    """
    cam = apply_transform(points, transform)
    z = cam[:, 2]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u = fx * cam[:, 0] / (z + eps) + cx
    v = fy * cam[:, 1] / (z + eps) + cy
    in_frustum = (z > 0) & (u >= 0) & (u < image_w) & (v >= 0) & (v < image_h)

    n_pos = num_queries // 2
    pos_sel = _resample(np.where(in_frustum)[0], n_pos)
    neg_sel = _resample(np.where(~in_frustum)[0], num_queries - n_pos)
    sel = np.concatenate([pos_sel, neg_sel])

    queries = normalize_coord_corr_points(points[sel], points).astype(np.float32)
    labels = in_frustum[sel].astype(np.float32)
    return queries, labels
