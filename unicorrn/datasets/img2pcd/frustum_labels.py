"""Geometric frustum labels and signed-distance targets for 3D point queries.

Given a point cloud and the ground-truth camera (intrinsics K and cloud->image
extrinsic T), a point P is labelled in-frustum iff it projects in front of the
camera and inside the image extent:

    P_cam = R @ P + t,        z = P_cam_z
    u = fx * x / z + cx,      v = fy * y / z + cy
    in_frustum = (z > z_min) and (0 <= u < W) and (0 <= v < H)

Membership is also emitted as a signed distance, so supervision is graded across
the boundary instead of being a step function of position. For the projection
normalised to the unit rectangle and ``q = |u - 1/2| - 1/2``,

    s(u) = -( ||max(q, 0)||_2 + min(max(q_x, q_y), 0) )

is the exact Euclidean signed distance to that rectangle, positive inside and negative
outside, clamped so its range stays bounded. Points at or behind ``z_min`` take the
fully-outside value: their projection is the mirrored image and would otherwise land
inside the rectangle carrying an out-of-frustum label.

T and K are used here for *labelling only*; the model never receives them as input, so
the learned classifier stays pose-independent.
"""

import numpy as np
from numpy import ndarray

from ...utils.vision3d.array_ops import apply_transform, normalize_coord_corr_points

_MIN_DEPTH = 1e-3
_BOX_CENTRE = 0.5
_SATURATION = 0.5


def _resample(idx: ndarray, n: int) -> ndarray:
    """Resample indices to a fixed count, with replacement when scarce.

    Args:
        idx: Candidate indices to draw from.
        n: Number of indices to return.
    """
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    if len(idx) == 0:
        return np.zeros(n, dtype=np.int64)
    return np.random.choice(idx, n, replace=len(idx) < n)


def _signed_distance(uv: ndarray, z: ndarray, image_h: int, image_w: int) -> ndarray:
    """Signed distance of each projection to the image rectangle.

    Args:
        uv: Pixel coordinates of the projections.
        z: Camera-frame depth of the same points.
        image_h: Image height in pixels.
        image_w: Image width in pixels.
    """
    normalised = uv / np.array([image_w, image_h], dtype=uv.dtype)
    q = np.abs(normalised - _BOX_CENTRE) - _BOX_CENTRE
    outside = np.linalg.norm(np.clip(q, 0.0, None), axis=-1)
    inside = np.clip(q.max(axis=-1), None, 0.0)
    distance = np.where(z > _MIN_DEPTH, -(outside + inside), -_SATURATION)
    return np.clip(distance, -_SATURATION, _SATURATION)


def build_frustum_queries(
    points: ndarray,
    transform: ndarray,
    intrinsics: ndarray,
    image_h: int,
    image_w: int,
    num_queries: int,
) -> tuple[ndarray, ndarray, ndarray]:
    """Sample a class-balanced set of point queries with in/out-frustum supervision.

    Returns normalised 3D query coordinates (matching the model's point normalisation),
    float {0, 1} membership labels, and the signed distance of each query's projection
    to the frustum boundary.

    Args:
        points: Point cloud in the cloud frame.
        transform: Ground-truth cloud-to-camera extrinsic.
        intrinsics: Camera intrinsic matrix in the model's image frame.
        image_h: Image height in pixels.
        image_w: Image width in pixels.
        num_queries: Total queries to draw, split evenly between the two classes.
    """
    cam = apply_transform(points, transform)
    z = cam[:, 2]
    safe_z = np.maximum(z, _MIN_DEPTH)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    uv = np.stack([fx * cam[:, 0] / safe_z + cx, fy * cam[:, 1] / safe_z + cy], axis=-1)
    in_frustum = (
        (z > _MIN_DEPTH)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < image_w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < image_h)
    )

    n_pos = num_queries // 2
    pos_sel = _resample(np.where(in_frustum)[0], n_pos)
    neg_sel = _resample(np.where(~in_frustum)[0], num_queries - n_pos)
    sel = np.concatenate([pos_sel, neg_sel])

    queries = normalize_coord_corr_points(points[sel], points).astype(np.float32)
    labels = in_frustum[sel].astype(np.float32)
    distances = _signed_distance(uv[sel], z[sel], image_h, image_w).astype(np.float32)
    return queries, labels, distances
