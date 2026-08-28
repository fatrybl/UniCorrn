"""Geometric frustum labels and signed-distance targets for 3D point queries.

Given a point cloud and the ground-truth camera (intrinsics K and cloud->image
extrinsic T), a point P is labelled in-frustum iff it projects in front of the
camera and inside the image extent:

    P_cam = R @ P + t,        z = P_cam_z
    u = fx * x / z + cx,      v = fy * y / z + cy
    in_frustum = (z > z_min) and (0 <= u < W) and (0 <= v < H)

Membership is also emitted as a signed distance, so supervision is graded across
the boundary instead of being a step function of position. The frustum is the
intersection of five half-spaces of the camera frame - the side planes through the
centre, ``a_i . P_cam >= 0`` for ``a = (fx, 0, cx)``, ``(-fx, 0, W - cx)``,
``(0, fy, cy)``, ``(0, -fy, H - cy)``, and the near plane ``z - z_min >= 0`` - and the
target is the angular margin to the nearest face,

    s(P) = min_i ( a_i . P_cam + b_i ) / (|a_i| * ||P_cam||),

the sine of the angle to a side plane and the cosine of the polar angle at the near
plane: positive inside and negative outside with the sign exact everywhere, including
behind the camera, continuous, bounded and free of any scene scale.

T and K are used here for *labelling only*; the model never receives them as input, so
the learned classifier stays pose-independent.
"""

import numpy as np
from numpy import ndarray

from ...utils.vision3d.array_ops import apply_transform, normalize_coord_corr_points

_MIN_DEPTH = 1e-3


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


def _angular_sdf(cam: ndarray, intrinsics: ndarray, image_h: int, image_w: int) -> ndarray:
    """Angular signed distance of camera-frame points to the frustum (module formula).

    Args:
        cam: Points in the camera frame.
        intrinsics: Camera intrinsic matrix in the model's image frame.
        image_h: Image height in pixels.
        image_w: Image width in pixels.
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    normals = np.array(
        [[fx, 0.0, cx], [-fx, 0.0, image_w - cx], [0.0, fy, cy], [0.0, -fy, image_h - cy], [0.0, 0.0, 1.0]],
        dtype=cam.dtype,
    )
    offsets = np.array([0.0, 0.0, 0.0, 0.0, -_MIN_DEPTH], dtype=cam.dtype)
    margins = (cam @ normals.T + offsets) / np.linalg.norm(normals, axis=-1)
    radial = np.maximum(np.linalg.norm(cam, axis=-1), _MIN_DEPTH)
    return margins.min(axis=-1) / radial


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
    float {0, 1} membership labels, and the angular signed distance of each query to the
    frustum boundary.

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
    distances = _angular_sdf(cam[sel], intrinsics, image_h, image_w).astype(np.float32)
    return queries, labels, distances
