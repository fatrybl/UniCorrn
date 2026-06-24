from typing import Tuple, Optional, Union

import numpy as np
from numpy import ndarray

from .ball_query import ball_query
from .depth_image import back_project, render
from .metrics import point_cloud_overlap, registration_corr_distance, registration_inlier_ratio
from .mutual_select import mutual_select
from .se3 import apply_transform, inverse_transform


# Ground Truth Utilities


def get_correspondences(src_points, tgt_points, transform, positive_radius):
    """Find the ground truth correspondences within the matching radius between two point clouds."""
    src_points = src_points[..., :3]
    tgt_points = tgt_points[..., :3]
    src_points = apply_transform(src_points, transform)
    indices_list = ball_query(tgt_points, src_points, positive_radius)
    corr_indices = np.array([(j, i) for i, indices in enumerate(indices_list) for j in indices], dtype=np.int32)
    src_corr_indices = corr_indices[:, 0]
    tgt_corr_indices = corr_indices[:, 1]
    return src_corr_indices, tgt_corr_indices


def get_2d3d_correspondences_mutual(
        depth_img: ndarray,
        pcd_points: ndarray,
        intrinsic: ndarray,
        transform: ndarray,
        matching_radius_2d: float,
        matching_radius_3d: float,
        depth_limit: float = 6.0,
) -> Tuple[ndarray, ndarray, ndarray]:
    """Find the ground-truth correspondences between an image and a point cloud.

    Method: Find the mutually nearest point pairs in 3D space, and select those satisfying distance restrictions.

    Returns:
        An array of the corresponding pixels in the shape of (C, 2), h first w last.
        An array of the corresponding point indices in the shape of (C).
        An array of the corresponding point in the shape of (C, 3).
    """
    img_points, img_pixels = back_project(depth_img, intrinsic, depth_limit=depth_limit, return_pixels=True)
    pcd_points_aligned = apply_transform(pcd_points, transform)
    img_corr_indices, pcd_corr_indices = mutual_select(img_points[..., :3], pcd_points_aligned, mutual=True)
    img_corr_points = img_points[img_corr_indices]
    pcd_corr_points = pcd_points_aligned[pcd_corr_indices]
    masks_3d = np.linalg.norm(img_corr_points[..., :3] - pcd_corr_points, axis=1) < matching_radius_3d
    img_corr_pixels = img_pixels[img_corr_indices]
    pcd_corr_pixels = render(pcd_corr_points, intrinsic)
    masks_2d = np.linalg.norm(img_corr_pixels[..., :3] - pcd_corr_pixels, axis=1) < matching_radius_2d
    masks = masks_2d & masks_3d
    img_corr_indices = img_corr_indices[masks]
    img_corr_pixels = img_pixels[img_corr_indices]
    pcd_corr_indices = pcd_corr_indices[masks]
    return img_corr_pixels, pcd_corr_indices, pcd_points[pcd_corr_indices]


def get_2d3d_correspondences_radius(
        depth_img: ndarray,
        pcd_points: ndarray,
        intrinsic: ndarray,
        transform: ndarray,
        matching_radius_2d: float,
        matching_radius_3d: float,
        depth_limit: float = 6.0,
) -> Tuple[ndarray, ndarray, ndarray]:
    """Find the ground-truth correspondences between an image and a point cloud.

    Method: Find the point pairs in 3D space with a 3D threshold, and select those satisfying 2D threshold.

    Returns:
        An array of the corresponding pixels in the shape of (C, 2), h first w last.
        An array of the corresponding point indices in the shape of (C).
        An array of the corresponding point in the shape of (C, 3).
    """
    img_points, img_pixels = back_project(depth_img, intrinsic, depth_limit=depth_limit, return_pixels=True)
    pcd_corr_indices, img_corr_indices = get_correspondences(pcd_points, img_points, transform, matching_radius_3d)
    pcd_corr_points = pcd_points[pcd_corr_indices]
    img_corr_pixels = img_pixels[img_corr_indices]
    pcd_corr_pixels = render(pcd_corr_points, intrinsic, extrinsics=transform)
    masks = np.linalg.norm(img_corr_pixels[..., :3] - pcd_corr_pixels, axis=1) < matching_radius_2d
    img_corr_indices = img_corr_indices[masks]
    img_corr_pixels = img_pixels[img_corr_indices]
    pcd_corr_indices = pcd_corr_indices[masks]
    return img_corr_pixels, pcd_corr_indices, pcd_corr_points[masks]


def get_3d3d_correspondences_mutual(
        src_pcd: ndarray,
        tgt_pcd: ndarray,
        tgt2src_transform: Optional[ndarray] = None,
        matching_radius_3d: Optional[float] = 0.0375,
        atol: Optional[float] = 0.5,
        return_indices: bool = False,
) -> Union[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray, ndarray, ndarray]]:
    tgt_pcd_aligned = apply_transform(tgt_pcd, tgt2src_transform) if tgt2src_transform is not None else tgt_pcd
    src_corr_indices, tgt_corr_indices = mutual_select(src_pcd, tgt_pcd_aligned, mutual=True)
    src_corr_points = src_pcd[src_corr_indices]
    tgt_corr_points = tgt_pcd_aligned[tgt_corr_indices]
    if matching_radius_3d is None:
        matching_radius_3d = np.min(np.linalg.norm(src_corr_points - tgt_corr_points, axis=-1)) + atol

    mask_3d = np.linalg.norm(src_corr_points - tgt_corr_points, axis=-1) < matching_radius_3d
    src_corr_indices = src_corr_indices[mask_3d]
    tgt_corr_indices = tgt_corr_indices[mask_3d]

    if return_indices:
        return src_pcd[src_corr_indices], tgt_pcd[tgt_corr_indices], src_corr_indices, tgt_corr_indices
    else:
        return src_pcd[src_corr_indices], tgt_pcd[tgt_corr_indices]


def get_3d3d_correspondences(
        src_pcd: ndarray,
        tgt_pcd: ndarray,
        tgt2src_transform: Optional[ndarray] = None,
        matching_radius_3d: Optional[float] = 0.0375,
        atol: Optional[float] = 0.5,
        return_indices: bool = False,
) -> Union[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray, ndarray, ndarray]]:
    src_pcd_aligned = apply_transform(src_pcd, inverse_transform(
        tgt2src_transform)) if tgt2src_transform is not None else src_pcd
    src_corr_indices, tgt_corr_indices = mutual_select(src_pcd_aligned, tgt_pcd, mutual=False)
    src_corr_points = src_pcd_aligned[src_corr_indices]
    tgt_corr_points = tgt_pcd[tgt_corr_indices]
    if matching_radius_3d is None:
        matching_radius_3d = np.min(np.linalg.norm(src_corr_points - tgt_corr_points, axis=-1)) + atol

    mask_3d = np.linalg.norm(src_corr_points - tgt_corr_points, axis=-1) < matching_radius_3d
    src_corr_indices = src_corr_indices[mask_3d]

    if return_indices:
        return src_pcd[src_corr_indices], src_pcd_aligned[src_corr_indices], src_corr_indices, src_corr_indices
    else:
        return src_pcd[src_corr_indices], src_pcd_aligned[src_corr_indices]


# Matching Utilities

def extract_correspondences_from_feats(
        src_points: ndarray,
        tgt_points: ndarray,
        src_feats: ndarray,
        tgt_feats: ndarray,
        mutual: bool = False,
        return_feat_dist: bool = False,
):
    """Extract correspondences from features."""
    src_corr_indices, tgt_corr_indices = mutual_select(src_feats, tgt_feats, mutual=mutual)

    src_corr_points = src_points[src_corr_indices]
    tgt_corr_points = tgt_points[tgt_corr_indices]
    outputs = [src_corr_points, tgt_corr_points]

    if return_feat_dist:
        src_corr_feats = src_feats[src_corr_indices]
        tgt_corr_feats = tgt_feats[tgt_corr_indices]
        feat_dists = np.linalg.norm(tgt_corr_feats - src_corr_feats, axis=1)
        outputs.append(feat_dists)

    return outputs


# Evaluation Utilities
def evaluate_correspondences(src_corr_points, tgt_corr_points, transform, positive_radius=0.1):
    overlap = point_cloud_overlap(
        src_corr_points,
        tgt_corr_points,
        transform,
        positive_radius=positive_radius,
    )

    inlier_ratio = registration_inlier_ratio(
        src_corr_points,
        tgt_corr_points,
        transform,
        positive_radius=positive_radius,
    )

    distance = registration_corr_distance(
        src_corr_points,
        tgt_corr_points,
        transform,
    )

    return {"overlap": overlap, "inlier_ratio": inlier_ratio, "distance": distance}


def evaluate_sparse_correspondences(
        src_length, tgt_length, src_corr_indices, tgt_corr_indices, gt_src_corr_indices, gt_tgt_corr_indices
):
    gt_corr_mat = np.zeros(shape=(src_length, tgt_length))
    gt_corr_mat[gt_src_corr_indices, gt_tgt_corr_indices] = 1.0
    num_gt_correspondences = gt_corr_mat.sum()

    pred_corr_mat = np.zeros_like(gt_corr_mat)
    pred_corr_mat[src_corr_indices, tgt_corr_indices] = 1.0
    num_pred_correspondences = pred_corr_mat.sum()

    pos_corr_mat = gt_corr_mat * pred_corr_mat
    num_pos_correspondences = pos_corr_mat.sum()

    precision = num_pos_correspondences / (num_pred_correspondences + 1e-12)
    recall = num_pos_correspondences / (num_gt_correspondences + 1e-12)

    pos_corr_mat = pos_corr_mat > 0
    gt_corr_mat = gt_corr_mat > 0
    src_hit_ratio = np.any(pos_corr_mat, axis=1).sum() / (np.any(gt_corr_mat, axis=1).sum() + 1e-12)
    tgt_hit_ratio = np.any(pos_corr_mat, axis=0).sum() / (np.any(gt_corr_mat, axis=0).sum() + 1e-12)
    hit_ratio = 0.5 * (src_hit_ratio + tgt_hit_ratio)

    return {"precision": precision, "recall": recall, "hit_ratio": hit_ratio}
