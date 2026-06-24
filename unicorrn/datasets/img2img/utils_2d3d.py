import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ...utils.vision3d.array_ops import (
    apply_transform,
    back_project,
    mutual_select,
    render,
)


def random_se3(max_rotation=5, max_translation=1.0):
    # Generate a random rotation with a limited angle
    axis = np.random.randn(3)  # Random axis
    axis /= np.linalg.norm(axis)  # Normalize to unit vector
    angle = np.random.uniform(-max_rotation, max_rotation)  # Random angle within limits
    R = Rotation.from_rotvec(np.deg2rad(angle) * axis).as_matrix()

    # Generate a random translation within the given range
    t = np.random.uniform(-max_translation, max_translation, size=(3, 1))

    # Construct the SE(3) matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()

    return T


def inverse_se3(T):
    R = T[:3, :3]  # Rotation part
    t = T[:3, 3]  # Translation part

    # Compute inverse
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv


def grid_sample_3d(points, grid_size=0.05, kdtree_workers=32):
    # Compute the bounding box
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)

    # Generate a 3D grid
    x_vals = np.arange(min_bounds[0], max_bounds[0], grid_size)
    y_vals = np.arange(min_bounds[1], max_bounds[1], grid_size)
    z_vals = np.arange(min_bounds[2], max_bounds[2], grid_size)
    grid_x, grid_y, grid_z = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    # Find nearest neighbors from the original points
    tree = cKDTree(points)
    _, indices = tree.query(grid_points, k=1, p=1, workers=kdtree_workers)

    unique_indices = np.unique(indices)
    sampled_points = points[unique_indices]

    return sampled_points


def voxel_downsample(points, voxel_size=0.05):
    # Downsample a point cloud using voxel grid filtering
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    # get unique voxel indices and corresponding first occurrence
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)

    return points[unique_indices]


def get_2d3d_correspondences_mutual_select(
    img_points,
    pcd_points,
    img_valid_mask,
    img_intrinsic,
    img_extrinsic,
    pcd2img_transform=None,
    matching_radius_2d=8.0,
    matching_radius_3d=0.0375,
    return_indices=False,
):
    if pcd2img_transform is not None:
        pcd_points = apply_transform(pcd_points, pcd2img_transform)

    # extract mutual correspondences
    img_corr_indices, pcd_corr_indices = mutual_select(
        img_points, pcd_points, mutual=True
    )

    v_indices, u_indices = np.nonzero(img_valid_mask)
    img_pixels = np.stack([v_indices, u_indices], axis=1)  # (H, W) or (y, x)

    img_corr_points = img_points[img_corr_indices]
    pcd_corr_points = pcd_points[pcd_corr_indices]

    # keep points within 3D matching radius
    masks_3d = (
        np.linalg.norm(img_corr_points[..., :3] - pcd_corr_points, axis=1)
        < matching_radius_3d
    )

    # keep points within 2D matching radius
    img_corr_pixels = img_pixels[img_corr_indices]
    world2cam = inverse_se3(img_extrinsic)
    pcd_corr_pixels = render(pcd_corr_points, img_intrinsic, world2cam)

    masks_2d = (
        np.linalg.norm(img_corr_pixels[..., :3] - pcd_corr_pixels, axis=1)
        < matching_radius_2d
    )
    masks = masks_2d & masks_3d

    img_corr_indices = img_corr_indices[masks]
    pcd_corr_indices = pcd_corr_indices[masks]

    img_corr_pixels = img_pixels[img_corr_indices]
    pcd_corr_points = pcd_points[pcd_corr_indices]

    if return_indices:
        return img_corr_pixels, pcd_corr_points, img_corr_indices, pcd_corr_indices
    else:
        return img_corr_pixels, pcd_corr_points
