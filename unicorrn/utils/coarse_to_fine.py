# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# ----------------------------------------------------------------------
# coarse to fine utilities borrowed from https://github.com/naver/mast3r
# ----------------------------------------------------------------------
import numpy as np


def crop_tag(cell):
    return f'[{cell[1]}:{cell[3]},{cell[0]}:{cell[2]}]'


def crop_slice(cell):
    return slice(cell[1], cell[3]), slice(cell[0], cell[2])


def _start_pos(total_size, win_size, overlap):
    # we must have AT LEAST overlap between segments
    # first segment starts at 0, last segment starts at total_size-win_size
    assert 0 <= overlap < 1
    assert total_size >= win_size
    spacing = win_size * (1 - overlap)
    last_pt = total_size - win_size
    n_windows = 2 + int((last_pt - 1) // spacing)
    return np.linspace(0, last_pt, n_windows).round().astype(int)


def multiple_of_16(x):
    return int(x // 16) * 16


def _make_overlapping_grid(H, W, window_size, overlap):
    H_win, W_win = window_size
    x = _start_pos(W, W_win, overlap)
    y = _start_pos(H, H_win, overlap)
    grid = np.stack(np.meshgrid(x, y, indexing='xy'), axis=-1)
    grid = np.concatenate((grid, grid + (W_win, H_win)), axis=-1)
    return grid.reshape(-1, 4)


def _cell_size(cell2):
    width, height = cell2[:, 2] - cell2[:, 0], cell2[:, 3] - cell2[:, 1]
    assert width.min() >= 0
    assert height.min() >= 0
    return width, height


def _norm_windows(cell2, H2, W2, forced_resolution=None):
    # make sure the window aspect ratio is 3/4, or the output resolution is forced_resolution  if defined
    outcell = cell2.copy()
    width, height = _cell_size(cell2)
    width2, height2 = width.clip(max=W2), height.clip(max=H2)
    if forced_resolution is None:
        width2[width < height] = (height2[width < height] * 3.01 / 4).clip(max=W2)
        height2[width >= height] = (width2[width >= height] * 3.01 / 4).clip(max=H2)
    else:
        forced_H, forced_W = forced_resolution
        width2[:] = forced_W
        height2[:] = forced_H

    half = (width2 - width) / 2
    outcell[:, 0] -= half
    outcell[:, 2] += half
    half = (height2 - height) / 2
    outcell[:, 1] -= half
    outcell[:, 3] += half

    # proj to integers
    outcell = np.floor(outcell).astype(int)
    # Take care of flooring errors
    tmpw, tmph = _cell_size(outcell)
    outcell[:, 0] += tmpw.astype(tmpw.dtype) - width2.astype(tmpw.dtype)
    outcell[:, 1] += tmph.astype(tmpw.dtype) - height2.astype(tmpw.dtype)

    # make sure 0 <= x < W2 and 0 <= y < H2
    outcell[:, 0::2] -= outcell[:, [0]].clip(max=0)
    outcell[:, 1::2] -= outcell[:, [1]].clip(max=0)
    outcell[:, 0::2] -= outcell[:, [2]].clip(min=W2) - W2
    outcell[:, 1::2] -= outcell[:, [3]].clip(min=H2) - H2

    width, height = _cell_size(outcell)
    assert np.all(width == width2.astype(width.dtype)) and np.all(
        height == height2.astype(height.dtype)), "Error, output is not of the expected shape."
    assert np.all(width <= W2)
    assert np.all(height <= H2)
    return outcell


def _weight_pixels(cell, pix, assigned, gauss_var=2):
    center = cell.reshape(-1, 2, 2).mean(axis=1)
    width, height = _cell_size(cell)

    # square distance between each cell center and each point
    dist = (center[:, None] - pix[None]) / np.c_[width, height][:, None]
    dist2 = np.square(dist).sum(axis=-1)

    assert assigned.shape == dist2.shape
    res = np.where(assigned, np.exp(-gauss_var * dist2), 0)
    return res


def pos2d_in_rect(p1, cell1):
    x, y = p1.T
    l, t, r, b = cell1
    assigned = (l <= x) & (x < r) & (t <= y) & (y < b)
    return assigned


def _score_cell(cell1, H2, W2, p1, p2, min_corres=10, forced_resolution=None):
    assert p1.shape == p2.shape

    # compute keypoint assignment
    assigned = pos2d_in_rect(p1, cell1[None].T)
    assert assigned.shape == (len(cell1), len(p1))

    # remove cells without correspondences
    valid_cells = assigned.sum(axis=1) >= min_corres
    cell1 = cell1[valid_cells]
    assigned = assigned[valid_cells]
    if not valid_cells.any():
        return cell1, cell1, assigned

    # fill-in the assigned points in both image
    assigned_p1 = np.empty((len(cell1), len(p1), 2), dtype=np.float32)
    assigned_p2 = np.empty((len(cell1), len(p2), 2), dtype=np.float32)
    assigned_p1[:] = p1[None]
    assigned_p2[:] = p2[None]
    assigned_p1[~assigned] = np.nan
    assigned_p2[~assigned] = np.nan

    # find the median center and scale of assigned points in each cell
    # cell_center1 = np.nanmean(assigned_p1, axis=1)
    cell_center2 = np.nanmean(assigned_p2, axis=1)
    im1_q25, im1_q75 = np.nanquantile(assigned_p1, (0.1, 0.9), axis=1)
    im2_q25, im2_q75 = np.nanquantile(assigned_p2, (0.1, 0.9), axis=1)

    robust_std1 = (im1_q75 - im1_q25).clip(20.)
    robust_std2 = (im2_q75 - im2_q25).clip(20.)

    cell_size1 = (cell1[:, 2:4] - cell1[:, 0:2])
    cell_size2 = cell_size1 * robust_std2 / robust_std1
    cell2 = np.c_[cell_center2 - cell_size2 / 2, cell_center2 + cell_size2 / 2]

    # make sure cell bounds are valid
    cell2 = _norm_windows(cell2, H2, W2, forced_resolution=forced_resolution)

    # compute correspondence weights
    corres_weights = _weight_pixels(cell1, p1, assigned) * _weight_pixels(cell2, p2, assigned)

    # return a list of window pairs and assigned correspondences
    return cell1, cell2, corres_weights


def greedy_selection(corres_weights, target=0.9):
    # corres_weight = (n_cell_pair, n_corres) matrix.
    # If corres_weight[c,p]>0, means that correspondence p is visible in cell pair p
    assert 0 < target <= 1
    corres_weights = corres_weights.copy()

    total = corres_weights.max(axis=0).sum()
    target *= total

    # init = empty
    res = []
    cur = np.zeros(corres_weights.shape[1])  # current selection

    while cur.sum() < target:
        # pick the nex best cell pair
        best = corres_weights.sum(axis=1).argmax()
        res.append(best)

        if corres_weights[best].sum() == 0.0:
            break

        # update current
        cur += corres_weights[best]
        # print('     appending', best, 'with score', corres_weights[best].sum(), '-->', cur.sum())

        # remove from all other views
        corres_weights = (corres_weights - corres_weights[best]).clip(min=0)

    return res


def select_pairs_of_crops(img_q, img_b, pos2d_in_query, pos2d_in_ref, window_sizes, overlap=.5, coarse_coverage=0.9,
                          forced_resolution=None):
    # prepare the overlapping cells
    if isinstance(window_sizes[0], int):
        window_sizes = (window_sizes, window_sizes)

    grid_q = _make_overlapping_grid(*img_q.shape[:2], window_sizes[0], overlap)
    grid_b = _make_overlapping_grid(*img_b.shape[:2], window_sizes[1], overlap)

    assert forced_resolution is None or len(forced_resolution) == 2
    if isinstance(forced_resolution[0], int) or not len(forced_resolution[0]) == 2:
        forced_resolution1 = forced_resolution2 = forced_resolution
    else:
        assert len(forced_resolution[1]) == 2
        forced_resolution1 = forced_resolution[0]
        forced_resolution2 = forced_resolution[1]

    # Make sure crops respect constraints
    grid_q = _norm_windows(grid_q.astype(float), *img_q.shape[:2], forced_resolution=forced_resolution1)
    grid_b = _norm_windows(grid_b.astype(float), *img_b.shape[:2], forced_resolution=forced_resolution2)

    # score cells
    pairs_q = _score_cell(grid_q, *img_b.shape[:2], pos2d_in_query, pos2d_in_ref, forced_resolution=forced_resolution2)
    pairs_b = _score_cell(grid_b, *img_q.shape[:2], pos2d_in_ref, pos2d_in_query, forced_resolution=forced_resolution1)
    pairs_b = pairs_b[1], pairs_b[0], pairs_b[2]  # cellq, cellb, corres_weights

    # greedy selection until all correspondences are generated
    cell1, cell2, corres_weights = map(np.concatenate, zip(pairs_q, pairs_b))
    if len(corres_weights) == 0:
        return  # tolerated for empty generators
    order = greedy_selection(corres_weights, target=coarse_coverage)

    for i in order:
        def pair_tag(qi, bi): return (str(qi) + crop_tag(cell1[i]), str(bi) + crop_tag(cell2[i]))

        yield cell1[i], cell2[i], pair_tag


def bilinear_interpolate_features(H, W, grid_coords, grid_features, selected_mask=None):
    """
    Strict bilinear interpolation of features on an image-sized grid.

    Args:
        H, W: image height and width
        grid_coords: (Gh, Gw, 2) tensor of grid coordinates (x, y)
        grid_features: (Gh, Gw, F) tensor of features at grid points
        selected_mask: (Gh, Gw,) boolean mask of selected grid points

    Returns:
        dense_features: (H, W, F) interpolated features
        valid_mask: (H, W) bool mask (True = inside selected hull)
    """
    Gh, Gw, _ = grid_coords.shape

    # pixel meshgrid
    yy, xx = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij"
    )

    # grid lines
    x_grid = grid_coords[0, :, 0]  # (Gw,)
    y_grid = grid_coords[:, 0, 1]  # (Gh,)

    # which cell are we in?
    gx = np.searchsorted(x_grid, xx, side="left") - 1
    gy = np.searchsorted(y_grid, yy, side="left") - 1

    # validity mask: strictly inside the grid
    inside_x = (gx >= 0) & (gx < Gw - 1)
    inside_y = (gy >= 0) & (gy < Gh - 1)
    valid_mask = inside_x & inside_y

    gx = np.clip(gx, 0, Gw - 2)
    gy = np.clip(gy, 0, Gh - 2)

    # gather the 4 cell corners
    p00 = grid_coords[gy, gx]  # top-left
    p10 = grid_coords[gy, gx + 1]  # top-right
    p01 = grid_coords[gy + 1, gx]  # bottom-left

    f00 = grid_features[gy, gx]
    f10 = grid_features[gy, gx + 1]
    f01 = grid_features[gy + 1, gx]
    f11 = grid_features[gy + 1, gx + 1]

    # relative coords (bilinear weights)
    denom_x = np.clip(p10[..., 0] - p00[..., 0], 1e-8, None)
    denom_y = np.clip(p01[..., 1] - p00[..., 1], 1e-8, None)
    u = (xx - p00[..., 0]) / denom_x
    v = (yy - p00[..., 1]) / denom_y

    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)

    dense_features = (
            (1 - u)[..., None] * (1 - v)[..., None] * f00 +
            u[..., None] * (1 - v)[..., None] * f10 +
            (1 - u)[..., None] * v[..., None] * f01 +
            u[..., None] * v[..., None] * f11
    )

    # validity mask
    if selected_mask is not None:
        m00 = selected_mask[gy, gx]
        m10 = selected_mask[gy, gx + 1]
        m01 = selected_mask[gy + 1, gx]
        m11 = selected_mask[gy + 1, gx + 1]
        valid_mask = valid_mask & m00 & m10 & m01 & m11

    # extract valid coords and features
    coords = np.stack([xx[valid_mask], yy[valid_mask]], axis=-1).astype(int)  # (M, 2)
    feats = dense_features[valid_mask]  # (M, F)

    return coords, feats
