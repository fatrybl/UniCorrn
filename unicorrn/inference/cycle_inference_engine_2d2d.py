from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as tvf

from ..datasets.img2img.cropping import crop_to_homography
from ..utils.coarse_to_fine import crop_slice, select_pairs_of_crops
from .tictoc import tic, toc


def crop(img, mask, pts3d, crop, intrinsics=None):
    out_cropped_img = img.clone()
    if mask is not None:
        out_cropped_mask = mask.clone()
    else:
        out_cropped_mask = None
    if pts3d is not None:
        out_cropped_pts3d = pts3d.clone()
    else:
        out_cropped_pts3d = None
    to_orig = torch.eye(3, device=img.device)

    # If intrinsics available, crop and apply rectifying homography. Otherwise, just crop
    if intrinsics is not None:
        K_old = intrinsics
        imsize, K_new, R, H = crop_to_homography(K_old, crop)
        # apply homography to image
        H /= H[2, 2]
        homo8 = H.ravel().tolist()[:8]
        # From float tensor to uint8 PIL Image
        pilim = Image.fromarray((255 * (img + 1.0) / 2).to(torch.uint8).numpy())
        pilout_cropped_img = pilim.transform(
            imsize,
            Image.Transform.PERSPECTIVE,
            homo8,
            resample=Image.Resampling.BICUBIC,
        )

        # From uint8 PIL Image to float tensor
        out_cropped_img = (
            2.0 * torch.tensor(np.array(pilout_cropped_img)).to(img) / 255.0 - 1.0
        )
        if out_cropped_mask is not None:
            pilmask = Image.fromarray((255 * out_cropped_mask).to(torch.uint8).numpy())
            pilout_cropped_mask = pilmask.transform(
                imsize,
                Image.Transform.PERSPECTIVE,
                homo8,
                resample=Image.Resampling.NEAREST,
            )
            out_cropped_mask = torch.from_numpy(np.array(pilout_cropped_mask) > 0).to(
                out_cropped_mask.dtype
            )
        if out_cropped_pts3d is not None:
            out_cropped_pts3d = out_cropped_pts3d.numpy()
            out_cropped_X = np.array(
                Image.fromarray(out_cropped_pts3d[:, :, 0]).transform(
                    imsize,
                    Image.Transform.PERSPECTIVE,
                    homo8,
                    resample=Image.Resampling.NEAREST,
                )
            )
            out_cropped_Y = np.array(
                Image.fromarray(out_cropped_pts3d[:, :, 1]).transform(
                    imsize,
                    Image.Transform.PERSPECTIVE,
                    homo8,
                    resample=Image.Resampling.NEAREST,
                )
            )
            out_cropped_Z = np.array(
                Image.fromarray(out_cropped_pts3d[:, :, 2]).transform(
                    imsize,
                    Image.Transform.PERSPECTIVE,
                    homo8,
                    resample=Image.Resampling.NEAREST,
                )
            )

            out_cropped_pts3d = torch.from_numpy(
                np.stack([out_cropped_X, out_cropped_Y, out_cropped_Z], axis=-1)
            )

        to_orig = torch.tensor(H, device=img.device)
    else:
        out_cropped_img = img[crop_slice(crop)]
        if out_cropped_mask is not None:
            out_cropped_mask = out_cropped_mask[crop_slice(crop)]
        if out_cropped_pts3d is not None:
            out_cropped_pts3d = out_cropped_pts3d[crop_slice(crop)]
        to_orig[:2, -1] = torch.tensor(crop[:2])

    return out_cropped_img, out_cropped_mask, out_cropped_pts3d, to_orig


def get_HW_resolution(maxdim=512, patch_size=16, subpixel_scale=1):
    if maxdim == 512:
        return (384 // subpixel_scale, 512 // subpixel_scale)
    else:
        raise NotImplementedError


def interpolate_image(img, resolution):
    img_res = img.shape[2:]
    if img_res[0] == resolution[0] and img_res[1] == resolution[1]:
        return img

    img = F.interpolate(img, size=resolution, mode="bilinear", align_corners=False)
    return img


"""
    Coarse only matching with cyclic consistency.

"""


@torch.no_grad()
def cycle_matching(
    img1, img2, queries, model, matching_radius_px=5.0, unified_model=False
):
    H, W = img1.shape[2:]

    # forward cycle
    tic()
    output = model.forward_img_to_img_bidirectional(img1, img2, queries)
    elapsed_time = toc()

    tgt_pred_forward = output["corr_predictions_src2tgt"]
    src_pred_backward = output["corr_predictions_tgt2src"]

    conf_pred = output["info_predictions_src2tgt"]

    # de-normalize coordinates
    tgt_pred_forward[:, :, 0] = tgt_pred_forward[:, :, 0] * W
    tgt_pred_forward[:, :, 1] = tgt_pred_forward[:, :, 1] * H

    # de-normalize coordinates
    src_pred_backward[:, :, 0] = src_pred_backward[:, :, 0] * W
    src_pred_backward[:, :, 1] = src_pred_backward[:, :, 1] * H

    cycle_matched = torch.norm(src_pred_backward - queries, dim=-1) < matching_radius_px

    return (
        tgt_pred_forward.squeeze(0).detach().cpu().numpy(),
        conf_pred.squeeze().detach().cpu().numpy(),
        cycle_matched.squeeze(0).cpu().numpy(),
        elapsed_time,
    )


@torch.no_grad()
def cycle_coarse_matching(
    img1, img2, queries, model, matching_radius_px=5.0, unified_model=False
):

    # orignal image dimensions
    H1, W1 = img1.shape[2:]
    H2, W2 = img2.shape[2:]

    maxdim = max(model.patch_embed.img_size)

    H, W = get_HW_resolution(maxdim=maxdim, patch_size=model.patch_embed.patch_size)

    # resize images : downscale image pairs to nearest aspect ratio of the first image
    img1 = F.interpolate(img1, size=(H, W), mode="bilinear", align_corners=False)
    img2 = F.interpolate(img2, size=(H, W), mode="bilinear", align_corners=False)

    # downscale query points
    scale_x = W1 / W
    scale_y = H1 / H
    scaled_queries = queries.clone()
    scaled_queries[:, :, 0] = (queries[:, :, 0] / scale_x).round().int()
    scaled_queries[:, :, 1] = (queries[:, :, 1] / scale_y).round().int()

    # predict
    img1 = img1.cuda()
    img2 = img2.cuda()
    scaled_queries = scaled_queries.cuda()
    model = model.cuda()

    corr_pred, conf_pred, cycle_matched, elapsed_time = cycle_matching(
        img1, img2, scaled_queries, model, matching_radius_px, unified_model
    )

    scaled_queries = scaled_queries.squeeze(0).detach().cpu().numpy()

    return scaled_queries, corr_pred, conf_pred, cycle_matched, elapsed_time


@torch.no_grad()
def cycle_coarse_inference(
    img1,
    img2,
    queries,
    model,
    map_data,
    matching_radius_px=5,
    unified_model=False,
):
    maxdim = max(model.patch_embed.img_size)
    H1, W1, _ = img1.shape
    H2, W2, _ = img2.shape

    img1 = tvf.to_tensor(img1)
    img2 = tvf.to_tensor(img2)
    norm_img1 = (
        tvf.normalize(img1, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        .float()[None]
        .contiguous()
    )
    norm_img2 = (
        tvf.normalize(img2, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        .float()[None]
        .contiguous()
    )
    queries = torch.from_numpy(queries)[None].contiguous()

    model.eval()
    coarse_queries, coarse_preds, coarse_confidence, cycle_matched, _ = (
        cycle_coarse_matching(
            norm_img1,
            norm_img2,
            queries,
            model,
            unified_model=unified_model,
            matching_radius_px=matching_radius_px,
        )
    )

    pts3d = map_data["pts3d_rescaled"]
    valid = map_data["valid_rescaled"]

    # assert pts3d.shape == (384, 512, 3), f"Invalid point map. Expected (512, 384, 3) but got {pts3d.shape}."
    # assert valid.shape == (384, 512), f"Invalid valid map. Expected (512, 384) but got {valid.shape}."

    queries = queries.squeeze(0)[cycle_matched]
    coarse_queries = coarse_queries[cycle_matched]
    coarse_preds = coarse_preds[cycle_matched]
    coarse_confidence = coarse_confidence[cycle_matched]

    # filter out predictions for valid 3D points only.

    _h, _w, _ = pts3d.shape
    is_portrait = _w < _h

    if is_portrait:
        # rescale to account for aspect ratio change.
        scale_x = 384 / 512
        scale_y = 512 / 384

        coarse_queries[:, 0] = coarse_queries[:, 0] * scale_x
        coarse_queries[:, 1] = coarse_queries[:, 1] * scale_y

    valid_pts = valid[
        np.round(coarse_queries[:, 1]).astype(int),
        np.round(coarse_queries[:, 0]).astype(int),
    ]

    queries = queries[valid_pts]
    coarse_queries = coarse_queries[valid_pts]
    coarse_preds = coarse_preds[valid_pts]
    coarse_confidence = coarse_confidence[valid_pts]

    valid_pts3d = pts3d[
        np.round(coarse_queries[:, 1]).astype(int),
        np.round(coarse_queries[:, 0]).astype(int),
    ]

    # rescale back to original resolution
    H, W = get_HW_resolution(maxdim=maxdim, patch_size=model.patch_embed.patch_size)
    scale_x = W2 / W
    scale_y = H2 / H

    coarse_preds[..., 0] = coarse_preds[..., 0] * scale_x
    coarse_preds[..., 1] = coarse_preds[..., 1] * scale_y

    queries = queries.cpu().numpy()

    return queries, coarse_preds, coarse_confidence, valid_pts3d


"""
    Unform grid based matching with cyclic consistency.

"""


def init_query_points(img_h, img_w, grid_size=1):
    img_h = img_h // grid_size
    img_w = img_w // grid_size

    x = torch.arange(img_w, dtype=torch.int32) * grid_size
    y = torch.arange(img_h, dtype=torch.int32) * grid_size

    queries = torch.cartesian_prod(x, y).reshape(img_w, img_h, 2)

    return queries.permute(1, 0, 2).contiguous()


@torch.no_grad()
def cycle_uniform_grid_matching(
    img1, img2, model, grid_size=1, matching_radius_px=5.0, unified_model=False
):

    # orignal image dimensions
    H1, W1 = img1.shape[2:]
    H2, W2 = img2.shape[2:]

    maxdim = max(model.patch_embed.img_size)

    H, W = get_HW_resolution(maxdim=maxdim, patch_size=model.patch_embed.patch_size)

    # resize images : downscale image pairs to nearest aspect ratio of the first image
    img1 = F.interpolate(img1, size=(H, W), mode="bilinear", align_corners=False)
    img2 = F.interpolate(img2, size=(H, W), mode="bilinear", align_corners=False)

    # initialize
    queries = (
        init_query_points(H, W, grid_size=grid_size)
        .view(-1, 2)
        .contiguous()
        .round()
        .int()[None]
    )

    # predict
    img1 = img1.cuda()
    img2 = img2.cuda()
    queries = queries.cuda()
    model = model.cuda()

    corr_pred, conf_pred, cycle_matched, elapsed_time = cycle_matching(
        img1, img2, queries, model, matching_radius_px, unified_model
    )

    queries = queries.squeeze(0).detach().cpu().numpy()

    return queries, corr_pred, conf_pred, cycle_matched, elapsed_time


@torch.no_grad()
def cycle_uniform_grid_inference(
    img1,
    img2,
    model,
    map_data=None,
    grid_size=1,
    matching_radius_px=5,
    unified_model=False,
):
    maxdim = max(model.patch_embed.img_size)
    H1, W1, _ = img1.shape
    H2, W2, _ = img2.shape

    img1 = tvf.to_tensor(img1)
    img2 = tvf.to_tensor(img2)
    norm_img1 = (
        tvf.normalize(img1, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        .float()[None]
        .contiguous()
    )
    norm_img2 = (
        tvf.normalize(img2, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        .float()[None]
        .contiguous()
    )

    model.eval()
    coarse_queries, coarse_preds, coarse_confidence, cycle_matched, _ = (
        cycle_uniform_grid_matching(
            norm_img1,
            norm_img2,
            model,
            grid_size=grid_size,
            unified_model=unified_model,
            matching_radius_px=matching_radius_px,
        )
    )

    coarse_queries = coarse_queries[cycle_matched]
    coarse_preds = coarse_preds[cycle_matched]
    coarse_confidence = coarse_confidence[cycle_matched]

    # filter out predictions for valid 3D points only.
    valid_pts3d = None
    if map_data is not None:
        pts3d = map_data["pts3d_rescaled"]
        valid = map_data["valid_rescaled"]

        _h, _w, _ = pts3d.shape
        is_portrait = _w < _h

        if is_portrait:
            # rescale to account for aspect ratio change.
            scale_x = 384 / 512
            scale_y = 512 / 384

            coarse_queries[:, 0] = coarse_queries[:, 0] * scale_x
            coarse_queries[:, 1] = coarse_queries[:, 1] * scale_y

        valid_pts = valid[
            np.round(coarse_queries[:, 1]).astype(int),
            np.round(coarse_queries[:, 0]).astype(int),
        ]

        coarse_queries = coarse_queries[valid_pts]
        coarse_preds = coarse_preds[valid_pts]
        coarse_confidence = coarse_confidence[valid_pts]

        valid_pts3d = pts3d[
            np.round(coarse_queries[:, 1]).astype(int),
            np.round(coarse_queries[:, 0]).astype(int),
        ]

    # rescale back to original resolution
    H, W = get_HW_resolution(maxdim=maxdim, patch_size=model.patch_embed.patch_size)

    scale_x1 = W1 / W
    scale_y1 = H1 / H
    coarse_queries[..., 0] = coarse_queries[..., 0] * scale_x1
    coarse_queries[..., 1] = coarse_queries[..., 1] * scale_y1

    scale_x2 = W2 / W
    scale_y2 = H2 / H

    coarse_preds[..., 0] = coarse_preds[..., 0] * scale_x2
    coarse_preds[..., 1] = coarse_preds[..., 1] * scale_y2

    return coarse_queries, coarse_preds, coarse_confidence, valid_pts3d


"""
    Coarse only matching.

"""


@torch.no_grad()
def coarse_inference(
    img1, img2, queries, model, map_data, unified_model=False, **kwargs
):
    maxdim = max(model.patch_embed.img_size)
    H1, W1, _ = img1.shape
    H2, W2, _ = img2.shape

    img1 = tvf.to_tensor(img1)
    img2 = tvf.to_tensor(img2)
    norm_img1 = (
        tvf.normalize(img1, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        .float()[None]
        .contiguous()
    )
    norm_img2 = (
        tvf.normalize(img2, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        .float()[None]
        .contiguous()
    )
    queries = torch.from_numpy(queries)[None].contiguous()

    model.eval()
    coarse_queries, coarse_preds, coarse_confidence, _ = coarse_matching(
        norm_img1,
        norm_img2,
        queries,
        model,
        select_layer=-1,
        unified_model=unified_model,
        coarse_only=True,
    )

    pts3d = map_data["pts3d_rescaled"]
    valid = map_data["valid_rescaled"]

    # assert pts3d.shape == (384, 512, 3), f"Invalid point map. Expected (512, 384, 3) but got {pts3d.shape}."
    # assert valid.shape == (384, 512), f"Invalid valid map. Expected (512, 384) but got {valid.shape}."

    # filter out predictions for valid 3D points only.

    _h, _w, _ = pts3d.shape
    is_portrait = _w < _h

    if is_portrait:
        # rescale to account for aspect ratio change.
        scale_x = 384 / 512
        scale_y = 512 / 384

        coarse_queries[:, 0] = coarse_queries[:, 0] * scale_x
        coarse_queries[:, 1] = coarse_queries[:, 1] * scale_y

    valid_pts = valid[
        np.round(coarse_queries[:, 1]).astype(int),
        np.round(coarse_queries[:, 0]).astype(int),
    ]

    queries = queries.squeeze(0)[valid_pts]
    coarse_queries = coarse_queries[valid_pts]
    coarse_preds = coarse_preds[valid_pts]
    coarse_confidence = coarse_confidence[valid_pts]

    valid_pts3d = pts3d[
        np.round(coarse_queries[:, 1]).astype(int),
        np.round(coarse_queries[:, 0]).astype(int),
    ]

    # rescale back to original resolution
    H, W = get_HW_resolution(maxdim=maxdim, patch_size=model.patch_embed.patch_size)
    scale_x = W2 / W
    scale_y = H2 / H

    coarse_preds[..., 0] = coarse_preds[..., 0] * scale_x
    coarse_preds[..., 1] = coarse_preds[..., 1] * scale_y

    queries = queries.cpu().numpy()

    return queries, coarse_preds, coarse_confidence, valid_pts3d


@torch.no_grad()
def coarse_matching(
    img1,
    img2,
    queries,
    model,
    select_layer=-1,
    unified_model=False,
    raw_output=False,
    coarse_only=False,
):
    # orignal image dimensions
    H1, W1 = img1.shape[2:]
    H2, W2 = img2.shape[2:]

    maxdim = max(model.patch_embed.img_size)

    H, W = get_HW_resolution(maxdim=maxdim, patch_size=model.patch_embed.patch_size)

    # resize images : downscale image pairs to nearest aspect ratio of the first image
    img1 = interpolate_image(img1, (H, W))
    img2 = interpolate_image(img2, (H, W))

    # downscale query points
    scale_x = W1 / W
    scale_y = H1 / H
    scaled_queries = queries.clone()
    scaled_queries[:, :, 0] = queries[:, :, 0] / scale_x
    scaled_queries[:, :, 1] = queries[:, :, 1] / scale_y

    # predict
    img1 = img1.cuda()
    img2 = img2.cuda()
    scaled_queries = scaled_queries.cuda()
    model = model.cuda()

    # print('coarse: ', scaled_queries.shape, scaled_queries.stride(), scaled_queries.is_contiguous())
    # with record_function("model_inference"):
    tic()
    if not unified_model:
        output = model(img1, img2, scaled_queries)
    else:
        output = model.forward_img_to_img(img1, img2, scaled_queries)
    elapsed_time = toc()

    corr_pred = output["corr_predictions"]
    conf_pred = output["info_predictions"]

    if isinstance(corr_pred, List):
        corr_pred = corr_pred[select_layer]
        conf_pred = conf_pred[select_layer]

    # de-normalize coordinates
    corr_pred[:, :, 0] = corr_pred[:, :, 0] * W
    corr_pred[:, :, 1] = corr_pred[:, :, 1] * H

    if coarse_only:
        return (
            scaled_queries.squeeze(0).detach().cpu().numpy(),
            corr_pred.squeeze(0).detach().cpu().numpy(),
            conf_pred.squeeze(0).detach().cpu().numpy(),
            elapsed_time,
        )

    # upscale predicted points to original resolution
    scale_x = W2 / W
    scale_y = H2 / H

    corr_pred[:, :, 0] = corr_pred[:, :, 0] * scale_x
    corr_pred[:, :, 1] = corr_pred[:, :, 1] * scale_y

    corr_pred = corr_pred.squeeze(0)
    if not raw_output:
        corr_pred = corr_pred.round().int()
    conf_pred = conf_pred.detach().cpu()

    return corr_pred.detach().cpu().numpy(), conf_pred, elapsed_time


"""
    Coarse to fine matching.

"""


def get_overlapping_crops(
    img1,
    img2,
    queries,
    coarse_preds,
    pts3d,
    valid_all,
    map_K,
    query_K,
    maxdim=512,
    overlap=0.5,
    coarse_coverage=1.0,
    patch_size=16,
    subpixel_scale=1,
):
    crops1, crops2, overlapped_queries, pair_tags = [], [], [], []
    crops_v1, crops_p1 = [], []

    # orignal image dimensions
    H1, W1, _ = img1.shape
    H2, W2, _ = img2.shape

    # crop size
    H, W = get_HW_resolution(
        maxdim=maxdim, patch_size=patch_size, subpixel_scale=subpixel_scale
    )
    forced_resolution = [(H, W), (H, W)]
    idx = 0
    for crop_1, crop_2, pair_tag in select_pairs_of_crops(
        img1,
        img2,
        queries,
        coarse_preds,
        window_sizes=forced_resolution,
        overlap=overlap,
        coarse_coverage=coarse_coverage,
        forced_resolution=forced_resolution,
    ):
        # c1 = crop(img1, crop_1)
        # c2 = crop(img2, crop_2)
        # crops1.append(c1)
        # crops2.append(c2)

        c1, v1, p1, trf1 = crop(
            img1, valid_all, pts3d, crop_1, map_K
        )  # map image has 3D points
        c2, _, _, trf2 = crop(img2, None, None, crop_2, query_K)
        crops1.append(c1)
        crops2.append(c2)
        crops_v1.append(v1)
        crops_p1.append(p1)

        x1, x2, y1, y2 = crop_1[0], crop_1[2], crop_1[1], crop_1[3]

        # print('crops1', crop_1, 'crops2', crop_2, 'x1', x1, 'x2', x2, 'y1', y1, 'y2', y2)

        condition = (
            (queries[:, 0] >= x1)
            & (queries[:, 0] < x2)
            & (queries[:, 1] >= y1)
            & (queries[:, 1] < y2)
        )
        selected_query = queries[condition] - (x1, y1)
        overlapped_queries.append(selected_query)

        pair_tags.append({"idx": idx, "img1": crop_1[:2], "img2": crop_2[:2]})
        idx += 1

    # print(len(crops1), len(crops2), len(overlapped_queries), len(pair_tags))

    max_rows = max(q.shape[0] for q in overlapped_queries)
    # Pad queries to have the same shape
    padded_queries = [
        np.pad(query, ((0, max_rows - query.shape[0]), (0, 0)), mode="constant")
        for query in overlapped_queries
    ]

    crops1 = torch.stack(crops1).permute(0, 3, 1, 2)
    crops2 = torch.stack(crops2).permute(0, 3, 1, 2)

    crops1 = tvf.normalize(crops1, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    crops2 = tvf.normalize(crops2, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    crops_p1 = torch.stack(crops_p1)
    crops_v1 = torch.stack(crops_v1)

    padded_queries = torch.from_numpy(np.stack(padded_queries))

    return crops1, crops2, crops_p1, crops_v1, padded_queries, pair_tags


@torch.no_grad()
def crops_inference(
    img1,
    img2,
    queries,
    model,
    device="cuda",
    batch_size=32,
    select_layer=-1,
    subpixel_scale=1,
    unified_model=False,
):
    B, C, H, W = img1.shape

    if subpixel_scale != 1:
        img1 = F.interpolate(
            img1, scale_factor=subpixel_scale, mode="bilinear", align_corners=False
        )
        img2 = F.interpolate(
            img2, scale_factor=subpixel_scale, mode="bilinear", align_corners=False
        )
    img1 = img1.to(device)
    img2 = img2.to(device)
    queries = queries.to(device) * subpixel_scale
    model = model.to(device)
    # print('fine: ', queries.shape, queries.stride(), queries.is_contiguous())

    if B < batch_size:
        # with record_function("model_inference"):
        if not unified_model:
            output = model(img1, img2, queries)
        else:
            output = model.forward_img_to_img(img1, img2, queries)

        corr_pred = output["corr_predictions"]
        conf_pred = output["info_predictions"]

        if isinstance(corr_pred, List):
            corr_pred = corr_pred[select_layer]
            conf_pred = conf_pred[select_layer]

        corr_pred = corr_pred.detach().cpu().numpy()
        conf_pred = conf_pred.detach().cpu().numpy()

        # de-normalize coordinates
        corr_pred[:, :, 0] = corr_pred[:, :, 0] * W
        corr_pred[:, :, 1] = corr_pred[:, :, 1] * H

        return corr_pred, conf_pred

    corr_preds, conf_preds = [], []
    for i in range(0, B, batch_size):
        selected = slice(i, i + min(B - i, batch_size))
        _img1, _img2, _queries = (
            img1[selected].contiguous(),
            img2[selected].contiguous(),
            queries[selected].contiguous(),
        )

        # with record_function("model_inference"):
        if not unified_model:
            output = model(_img1, _img2, _queries)
        else:
            output = model.forward_img_to_img(_img1, _img2, _queries)

        corr_pred = output["corr_predictions"]
        conf_pred = output["info_predictions"]

        if isinstance(corr_pred, List):
            corr_pred = corr_pred[select_layer]
            conf_pred = conf_pred[select_layer]

        corr_preds.append(corr_pred.detach().cpu().numpy())
        conf_preds.append(conf_pred.detach().cpu().numpy())

    # Merge all preds
    corr_preds = np.vstack(corr_preds)
    conf_preds = np.vstack(conf_preds)

    # de-normalize coordinates
    corr_preds[:, :, 0] = corr_preds[:, :, 0] * W
    corr_preds[:, :, 1] = corr_preds[:, :, 1] * H

    return corr_preds, conf_preds


@torch.no_grad()
def fine_matching(
    img1,
    img2,
    crop_pts3d,
    crop_valid,
    overlapped_queries,
    queries,
    pair_tags,
    model,
    device="cuda",
    batch_size=32,
    sort=True,
    select_layer=-1,
    subpixel_scale=1,
    unified_model=False,
):
    corr_preds, conf_preds = crops_inference(
        img1,
        img2,
        overlapped_queries,
        model,
        device=device,
        batch_size=batch_size,
        select_layer=select_layer,
        subpixel_scale=subpixel_scale,
        unified_model=unified_model,
    )

    # overlapped_queries = rearrange(overlapped_queries.numpy(), 'n1 n2 c -> (n1 n2) c')
    overlapped_queries = overlapped_queries.numpy()
    valid_pts3d, corrs, confs, overlapped_q = [], [], [], []

    for tag, query, corr, conf in zip(
        pair_tags, overlapped_queries, corr_preds, conf_preds
    ):
        # valid pts3d
        idx = tag["idx"]
        pts3d_i = crop_pts3d[idx].cpu().numpy()
        valid_i = crop_valid[idx].cpu().numpy()

        # filter out predictions for valid 3D points only.
        valid_pts = valid_i[
            np.round(query[:, 1]).astype(int), np.round(query[:, 0]).astype(int)
        ]

        query = query[valid_pts]
        corr = corr[valid_pts]
        conf = conf[valid_pts]

        valid_pts3d_i = pts3d_i[
            np.round(query[:, 1]).astype(int), np.round(query[:, 0]).astype(int)
        ]
        valid_pts3d.append(valid_pts3d_i)

        # to original image resolution
        x1, y1 = tag["img1"]
        query = query + (x1, y1)

        x2, y2 = tag["img2"]
        corr = corr + (x2, y2)

        overlapped_q.append(query)
        corrs.append(corr)
        confs.append(conf)

    overlapped_q = np.vstack(overlapped_q)
    corrs = np.vstack(corrs)
    confs = np.vstack(confs)
    valid_pts3d = np.vstack(valid_pts3d)

    # print(overlapped_q.shape, corrs.shape, confs.shape)
    # print(queries.min(), queries.max(), overlapped_q.min(), overlapped_q.max())

    # Consolidate predicted correspondences
    queries = queries[np.newaxis, :, :]  # (1, Nq, 2)
    temp_q = overlapped_q[:, np.newaxis, :]  # (Nv, 1, 2)
    matches = np.equal(temp_q, queries)  # (Nv, Nq, 2)
    matches_all = np.all(matches, axis=2)  # (Nv, Nq)

    matching_indices = np.any(matches_all, axis=1).nonzero()[0].tolist()

    overlapped_q = overlapped_q[matching_indices]
    corrs = corrs[matching_indices]
    confs = confs[matching_indices]
    valid_pts3d = valid_pts3d[matching_indices]

    unique_rows = {}
    for index, row in enumerate(overlapped_q):
        row = tuple(row)
        confidence = confs[index][0]

        # If row not in dictionary or current confidence is higher, update
        if row not in unique_rows or confidence > unique_rows[row][1]:
            unique_rows[row] = (index, confidence)

    # Extract indices with the highest confidence for each unique row
    confidence_indices = [index for index, _ in unique_rows.values()]
    # print(len(confidence_indices), overlapped_q.shape, corrs.shape)
    overlapped_q = overlapped_q[confidence_indices]
    corrs = corrs[confidence_indices]
    confs = confs[confidence_indices]
    valid_pts3d = valid_pts3d[confidence_indices]

    if sort:
        indices = np.argsort(overlapped_q[:, 1])
        overlapped_q = overlapped_q[indices]
        corrs = corrs[indices]
        confs = confs[indices]
        valid_pts3d = valid_pts3d[indices]

    return overlapped_q, corrs, confs, valid_pts3d


@torch.no_grad()
def coarse_to_fine(
    img1,
    img2,
    queries,
    model,
    map_data,
    conf_threshold=1.0001,
    conf_topk=3000,
    coarse_coverage=0.9,
    overlap=0.5,
    batch_size=32,
    select_layer=-1,
    unified_model=False,
    coarse_only=False,
):
    maxdim = max(model.patch_embed.img_size)
    H1, W1, _ = img1.shape
    H2, W2, _ = img2.shape

    img1 = tvf.to_tensor(img1)
    img2 = tvf.to_tensor(img2)
    norm_img1 = (
        tvf.normalize(img1, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        .float()[None]
        .contiguous()
    )
    norm_img2 = (
        tvf.normalize(img2, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        .float()[None]
        .contiguous()
    )
    queries = torch.from_numpy(queries)[None].contiguous()

    model.eval()
    coarse_preds, coarse_confidence, _ = coarse_matching(
        norm_img1,
        norm_img2,
        queries,
        model,
        select_layer=select_layer,
        unified_model=unified_model,
        coarse_only=False,
    )

    raw_confidence = coarse_confidence.squeeze()

    # Skip fine estimation if max confidence is below threshold
    if raw_confidence.max() < conf_threshold:
        pts3d = map_data["pts3d"]
        valid = map_data["valid_all"]

        queries = queries.detach().squeeze().cpu().numpy()
        raw_confidence = raw_confidence.cpu().numpy()

        # filter out predictions for valid 3D points only.
        valid_pts = valid[
            np.round(queries[:, 1]).astype(int), np.round(queries[:, 0]).astype(int)
        ]
        valid_pts = valid[
            np.round(queries[:, 1]).astype(int), np.round(queries[:, 0]).astype(int)
        ]

        queries = queries[valid_pts]
        coarse_preds = coarse_preds[valid_pts]
        raw_confidence = raw_confidence[valid_pts]

        valid_pts3d = pts3d[
            np.round(queries[:, 1]).astype(int), np.round(queries[:, 0]).astype(int)
        ]

        return queries, coarse_preds, raw_confidence, valid_pts3d

    selected_conf = (raw_confidence > conf_threshold).to(bool)
    if selected_conf.sum() > conf_topk:
        selected_conf = torch.topk(raw_confidence, conf_topk, dim=0)[1].tolist()
    elif selected_conf.sum() < conf_topk // 2:
        selected_conf = torch.topk(raw_confidence, conf_topk // 2, dim=0)[1].tolist()

    selected_queries = queries.squeeze(0).detach().cpu()[selected_conf]
    coarse_preds = coarse_preds[selected_conf]

    img1 = img1.permute(1, 2, 0)
    img2 = img2.permute(1, 2, 0)
    queries = selected_queries.numpy()

    pts3d = torch.from_numpy(map_data["pts3d"])
    valid = torch.from_numpy(map_data["valid_all"])
    map_intrinsic = map_data["map_K"]
    query_intrinsic = map_data["query_K"]

    crops1, crops2, crop_pts3d, crop_valid, overlapped_queries, pair_tags = (
        get_overlapping_crops(
            img1,
            img2,
            queries,
            coarse_preds,
            pts3d,
            valid,
            map_intrinsic,
            query_intrinsic,
            maxdim=maxdim,
            coarse_coverage=coarse_coverage,
            overlap=overlap,
            patch_size=model.patch_embed.patch_size,
        )
    )

    matched_queries, fine_preds, fine_confidence, valid_pts3d = fine_matching(
        crops1,
        crops2,
        crop_pts3d,
        crop_valid,
        overlapped_queries,
        queries,
        pair_tags,
        model,
        batch_size=batch_size,
        select_layer=select_layer,
        unified_model=unified_model,
    )

    return matched_queries, fine_preds, fine_confidence, valid_pts3d
