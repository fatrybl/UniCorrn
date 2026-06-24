"""
    Coarse to fine borrowed from MASt3R.

"""

import time
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.profiler import record_function
from torchvision.transforms import functional as tvf

from ..utils import cartesian_img_coord
from ..utils.coarse_to_fine import crop_slice, select_pairs_of_crops
from .tictoc import tic, toc


def crop(img, crop):
    out_cropped_img = img.clone()
    out_cropped_img = img[crop_slice(crop)]
    return out_cropped_img


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


@torch.no_grad()
def coarse_matching(
    img1, img2, queries, model, select_layer=-1, unified_model=False, raw_output=False
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


def get_overlapping_crops(
    img1,
    img2,
    queries,
    coarse_preds,
    maxdim=512,
    overlap=0.5,
    coarse_coverage=1.0,
    patch_size=16,
    subpixel_scale=1,
):
    crops1, crops2, overlapped_queries, pair_tags = [], [], [], []

    # orignal image dimensions
    H1, W1, _ = img1.shape
    H2, W2, _ = img2.shape

    # crop size
    H, W = get_HW_resolution(
        maxdim=maxdim, patch_size=patch_size, subpixel_scale=subpixel_scale
    )
    forced_resolution = [(H, W), (H, W)]

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
        c1 = crop(img1, crop_1)
        c2 = crop(img2, crop_2)
        crops1.append(c1)
        crops2.append(c2)

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

        pair_tags.append({"img1": crop_1[:2], "img2": crop_2[:2]})

    # print(len(crops1), len(crops2), len(overlapped_queries), len(pair_tags))

    max_rows = max(q.shape[0] for q in overlapped_queries)
    # Pad queries to have the same shape
    padded_queries = [
        np.pad(query, ((0, max_rows - query.shape[0]), (0, 0)), mode="constant")
        for query in overlapped_queries
    ]

    crops1 = torch.stack(crops1).permute(0, 3, 1, 2)
    crops2 = torch.stack(crops2).permute(0, 3, 1, 2)
    padded_queries = torch.from_numpy(np.stack(padded_queries))

    return crops1, crops2, padded_queries, pair_tags


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
    corrs, confs, overlapped_q = [], [], []

    for tag, query, corr, conf in zip(
        pair_tags, overlapped_queries, corr_preds, conf_preds
    ):
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

    if sort:
        indices = np.argsort(overlapped_q[:, 1])
        overlapped_q = overlapped_q[indices]
        corrs = corrs[indices]
        confs = confs[indices]

    return overlapped_q, corrs, confs


@torch.no_grad()
def coarse_only(img1, img2, queries, model, select_layer=-1, unified_model=False):
    H1, W1, _ = img1.shape
    H2, W2, _ = img2.shape

    img1 = (
        tvf.normalize(tvf.to_tensor(img1), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        .float()[None]
        .contiguous()
    )
    img2 = (
        tvf.normalize(tvf.to_tensor(img2), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        .float()[None]
        .contiguous()
    )
    queries = torch.from_numpy(queries)[None].contiguous()

    model.eval()
    coarse_preds, coarse_confidence, elapsed_time = coarse_matching(
        img1,
        img2,
        queries,
        model,
        select_layer=select_layer,
        unified_model=unified_model,
        raw_output=True,
    )

    return queries.squeeze(0).numpy(), coarse_preds, coarse_confidence, elapsed_time


@torch.no_grad()
def coarse_to_fine(
    img1,
    img2,
    queries,
    model,
    coarse_coverage=0.9,
    overlap=0.5,
    batch_size=32,
    select_layer=-1,
    unified_model=False,
):
    maxdim = max(model.patch_embed.img_size)

    H1, W1, _ = img1.shape
    H2, W2, _ = img2.shape

    img1 = (
        tvf.normalize(tvf.to_tensor(img1), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        .float()[None]
        .contiguous()
    )
    img2 = (
        tvf.normalize(tvf.to_tensor(img2), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        .float()[None]
        .contiguous()
    )
    queries = torch.from_numpy(queries)[None].contiguous()

    model.eval()
    coarse_preds, coarse_confidence, elapsed_time = coarse_matching(
        img1,
        img2,
        queries,
        model,
        select_layer=select_layer,
        unified_model=unified_model,
    )

    # backward compatibility
    # if maxdim == 560 and (min(img1.shape[2:]) < maxdim or min(img2.shape[2:]) < maxdim):
    #     print(f"Image is smaller than maxdim {maxdim}, img1 {img1.shape}, img2 {img2.shape}. Skipping coarse to fine")
    #     queries = queries.squeeze(0).numpy()
    #     queries, coarse_preds = to_portrait(queries, coarse_preds, is_img1_portrait, is_img2_portrait)
    #     return queries, coarse_preds, coarse_confidence
    #
    # if max(img1.shape[2:]) < maxdim or max(img2.shape[2:]) < maxdim:
    #     print(f"Image is smaller than maxdim {maxdim}, img1 {img1.shape}, img2 {img2.shape}. Skipping coarse to fine")
    #     queries = queries.squeeze(0).numpy()
    #     queries, coarse_preds = to_portrait(queries, coarse_preds, is_img1_portrait, is_img2_portrait)
    #     return queries, coarse_preds, coarse_confidence

    img1 = img1.squeeze(0).permute(1, 2, 0)
    img2 = img2.squeeze(0).permute(1, 2, 0)
    queries = queries.squeeze(0).numpy()

    crops1, crops2, overlapped_queries, pair_tags = get_overlapping_crops(
        img1,
        img2,
        queries,
        coarse_preds,
        maxdim=maxdim,
        coarse_coverage=coarse_coverage,
        overlap=overlap,
        patch_size=model.patch_embed.patch_size,
    )

    matched_queries, fine_preds, fine_confidence = fine_matching(
        crops1,
        crops2,
        overlapped_queries,
        queries,
        pair_tags,
        model,
        batch_size=batch_size,
        select_layer=select_layer,
        unified_model=unified_model,
    )

    return matched_queries, fine_preds, fine_confidence, elapsed_time


@torch.no_grad()
def conf_coarse_to_fine(
    img1,
    img2,
    queries,
    model,
    conf_threshold=1.0001,
    conf_topk=3000,
    coarse_coverage=0.9,
    overlap=0.5,
    batch_size=32,
    select_layer=-1,
    unified_model=False,
):
    maxdim = max(model.patch_embed.img_size)
    H1, W1, _ = img1.shape
    H2, W2, _ = img2.shape

    img1 = (
        tvf.normalize(tvf.to_tensor(img1), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        .float()[None]
        .contiguous()
    )
    img2 = (
        tvf.normalize(tvf.to_tensor(img2), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        .float()[None]
        .contiguous()
    )
    queries = torch.from_numpy(queries)[None].contiguous()

    model.eval()
    coarse_preds, coarse_confidence, _ = coarse_matching(
        img1,
        img2,
        queries,
        model,
        select_layer=select_layer,
        unified_model=unified_model,
    )

    raw_confidence = coarse_confidence.squeeze()
    selected_conf = (raw_confidence > conf_threshold).to(bool)
    if selected_conf.sum() > conf_topk:
        selected_conf = torch.topk(raw_confidence, conf_topk, dim=0)[1].tolist()
    elif selected_conf.sum() < conf_topk // 2:
        selected_conf = torch.topk(raw_confidence, conf_topk // 2, dim=0)[1].tolist()

    selected_queries = queries.squeeze(0).detach().cpu()[selected_conf]
    coarse_preds = coarse_preds[selected_conf]

    img1 = img1.squeeze(0).permute(1, 2, 0)
    img2 = img2.squeeze(0).permute(1, 2, 0)
    queries = selected_queries.numpy()

    crops1, crops2, overlapped_queries, pair_tags = get_overlapping_crops(
        img1,
        img2,
        queries,
        coarse_preds,
        maxdim=maxdim,
        coarse_coverage=coarse_coverage,
        overlap=overlap,
        patch_size=model.patch_embed.patch_size,
    )

    matched_queries, fine_preds, fine_confidence = fine_matching(
        crops1,
        crops2,
        overlapped_queries,
        queries,
        pair_tags,
        model,
        batch_size=batch_size,
        select_layer=select_layer,
        unified_model=unified_model,
    )

    return matched_queries, fine_preds, fine_confidence


@torch.no_grad()
def coarse_keypoint_matching(
    img1, img2, model, grid_size=4, select_layer=-1, unified_model=False
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
    queries = (
        cartesian_img_coord(H // grid_size, W // grid_size, patch_size=grid_size)
        .view(-1, 2)[None]
        .contiguous()
    )

    # predict
    img1 = img1.cuda()
    img2 = img2.cuda()
    queries = queries.cuda()
    model = model.cuda()

    if not unified_model:
        output = model(img1, img2, queries)
    else:
        output = model.forward_img_to_img(img1, img2, queries)

    conf_pred = output["info_predictions"]

    if isinstance(conf_pred, List):
        conf_pred = conf_pred[select_layer]

    queries[:, :, 0] = queries[:, :, 0] * scale_x
    queries[:, :, 1] = queries[:, :, 1] * scale_y

    raw_queries = queries.squeeze(0).detach().cpu()
    conf_pred = conf_pred.detach().cpu()

    return raw_queries, conf_pred


@torch.no_grad()
def keypoint_selection_coarse_to_fine(
    img1,
    img2,
    model,
    grid_size=4,
    conf_threshold=1.0001,
    conf_topk=5000,
    coarse_coverage=0.9,
    overlap=0.5,
    batch_size=32,
    select_layer=-1,
    unified_model=False,
):
    maxdim = max(model.patch_embed.img_size)

    H1, W1, _ = img1.shape
    H2, W2, _ = img2.shape

    img1 = (
        tvf.normalize(tvf.to_tensor(img1), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        .float()[None]
        .contiguous()
    )
    img2 = (
        tvf.normalize(tvf.to_tensor(img2), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        .float()[None]
        .contiguous()
    )

    model.eval()
    raw_queries, raw_confidence = coarse_keypoint_matching(
        img1,
        img2,
        model,
        grid_size,
        select_layer=select_layer,
        unified_model=unified_model,
    )

    raw_confidence = raw_confidence.squeeze()
    selected_conf = (raw_confidence > conf_threshold).to(bool)
    if selected_conf.sum() > conf_topk:
        selected_conf = torch.topk(raw_confidence, conf_topk, dim=0)[1].tolist()
    elif selected_conf.sum() < conf_topk // 2:
        selected_conf = torch.topk(raw_confidence, conf_topk // 2, dim=0)[1].tolist()

    selected_conf = torch.multinomial(
        F.softmax(raw_confidence / 0.2, dim=-1),
        num_samples=conf_topk,
        replacement=False,
    )
    selected_queries = raw_queries[selected_conf][None]

    coarse_preds, coarse_confidence, _ = coarse_matching(
        img1,
        img2,
        selected_queries,
        model,
        select_layer=select_layer,
        unified_model=unified_model,
    )

    img1 = img1.squeeze(0).permute(1, 2, 0)
    img2 = img2.squeeze(0).permute(1, 2, 0)
    queries = selected_queries.squeeze(0).numpy()

    crops1, crops2, overlapped_queries, pair_tags = get_overlapping_crops(
        img1,
        img2,
        queries,
        coarse_preds,
        maxdim=maxdim,
        coarse_coverage=coarse_coverage,
        overlap=overlap,
        patch_size=model.patch_embed.patch_size,
    )

    matched_queries, fine_preds, fine_confidence = fine_matching(
        crops1,
        crops2,
        overlapped_queries,
        queries,
        pair_tags,
        model,
        batch_size=batch_size,
        select_layer=select_layer,
        unified_model=unified_model,
    )

    return matched_queries, fine_preds, fine_confidence
