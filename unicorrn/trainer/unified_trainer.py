import torch
import numpy as np

from ..utils import endpointerror
from ..utils.registry import Registry

TRAINER_REGISTRY = Registry('TRAINERS')


@TRAINER_REGISTRY.register(name="unified_trainer")
class UnifiedTrainer:
    def __init__(self, img2img_wt=1.0, img2pcd_wt=1.0, pcd2pcd_wt=1.0, frustum_wt=0.0):
        self.img2img_wt = img2img_wt
        self.img2pcd_wt = img2pcd_wt
        self.pcd2pcd_wt = pcd2pcd_wt
        self.frustum_wt = frustum_wt

    def _run_step_img2img(self, model, batch, matching_loss_fn, feature_loss_fn, **kwargs):
        img1 = batch['img1']
        img2 = batch['img2']
        queries = batch['queries']
        targets = batch['norm_targets']

        pred = model(task="img2img", src_img=img1, tgt_img=img2, query_pos=queries)
        loss, loss_details = matching_loss_fn(pred, targets)
        if feature_loss_fn is not None:
            feature_loss = feature_loss_fn(pred)
            loss = loss + feature_loss
            loss_details["feature_loss"] = feature_loss.item()

        loss_details["total_loss"] = loss.item()
        return loss, loss_details

    def _run_step_img2pcd(
        self, model, batch, matching_loss_fn, feature_loss_fn, frustum_loss_fn=None, **kwargs
    ):
        img = batch["image"]

        queries_2d = batch['queries']
        queries_3d = batch['norm_targets']

        targets_2d = batch['norm_queries']
        targets_3d = batch['norm_targets']

        frustum_queries = batch.get('frustum_queries')
        pred = model(
            task="img2pcd", src_img=img, sample=batch, query_pos_2d=queries_2d,
            query_pos_3d=queries_3d, frustum_query_pos_3d=frustum_queries,
        )

        loss_img2pcd, loss_details_img2pcd = matching_loss_fn(pred['img2pcd'], targets_3d)
        loss_pcd2img, loss_details_pcd2img = matching_loss_fn(pred['pcd2img'], targets_2d)

        loss = loss_img2pcd + loss_pcd2img
        loss_details = {}

        if frustum_loss_fn is not None and frustum_queries is not None:
            frustum_loss, frustum_details = frustum_loss_fn(
                pred['frustum_intermediates'],
                batch['frustum_labels'],
                batch['frustum_distances'],
            )
            loss = loss + self.frustum_wt * frustum_loss
            loss_details.update(frustum_details)

        for key, val in loss_details_img2pcd.items():
            loss_details[f"img2pcd_{key}"] = val

        for key, val in loss_details_pcd2img.items():
            loss_details[f"pcd2img_{key}"] = val

        if feature_loss_fn is not None:
            feature_loss = feature_loss_fn(pred)
            loss = loss + feature_loss
            loss_details["img2pcd_feature_loss"] = feature_loss.item()

        loss_details["img2pcd_total_loss"] = loss.item()
        return loss, loss_details

    def _run_step_pcd2pcd(self, model, batch, matching_loss_fn, feature_loss_fn, **kwargs):
        queries = batch['queries']
        targets = batch['norm_targets']

        pred = model(task="pcd2pcd", sample=batch, query_pos=queries)

        loss, loss_details = matching_loss_fn(pred, targets)
        if feature_loss_fn is not None:
            feature_loss = feature_loss_fn(pred)
            loss = loss + feature_loss
            loss_details["feature_loss"] = feature_loss.item()

        loss_details['total_loss'] = loss.item()
        return loss, loss_details

    def run_step(
        self, model, batch, matching_loss_fn, feature_loss_fn=None, frustum_loss_fn=None, **kwargs
    ):
        loss = 0
        loss_details = {}

        if self.img2img_wt > 0.0 and "img2img" in batch.keys():
            loss_img2img, loss_details_img2img = self._run_step_img2img(
                model,
                batch["img2img"],
                matching_loss_fn,
                feature_loss_fn,
                **kwargs
            )

            loss += self.img2img_wt * loss_img2img
            for k, v in loss_details_img2img.items():
                loss_details['img2img_' + k] = v

        if self.img2pcd_wt > 0.0 and "img2pcd" in batch.keys():
            loss_img2pcd, loss_details_img2pcd = self._run_step_img2pcd(
                model,
                batch["img2pcd"],
                matching_loss_fn,
                feature_loss_fn,
                frustum_loss_fn=frustum_loss_fn,
                **kwargs
            )

            loss += self.img2pcd_wt * loss_img2pcd
            for k, v in loss_details_img2pcd.items():
                loss_details[k] = v

        if self.pcd2pcd_wt > 0.0 and "pcd2pcd" in batch.keys():
            loss_pcd2pcd, loss_details_pcd2pcd = self._run_step_pcd2pcd(
                model,
                batch["pcd2pcd"],
                matching_loss_fn,
                feature_loss_fn,
                **kwargs
            )

            loss += self.pcd2pcd_wt * loss_pcd2pcd
            for k, v in loss_details_pcd2pcd.items():
                loss_details['pcd2pcd_' + k] = v

        loss_details['total_loss'] = loss.item()
        return loss, loss_details

    @torch.no_grad()
    def evaluate(self, model, batch):
        assert "img2img" in batch.keys() or "img2pcd" in batch.keys() or "pcd2pcd" in batch.keys()
        metrics = {}

        if "img2img" in batch.keys():
            img1 = batch["img2img"]['img1']
            img2 = batch["img2img"]['img2']
            queries = batch["img2img"]['queries']
            targets = batch["img2img"]['norm_targets']

            preds = model(task="img2img", src_img=img1, tgt_img=img2, query_pos=queries)["corr_predictions"]
            metrics["img2img"] = endpointerror(preds, targets).data.item()

        if "img2pcd" in batch.keys():
            img = batch['img2pcd']["image"]
            queries = batch['img2pcd']['queries']
            targets = batch['img2pcd']['norm_targets']

            preds = model(task="img2pcd", src_img=img, sample=batch["img2pcd"], query_pos_2d=queries)["img2pcd"][
                "corr_predictions"]
            metrics["img2pcd"] = endpointerror(preds, targets).data.item()

        if "pcd2pcd" in batch.keys():
            queries = batch["pcd2pcd"]['norm_queries']
            targets = batch["pcd2pcd"]['norm_targets']

            preds = model(task="pcd2pcd", sample=batch["pcd2pcd"], query_pos=queries)["corr_predictions"]
            metrics["pcd2pcd"] = endpointerror(preds, targets).data.item()

        total = []
        for key, val in metrics.items():
            if not np.isnan(val):
                total.append(val)

        metrics["combined_average"] = np.mean(total)
        return metrics
