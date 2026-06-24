"""Loss functions for unified 2D-3D matching tasks.

Registers per-task loss implementations into the shared ``loss_functions``
registry used by the UniCorrn trainer. Includes the deep-supervised
geometric frustum focal-BCE loss added by the frustum-head fork.
"""

import torch
import torch.nn.functional as F

from .functions import loss_functions, ConfidenceMatchingLoss, InfoNCE


@loss_functions.register()
class FrustumClassificationLoss:
    """Deep-supervised focal BCE for geometric frustum membership.

    Applies focal binary cross-entropy to every decoder-layer frustum readout,
    weighting deeper layers more via an exponential decay (gamma), mirroring the
    auxiliary matching supervision.

        focal = (1 - p_t) ** focal_gamma,   loss = focal * BCE(logit, label)
    """

    def __init__(self, gamma=0.8, focal_gamma=2.0, pos_weight=1.0):
        """Initialise decay, focal-loss exponent and positive-class BCE weight."""
        self.gamma = gamma
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight

    def _focal_bce(self, logits, labels):
        """Focal binary cross-entropy for a single prediction layer."""
        pos_weight = torch.tensor(self.pos_weight, device=logits.device)
        ce = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none", pos_weight=pos_weight
        )
        p = torch.sigmoid(logits)
        focal = (1 - torch.where(labels > 0.5, p, 1 - p)) ** self.focal_gamma
        return (focal * ce).mean()

    def __call__(self, frustum_intermediates, labels, **kwargs):
        """Compute γ-decayed focal BCE over all decoder-layer frustum readouts."""
        labels = labels.float()
        num_layers = len(frustum_intermediates)
        loss = 0.0
        for idx in range(num_layers):
            decay = self.gamma ** (num_layers - idx - 1)
            loss = loss + decay * self._focal_bce(
                frustum_intermediates[idx].squeeze(-1), labels
            )
        return loss, {"frustum_loss": loss.item() if torch.is_tensor(loss) else loss}


class _GlobalMatchingLoss:
    def __init__(self, reg_loss='l1'):
        self.reg_loss = reg_loss

    def __call__(self, output, target, **kwargs):
        assert output.shape[2] == 2 or output.shape[2] == 3, f"expected channels 2 or 3 but received {output.shape[2]}"

        if self.reg_loss == "l1":
            loss = F.l1_loss(output, target, reduction="none").sum(dim=2, keepdim=True)
        elif self.reg_loss == "l2":
            loss = torch.norm(target - output, dim=2, keepdim=True)
        elif self.reg_loss == "smooth_l1":
            loss = F.smooth_l1_loss(output, target, reduction="none").sum(dim=2, keepdim=True)
        else:
            raise NotImplementedError

        total_loss = loss.mean() if loss.numel() > 0 else 0

        return total_loss


@loss_functions.register()
class AuxiliaryGlobalMatchingLoss:
    def __init__(self, reg_loss='l1', gamma=0.8, alpha=0.2, vmin=1, vmax=float('inf'), conf_mode='exp'):
        assert reg_loss == "l1" or reg_loss == "l2" or reg_loss == "smooth_l1"
        self.matching_loss = _GlobalMatchingLoss(reg_loss)
        self.conf_matching_loss_fn = ConfidenceMatchingLoss(
            reg_loss=reg_loss,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            mode=conf_mode
        )
        self.gamma = gamma

    def get_loss(self, gm_intermediates, predictions, target, **kwargs):
        num_layers = len(gm_intermediates)
        aux_loss = 0.0

        for layer_idx in range(num_layers):
            gamma = self.gamma ** (num_layers - layer_idx - 1)
            aux_loss += gamma * self.matching_loss(gm_intermediates[layer_idx], target)

        conf_loss, loss_details = self.conf_matching_loss_fn(predictions, target)
        loss_details['gm_aux_loss'] = aux_loss.item()

        return aux_loss + conf_loss, loss_details

    def __call__(self, output, target, **kwargs):
        predictions = {'corr_predictions': output['corr_predictions'], 'info_predictions': output['info_predictions']}
        return self.get_loss(output['gm_intermediates'], predictions, target)


@loss_functions.register()
class UnifiedInfoNCELoss:
    def __init__(
            self,
            info_nce_wt=1.0,
            temperature=0.05,
            eps=1e-8,
            mode='proper',
            use_euclidean_dist=False,
            enable_query2tgt=False,
            enable_query2src=False
    ):
        self.info_nce_loss_fn = InfoNCE(
            temperature=temperature,
            eps=eps,
            mode=mode
        )

        self.info_nce_wt = info_nce_wt
        self.use_euclidean_dist = use_euclidean_dist
        self.infonce_enable_query2tgt = enable_query2tgt
        self.infonce_enable_query2src = enable_query2src

    def __call__(self, output):
        desc_src = output["desc_src"]
        desc_tgt = output["desc_tgt"]
        qfeat_src = output["qfeat_src"]
        qfeat_tgt = output["qfeat_tgt"]

        desc_src = desc_src / desc_src.norm(dim=-1, keepdim=True)
        desc_tgt = desc_tgt / desc_tgt.norm(dim=-1, keepdim=True)
        qfeat_src = qfeat_src / qfeat_src.norm(dim=-1, keepdim=True)
        qfeat_tgt = qfeat_tgt / qfeat_tgt.norm(dim=-1, keepdim=True)

        valid = torch.ones(*desc_src.shape[:-1]).bool().to(desc_src.device)
        info_nce_loss = self.info_nce_loss_fn(desc_src, desc_tgt, valid_matches=valid, euc=self.use_euclidean_dist)
        if self.infonce_enable_query2tgt:
            info_nce_loss += self.info_nce_loss_fn(qfeat_src, desc_tgt, valid_matches=valid,
                                                   euc=self.use_euclidean_dist)
            info_nce_loss += self.info_nce_loss_fn(qfeat_tgt, desc_src, valid_matches=valid,
                                                   euc=self.use_euclidean_dist)

        if self.infonce_enable_query2src:
            info_nce_loss += self.info_nce_loss_fn(qfeat_src, desc_src, valid_matches=valid,
                                                   euc=self.use_euclidean_dist)
            info_nce_loss += self.info_nce_loss_fn(qfeat_tgt, desc_tgt, valid_matches=valid,
                                                   euc=self.use_euclidean_dist)

        return self.info_nce_wt * info_nce_loss


@loss_functions.register()
class GMAuxiliaryMatchingAndInfoNCELoss:
    def __init__(
            self,
            reg_loss='l1',
            gamma=0.8,
            alpha=0.2,
            vmin=1,
            vmax=float('inf'),
            conf_mode='exp',
            infonce_temperature=0.05,
            infonce_eps=1e-8,
            infonce_mode='proper',
            info_nce_wt=1.0,
            infonce_use_euclidean_dist=False,
            infonce_enable_query2tgt=False,
            infonce_enable_query2src=False,
    ):
        self.aux_matching_loss_fn = AuxiliaryGlobalMatchingLoss(
            reg_loss=reg_loss,
            gamma=gamma,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            conf_mode=conf_mode
        )

        self.info_nce_loss_fn = UnifiedInfoNCELoss(
            info_nce_wt=info_nce_wt,
            temperature=infonce_temperature,
            eps=infonce_eps,
            mode=infonce_mode,
            use_euclidean_dist=infonce_use_euclidean_dist,
            enable_query2tgt=infonce_enable_query2tgt,
            enable_query2src=infonce_enable_query2src
        )

    def __call__(self, output, target, **kwargs):
        predictions = {'corr_predictions': output['corr_predictions'], 'info_predictions': output['info_predictions']}
        loss, loss_details = self.aux_matching_loss_fn.get_loss(output['gm_intermediates'], predictions, target)
        info_nce_loss = self.info_nce_loss_fn(output)

        if not torch.isnan(info_nce_loss):
            loss += info_nce_loss
            loss_details["info_nce_loss"] = info_nce_loss.item()
        else:
            loss_details["info_nce_loss"] = info_nce_loss.item()

        return loss, loss_details
