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
    """Deep-supervised focal BCE plus signed-distance regression for frustum membership.

    Every decoder-layer readout emits an in/out logit and the signed distance of the
    query's projection to the frustum boundary; deeper layers weigh more via an
    exponential decay (gamma), mirroring the auxiliary matching supervision.

        focal = (1 - p_t) ** focal_gamma
        loss  = focal * BCE(logit, label) + SmoothL1(distance, target)

    The terms carry equal weight because they no longer compete: the head detaches the
    distance on its logit branch, so the BCE reaches only the temperature and the
    regression only the trunk. Their relative scale would merely retune the temperature's
    step size, so the only meaningful knob is one weight on the whole frustum objective,
    and that belongs to the caller.

    Since the label is exactly ``sign(target)`` and the predicted class exactly
    ``sign(distance)``, the regression - not the BCE - is what sets accuracy, and the BCE
    is left fitting the temperature alone.

    That temperature settles on the *pooled* distance scale, not the boundary one. At the
    optimum ``sum_i (sigmoid(s*d_i) - y_i) * d_i = 0``: correctly classified queries push
    ``s`` up with weight ``|d| * (1 - sigmoid(s|d|))``, which vanishes once ``s|d| >> 1``,
    while confidently wrong ones push it down at full focal weight. The balance is
    therefore set by the whole ``|d|`` distribution and the error rate. Measured over a
    4-epoch run, ``1/s`` tracked the pooled distance MAE at a near-constant 3.8-4.0x while
    its ratio to the boundary-band MAE drifted 4.4 -> 2.4, so the focal exponent tilts the
    fit toward the band without dominating it. ``1/s`` is consequently a usable running
    estimate of the head's own error scale, which ``consistency.LEARNED_TAU`` is defined
    from.

    ``huber_beta`` must be set to the scale of the boundary band, not left at a value
    sized for the target's full range. SmoothL1 has gradient ``min(|e| / beta, 1)``, so
    with beta well above the boundary scale the near-boundary errors that decide the
    class land in the quadratic regime and are down-weighted relative to large errors on
    saturated targets, whose precision changes no decision.
    """

    def __init__(self, gamma=0.8, focal_gamma=2.0, huber_beta=0.1):
        """Initialise decay, focal exponent and regression transition width."""
        self.gamma = gamma
        self.focal_gamma = focal_gamma
        self.huber_beta = huber_beta

    def _focal_bce(self, logits, labels):
        """Focal binary cross-entropy for a single prediction layer.

        No class weight: the query sampler emits an equal number of in- and
        out-of-frustum rows, so the two classes are balanced by construction.
        """
        ce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        p = torch.sigmoid(logits)
        focal = (1 - torch.where(labels > 0.5, p, 1 - p)) ** self.focal_gamma
        return (focal * ce).mean()

    def __call__(self, frustum_intermediates, labels, distances, **kwargs):
        """Compute γ-decayed focal BCE and signed-distance regression over all readouts.

        Args:
            frustum_intermediates: Per-layer readouts holding logit and signed distance.
            labels: Ground-truth in/out membership.
            distances: Ground-truth signed boundary distances.

        Raises:
            ValueError: If distances are missing. Since the head detaches the distance on
                its logit branch, the BCE reaches only the temperature, so dropping the
                regression would leave the head's trunk with no gradient at all.
        """
        if distances is None:
            raise ValueError("FrustumClassificationLoss requires signed-distance targets")
        labels = labels.float()
        num_layers = len(frustum_intermediates)
        loss, cls_loss, dist_loss = 0.0, 0.0, 0.0
        for idx in range(num_layers):
            decay = self.gamma ** (num_layers - idx - 1)
            layer = frustum_intermediates[idx]
            layer_cls = self._focal_bce(layer[..., 0], labels)
            cls_loss = cls_loss + layer_cls
            loss = loss + decay * layer_cls
            layer_dist = F.smooth_l1_loss(
                layer[..., 1], distances.float(), beta=self.huber_beta
            )
            dist_loss = dist_loss + layer_dist
            loss = loss + decay * layer_dist
        return loss, {
            "frustum_loss": loss.item() if torch.is_tensor(loss) else loss,
            "frustum_cls": cls_loss.item() if torch.is_tensor(cls_loss) else cls_loss,
            "frustum_dist": dist_loss.item() if torch.is_tensor(dist_loss) else dist_loss,
        }


class _GlobalMatchingLoss:
    """Per-layer coordinate regression, reduced over coordinates by a mean.

    The mean keeps 2D and 3D targets commensurate; see ``ConfidenceMatchingLoss``.
    """

    def __init__(self, reg_loss='l1'):
        self.reg_loss = reg_loss

    def __call__(self, output, target, **kwargs):
        assert output.shape[2] == 2 or output.shape[2] == 3, f"expected channels 2 or 3 but received {output.shape[2]}"

        if self.reg_loss == "l1":
            loss = F.l1_loss(output, target, reduction="none").mean(dim=2, keepdim=True)
        elif self.reg_loss == "l2":
            loss = torch.norm(target - output, dim=2, keepdim=True) / output.shape[2] ** 0.5
        elif self.reg_loss == "smooth_l1":
            loss = F.smooth_l1_loss(output, target, reduction="none").mean(dim=2, keepdim=True)
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
