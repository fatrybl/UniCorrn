import torch
import torch.nn.functional as F

from torch.nn import L1Loss, MSELoss
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR, OneCycleLR
from typing import List, Dict

from ..utils.registry import Registry

loss_functions = Registry("loss_functions")
optimizers = Registry("optimizers")
schedulers = Registry("schedulers")

loss_functions.register(MSELoss, "MSELoss")
loss_functions.register(L1Loss, "L1Loss")

optimizers.register(Adam, "Adam")
optimizers.register(AdamW, "AdamW")

schedulers.register(MultiStepLR, "MultiStepLR")
schedulers.register(OneCycleLR, "OneCycleLR")


@loss_functions.register()
class ConfidenceMatchingLoss:
    """ 
    Adapted from DUSt3R : https://github.com/naver/dust3r
    
    Weighted L1/L2 regression by learned confidence.
    

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10) 

        alpha: hyperparameter
    """

    def __init__(self, reg_loss='l1', alpha=0.2, vmin=1, vmax=float('inf'), mode='exp', robust=False):
        assert reg_loss == "l1" or reg_loss == "l2" or reg_loss == "smooth_l1"
        self.alpha = alpha
        self.vmin = vmin
        self.vmax = vmax
        self.mode = mode
        self.reg_loss = reg_loss
        self.robust = robust

    def __call__(self, output, target, batch=None, **kwargs):
        assert isinstance(output, Dict) and "corr_predictions" in output.keys() and "info_predictions" in output.keys()
        assert len(output["corr_predictions"]) == len(output["info_predictions"])

        corr_predictions, info_predictions = output["corr_predictions"], output["info_predictions"]

        assert corr_predictions.shape[2] == 2 or corr_predictions.shape[
            2] == 3, f"expected channels 2 or 3 but received {corr_predictions.shape[2]}"
        assert info_predictions.shape[2] == 1, f"expected channels 1 but received {info_predictions.shape[2]}"

        if self.reg_loss == "l1":
            loss = F.l1_loss(corr_predictions, target, reduction="none").sum(dim=2, keepdim=True)
        elif self.reg_loss == "l2":
            loss = torch.norm(target - corr_predictions, dim=2, keepdim=True)
        elif self.reg_loss == "smooth_l1":
            loss = F.smooth_l1_loss(corr_predictions, target, reduction="none").sum(dim=2, keepdim=True)

        if self.robust:
            epsilon = 0.01
            q = 0.4
            loss = torch.pow(loss + epsilon, q)

        if self.mode == 'exp':
            conf = self.vmin + info_predictions.exp().clip(max=self.vmax - self.vmin)
        elif self.mode == 'sigmoid':
            conf = (self.vmax - self.vmin) * torch.sigmoid(info_predictions) + self.vmin
        else:
            raise ValueError(f"Unsupported {self.mode}. ")

        log_conf = torch.log(conf)
        conf_loss = loss * conf - self.alpha * log_conf

        # average + nan protection (in case of no valid pixels at all)
        total_loss = conf_loss.mean() if conf_loss.numel() > 0 else 0

        loss_details = {
            self.reg_loss + '_loss': loss.mean().item() if loss.numel() > 0 else 0,
            'confidence_loss': log_conf.mean().item() if log_conf.numel() > 0 else 0
        }

        return total_loss, loss_details


def get_similarities(desc1, desc2, euc=False):
    if euc:  # negative squared l2 distance in same range than similarities
        # dists = (desc1[:, :, None] - desc2[:, None]).norm(dim=-1)
        # sim = 1 / (1 + dists)
        dists = torch.sum((desc1[:, :, None] - desc2[:, None]) ** 2, dim=-1)
        sim = -dists
    else:
        # Compute similarities
        sim = desc1 @ desc2.transpose(-2, -1)
    return sim


@loss_functions.register()
class InfoNCE:
    def __init__(self, temperature=0.07, eps=1e-8, mode='proper', **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.eps = eps
        assert mode in ['all', 'proper', 'dual']
        self.mode = mode

    def __call__(self, desc1, desc2, valid_matches=None, euc=False, neg_desc2=None):
        # valid positives are along diagonals
        B, N, D = desc1.shape
        B2, N2, D2 = desc2.shape
        assert B == B2 and D == D2
        if valid_matches is None:
            valid_matches = torch.ones([B, N], dtype=bool)
        # torch.all(valid_matches.sum(dim=-1) > 0) some pairs have no matches????
        assert valid_matches.shape == torch.Size([B, N]) and valid_matches.sum() > 0

        # Tempered similarities
        sim = get_similarities(desc1, desc2, euc) / self.temperature
        sim[sim.isnan()] = -torch.inf  # ignore nans
        # Softmax of positives with temperature
        sim = sim.exp_()  # save peak memory
        positives = sim.diagonal(dim1=-2, dim2=-1)

        # Loss
        if self.mode == 'all':  # Previous InfoNCE
            loss = -torch.log((positives / sim.sum(dim=-1).sum(dim=-1, keepdim=True)).clip(self.eps))
        elif self.mode == 'proper':  # Proper InfoNCE
            loss = -(torch.log((positives / sim.sum(dim=-2)).clip(self.eps)) +
                     torch.log((positives / sim.sum(dim=-1)).clip(self.eps)))
        elif self.mode == 'dual':  # Dual Softmax
            loss = -(torch.log((positives ** 2 / sim.sum(dim=-1) / sim.sum(dim=-2)).clip(self.eps)))
        else:
            raise ValueError("This should not happen...")
        return loss[valid_matches].mean()
