# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Main encoder/decoder blocks
# --------------------------------------------------------
# References: 
# timm
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/helpers.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/mlp.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/patch_embed.py

import collections.abc

import torch
import torch.nn as nn

from itertools import repeat
from .build import MODULES_REGISTRY
from .functional import get_functional
from ...utils.config import configurable

from unicorrn.utils.vision3d.ops import index_select, knn

try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    print("xFormers is available.")
    XFORMERS_AVAILABLE = True
except ImportError:
    print("xFormers is not available.")
    XFORMERS_AVAILABLE = False

try:
    from flash_attn import flash_attn_func

    print("Flash attention is available.")
    FLASH_AVAILABLE = True


    def flash_attention(q, k, v):
        return flash_attn_func(
            q.to(torch.float16),
            k.to(torch.float16),
            v.to(torch.float16),
        ).to(torch.float32)

except ImportError:
    print("Flash attention is not available")
    FLASH_AVAILABLE = False


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


@MODULES_REGISTRY.register()
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    @configurable
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer="gelu", bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = get_functional(act_layer)
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_features": cfg.IN_FEATURES,
            "hidden_features": cfg.HIDDEN_FEATURES,
            "out_features": cfg.OUT_FEATURES,
            "act_layer": cfg.ACTIVATION,
            "bias": cfg.BIAS,
            "drop": cfg.DROP_OUT,
        }

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


@MODULES_REGISTRY.register(name="crocov2_attn_module")
class Attention(nn.Module):

    @configurable
    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    @classmethod
    def from_config(cls, cfg):
        return {
            "dim": cfg.DIM,
            "num_heads": cfg.NUM_HEADS,
            "qkv_bias": cfg.QKV_BIAS,
            "attn_drop": cfg.ATTN_DROP_OUT,
            "proj_drop": cfg.PROJ_DROP_OUT,
        }

    def forward(self, x, xpos):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        # q,k,v = qkv.unbind(2)  # make torchscript happy (cannot use tensor as tuple)

        if self.rope is not None:
            q = self.rope(q, xpos.long())
            k = self.rope(k, xpos.long())

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


@MODULES_REGISTRY.register(name="crocov2_memeff_attn_module")
class EfficientAttention(Attention):
    def forward(self, x, xpos):
        if not XFORMERS_AVAILABLE:
            return super().forward(x, xpos)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]  # B x num_heads x N x C // num_heads

        if self.rope is not None:
            q = self.rope(q, xpos.long())
            k = self.rope(k, xpos.long())

        # (batch_size, seqlen, nheads, headdim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        x = memory_efficient_attention(q, k, v)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


@MODULES_REGISTRY.register(name="crocov2_flash_attn_module")
class FlashAttention(Attention):
    def forward(self, x, xpos):
        if not FLASH_AVAILABLE:
            return super().forward(x, xpos)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]  # B x num_heads x N x C // num_heads

        if self.rope is not None:
            q = self.rope(q, xpos.long())
            k = self.rope(k, xpos.long())

        # (batch_size, seqlen, nheads, headdim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        x = flash_attention(q, k, v)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


@MODULES_REGISTRY.register(name="crocov2_encoder_block")
class Block(nn.Module):

    @configurable
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer="gelu", norm_layer=nn.LayerNorm, rope=None, use_flash_attn=False):
        super().__init__()
        self.norm1 = norm_layer(dim)

        if use_flash_attn:
            self.attn = FlashAttention(
                dim,
                rope=rope,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop
            )
        else:
            self.attn = EfficientAttention(
                dim,
                rope=rope,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop
            )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    @classmethod
    def from_config(cls, cfg):
        return {
            "dim": cfg.DIM,
            "num_heads": cfg.NUM_HEADS,
            "mlp_ratio": cfg.MLP_RATIO,
            "qkv_bias": cfg.QKV_BIAS,
            "act_layer": cfg.ACTIVATION,
            "attn_drop": cfg.ATTN_DROP_OUT,
            "drop": cfg.DROP_OUT,
            "drop_path": cfg.DROP_PATH,
            "use_flash_attn": cfg.USE_FLASH_ATTN
        }

    def forward(self, x, xpos):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GaussianKNNSample(nn.Module):
    def __init__(self, k=16):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(1.0))
        self.k = k

    def forward(self, src_feat, src_coord, sample_coord, return_indices=False):
        return knn_sample_3d(
            src_feat,
            src_coord,
            sample_coord,
            self.k,
            self.sigma,
            return_indices=return_indices,
        )


def knn_sample_3d(
        input_t, coord_t, sample_coord, k, sigma, return_indices=False, eps=1e-8
):
    """
    KNN sample a 3D point cloud, CANNOT handle batched data
    """
    # N_sample, k
    knn_distances, knn_indices = knn(
        sample_coord.float(), coord_t.float(), k, return_distance=True
    )
    sample_weights = torch.exp(-(knn_distances ** 2) / (2 * sigma ** 2))
    sample_weights_norm = sample_weights / (
            torch.sum(sample_weights, dim=-1, keepdim=True) + eps
    )

    sample_input = index_select(input_t, knn_indices, dim=0)
    if return_indices:
        return (
            torch.sum(sample_weights_norm[..., None] * sample_input, dim=-2),
            knn_indices,
        )
    return torch.sum(sample_weights_norm[..., None] * sample_input, dim=-2)
