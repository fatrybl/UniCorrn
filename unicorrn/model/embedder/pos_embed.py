# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, n_cls_token=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [n_cls_token+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if n_cls_token>0:
        pos_embed = np.concatenate([np.zeros([n_cls_token, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


#----------------------------------------------------------
# RoPE2D: RoPE implementation in 2D
#----------------------------------------------------------
try:
    from unicorrn.model.embedder.curope import cuRoPE2D
    RoPE2D = cuRoPE2D
    print('cuda-compiled RoPE2d successfully loaded!!')
except ImportError:
    print('Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead')

    class RoPE2D(torch.nn.Module):
        
        def __init__(self, freq=100.0, F0=1.0):
            super().__init__()
            self.base = freq 
            self.F0 = F0
            self.cache = {}

        def get_cos_sin(self, D, seq_len, device, dtype):
            if (D,seq_len,device,dtype) not in self.cache:
                inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
                t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
                freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
                freqs = torch.cat((freqs, freqs), dim=-1)
                cos = freqs.cos() # (Seq, Dim)
                sin = freqs.sin()
                self.cache[D,seq_len,device,dtype] = (cos,sin)
            return self.cache[D,seq_len,device,dtype]
            
        @staticmethod
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
            
        def apply_rope1d(self, tokens, pos1d, cos, sin):
            assert pos1d.ndim==2
            cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
            sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
            return (tokens * cos) + (self.rotate_half(tokens) * sin)
            
        def forward(self, tokens, positions):
            """
            input:
                * tokens: batch_size x nheads x ntokens x dim
                * positions: batch_size x ntokens x 2 (y and x position of each token)
            output:
                * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
            """
            assert tokens.size(3)%2==0, "number of dimensions should be a multiple of two"
            D = tokens.size(3) // 2
            assert positions.ndim==3 and positions.shape[-1] == 2 # Batch, Seq, 2
            cos, sin = self.get_cos_sin(D, int(positions.max())+1, tokens.device, tokens.dtype)
            # split features into two along the feature dimension, and apply rope1d on each half
            y, x = tokens.chunk(2, dim=-1)
            y = self.apply_rope1d(y, positions[:,:,0], cos, sin)
            x = self.apply_rope1d(x, positions[:,:,1], cos, sin)
            tokens = torch.cat((y, x), dim=-1)
            return tokens


class RoPE3D(nn.Module):
    """
    Adapted from https://github.com/rabbityl/lepard/blob/main/models/position_encoding.py

    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def rotate_half(x):
        return torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()

    def embed_rotary(self, x, cos, sin):
        x = x * cos + self.rotate_half(x) * sin
        return x

    def get_cos_sin(self, dim, positions):
        bsize, npoint, _ = positions.shape

        # Ensure dim is divisible by 3
        padded_dim = math.ceil(dim / 3) * 3
        pad_size = padded_dim - dim

        x_position, y_position, z_position = positions[..., 0:1], positions[...,1:2], positions[...,2:3]
        div_term = torch.exp( torch.arange(0, padded_dim // 3, 2, dtype=torch.float, device=positions.device) *  (-math.log(10000.0) / (padded_dim // 3)))
        div_term = div_term.view( 1,1, -1) # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term) # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        # sin/cos [θ0,θ1,θ2......θd/6-1] -> sin/cos [θ0,θ0,θ1,θ1,θ2,θ2......θd/6-1,θd/6-1]
        sinx, cosx, siny, cosy, sinz, cosz = map( lambda  feat:torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
              [ sinx, cosx, siny, cosy, sinz, cosz] )
        sin_pos = torch.cat([sinx,siny,sinz], dim=-1)[:,None,:,:]
        cos_pos = torch.cat([cosx,cosy,cosz], dim=-1)[:,None,:,:]

        # Trim back to the original dimension
        sin_pos = sin_pos[..., :dim]
        cos_pos = cos_pos[..., :dim]

        if cos_pos.requires_grad:
            cos_pos = cos_pos.detach()
        if sin_pos.requires_grad:
            sin_pos = sin_pos.detach()

        return cos_pos, sin_pos

    def forward(self, tokens, positions):
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 3 (x, y and z position of each token)
        output:
            * tokens after appplying RoPE3D (batch_size x nheads x ntokens x dim)
        """
        assert tokens.ndim==4 
        assert positions.ndim==3 and positions.shape[-1] == 3 # Batch, Seq, 3

        head_dim = tokens.shape[-1]

        cos, sin = self.get_cos_sin(head_dim, positions)
        tokens = self.embed_rotary(tokens, cos, sin)

        return tokens


class RoPE2D_Continuous(nn.Module):
    """
    2D Rotary Positional Encoding for image tokens.
    Applies RoPE based on (x, y) coordinates (e.g., pixel or patch positions).
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def rotate_half(x):
        return torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()

    def embed_rotary(self, x, cos, sin):
        return x * cos + self.rotate_half(x) * sin

    def get_cos_sin(self, dim, positions):
        """
        Generate rotary position encoding for 2D (x, y) positions.
        positions: [B, N, 2]
        Returns cos, sin: [B, 1, N, D]
        """
        assert dim % 2 == 0, "Embedding dimension must be divisible by 2"
        bsize, npoint, _ = positions.shape

        x_position, y_position = positions[..., 0:1], positions[..., 1:2]
        half_dim = dim // 2

        div_term = torch.exp(
            torch.arange(0, half_dim, 2, dtype=torch.float, device=positions.device) *
            (-math.log(10000.0) / half_dim)
        )  # [dim/4]

        div_term = div_term.view(1, 1, -1)  # [1, 1, dim//4]

        sinx = torch.sin(x_position * div_term)  # [B, N, dim//4]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)

        # Repeat each sin/cos twice for interleaving
        sinx, cosx, siny, cosy = map(
            lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
            [sinx, cosx, siny, cosy]
        )

        sin_pos = torch.cat([sinx, siny], dim=-1)[:, None, :, :]  # [B, 1, N, D]
        cos_pos = torch.cat([cosx, cosy], dim=-1)[:, None, :, :]  # [B, 1, N, D]

        return cos_pos.detach(), sin_pos.detach()

    def forward(self, tokens, positions):
        """
        tokens: [B, H, N, D] - token features
        positions: [B, N, 2] - (x, y) coordinates for each token
        Returns:
            RoPE-encoded tokens: [B, H, N, D]
        """
        assert tokens.ndim == 4
        assert positions.ndim == 3 and positions.shape[-1] == 2

        head_dim = tokens.shape[-1]
        cos, sin = self.get_cos_sin(head_dim, positions)
        tokens = self.embed_rotary(tokens, cos, sin)
        return tokens


class InvertibleLinearPositionEmbedding(nn.Module):
    def __init__(self, input_dim, feat_dim):
        super().__init__()
        self.position_encoder = nn.Linear(input_dim, feat_dim)

    def forward(self, pos):
        return self.position_encoder(pos)

    def decode(self, feat):
        W_inv = torch.pinverse(self.position_encoder.weight)
        return (feat - self.position_encoder.bias) @ W_inv.T
