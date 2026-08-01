# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# References: https://github.com/naver/croco/blob/master/models/croco.py
# --------------------------------------------------------


import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..grad_ckpt import grad_checkpointing_enabled

from functools import partial

from .build import ENCODER_REGISTRY
from ..blocks import Block
from ...utils.config import configurable


@ENCODER_REGISTRY.register()
class CrocoV2_Encoder(nn.Module):

    @configurable
    def __init__(
            self,
            pos_embed,
            depth=12,
            num_heads=12,
            embed_dim=768,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            qkv_bias=True,
            rope=None,
            act_layer="gelu",
            attn_drop=0.,
            proj_drop=0.,
            drop_path=0.,
            use_flash_attn=False,
            **kwargs
    ):
        super(CrocoV2_Encoder, self).__init__()

        assert pos_embed is not None or rope is not None, "No position embedding provided."

        if pos_embed is not None:
            self.register_buffer('enc_pos_embed', torch.from_numpy(pos_embed).float())
        else:
            self.enc_pos_embed = None

        self.enc_blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                rope=rope,
                act_layer=act_layer,
                drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                use_flash_attn=use_flash_attn
            ) for i in range(depth)])

        self.enc_norm = norm_layer(embed_dim)

    @classmethod
    def from_config(cls, cfg):
        return {
            "embed_dim": cfg.EMBED_DIM,
            "depth": cfg.DEPTH,
            "num_heads": cfg.NUM_HEADS,
            "mlp_ratio": cfg.MLP_RATIO,
            "qkv_bias": cfg.QKV_BIAS,
            "act_layer": cfg.ACTIVATION,
            "attn_drop": cfg.ATTN_DROP,
            "proj_drop": cfg.PROJ_DROP,
            "drop_path": cfg.DROP_PATH,
            "use_flash_attn": cfg.USE_FLASH_ATTN
        }

    def forward(self, x, pos, do_mask=False, return_all_blocks=False, **kwargs):
        """
        x: tensor of shape B x num_patches x embed_dim 
        do_mask: whether to perform masking or not
        return_all_blocks: if True, return the features at the end of every block 
                           instead of just the features from the last block (eg for some prediction heads)
        """

        # embed the image into patches  (x has size B x Npatches x C) 
        # and get position if each return patch (pos has size B x Npatches x 2)

        # add positional embedding without cls token  
        if self.enc_pos_embed is not None:
            x = x + self.enc_pos_embed[None, ...]

        # apply masking 
        B, N, C = x.size()
        if do_mask:
            masks = self.mask_generator(x)
            x = x[~masks].view(B, -1, C)
            posvis = pos[~masks].view(B, -1, 2)
        else:
            B, N, C = x.size()
            masks = torch.zeros((B, N), dtype=bool)
            posvis = pos

        # now apply the transformer encoder and normalization        
        ckpt = grad_checkpointing_enabled() and self.training and torch.is_grad_enabled() and x.requires_grad
        if return_all_blocks:
            out = []
            for blk in self.enc_blocks:
                x = checkpoint(blk, x, posvis, use_reentrant=False) if ckpt else blk(x, posvis)
                out.append(x)
            out[-1] = self.enc_norm(out[-1])
            return out, pos, masks
        else:
            for blk in self.enc_blocks:
                x = checkpoint(blk, x, posvis, use_reentrant=False) if ckpt else blk(x, posvis)
            x = self.enc_norm(x)
            return x, pos, masks

    def freeze_croco_weights(self):
        self.enc_blocks.eval()
        self.enc_norm.eval()

        for params in self.enc_blocks.parameters():
            params.requires_grad = False

        for params in self.enc_norm.parameters():
            params.requires_grad = False
