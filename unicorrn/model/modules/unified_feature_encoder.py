from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..grad_ckpt import grad_checkpointing_enabled

from ...utils import cartesian_img_coord
from ...utils.config import configurable
from ..blocks import MMDecoderBlockBidirectional
from ..blocks.utils import freeze_modules
from .build import DECODER_REGISTRY, build_decoder


def _ckpt(training, fn, *args):
    """Gradient-checkpoint ``fn(*args)`` during training (recompute in backward, save mem).

    Skips when not training, grad is disabled, or no input needs grad (frozen stage),
    so inference and frozen-encoder training are unaffected.
    """
    needs_grad = any(torch.is_tensor(a) and a.requires_grad for a in args)
    if grad_checkpointing_enabled() and training and torch.is_grad_enabled() and needs_grad:
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


@DECODER_REGISTRY.register()
class UnifiedFeatureEncoder(nn.Module):
    """
    Unified feature fusion encoder for information sharing across images and point clouds.

    """

    @configurable
    def __init__(
        self,
        cfg,
        img_enc_embed_dim=768,
        pcd_enc_embed_dim=512,
        dec_embed_dim=512,
        dec_depth=8,
        num_heads=16,
        patch_size=16,
        mlp_ratio=4,
        act_layer="gelu",
        point_order=("z", "z-trans", "hilbert", "hilbert-trans"),
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        modal_norm=True,
        norm_mem=True,
        fusion_attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        pcd_patch_size=1024,
        **kwargs
    ):
        super().__init__()
        self.decoder_embed_img = nn.Sequential(
            nn.Linear(img_enc_embed_dim, dec_embed_dim, bias=True),
            norm_layer(dec_embed_dim) if modal_norm else nn.Identity(),
        )
        self.decoder_embed_pcd = nn.Sequential(
            nn.Linear(pcd_enc_embed_dim, dec_embed_dim, bias=True),
            norm_layer(dec_embed_dim) if modal_norm else nn.Identity(),
        )

        self.point_order = point_order
        self.bbone_patch_size = patch_size

        self.order_indices = [i % len(point_order) for i in range(dec_depth)]

        self.dec_blocks = nn.ModuleList(
            [
                MMDecoderBlockBidirectional(
                    dim=dec_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=proj_drop,
                    attn_drop=fusion_attn_drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    norm_mem=norm_mem,
                    order_index=self.order_indices[i],
                    pcd_patch_size=pcd_patch_size,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(dec_embed_dim)

        self.img_upsampler = build_decoder(cfg.IMG_UPSAMPLER, patch_size=patch_size)
        self.pcd_upsampler = build_decoder(cfg.PCD_UPSAMPLER)

    @property
    def patch_size(self):
        return self.img_upsampler.upscale_patch_size

    def freeze_2d_weights(self):
        freeze_modules(
            self.decoder_embed, self.dec_blocks, self.dec_norm, self.img_upsampler
        )

    def joint_finetune_trainable(self):
        freeze_modules(self.dec_blocks, self.dec_norm)

    @classmethod
    def from_config(cls, cfg):
        return {
            "cfg": cfg,
            "img_enc_embed_dim": cfg.IMG_ENC_EMBED_DIM,
            "pcd_enc_embed_dim": cfg.PCD_ENC_EMBED_DIM,
            "dec_embed_dim": cfg.DEC_EMBED_DIM,
            "dec_depth": cfg.DEC_DEPTH,
            "num_heads": cfg.NUM_HEADS,
            "mlp_ratio": cfg.MLP_RATIO,
            "qkv_bias": cfg.QKV_BIAS,
            "act_layer": cfg.ACTIVATION,
            "point_order": cfg.POINT_ORDER,
            "modal_norm": cfg.MODAL_NORM,
            "norm_mem": cfg.NORM_MEM,
            "fusion_attn_drop": cfg.FUSION_ATTN_DROP,
            "proj_drop": cfg.PROJ_DROP,
            "drop_path": cfg.DROP_PATH,
            "pcd_patch_size": cfg.POINT_CLOUD_PATCH_SIZE,
        }

    def forward_img_to_img(self, src_feat, tgt_feat, img_H, img_W):
        _b = src_feat.shape[0]

        src_feat_dec = self.decoder_embed_img(src_feat)
        tgt_feat_dec = self.decoder_embed_img(tgt_feat)

        src_patch_shape = (
            torch.tensor([img_W, img_H], device=src_feat.device)
            // self.bbone_patch_size
        )
        patch_coord_map = (
            cartesian_img_coord(
                src_patch_shape[1].item(),
                src_patch_shape[0].item(),
                patch_size=self.bbone_patch_size,
                norm=True,
            )
            .repeat(_b, 1, 1, 1)
            .to(src_feat.device)
            .view(_b, -1, 2)
        )
        for block in self.dec_blocks:
            src_feat_dec, tgt_feat_dec = _ckpt(
                self.training, block.forward_img_to_img,
                src_feat_dec, tgt_feat_dec, patch_coord_map, patch_coord_map,
            )
        src_feat_dec = self.dec_norm(src_feat_dec)
        tgt_feat_dec = self.dec_norm(tgt_feat_dec)

        src_upscale_dict = self.img_upsampler(src_feat, src_feat_dec, src_patch_shape)
        tgt_upscale_dict = self.img_upsampler(tgt_feat, tgt_feat_dec, src_patch_shape)
        src_upscale_patch_shape = torch.tensor(
            [src_upscale_dict["W"], src_upscale_dict["H"]], device=src_feat.device
        )
        upscale_patch_coord_map = (
            cartesian_img_coord(
                src_upscale_dict["H"],
                src_upscale_dict["W"],
                patch_size=self.patch_size,
                norm=True,
            )
            .repeat(_b, 1, 1, 1)
            .to(src_feat.device)
            .view(_b, -1, 2)
        )

        return (
            # self.dec_norm(src_upscale_dict['feat_f']),
            # self.dec_norm(tgt_upscale_dict['feat_f']),
            src_upscale_dict["feat_f"],
            tgt_upscale_dict["feat_f"],
            src_upscale_patch_shape,
            upscale_patch_coord_map,
        )

    def forward_img_to_pcd(self, src_feat, tgt_point, img_H, img_W):
        _b = src_feat.shape[0]

        src_feat_dec = self.decoder_embed_img(src_feat)
        tgt_feat_dec = self.decoder_embed_pcd(tgt_point.feat)

        src_patch_shape = (
            torch.tensor([img_W, img_H], device=src_feat.device)
            // self.bbone_patch_size
        )
        patch_coord_map = (
            cartesian_img_coord(
                src_patch_shape[1].item(),
                src_patch_shape[0].item(),
                patch_size=self.bbone_patch_size,
                norm=True,
            )
            .repeat(_b, 1, 1, 1)
            .to(src_feat.device)
            .view(_b, -1, 2)
        )
        for block in self.dec_blocks:
            src_feat_dec, tgt_feat_dec = _ckpt(
                self.training, block.forward_img_to_pcd,
                src_feat_dec, tgt_feat_dec, patch_coord_map, tgt_point,
            )
        src_feat_dec = self.dec_norm(src_feat_dec)
        tgt_feat_dec = self.dec_norm(tgt_feat_dec)

        src_upscale_dict = self.img_upsampler(src_feat, src_feat_dec, src_patch_shape)
        tgt_upscale_point = self.pcd_upsampler(tgt_feat_dec, tgt_point)
        src_upscale_patch_shape = torch.tensor(
            [src_upscale_dict["W"], src_upscale_dict["H"]], device=src_feat.device
        )
        upscale_patch_coord_map = (
            cartesian_img_coord(
                src_upscale_dict["H"],
                src_upscale_dict["W"],
                patch_size=self.patch_size,
                norm=True,
            )
            .repeat(_b, 1, 1, 1)
            .to(src_feat.device)
            .view(_b, -1, 2)
        )
        # tgt_upscale_point.feat = self.dec_norm(tgt_upscale_point.feat)

        return (
            # self.dec_norm(src_upscale_dict['feat_f']),
            src_upscale_dict["feat_f"],
            tgt_upscale_point,
            src_upscale_patch_shape,
            upscale_patch_coord_map,
        )

    def forward_pcd_to_img(self, src_point, tgt_feat, img_H, img_W):
        _b = tgt_feat.shape[0]

        src_feat_dec = self.decoder_embed_pcd(src_point.feat)
        tgt_feat_dec = self.decoder_embed_img(tgt_feat)

        tgt_patch_shape = (
            torch.tensor([img_W, img_H], device=tgt_feat.device)
            // self.bbone_patch_size
        )
        patch_coord_map = (
            cartesian_img_coord(
                tgt_patch_shape[1].item(),
                tgt_patch_shape[0].item(),
                patch_size=self.bbone_patch_size,
                norm=True,
            )
            .repeat(_b, 1, 1, 1)
            .to(tgt_feat.device)
            .view(_b, -1, 2)
        )
        for block in self.dec_blocks:
            src_feat_dec, tgt_feat_dec = _ckpt(
                self.training, block.forward_pcd_to_img,
                src_feat_dec, tgt_feat_dec, src_point, patch_coord_map,
            )
        src_feat_dec = self.dec_norm(src_feat_dec)
        tgt_feat_dec = self.dec_norm(tgt_feat_dec)

        src_upscale_point = self.pcd_upsampler(src_feat_dec, src_point)
        tgt_upscale_dict = self.img_upsampler(tgt_feat, tgt_feat_dec, tgt_patch_shape)
        tgt_upscale_patch_shape = torch.tensor(
            [tgt_upscale_dict["W"], tgt_upscale_dict["H"]], device=tgt_feat.device
        )
        upscale_patch_coord_map = (
            cartesian_img_coord(
                tgt_upscale_dict["H"],
                tgt_upscale_dict["W"],
                patch_size=self.patch_size,
                norm=True,
            )
            .repeat(_b, 1, 1, 1)
            .to(tgt_feat.device)
            .view(_b, -1, 2)
        )
        # src_upscale_point.feat = self.dec_norm(src_upscale_point.feat)

        return (
            src_upscale_point,
            # self.dec_norm(tgt_upscale_dict['feat_f']),
            tgt_upscale_dict["feat_f"],
            tgt_upscale_patch_shape,
            upscale_patch_coord_map,
        )

    def forward_pcd_to_pcd(self, src_point, tgt_point):
        src_feat_dec = self.decoder_embed_pcd(src_point.feat)
        tgt_feat_dec = self.decoder_embed_pcd(tgt_point.feat)

        for block in self.dec_blocks:
            src_feat_dec, tgt_feat_dec = _ckpt(
                self.training, block.forward_pcd_to_pcd,
                src_feat_dec, tgt_feat_dec, src_point, tgt_point,
            )
        src_feat_dec = self.dec_norm(src_feat_dec)
        tgt_feat_dec = self.dec_norm(tgt_feat_dec)

        src_upscale_point = self.pcd_upsampler(src_feat_dec, src_point)
        tgt_upscale_point = self.pcd_upsampler(tgt_feat_dec, tgt_point)
        # src_upscale_point.feat = self.dec_norm(src_upscale_point.feat)
        # tgt_upscale_point.feat = self.dec_norm(tgt_upscale_point.feat)

        return src_upscale_point, tgt_upscale_point
