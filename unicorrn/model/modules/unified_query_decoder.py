"""DETR-style query matching decoder with dual appearance+positional streams.

The decoder iterates point/image queries against fixed image memory using
Gaussian attention (L2-distance kernel via xformers). Three value streams
share one attention matrix: appearance features, positional embeddings, and
raw coordinate residuals for deep supervision. The frustum fork adds a
per-layer ``FrustumHead`` on the pcd→img path.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from ...utils.config import configurable
from ..blocks import DualStreamQueryDecoderBlock, FrustumHead, GaussianKNNSample, Mlp
from ..blocks.utils import freeze_modules, offset2batch
from ..embedder import InvertibleLinearPositionEmbedding
from .build import DECODER_REGISTRY


def _declone(x):
    """Clone inference-mode tensors so checkpoint can save them for backward."""
    return x.clone() if torch.is_tensor(x) and torch.is_inference(x) else x


def _ckpt(training, fn, *args, **kwargs):
    """Gradient-checkpoint ``fn`` during training (recompute in backward, save memory)."""
    needs_grad = any(torch.is_tensor(a) and a.requires_grad for a in args)
    if not (training and torch.is_grad_enabled() and needs_grad):
        return fn(*args, **kwargs)
    args = tuple(_declone(a) for a in args)
    kwargs = {k: _declone(v) for k, v in kwargs.items()}
    return checkpoint(fn, *args, use_reentrant=False, **kwargs)


@DECODER_REGISTRY.register()
class QueryMatchingDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        dec_embed_dim=512,
        pos_embed_dim=128,
        dec_depth=8,
        num_heads=16,
        patch_size=16,
        mlp_ratio=4,
        act_layer="gelu",
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_mem=True,
        k=5,
        query_cross_attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        last_layer_res_only=False,
        **kwargs,
    ):
        super().__init__()
        self.knn_sampler = GaussianKNNSample(k)

        self.patch_size = patch_size
        project_dim = dec_embed_dim // 2
        self.descriptor_2d = nn.Linear(dec_embed_dim, project_dim)
        self.descriptor_3d = nn.Linear(dec_embed_dim, project_dim)

        self.learnable_pos_encoder_2d = InvertibleLinearPositionEmbedding(
            2, pos_embed_dim
        )
        self.learnable_pos_encoder_3d = InvertibleLinearPositionEmbedding(
            3, pos_embed_dim
        )
        self.corr_embed_2d = self.learnable_pos_encoder_2d.decode
        self.corr_embed_3d = self.learnable_pos_encoder_3d.decode

        self.query_decoder_blocks = nn.ModuleList(
            [
                DualStreamQueryDecoderBlock(
                    dim=project_dim,
                    res_dim=pos_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=proj_drop,
                    cross_attn_drop=query_cross_attn_drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    norm_mem=norm_mem,
                    pos_decoder2d=self.corr_embed_2d,
                    pos_decoder3d=self.corr_embed_3d,
                )
                for i in range(dec_depth)
            ]
        )

        self.info_embed = Mlp(project_dim, hidden_features=project_dim, out_features=1)
        self.frustum_head = FrustumHead(project_dim, pos_embed_dim, coord_dim=2)
        self.last_layer_res_only = last_layer_res_only

    def freeze_2d_weights(self):
        freeze_modules(
            self.descriptor_2d, self.learnable_pos_encoder_2d, self.info_embed
        )
        for blk in self.query_decoder_blocks:
            blk.freeze_2d_weights()

    def joint_finetune_trainable(self):
        freeze_modules(
            self.learnable_pos_encoder_2d,
            self.learnable_pos_encoder_3d,
            self.query_decoder_blocks,
            self.info_embed,
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "dec_embed_dim": cfg.DEC_EMBED_DIM,
            "pos_embed_dim": cfg.POS_EMBED_DIM,
            "dec_depth": cfg.DEC_DEPTH,
            "num_heads": cfg.NUM_HEADS,
            "mlp_ratio": cfg.MLP_RATIO,
            "qkv_bias": cfg.QKV_BIAS,
            "act_layer": cfg.ACTIVATION,
            "norm_mem": cfg.NORM_MEM,
            "k": cfg.K,
            "fusion_attn_drop": cfg.FUSION_ATTN_DROP,
            "query_cross_attn_drop": cfg.QUERY_CROSS_ATTN_DROP,
            "proj_drop": cfg.PROJ_DROP,
            "drop_path": cfg.DROP_PATH,
            "last_layer_res_only": cfg.LAST_LAYER_RES_ONLY,
        }

    def _sample_2d_descriptors(self, src_feat, src_patch_shape, sample_pos):
        _b, _q, _ = sample_pos.shape
        norm_queries = 2 * (sample_pos / (self.patch_size * src_patch_shape)) - 1
        desc = F.grid_sample(
            rearrange(
                src_feat.float(),
                "b (h w) c -> b c h w",
                h=src_patch_shape[1],
                w=src_patch_shape[0],
            ),
            norm_queries[:, None, :, :].float(),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        desc = desc.permute(0, 2, 3, 1).reshape(_b, _q, -1)
        return desc

    def _sample_3d_descriptors(self, src_feat, src_point, sample_pos):
        _b, _q, _ = sample_pos.shape
        desc = []
        src_feat_batch, src_coord_batch = offset2batch(
            src_feat, src_point.coord, src_point.offset
        )
        for idx in range(_b):
            desc.append(
                self.knn_sampler(
                    src_feat_batch[idx], src_coord_batch[idx], sample_pos[idx]
                )
            )
        desc = torch.stack(desc)
        return desc

    def forward_img_to_img(
        self,
        src_feat,
        tgt_feat,
        query_pos,
        patch_shape,
        patch_coord_map,
        target_pos=None,
        **kwargs,
    ):
        assert (
            query_pos.shape[-1] == 2
        ), f"Invalid queries dim. Expected 2 but received {query_pos.shape[-1]}"

        src_desc = self._sample_2d_descriptors(
            self.descriptor_2d(src_feat), patch_shape, query_pos
        )
        mem = self.descriptor_2d(tgt_feat)
        tgt_desc = (
            self._sample_2d_descriptors(mem, patch_shape, target_pos)
            if target_pos is not None
            else None
        )

        mem_pos_embed = self.learnable_pos_encoder_2d(patch_coord_map)
        q = src_desc.clone()
        hidden_state = None

        _padding = torch.zeros(
            *patch_coord_map.shape[:-1], 2, device=patch_coord_map.device
        )
        gm_res = torch.cat([patch_coord_map, _padding], dim=-1)
        gm_out = []

        for idx, block in enumerate(self.query_decoder_blocks):
            gm_res_ = None
            if (
                not self.last_layer_res_only
                or idx == len(self.query_decoder_blocks) - 1
            ):
                gm_res_ = gm_res

            ret = _ckpt(
                self.training,
                block.forward_query_to_img,
                q,
                mem,
                patch_coord_map,
                mem_pos_embed,
                hidden_state,
                img_query=True,
                gm_res=gm_res_,
            )

            if len(ret) == 3:
                q, hidden_state, gm_tgt = ret
                gm_out.append(gm_tgt[..., :2])
            else:
                q, hidden_state = ret

        corr = self.corr_embed_2d(hidden_state)
        info = self.info_embed(q)

        return corr, info, q, src_desc, tgt_desc, gm_out

    def forward_img_to_pcd(
        self,
        src_feat,
        tgt_feat,
        tgt_point,
        query_pos,
        patch_shape,
        target_pos=None,
        **kwargs,
    ):
        assert (
            query_pos.shape[-1] == 2
        ), f"Invalid queries dim. Expected 2 but received {query_pos.shape[-1]}"

        src_desc = self._sample_2d_descriptors(
            self.descriptor_2d(src_feat), patch_shape, query_pos
        )
        mem = self.descriptor_3d(tgt_feat)
        tgt_desc = (
            self._sample_3d_descriptors(mem, tgt_point, target_pos)
            if target_pos is not None
            else None
        )

        mem_pos_embed = self.learnable_pos_encoder_3d(tgt_point.coord)
        q = src_desc.clone()
        hidden_state = None

        _padding = torch.zeros(
            *tgt_point.coord.shape[:-1], 1, device=tgt_point.coord.device
        )
        gm_res = torch.cat([tgt_point.coord, _padding], dim=-1)
        gm_out = []

        for idx, block in enumerate(self.query_decoder_blocks):
            gm_res_ = None
            if (
                not self.last_layer_res_only
                or idx == len(self.query_decoder_blocks) - 1
            ):
                gm_res_ = gm_res

            ret = _ckpt(
                self.training,
                block.forward_query_to_pcd,
                q,
                mem,
                tgt_point.coord,
                tgt_point.offset,
                mem_pos_embed,
                hidden_state,
                img_query=True,
                gm_res=gm_res_,
            )
            if len(ret) == 3:
                q, hidden_state, gm_tgt = ret
                gm_out.append(gm_tgt[..., :3])
            else:
                q, hidden_state = ret

        corr = self.corr_embed_3d(hidden_state)
        info = self.info_embed(q)

        return corr, info, q, src_desc, tgt_desc, gm_out

    def forward_pcd_to_img(
        self,
        src_feat,
        src_point,
        tgt_feat,
        query_pos,
        patch_coord_map,
        target_pos=None,
        **kwargs,
    ):
        """Decode 3D point queries against image memory.

        Returns a 7-tuple: ``(corr, info, frustum_out, q, src_desc, tgt_desc, gm_out)``.
        """
        assert (
            query_pos.shape[-1] == 3
        ), f"Invalid queries dim. Expected 3 but received {query_pos.shape[-1]}"

        src_desc = self._sample_3d_descriptors(
            self.descriptor_3d(src_feat), src_point, query_pos
        )
        mem = self.descriptor_2d(tgt_feat)
        tgt_desc = (
            self._sample_2d_descriptors(mem, kwargs["patch_shape"], target_pos)
            if target_pos is not None
            else None
        )

        mem_pos_embed = self.learnable_pos_encoder_2d(patch_coord_map)
        q = src_desc.clone()
        hidden_state = None

        _padding = torch.zeros(
            *patch_coord_map.shape[:-1], 2, device=patch_coord_map.device
        )
        gm_res = torch.cat([patch_coord_map, _padding], dim=-1)
        gm_out = []
        frustum_out = []

        for idx, block in enumerate(self.query_decoder_blocks):
            gm_res_ = None
            if (
                not self.last_layer_res_only
                or idx == len(self.query_decoder_blocks) - 1
            ):
                gm_res_ = gm_res

            ret = _ckpt(
                self.training,
                block.forward_query_to_img,
                q,
                mem,
                patch_coord_map,
                mem_pos_embed,
                hidden_state,
                img_query=False,
                gm_res=gm_res_,
            )
            if len(ret) == 3:
                q, hidden_state, gm_tgt = ret
                gm_out.append(gm_tgt[..., :2])
                frustum_out.append(
                    self.frustum_head(
                        q, hidden_state, gm_tgt[..., :2], self.corr_embed_2d(hidden_state)
                    )
                )
            else:
                q, hidden_state = ret

        corr = self.corr_embed_2d(hidden_state)
        info = self.info_embed(q)

        return corr, info, frustum_out, q, src_desc, tgt_desc, gm_out

    def forward_pcd_to_pcd(
        self,
        src_feat,
        src_point,
        tgt_feat,
        tgt_point,
        query_pos,
        target_pos=None,
        **kwargs,
    ):
        assert (
            query_pos.shape[-1] == 3
        ), f"Invalid queries dim. Expected 3 but received {query_pos.shape[-1]}"

        src_desc = self._sample_3d_descriptors(
            self.descriptor_3d(src_feat), src_point, query_pos
        )
        mem = self.descriptor_3d(tgt_feat)
        tgt_desc = (
            self._sample_3d_descriptors(mem, tgt_point, target_pos)
            if target_pos is not None
            else None
        )

        mem_pos_embed = self.learnable_pos_encoder_3d(tgt_point.coord)
        q = src_desc.clone()
        hidden_state = None

        _padding = torch.zeros(
            *tgt_point.coord.shape[:-1], 1, device=tgt_point.coord.device
        )
        gm_res = torch.cat([tgt_point.coord, _padding], dim=-1)
        gm_out = []

        for idx, block in enumerate(self.query_decoder_blocks):
            gm_res_ = None
            if (
                not self.last_layer_res_only
                or idx == len(self.query_decoder_blocks) - 1
            ):
                gm_res_ = gm_res

            ret = _ckpt(
                self.training,
                block.forward_query_to_pcd,
                q,
                mem,
                tgt_point.coord,
                tgt_point.offset,
                mem_pos_embed,
                hidden_state,
                img_query=False,
                gm_res=gm_res_,
            )
            if len(ret) == 3:
                q, hidden_state, gm_tgt = ret
                gm_out.append(gm_tgt[..., :3])
            else:
                q, hidden_state = ret

        corr = self.corr_embed_3d(hidden_state)
        info = self.info_embed(q)

        return corr, info, q, src_desc, tgt_desc, gm_out
