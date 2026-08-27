"""UniCorrn: Unified Correspondence Transformer Across 2D and 3D.

Dual-encoder (CroCo V2 image + PTv3 point cloud) model with a DETR-style
Gaussian-attention decoder for cross-modal matching. The frustum fork adds a
``FrustumHead`` in the pcd→img decoder path for geometric frustum classification.
"""

import torch
import torch.nn as nn

from .backbones import ENCODER_REGISTRY, build_encoder
from .blocks.utils import freeze_modules
from .build import MODEL_REGISTRY
from .embedder import ManyAR_PatchEmbed, RoPE2D, get_2d_sincos_pos_embed
from .modules import build_decoder


@MODEL_REGISTRY.register()
class UniCorrn(nn.Module):
    """
    UniCorrn: Unified Correspondence Transformer Across 2D and 3D

    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.bidirectional = cfg.BIDIRECTIONAL

        image_encoder_cls = ENCODER_REGISTRY.get(cfg.BACKBONE.IMAGE.NAME)
        if getattr(image_encoder_cls, "consumes_raw_image", False):
            self.patch_embed = nn.Identity()
            num_patches = 0
        else:
            self.patch_embed = ManyAR_PatchEmbed(
                cfg.IMG_SIZE, cfg.PATCH_SIZE, 3, cfg.BACKBONE.IMAGE.EMBED_DIM, upscale=False
            )
            num_patches = self.patch_embed.num_patches

        if cfg.IMG_POS_EMBED == "cosine":
            # positional embedding
            enc_pos_embed = get_2d_sincos_pos_embed(
                cfg.BACKBONE.IMAGE.EMBED_DIM, int(num_patches**0.5), n_cls_token=0
            )
            rope = None  # nothing for cosine
        elif cfg.IMG_POS_EMBED.startswith("RoPE"):  # eg RoPE100
            enc_pos_embed = None  # nothing to add in the encoder with RoPE

            if RoPE2D is None:
                raise ImportError(
                    "Cannot find cuRoPE2D, please install it following the README instructions"
                )
            freq = float(cfg.IMG_POS_EMBED[len("RoPE") :])
            rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError("Unknown IMG_POS_EMBED: " + cfg.IMG_POS_EMBED)

        self.img_encoder = build_encoder(
            cfg.BACKBONE.IMAGE, pos_embed=enc_pos_embed, rope=rope
        )
        self.pcd_encoder = build_encoder(cfg.BACKBONE.POINT_CLOUD)
        self.feat_encoder = build_decoder(cfg.FEAT_ENCODER, patch_size=cfg.PATCH_SIZE)
        self.decoder = build_decoder(
            cfg.DECODER, patch_size=self.feat_encoder.patch_size
        )

    def freeze_croco_encoder(self):
        freeze_modules(self.patch_embed)
        self.img_encoder.freeze_croco_weights()

    def freeze_encoder_weights(self):
        freeze_modules(self.img_encoder, self.feat_encoder)

    def freeze_2d_weights(self):
        freeze_modules(self.patch_embed, self.img_encoder)
        self.feat_encoder.freeze_2d_weights()
        self.decoder.freeze_2d_weights()

    def joint_finetune_trainable(self):
        freeze_modules(self.patch_embed, self.img_encoder, self.pcd_encoder)
        self.feat_encoder.joint_finetune_trainable()
        self.decoder.joint_finetune_trainable()

    def _encode_img(self, img):
        # DINOv3-style backbones patchify the raw image themselves; CrocoV2 needs
        # pre-patchified tokens + 2D positions from patch_embed.
        if getattr(self.img_encoder, "consumes_raw_image", False):
            return self.img_encoder(img)[0]
        B = img.shape[0]
        true_shape = torch.tensor(img.shape[-2:])[None].repeat(B, 1)
        patches, pos = self.patch_embed(img, true_shape)
        img_feat, _, _ = self.img_encoder(patches, pos)

        return img_feat

    def _encode_img_pair(self, img1, img2):
        if getattr(self.img_encoder, "consumes_raw_image", False):
            feat, _, _ = self.img_encoder(torch.cat((img1, img2), dim=0))
            return feat.chunk(2, dim=0)
        B = img1.shape[0]
        true_shape1 = torch.tensor(img1.shape[-2:])[None].repeat(B, 1)
        true_shape2 = torch.tensor(img2.shape[-2:])[None].repeat(B, 1)

        patches, pos = self.patch_embed(
            torch.cat((img1, img2), dim=0), torch.cat((true_shape1, true_shape2), dim=0)
        )
        feat, _, _ = self.img_encoder(patches, pos)
        feat1, feat2 = feat.chunk(2, dim=0)

        return feat1, feat2

    def forward_img_to_img(self, src_img, tgt_img, query_pos):
        assert src_img.shape == tgt_img.shape
        B, C, H, W = src_img.shape
        src_feat, tgt_feat = self._encode_img_pair(src_img, tgt_img)

        (src_feat, tgt_feat, patch_shape, patch_coord_map) = (
            self.feat_encoder.forward_img_to_img(src_feat, tgt_feat, img_H=H, img_W=W)
        )

        if self.bidirectional and self.training:
            query_src2tgt, query_tgt2src = torch.chunk(query_pos, 2, dim=1)
            (
                out_src2tgt,
                info_src2tgt,
                qfeat_src,
                desc_src,
                desc_tgt,
                gm_out_src2tgt,
            ) = self.decoder.forward_img_to_img(
                src_feat,
                tgt_feat,
                query_src2tgt,
                patch_shape=patch_shape,
                patch_coord_map=patch_coord_map,
                target_pos=query_tgt2src,
            )
            (out_tgt2src, info_tgt2src, qfeat_tgt, _, _, gm_out_tgt2src) = (
                self.decoder.forward_img_to_img(
                    tgt_feat,
                    src_feat,
                    query_tgt2src,
                    patch_shape=patch_shape,
                    patch_coord_map=patch_coord_map,
                )
            )

            gm_intermediates = []
            for _gm_out_src2tgt, _gm_out_tgt2src in zip(gm_out_src2tgt, gm_out_tgt2src):
                gm_out = torch.cat([_gm_out_src2tgt, _gm_out_tgt2src], dim=1)
                gm_intermediates.append(gm_out)

            out = torch.cat([out_src2tgt, out_tgt2src], dim=1)
            info = torch.cat([info_src2tgt, info_tgt2src], dim=1)

            return {
                "corr_predictions": out,
                "info_predictions": info,
                "qfeat_src": qfeat_src,
                "qfeat_tgt": qfeat_tgt,
                "desc_src": desc_src,
                "desc_tgt": desc_tgt,
                "gm_intermediates": gm_intermediates,
            }

        out, info, qfeat_src, desc_src, desc_tgt, gm_out = (
            self.decoder.forward_img_to_img(
                src_feat,
                tgt_feat,
                query_pos,
                patch_shape=patch_shape,
                patch_coord_map=patch_coord_map,
            )
        )

        return {
            "corr_predictions": out,
            "info_predictions": info,
            "qfeat_src": qfeat_src,
            "qfeat_tgt": None,
            "desc_src": desc_src,
            "desc_tgt": desc_tgt,
            "gm_intermediates": gm_out,
        }

    def _frustum_pcd_to_img(self, point, img_feat, query_pos, patch_coord_map):
        """Classify extra 3D queries as in/out camera frustum, reusing encoded features."""
        frustum_out = self.decoder.forward_pcd_to_img(
            point.feat,
            point,
            img_feat,
            query_pos,
            patch_coord_map=patch_coord_map,
        )[2]
        return frustum_out[-1], frustum_out

    def forward_img_to_pcd(
        self, src_img, sample, query_pos_2d, query_pos_3d=None, frustum_query_pos_3d=None
    ):
        B, C, H, W = src_img.shape

        tgt_point = {
            "feat": sample.get("feats", sample["points"]),
            "coord": sample["points"],
            "grid_coord": sample["grid_coord"].int(),
            "offset": torch.cumsum(sample["lengths"], dim=0).to(
                sample["points"].device
            ),
        }
        src_feat = self._encode_img(src_img)
        tgt_point = self.pcd_encoder(tgt_point)

        (src_feat, tgt_point, patch_shape, patch_coord_map) = (
            self.feat_encoder.forward_img_to_pcd(src_feat, tgt_point, img_H=H, img_W=W)
        )

        if self.bidirectional and self.training and query_pos_3d is not None:
            (
                out_img2pcd,
                info_img2pcd,
                qfeat_src,
                desc_src,
                desc_tgt,
                gm_out_img2pcd,
            ) = self.decoder.forward_img_to_pcd(
                src_feat,
                tgt_point.feat,
                tgt_point,
                query_pos_2d,
                patch_shape=patch_shape,
                target_pos=query_pos_3d,
            )
            (out_pcd2img, info_pcd2img, _, qfeat_tgt, _, _, gm_out_pcd2img) = (
                self.decoder.forward_pcd_to_img(
                    tgt_point.feat,
                    tgt_point,
                    src_feat,
                    query_pos_3d,
                    patch_coord_map=patch_coord_map,
                )
            )

            output = {
                "img2pcd": {
                    "corr_predictions": out_img2pcd,
                    "info_predictions": info_img2pcd,
                    "gm_intermediates": gm_out_img2pcd,
                },
                "pcd2img": {
                    "corr_predictions": out_pcd2img,
                    "info_predictions": info_pcd2img,
                    "gm_intermediates": gm_out_pcd2img,
                },
                "qfeat_src": qfeat_src,
                "qfeat_tgt": qfeat_tgt,
                "desc_src": desc_src,
                "desc_tgt": desc_tgt,
            }
            if frustum_query_pos_3d is not None:
                output["frustum_predictions"], output["frustum_intermediates"] = (
                    self._frustum_pcd_to_img(
                        tgt_point, src_feat, frustum_query_pos_3d, patch_coord_map
                    )
                )
            return output

        out, info, qfeat_src, desc_src, desc_tgt, gm_out = (
            self.decoder.forward_img_to_pcd(
                src_feat,
                tgt_point.feat,
                tgt_point,
                query_pos_2d,
                patch_shape=patch_shape,
            )
        )

        output = {
            "img2pcd": {"corr_predictions": out, "info_predictions": info},
            "qfeat_src": qfeat_src,
            "qfeat_tgt": None,
            "desc_src": desc_src,
            "desc_tgt": desc_tgt,
            "gm_intermediates": gm_out,
        }
        if frustum_query_pos_3d is not None:
            output["frustum_predictions"], output["frustum_intermediates"] = (
                self._frustum_pcd_to_img(
                    tgt_point, src_feat, frustum_query_pos_3d, patch_coord_map
                )
            )
        return output

    def forward_pcd_to_img(
        self, sample, tgt_img, query_pos_3d, query_pos_2d=None, cached_img_feat=None
    ):
        B, C, H, W = tgt_img.shape

        src_point = {
            "feat": sample.get("feats", sample["points"]),
            "coord": sample["points"],
            "grid_coord": sample["grid_coord"].int(),
            "offset": torch.cumsum(sample["lengths"], dim=0).to(
                sample["points"].device
            ),
        }
        src_point = self.pcd_encoder(src_point)
        tgt_feat = self._encode_img(tgt_img) if cached_img_feat is None else cached_img_feat

        (src_point, tgt_feat, patch_shape, patch_coord_map) = (
            self.feat_encoder.forward_pcd_to_img(src_point, tgt_feat, img_H=H, img_W=W)
        )

        if self.bidirectional and self.training and query_pos_2d is not None:
            (
                out_pcd2img,
                info_pcd2img,
                _,
                qfeat_src,
                desc_src,
                desc_tgt,
                gm_out_pcd2img,
            ) = self.decoder.forward_pcd_to_img(
                src_point.feat,
                src_point,
                tgt_feat,
                query_pos_3d,
                patch_coord_map=patch_coord_map,
                target_pos=query_pos_2d,
                patch_shape=patch_shape,
            )
            (out_img2pcd, info_img2pcd, qfeat_tgt, _, _, gm_out_img2pcd) = (
                self.decoder.forward_img_to_pcd(
                    tgt_feat,
                    src_point.feat,
                    src_point,
                    query_pos_2d,
                    patch_shape=patch_shape,
                )
            )

            return {
                "pcd2img": {
                    "corr_predictions": out_pcd2img,
                    "info_predictions": info_pcd2img,
                    "gm_intermediates": gm_out_pcd2img,
                },
                "img2pcd": {
                    "corr_predictions": out_img2pcd,
                    "info_predictions": info_img2pcd,
                    "gm_intermediates": gm_out_img2pcd,
                },
                "qfeat_src": qfeat_src,
                "qfeat_tgt": qfeat_tgt,
                "desc_src": desc_src,
                "desc_tgt": desc_tgt,
            }

        out, info, frustum_out, qfeat_src, desc_src, desc_tgt, gm_out = (
            self.decoder.forward_pcd_to_img(
                src_point.feat,
                src_point,
                tgt_feat,
                query_pos_3d,
                patch_coord_map=patch_coord_map,
            )
        )

        return {
            "pcd2img": {
                "corr_predictions": out,
                "info_predictions": info,
                "frustum_predictions": frustum_out[-1],
                "frustum_intermediates": frustum_out,
            },
            "qfeat_src": qfeat_src,
            "qfeat_tgt": None,
            "desc_src": desc_src,
            "desc_tgt": desc_tgt,
            "gm_intermediates": gm_out,
        }

    def forward_pcd_to_pcd(self, sample, query_pos):
        src_point = {
            "feat": sample.get("src_feats", sample["src_pcd"]),
            "coord": sample["src_pcd"],
            "grid_coord": sample["src_grid_coord"].int(),
            "offset": torch.cumsum(sample["src_length"], dim=0).to(
                sample["src_pcd"].device
            ),
        }
        tgt_point = {
            "feat": sample.get("tgt_feats", sample["tgt_pcd"]),
            "coord": sample["tgt_pcd"],
            "grid_coord": sample["tgt_grid_coord"].int(),
            "offset": torch.cumsum(sample["tgt_length"], dim=0).to(
                sample["tgt_pcd"].device
            ),
        }
        src_point = self.pcd_encoder(src_point)
        tgt_point = self.pcd_encoder(tgt_point)
        src_point, tgt_point = self.feat_encoder.forward_pcd_to_pcd(
            src_point, tgt_point
        )

        if self.bidirectional and self.training:
            query_src2tgt, query_tgt2src = torch.chunk(query_pos, 2, dim=1)
            (
                out_src2tgt,
                info_src2tgt,
                qfeat_src,
                desc_src,
                desc_tgt,
                gm_out_src2tgt,
            ) = self.decoder.forward_pcd_to_pcd(
                src_point.feat,
                src_point,
                tgt_point.feat,
                tgt_point,
                query_src2tgt,
                target_pos=query_tgt2src,
            )
            (out_tgt2src, info_tgt2src, qfeat_tgt, _, _, gm_out_tgt2src) = (
                self.decoder.forward_pcd_to_pcd(
                    tgt_point.feat, tgt_point, src_point.feat, src_point, query_tgt2src
                )
            )

            gm_intermediates = []
            for _gm_out_src2tgt, _gm_out_tgt2src in zip(gm_out_src2tgt, gm_out_tgt2src):
                gm_out = torch.cat([_gm_out_src2tgt, _gm_out_tgt2src], dim=1)
                gm_intermediates.append(gm_out)

            out = torch.cat([out_src2tgt, out_tgt2src], dim=1)
            info = torch.cat([info_src2tgt, info_tgt2src], dim=1)

            return {
                "corr_predictions": out,
                "info_predictions": info,
                "qfeat_src": qfeat_src,
                "qfeat_tgt": qfeat_tgt,
                "desc_src": desc_src,
                "desc_tgt": desc_tgt,
                "gm_intermediates": gm_intermediates,
            }

        out, info, qfeat_src, desc_src, desc_tgt, gm_out = (
            self.decoder.forward_pcd_to_pcd(
                src_point.feat, src_point, tgt_point.feat, tgt_point, query_pos
            )
        )

        return {
            "corr_predictions": out,
            "info_predictions": info,
            "qfeat_src": qfeat_src,
            "qfeat_tgt": None,
            "desc_src": desc_src,
            "desc_tgt": desc_tgt,
            "gm_intermediates": gm_out,
        }

    @torch.no_grad()
    def forward_img_to_img_bidirectional(self, src_img, tgt_img, query_pos):
        assert src_img.shape == tgt_img.shape
        B, C, H, W = src_img.shape
        src_feat, tgt_feat = self._encode_img_pair(src_img, tgt_img)

        (src_feat, tgt_feat, patch_shape, patch_coord_map) = (
            self.feat_encoder.forward_img_to_img(src_feat, tgt_feat, img_H=H, img_W=W)
        )

        out_src2tgt, info_src2tgt, _, _, _, _ = self.decoder.forward_img_to_img(
            src_feat,
            tgt_feat,
            query_pos,
            patch_shape=patch_shape,
            patch_coord_map=patch_coord_map,
        )

        # de-normalize coordinates
        cycle_queries = out_src2tgt.clone()
        cycle_queries[:, :, 0] = cycle_queries[:, :, 0] * W
        cycle_queries[:, :, 1] = cycle_queries[:, :, 1] * H

        out_tgt2src, info_tgt2src, _, _, _, _ = self.decoder.forward_img_to_img(
            tgt_feat,
            src_feat,
            cycle_queries,
            patch_shape=patch_shape,
            patch_coord_map=patch_coord_map,
        )

        return {
            "corr_predictions_src2tgt": out_src2tgt,
            "info_predictions_src2tgt": info_src2tgt,
            "corr_predictions_tgt2src": out_tgt2src,
            "info_predictions_tgt2src": info_tgt2src,
        }

    def forward(self, task, **kwargs):
        if task == "img2img":
            return self.forward_img_to_img(**kwargs)
        elif task == "img2pcd":
            return self.forward_img_to_pcd(**kwargs)
        elif task == "pcd2img":
            return self.forward_pcd_to_img(**kwargs)
        elif task == "pcd2pcd":
            return self.forward_pcd_to_pcd(**kwargs)
        else:
            raise NotImplementedError("Unknown task: " + task)
