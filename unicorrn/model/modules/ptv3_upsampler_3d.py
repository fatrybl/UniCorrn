import torch
import torch.nn as nn

from functools import partial

from .build import DECODER_REGISTRY
from ..blocks.point_transformer_v3 import PointSequential, SerializedUnpooling, Block
from ...utils.config import configurable


@DECODER_REGISTRY.register()
class PTv3Upsampler3D(nn.Module):
    @configurable
    def __init__(self,
                 bbone_dims=(32, 64, 128, 256, 512),
                 enc_embed_dim=512,
                 dec_dims=(64, 64, 128, 256),
                 dec_depths=(2, 2, 2, 2),
                 num_heads=(4, 4, 8, 16),
                 patch_size=(1024, 1024, 1024, 1024),
                 order=("z", "z-trans", "hilbert", "hilbert-trans"),
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 drop_path=0.,
                 pre_norm=True,
                 enable_rpe=False,
                 enable_flash=False,
                 upcast_attention=False,
                 upcast_softmax=False,
                 **kwargs):
        super().__init__()
        act_layer = nn.GELU
        dec_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
        ]
        self.num_stages = len(dec_depths)
        self.dec = PointSequential()
        dec_channels = list(dec_dims) + [enc_embed_dim]
        num_dec_stages = len(dec_channels) - 1
        for s in reversed(range(num_dec_stages)):
            dec_drop_path_ = dec_drop_path[
                             sum(dec_depths[:s]): sum(dec_depths[: s + 1])
                             ]
            dec_drop_path_.reverse()
            dec = PointSequential()
            skip_connection_index = s
            dec.add(
                SerializedUnpooling(
                    in_channels=dec_channels[s + 1],
                    skip_channels=bbone_dims[skip_connection_index],
                    out_channels=dec_channels[s],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                ),
                name="up",
            )
            for i in range(dec_depths[s]):
                dec.add(
                    Block(
                        channels=dec_channels[s],
                        num_heads=num_heads[s],
                        patch_size=patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=dec_drop_path_[i],
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            self.dec.add(module=dec, name=f"dec{s}")

    @classmethod
    def from_config(cls, cfg):
        return {
            'bbone_dims': cfg.BBONE_DIMS,
            'enc_embed_dim': cfg.ENC_EMBED_DIM,
            'dec_dims': cfg.DEC_DIMS,
            'dec_depths': cfg.DEC_DEPTHS,
            'num_heads': cfg.NUM_HEADS,
            'patch_size': cfg.PATCH_SIZE,
            'order': cfg.POINT_ORDER,
            'mlp_ratio': cfg.MLP_RATIO,
            'qkv_bias': cfg.QKV_BIAS,
            'qk_scale': cfg.QK_SCALE,
            'attn_drop': cfg.ATTN_DROP,
            'proj_drop': cfg.PROJ_DROP,
            'drop_path': cfg.DROP_PATH,
            'pre_norm': cfg.PRE_NORM,
            'enable_rpe': cfg.ENABLE_RPE
        }

    def forward(self, feat, point, *args, **kwargs):
        point.feat = feat
        point_f = self.dec(point)

        return point_f
