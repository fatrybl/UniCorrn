import torch.nn as nn

from ...utils.config import configurable
from ..blocks import PointTransformerV3
from .build import ENCODER_REGISTRY


@ENCODER_REGISTRY.register()
class PTv3_Encoder(PointTransformerV3):
    """
    Point Transformer V3 Encoder Wrapper class.

    """

    @configurable
    def __init__(
        self,
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=True,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        project_dim=None,
        replace_bn_norm=False,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super(PTv3_Encoder, self).__init__(
            in_channels=in_channels,
            order=order,
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            pre_norm=pre_norm,
            shuffle_orders=shuffle_orders,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            cls_mode=cls_mode,
            pdnorm_bn=pdnorm_bn,
            pdnorm_ln=pdnorm_ln,
            pdnorm_decouple=pdnorm_decouple,
            pdnorm_adaptive=pdnorm_adaptive,
            pdnorm_affine=pdnorm_affine,
            pdnorm_conditions=pdnorm_conditions,
            replace_bn_norm=replace_bn_norm,
        )

        self.input_project = nn.Identity()
        self.project = nn.Identity()

        if in_channels > 3:
            self.input_project = nn.Linear(3, in_channels, bias=False)

        if project_dim and project_dim != -1:
            self.project = nn.Linear(enc_channels[-1], project_dim, bias=False)

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_channels": cfg.IN_CHANNELS,
            "stride": cfg.STRIDE,
            "enc_depths": cfg.ENC_DEPTHS,
            "enc_channels": cfg.ENC_CHANNELS,
            "enc_num_head": cfg.ENC_NUM_HEADS,
            "enc_patch_size": cfg.ENC_PATCH_SIZE,
            "mlp_ratio": cfg.MLP_RATIO,
            "qkv_bias": cfg.QKV_BIAS,
            "qk_scale": cfg.QK_SCALE,
            "attn_drop": cfg.ATTN_DROP,
            "proj_drop": cfg.PROJ_DROP,
            "drop_path": cfg.DROP_PATH,
            "pre_norm": cfg.PRE_NORM,
            "shuffle_orders": cfg.SHUFFLE_ORDERS,
            "enable_rpe": cfg.ENABLE_RPE,
            "enable_flash": cfg.ENABLE_FLASH,
            "project_dim": cfg.PROJECT_DIM,
            "replace_bn_norm": cfg.REPLACE_BN_NORM,
        }

    def forward(self, pcd, **kwargs):
        pcd["feat"] = self.input_project(pcd["feat"])

        output = super().forward(pcd)
        output.feat = self.project(output.feat)

        return output


@ENCODER_REGISTRY.register()
class PTv3_EncoderV2(PointTransformerV3):
    """
    Point Transformer V3 Encoder Wrapper class.

    """

    @configurable
    def __init__(
        self,
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),  # (3, 3, 3, 6, 3),
        enc_channels=(48, 96, 192, 384, 768),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(48, 96, 192, 384),
        dec_num_head=(2, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        cls_mode=False,
        replace_bn_norm=False,
        project_dim=64,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super(PTv3_EncoderV2, self).__init__(
            in_channels=in_channels,
            order=order,
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=dec_patch_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            pre_norm=pre_norm,
            shuffle_orders=shuffle_orders,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            cls_mode=cls_mode,
            pdnorm_bn=pdnorm_bn,
            pdnorm_ln=pdnorm_ln,
            pdnorm_decouple=pdnorm_decouple,
            pdnorm_adaptive=pdnorm_adaptive,
            pdnorm_affine=pdnorm_affine,
            pdnorm_conditions=pdnorm_conditions,
        )

        self.project = nn.Linear(dec_channels[0], project_dim, bias=True)

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_channels": cfg.IN_CHANNELS,
            "stride": cfg.STRIDE,
            "enc_depths": cfg.ENC_DEPTHS,
            "enc_channels": cfg.ENC_CHANNELS,
            "enc_num_head": cfg.ENC_NUM_HEADS,
            "enc_patch_size": cfg.ENC_PATCH_SIZE,
            "dec_depths": cfg.DEC_DEPTHS,
            "dec_channels": cfg.DEC_CHANNELS,
            "dec_num_head": cfg.DEC_NUM_HEADS,
            "dec_patch_size": cfg.DEC_PATCH_SIZE,
            "mlp_ratio": cfg.MLP_RATIO,
            "qkv_bias": cfg.QKV_BIAS,
            "qk_scale": cfg.QK_SCALE,
            "attn_drop": cfg.ATTN_DROP,
            "proj_drop": cfg.PROJ_DROP,
            "drop_path": cfg.DROP_PATH,
            "pre_norm": cfg.PRE_NORM,
            "shuffle_orders": cfg.SHUFFLE_ORDERS,
            "enable_rpe": cfg.ENABLE_RPE,
            "enable_flash": cfg.ENABLE_FLASH,
            "cls_mode": cfg.CLS_MODE,
            "replace_bn_norm": cfg.REPLACE_BN_NORM,
            "project_dim": cfg.PROJECT_DIM,
        }

    def forward(self, pcd, **kwargs):
        output = super().forward(pcd)

        output.feat = self.project(output.feat)
        return output


@ENCODER_REGISTRY.register()
class PTv3_EncoderV3(PointTransformerV3):
    """
    Point Transformer V3 Encoder Wrapper class.

    """

    @configurable
    def __init__(
        self,
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=True,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        project_dim=None,
        replace_bn_norm=False,
        enable_input_bias=False,
        enable_input_relu=False,
        enable_output_bias=False,
        enable_output_relu=False,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super(PTv3_EncoderV3, self).__init__(
            in_channels=in_channels,
            order=order,
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            pre_norm=pre_norm,
            shuffle_orders=shuffle_orders,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            cls_mode=cls_mode,
            pdnorm_bn=pdnorm_bn,
            pdnorm_ln=pdnorm_ln,
            pdnorm_decouple=pdnorm_decouple,
            pdnorm_adaptive=pdnorm_adaptive,
            pdnorm_affine=pdnorm_affine,
            pdnorm_conditions=pdnorm_conditions,
            replace_bn_norm=replace_bn_norm,
        )

        self.input_project = nn.Identity()
        self.project = nn.Identity()

        if in_channels > 3:
            layers = [nn.Linear(3, in_channels, bias=enable_input_bias)]
            if enable_input_relu:
                layers.append(nn.ReLU())

            self.input_project = nn.Sequential(*layers)

        if project_dim:
            layers = [nn.Linear(enc_channels[-1], project_dim, bias=enable_output_bias)]
            if enable_output_bias:
                layers.append(nn.ReLU())

            self.project = nn.Sequential(*layers)

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_channels": cfg.IN_CHANNELS,
            "stride": cfg.STRIDE,
            "enc_depths": cfg.ENC_DEPTHS,
            "enc_channels": cfg.ENC_CHANNELS,
            "enc_num_head": cfg.ENC_NUM_HEADS,
            "enc_patch_size": cfg.ENC_PATCH_SIZE,
            "mlp_ratio": cfg.MLP_RATIO,
            "qkv_bias": cfg.QKV_BIAS,
            "qk_scale": cfg.QK_SCALE,
            "attn_drop": cfg.ATTN_DROP,
            "proj_drop": cfg.PROJ_DROP,
            "drop_path": cfg.DROP_PATH,
            "pre_norm": cfg.PRE_NORM,
            "shuffle_orders": cfg.SHUFFLE_ORDERS,
            "enable_rpe": cfg.ENABLE_RPE,
            "enable_flash": cfg.ENABLE_FLASH,
            "project_dim": cfg.PROJECT_DIM,
            "replace_bn_norm": cfg.REPLACE_BN_NORM,
            "enable_input_bias": cfg.ENABLE_INPUT_BIAS,
            "enable_input_relu": cfg.ENABLE_INPUT_RELU,
            "enable_output_bias": cfg.ENABLE_OUTPUT_BIAS,
            "enable_output_relu": cfg.ENABLE_OUTPUT_RELU,
        }

    def forward(self, pcd, **kwargs):
        pcd["feat"] = self.input_project(pcd["feat"])

        output = super().forward(pcd)
        output.feat = self.project(output.feat)

        return output
