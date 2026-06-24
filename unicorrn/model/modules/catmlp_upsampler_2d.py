import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import DECODER_REGISTRY
from ..blocks import Mlp
from ...utils.config import configurable


@DECODER_REGISTRY.register()
class CatMLPUpsampler2D(nn.Module):
    @configurable
    def __init__(self,
                 bbone_channel,
                 enc_channel,
                 patch_size=16,
                 upscale_factor=1,
                 dec_channel=16,
                 mlp_ratio=4.,
                 **kwargs):
        super().__init__()
        self.dec_channel = dec_channel

        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2 and isinstance(patch_size[0], int) and isinstance(
                patch_size[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
            assert patch_size[0] == patch_size[1], "Error, non square patches not managed"
            patch_size = patch_size[0]
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor

        idim = bbone_channel + enc_channel
        self.head_local_features = Mlp(in_features=idim,
                                       hidden_features=int(mlp_ratio * idim),
                                       out_features=self.dec_channel * upscale_factor ** 2)

    @property
    def upscale_patch_size(self):
        return self.patch_size // self.upscale_factor

    @classmethod
    def from_config(cls, cfg):
        return {
            'bbone_channel': cfg.BBONE_CHANNEL,
            'enc_channel': cfg.ENC_CHANNEL,
            'upscale_factor': cfg.UPSCALE_FACTOR,
            'dec_channel': cfg.DEC_CHANNEL,
            'mlp_ratio': cfg.MLP_RATIO
        }

    @torch.autocast('cuda', enabled=False)
    def forward(self, bbone_out, enc_out, patch_shape):
        cat_output = torch.cat([bbone_out, enc_out], dim=-1)  # concatenate
        patch_W, patch_H = patch_shape
        B, S, D = cat_output.shape

        # extract local_features
        local_features = self.head_local_features(cat_output)  # B,S,D
        local_features = local_features.transpose(-1, -2).view(B, -1, patch_H, patch_W)
        local_features = F.pixel_shuffle(local_features, self.upscale_factor)  # B,d,H,W

        ret_dict = {
            'H': local_features.size(2),
            'W': local_features.size(3),
            'patch_size': self.patch_size // self.upscale_factor,
            'feat_f': local_features.permute(0, 2, 3, 1).reshape(B, -1, self.dec_channel)
        }

        return ret_dict
