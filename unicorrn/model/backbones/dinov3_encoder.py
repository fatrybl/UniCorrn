"""DINOv3 image backbone: a drop-in replacement for the CrocoV2 image encoder.

Unlike ``CrocoV2_Encoder`` (which consumes pre-patchified tokens + 2D positions from
``ManyAR_PatchEmbed``), DINOv3 is a complete ViT that patchifies and position-encodes the
raw image internally. The ``consumes_raw_image`` flag tells ``UniCorrn._encode_img`` to
feed the raw pixels here and skip ``patch_embed``. The output matches CrocoV2's contract:
patch tokens ``[B, N, C]`` on the native ``(H/16, W/16)`` grid (prefix CLS/register tokens
stripped), returned as ``(tokens, None, None)``.

The backbone is loaded through ``timm`` (DINOv3 builds as timm's ``Eva`` class) at the
config's ``MODEL_NAME``/``EMBED_DIM`` (e.g. ViT-L/16, embed dim 1024); a mismatch between
the checkpoint's ``embed_dim`` and ``EMBED_DIM`` is rejected. DINOv3 carries 5 prefix
tokens (CLS + 4 registers), stripped here so only the ``(H/16, W/16)`` patch grid remains.
"""

import os
from typing import Self

import timm
import torch.nn as nn

from ...utils.config import configurable
from .build import ENCODER_REGISTRY

_PATCH_SIZE = 16


@ENCODER_REGISTRY.register()
class DinoV3_Encoder(nn.Module):
    """Frozen/trainable DINOv3 ViT that returns patch tokens on the image grid."""

    consumes_raw_image = True

    @configurable
    def __init__(
        self,
        model_name,
        embed_dim,
        cache_dir=None,
        pretrained=True,
        frozen=False,
        **kwargs,
    ):
        super().__init__()
        if cache_dir:
            cache = os.path.abspath(cache_dir)
            os.environ.setdefault("HF_HOME", os.path.join(cache, ".huggingface"))
            os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache, ".huggingface", "hub"))
        backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
            cache_dir=os.path.abspath(cache_dir) if cache_dir else None,
        )
        if backbone.embed_dim != embed_dim:
            raise ValueError(
                f"{model_name} embed_dim {backbone.embed_dim} != config EMBED_DIM {embed_dim}"
            )
        self._backbone = backbone
        self._num_prefix = backbone.num_prefix_tokens
        self.embed_dim = embed_dim
        self._frozen = frozen
        if frozen:
            self.freeze_croco_weights()

    @classmethod
    def from_config(cls, cfg):
        return {
            "model_name": cfg.MODEL_NAME,
            "embed_dim": cfg.EMBED_DIM,
            "cache_dir": cfg.get("CACHE_DIR", None),
            "pretrained": cfg.get("PRETRAINED", True),
            "frozen": cfg.get("FROZEN", False),
        }

    def forward(self, img, *args, **kwargs):
        """Return ``(patch_tokens [B, N, C], None, None)`` on the ``(H/16, W/16)`` grid.

        Raises:
            ValueError: If the input is not a multiple of the patch size, since the token
                grid would then desync from the caller's ``patch_coord_map``.
        """
        _, _, height, width = img.shape
        if height % _PATCH_SIZE or width % _PATCH_SIZE:
            raise ValueError(
                f"DINOv3 input {height}x{width} must be a multiple of patch {_PATCH_SIZE}"
            )
        tokens = self._backbone.forward_features(img)
        return tokens[:, self._num_prefix :, :], None, None

    def train(self, mode: bool = True) -> Self:
        """Set training mode, keeping a frozen backbone in eval.

        ``nn.Module.train`` recurses into children, so without this a later
        ``model.train()`` would undo the eval state ``freeze_croco_weights`` set.

        Args:
            mode: Whether the module enters training mode.
        """
        super().train(mode)
        if self._frozen:
            self._backbone.eval()
        return self

    def freeze_croco_weights(self):
        """Freeze the ViT (name mirrors ``CrocoV2_Encoder`` so UniCorrn's freeze calls work)."""
        self._frozen = True
        self._backbone.eval()
        for param in self._backbone.parameters():
            param.requires_grad = False
