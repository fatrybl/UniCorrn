"""Dual image backbone: CroCo v2 and DINOv3 tokens fused to the CroCo width.

CroCo v2 is pretrained by cross-view completion, so its features carry a two-view
correspondence prior; DINOv3 is pretrained by single-image self-distillation and carries
stronger semantics. Both are ViTs at patch 16, so on an ``H x W`` input they emit
``(H/16) * (W/16)`` tokens in the same row-major order and concatenate token-wise.

The concatenation ``[B, N, C_croco + C_dino]`` is projected back to ``C_croco`` by one
linear layer initialised as the block matrix ``[I | 0]`` with zero bias, so

    fuse([croco ; norm(dino)]) = croco

exactly at step zero: the module starts numerically identical to ``CrocoV2_Encoder`` and
the DINOv3 contribution grows as training moves the right-hand block. Holding the output
width fixed also keeps every downstream module's shape unchanged. Only the DINOv3 branch
is layer-normalised, since normalising CroCo would break that identity.

``consumes_raw_image`` tells ``UniCorrn._encode_img`` to hand over raw pixels; the patch
embedding CroCo needs is owned here instead of by the model.
"""

from collections.abc import Sequence

import torch
from numpy import ndarray
from torch import Tensor, nn

from ...utils.config import CfgNode, configurable
from ..embedder import ManyAR_PatchEmbed
from .build import ENCODER_REGISTRY, build_encoder

_IMAGE_CHANNELS = 3


@ENCODER_REGISTRY.register()
class DualImage_Encoder(nn.Module):
    """CroCo v2 + DINOv3 image backbone with an identity-initialised fusion."""

    consumes_raw_image = True

    @configurable
    def __init__(
        self,
        croco_cfg: CfgNode,
        dino_cfg: CfgNode,
        img_size: Sequence[int],
        patch_size: int,
        pos_embed: ndarray | None = None,
        rope: nn.Module | None = None,
        **kwargs,
    ) -> None:
        """Build both branches, the DINOv3 norm and the identity-initialised fusion.

        Args:
            croco_cfg: Config subtree for the CroCo v2 branch.
            dino_cfg: Config subtree for the DINOv3 branch.
            img_size: Model input resolution the patch embedding is built for.
            patch_size: Patch side length, shared by both branches.
            pos_embed: Optional absolute position embedding for the CroCo branch.
            rope: Optional rotary position embedding for the CroCo branch.
        """
        super().__init__()
        croco_dim = croco_cfg.EMBED_DIM
        self.patch_embed = ManyAR_PatchEmbed(
            img_size, patch_size, _IMAGE_CHANNELS, croco_dim, upscale=False
        )
        self.croco = build_encoder(croco_cfg, pos_embed=pos_embed, rope=rope)
        self.dino = build_encoder(dino_cfg)
        self.dino_norm = nn.LayerNorm(self.dino.embed_dim)
        self.fuse = nn.Linear(croco_dim + self.dino.embed_dim, croco_dim)
        self.embed_dim = croco_dim
        self._reset_fusion(croco_dim)

    @classmethod
    def from_config(cls, cfg):
        """Map the image-backbone config subtree to constructor arguments."""
        return {
            "croco_cfg": cfg.CROCO,
            "dino_cfg": cfg.DINO,
            "img_size": cfg.IMG_SIZE,
            "patch_size": cfg.PATCH_SIZE,
        }

    def forward(self, img: Tensor, *args, **kwargs) -> tuple[Tensor, None, None]:
        """Return ``(fused tokens, None, None)`` on the ``(H/16, W/16)`` grid.

        Args:
            img: Raw image batch ``[B, 3, H, W]``.
        """
        batch = img.shape[0]
        true_shape = torch.tensor(img.shape[-2:], device=img.device)[None].repeat(batch, 1)
        patches, pos = self.patch_embed(img, true_shape)
        croco_tokens, _, _ = self.croco(patches, pos)
        dino_tokens, _, _ = self.dino(img)
        fused = torch.cat([croco_tokens, self.dino_norm(dino_tokens)], dim=-1)
        return self.fuse(fused), None, None

    def freeze_croco_weights(self) -> None:
        """Freeze both branches and the fusion (name mirrors ``CrocoV2_Encoder``)."""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def _reset_fusion(self, croco_dim: int) -> None:
        """Initialise the fusion as ``[I | 0]`` so the output equals the CroCo branch.

        Args:
            croco_dim: Width of the CroCo branch, and of the fused output.
        """
        with torch.no_grad():
            self.fuse.weight.zero_()
            self.fuse.weight[:, :croco_dim].copy_(torch.eye(croco_dim))
            self.fuse.bias.zero_()
