"""Geometric frustum-membership head for point-cloud -> image queries.

Predicts a per-query binary logit answering whether a 3D point lies inside the
camera viewing cone (geometric frustum, occlusion-agnostic). The head reuses
the matching decoder's internal state so the task shares representation with
matching and confidence:

    logit = MLP([ appearance ; positional_embedding ; predicted_image_xy ])

The predicted image coordinate is the per-layer attention soft-argmax already
produced by the dual-stream decoder, so a point whose projection falls inside
the image extent is, by construction, a strong in-frustum cue.
"""

import torch
from torch import Tensor, nn

from .blocks import Mlp


class FrustumHead(nn.Module):
    """Binary geometric frustum-membership head for point queries."""

    def __init__(self, feat_dim: int, pos_dim: int, coord_dim: int = 2) -> None:
        """Build an MLP head over concatenated appearance, position and projection."""
        super().__init__()
        self.mlp = Mlp(
            feat_dim + pos_dim + coord_dim,
            hidden_features=feat_dim,
            out_features=1,
        )

    def forward(
        self, appearance: Tensor, position: Tensor, projection: Tensor
    ) -> Tensor:
        """Predict in/out-frustum logits from appearance, position and projection."""
        return self.mlp(torch.cat([appearance, position, projection], dim=-1))
