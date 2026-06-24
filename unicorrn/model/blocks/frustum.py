"""Geometric frustum-membership head for point-cloud -> image queries.

Predicts a per-query binary logit answering whether a 3D point lies inside the
camera viewing cone (geometric frustum, occlusion-agnostic). The head reuses
the matching decoder's internal state so the task shares representation with
matching and confidence:

    logit = MLP([ appearance ; positional_embedding ; predicted_xy ; frustum_margins ])

``predicted_xy`` is the per-layer attention soft-argmax in the **normalised image
frame** ``[0, 1]^2`` (``cartesian_img_coord(norm=True)``); a coordinate outside
``[0, 1]`` means the point projects outside the image — i.e. out of frustum. Raw
``xy`` alone forces the MLP to discover the image-rectangle boundary from two
scalars, so we hand it explicit **signed margins** to that boundary:

    m_x = min(x, 1 - x)        # signed distance to the nearest vertical edge
    m_y = min(y, 1 - y)        # signed distance to the nearest horizontal edge
    m   = min(m_x, m_y)        # signed distance to the image rectangle

Each margin is positive inside the extent, zero on an edge, negative outside, with
magnitude equal to the distance to the nearest border; ``m`` is the geometric
in/out-frustum margin whose boundary gradient supplies the forward (Z) observability
that reprojection lacks. The features are piecewise-linear in ``xy`` (``min`` /
affine), hence differentiable almost everywhere and cheap (per query, not per point).
"""

import torch
from torch import Tensor, nn

from .blocks import Mlp

_N_MARGINS = 3  # m_x, m_y, m derived from the predicted image xy


class FrustumHead(nn.Module):
    """Binary geometric frustum-membership head for point queries."""

    def __init__(self, feat_dim: int, pos_dim: int, coord_dim: int = 2) -> None:
        """Build an MLP over appearance, position, projection and frustum margins."""
        super().__init__()
        self.mlp = Mlp(
            feat_dim + pos_dim + coord_dim + _N_MARGINS,
            hidden_features=feat_dim,
            out_features=1,
        )

    @staticmethod
    def _frustum_margins(projection: Tensor) -> Tensor:
        """Signed distances of the normalised image ``xy`` to the ``[0, 1]`` rectangle."""
        x, y = projection[..., 0:1], projection[..., 1:2]
        m_x = torch.minimum(x, 1.0 - x)
        m_y = torch.minimum(y, 1.0 - y)
        return torch.cat([m_x, m_y, torch.minimum(m_x, m_y)], dim=-1)

    def forward(self, appearance: Tensor, position: Tensor, projection: Tensor) -> Tensor:
        """Predict in/out-frustum logits from appearance, position, projection + margins."""
        margins = self._frustum_margins(projection)
        return self.mlp(torch.cat([appearance, position, projection, margins], dim=-1))
