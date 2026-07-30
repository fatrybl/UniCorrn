"""Geometric frustum-membership head for point-cloud -> image queries.

Predicts, per 3D query, the signed distance of its image projection to the frustum
boundary and the in/out logit, reusing the matching decoder's state so membership
shares representation with matching and confidence. For a projection ``u`` and
``q = |u - 1/2| - 1/2`` the exact signed distance to the image rectangle is

    s(u) = -( ||max(q, 0)||_2 + min(max(q_x, q_y), 0) ),

positive inside, zero on the border, negative outside, with magnitude equal to the
Euclidean distance to the rectangle.

``u`` is the decoder's **unbounded** correspondence decode (the affine pseudo-inverse
``InvertibleLinearPositionEmbedding.decode``), not the attention soft-argmax: the
soft-argmax is a convex combination of in-image patch centres, so it is confined to
their hull, which makes ``s`` strictly positive and flat outside the image — exactly
where the boundary gradient is needed. The soft-argmax is still supplied together
with its offset from the free decode, since that discrepancy measures how far the
correspondence has saturated out of frame.

The head regresses a single signed distance ``d`` and derives the in/out logit as
``d / T`` with one global learnable temperature ``T`` (``logit = d * exp(log_scale)``).
The two readouts therefore cannot disagree, and the membership probability
``sigmoid(d / T)`` is a calibrated level-set of the predicted distance field: it is
``0.5`` on the boundary and saturates with distance, which is the desired behaviour at
edges. This mirrors distance-to-logit level-set classification and needs no per-dataset
parameter beyond the single temperature.
"""

import torch
from torch import Tensor, nn

_N_MARGINS = 3
_N_OFFSET = 2
_BOX_CENTRE = 0.5
_NORM_EPS = 1e-12
_INIT_LOG_SCALE = 0.0


class FrustumHead(nn.Module):
    """Signed-distance and in/out logit head for point queries."""

    def __init__(
        self, feat_dim: int, pos_dim: int, coord_dim: int = 2, hidden_depth: int = 2
    ) -> None:
        """Build the trunk over appearance, position, projections and margins.

        Args:
            feat_dim: Width of the decoder appearance stream.
            pos_dim: Width of the decoder position stream.
            coord_dim: Image-coordinate dimensionality.
            hidden_depth: Number of hidden blocks in the trunk.
        """
        super().__init__()
        in_dim = feat_dim + pos_dim + 2 * coord_dim + _N_MARGINS + _N_OFFSET
        layers: list[nn.Module] = []
        for _ in range(hidden_depth):
            layers += [nn.Linear(in_dim, feat_dim), nn.LayerNorm(feat_dim), nn.GELU()]
            in_dim = feat_dim
        self.trunk = nn.Sequential(*layers)
        self.distance = nn.Linear(feat_dim, 1)
        self.log_scale = nn.Parameter(torch.full((), _INIT_LOG_SCALE))

    @staticmethod
    def signed_distance(projection: Tensor) -> Tensor:
        """Exact signed distance of a projection to the unit image rectangle.

        Args:
            projection: Normalised image coordinates.
        """
        q = torch.abs(projection - _BOX_CENTRE) - _BOX_CENTRE
        relative = torch.clamp(q, min=0.0)
        outside = torch.sqrt((relative * relative).sum(-1, keepdim=True) + _NORM_EPS)
        inside = torch.clamp(q.amax(dim=-1, keepdim=True), max=0.0)
        return -(outside + inside)

    def forward(
        self,
        appearance: Tensor,
        position: Tensor,
        projection: Tensor,
        free_projection: Tensor,
    ) -> Tensor:
        """Predict the in/out logit and signed boundary distance per query.

        The logit is the distance scaled by the learnable inverse temperature, so the
        two returned channels share a sign and the probability is a level-set of the
        distance field.

        Args:
            appearance: Decoder appearance stream.
            position: Decoder position stream.
            projection: Bounded attention soft-argmax image coordinates.
            free_projection: Unbounded correspondence decode of the position stream.
        """
        axis_margins = _BOX_CENTRE - torch.abs(free_projection - _BOX_CENTRE)
        features = torch.cat(
            [
                appearance,
                position,
                projection,
                free_projection,
                axis_margins,
                self.signed_distance(free_projection),
                free_projection - projection,
            ],
            dim=-1,
        )
        distance = self.distance(self.trunk(features))
        return torch.cat([distance * self.log_scale.exp(), distance], dim=-1)
