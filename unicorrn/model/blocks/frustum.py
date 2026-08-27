"""Geometric frustum-membership head for point-cloud -> image queries.

Predicts, per 3D query, the signed distance of its image projection to the frustum
boundary and the in/out logit, reusing the matching decoder's state so membership
shares representation with matching and confidence. For a projection ``u`` and
``q = |u - 1/2| - 1/2`` the exact signed distance to the image rectangle is

    s(u) = -( ||max(q, 0)||_2 + min(max(q_x, q_y), 0) ),

positive inside, zero on the border, negative outside, with magnitude equal to the
Euclidean distance to the rectangle.

``u`` is the decoder's correspondence decode (the affine pseudo-inverse
``InvertibleLinearPositionEmbedding.decode``); the soft-argmax is supplied alongside it
together with their offset, since that discrepancy measures how far the correspondence
has saturated out of frame.

**Measured caveat.** This decode was intended to be unbounded, unlike the soft-argmax,
so that ``s`` could go negative outside the image. It is not: over 12 288 queries of a
trained checkpoint it lay inside the unit rectangle **100%** of the time
(``u`` in [0.113, 0.969], ``v`` in [0.106, 0.975], closest approach to the frame 0.025).
The positional stream accumulates ``A · AbsPE(X_t)`` over layers, and each attention row
is a convex combination of in-image patch encodings, so the affine decode stays inside
their hull by construction. Consequently ``signed_distance(free_projection)`` and
``axis_margins`` are **strictly positive for every query** and carry no sign information:
no projection-derived input can express "outside". The head must infer the negative half
of the target from the appearance stream (attention quality) and the positional stream
(the query's own 3D coordinates) instead, which is why its predicted distance is
uncorrelated with the analytic SDF near the boundary (band R² = 1e-4). Making the target
a metric 3D margin — a function of coordinates the head does receive — is the change that
would make the negative half representable; see ``doc/vape/proposals.md`` P1.

The head regresses a single signed distance ``d`` and derives the in/out logit as
``d / T`` with one global learnable temperature ``T`` (``logit = d * exp(log_scale)``).
The two readouts therefore cannot disagree, and the membership probability
``sigmoid(d / T)`` is a level-set of the predicted distance field: it is ``0.5`` on the
boundary and saturates with distance, which is the desired behaviour at edges. This
mirrors distance-to-logit level-set classification and needs no per-dataset parameter
beyond the single temperature.

``d`` is **detached** on the logit branch. Because ``exp(log_scale) > 0`` the predicted
class is exactly ``sign(d)``, so the classification loss never casts a vote; through a
live branch all it does is pull ``d`` off the regression target, and that pull scales
with ``sigmoid(s* / T) - y``, which is maximal as ``s* -> 0`` — precisely in the band the
head exists to resolve. Detaching leaves the classification loss training the temperature
alone, its one remaining job, and the regression owning ``d`` outright, so the two
objectives share no parameter and cannot conflict.
"""

import torch
from torch import Tensor, nn

_N_MARGINS = 3
_N_OFFSET = 2
_N_STATS = 4
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
        # Attention statistics enter through a zero-initialised projection, so a head
        # trained without them is reproduced exactly at load.
        self.stats = nn.Linear(_N_STATS, feat_dim)
        nn.init.zeros_(self.stats.weight)
        nn.init.zeros_(self.stats.bias)
        # An unbounded projection regressor; its box distance feeds ``d`` through a gain
        # that starts at zero, so the readout is unchanged until the regressor is trained.
        self.projection = nn.Linear(feat_dim, coord_dim)
        self.sdf_gain = nn.Parameter(torch.zeros(()))

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
        stats: Tensor | None = None,
    ) -> Tensor:
        """Predict the in/out logit, signed boundary distance and unbounded projection.

        The logit is the detached distance scaled by the learnable inverse temperature,
        so the first two channels share a sign, the probability is a level-set of the
        distance field, and the classification loss trains only the temperature. The last
        two channels are the regressed image coordinates, supervised on every query in
        front of the camera, inside the frame or not.

        Args:
            appearance: Decoder appearance stream.
            position: Decoder position stream.
            projection: Attention soft-argmax image coordinates.
            free_projection: Correspondence decode of the position stream.
            stats: Attention existence cues ``[..., 4]`` (slot mass, expected logit,
                normalised entropy, border mass); ``None`` when the kernel has none.
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
        hidden = self.trunk[0](features)
        if stats is not None:
            hidden = hidden + self.stats(stats.to(hidden.dtype))
        hidden = self.trunk[1:](hidden)
        unbounded = self.projection(hidden)
        distance = self.distance(hidden) + self.sdf_gain * self.signed_distance(unbounded)
        return torch.cat([distance.detach() * self.log_scale.exp(), distance, unbounded], dim=-1)
