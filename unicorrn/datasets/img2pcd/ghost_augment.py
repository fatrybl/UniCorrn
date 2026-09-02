"""Ghost-copy augmentation for img2pcd samples (distortion_training.md §3).

Duplicates the cloud into parents ``P`` and an equal set of children, screw-displaces
every child by a single random ``G p = R(theta, a) p + t`` (``a`` a uniform axis,
``theta ~ N(0, sigma_rot^2)`` degrees, ``t ~ N(0, sigma_trans^2 I)`` metres), then
uniformly subsamples the union ``[P; G P]`` back to the base size ``N``. A child inherits
its parent's frustum label, pixel and channels — only its coordinates move — so both
matching directions and the frustum labels follow from gathering parent attributes
through the returned provenance.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Self

import numpy as np
from numpy.random import Generator

_AXIS_EPS = 1e-12


@dataclass(frozen=True)
class GhostAugmentConfig:
    """Knobs for ghost-copy augmentation.

    Attributes:
        prob: Probability a sample is ghosted.
        rot_std_deg: Std of the screw rotation angle in degrees.
        trans_std: Per-axis translation std in metres.

    Defaults mirror ``vape/models/vape/distortion.py`` — the single place the level is
    set; the training profile passes all three explicitly.
    """

    prob: float = 0.5
    rot_std_deg: float = 30.0
    trans_std: float = 1.2

    @classmethod
    def from_mapping(cls, params: Mapping[str, float]) -> Self:
        """Build a config from a params mapping, rejecting unknown keys.

        Args:
            params: Field-name to value mapping.

        Raises:
            ValueError: If ``params`` holds a key that is not a config field.
        """
        unknown = set(params) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown ghost_params: {sorted(unknown)}")
        return cls(**params)


def _rotation(rot_std_deg: float, rng: Generator) -> np.ndarray:
    """Rodrigues rotation about a uniform axis, angle ``N(0, rot_std_deg^2)``."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + _AXIS_EPS
    theta = np.radians(rng.normal(0.0, rot_std_deg))
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)


def augment_sample(
    points: np.ndarray, cfg: GhostAugmentConfig, rng: Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Displace a full copy of the cloud and subsample the union back to ``N``.

    Args:
        points: Base cloud ``[N, 3]``.
        cfg: Augmentation knobs.
        rng: Sample-deterministic generator.

    Returns:
        The retained ``[N, 3]`` mixture of parents and displaced children and their
        base-row provenance ``[N]`` (each child maps to its parent).
    """
    n = points.shape[0]
    rotation = _rotation(cfg.rot_std_deg, rng)
    children = points @ rotation.T + rng.normal(0.0, cfg.trans_std, size=3)
    joint = np.concatenate([points, children], axis=0)
    source = np.tile(np.arange(n), 2)
    keep = rng.choice(2 * n, size=n, replace=False)
    return joint[keep], source[keep]
