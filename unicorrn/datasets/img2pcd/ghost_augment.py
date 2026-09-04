"""Ghost-copy augmentation for img2pcd samples (distortion_training.md §3).

Duplicates the cloud into parents ``P`` and an equal set of children, screw-displaces
every child by a single random ``G p = R(theta, a) p + t`` and uniformly subsamples the
union ``[P; G P]`` back to the base size ``N``. The screw is uniform within the config's
limits: ``a`` a uniform axis, ``theta ~ U[0, rot_max_deg]``; ``t`` a uniform direction
times ``U[0, trans_max_m]`` -- the same law VAPE's validation draws, so no draw exceeds a
limit. A child inherits its parent's frustum label, pixel and channels — only its
coordinates move — so both matching directions and the frustum labels follow from
gathering parent attributes through the returned provenance.
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
        rot_max_deg: Largest rotation angle in degrees.
        trans_max_m: Largest translation norm in metres.

    The training profile passes all three explicitly (``distortion:`` in
    ``configs/train``); the defaults only mirror it.
    """

    prob: float = 0.5
    rot_max_deg: float = 90.0
    trans_max_m: float = 3.6

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


def random_screw(
    cfg: GhostAugmentConfig, rng: Generator
) -> tuple[np.ndarray, np.ndarray]:
    """One random screw ``(R, t)`` uniform within the config's limits.

    Args:
        cfg: Augmentation knobs carrying the limits.
        rng: Sample-deterministic generator.
    """
    x, y, z = _unit_vector(rng)
    theta = np.radians(rng.uniform(0.0, cfg.rot_max_deg))
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    rotation = (
        np.eye(3) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)
    )
    return rotation, _unit_vector(rng) * rng.uniform(0.0, cfg.trans_max_m)


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
    rotation, translation = random_screw(cfg, rng)
    children = points @ rotation.T + translation
    joint = np.concatenate([points, children], axis=0)
    source = np.tile(np.arange(n), 2)
    keep = rng.choice(2 * n, size=n, replace=False)
    return joint[keep], source[keep]


def _unit_vector(rng: Generator) -> np.ndarray:
    """A direction uniform on the sphere."""
    vector = rng.normal(size=3)
    return vector / (np.linalg.norm(vector) + _AXIS_EPS)
