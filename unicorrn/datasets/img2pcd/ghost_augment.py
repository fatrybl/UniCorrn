"""Ghost-copy augmentation for img2pcd samples (distortion_training.md §3).

Builds the joint cloud ``[P; G·P_sub]`` where a random screw
``G p = R(θ, â) p + t`` displaces a copied subset: ``â`` is a uniform axis,
``θ ~ N(0, σ_rot²)`` (degrees) and ``t ~ N(0, σ_trans² I₃)`` (metres). Ghost
queries carry the copy positions ``G·pₙ`` supervised by the base twin's pixel
``uₙ``; clean draws (probability ``1 − prob``) leave the cloud untouched and fill
the query slots with resampled base correspondence pairs. The copied subset is the
full cloud or a ``region_frac·r`` ball, optionally Bernoulli-thinned by
``copy_frac`` with every selected twin retained.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields

import numpy as np

_AXIS_EPS = 1e-12


@dataclass(frozen=True)
class GhostAugmentConfig:
    """Knobs for ghost-copy augmentation (distortion_training.md §8).

    Attributes:
        prob: Probability a sample is ghosted.
        rot_std_deg: Std of the screw rotation angle in degrees.
        trans_std: Per-axis translation std in metres.
        partial_prob: Probability of a region copy instead of a full copy.
        region_frac: Region ball radius as a fraction of the cloud radius.
        copy_frac: Bernoulli keep-fraction thinning the copied points.
        num_queries: Ghost query slots emitted per sample.
    """

    prob: float = 0.35
    rot_std_deg: float = 30.0
    trans_std: float = 1.0
    partial_prob: float = 0.5
    region_frac: float = 0.35
    copy_frac: float = 1.0
    num_queries: int = 256

    @classmethod
    def from_mapping(cls, params: Mapping[str, float | int]) -> "GhostAugmentConfig":
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


@dataclass
class GhostSample:
    """A ghosted (or clean) img2pcd sample.

    Attributes:
        points: Joint cloud ``[P; G·P_sub]`` (the base ``P`` on clean draws).
        queries: Ghost query positions ``[Q, 3]`` (copies or base points).
        targets: Pixel targets ``[Q, 2]`` — the base twins' pixels.
        flag: ``1.0`` when the cloud carries twins, else ``0.0``.
        copy_indices: Base-row indices of the appended copies (empty on clean
            draws), so per-point attributes (e.g. intensity) can follow them.
    """

    points: np.ndarray
    queries: np.ndarray
    targets: np.ndarray
    flag: float
    copy_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))


def _rotation(rot_std_deg: float) -> np.ndarray:
    """Rodrigues rotation about a uniform axis, angle ``N(0, rot_std_deg²)``."""
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis) + _AXIS_EPS
    theta = np.radians(np.random.normal(0.0, rot_std_deg))
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def _apply(rot: np.ndarray, trans: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply the screw ``rot · pts + trans`` row-wise."""
    return pts @ rot.T + trans


def _copy_subset(points: np.ndarray, cfg: GhostAugmentConfig) -> np.ndarray:
    """Base indices to copy: a ``region_frac`` ball with prob ``partial_prob``, else all."""
    n = points.shape[0]
    if np.random.uniform() >= cfg.partial_prob:
        return np.arange(n)
    centroid = points.mean(axis=0)
    radius = float(np.linalg.norm(points - centroid, axis=1).max())
    center = points[np.random.randint(n)]
    within = np.linalg.norm(points - center, axis=1) <= cfg.region_frac * radius
    return np.nonzero(within)[0]


def _resample(pool: np.ndarray, num: int) -> np.ndarray:
    """Draw ``num`` rows from ``pool`` (with replacement only when too few)."""
    return pool[np.random.choice(pool.shape[0], size=num, replace=pool.shape[0] < num)]


def augment_sample(
    points: np.ndarray,
    corr_points: np.ndarray,
    corr_indices: np.ndarray,
    corr_pixels: np.ndarray,
    cfg: GhostAugmentConfig,
) -> GhostSample:
    """Ghost a sample or emit a clean one with resampled base pairs.

    Args:
        points: Base cloud ``[N, 3]``.
        corr_points: Ground-truth correspondence points ``[C, 3]``.
        corr_indices: Their row indices into ``points`` ``[C]``.
        corr_pixels: Their pixel targets ``[C, 2]``.
        cfg: Augmentation knobs.

    Raises:
        ValueError: If no correspondences are supplied.
    """
    if corr_indices.shape[0] == 0:
        raise ValueError("Ghosting needs at least one correspondence")
    if np.random.uniform() >= cfg.prob:
        pick = _resample(np.arange(corr_indices.shape[0]), cfg.num_queries)
        return GhostSample(points, corr_points[pick], corr_pixels[pick], 0.0)

    rot = _rotation(cfg.rot_std_deg)
    trans = np.random.normal(0.0, cfg.trans_std, size=3)
    sub = _copy_subset(points, cfg)
    eligible = np.nonzero(np.isin(corr_indices, sub))[0]
    if eligible.shape[0] == 0:
        eligible = np.arange(corr_indices.shape[0])
    pick = _resample(eligible, cfg.num_queries)
    twins = corr_indices[pick]
    sub = np.union1d(sub, twins)
    if cfg.copy_frac < 1.0:
        keep = (np.random.uniform(size=sub.shape[0]) < cfg.copy_frac) | np.isin(sub, twins)
        sub = sub[keep]
    joint = np.concatenate([points, _apply(rot, trans, points[sub])], axis=0)
    return GhostSample(joint, _apply(rot, trans, corr_points[pick]), corr_pixels[pick], 1.0, sub)
