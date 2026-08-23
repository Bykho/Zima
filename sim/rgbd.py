"""Generic RGB-D observation. Camera-agnostic; no MuJoCo or D405 types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RgbdObservation:
    """One RGB-D frame in OpenCV camera convention.

    Camera frame: +X right, +Y down, +Z forward. Depth is Z in meters.
    R, t take camera-frame points to world:  X_w = R @ X_c + t.
    near/far are the backend's valid-depth interval, not perception knobs.
    """

    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    near: float = 0.07
    far: float = 0.50


def K_from_fovy(width: int, height: int, fovy_deg: float) -> np.ndarray:
    fy = 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2.0)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    return np.array([[fy, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def unproject(obs: RgbdObservation) -> np.ndarray:
    """Depth image → world XYZ. Drops non-finite and out-of-range pixels."""
    depth = np.asarray(obs.depth, dtype=np.float64)
    h, w = depth.shape
    fx, fy = float(obs.K[0, 0]), float(obs.K[1, 1])
    cx, cy = float(obs.K[0, 2]), float(obs.K[1, 2])
    vs, us = np.indices((h, w))
    z = depth
    keep = np.isfinite(z) & (z > obs.near) & (z < obs.far)
    if not np.any(keep):
        return np.zeros((0, 3), dtype=np.float64)
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=-1)[keep]
    return pts_cam @ np.asarray(obs.R, dtype=np.float64).T + np.asarray(obs.t, dtype=np.float64)
