"""Target pose from a generic RGB-D observation. No camera or simulator APIs.

Frozen: depth → cloud → source-square crop → board removal → largest cluster → XY.
Do not retune for a camera model or a populated board.
"""

from __future__ import annotations

import numpy as np

from rgbd import RgbdObservation, unproject
from squares import square_to_world

# One source square. Neighbor square centers sit one square_size away.
_BOARD_LIFT = 0.006
_BOARD_CEIL = 0.055
_CLUSTER_CELL = 0.005
_MIN_POINTS = 20


def _crop_square(pts: np.ndarray, cfg: dict, square: str) -> np.ndarray:
    board = cfg.get("sim", {}).get("board", {})
    half = 0.5 * float(board.get("square_size", 0.0375))
    cx, cy, _ = square_to_world(square, cfg)
    inside = (np.abs(pts[:, 0] - cx) <= half) & (np.abs(pts[:, 1] - cy) <= half)
    return pts[inside]


def _remove_board(pts: np.ndarray, cfg: dict) -> np.ndarray:
    surface = float(cfg.get("sim", {}).get("board", {}).get("surface_z", 0.02))
    above = (pts[:, 2] > surface + _BOARD_LIFT) & (pts[:, 2] < surface + _BOARD_CEIL)
    return pts[above]


def _largest_cluster(pts: np.ndarray, cell: float = _CLUSTER_CELL) -> np.ndarray:
    if len(pts) == 0:
        return pts
    origin = pts[:, :2].min(axis=0)
    ij = np.floor((pts[:, :2] - origin) / cell).astype(np.int32)
    buckets: dict[tuple[int, int], list[int]] = {}
    for k, (i, j) in enumerate(ij):
        buckets.setdefault((int(i), int(j)), []).append(k)
    seen: set[tuple[int, int]] = set()
    best: list[int] = []
    for start in buckets:
        if start in seen:
            continue
        stack = [start]
        comp: list[int] = []
        while stack:
            c = stack.pop()
            if c in seen or c not in buckets:
                continue
            seen.add(c)
            comp.extend(buckets[c])
            i, j = c
            stack.extend(((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)))
        if len(comp) > len(best):
            best = comp
    return pts[best]


def estimate_target_pose(obs: RgbdObservation, cfg: dict, square: str) -> np.ndarray | None:
    """depth → cloud → square crop → board removal → largest cluster → XY.

    `square` is a known prior (the commanded source), not a detection result.
    Returns base XYZ on the board plane, or None.
    """
    pts = unproject(obs)
    if len(pts) == 0:
        return None
    pts = _crop_square(pts, cfg, square)
    pts = _remove_board(pts, cfg)
    pts = _largest_cluster(pts)
    if len(pts) < _MIN_POINTS:
        return None
    surface = float(cfg.get("sim", {}).get("board", {}).get("surface_z", 0.02))
    xy = pts.mean(axis=0)[:2]
    return np.array([xy[0], xy[1], surface], dtype=np.float64)


_TOP_CEIL = 0.075
_TOP_PCT = 95.0


def observe_piece(obs: RgbdObservation, cfg: dict, square: str, grow: float = 1.25) -> dict | None:
    """Largest blob on/near `square`: base XY on the board plane and top height above it.

    `grow` widens the square crop so a piece that slid off-center is still seen.
    """
    pts = unproject(obs)
    if len(pts) == 0:
        return None
    board = cfg.get("sim", {}).get("board", {})
    surface = float(board.get("surface_z", 0.02))
    half = 0.5 * float(board.get("square_size", 0.0375)) * float(grow)
    cx, cy, _ = square_to_world(square, cfg)
    keep = (
        (np.abs(pts[:, 0] - cx) <= half)
        & (np.abs(pts[:, 1] - cy) <= half)
        & (pts[:, 2] > surface + _BOARD_LIFT)
        & (pts[:, 2] < surface + _TOP_CEIL)
    )
    pts = _largest_cluster(pts[keep])
    if len(pts) < _MIN_POINTS:
        return None
    xy = pts.mean(axis=0)[:2]
    top = float(np.percentile(pts[:, 2], _TOP_PCT)) - surface
    return {
        "xy": [float(xy[0]), float(xy[1])],
        "top_h": top,
        "n": int(len(pts)),
        "pts": np.asarray(pts, dtype=np.float64).copy(),
    }
