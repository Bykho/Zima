"""Board heightmap: the controller's picture of what is on the table.

2.5D because the only sensor (wrist D405) looks down: each cell keeps the
highest surface seen above the board plane. Cells no view has covered stay
unknown and read as tall, so an unseen gap never looks free. One scan builds
one map; a later scan replaces it rather than max-merging into a stale one.

Built from RgbdObservation only. `truth_heightmap` ray-casts the simulator
to score the estimate and must not feed control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from squares import FILES

CELL_M = 0.0015
MARGIN_M = 0.02
# Tallest piece is 55 mm. Anything higher at scan height is the arm itself.
Z_MAX_M = 0.075
UNKNOWN_H_M = Z_MAX_M
OCCUPIED_H_M = 0.008


def _board(cfg: dict) -> tuple[np.ndarray, float, float, float]:
    b = cfg.get("sim", {}).get("board", {})
    origin = np.asarray(b.get("origin", [0.22, -0.10, 0.0])[:2], dtype=float)
    return origin, float(b.get("square_size", 0.0375)), float(b.get("yaw", 0.0)), float(b.get("surface_z", 0.02))


@dataclass
class HeightMap:
    origin: np.ndarray  # world XY of the (0, 0) cell corner
    cell: float
    surface_z: float
    height: np.ndarray  # (nx, ny) meters above the board plane; nan = never seen
    hits: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(s) for s in self.height.shape)

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        nx, ny = self.shape
        xs = self.origin[0] + (np.arange(nx) + 0.5) * self.cell
        ys = self.origin[1] + (np.arange(ny) + 0.5) * self.cell
        return np.meshgrid(xs, ys, indexing="ij")

    def known(self) -> np.ndarray:
        return ~np.isnan(self.height)

    def filled(self) -> np.ndarray:
        return np.where(self.known(), self.height, UNKNOWN_H_M)

    def square_cells(self, cfg: dict) -> np.ndarray:
        """(nx, ny) int: 8*file + rank-1 for cells inside the board, -1 outside."""
        origin, size, yaw, _ = _board(cfg)
        X, Y = self.centers()
        dx, dy = X - origin[0], Y - origin[1]
        c, s = math.cos(yaw), math.sin(yaw)
        f = np.floor((c * dx + s * dy) / size).astype(int)
        r = np.floor((-s * dx + c * dy) / size).astype(int)
        inside = (f >= 0) & (f < 8) & (r >= 0) & (r < 8)
        return np.where(inside, 8 * f + r, -1)

    def square_heights(self, cfg: dict) -> dict[str, tuple[float, float]]:
        """square -> (tallest filled height, fraction of cells seen)."""
        idx = self.square_cells(cfg)
        filled, known = self.filled(), self.known()
        out = {}
        for k in range(64):
            m = idx == k
            if not m.any():
                continue
            out[f"{FILES[k // 8]}{k % 8 + 1}"] = (float(filled[m].max()), float(known[m].mean()))
        return out

    def occupied_squares(self, cfg: dict, min_h: float = OCCUPIED_H_M) -> set[str]:
        return {sq for sq, (h, _) in self.square_heights(cfg).items() if h >= min_h}


def empty_heightmap(cfg: dict, cell: float = CELL_M, margin: float = MARGIN_M) -> HeightMap:
    origin, size, yaw, surface = _board(cfg)
    c, s = math.cos(yaw), math.sin(yaw)
    span = 8.0 * size
    corners = np.array([[0, 0], [span, 0], [0, span], [span, span]], dtype=float)
    world = origin + corners @ np.array([[c, s], [-s, c]])
    lo = world.min(axis=0) - margin
    hi = world.max(axis=0) + margin
    n = np.ceil((hi - lo) / cell).astype(int)
    return HeightMap(
        origin=lo,
        cell=float(cell),
        surface_z=surface,
        height=np.full((int(n[0]), int(n[1])), np.nan),
        hits=np.zeros((int(n[0]), int(n[1])), dtype=np.int32),
    )


def integrate_points(hm: HeightMap, pts: np.ndarray) -> int:
    """Max-fuse world points into the map. Returns points used."""
    if len(pts) == 0:
        return 0
    h = pts[:, 2] - hm.surface_z
    i = np.floor((pts[:, 0] - hm.origin[0]) / hm.cell).astype(int)
    j = np.floor((pts[:, 1] - hm.origin[1]) / hm.cell).astype(int)
    nx, ny = hm.shape
    keep = (h > -0.01) & (h < Z_MAX_M) & (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    if not np.any(keep):
        return 0
    i, j, h = i[keep], j[keep], np.clip(h[keep], 0.0, None)
    tmp = np.full(hm.shape, -np.inf)
    np.maximum.at(tmp, (i, j), h)
    seen = np.isfinite(tmp)
    hm.height[seen] = np.fmax(hm.height[seen], tmp[seen])
    np.add.at(hm.hits, (i, j), 1)
    return int(keep.sum())


def truth_heightmap(sim, like: HeightMap) -> HeightMap:
    """Ray-cast static scene geoms (group 0: board, pieces, trays). Evaluation only."""
    import mujoco

    X, Y = like.centers()
    top = like.surface_z + 0.3
    group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    geomid = np.array([-1], dtype=np.int32)
    down = np.array([0.0, 0.0, -1.0])
    height = np.full(like.shape, np.nan)
    mujoco.mj_forward(sim.model, sim.data)
    for idx in np.ndindex(like.shape):
        d = mujoco.mj_ray(
            sim.model, sim.data, np.array([X[idx], Y[idx], top]), down, group, 1, -1, geomid
        )
        if d >= 0:
            height[idx] = max(0.0, top - d - like.surface_z)
    return HeightMap(like.origin.copy(), like.cell, like.surface_z, height, np.ones(like.shape, dtype=np.int32))
