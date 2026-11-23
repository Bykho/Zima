"""Which square each piece body sits on: one snap to the nearest square centre."""

from __future__ import annotations

import numpy as np

from scenario import is_piece_body
from squares import FILES, shift_square, square_to_world


def neighbor_squares(source: str) -> list[str]:
    out = []
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            sq = shift_square(source, df, dr)
            if sq:
                out.append(sq)
    return out


def occupied_squares(sim, cfg: dict) -> dict[str, str]:
    """square -> body for anything sitting on the board."""
    import mujoco

    size = float(cfg.get("sim", {}).get("board", {}).get("square_size", 0.0375))
    found: dict[str, str] = {}
    for i in range(sim.model.nbody):
        name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, i)
        if not name or not is_piece_body(name):
            continue
        if float(sim.data.xpos[i][2]) < 0.01:
            continue
        sq = _snap_square(cfg, sim.data.xpos[i][:2], size)
        if sq:
            found[sq] = name
    return found


def neighbor_occupancy(sim, cfg: dict, source: str) -> dict[str, str]:
    """Occupied neighbor squares of `source` -> body name."""
    occ = occupied_squares(sim, cfg)
    want = set(neighbor_squares(source))
    return {sq: name for sq, name in occ.items() if sq in want}


def square_of_body(sim, cfg: dict, name: str) -> str | None:
    """Occupancy square for a free body, or None if off-board."""
    if not name:
        return None
    occ = occupied_squares(sim, cfg)
    for sq, body in occ.items():
        if body == name:
            return sq
    return None


_SQUARE_GRID: dict[tuple, tuple[list[str], np.ndarray]] = {}


def _square_grid(cfg: dict) -> tuple[list[str], np.ndarray]:
    """All 64 square centers as one array; keyed on the board geometry values."""
    board = cfg.get("sim", {}).get("board", {})
    key = (
        tuple(float(v) for v in board.get("origin", [0.22, -0.10, 0.0])),
        float(board.get("square_size", 0.0375)),
        float(board.get("yaw", 0.0)),
    )
    hit = _SQUARE_GRID.get(key)
    if hit is None:
        names = [f"{f}{r}" for f in FILES for r in range(1, 9)]
        pts = np.array([square_to_world(sq, cfg)[:2] for sq in names], dtype=float)
        hit = (names, pts)
        _SQUARE_GRID[key] = hit
    return hit


def _snap_square(cfg: dict, xy, size: float) -> str | None:
    names, pts = _square_grid(cfg)
    d = np.hypot(pts[:, 0] - float(xy[0]), pts[:, 1] - float(xy[1]))
    i = int(np.argmin(d))
    return names[i] if d[i] < 0.55 * size else None
