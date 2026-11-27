"""Upright piece position from a depth cloud by fitting the piece's known shape.

The mean of the visible points is biased: a top-down view sees the cap and the
camera-facing side, a square crop clips a piece that sits off-centre, and an
asymmetric piece (the knight's head) moves the centroid off its base. Fitting
the piece model (the robot's own scene model, static geometry) to the points
removes all three: every observed point should lie on the model surface, so
the pose minimises the points' distance to it. Occluded surfaces cost nothing.
"""

from __future__ import annotations

import mujoco
import numpy as np

from rgbd import RgbdObservation, unproject

_BOARD_LIFT = 0.006  # same board rejection as perception.observe_piece
_TOP_CEIL = 0.075
_WINDOW = 0.024  # crop half-width around the seed: piece radius plus a slid-piece margin
_INLIER_M = 0.003
_MIN_INLIER_FRAC = 0.6
_MIN_POINTS = 20
_ITERS = 12


def piece_shape(sim, body: str) -> list[tuple]:
    """(type, size, pos, R) for each geom of `body`, in the body frame."""
    m = sim.model
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        return []
    out = []
    for g in np.flatnonzero(m.geom_bodyid == bid):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        out.append((int(m.geom_type[g]), m.geom_size[g].copy(), m.geom_pos[g].copy(), R.reshape(3, 3)))
    return out


def _sdf(q: np.ndarray, shape: list[tuple]) -> np.ndarray:
    best = np.full(len(q), np.inf)
    for typ, size, pos, R in shape:
        p = (q - pos) @ R
        if typ == mujoco.mjtGeom.mjGEOM_BOX:
            d = np.abs(p) - size[:3]
            dist = np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)
        elif typ == mujoco.mjtGeom.mjGEOM_CYLINDER:
            d = np.stack([np.hypot(p[:, 0], p[:, 1]) - size[0], np.abs(p[:, 2]) - size[1]], axis=1)
            dist = np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)
        elif typ == mujoco.mjtGeom.mjGEOM_SPHERE:
            dist = np.linalg.norm(p, axis=1) - size[0]
        else:
            continue
        best = np.minimum(best, dist)
    return best


def _residual(pts: np.ndarray, shape: list[tuple], x: np.ndarray, surface: float) -> np.ndarray:
    c, s = np.cos(x[2]), np.sin(x[2])
    d = pts - np.array([x[0], x[1], surface])
    q = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]], axis=1)
    return _sdf(q, shape)


def _fit(pts, shape, x0, surface):
    x = np.asarray(x0, dtype=float)
    eps = np.array([1e-4, 1e-4, 1e-3])
    for _ in range(_ITERS):
        r = _residual(pts, shape, x, surface)
        w = np.abs(r) < max(_INLIER_M, 3.0 * float(np.median(np.abs(r))))
        if w.sum() < _MIN_POINTS:
            break
        J = np.stack([(_residual(pts[w], shape, x + e, surface) - r[w]) / e[i]
                      for i, e in enumerate(np.diag(eps))], axis=1)
        H = J.T @ J
        step = -np.linalg.solve(H + 1e-3 * np.trace(H) / 3.0 * np.eye(3) + 1e-12 * np.eye(3), J.T @ r[w])
        x = x + step
        if np.hypot(step[0], step[1]) < 1e-5:
            break
    r = np.abs(_residual(pts, shape, x, surface))
    inl = r < _INLIER_M
    cost = float(np.mean(np.minimum(r, _INLIER_M) ** 2))
    return x, float(inl.mean()), cost


def _half_extents(typ, size, R) -> np.ndarray:
    if typ == mujoco.mjtGeom.mjGEOM_BOX:
        local = size[:3]
    elif typ == mujoco.mjtGeom.mjGEOM_CYLINDER:
        local = np.array([size[0], size[0], size[1]])
    else:
        local = np.array([size[0], size[0], size[0]])
    return np.abs(R) @ local


def symmetric(shape: list[tuple]) -> bool:
    """Every geom is a round shape on the body's vertical axis: yaw does not change the piece."""
    for typ, size, pos, R in shape:
        if typ == mujoco.mjtGeom.mjGEOM_BOX or abs(pos[0]) > 1e-6 or abs(pos[1]) > 1e-6:
            return False
        if typ == mujoco.mjtGeom.mjGEOM_CYLINDER and abs(R[2, 2]) < 1 - 1e-6:
            return False
    return True


def section_centre(shape: list[tuple], z: float) -> np.ndarray | None:
    """Body-frame XY centre of the piece's cross-section at height `z` above its origin (upright)."""
    lo, hi = None, None
    for typ, size, pos, R in shape:
        h = _half_extents(typ, size, R)
        if abs(z - pos[2]) > h[2]:
            continue
        a, b = pos[:2] - h[:2], pos[:2] + h[:2]
        lo = a if lo is None else np.minimum(lo, a)
        hi = b if hi is None else np.maximum(hi, b)
    return None if lo is None else 0.5 * (lo + hi)


def _span(shape: list[tuple], u: np.ndarray, keep) -> tuple[float, float] | None:
    lo, hi = None, None
    for typ, size, pos, R in shape:
        h = _half_extents(typ, size, R)
        if not keep(pos, h):
            continue
        c, r = float(pos[:2] @ u), float(np.abs(u) @ h[:2])
        lo = c - r if lo is None else min(lo, c - r)
        hi = c + r if hi is None else max(hi, c + r)
    return None if lo is None else (lo, hi)


def pinch_axis(shape: list[tuple], z: float) -> float | None:
    """Body-frame heading a pinch at height `z` should take; None when the section there is round.

    Along a face normal of the widest flat-sided (box) section, so the pads land on faces, not
    corners; of its two normals, one where nothing higher sticks out past the faces (the pads
    would catch under it), then the narrower.
    """
    box, area = None, 0.0
    for typ, size, pos, R in shape:
        if typ != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        h = _half_extents(typ, size, R)
        if abs(z - pos[2]) <= h[2] and h[0] * h[1] > area:
            box, area = (size, R), h[0] * h[1]
    if box is None:
        return None
    best = None
    for k in (0, 1):
        u = box[1][:2, k] / max(float(np.linalg.norm(box[1][:2, k])), 1e-9)
        at = _span(shape, u, lambda pos, h: abs(z - pos[2]) <= h[2])
        above = _span(shape, u, lambda pos, h: pos[2] + h[2] > z + 1e-6)
        overhang = 0.0 if above is None or at is None else max(0.0, above[1] - at[1], at[0] - above[0])
        if at is None:
            continue
        key = (overhang > 5e-4, at[1] - at[0])
        if best is None or key < best[0]:
            best = (key, float(np.arctan2(u[1], u[0])))
    return None if best is None else best[1]


def chord_centre(shape: list[tuple], body_pos, body_quat, point, axis, reach: float = 0.03, step: float = 0.0002) -> float | None:
    """Where the line `point + t * axis` crosses the piece at its pose: the centre t of that chord, or None if it misses."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(body_quat, dtype=float))
    R = R.reshape(3, 3)
    t = np.arange(-reach, reach + step, step)
    line = np.asarray(point, dtype=float) + t[:, None] * np.asarray(axis, dtype=float)
    inside = _sdf((line - np.asarray(body_pos, dtype=float)) @ R, shape) <= 0.0
    if not inside.any():
        return None
    return float(0.5 * (t[inside].min() + t[inside].max()))


def fit_upright_xy(obs: RgbdObservation, cfg: dict, shape: list[tuple], seed_xy) -> dict | None:
    """Body XY of an upright piece near `seed_xy`, or None when the model does not fit."""
    if not shape:
        return None
    pts = unproject(obs)
    surface = float(cfg.get("sim", {}).get("board", {}).get("surface_z", 0.02))
    sx, sy = float(seed_xy[0]), float(seed_xy[1])
    keep = (
        (np.abs(pts[:, 0] - sx) <= _WINDOW)
        & (np.abs(pts[:, 1] - sy) <= _WINDOW)
        & (pts[:, 2] > surface + _BOARD_LIFT)
        & (pts[:, 2] < surface + _TOP_CEIL)
    )
    pts = pts[keep]
    if len(pts) < _MIN_POINTS:
        return None
    best = None
    starts = [0.0] if symmetric(shape) else np.radians(np.arange(0.0, 360.0, 30.0))
    for yaw in starts:
        x, frac, cost = _fit(pts, shape, [sx, sy, yaw], surface)
        if best is None or cost < best[2]:
            best = (x, frac, cost)
    x, frac, _ = best
    if frac < _MIN_INLIER_FRAC or np.hypot(x[0] - sx, x[1] - sy) > _WINDOW:
        return None
    return {"xy": [float(x[0]), float(x[1])], "yaw": float(x[2]), "inlier_frac": round(frac, 3), "n": int(len(pts))}
