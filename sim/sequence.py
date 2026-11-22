"""LIFO unstack: park blocking neighbors, then restore them in reverse.

On a full board each square is its own free body, so a pawn can park a
neighboring pawn. Same-type skip is only when the caller passes skip_kinds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clearance import neighbor_squares, obstacle_bodies, plan_approach
from ik import EE_SITE, grasp_xy_delta, solve_ik
from mj import probing
from scenario import PIECES, STARTING_AT, body_kind, is_piece_body
from squares import FILES, parse_square, square_to_world

MAX_UNSTACK = 4
_PARK_ORDER = tuple(f"{f}{r}" for r in (3, 4, 5, 6) for f in "decfbga")


@dataclass(frozen=True)
class Move:
    """One square-to-square transfer. Sequencer output; execute_move consumes it."""

    kind: str
    src: str
    dst: str


@dataclass
class UnstackFrame:
    kind: str
    home: str
    park: str
    body: str = ""


def piece_kind(square: str, body: str) -> str:
    return body_kind(body) or STARTING_AT.get(square, "pawn")


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


def restore_order(moves: list[Move]) -> list[Move]:
    """LIFO restore: park → home for each unstack move, last parked first."""
    return [Move(kind=m.kind, src=m.dst, dst=m.src) for m in reversed(moves)]


def plan_unstack_move(
    square: str,
    kind: str,
    occ: dict[str, str],
    skip_squares: set[str],
    skip_kinds: set[str],
    occupied: set[str] | None = None,
    prefer: str | None = None,
) -> Move | None:
    """Next park move to free `square`. Occupancy only; no arm motion."""
    _ = kind
    neigh = {sq: body for sq, body in occ.items() if sq in set(neighbor_squares(square))}
    if prefer:
        if (
            prefer not in neigh
            or prefer in skip_squares
            or piece_kind(prefer, neigh[prefer]) in skip_kinds
        ):
            return None
        home = prefer
    else:
        home = choose_blocker(square, neigh, skip_squares, skip_kinds)
    if not home:
        return None
    blocked = set(occupied if occupied is not None else occ) | skip_squares | {home}
    park = choose_park({}, blocked, occupied if occupied is not None else occ)
    if not park:
        return None
    return Move(kind=piece_kind(home, neigh[home]), src=home, dst=park)


def plan_next_unstack(
    sim,
    cfg: dict,
    square: str,
    kind: str,
    skip_squares: set[str],
    skip_kinds: set[str],
    prefer: str | None = None,
) -> Move | None:
    occ = occupied_squares(sim, cfg)
    return plan_unstack_move(square, kind, occ, skip_squares, skip_kinds, set(occ), prefer=prefer)


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


def choose_park(cfg: dict, blocked: set[str], occ: dict | set | None = None) -> str | None:
    """Empty square with the fewest occupied neighbors. Rank-3 was too tight
    once every man is a real mesh, not a proxy cylinder."""
    _ = cfg
    sitting = set(occ if occ is not None else blocked) | set(blocked)
    if occ is not None and not isinstance(occ, set):
        sitting = set(occ) | set(blocked)
    candidates = [sq for sq in _PARK_ORDER if sq not in blocked]
    if not candidates:
        return None

    def clutter(sq: str) -> tuple[int, int]:
        n = sum(1 for nb in neighbor_squares(sq) if nb in sitting)
        return (n, _PARK_ORDER.index(sq))

    candidates.sort(key=clutter)
    return candidates[0]


def choose_blocker(
    source: str,
    occ: dict[str, str],
    skip_squares: set[str],
    skip_kinds: set[str],
) -> str | None:
    """Neighbor most aligned with the clutter cluster. Pawns first. No same-type."""
    candidates = []
    for sq, body in occ.items():
        if sq in skip_squares:
            continue
        if piece_kind(sq, body) in skip_kinds:
            continue
        candidates.append(sq)
    if not candidates:
        return None
    fi, rank = parse_square(source)
    acc = np.zeros(2)
    for sq in candidates:
        f2, r2 = parse_square(sq)
        acc += np.array([float(f2 - fi), float(r2 - rank)])
    if float(np.linalg.norm(acc)) < 0.25:
        acc = np.array([3.5 - fi, 4.5 - rank])

    def score(sq: str) -> tuple:
        kind = piece_kind(sq, occ[sq])
        f2, r2 = parse_square(sq)
        v = np.array([float(f2 - fi), float(r2 - rank)])
        return (0 if kind == "pawn" else 1, -float(np.dot(v, acc)))

    candidates.sort(key=score)
    return candidates[0]


def forward_blockers(source: str, occ: dict[str, str], skip_kinds: set[str]) -> list[str]:
    """Occupied neighbors one rank toward the board center, not same-type."""
    fi, rank = parse_square(source)
    toward = 1 if rank <= 4 else -1
    out = []
    for sq, body in occ.items():
        if piece_kind(sq, body) in skip_kinds:
            continue
        f2, r2 = parse_square(sq)
        if r2 == rank + toward and abs(f2 - fi) <= 1:
            out.append(sq)
    return out


@probing
def probe_approach(
    sim, cfg: dict, source: str, tid: str, scenario: dict, at=None,
    require_extract: bool = True, force_roll: float | None = None, lock_roll: bool = False,
) -> dict:
    """plan_approach result for the live occupancy. Restores arm qpos.

    `at` overrides the piece pose so an empty destination square can be
    checked with the same extract envelope as a pickup.
    """
    p = cfg.get("sim", {}).get("pickup", {})
    geom = dict(scenario.get("geometry") or PIECES.get(body_kind(tid) or tid) or {})
    surface = float(cfg.get("sim", {}).get("board", {}).get("surface_z", 0.02))
    open_g = float(p.get("gripper_open", 0.45))
    pre_z = float(p.get("pregrasp_z", 0.10))
    grasp_z = float(geom.get("preferred_grasp_height", p.get("grasp_z", 0.016)))
    off = scenario.get(
        "grasp_xy_offset",
        geom.get("grasp_xy_offset", p.get("grasp_xy_offset", [0.0, 0.0])),
    )
    if at is not None:
        xyz = np.asarray(at, dtype=float)
    else:
        xyz, _ = sim.object_pose(tid)
    q0 = sim.qpos()
    if solve_ik(sim, [float(xyz[0]), float(xyz[1]), surface + grasp_z]):
        _, grot = sim.site_pose(EE_SITE)
        d = grasp_xy_delta(grot, off)
        dx, dy = float(d[0]), float(d[1])
    else:
        dx, dy = float(off[0]), float(off[1])
    sim.set_qpos(q0, hold=True)
    pre = [float(xyz[0] + dx), float(xyz[1] + dy), surface + pre_z]
    grasp = [float(xyz[0] + dx), float(xyz[1] + dy), surface + grasp_z]
    clutter = [n for n in obstacle_bodies(sim, skip={tid}) if n != "board"]
    plan = plan_approach(
        sim, pre, grasp, open_g, clutter, source=source, cfg=cfg, scenario=scenario,
        require_extract=require_extract, force_roll=force_roll, lock_roll=lock_roll,
        target=tid,
    )
    sim.set_qpos(q0, hold=True)
    return plan
