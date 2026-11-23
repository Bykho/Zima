"""Kinematic suites: IK reach and clearance only, no controller, no physics, no wall.

layout   board square size x arm-to-a-file gap (the old `geom_sweep.py`), plus the
         configured layout. Per layout: 64-square observe/pregrasp/grasp IK, and the
         open-gripper gap to the neighbours at each starting man's grasp with all 32
         real `{kind}_{sq}` chessmen on the board.
reach    arm-base position grid (the old `reach_map.py`): observe IK over the
         starting squares at observe height and at a low look.

Rows carry `metrics`; `compare.py` diffs them with a tolerance instead of ok flips.
"""

from __future__ import annotations

import copy
import time
import traceback

WALL = "none"
SQUARE_MM = (37.5, 40.0, 42.5, 45.0, 47.5, 50.0)
GAP_MM = (80.0, 100.0, 120.0, 140.0, 160.0, 180.0)
BASE_DX = (0.0, 0.04, 0.06, 0.08, 0.10, 0.12)
BASE_DY = (0.0, 0.03, 0.05)
COLLIDE_M = 0.0005


def layout_cases() -> list[dict]:
    cases = [{"id": "layout_config", "square_mm": None, "gap_mm": None}]
    return cases + [{"id": f"layout_{s:.1f}_{g:.0f}", "square_mm": s, "gap_mm": g} for s in SQUARE_MM for g in GAP_MM]


def reach_cases() -> list[dict]:
    return [{"id": f"base_{dx:.2f}_{dy:.2f}", "pos": [dx, dy, 0.0]} for dx in BASE_DX for dy in BASE_DY]


def layout_cfg(cfg: dict, size: float, gap: float) -> dict:
    """Ranks centred on the arm, the a-file `gap` in front of it."""
    cfg = copy.deepcopy(cfg)
    base = [float(v) for v in cfg["sim"]["arm_base_position"]]
    cfg["sim"]["board"]["square_size"] = float(size)
    cfg["sim"]["board"]["origin"] = [base[0] + gap, base[1] - 4.0 * size, 0.0]
    return cfg


def _seed(sim, cfg: dict, gripper: float | None = None) -> None:
    """The controller's vertical seed, set directly: a kinematic probe, nothing steps."""
    from pickup import _vertical_seed

    q = dict(_vertical_seed(cfg))
    if gripper is not None:
        q["gripper"] = gripper
    sim.set_qpos(q, hold=True)


def _ik_at(sim, cfg: dict, xyz, gripper: float | None = None) -> bool:
    from ik import solve_ik

    _seed(sim, cfg, gripper)
    return bool(solve_ik(sim, xyz))


def _measure_layout(sim, cfg: dict) -> dict:
    import numpy as np

    from clearance import neighbor_approach_gap
    from occupancy import neighbor_occupancy
    from pickup import pickup_cfg
    from scenario import PIECES, STARTING_POSITION
    from squares import FILES, square_to_world

    p = pickup_cfg(cfg)
    surface = float(cfg["sim"]["board"]["surface_z"])
    reach = {}
    for sq in (f"{f}{r}" for r in range(1, 9) for f in FILES):
        x, y, _ = square_to_world(sq, cfg, z=surface)
        reach[sq] = {
            "observe": _ik_at(sim, cfg, [x, y, surface + float(p.get("observe_z", 0.20))]),
            "pregrasp": _ik_at(sim, cfg, [x, y, surface + float(p.get("pregrasp_z", 0.10))]),
            "grasp": _ik_at(sim, cfg, [x, y, surface + float(p.get("grasp_z", 0.016))]),
        }
    open_g = float(p.get("gripper_open", 0.45))
    gaps, direct, collide = [], 0, 0
    for kind, sq in STARTING_POSITION:
        x, y, _ = square_to_world(sq, cfg, z=surface)
        ok = _ik_at(sim, cfg, [x, y, surface + float(PIECES[kind]["preferred_grasp_height"])], gripper=open_g)
        near = list(neighbor_occupancy(sim, cfg, sq).values())
        gap = float(neighbor_approach_gap(sim, near)) if ok and near else (1.0 if ok else None)
        if gap is not None and near:
            gaps.append(gap * 1000.0)
        hit = bool(ok and near and gap < COLLIDE_M)
        collide += hit
        direct += bool(ok and not hit)
    full = [sq for sq, v in reach.items() if all(v.values())]
    return {
        "metrics": {
            "n_reach64": len(full),
            "n_observe": sum(v["observe"] for v in reach.values()),
            "n_pregrasp": sum(v["pregrasp"] for v in reach.values()),
            "n_grasp": sum(v["grasp"] for v in reach.values()),
            "n_direct": direct,
            "n_collide": collide,
            "min_gap_mm": round(min(gaps), 2) if gaps else None,
            "median_gap_mm": round(float(np.median(gaps)), 2) if gaps else None,
        },
        "miss": sorted(set(reach) - set(full)),
    }


def run_layout(case: dict, wall: str, timeout: float) -> dict:
    from mj import Sim, load_config
    from pickup import FULL_BOARD_XML

    t0 = time.time()
    try:
        cfg = load_config()
        if case["square_mm"] is not None:
            cfg = layout_cfg(cfg, case["square_mm"] / 1000.0, case["gap_mm"] / 1000.0)
        sim = Sim(FULL_BOARD_XML, cfg=cfg)
        sim.quiet = True
        m = _measure_layout(sim, cfg)
        board = cfg["sim"]["board"]
        return {"id": case["id"], "wall": WALL, "ok": True, "reason": None, "stage": None, "t": round(time.time() - t0, 1),
                "square_mm": round(float(board["square_size"]) * 1000, 3), "origin": board["origin"], **m}
    except Exception as e:  # noqa: BLE001
        return fail_row(case, wall, f"ERROR {type(e).__name__}: {e}", traceback.format_exc(limit=3), time.time() - t0)


def _observe_ok(sim, cfg: dict, square: str, z_off: float) -> bool:
    from squares import square_to_world

    p = square_to_world(square, cfg)
    return _ik_at(sim, cfg, [p[0], p[1], p[2] + z_off])


def run_reach(case: dict, wall: str, timeout: float) -> dict:
    from mj import Sim, load_config
    from scenario import STARTING_POSITION

    t0 = time.time()
    try:
        cfg = load_config()
        sim = Sim(cfg=cfg)
        sim.quiet = True
        sim.apply_arm_base(case["pos"])
        squares = [sq for _, sq in STARTING_POSITION]
        z = float(cfg["sim"]["pickup"]["observe_z"])
        hits = [sq for sq in squares if _observe_ok(sim, cfg, sq, z)]
        look = [sq for sq in squares if _observe_ok(sim, cfg, sq, 0.08)]
        return {"id": case["id"], "wall": WALL, "ok": True, "reason": None, "stage": None, "t": round(time.time() - t0, 1),
                "pos": case["pos"], "metrics": {"n_observe": len(hits), "n_look": len(look)},
                "miss": sorted(set(squares) - set(hits))}
    except Exception as e:  # noqa: BLE001
        return fail_row(case, wall, f"ERROR {type(e).__name__}: {e}", traceback.format_exc(limit=3), time.time() - t0)


def fail_row(case: dict, wall: str, reason: str, stage: str, t: float) -> dict:
    return {"id": case["id"], "wall": WALL, "ok": False, "reason": reason, "stage": stage, "t": round(t, 1), "metrics": {}}


def report_layout(rows: list[dict]) -> str:
    rows = [r for r in rows if r.get("metrics")]
    if not rows:
        return ""
    lines = ["\nlayout (square mm, a-file gap): reach64, direct/32, open-gripper hits, min gap mm"]
    for r in sorted(rows, key=lambda r: (-(r["metrics"]["n_reach64"] + r["metrics"]["n_direct"] - r["metrics"]["n_collide"]), r["id"])):
        m = r["metrics"]
        gmin = "n/a" if m["min_gap_mm"] is None else f"{m['min_gap_mm']:.1f}"
        lines.append(f"  {r['id']:20s} {m['n_reach64']:3d}/64  {m['n_direct']:3d}/32  {m['n_collide']:3d}  {gmin:>6s}"
                     f"   miss {' '.join(r.get('miss') or []) or '-'}")
    return "\n".join(lines)


def report_reach(rows: list[dict]) -> str:
    rows = [r for r in rows if r.get("metrics")]
    if not rows:
        return ""
    lines = ["\nreach (arm base dx, dy): observe, low look over the 32 starting squares"]
    for r in sorted(rows, key=lambda r: (-r["metrics"]["n_observe"], -r["metrics"]["n_look"], r["id"])):
        m = r["metrics"]
        lines.append(f"  {r['id']:16s} {m['n_observe']:2d}/32  {m['n_look']:2d}/32   miss {' '.join(r.get('miss') or []) or '-'}")
    return "\n".join(lines)
