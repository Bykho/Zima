#!/usr/bin/env python3
"""Board geometry sweep. Frozen pickup controller. No config writes.

  python sim/geom_sweep.py

Sweeps square size × arm-to-a-file gap. For each layout: 64-square IK reach,
open-gripper neighbor clearance on the starting board, and frozen
plan_approach pickup feasibility. Does not retune IK, grasp XY, or aperture.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

from clearance import occupied_neighbor_bodies, neighbor_approach_gap
from ik import solve_ik
from mj import DEBUG_DIR, Sim, load_config
from pickup import FULL_BOARD_XML, pickup_cfg
from scenario import HIDE_XYZ, PIECE_BODIES, PIECES, STARTING_POSITION
from squares import FILES, square_to_world

SQUARE_MM = (37.5, 40.0, 42.5, 45.0, 47.5, 50.0)
GAP_MM = (80.0, 100.0, 120.0, 140.0, 160.0, 180.0)
ALL_SQUARES = tuple(f"{f}{r}" for r in range(1, 9) for f in FILES)
COLLIDE_M = 0.0005
OUT = DEBUG_DIR / "geom" / "sweep.json"


def layout_cfg(cfg: dict, size: float, gap: float) -> dict:
    cfg = copy.deepcopy(cfg)
    base = [float(v) for v in cfg["sim"]["arm_base_position"]]
    # Center ranks on the arm so a larger board does not only grow away in +Y.
    origin = [base[0] + gap, base[1] - 4.0 * size, 0.0]
    cfg["sim"]["board"]["square_size"] = float(size)
    cfg["sim"]["board"]["origin"] = origin
    return cfg


def apply_board_geom(sim: Sim, cfg: dict) -> None:
    board = cfg["sim"]["board"]
    origin = [float(v) for v in board["origin"]]
    size = float(board["square_size"])
    half = 4.0 * size
    bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "board")
    gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "board")
    sim.model.body_pos[bid] = np.array([origin[0] + half, origin[1] + half, 0.01], dtype=np.float64)
    sim.model.geom_size[gid, 0] = half
    sim.model.geom_size[gid, 1] = half
    mujoco.mj_forward(sim.model, sim.data)


def proxy_names(sim: Sim) -> list[str]:
    names = []
    for i in range(sim.model.nbody):
        n = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, i)
        if n and n.startswith("proxy_"):
            names.append(n)
    return names


def hide_pieces(sim: Sim) -> None:
    for name in PIECE_BODIES:
        sim.place_object(name, HIDE_XYZ)
    for name in proxy_names(sim):
        sim.hide_body(name, z=-1.0)


def place_starting(sim: Sim, cfg: dict, target_sq: str | None = None, target_kind: str | None = None) -> None:
    surface = float(cfg["sim"]["board"]["surface_z"])
    hide_pieces(sim)
    occupied = {sq: kind for kind, sq in STARTING_POSITION}
    for name in proxy_names(sim):
        sq = name[6:]
        if sq in occupied and sq != target_sq:
            sim.set_body_pos(name, square_to_world(sq, cfg, z=surface))
        else:
            sim.hide_body(name, z=-1.0)
    if target_sq and target_kind:
        sim.place_object(target_kind, square_to_world(target_sq, cfg, z=surface))


def seed_ik(sim: Sim, cfg: dict) -> None:
    """Set the frozen vertical seed. No physics settle — this is a kinematic probe."""
    seed = cfg.get("poses", {}).get("vertical_seed") or {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.47,
        "wrist_flex": 1.18,
        "wrist_roll": 1.58,
        "gripper": 0.45,
    }
    sim.set_qpos(seed, hold=True)


def ik_at(sim: Sim, cfg: dict, xyz) -> bool:
    seed_ik(sim, cfg)
    return bool(solve_ik(sim, xyz))


def reach_square(sim: Sim, cfg: dict, sq: str) -> dict:
    p = pickup_cfg(cfg)
    surface = float(cfg["sim"]["board"]["surface_z"])
    xy = square_to_world(sq, cfg, z=surface)
    off = [float(v) for v in p.get("grasp_xy_offset", [0.0, 0.0])]
    observe = [xy[0], xy[1], surface + float(p.get("observe_z", 0.20))]
    pre = [xy[0], xy[1], surface + float(p.get("pregrasp_z", 0.10))]
    grasp = [xy[0] + off[0], xy[1] + off[1], surface + float(p.get("grasp_z", 0.016))]
    obs = ik_at(sim, cfg, observe)
    pre_ok = ik_at(sim, cfg, pre)
    grasp_ok = ik_at(sim, cfg, grasp)
    return {
        "observe": obs,
        "pregrasp": pre_ok,
        "grasp": grasp_ok,
        "ok": bool(obs and pre_ok and grasp_ok),
    }


def open_gripper_gap(sim: Sim, cfg: dict, sq: str, kind: str) -> dict:
    p = pickup_cfg(cfg)
    surface = float(cfg["sim"]["board"]["surface_z"])
    geom = PIECES[kind]
    off = [float(v) for v in p.get("grasp_xy_offset", geom.get("grasp_xy_offset", [0.0, 0.0]))]
    open_g = float(p.get("gripper_open", 0.45))
    grasp_z = float(geom.get("preferred_grasp_height", p.get("grasp_z", 0.016)))
    xy = square_to_world(sq, cfg, z=surface)
    grasp = [xy[0] + off[0], xy[1] + off[1], surface + grasp_z]
    seed_ik(sim, cfg)
    q = sim.qpos()
    q["gripper"] = open_g
    sim.set_qpos(q, hold=True)
    ik_ok = bool(solve_ik(sim, grasp))
    occ = occupied_neighbor_bodies(sim, cfg, sq)
    neighbors = list(occ.values())
    if not ik_ok:
        gap = -1.0
    elif not neighbors:
        gap = 1.0
    else:
        gap = float(neighbor_approach_gap(sim, neighbors))
    return {
        "ik": ik_ok,
        "n_neighbors": len(neighbors),
        "gap_mm": None if not ik_ok else gap * 1000.0,
        "collide": bool(ik_ok and neighbors and gap < COLLIDE_M),
    }


def measure_layout(sim: Sim, cfg: dict) -> dict:
    apply_board_geom(sim, cfg)
    hide_pieces(sim)
    reach = {sq: reach_square(sim, cfg, sq) for sq in ALL_SQUARES}
    n_reach = sum(1 for v in reach.values() if v["ok"])
    n_obs = sum(1 for v in reach.values() if v["observe"])
    n_pre = sum(1 for v in reach.values() if v["pregrasp"])
    n_grasp = sum(1 for v in reach.values() if v["grasp"])
    miss = [sq for sq, v in reach.items() if not v["ok"]]

    clutter = {}
    n_collide = 0
    n_direct = 0
    gaps = []
    for kind, sq in STARTING_POSITION:
        place_starting(sim, cfg, target_sq=sq, target_kind=kind)
        clr = open_gripper_gap(sim, cfg, sq, kind)
        # Frozen default pose: open gripper at the controller grasp XY/Z.
        # Extra wrist-roll search is the same planner for every layout; skip it
        # here so the sweep stays a geometry comparison, not a retune.
        direct = bool(clr["ik"] and not clr["collide"])
        clutter[sq] = {**clr, "kind": kind, "direct": direct}
        if clr["collide"]:
            n_collide += 1
        if clr["gap_mm"] is not None:
            gaps.append(float(clr["gap_mm"]))
        if direct:
            n_direct += 1

    return {
        "n_reach64": n_reach,
        "n_observe": n_obs,
        "n_pregrasp": n_pre,
        "n_grasp": n_grasp,
        "miss": miss,
        "n_collide": n_collide,
        "n_direct": n_direct,
        "n_start": len(STARTING_POSITION),
        "min_gap_mm": min(gaps) if gaps else None,
        "median_gap_mm": float(np.median(gaps)) if gaps else None,
        "score": n_reach + n_direct - n_collide,
        "reach": {sq: v["ok"] for sq, v in reach.items()},
        "clutter": clutter,
    }


def rank_map(reach: dict[str, bool]) -> str:
    lines = ["      " + " ".join(f"{f:>3s}" for f in FILES)]
    for r in range(8, 0, -1):
        cells = ["  +" if reach.get(f"{f}{r}") else "  ." for f in FILES]
        lines.append(f"   {r} " + " ".join(cells))
    return "\n".join(lines)


def main() -> None:
    t0 = time.time()
    base_cfg = load_config()
    sim = Sim(FULL_BOARD_XML, cfg=base_cfg)
    sim.quiet = True
    rows = []
    print(
        f"{'sq_mm':>6} {'gap_mm':>7} {'origin_xy':>18} "
        f"{'reach':>7} {'obs':>6} {'pre':>6} {'grasp':>6} "
        f"{'direct':>7} {'hit':>4} {'min_mm':>7} {'score':>5}"
    )
    for sq_mm in SQUARE_MM:
        for gap_mm in GAP_MM:
            cfg = layout_cfg(base_cfg, sq_mm / 1000.0, gap_mm / 1000.0)
            m = measure_layout(sim, cfg)
            origin = cfg["sim"]["board"]["origin"]
            row = {
                "square_mm": sq_mm,
                "gap_mm": gap_mm,
                "origin": [round(origin[0], 4), round(origin[1], 4), 0.0],
                **{k: m[k] for k in (
                    "n_reach64", "n_observe", "n_pregrasp", "n_grasp",
                    "n_direct", "n_collide", "n_start", "min_gap_mm",
                    "median_gap_mm", "score", "miss",
                )},
                "reach": m["reach"],
                "clutter": m["clutter"],
            }
            rows.append(row)
            gmin = "n/a" if m["min_gap_mm"] is None else f"{m['min_gap_mm']:.1f}"
            print(
                f"{sq_mm:6.1f} {gap_mm:7.1f} "
                f"({origin[0]:5.2f},{origin[1]:5.2f}) "
                f"{m['n_reach64']:3d}/64 {m['n_observe']:3d}/64 "
                f"{m['n_pregrasp']:3d}/64 {m['n_grasp']:3d}/64 "
                f"{m['n_direct']:3d}/32 {m['n_collide']:4d} {gmin:>7} {m['score']:5d}"
            )
            sys.stdout.flush()

    rows.sort(
        key=lambda r: (
            -r["score"],
            -r["n_reach64"],
            -r["n_direct"],
            r["n_collide"],
            -(r["min_gap_mm"] or -1e9),
        )
    )
    best = rows[0]
    payload = {
        "elapsed_s": time.time() - t0,
        "best": {
            "square_mm": best["square_mm"],
            "gap_mm": best["gap_mm"],
            "origin": best["origin"],
            "n_reach64": best["n_reach64"],
            "n_direct": best["n_direct"],
            "n_collide": best["n_collide"],
            "score": best["score"],
            "min_gap_mm": best["min_gap_mm"],
            "miss": best["miss"],
        },
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print("\nBEST", f"{best['square_mm']} mm squares, {best['gap_mm']} mm gap")
    print(
        f"  reach {best['n_reach64']}/64  direct {best['n_direct']}/32  "
        f"open-gripper hits {best['n_collide']}/32  min gap {best['min_gap_mm']:.1f} mm  "
        f"score {best['score']}"
    )
    print("  miss", " ".join(best["miss"]) or "-")
    print(rank_map(best["reach"]))
    print(f"wrote {OUT}  ({time.time() - t0:.0f}s)")
    sim.close()


if __name__ == "__main__":
    main()
