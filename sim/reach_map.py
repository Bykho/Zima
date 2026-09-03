#!/usr/bin/env python3
"""Observe-pose IK vs arm_base_position. No pickup, no perception.

  python sim/reach_map.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

from ik import solve_ik
from mj import Sim, load_config
from pickup import seed_vertical
from scenario import STARTING_POSITION
from squares import square_to_world

# Far-file / far-rank squares that currently miss observe IK at the origin.
PROBE = ("a1", "d1", "h1", "a8", "d8", "h8", "g1", "g2", "g7", "g8", "h2", "h7")


def observe_ok(sim, cfg, square: str, z_off: float) -> bool:
    seed_vertical(sim, cfg)
    p = square_to_world(square, cfg)
    return bool(solve_ik(sim, [p[0], p[1], p[2] + z_off]))


def score_base(sim, cfg, pos, squares) -> dict:
    sim.apply_arm_base(pos)
    z = float(cfg["sim"]["pickup"]["observe_z"])
    hits = [sq for sq in squares if observe_ok(sim, cfg, sq, z)]
    vis = [sq for sq in squares if observe_ok(sim, cfg, sq, 0.08)]
    return {
        "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
        "observe": hits,
        "look": vis,
        "n_observe": len(hits),
        "n_look": len(vis),
        "miss_observe": [sq for sq in squares if sq not in hits],
    }


def main():
    cfg = load_config()
    squares = [sq for _, sq in STARTING_POSITION]
    sim = Sim(cfg=cfg)
    grid = []
    for dx in (0.0, 0.04, 0.06, 0.08, 0.10, 0.12):
        for dy in (0.0, 0.03, 0.05):
            grid.append((dx, dy, 0.0))
    print("sq probe:", " ".join(PROBE))
    print(f"{'base xy':>16s}  obs  look  miss observe")
    best = None
    for pos in grid:
        row = score_base(sim, cfg, pos, squares)
        miss = " ".join(row["miss_observe"]) or "-"
        print(
            f"  ({pos[0]:5.2f},{pos[1]:5.2f})  "
            f"{row['n_observe']:2d}/{len(squares)}  {row['n_look']:2d}/{len(squares)}  {miss}"
        )
        if best is None or row["n_observe"] > best["n_observe"] or (
            row["n_observe"] == best["n_observe"] and row["n_look"] > best["n_look"]
        ):
            best = row
        # Prefer smaller |pos| on ties already handled by grid order + strict >
    print("\nbest", best["pos"], f"{best['n_observe']}/{len(squares)}", "miss", best["miss_observe"])
    sim.close()


if __name__ == "__main__":
    main()
