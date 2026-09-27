"""One move of a single piece on an empty board, swept over piece yaw and grasp height.

  python sim/grasp_bench.py                       # every kind, 12 yaws, table heights
  python sim/grasp_bench.py --kinds knight --heights 6,10,14,18,22 --tag knight_h

Cheat wall, so only the grasp varies. Scored from the world (piece upright on
dest); the carry tilt is the worst in-hand tilt from lift to lower.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

OUT = ROOT / "sim" / "debug" / "grasp_bench"
BODIES = {"pawn": "pawn_e2", "rook": "rook_h1", "knight": "knight_g1", "bishop": "bishop_c1", "queen": "queen_d1", "king": "king_e1"}
HELD_TILT_DEG = 15.0


def run_one(args: tuple[str, float, float | None, str, float | None, str]) -> dict:
    kind, yaw, height, square, closed, dest = args
    os.environ.setdefault("ZIMA_SIM_CLOCK", "1")
    from harness import MemoryLimit, _pads_in_piece, _tilt_deg, _watch_memory

    def _mem(*_):
        raise MemoryLimit

    signal.signal(signal.SIGUSR1, _mem)
    done = threading.Event()
    threading.Thread(target=_watch_memory, args=(threading.get_ident(), done), daemon=True).start()
    row = {"kind": kind, "yaw": yaw, "height_mm": None if height is None else round(height * 1000, 1), "square": square}
    t0 = time.time()
    try:
        import pickup
        from mj import Sim, load_config
        from robot import jaw_squeezing, open_robot, world_of
        from scenario import PIECES
        from sequence import occupied_squares

        cfg = load_config()
        world = Sim(xml_path=pickup.FULL_BOARD_XML, cfg=cfg)
        body = BODIES[kind]
        for k, name in enumerate(occupied_squares(world, cfg).values()):
            if name != body:
                world.place_object(name, [3.0 + 0.1 * k, 3.0, 0.02])
        sim = open_robot(world, "cheat")
        sim.quiet = True
        pickup.seed_vertical(sim, cfg, snap=True)
        seen: dict[str, dict] = {}
        tilts: list[tuple[str, float]] = []
        orig = pickup.log_state

        def _log(s, name, extra=None):
            w = world_of(s)
            tilts.append((name, round(_tilt_deg(w.object_pose(body)[1]), 1)))
            if name in ("005_gripper_closed", "006_lift"):
                seen[name] = _pads_in_piece(w, body) | {"tilt": tilts[-1][1], "squeezing": bool(jaw_squeezing(s))}
            return orig(s, name, extra)

        pickup.log_state = _log
        geom = dict(PIECES[kind])
        if height is not None:
            geom["preferred_grasp_height"] = height
        if closed is not None:
            geom["gripper_closed"] = closed
        r = pickup.execute_move(sim, cfg, scenario={
            "target_id": body, "source": square, "dest": dest, "yaw_deg": yaw, "geometry": geom,
            "quiet": True, "_grasp_attempt": 99,
        })
        wq = world.object_pose(body)
        close, lift = seen.get("005_gripper_closed", {}), seen.get("006_lift", {})
        row.update(
            reason=r.get("failure_reason"),
            close_tilt=close.get("tilt"),
            lift_tilt=lift.get("tilt"),
            squeeze_close=close.get("squeezing"),
            squeeze_lift=lift.get("squeezing"),
            pads=[close.get("fixed_jaw_box4"), close.get("moving_jaw_box2")],
            z_mm=round(float(wq[0][2]) * 1000, 1),
            carry_tilt=max([t for n, t in tilts if n in ("006_lift", "007_transport", "008_lower")], default=None),
            tilts=tilts,
            plumb=r.get("plumb"),
            held=bool(r.get("world_success")),
        )
    except MemoryLimit:
        row.update(reason="MEM_LIMIT", held=False)
    except Exception as e:  # noqa: BLE001 — a crash is a result
        row.update(reason=f"EXC {type(e).__name__}: {str(e)[:120]}", held=False)
    finally:
        done.set()
    row["t"] = round(time.time() - t0, 1)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default=",".join(BODIES))
    ap.add_argument("--yaws", default=",".join(str(a) for a in range(0, 360, 30)))
    ap.add_argument("--heights", default="", help="mm, comma separated; empty = the PIECES table")
    ap.add_argument("--square", default="d4")
    ap.add_argument("--dest", default="d6")
    ap.add_argument("--closed", type=float, default=None, help="gripper close target; default = the PIECES table")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--tag", default="latest")
    args = ap.parse_args()
    kinds = args.kinds.split(",")
    yaws = [float(a) for a in args.yaws.split(",")]
    heights = [float(h) / 1000 for h in args.heights.split(",")] if args.heights else [None]
    work = [(k, y, h, args.square, args.closed, args.dest) for k in kinds for h in heights for y in yaws]
    os.environ["PYTHONHASHSEED"] = "0"
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with mp.get_context("spawn").Pool(args.jobs, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(run_one, work):
            rows.append(row)
    rows.sort(key=lambda r: (r["kind"], r["height_mm"] or 0, r["yaw"]))
    (OUT / f"{args.tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    table = defaultdict(list)
    for r in rows:
        table[(r["kind"], r["height_mm"])].append(r)
    for (kind, h), rs in table.items():
        marks = " ".join(("#" if r["held"] else ".") for r in rs)
        print(f"{kind:7s} h={str(h):5s} held {sum(r['held'] for r in rs):2d}/{len(rs)}  yaw 0..330: {marks}")
    fails = [r for r in rows if not r["held"]]
    for r in fails[:12]:
        print(f"  {r['kind']} yaw={r['yaw']:.0f} h={r['height_mm']} {r.get('reason')} close_tilt={r.get('close_tilt')} carry_tilt={r.get('carry_tilt')} plumb={r.get('plumb')} squeeze={r.get('squeeze_close')}/{r.get('squeeze_lift')}")


if __name__ == "__main__":
    main()
