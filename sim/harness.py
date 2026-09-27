#!/usr/bin/env python3
"""Parallel move benchmark. Scores from the world, not from the controller's own flags.

Each case is one move from a position taken from real play: the position is
set up directly in the physics world, the controller then makes the move
through the robot wall. A fresh process per case keeps episodes independent.

  python sim/harness.py                              # opening suite, cheat + perception
  python sim/harness.py --suite opera --walls perception
  python sim/harness.py --suite all --jobs 5 --tag baseline
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import resource
import signal
import threading
import traceback
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

OUT = ROOT / "sim" / "debug" / "harness"
OPENING = [
    "a2a4", "b2b3", "c2c4", "d2d4", "e2e4", "f2f4", "g2g3", "h2h4", "b1c3", "g1f3",
    "a7a5", "b7b6", "c7c5", "d7d5", "e7e5", "f7f5", "g7g6", "h7h5", "b8c6", "g8f6",
]
GAMES = ("opera", "immortal", "evergreen")
DISPLACED_MM = 8.0
TILTED_DEG = 30.0
# A healthy episode peaks near 1.6 GB; one runaway once took the machine to 67 GB.
WORKER_MEM_CAP_GB = float(os.environ.get("ZIMA_MEM_CAP_GB", 4.0))


class MemoryLimit(Exception):
    pass


def _peak_gb() -> float:
    kb = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * kb / 1e9


def _watch_memory(main_ident: int, done: threading.Event) -> None:
    while not done.wait(0.25):
        if _peak_gb() > WORKER_MEM_CAP_GB:
            signal.pthread_kill(main_ident, signal.SIGUSR1)
            return


def game_cases(name: str) -> list[dict]:
    """Every plain ply of a famous game, with the moves before it as setup. Castling/ep/promotion skipped."""
    import chess

    from game import _load_game

    _, _, moves = _load_game(name, 400)
    board = chess.Board()
    done: list[str] = []
    cases = []
    for i, mv in enumerate(moves, start=1):
        special = board.is_castling(mv) or board.is_en_passant(mv) or mv.promotion
        uci = mv.uci()
        if not special:
            cases.append({"id": f"{name}_{i:02d}_{uci}", "setup": list(done), "move": uci})
        tag = "c" if board.is_castling(mv) else "e" if board.is_en_passant(mv) else ""
        board.push(mv)
        done.append(uci + tag)
    return cases


def suite_cases(suite: str) -> list[dict]:
    if suite == "opening":
        return [{"id": f"open_{m}", "setup": [], "move": m} for m in OPENING]
    if suite in GAMES:
        return game_cases(suite)
    if suite == "games":
        return [c for g in GAMES for c in game_cases(g)]
    if suite == "all":
        return suite_cases("opening") + suite_cases("games")
    if suite == "core":
        return suite_cases("opening") + suite_cases("opera")
    raise SystemExit(f"unknown suite {suite}")


def _setup_world(world, cfg: dict, setup: list[str]) -> set[str]:
    """Apply prior moves by placing bodies. Returns captured bodies (parked off the table)."""
    from sequence import occupied_squares
    from squares import square_to_world

    surface = float(cfg["sim"]["board"]["surface_z"])
    at = dict(occupied_squares(world, cfg))
    quat = {b: world.body_pose(b)[1].copy() for b in at.values()}
    captured: list[str] = []
    for uci in setup:
        src, dst = uci[:2], uci[2:4]
        body = at.pop(src)
        if dst in at:
            captured.append(at.pop(dst))
        if uci.endswith("e"):
            captured.append(at.pop(dst[0] + src[1]))
        at[dst] = body
        if uci.endswith("c"):
            rank = src[1]
            rook_src, rook_dst = ("h" + rank, "f" + rank) if dst[0] == "g" else ("a" + rank, "d" + rank)
            at[rook_dst] = at.pop(rook_src)
    for sq, body in at.items():
        world.place_object(body, square_to_world(sq, cfg, z=surface), quat[body])
    for k, body in enumerate(captured):
        world.place_object(body, [3.0 + 0.2 * k, 3.0, 0.02])
    return set(captured)


def _world_snapshot(world, bodies) -> dict:
    return {b: (world.body_pose(b)[0].copy(), world.body_pose(b)[1].copy()) for b in bodies}


def _tilt_deg(q) -> float:
    import numpy as np

    w, x, y, z = q
    zz = 1 - 2 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(zz, -1, 1))))


def _pads_in_piece(world, tid: str) -> dict:
    """Pad centres in the piece body frame (mm), plus contacts per pad."""
    import mujoco

    from robot import PAD_GEOMS

    m, d = world.model, world.data
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, tid)
    R, p = d.xmat[bid].reshape(3, 3), d.xpos[bid]
    out = {}
    for g in PAD_GEOMS:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        out[g] = [round(float(v) * 1000, 1) for v in R.T @ (d.geom_xpos[gid] - p)]
    hits = []
    for i in range(d.ncon):
        c = d.contact[i]
        names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2))}
        if int(m.geom_bodyid[c.geom1]) == bid or int(m.geom_bodyid[c.geom2]) == bid:
            hits.append(sorted(n or "?" for n in names))
    out["contacts"] = hits
    return out


def run_case(args: tuple[dict, str, float]) -> dict:
    case, wall, timeout = args
    os.environ.setdefault("ZIMA_SIM_CLOCK", "1")

    def _alarm(*_):
        raise TimeoutError

    def _mem(*_):
        raise MemoryLimit

    signal.signal(signal.SIGALRM, _alarm)
    signal.signal(signal.SIGUSR1, _mem)
    done = threading.Event()
    threading.Thread(target=_watch_memory, args=(threading.get_ident(), done), daemon=True).start()
    row = {"id": case["id"], "move": case["move"], "wall": wall, "plies_before": len(case["setup"])}
    t0 = time.time()
    phases: list[tuple[str, float]] = []
    signal.alarm(int(timeout))
    try:
        import numpy as np

        import episode
        import perceive
        import pickup
        from mj import Sim, load_config
        from robot import jaw_squeezing, open_robot, world_of
        from sequence import occupied_squares
        from squares import square_to_world

        orig_log = pickup.log_state

        track: list[list] = []
        close: list[dict] = []
        row["close"] = close

        def _log(sim, name, extra=None):
            phases.append((name, time.time()))
            tid = getattr(sim, "target_id", None)
            try:
                q, quat = world_of(sim).object_pose(tid)
                track.append([name, round(_tilt_deg(quat)), round(pickup.gripper_tilt_deg(sim)), [round(float(v) * 1000) for v in q]])
            except Exception:  # noqa: BLE001 — target not a body (e.g. before setup)
                pass
            if name in ("005_gripper_closed", "006_lift") and tid:
                close.append(_pads_in_piece(world_of(sim), tid) | {"phase": name, "tilt": round(_tilt_deg(world_of(sim).object_pose(tid)[1]), 1), "extra": extra, "squeezing": bool(jaw_squeezing(sim))})
            return orig_log(sim, name, extra)

        pickup.log_state = _log
        orig_scan = perceive.perceive_board

        def _scan(sim, cfg, refine=True):
            phases.append(("scan_start", time.time()))
            out = orig_scan(sim, cfg, refine)
            phases.append(("scan", time.time()))
            return out

        perceive.perceive_board = _scan

        looks: list[dict] = []
        orig_verify = pickup.verify_piece

        def _verify(sim, cfg, tid, square):
            before = round(_tilt_deg(world_of(sim).object_pose(tid)[1]), 1)
            out = orig_verify(sim, cfg, tid, square)
            q, quat = world_of(sim).object_pose(tid)
            looks.append({"look": {k: v for k, v in out.items() if k != "xy"}, "tilt_before": before, "world_tilt": round(_tilt_deg(quat), 1),
                          "world_off_mm": round(float(np.hypot(*(np.asarray(q[:2]) - np.asarray(square_to_world(square, cfg)[:2])))) * 1000, 1)})
            return out

        pickup.verify_piece = _verify
        row["looks"] = looks
        row["track"] = track

        cfg = load_config()
        world = Sim(xml_path=pickup.FULL_BOARD_XML, cfg=cfg)
        captured_before = _setup_world(world, cfg, case["setup"])
        world.step(60)
        sim = open_robot(world, wall)
        sim.quiet = True
        pickup.seed_vertical(sim, cfg, snap=True)
        src, dst = case["move"][:2], case["move"][2:4]
        occ = occupied_squares(world, cfg)
        mover, victim = occ.get(src), occ.get(dst)
        pieces = [b for b in occ.values()]
        before = _world_snapshot(world, pieces)
        t_setup = time.time()
        r = episode.play_move(sim, cfg, case["move"], use_vision=True)
        row["t_move"] = round(time.time() - t_setup, 1)
        after = _world_snapshot(world, pieces)
        occ_after = occupied_squares(world, cfg)
        mover_ok = mover is not None and occ_after.get(dst) == mover and _tilt_deg(after[mover][1]) < TILTED_DEG
        victim_gone = victim is None or victim not in occ_after.values()
        disturbed = []
        for b in pieces:
            if b in (mover, victim):
                continue
            d = float(np.linalg.norm(after[b][0][:2] - before[b][0][:2])) * 1000
            tilt = _tilt_deg(after[b][1])
            if d > DISPLACED_MM or tilt > TILTED_DEG:
                disturbed.append({"body": b, "mm": round(d, 1), "tilt": round(tilt, 1)})
        if mover is not None:
            dxy = square_to_world(dst, cfg, z=0.0)
            row["mover_final"] = {
                "xyz_mm": [round(float(v) * 1000, 1) for v in after[mover][0]],
                "off_dest_mm": round(float(np.hypot(*(after[mover][0][:2] - np.asarray(dxy[:2])))) * 1000, 1),
                "tilt": round(_tilt_deg(after[mover][1]), 1),
            }
            if hasattr(sim, "known"):
                row["mover_believed"] = mover in sim.known
        row.update(
            ok=bool(mover_ok and victim_gone and not disturbed),
            mover_ok=bool(mover_ok),
            victim_gone=bool(victim_gone),
            disturbed=disturbed,
            claimed=bool(r.get("success")),
            reason=r.get("failure_reason"),
            fallen_after=r.get("fallen_after"),
            grasp_check=r.get("grasp_check"),
            stage=r.get("failure_stage"),
            capture=victim is not None,
            plumb=r.get("plumb"),
            retreat_straight=(r.get("place_diag") or {}).get("retreat_straight"),
        )
    except TimeoutError:
        row.update(ok=False, claimed=False, reason="TIMEOUT", stage="timeout")
    except MemoryLimit:
        row.update(ok=False, claimed=False, reason="MEM_LIMIT", stage=phases[-1][0] if phases else "setup")
    except Exception as e:  # noqa: BLE001 — a crash is a result, not a harness failure
        row.update(ok=False, claimed=False, reason=f"EXC {type(e).__name__}", stage="crash", error=str(e)[:300],
                   trace=traceback.format_exc(limit=-6))
    finally:
        signal.alarm(0)
        done.set()
    row["peak_gb"] = round(_peak_gb(), 2)
    row["t"] = round(time.time() - t0, 1)
    row["phases"] = _phase_times(phases, t0)
    return row


def _phase_times(phases: list[tuple[str, float]], t0: float) -> dict[str, float]:
    """Seconds spent reaching each logged phase (interval ending at that log)."""
    out: dict[str, float] = {}
    prev = t0
    for name, t in phases:
        if name == "scan_start":
            prev = t
            continue
        key = name.split("_", 1)[1] if name[:3].isdigit() else name
        out[key] = round(out.get(key, 0.0) + (t - prev), 2)
        prev = t
    return out


def summarize(rows: list[dict]) -> str:
    lines = []
    for wall in sorted({r["wall"] for r in rows}):
        rs = [r for r in rows if r["wall"] == wall]
        ok = sum(r["ok"] for r in rs)
        claimed = sum(r.get("claimed", False) for r in rs)
        lying = sum(1 for r in rs if r.get("claimed") and not r["ok"])
        ts = [r["t"] for r in rs]
        lines.append(
            f"\n[{wall}] ok {ok}/{len(rs)} ({100 * ok / max(1, len(rs)):.0f}%)   claimed {claimed}   "
            f"claimed-but-wrong {lying}   time median {median(ts):.1f}s  max {max(ts):.1f}s  total {sum(ts):.0f}s"
        )
        caps = [r for r in rs if r.get("capture")]
        if caps:
            lines.append(f"  captures ok {sum(r['ok'] for r in caps)}/{len(caps)}")
        fails = Counter((r.get("stage") or "?", r.get("reason") or ("DISTURBED" if r.get("disturbed") else "?"))
                        for r in rs if not r["ok"])
        for (stage, reason), n in fails.most_common():
            lines.append(f"  fail {n:3d}  stage={stage:10s} reason={reason}")
        dist = sum(len(r.get("disturbed") or []) for r in rs)
        lines.append(f"  pieces disturbed: {dist}")
        agg: Counter = Counter()
        for r in rs:
            for k, v in (r.get("phases") or {}).items():
                agg[k] += v
        top = ", ".join(f"{k} {v / len(rs):.1f}s" for k, v in agg.most_common(8))
        lines.append(f"  mean time per case by phase: {top}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="opening", help="opening | opera | immortal | evergreen | games | core (opening+opera) | all")
    ap.add_argument("--walls", default="cheat,perception", help="comma list of robot walls")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=90.0, help="seconds per case")
    ap.add_argument("--only", default="", help="comma-separated substrings; keep cases whose id contains any")
    ap.add_argument("--tag", default="latest")
    args = ap.parse_args()

    keys = [k for k in args.only.split(",") if k] or [""]
    cases = [c for c in suite_cases(args.suite) if any(k in c["id"] for k in keys)]
    walls = [w.strip() for w in args.walls.split(",") if w.strip()]
    work = [(c, w, args.timeout) for w in walls for c in cases]
    print(f"{len(cases)} cases x {len(walls)} walls = {len(work)} episodes on {args.jobs} workers")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.tag}.jsonl"
    path.write_text("")
    rows = []
    t0 = time.time()
    os.environ["PYTHONHASHSEED"] = "0"  # set iteration order feeds association and planning; runs must repeat
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.jobs, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(run_case, work):
            rows.append(row)
            with path.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            mark = "ok  " if row["ok"] else "FAIL"
            print(f"  {mark} {row['wall']:10s} {row['id']:28s} {row['t']:5.1f}s  {row.get('reason') or ''}"
                  f"{'  disturbed=' + str(len(row['disturbed'])) if row.get('disturbed') else ''}", flush=True)
    report = summarize(rows) + f"\n\nwall clock {time.time() - t0:.0f}s   rows: {path}"
    print(report)
    (OUT / f"{args.tag}.txt").write_text(report + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
