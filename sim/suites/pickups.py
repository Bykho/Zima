"""Pickup suites from the old `bench.py` studies, run through `episode.run_pickup` on a wall.

matrix     piece x square on an empty board (`chess_scene_pieces.xml`, bare body names):
           every kind from each source square, the queen from d1 to each destination,
           and every kind by vision from six sources
starting   pick-and-lift of every man on the full starting board, gt and vision

Scored from the world: `matrix` by `score_board_move`, `starting` by `score_lift`
(mover lifted, or placed on the destination when an unstack ran the move on).
"""

from __future__ import annotations

import os
import time
import traceback

import runner

VISION_SOURCES = ("d1", "c2", "d4", "a4", "d6", "h4")


def matrix_cases() -> list[dict]:
    from scenario import dest_reach_scenarios, empty_matrix_scenarios

    out = [{"id": f"empty_{sc['name']}", "sc": sc} for sc in empty_matrix_scenarios("gt")]
    out += [{"id": f"dest_{sc['name']}", "sc": sc} for sc in dest_reach_scenarios("gt")]
    out += [{"id": f"empty_{sc['name']}", "sc": sc} for sc in empty_matrix_scenarios("vision") if sc["source"] in VISION_SOURCES]
    return out


def starting_cases() -> list[dict]:
    from scenario import starting_pickup_scenarios

    return [{"id": f"pick_{sc['target_id']}_{sc['perception']}", "sc": sc}
            for perc in ("gt", "vision") for sc in starting_pickup_scenarios(perc)]


def _head(case: dict, wall: str) -> dict:
    sc = case["sc"]
    from scenario import body_kind

    return {"id": case["id"], "wall": wall, "piece": body_kind(sc["target_id"]) or sc["target_id"], "target": sc["target_id"],
            "source": sc["source"], "dest": sc["dest"], "perception": sc["perception"]}


def _run(case: dict, wall: str, timeout: float, lift_only: bool) -> dict:
    os.environ.setdefault("ZIMA_SIM_CLOCK", "1")
    row = _head(case, wall)
    t0 = time.time()
    phases: list[tuple[str, float]] = []
    try:
        import episode
        import evaluate_move
        import pickup
        from mj import Sim, load_config
        from robot import open_robot
        from scenario import HIDE_XYZ, PIECE_BODIES
        from squares import square_to_world

        orig_log = pickup.log_state

        def _log(sim, name, extra=None):
            phases.append((name, time.time()))
            runner.send_phase(name)
            return orig_log(sim, name, extra)

        pickup.log_state = _log
        cfg = load_config()
        sc = dict(case["sc"])
        tid = sc["target_id"]
        world = Sim(xml_path=pickup.FULL_BOARD_XML if lift_only else pickup.PIECES_XML, cfg=cfg)
        if not lift_only:
            surface = float(cfg["sim"]["board"]["surface_z"])
            for name in PIECE_BODIES:
                if name != tid:
                    world.place_object(name, HIDE_XYZ)
            world.place_object(tid, square_to_world(sc["source"], cfg, z=surface), pickup.yaw_quat(float(sc.get("yaw_deg", 0.0))))
        _, before = evaluate_move.board_snapshot(world, cfg)
        sim = open_robot(world, wall)
        sim.quiet = True
        r = episode.run_pickup(sim, cfg, scenario=sc)
        if lift_only:
            score = evaluate_move.score_lift(world, cfg, before, tid, sc["dest"])
            claimed = bool(r.get("lift_success")) if r.get("stop_after") == "lift" else bool(r.get("success"))
        else:
            score = evaluate_move.score_board_move(world, cfg, before, tid, None, sc["dest"])
            claimed = bool(r.get("success"))
        ok = score["ok"]
        unstack = r.get("unstack") or []
        fc = r.get("first_collision") or {}
        row.update(
            ok=ok,
            claimed=claimed,
            reason=r.get("failure_reason") if not claimed else (None if ok else ("COLLATERAL_KNOCK" if score["disturbed"] else "BELIEF_WRONG")),
            stage=None if claimed else episode._failure_stage(r),
            disturbed=score["disturbed"],
            mover_final=score.get("mover_final"),
            lift_success=r.get("lift_success"),
            transport_success=r.get("transport_success"),
            placement_success=r.get("placement_success"),
            pickup_success=bool(r.get("lift_success")) and not r.get("collateral"),
            unstack=[f"{e.get('kind')}@{e.get('home')}->{e.get('park')}{'' if e.get('ok') else ' FAIL'}" for e in unstack],
            restack_ok=r.get("restack_ok"),
            obstruction=(f"{fc.get('body')}/{fc.get('robot_part')}@{fc.get('phase')}" if fc else None),
            plumb=r.get("plumb"),
        )
    except Exception as e:  # noqa: BLE001 — a crash is a result
        if type(e).__name__ == "GroundTruthAccess":  # a `gt` case on a wall without cheat_state
            row.update(ok=False, claimed=False, reason="NOT_PERCEIVED", stage="perception", error=str(e)[:300])
        else:
            row.update(ok=False, claimed=False, reason=f"EXC {type(e).__name__}", stage="crash", error=str(e)[:300],
                       trace=traceback.format_exc(limit=-6))
    row["t"] = round(time.time() - t0, 1)
    row["phases"] = runner.phase_times(phases, t0)
    return row


def run_matrix(case: dict, wall: str, timeout: float) -> dict:
    return _run(case, wall, timeout, lift_only=False)


def run_starting(case: dict, wall: str, timeout: float) -> dict:
    return _run(case, wall, timeout, lift_only=True)


def fail_row(case: dict, wall: str, reason: str, stage: str, t: float) -> dict:
    return _head(case, wall) | {"ok": False, "claimed": False, "reason": reason, "stage": stage, "t": t, "phases": {}}


def _grid(rows: list[dict], squares: list[str], key: str) -> list[str]:
    kinds = sorted({r["piece"] for r in rows}, key=lambda k: ("pawn", "rook", "knight", "bishop", "queen", "king").index(k))
    cell = {(r["piece"], r[key]): ("P" if r["ok"] else "F") for r in rows}
    lines = ["        " + "".join(f"{s:>5s}" for s in squares)]
    lines += [f"  {k:6s}" + "".join(f"{cell.get((k, s), '-'):>5s}" for s in squares) for k in kinds]
    return lines


def report_matrix(rows: list[dict]) -> str:
    from scenario import DEST_SQUARES, SOURCE_SQUARES

    out = []
    for wall in sorted({r["wall"] for r in rows}):
        rs = [r for r in rows if r["wall"] == wall]
        for label, sub, key, squares in (
            ("piece x source, empty board, gt", [r for r in rs if r["id"].startswith("empty_") and r["perception"] == "gt"], "source", SOURCE_SQUARES),
            ("piece x source, empty board, vision", [r for r in rs if r["id"].startswith("empty_") and r["perception"] == "vision"], "source", VISION_SOURCES),
            ("queen d1 x dest, gt", [r for r in rs if r["id"].startswith("dest_")], "dest", tuple(d for d in DEST_SQUARES if d != "d1")),
        ):
            if sub:
                out += [f"\n[{wall}] {label}: {sum(r['ok'] for r in sub)}/{len(sub)}", *_grid(sub, list(squares), key)]
    return "\n".join(out)


def report_starting(rows: list[dict]) -> str:
    out = []
    for wall in sorted({r["wall"] for r in rows}):
        for perc in ("gt", "vision"):
            rs = {r["source"]: r for r in rows if r["wall"] == wall and r["perception"] == perc}
            if not rs:
                continue
            out.append(f"\n[{wall}] starting pick+lift, {perc}: {sum(r['ok'] for r in rs.values())}/{len(rs)}   (P ok, F fail)")
            out.append("     " + "  ".join("abcdefgh"))
            for rank in "87654321":
                marks = [("P" if rs[f + rank]["ok"] else "F") if f + rank in rs else "." for f in "abcdefgh"]
                out.append(f"  {rank}  " + "  ".join(marks))
            for sq, r in sorted(rs.items()):
                if not r["ok"]:
                    out.append(f"  {r['piece']:6s} {sq} {r.get('reason')} stage={r.get('stage')} hit={r.get('obstruction')} unstack={r.get('unstack')}")
    return "\n".join(out)
