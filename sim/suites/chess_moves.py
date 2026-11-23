"""Chess-move suites: one move from a real-play position, set up in the world, made through the wall.

opening   20 first moves from the start position
opera, immortal, evergreen   every plain ply of the game (castling, en passant, promotion skipped)
core      opening + opera (52 moves)
"""

from __future__ import annotations

import os
import signal
import time
import traceback
from contextlib import ExitStack

import diag
import runner
from evaluate_move import tilt_deg as _tilt_deg

OPENING = [
    "a2a4", "b2b3", "c2c4", "d2d4", "e2e4", "f2f4", "g2g3", "h2h4", "b1c3", "g1f3",
    "a7a5", "b7b6", "c7c5", "d7d5", "e7e5", "f7f5", "g7g6", "h7h5", "b8c6", "g8f6",
]
GAMES = ("opera", "immortal", "evergreen")

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
    from occupancy import occupied_squares
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


def run_case(case: dict, wall: str, timeout: float) -> dict:
    os.environ.setdefault("ZIMA_SIM_CLOCK", "1")

    def _alarm(*_):
        raise TimeoutError

    signal.signal(signal.SIGALRM, _alarm)
    row = {"id": case["id"], "move": case["move"], "wall": wall, "plies_before": len(case["setup"])}
    t0 = time.time()
    phases: list[tuple[str, float]] = []
    stack = ExitStack()
    signal.alarm(int(timeout))
    try:
        import numpy as np

        import episode
        import evaluate_move
        import pickup
        from mj import Sim, load_config
        from robot import jaw_squeezing, open_robot, world_of
        from squares import square_to_world

        track: list[list] = []
        close: list[dict] = []
        looks: list[dict] = []
        row["close"] = close
        row["looks"] = looks
        row["track"] = track
        cfg = load_config()
        tilt_before: list[float] = []

        def _phase(sim, name, extra):
            phases.append((name, time.time()))
            runner.send_phase(name)
            tid = getattr(sim, "target_id", None)
            try:
                q, quat = world_of(sim).object_pose(tid)
                belief_err = None
                if sim is not world_of(sim) and tid in getattr(sim, "known", ()):
                    belief_err = [round(float(b - w) * 1000, 1) for b, w in zip(sim.object_pose(tid)[0], q)]
                track.append([name, round(_tilt_deg(quat)), round(pickup.gripper_tilt_deg(sim)), [round(float(v) * 1000) for v in q], belief_err])
            except Exception:  # noqa: BLE001 — target not a body (e.g. before setup)
                pass
            if name in ("005_gripper_closed", "006_lift") and tid:
                close.append(_pads_in_piece(world_of(sim), tid) | {"phase": name, "tilt": round(_tilt_deg(world_of(sim).object_pose(tid)[1]), 1), "extra": extra, "squeezing": bool(jaw_squeezing(sim))})

        def _on(kind, sim, name, data):
            if kind == "phase":
                _phase(sim, name, data.get("extra"))
            elif kind in ("scan_start", "scan_end"):
                phases.append(("scan_start" if kind == "scan_start" else "scan", time.time()))
            elif kind == "look_start":
                tilt_before.append(round(_tilt_deg(world_of(sim).object_pose(data["tid"])[1]), 1))
            elif kind == "look":
                q, quat = world_of(sim).object_pose(data["tid"])
                looks.append({"look": {k: v for k, v in data["out"].items() if k != "xy"}, "tilt_before": tilt_before.pop(), "world_tilt": round(_tilt_deg(quat), 1),
                              "world_off_mm": round(float(np.hypot(*(np.asarray(q[:2]) - np.asarray(square_to_world(data["square"], cfg)[:2])))) * 1000, 1)})

        stack.enter_context(diag.subscribe(_on))
        world = Sim(xml_path=pickup.FULL_BOARD_XML, cfg=cfg)
        _setup_world(world, cfg, case["setup"])
        world.step(60)
        sim = open_robot(world, wall)
        sim.quiet = True
        _, at_start = evaluate_move.board_snapshot(world, cfg)
        pickup.seed_vertical(sim, cfg)
        row["home_disturbed"] = evaluate_move.disturbed(world, at_start)
        src, dst = case["move"][:2], case["move"][2:4]
        occ, before = evaluate_move.board_snapshot(world, cfg)
        mover, victim = occ.get(src), occ.get(dst)
        t_setup = time.time()
        r = episode.play_move(sim, cfg, case["move"], use_vision=True)
        row["t_move"] = round(time.time() - t_setup, 1)
        score = evaluate_move.score_board_move(world, cfg, before, mover, victim, dst)
        ok = score["ok"]
        if "mover_final" in score:
            row["mover_final"] = score["mover_final"]
            if hasattr(sim, "known"):
                row["mover_believed"] = mover in sim.known
        row.update(
            ok=ok,
            ok_loose=score["ok_loose"],
            mover_ok=score["mover_ok"],
            victim_gone=score["victim_gone"],
            disturbed=score["disturbed"],
            claimed=bool(r.get("success")),
            reason=r.get("failure_reason") if not r.get("success") else (None if ok else ("COLLATERAL_KNOCK" if score["disturbed"] else "BELIEF_WRONG")),
            fallen_after=r.get("fallen_after"),
            grasp_check=r.get("grasp_check"),
            stage=r.get("failure_stage"),
            capture=victim is not None,
            plumb=r.get("plumb"),
            retreat_straight=(r.get("place_diag") or {}).get("retreat_straight"),
        )
    except TimeoutError:
        row.update(ok=False, claimed=False, reason="TIMEOUT", stage="timeout")
    except Exception as e:  # noqa: BLE001 — a crash is a result, not a harness failure
        row.update(ok=False, claimed=False, reason=f"EXC {type(e).__name__}", stage="crash", error=str(e)[:300],
                   trace=traceback.format_exc(limit=-6))
    finally:
        signal.alarm(0)
        stack.close()
    row["t"] = round(time.time() - t0, 1)
    row["phases"] = runner.phase_times(phases, t0)
    return row


def fail_row(case: dict, wall: str, reason: str, stage: str, t: float) -> dict:
    return {"id": case["id"], "move": case["move"], "wall": wall, "plies_before": len(case["setup"]),
            "ok": False, "claimed": False, "reason": reason, "stage": stage, "t": t, "phases": {}}
