"""Grasp suite: one move of a single piece on an otherwise empty board, swept over piece yaw.

grasp   every kind x 12 yaws (0..330), PIECES table heights, d4 -> d6

Scored from the world (piece upright on dest); the carry tilt is the worst
in-hand tilt from lift to lower. `sim/grasp_bench.py` runs custom grids
(heights, close targets, squares) through the same `run_case`.
"""

from __future__ import annotations

import os
import time
import traceback

import diag
import runner

BODIES = {"pawn": "pawn_e2", "rook": "rook_h1", "knight": "knight_g1", "bishop": "bishop_c1", "queen": "queen_d1", "king": "king_e1"}
YAWS = tuple(float(a) for a in range(0, 360, 30))


def case_id(kind: str, yaw: float, height: float | None = None, closed: float | None = None,
            square: str = "d4", dest: str = "d6") -> str:
    parts = [kind, f"y{yaw:03.0f}"]
    if height is not None:
        parts.append(f"h{height * 1000:g}")
    if closed is not None:
        parts.append(f"c{closed:g}")
    if (square, dest) != ("d4", "d6"):
        parts.append(square + dest)
    return "_".join(parts)


def make_case(kind: str, yaw: float, height: float | None = None, closed: float | None = None,
              square: str = "d4", dest: str = "d6") -> dict:
    return {"id": case_id(kind, yaw, height, closed, square, dest), "kind": kind, "yaw": yaw, "height": height,
            "closed": closed, "square": square, "dest": dest}


def suite_cases() -> list[dict]:
    return [make_case(k, y) for k in BODIES for y in YAWS]


def _head(case: dict, wall: str) -> dict:
    h = case["height"]
    return {"id": case["id"], "wall": wall, "kind": case["kind"], "yaw": case["yaw"],
            "height_mm": None if h is None else round(h * 1000, 1), "square": case["square"]}


def run_case(case: dict, wall: str, timeout: float) -> dict:
    os.environ.setdefault("ZIMA_SIM_CLOCK", "1")
    kind = case["kind"]
    row = _head(case, wall)
    t0 = time.time()
    try:
        import episode
        import evaluate_move
        import pickup
        from evaluate_move import tilt_deg
        from mj import Sim, load_config
        from robot import jaw_squeezing, open_robot, world_of
        from scenario import PIECES
        from occupancy import occupied_squares
        from squares import square_to_world
        from suites.chess_moves import _pads_in_piece

        cfg = load_config()
        world = Sim(xml_path=pickup.FULL_BOARD_XML, cfg=cfg)
        body = BODIES[kind]
        for k, name in enumerate(occupied_squares(world, cfg).values()):
            if name != body:
                world.place_object(name, [3.0 + 0.1 * k, 3.0, 0.02])
        surface = float(cfg["sim"]["board"]["surface_z"])
        world.place_object(body, square_to_world(case["square"], cfg, z=surface), pickup.yaw_quat(case["yaw"]))
        sim = open_robot(world, wall)
        sim.quiet = True
        pickup.seed_vertical(sim, cfg)
        seen: dict[str, dict] = {}
        tilts: list[tuple[str, float]] = []

        def _on(kind, s, name, data):
            if kind != "phase":
                return
            runner.send_phase(name)
            w = world_of(s)
            tilts.append((name, round(tilt_deg(w.object_pose(body)[1]), 1)))
            if name in ("005_gripper_closed", "006_lift"):
                seen[name] = _pads_in_piece(w, body) | {"tilt": tilts[-1][1], "squeezing": bool(jaw_squeezing(s))}

        geom = dict(PIECES[kind])
        if case["height"] is not None:
            geom["preferred_grasp_height"] = case["height"]
        if case["closed"] is not None:
            geom["gripper_closed"] = case["closed"]
        with diag.subscribe(_on):
            r = pickup.execute_move(sim, cfg, scenario={
                "target_id": body, "source": case["square"], "dest": case["dest"], "yaw_deg": case["yaw"], "geometry": geom,
                "quiet": True, "_grasp_attempt": 99,
            })
        wq = world.object_pose(body)
        close, lift = seen.get("005_gripper_closed", {}), seen.get("006_lift", {})
        held = bool(r.get("world_success"))
        place = evaluate_move.placement(world, cfg, body, case["dest"])
        row.update(
            ok=held,
            ok_loose=bool(place["ok_loose"] and not r.get("collateral")),
            mover_final={k: place[k] for k in ("off_dest_mm", "z_mm", "tilt", "at_rest", "robot_contact")},
            claimed=bool(r.get("success")),
            reason=r.get("failure_reason"),
            stage=None if held else (r.get("failure_stage") or episode._failure_stage(r)),
            close_tilt=close.get("tilt"),
            lift_tilt=lift.get("tilt"),
            squeeze_close=close.get("squeezing"),
            squeeze_lift=lift.get("squeezing"),
            pads=[close.get("fixed_jaw_box4"), close.get("moving_jaw_box2")],
            z_mm=round(float(wq[0][2]) * 1000, 1),
            carry_tilt=max([t for n, t in tilts if n in ("006_lift", "007_transport", "008_lower")], default=None),
            tilts=tilts,
            plumb=r.get("plumb"),
            held=held,
        )
    except Exception as e:  # noqa: BLE001 — a crash is a result
        row.update(ok=False, held=False, reason=f"EXC {type(e).__name__}: {str(e)[:120]}", stage="crash",
                   trace=traceback.format_exc(limit=-6))
    row["t"] = round(time.time() - t0, 1)
    return row


def fail_row(case: dict, wall: str, reason: str, stage: str, t: float) -> dict:
    return _head(case, wall) | {"ok": False, "held": False, "reason": f"{reason} @ {stage}", "stage": stage, "t": t}
