"""Pickup suites from the old `bench.py` and `evaluate.py` studies, run through `episode.run_pickup` on a wall.

matrix            piece x square on an empty board (`chess_scene_pieces.xml`, bare body names):
                  every kind from each source square, the queen from d1 to each destination,
                  and every kind by vision from six sources
starting          pick-and-lift of every man on the full starting board, gt and vision

Queen grasp studies on `chess_scene.xml` (queen d1 -> d4, bare names):
offsets           queen spawned off the d1 centre (1-D and 2-D offsets), pick-and-lift
perception_error  perceived queen XY off by -8..+8 mm on both axes, pick-and-lift; reports the robust radius
neighbors         one pawn beside d1 (left, right, front), pick-and-lift
transport         d1 -> d4 with a pawn on c1, e1, d2, c2, e2 or none, full move

Scored from the world: full moves by `score_board_move`, pick-and-lift by `score_lift`.
Neighbour pawns are placed by the suite and tracked by `evaluate_move.NeighborTracker`;
perception error is injected by wrapping `pickup.perceived_xy`.
"""

from __future__ import annotations

import os
import time
import traceback

import diag
import runner

VISION_SOURCES = ("d1", "c2", "d4", "a4", "d6", "h4")
KINDS = ("pawn", "rook", "knight", "bishop", "queen", "king")


def _cases(prefix: str, scs: list[dict], **extra) -> list[dict]:
    return [{"id": f"{prefix}{sc['name']}", "sc": sc | extra} for sc in scs]


def matrix_cases() -> list[dict]:
    from scenario import dest_reach_scenarios, empty_matrix_scenarios

    vision = [sc for sc in empty_matrix_scenarios("vision") if sc["source"] in VISION_SOURCES]
    return (_cases("empty_", empty_matrix_scenarios("gt")) + _cases("dest_", dest_reach_scenarios("gt"))
            + _cases("empty_", vision))


def starting_cases() -> list[dict]:
    from scenario import starting_pickup_scenarios

    return [{"id": f"pick_{sc['target_id']}_{sc['perception']}", "sc": sc}
            for perc in ("gt", "vision") for sc in starting_pickup_scenarios(perc)]


def offsets_cases() -> list[dict]:
    from scenario import position_offset_scenarios

    return _cases("", position_offset_scenarios(), stop_after="lift")


def perception_error_cases() -> list[dict]:
    from scenario import perception_error_scenarios

    return _cases("", perception_error_scenarios(), stop_after="lift")


def neighbors_cases() -> list[dict]:
    from scenario import neighbor_scenarios

    return _cases("", neighbor_scenarios(), stop_after="lift")


def transport_cases() -> list[dict]:
    from scenario import transport_scenarios

    return _cases("", transport_scenarios())


def _head(case: dict, wall: str) -> dict:
    from scenario import body_kind

    sc = case["sc"]
    row = {"id": case["id"], "wall": wall, "piece": body_kind(sc["target_id"]) or sc["target_id"], "target": sc["target_id"],
           "source": sc["source"], "dest": sc["dest"], "perception": sc["perception"]}
    for k in ("offset_xy", "pose_error_xy"):
        if any(sc.get(k) or ()):
            row[k + "_mm"] = [round(float(v) * 1000, 1) for v in sc[k]]
    if sc.get("neighbors"):
        row["neighbors"] = [n["square"] for n in sc["neighbors"]]
    return row


def _setup_world(world, cfg: dict, sc: dict) -> None:
    """Bare-name scenes: park every other piece off the table, place the target and the neighbour pawns."""
    from pickup import yaw_quat
    from scenario import HIDE_XYZ, PIECE_BODIES
    from squares import square_to_world

    surface = float(cfg["sim"]["board"]["surface_z"])
    nbs = {n.get("id", "pawn"): n for n in sc.get("neighbors") or ()}
    for name in PIECE_BODIES:
        if name != sc["target_id"] and name not in nbs:
            world.place_object(name, HIDE_XYZ)
    base = square_to_world(sc["source"], cfg, z=surface)
    off = sc.get("offset_xy") or (0.0, 0.0)
    world.place_object(sc["target_id"], [base[0] + off[0], base[1] + off[1], base[2]], yaw_quat(float(sc.get("yaw_deg", 0.0))))
    for name, n in nbs.items():
        world.place_object(name, square_to_world(n["square"], cfg, z=surface), yaw_quat(float(n.get("yaw_deg", 0.0))))


def _run(case: dict, wall: str, xml: str) -> dict:
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


        def _on(kind, sim, name, data):
            if kind == "phase":
                phases.append((name, time.time()))
                runner.send_phase(name)

        cfg = load_config()
        sc = dict(case["sc"])
        neighbors = sc.pop("neighbors", None) or []
        err = sc.pop("pose_error_xy", None) or (0.0, 0.0)
        tid = sc["target_id"]
        lift_only = sc.get("stop_after") == "lift"
        world = Sim(xml_path={"full": pickup.FULL_BOARD_XML, "pieces": pickup.PIECES_XML, "queen": None}[xml], cfg=cfg)
        if xml != "full":
            _setup_world(world, cfg, case["sc"])
        if neighbors:
            sc.update(populated=True, unstack=False)
        _, before = evaluate_move.board_snapshot(world, cfg)
        sim = open_robot(world, wall)
        sim.quiet = True
        if any(err):
            ex, ey = float(err[0]), float(err[1])
            pickup.perceived_xy = lambda s, xyz: [float(xyz[0] + ex), float(xyz[1] + ey)]
        tracker = None
        if neighbors:
            tracker = evaluate_move.NeighborTracker(world, [n.get("id", "pawn") for n in neighbors])
            sim.on_step = tracker.step
        with diag.subscribe(_on):
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
            ok_loose=score.get("ok_loose"),
            claimed=claimed,
            reason=r.get("failure_reason") if not claimed else (None if ok else ("COLLATERAL_KNOCK" if score["disturbed"] else "BELIEF_WRONG")),
            stage=None if claimed else episode._failure_stage(r),
            disturbed=score["disturbed"],
            mover_final=score.get("mover_final"),
            lift_success=r.get("lift_success"),
            transport_success=r.get("transport_success"),
            placement_success=r.get("placement_success"),
            pickup_success=bool(r.get("lift_success")) and not r.get("collateral"),
            close_shove_mm=r.get("close_shove_xy_mm"),
            close_tilt=r.get("close_tilt_deg"),
            unstack=[f"{e.get('kind')}@{e.get('home')}->{e.get('park')}{'' if e.get('ok') else ' FAIL'}" for e in unstack],
            restack_ok=r.get("restack_ok"),
            obstruction=(f"{fc.get('body')}/{fc.get('robot_part')}@{fc.get('phase')}" if fc else None),
            plumb=r.get("plumb"),
        )
        if tracker is not None:
            row["neighbor_eval"] = tracker.finish()
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
    return _run(case, wall, "pieces")


def run_starting(case: dict, wall: str, timeout: float) -> dict:
    return _run(case, wall, "full")


def run_queen(case: dict, wall: str, timeout: float) -> dict:
    return _run(case, wall, "queen")


def fail_row(case: dict, wall: str, reason: str, stage: str, t: float) -> dict:
    return _head(case, wall) | {"ok": False, "claimed": False, "reason": reason, "stage": stage, "t": t, "phases": {}}


def _grid(rows: list[dict], squares: list[str], key: str) -> list[str]:
    kinds = sorted({r["piece"] for r in rows}, key=KINDS.index)
    cell = {(r["piece"], r[key]): ("P" if r["ok"] else "F") for r in rows}
    lines = ["        " + "".join(f"{s:>5s}" for s in squares)]
    lines += [f"  {k:6s}" + "".join(f"{cell.get((k, s), '-'):>5s}" for s in squares) for k in kinds]
    return lines


def _walls(rows: list[dict]):
    for wall in sorted({r["wall"] for r in rows}):
        yield wall, [r for r in rows if r["wall"] == wall]


def report_matrix(rows: list[dict]) -> str:
    from scenario import DEST_SQUARES, SOURCE_SQUARES

    out = []
    for wall, rs in _walls(rows):
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
    for wall, ws in _walls(rows):
        for perc in ("gt", "vision"):
            rs = {r["source"]: r for r in ws if r["perception"] == perc}
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


def _xy_grid(rows: list[dict], key: str) -> list[str]:
    cell = {tuple(int(round(v)) for v in r.get(key) or (0, 0)): r for r in rows}
    xs = sorted({x for x, _ in cell})
    ys = sorted({y for _, y in cell}, reverse=True)
    lines = ["  y\\x " + " ".join(f"{x:+4d}" for x in xs)]
    for y in ys:
        lines.append(f"  {y:+4d} " + " ".join(f"{('Y' if cell[(x, y)]['ok'] else 'n') if (x, y) in cell else '.':>4s}" for x in xs))
    return lines


def report_offsets(rows: list[dict]) -> str:
    out = []
    for wall, rs in _walls(rows):
        out += [f"\n[{wall}] queen spawn offset (mm) -> lift: {sum(r['ok'] for r in rs)}/{len(rs)}   (Y lift, n fail)", *_xy_grid(rs, "offset_xy_mm")]
    return "\n".join(out)


def report_perception_error(rows: list[dict]) -> str:
    out = []
    for wall, rs in _walls(rows):
        out += [f"\n[{wall}] perception error (mm) -> lift: {sum(r['ok'] for r in rs)}/{len(rs)}   (Y lift, n fail)", *_xy_grid(rs, "pose_error_xy_mm"),
                f"  robust radius: {robust_radius(rs)} mm"]
    return "\n".join(out)


def robust_radius(rows: list[dict]) -> int:
    """Largest error ring (mm) the grasp survives: 0, 2 (+-2 axes and diagonals) or 4 (also +-4 on the axes)."""
    ok = {tuple(int(round(v)) for v in r.get("pose_error_xy_mm") or (0, 0)): r["ok"] for r in rows}
    if not ok.get((0, 0)) or not all(ok.get(p) for p in ((2, 0), (-2, 0), (0, 2), (0, -2))):
        return 0
    diag = all(ok.get(p) for p in ((2, 2), (2, -2), (-2, 2), (-2, -2)))
    axis4 = all(ok.get(p) for p in ((4, 0), (-4, 0), (0, 4), (0, -4)))
    return 4 if diag and axis4 else 2


def report_neighbors(rows: list[dict]) -> str:
    out = []
    for wall, rs in _walls(rows):
        out.append(f"\n[{wall}] {rs[0].get('suite', '')} (one pawn near d1): {sum(r['ok'] for r in rs)}/{len(rs)}")
        for r in rs:
            for n in r.get("neighbor_eval") or [{"id": None}]:
                out.append(f"  {r['id']:24s} {'ok  ' if r['ok'] else 'FAIL'} {r.get('reason') or '':18s} contact={n.get('contact')} "
                           f"robot={n.get('robot_contact')} disp={n.get('xy_disp_mm')}mm tilt={n.get('tilt_deg')} clear={n.get('clearance_mm')}mm")
    return "\n".join(out)
