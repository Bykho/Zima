"""Episode orchestration: unstack, one execute_move, restack LIFO. No snap-home."""

from __future__ import annotations

from metrics import annotate_episode
from pickup import (
    _js_raise_ee,
    _perceiving,
    execute_move,
    in_trash,
    open_gripper,
    pickup_cfg,
    seed_vertical,
    settle_queen,
    yaw_quat,
)
from scenario import HIDE_XYZ, PIECE_BODIES, body_kind, piece_scenario
from sequence import (
    MAX_UNSTACK,
    UnstackFrame,
    occupied_squares,
    piece_kind,
    plan_next_unstack,
    probe_approach,
)
from squares import bowl_xyz, parse_move, square_to_world


def _setup_board(sim, cfg: dict, sc: dict, tid: str, source: str, spawn, surface: float, yaw_deg: float) -> dict:
    """Park unused free bodies off the table, place the target and its neighbours.

    A 32-piece board already has every man on its square — leave it alone.
    On a populated board, pieces sitting on other squares stay put.
    """
    if sim.body_model_pos("pawn_e2") is not None:
        return {}
    keep = {tid}
    for nb in sc.get("neighbors") or []:
        keep.add(nb.get("id", "pawn"))
    populated = bool(sc.get("populated"))
    occ = occupied_squares(sim, cfg)
    for name in PIECE_BODIES:
        if name in keep:
            continue
        sitting = [sq for sq, body in occ.items() if body == name]
        if populated and sitting and source not in sitting:
            continue
        sim.place_object(name, HIDE_XYZ)
    sim.place_object(tid, spawn, yaw_quat(yaw_deg))
    neighbor_start = {}
    for nb in sc.get("neighbors") or []:
        nid = nb.get("id", "pawn")
        nsq = nb.get("square")
        npos = square_to_world(nsq, cfg, z=surface) if nsq else list(nb.get("xyz", [0.37, 0.05, -0.5]))
        sim.place_object(nid, npos, yaw_quat(float(nb.get("yaw_deg", 0.0))))
        neighbor_start[nid] = npos[:2]
    return neighbor_start


def run_episode(sim, cfg: dict, use_vision: bool = False, spawn_xy=None, scenario: dict | None = None) -> dict:
    """Reset once, unstack via execute_move, run the requested move, restack LIFO (no snap-home)."""
    sc = dict(scenario or {})
    p = pickup_cfg(cfg)
    tid = sc.get("target_id", "queen")
    kind = body_kind(tid) or tid
    sim.target_id = tid
    sim.quiet = bool(sc.get("quiet", False) or getattr(sim, "quiet", False))
    source = sc.get("source", p.get("source", "d1"))
    dest = sc.get("dest", p.get("dest", "d4"))
    surface = float(cfg.get("sim", {}).get("board", {}).get("surface_z", 0.02))
    open_g = float(p.get("gripper_open", 0.45))
    yaw_deg = float(sc.get("yaw_deg", 0.0))
    offset_xy = [float(v) for v in sc.get("offset_xy", [0.0, 0.0])]
    want_lift = sc.get("stop_after") == "lift"
    populated = bool(sc.get("populated"))
    reset_board = bool(sc.get("reset_board", True))

    if spawn_xy is not None:
        spawn = list(spawn_xy)
    else:
        base = square_to_world(source, cfg, z=surface)
        spawn = [base[0] + offset_xy[0], base[1] + offset_xy[1], base[2]]

    neighbor_start = {}
    if reset_board:
        neighbor_start = _setup_board(sim, cfg, sc, tid, source, spawn, surface, yaw_deg)

    stack: list[UnstackFrame] = []
    unstack_log: list[dict] = []
    wrist = {"roll": None, "set": False}

    def _plan(square: str, kind: str) -> dict:
        occ = occupied_squares(sim, cfg)
        sitting = occ.get(square)
        sc_b = sc if (square == source and sitting == tid) else piece_scenario(
            kind, square, dest, "gt", quiet=True, target_id=sitting or kind
        )
        at = None
        if sitting is None or piece_kind(square, sitting) != kind:
            at = square_to_world(square, cfg, z=surface)
        primary = square in (source, dest)
        return probe_approach(
            sim, cfg, square, sitting or kind, sc_b, at=at,
            require_extract=primary,
            force_roll=wrist["roll"] if square == dest and wrist["set"] else None,
            lock_roll=bool(square == dest and wrist["set"]),
        )

    def _prefer_square(plan: dict) -> str | None:
        body = ((plan or {}).get("block") or {}).get("body")
        if not body:
            return None
        occ = occupied_squares(sim, cfg)
        for sq, name in occ.items():
            if name == body:
                return sq
        return None

    def _execute(kind: str, src: str, dst: str, **extra) -> dict:
        prev_tid = getattr(sim, "target_id", tid)
        prev_step = sim.on_step
        prev_quiet = getattr(sim, "quiet", False)
        body = extra.get("target_id") or occupied_squares(sim, cfg).get(src) or kind
        aux = piece_scenario(
            kind,
            src,
            dst,
            extra.get("perception", "gt"),
            name=f"seq_{kind}_{src}_{dst}",
            quiet=True,
            reset_board=False,
            stop_after=extra.get("stop_after"),
            place_target=extra.get("place_target", False),
            target_id=body,
        )
        try:
            return execute_move(sim, cfg, use_vision=aux.get("perception") == "vision", scenario=aux)
        finally:
            sim.target_id = prev_tid
            sim.on_step = prev_step
            sim.quiet = prev_quiet

    def _unstack(square: str, kind: str, force: bool = False) -> bool:
        plan = _plan(square, kind)
        if plan.get("ok") and not force:
            return True
        in_progress.add(square)
        try:
            while len(stack) < MAX_UNSTACK and (force or not plan.get("ok")):
                skip_sq = {
                    source,
                    dest,
                    square,
                    *in_progress,
                    *failed_homes,
                    *(f.home for f in stack),
                    *(f.park for f in stack),
                    *sc.get("keep_squares", ()),
                }
                skip_kinds: set[str] = set()
                prefer = _prefer_square(plan)
                move = plan_next_unstack(sim, cfg, square, kind, skip_sq, skip_kinds, prefer=prefer)
                if move is None and prefer is not None:
                    move = plan_next_unstack(sim, cfg, square, kind, skip_sq, skip_kinds, prefer=None)
                if move is None:
                    break
                sitting = occupied_squares(sim, cfg).get(move.src)
                moved = _execute(move.kind, move.src, move.dst, target_id=sitting)
                entry = {
                    "kind": move.kind,
                    "home": move.src,
                    "park": move.dst,
                    "ok": bool(moved.get("success")),
                }
                if not moved.get("success"):
                    failed_homes.add(move.dst)
                    entry["reason"] = moved.get("failure_reason")
                    unstack_log.append(entry)
                    continue
                failed_homes.clear()
                stack.append(
                    UnstackFrame(
                        kind=move.kind,
                        home=move.src,
                        park=move.dst,
                        body=sitting or "",
                    )
                )
                unstack_log.append(entry)
                plan = _plan(square, kind)
            return bool(plan.get("ok"))
        finally:
            in_progress.discard(square)

    # Bowl dumps must not park neighbors (Opera dxe5 was unstacking e4/Nf3
    # to "clear" e5, then knocking them). Live play still unstacks.
    if populated and not sc.get("place_xyz") and sc.get("unstack", True):
        failed_homes: set[str] = set()
        in_progress: set[str] = set()
        _unstack(source, kind)
        sim.target_id = tid
    # Fresh episode only. A live game already has the arm over the last
    # square; folding to the seed between plies looks like a reset.
    if reset_board:
        seed_vertical(sim, cfg)
        open_gripper(sim, open_g, steps=40)
        settle_queen(sim)

    target_sc = dict(sc)
    target_sc["reset_board"] = False
    target_sc["seed_home"] = False
    target_sc["place_target"] = False
    if want_lift and stack:
        target_sc["stop_after"] = None
    result = execute_move(sim, cfg, use_vision=use_vision, spawn_xy=spawn_xy, scenario=target_sc)
    result["unstack"] = unstack_log
    result["stop_after"] = sc.get("stop_after")
    if neighbor_start and not result.get("_neighbor_start"):
        result["_neighbor_start"] = neighbor_start

    still_held = bool((result.get("place_diag") or {}).get("still_held"))
    gripper_busy = still_held and not result.get("placement_success")
    restack_ok = None
    restack_log: list[dict] = []
    restack_skipped = None
    if stack and gripper_busy:
        restack_skipped = "gripper_busy"
        restack_ok = False
        restack_log.append({"ok": False, "reason": "gripper_busy", "pending": len(stack)})
    elif stack and sc.get("defer_restack"):
        restack_skipped = "deferred"
        restack_ok = True
        result["parked"] = [
            {"kind": match.kind, "home": match.home, "park": match.park, "body": match.body}
            for match in stack
        ]
    elif stack:
        if sc.get("place_xyz"):
            seed_vertical(sim, cfg)
            open_gripper(sim, open_g, steps=40)
            settle_queen(sim)
        restack_ok = True
        for match in reversed(stack):
            moved = _execute(match.kind, match.park, match.home, target_id=match.body or None)
            if not moved.get("success"):
                restack_ok = False
                restack_log.append(
                    {
                        "kind": match.kind,
                        "home": match.home,
                        "park": match.park,
                        "ok": False,
                        "reason": moved.get("failure_reason"),
                        "place_diag": moved.get("place_diag"),
                    }
                )
                continue
            restack_log.append({"kind": match.kind, "home": match.home, "park": match.park, "ok": True})
        sim.target_id = tid
        if not restack_ok and result.get("success") and not want_lift:
            result["success"] = False
            result["failure_reason"] = "RESTACK_FAIL"
    result["restack"] = restack_log
    result["restack_ok"] = restack_ok
    result["restack_skipped"] = restack_skipped

    if want_lift:
        result["stop_after"] = "lift"
        annotate_episode(result)
    return result


def occupancy_text(sim, cfg: dict) -> str:
    occ = occupied_squares(sim, cfg)
    rows = ["  a b c d e f g h"]
    for rank in range(8, 0, -1):
        cells = []
        for file in "abcdefgh":
            sq = f"{file}{rank}"
            if sq not in occ:
                cells.append(".")
            else:
                kind = piece_kind(sq, occ[sq])
                cells.append(kind[0].upper() if kind != "knight" else "N")
        rows.append(f"{rank} " + " ".join(cells))
    return "\n".join(rows)


def _restore_capture(sim, cfg: dict, dst: str, snap: dict) -> None:
    """No-op. A failed capture leaves physics as-is."""
    _ = (sim, cfg, dst, snap)


def discard_square(
    sim, cfg: dict, square: str, use_vision: bool = False, keep_squares: set[str] | None = None,
    defer_restack: bool = False,
) -> dict:
    """Pick the piece on `square` and drop it in the capture bowl."""
    occ = occupied_squares(sim, cfg)
    body = occ.get(square)
    if not body:
        return {"success": True, "empty": True, "source": square}
    kind = piece_kind(square, body)
    xyz, quat = sim.object_pose(body)
    snap = {"type": "free", "body": body, "xyz": xyz.tolist(), "quat": quat.tolist(), "kind": kind, "home": square}
    sc = piece_scenario(
        kind,
        square,
        square,
        "vision" if use_vision else "gt",
        name=f"discard_{kind}_{square}",
        populated=True,
        reset_board=False,
        quiet=True,
        place_xyz=bowl_xyz(cfg),
        place_target=False,
        target_id=body,
        keep_squares=sorted(keep_squares or ()),
        defer_restack=defer_restack,
    )
    result = run_episode(sim, cfg, use_vision=use_vision, scenario=sc)
    if result.get("success"):
        try:
            dumped_xyz, _ = sim.object_pose(body)
        except Exception:
            dumped_xyz = None
        if dumped_xyz is None or not in_trash(dumped_xyz, cfg):
            result["success"] = False
            result["failure_reason"] = result.get("failure_reason") or "PLACEMENT_POSITION_ERROR"
    if not result.get("success"):
        result["source"] = square
        result["dest"] = "bowl"
        result["piece"] = kind
        result["failure_stage"] = result.get("failure_stage") or "capture"
        return result
    # A bare-name body is the scene's only one of its kind; free it for the next move of that kind.
    if body in PIECE_BODIES:
        sim.place_object(body, HIDE_XYZ)
    result["source"] = square
    result["dest"] = "bowl"
    result["piece"] = kind
    result["captured"] = kind
    result["capture_snap"] = snap
    return result


def restack_parked(sim, cfg: dict, frames: list[dict]) -> dict:
    """Put pieces parked during a capture dump back on their squares."""
    if not frames:
        return {"ok": True, "restack": []}
    p = pickup_cfg(cfg)
    open_g = float(p.get("gripper_open", 0.45))
    seed_vertical(sim, cfg)
    open_gripper(sim, open_g, steps=40)
    settle_queen(sim)
    log = []
    ok = True
    for frame in reversed(frames):
        kind = frame["kind"]
        body = frame.get("body") or kind
        sc = piece_scenario(
            kind,
            frame["park"],
            frame["home"],
            "gt",
            name=f"restack_{kind}_{frame['park']}_{frame['home']}",
            quiet=True,
            reset_board=False,
            place_target=False,
            seed_home=False,
            target_id=body,
        )
        moved = execute_move(sim, cfg, use_vision=False, scenario=sc)
        entry = {
            "kind": kind,
            "home": frame["home"],
            "park": frame["park"],
            "ok": bool(moved.get("success")),
        }
        if not moved.get("success"):
            ok = False
            entry["reason"] = moved.get("failure_reason")
        log.append(entry)
    return {"ok": ok, "restack": log}


def play_move(sim, cfg: dict, text: str, use_vision: bool = True) -> dict:
    """Parse a text move, run it through run_episode on the live board. No chess rules."""
    scan_info = None
    if _perceiving(sim):
        from perceive import perceive_board

        seen = perceive_board(sim, cfg)
        scan_info = {"views": seen["views"], "looked": seen["looked"], "pieces": len(seen["pieces"])}
    try:
        src, dst = parse_move(text)
    except ValueError as exc:
        return {
            "success": False,
            "source": None,
            "dest": None,
            "piece": None,
            "failure_reason": "BAD_MOVE",
            "failure_stage": "parse",
            "error": str(exc),
        }
    occ = occupied_squares(sim, cfg)
    if src not in occ:
        return {
            "success": False,
            "source": src,
            "dest": dst,
            "piece": None,
            "failure_reason": "EMPTY_SRC",
            "failure_stage": "occupancy",
            "occupancy": occupancy_text(sim, cfg),
        }
    body = occ[src]
    kind = piece_kind(src, body)
    captured = None
    capture_snap = None
    if dst in occ:
        captured = piece_kind(dst, occ[dst])
        dumped = discard_square(
            sim, cfg, dst, use_vision=use_vision, keep_squares={src}, defer_restack=True
        )
        if not dumped.get("success"):
            dumped["source"] = src
            dumped["dest"] = dst
            dumped["piece"] = kind
            dumped["captured"] = captured
            dumped["occupancy"] = occupancy_text(sim, cfg)
            dumped["failure_stage"] = dumped.get("failure_stage") or "capture"
            return dumped
        capture_snap = dumped.get("capture_snap")
        parked = dumped.get("parked") or []
        # Bowl dump leaves the arm over the tray. Raise there — folding to
        # the seed from the last square swept through Nf3 (Opera dxe5).
        _js_raise_ee(sim, z_min=0.12)
        settle_queen(sim, steps=24)
        occ = occupied_squares(sim, cfg)
        body = occ.get(src, body)
        kind = piece_kind(src, body) if src in occ else kind
    else:
        parked = []
    sc = piece_scenario(
        kind,
        src,
        dst,
        "vision" if use_vision else "gt",
        name=f"play_{kind}_{src}_{dst}",
        populated=True,
        reset_board=False,
        quiet=True,
        target_id=body,
    )
    result = run_episode(sim, cfg, use_vision=use_vision, scenario=sc)
    if parked:
        restored = restack_parked(sim, cfg, parked)
        result["capture_restack"] = restored.get("restack")
        if result.get("success") and not restored.get("ok"):
            result["success"] = False
            result["failure_reason"] = "RESTACK_FAIL"
    if not result.get("success") and capture_snap:
        _restore_capture(sim, cfg, dst, capture_snap)
    result["source"] = src
    result["dest"] = dst
    result["piece"] = kind
    result["captured"] = captured
    result["occupancy"] = occupancy_text(sim, cfg)
    result["board_scan"] = scan_info
    if result.get("success"):
        result["failure_stage"] = None
    else:
        result["failure_stage"] = _failure_stage(result)
    return result


def _failure_stage(result: dict) -> str:
    if result.get("unstack") and any(not e.get("ok", True) for e in result["unstack"]):
        return "unstack"
    reason = result.get("failure_reason") or ""
    if reason in ("PERCEPTION_FAIL", "PERCEPTION_FAILURE"):
        return "perception"
    if reason in ("IK_NO_SOLUTION", "IK_FAILURE", "NO_COLLISION_FREE_APPROACH", "APPROACH_COLLISION"):
        if result.get("lift_success") and not result.get("transport_success"):
            return "transport"
        if result.get("transport_success"):
            return "place"
        return "approach"
    if not result.get("grasp_acquired") and not result.get("lift_success"):
        if result.get("pregrasp_reached"):
            return "pick"
        return "approach"
    if not result.get("lift_success"):
        return "lift"
    if not result.get("transport_success"):
        return "transport"
    if not result.get("placement_success"):
        return "place"
    if result.get("restack_ok") is False:
        return "restack"
    return reason or "unknown"


def run_pickup(sim, cfg: dict, use_vision: bool = False, spawn_xy=None, scenario: dict | None = None) -> dict:
    return run_episode(sim, cfg, use_vision=use_vision, spawn_xy=spawn_xy, scenario=scenario)
