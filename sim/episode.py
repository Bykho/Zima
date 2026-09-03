"""Episode orchestration: unstack, one execute_move, restack LIFO. No snap-home."""

from __future__ import annotations

from metrics import annotate_episode
from pickup import (
    execute_move,
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
    choose_blocker,
    neighbor_occupancy,
    occupied_squares,
    piece_kind,
    plan_next_unstack,
    probe_approach,
    promote,
    rgba_for_home,
    restore_proxy,
    stow_idle_free_bodies,
)
from squares import bowl_xyz, parse_move, square_to_world


def _setup_board(sim, cfg: dict, sc: dict, tid: str, source: str, spawn, surface: float, yaw_deg: float) -> dict:
    """Hide unused free bodies, hide the target proxy, restore vacated proxies, place the target.

    A 32-piece board already has every man on its square — leave it alone.
    """
    if sim.body_model_pos("pawn_e2") is not None:
        return {}
    keep = {tid}
    for nb in sc.get("neighbors") or []:
        keep.add(nb.get("id", "pawn"))
    hide_sq = sc.get("hide_proxy")
    populated = bool(hide_sq)
    occ = occupied_squares(sim, cfg)
    for name in PIECE_BODIES:
        if name in keep:
            continue
        sitting = [sq for sq, body in occ.items() if body == name]
        if populated and sitting and hide_sq not in sitting:
            continue
        sim.place_object(name, HIDE_XYZ)
    if hide_sq:
        sim.hide_body(f"proxy_{hide_sq}")
    for sq, body in occ.items():
        if body != tid or sq == source:
            continue
        saved = sim.body_model_pos(f"proxy_{sq}")
        if saved is not None:
            sim.set_body_pos(f"proxy_{sq}", [float(saved[0]), float(saved[1]), surface])
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
    populated = bool(sc.get("hide_proxy"))
    reset_board = bool(sc.get("reset_board", True))

    if spawn_xy is not None:
        spawn = list(spawn_xy)
    else:
        base = square_to_world(source, cfg, z=surface)
        spawn = [base[0] + offset_xy[0], base[1] + offset_xy[1], base[2]]

    neighbor_start = {}
    if reset_board:
        neighbor_start = _setup_board(sim, cfg, sc, tid, source, spawn, surface, yaw_deg)
    else:
        stow_idle_free_bodies(sim, cfg)

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
        if str(body).startswith("proxy_"):
            home = body[6:]
            if occ.get(home) == body:
                return home
        return None

    def _undo_promote(kind: str, home: str, xyz: list[float], as_proxy: bool) -> None:
        if as_proxy:
            restore_proxy(sim, kind, xyz, home)
        else:
            sim.place_object(kind, square_to_world(home, cfg, z=surface))

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
                        proxy_xyz=[],
                        as_proxy=False,
                        body=sitting or "",
                    )
                )
                unstack_log.append(entry)
                plan = _plan(square, kind)
            return bool(plan.get("ok"))
        finally:
            in_progress.discard(square)

    if populated:
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
            {
                "kind": match.kind,
                "home": match.home,
                "park": match.park,
                "body": match.body,
                "as_proxy": match.as_proxy,
                "proxy_xyz": match.proxy_xyz,
            }
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
            if match.as_proxy:
                restore_proxy(sim, match.kind, match.proxy_xyz, match.home)
            restack_log.append({"kind": match.kind, "home": match.home, "park": match.park, "ok": True})
        sim.target_id = tid
        if not restack_ok and result.get("success") and not want_lift:
            result["success"] = False
            result["failure_reason"] = "RESTACK_FAIL"
    result["restack"] = restack_log
    result["restack_ok"] = restack_ok
    result["restack_skipped"] = restack_skipped
    stow_idle_free_bodies(sim, cfg)

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


def _park_free_kind(sim, cfg: dict, kind: str, src: str) -> None:
    """One free body per type: park it as a proxy so another piece of that type can move."""
    occ = occupied_squares(sim, cfg)
    if occ.get(src) == kind:
        return
    old = next((sq for sq, body in occ.items() if body == kind), None)
    if old is None:
        return
    homes = getattr(sim, "_kind_proxy_home", {})
    home = homes.get(kind, old)
    if sim.body_model_pos(f"proxy_{home}") is None:
        return
    restore_proxy(sim, kind, square_to_world(old, cfg), home)


def _restore_capture(sim, cfg: dict, dst: str, snap: dict) -> None:
    """No-op. A failed capture leaves physics as-is."""
    _ = (sim, cfg, dst, snap)


def _freeze_loot(sim, kind: str) -> dict | None:
    """Leave a static token in the bowl and free the reusable body."""
    xyz, quat = sim.object_pose(kind)
    n = int(getattr(sim, "_loot_n", 0))
    name = f"loot_{n}"
    if sim.body_model_pos(name) is None:
        sim.place_object(kind, HIDE_XYZ)
        return None
    sim.set_body_pos(name, [float(xyz[0]), float(xyz[1]), max(float(xyz[2]), 0.02)])
    color = sim.body_rgba(kind)
    if color is not None:
        sim.tint_body(name, color)
    sim.place_object(kind, HIDE_XYZ)
    homes = getattr(sim, "_kind_proxy_home", {})
    homes.pop(kind, None)
    sim._kind_proxy_home = homes
    sim._loot_n = n + 1
    return {"type": "loot", "loot": name, "kind": kind, "xyz": xyz.tolist(), "quat": quat.tolist()}


def _demote_to_proxy(sim, cfg: dict, kind: str, square: str) -> None:
    """Park a free body as its home proxy so another piece of that type can use it."""
    homes = getattr(sim, "_kind_proxy_home", {})
    home = homes.get(kind, square)
    if sim.body_model_pos(f"proxy_{home}") is None:
        return
    restore_proxy(sim, kind, square_to_world(square, cfg), home)
    homes.pop(kind, None)
    sim._kind_proxy_home = homes


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
    homes = getattr(sim, "_kind_proxy_home", {})
    snap = None
    promoted = False
    proxy_xyz = None
    if str(body).startswith("proxy_"):
        pos = sim.body_model_pos(body)
        snap = {
            "type": "proxy",
            "body": body,
            "xyz": None if pos is None else pos.tolist(),
            "kind": kind,
            "home": body[6:],
        }
        _park_free_kind(sim, cfg, kind, square)
        proxy_xyz, as_proxy = promote(sim, cfg, square, kind)
        promoted = as_proxy
        homes = getattr(sim, "_kind_proxy_home", {})
        homes[kind] = body[6:]
        sim._kind_proxy_home = homes
    elif body:
        xyz, quat = sim.object_pose(body)
        snap = {
            "type": "free",
            "body": body,
            "xyz": xyz.tolist(),
            "quat": quat.tolist(),
            "kind": kind,
            "home": homes.get(kind, square),
        }
    bowl = bowl_xyz(cfg)
    sc = piece_scenario(
        kind,
        square,
        square,
        "vision" if use_vision else "gt",
        name=f"discard_{kind}_{square}",
        hide_proxy=square,
        reset_board=False,
        quiet=True,
        place_xyz=bowl,
        place_target=False,
        target_id=kind if str(body).startswith("proxy_") else body,
        keep_squares=sorted(keep_squares or ()),
        defer_restack=defer_restack,
    )
    result = run_episode(sim, cfg, use_vision=use_vision, scenario=sc)
    if not result.get("success"):
        if promoted and proxy_xyz is not None and snap and snap.get("home"):
            restore_proxy(sim, kind, snap.get("xyz") or proxy_xyz, snap["home"])
        result["source"] = square
        result["dest"] = "bowl"
        result["piece"] = kind
        result["failure_stage"] = result.get("failure_stage") or "capture"
        return result
    if str(body).startswith("proxy_") or body in PIECE_BODIES:
        loot = _freeze_loot(sim, kind)
        if loot is not None:
            loot["home"] = (snap or {}).get("home")
            snap = loot
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
        elif frame.get("as_proxy"):
            restore_proxy(sim, kind, frame.get("proxy_xyz") or [], frame["home"])
        log.append(entry)
    return {"ok": ok, "restack": log}


def play_move(sim, cfg: dict, text: str, use_vision: bool = True) -> dict:
    """Parse a text move, run it through run_episode on the live board. No chess rules."""
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
        if occ.get(src) == captured:
            _demote_to_proxy(sim, cfg, captured, src)
            occ = occupied_squares(sim, cfg)
            body = occ.get(src, body)
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
        occ = occupied_squares(sim, cfg)
        body = occ.get(src, body)
        kind = piece_kind(src, body) if src in occ else kind
    else:
        parked = []
    if str(body).startswith("proxy_"):
        _park_free_kind(sim, cfg, kind, src)
        promote(sim, cfg, src, kind)
        homes = getattr(sim, "_kind_proxy_home", {})
        homes[kind] = body[6:] if str(body).startswith("proxy_") else src
        sim._kind_proxy_home = homes
    sc = piece_scenario(
        kind,
        src,
        dst,
        "vision" if use_vision else "gt",
        name=f"play_{kind}_{src}_{dst}",
        hide_proxy=src,
        reset_board=False,
        quiet=True,
        target_id=kind if str(body).startswith("proxy_") else body,
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
