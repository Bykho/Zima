"""Carry stage: lift, transport and lower over the destination.

Plumb plan first, legacy lift/transport/lower if plumb finds none. The carry
code itself is unchanged from `pickup.py`.
"""

from __future__ import annotations

import numpy as np

import carry as plumb
from clearance import APPROACH_BODIES, near_bodies, obstacle_bodies, safe_transit_ee_z, transit_clearances, transit_nontarget_hit
from ik import CARRY_ORI_WEIGHT, EE_SITE, gripper_tilt_deg, move_lin, solve_ik_down
from mj import ARM_JOINTS, Sim, probing
from pickup import (
    _WORKSPACE_R_M,
    _arm_base_xy,
    _chain_ik_ok,
    _drive_arm,
    _floor_wp,
    _follow_wps,
    _line_wps,
    _outward_xy,
    _pitch_path,
    _vertical_wrist,
    _xy_floor,
    _xyz,
    bystander_contact,
    grasp_acquired,
    gripper_object_transform,
    log_state,
    queen_upright,
    robot_part,
)
from scenario import STARTING_AT
from sequence import square_of_body
from squares import nest_xyz
from stages import register
from stages.types import Fail, Move, Stop


_LEVEL_PINCH_MIN_DEG = 2.5


_PLACE_MARGIN_M = 0.003


def _carry_candidates(start, goal, cfg: dict, n_wp: int, lock: tuple[str, ...]):
    """Carry lines that end on the real goal.

    Midpoints may sit on the 190 mm ring so a high transit does not ask IK
    for a hover that does not exist. The last point is never floored — that
    parked the piece 40+ mm off dest and scraped in at place height (b5).
    """
    n_wp = max(8, int(n_wp))
    raw = _line_wps(start, goal, n_wp)
    dest = [float(goal[0]), float(goal[1]), float(goal[2])]
    yield raw, lock, "straight"
    yield [[w[0], w[1], w[2] + 0.03] for w in raw], lock, "high"
    ring = [_floor_wp(w, cfg) for w in raw[:-1]] + [dest]
    if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) > 0.002 for a, b in zip(raw, ring)):
        yield ring, lock, "ring_then_in"
        yield [[w[0], w[1], w[2] + 0.03] for w in ring], lock, "ring_then_in_high"
    if "wrist_flex" in lock:
        unlock = tuple(j for j in lock if j != "wrist_flex")
        yield raw, unlock, "straight_unlock_flex"


def _carry_lin(
    sim: Sim,
    target,
    duration: float,
    n_wp: int,
    hold: tuple[str, ...],
    extract: tuple[str, ...],
    ori_weight: float = CARRY_ORI_WEIGHT,
) -> bool:
    """Carry/place Cartesian move: keep the piece upright, else lock the close wrist."""
    if move_lin(
        sim,
        target,
        duration=duration,
        physics=True,
        n_wp=n_wp,
        lock_joints=hold,
        ori_weight=ori_weight,
    ):
        return True
    return move_lin(
        sim, target, duration=duration, physics=True, n_wp=n_wp, lock_joints=extract
    )


def _level_pinch_axis(sim: Sim) -> float:
    """Roll the wrist so the pinch axis (gripper local Z) is horizontal.

    Returns the roll change applied, in degrees (0 if already level or no
    reachable roll levels it). Kinematic search over the candidate angles,
    then a physics move so the held piece follows.
    """
    _, R = sim.site_pose(EE_SITE)
    z_z = float(R[2, 2])
    if abs(z_z) < np.sin(np.radians(_LEVEL_PINCH_MIN_DEG)):
        return 0.0
    q0 = sim.qpos()
    cur = float(q0["wrist_roll"])
    lo, hi = sim.model.jnt_range[sim._joint["wrist_roll"]]
    phi0 = float(np.arctan2(z_z, float(R[2, 1])))
    best = None
    ctrl0 = sim.data.ctrl.copy()
    with sim.probe():
        try:
            for phi in (phi0, -phi0, phi0 + np.pi, phi0 - np.pi, -phi0 + np.pi, -phi0 - np.pi):
                cand = float(np.clip(cur + phi, lo, hi))
                sim.set_qpos({"wrist_roll": cand}, hold=False)
                _, Rc = sim.site_pose(EE_SITE)
                if abs(float(Rc[2, 2])) > np.sin(np.radians(_LEVEL_PINCH_MIN_DEG)):
                    continue
                if best is None or abs(cand - cur) < abs(best - cur):
                    best = cand
        finally:
            sim.set_qpos({"wrist_roll": cur}, hold=False)
            sim.data.ctrl[:] = ctrl0
    if best is None or abs(best - cur) < 1e-3:
        return 0.0
    _roll_holding(sim, best)
    return float(np.degrees(best - cur))


def _lower_vertical(
    sim: Sim, tid: str, dest_xy, place_z: float, hold: tuple[str, ...], result: dict, x_axis=None
) -> list[float] | None:
    """Slide the held piece over dest, then drop. Wrists stay put.

    Re-solving a vertical pinch on the way down swung the jaws and dumped
    the bishop (f1c4). Lock flex+roll and translate.
    """
    lock = tuple(dict.fromkeys((*hold, "wrist_flex", "wrist_roll")))
    bystanders = near_bodies(
        sim,
        [n for n in obstacle_bodies(sim, skip={tid}) if n != "board"],
        dest_xy,
        radius=0.055,
    )
    guard = lambda: _place_lower_guard(sim, tid, bystanders)
    gpos, _ = sim.site_pose(EE_SITE)
    q, _ = sim.object_pose(tid)
    align = [
        float(gpos[0] + dest_xy[0] - q[0]),
        float(gpos[1] + dest_xy[1] - q[1]),
        float(gpos[2]),
    ]
    if float(np.hypot(align[0] - gpos[0], align[1] - gpos[1])) > 0.001:
        if not move_lin(sim, align, duration=0.8, physics=True, n_wp=8, lock_joints=lock, guard=guard):
            if getattr(sim, "_last_abort", None) not in (None, "board"):
                result["place_diag"]["lower_bystander"] = getattr(sim, "_last_abort", None)
            # High hover over a near-base dest is often dead. Drop, then slide.
            for z in (0.07, 0.055, 0.045):
                drop = [float(gpos[0]), float(gpos[1]), float(z)]
                if not move_lin(sim, drop, duration=0.6, physics=True, n_wp=6, lock_joints=lock, guard=guard):
                    continue
                gpos, _ = sim.site_pose(EE_SITE)
                q, _ = sim.object_pose(tid)
                slide = [
                    float(gpos[0] + dest_xy[0] - q[0]),
                    float(gpos[1] + dest_xy[1] - q[1]),
                    float(z),
                ]
                if move_lin(sim, slide, duration=0.8, physics=True, n_wp=8, lock_joints=lock, guard=guard):
                    break
            else:
                return None
    gpos, _ = sim.site_pose(EE_SITE)
    q, _ = sim.object_pose(tid)
    place = [
        float(gpos[0] + dest_xy[0] - q[0]),
        float(gpos[1] + dest_xy[1] - q[1]),
        float(place_z),
    ]
    result["place_diag"]["lower_target_xy_mm"] = [round(place[0] * 1000, 1), round(place[1] * 1000, 1)]
    result["place_diag"]["lower_vertical"] = True
    sim._last_abort = None
    ok = move_lin(
        sim, place, duration=1.2, physics=True, n_wp=12, lock_joints=lock, guard=guard,
    )
    aborted = getattr(sim, "_last_abort", None)
    if aborted and aborted != "board":
        result["place_diag"]["lower_bystander"] = aborted
        cur = float(sim.qpos()["wrist_roll"])
        lo, hi = sim.joint_limits()["wrist_roll"]
        for d in (0.4, -0.4, 0.8, -0.8):
            _roll_holding(sim, float(np.clip(cur + d, lo, hi)))
            sim._last_abort = None
            gpos, _ = sim.site_pose(EE_SITE)
            q, _ = sim.object_pose(tid)
            place = [
                float(gpos[0] + dest_xy[0] - q[0]),
                float(gpos[1] + dest_xy[1] - q[1]),
                float(place_z),
            ]
            ok = move_lin(
                sim, place, duration=0.8, physics=True, n_wp=8, lock_joints=lock, guard=guard,
            )
            aborted = getattr(sim, "_last_abort", None)
            if ok or aborted == "board":
                result["place_diag"]["lower_roll_retry"] = round(d, 2)
                break
    if not ok and aborted not in (None, "board"):
        # Stopped on a neighbor instead of shoving it. Release from here.
        result["place_diag"]["lower_stopped_clear"] = True
        return place
    if not ok and aborted != "board":
        return None
    if aborted == "board":
        result["place_diag"]["lower_touchdown_early"] = True
    return place


def _pitch_holding(sim: Sim, flex_goal: float, closed: float, tid: str | None = None, n: int = 10) -> None:
    """Same as _pitch_in_place, but keep the gripper shut so a held piece follows."""
    if abs(flex_goal - float(sim.qpos()["wrist_flex"])) < 1e-3:
        return
    path = _pitch_path(sim, flex_goal, n)
    _drive_arm(sim, path, closed, duration=0.6, until=lambda: grasp_acquired(sim, tid))


@probing
def _place_ee_target(
    sim: Sim, tid: str, dest_xy, z_ee: float, lock: tuple[str, ...], p_local=None, x_axis=None
) -> list[float] | None:
    """EE point at height z_ee that puts the held piece over dest_xy.

    piece = ee + R p_local. R is the gripper rotation at dest (vertical unless
    `x_axis` freezes the current approach). Restores qpos.
    """
    if p_local is None:
        gpos, grot = sim.site_pose(EE_SITE)
        q, _ = sim.object_pose(tid)
        p_local = grot.T @ (q - gpos)
    snap = sim.qpos()
    try:
        ee = np.array([float(dest_xy[0]), float(dest_xy[1]), float(z_ee)])
        for _ in range(2):
            if not solve_ik_down(sim, ee, lock_joints=lock, x_axis=x_axis):
                return None
            _, R = sim.site_pose(EE_SITE)
            off = R @ p_local
            ee = np.array([float(dest_xy[0]) - off[0], float(dest_xy[1]) - off[1], float(z_ee)])
        return [float(v) for v in ee]
    finally:
        sim.set_qpos(snap, hold=True)


def _place_lower_guard(sim: Sim, tid: str, bystanders: list[str]):
    """Stop a place-lower before jaws shove a neighbor (Rxd7 / Qd8)."""
    if _touching_board(sim, tid):
        return "board"
    if bystanders:
        return bystander_contact(sim, bystanders)
    return None


@probing
def _plan_place(
    sim: Sim, tid: str, dest_xy, place_z: float, hold: tuple[str, ...], release_open: float, margin: float = _PLACE_MARGIN_M, x_axis=None
) -> dict:
    """Pick the wrist roll (level pinch, either side) that keeps jaws off dest neighbors.

    Release aperture stays at `release_open`. Narrowing it (0.15 on b8c6) left
    the knight hooked in the pads; it tipped on the square.
    """
    gpos, grot = sim.site_pose(EE_SITE)
    q, _ = sim.object_pose(tid)
    p_local = grot.T @ (q - gpos)
    cur = float(sim.qpos()["wrist_roll"])
    lo, hi = sim.model.jnt_range[sim._joint["wrist_roll"]]
    rolls = [cur]
    for d in (0.4, -0.4, 0.8, -0.8):
        alt = float(np.clip(cur + d, lo, hi))
        if all(abs(alt - r) > 0.05 for r in rolls):
            rolls.append(alt)
    neighbors = [n for n in obstacle_bodies(sim, skip={tid}) if n != "board"]
    from clearance import _min_gap, near_bodies

    neighbors = near_bodies(sim, neighbors, dest_xy)
    out = {"roll": cur, "release": float(release_open), "planned": False, "min_gap_mm": None}
    if not neighbors:
        return out
    snap = sim.qpos()
    ctrl0 = sim.data.ctrl.copy()
    best = None
    try:
        for roll in rolls:
            sim.set_qpos({**snap, "wrist_roll": roll}, hold=False)
            ee = _place_ee_target(sim, tid, dest_xy, place_z, hold, p_local=p_local, x_axis=x_axis)
            if ee is None:
                continue
            sim.set_qpos({**snap, "wrist_roll": roll}, hold=False)
            if not solve_ik_down(sim, ee, lock_joints=hold, x_axis=x_axis):
                continue
            q_low = sim.qpos()
            sim.set_qpos({**q_low, "gripper": float(release_open)}, hold=False)
            gap = _min_gap(sim, APPROACH_BODIES, neighbors)
            sim.set_qpos(q_low, hold=False)
            gap = min(gap, _min_gap(sim, APPROACH_BODIES, neighbors))
            cand = {"roll": roll, "release": float(release_open), "planned": True, "min_gap_mm": round(gap * 1000.0, 2)}
            if gap >= margin:
                return cand
            if best is None or gap > best["min_gap_mm"] / 1000.0:
                best = {**cand, "degraded": True}
    finally:
        sim.set_qpos(snap, hold=False)
        sim.data.ctrl[:] = ctrl0
    return best if best is not None else out


def _plumb_transport(
    sim: Sim, tid: str, result: dict, dest_xy, piece_place_z: float, lift_z: float, obstacles: list[str]
) -> dict | None:
    """Lift, carry and lower with the held piece kept upright (carry.py).

    None: no plumb path exists, nothing moved. Otherwise {"place": ee} or
    {"failure_reason": ...} after the arm has moved.
    """
    v_local, p_local = plumb.hand_frame(sim, tid)
    avoid = [n for n in obstacles if n not in (tid, "board")]
    gpos, _ = sim.site_pose(EE_SITE)
    z_t, _ = safe_transit_ee_z(sim, float(gpos[2]), tid, obstacles)
    z_t = max(z_t, float(gpos[2]) + plumb.PLUMB_END_M)
    end = plumb.PLUMB_END_M
    low = [float(gpos[0]), float(gpos[1]), float(gpos[2]) + end]
    up = [float(gpos[0]), float(gpos[1]), z_t]
    place = plumb.ee_for_piece(sim, [dest_xy[0], dest_xy[1], piece_place_z], v_local, p_local)
    if place is None:
        result["plumb"] = {"no_pose": getattr(sim, "plumb_fail", None)}
        return None
    above = [place[0], place[1], place[2] + end]
    hover = [place[0], place[1], max(z_t, above[2])]
    loose = plumb.AIRBORNE_TILT_DEG
    over = plumb.plan(sim, [(low, plumb.LIFT_TILT_DEG), (up, loose), (hover, loose)], v_local, avoid)
    if over is None:
        result["plumb"] = {"no_carry": getattr(sim, "plumb_fail", None)}
        return None
    snap = sim.qpos()
    with sim.probe():
        sim.set_qpos(over[-1], hold=True)
        down = plumb.plan(sim, [(above, loose), (place, plumb.PLACE_TILT_DEG)], v_local, avoid, gap=0.002)
        sim.set_qpos({n: snap[n] for n in ARM_JOINTS}, hold=True)
    if down is None:
        result["plumb"] = {"no_lower": getattr(sim, "plumb_fail", None)}
        return None
    result["plumb"] = "ok"
    result["transport_path"] = "plumb"
    result["transit_z"] = z_t
    log_state(sim, "006_lift")
    plumb.follow(sim, over, plumb.PLUMB_SPEED_MPS)
    held = grasp_acquired(sim, tid)
    result["lift_success"] = held
    result["transport_success"] = held
    log_state(sim, "007_transport", extra={"target": _xyz(hover), "path": "plumb"})
    if not held:
        return {"failure_reason": "TRANSPORT_DROP"}
    bystanders = near_bodies(sim, avoid, dest_xy, radius=0.055)
    sim._last_abort = None
    plumb.follow(sim, down, plumb.PLUMB_LOWER_SPEED_MPS, guard=lambda: _place_lower_guard(sim, tid, bystanders))
    if getattr(sim, "_last_abort", None) not in (None, "board"):
        result.setdefault("place_diag", {})["lower_bystander"] = sim._last_abort
    return {"place": [float(v) for v in place]}


def _roll_holding(sim: Sim, target_roll: float) -> None:
    """Roll the wrist under physics while every other ctrl (the gripper
    squeeze in particular) stays exactly as the close left it."""
    aid = sim._act["wrist_roll"]
    cur = float(sim.qpos()["wrist_roll"])
    n = max(20, int(max(0.3, 0.4 * abs(target_roll - cur)) / sim.dt))
    for i in range(n):
        u = (i + 1) / n
        u = u * u * (3 - 2 * u)
        sim.data.ctrl[aid] = cur + (target_roll - cur) * u
        sim.step()
    sim.step(10)


def _touching_board(sim: Sim, tid: str):
    return "board" if any(h.get("other") == "board" for h in sim.contacts_with(tid)) else None


def _transport_to(
    sim: Sim,
    goal,
    lock: tuple[str, ...],
    cfg: dict,
    n_wp: int,
    duration: float,
    obstacles=None,
    skip=None,
) -> tuple[bool, str]:
    """Carry EE to goal XY. Probe candidates before moving so a failed line does not drop the piece."""
    goal = np.asarray(goal, dtype=float)
    skip = set(skip or ())
    extract = tuple(dict.fromkeys((*lock, "wrist_flex")))
    moved = False
    for ori, lk_extra in ((CARRY_ORI_WEIGHT, ()), (0.0, ("wrist_flex",))):
        if moved:
            break
        start, _ = sim.site_pose(EE_SITE)
        for wps, lk, label in _carry_candidates(start, goal, cfg, n_wp, lock):
            use = tuple(dict.fromkeys((*lk, *lk_extra)))
            if not _chain_ik_ok(sim, wps, use, obstacles=obstacles, skip=skip, ori_weight=ori):
                continue
            if _follow_wps(sim, wps, duration, use, ori_weight=ori):
                return True, label
            moved = True
            break
    if _carry_lin(sim, goal, duration, max(8, n_wp), lock, extract):
        return True, "straight"
    # Near-base dest: a 10 cm hover is often dead while a low pinch is not.
    low_z = min(float(goal[2]), 0.06)
    if float(goal[2]) > low_z + 0.008:
        low = [float(goal[0]), float(goal[1]), low_z]
        if _carry_lin(sim, low, duration, max(8, n_wp), lock, extract):
            return True, "low"
    return False, "straight"


def carry(m: Move, use_plumb: bool = True, legacy_fallback: bool = True):
    """Lift, transport and lower to just above the destination (plumb, else legacy).

    Plumb is not tried for bowl dumps, nest re-picks and lift-only moves; those always go legacy.
    """
    sim, cfg, sc, result, tid, surface, p, dest, dest_xy, off_board, grasp, plan_xy, hold, clutter, lift_z, place_hover, closed_g, release_open = m.sim, m.cfg, m.sc, m.result, m.tid, m.surface, m.p, m.dest, m.dest_xy, m.off_board, m.grasp, m.plan_xy, m.hold, m.clutter, m.lift_z, m.place_hover, m.closed_g, m.release_open
    g_close, _ = sim.site_pose(EE_SITE)
    pick_xy = np.array([grasp[0], grasp[1]], dtype=np.float64)
    q_gt, _ = sim.object_pose(tid)
    # Held piece XY, not the pre-grasp plan. Close shove would otherwise
    # carry through to dest if we kept object_xy = plan_xy.
    object_xy = np.array(q_gt[:2], dtype=np.float64)
    result["grasp_offset_error_mm"] = float(np.linalg.norm(q_gt[:2] - pick_xy)) * 1000
    result["place_diag"] = {
        "dest_xy_mm": [round(float(dest_xy[0]) * 1000, 1), round(float(dest_xy[1]) * 1000, 1)],
        "plan_xy_mm": [round(float(plan_xy[0]) * 1000, 1), round(float(plan_xy[1]) * 1000, 1)],
        "held_xy_after_close_mm": [round(float(q_gt[0]) * 1000, 1), round(float(q_gt[1]) * 1000, 1)],
        "close_vs_plan_mm": round(float(np.linalg.norm(q_gt[:2] - plan_xy[:2])) * 1000, 1),
        "T_ee_after_close_mm": [round(v, 1) for v in gripper_object_transform(sim, tid)["p_ee_mm"]],
    }

    obstacles = obstacle_bodies(sim, skip={tid})
    mins = {"queen_mm": 1e9, "gripper_mm": 1e9, "camera_mm": 1e9}
    prev_step = sim.on_step

    n_transit = [0]

    def _transit_hook():
        phase = getattr(sim, "_pickup_phase", "") or ""
        n_transit[0] += 1
        if n_transit[0] % 4 == 0 and (
            phase.startswith("007") or phase.startswith("fail_transport")
        ):
            cl = transit_clearances(sim, tid, obstacles)
            for k in mins:
                mins[k] = min(mins[k], cl[k])
        hit = transit_nontarget_hit(sim, tid, obstacles)
        if hit is None:
            return
        if result["transport_first_collision"] is None:
            part = robot_part(hit.get("other"))
            sq = square_of_body(sim, cfg, hit.get("body") or "")
            result["transport_first_collision"] = {
                **hit,
                "phase": phase,
                "robot_part": part,
                "square": sq,
                "piece": STARTING_AT.get(sq) if sq else None,
            }
        if phase.startswith("006") or phase == "fail_lift":
            result["transport_contact_lift"] = True
        elif phase.startswith("007") or phase.startswith("fail_transport"):
            result["transport_contact_transit"] = True
        elif phase.startswith("008") or phase == "fail_lower":
            result["transport_contact_descend"] = True

    def _on_step():
        if prev_step:
            prev_step()
        _transit_hook()

    sim.on_step = _on_step
    plumbed = None
    plumb_applies = not off_board and not sc.get("_from_nest") and sc.get("stop_after") != "lift"
    if use_plumb and plumb_applies:
        plumbed = _plumb_transport(sim, tid, result, dest_xy, surface + place_hover, lift_z, obstacles)
        if plumbed is None and not legacy_fallback:
            result["failure_reason"] = "IK_NO_SOLUTION"
            log_state(sim, "fail_lift", extra={"path": "plumb"})
            return Fail("carry", result["failure_reason"])
    if plumbed is not None and plumbed.get("failure_reason"):
        result["failure_reason"] = plumbed["failure_reason"]
        return Fail("carry", result["failure_reason"], "pick")
    if plumbed is not None:
        place = plumbed["place"]
    else:
        log_state(sim, "006_lift")
        # Locked-wrist +Z from inside ~190 mm of the base is poorly tracked or
        # unreachable. Push the lift target out to that radius (generic workspace
        # floor, not a file/square table). Outside it, lift is still +Z.
        radial = float(np.linalg.norm(np.asarray(g_close[:2], dtype=float) - _arm_base_xy(cfg)))
        extra = max(0.0, _WORKSPACE_R_M - radial)
        lift_xy = _outward_xy(g_close[:2], cfg, extra) if extra > 0.001 else np.asarray(g_close[:2], dtype=float)
        if sc.get("_from_nest"):
            nest_xy = np.asarray(nest_xyz(cfg)[:2], dtype=float)
            away = np.asarray(g_close[:2], dtype=float) - nest_xy
            n = float(np.linalg.norm(away))
            if n > 1e-6:
                lift_xy = np.asarray(g_close[:2], dtype=float) + away / n * 0.025
        lift = [float(lift_xy[0]), float(lift_xy[1]), float(g_close[2] + lift_z)]
        # Extract +Z with the close wrist so the pinch does not peel. Transit and
        # lower then free wrist_flex to keep the piece world-upright.
        extract = tuple(dict.fromkeys((*hold, "wrist_flex")))

        def _lift(lock: tuple[str, ...], down: bool = False, x_axis=None) -> bool:
            return move_lin(sim, lift, duration=1.2, physics=True, n_wp=8, lock_joints=lock, down=down, x_axis=x_axis)

        # Vertical extract: the pinch is a hinge, so the piece hangs plumb; keeping
        # gripper +X world-down while rising is what stops it peeling. A locked
        # flex rotated the gripper up to 40 deg over a 30 mm lift near the base.
        # If the grasp itself was not vertical (near-base squares, g/h files),
        # freeze the *current* approach axis instead: re-verticalising while low
        # swung the jaws 45 deg through the rank above (rook a1 knocked pawn a2).
        _, R_g = sim.site_pose(EE_SITE)
        grasp_axis = np.asarray(R_g, dtype=float).reshape(3, 3)[:, 0]
        keep_axis = grasp_axis if gripper_tilt_deg(sim) > 6.0 else None
        result["lift_axis_frozen"] = keep_axis is not None
        lift_ok = _lift(tuple(hold), down=True, x_axis=keep_axis)
        if not lift_ok:
            result["lift_fallback"] = "locked_wrist"
            lift_ok = _lift(extract)
        if not lift_ok:
            # With flex+roll locked only shoulder_lift/elbow remain, and near the
            # folded (inner) edge of that 2R annulus a straight +Z is unreachable
            # (rank-7 pawns: 6/32 board failures were exactly this). The pinch
            # already holds the piece; a few degrees of flex beat abandoning it.
            result["lift_fallback"] = "free_wrist"
            lift_ok = _lift(tuple(hold))
        if not lift_ok:
            result["failure_reason"] = "IK_FAILURE"
            log_state(sim, "fail_lift", extra={"target": lift})
            return Fail("carry", result["failure_reason"])
        for _ in range(40):
            sim.step()
        log_state(sim, "006_lift")

        qpos, qquat = sim.object_pose(tid)
        object_xy = np.array(qpos[:2], dtype=np.float64)
        result["T_gripper_queen_lift"] = gripper_object_transform(sim, tid)
        result["place_diag"]["held_xy_after_lift_mm"] = [round(float(qpos[0]) * 1000, 1), round(float(qpos[1]) * 1000, 1)]
        result["place_diag"]["T_ee_after_lift_mm"] = [round(v, 1) for v in result["T_gripper_queen_lift"]["p_ee_mm"]]
        clearance = float(p.get("lift_clearance", 0.04))
        held = grasp_acquired(sim, tid)
        lifted = qpos[2] > surface + clearance
        result["max_queen_height_mm"] = float(qpos[2] * 1000)
        result["lift_success"] = bool(lifted and held and queen_upright(qquat, tilt_max=0.7))
        if not result["lift_success"]:
            result["failure_reason"] = "LIFT_DROP" if result["grasp_acquired"] else "GRASP_MISS"
            return Fail("carry", result["failure_reason"], "pick")
        if sc.get("stop_after") == "lift":
            return Stop("carry")

        # A two-pad pinch is a hinge about the pinch axis. Once that axis has a
        # vertical component (wrist at its flex limit on rank 7 + a roll chosen
        # for clearance) the piece rolls on the pads and creeps out over the
        # carry (e7e5 dropped after 60 mm). Roll the wrist so the axis is level;
        # the piece only yaws.
        # Leveling the pinch peels a short carry (g4→f3). Keep it for the
        # long rank-7 transits it was added for.
        carry_xy = float(np.linalg.norm(dest_xy[:2] - object_xy))
        # Bowl dumps are a long carry, but orientation does not matter and
        # leveling the pinch peels L-shaped men off the pads (Opera Bxf3).
        if carry_xy > 0.08 and not off_board:
            result["carry_level_roll_deg"] = round(_level_pinch_axis(sim), 2)
        else:
            result["carry_level_roll_deg"] = 0.0
        if not grasp_acquired(sim, tid):
            result["failure_reason"] = "LIFT_DROP"
            result["lift_success"] = False
            return Fail("carry", result["failure_reason"], "pick")

        gpos, _ = sim.site_pose(EE_SITE)
        # Translate the believed object XY onto dest. Using the EE grasp point here
        # would park the jaws on dest and leave the piece short by grasp_xy_offset.
        delta = dest_xy[:2] - object_xy
        dest_ee = np.array([float(gpos[0] + delta[0]), float(gpos[1] + delta[1])])
        _, standoff = _xy_floor(dest_ee, cfg)
        result["transport_standoff"] = bool(standoff)
        z_ee, derived = safe_transit_ee_z(sim, float(gpos[2]), tid, obstacles)
        if off_board:
            # Extra hover so a long bowl carry does not kiss the file it just left.
            z_ee = max(float(z_ee), float(gpos[2]) + 0.03)
        result["transit_z"] = z_ee
        result["transit_z_derived"] = derived
        result["transport_path"] = "over_board"
        if z_ee > float(gpos[2]) + 0.004:
            high = [float(gpos[0]), float(gpos[1]), z_ee]
            raised = _carry_lin(sim, high, 0.8, 4, hold, extract)
            if not raised:
                out = _outward_xy(gpos[:2], cfg, 0.05)
                high = [float(out[0]), float(out[1]), z_ee]
                raised = _carry_lin(sim, high, 0.8, 4, hold, extract)
            if not raised:
                result["failure_reason"] = "IK_FAILURE"
                log_state(sim, "fail_transport_raise", extra={"target": _xyz(high)})
                return Fail("carry", result["failure_reason"])
            gpos, _ = sim.site_pose(EE_SITE)
            z_ee, derived = safe_transit_ee_z(sim, float(gpos[2]), tid, obstacles)
            result["transit_z"] = float(gpos[2])
            result["transit_z_derived"] = derived
        carry = [float(gpos[0] + delta[0]), float(gpos[1] + delta[1]), float(gpos[2])]
        xy_dist = float(np.linalg.norm(delta))
        n_xy = max(10, int(np.ceil(xy_dist / 0.02)))
        log_state(sim, "007_transport")
        ok, how = _transport_to(sim, carry, hold, cfg, n_xy, max(1.5, 0.4 * n_xy), obstacles=clutter, skip={tid})
        result["transport_detour"] = how
        if not ok:
            result["failure_reason"] = "IK_FAILURE"
            log_state(sim, "fail_transport", extra={"target": _xyz(carry), "detour": how})
            return Fail("carry", result["failure_reason"])
        for _ in range(30):
            sim.step()
        qpos, qquat = sim.object_pose(tid)
        over_dest = float(np.hypot(qpos[0] - dest_xy[0], qpos[1] - dest_xy[1])) < 0.04
        # Near-base dests carry low (the 10 cm hover is dead). 40 mm clearance
        # then called a held piece on dest a drop and scooped it into the cup.
        held_over = bool(
            grasp_acquired(sim, tid)
            and (float(qpos[2]) > surface + 0.012 or over_dest)
        )
        result["transport_success"] = bool(held_over and over_dest and not result["transport_contact_transit"])
        result["min_queen_piece_clearance_mm"] = None if mins["queen_mm"] >= 1e8 else float(mins["queen_mm"])
        result["min_gripper_piece_clearance_mm"] = None if mins["gripper_mm"] >= 1e8 else float(mins["gripper_mm"])
        result["min_camera_piece_clearance_mm"] = None if mins["camera_mm"] >= 1e8 else float(mins["camera_mm"])
        log_state(
            sim,
            "007_transport",
            extra={
                "target": _xyz(carry),
                "dest": dest,
                "path": result["transport_path"],
                "detour": result.get("transport_detour"),
                "transit_z": result.get("transit_z"),
            },
        )
        if not result["transport_success"]:
            result["failure_reason"] = "TRANSPORT_COLLISION" if result["transport_contact_transit"] else "TRANSPORT_DROP"
            return Fail("carry", result["failure_reason"], "pick")

        gpos, _ = sim.site_pose(EE_SITE)
        q_hold, _ = sim.object_pose(tid)
        shift = dest_xy[:2] - q_hold[:2]
        result["place_diag"]["piece_xy_before_lower_mm"] = [
            round(float(q_hold[0]) * 1000, 1),
            round(float(q_hold[1]) * 1000, 1),
        ]
        result["place_diag"]["piece_err_before_lower_mm"] = round(
            float(np.hypot(q_hold[0] - dest_xy[0], q_hold[1] - dest_xy[1])) * 1000, 1
        )
        result["place_diag"]["ee_xy_before_lower_mm"] = [
            round(float(gpos[0]) * 1000, 1),
            round(float(gpos[1]) * 1000, 1),
        ]
        result["place_diag"]["T_ee_before_lower_mm"] = [
            round(v, 1) for v in gripper_object_transform(sim, tid)["p_ee_mm"]
        ]
        # Aim held piece XY at dest, then lower vertically. Combined diagonal
        # lower would drag the piece by the leftover grasp/carry offset.
        place_z = float(g_close[2]) + place_hover
        standoff = bool(result.get("transport_standoff"))
        # Least-tilted reachable pinch at this hover, then lock and translate.
        # Re-solving vertical *during* the lower swings the jaws (f1c4). Far
        # squares cannot be truly vertical; this is the same compromise the
        # real wrist will have to make.
        gpos, _ = sim.site_pose(EE_SITE)
        hover = [float(gpos[0]), float(gpos[1]), float(gpos[2])]
        vert_place = _vertical_wrist(sim, hover)
        if vert_place is not None:
            _pitch_holding(sim, vert_place["wrist_flex"], closed_g, tid)
            result["place_hover_tilt_deg"] = round(vert_place["tilt_deg"], 2)
        _, R_place = sim.site_pose(EE_SITE)
        place_axis = np.asarray(R_place, dtype=float).reshape(3, 3)[:, 0]
        keep_place = place_axis if gripper_tilt_deg(sim) > 6.0 else None
        # Destination clearance: which side the moving jaw faces (roll, level
        # pinch either way) and how wide to open without slapping a neighbour.
        place_plan = _plan_place(sim, tid, dest_xy, place_z, hold, release_open, x_axis=keep_place)
        result["place_plan"] = place_plan
        d_roll = abs(float(place_plan["roll"]) - float(sim.qpos()["wrist_roll"]))
        if 0.05 < d_roll < 1.0:
            _roll_holding(sim, float(place_plan["roll"]))
        # If the grasp was not already world-down, freeze that axis so the jaws
        # translate instead of swinging the piece onto its side (f1c4).
        vertical_place = _lower_vertical(sim, tid, dest_xy, place_z, hold, result, x_axis=keep_place)
        if vertical_place:
            place = vertical_place
        else:
            align = [float(gpos[0] + shift[0]), float(gpos[1] + shift[1]), float(gpos[2])]
            place = [align[0], align[1], place_z]
            if float(np.hypot(shift[0], shift[1])) > 0.001:
                if _carry_lin(sim, align, 0.5, 6, hold, extract):
                    gpos, _ = sim.site_pose(EE_SITE)
                    q_hold, _ = sim.object_pose(tid)
                    shift = dest_xy[:2] - q_hold[:2]
                    place = [
                        float(gpos[0] + shift[0]),
                        float(gpos[1] + shift[1]),
                        place_z,
                    ]
            result["place_diag"]["lower_target_xy_mm"] = [round(place[0] * 1000, 1), round(place[1] * 1000, 1)]
            lower_ok = _carry_lin(sim, place, 1.2, 12, hold, extract)
            if not lower_ok:
                result["failure_reason"] = "IK_FAILURE"
                result["ik_fail_phase"] = "008_lower"
                log_state(sim, "fail_lower", extra={"target": _xyz(place)})
                return Fail("carry", result["failure_reason"])
        q_lo, _ = sim.object_pose(tid)
        off = float(np.hypot(q_lo[0] - dest_xy[0], q_lo[1] - dest_xy[1]))
        if standoff and 0.012 < off < 0.022:
            gpos, _ = sim.site_pose(EE_SITE)
            shift = dest_xy[:2] - q_lo[:2]
            inch = [float(gpos[0] + shift[0]), float(gpos[1] + shift[1]), float(gpos[2])]
            if _carry_lin(sim, inch, 0.8, 8, hold, extract):
                place = inch
                result["place_diag"]["standoff_inch"] = True
    log_state(sim, "008_lower", extra={"target": _xyz(place)})


@register("carry", "plumb_then_legacy")
class PlumbThenLegacy:
    """Lift, transport and lower over the destination: plumb plan first, legacy lift/transport/lower otherwise."""

    def carry(self, m):
        return carry(m)


@register("carry", "legacy_only")
class LegacyOnly:
    """A/B variant: skip the plumb plan, always the legacy lift/transport/lower."""

    def carry(self, m):
        return carry(m, use_plumb=False)


@register("carry", "plumb_only")
class PlumbOnly:
    """A/B variant: no legacy fallback when plumb finds no plan (the move fails IK_NO_SOLUTION)."""

    def carry(self, m):
        return carry(m, legacy_fallback=False)
