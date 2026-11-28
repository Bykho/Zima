"""Approach stage: plan a collision-free vertical approach, reach pregrasp, re-look, descend to the grasp."""

from __future__ import annotations

import numpy as np

import carry as plumb
from clearance import HARD_MARGIN, JAW_BODIES, _ROLL_GRID_RAD, obstacle_bodies, plan_approach, verify_approach
from ik import EE_SITE, gripper_tilt_deg, joint_limit_margin, move_ik, move_lin, solve_ik_down
from mj import ARM_JOINTS, JOINT_NAMES, Sim
import pickup
from pickup import (
    _drive_arm,
    _grasp_xy_for_roll,
    _safe_lin,
    _tid,
    _transit_z,
    _xyz,
    bystander_contact,
    locate_queen,
    log_state,
    min_jaw_gap,
    open_gripper,
    queen_tilt_deg,
    queen_upright,
)
from stages import register
from stages.types import Fail, Move


_COLUMN_TILT_DEG = 2.0


def _approach_contact(desc: dict) -> bool:
    """True if descend/IK goal involved target or environment contact."""
    if desc.get("first_contact_jaw"):
        return True
    if int(desc.get("pre_close_contacts") or 0) > 0:
        return True
    if float(desc.get("queen_xy_mm") or 0.0) > 1.0:
        return True
    ikj = desc.get("ik_jaw") or {}
    if int(ikj.get("n_contact") or 0) > 0:
        return True
    gap = ikj.get("min_gap_mm")
    if gap is not None and float(gap) < 0.0:
        return True
    return False


def _contact_jaws(hits: list[dict]) -> list[str]:
    names = []
    if any(h["other"] == "gripper" for h in hits):
        names.append("fixed")
    if any(h["other"] == "moving_jaw_so101_v1" for h in hits):
        names.append("moving")
    return names


def _descend_column(sim: Sim, grasp, lock: tuple[str, ...]) -> list[dict] | None:
    """Turn the claw to its grasp angle at this height, then plan straight down.

    The claw only changes angle up here: rotating on the way down sweeps
    the jaws sideways through the piece. Returns the vertical joint chain
    (the arm is already over the grasp at the grasp angle), or None with
    nothing moved.
    """
    snap = sim.qpos()
    with sim.probe():
        try:
            if not solve_ik_down(sim, grasp, lock_joints=lock):
                return None
            _, R = sim.site_pose(EE_SITE)
            v_local = np.asarray(R, dtype=float).reshape(3, 3).T @ plumb.UP
            sim.set_qpos(snap, hold=True)
            gpos, _ = sim.site_pose(EE_SITE)
            above = [float(grasp[0]), float(grasp[1]), float(gpos[2])]
            if not plumb.solve_plumb(sim, above, v_local, _COLUMN_TILT_DEG):
                return None
            q_above = {n: float(sim.qpos()[n]) for n in ARM_JOINTS}
            down = plumb.plan(sim, [(grasp, _COLUMN_TILT_DEG)], v_local, [])
        finally:
            sim.set_qpos(snap, hold=True)
    if down is None:
        return None
    _follow_q_path(sim, [q_above], 0.6)
    return down


def _descend_fail_reason(desc: dict) -> str:
    if not desc.get("ik_solved"):
        return "IK_NO_SOLUTION"
    if _approach_contact(desc):
        return "APPROACH_COLLISION"
    return "APPROACH_TRACK_FAIL"


def _follow_q_path(sim: Sim, path_q: list[dict], duration: float, guard=None) -> bool:
    """Track a list of joint configurations with ctrl only, guard checked every step."""
    segs = max(1, len(path_q) - 1)
    n_total = max(30, int(duration / sim.dt))
    per = max(4, n_total // segs)
    cur = sim.qpos()
    for goal in path_q:
        for i in range(per):
            u = (i + 1) / per
            cmd = {k: cur[k] + (goal.get(k, cur[k]) - cur[k]) * u for k in cur}
            sim.set_ctrl(cmd)
            sim.step()
            if guard is not None:
                hit = guard()
                if hit:
                    sim._last_abort = hit
                    sim.hold(sim.qpos())
                    return False
        cur = {k: goal.get(k, cur[k]) for k in cur}
    sim.hold(cur)
    sim.step(20)
    return True


def _physical_descend_path(sim: Sim, target, path_q: list[dict], duration: float) -> dict:
    """Replay the planner's verified joint path (pre -> grasp) with the bystander guard."""
    tid = _tid(sim)
    q0, _ = sim.object_pose(tid)
    target_ee = np.asarray(target, dtype=np.float64)
    bystanders = [n for n in obstacle_bodies(sim, skip={tid}) if n != "board"]
    sim._last_abort = None
    guard = (lambda: bystander_contact(sim, bystanders)) if bystanders else None
    _follow_q_path(sim, path_q, duration, guard=guard)
    aborted = getattr(sim, "_last_abort", None)
    pos, _ = sim.site_pose(EE_SITE)
    qpos, qquat = sim.object_pose(tid)
    err_mm = float(np.linalg.norm(pos - target_ee)) * 1000
    hits = jaw_contacts(sim, tid)
    return {
        "ok": err_mm < 20.0 and not aborted,
        "aborted": aborted,
        "tilt_deg": gripper_tilt_deg(sim),
        "ik_solved": True,
        "first_contact_step": None,
        "first_contact_jaw": _contact_jaws(hits) if hits else None,
        "peak_force": {n: 0.0 for n in JOINT_NAMES},
        "samples": [],
        "ee_requested": _xyz(target_ee),
        "ee_actual": _xyz(pos),
        "ee_err_mm": err_mm,
        "ik_residual_mm": err_mm,
        "approach_reached": err_mm < 8.0,
        "joint_limit_margin_rad": joint_limit_margin(sim),
        "ik_jaw": jaw_snapshot(sim, tid),
        "queen_xy_mm": float(np.linalg.norm(qpos[:2] - q0[:2])) * 1000,
        "queen_tilt_deg": queen_tilt_deg(qquat),
        "pre_close_contacts": len(hits),
        "pre_close_gap_mm": min_jaw_gap(sim, tid) * 1000,
        "jaw_before_close": jaw_snapshot(sim, tid),
        "saturated": [],
        "path_replay": True,
    }


# A servo at its force limit that has left the pose. The far-file look holds
# with the shoulder there, so the shoulder is not a reason to change rolls.
# The a-file elbow and wrist flex leave the pose; that grasp does not hold.
_CLIP_SAG_RAD = 0.01
_CLIP_HOLD_STEPS = 80
_CLIP_JOINTS = ("elbow_flex", "wrist_flex")


def _roll_choices(sim: Sim, prefer: float) -> list[float]:
    lo, hi = sim.joint_limits()["wrist_roll"]
    grid = np.arange(float(lo) + 0.05, float(hi) - 0.05 + 1e-9, _ROLL_GRID_RAD)
    rolls = [float(prefer)]
    for roll in sorted((float(r) for r in grid), key=lambda r: abs(r - prefer)):
        if all(abs(roll - have) > 0.15 for have in rolls):
            rolls.append(roll)
    return rolls


def _scratch_grasp(sim: Sim, roll: float, piece_xy, z: float, shift: float):
    """IK the grasp at `roll` on a private copy. 'hold', 'clip', or 'miss'.

    The live arm is not written. A pose is on the clip only when elbow_flex
    or wrist_flex is at its force limit and has sagged off the solution.
    """
    import mujoco

    model = sim.model
    scratch = mujoco.MjData(model)
    mujoco.mj_copyData(scratch, model, sim.data)
    live = sim.data
    sim.data = scratch
    try:
        jid = sim._joint["wrist_roll"]
        lo, hi = model.jnt_range[jid]
        scratch.qpos[int(model.jnt_qposadr[jid])] = float(np.clip(roll, lo, hi))
        mujoco.mj_kinematics(model, scratch)
        seed = [float(piece_xy[0]), float(piece_xy[1]), float(z)]
        if not solve_ik_down(sim, seed, lock_joints=("wrist_roll",)):
            return "miss", None
        _, rot = sim.site_pose(EE_SITE)
        xy = [float(piece_xy[0] + rot[0, 2] * shift), float(piece_xy[1] + rot[1, 2] * shift)]
        if not solve_ik_down(sim, [xy[0], xy[1], float(z)], lock_joints=("wrist_roll",)):
            return "miss", None
        goal = {}
        for name in _CLIP_JOINTS:
            qadr = int(model.jnt_qposadr[sim._joint[name]])
            goal[name] = float(scratch.qpos[qadr])
            scratch.ctrl[sim._act[name]] = scratch.qpos[qadr]
        for name in ARM_JOINTS:
            if name in _CLIP_JOINTS:
                continue
            qadr = int(model.jnt_qposadr[sim._joint[name]])
            scratch.ctrl[sim._act[name]] = scratch.qpos[qadr]
        scratch.qvel[:] = 0.0
        for _ in range(_CLIP_HOLD_STEPS):
            mujoco.mj_step(model, scratch)
        for name in _CLIP_JOINTS:
            aid = sim._act[name]
            lo, hi = model.actuator_forcerange[aid]
            limit = float(min(abs(lo), abs(hi)))
            if abs(float(scratch.actuator_force[aid])) < limit - 1e-3:
                continue
            qadr = int(model.jnt_qposadr[sim._joint[name]])
            if abs(float(scratch.qpos[qadr]) - goal[name]) > _CLIP_SAG_RAD:
                return "clip", xy
        return "hold", xy
    finally:
        sim.data = live


def _roll_that_holds(sim: Sim, piece_xy, z: float, geom, prefer: float):
    """Keep `prefer` unless that grasp is on the elbow/wrist clip.

    Returns ('keep', None) when the current roll stays, ('roll', (roll, xy))
    when another roll holds, or ('none', None) when every roll is on the clip.
    """
    import mujoco

    live = sim.data
    qpos = live.qpos.copy()
    qvel = live.qvel.copy()
    ctrl = live.ctrl.copy()
    posed = getattr(sim, "_posed", None)
    try:
        shift = pickup._pinch_shift_m(sim, geom)
        kind, _xy = _scratch_grasp(sim, prefer, piece_xy, z, shift)
        if kind != "clip":
            return "keep", None
        for roll in _roll_choices(sim, prefer):
            if abs(roll - prefer) <= 0.15:
                continue
            kind, xy = _scratch_grasp(sim, roll, piece_xy, z, shift)
            if kind == "hold":
                return "roll", (float(roll), xy)
        return "none", None
    finally:
        if sim.data is not live:
            sim.data = live
        if (
            not np.array_equal(live.qpos, qpos)
            or not np.array_equal(live.qvel, qvel)
            or not np.array_equal(live.ctrl, ctrl)
        ):
            live.qpos[:] = qpos
            live.qvel[:] = qvel
            live.ctrl[:] = ctrl
            mujoco.mj_forward(sim.model, live)
        if getattr(sim, "_posed", None) != posed:
            sim._posed = posed


def _roll_in_place(sim: Sim, roll_goal: float, rad_per_s: float = 2.0) -> None:
    """Turn wrist_roll to `roll_goal` under physics, the other arm joints held at their current targets.

    A 180 deg flip in 0.35 s saturated the shoulder/elbow servos and the arm
    sagged onto the rank below (jaw swept c2/d2/e2 at ee_z=25 mm), so the turn
    runs at a bounded wrist speed in short legs and settles before returning.
    """
    q0 = {j: float(sim.qpos()[j]) for j in ARM_JOINTS}
    d = roll_goal - q0["wrist_roll"]
    if abs(d) < 1e-3:
        return
    legs = max(1, int(np.ceil(abs(d) / 0.4)))
    path = [{**q0, "wrist_roll": q0["wrist_roll"] + d * i / legs} for i in range(1, legs + 1)]
    _drive_arm(sim, path, float(sim.qpos()["gripper"]), duration=max(0.3, abs(d) / rad_per_s))
    for _ in range(4):
        if abs(float(sim.qpos()["wrist_roll"]) - roll_goal) <= 0.05:
            break
        sim.step(20)


def jaw_contacts(sim: Sim, object_id: str | None = None) -> list[dict]:
    tid = object_id or _tid(sim)
    return [
        c
        for c in sim.contacts_with(tid)
        if c["other"] in JAW_BODIES and (c["force"] > 0.02 or c["dist"] < 0.0)
    ]


def jaw_snapshot(sim: Sim, object_id: str | None = None) -> dict:
    """Open-jaw geometry relative to the target. Evaluation only."""
    tid = object_id or _tid(sim)
    qpos, _ = sim.object_pose(tid)
    ee, _ = sim.site_pose(EE_SITE)
    fix = sim.geom_xpos("fixed_jaw_sph_tip1")
    mov = sim.geom_xpos("moving_jaw_sph_tip1")
    pinch = 0.5 * (fix + mov)
    hits = jaw_contacts(sim, tid)
    return {
        "ee_xy_mm": [float(ee[0] * 1000), float(ee[1] * 1000)],
        "queen_xy_mm": [float(qpos[0] * 1000), float(qpos[1] * 1000)],
        "fixed_tip_xy_mm": [float(fix[0] * 1000), float(fix[1] * 1000)],
        "moving_tip_xy_mm": [float(mov[0] * 1000), float(mov[1] * 1000)],
        "pinch_xy_mm": [float(pinch[0] * 1000), float(pinch[1] * 1000)],
        "fixed_to_queen_xy_mm": [float((fix[0] - qpos[0]) * 1000), float((fix[1] - qpos[1]) * 1000)],
        "moving_to_queen_xy_mm": [float((mov[0] - qpos[0]) * 1000), float((mov[1] - qpos[1]) * 1000)],
        "pinch_to_queen_xy_mm": [float((pinch[0] - qpos[0]) * 1000), float((pinch[1] - qpos[1]) * 1000)],
        "jaw_gap_mm": float(sim.jaw_gap() * 1000),
        "min_gap_mm": float(min_jaw_gap(sim, tid) * 1000),
        "contact_jaws": _contact_jaws(hits),
        "n_contact": len(hits),
    }


def physical_descend(sim: Sim, target, duration: float = 1.2, lock_joints=(), path_q=None) -> dict:
    """Joint-space interpolation with actuator commands only (no qpos overwrite).

    path_q: planner-verified joint path (pre -> grasp); replayed as-is.
    """
    if path_q:
        return _physical_descend_path(sim, target, list(path_q), duration)
    quiet = getattr(sim, "quiet", False)
    tid = _tid(sim)
    start = sim.qpos()
    q0, _ = sim.object_pose(tid)
    target_ee = np.asarray(target, dtype=np.float64)
    report = {
        "ok": False,
        "ik_solved": False,
        "first_contact_step": None,
        "first_contact_jaw": None,
        "peak_force": {n: 0.0 for n in JOINT_NAMES},
        "samples": [],
        "ee_requested": _xyz(target_ee),
    }
    with sim.probe():
        solved = solve_ik_down(sim, target, lock_joints=lock_joints)
        pos, _ = sim.site_pose(EE_SITE)
        report["joint_limit_margin_rad"] = joint_limit_margin(sim)
        if not solved:
            report["ee_actual"] = _xyz(pos)
            report["ee_err_mm"] = float(np.linalg.norm(pos - target_ee)) * 1000
            report["ik_residual_mm"] = report["ee_err_mm"]
        else:
            goal = sim.qpos()
            report["ik_solved"] = True
            report["tilt_deg"] = gripper_tilt_deg(sim)
            report["ik_residual_mm"] = float(np.linalg.norm(pos - target_ee)) * 1000
            report["ik_jaw"] = jaw_snapshot(sim, tid)
        sim.set_qpos(start, hold=True)
    if not solved:
        return report
    n = max(30, int(duration / sim.dt))
    peak = np.zeros(len(JOINT_NAMES))
    for i in range(n):
        u = (i + 1) / n
        u = u * u * (3 - 2 * u)
        cmd = {k: start[k] + (goal[k] - start[k]) * u for k in start}
        sim.set_ctrl(cmd)
        sim.step()
        force = sim.actuator_forces()
        np.maximum(peak, np.abs(force), out=peak)
        hits = jaw_contacts(sim, tid)
        if hits and report["first_contact_step"] is None:
            report["first_contact_step"] = i
            report["first_contact_jaw"] = _contact_jaws(hits)
        if i % 40 == 0 or i == n - 1 or hits:
            pos, _ = sim.site_pose(EE_SITE)
            qpos, qquat = sim.object_pose(tid)
            actual = sim.qpos()
            sample = {
                "t": round((i + 1) * sim.dt, 3),
                "ee_target": _xyz(target_ee),
                "ee_actual": _xyz(pos),
                "ee_err_mm": round(float(np.linalg.norm(pos - target_ee)) * 1000, 2),
                "joint_err": {k: round(actual[k] - cmd[k], 4) for k in ARM_JOINTS},
                "force": {n: round(float(force[j]), 3) for j, n in enumerate(JOINT_NAMES)},
                "jaw_gap_mm": round(min_jaw_gap(sim, tid) * 1000, 2),
                "n_jaw_contact": len(hits),
                "queen_xy_disp_mm": round(float(np.linalg.norm(qpos[:2] - q0[:2])) * 1000, 2),
                "queen_tilt_deg": round(queen_tilt_deg(qquat), 2),
            }
            report["samples"].append(sample)
            if not quiet:
                fmax = max(abs(v) for v in sample["force"].values())
                print(
                    f"  descend t={sample['t']:.2f}s ee_err={sample['ee_err_mm']:.1f}mm "
                    f"jaw_gap={sample['jaw_gap_mm']:.2f}mm ncon={sample['n_jaw_contact']} "
                    f"q_xy={sample['queen_xy_disp_mm']:.2f}mm fmax={fmax:.2f}"
                )
    sim.hold(goal)
    for _ in range(20):
        sim.step()
        force = np.abs(sim.actuator_forces())
        np.maximum(peak, force, out=peak)
        if jaw_contacts(sim, tid) and report["first_contact_step"] is None:
            report["first_contact_step"] = n
            report["first_contact_jaw"] = _contact_jaws(jaw_contacts(sim, tid))
    pos, _ = sim.site_pose(EE_SITE)
    qpos, qquat = sim.object_pose(tid)
    report["ok"] = float(np.linalg.norm(pos - target_ee)) < 0.02
    report["ee_actual"] = _xyz(pos)
    report["ee_err_mm"] = float(np.linalg.norm(pos - target_ee)) * 1000
    report["approach_reached"] = report["ee_err_mm"] < 8.0
    report["peak_force"] = {n: float(peak[j]) for j, n in enumerate(JOINT_NAMES)}
    report["queen_xy_mm"] = float(np.linalg.norm(qpos[:2] - q0[:2])) * 1000
    report["queen_tilt_deg"] = queen_tilt_deg(qquat)
    report["pre_close_contacts"] = len(jaw_contacts(sim, tid))
    report["pre_close_gap_mm"] = min_jaw_gap(sim, tid) * 1000
    report["jaw_before_close"] = jaw_snapshot(sim, tid)
    report["saturated"] = [n for n, v in report["peak_force"].items() if v >= 2.94 - 1e-6]
    if not quiet:
        print(
            f"  descend done ee_err={report['ee_err_mm']:.1f}mm "
            f"pre_close_ncon={report['pre_close_contacts']} gap={report['pre_close_gap_mm']:.2f}mm "
            f"queen_xy={report['queen_xy_mm']:.2f}mm sat={report['saturated']}"
        )
    return report


def run(m: Move, planner=plan_approach):
    """Plan the approach, reach pregrasp, re-look and re-verify, descend to the grasp."""
    sim, cfg, sc, result, tid, surface, source, use_vision, queen_xyz, grasp_z, geom, open_g, plan_xy, grasp_xy, pre, actual = m.sim, m.cfg, m.sc, m.result, m.tid, m.surface, m.source, m.use_vision, m.queen_xyz, m.grasp_z, m.geom, m.open_g, m.plan_xy, m.grasp_xy, m.pre, m.actual
    clutter = [n for n in obstacle_bodies(sim, skip={tid}) if n != "board"]
    result["occupied_neighbors"] = sorted(clutter)
    grasp_plan = [grasp_xy[0], grasp_xy[1], surface + grasp_z]

    def _grasp_for_roll(roll: float | None) -> list[float]:
        # The pinch axis rotates with wrist roll, so the fixed-pad shift does too.
        xy = _grasp_xy_for_roll(sim, plan_xy, surface + grasp_z, geom, roll)
        return [xy[0], xy[1], surface + grasp_z]

    def _plan(pre_seed, grasp_seed):
        return planner(
            sim,
            pre_seed,
            grasp_seed,
            open_g,
            clutter,
            source=None if sc.get("_from_nest") else source,
            cfg=cfg,
            scenario=sc,
            grasp_for_roll=_grasp_for_roll,
            target=tid,
        )

    approach = _plan(pre, grasp_plan)
    if approach.get("pre"):
        pre = list(approach["pre"])
    if approach.get("grasp"):
        grasp_plan = list(approach["grasp"])
        grasp_xy = grasp_plan[:2]
    _plan_keys = ("ok", "roll", "gripper", "n_wp", "tried", "reachable", "rank", "reason", "pre_offset_xy", "block", "statuses", "degraded", "min_gap_mm")
    result["approach_plan"] = {k: approach.get(k) for k in _plan_keys}
    if not approach["ok"]:
        result["failure_reason"] = (
            "NO_COLLISION_FREE_APPROACH" if approach.get("reachable") else "IK_NO_SOLUTION"
        )
        log_state(sim, "fail_pregrasp", extra={"target": _xyz(pre), "approach": result["approach_plan"]})
        return Fail("approach", result["failure_reason"], "pick")
    lock: tuple[str, ...] = tuple(approach["lock_joints"])
    # After close, lock wrist_roll so the pinch does not yaw around the stem.
    # Leave wrist_flex free: carry IK pulls gripper +X toward world -Z so the
    # piece stays upright as the shoulder/elbow reach across the board.
    hold: tuple[str, ...] = tuple(
        j for j in dict.fromkeys((*lock, "wrist_roll")) if j != "wrist_flex"
    )
    pickup_clear = bool(clutter) or approach["n_wp"] > 1
    # Verified joint path for the descend (pre -> grasp); replayed verbatim.
    descend_path = approach.get("path_q") if clutter else None
    # Vertical pinch: pre, descend and grasp are all solved with gripper +X
    # world-down (solve_ik_down); flex is a free joint the solver uses to keep
    # the approach vertical while shoulder/elbow move. Roll stays locked when
    # the planner chose one for neighbor clearance.
    grasp_lock = tuple(j for j in lock if j != "wrist_flex")

    def _apply_approach(plan: dict) -> bool:
        """Set the planned roll + aperture, then bring the EE to its pregrasp."""
        if plan["roll"] is not None:
            _roll_in_place(sim, float(plan["roll"]))
            result["approach_roll_err"] = round(abs(float(sim.qpos()["wrist_roll"]) - float(plan["roll"])), 3)
        g_ap = float(plan["gripper"])
        result["approach_gripper"] = None if abs(g_ap - open_g) < 0.02 else g_ap
        if abs(g_ap - float(sim.qpos()["gripper"])) > 0.01:
            open_gripper(sim, g_ap, steps=40)
        # A straight line to the pregrasp from the last square (g4) sweeps the
        # file and knocks whoever is in the way (fallen Nf3, then the e5 pawn
        # → APPROACH_COLLISION). Hover over the target first so the pregrasp
        # move is a short local drop.
        gpos, _ = sim.site_pose(EE_SITE)
        pre_xyz = np.asarray(plan["pre"], dtype=float)
        if float(np.hypot(pre_xyz[0] - gpos[0], pre_xyz[1] - gpos[1])) > 0.05:
            hover_z = max(float(pre_xyz[2]), _transit_z(sim, cfg, skip={tid}))
            hover = [float(pre_xyz[0]), float(pre_xyz[1]), hover_z]
            _safe_lin(
                sim,
                hover,
                cfg,
                ("wrist_flex", "wrist_roll"),
                obstacles=clutter,
                skip={tid},
            )
        if pickup_clear:
            return move_lin(sim, plan["pre"], duration=1.0, physics=True, n_wp=6, lock_joints=grasp_lock, down=True)
        return move_ik(sim, plan["pre"], duration=1.0, physics=True, lock_joints=grasp_lock, down=True)

    if not _apply_approach(approach):
        result["failure_reason"] = "IK_NO_SOLUTION"
        log_state(sim, "fail_pregrasp", extra={"target": _xyz(pre)})
        return Fail("approach", result["failure_reason"], "pick")
    result["pregrasp_reached"] = True
    log_state(sim, "003_pregrasp", extra={"target": _xyz(pre), "approach_gripper": result.get("approach_gripper"), "approach_roll": approach.get("roll")})

    if not queen_upright(sim.object_pose(tid)[1]):
        result["failure_reason"] = "APPROACH_COLLISION"
        log_state(sim, "fail_pregrasp_knock")
        return Fail("approach", result["failure_reason"], "pick")

    if use_vision and not sc.get("_from_nest"):
        refined = locate_queen(sim, cfg, result, source)
        if refined is not None:
            queen_xyz = refined
            # Looked up on the module: the perception_error suite wraps it to inject error.
            plan_xy = pickup.perceived_xy(sim, queen_xyz)
            grasp_xy = _grasp_xy_for_roll(sim, plan_xy, surface + grasp_z, geom, approach.get("roll"))
            # The plan was verified against the long-range estimate. Re-check
            # the chosen approach on the refined grasp; if the jaws would now
            # clip a bystander, re-plan from here before going down.
            if clutter:
                status = verify_approach(
                    sim, pre, [grasp_xy[0], grasp_xy[1], surface + grasp_z], clutter,
                    float(approach["gripper"]), approach.get("roll"), lock, cfg=cfg, target=tid,
                    margin=HARD_MARGIN if approach.get("degraded") else None,
                )
                result["approach_reverify"] = status
                if status == "ok":
                    descend_path = getattr(sim, "_cand_path", None) or descend_path
                if status != "ok":
                    again = _plan(pre, [grasp_xy[0], grasp_xy[1], surface + grasp_z])
                    result["approach_replan"] = {k: again.get(k) for k in _plan_keys}
                    if not again["ok"]:
                        result["failure_reason"] = (
                            "NO_COLLISION_FREE_APPROACH" if again.get("reachable") else "IK_NO_SOLUTION"
                        )
                        log_state(sim, "fail_pregrasp", extra={"target": _xyz(pre), "approach": result["approach_replan"]})
                        return Fail("approach", result["failure_reason"])
                    approach = again
                    lock = tuple(approach["lock_joints"])
                    hold = tuple(j for j in dict.fromkeys((*lock, "wrist_roll")) if j != "wrist_flex")
                    grasp_lock = tuple(j for j in lock if j != "wrist_flex")
                    pre = list(approach["pre"])
                    grasp_xy = list(approach["grasp"][:2])
                    descend_path = approach.get("path_q")
                    result["approach_plan"] = {k: approach.get(k) for k in _plan_keys}
                    if not _apply_approach(approach):
                        result["failure_reason"] = "IK_NO_SOLUTION"
                        log_state(sim, "fail_pregrasp", extra={"target": _xyz(pre)})
                        return Fail("approach", result["failure_reason"])
                    log_state(sim, "003_pregrasp", extra={"target": _xyz(pre), "replanned": True})

    result["pose_used_for_grasp"] = [float(plan_xy[0]), float(plan_xy[1]), float(np.asarray(queen_xyz)[2])]
    result["perception_error"] = [
        float((plan_xy[0] - actual[0]) * 1000.0),
        float((plan_xy[1] - actual[1]) * 1000.0),
    ]
    prefer_roll = approach.get("roll")
    if prefer_roll is not None:
        decision, found = _roll_that_holds(
            sim, plan_xy, surface + grasp_z, geom, float(prefer_roll)
        )
        if decision == "none":
            result["failure_reason"] = "IK_NO_SOLUTION"
            log_state(sim, "fail_pregrasp", extra={"target": _xyz(pre), "clip": True})
            return Fail("approach", result["failure_reason"], "pick")
        if decision == "roll":
            roll, xy = found
            _roll_in_place(sim, roll)
            grasp_xy = list(xy)
            if "wrist_roll" not in grasp_lock:
                grasp_lock = tuple(dict.fromkeys((*grasp_lock, "wrist_roll")))
            descend_path = None
            result["grasp_roll"] = float(roll)
    grasp = [grasp_xy[0], grasp_xy[1], surface + grasp_z]
    result["ee_requested"] = _xyz(grasp)
    column = _descend_column(sim, grasp, grasp_lock)
    result["descend_column"] = column is not None or getattr(sim, "plumb_fail", None)
    desc = physical_descend(
        sim, grasp, duration=1.2, lock_joints=grasp_lock, path_q=column or descend_path
    )
    # Guard tripped: a robot link touched a bystander on the way down.
    if desc.get("aborted"):
        result["descend_aborted"] = desc["aborted"]
        move_lin(sim, pre, duration=0.6, physics=True, n_wp=4, lock_joints=grasp_lock, down=True)
        result["failure_reason"] = "APPROACH_COLLISION"
        log_state(sim, "fail_descend", extra={"target": _xyz(grasp), "aborted": desc["aborted"]})
        return Fail("approach", result["failure_reason"])
    result["descend"] = {k: v for k, v in desc.items() if k not in ("samples", "ik_jaw", "jaw_before_close")}
    result["ik_solved"] = bool(desc.get("ik_solved"))
    result["ik_residual_mm"] = desc.get("ik_residual_mm")
    result["ee_actual"] = desc.get("ee_actual")
    result["ee_err_mm"] = desc.get("ee_err_mm")
    result["approach_reached"] = bool(desc.get("approach_reached"))
    result["joint_limit_margin_rad"] = desc.get("joint_limit_margin_rad")
    result["pre_close_contacts"] = desc.get("pre_close_contacts", 0)
    result["pre_close_gap_mm"] = desc.get("pre_close_gap_mm")
    result["first_contact_jaw"] = desc.get("first_contact_jaw")
    result["ik_jaw"] = desc.get("ik_jaw")
    result["jaw_before_close"] = desc.get("jaw_before_close")
    if not desc.get("ik_solved") or not desc["ok"]:
        result["failure_reason"] = _descend_fail_reason(desc)
        log_state(sim, "fail_descend", extra={"target": _xyz(grasp), "ee_err_mm": desc.get("ee_err_mm")})
        return Fail("approach", result["failure_reason"])
    log_state(
        sim,
        "004_descend",
        extra={
            "target": _xyz(grasp),
            "ee_err_mm": round(desc["ee_err_mm"], 2),
            "pre_close_contacts": desc["pre_close_contacts"],
            "pre_close_gap_mm": round(desc["pre_close_gap_mm"], 2),
            "queen_xy_mm": round(desc["queen_xy_mm"], 2),
            "saturated": desc["saturated"],
        },
    )

    if not queen_upright(sim.object_pose(tid)[1]):
        result["failure_reason"] = "APPROACH_COLLISION"
        log_state(sim, "fail_descend_knock")
        return Fail("approach", result["failure_reason"], "pick")
    m.plan_xy = plan_xy
    m.grasp = grasp
    m.hold = hold
    m.clutter = clutter


@register("approach", "vertical_column")
class VerticalColumn:
    """Collision-free vertical approach, pregrasp, re-look and re-verify, descend to the grasp."""

    def plan(self, sim, pre, grasp, open_g, clutter, **kw) -> dict:
        """Collision-free pregrasp/roll/aperture for a grasp; also how unstack judges blockers."""
        return plan_approach(sim, pre, grasp, open_g, clutter, **kw)

    def run(self, m):
        return run(m, planner=self.plan)
