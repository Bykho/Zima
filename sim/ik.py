"""Damped-least-squares IK for the 5-DoF arm.

SO-101 cannot reach arbitrary 6-DoF poses. Position is required.
A soft pull on gripperframe +X toward world -Z is optional; it must not
override reaching the point. Rotation about the approach axis is free.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np

from fast import enabled as fast_enabled, motion_scale
from mj import ARM_JOINTS, Sim

EE_SITE = "gripperframe"
DOWN = np.array([0.0, 0.0, -1.0])
# Carry/place: after position IK, nudge wrist_flex toward gripper +X || world -Z.
# The 6-DoF stacked Jacobian slammed flex in one waypoint (elbow flip) and
# physics tracking dropped the piece. One bounded step per solve keeps the
# same IK family; many waypoints accumulate a full straighten.
CARRY_ORI_WEIGHT = 0.12
CARRY_FLEX_STEP = 0.20
# Unreachable targets plateau at the workspace boundary; stop iterating once
# the residual has not improved by 0.01 mm for this many iterations.
_STALL_EPS = 1e-5
_STALL_ITERS = 20


def grasp_xy_delta(grot, off) -> np.ndarray:
    """Map gripper-frame (ox, oy) into world XY.

    ``R[:,0]*ox + R[:,1]*oy`` is 3D. Gripper X is the approach axis (mostly
    world down), so its table-plane image is the small tilt remainder and
    lies anti-parallel to Y's image. A steeper wrist shortens that remainder
    and leaves oy unopposed; d2 then pinches on the wrong side of the jaws.

    Keep the 3D axes' XY projections (do not unit-normalize — that cancels —
    and do not replace X with the perpendicular — that dumps full ox onto
    the table and overshoots). Floor X's projected length so a near-vertical
    approach cannot drop ox out of the table plane.
    """
    ox, oy = float(off[0]), float(off[1])
    gx = np.array([float(grot[0, 0]), float(grot[1, 0])])
    gy = np.array([float(grot[0, 1]), float(grot[1, 1])])
    nx = float(np.linalg.norm(gx))
    # sin(30°). Steeper than this, ox's XY image is too short to balance oy.
    min_nx = 0.5
    if nx < 1e-6:
        gx = np.array([float(grot[0, 2]), float(grot[1, 2])])
        nx = float(np.linalg.norm(gx))
    if nx > 1e-6 and nx < min_nx:
        gx = gx * (min_nx / nx)
    return gx * ox + gy * oy


def _solve_position(
    sim: Sim,
    target: np.ndarray,
    lock: tuple[str, ...],
    sid: int,
    max_iters: int,
    pos_tol: float,
    damping: float,
    step: float,
) -> bool:
    import mujoco

    model, data = sim.model, sim.data
    names = [n for n in ARM_JOINTS if n not in lock]
    jids = [sim._joint[n] for n in names]
    dof_ids = np.array([int(model.jnt_dofadr[j]) for j in jids], dtype=int)
    qadr = np.array([int(model.jnt_qposadr[j]) for j in jids], dtype=int)
    lo = model.jnt_range[jids, 0]
    hi = model.jnt_range[jids, 1]
    jacp = np.zeros((3, model.nv))
    eye = damping * np.eye(3)
    # Kinematics-only forward: the solver needs site_xpos and cdof, not
    # collision detection over the whole board (that was ~90% of IK time).
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    best = float("inf")
    since_best = 0
    for _ in range(max_iters):
        err_pos = target - data.site_xpos[sid]
        err = float(np.linalg.norm(err_pos))
        if err < pos_tol:
            break
        if err < best - _STALL_EPS:
            best = err
            since_best = 0
        else:
            since_best += 1
            if since_best >= _STALL_ITERS:
                break
        mujoco.mj_jacSite(model, data, jacp, None, sid)
        J = jacp[:, dof_ids]
        dq = J.T @ np.linalg.solve(J @ J.T + eye, err_pos)
        data.qpos[qadr] = np.clip(data.qpos[qadr] + step * dq, lo, hi)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
    # Leave data fully consistent for callers (contacts, forces, sensors).
    mujoco.mj_forward(model, data)
    err = float(np.linalg.norm(target - data.site_xpos[sid]))
    sim.hold({n: sim.qpos()[n] for n in ARM_JOINTS})
    return err < pos_tol * 2.5


def _x_align(sim: Sim, sid: int, axis: np.ndarray) -> float:
    rot = sim.data.site_xmat[sid].reshape(3, 3)
    return float(np.dot(rot[:, 0], axis))


def _nudge_flex_upright(
    sim: Sim,
    target: np.ndarray,
    lock: tuple[str, ...],
    sid: int,
    axis: np.ndarray,
    ori_weight: float,
    max_iters: int,
    pos_tol: float,
    damping: float,
    step: float,
) -> bool:
    """Move wrist_flex at most one step toward +X || axis; re-reach the point."""
    import mujoco

    pos_q = sim.qpos()
    mujoco.mj_forward(sim.model, sim.data)
    best_align = _x_align(sim, sid, axis)
    best_q = dict(pos_q)
    flex0 = float(pos_q["wrist_flex"])
    lo, hi = sim.model.jnt_range[sim._joint["wrist_flex"]]
    df = float(np.clip(ori_weight * 1.67, 0.08, CARRY_FLEX_STEP))
    locked = tuple(dict.fromkeys((*lock, "wrist_flex")))
    for cand in (flex0 + df, flex0 - df):
        q = dict(pos_q)
        q["wrist_flex"] = float(np.clip(cand, lo, hi))
        if abs(q["wrist_flex"] - flex0) < 1e-4:
            continue
        sim.set_qpos(q, hold=True)
        if not _solve_position(sim, target, locked, sid, max_iters, pos_tol, damping, step):
            continue
        mujoco.mj_forward(sim.model, sim.data)
        align = _x_align(sim, sid, axis)
        if align > best_align + 1e-4:
            best_align = align
            best_q = sim.qpos()
    sim.set_qpos(best_q, hold=True)
    return True


def solve_ik(
    sim: Sim,
    target_pos,
    x_axis=DOWN,
    ori_weight: float = 0.0,
    max_iters: int = 200,
    pos_tol: float = 0.003,
    damping: float = 1e-3,
    step: float = 0.5,
    lock_joints: tuple[str, ...] = (),
    ori_steps: int = 1,
) -> bool:
    """Move arm qpos so gripperframe reaches target_pos."""
    import mujoco

    target = np.asarray(target_pos, dtype=np.float64)
    axis = np.asarray(x_axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    sid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    lock = tuple(lock_joints)
    if not _solve_position(sim, target, lock, sid, max_iters, pos_tol, damping, step):
        return False
    if ori_weight > 0 and "wrist_flex" not in lock:
        n = max(1, int(ori_steps))
        for _ in range(n):
            mujoco.mj_forward(sim.model, sim.data)
            before = _x_align(sim, sid, axis)
            _nudge_flex_upright(
                sim, target, lock, sid, axis, ori_weight, max_iters, pos_tol, damping, step
            )
            mujoco.mj_forward(sim.model, sim.data)
            if _x_align(sim, sid, axis) <= before + 1e-4:
                break
    return True


def gripper_tilt_deg(sim: Sim) -> float:
    """Angle between gripper +X (approach axis) and world down."""
    _, rot = sim.site_pose(EE_SITE)
    return float(np.degrees(np.arccos(np.clip(-rot[2, 0], -1.0, 1.0))))


def solve_ik_down(
    sim: Sim,
    target_pos,
    lock_joints: tuple[str, ...] = (),
    max_iters: int = 150,
    pos_tol: float = 0.003,
    tilt_tol_deg: float = 3.0,
    damping: float = 1e-3,
    step: float = 0.6,
    ori_weight: float = 0.05,
    x_axis=None,
    local_axis=None,
) -> bool:
    """Reach target_pos with gripper +X pointing world-down (a vertical pinch).

    x_axis: optional world direction for gripper +X instead of straight down.
    Used to lift with the *current* orientation frozen when a vertical pinch
    was not reachable, so the jaws translate instead of swinging.
    local_axis: gripper-frame vector to align with x_axis instead of +X
    (a held piece's own up axis, to carry it plumb).

    Joint 5-row task: position (3) + the x/y world components of gripper +X
    (2), which are zero when the approach axis is vertical. With pan, lift,
    elbow and flex free this is exactly determined, so it converges wherever
    a vertical pinch exists (files a–f on the starting board) and reports the
    residual tilt otherwise. Roll never enters the task, so it may stay locked
    for neighbor clearance.
    """
    import mujoco

    model, data = sim.model, sim.data
    target = np.asarray(target_pos, dtype=np.float64)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    names = [n for n in ARM_JOINTS if n not in lock_joints]
    jids = [sim._joint[n] for n in names]
    dof_ids = np.array([int(model.jnt_dofadr[j]) for j in jids], dtype=int)
    qadr = np.array([int(model.jnt_qposadr[j]) for j in jids], dtype=int)
    lo = model.jnt_range[jids, 0]
    hi = model.jnt_range[jids, 1]
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    w = float(ori_weight)
    tilt_tol = np.sin(np.radians(tilt_tol_deg))
    axis = None
    if x_axis is not None:
        axis = np.asarray(x_axis, dtype=np.float64)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
    if local_axis is not None:
        local = np.asarray(local_axis, dtype=np.float64)
        local = local / (np.linalg.norm(local) + 1e-12)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    best = float("inf")
    since_best = 0
    for _ in range(max_iters):
        R = data.site_xmat[sid].reshape(3, 3)
        x = R[:, 0] if local_axis is None else R @ local
        e_pos = target - data.site_xpos[sid]
        if axis is None:
            e_ori = -x[:2]  # want x_world = (0, 0, -1)
        else:
            # residual c = x × a (zero when aligned); error = 0 - c
            e_ori = -np.cross(x, axis)
        pos_err = float(np.linalg.norm(e_pos))
        tilt = float(np.linalg.norm(e_ori))
        if pos_err < pos_tol and tilt < tilt_tol:
            break
        score = pos_err + w * tilt
        if score < best - _STALL_EPS:
            best = score
            since_best = 0
        else:
            since_best += 1
            if since_best >= _STALL_ITERS:
                break
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        Jp = jacp[:, dof_ids]
        # d(x_axis)/dq = omega x x_axis = -[x]_x omega
        Jx = -np.cross(np.broadcast_to(x, (len(dof_ids), 3)), jacr[:, dof_ids].T).T
        if axis is None:
            Jo = Jx[:2]
        else:
            # d(x × a)/dq = -[a]_× dx
            Jo = -np.cross(np.broadcast_to(axis, (len(dof_ids), 3)), Jx.T).T
        J = np.vstack([Jp, w * Jo])
        e = np.concatenate([e_pos, w * e_ori])
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(J.shape[0]), e)
        data.qpos[qadr] = np.clip(data.qpos[qadr] + step * dq, lo, hi)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
    mujoco.mj_forward(model, data)
    pos_err = float(np.linalg.norm(target - data.site_xpos[sid]))
    if pos_err >= pos_tol:
        # Vertical not reachable here (g/h files): keep the least-tilted flex
        # found and polish position with it locked, so reachability is never
        # worse than position-only IK.
        polish_lock = tuple(dict.fromkeys((*lock_joints, "wrist_flex")))
        return _solve_position(sim, target, polish_lock, sid, 60, pos_tol, damping, 0.5)
    sim.hold({n: sim.qpos()[n] for n in ARM_JOINTS})
    return True


def solve_ik_upright(sim: Sim, target_pos, lock_joints: tuple[str, ...] = (), **kwargs) -> bool:
    """Reach the point while keeping gripper +X world-down when that is possible."""
    lock = tuple(lock_joints)
    flex_lock = tuple(dict.fromkeys((*lock, "wrist_flex")))
    kw = {k: v for k, v in kwargs.items() if k not in ("ori_weight", "ori_steps")}
    if solve_ik(sim, target_pos, lock_joints=flex_lock, **kw):
        return True
    kwargs.setdefault("ori_weight", CARRY_ORI_WEIGHT)
    kwargs.setdefault("ori_steps", 2)
    return solve_ik(sim, target_pos, lock_joints=lock, **kwargs)


# Same gate as solve_ik's polish: 2.5 × 3 mm. A plateau past this is not a pose.
_REACH_ERR_M = 0.0075


def pose_reaches(
    sim: Sim,
    target,
    *,
    lock_joints: tuple[str, ...] = (),
    max_err: float = _REACH_ERR_M,
    restore: bool = True,
    upright: bool = False,
    **ik_kwargs,
) -> bool:
    """True if IK can put the EE within `max_err` of `target`. Restores qpos (restore=False: caller's probe)."""
    snap = sim.qpos() if restore else None
    tgt = np.asarray(target, dtype=float)
    with sim.probe() if restore else nullcontext():
        try:
            fn = solve_ik_upright if upright else solve_ik
            ok = bool(fn(sim, tgt, lock_joints=lock_joints, **ik_kwargs))
            pos, _ = sim.site_pose(EE_SITE)
            return bool(ok and float(np.linalg.norm(pos - tgt)) <= float(max_err))
        finally:
            if snap is not None:
                sim.set_qpos(snap, hold=True)


def reachable_approaches(
    sim: Sim,
    candidates,
    goal=None,
    *,
    lock_joints: tuple[str, ...] = (),
    max_err: float = _REACH_ERR_M,
    reset=None,
    accept=None,
    upright: bool = False,
    **ik_kwargs,
) -> list[list[float]]:
    """Candidates that IK, and from which `goal` still IKs, in caller order.

    The preferred hover is often dead near the base (a 10 cm approach over a5)
    while the pinch itself is fine. Callers list high/far poses first so the
    first hit is the safest reachable approach, not a scrape. Restores qpos.
    """
    with sim.probe():
        return _reachable_approaches(
            sim, candidates, goal, lock_joints, max_err, reset, accept, upright, ik_kwargs
        )


def _reachable_approaches(sim, candidates, goal, lock_joints, max_err, reset, accept, upright, ik_kwargs):
    snap = sim.qpos()
    hits: list[list[float]] = []
    try:
        for raw in candidates:
            cand = [float(raw[0]), float(raw[1]), float(raw[2])]
            if reset is not None:
                reset()
            else:
                sim.set_qpos(snap, hold=True)
            if not pose_reaches(
                sim, cand, lock_joints=lock_joints, max_err=max_err,
                restore=False, upright=upright, **ik_kwargs,
            ):
                continue
            if goal is not None and not pose_reaches(
                sim, goal, lock_joints=lock_joints, max_err=max_err,
                restore=False, upright=upright, **ik_kwargs,
            ):
                continue
            if accept is not None and not accept(cand):
                continue
            hits.append(cand)
        return hits
    finally:
        sim.set_qpos(snap, hold=True)


def ik_fail_snapshot(sim: Sim, requested) -> dict:
    """EE requested vs actual plus residual and joint-limit margin. No motion."""
    pos, _ = sim.site_pose(EE_SITE)
    req = np.asarray(requested, dtype=np.float64)
    q = sim.qpos()
    per: dict[str, float] = {}
    best = 1e9
    tightest = None
    for n in ARM_JOINTS:
        lo, hi = sim.model.jnt_range[sim._joint[n]]
        m = float(min(q[n] - lo, hi - q[n]))
        per[n] = round(m, 4)
        if m < best:
            best = m
            tightest = n
    return {
        "ee_requested": [round(float(req[0]), 4), round(float(req[1]), 4), round(float(req[2]), 4)],
        "ee_actual": [round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4)],
        "ik_residual_mm": round(float(np.linalg.norm(pos - req)) * 1000.0, 2),
        "joint_limit_margin_rad": {
            "min_rad": round(float(best), 4),
            "tightest": tightest,
            "per_joint": per,
        },
    }


def _store_ik_fail(sim: Sim, requested, extra: dict | None = None) -> None:
    snap = ik_fail_snapshot(sim, requested)
    if extra:
        snap.update(extra)
    sim._last_ik_fail = snap


# Empty-arm cruise vs the last 2 inches of a pick. Not a video-clock scale:
# duration shrinks, dt stays the same, so joint velocity actually rises.
NEAR_M = 0.0508
FAR_VEL = 0.50
NEAR_VEL = 0.09
HOLD_FAR_VEL = 0.18
HOLD_NEAR_VEL = 0.08


def _vel_at(dist_to_ref: float, holding: bool) -> float:
    far = HOLD_FAR_VEL if holding else FAR_VEL
    near = HOLD_NEAR_VEL if holding else NEAR_VEL
    if dist_to_ref >= NEAR_M:
        return far
    u = max(0.0, float(dist_to_ref) / NEAR_M)
    return near + (far - near) * (u * u)


def cart_duration(
    sim: Sim,
    target_pos,
    holding: bool = False,
    ref=None,
    min_t: float | None = None,
) -> float:
    """Sim-seconds for a straight EE move. Slow inside 2 inches of `ref`."""
    start, _ = sim.site_pose(EE_SITE)
    goal = np.asarray(target_pos, dtype=np.float64)
    if ref is None:
        ref = goal
    ref = np.asarray(ref, dtype=np.float64)
    floor = 0.20 if holding else 0.10
    if min_t is not None:
        floor = float(min_t)
    return float(np.clip(_path_time(start, goal, ref, holding), floor, 2.5))


def _path_time(a, b, ref, holding: bool, n: int = 24) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    t = 0.0
    prev = a
    for i in range(1, n + 1):
        p = a + (b - a) * (i / n)
        ds = float(np.linalg.norm(p - prev))
        t += ds / max(_vel_at(float(np.linalg.norm(p - ref)), holding), 1e-6)
        prev = p
    return t


def _speed_wps(a, b, ref, holding: bool, dt: float):
    """Waypoints along a→b with ds = v(dist_to_ref) * dt."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    delta = b - a
    length = float(np.linalg.norm(delta))
    if length < 1e-5:
        return [b.copy()]
    uhat = delta / length
    pts = []
    s = 0.0
    for _ in range(20000):
        p = a + uhat * s
        s = min(length, s + _vel_at(float(np.linalg.norm(p - ref)), holding) * dt)
        pts.append(a + uhat * s)
        if s >= length - 1e-9:
            break
    pts[-1] = b.copy()
    return pts


def servo_lin(
    sim: Sim,
    target_pos,
    duration: float | None = None,
    physics: bool = True,
    lock_joints: tuple[str, ...] = (),
    holding: bool = False,
    ref=None,
    column: bool = False,
    min_t: float | None = None,
    **ik_kwargs,
) -> bool:
    """Track a Cartesian line by re-solving IK every physics step.

    Proximal joints compensate as the wrist changes. `column=True` holds
    goal XY and only moves Z (after a level XY align if needed).
    """
    start, _ = sim.site_pose(EE_SITE)
    _no_kinematic(physics)
    goal = np.asarray(target_pos, dtype=np.float64)
    if ref is None:
        ref = goal
    if column and float(np.hypot(goal[0] - start[0], goal[1] - start[1])) > 0.004:
        above = np.array([goal[0], goal[1], start[2]], dtype=np.float64)
        if not servo_lin(
            sim,
            above,
            duration=None,
            physics=physics,
            lock_joints=lock_joints,
            holding=holding,
            ref=ref,
            column=False,
            min_t=min_t,
            **ik_kwargs,
        ):
            return False
        start, _ = sim.site_pose(EE_SITE)
    if column:
        start = np.array([goal[0], goal[1], start[2]], dtype=np.float64)
        goal = np.array([goal[0], goal[1], goal[2]], dtype=np.float64)
    settle = ik_kwargs.pop("settle", None)
    # Dense servo + flex nudges slams wrist_flex. Position only along the line.
    track_kw = {k: v for k, v in ik_kwargs.items() if k not in ("ori_weight", "ori_steps")}
    track_kw["lock_joints"] = tuple(lock_joints)
    track_kw.setdefault("max_iters", 40)
    track_kw.setdefault("pos_tol", 0.002)
    wps = _speed_wps(start, goal, np.asarray(ref, dtype=np.float64), holding, sim.dt)
    if duration is not None:
        n = max(2, int(round(float(duration) / max(float(sim.dt), 1e-6))))
        wps = [start + (goal - start) * ((i + 1) / n) for i in range(n)]
        wps[-1] = goal.copy()
    for i, wp in enumerate(wps):
        arm = sim.qpos()
        with sim.probe():
            solved = solve_ik(sim, wp, **track_kw)
            if not solved:
                track_kw["max_iters"] = 80
                solved = solve_ik(sim, wp, **track_kw)
            if not solved:
                sim.set_qpos(arm, hold=True)
                _store_ik_fail(sim, target_pos, extra={"servo_wp": f"{i + 1}/{len(wps)}"})
            cmd = sim.qpos()
            sim.set_qpos(arm, hold=False)
        if not solved:
            return False
        sim.set_ctrl(cmd)
        sim.step()
    sim.hold(sim.qpos())
    extra = (4 if fast_enabled() else 12) if settle is None else max(0, int(settle))
    if extra:
        sim.step(extra)
    pos, _ = sim.site_pose(EE_SITE)
    ok = float(np.linalg.norm(goal - pos)) < 0.02
    if not ok:
        _store_ik_fail(sim, target_pos)
    return ok


def _no_kinematic(physics: bool) -> None:
    if not physics:
        raise ValueError("kinematic moves were removed: the arm moves only by its servos (physics=True)")


def _interp_n(sim: Sim, duration: float) -> int:
    """Physics steps for a move. Carries keep the old floor so the grasp can track."""
    duration = max(1e-3, float(duration) * motion_scale())
    n = int(round(duration / sim.dt))
    return max(8 if fast_enabled() else 30, n)


def move_ik(
    sim: Sim,
    target_pos,
    duration: float = 1.0,
    physics: bool = True,
    settle: int | None = None,
    **ik_kwargs,
) -> bool:
    """Solve IK from a snapshot, then servo the joints to that configuration under physics.

    down=True: solve_ik_down (gripper +X world-down) instead of position IK.
    """
    _no_kinematic(physics)
    start = sim.qpos()
    down = bool(ik_kwargs.pop("down", False))
    # guard: callable checked after every physics step; returning truthy
    # aborts the motion in place (the arm holds where it is) and move_ik
    # returns False with sim._last_abort set to the returned value.
    guard = ik_kwargs.pop("guard", None)
    x_axis = ik_kwargs.pop("x_axis", None)
    with sim.probe():
        if down:
            lock = tuple(ik_kwargs.get("lock_joints", ()))
            solved = solve_ik_down(sim, target_pos, lock_joints=lock, x_axis=x_axis)
        else:
            solved = solve_ik(sim, target_pos, **ik_kwargs)
        if not solved:
            _store_ik_fail(sim, target_pos)
        goal = sim.qpos()
        sim.set_qpos(start, hold=True)
    if not solved:
        return False
    n = _interp_n(sim, duration)
    for i in range(n):
        u = (i + 1) / n
        u = u * u * (3 - 2 * u)
        sim.set_ctrl({k: start[k] + (goal[k] - start[k]) * u for k in start})
        sim.step()
        if guard is not None:
            hit = guard()
            if hit:
                sim._last_abort = hit
                sim.hold(sim.qpos())
                return False
    sim.hold(goal)
    extra = (4 if fast_enabled() else 20) if settle is None else max(0, int(settle))
    if extra:
        sim.step(extra)
    pos, _ = sim.site_pose(EE_SITE)
    ok = float(np.linalg.norm(np.asarray(target_pos) - pos)) < 0.02
    if not ok:
        _store_ik_fail(sim, target_pos)
    return ok


def move_lin(
    sim: Sim,
    target_pos,
    duration: float = 1.0,
    physics: bool = True,
    n_wp: int = 8,
    **ik_kwargs,
) -> bool:
    """Follow a straight Cartesian line by solving IK at several waypoints.

    Joint-space interpolation between a distant start and goal can arc the
    wrist through a placed piece. Short Cartesian segments keep the fingers
    on the intended extract path.
    """
    _no_kinematic(physics)
    settle = ik_kwargs.pop("settle", None)
    start, _ = sim.site_pose(EE_SITE)
    goal = np.asarray(target_pos, dtype=np.float64)
    dt = duration / n_wp
    for i in range(n_wp):
        u = (i + 1) / n_wp
        wp = start + (goal - start) * u
        if not move_ik(sim, wp, duration=dt, settle=settle, **ik_kwargs):
            fail = getattr(sim, "_last_ik_fail", None)
            if isinstance(fail, dict):
                fail["lin_wp"] = f"{i + 1}/{n_wp}"
                fail["lin_goal"] = [round(float(goal[0]), 4), round(float(goal[1]), 4), round(float(goal[2]), 4)]
            return False
    return True


def move_qpos(sim: Sim, goal: dict[str, float], duration: float = 1.0, physics: bool = True):
    """Servo to a known joint configuration under physics."""
    _no_kinematic(physics)
    start = sim.qpos()
    n = _interp_n(sim, duration)
    for i in range(n):
        u = (i + 1) / n
        u = u * u * (3 - 2 * u)
        sim.set_ctrl({k: start[k] + (goal.get(k, start[k]) - start[k]) * u for k in start})
        sim.step()
    sim.hold({**start, **goal})
    sim.step(4 if fast_enabled() else 16)
