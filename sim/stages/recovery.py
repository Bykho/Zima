"""Recovery stage (gated by `pickup.recover_fallen`, off): nest-and-retry for a fallen piece. Code unchanged."""

from __future__ import annotations

import mujoco
import numpy as np

from ik import EE_SITE, move_lin, move_qpos, pose_reaches, reachable_approaches, solve_ik
from mj import ARM_JOINTS, Sim, probing
from pickup import (
    _WORKSPACE_R_M,
    _arm_base_xy,
    _backoff_grasp,
    _move_q_hold,
    _outward_xy,
    _piece_still_upright,
    _reset_ik_seed,
    _xyz,
    grasp_acquired,
    in_well,
    log_state,
    object_on_board,
    open_gripper,
    pickup_cfg,
    queen_tilt_deg,
    queen_upright,
    ray_approach_candidates,
    settle_queen,
)
from scenario import PIECES, body_kind
from sequence import square_of_body
from squares import nest_xyz
from stages import register
from stages.types import Retry


def _piece_com(sim: Sim, tid: str) -> np.ndarray:
    bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, tid)
    return np.asarray(sim.data.subtree_com[bid], dtype=float).copy()


def _fallen_stem_xy(sim: Sim, tid: str) -> np.ndarray | None:
    """World-XY of the piece's body +Z if it is lying on its side."""
    bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, tid)
    axis = np.asarray(sim.data.xmat[bid].reshape(3, 3)[:, 2], dtype=float)
    v = axis[:2]
    n = float(np.linalg.norm(v))
    if n >= 0.25:
        return v / n
    # Upside-down / nearly vertical: body +Z has no usable XY. Scoop
    # along COM vs origin so recover still has a direction (Nf6).
    com = _piece_com(sim, tid)
    qpos, _ = sim.object_pose(tid)
    v = com[:2] - qpos[:2]
    n = float(np.linalg.norm(v))
    if n >= 1e-4:
        return v / n
    return np.array([1.0, 0.0], dtype=float)


_SCOOP_DIP = 0.70


_RECOVER_OPEN = 0.40


_RECOVER_CLOSED = -0.10


_JAW_Z_GEOMS = (
    "moving_jaw_sph_tip1",
    "moving_jaw_box3",
    "fixed_jaw_sph_tip1",
    "fixed_jaw_box5",
)


def _scoop_axis(stem_xy, sign: float, dip: float = _SCOOP_DIP) -> np.ndarray:
    a = np.array([float(stem_xy[0]) * sign, float(stem_xy[1]) * sign, -float(dip)])
    return a / (float(np.linalg.norm(a)) + 1e-12)


def _jaw_open_dir(sim: Sim) -> np.ndarray:
    fix = sim.geom_xpos("fixed_jaw_sph_tip1")
    mov = sim.geom_xpos("moving_jaw_sph_tip1")
    od = mov - fix
    return od / (float(np.linalg.norm(od)) + 1e-12)


def _jaw_zmin(sim: Sim) -> float:
    zmin = 1.0
    for name in _JAW_Z_GEOMS:
        gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            zmin = min(zmin, float(sim.data.geom_xpos[gid][2]))
    return float(zmin)


@probing
def _choose_scoop_axis(sim: Sim, com, stem_xy) -> np.ndarray:
    """Approach along ±stem (with a downward dip) — pick the reachable end."""
    snap = sim.qpos()
    best = None
    for sign in (1.0, -1.0):
        axis = _scoop_axis(stem_xy, sign)
        ok = solve_ik(sim, com, x_axis=axis, ori_weight=0.5, ori_steps=12)
        gpos, grot = sim.site_pose(EE_SITE)
        align = float(np.dot(grot[:, 0], axis))
        err = float(np.linalg.norm(gpos - com))
        score = align - 8.0 * err
        if ok and (best is None or score > best[0]):
            best = (score, axis.copy())
        sim.set_qpos(snap, hold=True)
    return best[1] if best is not None else _scoop_axis(stem_xy, 1.0)


_WRIST_HOLD = ("wrist_flex", "wrist_roll")


_IK_HOLD_ERR = 0.012


def _pinch_xyz(sim: Sim, tid: str, surface: float) -> np.ndarray:
    """Stem-height target a few mm below COM so the pads wrap, not sit on top."""
    com = _piece_com(sim, tid)
    z = max(float(surface) + 0.005, float(com[2]) - 0.005)
    return np.array([float(com[0]), float(com[1]), z], dtype=float)


def _pinch_xyz_base(sim: Sim, tid: str, surface: float) -> np.ndarray:
    """Fat-end pinch. Body origin is the pedestal; COM sits toward the crown."""
    com = _piece_com(sim, tid)
    qpos, _ = sim.object_pose(tid)
    xy = 0.7 * np.asarray(qpos[:2], dtype=float) + 0.3 * com[:2]
    z = max(float(surface) + 0.005, float(com[2]) - 0.005)
    return np.array([float(xy[0]), float(xy[1]), z], dtype=float)


@probing
def _roll_horizontal_pinch(sim: Sim, com, axis) -> float:
    """wrist_roll so jaws open sideways and wrap the stem above the board."""
    start = sim.qpos()
    lo, hi = sim.joint_limits()["wrist_roll"]
    best = float(start["wrist_roll"])
    best_score = -1e9
    for roll in np.linspace(float(lo), float(hi), 25):
        q = dict(start)
        q["wrist_roll"] = float(roll)
        sim.set_qpos(q, hold=True)
        if not solve_ik(
            sim, com, x_axis=axis, ori_weight=0.5, ori_steps=10, lock_joints=("wrist_roll",)
        ):
            continue
        od = _jaw_open_dir(sim)
        zmin = _jaw_zmin(sim)
        _, grot = sim.site_pose(EE_SITE)
        align = float(np.dot(grot[:, 0], axis))
        if align < 0.55:
            continue
        score = -abs(float(od[2])) - 2.0 * abs(zmin - 0.025) + 0.1 * align
        if zmin < 0.018 or zmin > 0.040:
            score -= 2.0
        if score > best_score:
            best_score = score
            best = float(roll)
    sim.set_qpos(start, hold=True)
    return best


def _close_hold(sim: Sim, closed: float, steps: int = 220, hold=None) -> None:
    """Close the gripper while commanding a frozen arm pose."""
    hold = dict(hold or sim.qpos())
    start_g = float(hold.get("gripper", sim.qpos()["gripper"]))
    for i in range(max(1, int(steps))):
        u = (i + 1) / steps
        cmd = {n: hold[n] for n in ARM_JOINTS}
        cmd["gripper"] = start_g + (float(closed) - start_g) * u
        sim.set_ctrl(cmd)
        sim.step()
    sim.hold({**{n: hold[n] for n in ARM_JOINTS}, "gripper": float(closed)})
    sim.step(12)


@probing
def _ik_goal(
    sim: Sim,
    target,
    lock: tuple[str, ...] = _WRIST_HOLD,
    max_err: float = _IK_HOLD_ERR,
    **ik_kwargs,
) -> dict[str, float] | None:
    """IK solution that stays on the locked-wrist family, or None."""
    snap = sim.qpos()
    tgt = np.asarray(target, dtype=float)
    ok = bool(solve_ik(sim, tgt, lock_joints=lock, **ik_kwargs))
    p, _ = sim.site_pose(EE_SITE)
    err = float(np.linalg.norm(p - tgt))
    too_low = float(p[2]) < 0.040 and float(tgt[2]) >= 0.040
    goal = {n: sim.qpos()[n] for n in ARM_JOINTS} if ok and err <= max_err and not too_low else None
    sim.set_qpos(snap, hold=True)
    return goal


def _move_hold(
    sim: Sim,
    target,
    closed: float,
    duration: float = 0.8,
    n_wp: int = 6,
    lock: tuple[str, ...] = _WRIST_HOLD,
    require_grasp: bool = True,
    tid: str | None = None,
    **ik_kwargs,
) -> bool:
    """Cartesian move that keeps the gripper shut and holds the IK goal."""
    start, _ = sim.site_pose(EE_SITE)
    goal = np.asarray(target, dtype=float)
    n_wp = max(2, int(n_wp))
    dt = float(duration) / n_wp
    for i in range(n_wp):
        u = (i + 1) / n_wp
        wp = start + (goal - start) * u
        gq = _ik_goal(sim, wp, lock=lock, **ik_kwargs)
        if gq is None:
            return False
        _move_q_hold(sim, gq, closed, duration=dt)
        if require_grasp and not grasp_acquired(sim, tid):
            return False
    return True


def _js_extract(sim: Sim, tid: str, hold: dict, closed: float, surface: float) -> dict | None:
    """Raise the scoop wrist in joint space so the wrap lifts the piece off the board."""
    lo, hi = sim.joint_limits()["shoulder_lift"]
    pose = {n: hold[n] for n in ARM_JOINTS}
    for _ in range(10):
        nxt = dict(pose)
        nxt["shoulder_lift"] = float(np.clip(pose["shoulder_lift"] - 0.02, lo, hi))
        if abs(nxt["shoulder_lift"] - pose["shoulder_lift"]) < 1e-4:
            break
        _move_q_hold(sim, nxt, closed, duration=0.55)
        pose = nxt
        qpos, _ = sim.object_pose(tid)
        board = any(h["other"] == "board" and h["force"] > 0.05 for h in sim.contacts_with(tid))
        if not grasp_acquired(sim, tid):
            return None
        ee, _ = sim.site_pose(EE_SITE)
        # L-shaped men can have COM up while a prong still kisses the board.
        # Waiting for zero board contact then fails extract (Nf6 / Bxf3).
        lifted = float(qpos[2]) > surface + 0.016
        clear = (not board) or float(qpos[2]) > surface + 0.024
        if lifted and clear and float(ee[2]) >= 0.040:
            return pose
    qpos, _ = sim.object_pose(tid)
    board = any(h["other"] == "board" and h["force"] > 0.05 for h in sim.contacts_with(tid))
    ee, _ = sim.site_pose(EE_SITE)
    lifted = float(qpos[2]) > surface + 0.016
    clear = (not board) or float(qpos[2]) > surface + 0.024
    if grasp_acquired(sim, tid) and lifted and clear:
        return pose
    return None


def _js_escape_hole(sim: Sim, cfg: dict, tid: str, pose: dict, closed: float) -> dict | None:
    """Inch the EE outward while holding. A 60 mm jump or JS pan peels the scoop."""
    del pose
    base = _arm_base_xy(cfg)
    lock = ("wrist_roll",)
    last = {n: sim.qpos()[n] for n in ARM_JOINTS}
    for _ in range(12):
        if not grasp_acquired(sim, tid):
            _move_q_hold(sim, last, closed, duration=0.2)
            return last if grasp_acquired(sim, tid) else None
        ee, _ = sim.site_pose(EE_SITE)
        r = float(np.linalg.norm(ee[:2] - base))
        # Far enough that +Z extract works; going all the way to 190 mm peels.
        if r >= 0.140:
            return {n: sim.qpos()[n] for n in ARM_JOINTS}
        away = _outward_xy(ee[:2], cfg, 0.006)
        tgt = [float(away[0]), float(away[1]), float(ee[2])]
        if not _move_hold(sim, tgt, closed, duration=0.28, n_wp=3, lock=lock, tid=tid):
            _move_q_hold(sim, last, closed, duration=0.2)
            return last if grasp_acquired(sim, tid) else None
        last = {n: sim.qpos()[n] for n in ARM_JOINTS}
    return last if grasp_acquired(sim, tid) else None


def _nest_step_candidates(ee, nest, z: float) -> list[np.ndarray]:
    cur = np.asarray(ee, dtype=float)
    nest = np.asarray(nest, dtype=float)
    d = nest[:2] - cur[:2]
    dist = float(np.linalg.norm(d))
    step = min(0.012, dist) if dist > 1e-9 else 0.0
    out: list[np.ndarray] = []
    if dist > 1e-9:
        nrm = d / dist
        out.append(np.array([cur[0] + nrm[0] * step, cur[1] + nrm[1] * step, z]))
        for zz in (z - 0.012, z + 0.012):
            out.append(np.array([cur[0] + nrm[0] * step, cur[1] + nrm[1] * step, zz]))
    for r in (0.012, 0.020):
        for ang in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
            out.append(np.array([cur[0] + r * np.cos(ang), cur[1] + r * np.sin(ang), z]))
    floor_z = 0.050
    kept = []
    for p in out:
        q = np.asarray(p, dtype=float).copy()
        if q[2] < floor_z:
            q[2] = floor_z
        kept.append(q)
    return kept


def _piece_nest_dist(sim: Sim, tid: str, nest) -> float:
    qpos, _ = sim.object_pose(tid)
    return float(np.hypot(qpos[0] - nest[0], qpos[1] - nest[1]))


def _best_carry_goal(
    sim: Sim, tid: str, nest, z: float, lock: tuple[str, ...] = _WRIST_HOLD
) -> np.ndarray | None:
    """Locked-wrist step that puts the piece closer to the well."""
    cur, _ = sim.site_pose(EE_SITE)
    qpos, _ = sim.object_pose(tid)
    off = np.asarray(qpos[:2], dtype=float) - cur[:2]
    nest = np.asarray(nest, dtype=float)
    want_ee = np.array([nest[0] - off[0], nest[1] - off[1], z], dtype=float)
    cur_d = _piece_nest_dist(sim, tid, nest)
    best = None
    best_d = cur_d
    step_lim = 0.028 if cur_d > 0.08 else 0.018
    cands = _nest_step_candidates(cur, want_ee, z)
    for tgt in cands:
        if float(np.linalg.norm(tgt[:2] - cur[:2])) > step_lim:
            continue
        if _ik_goal(sim, tgt, lock=lock) is None:
            continue
        pred = tgt[:2] + off
        d = float(np.hypot(pred[0] - nest[0], pred[1] - nest[1]))
        if d < best_d - 0.002:
            best_d = d
            best = tgt
    return best


def _recover_descend(
    sim: Sim,
    cfg: dict,
    pinch,
    stem,
    axis,
    rec_open: float,
    lock_r: tuple[str, ...],
    out: dict,
) -> np.ndarray | None:
    """Reach the scoop pinch from a hover that actually IKs.

    The nominal 110 mm / 60 mm-back hover is dead next to the base; the pinch
    is not. Walk the same family as pick (safer poses first), then the other
    stem direction. Physics-tracks each kinematic hit; a failed move reseeds.
    """
    last_high = None
    for flip in (False, True):
        if flip:
            axis = _scoop_axis(stem, -1.0 if float(axis[0]) * float(stem[0]) >= 0 else 1.0)
            roll = _roll_horizontal_pinch(sim, pinch, axis)
            qroll = sim.qpos()
            qroll["wrist_roll"] = float(roll)
            qroll["gripper"] = rec_open
            move_qpos(sim, qroll, duration=0.30, physics=True)
            out["scoop"] = {
                "axis": [round(float(v), 3) for v in axis],
                "roll": round(float(roll), 3),
                "flipped": True,
            }
        roll = float(sim.qpos()["wrist_roll"])
        ikw = dict(x_axis=axis, ori_weight=0.45, ori_steps=8)

        def _reset():
            _reset_ik_seed(sim, cfg, keep={"wrist_roll": roll, "gripper": rec_open})

        if not pose_reaches(sim, pinch, lock_joints=lock_r, **ikw):
            continue
        hits = reachable_approaches(
            sim,
            ray_approach_candidates(pinch, axis[:2]),
            pinch,
            reset=_reset,
            lock_joints=lock_r,
            **ikw,
        )
        for high in hits:
            last_high = high
            with sim.probe():
                _reset()
            if not move_lin(sim, high, duration=0.9, physics=True, n_wp=6, lock_joints=lock_r, **ikw):
                continue
            log_state(sim, "recover_pregrasp", extra={"target": _xyz(high), "pinch": _xyz(pinch)})
            if move_lin(sim, pinch, duration=1.1, physics=True, n_wp=10, lock_joints=lock_r, **ikw):
                return np.asarray(axis, dtype=float)
    out["reason"] = "RECOVER_IK"
    log_state(
        sim,
        "fail_recover_pregrasp",
        extra={"target": _xyz(last_high if last_high is not None else pinch), "pinch": _xyz(pinch)},
    )
    return None


def recover_to_nest(sim: Sim, cfg: dict, tid: str) -> dict:
    """Approach along the lying stem, pinch across it, drop into the well."""
    p = pickup_cfg(cfg)
    surface = float(cfg.get("sim", {}).get("board", {}).get("surface_z", 0.02))
    release_open = float(p.get("gripper_release", 0.30))
    rec_open = min(float(p.get("gripper_open", 0.60)), _RECOVER_OPEN)
    geom = dict(PIECES.get(body_kind(tid) or tid) or {})
    closed_g = float(geom.get("gripper_closed", _RECOVER_CLOSED))
    nest = nest_xyz(cfg)
    out = {"ok": False, "reason": None, "nest_xyz": list(nest)}
    if _piece_still_upright(sim, tid, surface, tilt_max=0.5):
        out["ok"] = True
        out["reason"] = "already_upright"
        return out
    qpos, _ = sim.object_pose(tid)
    if float(qpos[2]) < surface - 0.02:
        out["reason"] = "RECOVER_OFF_TABLE"
        return out
    stem = _fallen_stem_xy(sim, tid)
    if stem is None:
        out["reason"] = "RECOVER_NO_STEM"
        return out

    _backoff_grasp(sim, cfg, tid, release_open)
    open_gripper(sim, rec_open, steps=40)
    settle_queen(sim, steps=16)
    sim.target_id = tid
    stem = _fallen_stem_xy(sim, tid)
    if stem is None:
        out["reason"] = "RECOVER_NO_STEM"
        return out
    com = _piece_com(sim, tid)
    pinch = _pinch_xyz(sim, tid, surface)
    axis = _choose_scoop_axis(sim, pinch, stem)
    roll = _roll_horizontal_pinch(sim, pinch, axis)
    out["scoop"] = {"axis": [round(float(v), 3) for v in axis], "roll": round(float(roll), 3)}

    qroll = sim.qpos()
    qroll["wrist_roll"] = float(roll)
    qroll["gripper"] = rec_open
    move_qpos(sim, qroll, duration=0.35, physics=True)
    lock_r = ("wrist_roll",)
    axis = _recover_descend(sim, cfg, pinch, stem, axis, rec_open, lock_r, out)
    if axis is None:
        return out
    ikw = dict(x_axis=axis, ori_weight=0.45, ori_steps=8)
    log_state(sim, "recover_grasp", extra={"pinch": _xyz(pinch), "com": _xyz(com)})
    scoop_q = {n: sim.qpos()[n] for n in ARM_JOINTS}
    _close_hold(sim, closed_g, steps=int(p.get("close_steps", 220)), hold=scoop_q)
    log_state(sim, "recover_closed")
    if not grasp_acquired(sim, tid):
        open_gripper(sim, rec_open, steps=40)
        pinch = _pinch_xyz(sim, tid, surface)
        axis = _scoop_axis(stem, -1.0 if float(axis[0]) * float(stem[0]) >= 0 else 1.0)
        ikw = dict(x_axis=axis, ori_weight=0.45, ori_steps=8)
        move_lin(sim, pinch, duration=0.8, physics=True, n_wp=8, lock_joints=lock_r, **ikw)
        scoop_q = {n: sim.qpos()[n] for n in ARM_JOINTS}
        _close_hold(sim, closed_g, steps=int(p.get("close_steps", 220)), hold=scoop_q)
        log_state(sim, "recover_closed")
    if not grasp_acquired(sim, tid):
        # COM pinch misses a thin stem. Try the fat pedestal.
        open_gripper(sim, rec_open, steps=40)
        pinch = _pinch_xyz_base(sim, tid, surface)
        axis = _choose_scoop_axis(sim, pinch, stem)
        roll = _roll_horizontal_pinch(sim, pinch, axis)
        qroll = sim.qpos()
        qroll["wrist_roll"] = float(roll)
        qroll["gripper"] = rec_open
        move_qpos(sim, qroll, duration=0.30, physics=True)
        base_out = {}
        axis2 = _recover_descend(sim, cfg, pinch, stem, axis, rec_open, lock_r, base_out)
        if axis2 is not None:
            axis = axis2
            scoop_q = {n: sim.qpos()[n] for n in ARM_JOINTS}
            _close_hold(sim, closed_g, steps=int(p.get("close_steps", 220)), hold=scoop_q)
            log_state(sim, "recover_closed", extra={"pinch": "base"})
    if not grasp_acquired(sim, tid):
        out["reason"] = "RECOVER_GRASP"
        log_state(sim, "fail_recover_grasp")
        return out
    # Pure +Z extract inside ~190 mm peels the scoop. Walk out first.
    gpos, _ = sim.site_pose(EE_SITE)
    if float(np.linalg.norm(gpos[:2] - _arm_base_xy(cfg))) < _WORKSPACE_R_M:
        escaped = _js_escape_hole(sim, cfg, tid, {n: sim.qpos()[n] for n in ARM_JOINTS}, closed_g)
        if escaped is None or not grasp_acquired(sim, tid):
            out["reason"] = "RECOVER_DROP"
            log_state(sim, "fail_recover_drop")
            return out
        scoop_q = escaped
    qpos, _ = sim.object_pose(tid)
    already_up = float(qpos[2]) > surface + 0.016 and grasp_acquired(sim, tid)
    hold = scoop_q if already_up else _js_extract(sim, tid, scoop_q, closed_g, surface)
    if hold is None:
        out["reason"] = "RECOVER_DROP"
        log_state(sim, "fail_recover_drop")
        return out
    qpos, _ = sim.object_pose(tid)
    log_state(sim, "recover_extract", extra={"xyz": _xyz(qpos)})

    gpos, _ = sim.site_pose(EE_SITE)
    # Hole scoop is a low wrap. Climbing to the usual 80 mm hover peels it
    # whether we skipped extract or just finished it.
    in_hole = float(np.linalg.norm(gpos[:2] - _arm_base_xy(cfg))) < _WORKSPACE_R_M
    if in_hole:
        z_carry = max(float(gpos[2]), 0.040)
    else:
        z_carry = max(float(gpos[2]), 0.055, float(nest[2]) + 0.06)
    if float(gpos[2]) < z_carry - 0.004:
        up = [float(gpos[0]), float(gpos[1]), z_carry]
        if not _move_hold(sim, up, closed_g, duration=0.7, n_wp=5, lock=_WRIST_HOLD, tid=tid):
            if not _move_hold(sim, up, closed_g, duration=0.7, n_wp=5, lock=("wrist_roll",), tid=tid):
                out["reason"] = "RECOVER_DROP" if not grasp_acquired(sim, tid) else "RECOVER_LIFT"
                log_state(sim, "fail_recover_drop")
                return out
    for _ in range(70):
        if not grasp_acquired(sim, tid):
            out["reason"] = "RECOVER_DROP"
            log_state(sim, "fail_recover_transit")
            return out
        if _piece_nest_dist(sim, tid, nest) <= 0.010:
            break
        nxt = _best_carry_goal(sim, tid, nest, z_carry)
        if nxt is None:
            break
        if not _move_hold(sim, nxt, closed_g, duration=0.45, n_wp=3, lock=_WRIST_HOLD, tid=tid):
            if not _move_hold(
                sim, nxt, closed_g, duration=0.5, n_wp=3, lock=("wrist_roll",), tid=tid, require_grasp=False
            ):
                out["reason"] = "RECOVER_DROP" if not grasp_acquired(sim, tid) else "RECOVER_TRANSIT"
                log_state(sim, "fail_recover_transit", extra={"target": _xyz(nxt)})
                return out
            if not grasp_acquired(sim, tid):
                out["reason"] = "RECOVER_DROP"
                log_state(sim, "fail_recover_transit")
                return out
            continue
        qpos, _ = sim.object_pose(tid)
        if float(qpos[2]) < surface + 0.008:
            out["reason"] = "RECOVER_DROP"
            log_state(sim, "fail_recover_drop")
            return out
    dist = _piece_nest_dist(sim, tid, nest)
    if dist > 0.018:
        out["reason"] = "RECOVER_TRANSIT"
        log_state(sim, "fail_recover_transit", extra={"piece_xy_mm": round(dist * 1000.0, 1)})
        return out
    # Center the held man over the hole, then seat on the lip. Carry steps
    # can stop 15 mm off-axis (bishop) where a 28 mm cup cannot stand it.
    qpos, _ = sim.object_pose(tid)
    ee, _ = sim.site_pose(EE_SITE)
    off = np.asarray(qpos[:2], dtype=float) - np.asarray(ee[:2], dtype=float)
    mid_z = max(float(ee[2]), float(nest[2]) + 0.028)
    center = [float(nest[0] - off[0]), float(nest[1] - off[1]), mid_z]
    if float(np.hypot(qpos[0] - nest[0], qpos[1] - nest[1])) > 0.006:
        if not _move_hold(sim, center, closed_g, duration=0.6, n_wp=5, lock=_WRIST_HOLD, tid=tid):
            _move_hold(sim, center, closed_g, duration=0.6, n_wp=5, lock=("wrist_roll",), tid=tid, require_grasp=False)
        if not grasp_acquired(sim, tid):
            out["reason"] = "RECOVER_DROP"
            log_state(sim, "fail_recover_drop")
            return out
    # Seat on the lip while still holding, then let go. Opening 8 cm up
    # throws the man past a 28 mm hole; the real arm has the same problem.
    lip_z = float(nest[2]) + 0.016
    for _ in range(14):
        qpos, _ = sim.object_pose(tid)
        ee, _ = sim.site_pose(EE_SITE)
        if float(qpos[2]) <= lip_z + 0.006 or float(ee[2]) <= lip_z + 0.004:
            break
        down = [float(ee[0]), float(ee[1]), max(lip_z, float(ee[2]) - 0.008)]
        if not _move_hold(sim, down, closed_g, duration=0.35, n_wp=3, lock=_WRIST_HOLD, tid=tid, require_grasp=False):
            if not _move_hold(
                sim, down, closed_g, duration=0.35, n_wp=3, lock=("wrist_roll",), tid=tid, require_grasp=False
            ):
                break
    log_state(sim, "recover_insert", extra={"target": _xyz(nest), "piece_xy_mm": round(_piece_nest_dist(sim, tid, nest) * 1000.0, 1)})
    arm = {n: sim.qpos()[n] for n in ARM_JOINTS}
    g0 = float(sim.qpos()["gripper"])
    for i in range(110):
        u = (i + 1) / 110
        cmd = dict(arm)
        cmd["gripper"] = g0 + (release_open - g0) * u
        sim.set_ctrl(cmd)
        sim.step()
    open_gripper(sim, max(release_open, rec_open), steps=60)
    settle_queen(sim, steps=240)
    # Leave the well on +Z after the piece has settled. An XY pad slide
    # hits the walls; a hop while the pads still wrap lifts it out (Nf6
    # ended airborne at 85 mm, 39 deg, then failed the from-nest place).
    if grasp_acquired(sim, tid):
        open_gripper(sim, max(release_open, rec_open, 0.45), steps=50)
        settle_queen(sim, steps=80)
    ee, _ = sim.site_pose(EE_SITE)
    if not grasp_acquired(sim, tid):
        move_lin(
            sim,
            [float(ee[0]), float(ee[1]), float(ee[2]) + 0.05],
            duration=0.6,
            physics=True,
            n_wp=5,
            lock_joints=_WRIST_HOLD,
        )
        settle_queen(sim, steps=160)
    else:
        settle_queen(sim, steps=80)
    qpos, quat = sim.object_pose(tid)
    out["final_xyz"] = _xyz(qpos)
    out["tilt_deg"] = round(queen_tilt_deg(quat), 1)
    seated = in_well(qpos, cfg) and float(qpos[2]) <= float(nest[2]) + 0.040
    if (not seated) or (not queen_upright(quat, tilt_max=0.70)):
        out["reason"] = "RECOVER_STILL_DOWN"
        log_state(
            sim,
            "fail_recover_down",
            extra={"tilt_deg": out["tilt_deg"], "seated": bool(seated), "z_mm": round(float(qpos[2]) * 1000, 1)},
        )
        return out
    out["ok"] = True
    log_state(sim, "recover_upright", extra={"tilt_deg": out["tilt_deg"], "xyz": out["final_xyz"]})
    return out


def _recover_fallen(sim: Sim, cfg: dict, sc: dict, failed: dict, tid: str, surface: float) -> Retry | None:
    """A fallen piece is tagged PIECE_FALLEN; with recovery on, scooped into the nest and picked again from there."""
    if int(sc.get("_recover_attempt", 0)) >= 2:
        return None
    dest = sc.get("dest")
    if (
        dest
        and sc.get("place_xyz") is None
        and square_of_body(sim, cfg, tid) == dest
        and object_on_board(sim, surface, object_id=tid)
    ):
        # On dest already. Nest scoop from f6 hits f7 (Opera Nf6).
        return None
    if _piece_still_upright(sim, tid, surface, tilt_max=0.5):
        return None
    qpos, _ = sim.object_pose(tid)
    if float(qpos[2]) < surface - 0.02:
        return None
    if not pickup_cfg(cfg).get("recover_fallen", False):
        failed["fallen_after"] = failed.get("failure_reason")
        failed["failure_reason"] = "PIECE_FALLEN"
        return None
    log_state(sim, "recover_start", extra={"after": failed.get("failure_reason")})
    rec = {"ok": False}
    for _try in range(2):
        rec = recover_to_nest(sim, cfg, tid)
        if rec.get("ok"):
            break
        if rec.get("reason") in ("RECOVER_OFF_TABLE", "RECOVER_NO_STEM"):
            break
        log_state(sim, "recover_retry", extra={"try": _try + 1, "reason": rec.get("reason")})
    failed["recover"] = rec
    if not rec.get("ok"):
        failed["failure_reason"] = rec.get("reason") or "RECOVER_FAIL"
        return None
    nxt = dict(sc)
    nxt["seed_home"] = True
    nxt["reset_board"] = False
    nxt["_from_nest"] = True
    nxt["_recover_attempt"] = int(sc.get("_recover_attempt", 0)) + 1
    nxt["_grasp_attempt"] = 0
    return Retry(nxt, {"recovered": True, "recover": rec, "first_grasp_reason": failed.get("failure_reason")})


@register("recovery", "nest")
class Nest:
    """Tag a fallen piece PIECE_FALLEN; with `pickup.recover_fallen` on, scoop it into the nest, stand it up and pick again."""

    def recover(self, sim, cfg, sc, failed, tid, surface):
        return _recover_fallen(sim, cfg, sc, failed, tid, surface)
