"""Plumb carry: lift, traverse and lower a held piece without tipping it.

A pinched piece hangs at whatever angle the gripper holds it. On this arm
the pan joint spins everything about world Z, so a piece that was upright
at the close stays upright if the gripper-frame vector that pointed up at
the close keeps pointing up. That is one 2-DOF task (plus position) the
four non-roll joints can hold anywhere a solution exists; roll stays locked
so the jaws never spin about the piece.

The wrist flex stops short of plumb at transit height over mid-board, so
only the ends are strict: the last PLUMB_END_M down keeps the piece within
PLACE_TILT_DEG (a pawn tips past ~31 deg; pieces set down at <= 19 deg stood),
the first PLUMB_END_M up within LIFT_TILT_DEG. Up high it may lean up to
AIRBORNE_TILT_DEG, which a firm pinch carries safely.

Every segment is planned in full before the arm moves: a waypoint every
PLUMB_STEP_M, each solved from the previous one, rejected on tilt, a joint
jump, or a robot link too close to a board piece. A plan that fails costs
no motion, so the caller can fall back.
"""

from __future__ import annotations

import numpy as np

from clearance import neighbor_approach_gap
from ik import ARM_JOINTS, EE_SITE, solve_ik_down
from mj import probing
from robot import PAD_GEOMS

UP = np.array([0.0, 0.0, 1.0])
PLACE_TILT_DEG = 6.0
LIFT_TILT_DEG = 10.0
AIRBORNE_TILT_DEG = 35.0
PLUMB_END_M = 0.03
PLUMB_STEP_M = 0.008
PLUMB_JOINT_STEP_RAD = 0.2  # larger jump between neighbouring waypoints = IK branch switch
PLUMB_BISECT_DEPTH = 3  # halve a jumping step down to 1 mm before calling it a switch
PLUMB_POS_TOL_M = 0.003
PLUMB_GAP_M = 0.004
PLUMB_LOCK = ("wrist_roll",)
PLUMB_SPEED_MPS = 0.12
PLUMB_LOWER_SPEED_MPS = 0.05


def hand_frame(sim, tid: str) -> tuple[np.ndarray, np.ndarray]:
    """(piece up axis, piece origin) in the gripper frame, taking the piece as upright now."""
    gpos, R = sim.site_pose(EE_SITE)
    q, _ = sim.object_pose(tid)
    R = np.asarray(R, dtype=float).reshape(3, 3)
    return R.T @ UP, R.T @ (np.asarray(q, dtype=float) - np.asarray(gpos, dtype=float))


def _tilt_deg(sim, v_local) -> float:
    _, R = sim.site_pose(EE_SITE)
    u = np.asarray(R, dtype=float).reshape(3, 3) @ v_local
    return float(np.degrees(np.arccos(np.clip(u @ UP, -1.0, 1.0))))


def solve_plumb(sim, ee_target, v_local, tilt_max: float = PLACE_TILT_DEG) -> bool:
    ok = solve_ik_down(sim, ee_target, lock_joints=PLUMB_LOCK, x_axis=UP, local_axis=v_local, tilt_tol_deg=1.0,
                       ori_weight=0.3)
    if not ok:
        return False
    gpos, _ = sim.site_pose(EE_SITE)
    if float(np.linalg.norm(np.asarray(gpos) - np.asarray(ee_target))) > PLUMB_POS_TOL_M:
        return False
    return _tilt_deg(sim, v_local) <= tilt_max


@probing
def ee_for_piece(sim, piece_xyz, v_local, p_local) -> np.ndarray | None:
    """EE point that puts the held piece origin at piece_xyz, gripper plumb. Restores qpos."""
    snap = sim.qpos()
    try:
        ee = np.asarray(piece_xyz, dtype=float).copy()
        for _ in range(3):
            if not solve_plumb(sim, ee, v_local):
                gpos, _ = sim.site_pose(EE_SITE)
                sim.plumb_fail = {"why": "place_pose", "tilt": round(_tilt_deg(sim, v_local), 1),
                                  "pos_mm": round(float(np.linalg.norm(np.asarray(gpos) - ee)) * 1000, 1),
                                  "up_in_hand_deg": round(float(np.degrees(np.arccos(np.clip(np.asarray(v_local)[0], -1, 1)))), 1)}
                return None
            _, R = sim.site_pose(EE_SITE)
            ee = np.asarray(piece_xyz, dtype=float) - np.asarray(R, dtype=float).reshape(3, 3) @ p_local
        return ee
    finally:
        sim.set_qpos({n: snap[n] for n in ARM_JOINTS}, hold=True)


def _wps(a, b) -> list[np.ndarray]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = max(1, int(np.ceil(np.linalg.norm(b - a) / PLUMB_STEP_M)))
    return [a + (b - a) * ((i + 1) / n) for i in range(n)]


def _bridge(sim, a, b, prev: dict, v_local, tilt_max: float, where: dict, depth: int) -> list[dict] | None:
    """Joint states from `prev` (at EE `a`) to EE `b`, halving the step while a joint jumps.

    A steep but continuous stretch shrinks when halved; an IK branch flip
    does not, and is refused once the step is down to PLUMB_STEP_M / 2**depth.
    """
    sim.set_qpos(prev, hold=True)
    if not solve_plumb(sim, b, v_local, tilt_max):
        gpos, _ = sim.site_pose(EE_SITE)
        sim.plumb_fail = {**where, "why": "ik", "tilt": round(_tilt_deg(sim, v_local), 1),
                          "pos_mm": round(float(np.linalg.norm(np.asarray(gpos) - b)) * 1000, 1)}
        return None
    q = sim.qpos()
    q = {n: float(q[n]) for n in ARM_JOINTS}
    jump = max(abs(q[n] - prev[n]) for n in ARM_JOINTS)
    if jump <= PLUMB_JOINT_STEP_RAD:
        return [q]
    if depth == 0:
        sim.plumb_fail = {**where, "why": "jump", "rad": round(jump, 2)}
        return None
    mid = 0.5 * (np.asarray(a, dtype=float) + np.asarray(b, dtype=float))
    first = _bridge(sim, a, mid, prev, v_local, tilt_max, where, depth - 1)
    if first is None:
        return None
    second = _bridge(sim, mid, b, first[-1], v_local, tilt_max, where, depth - 1)
    return None if second is None else first + second


@probing
def plan(sim, points, v_local, avoid: list[str], gap: float = PLUMB_GAP_M) -> list[dict] | None:
    """Joint chain along straight EE segments through `points`, or None. Restores qpos.

    `points` are (xyz, tilt_max) pairs; tilt_max applies along the segment ending there.
    A waypoint closer than `gap` to an avoided piece is refused only if it is
    closer than the one before: starting tight (jaws beside a neighbour at
    release) is fine as long as the path leaves.
    """
    snap = sim.qpos()
    try:
        chain: list[dict] = []
        prev = {n: float(snap[n]) for n in ARM_JOINTS}
        start, _ = sim.site_pose(EE_SITE)
        a = np.asarray(start, dtype=float)
        last_gap = neighbor_approach_gap(sim, avoid) if avoid else 1.0
        for seg, (b, tilt_max) in enumerate(points):
            here = a
            for k, wp in enumerate(_wps(a, b)):
                where = {"seg": seg, "wp": k, "z": round(float(wp[2]), 3)}
                steps = _bridge(sim, here, wp, prev, v_local, tilt_max, where, PLUMB_BISECT_DEPTH)
                if steps is None:
                    return None
                for q in steps:
                    sim.set_qpos(q, hold=True)
                    g = neighbor_approach_gap(sim, avoid) if avoid else 1.0
                    if g < gap and g < last_gap - 0.001:
                        sim.plumb_fail = {**where, "why": "gap", "mm": round(g * 1000, 1)}
                        return None
                    last_gap = g
                chain.extend(steps)
                prev, here = chain[-1], wp
            a = np.asarray(b, dtype=float)
        return chain
    finally:
        sim.set_qpos({n: snap[n] for n in ARM_JOINTS}, hold=True)


def _pad_xyz(sim, geom: str) -> np.ndarray:
    import mujoco

    return np.asarray(sim.data.geom_xpos[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, geom)], dtype=float)


PAD_BACKOFF_M = 0.003


def pad_backoff(sim) -> np.ndarray:
    """Small horizontal EE move that takes the fixed pad off the piece it just released.

    Only the moving jaw opens, so the piece is still against the fixed pad.
    A full centre (by stem width) drives the angled moving jaw into the
    wider base and collar.
    """
    fixed, moving = _pad_xyz(sim, PAD_GEOMS[0]), _pad_xyz(sim, PAD_GEOMS[1])
    u = moving - fixed
    u[2] = 0.0
    width = float(np.linalg.norm(u))
    if width < 1e-6:
        return np.zeros(3)
    u /= width
    return -u * PAD_BACKOFF_M


RETREAT_TRIES = 3


def retreat(sim, avoid: list[str], z_up: float, jaw_clear_m: float, touching=None) -> list[dict] | None:
    """After release: back the fixed pad off, straight up at the release
    orientation until the jaws clear the piece top (`jaw_clear_m`), then up
    to z_up leaning freely.

    Rising with the fixed pad still on the piece drags it over. A
    locked-wrist rise re-pitches the gripper as shoulder and elbow move,
    sweeping the jaws sideways through the piece.

    `touching()` (hand on the released piece) stops the rise; the pad backs
    off another step and the rise is re-planned from there, RETREAT_TRIES times.
    """
    gpos0, R = sim.site_pose(EE_SITE)
    v_local = np.asarray(R, dtype=float).reshape(3, 3).T @ UP
    z_clear = float(gpos0[2]) + jaw_clear_m
    for _ in range(RETREAT_TRIES):
        gpos, _ = sim.site_pose(EE_SITE)
        x, y, z = (float(v) for v in np.asarray(gpos, dtype=float) + pad_backoff(sim))
        points = [([x, y, z], LIFT_TILT_DEG), ([x, y, max(z, z_clear)], LIFT_TILT_DEG)]
        if z_up > z_clear:
            points.append(([x, y, float(z_up)], AIRBORNE_TILT_DEG))
        chain = plan(sim, points, v_local, avoid, gap=0.002)
        if chain is None:
            return None
        sim._last_abort = None
        if follow(sim, chain, PLUMB_LOWER_SPEED_MPS, guard=touching):
            return chain
    return None


def follow(sim, chain: list[dict], speed: float, guard=None) -> bool:
    """Track the chain with ctrl only, one smooth accel/decel over the whole path."""
    if not chain:
        return True
    q0 = sim.qpos()
    path = [{n: float(q0[n]) for n in ARM_JOINTS}, *chain]
    length = PLUMB_STEP_M * len(chain)
    n = max(8, int(np.ceil(length / speed / sim.dt)))
    last = len(path) - 1
    for i in range(n):
        u = (i + 1) / n
        s = (u * u * (3 - 2 * u)) * last
        k = min(int(s), last - 1)
        f = s - k
        sim.set_ctrl({j: path[k][j] + (path[k + 1][j] - path[k][j]) * f for j in ARM_JOINTS})
        sim.step()
        if guard is not None:
            hit = guard()
            if hit:
                sim._last_abort = hit
                sim.hold(sim.qpos())
                return False
    sim.hold(path[-1])
    sim.step(20)
    return True
