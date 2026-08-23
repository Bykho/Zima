"""Damped-least-squares IK for the 5-DoF arm.

SO-101 cannot reach arbitrary 6-DoF poses. Position is required.
A soft pull on gripperframe +X toward world -Z is optional; it must not
override reaching the point. Rotation about the approach axis is free.
"""

from __future__ import annotations

import numpy as np

from mj import ARM_JOINTS, Sim

EE_SITE = "gripperframe"
DOWN = np.array([0.0, 0.0, -1.0])


def _dof_ids(sim: Sim) -> list[int]:
    return [int(sim.model.jnt_dofadr[sim._joint[n]]) for n in ARM_JOINTS]


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
) -> bool:
    """Move arm qpos so gripperframe reaches target_pos."""
    import mujoco

    target = np.asarray(target_pos, dtype=np.float64)
    axis = np.asarray(x_axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    sid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    names = [n for n in ARM_JOINTS if n not in lock_joints]
    dof_ids = [int(sim.model.jnt_dofadr[sim._joint[n]]) for n in names]
    jacp = np.zeros((3, sim.model.nv))
    jacr = np.zeros((3, sim.model.nv))

    for _ in range(max_iters):
        mujoco.mj_forward(sim.model, sim.data)
        pos = sim.data.site_xpos[sid]
        err_pos = target - pos
        if float(np.linalg.norm(err_pos)) < pos_tol:
            sim.hold({n: sim.qpos()[n] for n in ARM_JOINTS})
            return True
        mujoco.mj_jacSite(sim.model, sim.data, jacp, jacr, sid)
        if ori_weight <= 0:
            J = jacp[:, dof_ids]
            jjt = J @ J.T + damping * np.eye(3)
            dq = J.T @ np.linalg.solve(jjt, err_pos)
        else:
            rot = sim.data.site_xmat[sid].reshape(3, 3)
            err_rot = np.cross(rot[:, 0], axis)
            J = np.vstack([jacp, ori_weight * jacr])[:, dof_ids]
            err = np.concatenate([err_pos, ori_weight * err_rot])
            jjt = J @ J.T + damping * np.eye(6)
            dq = J.T @ np.linalg.solve(jjt, err)
        q = sim.qpos()
        for i, name in enumerate(names):
            lo, hi = sim.model.jnt_range[sim._joint[name]]
            q[name] = float(np.clip(q[name] + step * dq[i], lo, hi))
        sim.set_qpos(q, hold=True)

    mujoco.mj_forward(sim.model, sim.data)
    err = float(np.linalg.norm(target - sim.data.site_xpos[sid]))
    sim.hold({n: sim.qpos()[n] for n in ARM_JOINTS})
    return err < pos_tol * 2.5


def move_ik(sim: Sim, target_pos, duration: float = 1.0, physics: bool = False, **ik_kwargs) -> bool:
    """Solve IK from a snapshot, then interpolate joints to that configuration.

    physics=False: re-assert qpos each step (stiff free-space motion).
    physics=True:  ctrl-only tracking so contacts can lift a grasped object.
    """
    start = sim.qpos()
    if not solve_ik(sim, target_pos, **ik_kwargs):
        sim.set_qpos(start, hold=True)
        return False
    goal = sim.qpos()
    sim.set_qpos(start, hold=True)
    n = max(30, int(duration / sim.dt))
    for i in range(n):
        u = (i + 1) / n
        u = u * u * (3 - 2 * u)
        cmd = {k: start[k] + (goal[k] - start[k]) * u for k in start}
        if physics:
            sim.set_ctrl(cmd)
        else:
            sim.set_qpos(cmd, hold=True)
        sim.step()
        if not physics:
            sim.set_qpos(cmd, hold=True)
    if physics:
        sim.hold(goal)
        for _ in range(20):
            sim.step()
    else:
        sim.set_qpos({**start, **goal}, hold=True)
    pos, _ = sim.site_pose(EE_SITE)
    return float(np.linalg.norm(np.asarray(target_pos) - pos)) < 0.02


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
    start, _ = sim.site_pose(EE_SITE)
    goal = np.asarray(target_pos, dtype=np.float64)
    dt = duration / n_wp
    for i in range(n_wp):
        u = (i + 1) / n_wp
        wp = start + (goal - start) * u
        if not move_ik(sim, wp, duration=dt, physics=physics, **ik_kwargs):
            return False
    return True


def move_qpos(sim: Sim, goal: dict[str, float], duration: float = 1.0, physics: bool = True):
    """Interpolate to a known joint configuration."""
    start = sim.qpos()
    n = max(30, int(duration / sim.dt))
    for i in range(n):
        u = (i + 1) / n
        u = u * u * (3 - 2 * u)
        cmd = {k: start[k] + (goal.get(k, start[k]) - start[k]) * u for k in start}
        if physics:
            sim.set_ctrl(cmd)
        else:
            sim.set_qpos(cmd, hold=True)
        sim.step()
        if not physics:
            sim.set_qpos(cmd, hold=True)
    if physics:
        sim.hold({**start, **goal})
        for _ in range(16):
            sim.step()
    else:
        sim.set_qpos({**start, **goal}, hold=True)
