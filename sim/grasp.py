"""Contact-triggered grasp for sim — jaws close on piece, weld on contact, release on open."""

from __future__ import annotations

import time

import pybullet as p

from arm import SimArm, JOINT_NAMES
from board import link_index

JAW_LINKS = ("gripper_link", "moving_jaw_so101_v1_link")
GRIPPER_PARENT = "gripper_link"


def _grasp_cfg(cfg: dict) -> dict:
    return cfg.get("sim", {}).get("grasp", {})


def jaw_link_indices(arm: SimArm) -> list[int]:
    return [link_index(arm, name) for name in JAW_LINKS]


def tune_grasp_physics(arm: SimArm, piece_id: int, cfg: dict):
    """Friction and solver tweaks so jaw–piece contacts are reliable."""
    g = _grasp_cfg(cfg)
    friction = g.get("friction", 1.2)
    for idx in jaw_link_indices(arm):
        p.changeDynamics(arm.robot_id, idx, lateralFriction=friction, rollingFriction=0.02)
    p.changeDynamics(piece_id, -1, lateralFriction=friction, rollingFriction=0.02)
    p.changeDynamics(piece_id, -1, contactStiffness=5000, contactDamping=100)


def jaw_contacts(arm: SimArm, piece_id: int, cfg: dict) -> list:
    jaws = set(jaw_link_indices(arm))
    min_force = _grasp_cfg(cfg).get("min_normal_force", 0.05)
    hits = []
    for c in p.getContactPoints(bodyA=arm.robot_id, bodyB=piece_id):
        if c[3] in jaws and c[9] >= min_force:
            hits.append(c)
    return hits


def grasp_confirmed(arm: SimArm, piece_id: int, cfg: dict) -> bool:
    return len(jaw_contacts(arm, piece_id, cfg)) >= _grasp_cfg(cfg).get("min_contact_points", 1)


def attach(arm: SimArm, piece_id: int) -> int:
    parent = link_index(arm, GRIPPER_PARENT)
    return p.createConstraint(
        arm.robot_id,
        parent,
        piece_id,
        -1,
        p.JOINT_FIXED,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    )


def detach(constraint_id: int | None):
    if constraint_id is not None:
        p.removeConstraint(constraint_id)


def set_gripper(arm: SimArm, angle: float, force: float | None = None):
    force = force if force is not None else arm.motor_force
    idx = arm.joint_indices["gripper"]
    p.setJointMotorControl2(
        arm.robot_id,
        idx,
        p.POSITION_CONTROL,
        targetPosition=angle,
        force=force,
    )


def hold_joints(arm: SimArm, positions: dict[str, float]):
    for name in JOINT_NAMES:
        if name == "gripper":
            continue
        p.setJointMotorControl2(
            arm.robot_id,
            arm.joint_indices[name],
            p.POSITION_CONTROL,
            targetPosition=positions[name],
            force=arm.motor_force,
        )


def close_until_grasp(arm: SimArm, piece_id: int, cfg: dict, hold: dict[str, float]) -> int | None:
    """Close gripper slowly; weld on first jaw contact."""
    if grasp_confirmed(arm, piece_id, cfg):
        return attach(arm, piece_id)

    g = _grasp_cfg(cfg)
    open_rad = hold.get("gripper", cfg.get("poses", {}).get("home", {}).get("gripper", 0.8))
    closed_rad = cfg.get("poses", {}).get("gripper_closed", {}).get("gripper", 0.0)
    steps = g.get("close_steps", 80)
    grip_force = g.get("gripper_force", 120.0)
    timeout = g.get("timeout_s", 4.0)
    t0 = time.time()

    for i in range(steps):
        if time.time() - t0 > timeout:
            break
        u = (i + 1) / steps
        grip = open_rad + (closed_rad - open_rad) * u
        hold_joints(arm, hold)
        set_gripper(arm, grip, force=grip_force)
        arm.step()
        if grasp_confirmed(arm, piece_id, cfg):
            return attach(arm, piece_id)

    hold_joints(arm, hold)
    set_gripper(arm, closed_rad, force=grip_force)
    for _ in range(g.get("settle_steps", 30)):
        arm.step()
        if grasp_confirmed(arm, piece_id, cfg):
            return attach(arm, piece_id)
    return None


def grasp_held(constraint_id: int | None) -> bool:
    return constraint_id is not None


def open_gripper(arm: SimArm, cfg: dict, hold: dict[str, float], steps: int = 40):
    g = _grasp_cfg(cfg)
    open_rad = cfg.get("poses", {}).get("home", {}).get("gripper", 0.8)
    closed_rad = hold.get("gripper", 0.0)
    grip_force = g.get("gripper_force", 120.0)
    for i in range(steps):
        u = (i + 1) / steps
        grip = closed_rad + (open_rad - closed_rad) * u
        hold_joints(arm, hold)
        set_gripper(arm, grip, force=grip_force)
        arm.step()
