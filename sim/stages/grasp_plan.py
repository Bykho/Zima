"""Grasp-plan stage: grasp point, height and pregrasp for the piece; wrist pitched vertical."""

from __future__ import annotations

from mj import Sim
import pickup
from pickup import (
    _drive_arm,
    _grasp_xy_for_roll,
    _pitch_path,
    _vertical_wrist,
    _xyz,
    find_pregrasp_pose,
    log_state,
)
from stages import register
from stages.types import Fail, Move


def _pitch_in_place(sim: Sim, flex_goal: float, n: int = 8) -> None:
    """Rotate wrist_flex to `flex_goal` while shoulder/elbow hold the EE put.

    A wrist-only move at a stretched observe pose swings the gripper on a
    ~15 cm lever: traced dropping the EE from z=0.14 to z=0.024 and sweeping
    the jaw through the rank below. Plan small pitch steps with position IK
    re-solved at each (wrist locked), then servo through them under physics.
    """
    if abs(flex_goal - float(sim.qpos()["wrist_flex"])) < 1e-3:
        return
    path = _pitch_path(sim, flex_goal, n)
    _drive_arm(sim, path, float(sim.qpos()["gripper"]), duration=0.5)


def _seeded_grasp_xy(sim: Sim, cfg: dict, piece_xy, geom: dict | None, z: float) -> list[float]:
    """Grasp EE XY from the pinch model at the current wrist roll."""
    del cfg
    return _grasp_xy_for_roll(sim, piece_xy, z, geom, None)


def plan(m: Move):
    """Grasp XY from the pinch model on the piece's cross-section at grasp height, wrist pitched vertical, pregrasp pose."""
    sim, cfg, result, surface, queen_xyz, grasp_z, geom, pre_z = m.sim, m.cfg, m.result, m.surface, m.queen_xyz, m.grasp_z, m.geom, m.pre_z
    # Looked up on the module: the perception_error suite wraps it to inject error.
    plan_xy = pickup.perceived_xy(sim, queen_xyz)
    # Vertical pinch: put the wrist where gripper +X is world-down at the
    # grasp point before anything downstream snapshots the heading. Without
    # this the observe pose's 30-50 deg flex was inherited (approach planner
    # locks flex at its current value) and the fixed jaw tipped pieces over on
    # close. Files a-f reach <3 deg; g/h get the least-tilted pose available.
    aim_xy = pickup.pinch_aim_xy(sim, m.tid, plan_xy, surface + grasp_z)
    vert = _vertical_wrist(sim, [aim_xy[0], aim_xy[1], surface + grasp_z])
    if vert is not None:
        result["vertical_flex"] = round(vert["wrist_flex"], 4)
        result["vertical_tilt_deg"] = round(vert["tilt_deg"], 2)
        _pitch_in_place(sim, vert["wrist_flex"])
    grasp_xy = _seeded_grasp_xy(sim, cfg, aim_xy, geom, surface + grasp_z)
    pre = find_pregrasp_pose(sim, cfg, grasp_xy)
    if pre is None:
        result["failure_reason"] = "IK_NO_SOLUTION"
        result["ee_requested"] = _xyz([grasp_xy[0], grasp_xy[1], surface + pre_z])
        log_state(sim, "fail_pregrasp", extra={"target": result["ee_requested"]})
        return Fail("grasp_plan", result["failure_reason"], "pick")
    m.plan_xy = plan_xy
    m.aim_xy = aim_xy
    m.grasp_xy = grasp_xy
    m.pre = pre


@register("grasp_plan", "piece_table")
class PieceTable:
    """Grasp point, height and pregrasp from the piece table plus the fixed-pad shift; pitch the wrist vertical."""

    def plan(self, m):
        return plan(m)
