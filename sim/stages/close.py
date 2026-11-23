"""Close stage: close the jaws on the piece; the grasp counts from gripper load."""

from __future__ import annotations

import numpy as np

from pickup import (
    close_gripper,
    grasp_acquired,
    gripper_object_transform,
    log_state,
    queen_tilt_deg,
)
from stages import register
from stages.types import Fail, Move


def close(m: Move):
    """Close the jaws and check the grasp from gripper load."""
    sim, result, tid, p, closed_g = m.sim, m.result, m.tid, m.p, m.closed_g
    result["grasp_attempted"] = True
    q_pre, _ = sim.object_pose(tid)
    grip_peak = [0.0]
    prev = sim.on_step

    def _close_step():
        if prev:
            prev()
        grip_peak[0] = max(grip_peak[0], abs(float(sim.actuator_forces()[sim._act["gripper"]])))

    sim.on_step = _close_step
    close_gripper(sim, closed_g, steps=int(p.get("close_steps", 220)))
    sim.on_step = prev
    q_post, qquat_post = sim.object_pose(tid)
    result["close_shove_xy_mm"] = float(np.linalg.norm(q_post[:2] - q_pre[:2])) * 1000
    result["pre_close_queen_disp_mm"] = float(
        np.linalg.norm(q_pre[:2] - np.asarray(result["queen_start_xyz"][:2]))
    ) * 1000
    result["close_tilt_deg"] = queen_tilt_deg(qquat_post)
    result["T_gripper_queen_close"] = gripper_object_transform(sim, tid)
    result["gripper_peak_Nm"] = grip_peak[0]
    result["gripper_saturated"] = grip_peak[0] >= 2.94 - 1e-6
    log_state(
        sim,
        "005_gripper_closed",
        extra={
            "close_shove_xy_mm": round(result["close_shove_xy_mm"], 2),
            "close_tilt_deg": round(result["close_tilt_deg"], 2),
            "T_p_ee_mm": [round(v, 2) for v in result["T_gripper_queen_close"]["p_ee_mm"]],
            "gripper_peak_Nm": round(grip_peak[0], 3),
        },
    )
    result["grasp_acquired"] = grasp_acquired(sim, tid)
    if not result["grasp_acquired"]:
        result["failure_reason"] = "GRASP_MISS"
        return Fail("close", result["failure_reason"], "pick")


@register("close", "load_check")
class LoadCheck:
    """Close the jaws; the grasp counts from gripper load."""

    def close(self, m):
        return close(m)
