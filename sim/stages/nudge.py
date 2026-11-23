"""Carry for a placement correction: lift just clear of the board, shift by the measured miss, set down.

The piece already stands on its destination, off-centre by what the confirm
look measured. A full lift and transport re-runs the motion that left it off
(arm tracking under load); a short hop over its own square only has to move
it by the miss.
"""

from __future__ import annotations

import numpy as np

from ik import EE_SITE
from pickup import grasp_acquired, gripper_object_transform, log_state, queen_upright
from stages import register
from stages.carry import _carry_lin
from stages.types import Fail, Move


_SERVO_TRIES = 3
_SERVO_TOL_M = 0.0005


def _servo_to(sim, target, duration: float, hold, extract) -> bool:
    """Cartesian move closed on the joint encoders: under load the arm stops short, so re-aim by the remainder."""
    goal = np.asarray(target, dtype=float)
    aim = goal.copy()
    for _ in range(_SERVO_TRIES):
        if not _carry_lin(sim, aim.tolist(), duration, 6, hold, extract):
            return False
        ee, _ = sim.site_pose(EE_SITE)
        miss = goal - np.asarray(ee, dtype=float)
        if float(np.linalg.norm(miss)) < _SERVO_TOL_M:
            break
        aim = aim + miss
        duration = 0.3
    return True


def nudge(m: Move, hop: float):
    sim, result, tid, dest_xy, hold, place_hover = m.sim, m.result, m.tid, m.dest_xy, m.hold, m.place_hover
    extract = tuple(dict.fromkeys((*hold, "wrist_flex")))
    g_close, _ = sim.site_pose(EE_SITE)
    q, _ = sim.object_pose(tid)
    shift = np.asarray(dest_xy[:2], dtype=float) - np.asarray(q[:2], dtype=float)
    result["grasp_offset_error_mm"] = float(np.linalg.norm(q[:2] - np.asarray(m.grasp[:2]))) * 1000
    result["transport_path"] = "nudge"
    result["place_diag"] = {
        "dest_xy_mm": [round(float(dest_xy[0]) * 1000, 1), round(float(dest_xy[1]) * 1000, 1)],
        "held_xy_after_close_mm": [round(float(q[0]) * 1000, 1), round(float(q[1]) * 1000, 1)],
        "nudge_shift_mm": [round(float(shift[0]) * 1000, 1), round(float(shift[1]) * 1000, 1)],
        "T_ee_after_close_mm": [round(v, 1) for v in gripper_object_transform(sim, tid)["p_ee_mm"]],
    }
    up = [float(g_close[0]), float(g_close[1]), float(g_close[2]) + hop]
    log_state(sim, "006_lift", extra={"target": [round(v, 4) for v in up], "path": "nudge"})
    lifted = _servo_to(sim, up, 0.5, hold, extract)
    _, qquat = sim.object_pose(tid)
    result["lift_success"] = bool(lifted and grasp_acquired(sim, tid) and queen_upright(qquat, tilt_max=0.7))
    if not result["lift_success"]:
        result["failure_reason"] = "LIFT_DROP" if lifted else "IK_FAILURE"
        log_state(sim, "fail_lift", extra={"path": "nudge"})
        return Fail("carry", result["failure_reason"], "pick" if lifted else None)
    over = [up[0] + float(shift[0]), up[1] + float(shift[1]), up[2]]
    log_state(sim, "007_transport", extra={"target": [round(v, 4) for v in over], "path": "nudge"})
    moved = _servo_to(sim, over, 0.6, hold, extract)
    result["transport_success"] = bool(moved and grasp_acquired(sim, tid))
    if not result["transport_success"]:
        result["failure_reason"] = "TRANSPORT_DROP" if moved else "IK_FAILURE"
        log_state(sim, "fail_transport", extra={"path": "nudge"})
        return Fail("carry", result["failure_reason"], "pick" if moved else None)
    place = [over[0], over[1], float(g_close[2]) + place_hover]
    if not _carry_lin(sim, place, 0.5, 6, hold, extract):
        result["failure_reason"] = "IK_FAILURE"
        result["ik_fail_phase"] = "008_lower"
        log_state(sim, "fail_lower", extra={"target": [round(v, 4) for v in place]})
        return Fail("carry", result["failure_reason"])
    log_state(sim, "008_lower", extra={"target": [round(v, 4) for v in place], "path": "nudge"})


@register("carry", "nudge")
class Nudge:
    """Placement correction only: hop clear of the board, shift by the measured miss, set down."""

    def carry(self, m):
        return nudge(m, float(self.params.get("hop", m.p.get("place_nudge_hop", 0.004))))
