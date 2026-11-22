"""Episode result flags. Metrics only — no motion."""

from __future__ import annotations

_REASON_CANON = {
    "PERCEPTION_FAILURE": "PERCEPTION_FAIL",
    "IK_FAILURE": "IK_NO_SOLUTION",
}


def annotate_episode(result: dict) -> dict:
    result["grasp_success"] = bool(result.get("grasp_acquired"))
    result["close_shove"] = result.get("close_shove_xy_mm")
    result["tilt_after_close"] = result.get("close_tilt_deg")
    t = result.get("T_gripper_queen_lift")
    result["T_gripper_object"] = t
    result["placement_error"] = result.get("release_position_error_mm")
    result["perception_success"] = bool(result.get("queen_detected"))
    pickup_ok = bool(result.get("lift_success"))
    trans_ok = bool(result.get("transport_success"))
    place_ok = bool(result.get("placement_success"))
    retreat_ok = bool(result.get("retreat_success"))
    flags = []
    flags.append("GRASP_PASS" if result["grasp_success"] else "GRASP_FAIL")
    flags.append("LIFT_PASS" if pickup_ok else "LIFT_FAIL")
    skip_rest = result.get("stop_after") == "lift" or not pickup_ok
    flags.append("TRANSPORT_PASS" if trans_ok else ("TRANSPORT_SKIP" if skip_rest else "TRANSPORT_FAIL"))
    if result.get("stop_after") == "lift" or (result.get("release_position_error_mm") is None and not place_ok):
        flags.append("PLACEMENT_SKIP")
    else:
        flags.append("PLACEMENT_PASS" if place_ok else "PLACEMENT_FAIL")
    if result.get("retreat_success") is None:
        flags.append("RETREAT_SKIP")
    else:
        flags.append("RETREAT_PASS" if retreat_ok else "RETREAT_FAIL")
    result["stage_flags"] = flags
    reason = result.get("failure_reason")
    if reason in _REASON_CANON:
        reason = _REASON_CANON[reason]
        result["failure_reason"] = reason
    if result.get("stop_after") == "lift":
        if not result.get("perception_success") or not pickup_ok:
            result["failure_category"] = "PICKUP_FAIL"
        else:
            result["failure_category"] = None
            result["success"] = True
    elif not result.get("perception_success"):
        result["failure_category"] = "PICKUP_FAIL"
    elif not pickup_ok:
        result["failure_category"] = "PICKUP_FAIL"
    elif not trans_ok:
        result["failure_category"] = "TRANSPORT_FAIL"
    elif not place_ok:
        result["failure_category"] = "PLACEMENT_FAIL"
    elif result.get("retreat_success") is False:
        result["failure_category"] = "RETREAT_FAIL"
    else:
        result["failure_category"] = None
    hit = result.get("first_collision") or result.get("transport_first_collision")
    result["neighbor_collision_phase"] = None if not hit else hit.get("phase")
    if hit:
        result["obstruction_body"] = hit.get("body")
        result["obstruction_square"] = hit.get("square")
        result["obstruction_piece"] = hit.get("piece")
        result["obstruction_robot_part"] = hit.get("robot_part")
        result["obstruction_phase"] = hit.get("phase")
    else:
        result["obstruction_body"] = None
        result["obstruction_square"] = None
        result["obstruction_piece"] = None
        result["obstruction_robot_part"] = None
        result["obstruction_phase"] = None
    if result.get("ik_fail_phase") is None and reason in ("IK_FAILURE", "IK_NO_SOLUTION"):
        result["ik_fail_phase"] = result.get("fail_phase")
    knocked = result.get("failure_reason") in ("APPROACH_COLLISION",)
    reached = bool(result.get("approach_reached"))
    ik = bool(result.get("ik_solved"))
    ncon = int(result.get("pre_close_contacts") or 0)
    result["approach_ok"] = bool(ik and reached and ncon == 0 and not knocked)
    if result.get("lift_success"):
        result["grasp_ok"] = True
    elif result["approach_ok"]:
        result["grasp_ok"] = False
    else:
        result["grasp_ok"] = None
    return result
