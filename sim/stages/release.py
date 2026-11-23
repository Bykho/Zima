"""Release stage: open to the release aperture at the place pose and record where the piece is left."""

from __future__ import annotations

import mujoco
import numpy as np

from ik import EE_SITE
from mj import Sim
from pickup import (
    JAW_BODIES,
    _perceiving,
    _tid,
    bowl_contains,
    grasp_acquired,
    gripper_object_transform,
    log_state,
    min_jaw_gap,
    object_on_board,
    open_gripper,
    queen_tilt_deg,
    queen_upright,
    settle_queen,
)
from robot import world_of
from stages import register
from stages.types import Move


def _geom_name(sim: Sim, gid: int) -> str:
    return mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"geom{gid}"


def _jaw_role(name: str) -> str:
    if "sph_tip" in name:
        return "tip"
    if "part0" in name or "part1" in name:
        return "hinge_mesh"
    if "box1" in name:
        return "palm"
    if "box" in name:
        return "finger"
    return "other"


def release_snapshot(sim: Sim, object_id: str | None = None) -> dict:
    """Jaw/piece contact at the current release pose. Evaluation only."""
    tid = object_id or _tid(sim)
    sim = world_of(sim)
    qpos, qquat = sim.object_pose(tid)
    gpos, _ = sim.site_pose(EE_SITE)
    bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, tid)
    hits = []
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        b1 = int(sim.model.geom_bodyid[c.geom1])
        b2 = int(sim.model.geom_bodyid[c.geom2])
        if bid not in (b1, b2):
            continue
        other = b2 if b1 == bid else b1
        other_name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, other)
        if other_name not in JAW_BODIES:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(sim.model, sim.data, i, force)
        g_other = c.geom2 if b1 == bid else c.geom1
        g_piece = c.geom1 if b1 == bid else c.geom2
        gname = _geom_name(sim, g_other)
        n = np.asarray(c.frame[:3], dtype=float)
        hits.append(
            {
                "other": other_name,
                "geom": gname,
                "piece_geom": _geom_name(sim, g_piece),
                "role": _jaw_role(gname),
                "dist_mm": float(c.dist) * 1000,
                "force": float(np.linalg.norm(force[:3])),
                "normal_z": float(n[2]),
                "normal": [round(float(v), 3) for v in n],
                "friction": [round(float(v), 4) for v in sim.model.geom_friction[g_other]],
            }
        )
    T = gripper_object_transform(sim, tid)
    roles = sorted({h["role"] for h in hits if h["force"] > 0.02})
    return {
        "jaw_gap_mm": float(sim.jaw_gap() * 1000),
        "min_jaw_gap_mm": float(min_jaw_gap(sim, tid) * 1000),
        "gripper_qpos": float(sim.qpos()["gripper"]),
        "piece_z": float(qpos[2]),
        "ee_z": float(gpos[2]),
        "hang_mm": float((gpos[2] - qpos[2]) * 1000),
        "p_ee_mm": [round(float(v), 2) for v in T["p_ee_mm"]],
        "tilt_deg": float(T["tilt_deg"]),
        "held": grasp_acquired(sim, tid),
        "n_jaw_contact": len([h for h in hits if h["force"] > 0.02]),
        "roles": roles,
        "contacts": hits[:8],
    }


def _belief_set_down(sim, tid: str, surface: float) -> None:
    """Released piece: belief stands it on its base under the jaws (a placed man settles upright)."""
    if not _perceiving(sim) or tid not in sim.known:
        return
    p, q = sim.object_pose(tid)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    x = R.reshape(3, 3)[:, 0]
    yaw = float(np.arctan2(x[1], x[0]))
    sim.believe(tid, [float(p[0]), float(p[1]), surface], (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)))


def release(m: Move):
    """Open to the release aperture at the place pose; record where the piece is left."""
    sim, result, tid, surface, p, dest_xy, off_board, release_open = m.sim, m.result, m.tid, m.surface, m.p, m.dest_xy, m.off_board, m.release_open
    q_lo, _ = sim.object_pose(tid)
    g_lo, r_lo = sim.site_pose(EE_SITE)
    result["place_diag"]["piece_xy_after_lower_mm"] = [
        round(float(q_lo[0]) * 1000, 1),
        round(float(q_lo[1]) * 1000, 1),
    ]
    result["place_diag"]["piece_err_after_lower_mm"] = round(
        float(np.hypot(q_lo[0] - dest_xy[0], q_lo[1] - dest_xy[1])) * 1000, 1
    )
    result["place_diag"]["ee_xy_after_lower_mm"] = [
        round(float(g_lo[0]) * 1000, 1),
        round(float(g_lo[1]) * 1000, 1),
    ]
    result["place_diag"]["ee_x_z_after_lower"] = round(float(r_lo[2, 0]), 3)
    result["place_diag"]["T_ee_after_lower_mm"] = [
        round(v, 1) for v in gripper_object_transform(sim, tid)["p_ee_mm"]
    ]

    _, g_hi = sim.joint_limits()["gripper"]
    # Narrower than approach open: a full 0.60 swing slaps neighbors
    # once the piece is already on the square.
    release_g = min(float(g_hi), float(release_open))
    open_gripper(sim, release_g, steps=100)
    settle_queen(sim, steps=16)
    _belief_set_down(sim, tid, surface)
    result["release_after_open"] = release_snapshot(sim, tid)
    q_open, _ = sim.object_pose(tid)
    result["place_diag"]["piece_xy_after_open_mm"] = [
        round(float(q_open[0]) * 1000, 1),
        round(float(q_open[1]) * 1000, 1),
    ]
    result["place_diag"]["piece_err_after_open_mm"] = round(
        float(np.hypot(q_open[0] - dest_xy[0], q_open[1] - dest_xy[1])) * 1000, 1
    )
    result["place_diag"]["still_held_after_open"] = bool(grasp_acquired(sim, tid))
    result["place_diag"]["piece_z_after_open_mm"] = round(float(q_open[2]) * 1000, 1)
    result["place_diag"]["release_gripper"] = round(release_g, 3)

    wrist = ("wrist_flex", "wrist_roll")
    release_lift = float(p.get("release_lift_z", 0.08))
    result["place_diag"]["airborne_after_open"] = float(q_open[2]) > surface + 0.012
    result["release_after_lift"] = release_snapshot(sim, tid)

    q_rel, qquat_rel = sim.object_pose(tid)
    result["release_position_error_mm"] = float(np.hypot(q_rel[0] - dest_xy[0], q_rel[1] - dest_xy[1])) * 1000
    result["place_diag"]["release_xy_mm"] = [
        round(float(q_rel[0]) * 1000, 1),
        round(float(q_rel[1]) * 1000, 1),
    ]
    result["place_diag"]["release_err_mm"] = round(result["release_position_error_mm"], 1)
    still_held = bool(grasp_acquired(sim, tid) and float(q_rel[2]) > surface + 0.012)
    if off_board:
        seated = bowl_contains(q_rel, dest_xy)
        upright = True
    else:
        seated = object_on_board(sim, surface, object_id=tid)
        result["place_diag"]["tilt_deg"] = round(float(queen_tilt_deg(qquat_rel)), 1)
        upright = queen_upright(qquat_rel, tilt_max=0.5)
    result["place_diag"]["still_held"] = still_held
    result["place_diag"]["jaw_contact"] = bool(grasp_acquired(sim, tid))
    result["place_diag"]["seated"] = bool(seated)
    result["place_diag"]["upright"] = bool(upright)
    result["place_diag"]["piece_z_mm"] = round(float(q_rel[2]) * 1000, 1)
    result["place_diag"]["off_board"] = bool(off_board)
    log_state(sim, "009_release", extra={"release_err_mm": round(result["release_position_error_mm"], 2)})
    m.q_rel = q_rel
    m.seated = seated
    m.upright = upright
    m.release_lift = release_lift
    m.wrist = wrist


@register("release", "open_at_place")
class OpenAtPlace:
    """Open to the release aperture at the place pose."""

    def release(self, m):
        return release(m)
