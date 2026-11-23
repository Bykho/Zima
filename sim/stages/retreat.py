"""Retreat stage: back the pads off the placed piece and rise clear; unhook frees the hand from the last move."""

from __future__ import annotations

import numpy as np

import carry as plumb
from clearance import neighbor_approach_gap, obstacle_bodies
from ik import EE_SITE
from mj import ARM_JOINTS, Sim, probing
from pickup import (
    _clear_then_rise,
    _js_raise_ee,
    _outward_xy,
    _robot_bid_array,
    _safe_lin,
    _tid,
    _transit_z,
    bystander_contact,
    grasp_acquired,
    open_gripper,
    settle_queen,
)
from robot import world_of
from scenario import PIECES, body_kind
from stages import register
from stages.types import Move


def _jaws_on_piece(sim: Sim) -> bool:
    return _piece_in_jaws(sim) is not None


def _piece_in_jaws(sim: Sim) -> str | None:
    if not grasp_acquired(sim):
        return None
    return getattr(sim, "held", None) or _tid(sim)


_CENTRE_SAMPLES = 8


@probing
def _roomiest_along(sim: Sim, tid: str, start, shift, v_local) -> tuple[np.ndarray | None, float]:
    """EE point on start -> start + shift where the open hand stands farthest
    from the piece, and that gap. The pad midpoint is only the centre at pad
    height; the hinged moving jaw leans in above it. Restores qpos."""
    snap = sim.qpos()
    best, room = None, -1.0
    try:
        for k in range(_CENTRE_SAMPLES + 1):
            p = start + shift * (k / _CENTRE_SAMPLES)
            if not plumb.solve_plumb(sim, p, v_local, plumb.LIFT_TILT_DEG):
                continue
            g = float(neighbor_approach_gap(sim, [tid]))
            if g > room + 1e-4:
                best, room = p, g
        return best, room
    finally:
        sim.set_qpos({n: snap[n] for n in ARM_JOINTS}, hold=True)


# A piece left leaning on the fixed jaw rocks back upright if the jaw leaves
# slowly; a quick slide drags it past its tipping angle.
_RELEASE_SLIDE_MPS = 0.02


@probing
def _plan_slide_rise(sim: Sim, centre, top, v_local, avoid: list[str]) -> tuple[list | None, list | None]:
    snap = sim.qpos()
    try:
        slide = plumb.plan(sim, [(centre, plumb.LIFT_TILT_DEG)], v_local, avoid, gap=0.002)
        if slide is None:
            return None, None
        sim.set_qpos(slide[-1], hold=True)
        return slide, plumb.plan(sim, [(top, plumb.LIFT_TILT_DEG)], v_local, avoid, gap=0.002)
    finally:
        sim.set_qpos({n: snap[n] for n in ARM_JOINTS}, hold=True)


def _centred_retreat(sim: Sim, tid: str, avoid: list[str], z_up: float, jaw_clear_m: float) -> dict:
    """After release: slide the open hand along the pinch axis to where it
    stands farthest from the piece, then rise holding the hand's orientation
    until the jaws clear the piece top, then up to z_up leaning freely.

    Only the moving jaw opens, so the piece is left against the fixed pad.
    Rising from there drags it; a rise that re-pitches the hand swings a jaw
    into it. The room is measured on the robot's own model at the pose it set
    the piece down, and the rise is refused if it closes on the piece.
    """
    fixed, moving = plumb._pad_xyz(sim, plumb.PAD_GEOMS[0]), plumb._pad_xyz(sim, plumb.PAD_GEOMS[1])
    u = moving - fixed
    u[2] = 0.0
    if float(np.linalg.norm(u)) < 1e-6:
        return {"ok": False, "why": "pads"}
    u /= float(np.linalg.norm(u))
    q, _ = sim.object_pose(tid)
    mid = 0.5 * (fixed + moving)
    shift = float((np.asarray(q, dtype=float) - mid) @ u) * u
    gpos, R = sim.site_pose(EE_SITE)
    v_local = np.asarray(R, dtype=float).reshape(3, 3).T @ plumb.UP
    centre, room = _roomiest_along(sim, tid, np.asarray(gpos, dtype=float), shift, v_local)
    if centre is None:
        return {"ok": False, "why": "ik"}
    z_clear = float(gpos[2]) + jaw_clear_m
    top = [centre[0], centre[1], max(float(centre[2]), z_clear)]
    out = {"shift_mm": round(float(np.linalg.norm(centre - np.asarray(gpos))) * 1000, 1), "room_mm": round(room * 1000, 1)}
    slide, rise = _plan_slide_rise(sim, centre, top, v_local, [*avoid, tid])
    if rise is None:
        return {**out, "ok": False, "why": getattr(sim, "plumb_fail", None)}
    plumb.follow(sim, slide, _RELEASE_SLIDE_MPS)
    plumb.follow(sim, rise, plumb.PLUMB_LOWER_SPEED_MPS)
    out["ok"] = True
    high = top[2] >= z_up or None
    if not high:
        rest = plumb.plan(sim, [([centre[0], centre[1], float(z_up)], plumb.AIRBORNE_TILT_DEG)], v_local, avoid, gap=0.002)
        high = rest is not None and plumb.follow(sim, rest, plumb.PLUMB_SPEED_MPS)
    out["high"] = bool(high)
    return out


_BEARING_NZ = 0.7  # contact normal within ~45 deg of vertical


def _hand_bears_on(sim: Sim, tid: str) -> bool:
    """A robot link presses on the piece from above or below (a lip or crown
    caught on a jaw). Sliding the hand out then drags the piece by friction."""
    w = world_of(sim) if getattr(sim, "cheat_state", True) else sim
    b1, b2 = w.contact_bodies()
    if b1.size == 0:
        return False
    robot = _robot_bid_array(sim)
    me = sim.body_id(tid)
    hit = (np.isin(b1, robot) & (b2 == me)) | (np.isin(b2, robot) & (b1 == me))
    nz = np.abs(w.data.contact.frame[: b1.size, 2])
    return bool((hit & (nz > _BEARING_NZ)).any())


def unhook(m: Move):
    """Free the hand from the last move: release, back off, open."""
    sim, cfg, tid, open_g, release_open = m.sim, m.cfg, m.tid, m.open_g, m.release_open
    gpos, _ = sim.site_pose(EE_SITE)
    hugging = _jaws_on_piece(sim)
    sim._pickup_phase = "000_unhook"
    if hugging:
        # Pads still wrap the last man. Open to release, then the same
        # clear-then-rise as after place — a raw +Z here combs the lip.
        wrap = _piece_in_jaws(sim)
        open_gripper(sim, release_open, steps=80)
        settle_queen(sim, steps=24)
        if wrap:
            _clear_then_rise(sim, cfg, wrap, z_min=0.11)
        else:
            _js_raise_ee(sim, z_min=0.11)
        open_gripper(sim, open_g, steps=40)
    elif float(gpos[2]) < 0.09:
        # Clear of the last man: step out, then up.
        away = _outward_xy(gpos[:2], cfg, 0.035)
        wrist = ("wrist_flex", "wrist_roll")
        others = obstacle_bodies(sim, skip={tid})
        _safe_lin(
            sim,
            [float(away[0]), float(away[1]), float(gpos[2])],
            cfg,
            wrist,
            obstacles=others,
            skip={tid},
            duration=0.45,
        )
        _js_raise_ee(sim, z_min=0.11)
        open_gripper(sim, open_g, steps=40)
    else:
        open_gripper(sim, open_g, steps=40)
    settle_queen(sim, steps=48)


def back_off(m: Move):
    """Back the pads off the piece and rise clear."""
    sim, cfg, result, tid, dest_xy, off_board, q_rel, seated, upright, release_lift, wrist = m.sim, m.cfg, m.result, m.tid, m.dest_xy, m.off_board, m.q_rel, m.seated, m.upright, m.release_lift, m.wrist
    gpos, _ = sim.site_pose(EE_SITE)
    touching = bool(grasp_acquired(sim, tid))
    result["place_diag"]["jaw_contact_before_retreat"] = touching
    z_up = max(
        float(gpos[2]) + release_lift,
        float(result.get("transit_z") or 0.0),
        _transit_z(sim, cfg, skip={tid}),
    )
    high = [float(gpos[0]), float(gpos[1]), z_up]
    retreated = True
    near_dest = float(np.hypot(q_rel[0] - dest_xy[0], q_rel[1] - dest_xy[1])) < 0.028
    straight_up = None
    if not off_board:
        spec = PIECES.get(body_kind(tid) or tid) or {}
        jaw_clear = float(spec.get("height", 0.055)) - float(spec.get("preferred_grasp_height", 0.01)) + 0.01
        avoid = [n for n in obstacle_bodies(sim, skip={tid}) if n != "board"]
        bearing = _hand_bears_on(sim, tid)
        result["place_diag"]["hand_bears_on_piece"] = bearing
        centred = {"ok": False, "why": "bearing"} if bearing else _centred_retreat(sim, tid, avoid, z_up, jaw_clear)
        result["place_diag"]["retreat_centred"] = centred
        if centred["ok"]:
            straight_up = []
            if not centred["high"]:
                g_top, _ = sim.site_pose(EE_SITE)
                _safe_lin(sim, [float(g_top[0]), float(g_top[1]), z_up], cfg, wrist,
                          obstacles=obstacle_bodies(sim, skip={tid}), skip={tid})
        else:
            straight_up = plumb.retreat(sim, avoid, z_up, jaw_clear, touching=lambda: bystander_contact(sim, [tid]))
        result["place_diag"]["retreat_straight"] = straight_up is not None or getattr(sim, "plumb_fail", None)
    if straight_up is not None:
        retreated = True
    elif (seated or (upright and near_dest)) and not off_board:
        # Get the pads off the lip, then rise. Skipping that check (or
        # `_safe_lin(skip=tid)`) combs the crown and knocks the piece.
        # "Seated" needs board contact; a man still on the pads is upright
        # on dest and must take this path, not a floored +Z.
        leave = _clear_then_rise(sim, cfg, tid, z_min=z_up)
        result["place_diag"]["pad_clear_mm"] = leave.get("pad_clear_mm", 0.0)
        result["place_diag"]["retreat_cleared"] = bool(leave.get("cleared"))
        result["place_diag"]["short_retreat"] = not bool(leave.get("rose"))
        retreated = True
    elif float(gpos[2]) < z_up - 0.01:
        retreated = _safe_lin(
            sim, high, cfg, wrist, obstacles=obstacle_bodies(sim, skip={tid}), skip={tid}
        )
    settle_queen(sim, steps=48)
    m.retreated = retreated
    m.high = high


@register("retreat", "pad_backoff")
class PadBackoff:
    """Back the pads off the piece and rise clear; also frees the hand from the last move (unhook)."""

    def unhook(self, m):
        return unhook(m)

    def back_off(self, m):
        return back_off(m)
