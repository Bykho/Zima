"""Observe stage: where the piece is before the pick, and whether it is placed after."""

from __future__ import annotations

import numpy as np

from pickup import (
    _perceiving,
    _reach_observe,
    _xyz,
    bowl_contains,
    grasp_acquired,
    locate_queen,
    log_state,
    object_on_board,
    queen_upright,
    verify_piece,
)
from squares import nest_xyz, square_to_world
from stages import register
from stages.types import Fail, Move


def _placed(sim, cfg: dict, tid: str, dest: str, dest_xy, surface: float, off_board: bool, xy_tol: float) -> tuple[bool, float | None]:
    """Is the released piece where it should be, and how far off is it (mm)? The wrist camera judges when perceiving."""
    xyz, quat = sim.object_pose(tid)
    if off_board:
        return bool(bowl_contains(xyz, dest_xy)), None
    if _perceiving(sim):
        look = verify_piece(sim, cfg, tid, dest)
        if not look.get("seen"):
            return False, None
        standing = bool(look.get("upright"))
        xyz, _ = sim.object_pose(tid)
    else:
        standing = bool(queen_upright(quat, tilt_max=0.5) and object_on_board(sim, surface, object_id=tid))
    off = float(np.hypot(xyz[0] - dest_xy[0], xyz[1] - dest_xy[1]))
    return bool(standing and off <= xy_tol), round(off * 1000.0, 1)


def locate(m: Move):
    """Wrist look over the source; `queen_xyz` is the believed piece position."""
    sim, cfg, sc, result, tid, surface, source, use_vision, grasp_z = m.sim, m.cfg, m.sc, m.result, m.tid, m.surface, m.source, m.use_vision, m.grasp_z
    if sc.get("_from_nest"):
        # Nest is off the commanded square; do not hover the empty source.
        queen_xyz, _ = sim.object_pose(tid)
        nest_z = float(nest_xyz(cfg)[2])
        grasp_z = max(0.030, float(queen_xyz[2]) - surface + 0.022, nest_z - surface + 0.008)
        log_state(sim, "002_observe", extra={"from_nest": True, "target": _xyz(queen_xyz)})
    elif use_vision:
        prior = square_to_world(source, cfg, z=surface)
        look = _reach_observe(sim, cfg, prior, tid)
        if look is None:
            result["failure_reason"] = "IK_NO_SOLUTION"
            result["ee_requested"] = _xyz(prior)
            log_state(sim, "fail_observe", extra={"target": result["ee_requested"]})
            return Fail("observe", result["failure_reason"], "pick")
        log_state(sim, "002_observe", extra={"target": _xyz(look)})
        queen_xyz = locate_queen(sim, cfg, result, source)
        if queen_xyz is None:
            # Commanded square is already the prior. Empty cloud on rank 8
            # (b8c6 after g1f3) should not abort the move.
            queen_xyz = prior
            result["queen_detected"] = True
            result["perception_fallback"] = "square"
        else:
            result["queen_detected"] = True
    else:
        queen_xyz, _ = sim.object_pose(tid)
        observe = _reach_observe(sim, cfg, queen_xyz, tid)
        if observe is None:
            result["failure_reason"] = "IK_NO_SOLUTION"
            result["ee_requested"] = _xyz(queen_xyz)
            log_state(sim, "fail_observe", extra={"target": result["ee_requested"]})
            return Fail("observe", result["failure_reason"], "pick")
        log_state(sim, "002_observe", extra={"target": _xyz(observe)})
        queen_xyz, _ = sim.object_pose(tid)
    m.queen_xyz = queen_xyz
    m.grasp_z = grasp_z


def confirm(m: Move):
    """Placed? The robot's own check (wrist look at the destination), then the final phase log."""
    sim, cfg, result, tid, surface, dest, dest_xy, off_board, retreated, high = m.sim, m.cfg, m.result, m.tid, m.surface, m.dest, m.dest_xy, m.off_board, m.retreated, m.high
    still_held = bool(grasp_acquired(sim, tid) and float(sim.object_pose(tid)[0][2]) > surface + 0.012)
    placed, result["placed_off_mm"] = _placed(sim, cfg, tid, dest, dest_xy, surface, off_board, m.xy_tol)
    result["placement_success"] = placed and not still_held
    result["retreat_success"] = bool(retreated)
    if not retreated:
        result["ik_fail_phase"] = "010_final"
        log_state(sim, "fail_retreat", extra={"target": _xyz(high)})
    log_state(sim, "010_final")

    if not result["placement_success"]:
        result["failure_reason"] = "RELEASE_FAILURE" if still_held else "PLACEMENT_POSITION_ERROR"
        return Fail("confirm", result["failure_reason"], "place")


@register("board_observe", "wrist_scan")
class WristScan:
    """Whole-board wrist scan, classified and written into the belief (`perceive.perceive_board`)."""

    def scan(self, sim, cfg: dict) -> dict:
        from perceive import perceive_board

        return perceive_board(sim, cfg)


@register("observe", "wrist_vision")
class WristVision:
    """Reach an observe pose over the source and estimate the piece XY from the wrist depth image."""

    def locate(self, m):
        return locate(m)

    def confirm(self, m):
        return confirm(m)
