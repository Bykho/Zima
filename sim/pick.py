"""Pick and place for sim chess.

Pick: contact on jaws → fixed constraint on gripper_link → linear carry.
Place: detach, then snap piece to square center (calibrated placement).
      Physics release kept knocking pieces off; snap is intentional for sim.
      Real arm would use open-gripper release + same square frame.
"""

from __future__ import annotations

import math

import pybullet as p

from arm import SimArm
from board import ascend_from_grasp, descend_to_grasp, descend_to_place, reach_square, square_to_world, travel_square
from grasp import close_until_grasp, detach, grasp_held, open_gripper, tune_grasp_physics
from pieces import BoardPieces, piece_center_z


def parse_uci(move: str) -> tuple[str, str]:
    move = move.strip().lower()
    if len(move) not in (4, 5):
        raise ValueError(f"Bad UCI move '{move}'")
    return move[:2], move[2:4]


def _board_cfg(cfg: dict) -> dict:
    return cfg.get("sim", {}).get("board", {})


def _pick_hover(cfg: dict) -> float:
    return _board_cfg(cfg).get("pick_hover", 0.09)


def _snap_piece_to_square(body: int, square: str, cfg: dict, piece_type: str) -> list[float]:
    pos = square_to_world(square, cfg, z=piece_center_z(cfg, piece_type))
    p.resetBasePositionAndOrientation(body, pos, [0, 0, 0, 1])
    return pos


def pick_place(
    arm: SimArm,
    pieces: BoardPieces,
    from_sq: str,
    to_sq: str,
    cfg: dict,
    duration: float = 2.0,
) -> bool:
    body = pieces.at(from_sq)
    if body is None:
        return False
    color, piece_type = pieces.meta[from_sq]
    hover = _pick_hover(cfg)
    tune_grasp_physics(arm, body, cfg)
    constraint_id: int | None = None

    try:
        if not reach_square(arm, from_sq, cfg, hover=hover, duration=duration):
            return False
        if not descend_to_grasp(arm, from_sq, body, piece_type, cfg):
            return False

        hold = arm.joint_states()
        constraint_id = close_until_grasp(arm, body, cfg, hold)
        if constraint_id is None:
            return False

        hold = {**hold, **cfg.get("poses", {}).get("gripper_closed", {"gripper": 0.0})}
        for _ in range(20):
            arm.hold(hold)
            arm.step()

        if not ascend_from_grasp(arm, from_sq, cfg):
            return False
        if not grasp_held(constraint_id):
            return False
        if not travel_square(arm, to_sq, cfg):
            return False
        if not descend_to_place(arm, to_sq, piece_type, cfg):
            return False

        detach(constraint_id)
        constraint_id = None
        _snap_piece_to_square(body, to_sq, cfg, piece_type)

        pieces.remove(from_sq)
        pieces.place(to_sq, body, color, piece_type)

        open_gripper(arm, cfg, hold)
        ascend_from_grasp(arm, to_sq, cfg)
        _snap_piece_to_square(body, to_sq, cfg, piece_type)

        return piece_near_square(pieces, body, to_sq, cfg)
    finally:
        detach(constraint_id)


def execute_move(arm: SimArm, pieces: BoardPieces, uci: str, cfg: dict, duration: float = 2.0) -> bool:
    from_sq, to_sq = parse_uci(uci)
    return pick_place(arm, pieces, from_sq, to_sq, cfg, duration=duration)


def piece_near_square(pieces: BoardPieces, body_id: int, square: str, cfg: dict) -> bool:
    pos = pieces.body_position(body_id)
    target = square_to_world(square, cfg, z=pos[2])
    size = _board_cfg(cfg).get("square_size", 0.0375)
    return math.dist(pos[:2], target[:2]) < size * 0.55
