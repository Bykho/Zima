"""Chess board frame in sim — square coords, visuals, reach helpers.

EE links (config.yaml):
  board.ee_link   — hover / reach_square (gripper_frame_link, jaw clearance)
  grasp.ik_link   — descend / carry while welded (gripper_link, jaw contact point)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pybullet as p

if TYPE_CHECKING:
    from arm import SimArm

FILES = "abcdefgh"


def parse_square(name: str) -> tuple[int, int]:
    name = name.strip().lower()
    if len(name) != 2 or name[0] not in FILES or name[1] not in "12345678":
        raise ValueError(f"Bad square '{name}' (use e.g. e4)")
    file_i = FILES.index(name[0])
    rank = int(name[1])
    return file_i, rank


def square_to_xy(square: str, origin: list[float], square_size: float, yaw: float = 0.0) -> tuple[float, float]:
    """Board frame: a1 at origin corner, +file = +x, +rank = +y."""
    file_i, rank = parse_square(square)
    lx = (file_i + 0.5) * square_size
    ly = (rank - 0.5) * square_size
    cx, cy = origin[0], origin[1]
    c, s = math.cos(yaw), math.sin(yaw)
    x = cx + c * lx - s * ly
    y = cy + s * lx + c * ly
    return x, y


def square_to_world(square: str, cfg: dict, z: float | None = None) -> list[float]:
    board = cfg.get("sim", {}).get("board", {})
    origin = board.get("origin", [0.22, -0.10, 0.0])
    size = board.get("square_size", 0.0375)
    yaw = board.get("yaw", 0.0)
    height = z if z is not None else board.get("surface_z", 0.02)
    x, y = square_to_xy(square, origin, size, yaw)
    return [x, y, height]


def add_checkerboard(cfg: dict) -> list[int]:
    """Colored 8x8 squares (visual only, no collision)."""
    board = cfg.get("sim", {}).get("board", {})
    origin = board.get("origin", [0.22, -0.10, 0.0])
    size = board.get("square_size", 0.0375)
    yaw = board.get("yaw", 0.0)
    z = board.get("surface_z", 0.02) + 0.0005
    half = size * 0.49
    thickness = 0.001
    light = board.get("light_square", [0.86, 0.78, 0.58, 1.0])
    dark = board.get("dark_square", [0.42, 0.26, 0.12, 1.0])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, thickness / 2])
    ids = []
    for file_i in range(8):
        for rank_i in range(8):
            square = f"{FILES[file_i]}{rank_i + 1}"
            rgba = light if (file_i + rank_i) % 2 == 0 else dark
            pos = square_to_world(square, cfg, z=z)
            body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=vis,
                basePosition=pos,
                baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
            )
            p.changeVisualShape(body, -1, rgbaColor=rgba)
            ids.append(body)
    return ids


def add_board_visuals(cfg: dict) -> list[int]:
    """Checker squares + optional grid lines."""
    board = cfg.get("sim", {}).get("board", {})
    ids = []
    if board.get("show_checker", True):
        ids.extend(add_checkerboard(cfg))
    if not board.get("show_grid", True):
        return ids

    origin = board.get("origin", [0.22, -0.10, 0.0])
    size = board.get("square_size", 0.0375)
    yaw = board.get("yaw", 0.0)
    z = board.get("surface_z", 0.02)
    span = 8 * size
    ids = []
    c, s = math.cos(yaw), math.sin(yaw)

    def rot(lx, ly):
        return origin[0] + c * lx - s * ly, origin[1] + s * lx + c * ly

    for i in range(9):
        t = i * size
        x0, y0 = rot(t, 0)
        x1, y1 = rot(t, span)
        ids.append(p.addUserDebugLine([x0, y0, z], [x1, y1, z], [0.2, 0.2, 0.2], 1.5, 0))
        x0, y0 = rot(0, t)
        x1, y1 = rot(span, t)
        ids.append(p.addUserDebugLine([x0, y0, z], [x1, y1, z], [0.2, 0.2, 0.2], 1.5, 0))
    return ids


def mark_square(square: str, cfg: dict, color: list[float] | None = None) -> int:
    color = color or [0.1, 0.7, 0.2]
    pos = square_to_world(square, cfg, z=cfg.get("sim", {}).get("board", {}).get("surface_z", 0.02) + 0.001)
    size = cfg.get("sim", {}).get("board", {}).get("square_size", 0.0375) * 0.9
    return p.addUserDebugText(square, [pos[0], pos[1], pos[2] + 0.01], textColorRGB=color, textSize=1.2)


def link_index(arm: SimArm, link_name: str) -> int:
    for i in range(p.getNumJoints(arm.robot_id)):
        if p.getJointInfo(arm.robot_id, i)[12].decode() == link_name:
            return i
    raise RuntimeError(f"Link '{link_name}' not found")


def ik_joints(arm: SimArm, sol: tuple) -> dict[str, float]:
    from arm import JOINT_NAMES

    out: dict[str, float] = {}
    for i, name in enumerate(JOINT_NAMES):
        if name in arm.joint_indices and i < len(sol):
            out[name] = float(sol[i])
    return out


def ee_error(arm: SimArm, ee_idx: int, target: list[float]) -> float:
    pos = p.getLinkState(arm.robot_id, ee_idx)[0]
    return math.dist(pos, target)


def reach_link(
    arm: SimArm,
    target: list[float],
    link_name: str,
    cfg: dict,
    duration: float = 2.5,
    tolerance: float = 0.08,
    each_step=None,
    rpy: list[float] | None = None,
) -> bool:
    """IK so a specific link (e.g. gripper_link) reaches a world point."""
    ee = link_index(arm, link_name)
    kwargs: dict = dict(maxNumIterations=400, residualThreshold=1e-5)
    if rpy is not None:
        kwargs["targetOrientation"] = p.getQuaternionFromEuler(rpy)
    sol = p.calculateInverseKinematics(arm.robot_id, ee, target, **kwargs)
    joints = ik_joints(arm, sol)
    if not arm.move_to(joints, duration=duration, tolerance=0.03, each_step=each_step):
        return False
    return ee_error(arm, ee, target) < tolerance


def reach_world(
    arm: SimArm,
    target: list[float],
    cfg: dict,
    duration: float = 2.5,
    tolerance: float = 0.08,
    each_step=None,
    use_orientation: bool = False,
) -> bool:
    """IK to a world-frame point with the configured EE link."""
    board = cfg.get("sim", {}).get("board", {})
    link = board.get("ee_link", "gripper_frame_link")
    rpy = board.get("ee_rpy") if use_orientation else None
    return reach_link(arm, target, link, cfg, duration=duration, tolerance=tolerance, each_step=each_step, rpy=rpy)


def _gripper_travel_z(cfg: dict) -> float:
    b = cfg.get("sim", {}).get("board", {})
    return b.get("gripper_travel_z", b.get("surface_z", 0.02) + b.get("pick_hover", 0.09) + 0.03)


def move_gripper_linear(
    arm: SimArm,
    end: list[float],
    cfg: dict,
    steps: int | None = None,
    duration: float = 0.3,
) -> bool:
    """Move grasp IK link in small straight-line steps (safe while carrying)."""
    g = cfg.get("sim", {}).get("grasp", {})
    link = g.get("ik_link", "gripper_link")
    link_idx = link_index(arm, link)
    steps = steps or g.get("travel_steps", 10)
    tol = g.get("travel_tolerance", 0.14)
    start = list(p.getLinkState(arm.robot_id, link_idx)[0])
    for i in range(steps):
        u = (i + 1) / steps
        target = [start[j] + (end[j] - start[j]) * u for j in range(3)]
        reach_link(arm, target, link, cfg, duration=duration, tolerance=0.12)
    return ee_error(arm, link_idx, end) < tol


def travel_square(arm: SimArm, square: str, cfg: dict) -> bool:
    """Move carried piece over a square at travel height."""
    sq_xy = square_to_world(square, cfg)[:2]
    z = _gripper_travel_z(cfg)
    return move_gripper_linear(arm, [sq_xy[0], sq_xy[1], z], cfg)


def ascend_from_grasp(arm: SimArm, square: str, cfg: dict) -> bool:
    """Lift after grasp without breaking the weld."""
    sq_xy = square_to_world(square, cfg)[:2]
    z = _gripper_travel_z(cfg)
    return move_gripper_linear(arm, [sq_xy[0], sq_xy[1], z], cfg)


def descend_to_grasp(
    arm: SimArm,
    square: str,
    piece_id: int,
    piece_type: str,
    cfg: dict,
    require_contact: bool = True,
) -> bool:
    """Small Z steps toward piece — avoids swooping IK that knocks pieces off the square."""
    import math

    from grasp import jaw_contacts
    from pieces import piece_center_z

    g = cfg.get("sim", {}).get("grasp", {})
    b = cfg.get("sim", {}).get("board", {})
    link = g.get("ik_link", "gripper_link")
    link_idx = link_index(arm, link)
    size = b.get("square_size", 0.0375)
    sq_xy = square_to_world(square, cfg)[:2]
    grasp_z = piece_center_z(cfg, piece_type)
    steps = g.get("descend_steps", 14)
    tol = g.get("travel_tolerance", 0.14)

    start = list(p.getLinkState(arm.robot_id, link_idx)[0])
    end = [sq_xy[0], sq_xy[1], grasp_z]

    for i in range(steps):
        u = (i + 1) / steps
        target = [start[j] + (end[j] - start[j]) * u for j in range(3)]
        reach_link(arm, target, link, cfg, duration=0.25, tolerance=0.12)
        if piece_id >= 0:
            piece_xy = p.getBasePositionAndOrientation(piece_id)[0][:2]
            if math.dist(piece_xy, sq_xy) > size * 0.55:
                return False
        if require_contact and jaw_contacts(arm, piece_id, cfg):
            return True

    if require_contact:
        return len(jaw_contacts(arm, piece_id, cfg)) > 0
    return ee_error(arm, link_idx, end) < tol


def descend_to_place(arm: SimArm, square: str, piece_type: str, cfg: dict) -> bool:
    """Lower welded piece onto square — no contact check."""
    return descend_to_grasp(arm, square, -1, piece_type, cfg, require_contact=False)


def reach_square(
    arm: SimArm,
    square: str,
    cfg: dict,
    hover: float = 0.08,
    duration: float = 2.5,
    each_step=None,
) -> bool:
    """IK to hover above square center."""
    board = cfg.get("sim", {}).get("board", {})
    target = square_to_world(square, cfg, z=board.get("surface_z", 0.02) + hover)
    return reach_world(arm, target, cfg, duration=duration, each_step=each_step)
