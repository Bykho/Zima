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


def corner_xy(cfg: dict) -> dict[str, tuple[float, float]]:
    """Outer board corners in world XY — a1 is the origin corner."""
    board = cfg.get("sim", {}).get("board", {})
    origin = board.get("origin", [0.22, -0.10, 0.0])
    size = float(board.get("square_size", 0.0375))
    yaw = float(board.get("yaw", 0.0))
    span = 8.0 * size
    c, s = math.cos(yaw), math.sin(yaw)

    def rot(lx: float, ly: float) -> tuple[float, float]:
        return origin[0] + c * lx - s * ly, origin[1] + s * lx + c * ly

    return {"a1": rot(0.0, 0.0), "h1": rot(span, 0.0), "a8": rot(0.0, span), "h8": rot(span, span)}


def corner_world(cfg: dict, z: float | None = None) -> dict[str, list[float]]:
    board = cfg.get("sim", {}).get("board", {})
    height = float(board.get("surface_z", 0.02) if z is None else z)
    return {name: [xy[0], xy[1], height] for name, xy in corner_xy(cfg).items()}


def apply_board_pose(cfg: dict, origin_xy, yaw: float, square_size: float | None = None) -> None:
    """Make square_to_world / reach use a measured board frame."""
    board = cfg.setdefault("sim", {}).setdefault("board", {})
    z = float(board.get("origin", [0.22, -0.10, 0.0])[2]) if board.get("origin") else 0.0
    board["origin"] = [float(origin_xy[0]), float(origin_xy[1]), z]
    board["yaw"] = float(yaw)
    if square_size is not None:
        board["square_size"] = float(square_size)


def add_table(cfg: dict) -> int:
    """Collision slab whose top is board surface_z — pieces were falling through the visual-only squares onto the ground plane."""
    board = cfg.get("sim", {}).get("board", {})
    origin = board.get("origin", [0.22, -0.10, 0.0])
    size = float(board.get("square_size", 0.0375))
    yaw = float(board.get("yaw", 0.0))
    surface = float(board.get("surface_z", 0.02))
    span = 8.0 * size
    margin = 0.04
    thick = 0.02
    half = [span / 2 + margin, span / 2 + margin, thick / 2]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.18, 0.16, 0.14, 1.0])
    cx = origin[0] + span / 2
    cy = origin[1] + span / 2
    cz = surface - thick / 2
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[cx, cy, cz],
        baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
    )


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


def add_corner_markers(cfg: dict) -> list[int]:
    """Red spheres on the four outer corners — wrist-cam fiducials for board pose."""
    board = cfg.get("sim", {}).get("board", {})
    radius = float(board.get("marker_radius", 0.012))
    rgba = list(board.get("marker_rgba", [0.95, 0.05, 0.05, 1.0]))
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
    ids = []
    for pos in corner_world(cfg).values():
        ids.append(
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=vis,
                basePosition=[pos[0], pos[1], pos[2] + radius],
            )
        )
    return ids


def add_grid_lines(cfg: dict) -> list[int]:
    board = cfg.get("sim", {}).get("board", {})
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


def add_board_visuals(cfg: dict) -> list[int]:
    """Table collision + checker squares + red corner dots + optional grid lines."""
    board = cfg.get("sim", {}).get("board", {})
    ids = [add_table(cfg)]
    if board.get("show_checker", True):
        ids.extend(add_checkerboard(cfg))
    if board.get("show_markers", True):
        ids.extend(add_corner_markers(cfg))
    if board.get("show_grid", True):
        ids.extend(add_grid_lines(cfg))
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
