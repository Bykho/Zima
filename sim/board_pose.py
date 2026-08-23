"""Board pose from four red corner dots in the wrist camera.

No extra cameras. Detect the markers, unproject with depth, label them with
the yaml prior, then every square is relative to that measured frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pybullet as p

from arm import ROOT, SimArm
from board import apply_board_pose, corner_world, reach_square, square_to_world
from camera import WristCamera
from world import board_center, unproject

FRAMES = ROOT / "sim" / "frames"
CORNER_NAMES = ("a1", "h1", "a8", "h8")


@dataclass
class BoardPose:
    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray
    square_size: float
    yaw: float
    corners: dict[str, np.ndarray]
    n_views: int = 0
    rgb: np.ndarray | None = None
    shot: dict | None = None
    debug_ids: list[int] = field(default_factory=list)

    def apply(self, cfg: dict) -> None:
        apply_board_pose(cfg, self.origin[:2], self.yaw, self.square_size)

    def world_to_board(self, pts: np.ndarray) -> np.ndarray:
        d = np.asarray(pts, dtype=np.float64) - self.origin
        return np.stack([d @ self.x_axis, d @ self.y_axis, d @ self.z_axis], axis=-1)

    def board_to_world(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        return self.origin + pts[:, 0:1] * self.x_axis + pts[:, 1:2] * self.y_axis + pts[:, 2:3] * self.z_axis


def expected_marker_xyz(cfg: dict) -> dict[str, np.ndarray]:
    board = cfg.get("sim", {}).get("board", {})
    radius = float(board.get("marker_radius", 0.012))
    out = {}
    for name, pos in corner_world(cfg).items():
        out[name] = np.array([pos[0], pos[1], pos[2] + radius], dtype=np.float64)
    return out


def red_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return (r > 150) & (g < 80) & (b < 80) & (r > g + 50) & (r > b + 50)


def cluster_xyz(points: np.ndarray, merge: float = 0.03, min_count: int = 4) -> np.ndarray:
    """Greedy 3D clusters — four well-separated spheres, no sklearn."""
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    unused = np.ones(len(points), dtype=bool)
    centers: list[np.ndarray] = []
    counts: list[int] = []
    while unused.any():
        seed = int(np.where(unused)[0][0])
        member = unused & (np.linalg.norm(points - points[seed], axis=1) < merge)
        c = points[member].mean(axis=0)
        member = unused & (np.linalg.norm(points - c, axis=1) < merge)
        c = points[member].mean(axis=0)
        n = int(member.sum())
        unused[member] = False
        if n >= min_count:
            centers.append(c)
            counts.append(n)
    if not centers:
        return np.zeros((0, 3), dtype=np.float64)
    order = np.argsort(counts)[::-1]
    return np.asarray(centers, dtype=np.float64)[order]


def red_clusters(shot: dict, skip_bodies: set[int] | None = None) -> np.ndarray:
    pts, cols = unproject(shot, skip_bodies=skip_bodies, stride=1)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    keep = (cols[:, 0] > 150) & (cols[:, 1] < 80) & (cols[:, 2] < 80)
    keep &= cols[:, 0] > cols[:, 1] + 50
    keep &= cols[:, 0] > cols[:, 2] + 50
    return cluster_xyz(pts[keep])


def merge_clusters(groups: list[np.ndarray], merge: float = 0.03) -> np.ndarray:
    nonempty = [g for g in groups if len(g)]
    if not nonempty:
        return np.zeros((0, 3), dtype=np.float64)
    return cluster_xyz(np.concatenate(nonempty, axis=0), merge=merge, min_count=1)


def project_world(shot: dict, xyz) -> tuple[int, int] | None:
    view = np.asarray(shot["view"], dtype=np.float64).reshape(4, 4).T
    proj = np.asarray(shot["proj"], dtype=np.float64).reshape(4, 4).T
    h, w = shot["rgb"].shape[:2]
    clip = proj @ view @ np.array([float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0])
    if abs(clip[3]) < 1e-9:
        return None
    ndc = clip[:3] / clip[3]
    u = int((ndc[0] + 1.0) * 0.5 * w)
    v = int((1.0 - ndc[1]) * 0.5 * h)
    if u < 0 or v < 0 or u >= w or v >= h:
        return None
    return u, v


def label_corners(detected: np.ndarray, expected: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    remaining = list(range(len(detected)))
    out: dict[str, np.ndarray] = {}
    for name in CORNER_NAMES:
        e = expected[name]
        best = min(remaining, key=lambda i: float(np.linalg.norm(detected[i] - e)))
        out[name] = detected[best]
        remaining.remove(best)
    return out


def fit_pose(detected: np.ndarray, cfg: dict) -> BoardPose:
    if len(detected) < 4:
        raise ValueError(f"need 4 corner dots, got {len(detected)}")
    labeled = label_corners(detected, expected_marker_xyz(cfg))
    a1, h1, a8 = labeled["a1"], labeled["h1"], labeled["a8"]
    x = h1 - a1
    y = a8 - a1
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    if nx < 1e-6 or ny < 1e-6:
        raise ValueError("degenerate corner triangle")
    x_axis = x / nx
    y_axis = y / ny
    z_axis = np.cross(x_axis, y_axis)
    zn = float(np.linalg.norm(z_axis))
    if zn < 1e-6:
        raise ValueError("corners are colinear")
    z_axis = z_axis / zn
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
    x_axis = np.cross(y_axis, z_axis)
    square_size = 0.5 * (nx + ny) / 8.0
    origin = a1.copy()
    origin[2] = float(cfg.get("sim", {}).get("board", {}).get("origin", [0.0, 0.0, 0.0])[2])
    yaw = math.atan2(float(x_axis[1]), float(x_axis[0]))
    return BoardPose(
        origin=origin,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        square_size=float(square_size),
        yaw=yaw,
        corners=labeled,
    )


def locate(arm: SimArm, cam: WristCamera | None = None, cfg: dict | None = None) -> BoardPose:
    """Hover, find the four red dots, solve board frame, write it into cfg."""
    cfg = cfg or arm.cfg
    cam = cam or WristCamera(arm, cfg)
    loc = cfg.get("sim", {}).get("locate", {})
    squares = list(loc.get("squares") or ("e4", "a1", "a8", "h1", "h8"))
    hover = float(loc.get("hover", 0.28))
    duration = float(loc.get("move_duration", 0.5))
    settle = int(loc.get("settle_steps", 8))
    look = board_center(cfg)
    skip = {0, int(arm.robot_id)}

    prev_far = cam.far
    cam.far = float(loc.get("far", 0.85))
    groups: list[np.ndarray] = []
    last_shot = None
    try:
        for square in squares:
            if not reach_square(arm, square, cfg, hover=hover, duration=duration):
                continue
            arm.hold(arm.joint_states())
            for _ in range(settle):
                arm.step()
            shot = cam.capture_rgbd(target=look, up=[0.0, 0.0, 1.0])
            if shot is None:
                continue
            last_shot = shot
            groups.append(red_clusters(shot, skip_bodies=skip))
            if len(merge_clusters(groups)) >= 4:
                break
    finally:
        cam.far = prev_far

    detected = merge_clusters(groups)
    pose = fit_pose(detected, cfg)
    pose.n_views = len(groups)
    pose.shot = last_shot
    pose.rgb = None if last_shot is None else last_shot["rgb"]
    pose.apply(cfg)
    return pose


def draw(pose: BoardPose, cfg: dict | None = None) -> None:
    for i in pose.debug_ids:
        try:
            p.removeUserDebugItem(i)
        except Exception:
            pass
    pose.debug_ids.clear()
    if not p.isConnected():
        return
    z = float((cfg or {}).get("sim", {}).get("board", {}).get("surface_z", 0.02)) + 0.004
    o = np.array([pose.origin[0], pose.origin[1], z])
    span = 8.0 * pose.square_size
    x_end = o + pose.x_axis * span
    y_end = o + pose.y_axis * span
    pose.debug_ids.append(p.addUserDebugLine(o.tolist(), x_end.tolist(), [1, 0.15, 0.1], 4, 0))
    pose.debug_ids.append(p.addUserDebugLine(o.tolist(), y_end.tolist(), [0.15, 0.85, 0.2], 4, 0))
    for name, c in pose.corners.items():
        pose.debug_ids.append(
            p.addUserDebugText(name, [float(c[0]), float(c[1]), float(c[2]) + 0.02], [1, 0.2, 0.2], 1.1)
        )


def render(
    pose: BoardPose,
    cfg: dict,
    cloud: tuple[np.ndarray, np.ndarray] | None = None,
    path: str | Path | None = None,
) -> np.ndarray:
    """RGB with corner labels + top-down board-frame cloud."""
    import cv2

    rgb = pose.rgb
    if rgb is None:
        rgb = np.full((240, 320, 3), 18, dtype=np.uint8)
    left = rgb.copy()
    cv2.putText(left, "red dots -> board frame", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 80), 1, cv2.LINE_AA)
    for name in CORNER_NAMES:
        c = pose.corners.get(name)
        if c is None:
            continue
        uv = project_world(pose.shot, c) if pose.shot is not None else None
        if uv is not None:
            cv2.circle(left, uv, 8, (255, 40, 40), 2)
            cv2.putText(left, name, (uv[0] + 10, uv[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 80), 1, cv2.LINE_AA)

    span = 8.0 * pose.square_size
    w = h = 400
    right = np.full((h, w, 3), 22, dtype=np.uint8)
    # board rectangle
    cv2.rectangle(right, (20, 20), (w - 20, h - 20), (60, 60, 60), 1)
    if cloud is not None and len(cloud[0]):
        local = pose.world_to_board(cloud[0])
        cols = cloud[1]
        keep = (
            (local[:, 0] > -0.02)
            & (local[:, 0] < span + 0.02)
            & (local[:, 1] > -0.02)
            & (local[:, 1] < span + 0.02)
            & (local[:, 2] > -0.02)
            & (local[:, 2] < 0.08)
        )
        local, cols = local[keep], cols[keep]
        if len(local):
            u = (20 + (w - 40) * (local[:, 0] / span)).astype(int)
            v = (h - 20 - (h - 40) * (local[:, 1] / span)).astype(int)
            z = local[:, 2]
            ok = (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1)
            u, v, cols, z = u[ok], v[ok], cols[ok], z[ok]
            order = np.argsort(z)
            u, v, cols = u[order], v[order], cols[order]
            right[v, u] = cols
            right[np.clip(v + 1, 0, h - 1), u] = cols
            right[v, np.clip(u + 1, 0, w - 1)] = cols
    cv2.putText(right, "board frame (a1 bottom-left)", (12, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(
        right,
        f"yaw {math.degrees(pose.yaw):+.1f} deg   sq {pose.square_size * 1000:.1f} mm",
        (12, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    left = cv2.resize(left, (400, 300))
    pad = np.full((100, 400, 3), 18, dtype=np.uint8)
    left = np.vstack([left, pad])
    header = np.full((36, 800, 3), 18, dtype=np.uint8)
    cv2.putText(
        header,
        f"board pose   origin ({pose.origin[0]:.3f}, {pose.origin[1]:.3f})   {pose.n_views} views",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    canvas = np.vstack([header, np.hstack([left, right])])
    path = Path(path) if path else FRAMES / "board_pose.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return canvas


def capture_board_cloud(arm: SimArm, cam: WristCamera, cfg: dict, pose: BoardPose) -> tuple[np.ndarray, np.ndarray]:
    """One wrist shot after pose is known, cropped in the board frame."""
    look = square_to_world("e4", cfg)
    loc = cfg.get("sim", {}).get("locate", {})
    hover = float(loc.get("hover", 0.28))
    reach_square(arm, "e4", cfg, hover=hover, duration=float(loc.get("move_duration", 0.5)))
    arm.hold(arm.joint_states())
    for _ in range(int(loc.get("settle_steps", 8))):
        arm.step()
    shot = cam.capture_rgbd(target=look, up=[0.0, 0.0, 1.0])
    if shot is None:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    pose.rgb = shot["rgb"]
    pose.shot = shot
    return unproject(shot, skip_bodies={0, int(arm.robot_id)}, stride=1)
