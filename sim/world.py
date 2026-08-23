"""Wrist-cam scan → one 3D world-frame point cloud.

Move the arm, take RGB-D from board corners + center, unproject with that
frame's camera pose, keep the table volume, fuse by closest camera per voxel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pybullet as p

from arm import ROOT, SimArm
from board import reach_square, square_to_world
from camera import WristCamera

DEFAULT_SQUARES = ("a1", "a8", "h1", "h8", "e4")
FRAMES = ROOT / "sim" / "frames"


@dataclass
class World:
    points: np.ndarray
    colors: np.ndarray
    n_views: int = 0
    debug_ids: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self) == 0:
            z = np.zeros(3)
            return z, z
        return self.points.min(axis=0), self.points.max(axis=0)


def workspace_aabb(cfg: dict) -> np.ndarray:
    """Loose box around the board — prior for cropping, not the world itself."""
    board = cfg.get("sim", {}).get("board", {})
    scan_cfg = cfg.get("sim", {}).get("scan", {})
    origin = np.asarray(board.get("origin", [0.22, -0.10, 0.0]), dtype=np.float32)
    span = 8.0 * float(board.get("square_size", 0.0375))
    margin = float(scan_cfg.get("margin", 0.04))
    zmin = float(scan_cfg.get("z_min", 0.015))
    zmax = float(scan_cfg.get("z_max", 0.12))
    return np.array(
        [origin[0] - margin, origin[1] - margin, zmin, origin[0] + span + margin, origin[1] + span + margin, zmax],
        dtype=np.float32,
    )


def crop_workspace(points: np.ndarray, colors: np.ndarray, aabb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points, colors
    keep = (
        (points[:, 0] >= aabb[0])
        & (points[:, 1] >= aabb[1])
        & (points[:, 2] >= aabb[2])
        & (points[:, 0] <= aabb[3])
        & (points[:, 1] <= aabb[4])
        & (points[:, 2] <= aabb[5])
    )
    return points[keep], colors[keep]


def depth_meters(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    """PyBullet depth buffer [0,1] → eye-space Z in meters."""
    return far * near / (far - (far - near) * np.clip(depth, 0.0, 1.0 - 1e-6))


def unproject(shot: dict, skip_bodies: set[int] | None = None, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Pixels + depth + view/proj → world XYZ and RGB."""
    rgb = shot["rgb"][::stride, ::stride]
    depth = shot["depth"][::stride, ::stride]
    seg = shot["seg"][::stride, ::stride]
    h, w = depth.shape
    near, far = float(shot["near"]), float(shot["far"])

    z_eye = depth_meters(depth, near, far)
    keep = (depth < 0.999) & (z_eye > near) & (z_eye < far * 0.95)
    if skip_bodies:
        keep &= ~np.isin(seg & 0xFFFFFF, list(skip_bodies))
    if not np.any(keep):
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    ys, xs = np.where(keep)
    ndc_x = (xs.astype(np.float32) + 0.5) / w * 2.0 - 1.0
    ndc_y = 1.0 - (ys.astype(np.float32) + 0.5) / h * 2.0
    ndc_z = depth[ys, xs] * 2.0 - 1.0
    clip = np.stack([ndc_x, ndc_y, ndc_z, np.ones(len(xs), dtype=np.float32)], axis=1)

    view = np.asarray(shot["view"], dtype=np.float32).reshape(4, 4).T
    proj = np.asarray(shot["proj"], dtype=np.float32).reshape(4, 4).T
    inv = np.linalg.inv(proj @ view)
    world = clip @ inv.T
    world = world[:, :3] / world[:, 3:4]
    colors = rgb[ys, xs]
    finite = np.isfinite(world).all(axis=1)
    return world[finite], colors[finite]


def board_center(cfg: dict) -> list[float]:
    a1 = square_to_world("a1", cfg)
    h8 = square_to_world("h8", cfg)
    z = cfg.get("sim", {}).get("board", {}).get("surface_z", 0.02)
    return [(a1[0] + h8[0]) / 2, (a1[1] + h8[1]) / 2, z]


def fuse(
    views: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    voxel: float,
    aabb: np.ndarray | None = None,
) -> World:
    """3D stitch: one sample per voxel, closest camera wins."""
    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for pts, cols, eye in views:
        if aabb is not None:
            pts, cols = crop_workspace(pts, cols, aabb)
        if len(pts) == 0:
            continue
        rng = np.linalg.norm(pts - np.asarray(eye, dtype=np.float32), axis=1)
        chunks.append((pts, cols, rng))
    if not chunks:
        return World(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), n_views=len(views))
    points = np.concatenate([c[0] for c in chunks], axis=0)
    colors = np.concatenate([c[1] for c in chunks], axis=0)
    ranges = np.concatenate([c[2] for c in chunks], axis=0)
    keys = np.floor(points / voxel).astype(np.int32)
    order = np.lexsort((ranges, keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
    keep = order[first]
    return World(points[keep], colors[keep], n_views=len(views))


def scan(arm: SimArm, cam: WristCamera | None = None, cfg: dict | None = None) -> World:
    """Hover over board corners + center, look at board center, fuse in 3D."""
    cfg = cfg or arm.cfg
    cam = cam or WristCamera(arm, cfg)
    scan_cfg = cfg.get("sim", {}).get("scan", {})
    squares = list(scan_cfg.get("squares") or DEFAULT_SQUARES)
    stride = int(scan_cfg.get("stride", 1))
    voxel = float(scan_cfg.get("voxel", 0.003))
    duration = float(scan_cfg.get("move_duration", 0.5))
    settle = int(scan_cfg.get("settle_steps", 8))
    hover = float(scan_cfg.get("hover", 0.24))
    aabb = workspace_aabb(cfg)
    look = board_center(cfg)

    prev_far = cam.far
    cam.far = float(scan_cfg.get("far", 0.75))

    skip = {0, int(arm.robot_id)}
    views: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
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
            pts, cols = unproject(shot, skip_bodies=skip, stride=stride)
            views.append((pts, cols, np.asarray(shot["eye"], dtype=np.float32)))
    finally:
        cam.far = prev_far

    return fuse(views, voxel=voxel, aabb=aabb)


def draw(world: World, point_size: float = 5.0, max_points: int = 20000) -> None:
    """Draw the 3D cloud in the PyBullet window (orbit with the mouse)."""
    for i in world.debug_ids:
        try:
            p.removeUserDebugItem(i)
        except Exception:
            pass
    world.debug_ids.clear()
    if len(world) == 0 or not p.isConnected():
        return
    pts = world.points
    cols = world.colors.astype(np.float32) / 255.0
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts, cols = pts[idx], cols[idx]
    world.debug_ids.append(p.addUserDebugPoints(pts.tolist(), cols.tolist(), pointSize=point_size))


def look_at_board(cfg: dict, distance: float = 0.55, yaw: float = 45, pitch: float = -40) -> None:
    if not p.isConnected():
        return
    c = board_center(cfg)
    p.resetDebugVisualizerCamera(distance, yaw, pitch, [c[0], c[1], c[2] + 0.02])


def write_ply(world: World, path: str | Path | None = None) -> Path:
    path = Path(path) if path else FRAMES / "world_scan.ply"
    path.parent.mkdir(parents=True, exist_ok=True)
    pts, rgb = world.points, world.colors
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for (x, y, z), (r, g, b) in zip(pts, rgb):
            f.write(f"{x:.5f} {y:.5f} {z:.5f} {int(r)} {int(g)} {int(b)}\n")
    return path


def _view_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z = eye - target
    z = z / (np.linalg.norm(z) + 1e-9)
    x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    m = np.eye(4)
    m[0, :3] = x
    m[1, :3] = y
    m[2, :3] = z
    m[0, 3] = -np.dot(x, eye)
    m[1, 3] = -np.dot(y, eye)
    m[2, 3] = -np.dot(z, eye)
    return m


def _project_points(points: np.ndarray, colors: np.ndarray, eye, target, up, size: tuple[int, int], fov_deg: float = 50.0) -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), 18, dtype=np.uint8)
    if len(points) == 0:
        return img
    view = _view_matrix(np.asarray(eye, dtype=np.float64), np.asarray(target, dtype=np.float64), np.asarray(up, dtype=np.float64))
    hom = np.concatenate([points.astype(np.float64), np.ones((len(points), 1))], axis=1)
    cam = (view @ hom.T).T
    z = cam[:, 2]
    keep = z < -0.02
    cam, cols, z = cam[keep], colors[keep], z[keep]
    f = 0.5 * h / np.tan(np.deg2rad(fov_deg) / 2)
    u = (f * cam[:, 0] / -z + w / 2).astype(int)
    v = (-f * cam[:, 1] / -z + h / 2).astype(int)
    order = np.argsort(z)
    u, v, cols = u[order], v[order], cols[order]
    ok = (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1)
    u, v, cols = u[ok], v[ok], cols[ok]
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            img[v + dv, u + du] = cols
    return img


def render(world: World, cfg: dict | None = None, path: str | Path | None = None) -> np.ndarray:
    """Three real 3D orbits of the fused cloud (not a heightmap). Also writes PLY."""
    import cv2

    cfg = cfg or {}
    write_ply(world)
    c = np.array(board_center(cfg) if cfg else [0.37, 0.05, 0.03], dtype=np.float64)
    c[2] = 0.03
    up = np.array([0.0, 0.0, 1.0])
    orbits = [
        ("High", c + np.array([0.12, -0.28, 0.32])),
        ("Side", c + np.array([0.42, 0.05, 0.12])),
        ("Oblique", c + np.array([-0.08, -0.32, 0.18])),
    ]
    panels = []
    for name, eye in orbits:
        img = _project_points(world.points, world.colors, eye, c, up, (640, 400))
        cv2.putText(img, name, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        panels.append(img)
    header = np.full((40, 640, 3), 18, dtype=np.uint8)
    cv2.putText(
        header,
        f"3D wrist-cam cloud   {world.n_views} views   {len(world)} pts   + world_scan.ply",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    canvas = np.vstack([header, *panels])
    path = Path(path) if path else FRAMES / "world_scan.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return canvas
