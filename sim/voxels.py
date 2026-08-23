"""Board-frame occupancy voxels from a four-corner wrist scan.

Locate the board first (red dots → pose). Then hover the four corners, unproject
RGB-D into that frame, and bin into a 3D occupancy grid. Square occupancy is
just the voxels above the table, grouped 8×8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pybullet as p

from arm import ROOT, SimArm
from board import FILES, reach_square
from board_pose import BoardPose
from camera import WristCamera
from world import board_center, unproject

FRAMES = ROOT / "sim" / "frames"
DEFAULT_SQUARES = ("a1", "a8", "h1", "h8")


@dataclass
class BoardGrid:
    pose: BoardPose
    occ: np.ndarray
    color: np.ndarray
    voxel: float
    zmin: float
    n_views: int = 0
    debug_ids: list[int] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.occ.shape)

    def __len__(self) -> int:
        return int(self.occ.sum())

    def centers_board(self) -> tuple[np.ndarray, np.ndarray]:
        ix, iy, iz = np.where(self.occ)
        if len(ix) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
        xyz = np.stack(
            [
                (ix + 0.5) * self.voxel,
                (iy + 0.5) * self.voxel,
                self.zmin + (iz + 0.5) * self.voxel,
            ],
            axis=1,
        )
        return xyz, self.color[ix, iy, iz]

    def centers_world(self) -> tuple[np.ndarray, np.ndarray]:
        xyz, cols = self.centers_board()
        if len(xyz) == 0:
            return xyz, cols
        return self.pose.board_to_world(xyz), cols

    def occupancy_8x8(self, z_clear: float) -> np.ndarray:
        """True if a square has any voxel whose center is above z_clear (board frame)."""
        nx, ny, nz = self.occ.shape
        iz0 = int(np.floor((z_clear - self.zmin) / self.voxel))
        iz0 = max(0, min(nz, iz0))
        slab = self.occ[:, :, iz0:]
        grid = np.zeros((8, 8), dtype=bool)
        s = self.pose.square_size
        for f in range(8):
            for r in range(8):
                x0 = int(np.floor(f * s / self.voxel))
                x1 = int(np.ceil((f + 1) * s / self.voxel))
                y0 = int(np.floor(r * s / self.voxel))
                y1 = int(np.ceil((r + 1) * s / self.voxel))
                x0, x1 = max(0, x0), min(nx, x1)
                y0, y1 = max(0, y0), min(ny, y1)
                if x1 > x0 and y1 > y0 and slab.size:
                    grid[f, r] = bool(slab[x0:x1, y0:y1].any())
        return grid


def table_z_board(pose: BoardPose, cfg: dict) -> float:
    surface = float(cfg.get("sim", {}).get("board", {}).get("surface_z", 0.02))
    p = pose.origin.copy()
    p[2] = surface
    return float(pose.world_to_board(p.reshape(1, 3))[0, 2])


def voxelize(
    local: np.ndarray,
    colors: np.ndarray,
    voxel: float,
    span: float,
    zmin: float,
    zmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    nx = max(1, int(np.ceil(span / voxel)))
    ny = nx
    nz = max(1, int(np.ceil((zmax - zmin) / voxel)))
    occ = np.zeros((nx, ny, nz), dtype=bool)
    rgb = np.zeros((nx, ny, nz, 3), dtype=np.uint8)
    if len(local) == 0:
        return occ, rgb
    ix = np.floor(local[:, 0] / voxel).astype(np.int32)
    iy = np.floor(local[:, 1] / voxel).astype(np.int32)
    iz = np.floor((local[:, 2] - zmin) / voxel).astype(np.int32)
    keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
    if not np.any(keep):
        return occ, rgb
    ix, iy, iz = ix[keep], iy[keep], iz[keep]
    cols = colors[keep].astype(np.float64)
    lin = ix.astype(np.int64) * (ny * nz) + iy.astype(np.int64) * nz + iz.astype(np.int64)
    uniq, inv, counts = np.unique(lin, return_inverse=True, return_counts=True)
    acc = np.zeros((len(uniq), 3), dtype=np.float64)
    np.add.at(acc, inv, cols)
    mean = (acc / counts[:, None]).astype(np.uint8)
    ux = uniq // (ny * nz)
    uy = (uniq // nz) % ny
    uz = uniq % nz
    occ[ux, uy, uz] = True
    rgb[ux, uy, uz] = mean
    return occ, rgb


def scan(arm: SimArm, cam: WristCamera, cfg: dict, pose: BoardPose) -> BoardGrid:
    """Four-corner wrist RGB-D → occupancy voxels in the measured board frame."""
    vcfg = cfg.get("sim", {}).get("voxels", {})
    scan_cfg = cfg.get("sim", {}).get("scan", {})
    loc = cfg.get("sim", {}).get("locate", {})
    squares = list(vcfg.get("squares") or DEFAULT_SQUARES)
    stride = int(vcfg.get("stride", scan_cfg.get("stride", 1)))
    voxel = float(vcfg.get("size", scan_cfg.get("voxel", 0.004)))
    hover = float(vcfg.get("hover", loc.get("hover", 0.28)))
    duration = float(vcfg.get("move_duration", loc.get("move_duration", 0.5)))
    settle = int(vcfg.get("settle_steps", loc.get("settle_steps", 8)))
    look = board_center(cfg)
    skip = {0, int(arm.robot_id)}
    z_table = table_z_board(pose, cfg)
    zmin = z_table + float(vcfg.get("z_clear", 0.008))
    zmax = z_table + float(vcfg.get("z_max", 0.055))
    span = 8.0 * pose.square_size

    prev_far = cam.far
    cam.far = float(vcfg.get("far", loc.get("far", 0.85)))
    chunks: list[tuple[np.ndarray, np.ndarray]] = []
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
            if len(pts) == 0:
                continue
            chunks.append((pose.world_to_board(pts), cols))
    finally:
        cam.far = prev_far

    if chunks:
        local = np.concatenate([c[0] for c in chunks], axis=0)
        colors = np.concatenate([c[1] for c in chunks], axis=0)
    else:
        local = np.zeros((0, 3))
        colors = np.zeros((0, 3), dtype=np.uint8)
    occ, rgb = voxelize(local, colors, voxel, span, zmin, zmax)
    return BoardGrid(pose=pose, occ=occ, color=rgb, voxel=voxel, zmin=zmin, n_views=len(chunks))


def draw(grid: BoardGrid, point_size: float = 6.0, max_points: int = 18000) -> None:
    for i in grid.debug_ids:
        try:
            p.removeUserDebugItem(i)
        except Exception:
            pass
    grid.debug_ids.clear()
    if not p.isConnected():
        return
    pts, cols = grid.centers_world()
    if len(pts) == 0:
        return
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts, cols = pts[idx], cols[idx]
    grid.debug_ids.append(
        p.addUserDebugPoints(pts.tolist(), (cols.astype(np.float32) / 255.0).tolist(), pointSize=point_size)
    )


def render(grid: BoardGrid, cfg: dict, path: str | Path | None = None) -> np.ndarray:
    """Top-down occupied voxels + 8×8 square occupancy."""
    import cv2

    nx, ny, nz = grid.occ.shape
    top = np.full((ny, nx, 3), 18, dtype=np.uint8)
    if grid.occ.any():
        filled = grid.occ.any(axis=2)
        iz_top = nz - 1 - grid.occ[:, :, ::-1].argmax(axis=2)
        ix, iy = np.where(filled)
        top[iy, ix] = grid.color[ix, iy, iz_top[ix, iy]]
    top = np.flipud(top)  # a1 bottom-left, files → +x
    top = cv2.resize(top, (400, 400), interpolation=cv2.INTER_NEAREST)
    cv2.putText(top, "voxels (a1 bottom-left)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

    z_table = table_z_board(grid.pose, cfg)
    z_clear = z_table + float(cfg.get("sim", {}).get("voxels", {}).get("z_clear", 0.008))
    occ8 = grid.occupancy_8x8(z_clear)
    cell = 44
    pad = 28
    board = np.full((pad + 8 * cell + 20, pad + 8 * cell + 12, 3), 18, dtype=np.uint8)
    for f in range(8):
        for r in range(8):
            x0 = pad + f * cell
            y0 = pad + (7 - r) * cell
            light = (f + r) % 2 == 0
            base = (70, 62, 48) if light else (42, 32, 24)
            if occ8[f, r]:
                base = (46, 196, 88)
            cv2.rectangle(board, (x0, y0), (x0 + cell - 2, y0 + cell - 2), base, -1)
    for i, name in enumerate(FILES):
        cv2.putText(board, name, (pad + i * cell + 14, pad + 8 * cell + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(board, str(i + 1), (8, pad + (7 - i) * cell + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(
        board,
        f"8x8  {int(occ8.sum())}/64 occupied",
        (pad, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    board = cv2.resize(board, (400, 400))

    header = np.full((40, 800, 3), 18, dtype=np.uint8)
    nx, ny, nz = grid.shape
    cv2.putText(
        header,
        f"board voxels   {grid.n_views} views   {len(grid)} occ   {nx}x{ny}x{nz} @ {grid.voxel*1000:.1f}mm",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    canvas = np.vstack([header, np.hstack([top, board])])
    path = Path(path) if path else FRAMES / "board_voxels.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return canvas
