"""Board scan: wrist-camera sweep → heightmap + brightness → detected pieces.

Scan poses are chosen by where the camera looks, not where the gripper is:
at the high "vertical" IK poses the D405 actually points 27-46 deg below
horizontal. Candidates come from a joint grid; each keeps every robot geom
well above the tallest piece, and a greedy cover picks views until every
square is seen steeply from two of them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

import scene
from mj import probing
from rgbd import unproject
from sensor import D405_FAR, DEFAULT_CAMERA, _mujoco_cv_extrinsics, capture
from squares import FILES, square_to_world

ROBOT_BODIES = ("upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1", "camera_mount")
SCAN_CLEAR_M = 0.07  # robot geoms over the board stay above this (tallest man 55 mm)
VIEW_MIN_M = 0.12  # square center at least this far from the lens (55 mm tops stay past the 70 mm near clip)
VIEW_MAX_M = 0.42
VIEW_MAX_TILT_DEG = 60.0
SCAN_GRIPPER = 0.0  # jaws shut: smallest silhouette in the wrist view
COVER_TARGET = 2
MAX_VIEWS = 10
PIECE_MIN_CELLS = 12
BRIGHT_WHITE = 0.5  # mean top luminance: white men ~0.9, black ~0.2


SCAN_SAMPLES = 30000
SCAN_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TRANSIT_CHECKS = 12


def _samples(sim, n: int = SCAN_SAMPLES, seed: int = 0):
    """Uniform joint samples. Near-base squares are only seen from extreme lift/flex/roll combos a grid misses."""
    rng = np.random.default_rng(seed)
    lim = {j: sim.model.jnt_range[sim._joint[j]] for j in SCAN_JOINTS}
    for _ in range(n):
        yield {j: float(rng.uniform(*lim[j])) for j in SCAN_JOINTS}


def _robot_geoms(sim) -> np.ndarray:
    bids = {sim.body_id(n) for n in ROBOT_BODIES}
    bids.discard(-1)
    return np.array([g for g in range(sim.model.ngeom) if int(sim.model.geom_bodyid[g]) in bids], dtype=int)


_CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)


def _geom_corners(sim, geoms: np.ndarray) -> np.ndarray:
    """(n, 8, 3) world corners of each geom's bounding box."""
    aabb = sim.model.geom_aabb[geoms]
    local = aabb[:, None, :3] + _CORNERS[None] * aabb[:, None, 3:]
    R = sim.data.geom_xmat[geoms].reshape(-1, 3, 3)
    return np.einsum("gij,gkj->gki", R, local) + sim.data.geom_xpos[geoms][:, None, :]


def _board_box(cfg: dict, margin: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    hm = scene.empty_heightmap(cfg, margin=margin)
    lo = hm.origin
    hi = hm.origin + np.array(hm.shape) * hm.cell
    return lo, hi


@probing
def plan_scan_poses(sim, cfg: dict, camera: str = DEFAULT_CAMERA) -> list[dict[str, float]]:
    """Greedy set of arm poses whose wrist views cover every square COVER_TARGET times."""
    surface = scene._board(cfg)[3]
    squares = [f"{f}{r}" for f in FILES for r in range(1, 9)]
    pts = np.array([square_to_world(sq, cfg, z=surface + 0.01) for sq in squares], dtype=float)
    cid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    cam_cfg = (sim.cfg or {}).get("sim", {}).get("camera", {})
    w, h = int(cam_cfg.get("width", 320)), int(cam_cfg.get("height", 240))
    fy = 0.5 * h / math.tan(math.radians(float(sim.model.cam_fovy[cid])) / 2.0)
    geoms = _robot_geoms(sim)
    lo, hi = _board_box(cfg)
    cos_tilt = math.cos(math.radians(VIEW_MAX_TILT_DEG))
    snap = sim.qpos()
    cands: list[tuple[dict, np.ndarray]] = []
    try:
        for q in _samples(sim):
            sim.set_qpos({**snap, **q, "gripper": SCAN_GRIPPER}, hold=False)
            if not _clear(sim, geoms, lo, hi, surface):
                continue
            R, t = _mujoco_cv_extrinsics(sim, camera)
            pc = (pts - t) @ R
            z = pc[:, 2]
            u = fy * pc[:, 0] / np.maximum(z, 1e-6) + (w - 1) / 2
            v = fy * pc[:, 1] / np.maximum(z, 1e-6) + (h - 1) / 2
            ray = (pts - t) / np.linalg.norm(pts - t, axis=1, keepdims=True)
            seen = (
                (z > VIEW_MIN_M) & (z < min(VIEW_MAX_M, D405_FAR))
                & (u > 8) & (u < w - 8) & (v > 8) & (v < h - 8)
                & (-ray[:, 2] > cos_tilt)
            )
            if seen.sum() >= 4:
                cands.append((q, seen))
    finally:
        sim.set_qpos(snap, hold=True)
    count = np.zeros(len(squares), dtype=int)
    chosen: list[dict] = []
    while len(chosen) < MAX_VIEWS and cands:
        need = count < COVER_TARGET
        gains = [int((s & need).sum()) for _, s in cands]
        k = int(np.argmax(gains))
        if gains[k] == 0:
            break
        q, s = cands.pop(k)
        chosen.append(q)
        count += s
    return _tour(sim, cfg, snap, chosen)


def _clear(sim, geoms, lo, hi, surface) -> bool:
    c = _geom_corners(sim, geoms)
    over = (c[..., 0] > lo[0]) & (c[..., 0] < hi[0]) & (c[..., 1] > lo[1]) & (c[..., 1] < hi[1])
    return not (np.any(over & (c[..., 2] < surface + SCAN_CLEAR_M)) or np.any(c[..., 2] < surface + 0.01))


@probing
def transit_clear(sim, cfg: dict, a: dict, b: dict) -> bool:
    """Joint-space line a→b keeps every robot geom above the pieces."""
    geoms = _robot_geoms(sim)
    lo, hi = _board_box(cfg)
    surface = scene._board(cfg)[3]
    snap = sim.qpos()
    try:
        for k in range(1, TRANSIT_CHECKS + 1):
            u = k / TRANSIT_CHECKS
            sim.set_qpos({**snap, **{j: a[j] + (b[j] - a[j]) * u for j in SCAN_JOINTS}}, hold=False)
            if not _clear(sim, geoms, lo, hi, surface):
                return False
        return True
    finally:
        sim.set_qpos(snap, hold=True)


def _tour(sim, cfg: dict, start: dict, poses: list[dict]) -> list[dict]:
    """Nearest-neighbor order from `start`, preferring joint moves that stay clear of the pieces."""
    left = list(poses)
    out: list[dict] = []
    cur = {j: float(start[j]) for j in SCAN_JOINTS}
    while left:
        dist = [sum((p[j] - cur[j]) ** 2 for j in SCAN_JOINTS) for p in left]
        order = np.argsort(dist)
        pick = next((int(k) for k in order if transit_clear(sim, cfg, cur, left[int(k)])), int(order[0]))
        cur = left.pop(pick)
        out.append(cur)
    return out


_POSE_CACHE: dict[str, list[dict]] = {}
POSE_FILE = Path(__file__).resolve().parents[1] / "data" / "scan_poses.json"


def camera_mount_sig(sim) -> list[float]:
    """Camera origin and look in the gripper frame. Changes when the mount moves, so scan poses are replanned."""
    snap = sim.qpos()
    try:
        sim.set_qpos({**snap, "gripper": SCAN_GRIPPER}, hold=False)
        gid = sim.body_id("gripper")
        R = sim.data.xmat[gid].reshape(3, 3)
        origin = sim.data.xpos[gid]
        cid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, DEFAULT_CAMERA)
        off = R.T @ (sim.data.cam_xpos[cid] - origin)
        look = R.T @ (-sim.data.cam_xmat[cid].reshape(3, 3)[:, 2])
        return [round(float(x), 4) for x in (*off, *look)]
    finally:
        sim.set_qpos(snap, hold=True)


def _pose_key(cfg: dict, mount: list[float] | None = None) -> str:
    b = cfg.get("sim", {}).get("board", {})
    parts = [b.get("origin", []), b.get("square_size"), b.get("yaw", 0.0), b.get("surface_z"),
             cfg.get("sim", {}).get("arm_base_position", []), SCAN_CLEAR_M, VIEW_MIN_M, VIEW_MAX_TILT_DEG, SCAN_SAMPLES,
             SCAN_GRIPPER]
    if mount is not None:
        parts.append(mount)
    return json.dumps(parts)


def scan_poses(sim, cfg: dict) -> list[dict[str, float]]:
    """Planned scan poses for this board/arm layout; planned once, then read from data/scan_poses.json."""
    key = _pose_key(cfg, camera_mount_sig(sim))
    if key in _POSE_CACHE:
        return _POSE_CACHE[key]
    disk = json.loads(POSE_FILE.read_text()) if POSE_FILE.exists() else {}
    if key not in disk:
        disk[key] = plan_scan_poses(sim, cfg)
        POSE_FILE.parent.mkdir(parents=True, exist_ok=True)
        POSE_FILE.write_text(json.dumps(disk, indent=1))
    _POSE_CACHE[key] = disk[key]
    return disk[key]


@dataclass
class Scan:
    hm: scene.HeightMap
    bright: np.ndarray  # (nx, ny) mean luminance of piece-height points; nan = none
    views: int = 0
    pieces: list[dict] = field(default_factory=list)


def arm_geoms(sim) -> np.ndarray:
    """Every geom on the arm's kinematic tree (base included)."""
    base = sim.body_id("base")
    geoms = []
    for g in range(sim.model.ngeom):
        b = int(sim.model.geom_bodyid[g])
        while b > 0 and b != base:
            b = int(sim.model.body_parentid[b])
        if b == base:
            geoms.append(g)
    return np.array(geoms, dtype=int)


def robot_pixels(sim, obs, geoms: np.ndarray, camera: str = DEFAULT_CAMERA) -> np.ndarray:
    """(H, W) True where the arm model at its measured pose covers the pixel.

    Renders the controller's own model (for a Robot, the belief copy whose arm
    follows joint readings), never the world.
    """
    h, w = np.asarray(obs.depth).shape
    r = getattr(sim, "_seg_renderer", None)
    if r is None or r.width != w or r.height != h:
        r = mujoco.Renderer(sim.model, height=h, width=w)
        r.enable_segmentation_rendering()
        sim._seg_renderer = r
    r.update_scene(sim.data, camera=camera)
    seg = r.render()
    ids, types = seg[..., 0], seg[..., 1]
    return (types == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(ids, geoms)


def _observe(sim, obs, geoms) -> tuple[np.ndarray, np.ndarray]:
    """World points and their luminance, arm self-returns removed."""
    depth = np.asarray(obs.depth, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > obs.near) & (depth < obs.far)
    pts = unproject(obs)
    lum = (np.asarray(obs.rgb, dtype=np.float64)[valid] / 255.0).mean(axis=1)
    keep = ~robot_pixels(sim, obs, geoms)[valid]
    return pts[keep], lum[keep]


def _fuse_brightness(hm: scene.HeightMap, acc: np.ndarray, cnt: np.ndarray, pts: np.ndarray, lum: np.ndarray) -> None:
    hgt = pts[:, 2] - hm.surface_z
    i = np.floor((pts[:, 0] - hm.origin[0]) / hm.cell).astype(int)
    j = np.floor((pts[:, 1] - hm.origin[1]) / hm.cell).astype(int)
    nx, ny = hm.shape
    m = (hgt > scene.OCCUPIED_H_M) & (hgt < scene.Z_MAX_M) & (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    np.add.at(acc, (i[m], j[m]), lum[m])
    np.add.at(cnt, (i[m], j[m]), 1)


def scan_board(sim, cfg: dict, duration: float = 0.6, physics: bool = True) -> Scan:
    """Visit scan poses, fuse depth and brightness. Caller raises the arm first."""
    from ik import move_qpos

    hm = scene.empty_heightmap(cfg)
    acc = np.zeros(hm.shape)
    cnt = np.zeros(hm.shape, dtype=np.int32)
    views = 0
    geoms = arm_geoms(sim)
    for q in scan_poses(sim, cfg):
        move_qpos(sim, {**q, "gripper": SCAN_GRIPPER}, duration=duration, physics=physics)
        pts, lum = _observe(sim, capture(sim), geoms)
        scene.integrate_points(hm, pts)
        _fuse_brightness(hm, acc, cnt, pts, lum)
        views += 1
    bright = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    out = Scan(hm=hm, bright=bright, views=views)
    out.pieces = detect_pieces(out, cfg)
    return out


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components of a boolean grid → list of (k, 2) index arrays."""
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    nx, ny = mask.shape
    for i0, j0 in zip(*np.nonzero(mask)):
        if seen[i0, j0]:
            continue
        stack = [(i0, j0)]
        seen[i0, j0] = True
        cells = []
        while stack:
            i, j = stack.pop()
            cells.append((i, j))
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < nx and 0 <= b < ny and mask[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    stack.append((a, b))
        comps.append(np.array(cells, dtype=int))
    return comps


MERGE_M = 0.02  # blob centers closer than this are one man split by an occlusion gap (squares are 40 mm apart)


def _merge_fragments(comps: list[np.ndarray], X: np.ndarray, Y: np.ndarray) -> list[np.ndarray]:
    comps = sorted(comps, key=len, reverse=True)
    out: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    for c in comps:
        xy = np.array([X[c[:, 0], c[:, 1]].mean(), Y[c[:, 0], c[:, 1]].mean()])
        near = [k for k, o in enumerate(centers) if np.linalg.norm(o - xy) < MERGE_M]
        if near:
            k = near[0]
            out[k] = np.vstack([out[k], c])
        else:
            out.append(c)
            centers.append(xy)
    return out


TALL_M = 0.030  # only upright men reach this; lying ones stay under their ~24 mm width
UPRIGHT_R_M = 0.016  # widest base radius (12 mm) + margin


def _split_tall(comps: list[np.ndarray], h: np.ndarray, X: np.ndarray, Y: np.ndarray) -> list[np.ndarray]:
    """Blobs touching each other: each tall core keeps its base disk, leftovers become their own blobs."""
    out: list[np.ndarray] = []
    for c in comps:
        tall = h[c[:, 0], c[:, 1]] > TALL_M
        if not tall.any():
            out.append(c)
            continue
        mask = np.zeros(h.shape, dtype=bool)
        mask[c[tall, 0], c[tall, 1]] = True
        cores = _merge_fragments(_components(mask), X, Y)
        centers = np.array([[X[k[:, 0], k[:, 1]].mean(), Y[k[:, 0], k[:, 1]].mean()] for k in cores])
        xy = np.stack([X[c[:, 0], c[:, 1]], Y[c[:, 0], c[:, 1]]], axis=1)
        d = np.linalg.norm(xy[:, None, :] - centers[None, :, :], axis=2)
        near, owner = d.min(axis=1) <= UPRIGHT_R_M, d.argmin(axis=1)
        out.extend(c[near & (owner == k)] for k in range(len(cores)))
        rest = np.zeros(h.shape, dtype=bool)
        rest[c[~near, 0], c[~near, 1]] = True
        out.extend(_components(rest))
    return out


def detect_pieces(scan: Scan, cfg: dict) -> list[dict]:
    """Blobs above the board → xy, square, height, footprint, color."""
    hm = scan.hm
    known = hm.known()
    h = np.where(known, hm.height, 0.0)
    occ = h > scene.OCCUPIED_H_M
    X, Y = hm.centers()
    sq_idx = hm.square_cells(cfg)
    comps = _split_tall(_merge_fragments(_components(occ), X, Y), h, X, Y)
    comps = [c for c in comps if len(c) >= PIECE_MIN_CELLS]
    label = np.full(hm.shape, -1, dtype=int)
    for k, comp in enumerate(comps):
        label[comp[:, 0], comp[:, 1]] = k
    out = []
    for k, comp in enumerate(comps):
        ci, cj = comp[:, 0], comp[:, 1]
        hs = h[ci, cj]
        top = float(hs.max())
        upper = hs > 0.5 * top
        x = float(X[ci[upper], cj[upper]].mean())
        y = float(Y[ci[upper], cj[upper]].mean())
        sq = int(sq_idx[int(round(np.mean(ci))), int(round(np.mean(cj)))])
        b = scan.bright[ci[upper], cj[upper]]
        lum = float(np.nanmean(b)) if np.isfinite(b).any() else float("nan")
        out.append(
            {
                "xy": [x, y],
                "square": None if sq < 0 else f"{FILES[sq // 8]}{sq % 8 + 1}",
                "top_h": top,
                "cells": int(len(comp)),
                "white": bool(lum > BRIGHT_WHITE) if math.isfinite(lum) else None,
                "lum": lum,
                "patch": _patch(hm, x, y, np.where((label >= 0) & (label != k), np.nan, hm.height)),
            }
        )
    return out


PATCH_HALF = 16  # cells: 49 x 49 at 1.5 mm = 73 mm


def _patch(hm: scene.HeightMap, x: float, y: float, height: np.ndarray | None = None) -> np.ndarray:
    """Height patch centered on (x, y); nan where unknown, off-map, or another blob."""
    height = hm.height if height is None else height
    ci = int(np.floor((x - hm.origin[0]) / hm.cell))
    cj = int(np.floor((y - hm.origin[1]) / hm.cell))
    n = 2 * PATCH_HALF + 1
    out = np.full((n, n), np.nan)
    nx, ny = hm.shape
    for a in range(n):
        i = ci - PATCH_HALF + a
        if not 0 <= i < nx:
            continue
        j0, j1 = cj - PATCH_HALF, cj + PATCH_HALF + 1
        lo, hi = max(0, j0), min(ny, j1)
        out[a, lo - j0 : hi - j0] = height[i, lo:hi]
    return out


TEMPLATE_YAWS = 12
_TEMPLATES: dict | None = None


def _isolated_patch(sim, cfg: dict, body: str, xy, yaw: float) -> np.ndarray | None:
    surface = scene._board(cfg)[3]
    half = 0.045
    n = int(round(2 * half / scene.CELL_M))
    hm = scene.HeightMap(
        origin=np.array([xy[0] - half, xy[1] - half]),
        cell=scene.CELL_M,
        surface_z=surface,
        height=np.full((n, n), np.nan),
        hits=np.zeros((n, n), dtype=np.int32),
    )
    sim.place_object(body, [xy[0], xy[1], surface], (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
    truth = scene.truth_heightmap(sim, hm)
    blobs = detect_pieces(Scan(hm=truth, bright=np.full(truth.shape, np.nan)), cfg)
    if not blobs:
        return None
    return max(blobs, key=lambda b: b["cells"])["patch"]


def piece_templates(cfg: dict) -> dict[str, list[np.ndarray]]:
    """Height patches of each piece model alone on the board, at TEMPLATE_YAWS yaws.

    Built from the piece CAD (the scene XML), never from the live world.
    """
    global _TEMPLATES
    if _TEMPLATES is not None:
        return _TEMPLATES
    from mj import Sim
    from pickup import FULL_BOARD_XML
    from scenario import STARTING_AT

    sim = Sim(xml_path=FULL_BOARD_XML, cfg=cfg)
    free = [mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, int(sim.model.jnt_bodyid[j]))
            for j in range(sim.model.njnt) if int(sim.model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
    park = {name: [2.0 + 0.1 * k, 0.0, -1.0] for k, name in enumerate(free)}
    for name, xyz in park.items():
        sim.place_object(name, xyz)
    xy = square_to_world("d4", cfg)
    out: dict[str, list[np.ndarray]] = {}
    for kind in ("pawn", "rook", "knight", "bishop", "queen", "king"):
        body = next(n for n in free if n.startswith(kind + "_"))
        pats = []
        for k in range(TEMPLATE_YAWS):
            p = _isolated_patch(sim, cfg, body, xy, 2 * math.pi * k / TEMPLATE_YAWS)
            if p is not None:
                pats.append(p)
        sim.place_object(body, park[body])
        out[kind] = pats
    sim.close()
    _TEMPLATES = out
    return out


MATCH_SHIFT = 2
MATCH_MIN_H = 0.003


def match(patch: np.ndarray, templates: dict[str, list[np.ndarray]]) -> dict:
    """Best piece kind and yaw for an observed height patch: mean |dh| over seen cells, small shifts allowed."""
    known = np.isfinite(patch)
    P = np.where(known, patch, 0.0)
    scores: dict[str, float] = {}
    yaws: dict[str, float] = {}
    for kind, pats in templates.items():
        best, best_yaw = float("inf"), 0.0
        for y, T in enumerate(pats):
            T0 = np.nan_to_num(T, nan=0.0)
            for di in range(-MATCH_SHIFT, MATCH_SHIFT + 1):
                for dj in range(-MATCH_SHIFT, MATCH_SHIFT + 1):
                    Ts = np.roll(np.roll(T0, di, axis=0), dj, axis=1)
                    m = known & ((P > MATCH_MIN_H) | (Ts > MATCH_MIN_H))
                    if m.sum() < PIECE_MIN_CELLS:
                        continue
                    d = float(np.abs(P[m] - Ts[m]).mean())
                    if d < best:
                        best, best_yaw = d, 2 * math.pi * y / TEMPLATE_YAWS
        scores[kind] = best
        yaws[kind] = best_yaw
    ranked = sorted(scores, key=scores.get)
    kind = ranked[0]
    margin = scores[ranked[1]] - scores[kind] if len(ranked) > 1 else float("inf")
    return {"kind": kind, "score": scores[kind], "margin": margin, "yaw": yaws[kind], "scores": scores,
            "seen": float(known.mean())}


def classify(patch: np.ndarray, templates: dict[str, list[np.ndarray]]) -> tuple[str, float, dict[str, float]]:
    m = match(patch, templates)
    return m["kind"], m["score"], m["scores"]
