#!/usr/bin/env python3
"""Pick and place: queen → lift → d4 → release.

  python sim/pickup.py --headless
  python sim/pickup.py --headless                # wall real: wrist look
  python sim/pickup.py --headless --wall cheat   # true poses, no look (the perception-free control)
  python sim/pickup.py --headless --random --seed 0 --video
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

import mujoco

import diag
import evaluate_move
import stages

from ik import (
    EE_SITE,
    cart_duration,
    gripper_tilt_deg,
    joint_limit_margin,
    move_ik,
    move_lin,
    move_qpos,
    reachable_approaches,
    solve_ik,
    solve_ik_aim,
    solve_ik_down,
)
from mj import ARM_JOINTS, DEBUG_DIR, Sim, load_config, probing
from clearance import (
    APPROACH_BODIES,
    CAMERA_BODIES,
    _WORKSPACE_R_M,
    _arm_base_xy,
    board_top_z,
    closest_pair,
    neighbor_approach_gap,
    obstacle_bodies,
)
from perception import estimate_target_pose, observe_piece
from piece_fit import fit_upright_xy, piece_shape
from sensor import capture as capture_wrist, robot_blocks_view, wrist_sees_point
from stages.types import Move, Retry, Stop
from squares import bowl_xyz, jitter_near_square, nest_xyz, square_to_world
from metrics import annotate_episode
from scenario import PIECE_BODIES, PIECES, STARTING_AT, body_kind
from occupancy import square_of_body
from robot import GRIP_SETTLE_STEPS, GRIP_STILL_RADS, GRIP_STILL_READS, WALLS, jaw_squeezing, open_robot, world_of

ARM_BODIES = ("base", "shoulder", "upper_arm", "lower_arm")
FULL_BOARD_XML = ROOT / "assets/so101/chess_scene_full.xml"
PIECES_XML = ROOT / "assets/so101/chess_scene_pieces.xml"


def _tid(sim: Sim) -> str:
    return getattr(sim, "target_id", "queen")


def _camera_involved(other: str | None) -> bool:
    name = other or ""
    return name in CAMERA_BODIES or name.startswith("d405") or "camera" in name


def robot_part(other: str | None) -> str | None:
    """Map a contact body to jaw / palm / wrist / camera / arm, or None if not robot."""
    name = other or ""
    if name == "moving_jaw_so101_v1" or "jaw" in name:
        return "jaw"
    if name == "gripper":
        return "palm"
    if name == "wrist" or name.startswith("wrist"):
        return "wrist"
    if _camera_involved(name):
        return "camera"
    if name in ARM_BODIES:
        return "arm"
    return None


def robot_part_walk(sim: Sim, other: str | None) -> str | None:
    part = robot_part(other)
    if part or not other:
        return part
    bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, other)
    while bid > 0:
        bid = int(sim.model.body_parentid[bid])
        parent = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, bid)
        part = robot_part(parent)
        if part:
            return part
    return None


def perceived_xy(sim: Sim, xyz) -> list[float]:
    """Target XY handed from observe to grasp planning. Study suites wrap this to inject perception error."""
    return [float(xyz[0]), float(xyz[1])]


def yaw_quat(deg: float) -> tuple[float, float, float, float]:
    a = np.radians(float(deg))
    return (float(np.cos(a / 2.0)), 0.0, 0.0, float(np.sin(a / 2.0)))


class VideoRecorder:
    """Capture overview+side composites on physics steps, then encode H.264."""

    def __init__(self, sim: Sim, path: Path, every: int = 6, width: int = 480, height: int = 360):
        self.sim = sim
        self.path = Path(path)
        self.every = every
        self.width = width
        self.height = height
        self.n_step = 0
        self.frames: list[np.ndarray] = []

    def grab(self):
        self.n_step += 1
        if self.n_step % self.every:
            return
        ov = self.sim.render("debug_overview", self.width, self.height)
        side = self.sim.render("debug_side", self.width, self.height)
        self.frames.append(np.hstack([ov, side]))

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.frames:
            return self.path
        h, w = self.frames[0].shape[:2]
        proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                "-r", "24", "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(self.path),
            ],
            stdin=subprocess.PIPE,
        )
        assert proc.stdin is not None
        for frame in self.frames:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")
        return self.path


def pickup_cfg(cfg: dict) -> dict:
    return cfg.get("sim", {}).get("pickup", {})


def log_state(sim: Sim, name: str, extra: dict | None = None):
    diag.emit("phase", sim, name, extra=extra)
    sim._pickup_phase = name
    if getattr(sim, "quiet", False):
        return
    tid = _tid(sim)
    W = world_of(sim)
    qpos, qquat = W.object_pose(tid)
    gpos, grot = sim.site_pose(EE_SITE)
    gap = sim.jaw_gap()
    contacts = W.contacts_with(tid)
    print(f"\nSTATE: {name}")
    print(f"  {tid}_ground_truth_xyz: {np.round(qpos, 4).tolist()}")
    print(f"  {tid}_quat:             {np.round(qquat, 4).tolist()}")
    print(f"  gripperframe_xyz:       {np.round(gpos, 4).tolist()}")
    print(f"  gripperframe_x_axis:    {np.round(grot[:, 0], 3).tolist()}")
    print(f"  jaw_gap_mm:             {gap * 1000:.1f}")
    print(f"  {tid}_gripper_dist_mm:  {np.linalg.norm(qpos - gpos) * 1000:.1f}")
    print(f"  {tid}_contacts:         {len(contacts)}")
    for c in contacts[:6]:
        print(f"    {c['other']:20s} dist={c['dist']*1000:.2f}mm  F={c['force']:.3f}")
    if extra:
        for k, v in extra.items():
            print(f"  {k}: {v}")
    run = DEBUG_DIR / "pickup"
    sim.save_frame("debug_overview", run / f"{name}_overview.png")
    sim.save_frame("debug_side", run / f"{name}_side.png")
    sim.save_frame("wrist_cam", run / f"{name}_wrist.png", width=320, height=240)


def _vertical_seed(cfg: dict) -> dict:
    return cfg.get("poses", {}).get(
        "vertical_seed",
        {
            "shoulder_pan": 0.0,
            "shoulder_lift": -0.877,
            "elbow_flex": 0.642,
            "wrist_flex": 1.30,
            "wrist_roll": 1.58,
            "gripper": 0.60,
        },
    )


_CLEAR_GAP = 0.004


def seed_vertical(sim: Sim, cfg: dict):
    """Return to the vertical seed under physics: rise over the pieces, transit, fold at the seed."""
    seed = _vertical_seed(cfg)
    cur = sim.qpos()
    err = max(abs(cur.get(k, 0.0) - float(seed.get(k, 0.0))) for k in seed)
    if err < 0.03:
        sim.hold(seed)
        return
    _clear_to_seed(sim, cfg, seed)


@probing
def _seed_ee_xyz(sim: Sim, cfg: dict):
    snap = sim.qpos()
    try:
        sim.set_qpos(_vertical_seed(cfg), hold=True)
        pos, _ = sim.site_pose(EE_SITE)
        return pos
    finally:
        sim.set_qpos(snap, hold=True)


def _transit_z(sim: Sim, cfg: dict, skip=None) -> float:
    p = pickup_cfg(cfg)
    surface = float(cfg.get("sim", {}).get("board", {}).get("surface_z", 0.02))
    gpos, _ = sim.site_pose(EE_SITE)
    obstacles = obstacle_bodies(sim, skip=skip)
    top = board_top_z(sim, obstacles)
    return max(
        float(gpos[2]),
        top + 0.04,
        surface + float(p.get("pregrasp_z", 0.10)),
        surface + float(p.get("transport_clearance_z", 0.08)),
    )


@probing
def _chain_ik_ok(
    sim: Sim,
    wps,
    lock_joints: tuple[str, ...],
    obstacles=None,
    skip=None,
    margin: float = _CLEAR_GAP,
    ori_weight: float = 0.0,
) -> bool:
    """True if every waypoint solves from the previous and clears board pieces."""
    snap = sim.qpos()
    skip = set(skip or ())
    try:
        for wp in wps:
            if not solve_ik(sim, wp, lock_joints=lock_joints, ori_weight=ori_weight):
                return False
            if obstacles:
                pieces = [n for n in obstacles if n not in skip and n != "board"]
                if pieces and neighbor_approach_gap(sim, pieces) < margin:
                    return False
        return True
    finally:
        sim.set_qpos(snap, hold=True)


def _bodies_under(sim: Sim, xy, skip: set[str], radius: float = 0.03) -> set[str]:
    """Bodies already in this gripper column. Going +Z is the exit, not a sideswipe."""
    under = set(skip)
    for name in obstacle_bodies(sim, skip=skip):
        if name == "board":
            continue
        bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue
        p = sim.data.xpos[bid]
        if float(np.hypot(float(p[0]) - float(xy[0]), float(p[1]) - float(xy[1]))) < radius:
            under.add(name)
    return under


def _safe_lin(
    sim: Sim,
    goal,
    cfg: dict,
    lock: tuple[str, ...],
    obstacles=None,
    skip=None,
    duration: float = 0.9,
) -> bool:
    """Cartesian move that lifts over pieces, transits, then descends. No low XY sweep."""
    goal = np.asarray(goal, dtype=float)
    skip = set(skip or ())
    if obstacles is None:
        obstacles = obstacle_bodies(sim, skip=skip)
    start, _ = sim.site_pose(EE_SITE)
    z_safe = _transit_z(sim, cfg, skip=skip)
    xy = float(np.hypot(goal[0] - start[0], goal[1] - start[1]))
    under = _bodies_under(sim, start[:2], skip)
    wrist = ("wrist_flex", "wrist_roll")

    segs: list[tuple[list[float], set[str], bool]] = []
    if xy > 0.012 and float(start[2]) < z_safe - 0.008:
        segs.append(([float(start[0]), float(start[1]), z_safe], under, True))
    if xy > 0.012:
        cruise_z = max(float(goal[2]), z_safe)
        segs.append(([float(goal[0]), float(goal[1]), cruise_z], skip, True))
    going_up = xy <= 0.012 and float(goal[2]) > float(start[2]) + 0.008
    segs.append(
        ([float(goal[0]), float(goal[1]), float(goal[2])], under if going_up else skip, going_up)
    )
    cleaned: list[tuple[list[float], set[str], bool]] = []
    for seg in segs:
        if cleaned and float(np.linalg.norm(np.asarray(seg[0]) - cleaned[-1][0])) < 0.008:
            continue
        cleaned.append(seg)

    def _run(pt, sk, lock_wrist, dur) -> bool:
        use_lock = tuple(dict.fromkeys((*lock, *wrist))) if lock_wrist else lock
        cur, _ = sim.site_pose(EE_SITE)
        dest = [float(pt[0]), float(pt[1]), float(pt[2])]
        dist = float(np.linalg.norm(np.asarray(dest) - cur))
        n_wp = max(6, int(np.ceil(max(dist, 0.03) / 0.03)))
        for extra_z in (0.0, 0.03, 0.06):
            g = [dest[0], dest[1], dest[2] + extra_z]
            # Floor long XY cruises so they do not ask for a dead hover.
            # A +Z rise over a just-placed man must stay on that XY — flooring
            # it walks out to the 190 mm ring and combs the piece (b3).
            xy = float(np.hypot(g[0] - cur[0], g[1] - cur[1]))
            raw = _line_wps(cur, g, n_wp)
            wps = [_floor_wp(wp, cfg) for wp in raw] if xy > 0.02 else raw
            if not _chain_ik_ok(sim, wps, use_lock, obstacles=obstacles, skip=sk):
                continue
            if _follow_wps(sim, wps, dur, use_lock):
                return True
        cur, _ = sim.site_pose(EE_SITE)
        if float(np.hypot(dest[0] - cur[0], dest[1] - cur[1])) < 0.015:
            return move_lin(sim, dest, duration=dur, physics=True, n_wp=n_wp, lock_joints=use_lock)
        return False

    for i, (pt, sk, lock_wrist) in enumerate(cleaned):
        cur, _ = sim.site_pose(EE_SITE)
        dt = cart_duration(sim, pt, holding=False, ref=pt)
        if _run(pt, sk, lock_wrist, dt):
            continue
        if i == 0 and lock_wrist:
            out = _outward_xy(start[:2], cfg, 0.04)
            if _run([float(out[0]), float(out[1]), float(pt[2])], sk, True, dt):
                continue
        return False
    return True


def _clear_to_seed(sim: Sim, cfg: dict, seed: dict):
    """Lift over the pieces, transit at that height, then fold to the seed."""
    lock = ("wrist_flex", "wrist_roll")
    obstacles = obstacle_bodies(sim)
    z_safe = _transit_z(sim, cfg)
    gpos, _ = sim.site_pose(EE_SITE)
    if float(gpos[2]) < z_safe - 0.008:
        high = [float(gpos[0]), float(gpos[1]), z_safe]
        if not _safe_lin(sim, high, cfg, lock, obstacles=obstacles):
            out = _outward_xy(gpos[:2], cfg, 0.04)
            _safe_lin(sim, [float(out[0]), float(out[1]), z_safe], cfg, lock, obstacles=obstacles)
        gpos, _ = sim.site_pose(EE_SITE)
        z_safe = max(z_safe, float(gpos[2]))
    seed_ee = _seed_ee_xyz(sim, cfg)
    park = [float(seed_ee[0]), float(seed_ee[1]), z_safe]
    _safe_lin(sim, park, cfg, lock, obstacles=obstacles, duration=0.55)
    # High transit is over the pieces; the last fold is local at the seed XY
    # so the gripper points down again without sweeping rank 1.
    move_qpos(sim, seed, duration=0.4, physics=True)
    sim.hold(seed)


def _reset_ik_seed(sim: Sim, cfg: dict, keep: dict | None = None):
    seed = dict(_vertical_seed(cfg))
    if keep:
        seed.update(keep)
    sim.set_qpos(seed, hold=True)


@probing
def _vertical_wrist(sim: Sim, target) -> dict | None:
    """wrist_flex that makes gripper +X world-down at target (roll untouched).

    Best effort: returns the least-tilted reachable pose even when a true
    vertical is outside the workspace, or None if the point is unreachable.
    Restores qpos.
    """
    snap = sim.qpos()
    try:
        solve_ik_down(sim, target, lock_joints=("wrist_roll",))
        pos, _ = sim.site_pose(EE_SITE)
        if float(np.linalg.norm(pos - np.asarray(target, dtype=float))) > 0.0075:
            return None
        return {"wrist_flex": float(sim.qpos()["wrist_flex"]), "tilt_deg": gripper_tilt_deg(sim)}
    finally:
        sim.set_qpos(snap, hold=True)


@probing
def _pitch_path(sim: Sim, flex_goal: float, n: int) -> list[dict]:
    """Arm waypoints that pitch wrist_flex to `flex_goal` with the EE held put (planned, not executed)."""
    keep, _ = sim.site_pose(EE_SITE)
    snap = sim.qpos()
    f0 = float(snap["wrist_flex"])
    path: list[dict] = []
    try:
        for i in range(1, n + 1):
            q = sim.qpos()
            q["wrist_flex"] = f0 + (flex_goal - f0) * i / n
            sim.set_qpos(q, hold=False)
            solve_ik(sim, keep, lock_joints=("wrist_flex", "wrist_roll"))
            path.append({j: float(sim.qpos()[j]) for j in ARM_JOINTS})
    finally:
        sim.set_qpos(snap, hold=True)
    return path


def _drive_arm(sim: Sim, path: list[dict], gripper: float, duration: float, until=None) -> bool:
    """Servo the arm through joint waypoints under physics, gripper target fixed. False if `until()` fails."""
    steps = max(8, int(round(duration / max(sim.dt, 1e-6) / max(1, len(path)))))
    for q in path:
        start = {j: float(sim.qpos()[j]) for j in ARM_JOINTS}
        for k in range(1, steps + 1):
            u = k / steps
            u = u * u * (3 - 2 * u)
            sim.set_ctrl({**{j: start[j] + (q[j] - start[j]) * u for j in ARM_JOINTS}, "gripper": float(gripper)})
            sim.step()
        if until is not None and not until():
            return False
    sim.set_ctrl({**path[-1], "gripper": float(gripper)} if path else {"gripper": float(gripper)})
    sim.step(10)
    return True


_PAD_RADIUS_M = 0.00075
_PINCH_CLEAR_MIN_M = 0.002


def _pinch_shift_m(sim: Sim, geom: dict | None) -> float:
    """EE shift along gripper local +Z (the pinch axis) that parks the fixed
    jaw pad just outside the piece so the moving jaw does the closing.

    The SO-101 jaw is asymmetric: the fixed pad sits ~20 mm from the
    gripperframe site along local -Z and the moving pad sweeps toward it.
    Centering the piece between open jaws shoves it half the aperture on
    close; biasing to the fixed pad shoves it only the clearance. Clearance
    covers the piece's widest overhang above the stem (head/collar) so the pad
    descends past it, plus a margin for IK residual.
    """
    cache = sim.__dict__.get("_fixed_pad_local_z")
    if cache is None:
        ee, R = sim.site_pose(EE_SITE)
        fl = R.T @ (sim.geom_xpos("fixed_jaw_sph_tip1") - ee)
        cache = float(fl[2])
        sim._fixed_pad_local_z = cache
    g = geom or {}
    stem_r = 0.5 * float(g.get("stem_width", 0.011))
    overhang = max(0.0, 0.5 * float(g.get("width", 2.0 * stem_r)) - stem_r)
    clear = max(_PINCH_CLEAR_MIN_M, overhang + 0.002)
    return abs(cache) - _PAD_RADIUS_M - stem_r - clear


@probing
def _grasp_xy_for_roll(sim: Sim, piece_xy, z: float, geom: dict | None, roll: float | None) -> list[float]:
    """Grasp EE XY for a vertical pinch at the given wrist roll (None = current).

    Solves the vertical pose over the piece so the pinch axis (gripper local
    Z) is known in world, then offsets along it by the fixed-pad shift.
    Restores qpos.
    """
    snap = sim.qpos()
    try:
        if roll is not None:
            q = dict(snap)
            q["wrist_roll"] = float(roll)
            sim.set_qpos(q, hold=True)
        seed = [float(piece_xy[0]), float(piece_xy[1]), float(z)]
        solve_ik_down(sim, seed, lock_joints=("wrist_roll",))
        _, R = sim.site_pose(EE_SITE)
        d = _pinch_shift_m(sim, geom)
        return [float(piece_xy[0] + R[0, 2] * d), float(piece_xy[1] + R[1, 2] * d)]
    finally:
        sim.set_qpos(snap, hold=True)


def _outward_xy(xy, cfg: dict, extra_m: float) -> np.ndarray:
    """Shift XY further from the arm base along the base→target ray."""
    base = _arm_base_xy(cfg)
    p = np.asarray(xy[:2], dtype=float)
    v = p - base
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        v, n = np.array([1.0, 0.0]), 1.0
    return p + v * (extra_m / n)


def _xy_floor(xy, cfg: dict, rmin: float = _WORKSPACE_R_M) -> tuple[np.ndarray, bool]:
    """Push XY out to `rmin` from the arm base if it sits inside that radius."""
    base = _arm_base_xy(cfg)
    p = np.asarray(xy[:2], dtype=float)
    v = p - base
    n = float(np.linalg.norm(v))
    if n >= rmin:
        return p, False
    if n < 1e-9:
        v, n = np.array([1.0, 0.0]), 1.0
    return base + v / n * rmin, True


def _floor_wp(wp, cfg: dict, rmin: float = _WORKSPACE_R_M) -> list[float]:
    xy, _ = _xy_floor(wp[:2], cfg, rmin)
    return [float(xy[0]), float(xy[1]), float(wp[2])]


def _line_wps(a, b, n: int) -> list[list[float]]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = max(2, int(n))
    return [[float(v) for v in a + (b - a) * ((i + 1) / n)] for i in range(n)]


def _follow_wps(
    sim: Sim, wps, duration: float, lock_joints: tuple[str, ...], ori_weight: float = 0.0
) -> bool:
    n = max(1, len(wps))
    dt = max(0.08, duration / n)
    for wp in wps:
        if not move_ik(
            sim, wp, duration=dt, physics=True, lock_joints=lock_joints, ori_weight=ori_weight
        ):
            return False
    return True


def settle_queen(sim: Sim, steps: int = 24):
    sim.step(max(1, int(steps)))


def queen_tilt_deg(quat) -> float:
    _, x, y, _ = quat
    zz = 1 - 2 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def _robot_bid_array(sim: Sim) -> np.ndarray:
    cache = sim.__dict__.get("_robot_bid_array")
    if cache is None:
        from clearance import _robot_body_ids

        ids = {int(i) for i in _robot_body_ids(sim)}
        ids.discard(0)  # world is a chain root, not a robot part
        cache = np.fromiter(sorted(ids), dtype=int)
        sim._robot_bid_array = cache
    return cache


def bystander_contact(sim: Sim, bodies: list[str]) -> dict | None:
    """First active contact between any robot link and one of `bodies`.

    Vectorized over the contact array so it is cheap enough to run every
    physics step as a motion guard.
    """
    w = world_of(sim) if getattr(sim, "cheat_state", True) else sim
    b1, b2 = w.contact_bodies()
    if b1.size == 0:
        return None
    by = np.fromiter((sim.body_id(n) for n in bodies), dtype=int)
    robot = _robot_bid_array(sim)
    hit = (np.isin(b1, robot) & np.isin(b2, by)) | (np.isin(b2, robot) & np.isin(b1, by))
    if not hit.any():
        return None
    k = int(np.flatnonzero(hit)[0])
    rb, pb = (b1[k], b2[k]) if b1[k] in robot else (b2[k], b1[k])
    return {
        "body": sim.body_name(int(pb)),
        "robot": sim.body_name(int(rb)),
        "dist_mm": float(w.data.contact.dist[k] * 1000.0),
    }


def _jaw_geom_ids(sim: Sim) -> list[int]:
    ids = sim.__dict__.get("_jaw_geom_ids")
    if ids is None:
        ids = []
        for i in range(sim.model.ngeom):
            ni = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if ni.startswith("fixed_jaw") or ni.startswith("moving_jaw"):
                ids.append(i)
        sim._jaw_geom_ids = ids
    return ids


def _body_geom_ids(sim: Sim, body: str) -> list[int]:
    cache = sim.__dict__.setdefault("_body_geom_ids", {})
    ids = cache.get(body)
    if ids is None:
        bid = sim.body_id(body)
        ids = [int(j) for j in np.flatnonzero(sim.model.geom_bodyid == bid)]
        cache[body] = ids
    return ids


def min_jaw_gap(sim: Sim, object_id: str | None = None) -> float:
    tid = object_id or _tid(sim)
    fromto = np.zeros(6)
    best = 1.0
    for i in _jaw_geom_ids(sim):
        for j in _body_geom_ids(sim, tid):
            d = mujoco.mj_geomDistance(sim.model, sim.data, i, j, 0.08, fromto)
            if d < best:
                best = d
    return float(best)


def min_body_gap(sim: Sim, body_a: str, body_b: str) -> float:
    fromto = np.zeros(6)
    best = 1.0
    for i in _body_geom_ids(sim, body_a):
        for j in _body_geom_ids(sim, body_b):
            d = mujoco.mj_geomDistance(sim.model, sim.data, i, j, 0.12, fromto)
            if d < best:
                best = d
    return float(best)


def close_gripper(sim: Sim, closed: float, steps: int = 200):
    q = sim.qpos()
    start = q["gripper"]
    for i in range(steps):
        u = (i + 1) / steps
        q["gripper"] = start + (closed - start) * u
        sim.set_ctrl(q)
        sim.step()
    sim.hold({**sim.qpos(), "gripper": closed})
    sim.step(30)
    if closed >= start:
        return
    # A jaw closing on a piece creeps for a while before it stalls; reading
    # the grasp before then is a coin flip on the speed threshold.
    still = 0
    for _ in range(GRIP_SETTLE_STEPS):
        still = still + 1 if abs(float(sim.qvel()["gripper"])) < GRIP_STILL_RADS else 0
        if still >= GRIP_STILL_READS:
            break
        sim.step()


def open_gripper(sim: Sim, opened: float, steps: int = 80):
    close_gripper(sim, opened, steps=steps)


_PAD_CLEAR_M = 0.01


def _fixed_pad_clear(sim: Sim, piece_xy, dist: float = _PAD_CLEAR_M):
    """EE XY that pulls the stationary pad `dist` off the piece, same height."""
    fix = np.asarray(sim.geom_xpos("fixed_jaw_sph_tip1")[:2], dtype=float)
    p = np.asarray(piece_xy[:2], dtype=float)
    v = fix - p
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        mov = np.asarray(sim.geom_xpos("moving_jaw_sph_tip1")[:2], dtype=float)
        v = fix - mov
        n = float(np.linalg.norm(v))
    if n < 1e-6:
        return None
    delta = v / n * float(dist)
    gpos, _ = sim.site_pose(EE_SITE)
    return [float(gpos[0] + delta[0]), float(gpos[1] + delta[1]), float(gpos[2])]


_RETREAT_GAP_M = 0.003


def _hand_clear_of(sim: Sim, tid: str, margin: float = _RETREAT_GAP_M) -> bool:
    """True when the gripper envelope is off this piece (not just 'not grasping')."""
    if grasp_acquired(sim, tid) or bystander_contact(sim, [tid]):
        return False
    return float(neighbor_approach_gap(sim, [tid])) >= float(margin)


def _clear_then_rise(sim: Sim, cfg: dict, tid: str, z_min: float = 0.11) -> dict:
    """Unhook the pads, then rise. Abort +Z if the hand hits the piece.

    Approach already guards bystanders on the way down. After release we used
    to skip this piece (`short_retreat` / `_safe_lin(skip=tid)`) and comb the
    lip on the way up.
    """
    lock = ("wrist_flex", "wrist_roll")
    out = {"cleared": False, "pad_clear_mm": 0.0, "rose": False, "aborted": None}
    p0, q0 = sim.object_pose(tid)
    tilt0 = float(queen_tilt_deg(q0))
    travelled = 0.0
    while travelled < 0.016 and not _hand_clear_of(sim, tid):
        q, _ = sim.object_pose(tid)
        slide = _fixed_pad_clear(sim, q, dist=0.004)
        if slide is None:
            break
        if not move_lin(sim, slide, duration=0.28, physics=True, n_wp=3, lock_joints=lock):
            break
        settle_queen(sim, steps=8)
        travelled += 0.004
        p1, q1 = sim.object_pose(tid)
        if float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) > 0.005:
            break
        if float(queen_tilt_deg(q1)) > tilt0 + 10.0:
            break
    out["pad_clear_mm"] = round(travelled * 1000.0, 1)
    out["cleared"] = _hand_clear_of(sim, tid)
    if not out["cleared"]:
        gpos, _ = sim.site_pose(EE_SITE)
        away = _outward_xy(gpos[:2], cfg, 0.012)
        move_lin(
            sim,
            [float(away[0]), float(away[1]), float(gpos[2])],
            duration=0.28,
            physics=True,
            n_wp=3,
            lock_joints=lock,
        )
        settle_queen(sim, steps=8)
        out["cleared"] = _hand_clear_of(sim, tid)
    gpos, _ = sim.site_pose(EE_SITE)
    z_up = max(float(gpos[2]) + 0.05, float(z_min), _transit_z(sim, cfg, skip={tid}))
    if float(gpos[2]) >= z_up - 0.01:
        out["rose"] = True
        return out
    if not out["cleared"]:
        # Still in the lip. A +Z here is the comb the viewer saw.
        return out
    high = [float(gpos[0]), float(gpos[1]), z_up]
    sim._last_abort = None
    ok = move_lin(
        sim,
        high,
        duration=0.7,
        physics=True,
        n_wp=6,
        lock_joints=lock,
        guard=lambda: bystander_contact(sim, [tid]),
    )
    aborted = getattr(sim, "_last_abort", None)
    if aborted:
        out["aborted"] = aborted
        gpos, _ = sim.site_pose(EE_SITE)
        q, _ = sim.object_pose(tid)
        slide = _fixed_pad_clear(sim, q, dist=0.008)
        if slide is not None:
            move_lin(
                sim,
                [float(slide[0]), float(slide[1]), float(gpos[2])],
                duration=0.3,
                physics=True,
                n_wp=3,
                lock_joints=lock,
            )
        if _hand_clear_of(sim, tid):
            sim._last_abort = None
            gpos, _ = sim.site_pose(EE_SITE)
            high = [float(gpos[0]), float(gpos[1]), z_up]
            ok = move_lin(
                sim,
                high,
                duration=0.7,
                physics=True,
                n_wp=6,
                lock_joints=lock,
                guard=lambda: bystander_contact(sim, [tid]),
            )
            aborted = getattr(sim, "_last_abort", None)
    out["rose"] = bool(ok) and not aborted
    return out


def queen_upright(quat, tilt_max: float = 0.35) -> bool:
    """quat is wxyz; world +Z of the piece ≈ body +Z."""
    w, x, y, z = quat
    # body Z in world from quaternion
    zz = 1 - 2 * (x * x + y * y)
    return float(zz) > np.cos(tilt_max)


def _piece_still_upright(sim: Sim, tid: str, surface: float, tilt_max: float = 0.5) -> bool:
    qpos, quat = sim.object_pose(tid)
    if float(qpos[2]) < surface - 0.01:
        return False
    return queen_upright(quat, tilt_max=tilt_max)


def _backoff_grasp(sim: Sim, cfg: dict, tid: str, release_open: float) -> None:
    """Open just enough to free the stem, then lift so the next approach is clear."""
    open_gripper(sim, float(release_open), steps=80)
    settle_queen(sim, steps=24)
    gpos, _ = sim.site_pose(EE_SITE)
    z_up = max(
        float(gpos[2]) + 0.05,
        float(pickup_cfg(cfg).get("pregrasp_z", 0.10)) + float(
            cfg.get("sim", {}).get("board", {}).get("surface_z", 0.02)
        ),
    )
    high = [float(gpos[0]), float(gpos[1]), z_up]
    _safe_lin(
        sim,
        high,
        cfg,
        ("wrist_flex", "wrist_roll"),
        obstacles=obstacle_bodies(sim, skip={tid}),
        skip={tid},
        duration=0.6,
    )
    settle_queen(sim, steps=16)


def _reperceive(sim: Sim, cfg: dict, tid: str, near: str | None) -> str | None:
    """After a drop the in-hand belief is stale: look where it fell, scan the board if it is not there.

    Returns the square the piece was seen on, or None if it is off the board or not found.
    """
    near = square_of_body(sim, cfg, tid) or near
    if near and verify_piece(sim, cfg, tid, near).get("seen"):
        return square_of_body(sim, cfg, tid)
    stages.resolve(cfg)["board_observe"].scan(sim, cfg)
    return square_of_body(sim, cfg, tid) if tid in sim.known else None


def _grasp_retry(sim: Sim, cfg: dict, sc: dict, failed: dict, tid: str, surface: float, release_open: float) -> Retry | None:
    """Pick again from the current pose after a miss/drop, if the piece is still up."""
    attempt = int(sc.get("_grasp_attempt", 0))
    budget = int(pickup_cfg(cfg).get("grasp_retries", 1))
    if attempt >= budget:
        return None
    here = sc.get("source")
    if _perceiving(sim):
        _backoff_grasp(sim, cfg, tid, release_open)
        here = _reperceive(sim, cfg, tid, here)
        if here is None:
            return None
    elif not _piece_still_upright(sim, tid, surface, tilt_max=1.0):
        return None
    else:
        _backoff_grasp(sim, cfg, tid, release_open)
    if not _piece_still_upright(sim, tid, surface, tilt_max=0.5):
        return None
    log_state(
        sim,
        "retry_grasp",
        extra={"attempt": attempt + 2, "after": failed.get("failure_reason")},
    )
    nxt = dict(sc, source=here)
    nxt["seed_home"] = False
    nxt["reset_board"] = False
    nxt["_grasp_attempt"] = attempt + 1
    log = list(failed.get("grasp_retry_log") or [])
    log.append({"attempt": attempt + 1, "reason": failed.get("failure_reason")})
    return Retry(nxt, {"grasp_retry": True, "grasp_attempts": attempt + 2, "first_grasp_reason": failed.get("failure_reason")}, log)


def _place_correction(sim: Sim, cfg: dict, sc: dict, failed: dict, tid: str, surface: float) -> Retry | None:
    """Re-grasp a standing piece where the confirm look saw it and nudge it onto dest.

    The piece can shift in the jaws during the carry, so the in-hand belief
    drifts; the post-release look measures the real offset and closes the loop.
    The correction is a hop over its own square (carry `nudge`), not a new carry.
    """
    done = int(sc.get("_place_correction", 0))
    if done >= int(pickup_cfg(cfg).get("place_corrections", 2)):
        return None
    if _perceiving(sim) and tid not in sim.known:
        return None
    if not _piece_still_upright(sim, tid, surface, tilt_max=0.5):
        return None
    here = square_of_body(sim, cfg, tid)
    if here is None:
        return None
    off_mm = failed.get("placed_off_mm")
    log_state(sim, "retry_place", extra={"attempt": done + 1, "from": here, "off_mm": off_mm})
    nxt = dict(sc, source=here, seed_home=False, reset_board=False, _grasp_attempt=0, _place_correction=done + 1, _nudge=True)
    nxt.setdefault("_first_place_off_mm", off_mm)
    return Retry(nxt, {"place_corrections": done + 1, "first_place_off_mm": nxt["_first_place_off_mm"]})


def _move_q_hold(sim: Sim, goal_arm: dict, closed: float, duration: float = 0.6) -> None:
    """Joint-space move that keeps commanding the goal pose (not the sagged qpos)."""
    start = {n: sim.qpos()[n] for n in ARM_JOINTS}
    n = max(24, int(float(duration) / sim.dt))
    for i in range(n):
        u = (i + 1) / n
        u = u * u * (3 - 2 * u)
        cmd = {j: start[j] + (goal_arm[j] - start[j]) * u for j in ARM_JOINTS}
        cmd["gripper"] = float(closed)
        sim.set_ctrl(cmd)
        sim.step()
    sim.hold({**{j: goal_arm[j] for j in ARM_JOINTS}, "gripper": float(closed)})
    sim.step(8)


def _js_raise_ee(sim: Sim, z_min: float = 0.10) -> bool:
    """Raise the EE in joint space so Cartesian +Z does not comb a seated man."""
    g = float(sim.qpos()["gripper"])
    pose = {n: sim.qpos()[n] for n in ARM_JOINTS}
    lo_s, hi_s = sim.joint_limits()["shoulder_lift"]
    lo_e, hi_e = sim.joint_limits()["elbow_flex"]
    for _ in range(10):
        ee, _ = sim.site_pose(EE_SITE)
        if float(ee[2]) >= float(z_min) - 1e-4:
            return True
        nxt = dict(pose)
        sl = float(np.clip(pose["shoulder_lift"] - 0.05, lo_s, hi_s))
        if abs(sl - pose["shoulder_lift"]) < 1e-4:
            nxt["elbow_flex"] = float(np.clip(pose["elbow_flex"] + 0.05, lo_e, hi_e))
        else:
            nxt["shoulder_lift"] = sl
        if all(abs(nxt[n] - pose[n]) < 1e-4 for n in ARM_JOINTS):
            break
        _move_q_hold(sim, nxt, g, duration=0.28)
        pose = nxt
    return True


def _retry_policy(sim: Sim, cfg: dict, sc: dict, failed: dict, tid: str, surface: float, release_open: float, after: str) -> Retry | None:
    """The one place a failed attempt becomes another attempt, or none.

    after="pick": re-grasp a missed or dropped piece that is still up; re-pick a
    standing piece dropped short of dest; else the recovery stage.
    after="place": the recovery stage, except bowl dumps (they stay down on
    purpose) and a piece still in the jaws; re-scan first if the piece left the belief.
    """
    reason = failed.get("failure_reason")
    if after == "pick":
        if reason in ("GRASP_MISS", "LIFT_DROP"):
            again = _grasp_retry(sim, cfg, sc, failed, tid, surface, release_open)
            if again is not None:
                return again
        dest = sc.get("dest")
        here = sc.get("source")
        if reason == "TRANSPORT_DROP" and _perceiving(sim):
            here = _reperceive(sim, cfg, tid, here)
            if here is None:
                return None
        # Carry can drop a standing man a square short (Bxf3 → g3). Re-pick
        # from here; a nest scoop is for pieces that are actually down.
        # Bowl dumps use dest=source, so "already on dest" still means retry.
        on_dest = bool(dest) and square_of_body(sim, cfg, tid) == dest
        if (
            reason == "TRANSPORT_DROP"
            and on_dest
            and _piece_still_upright(sim, tid, surface, tilt_max=0.5)
        ):
            # Already over dest. Scooping it into the cup is the wrong recovery.
            return None
        if (
            reason == "TRANSPORT_DROP"
            and int(sc.get("_recover_attempt", 0)) < 2
            and _piece_still_upright(sim, tid, surface, tilt_max=0.5)
            and (sc.get("place_xyz") is not None or not on_dest)
        ):
            nxt = dict(sc, source=here)
            nxt["seed_home"] = True
            nxt["reset_board"] = False
            nxt["_from_nest"] = False
            nxt["_recover_attempt"] = int(sc.get("_recover_attempt", 0)) + 1
            nxt["_grasp_attempt"] = 0
            return Retry(nxt, {"replaced_after_drop": True, "first_grasp_reason": reason})
    else:
        if sc.get("place_xyz") is not None:
            return None
        if reason == "RELEASE_FAILURE":
            return None
        if reason == "PLACEMENT_POSITION_ERROR":
            again = _place_correction(sim, cfg, sc, failed, tid, surface)
            if again is not None:
                return again
        if _perceiving(sim) and tid not in sim.known:
            stages.resolve(cfg)["board_observe"].scan(sim, cfg)
            if tid not in sim.known:
                failed["failure_reason"] = "PIECE_LOST"
                failed["failure_stage"] = "place"
                return None
    return stages.resolve(cfg)["recovery"].recover(sim, cfg, sc, failed, tid, surface)


def grasp_acquired(sim: Sim, object_id: str | None = None) -> bool:
    """Jaws are squeezing something: gripper servo load and speed, the only grasp sensors on the arm."""
    return jaw_squeezing(sim)


def object_on_board(sim: Sim, surface: float, z_tol: float = 0.012, object_id: str | None = None) -> bool:
    tid = object_id or _tid(sim)
    qpos, _ = sim.object_pose(tid)
    if abs(float(qpos[2]) - surface) > z_tol:
        return False
    if _perceiving(sim):
        return True
    return any(h["other"] == "board" for h in sim.contacts_with(tid))


def bowl_contains(xyz, dest_xy, half: float = 0.05) -> bool:
    """True if a piece is sitting in the capture tray (not on a chess square)."""
    dx = abs(float(xyz[0]) - float(dest_xy[0]))
    dy = abs(float(xyz[1]) - float(dest_xy[1]))
    z = float(xyz[2])
    return dx <= half and dy <= half and 0.01 <= z <= 0.09


def in_trash(xyz, cfg: dict) -> bool:
    """True if this pose is in the capture bowl, not on the board."""
    return bowl_contains(xyz, bowl_xyz(cfg))


def in_well(xyz, cfg: dict) -> bool:
    """True if this pose is in the reorient well."""
    return bowl_contains(xyz, nest_xyz(cfg), half=0.04)


def gripper_object_transform(sim: Sim, object_id: str | None = None) -> dict:
    """Object pose in gripperframe. Evaluation only — not used for control."""
    tid = object_id or _tid(sim)
    gpos, grot = sim.site_pose(EE_SITE)
    qpos, qquat = sim.object_pose(tid)
    p_ee = grot.T @ (qpos - gpos)
    return {
        "p_ee_mm": (p_ee * 1000).tolist(),
        "tilt_deg": queen_tilt_deg(qquat),
    }


def _xyz(v) -> list[float]:
    return [float(v[0]), float(v[1]), float(v[2])]


def _unique_xyz(cands, tol: float = 0.002) -> list[list[float]]:
    out: list[list[float]] = []
    for raw in cands:
        p = [float(raw[0]), float(raw[1]), float(raw[2])]
        if any(float(np.linalg.norm(np.asarray(p) - np.asarray(q))) < tol for q in out):
            continue
        out.append(p)
    return out


def hover_approach_candidates(
    xy,
    cfg: dict,
    prefer_z: float,
    lower_z: tuple[float, ...] = (0.08, 0.06, 0.04, 0.03),
    ring_zs: tuple[float, ...] | None = None,
) -> list[list[float]]:
    """High hover first, then outward, then lower — including outward+low.

    Near the base a 10 cm hover is often dead while a 3 cm pinch is fine.
    Outward+low is last so mid-board picks still take the overhead pose.
    """
    board = cfg["sim"]["board"]
    surface = float(board["surface_z"])
    size = float(board["square_size"])
    gx, gy = float(xy[0]), float(xy[1])
    zs = ring_zs if ring_zs is not None else (float(prefer_z),)
    cands: list[list[float]] = []
    for extra in (0.0, 0.5 * size, 1.0 * size, 1.5 * size):
        p = _outward_xy([gx, gy], cfg, extra)
        for z in zs:
            cands.append([float(p[0]), float(p[1]), float(z)])
    for z_off in lower_z:
        cands.append([gx, gy, surface + float(z_off)])
    for extra in (0.5 * size, 1.0 * size):
        p = _outward_xy([gx, gy], cfg, extra)
        for z_off in (0.06, 0.04):
            cands.append([float(p[0]), float(p[1]), surface + float(z_off)])
    return _unique_xyz(cands)


def ray_approach_candidates(
    pinch,
    back_xy,
    *,
    backs: tuple[float, ...] = (0.06, 0.08, 0.045, 0.030),
    zs: tuple[float, ...] = (0.11, 0.09, 0.07, 0.055),
) -> list[list[float]]:
    """Approaches along -back_xy from the pinch. High/far first, then shorter/lower."""
    axy = np.asarray(back_xy[:2], dtype=float)
    n = float(np.linalg.norm(axy))
    if n < 1e-9:
        axy, n = np.array([1.0, 0.0]), 1.0
    axy = axy / n
    px, py = float(pinch[0]), float(pinch[1])
    cands: list[list[float]] = []
    for z in zs:
        for back in backs:
            cands.append([px - float(axy[0]) * back, py - float(axy[1]) * back, float(z)])
    return _unique_xyz(cands)


def _cam_in_site(sim: Sim) -> tuple[np.ndarray, np.ndarray]:
    """Camera look direction and origin, both in the gripperframe. Fixed for a gripper opening."""
    sid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    cid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    mujoco.mj_forward(sim.model, sim.data)
    Rs = sim.data.site_xmat[sid].reshape(3, 3)
    look = Rs.T @ (-sim.data.cam_xmat[cid].reshape(3, 3)[:, 2])
    off = Rs.T @ (sim.data.cam_xpos[cid] - sim.data.site_xpos[sid])
    return look, off


def aim_camera_at(sim: Sim, ee_xyz, target_xyz) -> bool:
    """Point the wrist camera at `target_xyz` with the gripperframe held at `ee_xyz`.

    The wedge camera looks along the fingers, so the aim is that axis, not the
    old pinch-facing mount. One rule for every square.
    """
    look_site, cam_off = _cam_in_site(sim)
    ee = np.asarray(ee_xyz, dtype=float)
    tgt = np.asarray(target_xyz, dtype=float)
    sid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    for _ in range(4):
        mujoco.mj_forward(sim.model, sim.data)
        Rs = sim.data.site_xmat[sid].reshape(3, 3)
        cam = sim.data.site_xpos[sid] + Rs @ cam_off
        delta = tgt - cam
        dist = float(np.linalg.norm(delta))
        if dist < 1e-4:
            return False
        solve_ik_aim(sim, ee, look_site, delta / dist)
        mujoco.mj_forward(sim.model, sim.data)
        # The lens is wide. A stalled aim still sees the square when the point
        # sits in the frustum and the fingertips stayed on the hover.
        if wrist_sees_point(sim, tgt) and float(np.linalg.norm(sim.data.site_xpos[sid] - ee)) < 0.012:
            return True
    return False


def _servo_aim(sim: Sim, ee_xyz, target_xyz) -> bool:
    """Physically settle the aim found above. The transit locked the wrist."""
    start = sim.qpos()
    with sim.probe():
        seen = aim_camera_at(sim, ee_xyz, target_xyz)
        goal = sim.qpos()
        sim.set_qpos(start, hold=True)
    if not seen:
        return False
    _drive_arm(sim, [goal], float(start.get("gripper", 0.0)), 0.35)
    return wrist_sees_point(sim, target_xyz)


def _servo_other_roll(sim: Sim, ee_xyz, target_xyz) -> bool:
    """Drive the in-range roll that sees the square. Used when the aim has none."""
    start = sim.qpos()
    with sim.probe():
        seen = _settle_other_roll(sim, ee_xyz, target_xyz)
        goal = sim.qpos()
        sim.set_qpos(start, hold=True)
    if not seen:
        return False
    _drive_arm(sim, [goal], float(start.get("gripper", 0.0)), 0.35)
    return wrist_sees_point(sim, target_xyz)


def _settle_other_roll(sim: Sim, ee_xyz, target_xyz) -> bool:
    """Hold the hover and roll the wrist, inside its range, until the square is seen.

    The orientation aim walks roll into one stop and then either misses the
    square or pulls the fingertips off the hover. Leaves qpos on a pose that
    sees `target_xyz` with the gripper still on `ee_xyz`.
    """
    base = sim.qpos()
    jid = sim._joint["wrist_roll"]
    lo, hi = (float(v) for v in sim.model.jnt_range[jid])
    ee = np.asarray(ee_xyz, dtype=float)
    tgt = np.asarray(target_xyz, dtype=float)
    # The aim already tried the high stop. Start at the low stop and walk back.
    for roll in np.linspace(lo + 0.05, hi - 0.05, 17):
        sim.set_qpos(base, hold=True)
        sim.set_qpos({**sim.qpos(), "wrist_roll": float(roll)}, hold=True)
        if not solve_ik(sim, ee, lock_joints=("wrist_roll",)):
            continue
        pos, _ = sim.site_pose(EE_SITE)
        if float(np.linalg.norm(pos - ee)) > 0.012:
            continue
        if wrist_sees_point(sim, tgt):
            return True
    sim.set_qpos(base, hold=True)
    return False


def _observe_candidates(cfg: dict, target_xyz) -> tuple[np.ndarray, list]:
    p = pickup_cfg(cfg)
    board = cfg["sim"]["board"]
    surface = float(board["surface_z"])
    observe_z = float(p.get("observe_z", 0.12))
    target = np.asarray(target_xyz, dtype=float)
    cands = hover_approach_candidates(
        target[:2],
        cfg,
        surface + observe_z,
        lower_z=(),
        ring_zs=(surface + observe_z, surface + 0.08),
    )
    return target, cands


def _other_roll_looks(sim: Sim, cfg: dict, target_xyz) -> list[list[float]]:
    """Hovers the aim cannot use, that still see the square with roll in range."""
    target, cands = _observe_candidates(cfg, target_xyz)
    found: list[list[float]] = []
    for raw in cands:
        _reset_ik_seed(sim, cfg)
        if _settle_other_roll(sim, raw, target):
            found.append([float(raw[0]), float(raw[1]), float(raw[2])])
    return found


def find_observe_poses(sim: Sim, cfg: dict, target_xyz) -> list[list[float]]:
    """Reachable wrist poses that still see target_xyz, safest first."""
    target, cands = _observe_candidates(cfg, target_xyz)
    hits = reachable_approaches(
        sim,
        cands,
        reset=lambda: _reset_ik_seed(sim, cfg),
        upright=True,
        accept=lambda c: aim_camera_at(sim, c, target),
    )
    return hits


def _own_hand_hides(sim: Sim, target_xyz) -> bool:
    """The robot's own links cover the line of sight to the column a piece on
    target_xyz could fill; its top would read low (lying)."""
    base = np.asarray(target_xyz, dtype=float)
    top = base + np.array([0.0, 0.0, max(float(s["height"]) for s in PIECES.values())])
    return any(robot_blocks_view(sim, p, _robot_bid_array(sim)) for p in (base, top))


def _reach_observe(sim: Sim, cfg: dict, target_xyz, tid: str) -> list[float] | None:
    """Move to a wrist pose that sees target_xyz. Raise or reseeds if stuck."""
    lock = ("wrist_flex", "wrist_roll")
    obstacles = obstacle_bodies(sim, skip={tid})

    def _try(looks, duration: float) -> list[float] | None:
        hidden = None
        for look in looks:
            if _safe_lin(sim, look, cfg, lock, obstacles=obstacles, skip={tid}, duration=duration):
                if not _servo_aim(sim, look, target_xyz):
                    continue
                if not _own_hand_hides(sim, target_xyz):
                    return look
                hidden = hidden or look
        if hidden is not None and _safe_lin(sim, hidden, cfg, lock, obstacles=obstacles, skip={tid}, duration=duration):
            return hidden
        return None

    hit = _try(find_observe_poses(sim, cfg, target_xyz), 0.8)
    if hit is not None:
        return hit
    _js_raise_ee(sim, z_min=0.12)
    hit = _try(find_observe_poses(sim, cfg, target_xyz), 0.8)
    if hit is not None:
        return hit
    seed_vertical(sim, cfg)
    hit = _try(find_observe_poses(sim, cfg, target_xyz), 0.9)
    if hit is not None:
        return hit
    # Aimed looks are exhausted. The aim walks wrist_roll into one stop; on the
    # far squares that stop leaves the hover. The other side of the range still
    # reaches the same hover and holds the square in the lens.
    for look in _other_roll_looks(sim, cfg, target_xyz):
        if _safe_lin(sim, look, cfg, lock, obstacles=obstacles, skip={tid}, duration=0.9):
            if _servo_other_roll(sim, look, target_xyz) and not _own_hand_hides(sim, target_xyz):
                return look
    return None


def find_pregrasp_poses(sim: Sim, cfg: dict, grasp_xy) -> list[list[float]]:
    """Reachable pregrasps that can still IK to the grasp, safest first."""
    p = pickup_cfg(cfg)
    board = cfg["sim"]["board"]
    surface = float(board["surface_z"])
    pre_z = float(p.get("pregrasp_z", 0.10))
    grasp_z = float(p.get("grasp_z", 0.015))
    gx, gy = float(grasp_xy[0]), float(grasp_xy[1])
    grasp = [gx, gy, surface + grasp_z]
    cands = hover_approach_candidates([gx, gy], cfg, surface + pre_z)
    return reachable_approaches(
        sim, cands, grasp, reset=lambda: _reset_ik_seed(sim, cfg), upright=True
    )


def find_pregrasp_pose(sim: Sim, cfg: dict, grasp_xy) -> list[float] | None:
    """Reachable pregrasp that can still IK to the given grasp XY.

    Prefers the default hover; if that sits in the near-base hover hole,
    steps outward, then lower, then outward+low. Restores qpos.
    """
    hits = find_pregrasp_poses(sim, cfg, grasp_xy)
    return hits[0] if hits else None


def _already_upright(sim, tid: str) -> bool:
    known = getattr(sim, "known", None)
    if known is None or tid not in known:
        return False
    _, quat = sim.object_pose(tid)
    return queen_upright(quat)


def _short_view_of_standing(sim, cfg: dict, square: str, seen, height: float) -> bool:
    """The crown is outside the frame, so a low cloud is the view, not a fallen piece.

    A lying body shows a low extent while the standing crown is still in the
    lens. A column cut off above the base does not.
    """
    if seen is None or seen["top_h"] > VERIFY_UPRIGHT_FRAC * height:
        return False
    surface = float(cfg.get("sim", {}).get("board", {}).get("surface_z", 0.02))
    crown = square_to_world(square, cfg, z=surface + float(height))
    return not wrist_sees_point(sim, crown)


def locate_queen(sim: Sim, cfg: dict, result: dict, square: str) -> np.ndarray | None:
    """RGB-D estimate from the wrist sensor. Ground truth is logged for evaluation only."""
    tid = _tid(sim)
    obs = capture_wrist(sim)
    est = estimate_target_pose(obs, cfg, square)
    gt, _ = world_of(sim).object_pose(tid)
    if est is None:
        result["estimated_position_error_mm"] = None
        return None
    if hasattr(sim, "believe"):
        surface = float(cfg.get("sim", {}).get("board", {}).get("surface_z", 0.02))
        # The wedge lens looks along the fingers. At the pregrasp that aim has
        # carried it past the square: a sliver of the base is still in the crop,
        # and taking it as the pose shifts the pinch onto the piece. A frame
        # that no longer sees the square does not replace one that did.
        center = square_to_world(square, cfg, z=surface)
        if tid in sim.known and not wrist_sees_point(sim, center):
            return None
        seen = observe_piece(obs, cfg, square, grow=1.0)
        height = float((PIECES.get(body_kind(tid) or tid) or {}).get("height", 0.04))
        hidden = _own_hand_hides(sim, square_to_world(square, cfg, z=surface))
        # A later frame may replace an upright belief with a fallen one only
        # when the cloud is what a lying piece shows. A short view, crown
        # outside the frame, is not that.
        upright = (
            seen is None
            or hidden
            or seen["top_h"] > VERIFY_UPRIGHT_FRAC * height
            or (_already_upright(sim, tid) and _short_view_of_standing(sim, cfg, square, seen, height))
        )
        quat = (1.0, 0.0, 0.0, 0.0) if upright else (0.7071068, 0.7071068, 0.0, 0.0)
        result["observed_top_h_mm"] = None if seen is None else round(seen["top_h"] * 1000, 1)
        result["observe_hidden"] = hidden
        sim.believe(tid, [float(est[0]), float(est[1]), surface], quat)
    err = float(np.linalg.norm(est[:2] - gt[:2]))
    result["estimated_position_error_mm"] = err * 1000.0
    result["queen_estimated_xyz"] = est[:3].tolist()
    if not getattr(sim, "quiet", False):
        print(f"  estimated_xyz: {np.round(est[:3], 4).tolist()}")
        print(f"  gt_xyz:        {np.round(gt, 4).tolist()}")
        print(f"  position_error_mm: {err * 1000:.1f}")
    return est[:3]


def _perceiving(sim) -> bool:
    """Controller state comes from perception (Robot wall without cheat_state)."""
    return hasattr(sim, "believe") and not sim.cheat_state


def verify_piece(sim, cfg: dict, tid: str, square: str) -> dict:
    """Wrist look at `square`; belief takes the observed XY and upright/lying."""
    diag.emit("look_start", sim, tid=tid, square=square)
    out = _verify_piece(sim, cfg, tid, square)
    diag.emit("look", sim, tid=tid, square=square, out=out)
    return out


def _verify_piece(sim, cfg: dict, tid: str, square: str) -> dict:
    surface = float(cfg.get("sim", {}).get("board", {}).get("surface_z", 0.02))
    target = square_to_world(square, cfg, z=surface)
    if _reach_observe(sim, cfg, target, tid) is None:
        return {"looked": False}
    obs = capture_wrist(sim)
    seen = observe_piece(obs, cfg, square)
    if seen is None:
        sim.forget(tid)
        return {"looked": True, "seen": False}
    height = float((PIECES.get(body_kind(tid) or tid) or {}).get("height", 0.04))
    upright = seen["top_h"] > VERIFY_UPRIGHT_FRAC * height or (
        _already_upright(sim, tid) and _short_view_of_standing(sim, cfg, square, seen, height)
    )
    xy, fitted = seen["xy"], False
    if upright:
        fit = fit_upright_xy(obs, cfg, piece_shape(sim, tid), seen["xy"])
        if fit is not None:
            xy, fitted = fit["xy"], True
    quat = (1.0, 0.0, 0.0, 0.0) if upright else (0.7071068, 0.7071068, 0.0, 0.0)
    sim.believe(tid, [xy[0], xy[1], surface], quat)
    return {"looked": True, "seen": True, "upright": upright, "top_h_mm": round(seen["top_h"] * 1000, 1), "xy": xy, "fitted": fitted}


VERIFY_UPRIGHT_FRAC = 0.7


def execute_move(sim: Sim, cfg: dict, spawn_xy=None, scenario: dict | None = None) -> dict:
    """Run one move through the stage pipeline (`stages.resolve(cfg)`); world-side scoring stays here."""
    sc = dict(scenario or {})
    p = pickup_cfg(cfg)
    tid = sc.get("target_id", "queen")
    sim.target_id = tid
    sim.quiet = bool(sc.get("quiet", False) or getattr(sim, "quiet", False))
    source = sc.get("source", p.get("source", "d1"))
    dest = sc.get("dest", p.get("dest", "d4"))
    # The wrist look is how a robot without truth knows where pieces are; cheat/off read the true pose (the control).
    use_vision = _perceiving(sim)
    perception = "vision" if use_vision else "truth"
    offset_xy = [float(v) for v in sc.get("offset_xy", [0.0, 0.0])]
    yaw_deg = float(sc.get("yaw_deg", 0.0))
    geom = dict(sc.get("geometry") or PIECES.get(body_kind(tid) or tid) or {})
    off = sc.get("grasp_xy_offset", geom.get("grasp_xy_offset", p.get("grasp_xy_offset", [0.0, 0.0])))
    surface = float(cfg.get("sim", {}).get("board", {}).get("surface_z", 0.02))
    open_g = float(p.get("gripper_open", 0.60))
    release_open = float(p.get("gripper_release", 0.30))
    closed_g = float(geom.get("gripper_closed", p.get("gripper_closed", -0.05)))
    pre_z = float(p.get("pregrasp_z", 0.10))
    grasp_z = float(geom.get("preferred_grasp_height", p.get("grasp_z", 0.016)))
    lift_z = float(p.get("lift_z", 0.08))
    if sc.get("_from_nest"):
        lift_z = max(lift_z, 0.06)
        # Pinch is on the exposed stem above the lip; 0.30 still wraps a crown.
        release_open = max(release_open, 0.40)
    place_hover = float(p.get("place_hover", 0.003))
    xy_tol = float(p.get("place_xy_tol", 0.010))
    place_xyz = sc.get("place_xyz")
    off_board = place_xyz is not None
    if off_board:
        dest_xy = np.asarray(place_xyz, dtype=float)
        if dest_xy.size == 2:
            dest_xy = np.array([dest_xy[0], dest_xy[1], surface], dtype=float)
        xy_tol = max(xy_tol, 0.05)
    else:
        dest_xy = np.asarray(square_to_world(dest, cfg, z=surface), dtype=float)
    result = {
        "success": False,
        "queen_detected": True,
        "pregrasp_reached": False,
        "grasp_attempted": False,
        "grasp_acquired": False,
        "lift_success": False,
        "transport_success": False,
        "placement_success": False,
        "retreat_success": None,
        "failure_reason": None,
        "failure_category": None,
        "ik_fail_phase": None,
        "neighbor_collision_phase": None,
        "piece": geom.get("kind", tid),
        "scenario": sc.get("name", "queen_d1_d4"),
        "seed": sc.get("seed", 0),
        "target_id": tid,
        "source": source if spawn_xy is None else "random",
        "destination": dest,
        "dest": dest,
        "use_vision": use_vision,
        "perception": perception,
        "offset_xy_mm": [offset_xy[0] * 1000.0, offset_xy[1] * 1000.0],
        "yaw_deg": yaw_deg,
        "geometry": sc.get("geometry"),
        "grasp_xy_offset_mm": [float(off[0]) * 1000.0, float(off[1]) * 1000.0],
        "ik_solved": False,
        "approach_reached": False,
        "approach_ok": False,
        "first_collision": None,
        "transit_z": None,
        "transit_z_derived": None,
        "min_queen_piece_clearance_mm": None,
        "min_gripper_piece_clearance_mm": None,
        "min_camera_piece_clearance_mm": None,
        "transport_contact_lift": False,
        "transport_contact_transit": False,
        "transport_contact_descend": False,
        "transport_first_collision": None,
        "stop_after": sc.get("stop_after"),
        "obstruction_body": None,
        "obstruction_square": None,
        "obstruction_piece": None,
        "obstruction_robot_part": None,
        "obstruction_phase": None,
    }
    sim._last_ik_fail = None

    W = world_of(sim)  # evaluator view: collateral, first-collision, neighbor stats
    first_hit = [None]

    def _clutter_names() -> list[str]:
        names = []
        for i in range(W.model.nbody):
            name = mujoco.mj_id2name(W.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not name or name == tid or body_kind(name) is None:
                continue
            if float(W.data.xpos[i][2]) > 0.0 and not in_trash(W.data.xpos[i], cfg) and not in_well(W.data.xpos[i], cfg):
                names.append(name)
        return names

    # Fast reject for the per-step hook: with 32 pieces resting on the board
    # there are ~160 contacts per step, almost all piece<->board. Only a
    # clutter piece touching a robot part or the target needs the slow path.
    tid_bid = W.body_id(tid)
    inert_ids = np.array(
        [b for b in (W.body_id("board"), W.body_id("floor"), 0) if b >= 0], dtype=int
    )
    piece_ids = np.array(
        [
            i
            for i in range(W.model.nbody)
            if body_kind(W.body_name(i)) is not None
        ],
        dtype=int,
    )

    def _clutter_hit_possible() -> bool:
        b1, b2 = W.contact_bodies()
        if b1.size == 0:
            return False
        z = W.data.xpos[:, 2]
        live = piece_ids[(z[piece_ids] > 0.0) & (piece_ids != tid_bid)]
        in1 = np.isin(b1, live)
        in2 = np.isin(b2, live)
        cand = in1 | in2
        if not cand.any():
            return False
        other = np.where(in1, b2, b1)[cand]
        # other must be neither inert scenery nor another clutter piece
        # (clutter<->clutter is skipped by the slow path unless it is the target)
        ok = ~np.isin(other, inert_ids) & (~np.isin(other, live) | (other == tid_bid))
        return bool(ok.any())

    def _track_clutter():
        if first_hit[0] is not None:
            return
        phase = getattr(sim, "_pickup_phase", None)
        if phase in (None, "001_initial"):
            return
        if not _clutter_hit_possible():
            return

        def _part_touching(*bodies: str) -> str | None:
            for b in bodies:
                for h in W.contacts_with(b):
                    part = robot_part_walk(W, h.get("other"))
                    if part:
                        return part
            return None

        for name in _clutter_names():
            for h in W.contacts_with(name):
                other = h.get("other")
                if other in (None, "board", "floor", "world"):
                    continue
                part = robot_part_walk(W, other)
                if part is None and other != tid:
                    continue
                if part is None:
                    part = _part_touching(name, tid)
                sq = square_of_body(W, cfg, name)
                first_hit[0] = {
                    "body": name,
                    "other": other,
                    "phase": phase,
                    "dist_mm": float(h["dist"] * 1000),
                    "camera_involved": part == "camera",
                    "robot_part": part,
                    "square": sq,
                    "piece": body_kind(name) or (
                        STARTING_AT.get(sq) if sq else (other if other in PIECE_BODIES else None)
                    ),
                }
                result["first_collision"] = first_hit[0]
                return

    prev_all = sim.on_step

    def _clutter_hook():
        if prev_all:
            prev_all()
        _track_clutter()

    sim.on_step = _clutter_hook

    def _fill_fail_diag():
        pos, _ = W.site_pose(EE_SITE)
        reason = result.get("failure_reason")
        stash = getattr(sim, "_last_ik_fail", None)
        ik_reasons = ("IK_FAILURE", "IK_NO_SOLUTION")
        if isinstance(stash, dict) and reason in ik_reasons:
            result["ee_requested"] = stash.get("ee_requested")
            result["ee_actual"] = stash.get("ee_actual")
            result["ik_residual_mm"] = stash.get("ik_residual_mm")
            result["joint_limit_margin_rad"] = stash.get("joint_limit_margin_rad")
            if stash.get("lin_wp"):
                result["ik_lin_wp"] = stash["lin_wp"]
            if stash.get("lin_goal"):
                result["ik_lin_goal"] = stash["lin_goal"]
        else:
            if result.get("ee_actual") is None:
                result["ee_actual"] = _xyz(pos)
            req = result.get("ee_requested")
            if req is not None and result.get("ik_residual_mm") is None:
                result["ik_residual_mm"] = round(
                    float(np.linalg.norm(np.asarray(result["ee_actual"], dtype=float) - np.asarray(req, dtype=float)))
                    * 1000.0,
                    2,
                )
            if result.get("joint_limit_margin_rad") is None:
                result["joint_limit_margin_rad"] = joint_limit_margin(W)

        hit = result.get("first_collision") or result.get("transport_first_collision")
        plan = result.get("approach_plan") or {}
        collision_reasons = (
            "APPROACH_COLLISION",
            "NO_COLLISION_FREE_APPROACH",
            "TRANSPORT_COLLISION",
        )
        if hit:
            robot = hit.get("robot_part") or hit.get("other")
            body = hit.get("body")
            dist = hit.get("dist_mm")
            clear = dist
            if clear is None and robot and body:
                try:
                    clear = round(min_body_gap(W, robot, body) * 1000.0, 2)
                except Exception:
                    pass
            result["collision_pair"] = {
                "robot": robot,
                "body": body,
                "dist_mm": None if dist is None else round(float(dist), 2),
                "clearance_mm": clear,
                "square": hit.get("square"),
                "phase": hit.get("phase"),
            }
        elif plan.get("block"):
            blk = plan["block"]
            result["collision_pair"] = {
                "robot": blk.get("robot"),
                "body": blk.get("body"),
                "dist_mm": blk.get("clearance_mm"),
                "clearance_mm": blk.get("clearance_mm"),
                "square": square_of_body(W, cfg, blk.get("body") or ""),
                "phase": "approach_plan",
            }
        elif reason in collision_reasons:
            clutter = list(result.get("occupied_neighbors") or [])
            pair = closest_pair(W, APPROACH_BODIES, clutter) if clutter else None
            if pair:
                result["collision_pair"] = {
                    "robot": pair.get("robot"),
                    "body": pair.get("body"),
                    "dist_mm": pair.get("clearance_mm"),
                    "clearance_mm": pair.get("clearance_mm"),
                    "square": square_of_body(W, cfg, pair.get("body") or ""),
                    "phase": getattr(sim, "_pickup_phase", None),
                }

    collateral_start: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _collateral_snapshot():
        collateral_start.clear()
        for name in _clutter_names():
            p, q = W.object_pose(name)
            collateral_start[name] = (np.asarray(p, dtype=float).copy(), np.asarray(q, dtype=float).copy())

    def _collateral_check():
        """Evaluator: bystanders knocked over or shoved off their squares (fails `world_success`). Never fed back."""
        hits = []
        for name, (p0, _q0) in collateral_start.items():
            try:
                p, q = W.object_pose(name)
            except Exception:
                continue
            if float(p[2]) <= 0.0 or float(p0[2]) <= 0.0:
                continue
            # Captured men already in the bowl may tumble when the next
            # capture lands on them. That is trash, not a board failure.
            if in_trash(p0, cfg) or in_well(p0, cfg):
                continue
            disp, tilt, hit = evaluate_move.disturbance(p0, _q0, p, q)
            if hit:
                hits.append(
                    {
                        "body": name,
                        "square": square_of_body(W, cfg, name),
                        "xy_disp_mm": round(disp, 1),
                        "tilt_deg": round(tilt, 1),
                        "first_contact": first_hit[0] if first_hit[0] and first_hit[0].get("body") == name else None,
                    }
                )
        result["collateral"] = hits
        if hits:
            fc = hits[0].get("first_contact") or first_hit[0] or {}
            result["neighbor_collision_phase"] = fc.get("phase")

    def _done():
        _track_clutter()
        _collateral_check()
        if result.get("failure_reason") in ("IK_FAILURE", "IK_NO_SOLUTION") and not result.get("ik_fail_phase"):
            result["ik_fail_phase"] = getattr(sim, "_pickup_phase", None)
        _fill_fail_diag()
        wp, wq = W.object_pose(tid)
        if off_board:
            world_ok = bowl_contains(wp, dest_xy)
        else:
            world_ok = evaluate_move.placement(W, cfg, tid, dest)["ok"]
        result["world_success"] = bool(world_ok and not result["collateral"])
        annotate_episode(result)
        return result

    if spawn_xy is not None:
        spawn = list(spawn_xy)
    else:
        base = square_to_world(source, cfg, z=surface)
        spawn = [base[0] + offset_xy[0], base[1] + offset_xy[1], base[2]]
    result["spawn_xyz"] = _xyz(spawn)
    if sc.get("seed_home", sc.get("reset_board", True)):
        seed_vertical(sim, cfg)
    m = Move(sim, cfg, sc, result)
    for k, v in dict(p=p, tid=tid, source=source, dest=dest, use_vision=use_vision, geom=geom, surface=surface,
                     open_g=open_g, release_open=release_open, closed_g=closed_g, pre_z=pre_z, grasp_z=grasp_z,
                     lift_z=lift_z, place_hover=place_hover, off_board=off_board, dest_xy=dest_xy, xy_tol=xy_tol).items():
        setattr(m, k, v)
    P = stages.resolve(cfg, "carry:nudge" if sc.get("_nudge") else None)

    def _end(out):
        if isinstance(out, Stop):
            result["success"] = True
            result["failure_reason"] = None
            return _done()
        nxt = _retry_policy(sim, cfg, sc, result, tid, surface, release_open, out.retry) if out.retry else None
        if nxt is None:
            return _done()
        again = execute_move(sim, cfg, spawn_xy=None, scenario=nxt.scenario)
        again.update(nxt.notes)
        if nxt.retry_log is not None:
            again["grasp_retry_log"] = nxt.retry_log + list(again.get("grasp_retry_log") or [])
        return again

    P["retreat"].unhook(m)
    _collateral_snapshot()

    actual, _ = W.object_pose(tid)
    result["actual_start_pose"] = actual.tolist()
    result["queen_start_xyz"] = actual.tolist()
    log_state(sim, "001_initial", extra={"spawn": _xyz(spawn)})
    m.actual = actual

    steps = (P["observe"].locate, P["grasp_plan"].plan, P["approach"].run, P["close"].close,
             P["carry"].carry, P["release"].release, P["retreat"].back_off)
    for step in steps:
        out = step(m)
        if out is not None:
            return _end(out)
    out = P["observe"].confirm(m)

    qpos, qquat = W.object_pose(tid)
    result["final_position_error_mm"] = float(np.hypot(qpos[0] - dest_xy[0], qpos[1] - dest_xy[1])) * 1000
    result["final_queen_xyz"] = qpos.tolist()
    result["place_diag"]["final_err_mm"] = round(result["final_position_error_mm"], 1)
    result["place_diag"]["final_upright"] = bool(queen_upright(qquat, tilt_max=0.5))
    if out is not None:
        return _end(out)
    result["success"] = True
    result["failure_reason"] = None
    return _done()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--full-board", action="store_true", help="load starting-position chess_scene_full.xml")
    parser.add_argument("--random", action="store_true", help="spawn queen at a random on-board XY, not a square")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", action="store_true", help="write overview+side mp4 under sim/debug/pickup/")
    parser.add_argument("--wall", choices=WALLS, default="real", help="controller/world separation (robot.py)")
    stages.cli_arg(parser)
    args = parser.parse_args()
    stages.select(args.stage)
    cfg = load_config()
    spawn_xy = None
    if args.random:
        rng = np.random.default_rng(args.seed)
        spawn_xy = jitter_near_square(cfg, pickup_cfg(cfg).get("source", "d1"), rng)
        print(f"random spawn seed={args.seed} xyz={np.round(spawn_xy, 4).tolist()}")
    xml = FULL_BOARD_XML if args.full_board else None
    sc = None
    if args.full_board:
        from scenario import full_board_scenario

        sc = full_board_scenario()
    from episode import run_pickup, setup_board

    world = Sim(xml_path=xml, cfg=cfg)
    setup_board(world, cfg, sc or {}, spawn_xy=spawn_xy)
    sim = open_robot(world, args.wall)
    rec = None
    if args.video:
        tag = args.wall
        if args.random:
            tag += f"_random{args.seed}"
        rec = VideoRecorder(sim, DEBUG_DIR / "pickup" / f"{tag}.mp4")
        sim.on_step = rec.grab
    try:
        result = run_pickup(sim, cfg, spawn_xy=spawn_xy, scenario=sc)
        if rec is not None:
            rec.save()
            print(f"video: {rec.path} ({len(rec.frames)} frames)")
    finally:
        sim.close()
    print("\n" + json.dumps(result, indent=2))
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    (DEBUG_DIR / "pickup" / "result.json").write_text(json.dumps(result, indent=2))
    sys.exit(0 if result["success"] and result.get("world_success", True) else 1)


if __name__ == "__main__":
    main()
