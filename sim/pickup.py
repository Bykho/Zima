#!/usr/bin/env python3
"""Pick and place: queen → lift → d4 → release.

  python sim/pickup.py --headless
  python sim/pickup.py --headless --vision
  python sim/pickup.py --headless --vision --random --seed 0 --video
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

from ik import (
    CARRY_ORI_WEIGHT,
    EE_SITE,
    _store_ik_fail,
    cart_duration,
    grasp_xy_delta,
    gripper_tilt_deg,
    move_ik,
    move_lin,
    move_qpos,
    solve_ik,
    solve_ik_down,
    solve_ik_upright,
)
from mj import ARM_JOINTS, DEBUG_DIR, JOINT_NAMES, Sim, load_config
from clearance import (
    APPROACH_BODIES,
    board_top_z,
    closest_pair,
    neighbor_approach_gap,
    obstacle_bodies,
    plan_approach,
    safe_transit_ee_z,
    transit_clearances,
    transit_nontarget_hit,
)
from perception import estimate_target_pose
from sensor import capture as capture_wrist, wrist_sees_point
from squares import jitter_near_square, nest_xyz, square_to_world
from metrics import annotate_episode
from scenario import PIECE_BODIES, PIECES, STARTING_AT, body_kind
from sequence import square_of_body

JAW_BODIES = ("gripper", "moving_jaw_so101_v1")
CAMERA_BODIES = ("camera_mount",)
ARM_BODIES = ("base", "shoulder", "upper_arm", "lower_arm")
FULL_BOARD_XML = ROOT / "assets/so101/chess_scene_full.xml"
PIECES_XML = ROOT / "assets/so101/chess_scene_pieces.xml"


def _tid(sim: Sim) -> str:
    return getattr(sim, "target_id", "queen")


def _proxy_body_names(sim: Sim) -> list[str]:
    names = []
    for i in range(sim.model.nbody):
        n = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, i)
        if n and n.startswith("proxy_"):
            names.append(n)
    return names


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


def _approach_contact(desc: dict) -> bool:
    """True if descend/IK goal involved target or environment contact."""
    if desc.get("first_contact_jaw"):
        return True
    if int(desc.get("pre_close_contacts") or 0) > 0:
        return True
    if float(desc.get("queen_xy_mm") or 0.0) > 1.0:
        return True
    ikj = desc.get("ik_jaw") or {}
    if int(ikj.get("n_contact") or 0) > 0:
        return True
    gap = ikj.get("min_gap_mm")
    if gap is not None and float(gap) < 0.0:
        return True
    return False


def _descend_fail_reason(desc: dict) -> str:
    if not desc.get("ik_solved"):
        return "IK_NO_SOLUTION"
    if _approach_contact(desc):
        return "APPROACH_COLLISION"
    return "APPROACH_TRACK_FAIL"


def _xy_seg_dist(p, a, b) -> float:
    p = np.asarray(p[:2], dtype=float)
    a = np.asarray(a[:2], dtype=float)
    b = np.asarray(b[:2], dtype=float)
    ab = b - a
    den = float(np.dot(ab, ab))
    t = 0.0 if den < 1e-12 else float(np.clip(np.dot(p - a, ab) / den, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def carry_blocked(obstacles_xy, start_xy, dest_xy, radius: float = 0.028) -> bool:
    """True if any obstacle XY is inside `radius` of the start→dest segment."""
    for o in obstacles_xy:
        if _xy_seg_dist(o, start_xy, dest_xy) < radius:
            return True
    return False


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
    sim._pickup_phase = name
    if getattr(sim, "quiet", False):
        return
    tid = _tid(sim)
    qpos, qquat = sim.object_pose(tid)
    gpos, grot = sim.site_pose(EE_SITE)
    gap = sim.jaw_gap()
    contacts = sim.contacts_with(tid)
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
            "shoulder_lift": 0.0,
            "elbow_flex": 0.47,
            "wrist_flex": 1.18,
            "wrist_roll": 1.58,
            "gripper": 0.60,
        },
    )


_CLEAR_GAP = 0.004


def seed_vertical(sim: Sim, cfg: dict, snap: bool = False):
    """Return to the vertical seed. Interpolate unless `snap` (game start)."""
    seed = _vertical_seed(cfg)
    if snap:
        sim.set_qpos(seed)
        sim.hold(seed)
        for _ in range(40):
            sim.step()
        return
    cur = sim.qpos()
    err = max(abs(cur.get(k, 0.0) - float(seed.get(k, 0.0))) for k in seed)
    if err < 0.03:
        sim.hold(seed)
        return
    _clear_to_seed(sim, cfg, seed)


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
            wps = [_floor_wp(wp, cfg) for wp in _line_wps(cur, g, n_wp)]
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


def _reset_ik_seed(sim: Sim, cfg: dict):
    seed = _vertical_seed(cfg)
    sim.set_qpos(seed, hold=True)


def _offset_grasp_xy(sim: Sim, piece_xy, off) -> list[float]:
    """Apply grasp XY in the current gripper frame, projected onto world XY."""
    _, grot = sim.site_pose(EE_SITE)
    d = grasp_xy_delta(grot, off)
    return [float(piece_xy[0] + d[0]), float(piece_xy[1] + d[1])]


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


def _seeded_grasp_xy(sim: Sim, cfg: dict, piece_xy, geom: dict | None, z: float) -> list[float]:
    """Grasp EE XY from the pinch model at the current wrist roll."""
    del cfg
    return _grasp_xy_for_roll(sim, piece_xy, z, geom, None)


_WORKSPACE_R_M = 0.19


def _arm_base_xy(cfg: dict) -> np.ndarray:
    sim = cfg.get("sim", {})
    raw = sim.get("arm_base_position") or sim.get("arm_base_position") or [0.08, 0.045, 0.0]
    return np.asarray(raw[:2], dtype=float)


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


def _carry_lin(
    sim: Sim,
    target,
    duration: float,
    n_wp: int,
    hold: tuple[str, ...],
    extract: tuple[str, ...],
    ori_weight: float = CARRY_ORI_WEIGHT,
) -> bool:
    """Carry/place Cartesian move: keep the piece upright, else lock the close wrist."""
    if move_lin(
        sim,
        target,
        duration=duration,
        physics=True,
        n_wp=n_wp,
        lock_joints=hold,
        ori_weight=ori_weight,
    ):
        return True
    return move_lin(
        sim, target, duration=duration, physics=True, n_wp=n_wp, lock_joints=extract
    )


def _place_ee_target(sim: Sim, tid: str, dest_xy, z_ee: float, lock: tuple[str, ...]) -> list[float] | None:
    """EE point at height z_ee that puts the held piece over dest_xy with gripper +X world-down.

    piece = ee + R p_local, with p_local measured now (the pinch offset) and R
    the vertical-gripper rotation at dest (solved, roll locked). Restores qpos.
    """
    gpos, grot = sim.site_pose(EE_SITE)
    q, _ = sim.object_pose(tid)
    p_local = grot.T @ (q - gpos)
    snap = sim.qpos()
    try:
        ee = np.array([float(dest_xy[0]), float(dest_xy[1]), float(z_ee)])
        for _ in range(2):
            if not solve_ik_down(sim, ee, lock_joints=lock):
                return None
            _, R = sim.site_pose(EE_SITE)
            off = R @ p_local
            ee = np.array([float(dest_xy[0]) - off[0], float(dest_xy[1]) - off[1], float(z_ee)])
        return [float(v) for v in ee]
    finally:
        sim.set_qpos(snap, hold=True)


def _lower_vertical(sim: Sim, tid: str, dest_xy, place_z: float, hold: tuple[str, ...], result: dict) -> list[float] | None:
    """Align the held piece over dest at the current height, then descend plumb."""
    gpos, _ = sim.site_pose(EE_SITE)
    align = _place_ee_target(sim, tid, dest_xy, float(gpos[2]), hold)
    if align is not None:
        if not move_lin(sim, align, duration=0.8, physics=True, n_wp=8, lock_joints=hold, down=True):
            return None
    else:
        # Dest inside the near-base hole: a vertical wrist over it only exists
        # low (r=100 mm: z<=0.04). Drop to skim height here on the ring, slide
        # in plumb at that height, then finish the last few mm straight down.
        skim_z = place_z + 0.005
        down = [float(gpos[0]), float(gpos[1]), skim_z]
        if not move_lin(sim, down, duration=0.8, physics=True, n_wp=8, lock_joints=hold, down=True):
            return None
        skim = _place_ee_target(sim, tid, dest_xy, skim_z, hold)
        if skim is None:
            return None
        if not move_lin(sim, skim, duration=1.0, physics=True, n_wp=12, lock_joints=hold, down=True):
            return None
        result["place_diag"]["lower_standoff_skim"] = True
    place = _place_ee_target(sim, tid, dest_xy, place_z, hold)
    if place is None:
        return None
    result["place_diag"]["lower_target_xy_mm"] = [round(place[0] * 1000, 1), round(place[1] * 1000, 1)]
    result["place_diag"]["lower_vertical"] = True
    if not move_lin(sim, place, duration=1.2, physics=True, n_wp=12, lock_joints=hold, down=True):
        return None
    return place


def _carry_candidates(start, goal, cfg: dict, n_wp: int, lock: tuple[str, ...]):
    """Locked straight (floored to r≥190 mm), then a higher line, then unlock wrist_flex."""
    n_wp = max(8, int(n_wp))
    wps = [_floor_wp(wp, cfg) for wp in _line_wps(start, goal, n_wp)]
    yield wps, lock, "straight"
    high = [[w[0], w[1], w[2] + 0.03] for w in wps]
    yield high, lock, "high"
    if "wrist_flex" in lock:
        unlock = tuple(j for j in lock if j != "wrist_flex")
        yield wps, unlock, "straight_unlock_flex"


def _transport_to(
    sim: Sim,
    goal,
    lock: tuple[str, ...],
    cfg: dict,
    n_wp: int,
    duration: float,
    obstacles=None,
    skip=None,
) -> tuple[bool, str]:
    """Carry EE to goal XY. Probe candidates before moving so a failed line does not drop the piece."""
    goal = np.asarray(goal, dtype=float)
    skip = set(skip or ())
    extract = tuple(dict.fromkeys((*lock, "wrist_flex")))
    moved = False
    for ori, lk_extra in ((CARRY_ORI_WEIGHT, ()), (0.0, ("wrist_flex",))):
        if moved:
            break
        start, _ = sim.site_pose(EE_SITE)
        for wps, lk, label in _carry_candidates(start, goal, cfg, n_wp, lock):
            use = tuple(dict.fromkeys((*lk, *lk_extra)))
            if not _chain_ik_ok(sim, wps, use, obstacles=obstacles, skip=skip, ori_weight=ori):
                continue
            if _follow_wps(sim, wps, duration, use, ori_weight=ori):
                return True, label
            moved = True
            break
    goal_f = _floor_wp(goal, cfg)
    if _carry_lin(sim, goal_f, duration, max(8, n_wp), lock, extract):
        return True, "straight"
    return False, "straight"


def settle_queen(sim: Sim, steps: int = 24):
    sim.step(max(1, int(steps)))


def queen_tilt_deg(quat) -> float:
    _, x, y, _ = quat
    zz = 1 - 2 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def jaw_contacts(sim: Sim, object_id: str | None = None) -> list[dict]:
    tid = object_id or _tid(sim)
    return [
        c
        for c in sim.contacts_with(tid)
        if c["other"] in JAW_BODIES and (c["force"] > 0.02 or c["dist"] < 0.0)
    ]


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


def _contact_jaws(hits: list[dict]) -> list[str]:
    names = []
    if any(h["other"] == "gripper" for h in hits):
        names.append("fixed")
    if any(h["other"] == "moving_jaw_so101_v1" for h in hits):
        names.append("moving")
    return names


def joint_limit_margin(sim: Sim) -> dict:
    q = sim.qpos()
    margins = {}
    best = 1e9
    tightest = None
    for n in ARM_JOINTS:
        lo, hi = sim.model.jnt_range[sim._joint[n]]
        v = q[n]
        m = float(min(v - lo, hi - v))
        margins[n] = round(m, 4)
        if m < best:
            best = m
            tightest = n
    return {"min_rad": float(best), "tightest": tightest, "per_joint": margins}


def jaw_snapshot(sim: Sim, object_id: str | None = None) -> dict:
    """Open-jaw geometry relative to the target. Evaluation only."""
    tid = object_id or _tid(sim)
    qpos, _ = sim.object_pose(tid)
    ee, _ = sim.site_pose(EE_SITE)
    fix = sim.geom_xpos("fixed_jaw_sph_tip1")
    mov = sim.geom_xpos("moving_jaw_sph_tip1")
    pinch = 0.5 * (fix + mov)
    hits = jaw_contacts(sim, tid)
    return {
        "ee_xy_mm": [float(ee[0] * 1000), float(ee[1] * 1000)],
        "queen_xy_mm": [float(qpos[0] * 1000), float(qpos[1] * 1000)],
        "fixed_tip_xy_mm": [float(fix[0] * 1000), float(fix[1] * 1000)],
        "moving_tip_xy_mm": [float(mov[0] * 1000), float(mov[1] * 1000)],
        "pinch_xy_mm": [float(pinch[0] * 1000), float(pinch[1] * 1000)],
        "fixed_to_queen_xy_mm": [float((fix[0] - qpos[0]) * 1000), float((fix[1] - qpos[1]) * 1000)],
        "moving_to_queen_xy_mm": [float((mov[0] - qpos[0]) * 1000), float((mov[1] - qpos[1]) * 1000)],
        "pinch_to_queen_xy_mm": [float((pinch[0] - qpos[0]) * 1000), float((pinch[1] - qpos[1]) * 1000)],
        "jaw_gap_mm": float(sim.jaw_gap() * 1000),
        "min_gap_mm": float(min_jaw_gap(sim, tid) * 1000),
        "contact_jaws": _contact_jaws(hits),
        "n_contact": len(hits),
    }


def _physical_descend_lin(sim: Sim, target, duration: float, n_wp: int, lock_joints=()) -> dict:
    """Straight Cartesian descend. Used only when a neighbor blocks the carry corridor."""
    tid = _tid(sim)
    q0, _ = sim.object_pose(tid)
    target_ee = np.asarray(target, dtype=np.float64)
    start = sim.qpos()
    ik_ok = solve_ik_down(sim, target, lock_joints=lock_joints)
    if not ik_ok:
        _store_ik_fail(sim, target)
    sim.set_qpos(start, hold=True)
    if not ik_ok:
        pos, _ = sim.site_pose(EE_SITE)
        err = float(np.linalg.norm(pos - target_ee)) * 1000
        return {
            "ok": False,
            "ik_solved": False,
            "first_contact_step": None,
            "first_contact_jaw": None,
            "peak_force": {n: 0.0 for n in JOINT_NAMES},
            "samples": [],
            "ee_requested": _xyz(target_ee),
            "ee_actual": _xyz(pos),
            "ee_err_mm": err,
            "ik_residual_mm": err,
            "joint_limit_margin_rad": joint_limit_margin(sim),
        }
    move_lin(sim, target, duration=duration, physics=True, n_wp=n_wp, lock_joints=lock_joints, down=True)
    pos, _ = sim.site_pose(EE_SITE)
    qpos, qquat = sim.object_pose(tid)
    err_mm = float(np.linalg.norm(pos - target_ee)) * 1000
    hits = jaw_contacts(sim, tid)
    report = {
        "ok": err_mm < 20.0,
        "tilt_deg": gripper_tilt_deg(sim),
        "ik_solved": True,
        "first_contact_step": None,
        "first_contact_jaw": _contact_jaws(hits) if hits else None,
        "peak_force": {n: 0.0 for n in JOINT_NAMES},
        "samples": [],
        "ee_requested": _xyz(target_ee),
        "ee_actual": _xyz(pos),
        "ee_err_mm": err_mm,
        "ik_residual_mm": err_mm,
        "approach_reached": err_mm < 8.0,
        "joint_limit_margin_rad": joint_limit_margin(sim),
        "ik_jaw": jaw_snapshot(sim, tid),
        "queen_xy_mm": float(np.linalg.norm(qpos[:2] - q0[:2])) * 1000,
        "queen_tilt_deg": queen_tilt_deg(qquat),
        "pre_close_contacts": len(hits),
        "pre_close_gap_mm": min_jaw_gap(sim, tid) * 1000,
        "jaw_before_close": jaw_snapshot(sim, tid),
        "saturated": [],
    }
    return report


def physical_descend(sim: Sim, target, duration: float = 1.2, n_wp: int = 1, lock_joints=()) -> dict:
    """Joint-space interpolation with actuator commands only (no qpos overwrite).

    n_wp>1 follows a Cartesian line so a neighboring piece beside the grasp
    column is not swept by a joint-space arc. Unobstructed pickup keeps n_wp=1.
    """
    if int(n_wp) > 1:
        return _physical_descend_lin(sim, target, duration, int(n_wp), lock_joints=lock_joints)
    quiet = getattr(sim, "quiet", False)
    tid = _tid(sim)
    start = sim.qpos()
    q0, _ = sim.object_pose(tid)
    target_ee = np.asarray(target, dtype=np.float64)
    report = {
        "ok": False,
        "ik_solved": False,
        "first_contact_step": None,
        "first_contact_jaw": None,
        "peak_force": {n: 0.0 for n in JOINT_NAMES},
        "samples": [],
        "ee_requested": _xyz(target_ee),
    }
    if not solve_ik_down(sim, target, lock_joints=lock_joints):
        pos, _ = sim.site_pose(EE_SITE)
        report["ee_actual"] = _xyz(pos)
        report["ee_err_mm"] = float(np.linalg.norm(pos - target_ee)) * 1000
        report["ik_residual_mm"] = report["ee_err_mm"]
        report["joint_limit_margin_rad"] = joint_limit_margin(sim)
        sim.set_qpos(start, hold=True)
        return report
    goal = sim.qpos()
    report["ik_solved"] = True
    report["tilt_deg"] = gripper_tilt_deg(sim)
    pos, _ = sim.site_pose(EE_SITE)
    report["ik_residual_mm"] = float(np.linalg.norm(pos - target_ee)) * 1000
    report["joint_limit_margin_rad"] = joint_limit_margin(sim)
    report["ik_jaw"] = jaw_snapshot(sim, tid)
    sim.set_qpos(start, hold=True)
    n = max(30, int(duration / sim.dt))
    peak = np.zeros(len(JOINT_NAMES))
    for i in range(n):
        u = (i + 1) / n
        u = u * u * (3 - 2 * u)
        cmd = {k: start[k] + (goal[k] - start[k]) * u for k in start}
        sim.set_ctrl(cmd)
        sim.step()
        force = np.asarray(sim.data.actuator_force, dtype=float)
        np.maximum(peak, np.abs(force), out=peak)
        hits = jaw_contacts(sim, tid)
        if hits and report["first_contact_step"] is None:
            report["first_contact_step"] = i
            report["first_contact_jaw"] = _contact_jaws(hits)
        if i % 40 == 0 or i == n - 1 or hits:
            pos, _ = sim.site_pose(EE_SITE)
            qpos, qquat = sim.object_pose(tid)
            actual = sim.qpos()
            sample = {
                "t": round((i + 1) * sim.dt, 3),
                "ee_target": _xyz(target_ee),
                "ee_actual": _xyz(pos),
                "ee_err_mm": round(float(np.linalg.norm(pos - target_ee)) * 1000, 2),
                "joint_err": {k: round(actual[k] - cmd[k], 4) for k in ARM_JOINTS},
                "force": {n: round(float(force[j]), 3) for j, n in enumerate(JOINT_NAMES)},
                "jaw_gap_mm": round(min_jaw_gap(sim, tid) * 1000, 2),
                "n_jaw_contact": len(hits),
                "queen_xy_disp_mm": round(float(np.linalg.norm(qpos[:2] - q0[:2])) * 1000, 2),
                "queen_tilt_deg": round(queen_tilt_deg(qquat), 2),
            }
            report["samples"].append(sample)
            if not quiet:
                fmax = max(abs(v) for v in sample["force"].values())
                print(
                    f"  descend t={sample['t']:.2f}s ee_err={sample['ee_err_mm']:.1f}mm "
                    f"jaw_gap={sample['jaw_gap_mm']:.2f}mm ncon={sample['n_jaw_contact']} "
                    f"q_xy={sample['queen_xy_disp_mm']:.2f}mm fmax={fmax:.2f}"
                )
    sim.hold(goal)
    for _ in range(20):
        sim.step()
        force = np.abs(np.asarray(sim.data.actuator_force, dtype=float))
        np.maximum(peak, force, out=peak)
        if jaw_contacts(sim, tid) and report["first_contact_step"] is None:
            report["first_contact_step"] = n
            report["first_contact_jaw"] = _contact_jaws(jaw_contacts(sim, tid))
    pos, _ = sim.site_pose(EE_SITE)
    qpos, qquat = sim.object_pose(tid)
    report["ok"] = float(np.linalg.norm(pos - target_ee)) < 0.02
    report["ee_actual"] = _xyz(pos)
    report["ee_err_mm"] = float(np.linalg.norm(pos - target_ee)) * 1000
    report["approach_reached"] = report["ee_err_mm"] < 8.0
    report["peak_force"] = {n: float(peak[j]) for j, n in enumerate(JOINT_NAMES)}
    report["queen_xy_mm"] = float(np.linalg.norm(qpos[:2] - q0[:2])) * 1000
    report["queen_tilt_deg"] = queen_tilt_deg(qquat)
    report["pre_close_contacts"] = len(jaw_contacts(sim, tid))
    report["pre_close_gap_mm"] = min_jaw_gap(sim, tid) * 1000
    report["jaw_before_close"] = jaw_snapshot(sim, tid)
    report["saturated"] = [n for n, v in report["peak_force"].items() if v >= 2.94 - 1e-6]
    if not quiet:
        print(
            f"  descend done ee_err={report['ee_err_mm']:.1f}mm "
            f"pre_close_ncon={report['pre_close_contacts']} gap={report['pre_close_gap_mm']:.2f}mm "
            f"queen_xy={report['queen_xy_mm']:.2f}mm sat={report['saturated']}"
        )
    return report


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


def open_gripper(sim: Sim, opened: float, steps: int = 80):
    close_gripper(sim, opened, steps=steps)


_PAD_CLEAR_M = 0.01
_PAD_CLEAR_MAX_M = 0.02
_PAD_CLEAR_GAP = 0.004


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


def _slide_pad_off_piece(sim: Sim, tid: str, lock_joints: tuple[str, ...]):
    """XY-slide the fixed pad off the piece until jaws are clear, before any +Z."""
    last = None
    travelled = 0.0
    while travelled < _PAD_CLEAR_MAX_M - 1e-6:
        if (not grasp_acquired(sim, tid)) and float(min_jaw_gap(sim, tid)) > _PAD_CLEAR_GAP:
            break
        q, _ = sim.object_pose(tid)
        slide = _fixed_pad_clear(sim, q, dist=_PAD_CLEAR_M)
        if slide is None:
            break
        if not move_lin(sim, slide, duration=0.35, physics=True, n_wp=4, lock_joints=lock_joints):
            break
        settle_queen(sim, steps=12)
        last = slide
        travelled += _PAD_CLEAR_M
    return last, travelled


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


def _retry_missed_grasp(
    sim: Sim,
    cfg: dict,
    use_vision: bool,
    sc: dict,
    failed: dict,
    tid: str,
    surface: float,
    release_open: float,
) -> dict | None:
    """Re-run pickup from the current pose after a miss/drop, if the piece is still up."""
    attempt = int(sc.get("_grasp_attempt", 0))
    budget = int(pickup_cfg(cfg).get("grasp_retries", 1))
    if attempt >= budget:
        return None
    if not _piece_still_upright(sim, tid, surface, tilt_max=1.0):
        return None
    _backoff_grasp(sim, cfg, tid, release_open)
    if not _piece_still_upright(sim, tid, surface, tilt_max=0.5):
        return None
    log_state(
        sim,
        "retry_grasp",
        extra={"attempt": attempt + 2, "after": failed.get("failure_reason")},
    )
    nxt = dict(sc)
    nxt["place_target"] = False
    nxt["seed_home"] = False
    nxt["reset_board"] = False
    nxt["_grasp_attempt"] = attempt + 1
    out = execute_move(sim, cfg, use_vision=use_vision, spawn_xy=None, scenario=nxt)
    out["grasp_retry"] = True
    out["grasp_attempts"] = attempt + 2
    out["first_grasp_reason"] = failed.get("failure_reason")
    log = list(failed.get("grasp_retry_log") or [])
    log.append({"attempt": attempt + 1, "reason": failed.get("failure_reason")})
    out["grasp_retry_log"] = log + list(out.get("grasp_retry_log") or [])
    return out


def _piece_com(sim: Sim, tid: str) -> np.ndarray:
    bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, tid)
    return np.asarray(sim.data.subtree_com[bid], dtype=float).copy()


def _fallen_stem_xy(sim: Sim, tid: str) -> np.ndarray | None:
    """World-XY of the piece's body +Z if it is lying on its side."""
    bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, tid)
    axis = np.asarray(sim.data.xmat[bid].reshape(3, 3)[:, 2], dtype=float)
    v = axis[:2]
    n = float(np.linalg.norm(v))
    if n < 0.25:
        return None
    return v / n


_SCOOP_DIP = 0.70
_RECOVER_OPEN = 0.40
_RECOVER_CLOSED = -0.10
_JAW_Z_GEOMS = (
    "moving_jaw_sph_tip1",
    "moving_jaw_box3",
    "fixed_jaw_sph_tip1",
    "fixed_jaw_box5",
)


def _scoop_axis(stem_xy, sign: float, dip: float = _SCOOP_DIP) -> np.ndarray:
    a = np.array([float(stem_xy[0]) * sign, float(stem_xy[1]) * sign, -float(dip)])
    return a / (float(np.linalg.norm(a)) + 1e-12)


def _jaw_open_dir(sim: Sim) -> np.ndarray:
    fix = sim.geom_xpos("fixed_jaw_sph_tip1")
    mov = sim.geom_xpos("moving_jaw_sph_tip1")
    od = mov - fix
    return od / (float(np.linalg.norm(od)) + 1e-12)


def _jaw_zmin(sim: Sim) -> float:
    zmin = 1.0
    for name in _JAW_Z_GEOMS:
        gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            zmin = min(zmin, float(sim.data.geom_xpos[gid][2]))
    return float(zmin)


def _choose_scoop_axis(sim: Sim, com, stem_xy) -> np.ndarray:
    """Approach along ±stem (with a downward dip) — pick the reachable end."""
    snap = sim.qpos()
    best = None
    for sign in (1.0, -1.0):
        axis = _scoop_axis(stem_xy, sign)
        ok = solve_ik(sim, com, x_axis=axis, ori_weight=0.5, ori_steps=12)
        gpos, grot = sim.site_pose(EE_SITE)
        align = float(np.dot(grot[:, 0], axis))
        err = float(np.linalg.norm(gpos - com))
        score = align - 8.0 * err
        if ok and (best is None or score > best[0]):
            best = (score, axis.copy())
        sim.set_qpos(snap, hold=True)
    return best[1] if best is not None else _scoop_axis(stem_xy, 1.0)


_PINCH_DROP = 0.005
_WRIST_HOLD = ("wrist_flex", "wrist_roll")
_IK_HOLD_ERR = 0.012


def _pinch_xyz(sim: Sim, tid: str, surface: float) -> np.ndarray:
    """Stem-height target a few mm below COM so the pads wrap, not sit on top."""
    com = _piece_com(sim, tid)
    z = max(float(surface) + 0.005, float(com[2]) - _PINCH_DROP)
    return np.array([float(com[0]), float(com[1]), z], dtype=float)


def _roll_horizontal_pinch(sim: Sim, com, axis) -> float:
    """wrist_roll so jaws open sideways and wrap the stem above the board."""
    start = sim.qpos()
    lo, hi = sim.joint_limits()["wrist_roll"]
    best = float(start["wrist_roll"])
    best_score = -1e9
    for roll in np.linspace(float(lo), float(hi), 25):
        q = dict(start)
        q["wrist_roll"] = float(roll)
        sim.set_qpos(q, hold=True)
        if not solve_ik(
            sim, com, x_axis=axis, ori_weight=0.5, ori_steps=10, lock_joints=("wrist_roll",)
        ):
            continue
        od = _jaw_open_dir(sim)
        zmin = _jaw_zmin(sim)
        _, grot = sim.site_pose(EE_SITE)
        align = float(np.dot(grot[:, 0], axis))
        if zmin < 0.022 or zmin > 0.032 or align < 0.75:
            continue
        score = -abs(float(od[2])) - 2.0 * abs(zmin - 0.025) + 0.1 * align
        if score > best_score:
            best_score = score
            best = float(roll)
    sim.set_qpos(start, hold=True)
    return best


def _close_hold(sim: Sim, closed: float, steps: int = 220, hold=None) -> None:
    """Close the gripper while commanding a frozen arm pose."""
    hold = dict(hold or sim.qpos())
    start_g = float(hold.get("gripper", sim.qpos()["gripper"]))
    for i in range(max(1, int(steps))):
        u = (i + 1) / steps
        cmd = {n: hold[n] for n in ARM_JOINTS}
        cmd["gripper"] = start_g + (float(closed) - start_g) * u
        sim.set_ctrl(cmd)
        sim.step()
    sim.hold({**{n: hold[n] for n in ARM_JOINTS}, "gripper": float(closed)})
    sim.step(12)


def _ik_goal(
    sim: Sim,
    target,
    lock: tuple[str, ...] = _WRIST_HOLD,
    max_err: float = _IK_HOLD_ERR,
    **ik_kwargs,
) -> dict[str, float] | None:
    """IK solution that stays on the locked-wrist family, or None."""
    snap = sim.qpos()
    tgt = np.asarray(target, dtype=float)
    ok = bool(solve_ik(sim, tgt, lock_joints=lock, **ik_kwargs))
    p, _ = sim.site_pose(EE_SITE)
    err = float(np.linalg.norm(p - tgt))
    too_low = float(p[2]) < 0.040 and float(tgt[2]) >= 0.040
    goal = {n: sim.qpos()[n] for n in ARM_JOINTS} if ok and err <= max_err and not too_low else None
    sim.set_qpos(snap, hold=True)
    return goal


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


def _move_hold(
    sim: Sim,
    target,
    closed: float,
    duration: float = 0.8,
    n_wp: int = 6,
    lock: tuple[str, ...] = _WRIST_HOLD,
    require_grasp: bool = True,
    **ik_kwargs,
) -> bool:
    """Cartesian move that keeps the gripper shut and holds the IK goal."""
    start, _ = sim.site_pose(EE_SITE)
    goal = np.asarray(target, dtype=float)
    n_wp = max(2, int(n_wp))
    dt = float(duration) / n_wp
    for i in range(n_wp):
        u = (i + 1) / n_wp
        wp = start + (goal - start) * u
        gq = _ik_goal(sim, wp, lock=lock, **ik_kwargs)
        if gq is None:
            return False
        _move_q_hold(sim, gq, closed, duration=dt)
        if require_grasp and not grasp_acquired(sim):
            return False
    return True


def _js_extract(sim: Sim, tid: str, hold: dict, closed: float, surface: float) -> dict | None:
    """Raise the scoop wrist in joint space so the wrap lifts the piece off the board."""
    lo, hi = sim.joint_limits()["shoulder_lift"]
    pose = {n: hold[n] for n in ARM_JOINTS}
    for _ in range(6):
        nxt = dict(pose)
        nxt["shoulder_lift"] = float(np.clip(pose["shoulder_lift"] - 0.02, lo, hi))
        if abs(nxt["shoulder_lift"] - pose["shoulder_lift"]) < 1e-4:
            break
        _move_q_hold(sim, nxt, closed, duration=0.55)
        pose = nxt
        qpos, _ = sim.object_pose(tid)
        board = any(h["other"] == "board" and h["force"] > 0.05 for h in sim.contacts_with(tid))
        if not grasp_acquired(sim, tid):
            return None
        ee, _ = sim.site_pose(EE_SITE)
        if float(qpos[2]) > surface + 0.012 and not board and float(ee[2]) >= 0.040:
            return pose
    qpos, _ = sim.object_pose(tid)
    board = any(h["other"] == "board" and h["force"] > 0.05 for h in sim.contacts_with(tid))
    ee, _ = sim.site_pose(EE_SITE)
    if grasp_acquired(sim, tid) and float(qpos[2]) > surface + 0.012 and not board:
        return pose
    return None


def _nest_step_candidates(ee, nest, z: float) -> list[np.ndarray]:
    cur = np.asarray(ee, dtype=float)
    nest = np.asarray(nest, dtype=float)
    d = nest[:2] - cur[:2]
    dist = float(np.linalg.norm(d))
    step = min(0.012, dist) if dist > 1e-9 else 0.0
    out: list[np.ndarray] = []
    if dist > 1e-9:
        nrm = d / dist
        out.append(np.array([cur[0] + nrm[0] * step, cur[1] + nrm[1] * step, z]))
        for zz in (z - 0.012, z + 0.012):
            out.append(np.array([cur[0] + nrm[0] * step, cur[1] + nrm[1] * step, zz]))
    for r in (0.012, 0.020):
        for ang in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
            out.append(np.array([cur[0] + r * np.cos(ang), cur[1] + r * np.sin(ang), z]))
    floor_z = 0.050
    kept = []
    for p in out:
        q = np.asarray(p, dtype=float).copy()
        if q[2] < floor_z:
            q[2] = floor_z
        kept.append(q)
    return kept


def _piece_nest_dist(sim: Sim, tid: str, nest) -> float:
    qpos, _ = sim.object_pose(tid)
    return float(np.hypot(qpos[0] - nest[0], qpos[1] - nest[1]))


def _best_carry_goal(sim: Sim, tid: str, nest, z: float) -> np.ndarray | None:
    """Locked-wrist step that puts the piece closer to the well."""
    cur, _ = sim.site_pose(EE_SITE)
    qpos, _ = sim.object_pose(tid)
    off = np.asarray(qpos[:2], dtype=float) - cur[:2]
    nest = np.asarray(nest, dtype=float)
    want_ee = np.array([nest[0] - off[0], nest[1] - off[1], z], dtype=float)
    cur_d = _piece_nest_dist(sim, tid, nest)
    best = None
    best_d = cur_d
    cands = _nest_step_candidates(cur, want_ee, z)
    for tgt in cands:
        if float(np.linalg.norm(tgt[:2] - cur[:2])) > 0.018:
            continue
        if _ik_goal(sim, tgt) is None:
            continue
        pred = tgt[:2] + off
        d = float(np.hypot(pred[0] - nest[0], pred[1] - nest[1]))
        if d < best_d - 0.003:
            best_d = d
            best = tgt
    return best


def recover_to_nest(sim: Sim, cfg: dict, tid: str) -> dict:
    """Approach along the lying stem, pinch across it, drop into the well."""
    p = pickup_cfg(cfg)
    surface = float(cfg.get("sim", {}).get("board", {}).get("surface_z", 0.02))
    release_open = float(p.get("gripper_release", 0.30))
    rec_open = min(float(p.get("gripper_open", 0.60)), _RECOVER_OPEN)
    geom = dict(PIECES.get(body_kind(tid) or tid) or {})
    closed_g = max(float(geom.get("gripper_closed", _RECOVER_CLOSED)), _RECOVER_CLOSED)
    nest = nest_xyz(cfg)
    out = {"ok": False, "reason": None, "nest_xyz": list(nest)}
    if _piece_still_upright(sim, tid, surface, tilt_max=0.5):
        out["ok"] = True
        out["reason"] = "already_upright"
        return out
    qpos, _ = sim.object_pose(tid)
    if float(qpos[2]) < surface - 0.02:
        out["reason"] = "RECOVER_OFF_TABLE"
        return out
    stem = _fallen_stem_xy(sim, tid)
    if stem is None:
        out["reason"] = "RECOVER_NO_STEM"
        return out

    _backoff_grasp(sim, cfg, tid, release_open)
    open_gripper(sim, rec_open, steps=40)
    settle_queen(sim, steps=16)
    com = _piece_com(sim, tid)
    pinch = _pinch_xyz(sim, tid, surface)
    axis = _choose_scoop_axis(sim, pinch, stem)
    roll = _roll_horizontal_pinch(sim, pinch, axis)
    out["scoop"] = {"axis": [round(float(v), 3) for v in axis], "roll": round(float(roll), 3)}

    qroll = sim.qpos()
    qroll["wrist_roll"] = float(roll)
    qroll["gripper"] = rec_open
    move_qpos(sim, qroll, duration=0.35, physics=True)
    axy = axis[:2] / (float(np.linalg.norm(axis[:2])) + 1e-12)
    high = np.array([float(pinch[0]) - float(axy[0]) * 0.06, float(pinch[1]) - float(axy[1]) * 0.06, 0.11])
    ikw = dict(x_axis=axis, ori_weight=0.45, ori_steps=8)
    lock_r = ("wrist_roll",)
    if not move_lin(sim, high, duration=0.9, physics=True, n_wp=6, lock_joints=lock_r, **ikw):
        out["reason"] = "RECOVER_IK"
        log_state(sim, "fail_recover_pregrasp", extra={"target": _xyz(high)})
        return out
    log_state(sim, "recover_pregrasp", extra={"target": _xyz(high), "pinch": _xyz(pinch)})
    if not move_lin(
        sim, pinch, duration=1.1, physics=True, n_wp=10, lock_joints=lock_r, **ikw
    ):
        out["reason"] = "RECOVER_DESCEND"
        log_state(sim, "fail_recover_descend")
        return out
    log_state(sim, "recover_grasp", extra={"pinch": _xyz(pinch), "com": _xyz(com)})
    scoop_q = {n: sim.qpos()[n] for n in ARM_JOINTS}
    _close_hold(sim, closed_g, steps=int(p.get("close_steps", 220)), hold=scoop_q)
    log_state(sim, "recover_closed")
    if not grasp_acquired(sim, tid):
        out["reason"] = "RECOVER_GRASP"
        log_state(sim, "fail_recover_grasp")
        return out
    hold = _js_extract(sim, tid, scoop_q, closed_g, surface)
    if hold is None:
        out["reason"] = "RECOVER_DROP"
        log_state(sim, "fail_recover_drop")
        return out
    qpos, _ = sim.object_pose(tid)
    log_state(sim, "recover_extract", extra={"xyz": _xyz(qpos)})

    gpos, _ = sim.site_pose(EE_SITE)
    z_carry = max(float(gpos[2]), 0.055)
    if float(gpos[2]) < z_carry - 0.004:
        up = [float(gpos[0]), float(gpos[1]), z_carry]
        if not _move_hold(sim, up, closed_g, duration=0.7, n_wp=5, lock=_WRIST_HOLD):
            out["reason"] = "RECOVER_DROP" if not grasp_acquired(sim, tid) else "RECOVER_LIFT"
            log_state(sim, "fail_recover_drop")
            return out
    for _ in range(40):
        if not grasp_acquired(sim, tid):
            out["reason"] = "RECOVER_DROP"
            log_state(sim, "fail_recover_transit")
            return out
        if _piece_nest_dist(sim, tid, nest) <= 0.018:
            break
        nxt = _best_carry_goal(sim, tid, nest, z_carry)
        if nxt is None:
            break
        if not _move_hold(sim, nxt, closed_g, duration=0.45, n_wp=3, lock=_WRIST_HOLD):
            out["reason"] = "RECOVER_DROP" if not grasp_acquired(sim, tid) else "RECOVER_TRANSIT"
            log_state(sim, "fail_recover_transit", extra={"target": _xyz(nxt)})
            return out
        qpos, _ = sim.object_pose(tid)
        if float(qpos[2]) < surface + 0.008:
            out["reason"] = "RECOVER_DROP"
            log_state(sim, "fail_recover_drop")
            return out
    dist = _piece_nest_dist(sim, tid, nest)
    if dist > 0.030:
        out["reason"] = "RECOVER_TRANSIT"
        log_state(sim, "fail_recover_transit", extra={"piece_xy_mm": round(dist * 1000.0, 1)})
        return out
    ee, _ = sim.site_pose(EE_SITE)
    stand = np.array([float(ee[0]), float(ee[1]), max(float(ee[2]), 0.055)])
    gq = _ik_goal(
        sim,
        stand,
        lock=("wrist_roll",),
        x_axis=np.array([0.15, 0.0, 0.99]),
        ori_weight=0.55,
        ori_steps=10,
        max_err=0.02,
    )
    if gq is not None:
        _move_q_hold(sim, gq, closed_g, duration=1.2)
    if grasp_acquired(sim, tid):
        ee, _ = sim.site_pose(EE_SITE)
        seat = [float(ee[0]), float(ee[1]), float(nest[2]) + 0.028]
        _move_hold(sim, seat, closed_g, duration=0.8, n_wp=6, lock=_WRIST_HOLD)
    log_state(sim, "recover_insert", extra={"target": _xyz(nest), "piece_xy_mm": round(_piece_nest_dist(sim, tid, nest) * 1000.0, 1)})
    arm = {n: sim.qpos()[n] for n in ARM_JOINTS}
    g0 = float(sim.qpos()["gripper"])
    for i in range(110):
        u = (i + 1) / 110
        cmd = dict(arm)
        cmd["gripper"] = g0 + (release_open - g0) * u
        sim.set_ctrl(cmd)
        sim.step()
    settle_queen(sim, steps=220)
    qpos, quat = sim.object_pose(tid)
    out["final_xyz"] = _xyz(qpos)
    out["tilt_deg"] = round(queen_tilt_deg(quat), 1)
    if not queen_upright(quat, tilt_max=0.55):
        out["reason"] = "RECOVER_STILL_DOWN"
        log_state(sim, "fail_recover_down", extra={"tilt_deg": out["tilt_deg"]})
        return out
    out["ok"] = True
    log_state(sim, "recover_upright", extra={"tilt_deg": out["tilt_deg"], "xyz": out["final_xyz"]})
    return out


def _recover_fallen_then_finish(
    sim: Sim,
    cfg: dict,
    use_vision: bool,
    sc: dict,
    failed: dict,
    tid: str,
    surface: float,
    release_open: float,
) -> dict | None:
    if int(sc.get("_recover_attempt", 0)) >= 1:
        return None
    if _piece_still_upright(sim, tid, surface, tilt_max=0.5):
        return None
    qpos, _ = sim.object_pose(tid)
    if float(qpos[2]) < surface - 0.02:
        return None
    log_state(sim, "recover_start", extra={"after": failed.get("failure_reason")})
    rec = recover_to_nest(sim, cfg, tid)
    failed["recover"] = rec
    if not rec.get("ok"):
        failed["failure_reason"] = rec.get("reason") or "RECOVER_FAIL"
        return None
    nxt = dict(sc)
    nxt["place_target"] = False
    nxt["seed_home"] = True
    nxt["reset_board"] = False
    nxt["_from_nest"] = True
    nxt["_recover_attempt"] = int(sc.get("_recover_attempt", 0)) + 1
    nxt["_grasp_attempt"] = 0
    out = execute_move(sim, cfg, use_vision=use_vision, spawn_xy=None, scenario=nxt)
    out["recovered"] = True
    out["recover"] = rec
    out["first_grasp_reason"] = failed.get("failure_reason")
    return out


def _after_pick_fail(
    sim: Sim,
    cfg: dict,
    use_vision: bool,
    sc: dict,
    failed: dict,
    tid: str,
    surface: float,
    release_open: float,
) -> dict | None:
    reason = failed.get("failure_reason")
    if reason in ("GRASP_MISS", "LIFT_DROP"):
        again = _retry_missed_grasp(
            sim, cfg, use_vision, sc, failed, tid, surface, release_open
        )
        if again is not None:
            return again
    return _recover_fallen_then_finish(
        sim, cfg, use_vision, sc, failed, tid, surface, release_open
    )


def grasp_acquired(sim: Sim, object_id: str | None = None) -> bool:
    tid = object_id or _tid(sim)
    hits = sim.contacts_with(tid)
    jaws = {h["other"] for h in hits if h["force"] > 0.02}
    return "gripper" in jaws or "moving_jaw_so101_v1" in jaws


def object_on_board(sim: Sim, surface: float, z_tol: float = 0.012, object_id: str | None = None) -> bool:
    tid = object_id or _tid(sim)
    qpos, _ = sim.object_pose(tid)
    if abs(float(qpos[2]) - surface) > z_tol:
        return False
    return any(h["other"] == "board" for h in sim.contacts_with(tid))


def bowl_contains(xyz, dest_xy, half: float = 0.05) -> bool:
    """True if a piece is sitting in the capture tray (not on a chess square)."""
    dx = abs(float(xyz[0]) - float(dest_xy[0]))
    dy = abs(float(xyz[1]) - float(dest_xy[1]))
    z = float(xyz[2])
    return dx <= half and dy <= half and 0.01 <= z <= 0.09


def in_capture_bowl(sim: Sim, dest_xy, object_id: str | None = None, half: float = 0.05) -> bool:
    tid = object_id or _tid(sim)
    qpos, _ = sim.object_pose(tid)
    return bowl_contains(qpos, dest_xy, half=half)


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


def _geom_name(sim: Sim, gid: int) -> str:
    return mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"geom{gid}"


def _jaw_role(name: str) -> str:
    if "sph_tip" in name:
        return "tip"
    if "part0" in name or "part1" in name:
        return "hinge_mesh"
    if "box1" in name:
        return "palm"
    if "box" in name:
        return "finger"
    return "other"


def release_snapshot(sim: Sim, object_id: str | None = None) -> dict:
    """Jaw/piece contact at the current release pose. Evaluation only."""
    tid = object_id or _tid(sim)
    qpos, qquat = sim.object_pose(tid)
    gpos, _ = sim.site_pose(EE_SITE)
    bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, tid)
    hits = []
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        b1 = int(sim.model.geom_bodyid[c.geom1])
        b2 = int(sim.model.geom_bodyid[c.geom2])
        if bid not in (b1, b2):
            continue
        other = b2 if b1 == bid else b1
        other_name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, other)
        if other_name not in JAW_BODIES:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(sim.model, sim.data, i, force)
        g_other = c.geom2 if b1 == bid else c.geom1
        g_piece = c.geom1 if b1 == bid else c.geom2
        gname = _geom_name(sim, g_other)
        n = np.asarray(c.frame[:3], dtype=float)
        hits.append(
            {
                "other": other_name,
                "geom": gname,
                "piece_geom": _geom_name(sim, g_piece),
                "role": _jaw_role(gname),
                "dist_mm": float(c.dist) * 1000,
                "force": float(np.linalg.norm(force[:3])),
                "normal_z": float(n[2]),
                "normal": [round(float(v), 3) for v in n],
                "friction": [round(float(v), 4) for v in sim.model.geom_friction[g_other]],
            }
        )
    T = gripper_object_transform(sim, tid)
    roles = sorted({h["role"] for h in hits if h["force"] > 0.02})
    return {
        "jaw_gap_mm": float(sim.jaw_gap() * 1000),
        "min_jaw_gap_mm": float(min_jaw_gap(sim, tid) * 1000),
        "gripper_qpos": float(sim.qpos()["gripper"]),
        "piece_z": float(qpos[2]),
        "ee_z": float(gpos[2]),
        "hang_mm": float((gpos[2] - qpos[2]) * 1000),
        "p_ee_mm": [round(float(v), 2) for v in T["p_ee_mm"]],
        "tilt_deg": float(T["tilt_deg"]),
        "held": grasp_acquired(sim, tid),
        "n_jaw_contact": len([h for h in hits if h["force"] > 0.02]),
        "roles": roles,
        "contacts": hits[:8],
    }


def _xyz(v) -> list[float]:
    return [float(v[0]), float(v[1]), float(v[2])]


def find_observe_pose(sim: Sim, cfg: dict, target_xyz) -> list[float] | None:
    """Reachable wrist pose that still sees target_xyz. Not necessarily overhead.

    Tries the default hover first, then a few heights and outward XY shifts
    along the base→target ray. Restores qpos.
    """
    p = pickup_cfg(cfg)
    board = cfg["sim"]["board"]
    surface = float(board["surface_z"])
    size = float(board["square_size"])
    observe_z = float(p.get("observe_z", 0.12))
    target = np.asarray(target_xyz, dtype=float)
    look_z = (observe_z, 0.08)
    start = sim.qpos()
    try:
        for extra in (0.0, 0.5 * size, 1.0 * size, 1.5 * size):
            xy = _outward_xy(target[:2], cfg, extra)
            for z_off in look_z:
                cand = [float(xy[0]), float(xy[1]), surface + z_off]
                _reset_ik_seed(sim, cfg)
                if not solve_ik_upright(sim, cand):
                    continue
                if wrist_sees_point(sim, target):
                    return cand
        return None
    finally:
        sim.set_qpos(start, hold=True)


def find_pregrasp_pose(sim: Sim, cfg: dict, grasp_xy) -> list[float] | None:
    """Reachable pregrasp that can still IK to the given grasp XY.

    Prefers the default hover height; if that is inside the inner workspace
    hole, steps outward, then lowers. Restores qpos.
    """
    p = pickup_cfg(cfg)
    board = cfg["sim"]["board"]
    surface = float(board["surface_z"])
    size = float(board["square_size"])
    pre_z = float(p.get("pregrasp_z", 0.10))
    grasp_z = float(p.get("grasp_z", 0.015))
    gx, gy = float(grasp_xy[0]), float(grasp_xy[1])
    grasp = [gx, gy, surface + grasp_z]
    cands: list[list[float]] = []
    for extra in (0.0, 0.5 * size, 1.0 * size, 1.5 * size):
        xy = _outward_xy([gx, gy], cfg, extra)
        cands.append([float(xy[0]), float(xy[1]), surface + pre_z])
    for z_off in (0.08, 0.06, 0.04, 0.03):
        cands.append([gx, gy, surface + z_off])
    start = sim.qpos()
    try:
        for cand in cands:
            _reset_ik_seed(sim, cfg)
            if not solve_ik_upright(sim, cand):
                continue
            if solve_ik_upright(sim, grasp):
                return cand
        return None
    finally:
        sim.set_qpos(start, hold=True)


def locate_queen(sim: Sim, cfg: dict, result: dict, square: str) -> np.ndarray | None:
    """RGB-D estimate from the wrist sensor. Ground truth is logged for evaluation only."""
    tid = _tid(sim)
    obs = capture_wrist(sim)
    est = estimate_target_pose(obs, cfg, square)
    gt, _ = sim.object_pose(tid)
    if est is None:
        result["estimated_position_error_mm"] = None
        return None
    err = float(np.linalg.norm(est[:2] - gt[:2]))
    result["estimated_position_error_mm"] = err * 1000.0
    result["queen_estimated_xyz"] = est[:3].tolist()
    if not getattr(sim, "quiet", False):
        print(f"  estimated_xyz: {np.round(est[:3], 4).tolist()}")
        print(f"  gt_xyz:        {np.round(gt, 4).tolist()}")
        print(f"  position_error_mm: {err * 1000:.1f}")
    return est[:3]


def execute_move(sim: Sim, cfg: dict, use_vision: bool = False, spawn_xy=None, scenario: dict | None = None) -> dict:
    sc = dict(scenario or {})
    p = pickup_cfg(cfg)
    tid = sc.get("target_id", "queen")
    sim.target_id = tid
    sim.quiet = bool(sc.get("quiet", False) or getattr(sim, "quiet", False))
    source = sc.get("source", p.get("source", "d1"))
    dest = sc.get("dest", p.get("dest", "d4"))
    perception = sc.get("perception", "vision" if use_vision else "gt")
    use_vision = perception == "vision"
    offset_xy = [float(v) for v in sc.get("offset_xy", [0.0, 0.0])]
    pose_err = [float(v) for v in sc.get("pose_error_xy", [0.0, 0.0])]
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
        grasp_z = max(grasp_z, 0.022)
        lift_z = max(lift_z, 0.06)
    observe_z = float(p.get("observe_z", 0.20))
    place_hover = float(p.get("place_hover", 0.003))
    xy_tol = float(p.get("place_xy_tol", 0.018))
    transport_z = float(sc.get("transport_clearance_z", p.get("transport_clearance_z", 0.08)))
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
        "pose_error_xy_mm": [pose_err[0] * 1000.0, pose_err[1] * 1000.0],
        "yaw_deg": yaw_deg,
        "neighbors": sc.get("neighbors", []),
        "geometry": sc.get("geometry"),
        "neighbor_contact": False,
        "neighbor_displacement": None,
        "grasp_xy_offset_mm": [float(off[0]) * 1000.0, float(off[1]) * 1000.0],
        "ik_solved": False,
        "approach_reached": False,
        "approach_ok": False,
        "transport_path": None,
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

    first_hit = [None]

    def _clutter_names() -> list[str]:
        names = []
        for name in _proxy_body_names(sim):
            bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0 and float(sim.data.xpos[bid][2]) > 0.0:
                names.append(name)
        for i in range(sim.model.nbody):
            name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not name or name == tid or body_kind(name) is None:
                continue
            if float(sim.data.xpos[i][2]) > 0.0:
                names.append(name)
        return names

    # Fast reject for the per-step hook: with 32 pieces resting on the board
    # there are ~160 contacts per step, almost all piece<->board. Only a
    # clutter piece touching a robot part or the target needs the slow path.
    tid_bid = sim.body_id(tid)
    inert_ids = np.array(
        [b for b in (sim.body_id("board"), sim.body_id("floor"), 0) if b >= 0], dtype=int
    )
    piece_ids = np.array(
        [
            i
            for i in range(sim.model.nbody)
            if (sim.body_name(i) or "").startswith("proxy_") or body_kind(sim.body_name(i)) is not None
        ],
        dtype=int,
    )

    def _clutter_hit_possible() -> bool:
        b1, b2 = sim.contact_bodies()
        if b1.size == 0:
            return False
        z = sim.data.xpos[:, 2]
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

    def _track_proxies():
        if first_hit[0] is not None:
            return
        phase = getattr(sim, "_pickup_phase", None)
        if phase in (None, "001_initial"):
            return
        if not _clutter_hit_possible():
            return

        def _part_touching(*bodies: str) -> str | None:
            for b in bodies:
                for h in sim.contacts_with(b):
                    part = robot_part_walk(sim, h.get("other"))
                    if part:
                        return part
            return None

        for name in _clutter_names():
            for h in sim.contacts_with(name):
                other = h.get("other")
                if other in (None, "board", "floor", "world"):
                    continue
                part = robot_part_walk(sim, other)
                if part is None and other != tid:
                    continue
                if part is None:
                    part = _part_touching(name, tid)
                sq = square_of_body(sim, cfg, name)
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

    def _proxy_hook():
        if prev_all:
            prev_all()
        _track_proxies()

    sim.on_step = _proxy_hook

    nb_peak_disp = [0.0]
    nb_peak_tilt = [0.0]
    nb_peak_contact = [False]
    nb_min_clear = [None]
    nb_hook_n = [0]

    def _track_neighbors(*, with_gap: bool = False):
        nids = [n.get("id", "pawn") for n in (sc.get("neighbors") or [])]
        if not nids:
            return
        starts = result.get("_neighbor_start") or {}
        stats = []
        any_contact = nb_peak_contact[0]
        for nid in nids:
            p, q = sim.object_pose(nid)
            p0 = np.asarray(starts.get(nid, p[:2]), dtype=float)
            hits = [h for h in sim.contacts_with(nid) if h["other"] not in ("board", None)]
            robot_hit = any(
                h["other"] in ("gripper", "moving_jaw_so101_v1") or (h["other"] or "").startswith("link")
                for h in hits
            )
            any_contact = any_contact or bool(hits)
            disp = float(np.linalg.norm(p[:2] - p0[:2]) * 1000)
            tilt = queen_tilt_deg(q)
            nb_peak_disp[0] = max(nb_peak_disp[0], disp)
            nb_peak_tilt[0] = max(nb_peak_tilt[0], tilt)
            cl = None
            if with_gap:
                cl = min(
                    min_body_gap(sim, nid, "gripper"),
                    min_body_gap(sim, nid, "moving_jaw_so101_v1"),
                )
                if nb_min_clear[0] is None:
                    nb_min_clear[0] = cl
                else:
                    nb_min_clear[0] = min(nb_min_clear[0], cl)
            stats.append(
                {
                    "id": nid,
                    "contact": bool(hits),
                    "robot_contact": robot_hit,
                    "xy_disp_mm": nb_peak_disp[0],
                    "tilt_deg": nb_peak_tilt[0],
                    "clearance_mm": None if cl is None else float(cl * 1000),
                }
            )
        nb_peak_contact[0] = any_contact
        result["neighbor_eval"] = stats
        result["neighbor_contact"] = nb_peak_contact[0]
        result["neighbor_displacement"] = nb_peak_disp[0]
        result["neighbor_tilt"] = nb_peak_tilt[0]
        if with_gap:
            result["min_robot_neighbor_clearance_mm"] = (
                None if nb_min_clear[0] is None else float(nb_min_clear[0] * 1000)
            )

    def _nb_hook():
        nb_hook_n[0] += 1
        if nb_hook_n[0] % 4 == 0:
            _track_neighbors()

    def _fill_fail_diag():
        pos, _ = sim.site_pose(EE_SITE)
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
                result["joint_limit_margin_rad"] = joint_limit_margin(sim)

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
                    clear = round(min_body_gap(sim, robot, body) * 1000.0, 2)
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
                "square": square_of_body(sim, cfg, blk.get("body") or ""),
                "phase": "approach_plan",
            }
        elif reason in collision_reasons:
            clutter = list(result.get("occupied_neighbors") or [])
            pair = closest_pair(sim, APPROACH_BODIES, clutter) if clutter else None
            if pair:
                result["collision_pair"] = {
                    "robot": pair.get("robot"),
                    "body": pair.get("body"),
                    "dist_mm": pair.get("clearance_mm"),
                    "clearance_mm": pair.get("clearance_mm"),
                    "square": square_of_body(sim, cfg, pair.get("body") or ""),
                    "phase": getattr(sim, "_pickup_phase", None),
                }

    def _done():
        _track_neighbors(with_gap=True)
        _track_proxies()
        if result.get("failure_reason") in ("IK_FAILURE", "IK_NO_SOLUTION") and not result.get("ik_fail_phase"):
            result["ik_fail_phase"] = getattr(sim, "_pickup_phase", None)
        _fill_fail_diag()
        annotate_episode(result)
        return result

    if spawn_xy is not None:
        spawn = list(spawn_xy)
    else:
        base = square_to_world(source, cfg, z=surface)
        spawn = [base[0] + offset_xy[0], base[1] + offset_xy[1], base[2]]
    result["spawn_xyz"] = _xyz(spawn)
    if sc.get("place_target", True):
        sim.place_object(tid, spawn, yaw_quat(yaw_deg))
        neighbor_start = {}
        for nb in sc.get("neighbors") or []:
            nid = nb.get("id", "pawn")
            nsq = nb.get("square")
            npos = square_to_world(nsq, cfg, z=surface) if nsq else list(nb.get("xyz", [0.37, 0.05, -0.5]))
            sim.place_object(nid, npos, yaw_quat(float(nb.get("yaw_deg", 0.0))))
            neighbor_start[nid] = npos[:2]
        result["_neighbor_start"] = neighbor_start
    if sc.get("seed_home", sc.get("reset_board", True)):
        seed_vertical(sim, cfg)
    open_gripper(sim, open_g, steps=40)
    settle_queen(sim)

    actual, _ = sim.object_pose(tid)
    result["actual_start_pose"] = actual.tolist()
    result["queen_start_xyz"] = actual.tolist()
    log_state(sim, "001_initial", extra={"spawn": _xyz(spawn)})

    if use_vision:
        prior = square_to_world(source, cfg, z=surface)
        look = find_observe_pose(sim, cfg, prior)
        if look is None or not _safe_lin(
            sim,
            look,
            cfg,
            ("wrist_flex", "wrist_roll"),
            obstacles=obstacle_bodies(sim, skip={tid}),
            skip={tid},
            duration=0.8,
        ):
            result["failure_reason"] = "IK_NO_SOLUTION"
            result["ee_requested"] = _xyz(look or prior)
            log_state(sim, "fail_observe", extra={"target": result["ee_requested"]})
            return _done()
        log_state(sim, "002_observe", extra={"target": _xyz(look)})
        queen_xyz = locate_queen(sim, cfg, result, source)
        if queen_xyz is None:
            result["queen_detected"] = False
            result["failure_reason"] = "PERCEPTION_FAIL"
            return _done()
        result["queen_detected"] = True
    else:
        queen_xyz, _ = sim.object_pose(tid)
        observe = find_observe_pose(sim, cfg, queen_xyz)
        if observe is None or not _safe_lin(
            sim,
            observe,
            cfg,
            ("wrist_flex", "wrist_roll"),
            obstacles=obstacle_bodies(sim, skip={tid}),
            skip={tid},
            duration=0.8,
        ):
            result["failure_reason"] = "IK_NO_SOLUTION"
            result["ee_requested"] = _xyz(observe or queen_xyz)
            log_state(sim, "fail_observe", extra={"target": result["ee_requested"]})
            return _done()
        log_state(sim, "002_observe", extra={"target": _xyz(observe)})
        queen_xyz, _ = sim.object_pose(tid)
    plan_xy = [float(queen_xyz[0] + pose_err[0]), float(queen_xyz[1] + pose_err[1])]
    obstacles = []
    for nb in sc.get("neighbors") or []:
        nid = nb.get("id", "pawn")
        npos, _ = sim.object_pose(nid)
        if npos[2] > 0.0:
            obstacles.append(npos[:2])
    # Vertical pinch: put the wrist where gripper +X is world-down at the
    # grasp point before anything downstream snapshots the heading. Without
    # this the observe pose's 30-50 deg flex was inherited (approach planner
    # locks flex at its current value) and the fixed jaw tipped pieces over on
    # close. Files a-f reach <3 deg; g/h get the least-tilted pose available.
    vert = _vertical_wrist(sim, [plan_xy[0], plan_xy[1], surface + grasp_z])
    if vert is not None:
        result["vertical_flex"] = round(vert["wrist_flex"], 4)
        result["vertical_tilt_deg"] = round(vert["tilt_deg"], 2)
        qv = sim.qpos()
        qv["wrist_flex"] = vert["wrist_flex"]
        move_qpos(sim, qv, duration=0.3, physics=False)
    grasp_xy = _seeded_grasp_xy(sim, cfg, plan_xy, geom, surface + grasp_z)
    pre = find_pregrasp_pose(sim, cfg, grasp_xy)
    if pre is None:
        result["failure_reason"] = "IK_NO_SOLUTION"
        result["ee_requested"] = _xyz([grasp_xy[0], grasp_xy[1], surface + pre_z])
        log_state(sim, "fail_pregrasp", extra={"target": result["ee_requested"]})
        return _done()
    blocked = carry_blocked(obstacles, pre[:2], dest_xy[:2])
    result["transport_path"] = "clearance" if blocked else "straight"
    clutter = [n for n in obstacle_bodies(sim, skip={tid}) if n != "board"]
    result["occupied_neighbors"] = sorted(clutter)
    grasp_plan = [grasp_xy[0], grasp_xy[1], surface + grasp_z]

    def _grasp_for_roll(roll: float | None) -> list[float]:
        # The pinch axis rotates with wrist roll, so the fixed-pad shift does too.
        xy = _grasp_xy_for_roll(sim, plan_xy, surface + grasp_z, geom, roll)
        return [xy[0], xy[1], surface + grasp_z]

    approach = plan_approach(
        sim,
        pre,
        grasp_plan,
        open_g,
        clutter,
        source=None if sc.get("_from_nest") else source,
        cfg=cfg,
        scenario=sc,
        grasp_for_roll=_grasp_for_roll,
    )
    if approach.get("pre"):
        pre = list(approach["pre"])
    if approach.get("grasp"):
        grasp_plan = list(approach["grasp"])
        grasp_xy = grasp_plan[:2]
    result["approach_plan"] = {
        k: approach.get(k)
        for k in ("ok", "roll", "gripper", "n_wp", "tried", "reachable", "rank", "reason", "pre_offset_xy", "block")
    }
    if not approach["ok"]:
        result["failure_reason"] = (
            "NO_COLLISION_FREE_APPROACH" if approach.get("reachable") else "IK_NO_SOLUTION"
        )
        log_state(sim, "fail_pregrasp", extra={"target": _xyz(pre), "approach": result["approach_plan"]})
        return _done()
    lock: tuple[str, ...] = tuple(approach["lock_joints"])
    # After close, lock wrist_roll so the pinch does not yaw around the stem.
    # Leave wrist_flex free: carry IK pulls gripper +X toward world -Z so the
    # piece stays upright as the shoulder/elbow reach across the board.
    hold: tuple[str, ...] = tuple(
        j for j in dict.fromkeys((*lock, "wrist_roll")) if j != "wrist_flex"
    )
    if approach["roll"] is not None:
        qpre = sim.qpos()
        qpre["wrist_roll"] = float(approach["roll"])
        move_qpos(sim, qpre, duration=0.35, physics=True)
    g_ap = float(approach["gripper"])
    result["approach_gripper"] = None if abs(g_ap - open_g) < 0.02 else g_ap
    if abs(g_ap - float(sim.qpos()["gripper"])) > 0.01:
        qg = sim.qpos()
        qg["gripper"] = g_ap
        sim.set_qpos(qg, hold=True)
        sim.set_ctrl({**sim.qpos(), "gripper": g_ap})
    pickup_clear = bool(clutter) or approach["n_wp"] > 1
    # Vertical pinch: pre, descend and grasp are all solved with gripper +X
    # world-down (solve_ik_down); flex is a free joint the solver uses to keep
    # the approach vertical while shoulder/elbow move. Roll stays locked when
    # the planner chose one for neighbor clearance.
    grasp_lock = tuple(j for j in lock if j != "wrist_flex")
    pre_move = (
        move_lin(sim, pre, duration=1.0, physics=False, n_wp=6, lock_joints=grasp_lock, down=True)
        if pickup_clear
        else move_ik(sim, pre, duration=1.0, lock_joints=grasp_lock, down=True)
    )
    if not pre_move:
        result["failure_reason"] = "IK_NO_SOLUTION"
        log_state(sim, "fail_pregrasp", extra={"target": _xyz(pre)})
        return _done()
    result["pregrasp_reached"] = True
    log_state(sim, "003_pregrasp", extra={"target": _xyz(pre), "approach_gripper": result.get("approach_gripper"), "approach_roll": approach.get("roll")})

    if not queen_upright(sim.object_pose(tid)[1]):
        result["failure_reason"] = "APPROACH_COLLISION"
        log_state(sim, "fail_pregrasp_knock")
        again = _after_pick_fail(
            sim, cfg, use_vision, sc, result, tid, surface, release_open
        )
        return again if again is not None else _done()

    if use_vision:
        refined = locate_queen(sim, cfg, result, source)
        if refined is not None:
            queen_xyz = refined
            plan_xy = [float(queen_xyz[0] + pose_err[0]), float(queen_xyz[1] + pose_err[1])]
            grasp_xy = _seeded_grasp_xy(sim, cfg, plan_xy, geom, surface + grasp_z)

    result["pose_used_for_grasp"] = [float(plan_xy[0]), float(plan_xy[1]), float(np.asarray(queen_xyz)[2])]
    result["perception_error"] = [
        float((plan_xy[0] - actual[0]) * 1000.0),
        float((plan_xy[1] - actual[1]) * 1000.0),
    ]
    grasp = [grasp_xy[0], grasp_xy[1], surface + grasp_z]
    result["ee_requested"] = _xyz(grasp)
    blocked = carry_blocked(obstacles, grasp[:2], dest_xy[:2])
    result["transport_path"] = "clearance" if blocked else "straight"
    desc = physical_descend(
        sim, grasp, duration=1.2, n_wp=8 if pickup_clear else 1, lock_joints=grasp_lock
    )
    result["descend"] = {k: v for k, v in desc.items() if k not in ("samples", "ik_jaw", "jaw_before_close")}
    result["ik_solved"] = bool(desc.get("ik_solved"))
    result["ik_residual_mm"] = desc.get("ik_residual_mm")
    result["ee_actual"] = desc.get("ee_actual")
    result["ee_err_mm"] = desc.get("ee_err_mm")
    result["approach_reached"] = bool(desc.get("approach_reached"))
    result["joint_limit_margin_rad"] = desc.get("joint_limit_margin_rad")
    result["pre_close_contacts"] = desc.get("pre_close_contacts", 0)
    result["pre_close_gap_mm"] = desc.get("pre_close_gap_mm")
    result["first_contact_jaw"] = desc.get("first_contact_jaw")
    result["ik_jaw"] = desc.get("ik_jaw")
    result["jaw_before_close"] = desc.get("jaw_before_close")
    if not desc.get("ik_solved") or not desc["ok"]:
        result["failure_reason"] = _descend_fail_reason(desc)
        log_state(sim, "fail_descend", extra={"target": _xyz(grasp), "ee_err_mm": desc.get("ee_err_mm")})
        return _done()
    log_state(
        sim,
        "004_descend",
        extra={
            "target": _xyz(grasp),
            "ee_err_mm": round(desc["ee_err_mm"], 2),
            "pre_close_contacts": desc["pre_close_contacts"],
            "pre_close_gap_mm": round(desc["pre_close_gap_mm"], 2),
            "queen_xy_mm": round(desc["queen_xy_mm"], 2),
            "saturated": desc["saturated"],
        },
    )

    if not queen_upright(sim.object_pose(tid)[1]):
        result["failure_reason"] = "APPROACH_COLLISION"
        log_state(sim, "fail_descend_knock")
        again = _after_pick_fail(
            sim, cfg, use_vision, sc, result, tid, surface, release_open
        )
        return again if again is not None else _done()

    result["grasp_attempted"] = True
    q_pre, _ = sim.object_pose(tid)
    grip_peak = [0.0]
    prev = sim.on_step

    def _close_step():
        if prev:
            prev()
        grip_peak[0] = max(grip_peak[0], abs(float(sim.data.actuator_force[sim._act["gripper"]])))

    sim.on_step = _close_step
    close_gripper(sim, closed_g, steps=int(p.get("close_steps", 220)))
    sim.on_step = prev
    q_post, qquat_post = sim.object_pose(tid)
    result["close_shove_xy_mm"] = float(np.linalg.norm(q_post[:2] - q_pre[:2])) * 1000
    result["pre_close_queen_disp_mm"] = float(
        np.linalg.norm(q_pre[:2] - np.asarray(result["queen_start_xyz"][:2]))
    ) * 1000
    result["close_tilt_deg"] = queen_tilt_deg(qquat_post)
    result["T_gripper_queen_close"] = gripper_object_transform(sim, tid)
    result["gripper_peak_Nm"] = grip_peak[0]
    result["gripper_saturated"] = grip_peak[0] >= 2.94 - 1e-6
    log_state(
        sim,
        "005_gripper_closed",
        extra={
            "close_shove_xy_mm": round(result["close_shove_xy_mm"], 2),
            "close_tilt_deg": round(result["close_tilt_deg"], 2),
            "T_p_ee_mm": [round(v, 2) for v in result["T_gripper_queen_close"]["p_ee_mm"]],
            "gripper_peak_Nm": round(grip_peak[0], 3),
        },
    )
    result["grasp_acquired"] = grasp_acquired(sim, tid)
    if not result["grasp_acquired"]:
        result["failure_reason"] = "GRASP_MISS"
        again = _after_pick_fail(
            sim, cfg, use_vision, sc, result, tid, surface, release_open
        )
        return again if again is not None else _done()

    g_close, _ = sim.site_pose(EE_SITE)
    pick_xy = np.array([grasp[0], grasp[1]], dtype=np.float64)
    q_gt, _ = sim.object_pose(tid)
    # Held piece XY, not the pre-grasp plan. Close shove would otherwise
    # carry through to dest if we kept object_xy = plan_xy.
    object_xy = np.array(q_gt[:2], dtype=np.float64)
    result["grasp_offset_error_mm"] = float(np.linalg.norm(q_gt[:2] - pick_xy)) * 1000
    result["place_diag"] = {
        "dest_xy_mm": [round(float(dest_xy[0]) * 1000, 1), round(float(dest_xy[1]) * 1000, 1)],
        "plan_xy_mm": [round(float(plan_xy[0]) * 1000, 1), round(float(plan_xy[1]) * 1000, 1)],
        "held_xy_after_close_mm": [round(float(q_gt[0]) * 1000, 1), round(float(q_gt[1]) * 1000, 1)],
        "close_vs_plan_mm": round(float(np.linalg.norm(q_gt[:2] - plan_xy[:2])) * 1000, 1),
        "T_ee_after_close_mm": [round(v, 1) for v in gripper_object_transform(sim, tid)["p_ee_mm"]],
    }

    obstacles = obstacle_bodies(sim, skip={tid})
    mins = {"queen_mm": 1e9, "gripper_mm": 1e9, "camera_mm": 1e9}
    prev_step = sim.on_step

    n_transit = [0]

    def _transit_hook():
        phase = getattr(sim, "_pickup_phase", "") or ""
        n_transit[0] += 1
        if n_transit[0] % 4 == 0 and (
            phase.startswith("007") or phase.startswith("fail_transport")
        ):
            cl = transit_clearances(sim, tid, obstacles)
            for k in mins:
                mins[k] = min(mins[k], cl[k])
        hit = transit_nontarget_hit(sim, tid, obstacles)
        if hit is None:
            return
        if result["transport_first_collision"] is None:
            part = robot_part(hit.get("other"))
            sq = square_of_body(sim, cfg, hit.get("body") or "")
            result["transport_first_collision"] = {
                **hit,
                "phase": phase,
                "robot_part": part,
                "square": sq,
                "piece": STARTING_AT.get(sq) if sq else None,
            }
        if phase.startswith("006") or phase == "fail_lift":
            result["transport_contact_lift"] = True
        elif phase.startswith("007") or phase.startswith("fail_transport"):
            result["transport_contact_transit"] = True
        elif phase.startswith("008") or phase == "fail_lower":
            result["transport_contact_descend"] = True

    def _on_step():
        if prev_step:
            prev_step()
        _nb_hook()
        _transit_hook()

    sim.on_step = _on_step
    populated = any(n != "board" for n in obstacles)
    log_state(sim, "006_lift")
    # Locked-wrist +Z from inside ~190 mm of the base is poorly tracked or
    # unreachable. Push the lift target out to that radius (generic workspace
    # floor, not a file/square table). Outside it, lift is still +Z.
    radial = float(np.linalg.norm(np.asarray(g_close[:2], dtype=float) - _arm_base_xy(cfg)))
    extra = max(0.0, _WORKSPACE_R_M - radial)
    lift_xy = _outward_xy(g_close[:2], cfg, extra) if extra > 0.001 else np.asarray(g_close[:2], dtype=float)
    if sc.get("_from_nest"):
        nest_xy = np.asarray(nest_xyz(cfg)[:2], dtype=float)
        away = np.asarray(g_close[:2], dtype=float) - nest_xy
        n = float(np.linalg.norm(away))
        if n > 1e-6:
            lift_xy = np.asarray(g_close[:2], dtype=float) + away / n * 0.025
    lift = [float(lift_xy[0]), float(lift_xy[1]), float(g_close[2] + lift_z)]
    # Extract +Z with the close wrist so the pinch does not peel. Transit and
    # lower then free wrist_flex to keep the piece world-upright.
    extract = tuple(dict.fromkeys((*hold, "wrist_flex")))
    lin_lift = bool(blocked or populated or extra > 0.001 or sc.get("_from_nest"))

    def _lift(lock: tuple[str, ...], down: bool = False) -> bool:
        if lin_lift or down:
            return move_lin(sim, lift, duration=1.2, physics=True, n_wp=8, lock_joints=lock, down=down)
        return move_ik(sim, lift, duration=1.2, physics=True, lock_joints=lock)

    # Vertical extract: the pinch is a hinge, so the piece hangs plumb; keeping
    # gripper +X world-down while rising is what stops it peeling. A locked
    # flex rotated the gripper up to 40 deg over a 30 mm lift near the base.
    lift_ok = _lift(tuple(hold), down=True)
    if not lift_ok:
        result["lift_fallback"] = "locked_wrist"
        lift_ok = _lift(extract)
    if not lift_ok:
        # With flex+roll locked only shoulder_lift/elbow remain, and near the
        # folded (inner) edge of that 2R annulus a straight +Z is unreachable
        # (rank-7 pawns: 6/32 board failures were exactly this). The pinch
        # already holds the piece; a few degrees of flex beat abandoning it.
        result["lift_fallback"] = "free_wrist"
        lift_ok = _lift(tuple(hold))
    if not lift_ok:
        result["failure_reason"] = "IK_FAILURE"
        log_state(sim, "fail_lift", extra={"target": lift})
        return _done()
    for _ in range(40):
        sim.step()
    log_state(sim, "006_lift")

    qpos, qquat = sim.object_pose(tid)
    object_xy = np.array(qpos[:2], dtype=np.float64)
    result["T_gripper_queen_lift"] = gripper_object_transform(sim, tid)
    result["place_diag"]["held_xy_after_lift_mm"] = [round(float(qpos[0]) * 1000, 1), round(float(qpos[1]) * 1000, 1)]
    result["place_diag"]["T_ee_after_lift_mm"] = [round(v, 1) for v in result["T_gripper_queen_lift"]["p_ee_mm"]]
    clearance = float(p.get("lift_clearance", 0.04))
    held = grasp_acquired(sim, tid)
    lifted = qpos[2] > surface + clearance
    result["max_queen_height_mm"] = float(qpos[2] * 1000)
    result["lift_success"] = bool(lifted and held and queen_upright(qquat, tilt_max=0.7))
    if not result["lift_success"]:
        result["failure_reason"] = "LIFT_DROP" if result["grasp_acquired"] else "GRASP_MISS"
        again = _after_pick_fail(
            sim, cfg, use_vision, sc, result, tid, surface, release_open
        )
        return again if again is not None else _done()
    if sc.get("stop_after") == "lift":
        result["success"] = True
        result["failure_reason"] = None
        return _done()

    gpos, _ = sim.site_pose(EE_SITE)
    # Translate the believed object XY onto dest. Using the EE grasp point here
    # would park the jaws on dest and leave the piece short by grasp_xy_offset.
    delta = dest_xy[:2] - object_xy
    dest_ee = np.array([float(gpos[0] + delta[0]), float(gpos[1] + delta[1])])
    _, standoff = _xy_floor(dest_ee, cfg)
    result["transport_standoff"] = bool(standoff)
    z_ee, derived = safe_transit_ee_z(sim, float(gpos[2]), tid, obstacles)
    result["transit_z"] = z_ee
    result["transit_z_derived"] = derived
    if populated:
        result["transport_path"] = "over_board"
        if z_ee > float(gpos[2]) + 0.004:
            high = [float(gpos[0]), float(gpos[1]), z_ee]
            raised = _carry_lin(sim, high, 0.8, 4, hold, extract)
            if not raised:
                out = _outward_xy(gpos[:2], cfg, 0.05)
                high = [float(out[0]), float(out[1]), z_ee]
                raised = _carry_lin(sim, high, 0.8, 4, hold, extract)
            if not raised:
                result["failure_reason"] = "IK_FAILURE"
                log_state(sim, "fail_transport_raise", extra={"target": _xyz(high)})
                return _done()
            gpos, _ = sim.site_pose(EE_SITE)
            z_ee, derived = safe_transit_ee_z(sim, float(gpos[2]), tid, obstacles)
            result["transit_z"] = float(gpos[2])
            result["transit_z_derived"] = derived
        carry = [float(gpos[0] + delta[0]), float(gpos[1] + delta[1]), float(gpos[2])]
        xy_dist = float(np.linalg.norm(delta))
        n_xy = max(10, int(np.ceil(xy_dist / 0.02)))
        log_state(sim, "007_transport")
        ok, how = _transport_to(sim, carry, hold, cfg, n_xy, max(1.5, 0.4 * n_xy), obstacles=clutter, skip={tid})
        result["transport_detour"] = how
        if not ok:
            result["failure_reason"] = "IK_FAILURE"
            log_state(sim, "fail_transport", extra={"target": _xyz(carry), "detour": how})
            return _done()
    elif blocked:
        result["transport_path"] = "clearance"
        z_safe = max(float(gpos[2]), surface + transport_z)
        if z_safe > float(gpos[2]) + 0.004:
            high = [float(gpos[0]), float(gpos[1]), z_safe]
            if not _carry_lin(sim, high, 0.8, 4, hold, extract):
                result["failure_reason"] = "IK_FAILURE"
                log_state(sim, "fail_transport_raise", extra={"target": _xyz(high)})
                return _done()
            gpos, _ = sim.site_pose(EE_SITE)
        carry = [float(gpos[0] + delta[0]), float(gpos[1] + delta[1]), float(gpos[2])]
        ok, how = _transport_to(sim, carry, hold, cfg, 10, 1.5, obstacles=clutter, skip={tid})
        result["transport_detour"] = how
        if not ok:
            result["failure_reason"] = "IK_FAILURE"
            log_state(sim, "fail_transport", extra={"target": _xyz(carry), "detour": how})
            return _done()
    else:
        result["transport_path"] = "straight"
        carry = [float(gpos[0] + delta[0]), float(gpos[1] + delta[1]), float(gpos[2])]
        ok, how = _transport_to(sim, carry, hold, cfg, 10, 1.5, obstacles=clutter, skip={tid})
        result["transport_detour"] = how
        if not ok:
            result["failure_reason"] = "IK_FAILURE"
            log_state(sim, "fail_transport", extra={"target": _xyz(carry), "detour": how})
            return _done()
    for _ in range(30):
        sim.step()
    qpos, qquat = sim.object_pose(tid)
    held_over = bool(grasp_acquired(sim, tid) and qpos[2] > surface + clearance)
    over_dest = float(np.hypot(qpos[0] - dest_xy[0], qpos[1] - dest_xy[1])) < 0.04
    if result.get("transport_standoff"):
        over_dest = True
    if populated:
        result["transport_success"] = bool(
            held_over and over_dest and not result["transport_contact_transit"]
        )
    else:
        result["transport_success"] = held_over
    result["min_queen_piece_clearance_mm"] = None if mins["queen_mm"] >= 1e8 else float(mins["queen_mm"])
    result["min_gripper_piece_clearance_mm"] = None if mins["gripper_mm"] >= 1e8 else float(mins["gripper_mm"])
    result["min_camera_piece_clearance_mm"] = None if mins["camera_mm"] >= 1e8 else float(mins["camera_mm"])
    log_state(
        sim,
        "007_transport",
        extra={
            "target": _xyz(carry),
            "dest": dest,
            "path": result["transport_path"],
            "detour": result.get("transport_detour"),
            "transit_z": result.get("transit_z"),
        },
    )
    if not result["transport_success"]:
        result["failure_reason"] = "TRANSPORT_COLLISION" if result["transport_contact_transit"] else "TRANSPORT_DROP"
        return _done()

    gpos, _ = sim.site_pose(EE_SITE)
    q_hold, _ = sim.object_pose(tid)
    shift = dest_xy[:2] - q_hold[:2]
    result["place_diag"]["piece_xy_before_lower_mm"] = [
        round(float(q_hold[0]) * 1000, 1),
        round(float(q_hold[1]) * 1000, 1),
    ]
    result["place_diag"]["piece_err_before_lower_mm"] = round(
        float(np.hypot(q_hold[0] - dest_xy[0], q_hold[1] - dest_xy[1])) * 1000, 1
    )
    result["place_diag"]["ee_xy_before_lower_mm"] = [
        round(float(gpos[0]) * 1000, 1),
        round(float(gpos[1]) * 1000, 1),
    ]
    result["place_diag"]["T_ee_before_lower_mm"] = [
        round(v, 1) for v in gripper_object_transform(sim, tid)["p_ee_mm"]
    ]
    # Aim held piece XY at dest, then lower vertically. Combined diagonal
    # lower would drag the piece by the leftover grasp/carry offset.
    # Dest inside the near-base hole: carry stopped on the workspace ring;
    # lower first, then slide in at place height (same radius rule as lift).
    place_z = float(g_close[2]) + place_hover
    standoff = bool(result.get("transport_standoff"))
    # Vertical place: align the *piece* over dest with the gripper world-down,
    # then descend straight. The pinch offset is measured in the gripper
    # frame and mapped through the vertical rotation at dest, so a carry that
    # tilted the wrist no longer lands the piece 5-25 mm off. Falls back to
    # the legacy carry-lock lower when the vertical pose is out of reach.
    vertical_place = _lower_vertical(sim, tid, dest_xy, place_z, hold, result)
    if vertical_place:
        place = vertical_place
    elif standoff:
        down = [float(gpos[0]), float(gpos[1]), place_z]
        if not _carry_lin(sim, down, 0.8, 8, hold, extract):
            result["failure_reason"] = "IK_FAILURE"
            result["ik_fail_phase"] = "008_lower"
            log_state(sim, "fail_lower", extra={"target": _xyz(down)})
            return _done()
        gpos, _ = sim.site_pose(EE_SITE)
        q_hold, _ = sim.object_pose(tid)
        shift = dest_xy[:2] - q_hold[:2]
        place = [float(gpos[0] + shift[0]), float(gpos[1] + shift[1]), place_z]
        result["place_diag"]["lower_target_xy_mm"] = [round(place[0] * 1000, 1), round(place[1] * 1000, 1)]
        if float(np.hypot(shift[0], shift[1])) > 0.001:
            if not _carry_lin(sim, place, 1.2, 12, hold, extract):
                result["failure_reason"] = "IK_FAILURE"
                result["ik_fail_phase"] = "008_lower"
                log_state(sim, "fail_lower", extra={"target": _xyz(place)})
                return _done()
    else:
        align = [float(gpos[0] + shift[0]), float(gpos[1] + shift[1]), float(gpos[2])]
        place = [align[0], align[1], place_z]
        if float(np.hypot(shift[0], shift[1])) > 0.001:
            if _carry_lin(sim, align, 0.5, 6, hold, extract):
                gpos, _ = sim.site_pose(EE_SITE)
                q_hold, _ = sim.object_pose(tid)
                shift = dest_xy[:2] - q_hold[:2]
                place = [
                    float(gpos[0] + shift[0]),
                    float(gpos[1] + shift[1]),
                    place_z,
                ]
        result["place_diag"]["lower_target_xy_mm"] = [round(place[0] * 1000, 1), round(place[1] * 1000, 1)]
        if blocked or populated:
            lower_ok = _carry_lin(sim, place, 1.2, 12, hold, extract)
        else:
            lower_ok = move_ik(
                sim, place, duration=1.2, physics=True, lock_joints=hold, ori_weight=CARRY_ORI_WEIGHT
            ) or move_ik(sim, place, duration=1.2, physics=True, lock_joints=extract)
        if not lower_ok:
            result["failure_reason"] = "IK_FAILURE"
            result["ik_fail_phase"] = "008_lower"
            log_state(sim, "fail_lower", extra={"target": _xyz(place)})
            return _done()
    log_state(sim, "008_lower", extra={"target": _xyz(place)})
    q_lo, _ = sim.object_pose(tid)
    g_lo, r_lo = sim.site_pose(EE_SITE)
    result["place_diag"]["piece_xy_after_lower_mm"] = [
        round(float(q_lo[0]) * 1000, 1),
        round(float(q_lo[1]) * 1000, 1),
    ]
    result["place_diag"]["piece_err_after_lower_mm"] = round(
        float(np.hypot(q_lo[0] - dest_xy[0], q_lo[1] - dest_xy[1])) * 1000, 1
    )
    result["place_diag"]["ee_xy_after_lower_mm"] = [
        round(float(g_lo[0]) * 1000, 1),
        round(float(g_lo[1]) * 1000, 1),
    ]
    result["place_diag"]["ee_x_z_after_lower"] = round(float(r_lo[2, 0]), 3)
    result["place_diag"]["T_ee_after_lower_mm"] = [
        round(v, 1) for v in gripper_object_transform(sim, tid)["p_ee_mm"]
    ]

    _, g_hi = sim.joint_limits()["gripper"]
    # Narrower than approach open: a full 0.60 swing slaps neighbors
    # once the piece is already on the square.
    release_g = min(float(g_hi), float(release_open))
    open_gripper(sim, release_g, steps=100)
    settle_queen(sim, steps=16)
    result["release_after_open"] = release_snapshot(sim, tid)
    q_open, _ = sim.object_pose(tid)
    result["place_diag"]["piece_xy_after_open_mm"] = [
        round(float(q_open[0]) * 1000, 1),
        round(float(q_open[1]) * 1000, 1),
    ]
    result["place_diag"]["piece_err_after_open_mm"] = round(
        float(np.hypot(q_open[0] - dest_xy[0], q_open[1] - dest_xy[1])) * 1000, 1
    )
    result["place_diag"]["still_held_after_open"] = bool(grasp_acquired(sim, tid))
    result["place_diag"]["piece_z_after_open_mm"] = round(float(q_open[2]) * 1000, 1)
    result["place_diag"]["release_gripper"] = round(release_g, 3)

    wrist = ("wrist_flex", "wrist_roll")
    release_lift = float(p.get("release_lift_z", 0.08))
    # Residual jaw contact is not a carry if the piece is already on the board.
    airborne = float(q_open[2]) > surface + 0.012
    result["place_diag"]["airborne_after_open"] = airborne
    if grasp_acquired(sim, tid) and airborne:
        gpos, _ = sim.site_pose(EE_SITE)
        clear = [float(gpos[0]), float(gpos[1]), float(gpos[2]) + release_lift]
        move_lin(sim, clear, duration=0.8, physics=True, n_wp=6, lock_joints=wrist)
        settle_queen(sim, steps=24)
    elif grasp_acquired(sim, tid):
        for _ in range(3):
            open_gripper(sim, release_g, steps=80)
            settle_queen(sim, steps=30)
            if not grasp_acquired(sim, tid):
                break
    q_open, _ = sim.object_pose(tid)
    airborne = float(q_open[2]) > surface + 0.012
    # Pull the stationary pad off the piece before any +Z so a lip/head
    # does not stay hooked on the fixed finger hull.
    if not airborne:
        slide, travelled = _slide_pad_off_piece(sim, tid, wrist)
        result["place_diag"]["pad_clear_mm"] = round(travelled * 1000, 1)
        if slide is not None:
            result["place_diag"]["pad_clear_xy_mm"] = [
                round(slide[0] * 1000, 1),
                round(slide[1] * 1000, 1),
            ]
    result["release_after_lift"] = release_snapshot(sim, tid)

    q_rel, qquat_rel = sim.object_pose(tid)
    result["release_position_error_mm"] = float(np.hypot(q_rel[0] - dest_xy[0], q_rel[1] - dest_xy[1])) * 1000
    result["place_diag"]["release_xy_mm"] = [
        round(float(q_rel[0]) * 1000, 1),
        round(float(q_rel[1]) * 1000, 1),
    ]
    result["place_diag"]["release_err_mm"] = round(result["release_position_error_mm"], 1)
    still_held = bool(grasp_acquired(sim, tid) and float(q_rel[2]) > surface + 0.012)
    if off_board:
        seated = bowl_contains(q_rel, dest_xy)
        upright = True
        on_sq = seated
        result["placement_success"] = bool(seated and not still_held)
    else:
        on_sq = square_of_body(sim, cfg, tid) == dest
        seated = object_on_board(sim, surface, object_id=tid)
        upright = queen_upright(qquat_rel, tilt_max=0.5)
        result["placement_success"] = bool(on_sq and upright and seated and not still_held)
    result["place_diag"]["still_held"] = still_held
    result["place_diag"]["jaw_contact"] = bool(grasp_acquired(sim, tid))
    result["place_diag"]["seated"] = bool(seated)
    result["place_diag"]["upright"] = bool(upright)
    result["place_diag"]["piece_z_mm"] = round(float(q_rel[2]) * 1000, 1)
    result["place_diag"]["off_board"] = bool(off_board)
    log_state(sim, "009_release", extra={"release_err_mm": round(result["release_position_error_mm"], 2)})

    gpos, _ = sim.site_pose(EE_SITE)
    touching = bool(grasp_acquired(sim, tid))
    result["place_diag"]["jaw_contact_before_retreat"] = touching
    # Lift above the piece before any later XY so the next grab does not
    # sideswipe it. Skip hop only while jaws still touch a seated piece
    # (a +Z comb of a tall crown would lift it).
    z_up = max(
        float(gpos[2]) + release_lift,
        float(result.get("transit_z") or 0.0),
        _transit_z(sim, cfg, skip={tid}),
    )
    high = [float(gpos[0]), float(gpos[1]), z_up]
    retreated = True
    if seated and not off_board and touching:
        high = [float(gpos[0]), float(gpos[1]), float(gpos[2])]
    elif float(gpos[2]) < z_up - 0.01:
        retreated = _safe_lin(
            sim, high, cfg, wrist, obstacles=obstacle_bodies(sim, skip={tid}), skip={tid}
        )
    if not retreated:
        result["ik_fail_phase"] = "010_final"
        result["retreat_success"] = False
        if not result["placement_success"]:
            result["failure_reason"] = "RELEASE_FAILURE" if still_held else "PLACEMENT_POSITION_ERROR"
        else:
            result["failure_reason"] = "IK_FAILURE"
        log_state(sim, "fail_retreat", extra={"target": _xyz(high)})
        return _done()
    settle_queen(sim, steps=24)
    log_state(sim, "010_final")

    qpos, qquat = sim.object_pose(tid)
    result["final_position_error_mm"] = float(np.hypot(qpos[0] - dest_xy[0], qpos[1] - dest_xy[1])) * 1000
    result["final_queen_xyz"] = qpos.tolist()
    result["place_diag"]["final_xy_mm"] = [
        round(float(qpos[0]) * 1000, 1),
        round(float(qpos[1]) * 1000, 1),
    ]
    result["place_diag"]["final_err_mm"] = round(result["final_position_error_mm"], 1)
    opened = result["place_diag"].get("piece_err_after_open_mm")
    released = result["place_diag"].get("release_err_mm")
    if opened is not None and released is not None:
        result["place_diag"]["slide_after_release_mm"] = round(float(released) - float(opened), 1)
    result["place_diag"]["jaw_contact_after_retreat"] = bool(grasp_acquired(sim, tid))
    result["place_diag"]["final_z_mm"] = round(float(qpos[2]) * 1000, 1)
    result["place_diag"]["final_upright"] = bool(queen_upright(qquat, tilt_max=0.5))
    on_dest = square_of_body(sim, cfg, tid) == dest
    if off_board:
        result["retreat_success"] = bool(
            bowl_contains(qpos, dest_xy) and not grasp_acquired(sim, tid)
        )
    else:
        result["retreat_success"] = bool(
            on_dest
            and queen_upright(qquat, tilt_max=0.5)
            and abs(float(qpos[2]) - surface) < 0.02
        )
    if not result["placement_success"]:
        result["failure_reason"] = "RELEASE_FAILURE" if still_held else "PLACEMENT_POSITION_ERROR"
        return _done()
    if not result["retreat_success"]:
        result["failure_reason"] = "RETREAT_COLLISION"
        return _done()
    result["success"] = True
    result["failure_reason"] = None
    return _done()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--vision", action="store_true", help="use wrist RGB-D instead of ground-truth pose")
    parser.add_argument("--full-board", action="store_true", help="load starting-position chess_scene_full.xml")
    parser.add_argument("--random", action="store_true", help="spawn queen at a random on-board XY, not a square")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", action="store_true", help="write overview+side mp4 under sim/debug/pickup/")
    args = parser.parse_args()
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

        sc = full_board_scenario(perception="vision" if args.vision else "gt")
    sim = Sim(xml_path=xml, cfg=cfg)
    rec = None
    if args.video:
        tag = "vision" if args.vision else "gt"
        if args.random:
            tag += f"_random{args.seed}"
        rec = VideoRecorder(sim, DEBUG_DIR / "pickup" / f"{tag}.mp4")
        sim.on_step = rec.grab
    try:
        from episode import run_pickup

        result = run_pickup(sim, cfg, use_vision=args.vision, spawn_xy=spawn_xy, scenario=sc)
        if rec is not None:
            rec.save()
            print(f"video: {rec.path} ({len(rec.frames)} frames)")
    finally:
        sim.close()
    print("\n" + json.dumps(result, indent=2))
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    (DEBUG_DIR / "pickup" / "result.json").write_text(json.dumps(result, indent=2))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
