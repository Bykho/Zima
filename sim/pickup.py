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

import carry as plumb
import diag
import evaluate_move

from ik import (
    CARRY_ORI_WEIGHT,
    EE_SITE,
    cart_duration,
    gripper_tilt_deg,
    move_ik,
    move_lin,
    move_qpos,
    pose_reaches,
    reachable_approaches,
    solve_ik,
    solve_ik_down,
)
from mj import ARM_JOINTS, DEBUG_DIR, JOINT_NAMES, Sim, load_config, probing
from clearance import (
    APPROACH_BODIES,
    HARD_MARGIN,
    board_top_z,
    closest_pair,
    near_bodies,
    neighbor_approach_gap,
    obstacle_bodies,
    plan_approach,
    safe_transit_ee_z,
    transit_clearances,
    transit_nontarget_hit,
    verify_approach,
)
from perception import estimate_target_pose, observe_piece
from sensor import capture as capture_wrist, robot_blocks_view, wrist_sees_point
from squares import bowl_xyz, jitter_near_square, nest_xyz, square_to_world
from metrics import annotate_episode
from scenario import PIECE_BODIES, PIECES, STARTING_AT, body_kind
from sequence import square_of_body
from robot import GRIP_SETTLE_STEPS, GRIP_STILL_RADS, GRIP_STILL_READS, WALLS, jaw_squeezing, open_robot, world_of

JAW_BODIES = ("gripper", "moving_jaw_so101_v1")
CAMERA_BODIES = ("camera_mount",)
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


def _pitch_in_place(sim: Sim, flex_goal: float, n: int = 8) -> None:
    """Rotate wrist_flex to `flex_goal` while shoulder/elbow hold the EE put.

    A wrist-only move at a stretched observe pose swings the gripper on a
    ~15 cm lever: traced dropping the EE from z=0.14 to z=0.024 and sweeping
    the jaw through the rank below. Plan small pitch steps with position IK
    re-solved at each (wrist locked), then servo through them under physics.
    """
    if abs(flex_goal - float(sim.qpos()["wrist_flex"])) < 1e-3:
        return
    path = _pitch_path(sim, flex_goal, n)
    _drive_arm(sim, path, float(sim.qpos()["gripper"]), duration=0.5)


def _roll_in_place(sim: Sim, roll_goal: float, rad_per_s: float = 2.0) -> None:
    """Turn wrist_roll to `roll_goal` under physics, the other arm joints held at their current targets.

    A 180 deg flip in 0.35 s saturated the shoulder/elbow servos and the arm
    sagged onto the rank below (jaw swept c2/d2/e2 at ee_z=25 mm), so the turn
    runs at a bounded wrist speed in short legs and settles before returning.
    """
    q0 = {j: float(sim.qpos()[j]) for j in ARM_JOINTS}
    d = roll_goal - q0["wrist_roll"]
    if abs(d) < 1e-3:
        return
    legs = max(1, int(np.ceil(abs(d) / 0.4)))
    path = [{**q0, "wrist_roll": q0["wrist_roll"] + d * i / legs} for i in range(1, legs + 1)]
    _drive_arm(sim, path, float(sim.qpos()["gripper"]), duration=max(0.3, abs(d) / rad_per_s))
    for _ in range(4):
        if abs(float(sim.qpos()["wrist_roll"]) - roll_goal) <= 0.05:
            break
        sim.step(20)


def _pitch_holding(sim: Sim, flex_goal: float, closed: float, tid: str | None = None, n: int = 10) -> None:
    """Same as _pitch_in_place, but keep the gripper shut so a held piece follows."""
    if abs(flex_goal - float(sim.qpos()["wrist_flex"])) < 1e-3:
        return
    path = _pitch_path(sim, flex_goal, n)
    _drive_arm(sim, path, closed, duration=0.6, until=lambda: grasp_acquired(sim, tid))


_LEVEL_PINCH_MIN_DEG = 2.5


def _level_pinch_axis(sim: Sim) -> float:
    """Roll the wrist so the pinch axis (gripper local Z) is horizontal.

    Returns the roll change applied, in degrees (0 if already level or no
    reachable roll levels it). Kinematic search over the candidate angles,
    then a physics move so the held piece follows.
    """
    _, R = sim.site_pose(EE_SITE)
    z_z = float(R[2, 2])
    if abs(z_z) < np.sin(np.radians(_LEVEL_PINCH_MIN_DEG)):
        return 0.0
    q0 = sim.qpos()
    cur = float(q0["wrist_roll"])
    lo, hi = sim.model.jnt_range[sim._joint["wrist_roll"]]
    phi0 = float(np.arctan2(z_z, float(R[2, 1])))
    best = None
    ctrl0 = sim.data.ctrl.copy()
    with sim.probe():
        try:
            for phi in (phi0, -phi0, phi0 + np.pi, phi0 - np.pi, -phi0 + np.pi, -phi0 - np.pi):
                cand = float(np.clip(cur + phi, lo, hi))
                sim.set_qpos({"wrist_roll": cand}, hold=False)
                _, Rc = sim.site_pose(EE_SITE)
                if abs(float(Rc[2, 2])) > np.sin(np.radians(_LEVEL_PINCH_MIN_DEG)):
                    continue
                if best is None or abs(cand - cur) < abs(best - cur):
                    best = cand
        finally:
            sim.set_qpos({"wrist_roll": cur}, hold=False)
            sim.data.ctrl[:] = ctrl0
    if best is None or abs(best - cur) < 1e-3:
        return 0.0
    _roll_holding(sim, best)
    return float(np.degrees(best - cur))


def _roll_holding(sim: Sim, target_roll: float) -> None:
    """Roll the wrist under physics while every other ctrl (the gripper
    squeeze in particular) stays exactly as the close left it."""
    aid = sim._act["wrist_roll"]
    cur = float(sim.qpos()["wrist_roll"])
    n = max(20, int(max(0.3, 0.4 * abs(target_roll - cur)) / sim.dt))
    for i in range(n):
        u = (i + 1) / n
        u = u * u * (3 - 2 * u)
        sim.data.ctrl[aid] = cur + (target_roll - cur) * u
        sim.step()
    sim.step(10)


_PLACE_MARGIN_M = 0.003


@probing
def _plan_place(
    sim: Sim, tid: str, dest_xy, place_z: float, hold: tuple[str, ...], release_open: float, margin: float = _PLACE_MARGIN_M, x_axis=None
) -> dict:
    """Pick the wrist roll (level pinch, either side) that keeps jaws off dest neighbors.

    Release aperture stays at `release_open`. Narrowing it (0.15 on b8c6) left
    the knight hooked in the pads; it tipped on the square.
    """
    gpos, grot = sim.site_pose(EE_SITE)
    q, _ = sim.object_pose(tid)
    p_local = grot.T @ (q - gpos)
    cur = float(sim.qpos()["wrist_roll"])
    lo, hi = sim.model.jnt_range[sim._joint["wrist_roll"]]
    rolls = [cur]
    for d in (0.4, -0.4, 0.8, -0.8):
        alt = float(np.clip(cur + d, lo, hi))
        if all(abs(alt - r) > 0.05 for r in rolls):
            rolls.append(alt)
    neighbors = [n for n in obstacle_bodies(sim, skip={tid}) if n != "board"]
    from clearance import _min_gap, near_bodies

    neighbors = near_bodies(sim, neighbors, dest_xy)
    out = {"roll": cur, "release": float(release_open), "planned": False, "min_gap_mm": None}
    if not neighbors:
        return out
    snap = sim.qpos()
    ctrl0 = sim.data.ctrl.copy()
    best = None
    try:
        for roll in rolls:
            sim.set_qpos({**snap, "wrist_roll": roll}, hold=False)
            ee = _place_ee_target(sim, tid, dest_xy, place_z, hold, p_local=p_local, x_axis=x_axis)
            if ee is None:
                continue
            sim.set_qpos({**snap, "wrist_roll": roll}, hold=False)
            if not solve_ik_down(sim, ee, lock_joints=hold, x_axis=x_axis):
                continue
            q_low = sim.qpos()
            sim.set_qpos({**q_low, "gripper": float(release_open)}, hold=False)
            gap = _min_gap(sim, APPROACH_BODIES, neighbors)
            sim.set_qpos(q_low, hold=False)
            gap = min(gap, _min_gap(sim, APPROACH_BODIES, neighbors))
            cand = {"roll": roll, "release": float(release_open), "planned": True, "min_gap_mm": round(gap * 1000.0, 2)}
            if gap >= margin:
                return cand
            if best is None or gap > best["min_gap_mm"] / 1000.0:
                best = {**cand, "degraded": True}
    finally:
        sim.set_qpos(snap, hold=False)
        sim.data.ctrl[:] = ctrl0
    return best if best is not None else out


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


@probing
def _place_ee_target(
    sim: Sim, tid: str, dest_xy, z_ee: float, lock: tuple[str, ...], p_local=None, x_axis=None
) -> list[float] | None:
    """EE point at height z_ee that puts the held piece over dest_xy.

    piece = ee + R p_local. R is the gripper rotation at dest (vertical unless
    `x_axis` freezes the current approach). Restores qpos.
    """
    if p_local is None:
        gpos, grot = sim.site_pose(EE_SITE)
        q, _ = sim.object_pose(tid)
        p_local = grot.T @ (q - gpos)
    snap = sim.qpos()
    try:
        ee = np.array([float(dest_xy[0]), float(dest_xy[1]), float(z_ee)])
        for _ in range(2):
            if not solve_ik_down(sim, ee, lock_joints=lock, x_axis=x_axis):
                return None
            _, R = sim.site_pose(EE_SITE)
            off = R @ p_local
            ee = np.array([float(dest_xy[0]) - off[0], float(dest_xy[1]) - off[1], float(z_ee)])
        return [float(v) for v in ee]
    finally:
        sim.set_qpos(snap, hold=True)


def _place_lower_guard(sim: Sim, tid: str, bystanders: list[str]):
    """Stop a place-lower before jaws shove a neighbor (Rxd7 / Qd8)."""
    if _touching_board(sim, tid):
        return "board"
    if bystanders:
        return bystander_contact(sim, bystanders)
    return None


def _lower_vertical(
    sim: Sim, tid: str, dest_xy, place_z: float, hold: tuple[str, ...], result: dict, x_axis=None
) -> list[float] | None:
    """Slide the held piece over dest, then drop. Wrists stay put.

    Re-solving a vertical pinch on the way down swung the jaws and dumped
    the bishop (f1c4). Lock flex+roll and translate.
    """
    lock = tuple(dict.fromkeys((*hold, "wrist_flex", "wrist_roll")))
    bystanders = near_bodies(
        sim,
        [n for n in obstacle_bodies(sim, skip={tid}) if n != "board"],
        dest_xy,
        radius=0.055,
    )
    guard = lambda: _place_lower_guard(sim, tid, bystanders)
    gpos, _ = sim.site_pose(EE_SITE)
    q, _ = sim.object_pose(tid)
    align = [
        float(gpos[0] + dest_xy[0] - q[0]),
        float(gpos[1] + dest_xy[1] - q[1]),
        float(gpos[2]),
    ]
    if float(np.hypot(align[0] - gpos[0], align[1] - gpos[1])) > 0.001:
        if not move_lin(sim, align, duration=0.8, physics=True, n_wp=8, lock_joints=lock, guard=guard):
            if getattr(sim, "_last_abort", None) not in (None, "board"):
                result["place_diag"]["lower_bystander"] = getattr(sim, "_last_abort", None)
            # High hover over a near-base dest is often dead. Drop, then slide.
            for z in (0.07, 0.055, 0.045):
                drop = [float(gpos[0]), float(gpos[1]), float(z)]
                if not move_lin(sim, drop, duration=0.6, physics=True, n_wp=6, lock_joints=lock, guard=guard):
                    continue
                gpos, _ = sim.site_pose(EE_SITE)
                q, _ = sim.object_pose(tid)
                slide = [
                    float(gpos[0] + dest_xy[0] - q[0]),
                    float(gpos[1] + dest_xy[1] - q[1]),
                    float(z),
                ]
                if move_lin(sim, slide, duration=0.8, physics=True, n_wp=8, lock_joints=lock, guard=guard):
                    break
            else:
                return None
    gpos, _ = sim.site_pose(EE_SITE)
    q, _ = sim.object_pose(tid)
    place = [
        float(gpos[0] + dest_xy[0] - q[0]),
        float(gpos[1] + dest_xy[1] - q[1]),
        float(place_z),
    ]
    result["place_diag"]["lower_target_xy_mm"] = [round(place[0] * 1000, 1), round(place[1] * 1000, 1)]
    result["place_diag"]["lower_vertical"] = True
    sim._last_abort = None
    ok = move_lin(
        sim, place, duration=1.2, physics=True, n_wp=12, lock_joints=lock, guard=guard,
    )
    aborted = getattr(sim, "_last_abort", None)
    if aborted and aborted != "board":
        result["place_diag"]["lower_bystander"] = aborted
        cur = float(sim.qpos()["wrist_roll"])
        lo, hi = sim.joint_limits()["wrist_roll"]
        for d in (0.4, -0.4, 0.8, -0.8):
            _roll_holding(sim, float(np.clip(cur + d, lo, hi)))
            sim._last_abort = None
            gpos, _ = sim.site_pose(EE_SITE)
            q, _ = sim.object_pose(tid)
            place = [
                float(gpos[0] + dest_xy[0] - q[0]),
                float(gpos[1] + dest_xy[1] - q[1]),
                float(place_z),
            ]
            ok = move_lin(
                sim, place, duration=0.8, physics=True, n_wp=8, lock_joints=lock, guard=guard,
            )
            aborted = getattr(sim, "_last_abort", None)
            if ok or aborted == "board":
                result["place_diag"]["lower_roll_retry"] = round(d, 2)
                break
    if not ok and aborted not in (None, "board"):
        # Stopped on a neighbor instead of shoving it. Release from here.
        result["place_diag"]["lower_stopped_clear"] = True
        return place
    if not ok and aborted != "board":
        return None
    if aborted == "board":
        result["place_diag"]["lower_touchdown_early"] = True
    return place


def _plumb_transport(
    sim: Sim, tid: str, result: dict, dest_xy, piece_place_z: float, lift_z: float, obstacles: list[str]
) -> dict | None:
    """Lift, carry and lower with the held piece kept upright (carry.py).

    None: no plumb path exists, nothing moved. Otherwise {"place": ee} or
    {"failure_reason": ...} after the arm has moved.
    """
    v_local, p_local = plumb.hand_frame(sim, tid)
    avoid = [n for n in obstacles if n not in (tid, "board")]
    gpos, _ = sim.site_pose(EE_SITE)
    z_t, _ = safe_transit_ee_z(sim, float(gpos[2]), tid, obstacles)
    z_t = max(z_t, float(gpos[2]) + plumb.PLUMB_END_M)
    end = plumb.PLUMB_END_M
    low = [float(gpos[0]), float(gpos[1]), float(gpos[2]) + end]
    up = [float(gpos[0]), float(gpos[1]), z_t]
    place = plumb.ee_for_piece(sim, [dest_xy[0], dest_xy[1], piece_place_z], v_local, p_local)
    if place is None:
        result["plumb"] = {"no_pose": getattr(sim, "plumb_fail", None)}
        return None
    above = [place[0], place[1], place[2] + end]
    hover = [place[0], place[1], max(z_t, above[2])]
    loose = plumb.AIRBORNE_TILT_DEG
    over = plumb.plan(sim, [(low, plumb.LIFT_TILT_DEG), (up, loose), (hover, loose)], v_local, avoid)
    if over is None:
        result["plumb"] = {"no_carry": getattr(sim, "plumb_fail", None)}
        return None
    snap = sim.qpos()
    with sim.probe():
        sim.set_qpos(over[-1], hold=True)
        down = plumb.plan(sim, [(above, loose), (place, plumb.PLACE_TILT_DEG)], v_local, avoid, gap=0.002)
        sim.set_qpos({n: snap[n] for n in ARM_JOINTS}, hold=True)
    if down is None:
        result["plumb"] = {"no_lower": getattr(sim, "plumb_fail", None)}
        return None
    result["plumb"] = "ok"
    result["transport_path"] = "plumb"
    result["transit_z"] = z_t
    log_state(sim, "006_lift")
    plumb.follow(sim, over, plumb.PLUMB_SPEED_MPS)
    held = grasp_acquired(sim, tid)
    result["lift_success"] = held
    result["transport_success"] = held
    log_state(sim, "007_transport", extra={"target": _xyz(hover), "path": "plumb"})
    if not held:
        return {"failure_reason": "TRANSPORT_DROP"}
    bystanders = near_bodies(sim, avoid, dest_xy, radius=0.055)
    sim._last_abort = None
    plumb.follow(sim, down, plumb.PLUMB_LOWER_SPEED_MPS, guard=lambda: _place_lower_guard(sim, tid, bystanders))
    if getattr(sim, "_last_abort", None) not in (None, "board"):
        result.setdefault("place_diag", {})["lower_bystander"] = sim._last_abort
    return {"place": [float(v) for v in place]}


def _touching_board(sim: Sim, tid: str):
    return "board" if any(h.get("other") == "board" for h in sim.contacts_with(tid)) else None


def _carry_candidates(start, goal, cfg: dict, n_wp: int, lock: tuple[str, ...]):
    """Carry lines that end on the real goal.

    Midpoints may sit on the 190 mm ring so a high transit does not ask IK
    for a hover that does not exist. The last point is never floored — that
    parked the piece 40+ mm off dest and scraped in at place height (b5).
    """
    n_wp = max(8, int(n_wp))
    raw = _line_wps(start, goal, n_wp)
    dest = [float(goal[0]), float(goal[1]), float(goal[2])]
    yield raw, lock, "straight"
    yield [[w[0], w[1], w[2] + 0.03] for w in raw], lock, "high"
    ring = [_floor_wp(w, cfg) for w in raw[:-1]] + [dest]
    if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) > 0.002 for a, b in zip(raw, ring)):
        yield ring, lock, "ring_then_in"
        yield [[w[0], w[1], w[2] + 0.03] for w in ring], lock, "ring_then_in_high"
    if "wrist_flex" in lock:
        unlock = tuple(j for j in lock if j != "wrist_flex")
        yield raw, unlock, "straight_unlock_flex"


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
    if _carry_lin(sim, goal, duration, max(8, n_wp), lock, extract):
        return True, "straight"
    # Near-base dest: a 10 cm hover is often dead while a low pinch is not.
    low_z = min(float(goal[2]), 0.06)
    if float(goal[2]) > low_z + 0.008:
        low = [float(goal[0]), float(goal[1]), low_z]
        if _carry_lin(sim, low, duration, max(8, n_wp), lock, extract):
            return True, "low"
    return False, "straight"


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


_BEARING_NZ = 0.7  # contact normal within ~45 deg of vertical


def _hand_bears_on(sim: Sim, tid: str) -> bool:
    """A robot link presses on the piece from above or below (a lip or crown
    caught on a jaw). Sliding the hand out then drags the piece by friction."""
    w = world_of(sim) if getattr(sim, "cheat_state", True) else sim
    b1, b2 = w.contact_bodies()
    if b1.size == 0:
        return False
    robot = _robot_bid_array(sim)
    me = sim.body_id(tid)
    hit = (np.isin(b1, robot) & (b2 == me)) | (np.isin(b2, robot) & (b1 == me))
    nz = np.abs(w.data.contact.frame[: b1.size, 2])
    return bool((hit & (nz > _BEARING_NZ)).any())


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


def _follow_q_path(sim: Sim, path_q: list[dict], duration: float, guard=None) -> bool:
    """Track a list of joint configurations with ctrl only, guard checked every step."""
    segs = max(1, len(path_q) - 1)
    n_total = max(30, int(duration / sim.dt))
    per = max(4, n_total // segs)
    cur = sim.qpos()
    for goal in path_q:
        for i in range(per):
            u = (i + 1) / per
            cmd = {k: cur[k] + (goal.get(k, cur[k]) - cur[k]) * u for k in cur}
            sim.set_ctrl(cmd)
            sim.step()
            if guard is not None:
                hit = guard()
                if hit:
                    sim._last_abort = hit
                    sim.hold(sim.qpos())
                    return False
        cur = {k: goal.get(k, cur[k]) for k in cur}
    sim.hold(cur)
    sim.step(20)
    return True


_COLUMN_TILT_DEG = 2.0


def _descend_column(sim: Sim, grasp, lock: tuple[str, ...]) -> list[dict] | None:
    """Turn the claw to its grasp angle at this height, then plan straight down.

    The claw only changes angle up here: rotating on the way down sweeps
    the jaws sideways through the piece. Returns the vertical joint chain
    (the arm is already over the grasp at the grasp angle), or None with
    nothing moved.
    """
    snap = sim.qpos()
    with sim.probe():
        try:
            if not solve_ik_down(sim, grasp, lock_joints=lock):
                return None
            _, R = sim.site_pose(EE_SITE)
            v_local = np.asarray(R, dtype=float).reshape(3, 3).T @ plumb.UP
            sim.set_qpos(snap, hold=True)
            gpos, _ = sim.site_pose(EE_SITE)
            above = [float(grasp[0]), float(grasp[1]), float(gpos[2])]
            if not plumb.solve_plumb(sim, above, v_local, _COLUMN_TILT_DEG):
                return None
            q_above = {n: float(sim.qpos()[n]) for n in ARM_JOINTS}
            down = plumb.plan(sim, [(grasp, _COLUMN_TILT_DEG)], v_local, [])
        finally:
            sim.set_qpos(snap, hold=True)
    if down is None:
        return None
    _follow_q_path(sim, [q_above], 0.6)
    return down


def _physical_descend_path(sim: Sim, target, path_q: list[dict], duration: float) -> dict:
    """Replay the planner's verified joint path (pre -> grasp) with the bystander guard."""
    tid = _tid(sim)
    q0, _ = sim.object_pose(tid)
    target_ee = np.asarray(target, dtype=np.float64)
    bystanders = [n for n in obstacle_bodies(sim, skip={tid}) if n != "board"]
    sim._last_abort = None
    guard = (lambda: bystander_contact(sim, bystanders)) if bystanders else None
    _follow_q_path(sim, path_q, duration, guard=guard)
    aborted = getattr(sim, "_last_abort", None)
    pos, _ = sim.site_pose(EE_SITE)
    qpos, qquat = sim.object_pose(tid)
    err_mm = float(np.linalg.norm(pos - target_ee)) * 1000
    hits = jaw_contacts(sim, tid)
    return {
        "ok": err_mm < 20.0 and not aborted,
        "aborted": aborted,
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
        "path_replay": True,
    }


def physical_descend(sim: Sim, target, duration: float = 1.2, lock_joints=(), path_q=None) -> dict:
    """Joint-space interpolation with actuator commands only (no qpos overwrite).

    path_q: planner-verified joint path (pre -> grasp); replayed as-is.
    """
    if path_q:
        return _physical_descend_path(sim, target, list(path_q), duration)
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
    with sim.probe():
        solved = solve_ik_down(sim, target, lock_joints=lock_joints)
        pos, _ = sim.site_pose(EE_SITE)
        report["joint_limit_margin_rad"] = joint_limit_margin(sim)
        if not solved:
            report["ee_actual"] = _xyz(pos)
            report["ee_err_mm"] = float(np.linalg.norm(pos - target_ee)) * 1000
            report["ik_residual_mm"] = report["ee_err_mm"]
        else:
            goal = sim.qpos()
            report["ik_solved"] = True
            report["tilt_deg"] = gripper_tilt_deg(sim)
            report["ik_residual_mm"] = float(np.linalg.norm(pos - target_ee)) * 1000
            report["ik_jaw"] = jaw_snapshot(sim, tid)
        sim.set_qpos(start, hold=True)
    if not solved:
        return report
    n = max(30, int(duration / sim.dt))
    peak = np.zeros(len(JOINT_NAMES))
    for i in range(n):
        u = (i + 1) / n
        u = u * u * (3 - 2 * u)
        cmd = {k: start[k] + (goal[k] - start[k]) * u for k in start}
        sim.set_ctrl(cmd)
        sim.step()
        force = sim.actuator_forces()
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
        force = np.abs(sim.actuator_forces())
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


_CENTRE_SAMPLES = 8


@probing
def _roomiest_along(sim: Sim, tid: str, start, shift, v_local) -> tuple[np.ndarray | None, float]:
    """EE point on start -> start + shift where the open hand stands farthest
    from the piece, and that gap. The pad midpoint is only the centre at pad
    height; the hinged moving jaw leans in above it. Restores qpos."""
    snap = sim.qpos()
    best, room = None, -1.0
    try:
        for k in range(_CENTRE_SAMPLES + 1):
            p = start + shift * (k / _CENTRE_SAMPLES)
            if not plumb.solve_plumb(sim, p, v_local, plumb.LIFT_TILT_DEG):
                continue
            g = float(neighbor_approach_gap(sim, [tid]))
            if g > room + 1e-4:
                best, room = p, g
        return best, room
    finally:
        sim.set_qpos({n: snap[n] for n in ARM_JOINTS}, hold=True)


# A piece left leaning on the fixed jaw rocks back upright if the jaw leaves
# slowly; a quick slide drags it past its tipping angle.
_RELEASE_SLIDE_MPS = 0.02


@probing
def _plan_slide_rise(sim: Sim, centre, top, v_local, avoid: list[str]) -> tuple[list | None, list | None]:
    snap = sim.qpos()
    try:
        slide = plumb.plan(sim, [(centre, plumb.LIFT_TILT_DEG)], v_local, avoid, gap=0.002)
        if slide is None:
            return None, None
        sim.set_qpos(slide[-1], hold=True)
        return slide, plumb.plan(sim, [(top, plumb.LIFT_TILT_DEG)], v_local, avoid, gap=0.002)
    finally:
        sim.set_qpos({n: snap[n] for n in ARM_JOINTS}, hold=True)


def _centred_retreat(sim: Sim, tid: str, avoid: list[str], z_up: float, jaw_clear_m: float) -> dict:
    """After release: slide the open hand along the pinch axis to where it
    stands farthest from the piece, then rise holding the hand's orientation
    until the jaws clear the piece top, then up to z_up leaning freely.

    Only the moving jaw opens, so the piece is left against the fixed pad.
    Rising from there drags it; a rise that re-pitches the hand swings a jaw
    into it. The room is measured on the robot's own model at the pose it set
    the piece down, and the rise is refused if it closes on the piece.
    """
    fixed, moving = plumb._pad_xyz(sim, plumb.PAD_GEOMS[0]), plumb._pad_xyz(sim, plumb.PAD_GEOMS[1])
    u = moving - fixed
    u[2] = 0.0
    if float(np.linalg.norm(u)) < 1e-6:
        return {"ok": False, "why": "pads"}
    u /= float(np.linalg.norm(u))
    q, _ = sim.object_pose(tid)
    mid = 0.5 * (fixed + moving)
    shift = float((np.asarray(q, dtype=float) - mid) @ u) * u
    gpos, R = sim.site_pose(EE_SITE)
    v_local = np.asarray(R, dtype=float).reshape(3, 3).T @ plumb.UP
    centre, room = _roomiest_along(sim, tid, np.asarray(gpos, dtype=float), shift, v_local)
    if centre is None:
        return {"ok": False, "why": "ik"}
    z_clear = float(gpos[2]) + jaw_clear_m
    top = [centre[0], centre[1], max(float(centre[2]), z_clear)]
    out = {"shift_mm": round(float(np.linalg.norm(centre - np.asarray(gpos))) * 1000, 1), "room_mm": round(room * 1000, 1)}
    slide, rise = _plan_slide_rise(sim, centre, top, v_local, [*avoid, tid])
    if rise is None:
        return {**out, "ok": False, "why": getattr(sim, "plumb_fail", None)}
    plumb.follow(sim, slide, _RELEASE_SLIDE_MPS)
    plumb.follow(sim, rise, plumb.PLUMB_LOWER_SPEED_MPS)
    out["ok"] = True
    high = top[2] >= z_up or None
    if not high:
        rest = plumb.plan(sim, [([centre[0], centre[1], float(z_up)], plumb.AIRBORNE_TILT_DEG)], v_local, avoid, gap=0.002)
        high = rest is not None and plumb.follow(sim, rest, plumb.PLUMB_SPEED_MPS)
    out["high"] = bool(high)
    return out


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
    if n >= 0.25:
        return v / n
    # Upside-down / nearly vertical: body +Z has no usable XY. Scoop
    # along COM vs origin so recover still has a direction (Nf6).
    com = _piece_com(sim, tid)
    qpos, _ = sim.object_pose(tid)
    v = com[:2] - qpos[:2]
    n = float(np.linalg.norm(v))
    if n >= 1e-4:
        return v / n
    return np.array([1.0, 0.0], dtype=float)


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


@probing
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


_WRIST_HOLD = ("wrist_flex", "wrist_roll")
_IK_HOLD_ERR = 0.012


def _pinch_xyz(sim: Sim, tid: str, surface: float) -> np.ndarray:
    """Stem-height target a few mm below COM so the pads wrap, not sit on top."""
    com = _piece_com(sim, tid)
    z = max(float(surface) + 0.005, float(com[2]) - 0.005)
    return np.array([float(com[0]), float(com[1]), z], dtype=float)


def _pinch_xyz_base(sim: Sim, tid: str, surface: float) -> np.ndarray:
    """Fat-end pinch. Body origin is the pedestal; COM sits toward the crown."""
    com = _piece_com(sim, tid)
    qpos, _ = sim.object_pose(tid)
    xy = 0.7 * np.asarray(qpos[:2], dtype=float) + 0.3 * com[:2]
    z = max(float(surface) + 0.005, float(com[2]) - 0.005)
    return np.array([float(xy[0]), float(xy[1]), z], dtype=float)


@probing
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
        if align < 0.55:
            continue
        score = -abs(float(od[2])) - 2.0 * abs(zmin - 0.025) + 0.1 * align
        if zmin < 0.018 or zmin > 0.040:
            score -= 2.0
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


@probing
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
    tid: str | None = None,
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
        if require_grasp and not grasp_acquired(sim, tid):
            return False
    return True


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


def _js_extract(sim: Sim, tid: str, hold: dict, closed: float, surface: float) -> dict | None:
    """Raise the scoop wrist in joint space so the wrap lifts the piece off the board."""
    lo, hi = sim.joint_limits()["shoulder_lift"]
    pose = {n: hold[n] for n in ARM_JOINTS}
    for _ in range(10):
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
        # L-shaped men can have COM up while a prong still kisses the board.
        # Waiting for zero board contact then fails extract (Nf6 / Bxf3).
        lifted = float(qpos[2]) > surface + 0.016
        clear = (not board) or float(qpos[2]) > surface + 0.024
        if lifted and clear and float(ee[2]) >= 0.040:
            return pose
    qpos, _ = sim.object_pose(tid)
    board = any(h["other"] == "board" and h["force"] > 0.05 for h in sim.contacts_with(tid))
    ee, _ = sim.site_pose(EE_SITE)
    lifted = float(qpos[2]) > surface + 0.016
    clear = (not board) or float(qpos[2]) > surface + 0.024
    if grasp_acquired(sim, tid) and lifted and clear:
        return pose
    return None


def _js_escape_hole(sim: Sim, cfg: dict, tid: str, pose: dict, closed: float) -> dict | None:
    """Inch the EE outward while holding. A 60 mm jump or JS pan peels the scoop."""
    del pose
    base = _arm_base_xy(cfg)
    lock = ("wrist_roll",)
    last = {n: sim.qpos()[n] for n in ARM_JOINTS}
    for _ in range(12):
        if not grasp_acquired(sim, tid):
            _move_q_hold(sim, last, closed, duration=0.2)
            return last if grasp_acquired(sim, tid) else None
        ee, _ = sim.site_pose(EE_SITE)
        r = float(np.linalg.norm(ee[:2] - base))
        # Far enough that +Z extract works; going all the way to 190 mm peels.
        if r >= 0.140:
            return {n: sim.qpos()[n] for n in ARM_JOINTS}
        away = _outward_xy(ee[:2], cfg, 0.006)
        tgt = [float(away[0]), float(away[1]), float(ee[2])]
        if not _move_hold(sim, tgt, closed, duration=0.28, n_wp=3, lock=lock, tid=tid):
            _move_q_hold(sim, last, closed, duration=0.2)
            return last if grasp_acquired(sim, tid) else None
        last = {n: sim.qpos()[n] for n in ARM_JOINTS}
    return last if grasp_acquired(sim, tid) else None


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


def _best_carry_goal(
    sim: Sim, tid: str, nest, z: float, lock: tuple[str, ...] = _WRIST_HOLD
) -> np.ndarray | None:
    """Locked-wrist step that puts the piece closer to the well."""
    cur, _ = sim.site_pose(EE_SITE)
    qpos, _ = sim.object_pose(tid)
    off = np.asarray(qpos[:2], dtype=float) - cur[:2]
    nest = np.asarray(nest, dtype=float)
    want_ee = np.array([nest[0] - off[0], nest[1] - off[1], z], dtype=float)
    cur_d = _piece_nest_dist(sim, tid, nest)
    best = None
    best_d = cur_d
    step_lim = 0.028 if cur_d > 0.08 else 0.018
    cands = _nest_step_candidates(cur, want_ee, z)
    for tgt in cands:
        if float(np.linalg.norm(tgt[:2] - cur[:2])) > step_lim:
            continue
        if _ik_goal(sim, tgt, lock=lock) is None:
            continue
        pred = tgt[:2] + off
        d = float(np.hypot(pred[0] - nest[0], pred[1] - nest[1]))
        if d < best_d - 0.002:
            best_d = d
            best = tgt
    return best


def _recover_descend(
    sim: Sim,
    cfg: dict,
    pinch,
    stem,
    axis,
    rec_open: float,
    lock_r: tuple[str, ...],
    out: dict,
) -> np.ndarray | None:
    """Reach the scoop pinch from a hover that actually IKs.

    The nominal 110 mm / 60 mm-back hover is dead next to the base; the pinch
    is not. Walk the same family as pick (safer poses first), then the other
    stem direction. Physics-tracks each kinematic hit; a failed move reseeds.
    """
    last_high = None
    for flip in (False, True):
        if flip:
            axis = _scoop_axis(stem, -1.0 if float(axis[0]) * float(stem[0]) >= 0 else 1.0)
            roll = _roll_horizontal_pinch(sim, pinch, axis)
            qroll = sim.qpos()
            qroll["wrist_roll"] = float(roll)
            qroll["gripper"] = rec_open
            move_qpos(sim, qroll, duration=0.30, physics=True)
            out["scoop"] = {
                "axis": [round(float(v), 3) for v in axis],
                "roll": round(float(roll), 3),
                "flipped": True,
            }
        roll = float(sim.qpos()["wrist_roll"])
        ikw = dict(x_axis=axis, ori_weight=0.45, ori_steps=8)

        def _reset():
            _reset_ik_seed(sim, cfg, keep={"wrist_roll": roll, "gripper": rec_open})

        if not pose_reaches(sim, pinch, lock_joints=lock_r, **ikw):
            continue
        hits = reachable_approaches(
            sim,
            ray_approach_candidates(pinch, axis[:2]),
            pinch,
            reset=_reset,
            lock_joints=lock_r,
            **ikw,
        )
        for high in hits:
            last_high = high
            with sim.probe():
                _reset()
            if not move_lin(sim, high, duration=0.9, physics=True, n_wp=6, lock_joints=lock_r, **ikw):
                continue
            log_state(sim, "recover_pregrasp", extra={"target": _xyz(high), "pinch": _xyz(pinch)})
            if move_lin(sim, pinch, duration=1.1, physics=True, n_wp=10, lock_joints=lock_r, **ikw):
                return np.asarray(axis, dtype=float)
    out["reason"] = "RECOVER_IK"
    log_state(
        sim,
        "fail_recover_pregrasp",
        extra={"target": _xyz(last_high if last_high is not None else pinch), "pinch": _xyz(pinch)},
    )
    return None


def recover_to_nest(sim: Sim, cfg: dict, tid: str) -> dict:
    """Approach along the lying stem, pinch across it, drop into the well."""
    p = pickup_cfg(cfg)
    surface = float(cfg.get("sim", {}).get("board", {}).get("surface_z", 0.02))
    release_open = float(p.get("gripper_release", 0.30))
    rec_open = min(float(p.get("gripper_open", 0.60)), _RECOVER_OPEN)
    geom = dict(PIECES.get(body_kind(tid) or tid) or {})
    closed_g = float(geom.get("gripper_closed", _RECOVER_CLOSED))
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
    sim.target_id = tid
    stem = _fallen_stem_xy(sim, tid)
    if stem is None:
        out["reason"] = "RECOVER_NO_STEM"
        return out
    com = _piece_com(sim, tid)
    pinch = _pinch_xyz(sim, tid, surface)
    axis = _choose_scoop_axis(sim, pinch, stem)
    roll = _roll_horizontal_pinch(sim, pinch, axis)
    out["scoop"] = {"axis": [round(float(v), 3) for v in axis], "roll": round(float(roll), 3)}

    qroll = sim.qpos()
    qroll["wrist_roll"] = float(roll)
    qroll["gripper"] = rec_open
    move_qpos(sim, qroll, duration=0.35, physics=True)
    lock_r = ("wrist_roll",)
    axis = _recover_descend(sim, cfg, pinch, stem, axis, rec_open, lock_r, out)
    if axis is None:
        return out
    ikw = dict(x_axis=axis, ori_weight=0.45, ori_steps=8)
    log_state(sim, "recover_grasp", extra={"pinch": _xyz(pinch), "com": _xyz(com)})
    scoop_q = {n: sim.qpos()[n] for n in ARM_JOINTS}
    _close_hold(sim, closed_g, steps=int(p.get("close_steps", 220)), hold=scoop_q)
    log_state(sim, "recover_closed")
    if not grasp_acquired(sim, tid):
        open_gripper(sim, rec_open, steps=40)
        pinch = _pinch_xyz(sim, tid, surface)
        axis = _scoop_axis(stem, -1.0 if float(axis[0]) * float(stem[0]) >= 0 else 1.0)
        ikw = dict(x_axis=axis, ori_weight=0.45, ori_steps=8)
        move_lin(sim, pinch, duration=0.8, physics=True, n_wp=8, lock_joints=lock_r, **ikw)
        scoop_q = {n: sim.qpos()[n] for n in ARM_JOINTS}
        _close_hold(sim, closed_g, steps=int(p.get("close_steps", 220)), hold=scoop_q)
        log_state(sim, "recover_closed")
    if not grasp_acquired(sim, tid):
        # COM pinch misses a thin stem. Try the fat pedestal.
        open_gripper(sim, rec_open, steps=40)
        pinch = _pinch_xyz_base(sim, tid, surface)
        axis = _choose_scoop_axis(sim, pinch, stem)
        roll = _roll_horizontal_pinch(sim, pinch, axis)
        qroll = sim.qpos()
        qroll["wrist_roll"] = float(roll)
        qroll["gripper"] = rec_open
        move_qpos(sim, qroll, duration=0.30, physics=True)
        base_out = {}
        axis2 = _recover_descend(sim, cfg, pinch, stem, axis, rec_open, lock_r, base_out)
        if axis2 is not None:
            axis = axis2
            scoop_q = {n: sim.qpos()[n] for n in ARM_JOINTS}
            _close_hold(sim, closed_g, steps=int(p.get("close_steps", 220)), hold=scoop_q)
            log_state(sim, "recover_closed", extra={"pinch": "base"})
    if not grasp_acquired(sim, tid):
        out["reason"] = "RECOVER_GRASP"
        log_state(sim, "fail_recover_grasp")
        return out
    # Pure +Z extract inside ~190 mm peels the scoop. Walk out first.
    gpos, _ = sim.site_pose(EE_SITE)
    if float(np.linalg.norm(gpos[:2] - _arm_base_xy(cfg))) < _WORKSPACE_R_M:
        escaped = _js_escape_hole(sim, cfg, tid, {n: sim.qpos()[n] for n in ARM_JOINTS}, closed_g)
        if escaped is None or not grasp_acquired(sim, tid):
            out["reason"] = "RECOVER_DROP"
            log_state(sim, "fail_recover_drop")
            return out
        scoop_q = escaped
    qpos, _ = sim.object_pose(tid)
    already_up = float(qpos[2]) > surface + 0.016 and grasp_acquired(sim, tid)
    hold = scoop_q if already_up else _js_extract(sim, tid, scoop_q, closed_g, surface)
    if hold is None:
        out["reason"] = "RECOVER_DROP"
        log_state(sim, "fail_recover_drop")
        return out
    qpos, _ = sim.object_pose(tid)
    log_state(sim, "recover_extract", extra={"xyz": _xyz(qpos)})

    gpos, _ = sim.site_pose(EE_SITE)
    # Hole scoop is a low wrap. Climbing to the usual 80 mm hover peels it
    # whether we skipped extract or just finished it.
    in_hole = float(np.linalg.norm(gpos[:2] - _arm_base_xy(cfg))) < _WORKSPACE_R_M
    if in_hole:
        z_carry = max(float(gpos[2]), 0.040)
    else:
        z_carry = max(float(gpos[2]), 0.055, float(nest[2]) + 0.06)
    if float(gpos[2]) < z_carry - 0.004:
        up = [float(gpos[0]), float(gpos[1]), z_carry]
        if not _move_hold(sim, up, closed_g, duration=0.7, n_wp=5, lock=_WRIST_HOLD, tid=tid):
            if not _move_hold(sim, up, closed_g, duration=0.7, n_wp=5, lock=("wrist_roll",), tid=tid):
                out["reason"] = "RECOVER_DROP" if not grasp_acquired(sim, tid) else "RECOVER_LIFT"
                log_state(sim, "fail_recover_drop")
                return out
    for _ in range(70):
        if not grasp_acquired(sim, tid):
            out["reason"] = "RECOVER_DROP"
            log_state(sim, "fail_recover_transit")
            return out
        if _piece_nest_dist(sim, tid, nest) <= 0.010:
            break
        nxt = _best_carry_goal(sim, tid, nest, z_carry)
        if nxt is None:
            break
        if not _move_hold(sim, nxt, closed_g, duration=0.45, n_wp=3, lock=_WRIST_HOLD, tid=tid):
            if not _move_hold(
                sim, nxt, closed_g, duration=0.5, n_wp=3, lock=("wrist_roll",), tid=tid, require_grasp=False
            ):
                out["reason"] = "RECOVER_DROP" if not grasp_acquired(sim, tid) else "RECOVER_TRANSIT"
                log_state(sim, "fail_recover_transit", extra={"target": _xyz(nxt)})
                return out
            if not grasp_acquired(sim, tid):
                out["reason"] = "RECOVER_DROP"
                log_state(sim, "fail_recover_transit")
                return out
            continue
        qpos, _ = sim.object_pose(tid)
        if float(qpos[2]) < surface + 0.008:
            out["reason"] = "RECOVER_DROP"
            log_state(sim, "fail_recover_drop")
            return out
    dist = _piece_nest_dist(sim, tid, nest)
    if dist > 0.018:
        out["reason"] = "RECOVER_TRANSIT"
        log_state(sim, "fail_recover_transit", extra={"piece_xy_mm": round(dist * 1000.0, 1)})
        return out
    # Center the held man over the hole, then seat on the lip. Carry steps
    # can stop 15 mm off-axis (bishop) where a 28 mm cup cannot stand it.
    qpos, _ = sim.object_pose(tid)
    ee, _ = sim.site_pose(EE_SITE)
    off = np.asarray(qpos[:2], dtype=float) - np.asarray(ee[:2], dtype=float)
    mid_z = max(float(ee[2]), float(nest[2]) + 0.028)
    center = [float(nest[0] - off[0]), float(nest[1] - off[1]), mid_z]
    if float(np.hypot(qpos[0] - nest[0], qpos[1] - nest[1])) > 0.006:
        if not _move_hold(sim, center, closed_g, duration=0.6, n_wp=5, lock=_WRIST_HOLD, tid=tid):
            _move_hold(sim, center, closed_g, duration=0.6, n_wp=5, lock=("wrist_roll",), tid=tid, require_grasp=False)
        if not grasp_acquired(sim, tid):
            out["reason"] = "RECOVER_DROP"
            log_state(sim, "fail_recover_drop")
            return out
    # Seat on the lip while still holding, then let go. Opening 8 cm up
    # throws the man past a 28 mm hole; the real arm has the same problem.
    lip_z = float(nest[2]) + 0.016
    for _ in range(14):
        qpos, _ = sim.object_pose(tid)
        ee, _ = sim.site_pose(EE_SITE)
        if float(qpos[2]) <= lip_z + 0.006 or float(ee[2]) <= lip_z + 0.004:
            break
        down = [float(ee[0]), float(ee[1]), max(lip_z, float(ee[2]) - 0.008)]
        if not _move_hold(sim, down, closed_g, duration=0.35, n_wp=3, lock=_WRIST_HOLD, tid=tid, require_grasp=False):
            if not _move_hold(
                sim, down, closed_g, duration=0.35, n_wp=3, lock=("wrist_roll",), tid=tid, require_grasp=False
            ):
                break
    log_state(sim, "recover_insert", extra={"target": _xyz(nest), "piece_xy_mm": round(_piece_nest_dist(sim, tid, nest) * 1000.0, 1)})
    arm = {n: sim.qpos()[n] for n in ARM_JOINTS}
    g0 = float(sim.qpos()["gripper"])
    for i in range(110):
        u = (i + 1) / 110
        cmd = dict(arm)
        cmd["gripper"] = g0 + (release_open - g0) * u
        sim.set_ctrl(cmd)
        sim.step()
    open_gripper(sim, max(release_open, rec_open), steps=60)
    settle_queen(sim, steps=240)
    # Leave the well on +Z after the piece has settled. An XY pad slide
    # hits the walls; a hop while the pads still wrap lifts it out (Nf6
    # ended airborne at 85 mm, 39 deg, then failed the from-nest place).
    if grasp_acquired(sim, tid):
        open_gripper(sim, max(release_open, rec_open, 0.45), steps=50)
        settle_queen(sim, steps=80)
    ee, _ = sim.site_pose(EE_SITE)
    if not grasp_acquired(sim, tid):
        move_lin(
            sim,
            [float(ee[0]), float(ee[1]), float(ee[2]) + 0.05],
            duration=0.6,
            physics=True,
            n_wp=5,
            lock_joints=_WRIST_HOLD,
        )
        settle_queen(sim, steps=160)
    else:
        settle_queen(sim, steps=80)
    qpos, quat = sim.object_pose(tid)
    out["final_xyz"] = _xyz(qpos)
    out["tilt_deg"] = round(queen_tilt_deg(quat), 1)
    seated = in_well(qpos, cfg) and float(qpos[2]) <= float(nest[2]) + 0.040
    if (not seated) or (not queen_upright(quat, tilt_max=0.70)):
        out["reason"] = "RECOVER_STILL_DOWN"
        log_state(
            sim,
            "fail_recover_down",
            extra={"tilt_deg": out["tilt_deg"], "seated": bool(seated), "z_mm": round(float(qpos[2]) * 1000, 1)},
        )
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
    if int(sc.get("_recover_attempt", 0)) >= 2:
        return None
    dest = sc.get("dest")
    if (
        dest
        and sc.get("place_xyz") is None
        and square_of_body(sim, cfg, tid) == dest
        and object_on_board(sim, surface, object_id=tid)
    ):
        # On dest already. Nest scoop from f6 hits f7 (Opera Nf6).
        return None
    if _piece_still_upright(sim, tid, surface, tilt_max=0.5):
        return None
    qpos, _ = sim.object_pose(tid)
    if float(qpos[2]) < surface - 0.02:
        return None
    if not pickup_cfg(cfg).get("recover_fallen", False):
        failed["fallen_after"] = failed.get("failure_reason")
        failed["failure_reason"] = "PIECE_FALLEN"
        return None
    log_state(sim, "recover_start", extra={"after": failed.get("failure_reason")})
    rec = {"ok": False}
    for _try in range(2):
        rec = recover_to_nest(sim, cfg, tid)
        if rec.get("ok"):
            break
        if rec.get("reason") in ("RECOVER_OFF_TABLE", "RECOVER_NO_STEM"):
            break
        log_state(sim, "recover_retry", extra={"try": _try + 1, "reason": rec.get("reason")})
    failed["recover"] = rec
    if not rec.get("ok"):
        failed["failure_reason"] = rec.get("reason") or "RECOVER_FAIL"
        return None
    nxt = dict(sc)
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
    dest = sc.get("dest")
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
        nxt = dict(sc)
        nxt["seed_home"] = True
        nxt["reset_board"] = False
        nxt["_from_nest"] = False
        nxt["_recover_attempt"] = int(sc.get("_recover_attempt", 0)) + 1
        nxt["_grasp_attempt"] = 0
        out = execute_move(sim, cfg, use_vision=use_vision, spawn_xy=None, scenario=nxt)
        out["replaced_after_drop"] = True
        out["first_grasp_reason"] = reason
        return out
    return _recover_fallen_then_finish(
        sim, cfg, use_vision, sc, failed, tid, surface, release_open
    )


def _after_place_fail(
    sim: Sim,
    cfg: dict,
    use_vision: bool,
    sc: dict,
    failed: dict,
    tid: str,
    surface: float,
    release_open: float,
) -> dict | None:
    """Fallen after put-down: scoop into the well, stand up, place again.

    Same path for every piece. Bowl dumps stay down on purpose.
    """
    if sc.get("place_xyz") is not None:
        return None
    if failed.get("failure_reason") == "RELEASE_FAILURE":
        return None
    if _perceiving(sim) and tid not in sim.known:
        from perceive import perceive_board

        perceive_board(sim, cfg)
        if tid not in sim.known:
            failed["failure_reason"] = "PIECE_LOST"
            failed["failure_stage"] = "place"
            return None
    return _recover_fallen_then_finish(
        sim, cfg, use_vision, sc, failed, tid, surface, release_open
    )


def grasp_acquired(sim: Sim, object_id: str | None = None) -> bool:
    """Jaws are squeezing something: gripper servo load and speed, the only grasp sensors on the arm."""
    return jaw_squeezing(sim)


def _jaws_on_piece(sim: Sim) -> bool:
    return _piece_in_jaws(sim) is not None


def _piece_in_jaws(sim: Sim) -> str | None:
    if not grasp_acquired(sim):
        return None
    return getattr(sim, "held", None) or _tid(sim)


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
    sim = world_of(sim)
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


def find_observe_poses(sim: Sim, cfg: dict, target_xyz) -> list[list[float]]:
    """Reachable wrist poses that still see target_xyz, safest first."""
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
    return reachable_approaches(
        sim,
        cands,
        reset=lambda: _reset_ik_seed(sim, cfg),
        upright=True,
        accept=lambda _c: wrist_sees_point(sim, target),
    )


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
    return _try(find_observe_poses(sim, cfg, target_xyz), 0.9)


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
        seen = observe_piece(obs, cfg, square, grow=1.0)
        height = float((PIECES.get(body_kind(tid) or tid) or {}).get("height", 0.04))
        hidden = _own_hand_hides(sim, square_to_world(square, cfg, z=surface))
        upright = seen is None or hidden or seen["top_h"] > VERIFY_UPRIGHT_FRAC * height
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


def _belief_set_down(sim, tid: str, surface: float) -> None:
    """Released piece: belief stands it on its base under the jaws (a placed man settles upright)."""
    if not _perceiving(sim) or tid not in sim.known:
        return
    p, q = sim.object_pose(tid)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    x = R.reshape(3, 3)[:, 0]
    yaw = float(np.arctan2(x[1], x[0]))
    sim.believe(tid, [float(p[0]), float(p[1]), surface], (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)))


def _placed(sim, cfg: dict, tid: str, dest: str, dest_xy, surface: float, off_board: bool) -> bool:
    """Is the released piece where it should be? The wrist camera judges when perceiving."""
    if off_board:
        return bool(bowl_contains(sim.object_pose(tid)[0], dest_xy))
    if _perceiving(sim):
        look = verify_piece(sim, cfg, tid, dest)
        return bool(look.get("seen") and look.get("upright") and square_of_body(sim, cfg, tid) == dest)
    _, quat = sim.object_pose(tid)
    return bool(
        square_of_body(sim, cfg, tid) == dest
        and queen_upright(quat, tilt_max=0.5)
        and object_on_board(sim, surface, object_id=tid)
    )


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
    seen = observe_piece(capture_wrist(sim), cfg, square)
    if seen is None:
        sim.forget(tid)
        return {"looked": True, "seen": False}
    height = float((PIECES.get(body_kind(tid) or tid) or {}).get("height", 0.04))
    upright = seen["top_h"] > VERIFY_UPRIGHT_FRAC * height
    quat = (1.0, 0.0, 0.0, 0.0) if upright else (0.7071068, 0.7071068, 0.0, 0.0)
    sim.believe(tid, [seen["xy"][0], seen["xy"][1], surface], quat)
    return {"looked": True, "seen": True, "upright": upright, "top_h_mm": round(seen["top_h"] * 1000, 1), "xy": seen["xy"]}


VERIFY_UPRIGHT_FRAC = 0.7


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
    xy_tol = float(p.get("place_xy_tol", 0.018))
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
    gpos, _ = sim.site_pose(EE_SITE)
    hugging = _jaws_on_piece(sim)
    sim._pickup_phase = "000_unhook"
    if hugging:
        # Pads still wrap the last man. Open to release, then the same
        # clear-then-rise as after place — a raw +Z here combs the lip.
        wrap = _piece_in_jaws(sim)
        open_gripper(sim, release_open, steps=80)
        settle_queen(sim, steps=24)
        if wrap:
            _clear_then_rise(sim, cfg, wrap, z_min=0.11)
        else:
            _js_raise_ee(sim, z_min=0.11)
        open_gripper(sim, open_g, steps=40)
    elif float(gpos[2]) < 0.09:
        # Clear of the last man: step out, then up.
        away = _outward_xy(gpos[:2], cfg, 0.035)
        wrist = ("wrist_flex", "wrist_roll")
        others = obstacle_bodies(sim, skip={tid})
        _safe_lin(
            sim,
            [float(away[0]), float(away[1]), float(gpos[2])],
            cfg,
            wrist,
            obstacles=others,
            skip={tid},
            duration=0.45,
        )
        _js_raise_ee(sim, z_min=0.11)
        open_gripper(sim, open_g, steps=40)
    else:
        open_gripper(sim, open_g, steps=40)
    settle_queen(sim, steps=48)
    _collateral_snapshot()

    actual, _ = W.object_pose(tid)
    result["actual_start_pose"] = actual.tolist()
    result["queen_start_xyz"] = actual.tolist()
    log_state(sim, "001_initial", extra={"spawn": _xyz(spawn)})

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
            again = _after_pick_fail(
                sim, cfg, use_vision, sc, result, tid, surface, release_open
            )
            return again if again is not None else _done()
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
            again = _after_pick_fail(
                sim, cfg, use_vision, sc, result, tid, surface, release_open
            )
            return again if again is not None else _done()
        log_state(sim, "002_observe", extra={"target": _xyz(observe)})
        queen_xyz, _ = sim.object_pose(tid)
    plan_xy = perceived_xy(sim, queen_xyz)
    # Vertical pinch: put the wrist where gripper +X is world-down at the
    # grasp point before anything downstream snapshots the heading. Without
    # this the observe pose's 30-50 deg flex was inherited (approach planner
    # locks flex at its current value) and the fixed jaw tipped pieces over on
    # close. Files a-f reach <3 deg; g/h get the least-tilted pose available.
    vert = _vertical_wrist(sim, [plan_xy[0], plan_xy[1], surface + grasp_z])
    if vert is not None:
        result["vertical_flex"] = round(vert["wrist_flex"], 4)
        result["vertical_tilt_deg"] = round(vert["tilt_deg"], 2)
        _pitch_in_place(sim, vert["wrist_flex"])
    grasp_xy = _seeded_grasp_xy(sim, cfg, plan_xy, geom, surface + grasp_z)
    pre = find_pregrasp_pose(sim, cfg, grasp_xy)
    if pre is None:
        result["failure_reason"] = "IK_NO_SOLUTION"
        result["ee_requested"] = _xyz([grasp_xy[0], grasp_xy[1], surface + pre_z])
        log_state(sim, "fail_pregrasp", extra={"target": result["ee_requested"]})
        again = _after_pick_fail(
            sim, cfg, use_vision, sc, result, tid, surface, release_open
        )
        return again if again is not None else _done()
    result["transport_path"] = "straight"
    clutter = [n for n in obstacle_bodies(sim, skip={tid}) if n != "board"]
    result["occupied_neighbors"] = sorted(clutter)
    grasp_plan = [grasp_xy[0], grasp_xy[1], surface + grasp_z]

    def _grasp_for_roll(roll: float | None) -> list[float]:
        # The pinch axis rotates with wrist roll, so the fixed-pad shift does too.
        xy = _grasp_xy_for_roll(sim, plan_xy, surface + grasp_z, geom, roll)
        return [xy[0], xy[1], surface + grasp_z]

    def _plan(pre_seed, grasp_seed):
        return plan_approach(
            sim,
            pre_seed,
            grasp_seed,
            open_g,
            clutter,
            source=None if sc.get("_from_nest") else source,
            cfg=cfg,
            scenario=sc,
            grasp_for_roll=_grasp_for_roll,
            target=tid,
        )

    approach = _plan(pre, grasp_plan)
    if approach.get("pre"):
        pre = list(approach["pre"])
    if approach.get("grasp"):
        grasp_plan = list(approach["grasp"])
        grasp_xy = grasp_plan[:2]
    _plan_keys = ("ok", "roll", "gripper", "n_wp", "tried", "reachable", "rank", "reason", "pre_offset_xy", "block", "statuses", "degraded", "min_gap_mm")
    result["approach_plan"] = {k: approach.get(k) for k in _plan_keys}
    if not approach["ok"]:
        result["failure_reason"] = (
            "NO_COLLISION_FREE_APPROACH" if approach.get("reachable") else "IK_NO_SOLUTION"
        )
        log_state(sim, "fail_pregrasp", extra={"target": _xyz(pre), "approach": result["approach_plan"]})
        again = _after_pick_fail(
            sim, cfg, use_vision, sc, result, tid, surface, release_open
        )
        return again if again is not None else _done()
    lock: tuple[str, ...] = tuple(approach["lock_joints"])
    # After close, lock wrist_roll so the pinch does not yaw around the stem.
    # Leave wrist_flex free: carry IK pulls gripper +X toward world -Z so the
    # piece stays upright as the shoulder/elbow reach across the board.
    hold: tuple[str, ...] = tuple(
        j for j in dict.fromkeys((*lock, "wrist_roll")) if j != "wrist_flex"
    )
    pickup_clear = bool(clutter) or approach["n_wp"] > 1
    # Verified joint path for the descend (pre -> grasp); replayed verbatim.
    descend_path = approach.get("path_q") if clutter else None
    # Vertical pinch: pre, descend and grasp are all solved with gripper +X
    # world-down (solve_ik_down); flex is a free joint the solver uses to keep
    # the approach vertical while shoulder/elbow move. Roll stays locked when
    # the planner chose one for neighbor clearance.
    grasp_lock = tuple(j for j in lock if j != "wrist_flex")

    def _apply_approach(plan: dict) -> bool:
        """Set the planned roll + aperture, then bring the EE to its pregrasp."""
        if plan["roll"] is not None:
            _roll_in_place(sim, float(plan["roll"]))
            result["approach_roll_err"] = round(abs(float(sim.qpos()["wrist_roll"]) - float(plan["roll"])), 3)
        g_ap = float(plan["gripper"])
        result["approach_gripper"] = None if abs(g_ap - open_g) < 0.02 else g_ap
        if abs(g_ap - float(sim.qpos()["gripper"])) > 0.01:
            open_gripper(sim, g_ap, steps=40)
        # A straight line to the pregrasp from the last square (g4) sweeps the
        # file and knocks whoever is in the way (fallen Nf3, then the e5 pawn
        # → APPROACH_COLLISION). Hover over the target first so the pregrasp
        # move is a short local drop.
        gpos, _ = sim.site_pose(EE_SITE)
        pre_xyz = np.asarray(plan["pre"], dtype=float)
        if float(np.hypot(pre_xyz[0] - gpos[0], pre_xyz[1] - gpos[1])) > 0.05:
            hover_z = max(float(pre_xyz[2]), _transit_z(sim, cfg, skip={tid}))
            hover = [float(pre_xyz[0]), float(pre_xyz[1]), hover_z]
            _safe_lin(
                sim,
                hover,
                cfg,
                ("wrist_flex", "wrist_roll"),
                obstacles=clutter,
                skip={tid},
            )
        if pickup_clear:
            return move_lin(sim, plan["pre"], duration=1.0, physics=True, n_wp=6, lock_joints=grasp_lock, down=True)
        return move_ik(sim, plan["pre"], duration=1.0, physics=True, lock_joints=grasp_lock, down=True)

    if not _apply_approach(approach):
        result["failure_reason"] = "IK_NO_SOLUTION"
        log_state(sim, "fail_pregrasp", extra={"target": _xyz(pre)})
        again = _after_pick_fail(
            sim, cfg, use_vision, sc, result, tid, surface, release_open
        )
        return again if again is not None else _done()
    result["pregrasp_reached"] = True
    log_state(sim, "003_pregrasp", extra={"target": _xyz(pre), "approach_gripper": result.get("approach_gripper"), "approach_roll": approach.get("roll")})

    if not queen_upright(sim.object_pose(tid)[1]):
        result["failure_reason"] = "APPROACH_COLLISION"
        log_state(sim, "fail_pregrasp_knock")
        again = _after_pick_fail(
            sim, cfg, use_vision, sc, result, tid, surface, release_open
        )
        return again if again is not None else _done()

    if use_vision and not sc.get("_from_nest"):
        refined = locate_queen(sim, cfg, result, source)
        if refined is not None:
            queen_xyz = refined
            plan_xy = perceived_xy(sim, queen_xyz)
            grasp_xy = _grasp_xy_for_roll(sim, plan_xy, surface + grasp_z, geom, approach.get("roll"))
            # The plan was verified against the long-range estimate. Re-check
            # the chosen approach on the refined grasp; if the jaws would now
            # clip a bystander, re-plan from here before going down.
            if clutter:
                status = verify_approach(
                    sim, pre, [grasp_xy[0], grasp_xy[1], surface + grasp_z], clutter,
                    float(approach["gripper"]), approach.get("roll"), lock, cfg=cfg, target=tid,
                    margin=HARD_MARGIN if approach.get("degraded") else None,
                )
                result["approach_reverify"] = status
                if status == "ok":
                    descend_path = getattr(sim, "_cand_path", None) or descend_path
                if status != "ok":
                    again = _plan(pre, [grasp_xy[0], grasp_xy[1], surface + grasp_z])
                    result["approach_replan"] = {k: again.get(k) for k in _plan_keys}
                    if not again["ok"]:
                        result["failure_reason"] = (
                            "NO_COLLISION_FREE_APPROACH" if again.get("reachable") else "IK_NO_SOLUTION"
                        )
                        log_state(sim, "fail_pregrasp", extra={"target": _xyz(pre), "approach": result["approach_replan"]})
                        return _done()
                    approach = again
                    lock = tuple(approach["lock_joints"])
                    hold = tuple(j for j in dict.fromkeys((*lock, "wrist_roll")) if j != "wrist_flex")
                    grasp_lock = tuple(j for j in lock if j != "wrist_flex")
                    pre = list(approach["pre"])
                    grasp_xy = list(approach["grasp"][:2])
                    descend_path = approach.get("path_q")
                    result["approach_plan"] = {k: approach.get(k) for k in _plan_keys}
                    if not _apply_approach(approach):
                        result["failure_reason"] = "IK_NO_SOLUTION"
                        log_state(sim, "fail_pregrasp", extra={"target": _xyz(pre)})
                        return _done()
                    log_state(sim, "003_pregrasp", extra={"target": _xyz(pre), "replanned": True})

    result["pose_used_for_grasp"] = [float(plan_xy[0]), float(plan_xy[1]), float(np.asarray(queen_xyz)[2])]
    result["perception_error"] = [
        float((plan_xy[0] - actual[0]) * 1000.0),
        float((plan_xy[1] - actual[1]) * 1000.0),
    ]
    grasp = [grasp_xy[0], grasp_xy[1], surface + grasp_z]
    result["ee_requested"] = _xyz(grasp)
    result["transport_path"] = "straight"
    column = _descend_column(sim, grasp, grasp_lock)
    result["descend_column"] = column is not None or getattr(sim, "plumb_fail", None)
    desc = physical_descend(
        sim, grasp, duration=1.2, lock_joints=grasp_lock, path_q=column or descend_path
    )
    # Guard tripped: a robot link touched a bystander on the way down.
    if desc.get("aborted"):
        result["descend_aborted"] = desc["aborted"]
        move_lin(sim, pre, duration=0.6, physics=True, n_wp=4, lock_joints=grasp_lock, down=True)
        result["failure_reason"] = "APPROACH_COLLISION"
        log_state(sim, "fail_descend", extra={"target": _xyz(grasp), "aborted": desc["aborted"]})
        return _done()
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
        grip_peak[0] = max(grip_peak[0], abs(float(sim.actuator_forces()[sim._act["gripper"]])))

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
        _transit_hook()

    sim.on_step = _on_step
    plumbed = None
    if not off_board and not sc.get("_from_nest") and sc.get("stop_after") != "lift":
        plumbed = _plumb_transport(sim, tid, result, dest_xy, surface + place_hover, lift_z, obstacles)
    if plumbed is not None and plumbed.get("failure_reason"):
        result["failure_reason"] = plumbed["failure_reason"]
        again = _after_pick_fail(sim, cfg, use_vision, sc, result, tid, surface, release_open)
        return again if again is not None else _done()
    if plumbed is not None:
        place = plumbed["place"]
    else:
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

        def _lift(lock: tuple[str, ...], down: bool = False, x_axis=None) -> bool:
            return move_lin(sim, lift, duration=1.2, physics=True, n_wp=8, lock_joints=lock, down=down, x_axis=x_axis)

        # Vertical extract: the pinch is a hinge, so the piece hangs plumb; keeping
        # gripper +X world-down while rising is what stops it peeling. A locked
        # flex rotated the gripper up to 40 deg over a 30 mm lift near the base.
        # If the grasp itself was not vertical (near-base squares, g/h files),
        # freeze the *current* approach axis instead: re-verticalising while low
        # swung the jaws 45 deg through the rank above (rook a1 knocked pawn a2).
        _, R_g = sim.site_pose(EE_SITE)
        grasp_axis = np.asarray(R_g, dtype=float).reshape(3, 3)[:, 0]
        keep_axis = grasp_axis if gripper_tilt_deg(sim) > 6.0 else None
        result["lift_axis_frozen"] = keep_axis is not None
        lift_ok = _lift(tuple(hold), down=True, x_axis=keep_axis)
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

        # A two-pad pinch is a hinge about the pinch axis. Once that axis has a
        # vertical component (wrist at its flex limit on rank 7 + a roll chosen
        # for clearance) the piece rolls on the pads and creeps out over the
        # carry (e7e5 dropped after 60 mm). Roll the wrist so the axis is level;
        # the piece only yaws.
        # Leveling the pinch peels a short carry (g4→f3). Keep it for the
        # long rank-7 transits it was added for.
        carry_xy = float(np.linalg.norm(dest_xy[:2] - object_xy))
        # Bowl dumps are a long carry, but orientation does not matter and
        # leveling the pinch peels L-shaped men off the pads (Opera Bxf3).
        if carry_xy > 0.08 and not off_board:
            result["carry_level_roll_deg"] = round(_level_pinch_axis(sim), 2)
        else:
            result["carry_level_roll_deg"] = 0.0
        if not grasp_acquired(sim, tid):
            result["failure_reason"] = "LIFT_DROP"
            result["lift_success"] = False
            again = _after_pick_fail(
                sim, cfg, use_vision, sc, result, tid, surface, release_open
            )
            return again if again is not None else _done()

        gpos, _ = sim.site_pose(EE_SITE)
        # Translate the believed object XY onto dest. Using the EE grasp point here
        # would park the jaws on dest and leave the piece short by grasp_xy_offset.
        delta = dest_xy[:2] - object_xy
        dest_ee = np.array([float(gpos[0] + delta[0]), float(gpos[1] + delta[1])])
        _, standoff = _xy_floor(dest_ee, cfg)
        result["transport_standoff"] = bool(standoff)
        z_ee, derived = safe_transit_ee_z(sim, float(gpos[2]), tid, obstacles)
        if off_board:
            # Extra hover so a long bowl carry does not kiss the file it just left.
            z_ee = max(float(z_ee), float(gpos[2]) + 0.03)
        result["transit_z"] = z_ee
        result["transit_z_derived"] = derived
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
        for _ in range(30):
            sim.step()
        qpos, qquat = sim.object_pose(tid)
        over_dest = float(np.hypot(qpos[0] - dest_xy[0], qpos[1] - dest_xy[1])) < 0.04
        # Near-base dests carry low (the 10 cm hover is dead). 40 mm clearance
        # then called a held piece on dest a drop and scooped it into the cup.
        held_over = bool(
            grasp_acquired(sim, tid)
            and (float(qpos[2]) > surface + 0.012 or over_dest)
        )
        result["transport_success"] = bool(held_over and over_dest and not result["transport_contact_transit"])
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
            again = _after_pick_fail(
                sim, cfg, use_vision, sc, result, tid, surface, release_open
            )
            return again if again is not None else _done()

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
        place_z = float(g_close[2]) + place_hover
        standoff = bool(result.get("transport_standoff"))
        # Least-tilted reachable pinch at this hover, then lock and translate.
        # Re-solving vertical *during* the lower swings the jaws (f1c4). Far
        # squares cannot be truly vertical; this is the same compromise the
        # real wrist will have to make.
        gpos, _ = sim.site_pose(EE_SITE)
        hover = [float(gpos[0]), float(gpos[1]), float(gpos[2])]
        vert_place = _vertical_wrist(sim, hover)
        if vert_place is not None:
            _pitch_holding(sim, vert_place["wrist_flex"], closed_g, tid)
            result["place_hover_tilt_deg"] = round(vert_place["tilt_deg"], 2)
        _, R_place = sim.site_pose(EE_SITE)
        place_axis = np.asarray(R_place, dtype=float).reshape(3, 3)[:, 0]
        keep_place = place_axis if gripper_tilt_deg(sim) > 6.0 else None
        # Destination clearance: which side the moving jaw faces (roll, level
        # pinch either way) and how wide to open without slapping a neighbour.
        place_plan = _plan_place(sim, tid, dest_xy, place_z, hold, release_open, x_axis=keep_place)
        result["place_plan"] = place_plan
        d_roll = abs(float(place_plan["roll"]) - float(sim.qpos()["wrist_roll"]))
        if 0.05 < d_roll < 1.0:
            _roll_holding(sim, float(place_plan["roll"]))
        # If the grasp was not already world-down, freeze that axis so the jaws
        # translate instead of swinging the piece onto its side (f1c4).
        vertical_place = _lower_vertical(sim, tid, dest_xy, place_z, hold, result, x_axis=keep_place)
        if vertical_place:
            place = vertical_place
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
            lower_ok = _carry_lin(sim, place, 1.2, 12, hold, extract)
            if not lower_ok:
                result["failure_reason"] = "IK_FAILURE"
                result["ik_fail_phase"] = "008_lower"
                log_state(sim, "fail_lower", extra={"target": _xyz(place)})
                return _done()
        q_lo, _ = sim.object_pose(tid)
        off = float(np.hypot(q_lo[0] - dest_xy[0], q_lo[1] - dest_xy[1]))
        if standoff and 0.012 < off < 0.022:
            gpos, _ = sim.site_pose(EE_SITE)
            shift = dest_xy[:2] - q_lo[:2]
            inch = [float(gpos[0] + shift[0]), float(gpos[1] + shift[1]), float(gpos[2])]
            if _carry_lin(sim, inch, 0.8, 8, hold, extract):
                place = inch
                result["place_diag"]["standoff_inch"] = True
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
    _belief_set_down(sim, tid, surface)
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
    result["place_diag"]["airborne_after_open"] = float(q_open[2]) > surface + 0.012
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
    else:
        seated = object_on_board(sim, surface, object_id=tid)
        result["place_diag"]["tilt_deg"] = round(float(queen_tilt_deg(qquat_rel)), 1)
        upright = queen_upright(qquat_rel, tilt_max=0.5)
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
    z_up = max(
        float(gpos[2]) + release_lift,
        float(result.get("transit_z") or 0.0),
        _transit_z(sim, cfg, skip={tid}),
    )
    high = [float(gpos[0]), float(gpos[1]), z_up]
    retreated = True
    near_dest = float(np.hypot(q_rel[0] - dest_xy[0], q_rel[1] - dest_xy[1])) < 0.028
    straight_up = None
    if not off_board:
        spec = PIECES.get(body_kind(tid) or tid) or {}
        jaw_clear = float(spec.get("height", 0.055)) - float(spec.get("preferred_grasp_height", 0.01)) + 0.01
        avoid = [n for n in obstacle_bodies(sim, skip={tid}) if n != "board"]
        bearing = _hand_bears_on(sim, tid)
        result["place_diag"]["hand_bears_on_piece"] = bearing
        centred = {"ok": False, "why": "bearing"} if bearing else _centred_retreat(sim, tid, avoid, z_up, jaw_clear)
        result["place_diag"]["retreat_centred"] = centred
        if centred["ok"]:
            straight_up = []
            if not centred["high"]:
                g_top, _ = sim.site_pose(EE_SITE)
                _safe_lin(sim, [float(g_top[0]), float(g_top[1]), z_up], cfg, wrist,
                          obstacles=obstacle_bodies(sim, skip={tid}), skip={tid})
        else:
            straight_up = plumb.retreat(sim, avoid, z_up, jaw_clear, touching=lambda: bystander_contact(sim, [tid]))
        result["place_diag"]["retreat_straight"] = straight_up is not None or getattr(sim, "plumb_fail", None)
    if straight_up is not None:
        retreated = True
    elif (seated or (upright and near_dest)) and not off_board:
        # Get the pads off the lip, then rise. Skipping that check (or
        # `_safe_lin(skip=tid)`) combs the crown and knocks the piece.
        # "Seated" needs board contact; a man still on the pads is upright
        # on dest and must take this path, not a floored +Z.
        leave = _clear_then_rise(sim, cfg, tid, z_min=z_up)
        result["place_diag"]["pad_clear_mm"] = leave.get("pad_clear_mm", 0.0)
        result["place_diag"]["retreat_cleared"] = bool(leave.get("cleared"))
        result["place_diag"]["short_retreat"] = not bool(leave.get("rose"))
        retreated = True
    elif float(gpos[2]) < z_up - 0.01:
        retreated = _safe_lin(
            sim, high, cfg, wrist, obstacles=obstacle_bodies(sim, skip={tid}), skip={tid}
        )
    settle_queen(sim, steps=48)
    still_held = bool(grasp_acquired(sim, tid) and float(sim.object_pose(tid)[0][2]) > surface + 0.012)
    result["placement_success"] = _placed(sim, cfg, tid, dest, dest_xy, surface, off_board) and not still_held
    result["retreat_success"] = bool(retreated)
    if not retreated:
        result["ik_fail_phase"] = "010_final"
        log_state(sim, "fail_retreat", extra={"target": _xyz(high)})
    log_state(sim, "010_final")

    qpos, qquat = W.object_pose(tid)
    result["final_position_error_mm"] = float(np.hypot(qpos[0] - dest_xy[0], qpos[1] - dest_xy[1])) * 1000
    result["final_queen_xyz"] = qpos.tolist()
    result["place_diag"]["final_err_mm"] = round(result["final_position_error_mm"], 1)
    result["place_diag"]["final_upright"] = bool(queen_upright(qquat, tilt_max=0.5))
    if not result["placement_success"]:
        result["failure_reason"] = "RELEASE_FAILURE" if still_held else "PLACEMENT_POSITION_ERROR"
        again = _after_place_fail(
            sim, cfg, use_vision, sc, result, tid, surface, release_open
        )
        return again if again is not None else _done()
    result["success"] = True
    result["failure_reason"] = None
    return _done()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--vision", action="store_true", help="wrist RGB-D pose (the default; kept for old commands)")
    parser.add_argument("--gt", action="store_true", help="belief pose instead of a wrist look (truth only under --wall cheat/off)")
    parser.add_argument("--full-board", action="store_true", help="load starting-position chess_scene_full.xml")
    parser.add_argument("--random", action="store_true", help="spawn queen at a random on-board XY, not a square")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", action="store_true", help="write overview+side mp4 under sim/debug/pickup/")
    parser.add_argument("--wall", choices=WALLS, default="real", help="controller/world separation (robot.py)")
    args = parser.parse_args()
    args.vision = not args.gt
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
    from episode import run_pickup, setup_board

    world = Sim(xml_path=xml, cfg=cfg)
    setup_board(world, cfg, sc or {}, spawn_xy=spawn_xy)
    sim = open_robot(world, args.wall)
    rec = None
    if args.video:
        tag = "vision" if args.vision else "gt"
        if args.random:
            tag += f"_random{args.seed}"
        rec = VideoRecorder(sim, DEBUG_DIR / "pickup" / f"{tag}.mp4")
        sim.on_step = rec.grab
    try:
        result = run_pickup(sim, cfg, use_vision=args.vision, spawn_xy=spawn_xy, scenario=sc)
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
