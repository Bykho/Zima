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

from ik import EE_SITE, move_ik, move_lin, solve_ik
from mj import ARM_JOINTS, DEBUG_DIR, JOINT_NAMES, Sim, load_config
from clearance import (
    obstacle_bodies,
    plan_approach,
    safe_transit_ee_z,
    transit_clearances,
    transit_nontarget_hit,
)
from perception import estimate_target_pose
from sensor import capture as capture_wrist
from squares import jitter_near_square, square_to_world
from scenario import HIDE_XYZ, PIECE_BODIES, STARTING_AT

JAW_BODIES = ("gripper", "moving_jaw_so101_v1")
CAMERA_BODIES = ("camera_mount",)
ARM_BODIES = ("base", "shoulder", "upper_arm", "lower_arm")
FULL_BOARD_XML = ROOT / "assets/so101/chess_scene_full.xml"
PIECES_XML = ROOT / "assets/so101/chess_scene_pieces.xml"
_REASON_CANON = {
    "PERCEPTION_FAILURE": "PERCEPTION_FAIL",
    "IK_FAILURE": "IK_NO_SOLUTION",
}


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


def _clutter_square(name: str) -> str | None:
    if name.startswith("proxy_"):
        return name[6:]
    if name == "queen":
        return "d1"
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


def seed_vertical(sim: Sim, cfg: dict):
    """Menagerie box-pickup wrist: gripperframe +X already points down."""
    seed = cfg.get("poses", {}).get(
        "vertical_seed",
        {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.47,
            "wrist_flex": 1.18,
            "wrist_roll": 1.58,
            "gripper": 0.45,
        },
    )
    sim.set_qpos(seed)
    sim.hold(seed)
    for _ in range(40):
        sim.step()


def settle_queen(sim: Sim, steps: int = 80):
    for _ in range(steps):
        sim.step()


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


def jaw_queen_contacts(sim: Sim) -> list[dict]:
    return jaw_contacts(sim, "queen")


def min_jaw_gap(sim: Sim, object_id: str | None = None) -> float:
    tid = object_id or _tid(sim)
    bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, tid)
    fromto = np.zeros(6)
    best = 1.0
    for i in range(sim.model.ngeom):
        ni = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if not (ni.startswith("fixed_jaw") or ni.startswith("moving_jaw")):
            continue
        for j in range(sim.model.ngeom):
            if int(sim.model.geom_bodyid[j]) != bid:
                continue
            d = mujoco.mj_geomDistance(sim.model, sim.data, i, j, 0.08, fromto)
            if d < best:
                best = d
    return float(best)


def min_body_gap(sim: Sim, body_a: str, body_b: str) -> float:
    ba = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, body_a)
    bb = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, body_b)
    fromto = np.zeros(6)
    best = 1.0
    for i in range(sim.model.ngeom):
        if int(sim.model.geom_bodyid[i]) != ba:
            continue
        for j in range(sim.model.ngeom):
            if int(sim.model.geom_bodyid[j]) != bb:
                continue
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
    for n in ARM_JOINTS:
        lo, hi = sim.model.jnt_range[sim._joint[n]]
        v = q[n]
        m = float(min(v - lo, hi - v))
        margins[n] = round(m, 4)
        best = min(best, m)
    return {"min_rad": float(best), "per_joint": margins}


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
    ik_ok = solve_ik(sim, target, lock_joints=lock_joints)
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
    move_lin(sim, target, duration=duration, physics=True, n_wp=n_wp, lock_joints=lock_joints)
    pos, _ = sim.site_pose(EE_SITE)
    qpos, qquat = sim.object_pose(tid)
    err_mm = float(np.linalg.norm(pos - target_ee)) * 1000
    hits = jaw_contacts(sim, tid)
    report = {
        "ok": err_mm < 20.0,
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
    if not solve_ik(sim, target):
        pos, _ = sim.site_pose(EE_SITE)
        report["ee_actual"] = _xyz(pos)
        report["ee_err_mm"] = float(np.linalg.norm(pos - target_ee)) * 1000
        report["ik_residual_mm"] = report["ee_err_mm"]
        report["joint_limit_margin_rad"] = joint_limit_margin(sim)
        sim.set_qpos(start, hold=True)
        return report
    goal = sim.qpos()
    report["ik_solved"] = True
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
    for _ in range(30):
        sim.step()


def open_gripper(sim: Sim, opened: float, steps: int = 80):
    close_gripper(sim, opened, steps=steps)


def queen_upright(quat, tilt_max: float = 0.35) -> bool:
    """quat is wxyz; world +Z of the piece ≈ body +Z."""
    w, x, y, z = quat
    # body Z in world from quaternion
    zz = 1 - 2 * (x * x + y * y)
    return float(zz) > np.cos(tilt_max)


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


def queen_on_board(sim: Sim, surface: float, z_tol: float = 0.012) -> bool:
    return object_on_board(sim, surface, z_tol, "queen")


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


def gripper_queen_transform(sim: Sim) -> dict:
    return gripper_object_transform(sim, "queen")


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


def _annotate_episode(result: dict):
    result["grasp_success"] = bool(result.get("grasp_acquired"))
    result["close_shove"] = result.get("close_shove_xy_mm")
    result["tilt_after_close"] = result.get("close_tilt_deg")
    t = result.get("T_gripper_queen_lift")
    result["T_gripper_object"] = t
    result["placement_error"] = result.get("release_position_error_mm")
    result["perception_success"] = bool(result.get("queen_detected"))
    pickup_ok = bool(result.get("lift_success"))
    trans_ok = bool(result.get("transport_success"))
    place_ok = bool(result.get("placement_success"))
    retreat_ok = bool(result.get("retreat_success"))
    reason = result.get("failure_reason")
    flags = []
    flags.append("GRASP_PASS" if result["grasp_success"] else "GRASP_FAIL")
    flags.append("LIFT_PASS" if pickup_ok else "LIFT_FAIL")
    skip_rest = result.get("stop_after") == "lift" or not pickup_ok
    flags.append("TRANSPORT_PASS" if trans_ok else ("TRANSPORT_SKIP" if skip_rest else "TRANSPORT_FAIL"))
    if result.get("stop_after") == "lift" or (result.get("release_position_error_mm") is None and not place_ok):
        flags.append("PLACEMENT_SKIP")
    else:
        flags.append("PLACEMENT_PASS" if place_ok else "PLACEMENT_FAIL")
    if result.get("retreat_success") is None:
        flags.append("RETREAT_SKIP")
    else:
        flags.append("RETREAT_PASS" if retreat_ok else "RETREAT_FAIL")
    result["stage_flags"] = flags
    reason = result.get("failure_reason")
    if reason in _REASON_CANON:
        reason = _REASON_CANON[reason]
        result["failure_reason"] = reason
    if result.get("stop_after") == "lift":
        if not result.get("perception_success") or not pickup_ok:
            result["failure_category"] = "PICKUP_FAIL"
        else:
            result["failure_category"] = None
            result["success"] = True
    elif not result.get("perception_success"):
        result["failure_category"] = "PICKUP_FAIL"
    elif not pickup_ok:
        result["failure_category"] = "PICKUP_FAIL"
    elif not trans_ok:
        result["failure_category"] = "TRANSPORT_FAIL"
    elif not place_ok:
        result["failure_category"] = "PLACEMENT_FAIL"
    elif result.get("retreat_success") is False:
        result["failure_category"] = "RETREAT_FAIL"
    else:
        result["failure_category"] = None
    hit = result.get("first_collision") or result.get("transport_first_collision")
    result["neighbor_collision_phase"] = None if not hit else hit.get("phase")
    if hit:
        result["obstruction_body"] = hit.get("body")
        result["obstruction_square"] = hit.get("square")
        result["obstruction_piece"] = hit.get("piece")
        result["obstruction_robot_part"] = hit.get("robot_part")
        result["obstruction_phase"] = hit.get("phase")
    else:
        result["obstruction_body"] = None
        result["obstruction_square"] = None
        result["obstruction_piece"] = None
        result["obstruction_robot_part"] = None
        result["obstruction_phase"] = None
    if result.get("ik_fail_phase") is None and reason in ("IK_FAILURE", "IK_NO_SOLUTION"):
        result["ik_fail_phase"] = result.get("fail_phase")
    if result.get("neighbor_contact") is None:
        result["neighbor_contact"] = False
    if "neighbor_displacement" not in result:
        result["neighbor_displacement"] = None
    knocked = result.get("failure_reason") in ("APPROACH_COLLISION",)
    reached = bool(result.get("approach_reached"))
    ik = bool(result.get("ik_solved"))
    ncon = int(result.get("pre_close_contacts") or 0)
    result["approach_ok"] = bool(ik and reached and ncon == 0 and not knocked)
    if result.get("lift_success"):
        result["grasp_ok"] = True
    elif result["approach_ok"]:
        result["grasp_ok"] = False
    else:
        result["grasp_ok"] = None


def run_pickup(sim: Sim, cfg: dict, use_vision: bool = False, spawn_xy=None, scenario: dict | None = None) -> dict:
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
    geom = dict(sc.get("geometry") or {})
    off = sc.get("grasp_xy_offset", geom.get("grasp_xy_offset", p.get("grasp_xy_offset", [0.0, 0.0])))
    surface = float(cfg.get("sim", {}).get("board", {}).get("surface_z", 0.02))
    open_g = float(p.get("gripper_open", 0.45))
    closed_g = float(geom.get("gripper_closed", p.get("gripper_closed", -0.05)))
    pre_z = float(p.get("pregrasp_z", 0.10))
    grasp_z = float(geom.get("preferred_grasp_height", p.get("grasp_z", 0.016)))
    lift_z = float(p.get("lift_z", 0.08))
    observe_z = float(p.get("observe_z", 0.20))
    place_hover = float(p.get("place_hover", 0.003))
    xy_tol = float(p.get("place_xy_tol", 0.018))
    transport_z = float(sc.get("transport_clearance_z", p.get("transport_clearance_z", 0.08)))
    dest_xy = square_to_world(dest, cfg, z=surface)
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

    first_hit = [None]

    def _clutter_names() -> list[str]:
        names = []
        for name in _proxy_body_names(sim):
            bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0 and float(sim.data.xpos[bid][2]) > 0.0:
                names.append(name)
        for piece in PIECE_BODIES:
            if piece == tid:
                continue
            p, _ = sim.object_pose(piece)
            if float(p[2]) > 0.0:
                names.append(piece)
        return names

    def _track_proxies():
        if first_hit[0] is not None:
            return
        phase = getattr(sim, "_pickup_phase", None)
        if phase in (None, "001_initial"):
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
                sq = _clutter_square(name)
                first_hit[0] = {
                    "body": name,
                    "other": other,
                    "phase": phase,
                    "dist_mm": float(h["dist"] * 1000),
                    "camera_involved": part == "camera",
                    "robot_part": part,
                    "square": sq,
                    "piece": STARTING_AT.get(sq) if sq else (other if other in PIECE_BODIES else None),
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

    def _done():
        _track_neighbors(with_gap=True)
        _track_proxies()
        if result.get("failure_reason") in ("IK_FAILURE", "IK_NO_SOLUTION") and not result.get("ik_fail_phase"):
            result["ik_fail_phase"] = getattr(sim, "_pickup_phase", None)
        _annotate_episode(result)
        return result

    if spawn_xy is not None:
        spawn = list(spawn_xy)
    else:
        base = square_to_world(source, cfg, z=surface)
        spawn = [base[0] + offset_xy[0], base[1] + offset_xy[1], base[2]]
    result["spawn_xyz"] = _xyz(spawn)
    keep = {tid}
    for nb in sc.get("neighbors") or []:
        keep.add(nb.get("id", "pawn"))
    populated_board = bool(sc.get("hide_proxy"))
    for name in PIECE_BODIES:
        if name in keep:
            continue
        # Full-board XML already has the queen on d1; leave it unless we occupy that square.
        if populated_board and name == "queen" and sc.get("hide_proxy") != "d1":
            continue
        sim.place_object(name, HIDE_XYZ)
    hide_sq = sc.get("hide_proxy")
    if hide_sq:
        sim.hide_body(f"proxy_{hide_sq}")
    if tid == "queen" and source != "d1":
        d1 = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "proxy_d1")
        if d1 >= 0:
            sim.model.body_pos[d1, 2] = surface
    sim.place_object(tid, spawn, yaw_quat(yaw_deg))
    neighbor_start = {}
    for nb in sc.get("neighbors") or []:
        nid = nb.get("id", "pawn")
        nsq = nb.get("square")
        npos = square_to_world(nsq, cfg, z=surface) if nsq else list(nb.get("xyz", [0.37, 0.05, -0.5]))
        sim.place_object(nid, npos, yaw_quat(float(nb.get("yaw_deg", 0.0))))
        neighbor_start[nid] = npos[:2]
    result["_neighbor_start"] = neighbor_start
    seed_vertical(sim, cfg)
    open_gripper(sim, open_g, steps=40)
    settle_queen(sim)

    actual, _ = sim.object_pose(tid)
    result["actual_start_pose"] = actual.tolist()
    result["queen_start_xyz"] = actual.tolist()
    log_state(sim, "001_initial", extra={"spawn": _xyz(spawn)})

    if use_vision:
        prior = square_to_world(source, cfg, z=surface)
        look = [prior[0], prior[1], surface + 0.08]
        if not move_ik(sim, look, duration=1.0):
            result["failure_reason"] = "IK_NO_SOLUTION"
            log_state(sim, "fail_observe", extra={"target": _xyz(look)})
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
        observe = [queen_xyz[0], queen_xyz[1], surface + observe_z]
        if not move_ik(sim, observe, duration=1.2):
            result["failure_reason"] = "IK_NO_SOLUTION"
            log_state(sim, "fail_observe", extra={"target": _xyz(observe)})
            return _done()
        log_state(sim, "002_observe", extra={"target": _xyz(observe)})
        queen_xyz, _ = sim.object_pose(tid)
    plan_xy = [float(queen_xyz[0] + pose_err[0]), float(queen_xyz[1] + pose_err[1])]
    ox, oy = float(off[0]), float(off[1])
    obstacles = []
    for nb in sc.get("neighbors") or []:
        nid = nb.get("id", "pawn")
        npos, _ = sim.object_pose(nid)
        if npos[2] > 0.0:
            obstacles.append(npos[:2])
    pre = [plan_xy[0] + ox, plan_xy[1] + oy, surface + pre_z]
    grasp_xy = [plan_xy[0] + ox, plan_xy[1] + oy]
    blocked = carry_blocked(obstacles, pre[:2], dest_xy[:2])
    result["transport_path"] = "clearance" if blocked else "straight"
    clutter = [n for n in obstacle_bodies(sim, skip={tid}) if n != "board"]
    result["occupied_neighbors"] = sorted(clutter)
    grasp_plan = [grasp_xy[0], grasp_xy[1], surface + grasp_z]
    approach = plan_approach(
        sim,
        pre,
        grasp_plan,
        open_g,
        clutter,
        source=source,
        cfg=cfg,
        scenario=sc,
    )
    result["approach_plan"] = {
        k: approach.get(k) for k in ("ok", "roll", "gripper", "n_wp", "tried", "reachable", "rank", "reason")
    }
    if not approach["ok"]:
        result["failure_reason"] = (
            "NO_COLLISION_FREE_APPROACH" if approach.get("reachable") else "IK_NO_SOLUTION"
        )
        log_state(sim, "fail_pregrasp", extra={"target": _xyz(pre), "approach": result["approach_plan"]})
        return _done()
    lock: tuple[str, ...] = tuple(approach["lock_joints"])
    if approach["roll"] is not None:
        qpre = sim.qpos()
        qpre["wrist_roll"] = float(approach["roll"])
        sim.set_qpos(qpre, hold=True)
    g_ap = float(approach["gripper"])
    result["approach_gripper"] = None if abs(g_ap - open_g) < 0.02 else g_ap
    if abs(g_ap - float(sim.qpos()["gripper"])) > 0.01:
        qg = sim.qpos()
        qg["gripper"] = g_ap
        sim.set_qpos(qg, hold=True)
        sim.set_ctrl({**sim.qpos(), "gripper": g_ap})
    pickup_clear = bool(clutter) or approach["n_wp"] > 1
    pre_move = (
        move_lin(sim, pre, duration=1.0, physics=False, n_wp=6, lock_joints=lock)
        if pickup_clear
        else move_ik(sim, pre, duration=1.0, lock_joints=lock)
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
        return _done()

    if use_vision:
        refined = locate_queen(sim, cfg, result, source)
        if refined is not None:
            queen_xyz = refined
            plan_xy = [float(queen_xyz[0] + pose_err[0]), float(queen_xyz[1] + pose_err[1])]

    result["pose_used_for_grasp"] = [float(plan_xy[0]), float(plan_xy[1]), float(np.asarray(queen_xyz)[2])]
    result["perception_error"] = [
        float((plan_xy[0] - actual[0]) * 1000.0),
        float((plan_xy[1] - actual[1]) * 1000.0),
    ]
    grasp = [plan_xy[0] + ox, plan_xy[1] + oy, surface + grasp_z]
    result["ee_requested"] = _xyz(grasp)
    blocked = carry_blocked(obstacles, grasp[:2], dest_xy[:2])
    result["transport_path"] = "clearance" if blocked else "straight"
    desc = physical_descend(
        sim, grasp, duration=1.2, n_wp=8 if pickup_clear else 1, lock_joints=lock
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
        return _done()

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
        return _done()

    g_close, _ = sim.site_pose(EE_SITE)
    pick_xy = np.array([grasp[0], grasp[1]], dtype=np.float64)
    q_gt, _ = sim.object_pose(tid)
    result["grasp_offset_error_mm"] = float(np.linalg.norm(q_gt[:2] - pick_xy)) * 1000

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
            sq = _clutter_square(hit.get("body") or "")
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
    lift = [float(g_close[0]), float(g_close[1]), float(g_close[2] + lift_z)]
    lift_ok = (
        move_lin(sim, lift, duration=1.2, physics=True, n_wp=8, lock_joints=lock)
        if (blocked or populated)
        else move_ik(sim, lift, duration=1.2, physics=True)
    )
    if not lift_ok:
        result["failure_reason"] = "IK_FAILURE"
        log_state(sim, "fail_lift", extra={"target": lift})
        return _done()
    for _ in range(40):
        sim.step()
    log_state(sim, "006_lift")

    qpos, qquat = sim.object_pose(tid)
    result["T_gripper_queen_lift"] = gripper_object_transform(sim, tid)
    clearance = float(p.get("lift_clearance", 0.04))
    held = grasp_acquired(sim, tid)
    lifted = qpos[2] > surface + clearance
    result["max_queen_height_mm"] = float(qpos[2] * 1000)
    result["lift_success"] = bool(lifted and held and queen_upright(qquat, tilt_max=0.7))
    if not result["lift_success"]:
        result["failure_reason"] = "LIFT_DROP" if result["grasp_acquired"] else "GRASP_MISS"
        return _done()
    if sc.get("stop_after") == "lift":
        result["success"] = True
        result["failure_reason"] = None
        return _done()

    gpos, _ = sim.site_pose(EE_SITE)
    delta = dest_xy[:2] - pick_xy
    z_ee, derived = safe_transit_ee_z(sim, float(gpos[2]), tid, obstacles)
    result["transit_z"] = z_ee
    result["transit_z_derived"] = derived
    if populated:
        result["transport_path"] = "over_board"
        if z_ee > float(gpos[2]) + 0.004:
            high = [float(gpos[0]), float(gpos[1]), z_ee]
            if not move_lin(sim, high, duration=0.8, physics=True, n_wp=4, lock_joints=lock):
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
        if not move_lin(sim, carry, duration=max(1.5, 0.4 * n_xy), physics=True, n_wp=n_xy, lock_joints=lock):
            result["failure_reason"] = "IK_FAILURE"
            log_state(sim, "fail_transport", extra={"target": _xyz(carry)})
            return _done()
    elif blocked:
        result["transport_path"] = "clearance"
        z_safe = max(float(gpos[2]), surface + transport_z)
        if z_safe > float(gpos[2]) + 0.004:
            high = [float(gpos[0]), float(gpos[1]), z_safe]
            if not move_lin(sim, high, duration=0.8, physics=True, n_wp=4, lock_joints=lock):
                result["failure_reason"] = "IK_FAILURE"
                log_state(sim, "fail_transport_raise", extra={"target": _xyz(high)})
                return _done()
            gpos, _ = sim.site_pose(EE_SITE)
        carry = [float(gpos[0] + delta[0]), float(gpos[1] + delta[1]), float(gpos[2])]
        if not move_lin(sim, carry, duration=1.5, physics=True, n_wp=10, lock_joints=lock):
            result["failure_reason"] = "IK_FAILURE"
            log_state(sim, "fail_transport", extra={"target": _xyz(carry)})
            return _done()
    else:
        result["transport_path"] = "straight"
        carry = [float(gpos[0] + delta[0]), float(gpos[1] + delta[1]), float(gpos[2])]
        if not move_ik(sim, carry, duration=1.5, physics=True):
            result["failure_reason"] = "IK_FAILURE"
            log_state(sim, "fail_transport", extra={"target": _xyz(carry)})
            return _done()
    for _ in range(30):
        sim.step()
    qpos, qquat = sim.object_pose(tid)
    held_over = bool(grasp_acquired(sim, tid) and qpos[2] > surface + clearance)
    over_dest = float(np.hypot(qpos[0] - dest_xy[0], qpos[1] - dest_xy[1])) < 0.04
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
        extra={"target": _xyz(carry), "dest": dest, "path": result["transport_path"], "transit_z": result.get("transit_z")},
    )
    if not result["transport_success"]:
        result["failure_reason"] = "TRANSPORT_COLLISION" if result["transport_contact_transit"] else "TRANSPORT_DROP"
        return _done()

    gpos, _ = sim.site_pose(EE_SITE)
    place = [float(gpos[0]), float(gpos[1]), float(g_close[2]) + place_hover]
    lower_ok = (
        move_lin(sim, place, duration=1.2, physics=True, n_wp=8, lock_joints=lock)
        if (blocked or populated)
        else move_ik(sim, place, duration=1.2, physics=True)
    )
    if not lower_ok:
        result["failure_reason"] = "IK_FAILURE"
        result["ik_fail_phase"] = "008_lower"
        log_state(sim, "fail_lower", extra={"target": _xyz(place)})
        return _done()
    log_state(sim, "008_lower", extra={"target": _xyz(place)})

    open_gripper(sim, open_g, steps=int(p.get("close_steps", 220)))
    settle_queen(sim, steps=80)
    result["release_after_open"] = release_snapshot(sim, tid)

    wrist = ("wrist_flex", "wrist_roll")
    release_lift = float(p.get("release_lift_z", 0.08))
    if grasp_acquired(sim, tid):
        gpos, _ = sim.site_pose(EE_SITE)
        clear = [float(gpos[0]), float(gpos[1]), float(gpos[2]) + release_lift]
        move_lin(sim, clear, duration=0.8, physics=True, n_wp=6, lock_joints=wrist)
        settle_queen(sim, steps=80)
    result["release_after_lift"] = release_snapshot(sim, tid)

    q_rel, qquat_rel = sim.object_pose(tid)
    result["release_position_error_mm"] = float(np.hypot(q_rel[0] - dest_xy[0], q_rel[1] - dest_xy[1])) * 1000
    still_held = grasp_acquired(sim, tid)
    on_sq = result["release_position_error_mm"] / 1000.0 < xy_tol
    seated = object_on_board(sim, surface, object_id=tid)
    upright = queen_upright(qquat_rel, tilt_max=0.4)
    result["placement_success"] = bool(on_sq and upright and seated and not still_held)
    log_state(sim, "009_release", extra={"release_err_mm": round(result["release_position_error_mm"], 2)})

    gpos, _ = sim.site_pose(EE_SITE)
    z_up = max(float(gpos[2]) + release_lift, float(result.get("transit_z") or 0.0))
    high = [float(gpos[0]), float(gpos[1]), z_up]
    retreated = True
    if float(gpos[2]) < z_up - 0.01:
        retreated = move_lin(sim, high, duration=0.8, physics=True, n_wp=6, lock_joints=wrist)
    if not retreated:
        result["ik_fail_phase"] = "010_final"
        result["retreat_success"] = False
        if not result["placement_success"]:
            result["failure_reason"] = "RELEASE_FAILURE" if still_held else "PLACEMENT_POSITION_ERROR"
        else:
            result["failure_reason"] = "IK_FAILURE"
        log_state(sim, "fail_retreat", extra={"target": _xyz(high)})
        return _done()
    settle_queen(sim, steps=80)
    log_state(sim, "010_final")

    qpos, qquat = sim.object_pose(tid)
    result["final_position_error_mm"] = float(np.hypot(qpos[0] - dest_xy[0], qpos[1] - dest_xy[1])) * 1000
    result["final_queen_xyz"] = qpos.tolist()
    result["retreat_success"] = bool(
        object_on_board(sim, surface, object_id=tid)
        and queen_upright(qquat, tilt_max=0.5)
        and not grasp_acquired(sim, tid)
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
