"""MuJoCo sim tests — python sim/test_mj.py [--quick]"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

from ik import solve_ik
from mj import DEBUG_DIR, JOINT_NAMES, Sim, load_config
from squares import square_to_world


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def test_load_and_render():
    sim = Sim()
    q, _ = sim.queen_pose()
    frames = []
    for cam in ("debug_overview", "debug_side", "debug_top", "wrist_cam"):
        path = sim.save_frame(cam, DEBUG_DIR / "smoke" / f"{cam}.png", width=480, height=360)
        frames.append(path)
    rgb = sim.render("debug_overview", 64, 48)
    sim.close()
    ok = q[2] > 0.015 and rgb.std() > 10
    return check("load + render", ok, f"queen_z={q[2]:.3f} frames={len(frames)}")


def test_joints_move():
    sim = Sim()
    sim.set_qpos({n: 0.0 for n in JOINT_NAMES})
    all_ok = True
    targets = {
        "shoulder_pan": 0.4,
        "shoulder_lift": -0.3,
        "elbow_flex": 0.5,
        "wrist_flex": 0.4,
        "wrist_roll": 0.5,
        "gripper": 0.6,
    }
    for name, target in targets.items():
        start = {n: 0.0 for n in JOINT_NAMES}
        sim.set_qpos(start)
        sim.hold(start)
        cmd = {**start, name: target}
        sim.set_ctrl(cmd)
        for _ in range(80):
            sim.step()
        actual = sim.qpos()[name]
        joint_ok = abs(actual - target) < 0.12
        if not check(f"move {name}", joint_ok, f"target={target:.2f} actual={actual:.2f}"):
            all_ok = False
    sim.close()
    return all_ok


def test_gripper_gap():
    sim = Sim()
    sim.set_qpos({**{n: 0.0 for n in JOINT_NAMES}, "gripper": -0.1})
    for _ in range(60):
        sim.step()
    closed = sim.jaw_gap()
    sim.set_ctrl({**sim.qpos(), "gripper": 0.8})
    for _ in range(100):
        sim.step()
    opened = sim.jaw_gap()
    sim.close()
    ok = opened > closed + 0.03
    return check("gripper opens", ok, f"closed={closed*1000:.1f}mm open={opened*1000:.1f}mm")


def test_ik_above_d1():
    sim = Sim()
    cfg = load_config()
    seed = cfg["poses"]["vertical_seed"]
    sim.set_qpos(seed)
    xy = square_to_world("d1", cfg)
    target = [xy[0], xy[1], 0.12]
    ok_solve = solve_ik(sim, target)
    pos, rot = sim.site_pose("gripperframe")
    err = float(((pos[0] - target[0]) ** 2 + (pos[1] - target[1]) ** 2 + (pos[2] - target[2]) ** 2) ** 0.5)
    down = float(rot[2, 0])
    sim.save_frame("debug_overview", DEBUG_DIR / "smoke" / "ik_d1.png")
    sim.save_frame("debug_side", DEBUG_DIR / "smoke" / "ik_d1_side.png")
    sim.close()
    ok = ok_solve and err < 0.008
    return check("IK above d1", ok, f"err={err*1000:.1f}mm  x_z={down:.2f}")


def test_wrist_localize():
    import numpy as np
    from ik import move_ik
    from perception import estimate_target_pose
    from pickup import open_gripper, seed_vertical, settle_queen
    from sensor import capture as capture_wrist

    cfg = load_config()
    sim = Sim(cfg=cfg)
    surface = float(cfg["sim"]["board"]["surface_z"])
    xy = square_to_world("d1", cfg)
    sim.place_queen(square_to_world("d1", cfg, z=surface))
    seed_vertical(sim, cfg)
    open_gripper(sim, 0.45, steps=30)
    settle_queen(sim, 40)
    move_ik(sim, [xy[0], xy[1], 0.08], duration=0.6)
    obs = capture_wrist(sim)
    sim.save_frame("wrist_cam", DEBUG_DIR / "perc" / "localize_wrist.png", 320, 240)
    est = estimate_target_pose(obs, cfg, "d1")
    gt, _ = sim.queen_pose()
    sim.close()
    if est is None:
        return check("wrist localize queen", False, "no estimate")
    err = float(np.linalg.norm(est[:2] - gt[:2]))
    return check(
        "wrist localize queen",
        err < 0.005,
        f"xy_err={err*1000:.1f}mm est={np.round(est[:2], 4).tolist()} gt={np.round(gt[:2], 4).tolist()}",
    )


def main():
    quick = "--quick" in sys.argv
    print("Zima MuJoCo tests" + (" (quick)" if quick else "") + "\n")
    results = [
        test_load_and_render(),
        test_joints_move(),
        test_gripper_gap(),
        test_ik_above_d1(),
    ]
    if not quick:
        results.append(test_wrist_localize())
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
