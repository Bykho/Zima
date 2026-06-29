#!/usr/bin/env python3
"""Align sim C-space to real arm.

  python sync.py calibrate   # measure tick/rad scale (moves arm once)
  python sync.py align       # current pose = sim home (no motion)
  python sync.py run --gui   # auto-aligns on start, then sync
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))


def cfg():
    import yaml

    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def calibrate_cmd(args):
    from arm import SimArm
    from real.arm import RealArm
    from real.calib import CALIB_PATH, calibrate, find_port

    c = cfg()
    port = find_port(args.port or c.get("real", {}).get("port"))
    if not port:
        sys.exit("No arm port. Plug in USB, run status.")

    home = c["poses"]["home"]
    joints = c["sync"]["joints"]
    nudge = c["sync"].get("nudge_rad", 0.2)

    sim = SimArm(gui=args.gui)
    real = RealArm(port)
    try:
        real.connect()
        cal = calibrate(sim, real, home, joints, nudge)
        cal.save(CALIB_PATH)
        print(f"saved {CALIB_PATH}")
    finally:
        real.close()
        sim.disconnect()


def run_cmd(args):
    import pybullet as p
    from arm import JOINT_NAMES, SimArm
    from controls import apply_key_event, nudge
    from real.arm import RealArm
    from real.calib import CALIB_PATH, Calib, find_port
    from real.safety import SafetyConfig, Watchdog

    c = cfg()
    if not CALIB_PATH.exists():
        sys.exit(f"No calib at {CALIB_PATH}. Run: python sync.py calibrate")

    cal = Calib.load()
    home = c["poses"]["home"]
    joints = c["sync"]["joints"]
    rate = c["sync"].get("rate_hz", 30)
    dt = 1.0 / rate

    port = find_port(args.port or c.get("real", {}).get("port"))
    if not port:
        sys.exit("No arm port.")

    sim = SimArm(gui=args.gui)
    real = RealArm(port, joints=joints)
    real.connect()

    from real.calib import align

    cal, present = align(real, cal, home, joints, save=args.save_align)
    print("aligned to present pose (sim home = where arm is now)")
    for n in joints:
        print(f"  {n}: {present[n]} tick")

    limits = {
        n: (float(p.getJointInfo(sim.robot_id, idx)[8]), float(p.getJointInfo(sim.robot_id, idx)[9]))
        for n, idx in sim.joint_indices.items()
    }
    targets = {n: home.get(n, 0.0) for n in JOINT_NAMES}
    sim.reset_pose(targets)
    state = {"targets": targets, "selected": 0, "step": 0.05, "grip_open": True}

    sc = SafetyConfig.from_dict(c.get("safety"))
    wd = Watchdog(sc, {n: (cal.joints[n].min_tick, cal.joints[n].max_tick) for n in joints})
    wd.seed(present)
    print("1-6 W/S R G Q — sim drives real (watchdog on)")
    print(f"safety: delta≤{sc.max_delta_ticks} err≤{sc.max_tracking_error} load≤{sc.max_load}")

    try:
        while p.isConnected():
            keys = p.getKeyboardEvents()
            if keys.get(ord("q"), 0) & p.KEY_WAS_TRIGGERED:
                break
            n = JOINT_NAMES[state["selected"]]
            if keys.get(ord("w"), 0) & p.KEY_IS_DOWN or keys.get(ord("W"), 0) & p.KEY_IS_DOWN:
                state["targets"] = nudge(state["targets"], limits, n, state["step"])
            if keys.get(ord("s"), 0) & p.KEY_IS_DOWN or keys.get(ord("S"), 0) & p.KEY_IS_DOWN:
                state["targets"] = nudge(state["targets"], limits, n, -state["step"])
            for key, st in keys.items():
                if st & p.KEY_WAS_TRIGGERED:
                    state["targets"], state["selected"], state["step"], state["grip_open"], _ = apply_key_event(
                        state["targets"],
                        limits,
                        home,
                        selected=state["selected"],
                        step=state["step"],
                        key=key,
                        gripper_open=state["grip_open"],
                    )

            sim.set_joints(state["targets"])
            sim.step(dt)

            desired = cal.sim_to_ticks({j: state["targets"][j] for j in joints})
            cmd = wd.filter(desired)
            real.write_ticks(cmd)
            fault = wd.check(cmd, real.read_ticks(joints), real.read_load(joints))
            if fault and wd.note(fault):
                print(f"ESTOP: {fault}")
                real.estop()
                break
            wd.note(None)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nCtrl+C — estop")
        real.estop()
    finally:
        real.close()
        sim.disconnect()


def align_cmd(args):
    from real.arm import RealArm
    from real.calib import CALIB_PATH, Calib, align, find_port

    c = cfg()
    if not CALIB_PATH.exists():
        sys.exit(f"No calib at {CALIB_PATH}. Run: python sync.py calibrate first")

    port = find_port(args.port or c.get("real", {}).get("port"))
    if not port:
        sys.exit("No arm port.")

    cal = Calib.load()
    home = c["poses"]["home"]
    joints = c["sync"]["joints"]
    real = RealArm(port, joints=joints)
    try:
        real.connect()
        cal, present = align(real, cal, home, joints, save=not args.no_save)
        print("aligned — current pose is sim home")
        for n in joints:
            print(f"  {n}: {present[n]} tick")
        if not args.no_save:
            print(f"saved {CALIB_PATH}")
    finally:
        real.close()


def status_cmd(_args):
    from real.calib import CALIB_PATH, find_port

    c = cfg()
    port = find_port(c.get("real", {}).get("port"))
    print("port:", port or "(not connected)")
    print("calib:", CALIB_PATH, CALIB_PATH.exists())
    s = c.get("safety", {})
    print(f"safety: delta={s.get('max_delta_ticks', 12)} err={s.get('max_tracking_error', 120)} load={s.get('max_load', 850)}")
    if CALIB_PATH.exists():
        from real.calib import Calib

        cal = Calib.load()
        for n, m in cal.joints.items():
            print(f"  {n}: home tick {m.real_home}  {m.ticks_per_rad:.1f} tick/rad")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("calibrate")
    pc.add_argument("--port")
    pc.add_argument("--gui", action="store_true")
    pr = sub.add_parser("run")
    pr.add_argument("--port")
    pr.add_argument("--gui", action="store_true")
    pr.add_argument("--save-align", action="store_true", help="persist re-anchor on run")
    pa = sub.add_parser("align", help="re-anchor: current pose = sim home")
    pa.add_argument("--port")
    pa.add_argument("--no-save", action="store_true")
    sub.add_parser("status")
    args = p.parse_args()
    if args.cmd == "calibrate":
        calibrate_cmd(args)
    elif args.cmd == "run":
        run_cmd(args)
    elif args.cmd == "align":
        align_cmd(args)
    else:
        status_cmd(args)


if __name__ == "__main__":
    main()
