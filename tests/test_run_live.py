#!/usr/bin/env python3
"""Live sync smoke test — headless, ~5s, real arm must be connected."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))

from arm import SimArm, load_config
from real.arm import RealArm
from real.calib import CALIB_PATH, Calib, find_port
from real.safety import SafetyConfig, Watchdog


def main():
    if not CALIB_PATH.exists():
        sys.exit("no calib — run sync.py calibrate first")

    port = find_port()
    if not port:
        sys.exit("no arm port")

    cfg = load_config()
    cal = Calib.load()
    home = cfg["poses"]["home"]
    joints = cfg["sync"]["joints"]
    rate = cfg["sync"].get("rate_hz", 30)
    dt = 1.0 / rate

    sim = SimArm(gui=False)
    real = RealArm(port, joints=joints)
    real.connect()

    sc = SafetyConfig.from_dict(cfg.get("safety"))
    wd = Watchdog(sc, {n: (cal.joints[n].min_tick, cal.joints[n].max_tick) for n in joints})
    from real.calib import align

    cal, present = align(real, cal, home, joints)
    wd.seed(present)
    targets = {j: home[j] for j in home}

    pan0 = real.read_ticks(["shoulder_pan"])["shoulder_pan"]
    frames = 0
    estop = False

    print("phase 1: hold home 2s")
    t_end = time.time() + 2.0
    while time.time() < t_end:
        sim.set_joints(targets)
        sim.step(dt)
        desired = cal.sim_to_ticks({j: targets[j] for j in joints})
        cmd = wd.filter(desired)
        real.write_ticks(cmd)
        fault = wd.check(cmd, real.read_ticks(joints), real.read_load(joints))
        if fault and wd.note(fault):
            print("ESTOP:", fault)
            estop = True
            real.estop()
            break
        wd.note(None)
        frames += 1
        time.sleep(dt)

    if estop:
        sim.disconnect()
        real.close()
        sys.exit(1)

    print("phase 2: nudge shoulder_pan +0.3 rad over 1.5s")
    targets["shoulder_pan"] = home["shoulder_pan"] + 0.3
    t_end = time.time() + 1.5
    while time.time() < t_end:
        sim.set_joints(targets)
        sim.step(dt)
        desired = cal.sim_to_ticks({j: targets[j] for j in joints})
        cmd = wd.filter(desired)
        real.write_ticks(cmd)
        fault = wd.check(cmd, real.read_ticks(joints), real.read_load(joints))
        if fault and wd.note(fault):
            print("ESTOP:", fault)
            estop = True
            real.estop()
            break
        wd.note(None)
        frames += 1
        time.sleep(dt)

    pan1 = real.read_ticks(["shoulder_pan"])["shoulder_pan"]
    sim.disconnect()
    real.close()

    if estop:
        sys.exit(1)

    moved = pan1 - pan0
    print(f"frames={frames} pan {pan0}->{pan1} (delta {moved})")
    if moved < 50:
        sys.exit(f"shoulder_pan barely moved ({moved} ticks) — sync may be broken")
    print("ok")

if __name__ == "__main__":
    main()
