"""Sim radians <-> real encoder ticks."""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CALIB_PATH = ROOT / "data" / "calib" / "sim_to_real.yaml"
TICKS_PER_REV = 4096.0


@dataclass
class JointMap:
    sim_home: float
    real_home: int
    ticks_per_rad: float
    min_tick: int
    max_tick: int

    def to_ticks(self, sim_rad: float) -> int:
        tick = self.real_home + self.ticks_per_rad * (sim_rad - self.sim_home)
        return int(round(max(self.min_tick, min(self.max_tick, tick))))

    def to_sim(self, tick: int) -> float:
        return self.sim_home + (tick - self.real_home) / self.ticks_per_rad


@dataclass
class Calib:
    joints: dict[str, JointMap]

    def sim_to_ticks(self, sim: dict[str, float]) -> dict[str, int]:
        return {n: self.joints[n].to_ticks(r) for n, r in sim.items() if n in self.joints}

    def ticks_to_sim(self, ticks: dict[str, int]) -> dict[str, float]:
        return {n: self.joints[n].to_sim(t) for n, t in ticks.items() if n in self.joints}

    def save(self, path: Path = CALIB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "joints": {
                n: {
                    "sim_home": m.sim_home,
                    "real_home": m.real_home,
                    "ticks_per_rad": m.ticks_per_rad,
                    "min_tick": m.min_tick,
                    "max_tick": m.max_tick,
                }
                for n, m in self.joints.items()
            }
        }
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    @classmethod
    def load(cls, path: Path = CALIB_PATH) -> Calib:
        with open(path) as f:
            raw = yaml.safe_load(f)
        joints = {n: JointMap(**d) for n, d in raw["joints"].items()}
        return cls(joints=joints)


def reanchor(cal: Calib, present: dict[str, int], sim_home: dict[str, float], joints: list[str]) -> Calib:
    """Current physical pose becomes sim home — no motion required."""
    updated = dict(cal.joints)
    for name in joints:
        m = cal.joints[name]
        updated[name] = JointMap(
            sim_home=sim_home[name],
            real_home=int(present[name]),
            ticks_per_rad=m.ticks_per_rad,
            min_tick=m.min_tick,
            max_tick=m.max_tick,
        )
    return Calib(joints=updated)


def align(real, cal: Calib, sim_home: dict[str, float], joints: list[str], save: bool = False) -> Calib:
    present = real.read_ticks(joints)
    cal = reanchor(cal, present, sim_home, joints)
    real.hold(present)
    if save:
        cal.save()
    return cal, present


def find_port(preferred: str | None = None) -> str | None:
    if preferred and os.path.exists(preferred):
        return preferred
    ports = []
    try:
        from serial.tools import list_ports

        ports = [p.device for p in list_ports.comports() if p.device]
    except Exception:
        pass
    ports.extend(glob.glob("/dev/cu.usbmodem*"))
    ports.extend(glob.glob("/dev/cu.usbserial*"))
    ports = sorted(set(ports), key=lambda d: (0 if "/cu." in d else 1, d))
    for dev in ports:
        if "usbmodem" in dev or "usbserial" in dev:
            return dev
    return None


def calibrate(sim, real, home_sim: dict[str, float], joints: list[str], nudge: float = 0.2) -> Calib:
    """Anchor at home, nudge each joint once, fit ticks_per_rad."""
    sim.reset_pose(home_sim)
    sim.set_joints(home_sim)
    sim.step(0.3)

    limits = real.read_limits(joints)
    maps: dict[str, JointMap] = {}
    home_real = real.read_ticks(joints)

    for name in joints:
        maps[name] = JointMap(
            sim_home=home_sim[name],
            real_home=home_real[name],
            ticks_per_rad=TICKS_PER_REV / (2 * 3.1415926535),
            min_tick=limits[name][0],
            max_tick=limits[name][1],
        )

    for name in joints:
        target = dict(home_sim)
        target[name] = home_sim[name] + nudge
        sim.move_to({name: target[name]}, duration=1.0)
        sim_rad = sim.joint_states()[name]

        tick_goal = maps[name].to_ticks(sim_rad)
        real.write_ticks({name: tick_goal})
        settled = real.wait_settled({name: tick_goal})[name]

        m = maps[name]
        dr = sim_rad - m.sim_home
        if abs(dr) > 1e-4:
            m.ticks_per_rad = (settled - m.real_home) / dr
        print(f"{name}: real_home={m.real_home} ticks_per_rad={m.ticks_per_rad:.1f}")

        sim.move_to({name: home_sim[name]}, duration=0.8)
        time.sleep(0.1)

    return Calib(joints=maps)
