"""SO-101 follower over FeetechMotorsBus."""

from __future__ import annotations

import time
from pathlib import Path

import draccus
import yaml
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

ROOT = Path(__file__).resolve().parents[1]

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

MOTORS = {
    n: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100)
    for i, n in enumerate(JOINTS, start=1)
}


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _lerobot_calib(robot_id: str) -> dict[str, MotorCalibration] | None:
    path = HF_LEROBOT_CALIBRATION / ROBOTS / "so101_follower" / f"{robot_id}.json"
    if not path.exists():
        return None
    with open(path) as f, draccus.config_type("json"):
        c = draccus.load(dict[str, MotorCalibration], f)
    return {k: v for k, v in c.items() if k in MOTORS}


class RealArm:
    def __init__(self, port: str, joints: list[str] | None = None):
        cfg = load_config().get("real", {})
        self.port = port
        self.joints = joints or cfg.get("joints", JOINTS)
        self.vel = cfg.get("goal_velocity", 120)
        self.bus: FeetechMotorsBus | None = None

    def connect(self):
        robot_id = load_config().get("real", {}).get("robot_id", "follower_arm")
        self.bus = FeetechMotorsBus(port=self.port, motors=MOTORS, calibration=_lerobot_calib(robot_id))
        self.bus.connect(handshake=False)
        self.bus.enable_torque(self.joints)
        for n in self.joints:
            self.bus.write("Goal_Time", n, 0)
            self.bus.write("Goal_Velocity", n, self.vel)
            self.bus.write("Acceleration", n, 40)

    def hold(self, ticks: dict[str, int] | None = None):
        """Command present position — no movement."""
        ticks = ticks or self.read_ticks()
        self.write_ticks(ticks)

    def read_ticks(self, joints: list[str] | None = None) -> dict[str, int]:
        joints = joints or self.joints
        assert self.bus
        return {n: int(self.bus.read("Present_Position", n, normalize=False)) for n in joints}

    def read_load(self, joints: list[str] | None = None) -> dict[str, int]:
        joints = joints or self.joints
        assert self.bus
        return {n: int(self.bus.read("Present_Load", n, normalize=False)) for n in joints}

    def read_limits(self, joints: list[str] | None = None) -> dict[str, tuple[int, int]]:
        joints = joints or self.joints
        assert self.bus
        return {
            n: (
                int(self.bus.read("Min_Position_Limit", n, normalize=False)),
                int(self.bus.read("Max_Position_Limit", n, normalize=False)),
            )
            for n in joints
        }

    def write_ticks(self, ticks: dict[str, int]):
        assert self.bus
        for n, t in ticks.items():
            if n not in self.joints:
                continue
            self.bus.write("Goal_Velocity", n, self.vel, num_retry=2)
            self.bus.write("Goal_Position", n, int(t), normalize=False, num_retry=2)

    def estop(self):
        assert self.bus
        self.bus.disable_torque(self.joints)

    def wait_settled(self, target: dict[str, int], tol: int = 30, timeout: float = 4.0) -> dict[str, int]:
        assert self.bus
        t0 = time.time()
        ok_since = t0
        last = self.read_ticks(list(target))
        while time.time() - t0 < timeout:
            cur = self.read_ticks(list(target))
            err = max(abs(cur[n] - target[n]) for n in target)
            if err <= tol:
                if time.time() - ok_since > 0.2:
                    return cur
            else:
                ok_since = time.time()
            last = cur
            time.sleep(0.05)
        return last

    def close(self):
        if self.bus:
            self.bus.disconnect()
            self.bus = None
