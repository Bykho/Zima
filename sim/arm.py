"""PyBullet SO-101 sim."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pybullet as p
import pybullet_data
import yaml

from fast import apply_config

ROOT = Path(__file__).resolve().parents[1]
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


class SimArm:
    def __init__(self, gui: bool | None = None, fast: bool | None = None):
        cfg = load_config()
        if fast is None:
            fast = not gui
        if fast:
            os.environ.setdefault("ZIMA_SIM_FAST", "1")
            apply_config(cfg)

        sim = cfg["sim"]
        if gui is None:
            gui = sim.get("use_gui", True)

        self.cfg = cfg
        self.fast = fast
        self.motor_force = sim.get("motor_force", 200.0)
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, sim.get("gravity", -9.81))
        p.setTimeStep(sim.get("timestep", 1 / 240))
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1 if gui else 0)

        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(
            str(ROOT / sim["urdf"]),
            basePosition=sim.get("arm_base_position", [0, 0, 0]),
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )

        self.joint_indices: dict[str, int] = {}
        self._limits: dict[str, tuple[float, float]] = {}
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            name = info[1].decode("utf-8")
            if name in JOINT_NAMES:
                self.joint_indices[name] = i
                self._limits[name] = (float(info[8]), float(info[9]))

        missing = [n for n in JOINT_NAMES if n not in self.joint_indices]
        if missing:
            raise RuntimeError(f"Missing joints in URDF: {missing}")

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        return dict(self._limits)

    def reset_pose(self, positions: dict[str, float] | None = None):
        positions = positions or {n: 0.0 for n in JOINT_NAMES}
        for name in JOINT_NAMES:
            p.resetJointState(
                self.robot_id, self.joint_indices[name], positions.get(name, 0.0)
            )

    def set_joints(self, positions: dict[str, float], force: float | None = None):
        force = self.motor_force if force is None else force
        for name, angle in positions.items():
            if name not in self.joint_indices:
                continue
            p.setJointMotorControl2(
                self.robot_id,
                self.joint_indices[name],
                p.POSITION_CONTROL,
                targetPosition=angle,
                force=force,
            )

    def hold(self, positions: dict[str, float] | None = None):
        positions = positions or self.joint_states()
        self.set_joints(positions)

    def step(self, seconds: float = 0.02):
        dt = p.getPhysicsEngineParameters()["fixedTimeStep"]
        for _ in range(max(1, int(seconds / dt))):
            p.stepSimulation()

    def joint_states(self) -> dict[str, float]:
        return {
            name: p.getJointState(self.robot_id, idx)[0]
            for name, idx in self.joint_indices.items()
        }

    def resolve_pose(self, pose_name: str) -> dict[str, float]:
        from poses import load_all

        poses = load_all()
        if pose_name not in poses:
            raise KeyError(f"Unknown pose '{pose_name}'. Available: {list(poses)}")
        partial = poses[pose_name]
        return {**self.joint_states(), **partial}

    def move_to(
        self,
        target: dict[str, float],
        duration: float = 2.0,
        tolerance: float = 0.02,
        each_step=None,
    ) -> bool:
        start = self.joint_states()
        full_target = {**start, **target}

        if self.fast:
            dt = p.getPhysicsEngineParameters()["fixedTimeStep"]
            n_steps = max(36, int(duration / dt * 0.55))
            for i in range(n_steps):
                u = (i + 1) / n_steps
                u = u * u * (3 - 2 * u)
                cmd = {name: start[name] + (full_target[name] - start[name]) * u for name in JOINT_NAMES}
                self.set_joints(cmd)
                self.step()
                if each_step:
                    each_step()
            err = max(abs(self.joint_states()[n] - full_target[n]) for n in full_target)
            return err < tolerance

        t0 = time.time()
        while time.time() - t0 < duration + 1.0:
            u = min(1.0, (time.time() - t0) / duration)
            u = u * u * (3 - 2 * u)
            cmd = {name: start[name] + (full_target[name] - start[name]) * u for name in JOINT_NAMES}
            self.set_joints(cmd)
            self.step()
            if each_step:
                each_step()

            if u >= 1.0:
                err = max(abs(self.joint_states()[n] - full_target[n]) for n in full_target)
                if err < tolerance:
                    return True
        return False

    def go_pose(self, pose_name: str, duration: float = 2.0) -> bool:
        target = self.resolve_pose(pose_name)
        dur = duration * 0.5 if pose_name == "gripper_closed" else duration
        return self.move_to(target, duration=dur)

    def disconnect(self):
        p.disconnect(self.client)


def smoke_test():
    arm = SimArm(gui=False)
    ok = arm.move_to({"shoulder_pan": 0.5, "gripper": 0.2}, duration=0.5)
    states = arm.joint_states()
    arm.reset_pose(load_config().get("poses", {}).get("home"))
    for _ in range(60):
        arm.hold()
        arm.step(1 / 240)
    drift = max(abs(arm.joint_states()[n] - load_config()["poses"]["home"][n]) for n in JOINT_NAMES)
    arm.disconnect()
    assert ok and abs(states["shoulder_pan"] - 0.5) < 0.05 and drift < 0.03
    print("smoke test ok", f"drift={drift:.4f}")


def main():
    parser = argparse.ArgumentParser(description="SO-101 PyBullet sim smoke test")
    parser.add_argument("--test", action="store_true", help="run smoke test (default)")
    args = parser.parse_args()
    smoke_test()


if __name__ == "__main__":
    main()
