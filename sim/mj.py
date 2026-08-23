"""MuJoCo SO-101 world — joints, rendering, ground-truth queries."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
ARM_JOINTS = JOINT_NAMES[:-1]
DEBUG_DIR = ROOT / "sim" / "debug"


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


class Sim:
    def __init__(self, xml_path: str | Path | None = None, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        xml_path = Path(xml_path or self.cfg.get("sim", {}).get("mjcf", "assets/so101/chess_scene.xml"))
        if not xml_path.is_absolute():
            xml_path = ROOT / xml_path
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self._joint = {n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES}
        self._act = {n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES}
        self._renderer: mujoco.Renderer | None = None
        self._depth_renderer: mujoco.Renderer | None = None
        self.on_step = None
        self.quiet = False
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.on_step = None

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        out = {}
        for name, jid in self._joint.items():
            lo, hi = self.model.jnt_range[jid]
            out[name] = (float(lo), float(hi))
        return out

    def qpos(self) -> dict[str, float]:
        return {n: float(self.data.qpos[self.model.jnt_qposadr[j]]) for n, j in self._joint.items()}

    def set_qpos(self, positions: dict[str, float], hold: bool = True):
        for name, value in positions.items():
            jid = self._joint[name]
            lo, hi = self.model.jnt_range[jid]
            self.data.qpos[self.model.jnt_qposadr[jid]] = float(np.clip(value, lo, hi))
            if hold:
                self.data.ctrl[self._act[name]] = float(np.clip(value, lo, hi))
        mujoco.mj_forward(self.model, self.data)

    def set_ctrl(self, positions: dict[str, float]):
        for name, value in positions.items():
            aid = self._act[name]
            lo, hi = self.model.actuator_ctrlrange[aid]
            self.data.ctrl[aid] = float(np.clip(value, lo, hi))

    def hold(self, positions: dict[str, float] | None = None):
        self.set_ctrl(positions or self.qpos())

    def step(self, n: int = 1):
        for _ in range(max(1, n)):
            mujoco.mj_step(self.model, self.data)
            if self.on_step is not None:
                self.on_step()

    def site_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return self.data.site_xpos[sid].copy(), self.data.site_xmat[sid].reshape(3, 3).copy()

    def body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[bid].copy(), self.data.xquat[bid].copy()

    def object_pose(self, body: str) -> tuple[np.ndarray, np.ndarray]:
        """Simulator ground truth for a free body."""
        return self.body_pose(body)

    def queen_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Simulator ground truth — evaluator / Phase-1 scaffold only."""
        return self.object_pose("queen")

    def geom_xpos(self, name: str) -> np.ndarray:
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return self.data.geom_xpos[gid].copy()

    def jaw_gap(self) -> float:
        a = self.geom_xpos("fixed_jaw_sph_tip1")
        b = self.geom_xpos("moving_jaw_sph_tip1")
        return float(np.linalg.norm(a - b))

    def place_object(self, body: str, xyz, quat=(1.0, 0.0, 0.0, 0.0)):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{body}_free")
        if jid < 0:
            return False
        adr = self.model.jnt_qposadr[jid]
        dof = self.model.jnt_dofadr[jid]
        self.data.qpos[adr : adr + 3] = np.asarray(xyz, dtype=np.float64)
        self.data.qpos[adr + 3 : adr + 7] = np.asarray(quat, dtype=np.float64)
        self.data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return True

    def place_queen(self, xyz, quat=(1.0, 0.0, 0.0, 0.0)):
        self.place_object("queen", xyz, quat)

    def hide_body(self, name: str, z: float = -1.0) -> bool:
        """Drop a static body below the board (proxies). Model-level; new Sim to undo."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            return False
        self.model.body_pos[bid, 2] = float(z)
        mujoco.mj_forward(self.model, self.data)
        return True

    def contacts_with(self, body: str) -> list[dict]:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        hits = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if bid not in (b1, b2):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            other = b2 if b1 == bid else b1
            hits.append(
                {
                    "other": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(other)),
                    "dist": float(c.dist),
                    "force": float(np.linalg.norm(force[:3])),
                    "pos": c.pos.copy(),
                }
            )
        return hits

    def _renderer_for(self, width: int, height: int, depth: bool = False) -> mujoco.Renderer:
        if depth:
            if self._depth_renderer is None or self._depth_renderer.width != width:
                self._depth_renderer = mujoco.Renderer(self.model, height=height, width=width)
                self._depth_renderer.enable_depth_rendering()
            return self._depth_renderer
        if self._renderer is None or self._renderer.width != width:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        return self._renderer

    def render(self, camera: str, width: int = 640, height: int = 480) -> np.ndarray:
        r = self._renderer_for(width, height)
        r.update_scene(self.data, camera=camera)
        return r.render()

    def render_depth(self, camera: str, width: int = 320, height: int = 240) -> np.ndarray:
        r = self._renderer_for(width, height, depth=True)
        r.update_scene(self.data, camera=camera)
        return r.render()

    def wrist_rgbd(self, width: int = 320, height: int = 240):
        """Debug helper. Manipulation should call sensor.capture, not this."""
        from sensor import capture

        return capture(self, width, height)

    def save_frame(self, camera: str, path: str | Path, width: int = 640, height: int = 480) -> Path:
        import cv2

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rgb = self.render(camera, width, height)
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return path

    def close(self):
        self._renderer = None
        self._depth_renderer = None


def smoke_test():
    sim = Sim()
    closed = {n: 0.0 for n in JOINT_NAMES}
    sim.set_qpos(closed)
    sim.hold(closed)
    for _ in range(40):
        sim.step()
    gap0 = sim.jaw_gap()
    sim.set_ctrl({**sim.qpos(), "gripper": 0.8})
    for _ in range(100):
        sim.step()
    gap1 = sim.jaw_gap()
    qpos, _ = sim.queen_pose()
    sim.close()
    assert gap1 > gap0 + 0.02, f"gripper did not open ({gap0:.3f} -> {gap1:.3f})"
    assert qpos[2] > 0.01, f"queen fell through board z={qpos[2]}"
    print(f"smoke ok  gap {gap0*1000:.1f}->{gap1*1000:.1f} mm  queen_z={qpos[2]:.3f}")


if __name__ == "__main__":
    smoke_test()
