"""Controller-side robot handle: the wall between control and simulator truth.

The controller gets a Robot, never the physics Sim. A Robot is a Sim over its
own copy of the scene model (the belief). Its arm follows joint readings from
the world after every step; its pieces sit wherever perception last put them.
Planning, IK and clearance run on that copy unchanged. Actuator commands and
camera renders go to the world.

Two flags open the wall for reference runs:
  cheat_state   belief pieces and contact queries mirror the world (perfect perception)
  cheat_motion  qpos writes with hold=True also teleport the world arm (legacy kinematic moves)
Both on reproduces the old ground-truth controller. Both off is the real robot.
"""

from __future__ import annotations

import mujoco
import numpy as np

from mj import JOINT_NAMES, Sim

HIDE_Z = -1.0


class GroundTruthAccess(RuntimeError):
    """Control code asked for something only the simulator knows."""


def world_of(sim) -> Sim:
    """Physics world behind a Robot. Evaluator privilege, never control."""
    return getattr(sim, "_world", None) or sim


def gt(sim) -> Sim:
    """Simulator truth for control code. Allowed only with cheat_state."""
    world = getattr(sim, "_world", None)
    if world is None:
        return sim
    if sim.cheat_state:
        return world
    raise GroundTruthAccess("control code read simulator truth; perception must supply this")


WALLS = ("off", "cheat", "perception", "real")


def open_robot(world: Sim, wall: str = "cheat"):
    """Controller handle for `world`.

    off         the world itself (no wall, legacy)
    cheat       Robot with perfect state and teleporting motion (must reproduce off)
    perception  state from perception only, legacy kinematic motion
    real        state from perception only, actuated motion only
    """
    if wall == "off":
        return world
    if wall == "cheat":
        return Robot(world, cheat_state=True, cheat_motion=True)
    if wall == "perception":
        return Robot(world, cheat_state=False, cheat_motion=True)
    if wall == "real":
        return Robot(world, cheat_state=False, cheat_motion=False)
    raise ValueError(f"wall must be one of {WALLS}")


def piece_free_joints(model) -> list[int]:
    """Free joints of movable scene bodies (the pieces)."""
    return [j for j in range(model.njnt) if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]


class Robot(Sim):
    def __init__(self, world: Sim, cheat_state: bool = False, cheat_motion: bool = False):
        super().__init__(xml_path=world.xml_path, cfg=world.cfg)
        self._world = world
        self.cheat_state = bool(cheat_state)
        self.cheat_motion = bool(cheat_motion)
        self.model.opt.timestep = world.model.opt.timestep
        self.dt = world.dt
        self.model.body_pos[:] = world.model.body_pos
        self._arm_adr = np.array([self.model.jnt_qposadr[self._joint[n]] for n in JOINT_NAMES], dtype=int)
        self._free = piece_free_joints(self.model)
        self._piece_names = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.jnt_bodyid[j])) for j in self._free}
        self.known: set[str] = set(self._piece_names) if self.cheat_state else set()
        if not self.cheat_state:
            for j in self._free:
                adr = int(self.model.jnt_qposadr[j])
                self.data.qpos[adr : adr + 3] = [0.0, 0.0, HIDE_Z]
                self.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.ctrl[:] = world.data.ctrl
        self._sync()

    # -- world link -------------------------------------------------------

    def _sync(self) -> None:
        w = self._world
        if self.cheat_state:
            self.model.body_pos[:] = w.model.body_pos
            self.data.qpos[:] = w.data.qpos
        else:
            self.data.qpos[self._arm_adr] = w.data.qpos[self._arm_adr]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1):
        w = self._world
        w.data.ctrl[:] = self.data.ctrl
        n = max(1, int(n))
        if self.on_step is not None:
            for _ in range(n):
                w.step()
                self.on_step()
        else:
            w.step(n)
        self._sync()

    def set_qpos(self, positions: dict[str, float], hold: bool = True):
        super().set_qpos(positions, hold=hold)
        if hold and self.cheat_motion:
            self._world.set_qpos(positions, hold=True)

    def actuator_forces(self) -> np.ndarray:
        return self._world.actuator_forces()

    def render(self, camera: str, width: int = 640, height: int = 480) -> np.ndarray:
        return self._world.render(camera, width, height)

    def render_depth(self, camera: str, width: int = 320, height: int = 240) -> np.ndarray:
        return self._world.render_depth(camera, width, height)

    def save_frame(self, camera: str, path, width: int = 640, height: int = 480):
        return self._world.save_frame(camera, path, width, height)

    # -- belief ------------------------------------------------------------

    def believe(self, body: str, xyz, quat=(1.0, 0.0, 0.0, 0.0)) -> bool:
        """Perception result: put belief `body` at `xyz`."""
        ok = super().place_object(body, xyz, quat)
        if ok:
            self.known.add(body)
        return ok

    def forget(self, body: str) -> None:
        super().place_object(body, [0.0, 0.0, HIDE_Z])
        self.known.discard(body)

    def object_pose(self, body: str):
        if body in self._piece_names and body not in self.known:
            raise GroundTruthAccess(f"{body} has not been perceived")
        return super().object_pose(body)

    # -- truth the real arm does not have ---------------------------------

    def contact_bodies(self):
        return gt(self).contact_bodies()

    def contacts_with(self, body: str) -> list[dict]:
        return gt(self).contacts_with(body)

    # -- scene edits act on the world; belief learns them only by cheating -

    def place_object(self, body: str, xyz, quat=(1.0, 0.0, 0.0, 0.0)):
        if getattr(self, "_world", None) is None:
            return super().place_object(body, xyz, quat)
        ok = self._world.place_object(body, xyz, quat)
        if self.cheat_state:
            self._sync()
        return ok

    def hide_body(self, name: str, z: float = -1.0) -> bool:
        ok = self._world.hide_body(name, z)
        super().hide_body(name, z)
        return ok

    def set_body_pos(self, name: str, xyz) -> bool:
        ok = self._world.set_body_pos(name, xyz)
        super().set_body_pos(name, xyz)
        return ok

    def tint_body(self, name: str, rgba) -> bool:
        return self._world.tint_body(name, rgba)

    def apply_arm_base(self, pos=None) -> None:
        super().apply_arm_base(pos)
        world = getattr(self, "_world", None)
        if world is not None:
            world.apply_arm_base(pos)
            self._sync()

    def reset(self):
        self._world.reset()
        super().reset()
        if not self.cheat_state:
            for j in self._free:
                adr = int(self.model.jnt_qposadr[j])
                self.data.qpos[adr : adr + 3] = [0.0, 0.0, HIDE_Z]
        self._sync()

    def close(self):
        super().close()
        self._world.close()
