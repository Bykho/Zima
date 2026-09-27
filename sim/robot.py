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
EE_SITE = "gripperframe"
GRIP_LOAD_HELD = 0.08  # |gripper effort| while squeezing (relaxed pawn carry ~0.2); empty close reads ~0.003
GRIP_STILL_RADS = 0.03  # jaw speed below this with closing load = stalled on a piece
GRIP_ATTACH_XY_M = 0.025
PAD_GEOMS = ("fixed_jaw_box4", "moving_jaw_box2")  # opposing contact plates at pinch height
GRIP_LOAD_RELEASED = 0.04  # below this effort the jaws are no longer holding anything
GRIP_RELEASE_READS = 5


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


def jaw_squeezing(sim) -> bool:
    """Gripper stalled on something: closing effort with the jaw at rest (servo load + speed)."""
    load = float(sim.actuator_forces()[sim._act["gripper"]])
    return load < -GRIP_LOAD_HELD and abs(sim.qvel()["gripper"]) < GRIP_STILL_RADS


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
        self.last_seen: dict[str, np.ndarray] = {}
        self.held: str | None = None
        self._held_off = None
        self._slack = 0
        if not self.cheat_state:
            self._hide_all()
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
            mujoco.mj_kinematics(self.model, self.data)
            self._track_held()
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

    def qvel(self) -> dict[str, float]:
        return self._world.qvel()

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

    def _hide_xyz(self, body: str) -> list[float]:
        """Unperceived bodies wait apart below the table (a shared spot explodes the contact solver)."""
        k = sorted(self._piece_names).index(body)
        return [5.0 + 0.2 * k, 0.0, HIDE_Z]

    def _hide_all(self) -> None:
        for j in self._free:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.jnt_bodyid[j]))
            adr = int(self.model.jnt_qposadr[j])
            self.data.qpos[adr : adr + 3] = self._hide_xyz(name)
            self.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]

    def forget(self, body: str) -> None:
        if body in self.known:
            self.last_seen[body] = np.array(super().object_pose(body)[0][:2], dtype=float)
        super().place_object(body, self._hide_xyz(body))
        self.known.discard(body)

    def object_pose(self, body: str):
        if body in self._piece_names and body not in self.known:
            raise GroundTruthAccess(f"{body} has not been perceived")
        return super().object_pose(body)

    # -- contacts: measured in the world when cheating, else predicted by belief

    def contact_bodies(self):
        if self.cheat_state:
            return self._world.contact_bodies()
        return super().contact_bodies()

    def contacts_with(self, body: str) -> list[dict]:
        if self.cheat_state:
            return self._world.contacts_with(body)
        return super().contacts_with(body)

    # -- held piece: belief rides on the gripper while the jaws squeeze -----

    def gripper_load(self) -> float:
        return float(self.actuator_forces()[self._act["gripper"]])

    def squeezing(self) -> bool:
        return jaw_squeezing(self)

    def _jaw_center(self) -> np.ndarray:
        return 0.5 * (self.geom_xpos("fixed_jaw_sph_tip1") + self.geom_xpos("moving_jaw_sph_tip1"))

    def _between_jaws(self) -> str | None:
        c = self._jaw_center()
        best, best_d = None, GRIP_ATTACH_XY_M
        for name in self.known:
            p = self.body_pose(name)[0]
            d = float(np.hypot(p[0] - c[0], p[1] - c[1]))
            if d < best_d and p[2] < c[2] + 0.01:
                best, best_d = name, d
        return best

    def attach(self, body: str) -> None:
        """Both pads touch the piece: along the pinch axis it sits midway between them."""
        a = self.geom_xpos(PAD_GEOMS[0])
        b = self.geom_xpos(PAD_GEOMS[1])
        axis = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
        pb, qb = self.body_pose(body)
        pb = pb + float(np.dot(0.5 * (a + b) - pb, axis)) * axis
        super().place_object(body, pb, qb)
        pe, Re = self.site_pose(EE_SITE)
        pb, qb = self.body_pose(body)
        Rb = np.zeros(9)
        mujoco.mju_quat2Mat(Rb, qb)
        Rb = Rb.reshape(3, 3)
        self.held = body
        self._slack = 0
        self._held_off = (Re.T @ (pb - pe), Re.T @ Rb)

    def detach(self) -> None:
        self._slack = 0
        self.held = None
        self._held_off = None

    def _track_held(self) -> None:
        if self.held is None:
            if self.squeezing():
                body = self._between_jaws()
                if body is not None:
                    self.attach(body)
            return
        if self.gripper_load() > -GRIP_LOAD_RELEASED:
            self._slack += 1
            if self._slack >= GRIP_RELEASE_READS:
                self.detach()
                return
        else:
            self._slack = 0
        pe, Re = self.site_pose(EE_SITE)
        p_off, R_off = self._held_off
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (Re @ R_off).reshape(-1))
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.held}_free")
        adr = int(self.model.jnt_qposadr[jid])
        self.data.qpos[adr : adr + 3] = pe + Re @ p_off
        self.data.qpos[adr + 3 : adr + 7] = q

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
            self._hide_all()
            self.known.clear()
            self.last_seen.clear()
            self.detach()
        self._sync()

    def close(self):
        super().close()
        self._world.close()
