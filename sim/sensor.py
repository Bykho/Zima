"""Wrist RGB-D backends. Each capture() returns a generic RgbdObservation.

The MuJoCo D405 backend is the current default. A physical RealSense (or any
other camera) should implement the same capture(sim) → RgbdObservation shape
so perception and manipulation stay unchanged.
"""

from __future__ import annotations

import numpy as np

from realism import noisy_depth
from rgbd import K_from_fovy, RgbdObservation

# Published D405 working range. Perception reads these from the observation.
D405_NEAR = 0.07
D405_FAR = 0.50
DEFAULT_CAMERA = "wrist_cam"


def _mujoco_cv_extrinsics(sim, camera: str) -> tuple[np.ndarray, np.ndarray]:
    """MuJoCo camera (OpenGL, look -Z, Y up) → OpenCV (look +Z, Y down)."""
    import mujoco

    cid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    R_gl = sim.data.cam_xmat[cid].reshape(3, 3).copy()
    t = sim.data.cam_xpos[cid].copy()
    R_cv = np.column_stack([R_gl[:, 0], -R_gl[:, 1], -R_gl[:, 2]])
    return R_cv, t


def wrist_sees_point(sim, xyz, camera: str = DEFAULT_CAMERA) -> bool:
    """True if world XYZ is in the wrist camera frustum and D405 range."""
    import mujoco

    xyz = np.asarray(xyz, dtype=np.float64)
    cid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cid < 0:
        return False
    mujoco.mj_forward(sim.model, sim.data)
    R, t = _mujoco_cv_extrinsics(sim, camera)
    x_c = R.T @ (xyz - t)
    z = float(x_c[2])
    if z < D405_NEAR or z > D405_FAR:
        return False
    cam_cfg = (sim.cfg or {}).get("sim", {}).get("camera", {})
    w = int(cam_cfg.get("width", 320))
    h = int(cam_cfg.get("height", 240))
    K = K_from_fovy(w, h, float(sim.model.cam_fovy[cid]))
    u = float(K[0, 0] * x_c[0] / z + K[0, 2])
    v = float(K[1, 1] * x_c[1] / z + K[1, 2])
    return 4.0 <= u < w - 4.0 and 4.0 <= v < h - 4.0


def robot_blocks_view(sim, xyz, robot_bodies, camera: str = DEFAULT_CAMERA) -> bool:
    """True if one of the robot's own links is on the line of sight from the
    wrist camera to world XYZ (the jaws hang in its view)."""
    import mujoco

    cid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    mujoco.mj_forward(sim.model, sim.data)
    _, t = _mujoco_cv_extrinsics(sim, camera)
    vec = np.asarray(xyz, dtype=np.float64) - t
    reach = float(np.linalg.norm(vec))
    if reach < 1e-6:
        return False
    vec /= reach
    geom = np.zeros(1, dtype=np.int32)
    d = mujoco.mj_ray(sim.model, sim.data, t, vec, None, 0, int(sim.model.cam_bodyid[cid]), geom)
    if d < 0 or d >= reach - 0.002 or geom[0] < 0:
        return False
    return int(sim.model.geom_bodyid[geom[0]]) in set(int(b) for b in robot_bodies)


def capture(sim, width: int | None = None, height: int | None = None, camera: str = DEFAULT_CAMERA) -> RgbdObservation:
    """Current backend: simulated D405 on the SO-ARM wrist."""
    cam_cfg = (sim.cfg or {}).get("sim", {}).get("camera", {})
    w = int(width or cam_cfg.get("width", 320))
    h = int(height or cam_cfg.get("height", 240))
    import mujoco

    cid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    fovy = float(sim.model.cam_fovy[cid])
    R, t = _mujoco_cv_extrinsics(sim, camera)
    return RgbdObservation(
        rgb=sim.render(camera, w, h),
        depth=noisy_depth(sim.render_depth(camera, w, h)),
        K=K_from_fovy(w, h, fovy),
        R=R,
        t=t,
        near=D405_NEAR,
        far=D405_FAR,
    )
