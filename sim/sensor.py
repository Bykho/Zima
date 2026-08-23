"""Wrist RGB-D backends. Each capture() returns a generic RgbdObservation.

The MuJoCo D405 backend is the current default. A physical RealSense (or any
other camera) should implement the same capture(sim) → RgbdObservation shape
so perception and manipulation stay unchanged.
"""

from __future__ import annotations

import numpy as np

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
        depth=sim.render_depth(camera, w, h),
        K=K_from_fovy(w, h, fovy),
        R=R,
        t=t,
        near=D405_NEAR,
        far=D405_FAR,
    )
