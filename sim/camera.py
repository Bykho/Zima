"""Wrist camera — rigidly on gripper_link (32×32 mount), tracks via link FK each frame."""

from __future__ import annotations

import time

import numpy as np
import pybullet as p

from board import link_index

WINDOW = "Wrist Camera"
IDENT = [0.0, 0.0, 0.0, 1.0]


class WristCamera:
    def __init__(self, arm, cfg=None):
        cfg = cfg or arm.cfg
        cam = cfg.get("sim", {}).get("camera", {})
        self.arm = arm
        # SO-101 wrist cam PCB bolts to wrist-roll housing on gripper_link
        self.mount = cam.get("mount_link", "gripper_link")
        self.aim = cam.get("aim_link", "gripper_frame_link")
        # Lens on wrist-roll housing; target along gripper_frame +X/+Y toward board
        self.eye_local = list(cam.get("eye_xyz", [0.03, 0.05, -0.02]))
        self.target_local = list(cam.get("target_xyz", [0.16, 0.13, 0.02]))
        self.look_at = cam.get("look_at", "offset")
        self.lookahead = list(cam.get("lookahead", [0.05, 0.0, 0.0]))
        board = cfg.get("sim", {}).get("board", {})
        self.board_z = float(board.get("surface_z", 0.02)) + float(cam.get("board_z_offset", 0.01))
        self.up_link = cam.get("up_link", self.mount)
        self.up_local = list(cam.get("up_local", [0.0, 1.0, 0.0]))
        self.up = list(cam.get("up", [0.0, 0.0, 1.0]))
        self.w = int(cam.get("width", 320))
        self.h = int(cam.get("height", 240))
        self.fov = float(cam.get("fov", 75))
        self.near = float(cam.get("near", 0.01))
        self.far = float(cam.get("far", 1.0))
        self.hz = float(cam.get("preview_hz", 12))
        self._last = 0.0

    def _link_frame(self, name: str) -> tuple[list[float], list[float]]:
        idx = link_index(self.arm, name)
        st = p.getLinkState(self.arm.robot_id, idx, computeForwardKinematics=True)
        return list(st[4]), list(st[5])

    def _world_from_local(self, link: str, local_xyz: list[float]) -> list[float]:
        pos, orn = self._link_frame(link)
        return list(p.multiplyTransforms(pos, orn, local_xyz, IDENT)[0])

    def _camera_up(self) -> list[float]:
        if self.up_link:
            return self._world_from_local(self.up_link, self.up_local)
        return self.up

    def eye_target_world(self) -> tuple[list[float], list[float]]:
        """Recomputed every frame so the camera rides the arm."""
        eye = self._world_from_local(self.mount, self.eye_local)
        apos, aorn = self._link_frame(self.aim)
        if self.look_at == "board_plane":
            lead = p.multiplyTransforms(apos, aorn, self.lookahead, IDENT)[0]
            target = [lead[0], lead[1], self.board_z]
        else:
            target = p.multiplyTransforms(apos, aorn, self.target_local, IDENT)[0]
        return list(eye), list(target)

    def capture_rgbd(self, target: list[float] | None = None, up: list[float] | None = None) -> dict | None:
        """RGB, depth buffer, segmentation, and the matrices needed to unproject."""
        if not p.isConnected():
            return None
        eye, default_target = self.eye_target_world()
        target = list(target) if target is not None else default_target
        up = list(up) if up is not None else self._camera_up()
        view = p.computeViewMatrix(eye, target, up)
        proj = p.computeProjectionMatrixFOV(self.fov, self.w / self.h, self.near, self.far)
        _, _, rgb, depth, seg = p.getCameraImage(
            self.w, self.h, view, proj, renderer=p.ER_TINY_RENDERER
        )
        rgb = np.array(rgb, dtype=np.uint8).reshape(self.h, self.w, 4)[:, :, :3]
        depth = np.array(depth, dtype=np.float32).reshape(self.h, self.w)
        seg = np.array(seg, dtype=np.int32).reshape(self.h, self.w)
        return {
            "rgb": rgb,
            "depth": depth,
            "seg": seg,
            "view": view,
            "proj": proj,
            "near": self.near,
            "far": self.far,
            "eye": eye,
            "target": target,
        }

    def capture(self) -> np.ndarray | None:
        shot = self.capture_rgbd()
        return None if shot is None else shot["rgb"]

    def show(self) -> bool:
        if not p.isConnected():
            return False
        now = time.time()
        if now - self._last < 1.0 / self.hz:
            return True
        self._last = now
        import cv2

        frame = self.capture()
        if frame is None:
            return False
        cv2.imshow(WINDOW, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            cv2.destroyWindow(WINDOW)
            return False
        return True

    @staticmethod
    def close_window():
        import cv2

        try:
            cv2.destroyWindow(WINDOW)
        except cv2.error:
            pass
