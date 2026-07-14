from __future__ import annotations

import numpy as np
import mujoco
from twin_types.poses import TwinState

# RGBA for object geom feedback (tracking / grasp).
_RGBA_TRACKING_OK = np.array([0.9, 0.4, 0.1, 1.0], dtype=np.float64)
_RGBA_TRACKING_LOST = np.array([0.85, 0.15, 0.15, 1.0], dtype=np.float64)
_RGBA_NEAR_CONTACT = np.array([0.15, 0.85, 0.25, 1.0], dtype=np.float64)


class TwinMujocoDriver:
    def __init__(self, model_path: str, object_body: str) -> None:
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.object_body = object_body
        self._body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_body)
        if self._body_id < 0:
            raise RuntimeError(f"body not found: {object_body}")
        self._mocap_id = int(self.model.body_mocapid[self._body_id])
        if self._mocap_id < 0:
            raise RuntimeError(f"body {object_body} is not mocap")

        # Prefer geom named ``{body}_geom``; else first geom on the body.
        preferred = f"{object_body}_geom"
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, preferred)
        if gid < 0:
            gid = -1
            for i in range(self.model.ngeom):
                if int(self.model.geom_bodyid[i]) == self._body_id:
                    gid = i
                    break
        self._object_geom_id = gid

        # Optional FR3 joints if present
        self._joint_qpos: dict[str, int] = {}
        for i in range(1, 8):
            name = f"fr3_joint{i}"
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self._joint_qpos[name] = self.model.jnt_qposadr[jid]
        self._finger_qpos: list[int] = []
        for name in ("fr3_finger_joint1", "fr3_finger_joint2"):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self._finger_qpos.append(self.model.jnt_qposadr[jid])

    def apply(self, state: TwinState) -> None:
        if len(self._joint_qpos) == 7:
            for idx, name in enumerate([f"fr3_joint{i}" for i in range(1, 8)]):
                self.data.qpos[self._joint_qpos[name]] = state.arm_qpos[idx]
        for adr in self._finger_qpos:
            half = max(0.0, min(0.04, state.gripper_width))
            self.data.qpos[adr] = half
        if state.objects:
            T = state.objects[0].pose_in_world.matrix
            self.data.mocap_pos[self._mocap_id] = T[:3, 3]
            # convert rotm to quat wxyz for mocap_quat
            _x, _y, _z, qw, qx, qy, qz = state.objects[0].pose_in_world.xyz_quat_wxyz()
            self.data.mocap_quat[self._mocap_id] = [qw, qx, qy, qz]
            self._apply_object_rgba(state)
        mujoco.mj_forward(self.model, self.data)

    def _apply_object_rgba(self, state: TwinState) -> None:
        if self._object_geom_id < 0 or not state.objects:
            return
        obj = state.objects[0]
        if not obj.tracking_ok:
            rgba = _RGBA_TRACKING_LOST
        elif state.grasp is not None and state.grasp.near_contact:
            rgba = _RGBA_NEAR_CONTACT
        else:
            rgba = _RGBA_TRACKING_OK
        self.model.geom_rgba[self._object_geom_id] = rgba

    def object_xpos(self) -> np.ndarray:
        return self.data.xpos[self._body_id].copy()

    def object_rgba(self) -> np.ndarray:
        if self._object_geom_id < 0:
            raise RuntimeError("object geom not found")
        return self.model.geom_rgba[self._object_geom_id].copy()
