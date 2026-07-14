from __future__ import annotations

import numpy as np
import mujoco
from twin_types.poses import TwinState


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
            x, y, z, qw, qx, qy, qz = state.objects[0].pose_in_world.xyz_quat_wxyz()
            self.data.mocap_quat[self._mocap_id] = [qw, qx, qy, qz]
        mujoco.mj_forward(self.model, self.data)

    def object_xpos(self) -> np.ndarray:
        return self.data.xpos[self._body_id].copy()
