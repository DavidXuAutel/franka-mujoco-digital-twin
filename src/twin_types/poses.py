from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Pose3D:
    """Rigid transform. Quaternion is (w, x, y, z)."""

    matrix: np.ndarray  # 4x4

    @staticmethod
    def identity() -> Pose3D:
        return Pose3D(np.eye(4))

    @staticmethod
    def from_xyz_quat(
        x: float, y: float, z: float, qw: float, qx: float, qy: float, qz: float
    ) -> Pose3D:
        n = np.array([qw, qx, qy, qz], dtype=float)
        n = n / np.linalg.norm(n)
        w, xq, yq, zq = n
        r = np.array(
            [
                [1 - 2 * (yq * yq + zq * zq), 2 * (xq * yq - zq * w), 2 * (xq * zq + yq * w)],
                [2 * (xq * yq + zq * w), 1 - 2 * (xq * xq + zq * zq), 2 * (yq * zq - xq * w)],
                [2 * (xq * zq - yq * w), 2 * (yq * zq + xq * w), 1 - 2 * (xq * xq + yq * yq)],
            ],
            dtype=float,
        )
        m = np.eye(4)
        m[:3, :3] = r
        m[:3, 3] = [x, y, z]
        return Pose3D(m)

    def as_matrix(self) -> np.ndarray:
        return self.matrix.copy()

    def xyz_quat_wxyz(self) -> tuple[float, float, float, float, float, float, float]:
        r = self.matrix[:3, :3]
        t = self.matrix[:3, 3]
        trace = float(np.trace(r))
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (r[2, 1] - r[1, 2]) * s
            y = (r[0, 2] - r[2, 0]) * s
            z = (r[1, 0] - r[0, 1]) * s
        else:
            # stable branch for rare cases
            i = int(np.argmax([r[0, 0], r[1, 1], r[2, 2]]))
            if i == 0:
                s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
                w = (r[2, 1] - r[1, 2]) / s
                x = 0.25 * s
                y = (r[0, 1] + r[1, 0]) / s
                z = (r[0, 2] + r[2, 0]) / s
            elif i == 1:
                s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
                w = (r[0, 2] - r[2, 0]) / s
                x = (r[0, 1] + r[1, 0]) / s
                y = 0.25 * s
                z = (r[1, 2] + r[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
                w = (r[1, 0] - r[0, 1]) / s
                x = (r[0, 2] + r[2, 0]) / s
                y = (r[1, 2] + r[2, 1]) / s
                z = 0.25 * s
        return (float(t[0]), float(t[1]), float(t[2]), float(w), float(x), float(y), float(z))


@dataclass(frozen=True)
class ObjectPose:
    object_id: str
    pose_in_world: Pose3D
    stamp_s: float
    confidence: float
    tracking_ok: bool


@dataclass(frozen=True)
class GraspMetrics:
    distance_m: float
    near_contact: bool


@dataclass
class TwinState:
    arm_qpos: list[float]  # len 7
    gripper_width: float
    objects: list[ObjectPose]
    grasp: GraspMetrics | None
    stamp_s: float
    latency_s: float | None = None
