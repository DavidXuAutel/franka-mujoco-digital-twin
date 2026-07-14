from __future__ import annotations

from twin_types.grasp import compute_grasp_metrics
from twin_types.poses import GraspMetrics, ObjectPose, Pose3D, TwinState


class TwinAggregator:
    def __init__(self, object_half_extent_m: float, near_contact_m: float = 0.02) -> None:
        self.object_half_extent_m = object_half_extent_m
        self.near_contact_m = near_contact_m
        self._arm: list[float] | None = None
        self._gripper: float = 0.0
        self._arm_stamp: float | None = None
        self._objects: list[ObjectPose] = []

    def update_arm(self, qpos7: list[float], gripper_width: float, stamp_s: float) -> None:
        if len(qpos7) != 7:
            raise ValueError("arm qpos must have length 7")
        self._arm = [float(v) for v in qpos7]
        self._gripper = float(gripper_width)
        self._arm_stamp = stamp_s

    def update_objects(self, objects: list[ObjectPose]) -> None:
        self._objects = list(objects)

    def build(self, stamp_s: float, tcp_pose_world: Pose3D | None) -> TwinState:
        if self._arm is None:
            raise RuntimeError("arm state not yet received")
        grasp: GraspMetrics | None = None
        if tcp_pose_world is not None and self._objects:
            grasp = compute_grasp_metrics(
                tcp_pose_world,
                self._objects[0].pose_in_world,
                object_half_extent_m=self.object_half_extent_m,
                near_contact_m=self.near_contact_m,
            )
        latency = None
        if self._arm_stamp is not None:
            latency = max(0.0, stamp_s - self._arm_stamp)
        return TwinState(
            arm_qpos=list(self._arm),
            gripper_width=self._gripper,
            objects=list(self._objects),
            grasp=grasp,
            stamp_s=stamp_s,
            latency_s=latency,
        )
