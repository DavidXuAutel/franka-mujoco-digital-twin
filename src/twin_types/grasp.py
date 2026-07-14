from __future__ import annotations

import numpy as np

from twin_types.poses import GraspMetrics, Pose3D


def compute_grasp_metrics(
    tcp_pose_world: Pose3D,
    object_pose_world: Pose3D,
    object_half_extent_m: float,
    near_contact_m: float = 0.02,
) -> GraspMetrics:
    tcp = tcp_pose_world.matrix[:3, 3]
    center = object_pose_world.matrix[:3, 3]
    center_dist = float(np.linalg.norm(tcp - center))
    distance = max(0.0, center_dist - float(object_half_extent_m))
    return GraspMetrics(distance_m=distance, near_contact=distance <= near_contact_m)
