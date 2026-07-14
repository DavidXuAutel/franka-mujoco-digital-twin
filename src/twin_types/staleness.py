"""Stale object-pose handling when the camera / pose stream stops."""
from __future__ import annotations

from twin_types.poses import ObjectPose


def apply_pose_staleness(
    pose: ObjectPose | None,
    now_s: float,
    max_age_s: float,
) -> ObjectPose | None:
    """Return pose with ``tracking_ok=False`` if its stamp is older than ``max_age_s``.

    Used when ``pose_node`` stops publishing (camera death): twin_node keeps the
    last message but must mark the object lost after ``max_age_s``.
    """
    if pose is None:
        return None
    if max_age_s < 0:
        raise ValueError("max_age_s must be >= 0")
    age = now_s - pose.stamp_s
    if age <= max_age_s:
        return pose
    if not pose.tracking_ok and pose.confidence == 0.0:
        return pose
    return ObjectPose(
        object_id=pose.object_id,
        pose_in_world=pose.pose_in_world,
        stamp_s=pose.stamp_s,
        confidence=0.0,
        tracking_ok=False,
    )
