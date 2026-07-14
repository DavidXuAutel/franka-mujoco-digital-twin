from __future__ import annotations

import json
from typing import Any

from twin_types.poses import ObjectPose, Pose3D

_REQUIRED_KEYS = ("object_id", "xyz_quat_wxyz", "stamp_s", "confidence", "tracking_ok")


def object_pose_to_dict(pose: ObjectPose) -> dict[str, Any]:
    """Pack an :class:`ObjectPose` into the wire dict published on ``object_poses``.

    Kept ROS-free (no rclpy/std_msgs import) so it can be unit tested on a
    machine without ROS2 installed, and shared between ``pose_node`` (packing)
    and ``twin_node`` (unpacking).
    """
    return {
        "object_id": pose.object_id,
        "xyz_quat_wxyz": [float(v) for v in pose.pose_in_world.xyz_quat_wxyz()],
        "stamp_s": float(pose.stamp_s),
        "confidence": float(pose.confidence),
        "tracking_ok": bool(pose.tracking_ok),
    }


def object_pose_to_json(pose: ObjectPose) -> str:
    return json.dumps(object_pose_to_dict(pose))


def dict_to_object_pose(raw: dict[str, Any]) -> ObjectPose:
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(f"object pose JSON missing keys: {missing}")
    xyz_quat = raw["xyz_quat_wxyz"]
    if len(xyz_quat) != 7:
        raise ValueError("xyz_quat_wxyz must have 7 numbers [x,y,z,qw,qx,qy,qz]")
    return ObjectPose(
        object_id=str(raw["object_id"]),
        pose_in_world=Pose3D.from_xyz_quat(*[float(v) for v in xyz_quat]),
        stamp_s=float(raw["stamp_s"]),
        confidence=float(raw["confidence"]),
        tracking_ok=bool(raw["tracking_ok"]),
    )


def json_to_object_pose(text: str) -> ObjectPose:
    return dict_to_object_pose(json.loads(text))
