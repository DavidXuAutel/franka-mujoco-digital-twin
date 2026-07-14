from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_TOPICS: dict[str, str] = {
    "image": "/cam1/cam1/color/image_raw",
    "camera_info": "/cam1/cam1/color/camera_info",
    "joint_states": "/franka_robot_state_broadcaster/measured_joint_states",
    "gripper_joint_states": "/franka_gripper/joint_states",
    "object_poses": "/twin/object_poses",
}


def load_topics(path: str | Path | None) -> dict[str, str]:
    """Load topic name overrides from ``configs/topics.yaml``.

    Always returns a dict containing every key in :data:`DEFAULT_TOPICS`, with
    any values present in the YAML file overriding the default. ``path=None``
    returns the defaults unchanged (ROS-free — no rclpy import required).
    """
    topics = dict(DEFAULT_TOPICS)
    if path is None:
        return topics
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"topics config not found: {resolved}")
    raw = yaml.safe_load(resolved.read_text()) or {}
    topics.update({str(k): str(v) for k, v in raw.items()})
    return topics
