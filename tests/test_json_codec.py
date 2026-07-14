import pytest

from twin_ros.json_codec import (
    dict_to_object_pose,
    json_to_object_pose,
    object_pose_to_dict,
    object_pose_to_json,
)
from twin_types.poses import ObjectPose, Pose3D


def _sample_pose() -> ObjectPose:
    return ObjectPose(
        object_id="cube_a",
        pose_in_world=Pose3D.from_xyz_quat(0.2, 0.1, 0.05, 1.0, 0.0, 0.0, 0.0),
        stamp_s=12.5,
        confidence=0.9,
        tracking_ok=True,
    )


def test_object_pose_to_dict_shape():
    raw = object_pose_to_dict(_sample_pose())
    assert raw["object_id"] == "cube_a"
    assert len(raw["xyz_quat_wxyz"]) == 7
    assert raw["stamp_s"] == 12.5
    assert raw["confidence"] == 0.9
    assert raw["tracking_ok"] is True


def test_json_roundtrip():
    pose = _sample_pose()
    text = object_pose_to_json(pose)
    recovered = json_to_object_pose(text)
    assert recovered.object_id == pose.object_id
    assert recovered.stamp_s == pose.stamp_s
    assert recovered.confidence == pose.confidence
    assert recovered.tracking_ok == pose.tracking_ok
    import numpy as np

    np.testing.assert_allclose(
        recovered.pose_in_world.matrix, pose.pose_in_world.matrix, atol=1e-9
    )


def test_dict_to_object_pose_missing_key_raises():
    raw = object_pose_to_dict(_sample_pose())
    del raw["confidence"]
    with pytest.raises(ValueError):
        dict_to_object_pose(raw)


def test_dict_to_object_pose_bad_length_raises():
    raw = object_pose_to_dict(_sample_pose())
    raw["xyz_quat_wxyz"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        dict_to_object_pose(raw)
