from twin_types.poses import ObjectPose, Pose3D
from twin_types.staleness import apply_pose_staleness


def test_fresh_pose_unchanged():
    pose = ObjectPose("cube_a", Pose3D.identity(), stamp_s=10.0, confidence=1.0, tracking_ok=True)
    out = apply_pose_staleness(pose, now_s=10.2, max_age_s=0.5)
    assert out is pose
    assert out.tracking_ok is True


def test_stale_pose_marked_lost():
    pose = ObjectPose("cube_a", Pose3D.identity(), stamp_s=10.0, confidence=1.0, tracking_ok=True)
    out = apply_pose_staleness(pose, now_s=11.0, max_age_s=0.5)
    assert out is not None
    assert out.tracking_ok is False
    assert out.confidence == 0.0
    assert out.stamp_s == 10.0


def test_none_passthrough():
    assert apply_pose_staleness(None, now_s=1.0, max_age_s=0.5) is None
