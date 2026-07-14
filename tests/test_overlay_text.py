from twin_mujoco.viewer_app import format_overlay
from twin_types.poses import GraspMetrics, ObjectPose, Pose3D, TwinState


def test_format_overlay_near_contact():
    state = TwinState(
        arm_qpos=[0] * 7,
        gripper_width=0.02,
        objects=[ObjectPose("cube_a", Pose3D.identity(), 1.0, 1.0, True)],
        grasp=GraspMetrics(0.01, True),
        stamp_s=1.0,
        latency_s=0.04,
    )
    text = format_overlay(state)
    assert "NEAR" in text
    assert "ok=True" in text or "tracking_ok=True" in text
