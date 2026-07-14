from twin_core.aggregator import TwinAggregator
from twin_types.poses import ObjectPose, Pose3D


def test_aggregate_arm_and_object():
    agg = TwinAggregator(object_half_extent_m=0.05, near_contact_m=0.02)
    agg.update_arm([0.1] * 7, gripper_width=0.04, stamp_s=1.0)
    agg.update_objects(
        [
            ObjectPose(
                object_id="cube_a",
                pose_in_world=Pose3D.from_xyz_quat(0.5, 0.0, 0.05, 1, 0, 0, 0),
                stamp_s=1.0,
                confidence=1.0,
                tracking_ok=True,
            )
        ]
    )
    state = agg.build(
        stamp_s=1.05,
        tcp_pose_world=Pose3D.from_xyz_quat(0.5, 0.0, 0.15, 1, 0, 0, 0),
    )
    assert state.arm_qpos[0] == 0.1
    assert state.objects[0].tracking_ok is True
    assert state.grasp is not None
    assert state.grasp.distance_m > 0
