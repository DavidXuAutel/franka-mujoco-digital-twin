import pytest

mujoco = pytest.importorskip("mujoco")
from pathlib import Path
from twin_mujoco.driver import TwinMujocoDriver
from twin_types.poses import GraspMetrics, ObjectPose, Pose3D, TwinState


def test_apply_object_pose():
    xml = Path(__file__).resolve().parents[1] / "src/twin_mujoco/scene_mvp.xml"
    driver = TwinMujocoDriver(str(xml), object_body="object_cube_a")
    state = TwinState(
        arm_qpos=[0.0] * 7,
        gripper_width=0.04,
        objects=[
            ObjectPose(
                "cube_a",
                Pose3D.from_xyz_quat(0.2, 0.1, 0.05, 1, 0, 0, 0),
                stamp_s=1.0,
                confidence=1.0,
                tracking_ok=True,
            )
        ],
        grasp=GraspMetrics(0.1, False),
        stamp_s=1.0,
    )
    driver.apply(state)
    pos = driver.object_xpos()
    assert abs(pos[0] - 0.2) < 1e-6
    assert abs(pos[1] - 0.1) < 1e-6
