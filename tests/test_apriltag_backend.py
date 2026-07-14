import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from pose_backend.apriltag_backend import AprilTagBackend
from twin_types.object_library import ObjectSpec
from twin_types.poses import Pose3D


def _synthetic_intrinsics():
    fx = fy = 600.0
    cx = cy = 320.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)


def test_detect_single_tag_identity_extrinsics():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag = cv2.aruco.generateImageMarker(dictionary, 0, 200)
    img = np.full((480, 640), 255, dtype=np.uint8)
    img[140:340, 220:420] = tag

    spec = ObjectSpec(
        object_id="cube_a",
        mesh="meshes/cube_a.obj",
        mujoco_body="object_cube_a",
        tag_family="tag36h11",
        tag_id=0,
        tag_size_m=0.04,
        T_object_tag=Pose3D.identity(),
    )
    backend = AprilTagBackend(
        object_spec=spec,
        T_base_camera=Pose3D.identity(),
        lose_track_timeout_s=0.5,
    )
    poses = backend.estimate(
        image_bgr=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
        camera_matrix=_synthetic_intrinsics(),
        dist_coeffs=np.zeros(5),
        stamp_s=1.0,
    )
    assert len(poses) == 1
    assert poses[0].object_id == "cube_a"
    assert poses[0].tracking_ok is True


def test_miss_holds_last_pose_within_timeout():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag = cv2.aruco.generateImageMarker(dictionary, 0, 200)
    img_with_tag = np.full((480, 640), 255, dtype=np.uint8)
    img_with_tag[140:340, 220:420] = tag
    img_blank = np.full((480, 640), 255, dtype=np.uint8)

    spec = ObjectSpec(
        object_id="cube_a",
        mesh="meshes/cube_a.obj",
        mujoco_body="object_cube_a",
        tag_family="tag36h11",
        tag_id=0,
        tag_size_m=0.04,
        T_object_tag=Pose3D.identity(),
    )
    backend = AprilTagBackend(
        object_spec=spec,
        T_base_camera=Pose3D.identity(),
        lose_track_timeout_s=0.5,
    )
    first = backend.estimate(
        image_bgr=cv2.cvtColor(img_with_tag, cv2.COLOR_GRAY2BGR),
        camera_matrix=_synthetic_intrinsics(),
        dist_coeffs=np.zeros(5),
        stamp_s=1.0,
    )
    assert len(first) == 1
    assert first[0].tracking_ok is True

    held = backend.estimate(
        image_bgr=cv2.cvtColor(img_blank, cv2.COLOR_GRAY2BGR),
        camera_matrix=_synthetic_intrinsics(),
        dist_coeffs=np.zeros(5),
        stamp_s=1.2,
    )
    assert len(held) == 1
    assert held[0].object_id == "cube_a"
    assert held[0].tracking_ok is False
    np.testing.assert_allclose(
        held[0].pose_in_world.as_matrix(), first[0].pose_in_world.as_matrix()
    )


def test_miss_without_any_prior_detection_returns_empty():
    img_blank = np.full((480, 640), 255, dtype=np.uint8)

    spec = ObjectSpec(
        object_id="cube_a",
        mesh="meshes/cube_a.obj",
        mujoco_body="object_cube_a",
        tag_family="tag36h11",
        tag_id=0,
        tag_size_m=0.04,
        T_object_tag=Pose3D.identity(),
    )
    backend = AprilTagBackend(
        object_spec=spec,
        T_base_camera=Pose3D.identity(),
        lose_track_timeout_s=0.5,
    )
    poses = backend.estimate(
        image_bgr=cv2.cvtColor(img_blank, cv2.COLOR_GRAY2BGR),
        camera_matrix=_synthetic_intrinsics(),
        dist_coeffs=np.zeros(5),
        stamp_s=1.0,
    )
    assert poses == []


def test_miss_after_timeout_still_holds_last_pose():
    """MVP intentionally holds the last pose forever, not only within the timeout.

    ``lose_track_timeout_s`` is stored but not enforced yet; a miss long after
    that window still returns the held pose with ``tracking_ok=False``.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag = cv2.aruco.generateImageMarker(dictionary, 0, 200)
    img_with_tag = np.full((480, 640), 255, dtype=np.uint8)
    img_with_tag[140:340, 220:420] = tag
    img_blank = np.full((480, 640), 255, dtype=np.uint8)

    spec = ObjectSpec(
        object_id="cube_a",
        mesh="meshes/cube_a.obj",
        mujoco_body="object_cube_a",
        tag_family="tag36h11",
        tag_id=0,
        tag_size_m=0.04,
        T_object_tag=Pose3D.identity(),
    )
    backend = AprilTagBackend(
        object_spec=spec,
        T_base_camera=Pose3D.identity(),
        lose_track_timeout_s=0.5,
    )
    first = backend.estimate(
        image_bgr=cv2.cvtColor(img_with_tag, cv2.COLOR_GRAY2BGR),
        camera_matrix=_synthetic_intrinsics(),
        dist_coeffs=np.zeros(5),
        stamp_s=1.0,
    )

    held = backend.estimate(
        image_bgr=cv2.cvtColor(img_blank, cv2.COLOR_GRAY2BGR),
        camera_matrix=_synthetic_intrinsics(),
        dist_coeffs=np.zeros(5),
        stamp_s=10.0,  # well past lose_track_timeout_s
    )
    assert len(held) == 1
    assert held[0].tracking_ok is False
    np.testing.assert_allclose(
        held[0].pose_in_world.as_matrix(), first[0].pose_in_world.as_matrix()
    )
