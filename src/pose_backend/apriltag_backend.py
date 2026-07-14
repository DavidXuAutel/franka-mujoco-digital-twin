from __future__ import annotations

import cv2
import numpy as np

from twin_types.object_library import ObjectSpec
from twin_types.poses import ObjectPose, Pose3D
from twin_types.transforms import compose, invert

_TAG_FAMILY_TO_DICT = {
    "tag36h11": "DICT_APRILTAG_36h11",
}


def _resolve_dictionary(tag_family: str) -> cv2.aruco.Dictionary:
    attr_name = _TAG_FAMILY_TO_DICT.get(tag_family)
    if attr_name is None or not hasattr(cv2.aruco, attr_name):
        raise ValueError(f"Unsupported tag family: {tag_family!r}")
    dict_id = getattr(cv2.aruco, attr_name)
    return cv2.aruco.getPredefinedDictionary(dict_id)


class AprilTagBackend:
    """PoseBackend that detects a single AprilTag per object via cv2.aruco.

    On a missed detection, the last successfully observed pose is held indefinitely
    with ``tracking_ok=False``. There is no time-based eviction in the MVP.

    Adapts to whichever cv2.aruco API is installed:
    - Detection: the modern ``cv2.aruco.ArucoDetector`` class if available,
      otherwise the legacy free-function ``cv2.aruco.detectMarkers``.
    - Pose estimation: the legacy ``cv2.aruco.estimatePoseSingleMarkers`` if
      available, otherwise ``cv2.solvePnP`` with ``SOLVEPNP_IPPE_SQUARE``
      (which the legacy helper itself is a thin wrapper around).
    """

    def __init__(
        self,
        object_spec: ObjectSpec,
        T_base_camera: Pose3D,
        lose_track_timeout_s: float = 0.5,
    ) -> None:
        """Configure detection for one known object.

        Args:
            object_spec: Known object and tag metadata.
            T_base_camera: Static extrinsic from the robot/world base frame to the
                camera.
            lose_track_timeout_s: Reserved for a future eviction policy once the tag
                has been unseen for this duration. Currently unused by design: on
                miss, the last pose is held forever with ``tracking_ok=False``.
        """
        self._spec = object_spec
        self._T_base_camera = T_base_camera
        self._lose_track_timeout_s = lose_track_timeout_s

        self._dictionary = _resolve_dictionary(object_spec.tag_family)
        self._detector = None
        if hasattr(cv2.aruco, "ArucoDetector"):
            params = cv2.aruco.DetectorParameters()
            self._detector = cv2.aruco.ArucoDetector(self._dictionary, params)

        half = object_spec.tag_size_m / 2.0
        self._object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )

        self._last_pose: ObjectPose | None = None
        self._last_seen_stamp_s: float | None = None

    def _detect_markers(self, gray: np.ndarray):
        if self._detector is not None:
            corners, ids, _rejected = self._detector.detectMarkers(gray)
        else:
            corners, ids, _rejected = cv2.aruco.detectMarkers(gray, self._dictionary)
        return corners, ids

    def _estimate_tag_pose(
        self,
        corners_for_tag: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> Pose3D | None:
        if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
            rvecs, tvecs, _obj_pts = cv2.aruco.estimatePoseSingleMarkers(
                [corners_for_tag], self._spec.tag_size_m, camera_matrix, dist_coeffs
            )
            rvec = rvecs[0]
            tvec = tvecs[0]
        else:
            image_points = np.asarray(corners_for_tag, dtype=np.float64).reshape(4, 2)
            ok, rvec, tvec = cv2.solvePnP(
                self._object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                return None

        rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
        return Pose3D(matrix)

    def estimate(
        self,
        image_bgr: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        stamp_s: float,
    ) -> list[ObjectPose]:
        if image_bgr.ndim == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr

        corners, ids = self._detect_markers(gray)

        T_cam_tag: Pose3D | None = None
        if ids is not None:
            ids_flat = np.asarray(ids).reshape(-1)
            matches = np.flatnonzero(ids_flat == self._spec.tag_id)
            if matches.size > 0:
                T_cam_tag = self._estimate_tag_pose(
                    corners[int(matches[0])], camera_matrix, dist_coeffs
                )

        if T_cam_tag is not None:
            T_base_tag = compose(self._T_base_camera, T_cam_tag)
            T_base_object = compose(T_base_tag, invert(self._spec.T_object_tag))
            object_pose = ObjectPose(
                object_id=self._spec.object_id,
                pose_in_world=T_base_object,
                stamp_s=stamp_s,
                confidence=1.0,
                tracking_ok=True,
            )
            self._last_pose = object_pose
            self._last_seen_stamp_s = stamp_s
            return [object_pose]

        if self._last_pose is None:
            return []

        held_pose = ObjectPose(
            object_id=self._last_pose.object_id,
            pose_in_world=self._last_pose.pose_in_world,
            stamp_s=stamp_s,
            confidence=0.0,
            tracking_ok=False,
        )
        return [held_pose]
