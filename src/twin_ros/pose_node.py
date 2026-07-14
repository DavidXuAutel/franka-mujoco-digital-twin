#!/usr/bin/env python3
"""ROS2 node: AprilTag object pose estimation -> ``object_poses`` JSON topic.

Observation only: this node never publishes Franka commands. See
``docs/superpowers/specs/2026-07-14-online-digital-twin-design.md``.

Importable without ``rclpy`` installed. All ROS2-specific imports (rclpy,
sensor_msgs, std_msgs) are deferred into :func:`_build_node_class` /
:func:`main`, so plain ``import twin_ros.pose_node`` and
``python -m py_compile`` succeed on a machine without ROS2, and the pure
argument-parsing / image-decoding helpers below stay unit testable. JSON
wire packing lives in the ROS-free ``twin_ros.json_codec`` module.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from pose_backend.apriltag_backend import AprilTagBackend
from twin_ros.extrinsics import load_camera_extrinsics
from twin_ros.json_codec import object_pose_to_json
from twin_ros.topics import load_topics
from twin_types.object_library import load_object_spec

try:  # pragma: no cover - depends on host ROS/vision install
    import cv_bridge

    _CV_BRIDGE: Any = cv_bridge.CvBridge()
except Exception:  # noqa: BLE001
    _CV_BRIDGE = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AprilTag object pose estimation ROS2 node (observation only)"
    )
    parser.add_argument(
        "--object-yaml", required=True, help="Path to object library YAML (ObjectSpec)"
    )
    parser.add_argument(
        "--extrinsics-yaml",
        required=True,
        help="Path to camera_extrinsics.yaml (T_base_camera); aborts loudly if missing",
    )
    parser.add_argument(
        "--topics-yaml", default=None, help="Path to configs/topics.yaml (built-in defaults if omitted)"
    )
    parser.add_argument("--image-topic", default=None, help="Override the image topic from topics.yaml")
    parser.add_argument(
        "--camera-info-topic", default=None, help="Override the camera_info topic from topics.yaml"
    )
    parser.add_argument(
        "--object-poses-topic", default=None, help="Override the object_poses topic from topics.yaml"
    )
    parser.add_argument(
        "--lose-track-timeout-s",
        type=float,
        default=0.5,
        help="Passed through to AprilTagBackend (reserved for future eviction policy)",
    )
    return parser


def resolve_topics(args: argparse.Namespace) -> dict[str, str]:
    topics = load_topics(args.topics_yaml)
    if args.image_topic:
        topics["image"] = args.image_topic
    if args.camera_info_topic:
        topics["camera_info"] = args.camera_info_topic
    if args.object_poses_topic:
        topics["object_poses"] = args.object_poses_topic
    return topics


def image_msg_to_bgr(msg: Any) -> np.ndarray:
    """Convert a ``sensor_msgs/Image``-like object into an HxWx3 BGR ``uint8`` array.

    Prefers ``cv_bridge`` when importable. Otherwise builds the array manually
    from ``encoding`` + raw ``data``, supporting ``bgr8``, ``rgb8``, and
    ``mono8``. Duck-typed on ``height``/``width``/``encoding``/``data`` so it
    can be unit tested with a plain stand-in object instead of a real ROS
    message.
    """
    if _CV_BRIDGE is not None:
        return _CV_BRIDGE.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    encoding = msg.encoding
    height, width = msg.height, msg.width
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if encoding == "bgr8":
        return buf.reshape(height, width, 3).copy()
    if encoding == "rgb8":
        rgb = buf.reshape(height, width, 3)
        return rgb[:, :, ::-1].copy()
    if encoding == "mono8":
        gray = buf.reshape(height, width)
        return np.repeat(gray[:, :, None], 3, axis=2)
    raise ValueError(
        f"Unsupported image encoding for manual conversion (install cv_bridge for "
        f"broader support): {encoding!r}"
    )


def camera_info_to_intrinsics(msg: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(camera_matrix[3x3], dist_coeffs)`` from a ``sensor_msgs/CameraInfo``-like object.

    Accepts both the modern lowercase (``k``/``d``) and legacy uppercase
    (``K``/``D``) field names used across ROS2 message generator versions.
    """
    k = np.asarray(getattr(msg, "k", None) if hasattr(msg, "k") else msg.K, dtype=np.float64)
    d = np.asarray(getattr(msg, "d", None) if hasattr(msg, "d") else msg.D, dtype=np.float64)
    camera_matrix = k.reshape(3, 3)
    dist_coeffs = d if d.size > 0 else np.zeros(5, dtype=np.float64)
    return camera_matrix, dist_coeffs


def _build_node_class():
    """Deferred rclpy import; returns ``(PoseNode, rclpy)``.

    Keeping this behind a function call (rather than at module scope) means
    the class body — which subclasses ``rclpy.node.Node`` — is only
    evaluated when ROS2 is actually installed.
    """
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String

    sensor_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
    )

    class PoseNode(Node):
        def __init__(self, backend: AprilTagBackend, topics: dict[str, str]) -> None:
            super().__init__("twin_pose_node")
            self._backend = backend
            self._camera_matrix: np.ndarray | None = None
            self._dist_coeffs: np.ndarray | None = None
            self._publisher = self.create_publisher(String, topics["object_poses"], 10)
            self.create_subscription(Image, topics["image"], self._on_image, sensor_qos)
            self.create_subscription(
                CameraInfo, topics["camera_info"], self._on_camera_info, sensor_qos
            )
            self.get_logger().info(
                f"twin_pose_node: image={topics['image']} camera_info={topics['camera_info']} "
                f"-> publishing {topics['object_poses']}"
            )

        def _on_camera_info(self, msg: "CameraInfo") -> None:
            self._camera_matrix, self._dist_coeffs = camera_info_to_intrinsics(msg)

        def _on_image(self, msg: "Image") -> None:
            if self._camera_matrix is None:
                return
            stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            try:
                image_bgr = image_msg_to_bgr(msg)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"failed to decode image: {exc!r}")
                return
            poses = self._backend.estimate(
                image_bgr=image_bgr,
                camera_matrix=self._camera_matrix,
                dist_coeffs=self._dist_coeffs,
                stamp_s=stamp_s,
            )
            for pose in poses:
                out = String()
                out.data = object_pose_to_json(pose)
                self._publisher.publish(out)

    return PoseNode, rclpy


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    topics = resolve_topics(args)

    # Fail loud on missing/invalid extrinsics or object spec before touching ROS.
    object_spec = load_object_spec(Path(args.object_yaml))
    T_base_camera = load_camera_extrinsics(Path(args.extrinsics_yaml))
    backend = AprilTagBackend(
        object_spec=object_spec,
        T_base_camera=T_base_camera,
        lose_track_timeout_s=args.lose_track_timeout_s,
    )

    try:
        PoseNode, rclpy = _build_node_class()
    except ImportError as exc:
        raise SystemExit(
            "rclpy (ROS2 Python client library) is required to run pose_node as a "
            f"live ROS2 node. Import failed: {exc!r}"
        ) from exc

    rclpy.init(args=argv)
    node = PoseNode(backend, topics)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
