"""Unit tests for the ROS-free helpers in twin_ros.pose_node.

Deliberately does not import rclpy: pose_node.py defers all ROS2 imports
into `_build_node_class()` / `main()`, so this module (and its argparse /
image-decoding helpers) must be importable and testable without ROS2
installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from twin_ros import pose_node


def test_build_arg_parser_requires_object_and_extrinsics_yaml():
    parser = pose_node.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["--object-yaml", "obj.yaml", "--extrinsics-yaml", "ext.yaml"])
    assert args.object_yaml == "obj.yaml"
    assert args.extrinsics_yaml == "ext.yaml"
    assert args.topics_yaml is None
    assert args.lose_track_timeout_s == 0.5


def test_resolve_topics_applies_overrides():
    parser = pose_node.build_arg_parser()
    args = parser.parse_args(
        [
            "--object-yaml",
            "obj.yaml",
            "--extrinsics-yaml",
            "ext.yaml",
            "--image-topic",
            "/my/image",
        ]
    )
    topics = pose_node.resolve_topics(args)
    assert topics["image"] == "/my/image"
    assert topics["camera_info"] == "/cam1/cam1/color/camera_info"


@dataclass
class _FakeImage:
    height: int
    width: int
    encoding: str
    data: bytes = field(default=b"")


def test_image_msg_to_bgr_bgr8_roundtrip():
    height, width = 4, 3
    arr = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    msg = _FakeImage(height=height, width=width, encoding="bgr8", data=arr.tobytes())
    out = pose_node.image_msg_to_bgr(msg)
    np.testing.assert_array_equal(out, arr)


def test_image_msg_to_bgr_rgb8_swaps_channels():
    height, width = 2, 2
    arr = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    msg = _FakeImage(height=height, width=width, encoding="rgb8", data=arr.tobytes())
    out = pose_node.image_msg_to_bgr(msg)
    np.testing.assert_array_equal(out, arr[:, :, ::-1])


def test_image_msg_to_bgr_mono8_expands_channels():
    height, width = 2, 2
    arr = np.arange(height * width, dtype=np.uint8).reshape(height, width)
    msg = _FakeImage(height=height, width=width, encoding="mono8", data=arr.tobytes())
    out = pose_node.image_msg_to_bgr(msg)
    assert out.shape == (height, width, 3)
    np.testing.assert_array_equal(out[:, :, 0], arr)


def test_image_msg_to_bgr_unsupported_encoding_raises():
    msg = _FakeImage(height=1, width=1, encoding="yuv422", data=b"\x00\x00")
    with pytest.raises(ValueError):
        pose_node.image_msg_to_bgr(msg)


@dataclass
class _FakeCameraInfo:
    k: list[float]
    d: list[float]


def test_camera_info_to_intrinsics():
    fx = fy = 600.0
    cx = cy = 320.0
    msg = _FakeCameraInfo(k=[fx, 0, cx, 0, fy, cy, 0, 0, 1], d=[0.0, 0.0, 0.0, 0.0, 0.0])
    camera_matrix, dist_coeffs = pose_node.camera_info_to_intrinsics(msg)
    assert camera_matrix.shape == (3, 3)
    assert camera_matrix[0, 0] == fx
    assert camera_matrix[1, 1] == fy
    assert dist_coeffs.shape == (5,)


def test_camera_info_to_intrinsics_empty_dist_defaults_to_zeros():
    msg = _FakeCameraInfo(k=[1, 0, 0, 0, 1, 0, 0, 0, 1], d=[])
    _camera_matrix, dist_coeffs = pose_node.camera_info_to_intrinsics(msg)
    np.testing.assert_array_equal(dist_coeffs, np.zeros(5))
