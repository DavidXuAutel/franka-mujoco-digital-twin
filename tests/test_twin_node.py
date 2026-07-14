"""Unit tests for the ROS-free helpers in twin_ros.twin_node.

twin_node.py defers all rclpy/sensor_msgs imports into `_build_node_class()`
/ `main()`, so this module must import and its helpers must run fine without
ROS2 installed.
"""
from __future__ import annotations

from pathlib import Path

from twin_ros import twin_node

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_build_arg_parser_defaults():
    args = twin_node.build_arg_parser().parse_args([])
    assert args.model.endswith("scene_mvp.xml")
    assert args.object_body == "object_cube_a"
    assert args.object_half_extent_m is None
    assert args.tcp_name == "tcp_proxy"
    assert args.rate_hz == 30.0
    assert args.object_stale_s == twin_node.DEFAULT_OBJECT_STALE_S


def test_resolve_half_extent_default_is_fixed_value():
    args = twin_node.build_arg_parser().parse_args([])
    assert twin_node.resolve_half_extent(args) == twin_node.DEFAULT_OBJECT_HALF_EXTENT_M


def test_resolve_half_extent_explicit_override():
    args = twin_node.build_arg_parser().parse_args(["--object-half-extent-m", "0.03"])
    assert twin_node.resolve_half_extent(args) == 0.03


def test_resolve_half_extent_from_object_yaml():
    object_yaml = _CONFIGS / "objects/cube_a.yaml"
    args = twin_node.build_arg_parser().parse_args(["--object-yaml", str(object_yaml)])
    # cube_a.yaml: tag_size_m: 0.04 -> half extent 0.02
    assert abs(twin_node.resolve_half_extent(args) - 0.02) < 1e-9


def test_parse_joint_positions_success():
    names = ["fr3_joint1", "fr3_joint2"]
    msg_names = ["fr3_joint2", "fr3_joint1", "fr3_finger_joint1"]
    msg_positions = [0.2, 0.1, 0.04]
    result = twin_node.parse_joint_positions(names, msg_names, msg_positions)
    assert result == [0.1, 0.2]


def test_parse_joint_positions_missing_returns_none():
    result = twin_node.parse_joint_positions(["fr3_joint1"], ["fr3_joint2"], [0.1])
    assert result is None


def test_parse_finger_position_found():
    result = twin_node.parse_finger_position(["fr3_finger_joint1", "fr3_finger_joint2"], [0.02, 0.02])
    assert result == 0.02


def test_parse_finger_position_missing_returns_none():
    result = twin_node.parse_finger_position(["other_joint"], [0.5])
    assert result is None
