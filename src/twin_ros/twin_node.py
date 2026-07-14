#!/usr/bin/env python3
"""ROS2 node: mirror Franka arm + gripper + tracked object into MuJoCo.

Observation only: this node never publishes Franka motion commands. See
``docs/superpowers/specs/2026-07-14-online-digital-twin-design.md``.

Importable without ``rclpy`` installed. All ROS2-specific imports (rclpy,
sensor_msgs, std_msgs) are deferred into :func:`_build_node_class` /
:func:`main`, so plain ``import twin_ros.twin_node`` and
``python -m py_compile`` succeed on a machine without ROS2, and the pure
argument-parsing / joint-name-matching helpers below stay unit testable.

Viewer loop pattern (rclpy spin in a daemon thread, MuJoCo passive viewer on
the main thread) is ported from
``~/Projects/franka_teleop_stable/gello_desk/mujoco_ros_mirror.py``.
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

from twin_core.aggregator import TwinAggregator
from twin_mujoco.driver import TwinMujocoDriver
from twin_mujoco.viewer_app import format_overlay
from twin_ros.json_codec import json_to_object_pose
from twin_ros.tcp_fk import get_tcp_pose
from twin_ros.topics import load_topics
from twin_types.object_library import load_object_spec
from twin_types.poses import ObjectPose

_DEFAULT_MODEL = str(Path(__file__).resolve().parents[1] / "twin_mujoco" / "scene_mvp.xml")
FRANKA_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
GRIPPER_FINGER_JOINTS = ["fr3_finger_joint1", "fr3_finger_joint2"]
FINGER_OPEN_M = 0.04
DEFAULT_OBJECT_HALF_EXTENT_M = 0.05


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Digital twin viewer ROS2 node (observation only, no Franka commands)"
    )
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="MuJoCo XML model path")
    parser.add_argument(
        "--object-body", default="object_cube_a", help="Mocap body name for the tracked object"
    )
    parser.add_argument(
        "--object-yaml",
        default=None,
        help="Object library YAML; if given, half-extent defaults to tag_size_m / 2",
    )
    parser.add_argument(
        "--object-half-extent-m",
        type=float,
        default=None,
        help="Override half-extent used for grasp metrics (default: tag_size_m/2 or 0.05 m)",
    )
    parser.add_argument("--topics-yaml", default=None, help="Path to configs/topics.yaml")
    parser.add_argument("--near-contact-m", type=float, default=0.02)
    parser.add_argument(
        "--tcp-name",
        default="tcp_proxy",
        help="MuJoCo site/body name used as the TCP proxy for grasp metrics",
    )
    parser.add_argument("--rate-hz", type=float, default=30.0)
    return parser


def resolve_half_extent(args: argparse.Namespace) -> float:
    if args.object_half_extent_m is not None:
        return float(args.object_half_extent_m)
    if args.object_yaml:
        spec = load_object_spec(Path(args.object_yaml))
        return float(spec.tag_size_m) / 2.0
    return DEFAULT_OBJECT_HALF_EXTENT_M


def parse_joint_positions(
    names: list[str], msg_names: list[str], msg_positions: list[float]
) -> list[float] | None:
    """Map a ``JointState``-like ``(name[], position[])`` pair to an ordered list.

    Returns ``None`` if any of ``names`` is missing from ``msg_names``. Kept
    ROS-free (plain lists in, plain list out) so it is unit testable without
    constructing a real ``sensor_msgs/JointState``.
    """
    name_to_pos = dict(zip(msg_names, msg_positions))
    try:
        return [float(name_to_pos[name]) for name in names]
    except KeyError:
        return None


def parse_finger_position(msg_names: list[str], msg_positions: list[float]) -> float | None:
    name_to_pos = dict(zip(msg_names, msg_positions))
    for name in GRIPPER_FINGER_JOINTS:
        if name in name_to_pos:
            return float(name_to_pos[name])
    return None


def _build_node_class():
    """Deferred rclpy import; returns ``(TwinNode, rclpy)``."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    sensor_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )

    class TwinNode(Node):
        """Subscribes joints/gripper/object pose; never publishes robot commands."""

        def __init__(self, topics: dict[str, str]) -> None:
            super().__init__("twin_node")
            self._lock = threading.Lock()
            self._latest_arm: list[float] | None = None
            self._latest_gripper: float = FINGER_OPEN_M
            self._latest_arm_stamp: float | None = None
            self._latest_object: ObjectPose | None = None

            self.create_subscription(
                JointState, topics["joint_states"], self._on_joint_states, sensor_qos
            )
            self.create_subscription(
                JointState, topics["gripper_joint_states"], self._on_gripper_states, sensor_qos
            )
            self.create_subscription(String, topics["object_poses"], self._on_object_pose, sensor_qos)
            self.get_logger().info(
                f"twin_node: joints={topics['joint_states']} "
                f"gripper={topics['gripper_joint_states']} objects={topics['object_poses']} "
                "(observation only)"
            )

        def _on_joint_states(self, msg: "JointState") -> None:
            positions = parse_joint_positions(FRANKA_JOINT_NAMES, list(msg.name), list(msg.position))
            if positions is None:
                return
            stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            with self._lock:
                self._latest_arm = positions
                self._latest_arm_stamp = stamp_s

        def _on_gripper_states(self, msg: "JointState") -> None:
            finger_pos = parse_finger_position(list(msg.name), list(msg.position))
            if finger_pos is not None:
                with self._lock:
                    self._latest_gripper = finger_pos

        def _on_object_pose(self, msg: "String") -> None:
            try:
                pose = json_to_object_pose(msg.data)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"failed to parse object pose JSON: {exc!r}")
                return
            with self._lock:
                self._latest_object = pose

        def snapshot(
            self,
        ) -> tuple[list[float] | None, float, float | None, ObjectPose | None]:
            with self._lock:
                return (
                    self._latest_arm,
                    self._latest_gripper,
                    self._latest_arm_stamp,
                    self._latest_object,
                )

    return TwinNode, rclpy


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    topics = load_topics(args.topics_yaml)
    half_extent = resolve_half_extent(args)

    driver = TwinMujocoDriver(args.model, object_body=args.object_body)
    aggregator = TwinAggregator(object_half_extent_m=half_extent, near_contact_m=args.near_contact_m)

    try:
        TwinNode, rclpy = _build_node_class()
    except ImportError as exc:
        raise SystemExit(
            "rclpy (ROS2 Python client library) is required to run twin_node as a "
            f"live ROS2 node. Import failed: {exc!r}"
        ) from exc

    import os

    os.environ.setdefault("MUJOCO_GL", "glfw")
    import mujoco.viewer

    rclpy.init(args=argv)
    node = TwinNode(topics)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    period = 1.0 / args.rate_hz if args.rate_hz > 0 else 0.0
    print(
        f"twin_node viewer at {args.rate_hz} Hz | model={args.model} | "
        f"object_body={args.object_body} | half_extent_m={half_extent} | observation only",
        flush=True,
    )

    try:
        with mujoco.viewer.launch_passive(driver.model, driver.data) as viewer:
            while viewer.is_running():
                arm, gripper, arm_stamp, obj_pose = node.snapshot()
                if obj_pose is not None:
                    aggregator.update_objects([obj_pose])
                if arm is not None:
                    aggregator.update_arm(arm, gripper_width=gripper, stamp_s=arm_stamp or time.time())
                    state = aggregator.build(stamp_s=time.time(), tcp_pose_world=None)
                    driver.apply(state)
                    try:
                        tcp_pose = get_tcp_pose(driver.model, driver.data, name=args.tcp_name)
                    except RuntimeError:
                        tcp_pose = None
                    if tcp_pose is not None:
                        state = aggregator.build(stamp_s=state.stamp_s, tcp_pose_world=tcp_pose)
                    print(format_overlay(state), flush=True)
                viewer.sync()
                if period:
                    time.sleep(period)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
