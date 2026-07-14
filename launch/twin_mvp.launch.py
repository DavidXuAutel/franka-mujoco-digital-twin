#!/usr/bin/env python3
"""MVP digital twin bringup: pose_node + twin_node.

This repo is a plain Python package (no ament/colcon), so this file is a
**documented launcher** rather than a colcon-installed ROS launch entry.

Usage (recommended on the remote ROS host)::

    cd ~/franka_mujoco_digital_twin
    source /opt/ros/humble/setup.bash   # or your existing teleop overlay
    python launch/twin_mvp.launch.py

Equivalent module invocations (same CLI flags)::

    python -m twin_ros.pose_node \\
        --object-yaml configs/objects/cube_a.yaml \\
        --extrinsics-yaml configs/camera_extrinsics.yaml \\
        --topics-yaml configs/topics.yaml

    python -m twin_ros.twin_node \\
        --model src/twin_mujoco/scene_mvp.xml \\
        --object-yaml configs/objects/cube_a.yaml \\
        --topics-yaml configs/topics.yaml

If ``launch`` / ``launch_ros`` are importable, ``generate_launch_description()``
is also provided for ``ros2 launch`` when this file is on ``PYTHONPATH``::

    ros2 launch launch/twin_mvp.launch.py

Observation only: neither node publishes Franka motion commands.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OBJECT_YAML = _REPO_ROOT / "configs/objects/cube_a.yaml"
_DEFAULT_EXTRINSICS_YAML = _REPO_ROOT / "configs/camera_extrinsics.yaml"
_DEFAULT_TOPICS_YAML = _REPO_ROOT / "configs/topics.yaml"
_DEFAULT_MODEL = _REPO_ROOT / "src/twin_mujoco/scene_mvp.xml"


def _repo_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "object_yaml": Path(args.object_yaml),
        "extrinsics_yaml": Path(args.extrinsics_yaml),
        "topics_yaml": Path(args.topics_yaml),
        "model": Path(args.model),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start pose_node and twin_node for the online digital twin MVP"
    )
    parser.add_argument(
        "--object-yaml",
        default=str(_DEFAULT_OBJECT_YAML),
        help="Object library YAML (ObjectSpec); required by pose_node and optional for twin_node half-extent",
    )
    parser.add_argument(
        "--extrinsics-yaml",
        default=str(_DEFAULT_EXTRINSICS_YAML),
        help="camera_extrinsics.yaml with T_base_camera; pose_node aborts if missing",
    )
    parser.add_argument(
        "--topics-yaml",
        default=str(_DEFAULT_TOPICS_YAML),
        help="configs/topics.yaml; built-in defaults if omitted in each node",
    )
    parser.add_argument(
        "--model",
        default=str(_DEFAULT_MODEL),
        help="MuJoCo XML model path for twin_node",
    )
    parser.add_argument(
        "--object-body",
        default="object_cube_a",
        help="Mocap body name in the MuJoCo model (twin_node --object-body)",
    )
    parser.add_argument(
        "--pose-only",
        action="store_true",
        help="Start pose_node only (AprilTag object pose publisher)",
    )
    parser.add_argument(
        "--twin-only",
        action="store_true",
        help="Start twin_node only (MuJoCo viewer; assumes object_poses already published)",
    )
    return parser


def pose_node_argv(paths: dict[str, Path], extra: list[str] | None = None) -> list[str]:
    """CLI for ``twin_ros.pose_node`` (see ``build_arg_parser`` there).

    Required:
      --object-yaml, --extrinsics-yaml
    Optional:
      --topics-yaml, --image-topic, --camera-info-topic, --object-poses-topic,
      --lose-track-timeout-s
    """
    argv = [
        sys.executable,
        "-m",
        "twin_ros.pose_node",
        "--object-yaml",
        str(paths["object_yaml"]),
        "--extrinsics-yaml",
        str(paths["extrinsics_yaml"]),
        "--topics-yaml",
        str(paths["topics_yaml"]),
    ]
    if extra:
        argv.extend(extra)
    return argv


def twin_node_argv(paths: dict[str, Path], args: argparse.Namespace, extra: list[str] | None = None) -> list[str]:
    """CLI for ``twin_ros.twin_node`` (see ``build_arg_parser`` there).

    Optional:
      --model, --object-body, --object-yaml, --object-half-extent-m,
      --topics-yaml, --near-contact-m, --tcp-name, --rate-hz
    """
    argv = [
        sys.executable,
        "-m",
        "twin_ros.twin_node",
        "--model",
        str(paths["model"]),
        "--object-body",
        args.object_body,
        "--object-yaml",
        str(paths["object_yaml"]),
        "--topics-yaml",
        str(paths["topics_yaml"]),
    ]
    if extra:
        argv.extend(extra)
    return argv


def _preflight(paths: dict[str, Path], start_pose: bool, start_twin: bool) -> None:
    if start_pose and not paths["extrinsics_yaml"].is_file():
        raise SystemExit(
            f"Missing {paths['extrinsics_yaml']}\n"
            "Copy configs/camera_extrinsics.yaml.example → configs/camera_extrinsics.yaml "
            "and fill T_base_camera before starting pose_node."
        )
    if start_pose and not paths["object_yaml"].is_file():
        raise SystemExit(f"Missing object library: {paths['object_yaml']}")
    if start_twin and not paths["model"].is_file():
        raise SystemExit(f"Missing MuJoCo model: {paths['model']}")


def run_subprocess_launcher(args: argparse.Namespace) -> int:
    paths = _repo_paths(args)
    start_pose = not args.twin_only
    start_twin = not args.pose_only
    _preflight(paths, start_pose, start_twin)

    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "glfw")
    children: list[subprocess.Popen[bytes]] = []

    def _shutdown(*_sig: object) -> None:
        for proc in children:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        for proc in children:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if start_pose:
        cmd = pose_node_argv(paths)
        print("+", " ".join(cmd), flush=True)
        children.append(subprocess.Popen(cmd, cwd=_REPO_ROOT, env=env))
    if start_twin:
        cmd = twin_node_argv(paths, args)
        print("+", " ".join(cmd), flush=True)
        children.append(subprocess.Popen(cmd, cwd=_REPO_ROOT, env=env))

    exit_code = 0
    try:
        while children:
            for proc in list(children):
                code = proc.poll()
                if code is not None:
                    children.remove(proc)
                    if code != 0:
                        exit_code = code
                        _shutdown()
                        break
            else:
                if children:
                    children[0].wait(timeout=0.2)
    except KeyboardInterrupt:
        _shutdown()
    return exit_code


def generate_launch_description():
    """Optional ROS2 launch description when ``launch`` packages are installed."""
    from launch import LaunchDescription
    from launch.actions import ExecuteProcess

    args = build_arg_parser().parse_args([])
    paths = _repo_paths(args)
    _preflight(paths, start_pose=True, start_twin=True)

    return LaunchDescription(
        [
            ExecuteProcess(
                cmd=pose_node_argv(paths),
                cwd=str(_REPO_ROOT),
                output="screen",
                shell=False,
            ),
            ExecuteProcess(
                cmd=twin_node_argv(paths, args),
                cwd=str(_REPO_ROOT),
                output="screen",
                shell=False,
            ),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_subprocess_launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
