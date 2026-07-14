#!/usr/bin/env python3
"""Replay TwinState JSONL frames into MuJoCo without ROS."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from twin_mujoco.driver import TwinMujocoDriver
from twin_mujoco.viewer_app import format_overlay
from twin_types.poses import GraspMetrics, ObjectPose, Pose3D, TwinState

_DEFAULT_MODEL = str(
    Path(__file__).resolve().parents[1] / "src" / "twin_mujoco" / "scene_mvp.xml"
)


def dict_to_object_pose(raw: dict[str, Any], *, default_stamp_s: float) -> ObjectPose:
    xyz_quat = raw.get("xyz_quat_wxyz")
    if xyz_quat is None or len(xyz_quat) != 7:
        raise ValueError("object xyz_quat_wxyz must have 7 numbers [x,y,z,qw,qx,qy,qz]")
    return ObjectPose(
        object_id=str(raw["object_id"]),
        pose_in_world=Pose3D.from_xyz_quat(*[float(v) for v in xyz_quat]),
        stamp_s=float(raw.get("stamp_s", default_stamp_s)),
        confidence=float(raw.get("confidence", 1.0)),
        tracking_ok=bool(raw.get("tracking_ok", True)),
    )


def dict_to_grasp_metrics(raw: dict[str, Any] | None) -> GraspMetrics | None:
    if raw is None:
        return None
    return GraspMetrics(
        distance_m=float(raw["distance_m"]),
        near_contact=bool(raw["near_contact"]),
    )


def dict_to_twin_state(raw: dict[str, Any]) -> TwinState:
    stamp_s = float(raw["stamp_s"])
    arm_qpos = [float(v) for v in raw["arm_qpos"]]
    if len(arm_qpos) != 7:
        raise ValueError("arm_qpos must have 7 joint values")
    objects = [
        dict_to_object_pose(obj, default_stamp_s=stamp_s) for obj in raw.get("objects", [])
    ]
    return TwinState(
        arm_qpos=arm_qpos,
        gripper_width=float(raw["gripper_width"]),
        objects=objects,
        grasp=dict_to_grasp_metrics(raw.get("grasp")),
        stamp_s=stamp_s,
        latency_s=None if raw.get("latency_s") is None else float(raw["latency_s"]),
    )


def iter_jsonl_states(path: str | Path) -> Iterator[TwinState]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            yield dict_to_twin_state(raw)


def load_jsonl_states(path: str | Path) -> list[TwinState]:
    return list(iter_jsonl_states(path))


def apply_all_frames(driver: TwinMujocoDriver, states: list[TwinState]) -> None:
    for state in states:
        driver.apply(state)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay TwinState JSONL into MuJoCo (observation only, no ROS)"
    )
    parser.add_argument("--jsonl", required=True, help="Path to TwinState JSONL recording")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="MuJoCo XML model path")
    parser.add_argument(
        "--object-body",
        default="object_cube_a",
        help="Mocap body name for the tracked object",
    )
    parser.add_argument("--hz", type=float, default=30.0, help="Replay rate in Hz")
    return parser


def replay_with_viewer(
    driver: TwinMujocoDriver,
    states: list[TwinState],
    *,
    hz: float,
) -> None:
    import mujoco.viewer

    period = 1.0 / hz if hz > 0 else 0.0
    idx = 0
    with mujoco.viewer.launch_passive(driver.model, driver.data) as viewer:
        while viewer.is_running():
            if idx < len(states):
                state = states[idx]
                driver.apply(state)
                print(format_overlay(state), flush=True)
                idx += 1
            viewer.sync()
            if period:
                time.sleep(period)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    states = load_jsonl_states(args.jsonl)
    if not states:
        print(f"no frames in {args.jsonl}", file=sys.stderr)
        return 1

    driver = TwinMujocoDriver(args.model, object_body=args.object_body)
    print(
        f"offline_replay {len(states)} frames at {args.hz} Hz | "
        f"model={args.model} | object_body={args.object_body}",
        flush=True,
    )
    replay_with_viewer(driver, states, hz=args.hz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
