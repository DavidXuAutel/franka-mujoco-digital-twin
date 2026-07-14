from __future__ import annotations

from pathlib import Path

import yaml

from twin_types.poses import Pose3D

_CHECKLIST = (
    "Camera extrinsics missing or invalid. Checklist:\n"
    "1) Copy configs/camera_extrinsics.yaml.example → camera_extrinsics.yaml\n"
    "2) Fill T_base_camera [x,y,z,qw,qx,qy,qz] in robot base frame\n"
    "3) Confirm optical vs ROS frame convention\n"
)


def load_camera_extrinsics(path: str | Path) -> Pose3D:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(_CHECKLIST + f"Expected file: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if "T_base_camera" not in raw:
        raise ValueError(_CHECKLIST + f"Missing T_base_camera in {path}")
    t = raw["T_base_camera"]
    if len(t) != 7:
        raise ValueError(_CHECKLIST + "T_base_camera must have 7 numbers")
    return Pose3D.from_xyz_quat(*[float(v) for v in t])
