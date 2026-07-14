from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from twin_types.poses import Pose3D


@dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    mesh: str
    mujoco_body: str
    tag_family: str
    tag_id: int
    tag_size_m: float
    T_object_tag: Pose3D


def load_object_spec(path: str | Path) -> ObjectSpec:
    raw = yaml.safe_load(Path(path).read_text())
    required = [
        "object_id",
        "mesh",
        "mujoco_body",
        "tag_family",
        "tag_id",
        "tag_size_m",
        "T_object_tag",
    ]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    t = raw["T_object_tag"]
    if len(t) != 7:
        raise ValueError(f"{path}: T_object_tag must be [x,y,z,qw,qx,qy,qz]")
    return ObjectSpec(
        object_id=str(raw["object_id"]),
        mesh=str(raw["mesh"]),
        mujoco_body=str(raw["mujoco_body"]),
        tag_family=str(raw["tag_family"]),
        tag_id=int(raw["tag_id"]),
        tag_size_m=float(raw["tag_size_m"]),
        T_object_tag=Pose3D.from_xyz_quat(*[float(v) for v in t]),
    )
