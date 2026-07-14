from __future__ import annotations

import numpy as np
from twin_types.poses import Pose3D


def compose(a: Pose3D, b: Pose3D) -> Pose3D:
    """Return a ∘ b (apply b first, then a), i.e. T_a @ T_b."""
    return Pose3D(a.matrix @ b.matrix)


def invert(pose: Pose3D) -> Pose3D:
    r = pose.matrix[:3, :3]
    t = pose.matrix[:3, 3]
    m = np.eye(4)
    m[:3, :3] = r.T
    m[:3, 3] = -r.T @ t
    return Pose3D(m)


def transform_point(pose: Pose3D, point: np.ndarray) -> np.ndarray:
    p = np.asarray(point, dtype=float).reshape(3)
    r = pose.matrix[:3, :3]
    t = pose.matrix[:3, 3]
    return r @ p + t
