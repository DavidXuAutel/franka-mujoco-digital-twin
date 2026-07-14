from __future__ import annotations

from typing import Protocol

import numpy as np

from twin_types.poses import ObjectPose


class PoseBackend(Protocol):
    """Estimates object poses from a single camera frame."""

    def estimate(
        self,
        image_bgr: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        stamp_s: float,
    ) -> list[ObjectPose]:
        ...
