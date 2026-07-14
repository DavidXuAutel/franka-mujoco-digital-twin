from __future__ import annotations

import mujoco
import numpy as np

from twin_types.poses import Pose3D


def _pose_from_xpos_xmat(xpos: np.ndarray, xmat: np.ndarray) -> Pose3D:
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(xmat, dtype=float).reshape(3, 3)
    matrix[:3, 3] = np.asarray(xpos, dtype=float).reshape(3)
    return Pose3D(matrix)


def get_tcp_pose(
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    name: str = "tcp_proxy",
) -> Pose3D:
    """Return the world-frame TCP pose after ``mujoco.mj_forward``.

    Used by ``twin_node`` to compute grasp-consistency metrics once the arm
    qpos has been applied to the driver. Sites are preferred over bodies
    because they carry no inertial/geom offset ambiguity. Lookup order:

    1. Site named ``name`` (e.g. ``tcp_site`` from ``scene_mvp.xml``).
    2. Body named ``name`` (e.g. ``tcp_proxy`` from ``scene_mvp.xml``).
    3. Site/body named ``tcp_site`` / ``tcp_proxy`` as a last-resort fallback,
       so a real FR3 model can pass its own hand site/body name and still
       fall back to the MVP stub names if not found.

    Raises:
        RuntimeError: if none of the candidate site/body names exist in the
            model.
    """
    site_candidates = [name, "tcp_site"]
    for site_name in site_candidates:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id >= 0:
            return _pose_from_xpos_xmat(data.site_xpos[site_id], data.site_xmat[site_id])

    body_candidates = [name, "tcp_proxy"]
    for body_name in body_candidates:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            return _pose_from_xpos_xmat(data.xpos[body_id], data.xmat[body_id])

    raise RuntimeError(
        f"No TCP proxy site or body found in model "
        f"(tried sites={site_candidates}, bodies={body_candidates})"
    )
