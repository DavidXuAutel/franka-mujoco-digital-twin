import numpy as np
from twin_types.grasp import compute_grasp_metrics
from twin_types.poses import Pose3D


def test_distance_and_near_contact():
    tcp = Pose3D.from_xyz_quat(0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0)
    obj = Pose3D.from_xyz_quat(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    m = compute_grasp_metrics(tcp, obj, object_half_extent_m=0.05, near_contact_m=0.02)
    assert abs(m.distance_m - 0.05) < 1e-9
    assert m.near_contact is False

    m2 = compute_grasp_metrics(
        Pose3D.from_xyz_quat(0.0, 0.0, 0.055, 1, 0, 0, 0),
        obj,
        object_half_extent_m=0.05,
        near_contact_m=0.02,
    )
    assert m2.near_contact is True
