import numpy as np
from twin_types.poses import Pose3D
from twin_types.transforms import compose, invert, transform_point


def test_compose_invert_roundtrip():
    a = Pose3D.from_xyz_quat(0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    b = Pose3D.from_xyz_quat(0.0, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0)
    ab = compose(a, b)
    recovered = compose(invert(a), ab)
    np.testing.assert_allclose(recovered.as_matrix(), b.as_matrix(), atol=1e-9)


def test_transform_point():
    pose = Pose3D.from_xyz_quat(1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0)
    out = transform_point(pose, np.array([0.5, 0.0, 0.0]))
    np.testing.assert_allclose(out, [1.5, 2.0, 3.0], atol=1e-9)
