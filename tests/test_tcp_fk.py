from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from twin_ros.tcp_fk import get_tcp_pose

_SCENE = Path(__file__).resolve().parents[1] / "src/twin_mujoco/scene_mvp.xml"


def test_get_tcp_pose_from_site():
    model = mujoco.MjModel.from_xml_path(str(_SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pose = get_tcp_pose(model, data, name="tcp_site")
    x, y, z, qw, qx, qy, qz = pose.xyz_quat_wxyz()
    assert abs(x - 0.0) < 1e-6
    assert abs(y - 0.0) < 1e-6
    assert abs(z - 0.3) < 1e-6
    assert abs(qw - 1.0) < 1e-6


def test_get_tcp_pose_falls_back_to_body_name():
    model = mujoco.MjModel.from_xml_path(str(_SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # "hand" doesn't exist as a site or body -> falls back to tcp_site/tcp_proxy.
    pose = get_tcp_pose(model, data, name="hand")
    _x, _y, z, *_ = pose.xyz_quat_wxyz()
    assert abs(z - 0.3) < 1e-6


def test_get_tcp_pose_missing_raises_on_model_without_tcp():
    xml = """
    <mujoco model="no_tcp">
      <worldbody>
        <geom name="floor" type="plane" size="1 1 0.05"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with pytest.raises(RuntimeError):
        get_tcp_pose(model, data)
