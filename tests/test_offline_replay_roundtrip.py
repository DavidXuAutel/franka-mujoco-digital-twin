import json
import sys
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.offline_replay import apply_all_frames, dict_to_twin_state, load_jsonl_states
from twin_mujoco.driver import TwinMujocoDriver

_SCENE = Path(__file__).resolve().parents[1] / "src/twin_mujoco/scene_mvp.xml"


def _frame(stamp_s: float, x: float) -> dict:
    return {
        "stamp_s": stamp_s,
        "arm_qpos": [0.0] * 7,
        "gripper_width": 0.04,
        "objects": [
            {
                "object_id": "cube_a",
                "xyz_quat_wxyz": [x, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0],
                "tracking_ok": True,
            }
        ],
        "grasp": None,
        "latency_s": None,
    }


def test_dict_to_twin_state_roundtrip_shape():
    raw = _frame(1.0, 0.2)
    state = dict_to_twin_state(raw)
    assert state.stamp_s == 1.0
    assert len(state.arm_qpos) == 7
    assert state.objects[0].object_id == "cube_a"
    assert state.grasp is None


def test_offline_replay_moves_object_x(tmp_path: Path):
    jsonl_path = tmp_path / "frames.jsonl"
    xs = [0.2, 0.25, 0.3]
    jsonl_path.write_text(
        "\n".join(json.dumps(_frame(float(i + 1), x)) for i, x in enumerate(xs)),
        encoding="utf-8",
    )

    states = load_jsonl_states(jsonl_path)
    assert len(states) == 3

    driver = TwinMujocoDriver(str(_SCENE), object_body="object_cube_a")
    positions = []
    for state in states:
        driver.apply(state)
        positions.append(float(driver.object_xpos()[0]))

    assert positions == pytest.approx(xs, abs=1e-6)
    assert positions[-1] > positions[0]


def test_apply_all_frames_helper(tmp_path: Path):
    jsonl_path = tmp_path / "frames.jsonl"
    jsonl_path.write_text(
        "\n".join([json.dumps(_frame(1.0, 0.2)), json.dumps(_frame(2.0, 0.35))]),
        encoding="utf-8",
    )
    driver = TwinMujocoDriver(str(_SCENE), object_body="object_cube_a")
    apply_all_frames(driver, load_jsonl_states(jsonl_path))
    assert float(driver.object_xpos()[0]) == pytest.approx(0.35, abs=1e-6)
