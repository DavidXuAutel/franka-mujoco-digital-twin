from pathlib import Path

from twin_types.object_library import load_object_spec


def test_load_cube_a():
    path = Path(__file__).resolve().parents[1] / "configs/objects/cube_a.yaml"
    spec = load_object_spec(path)
    assert spec.object_id == "cube_a"
    assert spec.tag_id == 0
    assert abs(spec.tag_size_m - 0.04) < 1e-9
    assert spec.mujoco_body == "object_cube_a"
