from pathlib import Path

import pytest

from twin_ros.extrinsics import load_camera_extrinsics


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError) as ei:
        load_camera_extrinsics(tmp_path / "nope.yaml")
    assert "checklist" in str(ei.value).lower() or "Camera extrinsics" in str(ei.value)


def test_load_example_shape():
    # copy example into tmp and ensure keys work
    example = Path(__file__).resolve().parents[1] / "configs/camera_extrinsics.yaml.example"
    text = example.read_text().replace(".example", "")
    # example uses placeholder values — loader must accept them
    p = Path(__file__).resolve().parents[1] / "configs/camera_extrinsics.yaml.example"
    pose = load_camera_extrinsics(p)
    assert pose.matrix.shape == (4, 4)
