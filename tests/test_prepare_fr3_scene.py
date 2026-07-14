from pathlib import Path

from twin_mujoco.prepare_fr3_scene import write_fr3_wrapper


def test_write_fr3_wrapper_include(tmp_path: Path):
    fr3 = tmp_path / "fake_fr3.urdf"
    fr3.write_text("<mujoco/>")
    out = write_fr3_wrapper(tmp_path / "scene_fr3_wrapper.xml", fr3)
    text = out.read_text()
    assert f'include file="{fr3}"' in text
    assert 'name="object_cube_a"' in text


def test_write_fr3_wrapper_require_exists():
    import pytest

    with pytest.raises(FileNotFoundError):
        write_fr3_wrapper("/tmp/nope_wrapper.xml", "/no/such/fr3.urdf", require_exists=True)
