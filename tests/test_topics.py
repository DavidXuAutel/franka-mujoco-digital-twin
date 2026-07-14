from pathlib import Path

import pytest

from twin_ros.topics import DEFAULT_TOPICS, load_topics


def test_load_topics_none_returns_defaults():
    topics = load_topics(None)
    assert topics == DEFAULT_TOPICS


def test_load_topics_repo_config_matches_defaults():
    path = Path(__file__).resolve().parents[1] / "configs/topics.yaml"
    topics = load_topics(path)
    assert topics == DEFAULT_TOPICS


def test_load_topics_override(tmp_path):
    custom = tmp_path / "topics.yaml"
    custom.write_text("image: /custom/image\n")
    topics = load_topics(custom)
    assert topics["image"] == "/custom/image"
    # unspecified keys keep their defaults
    assert topics["object_poses"] == DEFAULT_TOPICS["object_poses"]


def test_load_topics_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_topics(tmp_path / "nope.yaml")
