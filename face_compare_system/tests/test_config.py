"""Tests for strict, typed configuration loading."""

from __future__ import annotations

import json

import pytest

from face_compare.config import load_config


def test_default_configuration_is_consistent() -> None:
    config = load_config()
    assert config.camera.preferred_index == 0
    assert config.recognition.threshold_min < config.recognition.default_distance_threshold
    assert config.recognition.default_distance_threshold < config.recognition.threshold_max


def test_json_override_and_tuple_conversion(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"feature": {"radii": [1]}, "camera": {"preferred_index": 3}}),
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.feature.radii == (1,)
    assert config.camera.preferred_index == 3
    assert config.feature.grid_rows == 8


def test_unknown_configuration_key_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"camera": {"mystery": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="未知配置项"):
        load_config(path)
