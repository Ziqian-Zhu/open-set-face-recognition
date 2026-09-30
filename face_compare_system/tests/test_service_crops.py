"""Detector-free service integration tests using synthetic face-like crops."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from face_compare.config import AppConfig, StorageConfig
from face_compare.service import FaceComparisonSystem


def _synthetic_crop(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows, columns = np.indices((128, 128))
    base = 90 + 45 * np.sin(rows / 5.0) + 30 * np.cos(columns / 7.0)
    noise = rng.normal(0, 10, size=(128, 128))
    gray = np.clip(base + noise, 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def test_face_crop_enrollment_and_recognition_need_no_detector_call(tmp_path, monkeypatch) -> None:
    config = replace(
        AppConfig(),
        storage=StorageConfig(
            database_directory=str(tmp_path / "data"),
            event_log=str(tmp_path / "events.jsonl"),
            save_face_images=False,
        ),
    )
    system = FaceComparisonSystem(config, tmp_path)
    monkeypatch.setattr(
        system.detector,
        "detect",
        lambda _frame: (_ for _ in ()).throw(AssertionError("detector must not run")),
    )
    crop = _synthetic_crop(7)
    person = system.enroll_face_crops("测试人员", [crop, _synthetic_crop(8)])
    result = system.recognize_face_crop(crop)
    assert person.sample_count == 2
    assert result.known
    assert result.name == "测试人员"
