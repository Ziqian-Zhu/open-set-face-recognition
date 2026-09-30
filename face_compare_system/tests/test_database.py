"""Tests for dynamic identity expansion and threshold calibration."""

from __future__ import annotations

import numpy as np

from face_compare.database import FaceDatabase
from face_compare.models import PreparedSample, QualityReport


QUALITY = QualityReport(
    accepted=True,
    brightness=120.0,
    contrast=45.0,
    blur_variance=100.0,
    face_size_ratio=0.2,
)


def _sample(value: float) -> PreparedSample:
    return PreparedSample(
        crop=np.full((32, 32, 3), 127, dtype=np.uint8),
        feature=np.asarray([value], dtype=np.float32),
        quality=QUALITY,
    )


def test_new_people_and_same_name_samples_expand_database(tmp_path) -> None:
    database = FaceDatabase(tmp_path / "db", save_face_images=False)
    alice = database.add_samples("Alice", [_sample(0.10), _sample(0.12)])
    expanded = database.add_samples(" alice ", [_sample(0.11)])
    bob = database.add_samples("Bob", [_sample(0.80), _sample(0.82)])

    assert alice.person_id == expanded.person_id
    assert bob.person_id != alice.person_id
    assert [(person.name, person.sample_count) for person in database.list_people()] == [
        ("Alice", 3),
        ("Bob", 2),
    ]
    assert database.sample_count() == 5

    reloaded = FaceDatabase(tmp_path / "db", save_face_images=False)
    assert reloaded.sample_count() == 5
    assert len(reloaded.feature_sets()) == 2


def test_calibration_persists_threshold_and_counts_pairs(tmp_path) -> None:
    database = FaceDatabase(tmp_path / "db", save_face_images=False)
    database.add_samples("A", [_sample(0.10), _sample(0.12), _sample(0.11)])
    database.add_samples("B", [_sample(0.80), _sample(0.82), _sample(0.79)])

    def distance(first, second) -> float:
        return abs(float(first[0]) - float(second[0]))

    result = database.calibrate(distance, fallback=0.43, minimum=0.20, maximum=0.62)
    assert result.source == "库内配对自动校准"
    assert result.intra_count == 6
    assert result.inter_count == 9
    assert 0.20 <= result.threshold <= 0.62
    assert result.false_accept_rate == 0.0
    assert result.false_reject_rate == 0.0

    reloaded = FaceDatabase(tmp_path / "db", save_face_images=False)
    assert reloaded.calibrated_threshold == result.threshold
