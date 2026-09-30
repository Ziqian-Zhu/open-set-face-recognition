"""Regression checks for concrete enrollment, storage, tracking and evaluation failures."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from face_compare.config import AppConfig, EngineConfig, load_config
from face_compare.database import FaceDatabase, normalize_person_name
from face_compare.deep_engine import SFaceExtractor
from face_compare.enrollment import EnrollmentSession, TrackContinuity, validate_identity_set
from face_compare.evaluation import interval, read_manifest, summarize
from face_compare.models import BoundingBox, FaceObservation, PreparedSample, QualityReport


QUALITY = QualityReport(True, 120, 40, 100, 0.15)


def sample(seed=1, feature=(1., 0.)):
    return PreparedSample(np.random.default_rng(seed).integers(0, 255, (112, 112, 3), dtype=np.uint8),
                          np.array(feature, np.float32), QUALITY)


def test_enrollment_binds_name_and_clear_releases_it():
    session = EnrollmentSession()
    session.add("A", sample(), now=1)
    with pytest.raises(ValueError, match="属于"):
        session.add("B", sample(2), now=2)
    assert len(session.samples) == 1
    session.clear()
    session.add("B", sample(2), now=3)
    assert session.name == "B"


def test_duplicate_and_rapid_capture_do_not_increment_count():
    session = EnrollmentSession()
    first = sample()
    session.add("A", first, now=0)
    with pytest.raises(ValueError, match="太快"):
        session.add("A", sample(2), now=.1)
    with pytest.raises(ValueError, match="重复"):
        session.add("A", first, now=2)
    assert len(session.samples) == 1


def test_mixed_person_sample_rejected_and_state_unchanged():
    session = EnrollmentSession()
    session.add("A", sample(), extractor=SFaceExtractor, now=1)
    with pytest.raises(ValueError, match="混入"):
        session.add("A", sample(2, (0, 1)), extractor=SFaceExtractor, now=2)
    assert len(session.samples) == 1


@pytest.mark.parametrize("value", ["A\x00B", "A\nB", "\tAlice", "A\u200bB"])
def test_control_characters_cannot_enter_identity_names(value):
    with pytest.raises(ValueError):
        normalize_person_name(value)


def test_partial_write_failure_never_exposes_half_enrollment(tmp_path, monkeypatch):
    db = FaceDatabase(tmp_path, save_face_images=False)
    original = np.save
    calls = []

    def failing_save(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("disk full")
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "save", failing_save)
    with pytest.raises(OSError):
        db.add_samples("A", [sample(), sample(2)])
    assert db.sample_count() == 0
    assert not db.feature_sets()
    assert FaceDatabase(tmp_path, save_face_images=False).sample_count() == 0
    monkeypatch.setattr(np, "save", original)
    assert db.add_samples("A", [sample()]).sample_count == 1


def test_stale_process_cannot_restore_archived_people(tmp_path):
    db = FaceDatabase(tmp_path, save_face_images=False)
    person = db.add_samples("A", [sample()])
    stale = FaceDatabase(tmp_path, save_face_images=False)
    archive = db.archive_people(person.person_id)
    assert json.loads(archive.read_text())["people"]
    assert db.sample_count() == 0
    with pytest.raises(RuntimeError, match="另一个程序"):
        stale.add_samples("B", [sample(2)])
    assert FaceDatabase(tmp_path, save_face_images=False).sample_count() == 0


@pytest.mark.parametrize("feature", [np.array([np.nan]), np.zeros((1, 2)), np.array([])])
def test_invalid_features_never_persist(tmp_path, feature):
    db = FaceDatabase(tmp_path, save_face_images=False)
    with pytest.raises(ValueError):
        db.add_samples("A", [replace(sample(), feature=feature)])
    assert db.sample_count() == 0


def test_incompatible_model_database_fails_explicitly(tmp_path):
    db = FaceDatabase(tmp_path, save_face_images=False, feature_signature="old")
    db.add_samples("A", [sample()])
    with pytest.raises(RuntimeError, match="不一致"):
        FaceDatabase(tmp_path, feature_signature="new")


def test_missing_feature_is_reported_instead_of_silent_person_loss(tmp_path):
    db = FaceDatabase(tmp_path, save_face_images=False)
    db.add_samples("A", [sample()])
    next(tmp_path.rglob("*.npy")).unlink()
    with pytest.raises(RuntimeError, match="缺少特征"):
        FaceDatabase(tmp_path, save_face_images=False)


def test_continuity_resets_on_same_position_new_identity_or_gap():
    gate = TrackContinuity(SFaceExtractor)
    a = FaceObservation(BoundingBox(10, 10, 100, 100), QUALITY, None, feature=np.array([1., 0.]))
    b = replace(a, feature=np.array([0., 1.]))
    assert not gate.update(a, 0)
    assert gate.update(a, .1)
    assert not gate.update(b, .2)
    assert not gate.update(b, 3)


def test_manifest_blocks_same_image_across_enrollment_and_test(tmp_path):
    cv2.imwrite(str(tmp_path/"one.png"), sample().crop)
    cv2.imwrite(str(tmp_path/"copy.png"), sample().crop)
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps({"split": "test", "gallery": [{"image": "one.png", "label": "A"}],
                                "probes": [{"image": "copy.png", "label": "A"}]}))
    with pytest.raises(ValueError, match="泄漏"):
        read_manifest(path)


def test_no_face_is_not_counted_as_successful_unknown_rejection():
    result = summarize([
        {"expected": None, "predicted": None, "status": "no_face", "ms": 1},
        {"expected": None, "predicted": "A", "status": "recognized", "ms": 2},
        {"expected": "A", "predicted": "B", "status": "recognized", "ms": 3},
    ])
    assert result["unknown_correct_rejection_all_attempts"]["rate"] == 0
    assert result["unknown_false_accept_usable_faces"]["rate"] == 1
    assert result["known_misidentified"]["rate"] == 1
    assert result["known_failed_identification"]["rate"] == 1
    assert interval(0, 0)["rate"] is None
    assert interval(0, 10)["wilson95"][1] > .20


def test_neural_models_load_and_blank_frame_has_no_face(tmp_path):
    from face_compare.config import StorageConfig
    from face_compare.service import FaceComparisonSystem
    models = Path(__file__).resolve().parents[1]/"models"
    if not (models/"face_recognition_sface_2021dec.onnx").exists():
        pytest.skip("官方模型尚未下载；运行 scripts/download_models.py 后重测")
    config = replace(AppConfig(), engine=EngineConfig(backend="sface", model_directory=str(models)),
                     storage=StorageConfig(str(tmp_path/"db"), str(tmp_path/"events.jsonl"), False))
    system = FaceComparisonSystem(config, tmp_path)
    assert system.analyze_frame(np.zeros((480, 640, 3), np.uint8), log_events=False) == []
    feature = system.extractor.extract(sample().crop)
    assert feature.shape == (128,)
    assert np.linalg.norm(feature) == pytest.approx(1, abs=1e-5)
    assert system.recognizer.threshold == pytest.approx(.275)
    assert system.calibrate_threshold().false_accept_rate is None
    with pytest.raises(ValueError, match="五点对齐"):
        system.prepare_face_crop(sample().crop)


def test_evaluation_uses_temporary_gallery_and_keeps_acquisition_failures_separate(tmp_path, monkeypatch):
    from face_compare.detector import FaceDetector
    from face_compare.evaluation import evaluate
    # Synthetic images and deterministic boxes test orchestration, not face accuracy.
    monkeypatch.setattr(FaceDetector, "detect", lambda self, frame: [BoundingBox(0, 0, 112, 112)])
    gallery, probe = sample(5).crop, sample(6).crop
    cv2.imwrite(str(tmp_path/"gallery.png"), gallery)
    cv2.imwrite(str(tmp_path/"probe.png"), probe)
    cv2.imwrite(str(tmp_path/"dark.png"), np.zeros((112, 112, 3), np.uint8))
    manifest = tmp_path/"test.json"
    manifest.write_text(json.dumps({"split": "test", "gallery": [{"image": "gallery.png", "label": "A"}],
                                   "probes": [{"image": "probe.png", "label": "A"},
                                              {"image": "dark.png", "label": None}]}))
    production = tmp_path/"production"
    production.mkdir()
    marker = production/"keep.txt"
    marker.write_text("unchanged")
    report = evaluate(AppConfig(), production, manifest)
    assert report["summary"]["total"] == 2
    assert report["summary"]["status_counts"]["quality_rejected"] == 1
    assert report["summary"]["unknown_correct_rejection_all_attempts"]["rate"] == 0
    assert report["summary"]["unknown_false_accept_usable_faces"]["rate"] is None
    assert list(production.iterdir()) == [marker]
