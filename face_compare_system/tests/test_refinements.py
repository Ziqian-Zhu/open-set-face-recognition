"""Boundary regressions; synthetic features test rules, not biometric accuracy."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from face_compare.config import EngineConfig, QualityConfig, RecognitionConfig, load_config
from face_compare.database import FaceDatabase
from face_compare.deep_engine import SFaceExtractor
from face_compare.models import BoundingBox, PreparedSample, QualityReport, RecognitionResult
from face_compare.quality import FaceQualityAssessor
from face_compare.recognizer import FaceRecognizer, MultiFrameVoter
from face_compare.service import FaceComparisonSystem


def sample(angle):
    radians = np.deg2rad(angle)
    return PreparedSample(np.full((112, 112, 3), 120, np.uint8),
                          np.array([np.cos(radians), np.sin(radians)], np.float32),
                          QualityReport(True, 120, 40, 100, .2))


def service(tmp_path):
    # Exercise the real enrollment service and storage without neural inference.
    system = FaceComparisonSystem.__new__(FaceComparisonSystem)
    system.config = SimpleNamespace(engine=EngineConfig(backend="sface"))
    system.database = FaceDatabase(tmp_path, save_face_images=False)
    system.extractor = SFaceExtractor
    system.logger = SimpleNamespace(write=lambda *args, **kwargs: None)
    return system


def test_each_appended_sample_must_match_existing_identity(tmp_path):
    system = service(tmp_path)
    system.enroll_samples("A", [sample(0)])
    with pytest.raises(ValueError, match="已有样本"):
        system.enroll_samples("A", [sample(40), sample(40), sample(80)])
    assert system.database.sample_count() == 1


def test_one_duplicate_identity_cannot_hide_in_batch_median(tmp_path):
    system = service(tmp_path)
    system.enroll_samples("A", [sample(90)])
    with pytest.raises(ValueError, match="另一已录入"):
        system.enroll_samples("B", [sample(0), sample(0), sample(60)])
    assert system.database.sample_count() == 1


def test_name_normalization_is_shared_with_database(tmp_path):
    system = service(tmp_path)
    system.enroll_samples("Alice Smith", [sample(0)])
    person = system.enroll_samples(" Alice   Smith ", [sample(0)])
    assert person.name == "Alice Smith"
    assert person.sample_count == 2


@pytest.mark.parametrize("landmarks, reason", [
    ((-1, 20, 40, 20, 30, 30, 20, 40, 40, 40), "超出画面"),
    ((20, 20, 128, 20, 30, 30, 20, 40, 40, 40), "超出画面"),
    ((20, 20, 40, 20, 30, 30, 20, 40, 40, float("nan")), "无效"),
    ((20, 20), "无效"),
])
def test_offscreen_or_invalid_landmarks_rejected(landmarks, reason):
    crop = np.random.default_rng(1).integers(40, 210, (128, 128, 3), np.uint8)
    quality = FaceQualityAssessor(QualityConfig())
    report = quality.assess(crop, BoundingBox(0, 0, 128, 128, landmarks), crop.shape)
    assert not report.accepted
    assert any(reason in item for item in report.reasons)


def test_valid_landmarks_not_rejected():
    crop = np.random.default_rng(1).integers(40, 210, (128, 128, 3), np.uint8)
    report = FaceQualityAssessor(QualityConfig()).assess(
        crop, BoundingBox(0, 0, 128, 128, (20, 20, 40, 20, 30, 30, 20, 40, 40, 40)), crop.shape)
    assert report.accepted


def test_zero_calibrated_threshold_is_not_replaced_with_looser_default():
    db = SimpleNamespace(calibrated_threshold=0.)
    recognizer = FaceRecognizer(db, None, RecognitionConfig())
    assert recognizer.threshold == 0.


def test_stable_unknown_keeps_current_explanation():
    voter = MultiFrameVoter(RecognitionConfig())
    result = RecognitionResult.unknown(threshold=.275, reason="前两名距离过近，拒绝歧义匹配")
    for _ in range(4):
        stable = voter.update(result)
    assert stable is not None
    assert "歧义" in stable.reason
    assert "多帧" in stable.reason


@pytest.mark.parametrize("section,key,value", [
    ("recognition", "ambiguity_margin", float("nan")),
    ("quality", "min_blur_variance", float("inf")),
    ("camera", "width", 640.5),
    ("camera", "fps", True),
    ("camera", "preferred_index", -1),
    ("camera", "width", 0),
    ("ui", "process_every_n_frames", 0),
    ("quality", "min_contrast", -1),
    ("storage", "save_face_images", "false"),
    ("feature", "radii", [1.5]),
    ("feature", "radii", "12"),
])
def test_invalid_config_is_rejected_before_startup(tmp_path, section, key, value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({section: {key: value}}))
    with pytest.raises(ValueError):
        load_config(path)
