"""Deterministic signal regressions, NOT human-face accuracy validation."""

from dataclasses import replace
import json

import cv2
import numpy as np
import pytest

from face_compare.config import QualityConfig, load_config
from face_compare.models import BoundingBox
from face_compare.quality import FaceQualityAssessor


def detail_pattern(size=256, glasses=False):
    """Sparse face-like contours with smooth interiors (no biometric data)."""
    gray = np.tile(np.linspace(110, 190, 128).astype(np.uint8), (128, 1))
    cv2.ellipse(gray, (64, 66), (44, 60), 0, 0, 360, 100, 2)
    for x in (42, 86):
        cv2.ellipse(gray, (x, 40), (10, 4), 0, 0, 180, 45, 2)
    cv2.line(gray, (63, 55), (58, 76), 75, 2)
    cv2.line(gray, (58, 76), (70, 76), 75, 2)
    cv2.ellipse(gray, (64, 94), (20, 5), 0, 0, 180, 60, 2)
    if glasses:
        for x in (24, 70):
            cv2.rectangle(gray, (x, 28), (x+34, 49), 20, 2)
        cv2.line(gray, (58, 35), (70, 35), 20, 2)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_CUBIC)


def assess(gray, **kwargs):
    config = replace(QualityConfig(), focus_method="regional", **kwargs)
    height, width = gray.shape[:2]
    return FaceQualityAssessor(config).assess(gray, BoundingBox(0, 0, width, height), gray.shape)


def test_default_project_uses_regional_and_baseline_stays_legacy():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert load_config(root / "config.json").quality.focus_method == "regional"
    assert load_config(root / "config_lbph.json").quality.focus_method == "legacy"


def test_scale_normalization_accepts_sparse_clear_detail_without_glasses():
    reports = [assess(detail_pattern(size)) for size in (128, 256, 512, 768)]
    assert all(report.accepted for report in reports)
    # Original raw Laplacian gate rejects the large but clear sparse pattern.
    assert reports[-1].blur_variance < 55
    assert max(r.focus_score for r in reports) / min(r.focus_score for r in reports) < 1.5


def test_with_and_without_glasses_contours_both_pass():
    assert assess(detail_pattern(glasses=False)).accepted
    assert assess(detail_pattern(glasses=True)).accepted


@pytest.mark.parametrize("sigma", [5, 9])
@pytest.mark.parametrize("glasses", [False, True])
def test_defocus_is_rejected_even_with_glasses(glasses, sigma):
    blurred = cv2.GaussianBlur(detail_pattern(glasses=glasses), (0, 0), sigma)
    assert "画面模糊" in assess(blurred).reasons


@pytest.mark.parametrize("axis", [0, 1])
def test_severe_motion_blur_is_rejected(axis):
    kernel = np.zeros((31, 31), np.float32)
    if axis == 0:
        kernel[15, :] = 1/31
    else:
        kernel[:, 15] = 1/31
    report = assess(cv2.filter2D(detail_pattern(), -1, kernel))
    assert "画面模糊" in report.reasons


def test_single_strong_eye_band_cannot_rescue_blurred_lower_face():
    sharp = detail_pattern()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 9)
    blurred[:100] = sharp[:100]
    report = assess(blurred)
    assert report.focus_regions[0] >= report.focus_threshold
    assert "画面模糊" in report.reasons


def test_contrast_gain_cancels_without_bypassing_contrast_gate():
    gray = detail_pattern()
    lower_contrast = np.round(128 + (gray.astype(float)-128)*.7).astype(np.uint8)
    original = assess(gray)
    changed = assess(lower_contrast)
    assert changed.focus_score == pytest.approx(original.focus_score, rel=.08)
    assert "画面模糊" not in changed.reasons
    flat = assess(np.full((128, 128), 128, np.uint8))
    assert not flat.accepted and flat.focus_score == 0
    assert "对比度过低" in flat.reasons


def test_other_quality_gates_still_apply():
    report = assess((detail_pattern().astype(float)*.2).astype(np.uint8))
    assert not report.accepted and "光线过暗" in report.reasons
    assessor = FaceQualityAssessor(replace(QualityConfig(), focus_method="regional"))
    crop = detail_pattern(128)
    report = assessor.assess(crop, BoundingBox(0, 0, 128, 128), (1080, 1920))
    assert "人脸距离过远" in report.reasons


def test_weak_noise_on_smooth_illumination_cannot_supply_detail():
    gray = np.tile(np.linspace(60, 190, 256), (256, 1))
    gray = np.clip(gray + np.random.default_rng(17).normal(0, 3, gray.shape), 0, 255).astype(np.uint8)
    report = assess(gray)
    assert report.contrast > 22
    assert "画面模糊" in report.reasons


@pytest.mark.parametrize("noise", [3, 5, 8, 12, 20])
@pytest.mark.parametrize("size", [128, 256])
def test_noise_cannot_rescue_defocused_face_pattern(noise, size):
    gray = cv2.GaussianBlur(detail_pattern(size), (0, 0), size/32)
    noisy = np.clip(gray.astype(float) + np.random.default_rng(37).normal(0, noise, gray.shape), 0, 255).astype(np.uint8)
    report = assess(noisy)
    assert not report.accepted and "画面模糊" in report.reasons
    assert report.focus_version == "regional-v2-noise-corrected"
    assert report.noise_estimate > 0


@pytest.mark.parametrize("noise", [0, 3, 5])
def test_clear_face_pattern_with_mild_noise_still_passes(noise):
    gray = detail_pattern(128)
    noisy = np.clip(gray.astype(float) + np.random.default_rng(37).normal(0, noise, gray.shape), 0, 255).astype(np.uint8)
    assert assess(noisy).accepted


def test_invalid_landmarks_still_rejected_in_regional_mode():
    crop = detail_pattern(128)
    config = replace(QualityConfig(), focus_method="regional")
    report = FaceQualityAssessor(config).assess(
        crop, BoundingBox(0, 0, 128, 128, (-1, 20, 40, 20, 30, 30, 20, 40, 40, 40)), crop.shape)
    assert not report.accepted and any("超出画面" in reason for reason in report.reasons)


def test_enrollment_rejection_logs_only_metrics_and_explains_threshold(tmp_path):
    from types import SimpleNamespace
    from face_compare.event_log import JsonlEventLogger
    from face_compare.service import FaceComparisonSystem
    system = FaceComparisonSystem.__new__(FaceComparisonSystem)
    system.quality = FaceQualityAssessor(replace(QualityConfig(), focus_method="regional"))
    system.detector = SimpleNamespace(detect=lambda frame: [BoundingBox(0, 0, 256, 256)],
                                      crop=lambda frame, box: frame)
    system.logger = JsonlEventLogger(tmp_path / "events.jsonl")
    # No extractor/database is installed: quality rejection must precede both.
    with pytest.raises(ValueError, match="门槛"):
        system.prepare_sample(cv2.GaussianBlur(detail_pattern(), (0, 0), 9))
    event = json.loads((tmp_path / "events.jsonl").read_text())
    assert event["event"] == "enrollment_quality_rejected"
    assert set(event) == {"timestamp", "event", "quality"}
    assert len(event["quality"]["focus_regions"]) == 3


def test_diagnostics_serializable_and_not_probability():
    report = assess(detail_pattern())
    value = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert len(value["focus_regions"]) == 3
    assert "门槛" in report.focus_description
    assert "%" not in report.focus_description


@pytest.mark.parametrize("field,value", [
    ("focus_method", "auto"), ("min_focus_score", 0),
    ("min_gradient_energy", -1), ("min_focus_score", float("nan")),
])
def test_invalid_regional_configuration_rejected(tmp_path, field, value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"quality": {field: value}}))
    with pytest.raises(ValueError):
        load_config(path)
