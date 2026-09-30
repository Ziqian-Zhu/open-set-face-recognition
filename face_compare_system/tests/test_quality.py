"""Tests for explainable image-quality rejection gates."""

from __future__ import annotations

import numpy as np

from face_compare.config import QualityConfig
from face_compare.models import BoundingBox
from face_compare.quality import FaceQualityAssessor


def test_dark_flat_crop_reports_multiple_actionable_reasons() -> None:
    assessor = FaceQualityAssessor(QualityConfig())
    crop = np.full((128, 128, 3), 20, dtype=np.uint8)
    report = assessor.assess(crop, BoundingBox(0, 0, 128, 128), crop.shape)
    assert not report.accepted
    assert "光线过暗" in report.reasons
    assert "对比度过低" in report.reasons
    assert "画面模糊" in report.reasons


def test_sharp_but_tiny_face_is_rejected_for_distance() -> None:
    assessor = FaceQualityAssessor(QualityConfig())
    checker = (np.indices((48, 48)).sum(axis=0) % 2 * 255).astype(np.uint8)
    crop = np.repeat(checker[:, :, None], 3, axis=2)
    report = assessor.assess(crop, BoundingBox(0, 0, 48, 48), (720, 1280, 3))
    assert not report.accepted
    assert "人脸距离过远" in report.reasons
    assert "画面模糊" not in report.reasons
