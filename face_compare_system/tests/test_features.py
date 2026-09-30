"""Tests for the explainable uniform-LBP descriptor."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from face_compare.config import FeatureConfig
from face_compare.features import LBPHExtractor, UNIFORM_LBP_BINS, uniform_lbp_mapping


def _texture() -> np.ndarray:
    rows, columns = np.indices((128, 128))
    gray = ((rows * 5 + columns * 3 + ((rows // 8) % 2) * 35) % 256).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_u2_mapping_has_58_unique_uniform_patterns_and_one_other_bucket() -> None:
    mapping = uniform_lbp_mapping()
    assert mapping.shape == (256,)
    assert UNIFORM_LBP_BINS == 59
    counts = np.bincount(mapping, minlength=UNIFORM_LBP_BINS)
    assert np.all(counts[:58] == 1)
    assert counts[58] == 198


def test_descriptor_dimension_and_every_cell_histogram_is_normalized() -> None:
    config = FeatureConfig(image_size=128, grid_rows=8, grid_columns=8, radii=(1, 2))
    extractor = LBPHExtractor(config)
    feature = extractor.extract(_texture())
    assert extractor.dimension == 8 * 8 * 59 * 2 == 7552
    assert feature.shape == (7552,)
    cell_histograms = feature.reshape(-1, UNIFORM_LBP_BINS)
    np.testing.assert_allclose(cell_histograms.sum(axis=1), 1.0, atol=1e-6)


def test_chi_square_distance_is_zero_for_identical_feature() -> None:
    extractor = LBPHExtractor(FeatureConfig())
    feature = extractor.extract(_texture())
    assert extractor.distance(feature, feature.copy()) == 0.0


def test_different_textures_have_positive_symmetric_distance() -> None:
    extractor = LBPHExtractor(FeatureConfig())
    first = extractor.extract(_texture())
    checker = np.indices((128, 128)).sum(axis=0) % 2
    second = extractor.extract((checker * 255).astype(np.uint8))
    forward = extractor.distance(first, second)
    backward = extractor.distance(second, first)
    assert 0.05 < forward <= 1.0
    assert forward == pytest.approx(backward)
