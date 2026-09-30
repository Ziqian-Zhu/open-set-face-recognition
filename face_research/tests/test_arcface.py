from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from face_compare_system.face_compare.config import AppConfig, EngineConfig, StorageConfig
from face_compare_system.face_compare.embedding import FaceEmbedder, validate_embedder
from face_compare_system.face_compare.deep_engine import SFaceExtractor
from face_compare_system.face_compare.models import BoundingBox
from face_compare_system.face_compare.service import FaceComparisonSystem
from face_research.arcface import ArcFaceEmbedder, REFERENCE_POINTS, similarity_transform


def test_similarity_recovers_rotation_scale_translation():
    angle, scale = .2, 1.7
    forward = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = REFERENCE_POINTS @ forward.T + [34, -10]
    matrix = similarity_transform(points)
    recovered = points @ matrix[:, :2].T + matrix[:, 2]
    np.testing.assert_allclose(recovered, REFERENCE_POINTS, atol=1e-5)
    assert np.linalg.det(matrix[:, :2]) > 0


def test_similarity_matches_closed_form_procrustes_on_noisy_points():
    rng = np.random.default_rng(42)
    for _ in range(20):
        src = REFERENCE_POINTS.astype(float) + rng.normal(size=(5, 2)) * 2
        dst = REFERENCE_POINTS.astype(float)
        x, y = src - src.mean(0), dst - dst.mean(0)
        u, singular, vt = np.linalg.svd(y.T @ x / len(src))
        rotation = u @ vt
        scale = sum(singular) / np.mean(np.sum(x * x, axis=1))
        expected = np.column_stack((scale * rotation, dst.mean(0) - scale * rotation @ src.mean(0)))
        np.testing.assert_allclose(similarity_transform(src), expected, atol=1e-10)


@pytest.mark.parametrize("points", [np.zeros((5, 2)), np.ones((4, 2)), np.full((5, 2), np.nan),
                                  np.column_stack((np.arange(5), np.arange(5)))])
def test_similarity_rejects_invalid_geometry(points):
    with pytest.raises(ValueError):
        similarity_transform(points)


class Net:
    def __init__(self, output=None):
        self.output = np.arange(1, 513, dtype=np.float32) if output is None else output
    def setInput(self, blob):
        self.blob = blob
    def forward(self):
        return self.output


def _adapter(output=None):
    adapter = ArcFaceEmbedder.__new__(ArcFaceEmbedder)
    adapter._net = Net(output)
    return adapter


def test_rgb_normalization_dimension_and_unit_norm():
    adapter = _adapter()
    assert isinstance(adapter, FaceEmbedder)
    validate_embedder(adapter)
    image = np.full((112, 112, 3), [0, 127, 255], np.uint8)
    feature = adapter.extract(image)
    np.testing.assert_allclose(adapter._net.blob[0, :, 0, 0], [1, (127 - 127.5) / 127.5, -1], atol=1e-7)
    assert feature.shape == (512,) and feature.dtype == np.float32
    assert np.linalg.norm(feature) == pytest.approx(1)
    assert adapter.signature != SFaceExtractor.signature
    assert adapter.distance(feature, feature) == pytest.approx(0, abs=1e-7)


@pytest.mark.parametrize("output", [np.zeros(512), np.ones(128), np.full(512, np.inf), np.full(512, np.nan)])
def test_bad_output_is_rejected(output):
    with pytest.raises(ValueError, match="512维"):
        _adapter(output).extract(np.zeros((112, 112, 3), np.uint8))


@pytest.mark.parametrize("image", [np.zeros((112, 112, 3)), np.zeros((120, 120, 3), np.uint8), None])
def test_no_implicit_resizing_or_float_input(image):
    with pytest.raises(ValueError):
        _adapter().extract(image)


def test_alignment_uint8_and_exact_template():
    adapter = _adapter()
    image = np.random.default_rng(1).integers(0, 256, (112, 112, 3), dtype=np.uint8)
    box = BoundingBox(0, 0, 112, 112, tuple(REFERENCE_POINTS.ravel()), 1)
    np.testing.assert_array_equal(adapter.align(image, box), image)
    with pytest.raises(ValueError):
        adapter.align(image, BoundingBox(0, 0, 112, 112))


def test_license_and_file_validation_before_model_load(tmp_path):
    path = tmp_path / "model.onnx"
    with pytest.raises(ValueError, match="非商业研究"):
        ArcFaceEmbedder(path)
    with pytest.raises(ValueError, match="缺少固定版本"):
        ArcFaceEmbedder(path, noncommercial_research=True)


def test_wrong_model_hash_is_rejected_before_inference(tmp_path, monkeypatch):
    from face_research import arcface
    path = tmp_path / "model.onnx"
    path.write_bytes(b"corrupt")
    monkeypatch.setattr(arcface, "MODEL_SIZE", 7)
    with pytest.raises(ValueError, match="SHA256"):
        ArcFaceEmbedder(path, noncommercial_research=True)


def test_injected_feature_space_isolated_before_database_use(tmp_path, monkeypatch):
    from face_compare_system.face_compare import deep_engine
    monkeypatch.setattr(deep_engine, "YuNetDetector", lambda *args: SimpleNamespace())
    config = replace(AppConfig(), engine=EngineConfig(backend="sface"),
                     storage=StorageConfig("data", "events.jsonl", False, "sqlite"))
    adapter = _adapter()
    system = FaceComparisonSystem(config, tmp_path, embedder=adapter)
    assert system.database.expected_dimension == 512
    assert system.extractor is adapter
    assert system.recognition_config.default_distance_threshold == .275
    system.database.close()
    other = _adapter()
    other.signature = "different-weights-same-dimension"
    with pytest.raises(RuntimeError, match="特征|模型"):
        FaceComparisonSystem(config, tmp_path, embedder=other)
    with pytest.raises(ValueError, match="LBPH"):
        FaceComparisonSystem(AppConfig(), tmp_path / "unused", embedder=adapter)
    assert not (tmp_path / "unused").exists()


def test_default_service_still_constructs_sface(tmp_path, monkeypatch):
    from face_compare_system.face_compare import deep_engine
    fake = _adapter()
    fake.label, fake.signature, fake.dimension = "SFace", SFaceExtractor.signature, 128
    monkeypatch.setattr(deep_engine, "YuNetDetector", lambda *args: SimpleNamespace())
    monkeypatch.setattr(deep_engine, "SFaceExtractor", lambda *args: fake)
    config = replace(AppConfig(), engine=EngineConfig(backend="sface"),
                     storage=StorageConfig("data", "events.jsonl", False, "sqlite"))
    system = FaceComparisonSystem(config, tmp_path)
    assert system.extractor is fake
    assert system.engine_label == "YuNet + 五点对齐 + SFace"
    assert system.database.feature_signature == "sface-2021dec-aligned-cosine-v1"
    system.database.close()
