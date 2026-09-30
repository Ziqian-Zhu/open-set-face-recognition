import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from face_research import model_comparison as comparison
from face_research.tests.test_verification import _fixture
from face_research.evaluation.artifacts import RunJournal, verify_result_directory
from face_compare_system.face_compare.config import AppConfig, EngineConfig, StorageConfig


class Embedder:
    dimension = 2
    signature = "fixture-model"
    label = "fixture"
    def __init__(self, *args, **kwargs):
        pass
    def align(self, image, box):
        return image
    def extract(self, image):
        return np.array([1, 0] if image[0, 0, 0] < 50 else [0, 1], np.float32)
    def distance(self, a, b):
        return float((1 - a @ b) / 2)
    def metadata(self):
        return {"source": "synthetic-test-fixture"}


class System:
    def __init__(self, config, root, *, embedder):
        self.extractor = embedder
        self.database = SimpleNamespace(close=lambda: None)
    def prepare_sample(self, image):
        return SimpleNamespace(feature=self.extractor.extract(image), crop=image)


def setup(tmp_path, monkeypatch):
    validation = _fixture(tmp_path, "validation")
    test = _fixture(tmp_path, "test", offset=1)
    project = tmp_path / "project"
    models = project / "models"
    models.mkdir(parents=True)
    for name in ("face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx", "arcface.onnx"):
        (models / name).write_bytes(b"fixture")
    (project / "config.json").write_text("{}")
    config = replace(AppConfig(), engine=EngineConfig(backend="sface"), storage=StorageConfig(backend="sqlite"))
    monkeypatch.setattr(comparison, "PROJECT", project)
    monkeypatch.setattr(comparison, "load_config", lambda path: config)
    monkeypatch.setattr(comparison, "SFaceExtractor", Embedder)
    monkeypatch.setattr(comparison, "ArcFaceEmbedder", Embedder)
    monkeypatch.setattr(comparison, "FaceComparisonSystem", System)
    return validation, test, models / "arcface.onnx"


def test_all_models_calibrated_before_any_test_and_artifacts_verified(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    events = []
    original = comparison.score_pair_manifest
    def score(*args, **kwargs):
        if kwargs["split"] == "test":
            assert events[:4] == ["validation", "validation", "freeze", "test_open"]
        events.append(kwargs["split"])
        return original(*args, **kwargs)
    monkeypatch.setattr(comparison, "score_pair_manifest", score)
    output = tmp_path / "result"
    with RunJournal(output, kind="model-comparison") as journal:
        def freeze(receipt):
            assert set(receipt["calibration"]) == {"sface", "arcface"}
            journal.freeze(receipt)
            events.append("freeze")
        def opening():
            journal.test_opening()
            events.append("test_open")
        report = comparison.run_model_comparison(*args, noncommercial_research=True,
                                                 require_session_ids=False, on_freeze=freeze, on_test_open=opening)
        assert report["paired_arcface_vs_sface"]["both_usable"] == 3
        assert report["paired_arcface_vs_sface"]["genuine_accept_gained"] == 0
        assert report["variants"]["arcface"]["extraction_timing_ms"]["count"] == 8
        comparison.write_model_comparison(report, output)
        journal.completed(output)
    assert verify_result_directory(output) == {"status": "verified", "files": 10}
    assert str(tmp_path) not in (output / "metrics.json").read_text()
    with pytest.raises(FileExistsError):
        comparison.write_model_comparison(report, output)


def test_comparison_test_labels_cannot_change_either_threshold(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    first = comparison.run_model_comparison(*args, noncommercial_research=True, require_session_ids=False)
    raw = json.loads(args[1].read_text())
    raw["pairs"].pop(0)
    args[1].write_text(json.dumps(raw))
    second = comparison.run_model_comparison(*args, noncommercial_research=True, require_session_ids=False)
    assert first["frozen_validation"]["calibration"] == second["frozen_validation"]["calibration"]


def test_comparison_rejects_mutation_after_freeze(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    def freeze(receipt):
        args[2].write_bytes(b"changed model")
    with pytest.raises(ValueError, match="冻结后"):
        comparison.run_model_comparison(*args, noncommercial_research=True,
                                        require_session_ids=False, on_freeze=freeze)


def test_comparison_rejects_missing_license_acknowledgement(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="非商业"):
        comparison.run_model_comparison(*args)


def test_paired_counts_use_same_hashes_and_keep_acquisition_failures():
    def row(index, genuine, score):
        return {"left_sha256": f"left{index}", "right_sha256": f"right{index}", "genuine": genuine,
                "status": "scored" if score is not None else "acquisition_failed", "distance": score}
    a = {"records": [row(0, True, None), row(1, True, .2), row(2, False, .3), row(3, False, .6)]}
    b = {"records": [row(0, True, .2), row(1, True, None), row(2, False, .7), row(3, False, .1)]}
    counts = comparison.paired_model_counts(a, b, .4, .4)
    assert counts == {"genuine_accept_gained": 1, "genuine_accept_lost": 1,
                      "impostor_false_accept_added": 1, "impostor_false_accept_prevented": 1,
                      "both_usable": 2, "baseline_only_usable": 1, "challenger_only_usable": 1,
                      "neither_usable": 0}
    b["records"].reverse()
    with pytest.raises(ValueError, match="同一批"):
        comparison.paired_model_counts(a, b, .4, .4)
