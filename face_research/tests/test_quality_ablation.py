import csv
from types import SimpleNamespace

import numpy as np
import pytest

from face_research import quality_ablation as qa
from face_research.experiment import Probe
from face_research.selection import Template
from face_compare_system.face_compare.config import QualityConfig
from face_compare_system.face_compare.models import QualityReport
from face_research.evaluation.artifacts import RunJournal, verify_result_directory
from face_research.evaluation.metrics import IdentificationAttempt


def quality(*, brightness=100, focus=0.4, accepted=True, reasons=()):
    return QualityReport(accepted, brightness, 30, 80, 0.1, reasons,
                         focus_method="regional", focus_score=focus,
                         focus_threshold=0.3)


def test_policy_gates_are_independent_and_none_only_bypasses_optical_quality():
    config = QualityConfig()
    dark = quality(brightness=20, focus=.5, accepted=False, reasons=("光线过暗",))
    soft = quality(brightness=100, focus=.1, accepted=False, reasons=("画面模糊",))
    assert qa.policy_accepts("none", dark, config)
    assert not qa.policy_accepts("brightness_only", dark, config)
    assert qa.policy_accepts("sharpness_only", dark, config)
    assert not qa.policy_accepts("full", dark, config)
    assert qa.policy_accepts("brightness_only", soft, config)
    assert not qa.policy_accepts("sharpness_only", soft, config)
    assert not qa.policy_accepts("full", soft, config)
    with pytest.raises(ValueError, match="未知画质策略"):
        qa.policy_accepts("invented", dark, config)


def test_raw_pool_rejects_invalid_geometry_even_under_no_filter(monkeypatch):
    image = np.zeros((16, 16, 3), np.uint8)
    monkeypatch.setattr(qa, "load_image", lambda entry: image)
    invalid = quality(accepted=False, reasons=("人脸关键点无效",))
    extractor = SimpleNamespace(align=lambda *_: pytest.fail("不得对齐无效关键点"))
    system = SimpleNamespace(
        analyze_frame=lambda *_args, **_kwargs: [SimpleNamespace(quality=invalid, box=object())],
        extractor=extractor,
    )
    entry = {"label": "A", "condition": "normal", "sha256": "sha", "path": "unused"}
    raw = qa._read_raw_probes(system, [entry])
    assert raw[0].probe.status == "invalid_geometry"
    assert qa._apply_policy(raw, "none", QualityConfig())[0].feature is None


def test_same_raw_feature_is_used_for_gate_variants():
    feature = np.array([1, 0], np.float32)
    base = Probe("A", "dark", "recognized", feature, "hash", "aligned", 1.)
    row = qa.RawProbe(base, quality(brightness=20, accepted=False, reasons=("光线过暗",)))
    assert qa._apply_policy([row], "none", QualityConfig())[0].feature is feature
    rejected = qa._apply_policy([row], "brightness_only", QualityConfig())[0]
    assert rejected.status == "quality_rejected" and rejected.feature is None
    assert rejected.aligned_sha256 == "aligned"


def test_extraction_failure_is_not_reclassified_as_quality_success(monkeypatch):
    monkeypatch.setattr(qa, "load_image", lambda entry: np.zeros((16, 16, 3), np.uint8))
    system = SimpleNamespace(
        analyze_frame=lambda *_args, **_kwargs: [
            SimpleNamespace(quality=quality(accepted=False, reasons=("画面模糊",)), box=object())
        ],
        extractor=SimpleNamespace(
            align=lambda *_: np.ones((8, 8, 3), np.uint8),
            extract=lambda *_: np.zeros(128, np.float32),
        ),
    )
    raw = qa._read_raw_probes(system, [{
        "label": "A", "condition": "blur", "sha256": "sha", "path": "unused"
    }])
    assert raw[0].probe.status == "extraction_failed"
    assert qa._apply_policy(raw, "none", QualityConfig())[0].status == "extraction_failed"


def test_paired_quality_comparison_allows_status_change_and_counts_tradeoff():
    baseline = [
        {"sha256": "a", "expected": "A", "predicted": "A", "status": "recognized"},
        {"sha256": "b", "expected": None, "predicted": "A", "status": "recognized"},
    ]
    gated = [
        {"sha256": "a", "expected": "A", "predicted": None, "status": "quality_rejected"},
        {"sha256": "b", "expected": None, "predicted": None, "status": "quality_rejected"},
    ]
    result = qa.paired_quality_comparison(baseline, gated)
    assert result["known_correct_lost"] == 1
    assert result["unknown_false_accepts_prevented"] == 1
    assert result["known_correct_rate_delta_all_attempts"] == -1
    with pytest.raises(ValueError, match="同一批"):
        qa.paired_quality_comparison(baseline, gated[::-1])


def test_writer_marks_unavailable_without_fabricating_metrics(tmp_path):
    gate = {"total": 2, "status_counts": {"quality_rejected": 2}}
    variants = {policy: {"status": "unavailable", "reason": "insufficient validation",
                         "test_identification": None, "validation_gate": gate, "test_gate": gate}
                for policy in qa.POLICIES}
    destination = tmp_path / "run"
    qa.write_quality_results({"variants": variants}, destination)
    assert "insufficient validation" in (destination / "metrics.csv").read_text()
    assert (destination / "threshold_selection.csv").exists()
    with pytest.raises(FileExistsError):
        qa.write_quality_results({"variants": variants}, destination)


def test_quality_csv_retains_corrected_accuracy_numerators_and_failures(tmp_path):
    attempts = [
        IdentificationAttempt("A", "recognized", ((0.1, "A"),)),
        IdentificationAttempt(None, "recognized", ((0.8, "A"),)),
        IdentificationAttempt(None, "no_face", ()),
    ]
    frozen = qa.calibrate_open_set(attempts, split="validation", target_fpir=0, margins=(0,))
    gate = {"total": 3, "status_counts": {"recognized": 2, "no_face": 1}}
    variant = {
        "status": "evaluated", "validation_gate": gate, "test_gate": gate,
        "test_identification": frozen.validation,
        "threshold_selection": list(frozen.sweep),
        "threshold_from_validation": frozen.threshold, "margin_from_validation": frozen.margin,
    }
    output = qa.write_quality_results({"variants": {"full": variant}}, tmp_path / "run")
    for filename in ("metrics.csv", "threshold_selection.csv"):
        with (output / filename).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        row = next(row for row in rows if float(row["threshold"]) == frozen.threshold)
        assert row["correct_all_attempts"] == "2"
        assert row["total_attempts"] == "3"
        assert row["unknown_acquisition_failures"] == "1"
        assert row["known_acquisition_failures"] == "0"
        assert float(row["accuracy_all_attempts"]) == 2 / 3


def test_all_four_gates_freeze_before_test_manifest_is_opened(monkeypatch, tmp_path):
    vectors = {
        "A": np.array([1, 0, 0], np.float32),
        "B": np.array([0, 1, 0], np.float32),
        "C": np.array([0, 0, 1], np.float32),
    }
    gallery = [{"label": name, "sha256": f"{name}{i}", "session_id": "gallery"}
               for name in vectors for i in range(4)]
    validation = {"gallery": gallery, "probes": [
        {"label": "A", "sha256": "va", "session_id": "validation"},
        {"label": None, "subject_id": "V", "sha256": "vu", "session_id": "validation"},
    ]}
    test = {"gallery": gallery, "probes": [
        {"label": "B", "sha256": "tb", "session_id": "test"},
        {"label": None, "subject_id": "T", "sha256": "tu", "session_id": "test"},
    ]}
    calibration_count = 0
    original_calibrate = qa.calibrate_open_set

    def calibration(*args, **kwargs):
        nonlocal calibration_count
        calibration_count += 1
        return original_calibrate(*args, **kwargs)

    def manifest(path):
        if path == "test":
            assert calibration_count == 4
            return "test", test
        return "validation", validation

    class FakeSystem:
        def __init__(self, config, root):
            assert not config.storage.save_face_images
            self.database = SimpleNamespace(close=lambda: None, feature_signature="fake")
            self.recognition_config = SimpleNamespace(nearest_samples=3)

    def prepared(_system, entries, _quality, _budget):
        return {
            name: [Template(f"{name}{i}", vector, .9) for i in range(4)]
            for name, vector in vectors.items()
        }, {row["sha256"] for row in entries}

    def probes(_system, entries):
        return [qa.RawProbe(Probe(
            row["label"], "normal", "recognized",
            vectors[row["label"]] if row["label"] else np.array([-1, -1, -1], np.float32) / np.sqrt(3),
            row["sha256"], row["sha256"] + "-aligned", 1.,
        ), quality()) for row in entries]

    monkeypatch.setattr(qa, "read_research_manifest", manifest)
    monkeypatch.setattr(qa, "FaceComparisonSystem", FakeSystem)
    monkeypatch.setattr(qa, "_prepare_gallery", prepared)
    monkeypatch.setattr(qa, "_read_raw_probes", probes)
    monkeypatch.setattr(qa, "_sha256_file", lambda path: "fake-hash")
    monkeypatch.setattr(qa, "calibrate_open_set", calibration)
    result = qa.run_quality_ablation("validation", "test", budget=3)
    assert all(value["status"] == "evaluated" for value in result["variants"].values())
    assert result["paired_vs_no_filter"]["full"]["known_correct_lost"] == 0
    output = qa.write_quality_results(result, tmp_path / "quality")
    assert (output / "metrics.json").exists()
    assert "fpir_unknown_all_attempts" in (output / "metrics.csv").read_text()
    assert verify_result_directory(output)["files"] == 4
    assert result["frozen_validation"]["policies"]["full"]["threshold"] == result["variants"]["full"]["threshold_from_validation"]
    full_point = result["variants"]["full"]["test_identification"]
    assert result["variants"]["none"]["test_at_shared_full_gate_validation_point"]["threshold"] == full_point["threshold"]
    calibration_count = 0
    with RunJournal(tmp_path / "quality-audited", kind="quality_ablation") as journal:
        rerun = qa.run_quality_ablation("validation", "test", budget=3,
                                       on_freeze=journal.freeze, on_test_open=journal.test_opening)
        output = qa.write_quality_results(rerun, journal.output)
        journal.completed(output)
    assert (journal.directory / "completed.json").exists()
