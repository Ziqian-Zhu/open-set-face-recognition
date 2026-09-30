"""P0 full-grid regressions use explicit synthetic vectors, never real people."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from face_research import matrix
from face_research.evaluation.artifacts import RunJournal, verify_result_directory
from face_research.evaluation.metrics import VerificationPair
from face_research.evaluation.reporting import write_matrix_results
from face_research.experiment import Probe
from face_research.selection import Template


@pytest.fixture
def synthetic_matrix(monkeypatch):
    bases = dict(zip(("A", "B", "C"), np.eye(3, dtype=np.float32)))
    gallery = [{"label": name, "sha256": f"g-{name}-{i}", "session_id": "gallery"}
               for name in bases for i in range(6)]
    def manifest_rows(split):
        return {
            "gallery": gallery,
            "probes": [
                {"label": name, "sha256": f"{split}-{name}", "session_id": split}
                for name in bases
            ] + [{"label": None, "subject_id": f"unknown-{split}",
                  "sha256": f"{split}-u", "session_id": split}],
        }
    validation, test = manifest_rows("validation"), manifest_rows("test")
    feature_by_hash = {
        row["sha256"]: bases[row["label"]] if row["label"] else -np.ones(3, np.float32) / np.sqrt(3)
        for groups in (validation, test) for row in groups["probes"]
    }
    state = {"test_reads": 0, "freeze": None, "calibrations": 0,
             "validation_pair_sessions": set(), "source_hash": "fixture"}

    def read(path):
        if path == "test":
            state["test_reads"] += 1
            assert state["freeze"] is not None
            return "test", test
        return "validation", validation

    class FakeSystem:
        def __init__(self, config, root):
            self.database = SimpleNamespace(close=lambda: None, feature_signature="SYNTHETIC-NOT-A-FACE-MODEL")

    def prepared(_system, rows, _quality, budget):
        assert budget <= 5
        grouped = {}
        for name, base in bases.items():
            grouped[name] = []
            for i in range(6):
                vector = base + (i + 1) * 0.01
                vector /= np.linalg.norm(vector)
                grouped[name].append(Template(f"g-{name}-{i}", vector, 0.95 - 0.01 * i))
        return grouped, {row["sha256"] for row in rows}

    def probes(_system, rows):
        return [Probe(row["label"], "synthetic", "recognized", feature_by_hash[row["sha256"]],
                      row["sha256"], row["sha256"] + "-aligned", 0.0) for row in rows]

    original_calibrate = matrix.calibrate_open_set
    def calibrate(*args, **kwargs):
        assert state["test_reads"] == 0, "no calibration may happen after test is opened"
        state["calibrations"] += 1
        return original_calibrate(*args, **kwargs)

    def pairs(_system, path, **kwargs):
        if path == "test-pairs":
            assert state["freeze"] is not None
        return {
            "pairs": [VerificationPair(0.01, True), VerificationPair(0.8, False, "unique-attempt")],
            "manifest_sha256": path, "image_hashes": set(), "aligned_hashes": set(),
            "sessions": state["validation_pair_sessions"] if path == "validation-pairs" else set(),
            "total_pairs": 2, "usable_pairs": 2, "acquisition_failures": {"genuine": 0, "impostor": 0},
            "note": "SYNTHETIC FIXTURE ONLY",
        }

    monkeypatch.setattr(matrix, "read_research_manifest", read)
    monkeypatch.setattr(matrix, "FaceComparisonSystem", FakeSystem)
    monkeypatch.setattr(matrix, "_prepare_gallery", prepared)
    monkeypatch.setattr(matrix, "_read_probes", probes)
    monkeypatch.setattr(matrix, "_sha256_file", lambda path: (
        hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if Path(path).name == "primary.json" else state["source_hash"]
    ))
    monkeypatch.setattr(matrix, "_source_hashes", lambda _: {"fixture.py": state["source_hash"]})
    monkeypatch.setattr(matrix, "calibrate_open_set", calibrate)
    monkeypatch.setattr(matrix, "score_pair_manifest", pairs)
    return state, test


def _run(state, *, on_freeze=None, **kwargs):
    state.update(test_reads=0, freeze=None, calibrations=0)
    def freeze(receipt):
        assert state["test_reads"] == 0
        state["freeze"] = copy.deepcopy(receipt)
        if on_freeze:
            on_freeze(receipt)
    return matrix.run_matrix_experiment("validation", "test", on_freeze=freeze, **kwargs)


def test_full_p0_grid_pair_metrics_artifacts_and_journal(synthetic_matrix, tmp_path):
    state, _ = synthetic_matrix
    with RunJournal(tmp_path / "synthetic-only") as journal:
        report = _run(state, on_freeze=journal.freeze, on_test_open=journal.test_opening,
                      verification_validation_pairs="validation-pairs", verification_test_pairs="test-pairs")
        report["evidence_kind"] = "synthetic_regression_only_not_accuracy"
        output = write_matrix_results(report, journal.output)
        journal.completed(output)
    assert len(report["variants"]) == 4 * 7 * 6 == 168
    assert state["calibrations"] == 168 * 3
    assert verify_result_directory(output)["files"] == 8
    assert json.loads((journal.directory / "frozen_validation.json").read_text()) == report["frozen_validation"]
    assert report["verification"]["report"]["auc"] == pytest.approx(1.0)
    assert report["verification"]["report"]["tar_at_far"]["0.001"]["status"] == "insufficient samples"
    for variant in report["variants"].values():
        assert variant["test"]["verification_at_frozen_threshold"]["impostor_count"] == 1
        assert all(row["FAR"] is not None for row in variant["threshold_selection"])
        fixed = variant["test"]["fixed_engineering_operating_point"]
        assert fixed["threshold"] == pytest.approx(0.275)
        assert fixed["margin"] == pytest.approx(0.04)
    assert len(report["frozen_validation"]["variants"]) == 168


def test_repeated_seed_and_changed_test_labels_do_not_change_frozen_choices(synthetic_matrix):
    state, test = synthetic_matrix
    first = _run(state, budgets=(1, 3, 5))
    repeated = _run(state, budgets=(1, 3, 5))
    assert first["frozen_validation_sha256"] == repeated["frozen_validation_sha256"]
    for key, value in first["variants"].items():
        assert value["test"]["identification"] == repeated["variants"][key]["test"]["identification"]
    test["probes"][0]["label"], test["probes"][1]["label"] = "B", "A"
    corrupted = _run(state, budgets=(1, 3, 5))
    assert first["frozen_validation"] == corrupted["frozen_validation"]
    key = "k3__first__s42__top3_median"
    assert first["variants"][key]["test"]["identification"] != corrupted["variants"][key]["test"]["identification"]


def test_validation_pairs_cannot_reuse_a_test_probe_session(synthetic_matrix):
    state, _ = synthetic_matrix
    state["validation_pair_sessions"] = {("a", "test")}
    with pytest.raises(ValueError, match="会话级泄漏"):
        _run(state, budgets=(1,), random_seeds=(42,), verification_validation_pairs="validation-pairs")


def test_changed_source_version_prevents_publication(synthetic_matrix):
    state, _ = synthetic_matrix
    def changed_after_freeze(_receipt):
        state["source_hash"] = "changed"
    with pytest.raises(ValueError, match="混合版本结果"):
        _run(state, budgets=(1,), random_seeds=(42,), on_freeze=changed_after_freeze)


def test_primary_protocol_is_bound_before_test_and_reports_fixed_pair(synthetic_matrix, tmp_path):
    state, _ = synthetic_matrix
    specification = tmp_path / "primary.json"
    source = Path(matrix.__file__).parent / "protocols" / "live_camera_mixed_v1.json"
    specification.write_bytes(source.read_bytes())
    report = _run(state, budgets=(1, 3, 5), protocol_spec=specification)
    primary = report["primary_comparison"]
    assert primary["status"] == "specified_before_test"
    assert primary["baseline_variant"] == "k3__first__s42__top3_median"
    assert primary["challenger_variant"] == "k3__coverage__s42__top3_median"
    assert primary["baseline"]["known_attempts"] == 3
    assert primary["protocol_sha256"] == report["frozen_validation"]["protocol_spec_sha256"]


def test_incompatible_fixed_protocol_fails_before_opening_test(synthetic_matrix, tmp_path):
    state, _ = synthetic_matrix
    source = Path(matrix.__file__).parent / "protocols" / "meglass_mixed_v1.json"
    with pytest.raises(ValueError, match="budgets"):
        _run(state, budgets=(1, 3), protocol_spec=source)
    assert state["test_reads"] == 0
