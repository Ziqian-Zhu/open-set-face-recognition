import csv
import json

import cv2
import pytest

from face_research.evaluation.calibration import calibrate_open_set
from face_research.evaluation.metrics import IdentificationAttempt, VerificationPair, identification_report, verification_report
from face_research.evaluation.reporting import write_matrix_results
from face_research.evaluation.artifacts import verify_result_directory


def test_matrix_results_are_create_only_and_explicitly_do_not_fake_roc(tmp_path):
    attempts = [
        IdentificationAttempt("a", "recognized", ((0.1, "a"), (0.5, "b"))),
        IdentificationAttempt(None, "recognized", ((0.8, "a"), (0.9, "b"))),
        IdentificationAttempt(None, "no_face", ()),
    ]
    chosen = calibrate_open_set(attempts, split="validation", target_fpir=0, margins=(0, 0.04))
    variant = {
        "budget": 3, "selection": "first", "seed": 42, "aggregation": "top3_median",
        "threshold_from_validation": chosen.threshold,
        "margin_from_validation": chosen.margin,
        "threshold_selection": list(chosen.sweep),
        "test": {"identification": identification_report(attempts, chosen.threshold, chosen.margin)},
    }
    report = {"protocol": "synthetic-test-only", "verification": {"status": "not_evaluated"},
              "variants": {"k3__first__s42__top3_median": variant}}
    folder = tmp_path / "results" / "run-001"
    assert write_matrix_results(report, folder) == folder
    assert {path.name for path in folder.iterdir()} == {
        "experiment_config.json", "metrics.json", "metrics.csv",
        "threshold_selection.csv", "roc_curve.png", "threshold_curve.png",
        "artifacts.json",
    }
    assert verify_result_directory(folder) == {"status": "verified", "files": 6}
    assert json.loads((folder / "metrics.json").read_text())["verification"]["status"] == "not_evaluated"
    with (folder / "threshold_selection.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and rows[0]["FAR"] == "" and rows[0]["fpir_usable"] != ""
    assert cv2.imread(str(folder / "roc_curve.png")) is not None
    assert cv2.imread(str(folder / "threshold_curve.png")) is not None
    with pytest.raises(FileExistsError):
        write_matrix_results(report, folder)

    report["verification"] = {
        "status": "evaluated",
        "report": verification_report([
            VerificationPair(0.1, True, "g"), VerificationPair(0.8, False, "i")
        ], 0.2),
    }
    evaluated = tmp_path / "results" / "run-002"
    write_matrix_results(report, evaluated)
    assert cv2.imread(str(evaluated / "roc_curve.png")) is not None
    with (evaluated / "metrics.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["known_correct_count"] == "1"
    assert row["unknown_false_accept_usable_count"] == "0"
    assert row["unknown_acquisition_failures_count"] == "1"
    assert row["unknown_correct_rejection_count"] == "1"
    assert row["correct_all_attempts_count"] == "2"
    assert row["correct_all_attempts_total"] == "3"
    assert float(row["accuracy_all_attempts"]) == 2 / 3
    assert float(row["unknown_rejection_rate"]) == 1
    assert float(row["unknown_correct_rejection_rate"]) == 1 / 2
    assert all(point["unknown_acquisition_failures"] == "1" for point in rows)
    assert float(row["unknown_false_accept_usable_wilson95_high"]) > 0
    (folder / "metrics.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="校验不通过"):
        verify_result_directory(folder)
