import numpy as np
import pytest

from face_research.evaluation.calibration import calibrate_open_set
from face_research.evaluation.metrics import IdentificationAttempt, identification_report
from face_research.selection import decide


def _attempts():
    return [
        IdentificationAttempt("a", "recognized", ((0.1, "a"), (0.13, "b"))),
        IdentificationAttempt("b", "recognized", ((0.2, "b"), (0.6, "a"))),
        IdentificationAttempt(None, "recognized", ((0.15, "a"), (0.16, "b"))),
        IdentificationAttempt(None, "recognized", ((0.25, "a"), (0.5, "b"))),
    ]


def test_joint_threshold_margin_search_uses_validation_only():
    selected = calibrate_open_set(_attempts(), split="validation", target_fpir=0, margins=(0, 0.02))
    assert selected.threshold == pytest.approx(0.2)
    assert selected.margin == pytest.approx(0.02)
    assert selected.validation["dir_known_correct_all_attempts"]["count"] == 2
    assert selected.validation["fpir_unknown_usable_faces"]["count"] == 0
    assert selected.sweep
    assert all(row["FAR"] is None and row["FRR"] is None and row["TAR"] is None for row in selected.sweep)


def test_calibration_refuses_test_split_and_unusable_data():
    with pytest.raises(ValueError, match="只能用validation"):
        calibrate_open_set(_attempts(), split="test", target_fpir=0, margins=(0,))
    with pytest.raises(ValueError, match="可用未知"):
        calibrate_open_set(_attempts()[:2], split="validation", target_fpir=0, margins=(0,))


def test_acquisition_failures_are_not_successful_unknown_rejections():
    usable = [
        IdentificationAttempt("a", "recognized", ((0.1, "a"),)),
        IdentificationAttempt(None, "recognized", ((0.8, "a"),)),
    ]
    rows = usable + [
        IdentificationAttempt(None, "no_face", ()),
        IdentificationAttempt(None, "quality_rejected", ()),
    ]
    baseline = calibrate_open_set(usable, split="validation", target_fpir=0, margins=(0,))
    actual = calibrate_open_set(rows, split="validation", target_fpir=0, margins=(0,))
    assert (actual.threshold, actual.margin) == (baseline.threshold, baseline.margin)
    point = next(row for row in actual.sweep if row["threshold"] == actual.threshold)
    assert point["accuracy_all_attempts"] == 0.5
    assert point["unknown_correct_rejection"] == 1
    assert point["unknown_acquisition_failures"] == 2
    assert point["known_acquisition_failures"] == 0
    assert point["correct_all_attempts"] == 2
    assert point["total_attempts"] == 4


def test_joint_event_sweep_matches_exhaustive_grid_with_ties_and_failures():
    rng = np.random.default_rng(72)
    margins = (0, 0.02, 0.04, 0.1)
    for _ in range(25):
        rows = []
        for i in range(24):
            expected = ("a", "b", None)[i % 3]
            best = float(rng.choice([0, 0.1, 0.2, 0.4]))
            gap = float(rng.choice([0, 0.01, 0.04, 0.2]))
            winner = str(rng.choice(["a", "b"]))
            candidates = ((best, winner), (best + gap, "b" if winner == "a" else "a"))
            rows.append(IdentificationAttempt(expected, "no_face" if i % 7 == 0 else "recognized",
                                              () if i % 7 == 0 else candidates))
        target = float(rng.choice([0, 0.1, 0.3]))
        reference = []
        for margin in margins:
            for threshold in (0, 0.1, 0.2, 0.4, 1):
                report = identification_report(rows, threshold, margin)
                if report["fpir_unknown_usable_faces"]["rate"] <= target + 1e-12:
                    score = (report["dir_known_correct_all_attempts"]["count"],
                             -report["known_misidentification_all_attempts"]["count"],
                             -report["fpir_unknown_usable_faces"]["count"], -threshold, -margin)
                    reference.append((score, threshold, margin))
        if reference:
            _, threshold, margin = max(reference)
            actual = calibrate_open_set(rows, split="validation", target_fpir=target, margins=margins)
            assert (actual.threshold, actual.margin) == (threshold, margin)
            # Check every sweep row, not only the winning operating point.
            for point in actual.sweep:
                correct = sum(
                    row.status == "recognized"
                    and decide(list(row.candidates), point["threshold"], point["margin"]) == row.expected
                    for row in rows
                )
                known_failures = sum(row.expected is not None and row.status != "recognized" for row in rows)
                unknown_failures = sum(row.expected is None and row.status != "recognized" for row in rows)
                assert point["correct_all_attempts"] == correct
                assert point["total_attempts"] == len(rows)
                assert point["accuracy_all_attempts"] == correct / len(rows)
                assert point["known_acquisition_failures"] == known_failures
                assert point["unknown_acquisition_failures"] == unknown_failures
                report = identification_report(rows, point["threshold"], point["margin"])
                assert report["accuracy_all_attempts"]["count"] == correct
                assert report["unknown_correct_rejection_all_attempts"]["count"] == point["unknown_correct_rejection"]
        else:
            with pytest.raises(ValueError, match="无法满足"):
                calibrate_open_set(rows, split="validation", target_fpir=target, margins=margins)
