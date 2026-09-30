import math

import pytest
from face_compare.decision import accept_identity
from face_compare.operating_point import select_operating_point


def report():
    return dict(split="validation", feature_signature="sface-test", margin=.04, records=[
        dict(status="recognized", expected="A", candidate_name="A", predicted="A", distance=.10, second_best_distance=.5, ms=1),
        dict(status="recognized", expected="A", candidate_name="A", predicted=None, distance=.24, second_best_distance=.5, ms=1),
        dict(status="recognized", expected=None, candidate_name="A", predicted="A", distance=.20, second_best_distance=.5, ms=1),
        dict(status="no_face", expected=None, predicted=None, ms=1),
    ])


def test_threshold_selection_respects_usable_unknown_denominator():
    selected = select_operating_point(report(), 0)
    assert .1 <= selected["distance_threshold"] < .2
    assert selected["validation_summary"]["unknown_false_accept_usable_faces"]["total"] == 1
    assert selected["validation_summary"]["known_correct"]["rate"] == .5
    assert selected["validation_summary"]["unknown_false_accept_usable_faces"]["wilson95"][1] > .5


def test_test_split_cannot_be_used_to_tune():
    source = report()
    source["split"] = "test"
    with pytest.raises(ValueError, match="禁止"):
        select_operating_point(source)


def test_ambiguity_margin_is_preserved_during_threshold_sweep():
    source = report()
    source["records"][1]["second_best_distance"] = .25
    selected = select_operating_point(source, 1.)
    assert selected["validation_summary"]["known_correct"]["rate"] == .5


def test_missing_unknown_or_invalid_distance_fails():
    source = report()
    source["records"] = source["records"][:2]
    with pytest.raises(ValueError):
        select_operating_point(source)
    source = report()
    source["records"][0]["distance"] = float("nan")
    with pytest.raises(ValueError):
        select_operating_point(source)


def boundary_report(second, *, best=.275):
    return dict(split="validation", feature_signature="sface-test", margin=.04, records=[
        dict(status="recognized", expected="A", candidate_name="A", predicted=None,
             distance=best, second_best_distance=second, ms=1),
        dict(status="recognized", expected=None, candidate_name="A", predicted=None,
             distance=.6, second_best_distance=.7, ms=1),
    ])


@pytest.mark.parametrize("second,expected_threshold,accepted", [
    (.315, .275, True),              # Decimal equality rounds just below 0.04.
    (.315 + 2e-12, .275, True),      # Clearly above the margin.
    (.315 - 2e-12, 1e-8, False),     # Below the permitted float tolerance.
    (None, .275, True),              # One enrolled identity has no runner-up.
])
def test_cli_margin_scan_matches_shared_decision(second, expected_threshold, accepted):
    source = boundary_report(second)
    selected = select_operating_point(source, target_fpir=0)
    assert accept_identity(.275, second, .275, .04) is accepted
    assert selected["distance_threshold"] == expected_threshold
    at_boundary = next(row for row in selected["curve"] if row["distance_threshold"] == .275)
    assert at_boundary["known_correct_all_attempts"] == int(accepted)


def test_cli_best_distance_threshold_boundary_is_inclusive():
    selected = select_operating_point(boundary_report(None), target_fpir=0)
    below = math.nextafter(.275, -math.inf)
    assert not accept_identity(.275, None, below, .04)
    assert accept_identity(.275, None, .275, .04)
    curve = {row["distance_threshold"]: row for row in selected["curve"]}
    assert curve[below]["known_correct_all_attempts"] == 0
    assert curve[.275]["known_correct_all_attempts"] == 1
