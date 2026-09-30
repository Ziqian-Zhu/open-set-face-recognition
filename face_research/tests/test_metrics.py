import pytest

from face_research.evaluation.metrics import (
    IdentificationAttempt,
    VerificationPair,
    identification_report,
    verification_at_threshold,
    verification_report,
)


def test_verification_roc_auc_and_fixed_operating_point_are_hand_checkable():
    pairs = [
        VerificationPair(0.1, True, "g1"),
        VerificationPair(0.2, True, "g2"),
        VerificationPair(0.3, False, "i1"),
        VerificationPair(0.4, False, "i2"),
    ]
    report = verification_report(pairs, 0.2, target_fars=(0.5,))
    assert report["operating_point"]["tar"] == 1
    assert report["operating_point"]["far"] == 0
    assert report["auc"] == pytest.approx(1)
    assert report["roc"][0]["tar"] == 0
    assert report["roc"][-1]["far"] == 1
    assert report["tar_at_far"]["0.5"]["status"] == "insufficient samples"
    assert verification_at_threshold(pairs, 0.3)["false_accept_count"] == 1


def test_low_far_never_fabricated_from_tiny_or_correlated_sample():
    pairs = [VerificationPair(0.1, True, "g"), VerificationPair(0.9, False, "same")]
    result = verification_report(pairs, 0.3)
    assert result["tar_at_far"]["0.01"]["status"] == "insufficient samples"
    pairs.append(VerificationPair(0.8, False, "same"))
    result = verification_report(pairs, 0.3)
    assert "重复" in result["tar_at_far"]["0.01"]["reason"]


def test_identification_separates_top1_dir_misid_reject_and_unknown_fpir():
    attempts = [
        IdentificationAttempt("a", "recognized", ((0.1, "a"), (0.3, "b"))),
        IdentificationAttempt("b", "recognized", ((0.1, "a"), (0.12, "b"))),
        IdentificationAttempt("b", "no_face", ()),
        IdentificationAttempt(None, "recognized", ((0.1, "a"), (0.4, "b"))),
        IdentificationAttempt(None, "quality_rejected", ()),
    ]
    report = identification_report(attempts, 0.2, 0.04)
    assert report["top1_closed_set_usable_known"]["count"] == 1
    assert report["top1_closed_set_usable_known"]["total"] == 2
    assert report["dir_known_correct_all_attempts"]["count"] == 1
    assert report["known_reject_all_attempts"]["count"] == 2
    assert report["known_misidentification_all_attempts"]["count"] == 0
    assert report["fpir_unknown_usable_faces"]["count"] == 1
    assert report["fpir_unknown_usable_faces"]["total"] == 1
    assert report["fpir_unknown_all_attempts"]["total"] == 2


def test_invalid_verification_pair_or_candidate_fails():
    with pytest.raises(ValueError, match="验证对"):
        verification_report([VerificationPair(float("nan"), True)], 0.3)
    with pytest.raises(ValueError, match="候选身份距离"):
        identification_report([IdentificationAttempt("a", "recognized", ((float("inf"), "a"),))], 0.3, 0)
    with pytest.raises(ValueError, match="升序"):
        identification_report([IdentificationAttempt("a", "recognized", ((0.2, "a"), (0.1, "b")))], 0.3, 0)
    with pytest.raises(ValueError, match="不可用人脸"):
        identification_report([IdentificationAttempt("a", "no_face", ((0.1, "a"),))], 0.3, 0)


def test_non_acceptance_is_distinct_from_successful_unknown_rejection():
    rows = [
        IdentificationAttempt("a", "recognized", ((0.1, "a"),)),
        IdentificationAttempt("a", "no_face", ()),
        IdentificationAttempt(None, "recognized", ((0.8, "a"),)),
        IdentificationAttempt(None, "recognized", ((0.1, "a"),)),
        IdentificationAttempt(None, "no_face", ()),
        IdentificationAttempt(None, "quality_rejected", ()),
    ]
    report = identification_report(rows, 0.2, 0)
    assert report["unknown_rejection_all_attempts"]["rate"] == 3 / 4
    assert report["unknown_rejection_all_attempts"]["rate"] == 1 - report["fpir_unknown_all_attempts"]["rate"]
    assert report["unknown_correct_rejection_all_attempts"]["count"] == 1
    assert report["unknown_correct_rejection_all_attempts"]["rate"] == 1 / 4
    assert report["unknown_acquisition_failures_all_attempts"]["count"] == 2
    assert report["known_acquisition_failures_all_attempts"]["count"] == 1
    assert report["accuracy_all_attempts"]["count"] == 2
    assert report["accuracy_all_attempts"]["total"] == 6
    assert report["accuracy_all_attempts"]["rate"] == 2 / 6


@pytest.mark.parametrize("rows", [[], [IdentificationAttempt(None, "no_face", ())]])
def test_no_usable_attempt_never_creates_a_correct_classification(rows):
    report = identification_report(rows, 0.2, 0)
    assert report["accuracy_all_attempts"]["count"] == 0
    assert report["accuracy_all_attempts"]["total"] == len(rows)
    assert report["accuracy_all_attempts"]["rate"] == (0 if rows else None)
    assert report["unknown_correct_rejection_all_attempts"]["count"] == 0


def test_verification_ties_zero_threshold_and_missing_class_are_explicit():
    tied = verification_report([VerificationPair(0, True), VerificationPair(0, False, "i")], 0)
    assert tied["auc"] == pytest.approx(0.5)
    assert tied["operating_point"]["false_accept_count"] == 1
    assert tied["roc"][0]["false_accept_count"] == 0
    missing = verification_report([VerificationPair(0.1, True)], 0.2)
    assert missing["auc"] is None
    assert missing["operating_point"]["far"] is None
    assert missing["tar_at_far"]["0.01"]["status"] == "insufficient samples"
