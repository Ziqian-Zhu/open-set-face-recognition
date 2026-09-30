import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from face_research.evaluation.artifacts import RunJournal, verify_result_directory
from face_research.evaluation.metrics import VerificationPair, verification_report
from face_research.evaluation.pairs import score_pair_manifest
from face_research.evaluation.verification_calibration import calibrate_verification
from face_research import verification as runner


def test_calibration_is_validation_only_and_inclusive():
    pairs = [VerificationPair(.1, True), VerificationPair(.3, True),
             VerificationPair(.3, False), VerificationPair(.6, False)]
    point = calibrate_verification(pairs, split="validation", target_far=.5)
    assert point.threshold == .3
    assert point.validation["true_accept_count"] == 2
    assert point.validation["false_accept_count"] == 1
    assert calibrate_verification(pairs, split="validation", target_far=0).threshold == .1
    with pytest.raises(ValueError, match="validation"):
        calibrate_verification(pairs, split="test")
    with pytest.raises(ValueError, match="没有阈值"):
        calibrate_verification([VerificationPair(0, False), VerificationPair(.1, True)], split="validation")


@pytest.mark.parametrize("target", [-1, 1, float("nan"), float("inf")])
def test_calibration_invalid_targets(target):
    with pytest.raises(ValueError):
        calibrate_verification([VerificationPair(.1, True)], split="validation", target_far=target)


def _fixture(tmp_path, split, offset=0):
    directory = tmp_path / split
    directory.mkdir()
    for i, value in enumerate((10, 20, 100, 110)):
        assert cv2.imwrite(str(directory / f"{i}.png"), np.full((16, 16, 3), value + offset, np.uint8))
    rows = []
    for left, right, genuine in ((0, 1, True), (2, 3, True), (0, 3, False)):
        rows.append({"left_image": f"{left}.png", "right_image": f"{right}.png",
                     "left_subject_id": f"{split}_{left // 2}", "right_subject_id": f"{split}_{right // 2}",
                     "genuine": genuine})
    path = directory / "pairs.json"
    path.write_text(json.dumps({"split": split, "session_metadata": "unavailable", "pairs": rows}))
    return path


class FakeSystem:
    extractor = SimpleNamespace(distance=lambda left, right: float((1 - left @ right) / 2))
    database = SimpleNamespace(close=lambda: None, feature_signature="fixture")

    def __init__(self, *args):
        pass

    def prepare_sample(self, image):
        return SimpleNamespace(feature=np.array([1, 0] if image[0, 0, 0] < 50 else [0, 1], np.float32),
                               crop=image.copy())


def test_missing_sessions_explicit_and_never_support_low_far(tmp_path):
    path = _fixture(tmp_path, "test")
    with pytest.raises(ValueError, match="session_id"):
        score_pair_manifest(FakeSystem(), path)
    scored = score_pair_manifest(FakeSystem(), path, require_session_ids=False)
    assert scored["sessions"] == set()
    assert scored["missing_session_pairs"] == 3
    assert all(p.attempt_id is None for p in scored["pairs"])
    report = verification_report(scored["pairs"], .275)
    assert all(value["status"] == "insufficient samples" for value in report["tar_at_far"].values())
    with pytest.raises(ValueError, match="身份级泄漏"):
        score_pair_manifest(FakeSystem(), path, require_session_ids=False, forbidden_subjects={"test_0"})
    raw = json.loads(path.read_text())
    del raw["session_metadata"]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="显式声明"):
        score_pair_manifest(FakeSystem(), path, require_session_ids=False)


def test_present_sessions_still_checked_in_exploratory_mode(tmp_path):
    path = _fixture(tmp_path, "test")
    raw = json.loads(path.read_text())
    raw["pairs"][0].update(left_session_id="same", right_session_id="same")
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="不同采集批次"):
        score_pair_manifest(FakeSystem(), path, require_session_ids=False)


def test_image_mutation_is_rejected(tmp_path):
    path = _fixture(tmp_path, "test")
    class Mutating(FakeSystem):
        def prepare_sample(self, image):
            if image[0, 0, 0] == 110:
                cv2.imwrite(str(path.parent / "1.png"), np.full((16, 16, 3), 30, np.uint8))
            return super().prepare_sample(image)
    with pytest.raises(ValueError, match="图片在运行期间发生变化"):
        score_pair_manifest(Mutating(), path, require_session_ids=False)


def test_runner_freezes_before_test_and_publishes_all_artifacts(tmp_path, monkeypatch):
    validation = _fixture(tmp_path, "validation")
    test = _fixture(tmp_path, "test", offset=1)
    monkeypatch.setattr(runner, "FaceComparisonSystem", FakeSystem)
    events = []
    original = runner.score_pair_manifest
    def score(*args, **kwargs):
        if kwargs["split"] == "test":
            assert events == ["frozen", "test_opening"]
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, "score_pair_manifest", score)
    output = tmp_path / "result"
    with RunJournal(output, kind="verification") as journal:
        def freeze(receipt):
            journal.freeze(receipt)
            events.append("frozen")
        def opening():
            journal.test_opening()
            events.append("test_opening")
        report = runner.run_verification_experiment(validation, test, require_session_ids=False,
                                                    on_freeze=freeze, on_test_open=opening)
        assert report["threshold_from_validation"] == 0
        assert report["test"]["report"]["auc"] == 1
        destination = runner.write_verification_results(report, output)
        journal.completed(destination)
    assert verify_result_directory(output) == {"status": "verified", "files": 8}
    assert str(tmp_path) not in (output / "metrics.json").read_text()
    with pytest.raises(FileExistsError):
        runner.write_verification_results(report, output)
    (output / "pairs.csv").write_text("tampered")
    with pytest.raises(ValueError):
        verify_result_directory(output)


def test_runner_manifest_mutation_after_freeze_fails_before_test(tmp_path, monkeypatch):
    validation = _fixture(tmp_path, "validation")
    test = _fixture(tmp_path, "test", offset=1)
    monkeypatch.setattr(runner, "FaceComparisonSystem", FakeSystem)
    def freeze(receipt):
        test.write_text(test.read_text() + " ")
    with pytest.raises(ValueError, match="冻结后"):
        runner.run_verification_experiment(validation, test, require_session_ids=False, on_freeze=freeze)


def test_test_labels_do_not_change_calibration(tmp_path, monkeypatch):
    validation = _fixture(tmp_path, "validation")
    test = _fixture(tmp_path, "test", offset=1)
    monkeypatch.setattr(runner, "FaceComparisonSystem", FakeSystem)
    first = runner.run_verification_experiment(validation, test, require_session_ids=False)
    raw = json.loads(test.read_text())
    # Remove a genuine pair, change test class proportions but not calibration.
    raw["pairs"].pop(0)
    test.write_text(json.dumps(raw))
    second = runner.run_verification_experiment(validation, test, require_session_ids=False)
    assert first["frozen_validation"]["calibration"] == second["frozen_validation"]["calibration"]


def test_acquisition_denominators_are_not_hidden(tmp_path):
    path = _fixture(tmp_path, "test")
    class Failing(FakeSystem):
        def prepare_sample(self, image):
            if image[0, 0, 0] >= 100:
                raise ValueError("quality")
            return super().prepare_sample(image)
    scored = score_pair_manifest(Failing(), path, require_session_ids=False)
    public = runner._public_split(scored, .275)
    assert public["acquisition_failures"] == {"genuine": 1, "impostor": 1}
    assert public["all_attempts"]["tar"] == .5
    assert public["report"]["operating_point"]["tar"] == 1
    assert public["report"]["operating_point"]["far"] is None


def test_protocol_target_cannot_drift(tmp_path, monkeypatch):
    validation = _fixture(tmp_path, "validation")
    test = _fixture(tmp_path, "test", offset=1)
    raw = json.loads(validation.read_text())
    raw["protocol"] = {"target_empirical_far": .01}
    validation.write_text(json.dumps(raw))
    monkeypatch.setattr(runner, "FaceComparisonSystem", FakeSystem)
    with pytest.raises(ValueError, match="预定FAR目标"):
        runner.run_verification_experiment(validation, test, require_session_ids=False, target_far=.05)


def test_entire_test_failure_retains_denominators_and_no_roc(tmp_path):
    path = _fixture(tmp_path, "test")
    class Failing(FakeSystem):
        def prepare_sample(self, image):
            raise ValueError("no_face")
    result = runner._public_split(score_pair_manifest(Failing(), path, require_session_ids=False), .275)
    assert result["report"] is None
    assert result["all_attempts"]["genuine_count"] == 2
    assert result["all_attempts"]["impostor_count"] == 1
    assert result["acquisition_failures"] == {"genuine": 2, "impostor": 1}
