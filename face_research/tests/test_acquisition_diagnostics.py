import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from face_compare_system.face_compare.models import BoundingBox, QualityReport
from face_compare_system.face_compare.service import FaceComparisonSystem
from face_research.acquisition import AcquisitionInspector
from face_research import verification, verification_diagnostics as diagnostic
from face_research.evaluation.artifacts import RunJournal, verify_result_directory


class FakeSystem(FaceComparisonSystem):
    """Real service preparation logic; only inference/storage dependencies fake."""

    def __init__(self, *_):
        self.config = SimpleNamespace(engine=SimpleNamespace(backend="sface"))
        self.database = SimpleNamespace(feature_signature="fixture", close=lambda: None)
        self.logger = SimpleNamespace(write=lambda *a, **kw: None)
        box = BoundingBox(0, 0, 16, 16)
        self.detector = SimpleNamespace(
            detect=lambda frame: [] if frame[0, 0, 0] == 10 else [box, box] if frame[0, 0, 0] == 20 else [box],
            crop=lambda frame, box: frame.copy(),
        )
        self.quality = SimpleNamespace(assess=lambda crop, *_: QualityReport(
            bool(crop[0, 0, 0] != 30), 100., 30., 80., .2,
            ("画面模糊",) if crop[0, 0, 0] == 30 else ()))

        def align(frame, _box):
            if frame[0, 0, 0] == 40:
                raise ValueError("private diagnostic text must not be exported")
            return frame.copy()

        def extract(frame):
            if frame[0, 0, 0] == 50:
                raise ValueError("private extraction text")
            return np.array([1, 0] if frame[0, 0, 0] < 100 else [0, 1], np.float32)

        self.extractor = SimpleNamespace(align=align, extract=extract,
                                         distance=lambda a, b: float((1 - a @ b) / 2))


@pytest.mark.parametrize("value,status,last_stage", [
    (10, "no_face", "detection"), (20, "multiple_faces", "detection"),
    (30, "quality_rejected", "quality"), (40, "alignment_failed", "alignment"),
    (50, "extraction_failed", "extraction"), (60, "usable", "extraction"),
])
def test_inspector_uses_actual_service_once_and_preserves_values_errors(value, status, last_stage):
    system = FakeSystem()
    image = np.full((16, 16, 3), value, np.uint8)
    before = {name: dict(vars(obj)) for name, obj in
              (("detector", system.detector), ("quality", system.quality), ("extractor", system.extractor))}
    try:
        plain = system.prepare_sample(image)
    except ValueError as exc:
        plain_error = str(exc)
    with AcquisitionInspector(system) as observer:
        if status == "usable":
            traced = observer.prepare_sample(image)
            np.testing.assert_array_equal(traced.feature, plain.feature)
            np.testing.assert_array_equal(traced.crop, plain.crop)
            assert traced.quality == plain.quality
        else:
            with pytest.raises(ValueError) as exc:
                observer.prepare_sample(image)
            assert str(exc.value) == plain_error
        row, = observer.records.values()
        assert row["status"] == status
        assert row["last_stage"] == last_stage
        assert row["stage_calls"]["detection"] == 1
        assert max(row["stage_calls"].values()) == 1
        assert "private" not in json.dumps(row)
        if status in {"no_face", "multiple_faces", "quality_rejected"}:
            assert row["stage_calls"]["alignment"] == row["stage_calls"]["extraction"] == 0
    for name, expected in before.items():
        assert vars(getattr(system, name)) == expected


def test_partial_install_and_unexpected_error_restore_methods():
    system = FakeSystem()
    original_detect = system.detector.detect
    system.extractor.extract = None
    with pytest.raises(TypeError):
        with AcquisitionInspector(system):
            pass
    assert system.detector.detect is original_detect
    system = FakeSystem()
    original_align = system.extractor.align
    def crash(*_):
        raise RuntimeError("not a sample rejection")
    system.extractor.extract = crash
    with pytest.raises(RuntimeError, match="not a sample"):
        with AcquisitionInspector(system) as inspector:
            inspector.prepare_sample(np.full((16, 16, 3), 60, np.uint8))
    assert system.extractor.align is original_align
    assert system.extractor.extract is crash


def test_inspector_requires_context_and_rejects_nesting():
    inspector = AcquisitionInspector(FakeSystem())
    with pytest.raises(RuntimeError):
        inspector.prepare_sample(np.ones((16, 16, 3), np.uint8))
    with inspector:
        with pytest.raises(RuntimeError):
            inspector.__enter__()


def _manifest(folder, split, offset):
    directory = folder / split
    directory.mkdir()
    for i, value in enumerate((60, 61, 120, 121)):
        assert cv2.imwrite(str(directory / f"{i}.png"), np.full((16, 16, 3), value + offset, np.uint8))
    pairs = [{"left_image": f"{a}.png", "right_image": f"{b}.png", "genuine": genuine,
              "left_subject_id": f"private_{split}_{a//2}", "right_subject_id": f"private_{split}_{b//2}",
              "left_condition": "glasses", "right_condition": "no_glasses"}
             for a, b, genuine in ((0, 1, True), (2, 3, True), (0, 3, False))]
    path = directory / "manifest.json"
    path.write_text(json.dumps({"split": split, "session_metadata": "unavailable", "pairs": pairs}))
    return path


def _baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(verification, "FaceComparisonSystem", FakeSystem)
    monkeypatch.setattr(diagnostic, "FaceComparisonSystem", FakeSystem)
    validation = _manifest(tmp_path, "validation", 0)
    test = _manifest(tmp_path, "test", 2)
    report = verification.run_verification_experiment(validation, test, require_session_ids=False)
    folder = verification.write_verification_results(report, tmp_path / "baseline")
    return folder, validation, test


def test_full_diagnostic_join_deduplicates_and_publishes_no_paths_or_names(tmp_path, monkeypatch):
    folder, validation, test = _baseline(tmp_path, monkeypatch)
    events = []
    original = diagnostic.score_pair_manifest
    def score(*args, **kwargs):
        assert events == (["plan"] if kwargs["split"] == "validation" else ["plan", "test"])
        result = original(*args, **kwargs)
        assert len(args[0].records) == 4
        assert all(row["stage_calls"]["detection"] == 1 for row in args[0].records.values())
        return result
    monkeypatch.setattr(diagnostic, "score_pair_manifest", score)
    with RunJournal(tmp_path / "diagnostic", kind="posthoc-verification-diagnostics") as journal:
        def plan(receipt):
            events.append("plan")
            assert not receipt["calibration_performed"]
            journal.freeze(receipt)
        def opening():
            events.append("test")
            journal.test_opening()
        report = diagnostic.run_diagnostics(folder, validation, test, on_plan=plan, on_test_open=opening)
        output = diagnostic.write_diagnostics(report, journal.output)
        journal.completed(output)
    assert verify_result_directory(output) == {"status": "verified", "files": 5}
    assert report["baseline_pair_records_identical"]
    data = report["splits"]["test"]
    assert data["all_images"]["unique_images"] == 4  # 6 pair sides, 4 distinct images
    assert data["grouped_statistics"]["components"] == 1  # Impostor connects both subjects
    assert data["within_subject_appearance_usability"]["both_usable"] == 2
    for path in output.iterdir():
        text = path.read_text()
        assert str(tmp_path) not in text and "private_test" not in text and "private_validation" not in text
    with pytest.raises(FileExistsError):
        diagnostic.write_diagnostics(report, output)


@pytest.mark.parametrize("mutation", ["manifest", "score", "code", "image", "baseline"])
def test_diagnostics_rejects_drift_instead_of_claiming_improvement(tmp_path, monkeypatch, mutation):
    folder, validation, test = _baseline(tmp_path, monkeypatch)
    def change(_):
        if mutation == "manifest":
            test.write_text(test.read_text() + " ")
        elif mutation == "score":
            monkeypatch.setattr(FakeSystem, "prepare_sample", lambda self, image: (_ for _ in ()).throw(ValueError("changed")))
        elif mutation == "code":
            monkeypatch.setattr(diagnostic, "_source_hashes", lambda path: {"changed": "value"})
        elif mutation == "image":
            cv2.imwrite(str(test.parent / "0.png"), np.full((16, 16, 3), 99, np.uint8))
        else:
            (folder / "pairs.csv").write_text("changed")
    with pytest.raises(ValueError):
        diagnostic.run_diagnostics(folder, validation, test, on_plan=change)


def test_diagnostic_config_must_match_baseline_bytes(tmp_path, monkeypatch):
    folder, validation, test = _baseline(tmp_path, monkeypatch)
    config = tmp_path / "config.json"
    config.write_text((diagnostic.PROJECT / "config.json").read_text() + "\n")
    with pytest.raises(ValueError, match="配置/模型"):
        diagnostic.run_diagnostics(folder, validation, test, config_path=config)


def test_failed_pairs_preserve_both_sides_and_condition_counts(tmp_path, monkeypatch):
    folder, validation, test = _baseline(tmp_path, monkeypatch)
    # Direct join fixture verifies two different failures are one failed pair.
    scored = {"records": [{"pair_index": 0, "left_sha256": "a", "right_sha256": "b",
                           "genuine": True, "status": "acquisition_failed", "distance": None}]}
    manifest = {"pairs": [{"left_subject_id": "private", "right_subject_id": "private",
                            "left_condition": "glasses", "right_condition": "no_glasses"}]}
    details = {digest: {"image_sha256": digest, "status": status, "quality": quality}
               for digest, status, quality in (("a", "no_face", None), ("b", "quality_rejected", {
                   "accepted": False, "reasons": ["画面模糊", "光线过暗"]}))}
    result = diagnostic.summarize_diagnostics(scored, manifest, details, split="test", threshold=.3,
                                            comparator=.275, bootstrap={"iterations": 100})
    assert result["pair_failures"]["genuine"]["failure_side_counts"] == {"both": 1}
    assert result["pair_failures"]["genuine"]["failure_combination_counts"] == {"no_face+quality_rejected": 1}
    assert result["all_images"]["quality_reason_counts_overlapping"] == {"光线过暗": 1, "画面模糊": 1}
    assert result["within_subject_appearance_usability"]["neither_usable"] == 1
    with pytest.raises(ValueError, match="缺少同图"):
        diagnostic.summarize_diagnostics(scored, manifest, {}, split="test", threshold=.3,
                                         comparator=.275, bootstrap={"iterations": 100})
    manifest["pairs"][0]["left_condition"] = "undeclared condition"
    with pytest.raises(ValueError, match="未声明"):
        diagnostic.summarize_diagnostics(scored, manifest, details, split="test", threshold=.3,
                                         comparator=.275, bootstrap={"iterations": 100})
    manifest["pairs"][0]["left_condition"] = "glasses"
    manifest["pairs"].append({**manifest["pairs"][0], "left_condition": "no_glasses"})
    scored["records"].append({**scored["records"][0], "pair_index": 1})
    with pytest.raises(ValueError, match="标签冲突"):
        diagnostic.summarize_diagnostics(scored, manifest, details, split="test", threshold=.3,
                                         comparator=.275, bootstrap={"iterations": 100})
