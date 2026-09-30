import json
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import face_research.experiment as experiment
from face_research.selection import Template, rank_identities
from face_research.manifest import load_image, read_research_manifest
from face_compare_system.face_compare.models import PreparedSample, QualityReport
from face_compare_system.face_compare.vector_database import SQLiteVectorDatabase
from face_research.experiment import (
    Probe,
    calibrate_threshold,
    check_aligned_leakage,
    check_manifest_splits,
    paired_comparison,
    write_report,
)


def probe(expected, digest, aligned=None, status="recognized"):
    return Probe(
        expected,
        "normal",
        status,
        np.array([1, 0], dtype=np.float32) if status == "recognized" else None,
        digest,
        aligned,
        2.0,
    )


def test_validation_threshold_respects_unknown_target_without_test_data():
    predictions = [
        (probe("A", "ka"), [(0.10, "A")], 1.0),
        (probe("A", "kb"), [(0.21, "A")], 1.0),
        (probe(None, "ua"), [(0.20, "A")], 1.0),
        (probe(None, "ub"), [(0.30, "A")], 1.0),
    ]
    threshold, report = calibrate_threshold(predictions, 0.04, 0)
    assert 0.10 <= threshold < 0.20
    assert report["summary"]["known_correct"]["count"] == 1
    assert report["validation_unknown_false_accept"]["count"] == 0
    assert report["validation_unknown_false_accept"]["wilson95"][1] > 0


def test_validation_requires_usable_known_and_unknown():
    predictions = [(probe("A", "k"), [(0.10, "A")], 0)]
    with pytest.raises(ValueError, match="可用"):
        calibrate_threshold(predictions, 0.04, 0.05)


def test_threshold_sweep_matches_exhaustive_reference():
    rng = np.random.default_rng(13)
    for trial in range(30):
        predictions = []
        for index in range(35):
            expected = ("A", "B", None)[index % 3]
            status = "quality_rejected" if index % 11 == 0 else "recognized"
            distance = float(rng.choice([0, 0.05, 0.12, 0.20, 0.28, 0.40, 0.55]))
            name = ("A", "B")[int(rng.integers(0, 2))]
            gap = float(rng.choice([0, 0.02, 0.06]))
            candidates = [
                (distance, name),
                (distance + gap, "B" if name == "A" else "A"),
            ]
            predictions.append(
                (
                    probe(expected, f"{trial}-{index}", status=status),
                    candidates if status == "recognized" else [],
                    0.0,
                )
            )
        target = (0, 0.1, 0.3)[trial % 3]
        usable_unknown = sum(
            item.expected is None and item.status == "recognized"
            for item, _, _ in predictions
        )
        candidates = {0.0, 1.0}
        for _, ranked, _ in predictions:
            if ranked:
                candidates.add(ranked[0][0])
                candidates.add(float(np.nextafter(ranked[0][0], -np.inf)))
        reference = None
        for threshold in sorted(x for x in candidates if 0 <= x <= 1):
            records = experiment._records(predictions, threshold, 0.04)
            false_accept = sum(
                row["expected"] is None and row["predicted"] is not None
                for row in records
            )
            if false_accept / usable_unknown > target + 1e-12:
                continue
            correct = sum(
                row["expected"] is not None and row["predicted"] == row["expected"]
                for row in records
            )
            misid = sum(
                row["expected"] is not None
                and row["predicted"] is not None
                and row["predicted"] != row["expected"]
                for row in records
            )
            score = (correct, -misid, -false_accept, -threshold)
            if reference is None or score > reference[0]:
                reference = (score, threshold)
        if reference is None:
            with pytest.raises(ValueError, match="无法满足"):
                calibrate_threshold(predictions, 0.04, target)
        else:
            chosen, _ = calibrate_threshold(predictions, 0.04, target)
            assert chosen == reference[1]


def test_manifest_requires_identical_gallery_and_independent_probes():
    validation = {
        "gallery": [{"sha256": "g", "label": "A"}],
        "probes": [
            {"sha256": "v1", "label": "A"},
            {"sha256": "v2", "label": None, "subject_id": "U1"},
        ],
    }
    test = {
        "gallery": [{"sha256": "g", "label": "A"}],
        "probes": [
            {"sha256": "t1", "label": "A"},
            {"sha256": "t2", "label": None, "subject_id": "U2"},
        ],
    }
    check_manifest_splits(validation, test)
    test["probes"][0]["sha256"] = "v1"
    with pytest.raises(ValueError, match="数据泄漏"):
        check_manifest_splits(validation, test)
    test["probes"][0]["sha256"] = "t1"
    test["gallery"][0]["label"] = "B"
    with pytest.raises(ValueError, match="gallery"):
        check_manifest_splits(validation, test)
    test["gallery"][0]["label"] = "A"
    test["probes"][1]["subject_id"] = "U1"
    with pytest.raises(ValueError, match="身份级泄漏"):
        check_manifest_splits(validation, test)


def test_manifest_loads_lazily_and_rejects_changed_images(tmp_path):
    gallery_path = tmp_path / "gallery.png"
    known_path = tmp_path / "known.png"
    unknown_path = tmp_path / "unknown.png"
    for index, path in enumerate((gallery_path, known_path, unknown_path), start=1):
        assert cv2.imwrite(str(path), np.full((12, 12, 3), index * 40, np.uint8))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "split": "validation",
                "gallery": [{"image": "gallery.png", "label": "A"}],
                "probes": [
                    {"image": "known.png", "label": "A", "condition": "glasses"},
                    {"image": "unknown.png", "label": None, "subject_id": "U1"},
                ],
            }
        ),
        encoding="utf-8",
    )
    split, groups = read_research_manifest(manifest_path)
    assert split == "validation"
    assert "image" not in groups["gallery"][0]
    assert load_image(groups["gallery"][0]).shape == (12, 12, 3)
    assert cv2.imwrite(str(gallery_path), np.full((12, 12, 3), 255, np.uint8))
    with pytest.raises(ValueError, match="发生变化"):
        load_image(groups["gallery"][0])

    content = json.loads(manifest_path.read_text(encoding="utf-8"))
    del content["probes"][1]["subject_id"]
    manifest_path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="subject_id"):
        read_research_manifest(manifest_path)
    content["probes"][1]["subject_id"] = "U1"
    content["probes"][0]["image"] = "gallery.png"
    manifest_path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="重复解码图像"):
        read_research_manifest(manifest_path)


def test_aligned_crop_leakage_detected_in_all_splits():
    check_aligned_leakage(
        {"g"}, [probe("A", "v", "v-aligned")], [probe(None, "t", "t-aligned")]
    )
    with pytest.raises(ValueError, match="重复"):
        check_aligned_leakage({"g"}, [probe("A", "v", "g")], [])
    with pytest.raises(ValueError, match="重复"):
        check_aligned_leakage(
            set(), [probe("A", "v", "same")], [probe(None, "t", "same")]
        )


def test_output_is_create_only(tmp_path):
    target = tmp_path / "results.json"
    write_report({"ok": True}, target)
    with pytest.raises(FileExistsError):
        write_report({"ok": False}, target)
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}
    assert list(tmp_path.glob("*.tmp")) == []


def test_cli_reports_expected_input_errors_without_traceback(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "face_research",
            "--validation",
            str(tmp_path / "missing.json"),
            "--test",
            str(tmp_path / "also-missing.json"),
            "--output",
            str(tmp_path / "out.json"),
        ],
        cwd=experiment.PROJECT.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "实验未完成" in result.stderr
    assert "Traceback" not in result.stderr


def test_paired_test_deltas_keep_gains_and_regressions_visible():
    first = [
        {"sha256": "a", "expected": "A", "predicted": None, "status": "recognized"},
        {"sha256": "b", "expected": "B", "predicted": "B", "status": "recognized"},
        {"sha256": "u", "expected": None, "predicted": "A", "status": "recognized"},
    ]
    other = [
        {"sha256": "a", "expected": "A", "predicted": "A", "status": "recognized"},
        {"sha256": "b", "expected": "B", "predicted": None, "status": "recognized"},
        {"sha256": "u", "expected": None, "predicted": None, "status": "recognized"},
    ]
    result = paired_comparison(first, other)
    assert result["known_correct_gained"] == 1
    assert result["known_correct_lost"] == 1
    assert result["known_accuracy_delta"] == 0
    assert result["unknown_false_accepts_prevented"] == 1
    assert result["usable_unknown_fpir_delta"] == -1
    other[0]["sha256"] = "different-photo"
    with pytest.raises(ValueError, match="完全相同"):
        paired_comparison(first, other)


def test_research_rank_matches_production_sqlite_formula(tmp_path):
    first = np.eye(128, dtype=np.float32)[0]
    second = np.eye(128, dtype=np.float32)[1]
    report = QualityReport(True, 120, 40, 100, 0.1)

    def sample(vector):
        return PreparedSample(np.zeros((112, 112, 3), dtype=np.uint8), vector, report)

    database = SQLiteVectorDatabase(
        tmp_path,
        save_face_images=False,
        feature_signature="sface-test",
        expected_dimension=128,
    )
    try:
        database.add_samples("A", [sample(first), sample(first), sample(second)])
        database.add_samples("B", [sample(second)])
        research = rank_identities(
            {
                "A": [
                    Template(str(i), vector, 0.8)
                    for i, vector in enumerate((first, first, second))
                ],
                "B": [Template("b", second, 0.8)],
            },
            first,
            3,
        )
        production = database.rank_candidates(first, 3)
        assert [(distance, name) for distance, _, name in production] == research
    finally:
        database.close()


def test_end_to_end_orchestration_freezes_all_methods_before_test(monkeypatch):
    names = ("A", "B", "C")
    vectors = {
        "A": np.array([1, 0, 0], np.float32),
        "B": np.array([0, 1, 0], np.float32),
        "C": np.array([0, 0, 1], np.float32),
    }
    gallery = [
        {"label": name, "sha256": f"{name}{i}", "image": (name, i)}
        for name in names
        for i in range(4)
    ]
    validation = {
        "gallery": gallery,
        "probes": [
            {"label": "A", "sha256": "vk", "condition": "normal"},
            {"label": None, "sha256": "vu", "subject_id": "UV", "condition": "normal"},
        ],
    }
    test = {
        "gallery": gallery,
        "probes": [
            {"label": "B", "sha256": "tk", "condition": "normal"},
            {"label": None, "sha256": "tu", "subject_id": "UT", "condition": "normal"},
        ],
    }
    calibrations = []
    original_calibrate = experiment.calibrate_threshold

    def calibrated(*args, **kwargs):
        calibrations.append(True)
        return original_calibrate(*args, **kwargs)

    def manifest(path):
        if path == "test":
            assert len(calibrations) == len(experiment.METHODS)
            return "test", test
        return "validation", validation

    class FakeSystem:
        def __init__(self, config, root):
            assert str(root) != str(experiment.PROJECT)
            assert not config.storage.save_face_images
            self.database = SimpleNamespace(
                close=lambda: None, feature_signature="fake-sface"
            )
            self.recognition_config = SimpleNamespace(
                ambiguity_margin=0.04,
                nearest_samples=3,
                default_distance_threshold=0.275,
            )
            self.engine_label = "mock YuNet + SFace"

        def prepare_sample(self, item):
            name, index = item
            quality = SimpleNamespace(
                accepted=True,
                brightness=132,
                focus_score=0.7,
                blur_variance=100,
                contrast=50,
                face_size_ratio=0.1,
            )
            return SimpleNamespace(
                crop=np.array([ord(name), index], dtype=np.uint8),
                feature=vectors[name],
                quality=quality,
            )

        def enroll_samples(self, name, samples):
            assert name in names and len(samples) == 4

    def read_probes(_system, entries):
        return [
            Probe(
                row["label"],
                row["condition"],
                "recognized",
                (
                    vectors[row["label"]]
                    if row["label"]
                    else np.array([-1, -1, -1], np.float32) / np.sqrt(3)
                ),
                row["sha256"],
                row["sha256"] + "-aligned",
                1.0,
            )
            for row in entries
        ]

    monkeypatch.setattr(experiment, "read_research_manifest", manifest)
    monkeypatch.setattr(experiment, "load_image", lambda entry: entry["image"])
    monkeypatch.setattr(experiment, "FaceComparisonSystem", FakeSystem)
    monkeypatch.setattr(experiment, "_read_probes", read_probes)
    monkeypatch.setattr(experiment, "calibrate_threshold", calibrated)
    report = experiment.run_experiment("validation", "test", budget=3)
    assert set(report["methods"]) == set(experiment.METHODS)
    assert "coverage" in report["paired_vs_first"]
    assert all(
        row["test"]["summary"]["known_correct"]["count"] == 1
        for row in report["methods"].values()
    )
