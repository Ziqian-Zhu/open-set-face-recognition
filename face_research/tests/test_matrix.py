from types import SimpleNamespace

import numpy as np

from face_research import matrix
from face_research.experiment import Probe
from face_research.selection import Template


def test_matrix_freezes_every_variant_before_test_is_opened(monkeypatch):
    vectors = {
        "A": np.array([1, 0, 0], np.float32),
        "B": np.array([0, 1, 0], np.float32),
        "C": np.array([0, 0, 1], np.float32),
    }
    gallery = [
        {"label": name, "sha256": f"{name}{i}", "session_id": "gallery"}
        for name in vectors for i in range(4)
    ]
    validation = {
        "gallery": gallery,
        "probes": [
            {"label": "A", "sha256": "va", "session_id": "validation"},
            {"label": None, "subject_id": "V", "sha256": "vu", "session_id": "validation"},
        ],
    }
    test = {
        "gallery": gallery,
        "probes": [
            {"label": "B", "sha256": "tb", "session_id": "test"},
            {"label": None, "subject_id": "T", "sha256": "tu", "session_id": "test"},
        ],
    }
    calibrations = []
    original_calibrate = matrix.calibrate_open_set

    def counted_calibration(*args, **kwargs):
        calibrations.append(True)
        return original_calibrate(*args, **kwargs)

    def manifest(path):
        if path == "test":
            assert len(calibrations) == 2 * len(matrix.METHODS) * len(matrix.AGGREGATIONS) * 3
            return "test", test
        return "validation", validation

    class FakeSystem:
        def __init__(self, config, root):
            assert not config.storage.save_face_images
            self.database = SimpleNamespace(close=lambda: None, feature_signature="fake")

    def prepared(_system, entries, _quality, _budget):
        grouped = {
            name: [Template(f"{name}{i}", vector, 0.9 - 0.01 * i) for i in range(4)]
            for name, vector in vectors.items()
        }
        return grouped, {row["sha256"] for row in entries}

    def probes(_system, entries):
        return [
            Probe(
                row["label"], "normal", "recognized",
                vectors[row["label"]] if row["label"] else np.array([-1, -1, -1], np.float32) / np.sqrt(3),
                row["sha256"], row["sha256"] + "-aligned", 1.0,
            )
            for row in entries
        ]

    monkeypatch.setattr(matrix, "read_research_manifest", manifest)
    monkeypatch.setattr(matrix, "FaceComparisonSystem", FakeSystem)
    monkeypatch.setattr(matrix, "_prepare_gallery", prepared)
    monkeypatch.setattr(matrix, "_read_probes", probes)
    monkeypatch.setattr(matrix, "_sha256_file", lambda path: "fake-hash")
    monkeypatch.setattr(matrix, "calibrate_open_set", counted_calibration)
    report = matrix.run_matrix_experiment(
        "validation", "test", budgets=(1, 2), random_seeds=(42,), margins=(0, 0.04)
    )
    assert len(report["variants"]) == 2 * len(matrix.METHODS) * len(matrix.AGGREGATIONS)
    assert report["verification"]["status"] == "not_evaluated"
    assert all(item["test"]["identification"]["known_attempts"] == 1 for item in report["variants"].values())
    assert all("paired_fixed_margin_vs_threshold_only" in item["margin_ablation"]
               for item in report["variants"].values())
