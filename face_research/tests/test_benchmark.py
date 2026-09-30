from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from face_research import benchmark
from face_research.benchmark import _summary, run_benchmark, write_benchmark_results
from face_research.evaluation.artifacts import verify_result_directory


def test_latency_summary_keeps_stages_and_processing_fps_distinct():
    rows = [
        {"detected_faces": 1, "usable_faces": 1, "detection_ms": 1.0,
         "quality_ms": 1.0, "alignment_ms": 1.0, "embedding_ms": 1.0,
         "matching_ms": 1.0, "end_to_end_ms": 10.0},
        {"detected_faces": 2, "usable_faces": 1, "detection_ms": 2.0,
         "quality_ms": 2.0, "alignment_ms": 2.0, "embedding_ms": 2.0,
         "matching_ms": 2.0, "end_to_end_ms": 20.0},
    ]
    result = _summary(rows)
    assert result["end_to_end_ms"]["p50"] == pytest.approx(15.0)
    assert result["processing_fps_from_p50"] == pytest.approx(1000 / 15)
    assert result["detected_faces_per_frame"] == [1, 2]
    assert result["usable_faces_per_frame"] == [1]


def test_benchmark_rejects_invalid_gallery_grid_before_reading_images():
    with pytest.raises(ValueError, match="gallery_sizes"):
        run_benchmark("no-file", "no-file", gallery_sizes=(0,))


def test_thread_configuration_is_restored_after_failure(monkeypatch):
    current = [8]
    monkeypatch.setattr(cv2, "getNumThreads", lambda: current[0])
    monkeypatch.setattr(cv2, "setNumThreads", lambda value: current.__setitem__(0, value))
    with pytest.raises(RuntimeError):
        with benchmark._thread_limit(1):
            assert current[0] == 1
            raise RuntimeError("simulated")
    assert current[0] == 8


def test_benchmark_saves_raw_samples_and_labels_repeated_images_as_synthetic(tmp_path, monkeypatch):
    class FakeDatabase:
        def __init__(self):
            self.people = 0
            self.closed = False
        def add_samples(self, name, samples):
            assert name.startswith("synthetic_")
            assert len(samples) == 3
            assert all(np.linalg.norm(item.feature) == pytest.approx(1) for item in samples)
            self.people += 1
        def close(self):
            self.closed = True
        def refresh_cache(self):
            pass
        def rank_candidates(self, feature, count):
            return []
    database = FakeDatabase()
    system = SimpleNamespace(
        database=database, recognition_config=SimpleNamespace(nearest_samples=3),
        detector=SimpleNamespace(detect=lambda frame: [object()] * (1 if frame.shape[1] == 16 else 2),
                                 crop=lambda frame, box: frame),
        quality=SimpleNamespace(assess=lambda *args: SimpleNamespace(accepted=True)),
        extractor=SimpleNamespace(dimension=3, align=lambda frame, box: frame,
                                  extract=lambda frame: np.array([1, 0, 0], np.float32)),
    )
    monkeypatch.setattr(benchmark, "FaceComparisonSystem", lambda *_: system)
    monkeypatch.setattr(benchmark, "_sha256_file", lambda _: "hash")
    monkeypatch.setattr(benchmark, "_source_hashes", lambda _: {"fixture": "hash"})
    monkeypatch.setattr(benchmark, "_machine_info", lambda: {"cpu": "SYNTHETIC FIXTURE"})
    # A no-parallel-framework OpenCV build may report its CPU count and ignore
    # setNumThreads. Do not label this workload as measured single-threaded.
    monkeypatch.setattr(cv2, "setNumThreads", lambda _: None)
    monkeypatch.setattr(cv2, "getNumThreads", lambda: 8)
    monkeypatch.setattr(cv2, "imread", lambda _: np.ones((16, 16, 3), np.uint8))
    report = run_benchmark("fixture.png", repeat_single=2, gallery_sizes=(2, 4), warmup=1, iterations=3)
    assert database.closed and database.people == 4
    assert report["scene_kinds"]["multiple"].startswith("synthetic_layout")
    assert report["results"]["4"]["multiple"]["usable_faces_per_frame"] == [2]
    assert len(report["isolated_cached_retrieval"]["4"]["raw_ms"]) == 3
    assert len(report["raw_stage_samples"]["2"]["single"]) == 3
    assert report["machine"]["opencv_thread_control"]["requested"] == 1
    assert report["machine"]["opencv_thread_control"]["reported"] == 8
    assert not report["machine"]["opencv_thread_control"]["request_matched_report"]
    assert "cached_revision_check_ms" in report["isolated_cached_retrieval"]["4"]
    assert report["isolated_cached_retrieval"]["4"]["vector_payload_bytes"] == 4 * 3 * 3 * 4
    output = write_benchmark_results(report, tmp_path / "synthetic-only")
    assert verify_result_directory(output)["files"] == 3
    with pytest.raises(FileExistsError):
        write_benchmark_results(report, output)


@pytest.mark.parametrize("kwargs", [{"repeat_single": 1}, {"repeat_single": 2, "multiple_image": "x"},
                                    {"opencv_threads": 0}])
def test_benchmark_rejects_invalid_workload_before_loading_images(kwargs):
    with pytest.raises(ValueError):
        run_benchmark("no-image", **kwargs)
