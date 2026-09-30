from dataclasses import replace
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from face_research import video_benchmark as runner
from face_research.profiling import FrameProfiler, STAGES
from face_research.evaluation.artifacts import verify_result_directory
from face_compare_system.face_compare.config import AppConfig, EngineConfig, StorageConfig
from face_compare_system.face_compare.models import BoundingBox, QualityReport, RecognitionResult
from face_compare_system.face_compare.service import FaceComparisonSystem
from face_compare_system.face_compare.tracking import MultiFaceTracker


class FakeDatabase:
    def __init__(self):
        self.people = 0
        self.closed = False
        self.features = []

    def add_samples(self, name, samples):
        assert name.startswith("synthetic_")
        self.people += 1
        self.features.extend(sample.feature.copy() for sample in samples)

    def refresh_cache(self):
        pass

    def close(self):
        self.closed = True


def fake_system(config=None):
    """Use actual analyze_frame/tracker, fake only model/numeric inputs."""
    system = object.__new__(FaceComparisonSystem)
    system.config = config or AppConfig(engine=EngineConfig(backend="sface"))
    system.recognition_config = system.config.recognition
    system.database = FakeDatabase()
    system.detector = SimpleNamespace(
        detect=lambda frame: [BoundingBox(0, 0, 8, 8)] if frame[0, 0, 0] else [],
        crop=lambda frame, box: frame[:8, :8],
    )
    system.extractor = SimpleNamespace(dimension=2, align=lambda frame, box: frame.copy(),
        extract=lambda aligned: np.array([1., 0.], np.float32), distance=lambda a, b: float((1-a@b)/2))
    system.quality = SimpleNamespace(assess=lambda *_: QualityReport(True, 128, 40, 100, .2))
    system.recognizer = SimpleNamespace(threshold=.275, match=lambda _: RecognitionResult(True, "A", "a", 95, .05, .275, "fixture"))
    return system


def decisions(observations):
    return [(row.box, row.quality, row.recognition, row.stable_recognition, row.track_id, row.tracking_state,
             row.feature.tolist() if row.feature is not None else None) for row in observations]


def test_profiler_uses_actual_path_once_and_restores_instance_methods():
    system = fake_system()
    plain = MultiFaceTracker(system.extractor, system.recognition_config)
    profiled = MultiFaceTracker(system.extractor, system.recognition_config)
    originals = [dict(vars(obj)) for obj in (system.detector, system.quality, system.extractor, system.recognizer)]
    frame = np.ones((16, 16, 3), np.uint8)
    with FrameProfiler(system, profiled) as profiler:
        for index in range(8):
            expected = plain.update(system.analyze_frame(frame, log_events=False), index / 10)
            actual, row = profiler.analyze(frame, index / 10)
            assert decisions(expected) == decisions(actual)
            assert row["stage_calls"] == dict(zip(STAGES, (1, 2, 1, 1, 1, 1), strict=True))
            assert row["detected_faces"] == row["usable_faces"] == 1
            assert row["processing_ms"] >= sum(row[stage] for stage in STAGES)
    assert "update" not in vars(profiled)  # class method isn't left shadowed
    assert originals == [dict(vars(obj)) for obj in (system.detector, system.quality, system.extractor, system.recognizer)]


def test_profiler_quality_failure_and_empty_frame_do_not_infer_extra_faces():
    system = fake_system()
    system.quality.assess = lambda *_: QualityReport(False, 128, 40, 0, .2, ("blur",))
    tracker = MultiFaceTracker(system.extractor, system.recognition_config)
    with FrameProfiler(system, tracker) as profiler:
        observations, row = profiler.analyze(np.ones((16, 16, 3), np.uint8), 0)
        assert observations[0].tracking_state == "quality_rejected"
        assert row["stage_calls"]["embedding_ms"] == row["stage_calls"]["matching_ms"] == 0
        assert row["usable_faces"] == 0
        empty, row = profiler.analyze(np.zeros((16, 16, 3), np.uint8), .1)
        assert not empty and row["detected_faces"] == 0
        assert row["stage_calls"]["quality_ms"] == 0


def test_profiler_cleanup_on_inference_error_and_partial_install_error():
    system = fake_system()
    def fail(_):
        raise RuntimeError("inference failed")
    system.extractor.extract = fail
    before = dict(vars(system.detector))
    tracker = MultiFaceTracker(system.extractor, system.recognition_config)
    profiler = FrameProfiler(system, tracker)
    with pytest.raises(RuntimeError, match="inference"):
        with profiler:
            profiler.analyze(np.ones((16, 16, 3), np.uint8), 0)
    assert system.extractor.extract is fail
    assert dict(vars(system.detector)) == before
    assert "update" not in vars(tracker)
    with pytest.raises(RuntimeError, match="上下文"):
        profiler.analyze(np.ones((16, 16, 3), np.uint8), 0)
    del system.extractor.align
    with pytest.raises(AttributeError):
        with profiler:
            pass
    assert dict(vars(system.detector)) == before


class FakeCapture:
    def __init__(self, *, declared=6, decoded=6, fps=10., pixels=1):
        self.position = 0
        self.declared = declared
        self.decoded = decoded
        self.fps = fps
        self.pixels = pixels
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_COUNT: self.declared,
                cv2.CAP_PROP_FRAME_WIDTH: 16, cv2.CAP_PROP_FRAME_HEIGHT: 16}[prop]

    def read(self):
        self.position += 1
        return (True, np.full((16, 16, 3), self.pixels, np.uint8)) if self.position <= self.decoded else (False, None)

    def release(self):
        self.released = True


def patch_capture(monkeypatch, **kwargs):
    captures = []
    def factory(path):
        assert isinstance(path, str)
        capture = FakeCapture(**kwargs)
        captures.append(capture)
        return capture
    monkeypatch.setattr(runner.cv2, "VideoCapture", factory)
    return captures


def test_measure_video_sampling_warmup_and_state_reset(monkeypatch):
    captures = patch_capture(monkeypatch)
    system = fake_system()
    observed_states = []
    original = FrameProfiler.analyze
    def inspect(self, *args):
        result = original(self, *args)
        observed_states.append([row.tracking_state for row in result[0]])
        return result
    monkeypatch.setattr(FrameProfiler, "analyze", inspect)
    result = runner._measure_video(system, "fixture.avi", every=2, max_frames=0, warmup=5)
    assert [row["frame"] for row in result["raw_frames"]] == [0, 2, 4]
    assert [row["timestamp"] for row in result["raw_frames"]] == [0, .2, .4]
    assert observed_states[4] == ["known"]
    assert observed_states[5] == ["confirming"]  # warmup votes were discarded
    assert result["decoded_frames"] == len(result["decode_samples_ms"]) == 6
    assert result["analyzed_frames"] == 3
    assert result["scene_timings"]["one_detection"]["frames"] == 3
    assert result["scene_timings"]["multiple_detections"]["latency_ms"] is None
    assert all(capture.released for capture in captures)


@pytest.mark.parametrize("kwargs", [{"fps": 0}, {"fps": float("nan")}, {"fps": 241},
                                     {"declared": 0}, {"declared": 6.5}, {"decoded": 3}])
def test_bad_metadata_or_truncated_decode_fails_with_release(monkeypatch, kwargs):
    captures = patch_capture(monkeypatch, **kwargs)
    with pytest.raises(ValueError):
        runner._measure_video(fake_system(), "fixture.avi", every=1, max_frames=0, warmup=1)
    assert all(capture.released for capture in captures)


def test_zero_detections_preserves_timing_and_unexercised_warmup_stages(monkeypatch):
    patch_capture(monkeypatch, pixels=0)
    result = runner._measure_video(fake_system(), "fixture.avi", every=2, max_frames=4, warmup=2)
    assert result["decoded_frames"] == 4
    assert result["scene_timings"]["no_detection"]["frames"] == 2
    assert "embedding_ms" in result["stage_not_exercised_during_warmup"]
    assert result["scene_timings"]["one_detection"]["frames"] == 0


@pytest.mark.parametrize("kwargs", [{"every": 0}, {"every": True}, {"warmup": -1},
    {"max_frames": .5}, {"gallery_sizes": (1, True)}, {"gallery_sizes": ()},
    {"templates_per_identity": 0}, {"opencv_threads": False}, {"seed": -1}, {"synthetic_input": 1}])
def test_invalid_request_rejected_before_reading_data(kwargs):
    with pytest.raises(ValueError):
        runner.run_video_benchmark("missing.avi", **kwargs)


def _patch_runner(tmp_path, monkeypatch):
    path = tmp_path / "fixture.avi"
    path.write_bytes(b"fixture not real video")
    config = AppConfig(engine=EngineConfig(backend="sface"), storage=StorageConfig(backend="sqlite"))
    monkeypatch.setattr(runner, "load_config", lambda _: config)
    monkeypatch.setattr(runner, "_sha256_file", lambda _: "hash")
    monkeypatch.setattr(runner, "_source_hashes", lambda _: {"fixture": "hash"})
    monkeypatch.setattr(runner, "_machine_info", lambda: {"cpu": "SYNTHETIC FIXTURE"})
    monkeypatch.setattr(cv2, "setNumThreads", lambda _: None)
    monkeypatch.setattr(cv2, "getNumThreads", lambda: 8)
    captures = patch_capture(monkeypatch)
    systems = []
    def factory(isolated, root):
        assert isolated.storage.backend == "sqlite" and not isolated.storage.save_face_images
        assert isolated.storage.database_directory.startswith(str(root))
        assert root != runner.PROJECT
        system = fake_system(isolated)
        systems.append(system)
        return system
    monkeypatch.setattr(runner, "FaceComparisonSystem", factory)
    return path, systems, captures


def test_runner_isolated_deterministic_and_artifacts_private_create_only(tmp_path, monkeypatch):
    path, systems, captures = _patch_runner(tmp_path, monkeypatch)
    report = runner.run_video_benchmark(path, gallery_sizes=(2, 3), every=2, warmup=2, synthetic_input=True)
    vectors = systems[0].database.features
    again = runner.run_video_benchmark(path, gallery_sizes=(2, 3), every=2, warmup=2, synthetic_input=True)
    assert all(np.array_equal(a, b) for a, b in zip(vectors, systems[1].database.features, strict=True))
    assert all(np.linalg.norm(vector) == pytest.approx(1) for vector in vectors)
    assert all(system.database.closed for system in systems) and all(c.released for c in captures)
    assert report["machine"]["opencv_thread_control"]["request_matched_report"] is False
    assert report["input_kind"] == again["input_kind"] == "synthetic_wiring_not_real_motion"
    assert len(report["results"]["3"]["raw_frames"]) == 3
    output = runner.write_video_benchmark(report, tmp_path / "result")
    assert verify_result_directory(output) == {"status": "verified", "files": 3}
    assert str(tmp_path) not in (output / "metrics.json").read_text()
    assert "A" not in (output / "timing.csv").read_text()
    assert json.loads((output / "metrics.json").read_text())["gallery_kind"].startswith("synthetic_")
    with pytest.raises(FileExistsError):
        runner.write_video_benchmark(report, output)


def test_runner_rejects_mutation_and_closes_database(tmp_path, monkeypatch):
    path, systems, captures = _patch_runner(tmp_path, monkeypatch)
    reads = []
    def hashing(candidate):
        if candidate == path:
            reads.append(candidate)
            return "first" if len(reads) == 1 else "changed"
        return "hash"
    monkeypatch.setattr(runner, "_sha256_file", hashing)
    with pytest.raises(ValueError, match="发生变化"):
        runner.run_video_benchmark(path, gallery_sizes=(1,), warmup=0)
    assert systems[0].database.closed and all(c.released for c in captures)


def test_smoke_failure_reserves_run_id_and_does_not_lose_attempt(tmp_path, monkeypatch):
    from face_research import smoke_video_benchmark as smoke
    output = tmp_path / "failed-smoke"
    def fail(_):
        raise RuntimeError("fixture coverage failed")
    monkeypatch.setattr(smoke, "_smoke", fail)
    with pytest.raises(RuntimeError, match="coverage"):
        smoke.smoke(output)
    receipt = json.loads((tmp_path / "failed-smoke.audit/failed.json").read_text())
    assert receipt["error_type"] == "RuntimeError"
    assert not receipt["test_may_have_been_accessed"]
    with pytest.raises(FileExistsError):
        smoke.smoke(output)
