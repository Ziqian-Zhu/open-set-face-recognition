"""Only the newest frame should wait for a busy model; faults must not kill it."""

import queue
import threading
import time
from types import SimpleNamespace

import numpy as np

from face_compare.worker import InferenceWorker


def test_video_media_time_is_separate_from_stale_result_wall_time():
    from unittest.mock import Mock
    system = SimpleNamespace(lock=threading.RLock(), database=SimpleNamespace(revision=0),
                             analyze_frame=lambda *args, **kwargs: [])
    worker = InferenceWorker(system)
    worker.tracker = Mock()
    worker.tracker.update.return_value = []
    try:
        captured = time.monotonic()
        worker.submit(np.zeros((2, 2)), 1, captured, tracking_time=.125)
        packet = worker.outputs.get(timeout=2)
        assert packet[1] == captured
        worker.tracker.update.assert_called_once_with([], .125)
        assert packet[3] is None
    finally:
        worker.close()
        worker.thread.join(timeout=2)


def test_worker_discards_intermediate_backlog_and_survives_error():
    entered = threading.Event()
    release = threading.Event()
    done = threading.Event()
    seen = []

    def analyze(frame, log_events=False):
        value = int(frame[0, 0])
        seen.append(value)
        if value == 1:
            entered.set()
            assert release.wait(2)
            raise ValueError("bad frame")
        done.set()
        return [value]

    worker = InferenceWorker(SimpleNamespace(lock=threading.RLock(), analyze_frame=analyze))
    try:
        worker.submit(np.full((2, 2), 1), 1, time.monotonic())
        assert entered.wait(2)
        worker.submit(np.full((2, 2), 2), 1, time.monotonic())
        worker.submit(np.full((2, 2), 3), 2, time.monotonic())
        release.set()
        assert done.wait(2)
        packet = worker.outputs.get(timeout=2)
        if packet[0] == 1:
            packet = worker.outputs.get(timeout=2)
        assert seen == [1, 3]
        assert packet[0] == 2 and packet[2] == [3] and packet[3] is None
    finally:
        release.set()
        worker.close()
        worker.thread.join(timeout=2)


def test_log_failure_does_not_turn_success_into_a_retry(tmp_path):
    import pytest
    from face_compare.event_log import JsonlEventLogger
    logger = JsonlEventLogger(tmp_path/"directory")
    logger.path.mkdir()
    with pytest.warns(RuntimeWarning, match="日志"):
        logger.write("enrollment", added_samples=3)
    assert logger.last_error is not None
