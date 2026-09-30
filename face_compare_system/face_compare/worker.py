"""Bounded latest-frame inference: avoid queue latency and stale camera results."""

from __future__ import annotations

import queue
import threading
import time


class InferenceWorker:
    def __init__(self, system):
        self.system = system
        self.inputs = queue.Queue(maxsize=1)
        self.outputs = queue.Queue(maxsize=1)
        self.stopped = threading.Event()
        self.tracker = None
        self._stream_key = None
        if hasattr(system, "config") and hasattr(system, "extractor"):
            from .tracking import MultiFaceTracker
            self.tracker = MultiFaceTracker(system.extractor, system.recognition_config, system.config.tracking)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @staticmethod
    def _latest(target, value):
        try:
            target.get_nowait()
        except queue.Empty:
            pass
        try:
            target.put_nowait(value)
        except queue.Full:
            pass

    def submit(self, frame, generation, captured, *, tracking_time=None):
        self._latest(self.inputs, (frame.copy(), generation, captured,
                                   captured if tracking_time is None else tracking_time))

    def _run(self):
        while not self.stopped.is_set():
            try:
                frame, generation, captured, tracking_time = self.inputs.get(timeout=0.1)
            except queue.Empty:
                continue
            start = time.perf_counter()
            try:
                with self.system.lock:
                    if self.tracker is not None:
                        key = (generation, getattr(self.system.database, "revision", None))
                        if key != self._stream_key:
                            self.tracker.reset()
                            self._stream_key = key
                    observations = self.system.analyze_frame(frame, log_events=False)
                    if self.tracker is not None:
                        if getattr(self.system.database, "revision", None) != key[1]:
                            raise RuntimeError("人员库在分析期间更新，正在重新确认")
                        observations = self.tracker.update(observations, tracking_time)
                error = None
            except Exception as exc:
                observations, error = [], str(exc)
                if self.tracker is not None:
                    self.tracker.reset()
            self._latest(self.outputs, (generation, captured, observations, error,
                                        (time.perf_counter()-start)*1000))

    def poll(self):
        try:
            return self.outputs.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        self.stopped.set()
