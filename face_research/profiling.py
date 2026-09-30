"""Opt-in instance-local timings of the real service/tracker call path.

Only use on a private, single-threaded research instance. No class-level patch,
second inference pass, changed decision, or persistent application modification.
"""

from __future__ import annotations

from functools import wraps
import time


STAGES = ("detection_ms", "quality_ms", "alignment_ms", "embedding_ms", "matching_ms", "tracking_ms")
_MISSING = object()


class FrameProfiler:
    """Instrument a private instance temporarily; restore even on exceptions."""

    def __init__(self, system, tracker):
        self.system = system
        self.tracker = tracker
        self._restores = []
        self._entered = False
        self._measuring = False
        self._times = {}
        self._counts = {}

    def _patch(self, instance, name, stage):
        original = getattr(instance, name)
        previous = vars(instance).get(name, _MISSING)
        if not callable(original):
            raise TypeError(f"{name}必须可调用")

        @wraps(original)
        def measured(*args, **kwargs):
            if not self._measuring:
                return original(*args, **kwargs)
            start = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                self._times[stage] += (time.perf_counter_ns() - start) / 1_000_000
                self._counts[stage] += 1

        setattr(instance, name, measured)
        self._restores.append((instance, name, previous))

    def __enter__(self):
        if self._entered:
            raise RuntimeError("同一profiler不可嵌套使用")
        self._entered = True
        try:
            for instance, name, stage in (
                (self.system.detector, "detect", "detection_ms"),
                (self.system.detector, "crop", "quality_ms"),
                (self.system.quality, "assess", "quality_ms"),
                (self.system.extractor, "align", "alignment_ms"),
                (self.system.extractor, "extract", "embedding_ms"),
                (self.system.recognizer, "match", "matching_ms"),
                (self.tracker, "update", "tracking_ms"),
            ):
                self._patch(instance, name, stage)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        for instance, name, previous in reversed(self._restores):
            if previous is _MISSING:
                delattr(instance, name)
            else:
                setattr(instance, name, previous)
        self._restores.clear()
        self._entered = False
        self._measuring = False
        return False

    def analyze(self, frame, timestamp):
        if not self._entered or self._measuring:
            raise RuntimeError("须在profiler上下文内逐帧调用，不可并发或递归")
        self._times = dict.fromkeys(STAGES, 0.)
        self._counts = dict.fromkeys(STAGES, 0)
        self._measuring = True
        start = time.perf_counter_ns()
        try:
            observations = self.tracker.update(self.system.analyze_frame(frame, log_events=False), timestamp)
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
        finally:
            self._measuring = False
        return observations, {
            **self._times, "processing_ms": elapsed,
            "stage_calls": dict(self._counts),
            "detected_faces": len(observations),
            "usable_faces": sum(row.quality.accepted and row.feature is not None for row in observations),
        }
