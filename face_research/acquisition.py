"""Opt-in acquisition telemetry on a private, single-threaded service instance.

Observe the actual prepare_sample call once: no second detection pass, relaxed
quality gate, copied preparation algorithm, raw image, vector, or exception text.
"""

from __future__ import annotations

import hashlib
from functools import wraps


_MISSING = object()


class AcquisitionInspector:
    """Proxy accepted by score_pair_manifest; always restores instance methods."""

    def __init__(self, system):
        self.system = system
        self.extractor = system.extractor
        self.records: dict[str, dict] = {}
        self._restores = []
        self._entered = False
        self._active = False
        self._current = None

    def _observe(self, instance, name, stage):
        original = getattr(instance, name)
        previous = vars(instance).get(name, _MISSING)
        if not callable(original):
            raise TypeError(f"{name}必须可调用")

        @wraps(original)
        def observed(*args, **kwargs):
            if not self._active:
                return original(*args, **kwargs)
            self._current["last_stage"] = stage
            self._current["stage_calls"][stage] += 1
            value = original(*args, **kwargs)
            if stage == "detection":
                self._current["detected_faces"] = len(value)
            elif stage == "quality":
                self._current["quality"] = value.to_dict()
            return value

        setattr(instance, name, observed)
        self._restores.append((instance, name, previous))

    def __enter__(self):
        if self._entered:
            raise RuntimeError("同一采集诊断器不可嵌套")
        self._entered = True
        try:
            for instance, name, stage in (
                (self.system.detector, "detect", "detection"),
                (self.system.detector, "crop", "crop"),
                (self.system.quality, "assess", "quality"),
                (self.system.extractor, "align", "alignment"),
                (self.system.extractor, "extract", "extraction"),
            ):
                self._observe(instance, name, stage)
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
        self._entered = self._active = False
        self._current = None
        return False

    def prepare_sample(self, image):
        if not self._entered or self._active:
            raise RuntimeError("须在诊断上下文内顺序调用，不可并发或递归")
        digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        self._current = {
            "image_sha256": digest, "status": "preparation_failed",
            "last_stage": "prepare", "detected_faces": None, "quality": None,
            "stage_calls": dict.fromkeys(("detection", "crop", "quality", "alignment", "extraction"), 0),
        }
        self._active = True
        try:
            sample = self.system.prepare_sample(image)
        except ValueError:
            count = self._current["detected_faces"]
            quality = self._current["quality"]
            if count == 0:
                status = "no_face"
            elif count is not None and count != 1:
                status = "multiple_faces"
            elif quality is not None and not quality["accepted"]:
                status = "quality_rejected"
            else:
                status = self._current["last_stage"] + "_failed"
            self._current["status"] = status
            raise
        else:
            self._current["status"] = "usable"
            return sample
        finally:
            self.records[digest] = self._current
            self._active = False
            self._current = None
