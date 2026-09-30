"""Name-bound enrollment session with quality, diversity and identity checks."""

from __future__ import annotations

import time
import cv2
import numpy as np

from .database import normalize_person_name


def validate_identity_set(samples, extractor, cosine_threshold):
    if not samples:
        raise ValueError("请先采集人脸")
    limit = (1-cosine_threshold)/2
    # Each sample must agree with the median of all other samples, not only its nearest peer.
    for index, sample in enumerate(samples):
        distances = [extractor.distance(sample.feature, other.feature)
                     for j, other in enumerate(samples) if j != index]
        if distances and float(np.median(distances)) > limit:
            raise ValueError("本组样本可能混入了其他人或姿态过大，请清空本次采集后重新录入")


class EnrollmentSession:
    def __init__(self):
        self.name = None
        self.samples = []
        self._last_time = -float("inf")
        self._thumbnails = []

    def clear(self):
        self.name = None
        self.samples.clear()
        self._thumbnails.clear()
        self._last_time = -float("inf")

    def validate_name(self, name):
        normalized = normalize_person_name(name)
        if self.name is not None and normalized.casefold() != self.name.casefold():
            raise ValueError(f"本次样本属于“{self.name}”。换人前请先保存或点“清空本次采集”")
        return normalized

    def add(self, name, sample, *, extractor=None, consistency=0.45, now=None):
        normalized = self.validate_name(name)
        now = time.monotonic() if now is None else now
        if now-self._last_time < 0.8:
            raise ValueError("采集太快，请稍微改变表情或角度，停稳后再采集")
        if not sample.quality.accepted:
            raise ValueError("样本质量不合格：" + "、".join(sample.quality.reasons))
        gray = cv2.cvtColor(sample.crop, cv2.COLOR_BGR2GRAY) if sample.crop.ndim == 3 else sample.crop
        thumb = cv2.resize(gray, (32, 32)).astype(np.float32)
        if any(float(np.mean(np.abs(thumb-old))) < 1.5 for old in self._thumbnails):
            raise ValueError("样本几乎重复，请轻微转头或改变表情后再采集")
        if extractor is not None:
            validate_identity_set([*self.samples, sample], extractor, consistency)
        self.name = normalized
        self.samples.append(sample)
        self._thumbnails.append(thumb)
        self._last_time = now


class TrackContinuity:
    """Conservative single-person continuity gate; not a multi-object tracker."""

    def __init__(self, extractor, max_gap=0.9, max_distance=0.30):
        self.extractor = extractor
        self.max_gap = max_gap
        self.max_distance = max_distance
        self.reset()

    def reset(self):
        self.box = self.feature = self.timestamp = None

    def update(self, observation, now):
        from .detector import FaceDetector
        same = self.box is not None and now-self.timestamp <= self.max_gap
        if same:
            same = FaceDetector._intersection_over_union(self.box, observation.box) >= 0.25
        if same and self.feature is not None and observation.feature is not None:
            same = self.extractor.distance(self.feature, observation.feature) <= self.max_distance
        self.box, self.feature, self.timestamp = observation.box, observation.feature, now
        return bool(same)
