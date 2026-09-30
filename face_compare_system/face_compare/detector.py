"""Offline Haar-cascade face detection with profile and lighting support."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import DetectionConfig
from .models import BoundingBox


class FaceDetector:
    """Detect frontal and profile faces without model downloads."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        cascade_root = Path(cv2.data.haarcascades)
        self._frontal = self._load_cascade(cascade_root / "haarcascade_frontalface_default.xml")
        profile_path = cascade_root / "haarcascade_profileface.xml"
        self._profile = (
            self._load_cascade(profile_path)
            if config.use_profile_detector and profile_path.exists()
            else None
        )
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    @staticmethod
    def _load_cascade(path: Path) -> cv2.CascadeClassifier:
        """Load a bundled cascade and fail with an actionable error."""

        cascade = cv2.CascadeClassifier(str(path))
        if cascade.empty():
            raise RuntimeError(f"无法加载 OpenCV 人脸检测器: {path}")
        return cascade

    def detect(self, frame: NDArray[np.uint8]) -> list[BoundingBox]:
        """Return de-duplicated face rectangles sorted from largest to smallest."""

        if frame is None or frame.size == 0:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        normalized = self._clahe.apply(gray)
        minimum = (self.config.min_face_pixels, self.config.min_face_pixels)
        raw_boxes: list[tuple[int, int, int, int]] = list(
            self._frontal.detectMultiScale(
                normalized,
                scaleFactor=self.config.scale_factor,
                minNeighbors=self.config.min_neighbors,
                minSize=minimum,
            )
        )
        if self._profile is not None:
            raw_boxes.extend(
                self._profile.detectMultiScale(
                    normalized,
                    scaleFactor=self.config.scale_factor,
                    minNeighbors=self.config.min_neighbors,
                    minSize=minimum,
                )
            )
            flipped = cv2.flip(normalized, 1)
            flipped_boxes = self._profile.detectMultiScale(
                flipped,
                scaleFactor=self.config.scale_factor,
                minNeighbors=self.config.min_neighbors,
                minSize=minimum,
            )
            width = gray.shape[1]
            raw_boxes.extend((width - x - w, y, w, h) for x, y, w, h in flipped_boxes)
        boxes = [BoundingBox(int(x), int(y), int(w), int(h)) for x, y, w, h in raw_boxes]
        return self._non_maximum_suppression(boxes, iou_threshold=0.35)

    @staticmethod
    def _non_maximum_suppression(
        boxes: list[BoundingBox],
        *,
        iou_threshold: float,
    ) -> list[BoundingBox]:
        """Merge overlapping detections emitted by multiple cascades."""

        kept: list[BoundingBox] = []
        for candidate in sorted(boxes, key=lambda box: box.area, reverse=True):
            if all(FaceDetector._intersection_over_union(candidate, item) < iou_threshold for item in kept):
                kept.append(candidate)
        return kept

    @staticmethod
    def _intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
        """Return rectangle intersection-over-union."""

        x1 = max(first.x, second.x)
        y1 = max(first.y, second.y)
        x2 = min(first.right, second.right)
        y2 = min(first.bottom, second.bottom)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = first.area + second.area - intersection
        return intersection / union if union else 0.0

    def crop(self, frame: NDArray[np.uint8], box: BoundingBox) -> NDArray[np.uint8]:
        """Crop a face with proportional context padding, clipped to the frame."""

        padding_x = int(box.width * self.config.padding_ratio)
        padding_y = int(box.height * self.config.padding_ratio)
        frame_height, frame_width = frame.shape[:2]
        left = max(0, box.x - padding_x)
        top = max(0, box.y - padding_y)
        right = min(frame_width, box.right + padding_x)
        bottom = min(frame_height, box.bottom + padding_y)
        return frame[top:bottom, left:right].copy()


def largest_box(boxes: list[BoundingBox]) -> BoundingBox | None:
    """Return the largest box, or ``None`` when the input is empty."""

    return max(boxes, key=lambda box: box.area, default=None)
