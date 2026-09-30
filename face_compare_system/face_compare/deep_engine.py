"""OpenCV Zoo YuNet landmarks + aligned SFace unit embeddings, CPU inference."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from .config import DetectionConfig, EngineConfig
from .models import BoundingBox


MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "face_detection_yunet", "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", 232589,
    ),
    "face_recognition_sface_2021dec.onnx": (
        "face_recognition_sface", "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79", 38696353,
    ),
}


def verified_model(directory: Path, filename: str) -> str:
    path = directory / filename
    _, digest, size = MODELS[filename]
    if not path.is_file() or path.stat().st_size != size:
        raise RuntimeError(f"模型缺失或不完整：{filename}。请运行 python scripts/download_models.py")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"模型校验失败：{filename}。请重新下载官方模型")
    return str(path)


class YuNetDetector:
    def __init__(self, directory: Path, engine: EngineConfig, config: DetectionConfig):
        self.engine = engine
        self.config = config
        self._net = cv2.FaceDetectorYN.create(
            verified_model(directory, "face_detection_yunet_2023mar.onnx"), "", (320, 320),
            engine.detection_score, 0.3, 5000,
        )

    def detect(self, frame):
        if frame is None or frame.size == 0:
            return []
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        height, width = frame.shape[:2]
        scale = min(1.0, self.engine.max_detection_side / max(height, width))
        small = cv2.resize(frame, (round(width * scale), round(height * scale)))
        sx, sy = small.shape[1] / width, small.shape[0] / height
        self._net.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self._net.detect(small)
        boxes = []
        for row in faces if faces is not None else []:
            if not np.isfinite(row).all():
                continue
            points = row[4:14].reshape(5, 2) / np.array([sx, sy])
            left, top = max(0, round(row[0] / sx)), max(0, round(row[1] / sy))
            right = min(width, round((row[0] + row[2]) / sx))
            bottom = min(height, round((row[1] + row[3]) / sy))
            if min(right - left, bottom - top) < self.config.min_face_pixels:
                continue
            if np.linalg.norm(points[0] - points[1]) < 8:
                continue
            boxes.append(BoundingBox(left, top, right-left, bottom-top,
                                     tuple(float(x) for x in points.ravel()), float(row[14])))
        return sorted(boxes, key=lambda box: box.area, reverse=True)

    @staticmethod
    def crop(frame, box):
        return frame[box.y:box.bottom, box.x:box.right].copy()


class SFaceExtractor:
    dimension = 128
    signature = "sface-2021dec-aligned-cosine-v1"
    label = "SFace"

    def __init__(self, directory: Path):
        self._net = cv2.FaceRecognizerSF.create(
            verified_model(directory, "face_recognition_sface_2021dec.onnx"), "",
        )

    def align(self, frame, box):
        if len(box.landmarks) != 10:
            raise ValueError("SFace需要五点对齐，不能使用无关键点的人脸框")
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        row = np.asarray([*box.as_tuple(), *box.landmarks, box.confidence], dtype=np.float32)
        return self._net.alignCrop(frame, row)

    def extract(self, aligned):
        if aligned.shape != (112, 112, 3):
            raise ValueError("SFace输入必须是经过五点对齐的112×112 BGR人脸")
        feature = self._net.feature(aligned).reshape(-1).astype(np.float32).copy()
        norm = float(np.linalg.norm(feature))
        if feature.size != self.dimension or not np.isfinite(feature).all() or norm < 1e-8:
            raise ValueError("人脸特征无效，请重新采集")
        return feature / norm

    @staticmethod
    def distance(first, second):
        a, b = np.asarray(first).reshape(-1), np.asarray(second).reshape(-1)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("特征维度或数值无效")
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator < 1e-8:
            raise ValueError("不能比较零向量")
        # Half cosine distance lies in [0,1], matching the existing distance API.
        return float((1 - np.clip(np.dot(a, b) / denominator, -1, 1)) / 2)
