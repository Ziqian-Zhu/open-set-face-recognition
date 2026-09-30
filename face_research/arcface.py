"""Explicit optional ArcFace-family MobileFaceNet research adapter.

Only the pinned official Buffalo_SC recognition ONNX is supported. Weights are
non-commercial research-only, NOT covered by the library's MIT code license.
No auto-download, InsightFace import, attribute model, or extra detector.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from face_compare_system.face_compare.deep_engine import SFaceExtractor


MODEL_SHA256 = "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f"
MODEL_SIZE = 13616099
ARCHIVE_SHA256 = "57d31b56b6ffa911c8a73cfc1707c73cab76efe7f13b675a05223bf42de47c72"
SOURCE_URL = "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_sc.zip"
LICENSE_URL = "https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md#model-licenses"
# Standard five-point ArcFace template, in YuNet's image-left to image-right order.
REFERENCE_POINTS = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
REFERENCE_POINTS.setflags(write=False)


def similarity_transform(landmarks: np.ndarray) -> np.ndarray:
    """Least-squares orientation-preserving 2D similarity, not full affine/RANSAC.

    Solves x'=a*x-b*y+tx, y'=b*x+a*y+ty using all five points. Equivalent
    least-squares objective to upstream SimilarityTransform for this nonreflective
    112-pixel template. Reject degenerate/nonfinite geometry rather than guessing.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("ArcFace需要五个有限二维关键点")
    centered = points - points.mean(axis=0)
    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError("ArcFace关键点退化，不能对齐")
    design = np.zeros((10, 4), dtype=np.float64)
    design[0::2, 0], design[0::2, 1], design[0::2, 2] = points[:, 0], -points[:, 1], 1
    design[1::2, 0], design[1::2, 1], design[1::2, 3] = points[:, 1], points[:, 0], 1
    a, b, tx, ty = np.linalg.lstsq(design, REFERENCE_POINTS.astype(np.float64).ravel(), rcond=None)[0]
    if not np.isfinite([a, b, tx, ty]).all() or np.hypot(a, b) < 1e-8:
        raise ValueError("ArcFace对齐变换无效")
    return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float64)


class ArcFaceEmbedder:
    dimension = 512
    label = "ArcFace MBF（研究）"
    signature = f"arcface-w600k-mbf-{MODEL_SHA256}-ls5p112-rgb127.5-l2-opencv-v1"
    distance = staticmethod(SFaceExtractor.distance)

    def __init__(self, model_path: str | Path, *, noncommercial_research: bool = False):
        if noncommercial_research is not True:
            raise ValueError("此官方权重仅用于非商业研究；必须显式确认noncommercial_research=True")
        self.model_path = Path(model_path).resolve()
        self.verify_model()
        self._net = cv2.dnn.readNetFromONNX(str(self.model_path))
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def verify_model(self) -> None:
        if not self.model_path.is_file() or self.model_path.stat().st_size != MODEL_SIZE:
            raise ValueError("缺少固定版本的w600k_mbf.onnx或文件大小不符；不会自动下载")
        if hashlib.sha256(self.model_path.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError("ArcFace模型SHA256不匹配，不加载其他权重或混用特征空间")

    def align(self, frame, box):
        if frame is None or frame.size == 0 or frame.dtype != np.uint8:
            raise ValueError("ArcFace输入须为非空uint8图像")
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim != 3 or frame.shape[2] != 3 or len(box.landmarks) != 10:
            raise ValueError("ArcFace需要BGR帧及YuNet五点关键点")
        matrix = similarity_transform(np.asarray(box.landmarks).reshape(5, 2))
        return cv2.warpAffine(frame, matrix, (112, 112), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def extract(self, aligned):
        if not isinstance(aligned, np.ndarray) or aligned.shape != (112, 112, 3) or aligned.dtype != np.uint8:
            raise ValueError("ArcFace输入须为已对齐112×112 uint8 BGR人脸")
        # Pinned w600k_mbf has no embedded input normalization; upstream formula.
        blob = cv2.dnn.blobFromImage(aligned, 1 / 127.5, (112, 112),
                                     (127.5, 127.5, 127.5), swapRB=True, crop=False)
        self._net.setInput(blob)
        feature = self._net.forward().reshape(-1).astype(np.float32).copy()
        norm = float(np.linalg.norm(feature))
        if feature.size != self.dimension or not np.isfinite(feature).all() or not np.isfinite(norm) or norm < 1e-8:
            raise ValueError("ArcFace输出必须是512维有限非零向量")
        return feature / norm

    def metadata(self) -> dict:
        return {"model": "InsightFace buffalo_sc / w600k_mbf", "sha256": MODEL_SHA256,
                "bytes": MODEL_SIZE, "archive_sha256": ARCHIVE_SHA256, "source_url": SOURCE_URL,
                "license_url": LICENSE_URL, "use": "non-commercial research only",
                "input": "112x112 uint8 BGR -> RGB NCHW float32 (x-127.5)/127.5",
                "output": "512-d float32 L2 normalized", "signature": self.signature,
                "runtime": f"OpenCV {cv2.__version__} DNN CPU"}
