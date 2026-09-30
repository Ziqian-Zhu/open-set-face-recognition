"""Public-image scale/contrast/blur regressions; no camera, DB, or accuracy claim."""

import json
from dataclasses import replace
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from face_compare.config import load_config
from face_compare.deep_engine import YuNetDetector
from face_compare.models import BoundingBox
from face_compare.quality import FaceQualityAssessor


def main():
    config = load_config(ROOT / "config.json")
    detector = YuNetDetector(ROOT / "models", config.engine, replace(config.detection, min_face_pixels=20))
    assessor = FaceQualityAssessor(config.quality)
    image = cv2.imread(str(ROOT / ".qa/public_samples/lena.jpg"))
    if image is None:
        raise RuntimeError("缺少公开样例 .qa/public_samples/lena.jpg")
    boxes = detector.detect(image)
    if not boxes:
        raise RuntimeError("公开样例未检测到人脸")
    crop = detector.crop(image, boxes[0])
    cases = [(f"clear_scale_{size}", cv2.resize(crop, (size, size)), True)
             for size in (128, 256, 512)]
    base = cv2.resize(crop, (256, 256))
    cases.append(("contrast_0.7", np.round(128+(base.astype(float)-128)*.7).astype(np.uint8), True))
    for sigma in (4, 8):
        cases.append((f"defocus_sigma_{sigma}", cv2.GaussianBlur(base, (0, 0), sigma), False))
    for size in (128, 256):
        blurred = cv2.GaussianBlur(cv2.resize(crop, (size, size)), (0, 0), size/32)
        for sigma in (3, 5, 8, 12, 20):
            noise = np.random.default_rng(37).normal(0, sigma, (size, size))
            noisy = np.clip(blurred.astype(float)+noise[:, :, None], 0, 255).astype(np.uint8)
            cases.append((f"defocus_with_noise_{sigma}_size_{size}", noisy, False))
    for axis in (0, 1):
        kernel = np.zeros((31, 31), np.float32)
        if axis:
            kernel[:, 15] = 1/31
        else:
            kernel[15, :] = 1/31
        cases.append((f"motion_axis_{axis}", cv2.filter2D(base, -1, kernel), False))
    rows = []
    for name, face, expected in cases:
        height, width = face.shape[:2]
        report = assessor.assess(face, BoundingBox(0, 0, width, height), face.shape)
        assert report.accepted == expected, (name, report)
        if not expected:
            assert "画面模糊" in report.reasons, (name, report)
        rows.append({"case": name, **report.to_dict()})
    print(json.dumps({"kind": "public-image signal regression; NOT glasses or biometric accuracy validation",
                      "production_quality_config": True, "results": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
