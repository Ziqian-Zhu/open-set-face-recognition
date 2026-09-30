"""Two public OpenCV images: integration smoke only, NOT an accuracy benchmark."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from face_compare.config import load_config, StorageConfig
from face_compare.service import FaceComparisonSystem


def main():
    config = load_config(ROOT/"config.json")
    with tempfile.TemporaryDirectory(prefix="sface-smoke-") as temporary:
        config = replace(config, engine=replace(config.engine, model_directory=str(ROOT/"models")),
                         storage=StorageConfig(str(Path(temporary)/"db"), str(Path(temporary)/"events.jsonl"), False, config.storage.backend),
                         detection=replace(config.detection, min_face_pixels=20))
        system = FaceComparisonSystem(config, temporary)
        samples = []
        latency = []
        quality = []
        for filename in ("lena.jpg", "messi5.jpg"):
            image = cv2.imread(str(ROOT/".qa/public_samples"/filename))
            if image is None:
                raise RuntimeError(f"缺少公开测试图：{filename}")
            started = time.perf_counter()
            boxes = system.detector.detect(image)
            if not boxes:
                raise RuntimeError(f"未检测到公开样例人脸：{filename}")
            box = boxes[0]
            # A tight context ROI lets us exercise the normal enrollment quality gate.
            pad = int(max(box.width, box.height)*0.45)
            roi = image[max(0, box.y-pad):min(image.shape[0], box.bottom+pad),
                        max(0, box.x-pad):min(image.shape[1], box.right+pad)]
            roi = cv2.resize(roi, (480, round(480*roi.shape[0]/roi.shape[1])))
            # Messi's public sample has a very small face; record quality rejection
            # explicitly, then bypass it ONLY for model wiring verification.
            prepared = system.prepare_sample(roi, allow_low_quality=True)
            quality.append({"image": filename, **prepared.quality.to_dict()})
            samples.append((roi, prepared))
            latency.append((time.perf_counter()-started)*1000)
        system.enroll_samples("public_A", [samples[0][1]])
        own = system.recognizer.match(samples[0][1].feature)
        other = system.recognizer.match(samples[1][1].feature)
        assert own.known and not other.known
        result = {"kind": "public-image integration smoke; not held-out accuracy",
                  "engine": system.engine_label, "images": ["OpenCV lena.jpg", "OpenCV messi5.jpg"],
                  "landmark_alignment_and_embedding": "passed", "embedding_dimensions": 128,
                  "same_image_distance": own.distance, "different_person_distance": other.distance,
                  "threshold": other.threshold, "different_person_rejected": not other.known,
                  "cold_detect_align_extract_ms": latency,
                  "input_quality": quality,
                  "quality_gate_bypassed_for_model_smoke": True,
                  "warning": "Same-image acceptance only tests pipeline wiring; two people do not establish FAR/FRR."}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result


if __name__ == "__main__":
    main()
