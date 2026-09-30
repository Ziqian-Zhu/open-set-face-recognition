"""Real model + SQLite + video decoding + tracks; public reused images, NOT accuracy."""

import json
from pathlib import Path
import sys
import tempfile
from dataclasses import replace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from face_compare.config import load_config, StorageConfig
from face_compare.service import FaceComparisonSystem
from face_compare.video import analyze_video


def main():
    with tempfile.TemporaryDirectory(prefix="face-video-smoke-") as folder:
        directory = Path(folder)
        config = load_config(ROOT/"config.json")
        config = replace(config, storage=StorageConfig(str(directory/"db"), str(directory/"events.jsonl"), False, "sqlite"),
                         engine=replace(config.engine, max_detection_side=1024),
                         # This wiring-only smoke deliberately bypasses focus for
                         # tiny public faces; never reuse it as quality validation.
                         quality=replace(config.quality, focus_method="legacy", min_blur_variance=0,
                                         min_contrast=0, min_face_ratio=.001),
                         detection=replace(config.detection, min_face_pixels=20))
        system = FaceComparisonSystem(config, ROOT)
        panels = []
        for filename in ("lena.jpg", "messi5.jpg"):
            image = cv2.imread(str(ROOT/".qa/public_samples"/filename))
            if image is None:
                raise RuntimeError("先准备已有的OpenCV公开测试图，再运行本连通性检查")
            b = system.detector.detect(image)[0]
            pad = round(max(b.width, b.height)*.6)
            image = image[max(0,b.y-pad):min(image.shape[0],b.bottom+pad), max(0,b.x-pad):min(image.shape[1],b.right+pad)]
            scale = min(360/image.shape[1], 440/image.shape[0])
            image = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)))
            panel = np.full((480, 400, 3), 65, np.uint8)
            y, x = (480-image.shape[0])//2, (400-image.shape[1])//2
            panel[y:y+image.shape[0], x:x+image.shape[1]] = image
            panels.append(panel)
        system.enroll_samples("Public_A", [system.prepare_sample(panels[0])])
        system.database.close()
        path = directory/"two-people.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (800, 480))
        if not writer.isOpened():
            raise RuntimeError("本机缺少MJPG测试编码器")
        for _ in range(12):
            writer.write(np.hstack(panels))
        writer.release()
        output = directory/"video.jsonl"
        summary = analyze_video(config, ROOT, path, output, every=1)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        frames = [row for row in rows if row["type"] == "frame"]
        final = frames[-1]["faces"]
        assert len(final) == 2, final
        assert len({face["track_id"] for face in final}) == 2
        assert {face["state"] for face in final} == {"known", "unknown"}, final
        assert all(face["single_frame"]["state"] in {"known", "unknown", "quality_rejected"}
                   for frame in frames for face in frame["faces"])
        print(json.dumps({"kind": "public still-image video wiring smoke, NOT held-out accuracy", "summary": summary,
                          "final_faces": final, "quality_thresholds_relaxed_only_for_smoke": True,
                          "note": "复用公开静态图，未测试真实运动、遮挡或活体；临时库已隔离生产数据"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
