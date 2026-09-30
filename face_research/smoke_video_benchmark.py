"""Real-model wiring smoke on repeated public stills, NEVER motion evidence."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

from ._baseline import PROJECT, FaceComparisonSystem, StorageConfig, load_config
from .experiment import _sha256_file
from .evaluation.artifacts import RunJournal
from .profiling import FrameProfiler
from .video_benchmark import run_video_benchmark, write_video_benchmark
from face_compare_system.face_compare.tracking import MultiFaceTracker


def _same_observations(first, second):
    assert len(first) == len(second)
    for left, right in zip(first, second, strict=True):
        for field in ("box", "quality", "recognition", "stable_recognition", "track_id", "tracking_state"):
            assert getattr(left, field) == getattr(right, field), field
        assert np.array_equal(left.crop, right.crop)
        assert ((left.feature is None and right.feature is None)
                or (left.feature is not None and right.feature is not None
                    and np.array_equal(left.feature, right.feature)))


def _smoke(output):
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError("结果已存在，请使用新的run ID")
    source = PROJECT / ".qa/public_samples/lena.jpg"
    original_hash = _sha256_file(source)
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError("需要已有本地OpenCV测试静态图；不联网下载或调用摄像头")
    with tempfile.TemporaryDirectory(prefix="video-profile-smoke-") as temporary, ExitStack() as cleanup:
        directory = Path(temporary)
        config = load_config(PROJECT / "config.json")
        isolated = replace(config, storage=StorageConfig(str(directory / "equivalence-db"),
                            str(directory / "events.jsonl"), False, "sqlite"))
        system = FaceComparisonSystem(isolated, PROJECT)
        cleanup.callback(system.database.close)
        boxes = system.detector.detect(image)
        if len(boxes) != 1:
            raise ValueError("固定测试图须检出单脸，不能用放宽检测阈值来通过自检")
        box = boxes[0]
        padding = round(max(box.width, box.height) * .6)
        portrait = image[max(0, box.y-padding):min(image.shape[0], box.bottom+padding),
                         max(0, box.x-padding):min(image.shape[1], box.right+padding)]
        portrait = cv2.resize(portrait, (512, 512))
        blank = np.full_like(portrait, 65)
        panels = [np.hstack((portrait, blank)), np.hstack((portrait, portrait)), np.hstack((blank, blank))]
        plain = MultiFaceTracker(system.extractor, system.recognition_config, config.tracking)
        measured = MultiFaceTracker(system.extractor, system.recognition_config, config.tracking)
        checked = 0
        with FrameProfiler(system, measured) as profiler:
            for index in range(12):
                frame = panels[(index // 4) % 3]
                expected = plain.update(system.analyze_frame(frame, log_events=False), index / 10)
                actual, _ = profiler.analyze(frame, index / 10)
                _same_observations(expected, actual)
                checked += 1
        video = directory / "repeated-stills-not-motion.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10., (1024, 512))
        try:
            if not writer.isOpened():
                raise RuntimeError("无法打开MJPG测试编码器")
            for index in range(24):
                writer.write(panels[(index // 8) % 3])
        finally:
            writer.release()
        report = run_video_benchmark(video, gallery_sizes=(10, 100), every=1, warmup=3, synthetic_input=True)
        # Both fully accepted and rejected/empty paths must be represented;
        # never relax production quality just to make this assertion pass.
        checks = {}
        for size, measured in report["results"].items():
            checks[size] = {
                "analyzed_all_frames": measured["analyzed_frames"] == 24,
                "has_single": measured["scene_timings"]["one_detection"]["frames"] > 0,
                "has_multiple": measured["scene_timings"]["multiple_detections"]["frames"] > 0,
                "has_empty": measured["scene_timings"]["no_detection"]["frames"] > 0,
                "ran_embedding": sum(row["stage_calls"]["embedding_ms"] for row in measured["raw_frames"]) > 0,
            }
        if _sha256_file(source) != original_hash:
            raise ValueError("测试静态图在运行期间变化")
        report["smoke_fixture"] = {
            "source_image_sha256": original_hash, "frames": 24,
            "layout": "8 single + 8 duplicated-identical-face + 8 blank frames; static wiring fixture",
            "portrait_recipe": "YuNet box padded by .6*max(width,height), clipped to source, resized to512x512",
            "real_model_equivalence_frames": checked, "observations_equal": True,
            "production_quality_unchanged": True,
            "coverage_checks": checks, "passed": all(all(row.values()) for row in checks.values()),
            "limitations": "不是两个不同的人，不是真实运动或准确率；等价检查使用临时空库",
        }
        # Persist the actual measurements even when the fixture fails to cover
        # a branch. A future retry must use a different run ID.
        write_video_benchmark(report, destination)
        if not report["smoke_fixture"]["passed"]:
            raise RuntimeError("固定自检布局未覆盖必要路径；失败测量已保存，不放宽默认门槛")
    return {"output": str(destination), "input_kind": report["input_kind"],
            "equivalence_frames": checked, "artifact_files": 3,
            "scene_frames": {size: {scene: values["frames"] for scene, values in result["scene_timings"].items()}
                             for size, result in report["results"].items()}}


def smoke(output):
    # A failed layout/codec/model check reserves this ID just like other
    # research runs. This is a wiring fixture, not a validation/test split.
    with RunJournal(output, kind="synthetic-video-profiler-smoke") as journal:
        result = _smoke(output)
        journal.completed(Path(output).resolve())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(smoke(args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
