"""Offline video stage benchmark: real service path, private synthetic gallery.

This measures processing cost, NOT accuracy, camera/UI FPS or independent
recognition evidence. It cannot certify that an input is real motion/authorized.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np

from ._baseline import PROJECT, FaceComparisonSystem, StorageConfig, load_config
from .benchmark import _machine_info, _thread_limit
from .evaluation.artifacts import artifact_hashes, new_result_directory
from .evaluation.reporting import _csv, _write_json
from .experiment import _sha256_file, _source_hashes
from .profiling import FrameProfiler, STAGES
from face_compare_system.face_compare.models import PreparedSample, QualityReport
from face_compare_system.face_compare.tracking import MultiFaceTracker


def _positive_integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}必须为至少{minimum}的整数")


@contextmanager
def _capture(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("无法打开本地视频文件")
        yield capture
    finally:
        capture.release()


def _video_metadata(capture):
    values = [float(capture.get(prop)) for prop in (
        cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("性能实验要求可靠的fps、帧数、宽高元数据；不自动以25fps补齐")
    fps, count, width, height = values
    if fps > 240 or any(value != int(value) for value in (count, width, height)):
        raise ValueError("视频元数据不在受支持范围；要求整数帧数/尺寸及fps≤240")
    return {"fps": fps, "declared_frames": int(count), "width": int(width), "height": int(height)}


def _check_frame(frame, metadata):
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.shape != (metadata["height"], metadata["width"], 3)):
        raise ValueError("解码帧格式或尺寸与固定视频元数据不一致")


def _percentiles(values):
    return {"p50": float(np.median(values)), "p95": float(np.percentile(values, 95))}


def summarize_timings(rows):
    """Conditional on detected workload, not on ground-truth face counts."""
    def summarize(subset):
        if not subset:
            return {"frames": 0, "latency_ms": None, "processing_fps": None}
        values = {stage: _percentiles([row[stage] for row in subset])
                  for stage in (*STAGES, "processing_ms", "read_ms")}
        total = sum(row["processing_ms"] for row in subset)
        return {"frames": len(subset), "latency_ms": values,
                "processing_fps": len(subset) * 1000 / total if total > 0 else None,
                "detected_faces": sum(row["detected_faces"] for row in subset),
                "usable_faces": sum(row["usable_faces"] for row in subset)}
    return {
        "all": summarize(rows),
        "no_detection": summarize([row for row in rows if row["detected_faces"] == 0]),
        "one_detection": summarize([row for row in rows if row["detected_faces"] == 1]),
        "multiple_detections": summarize([row for row in rows if row["detected_faces"] >= 2]),
    }


def _measure_video(system, path, *, every, max_frames, warmup):
    with _capture(path) as capture:
        metadata = _video_metadata(capture)
        ok, first = capture.read()
        if not ok:
            raise ValueError("视频第一帧无法解码")
        _check_frame(first, metadata)
    tracker = MultiFaceTracker(system.extractor, system.recognition_config, system.config.tracking)
    with FrameProfiler(system, tracker) as profiler:
        warmup_calls = dict.fromkeys(STAGES, 0)
        for index in range(warmup):
            _, measured = profiler.analyze(first, index / metadata["fps"])
            for stage in STAGES:
                warmup_calls[stage] += measured["stage_calls"][stage]
        # Warmup must never seed a measured track or identity vote. Recreate
        # rather than reset so even monotonically incremented track IDs restart.
    tracker = MultiFaceTracker(system.extractor, system.recognition_config, system.config.tracking)
    rows, decode_samples = [], []
    expected = min(metadata["declared_frames"], max_frames) if max_frames else metadata["declared_frames"]
    with FrameProfiler(system, tracker) as profiler, _capture(path) as capture:
        if _video_metadata(capture) != metadata:
            raise ValueError("两次打开视频的元数据发生变化")
        started = time.perf_counter_ns()
        decoded = 0
        while not max_frames or decoded < max_frames:
            tick = time.perf_counter_ns()
            ok, frame = capture.read()
            read_ms = (time.perf_counter_ns() - tick) / 1_000_000
            if not ok:
                break
            _check_frame(frame, metadata)
            index = decoded
            decoded += 1
            decode_samples.append(read_ms)
            if index % every:
                continue
            _, measured = profiler.analyze(frame, index / metadata["fps"])
            rows.append({"frame": index, "timestamp": index / metadata["fps"],
                         "read_ms": read_ms, **measured})
        stream_ms = (time.perf_counter_ns() - started) / 1_000_000
    if decoded != expected:
        raise ValueError("实际解码帧数与声明帧数/固定上限不一致，不发布截断视频的基准")
    return {
        "video_metadata": metadata, "decoded_frames": decoded, "analyzed_frames": len(rows),
        "warmup_frames": warmup, "warmup_stage_calls": warmup_calls,
        "stage_not_exercised_during_warmup": [stage for stage, count in warmup_calls.items() if count == 0],
        "decode_samples_ms": decode_samples, "stream_ms": stream_ms,
        "sampled_processing_fps_including_all_decode_and_loop": len(rows) * 1000 / stream_ms if stream_ms > 0 else None,
        "scene_timings": summarize_timings(rows), "raw_frames": rows,
    }


def run_video_benchmark(video, *, config_path=None, gallery_sizes=(10, 100, 1000),
                        templates_per_identity=3, every=3, max_frames=0, warmup=5,
                        seed=42, opencv_threads=1, synthetic_input=False):
    supplied_sizes = tuple(gallery_sizes)
    if not supplied_sizes:
        raise ValueError("gallery_sizes不能为空")
    for size in supplied_sizes:
        _positive_integer(size, "gallery_sizes")
    sizes = tuple(sorted(set(supplied_sizes)))
    for name, value, minimum in (("templates_per_identity", templates_per_identity, 1),
                                 ("every", every, 1), ("max_frames", max_frames, 0),
                                 ("warmup", warmup, 0), ("seed", seed, 0)):
        _positive_integer(value, name, minimum)
    if opencv_threads is not None:
        _positive_integer(opencv_threads, "opencv_threads")
    if type(synthetic_input) is not bool:
        raise ValueError("synthetic_input须为bool")
    path = Path(video).resolve()
    if not path.is_file():
        raise ValueError("只允许已有本地视频文件；不接受摄像头编号/网络URL")
    config_path = Path(config_path).resolve() if config_path else PROJECT / "config.json"
    config = load_config(config_path)
    if config.engine.backend != "sface" or config.storage.backend != "sqlite":
        raise ValueError("视频性能入口要求YuNet+SFace及SQLite")
    models = (PROJECT / config.engine.model_directory).resolve()

    def snapshot():
        return {"video_sha256": _sha256_file(path), "config_sha256": _sha256_file(config_path),
                "model_sha256": {name: _sha256_file(models / name) for name in (
                    "face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx")},
                "research_code_sha256": _source_hashes(Path(__file__).parent),
                "application_code_sha256": _source_hashes(PROJECT / "face_compare")}

    provenance = snapshot()
    with _thread_limit(opencv_threads), tempfile.TemporaryDirectory(prefix="face-video-benchmark-") as temporary, ExitStack() as cleanup:
        isolated = replace(config, engine=replace(config.engine, model_directory=str(models)),
                           storage=StorageConfig(str(Path(temporary) / "db"),
                                                 str(Path(temporary) / "events.jsonl"), False, "sqlite"))
        system = FaceComparisonSystem(isolated, temporary)
        cleanup.callback(system.database.close)
        rng = np.random.default_rng(seed)
        crop = np.zeros((112, 112, 3), np.uint8)
        quality = QualityReport(True, 128, 40, 100, .1)
        results, previous_size = {}, 0
        for size in sizes:
            for identity in range(previous_size, size):
                samples = []
                for _ in range(templates_per_identity):
                    feature = rng.normal(size=system.extractor.dimension).astype(np.float32)
                    feature /= np.linalg.norm(feature)
                    samples.append(PreparedSample(crop, feature, quality))
                system.database.add_samples(f"synthetic_{identity:05d}", samples)
            previous_size = size
            system.database.refresh_cache()
            results[str(size)] = _measure_video(system, path, every=every, max_frames=max_frames, warmup=warmup)
        if snapshot() != provenance:
            raise ValueError("运行期间视频、配置、代码或模型发生变化，拒绝发布混合结果")
        machine = _machine_info()
        machine["opencv_thread_control"] = {"requested": opencv_threads, "reported": cv2.getNumThreads(),
            "request_matched_report": opencv_threads is None or opencv_threads == cv2.getNumThreads()}
    return {
        "schema": "video-stage-benchmark-v1", "kind": "latency_only_not_biometric_accuracy",
        "input_kind": "synthetic_wiring_not_real_motion" if synthetic_input else "provided_video_unverified_provenance",
        "provenance": provenance, "machine": machine, "opencv": cv2.__version__,
        "gallery_kind": "synthetic_unit_vectors_in_temporary_sqlite", "gallery_sizes": list(sizes),
        "templates_per_identity": templates_per_identity, "seed": seed,
        "every": every, "max_frames": max_frames, "warmup": warmup,
        "results": results,
        "limitations": [
            "只测CPU性能；合成gallery不能提供已知/未知准确率或真实身份投票负载结论",
            "原始视频须由调用者合法获取；工具不能认证运动真实性、授权或采集会话独立性",
            "计时包装实际analyze_frame和tracker.update，不重做推理；计时本身仍有少量开销",
            "quality包括裁剪及质量评估；matching包括真实身份判定，tracking包括关联/投票/冲突处理",
            "processing不含视频打开/解码/序列化/UI；stream包含全部解码、采样跳帧与循环，但不含打开、预热或产物写入",
            "处理FPS为分析帧数/处理总秒数，不是源视频FPS、相机FPS、GUI FPS或硬实时保证",
            "按检测数量分组，不是真值人数；漏检也可能进入no_detection分组，不能解释成无人场景",
            "每种规模重复首帧预热并重建tracker后从0帧测量；未触发的阶段显式列出，不能声称它们已预热",
            "各规模顺序运行而非交替配对；磁盘缓存/热状态会变化，不用于严谨新旧实现加速结论",
            "使用恒定fps的frame/fps时间轴；变帧率/缺帧/可变分辨率不属于本入口有效协议",
        ],
    }


def write_video_benchmark(report, output):
    destination = Path(output).resolve()
    with new_result_directory(destination) as staging:
        _write_json(staging / "metrics.json", report)
        _write_json(staging / "experiment_config.json", {key: value for key, value in report.items() if key != "results"})
        rows = [{"gallery_size": int(size), **{key: value for key, value in row.items() if key != "stage_calls"},
                 **{stage.removesuffix("_ms") + "_calls": value for stage, value in row["stage_calls"].items()}}
                for size, result in report["results"].items() for row in result["raw_frames"]]
        if not rows:
            raise ValueError("没有分析帧，不发布空基准")
        _csv(staging / "timing.csv", rows, list(rows[0]))
        _write_json(staging / "artifacts.json", {
            "schema": "video-benchmark-artifacts-v1", "sha256": artifact_hashes(staging),
        })
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--gallery-sizes", default="10,100,1000")
    parser.add_argument("--templates-per-identity", type=int, default=3)
    parser.add_argument("--every", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opencv-threads", type=int, default=1)
    parser.add_argument("--synthetic-input", action="store_true")
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("结果目录已存在，请使用新的run ID")
    report = run_video_benchmark(args.video, config_path=args.config,
        gallery_sizes=tuple(int(value) for value in args.gallery_sizes.split(",")),
        templates_per_identity=args.templates_per_identity, every=args.every, max_frames=args.max_frames,
        warmup=args.warmup, seed=args.seed, opencv_threads=args.opencv_threads, synthetic_input=args.synthetic_input)
    destination = write_video_benchmark(report, args.output)
    print(json.dumps({"output": str(destination), "input_kind": report["input_kind"],
                      "analyzed_frames": {size: result["analyzed_frames"] for size, result in report["results"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
