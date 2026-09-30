"""CPU stage benchmark with an isolated synthetic gallery; never an accuracy test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from ._baseline import PROJECT, FaceComparisonSystem, StorageConfig, load_config
from .experiment import _sha256_file, _source_hashes, write_report
from .evaluation.artifacts import artifact_hashes, new_result_directory
from .evaluation.reporting import _csv, _write_json, _save_png
from face_compare_system.face_compare.models import PreparedSample, QualityReport


def _milliseconds(start: int) -> float:
    return (time.perf_counter_ns() - start) / 1_000_000


def _profile_frame(system: FaceComparisonSystem, frame: np.ndarray) -> dict:
    started = time.perf_counter_ns()
    boxes = system.detector.detect(frame)
    detection_ms = _milliseconds(started)
    stages = {"quality_ms": 0.0, "alignment_ms": 0.0, "embedding_ms": 0.0, "matching_ms": 0.0}
    usable = 0
    for box in boxes:
        tick = time.perf_counter_ns()
        crop = system.detector.crop(frame, box)
        quality = system.quality.assess(crop, box, frame.shape)
        stages["quality_ms"] += _milliseconds(tick)
        if not quality.accepted:
            continue
        tick = time.perf_counter_ns()
        aligned = system.extractor.align(frame, box)
        stages["alignment_ms"] += _milliseconds(tick)
        tick = time.perf_counter_ns()
        feature = system.extractor.extract(aligned)
        stages["embedding_ms"] += _milliseconds(tick)
        tick = time.perf_counter_ns()
        system.database.rank_candidates(feature, system.recognition_config.nearest_samples)
        stages["matching_ms"] += _milliseconds(tick)
        usable += 1
    return {"detected_faces": len(boxes), "usable_faces": usable, "detection_ms": detection_ms,
            **stages, "end_to_end_ms": _milliseconds(started)}


def _summary(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("benchmark没有采样结果")
    stages = ("detection_ms", "quality_ms", "alignment_ms", "embedding_ms", "matching_ms", "end_to_end_ms")
    result = {
        stage: {"p50": float(np.median([row[stage] for row in rows])),
                "p95": float(np.percentile([row[stage] for row in rows], 95))}
        for stage in stages
    }
    result["processing_fps_from_p50"] = 1000 / result["end_to_end_ms"]["p50"] if result["end_to_end_ms"]["p50"] > 0 else None
    result["detected_faces_per_frame"] = sorted({row["detected_faces"] for row in rows})
    result["usable_faces_per_frame"] = sorted({row["usable_faces"] for row in rows})
    return result


@contextmanager
def _thread_limit(count: int | None):
    """Restore the caller's OpenCV thread setting even when measurement fails."""

    previous = cv2.getNumThreads()
    try:
        if count is not None:
            cv2.setNumThreads(count)
        yield
    finally:
        if count is not None:
            cv2.setNumThreads(previous)


def _machine_info() -> dict:
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            value = subprocess.run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                                   capture_output=True, text=True, check=True, timeout=2).stdout.strip()
            cpu = value or cpu
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "os": platform.system(), "os_release": platform.release(),
        "architecture": platform.machine(), "cpu": cpu,
        "logical_cpus": os.cpu_count(), "python": platform.python_version(),
        "numpy": np.__version__, "opencv_threads": cv2.getNumThreads(),
        "numpy_thread_environment": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "opencv_build_sha256": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
    }


def run_benchmark(
    single_image: str | Path,
    multiple_image: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
    gallery_sizes: tuple[int, ...] = (10, 100, 1000),
    templates_per_identity: int = 3,
    iterations: int = 30,
    warmup: int = 5,
    seed: int = 42,
    repeat_single: int = 0,
    opencv_threads: int | None = 1,
) -> dict:
    """Measure steady-state CPU inference; gallery vectors are synthetic only."""

    sizes = tuple(sorted(set(gallery_sizes)))
    if not sizes or any(type(size) is not int or size < 1 for size in sizes):
        raise ValueError("gallery_sizes必须是非空正整数集合")
    if type(templates_per_identity) is not int or templates_per_identity < 1:
        raise ValueError("templates_per_identity必须是正整数")
    if type(iterations) is not int or iterations < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("iterations必须为正整数，warmup不能为负数")
    if type(repeat_single) is not int or repeat_single not in (0, 2, 3, 4):
        raise ValueError("repeat_single只能是0或2/3/4；这是重复静态图负载，不是真实多人")
    if multiple_image is not None and repeat_single:
        raise ValueError("multiple_image与repeat_single不能同时指定")
    if opencv_threads is not None and (type(opencv_threads) is not int or opencv_threads < 1):
        raise ValueError("opencv_threads必须为正整数或None")
    paths = [Path(single_image).resolve()]
    if multiple_image is not None:
        paths.append(Path(multiple_image).resolve())
    input_hashes = [_sha256_file(path) for path in paths]
    frames = [cv2.imread(str(path)) for path in paths]
    if any(frame is None for frame in frames):
        raise ValueError("benchmark图像无法读取")
    scene_kinds = {"single": "provided_static_image"}
    if repeat_single:
        frames.append(np.concatenate([frames[0]] * repeat_single, axis=1))
        scene_kinds["multiple"] = f"synthetic_layout_of_{repeat_single}_identical_images_not_distinct_people"
    elif multiple_image is not None:
        scene_kinds["multiple"] = "provided_static_image_not_video"
    config_path = Path(config_path).resolve() if config_path else PROJECT / "config.json"
    config = load_config(config_path)
    if config.engine.backend != "sface" or config.storage.backend != "sqlite":
        raise ValueError("benchmark要求YuNet+SFace和SQLite")
    model_dir = (PROJECT / config.engine.model_directory).resolve()
    config_hash = _sha256_file(config_path)
    model_hashes = {name: _sha256_file(model_dir / name) for name in (
        "face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx")}
    code_hashes = _source_hashes(Path(__file__).parent)
    course_hashes = _source_hashes(PROJECT / "face_compare")
    with _thread_limit(opencv_threads), tempfile.TemporaryDirectory(prefix="face-benchmark-") as temporary, ExitStack() as cleanup:
        isolated = replace(config, engine=replace(config.engine, model_directory=str(model_dir)),
                           storage=StorageConfig(str(Path(temporary) / "data"),
                                                  str(Path(temporary) / "events.jsonl"), False, "sqlite"))
        system = FaceComparisonSystem(isolated, temporary)
        cleanup.callback(system.database.close)
        initial = [_profile_frame(system, frame) for frame in frames]
        if initial[0]["detected_faces"] != 1 or (len(initial) > 1 and initial[1]["detected_faces"] < 2):
            raise ValueError("single图像须恰好检测1张脸，multiple图像须至少检测2张脸")
        if any(row["usable_faces"] < 1 for row in initial):
            raise ValueError("每个基准场景至少需一张通过原有质量门禁的脸；不自动放宽门槛")
        rng = np.random.default_rng(seed)
        queries = np.random.default_rng(seed + 1).normal(size=(iterations, system.extractor.dimension)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        empty_crop = np.zeros((112, 112, 3), np.uint8)
        quality = QualityReport(True, 128, 40, 100, 0.1)
        current_size = 0
        results = {}
        retrieval, raw_samples = {}, {}
        for size in sizes:
            build_started = time.perf_counter_ns()
            for identity in range(current_size, size):
                samples = []
                for _ in range(templates_per_identity):
                    vector = rng.normal(size=system.extractor.dimension).astype(np.float32)
                    vector /= np.linalg.norm(vector)
                    samples.append(PreparedSample(empty_crop, vector, quality))
                system.database.add_samples(f"synthetic_{identity:05d}", samples)
            current_size = size
            build_ms = _milliseconds(build_started)
            cache_started = time.perf_counter_ns()
            system.database.refresh_cache()
            cache_ms = _milliseconds(cache_started)
            for i in range(warmup):
                system.database.rank_candidates(queries[i % iterations], system.recognition_config.nearest_samples)
            query_ms = []
            for query in queries:
                tick = time.perf_counter_ns()
                system.database.rank_candidates(query, system.recognition_config.nearest_samples)
                query_ms.append(_milliseconds(tick))
            retrieval[str(size)] = {
                "p50_ms": float(np.median(query_ms)), "p95_ms": float(np.percentile(query_ms, 95)),
                "raw_ms": query_ms, "incremental_gallery_build_ms": build_ms,
                "cached_revision_check_ms": cache_ms,
                "build_note": "add_samples includes cache rebuilds; the subsequent refresh is only a revision check",
                "vector_payload_bytes": size * templates_per_identity * system.extractor.dimension * 4,
                "vector_payload_note": "one float32 matrix payload only; NOT total process RSS or total cache memory",
            }
            scenes = {}
            raw_samples[str(size)] = {}
            for label, frame in zip(("single", "multiple"), frames):
                for _ in range(warmup):
                    _profile_frame(system, frame)
                rows = [_profile_frame(system, frame) for _ in range(iterations)]
                scenes[label] = _summary(rows)
                raw_samples[str(size)][label] = rows
            results[str(size)] = scenes
        if ([_sha256_file(path) for path in paths] != input_hashes or _sha256_file(config_path) != config_hash
                or _source_hashes(Path(__file__).parent) != code_hashes
                or _source_hashes(PROJECT / "face_compare") != course_hashes
                or any(_sha256_file(model_dir / name) != digest for name, digest in model_hashes.items())):
            raise ValueError("基准运行期间输入、配置、模型或代码发生变化")
        machine = _machine_info()
        machine["opencv_thread_control"] = {
            "requested": opencv_threads,
            "reported": cv2.getNumThreads(),
            "request_matched_report": opencv_threads is None or cv2.getNumThreads() == opencv_threads,
            "note": "OpenCV may ignore setNumThreads in some builds; this is not an OS-level worker count or a BLAS limit",
        }
        return {
            "kind": "CPU stage latency and synthetic-gallery scaling; NOT biometric accuracy",
            "config_sha256": config_hash, "model_sha256": model_hashes,
            "research_code_sha256": code_hashes, "course_code_sha256": course_hashes,
            "machine": machine,
            "opencv": cv2.__version__,
            "image_sha256": input_hashes, "scene_kinds": scene_kinds,
            "image_shapes": [list(frame.shape) for frame in frames],
            "gallery_sizes": list(sizes),
            "templates_per_identity": templates_per_identity,
            "iterations": iterations,
            "warmup": warmup,
            "seed": seed,
            "results": results,
            "raw_stage_samples": raw_samples, "isolated_cached_retrieval": retrieval,
            "scene_order": "single then multiple; ascending gallery sizes; one Python worker",
            "limitations": ["图库使用合成单位向量，只能评估检索规模/时延，不能评估识别准确率",
                            "FPS是单Python工作线程的处理吞吐估计，原生库线程数另列；不含相机、UI、解码或轨迹投票",
                            "同一静态图重复计时，只描述该工作负载；重复拼接布局不是真实多人场景",
                            "vector_payload_bytes只是单个float32矩阵的大小，不是总内存/RSS",
                            "质量拒绝的脸不进入对齐/特征/匹配，须同时看usable_faces_per_frame"],
        }


def write_benchmark_results(report: dict, output_dir: str | Path) -> Path:
    """Export raw timings, machine metadata, tabular stages and a scale curve."""

    destination = Path(output_dir).resolve()
    with new_result_directory(destination) as staging:
        _write_json(staging / "metrics.json", report)
        rows = []
        for size, scenes in report["results"].items():
            for scene, result in scenes.items():
                for stage, summary in result.items():
                    if not stage.endswith("_ms"):
                        continue
                    rows.append({"gallery_identities": size, "scene": scene,
                                 "scene_kind": report["scene_kinds"][scene], "stage": stage,
                                 "p50_ms": summary["p50"], "p95_ms": summary["p95"],
                                 "detected_faces": json.dumps(result["detected_faces_per_frame"]),
                                 "usable_faces": json.dumps(result["usable_faces_per_frame"])})
        _csv(staging / "metrics.csv", rows, list(rows[0]))
        canvas = np.full((600, 900, 3), 250, np.uint8)
        cv2.putText(canvas, "Cached exact retrieval / SYNTHETIC gallery", (55, 50), cv2.FONT_HERSHEY_SIMPLEX, .75, (40, 40, 40), 2)
        sizes = report["gallery_sizes"]
        values = report["isolated_cached_retrieval"]
        ceiling = max(values[str(size)]["p95_ms"] for size in sizes) * 1.15 or 1.0
        denominator = max(np.log10(sizes[-1]) - np.log10(sizes[0]), 1)
        for tick in np.linspace(0, ceiling, 6):
            y = round(500 - 370 * tick / ceiling)
            cv2.line(canvas, (90, y), (815, y), (215, 215, 215), 1)
            cv2.putText(canvas, f"{tick:.2f}", (20, y + 4), cv2.FONT_HERSHEY_SIMPLEX, .4, (50, 50, 50), 1)
        for field, color, legend_x in (("p50_ms", (50, 120, 20), 150), ("p95_ms", (160, 60, 80), 450)):
            points = []
            for size in sizes:
                x = round(90 + 725 * (np.log10(size) - np.log10(sizes[0])) / denominator)
                y = round(500 - 370 * values[str(size)][field] / ceiling)
                points.append((x, y))
                cv2.circle(canvas, (x, y), 4, color, -1)
                cv2.putText(canvas, str(size), (x - 15, 525), cv2.FONT_HERSHEY_SIMPLEX, .4, (40, 40, 40), 1)
            cv2.polylines(canvas, [np.asarray(points, np.int32)], False, color, 2)
            cv2.putText(canvas, field, (legend_x, 565), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
        cv2.putText(canvas, "ms / query", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, .45, (40, 40, 40), 1)
        cv2.putText(canvas, "identities (log scale)", (590, 565), cv2.FONT_HERSHEY_SIMPLEX, .45, (40, 40, 40), 1)
        _save_png(staging / "scaling_curve.png", canvas)
        _write_json(staging / "artifacts.json", {"schema": "benchmark-artifacts-v1", "sha256": artifact_hashes(staging)})
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="YuNet/SFace/SQLite分阶段CPU基准（非准确率）")
    parser.add_argument("--single-image", required=True)
    scenes = parser.add_mutually_exclusive_group()
    scenes.add_argument("--multiple-image", help="真实静态多人图；不是视频精度评估")
    scenes.add_argument("--repeat-single", type=int, default=0, help="2/3/4张相同单人图的合成布局，只测吞吐")
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument("--output", help="兼容旧用法：新建结果JSON")
    outputs.add_argument("--output-dir", help="新建JSON/CSV/曲线及校验表结果目录")
    parser.add_argument("--config")
    parser.add_argument("--gallery-sizes", default="10,100,1000")
    parser.add_argument("--templates-per-identity", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opencv-threads", type=int, default=1)
    args = parser.parse_args()
    try:
        if Path(args.output or args.output_dir).exists():
            raise FileExistsError("基准输出已存在，请换新的run ID")
        report = run_benchmark(
            args.single_image, args.multiple_image, config_path=args.config,
            gallery_sizes=tuple(int(item) for item in args.gallery_sizes.split(",")),
            templates_per_identity=args.templates_per_identity,
            iterations=args.iterations, warmup=args.warmup, seed=args.seed,
            repeat_single=args.repeat_single, opencv_threads=args.opencv_threads,
        )
        output = write_benchmark_results(report, args.output_dir) if args.output_dir else write_report(report, args.output)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"benchmark未完成：{exc}\n")
    print(json.dumps({"output": str(output), "gallery_sizes": report["gallery_sizes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
