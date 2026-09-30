"""Pinned SFace/ArcFace public 1:1 comparison; freeze both before reading test."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

from ._baseline import PROJECT, FaceComparisonSystem, StorageConfig, load_config
from .arcface import ArcFaceEmbedder
from .evaluation.artifacts import RunJournal, artifact_hashes, new_result_directory
from .evaluation.pairs import score_pair_manifest
from .evaluation.reporting import _axes, _canvas, _csv, _roc_curve, _save_png, _write_json
from .evaluation.verification_calibration import calibrate_verification
from .experiment import _sha256_file, _source_hashes
from .verification import _public_split
from face_compare_system.face_compare.deep_engine import SFaceExtractor


class TimedEmbedder:
    """Timing only; preserve alignment, vectors, signature and distance exactly."""
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.dimension, self.signature, self.label = wrapped.dimension, wrapped.signature, wrapped.label
        self.times_ms = []

    def align(self, frame, box):
        return self.wrapped.align(frame, box)

    def extract(self, aligned):
        started = time.perf_counter_ns()
        result = self.wrapped.extract(aligned)
        self.times_ms.append((time.perf_counter_ns() - started) / 1e6)
        return result

    def distance(self, left, right):
        return self.wrapped.distance(left, right)


def paired_model_counts(baseline: dict, challenger: dict, left_tau: float, right_tau: float) -> dict:
    left, right = baseline["records"], challenger["records"]
    if len(left) != len(right) or any(
        (a["left_sha256"], a["right_sha256"], a["genuine"]) !=
        (b["left_sha256"], b["right_sha256"], b["genuine"])
        for a, b in zip(left, right, strict=True)
    ):
        raise ValueError("模型对照必须使用同一批且顺序一致的配对")
    result = {"genuine_accept_gained": 0, "genuine_accept_lost": 0,
              "impostor_false_accept_added": 0, "impostor_false_accept_prevented": 0,
              "both_usable": 0, "baseline_only_usable": 0, "challenger_only_usable": 0,
              "neither_usable": 0}
    for a, b in zip(left, right, strict=True):
        usable_a, usable_b = a["status"] == "scored", b["status"] == "scored"
        key = ("both_usable" if usable_a and usable_b else "baseline_only_usable" if usable_a else
               "challenger_only_usable" if usable_b else "neither_usable")
        result[key] += 1
        accepted_a = usable_a and a["distance"] <= left_tau
        accepted_b = usable_b and b["distance"] <= right_tau
        if a["genuine"]:
            result["genuine_accept_gained"] += accepted_b and not accepted_a
            result["genuine_accept_lost"] += accepted_a and not accepted_b
        else:
            result["impostor_false_accept_added"] += accepted_b and not accepted_a
            result["impostor_false_accept_prevented"] += accepted_a and not accepted_b
    return result


def run_model_comparison(validation_manifest, test_manifest, arcface_model, *,
                         noncommercial_research=False, config_path=None, target_far=.05,
                         require_session_ids=True, on_freeze=None, on_test_open=None) -> dict:
    if noncommercial_research is not True:
        raise ValueError("ArcFace对照须显式确认非商业研究用途")
    config_path = Path(config_path).resolve() if config_path else PROJECT / "config.json"
    config = load_config(config_path)
    if config.engine.backend != "sface" or config.storage.backend != "sqlite":
        raise ValueError("模型对照要求保留YuNet质量前端及独立SQLite")
    model_dir = (PROJECT / config.engine.model_directory).resolve()
    model_paths = {"yunet": model_dir / "face_detection_yunet_2023mar.onnx",
                   "sface": model_dir / "face_recognition_sface_2021dec.onnx",
                   "arcface": Path(arcface_model).resolve()}

    def snapshot():
        return {"manifest_sha256": {"validation": _sha256_file(Path(validation_manifest)),
                                     "test": _sha256_file(Path(test_manifest))},
                "config_sha256": _sha256_file(config_path),
                "model_sha256": {name: _sha256_file(path) for name, path in model_paths.items()},
                "research_code_sha256": _source_hashes(Path(__file__).parent),
                "course_code_sha256": _source_hashes(PROJECT / "face_compare")}

    provenance = snapshot()
    raw_embedders = {"sface": SFaceExtractor(model_dir),
                     "arcface": ArcFaceEmbedder(arcface_model, noncommercial_research=True)}
    embedders = {name: TimedEmbedder(model) for name, model in raw_embedders.items()}
    # Deterministic non-biometric warm-up; never warm on test pixels.
    for embedder in raw_embedders.values():
        for _ in range(5):
            embedder.extract(np.full((112, 112, 3), 127, np.uint8))
    systems, validation, test, points = {}, {}, {}, {}
    with tempfile.TemporaryDirectory(prefix="face-model-comparison-") as temporary, ExitStack() as cleanup:
        for name, embedder in embedders.items():
            isolated = replace(config, engine=replace(config.engine, model_directory=str(model_dir)),
                               storage=StorageConfig(str(Path(temporary) / name),
                                                     str(Path(temporary) / f"{name}.jsonl"), False, "sqlite"))
            systems[name] = FaceComparisonSystem(isolated, temporary, embedder=embedder)
            cleanup.callback(systems[name].database.close)
            validation[name] = score_pair_manifest(systems[name], validation_manifest, split="validation",
                                                   require_session_ids=require_session_ids)
            points[name] = calibrate_verification(validation[name]["pairs"], split="validation", target_far=target_far)
        protocol = validation["sface"]["protocol_spec"]
        if protocol is not None and (not isinstance(protocol, dict) or protocol.get("target_empirical_far") != target_far):
            raise ValueError("运行FAR目标与固定配对协议不一致")
        if snapshot() != provenance:
            raise ValueError("冻结前输入或代码发生变化")
        receipt = {"schema": "model-comparison-validation-freeze-v1", **provenance,
                   "models": {name: {"signature": embedder.signature, "dimension": embedder.dimension,
                                      "bytes": model_paths[name].stat().st_size}
                              for name, embedder in embedders.items()},
                   "target_empirical_far": target_far, "require_session_ids": require_session_ids,
                   "calibration": {name: asdict(point) for name, point in points.items()},
                   "validation_image_hashes": sorted(validation["sface"]["image_hashes"]),
                   "validation_file_hashes": sorted(validation["sface"]["input_files"].values()),
                   "comparison": "same pairs; YuNet/quality unchanged; model-specific alignment/preprocessing; independently calibrated",
                   "test_reuse": "existing public protocol test has been used; exploratory model comparison, not a new blind test"}
        if on_freeze:
            on_freeze(json.loads(json.dumps(receipt, allow_nan=False)))
        if on_test_open:
            on_test_open()
        if snapshot() != provenance:
            raise ValueError("冻结后输入或代码发生变化")
        for name, system in systems.items():
            earlier = validation[name]
            test[name] = score_pair_manifest(system, test_manifest, split="test", require_session_ids=require_session_ids,
                                             forbidden_hashes=earlier["image_hashes"], forbidden_aligned_hashes=earlier["aligned_hashes"],
                                             forbidden_sessions=earlier["sessions"], forbidden_subjects=earlier["subject_ids"])
            if test[name]["protocol_spec"] != protocol:
                raise ValueError("验证/测试配对协议不一致")
        if snapshot() != provenance or any(
            _sha256_file(path) != digest for group in (validation, test) for scored in group.values()
            for path, digest in scored["input_files"].items()
        ):
            raise ValueError("运行期间输入或代码发生变化")
        variants = {}
        for name, embedder in embedders.items():
            times = embedder.times_ms
            variants[name] = {"threshold": points[name].threshold,
                              "validation": _public_split(validation[name], points[name].threshold),
                              "test": _public_split(test[name], points[name].threshold),
                              "extraction_timing_ms": {"count": len(times), "raw": times,
                                                       "p50": float(np.percentile(times, 50)),
                                                       "p95": float(np.percentile(times, 95))},
                              "model": receipt["models"][name]}
        return {"protocol": "sface-arcface-verification-comparison-v1", "frozen_validation": receipt,
                "variants": variants,
                "paired_arcface_vs_sface": paired_model_counts(test["sface"], test["arcface"], points["sface"].threshold, points["arcface"].threshold),
                "arcface_provenance": raw_embedders["arcface"].metadata(),
                "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                "opencv": cv2.__version__, "numpy": np.__version__,
                                "opencv_reported_threads": cv2.getNumThreads()},
                "limitations": ["公开裁剪图、缺会话，两个模型均不能给低FAR保证；test已经使用，不是新盲测",
                                "只比较固定模型完整特征路径，不能把不同训练数据/容量/对齐差异归因于损失函数本身",
                                "计时仅BGR对齐图→预处理/特征/L2，排除检测、对齐、检索、IO和UI；每模型先做5次合成暖机，随后按模型顺序处理固定数据",
                                "逐图时延是描述性测量，非交替配对速度基准，不能宣称整个应用同倍加速",
                                "Wilson区间仅为描述性近似；图像跨genuine/impostor复用，预训练重叠与身份标签误差未知",
                                "保留所有采集失败，不以可用样本TAR代替全尝试比例；不自动更新线上模型或阈值"]}


def write_model_comparison(report, output):
    json.dumps(report, allow_nan=False)
    destination = Path(output).resolve()
    with new_result_directory(destination) as staging:
        _write_json(staging / "metrics.json", report)
        receipt = report["frozen_validation"]
        _write_json(staging / "experiment_config.json", {"frozen_validation": receipt,
                    "environment": report["environment"], "limitations": report["limitations"],
                    "arcface_provenance": report["arcface_provenance"]})
        _write_json(staging / "frozen_validation.json", receipt)
        metrics, sweeps, pairs, timings = [], [], [], []
        for name, variant in report["variants"].items():
            for split in ("validation", "test"):
                scored = variant[split]
                metrics.append({"model": name, "split": split, "total_pairs": scored["total_pairs"],
                                "usable_pairs": scored["usable_pairs"],
                                "genuine_failures": scored["acquisition_failures"]["genuine"],
                                "impostor_failures": scored["acquisition_failures"]["impostor"],
                                "tar_all_attempts": scored["all_attempts"]["tar"],
                                **((scored["report"] or {}).get("operating_point") or {})})
                pairs.extend({"model": name, "split": split, **row} for row in scored["records"])
            sweep = receipt["calibration"][name]["sweep"]
            sweeps.extend({"model": name, **row} for row in sweep)
            timings.extend({"model": name, "index": i, "extraction_ms": value}
                           for i, value in enumerate(variant["extraction_timing_ms"]["raw"]))
            _roc_curve(staging / f"roc_{name}.png", variant["test"])
        _csv(staging / "metrics.csv", metrics, list(metrics[0]))
        _csv(staging / "pairs.csv", pairs, list(pairs[0]))
        _csv(staging / "threshold_selection.csv", sweeps, list(sweeps[0]))
        _csv(staging / "extraction_timing.csv", timings, list(timings[0]))
        canvas = _canvas("Model-specific threshold curves (validation only)")
        left, right, top, bottom = _axes(canvas, "threshold", "rate")
        colors = {("sface", "tar"): (50, 120, 20), ("sface", "far"): (160, 60, 80),
                  ("arcface", "tar"): (40, 160, 190), ("arcface", "far"): (170, 100, 20)}
        for idx, ((name, field), color) in enumerate(colors.items()):
            points = [(left, bottom)]
            for row in receipt["calibration"][name]["sweep"]:
                x = round(left + (right - left) * row["threshold"])
                y = round(bottom - (bottom - top) * row[field])
                points.extend(((x, points[-1][1]), (x, y)))
            points.append((right, points[-1][1]))
            cv2.polylines(canvas, [np.array(points, np.int32)], False, color, 2, cv2.LINE_AA)
            cv2.putText(canvas, f"{name} {field.upper()}", (90 + idx * 195, 575), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        _save_png(staging / "threshold_curve.png", canvas)
        _write_json(staging / "artifacts.json", {"schema": "model-comparison-artifacts-v1", "sha256": artifact_hashes(staging)})
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--arcface-model", required=True)
    parser.add_argument("--noncommercial-research", action="store_true")
    parser.add_argument("--allow-missing-sessions", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--target-far", type=float, default=.05)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with RunJournal(args.output, kind="model-comparison") as journal:
        report = run_model_comparison(args.validation, args.test, args.arcface_model,
                                      noncommercial_research=args.noncommercial_research,
                                      require_session_ids=not args.allow_missing_sessions, config_path=args.config,
                                      target_far=args.target_far, on_freeze=journal.freeze, on_test_open=journal.test_opening)
        output = write_model_comparison(report, args.output)
        journal.completed(output)
    print(json.dumps({"output": str(output), "paired": report["paired_arcface_vs_sface"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
