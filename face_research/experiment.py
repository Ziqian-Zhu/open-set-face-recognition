"""Isolated, label-blind template-selection ablation for the SFace project.

Gallery images determine templates; validation probes determine operating points;
test probes are touched only after those choices are frozen. This module does not
change the GUI, the course database, or the production matching implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from ._baseline import (
    PROJECT,
    FaceComparisonSystem,
    StorageConfig,
    interval,
    load_config,
    summarize,
)
from .manifest import check_manifest_splits, load_image, read_research_manifest
from .selection import (
    COHERENCE_FLOOR,
    COVERAGE_DISTANCE_SCALE,
    DENSITY_SMOOTHING,
    METHODS,
    QUALITY_BONUS,
    GalleryIndex,
    Template,
    build_gallery_index,
    decide,
    quality_proxy,
    rank_with_index,
    select_templates,
)


@dataclass(frozen=True)
class Probe:
    expected: str | None
    condition: str
    status: str
    feature: NDArray[np.float32] | None
    sha256: str
    aligned_sha256: str | None
    preprocessing_ms: float


@dataclass(frozen=True)
class FrozenMethod:
    """All choices made without opening the test manifest."""

    selected: dict[str, list[Template]]
    index: GalleryIndex
    selection_ms: float
    threshold: float
    calibration: dict


def _read_probes(system: FaceComparisonSystem, entries: list[dict]) -> list[Probe]:
    probes = []
    for entry in entries:
        image = load_image(entry)
        start = time.perf_counter()
        observations = system.analyze_frame(image, recognize=False, log_events=False)
        feature = None
        aligned_sha = None
        if not observations:
            status = "no_face"
        elif len(observations) != 1:
            status = "multiple_faces"
        elif not observations[0].quality.accepted:
            status = "quality_rejected"
        else:
            status = "recognized"
            aligned = system.extractor.align(image, observations[0].box)
            aligned_sha = hashlib.sha256(aligned.tobytes()).hexdigest()
            feature = system.extractor.extract(aligned)
        probes.append(
            Probe(
                entry["label"],
                entry["condition"],
                status,
                feature,
                entry["sha256"],
                aligned_sha,
                (time.perf_counter() - start) * 1000,
            )
        )
    return probes


def check_aligned_leakage(
    gallery_hashes: set[str], validation: list[Probe], test: list[Probe]
) -> None:
    """Catch re-exported identical aligned crops in addition to decoded images."""

    val_hashes = {row.aligned_sha256 for row in validation if row.aligned_sha256}
    test_hashes = {row.aligned_sha256 for row in test if row.aligned_sha256}
    if gallery_hashes & (val_hashes | test_hashes) or val_hashes & test_hashes:
        raise ValueError("五点对齐后的人脸裁剪完全重复，拒绝自测/跨集泄漏")
    if len(val_hashes) != sum(row.aligned_sha256 is not None for row in validation):
        raise ValueError("验证集中存在重复的对齐人脸裁剪")
    if len(test_hashes) != sum(row.aligned_sha256 is not None for row in test):
        raise ValueError("测试集中存在重复的对齐人脸裁剪")


def _predictions(
    probes: list[Probe],
    index: GalleryIndex,
    nearest_samples: int,
    *,
    strategy: str = "topk_median",
) -> list[tuple[Probe, list[tuple[float, str]], float]]:
    result = []
    for probe in probes:
        start = time.perf_counter()
        candidates = (
            rank_with_index(index, probe.feature, nearest_samples, strategy=strategy)
            if probe.feature is not None
            else []
        )
        result.append((probe, candidates, (time.perf_counter() - start) * 1000))
    return result


def _records(predictions, threshold: float, margin: float) -> list[dict]:
    return [
        {
            "sha256": probe.sha256,
            "expected": probe.expected,
            "predicted": decide(candidates, threshold, margin),
            "condition": probe.condition,
            "status": probe.status,
            "ms": probe.preprocessing_ms + matching_ms,
        }
        for probe, candidates, matching_ms in predictions
    ]


def calibrate_threshold(
    predictions,
    margin: float,
    target_fpir: float,
) -> tuple[float, dict]:
    """Optimize validation known-correct subject to empirical usable-face FPIR.

    This does not confer a population-level FPIR guarantee. Test labels and
    images are never supplied to this function.
    """

    if not 0 <= target_fpir < 1:
        raise ValueError("target_fpir必须在[0,1)之间")
    usable_unknown = sum(
        probe.expected is None and probe.status == "recognized"
        for probe, _, _ in predictions
    )
    usable_known = sum(
        probe.expected is not None and probe.status == "recognized"
        for probe, _, _ in predictions
    )
    if usable_unknown < 1 or usable_known < 1:
        raise ValueError("验证集需要至少1张可用的已知人脸和1张可用的未知人脸")

    # Acceptance changes only when the threshold crosses a best distance.
    # Sweep those events once: O(N log N), rather than rebuilding N records
    # for each of N candidate thresholds.
    events: dict[float, list[tuple[str | None, str]]] = defaultdict(list)
    for probe, candidates, _ in predictions:
        if probe.status != "recognized" or not candidates:
            continue
        eligible_name = decide(candidates, 1.0, margin)
        if eligible_name is not None:
            distance = float(candidates[0][0])
            if not np.isfinite(distance) or not 0 <= distance <= 1:
                raise ValueError("验证集匹配距离无效")
            events[distance].append((probe.expected, eligible_name))
    correct = misid = false_accept = 0
    best: tuple[tuple[int, int, int, float], float, int] | None = None

    def consider(threshold: float) -> None:
        nonlocal best
        if false_accept / usable_unknown > target_fpir + 1e-12:
            return
        score = (correct, -misid, -false_accept, -threshold)
        if best is None or score > best[0]:
            best = (score, threshold, false_accept)

    if 0.0 not in events:
        consider(0.0)
    for threshold in sorted(events):
        for expected, predicted in events[threshold]:
            if expected is None:
                false_accept += 1
            elif predicted == expected:
                correct += 1
            else:
                misid += 1
        consider(threshold)
        if false_accept / usable_unknown > target_fpir + 1e-12:
            break
    if best is None:
        raise ValueError("验证集无法满足目标未知人员误接受率")
    _, threshold, fp = best
    records = _records(predictions, threshold, margin)
    return threshold, {
        "target_empirical_fpir": target_fpir,
        "validation_usable_unknown": usable_unknown,
        "validation_unknown_false_accept": interval(fp, usable_unknown),
        "summary": summarize(records),
    }


def _condition_summaries(records: list[dict]) -> dict[str, dict]:
    return {
        condition: summarize([row for row in records if row["condition"] == condition])
        for condition in sorted({row["condition"] for row in records})
    }


def paired_comparison(baseline: list[dict], challenger: list[dict]) -> dict:
    """Descriptive paired changes on the *same* test probes, without p-hacking."""

    if len(baseline) != len(challenger) or any(
        first["sha256"] != second["sha256"]
        or first["expected"] != second["expected"]
        or first["status"] != second["status"]
        for first, second in zip(baseline, challenger, strict=True)
    ):
        raise ValueError("配对比较要求完全相同且顺序一致的测试样本")
    known = [
        (a, b)
        for a, b in zip(baseline, challenger, strict=True)
        if a["expected"] is not None
    ]
    unknown = [
        (a, b)
        for a, b in zip(baseline, challenger, strict=True)
        if a["expected"] is None and a["status"] == "recognized"
    ]
    known_wins = sum(
        a["predicted"] != a["expected"] and b["predicted"] == b["expected"]
        for a, b in known
    )
    known_losses = sum(
        a["predicted"] == a["expected"] and b["predicted"] != b["expected"]
        for a, b in known
    )
    unknown_wins = sum(
        a["predicted"] is not None and b["predicted"] is None for a, b in unknown
    )
    unknown_losses = sum(
        a["predicted"] is None and b["predicted"] is not None for a, b in unknown
    )
    return {
        "known_correct_gained": known_wins,
        "known_correct_lost": known_losses,
        "known_accuracy_delta": (
            (known_wins - known_losses) / len(known) if known else None
        ),
        "unknown_false_accepts_prevented": unknown_wins,
        "unknown_false_accepts_added": unknown_losses,
        "usable_unknown_fpir_delta": (
            (unknown_losses - unknown_wins) / len(unknown) if unknown else None
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(directory: Path) -> dict[str, str]:
    """Include nested evaluation/dataset code, not only top-level modules."""

    paths = []
    for folder, subdirs, filenames in os.walk(directory):
        subdirs[:] = sorted(name for name in subdirs if not name.startswith(".") and name not in {
            "tests", "__pycache__", "data", "results"
        })
        paths.extend(Path(folder) / name for name in filenames if name.endswith(".py"))
    return {path.relative_to(directory).as_posix(): _sha256_file(path) for path in sorted(paths)}


def _prepare_gallery(
    system: FaceComparisonSystem, entries: list[dict], quality_config, budget: int
) -> tuple[dict[str, list[Template]], set[str]]:
    """Check every enrollment image in an isolated DB, retaining embeddings only."""

    grouped: dict[str, list[Template]] = defaultdict(list)
    prepared = defaultdict(list)
    aligned_hashes: set[str] = set()
    for entry in entries:
        sample = system.prepare_sample(load_image(entry))
        aligned_sha = hashlib.sha256(sample.crop.tobytes()).hexdigest()
        if aligned_sha in aligned_hashes:
            raise ValueError("gallery含重复的对齐人脸裁剪")
        aligned_hashes.add(aligned_sha)
        grouped[entry["label"]].append(
            Template(
                entry["sha256"],
                sample.feature,
                quality_proxy(sample.quality, quality_config),
            )
        )
        prepared[entry["label"]].append(sample)
    if len(grouped) < 3:
        raise ValueError("为支撑多人开集结论，gallery至少需要3名已知人员")
    if any(len(templates) <= budget for templates in grouped.values()):
        raise ValueError(
            "每名已知人员的录入样本都必须多于模板预算，否则无法公平比较选样策略"
        )
    # The course system's existing identity-consistency and cross-person
    # checks run against a disposable SQLite database, never the real DB.
    for name in prepared:
        system.enroll_samples(name, prepared[name])
    prepared.clear()  # Release aligned image crops before loading probes.
    return dict(grouped), aligned_hashes


def run_experiment(
    validation_manifest: str | Path,
    test_manifest: str | Path,
    *,
    budget: int = 3,
    target_fpir: float = 0.05,
    seed: int = 42,
    config_path: str | Path | None = None,
) -> dict:
    """Return a report; call ``write_report`` explicitly to save it."""

    if budget < 1:
        raise ValueError("每人模板预算必须至少为1")
    if not 0 <= target_fpir < 1:
        raise ValueError("target_fpir必须在[0,1)之间")
    config_path = (
        Path(config_path).resolve() if config_path else PROJECT / "config.json"
    )
    config = load_config(config_path)
    if config.engine.backend != "sface":
        raise ValueError("本实验仅支持YuNet+SFace，不比较不同特征空间")
    if config.storage.backend != "sqlite":
        raise ValueError("本实验要求临时SQLite，绝不访问课程项目的人脸库")
    val_split, val_groups = read_research_manifest(validation_manifest)
    if val_split != "validation":
        raise ValueError("验证清单必须标记split=validation")

    model_dir = (PROJECT / config.engine.model_directory).resolve()
    with tempfile.TemporaryDirectory(
        prefix="face-research-"
    ) as temporary, ExitStack() as cleanup:
        isolated = replace(
            config,
            engine=replace(config.engine, model_directory=str(model_dir)),
            storage=StorageConfig(
                str(Path(temporary) / "data"),
                str(Path(temporary) / "events.jsonl"),
                False,
                "sqlite",
            ),
        )
        system = FaceComparisonSystem(isolated, temporary)
        if hasattr(system.database, "close"):
            cleanup.callback(system.database.close)
        grouped, gallery_aligned_hashes = _prepare_gallery(
            system, val_groups["gallery"], config.quality, budget
        )
        val_probes = _read_probes(system, val_groups["probes"])
        check_aligned_leakage(gallery_aligned_hashes, val_probes, [])
        margin = system.recognition_config.ambiguity_margin
        nearest = system.recognition_config.nearest_samples
        fixed_threshold = system.recognition_config.default_distance_threshold
        frozen: dict[str, FrozenMethod] = {}
        for method in METHODS:
            select_start = time.perf_counter()
            selected = {
                name: select_templates(
                    templates,
                    method,
                    budget,
                    seed=seed,
                    consistency_distance=(1 - config.engine.enrollment_consistency) / 2,
                )
                for name, templates in grouped.items()
            }
            index = build_gallery_index(selected)
            selection_ms = (time.perf_counter() - select_start) * 1000
            val_predictions = _predictions(val_probes, index, nearest)
            threshold, calibration = calibrate_threshold(
                val_predictions, margin, target_fpir
            )
            frozen[method] = FrozenMethod(
                selected, index, selection_ms, threshold, calibration
            )

        # Only now load test images: no test probe may influence selection,
        # threshold choice, or any frozen method parameter.
        test_split, test_groups = read_research_manifest(test_manifest)
        if test_split != "test":
            raise ValueError("测试清单必须标记split=test")
        check_manifest_splits(val_groups, test_groups)
        test_probes = _read_probes(system, test_groups["probes"])
        check_aligned_leakage(gallery_aligned_hashes, val_probes, test_probes)
        result = {}
        test_records_by_method = {}
        for method, choice in frozen.items():
            test_predictions = _predictions(test_probes, choice.index, nearest)
            test_records = _records(test_predictions, choice.threshold, margin)
            test_records_by_method[method] = test_records
            fixed_records = _records(test_predictions, fixed_threshold, margin)
            result[method] = {
                "threshold_from_validation": choice.threshold,
                "validation": choice.calibration,
                "test": {
                    "summary": summarize(test_records),
                    "by_condition": _condition_summaries(test_records),
                },
                "test_at_fixed_course_threshold": summarize(fixed_records),
                "selection_ms_total": choice.selection_ms,
                "selected_template_hashes": {
                    name: [item.key for item in templates]
                    for name, templates in choice.selected.items()
                },
                "selected_template_quality": {
                    name: [item.quality for item in templates]
                    for name, templates in choice.selected.items()
                },
            }
        return {
            "protocol": "frozen-sface-fixed-budget-v1",
            "manifest_validation": str(Path(validation_manifest).resolve()),
            "manifest_test": str(Path(test_manifest).resolve()),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "model_sha256": {
                filename: _sha256_file(model_dir / filename)
                for filename in (
                    "face_detection_yunet_2023mar.onnx",
                    "face_recognition_sface_2021dec.onnx",
                )
            },
            "research_code_sha256": _source_hashes(Path(__file__).parent),
            "course_code_sha256": _source_hashes(PROJECT / "face_compare"),
            "engine": system.engine_label,
            "feature_signature": system.database.feature_signature,
            "opencv": cv2.__version__,
            "gallery_people": len(grouped),
            "gallery_samples": sum(map(len, grouped.values())),
            "per_person_budget": budget,
            "coverage_constants": {
                "distance_scale": COVERAGE_DISTANCE_SCALE,
                "density_smoothing": DENSITY_SMOOTHING,
                "quality_bonus": QUALITY_BONUS,
                "coherence_floor": COHERENCE_FLOOR,
                "consistency_distance": (1 - config.engine.enrollment_consistency) / 2,
            },
            "nearest_samples": nearest,
            "seed": seed,
            "margin": margin,
            "course_fixed_threshold": fixed_threshold,
            "methods": result,
            "paired_vs_first": {
                method: paired_comparison(test_records_by_method["first"], records)
                for method, records in test_records_by_method.items()
                if method != "first"
            },
            "limitations": [
                "静态单人图像实验；不评估摄像头实时性、多人跟踪或活体检测",
                "画质分为手工图像代理指标，不是经校准的识别成功概率",
                "目标FPIR只在验证集经验满足，不构成真实场景风险保证",
                "精确像素/对齐裁剪去重无法发现近重复连拍；采集者须使用独立拍摄批次",
                "所有选样方法共享SFace特征和同一模板预算；测试结果不可反复用来调参",
                "单图延迟仅含检测+画质+特征+Python匹配，不含磁盘读取/哈希、摄像头、多帧投票或UI",
            ],
        }


def write_report(report: dict, path: str | Path) -> Path:
    """Publish a private report atomically, without overwriting prior runs."""

    serialized = (
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        # Linking is an atomic create-only operation on the same filesystem.
        os.link(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination
