"""Probe-side quality-gate ablation with one fixed, production-valid gallery.

The four gates see the same decoded probe pool and extracted SFace vectors.
Only validation determines each gate's open-set operating point; test is opened
after all four choices have been frozen. This never changes the live database.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from ._baseline import PROJECT, FaceComparisonSystem, StorageConfig, load_config
from .evaluation.calibration import FrozenOperatingPoint, calibrate_open_set
from .evaluation.metrics import IdentificationAttempt, identification_report
from .evaluation.reporting import _csv, _write_json
from .evaluation.artifacts import artifact_hashes, new_result_directory
from .experiment import (
    Probe,
    _condition_summaries,
    _predictions,
    _prepare_gallery,
    _records,
    _sha256_file,
    _source_hashes,
    check_aligned_leakage,
)
from .manifest import (
    check_manifest_splits,
    check_session_isolation,
    load_image,
    near_duplicate_warnings,
    read_research_manifest,
)
from .selection import build_gallery_index, select_templates


POLICIES = ("none", "brightness_only", "sharpness_only", "full")
_STRUCTURAL_REASONS = {
    "人脸关键点无效",
    "人脸超出画面，请将完整面部移入画面",
    "人脸裁剪为空",
}


@dataclass(frozen=True)
class RawProbe:
    probe: Probe
    quality: object | None


def policy_accepts(policy: str, quality, config) -> bool:
    """Optical gates only; invalid geometry is rejected before this function."""

    if policy not in POLICIES:
        raise ValueError(f"未知画质策略：{policy}")
    if quality is None:
        return False
    if policy == "none":
        return True
    if policy == "brightness_only":
        return config.min_brightness <= quality.brightness <= config.max_brightness
    if policy == "sharpness_only":
        score = quality.focus_score if quality.focus_score is not None else quality.blur_variance
        threshold = config.min_focus_score if quality.focus_score is not None else config.min_blur_variance
        return score >= threshold
    return quality.accepted


def _read_raw_probes(system: FaceComparisonSystem, entries: list[dict]) -> list[RawProbe]:
    result = []
    for entry in entries:
        image = load_image(entry)
        started = time.perf_counter()
        observations = system.analyze_frame(image, recognize=False, log_events=False)
        feature = aligned_hash = quality = None
        if not observations:
            status = "no_face"
        elif len(observations) != 1:
            status = "multiple_faces"
        else:
            observation = observations[0]
            quality = observation.quality
            if any(reason in _STRUCTURAL_REASONS for reason in quality.reasons):
                status = "invalid_geometry"
            else:
                try:
                    aligned = system.extractor.align(image, observation.box)
                    aligned_hash = hashlib.sha256(aligned.tobytes()).hexdigest()
                    feature = system.extractor.extract(aligned)
                    if (
                        feature is None
                        or not np.isfinite(feature).all()
                        or np.linalg.norm(feature) < 1e-8
                    ):
                        raise ValueError("特征无效")
                    status = "recognized"
                except (TypeError, ValueError, cv2.error):
                    status = "extraction_failed"
        result.append(RawProbe(Probe(
            entry["label"], entry["condition"], status, feature, entry["sha256"],
            aligned_hash, (time.perf_counter() - started) * 1000,
        ), quality))
    return result


def _apply_policy(rows: list[RawProbe], policy: str, config) -> list[Probe]:
    return [
        replace(row.probe, status="quality_rejected", feature=None)
        if row.probe.status == "recognized" and not policy_accepts(policy, row.quality, config)
        else row.probe
        for row in rows
    ]


def _gate_counts(rows: list[Probe]) -> dict:
    return {
        "total": len(rows),
        "known": sum(row.expected is not None for row in rows),
        "unknown": sum(row.expected is None for row in rows),
        "status_counts": dict(sorted(Counter(row.status for row in rows).items())),
        "by_condition": {
            condition: dict(sorted(Counter(row.status for row in rows if row.condition == condition).items()))
            for condition in sorted({row.condition for row in rows})
        },
    }


def paired_quality_comparison(baseline: list[dict], challenger: list[dict]) -> dict:
    """Paired all-attempt effects; status may change because that is the treatment."""

    if len(baseline) != len(challenger) or any(
        a["sha256"] != b["sha256"] or a["expected"] != b["expected"]
        for a, b in zip(baseline, challenger, strict=True)
    ):
        raise ValueError("画质门控配对比较需要同一批且顺序一致的样本")
    known = [(a, b) for a, b in zip(baseline, challenger, strict=True) if a["expected"] is not None]
    unknown = [(a, b) for a, b in zip(baseline, challenger, strict=True) if a["expected"] is None]
    gained = sum(a["predicted"] != a["expected"] and b["predicted"] == b["expected"] for a, b in known)
    lost = sum(a["predicted"] == a["expected"] and b["predicted"] != b["expected"] for a, b in known)
    prevented = sum(a["predicted"] is not None and b["predicted"] is None for a, b in unknown)
    added = sum(a["predicted"] is None and b["predicted"] is not None for a, b in unknown)
    return {
        "known_correct_gained": gained, "known_correct_lost": lost,
        "unknown_false_accepts_prevented": prevented, "unknown_false_accepts_added": added,
        "known_correct_rate_delta_all_attempts": (gained - lost) / len(known) if known else None,
        "fpir_delta_all_unknown_attempts": (added - prevented) / len(unknown) if unknown else None,
    }


def run_quality_ablation(
    validation_manifest: str | Path,
    test_manifest: str | Path,
    *,
    budget: int = 3,
    target_fpir: float = 0.05,
    margins: tuple[float, ...] = (0.0, 0.02, 0.04, 0.06),
    config_path: str | Path | None = None,
    require_session_ids: bool = True,
    on_freeze: Callable[[dict], None] | None = None,
    on_test_open: Callable[[], None] | None = None,
) -> dict:
    """Compare probe gates at a fixed First-K gallery and fixed SFace backend."""

    if type(budget) is not int or budget < 1:
        raise ValueError("budget必须是正整数")
    config_path = Path(config_path).resolve() if config_path else PROJECT / "config.json"
    config = load_config(config_path)
    if config.engine.backend != "sface" or config.storage.backend != "sqlite":
        raise ValueError("画质消融要求YuNet+SFace与隔离的SQLite向量库")
    model_dir = (PROJECT / config.engine.model_directory).resolve()
    def snapshot():
        return {
            "validation_manifest_sha256": _sha256_file(Path(validation_manifest)),
            "config_sha256": _sha256_file(config_path),
            "model_sha256": {name: _sha256_file(model_dir / name) for name in (
                "face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx")},
            "research_code_sha256": _source_hashes(Path(__file__).parent),
            "course_code_sha256": _source_hashes(PROJECT / "face_compare"),
        }
    provenance = snapshot()
    split, validation = read_research_manifest(validation_manifest)
    if split != "validation":
        raise ValueError("验证清单必须标记split=validation")
    if require_session_ids and any(
        row.get("session_id") is None
        for group in ("gallery", "probes")
        for row in validation[group]
    ):
        raise ValueError("严格画质实验要求gallery和validation都有session_id")
    with tempfile.TemporaryDirectory(prefix="face-quality-") as temporary, ExitStack() as cleanup:
        isolated = replace(config,
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
        grouped, gallery_hashes = _prepare_gallery(system, validation["gallery"], config.quality, budget)
        selected = {name: select_templates(templates, "first", budget) for name, templates in grouped.items()}
        index = build_gallery_index(selected)
        anonymous_ids = {name: f"subject_{i:03d}" for i, name in enumerate(sorted(grouped), 1)}
        validation_raw = _read_raw_probes(system, validation["probes"])
        check_aligned_leakage(gallery_hashes, [row.probe for row in validation_raw], [])
        frozen: dict[str, FrozenOperatingPoint | None] = {}
        unavailable = {}
        validation_gates = {}
        for policy in POLICIES:
            probes = _apply_policy(validation_raw, policy, config.quality)
            validation_gates[policy] = _gate_counts(probes)
            predictions = _predictions(
                probes, index, system.recognition_config.nearest_samples
            )
            attempts = [
                IdentificationAttempt(p.expected, p.status, tuple(c))
                for p, c, _ in predictions
            ]
            try:
                frozen[policy] = calibrate_open_set(
                    attempts, split="validation", target_fpir=target_fpir,
                    margins=margins,
                )
            except ValueError as exc:
                insufficient = "至少需要一张可用已知脸和一张可用未知脸" in str(exc)
                infeasible = "无法满足目标未知人员误接受率" in str(exc)
                if not (insufficient or infeasible):
                    raise
                frozen[policy] = None
                unavailable[policy] = str(exc)

        if snapshot() != provenance:
            raise ValueError("画质消融冻结前代码、模型、配置或清单发生变化")
        receipt = {
            "schema": "quality-validation-freeze-v1", **provenance,
            "budget": budget, "selection": "first", "aggregation": "topk_median",
            "nearest_samples": system.recognition_config.nearest_samples,
            "margins": list(margins), "target_empirical_fpir": target_fpir,
            "require_session_ids": require_session_ids,
            "primary_contrast": "full vs none; independently calibrated on validation; descriptive exploration",
            "selected_template_hashes": {
                anonymous_ids[name]: [template.key for template in templates]
                for name, templates in selected.items()
            },
            "validation_input_hashes": {group: [row["sha256"] for row in validation[group]]
                                        for group in ("gallery", "probes")},
            "policies": {
                policy: {"threshold": point.threshold, "margin": point.margin,
                         "validation": point.validation} if point else {
                    "status": "unavailable", "reason": unavailable[policy]
                } for policy, point in frozen.items()
            },
        }
        if on_freeze:
            on_freeze(json.loads(json.dumps(receipt, allow_nan=False)))
        # Validation calibration for every gate has finished before test is opened.
        if on_test_open:
            on_test_open()
        test_manifest_sha256 = _sha256_file(Path(test_manifest))
        test_split, test = read_research_manifest(test_manifest)
        if test_split != "test":
            raise ValueError("测试清单必须标记split=test")
        check_manifest_splits(validation, test)
        check_session_isolation(validation, test, require_session_ids=require_session_ids)
        warnings = [
            {**item, "subject": anonymous_ids[item["subject"]]}
            for item in near_duplicate_warnings(validation, test)
        ]
        test_raw = _read_raw_probes(system, test["probes"])
        check_aligned_leakage(
            gallery_hashes,
            [row.probe for row in validation_raw],
            [row.probe for row in test_raw],
        )
        variants, records_by_policy = {}, {}
        for policy in POLICIES:
            probes = _apply_policy(test_raw, policy, config.quality)
            point = frozen[policy]
            variant = {"validation_gate": validation_gates[policy], "test_gate": _gate_counts(probes)}
            if point is None:
                variant.update(status="unavailable", reason=unavailable[policy], test_identification=None)
            else:
                predictions = _predictions(probes, index, system.recognition_config.nearest_samples)
                attempts = [IdentificationAttempt(p.expected, p.status, tuple(c)) for p, c, _ in predictions]
                records = _records(predictions, point.threshold, point.margin)
                records_by_policy[policy] = records
                variant.update(
                    status="evaluated", threshold_from_validation=point.threshold,
                    margin_from_validation=point.margin, validation_identification=point.validation,
                    threshold_selection=list(point.sweep),
                    test_identification=identification_report(attempts, point.threshold, point.margin),
                    test_by_condition=_condition_summaries(records),
                )
                full_point = frozen["full"]
                variant["test_at_shared_full_gate_validation_point"] = (
                    identification_report(attempts, full_point.threshold, full_point.margin)
                    if full_point is not None else None
                )
            variants[policy] = variant
        if snapshot() != provenance or _sha256_file(Path(test_manifest)) != test_manifest_sha256:
            raise ValueError("运行期间代码、配置或清单发生变化，拒绝发布混合版本结果")
        return {
            "protocol": "fixed-gallery-probe-quality-ablation-v1",
            "scope": "仅比较probe画质门控；gallery始终使用生产完整质量检查，不衡量放宽录入质量门槛的效果",
            "manifest_sha256": {
                "validation": provenance["validation_manifest_sha256"],
                "test": test_manifest_sha256,
            },
            **provenance,
            "frozen_validation": receipt,
            "opencv": cv2.__version__, "feature_signature": system.database.feature_signature,
            "gallery_people": len(grouped), "gallery_samples": sum(map(len, grouped.values())),
            "per_person_budget": budget, "selection": "first", "aggregation": "topk_median",
            "selected_template_hashes": {
                anonymous_ids[name]: [template.key for template in templates]
                for name, templates in selected.items()
            },
            "nearest_samples": system.recognition_config.nearest_samples,
            "target_empirical_fpir_validation_usable_faces": target_fpir,
            "margins": list(margins), "near_duplicate_warnings": warnings,
            "variants": variants,
            "paired_vs_no_filter": {
                policy: paired_quality_comparison(records_by_policy["none"], records)
                for policy, records in records_by_policy.items() if policy != "none"
            } if "none" in records_by_policy else {},
            "limitations": [
                "静态单人probe实验，不评价录入门槛、相机时序、多人视频或活体检测",
                "无过滤仍拒绝检测/关键点/对齐/特征提取失败；不是允许任意坏图通过",
                "accuracy_all_attempts=(已知正确接受+可用未知脸正确拒绝)/全部尝试；采集失败不计正确，且该值受已知/未知比例影响",
                "各策略独立在validation校准，test只按冻结阈值评估；小样本经验FPIR不是人群风险保证",
                "条件分层和配对变化为描述性统计；连拍相关性需要人工复核",
            ],
        }


def _write_quality_results(report: dict, destination: Path) -> None:
    _write_json(destination / "metrics.json", report)
    if "frozen_validation" in report:
        _write_json(destination / "frozen_validation.json", report["frozen_validation"])
    rows = []
    sweep = []
    for policy, variant in report["variants"].items():
        metric = variant["test_identification"] or {}
        rows.append({
            "policy": policy,
            "status": variant["status"],
            "reason": variant.get("reason"),
            "validation_quality_rejected": variant["validation_gate"][
                "status_counts"
            ].get("quality_rejected", 0),
            "test_quality_rejected": variant["test_gate"]["status_counts"].get(
                "quality_rejected", 0
            ),
            "known_attempts": metric.get("known_attempts"),
            "unknown_attempts": metric.get("unknown_attempts"),
            "usable_known_attempts": metric.get("usable_known_attempts"),
            "usable_unknown_attempts": metric.get("usable_unknown_attempts"),
            "known_correct_count": (
                metric.get("dir_known_correct_all_attempts") or {}
            ).get("count"),
            "known_correct_rate_all_attempts": (
                metric.get("dir_known_correct_all_attempts") or {}
            ).get("rate"),
            "unknown_false_accept_count": (
                metric.get("fpir_unknown_all_attempts") or {}
            ).get("count"),
            "fpir_unknown_all_attempts": (
                metric.get("fpir_unknown_all_attempts") or {}
            ).get("rate"),
            "fpir_unknown_usable_faces": (
                metric.get("fpir_unknown_usable_faces") or {}
            ).get("rate"),
            "unknown_correct_rejection_count": (
                metric.get("unknown_correct_rejection_all_attempts") or {}
            ).get("count"),
            "unknown_acquisition_failures": (
                metric.get("unknown_acquisition_failures_all_attempts") or {}
            ).get("count"),
            "known_acquisition_failures": (
                metric.get("known_acquisition_failures_all_attempts") or {}
            ).get("count"),
            "correct_all_attempts": (
                metric.get("accuracy_all_attempts") or {}
            ).get("count"),
            "total_attempts": (
                metric.get("accuracy_all_attempts") or {}
            ).get("total"),
            "accuracy_all_attempts": (
                metric.get("accuracy_all_attempts") or {}
            ).get("rate"),
            "threshold": variant.get("threshold_from_validation"),
            "margin": variant.get("margin_from_validation"),
        })
        sweep.extend({"policy": policy, **point} for point in variant.get("threshold_selection", []))
    _csv(destination / "metrics.csv", rows, list(rows[0]))
    sweep_columns = [
        "policy", "threshold", "margin", "known_correct", "known_rejected",
        "known_misidentified", "known_total", "usable_known", "known_acquisition_failures",
        "unknown_false_accept", "unknown_total", "usable_unknown", "unknown_acquisition_failures",
        "unknown_correct_rejection", "correct_all_attempts", "total_attempts",
        "fpir_usable", "known_correct_rate", "known_reject_rate",
        "accuracy_all_attempts", "FAR", "FRR", "TAR",
    ]
    _csv(destination / "threshold_selection.csv", sweep, sweep_columns)


def write_quality_results(report: dict, output_directory: str | Path) -> Path:
    """Publish a complete, checksummed ablation bundle, never a partial run."""

    json.dumps(report, allow_nan=False)
    destination = Path(output_directory).resolve()
    with new_result_directory(destination) as staging:
        _write_quality_results(report, staging)
        _write_json(staging / "artifacts.json", {
            "schema": "quality-artifacts-v1", "sha256": artifact_hashes(staging),
        })
    return destination
