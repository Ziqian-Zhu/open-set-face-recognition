"""Freeze a template-budget/aggregation experiment grid before opening test data."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import cv2

from ._baseline import PROJECT, FaceComparisonSystem, StorageConfig, load_config, summarize
from .evaluation.calibration import FrozenOperatingPoint, calibrate_open_set
from .evaluation.metrics import IdentificationAttempt, identification_report, verification_at_threshold, verification_report
from .evaluation.pairs import score_pair_manifest
from .evaluation.protocol import read_primary_protocol
from .experiment import (
    _condition_summaries,
    _predictions,
    _prepare_gallery,
    _read_probes,
    _records,
    _sha256_file,
    _source_hashes,
    check_aligned_leakage,
    paired_comparison,
)
from .manifest import check_manifest_splits, check_session_isolation, near_duplicate_warnings, read_research_manifest
from .selection import METHODS, GalleryIndex, Template, build_gallery_index, select_templates


# Label, aggregation strategy, nearest-template count. All-template strategies
# ignore k; the value is retained for a uniform, explicit experiment config.
AGGREGATIONS = (
    ("nearest", "nearest", 1),
    ("mean", "mean", 3),
    ("median", "median", 3),
    ("top2_mean", "topk_mean", 2),
    ("top3_mean", "topk_mean", 3),
    ("top3_median", "topk_median", 3),
)


@dataclass(frozen=True)
class _FrozenVariant:
    index: GalleryIndex
    selected: dict[str, list[Template]]
    selection_ms: float
    strategy: str
    nearest: int
    operating: FrozenOperatingPoint
    margin_controls: dict[str, FrozenOperatingPoint | None]


def _attempts(predictions) -> list[IdentificationAttempt]:
    return [
        IdentificationAttempt(probe.expected, probe.status, tuple(candidates))
        for probe, candidates, _ in predictions
    ]


def _variant_key(budget: int, method: str, seed: int, label: str) -> str:
    return f"k{budget}__{method}__s{seed}__{label}"


def _manifest_sessions(rows: list[dict]) -> set[tuple[str, str]]:
    return {
        ((row["label"] if row["label"] is not None else row["subject_id"]).casefold(),
         row["session_id"].casefold())
        for row in rows if row.get("session_id") is not None
    }


def run_matrix_experiment(
    validation_manifest: str | Path,
    test_manifest: str | Path,
    *,
    budgets: Sequence[int] = (1, 2, 3, 5),
    random_seeds: Sequence[int] = (42, 43, 44),
    target_fpir: float = 0.05,
    margins: Sequence[float] = (0.0, 0.02, 0.04, 0.06),
    config_path: str | Path | None = None,
    require_session_ids: bool = True,
    verification_validation_pairs: str | Path | None = None,
    verification_test_pairs: str | Path | None = None,
    protocol_spec: str | Path | None = None,
    on_freeze: Callable[[dict], None] | None = None,
    on_test_open: Callable[[], None] | None = None,
) -> dict:
    """Run all validation choices before touching any test probe image.

    This remains a static 1:N identification experiment. A separate, labeled
    1:1 pair protocol is needed for verification ROC/TAR@FAR.
    """

    budget_grid = tuple(sorted(set(budgets)))
    seeds = tuple(sorted(set(random_seeds)))
    margin_grid = tuple(sorted(set(margins)))
    if not budget_grid or any(type(value) is not int or value < 1 for value in budget_grid):
        raise ValueError("budgets必须是非空正整数集合")
    if not seeds or any(type(value) is not int for value in seeds):
        raise ValueError("random_seeds必须是非空整数集合")
    primary = read_primary_protocol(
        protocol_spec, budgets=budget_grid, seeds=seeds, margins=margin_grid,
        target_fpir=target_fpir, methods=tuple(METHODS),
        aggregations=tuple(row[0] for row in AGGREGATIONS),
    ) if protocol_spec is not None else {
        "status": "not_specified", "reason": "未绑定主比较协议；各组合只能作探索性描述，不能事后挑最高分声称主结果",
    }
    config_path = Path(config_path).resolve() if config_path else PROJECT / "config.json"
    config = load_config(config_path)
    if config.engine.backend != "sface" or config.storage.backend != "sqlite":
        raise ValueError("矩阵实验要求YuNet+SFace与隔离的SQLite向量库")
    model_dir = (PROJECT / config.engine.model_directory).resolve()

    def source_snapshot() -> dict:
        return {
            "validation_manifest_sha256": _sha256_file(Path(validation_manifest)),
            "config_sha256": _sha256_file(config_path),
            "protocol_spec_sha256": _sha256_file(Path(protocol_spec)) if protocol_spec else None,
            "model_sha256": {
                filename: _sha256_file(model_dir / filename)
                for filename in (
                    "face_detection_yunet_2023mar.onnx",
                    "face_recognition_sface_2021dec.onnx",
                )
            },
            "research_code_sha256": _source_hashes(Path(__file__).parent),
            "course_code_sha256": _source_hashes(PROJECT / "face_compare"),
        }

    provenance = source_snapshot()
    if protocol_spec and provenance["protocol_spec_sha256"] != primary["protocol_sha256"]:
        raise ValueError("协议在读取与快照之间发生变化")
    validation_split, validation = read_research_manifest(validation_manifest)
    if validation_split != "validation":
        raise ValueError("验证清单必须标记split=validation")
    if require_session_ids and any(
        row.get("session_id") is None
        for group in ("gallery", "probes")
        for row in validation[group]
    ):
        raise ValueError("严格实验要求gallery和validation每张图都有session_id")

    with tempfile.TemporaryDirectory(prefix="face-matrix-") as temporary, ExitStack() as cleanup:
        isolated = replace(
            config,
            engine=replace(config.engine, model_directory=str(model_dir)),
            storage=StorageConfig(
                str(Path(temporary) / "data"), str(Path(temporary) / "events.jsonl"), False, "sqlite"
            ),
        )
        system = FaceComparisonSystem(isolated, temporary)
        if hasattr(system.database, "close"):
            cleanup.callback(system.database.close)
        grouped, gallery_hashes = _prepare_gallery(
            system, validation["gallery"], config.quality, max(budget_grid)
        )
        anonymous_ids = {
            name: f"subject_{index:03d}"
            for index, name in enumerate(sorted(grouped), 1)
        }
        validation_probes = _read_probes(system, validation["probes"])
        check_aligned_leakage(gallery_hashes, validation_probes, [])
        validation_pairs = None
        if verification_validation_pairs is not None:
            validation_pairs = score_pair_manifest(
                system, verification_validation_pairs, split="validation",
                forbidden_hashes={row["sha256"] for row in validation["gallery"]},
                forbidden_aligned_hashes=gallery_hashes,
                forbidden_sessions=_manifest_sessions(validation["gallery"]),
            )
        pair_metric_cache = {}
        frozen: dict[str, _FrozenVariant] = {}
        variant_meta: dict[str, dict] = {}
        for budget in budget_grid:
            for method in METHODS:
                method_seeds = seeds if method == "random" else seeds[:1]
                for seed in method_seeds:
                    started = time.perf_counter()
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
                    selection_ms = (time.perf_counter() - started) * 1000
                    for label, strategy, nearest in AGGREGATIONS:
                        predictions = _predictions(
                            validation_probes, index, nearest, strategy=strategy
                        )
                        validation_attempts = _attempts(predictions)
                        operating = calibrate_open_set(
                            validation_attempts,
                            split="validation",
                            target_fpir=target_fpir,
                            margins=margin_grid,
                        )
                        if validation_pairs is not None and validation_pairs["pairs"]:
                            enriched = []
                            for row in operating.sweep:
                                threshold = row["threshold"]
                                if threshold not in pair_metric_cache:
                                    pair_metric_cache[threshold] = verification_at_threshold(
                                        validation_pairs["pairs"], threshold
                                    )
                                pair_metric = pair_metric_cache[threshold]
                                enriched.append({**row, "FAR": pair_metric["far"],
                                                 "FRR": pair_metric["frr"], "TAR": pair_metric["tar"]})
                            operating = replace(operating, sweep=tuple(enriched))
                        controls = {}
                        for control_name, control_margin in (
                            ("threshold_only", 0.0),
                            ("threshold_plus_fixed_margin", config.engine.cosine_margin / 2),
                        ):
                            try:
                                controls[control_name] = calibrate_open_set(
                                    validation_attempts, split="validation",
                                    target_fpir=target_fpir, margins=(control_margin,),
                                )
                            except ValueError as exc:
                                if "无法满足目标未知人员误接受率" not in str(exc):
                                    raise
                                controls[control_name] = None
                        key = _variant_key(budget, method, seed, label)
                        frozen[key] = _FrozenVariant(
                            index, selected, selection_ms, strategy, nearest, operating, controls
                        )
                        variant_meta[key] = {
                            "budget": budget,
                            "selection": method,
                            "seed": seed,
                            "aggregation": label,
                            "aggregation_strategy": strategy,
                            "nearest_samples": nearest,
                        }

        if source_snapshot() != provenance:
            raise ValueError("冻结前配置、模型、代码或validation清单发生变化，请使用新的run ID")
        freeze = {
            "schema": "validation-freeze-v1",
            **provenance,
            "opencv": cv2.__version__,
            "feature_signature": system.database.feature_signature,
            "budgets": list(budget_grid),
            "random_seeds": list(seeds),
            "margin_grid": list(margin_grid),
            "target_empirical_fpir": target_fpir,
            "require_session_ids": require_session_ids,
            "primary_comparison": primary,
            "fixed_engineering_operating_point": {
                "threshold": (1 - config.engine.cosine_threshold) / 2,
                "margin": config.engine.cosine_margin / 2,
                "source": "unchanged_production_config_not_test_tuned",
            },
            "validation_input_hashes": {
                group: [row["sha256"] for row in validation[group]]
                for group in ("gallery", "probes")
            },
            "verification_validation_manifest_sha256": (
                validation_pairs["manifest_sha256"] if validation_pairs else None
            ),
            "variants": {
                key: {
                    **variant_meta[key],
                    "threshold": item.operating.threshold,
                    "margin": item.operating.margin,
                    "validation": item.operating.validation,
                    "selected_template_hashes": {
                        anonymous_ids[name]: [template.key for template in templates]
                        for name, templates in item.selected.items()
                    },
                    "margin_controls": {
                        name: {"threshold": point.threshold, "margin": point.margin} if point else None
                        for name, point in item.margin_controls.items()
                    },
                } for key, item in frozen.items()
            },
            "note": "冻结时尚未读取本次test清单；不能证明采集批次真实独立或研究者从未看过测试数据",
        }
        freeze_bytes = json.dumps(freeze, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        freeze_sha256 = hashlib.sha256(freeze_bytes).hexdigest()
        if on_freeze is not None:
            # A callback only sees a detached, JSON-safe receipt, not live indices.
            on_freeze(json.loads(freeze_bytes))

        # This line is deliberately after *all* gallery selection and validation
        # calibration. Test labels cannot affect a frozen method or margin.
        if on_test_open is not None:
            on_test_open()
        test_manifest_sha256 = _sha256_file(Path(test_manifest))
        test_split, test = read_research_manifest(test_manifest)
        if test_split != "test":
            raise ValueError("测试清单必须标记split=test")
        check_manifest_splits(validation, test)
        check_session_isolation(validation, test, require_session_ids=require_session_ids)
        if validation_pairs is not None and validation_pairs["sessions"] & _manifest_sessions(test["probes"]):
            raise ValueError("validation验证对与test probes复用同一人员采集批次，拒绝会话级泄漏")
        duplicate_warnings = [
            {**item, "subject": anonymous_ids[item["subject"]]}
            for item in near_duplicate_warnings(validation, test)
        ]
        test_probes = _read_probes(system, test["probes"])
        check_aligned_leakage(gallery_hashes, validation_probes, test_probes)
        if validation_pairs is not None:
            if validation_pairs["image_hashes"] & {row["sha256"] for row in test["probes"]}:
                raise ValueError("validation验证对与test probes重复，拒绝数据泄漏")
            if validation_pairs["aligned_hashes"] & {
                row.aligned_sha256 for row in test_probes if row.aligned_sha256
            }:
                raise ValueError("validation验证对与test对齐人脸重复，拒绝数据泄漏")
        verification = {
            "status": "not_evaluated",
            "reason": "未提供独立的1:1 genuine/impostor pair协议；不可把FPIR冒充FAR或生成ROC",
        }
        scored_pairs = []
        if verification_test_pairs is not None:
            scored = score_pair_manifest(
                system,
                verification_test_pairs,
                forbidden_hashes={
                    row["sha256"] for row in (*validation["gallery"], *validation["probes"])
                } | (validation_pairs["image_hashes"] if validation_pairs else set()),
                forbidden_aligned_hashes=gallery_hashes | {
                    row.aligned_sha256 for row in validation_probes if row.aligned_sha256
                } | (validation_pairs["aligned_hashes"] if validation_pairs else set()),
                forbidden_sessions=_manifest_sessions(
                    [*validation["gallery"], *validation["probes"]]
                ) | (validation_pairs["sessions"] if validation_pairs else set()),
            )
            scored_pairs = scored["pairs"]
            verification = {
                "status": "evaluated" if scored["pairs"] else "insufficient samples",
                "manifest_sha256": scored["manifest_sha256"],
                "total_pairs": scored["total_pairs"],
                "usable_pairs": scored["usable_pairs"],
                "acquisition_failures": scored["acquisition_failures"],
                "note": scored["note"],
                "fixed_reference_threshold": (1 - config.engine.cosine_threshold) / 2,
                "report": verification_report(
                    scored["pairs"], (1 - config.engine.cosine_threshold) / 2
                ) if scored["pairs"] else None,
            }
        results = {}
        records_by_key = {}
        for key, item in frozen.items():
            predictions = _predictions(test_probes, item.index, item.nearest, strategy=item.strategy)
            records = _records(predictions, item.operating.threshold, item.operating.margin)
            records_by_key[key] = records
            controls = {}
            control_records = {}
            for name, point in item.margin_controls.items():
                if point is None:
                    controls[name] = {
                        "status": "unavailable",
                        "reason": "验证集在此margin下无法满足预先设定的目标FPIR",
                    }
                    continue
                attempt_rows = _attempts(predictions)
                control_records[name] = _records(predictions, point.threshold, point.margin)
                controls[name] = {
                    "threshold_from_validation": point.threshold,
                    "margin_from_validation": point.margin,
                    "validation": point.validation,
                    "test": identification_report(attempt_rows, point.threshold, point.margin),
                }
            controls["paired_fixed_margin_vs_threshold_only"] = (
                paired_comparison(
                    control_records["threshold_only"],
                    control_records["threshold_plus_fixed_margin"],
                ) if len(control_records) == 2 else {
                    "status": "unavailable", "reason": "至少一个对照无法达到验证集目标FPIR"
                }
            )
            results[key] = {
                **variant_meta[key],
                "threshold_from_validation": item.operating.threshold,
                "margin_from_validation": item.operating.margin,
                "validation": item.operating.validation,
                "threshold_selection": list(item.operating.sweep),
                "margin_ablation": controls,
                "test": {
                    "identification": identification_report(
                        _attempts(predictions), item.operating.threshold, item.operating.margin
                    ),
                    "summary": summarize(records),
                    "by_condition": _condition_summaries(records),
                    "verification_at_frozen_threshold": verification_at_threshold(
                        scored_pairs, item.operating.threshold
                    ) if scored_pairs else None,
                    "fixed_engineering_operating_point": identification_report(
                        _attempts(predictions), (1 - config.engine.cosine_threshold) / 2,
                        config.engine.cosine_margin / 2,
                    ),
                },
                "selection_ms_total": item.selection_ms,
                "selected_template_hashes": {
                    anonymous_ids[name]: [template.key for template in templates]
                    for name, templates in item.selected.items()
                },
            }
        for key, result in results.items():
            meta = variant_meta[key]
            baseline = _variant_key(meta["budget"], "first", seeds[0], meta["aggregation"])
            if key != baseline:
                result["paired_vs_first"] = paired_comparison(
                    records_by_key[baseline], records_by_key[key]
                )
        if source_snapshot() != provenance or _sha256_file(Path(test_manifest)) != test_manifest_sha256:
            raise ValueError("运行期间配置、模型、代码或数据清单发生变化，拒绝发布混合版本结果")
        primary_result = dict(primary)
        if primary["status"] == "specified_before_test":
            baseline = results[primary["baseline_variant"]]["test"]["identification"]
            challenger = results[primary["challenger_variant"]]["test"]["identification"]
            primary_result.update({
                "baseline": baseline,
                "challenger": challenger,
                "paired": paired_comparison(records_by_key[primary["baseline_variant"]],
                                            records_by_key[primary["challenger_variant"]]),
                "known_correct_delta": (challenger["dir_known_correct_all_attempts"]["count"]
                                        - baseline["dir_known_correct_all_attempts"]["count"]),
                "interpretation": "仅描述固定主比较及分母；不以正差值自动宣称显著提升或低误接保证",
            })
        return {
            "protocol": "frozen-sface-budget-aggregation-matrix-v1",
            **provenance,
            "test_manifest_sha256": test_manifest_sha256,
            "frozen_validation": freeze,
            "frozen_validation_sha256": freeze_sha256,
            "primary_comparison": primary_result,
            "opencv": cv2.__version__,
            "feature_signature": system.database.feature_signature,
            "gallery_people": len(grouped),
            "gallery_samples": sum(map(len, grouped.values())),
            "validation_probes": len(validation_probes),
            "test_probes": len(test_probes),
            "budgets": list(budget_grid),
            "random_seeds": list(seeds),
            "margin_grid": list(margin_grid),
            "target_empirical_fpir": target_fpir,
            "require_session_ids": require_session_ids,
            "verification": verification,
            "verification_validation": {
                "status": ("evaluated" if validation_pairs["pairs"] else "insufficient samples")
                          if validation_pairs else "not_evaluated",
                "manifest_sha256": validation_pairs["manifest_sha256"] if validation_pairs else None,
                "total_pairs": validation_pairs["total_pairs"] if validation_pairs else 0,
                "usable_pairs": validation_pairs["usable_pairs"] if validation_pairs else 0,
                "acquisition_failures": validation_pairs["acquisition_failures"] if validation_pairs else None,
                "note": "验证对仅为阈值扫描提供verification列；当前1:N工作点仍按validation FPIR/已知正确率选择",
            },
            "near_duplicate_warnings": duplicate_warnings,
            "variants": results,
            "limitations": [
                "静态单脸1:N实验；多脸轨迹须另用带真值视频评估",
                "经验验证集FPIR不构成人群风险保证；低FAR指标需足量独立pair",
                "近重复连拍或同人换ID不能仅靠精确哈希自动发现",
                "测试结果不得反复用于选择K、聚合、margin网格或coverage常数",
            ],
        }
