"""Bind a declared primary comparison to a matrix before any test evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def read_primary_protocol(
    path: str | Path, *, budgets: tuple[int, ...], seeds: tuple[int, ...],
    margins: tuple[float, ...], target_fpir: float, methods: tuple[str, ...],
    aggregations: tuple[str, ...],
) -> dict:
    """Validate the fixed protocol's main comparison and any declared grids.

    The spec is data, not executable instructions. Binding its hash prevents
    accidental protocol drift; it cannot prove historical preregistration.
    """

    content = Path(path).read_bytes()
    raw = json.loads(content)
    if not isinstance(raw, dict) or not isinstance(raw.get("id"), str) or not raw["id"].strip():
        raise ValueError("协议必须具有非空id")
    experiment = raw.get("experiment", raw)
    if not isinstance(experiment, dict):
        raise ValueError("协议experiment必须是对象")
    primary = experiment.get("primary_comparison")
    if not isinstance(primary, dict):
        raise ValueError("协议缺少primary_comparison，不能事后挑选最佳组合")
    budget, seed = primary.get("budget"), primary.get("random_seed")
    if type(budget) is not int or budget not in budgets:
        raise ValueError("实验预算没有覆盖协议主比较")
    if type(seed) is not int or seed != seeds[0]:
        raise ValueError("主比较seed必须等于矩阵非随机方法的首个种子")
    if primary.get("selection") not in methods or primary.get("comparator") not in methods:
        raise ValueError("协议主比较的选样方法无效")
    if primary["selection"] == primary["comparator"] or primary.get("aggregation") not in aggregations:
        raise ValueError("协议必须比较不同选样方法，且聚合方法须受支持")
    if primary.get("metric") not in {
        "known correct identification / all known attempts",
        "test known correct identification / all known attempts",
        "dir_known_correct_all_attempts",
    }:
        raise ValueError("当前主比较只支持全部已知尝试的正确接受率，不能静默替换协议指标")
    target = experiment.get("empirical_validation_fpir_target", primary.get("empirical_validation_fpir_target"))
    if type(target) not in (int, float) or not math.isfinite(target) or abs(target - target_fpir) > 1e-12:
        raise ValueError("实验FPIR目标与固定协议不一致")
    for field, actual in (("budgets", budgets), ("random_baseline_seeds", seeds), ("margin_candidates", margins)):
        declared = experiment.get(field)
        if declared is not None and declared != list(actual):
            raise ValueError(f"实验{field}与固定协议不一致")
    label = primary["aggregation"]
    return {
        "status": "specified_before_test", "protocol_id": raw["id"].strip(),
        "protocol_sha256": hashlib.sha256(content).hexdigest(),
        "baseline_variant": f"k{budget}__{primary['comparator']}__s{seed}__{label}",
        "challenger_variant": f"k{budget}__{primary['selection']}__s{seed}__{label}",
        "primary_metric": "dir_known_correct_all_attempts",
        "safety_metric": "fpir_unknown_usable_faces",
        "note": "本次计算前绑定的主比较；不是第三方预注册，也不能消除已有数据筛选偏差",
    }
