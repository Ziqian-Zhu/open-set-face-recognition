"""Descriptive paired bootstrap over subject-connected components, not pairs.

Subjects shared across genuine/impostor pairs stay in the same resampling unit.
Missing capture sessions or source-label errors are NOT repaired by resampling.
"""

from __future__ import annotations

import math

import numpy as np


def grouped_verification(records: list[dict], threshold: float, comparator: float, *,
                         seed: int = 20260926, iterations: int = 2000,
                         minimum_components: int = 20) -> dict:
    """Use frozen thresholds; retain failed pairs and shared-subject dependence.

Each record contains genuine, status, distance, left_subject, right_subject.
Subjects are run-local pseudonyms. No threshold search or model inference occurs.
"""
    if any(type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1
           for value in (threshold, comparator)):
        raise ValueError("冻结阈值必须是[0,1]内的有限数")
    if type(seed) is not int or seed < 0 or type(iterations) is not int or not 100 <= iterations <= 100000:
        raise ValueError("seed必须非负整数，bootstrap次数须在100到100000之间")
    if type(minimum_components) is not int or minimum_components < 2:
        raise ValueError("minimum_components至少为2")
    parent = {}

    def find(value):
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    seen = set()
    for row in records:
        left, right = row["left_subject"], row["right_subject"]
        if any(not isinstance(x, str) or not x for x in (left, right)):
            raise ValueError("人员分组标识不能为空")
        if type(row["genuine"]) is not bool or row["genuine"] != (left == right):
            raise ValueError("genuine与人员分组矛盾")
        if row["status"] not in {"scored", "acquisition_failed"}:
            raise ValueError("配对状态无效")
        distance = row["distance"]
        if row["status"] == "scored":
            if type(distance) not in (int, float) or not math.isfinite(distance) or not 0 <= distance <= 1:
                raise ValueError("可用pair必须有有限距离")
        elif distance is not None:
            raise ValueError("失败pair不得携带分数")
        # Pair indices are unique within one scored split.
        index = row["pair_index"]
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError("pair_index无效或重复")
        seen.add(index)
        a, b = find(left), find(right)
        parent[max(a, b)] = min(a, b)

    groups = {}
    subject_rows = {}
    for row in records:
        # counts: genuine total / usable / accepted / comparator accepted;
        # impostor total / usable / false accepted.
        counts = groups.setdefault(find(row["left_subject"]), np.zeros(7, dtype=np.int64))
        usable = row["status"] == "scored"
        accepted = usable and row["distance"] <= threshold
        compared = usable and row["distance"] <= comparator
        if row["genuine"]:
            counts[:4] += (1, usable, accepted, compared)
            subject = subject_rows.setdefault(row["left_subject"], [0, 0])
            subject[0] += accepted
            subject[1] += 1
        else:
            counts[4:] += (1, usable, accepted)
    matrix = np.array([groups[key] for key in sorted(groups)], dtype=np.int64).reshape(-1, 7)
    total = matrix.sum(axis=0)
    fields = {
        "tar_all_attempts": (2, 0), "tar_usable_pairs": (2, 1),
        "genuine_acquisition_failure_rate": (None, 0),
        "far_usable_pairs": (6, 5),
        "tar_delta_vs_fixed_engineering_all_attempts": (None, 0),
    }

    def numerator(values, field):
        if field == "genuine_acquisition_failure_rate":
            return values[..., 0] - values[..., 1]
        if field == "tar_delta_vs_fixed_engineering_all_attempts":
            return values[..., 2] - values[..., 3]
        return values[..., fields[field][0]]

    # Chunk resampling to bound memory independently of iterations*components.
    draws = []
    if len(matrix) >= minimum_components:
        rng = np.random.default_rng(seed)
        for start in range(0, iterations, 128):
            indices = rng.integers(0, len(matrix), size=(min(128, iterations-start), len(matrix)))
            draws.append(matrix[indices].sum(axis=1))
    sampled = np.concatenate(draws) if draws else None
    estimates = {}
    for field, (_, denominator_index) in fields.items():
        count, denominator = int(numerator(total, field)), int(total[denominator_index])
        metric = {"count": count, "total": denominator,
                  "rate": count / denominator if denominator else None,
                  "percentile95": None, "valid_resamples": 0}
        if not denominator:
            metric["interval_status"] = "no_denominator"
        elif sampled is None:
            metric["interval_status"] = "insufficient_components"
        else:
            mask = sampled[:, denominator_index] > 0
            values = numerator(sampled, field)[mask] / sampled[mask, denominator_index]
            metric["valid_resamples"] = int(mask.sum())
            if not mask.all():
                metric["interval_status"] = "empty_denominator_resamples"
            elif np.ptp(values) == 0:
                # A zero-error bootstrap [0,0] is NOT evidence of zero risk.
                metric["interval_status"] = "degenerate_resamples_no_risk_bound"
            else:
                metric.update(interval_status="descriptive_only",
                              percentile95=np.quantile(values, [.025, .975]).tolist())
        estimates[field] = metric
    return {
        "method": "paired_subject_connected_component_percentile_bootstrap",
        "seed": seed, "iterations": iterations, "minimum_components": minimum_components,
        "components": len(matrix), "subjects": len(parent),
        "max_component_pairs": int((matrix[:, 0] + matrix[:, 4]).max()) if len(matrix) else 0,
        "genuine_subjects": len(subject_rows),
        "macro_genuine_accept_rate": float(np.mean([a/n for a, n in subject_rows.values()])) if subject_rows else None,
        "metrics": estimates,
        "limitations": ["按共享人员连通分量重采样，不将重复人员的pair当独立试验",
                        "区间仅描述本批数据和冻结工作点；不包含标定不确定性、缺失会话、源标签或预训练重叠偏差",
                        "minimum_components是保守的软件报告门槛，不是统计独立性的证明；零误差不输出[0,0]风险保证"],
    }
