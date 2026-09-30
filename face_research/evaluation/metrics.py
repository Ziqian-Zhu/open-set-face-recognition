"""Explicitly separate 1:1 verification from open-set 1:N identification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .._baseline import interval
from ..selection import decide


@dataclass(frozen=True)
class VerificationPair:
    distance: float
    genuine: bool
    # A unique ID for an independent impostor attempt. Repeated IDs mean
    # correlated pairs and are insufficient for low-FAR extrapolation.
    attempt_id: str | None = None


@dataclass(frozen=True)
class IdentificationAttempt:
    expected: str | None
    status: str
    candidates: tuple[tuple[float, str], ...]


def validate_identification_attempts(attempts: Sequence[IdentificationAttempt]) -> list[IdentificationAttempt]:
    """Keep ranking/metric assumptions explicit at the experiment boundary."""

    rows = list(attempts)
    for row in rows:
        if row.status != "recognized" and row.candidates:
            raise ValueError("不可用人脸不能携带身份候选")
        scores = []
        names = set()
        for score, name in row.candidates:
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("候选身份距离必须在[0,1]内")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("候选身份名称无效或重复")
            scores.append(score)
            names.add(name)
        if scores != sorted(scores):
            raise ValueError("候选身份必须按距离升序排列")
    return rows


def _checked_pairs(pairs: Sequence[VerificationPair]) -> list[VerificationPair]:
    checked = list(pairs)
    if not checked:
        raise ValueError("验证对不能为空")
    for pair in checked:
        if not isinstance(pair.genuine, bool) or not math.isfinite(pair.distance) or not 0 <= pair.distance <= 1:
            raise ValueError("验证对必须包含[0,1]内的半余弦距离及布尔真值")
    return checked


def _verification_counts(pairs: Sequence[VerificationPair], threshold: float) -> dict:
    genuine = [item for item in pairs if item.genuine]
    impostor = [item for item in pairs if not item.genuine]
    true_accept = sum(item.distance <= threshold for item in genuine)
    false_accept = sum(item.distance <= threshold for item in impostor)
    return {
        "threshold": threshold,
        "genuine_count": len(genuine),
        "impostor_count": len(impostor),
        "true_accept_count": true_accept,
        "false_reject_count": len(genuine) - true_accept,
        "false_accept_count": false_accept,
        "true_reject_count": len(impostor) - false_accept,
        "tar": true_accept / len(genuine) if genuine else None,
        "frr": (len(genuine) - true_accept) / len(genuine) if genuine else None,
        "far": false_accept / len(impostor) if impostor else None,
    }


def verification_at_threshold(pairs: Sequence[VerificationPair], threshold: float) -> dict:
    """Score a frozen threshold without searching on test pair labels."""

    checked = _checked_pairs(pairs)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("验证阈值必须在[0,1]内")
    return _verification_counts(checked, threshold)


def verification_report(
    pairs: Sequence[VerificationPair],
    threshold: float,
    *,
    target_fars: tuple[float, ...] = (1e-2, 1e-3),
) -> dict:
    """Return a descriptive ROC plus explicitly support-gated TAR@FAR.

    TAR@FAR is not reported unless every impostor pair has a *distinct*,
    nonempty attempt ID and the zero-error Wilson 95% bound could resolve the
    target FAR at the available sample count. IDs are protocol metadata, not
    proof of statistical independence; this remains a descriptive estimate.
    """

    checked = _checked_pairs(pairs)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("验证阈值必须在[0,1]内")
    if any(not math.isfinite(target) or not 0 < target < 1 for target in target_fars):
        raise ValueError("目标FAR必须在(0,1)内")
    fixed = _verification_counts(checked, threshold)
    fixed["tar_wilson95"] = interval(fixed["true_accept_count"], fixed["genuine_count"])
    fixed["far_wilson95"] = interval(fixed["false_accept_count"], fixed["impostor_count"])
    genuine_count = fixed["genuine_count"]
    impostor_count = fixed["impostor_count"]
    roc = []
    auc = None
    if genuine_count and impostor_count:
        distances = sorted({item.distance for item in checked})
        roc = [_verification_counts(checked, math.nextafter(distances[0], -math.inf))]
        roc.extend(_verification_counts(checked, value) for value in distances)
        auc = sum(
            (right["far"] - left["far"]) * (right["tar"] + left["tar"]) / 2
            for left, right in zip(roc, roc[1:])
        )
    impostor_ids = [item.attempt_id for item in checked if not item.genuine]
    independent = bool(impostor_ids) and all(impostor_ids) and len(set(impostor_ids)) == len(impostor_ids)
    target_results = {}
    z2 = 1.959963984540054**2
    for target in target_fars:
        key = f"{target:g}"
        minimum = math.ceil(z2 * (1 - target) / target)
        if not genuine_count or not impostor_count:
            target_results[key] = {"status": "insufficient samples", "reason": "缺少genuine或impostor对", "minimum_independent_impostors": minimum}
        elif not independent:
            target_results[key] = {"status": "insufficient samples", "reason": "impostor attempt_id缺失或重复，独立尝试数不可证实", "minimum_independent_impostors": minimum}
        elif impostor_count < minimum:
            target_results[key] = {"status": "insufficient samples", "reason": "独立impostor数不足以解析目标FAR", "independent_impostors": impostor_count, "minimum_independent_impostors": minimum}
        else:
            eligible = [point for point in roc if point["far"] <= target]
            best = max(eligible, key=lambda point: (point["tar"], -point["far"], -point["threshold"]))
            target_results[key] = {"status": "estimated", "tar": best["tar"], "far": best["far"], "threshold": best["threshold"], "tar_wilson95": interval(best["true_accept_count"], genuine_count), "far_wilson95": interval(best["false_accept_count"], impostor_count), "independent_impostors": impostor_count, "minimum_independent_impostors": minimum, "note": "测试样本的经验工作点，不是人群风险保证；不同ID也不保证统计独立，Wilson区间仅作近似"}
    return {
        "metric": "half_cosine_distance",
        "operating_point": fixed,
        "roc": roc,
        "auc": auc,
        "tar_at_far": target_results,
        "limitations": ["ROC使用固定pair清单；同人/同会话pair的相关性不能按独立样本计算置信度", "TAR@FAR只在impostor尝试标识唯一且样本量足够时报告；标识唯一不能证明样本真正独立"],
    }


def identification_report(
    attempts: Sequence[IdentificationAttempt], threshold: float, margin: float
) -> dict:
    """Report all-attempt and usable-face denominators without conflating FAR/FPIR."""

    rows = validate_identification_attempts(attempts)
    known = [row for row in rows if row.expected is not None]
    unknown = [row for row in rows if row.expected is None]
    usable_known = [row for row in known if row.status == "recognized"]
    usable_unknown = [row for row in unknown if row.status == "recognized"]
    def prediction(row: IdentificationAttempt) -> str | None:
        return decide(list(row.candidates), threshold, margin) if row.status == "recognized" else None
    correct = sum(prediction(row) == row.expected for row in known)
    rejected_known = sum(prediction(row) is None for row in known)
    misidentified = len(known) - correct - rejected_known
    false_accept_all = sum(prediction(row) is not None for row in unknown)
    correct_unknown_rejection = len(usable_unknown) - false_accept_all
    top1_correct = sum(bool(row.candidates) and row.candidates[0][1] == row.expected for row in usable_known)
    return {
        "known_attempts": len(known),
        "unknown_attempts": len(unknown),
        "usable_known_attempts": len(usable_known),
        "usable_unknown_attempts": len(usable_unknown),
        "top1_closed_set_usable_known": interval(top1_correct, len(usable_known)),
        "dir_known_correct_all_attempts": interval(correct, len(known)),
        "known_reject_all_attempts": interval(rejected_known, len(known)),
        "known_misidentification_all_attempts": interval(misidentified, len(known)),
        "fpir_unknown_all_attempts": interval(false_accept_all, len(unknown)),
        "fpir_unknown_usable_faces": interval(false_accept_all, len(usable_unknown)),
        "unknown_rejection_all_attempts": interval(len(unknown) - false_accept_all, len(unknown)),
        # Preserve the historical non-acceptance metric above (1 - FPIR).
        # Only usable faces can contribute to successful classification below.
        "unknown_correct_rejection_all_attempts": interval(correct_unknown_rejection, len(unknown)),
        "unknown_acquisition_failures_all_attempts": interval(len(unknown) - len(usable_unknown), len(unknown)),
        "known_acquisition_failures_all_attempts": interval(len(known) - len(usable_known), len(known)),
        "accuracy_all_attempts": interval(correct + correct_unknown_rejection, len(rows)),
        "status_counts": {status: sum(row.status == status for row in rows) for status in sorted({row.status for row in rows})},
        "threshold": threshold,
        "margin": margin,
    }
