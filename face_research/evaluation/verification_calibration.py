"""Validation-only event-threshold calibration for 1:1, not open-set 1:N."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .metrics import VerificationPair, _checked_pairs, verification_at_threshold


@dataclass(frozen=True)
class FrozenVerificationPoint:
    threshold: float
    target_far: float
    validation: dict
    sweep: tuple[dict, ...]


def calibrate_verification(
    pairs: Sequence[VerificationPair], *, split: str, target_far: float = 0.05,
) -> FrozenVerificationPoint:
    """Maximize usable-pair TAR under empirical FAR; tie: fewer FA, lower tau.

    Include zero and every observed distance (acceptance is inclusive). If
    zero-distance impostors make the target infeasible, fail instead of
    inventing a negative production threshold or accessing test to repair it.
    """
    if split != "validation":
        raise ValueError("只能使用validation验证对选择阈值")
    if not math.isfinite(target_far) or not 0 <= target_far < 1:
        raise ValueError("target_far必须在[0,1)内")
    checked = _checked_pairs(pairs)
    if not any(p.genuine for p in checked) or not any(not p.genuine for p in checked):
        raise ValueError("校准至少需要一个可用genuine和impostor验证对")
    sweep = tuple(verification_at_threshold(checked, threshold) for threshold in
                  sorted({0.0, *(p.distance for p in checked)}))
    eligible = [row for row in sweep if row["far"] <= target_far]
    if not eligible:
        raise ValueError("[0,1]内没有阈值满足验证集经验FAR目标")
    best = max(eligible, key=lambda row: (row["true_accept_count"],
                                        -row["false_accept_count"], -row["threshold"]))
    return FrozenVerificationPoint(best["threshold"], target_far, best, sweep)
