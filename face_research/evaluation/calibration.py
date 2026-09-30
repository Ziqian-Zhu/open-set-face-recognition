"""Validation-only joint threshold/margin search for open-set identification."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from .metrics import IdentificationAttempt, identification_report, validate_identification_attempts
from ..selection import decide


@dataclass(frozen=True)
class FrozenOperatingPoint:
    threshold: float
    margin: float
    target_fpir: float
    validation: dict
    sweep: tuple[dict, ...]


def calibrate_open_set(
    attempts: Sequence[IdentificationAttempt],
    *,
    split: str,
    target_fpir: float,
    margins: Sequence[float],
) -> FrozenOperatingPoint:
    """Maximize correct known accepts under empirical usable-face FPIR.

    The caller must pass the validation split explicitly. No test data enters
    this function. Results are empirical and do not guarantee population FPIR.
    All-attempt accuracy counts only successful usable-face decisions; failed
    acquisitions stay in the denominator, including failures on unknown faces.
    Accuracy is descriptive and never participates in operating-point selection.
    """

    if split != "validation":
        raise ValueError("只能用validation选择阈值与间隔；test必须使用已冻结参数")
    if not math.isfinite(target_fpir) or not 0 <= target_fpir < 1:
        raise ValueError("target_fpir必须在[0,1)内")
    grid = tuple(sorted(set(margins)))
    if not grid or any(not math.isfinite(value) or not 0 <= value <= 1 for value in grid):
        raise ValueError("margin候选必须是[0,1]内的有限数")
    rows = validate_identification_attempts(attempts)
    known_total = sum(row.expected is not None for row in rows)
    unknown_total = len(rows) - known_total
    usable_known = sum(row.expected is not None and row.status == "recognized" for row in rows)
    usable_unknown = sum(row.expected is None and row.status == "recognized" for row in rows)
    if usable_known < 1 or usable_unknown < 1:
        raise ValueError("验证集至少需要一张可用已知脸和一张可用未知脸")
    sweep = []
    winner = None
    for margin in grid:
        events: dict[float, list[tuple[str | None, str]]] = defaultdict(list)
        for row in rows:
            if row.status != "recognized" or not row.candidates:
                continue
            identity = decide(list(row.candidates), 1.0, margin)
            if identity is not None:
                distance = float(row.candidates[0][0])
                if not math.isfinite(distance) or not 0 <= distance <= 1:
                    raise ValueError("验证集身份距离必须在[0,1]内")
                events[distance].append((row.expected, identity))
        correct = misidentified = false_accept = 0
        thresholds = sorted({0.0, *events})
        for threshold in thresholds:
            for expected, predicted in events.get(threshold, ()):
                if expected is None:
                    false_accept += 1
                elif predicted == expected:
                    correct += 1
                else:
                    misidentified += 1
            rejected_known = known_total - correct - misidentified
            accepted_unknown = false_accept
            # Acquisition failure is non-acceptance, not successful rejection.
            rejected_unknown = usable_unknown - accepted_unknown
            correct_all = correct + rejected_unknown
            record = {
                "threshold": threshold,
                "margin": margin,
                "known_correct": correct,
                "known_rejected": rejected_known,
                "known_misidentified": misidentified,
                "known_total": known_total,
                "usable_known": usable_known,
                "known_acquisition_failures": known_total - usable_known,
                "unknown_false_accept": accepted_unknown,
                "unknown_total": unknown_total,
                "usable_unknown": usable_unknown,
                "unknown_acquisition_failures": unknown_total - usable_unknown,
                "unknown_correct_rejection": rejected_unknown,
                "correct_all_attempts": correct_all,
                "total_attempts": len(rows),
                "fpir_usable": accepted_unknown / usable_unknown,
                "known_correct_rate": correct / known_total,
                "known_reject_rate": rejected_known / known_total,
                "accuracy_all_attempts": correct_all / len(rows),
                # No verification pair protocol is present in a 1:N sweep.
                "FAR": None,
                "FRR": None,
                "TAR": None,
            }
            sweep.append(record)
            if record["fpir_usable"] > target_fpir + 1e-12:
                continue
            score = (correct, -misidentified, -false_accept, -threshold, -margin)
            if winner is None or score > winner[0]:
                winner = (score, threshold, margin)
    if winner is None:
        raise ValueError("验证集无法满足目标未知人员误接受率；不得使用测试集调参")
    _, threshold, margin = winner
    return FrozenOperatingPoint(
        threshold,
        margin,
        target_fpir,
        identification_report(rows, threshold, margin),
        tuple(sweep),
    )
