"""Shared open-set acceptance rule for production and offline experiments."""

from __future__ import annotations

import math


def accept_identity(best_distance: float, second_distance: float | None,
                    threshold: float, margin: float) -> bool:
    """Accept iff the best identity clears both inclusive distance constraints."""

    values = (best_distance, threshold, margin)
    if not all(math.isfinite(value) for value in values) or margin < 0:
        raise ValueError("身份距离、阈值或间隔无效")
    if second_distance is not None and not math.isfinite(second_distance):
        raise ValueError("第二名身份距离无效")
    if best_distance > threshold:
        return False
    if second_distance is None:
        return True
    gap = second_distance - best_distance
    # Decimal configuration values (e.g. 0.315 - 0.275) are not represented
    # exactly as binary floats. Only neutralize round-off at the equality edge.
    return gap >= margin or math.isclose(gap, margin, rel_tol=0.0, abs_tol=1e-12)
