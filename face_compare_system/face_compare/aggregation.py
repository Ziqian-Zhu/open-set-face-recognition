"""Per-identity aggregation of template distances.

Scores are distances: smaller means a better match. The production default is
the median of the nearest three templates; other strategies are opt-in research
controls and must be calibrated separately before comparing operating points.
"""

from __future__ import annotations

import numpy as np


STRATEGIES = (
    "nearest",
    "mean",
    "median",
    "topk_mean",
    "topk_median",
)


def aggregate_identity_score(distances, strategy: str = "topk_median", k: int = 3) -> float:
    """Reduce nonempty finite template distances to one comparable identity score.

    ``k`` is capped by the number of templates. It is ignored for all-template
    strategies and ``nearest``. The caller owns metric conversion/normalization.
    """

    values = np.asarray(distances)
    if (values.ndim != 1 or not values.size or not np.issubdtype(values.dtype, np.number)
            or np.issubdtype(values.dtype, np.complexfloating)
            or not np.isfinite(values).all()):
        raise ValueError("模板距离必须是非空的一维有限数数组")
    if strategy not in STRATEGIES:
        raise ValueError(f"未知模板聚合策略：{strategy}")
    if type(k) is not int or k < 1:
        raise ValueError("最近模板数量 k 必须是正整数")
    if strategy == "nearest":
        return float(np.min(values))
    if strategy == "mean":
        return float(np.mean(values))
    if strategy == "median":
        return float(np.median(values))
    count = min(k, values.size)
    nearest = np.partition(values, count - 1)[:count]
    if strategy == "topk_mean":
        return float(np.mean(nearest))
    return float(np.median(nearest))


def aggregate_identity_scores(distances, strategy: str = "topk_median", k: int = 3) -> np.ndarray:
    """Row-wise equivalent of :func:`aggregate_identity_score` for equal budgets.

    Rows are identities and columns are their template distances. Grouping by
    actual template count avoids padding: NaN/inf sentinels could change means,
    medians or the min(k, N) rule. No approximate candidate pruning is performed.
    The scalar implementation remains an independent reference for regressions.
    """

    values = np.asarray(distances)
    if (values.ndim != 2 or not values.size or not np.issubdtype(values.dtype, np.number)
            or np.issubdtype(values.dtype, np.complexfloating)
            or not np.isfinite(values).all()):
        raise ValueError("批量模板距离必须是非空的二维有限数数组")
    if strategy not in STRATEGIES:
        raise ValueError(f"未知模板聚合策略：{strategy}")
    if type(k) is not int or k < 1:
        raise ValueError("最近模板数量 k 必须是正整数")
    if strategy == "nearest":
        return np.min(values, axis=1)
    if strategy == "mean":
        return np.mean(values, axis=1)
    if strategy == "median":
        return np.median(values, axis=1)
    count = min(k, values.shape[1])
    nearest = np.partition(values, count - 1, axis=1)[:, :count]
    if strategy == "topk_mean":
        return np.mean(nearest, axis=1)
    return np.median(nearest, axis=1)
