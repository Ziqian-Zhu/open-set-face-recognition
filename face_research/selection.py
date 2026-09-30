"""Deterministic, budgeted selection over already verified face embeddings.

All methods use the same frozen SFace embeddings and the same recognition rule.
The coverage method is an engineering hypothesis, not a novel network or a
calibrated biometric quality estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from face_compare_system.face_compare.aggregation import aggregate_identity_score
from face_compare_system.face_compare.decision import accept_identity


METHODS = ("first", "random", "quality", "diversity", "coverage")
COVERAGE_DISTANCE_SCALE = 0.25
DENSITY_SMOOTHING = 0.20
QUALITY_BONUS = 0.05
COHERENCE_FLOOR = 0.25


@dataclass(frozen=True)
class Template:
    """One gallery sample, with a bounded image-quality proxy."""

    key: str
    feature: NDArray[np.float32]
    quality: float


@dataclass(frozen=True)
class GalleryIndex:
    """Validated, normalized gallery reused across all probes of one method."""

    identities: tuple[tuple[str, NDArray[np.float32]], ...]
    dimension: int


def quality_proxy(report, config) -> float:
    """Map accepted image diagnostics to [0, 1] without using probe labels.

    This is a hand-built image-quality proxy. It does not estimate recognition
    probability or infer whether a person wears glasses.
    """

    if not report.accepted:
        raise ValueError("被画质门禁拒绝的样本不能成为模板")
    midpoint = (config.min_brightness + config.max_brightness) / 2
    halfspan = (config.max_brightness - config.min_brightness) / 2
    exposure = 1 - 0.5 * min(1.0, abs(report.brightness - midpoint) / halfspan)
    focus = (
        min(1.0, report.focus_score / (2 * config.min_focus_score))
        if report.focus_score is not None
        else min(1.0, report.blur_variance / max(2 * config.min_blur_variance, 1e-8))
    )
    contrast = min(1.0, report.contrast / max(2 * config.min_contrast, 1e-8))
    area = min(1.0, report.face_size_ratio / (2 * config.min_face_ratio))
    return float(0.35 * focus + 0.25 * contrast + 0.20 * area + 0.20 * exposure)


def _matrix(templates: list[Template]) -> NDArray[np.float32]:
    if not templates:
        raise ValueError("人员没有可选模板")
    vectors = []
    dimensions = None
    for template in templates:
        feature = np.asarray(template.feature, dtype=np.float32)
        if (
            feature.ndim != 1
            or not feature.size
            or not np.isfinite(feature).all()
            or (dimensions is not None and feature.size != dimensions)
        ):
            raise ValueError("模板特征维度或数值无效")
        norm = float(np.linalg.norm(feature))
        if not np.isfinite(norm) or norm < 1e-8:
            raise ValueError("模板特征不能为零向量")
        if not np.isfinite(template.quality) or not 0 <= template.quality <= 1:
            raise ValueError("模板质量分须在[0,1]之间")
        dimensions = feature.size
        vectors.append(feature / norm)
    return np.stack(vectors)


def select_templates(
    templates: list[Template],
    method: str,
    budget: int,
    *,
    seed: int = 42,
    consistency_distance: float = 0.275,
) -> list[Template]:
    """Choose at most ``budget`` original samples using gallery information only.

    ``coverage`` greedily maximizes quality-weighted coverage of all gallery
    vectors. Inverse local density gives uncommon modes some representation;
    within-person coherence and a small quality bonus limit noisy outliers.
    The scale and bonus are fixed engineering defaults, not test-set tuning.
    """

    if method not in METHODS:
        raise ValueError(f"未知选样方法：{method}")
    if type(budget) is not int or budget < 1:
        raise ValueError("每人模板预算至少为1")
    if type(seed) is not int:
        raise ValueError("随机种子必须为整数")
    if not np.isfinite(consistency_distance) or not 0 < consistency_distance <= 1:
        raise ValueError("录入身份一致性距离必须位于(0,1]")
    matrix = _matrix(templates)
    count = len(templates)
    wanted = min(budget, count)
    if wanted == count:
        return list(templates)
    quality = np.array([item.quality for item in templates], dtype=np.float64)
    if method == "first":
        selected = list(range(wanted))
    elif method == "random":
        # Stable across Python processes and independent of dictionary hashes.
        salt = int.from_bytes(
            sha256(templates[0].key.encode("utf-8")).digest()[:8], "big"
        )
        selected = list(
            np.random.default_rng((seed + salt) % (2**64)).choice(
                count, wanted, replace=False
            )
        )
    elif method == "quality":
        selected = sorted(range(count), key=lambda index: (-quality[index], index))[
            :wanted
        ]
    else:
        distances = np.clip((1 - matrix @ matrix.T) / 2, 0, 1).astype(np.float64)
        selected = [int(np.argmax(quality))]
        if method == "diversity":
            while len(selected) < wanted:
                remaining = [i for i in range(count) if i not in selected]
                chosen = max(
                    remaining,
                    key=lambda i: (
                        float(np.min(distances[i, selected])),
                        quality[i],
                        -i,
                    ),
                )
                selected.append(chosen)
        else:
            similarity = np.clip(1 - distances / COVERAGE_DISTANCE_SCALE, 0, 1)
            # The production enrollment check is a hard gallery-level gate.
            # This soft factor further limits a borderline gallery outlier
            # without eliminating a genuine but less common appearance.
            coherence = np.empty(count, dtype=np.float64)
            for index in range(count):
                peers = np.delete(distances[index], index)
                coherence[index] = np.clip(
                    1 - np.median(peers) / (2 * consistency_distance),
                    COHERENCE_FLOOR,
                    1,
                )
            # Low-density modes receive more weight; poor or borderline
            # samples cannot win merely because they are rare.
            weights = quality * coherence / (DENSITY_SMOOTHING + similarity.sum(axis=1))
            if float(weights.sum()) <= 1e-12:
                # Public API also accepts a gallery whose proxy scores are all
                # zero; keep coverage deterministic instead of emitting NaNs.
                weights = coherence / (DENSITY_SMOOTHING + similarity.sum(axis=1))
            weights /= weights.sum()
            coverage = similarity[:, selected[0]].copy()
            while len(selected) < wanted:
                remaining = [i for i in range(count) if i not in selected]

                def value(index: int) -> tuple[float, float, int]:
                    gain = float(
                        np.dot(
                            weights,
                            np.maximum(coverage, similarity[:, index]) - coverage,
                        )
                    )
                    return (
                        gain + QUALITY_BONUS * quality[index] * coherence[index],
                        quality[index],
                        -index,
                    )

                chosen = max(remaining, key=value)
                selected.append(chosen)
                coverage = np.maximum(coverage, similarity[:, chosen])
    return [templates[index] for index in selected]


def build_gallery_index(gallery: Mapping[str, list[Template]]) -> GalleryIndex:
    """Normalize once, rather than once per identity for every probe."""

    if not gallery:
        raise ValueError("标准库不能为空")
    identities = tuple(
        (name, _matrix(templates)) for name, templates in gallery.items()
    )
    dimensions = {vectors.shape[1] for _, vectors in identities}
    if len(dimensions) != 1:
        raise ValueError("标准库模板维度不一致")
    return GalleryIndex(identities, dimensions.pop())


def rank_with_index(
    index: GalleryIndex,
    feature: NDArray[np.float32],
    nearest_samples: int = 3,
    *,
    strategy: str = "topk_median",
) -> list[tuple[float, str]]:
    """Use the production distance and nearest-k median on a compiled gallery."""

    if type(nearest_samples) is not int or nearest_samples < 1:
        raise ValueError("nearest_samples必须为正整数")
    query = np.asarray(feature, dtype=np.float32)
    if (
        query.ndim != 1
        or query.size != index.dimension
        or not np.isfinite(query).all()
        or float(np.linalg.norm(query)) < 1e-8
    ):
        raise ValueError("待测特征无效")
    query = query / np.linalg.norm(query)
    candidates = []
    for name, vectors in index.identities:
        distances = np.clip((1 - vectors @ query) / 2, 0, 1)
        value = aggregate_identity_score(distances, strategy, nearest_samples)
        candidates.append((value, name))
    return sorted(candidates, key=lambda item: (item[0], item[1]))


def rank_identities(
    gallery: Mapping[str, list[Template]],
    feature: NDArray[np.float32],
    nearest_samples: int = 3,
    *,
    strategy: str = "topk_median",
) -> list[tuple[float, str]]:
    """Convenience API for one query; use ``rank_with_index`` for batches."""

    return rank_with_index(build_gallery_index(gallery), feature, nearest_samples, strategy=strategy)


def decide(
    candidates: list[tuple[float, str]], threshold: float, margin: float
) -> str | None:
    """Return one identity only when distance and runner-up margin both pass."""

    if (
        not np.isfinite(threshold)
        or not np.isfinite(margin)
        or not 0 <= threshold <= 1
        or not 0 <= margin <= 1
    ):
        raise ValueError("匹配阈值或间隔无效")
    if not candidates:
        return None
    best, name = candidates[0]
    second = candidates[1][0] if len(candidates) > 1 else None
    return name if accept_identity(best, second, threshold, margin) else None
