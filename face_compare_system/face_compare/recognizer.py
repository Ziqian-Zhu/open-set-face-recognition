"""Known/unknown identity decisions and temporal vote stabilization."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import replace
from statistics import median

import numpy as np
from numpy.typing import NDArray

from .config import RecognitionConfig
from .aggregation import aggregate_identity_score
from .decision import accept_identity
from .database import FaceDatabase
from .features import LBPHExtractor
from .models import RecognitionResult


def rank_identity_candidates(database, extractor, feature, nearest_samples):
    """Shared per-person evidence for recognition AND enrollment validation."""
    if hasattr(database, "rank_candidates"):
        return database.rank_candidates(feature, nearest_samples)
    candidates = []
    for person, samples in database.feature_sets():
        distances = [extractor.distance(feature, sample) for sample in samples]
        if not distances:
            continue
        candidates.append((aggregate_identity_score(distances, k=nearest_samples), person.person_id, person.name))
    return sorted(candidates)


class FaceRecognizer:
    """Compare one LBPH feature against all enrolled identities."""

    def __init__(
        self,
        database: FaceDatabase,
        extractor: LBPHExtractor,
        config: RecognitionConfig,
    ) -> None:
        self.database = database
        self.extractor = extractor
        self.config = config
        self.use_calibration = True

    @property
    def threshold(self) -> float:
        """Return calibrated threshold when available, otherwise configured default."""

        calibrated = self.database.calibrated_threshold if self.use_calibration else None
        return self.config.default_distance_threshold if calibrated is None else calibrated

    def match(self, feature: NDArray[np.float32]) -> RecognitionResult:
        """Classify a feature as one enrolled person or explicitly unknown."""

        if np.asarray(feature).ndim != 1 or not np.size(feature) or not np.isfinite(feature).all():
            raise ValueError("待识别特征含无效值")
        candidates = rank_identity_candidates(self.database, self.extractor, feature, self.config.nearest_samples)
        if not candidates:
            return RecognitionResult.unknown(
                threshold=self.threshold,
                reason="人脸标准库为空，请先录入人员",
            )
        candidates.sort(key=lambda item: item[0])
        best_distance, person_id, name = candidates[0]
        second_distance = candidates[1][0] if len(candidates) > 1 else None
        similarity = max(0.0, min(100.0, (1.0 - best_distance) * 100.0))
        if best_distance > self.threshold:
            return RecognitionResult(
                known=False,
                name="未知人员",
                person_id=None,
                similarity=similarity,
                distance=best_distance,
                threshold=self.threshold,
                reason="最佳匹配距离超过阈值",
                second_best_distance=second_distance,
                candidate_name=name,
                candidate_person_id=person_id,
            )
        if not accept_identity(best_distance, second_distance, self.threshold,
                               self.config.ambiguity_margin):
            return RecognitionResult(
                known=False,
                name="未知人员",
                person_id=None,
                similarity=similarity,
                distance=best_distance,
                threshold=self.threshold,
                reason="前两名距离过近，拒绝歧义匹配",
                second_best_distance=second_distance,
                candidate_name=name,
                candidate_person_id=person_id,
            )
        return RecognitionResult(
            known=True,
            name=name,
            person_id=person_id,
            similarity=similarity,
            distance=best_distance,
            threshold=self.threshold,
            reason="距离低于阈值且无身份歧义",
            second_best_distance=second_distance,
            candidate_name=name,
            candidate_person_id=person_id,
        )


class MultiFrameVoter:
    """Suppress single-frame flicker by requiring a temporal majority."""

    UNKNOWN_KEY = "__unknown__"

    def __init__(self, config: RecognitionConfig) -> None:
        self.config = config
        self._history: deque[RecognitionResult] = deque(maxlen=config.vote_window)

    def reset(self) -> None:
        """Forget all previous frames."""

        self._history.clear()

    def update(self, result: RecognitionResult) -> RecognitionResult | None:
        """Add a frame and return a stable result once quorum is reached."""

        self._history.append(result)
        if len(self._history) < self.config.vote_min_frames:
            return None
        keys = [item.person_id if item.known else self.UNKNOWN_KEY for item in self._history]
        winner, votes = Counter(keys).most_common(1)[0]
        ratio = votes / len(self._history)
        if ratio < self.config.vote_required_ratio or winner != keys[-1]:
            return None
        # Do not confirm a single new observation using old votes from an earlier person.
        if len(keys) >= 2 and keys[-2] != winner:
            return None
        matching = [
            item
            for item, key in zip(self._history, keys)
            if key == winner
        ]
        representative = min(matching, key=lambda item: abs(item.distance - median(x.distance for x in matching)))
        return replace(
            representative,
            reason=f"多帧投票稳定：{votes}/{len(self._history)} 帧（{ratio:.0%}）；{result.reason}",
        )
