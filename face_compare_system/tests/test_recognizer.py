"""Tests for top-k distance, threshold, ambiguity margin, and voting."""

from __future__ import annotations

import numpy as np
import pytest

from face_compare.config import RecognitionConfig
from face_compare.database import PersonSummary
from face_compare.models import RecognitionResult
from face_compare.recognizer import FaceRecognizer, MultiFrameVoter


class FakeDatabase:
    def __init__(self, threshold: float, identities):
        self.calibrated_threshold = threshold
        self._identities = identities

    def feature_sets(self):
        return self._identities


class AbsoluteExtractor:
    @staticmethod
    def distance(first, second) -> float:
        return abs(float(first[0]) - float(second[0]))


def _person(person_id: str, name: str) -> PersonSummary:
    return PersonSummary(person_id, name, 3, "2026-01-01T00:00:00+08:00")


def test_top_k_median_returns_known_person_below_threshold() -> None:
    database = FakeDatabase(
        0.30,
        [
            (_person("a", "Alice"), tuple(np.asarray([x], np.float32) for x in (0.10, 0.11, 0.12))),
            (_person("b", "Bob"), tuple(np.asarray([x], np.float32) for x in (0.75, 0.80, 0.85))),
        ],
    )
    recognizer = FaceRecognizer(database, AbsoluteExtractor(), RecognitionConfig(nearest_samples=3))
    result = recognizer.match(np.asarray([0.11], dtype=np.float32))
    assert result.known
    assert result.name == "Alice"
    assert result.distance == pytest.approx(0.01)


def test_distance_threshold_rejects_unknown() -> None:
    database = FakeDatabase(
        0.20,
        [
            (_person("a", "Alice"), (np.asarray([0.10], np.float32),)),
            (_person("b", "Bob"), (np.asarray([0.90], np.float32),)),
        ],
    )
    recognizer = FaceRecognizer(database, AbsoluteExtractor(), RecognitionConfig())
    result = recognizer.match(np.asarray([0.50], dtype=np.float32))
    assert not result.known
    assert result.name == "未知人员"
    assert "超过阈值" in result.reason


def test_margin_rejects_ambiguous_first_and_second_place() -> None:
    database = FakeDatabase(
        0.50,
        [
            (_person("a", "Alice"), (np.asarray([0.40], np.float32),)),
            (_person("b", "Bob"), (np.asarray([0.42], np.float32),)),
        ],
    )
    config = RecognitionConfig(ambiguity_margin=0.03, nearest_samples=1)
    recognizer = FaceRecognizer(database, AbsoluteExtractor(), config)
    result = recognizer.match(np.asarray([0.41], dtype=np.float32))
    assert not result.known
    assert "歧义" in result.reason


def test_multiframe_vote_requires_configured_quorum() -> None:
    config = RecognitionConfig(vote_window=5, vote_min_frames=4, vote_required_ratio=0.60)
    voter = MultiFrameVoter(config)
    known = RecognitionResult(True, "Alice", "a", 95.0, 0.05, 0.30, "single")
    unknown = RecognitionResult.unknown(threshold=0.30, reason="single", distance=0.5)
    assert voter.update(known) is None
    assert voter.update(known) is None
    assert voter.update(unknown) is None
    assert voter.update(known) is None
    stable = voter.update(known)
    assert stable is not None
    assert stable.known
    assert "4/5" in stable.reason


def test_old_majority_never_confirms_current_unknown_or_new_person():
    voter = MultiFrameVoter(RecognitionConfig())
    alice = RecognitionResult(True, "Alice", "a", 95, 0.05, 0.3, "single")
    bob = RecognitionResult(True, "Bob", "b", 95, 0.05, 0.3, "single")
    for _ in range(7):
        voter.update(alice)
    assert voter.update(bob) is None
    assert voter.update(RecognitionResult.unknown(threshold=0.3, reason="quality")) is None
