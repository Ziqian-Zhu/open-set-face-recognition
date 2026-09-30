from dataclasses import replace
import numpy as np

from face_compare.config import RecognitionConfig, TrackingConfig
from face_compare.deep_engine import SFaceExtractor
from face_compare.models import BoundingBox, FaceObservation, QualityReport, RecognitionResult
from face_compare.tracking import MultiFaceTracker


def face(x=0, identity="A", vector=(1., 0.), quality=True):
    result = RecognitionResult(True, identity, identity, 95, .05, .275, "match") if identity else RecognitionResult.unknown(threshold=.275, reason="threshold")
    return FaceObservation(BoundingBox(x, 10, 80, 100), QualityReport(quality, 120, 40, 100, .2),
                           np.zeros((100, 80, 3), np.uint8), result, np.array(vector, np.float32) if quality else None)


def tracker(**kwargs):
    return MultiFaceTracker(SFaceExtractor, RecognitionConfig(vote_min_frames=3), TrackingConfig(**kwargs))


def test_two_people_have_independent_votes_and_survive_order_changes():
    t = tracker()
    for index in range(3):
        observations = [face(0), face(140, "B", (0., 1.))]
        if index == 1:
            observations.reverse()
        t.update(observations, index*.1)
    assert [(o.track_id, o.stable_recognition.name) for o in observations] == [(1, "A"), (2, "B")]


def test_different_people_cross_without_sharing_identity_votes():
    t = tracker()
    for index, positions in enumerate(((0, 160), (50, 110), (100, 60), (150, 10))):
        observations = [face(positions[0]), face(positions[1], "B", (0., 1.))]
        t.update(observations, index*.1)
    assert [o.track_id for o in observations] == [1, 2]
    assert [o.stable_recognition.name for o in observations] == ["A", "B"]


def test_new_person_in_same_position_cannot_inherit_identity():
    t = tracker()
    for i in range(4):
        t.update([face()], i*.1)
    b = face(0, "B", (0., 1.))
    t.update([b], .4)
    assert b.track_id == 2 and b.stable_recognition is None


def test_unknown_tracks_stay_separate():
    t = tracker()
    for i in range(3):
        observations = [face(0, None), face(140, None, (0., 1.))]
        t.update(observations, i*.1)
    assert observations[0].track_id != observations[1].track_id
    assert all(o.tracking_state == "unknown" for o in observations)


def test_quality_miss_gap_and_timestamp_rewind_clear_confirmation():
    for interruption in ("quality", "miss", "gap", "rewind"):
        t = tracker()
        for i in range(4):
            t.update([face()], i*.1)
        if interruption == "quality":
            t.update([face(quality=False)], .4)
        elif interruption == "miss":
            t.update([], .4)
        observation = face()
        t.update([observation], 2. if interruption == "gap" else .1 if interruption == "rewind" else .5)
        assert observation.stable_recognition is None


def test_ambiguous_crossing_restarts_instead_of_guessing():
    t = tracker()
    t.update([face(0), face(160, "B", (1., 0.))], 0)
    observations = [face(80), face(80, "B", (1., 0.))]
    t.update(observations, .1)
    assert all(o.track_id not in (1, 2) for o in observations)


def test_same_enrolled_identity_claimed_by_two_tracks_is_conflict():
    t = tracker(max_center_distance=.5)
    for i in range(3):
        observations = [face(0), face(200)]
        t.update(observations, i*.1)
    assert all(o.tracking_state == "identity_conflict" and o.stable_recognition is None for o in observations)


def test_track_capacity_bounds_memory():
    t = tracker(max_tracks=2)
    observations = [face(0), face(140), face(280)]
    t.update(observations, 0)
    assert len(t.tracks) == 2
    assert observations[2].tracking_state == "capacity"
