"""Conservative appearance + geometry association with independent identity votes.

Ambiguous associations start new tracks instead of transferring an old identity.
This is a bounded CPU baseline, not a claim of state-of-the-art MOT accuracy.
"""

from dataclasses import dataclass
import math

from .config import TrackingConfig
from .detector import FaceDetector
from .recognizer import MultiFrameVoter


@dataclass
class Track:
    track_id: int
    box: object
    feature: object
    timestamp: float
    voter: MultiFrameVoter


class MultiFaceTracker:
    def __init__(self, extractor, recognition_config, config=None):
        self.extractor = extractor
        self.recognition_config = recognition_config
        self.config = config or TrackingConfig()
        self._next_id = 1
        self._timestamp = None
        self.tracks = {}

    def reset(self):
        self.tracks.clear()
        self._timestamp = None

    def _cost(self, track, observation):
        a, b = track.box, observation.box
        center = math.hypot((a.x+a.width/2)-(b.x+b.width/2), (a.y+a.height/2)-(b.y+b.height/2))
        scale = max(math.hypot(a.width, a.height), math.hypot(b.width, b.height), 1)
        if center/scale > self.config.max_center_distance:
            return None
        overlap = FaceDetector._intersection_over_union(a, b)
        if track.feature is not None and observation.feature is not None:
            distance = self.extractor.distance(track.feature, observation.feature)
            if distance > self.config.max_feature_distance:
                return None
            return .75*distance + .25*(1-overlap)/2
        return (1-overlap)/2 if overlap >= .3 else None

    def update(self, observations, timestamp):
        if not math.isfinite(timestamp):
            raise ValueError("视频时间戳无效")
        if self._timestamp is not None and timestamp <= self._timestamp:
            self.reset()  # replay/seek/duplicate timestamps cannot inherit votes
        self._timestamp = timestamp
        self.tracks = {tid: track for tid, track in self.tracks.items()
                       if timestamp-track.timestamp <= self.config.max_gap_seconds}
        edges = []
        for tid, track in self.tracks.items():
            for index, observation in enumerate(observations):
                cost = self._cost(track, observation)
                if cost is not None:
                    edges.append((cost, tid, index))
        # If either endpoint has near-tied alternatives, do not carry identity.
        ambiguous_tracks, ambiguous_faces = set(), set()
        for tid in self.tracks:
            choices = sorted(cost for cost, candidate, _ in edges if candidate == tid)
            if len(choices) > 1 and choices[1]-choices[0] < self.config.ambiguity_gap:
                ambiguous_tracks.add(tid)
        for index in range(len(observations)):
            choices = sorted(cost for cost, _, candidate in edges if candidate == index)
            if len(choices) > 1 and choices[1]-choices[0] < self.config.ambiguity_gap:
                ambiguous_faces.add(index)
        assigned, used = {}, set()
        for _, tid, index in sorted(edges):
            if tid not in used and index not in assigned and tid not in ambiguous_tracks and index not in ambiguous_faces:
                assigned[index] = tid
                used.add(tid)
        for tid, track in self.tracks.items():
            if tid not in used:
                track.voter.reset()  # no confirmation survives a missed detection
        for index, observation in enumerate(observations):
            observation.stable_recognition = None
            tid = assigned.get(index)
            if tid is None:
                if len(self.tracks) >= self.config.max_tracks:
                    removable = [key for key in self.tracks if key not in used]
                    if not removable:
                        observation.track_id = None
                        observation.tracking_state = "capacity"
                        continue
                    self.tracks.pop(min(removable, key=lambda key: self.tracks[key].timestamp))
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = Track(tid, observation.box, observation.feature, timestamp,
                                         MultiFrameVoter(self.recognition_config))
                used.add(tid)
            track = self.tracks[tid]
            track.box, track.timestamp = observation.box, timestamp
            observation.track_id = tid
            if not observation.quality.accepted or observation.feature is None or observation.recognition is None:
                track.voter.reset()
                observation.tracking_state = "quality_rejected"
                continue
            track.feature = observation.feature.copy()
            stable = track.voter.update(observation.recognition)
            observation.stable_recognition = stable
            observation.tracking_state = ("known" if stable.known else "unknown") if stable else "confirming"
        # Two tracks matching one enrolled identity are not independent proof.
        claimed = {}
        for observation in observations:
            result = observation.stable_recognition
            if result is not None and result.known:
                claimed.setdefault(result.person_id, []).append(observation)
        for group in claimed.values():
            if len(group) > 1:
                for observation in group:
                    observation.stable_recognition = None
                    observation.tracking_state = "identity_conflict"
                    self.tracks[observation.track_id].voter.reset()
        return observations
