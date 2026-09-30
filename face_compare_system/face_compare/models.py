"""Shared data models used by the camera, algorithm, service, and UI layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    """Rectangle in image pixel coordinates."""

    x: int
    y: int
    width: int
    height: int
    landmarks: tuple[float, ...] = ()
    confidence: float = 1.0

    @property
    def area(self) -> int:
        """Return the rectangle area in pixels."""

        return max(0, self.width) * max(0, self.height)

    @property
    def right(self) -> int:
        """Return the exclusive right edge."""

        return self.x + self.width

    @property
    def bottom(self) -> int:
        """Return the exclusive bottom edge."""

        return self.y + self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return ``(x, y, width, height)`` for OpenCV calls."""

        return self.x, self.y, self.width, self.height


@dataclass(frozen=True)
class QualityReport:
    """Explainable quality measurements for one cropped face."""

    accepted: bool
    brightness: float
    contrast: float
    blur_variance: float
    face_size_ratio: float
    reasons: tuple[str, ...] = ()
    focus_method: str = "legacy"
    focus_score: float | None = None
    focus_threshold: float | None = None
    focus_regions: tuple[float, ...] = ()
    noise_estimate: float | None = None
    focus_version: str = "legacy-v1"

    @property
    def focus_description(self) -> str:
        if self.focus_method == "regional" and self.focus_score is not None:
            return f"分区细节 {self.focus_score:.2f} / 门槛 {self.focus_threshold:.2f}"
        return f"清晰度 {self.blur_variance:.0f}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


@dataclass(frozen=True)
class RecognitionResult:
    """Identity decision and the values that explain it."""

    known: bool
    name: str
    person_id: str | None
    similarity: float
    distance: float
    threshold: float
    reason: str
    second_best_distance: float | None = None
    candidate_name: str | None = None
    candidate_person_id: str | None = None

    @classmethod
    def unknown(
        cls,
        *,
        threshold: float,
        reason: str,
        distance: float = 1.0,
        similarity: float = 0.0,
    ) -> "RecognitionResult":
        """Build a consistent unknown-person result."""

        return cls(
            known=False,
            name="未知人员",
            person_id=None,
            similarity=similarity,
            distance=distance,
            threshold=threshold,
            reason=reason,
        )


@dataclass
class FaceObservation:
    """A detected face with quality and optional recognition result."""

    box: BoundingBox
    quality: QualityReport
    crop: Any = field(repr=False)
    recognition: RecognitionResult | None = None
    feature: Any = field(default=None, repr=False)
    track_id: int | None = None
    stable_recognition: RecognitionResult | None = None
    tracking_state: str = "untracked"


@dataclass(frozen=True)
class PreparedSample:
    """Validated enrollment sample ready for persistent storage."""

    crop: Any = field(repr=False)
    feature: Any = field(repr=False)
    quality: QualityReport
