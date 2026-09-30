"""Offline-friendly face comparison system.

The package intentionally uses an explainable Local Binary Pattern Histogram
(LBPH) implementation instead of downloading a pretrained network.  It is
designed for teaching and demonstrations, not security-critical identity
verification.
"""

from .config import AppConfig, load_config
from .models import BoundingBox, FaceObservation, QualityReport, RecognitionResult

__all__ = [
    "AppConfig",
    "BoundingBox",
    "FaceObservation",
    "QualityReport",
    "RecognitionResult",
    "load_config",
]

__version__ = "1.0.0"
