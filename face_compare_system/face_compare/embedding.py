"""Small structural interface for explicitly selected neural feature spaces.

Implementations own alignment and preprocessing. A changed model, alignment or
normalization contract must use a different signature, even at equal dimension.
The default application still constructs SFace; no optional dependency import
or model download occurs through this interface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .models import BoundingBox


@runtime_checkable
class FaceEmbedder(Protocol):
    dimension: int
    signature: str
    label: str

    def align(self, frame: NDArray[np.uint8], box: BoundingBox) -> NDArray[np.uint8]: ...

    def extract(self, aligned: NDArray[np.uint8]) -> NDArray[np.float32]: ...

    def distance(self, first: NDArray, second: NDArray) -> float: ...


def validate_embedder(embedder: FaceEmbedder) -> None:
    """Reject a malformed explicit adapter before opening the destination DB."""
    if (not isinstance(embedder, FaceEmbedder)
            or type(embedder.dimension) is not int or embedder.dimension < 1
            or not isinstance(embedder.signature, str) or not embedder.signature.strip()
            or not isinstance(embedder.label, str) or not embedder.label.strip()
            or any(not callable(getattr(embedder, name, None)) for name in ("align", "extract", "distance"))):
        raise ValueError("FaceEmbedder必须声明有效维度、特征签名、标签以及align/extract/distance")
