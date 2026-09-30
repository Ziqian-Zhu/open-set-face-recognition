"""Pure NumPy Local Binary Pattern Histogram feature extraction."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import FeatureConfig


FloatVector = NDArray[np.float32]
UNIFORM_LBP_BINS = 59


def uniform_lbp_mapping() -> NDArray[np.uint8]:
    """Map 256 raw 8-neighbour codes to 58 U2 patterns plus one bucket.

    A pattern is uniform when its circular bit string has at most two 0↔1
    transitions.  The 58 uniform raw codes receive stable bins 0–57 (numeric
    code order); every non-uniform code maps to bin 58.
    """

    mapping = np.full(256, UNIFORM_LBP_BINS - 1, dtype=np.uint8)
    uniform_codes: list[int] = []
    for code in range(256):
        bits = [(code >> bit) & 1 for bit in range(8)]
        transitions = sum(bits[index] != bits[(index + 1) % 8] for index in range(8))
        if transitions <= 2:
            uniform_codes.append(code)
    if len(uniform_codes) != 58:
        raise RuntimeError(f"U2 LBP 映射构造失败: {len(uniform_codes)} != 58")
    for bin_index, code in enumerate(uniform_codes):
        mapping[code] = bin_index
    return mapping


_U2_MAPPING = uniform_lbp_mapping()


class LBPHExtractor:
    """Extract a multi-radius, spatial uniform-LBP (U2) descriptor.

    Every pixel is encoded by comparing eight neighbours with its centre.
    Histograms retain the distribution of these explainable texture codes in a
    spatial grid.  CLAHE normalization reduces illumination sensitivity.
    """

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    @property
    def dimension(self) -> int:
        """Return descriptor length."""

        return (
            len(self.config.radii)
            * self.config.grid_rows
            * self.config.grid_columns
            * UNIFORM_LBP_BINS
        )

    def preprocess(self, face: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Convert a face crop to normalized fixed-size grayscale."""

        if face is None or face.size == 0:
            raise ValueError("人脸图像为空")
        if face.ndim == 3:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        elif face.ndim == 2:
            gray = face
        else:
            raise ValueError(f"不支持的图像维度: {face.shape}")
        size = self.config.image_size
        interpolation = cv2.INTER_AREA if max(gray.shape) > size else cv2.INTER_CUBIC
        normalized = cv2.resize(gray, (size, size), interpolation=interpolation)
        return self._clahe.apply(normalized)

    def extract(self, face: NDArray[np.uint8]) -> FloatVector:
        """Return a deterministic multi-radius LBPH feature vector."""

        gray = self.preprocess(face)
        components: list[NDArray[np.float32]] = []
        for radius in self.config.radii:
            lbp = self._lbp_image(gray, radius)
            components.extend(self._grid_histograms(lbp))
        vector = np.concatenate(components).astype(np.float32, copy=False)
        if vector.size != self.dimension:
            raise RuntimeError(f"LBPH 维度异常: {vector.size} != {self.dimension}")
        return vector

    def _lbp_image(self, gray: NDArray[np.uint8], radius: int) -> NDArray[np.uint8]:
        """Encode eight compass neighbours around every valid centre pixel."""

        padded = np.pad(gray, radius, mode="reflect")
        height, width = gray.shape
        centre = padded[radius : radius + height, radius : radius + width]
        offsets: Sequence[tuple[int, int]] = (
            (-radius, -radius),
            (-radius, 0),
            (-radius, radius),
            (0, radius),
            (radius, radius),
            (radius, 0),
            (radius, -radius),
            (0, -radius),
        )
        codes = np.zeros_like(gray, dtype=np.uint8)
        for bit, (row_offset, column_offset) in enumerate(offsets):
            neighbour = padded[
                radius + row_offset : radius + row_offset + height,
                radius + column_offset : radius + column_offset + width,
            ]
            codes |= ((neighbour >= centre).astype(np.uint8) << bit)
        return codes

    def _grid_histograms(self, codes: NDArray[np.uint8]) -> list[NDArray[np.float32]]:
        """Calculate L1-normalized 59-bin uniform-LBP histograms per cell."""

        uniform_codes = _U2_MAPPING[codes]
        row_cells = np.array_split(uniform_codes, self.config.grid_rows, axis=0)
        histograms: list[NDArray[np.float32]] = []
        for row_cell in row_cells:
            for cell in np.array_split(row_cell, self.config.grid_columns, axis=1):
                histogram = np.bincount(cell.ravel(), minlength=UNIFORM_LBP_BINS).astype(np.float32)
                histogram /= max(float(histogram.sum()), 1.0)
                histograms.append(histogram)
        return histograms

    def distance(self, first: FloatVector, second: FloatVector) -> float:
        """Return mean Chi-square distance in the approximate range [0, 1]."""

        if first.shape != second.shape:
            raise ValueError(f"特征维度不一致: {first.shape} 与 {second.shape}")
        cell_count = len(self.config.radii) * self.config.grid_rows * self.config.grid_columns
        numerator = np.square(first - second)
        denominator = first + second + np.float32(1e-10)
        value = 0.5 * float(np.sum(numerator / denominator)) / float(cell_count)
        return max(0.0, min(1.0, value))


def chi_square_distance(
    first: NDArray[np.floating],
    second: NDArray[np.floating],
    *,
    cells: int = 1,
) -> float:
    """Standalone Chi-square distance useful for tests and calibration."""

    first_array = np.asarray(first, dtype=np.float32)
    second_array = np.asarray(second, dtype=np.float32)
    if first_array.shape != second_array.shape:
        raise ValueError("特征维度不一致")
    return 0.5 * float(
        np.sum(np.square(first_array - second_array) / (first_array + second_array + 1e-10))
    ) / max(cells, 1)
