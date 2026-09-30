"""Face-image quality assessment and explainable rejection reasons."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import QualityConfig
from .models import BoundingBox, QualityReport


class FaceQualityAssessor:
    """Measure exposure, contrast, focus, and relative face size."""

    def __init__(self, config: QualityConfig) -> None:
        self.config = config
        # Expected output energy of unit white noise for our exact derivative
        # filters. Compute once, including the 3x3 Gaussian preprocessing.
        impulse = np.zeros((11, 11), np.float32)
        impulse[5, 5] = 1
        smooth = cv2.GaussianBlur(impulse, (3, 3), .5)
        self._derivatives = ((1, 0, 3, 1/8), (0, 1, 3, 1/8), (2, 0, 1, 1), (0, 2, 1, 1))
        self._noise_gains = [float(np.sum(cv2.Sobel(smooth, cv2.CV_32F, x, y, ksize=k, scale=s)**2))
                             for x, y, k, s in self._derivatives]

    def _regional_focus(self, gray: NDArray[np.uint8]) -> tuple[float, tuple[float, ...], float]:
        """Contrast-relative detail at a fixed face scale; not a learned FIQA model.

        Normalize to 128x128 (no sharpening or contrast enhancement), suppress
        pixel noise mildly, then measure second / first derivative energy in three
        interior horizontal bands. The ratio approximately cancels contrast gain.
        Use the weaker of horizontal/vertical detail so directional motion blur
        cannot pass on the strength of edges in its unaffected direction alone.
        Median voting requires two bands: a strong glasses/hair edge in just one
        band cannot compensate for smooth lower-face regions. An absolute energy
        floor prevents dividing nearly-flat regions into a misleading high score.
        Estimate local noise by robust diagonal Haar residuals before smoothing;
        subtract its predicted energy from each derivative before forming ratios.
        This models approximately white additive noise, not every camera artifact.

        Thresholds are engineering defaults, not validated biometric operating
        points. Texture/noise, pose and camera processing can still affect them.
        """
        interpolation = cv2.INTER_AREA if min(gray.shape[:2]) >= 128 else cv2.INTER_LINEAR
        normalized = cv2.resize(gray, (128, 128), interpolation=interpolation).astype(np.float32)
        scores, noises = [], []
        for top, bottom in ((20, 54), (54, 86), (86, 116)):
            region = normalized[top:bottom, 20:108]
            hh = (region[1:, 1:]-region[:-1, 1:]-region[1:, :-1]+region[:-1, :-1])/2
            sigma = float(np.median(np.abs(hh-np.median(hh))))/.67448975
            noises.append(sigma)
            region = cv2.GaussianBlur(region, (3, 3), .5)
            # Exclude derivative border artifacts, not just the crop boundary.
            energies = []
            for (x, y, k, s), gain in zip(self._derivatives, self._noise_gains):
                derivative = cv2.Sobel(region, cv2.CV_32F, x, y, ksize=k, scale=s)[2:-2, 2:-2]
                energies.append(max(0., float(np.mean(derivative*derivative))-sigma*sigma*gain))
            ex, ey, exx, eyy = energies
            floor = self.config.min_gradient_energy
            score = (min(exx/ex, eyy/ey)
                     if ex+ey >= floor and min(ex, ey) >= floor/4 else 0.0)
            scores.append(score)
        return float(np.median(scores)), tuple(scores), float(np.median(noises))

    def assess(
        self,
        face: NDArray[np.uint8],
        box: BoundingBox,
        frame_shape: tuple[int, ...],
    ) -> QualityReport:
        """Return quality metrics and localized rejection reasons."""

        if face is None or face.size == 0:
            return QualityReport(False, 0.0, 0.0, 0.0, 0.0, ("人脸裁剪为空",))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if face.ndim == 3 else face
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        focus_score, focus_regions, noise_estimate = None, (), None
        if self.config.focus_method == "regional":
            focus_score, focus_regions, noise_estimate = self._regional_focus(gray)
        frame_area = max(int(frame_shape[0]) * int(frame_shape[1]), 1)
        face_size_ratio = box.area / frame_area
        reasons: list[str] = []
        # YuNet boxes are clipped to the image, but its landmarks retain their
        # original coordinates. Refuse missing eyes/mouth instead of aligning
        # an off-screen face with synthesized border pixels.
        if box.landmarks:
            points = np.asarray(box.landmarks, dtype=float)
            if points.shape != (10,) or not np.isfinite(points).all():
                reasons.append("人脸关键点无效")
            else:
                points = points.reshape(5, 2)
                if (np.any(points < 0) or np.any(points[:, 0] >= frame_shape[1])
                        or np.any(points[:, 1] >= frame_shape[0])):
                    reasons.append("人脸超出画面，请将完整面部移入画面")
        if brightness < self.config.min_brightness:
            reasons.append("光线过暗")
        if brightness > self.config.max_brightness:
            reasons.append("画面过曝")
        if contrast < self.config.min_contrast:
            reasons.append("对比度过低")
        blurry = (focus_score < self.config.min_focus_score if focus_score is not None
                  else blur_variance < self.config.min_blur_variance)
        if blurry:
            reasons.append("画面模糊")
        if face_size_ratio < self.config.min_face_ratio:
            reasons.append("人脸距离过远")
        return QualityReport(
            accepted=not reasons,
            brightness=brightness,
            contrast=contrast,
            blur_variance=blur_variance,
            face_size_ratio=face_size_ratio,
            reasons=tuple(reasons),
            focus_method=self.config.focus_method,
            focus_score=focus_score,
            focus_threshold=self.config.min_focus_score if focus_score is not None else self.config.min_blur_variance,
            focus_regions=focus_regions,
            noise_estimate=noise_estimate,
            focus_version="regional-v2-noise-corrected" if focus_score is not None else "legacy-v1",
        )
