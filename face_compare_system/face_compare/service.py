"""Application service joining detection, quality, storage, and recognition."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from dataclasses import replace
import threading
from contextlib import nullcontext

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import AppConfig, RecognitionConfig
from .database import CalibrationResult, FaceDatabase, PersonSummary, normalize_person_name
from .detector import FaceDetector, largest_box
from .event_log import JsonlEventLogger
from .embedding import FaceEmbedder, validate_embedder
from .features import LBPHExtractor
from .models import BoundingBox, FaceObservation, PreparedSample, RecognitionResult
from .quality import FaceQualityAssessor
from .recognizer import FaceRecognizer, rank_identity_candidates


class FaceComparisonSystem:
    """High-level use cases shared by Tkinter and headless CLI."""

    def __init__(self, config: AppConfig, project_root: str | Path, *,
                 embedder: FaceEmbedder | None = None) -> None:
        if embedder is not None:
            validate_embedder(embedder)
            if config.engine.backend != "sface":
                raise ValueError("可选神经特征仅支持YuNet五点路径，不能注入LBPH分支")
        self.config = config
        self.lock = threading.RLock()
        self.project_root = Path(project_root).resolve()
        storage_path = self._resolve_path(config.storage.database_directory)
        log_path = self._resolve_path(config.storage.event_log)
        if config.engine.backend == "sface":
            from .deep_engine import SFaceExtractor, YuNetDetector
            directory = self._resolve_path(config.engine.model_directory)
            self.detector = YuNetDetector(directory, config.engine, config.detection)
            self.extractor = SFaceExtractor(directory) if embedder is None else embedder
            self.recognition_config = replace(
                config.recognition,
                default_distance_threshold=(1-config.engine.cosine_threshold)/2,
                ambiguity_margin=config.engine.cosine_margin/2,
            )
            self.engine_label = f"YuNet + 五点对齐 + {self.extractor.label}"
            signature = self.extractor.signature
        else:
            self.detector = FaceDetector(config.detection)
            self.extractor = LBPHExtractor(config.feature)
            self.recognition_config = config.recognition
            self.engine_label = "Haar + U2-LBPH（基线）"
            signature = f"lbph-u2-{config.feature.image_size}-{config.feature.grid_rows}-{config.feature.grid_columns}-{config.feature.radii}"
        self.quality = FaceQualityAssessor(config.quality)
        database_type = FaceDatabase
        if config.storage.backend == "sqlite":
            from .vector_database import SQLiteVectorDatabase
            database_type = SQLiteVectorDatabase
        self.database = database_type(
            storage_path,
            save_face_images=config.storage.save_face_images,
            feature_signature=signature,
            expected_dimension=self.extractor.dimension,
        )
        self.recognizer = FaceRecognizer(self.database, self.extractor, self.recognition_config)
        # A neural operating threshold cannot be inferred from enrollment pairs.
        self.recognizer.use_calibration = config.engine.backend == "lbph"
        self.logger = JsonlEventLogger(log_path)

    def _feature_input(self, frame, box, crop):
        if self.config.engine.backend == "sface":
            return self.extractor.align(frame, box)
        return crop

    def _resolve_path(self, path: str | Path) -> Path:
        """Resolve configuration paths relative to the project directory."""

        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.project_root / candidate

    def analyze_frame(
        self,
        frame: NDArray[np.uint8],
        *,
        recognize: bool = True,
        log_events: bool = True,
    ) -> list[FaceObservation]:
        """Detect every face, measure quality, and optionally identify each one."""

        observations: list[FaceObservation] = []
        for box in self.detector.detect(frame):
            crop = self.detector.crop(frame, box)
            report = self.quality.assess(crop, box, frame.shape)
            result: RecognitionResult | None = None
            feature = None
            if recognize:
                if report.accepted:
                    feature = self.extractor.extract(self._feature_input(frame, box, crop))
                    result = self.recognizer.match(feature)
                else:
                    result = RecognitionResult.unknown(
                        threshold=self.recognizer.threshold,
                        reason="图像质量不足：" + "、".join(report.reasons),
                    )
                if log_events:
                    self.logger.write(
                        "recognition",
                        known=result.known,
                        name=result.name,
                        similarity=round(result.similarity, 3),
                        distance=round(result.distance, 6),
                        threshold=round(result.threshold, 6),
                        reason=result.reason,
                        quality=report.to_dict(),
                    )
            observations.append(
                FaceObservation(box=box, crop=crop, quality=report, recognition=result, feature=feature)
            )
        if log_events and not observations:
            self.logger.write("no_face")
        return observations

    def prepare_sample(
        self,
        frame: NDArray[np.uint8],
        *,
        allow_low_quality: bool = False,
        require_single_face: bool = True,
    ) -> PreparedSample:
        """Extract one enrollment sample after face-count and quality checks."""

        boxes = self.detector.detect(frame)
        if not boxes:
            raise ValueError("图像中未检测到人脸")
        if require_single_face and len(boxes) != 1:
            raise ValueError(f"录入图像应恰好包含一张人脸，当前检测到 {len(boxes)} 张")
        box = largest_box(boxes)
        assert box is not None
        crop = self.detector.crop(frame, box)
        report = self.quality.assess(crop, box, frame.shape)
        if not report.accepted and not allow_low_quality:
            # Log metrics only: no frame, embedding or person's name on rejection.
            self.logger.write("enrollment_quality_rejected", quality=report.to_dict())
            detail = "样本质量不合格：" + "、".join(report.reasons)
            if "画面模糊" in report.reasons:
                detail += f"\n{report.focus_description}。请停稳、正面补光后重试。"
            raise ValueError(detail)
        prepared = self._feature_input(frame, box, crop)
        feature = self.extractor.extract(prepared)
        return PreparedSample(crop=prepared, feature=feature, quality=report)

    def prepare_face_crop(
        self,
        crop: NDArray[np.uint8],
        *,
        allow_low_quality: bool = False,
    ) -> PreparedSample:
        """Prepare an already-cropped face without camera or face detection.

        This explicit path is useful for repeatable automated tests and for
        upstream systems that already provide a face region.  The crop itself is
        treated as the complete frame, so the face-size gate always passes while
        exposure, contrast, and sharpness checks remain active.
        """

        if crop is None or crop.size == 0:
            raise ValueError("人脸裁剪为空")
        if self.config.engine.backend == "sface":
            raise ValueError("SFace录入必须通过prepare_sample执行检测和五点对齐，不能直接传未对齐裁剪")
        height, width = crop.shape[:2]
        box = BoundingBox(0, 0, width, height)
        report = self.quality.assess(crop, box, crop.shape)
        if not report.accepted and not allow_low_quality:
            raise ValueError("样本质量不合格：" + "、".join(report.reasons))
        return PreparedSample(
            crop=crop.copy(),
            feature=self.extractor.extract(crop),
            quality=report,
        )

    def enroll_face_crops(
        self,
        name: str,
        crops: Iterable[NDArray[np.uint8]],
        *,
        allow_low_quality: bool = False,
    ) -> PersonSummary:
        """Enroll pre-cropped faces through a detector-free API."""

        samples = [
            self.prepare_face_crop(crop, allow_low_quality=allow_low_quality)
            for crop in crops
        ]
        return self.enroll_samples(name, samples)

    def recognize_face_crop(
        self,
        crop: NDArray[np.uint8],
        *,
        allow_low_quality: bool = False,
    ) -> RecognitionResult:
        """Recognize one pre-cropped face without camera or detector state."""

        sample = self.prepare_face_crop(crop, allow_low_quality=allow_low_quality)
        result = self.recognizer.match(sample.feature)
        self.logger.write(
            "crop_recognition",
            known=result.known,
            name=result.name,
            similarity=round(result.similarity, 3),
            distance=round(result.distance, 6),
            threshold=round(result.threshold, 6),
            reason=result.reason,
        )
        return result

    def enroll_samples(self, name: str, samples: Sequence[PreparedSample]) -> PersonSummary:
        """Persist captured samples and invalidate the old calibration."""

        guard = self.database.write_guard() if hasattr(self.database, "write_guard") else nullcontext()
        with guard:
            person = self._enroll_samples(name, samples)
        self.logger.write("enrollment", person_id=person.person_id, name=person.name,
                          added_samples=len(samples), total_samples=person.sample_count)
        return person

    def _enroll_samples(self, name, samples):

        name = normalize_person_name(name)
        if self.config.engine.backend == "sface":
            from .enrollment import validate_identity_set
            validate_identity_set(samples, self.extractor, self.config.engine.enrollment_consistency)
            count = getattr(self.config, "recognition", RecognitionConfig()).nearest_samples
            append_limit = (1-self.config.engine.enrollment_consistency)/2
            match_limit = (1-self.config.engine.cosine_threshold)/2
            margin = self.config.engine.cosine_margin/2
            # Validate every sample against the unchanged gallery, within the
            # outer transaction. New samples cannot bootstrap their own approval.
            for sample in samples:
                candidates = rank_identity_candidates(self.database, self.extractor, sample.feature, count)
                own = next((row for row in candidates if row[2].casefold() == name.casefold()), None)
                others = [row for row in candidates if row[2].casefold() != name.casefold()]
                if own is not None:
                    if own[0] > append_limit:
                        raise ValueError("新样本与该姓名已有样本差异过大，请确认是否为本人")
                    if others and (others[0][0]-own[0] < margin or others[0][0] < (1-.65)/2):
                        raise ValueError("新样本与另一已录入身份接近，存在身份歧义；未保存")
                elif others and others[0][0] <= max(match_limit, (1-.65)/2):
                    raise ValueError("本次人脸与另一已录入身份匹配或高度相似，请使用原姓名补录，避免重复身份")
        person = self.database.add_samples(name, samples)
        return person

    def enroll_images(
        self,
        name: str,
        paths: Iterable[str | Path],
        *,
        allow_low_quality: bool = False,
    ) -> tuple[PersonSummary, list[str]]:
        """Enroll all valid image paths and report individual rejections."""

        samples: list[PreparedSample] = []
        rejected: list[str] = []
        for path in paths:
            image_path = Path(path)
            image = cv2.imread(str(image_path))
            if image is None:
                rejected.append(f"{image_path}: 无法读取图像")
                continue
            try:
                samples.append(
                    self.prepare_sample(image, allow_low_quality=allow_low_quality)
                )
            except ValueError as error:
                rejected.append(f"{image_path}: {error}")
        if not samples:
            details = "；".join(rejected) if rejected else "没有输入图像"
            raise ValueError(f"没有可录入的有效样本：{details}")
        return self.enroll_samples(name, samples), rejected

    def recognize_image(self, path: str | Path) -> list[FaceObservation]:
        """Read and recognize all faces in one still image."""

        image_path = Path(path)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"无法读取图像: {image_path}")
        return self.analyze_frame(frame, recognize=True, log_events=True)

    def calibrate_threshold(self) -> CalibrationResult:
        """Calibrate and persist the decision threshold from current samples."""

        if self.config.engine.backend == "sface":
            # Keep the fixed conservative operating point; real calibration needs a held-out set.
            return CalibrationResult(self.recognizer.threshold,
                                     "SFace固定阈值（须用独立验证集评估；不以录入样本自校准）", 0, 0)
        settings = self.recognition_config
        result = self.database.calibrate(
            self.extractor.distance,
            fallback=settings.default_distance_threshold,
            minimum=settings.threshold_min,
            maximum=settings.threshold_max,
        )
        self.logger.write(
            "calibration",
            threshold=result.threshold,
            source=result.source,
            intra_count=result.intra_count,
            inter_count=result.inter_count,
            false_accept_rate=result.false_accept_rate,
            false_reject_rate=result.false_reject_rate,
        )
        return result

    @staticmethod
    def annotate(
        frame: NDArray[np.uint8],
        observations: Sequence[FaceObservation],
    ) -> NDArray[np.uint8]:
        """Draw portable ASCII labels; Chinese details remain in the side panel."""

        output = frame.copy()
        for observation in observations:
            tracked = observation.tracking_state != "untracked"
            result = observation.stable_recognition if tracked else observation.recognition
            accepted = bool(result and result.known)
            color = (80, 210, 120) if accepted else (70, 120, 245)
            if not observation.quality.accepted:
                color = (30, 190, 245)
            x, y, width, height = observation.box.as_tuple()
            cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)
            if result is None:
                label = observation.tracking_state.upper() if tracked else "FACE"
            elif result.known:
                ascii_name = result.name if result.name.isascii() else "KNOWN"
                label = f"{ascii_name} score:{result.similarity:.1f}"
            else:
                label = f"UNKNOWN score:{result.similarity:.1f}"
            if observation.track_id is not None:
                label = f"T{observation.track_id} {label}"
            label_y = max(24, y - 9)
            cv2.putText(
                output,
                label,
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )
        return output
