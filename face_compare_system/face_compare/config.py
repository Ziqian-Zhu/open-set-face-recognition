"""Configuration loading with validated, typed defaults."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar


@dataclass(frozen=True)
class CameraConfig:
    """Camera discovery and capture settings."""

    preferred_index: int = 0
    scan_count: int = 6
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass(frozen=True)
class DetectionConfig:
    """Haar cascade detector settings."""

    scale_factor: float = 1.10
    min_neighbors: int = 5
    min_face_pixels: int = 80
    use_profile_detector: bool = True
    padding_ratio: float = 0.14


@dataclass(frozen=True)
class FeatureConfig:
    """Explainable LBPH descriptor settings."""

    image_size: int = 128
    grid_rows: int = 8
    grid_columns: int = 8
    radii: tuple[int, ...] = (1, 2)


@dataclass(frozen=True)
class EngineConfig:
    """Explicit feature space: never silently mix neural and LBP templates."""

    backend: str = "lbph"
    model_directory: str = "models"
    detection_score: float = 0.90
    max_detection_side: int = 640
    cosine_threshold: float = 0.45
    cosine_margin: float = 0.08
    enrollment_consistency: float = 0.45


@dataclass(frozen=True)
class QualityConfig:
    """Enrollment/recognition input quality gates."""

    min_brightness: float = 45.0
    max_brightness: float = 220.0
    min_contrast: float = 22.0
    min_blur_variance: float = 55.0
    min_face_ratio: float = 0.035
    # Keep the legacy baseline explicit; SFace's shipped config opts into regional.
    focus_method: str = "legacy"
    min_focus_score: float = 0.30
    min_gradient_energy: float = 16.0


@dataclass(frozen=True)
class RecognitionConfig:
    """Distance threshold, ambiguity rejection, and temporal voting settings."""

    default_distance_threshold: float = 0.43
    threshold_min: float = 0.20
    threshold_max: float = 0.62
    ambiguity_margin: float = 0.025
    nearest_samples: int = 3
    vote_window: int = 7
    vote_min_frames: int = 4
    vote_required_ratio: float = 0.60


@dataclass(frozen=True)
class StorageConfig:
    """Paths relative to the project directory unless absolute."""

    database_directory: str = "data"
    event_log: str = "logs/events.jsonl"
    save_face_images: bool = True
    backend: str = "files"


@dataclass(frozen=True)
class TrackingConfig:
    max_gap_seconds: float = 0.9
    max_feature_distance: float = 0.25
    ambiguity_gap: float = 0.025
    max_center_distance: float = 1.5
    max_tracks: int = 32


@dataclass(frozen=True)
class UIConfig:
    """Display and real-time processing settings."""

    title: str = "智能人脸比对系统"
    process_every_n_frames: int = 3
    preview_width: int = 960
    preview_height: int = 640
    enrollment_target_samples: int = 5
    reduce_motion: bool = False


@dataclass(frozen=True)
class AppConfig:
    """Complete application configuration."""

    camera: CameraConfig = CameraConfig()
    detection: DetectionConfig = DetectionConfig()
    feature: FeatureConfig = FeatureConfig()
    quality: QualityConfig = QualityConfig()
    recognition: RecognitionConfig = RecognitionConfig()
    storage: StorageConfig = StorageConfig()
    ui: UIConfig = UIConfig()
    engine: EngineConfig = EngineConfig()
    tracking: TrackingConfig = TrackingConfig()


T = TypeVar("T")


def _build_dataclass(cls: type[T], values: dict[str, Any]) -> T:
    """Create a dataclass while rejecting unknown configuration keys."""

    valid = {item.name for item in fields(cls)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"{cls.__name__} 包含未知配置项: {sorted(unknown)}")
    if cls is FeatureConfig and "radii" in values:
        values = dict(values)
        radii = values["radii"]
        if not isinstance(radii, (list, tuple)) or any(type(value) is not int for value in radii):
            raise ValueError("feature.radii 必须为整数列表")
        values["radii"] = tuple(radii)
    defaults = cls()
    for key, value in values.items():
        expected = type(getattr(defaults, key))
        valid_type = (type(value) in (int, float) if expected is float else type(value) is expected)
        if not valid_type or (type(value) in (int, float) and not math.isfinite(value)):
            raise ValueError(f"{cls.__name__}.{key} 类型错误或包含非有限数值")
    return cls(**values)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load JSON configuration, falling back to safe defaults.

    Args:
        path: Configuration file. ``None`` returns built-in defaults.

    Returns:
        A fully typed :class:`AppConfig`.
    """

    if path is None:
        return AppConfig()
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("配置文件根节点必须是 JSON 对象")
    allowed_sections = {item.name for item in fields(AppConfig)}
    unknown_sections = set(raw) - allowed_sections
    if unknown_sections:
        raise ValueError(f"未知配置分区: {sorted(unknown_sections)}")
    section_types: dict[str, type[Any]] = {
        "camera": CameraConfig,
        "detection": DetectionConfig,
        "feature": FeatureConfig,
        "quality": QualityConfig,
        "recognition": RecognitionConfig,
        "storage": StorageConfig,
        "ui": UIConfig,
        "engine": EngineConfig,
        "tracking": TrackingConfig,
    }
    values: dict[str, Any] = {}
    for name, section_type in section_types.items():
        section = raw.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f"配置分区 {name} 必须是 JSON 对象")
        values[name] = _build_dataclass(section_type, section)
    config = AppConfig(**values)
    _validate_config(config)
    return config


def _validate_config(config: AppConfig) -> None:
    """Raise ``ValueError`` for inconsistent numeric settings."""

    if config.camera.scan_count < 1:
        raise ValueError("camera.scan_count 必须大于 0")
    if config.camera.preferred_index < 0 or min(config.camera.width, config.camera.height, config.camera.fps) < 1:
        raise ValueError("摄像头编号不能为负数，分辨率与帧率必须为正数")
    if min(config.ui.process_every_n_frames, config.ui.preview_width,
           config.ui.preview_height, config.ui.enrollment_target_samples) < 1:
        raise ValueError("界面尺寸、采样间隔及录入目标必须为正数")
    if min(config.quality.min_contrast, config.quality.min_blur_variance) < 0:
        raise ValueError("对比度和清晰度阈值不能为负数")
    if config.quality.focus_method not in ("legacy", "regional"):
        raise ValueError("quality.focus_method 必须为 legacy 或 regional")
    if config.quality.min_focus_score <= 0 or config.quality.min_gradient_energy <= 0:
        raise ValueError("分区细节阈值和最小边缘能量必须大于 0")
    if config.engine.backend not in ("lbph", "sface"):
        raise ValueError("engine.backend 必须为 lbph 或 sface")
    if config.storage.backend not in ("files", "sqlite"):
        raise ValueError("storage.backend 必须为 files 或 sqlite")
    if config.storage.backend == "sqlite" and config.engine.backend != "sface":
        raise ValueError("SQLite余弦向量库仅用于SFace；LBPH请保留files基线")
    if (config.tracking.max_gap_seconds <= 0 or not 0 < config.tracking.max_feature_distance < 1
            or not 0 <= config.tracking.ambiguity_gap < 1 or config.tracking.max_center_distance <= 0
            or config.tracking.max_tracks < 1):
        raise ValueError("多人跟踪参数无效")
    if not 0 < config.engine.detection_score <= 1 or config.engine.max_detection_side < 160:
        raise ValueError("检测置信阈值或输入尺寸无效")
    if not 0 < config.engine.cosine_threshold < 1 or not 0 <= config.engine.cosine_margin < 1:
        raise ValueError("余弦阈值或身份间隔无效")
    if not 0 < config.engine.enrollment_consistency < 1:
        raise ValueError("录入一致性阈值须位于 (0,1)")
    if not 0 <= config.quality.min_brightness < config.quality.max_brightness <= 255:
        raise ValueError("亮度范围无效")
    if not 0 < config.quality.min_face_ratio <= 1:
        raise ValueError("人脸面积比例须位于 (0,1]")
    if not 1.01 <= config.detection.scale_factor <= 2.0:
        raise ValueError("detection.scale_factor 应位于 [1.01, 2.0]")
    if config.feature.image_size < 32:
        raise ValueError("feature.image_size 至少为 32")
    if min(config.feature.grid_rows, config.feature.grid_columns) < 1:
        raise ValueError("LBPH 网格数必须为正数")
    if max(config.feature.grid_rows, config.feature.grid_columns) > config.feature.image_size:
        raise ValueError("LBPH 网格数不能超过归一化图像尺寸")
    if not config.feature.radii or min(config.feature.radii) < 1:
        raise ValueError("feature.radii 必须包含正整数")
    recognition = config.recognition
    if not 0.0 <= recognition.threshold_min < recognition.threshold_max <= 1.0:
        raise ValueError("距离阈值上下限必须满足 0 <= min < max <= 1")
    if not recognition.threshold_min <= recognition.default_distance_threshold <= recognition.threshold_max:
        raise ValueError("默认阈值必须位于阈值上下限之间")
    if recognition.nearest_samples < 1:
        raise ValueError("nearest_samples 必须大于 0")
    if recognition.ambiguity_margin < 0:
        raise ValueError("ambiguity_margin 不能为负数")
    if not 1 <= recognition.vote_min_frames <= recognition.vote_window:
        raise ValueError("vote_min_frames 必须位于 [1, vote_window]")
    if not 0.5 <= recognition.vote_required_ratio <= 1.0:
        raise ValueError("vote_required_ratio 应位于 [0.5, 1.0]")
