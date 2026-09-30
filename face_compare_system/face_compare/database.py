"""File-backed, dynamically extensible face sample database."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
import copy
import hashlib
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .models import PreparedSample


FeatureVector = NDArray[np.float32]
DistanceFunction = Callable[[FeatureVector, FeatureVector], float]


@dataclass(frozen=True)
class PersonSummary:
    """Public identity metadata shown by the UI and CLI."""

    person_id: str
    name: str
    sample_count: int
    created_at: str


@dataclass(frozen=True)
class CalibrationResult:
    """Threshold selection details for reproducibility."""

    threshold: float
    source: str
    intra_count: int
    inter_count: int
    false_accept_rate: float | None = None
    false_reject_rate: float | None = None


def normalize_person_name(name: str) -> str:
    """Validate a display name while keeping Unicode names intact."""

    if not isinstance(name, str) or any(unicodedata.category(c).startswith("C") for c in name):
        raise ValueError("姓名中不能包含控制字符")
    normalized = re.sub(r"\s+", " ", name.strip())
    if not normalized:
        raise ValueError("姓名不能为空")
    if len(normalized) > 64:
        raise ValueError("姓名不能超过 64 个字符")
    if any(character in normalized for character in "\r\n\t"):
        raise ValueError("姓名中不能包含控制字符")
    return normalized


class FaceDatabase:
    """Persist images, NumPy descriptors, and metadata under one directory."""

    SCHEMA_VERSION = 1

    def __init__(self, root: str | Path, *, save_face_images: bool = True,
                 feature_signature: str | None = None, expected_dimension: int | None = None) -> None:
        self.root = Path(root)
        self.metadata_path = self.root / "people.json"
        self.samples_root = self.root / "samples"
        self.features_root = self.root / "features"
        self.save_face_images = save_face_images
        self.feature_signature = feature_signature
        self.expected_dimension = expected_dimension
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.samples_root.mkdir(parents=True, exist_ok=True)
        self.features_root.mkdir(parents=True, exist_ok=True)
        if not self.metadata_path.exists():
            initial = self._empty_metadata()
            initial["feature_signature"] = feature_signature
            self._write_metadata(initial)
        self._metadata = self._read_metadata()
        stored = self._metadata.get("feature_signature")
        if stored != feature_signature and self._metadata["people"]:
            # Unmarked legacy data may only be opened as LBPH.
            if not (stored is None and (feature_signature or "").startswith("lbph-")):
                raise RuntimeError("人脸库的模型/配置与当前特征空间不一致，请使用独立数据目录")
        self._revision = self.metadata_path.read_bytes()
        self._feature_cache: dict[str, list[FeatureVector]] = {}
        self.refresh_cache()

    @classmethod
    def _empty_metadata(cls) -> dict[str, Any]:
        """Return a new schema document."""

        return {
            "schema_version": cls.SCHEMA_VERSION,
            "people": [],
            "calibration": None,
        }

    def _read_metadata(self) -> dict[str, Any]:
        """Load and minimally validate metadata."""

        try:
            with self.metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"人脸库元数据损坏或不可读: {self.metadata_path}") from error
        if metadata.get("schema_version") != self.SCHEMA_VERSION:
            raise RuntimeError("人脸库版本不兼容")
        if not isinstance(metadata.get("people"), list):
            raise RuntimeError("人脸库 people 字段格式错误")
        return metadata

    def _write_metadata(self, metadata: dict[str, Any]) -> None:
        """Atomically replace the JSON metadata file."""

        temporary = self.root / f".people-{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.metadata_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def refresh_cache(self) -> None:
        """Reload all feature arrays referenced by metadata."""

        cache: dict[str, list[FeatureVector]] = {}
        with self._lock:
            for person in self._metadata["people"]:
                vectors: list[FeatureVector] = []
                for sample in person.get("samples", []):
                    feature_path = self.root / sample["feature_path"]
                    if not feature_path.resolve().is_relative_to(self.root.resolve()):
                        raise RuntimeError("特征路径越出标准库目录")
                    if not feature_path.is_file():
                        raise RuntimeError(f"标准库缺少特征文件：{feature_path.name}，请恢复备份")
                    vector = np.load(feature_path, allow_pickle=False).astype(np.float32, copy=False)
                    self._validate_vector(vector)
                    vectors.append(vector)
                cache[person["id"]] = vectors
            self._feature_cache = cache

    def _validate_vector(self, vector):
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise ValueError("特征必须是非空、有限数值的一维向量")
        if self.expected_dimension is not None and vector.size != self.expected_dimension:
            raise ValueError("特征维数与当前模型不一致，请重新录入到独立标准库")

    def _check_revision(self):
        if not self.metadata_path.exists() or self.metadata_path.read_bytes() != self._revision:
            raise RuntimeError("标准库已被另一个程序修改或清空，请重启本程序后再操作")

    def _commit(self, metadata):
        self._check_revision()
        metadata["feature_signature"] = self.feature_signature
        self._write_metadata(metadata)
        self._metadata = metadata
        self._revision = self.metadata_path.read_bytes()

    def list_people(self) -> list[PersonSummary]:
        """Return enrolled identities sorted by name."""

        with self._lock:
            result = [
                PersonSummary(
                    person_id=person["id"],
                    name=person["name"],
                    sample_count=len(person.get("samples", [])),
                    created_at=person["created_at"],
                )
                for person in self._metadata["people"]
            ]
        return sorted(result, key=lambda item: item.name.casefold())

    def feature_sets(self) -> list[tuple[PersonSummary, tuple[FeatureVector, ...]]]:
        """Return immutable views of identities and their cached descriptors."""

        people = {person.person_id: person for person in self.list_people()}
        with self._lock:
            return [
                (people[person_id], tuple(vectors))
                for person_id, vectors in self._feature_cache.items()
                if person_id in people and vectors
            ]

    def add_samples(self, name: str, samples: Sequence[PreparedSample]) -> PersonSummary:
        """Append validated samples to an existing or new person.

        The person lookup is case-insensitive, enabling dynamic expansion without
        accidentally creating duplicate identities such as ``Alice``/``alice``.
        """

        normalized_name = normalize_person_name(name)
        if not samples:
            raise ValueError("至少需要一张有效人脸样本")
        # Reject the whole batch before creating any files for invalid features.
        for sample in samples:
            self._validate_vector(np.asarray(sample.feature, dtype=np.float32))
        now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        with self._lock:
            self._check_revision()
            metadata = copy.deepcopy(self._metadata)
            person = next(
                (
                    item
                    for item in metadata["people"]
                    if item["name"].casefold() == normalized_name.casefold()
                ),
                None,
            )
            if person is None:
                person = {
                    "id": uuid.uuid4().hex,
                    "name": normalized_name,
                    "created_at": now,
                    "samples": [],
                }
                metadata["people"].append(person)
            person_id = person["id"]
            feature_directory = self.features_root / person_id
            image_directory = self.samples_root / person_id
            feature_directory.mkdir(parents=True, exist_ok=True)
            if self.save_face_images:
                image_directory.mkdir(parents=True, exist_ok=True)

            new_vectors: list[FeatureVector] = []
            for sample in samples:
                sample_id = uuid.uuid4().hex
                feature_relative = Path("features") / person_id / f"{sample_id}.npy"
                vector = np.asarray(sample.feature, dtype=np.float32)
                self._validate_vector(vector)
                np.save(self.root / feature_relative, vector, allow_pickle=False)
                image_relative: Path | None = None
                if self.save_face_images:
                    image_relative = Path("samples") / person_id / f"{sample_id}.jpg"
                    ok = cv2.imwrite(
                        str(self.root / image_relative),
                        sample.crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 94],
                    )
                    if not ok:
                        raise OSError(f"无法保存样本图像: {image_relative}")
                person["samples"].append(
                    {
                        "id": sample_id,
                        "created_at": now,
                        "feature_path": feature_relative.as_posix(),
                        "image_path": image_relative.as_posix() if image_relative else None,
                        "quality": sample.quality.to_dict(),
                        "image_sha256": hashlib.sha256(np.ascontiguousarray(sample.crop).tobytes()).hexdigest(),
                    }
                )
                new_vectors.append(vector)
            # New data invalidate the previous calibrated operating point.
            metadata["calibration"] = None
            self._commit(metadata)
            self._feature_cache.setdefault(person_id, []).extend(new_vectors)
            return PersonSummary(
                person_id=person_id,
                name=person["name"],
                sample_count=len(person["samples"]),
                created_at=person["created_at"],
            )

    def archive_people(self, person_id: str | None = None) -> Path:
        """Recoverable removal. Snapshot metadata retains references to original files."""
        with self._lock:
            self._check_revision()
            if person_id is not None and not any(p["id"] == person_id for p in self._metadata["people"]):
                raise ValueError("未找到该人员")
            archive = self.root / "archives"
            archive.mkdir(exist_ok=True)
            path = archive / f"people-{uuid.uuid4().hex}.json"
            path.write_bytes(self._revision)
            updated = copy.deepcopy(self._metadata)
            updated["people"] = [p for p in updated["people"] if person_id is not None and p["id"] != person_id]
            updated["calibration"] = None
            self._commit(updated)
            self.refresh_cache()
            return path

    @property
    def calibrated_threshold(self) -> float | None:
        """Return the stored operating threshold, if calibration is current."""

        calibration = self._metadata.get("calibration")
        return float(calibration["threshold"]) if calibration else None

    def calibrate(
        self,
        distance: DistanceFunction,
        *,
        fallback: float,
        minimum: float,
        maximum: float,
    ) -> CalibrationResult:
        """Choose a threshold minimizing twice FAR plus FRR on enrolled pairs.

        Same-person pairs estimate genuine distances and cross-person pairs
        estimate impostor distances.  False accepts carry double weight because
        accepting the wrong identity is usually worse than returning unknown.
        """

        feature_sets = self.feature_sets()
        intra: list[float] = []
        inter: list[float] = []
        for _, vectors in feature_sets:
            for first_index in range(len(vectors)):
                for second_index in range(first_index + 1, len(vectors)):
                    intra.append(distance(vectors[first_index], vectors[second_index]))
        for first_index in range(len(feature_sets)):
            first_vectors = feature_sets[first_index][1]
            for second_index in range(first_index + 1, len(feature_sets)):
                second_vectors = feature_sets[second_index][1]
                # Bound calibration time while sampling all identities fairly.
                for first_vector in first_vectors[:10]:
                    for second_vector in second_vectors[:10]:
                        inter.append(distance(first_vector, second_vector))

        if not intra or not inter:
            result = CalibrationResult(fallback, "默认阈值（样本不足）", len(intra), len(inter))
        else:
            # Include points just below impostor distances: acceptance is <= T.
            candidates = sorted(set(intra + inter + [fallback, minimum, maximum]
                                    + [float(np.nextafter(value, -np.inf)) for value in inter]))
            best: tuple[float, float, float, float, float] | None = None
            for candidate in candidates:
                threshold = max(minimum, min(maximum, candidate))
                false_accept_rate = sum(item <= threshold for item in inter) / len(inter)
                false_reject_rate = sum(item > threshold for item in intra) / len(intra)
                objective = 2.0 * false_accept_rate + false_reject_rate
                ranking = (
                    objective,
                    abs(threshold - fallback),
                    threshold,
                    false_accept_rate,
                    false_reject_rate,
                )
                if best is None or ranking < best:
                    best = ranking
            assert best is not None
            result = CalibrationResult(
                threshold=best[2],
                source="库内配对自动校准",
                intra_count=len(intra),
                inter_count=len(inter),
                false_accept_rate=best[3],
                false_reject_rate=best[4],
            )
        with self._lock:
            metadata = copy.deepcopy(self._metadata)
            metadata["calibration"] = {
                "threshold": result.threshold,
                "source": result.source,
                "intra_count": result.intra_count,
                "inter_count": result.inter_count,
                "false_accept_rate": result.false_accept_rate,
                "false_reject_rate": result.false_reject_rate,
                "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            }
            self._commit(metadata)
        return result

    def sample_count(self) -> int:
        """Return total enrolled sample count."""

        return sum(person.sample_count for person in self.list_people())

    def iter_feature_vectors(self) -> Iterable[FeatureVector]:
        """Yield all cached features; mainly useful for diagnostics."""

        for _, vectors in self.feature_sets():
            yield from vectors
