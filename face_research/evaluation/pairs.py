"""Consent-based, explicitly labeled 1:1 image-pair verification protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .metrics import VerificationPair


def _required_text(row: dict, key: str, index: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"pairs[{index}].{key}必须为非空字符串")
    return value.strip()


def score_pair_manifest(
    system, path: str | Path, *,
    split: str = "test",
    forbidden_hashes: set[str] | None = None,
    forbidden_aligned_hashes: set[str] | None = None,
    forbidden_sessions: set[tuple[str, str]] | None = None,
    forbidden_subjects: set[str] | None = None,
    require_session_ids: bool = True,
) -> dict:
    """Extract aligned SFace embeddings and score split-labeled pairs.

    Unknown quality/detection failures are counted, never silently treated as
    successful impostor rejections. Low-FAR support uses one left-subject/session
    as one independent impostor attempt; repeated sessions remain correlated.
    """

    manifest = Path(path).resolve()
    manifest_bytes = manifest.read_bytes()
    raw = json.loads(manifest_bytes)
    if split not in {"validation", "test"} or not isinstance(raw, dict) or raw.get("split") != split:
        raise ValueError(f"验证对清单必须标记split={split}；测试对不得用于选择参数")
    source = raw.get("pairs")
    if not isinstance(source, list) or not source:
        raise ValueError("验证对清单pairs必须为非空列表")
    if type(require_session_ids) is not bool:
        raise ValueError("require_session_ids必须为布尔值")
    if not require_session_ids and raw.get("session_metadata") != "unavailable":
        raise ValueError("公开探索模式必须显式声明session_metadata=unavailable")
    cache = {}
    failures = {"genuine": 0, "impostor": 0}
    usable: list[VerificationPair] = []
    forbidden = forbidden_hashes or set()
    forbidden_aligned = forbidden_aligned_hashes or set()
    blocked_sessions = forbidden_sessions or set()
    sessions: set[tuple[str, str]] = set()
    image_metadata: dict[str, tuple[str, str | None]] = {}
    aligned_metadata: dict[str, tuple[str, str | None]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    seen_aligned_pairs: set[tuple[str, str]] = set()
    subjects: set[str] = set()
    input_files: dict[Path, str] = {}
    records = []
    missing_session_pairs = 0

    def session_value(row, key, index):
        if not require_session_ids and row.get(key) is None:
            return None
        return _required_text(row, key, index).casefold()

    def embedding(image_path: Path):
        encoded = image_path.read_bytes()
        file_hash = hashlib.sha256(encoded).hexdigest()
        if input_files.setdefault(image_path, file_hash) != file_hash:
            raise ValueError("验证对图片在运行期间发生变化")
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取验证对图片：{image_path}")
        digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        if digest in forbidden:
            raise ValueError("验证对图片与录入/验证集重复，拒绝泄漏")
        if digest not in cache:
            try:
                sample = system.prepare_sample(image)
            except ValueError as exc:
                cache[digest] = (None, str(exc))
            else:
                aligned_digest = hashlib.sha256(sample.crop.tobytes()).hexdigest()
                if aligned_digest in forbidden_aligned:
                    raise ValueError("验证对的对齐人脸与录入/验证集重复，拒绝泄漏")
                cache[digest] = (sample.feature, aligned_digest)
        return digest, cache[digest]

    for index, row in enumerate(source):
        if not isinstance(row, dict) or type(row.get("genuine")) is not bool:
            raise ValueError(f"pairs[{index}]必须有布尔genuine真值")
        left_subject = _required_text(row, "left_subject_id", index)
        right_subject = _required_text(row, "right_subject_id", index)
        left_session = session_value(row, "left_session_id", index)
        right_session = session_value(row, "right_session_id", index)
        pair_subjects = {left_subject.casefold(), right_subject.casefold()}
        if pair_subjects & (forbidden_subjects or set()):
            raise ValueError(f"pairs[{index}]人员与验证集重复，拒绝身份级泄漏")
        subjects.update(pair_subjects)
        missing_session_pairs += left_session is None or right_session is None
        pair_sessions = {
            (subject.casefold(), session)
            for subject, session in ((left_subject, left_session), (right_subject, right_session))
            if session is not None
        }
        if pair_sessions & blocked_sessions:
            raise ValueError(f"pairs[{index}]采集批次与更早的数据集合重复，拒绝会话级泄漏")
        sessions.update(pair_sessions)
        genuine = row["genuine"]
        same_subject = left_subject.casefold() == right_subject.casefold()
        if genuine != same_subject:
            raise ValueError(f"pairs[{index}]的genuine与subject_id矛盾")
        if genuine and left_session is not None and left_session == right_session:
            raise ValueError(f"pairs[{index}]的genuine两侧必须来自不同采集批次")
        left_path = (manifest.parent / _required_text(row, "left_image", index)).resolve()
        right_path = (manifest.parent / _required_text(row, "right_image", index)).resolve()
        left_hash, (left_feature, left_aligned) = embedding(left_path)
        right_hash, (right_feature, right_aligned) = embedding(right_path)
        for digest, aligned, feature, subject, session in (
            (left_hash, left_aligned, left_feature, left_subject, left_session),
            (right_hash, right_aligned, right_feature, right_subject, right_session),
        ):
            identity_session = (subject.casefold(), session)
            if image_metadata.setdefault(digest, identity_session) != identity_session:
                raise ValueError(f"pairs[{index}]同一图像使用了矛盾的人员/批次标签")
            if feature is not None and aligned_metadata.setdefault(aligned, identity_session) != identity_session:
                raise ValueError(f"pairs[{index}]同一对齐人脸使用了矛盾的人员/批次标签")
        if left_hash == right_hash:
            raise ValueError(f"pairs[{index}]左右是同一张图像，拒绝自比对")
        if left_feature is not None and right_feature is not None and left_aligned == right_aligned:
            raise ValueError(f"pairs[{index}]左右对齐人脸完全重复，拒绝自比对")
        pair_key = tuple(sorted((left_hash, right_hash)))
        if pair_key in seen_pairs:
            raise ValueError(f"pairs[{index}]验证对重复（含左右交换），不能重复计数")
        seen_pairs.add(pair_key)
        record = {"pair_index": index, "left_sha256": left_hash, "right_sha256": right_hash,
                  "genuine": genuine, "status": "acquisition_failed", "distance": None}
        records.append(record)
        if left_feature is None or right_feature is None:
            failures["genuine" if genuine else "impostor"] += 1
            continue
        aligned_pair = tuple(sorted((left_aligned, right_aligned)))
        if aligned_pair in seen_aligned_pairs:
            raise ValueError(f"pairs[{index}]对齐验证对重复，不能用重复裁剪扩大样本量")
        seen_aligned_pairs.add(aligned_pair)
        distance = float(system.extractor.distance(left_feature, right_feature))
        if not np.isfinite(distance) or not 0 <= distance <= 1:
            raise ValueError("验证对距离必须是[0,1]内的有限半余弦距离")
        record.update(status="scored", distance=distance)
        attempt_id = (
            f"{left_subject.casefold()}\0{left_session}"
            if not genuine and require_session_ids else None
        )
        usable.append(VerificationPair(distance, genuine, attempt_id))
    if manifest.read_bytes() != manifest_bytes:
        raise ValueError("验证对清单在运行期间发生变化")
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != digest for path, digest in input_files.items()):
        raise ValueError("验证对图片在运行期间发生变化")
    return {
        "pairs": usable,
        "image_hashes": set(cache),
        "aligned_hashes": {value[1] for value in cache.values() if value[0] is not None},
        "sessions": sessions,
        "protocol_spec": raw.get("protocol"),
        "subject_ids": subjects,
        "input_files": input_files,
        "records": records,
        "session_metadata": "required" if require_session_ids else "unavailable_exploratory",
        "missing_session_pairs": missing_session_pairs,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "total_pairs": len(source),
        "usable_pairs": len(usable),
        "acquisition_failures": failures,
        "note": "验证ROC仅基于成功检测/画质合格的pair；失败尝试另列，不当作正确拒绝",
    }
