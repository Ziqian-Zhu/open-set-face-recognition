"""Validated, image-lazy research manifests and split-integrity checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from ._baseline import normalize_person_name


def _decoded_hash(image: NDArray[np.uint8]) -> str:
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def _dhash64(image: NDArray[np.uint8]) -> int:
    """A cheap near-duplicate *warning* signal, not identity evidence."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def load_image(entry: dict[str, Any]) -> NDArray[np.uint8]:
    """Decode one image and reject edits made after manifest validation."""

    image = cv2.imread(entry["path"])
    if image is None:
        raise ValueError(f"无法读取测试图片：{entry['path']}")
    if _decoded_hash(image) != entry["sha256"]:
        raise ValueError(f"图片在清单验证后发生变化：{entry['path']}")
    return image


def read_research_manifest(
    path: str | Path,
) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    """Validate schema and hashes while retaining paths, not decoded frames."""

    manifest_path = Path(path).resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("split") not in {"validation", "test"}:
        raise ValueError("清单split必须为validation或test")
    seen: set[str] = set()
    groups: dict[str, list[dict[str, Any]]] = {}
    for group in ("gallery", "probes"):
        source = raw.get(group)
        if not isinstance(source, list) or not source:
            raise ValueError(f"{group}必须为非空列表")
        loaded = []
        for index, item in enumerate(source):
            if not isinstance(item, dict):
                raise ValueError(f"{group}[{index}]必须为对象")
            relative_path = item.get("image")
            if not isinstance(relative_path, str) or not relative_path.strip():
                raise ValueError(f"{group}[{index}].image必须为非空路径")
            label = item.get("label")
            if label is None and group == "gallery":
                raise ValueError("录入集gallery不能使用未知标签null")
            if label is not None:
                label = normalize_person_name(label)
            condition = item.get("condition", "unspecified")
            if not isinstance(condition, str) or not condition.strip():
                raise ValueError(f"{group}[{index}].condition必须为非空字符串")
            session_id = item.get("session_id")
            if session_id is not None and (
                not isinstance(session_id, str) or not session_id.strip()
            ):
                raise ValueError(f"{group}[{index}].session_id必须为非空字符串")
            image_path = (manifest_path.parent / relative_path).resolve()
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"无法读取测试图片：{image_path}")
            digest = _decoded_hash(image)
            if digest in seen:
                raise ValueError("发现重复解码图像/训练测试泄漏")
            seen.add(digest)
            entry = {
                "path": str(image_path),
                "sha256": digest,
                "dhash64": _dhash64(image),
                "label": label,
                "condition": condition.strip(),
                "session_id": session_id.strip() if session_id is not None else None,
            }
            if label is None:
                subject_id = item.get("subject_id")
                if not isinstance(subject_id, str) or not subject_id.strip():
                    raise ValueError("未知人员probe必须填写非空subject_id")
                entry["subject_id"] = subject_id.strip()
            loaded.append(entry)
        groups[group] = loaded
    names = {entry["label"] for entry in groups["gallery"]}
    name_keys = {name.casefold() for name in names}
    if len(name_keys) != len(names):
        raise ValueError("姓名大小写不一致，请统一清单标签")
    for entry in groups["probes"]:
        if entry["label"] is None:
            if entry["subject_id"].casefold() in name_keys:
                raise ValueError("未知人员subject_id不能与已知人员姓名相同")
        elif entry["label"] not in names:
            raise ValueError("probe中未录入人员必须以label:null标记为未知")
    return raw["split"], groups


def check_manifest_splits(validation: dict, test: dict) -> None:
    """Require identical gallery and independent validation/test source images."""

    fields = ("sha256", "label", "condition", "session_id")
    val_gallery = [tuple(row.get(field) for field in fields) for row in validation["gallery"]]
    test_gallery = [tuple(row.get(field) for field in fields) for row in test["gallery"]]
    if val_gallery != test_gallery:
        raise ValueError("验证/测试清单的gallery必须完全相同，且顺序一致")
    if {row["sha256"] for row in validation["probes"]} & {
        row["sha256"] for row in test["probes"]
    }:
        raise ValueError("验证集与测试集含有相同解码图像，拒绝数据泄漏")
    for name, groups in (("验证", validation), ("测试", test)):
        probes = groups["probes"]
        if not any(row["label"] is None for row in probes):
            raise ValueError(f"{name}集必须包含未知人员(label:null)")
        if not any(row["label"] is not None for row in probes):
            raise ValueError(f"{name}集必须包含已知人员")
    val_unknown = {
        row["subject_id"].casefold()
        for row in validation["probes"]
        if row["label"] is None
    }
    test_unknown = {
        row["subject_id"].casefold() for row in test["probes"] if row["label"] is None
    }
    if val_unknown & test_unknown:
        raise ValueError("验证集与测试集使用了同一未知人员，拒绝身份级泄漏")
    check_session_isolation(validation, test)


def check_session_isolation(
    validation: dict, test: dict, *, require_session_ids: bool = False
) -> None:
    """Reject a subject's capture session reused across gallery/val/test.

    A missing session ID cannot establish independence; strict experiments may
    require one for every image. Identical gallery copies in the two manifests
    represent one training split and are intentionally counted only once.
    """

    splits = (
        ("gallery", validation["gallery"]),
        ("validation", validation["probes"]),
        ("test", test["probes"]),
    )
    seen: dict[tuple[str, str], str] = {}
    for split, rows in splits:
        for row in rows:
            session = row.get("session_id")
            if session is None:
                if require_session_ids:
                    raise ValueError(f"{split}存在未标注session_id的图像，不能证明采集批次独立")
                continue
            subject = row["label"] if row["label"] is not None else row["subject_id"]
            key = (subject.casefold(), session.casefold())
            previous = seen.setdefault(key, split)
            if previous != split:
                raise ValueError(f"同一人员采集批次跨{previous}/{split}复用，拒绝会话级泄漏")


def near_duplicate_warnings(validation: dict, test: dict, *, max_hamming: int = 4) -> list[dict]:
    """Flag similar full frames across splits for manual review.

    This is intentionally advisory: a 64-bit dHash may flag distinct photos
    with similar backgrounds and miss the same face after crop/re-encoding.
    Exact decoded/aligned duplicates remain hard errors elsewhere.
    """

    if type(max_hamming) is not int or not 0 <= max_hamming <= 64:
        raise ValueError("max_hamming必须是0到64的整数")
    sections = (
        ("gallery", validation["gallery"]),
        ("validation", validation["probes"]),
        ("test", test["probes"]),
    )
    warnings = []
    for left_index, (left_split, left_rows) in enumerate(sections):
        for right_split, right_rows in sections[left_index + 1:]:
            for left in left_rows:
                if left.get("label") is None or left.get("dhash64") is None:
                    continue
                for right in right_rows:
                    if right.get("label") != left["label"] or right.get("dhash64") is None:
                        continue
                    distance = (left["dhash64"] ^ right["dhash64"]).bit_count()
                    if distance <= max_hamming:
                        warnings.append({
                            "subject": left["label"],
                            "left_split": left_split,
                            "right_split": right_split,
                            "left_sha256": left["sha256"],
                            "right_sha256": right["sha256"],
                            "dhash_hamming": distance,
                        })
    return warnings
