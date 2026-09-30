"""Independent open-set evaluation; no production database writes or self-test leakage."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import time
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .config import StorageConfig
from .database import normalize_person_name
from .service import FaceComparisonSystem


def interval(successes, total):
    """Wilson 95% interval: zero observed errors is not proof of zero risk."""
    if total == 0:
        return {"count": successes, "total": total, "rate": None, "wilson95": None}
    p, z = successes / total, 1.959963984540054
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return {"count": successes, "total": total, "rate": p,
            "wilson95": [max(0, center-half), min(1, center+half)]}


def summarize(records):
    known = [r for r in records if r["expected"] is not None]
    unknown = [r for r in records if r["expected"] is None]
    successful_unknown = sum(r["status"] == "recognized" and r["predicted"] is None for r in unknown)
    accepted_unknown = sum(r["predicted"] is not None for r in unknown)
    correct_known = sum(r["predicted"] == r["expected"] for r in known)
    rejection = sum(r["predicted"] is None for r in known)
    misid = sum(r["predicted"] is not None and r["predicted"] != r["expected"] for r in known)
    usable_unknown = sum(r["status"] == "recognized" for r in unknown)
    return {
        "total": len(records),
        "status_counts": dict(Counter(r["status"] for r in records)),
        "known_correct": interval(correct_known, len(known)),
        "known_rejected_including_acquisition_failure": interval(rejection, len(known)),
        "known_misidentified": interval(misid, len(known)),
        "known_failed_identification": interval(rejection+misid, len(known)),
        "unknown_false_accept_all_attempts": interval(accepted_unknown, len(unknown)),
        "unknown_false_accept_usable_faces": interval(accepted_unknown, usable_unknown),
        "unknown_correct_rejection_all_attempts": interval(successful_unknown, len(unknown)),
        "latency_ms": {"median": float(np.median([r["ms"] for r in records])) if records else None,
                       "p95": float(np.percentile([r["ms"] for r in records], 95)) if records else None},
    }


def read_manifest(path):
    path = Path(path).resolve()
    content = json.loads(path.read_text(encoding="utf-8"))
    if content.get("split") not in ("validation", "test"):
        raise ValueError("清单split必须为validation或test；参数选择后另用test评估")
    seen = set()
    groups = {}
    for group in ("gallery", "probes"):
        entries = content.get(group)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{group}必须为非空列表")
        loaded = []
        for entry in entries:
            label = entry.get("label")
            if label is not None:
                label = normalize_person_name(label)
            elif group == "gallery":
                raise ValueError("录入集gallery不能使用未知标签null")
            candidate = (path.parent / entry["image"]).resolve()
            image = cv2.imread(str(candidate))
            if image is None:
                raise ValueError(f"无法读取测试图片：{candidate}")
            digest = hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
            if digest in seen:
                raise ValueError("发现重复图像/训练测试泄漏；相同解码图像不可重复计数")
            seen.add(digest)
            loaded.append({"image": image, "path": str(candidate), "label": label, "sha256": digest,
                           "condition": entry.get("condition", "unspecified")})
        groups[group] = loaded
    identities = {item["label"].casefold() for item in groups["gallery"]}
    if len(identities) != len({item["label"] for item in groups["gallery"]}):
        raise ValueError("姓名大小写不一致，请统一清单标签")
    for entry in groups["probes"]:
        if entry["label"] is not None and entry["label"] not in {item["label"] for item in groups["gallery"]}:
            raise ValueError("probe中未录入人员必须以label:null标记为未知")
    return content["split"], groups


def evaluate(config, project_root, manifest):
    split, groups = read_manifest(manifest)
    model_dir = (Path(project_root) / config.engine.model_directory).resolve()
    with tempfile.TemporaryDirectory(prefix="face-evaluation-") as temporary, ExitStack() as cleanup:
        isolated = replace(config, engine=replace(config.engine, model_directory=str(model_dir)),
                           storage=StorageConfig(str(Path(temporary)/"data"), str(Path(temporary)/"events.jsonl"), False, config.storage.backend))
        system = FaceComparisonSystem(isolated, temporary)
        if hasattr(system.database, "close"):
            cleanup.callback(system.database.close)
        gallery_hashes = set()
        for entry in groups["gallery"]:
            sample = system.prepare_sample(entry["image"])
            # Also reject an enrollment crop re-exported as a differently-sized test image.
            gallery_hashes.add(hashlib.sha256(sample.crop.tobytes()).hexdigest())
            system.enroll_samples(entry["label"], [sample])
        records = []
        for entry in groups["probes"]:
            start = time.perf_counter()
            observations = system.analyze_frame(entry["image"], log_events=False)
            record = {"image": entry["path"], "sha256": entry["sha256"], "condition": entry["condition"],
                      "expected": entry["label"], "predicted": None, "distance": None,
                      "threshold": system.recognizer.threshold}
            if not observations:
                record["status"] = "no_face"
            elif len(observations) != 1:
                record["status"] = "multiple_faces"
            elif not observations[0].quality.accepted:
                record["status"] = "quality_rejected"
            else:
                sample = system.prepare_sample(entry["image"])
                if hashlib.sha256(sample.crop.tobytes()).hexdigest() in gallery_hashes:
                    raise ValueError("测试人脸裁剪与录入裁剪完全相同，拒绝自测泄漏")
                result = observations[0].recognition
                record.update(status="recognized", predicted=result.name if result.known else None,
                              distance=result.distance, reason=result.reason, candidate_name=result.candidate_name,
                              second_best_distance=result.second_best_distance)
            record["ms"] = (time.perf_counter()-start)*1000
            records.append(record)
        report = {"split": split, "engine": system.engine_label, "feature_signature": system.database.feature_signature,
                  "opencv": cv2.__version__, "threshold": system.recognizer.threshold,
                  "margin": system.recognition_config.ambiguity_margin,
                  "gallery_samples": len(groups["gallery"]), "gallery_people": len(system.database.list_people()),
                  "summary": summarize(records),
                  "by_condition": {c: summarize([r for r in records if r["condition"] == c])
                                   for c in sorted({r["condition"] for r in records})},
                  "records": records,
                  "limitations": ["仅静态图片；延迟包含检测、质量、识别和泄漏复核，不包含视频多帧确认", 
                                   "字节去重不能发现近重复连拍，应由采集者保证不同拍摄批次", 
                                   "测试集不能反复用于调阈值；置信区间假定试验近似独立"]}
        return report
