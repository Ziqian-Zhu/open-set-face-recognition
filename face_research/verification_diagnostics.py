"""Explain failures on an existing verification run without tuning or new data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from ._baseline import PROJECT, FaceComparisonSystem, StorageConfig, load_config
from .acquisition import AcquisitionInspector
from .evaluation.artifacts import RunJournal, artifact_hashes, new_result_directory, verify_result_directory
from .evaluation.grouped import grouped_verification
from .evaluation.pairs import score_pair_manifest
from .evaluation.reporting import _csv, _write_json
from .experiment import _sha256_file, _source_hashes


PROTOCOL = Path(__file__).parent / "protocols" / "verification_diagnostics_v1.json"


def _image_summary(images: list[dict]) -> dict:
    reasons = Counter(reason for row in images if row["status"] == "quality_rejected"
                      for reason in row["quality"]["reasons"])
    measurements = {}
    for key in ("brightness", "contrast", "blur_variance", "face_size_ratio", "focus_score"):
        values = [row["quality"][key] for row in images
                  if row["quality"] is not None and row["quality"].get(key) is not None]
        if any(not np.isfinite(value) for value in values):
            raise ValueError("质量测量包含非有限数，不发布")
        measurements[key] = {"count": len(values),
                             "p25_p50_p75": np.quantile(values, [.25, .5, .75]).tolist() if values else None}
    return {"unique_images": len(images), "subjects": len({row["subject"] for row in images}),
            "status_counts": dict(sorted(Counter(row["status"] for row in images).items())),
            "quality_reason_counts_overlapping": dict(sorted(reasons.items())),
            "measurements_on_quality_assessed_images": measurements}


def summarize_diagnostics(scored: dict, manifest: dict, details: dict, *, split: str,
                          threshold: float, comparator: float, bootstrap: dict) -> dict:
    """Join by decoded hashes; subjects are pseudonymized before publication."""
    if len(scored["records"]) != len(manifest["pairs"]):
        raise ValueError("诊断与清单配对数量不同")
    images, pairs = {}, []
    for record, metadata in zip(scored["records"], manifest["pairs"], strict=True):
        joined = dict(record)
        statuses = []
        for side in ("left", "right"):
            digest = record[f"{side}_sha256"]
            if digest not in details or details[digest]["image_sha256"] != digest:
                raise ValueError("缺少同图采集诊断")
            condition = metadata.get(f"{side}_condition", "unspecified")
            if condition not in {"glasses", "no_glasses", "unspecified"}:
                raise ValueError("未声明的外观条件；不从文件名或模型结果推测")
            subject = hashlib.sha256(
                f"{split}\0{metadata[f'{side}_subject_id'].strip().casefold()}".encode()
            ).hexdigest()
            image = {**details[digest], "subject": subject, "condition": condition}
            if images.setdefault(digest, image) != image:
                raise ValueError("同一图像的人员/外观诊断标签冲突")
            joined[f"{side}_subject"] = subject
            joined[f"{side}_status"] = image["status"]
            statuses.append(image["status"])
        usable = all(status == "usable" for status in statuses)
        if usable != (record["status"] == "scored"):
            raise ValueError("诊断终点与正式采集判定不一致")
        failed = [status != "usable" for status in statuses]
        joined["failure_side"] = "both" if all(failed) else "left" if failed[0] else "right" if failed[1] else "none"
        joined["failure_combination"] = "+".join(sorted(set(status for status in statuses if status != "usable"))) or "none"
        joined["accepted"] = usable and record["distance"] <= threshold
        pairs.append(joined)
    if set(images) != set(details):
        raise ValueError("诊断包含非本清单图像")
    image_rows = [images[digest] for digest in sorted(images)]
    # Only exactly one image per appearance yields a within-subject contrast.
    subjects = defaultdict(lambda: defaultdict(list))
    for row in image_rows:
        subjects[row["subject"]][row["condition"]].append(row["status"] == "usable")
    paired = Counter({"both_usable": 0, "glasses_only_usable": 0,
                      "no_glasses_only_usable": 0, "neither_usable": 0, "ineligible_subjects": 0})
    for conditions in subjects.values():
        if len(conditions["glasses"]) != 1 or len(conditions["no_glasses"]) != 1:
            paired["ineligible_subjects"] += 1
            continue
        glasses, bare = conditions["glasses"][0], conditions["no_glasses"][0]
        key = "both_usable" if glasses and bare else "glasses_only_usable" if glasses else "no_glasses_only_usable" if bare else "neither_usable"
        paired[key] += 1
    failures = {}
    for genuine, name in ((True, "genuine"), (False, "impostor")):
        selected = [row for row in pairs if row["genuine"] == genuine]
        failures[name] = {"total_pairs": len(selected),
                          "failure_side_counts": dict(sorted(Counter(row["failure_side"] for row in selected).items())),
                          "failure_combination_counts": dict(sorted(Counter(row["failure_combination"] for row in selected).items()))}
    return {"images": image_rows, "pairs": pairs,
            "all_images": _image_summary(image_rows),
            "by_condition": {condition: _image_summary([row for row in image_rows if row["condition"] == condition])
                             for condition in sorted({row["condition"] for row in image_rows})},
            "within_subject_appearance_usability": dict(paired),
            "pair_failures": failures,
            "grouped_statistics": grouped_verification(pairs, threshold, comparator, **bootstrap)}


def run_diagnostics(baseline_directory, validation_manifest, test_manifest, *,
                    config_path=None, on_plan=None, on_test_open=None) -> dict:
    """Reuse and verify an immutable baseline; reject any score/status drift."""
    baseline_directory = Path(baseline_directory).resolve()
    verify_result_directory(baseline_directory)
    baseline_hashes = artifact_hashes(baseline_directory)
    baseline = json.loads((baseline_directory / "metrics.json").read_text())
    if baseline.get("protocol") != "subject-disjoint-verification-v1":
        raise ValueError("只支持完整的独立1:1实验基线")
    plan = json.loads(PROTOCOL.read_text())
    receipt = baseline["frozen_validation"]
    threshold = baseline["threshold_from_validation"]
    if threshold != receipt["calibration"]["threshold"]:
        raise ValueError("基线阈值与冻结凭据不一致")
    config_path = Path(config_path).resolve() if config_path else PROJECT / "config.json"
    config = load_config(config_path)
    if config.engine.backend != "sface" or config.storage.backend != "sqlite":
        raise ValueError("诊断要求原SFace路径与隔离SQLite")
    model_dir = (PROJECT / config.engine.model_directory).resolve()
    manifests = {"validation": Path(validation_manifest), "test": Path(test_manifest)}

    def snapshot():
        return {"config_sha256": _sha256_file(config_path), "protocol_sha256": _sha256_file(PROTOCOL),
                "manifest_sha256": {split: _sha256_file(path) for split, path in manifests.items()},
                "model_sha256": {name: _sha256_file(model_dir / name) for name in receipt["model_sha256"]},
                "research_code_sha256": _source_hashes(Path(__file__).parent),
                "course_code_sha256": _source_hashes(PROJECT / "face_compare")}

    provenance = snapshot()
    for key in ("config_sha256", "model_sha256"):
        if provenance[key] != receipt[key]:
            raise ValueError("诊断的配置/模型必须与基线完全相同")
    if any(provenance["manifest_sha256"][split] != receipt[f"{split}_manifest_sha256"] for split in manifests):
        raise ValueError("诊断必须使用原始完整清单，不能换图或补选")
    fixed = (1 - config.engine.cosine_threshold) / 2
    frozen_plan = {"schema": "posthoc-diagnostic-plan-v1", "plan": plan, **provenance,
                   "baseline_artifact_sha256": baseline_hashes,
                   "threshold_reused_from_baseline": threshold, "fixed_engineering_threshold": fixed,
                   "test_already_used": True, "calibration_performed": False}
    if on_plan:
        on_plan(json.loads(json.dumps(frozen_plan, allow_nan=False)))
    if snapshot() != provenance:
        raise ValueError("诊断计划记录后代码/输入发生变化")
    scored, outputs = {}, {}
    with tempfile.TemporaryDirectory(prefix="face-diagnostics-") as temporary, ExitStack() as cleanup:
        isolated = replace(config, engine=replace(config.engine, model_directory=str(model_dir)),
                           storage=StorageConfig(str(Path(temporary) / "data"),
                                                 str(Path(temporary) / "events.jsonl"), False, "sqlite"))
        system = FaceComparisonSystem(isolated, temporary)
        cleanup.callback(system.database.close)
        if system.database.feature_signature != baseline["feature_signature"]:
            raise ValueError("特征签名与基线不一致")
        for split, path in manifests.items():
            if split == "test" and on_test_open:
                on_test_open()
            if snapshot() != provenance:
                raise ValueError("读取图像前代码/输入发生变化")
            earlier = scored.get("validation", {})
            with AcquisitionInspector(system) as inspector:
                result = score_pair_manifest(
                    inspector, path, split=split, require_session_ids=receipt["require_session_ids"],
                    forbidden_hashes=earlier.get("image_hashes"),
                    forbidden_aligned_hashes=earlier.get("aligned_hashes"),
                    forbidden_sessions=earlier.get("sessions"), forbidden_subjects=earlier.get("subject_ids"))
            if result["records"] != baseline[split]["records"]:
                raise ValueError("配对分数/状态与原基线不同，拒绝把变化冒充诊断")
            outputs[split] = summarize_diagnostics(
                result, json.loads(path.read_text()), inspector.records, split=split,
                threshold=threshold, comparator=fixed, bootstrap=plan["bootstrap"])
            scored[split] = result
        if snapshot() != provenance or any(_sha256_file(path) != digest
                                          for result in scored.values() for path, digest in result["input_files"].items()):
            raise ValueError("运行期间图像、代码或配置发生变化")
        verify_result_directory(baseline_directory)
        if artifact_hashes(baseline_directory) != baseline_hashes:
            raise ValueError("基线产物在运行期间发生变化")
    return {"protocol": plan["id"], "diagnostic_plan": frozen_plan,
            "baseline_pair_records_identical": True, "splits": outputs,
            "environment": {"python": platform.python_version(), "opencv": cv2.__version__,
                            "numpy": np.__version__, "platform": platform.platform()},
            "limitations": ["事后诊断已使用的test，不是新盲测；不更改阈值、不补图、不绕过任何门控",
                            "未执行下游阶段就不推断其潜在结果；no_face是正式检测器输出为空，不能区分网络未检出与内部尺寸/关键点过滤",
                            "质量原因可能重叠，不可相加当作独立失败图片数；外观分层不是眼镜的因果实验",
                            "仅本轮伪名和哈希用于复核，不等于不可重新关联；对外展示优先用聚合统计",
                            "真实运动、跨会话和低FAR仍未验证，不自动改变默认产品行为"]}


def write_diagnostics(report: dict, output) -> Path:
    json.dumps(report, allow_nan=False)
    destination = Path(output).resolve()
    with new_result_directory(destination) as staging:
        _write_json(staging / "metrics.json", report)
        _write_json(staging / "diagnostic_plan.json", report["diagnostic_plan"])
        _write_json(staging / "experiment_config.json", {key: report[key] for key in ("protocol", "environment", "limitations")})
        images, pairs = [], []
        for split, data in report["splits"].items():
            for row in data["images"]:
                images.append({"split": split, "image_sha256": row["image_sha256"], "subject": row["subject"],
                               "condition": row["condition"], "status": row["status"], "last_stage": row["last_stage"],
                               "detected_faces": row["detected_faces"],
                               "quality_json": json.dumps(row["quality"], ensure_ascii=False, allow_nan=False),
                               "stage_calls_json": json.dumps(row["stage_calls"], sort_keys=True)})
            pairs.extend({"split": split, **row} for row in data["pairs"])
        _csv(staging / "images.csv", images, list(images[0]))
        _csv(staging / "pairs.csv", pairs, list(pairs[0]))
        _write_json(staging / "artifacts.json", {"schema": "diagnostic-artifacts-v1", "sha256": artifact_hashes(staging)})
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    with RunJournal(args.output, kind="posthoc-verification-diagnostics") as journal:
        report = run_diagnostics(args.baseline, args.validation, args.test, config_path=args.config,
                                 on_plan=journal.freeze, on_test_open=journal.test_opening)
        destination = write_diagnostics(report, args.output)
        journal.completed(destination)
    print(json.dumps({"output": str(destination), "baseline_pair_records_identical": True,
                      "test_images": report["splits"]["test"]["all_images"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
