"""Standalone reproducible 1:1 experiment; temporary storage, no camera access."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

from ._baseline import PROJECT, FaceComparisonSystem, StorageConfig, load_config
from .evaluation.artifacts import RunJournal, artifact_hashes, new_result_directory
from .evaluation.metrics import verification_report
from .evaluation.pairs import score_pair_manifest
from .evaluation.reporting import _axes, _canvas, _csv, _roc_curve, _save_png, _write_json
from .evaluation.verification_calibration import calibrate_verification
from .experiment import _sha256_file, _source_hashes


def _public_split(scored: dict, threshold: float) -> dict:
    report = verification_report(scored["pairs"], threshold) if scored["pairs"] else None
    failures = scored["acquisition_failures"]
    counts = (report or {}).get("operating_point", {})
    genuine_total = counts.get("genuine_count", 0) + failures["genuine"]
    impostor_total = counts.get("impostor_count", 0) + failures["impostor"]
    return {
        "total_pairs": scored["total_pairs"], "usable_pairs": scored["usable_pairs"],
        "subject_count": len(scored["subject_ids"]),
        "image_count": len(scored["image_hashes"]),
        "acquisition_failures": failures,
        "session_metadata": scored["session_metadata"],
        "missing_session_pairs": scored["missing_session_pairs"],
        "report": report, "records": scored["records"],
        "all_attempts": {
            "genuine_count": genuine_total, "impostor_count": impostor_total,
            "true_accept_count": counts.get("true_accept_count", 0),
            "false_accept_count": counts.get("false_accept_count", 0),
            "tar": counts.get("true_accept_count", 0) / genuine_total if genuine_total else None,
            "false_accept_fraction": counts.get("false_accept_count", 0) / impostor_total if impostor_total else None,
            "note": "失败单独计数；全尝试分母不可把无法采集的impostor当成成功拒绝；ROC仅基于可用pair",
        },
    }


def run_verification_experiment(
    validation_manifest: str | Path, test_manifest: str | Path, *,
    target_far: float = 0.05, require_session_ids: bool = True,
    config_path: str | Path | None = None, on_freeze=None, on_test_open=None,
) -> dict:
    """Subject-disjoint validation/test; never choose threshold on test.

    Missing-session exploration must be explicit both in CLI and manifests.
    Test manifest bytes are committed before calibration; test pixels are not
    accessed until a durable freeze callback has returned in the CLI workflow.
    """
    config_path = Path(config_path).resolve() if config_path else PROJECT / "config.json"
    config = load_config(config_path)
    if config.engine.backend != "sface" or config.storage.backend != "sqlite":
        raise ValueError("1:1实验要求SFace与隔离SQLite")
    model_dir = (PROJECT / config.engine.model_directory).resolve()

    def snapshot():
        return {
            "validation_manifest_sha256": _sha256_file(Path(validation_manifest)),
            "test_manifest_sha256": _sha256_file(Path(test_manifest)),
            "config_sha256": _sha256_file(config_path),
            "model_sha256": {name: _sha256_file(model_dir / name) for name in (
                "face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx")},
            "research_code_sha256": _source_hashes(Path(__file__).parent),
            "course_code_sha256": _source_hashes(PROJECT / "face_compare"),
        }

    provenance = snapshot()
    with tempfile.TemporaryDirectory(prefix="face-verification-") as temporary, ExitStack() as cleanup:
        isolated = replace(config, engine=replace(config.engine, model_directory=str(model_dir)),
                           storage=StorageConfig(str(Path(temporary) / "data"),
                                                 str(Path(temporary) / "events.jsonl"), False, "sqlite"))
        system = FaceComparisonSystem(isolated, temporary)
        cleanup.callback(system.database.close)
        validation = score_pair_manifest(system, validation_manifest, split="validation",
                                         require_session_ids=require_session_ids)
        protocol = validation["protocol_spec"]
        if protocol is not None and (
            not isinstance(protocol, dict) or protocol.get("target_empirical_far") != target_far
        ):
            raise ValueError("配对协议的预定FAR目标与运行参数不一致")
        point = calibrate_verification(validation["pairs"], split="validation", target_far=target_far)
        if snapshot() != provenance:
            raise ValueError("冻结前代码、模型、配置或清单发生变化")
        receipt = {
            "schema": "verification-validation-freeze-v1", **provenance,
            "require_session_ids": require_session_ids, "identity_disjoint_splits": True,
            "target_empirical_far": target_far,
            "tie_rule": "max true accepts, min false accepts, min threshold",
            "calibration": asdict(point),
            "validation_image_hashes": sorted(validation["image_hashes"]),
            "validation_file_hashes": sorted(validation["input_files"].values()),
            "pair_protocol_sha256": hashlib.sha256(
                json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest() if protocol is not None else None,
        }
        if on_freeze:
            on_freeze(json.loads(json.dumps(receipt, allow_nan=False)))
        if on_test_open:
            on_test_open()
        # Check callback-time changes before opening test pixels as well as at end.
        if snapshot() != provenance:
            raise ValueError("冻结后代码、模型、配置或清单发生变化")
        test = score_pair_manifest(
            system, test_manifest, split="test", require_session_ids=require_session_ids,
            forbidden_hashes=validation["image_hashes"],
            forbidden_aligned_hashes=validation["aligned_hashes"],
            forbidden_sessions=validation["sessions"], forbidden_subjects=validation["subject_ids"],
        )
        if test["protocol_spec"] != protocol:
            raise ValueError("验证/测试配对协议定义不一致")
        if snapshot() != provenance or any(
            _sha256_file(path) != digest for scored in (validation, test)
            for path, digest in scored["input_files"].items()
        ):
            raise ValueError("运行期间输入或代码发生变化，拒绝发布混合版本结果")
        return {
            "protocol": "subject-disjoint-verification-v1", "frozen_validation": receipt,
            "feature_signature": system.database.feature_signature,
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "opencv": cv2.__version__, "numpy": np.__version__},
            "threshold_from_validation": point.threshold,
            "validation": _public_split(validation, point.threshold),
            "test": _public_split(test, point.threshold),
            "test_at_fixed_engineering_threshold": _public_split(test, (1 - config.engine.cosine_threshold) / 2),
            "limitations": [
                "1:1验证，不是1:N开放集FPIR；仅在验证集选阈值，test ROC是描述性曲线而非重新选部署参数",
                "同人参与genuine和impostor产生相关性；Wilson仅作描述性近似，不是人群风险保证",
                "缺会话探索模式不生成独立attempt ID，TAR@低FAR保持insufficient samples",
                "身份隔离依赖源标签，精确解码/对齐哈希无法证明不存在近重复或同人多ID",
                "公开裁剪图不能证明真实视频/相机泛化，亦无法排除预训练数据重叠",
            ],
        }


def write_verification_results(report: dict, output: str | Path) -> Path:
    json.dumps(report, allow_nan=False)
    destination = Path(output).resolve()
    with new_result_directory(destination) as staging:
        _write_json(staging / "metrics.json", report)
        receipt = report["frozen_validation"]
        _write_json(staging / "frozen_validation.json", receipt)
        _write_json(staging / "experiment_config.json", {
            "protocol": report["protocol"], "frozen_validation": receipt,
            "environment": report["environment"], "limitations": report["limitations"],
        })
        rows = []
        records = []
        for split in ("validation", "test", "test_at_fixed_engineering_threshold"):
            data = report[split]
            rows.append({"split": split, "total_pairs": data["total_pairs"],
                         "usable_pairs": data["usable_pairs"],
                         "genuine_acquisition_failures": data["acquisition_failures"]["genuine"],
                         "impostor_acquisition_failures": data["acquisition_failures"]["impostor"],
                         **((data["report"] or {}).get("operating_point") or {})})
            if split in {"validation", "test"}:
                records.extend({"split": split, **record} for record in data["records"])
        _csv(staging / "metrics.csv", rows, list(rows[0]))
        _csv(staging / "pairs.csv", records, list(records[0]))
        sweep = receipt["calibration"]["sweep"]
        _csv(staging / "threshold_selection.csv", sweep, list(sweep[0]))
        _roc_curve(staging / "roc_curve.png", report["test"])
        canvas = _canvas("1:1 threshold selection (validation usable pairs)")
        left, right, top, bottom = _axes(canvas, "threshold", "rate")
        for field, color, label_x in (("tar", (50, 120, 20), 100), ("far", (160, 60, 80), 350)):
            coords = [(left, bottom)]
            for row in sweep:
                x = round(left + (right - left) * row["threshold"])
                y = round(bottom - (bottom - top) * row[field])
                coords.extend(((x, coords[-1][1]), (x, y)))
            coords.append((right, coords[-1][1]))
            cv2.polylines(canvas, [np.asarray(coords, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
            cv2.putText(canvas, field.upper(), (label_x, 570), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
        threshold = report["threshold_from_validation"]
        x = round(left + (right - left) * threshold)
        cv2.line(canvas, (x, top), (x, bottom), (60, 60, 60), 1)
        cv2.putText(canvas, f"frozen tau={threshold:.4f}; empirical target FAR={receipt['target_empirical_far']:.3f}",
                    (90, 92), cv2.FONT_HERSHEY_SIMPLEX, .45, (60, 60, 60), 1)
        _save_png(staging / "threshold_curve.png", canvas)
        _write_json(staging / "artifacts.json", {
            "schema": "verification-artifacts-v1", "sha256": artifact_hashes(staging),
        })
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--target-far", type=float, default=.05)
    parser.add_argument("--allow-missing-sessions", action="store_true",
                        help="明确声明缺少会话的公开探索，不支持低FAR推断")
    args = parser.parse_args()
    with RunJournal(args.output, kind="verification") as journal:
        report = run_verification_experiment(
            args.validation, args.test, target_far=args.target_far,
            config_path=args.config, require_session_ids=not args.allow_missing_sessions,
            on_freeze=journal.freeze, on_test_open=journal.test_opening,
        )
        destination = write_verification_results(report, args.output)
        journal.completed(destination)
    print(json.dumps({"output": str(destination), "threshold": report["threshold_from_validation"],
                      "test_usable_pairs": report["test"]["usable_pairs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
