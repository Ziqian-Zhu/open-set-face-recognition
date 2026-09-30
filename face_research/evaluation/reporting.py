"""Create-only, privacy-conscious experiment artifacts without extra plotting deps."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .artifacts import artifact_hashes, new_result_directory


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _metric_rate(summary: dict, name: str):
    return summary[name]["rate"]


def _csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _canvas(title: str) -> np.ndarray:
    canvas = np.full((600, 900, 3), 250, dtype=np.uint8)
    cv2.putText(canvas, title, (65, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (36, 46, 46), 2, cv2.LINE_AA)
    return canvas


def _save_png(path: Path, canvas: np.ndarray) -> None:
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"无法生成图片：{path}")


def _axes(canvas: np.ndarray, x_label: str, y_label: str) -> tuple[int, int, int, int]:
    """Draw explicit [0, 1] axes so a saved figure has readable numeric units."""

    left, right, top, bottom = 90, 825, 115, 500
    for tick in np.linspace(0, 1, 6):
        x, y = round(left + (right - left) * tick), round(bottom - (bottom - top) * tick)
        cv2.line(canvas, (x, top), (x, bottom), (226, 226, 226), 1)
        cv2.line(canvas, (left, y), (right, y), (226, 226, 226), 1)
        cv2.putText(canvas, f"{tick:.1f}", (x - 12, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
        cv2.putText(canvas, f"{tick:.1f}", (left - 33, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
    cv2.line(canvas, (left, bottom), (right, bottom), (70, 70, 70), 2)
    cv2.line(canvas, (left, bottom), (left, top), (70, 70, 70), 2)
    cv2.putText(canvas, x_label, (right - 105, bottom + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (50, 50, 50), 1)
    cv2.putText(canvas, y_label, (left - 45, top - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1)
    return left, right, top, bottom


def _roc_curve(path: Path, verification: dict) -> None:
    points = (verification.get("report") or {}).get("roc") or []
    if not points:
        canvas = _canvas("Verification ROC: NOT EVALUATED")
        cv2.putText(canvas, "No usable independent genuine + impostor pair protocol.",
                    (70, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 150), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Open-set FPIR is not verification FAR.",
                    (70, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 150), 1, cv2.LINE_AA)
        _save_png(path, canvas)
        return
    canvas = _canvas("Verification ROC (usable test pairs only)")
    left, right, top, bottom = _axes(canvas, "FAR", "TAR")
    coordinates = np.asarray([
        (round(left + (right - left) * point["far"]), round(bottom - (bottom - top) * point["tar"]))
        for point in points
    ], dtype=np.int32)
    cv2.polylines(canvas, [coordinates], False, (130, 75, 20), 2, cv2.LINE_AA)
    auc = verification["report"]["auc"]
    cv2.putText(canvas, f"AUC={auc:.4f}; descriptive, not a deployment guarantee",
                (90, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    _save_png(path, canvas)


def _threshold_curve(path: Path, report: dict) -> None:
    variants = report["variants"]
    preferred = next(
        (
            key for key, value in variants.items()
            if value["budget"] == 3 and value["selection"] == "first"
            and value["aggregation"] == "top3_median"
        ),
        sorted(variants)[0],
    )
    variant = variants[preferred]
    margin = variant["margin_from_validation"]
    rows = sorted(
        (
            row for row in variant["threshold_selection"]
            if abs(row["margin"] - margin) < 1e-12
        ),
        key=lambda row: row["threshold"],
    )
    canvas = _canvas("Open-set threshold curve (validation only)")
    left, right, top, bottom = _axes(canvas, "threshold", "rate")
    cv2.putText(canvas, "known correct / all known", (120, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 120, 20), 2)
    cv2.putText(canvas, "FPIR / usable unknown", (480, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 60, 80), 2)
    for field, color in (("known_correct_rate", (50, 120, 20)), ("fpir_usable", (160, 60, 80))):
        # Acceptance changes at observed score events, not linearly in between.
        coordinates = []
        previous_y = bottom
        for row in rows:
            x = round(left + (right - left) * row["threshold"])
            y = round(bottom - (bottom - top) * row[field])
            coordinates.extend(((x, previous_y), (x, y)))
            previous_y = y
        coordinates.append((right, previous_y))
        points = np.asarray(coordinates, dtype=np.int32)
        if len(points) >= 2:
            cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
    frozen_x = round(left + (right - left) * variant["threshold_from_validation"])
    for y in range(top, bottom, 12):
        cv2.line(canvas, (frozen_x, y), (frozen_x, min(y + 6, bottom)), (85, 85, 85), 1)
    cv2.putText(canvas, f"frozen threshold={variant['threshold_from_validation']:.4f}",
                (470, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (60, 60, 60), 1)
    cv2.putText(canvas, f"variant={preferred}; margin={margin:.4f}",
                (90, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (80, 80, 80), 1)
    _save_png(path, canvas)


def _write_matrix_results(report: dict, destination: Path) -> None:
    """Build a complete artifact set inside an unpublished staging directory."""

    variants = report["variants"]
    if not variants:
        raise ValueError("实验结果没有任何组合")
    config = {key: value for key, value in report.items() if key not in {
        "variants", "limitations", "verification", "frozen_validation"
    }}
    config["metric_definitions"] = {
        "threshold": "half_cosine_distance; lower is more similar",
        "margin": "second_best_identity_distance - best_identity_distance",
        "FPIR": "accepted unknown / unknown probes with usable face",
        "FAR": "verification false accepts / usable impostor pairs; null without pairs; pair count does not prove independence",
        "FRR_TAR": "verification genuine-pair metrics at validation-frozen threshold; null without pairs",
        "unknown_rejection_rate": "all unknown attempts not accepted, including acquisition failures; not successful identification",
        "unknown_correct_rejection_rate": "usable unknown faces not accepted / all unknown attempts; acquisition failures are not correct rejections",
        "accuracy_all_attempts": "(correct known accepts + usable unknown faces not accepted) / all attempts; every acquisition failure is unsuccessful; depends on known/unknown mix",
        "confidence_intervals": "Wilson 95% descriptive approximation; repeated subjects/sessions are not independent trials",
    }
    _write_json(destination / "experiment_config.json", config)
    _write_json(destination / "metrics.json", report)
    if "frozen_validation" in report:
        _write_json(destination / "frozen_validation.json", report["frozen_validation"])
    if "primary_comparison" in report:
        _write_json(destination / "primary_comparison.json", report["primary_comparison"])
    metric_rows = []
    threshold_rows = []
    for key, value in variants.items():
        summary = value["test"]["identification"]
        pair_metric = value["test"].get("verification_at_frozen_threshold") or {}
        metric_rows.append({
            "variant": key,
            "budget": value["budget"],
            "selection": value["selection"],
            "random_seed": value["seed"],
            "aggregation": value["aggregation"],
            "threshold": value["threshold_from_validation"],
            "margin": value["margin_from_validation"],
            "known_attempts": summary["known_attempts"],
            "unknown_attempts": summary["unknown_attempts"],
            "usable_known_attempts": summary["usable_known_attempts"],
            "usable_unknown_attempts": summary["usable_unknown_attempts"],
            "top1_usable_known": _metric_rate(summary, "top1_closed_set_usable_known"),
            "known_recognition_rate": _metric_rate(summary, "dir_known_correct_all_attempts"),
            "known_reject_rate": _metric_rate(summary, "known_reject_all_attempts"),
            "known_misidentification_rate": _metric_rate(summary, "known_misidentification_all_attempts"),
            "FPIR_usable_unknown": _metric_rate(summary, "fpir_unknown_usable_faces"),
            "unknown_rejection_rate": _metric_rate(summary, "unknown_rejection_all_attempts"),
            "unknown_correct_rejection_rate": _metric_rate(summary, "unknown_correct_rejection_all_attempts"),
            "accuracy_all_attempts": _metric_rate(summary, "accuracy_all_attempts"),
            "FAR": pair_metric.get("far"),
            "FRR": pair_metric.get("frr"),
            "TAR": pair_metric.get("tar"),
            "verification_status": "evaluated" if pair_metric else "not_evaluated",
            "verification_reason": "" if pair_metric else "no usable independent 1:1 test pair protocol supplied",
            "verification_genuine_pairs": pair_metric.get("genuine_count"),
            "verification_impostor_pairs": pair_metric.get("impostor_count"),
            "verification_true_accepts": pair_metric.get("true_accept_count"),
            "verification_false_accepts": pair_metric.get("false_accept_count"),
        })
        for prefix, name in (
            ("known_correct", "dir_known_correct_all_attempts"),
            ("known_rejected", "known_reject_all_attempts"),
            ("known_misidentified", "known_misidentification_all_attempts"),
            ("unknown_false_accept_all", "fpir_unknown_all_attempts"),
            ("unknown_false_accept_usable", "fpir_unknown_usable_faces"),
            ("unknown_correct_rejection", "unknown_correct_rejection_all_attempts"),
            ("unknown_acquisition_failures", "unknown_acquisition_failures_all_attempts"),
            ("known_acquisition_failures", "known_acquisition_failures_all_attempts"),
            ("correct_all_attempts", "accuracy_all_attempts"),
        ):
            metric = summary[name]
            metric_rows[-1].update({
                f"{prefix}_count": metric["count"],
                f"{prefix}_total": metric["total"],
                f"{prefix}_wilson95_low": metric["wilson95"][0] if metric["wilson95"] else None,
                f"{prefix}_wilson95_high": metric["wilson95"][1] if metric["wilson95"] else None,
            })
        threshold_rows.extend({"variant": key, **row} for row in value["threshold_selection"])
    _csv(destination / "metrics.csv", metric_rows, list(metric_rows[0]))
    _csv(destination / "threshold_selection.csv", threshold_rows, ["variant", *variants[next(iter(variants))]["threshold_selection"][0]])
    _roc_curve(destination / "roc_curve.png", report["verification"])
    _threshold_curve(destination / "threshold_curve.png", report)


def write_matrix_results(report: dict, output_directory: str | Path) -> Path:
    """Publish all artifacts together, with checksums; never overwrite a run."""

    # Reject non-JSON values/NaN before creating anything at the destination.
    json.dumps(report, allow_nan=False)
    destination = Path(output_directory).resolve()
    with new_result_directory(destination) as staging:
        _write_matrix_results(report, staging)
        _write_json(staging / "artifacts.json", {
            "schema": "matrix-artifacts-v1", "sha256": artifact_hashes(staging),
            "note": "校验产物完整性，不是第三方时间戳或防篡改签名",
        })
    return destination
