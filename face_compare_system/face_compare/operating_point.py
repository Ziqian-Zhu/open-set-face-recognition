"""Choose an empirical SFace threshold using validation results, never test data."""

import math
import numpy as np

from .decision import accept_identity
from .evaluation import summarize


def select_operating_point(report, target_fpir=.01):
    if report.get("split") != "validation":
        raise ValueError("只允许使用validation报告选择阈值，禁止用test结果调参")
    if not str(report.get("feature_signature", "")).startswith("sface-"):
        raise ValueError("此阈值工具仅支持SFace余弦距离")
    if not math.isfinite(target_fpir) or not 0 <= target_fpir <= 1:
        raise ValueError("目标FPIR必须位于[0,1]")
    margin = float(report["margin"])
    if not math.isfinite(margin) or not 0 <= margin < 1:
        raise ValueError("验证报告中的身份间隔无效")
    records = report.get("records", [])
    usable = [row for row in records if row["status"] == "recognized"]
    if not any(row["expected"] is None for row in usable) or not any(row["expected"] is not None for row in usable):
        raise ValueError("验证集必须同时包含可识别的已知及未知人员")
    candidates = {1e-8, .275, float(np.nextafter(.5, 0))}
    for row in usable:
        distance = row.get("distance")
        if distance is None or not math.isfinite(distance) or not 0 <= distance <= 1 or not row.get("candidate_name"):
            raise ValueError("验证报告缺少有效的最近候选或距离，请重新evaluate")
        second = row.get("second_best_distance")
        if second is not None and (not math.isfinite(second) or not distance <= second <= 1):
            raise ValueError("第二候选距离无效")
        if 0 < distance < .5:
            candidates.update([distance, float(np.nextafter(distance, -np.inf))])
    curve, best = [], None
    for threshold in sorted(candidates):
        simulated = []
        for row in records:
            predicted = None
            if row["status"] == "recognized":
                second = row.get("second_best_distance")
                if accept_identity(row["distance"], second, threshold, margin):
                    predicted = row["candidate_name"]
            simulated.append({**row, "predicted": predicted})
        summary = summarize(simulated)
        fpir = summary["unknown_false_accept_usable_faces"]["rate"]
        correct = summary["known_correct"]["rate"]
        curve.append({"distance_threshold": threshold, "fpir_usable_unknown": fpir,
                      "known_correct_all_attempts": correct})
        if fpir <= target_fpir:
            rank = (-correct, summary["known_misidentified"]["rate"], fpir, threshold)
            if best is None or rank < best[0]:
                best = (rank, threshold, summary)
    if best is None:
        raise ValueError("当前特征及身份间隔下没有满足目标的阈值；不能宣称达到该FPIR")
    return {"source_split": "validation", "feature_signature": report["feature_signature"],
            "target_empirical_fpir": target_fpir, "distance_threshold": best[1],
            "suggested_config_patch": {"engine": {"cosine_threshold": 1-2*best[1]}},
            "fixed_distance_margin": margin, "validation_summary": best[2], "curve": curve,
            "limitations": ["未自动修改配置；应用建议后须另采test评估", "目标约束是样本上的经验比例，不是总体风险保证",
                            "同一人的连拍不等于独立样本；未知身份应尽量与最终测试不同",
                            "此工具优化静态1:N拒识，不等同于视频级确认或活体检测"]}
