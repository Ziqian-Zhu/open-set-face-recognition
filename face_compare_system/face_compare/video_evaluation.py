"""Evaluate exported video tracks against explicit per-frame person annotations."""

import json
import hashlib
import math
from pathlib import Path

from .database import normalize_person_name
from .detector import FaceDetector
from .evaluation import interval, summarize
from .models import BoundingBox


_STATES = {"known", "unknown", "confirming", "quality_rejected", "capacity", "identity_conflict"}


def _finite_nonnegative(value, field):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field}必须为有限非负数")


def _frame_number(value):
    if type(value) is not int or value < 0:
        raise ValueError("frame必须为非负整数，不能用布尔值或小数")


def _validate_video_inputs(predictions, truth):
    """Fail closed, including empty scenes that would never enter IoU matching.

    Legacy exports without metadata/counts are supported. When v2 export
    metadata/counts exist they are binding, not silently ignored hints.
    """
    if not isinstance(truth, dict) or truth.get("split") not in ("validation", "test"):
        raise ValueError("视频标注须注明validation或test")
    if (not isinstance(predictions, list) or not predictions
            or any(not isinstance(row, dict) for row in predictions)
            or predictions[-1].get("type") != "summary"):
        raise ValueError("视频导出不完整：缺少最终summary")
    types = [row.get("type") for row in predictions]
    if (types.count("summary") != 1 or types.count("metadata") > 1
            or any(not isinstance(value, str) or value not in {"metadata", "frame", "summary"} for value in types)
            or ("metadata" in types and types[0] != "metadata")):
        raise ValueError("视频记录须为可选metadata、连续frame及唯一末尾summary")
    frames = [row for row in predictions if row["type"] == "frame"]
    if not frames:
        raise ValueError("预测帧为空或重复")
    previous_frame, previous_timestamp = -1, -1.
    for frame in frames:
        _frame_number(frame.get("frame"))
        _finite_nonnegative(frame.get("timestamp"), "视频时间戳")
        _finite_nonnegative(frame.get("processing_ms"), "processing_ms")
        if frame["frame"] <= previous_frame or frame["timestamp"] <= previous_timestamp:
            raise ValueError("视频帧号和时间戳须严格递增")
        previous_frame, previous_timestamp = frame["frame"], frame["timestamp"]
        if not isinstance(frame.get("faces"), list):
            raise ValueError("预测faces必须为列表")
        track_ids = set()
        for face in frame["faces"]:
            if not isinstance(face, dict):
                raise ValueError("预测人脸必须为对象")
            box(face.get("box"))
            if not isinstance(face.get("state"), str) or face["state"] not in _STATES:
                raise ValueError("预测state不合法")
            if face["state"] == "known":
                if not isinstance(face.get("name"), str) or not face["name"].strip():
                    raise ValueError("known输出必须有非空身份名称")
                normalize_person_name(face["name"])
            if face.get("distance") is not None:
                _finite_nonnegative(face["distance"], "distance")
            tid = face.get("track_id")
            if tid is not None:
                if type(tid) is not int or tid < 1 or tid in track_ids:
                    raise ValueError("同帧track_id必须为唯一正整数或null")
                track_ids.add(tid)
    annotations = truth.get("frames")
    if not isinstance(annotations, list):
        raise ValueError("标注frames必须为列表")
    seen = set()
    for frame in annotations:
        if not isinstance(frame, dict):
            raise ValueError("标注帧必须为对象")
        _frame_number(frame.get("frame"))
        if frame["frame"] in seen:
            raise ValueError("标注帧重复")
        seen.add(frame["frame"])
        if not isinstance(frame.get("faces"), list):
            raise ValueError("标注faces必须为列表，无人帧须显式写[]")
        subjects = set()
        for face in frame["faces"]:
            if not isinstance(face, dict):
                raise ValueError("标注人脸必须为对象")
            box(face.get("box"))
            subject = face.get("subject")
            if not isinstance(subject, str) or not subject.strip() or subject in subjects:
                raise ValueError("同帧subject必须为非空且唯一的字符串；不同未知人员也必须区分")
            subjects.add(subject)
            if "label" not in face:
                raise ValueError("未知人员也须显式写label:null，不能遗漏label")
            if face["label"] is not None:
                if not isinstance(face["label"], str):
                    raise ValueError("label必须为字符串或null")
                normalize_person_name(face["label"])
    if seen != {frame["frame"] for frame in frames}:
        raise ValueError("必须标注全部已分析帧（无人帧用faces:[]），不能只挑容易的帧")
    session = truth.get("session_id")
    if session is not None and (not isinstance(session, str) or not session.strip()):
        raise ValueError("session_id必须为非空字符串或null；不能伪造缺失会话")
    summary = predictions[-1]
    if "analyzed_frames" in summary and (type(summary["analyzed_frames"]) is not int
                                        or summary["analyzed_frames"] != len(frames)):
        raise ValueError("summary.analyzed_frames与实际记录不一致")
    if "decoded_frames" in summary:
        decoded = summary["decoded_frames"]
        if type(decoded) is not int or decoded <= frames[-1]["frame"]:
            raise ValueError("summary.decoded_frames无效")
    if types[0] == "metadata":
        metadata = predictions[0]
        if "fps" in metadata:
            _finite_nonnegative(metadata["fps"], "fps")
            if metadata["fps"] == 0:
                raise ValueError("fps必须大于0")
            if any(not math.isclose(frame["timestamp"], frame["frame"] / metadata["fps"],
                                    rel_tol=1e-7, abs_tol=1e-7) for frame in frames):
                raise ValueError("时间戳与导出的frame/fps不一致")
        if "every" in metadata:
            every = metadata["every"]
            if type(every) is not int or every < 1:
                raise ValueError("every必须为正整数")
            if ("decoded_frames" not in summary
                    or len(frames) != (summary["decoded_frames"] + every - 1) // every
                    or any(row["frame"] != i * every for i, row in enumerate(frames))):
                raise ValueError("帧记录与完整采样网格不一致，不能同时删掉预测和真值中的困难帧")


def _subject_outcomes(episodes):
    """One descriptive row per annotated identity, not one independent trial/frame."""
    grouped = {}
    for episode in episodes:
        subject = episode["subject"]
        if subject not in grouped:
            grouped[subject] = {"subject": subject, "expected": episode["expected"], "episodes": 0,
                                "confirmed_episodes": 0, "visible_frames": 0, "detected_frames": 0,
                                "usable_frames": 0, "correct_frames": 0, "false_accept_frames": 0,
                                "misidentified_frames": 0, "track_id_switches": 0,
                                "recognized_label_switches": 0, "output_state_switches": 0}
        row = grouped[subject]
        row["episodes"] += 1
        row["confirmed_episodes"] += episode["first_correct_seconds"] is not None
        for field in ("visible_frames", "detected_frames", "usable_frames", "correct_frames",
                      "false_accept_frames", "misidentified_frames", "track_id_switches",
                      "recognized_label_switches", "output_state_switches"):
            row[field] += episode[field]
    rows = [grouped[key] for key in sorted(grouped)]
    for row in rows:
        row["correct_frame_fraction"] = row["correct_frames"] / row["visible_frames"]
        row["false_accept_frame_fraction"] = row["false_accept_frames"] / row["visible_frames"]
    known = [row for row in rows if row["expected"] is not None]
    unknown = [row for row in rows if row["expected"] is None]
    return rows, {
        "known_subjects": len(known), "unknown_subjects": len(unknown),
        "known_ever_correct": sum(row["correct_frames"] > 0 for row in known),
        "known_never_correct": sum(row["correct_frames"] == 0 for row in known),
        "unknown_ever_false_accepted": sum(row["false_accept_frames"] > 0 for row in unknown),
        "known_macro_correct_frame_fraction": (
            sum(row["correct_frame_fraction"] for row in known) / len(known) if known else None),
        "unknown_macro_false_accept_frame_fraction": (
            sum(row["false_accept_frame_fraction"] for row in unknown) / len(unknown) if unknown else None),
        "interval": None,
        "interval_reason": "同一视频/会话及同时多视角仍相关；不假设逐帧或逐身份独立",
    }


def box(value):
    if (not isinstance(value, list) or len(value) != 4
            or not all(type(v) in (int, float) and math.isfinite(v) for v in value)
            or min(value[2:]) <= 0):
        raise ValueError("标注框必须是有限的[x,y,width,height]，宽高须为正数")
    return BoundingBox(*value)


def _read_inputs(prediction_path, truth_path):
    prediction_bytes = Path(prediction_path).read_bytes()
    truth_bytes = Path(truth_path).read_bytes()
    predictions = [json.loads(line) for line in prediction_bytes.decode("utf-8").splitlines() if line.strip()]
    truth = json.loads(truth_bytes)
    hashes = {"prediction_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
              "truth_sha256": hashlib.sha256(truth_bytes).hexdigest()}
    return predictions, truth, hashes


def evaluate_video(prediction_path, truth_path, *, decision_mode="stable"):
    """Evaluate one output policy; old exports remain supported in stable mode."""

    predictions, truth, hashes = _read_inputs(prediction_path, truth_path)
    return {**_evaluate_video_rows(predictions, truth, decision_mode=decision_mode), **hashes}


def _evaluate_video_rows(predictions, truth, *, decision_mode="stable"):
    if decision_mode not in {"stable", "single_frame"}:
        raise ValueError("decision_mode必须为stable或single_frame")
    _validate_video_inputs(predictions, truth)
    frames = [p for p in predictions if p.get("type") == "frame"]
    if decision_mode == "single_frame":
        transformed = []
        for frame in frames:
            faces = []
            for face in frame["faces"]:
                raw = face.get("single_frame")
                if (not isinstance(raw, dict) or not isinstance(raw.get("state"), str)
                        or raw["state"] not in {"known", "unknown", "quality_rejected"}):
                    raise ValueError("缺少合法single_frame输出，请用新版重新导出，不能从稳定结果反推单帧")
                if raw["state"] == "known" and (not isinstance(raw.get("name"), str) or not raw["name"].strip()):
                    raise ValueError("单帧known输出必须有非空身份名称")
                if raw.get("distance") is not None:
                    _finite_nonnegative(raw["distance"], "single_frame.distance")
                faces.append({**face, "state": raw["state"], "name": raw.get("name"), "distance": raw.get("distance")})
            transformed.append({**frame, "faces": faces})
        frames = transformed
    annotations = truth.get("frames", [])
    by_frame = {item["frame"]: item for item in annotations}
    records, episodes, active, labels, previous = [], [], {}, {}, {}
    last_recognized, last_output = {}, {}
    count, matched_count, false_detections, switches, recognition_flickers = 0, 0, 0, 0, 0
    output_state_switches = 0
    for prediction in frames:
        timestamp = prediction["timestamp"]
        expected = by_frame[prediction["frame"]].get("faces", [])
        observed = prediction["faces"]
        ids = [item["subject"] for item in expected]
        for subject in list(active):
            if subject not in ids:
                del active[subject]
                previous.pop(subject, None)
                last_recognized.pop(subject, None)
                last_output.pop(subject, None)
        edges = []
        for i, target in enumerate(expected):
            for j, face in enumerate(observed):
                overlap = FaceDetector._intersection_over_union(box(target["box"]), box(face["box"]))
                if overlap >= .5:
                    edges.append((-overlap, i, j))
        assignments, used = {}, set()
        for _, i, j in sorted(edges):
            if i not in assignments and j not in used:
                assignments[i] = j
                used.add(j)
        count += len(expected)
        matched_count += len(assignments)
        false_detections += len(observed)-len(used)
        for i, target in enumerate(expected):
            subject, label = target["subject"], target.get("label")
            if label is not None:
                label = normalize_person_name(label)
            if subject in labels and labels[subject] != label:
                raise ValueError("同一subject的身份标签不能中途改变")
            labels[subject] = label
            if subject not in active:
                active[subject] = len(episodes)
                episodes.append({"subject": subject, "expected": label, "start": timestamp,
                                 "first_correct_seconds": None, "visible_frames": 0,
                                 "detected_frames": 0, "usable_frames": 0,
                                 "correct_frames": 0, "false_accept_frames": 0,
                                 "misidentified_frames": 0, "recognized_label_switches": 0,
                                 "output_state_switches": 0, "track_id_switches": 0})
            face = observed[assignments[i]] if i in assignments else None
            predicted = normalize_person_name(face["name"]) if face and face["state"] == "known" else None
            status = "no_face" if face is None else "recognized" if face["state"] in ("known", "unknown") else face["state"]
            records.append(dict(frame=prediction["frame"], subject=subject, expected=label, predicted=predicted,
                                status=status, ms=prediction["processing_ms"]))
            episode = episodes[active[subject]]
            episode["visible_frames"] += 1
            episode["detected_frames"] += face is not None
            episode["usable_frames"] += status == "recognized"
            output_key = (status, predicted)
            if subject in last_output and last_output[subject] != output_key:
                output_state_switches += 1
                episode["output_state_switches"] += 1
            last_output[subject] = output_key
            if face and face.get("track_id") is not None:
                old = previous.get(subject)
                if old is not None and old != face["track_id"]:
                    switches += 1
                    episode["track_id_switches"] += 1
                previous[subject] = face["track_id"]
            # Count stable *recognized-output* changes within a continuous
            # matched run. A miss or confirming frame breaks the run, while
            # known<->unknown and known A<->known B changes both count.
            if status == "recognized":
                if subject in last_recognized and last_recognized[subject] != predicted:
                    recognition_flickers += 1
                    episode["recognized_label_switches"] += 1
                last_recognized[subject] = predicted
            else:
                last_recognized.pop(subject, None)
            correct = status == "recognized" and predicted == label
            episode["correct_frames"] += correct
            episode["false_accept_frames"] += label is None and predicted is not None
            episode["misidentified_frames"] += label is not None and predicted is not None and predicted != label
            if correct and episode["first_correct_seconds"] is None:
                episode["first_correct_seconds"] = timestamp-episode["start"]
    summary = summarize(records)
    # Timing is per analyzed frame, not repeated once per ground-truth face.
    import numpy as np
    elapsed = [frame["processing_ms"] for frame in frames]
    summary["latency_ms"] = {"median": float(np.median(elapsed)), "p95": float(np.percentile(elapsed, 95))}
    scene = {
        "frames_with_multiple_truth_faces": sum(
            len(by_frame[frame["frame"]].get("faces", [])) >= 2 for frame in frames
        ),
        "frames_with_multiple_detected_faces": sum(
            len(frame["faces"]) >= 2 for frame in frames
        ),
        "max_simultaneous_truth_faces": max(
            len(by_frame[frame["frame"]].get("faces", [])) for frame in frames
        ),
    }
    known_episodes = [item for item in episodes if item["expected"] is not None]
    unknown_episodes = [item for item in episodes if item["expected"] is None]
    episode_summary = {
        "known_episodes": len(known_episodes), "unknown_episodes": len(unknown_episodes),
        "known_ever_correct": sum(item["first_correct_seconds"] is not None for item in known_episodes),
        "unknown_ever_false_accepted": sum(item["false_accept_frames"] > 0 for item in unknown_episodes),
        "known_any_misidentification": sum(item["misidentified_frames"] > 0 for item in known_episodes),
        "known_mean_correct_fraction": (
            float(np.mean([item["correct_frames"] / item["visible_frames"] for item in known_episodes]))
            if known_episodes else None
        ),
        "unknown_mean_false_accept_fraction": (
            float(np.mean([item["false_accept_frames"] / item["visible_frames"] for item in unknown_episodes]))
            if unknown_episodes else None
        ),
        "known_confirmed_delay_seconds": [item["first_correct_seconds"] for item in known_episodes
                                          if item["first_correct_seconds"] is not None],
        "known_never_confirmed": sum(item["first_correct_seconds"] is None for item in known_episodes),
    }
    subjects, subject_summary = _subject_outcomes(episodes)
    return {"split": truth["split"], "decision_mode": decision_mode,
            "session_id": truth.get("session_id"),
            "session_metadata": "provided_not_independently_verified" if truth.get("session_id") else "unavailable",
            "annotated_frames": len(frames), "face_attempts": count,
            "detection_recall": interval(matched_count, count), "unmatched_detections": false_detections,
            "track_id_switches": switches, "recognized_label_switches": recognition_flickers,
            "output_state_switches_including_pending_and_misses": output_state_switches,
            "summary": summary, "scene": scene,
            "subject_summary": subject_summary, "subjects": subjects,
            "episode_summary": episode_summary, "episodes": episodes,
            "unconfirmed_episodes": sum(e["first_correct_seconds"] is None for e in episodes), "records": records,
            "limitations": ["IoU≥0.5贪心一对一匹配；不是完整MOTChallenge指标实现",
                            "首次确认时间按视频时间而非实时计算速度；未确认片段单独保留",
                            "同一视频内帧高度相关，逐帧比例的Wilson区间不是独立人员级置信区间",
                            "不评估活体检测；漏检、质量失败、待确认均不能当成成功未知拒绝",
                            "按连续出现的subject计算episode；短暂离开后重新出现算新episode，轨迹标注subject须稳定",
                            "身份级macro按subject等权；会话ID仅来自标注，不自动把多视角或连续片段当作独立会话",
                            "recognized_label_switches只比较连续的已识别输出，漏检/待确认会中断序列；不等同完整MOT身份切换"]}


def evaluate_temporal_ablation(prediction_path, truth_path):
    """Compare recorded raw vs confirmed outputs on identical detection evidence.

    The temporal treatment includes confirmation and duplicate-identity conflict
    suppression. It is not an ablation of association or independent detector runs.
    """

    predictions, truth, hashes = _read_inputs(prediction_path, truth_path)
    baseline = _evaluate_video_rows(predictions, truth, decision_mode="single_frame")
    temporal = _evaluate_video_rows(predictions, truth, decision_mode="stable")
    pairs = list(zip(baseline["records"], temporal["records"], strict=True))
    if any((a["frame"], a["subject"], a["expected"]) != (b["frame"], b["subject"], b["expected"])
           for a, b in pairs):
        raise ValueError("时序对照没有使用完全相同的真值尝试")
    known = [(a, b) for a, b in pairs if a["expected"] is not None]
    unknown = [(a, b) for a, b in pairs if a["expected"] is None]
    def correct(row):
        return row["status"] == "recognized" and row["predicted"] == row["expected"]
    return {
        "protocol": "paired-single-frame-vs-temporal-v1", **hashes,
        "evidence_kind": truth.get("evidence_kind", "provided_annotations_unverified_provenance"),
        "single_frame": baseline, "temporal": temporal,
        "paired": {
            "known_attempts": len(known), "unknown_attempts": len(unknown),
            "known_correct_gained": sum(not correct(a) and correct(b) for a, b in known),
            "known_correct_lost": sum(correct(a) and not correct(b) for a, b in known),
            "unknown_false_accepts_prevented": sum(a["predicted"] is not None and b["predicted"] is None for a, b in unknown),
            "unknown_false_accepts_added": sum(a["predicted"] is None and b["predicted"] is not None for a, b in unknown),
            "recognized_output_attempts_single": sum(a["status"] == "recognized" for a, _ in pairs),
            "recognized_output_attempts_temporal": sum(b["status"] == "recognized" for _, b in pairs),
        },
        "limitations": [
            "两种输出共享同一帧、检测、质量、单帧匹配与轨迹关联；只比较单帧决策和时序/身份冲突处理",
            "减少闪烁可能以更多待确认和首次确认延迟为代价；应同时阅读输出覆盖与状态切换数",
            "逐帧尝试高度相关，不进行独立样本显著性检验；合成fixture仅验证代码，不是真实视频效果",
            "两组processing_ms来自同一次处理，不能据此比较单帧/时序的运行成本",
        ],
    }
