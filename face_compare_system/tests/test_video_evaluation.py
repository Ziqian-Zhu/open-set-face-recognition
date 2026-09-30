import json
import pytest
from face_compare.video_evaluation import evaluate_video, evaluate_temporal_ablation


def files(tmp_path):
    prediction = tmp_path/"video.jsonl"
    rows = [dict(type="frame", frame=i, timestamp=i*.1, processing_ms=10, faces=faces)
            for i, faces in enumerate([
                [dict(box=[0, 0, 100, 100], track_id=1, state="confirming", name=None)],
                [dict(box=[0, 0, 100, 100], track_id=2, state="known", name="A")],
                [],
            ])] + [dict(type="summary")]
    prediction.write_text("\n".join(json.dumps(row) for row in rows))
    truth = tmp_path/"truth.json"
    truth.write_text(json.dumps(dict(split="test", frames=[dict(frame=i, faces=[dict(subject="p1", label="A", box=[0, 0, 100, 100])]) for i in range(3)])))
    return prediction, truth


def test_video_metrics_count_misses_pending_switch_and_first_confirmation(tmp_path):
    result = evaluate_video(*files(tmp_path))
    assert result["detection_recall"]["rate"] == pytest.approx(2/3)
    assert result["track_id_switches"] == 1
    assert result["episodes"][0]["first_correct_seconds"] == .1
    assert result["summary"]["known_correct"]["rate"] == pytest.approx(1/3)
    assert result["episode_summary"]["known_ever_correct"] == 1
    assert result["episodes"][0]["visible_frames"] == 3
    assert result["episodes"][0]["correct_frames"] == 1


def test_cannot_cherry_pick_only_successful_frames(tmp_path):
    prediction, truth = files(tmp_path)
    data = json.loads(truth.read_text())
    data["frames"] = data["frames"][1:2]
    truth.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="全部"):
        evaluate_video(prediction, truth)


def test_unknown_pending_and_miss_are_not_successful_rejections(tmp_path):
    prediction, truth = files(tmp_path)
    data = json.loads(truth.read_text())
    for frame in data["frames"]:
        frame["faces"][0]["label"] = None
    truth.write_text(json.dumps(data))
    result = evaluate_video(prediction, truth)
    assert result["summary"]["unknown_correct_rejection_all_attempts"]["rate"] == 0
    assert result["summary"]["unknown_false_accept_all_attempts"]["rate"] == pytest.approx(1/3)
    assert result["episode_summary"]["unknown_ever_false_accepted"] == 1


def test_recognized_label_flicker_counts_known_unknown_and_name_changes(tmp_path):
    prediction, truth = files(tmp_path)
    predictions = [dict(type="frame", frame=i, timestamp=i*.1, processing_ms=10,
                        faces=[dict(box=[0, 0, 100, 100], track_id=1, state=state, name=name)])
                   for i, (state, name) in enumerate([
                       ("known", "A"), ("unknown", None), ("known", "B"),
                       ("confirming", None), ("known", "A"),
                   ])] + [dict(type="summary")]
    prediction.write_text("\n".join(json.dumps(row) for row in predictions))
    truth.write_text(json.dumps(dict(split="test", frames=[
        dict(frame=i, faces=[dict(subject="p1", label="A", box=[0, 0, 100, 100])])
        for i in range(5)
    ])))
    result = evaluate_video(prediction, truth)
    assert result["recognized_label_switches"] == 2
    assert result["episodes"][0]["recognized_label_switches"] == 2
    assert result["episode_summary"]["known_any_misidentification"] == 1
    assert result["episodes"][0]["misidentified_frames"] == 1


def test_two_faces_have_independent_episode_outcomes(tmp_path):
    prediction = tmp_path / "two.jsonl"
    truth = tmp_path / "two-truth.json"
    rows = [
        dict(type="frame", frame=0, timestamp=0., processing_ms=12, faces=[
            dict(box=[0, 0, 100, 100], track_id=1, state="known", name="A"),
            dict(box=[200, 0, 100, 100], track_id=2, state="unknown", name=None),
        ]),
        dict(type="frame", frame=1, timestamp=.1, processing_ms=13, faces=[
            dict(box=[0, 0, 100, 100], track_id=1, state="known", name="A"),
            dict(box=[200, 0, 100, 100], track_id=2, state="known", name="A"),
        ]),
        dict(type="summary"),
    ]
    prediction.write_text("\n".join(json.dumps(row) for row in rows))
    truth.write_text(json.dumps(dict(split="test", frames=[dict(frame=i, faces=[
        dict(subject="known-1", label="A", box=[0, 0, 100, 100]),
        dict(subject="unknown-1", label=None, box=[200, 0, 100, 100]),
    ]) for i in range(2)])))
    result = evaluate_video(prediction, truth)
    assert result["episode_summary"]["known_ever_correct"] == 1
    assert result["episode_summary"]["unknown_ever_false_accepted"] == 1
    assert result["recognized_label_switches"] == 1
    assert result["track_id_switches"] == 0
    assert result["scene"]["frames_with_multiple_truth_faces"] == 2
    assert result["scene"]["max_simultaneous_truth_faces"] == 2


def test_temporal_ablation_rejects_legacy_exports_without_raw_decisions(tmp_path):
    prediction, truth = files(tmp_path)
    with pytest.raises(ValueError, match="single_frame"):
        evaluate_temporal_ablation(prediction, truth)
    assert evaluate_video(prediction, truth)["decision_mode"] == "stable"


def test_temporal_ablation_pairs_same_attempts_and_reports_confirmation_cost(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import numpy as np
    from face_compare.config import AppConfig, RecognitionConfig
    from face_compare.deep_engine import SFaceExtractor
    from face_compare.models import BoundingBox, FaceObservation, QualityReport, RecognitionResult
    from face_compare.video import analyze_video

    def observation(x, label, feature):
        raw = (RecognitionResult(True, label, label, 95, .05, .275, "synthetic") if label
               else RecognitionResult.unknown(threshold=.275, reason="synthetic unknown"))
        return FaceObservation(BoundingBox(x, 0, 80, 100), QualityReport(True, 120, 40, 100, .2),
                               np.zeros((100, 80, 3), np.uint8), raw, np.asarray(feature, np.float32))
    config = AppConfig(recognition=RecognitionConfig(vote_window=7, vote_min_frames=4))
    observations = [[observation(0, "B" if i == 5 else "A", (1, 0)),
                     observation(200, "A" if i == 0 else None, (0, 1))] for i in range(10)]
    system = SimpleNamespace(extractor=SFaceExtractor, recognition_config=config.recognition,
                             database=SimpleNamespace(close=lambda: None), engine_label="SYNTHETIC",
                             recognizer=SimpleNamespace(threshold=.275), analyze_frame=Mock(side_effect=observations))
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.return_value = 10.
    capture.read.side_effect = [(True, np.zeros((120, 300, 3), np.uint8)) for _ in range(10)] + [(False, None)]
    monkeypatch.setattr("face_compare.video.cv2.VideoCapture", Mock(return_value=capture))
    monkeypatch.setattr("face_compare.video.FaceComparisonSystem", lambda *_: system)
    path = tmp_path / "synthetic.avi"
    path.touch()
    prediction, truth = tmp_path / "export.jsonl", tmp_path / "truth.json"
    analyze_video(config, tmp_path, path, prediction, every=1)
    truth.write_text(json.dumps({"split": "test", "evidence_kind": "synthetic_regression_only",
                                "frames": [{"frame": i, "faces": [
                                    {"subject": "known", "label": "A", "box": [0, 0, 80, 100]},
                                    {"subject": "unknown", "label": None, "box": [200, 0, 80, 100]},
                                ]} for i in range(10)]}))
    result = evaluate_temporal_ablation(prediction, truth)
    assert result["evidence_kind"] == "synthetic_regression_only"
    assert result["paired"]["known_attempts"] == result["paired"]["unknown_attempts"] == 10
    assert result["paired"]["unknown_false_accepts_prevented"] == 1
    assert result["paired"]["known_correct_lost"] > 0
    assert result["paired"]["recognized_output_attempts_temporal"] < 20
    assert result["single_frame"]["recognized_label_switches"] == 3
    assert result["temporal"]["recognized_label_switches"] == 0
    assert result["temporal"]["output_state_switches_including_pending_and_misses"] > 0
    assert result["temporal"]["episode_summary"]["known_confirmed_delay_seconds"][0] > 0
    assert result["single_frame"]["detection_recall"] == result["temporal"]["detection_recall"]


def _change_prediction(path, mutate):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows)
    path.write_text("\n".join(json.dumps(row) for row in rows))


@pytest.mark.parametrize("field,value", [
    ("processing_ms", -1), ("processing_ms", float("nan")), ("processing_ms", True),
    ("timestamp", -1), ("timestamp", float("inf")), ("timestamp", "0"),
    ("frame", -1), ("frame", .5), ("frame", True),
])
def test_invalid_frame_fields_are_rejected(tmp_path, field, value):
    prediction, truth = files(tmp_path)
    _change_prediction(prediction, lambda rows: rows[0].update({field: value}))
    with pytest.raises(ValueError):
        evaluate_video(prediction, truth)


@pytest.mark.parametrize("field,value", [
    ("state", "nonsense"), ("state", []), ("track_id", 0), ("track_id", True),
    ("box", [0, 0, 0, 2]), ("box", [0, 0, float("nan"), 2]),
    ("distance", float("inf")), ("distance", -1),
])
def test_invalid_detection_rejected_even_without_ground_truth_face(tmp_path, field, value):
    prediction, truth = files(tmp_path)
    data = json.loads(truth.read_text())
    data["frames"][0]["faces"] = []
    truth.write_text(json.dumps(data))
    _change_prediction(prediction, lambda rows: rows[0]["faces"][0].update({field: value}))
    with pytest.raises(ValueError):
        evaluate_video(prediction, truth)


def test_bad_truth_box_rejected_even_when_detector_misses_everything(tmp_path):
    prediction, truth = files(tmp_path)
    data = json.loads(truth.read_text())
    data["frames"][2]["faces"][0]["box"][2] = -1
    truth.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="标注框"):
        evaluate_video(prediction, truth)


def test_known_without_name_cannot_be_counted_as_unknown_rejection(tmp_path):
    prediction, truth = files(tmp_path)
    _change_prediction(prediction, lambda rows: rows[1]["faces"][0].update(name=None))
    with pytest.raises(ValueError, match="known"):
        evaluate_video(prediction, truth)


def test_duplicate_track_and_incomplete_summary_are_rejected(tmp_path):
    prediction, truth = files(tmp_path)
    _change_prediction(prediction, lambda rows: rows[0]["faces"].append(dict(rows[0]["faces"][0])))
    with pytest.raises(ValueError, match="track_id"):
        evaluate_video(prediction, truth)
    prediction, truth = files(tmp_path)
    _change_prediction(prediction, lambda rows: rows[-1].update(analyzed_frames=2))
    with pytest.raises(ValueError, match="analyzed_frames"):
        evaluate_video(prediction, truth)


def test_missing_truth_label_is_not_implicitly_unknown(tmp_path):
    prediction, truth = files(tmp_path)
    data = json.loads(truth.read_text())
    del data["frames"][0]["faces"][0]["label"]
    truth.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="label"):
        evaluate_video(prediction, truth)


def test_export_metadata_prevents_deleting_a_frame_from_both_inputs(tmp_path):
    prediction, truth = files(tmp_path)
    def metadata(rows):
        rows.insert(0, {"type": "metadata", "fps": 10., "every": 1})
        rows[-1].update(analyzed_frames=3, decoded_frames=3)
    _change_prediction(prediction, metadata)
    assert evaluate_video(prediction, truth)["annotated_frames"] == 3
    def delete(rows):
        del rows[2]
        rows[-1]["analyzed_frames"] = 2
    _change_prediction(prediction, delete)
    data = json.loads(truth.read_text())
    del data["frames"][1]
    truth.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="采样网格"):
        evaluate_video(prediction, truth)


def test_fps_and_timestamps_must_agree(tmp_path):
    prediction, truth = files(tmp_path)
    _change_prediction(prediction, lambda rows: rows.insert(0, {"type": "metadata", "fps": 30.}))
    with pytest.raises(ValueError, match="frame/fps"):
        evaluate_video(prediction, truth)


def test_subject_macro_does_not_overweight_long_tracks_and_keeps_sessions_explicit(tmp_path):
    prediction, truth = files(tmp_path)
    truth_data = {"split": "test", "session_id": "capture-event-1", "frames": []}
    rows = []
    for i in range(10):
        # Long track A always recognized; short track B always missed.
        faces = [{"subject": "a", "label": "A", "box": [0, 0, 100, 100]}]
        if i == 0:
            faces.append({"subject": "b", "label": "B", "box": [200, 0, 100, 100]})
        truth_data["frames"].append({"frame": i, "faces": faces})
        rows.append({"type": "frame", "frame": i, "timestamp": i / 10, "processing_ms": 10,
                     "faces": [{"box": [0, 0, 100, 100], "track_id": 1, "state": "known", "name": "A"}]})
    rows.append({"type": "summary"})
    prediction.write_text("\n".join(json.dumps(row) for row in rows))
    truth.write_text(json.dumps(truth_data))
    result = evaluate_video(prediction, truth)
    assert result["summary"]["known_correct"]["rate"] == pytest.approx(10 / 11)
    assert result["subject_summary"]["known_macro_correct_frame_fraction"] == .5
    assert result["subject_summary"]["known_never_correct"] == 1
    assert result["subject_summary"]["interval"] is None
    assert result["session_id"] == "capture-event-1"
    assert result["session_metadata"] == "provided_not_independently_verified"
    del truth_data["session_id"]
    truth.write_text(json.dumps(truth_data))
    assert evaluate_video(prediction, truth)["session_metadata"] == "unavailable"


def test_reappearance_is_another_episode_not_another_subject(tmp_path):
    prediction, truth = files(tmp_path)
    data = json.loads(truth.read_text())
    data["frames"][1]["faces"] = []
    truth.write_text(json.dumps(data))
    result = evaluate_video(prediction, truth)
    assert result["episode_summary"]["known_episodes"] == 2
    assert result["subject_summary"]["known_subjects"] == 1
    assert result["subjects"][0]["episodes"] == 2
    assert result["subjects"][0]["visible_frames"] == 2
