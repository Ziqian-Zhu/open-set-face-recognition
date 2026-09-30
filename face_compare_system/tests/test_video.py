import json
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest

from face_compare.config import AppConfig
from face_compare.video import VideoFileSource, analyze_video


def test_playback_wait_is_distinct_from_eof(tmp_path, monkeypatch):
    path = tmp_path/"movie.mp4"
    path.touch()
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.return_value = 10.
    frame = np.zeros((20, 20, 3), np.uint8)
    capture.read.side_effect = [(True, frame), (False, None)]
    monkeypatch.setattr("face_compare.video.cv2.VideoCapture", Mock(return_value=capture))
    now = [1.]
    monkeypatch.setattr("face_compare.video.time.monotonic", lambda: now[0])
    source = VideoFileSource(path)
    assert source.read()[0] is True
    now[0] = 1.01
    assert source.read() == (None, None)
    now[0] = 1.2
    assert source.read() == (False, None)
    source.release()
    capture.release.assert_called_once()


def test_offline_export_processes_sampled_frames_without_claiming_accuracy(tmp_path, monkeypatch):
    path = tmp_path/"movie.mp4"
    path.touch()
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.return_value = 25.
    capture.read.side_effect = [(True, np.zeros((20, 20, 3), np.uint8)) for _ in range(6)] + [(False, None)]
    monkeypatch.setattr("face_compare.video.cv2.VideoCapture", Mock(return_value=capture))
    config = AppConfig()
    system = SimpleNamespace(extractor=None, recognition_config=config.recognition, database=SimpleNamespace(),
                             engine_label="test", recognizer=SimpleNamespace(threshold=.275), analyze_frame=Mock(return_value=[]))
    monkeypatch.setattr("face_compare.video.FaceComparisonSystem", lambda *args: system)
    output = tmp_path/"result.jsonl"
    result = analyze_video(config, tmp_path, path, output, every=2)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [r["frame"] for r in records if r["type"] == "frame"] == [0, 2, 4]
    assert result["accuracy"] is None
    assert result["analyzed_frames"] == 3
    capture.release.assert_called_once()
    with pytest.raises(ValueError, match="已存在"):
        analyze_video(config, tmp_path, path, output)


@pytest.mark.parametrize("fps,expected", [(24., 24), (25., 25), (30., 30)])
def test_fixed_ui_polling_does_not_accumulate_playback_drift(tmp_path, monkeypatch, fps, expected):
    path = tmp_path / "movie.mp4"
    path.touch()
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.return_value = fps
    capture.read.return_value = (True, np.zeros((2, 2, 3), np.uint8))
    monkeypatch.setattr("face_compare.video.cv2.VideoCapture", Mock(return_value=capture))
    now = [0.]
    monkeypatch.setattr("face_compare.video.time.monotonic", lambda: now[0])
    source = VideoFileSource(path)
    for i in range(34):
        now[0] = i*.03
        source.read()
    assert capture.read.call_count == expected
    assert source.timestamp == pytest.approx((expected-1)/fps)


def test_long_pause_rebases_deadline_but_not_media_timestamp(tmp_path, monkeypatch):
    path = tmp_path / "movie.mp4"
    path.touch()
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.return_value = 30.
    capture.read.return_value = (True, np.zeros((2, 2, 3), np.uint8))
    monkeypatch.setattr("face_compare.video.cv2.VideoCapture", Mock(return_value=capture))
    now = [0.]
    monkeypatch.setattr("face_compare.video.time.monotonic", lambda: now[0])
    source = VideoFileSource(path)
    source.read()
    now[0] = 5.
    source.read()
    assert source.timestamp == pytest.approx(1/30)
    assert source.next_frame == pytest.approx(5+1/30)
    assert source.read() == (None, None)
    assert 1 <= source.poll_delay_ms() <= 30


@pytest.mark.parametrize("fps", [30., 60., 120.])
def test_adaptive_ui_delay_follows_high_frame_rate(tmp_path, monkeypatch, fps):
    path = tmp_path / "movie.mp4"
    path.touch()
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.return_value = fps
    capture.read.return_value = (True, np.zeros((2, 2, 3), np.uint8))
    monkeypatch.setattr("face_compare.video.cv2.VideoCapture", Mock(return_value=capture))
    now = [0.]
    monkeypatch.setattr("face_compare.video.time.monotonic", lambda: now[0])
    source = VideoFileSource(path)
    while now[0] < 1.:
        source.read()
        now[0] += source.poll_delay_ms()/1000
    assert abs(capture.read.call_count-fps) <= 1
