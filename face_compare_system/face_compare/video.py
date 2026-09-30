"""Local video playback and repeatable per-frame tracking export (no camera)."""

import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from .service import FaceComparisonSystem
from .tracking import MultiFaceTracker


class VideoFileSource:
    def __init__(self, path):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise ValueError("视频文件不存在")
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError("无法读取该视频格式")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(self.fps) or not 0 < self.fps <= 240:
            self.fps = 25.
        self.current_index = "视频"
        self.next_frame = 0.
        self.frame_index = -1
        self.timestamp = 0.

    @property
    def is_open(self):
        return self.capture.isOpened()

    def read(self):
        now = time.monotonic()
        if now < self.next_frame:
            return None, None  # not due; different from EOF/failed decoding
        ok, frame = self.capture.read()
        # Accumulate deadlines, not the (late) UI poll time: 30 ms polls must
        # not turn a 30 FPS file into a 16.7 FPS file. A long pause rebases
        # playback without an unbounded burst of catch-up decodes.
        deadline = self.next_frame if self.frame_index >= 0 and now-self.next_frame < 1.0 else now
        self.next_frame = deadline+1/self.fps
        if ok:
            self.frame_index += 1
            self.timestamp = self.frame_index/self.fps
        return bool(ok), frame if ok else None

    def poll_delay_ms(self):
        return max(1, min(30, math.ceil((self.next_frame-time.monotonic())*1000)))

    def release(self):
        self.capture.release()


def analyze_video(config, root, path, output, *, every=3, max_frames=0):
    if every < 1 or max_frames < 0:
        raise ValueError("采样间隔须为正数，帧上限不能为负数")
    output = Path(output)
    if output.exists():
        raise ValueError("输出已存在，请选择新路径")
    source = VideoFileSource(path)
    system = None
    durations, analyzed, decoded = [], 0, 0
    try:
        system = FaceComparisonSystem(config, root)
        tracker = MultiFaceTracker(system.extractor, system.recognition_config, config.tracking)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"type": "metadata", "schema": "video-dual-decision-v2", "video": str(source.path), "fps": source.fps,
                "every": every, "threshold": system.recognizer.threshold, "margin": system.recognition_config.ambiguity_margin,
                "engine": system.engine_label, "limitations": "无真值标注，不计算准确率；逐帧耗时不等于身份确认延迟"}, ensure_ascii=False)+"\n")
            while not max_frames or decoded < max_frames:
                ok, frame = source.capture.read()
                if not ok:
                    break
                index = decoded
                decoded += 1
                if index % every:
                    continue
                timestamp = index/source.fps
                start = time.perf_counter()
                observations = tracker.update(system.analyze_frame(frame, log_events=False), timestamp)
                elapsed = (time.perf_counter()-start)*1000
                durations.append(elapsed)
                analyzed += 1
                faces = []
                for observation in observations:
                    result = observation.stable_recognition
                    raw = observation.recognition
                    raw_usable = observation.quality.accepted and observation.feature is not None and raw is not None
                    single_frame = {
                        "state": ("known" if raw.known else "unknown") if raw_usable else "quality_rejected",
                        "name": raw.name if raw_usable and raw.known else None,
                        "distance": raw.distance if raw_usable else None,
                    }
                    faces.append({"track_id": observation.track_id, "box": observation.box.as_tuple(),
                                  "state": observation.tracking_state, "name": result.name if result else None,
                                  "person_id": result.person_id if result else None,
                                  "distance": result.distance if result else None,
                                  "quality": observation.quality.to_dict(), "single_frame": single_frame})
                handle.write(json.dumps({"type": "frame", "frame": index, "timestamp": timestamp,
                                         "processing_ms": elapsed, "faces": faces}, ensure_ascii=False, allow_nan=False)+"\n")
            if not decoded:
                raise ValueError("视频没有可解码帧；输出保留为不完整记录")
            summary = {"type": "summary", "decoded_frames": decoded, "analyzed_frames": analyzed,
                       "processing_ms_median": float(np.median(durations)),
                       "processing_ms_p95": float(np.percentile(durations, 95)),
                       "stopped_at_frame_limit": bool(max_frames and decoded >= max_frames),
                       "accuracy": None}
            handle.write(json.dumps(summary)+"\n")
            return summary
    finally:
        source.release()
        if system is not None and hasattr(system.database, "close"):
            system.database.close()
