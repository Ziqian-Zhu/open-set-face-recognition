"""Exercise actual Tk widgets with a temporary database; never opens a camera."""

from __future__ import annotations

import sys
import argparse
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from face_compare.config import load_config, StorageConfig
from face_compare.models import PreparedSample, QualityReport
from face_compare.ui import FaceCompareApp
from face_compare.camera import CameraDevice
from face_compare.models import BoundingBox, FaceObservation, RecognitionResult


def settle(app):
    """Allow macOS to commit a resize, without an unbounded nested update loop."""
    app.after(120, app.quit)
    app.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="Show idle design with temporary data; never opens a camera")
    args = parser.parse_args()
    config = load_config(ROOT/"config.json")
    with tempfile.TemporaryDirectory(prefix="face-ui-check-") as directory:
        config = replace(config, engine=replace(config.engine, model_directory=str(ROOT/"models")),
                         storage=StorageConfig(str(Path(directory)/"data"), str(Path(directory)/"events.jsonl"), False, config.storage.backend))
        with patch.object(FaceCompareApp, "_begin_camera_scan"), \
             patch("face_compare.ui.messagebox.showwarning"), \
             patch("face_compare.ui.messagebox.showerror"), \
             patch("face_compare.ui.messagebox.showinfo"):
            app = FaceCompareApp(config, ROOT)
            try:
                settle(app)
                if args.preview:
                    app.camera_choice.set("摄像头 0 · 预览模式")
                    app.footer_text.set("界面预览 · 临时空库 · 摄像头未开启")
                    app.start_button.configure(state="disabled")
                    app.scan_button.configure(state="disabled")
                    app.mainloop()
                    return
                app.update_idletasks()
                # Motion is tied to state, has one timer and can be disabled.
                app._set_motion_state("confirming")
                assert "status" in app.motion._motions
                assert app.motion._motions["status"].repeat
                app.reduce_motion_button.invoke()
                assert app.reduce_motion.get() and not app.motion._motions
                app._animate_sample_progress(2)
                assert float(app.sample_progress.cget("value")) == 2
                app._clear_enrollment()
                app.reduce_motion_button.invoke()
                assert not app.reduce_motion.get()
                app._set_motion_state("idle")
                assert app.capture_button.winfo_width() > 50, (app.geometry(), app.workspace_tabs.winfo_width(), app.capture_button.winfo_width())
                assert app.save_button.winfo_width() > 50
                app._device_queue.put(([CameraDevice(0, "外部设备"), CameraDevice(1, "内置设备", "built-in", True)], None))
                app._consume_device_scan()
                assert app.camera_choice.get() == "内置设备"
                app._device_queue.put(([], None))
                app._consume_device_scan()
                assert not app._devices
                assert str(app.start_button.cget("state")) == "disabled"
                for width, height in ((1280, 800), (960, 640)):
                    app.geometry(f"{width}x{height}")
                    quality = QualityReport(False, 120, 40, 42, .179, ("画面模糊",),
                                            "regional", .12, .30, (.1, .12, .15))
                    app._update_result_panel([FaceObservation(BoundingBox(0, 0, 128, 128), quality, None)])
                    assert "分区细节 0.12 / 门槛 0.30" in app.quality_text.get()
                    settle(app)
                    for tab in (2, 1, 0):
                        app.workspace_tabs.select(tab)
                        settle(app)
                        if tab == 1:
                            bottom = app.delete_saved_button.winfo_rooty()-app.winfo_rooty()+app.delete_saved_button.winfo_height()
                            assert bottom <= app.winfo_height()
                    for widget in (app.preview, app.start_button, app.name_entry,
                                   app.capture_button, app.save_button, app.result_label, app.manage_saved_button):
                        x = widget.winfo_rootx()-app.winfo_rootx()
                        y = widget.winfo_rooty()-app.winfo_rooty()
                        assert 0 <= x and x+widget.winfo_width() <= app.winfo_width(), (str(widget), x, widget.winfo_width())
                        assert 0 <= y and y+widget.winfo_height() <= app.winfo_height(), (str(widget), y, widget.winfo_height())
                    assert app.preview.winfo_height() >= 120, (width, height, app.preview.winfo_height(), app.winfo_height())
                    assert app.empty_preview.winfo_ismapped()
                    app._show_frame(np.zeros((480, 640, 3), np.uint8))
                    settle(app)
                    assert not app.empty_preview.winfo_ismapped()
                    app._stop_camera()
                    settle(app)
                    assert app.empty_preview.winfo_ismapped()
                app.person_name.set("UI测试A")
                for index in range(3):
                    crop = np.random.default_rng(index).integers(0, 255, (112, 112, 3), np.uint8)
                    feature = np.zeros(128, np.float32)
                    feature[0] = 1
                    prepared = PreparedSample(crop, feature, QualityReport(True, 128, 40, 100, .2))
                    app.current_frame = crop
                    app._current_frame_time = time.monotonic()
                    app.session._last_time = -float("inf")
                    with patch.object(app.system, "prepare_sample", return_value=prepared):
                        app.capture_button.invoke()
                assert len(app.session.samples) == 3
                assert str(app.name_entry.cget("state")) == "disabled"
                assert len(app.thumbnail_frame.winfo_children()) == 3
                app.after(300, app.quit)
                app.mainloop()
                assert float(app.sample_progress.cget("value")) == 3
                app.person_name.set("UI测试B")
                app.save_button.invoke()
                assert app.system.database.sample_count() == 0
                app.person_name.set("UI测试A")
                app.save_button.invoke()
                assert app.system.database.sample_count() == 3
                assert not app.session.samples
                assert app.person_name.get() == ""
                assert "已为 UI测试A 保存 3 张样本" in app.footer_text.get()
                app._display_stable_result(RecognitionResult(True, "UI测试A", "a", 95, .05, .275, "test"))
                assert "已为 UI测试A 保存 3 张样本" in app.footer_text.get()
                assert float(app.sample_progress.cget("value")) == 0
                # The real result panel must show two independently confirmed tracks.
                observations = []
                for track_id, name in ((1, "UI测试A"), (2, None)):
                    result = RecognitionResult(True, name, "a", 95, .05, .275, "test") if name else RecognitionResult.unknown(threshold=.275, reason="test")
                    observations.append(FaceObservation(BoundingBox(track_id*100, 0, 80, 100), prepared.quality,
                                                       prepared.crop, result, prepared.feature, track_id, result,
                                                       "known" if name else "unknown"))
                app._update_result_panel(observations)
                assert len(app.track_list.get_children()) == 2
                assert "2 张人脸" in app.result_name.get()
                assert "未知人员" in app.track_list.item("1", "values")
                app._stop_camera()
                assert not app.motion._motions.get("status", None) or not app.motion._motions["status"].repeat
                assert not app.track_list.get_children()
                person_a = app.system.database.list_people()[0]
                other = app.system.database.add_samples("UI测试B", [prepared])
                # An unsaved enrollment must survive deletion of a saved person.
                app.session.add("未保存草稿", prepared, now=100)
                app.manage_saved_button.invoke()
                settle(app)
                assert app.workspace_tabs.index(app.workspace_tabs.select()) == 1
                app.saved_people_list.selection_set(person_a.person_id)
                settle(app)
                assert "UI测试A" in app.delete_saved_button.cget("text")
                with patch("face_compare.ui.messagebox.askyesno", return_value=False) as confirm:
                    app.delete_saved_button.invoke()
                    assert "UI测试A" in confirm.call_args.args[1]
                    assert "3 张" in confirm.call_args.args[1]
                assert app.system.database.sample_count() == 4
                generation = app.generation
                with patch("face_compare.ui.messagebox.askyesno", return_value=True):
                    app.delete_saved_button.invoke()
                assert [p.person_id for p in app.system.database.list_people()] == [other.person_id]
                assert len(app.session.samples) == 1
                assert app.session.name == "未保存草稿"
                assert app.generation == generation+1
                assert list((Path(directory)/"data"/"archives").glob("people-*.json"))
                app.saved_people_list.selection_set(other.person_id)
                settle(app)
                with patch("face_compare.ui.messagebox.askyesno", return_value=True):
                    app.delete_saved_button.invoke()
                assert app.system.database.sample_count() == 0
                assert str(app.delete_saved_button.cget("state")) == "disabled"
                assert "还没有" in app.saved_hint.get()
                print("GUI smoke passed: both layouts/tabs, idle/video/stop, enrollment/save, visible deletion entry, cancel/confirm, delete selected only, preserve unsaved samples, backup and empty state; no camera opened.", flush=True)
            finally:
                try:
                    if app.winfo_exists():
                        app._on_close()
                except Exception:
                    pass  # Preview may have already destroyed the Tk interpreter.


if __name__ == "__main__":
    main()
