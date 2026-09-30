"""Single-mainloop startup test: real metadata discovery, no camera sessions."""

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import tkinter as tk
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from face_compare.config import load_config, StorageConfig
from face_compare.ui import FaceCompareApp


def main():
    assert sys.platform != "darwin" or tk.TkVersion >= 8.6, "Obsolete Mac Tk runtime"
    errors = []
    with tempfile.TemporaryDirectory(prefix="face-startup-") as folder:
        config = load_config(ROOT / "config.json")
        config = replace(config, storage=StorageConfig(folder, str(Path(folder)/"events.jsonl"), False, config.storage.backend))
        with patch("face_compare.camera.cv2.VideoCapture", side_effect=AssertionError("Startup must not open a camera")) as capture:
            app = FaceCompareApp(config, ROOT)
            app.report_callback_exception = lambda *exc: errors.append(exc)

            def verify():
                try:
                    assert app.winfo_ismapped()
                    assert app.capture_button.winfo_ismapped()
                    assert app.capture_button.winfo_width() > 50
                    assert app.preview.winfo_width() > 300
                    assert app.preview.winfo_height() > 150
                    assert app.empty_preview.find_all()
                    assert not app._scanning
                    assert not errors, errors
                    capture.assert_not_called()
                    print(f"STARTUP PASS: Python {sys.version.split()[0]}, Tk {app.tk.call('package','require','Tk')}, "
                          f"window={app.geometry()}, preview={app.preview.winfo_width()}x{app.preview.winfo_height()}, "
                          f"device={app.camera_choice.get()}; no camera opened", flush=True)
                except BaseException as error:
                    errors.append(error)
                finally:
                    app._on_close()

            app.after(2500, verify)
            app.mainloop()
    if errors:
        raise RuntimeError(f"Startup failed: {errors}")


if __name__ == "__main__":
    main()
