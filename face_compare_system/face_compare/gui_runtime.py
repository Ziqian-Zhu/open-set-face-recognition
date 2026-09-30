"""Keep macOS desktop launches away from Apple's obsolete Tk 8.5 runtime."""

import os
from pathlib import Path
import subprocess
import sys


def ensure_gui_runtime(project_root, arguments=None):
    if sys.platform != "darwin":
        return
    import tkinter
    if tkinter.TkVersion >= 8.6:
        return
    candidate = Path(project_root) / ".venv-ui" / "bin" / "python"
    if candidate.is_file() and str(candidate) != sys.executable:
        check = subprocess.run(
            [str(candidate), "-c", "import tkinter; raise SystemExit(0 if tkinter.TkVersion >= 8.6 else 1)"],
            capture_output=True, timeout=15,
        )
        if check.returncode == 0:
            print("检测到旧版 Tk 8.5，正在切换到项目的兼容界面环境…", flush=True)
            os.execv(str(candidate), [str(candidate), *(sys.argv if arguments is None else arguments)])
    raise RuntimeError("当前 Mac Python 使用不兼容的 Tk 8.5。请使用 .venv-ui/bin/python main.py；"
                       "若该环境已失效，请用带 Tk 8.6+ 的 Python 重建 .venv-ui。")
