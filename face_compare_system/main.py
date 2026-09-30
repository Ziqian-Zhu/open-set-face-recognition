"""Desktop entry point for the intelligent face comparison system."""

from __future__ import annotations

from pathlib import Path

from face_compare.config import load_config
from face_compare.gui_runtime import ensure_gui_runtime


def main() -> None:
    """Load project configuration and launch the Tkinter interface."""

    project_root = Path(__file__).resolve().parent
    ensure_gui_runtime(project_root)
    from face_compare.ui import launch_ui
    config = load_config(project_root / "config.json")
    launch_ui(config, project_root)


if __name__ == "__main__":
    main()
