"""Do not silently open another blank window with the obsolete Apple Tk."""

import tkinter
from unittest.mock import Mock

import pytest

from face_compare import gui_runtime


def test_supported_tk_does_not_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(gui_runtime.sys, "platform", "darwin")
    monkeypatch.setattr(tkinter, "TkVersion", 9.0)
    execute = Mock()
    monkeypatch.setattr(gui_runtime.os, "execv", execute)
    gui_runtime.ensure_gui_runtime(tmp_path)
    execute.assert_not_called()


def test_obsolete_tk_without_replacement_explains_how_to_recover(monkeypatch, tmp_path):
    monkeypatch.setattr(gui_runtime.sys, "platform", "darwin")
    monkeypatch.setattr(tkinter, "TkVersion", 8.5)
    with pytest.raises(RuntimeError, match="Tk 8.5"):
        gui_runtime.ensure_gui_runtime(tmp_path)


def test_obsolete_tk_relaunch_preserves_cli_arguments(monkeypatch, tmp_path):
    monkeypatch.setattr(gui_runtime.sys, "platform", "darwin")
    monkeypatch.setattr(tkinter, "TkVersion", 8.5)
    candidate = tmp_path / ".venv-ui/bin/python"
    candidate.parent.mkdir(parents=True)
    candidate.touch()
    monkeypatch.setattr(gui_runtime.subprocess, "run", Mock(return_value=Mock(returncode=0)))
    execute = Mock(side_effect=SystemExit(0))
    monkeypatch.setattr(gui_runtime.os, "execv", execute)
    arguments = ["-m", "face_compare.cli", "--config", "custom.json", "gui"]
    with pytest.raises(SystemExit):
        gui_runtime.ensure_gui_runtime(tmp_path, arguments)
    execute.assert_called_once_with(str(candidate), [str(candidate), *arguments])
