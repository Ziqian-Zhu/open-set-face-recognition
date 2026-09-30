"""Single import boundary to the unchanged sibling course application."""

from __future__ import annotations

from pathlib import Path

from face_compare_system.face_compare.config import StorageConfig, load_config
from face_compare_system.face_compare.database import normalize_person_name
from face_compare_system.face_compare.evaluation import interval, summarize
from face_compare_system.face_compare.service import FaceComparisonSystem


PROJECT = Path(__file__).resolve().parent.parent / "face_compare_system"
