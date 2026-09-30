"""Live-camera protocol tests use synthetic images and never open a camera."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from face_research.datasets.live_mixed import (
    FIELDS,
    build_manifests,
    capture_session,
    make_template,
    collection_status,
)
from face_research.manifest import (
    check_manifest_splits,
    check_session_isolation,
    read_research_manifest,
)


def filled_pilot(tmp_path: Path):
    index = tmp_path / "pilot.csv"
    assert make_template(index)["rows"] == 34
    with index.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rng = np.random.default_rng(99)
    for row in rows:
        row["session_id"] = f"{row['subject_id']}-{row['split']}-real-session"
        row["consent_confirmed"] = "yes"
        row["captured_at_utc"] = "2026-09-26T12:00:00+00:00"
        image_path = tmp_path / row["image"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        pixels = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        assert cv2.imwrite(str(image_path), pixels)
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return index, rows


def test_pilot_builds_balanced_mixed_session_isolated_manifests(tmp_path, monkeypatch):
    index, _ = filled_pilot(tmp_path)
    output = tmp_path / "manifests"
    original_imread = cv2.imread
    monkeypatch.setattr(cv2, "imread", lambda *_: (_ for _ in ()).throw(
        AssertionError("builder must not read held-out pixels")
    ))
    summary = build_manifests(index, output)
    monkeypatch.setattr(cv2, "imread", original_imread)
    assert summary["image_pixels_read_during_build"] is False
    assert summary["known_people"] == 3
    assert len(list(output.iterdir())) == 3
    val_json = json.loads((output / "mixed.validation.json").read_text())
    test_json = json.loads((output / "mixed.test.json").read_text())
    assert val_json["gallery"] == test_json["gallery"]
    assert len(val_json["gallery"]) == 18
    assert len(val_json["probes"]) == len(test_json["probes"]) == 8
    for manifest in (val_json, test_json):
        assert sum(row["condition"] == "glasses" for row in manifest["gallery"]) == 9
        assert sum(row["condition"] == "no_glasses" for row in manifest["gallery"]) == 9
        assert sum(row["condition"] == "glasses" for row in manifest["probes"]) == 4
        assert sum(row["condition"] == "no_glasses" for row in manifest["probes"]) == 4
        assert all(row["session_id"] for row in manifest["gallery"] + manifest["probes"])
    _, val = read_research_manifest(output / "mixed.validation.json")
    _, test = read_research_manifest(output / "mixed.test.json")
    check_manifest_splits(val, test)
    check_session_isolation(val, test, require_session_ids=True)


def test_missing_consent_and_reused_session_are_rejected(tmp_path):
    index, rows = filled_pilot(tmp_path)
    rows[0]["consent_confirmed"] = ""
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="未确认参与者同意"):
        build_manifests(index, tmp_path / "no-consent")

    rows[0]["consent_confirmed"] = "yes"
    for row in rows:
        if row["subject_id"] == "known_001" and row["split"] == "validation":
            row["session_id"] = "known_001-gallery-real-session"
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="三个不同拍摄批次"):
        build_manifests(index, tmp_path / "same-session")


def test_capture_cannot_start_without_explicit_consent(tmp_path):
    index = tmp_path / "pilot.csv"
    make_template(index)
    with pytest.raises(ValueError, match="参与者同意"):
        capture_session(index, subject_id="known_001", split="gallery",
                        session_id="s1", consent_confirmed=False)


def test_collection_readiness_never_opens_camera_or_test_pixels(tmp_path, monkeypatch):
    empty = tmp_path / "empty.csv"
    make_template(empty)
    ready, _ = filled_pilot(tmp_path / "filled")
    monkeypatch.setattr(cv2, "imread", lambda *_: pytest.fail("status must not read pixels"))
    status = collection_status(empty)
    assert status["existing_images"] == 0
    assert status["total_image_slots"] == 34
    assert status["expected_people"] == 7
    assert status["ready_to_build"] is False
    assert len(status["pending_sessions"]) == 13
    completed = collection_status(ready)
    assert completed["existing_images"] == 34
    assert completed["ready_to_build"] is True
    assert completed["pending_sessions"] == []
    assert completed["accuracy_evaluated"] is False
