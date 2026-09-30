"""The mixed protocol is tested with synthetic pixels, not human faces."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from face_research.datasets.meglass import MeGlassRow
from face_research.datasets.meglass_mixed import (
    _has_cross_split_near_duplicate,
    build_meglass_mixed_manifests,
)
from face_research.manifest import check_manifest_splits, read_research_manifest


def synthetic_meglass(tmp_path: Path, *, people: int = 8, each_condition: int = 6):
    images = tmp_path / "images"
    images.mkdir()
    metadata = tmp_path / "meta.txt"
    rng = np.random.default_rng(127)
    lines = []
    for person in range(people):
        for condition in (0, 1):
            for index in range(each_condition):
                photo = f"{person:03d}{condition}{index:03d}"
                name = f"{person:08d}@N00_identity_0@{photo}_0.jpg"
                pixels = rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)
                assert cv2.imwrite(str(images / name), pixels)
                lines.append(f"{name} {condition}")
    metadata.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metadata, images


class AcceptAllPreflight:
    def __init__(self):
        self.inspected = set()

    def __call__(self, row):
        self.inspected.add(row.filename)
        return True

    def check_set(self, rows):
        return True

    def check_cohort(self, name, galleries):
        return True


def test_balanced_mixed_manifest_is_deterministic_and_valid(tmp_path):
    metadata, images = synthetic_meglass(tmp_path)
    checker = AcceptAllPreflight()
    output = tmp_path / "run-a"
    summary = build_meglass_mixed_manifests(
        metadata, images, output_dir=output, image_kind="cropped",
        preflight=checker, preflight_config_sha256="synthetic",
        known_people=3, unknown_validation_people=2,
        unknown_test_people=2, gallery_per_condition=2,
        max_budget=3, seed=7,
    )
    assert summary["gallery_images_per_known_identity"] == 4
    assert not summary["matches_registered_default"]
    assert len(list(output.iterdir())) == 3
    assert not list(output.glob("*.jpg"))
    val_json = json.loads((output / "mixed.validation.json").read_text())
    test_json = json.loads((output / "mixed.test.json").read_text())
    assert val_json["gallery"] == test_json["gallery"]
    assert len(val_json["gallery"]) == 12
    assert len(val_json["probes"]) == len(test_json["probes"]) == 8
    for manifest in (val_json, test_json):
        assert sum(row["condition"] == "glasses" for row in manifest["gallery"]) == 6
        assert sum(row["condition"] == "no_glasses" for row in manifest["gallery"]) == 6
        assert sum(row["condition"] == "glasses" for row in manifest["probes"]) == 4
        assert sum(row["condition"] == "no_glasses" for row in manifest["probes"]) == 4
        assert all("session_id" not in row for row in manifest["gallery"] + manifest["probes"])
        assert not checker.inspected.intersection(
            Path(row["image"]).name for row in manifest["probes"]
        )
    val_split, val = read_research_manifest(output / "mixed.validation.json")
    test_split, test = read_research_manifest(output / "mixed.test.json")
    assert (val_split, test_split) == ("validation", "test")
    check_manifest_splits(val, test)

    second = tmp_path / "run-b"
    build_meglass_mixed_manifests(
        metadata, images, output_dir=second, image_kind="cropped",
        preflight=AcceptAllPreflight(), preflight_config_sha256="synthetic",
        known_people=3, unknown_validation_people=2,
        unknown_test_people=2, gallery_per_condition=2,
        max_budget=3, seed=7,
    )
    for name in summary["manifests"]:
        assert json.loads((output / name).read_text()) == json.loads((second / name).read_text())


def test_mixed_near_duplicate_check_catches_gallery_to_probe(tmp_path):
    pixels = np.full((48, 48, 3), 127, dtype=np.uint8)
    left = tmp_path / "a.jpg"
    right = tmp_path / "b.jpg"
    assert cv2.imwrite(str(left), pixels)
    assert cv2.imwrite(str(right), pixels)
    gallery = [MeGlassRow(left.name, "person", "1", "glasses", left)]
    validation = [MeGlassRow(right.name, "person", "2", "no_glasses", right)]
    assert _has_cross_split_near_duplicate(gallery, validation, [], {})


def test_mixed_rejects_unbalanced_unknown_population(tmp_path):
    metadata, images = synthetic_meglass(tmp_path)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="偶数"):
        build_meglass_mixed_manifests(
            metadata, images, output_dir=output, image_kind="cropped",
            preflight=AcceptAllPreflight(), preflight_config_sha256="synthetic",
            known_people=3, unknown_validation_people=3,
            unknown_test_people=2, gallery_per_condition=2,
            max_budget=3, seed=7,
        )
    assert not output.exists()
