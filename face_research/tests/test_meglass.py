"""Adapter tests use synthetic pixels and author-format filenames, not faces."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from face_research.datasets.meglass import (
    GalleryPreflight,
    MeGlassRow,
    _check_source_photo_isolation,
    build_meglass_manifests,
    read_meglass_metadata,
)
from face_compare_system.face_compare.models import PreparedSample, QualityReport
from face_compare_system.face_compare.vector_database import SQLiteVectorDatabase
from face_research.manifest import check_manifest_splits, read_research_manifest


def sample_dataset(tmp_path: Path, *, people: int = 5, each_condition: int = 4):
    images = tmp_path / "images"
    images.mkdir()
    metadata = tmp_path / "meta.txt"
    lines = []
    rng = np.random.default_rng(19)
    for identity_index in range(people):
        for condition in (0, 1):
            for image_index in range(each_condition):
                photo_id = f"{identity_index:03d}{condition}{image_index:03d}"
                filename = (
                    f"{identity_index:08d}@N00_identity_0@{photo_id}_0.jpg"
                )
                image = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
                assert cv2.imwrite(str(images / filename), image)
                lines.append(f"{filename} {condition}")
    metadata.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metadata, images


def test_author_filename_uses_second_at_for_identity(tmp_path):
    metadata = tmp_path / "meta.txt"
    metadata.write_text(
        "10032527@N08_identity_4@2582182573_0.jpg 0\n"
        "10032527@N08_identity_9@2582191559_0.jpg 1\n",
        encoding="utf-8",
    )
    rows = read_meglass_metadata(metadata)
    assert rows[0].identity == "10032527@N08_identity_4"
    assert rows[1].identity == "10032527@N08_identity_9"
    assert rows[0].source_photo_id == "2582182573"
    assert (rows[0].condition, rows[1].condition) == ("no_glasses", "glasses")


@pytest.mark.parametrize("line", [
    "../x@N00_identity_0@123_0.jpg 1",
    "x@N00_identity_0@123_0.jpg 2",
    "x@N00_identity_0@123.jpg 1",
    "x@N00_identity_0@123_0.jpg 1 extra",
])
def test_invalid_author_metadata_is_rejected(tmp_path, line):
    metadata = tmp_path / "meta.txt"
    metadata.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="第1行"):
        read_meglass_metadata(metadata)


def test_two_directions_are_reproducible_and_match_research_schema(tmp_path):
    metadata, images = sample_dataset(tmp_path)
    output = tmp_path / "private-results"
    summary = build_meglass_manifests(
        metadata, images, output_dir=output, image_kind="original",
        known_people=3, unknown_validation_people=1,
        unknown_test_people=1, max_budget=1, seed=7,
    )
    assert summary["session_metadata"].startswith("not_provided")
    assert summary["gallery_images_per_known_identity"] == 2
    assert summary["missing_images"] == 0
    assert len(list(output.iterdir())) == 5
    assert not list(output.glob("*.jpg"))
    for direction, gallery_condition, probe_condition in (
        ("glasses_to_no_glasses", "glasses", "no_glasses"),
        ("no_glasses_to_glasses", "no_glasses", "glasses"),
    ):
        val_path = output / f"{direction}.validation.json"
        test_path = output / f"{direction}.test.json"
        val_json = json.loads(val_path.read_text(encoding="utf-8"))
        test_json = json.loads(test_path.read_text(encoding="utf-8"))
        assert val_json["gallery"] == test_json["gallery"]
        assert all("session_id" not in row for row in val_json["gallery"])
        assert all(row["condition"] == gallery_condition for row in val_json["gallery"])
        assert all(row["condition"] == probe_condition for row in val_json["probes"])
        val_split, val = read_research_manifest(val_path)
        test_split, test = read_research_manifest(test_path)
        assert (val_split, test_split) == ("validation", "test")
        check_manifest_splits(val, test)
        assert {row["subject_id"] for row in val["probes"] if row["label"] is None}.isdisjoint(
            {row["subject_id"] for row in test["probes"] if row["label"] is None}
        )

    second = tmp_path / "another-private-results"
    build_meglass_manifests(
        metadata, images, output_dir=second, image_kind="original",
        known_people=3, unknown_validation_people=1,
        unknown_test_people=1, max_budget=1, seed=7,
    )
    for name in summary["manifests"]:
        assert json.loads((output / name).read_text()) == json.loads((second / name).read_text())
    with pytest.raises(FileExistsError, match="已存在"):
        build_meglass_manifests(
            metadata, images, output_dir=output, image_kind="original",
            known_people=3, unknown_validation_people=1,
            unknown_test_people=1, max_budget=1,
        )


def test_budget_shortage_fails_without_creating_output(tmp_path):
    metadata, images = sample_dataset(tmp_path, each_condition=4)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="符合双向K=3要求"):
        build_meglass_manifests(
            metadata, images, output_dir=output, image_kind="cropped",
            known_people=3, unknown_validation_people=1,
            unknown_test_people=1, max_budget=3,
        )
    assert not output.exists()


def test_gallery_preflight_never_inspects_selected_probes_or_reuses_failed_identity(tmp_path):
    metadata, images = sample_dataset(tmp_path, people=7)
    identities = [f"{index:08d}@N00_identity_0" for index in range(7)]
    random.Random(7).shuffle(identities)
    rejected_identity = identities[0]
    checked = set()

    def checker(row):
        checked.add(row.filename)
        return row.identity != rejected_identity

    output = tmp_path / "screened"
    summary = build_meglass_manifests(
        metadata, images, output_dir=output, image_kind="cropped",
        known_people=3, unknown_validation_people=1,
        unknown_test_people=1, max_budget=1, seed=7,
        gallery_checker=checker, preflight_config_sha256="fake-config",
    )
    assert summary["gallery_preflight"]["known_identities_examined"] >= 4
    assert summary["gallery_preflight"]["candidate_images_rejected"] > 0
    for filename in summary["manifests"]:
        manifest = json.loads((output / filename).read_text())
        for row in manifest["probes"]:
            probe_filename = Path(row["image"]).name
            assert probe_filename not in checked
            assert not probe_filename.startswith(rejected_identity + "@")


def test_group_and_cross_person_preflight_refill_without_inspecting_probes(tmp_path):
    metadata, images = sample_dataset(tmp_path, people=8, each_condition=6)
    inspected = set()
    set_calls = []
    cohort_calls = []

    def check_image(row):
        inspected.add(row.filename)
        return True

    def check_set(rows):
        set_calls.append(tuple(row.filename for row in rows))
        return len(set_calls) != 1

    def check_cohort(name, galleries):
        cohort_calls.append((name, galleries))
        return len(cohort_calls) != 1

    output = tmp_path / "preflight"
    summary = build_meglass_manifests(
        metadata, images, output_dir=output, image_kind="cropped",
        known_people=3, unknown_validation_people=1,
        unknown_test_people=1, max_budget=1, seed=7,
        gallery_checker=check_image, gallery_set_checker=check_set,
        gallery_cohort_checker=check_cohort,
    )
    preflight = summary["gallery_preflight"]
    assert preflight["candidate_sets_rejected"] == 1
    assert preflight["cross_identity_cohorts_rejected"] == 1
    assert preflight["known_identities_examined"] == 4
    assert [name for name, _ in cohort_calls] == ["known_001"] * 2 + ["known_002", "known_003"]
    for filename in summary["manifests"]:
        manifest = json.loads((output / filename).read_text())
        assert not inspected.intersection(Path(row["image"]).name for row in manifest["probes"])


def test_near_duplicate_photos_with_different_source_ids_are_excluded(tmp_path):
    metadata, images = sample_dataset(tmp_path, people=7, each_condition=4)
    identities = [f"{index:08d}@N00_identity_0" for index in range(7)]
    random.Random(7).shuffle(identities)
    duplicate_identity = identities[0]
    repeated = np.full((48, 48, 3), 120, dtype=np.uint8)
    for path in images.glob(f"{duplicate_identity}@*.jpg"):
        assert cv2.imwrite(str(path), repeated)

    output = tmp_path / "screened"
    summary = build_meglass_manifests(
        metadata, images, output_dir=output, image_kind="cropped",
        known_people=3, unknown_validation_people=1,
        unknown_test_people=1, max_budget=1, seed=7,
    )
    assert summary["near_duplicate_screen"]["known_identities_rejected"] == 1
    assert summary["near_duplicate_screen"]["max_hamming"] == 4
    for filename in summary["manifests"]:
        manifest = json.loads((output / filename).read_text())
        all_paths = [row["image"] for row in manifest["gallery"] + manifest["probes"]]
        assert not any(Path(path).name.startswith(duplicate_identity + "@") for path in all_paths)


def test_two_direction_cohort_preflight_rolls_back_both_sqlite_databases(tmp_path):
    databases = {
        condition: SQLiteVectorDatabase(
            tmp_path / condition, save_face_images=False,
            feature_signature="test-rollback", expected_dimension=2,
        )
        for condition in ("glasses", "no_glasses")
    }

    def enroll(database, *, fail):
        def operation(name, samples):
            database.add_samples(name, samples)
            if fail:
                raise ValueError("模拟第二方向录入冲突")
        return operation

    systems = {
        condition: SimpleNamespace(
            database=database,
            enroll_samples=enroll(database, fail=condition == "no_glasses"),
        )
        for condition, database in databases.items()
    }
    preflight = GalleryPreflight(systems)
    rows = {
        condition: MeGlassRow(
            f"a@N00_identity_0@{index}_0.jpg", "a@N00_identity_0", str(index), condition
        )
        for index, condition in enumerate(databases, 1)
    }
    sample = PreparedSample(
        crop=np.zeros((4, 4, 3), dtype=np.uint8),
        feature=np.array([1, 0], dtype=np.float32),
        quality=QualityReport(True, 100, 50, 100, 0.2),
    )
    preflight.cache = {row.filename: sample for row in rows.values()}
    try:
        assert not preflight.check_cohort("known_001", {
            condition: [row] for condition, row in rows.items()
        })
        assert all(not database.list_people() for database in databases.values())
    finally:
        for database in databases.values():
            database.close()


def test_source_photo_id_cannot_cross_splits():
    row_a = MeGlassRow("a@N00_identity_0@42_0.jpg", "a@N00_identity_0", "42", "glasses")
    row_b = MeGlassRow("b@N00_identity_0@42_1.jpg", "b@N00_identity_0", "42", "no_glasses")
    manifest = {
        "split": "validation",
        "gallery": [{"image": row_a.filename}],
        "probes": [{"image": row_b.filename}],
    }
    with pytest.raises(ValueError, match="源照片"):
        _check_source_photo_isolation(
            manifest, {row_a.filename: row_a, row_b.filename: row_b}
        )
