import json
from pathlib import Path

import pytest

from face_research.datasets.meglass_pairs import build_pair_protocol


def _metadata(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    lines = []
    for subject in range(10):
        for condition in (0, 1):
            name = f"user{subject}@identity_0@{subject * 2 + condition}_0.jpg"
            # Deliberately invalid image contents: metadata builder must not decode.
            (images / name).write_bytes(b"not decoded")
            lines.append(f"{name} {condition}")
    metadata = tmp_path / "meta.txt"
    metadata.write_text("\n".join(lines))
    return metadata, images


def test_builder_is_metadata_only_deterministic_disjoint_and_create_only(tmp_path):
    metadata, images = _metadata(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    for output in (first, second):
        result = build_pair_protocol(metadata, images, output, subjects_per_split=4)
        assert result["pairs_per_split"] == 6
    splits = []
    for split in ("validation", "test"):
        raw = json.loads((first / f"pairs.{split}.json").read_text())
        other = json.loads((second / f"pairs.{split}.json").read_text())
        assert raw == other
        pairs = raw["pairs"]
        assert sum(row["genuine"] for row in pairs) == 4
        assert not any("session_id" in key for row in pairs for key in row)
        subjects = {row[key] for row in pairs for key in ("left_subject_id", "right_subject_id")}
        photos = {row[key] for row in pairs for key in ("left_source_photo_id", "right_source_photo_id")}
        assert len(subjects) == 4 and len(photos) == 8
        splits.append((subjects, photos))
    assert not splits[0][0] & splits[1][0]
    assert not splits[0][1] & splits[1][1]
    with pytest.raises(FileExistsError):
        build_pair_protocol(metadata, images, first, subjects_per_split=4)


def test_previous_identity_is_excluded(tmp_path):
    metadata, images = _metadata(tmp_path)
    excluded = tmp_path / "excluded.json"
    name = "user0@identity_0@0_0.jpg"
    excluded.write_text(json.dumps({"split": "test", "gallery": [], "probes": [{"image": name}]}))
    result = build_pair_protocol(metadata, images, tmp_path / "pairs", subjects_per_split=4,
                                 exclude_manifests=(excluded,))
    assert result["excluded_subjects"] == 1
    for split in ("validation", "test"):
        raw = json.loads((tmp_path / "pairs" / f"pairs.{split}.json").read_text())
        assert all(not Path(row[key]).name.startswith("user0@") for row in raw["pairs"]
                   for key in ("left_image", "right_image"))


@pytest.mark.parametrize("count", [0, 1, 3, 12, True])
def test_builder_rejects_invalid_or_insufficient_counts(tmp_path, count):
    metadata, images = _metadata(tmp_path)
    with pytest.raises(ValueError):
        build_pair_protocol(metadata, images, tmp_path / "pairs", subjects_per_split=count)
