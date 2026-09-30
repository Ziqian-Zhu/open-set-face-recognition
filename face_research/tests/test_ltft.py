import json

import pytest

from face_research.datasets.ltft import parse_annotations, annotation_inventory, audit_annotations, write_audit
from face_research.evaluation.artifacts import verify_result_directory


def test_parser_retains_empty_frames_and_face_zero_without_quality_filtering():
    data = parse_annotations(b"3\n0 0\n1 2 1 0 0 10 20 1 0.2 2 5 5 20 30 0 0.99\n2 0\n")
    assert len(data.frames) == 3
    assert not data.frames[0].detections
    assert data.frames[1].detections[0].face
    assert data.frames[1].detections[0].confidence == .2
    assert not data.frames[1].detections[1].face
    assert len(data.sha256) == 64


@pytest.mark.parametrize("content", [
    b"", b"0\n", b"2\n0 0\n", b"2\n0 0\n0 0\n", b"1\n1 0\n", b"1\n0.0 0\n",
    b"1\n0 1 1 0 0 10 20 1\n", b"1\n0 -1\n", b"1\n0 1 0 0 0 10 20 1 .9\n",
    b"1\n0 1 1 0 0 0 20 1 .9\n", b"1\n0 1 1 nan 0 10 20 1 .9\n",
    b"1\n0 1 1 0 0 10 20 2 .9\n", b"1\n0 1 1 0 0 10 20 1 1.1\n",
    b"1\n0 1 1 0 0 10 20 1 nan\n", b"1\n0 2 1 0 0 10 20 1 .9 1 5 5 10 20 1 .9\n",
])
def test_parser_rejects_corrupt_metadata(content):
    with pytest.raises(ValueError):
        parse_annotations(content)


def _fixture(path, count):
    lines = [str(count), "0 2 1 0 0 10 20 1 .9 2 20 0 10 20 0 .9"]
    lines.extend(f"{i} 0" for i in range(1, count))
    path.write_text("\n".join(lines) + "\n")


def test_inventory_rejects_wrong_source_or_shortened_sequence():
    source = parse_annotations(b"1\n0 0\n")
    with pytest.raises(ValueError, match="帧数"):
        annotation_inventory(source, sequence="choke1")
    with pytest.raises(ValueError, match="ChokePoint"):
        annotation_inventory(source, sequence="street")


def test_audit_is_metadata_only_anonymous_create_only_and_checksummed(tmp_path):
    paths = [tmp_path / name for name in ("choke1.txt", "choke2.txt")]
    for path, count in zip(paths, (2526, 2139), strict=True):
        _fixture(path, count)
    report = audit_annotations(*paths)
    assert report["kind"] == "metadata_only_not_video_evaluation"
    row = report["sequences"][0]
    assert row["frames_without_face1"] == 2525
    assert row["face1_annotations"] == row["face0_annotations"] == row["face1_subjects"] == 1
    assert row["video_bytes_available"] is None
    assert row["session_id"] is None
    assert not row["open_set_gallery_ready"]
    output = write_audit(report, tmp_path / "result")
    assert verify_result_directory(output) == {"status": "verified", "files": 2}
    assert str(tmp_path) not in (output / "metadata.json").read_text()
    assert json.loads((output / "metadata.json").read_text()) == report
    with pytest.raises(FileExistsError):
        write_audit(report, output)
    (output / "sequence_inventory.csv").write_text("changed")
    with pytest.raises(ValueError, match="校验"):
        verify_result_directory(output)
