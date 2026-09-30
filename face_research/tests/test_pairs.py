import hashlib
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from face_research.evaluation.pairs import score_pair_manifest


def _manifest(tmp_path):
    for name, value in (("a1.png", 10), ("a2.png", 20), ("b1.png", 100)):
        assert cv2.imwrite(str(tmp_path / name), np.full((16, 16, 3), value, np.uint8))
    pairs = [
        dict(left_image="a1.png", right_image="a2.png", left_subject_id="A", right_subject_id="A",
             left_session_id="day1", right_session_id="day2", genuine=True),
        dict(left_image="a1.png", right_image="b1.png", left_subject_id="A", right_subject_id="B",
             left_session_id="day1", right_session_id="day1", genuine=False),
    ]
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps({"split": "test", "pairs": pairs}), encoding="utf-8")
    return path


class _System:
    extractor = SimpleNamespace(distance=lambda left, right: float((1 - left @ right) / 2))

    def prepare_sample(self, image):
        value = image[0, 0, 0]
        return SimpleNamespace(feature=np.array([1, 0] if value < 50 else [0, 1], np.float32),
                               crop=image.copy())


def test_pair_manifest_scores_separate_genuine_and_impostor(tmp_path):
    path = _manifest(tmp_path)
    result = score_pair_manifest(_System(), path)
    assert result["usable_pairs"] == 2
    assert [item.distance for item in result["pairs"]] == [0, 0.5]
    assert result["pairs"][1].attempt_id == "a\0day1"
    assert result["acquisition_failures"] == {"genuine": 0, "impostor": 0}
    raw = json.loads(path.read_text())
    raw["split"] = "validation"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert score_pair_manifest(_System(), path, split="validation")["usable_pairs"] == 2
    with pytest.raises(ValueError, match="split=test"):
        score_pair_manifest(_System(), path)
    with pytest.raises(ValueError, match="会话级泄漏"):
        score_pair_manifest(_System(), path, split="validation", forbidden_sessions={("a", "day1")})


def test_pair_manifest_blocks_gallery_reuse_and_bad_labels(tmp_path):
    path = _manifest(tmp_path)
    image = cv2.imread(str(tmp_path / "a1.png"))
    digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
    with pytest.raises(ValueError, match="重复"):
        score_pair_manifest(_System(), path, forbidden_hashes={digest})
    raw = json.loads(path.read_text())
    raw["pairs"][0]["genuine"] = False
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="矛盾"):
        score_pair_manifest(_System(), path)


def test_pair_acquisition_failure_is_not_counted_as_correct_rejection(tmp_path):
    path = _manifest(tmp_path)
    class Failing(_System):
        def prepare_sample(self, image):
            if image[0, 0, 0] == 100:
                raise ValueError("画质不足")
            return super().prepare_sample(image)
    result = score_pair_manifest(Failing(), path)
    assert result["usable_pairs"] == 1
    assert result["acquisition_failures"]["impostor"] == 1


@pytest.mark.parametrize("reverse", [False, True])
def test_duplicate_pairs_cannot_inflate_sample_size(tmp_path, reverse):
    path = _manifest(tmp_path)
    raw = json.loads(path.read_text())
    repeated = raw["pairs"][1].copy()
    if reverse:
        for suffix in ("image", "subject_id", "session_id"):
            repeated[f"left_{suffix}"], repeated[f"right_{suffix}"] = (
                repeated[f"right_{suffix}"], repeated[f"left_{suffix}"]
            )
    raw["pairs"].append(repeated)
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="验证对重复"):
        score_pair_manifest(_System(), path)


@pytest.mark.parametrize("field,value", [("left_subject_id", "C"), ("left_session_id", "another-session")])
def test_one_image_cannot_be_relabelled_as_another_identity_or_session(tmp_path, field, value):
    path = _manifest(tmp_path)
    raw = json.loads(path.read_text())
    raw["pairs"][1][field] = value
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="矛盾的人员/批次"):
        score_pair_manifest(_System(), path)
