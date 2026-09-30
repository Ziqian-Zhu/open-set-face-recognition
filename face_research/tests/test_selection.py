from types import SimpleNamespace

import numpy as np
import pytest

from face_research.selection import (
    METHODS,
    Template,
    decide,
    quality_proxy,
    build_gallery_index,
    rank_identities,
    rank_with_index,
    select_templates,
)


def template(key, vector, quality):
    return Template(key, np.asarray(vector, dtype=np.float32), quality)


def test_all_methods_obey_same_budget_without_mutating_gallery():
    gallery = [template(str(i), [1, i * 0.1, 0], 0.9 - i * 0.1) for i in range(5)]
    original = list(gallery)
    for method in METHODS:
        selected = select_templates(gallery, method, 2)
        assert len(selected) == 2
        assert len({item.key for item in selected}) == 2
        assert all(item in gallery for item in selected)
    assert gallery == original


def test_random_is_reproducible_and_quality_prefers_high_score():
    gallery = [template(str(i), [1, 0.2 * i], i / 10) for i in range(10)]
    assert [x.key for x in select_templates(gallery, "random", 4, seed=7)] == [
        x.key for x in select_templates(gallery, "random", 4, seed=7)
    ]
    assert [x.key for x in select_templates(gallery, "quality", 2)] == ["9", "8"]


def test_coverage_retains_a_rare_valid_mode():
    gallery = [
        template("a", [1, 0], 0.95),
        template("b", [1, 0.02], 0.9),
        template("c", [1, -0.02], 0.9),
        template("rare", [0.8, 0.6], 0.8),
    ]
    keys = {item.key for item in select_templates(gallery, "coverage", 2)}
    assert keys == {"a", "rare"}


def test_invalid_features_and_budget_are_rejected():
    with pytest.raises(ValueError):
        select_templates([template("a", [0, 0], 0.8)], "first", 1)
    with pytest.raises(ValueError):
        select_templates([template("a", [1, 0], 1.1)], "first", 1)
    with pytest.raises(ValueError):
        select_templates([template("a", [1, 0], 0.8)], "first", 0)
    with pytest.raises(ValueError):
        select_templates([template("a", [1, 0], 0.8)], "first", True)


def test_coverage_handles_zero_proxy_scores_without_nan():
    gallery = [
        template("a", [1, 0], 0),
        template("b", [1, 0.02], 0),
        template("rare", [0.8, 0.6], 0),
    ]
    with np.errstate(all="raise"):
        selected = select_templates(gallery, "coverage", 2)
    assert len({item.key for item in selected}) == 2


def test_matching_uses_nearest_three_median_and_margin():
    gallery = {
        "A": [
            template("a1", [1, 0], 0.8),
            template("a2", [1, 0], 0.8),
            template("a3", [-1, 0], 0.8),
        ],
        "B": [template("b1", [0, 1], 0.8)],
    }
    candidates = rank_identities(gallery, np.asarray([1, 0], np.float32), 3)
    assert candidates[0] == (0, "A")
    assert candidates[1] == (0.5, "B")
    assert decide(candidates, 0.2, 0.04) == "A"
    assert decide(candidates, 0.2, 0.6) is None
    index = build_gallery_index(gallery)
    assert rank_with_index(index, np.asarray([1, 0], np.float32), 3) == candidates
    with pytest.raises(ValueError):
        rank_with_index(index, np.asarray([1, 0], np.float32), 0)
    with pytest.raises(ValueError):
        decide(candidates, float("nan"), 0.04)


def test_quality_proxy_prefers_sharp_accepted_samples():
    config = SimpleNamespace(
        min_brightness=45,
        max_brightness=220,
        min_focus_score=0.3,
        min_blur_variance=55,
        min_contrast=22,
        min_face_ratio=0.035,
    )

    def report(focus):
        return SimpleNamespace(
            accepted=True,
            brightness=132,
            focus_score=focus,
            blur_variance=100,
            contrast=40,
            face_size_ratio=0.1,
        )

    assert (
        0
        <= quality_proxy(report(0.3), config)
        < quality_proxy(report(0.8), config)
        <= 1
    )
    with pytest.raises(ValueError):
        quality_proxy(SimpleNamespace(accepted=False), config)
