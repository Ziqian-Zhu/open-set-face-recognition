"""The research controls must retain the production top-three default."""

import numpy as np
import pytest

from face_compare.aggregation import STRATEGIES, aggregate_identity_score, aggregate_identity_scores


@pytest.mark.parametrize(
    ("strategy", "k", "expected"),
    [
        ("nearest", 3, 0.1),
        ("mean", 3, 0.4),
        ("median", 3, 0.45),
        ("topk_mean", 2, 0.15),
        ("topk_mean", 3, 0.7 / 3),
        ("topk_median", 3, 0.2),
        ("topk_median", 10, 0.45),
    ],
)
def test_known_aggregation_scores(strategy, k, expected):
    assert aggregate_identity_score([0.6, 0.1, 0.5, 0.2, 0.4, 0.6], strategy, k) == pytest.approx(expected)


def test_single_and_two_templates_preserve_default_semantics():
    assert aggregate_identity_score([0.21]) == pytest.approx(0.21)
    assert aggregate_identity_score([0.21, 0.31]) == pytest.approx(0.26)


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [[0.1, 0.2]]])
def test_bad_template_distances_fail(values):
    with pytest.raises(ValueError, match="模板距离"):
        aggregate_identity_score(values)


def test_invalid_strategy_or_k_fail():
    with pytest.raises(ValueError, match="未知模板聚合策略"):
        aggregate_identity_score([0.1], "unknown")
    for k in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="k 必须"):
            aggregate_identity_score(np.array([0.1]), k=k)


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batch_is_bitwise_equal_to_scalar_for_ragged_budgets_and_ties(strategy, dtype):
    rng = np.random.default_rng(141)
    for count in (1, 2, 3, 5, 7, 8, 17, 64, 257):
        values = rng.uniform(size=(33, count)).astype(dtype)
        values[0] = .275  # threshold equality and exact ties
        for k in (1, 2, 3, 5, 512):
            expected = [aggregate_identity_score(row, strategy, k) for row in values]
            np.testing.assert_array_equal(aggregate_identity_scores(values, strategy, k), expected)
            # Noncontiguous callers must have the same scalar reduction order.
            strided = values[:, ::2]
            expected = [aggregate_identity_score(row, strategy, k) for row in strided]
            np.testing.assert_array_equal(aggregate_identity_scores(strided, strategy, k), expected)


@pytest.mark.parametrize("values", [[], [[]], [1., 2.], [[float("nan")]], [[float("inf")]],
                                    [[1j]], [["1"]], [[True]]])
def test_batch_rejects_invalid_distance_matrices(values):
    with pytest.raises(ValueError, match="批量模板距离"):
        aggregate_identity_scores(values)


def test_batch_rejects_invalid_strategy_and_k():
    with pytest.raises(ValueError, match="未知模板聚合策略"):
        aggregate_identity_scores([[.1]], "unknown")
    for k in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="k 必须"):
            aggregate_identity_scores([[.1]], k=k)
