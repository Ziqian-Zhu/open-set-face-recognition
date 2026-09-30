import pytest

from face_compare.decision import accept_identity


def test_inclusive_threshold_and_margin():
    assert accept_identity(0.275, 0.315, 0.275, 0.04)
    assert not accept_identity(0.2751, 0.5, 0.275, 0.04)
    assert not accept_identity(0.25, 0.289, 0.275, 0.04)


def test_single_identity_has_no_runner_up_margin():
    assert accept_identity(0.2, None, 0.275, 0.5)


@pytest.mark.parametrize("args", [
    (float("nan"), None, 0.3, 0.0),
    (0.2, float("inf"), 0.3, 0.0),
    (0.2, None, float("nan"), 0.0),
    (0.2, None, 0.3, -0.01),
])
def test_invalid_decision_numbers_fail(args):
    with pytest.raises(ValueError, match="无效"):
        accept_identity(*args)
