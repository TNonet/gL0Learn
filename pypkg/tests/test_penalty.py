import numpy as np
import pytest

from gl0learn import Penalty
from gl0learn.utils import ClosedInterval


@pytest.mark.parametrize("value", [-1, np.ones([3, 2]), np.arange(4).reshape(2, 2)])
@pytest.mark.parametrize("penalty", ["l0", "l1", "l2"])
def test_bad_lx_Penalty(penalty, value):
    with pytest.raises(ValueError):
        _ = Penalty(**{penalty: value})


@pytest.mark.parametrize("value", [0, 1, np.ones([3, 3])])
@pytest.mark.parametrize("penalty", ["l0", "l1", "l2"])
def test_lx_Penalty(penalty, value):
    P = Penalty(**{penalty: value})
    np.testing.assert_equal(getattr(P, penalty), value)


@pytest.mark.parametrize("l0, l1", [(0.0, 0.0), (np.ones([2, 2]), np.ones([2, 2]))])
def test_l0l1_Penalty(l0, l1):
    P = Penalty(l0, l1)
    np.testing.assert_equal(P.l0, l0)
    np.testing.assert_equal(P.l1, l1)
    if isinstance(l0, float):
        assert P.num_features == ClosedInterval(1, np.inf)
    else:
        assert P.num_features == l0.shape[0]


@pytest.mark.parametrize("l0, l2", [(0.0, 0.0), (np.ones([2, 2]), np.ones([2, 2]))])
def test_l0l2_Penalty(l0, l2):
    P = Penalty(l0, l2=l2)
    np.testing.assert_equal(P.l0, l0)
    np.testing.assert_equal(P.l2, l2)
    if isinstance(l0, float):
        assert P.num_features == ClosedInterval(1, np.inf)
    else:
        assert P.num_features == l0.shape[0]


@pytest.mark.parametrize(
    "l0, l1, l2", [(0.0, 0.0, 0.0), (np.ones([2, 2]), np.ones([2, 2]), np.ones([2, 2]))]
)
def test_l0l1l2_Penalty(l0, l1, l2):
    P = Penalty(l0, l1, l2)
    np.testing.assert_equal(P.l0, l0)
    np.testing.assert_equal(P.l1, l1)
    np.testing.assert_equal(P.l2, l2)
    if isinstance(l0, float):
        assert P.num_features == ClosedInterval(1, np.inf)
    else:
        assert P.num_features == l0.shape[0]


@pytest.mark.parametrize(
    "l0, l1, l2", [(0.0, -1.0, 0.0), (np.ones([2, 2]), np.ones([3, 3]), 0)]
)
def test_bad_l0l1l2_Penalty(l0, l1, l2):
    with pytest.raises(ValueError):
        _ = Penalty(l0, l1, l2)
