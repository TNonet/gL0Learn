from typing import Union, Tuple, overload

import numpy as np
import pytest

from hypothesis.extra import numpy as npst
from hypothesis import strategies as st, given
from hypothesis.strategies import floats, tuples, integers
from hypothesis.strategies._internal import SearchStrategy

from gl0learn import Penalty
from gl0learn.utils import ClosedInterval

from utils import numpy_as_fortran


@pytest.mark.parametrize("value", [-1, np.ones([3, 2]), np.arange(4).reshape(2, 2)])
@pytest.mark.parametrize("penalty", ["l0", "l1", "l2"])
@numpy_as_fortran
def test_bad_lx_Penalty(penalty, value):
    with pytest.raises(ValueError):
        _ = Penalty(**{penalty: value})


@pytest.mark.parametrize("value", [0, 1, np.ones([3, 3])])
@pytest.mark.parametrize("penalty", ["l0", "l1", "l2"])
@numpy_as_fortran
def test_lx_Penalty(penalty, value):
    P = Penalty(**{penalty: value})
    np.testing.assert_equal(getattr(P, penalty), value)


@pytest.mark.parametrize("l0, l1", [(0.0, 0.0), (np.ones([2, 2]), np.ones([2, 2]))])
@numpy_as_fortran
def test_l0l1_Penalty(l0, l1):
    P = Penalty(l0, l1)
    np.testing.assert_equal(P.l0, l0)
    np.testing.assert_equal(P.l1, l1)
    if isinstance(l0, float):
        assert P.num_features == ClosedInterval(1, np.inf)
    else:
        assert P.num_features == l0.shape[0]


@pytest.mark.parametrize("l0, l2", [(0.0, 0.0), (np.ones([2, 2]), np.ones([2, 2]))])
@numpy_as_fortran
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
@numpy_as_fortran
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
@numpy_as_fortran
def test_bad_l0l1l2_Penalty(l0, l1, l2):
    with pytest.raises(ValueError):
        _ = Penalty(l0, l1, l2)


@overload
def penalty_and_theta(
    draw: st.DrawFn, shape: None
) -> SearchStrategy[Tuple[float, float]]:
    ...


@overload
def penalty_and_theta(
    draw: st.DrawFn, shape: Union[Tuple[int, ...], SearchStrategy[Tuple[int, ...]]]
) -> Tuple[np.ndarray, np.ndarray]:
    ...


@st.composite
def penalty_and_theta(draw, shape, max_value=1e6):
    penalty_float_strategy = floats(min_value=0, allow_nan=False, max_value=max_value)
    theta_float_strategy = floats(allow_nan=False, max_value=max_value)
    if shape is None:
        return draw(penalty_float_strategy), draw(theta_float_strategy)
    elif not isinstance(shape, tuple):
        shape = draw(shape)

    return draw(
        npst.arrays(dtype=np.float64, shape=shape, elements=penalty_float_strategy)
    ), draw(npst.arrays(dtype=np.float64, shape=shape, elements=theta_float_strategy))


@given(
    l0_theta=penalty_and_theta(
        tuples(integers(min_value=2, max_value=10), integers(min_value=2, max_value=10))
    )
)
@numpy_as_fortran
def test_l0_cost(l0_theta):
    l0, theta = l0_theta
    p = Penalty(l0=l0, validate=False)
    np.testing.assert_almost_equal(
        p.cxx_penalty.penalty_cost(theta), np.sum((theta != 0).astype(int) * l0)
    )
