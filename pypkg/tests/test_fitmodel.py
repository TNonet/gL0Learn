import numpy as np
from gl0learn.fitmodel import FitModel
from gl0learn.gl0learn import _fitmodel
from hypothesis import given
from hypothesis.strategies import lists, floats, integers
from hypothesis.extra import numpy as npst

from utils import numpy_as_fortran


@given(
    theta=npst.arrays(
        dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=2)
    ),
    R=npst.arrays(
        dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=2)
    ),
    costs=lists(floats(), min_size=1),
    active_set_size=lists(integers(0, 2**64 - 1), min_size=1),
)
def test_fitmodel_eq(theta, R, costs, active_set_size):
    """
    const arma::mat &, const arma::mat &,
                    const std::vector<double> &,
                    const std::vector<std::size_t>>

    """
    f1 = _fitmodel(theta, R, costs, active_set_size)
    f2 = _fitmodel(theta, R, costs, active_set_size)
    assert FitModel(f1) == FitModel(f2)


def test_fitmodel_neq():
    """
    const arma::mat &, const arma::mat &,
                    const std::vector<double> &,
                    const std::vector<std::size_t>>

    """
    f1 = _fitmodel(np.zeros([2, 2]), np.zeros([2, 2]), [1.2], [0, 1])
    f2 = _fitmodel(np.zeros([3, 3]), np.zeros([2, 2]), [1.2], [0, 1])
    assert FitModel(f1) != FitModel(f2)
