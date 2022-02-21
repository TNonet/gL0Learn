import numpy as np
import numpy.testing
import pandas as pd
import pytest
from gl0learn import Bounds
from gl0learn.utils import ClosedInterval


@pytest.mark.parametrize(
    "bounds", [(0, 0), (-1, -1), (1, 1), (np.NAN, np.NAN), (np.NAN, 1), (-1, np.NAN)]
)
def test_scalar_bad_bounds(bounds):
    with pytest.raises(ValueError):
        _ = Bounds(*bounds)


@pytest.mark.parametrize(
    "bounds",
    [
        (np.zeros([2, 2]), np.zeros([2, 2])),
        (-np.ones([2, 2]), -np.ones([2, 2])),
        (np.ones([2, 2]), np.ones([2, 2])),
        (np.NAN * np.ones([2, 2]), np.NAN * np.ones([2, 2])),
        (np.NAN * np.ones([2, 2]), np.ones([2, 2])),
        (-np.ones([2, 2]), np.NAN * np.ones([2, 2])),
        (-np.ones([3, 1]), np.ones([2, 2])),
        (-np.ones([2, 2]), np.ones([3, 1])),
        (-np.ones([3, 3]), np.ones([2, 2])),
        (-np.ones([3, 3]), np.arange(0, 9).reshape(3, 3)),
        (-np.arange(0, 9).reshape(3, 3), np.ones([3, 3])),
    ],
)
def test_matrix_bad_bounds(bounds):
    with pytest.raises(ValueError):
        _ = Bounds(*bounds)


@pytest.mark.parametrize(
    "bounds",
    [
        (np.zeros([2, 2]), 0),
        (0, np.zeros([2, 2])),
        (-1, -np.ones([2, 2])),
        (-np.ones([2, 2]), -1),
        (np.ones([2, 2]), 1),
        (1, np.ones([2, 2])),
        (np.NAN, np.NAN * np.ones([2, 2])),
        (np.NAN * np.ones([2, 2]), np.NAN),
        (np.NAN, np.ones([2, 2])),
        (np.NAN * np.ones([2, 2]), 1),
        (-np.ones([2, 2]), np.NAN),
        (-1, np.NAN * np.ones([2, 2])),
    ],
)
def test_mixed_bad_bounds(bounds):
    with pytest.raises(ValueError):
        _ = Bounds(*bounds)


def test_good_bounds_ex1():
    bounds = (-1, 1)
    b = Bounds(*bounds)
    numpy.testing.assert_equal(bounds[0], b.lows)
    numpy.testing.assert_equal(bounds[1], b.highs)
    assert b.num_features == ClosedInterval(1, np.inf)


def test_good_bounds_ex2(n=2):
    bounds = (-1, np.ones([n, n]))
    b = Bounds(*bounds)
    numpy.testing.assert_equal(-np.ones([n, n]), b.lows)
    numpy.testing.assert_equal(bounds[1], b.highs)
    # assert not numpy.shares_memory(bounds[1], b.highs). Unsure if this is the correct way to check for memory
    # locations

    assert b.num_features == n


def test_good_bounds_ex3(n=2):
    bounds = (-1, np.ones([n, n], order="F"))
    b = Bounds(*bounds)
    numpy.testing.assert_equal(-np.ones([n, n]), b.lows)
    numpy.testing.assert_equal(bounds[1], b.highs)
    # assert numpy.shares_memory(bounds[1], b.highs). Unsure if this is the correct way to check for memory locations
