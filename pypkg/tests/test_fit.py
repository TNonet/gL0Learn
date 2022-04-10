import numpy as np
import pytest
from gl0learn import fit


@pytest.mark.parametrize(
    "x",
    (
        [["not"]],  # not an array
        np,  # not an array
        np.ones([3, 3], dtype=int),  # wrong dtype
        np.ones([3, 3, 3]),  # wrong number of dimensions
        np.ones([3, 1]),  # wrong number of columns
        np.ones([1, 3]),  # wrong number of rows
    ),
)
def test_fit_bad_x(x):
    with pytest.raises(ValueError):
        _ = fit(x)
