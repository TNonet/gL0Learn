from gl0learn.gl0learn import (
    check_is_coordinate_subset,
)
from hypothesis import given
from hypothesis import strategies as st
import numpy as np


@st.composite
def size_and_subset(draw: st.DrawFn, max_columns: int = 10):
    n = draw(st.integers(min_value=1, max_value=max_columns))
    n = (n - 1) * n // 2

    if n > 1:
        subset_indices = draw(st.sets(elements=st.integers(0, n - 1), max_size=n))
    else:
        subset_indices = {}

    return n, np.asarray(sorted(subset_indices), dtype=int)


@given(size_and_subset(max_columns=20))
def test_check_is_coordinate_subset(x):
    n, subset_indices = x
    full = np.asarray(np.triu_indices(n, k=1)).T

    subset = full[subset_indices, :]

    assert check_is_coordinate_subset(full, subset)
