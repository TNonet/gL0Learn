import numpy as np
import pytest
from gl0learn import fit, synthetic
from hypothesis import given, settings, assume
from hypothesis.strategies import integers, floats, random_module, just, booleans

from conftest import MAX_OVERLAPS
from utils import (
    random_penalty_values,
    random_penalty,
    overlap_covariance_matrix,
    sample_from_cov,
    numpy_as_fortran
)


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
@numpy_as_fortran
def test_fit_bad_x(x):
    with pytest.raises(ValueError):
        _ = fit(x)


@pytest.mark.parametrize("algorithm", ["CD", "CDPSI"])
@given(
    max_iter=integers(1, 1000),
    active_set=floats(0, 2),
    tol=floats(1e-16, 1e-1),
    super_active_set=floats(0, 2),
    p=integers(2, 10),
    n=floats(0, 1000),
    overlaps=integers(1, MAX_OVERLAPS - 1),
    module=random_module(),
    lXs=random_penalty_values(
        penalty_strategies=random_penalty(l0=just(True), l1=booleans(), l2=booleans()),
        values_strategies={
            "l0": floats(0, 10),
            "l1": floats(0, 10),
            "l2": floats(0, 10),
        },
    ),
)
@settings(max_examples=1000, deadline=None)
def test_fit_is_reproducible(
    n, p, max_iter, module, overlaps, active_set, super_active_set, lXs, tol, algorithm
):
    assume(active_set > super_active_set)
    num_samples = max(2, int(n * p**2))
    theta_truth = overlap_covariance_matrix(
        p=p,
        seed=module.seed,
        max_overlaps=overlaps,
        decay=1 - np.exp(overlaps - MAX_OVERLAPS),
    )

    assume(all(np.linalg.eigvalsh(theta_truth) > 0))
    x = sample_from_cov(n=num_samples, cov=theta_truth)

    _, _, _, _, y, _ = synthetic.preprocess(x, assume_centered=False, cholesky=True)

    fit_dict = dict(
        **lXs,
        scale_x=False,
        theta_init=None,
        active_set=active_set,
        max_iter=max_iter,
        seed=module.seed,
        super_active_set=super_active_set,
        max_active_set_ratio=1.0,
        tol=tol
    )

    fit1 = fit(y, **fit_dict)

    fit2 = fit(y, **fit_dict)

    assert fit1 == fit2
