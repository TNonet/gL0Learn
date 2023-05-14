import warnings
import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis.strategies import integers, random_module, just, floats

from gl0learn.synthetic import preprocess

from utils import (
    random_penalty,
    random_penalty_values,
    overlap_covariance_matrix,
    sample_from_cov,
    is_mosek_installed,
)


@pytest.mark.skipif(not is_mosek_installed(), reason="`mosek` is not installed.")
@given(
    p=integers(3, 10),
    module=random_module(),
    lXs=random_penalty_values(
        penalty_strategies=random_penalty(l0=just(True), l1=just(False), l2=just(True)),
        values_strategies={"l0": floats(0.01, 10), "l2": floats(0.01, 10)},
    ),
)
@settings(deadline=None, max_examples=1000)
def test_init_levels(p, module, lXs):
    import mosek
    from gl0learn.opt import MIO_mosek, mosek_level_values

    theta_truth = overlap_covariance_matrix(p=p, seed=module.seed, decay=0.8)
    x = sample_from_cov(cov=theta_truth, n=30 * p**2, seed=module.seed)

    _, _, _, _, Y, _ = preprocess(x, assume_centered=False, cholesky=False)

    m = np.max(np.abs(theta_truth * (1 - np.eye(p))))
    int_tol = 1e-4
    try:
        with warnings.catch_warnings(record=True):
            results = MIO_mosek(
                Y, m=m, l0=lXs["l0"], l2=lXs["l2"], int_tol=int_tol, max_time=10
            )
    except mosek.MosekException:
        assume(False)
    else:
        theta_tril, z, s, t, lg, residuals = mosek_level_values(
            theta=results.theta_hat, Y=Y, int_tol=int_tol
        )

        np.testing.assert_array_equal(results.theta_hat[np.tril_indices(p)], theta_tril)

        np.testing.assert_array_equal(results.z[np.tril_indices(p)] > int_tol, z)

        decimals = int(-np.log10(int_tol) + 1)

        np.testing.assert_array_almost_equal(results.s, s, decimal=decimals)
        np.testing.assert_array_almost_equal(results.t, t, decimal=decimals)
        np.testing.assert_array_almost_equal(
            results.residuals.reshape(p, p), residuals, decimal=decimals
        )
