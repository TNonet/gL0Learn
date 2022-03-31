import random
from typing import Callable, Tuple

import numpy as np
import pytest
from gl0learn import fit, synthetic
from gl0learn.metrics import nonzeros
from gl0learn.opt import MIO_mosek
from hypothesis import given, settings, HealthCheck, assume, note
from hypothesis.strategies import just, booleans, floats, integers, random_module

from tests.utils.utils import (
    _sample_data,
    _sample_data2,
    sample_from_cov,
    overlap_covariance_matrix,
    is_scipy_installed,
    is_mosek_installed,
    make_bisect_func,
    random_penalty,
    random_penalty_values,
    top_n_triu_indicies,
)


@given(n=integers(3, 10), module=random_module())
def test_cd_limited_active_set(n, module):
    theta_truth = overlap_covariance_matrix(p=n, seed=module.seed, decay=0.8)
    x = sample_from_cov(theta_truth)
    _, _, _, _, Y, _ = synthetic.preprocess(x, assume_centered=False, cholesky=True)
    results = fit(
        x,
        l0=0,
        scale_x=True,
        max_active_set_size=1,
        initial_active_set=np.inf,
        super_active_set=0.0,
    )

    theta_truth_copy = np.copy(theta_truth)
    np.fill_diagonal(theta_truth_copy, 0)
    i, j = np.unravel_index(np.argmax(theta_truth_copy), theta_truth.shape)

    assert results.theta[i, j] > np.mean(theta_truth_copy)


@pytest.mark.skipif(not is_scipy_installed(), reason="`scipy` is not installed.")
@pytest.mark.parametrize("nnz", range(1, 10))
@pytest.mark.parametrize("algorithm", ["CD", "CDPSI"])
@given(
    p=integers(3, 10),
    module=random_module(),
    lXs=random_penalty_values(
        penalty_strategies=random_penalty(l0=just(False), l1=booleans(), l2=booleans()),
        values_strategies={"l1": floats(0.01, 10), "l2": floats(0.01, 10)},
    ),
)
@settings(max_examples=1000)
def test_cd_example_2(p, module, nnz, algorithm, lXs):
    theta_truth = overlap_covariance_matrix(p=p, seed=module.seed, decay=0.8)
    x = sample_from_cov(n=30 * p**2, cov=theta_truth)

    _, _, _, _, Y, _ = synthetic.preprocess(x, assume_centered=False, cholesky=True)

    fit_kwargs = dict(
        **lXs,
        scale_x=False,
        max_active_set_size=p * (p - 1) // 2,
        initial_active_set=0.0,
        super_active_set=0.0,
        algorithm=algorithm
    )

    f = make_bisect_func(nnz, Y, **fit_kwargs)

    from scipy.optimize import bisect

    try:
        opt_l0 = bisect(f, a=0, b=10)
    except ValueError:
        assume(False)

    results = fit(Y, l0=opt_l0, **fit_kwargs)

    theta = results.theta

    assume(nonzeros(theta[np.tril_indices(p, k=-1)]).sum() == nnz)

    cd_indices = top_n_triu_indicies(results.theta, nnz)
    indices = top_n_triu_indicies(theta_truth, nnz)

    if any(theta_truth[cd_indices] == 0):
        # CD algorithm has selected zero items. This can be fine if we ask for more non-zeros than are in theta_truth!
        # Check if indicies is contained in cd_indices
        indices_set = set(zip(*indices))
        cd_indices_set = set(zip(*cd_indices))
        assert cd_indices_set.issuperset(indices_set)
        should_be_zero_indices_set = cd_indices_set - indices_set

        for (i, j) in should_be_zero_indices_set:
            assert theta_truth[i, j] == 0

    else:
        np.testing.assert_array_equal(cd_indices, indices)


@pytest.mark.parametrize("algorithm", ["CD", "CDPSI"])
@given(
    p=integers(3, 10),
    module=random_module(),
    lXs=random_penalty_values(
        penalty_strategies=random_penalty(l0=just(True), l1=booleans(), l2=booleans()),
        values_strategies={
            "l0": floats(0.01, 10),
            "l1": floats(0.01, 10),
            "l2": floats(0.01, 10),
        },
    ),
)
@settings(max_examples=1000)
def test_super_active_set(algorithm, p, module, lXs):
    theta_truth = overlap_covariance_matrix(p=p, seed=module.seed, decay=0.8)
    x = sample_from_cov(n=30 * p**2, cov=theta_truth)

    _, _, _, _, Y, _ = synthetic.preprocess(x, assume_centered=False, cholesky=True)

    test_result = fit(
        Y,
        **lXs,
        initial_active_set=np.inf,
        super_active_set=0.0,
        max_active_set_size=p**2
    )

    print("----->", test_result.active_set_size[-1])
    assume(test_result.active_set_size[-1] > 0)

    possible_active_set = np.where(np.abs(np.triu(test_result.theta, k=1)) > 0)

    possible_active_set = np.asarray(possible_active_set).T
    active_set_size = test_result.active_set_size[-1]
    if possible_active_set.shape[0] > 1:
        num_selected = np.random.randint(1, active_set_size)
        idx = np.sort(
            np.random.choice(
                np.arange(active_set_size), size=num_selected, replace=False
            )
        )
    else:
        num_selected = 1
        idx = [0]

    initial_super_active_set = possible_active_set[idx, :]

    lXs["l0"] = 0

    theta_init = np.diag(np.diag(test_result.theta))
    for row, col in initial_super_active_set:
        theta_init[row, col] = theta_init[col, row] = test_result.theta[row, col]

    results = fit(
        Y,
        **lXs,
        theta_init=theta_init,
        initial_active_set=initial_super_active_set,
        super_active_set=initial_super_active_set,
        max_active_set_size=p**2
    )

    cd_indices = top_n_triu_indicies(results.theta, num_selected)

    np.testing.assert_array_equal(np.asarray(cd_indices).T, initial_super_active_set)


@pytest.mark.skipif(not is_mosek_installed(), reason="`mosek` is not installed.")
@given(
    p=integers(3, 10),
    overlaps=integers(1, 5),
    module=random_module(),
    lXs=random_penalty_values(
        penalty_strategies=random_penalty(l0=just(True), l1=just(False), l2=just(True)),
        values_strategies={"l0": floats(0.01, 10), "l2": floats(0.01, 10)},
    ),
)
def test_cd_vs_mosek_high_data(p, module, overlaps, lXs):
    num_samples = 30 * p**2
    theta_truth = overlap_covariance_matrix(
        p=p, seed=module.seed, max_overlaps=overlaps, decay=1 - np.exp(overlaps - 6)
    )

    assume(all(np.linalg.eigvalsh(theta_truth) > 0))
    x = sample_from_cov(n=num_samples, cov=theta_truth)

    _, _, _, _, Y, _ = synthetic.preprocess(x, assume_centered=False, cholesky=True)

    M = np.max(np.abs(theta_truth * (1 - np.eye(p))))
    int_tol = 1e-4

    results = MIO_mosek(Y, M=M, l0=lXs["l0"], l2=lXs["l2"], int_tol=int_tol, maxtime=10)
