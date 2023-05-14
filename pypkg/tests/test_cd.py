from copy import deepcopy
import warnings


import numpy as np
import pytest
from gl0learn import fit, synthetic, Penalty
from gl0learn.metrics import nonzeros, pseudo_likelihood_loss
from gl0learn.utils import triu_nnz_indicies
from hypothesis import given, settings, assume, HealthCheck
from hypothesis.strategies import just, booleans, floats, integers, random_module
from conftest import MAX_OVERLAPS


from utils import (
    sample_from_cov,
    overlap_covariance_matrix,
    is_scipy_installed,
    is_mosek_installed,
    make_bisect_func,
    random_penalty,
    random_penalty_values,
    top_n_triu_indicies_by_abs_value,
)


@given(p=integers(3, 10), module=random_module())
def test_cd_limited_active_set(p, module):
    theta_truth = overlap_covariance_matrix(p=p, seed=module.seed, decay=0.8)
    x = sample_from_cov(theta_truth, n=1000)
    _, _, _, _, Y, _ = synthetic.preprocess(x, assume_centered=False, cholesky=True)
    results = fit(
        x,
        l0=0,
        scale_x=True,
        max_active_set_ratio=1,
        active_set=np.inf,
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
@settings(max_examples=100, deadline=None)
def test_cd_example_2(p, module, nnz, algorithm, lXs):
    theta_truth = overlap_covariance_matrix(p=p, seed=module.seed, decay=0.8)
    x = sample_from_cov(n=30 * p**2, cov=theta_truth)

    _, _, _, _, Y, _ = synthetic.preprocess(x, assume_centered=False, cholesky=True)

    fit_kwargs = dict(
        **lXs,
        scale_x=False,
        max_active_set_ratio=1.0,
        active_set=0.0,
        super_active_set=0.0,
        algorithm=algorithm,
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

    cd_indices = top_n_triu_indicies_by_abs_value(results.theta, nnz)
    indices = top_n_triu_indicies_by_abs_value(theta_truth, nnz)

    if any(theta_truth[cd_indices] == 0):
        # CD algorithm has selected zero items. This can be fine if we ask for more non-zeros than are in theta_truth!
        # Check if indicies is contained in cd_indices
        indices_set = set(zip(*indices))
        cd_indices_set = set(zip(*cd_indices))
        assert cd_indices_set.issuperset(indices_set)
        should_be_zero_indices_set = cd_indices_set - indices_set

        for i, j in should_be_zero_indices_set:
            assert theta_truth[i, j] == 0

    else:
        np.testing.assert_array_equal(cd_indices, indices)


@pytest.mark.parametrize("algorithm", ["CD", "CDPSI"])
@given(
    p=integers(3, 10),
    module=random_module(),
    overlaps=integers(1, MAX_OVERLAPS - 1),
    lXs=random_penalty_values(
        penalty_strategies=random_penalty(l0=just(True), l1=booleans(), l2=booleans()),
        values_strategies={
            "l0": floats(0.01, 10),
            "l1": floats(0.01, 10),
            "l2": floats(0.01, 10),
        },
    ),
)
@settings(suppress_health_check=[HealthCheck(2)], deadline=None)
def test_super_active_set(algorithm, p, module, overlaps, lXs):
    # TODO: Figure out this hypothesis bug. When lXs aren't deep copied, the tracebacks provided by hypothesis are wrong.
    lX2s = deepcopy(lXs)
    theta_truth = overlap_covariance_matrix(
        p=p,
        seed=module.seed,
        max_overlaps=overlaps,
        decay=1 - np.exp(overlaps - MAX_OVERLAPS),
    )
    x = sample_from_cov(n=30 * p**2, cov=theta_truth)

    _, _, _, _, Y, _ = synthetic.preprocess(x, assume_centered=False, cholesky=True)

    test_result = fit(
        Y, **lXs, active_set=np.inf, super_active_set=0.0, max_active_set_ratio=1.0
    )

    assume(test_result.active_set_size[-1] > 0)

    # replace with triu_nnz_indicies
    # print(test_result.theta)
    possible_active_set = np.asarray(
        np.where(np.abs(np.triu(test_result.theta, k=1)) > 0)
    ).T

    active_set_size = possible_active_set.shape[0]
    # print(active_set_size)
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

    # print(idx)
    # print(possible_active_set)

    initial_super_active_set = possible_active_set[idx, :]

    lX2s["l0"] = 0

    theta_init = np.diag(np.diag(test_result.theta))
    for row, col in initial_super_active_set:
        theta_init[row, col] = theta_init[col, row] = test_result.theta[row, col]

    results = fit(
        Y,
        **lX2s,
        theta_init=theta_init,
        active_set=initial_super_active_set,
        super_active_set=initial_super_active_set,
        max_active_set_ratio=1.0,
    )

    cd_indices = top_n_triu_indicies_by_abs_value(results.theta, num_selected)

    np.testing.assert_array_equal(np.asarray(cd_indices).T, initial_super_active_set)


@pytest.mark.skipif(not is_mosek_installed(), reason="`mosek` is not installed.")
@given(
    p=integers(3, 10),
    n=floats(0, 1000),
    overlaps=integers(1, MAX_OVERLAPS - 1),
    module=random_module(),
    lXs=random_penalty_values(
        penalty_strategies=random_penalty(l0=just(True), l1=just(False), l2=just(True)),
        values_strategies={"l0": floats(0.1, 10), "l2": floats(0.1, 10)},
    ),
)
@settings(deadline=None, suppress_health_check=[HealthCheck(2)])
def test_cd_vs_mosek(n, p, module, overlaps, lXs):
    from gl0learn.opt import MIO_mosek
    from mosek.fusion import SolutionError

    num_samples = max(1, int(n * p**2))
    theta_truth = overlap_covariance_matrix(
        p=p,
        seed=module.seed,
        max_overlaps=overlaps,
        decay=1 - np.exp(overlaps - MAX_OVERLAPS),
    )

    assume(all(np.linalg.eigvalsh(theta_truth) > 0))
    x = sample_from_cov(n=num_samples, cov=theta_truth)

    _, _, _, _, y, _ = synthetic.preprocess(x, assume_centered=False, cholesky=True)
    y = np.asfortranarray(y)

    m = np.max(np.abs(theta_truth * (1 - np.eye(p))))
    int_tol = 1e-4

    try:
        with warnings.catch_warnings(record=True):
            MIO_results = MIO_mosek(y=y, m=m, **lXs, int_tol=int_tol, max_time=10)
    except SolutionError:
        assume(False)

    cd_results = fit(
        y,
        **lXs,
        scale_x=False,
        theta_init=None,
        active_set=0.0,
        max_iter=1000,
        super_active_set=0.0,
        max_active_set_ratio=1.0,
        tol=1e-12,
    )

    MIO_active_set = triu_nnz_indicies(MIO_results.theta_hat)
    CD_active_set = triu_nnz_indicies(cd_results.theta)

    if MIO_active_set.shape != CD_active_set.shape:
        assume(False)
    else:
        assume((MIO_active_set == CD_active_set).all())

    penalty = Penalty(**lXs)
    MIO_loss = pseudo_likelihood_loss(
        y, np.asfortranarray(MIO_results.theta_hat), penalty, active_set=MIO_active_set
    )
    cd_loss = pseudo_likelihood_loss(
        y, np.asfortranarray(cd_results.theta), penalty, active_set=CD_active_set
    )

    assert cd_loss <= MIO_loss + 1e-6 * abs(MIO_loss)


@pytest.mark.skipif(not is_mosek_installed(), reason="`mosek` is not installed.")
@pytest.mark.parametrize("max_iter", [1, 1000])
@pytest.mark.parametrize("algorithm", ["CD", "CDPSI"])
@given(
    n=integers(3, 1000),
    p=integers(3, 10),
    overlaps=integers(1, MAX_OVERLAPS - 1),
    module=random_module(),
    lXs=random_penalty_values(
        penalty_strategies=random_penalty(l0=just(True), l1=just(False), l2=just(True)),
        values_strategies={"l0": floats(0.1, 10), "l2": floats(0.1, 10)},
    ),
)
@settings(max_examples=250, deadline=None)
def test_cd_keeps_mio_results(max_iter, algorithm, n, p, module, overlaps, lXs):
    from gl0learn.opt import MIO_mosek

    theta_truth = overlap_covariance_matrix(
        p=p,
        seed=module.seed,
        max_overlaps=overlaps,
        decay=1 - np.exp(overlaps - MAX_OVERLAPS),
    )

    assume(all(np.linalg.eigvalsh(theta_truth) > 0))
    x = sample_from_cov(n=n, cov=theta_truth)

    _, _, _, _, y, _ = synthetic.preprocess(x, assume_centered=False, cholesky=True)

    m = np.max(np.abs(theta_truth * (1 - np.eye(p))))
    int_tol = 1e-4

    with warnings.catch_warnings(record=True):
        MIO_results = MIO_mosek(y=y, m=m, **lXs, int_tol=int_tol)
    cd_results = fit(
        y,
        **lXs,
        scale_x=False,
        theta_init=MIO_results.theta_hat,
        max_iter=max_iter,
        algorithm=algorithm,
        active_set=0.0,
        super_active_set=0.0,
        max_active_set_ratio=1.0,
    )

    try:
        np.testing.assert_array_equal(MIO_results.theta_hat, cd_results.theta)
    except AssertionError:
        penalty = Penalty(**lXs)
        active_set = np.asarray(np.triu_indices(p, k=1), order="C", dtype=np.uint64).T
        MIO_loss = pseudo_likelihood_loss(
            y, np.array(MIO_results.theta_hat), penalty, active_set=active_set
        )
        cd_loss = pseudo_likelihood_loss(
            y, np.array(cd_results.theta), penalty, active_set=active_set
        )

        assert cd_loss <= MIO_loss


@pytest.mark.skipif(not is_mosek_installed(), reason="`mosek` is not installed.")
@pytest.mark.parametrize("algorithm", ["CD", "CDPSI"])
@given(
    n=integers(3, 1000),
    p=integers(3, 10),
    overlaps=integers(1, MAX_OVERLAPS - 1),
    module=random_module(),
    lXs=random_penalty_values(
        penalty_strategies=random_penalty(l0=just(True), l1=just(False), l2=just(True)),
        values_strategies={"l0": floats(0.1, 10), "l2": floats(0.1, 10)},
    ),
)
@settings(max_examples=250, deadline=None)
def test_cd_learns_mio_results_from_support(algorithm, n, p, module, overlaps, lXs):
    # note({"n": n, "p": p, "module":module, overlaps: "overlaps", "lXs": lXs})
    from gl0learn.opt import MIO_mosek

    theta_truth = overlap_covariance_matrix(
        p=p,
        seed=module.seed,
        max_overlaps=overlaps,
        decay=1 - np.exp(overlaps - MAX_OVERLAPS),
    )

    assume(all(np.linalg.eigvalsh(theta_truth) > 0))
    x = sample_from_cov(n=n, cov=theta_truth)

    _, _, _, _, y, _ = synthetic.preprocess(x, assume_centered=False, cholesky=True)

    m = np.max(np.abs(theta_truth * (1 - np.eye(p))))
    int_tol = 1e-4

    with warnings.catch_warnings(record=True):
        MIO_results = MIO_mosek(y=y, m=m, **lXs, int_tol=int_tol)
    active_set = triu_nnz_indicies(MIO_results.theta_hat)
    cd_results = fit(
        y,
        **lXs,
        scale_x=False,
        theta_init=None,
        max_iter=1000,
        algorithm=algorithm,
        active_set=active_set,
        super_active_set=active_set,
        max_active_set_ratio=1.0,
    )

    penalty = Penalty(**lXs)
    MIO_loss = pseudo_likelihood_loss(
        y, np.array(MIO_results.theta_hat), penalty, active_set=active_set
    )
    cd_loss = pseudo_likelihood_loss(
        y, np.array(cd_results.theta), penalty, active_set=active_set
    )

    assert cd_loss <= MIO_loss + int_tol * abs(MIO_loss)
