from typing import List, Dict, Iterable

import hypothesis
import numpy as np
from gl0learn import fit
from hypothesis.strategies import composite


def is_mosek_installed() -> bool:
    try:
        import mosek
    except ModuleNotFoundError:
        return False
    else:
        return True


def is_scipy_installed() -> bool:
    try:
        import scipy
    except ModuleNotFoundError:
        return False
    else:
        return True


def top_n_triu_indicies_by_abs_value(x, n):
    """
    Parameters
    ----------
    n: int
        Number of indicies to return.
        If n is greather than p*(p-1)//2, the number of upper triangluer coordinates, an error is raised
        If there are only k non-zero vaues, st k < n. Only k values are returned.
    """
    if n <= 0:
        raise ValueError(f"Cannot request {n} non-zero items")

    p, p1 = x.shape
    if p != p1:
        raise ValueError(f"x is not a square matrix")

    if n > p * (p - 1) // 2:
        raise ValueError(f"n is to large for a {p} by {p} matrix")

    triu_x = np.abs(np.triu(x, k=1))

    if (triu_x == 0).all():
        raise ValueError("All triu values of x are 0.")

    triu_x_flat = triu_x.flatten()

    non_zero_triu_x = triu_x_flat[np.nonzero(triu_x_flat)]
    nnz = non_zero_triu_x.size
    if np.unique(non_zero_triu_x).size != nnz:
        raise NotImplementedError("Not implemented for arrays with duplicate values")

    sorted_triu_values = np.sort(triu_x_flat)[::-1]

    if sorted_triu_values[n] == 0:
        n = np.where(sorted_triu_values == 0)[0][0] - 1
        return np.where(triu_x >= sorted_triu_values[n])

    return np.where(triu_x > sorted_triu_values[n])


@composite
def random_penalty(
    draw,
    l0: hypothesis.strategies.SearchStrategy[bool],
    l1: hypothesis.strategies.SearchStrategy[bool],
    l2: hypothesis.strategies.SearchStrategy[bool],
) -> List[str]:
    penalties = []

    if draw(l0):
        penalties.append("l0")

    if draw(l1):
        penalties.append("l1")

    if draw(l2):
        penalties.append("l2")

    return penalties


@composite
def random_penalty_values(
    draw,
    values_strategies: Dict[str, hypothesis.strategies.SearchStrategy[float]],
    penalty_strategies: hypothesis.strategies.SearchStrategy[Iterable[str]],
) -> Dict[str, float]:
    penalties = draw(penalty_strategies)
    values = {}
    for penalty in penalties:
        values[penalty] = draw(values_strategies[penalty])

    return values


def make_bisect_func(desired_nnz: int, Y: np.ndarray, verbose: bool = True, **kwargs):
    def inner_bisect(l0):
        fit_gl0learn = fit(Y, l0=l0, **kwargs)
        theta_hat = fit_gl0learn.theta
        np.fill_diagonal(theta_hat, 0)

        nnz = np.count_nonzero(theta_hat) // 2
        cost = desired_nnz - nnz
        if verbose:
            print(f"gl0Learn found solution with {nnz} non-zeros with parameters:")
            print(f"\t l0 = {l0})")
            print(f"\t cost = {cost}")
        return cost

    return inner_bisect


def overlap_covariance_matrix(p: int, seed: int = 0, max_overlaps: int = 1, decay=0.99):
    v = 1

    rng = np.random.RandomState(seed=seed)
    cov = -1 * np.eye(1)
    while min(np.linalg.eigvals(cov)) < 0:
        overlaps = {i: 0 for i in range(p)}
        cov = np.eye(p)
        while len(overlaps) >= 2:
            rows = list(overlaps.keys())

            row, col = rng.choice(rows, size=2, replace=False)

            overlaps[row] += 1
            overlaps[col] += 1

            cov[row, col] += v
            v *= decay

            overlaps = {r: o for (r, o) in overlaps.items() if o < max_overlaps}

        cov = (cov + cov.T) / 2

    return np.asfortranarray(cov)


def sample_from_cov(cov: np.ndarray, n: int = 1000, seed: int = 0) -> np.ndarray:
    p, p2 = cov.shape
    assert p == p2

    mu = np.zeros(p)
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(mu, cov=np.linalg.inv(cov), size=n)

    return np.asfortranarray(x)
