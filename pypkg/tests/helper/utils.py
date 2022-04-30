from functools import cache
from typing import List, Dict, Iterable, Tuple

import hypothesis
import numpy as np
from gl0learn import fit
from hypothesis.strategies import composite, integers


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


def top_n_triu_indicies(x, n):
    x = np.copy(x)
    x = np.triu(x, k=1)
    value = np.sort(np.abs(x).flatten())[::-1][n - 1]

    return np.where(np.abs(x) >= value)


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
) -> hypothesis.strategies.SearchStrategy[Dict[str, float]]:
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


@cache
def _sample_data(n: int = 1000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """


    Example Data!


    >>>from tabulate import tabulate
    ...import numpy as np
    ...coords = np.array([str(t).replace('(','').replace(')','') for t in zip(*np.nonzero(np.ones([5,5])))]).reshape(5,5)
    ...table = tabulate(coords, tablefmt="fancy_grid")
    ...print(table)
    ╒══════╤══════╤══════╤══════╤══════╕
    │ 0, 0 │ 0, 1 │ 0, 2 │ 0, 3 │ 0, 4 │
    ├──────┼──────┼──────┼──────┼──────┤
    │ 1, 0 │ 1, 1 │ 1, 2 │ 1, 3 │ 1, 4 │
    ├──────┼──────┼──────┼──────┼──────┤
    │ 2, 0 │ 2, 1 │ 2, 2 │ 2, 3 │ 2, 4 │
    ├──────┼──────┼──────┼──────┼──────┤
    │ 3, 0 │ 3, 1 │ 3, 2 │ 3, 3 │ 3, 4 │
    ├──────┼──────┼──────┼──────┼──────┤
    │ 4, 0 │ 4, 1 │ 4, 2 │ 4, 3 │ 4, 4 │
    ╘══════╧══════╧══════╧══════╧══════╛

    Suppose:
        Coordinates (0,1) and (1,2) are the initial support
        Coordinates (0,2) and (1,3) are also in the active set
        Coordinates (0,3) and (1,4) are also in the super active set

    Supplying `theta_truth` as a upper triangular diagonally dominate matrix, we can set which of `theta_hat` should be learned first.

    This allows us to check if fit is behaving as expected!
    """
    N = 5
    mu = np.zeros(N)

    theta_truth_tril = (1 / 8) * np.asarray(
        [
            [8, 0, 0, 0, 1],
            [0, 8, 4, 2, 3],
            [0, 0, 8, 6, 5],
            [0, 0, 0, 8, 7],
            [0, 0, 0, 0, 8],
        ]
    )

    theta_truth = (theta_truth_tril + theta_truth_tril.T) / 2

    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(mu, cov=np.linalg.inv(theta_truth), size=n)

    return theta_truth, x


def overlap_covariance_tril_matrix(
    n: int,
    max_overlaps: int = 1,
    seed: int = 0,
    max_iters: int = 1000,
    decay: float = 1.0,
):
    rng = np.random.RandomState(seed=seed)

    row_overlaps = {i: 0 for i in range(n - 1)}
    col_overlaps = {i: 0 for i in range(1, n)}

    cov = np.eye(n)

    v = 1

    for _ in range(max_iters):
        rows = list(row_overlaps.keys())

        row_openings = {}
        for row in rows:
            row_openings[row] = sum(1 for k in col_overlaps if k > row)

        num_openings = sum(row_openings.values())

        if not num_openings:
            break

        row_probability = [r / num_openings for r in row_openings.values()]
        row = rng.choice(rows, p=row_probability)
        try:
            col = rng.choice(list(c for c in col_overlaps.keys() if c > row))
        except ValueError:
            continue
        cov[row, col] += v
        v *= decay

        row_overlaps[row] += 1
        col_overlaps[col] += 1

        row_overlaps = {r: o for (r, o) in row_overlaps.items() if o < max_overlaps}
        col_overlaps = {c: o for (c, o) in col_overlaps.items() if o < max_overlaps}

    return cov


def overlap_covariance_matrix(p: int, seed: int = 0, max_overlaps: int = 1, decay=0.99):

    overlaps = {i: 0 for i in range(p)}
    cov = np.eye(p)

    v = 1

    rng = np.random.RandomState(seed=seed)
    while len(overlaps) >= 2:
        rows = list(overlaps.keys())

        row, col = rng.choice(rows, size=2, replace=False)

        overlaps[row] += 1
        overlaps[col] += 1

        cov[row, col] += v
        v *= decay

        overlaps = {r: o for (r, o) in overlaps.items() if o < max_overlaps}

    cov = (cov + cov.T) / 2

    return cov


def sample_from_cov(cov: np.ndarray, n: int = 1000, seed: int = 0) -> np.ndarray:
    p, p2 = cov.shape
    assert p == p2

    mu = np.zeros(p)
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(mu, cov=np.linalg.inv(cov), size=n)

    return x


@cache
def _sample_data2(n: int = 1000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    p = 5
    mu = np.zeros(p)
    theta_truth_tril = overlap_covariance_matrix(p, 1, decay=0.8)

    theta_truth = (theta_truth_tril + theta_truth_tril.T) / 2

    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(mu, cov=np.linalg.inv(theta_truth), size=n)

    return theta_truth, x
