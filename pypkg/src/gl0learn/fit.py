from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from gl0learn.gl0learn_core import (
    _fit,
    check_is_coordinate_subset,
)

from gl0learn import Penalty, Bounds
from gl0learn.fitmodel import FitModel
from gl0learn.oracle import Oracle
from gl0learn.utils import check_make_valid_coordinate_matrix, ensure_well_behaved


def fit(
    x: npt.ArrayLike,
    theta_init: Optional[Union[npt.ArrayLike, float]] = None,
    l0: Optional[Union[npt.ArrayLike, float]] = None,
    l1: Optional[Union[npt.ArrayLike, float]] = None,
    l2: Optional[Union[npt.ArrayLike, float]] = None,
    lows: Optional[Union[npt.ArrayLike, float]] = None,
    highs: Optional[Union[npt.ArrayLike, float]] = None,
    max_iter: int = 100,
    max_swaps: int = 100,
    algorithm: str = "CD",
    max_active_set_ratio: float = 0.1,
    tol: float = 1e-6,
    active_set: Union[Union[npt.ArrayLike, float]] = 0.7,
    super_active_set: Union[Union[npt.ArrayLike, float]] = 0.5,
    scale_x: bool = True,
    check: bool = True,
    seed: int = 0,
    shuffle: bool = False,
) -> FitModel:
    """

    Parameters
    ----------
    x
    theta_init
    l0
    l1
    l2
    lows
    highs
    max_iter
    algorithm
    max_active_set_ratio
    atol
    rtol
    active_set:
        Highly corlelated components of X
        Coordinates of |YtY/S_diag| >= initial_active_set

    super_active_set
    swap_iters
    scale_x
    check

    Returns
    -------

    """

    if isinstance(x, np.ndarray):
        x = ensure_well_behaved(x, name="x")
    else:
        x = np.asarray(x, order="F")

    if not np.issubdtype(x.dtype, np.floating):
        raise ValueError(
            f"expected `x` to be an array of dtype {np.float_}, but got {x.dtype}"
        )

    try:
        n, p = x.shape
    except ValueError as e:
        raise ValueError(
            f"expected `x` to be a 2D array but got {x.ndim}D array."
        ) from e

    if n < 1 or p < 1:
        raise ValueError(
            f"expected `x` to have at least two rows and two columns, but got {n} rows and {p} columns."
        )

    oracle = Oracle(Penalty(l0, l1, l2, validate=check), Bounds(lows, highs))
    if p not in oracle.num_features:
        raise ValueError(
            "expected `x`, `l0`, `l1`, `l2`, `lows`, and `highs` to all have compatible shapes, but they are not."
        )

    # TODO: check theta_init
    if theta_init is None:
        theta_init = np.eye(p, order="F")
        theta_init_support = np.empty(shape=(0, 0), dtype="int", order="F")
    else:
        if isinstance(theta_init, np.ndarray):
            theta_init = ensure_well_behaved(theta_init, name="theta_init")
        else:
            theta_init = np.asarray(theta_init, dtype="float", order="F")

        if theta_init.shape != (p, p) or (theta_init != theta_init.T).all():
            raise ValueError(
                f"expected `theta_init` to be a square symmetric matrix of side length {p}, but is not."
            )

        theta_init_support = np.transpose(np.nonzero(theta_init * np.tri(p, k=-1).T))

    if scale_x:
        y = x / np.sqrt(n)
    else:
        y = x

    if max_active_set_ratio < 0:
        raise ValueError(
            f"expected `max_active_set_size` to be an positive number in, but got {max_active_set_ratio}"
        )

    max_active_set_size = int(max_active_set_ratio * (p * (p - 1) // 2))

    if max_iter < 1:
        raise ValueError(
            f"expected `max_iter` to be a positive integer, but got {max_iter}"
        )

    if algorithm not in ["CD", "CDPSI"]:
        raise ValueError(
            f"expected `algorithm` to be a 'CD' or 'CDPSI', but got {algorithm}"
        )

    if tol < 0:
        raise ValueError(f"expected `tol` to be a non-negative number, but got {tol}")

    active_set = check_make_valid_coordinate_matrix(
        active_set, y, "initial_active_set", check=check
    )

    # if algorithm == "CDPSI":
    #     super_active_set = check_make_valid_coordinate_matrix(super_active_set, y, 'super_active_set', check=check)
    # else:
    #     super_active_set = np.empty(shape=(0, 0), dtype='int', order='F')

    super_active_set = check_make_valid_coordinate_matrix(
        super_active_set, y, "super_active_set", check=check
    )

    if check:
        # if algorithm == "CDPSI" and not check_is_coordinate_subset(super_active_set, initial_active_set):
        if not check_is_coordinate_subset(super_active_set, active_set):
            raise ValueError(
                "executed `initial_active_set` to be a subset of `super_active_set` but is not."
            )

        if not check_is_coordinate_subset(active_set, theta_init_support):
            raise ValueError(
                "expected the support of `theta_init` to be a subset of `initial_active_set`, but is not"
            )

    if active_set.shape[0] > max_active_set_size:
        raise ValueError(
            "expected `initial_active_set` to be less than `max_active_set_size`, but isn't."
        )

    return FitModel(
        _fit(
            y,
            theta_init,
            oracle.penalty.cxx_penalty,
            oracle.bounds.cxx_bounds,
            algorithm,
            active_set,
            super_active_set,
            tol,
            max_active_set_size,
            max_iter,
            seed,
            max_swaps,
            shuffle,
        )
    )
