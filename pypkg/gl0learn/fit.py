from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from _gl0learn import (
    _fit,
    union_of_correlated_features2,
    upper_triangular_coords,
    check_coordinate_matrix,
    check_is_coordinate_subset,
)

from gl0learn import Penalty, Bounds
from gl0learn.fitmodel import FitModel
from gl0learn.oracle import Oracle
from gl0learn.utils import check_make_valid_coordinate_matrix


def fit(
    x: npt.ArrayLike,
    theta_init: Optional[Union[npt.ArrayLike, float]] = None,
    l0: Optional[Union[npt.ArrayLike, float]] = None,
    l1: Optional[Union[npt.ArrayLike, float]] = None,
    l2: Optional[Union[npt.ArrayLike, float]] = None,
    lows: Optional[Union[npt.ArrayLike, float]] = None,
    highs: Optional[Union[npt.ArrayLike, float]] = None,
    max_iter: int = 100,
    algorithm: str = "CD",
    max_active_set_size: float = 0.1,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    initial_active_set: Union[Union[npt.ArrayLike, float]] = 0.7,
    super_active_set: Union[Union[npt.ArrayLike, float]] = 0.5,
    swap_iters=None,
    scale_x: bool = True,
    check: bool = True,
) -> FitModel:

    n, p = x.shape

    if p < 1:
        raise ValueError("expected `x` to have at least 2 columns.")

    oracle = Oracle(Penalty(l0, l1, l2), Bounds(lows, highs))
    if p not in oracle.num_features:
        raise ValueError(
            f"expected `x`, `l0`, `l1`, `l2`, `lows`, and `highs` to all have compatible shapes but "
            f"they are not."
        )

    # TODO: check theta_init
    if theta_init is None:
        theta_init = np.eye(p)
        theta_init_support = np.empty(shape=(0, 0), dtype="int", order="F")
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

    if not isinstance(max_active_set_size, int) or max_active_set_size < 1:
        raise ValueError(f"expected `max_active_set_size` to be an positive integer.")

    if max_iter < 1:
        raise ValueError(
            f"expected `max_iter` to be a positive integer, but got {max_iter}"
        )

    if algorithm not in ["CD", "CDPSI"]:
        raise ValueError(
            f"expected `algorithm` to be a 'CD' or 'CDPSI', but got {algorithm}"
        )

    if atol < 0:
        raise ValueError(f"expected `atol` to be a non-negative number, but got {atol}")

    if rtol < 0 or rtol >= 1:
        raise ValueError(
            f"expected `rtol` to be a number between 0 and 1 (exclusive), but got {rtol}."
        )

    initial_active_set = check_make_valid_coordinate_matrix(
        initial_active_set, y, "initial_active_set", check=check
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
        if not check_is_coordinate_subset(super_active_set, initial_active_set):
            raise ValueError(
                f"executed `initial_active_set` to be a subset of `super_active_set` but is not."
            )

        if not check_is_coordinate_subset(initial_active_set, theta_init_support):
            raise ValueError(
                f"expected the support of `theta_init` to be a subset of `initial_active_set`, but is not"
            )

    if initial_active_set.shape[0] > max_active_set_size:
        raise ValueError(
            f"expected `initial_active_set` to be less than `max_active_set_size`, but isn't."
        )

    _fitmodel = _fit(
        y,
        theta_init,
        oracle.penalty._penalty,
        oracle.bounds._bounds,
        algorithm,
        initial_active_set,
        super_active_set,
        atol,
        rtol,
        max_active_set_size,
        max_iter,
    )

    return FitModel(_fitmodel)
