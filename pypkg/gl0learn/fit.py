from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from .gl0learn import _fit, check_is_coordinate_subset
from .penalty import Penalty
from .bounds import Bounds
from .fitmodel import FitModel
from .oracle import Oracle
from .utils import check_make_valid_coordinate_matrix, ensure_well_behaved


def fit(
    x: npt.ArrayLike,
    theta_init: Optional[Union[npt.ArrayLike, float]] = None,
    l0: Union[npt.ArrayLike, float] = 0,
    l1: Optional[Union[npt.ArrayLike, float]] = None,
    l2: Optional[Union[npt.ArrayLike, float]] = None,
    lows: Optional[Union[npt.ArrayLike, float]] = None,
    highs: Optional[Union[npt.ArrayLike, float]] = None,
    max_iter: int = 100,
    max_swaps: int = 100,
    algorithm: str = "CD",
    max_active_set_ratio: float = 0.1,
    tol: float = 1e-6,
    active_set: Union[npt.ArrayLike, float] = 0.7,
    super_active_set: Union[npt.ArrayLike, float] = 0.5,
    scale_x: bool = True,
    check: bool = True,
    seed: int = 0,
    shuffle: bool = False,
) -> FitModel:
    """

    Parameters
    ----------
    x: array like of shape (n, p)
         The data matrix of shape (n, p) where each row x[i, ] is believed to be drawn from N(0, theta)
    theta_init: array like of shape (p, p), optional
        The initial guess of theta. Default is the identity matrix.
        If provided, must be a symmetric matrix of shape (p, p) such that all non-zero upper triangle values of
        `theta_init` are included in `active_set`.
        Recommended that `check` be keep as `True` when providing `theta_init`
    l0: positive float or array like of shape (p, p), optional
        The L0 regularization penalty.
        If provided, must be one of:
            1. Positive scalar. Applies the same L0 regularization to each coordinate of `theta`
            2. Symmetric Matrix with only positive values of shape (p, p). Applies L0 regularization coordinate
                by coordinate to `theta`
        If not provided, L0 penalty is set to 0
    l1: positive float or array like of shape (p, p), optional
        The L1 regularization penalty.
        If provided, must be one of:
            1. Positive scalar. Applies the same L1 regularization to each coordinate of `theta`
            2. Symmetric Matrix with only positive values of shape (p, p). Applies L1 regularization coordinate
                by coordinate to `theta`
        If not provided, L1 penalty is set to 0
    l2: positive float or array like of shape (p, p), optional
        The L2 regularization penalty.
        If provided, must be one of:
            1. Positive scalar. Applies the same L2 regularization to each coordinate of `theta`
            2. Symmetric Matrix with only positive values of shape (p, p). Applies L2 regularization coordinate
                by coordinate to `theta`
        If not provided, L2 penalty is set to 0
    lows: float or array of floats with shape (P, P), optional
        If not provided, `lows` will be interpreted as boundless and the optimized values of theta will
            have no lower bounds.
        If provided as a float, the value must be non-positive. This will ensure that every value of
            theta will respect: theta[i , j] >= lows for i, j in 0 to p-1
        If provided as an `p` by `p` array of non-positive floats. This will ensure that every value of
            theta will respect: theta[i , j] >= lows[i, j] for i, j in 0 to p-1
        The same as `highs` but provides a lower bound for the optimized value of theta.
    highs: float or array of floats with shape (P, P), optional
        If not provided, `highs` will be interpreted as boundless and the optimized values of theta will
            have no upper bounds.
        If provided as a float, the value must be non-negative. This will ensure that every value of
            theta will respect: theta[i , j] <= highs for i, j in 0 to p-1
        If provided as an `p` by `p` array of non-negative floats. This will ensure that every value of
            theta will respect: theta[i , j] <= highs[i, j] for i, j in 0 to p-1
        The same as `lows` but provides an upper bound for the optimized value of theta.
    max_iter: int, optional
        The maximum number of iterations the algorithm can take before exiting.
        May exit before this number of iterations if convergence is found.
    max_swaps: int, optional
        The maximum number of swaps the "CDPSI" algorithm will perform per iteration.
        Ignored, if `algorithm` is `CD`
    algorithm: str, optional
        The type of algorithm used to minimize the objective function.
        Must be one of:
            1. "CD" A variant of cyclic coordinate descent and runs very fast.
            2. "CDPSI" performs local combinatorial search on top of CD and typically
                achieves higher quality solutions (at the expense of increased running time).
    max_active_set_ratio: float, optional
        The maximum number of non-zero values in `theta` expressed in terms of percentage of p**2
    tol: float, optional
        The tolerance for determining convergence. Graphical Models have non standard convergence criteria.
        See [TODO: Convergence Documentation] for more details.
    active_set: float or integer matrix of shape (m, 2), optional
        The set of coordinates that the local optimization algorithm quickly iterates
        as potential support values of theta.
        Can be one of:
            1. a scalar value, t, will be used to find the values of x'x that have
               an absolute value larger than t. The coordinates of these values are the
               initial active_set.
           2. Integer Matrix of shape (m, 2) encoding for the coordinates of the active_set.
              Row k (active_set[k, :]), corresponds to the coordinate in theta,
                (i.e theta[active_set[k, 1], active_set[k, 2]]) that is in the active_set.
              *NOTE* All rows of active_set must encode for valid upper triangle coordinates of theta
                    (i.e. all(x>0) and all(x<p+1)).
            *NOTE* The rows of active_set must be lexicographically sorted such that
                active_set[k] < active_set[j] -> k < j.
    super_active_set: float or integer matrix of shape (m, 2), optional
        The set of coordinates that the global optimization algorithm can swap in and out of `active_set`.
        See `active_set` parameter for valid values. When evaluated, all items in `active_set` must be contained
        in `super_active_set`. This can easily be obtained by setting `active_set` and `super_active_set` to be floats
        where `active_set` >= `super_active_set`
    scale_x: bool, optional
         A boolean flag whether x needs to be scaled by 1/sqrt(n).
         If scale_x is false (i.e the matrix is already scaled), the solver will not save a local copy of x and thus
         reduce memory usage.
    check: bool, optional
        If set, checks all values for appropriate dimensions and values.
        Only use this is speed is required and you know what you are doing.
    seed: int, optional
        The seed value used to set randomness.
        The same input values with the same seed run on the same version of `gL0learn` will always result
            in the same value
    shuffle: bool, optional
        A boolean flag whether or not to shuffle the iteration order of `active_set` when optimizing.
    Returns
    -------
    fitmodel: FitModel

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

    if n <= 1 or p <= 1:
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
