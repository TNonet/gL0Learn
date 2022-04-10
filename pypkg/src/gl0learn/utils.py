import warnings
from dataclasses import dataclass
from numbers import Real
from typing import Optional, Union

import numpy as np
from numpy import typing as npt

from gl0learn.gl0learn_core import (
    union_of_correlated_features2,
    upper_triangular_coords,
    check_coordinate_matrix,
)


class IntersectionError(ValueError):
    pass


@dataclass(frozen=True)
class ClosedInterval:
    low: Real
    high: Real

    def __contains__(self, item: Union["ClosedInterval", Real]) -> bool:
        if isinstance(item, Real):
            return self.low <= item <= self.high
        if isinstance(item, ClosedInterval):
            return self.low <= item.low and item.high <= self.high
        raise NotImplementedError()

    def intersect(
        self, other: Union["ClosedInterval", Real]
    ) -> Union["ClosedInterval", Real]:
        if isinstance(other, Real):
            if other in self:
                return other
            raise IntersectionError(f"No intersection between {other} and {self}")
        if isinstance(other, ClosedInterval):
            if other.low > self.high or self.low > other.high:
                raise IntersectionError(f"No intersection between {other} and {self}")
            lowest = max(other.low, self.low)
            highest = min(other.high, self.high)
            if lowest == highest:
                return lowest
            else:
                return ClosedInterval(lowest, highest)


def intersect(
    x1: Union["ClosedInterval", Real], x2: Union["ClosedInterval", Real]
) -> Union["ClosedInterval", Real]:
    if isinstance(x1, ClosedInterval):
        return x1.intersect(x2)
    if isinstance(x2, ClosedInterval):
        return x2.intersect(x1)
    if x1 == x2:
        return x1
    raise IntersectionError(f"{x1} and {x2} do not intersect.")


def overlaps(
    x1: Union["ClosedInterval", Real], x2: Union["ClosedInterval", Real]
) -> bool:
    try:
        intersect(x1, x2)
    except IntersectionError:
        return False
    else:
        return True


def check_make_valid_coordinate_matrix(
    x: Union[float, npt.NDArray[np.int_]],
    y: npt.NDArray[np.float64],
    scope_x_name: str,
    check: bool = True,
) -> npt.NDArray[np.int_]:
    """

    Parameters
    ----------
    x : float, str, or matrix
        if `x` is a float, then a method to calculate the highly correlated features of yTy is called using `x` as
            the threshold parameter.
        if `x` == "full", then a coordinate matrix that covers the full set of upper triangle coordinates is returned
        if `x` is an N by 2 integer numpy matrix, then a check is run on the validity of
    y : (N, 2) numpy array for which a coordinate matrix is being calculated for
    scope_x_name: Provided variable name of `x` to make error message more descriptive
    check: boolean flag, default True
        if False, will not check `x` for being a valid sorted coordinate matrix.

    Returns
    -------
    coords: (N, 2) integer numpy array that is sorted lexicographically and only consists of items in the upper
        triangle of y.

    """
    p = y.shape[1]

    if isinstance(x, float):
        if x <= 0:
            return upper_triangular_coords(p)
        else:
            return union_of_correlated_features2(y, x)

    x = np.asarray(x, order="F")  # Not setting dtype on purpose. Let `asarray` decide.

    if (
        x.ndim != 2
        or x.shape[1] != 2
        or not np.issubdtype(x.dtype, np.integer)
        or (check and not check_coordinate_matrix(x, True, True))
    ):
        raise ValueError(
            f"expected `{scope_x_name}`, when passed as a np.ndarray,"
            f"to be a N by 2 integer matrix that only refers to the upper triangle of a {p} by {p} "
            f"matrix and is lexicographically sorted, but got {x}"
        )

    return x


def triu_nnz_indicies(x, tol: float = 0):
    return np.asarray(np.where(np.abs(np.triu(x, k=1)) > tol)).T


def set_post_broadcasting_flags(arr: npt.NDArray):
    with warnings.catch_warnings():
        arr.flags["WRITEABLE"] = True


def ensure_well_behaved(
    arr: npt.NDArray, dtype=np.float_, name: Optional[str] = None
) -> npt.NDArray:
    if arr.flags["BEHAVED"] and arr.flags["F_CONTIGUOUS"]:
        if arr.dtype != dtype:
            arr = arr.astype(dtype)
        return arr
    else:
        if name is not None:
            name = f", provided scope name = '{name}',"
        else:
            name = ""
        warnings.warn(
            f"gl0learn requires Fortran-style array. Array {name} detected that is C-style. "
            f"Generating copy that is Fortran-style. To decrease memory usage, ensure array is provided as "
            f"Fortran-style and no copy will be made. See: "
            f"https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html."
        )
        return np.asarray(arr, order="F", dtype=dtype)
