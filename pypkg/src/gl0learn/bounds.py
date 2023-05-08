from typing import TypeVar, Union, Optional

import numpy as np
import numpy.typing as npt
from ._gl0learn import _NoBounds, _Bounds_double, _Bounds_mat
from .utils import ensure_well_behaved, ClosedInterval

FLOAT_TYPE = TypeVar("FLOAT_TYPE", bound=npt.NBitBase)


class Bounds:
    """
    Python wrapper around C++ Bounds object used to configure `gl0learn.fit` call.

    Bounds are used to apply limits on the acceptable value of the learned covariance matrix
    """
    def __init__(
        self,
        lows: Union[float, npt.NDArray[FLOAT_TYPE]] = -float('inf'),
        highs: Union[float, npt.NDArray[FLOAT_TYPE]] = +float('inf'),
        validate: bool = True,
    ):
        """
        Create a Bounds object for `gl0learn.fit`.

        If Bounds are supplied, the learned parameters for theta will respect the bounds provided.

        Parameters
        ----------
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
        validate: bool, optional
            Whether to validate the provided bounds.
            Validation does require a full scan of the bounds.
            If performance is important and the bounds source is trusted, this can be set to False.
        """

        if lows == -float("inf") and highs == float("inf"):
            self.cxx_bounds = _NoBounds()
            return
        elif lows != -float("inf") and highs != float("inf"):
            lows = np.asarray(lows)
            highs = np.asarray(highs)
            bounds_shape = np.broadcast_shapes(lows.shape, highs.shape)

            if lows.shape != bounds_shape:
                # TODO: Should be able to broadcast to an `order`. Raise numpy PR
                lows = np.array(
                    np.broadcast_to(lows, bounds_shape), order="F", dtype="float"
                )
            else:
                lows = ensure_well_behaved(lows, name="lows")

            if highs.shape != bounds_shape:
                # TODO: Should be able to broadcast to an `order`. Raise numpy PR
                highs = np.array(
                    np.broadcast_to(highs, bounds_shape), order="F", dtype="float"
                )
            else:
                highs = ensure_well_behaved(highs, name="highs")

        elif lows == -float("inf"):
            lows = np.inf * np.ones_like(highs, order="F", dtype="float")
        else:
            highs = np.inf * np.ones_like(lows, order="F", dtype="float")

        if lows.size == 1:
            self.cxx_bounds = _Bounds_double(lows, highs)
        elif len(lows.shape) == 2:
            self.cxx_bounds = _Bounds_mat(lows, highs)
        else:
            raise ValueError(
                f"expected bounds to be either 2D arrays of floats or a single float, "
                f"but got lows={lows}, highs={highs}."
            )

        if validate and not self.cxx_bounds.validate():
            raise ValueError(
                "expected bounds to follow (lows <= 0, highs >=0 and lows < highs) and if passed as an "
                "array, bounds must be a symmetric square 2D array, but are not."
            )

    @property
    def num_features(self) -> Union[ClosedInterval, int]:
        if isinstance(self.cxx_bounds, _NoBounds) or isinstance(self.lows, float):
            return ClosedInterval(low=1, high=np.inf)
        else:
            return self.lows.shape[0]

    @property
    def lows(self):
        return self.cxx_bounds.lows

    @property
    def highs(self):
        return self.cxx_bounds.highs

    def __repr__(self):
        return f"Bounds({self.lows}, {self.highs})"
