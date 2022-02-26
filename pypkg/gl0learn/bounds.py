from typing import TypeVar, Union, Optional

import numpy as np
import numpy.typing as npt

from gl0learn._gl0learn import _NoBounds, _Bounds_double, _Bounds_mat
from gl0learn.utils import (
    ensure_well_behaved,
    set_post_broadcasting_flags,
    ClosedInterval,
)

FLOAT_TYPE = TypeVar("FLOAT_TYPE", bound=npt.NBitBase)


class Bounds:
    def __init__(
        self,
        lows: Optional[Union[float, npt.NDArray[FLOAT_TYPE]]] = None,
        highs: Optional[Union[float, npt.NDArray[FLOAT_TYPE]]] = None,
    ):

        if lows is None and highs is None:
            self._bounds = _NoBounds()
            return
        elif lows is not None and highs is not None:
            # TODO Handle broadcasting properly.
            lows, highs = np.broadcast_arrays(lows, highs)
            set_post_broadcasting_flags(lows)
            set_post_broadcasting_flags(highs)
        elif lows is None:
            lows = np.inf * np.ones_like(highs)
        else:
            highs = np.inf * np.ones_like(lows)

        if lows.size == 1:
            self._bounds = _Bounds_double(lows, highs)
        elif len(lows.shape) == 2:
            lows = ensure_well_behaved(lows, name="lows")
            highs = ensure_well_behaved(highs, name="highs")
            self._bounds = _Bounds_mat(lows, highs)
        else:
            raise ValueError(
                f"expected bounds to be either 2D arrays of floats or a single float, "
                f"but got lows={lows}, highs={highs}."
            )

        if not self._bounds.validate():
            raise ValueError(
                f"expected bounds to follow (lows <= 0, highs >=0 and lows < highs) and if passed as an "
                f"array, bounds must be a symmetric square 2D array, but are not."
            )

    @property
    def num_features(self) -> Union[ClosedInterval, int]:
        if isinstance(self._bounds, _NoBounds) or isinstance(self.lows, float):
            return ClosedInterval(low=1, high=np.inf)
        else:
            return self.lows.shape[0]

    @property
    def lows(self):
        return self._bounds.lows

    @property
    def highs(self):
        return self._bounds.highs

    def __repr__(self):
        return f"Bounds({self.lows}, {self.highs})"
