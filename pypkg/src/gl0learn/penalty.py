from typing import TypeVar, Union, Optional, Callable, Dict, Tuple

import numpy as np
import numpy.typing as npt
from gl0learn.gl0learn_core import (
    _PenaltyL0_double,
    _PenaltyL0_mat,
    _PenaltyL1_double,
    _PenaltyL1_mat,
    _PenaltyL2_double,
    _PenaltyL2_mat,
    _PenaltyL0L1_double,
    _PenaltyL0L1_mat,
    _PenaltyL0L2_double,
    _PenaltyL0L2_mat,
    _PenaltyL0L1L2_double,
    _PenaltyL0L1L2_mat,
    check_coordinate_matrix,
    objective,
)

from .utils import ensure_well_behaved, set_post_broadcasting_flags, ClosedInterval

FLOAT_TYPE = TypeVar("FLOAT_TYPE", bound=npt.NBitBase)
P = TypeVar("P")


class Penalty:
    PENALTY_DICT: Dict[str, Callable[[Tuple[Union[float, npt.NDArray[FLOAT_TYPE]], ...]], P]] = {
        "_PenaltyL0_double": _PenaltyL0_double,
        "_PenaltyL0_mat": _PenaltyL0_mat,
        "_PenaltyL1_double": _PenaltyL1_double,
        "_PenaltyL1_mat": _PenaltyL1_mat,
        "_PenaltyL2_double": _PenaltyL2_double,
        "_PenaltyL2_mat": _PenaltyL2_mat,
        "_PenaltyL0L1_double": _PenaltyL0L1_double,
        "_PenaltyL0L1_mat": _PenaltyL0L1_mat,
        "_PenaltyL0L2_double": _PenaltyL0L2_double,
        "_PenaltyL0L2_mat": _PenaltyL0L2_mat,
        "_PenaltyL0L1L2_double": _PenaltyL0L1L2_double,
        "_PenaltyL0L1L2_mat": _PenaltyL0L1L2_mat,
    }

    def __init__(
        self,
        l0: Optional[Union[float, npt.NDArray[FLOAT_TYPE]]] = None,
        l1: Optional[Union[float, npt.NDArray[FLOAT_TYPE]]] = None,
        l2: Optional[Union[float, npt.NDArray[FLOAT_TYPE]]] = None,
        validate: bool = True,
    ):

        if l0 is None and l1 is None and l2 is None:
            raise ValueError("expected at least one of l0, l1, and l2 to be non-None, but got all None.")

        penalty_values = []
        penalty_names = []
        if l0 is not None:
            penalty_names.append("l0")
            penalty_values.append(l0)
        if l1 is not None:
            penalty_names.append("l1")
            penalty_values.append(l1)
        if l2 is not None:
            penalty_names.append("l2")
            penalty_values.append(l2)

        penalty_name = "".join(penalty_names)
        penalty_values = np.broadcast_arrays(*penalty_values)
        for penalty_value in penalty_values:
            set_post_broadcasting_flags(penalty_value)

        if penalty_values[0].size == 1:
            penalty_dtype = "double"
        elif len(penalty_values[0].shape) == 2:
            penalty_values = [ensure_well_behaved(p, name=name) for (p, name) in zip(penalty_values, penalty_names)]
            penalty_dtype = "mat"
        else:
            raise ValueError(
                "expected all penalty values to be either 2D arrays of floats or single floats, but are not."
            )

        penalty_lx = f"_Penalty{penalty_name.upper()}_"
        try:
            penalty = self.PENALTY_DICT[penalty_lx + penalty_dtype]
        except KeyError:
            raise ValueError("Currently, only L0, L1, L2, L0L1, L0L2 and L0L1L2 are supported.")

        self.cxx_penalty = penalty(*penalty_values)

        if validate and not self.cxx_penalty.validate():
            raise ValueError(
                "expected all penalty values to be non-negative and if passed as floats, "
                "to by a symmetric square 2D array, but are not."
            )
        self.penalty_names = penalty_names
        for name in self.penalty_names:
            setattr(self, name, getattr(self.cxx_penalty, name))

    @property
    def num_features(self) -> Union[ClosedInterval, int]:
        lx = getattr(self.cxx_penalty, self.penalty_names[0])
        if isinstance(lx, float):
            return ClosedInterval(low=1, high=np.inf)
        else:
            return lx.shape[0]

    def __getitem__(self, item) -> "Penalty":
        return Penalty(
            **{lX: getattr(self.cxx_penalty, lX)[item] for lX in self.penalty_names},
            validate=False,
        )

    def objective(
        self,
        theta: np.ndarray,
        residuals: np.ndarray,
        active_set: Optional[np.ndarray] = None,
    ) -> float:
        if active_set is None:
            p = theta.shape[0]
            active_set = np.asarray(np.triu_indices(p, k=1), order="F").T

        if not check_coordinate_matrix(active_set, for_order=True, for_upper_triangle=True):
            raise ValueError("expected `active_set` to be a lexicographically sorted coordinate matrix, but isn't.")

        return objective(theta, residuals, active_set, self.cxx_penalty)

    def cost(self, theta: Union[np.ndarray, float], *getitems: int) -> Union[np.ndarray, float]:
        """Calculate the regularization/penalty cost of `theta`

        Parameters
        ----------
        theta : numpy array of shape (P, P) or numpy array of shape (P, ) or double.
            See `getitems` parameter for more information on shape of theta!
            coefficient matrix/row/value that the regularization/penalty cost is calculated with.

        getitems : Variable length list of indices to slice/get items of penalty values.

            For example, suppose `Penalty.LX` is a matrix of values, but you would like to calculate the regularization
            cost for a specific row or value. This information is passed in as the variable length list `getitems`. If
            `getitems` is of length one, the `getitems[0]`th row of `Penalty.LX` will be used. If `theta` is passed in
            as a (P, P) array, the same row of `theta` will be sliced. If `theta` is a row of length P, then no slicing
            will be preformed. If `theta` is a single value, then it will broadcast to match the row of `Penalty.LX`

            If `Penalty.LX` is a double, `getitems` will be mainly ignored except to slice/getitem of `theta`.

        Returns
        -------
        penalty_cost: numpy array of shape (P, P) or numpy array of shape (P, ) or double
        """
        return self.cxx_penalty.cost(theta, *getitems)

    def __repr__(self):
        repr_str = ", ".join([f"{name}={getattr(self, name)}" for name in self.penalty_names])
        return f"Penalty({repr_str})"
