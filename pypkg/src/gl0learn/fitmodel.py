import numpy as np
import numpy.typing as npt


class FitModel:
    def __init__(self, _fitmodel):
        self._fitmodel = _fitmodel

    @property
    def theta(self) -> npt.NDArray[np.float]:
        return self._fitmodel.theta

    @property
    def R(self) -> npt.NDArray[np.float]:
        return self._fitmodel.R

    @property
    def costs(self) -> npt.NDArray[np.float]:
        return self._fitmodel.costs

    @property
    def active_set_size(self) -> npt.NDArray[np.int_]:
        return self._fitmodel.active_set_size

    def __repr__(self):
        return f"FitModel()"
