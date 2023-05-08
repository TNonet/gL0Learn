import numpy as np
import numpy.typing as npt


class FitModel:
    def __init__(self, _fitmodel):
        self._fitmodel = _fitmodel

    @property
    def theta(self) -> npt.NDArray[float]:
        return self._fitmodel.theta

    @property
    def R(self) -> npt.NDArray[float]:
        return self._fitmodel.R

    @property
    def costs(self) -> npt.NDArray[float]:
        return self._fitmodel.costs

    @property
    def active_set_size(self) -> npt.NDArray[np.int_]:
        return self._fitmodel.active_set_size

    def __eq__(self, other):
        if isinstance(other, FitModel):
            print(
                [
                    np.array_equal(self.theta, other.theta, equal_nan=True),
                    np.array_equal(self.R, other.R, equal_nan=True),
                    np.array_equal(self.costs, other.costs, equal_nan=True),
                    np.array_equal(
                        self.active_set_size, other.active_set_size, equal_nan=True
                    ),
                ]
            )
            return all(
                [
                    np.array_equal(self.theta, other.theta, equal_nan=True),
                    np.array_equal(self.R, other.R, equal_nan=True),
                    np.array_equal(self.costs, other.costs, equal_nan=True),
                    np.array_equal(
                        self.active_set_size, other.active_set_size, equal_nan=True
                    ),
                ]
            )
        else:
            return False

    def __repr__(self):
        return "FitModel()"
