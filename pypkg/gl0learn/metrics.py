import numpy as np
import numpy.typing as npt


def nonzeros(x: npt.ArrayLike, abs_tol: float = 1e-6) -> npt.NDArray[np.int_]:
    return np.abs(x) >= abs_tol


def zeros(x: npt.ArrayLike, abs_tol: float = 1e-6) -> npt.NDArray[np.int_]:
    return np.abs(x) < abs_tol


def indicator_matrix_to_coords(
    x: npt.ArrayLike, abs_tol: float = 1e-6, only_upper: bool = True
) -> npt.NDArray[np.int_]:
    x = np.asarray(x)

    if x.ndim != 2:
        raise ValueError(f"expected `x` to be a 2D array, but got {x.ndim}D.")

    if only_upper:
        p = x.shape[1]
        x = x * np.tri(p, k=-1).T

    return np.transpose(nonzeros(x, abs_tol=abs_tol))


def false_positives(
    x_pred: npt.ArrayLike, x_truth: npt.ArrayLike, abs_tol: float = 1e-6
) -> int:
    return np.sum(
        np.logical_and(
            nonzeros(x_pred, abs_tol=abs_tol),  # type: ignore # noqa
            zeros(x_truth, abs_tol=abs_tol),
        )
    )


def prediction_error(r_pred: np.ndarray, r_truth: np.ndarray) -> float:
    return np.linalg.norm(r_pred - r_truth) / np.linalg.norm(r_truth)
