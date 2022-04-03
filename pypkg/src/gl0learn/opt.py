from sys import stdout
from time import perf_counter
from collections import namedtuple
from typing import Optional, Dict

import numpy as np


def diag_index(i):
    return (i + 1) * (i + 2) // 2 - 1


MIOResults = namedtuple(
    "MIOResults",
    [
        "theta_hat",
        "z",
        "t",
        "s",
        "lg",
        "residuals",
        "objective",
        "elapsed",
        "upper_bound",
        "lower_bound",
        "gap",
        "status",
    ],
)


def mosek_level_values(theta: np.ndarray, Y: np.ndarray, int_tol: float = 1e-4):
    """

    Parameters
    ----------
    theta : (p, p) symmetric matrix
        The lower triangular will be selected automatically
    Y: (n, p) samples of from theta inverse
    int_tol : float, default = 1e-4
        tolerance such that any value of theta with absolute value less than int_tol is deemed to be 0.

    Returns
    -------
    theta: (p, p) array
        returns `theta` as passed
    theta_tril: (p*(p+1)//2, ) array
        Lower triangular section of theta including the main diagonal
    z_values: (p*(p+1)//2, ) array
        Indicator matrix of lower triangular section of theta including the main diagonal where:
            Any non zero item of the matrix is located
                AND
            Is not located on the main diagonal.
    s_values: (p*(p+1)//2, ) array
        Derived matrix of triangular section of theta including the main diagonal where the value is:
            theta[i, j]**2 if i != j else 0!
    t_values: (p, ) array
        t_values[i] <- 1/theta[i,i]||Ytheta[:, i]||^2

    lg: (p, ) array
        natural log of the main diagonal
    residuals:
    """
    n, p = Y.shape

    assert n > p

    assert theta.shape == (p, p), "Initial Theta must be passed as a (p by p matrix)!"
    np.testing.assert_array_almost_equal(theta, theta.T)

    tril_indicies = np.tril_indices(p, k=0)  # Used to select the lower triangular values including the main diagonal

    # Since mosek keeps main diagonal in the l0 and l2 variables.
    # We create a copy and set diagonal to zero to make l0, and l2 calculations easier!
    theta_no_diag = np.copy(theta)
    np.fill_diagonal(theta_no_diag, 0)

    non_zero_values = np.abs(theta_no_diag) > int_tol

    z_values = non_zero_values[tril_indicies]
    s_values = theta_no_diag[tril_indicies] ** 2

    t_values = np.linalg.norm(Y @ theta, axis=0) ** 2 / np.diag(theta)

    lg_values = np.log(np.diag(theta))

    YtY = np.linalg.cholesky(Y.T @ Y).T
    residuals = YtY @ theta

    return theta[tril_indicies], z_values, s_values, t_values, lg_values, residuals


def MIO_mosek(
    y,
    l0,
    l2,
    m,
    mio_gap=1e-4,
    int_tol=1e-4,
    max_time=None,
    initial_values: Optional[Dict[str, np.ndarray]] = None,
):
    start_time = perf_counter()
    n, p = y.shape
    num_coeffs = p * (p + 1) // 2
    try:
        import mosek.fusion as msk
    except ModuleNotFoundError:
        raise Exception("`mosek` is not installed. Refer ot installation documentation about how to install `mosek`")

    model = msk.Model()
    model.acceptedSolutionStatus(msk.AccSolutionStatus.Feasible)

    theta_tril = model.variable("theta_tril", num_coeffs, msk.Domain.unbounded())
    s = model.variable("s", num_coeffs, msk.Domain.greaterThan(0))
    z = model.variable("z", num_coeffs, msk.Domain.integral(msk.Domain.inRange(0, 1)))
    t = model.variable("t", p, msk.Domain.greaterThan(0))
    lg = model.variable("lg", p, msk.Domain.unbounded())
    residuals = model.variable("residuals", [min(n, p), p], msk.Domain.unbounded())

    theta = theta_tril.fromTril(p)
    if n <= p:
        expr = msk.Expr.mul(msk.Matrix.dense(y), theta)
    else:
        C = np.linalg.cholesky(y.T @ y)
        expr = msk.Expr.mul(msk.Matrix.dense(C.T), theta)
    model.constraint(msk.Expr.sub(residuals, expr), msk.Domain.equalsTo(0))

    for i in range(p):
        model.constraint(
            msk.Expr.vstack(
                theta_tril.index(diag_index(i)),
                msk.Expr.mul(0.5, t.index(i)),
                residuals.slice([0, i], [min(n, p), i + 1]).reshape(min(n, p)),
            ),
            msk.Domain.inRotatedQCone(),
        )
        model.constraint(
            msk.Expr.vstack(theta_tril.index(diag_index(i)), msk.Expr.constTerm(1), lg.index(i)),
            msk.Domain.inPExpCone(),
        )

    z_expr = msk.Expr.constTerm(0)
    s_expr = msk.Expr.constTerm(0)
    for i in range(1, p):
        theta_tmp = theta_tril.slice(diag_index(i - 1) + 1, diag_index(i))
        z_tmp = z.slice(diag_index(i - 1) + 1, diag_index(i))
        s_tmp = s.slice(diag_index(i - 1) + 1, diag_index(i))
        expr = msk.Expr.mul(z_tmp, m)
        model.constraint(msk.Expr.sub(expr, theta_tmp), msk.Domain.greaterThan(0))
        model.constraint(msk.Expr.add(theta_tmp, expr), msk.Domain.greaterThan(0))
        expr = msk.Expr.hstack(msk.Expr.mul(0.5, s_tmp), z_tmp, theta_tmp)
        model.constraint(expr, msk.Domain.inRotatedQCone())
        z_expr = msk.Expr.add(z_expr, msk.Expr.sum(z_tmp))
        s_expr = msk.Expr.add(s_expr, msk.Expr.sum(s_tmp))

    z_expr = msk.Expr.mul(l0, z_expr)
    s_expr = msk.Expr.mul(l2, s_expr)
    t_expr = msk.Expr.sum(msk.Expr.sub(t, lg))

    model.objective(msk.ObjectiveSense.Minimize, msk.Expr.add([t_expr, z_expr, s_expr]))

    model.setSolverParam("log", 0)
    model.setSolverParam("mioTolAbsRelaxInt", int_tol)
    model.setSolverParam("mioTolAbsGap", mio_gap)
    model.setSolverParam("mioTolRelGap", mio_gap)
    model.setSolverParam("mioRelGapConst", 1)

    if max_time is not None:
        model.setSolverParam("mioMaxTime", max_time)
    model.setLogHandler(stdout)

    if initial_values:
        theta_tril.setLevel(initial_values["theta_tril"])
        s.setLevel(initial_values["s"])
        z.setLevel(initial_values["z"])
        t.setLevel(initial_values["t"])
        lg.setLevel(initial_values["lg"])
        residuals.setLevel(initial_values["residuals"])
        print(model.getPrimalSolutionStatus())

    model.solve()

    status = model.getProblemStatus()

    lower_bound = model.getSolverDoubleInfo("mioObjBound")
    upper_bound = model.getSolverDoubleInfo("mioObjInt")
    gap = (upper_bound - lower_bound) / max(1, abs(upper_bound))

    return MIOResults(
        theta_hat=theta.level().reshape(p, p),
        z=z.fromTril(p).level().reshape(p, p),
        t=t.level(),
        s=s.level(),
        lg=lg.level(),
        residuals=residuals.level(),
        objective=model.primalObjValue(),
        elapsed=perf_counter() - start_time,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        gap=gap,
        status=status,
    )
