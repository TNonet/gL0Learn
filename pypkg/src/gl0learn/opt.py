import sys

import numpy as np
import time


def diag_index(i):
    return (i + 1) * (i + 2) // 2 - 1


def MIO_mosek(Y, l0, l2, M, mio_gap=1e-4, int_tol=1e-4, maxtime=None):
    st = time.time()
    n, p = Y.shape
    P = p * (p + 1) // 2
    try:
        import mosek.fusion as msk
        import mosek
    except ModuleNotFoundError:
        raise Exception(
            f"`mosek` is not installed. Refer ot installation documentation about how to install `mosek`"
        )

    model = msk.Model()
    model.acceptedSolutionStatus(msk.AccSolutionStatus.Feasible)

    theta_tril = model.variable("theta_tril", P, msk.Domain.unbounded())
    s = model.variable("s", P, msk.Domain.greaterThan(0))
    z = model.variable("z", P, msk.Domain.integral(msk.Domain.inRange(0, 1)))
    t = model.variable("t", p, msk.Domain.greaterThan(0))
    lg = model.variable("lg", p, msk.Domain.unbounded())
    residuals = model.variable("residuals", [min(n, p), p], msk.Domain.unbounded())

    theta = theta_tril.fromTril(p)
    if n <= p:
        expr = msk.Expr.mul(msk.Matrix.dense(Y), theta)
    else:
        C = np.linalg.cholesky(Y.T @ Y)
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
            msk.Expr.vstack(
                theta_tril.index(diag_index(i)), msk.Expr.constTerm(1), lg.index(i)
            ),
            msk.Domain.inPExpCone(),
        )

    z_expr = msk.Expr.constTerm(0)
    s_expr = msk.Expr.constTerm(0)
    for i in range(1, p):
        theta_tmp = theta_tril.slice(diag_index(i - 1) + 1, diag_index(i))
        z_tmp = z.slice(diag_index(i - 1) + 1, diag_index(i))
        s_tmp = s.slice(diag_index(i - 1) + 1, diag_index(i))
        expr = msk.Expr.mul(z_tmp, M)
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

    if maxtime is not None:
        model.setSolverParam("mioMaxTime", maxtime)
    model.setLogHandler(sys.stdout)
    model.solve()

    status = model.getProblemStatus()

    lower_bound = model.getSolverDoubleInfo("mioObjBound")
    upper_bound = model.getSolverDoubleInfo("mioObjInt")
    gap = (upper_bound - lower_bound) / max(1, abs(upper_bound))

    return (
        theta.level().reshape(p, p),
        z.fromTril(p).level().reshape(p, p),
        model.primalObjValue(),
        time.time() - st,
        upper_bound,
        lower_bound,
        gap,
    )
