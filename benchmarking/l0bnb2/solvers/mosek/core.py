import sys

import numpy as np
import time


from ..utils import diag_index, get_active_set_mask

    
def mosek_solve(Y, l0, l2, M, zlb, zub, input_tril = False, output_tril=False, relax=True, maxtime = None):
    n,p = Y.shape
    P = p*(p+1)//2
    try:
        import mosek.fusion as msk
        import mosek
    except ModuleNotFoundError:
        raise Exception('Mosek is not installed')
    
    if input_tril:
        zlb_tril = zlb
        zub_tril = zub
    else:
        zlb_tril = zlb[np.tril_indices(p,0)]
        zub_tril = zub[np.tril_indices(p,0)]

    
    model = msk.Model()
    model.acceptedSolutionStatus(msk.AccSolutionStatus.Feasible)
    theta_tril = model.variable('theta_tril', P, msk.Domain.unbounded())

    s = model.variable('s', P, msk.Domain.greaterThan(0))
    if relax:
        z = model.variable('z', P, msk.Domain.inRange(zlb_tril, zub_tril))
    else:
        z = model.variable('z', P, msk.Domain.integral(msk.Domain.inRange(zlb_tril, zub_tril)))
    t = model.variable('t', p, msk.Domain.greaterThan(0))
    lg = model.variable('lg', p, msk.Domain.unbounded())
    R = model.variable('R', [min(n,p),p], msk.Domain.unbounded())
    
    Theta = theta_tril.fromTril(p)
    if n <= p:
        expr = msk.Expr.mul(msk.Matrix.dense(Y), Theta)
    else:
        C = np.linalg.cholesky(Y.T@Y)
        expr = msk.Expr.mul(msk.Matrix.dense(C.T), Theta)
    model.constraint(msk.Expr.sub(R, expr), msk.Domain.equalsTo(0))
    
    for i in range(p):
        model.constraint(msk.Expr.vstack(theta_tril.index(diag_index(i)), msk.Expr.mul(0.5, t.index(i)), R.slice([0,i],[min(n,p),i+1]).reshape(min(n,p))), msk.Domain.inRotatedQCone())
        model.constraint(msk.Expr.vstack(theta_tril.index(diag_index(i)), msk.Expr.constTerm(1), lg.index(i)), msk.Domain.inPExpCone())
    
    z_expr = msk.Expr.constTerm(0)
    s_expr = msk.Expr.constTerm(0)
    for i in range(1,p):
        theta_tmp = theta_tril.slice(diag_index(i-1)+1,diag_index(i))
        z_tmp = z.slice(diag_index(i-1)+1,diag_index(i))
        s_tmp = s.slice(diag_index(i-1)+1,diag_index(i))
        expr = msk.Expr.mul(z_tmp, M)
        model.constraint(msk.Expr.sub(expr, theta_tmp), msk.Domain.greaterThan(0))
        model.constraint(msk.Expr.add(theta_tmp, expr), msk.Domain.greaterThan(0))
        expr = msk.Expr.hstack(msk.Expr.mul(0.5, s_tmp), z_tmp, theta_tmp)
        model.constraint(expr, msk.Domain.inRotatedQCone())
        z_expr = msk.Expr.add(z_expr, msk.Expr.sum(z_tmp))
        s_expr = msk.Expr.add(s_expr, msk.Expr.sum(s_tmp))

    z_expr = msk.Expr.mul(l0, z_expr)
    s_expr = msk.Expr.mul(l2, s_expr)
    t_expr = msk.Expr.sum(msk.Expr.sub(t,lg))
    
    
    model.objective(msk.ObjectiveSense.Minimize,msk.Expr.add([t_expr, z_expr, s_expr]))
    
    
    model.setSolverParam("log", 0)
    if maxtime is not None:
        if relax:
            model.setSolverParam('optimizerMaxTime', maxtime)
        else:
            model.setSolverParam('mioMaxTime', maxtime)
    model.setLogHandler(sys.stdout)
    model.solve()
    
    status = model.getProblemStatus()
    if status == msk.ProblemStatus.Unknown:
        symname, desc = mosek.Env.getcodedesc(mosek.rescode(int(model.getSolverIntInfo("optimizeResponse"))))
        raise Exception("   Termination code: {0} {1}".format(symname, desc))
    if output_tril:
        if relax:
            return theta_tril.level(), z.level(), model.primalObjValue(), model.dualObjValue()
        else:
            return theta_tril.level(), z.level(), model.primalObjValue()
    else:
        if relax:
            return Theta.level().reshape(p,p), z.fromTril(p).level().reshape(p,p), model.primalObjValue(), model.dualObjValue()
        else:
            return Theta.level().reshape(p,p), z.fromTril(p).level().reshape(p,p), model.primalObjValue()
        
def MIO_mosek(Y, l0, l2, M, mio_gap=1e-4, int_tol=1e-4, maxtime = None):
    st = time.time()
    n,p = Y.shape
    P = p*(p+1)//2
    try:
        import mosek.fusion as msk
        import mosek
    except ModuleNotFoundError:
        raise Exception('Mosek is not installed')
    

    model = msk.Model()
    model.acceptedSolutionStatus(msk.AccSolutionStatus.Feasible)
    theta_tril = model.variable('theta_tril', P, msk.Domain.unbounded())

    s = model.variable('s', P, msk.Domain.greaterThan(0))
    z = model.variable('z', P, msk.Domain.integral(msk.Domain.inRange(0, 1)))
    t = model.variable('t', p, msk.Domain.greaterThan(0))
    lg = model.variable('lg', p, msk.Domain.unbounded())
    R = model.variable('R', [min(n,p),p], msk.Domain.unbounded())
    
    Theta = theta_tril.fromTril(p)
    if n <= p:
        expr = msk.Expr.mul(msk.Matrix.dense(Y), Theta)
    else:
        C = np.linalg.cholesky(Y.T@Y)
        expr = msk.Expr.mul(msk.Matrix.dense(C.T), Theta)
    model.constraint(msk.Expr.sub(R, expr), msk.Domain.equalsTo(0))
    
    for i in range(p):
        model.constraint(msk.Expr.vstack(theta_tril.index(diag_index(i)), msk.Expr.mul(0.5, t.index(i)), R.slice([0,i],[min(n,p),i+1]).reshape(min(n,p))), msk.Domain.inRotatedQCone())
        model.constraint(msk.Expr.vstack(theta_tril.index(diag_index(i)), msk.Expr.constTerm(1), lg.index(i)), msk.Domain.inPExpCone())
    
    z_expr = msk.Expr.constTerm(0)
    s_expr = msk.Expr.constTerm(0)
    for i in range(1,p):
        theta_tmp = theta_tril.slice(diag_index(i-1)+1,diag_index(i))
        z_tmp = z.slice(diag_index(i-1)+1,diag_index(i))
        s_tmp = s.slice(diag_index(i-1)+1,diag_index(i))
        expr = msk.Expr.mul(z_tmp, M)
        model.constraint(msk.Expr.sub(expr, theta_tmp), msk.Domain.greaterThan(0))
        model.constraint(msk.Expr.add(theta_tmp, expr), msk.Domain.greaterThan(0))
        expr = msk.Expr.hstack(msk.Expr.mul(0.5, s_tmp), z_tmp, theta_tmp)
        model.constraint(expr, msk.Domain.inRotatedQCone())
        z_expr = msk.Expr.add(z_expr, msk.Expr.sum(z_tmp))
        s_expr = msk.Expr.add(s_expr, msk.Expr.sum(s_tmp))

    z_expr = msk.Expr.mul(l0, z_expr)
    s_expr = msk.Expr.mul(l2, s_expr)
    t_expr = msk.Expr.sum(msk.Expr.sub(t,lg))
    
    
    model.objective(msk.ObjectiveSense.Minimize,msk.Expr.add([t_expr, z_expr, s_expr]))
    
    
    model.setSolverParam("log", 0)
    model.setSolverParam("mioTolAbsRelaxInt", int_tol)
    model.setSolverParam("mioTolAbsGap", mio_gap)
    model.setSolverParam("mioTolRelGap", mio_gap)
    model.setSolverParam("mioRelGapConst",1)
    
    if maxtime is not None:
        model.setSolverParam('mioMaxTime', maxtime)
    model.setLogHandler(sys.stdout)
    model.solve()
    
    status = model.getProblemStatus()

    lower_bound = model.getSolverDoubleInfo("mioObjBound")
    upper_bound = model.getSolverDoubleInfo("mioObjInt")
    gap = (upper_bound-lower_bound)/max(1,abs(upper_bound))

    return Theta.level().reshape(p,p), z.fromTril(p).level().reshape(p,p), model.primalObjValue(), time.time()-st, upper_bound, lower_bound, gap

def relax_mosek(Y, l0, l2, M, zlb, zub, input_tril = False, output_tril=False):
    n,p = Y.shape
    P = p*(p+1)//2
    try:
        import mosek.fusion as msk
        import mosek
    except ModuleNotFoundError:
        raise Exception('Mosek is not installed')
    
    if input_tril:
        zlb_tril = zlb
        zub_tril = zub
    else:
        zlb_tril = zlb[np.tril_indices(p,0)]
        zub_tril = zub[np.tril_indices(p,0)]

    
    model = msk.Model()
    
    theta_tril = model.variable('theta_tril', P, msk.Domain.unbounded())

    s = model.variable('s', P, msk.Domain.greaterThan(0))
    z = model.variable('z', P, msk.Domain.inRange(zlb_tril, zub_tril))
    t = model.variable('t', p, msk.Domain.greaterThan(0))
    lg = model.variable('lg', p, msk.Domain.unbounded())
    R = model.variable('R', [min(n,p),p], msk.Domain.unbounded())
    
    Theta = theta_tril.fromTril(p)
    if n <= p:
        expr = msk.Expr.mul(msk.Matrix.dense(Y), Theta)
    else:
        C = np.linalg.cholesky(Y.T@Y)
        expr = msk.Expr.mul(msk.Matrix.dense(C.T), Theta)
    model.constraint(msk.Expr.sub(R, expr), msk.Domain.equalsTo(0))
    
    for i in range(p):
        model.constraint(msk.Expr.vstack(theta_tril.index(diag_index(i)), msk.Expr.mul(0.5, t.index(i)), R.slice([0,i],[min(n,p),i+1]).reshape(min(n,p))), msk.Domain.inRotatedQCone())
        model.constraint(msk.Expr.vstack(theta_tril.index(diag_index(i)), msk.Expr.constTerm(1), lg.index(i)), msk.Domain.inPExpCone())
    
    z_expr = msk.Expr.constTerm(0)
    s_expr = msk.Expr.constTerm(0)
    for i in range(1,p):
        theta_tmp = theta_tril.slice(diag_index(i-1)+1,diag_index(i))
        z_tmp = z.slice(diag_index(i-1)+1,diag_index(i))
        s_tmp = s.slice(diag_index(i-1)+1,diag_index(i))
        expr = msk.Expr.mul(z_tmp, M)
        model.constraint(msk.Expr.sub(expr, theta_tmp), msk.Domain.greaterThan(0))
        model.constraint(msk.Expr.add(theta_tmp, expr), msk.Domain.greaterThan(0))
        expr = msk.Expr.hstack(msk.Expr.mul(0.5, s_tmp), z_tmp, theta_tmp)
        model.constraint(expr, msk.Domain.inRotatedQCone())
        z_expr = msk.Expr.add(z_expr, msk.Expr.sum(z_tmp))
        s_expr = msk.Expr.add(s_expr, msk.Expr.sum(s_tmp))

    z_expr = msk.Expr.mul(l0, z_expr)
    s_expr = msk.Expr.mul(l2, s_expr)
    t_expr = msk.Expr.sum(msk.Expr.sub(t,lg))
    
    
    model.objective(msk.ObjectiveSense.Minimize,msk.Expr.add([t_expr, z_expr, s_expr]))
    
    
    model.setSolverParam("log", 0)
    model.setLogHandler(sys.stdout)
    model.solve()
    
    
    if output_tril:
        return theta_tril.level(), z.level(), model.primalObjValue(), model.dualObjValue()
    else:
        return Theta.level().reshape(p,p), z.fromTril(p).level().reshape(p,p), model.primalObjValue(), model.dualObjValue()



def L2_mosek(Y, l2, M, active_set):
    n,p = Y.shape
    P = p*(p+1)//2
    try:
        import mosek.fusion as msk
        import mosek
    except ModuleNotFoundError:
        raise Exception('Mosek is not installed')
    
    z = get_active_set_mask(active_set,p)
    z = z[np.tril_indices(p,0)].astype(float)

    
    model = msk.Model()
    
    theta_tril = model.variable('theta_tril', P, msk.Domain.unbounded())

    s = model.variable('s', P, msk.Domain.greaterThan(0))
    t = model.variable('t', p, msk.Domain.greaterThan(0))
    lg = model.variable('lg', p, msk.Domain.unbounded())
    R = model.variable('R', [min(n,p),p], msk.Domain.unbounded())
    
    Theta = theta_tril.fromTril(p)
    if n <= p:
        expr = msk.Expr.mul(msk.Matrix.dense(Y), Theta)
    else:
        C = np.linalg.cholesky(Y.T@Y)
        expr = msk.Expr.mul(msk.Matrix.dense(C.T), Theta)
    model.constraint(msk.Expr.sub(R, expr), msk.Domain.equalsTo(0))
    
    for i in range(p):
        model.constraint(msk.Expr.vstack(theta_tril.index(diag_index(i)), msk.Expr.mul(0.5, t.index(i)), R.slice([0,i],[min(n,p),i+1]).reshape(min(n,p))), msk.Domain.inRotatedQCone())
        model.constraint(msk.Expr.vstack(theta_tril.index(diag_index(i)), msk.Expr.constTerm(1), lg.index(i)), msk.Domain.inPExpCone())
    
    
    s_expr = msk.Expr.constTerm(0)
    for i in range(1,p):
        theta_tmp = theta_tril.slice(diag_index(i-1)+1,diag_index(i))
        z_tmp = msk.Expr.constTerm(z[diag_index(i-1)+1:diag_index(i)])
        s_tmp = s.slice(diag_index(i-1)+1,diag_index(i))
        expr = msk.Expr.mul(z_tmp, M)
        model.constraint(msk.Expr.sub(expr, theta_tmp), msk.Domain.greaterThan(0))
        model.constraint(msk.Expr.add(theta_tmp, expr), msk.Domain.greaterThan(0))
        z_tmp = msk.Expr.constTerm(np.ones_like(z[diag_index(i-1)+1:diag_index(i)]))
        expr = msk.Expr.hstack(msk.Expr.mul(0.5, s_tmp), z_tmp, theta_tmp)
        model.constraint(expr, msk.Domain.inRotatedQCone())
        s_expr = msk.Expr.add(s_expr, msk.Expr.sum(s_tmp))

    
    s_expr = msk.Expr.mul(l2, s_expr)
    t_expr = msk.Expr.sum(msk.Expr.sub(t,lg))
    
    
    model.objective(msk.ObjectiveSense.Minimize,msk.Expr.add([t_expr, s_expr]))
    
    
    model.setSolverParam("log", 0)
    model.setLogHandler(sys.stdout)
    model.solve()
    
    

    return Theta.level().reshape(p,p), model.primalObjValue(), model.dualObjValue()


def relax_ASmosek():
    pass