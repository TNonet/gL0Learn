import numpy as np
from numba import njit
from numba.typed import List

from ..utils import compute_relative_gap
from ._cost import get_L2_primal_cost, get_L0L2_cost
from ..oracle import Q_L2reg, Q_L0L2reg, R_nl


@njit(cache=True)
def L2_CD_loop(Y, Theta, l2, M, S_diag, active_set, R):
    p = Y.shape[1]
    for i,j in active_set:
        theta_old = Theta[i,j]
        Theta[i,j] = Theta[j,i] = Q_L2reg(S_diag[j]/Theta[i,i]+S_diag[i]/Theta[j,j], \
                                        (2*Y[:,j]@R[:,i]-2*Theta[i,j]*S_diag[j])/Theta[i,i]+(2*Y[:,i]@R[:,j]-2*Theta[i,j]*S_diag[i])/Theta[j,j], l2, M)
        R[:,i] = R[:,i] + (Theta[i,j]-theta_old)*Y[:,j]
        R[:,j] = R[:,j] + (Theta[i,j]-theta_old)*Y[:,i]
    
    for i in range(p):
        R[:,i] = R[:,i] - Theta[i,i]*Y[:,i]
        Theta[i,i] = R_nl(S_diag[i], R[:,i]@R[:,i])
        R[:,i] = R[:,i] + Theta[i,i]*Y[:,i]
    
    return Theta, R

@njit(cache=True)
def L2_CD(Y, Theta, cost, l2, M, S_diag, active_set, R, rel_tol=1e-8, maxiter=3000, verbose=False):
    tol = 1
    old_cost = cost
    curiter = 0
    while tol > rel_tol and curiter < maxiter:
        old_cost = cost
        Theta, R = L2_CD_loop(Y, Theta, l2, M, S_diag, active_set, R)
        cost = get_L2_primal_cost(Y, Theta, R, l2, M, active_set)
        if verbose:
            print(cost)
        tol = abs(compute_relative_gap(old_cost, cost))
        curiter += 1
    return Theta, cost, R


@njit(cache=True)
def L0L2_CD_loop(Y, Theta, l0, l2, M, S_diag, active_set, R):
    p = Y.shape[1]
    theta_old =0 
    for i,j in active_set:
        theta_old = Theta[i,j]
        Theta[i,j] = Theta[j,i] = Q_L0L2reg(S_diag[j]/Theta[i,i]+S_diag[i]/Theta[j,j], \
                                            (2*Y[:,j]@R[:,i]-2*Theta[i,j]*S_diag[j])/Theta[i,i]+(2*Y[:,i]@R[:,j]-2*Theta[i,j]*S_diag[i])/Theta[j,j], l0, l2, M)
        R[:,i] = R[:,i] + (Theta[i,j]-theta_old)*Y[:,j]
        R[:,j] = R[:,j] + (Theta[i,j]-theta_old)*Y[:,i]
    
    for i in range(p):
        R[:,i] = R[:,i] - Theta[i,i]*Y[:,i]
        Theta[i,i] = R_nl(S_diag[i], R[:,i]@R[:,i])
        R[:,i] = R[:,i] + Theta[i,i]*Y[:,i]
    
    return Theta, R


@njit(cache=True)
def L0L2_CD(Y, Theta, cost, l0, l2, M, S_diag, active_set, R, rel_tol=1e-8, maxiter=3000, verbose=False):
    tol = 1
    old_cost = cost
    curiter = 0
    while tol > rel_tol and curiter < maxiter:
        old_cost = cost
        Theta, R = L0L2_CD_loop(Y, Theta, l0, l2, M, S_diag, active_set, R)
        cost = get_L0L2_cost(Y, Theta, R, l0, l2, M, active_set)
        if verbose:
            print(cost)
        tol = abs(compute_relative_gap(old_cost, cost))
        curiter += 1
    return Theta, cost, R
