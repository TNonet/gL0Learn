import numpy as np
from numba import njit
from numba.typed import List


from ..utils import compute_relative_gap, get_active_set_mask
from ..oracle import Q_L2reg, Q_L0L2reg, Q_L0L2reg_obj, Q_L0L2reg_obj_vec, R_nl
from ._cost import get_L0L2_cost
from ._coordinate_descent import L0L2_CD


@njit(cache=True)
def PSI_by_row(row, Y, Theta, l0, l2, M, S_diag, nonzeros, zeros, R, verbose=False):
    i = row
    Theta_diag = np.diag(Theta)
    swap = False
    for j in nonzeros:
        R[:,i] = R[:,i] - Theta[i,j]*Y[:,j]
        R[:,j] = R[:,j] - Theta[i,j]*Y[:,i]
        Theta[i,j] = Theta[j,i] = 0
        aj = S_diag[i]/Theta[j,j] + S_diag[j]/Theta[i,i]
        bj = 2*Y[:,j]@R[:,i]/Theta[i,i] + 2*Y[:,i]@R[:,j]/Theta[j,j]
        theta, f = Q_L0L2reg_obj(aj,bj,l0,l2,M)
        a = S_diag[i]/Theta_diag[zeros] + S_diag[zeros]/Theta[i,i]
        b = 2*Y[:,zeros].T@R[:,i]/Theta[i,i] + 2*R[:,zeros].T@Y[:,i]/Theta_diag[zeros]
        thetas, fs = Q_L0L2reg_obj_vec(a,b,l0,l2,M)
        if f < np.min(fs):
            Theta[i,j] = Theta[j,i] = theta
            R[:,i] = R[:,i] + Theta[i,j]*Y[:,j]
            R[:,j] = R[:,j] + Theta[i,j]*Y[:,i]
        else:
            ell = np.argmin(fs)
            k = zeros[ell]
            Theta[i,k] = Theta[k,i] = thetas[ell]
            R[:,i] = R[:,i] + Theta[i,k]*Y[:,k]
            R[:,k] = R[:,k] + Theta[i,k]*Y[:,i]
            swap=True
            if verbose:
                print("row ",i,": ", j, " swapped with ", k)
            break
    return Theta, R, swap


@njit(cache=True)
def L0L2_CDPSI(Y, Theta, cost, l0, l2, M, S_diag, active_set, R, rel_tol=1e-8, cd_max_itr=3000, swap_max_itr=10, verbose=False):
    p = Y.shape[1]
    Theta, cost, R = L0L2_CD(Y, Theta, cost, l0, l2, M, S_diag, active_set, R, rel_tol, cd_max_itr, verbose)
    active_set_mask = get_active_set_mask(active_set,p)
    swap = False
    for itr in range(swap_max_itr):
        swap = False
        if verbose:
            print("PSI started...")
        for row in range(p):
            nonzeros = np.array([i for i in np.where(active_set_mask[row])[0] if Theta[row,i]!=0 and i!=row])
            zeros = np.array([i for i in np.where(active_set_mask[row])[0] if Theta[row,i]==0])
            if len(nonzeros)==0 or len(zeros) ==0:
                continue
            Theta, R, row_swap =  PSI_by_row(row, Y, Theta, l0, l2, M, S_diag, nonzeros, zeros, R, verbose)
            swap = swap | row_swap
        for i in range(p):
            R[:,i] = R[:,i] - Theta[i,i]*Y[:,i]
            Theta[i,i] = R_nl(S_diag[i], R[:,i]@R[:,i])
            R[:,i] = R[:,i] + Theta[i,i]*Y[:,i]
        cost = get_L0L2_cost(Y, Theta, R, l0, l2, M, active_set)
        if verbose:
            print("PSI finished...")
            print("cost:", cost)
            print("swap:", swap)
        if not swap:
            break
        Theta, cost, R = L0L2_CD(Y, Theta, cost, l0, l2, M, S_diag, active_set, R, rel_tol, cd_max_itr, verbose)
    return Theta, cost, R