import numpy as np
from numba import njit

@njit(cache=True)
def get_L0L2_cost(Y, Theta, R, l0, l2, M, active_set):
    n,p = Y.shape
    cost = 0
    for i in range(p):
        cost -= np.log(Theta[i,i])
        cost += R[:,i]@R[:,i]/Theta[i,i]
    
    for i,j in active_set:
        if Theta[i,j] != 0:
            cost += l0+l2*Theta[i,j]**2
            
    return cost

@njit(cache=True)
def get_L2_primal_cost(Y, Theta, R, l2, M, active_set):
    n,p = Y.shape
    cost = 0
    for i in range(p):
        cost -= np.log(Theta[i,i])
        cost += R[:,i]@R[:,i]/Theta[i,i]
    
    for i,j in active_set:
        if Theta[i,j] != 0:
            cost += l2*Theta[i,j]**2
            
    return cost

@njit(cache=True)
def get_L2_dual_cost(Y, Theta, R, l2, M, active_set):
    p = Y.shape[1]
    res = p
    tmp = 0.
    for i in range(p):
        tmp = -np.linalg.norm(R[:,i]/Theta[i,i], 2)**2 +2* R[:,i]@Y[:,i]/Theta[i,i]
        if tmp <= 0:
            return -np.inf
        res += np.log(tmp)
    
    a = 2*M*l2 if l2 != 0 else 0
    pen = 0.
    cur_nuy = 0
    for i,j in active_set:
        cur_nuy = 2*abs(R[:,i]@Y[:,j]/Theta[i,i]+R[:,j]@Y[:,i]/Theta[j,j])
        if l2 == 0:
            pen += M*cur_nuy
        elif cur_nuy <= a:
            pen += cur_nuy**2/(4*l2)
        else:
            pen += (M*cur_nuy-l2*M**2)
    return res-pen