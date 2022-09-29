import numpy as np
from numba import njit

import warnings

from numba.core.errors import NumbaDeprecationWarning, \
    NumbaPendingDeprecationWarning, NumbaPerformanceWarning

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


from .utils import get_ratio_threshold

@njit(cache=True)
def reverse_huber(t):
    return np.abs(t) if np.abs(t)<=1 else (t**2+1)/2

@njit(cache=True)
def psi(beta,l0,l2,M,ratio=0.,threshold=0.,suff=False):
    abs_beta = np.abs(beta)
    if abs_beta > M:
        return np.inf
    if not suff:
        ratio, _ = get_ratio_threshold(l0, l2, M)
    return 2*l0*reverse_huber(beta/ratio) if ratio <= M else abs_beta*(l0/M+l2*M)

@njit(cache=True)
def prox_psi(beta,l0,l2,M,ratio=0.,threshold=0.,suff=False):
    abs_beta = np.abs(beta)
    if not suff:
        ratio, threshold = get_ratio_threshold(l0, l2, M)
    if ratio <= M:
        res = np.maximum(abs_beta - threshold,0)
        res = res if res <= ratio else abs_beta / (1+2*l2)
    else:
        res = np.maximum(abs_beta - threshold,0)
        res = res if res <= M else M
    
    res *= np.sign(beta)
    
    return res

@njit(cache=True)
def quad_psi(x,a,b,l0,l2,M):
    return a*x**2+b*x+psi(x,l0,l2,M)


@njit(cache=True)
def Q_psi(a,b,l0,l2,M,ratio=0.,threshold=0.,suff=False):
    ####
    # argmin_x ax^2+bx+psi(x,l0,l2,M)
    ####
    beta = -b / (2*a)
    l0 /= (2*a)
    l2 /= (2*a)
    if not suff:
        return prox_psi(beta,l0,l2,M)
    else:
        threshold /= (2*a)
        return prox_psi(beta,l0,l2,M,ratio,threshold,suff)

@njit(cache=True)
def phi(beta,z,l0,l2,M):
    abs_beta = np.abs(beta)
    if abs_beta > M:
        return np.inf
    return l0*z+l2*abs_beta**2
    
@njit(cache=True)
def prox_phi(beta,z,l2,M):
    if z == 0:
        return 0.
    else:
        res = beta/(1+2*l2)
        return res if np.abs(res)<=M else M*np.sign(res)

@njit(cache=True)
def Q_phi(a,b,z,l2,M):
    return prox_phi(-b/(2*a),z,l2/(2*a),M)
    

@njit(cache=True)
def quad_neglog(x, a, b):
    ####
    # -log(x)+ax^2+bx
    ####
    return -np.log(x)+a*x**2+b*x

@njit(cache=True)
def Q_nl(a, b):
    ####
    # argmin_x -log(x)+ax^2+bx
    ####
    return (-b+np.sqrt(b**2+8*a))/(4*a)


@njit(cache=True)
def lin_recipr_neglog(x,a,b):
    ####
    # -log(x)+ax+b/x
    ####
    return -np.log(x)+a*x+b/x

@njit(cache=True)
def R_nl(a,b):
    ####
    # argmin_x -log(x)+ax+b/x
    ####
    return (1+np.sqrt(1+4*a*b))/(2*a)


@njit(cache=True)
def L0L2reg(beta,l0,l2,M):
    if np.abs(beta) > M:
        return np.inf
    return l0*(beta!=0) + l2*beta**2


@njit(cache=True)
def prox_L0L2reg(beta,l0,l2,M):
    val = np.abs(beta)/(1+2*l2)
    if val <= M:
        return np.sign(beta)*val if val>np.sqrt(2*l0/(1+2*l2)) else 0
    else:
        return np.sign(beta)*M if val>M/2 + l0/M/(1+2*l2) else 0


@njit(cache=True)
def quad_L0L2reg(x,a,b,l0,l2,M):
    return a*x**2+b*x+L0L2reg(x,l0,l2,M)


@njit(cache=True)
def Q_L0L2reg(a,b,l0,l2,M):
    beta = -b / (2*a)
    l0 /= (2*a)
    l2 /= (2*a)
    return prox_L0L2reg(beta,l0,l2,M)


@njit(cache=True)
def prox_L0L2reg_vec(beta,l0,l2,M):
    val = np.abs(beta)/(1+2*l2)
    thres1 = np.sqrt(2*l0/(1+2*l2))
    thres2 = M/2 + l0/M/(1+2*l2)
    return np.sign(beta)*np.where(val<=M, np.where(val>thres1, val, 0), np.where(val>thres2, M, 0))

@njit(cache=True)
def Q_L0L2reg_obj(a,b,l0,l2,M):
    beta = -b/(2*a)
    l0 = l0/(2*a)
    l2 = l2/(2*a)
    x = prox_L0L2reg(beta,l0,l2,M)
    return x, a*x**2+b*x+l0*(x!=0)+l2*x**2

@njit(cache=True)
def Q_L0L2reg_obj_vec(a,b,l0,l2,M):
    beta = -b / (2*a)
    l0_v = l0 / (2*a)
    l2_v = l2 / (2*a)
    x = prox_L0L2reg_vec(beta,l0_v,l2_v,M)
    return x, a*x**2+b*x+l0*(x!=0)+l2*x**2


@njit(cache=True)
def L2reg(beta,l2,M):
    if np.abs(beta) > M:
        return np.inf
    return l2*beta**2

@njit(cache=True)
def prox_L2reg(beta,l2,M):
    return np.sign(beta)*np.minimum(np.abs(beta)/(1+2*l2), M)

@njit(cache=True)
def quad_L2reg(x,a,b,l2,M):
    return a*x**2+b*x+L2reg(x,l0,l2,M)

@njit(cache=True)
def Q_L2reg(a,b,l2,M):
    beta = -b / (2*a)
    l2 /= (2*a)
    return prox_L2reg(beta,l2,M)