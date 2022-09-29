import numpy as np
from numba import njit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)



@njit(cache=True)
def get_ratio_threshold(l0, l2, m):
    ratio = np.sqrt(l0 / l2) if l2 != 0 else np.Inf
    threshold = 2 * np.sqrt(l0 * l2) if ratio <= m else l0 / m + l2 * m
    return ratio, threshold


@njit(cache=True)
def compute_relative_gap(cost1, cost2, which="both", one=True):
    if cost1 == np.inf or cost2 == -np.inf:
        return 1.
    if cost1 == -np.inf or cost2 == np.inf:
        return -1.
    if which == "both":
        benchmark = max(abs(cost1),abs(cost2))
    elif which == "first":
        benchmark = abs(cost1)
    elif which == "second":
        benchmark = abs(cost2)
    if one:
        benchmark = max(benchmark,1)
    return (cost1-cost2)/benchmark

@njit(cache=True)
def get_active_set_mask(active_set,p):
    mask = np.full((p,p),False)
    for i,j in active_set:
        mask[i,j] = mask[j,i] = True
    return mask


@njit(cache=True)
def trivial_soln(Y, S_diag):
    p = Y.shape[1]
    Theta = np.zeros((p,p))
    np.fill_diagonal(Theta, 1/S_diag)
    R = Y / S_diag
    return Theta, R

def support_to_active_set(support):
    if len(support) == 0:
        return np.zeros((0,2),dtype=int)
    else:
        return np.array(sorted(support))
    
@njit(cache=True)
def diag_index(i):
    return (i+1)*(i+2)//2-1


@njit(cache=True)
def fromTril(tril_values, p, diag=True):
    res = np.zeros((p,p))
    k = 0
    for i in range(p):
        for j in range(i+diag):
            res[i,j] = res[j,i] = tril_values[k]
            k += 1
    return res

@njit(cache=True)
def fillTril(arr, tril_values, p, diag=True):
    k = 0
    for i in range(p):
        for j in range(i+diag):
            arr[i,j] = arr[j,i] = tril_values[k]
            k += 1
    return arr

@njit(cache=True)
def toTril(mat, p, diag=True):
    if diag:
        res = np.zeros(p*(p+1)//2)
    else:
        res = np.zeros(p*(p-1)//2)
    k = 0
    for i in range(p):
        for j in range(i+diag):
            res[k] = mat[i,j]
    return res

@njit(cache=True)
def getTril(arr, mat, p, diag=True):
    k = 0
    for i in range(p):
        for j in range(i+diag):
            arr[k] = mat[i,j]
    return arr