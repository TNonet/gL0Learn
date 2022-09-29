import copy
from time import time
from collections import namedtuple
import math

import numpy as np
from numba.typed import List
from numba import njit

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

from ._coordinate_descent import L2_CD, L0L2_CD, L0L2_CD_loop
from ._local_search import L0L2_CDPSI
from ._cost import get_L2_primal_cost,get_L2_dual_cost, get_L0L2_cost
from ..utils import compute_relative_gap, support_to_active_set, trivial_soln, get_active_set_mask



EPSILON = np.finfo('float').eps

def L2_CD_solve(Y, l2, M, support, S_diag=None, warm_start=None, rel_tol=1e-4,cd_max_itr=100,kkt_max_itr=100,verbose=False):
    if S_diag is None:
        S_diag = np.linalg.norm(Y,axis=0)**2
    if warm_start is not None:
        Theta = np.copy(warm_start['Theta'])
        R = warm_start.get("R", Y@Theta)
    else:
        Theta, R = trivial_soln(Y,S_diag)
    
    active_set = support_to_active_set(support)
    cost = get_L2_primal_cost(Y, Theta, R, l2, M, active_set)
    cd_tol = rel_tol/2
    if verbose:
        print("cost", cost)
    curiter = 0
    while curiter < kkt_max_itr:
        Theta, cost, R = L2_CD(Y, Theta, cost, l2, M, S_diag, active_set, R, cd_tol, cd_max_itr, verbose)
        if verbose:
            print("iter", curiter+1)
            print("cost", cost)
            
        dual_cost = get_L2_dual_cost(Y, Theta, R, l2, M, active_set)
        if verbose:
            print("dual", dual_cost)
        
        if (compute_relative_gap(cost, dual_cost) < rel_tol) or (cd_tol < 1e-8):
            break
        else:
            cd_tol /= 100
        curiter += 1
    return Theta, cost, R

EPSILON = np.finfo('float').eps
def _initial_active_set(Y, Theta, support):
    p = Y.shape[1]
    corr = np.corrcoef(Y.T)
    k = min(int(0.2*p), math.floor(np.sqrt(p)))
    argpart = np.argpartition(-np.abs(corr), k, axis=1)[:,:k]
    active_set = set()
    
    for i in range(p):
        for j in argpart[i]:
            if i < j and ((i,j) in support):
                active_set.add((i,j))
            elif i > j and ((j,i) in support):
                active_set.add((j,i))
    
    argwhere = np.argwhere(np.abs(Theta)>EPSILON*1e10)
    for i,j in argwhere:
        if i<j  and ((i,j) in support):
            active_set.add((i,j))
    active_set = support_to_active_set(active_set)
    return active_set

@njit(cache=True)
def _refined_initial_active_set(Y, Theta, l0, l2, M, S_diag, active_set, support, R):
    support.clear()
    num_of_similar_supports = 0
    delta = 0
    while num_of_similar_supports < 3:
        delta = 0
        Theta, R = L0L2_CD_loop(Y, Theta, l0, l2, M, S_diag, active_set, R)
        for i,j in active_set:
            if (Theta[i,j]!=0)  and (i<j) and ((i,j) not in support):
                support.add((i,j))
                delta += 1
        if delta == 0:
            num_of_similar_supports += 1
        else:
            num_of_similar_supports = 0
    return support, Theta, R

def _initialize_active_set_algo(Y, l0, l2, M, S_diag, upper_support, warm_start):
    p = Y.shape[1]
    if S_diag is None:
        S_diag = np.linalg.norm(Y, axis=0)**2
#     if (warm_start is not None) and (np.count_nonzero(warm_start['Theta']) == p**2-p):
    if (warm_start is not None):
        support, Theta = warm_start['support'], np.copy(warm_start['Theta'])
        R = warm_start.get('R', Y@Theta)
    else:
        Theta, R = trivial_soln(Y, S_diag)
        active_set = _initial_active_set(Y, Theta, upper_support)
        support = {(0,0)}
        support, Theta, R = _refined_initial_active_set(Y, Theta, l0, l2, M, S_diag, active_set, support, R)
        
    return Theta, R, support, S_diag

@njit(cache=True, parallel=True)
def _above_threshold_indices(l0,l2,M,Y,S_diag,Theta,R,active_set_mask):
    Theta_diag = np.diag(Theta)
    a = S_diag/Theta_diag.reshape(-1,1)+S_diag.reshape(-1,1)/Theta_diag
    b = 2*Y.T@R/Theta_diag+2*R.T@Y/Theta_diag.reshape(-1,1)
    criterion = np.where(np.abs(b)/2/(a+l2)<=M, 4*l0*(a+l2)-b**2, a*M**2-np.abs(b)*M+l0+l2*M**2)
    above_threshold = np.argwhere(active_set_mask*criterion<0)
    return above_threshold


def L0L2_ASCD(Y, l0, l2, M, upper_support, S_diag=None, warm_start=None, cd_tol=1e-4, cd_max_itr=100, rel_tol=1e-6, kkt_max_itr=100, maxtime=np.inf, verbose=False):
    st = time()
    p = Y.shape[1]
    Theta, R, support, S_diag = _initialize_active_set_algo(Y, l0, l2, M, S_diag, upper_support, warm_start)
    active_set = support_to_active_set(support)
    
    upper_active_set_mask = get_active_set_mask(support_to_active_set(upper_support),p)
    cost = get_L0L2_cost(Y, Theta, R, l0, l2, M, active_set)
    old_cost = cost
    if verbose:
        print("cost", cost)
    curiter = 0
    while curiter < kkt_max_itr and time()-st < maxtime:
        Theta, cost, R = L0L2_CD(Y, Theta, cost, l0, l2, M, S_diag, active_set, R, cd_tol, cd_max_itr, verbose)
        if verbose:
            print("iter", curiter+1)
            print("cost", cost)
        above_threshold = _above_threshold_indices(l0,l2,M,Y,S_diag,Theta,R,upper_active_set_mask)
        outliers = [(i,j) for i,j in above_threshold if i < j and (i,j) not in support and (i,j) in upper_support]
        if not outliers:
            if verbose:
                print("no outliers, computing relative accuracy...")
            if compute_relative_gap(cost, old_cost) < rel_tol or cd_tol < 1e-8:
                break
            else:
                cd_tol /= 100
                old_cost = cost
        
        support = support | set(outliers)
        active_set = support_to_active_set(support)
        curiter += 1
    if curiter == kkt_max_itr:
        print('Maximum KKT check iterations reached, increase kkt_max_itr '
              'to avoid this warning')
    return Theta, cost, R, support


def L0L2_ASCDPSI(Y, l0, l2, M, upper_support, S_diag=None, warm_start=None, cd_tol=1e-4, cd_max_itr=100, swap_max_itr=10, rel_tol=1e-6, kkt_max_itr=100, maxtime=np.inf, verbose=False):
    st = time()
    p = Y.shape[1]
    Theta, R, support, S_diag = _initialize_active_set_algo(Y, l0, l2, M, S_diag, upper_support, warm_start)
    active_set = support_to_active_set(support)
    
    upper_active_set_mask = get_active_set_mask(support_to_active_set(upper_support),p)
    cost = get_L0L2_cost(Y, Theta, R, l0, l2, M, active_set)
    old_cost = cost
    if verbose:
        print("cost", cost)
    curiter = 0
    while curiter < kkt_max_itr and time()-st < maxtime:
        Theta, cost, R = L0L2_CDPSI(Y, Theta, cost, l0, l2, M, S_diag, active_set, R, cd_tol, cd_max_itr,swap_max_itr, verbose)
        if verbose:
            print("iter", curiter+1)
            print("cost", cost)
        above_threshold = _above_threshold_indices(l0,l2,M,Y,S_diag,Theta,R,upper_active_set_mask)
        outliers = [(i,j) for i,j in above_threshold if i < j and (i,j) not in support and (i,j) in upper_support]
        
        if not outliers:
            if verbose:
                print("no outliers, computing relative accuracy...")
            if compute_relative_gap(cost, old_cost) < rel_tol or cd_tol < 1e-8:
                break
            else:
                cd_tol /= 100
                old_cost = cost
        
        support = support | set(outliers)
        active_set = support_to_active_set(support)
        curiter += 1
    if curiter == kkt_max_itr:
        print('Maximum KKT check iterations reached, increase kkt_max_itr '
              'to avoid this warning')
    return Theta, cost, R, support


def _initialize_algo(Y, S_diag, Theta, z, solver, support_type, p):
    assert support_type in {"all", "nonzeros", "rounding"}
    if Theta is None and z is None:
        support = set([(i,j) for i in range(p) for j in range(i+1,p)])
    elif z is not None:
        if support_type == "all":
            support = set([(i,j) for i in range(p) for j in range(i+1,p)])
        elif support_type == "rounding":
            support = set([(i,j) for i,j in np.argwhere(np.round(z)) if i < j])
        elif support_type == "nonzeros":
            support = set([(i,j) for i,j in np.argwhere(z) if i < j])
    elif Theta is not None:
        if support_type == "all":
            support = set([(i,j) for i in range(p) for j in range(i+1,p)])
        else:
            support = set([(i,j) for i,j in np.argwhere(Theta) if i < j])
    
    active_set = support_to_active_set(support)
    active_set_mask = get_active_set_mask(active_set,p)
    np.fill_diagonal(active_set_mask, True)
    if Theta is not None:
        Theta = np.where(active_set_mask, Theta, 0)
        R = Y@Theta
    elif "AS" in solver:
        Theta, R = None, None
    else:
        Theta, R = trivial_soln(Y, S_diag)
    return Theta, R, support, active_set


def heuristic_solve(Y, l0, l2, M, solver="L0L2_CDPSI", support_type="all", Theta=None, z=None, \
                    S_diag=None, rel_tol=1e-4, cd_max_itr=100, verbose=False, **kwargs):
    p = Y.shape[1]
    assert solver in {"L0L2_CDPSI", "L0L2_CD", "L2_CDApprox", "L2_CD", "L0L2_ASCD", "L0L2_ASCDPSI"}
    if S_diag is None:
        S_diag = np.linalg.norm(Y,axis=0)**2
    
    Theta, R, support, active_set = _initialize_algo(Y, S_diag, Theta, z, solver, support_type, p)
    if solver in {"L0L2_CDPSI", "L0L2_CD"}:
        cost = get_L0L2_cost(Y, Theta, R, l0, l2, M, active_set)
    elif solver in {"L2_CD","L2_CDApprox"}:
        cost = get_L2_primal_cost(Y, Theta, R, l2, M, active_set)
    
    if solver == "L0L2_CD":
        Theta, cost, R = L0L2_CD(Y, Theta, cost, l0, l2, M, S_diag, active_set, R, rel_tol, cd_max_itr, verbose)
    elif solver == "L0L2_CDPSI":
        swap_max_itr = kwargs.get("swap_max_itr", 10)
        Theta, cost, R = L0L2_CDPSI(Y, Theta, cost, l0, l2, M, S_diag, active_set, R, rel_tol, cd_max_itr, swap_max_itr, verbose)
    elif solver == "L2_CDApprox":
        Theta, cost, R = L2_CD(Y, Theta, cost, l2, M, S_diag, active_set, R, rel_tol, maxiter, verbose)
        cost += len(support)*l0
    elif solver == "L2_CD":
        kkt_max_itr = kwargs.get("kkt_max_itr",100)
        warm_start = dict()
        warm_start['Theta'] = Theta
        warm_start['R'] = R
        Theta, cost, R = L2_CD_solve(Y, l2, M, support, S_diag, warm_start, rel_tol,cd_max_itr,kkt_max_itr,verbose)
        cost += len(support)*l0
    elif solver in {"L0L2_ASCD", "L0L2_ASCDPSI"}:
        if Theta is not None:
            warm_start = dict()
            warm_start['Theta'] = Theta
            warm_start['R'] = R
            warm_start['support'] = set([(i,j) for i,j in np.argwhere(Theta) if i < j])
        else:
            warm_start = None
        if solver == "L0L2_ASCD":
            kkt_max_itr = kwargs.get("kkt_max_itr",100)
            Theta, cost, R, support = L0L2_ASCD(Y, l0, l2, M, support, S_diag, warm_start, rel_tol, cd_max_itr, rel_tol, kkt_max_itr, verbose=verbose)
        elif solver == "L0L2_ASCDPSI":
            kkt_max_itr = kwargs.get("kkt_max_itr",100)
            swap_max_itr = kwargs.get("swap_max_itr", 10)
            Theta, cost, R, support = L0L2_ASCDPSI(Y, l0, l2, M, support, S_diag, warm_start, rel_tol, cd_max_itr, swap_max_itr, rel_tol, kkt_max_itr, verbose=verbose)
    support = set([(i,j) for [i,j] in np.argwhere(Theta) if i<j])
    return Theta, cost, R, support

class L0L2Solver:
    def __init__(self, X, assumed_centered=False, cholesky=False):
        self.n, self.p, self.X, self.X_mean, self.Y, self.S_diag = preprocess(X,assume_centered,cholesky)
        
    def solve(self, l0, l2, M, solver="ASCD", warm_start=None, verbose=False):
        pass