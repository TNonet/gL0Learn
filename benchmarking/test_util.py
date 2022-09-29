import numpy as np
import copy
from numpy import random
import math as math
from l0bnb2 import heuristic_solve, preprocess



import pandas as pd
import matplotlib.pyplot as plt






"""
Run clime, scad or glasso
Input is normalized and centered
"""


        
"""
Parameter selection for BnB
The value of lambda is chosen in the interval [lambda_core/a, lambda_core*a] over a logarithmic grid of n_lambda points.
c is the validation cost used (see validation loss).
The data is assumed to be centered and normalized.
"""


def validate_lambda_l0l2(X_train, X_val, n_train, n_val, p,  n_lambda,  M, lambda_0_core, lambda_2_core, a, c):
    
    Y = X_train
    min_loss = 1e10
    l0_best = 0
    l2_best = 0
    Lambda0 =  np.logspace(np.log10(lambda_0_core/a), np.log10(lambda_0_core*a), n_lambda)
    Lambda2 =  np.logspace(np.log10(lambda_2_core/a), np.log10(lambda_2_core*a), n_lambda)
    
    Y = X_train
    
    Theta_0 = np.ones((p,p))*0.001
    for i in range(n_lambda):
        l0 = Lambda0[n_lambda-i-1]
        for j in range(n_lambda):
            l2 = Lambda2[n_lambda-j-1]
            Theta_0, _, _, _ = heuristic_solve(Y,  l0,l2 ,M, solver="L0L2_ASCD", support_type="all", Theta=Theta_0, z=None, \
                                                S_diag=None, rel_tol=1e-6, cd_max_itr=100, verbose=False, kkt_max_itr=100, cd_tol=1e-4)
            cv_loss_temp= validation_loss(Theta_0, p, X_val, n_val,c)
            if cv_loss_temp< min_loss:
                min_loss = cv_loss_temp
                Best_theta = copy.deepcopy(Theta_0)
                l0_best = l0
                l2_best = l2
    return Best_theta, l0_best, l2_best, min_loss
    
"""
Validation loss
Input is assumed to be centered and normalized
c is the validation cost: pseudolikelihood or likelihood
"""
    
def validation_loss(Theta, p, X_val, n, c):
    
    assert c in {"pseudolikelihood", "likelihood"}
    
    Theta = (Theta + np.transpose(Theta))/2
    
    loss = 0
    if  c == "pseudolikelihood":
        R = np.matmul(X_val, Theta)
        th_d = np.tile( np.sqrt(np.diag(Theta)), (n, 1))
        R = np.divide(R,th_d)
        loss = -np.sum(np.log((np.diag(Theta))))
        loss = loss + np.square(np.linalg.norm(R, ord='fro'))
    else:
        S = np.matmul(np.transpose(X_val),X_val)
        loss = np.trace(np.matmul(S,Theta)) - np.log(np.linalg.det(Theta))
    return loss

    
"""
Report support recovery performance
"""
    
def check_support(Theta_truth, Theta_rec, p):
    mask_truth_bool = np.abs(Theta_truth)>1e-3
    mask_truth = mask_truth_bool.astype(int)
    mask_rec_bool = np.abs(Theta_rec)>1e-3
    mask_rec = mask_rec_bool.astype(int)
    
    nonzeors_true = np.sum(mask_truth)
    nonzeors_rec = np.sum(mask_rec)
    
    zeros_rec = p*p - nonzeors_rec

    
    fn_mask_bool = (mask_truth - mask_rec)>1e-4
    fn_mask = fn_mask_bool.astype(int)
    fp_mask_bool = (mask_truth - mask_rec)<-1e-4
    fp_mask = fp_mask_bool.astype(int)
    
    FP = np.sum(fp_mask)
    FN = np.sum(fn_mask)
    TP = nonzeors_rec - FP
    TN = zeros_rec - FN
    
    
    a = np.sqrt((TP+FP)*(TP+FN))
    b = np.sqrt((TN+FP)*(TN+FN))
    
    MCC = (TP*TN-FP*FN)/a/b
    
    return FP, FN, nonzeors_rec, nonzeors_true, MCC