from copy import deepcopy


import numpy as np
import pytest
from gl0learn import fit, synthetic, Penalty
from gl0learn.metrics import nonzeros, pseudo_likelihood_loss
from gl0learn.utils import triu_nnz_indicies
from hypothesis import given, settings, assume, HealthCheck, note
from hypothesis.strategies import just, booleans, floats, integers, random_module

from utils import (
    sample_from_cov,
    overlap_covariance_matrix,
    is_scipy_installed,
    is_mosek_installed,
    make_bisect_func,
    random_penalty,
    random_penalty_values,
    top_n_triu_indicies_by_abs_value,
)


theta_truth = overlap_covariance_matrix(p=p, seed=1, decay=0.8)
x = sample_from_cov(theta_truth, n=1000)
_, _, _, _, Y, _ = synthetic.preprocess(x, assume_centered=False, cholesky=True)
results = fit(
    x,
    l0=0,
    scale_x=True,
    max_active_set_ratio=1,
    active_set=np.inf,
    super_active_set=0.0,
)