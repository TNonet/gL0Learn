import os
import random
import sys
import time

import numpy as np
from gl0learn import fit, synthetic

from utils import (
    sample_from_cov,
    overlap_covariance_matrix,
)

MAX_OVERLAPS = 6


def test_cd_limited_active_set(p, seed):
    theta_truth = overlap_covariance_matrix(p=p, seed=seed, decay=0.8)
    x = sample_from_cov(theta_truth, n=1000)
    _, _, _, _, Y, _ = synthetic.preprocess(x, assume_centered=False, cholesky=True)

    print("fitting from python")
    results = fit(
        x,
        l0=0,
        scale_x=True,
        max_active_set_ratio=1,
        active_set=np.inf,
        super_active_set=0.0,
    )

    theta_truth_copy = np.copy(theta_truth)
    np.fill_diagonal(theta_truth_copy, 0)
    i, j = np.unravel_index(np.argmax(theta_truth_copy), theta_truth.shape)

    assert results.theta[i, j] > np.mean(theta_truth_copy)


if __name__ == "__main__":
    file_name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
    log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    save_path = os.path.join(log_path, f"output_{time.time_ns()}.txt")
    sys.stdout = open(save_path, "w")

    print("There should be at least one line!", flush=True)

    for _ in range(100):
        p = random.randint(2, 10)
        seed = random.randint(1, 100000)
        print(f"p = {p}, seed = {seed}", flush=True)
        test_cd_limited_active_set(p=p, seed=seed)
