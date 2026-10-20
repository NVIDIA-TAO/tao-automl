# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared GP / expected-improvement numerical kernel.

Extracted from tao_automl.brain.bayesian; acquisition originally based on
http://krasserm.github.io/2018/03/21/bayesian-optimization/ . Repository-specific
parameter mapping, fidelity and model semantics deliberately live elsewhere.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


def make_gp(dimensions, seed=95051):
    return GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * Matern(length_scale=[1.0] * dimensions, nu=2.5),
        alpha=1e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=10,
        random_state=seed,
    )


def negative_expected_improvement(gp, samples, point, xi=0.01):
    mean, std = gp.predict(np.asarray(point).reshape(1, -1), return_std=True)
    if std[0] <= 0:
        return 0.0
    improvement = mean[0] - np.max(gp.predict(np.asarray(samples))) - xi
    z = improvement / std[0]
    return -float(improvement * norm.cdf(z) + std[0] * norm.pdf(z))


def optimize_acquisition(acquisition, dimensions, random, restarts=5):
    """Minimize negative EI, retaining a finite candidate if scipy fails."""
    best = np.asarray(random(dimensions))
    best_value = acquisition(best)
    for _ in range(restarts):
        start = np.asarray(random(dimensions))
        result = minimize(
            acquisition, x0=start, bounds=[(0, 1)] * dimensions, method="L-BFGS-B"
        )
        for candidate in (start, result.x):
            value = acquisition(candidate)
            if np.isfinite(value) and (
                not np.isfinite(best_value) or value < best_value
            ):
                best, best_value = candidate, value
    return np.clip(best, 0, 1).reshape(-1)
