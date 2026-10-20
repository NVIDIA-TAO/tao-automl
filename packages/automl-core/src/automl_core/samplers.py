# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical samplers adapted from the original BFBO, BOHB and DEHB brains.

No network registry, spec mutation, SDK, or metric-name direction heuristic.
"""

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from automl_core.bayesian import optimize_acquisition
from automl_core.contracts import canonical, lookup, override
from automl_core.search import features


def score(spec, trial):
    return trial["result"]["metric"] * (
        1 if spec.objective.direction == "maximize" else -1
    )


def decode(spec, point):
    return override(
        spec.base_config,
        {
            name: spec.parameters[name].decode(float(value))
            for name, value in zip(sorted(spec.parameters), point)
        },
    )


def encode(spec, config):
    point = []
    for name in sorted(spec.parameters):
        parameter, value = spec.parameters[name], lookup(config, name)
        if parameter.kind == "categorical":
            choices = [canonical(choice) for choice in parameter.choices]
            unit = (choices.index(canonical(value)) + 0.5) / len(choices)
        else:
            if type(value) not in (int, float) or not np.isfinite(value):
                raise ValueError(f"Non-numeric proposal for {name}")
            if not parameter.low <= value <= parameter.high or (
                parameter.kind == "int" and type(value) is not int
            ):
                raise ValueError(f"Proposal outside approved domain: {name}")
            if parameter.log:
                unit = np.log(value / parameter.low) / np.log(
                    parameter.high / parameter.low
                )
            else:
                unit = (value - parameter.low) / (parameter.high - parameter.low)
        point.append(float(unit))
    return point


def bfbo(spec, history, rng):
    samples = np.asarray([features(spec, trial["point"]) for trial in history])
    options = spec.options
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF([1.0] * samples.shape[1], (1e-2, 1e2)),
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=spec.seed,
    )
    gp.fit(samples, [score(spec, trial) for trial in history])
    kappa = max(options.kappa_min, options.kappa * options.kappa_decay ** len(history))

    def acquisition(point):
        vector = np.asarray(features(spec, point)).reshape(1, -1)
        mean, std = gp.predict(vector, return_std=True)
        distances = np.linalg.norm(samples - vector, axis=1)
        penalty = np.prod(np.tanh(distances / options.penalization_radius))
        return -float(mean[0] + kappa * std[0] * penalty)

    return optimize_acquisition(
        acquisition, len(spec.parameters), rng.random, restarts=10
    )


def bohb(spec, history, rng):
    """Fit good/bad KDEs at the largest resource with sufficient observations.

    Singular or sparse populations explicitly fall back to random proposals,
    as in the source. Never mix scores from unequal training budgets.
    """
    dimensions = len(spec.parameters)
    minimum = max(spec.options.min_points_in_model, dimensions + 1)
    for resource in sorted({t["resource"] for t in history}, reverse=True):
        group = sorted(
            (t for t in history if t["resource"] == resource),
            key=lambda t: score(spec, t),
            reverse=True,
        )
        n_good = max(minimum, int(len(group) * spec.options.top_fraction))
        if len(group) - n_good < minimum:
            continue
        try:
            good = gaussian_kde(np.asarray([t["point"] for t in group[:n_good]]).T)
            bad = gaussian_kde(np.asarray([t["point"] for t in group[n_good:]]).T)
            candidates = np.clip(
                good.resample(spec.options.kde_samples, seed=rng).T, 0, 1
            )
            ratios = good.logpdf(candidates.T) - bad.logpdf(candidates.T)
            if np.isfinite(ratios).all():
                return candidates[int(np.argmax(ratios))], "kde"
        except (ValueError, np.linalg.LinAlgError):
            continue
    return rng.random(dimensions), "random_warmup_or_singular_kde"


def dehb(spec, history, rng):
    """DE/rand/1 plus binomial crossover from same-resource successful members."""
    dimensions = len(spec.parameters)
    if len(history) < 4:
        return rng.random(dimensions), "random_warmup"
    selected = rng.choice(len(history), size=4, replace=False)
    target, a, b, c = [np.asarray(history[int(i)]["point"]) for i in selected]
    mutant = np.clip(a + spec.options.mutation_factor * (b - c), 0, 1)
    mask = rng.random(dimensions) < spec.options.crossover_probability
    mask[int(rng.integers(dimensions))] = True
    return np.where(mask, mutant, target), "differential_evolution"
