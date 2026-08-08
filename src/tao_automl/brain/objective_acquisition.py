# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure utilities for mode-specific objective-aware Bayesian acquisition.

Final archive selection lives in :mod:`tao_automl.selection`.  This module
instead defines the response surfaces and acquisition values used while a
search is running:

* accuracy mode models accuracy directly;
* latency mode models accuracy and latency independently and uses constrained
  expected improvement after a deterministic calibration stage;
* multi-objective mode uses deterministic ParEGO scalarizations of the
  observed Pareto geometry.

The functions are intentionally independent of the controller and sklearn so
their directionality, normalization, and edge cases can be tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import norm


_SUCCESS_STATES = frozenset({"success", "done"})


def _finite_float(value: Any) -> float | None:
    """Return a finite non-boolean float or ``None``."""
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


@dataclass(frozen=True)
class ObjectiveObservation:
    """One complete, successful accuracy/latency observation."""

    candidate_id: str
    accuracy: float
    latency: float


@dataclass(frozen=True)
class AccuracyObservation:
    """One successful finite accuracy observation."""

    candidate_id: str
    accuracy: float


def valid_accuracy_observations(
    recommendations: Iterable[Any],
    *,
    accuracy_metric: str,
) -> list[AccuracyObservation]:
    """Extract finite accuracy without requiring a latency measurement."""
    observations: list[AccuracyObservation] = []
    for recommendation in recommendations:
        status = str(getattr(recommendation, "status", "")).lower()
        values = getattr(recommendation, "objective_values", None)
        if status not in _SUCCESS_STATES or not isinstance(values, Mapping):
            continue
        accuracy = _finite_float(values.get(accuracy_metric))
        if accuracy is None:
            continue
        observations.append(
            AccuracyObservation(
                candidate_id=str(getattr(recommendation, "id", "")),
                accuracy=accuracy,
            )
        )
    return observations


def valid_objective_observations(
    recommendations: Iterable[Any],
    *,
    accuracy_metric: str,
    latency_metric: str,
) -> list[ObjectiveObservation]:
    """Extract complete finite accuracy/positive-latency objective pairs."""
    observations: list[ObjectiveObservation] = []
    for recommendation in recommendations:
        status = str(getattr(recommendation, "status", "")).lower()
        values = getattr(recommendation, "objective_values", None)
        if status not in _SUCCESS_STATES or not isinstance(values, Mapping):
            continue
        accuracy = _finite_float(values.get(accuracy_metric))
        latency = _finite_float(values.get(latency_metric))
        if accuracy is None or latency is None or latency <= 0.0:
            continue
        observations.append(
            ObjectiveObservation(
                candidate_id=str(getattr(recommendation, "id", "")),
                accuracy=accuracy,
                latency=latency,
            )
        )
    return observations


def default_calibration_points(dimension: int) -> int:
    """Return the deterministic initial-design size for native objective search.

    Two points per dimension gives the surrogate at least minimal geometric
    coverage.  The cap prevents calibration from consuming an entire
    moderate-budget campaign, while the floor avoids treating two coincident
    observations as a calibrated quality envelope.
    """
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise TypeError("search-space dimension must be an integer")
    if dimension <= 0:
        raise ValueError("search-space dimension must be positive")
    return min(12, max(4, 2 * dimension))


def retained_accuracy_threshold(
    accuracies: Sequence[float],
    retained_fraction: float,
) -> tuple[float, float] | None:
    """Return ``(best_observed_accuracy, threshold)`` for latency acquisition.

    A relative threshold is not informative until a positive reference exists.
    Returning ``None`` keeps acquisition in quality-discovery mode instead of
    treating a zero/negative archive as evidence that degenerate candidates are
    feasible.
    """
    fraction = _finite_float(retained_fraction)
    if fraction is None or not 0.0 < fraction <= 1.0:
        raise ValueError("retained accuracy fraction must be finite and in (0, 1]")
    finite = [_finite_float(value) for value in accuracies]
    valid = [value for value in finite if value is not None]
    if not valid:
        return None
    reference = max(valid)
    if reference <= 0.0:
        return None
    return reference, reference * fraction


def expected_improvement_minimize(
    mean: np.ndarray,
    stddev: np.ndarray,
    *,
    incumbent: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected improvement for a minimized objective."""
    mean = np.asarray(mean, dtype=float)
    stddev = np.asarray(stddev, dtype=float)
    if mean.shape != stddev.shape:
        raise ValueError("mean and stddev must have the same shape")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(stddev)):
        raise ValueError("mean and stddev must be finite")
    if np.any(stddev < 0.0):
        raise ValueError("stddev cannot be negative")
    incumbent_value = _finite_float(incumbent)
    xi_value = _finite_float(xi)
    if incumbent_value is None or xi_value is None or xi_value < 0.0:
        raise ValueError("incumbent and non-negative xi must be finite")

    improvement = incumbent_value - mean - xi_value
    result = np.zeros_like(mean, dtype=float)
    positive_sigma = stddev > 0.0
    if np.any(positive_sigma):
        z_score = improvement[positive_sigma] / stddev[positive_sigma]
        result[positive_sigma] = (
            improvement[positive_sigma] * norm.cdf(z_score)
            + stddev[positive_sigma] * norm.pdf(z_score)
        )
    result[~positive_sigma] = np.maximum(
        improvement[~positive_sigma],
        0.0,
    )
    return np.maximum(result, 0.0)


def expected_improvement_maximize(
    mean: np.ndarray,
    stddev: np.ndarray,
    *,
    incumbent: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected improvement for a maximized objective."""
    return expected_improvement_minimize(
        -np.asarray(mean, dtype=float),
        np.asarray(stddev, dtype=float),
        incumbent=-float(incumbent),
        xi=xi,
    )


def probability_at_least(
    mean: np.ndarray,
    stddev: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return Gaussian probability that a maximized value meets a threshold."""
    mean = np.asarray(mean, dtype=float)
    stddev = np.asarray(stddev, dtype=float)
    threshold_value = _finite_float(threshold)
    if mean.shape != stddev.shape:
        raise ValueError("mean and stddev must have the same shape")
    if (
        threshold_value is None
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(stddev))
        or np.any(stddev < 0.0)
    ):
        raise ValueError("probability inputs must be finite with non-negative stddev")
    probability = np.empty_like(mean, dtype=float)
    positive_sigma = stddev > 0.0
    probability[positive_sigma] = norm.cdf(
        (mean[positive_sigma] - threshold_value) / stddev[positive_sigma]
    )
    probability[~positive_sigma] = (
        mean[~positive_sigma] >= threshold_value
    ).astype(float)
    return probability


def constrained_latency_ei(
    latency_mean: np.ndarray,
    latency_stddev: np.ndarray,
    accuracy_mean: np.ndarray,
    accuracy_stddev: np.ndarray,
    *,
    accuracy_threshold: float,
    feasible_latency_incumbent: float | None,
    xi: float = 0.01,
) -> np.ndarray:
    """Constrained EI for latency with an independently modelled quality gate.

    Before a feasible latency incumbent exists, acquisition maximizes the
    probability of crossing the quality boundary.  Once one exists, latency
    expected improvement is multiplied by that probability.
    """
    feasibility = probability_at_least(
        accuracy_mean,
        accuracy_stddev,
        accuracy_threshold,
    )
    if feasible_latency_incumbent is None:
        return feasibility
    latency_ei = expected_improvement_minimize(
        latency_mean,
        latency_stddev,
        incumbent=feasible_latency_incumbent,
        xi=xi,
    )
    return latency_ei * feasibility


def _nondominated_mask(accuracy: np.ndarray, latency: np.ndarray) -> np.ndarray:
    """Return the rank-zero mask for maximize-accuracy/minimize-latency data."""
    count = len(accuracy)
    mask = np.ones(count, dtype=bool)
    for index in range(count):
        for other in range(count):
            if index == other:
                continue
            if (
                accuracy[other] >= accuracy[index]
                and latency[other] <= latency[index]
                and (
                    accuracy[other] > accuracy[index]
                    or latency[other] < latency[index]
                )
            ):
                mask[index] = False
                break
    return mask


def pareto_regrets(
    accuracies: Sequence[float],
    latencies: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return front-relative accuracy and latency regrets.

    Ideals and ordinary nadirs come from the observed rank-zero front. If a
    front has zero range in one dimension (especially a one-point front), the
    nadir for that dimension falls back to the complete valid archive. This
    preserves evidence that a dominated point is worse instead of assigning
    every point zero regret. A dimension identical across the complete archive
    still maps to zero and never divides by zero.
    """
    accuracy = np.asarray(accuracies, dtype=float)
    latency = np.asarray(latencies, dtype=float)
    if accuracy.ndim != 1 or latency.ndim != 1 or accuracy.shape != latency.shape:
        raise ValueError("accuracy and latency must be equal-length vectors")
    if not len(accuracy):
        raise ValueError("at least one objective pair is required")
    if not np.all(np.isfinite(accuracy)) or not np.all(np.isfinite(latency)):
        raise ValueError("objective values must be finite")

    front = _nondominated_mask(accuracy, latency)
    front_accuracy = accuracy[front]
    front_latency = latency[front]
    front_accuracy_min = float(np.min(front_accuracy))
    accuracy_max = float(np.max(front_accuracy))
    latency_min = float(np.min(front_latency))
    front_latency_max = float(np.max(front_latency))
    accuracy_min = (
        front_accuracy_min
        if accuracy_max > front_accuracy_min
        else float(np.min(accuracy))
    )
    latency_max = (
        front_latency_max
        if front_latency_max > latency_min
        else float(np.max(latency))
    )
    accuracy_span = accuracy_max - accuracy_min
    latency_span = latency_max - latency_min
    accuracy_regret = (
        np.zeros_like(accuracy)
        if accuracy_span <= 0.0
        else (accuracy_max - accuracy) / accuracy_span
    )
    latency_regret = (
        np.zeros_like(latency)
        if latency_span <= 0.0
        else (latency - latency_min) / latency_span
    )
    return (
        accuracy_regret,
        latency_regret,
        {
            "accuracy_min": accuracy_min,
            "accuracy_max": accuracy_max,
            "latency_min": latency_min,
            "latency_max": latency_max,
            "accuracy_nadir_source": (
                "pareto_front"
                if accuracy_min == front_accuracy_min
                else "valid_archive_fallback"
            ),
            "latency_nadir_source": (
                "pareto_front"
                if latency_max == front_latency_max
                else "valid_archive_fallback"
            ),
        },
    )


def van_der_corput(index: int, base: int = 2) -> float:
    """Return a deterministic low-discrepancy value in ``[0, 1)``."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("index must be a non-negative integer")
    if isinstance(base, bool) or not isinstance(base, int) or base < 2:
        raise ValueError("base must be an integer >= 2")
    value = 0.0
    denominator = 1.0
    n = index
    while n:
        n, remainder = divmod(n, base)
        denominator *= base
        value += remainder / denominator
    return value


def parego_weights(iteration: int) -> tuple[float, float]:
    """Return deterministic two-objective ParEGO weights.

    The sequence begins at the balanced weight and then fills the simplex in
    low-discrepancy order: ``0.5, 0.25, 0.75, 0.125, ...``.
    """
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("iteration must be a non-negative integer")
    accuracy_weight = van_der_corput(iteration + 1, base=2)
    return accuracy_weight, 1.0 - accuracy_weight


def parego_utilities(
    accuracies: Sequence[float],
    latencies: Sequence[float],
    *,
    iteration: int,
    augmentation_rho: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return maximize-oriented augmented-Chebyshev ParEGO utilities."""
    rho = _finite_float(augmentation_rho)
    if rho is None or rho < 0.0:
        raise ValueError("augmentation_rho must be finite and non-negative")
    accuracy_regret, latency_regret, bounds = pareto_regrets(
        accuracies,
        latencies,
    )
    accuracy_weight, latency_weight = parego_weights(iteration)
    weighted_accuracy = accuracy_weight * accuracy_regret
    weighted_latency = latency_weight * latency_regret
    regret = (
        np.maximum(weighted_accuracy, weighted_latency)
        + rho * (weighted_accuracy + weighted_latency)
    )
    audit = {
        "method": "parego_augmented_chebyshev",
        "iteration": iteration,
        "weights": {
            "accuracy": accuracy_weight,
            "latency": latency_weight,
        },
        "normalization_bounds": bounds,
        "augmentation_rho": rho,
    }
    return -regret, audit
