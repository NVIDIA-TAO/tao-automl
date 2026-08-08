# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest

from tao_automl.brain.objective_acquisition import (
    constrained_latency_ei,
    default_calibration_points,
    parego_utilities,
    parego_weights,
    probability_at_least,
    retained_accuracy_threshold,
    valid_accuracy_observations,
    valid_objective_observations,
)
from tao_automl.types import JobStates, Recommendation


def _recommendation(identifier, accuracy, latency, status=JobStates.success):
    recommendation = Recommendation(identifier, {}, "mAP50")
    recommendation.objective_values = {
        "mAP50": accuracy,
        "latency": latency,
    }
    recommendation.status = status
    return recommendation


def test_valid_objective_observations_reject_failed_and_invalid():
    history = [
        _recommendation(0, 0.6, 10.0),
        _recommendation(1, 0.7, 11.0, status=JobStates.failure),
        _recommendation(2, math.nan, 9.0),
        _recommendation(3, True, 8.0),
        _recommendation(4, 0.8, 0.0),
    ]
    observations = valid_objective_observations(
        history,
        accuracy_metric="mAP50",
        latency_metric="latency",
    )
    assert [(item.candidate_id, item.accuracy, item.latency) for item in observations] == [
        ("0", 0.6, 10.0)
    ]


def test_valid_accuracy_observations_do_not_require_latency():
    valid_without_latency = _recommendation(0, 0.6, 10.0)
    valid_without_latency.objective_values.pop("latency")
    invalid_latency = _recommendation(1, 0.7, -1.0)
    failed = _recommendation(
        2,
        0.9,
        5.0,
        status=JobStates.failure,
    )

    accuracy = valid_accuracy_observations(
        [valid_without_latency, invalid_latency, failed],
        accuracy_metric="mAP50",
    )
    pairs = valid_objective_observations(
        [valid_without_latency, invalid_latency, failed],
        accuracy_metric="mAP50",
        latency_metric="latency",
    )

    assert [(item.candidate_id, item.accuracy) for item in accuracy] == [
        ("0", 0.6),
        ("1", 0.7),
    ]
    assert pairs == []


@pytest.mark.parametrize(
    ("dimension", "expected"),
    [(1, 4), (2, 4), (3, 6), (6, 12), (100, 12)],
)
def test_default_calibration_points_is_bounded(dimension, expected):
    assert default_calibration_points(dimension) == expected


def test_retained_accuracy_threshold_uses_best_observed_accuracy():
    assert retained_accuracy_threshold([0.4, 0.7, 0.6], 0.9) == pytest.approx(
        (0.7, 0.63)
    )


@pytest.mark.parametrize("values", [[], [0.0], [-1.0, 0.0]])
def test_retained_accuracy_threshold_requires_positive_reference(values):
    assert retained_accuracy_threshold(values, 0.9) is None


def test_probability_at_least_handles_zero_variance_deterministically():
    probability = probability_at_least(
        np.asarray([0.7, 0.5]),
        np.asarray([0.0, 0.0]),
        0.6,
    )
    assert probability.tolist() == [1.0, 0.0]


def test_constrained_latency_ei_rejects_fast_but_infeasible_prediction():
    acquisition = constrained_latency_ei(
        latency_mean=np.asarray([5.0, 7.0]),
        latency_stddev=np.asarray([0.1, 0.1]),
        accuracy_mean=np.asarray([0.1, 0.7]),
        accuracy_stddev=np.asarray([0.01, 0.01]),
        accuracy_threshold=0.6,
        feasible_latency_incumbent=8.0,
    )
    assert acquisition[1] > acquisition[0]


def test_parego_weight_schedule_is_deterministic_and_varied():
    weights = [parego_weights(index) for index in range(4)]
    assert weights == [
        (0.5, 0.5),
        (0.25, 0.75),
        (0.75, 0.25),
        (0.125, 0.875),
    ]


def test_parego_utilities_are_scale_invariant():
    accuracy = [0.9, 0.8, 0.7]
    latency = [30.0, 20.0, 10.0]
    original, original_audit = parego_utilities(
        accuracy,
        latency,
        iteration=0,
    )
    scaled, scaled_audit = parego_utilities(
        [value * 100.0 for value in accuracy],
        [value * 0.001 for value in latency],
        iteration=0,
    )
    assert scaled == pytest.approx(original)
    assert original_audit["weights"] == scaled_audit["weights"]


def test_parego_identical_objectives_do_not_divide_by_zero():
    utilities, audit = parego_utilities(
        [0.5, 0.5],
        [10.0, 10.0],
        iteration=2,
    )
    assert utilities.tolist() == [0.0, 0.0]
    assert audit["normalization_bounds"] == {
        "accuracy_min": 0.5,
        "accuracy_max": 0.5,
        "latency_min": 10.0,
        "latency_max": 10.0,
        "accuracy_nadir_source": "pareto_front",
        "latency_nadir_source": "pareto_front",
    }


def test_parego_single_dominating_point_keeps_dominated_points_worse():
    utilities, audit = parego_utilities(
        [0.9, 0.8, 0.7],
        [10.0, 12.0, 14.0],
        iteration=0,
    )

    assert utilities[0] == pytest.approx(0.0)
    assert utilities[0] > utilities[1] > utilities[2]
    assert audit["normalization_bounds"] == {
        "accuracy_min": 0.7,
        "accuracy_max": 0.9,
        "latency_min": 10.0,
        "latency_max": 14.0,
        "accuracy_nadir_source": "valid_archive_fallback",
        "latency_nadir_source": "valid_archive_fallback",
    }


def test_parego_collapsed_accuracy_uses_only_latency_regret():
    utilities, audit = parego_utilities(
        [0.9, 0.9],
        [10.0, 12.0],
        iteration=0,
    )

    assert utilities[0] == pytest.approx(0.0)
    assert utilities[1] < utilities[0]
    assert audit["normalization_bounds"]["accuracy_nadir_source"] == "pareto_front"
    assert audit["normalization_bounds"]["latency_nadir_source"] == (
        "valid_archive_fallback"
    )
