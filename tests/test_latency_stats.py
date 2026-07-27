# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for deterministic aggregation of synchronized latency samples."""

from dataclasses import replace

import numpy as np
import pytest

from tao_automl.latency_stats import (
    LatencyProtocol,
    LatencyValidityThresholds,
    aggregate_synchronized_latency,
    compare_latency,
)


def _protocol(**overrides):
    defaults = {
        "warmup_iterations": 3,
        "timed_iterations": 3,
        "repeated_rounds": 2,
        "bootstrap_resamples": 500,
        "bootstrap_seed": 17,
    }
    defaults.update(overrides)
    return LatencyProtocol(**defaults)


def _samples():
    return {
        0: {
            "cuda:0": [1.0, 2.0, 3.0],
            "cuda:1": [2.0, 1.0, 4.0],
        },
        1: {
            "cuda:0": [2.0, 3.0, 4.0],
            "cuda:1": [1.0, 5.0, 3.0],
        },
    }


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("warmup_iterations", -1, ValueError),
        ("warmup_iterations", 1.5, TypeError),
        ("timed_iterations", 0, ValueError),
        ("timed_iterations", True, TypeError),
        ("repeated_rounds", 0, ValueError),
        ("tail_percentile", 49.9, ValueError),
        ("tail_percentile", float("nan"), ValueError),
        ("bootstrap_resamples", 0, ValueError),
        ("bootstrap_confidence_level", 1.0, ValueError),
        ("bootstrap_seed", -1, ValueError),
    ],
)
def test_protocol_rejects_invalid_values(field, value, error):
    values = {
        "warmup_iterations": 3,
        "timed_iterations": 3,
        "repeated_rounds": 2,
    }
    values[field] = value

    with pytest.raises(error):
        LatencyProtocol(**values)


def test_protocol_rejects_invalid_expected_devices_and_thresholds():
    with pytest.raises(ValueError, match="duplicates"):
        _protocol(expected_devices=("cuda:0", "cuda:0"))
    with pytest.raises(TypeError, match="tuple"):
        _protocol(expected_devices=["cuda:0"])
    with pytest.raises(ValueError, match="finite non-negative"):
        LatencyValidityThresholds(max_robust_cv=-0.1)


def test_aggregate_uses_median_of_device_round_medians_as_primary_latency():
    statistics = aggregate_synchronized_latency(_samples(), _protocol())

    # Device-round medians are [2, 2] and [3, 3].  The slowest-device
    # synchronized samples [2, 2, 4] and [2, 5, 4] are diagnostics only.
    assert statistics.per_device_sample_count == 12
    assert statistics.device_round_cluster_count == 4
    assert statistics.synchronized_sample_count == 6
    assert statistics.median_ms == pytest.approx(2.5)
    assert statistics.tail_percentile == 95.0
    assert statistics.tail_latency_ms == pytest.approx(4.45)
    assert statistics.mad_ms == pytest.approx(1.0)
    assert statistics.iqr_ms == pytest.approx(1.5)
    assert statistics.robust_cv == pytest.approx(1.4826 / 2.5)
    assert statistics.synchronized_median_ms == pytest.approx(3.0)
    assert statistics.synchronized_tail_latency_ms == pytest.approx(4.75)
    assert statistics.per_round_median_ms == ((0, 2.0), (1, 3.0))
    assert statistics.per_device_median_ms == (
        ("cuda:0", 2.5),
        ("cuda:1", 2.5),
    )
    assert statistics.round_median_range_ms == pytest.approx(1.0)
    assert statistics.round_median_range_fraction == pytest.approx(1.0 / 2.5)
    assert statistics.round_drift_ms == pytest.approx(1.0)
    assert statistics.round_drift_fraction == pytest.approx(1.0 / 2.5)
    assert statistics.device_median_range_ms == pytest.approx(0.0)
    assert statistics.per_device_round_range_ms == (
        ("cuda:0", 1.0),
        ("cuda:1", 1.0),
    )
    assert statistics.per_device_round_drift_ms == (
        ("cuda:0", 1.0),
        ("cuda:1", 1.0),
    )
    assert statistics.is_valid
    assert statistics.invalid_reasons == ()
    assert statistics.validity_reason == "valid"


def test_primary_median_weights_device_round_clusters_equally():
    samples = {
        0: {
            "cuda:0": [1.0, 100.0, 100.0],
            "cuda:1": [2.0, 2.0, 200.0],
        },
        1: {
            "cuda:0": [3.0, 3.0, 300.0],
            "cuda:1": [4.0, 4.0, 400.0],
        },
    }

    statistics = aggregate_synchronized_latency(samples, _protocol())

    # The four cluster medians are [100, 2, 3, 4], with median 3.5.
    # Pooling all raw observations would incorrectly produce 4.0.
    assert statistics.median_ms == pytest.approx(3.5)
    assert statistics.median_ms != pytest.approx(
        np.median(np.asarray(list(samples[0].values()) + list(samples[1].values())))
    )


def test_single_device_single_round_has_zero_repeatability_spread():
    protocol = _protocol(
        timed_iterations=3,
        repeated_rounds=1,
        bootstrap_resamples=20,
    )

    statistics = aggregate_synchronized_latency(
        {7: {"cuda:0": [2.0, 2.0, 2.0]}},
        protocol,
    )

    assert statistics.median_ms == pytest.approx(2.0)
    assert statistics.round_median_range_ms == 0.0
    assert statistics.round_drift_ms == 0.0
    assert statistics.device_median_range_ms == 0.0
    assert statistics.bootstrap_median_ci_ms == (2.0, 2.0)
    assert statistics.is_valid


def test_aggregate_is_deterministic_and_independent_of_mapping_order():
    protocol = _protocol(bootstrap_resamples=1000, bootstrap_seed=1234)
    forward = _samples()
    reordered = {
        1: {
            "cuda:1": forward[1]["cuda:1"],
            "cuda:0": forward[1]["cuda:0"],
        },
        0: {
            "cuda:1": forward[0]["cuda:1"],
            "cuda:0": forward[0]["cuda:0"],
        },
    }

    first = aggregate_synchronized_latency(forward, protocol)
    repeated = aggregate_synchronized_latency(forward, protocol)
    permuted = aggregate_synchronized_latency(reordered, protocol)

    assert first == repeated
    assert first == permuted
    assert first.device_ids == ("cuda:0", "cuda:1")
    assert first.bootstrap_median_ci_ms[0] <= first.median_ms
    assert first.bootstrap_median_ci_ms[1] >= first.median_ms


def test_bootstrap_seed_is_explicit_and_reproducible():
    protocol = _protocol(bootstrap_resamples=31, bootstrap_seed=9)
    first = aggregate_synchronized_latency(_samples(), protocol)
    second = aggregate_synchronized_latency(_samples(), replace(protocol))

    assert first.bootstrap_median_ci_ms == second.bootstrap_median_ci_ms
    assert first.protocol.bootstrap_seed == 9


def test_configured_quality_thresholds_produce_stable_invalid_reasons():
    samples = _samples()
    samples[1]["cuda:1"] = [1.0, 5.0, 6.0]
    thresholds = LatencyValidityThresholds(
        max_robust_cv=0.0,
        max_round_median_range_fraction=0.0,
        max_absolute_round_drift_fraction=0.0,
        max_device_median_range_fraction=0.0,
        max_bootstrap_ci_width_fraction=0.0,
    )

    statistics = aggregate_synchronized_latency(
        samples,
        _protocol(validity_thresholds=thresholds),
    )

    assert not statistics.is_valid
    assert statistics.invalid_reasons == (
        "robust_cv_exceeds_threshold",
        "round_median_range_exceeds_threshold",
        "absolute_round_drift_exceeds_threshold",
        "device_median_range_exceeds_threshold",
        "bootstrap_ci_width_exceeds_threshold",
    )
    assert statistics.validity_reason == ", ".join(statistics.invalid_reasons)


def test_threshold_boundary_is_valid():
    unbounded = aggregate_synchronized_latency(_samples(), _protocol())
    thresholds = LatencyValidityThresholds(
        max_robust_cv=unbounded.robust_cv,
        max_round_median_range_fraction=unbounded.round_median_range_fraction,
        max_absolute_round_drift_fraction=abs(
            unbounded.round_drift_fraction
        ),
        max_device_median_range_fraction=unbounded.device_median_range_fraction,
        max_bootstrap_ci_width_fraction=unbounded.bootstrap_ci_width_fraction,
    )

    bounded = aggregate_synchronized_latency(
        _samples(),
        _protocol(validity_thresholds=thresholds),
    )

    assert bounded.is_valid


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        ({0: {"cuda:0": [1.0, 2.0, 3.0]}}, "1 rounds; expected 2"),
        (
            {
                0: {"cuda:0": [1.0, 2.0, 3.0]},
                1: {"cuda:1": [1.0, 2.0, 3.0]},
            },
            "same device",
        ),
        (
            {
                0: {"cuda:0": [1.0, 2.0]},
                1: {"cuda:0": [1.0, 2.0, 3.0]},
            },
            "contains 2 timed samples",
        ),
        (
            {
                0: {"cuda:0": [1.0, float("nan"), 3.0]},
                1: {"cuda:0": [1.0, 2.0, 3.0]},
            },
            "finite and > 0",
        ),
        (
            {
                0: {"cuda:0": [1.0, 0.0, 3.0]},
                1: {"cuda:0": [1.0, 2.0, 3.0]},
            },
            "finite and > 0",
        ),
    ],
)
def test_aggregate_rejects_incomplete_or_invalid_samples(samples, message):
    with pytest.raises(ValueError, match=message):
        aggregate_synchronized_latency(samples, _protocol())


def test_aggregate_rejects_non_numeric_and_boolean_samples():
    samples = _samples()
    samples[0]["cuda:0"][0] = "1.0"
    with pytest.raises(TypeError, match="must be numeric"):
        aggregate_synchronized_latency(samples, _protocol())

    samples = _samples()
    samples[0]["cuda:0"][0] = True
    with pytest.raises(TypeError, match="must be numeric"):
        aggregate_synchronized_latency(samples, _protocol())


def test_expected_devices_are_enforced():
    protocol = _protocol(expected_devices=("cuda:0", "cuda:2"))

    with pytest.raises(ValueError, match="do not match expected_devices"):
        aggregate_synchronized_latency(_samples(), protocol)


def _constant_statistics(
    first_round,
    second_round,
    *,
    validity_thresholds=None,
):
    protocol = _protocol(
        timed_iterations=len(first_round),
        bootstrap_resamples=1000,
        validity_thresholds=(
            validity_thresholds
            if validity_thresholds is not None
            else LatencyValidityThresholds()
        ),
    )
    return aggregate_synchronized_latency(
        {
            0: {"cuda:0": first_round},
            1: {"cuda:0": second_round},
        },
        protocol,
    )


def test_compare_latency_uses_absolute_or_relative_tolerance():
    first = _constant_statistics([10.0] * 3, [10.0] * 3)
    second = _constant_statistics([10.4] * 3, [10.4] * 3)

    absolute = compare_latency(
        first,
        second,
        absolute_tolerance_ms=0.5,
        use_confidence_interval=False,
    )
    relative = compare_latency(
        first,
        second,
        relative_tolerance=0.05,
        use_confidence_interval=False,
    )

    assert absolute.equivalent
    assert absolute.within_tolerance
    assert absolute.tolerance_ms == pytest.approx(0.5)
    assert absolute.reason == "within_tolerance"
    assert relative.equivalent
    assert relative.tolerance_ms == pytest.approx(0.52)


def test_compare_latency_can_treat_overlapping_confidence_intervals_as_equivalent():
    first = _constant_statistics([1.0] * 3, [2.0] * 3)
    second = _constant_statistics([1.5] * 3, [2.5] * 3)

    with_ci = compare_latency(first, second)
    without_ci = compare_latency(
        first,
        second,
        use_confidence_interval=False,
    )

    assert not with_ci.within_tolerance
    assert with_ci.confidence_intervals_overlap
    assert with_ci.equivalent
    assert with_ci.reason == "confidence_intervals_overlap"
    assert not without_ci.equivalent
    assert without_ci.reason == "meaningfully_different"


def test_compare_latency_rejects_invalid_statistics_by_default():
    noisy = _constant_statistics(
        [1.0, 2.0, 3.0],
        [3.0, 4.0, 5.0],
        validity_thresholds=LatencyValidityThresholds(max_robust_cv=0.0),
    )
    stable = _constant_statistics([2.0] * 3, [2.0] * 3)

    assert not noisy.is_valid
    with pytest.raises(ValueError, match="invalid latency measurements"):
        compare_latency(noisy, stable)

    comparison = compare_latency(noisy, stable, require_valid=False)
    assert isinstance(comparison.equivalent, bool)


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("absolute_tolerance_ms", -0.1, ValueError),
        ("absolute_tolerance_ms", float("nan"), ValueError),
        ("relative_tolerance", True, TypeError),
    ],
)
def test_compare_latency_rejects_invalid_tolerances(name, value, error):
    first = _constant_statistics([1.0] * 3, [1.0] * 3)

    with pytest.raises(error):
        compare_latency(first, first, **{name: value})


def test_numpy_real_samples_are_accepted():
    samples = {
        0: {"cuda:0": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
        1: {"cuda:0": np.array([2.0, 3.0, 4.0], dtype=np.float64)},
    }

    statistics = aggregate_synchronized_latency(samples, _protocol())

    assert statistics.median_ms == pytest.approx(2.5)
