# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic statistics for repeated, multi-device latency measurements.

The utility is deliberately independent of any model or benchmark launcher.  A
caller supplies timed samples as ``round -> device -> samples``.  Samples at the
same position within a round may be synchronized.  Devices are treated as
independent replicas for the primary latency estimate.  Slowest-device
synchronized latency is retained as a secondary distributed-step diagnostic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from numbers import Real
from typing import TypeAlias

import numpy as np


LatencySamples: TypeAlias = Mapping[int, Mapping[str, Sequence[Real]]]


def _validate_optional_threshold(name: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative number or None")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number or None")


@dataclass(frozen=True, slots=True)
class LatencyValidityThresholds:
    """Optional measurement-quality limits.

    Fractions are relative to the aggregate median.  ``None`` disables a
    threshold so product policy can choose limits without this low-level
    utility silently imposing hardware-specific assumptions.
    """

    max_robust_cv: float | None = None
    max_round_median_range_fraction: float | None = None
    max_absolute_round_drift_fraction: float | None = None
    max_device_median_range_fraction: float | None = None
    max_bootstrap_ci_width_fraction: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_robust_cv",
            "max_round_median_range_fraction",
            "max_absolute_round_drift_fraction",
            "max_device_median_range_fraction",
            "max_bootstrap_ci_width_fraction",
        ):
            _validate_optional_threshold(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class LatencyProtocol:
    """Configuration required to validate and summarize timed measurements."""

    warmup_iterations: int
    timed_iterations: int
    repeated_rounds: int
    tail_percentile: float = 95.0
    bootstrap_resamples: int = 2000
    bootstrap_confidence_level: float = 0.95
    bootstrap_seed: int = 0
    expected_devices: tuple[str, ...] = ()
    validity_thresholds: LatencyValidityThresholds = field(
        default_factory=LatencyValidityThresholds
    )

    def __post_init__(self) -> None:
        for name, minimum in (
            ("warmup_iterations", 0),
            ("timed_iterations", 1),
            ("repeated_rounds", 1),
            ("bootstrap_resamples", 1),
            ("bootstrap_seed", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < minimum:
                raise ValueError(f"{name} must be >= {minimum}")

        if (
            isinstance(self.tail_percentile, bool)
            or not isinstance(self.tail_percentile, Real)
        ):
            raise TypeError("tail_percentile must be a finite number")
        if not math.isfinite(float(self.tail_percentile)) or not (
            50.0 <= float(self.tail_percentile) <= 100.0
        ):
            raise ValueError("tail_percentile must be in [50, 100]")

        if (
            isinstance(self.bootstrap_confidence_level, bool)
            or not isinstance(self.bootstrap_confidence_level, Real)
        ):
            raise TypeError("bootstrap_confidence_level must be a finite number")
        if not math.isfinite(float(self.bootstrap_confidence_level)) or not (
            0.0 < float(self.bootstrap_confidence_level) < 1.0
        ):
            raise ValueError("bootstrap_confidence_level must be in (0, 1)")

        if not isinstance(self.expected_devices, tuple):
            raise TypeError("expected_devices must be a tuple of device identifiers")
        if any(
            not isinstance(device, str) or not device
            for device in self.expected_devices
        ):
            raise ValueError(
                "expected_devices must contain non-empty string identifiers"
            )
        if len(set(self.expected_devices)) != len(self.expected_devices):
            raise ValueError("expected_devices must not contain duplicates")
        if not isinstance(self.validity_thresholds, LatencyValidityThresholds):
            raise TypeError(
                "validity_thresholds must be a LatencyValidityThresholds instance"
            )


@dataclass(frozen=True, slots=True)
class LatencyStatistics:
    """Aggregate latency and repeatability diagnostics.

    ``raw_sample_count_total`` and ``samples_per_device`` are the unambiguous
    sample-count fields. ``per_device_sample_count`` is retained as a
    backward-compatible alias for the historical serialized field, whose
    value was the total sample count despite its name.
    """

    protocol: LatencyProtocol
    device_ids: tuple[str, ...]
    per_device_sample_count: int
    device_round_cluster_count: int
    synchronized_sample_count: int
    median_ms: float
    tail_percentile: float
    tail_latency_ms: float
    mad_ms: float
    iqr_ms: float
    robust_cv: float
    synchronized_median_ms: float
    synchronized_tail_latency_ms: float
    per_round_median_ms: tuple[tuple[int, float], ...]
    per_device_median_ms: tuple[tuple[str, float], ...]
    round_median_range_ms: float
    round_median_range_fraction: float
    round_drift_ms: float
    round_drift_fraction: float
    device_median_range_ms: float
    device_median_range_fraction: float
    per_device_round_range_ms: tuple[tuple[str, float], ...]
    per_device_round_drift_ms: tuple[tuple[str, float], ...]
    bootstrap_median_ci_ms: tuple[float, float]
    bootstrap_ci_width_ms: float
    bootstrap_ci_width_fraction: float
    is_valid: bool
    invalid_reasons: tuple[str, ...]
    # Derived fields use ``init=False`` so the historical constructor
    # signature remains valid for downstream callers.
    raw_sample_count_total: int = field(init=False)
    samples_per_device: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "raw_sample_count_total",
            int(self.per_device_sample_count),
        )
        object.__setattr__(
            self,
            "samples_per_device",
            int(
                self.protocol.repeated_rounds
                * self.protocol.timed_iterations
            ),
        )

    @property
    def validity_reason(self) -> str:
        """Return a concise, stable reason suitable for logs."""

        if self.is_valid:
            return "valid"
        return ", ".join(self.invalid_reasons)


@dataclass(frozen=True, slots=True)
class LatencyComparison:
    """Deterministic equivalence decision for two latency measurements."""

    equivalent: bool
    median_difference_ms: float
    tolerance_ms: float
    within_tolerance: bool
    confidence_intervals_overlap: bool
    reason: str


def _validated_array(
    samples: LatencySamples,
    protocol: LatencyProtocol,
) -> tuple[tuple[int, ...], tuple[str, ...], np.ndarray]:
    if not isinstance(samples, Mapping):
        raise TypeError("samples must be a round-to-device mapping")
    if len(samples) != protocol.repeated_rounds:
        raise ValueError(
            "samples contain "
            f"{len(samples)} rounds; expected {protocol.repeated_rounds}"
        )

    round_ids = tuple(samples)
    if any(isinstance(round_id, bool) or not isinstance(round_id, int) for round_id in round_ids):
        raise TypeError("round identifiers must be integers")
    round_ids = tuple(sorted(round_ids))

    discovered_devices: set[str] | None = None
    for round_id in round_ids:
        device_samples = samples[round_id]
        if not isinstance(device_samples, Mapping) or not device_samples:
            raise ValueError(f"round {round_id} must contain at least one device")
        devices = set(device_samples)
        if any(not isinstance(device, str) or not device for device in devices):
            raise ValueError("device identifiers must be non-empty strings")
        if discovered_devices is None:
            discovered_devices = devices
        elif devices != discovered_devices:
            raise ValueError("every round must contain the same device identifiers")

    assert discovered_devices is not None
    if protocol.expected_devices:
        expected_devices = set(protocol.expected_devices)
        if discovered_devices != expected_devices:
            raise ValueError(
                "sample devices do not match expected_devices: "
                f"got {sorted(discovered_devices)!r}, "
                f"expected {sorted(expected_devices)!r}"
            )
    device_ids = tuple(sorted(discovered_devices))

    values = np.empty(
        (len(round_ids), len(device_ids), protocol.timed_iterations),
        dtype=np.float64,
    )
    for round_position, round_id in enumerate(round_ids):
        for device_position, device_id in enumerate(device_ids):
            device_values = samples[round_id][device_id]
            try:
                count = len(device_values)
            except TypeError as error:
                raise TypeError(
                    f"samples for round {round_id}, device {device_id!r} "
                    "must be a sized sequence"
                ) from error
            if count != protocol.timed_iterations:
                raise ValueError(
                    f"round {round_id}, device {device_id!r} contains {count} "
                    f"timed samples; expected {protocol.timed_iterations}"
                )
            for iteration, value in enumerate(device_values):
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError(
                        f"latency at round {round_id}, device {device_id!r}, "
                        f"iteration {iteration} must be numeric"
                    )
                numeric_value = float(value)
                if not math.isfinite(numeric_value) or numeric_value <= 0.0:
                    raise ValueError(
                        f"latency at round {round_id}, device {device_id!r}, "
                        f"iteration {iteration} must be finite and > 0"
                    )
                values[round_position, device_position, iteration] = numeric_value

    return round_ids, device_ids, values


def _cluster_bootstrap_median_ci(
    device_round_medians: np.ndarray,
    protocol: LatencyProtocol,
) -> tuple[float, float]:
    """Bootstrap the primary median by whole device-round clusters."""

    rng = np.random.default_rng(protocol.bootstrap_seed)
    cluster_medians = device_round_medians.reshape(-1)
    cluster_count = cluster_medians.size
    bootstrapped_medians = np.empty(protocol.bootstrap_resamples, dtype=np.float64)
    for index in range(protocol.bootstrap_resamples):
        sampled_clusters = rng.integers(0, cluster_count, size=cluster_count)
        bootstrapped_medians[index] = np.median(cluster_medians[sampled_clusters])

    alpha = 1.0 - float(protocol.bootstrap_confidence_level)
    lower, upper = np.quantile(
        bootstrapped_medians,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    return float(lower), float(upper)


def aggregate_synchronized_latency(
    samples: LatencySamples,
    protocol: LatencyProtocol,
) -> LatencyStatistics:
    """Validate and aggregate repeated multi-device latency samples.

    The primary estimator is the median of device-round medians, preventing a
    noisy device or round with more observations from receiving extra weight.
    Tail latency, MAD, and IQR are computed from all per-device timed samples.
    For distributed diagnostics, the slowest device at each synchronized
    iteration is summarized separately.
    """

    round_ids, device_ids, values = _validated_array(samples, protocol)
    device_round_medians = np.median(values, axis=2)
    cluster_medians = device_round_medians.reshape(-1)
    flattened_per_device = values.reshape(-1)
    synchronized = np.max(values, axis=1)
    flattened_synchronized = synchronized.reshape(-1)

    median_ms = float(np.median(cluster_medians))
    tail_latency_ms = float(
        np.percentile(
            flattened_per_device,
            float(protocol.tail_percentile),
            method="linear",
        )
    )
    mad_ms = float(np.median(np.abs(flattened_per_device - median_ms)))
    first_quartile, third_quartile = np.percentile(
        flattened_per_device, [25.0, 75.0], method="linear"
    )
    iqr_ms = float(third_quartile - first_quartile)
    robust_cv = float(1.4826 * mad_ms / median_ms)
    synchronized_median_ms = float(np.median(flattened_synchronized))
    synchronized_tail_latency_ms = float(
        np.percentile(
            flattened_synchronized,
            float(protocol.tail_percentile),
            method="linear",
        )
    )

    round_medians = np.median(device_round_medians, axis=1)
    per_round_median_ms = tuple(
        (round_id, float(round_medians[position]))
        for position, round_id in enumerate(round_ids)
    )
    round_median_range_ms = float(np.max(round_medians) - np.min(round_medians))
    round_drift_ms = float(round_medians[-1] - round_medians[0])

    device_medians = np.median(device_round_medians, axis=0)
    per_device_median_ms = tuple(
        (device_id, float(device_medians[position]))
        for position, device_id in enumerate(device_ids)
    )
    device_median_range_ms = float(np.max(device_medians) - np.min(device_medians))

    per_device_round_range_ms = tuple(
        (
            device_id,
            float(
                np.max(device_round_medians[:, position])
                - np.min(device_round_medians[:, position])
            ),
        )
        for position, device_id in enumerate(device_ids)
    )
    per_device_round_drift_ms = tuple(
        (
            device_id,
            float(
                device_round_medians[-1, position]
                - device_round_medians[0, position]
            ),
        )
        for position, device_id in enumerate(device_ids)
    )

    bootstrap_median_ci_ms = _cluster_bootstrap_median_ci(
        device_round_medians,
        protocol,
    )
    bootstrap_ci_width_ms = (
        bootstrap_median_ci_ms[1] - bootstrap_median_ci_ms[0]
    )

    round_median_range_fraction = round_median_range_ms / median_ms
    round_drift_fraction = round_drift_ms / median_ms
    device_median_range_fraction = device_median_range_ms / median_ms
    bootstrap_ci_width_fraction = bootstrap_ci_width_ms / median_ms

    thresholds = protocol.validity_thresholds
    invalid_reasons: list[str] = []
    if (
        thresholds.max_robust_cv is not None
        and robust_cv > thresholds.max_robust_cv
    ):
        invalid_reasons.append("robust_cv_exceeds_threshold")
    if (
        thresholds.max_round_median_range_fraction is not None
        and round_median_range_fraction
        > thresholds.max_round_median_range_fraction
    ):
        invalid_reasons.append("round_median_range_exceeds_threshold")
    if (
        thresholds.max_absolute_round_drift_fraction is not None
        and abs(round_drift_fraction)
        > thresholds.max_absolute_round_drift_fraction
    ):
        invalid_reasons.append("absolute_round_drift_exceeds_threshold")
    if (
        thresholds.max_device_median_range_fraction is not None
        and device_median_range_fraction
        > thresholds.max_device_median_range_fraction
    ):
        invalid_reasons.append("device_median_range_exceeds_threshold")
    if (
        thresholds.max_bootstrap_ci_width_fraction is not None
        and bootstrap_ci_width_fraction
        > thresholds.max_bootstrap_ci_width_fraction
    ):
        invalid_reasons.append("bootstrap_ci_width_exceeds_threshold")

    return LatencyStatistics(
        protocol=protocol,
        device_ids=device_ids,
        # Deprecated compatibility field.  Its historical value was the
        # total across devices, not the number for one device.
        per_device_sample_count=int(flattened_per_device.size),
        device_round_cluster_count=int(cluster_medians.size),
        synchronized_sample_count=int(flattened_synchronized.size),
        median_ms=median_ms,
        tail_percentile=float(protocol.tail_percentile),
        tail_latency_ms=tail_latency_ms,
        mad_ms=mad_ms,
        iqr_ms=iqr_ms,
        robust_cv=robust_cv,
        synchronized_median_ms=synchronized_median_ms,
        synchronized_tail_latency_ms=synchronized_tail_latency_ms,
        per_round_median_ms=per_round_median_ms,
        per_device_median_ms=per_device_median_ms,
        round_median_range_ms=round_median_range_ms,
        round_median_range_fraction=round_median_range_fraction,
        round_drift_ms=round_drift_ms,
        round_drift_fraction=round_drift_fraction,
        device_median_range_ms=device_median_range_ms,
        device_median_range_fraction=device_median_range_fraction,
        per_device_round_range_ms=per_device_round_range_ms,
        per_device_round_drift_ms=per_device_round_drift_ms,
        bootstrap_median_ci_ms=bootstrap_median_ci_ms,
        bootstrap_ci_width_ms=bootstrap_ci_width_ms,
        bootstrap_ci_width_fraction=bootstrap_ci_width_fraction,
        is_valid=not invalid_reasons,
        invalid_reasons=tuple(invalid_reasons),
    )


def compare_latency(
    first: LatencyStatistics,
    second: LatencyStatistics,
    *,
    absolute_tolerance_ms: float = 0.0,
    relative_tolerance: float = 0.0,
    use_confidence_interval: bool = True,
    require_valid: bool = True,
) -> LatencyComparison:
    """Compare measurements using a symmetric tolerance and bootstrap CIs.

    The point-estimate tolerance is the larger of the configured absolute
    tolerance and the relative tolerance times the larger median.  When
    ``use_confidence_interval`` is true, overlapping median confidence
    intervals also make the measurements equivalent.
    """

    for name, value in (
        ("absolute_tolerance_ms", absolute_tolerance_ms),
        ("relative_tolerance", relative_tolerance),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a finite non-negative number")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number")
    if not isinstance(use_confidence_interval, bool):
        raise TypeError("use_confidence_interval must be a bool")
    if not isinstance(require_valid, bool):
        raise TypeError("require_valid must be a bool")
    if require_valid and (not first.is_valid or not second.is_valid):
        raise ValueError("cannot compare invalid latency measurements")

    median_difference_ms = abs(first.median_ms - second.median_ms)
    tolerance_ms = max(
        float(absolute_tolerance_ms),
        float(relative_tolerance) * max(first.median_ms, second.median_ms),
    )
    within_tolerance = median_difference_ms <= tolerance_ms
    confidence_intervals_overlap = (
        max(first.bootstrap_median_ci_ms[0], second.bootstrap_median_ci_ms[0])
        <= min(first.bootstrap_median_ci_ms[1], second.bootstrap_median_ci_ms[1])
    )
    equivalent = within_tolerance or (
        use_confidence_interval and confidence_intervals_overlap
    )

    if within_tolerance and use_confidence_interval and confidence_intervals_overlap:
        reason = "within_tolerance_and_confidence_intervals_overlap"
    elif within_tolerance:
        reason = "within_tolerance"
    elif use_confidence_interval and confidence_intervals_overlap:
        reason = "confidence_intervals_overlap"
    else:
        reason = "meaningfully_different"

    return LatencyComparison(
        equivalent=equivalent,
        median_difference_ms=median_difference_ms,
        tolerance_ms=tolerance_ms,
        within_tolerance=within_tolerance,
        confidence_intervals_overlap=confidence_intervals_overlap,
        reason=reason,
    )
