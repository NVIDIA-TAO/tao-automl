# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-neutral, synchronized latency measurement contract.

Model adapters own input preparation and the exact callable that is timed.
This module owns the parts that must be identical across AutoML candidates:
warm-up and round counts, accelerator synchronization, monotonic timing,
replica provenance, raw-sample preservation, and deterministic aggregation.

One process records one accelerator replica.  A distributed launcher runs the
same frozen contract on every replica and then combines the records with
``combine_replica_records``.  Matched post-selection measurements remain a
separate workflow; this module does not invoke a selector.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Any

from tao_automl.latency_stats import (
    LatencyProtocol,
    LatencyStatistics,
    LatencyValidityThresholds,
    aggregate_synchronized_latency,
)


LATENCY_BENCHMARK_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of strict canonical JSON."""
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class LatencyBenchmarkContract:
    """Frozen measurement and deployment context shared by every candidate."""

    warmup_iterations: int = 50
    timed_iterations: int = 100
    repeated_rounds: int = 5
    tail_percentile: float = 95.0
    bootstrap_resamples: int = 2000
    bootstrap_confidence_level: float = 0.95
    bootstrap_seed: int = 0
    batch_size_per_replica: int = 1
    precision: str = "fp32"
    timed_scope: str = "model_forward"
    input_sha256: str = ""
    runtime_sha256: str = ""
    expected_replicas: int = 1
    measurement_role: str = "selection_time"
    synchronization: str = "accelerator_sync_before_and_after_each_sample"
    validity_thresholds: LatencyValidityThresholds = LatencyValidityThresholds()

    def __post_init__(self) -> None:
        # Reuse the low-level protocol's strict numeric validation.
        LatencyProtocol(
            warmup_iterations=self.warmup_iterations,
            timed_iterations=self.timed_iterations,
            repeated_rounds=self.repeated_rounds,
            tail_percentile=self.tail_percentile,
            bootstrap_resamples=self.bootstrap_resamples,
            bootstrap_confidence_level=self.bootstrap_confidence_level,
            bootstrap_seed=self.bootstrap_seed,
            validity_thresholds=self.validity_thresholds,
        )
        if (
            isinstance(self.batch_size_per_replica, bool)
            or not isinstance(self.batch_size_per_replica, int)
            or self.batch_size_per_replica < 1
        ):
            raise ValueError("batch_size_per_replica must be an integer >= 1")
        if (
            isinstance(self.expected_replicas, bool)
            or not isinstance(self.expected_replicas, int)
            or self.expected_replicas < 1
        ):
            raise ValueError("expected_replicas must be an integer >= 1")
        _nonempty(self.precision, "precision")
        _nonempty(self.timed_scope, "timed_scope")
        _require_sha256(self.input_sha256, "input_sha256")
        _require_sha256(self.runtime_sha256, "runtime_sha256")
        if self.measurement_role not in {
            "selection_time",
            "validation_only",
        }:
            raise ValueError(
                "measurement_role must be 'selection_time' or "
                "'validation_only'"
            )
        if (
            self.synchronization
            != "accelerator_sync_before_and_after_each_sample"
        ):
            raise ValueError(
                "synchronization must require accelerator synchronization "
                "before and after each timed sample"
            )
        if not isinstance(self.validity_thresholds, LatencyValidityThresholds):
            raise TypeError(
                "validity_thresholds must be LatencyValidityThresholds"
            )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema_version"] = LATENCY_BENCHMARK_SCHEMA_VERSION
        return value

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def statistics_protocol(
        self,
        device_ids: Sequence[str],
    ) -> LatencyProtocol:
        return LatencyProtocol(
            warmup_iterations=self.warmup_iterations,
            timed_iterations=self.timed_iterations,
            repeated_rounds=self.repeated_rounds,
            tail_percentile=self.tail_percentile,
            bootstrap_resamples=self.bootstrap_resamples,
            bootstrap_confidence_level=self.bootstrap_confidence_level,
            bootstrap_seed=self.bootstrap_seed,
            expected_devices=tuple(device_ids),
            validity_thresholds=self.validity_thresholds,
        )


@dataclass(frozen=True, slots=True)
class ReplicaIdentity:
    """Immutable identity of one benchmark worker and accelerator."""

    rank: int
    world_size: int
    device_id: str
    hardware_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("rank must be an integer")
        if (
            isinstance(self.world_size, bool)
            or not isinstance(self.world_size, int)
            or self.world_size < 1
        ):
            raise ValueError("world_size must be an integer >= 1")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        _nonempty(self.device_id, "device_id")
        _require_sha256(self.hardware_sha256, "hardware_sha256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_samples(
    samples: Any,
    contract: LatencyBenchmarkContract,
) -> tuple[tuple[float, ...], ...]:
    if (
        not isinstance(samples, Sequence)
        or isinstance(samples, (str, bytes))
        or len(samples) != contract.repeated_rounds
    ):
        raise ValueError(
            "samples must contain exactly repeated_rounds sequences"
        )
    normalized = []
    for round_index, values in enumerate(samples):
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or len(values) != contract.timed_iterations
        ):
            raise ValueError(
                f"round {round_index} must contain exactly "
                f"{contract.timed_iterations} timed samples"
            )
        round_values = []
        for sample in values:
            if (
                isinstance(sample, bool)
                or not isinstance(sample, Real)
                or not math.isfinite(float(sample))
                or float(sample) <= 0.0
            ):
                raise ValueError(
                    "latency samples must be finite positive numbers"
                )
            round_values.append(float(sample))
        normalized.append(tuple(round_values))
    return tuple(normalized)


def _record_payload(
    *,
    contract: LatencyBenchmarkContract,
    identity: ReplicaIdentity,
    candidate_fingerprint: str,
    samples: Sequence[Sequence[Real]],
) -> dict[str, Any]:
    normalized = _validate_samples(samples, contract)
    payload = {
        "schema_version": LATENCY_BENCHMARK_SCHEMA_VERSION,
        "contract": contract.to_dict(),
        "contract_sha256": contract.sha256,
        "identity": identity.to_dict(),
        "candidate_fingerprint": _require_sha256(
            candidate_fingerprint,
            "candidate_fingerprint",
        ),
        "samples_ms": [list(values) for values in normalized],
        "selection_isolation": _selection_isolation(contract),
    }
    payload["record_sha256"] = canonical_sha256(payload)
    return payload


def _selection_isolation(
    contract: LatencyBenchmarkContract,
) -> dict[str, bool]:
    """Return the explicit selection/validation boundary for this record."""
    return {
        "selector_invoked_on_matched_measurements": False,
        "selection_time_objectives_replaced": False,
        "measurements_feed_selection": (
            contract.measurement_role == "selection_time"
        ),
        "measurements_feed_reselection": False,
        "algorithm_selected_candidate_overridden": False,
    }


def run_replica_benchmark(
    *,
    contract: LatencyBenchmarkContract,
    identity: ReplicaIdentity,
    candidate_fingerprint: str,
    step: Callable[[int, int], Any],
    synchronize: Callable[[], None],
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Measure one replica with synchronization outside the timed callable.

    ``step(round_index, iteration)`` must execute the exact preregistered timed
    scope using already prepared inputs.  Preprocessing, transfer, and
    postprocessing are included only when the model adapter puts them in that
    callable and declares them in ``contract.timed_scope``.
    """
    if not isinstance(contract, LatencyBenchmarkContract):
        raise TypeError("contract must be LatencyBenchmarkContract")
    if not isinstance(identity, ReplicaIdentity):
        raise TypeError("identity must be ReplicaIdentity")
    if identity.world_size != contract.expected_replicas:
        raise ValueError(
            "replica world_size does not match contract.expected_replicas"
        )
    if not callable(step) or not callable(synchronize) or not callable(clock_ns):
        raise TypeError("step, synchronize, and clock_ns must be callable")

    # Warm all caches and kernels through the same callable. Warmups are never
    # reported as timed samples.
    for warmup_index in range(contract.warmup_iterations):
        synchronize()
        step(-1, warmup_index)
        synchronize()

    samples: list[list[float]] = []
    for round_index in range(contract.repeated_rounds):
        values = []
        for iteration in range(contract.timed_iterations):
            synchronize()
            start = clock_ns()
            step(round_index, iteration)
            synchronize()
            stop = clock_ns()
            if (
                isinstance(start, bool)
                or isinstance(stop, bool)
                or not isinstance(start, int)
                or not isinstance(stop, int)
                or stop <= start
            ):
                raise ValueError(
                    "clock_ns must return strictly increasing integer "
                    "nanosecond timestamps"
                )
            values.append((stop - start) / 1_000_000.0)
        samples.append(values)

    return _record_payload(
        contract=contract,
        identity=identity,
        candidate_fingerprint=candidate_fingerprint,
        samples=samples,
    )


def validate_replica_record(record: Any) -> dict[str, Any]:
    """Validate record integrity and return a defensive copy."""
    if not isinstance(record, Mapping):
        raise TypeError("replica record must be a mapping")
    value = copy.deepcopy(dict(record))
    if value.get("schema_version") != LATENCY_BENCHMARK_SCHEMA_VERSION:
        raise ValueError("replica record schema version is unsupported")
    expected = value.pop("record_sha256", None)
    if not isinstance(expected, str) or canonical_sha256(value) != expected:
        raise ValueError("replica record integrity verification failed")
    value["record_sha256"] = expected
    contract_raw = value.get("contract")
    if not isinstance(contract_raw, Mapping):
        raise ValueError("replica record contract is missing")
    contract_values = dict(contract_raw)
    contract_values.pop("schema_version", None)
    thresholds = contract_values.get("validity_thresholds")
    if isinstance(thresholds, Mapping):
        contract_values["validity_thresholds"] = LatencyValidityThresholds(
            **dict(thresholds)
        )
    contract = LatencyBenchmarkContract(**contract_values)
    if value.get("contract_sha256") != contract.sha256:
        raise ValueError("replica record contract hash does not match")
    identity = ReplicaIdentity(**value.get("identity", {}))
    _require_sha256(
        value.get("candidate_fingerprint"),
        "candidate_fingerprint",
    )
    _validate_samples(value.get("samples_ms"), contract)
    expected_isolation = _selection_isolation(contract)
    if value.get("selection_isolation") != expected_isolation:
        raise ValueError("replica record selection-isolation flags changed")
    value["_contract"] = contract
    value["_identity"] = identity
    return value


def _statistics_dict(statistics: LatencyStatistics) -> dict[str, Any]:
    return {
        "median_ms": statistics.median_ms,
        "p95_ms": statistics.tail_latency_ms,
        "mad_ms": statistics.mad_ms,
        "iqr_ms": statistics.iqr_ms,
        "robust_cv": statistics.robust_cv,
        "bootstrap_median_ci_ms": list(
            statistics.bootstrap_median_ci_ms
        ),
        "round_median_range_ms": statistics.round_median_range_ms,
        "round_drift_ms": statistics.round_drift_ms,
        "device_median_range_ms": statistics.device_median_range_ms,
        "raw_sample_count_total": statistics.raw_sample_count_total,
        "samples_per_device": statistics.samples_per_device,
        "is_valid": statistics.is_valid,
        "invalid_reasons": list(statistics.invalid_reasons),
    }


def combine_replica_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate a complete replica set and deterministically aggregate it."""
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence")
    validated = [validate_replica_record(record) for record in records]
    if not validated:
        raise ValueError("at least one replica record is required")

    contract = validated[0]["_contract"]
    contract_hashes = {item["contract_sha256"] for item in validated}
    candidates = {item["candidate_fingerprint"] for item in validated}
    world_sizes = {item["_identity"].world_size for item in validated}
    ranks = [item["_identity"].rank for item in validated]
    device_ids = [item["_identity"].device_id for item in validated]
    hardware = {item["_identity"].hardware_sha256 for item in validated}
    if len(contract_hashes) != 1:
        raise ValueError("replica records use different benchmark contracts")
    if len(candidates) != 1:
        raise ValueError("replica records refer to different candidates")
    if world_sizes != {contract.expected_replicas}:
        raise ValueError("replica records use an unexpected world size")
    if sorted(ranks) != list(range(contract.expected_replicas)):
        raise ValueError("replica records must contain every rank exactly once")
    if len(set(device_ids)) != len(device_ids):
        raise ValueError("replica device IDs must be unique")
    if len(hardware) != 1:
        raise ValueError("replica records use different hardware contracts")

    ordered = sorted(validated, key=lambda item: item["_identity"].rank)
    samples = {
        round_index: {
            item["_identity"].device_id: item["samples_ms"][round_index]
            for item in ordered
        }
        for round_index in range(contract.repeated_rounds)
    }
    statistics = aggregate_synchronized_latency(
        samples,
        contract.statistics_protocol(
            [item["_identity"].device_id for item in ordered]
        ),
    )
    result = {
        "schema_version": LATENCY_BENCHMARK_SCHEMA_VERSION,
        "contract": contract.to_dict(),
        "contract_sha256": contract.sha256,
        "candidate_fingerprint": next(iter(candidates)),
        "hardware_sha256": next(iter(hardware)),
        "replica_record_sha256": [
            item["record_sha256"] for item in ordered
        ],
        "statistics": _statistics_dict(statistics),
        "selection_isolation": copy.deepcopy(
            ordered[0]["selection_isolation"]
        ),
    }
    result["aggregate_sha256"] = canonical_sha256(result)
    return result


def write_record_atomic(path: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
    """Validate and atomically persist one raw replica record."""
    validated = validate_replica_record(record)
    validated.pop("_contract", None)
    validated.pop("_identity", None)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(validated))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "LATENCY_BENCHMARK_SCHEMA_VERSION",
    "LatencyBenchmarkContract",
    "ReplicaIdentity",
    "canonical_sha256",
    "combine_replica_records",
    "run_replica_benchmark",
    "validate_replica_record",
    "write_record_atomic",
]
