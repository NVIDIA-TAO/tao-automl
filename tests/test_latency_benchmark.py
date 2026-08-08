# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the model-neutral synchronized latency benchmark contract."""

from __future__ import annotations

import copy
import json

import pytest

from tao_automl.latency_benchmark import (
    LatencyBenchmarkContract,
    ReplicaIdentity,
    canonical_sha256,
    combine_replica_records,
    run_replica_benchmark,
    validate_replica_record,
    write_record_atomic,
)


def _hash(label):
    return canonical_sha256({"label": label})


def _contract(**changes):
    values = {
        "warmup_iterations": 2,
        "timed_iterations": 3,
        "repeated_rounds": 2,
        "bootstrap_resamples": 50,
        "bootstrap_seed": 7,
        "input_sha256": _hash("input"),
        "runtime_sha256": _hash("runtime"),
        "expected_replicas": 2,
    }
    values.update(changes)
    return LatencyBenchmarkContract(**values)


class _Clock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 1_000_000
        return self.value


def _record(rank, *, contract=None, candidate=None, hardware=None):
    contract = contract or _contract()
    calls = []
    syncs = []
    return run_replica_benchmark(
        contract=contract,
        identity=ReplicaIdentity(
            rank=rank,
            world_size=2,
            device_id=f"cuda:{rank}",
            hardware_sha256=hardware or _hash("hardware"),
        ),
        candidate_fingerprint=candidate or _hash("candidate"),
        step=lambda round_index, iteration: calls.append(
            (round_index, iteration)
        ),
        synchronize=lambda: syncs.append(True),
        clock_ns=_Clock(),
    ), calls, syncs


def test_runner_warms_then_records_exact_round_sample_contract():
    record, calls, syncs = _record(0)

    assert calls[:2] == [(-1, 0), (-1, 1)]
    assert calls[2:] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    # Two synchronization calls around every warmup and timed invocation.
    assert len(syncs) == 2 * len(calls)
    assert record["samples_ms"] == [[1.0] * 3, [1.0] * 3]
    assert validate_replica_record(record)["record_sha256"] == (
        record["record_sha256"]
    )


def test_combiner_requires_complete_homogeneous_replica_set():
    first = _record(0)[0]
    second = _record(1)[0]

    aggregate = combine_replica_records([second, first])

    assert aggregate["statistics"]["median_ms"] == 1.0
    assert aggregate["statistics"]["p95_ms"] == 1.0
    assert aggregate["statistics"]["raw_sample_count_total"] == 12
    assert aggregate["statistics"]["samples_per_device"] == 6
    assert aggregate["statistics"]["is_valid"]
    assert aggregate["replica_record_sha256"] == [
        first["record_sha256"],
        second["record_sha256"],
    ]
    assert aggregate["aggregate_sha256"] == canonical_sha256(
        {key: value for key, value in aggregate.items() if key != "aggregate_sha256"}
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda records: records.__setitem__(1, _record(0)[0]),
            "every rank exactly once",
        ),
        (
            lambda records: records.__setitem__(
                1, _record(1, candidate=_hash("other"))[0]
            ),
            "different candidates",
        ),
        (
            lambda records: records.__setitem__(
                1, _record(1, hardware=_hash("other-hardware"))[0]
            ),
            "different hardware",
        ),
        (
            lambda records: records.__setitem__(
                1,
                _record(
                    1,
                    contract=_contract(precision="fp16"),
                )[0],
            ),
            "different benchmark contracts",
        ),
    ],
)
def test_combiner_rejects_provenance_drift(change, message):
    records = [_record(0)[0], _record(1)[0]]
    change(records)

    with pytest.raises(ValueError, match=message):
        combine_replica_records(records)


def test_integrity_and_selection_isolation_are_fail_closed():
    record = _record(0)[0]
    modified = copy.deepcopy(record)
    modified["samples_ms"][0][0] = 2.0
    with pytest.raises(ValueError, match="integrity"):
        validate_replica_record(modified)

    modified = copy.deepcopy(record)
    modified["selection_isolation"][
        "selector_invoked_on_matched_measurements"
    ] = True
    modified["record_sha256"] = canonical_sha256(
        {key: value for key, value in modified.items() if key != "record_sha256"}
    )
    with pytest.raises(ValueError, match="selection-isolation"):
        validate_replica_record(modified)


def test_atomic_record_write_preserves_validated_content(tmp_path):
    record = _record(0)[0]
    destination = tmp_path / "rank_0.json"

    write_record_atomic(destination, record)
    observed = json.loads(destination.read_text())

    assert observed == record
    validate_replica_record(observed)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"warmup_iterations": -1}, ValueError),
        ({"expected_replicas": True}, ValueError),
        ({"batch_size_per_replica": 0}, ValueError),
        ({"input_sha256": "bad"}, ValueError),
        ({"runtime_sha256": "bad"}, ValueError),
        ({"precision": ""}, ValueError),
        (
            {"synchronization": "none"},
            ValueError,
        ),
        ({"measurement_role": "reselection"}, ValueError),
    ],
)
def test_contract_rejects_invalid_or_unsynchronized_protocol(change, error):
    with pytest.raises(error):
        _contract(**change)


def test_runner_rejects_world_size_or_clock_drift():
    contract = _contract()
    identity = ReplicaIdentity(
        rank=0,
        world_size=1,
        device_id="cuda:0",
        hardware_sha256=_hash("hardware"),
    )
    with pytest.raises(ValueError, match="world_size"):
        run_replica_benchmark(
            contract=contract,
            identity=identity,
            candidate_fingerprint=_hash("candidate"),
            step=lambda *_: None,
            synchronize=lambda: None,
        )

    identity = ReplicaIdentity(
        rank=0,
        world_size=2,
        device_id="cuda:0",
        hardware_sha256=_hash("hardware"),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        run_replica_benchmark(
            contract=contract,
            identity=identity,
            candidate_fingerprint=_hash("candidate"),
            step=lambda *_: None,
            synchronize=lambda: None,
            clock_ns=lambda: 1,
        )


def test_validation_only_role_freezes_all_selection_isolation_flags_false():
    record = _record(
        0,
        contract=_contract(measurement_role="validation_only"),
    )[0]

    assert set(record["selection_isolation"].values()) == {False}


def test_selection_time_role_explicitly_feeds_only_initial_selection():
    record = _record(0)[0]

    assert record["selection_isolation"] == {
        "selector_invoked_on_matched_measurements": False,
        "selection_time_objectives_replaced": False,
        "measurements_feed_selection": True,
        "measurements_feed_reselection": False,
        "algorithm_selected_candidate_overridden": False,
    }
