from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mask2former_latency_tail_audit as audit  # noqa: E402


def _record(rank: int) -> dict:
    batches = [
        {
            "batch_index": index,
            "model_input_shape": [1, 3, 32, 32 + index],
        }
        for index in range(16)
    ]
    input_evidence = {"schema_version": 1, "batches": batches}
    input_evidence["sha256"] = audit.canonical_sha256(input_evidence)
    rounds = []
    for round_index in range(5):
        rounds.append(
            [
                float(10 + ((round_index * 100 + iteration) % 16))
                for iteration in range(100)
            ]
        )
    record = {
        "candidate_fingerprint": "a" * 64,
        "contract": {
            "batch_size_per_replica": 1,
            "precision": "fp32",
            "repeated_rounds": 5,
            "runtime_sha256": "r" * 64,
            "synchronization": "accelerator_sync_before_and_after_each_sample",
            "timed_iterations": 100,
            "timed_scope": "mask2former_model_forward",
            "warmup_iterations": 50,
        },
        "contract_sha256": "c" * 64,
        "identity": {
            "hardware_sha256": "h" * 64,
            "rank": rank,
            "world_size": 8,
        },
        "input_evidence": input_evidence,
        "record_sha256": "",
        "samples_ms": rounds,
        "tao_job_id": "job",
    }
    unsigned = dict(record)
    unsigned.pop("record_sha256")
    record["record_sha256"] = audit.canonical_sha256(unsigned)
    return record


def _write_records(root: Path) -> tuple[list[float], float]:
    all_values = []
    cluster_medians = []
    for rank in range(8):
        record = _record(rank)
        all_values.extend(value for row in record["samples_ms"] for value in row)
        cluster_medians.extend(
            audit.percentile(row, 0.5) for row in record["samples_ms"]
        )
        (root / f"rank_{rank}.json").write_text(json.dumps(record))
    return all_values, audit.percentile(cluster_medians, 0.5)


def test_percentile_uses_linear_interpolation() -> None:
    assert audit.percentile([0.0, 10.0], 0.95) == pytest.approx(9.5)


def test_trace_audit_reproduces_aggregate_and_cross_round_schedule(
    tmp_path: Path,
) -> None:
    values, primary_median = _write_records(tmp_path)
    candidate = {
        "candidate_id": "mode_rec_0",
        "candidate_fingerprint": "a" * 64,
        "objective_values": {
            "latency_ms": primary_median,
            "latency_p95_ms": audit.percentile(values, 0.95),
        },
        "tao_job_id": "job",
    }
    result = audit.analyze_trace_set(tmp_path, candidate=candidate)
    assert result["aggregate"]["median_ms"] == pytest.approx(
        primary_median
    )
    assert [
        row["median_ms"] for row in result["input_position_summaries"]
    ] == pytest.approx([float(10 + index) for index in range(16)])
    assert result["protocol"]["timed_samples"] == 4000
    assert result["selection_isolation"]["measurements_feed_reselection"] is False


def test_signed_record_tampering_is_rejected(tmp_path: Path) -> None:
    record = _record(0)
    record["samples_ms"][0][0] = 999.0
    path = tmp_path / "rank_0.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="record digest mismatch"):
        audit._verify_signed_record(path)


def test_input_evidence_tampering_is_rejected(tmp_path: Path) -> None:
    record = _record(0)
    record["input_evidence"] = copy.deepcopy(record["input_evidence"])
    record["input_evidence"]["batches"][0]["model_input_shape"][-1] = 999
    unsigned = dict(record)
    unsigned.pop("record_sha256")
    record["record_sha256"] = audit.canonical_sha256(unsigned)
    path = tmp_path / "rank_0.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="input-evidence digest mismatch"):
        audit._verify_signed_record(path)
