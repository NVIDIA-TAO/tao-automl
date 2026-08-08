#!/usr/bin/env python3

"""Audit frozen Mask2Former selection-time latency traces read-only.

The audit intentionally does not call the AutoML selector.  It verifies the
signed per-replica records for each persisted active-mode winner, reproduces
the aggregate median and p95, and partitions the timed samples by the frozen
16-input schedule.  The resulting artifact is diagnostic evidence only and
cannot replace a selection-time objective or trigger reselection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


MODES = ("accuracy", "latency", "multi_objective")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sample")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _verify_signed_record(path: Path) -> dict[str, Any]:
    record = _load_json(path)
    expected = record.get("record_sha256")
    unsigned = dict(record)
    unsigned.pop("record_sha256", None)
    if not isinstance(expected, str) or canonical_sha256(unsigned) != expected:
        raise ValueError(f"record digest mismatch: {path}")

    input_evidence = record.get("input_evidence")
    if not isinstance(input_evidence, dict):
        raise ValueError(f"input evidence is missing: {path}")
    input_expected = input_evidence.get("sha256")
    input_unsigned = dict(input_evidence)
    input_unsigned.pop("sha256", None)
    if (
        not isinstance(input_expected, str)
        or canonical_sha256(input_unsigned) != input_expected
    ):
        raise ValueError(f"input-evidence digest mismatch: {path}")
    return record


def _active_winner_id(result: Mapping[str, Any], mode: str) -> str:
    try:
        selection = result["result"]["selection_analysis"]["selections"][mode]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{mode} result has no active-mode selection") from exc
    if selection.get("status") != "selected":
        raise ValueError(f"{mode} active-mode selection is not terminal")
    return str(selection["winner_id"])


def _candidate(
    evidence: Mapping[str, Any], candidate_id: str
) -> dict[str, Any]:
    candidates = evidence.get("candidates")
    if not isinstance(candidates, Mapping):
        raise ValueError("candidate evidence does not contain a candidate map")
    value = candidates.get(candidate_id)
    if not isinstance(value, dict):
        matches = [
            candidate
            for candidate in candidates.values()
            if isinstance(candidate, dict)
            and str(candidate.get("rec_id")) == candidate_id
        ]
        value = matches[0] if len(matches) == 1 else None
    if not isinstance(value, dict):
        raise ValueError(f"candidate {candidate_id!r} is absent from evidence")
    return value


def _sample_summary(values: Sequence[float]) -> dict[str, float]:
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    mad = statistics.median(deviations)
    q25 = percentile(values, 0.25)
    q75 = percentile(values, 0.75)
    return {
        "median_ms": median,
        "p95_ms": percentile(values, 0.95),
        "mad_ms": mad,
        "iqr_ms": q75 - q25,
        "robust_cv": (1.4826 * mad / median) if median else 0.0,
    }


def analyze_trace_set(
    trace_dir: Path,
    *,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    paths = sorted(trace_dir.glob("rank_*.json"))
    if len(paths) != 8:
        raise ValueError(f"expected eight rank records in {trace_dir}")
    records = [_verify_signed_record(path) for path in paths]

    ranks = {record.get("identity", {}).get("rank") for record in records}
    if ranks != set(range(8)):
        raise ValueError(f"replica ranks are incomplete in {trace_dir}")
    fingerprints = {record.get("candidate_fingerprint") for record in records}
    candidate_fingerprint = candidate.get("candidate_fingerprint")
    if fingerprints != {candidate_fingerprint}:
        raise ValueError(f"candidate fingerprint mismatch in {trace_dir}")
    job_ids = {record.get("tao_job_id") for record in records}
    selection_latency = candidate.get("selection_time_latency", {})
    expected_job_id = selection_latency.get("tao_job_id")
    if expected_job_id is None:
        expected_job_id = (
            selection_latency.get("result_root", "")
            .rstrip("/")
            .rsplit("/", 1)[-1]
            or None
        )
    if expected_job_id is None:
        expected_job_id = candidate.get("tao_job_id")
    if expected_job_id is not None and job_ids != {expected_job_id}:
        raise ValueError(f"TAO job identity mismatch in {trace_dir}")

    contract_hashes = {record.get("contract_sha256") for record in records}
    input_hashes = {
        record.get("input_evidence", {}).get("sha256") for record in records
    }
    hardware_hashes = {
        record.get("identity", {}).get("hardware_sha256") for record in records
    }
    runtime_hashes = {
        record.get("contract", {}).get("runtime_sha256") for record in records
    }
    if any(len(values) != 1 for values in (
        contract_hashes,
        input_hashes,
        hardware_hashes,
        runtime_hashes,
    )):
        raise ValueError(f"replica provenance is inconsistent in {trace_dir}")

    contract = records[0]["contract"]
    rounds = int(contract["repeated_rounds"])
    iterations = int(contract["timed_iterations"])
    if rounds != 5 or iterations != 100:
        raise ValueError("frozen Mask2Former protocol must use 5 x 100 samples")

    batches = records[0]["input_evidence"].get("batches")
    if not isinstance(batches, list) or len(batches) != 16:
        raise ValueError("frozen Mask2Former input schedule must have 16 batches")
    all_samples: list[float] = []
    by_round: list[list[float]] = [[] for _ in range(rounds)]
    by_device: dict[str, list[float]] = {}
    device_round_medians: dict[str, list[float]] = {}
    by_position: list[list[float]] = [[] for _ in batches]
    for record in records:
        samples = record.get("samples_ms")
        if (
            not isinstance(samples, list)
            or len(samples) != rounds
            or any(not isinstance(row, list) or len(row) != iterations for row in samples)
        ):
            raise ValueError(f"sample matrix changed in {trace_dir}")
        device_key = f"rank_{record['identity']['rank']}"
        by_device[device_key] = []
        device_round_medians[device_key] = []
        for round_index, row in enumerate(samples):
            device_round_medians[device_key].append(
                statistics.median(float(value) for value in row)
            )
            for iteration, raw_value in enumerate(row):
                value = float(raw_value)
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(f"invalid latency sample in {trace_dir}")
                all_samples.append(value)
                by_round[round_index].append(value)
                by_device[device_key].append(value)
                linear_index = round_index * iterations + iteration
                by_position[linear_index % len(batches)].append(value)
    if len(all_samples) != 4000:
        raise ValueError(f"expected 4,000 timed samples in {trace_dir}")

    cluster_medians = [
        value
        for device_values in device_round_medians.values()
        for value in device_values
    ]
    primary_median = statistics.median(cluster_medians)
    aggregate = _sample_summary(all_samples)
    aggregate["raw_sample_median_ms"] = aggregate["median_ms"]
    aggregate["median_ms"] = primary_median
    aggregate["mad_ms"] = statistics.median(
        abs(value - primary_median) for value in all_samples
    )
    aggregate["robust_cv"] = (
        1.4826 * aggregate["mad_ms"] / primary_median
        if primary_median
        else 0.0
    )
    objectives = candidate.get("objective_values", {})
    if not math.isclose(
        aggregate["median_ms"], float(objectives["latency_ms"]), abs_tol=1e-9
    ):
        raise ValueError("raw median does not reproduce selection-time latency")
    if not math.isclose(
        aggregate["p95_ms"],
        float(objectives["latency_p95_ms"]),
        abs_tol=1e-9,
    ):
        raise ValueError("raw p95 does not reproduce selection-time latency")

    position_rows = []
    for index, values in enumerate(by_position):
        row = {
            "input_position": index,
            "model_input_shape": batches[index]["model_input_shape"],
            "sample_count": len(values),
            **_sample_summary(values),
        }
        position_rows.append(row)
    position_medians = [row["median_ms"] for row in position_rows]
    position_mads = [row["mad_ms"] for row in position_rows]
    position_range = max(position_medians) - min(position_medians)
    median_position_mad = statistics.median(position_mads)

    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_fingerprint": candidate_fingerprint,
        "tao_job_id": next(iter(job_ids)),
        "objective_values": objectives,
        "protocol": {
            "warmup_iterations": contract["warmup_iterations"],
            "timed_iterations": iterations,
            "repeated_rounds": rounds,
            "replicas": len(records),
            "timed_samples": len(all_samples),
            "precision": contract["precision"],
            "batch_size_per_replica": contract["batch_size_per_replica"],
            "timed_scope": contract["timed_scope"],
            "synchronization": contract["synchronization"],
        },
        "provenance": {
            "contract_sha256": next(iter(contract_hashes)),
            "input_evidence_sha256": next(iter(input_hashes)),
            "hardware_sha256": next(iter(hardware_hashes)),
            "runtime_sha256": next(iter(runtime_hashes)),
            "record_file_sha256": {
                path.name: file_sha256(path) for path in paths
            },
        },
        "aggregate": aggregate,
        "round_summaries": [
            {
                "round": index,
                **_sample_summary(values),
                "device_round_cluster_median_ms": statistics.median(
                    device_round_medians[device][index]
                    for device in sorted(device_round_medians)
                ),
            }
            for index, values in enumerate(by_round)
        ],
        "device_summaries": {
            device: {
                **_sample_summary(values),
                "device_round_cluster_median_ms": statistics.median(
                    device_round_medians[device]
                ),
            }
            for device, values in sorted(by_device.items())
        },
        "input_position_summaries": position_rows,
        "input_position_effect": {
            "position_median_min_ms": min(position_medians),
            "position_median_max_ms": max(position_medians),
            "position_median_range_ms": position_range,
            "median_within_position_mad_ms": median_position_mad,
            "range_to_within_position_mad_ratio": (
                position_range / median_position_mad
                if median_position_mad
                else None
            ),
            "interpretation": (
                "latency tail is reproducibly associated with positions in "
                "the frozen input-shape schedule; it is not an isolated "
                "cold-start sample"
            ),
        },
        "selection_isolation": {
            "selector_invoked_on_trace_audit": False,
            "trace_measurements_replaced": False,
            "measurements_feed_reselection": False,
            "algorithm_selected_candidate_overridden": False,
        },
    }


def build_audit(campaign_root: Path, trace_root: Path) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode in MODES:
        result_path = campaign_root / mode / "result.json"
        evidence_path = campaign_root / mode / "candidate_evidence.json"
        result = _load_json(result_path)
        evidence = _load_json(evidence_path)
        winner_id = _active_winner_id(result, mode)
        candidate = _candidate(evidence, winner_id)
        trace_dir = trace_root / f"{mode}_rec_{winner_id}"
        modes[mode] = {
            "persisted_active_winner_id": winner_id,
            "result_json_sha256": file_sha256(result_path),
            "candidate_evidence_json_sha256": file_sha256(evidence_path),
            "trace_analysis": analyze_trace_set(
                trace_dir,
                candidate=candidate,
            ),
        }
    document = {
        "schema_version": 1,
        "purpose": "read_only_mask2former_selection_time_latency_tail_audit",
        "campaign_root": str(campaign_root.resolve()),
        "trace_root": str(trace_root.resolve()),
        "selector_invoked": False,
        "selection_time_objectives_replaced": False,
        "measurements_feed_selection": False,
        "measurements_feed_reselection": False,
        "algorithm_selected_candidate_overridden": False,
        "modes": modes,
    }
    document["audit_sha256"] = canonical_sha256(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = build_audit(args.campaign_root, args.trace_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "audit_sha256": document["audit_sha256"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
