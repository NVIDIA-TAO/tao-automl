#!/usr/bin/env python3

"""Repeat the frozen algorithm-selected winner's latency measurement.

This historical validation-only driver reads a completed shared archive in
which all three modes happened to identify one common winner, and launches
three new independent eight-GPU SQSH benchmark jobs. The common-winner check is
a precondition of this archived experiment, not a product requirement that
mode winners be distinct or shared. The driver never invokes archive selection
and never writes ``combined_selection.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

from tao_sdk.platforms.slurm import SlurmSDK

from run_experiment import (
    BENCHMARK_SCRIPT,
    EXPERIMENT_DIR,
    GPU_COUNT,
    SQSH_PATH,
    atomic_json,
    configure_slurm_environment,
    launch_latency_benchmark,
)


SELECTION_PATH = EXPERIMENT_DIR / "combined_selection.json"
OUTPUT_PATH = EXPERIMENT_DIR / "winner_latency_repeats.json"
RUNTIME_DIR = EXPERIMENT_DIR / "winner_latency_repeats"
EVENT_PATH = RUNTIME_DIR / "events.jsonl"
STATE_PATH = RUNTIME_DIR / "slurm_state.json"
REPEAT_COUNT = 3
MODES = ("accuracy", "latency", "multi_objective")


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_unique_winner(
    selection: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    selections = selection.get("selections")
    if not isinstance(selections, dict):
        raise ValueError("combined selection has no selections mapping")

    mode_snapshot: dict[str, Any] = {}
    winner_ids: set[str] = set()
    for mode in MODES:
        mode_selection = selections.get(mode)
        if (
            not isinstance(mode_selection, dict)
            or mode_selection.get("status") != "selected"
            or not isinstance(mode_selection.get("winner_id"), str)
        ):
            raise ValueError(f"{mode} does not contain a completed selection")
        mode_snapshot[mode] = {
            "status": mode_selection["status"],
            "winner_id": mode_selection["winner_id"],
            "reason": mode_selection.get("reason"),
            "distinct_compromise": mode_selection.get("distinct_compromise"),
            "fallback_used": mode_selection.get("fallback_used"),
        }
        winner_ids.add(mode_selection["winner_id"])

    if len(winner_ids) != 1:
        raise ValueError(
            "validation-repeat driver requires one unique algorithm-selected "
            f"winner across all modes, got {sorted(winner_ids)}"
        )
    winner_id = next(iter(winner_ids))
    records = selection.get("candidate_records")
    if not isinstance(records, dict) or winner_id not in records:
        raise ValueError(f"selected winner {winner_id!r} has no candidate record")
    winner = records[winner_id]
    if winner.get("status") != "success":
        raise ValueError(f"selected winner {winner_id!r} was not successful")
    if winner.get("candidate_id") != winner_id:
        raise ValueError("selected winner ID does not match its candidate record")

    checkpoint = winner.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint.startswith("/lustre/"):
        raise ValueError("selected winner checkpoint is not an absolute Lustre path")
    num_queries = winner.get("num_queries")
    if (
        isinstance(num_queries, bool)
        or not isinstance(num_queries, int)
        or num_queries <= 0
    ):
        raise ValueError("selected winner has an invalid num_queries value")
    if int(winner["specs"]["model.num_queries"]) != num_queries:
        raise ValueError("selected winner num_queries conflicts with its specs")
    return winner_id, winner, mode_snapshot


def summarize(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    standard_deviation = statistics.pstdev(values)
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "range": max(values) - min(values),
        "mean": mean,
        "median": statistics.median(values),
        "population_standard_deviation": standard_deviation,
        "coefficient_of_variation": (
            standard_deviation / mean if not math.isclose(mean, 0.0) else 0.0
        ),
    }


def initial_payload(
    *,
    selection_sha256: str,
    benchmark_script_sha256: str,
    winner_id: str,
    winner: dict[str, Any],
    mode_snapshot: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_timestamp(),
        "purpose": (
            "Independent-allocation latency repeatability validation for the "
            "historical archive's common algorithm-selected winner."
        ),
        "validation_only": True,
        "feeds_selection": False,
        "selection_source": str(SELECTION_PATH),
        "selection_source_sha256": selection_sha256,
        "benchmark_script_sha256": benchmark_script_sha256,
        "repeat_count_requested": REPEAT_COUNT,
        "gpu_count_per_repeat": GPU_COUNT,
        "sqsh_path": SQSH_PATH,
        "mode_selection_snapshot": mode_snapshot,
        "selected_candidate": {
            "candidate_id": winner_id,
            "checkpoint": winner["checkpoint"],
            "num_queries": winner["num_queries"],
            "specs": winner["specs"],
            "mAP50": winner["objective_values"]["mAP50"],
            "original_latency_metrics": {
                key: value
                for key, value in winner["objective_values"].items()
                if key.startswith("latency_")
            },
        },
        "repeats": [],
    }


def load_or_initialize() -> tuple[dict[str, Any], dict[str, Any]]:
    selection_sha256 = file_sha256(SELECTION_PATH)
    benchmark_script_sha256 = file_sha256(BENCHMARK_SCRIPT)
    selection = json.loads(SELECTION_PATH.read_text())
    winner_id, winner, mode_snapshot = selected_unique_winner(selection)

    if OUTPUT_PATH.exists():
        payload = json.loads(OUTPUT_PATH.read_text())
        expected = {
            "selection_source_sha256": selection_sha256,
            "benchmark_script_sha256": benchmark_script_sha256,
            "repeat_count_requested": REPEAT_COUNT,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(
                    f"existing repeat evidence has incompatible {key}: "
                    f"{payload.get(key)!r} != {value!r}"
                )
        if payload.get("selected_candidate", {}).get("candidate_id") != winner_id:
            raise RuntimeError(
                "existing repeat evidence targets a different selected candidate"
            )
    else:
        payload = initial_payload(
            selection_sha256=selection_sha256,
            benchmark_script_sha256=benchmark_script_sha256,
            winner_id=winner_id,
            winner=winner,
            mode_snapshot=mode_snapshot,
        )
        atomic_json(OUTPUT_PATH, payload)
    return payload, winner


def main() -> int:
    payload, winner = load_or_initialize()
    repeats = payload["repeats"]
    if len(repeats) > REPEAT_COUNT:
        raise RuntimeError("repeat evidence contains more than three repeats")
    if payload.get("status") == "complete":
        if len(repeats) != REPEAT_COUNT:
            raise RuntimeError("complete repeat evidence has an incomplete job set")
        print(
            "WINNER_LATENCY_REPEATS_ALREADY_COMPLETE "
            + json.dumps(payload["aggregate"], sort_keys=True),
            flush=True,
        )
        return 0

    payload["status"] = "running"
    payload.pop("error", None)
    atomic_json(OUTPUT_PATH, payload)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    configure_slurm_environment()
    sdk = SlurmSDK(poll_interval=10, state_file=STATE_PATH)

    try:
        while len(repeats) < REPEAT_COUNT:
            repeat_index = len(repeats)
            metrics, benchmark = launch_latency_benchmark(
                sdk,
                winner["checkpoint"],
                int(winner["num_queries"]),
                event_path=EVENT_PATH,
                seed=int(winner["search_seed"]),
                rec_id=10_000 + repeat_index,
            )
            repeat = {
                "repeat_index": repeat_index,
                "completed_at": utc_timestamp(),
                "metrics": metrics,
                "benchmark": benchmark,
            }
            repeats.append(repeat)

            identities = {
                item["benchmark"]["benchmark_inputs"]["identity_sha256"]
                for item in repeats
            }
            if len(identities) != 1:
                raise RuntimeError(
                    "independent repeats used different benchmark input workloads: "
                    f"{sorted(identities)}"
                )
            payload["completed_repeat_count"] = len(repeats)
            payload["input_identity_sha256"] = next(iter(identities))
            atomic_json(OUTPUT_PATH, payload)

        medians = [float(item["metrics"]["latency_ms"]) for item in repeats]
        tails = [float(item["metrics"]["latency_p95_ms"]) for item in repeats]
        original_median = float(
            winner["objective_values"]["latency_ms"]
        )
        payload["aggregate"] = {
            "independent_job_count": len(repeats),
            "median_latency_ms": summarize(medians),
            "p95_latency_ms": summarize(tails),
            "original_selection_measurement_median_ms": original_median,
            "repeat_delta_from_original_ms": [
                value - original_median for value in medians
            ],
            "all_jobs_used_identical_inputs": True,
            "input_identity_sha256": payload["input_identity_sha256"],
        }
        payload["status"] = "complete"
        payload["completed_at"] = utc_timestamp()
        atomic_json(OUTPUT_PATH, payload)
    except BaseException as exc:
        payload["status"] = "failure"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["failed_at"] = utc_timestamp()
        atomic_json(OUTPUT_PATH, payload)
        raise

    print(
        "WINNER_LATENCY_REPEATS_COMPLETE "
        + json.dumps(payload["aggregate"], sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
