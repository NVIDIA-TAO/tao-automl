#!/usr/bin/env python3

"""Run one matched phase-2 latency block inside an eight-GPU allocation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--benchmark-script", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != 1:
        raise ValueError("allocation plan schema_version must be 1")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 6:
        raise ValueError("allocation plan must contain exactly six candidates")
    positions = [candidate.get("position") for candidate in candidates]
    if positions != list(range(6)):
        raise ValueError("allocation candidate positions must be 0..5 in order")
    candidate_ids = [candidate.get("candidate_id") for candidate in candidates]
    if len(set(candidate_ids)) != 6:
        raise ValueError("allocation candidate IDs must be unique")
    run_labels = [candidate.get("run_label") for candidate in candidates]
    if len(set(run_labels)) != 6 or any(
        not isinstance(label, str) or not label for label in run_labels
    ):
        raise ValueError("allocation run labels must be unique non-empty strings")
    if plan.get("gpu_count") != 8:
        raise ValueError("allocation plan must require exactly eight GPUs")
    if plan.get("feeds_selection") is not False:
        raise ValueError("phase-2 allocation must be validation-only")


def main() -> int:
    args = parse_args()
    plan_path = Path(args.plan)
    benchmark_script = Path(args.benchmark_script)
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    if not benchmark_script.is_file():
        raise FileNotFoundError(benchmark_script)

    tao_job_id = os.environ.get("TAO_JOB_ID")
    if not tao_job_id:
        raise RuntimeError("TAO_JOB_ID is required for collision-free output")
    allocation_root = (
        Path(args.output_root)
        / "dino_moo_phase2_20260728"
        / plan["allocation_id"]
        / tao_job_id
    )
    result_path = allocation_root / "allocation_result.json"
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_timestamp(),
        "manifest_id": plan["manifest_id"],
        "allocation_id": plan["allocation_id"],
        "tao_job_id": tao_job_id,
        "hostname": socket.gethostname(),
        "feeds_selection": False,
        "candidate_runs": [],
    }
    atomic_json(result_path, result)

    protocol = plan["latency_protocol"]
    benchmark = plan["latency_benchmark"]
    for candidate in plan["candidates"]:
        candidate_output_root = allocation_root / candidate["run_label"]
        command = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=8",
            str(benchmark_script),
            "--config",
            candidate["config_path"],
            "--checkpoint",
            candidate["checkpoint"],
            "--output-root",
            str(candidate_output_root),
            "--warmup-iterations",
            str(protocol["warmup_iterations"]),
            "--timed-iterations",
            str(protocol["timed_iterations"]),
            "--rounds",
            str(protocol["repeated_rounds"]),
            "--preloaded-batches",
            str(benchmark["preloaded_batches"]),
            "--seed",
            str(benchmark["benchmark_seed"]),
        ]
        run_record = {
            "candidate_id": candidate["candidate_id"],
            "position": candidate["position"],
            "run_label": candidate["run_label"],
            "status": "running",
            "started_at": utc_timestamp(),
            "command": command,
            "raw_samples_dir": str(
                candidate_output_root / tao_job_id / "latency"
            ),
        }
        result["candidate_runs"].append(run_record)
        atomic_json(result_path, result)
        completed = subprocess.run(command, check=False)
        run_record["exit_code"] = completed.returncode
        run_record["completed_at"] = utc_timestamp()
        if completed.returncode != 0:
            run_record["status"] = "failure"
            result["status"] = "failure"
            result["failed_candidate_id"] = candidate["candidate_id"]
            result["completed_at"] = utc_timestamp()
            atomic_json(result_path, result)
            return completed.returncode
        run_record["status"] = "success"
        atomic_json(result_path, result)

    result["status"] = "success"
    result["completed_at"] = utc_timestamp()
    atomic_json(result_path, result)
    print(
        "TAO_AUTOML_PHASE2_ALLOCATION_COMPLETE "
        f"allocation_id={plan['allocation_id']} candidates=6",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
