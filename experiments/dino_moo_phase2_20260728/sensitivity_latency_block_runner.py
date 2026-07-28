#!/usr/bin/env python3

"""Run one 14-profile matched DINO sensitivity block on eight GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--benchmark-script", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != 1:
        raise ValueError("block plan schema_version must be 1")
    if plan.get("feeds_final_selection") is not False:
        raise ValueError("block plan must be validation-only")
    if plan.get("manual_promotion_permitted") is not False:
        raise ValueError("manual promotion must be disabled")
    if plan.get("gpu_count") != 8:
        raise ValueError("block plan must request exactly eight GPUs")
    if plan.get("output_contract") != {
        "root_expression": "$TAO_RESULTS_ROOT/$TAO_JOB_ID",
        "sdk_job_scoped": True,
        "relative_layout": (
            "dino_moo_phase2_20260728/sensitivity_latency/"
            "<manifest_id>/seed_<seed>/<allocation_id>"
        ),
    }:
        raise ValueError("SDK job-scoped output contract drift")
    claimed_plan_digest = plan.get("block_plan_sha256")
    unhashed_plan = dict(plan)
    unhashed_plan.pop("block_plan_sha256", None)
    if claimed_plan_digest != sha256_value(unhashed_plan):
        raise ValueError("block plan digest mismatch")
    profiles = plan.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != 14:
        raise ValueError("block plan must contain exactly 14 profiles")
    if [item.get("position") for item in profiles] != list(range(14)):
        raise ValueError("profile positions must be 0..13 in execution order")
    profile_ids = [item.get("profile_id") for item in profiles]
    if len(set(profile_ids)) != 14 or "reference" not in profile_ids:
        raise ValueError("block requires 14 unique profiles including reference")
    seed = plan.get("seed")
    if any(item.get("seed") != seed for item in profiles):
        raise ValueError("all block profiles must use the matched block seed")
    run_labels = [item.get("run_label") for item in profiles]
    if len(set(run_labels)) != 14 or any(not label for label in run_labels):
        raise ValueError("run labels must be unique and non-empty")

    protocol = plan["latency_protocol"]
    required_protocol = {
        "warmup_iterations": 50,
        "timed_iterations": 100,
        "repeated_rounds": 5,
        "preloaded_batches": 16,
        "batch_size_per_gpu": 1,
        "precision": "fp32",
        "tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "benchmark_seed": 20260727,
    }
    for key, expected in required_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"latency protocol drift: {key}")
    if protocol.get("fixed_preprocessed_shapes") != {
        "model_input": [1, 4, 800, 1333],
        "image_tensor": [1, 3, 800, 1333],
        "padding_mask": [1, 1, 800, 1333],
    }:
        raise ValueError("fixed preprocessed input shape contract drift")


def validate_hardware(expected: dict[str, Any]) -> dict[str, Any]:
    import torch

    if torch.cuda.device_count() != 8:
        raise RuntimeError(
            f"expected exactly 8 visible GPUs, got {torch.cuda.device_count()}"
        )
    devices = []
    for index in range(8):
        properties = torch.cuda.get_device_properties(index)
        record = {
            "index": index,
            "name": properties.name,
            "compute_capability": (
                f"{properties.major}.{properties.minor}"
            ),
            "total_memory_bytes": properties.total_memory,
        }
        if record["name"] != expected["gpu_name"]:
            raise RuntimeError(f"GPU {index} model mismatch: {record['name']}")
        if record["compute_capability"] != expected["compute_capability"]:
            raise RuntimeError(
                f"GPU {index} compute capability mismatch: "
                f"{record['compute_capability']}"
            )
        if record["total_memory_bytes"] != expected["total_memory_bytes"]:
            raise RuntimeError(
                f"GPU {index} memory mismatch: {record['total_memory_bytes']}"
            )
        devices.append(record)
    runtime = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if runtime["torch"] != expected["torch"]:
        raise RuntimeError(f"torch mismatch: {runtime['torch']}")
    if runtime["cuda"] != expected["cuda"]:
        raise RuntimeError(f"CUDA mismatch: {runtime['cuda']}")
    if runtime["cudnn"] != expected["cudnn"]:
        raise RuntimeError(f"cuDNN mismatch: {runtime['cudnn']}")
    return {"devices": devices, "runtime": runtime}


def validate_configs_and_checkpoints(
    plan: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    config_digests = {}
    checkpoint_digests = {}
    for profile in plan["profiles"]:
        profile_id = profile["profile_id"]
        config_path = Path(profile["config_path"])
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        actual_config_digest = sha256_file(config_path)
        if actual_config_digest != profile["config_sha256"]:
            raise RuntimeError(f"{profile_id}: staged config digest mismatch")
        config = yaml.safe_load(config_path.read_text())
        if config["train"]["activation_checkpoint"] is not False:
            raise RuntimeError(
                f"{profile_id}: activation_checkpoint must remain false"
            )
        if config["train"]["seed"] != plan["seed"]:
            raise RuntimeError(f"{profile_id}: matched training seed drift")
        if (
            config["evaluate"]["num_gpus"] != 8
            or config["evaluate"]["gpu_ids"] != list(range(8))
            or config["evaluate"]["num_nodes"] != 1
            or config["evaluate"]["batch_size"] != 1
        ):
            raise RuntimeError(f"{profile_id}: evaluation topology drift")
        if sha256_value(config["model"]) != profile[
            "resolved_model_spec_sha256"
        ]:
            raise RuntimeError(f"{profile_id}: full model spec digest mismatch")
        if config["evaluate"]["checkpoint"] != profile["checkpoint_path"]:
            raise RuntimeError(f"{profile_id}: config checkpoint path mismatch")
        config_digests[profile_id] = actual_config_digest

        checkpoint_path = profile["checkpoint_path"]
        if checkpoint_path not in checkpoint_digests:
            path = Path(checkpoint_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            checkpoint_digests[checkpoint_path] = sha256_file(path)
        if checkpoint_digests[checkpoint_path] != profile[
            "checkpoint_sha256"
        ]:
            raise RuntimeError(f"{profile_id}: checkpoint digest mismatch")

    reference = next(
        item for item in plan["profiles"] if item["profile_id"] == "reference"
    )
    for profile in plan["profiles"]:
        if profile["checkpoint_source_profile_id"] == "reference":
            if (
                profile["checkpoint_path"] != reference["checkpoint_path"]
                or profile["checkpoint_sha256"]
                != reference["checkpoint_sha256"]
            ):
                raise RuntimeError(
                    f"{profile['profile_id']}: invalid reference checkpoint reuse"
                )
    return config_digests, checkpoint_digests


def validate_sdk_job_output_root(requested: Path) -> tuple[str, Path]:
    tao_job_id = os.environ.get("TAO_JOB_ID")
    if not tao_job_id:
        raise RuntimeError("TAO_JOB_ID is required for collision-free outputs")
    results_root = os.environ.get("TAO_RESULTS_ROOT")
    if not results_root:
        raise RuntimeError(
            "TAO_RESULTS_ROOT is required for SDK-owned outputs"
        )
    expected_output_root = Path(
        os.path.abspath(os.path.join(results_root, tao_job_id))
    )
    supplied_output_root = Path(os.path.abspath(requested))
    if supplied_output_root != expected_output_root:
        raise RuntimeError(
            "output root must be exactly "
            "$TAO_RESULTS_ROOT/$TAO_JOB_ID: "
            f"{supplied_output_root} != {expected_output_root}"
        )
    if supplied_output_root.name != tao_job_id:
        raise RuntimeError("SDK job-scoped output root identity mismatch")
    return tao_job_id, supplied_output_root


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    if sha256_file(args.benchmark_script) != plan["benchmark_sha256"]:
        raise RuntimeError("pinned benchmark digest mismatch")
    tao_job_id, supplied_output_root = validate_sdk_job_output_root(
        args.output_root
    )

    allocation_root = (
        supplied_output_root
        / "dino_moo_phase2_20260728"
        / "sensitivity_latency"
        / plan["manifest_id"]
        / f"seed_{plan['seed']:06d}"
        / plan["allocation_id"]
    )
    result_path = allocation_root / "allocation_result.json"
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "preflight",
        "started_at_utc": timestamp(),
        "manifest_id": plan["manifest_id"],
        "manifest_sha256": plan["manifest_sha256"],
        "checkpoint_artifact_sha256": plan[
            "checkpoint_artifact_sha256"
        ],
        "schedule_sha256": plan["schedule_sha256"],
        "allocation_id": plan["allocation_id"],
        "seed": plan["seed"],
        "repeat_index": plan["repeat_index"],
        "williams_row_index": plan["williams_row_index"],
        "block_plan_sha256": plan["block_plan_sha256"],
        "tao_job_id": tao_job_id,
        "sdk_job_scoped_result_root": str(supplied_output_root),
        "output_contract": {
            "root_env": "TAO_RESULTS_ROOT",
            "job_scope_env": "TAO_JOB_ID",
            "root": str(supplied_output_root),
            "layout": (
                "dino_moo_phase2_20260728/sensitivity_latency/"
                "<manifest_id>/seed_<seed>/<allocation_id>"
            ),
        },
        "hostname": socket.gethostname(),
        "feeds_final_selection": False,
        "manual_promotion_permitted": False,
        "profile_runs": [],
    }
    atomic_json(result_path, result)
    try:
        result["hardware"] = validate_hardware(plan["expected_hardware"])
        (
            result["verified_config_sha256"],
            result["verified_checkpoint_sha256"],
        ) = validate_configs_and_checkpoints(plan)
    except Exception as error:
        result["status"] = "preflight_failure"
        result["error"] = f"{type(error).__name__}: {error}"
        result["completed_at_utc"] = timestamp()
        atomic_json(result_path, result)
        raise

    result["status"] = "running"
    atomic_json(result_path, result)
    protocol = plan["latency_protocol"]
    for profile in plan["profiles"]:
        candidate_output_root = (
            allocation_root / "profiles" / profile["run_label"]
        )
        command = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=8",
            str(args.benchmark_script),
            "--config",
            profile["config_path"],
            "--checkpoint",
            profile["checkpoint_path"],
            "--output-root",
            str(candidate_output_root),
            "--warmup-iterations",
            str(protocol["warmup_iterations"]),
            "--timed-iterations",
            str(protocol["timed_iterations"]),
            "--rounds",
            str(protocol["repeated_rounds"]),
            "--preloaded-batches",
            str(protocol["preloaded_batches"]),
            "--seed",
            str(protocol["benchmark_seed"]),
        ]
        run = {
            "profile_id": profile["profile_id"],
            "seed": profile["seed"],
            "position": profile["position"],
            "run_label": profile["run_label"],
            "checkpoint_path": profile["checkpoint_path"],
            "checkpoint_sha256": profile["checkpoint_sha256"],
            "checkpoint_source_profile_id": profile[
                "checkpoint_source_profile_id"
            ],
            "resolved_model_spec_sha256": profile[
                "resolved_model_spec_sha256"
            ],
            "config_sha256": profile["config_sha256"],
            "status": "running",
            "started_at_utc": timestamp(),
            "raw_samples_dir": str(
                candidate_output_root / tao_job_id / "latency"
            ),
        }
        result["profile_runs"].append(run)
        atomic_json(result_path, result)
        completed = subprocess.run(command, check=False)
        run["exit_code"] = completed.returncode
        run["completed_at_utc"] = timestamp()
        if completed.returncode != 0:
            run["status"] = "failure"
            result["status"] = "failure"
            result["failed_profile_id"] = profile["profile_id"]
            result["completed_at_utc"] = timestamp()
            atomic_json(result_path, result)
            return completed.returncode
        run["status"] = "success"
        atomic_json(result_path, result)

    result["status"] = "success"
    result["completed_at_utc"] = timestamp()
    atomic_json(result_path, result)
    print(
        "TAO_DINO_SENSITIVITY_LATENCY_BLOCK_COMPLETE "
        f"allocation_id={plan['allocation_id']} seed={plan['seed']} "
        "profiles=14",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
