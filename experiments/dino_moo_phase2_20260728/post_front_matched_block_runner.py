#!/usr/bin/env python3

"""Run one complete expanded-front matched-latency block on eight GPUs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time
from typing import Any, Iterable

import yaml


HEX = frozenset("0123456789abcdef")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--benchmark-script", type=Path, required=True)
    parser.add_argument("--latency-stats-module", type=Path, required=True)
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


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _reject_duplicate_pairs(
    pairs: Iterable[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {item}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


def major_minor_patch(value: Any, label: str) -> str:
    match = re.match(r"^(\d+\.\d+\.\d+)", str(value))
    if match is None:
        raise RuntimeError(f"{label} has no major.minor.patch prefix: {value}")
    return match.group(1)


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != 1:
        raise ValueError("block plan schema_version must be 1")
    if plan.get("manifest_id") != (
        "dino_expanded_post_front_matched_20260728_v1"
    ):
        raise ValueError("block plan manifest identity drift")
    for key in (
        "manifest_sha256",
        "schedule_sha256",
        "benchmark_sha256",
        "latency_stats_sha256",
        "block_runner_sha256",
    ):
        require_sha256(plan.get(key), f"block plan {key}")
    allocation_index = plan.get("allocation_index")
    if (
        isinstance(allocation_index, bool)
        or not isinstance(allocation_index, int)
        or allocation_index not in range(6)
    ):
        raise ValueError("allocation_index must be an integer in [0, 5]")
    expected_allocation_id = f"post_front_allocation_{allocation_index:02d}"
    if plan.get("allocation_id") != expected_allocation_id:
        raise ValueError("allocation identity/index mismatch")
    design_row_index = plan.get("design_row_index")
    if (
        isinstance(design_row_index, bool)
        or not isinstance(design_row_index, int)
        or design_row_index < 0
    ):
        raise ValueError("design_row_index must be a non-negative integer")
    for key in (
        "feeds_final_selection",
        "feeds_reselection",
        "manual_candidate_addition_or_removal_permitted",
        "winner_override_permitted",
        "selection_time_objective_replacement_permitted",
    ):
        if plan.get(key) is not False:
            raise ValueError(f"validation-only plan flag drift: {key}")
    if plan.get("gpu_count") != 8 or plan.get("num_nodes") != 1:
        raise ValueError("block plan must use exactly one node and eight GPUs")
    claimed = require_sha256(
        plan.get("block_plan_sha256"),
        "block plan SHA256",
    )
    unhashed = dict(plan)
    del unhashed["block_plan_sha256"]
    if sha256_value(unhashed) != claimed:
        raise ValueError("block plan canonical digest mismatch")
    expected_output = {
        "root_expression": "$TAO_RESULTS_ROOT/$TAO_JOB_ID",
        "sdk_job_scoped": True,
        "relative_layout": (
            "dino_moo_phase2_20260728/post_front_matched/"
            "<manifest_id>/<allocation_id>"
        ),
    }
    if plan.get("output_contract") != expected_output:
        raise ValueError("SDK job-scoped output contract drift")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("block plan candidates must be non-empty")
    if plan.get("candidate_count") != len(candidates):
        raise ValueError("block candidate count mismatch")
    if [item.get("position") for item in candidates] != list(
        range(len(candidates))
    ):
        raise ValueError("candidate positions must be contiguous and ordered")
    candidate_ids = [item.get("candidate_id") for item in candidates]
    if (
        len(set(candidate_ids)) != len(candidates)
        or any(
            not isinstance(item, str)
            or re.fullmatch(r"[A-Za-z0-9_.-]+", item) is None
            for item in candidate_ids
        )
    ):
        raise ValueError("candidate IDs must be unique safe path components")
    run_labels = [item.get("run_label") for item in candidates]
    expected_run_labels = [
        (
            f"{expected_allocation_id}_p{position:03d}_"
            f"{candidate_id}"
        )
        for position, candidate_id in enumerate(candidate_ids)
    ]
    if run_labels != expected_run_labels:
        raise ValueError("candidate run labels are not deterministically bound")
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        checkpoint_path = candidate.get("checkpoint_path")
        if (
            not isinstance(checkpoint_path, str)
            or not Path(checkpoint_path).is_absolute()
        ):
            raise ValueError(f"{candidate_id}: checkpoint path must be absolute")
        expected_config = f"configs/{candidate_id}.yaml"
        if candidate.get("config_relative_path") != expected_config:
            raise ValueError(f"{candidate_id}: staged config path drift")
        for key in (
            "checkpoint_sha256",
            "resolved_model_spec_sha256",
            "candidate_table_record_sha256",
            "config_sha256",
        ):
            require_sha256(candidate.get(key), f"{candidate_id} {key}")
    protocol = plan.get("latency_protocol", {})
    required = {
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
        "tail_percentile": 95,
        "bootstrap_resamples": 5000,
        "bootstrap_confidence_level": 0.95,
        "bootstrap_seed": 424242,
        "synchronization": "cuda_sync_each_sample_and_nccl_barrier",
        "timed_scope": "model_forward_plus_dino_gpu_postprocess",
    }
    for key, expected in required.items():
        if protocol.get(key) != expected:
            raise ValueError(f"latency protocol drift: {key}")
    if (
        protocol.get("timed_iterations")
        * protocol.get("repeated_rounds")
        * plan["gpu_count"]
        != 4000
    ):
        raise ValueError("raw latency sample-count contract drift")
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
                f"GPU {index} memory mismatch: "
                f"{record['total_memory_bytes']}"
            )
        devices.append(record)
    runtime = {
        "torch": torch.__version__,
        "torch_major_minor_patch": major_minor_patch(
            torch.__version__,
            "torch version",
        ),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if expected.get("torch_version_match") != "major_minor_patch":
        raise RuntimeError("unsupported torch version-match policy")
    if runtime["torch_major_minor_patch"] != expected["torch"]:
        raise RuntimeError(f"torch mismatch: {runtime['torch']}")
    if runtime["cuda"] != expected["cuda"]:
        raise RuntimeError(f"CUDA mismatch: {runtime['cuda']}")
    if runtime["cudnn"] != expected["cudnn"]:
        raise RuntimeError(f"cuDNN mismatch: {runtime['cudnn']}")
    return {"devices": devices, "runtime": runtime}


def validate_configs_and_checkpoints(
    plan: dict[str, Any],
    staging_root: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    config_digests: dict[str, str] = {}
    checkpoint_digests: dict[str, str] = {}
    for candidate in plan["candidates"]:
        candidate_id = candidate["candidate_id"]
        config_path = staging_root / candidate["config_relative_path"]
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        actual_config = sha256_file(config_path)
        if actual_config != candidate["config_sha256"]:
            raise RuntimeError(f"{candidate_id}: staged config digest mismatch")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if sha256_value(config["model"]) != candidate[
            "resolved_model_spec_sha256"
        ]:
            raise RuntimeError(f"{candidate_id}: full model mapping drift")
        if config["evaluate"]["checkpoint"] != candidate["checkpoint_path"]:
            raise RuntimeError(f"{candidate_id}: config checkpoint drift")
        if (
            config["dataset"]["batch_size"] != 1
            or config["dataset"]["workers"] != 0
            or config["evaluate"]["batch_size"] != 1
            or config["evaluate"]["num_gpus"] != 8
            or config["evaluate"]["gpu_ids"] != list(range(8))
            or config["evaluate"]["num_nodes"] != 1
        ):
            raise RuntimeError(f"{candidate_id}: evaluation topology drift")
        config_digests[candidate_id] = actual_config

        checkpoint_path = candidate["checkpoint_path"]
        if checkpoint_path not in checkpoint_digests:
            path = Path(checkpoint_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            checkpoint_digests[checkpoint_path] = sha256_file(path)
        if checkpoint_digests[checkpoint_path] != candidate[
            "checkpoint_sha256"
        ]:
            raise RuntimeError(f"{candidate_id}: checkpoint digest mismatch")
    return config_digests, checkpoint_digests


def validated_staging_root(plan_path: Path) -> Path:
    """Revalidate the job-private staging directory immediately before use."""

    resolved_plan = plan_path.resolve(strict=True)
    root = resolved_plan.parent.parent
    info = root.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise RuntimeError("staging root ownership/type/mode contract failed")
    if resolved_plan.parent.name != "plans" or resolved_plan.suffix != ".json":
        raise RuntimeError("block plan escaped the staged plans directory")
    return root


def load_latency_stats_module(path: Path, expected_sha256: str) -> Any:
    resolved = path.resolve(strict=True)
    info = resolved.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise RuntimeError("latency statistics module is not a private file")
    if sha256_file(resolved) != expected_sha256:
        raise RuntimeError("latency statistics source digest mismatch")
    module_name = "_dino_post_front_latency_stats"
    specification = importlib.util.spec_from_file_location(
        module_name,
        resolved,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load latency statistics module")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def validate_candidate_latency(
    raw_samples_dir: Path,
    protocol_source: dict[str, Any],
    latency_stats: Any,
) -> dict[str, Any]:
    paths = sorted(raw_samples_dir.glob("rank_*.json"))
    expected = [
        raw_samples_dir / f"rank_{rank}.json"
        for rank in range(8)
    ]
    if paths != expected:
        raise RuntimeError("candidate does not contain exact rank_0..7 records")
    samples = {
        round_index: {}
        for round_index in range(protocol_source["repeated_rounds"])
    }
    record_sha256: dict[str, str] = {}
    for rank, path in enumerate(paths):
        record = load_json(path)
        if (
            record.get("rank") != rank
            or record.get("local_rank") != rank
            or record.get("world_size") != 8
        ):
            raise RuntimeError(f"rank_{rank} distributed identity mismatch")
        rank_samples = record.get("samples_ms")
        if (
            not isinstance(rank_samples, list)
            or len(rank_samples) != protocol_source["repeated_rounds"]
        ):
            raise RuntimeError(f"rank_{rank} samples have wrong round count")
        for round_index, values in enumerate(rank_samples):
            if (
                not isinstance(values, list)
                or len(values) != protocol_source["timed_iterations"]
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                    for value in values
                )
            ):
                raise RuntimeError(f"rank_{rank} timed samples are invalid")
            samples[round_index][str(rank)] = values
        record_sha256[str(rank)] = sha256_file(path)
    thresholds = protocol_source["validity_thresholds"]
    protocol = latency_stats.LatencyProtocol(
        warmup_iterations=protocol_source["warmup_iterations"],
        timed_iterations=protocol_source["timed_iterations"],
        repeated_rounds=protocol_source["repeated_rounds"],
        tail_percentile=protocol_source["tail_percentile"],
        bootstrap_resamples=protocol_source["bootstrap_resamples"],
        bootstrap_confidence_level=protocol_source[
            "bootstrap_confidence_level"
        ],
        bootstrap_seed=protocol_source["bootstrap_seed"],
        expected_devices=tuple(str(rank) for rank in range(8)),
        validity_thresholds=latency_stats.LatencyValidityThresholds(
            max_robust_cv=thresholds["max_robust_cv"],
            max_round_median_range_fraction=thresholds[
                "max_round_median_range_fraction"
            ],
            max_absolute_round_drift_fraction=thresholds[
                "max_absolute_round_drift_fraction"
            ],
            max_device_median_range_fraction=thresholds[
                "max_device_median_range_fraction"
            ],
            max_bootstrap_ci_width_fraction=thresholds[
                "max_bootstrap_ci_width_fraction"
            ],
        ),
    )
    statistics = latency_stats.aggregate_synchronized_latency(
        samples,
        protocol,
    )
    if not statistics.is_valid:
        raise RuntimeError(
            "latency validity thresholds failed: "
            f"{statistics.validity_reason}"
        )
    return {
        "is_valid": True,
        "validity_reason": statistics.validity_reason,
        "median_ms": statistics.median_ms,
        "p95_ms": statistics.tail_latency_ms,
        "robust_cv": statistics.robust_cv,
        "round_median_range_ms": statistics.round_median_range_ms,
        "device_median_range_ms": statistics.device_median_range_ms,
        "bootstrap_median_ci_ms": list(
            statistics.bootstrap_median_ci_ms
        ),
        "bootstrap_ci_width_ms": statistics.bootstrap_ci_width_ms,
        "raw_record_sha256_by_rank": record_sha256,
    }


def validate_sdk_output_root(requested: Path) -> tuple[str, Path]:
    tao_job_id = os.environ.get("TAO_JOB_ID")
    results_root = os.environ.get("TAO_RESULTS_ROOT")
    if not tao_job_id or not results_root:
        raise RuntimeError("TAO_JOB_ID and TAO_RESULTS_ROOT are required")
    expected = Path(
        os.path.abspath(os.path.join(results_root, tao_job_id))
    )
    supplied = Path(os.path.abspath(requested))
    if supplied != expected or supplied.name != tao_job_id:
        raise RuntimeError(
            "output root must be exactly $TAO_RESULTS_ROOT/$TAO_JOB_ID"
        )
    return tao_job_id, supplied


def main() -> int:
    args = parse_args()
    staging_root = validated_staging_root(args.plan)
    plan = load_json(args.plan)
    validate_plan(plan)
    if sha256_file(Path(__file__).resolve()) != plan["block_runner_sha256"]:
        raise RuntimeError("block runner source digest mismatch")
    if sha256_file(args.benchmark_script) != plan["benchmark_sha256"]:
        raise RuntimeError("DINO latency benchmark digest mismatch")
    latency_stats = load_latency_stats_module(
        args.latency_stats_module,
        plan["latency_stats_sha256"],
    )
    tao_job_id, output_root = validate_sdk_output_root(args.output_root)
    allocation_root = (
        output_root
        / "dino_moo_phase2_20260728"
        / "post_front_matched"
        / plan["manifest_id"]
        / plan["allocation_id"]
    )
    result_path = allocation_root / "allocation_result.json"
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "preflight",
        "started_at_utc": timestamp(),
        "manifest_id": plan["manifest_id"],
        "manifest_sha256": plan["manifest_sha256"],
        "schedule_sha256": plan["schedule_sha256"],
        "allocation_id": plan["allocation_id"],
        "allocation_index": plan["allocation_index"],
        "design_row_index": plan["design_row_index"],
        "block_plan_sha256": plan["block_plan_sha256"],
        "tao_job_id": tao_job_id,
        "sdk_job_scoped_result_root": str(output_root),
        "hostname": socket.gethostname(),
        "feeds_final_selection": False,
        "feeds_reselection": False,
        "selection_time_objective_replacement_permitted": False,
        "candidate_runs": [],
    }
    atomic_json(result_path, result)
    try:
        result["hardware"] = validate_hardware(plan["expected_hardware"])
        (
            result["verified_config_sha256"],
            result["verified_checkpoint_sha256"],
        ) = validate_configs_and_checkpoints(plan, staging_root)
    except Exception as error:
        result["status"] = "preflight_failure"
        result["error"] = f"{type(error).__name__}: {error}"
        result["completed_at_utc"] = timestamp()
        atomic_json(result_path, result)
        raise

    result["status"] = "running"
    atomic_json(result_path, result)
    protocol = plan["latency_protocol"]
    for candidate in plan["candidates"]:
        candidate_output_root = (
            allocation_root / "candidates" / candidate["run_label"]
        )
        command = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=8",
            str(args.benchmark_script),
            "--config",
            str(staging_root / candidate["config_relative_path"]),
            "--checkpoint",
            candidate["checkpoint_path"],
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
            "candidate_id": candidate["candidate_id"],
            "position": candidate["position"],
            "run_label": candidate["run_label"],
            "checkpoint_path": candidate["checkpoint_path"],
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "resolved_model_spec_sha256": candidate[
                "resolved_model_spec_sha256"
            ],
            "candidate_table_record_sha256": candidate[
                "candidate_table_record_sha256"
            ],
            "config_sha256": candidate["config_sha256"],
            "config_path": str(
                staging_root / candidate["config_relative_path"]
            ),
            "status": "running",
            "started_at_utc": timestamp(),
            "raw_samples_dir": str(
                candidate_output_root / tao_job_id / "latency"
            ),
        }
        result["candidate_runs"].append(run)
        atomic_json(result_path, result)
        completed = subprocess.run(command, check=False)
        run["exit_code"] = completed.returncode
        run["completed_at_utc"] = timestamp()
        if completed.returncode != 0:
            run["status"] = "failure"
            result["status"] = "failure"
            result["failed_candidate_id"] = candidate["candidate_id"]
            result["completed_at_utc"] = timestamp()
            atomic_json(result_path, result)
            return completed.returncode
        try:
            run["latency_validation"] = validate_candidate_latency(
                Path(run["raw_samples_dir"]),
                protocol,
                latency_stats,
            )
        except Exception as error:
            run["status"] = "validation_failure"
            run["validation_error"] = (
                f"{type(error).__name__}: {error}"
            )
            result["status"] = "failure"
            result["failed_candidate_id"] = candidate["candidate_id"]
            result["completed_at_utc"] = timestamp()
            atomic_json(result_path, result)
            raise
        run["status"] = "success"
        atomic_json(result_path, result)

    result["status"] = "success"
    result["completed_at_utc"] = timestamp()
    atomic_json(result_path, result)
    print(
        "TAO_DINO_POST_FRONT_MATCHED_BLOCK_COMPLETE "
        f"allocation_id={plan['allocation_id']} "
        f"candidates={len(plan['candidates'])}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
