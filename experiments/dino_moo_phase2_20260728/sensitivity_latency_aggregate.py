#!/usr/bin/env python3

"""Provenance-bound aggregation for matched DINO sensitivity latency blocks."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import yaml

from sensitivity_latency_common import (
    DEFAULT_MANIFEST,
    build_profiles,
    build_schedule,
    canonical_bytes,
    load_accuracy_artifact,
    load_checkpoint_artifact,
    load_contract,
    sha256_file,
    sha256_value,
)
from sensitivity_latency_launcher import (
    build_block_plan,
    evaluation_config,
    staged_command,
    validate_sources,
    yaml_payload,
)


HERE = Path(__file__).resolve().parent
AUTOML_SRC = HERE.parent.parent / "src"
if str(AUTOML_SRC) not in sys.path:
    sys.path.insert(0, str(AUTOML_SRC))

from tao_automl.latency_stats import (  # noqa: E402
    LatencyProtocol,
    LatencyValidityThresholds,
    aggregate_synchronized_latency,
)


EXPECTED_RANKS = 8
EXPECTED_PROFILES = 14
EXPECTED_ALLOCATIONS = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint-artifact", type=Path, required=True)
    parser.add_argument("--checkpoint-artifact-sha256", required=True)
    parser.add_argument("--accuracy-artifact", type=Path, required=True)
    parser.add_argument("--accuracy-artifact-sha256", required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--submission-ledger-sha256", required=True)
    parser.add_argument("--sdk-state", type=Path, required=True)
    parser.add_argument("--secrets-env", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def strict_json_bytes(raw: bytes, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> Any:
        raise ValueError(f"{label}: invalid JSON constant {value}")

    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=invalid_constant,
    )


def read_hashed_json(path: Path, label: str) -> tuple[Any, str]:
    raw = path.read_bytes()
    return strict_json_bytes(raw, label), hashlib.sha256(raw).hexdigest()


def load_env_file(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"required secrets env file not found: {path}")
    loaded = []
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"unsupported env line {line_number}")
        key, encoded = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ValueError(f"invalid env key on line {line_number}")
        tokens = shlex.split(encoded, comments=True, posix=True)
        if len(tokens) > 1:
            raise ValueError(f"unsupported env syntax on line {line_number}")
        os.environ.setdefault(key, tokens[0] if tokens else "")
        loaded.append(key)
    return sorted(loaded)


def ssh_target() -> str:
    host = os.environ.get("SLURM_HOSTNAME", "").split(",", 1)[0].strip()
    user = os.environ.get("SLURM_USER", "").strip()
    if not host or not user:
        raise RuntimeError("SLURM_USER and SLURM_HOSTNAME are required")
    return f"{user}@{host}"


def remote_output(command: str, *, timeout: int = 120) -> str:
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    key = os.environ.get("SSH_KEY_PATH")
    if key:
        ssh.extend(["-i", key])
    ssh.extend([ssh_target(), command])
    return subprocess.run(
        ssh,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    ).stdout


def local_lustre_path(uri: str) -> Path:
    if uri.startswith("lustre://"):
        value = uri.removeprefix("lustre://")
        return Path(value if value.startswith("/") else f"/{value}")
    if uri.startswith("/"):
        return Path(uri)
    raise ValueError(f"expected absolute Lustre result URI, got {uri!r}")


def sdk_db_path(state_path: Path) -> Path:
    if state_path.name.endswith(".json"):
        return state_path.with_suffix(".db")
    return Path(str(state_path) + ".db")


def sqlite_snapshot_sha256(database: Path) -> str:
    with tempfile.NamedTemporaryFile(
        prefix="sensitivity_sdk_snapshot_", suffix=".db", delete=False
    ) as temporary:
        snapshot = Path(temporary.name)
    try:
        source = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        target = sqlite3.connect(snapshot)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        return sha256_file(snapshot)
    finally:
        snapshot.unlink(missing_ok=True)


def latency_protocol(contract: dict[str, Any]) -> LatencyProtocol:
    source = contract["latency_protocol"]
    thresholds = source["validity_thresholds"]
    return LatencyProtocol(
        warmup_iterations=source["warmup_iterations"],
        timed_iterations=source["timed_iterations"],
        repeated_rounds=source["repeated_rounds"],
        tail_percentile=source["tail_percentile"],
        bootstrap_resamples=source["bootstrap_resamples"],
        bootstrap_confidence_level=source["bootstrap_confidence_level"],
        bootstrap_seed=source["bootstrap_seed"],
        expected_devices=tuple(str(index) for index in range(EXPECTED_RANKS)),
        validity_thresholds=LatencyValidityThresholds(
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


def regenerate_execution(
    manifest_path: Path,
    contract: dict[str, Any],
    one: dict[str, Any],
    profiles: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    artifact_entries: dict[tuple[int, str], dict[str, Any]],
    manifest_sha256: str,
    checkpoint_artifact_sha256: str,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    benchmark, template_path, source_checks = validate_sources(
        manifest_path, contract
    )
    template = yaml.safe_load(template_path.read_text())
    configs: dict[tuple[int, str], bytes] = {}
    for seed in one["design"]["seeds"]:
        for profile in profiles:
            artifact = artifact_entries[(seed, profile["profile_id"])]
            config = evaluation_config(
                contract,
                template,
                profile,
                artifact["checkpoint_path"],
                seed,
            )
            if sha256_value(config["model"]) != profile[
                "resolved_model_spec_sha256"
            ]:
                raise RuntimeError("resolved evaluation model mapping drift")
            configs[(seed, profile["profile_id"])] = yaml_payload(config)
    plans = {}
    summaries = {}
    for block in schedule:
        plan = build_block_plan(
            contract,
            manifest_sha256,
            checkpoint_artifact_sha256,
            profiles,
            artifact_entries,
            block,
            configs,
        )
        _command, summary = staged_command(
            benchmark, block, plan, configs
        )
        plans[block["allocation_id"]] = plan
        summaries[block["allocation_id"]] = summary
    return plans, summaries, source_checks


def load_submission_ledger(
    path: Path,
    expected_file_sha256: str,
    contract: dict[str, Any],
    manifest_sha256: str,
    checkpoint_artifact_sha256: str,
    schedule_sha256: str,
    schedule: list[dict[str, Any]],
    expected_summaries: dict[str, dict[str, Any]],
    current_source_checks: dict[str, str],
) -> tuple[dict[str, Any], str]:
    ledger, actual_sha256 = read_hashed_json(path, "submission ledger")
    if actual_sha256 != expected_file_sha256:
        raise RuntimeError("immutable submission ledger digest mismatch")
    if (
        ledger.get("schema_version") != 1
        or ledger.get("status") != "complete"
        or ledger.get("phase") != "sensitivity_latency_blocks"
        or ledger.get("manifest_id") != contract["manifest_id"]
        or ledger.get("manifest_sha256") != manifest_sha256
        or ledger.get("checkpoint_artifact_sha256")
        != checkpoint_artifact_sha256
        or ledger.get("schedule_sha256") != schedule_sha256
        or ledger.get("expected_allocation_count") != EXPECTED_ALLOCATIONS
        or ledger.get("allocation_count") != EXPECTED_ALLOCATIONS
        or ledger.get("feeds_final_selection") is not False
        or ledger.get("manual_promotion_permitted") is not False
    ):
        raise ValueError("submission ledger identity or policy mismatch")
    recorded_sources = ledger.get("source_checks")
    if not isinstance(recorded_sources, dict):
        raise ValueError("submission ledger source checks are absent")
    for key, expected in current_source_checks.items():
        if recorded_sources.get(key) != expected:
            raise ValueError(f"submission ledger source drift: {key}")
    if recorded_sources.get("submission_source_state") != "tracked_and_clean":
        raise ValueError("submission source was not recorded tracked and clean")

    expected_blocks = {
        block["allocation_id"]: block for block in schedule
    }
    submissions = ledger.get("submissions")
    if not isinstance(submissions, list) or len(submissions) != EXPECTED_ALLOCATIONS:
        raise ValueError("ledger must contain exactly nine submissions")
    tao_ids: set[str] = set()
    slurm_ids: set[str] = set()
    by_allocation: dict[str, dict[str, Any]] = {}
    for item in submissions:
        allocation_id = item.get("allocation_id")
        block = expected_blocks.get(allocation_id)
        summary = expected_summaries.get(allocation_id)
        if block is None or summary is None or allocation_id in by_allocation:
            raise ValueError("ledger allocation identity mismatch")
        for key, value in summary.items():
            if item.get(key) != value:
                raise ValueError(f"{allocation_id}: submitted {key} drift")
        for key in ("seed", "repeat_index", "williams_row_index"):
            if item.get(key) != block[key]:
                raise ValueError(f"{allocation_id}: submitted {key} drift")
        if item.get("profile_order") != block["profile_order"]:
            raise ValueError(f"{allocation_id}: submitted order drift")
        tao_id = item.get("tao_job_id")
        slurm_id = str(item.get("slurm_job_id", ""))
        if (
            not isinstance(tao_id, str)
            or not tao_id
            or not slurm_id.isdigit()
            or tao_id in tao_ids
            or slurm_id in slurm_ids
        ):
            raise ValueError("ledger TAO/SLURM identities must be distinct")
        uri = item.get("sdk_results_uri")
        if not isinstance(uri, str) or local_lustre_path(uri).name != tao_id:
            raise ValueError(f"{allocation_id}: ledger SDK result URI invalid")
        tao_ids.add(tao_id)
        slurm_ids.add(slurm_id)
        by_allocation[allocation_id] = item
    if set(by_allocation) != set(expected_blocks):
        raise ValueError("ledger does not cover the frozen nine-block schedule")
    supersedes = ledger.get("supersedes")
    retry_entries = [
        item for item in by_allocation.values() if "retry_of" in item
    ]
    if supersedes is None and retry_entries:
        raise ValueError("retry entry is missing superseded-ledger provenance")
    if supersedes is not None:
        if not isinstance(supersedes, dict) or len(retry_entries) != 1:
            raise ValueError("retry ledger must replace exactly one allocation")
        allocation_id = supersedes.get("replaced_allocation_id")
        replacement = by_allocation.get(allocation_id)
        replaced = supersedes.get("replaced_submission")
        prior_path = Path(str(supersedes.get("ledger_path", ""))).resolve()
        prior_sha = supersedes.get("ledger_sha256")
        evidence_path = Path(
            str(supersedes.get("retry_evidence_path", ""))
        ).resolve()
        evidence_sha = supersedes.get("retry_evidence_sha256")
        prior_raw = prior_path.read_bytes()
        evidence, actual_evidence_sha = read_hashed_json(
            evidence_path, "retry evidence"
        )
        if (
            replacement is None
            or not isinstance(replaced, dict)
            or not isinstance(prior_sha, str)
            or hashlib.sha256(prior_raw).hexdigest() != prior_sha
            or not isinstance(evidence_sha, str)
            or actual_evidence_sha != evidence_sha
            or replacement.get("retry_of", {}).get("tao_job_id")
            != replaced.get("tao_job_id")
            or replacement.get("tao_job_id")
            != supersedes.get("replacement_tao_job_id")
            or str(replacement.get("slurm_job_id"))
            != str(supersedes.get("replacement_slurm_job_id"))
            or replacement.get("tao_job_id") == replaced.get("tao_job_id")
            or str(replacement.get("slurm_job_id"))
            == str(replaced.get("slurm_job_id"))
        ):
            raise ValueError("retry ledger provenance chain mismatch")
        if (
            evidence.get("allocation_id") != allocation_id
            or evidence.get("prior_tao_job_id")
            != replaced.get("tao_job_id")
            or str(evidence.get("prior_slurm_job_id", ""))
            != str(replaced.get("slurm_job_id", ""))
            or evidence.get("reason_code")
            != supersedes.get("retry_reason_code")
            or evidence.get("retry_permitted") is not True
            or evidence.get("partial_measurements_reusable") is not False
        ):
            raise ValueError("retry evidence content mismatch")
    ledger["submissions"] = [
        by_allocation[block["allocation_id"]] for block in schedule
    ]
    return ledger, actual_sha256


def slurm_accounting(
    submissions: list[dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    slurm_ids = [str(item["slurm_job_id"]) for item in submissions]
    fields = [
        "JobIDRaw",
        "JobName",
        "State",
        "ExitCode",
        "DerivedExitCode",
        "Partition",
        "Account",
        "NNodes",
        "NTasks",
        "AllocTRES",
        "NodeList",
        "Start",
        "End",
        "ElapsedRaw",
    ]
    command = " ".join(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            shlex.quote(",".join(slurm_ids)),
            f"--format={','.join(fields)}",
        ]
    )
    rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for line in remote_output(command).splitlines():
        values = line.rstrip().split("|")
        if len(values) < len(fields) or values[0] not in slurm_ids:
            continue
        rows[values[0]].append(dict(zip(fields, values, strict=True)))
    result = {}
    runtime = contract["runtime_contract"]
    for slurm_id in slurm_ids:
        candidates = rows.get(slurm_id, [])
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected exactly one sacct root row for {slurm_id}"
            )
        row = candidates[0]
        state = row["State"].split("+", 1)[0].split(None, 1)[0]
        gpu_match = re.search(
            r"(?:^|,)(?:gres/)?gpu(?:=[^,:]+)?[:=](\d+)(?:,|$)",
            row["AllocTRES"],
        )
        if (
            state != "COMPLETED"
            or row["ExitCode"] != "0:0"
            or row["Partition"] != runtime["partition"]
            or row["Account"] != runtime["account"]
            or row["NNodes"] != "1"
            or gpu_match is None
            or int(gpu_match.group(1)) != EXPECTED_RANKS
            or not row["NodeList"]
        ):
            raise RuntimeError(f"SLURM completion/topology mismatch: {slurm_id}")
        expanded = [
            item.strip()
            for item in remote_output(
                f"scontrol show hostnames {shlex.quote(row['NodeList'])}"
            ).splitlines()
            if item.strip()
        ]
        if len(expanded) != 1:
            raise RuntimeError(f"{slurm_id}: expected exactly one allocated node")
        result[slurm_id] = {
            "slurm_job_id": slurm_id,
            "job_name": row["JobName"],
            "state": state,
            "exit_code": row["ExitCode"],
            "derived_exit_code": row["DerivedExitCode"],
            "partition": row["Partition"],
            "account": row["Account"],
            "node_count": 1,
            "task_count": row["NTasks"],
            "alloc_tres": row["AllocTRES"],
            "node_list": row["NodeList"],
            "expanded_nodes": expanded,
            "start": row["Start"],
            "end": row["End"],
            "elapsed_seconds": row["ElapsedRaw"],
        }
    return result


def inspect_sdk_jobs(
    contract: dict[str, Any],
    ledger: dict[str, Any],
    state_path: Path,
    accounting: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    database = sdk_db_path(state_path)
    if not state_path.is_file():
        raise FileNotFoundError(f"SDK monitor state missing: {state_path}")
    if not database.is_file():
        raise FileNotFoundError(f"SDK durable database missing: {database}")
    before_sha = sqlite_snapshot_sha256(database)
    sdk_path = contract["runtime_contract"]["sdk_path"]
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)
    from tao_sdk.platforms.slurm import SlurmSDK
    from tao_sdk.job_store import JobStore

    runtime_contract = contract["runtime_contract"]
    # Do not construct SlurmSDK normally: its monitor can poll failed jobs and
    # trigger the SDK's automatic infrastructure retry path.  Bind the exact
    # get_job_results_dir implementation to a read-only aggregation facade
    # backed by the durable store, with no handler and no monitor thread.
    sdk = SlurmSDK.__new__(SlurmSDK)
    sdk._store = JobStore(db_path=database)
    jobs = []
    try:
        for item in ledger["submissions"]:
            allocation_id = item["allocation_id"]
            tao_id = item["tao_job_id"]
            row = sdk._store.get_job(tao_id)
            if row is None:
                raise RuntimeError(f"{allocation_id}: absent from SDK state")
            specs = row.get("specs", {})
            if isinstance(specs, str):
                specs = strict_json_bytes(
                    specs.encode("utf-8"), f"{allocation_id} SDK specs"
                )
            runtime = specs.get("_slurm_runtime", {})
            artifacts = specs.get("_tao_artifacts", {})
            slurm_id = str(item["slurm_job_id"])
            sdk_uri = sdk.get_job_results_dir(tao_id)
            if (
                row.get("backend_type") != "slurm"
                or row.get("status") != "Complete"
                or row.get("image") != runtime_contract["sqsh_path"]
                or str(runtime.get("slurm_job_id", "")) != slurm_id
                or int(runtime.get("retry_count", -1)) != 0
                or runtime.get("failed_slurm_job_ids") != []
                or runtime.get("launch_uncertain") is not False
                or artifacts.get("kind") != "lustre"
                or artifacts.get("root") != sdk_uri
                or sdk_uri != item["sdk_results_uri"]
            ):
                raise RuntimeError(
                    f"{allocation_id}: durable SDK identity/state mismatch"
                )
            result_root = local_lustre_path(sdk_uri)
            if (
                result_root.name != tao_id
                or result_root.parent.name != "results"
            ):
                raise RuntimeError(
                    f"{allocation_id}: SDK result root is not job-scoped"
                )
            sacct = accounting[slurm_id]
            if sacct["job_name"] != row.get("backend_job_id"):
                raise RuntimeError(
                    f"{allocation_id}: SDK/sacct job-name mismatch"
                )
            jobs.append(
                {
                    "allocation_id": allocation_id,
                    "tao_job_id": tao_id,
                    "slurm_job_id": slurm_id,
                    "sdk_status": row["status"],
                    "sdk_results_uri": sdk_uri,
                    "sdk_job_scoped_result_root": str(result_root),
                    "sdk_backend_job_id": row["backend_job_id"],
                    "sdk_runtime_revision": runtime.get("revision"),
                    "sdk_retry_count": runtime.get("retry_count"),
                    "sdk_failed_slurm_job_ids": runtime.get(
                        "failed_slurm_job_ids", []
                    ),
                    "sdk_launch_uncertain": runtime.get("launch_uncertain"),
                    "scheduler": sacct,
                    "complete": True,
                }
            )
    finally:
        sdk._store.close()
    after_sha = sqlite_snapshot_sha256(database)
    if after_sha != before_sha:
        raise RuntimeError("read-only SDK inspection mutated durable state")
    return jobs, {
        "state_path": str(state_path),
        "state_file_sha256": (
            sha256_file(state_path) if state_path.is_file() else None
        ),
        "database_path": str(database),
        "consistent_sqlite_snapshot_sha256": before_sha,
        "read_only_snapshot_stable": True,
    }


def result_path_for_job(
    job: dict[str, Any],
    contract: dict[str, Any],
    block: dict[str, Any],
) -> Path:
    return (
        Path(job["sdk_job_scoped_result_root"])
        / "dino_moo_phase2_20260728"
        / "sensitivity_latency"
        / contract["manifest_id"]
        / f"seed_{block['seed']:06d}"
        / block["allocation_id"]
        / "allocation_result.json"
    )


def validate_input_identity(record: dict[str, Any], contract: dict[str, Any]) -> str:
    metadata = record.get("benchmark_inputs")
    if not isinstance(metadata, dict):
        raise ValueError("missing benchmark_inputs")
    batches = metadata.get("batches")
    expected_count = contract["latency_protocol"]["preloaded_batches"]
    if (
        not isinstance(batches, list)
        or len(batches) != expected_count
        or metadata.get("batch_count") != expected_count
        or metadata.get("example_count") != expected_count
    ):
        raise ValueError("benchmark input preload evidence mismatch")
    shapes = contract["latency_protocol"]["fixed_preprocessed_shapes"]
    for index, batch in enumerate(batches):
        if (
            batch.get("batch_index") != index
            or batch.get("batch_size") != 1
            or batch.get("model_input", {}).get("shape")
            != shapes["model_input"]
            or batch.get("image_tensor_shape") != shapes["image_tensor"]
            or batch.get("padding_mask", {}).get("shape")
            != shapes["padding_mask"]
        ):
            raise ValueError("benchmark input identity/shape mismatch")
    identity = metadata.get("identity_sha256")
    payload = {
        "schema_version": metadata.get("schema_version"),
        "batch_count": metadata.get("batch_count"),
        "example_count": metadata.get("example_count"),
        "batches": batches,
    }
    if identity != hashlib.sha256(canonical_bytes(payload)).hexdigest():
        raise ValueError("benchmark input canonical digest mismatch")
    return identity


def validate_rank_record(
    record: dict[str, Any],
    *,
    rank: int,
    contract: dict[str, Any],
    checkpoint_path: str,
    config_path: str,
    allocation_hostname: str,
) -> tuple[str, tuple[Any, ...], str]:
    if (
        record.get("rank") != rank
        or record.get("local_rank") != rank
        or record.get("world_size") != EXPECTED_RANKS
        or record.get("checkpoint") != checkpoint_path
        or record.get("config_path") != config_path
    ):
        raise ValueError(f"rank {rank} identity/config/checkpoint mismatch")
    expected_protocol = contract["latency_protocol"]
    actual = record.get("protocol", {})
    checks = {
        "warmup_iterations": expected_protocol["warmup_iterations"],
        "timed_iterations": expected_protocol["timed_iterations"],
        "repeated_rounds": expected_protocol["repeated_rounds"],
        "preloaded_batches": expected_protocol["preloaded_batches"],
        "batch_size_per_gpu": 1,
        "precision": "fp32",
        "tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "timed_scope": "model_forward_plus_dino_gpu_postprocess",
        "synchronization": "cuda_sync_each_sample_and_nccl_barrier",
        "seed": expected_protocol["benchmark_seed"],
    }
    for key, expected in checks.items():
        if actual.get(key) != expected:
            raise ValueError(f"rank benchmark protocol mismatch: {key}")
    excluded = {
        "checkpoint_load",
        "disk_io",
        "decode_resize_normalize",
        "host_to_device_transfer",
        "coco_accumulation",
        "distributed_gather",
    }
    if set(actual.get("excluded_scope", [])) != excluded:
        raise ValueError("rank excluded timing scope mismatch")
    runtime_contract = contract["runtime_contract"]
    hardware = record.get("hardware", {})
    if (
        hardware.get("hostname") != allocation_hostname
        or hardware.get("gpu_name") != runtime_contract["required_gpu_name"]
        or hardware.get("compute_capability")
        != runtime_contract["required_compute_capability"]
        or hardware.get("total_memory_bytes")
        != runtime_contract["required_total_memory_bytes"]
    ):
        raise ValueError("rank hardware/hostname mismatch")
    nvidia_smi = hardware.get("nvidia_smi", "")
    if not isinstance(nvidia_smi, str) or nvidia_smi.startswith("unavailable:"):
        raise ValueError("nvidia-smi evidence unavailable")
    gpu_uuid = nvidia_smi.split(",", 1)[0].strip()
    if not gpu_uuid:
        raise ValueError("GPU UUID is empty")
    runtime = record.get("runtime", {})
    if (
        runtime.get("torch") != runtime_contract["required_torch"]
        or runtime.get("cuda") != runtime_contract["required_cuda"]
        or runtime.get("cudnn") != runtime_contract["required_cudnn"]
    ):
        raise ValueError("rank runtime mismatch")
    signature = (
        runtime.get("python"),
        runtime.get("torch"),
        runtime.get("cuda"),
        runtime.get("cudnn"),
    )
    if any(value is None for value in signature):
        raise ValueError("rank runtime signature incomplete")
    return validate_input_identity(record, contract), signature, gpu_uuid


def stats_record(stats: Any) -> dict[str, Any]:
    return {
        "median_ms": stats.median_ms,
        "p95_ms": stats.tail_latency_ms,
        "bootstrap_median_ci95_ms": list(stats.bootstrap_median_ci_ms),
        "mad_ms": stats.mad_ms,
        "iqr_ms": stats.iqr_ms,
        "robust_cv": stats.robust_cv,
        "round_median_range_ms": stats.round_median_range_ms,
        "round_drift_ms": stats.round_drift_ms,
        "device_median_range_ms": stats.device_median_range_ms,
        "raw_sample_count_total": stats.raw_sample_count_total,
        "samples_per_device": stats.samples_per_device,
        "is_valid": stats.is_valid,
        "invalid_reasons": list(stats.invalid_reasons),
    }


def aggregate_job_results(
    contract: dict[str, Any],
    profiles: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    artifact_entries: dict[tuple[int, str], dict[str, Any]],
    accuracy_entries: dict[tuple[int, str], dict[str, Any]],
    jobs: list[dict[str, Any]],
    expected_plans: dict[str, dict[str, Any]],
    manifest_sha256: str,
    checkpoint_artifact_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    protocol = latency_protocol(contract)
    blocks = {block["allocation_id"]: block for block in schedule}
    profile_by_id = {
        profile["profile_id"]: profile for profile in profiles
    }
    input_identities: dict[int, str] = {}
    runtime_signature: tuple[Any, ...] | None = None
    gpu_uuid_by_allocation_rank: dict[str, dict[int, str]] = defaultdict(dict)
    measurements = []
    evidence = []

    for job in jobs:
        allocation_id = job["allocation_id"]
        block = blocks[allocation_id]
        plan = expected_plans[allocation_id]
        result_path = result_path_for_job(job, contract, block)
        result, result_sha256 = read_hashed_json(
            result_path, f"{allocation_id} result"
        )
        expected_result_values = {
            "schema_version": 1,
            "status": "success",
            "manifest_id": contract["manifest_id"],
            "manifest_sha256": manifest_sha256,
            "checkpoint_artifact_sha256": checkpoint_artifact_sha256,
            "schedule_sha256": contract["design"]["schedule_sha256"],
            "allocation_id": allocation_id,
            "seed": block["seed"],
            "repeat_index": block["repeat_index"],
            "williams_row_index": block["williams_row_index"],
            "block_plan_sha256": plan["block_plan_sha256"],
            "tao_job_id": job["tao_job_id"],
            "sdk_job_scoped_result_root": job[
                "sdk_job_scoped_result_root"
            ],
            "feeds_final_selection": False,
            "manual_promotion_permitted": False,
        }
        for key, expected in expected_result_values.items():
            if result.get(key) != expected:
                raise ValueError(f"{allocation_id}: result {key} drift")
        hostname = result.get("hostname")
        if (
            not isinstance(hostname, str)
            or hostname
            != job["scheduler"]["expanded_nodes"][0]
        ):
            raise ValueError(f"{allocation_id}: scheduler hostname mismatch")
        result_output = result.get("output_contract", {})
        if (
            result_output.get("root_env") != "TAO_RESULTS_ROOT"
            or result_output.get("job_scope_env") != "TAO_JOB_ID"
            or result_output.get("root")
            != job["sdk_job_scoped_result_root"]
        ):
            raise ValueError(f"{allocation_id}: result output contract mismatch")
        hardware = result.get("hardware", {})
        expected_runtime = contract["runtime_contract"]
        devices = hardware.get("devices", [])
        if (
            not isinstance(devices, list)
            or len(devices) != EXPECTED_RANKS
            or [device.get("index") for device in devices]
            != list(range(EXPECTED_RANKS))
            or any(
                device.get("name") != expected_runtime["required_gpu_name"]
                or device.get("compute_capability")
                != expected_runtime["required_compute_capability"]
                or device.get("total_memory_bytes")
                != expected_runtime["required_total_memory_bytes"]
                for device in devices
            )
            or hardware.get("runtime", {}).get("torch")
            != expected_runtime["required_torch"]
            or hardware.get("runtime", {}).get("cuda")
            != expected_runtime["required_cuda"]
            or hardware.get("runtime", {}).get("cudnn")
            != expected_runtime["required_cudnn"]
        ):
            raise ValueError(f"{allocation_id}: allocation hardware mismatch")
        runs = result.get("profile_runs")
        if (
            not isinstance(runs, list)
            or len(runs) != EXPECTED_PROFILES
            or [run.get("profile_id") for run in runs]
            != block["profile_order"]
            or [run.get("position") for run in runs]
            != list(range(EXPECTED_PROFILES))
            or any(
                run.get("status") != "success"
                or run.get("exit_code") != 0
                or run.get("seed") != block["seed"]
                for run in runs
            )
        ):
            raise ValueError(f"{allocation_id}: incomplete/reordered block")
        expected_config_digests = {
            item["profile_id"]: item["config_sha256"]
            for item in plan["profiles"]
        }
        if result.get("verified_config_sha256") != expected_config_digests:
            raise ValueError(f"{allocation_id}: verified config digest drift")

        result_root = Path(job["sdk_job_scoped_result_root"])
        block_measurements = []
        raw_file_count = 0
        raw_digest_by_profile: dict[str, dict[str, str]] = {}
        for run in runs:
            profile_id = run["profile_id"]
            expected_profile = plan["profiles"][run["position"]]
            artifact = artifact_entries[(block["seed"], profile_id)]
            if (
                expected_profile["profile_id"] != profile_id
                or run.get("run_label") != expected_profile["run_label"]
                or run.get("config_sha256")
                != expected_profile["config_sha256"]
                or run.get("checkpoint_path")
                != artifact["checkpoint_path"]
                or run.get("checkpoint_sha256")
                != artifact["checkpoint_sha256"]
                or run.get("checkpoint_source_profile_id")
                != artifact["checkpoint_source_profile_id"]
                or run.get("resolved_model_spec_sha256")
                != profile_by_id[profile_id]["resolved_model_spec_sha256"]
            ):
                raise ValueError(
                    f"{allocation_id}/{profile_id}: run provenance mismatch"
                )
            raw_dir = (
                result_path.parent
                / "profiles"
                / run["run_label"]
                / job["tao_job_id"]
                / "latency"
            )
            if run.get("raw_samples_dir") != str(raw_dir):
                raise ValueError(
                    f"{allocation_id}/{profile_id}: raw path drift"
                )
            samples = {
                round_index: {}
                for round_index in range(protocol.repeated_rounds)
            }
            rank_digests = {}
            for rank in range(EXPECTED_RANKS):
                rank_path = raw_dir / f"rank_{rank}.json"
                record, digest = read_hashed_json(
                    rank_path,
                    f"{allocation_id}/{profile_id}/rank_{rank}",
                )
                rank_digests[str(rank)] = digest
                raw_file_count += 1
                identity, signature, gpu_uuid = validate_rank_record(
                    record,
                    rank=rank,
                    contract=contract,
                    checkpoint_path=artifact["checkpoint_path"],
                    config_path=expected_profile["config_path"],
                    allocation_hostname=hostname,
                )
                prior_input = input_identities.setdefault(rank, identity)
                if identity != prior_input:
                    raise ValueError("benchmark input identity drift")
                if runtime_signature is None:
                    runtime_signature = signature
                elif runtime_signature != signature:
                    raise ValueError("runtime signature drift")
                prior_uuid = gpu_uuid_by_allocation_rank[
                    allocation_id
                ].setdefault(rank, gpu_uuid)
                if prior_uuid != gpu_uuid:
                    raise ValueError("GPU rank mapping changed within allocation")
                rank_samples = record.get("samples_ms")
                if (
                    not isinstance(rank_samples, list)
                    or len(rank_samples) != protocol.repeated_rounds
                ):
                    raise ValueError("rank sample round count mismatch")
                for round_index, values in enumerate(rank_samples):
                    samples[round_index][str(rank)] = values
            stats = aggregate_synchronized_latency(samples, protocol)
            if not stats.is_valid:
                raise ValueError(
                    f"{allocation_id}/{profile_id}: {stats.validity_reason}"
                )
            raw_digest_by_profile[profile_id] = rank_digests
            accuracy = accuracy_entries[(block["seed"], profile_id)]
            block_measurements.append(
                {
                    "allocation_id": allocation_id,
                    "tao_job_id": job["tao_job_id"],
                    "slurm_job_id": job["slurm_job_id"],
                    "node_list": job["scheduler"]["node_list"],
                    "hostname": hostname,
                    "seed": block["seed"],
                    "repeat_index": block["repeat_index"],
                    "williams_row_index": block["williams_row_index"],
                    "profile_id": profile_id,
                    "axis": profile_by_id[profile_id]["axis"],
                    "level": profile_by_id[profile_id]["level"],
                    "position": run["position"],
                    "map50": float(accuracy["mAP50"]),
                    "checkpoint_sha256": artifact["checkpoint_sha256"],
                    "resolved_model_spec_sha256": artifact[
                        "resolved_model_spec_sha256"
                    ],
                    "config_sha256": expected_profile["config_sha256"],
                    "raw_rank_file_sha256_by_rank": rank_digests,
                    **stats_record(stats),
                }
            )
        if raw_file_count != EXPECTED_PROFILES * EXPECTED_RANKS:
            raise RuntimeError(f"{allocation_id}: raw rank file count drift")
        if (
            len(set(gpu_uuid_by_allocation_rank[allocation_id].values()))
            != EXPECTED_RANKS
        ):
            raise ValueError(f"{allocation_id}: GPUs are not eight distinct UUIDs")
        measurements.extend(block_measurements)
        evidence.append(
            {
                "allocation_id": allocation_id,
                "tao_job_id": job["tao_job_id"],
                "slurm_job_id": job["slurm_job_id"],
                "node_list": job["scheduler"]["node_list"],
                "hostname": hostname,
                "sdk_job_scoped_result_root": str(result_root),
                "result_path": str(result_path),
                "result_sha256": result_sha256,
                "block_plan_sha256": plan["block_plan_sha256"],
                "raw_rank_file_count": raw_file_count,
                "raw_rank_file_sha256_by_profile": raw_digest_by_profile,
                "gpu_uuid_sha256_by_rank": {
                    str(rank): hashlib.sha256(uuid.encode()).hexdigest()
                    for rank, uuid in sorted(
                        gpu_uuid_by_allocation_rank[allocation_id].items()
                    )
                },
            }
        )
    if len(measurements) != EXPECTED_ALLOCATIONS * EXPECTED_PROFILES:
        raise RuntimeError("expected exactly 126 valid measurements")
    consistency = {
        "hardware_contract": "pass",
        "runtime_contract": "pass",
        "protocol_contract": "pass",
        "benchmark_input_identity": "pass",
        "rank_files_per_profile": EXPECTED_RANKS,
        "runtime_signature": list(runtime_signature or ()),
        "benchmark_input_identity_sha256_by_rank": {
            str(rank): digest
            for rank, digest in sorted(input_identities.items())
        },
        "distinct_gpu_count_by_allocation": {
            allocation_id: len(set(by_rank.values()))
            for allocation_id, by_rank in sorted(
                gpu_uuid_by_allocation_rank.items()
            )
        },
        "node_frequency": dict(
            sorted(
                {
                    node: sum(
                        item["hostname"] == node for item in evidence
                    )
                    for node in {item["hostname"] for item in evidence}
                }.items()
            )
        ),
        "distinct_tao_job_count": len(
            {item["tao_job_id"] for item in evidence}
        ),
        "distinct_slurm_allocation_count": len(
            {item["slurm_job_id"] for item in evidence}
        ),
        "position_balance": {
            profile["profile_id"]: sorted(
                row["position"]
                for row in measurements
                if row["profile_id"] == profile["profile_id"]
            )
            for profile in profiles
        },
    }
    return measurements, evidence, consistency


def deterministic_bootstrap_ci(
    values: list[float],
    label: str,
    *,
    resamples: int = 5000,
    confidence: float = 0.95,
) -> list[float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("bootstrap values must be finite and non-empty")
    seed_material = f"{label}|{resamples}|{confidence}"
    seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    source = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(source), size=(resamples, len(source)))
    estimates = np.median(source[indices], axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(
        estimates,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    return [float(low), float(high)]


def hierarchical_paired_bootstrap_ci(
    effects_by_seed: dict[int, list[float]],
    label: str,
    *,
    resamples: int,
    confidence: float,
) -> list[float]:
    seeds = sorted(effects_by_seed)
    if len(seeds) != 3 or any(len(effects_by_seed[seed]) != 3 for seed in seeds):
        raise ValueError("hierarchical bootstrap requires 3x3 paired effects")
    if any(
        not math.isfinite(value)
        for seed in seeds
        for value in effects_by_seed[seed]
    ):
        raise ValueError("hierarchical paired effects must be finite")
    seed_material = f"{label}|hierarchical|{resamples}|{confidence}"
    rng = np.random.default_rng(
        int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16)
    )
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled_seed_indices = rng.integers(0, len(seeds), size=len(seeds))
        seed_estimates = []
        for seed_index in sampled_seed_indices:
            values = np.asarray(
                effects_by_seed[seeds[int(seed_index)]], dtype=np.float64
            )
            allocation_indices = rng.integers(
                0, len(values), size=len(values)
            )
            seed_estimates.append(float(np.median(values[allocation_indices])))
        estimates[index] = float(np.median(seed_estimates))
    alpha = 1.0 - confidence
    low, high = np.quantile(
        estimates,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    return [float(low), float(high)]


def distribution_summary(values: list[float], label: str) -> dict[str, Any]:
    if not values:
        raise ValueError("distribution must be non-empty")
    array = np.asarray(values, dtype=np.float64)
    q1, q3 = np.quantile(array, [0.25, 0.75], method="linear")
    median = float(np.median(array))
    return {
        "count": len(values),
        "values": values,
        "median": median,
        "mean": float(np.mean(array)),
        "sample_stdev": (
            float(statistics.stdev(values)) if len(values) > 1 else 0.0
        ),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "range": float(np.max(array) - np.min(array)),
        "mad": float(np.median(np.abs(array - median))),
        "iqr": float(q3 - q1),
        "bootstrap_median_ci95": deterministic_bootstrap_ci(values, label),
    }


def qualification_analysis(
    contract: dict[str, Any],
    one: dict[str, Any],
    profiles: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    accuracy_entries: dict[tuple[int, str], dict[str, Any]],
    measurements: list[dict[str, Any]],
) -> dict[str, Any]:
    by_key = {
        (row["allocation_id"], row["profile_id"]): row
        for row in measurements
    }
    reference_id = one["reference"]["profile_id"]
    seeds = list(one["design"]["seeds"])
    reference_ranges = {}
    for seed in seeds:
        medians = [
            by_key[(block["allocation_id"], reference_id)]["median_ms"]
            for block in schedule
            if block["seed"] == seed
        ]
        reference_ranges[str(seed)] = max(medians) - min(medians)
    historical_floor = float(
        contract["aggregation"]["historical_noise_floor_ms"]
    )
    new_floor = max(reference_ranges.values())
    effective_floor = max(historical_floor, new_floor)
    protocol = contract["latency_protocol"]
    decisions = []
    paired_by_seed = []
    profile_summaries = []

    for profile in profiles:
        profile_id = profile["profile_id"]
        rows = [
            row for row in measurements if row["profile_id"] == profile_id
        ]
        profile_summaries.append(
            {
                "profile_id": profile_id,
                "axis": profile["axis"],
                "level": profile["level"],
                "within_allocation_variability": {
                    "robust_cv": distribution_summary(
                        [row["robust_cv"] for row in rows],
                        f"{contract['manifest_id']}:{profile_id}:robust_cv",
                    ),
                    "round_median_range_ms": distribution_summary(
                        [row["round_median_range_ms"] for row in rows],
                        f"{contract['manifest_id']}:{profile_id}:round_range",
                    ),
                    "device_median_range_ms": distribution_summary(
                        [row["device_median_range_ms"] for row in rows],
                        f"{contract['manifest_id']}:{profile_id}:device_range",
                    ),
                    "allocation_bootstrap_ci_width_ms": distribution_summary(
                        [
                            row["bootstrap_median_ci95_ms"][1]
                            - row["bootstrap_median_ci95_ms"][0]
                            for row in rows
                        ],
                        f"{contract['manifest_id']}:{profile_id}:ci_width",
                    ),
                },
                "between_allocation_variability": {
                    "median_ms": distribution_summary(
                        [row["median_ms"] for row in rows],
                        f"{contract['manifest_id']}:{profile_id}:median",
                    ),
                    "p95_ms": distribution_summary(
                        [row["p95_ms"] for row in rows],
                        f"{contract['manifest_id']}:{profile_id}:p95",
                    ),
                    "median_ms_by_seed": {
                        str(seed): distribution_summary(
                            [
                                row["median_ms"]
                                for row in rows
                                if row["seed"] == seed
                            ],
                            (
                                f"{contract['manifest_id']}:{profile_id}:"
                                f"seed:{seed}:median"
                            ),
                        )
                        for seed in seeds
                    },
                },
            }
        )
        if profile_id == reference_id:
            continue
        effects_by_seed: dict[int, list[float]] = {}
        p95_effects_by_seed: dict[int, list[float]] = {}
        accuracy_passes = []
        for seed in seeds:
            effects = []
            p95_effects = []
            for block in schedule:
                if block["seed"] != seed:
                    continue
                candidate = by_key[(block["allocation_id"], profile_id)]
                reference = by_key[(block["allocation_id"], reference_id)]
                effects.append(candidate["median_ms"] - reference["median_ms"])
                p95_effects.append(candidate["p95_ms"] - reference["p95_ms"])
            effects_by_seed[seed] = effects
            p95_effects_by_seed[seed] = p95_effects
            accuracy = float(accuracy_entries[(seed, profile_id)]["mAP50"])
            reference_accuracy = float(
                accuracy_entries[(seed, reference_id)]["mAP50"]
            )
            threshold = 0.98 * reference_accuracy
            accuracy_pass = accuracy >= threshold
            accuracy_passes.append(accuracy_pass)
            paired_by_seed.append(
                {
                    "profile_id": profile_id,
                    "axis": profile["axis"],
                    "level": profile["level"],
                    "seed": seed,
                    "allocation_median_differences_ms": effects,
                    "allocation_p95_differences_ms": p95_effects,
                    "median_paired_effect_ms": float(np.median(effects)),
                    "paired_effect_ci95_ms": deterministic_bootstrap_ci(
                        effects,
                        f"{contract['manifest_id']}:{profile_id}:{seed}",
                    ),
                    "median_p95_paired_effect_ms": float(
                        np.median(p95_effects)
                    ),
                    "paired_p95_effect_ci95_ms": deterministic_bootstrap_ci(
                        p95_effects,
                        (
                            f"{contract['manifest_id']}:{profile_id}:"
                            f"{seed}:p95"
                        ),
                    ),
                    "map50": accuracy,
                    "reference_map50": reference_accuracy,
                    "accuracy_threshold": threshold,
                    "accuracy_retention_pass": accuracy_pass,
                }
            )
        support = all(
            len(values) == 3
            and all(math.isfinite(value) for value in values)
            for values in effects_by_seed.values()
        )
        seed_medians = {
            str(seed): float(np.median(effects_by_seed[seed]))
            for seed in seeds
        }
        point_effect = float(np.median(list(seed_medians.values())))
        hierarchical_ci = hierarchical_paired_bootstrap_ci(
            effects_by_seed,
            f"{contract['manifest_id']}:{profile_id}",
            resamples=protocol["bootstrap_resamples"],
            confidence=protocol["bootstrap_confidence_level"],
        )
        reliably_faster = support and hierarchical_ci[1] < -effective_floor
        reliably_slower = support and hierarchical_ci[0] > effective_floor
        qualified = bool(reliably_faster or reliably_slower)
        if reliably_faster:
            direction = "faster"
        elif reliably_slower:
            direction = "slower"
        else:
            direction = "uncertain_or_within_practical_band"
        latency_mode_98pct_suitable = all(accuracy_passes)
        decisions.append(
            {
                "profile_id": profile_id,
                "axis": profile["axis"],
                "level": profile["level"],
                "support_validity_repeatability_gate": support,
                "latency_effect_qualified": qualified,
                "future_shared_multi_objective_eligible": qualified,
                "effect_direction": direction,
                "latency_reduction_qualified": bool(reliably_faster),
                "latency_mode_98pct_suitable": (
                    latency_mode_98pct_suitable
                ),
                "accuracy_retention_is_effect_gate": False,
                "same_seed_98pct_accuracy_passes": accuracy_passes,
                "seed_level_median_effects_ms": seed_medians,
                "negative_seed_count": sum(
                    value < 0.0 for value in seed_medians.values()
                ),
                "positive_seed_count": sum(
                    value > 0.0 for value in seed_medians.values()
                ),
                "median_across_seed_effect_ms": point_effect,
                "hierarchical_paired_effect_ci95_ms": hierarchical_ci,
                "effective_noise_floor_ms": effective_floor,
                "qualification_rule": (
                    "hierarchical_ci_upper < -effective_noise_floor_ms OR "
                    "hierarchical_ci_lower > +effective_noise_floor_ms"
                ),
                "feeds_final_selection": False,
                "winner_selected": False,
                "reason": (
                    f"reliably_{direction}_outside_practical_band"
                    if qualified
                    else "hierarchical_ci_overlaps_practical_band"
                ),
            }
        )
    return {
        "noise_floor": {
            "historical_noise_floor_ms": historical_floor,
            "reference_range_ms_by_seed": reference_ranges,
            "new_reference_range_ms": new_floor,
            "effective_noise_floor_ms": effective_floor,
        },
        "paired_by_seed": paired_by_seed,
        "profile_latency_summaries": profile_summaries,
        "latency_effect_decisions": decisions,
        "latency_effect_qualified_profiles": [
            item["profile_id"]
            for item in decisions
            if item["latency_effect_qualified"]
        ],
        "future_shared_multi_objective_profiles": [
            item["profile_id"]
            for item in decisions
            if item["future_shared_multi_objective_eligible"]
        ],
        "latency_reduction_qualified_profiles": [
            item["profile_id"]
            for item in decisions
            if item["latency_reduction_qualified"]
        ],
        "latency_mode_98pct_suitable_profiles": [
            item["profile_id"]
            for item in decisions
            if item["latency_mode_98pct_suitable"]
        ],
        "hierarchical_bootstrap": {
            "unit": "matched allocation nested within training seed",
            "seed_count": 3,
            "allocations_per_seed": 3,
            "resamples": protocol["bootstrap_resamples"],
            "confidence_level": protocol["bootstrap_confidence_level"],
            "statistic": (
                "median of within-seed allocation medians, then median "
                "across resampled seeds"
            ),
            "deterministic_seed_rule": (
                "first 64 bits of SHA256 of immutable profile label, "
                "method name, resample count, and confidence"
            ),
        },
    }


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    contract, one, one_path = load_contract(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    profiles = build_profiles(one)
    schedule = build_schedule(contract, profiles)
    if (
        len(profiles) != EXPECTED_PROFILES
        or len(schedule) != EXPECTED_ALLOCATIONS
    ):
        raise RuntimeError("frozen sensitivity design dimensions drifted")
    schedule_sha256 = sha256_value(schedule)
    checkpoint_artifact, artifact_entries = load_checkpoint_artifact(
        args.checkpoint_artifact.resolve(),
        args.checkpoint_artifact_sha256,
        contract,
        one,
        profiles,
    )
    accuracy_artifact, accuracy_entries = load_accuracy_artifact(
        args.accuracy_artifact.resolve(),
        args.accuracy_artifact_sha256,
        args.checkpoint_artifact_sha256,
        contract,
        one,
        profiles,
        artifact_entries,
    )
    plans, summaries, source_checks = regenerate_execution(
        manifest_path,
        contract,
        one,
        profiles,
        schedule,
        artifact_entries,
        manifest_sha256,
        args.checkpoint_artifact_sha256,
    )
    ledger, ledger_sha256 = load_submission_ledger(
        args.submission_ledger.resolve(),
        args.submission_ledger_sha256,
        contract,
        manifest_sha256,
        args.checkpoint_artifact_sha256,
        schedule_sha256,
        schedule,
        summaries,
        source_checks,
    )
    secrets_path = (
        args.secrets_env.resolve()
        if args.secrets_env
        else Path(contract["runtime_contract"]["secrets_env_path"])
    )
    loaded_keys = load_env_file(secrets_path)
    accounting = slurm_accounting(ledger["submissions"], contract)
    jobs, sdk_provenance = inspect_sdk_jobs(
        contract,
        ledger,
        args.sdk_state.resolve(),
        accounting,
    )
    measurements, allocation_evidence, consistency = aggregate_job_results(
        contract,
        profiles,
        schedule,
        artifact_entries,
        accuracy_entries,
        jobs,
        plans,
        manifest_sha256,
        args.checkpoint_artifact_sha256,
    )
    analysis = qualification_analysis(
        contract,
        one,
        profiles,
        schedule,
        accuracy_entries,
        measurements,
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "blockers": [],
        "checked_at_utc": utc_timestamp(),
        "manifest_id": contract["manifest_id"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "one_factor_manifest_path": str(one_path),
        "checkpoint_artifact_id": checkpoint_artifact["artifact_id"],
        "checkpoint_artifact_sha256": args.checkpoint_artifact_sha256,
        "accuracy_artifact_id": accuracy_artifact["artifact_id"],
        "accuracy_artifact_sha256": args.accuracy_artifact_sha256,
        "submission_ledger_path": str(args.submission_ledger.resolve()),
        "submission_ledger_sha256": ledger_sha256,
        "schedule_sha256": schedule_sha256,
        "source_checks": source_checks,
        "secrets_env_path": str(secrets_path),
        "loaded_secret_keys": loaded_keys,
        "secret_values_recorded": False,
        "sdk_state_provenance": sdk_provenance,
        "jobs": jobs,
        "allocation_evidence": allocation_evidence,
        "artifact_consistency": consistency,
        "allocation_measurements": measurements,
        **analysis,
        "winner_selected": False,
        "feeds_final_selection": False,
        "manual_promotion_permitted": False,
        "result_policy": (
            "Sensitivity qualification only. Future shared multi-objective "
            "axes/levels are driven exclusively by latency_effect_qualified; "
            "98 percent accuracy retention is an independent constrained-"
            "latency annotation and never an effect-qualification gate."
        ),
    }
    report["report_sha256"] = sha256_value(report)
    atomic_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "allocation_measurement_count": len(measurements),
                "latency_effect_qualified_profiles": report[
                    "latency_effect_qualified_profiles"
                ],
                "latency_reduction_qualified_profiles": report[
                    "latency_reduction_qualified_profiles"
                ],
                "latency_mode_98pct_suitable_profiles": report[
                    "latency_mode_98pct_suitable_profiles"
                ],
                "winner_selected": False,
                "feeds_final_selection": False,
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
