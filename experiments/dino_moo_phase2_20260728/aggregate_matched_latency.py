#!/usr/bin/env python3

"""Status and aggregation for the six preregistered matched-latency blocks.

This program is read-only with respect to TAO, SLURM, and the immutable
experiment manifests.  Its only write is the ignored runtime report.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
AUTOML_SRC = HERE.parent.parent / "src"
if str(AUTOML_SRC) not in sys.path:
    sys.path.insert(0, str(AUTOML_SRC))

from tao_automl.latency_stats import (  # noqa: E402
    LatencyProtocol,
    LatencyValidityThresholds,
    aggregate_synchronized_latency,
)


MANIFEST_PATH = HERE / "manifest.v2.json"
SUBMISSIONS_PATH = HERE / "runtime/matched_latency_blocks_submissions.json"
SDK_STATE_PATH = HERE / "runtime/slurm_state.db"
REPORT_PATH = HERE / "runtime/matched_latency_aggregation.json"
SECRETS_ENV_PATH = Path("/localhome/local-rarunachalam/.tao/config.env")
EXPECTED_MANIFEST_SHA256 = (
    "ccf88ad1a8c95a808bb9e217de50dc296b700e5af6e1dca474d56b967186e0d2"
)
PRACTICAL_TOLERANCE_MS = 0.75
PAIR_BOOTSTRAP_RESAMPLES = 5000
PAIR_BOOTSTRAP_CONFIDENCE = 0.95
EXPECTED_BLOCKS = 6
EXPECTED_CANDIDATES = 6
EXPECTED_RANKS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--submissions", type=Path, default=SUBMISSIONS_PATH)
    parser.add_argument("--sdk-state", type=Path, default=SDK_STATE_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--secrets-env", type=Path, default=SECRETS_ENV_PATH)
    return parser.parse_args()


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def load_env_file(path: Path) -> list[str]:
    """Load the existing TAO env file without ever returning secret values."""

    if not path.is_file():
        raise FileNotFoundError(f"required secrets env file not found: {path}")
    loaded: list[str] = []
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"unsupported env line {number}: missing '='")
        key, encoded = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ValueError(f"invalid env key on line {number}")
        tokens = shlex.split(encoded, comments=True, posix=True)
        if len(tokens) > 1:
            raise ValueError(f"unsupported env value syntax on line {number}")
        os.environ.setdefault(key, tokens[0] if tokens else "")
        loaded.append(key)
    return sorted(loaded)


def load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"immutable manifest v2 digest mismatch: {digest} != "
            f"{EXPECTED_MANIFEST_SHA256}"
        )
    manifest = json.loads(raw)
    if manifest.get("manifest_id") != "dino_moo_phase2_20260728_v2":
        raise ValueError("unexpected manifest identity")
    if manifest.get("feeds_selection") is not False:
        raise ValueError("matched latency experiment must be validation-only")
    candidates = manifest.get("candidates")
    schedule = manifest.get("schedule")
    if not isinstance(candidates, list) or len(candidates) != EXPECTED_CANDIDATES:
        raise ValueError("manifest must contain exactly six candidates")
    if not isinstance(schedule, list) or len(schedule) != EXPECTED_BLOCKS:
        raise ValueError("manifest must contain exactly six allocation blocks")
    candidate_ids = [item.get("candidate_id") for item in candidates]
    if len(set(candidate_ids)) != EXPECTED_CANDIDATES:
        raise ValueError("manifest candidate IDs must be unique")
    allocation_ids = [item.get("allocation_id") for item in schedule]
    if len(set(allocation_ids)) != EXPECTED_BLOCKS:
        raise ValueError("manifest allocation IDs must be unique")
    expected = set(candidate_ids)
    for block in schedule:
        order = block.get("candidate_order")
        if not isinstance(order, list) or len(order) != EXPECTED_CANDIDATES:
            raise ValueError(f"{block.get('allocation_id')}: invalid order")
        if set(order) != expected or len(set(order)) != EXPECTED_CANDIDATES:
            raise ValueError(
                f"{block.get('allocation_id')}: order is not a permutation"
            )
    if manifest["hardware_and_runtime"]["gpu_count"] != EXPECTED_RANKS:
        raise ValueError("manifest must require one eight-GPU node per block")
    return manifest, digest


def load_submissions(
    path: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, None
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    payload = json.loads(raw)
    if payload.get("manifest_id") != manifest["manifest_id"]:
        raise ValueError("submission ledger references another manifest")
    if payload.get("phase") != "matched_latency_blocks":
        raise ValueError("submission ledger has the wrong phase")
    if payload.get("feeds_selection") is not False:
        raise ValueError("submission ledger must set feeds_selection=false")
    submissions = payload.get("submissions")
    if not isinstance(submissions, list) or len(submissions) != EXPECTED_BLOCKS:
        raise ValueError("ledger must contain exactly six block submissions")
    expected = {
        block["allocation_id"]: block for block in manifest["schedule"]
    }
    actual_ids = [item.get("allocation_id") for item in submissions]
    if set(actual_ids) != set(expected) or len(set(actual_ids)) != EXPECTED_BLOCKS:
        raise ValueError("ledger block identities do not match the manifest")
    tao_ids: set[str] = set()
    slurm_ids: set[str] = set()
    by_id = {}
    for item in submissions:
        allocation_id = item["allocation_id"]
        if item.get("candidate_order") != expected[allocation_id][
            "candidate_order"
        ]:
            raise ValueError(f"{allocation_id}: submitted order drift")
        tao_id = item.get("tao_job_id")
        slurm_id = str(item.get("slurm_job_id", ""))
        if not isinstance(tao_id, str) or not tao_id:
            raise ValueError(f"{allocation_id}: missing TAO job ID")
        if not slurm_id.isdigit():
            raise ValueError(f"{allocation_id}: invalid SLURM job ID")
        if tao_id in tao_ids or slurm_id in slurm_ids:
            raise ValueError("TAO and SLURM job IDs must be unique")
        tao_ids.add(tao_id)
        slurm_ids.add(slurm_id)
        by_id[allocation_id] = item
    payload["submissions"] = [
        by_id[block["allocation_id"]] for block in manifest["schedule"]
    ]
    return payload, digest


def latency_protocol(manifest: dict[str, Any]) -> LatencyProtocol:
    source = manifest["latency_protocol"]
    thresholds = source["validity_thresholds"]
    return LatencyProtocol(
        warmup_iterations=source["warmup_iterations"],
        timed_iterations=source["timed_iterations"],
        repeated_rounds=source["repeated_rounds"],
        tail_percentile=source["tail_percentile"],
        bootstrap_resamples=source["bootstrap_resamples"],
        bootstrap_confidence_level=source["bootstrap_confidence_level"],
        bootstrap_seed=source["bootstrap_seed"],
        expected_devices=tuple(source["expected_devices"]),
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


def ssh_target() -> str:
    user = os.environ.get("SLURM_USER", "").strip()
    host = os.environ.get("SLURM_HOSTNAME", "").split(",", 1)[0].strip()
    if not user or not host:
        raise RuntimeError("SLURM_USER and SLURM_HOSTNAME are required")
    return f"{user}@{host}"


def remote_output(command: str, *, timeout: int = 900) -> str:
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    key_path = os.environ.get("SSH_KEY_PATH")
    if key_path:
        ssh.extend(["-i", key_path])
    ssh.extend([ssh_target(), command])
    return subprocess.run(
        ssh,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    ).stdout


def sdk_db_path(state_path: Path) -> Path:
    if state_path.name.endswith(".json"):
        return state_path.with_suffix(".db")
    return Path(str(state_path) + ".db")


def slurm_accounting(slurm_ids: list[str]) -> dict[str, dict[str, Any]]:
    command = " ".join(
        [
            "sacct",
            "-X",
            "-j",
            shlex.quote(",".join(slurm_ids)),
            "--noheader",
            "--parsable2",
            "--format=JobIDRaw,State,ExitCode,NodeList",
        ]
    )
    rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for line in remote_output(command, timeout=120).splitlines():
        fields = line.strip().split("|")
        if len(fields) < 4 or fields[0] not in slurm_ids:
            continue
        state = fields[1].split("+", 1)[0].split(None, 1)[0]
        rows[fields[0]].append(
            {
                "slurm_job_id": fields[0],
                "state": state,
                "exit_code": fields[2],
                "node_list": fields[3],
            }
        )
    result = {}
    for slurm_id in slurm_ids:
        candidates = rows.get(slurm_id, [])
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected one sacct row for {slurm_id}, found "
                f"{len(candidates)}"
            )
        row = candidates[0]
        row["complete"] = (
            row["state"] == "COMPLETED" and row["exit_code"] == "0:0"
        )
        result[slurm_id] = row
    return result


def inspect_jobs(
    manifest: dict[str, Any],
    submissions: dict[str, Any],
    state_path: Path,
) -> tuple[list[dict[str, Any]], Any]:
    database = sdk_db_path(state_path)
    if not database.is_file():
        raise FileNotFoundError(f"SDK durable state is missing: {database}")
    from tao_sdk.platforms.slurm import SlurmSDK

    hardware = manifest["hardware_and_runtime"]
    os.environ["SLURM_USE_SQSH"] = "false"
    os.environ["SLURM_PARTITION"] = hardware["partition"]
    os.environ["SLURM_ACCOUNT"] = hardware["account"]
    sdk = SlurmSDK(poll_interval=10, state_file=state_path)
    jobs = []
    try:
        accounting = slurm_accounting(
            [str(item["slurm_job_id"]) for item in submissions["submissions"]]
        )
        for item in submissions["submissions"]:
            status = sdk.get_job_status(item["tao_job_id"])
            identity = sdk._handler.get_job_runtime_identity(item["tao_job_id"])
            actual_slurm_id = str(identity.get("slurm_job_id", ""))
            if actual_slurm_id != str(item["slurm_job_id"]):
                raise RuntimeError(
                    f"{item['allocation_id']}: SDK/ledger SLURM ID mismatch"
                )
            result_root = local_lustre_path(
                sdk.get_job_results_dir(item["tao_job_id"])
            )
            sacct = accounting[actual_slurm_id]
            jobs.append(
                {
                    "allocation_id": item["allocation_id"],
                    "tao_job_id": item["tao_job_id"],
                    "slurm_job_id": actual_slurm_id,
                    "sdk_status": status.status,
                    "sdk_message": status.message,
                    "slurm_state": sacct["state"],
                    "slurm_exit_code": sacct["exit_code"],
                    "node_list": sacct["node_list"],
                    "result_root": result_root,
                    "complete": status.status == "Complete"
                    and sacct["complete"],
                    "feeds_selection": False,
                }
            )
    finally:
        sdk._monitor.stop()
        sdk._store.close()
    return jobs, database


def local_lustre_path(uri: str) -> str:
    if uri.startswith("lustre://"):
        path = uri.removeprefix("lustre://")
        return path if path.startswith("/") else f"/{path}"
    if uri.startswith("/"):
        return uri
    raise ValueError(f"expected Lustre results URI, got {uri!r}")


def fetch_allocation_bundle(job: dict[str, Any]) -> dict[str, Any]:
    result_path = (
        Path(job["result_root"])
        / "dino_moo_phase2_20260728"
        / job["allocation_id"]
        / job["tao_job_id"]
        / "allocation_result.json"
    )
    reader = "\n".join(
        [
            "import json,sys",
            "from pathlib import Path",
            "path=Path(sys.argv[1])",
            "result=json.loads(path.read_text())",
            "records={}",
            "for run in result.get('candidate_runs', []):",
            " root=Path(run['raw_samples_dir'])",
            " paths=sorted(root.glob('rank_*.json'))",
            " records[run['candidate_id']]={",
            "  'paths':[str(item) for item in paths],",
            "  'records':[json.loads(item.read_text()) for item in paths],",
            " }",
            "print(json.dumps({'result_path':str(path),'result':result,"
            "'rank_records':records},sort_keys=True))",
        ]
    )
    output = remote_output(
        f"python3 -c {shlex.quote(reader)} {shlex.quote(str(result_path))}",
        timeout=900,
    )
    return json.loads(output)


def expected_protocol_record(manifest: dict[str, Any]) -> dict[str, Any]:
    protocol = manifest["latency_protocol"]
    benchmark = manifest["latency_benchmark"]
    return {
        "warmup_iterations": protocol["warmup_iterations"],
        "timed_iterations": protocol["timed_iterations"],
        "repeated_rounds": protocol["repeated_rounds"],
        "preloaded_batches": benchmark["preloaded_batches"],
        "batch_size_per_gpu": benchmark["batch_size_per_gpu"],
        "precision": benchmark["precision"],
        "tf32": benchmark["tf32"],
        "cudnn_benchmark": benchmark["cudnn_benchmark"],
        "cudnn_deterministic": benchmark["cudnn_deterministic"],
        "timed_scope": benchmark["timed_scope"],
        "excluded_scope": benchmark["excluded_scope"],
        "synchronization": benchmark["synchronization"],
        "seed": benchmark["benchmark_seed"],
    }


def validate_input(record: dict[str, Any], manifest: dict[str, Any]) -> str:
    metadata = record.get("benchmark_inputs")
    if not isinstance(metadata, dict):
        raise ValueError("missing benchmark_inputs")
    batches = metadata.get("batches")
    expected_count = manifest["latency_benchmark"]["preloaded_batches"]
    if not isinstance(batches, list) or len(batches) != expected_count:
        raise ValueError("benchmark input preload count mismatch")
    if metadata.get("batch_count") != expected_count:
        raise ValueError("benchmark input batch_count mismatch")
    if metadata.get("example_count") != expected_count:
        raise ValueError("benchmark input example_count mismatch")
    for index, batch in enumerate(batches):
        if batch.get("batch_index") != index or batch.get("batch_size") != 1:
            raise ValueError("benchmark input batch order/size mismatch")
        if batch.get("model_input", {}).get("shape") != [1, 4, 800, 1333]:
            raise ValueError("model input shape mismatch")
        if batch.get("image_tensor_shape") != [1, 3, 800, 1333]:
            raise ValueError("image tensor shape mismatch")
        if batch.get("padding_mask", {}).get("shape") != [1, 1, 800, 1333]:
            raise ValueError("padding mask shape mismatch")
    identity = metadata.get("identity_sha256")
    if not isinstance(identity, str) or len(identity) != 64:
        raise ValueError("invalid benchmark input identity")
    identity_payload = {
        "schema_version": metadata.get("schema_version"),
        "batch_count": metadata.get("batch_count"),
        "example_count": metadata.get("example_count"),
        "batches": batches,
    }
    canonical = json.dumps(
        identity_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if sha256_bytes(canonical) != identity:
        raise ValueError("benchmark input identity digest does not match metadata")
    return identity


def validate_rank_record(
    record: dict[str, Any],
    *,
    rank: int,
    candidate: dict[str, Any],
    allocation_hostname: str,
    manifest: dict[str, Any],
) -> tuple[str, tuple[Any, ...], str]:
    if record.get("rank") != rank or record.get("local_rank") != rank:
        raise ValueError(f"rank identity mismatch for rank {rank}")
    if record.get("world_size") != EXPECTED_RANKS:
        raise ValueError("rank world_size is not eight")
    if record.get("checkpoint") != candidate["checkpoint"]:
        raise ValueError("checkpoint path differs from manifest v2")
    expected_config = (
        f"/tmp/dino_moo_phase2_20260728/configs/"
        f"{candidate['candidate_id']}.yaml"
    )
    if record.get("config_path") != expected_config:
        raise ValueError("staged evaluation config path mismatch")
    expected_protocol = expected_protocol_record(manifest)
    actual_protocol = record.get("protocol")
    if not isinstance(actual_protocol, dict):
        raise ValueError("missing rank protocol")
    for key, expected in expected_protocol.items():
        if actual_protocol.get(key) != expected:
            raise ValueError(f"rank protocol mismatch: {key}")

    hardware = record.get("hardware", {})
    expected_hardware = manifest["hardware_and_runtime"]["expected_hardware"]
    if hardware.get("hostname") != allocation_hostname:
        raise ValueError("rank hostname differs from allocation hostname")
    for key in ("gpu_name", "compute_capability", "total_memory_bytes"):
        if hardware.get(key) != expected_hardware[key]:
            raise ValueError(f"hardware mismatch: {key}")
    nvidia_smi = hardware.get("nvidia_smi", "")
    if not isinstance(nvidia_smi, str) or nvidia_smi.startswith("unavailable:"):
        raise ValueError("nvidia-smi evidence unavailable")
    gpu_uuid = nvidia_smi.split(",", 1)[0].strip()
    if not gpu_uuid:
        raise ValueError("GPU UUID is empty")

    runtime = record.get("runtime", {})
    for key in ("torch", "cuda", "cudnn"):
        if runtime.get(key) != expected_hardware[key]:
            raise ValueError(f"runtime mismatch: {key}")
    runtime_signature = (
        runtime.get("python"),
        runtime.get("torch"),
        runtime.get("cuda"),
        runtime.get("cudnn"),
    )
    if not all(value is not None for value in runtime_signature):
        raise ValueError("runtime signature is incomplete")
    return validate_input(record, manifest), runtime_signature, gpu_uuid


def stats_record(stats: Any) -> dict[str, Any]:
    return {
        "median_ms": stats.median_ms,
        "p95_ms": stats.tail_latency_ms,
        "mad_ms": stats.mad_ms,
        "iqr_ms": stats.iqr_ms,
        "robust_cv": stats.robust_cv,
        "round_median_range_ms": stats.round_median_range_ms,
        "round_drift_ms": stats.round_drift_ms,
        "device_median_range_ms": stats.device_median_range_ms,
        "bootstrap_median_ci95_ms": list(stats.bootstrap_median_ci_ms),
        "raw_sample_count_total": stats.raw_sample_count_total,
        "samples_per_device": stats.samples_per_device,
        "is_valid": stats.is_valid,
        "invalid_reasons": list(stats.invalid_reasons),
    }


def aggregate_bundles(
    manifest: dict[str, Any],
    jobs: list[dict[str, Any]],
    bundles: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol = latency_protocol(manifest)
    candidates = {
        item["candidate_id"]: item for item in manifest["candidates"]
    }
    schedule = {
        item["allocation_id"]: item for item in manifest["schedule"]
    }
    input_identities: dict[int, str] = {}
    runtime_signature: tuple[Any, ...] | None = None
    allocation_gpu_uuids: dict[str, dict[int, str]] = defaultdict(dict)
    measurements = []

    for job in jobs:
        allocation_id = job["allocation_id"]
        bundle = bundles[allocation_id]
        result = bundle.get("result", {})
        block = schedule[allocation_id]
        if result.get("schema_version") != 1:
            raise ValueError(f"{allocation_id}: invalid result schema")
        for key, expected in (
            ("status", "success"),
            ("manifest_id", manifest["manifest_id"]),
            ("allocation_id", allocation_id),
            ("tao_job_id", job["tao_job_id"]),
            ("feeds_selection", False),
        ):
            if result.get(key) != expected:
                raise ValueError(f"{allocation_id}: result mismatch: {key}")
        runs = result.get("candidate_runs")
        if not isinstance(runs, list) or len(runs) != EXPECTED_CANDIDATES:
            raise ValueError(f"{allocation_id}: expected six candidate runs")
        if [run.get("candidate_id") for run in runs] != block[
            "candidate_order"
        ]:
            raise ValueError(f"{allocation_id}: candidate order drift")
        if [run.get("position") for run in runs] != list(
            range(EXPECTED_CANDIDATES)
        ):
            raise ValueError(f"{allocation_id}: candidate position drift")
        if any(
            run.get("status") != "success" or run.get("exit_code") != 0
            for run in runs
        ):
            raise ValueError(f"{allocation_id}: candidate run is not successful")
        hostname = result.get("hostname")
        if not isinstance(hostname, str) or not hostname:
            raise ValueError(f"{allocation_id}: missing allocation hostname")

        for run in runs:
            candidate_id = run["candidate_id"]
            candidate = candidates[candidate_id]
            raw = bundle.get("rank_records", {}).get(candidate_id)
            if not isinstance(raw, dict):
                raise ValueError(
                    f"{allocation_id}/{candidate_id}: missing rank bundle"
                )
            paths = raw.get("paths")
            records = raw.get("records")
            expected_names = [f"rank_{rank}.json" for rank in range(8)]
            if (
                not isinstance(paths, list)
                or [Path(path).name for path in paths] != expected_names
                or not isinstance(records, list)
                or len(records) != EXPECTED_RANKS
            ):
                raise ValueError(
                    f"{allocation_id}/{candidate_id}: expected exactly rank_0..7"
                )
            samples = {
                round_index: {}
                for round_index in range(protocol.repeated_rounds)
            }
            for rank, record in enumerate(records):
                identity, signature, gpu_uuid = validate_rank_record(
                    record,
                    rank=rank,
                    candidate=candidate,
                    allocation_hostname=hostname,
                    manifest=manifest,
                )
                prior_input = input_identities.setdefault(rank, identity)
                if identity != prior_input:
                    raise ValueError(
                        f"{allocation_id}/{candidate_id}: rank {rank} "
                        "benchmark input drift"
                    )
                if runtime_signature is None:
                    runtime_signature = signature
                elif runtime_signature != signature:
                    raise ValueError("runtime differs between measurements")
                prior_uuid = allocation_gpu_uuids[allocation_id].setdefault(
                    rank, gpu_uuid
                )
                if prior_uuid != gpu_uuid:
                    raise ValueError(
                        f"{allocation_id}: rank {rank} GPU changed within block"
                    )
                rank_samples = record.get("samples_ms")
                if (
                    not isinstance(rank_samples, list)
                    or len(rank_samples) != protocol.repeated_rounds
                ):
                    raise ValueError("rank samples have the wrong round count")
                for round_index, values in enumerate(rank_samples):
                    samples[round_index][str(rank)] = values
            stats = aggregate_synchronized_latency(samples, protocol)
            if not stats.is_valid:
                raise ValueError(
                    f"{allocation_id}/{candidate_id}: "
                    f"{stats.validity_reason}"
                )
            measurements.append(
                {
                    "allocation_id": allocation_id,
                    "tao_job_id": job["tao_job_id"],
                    "slurm_job_id": job["slurm_job_id"],
                    "node_list": job["node_list"],
                    "hostname": hostname,
                    "candidate_id": candidate_id,
                    "position": run["position"],
                    "checkpoint_sha256": candidate["checkpoint_sha256"],
                    "input_identity_sha256_by_rank": {
                        str(rank): input_identities[rank]
                        for rank in range(EXPECTED_RANKS)
                    },
                    **stats_record(stats),
                }
            )
    if len(measurements) != EXPECTED_BLOCKS * EXPECTED_CANDIDATES:
        raise RuntimeError("expected exactly 36 valid allocation measurements")
    consistency = {
        "hardware_contract": "pass",
        "runtime_contract": "pass",
        "protocol_contract": "pass",
        "benchmark_input_identity": "pass",
        "rank_files_per_candidate": EXPECTED_RANKS,
        "runtime_signature": list(runtime_signature or ()),
        "gpu_uuid_count_by_allocation": {
            allocation_id: len(set(by_rank.values()))
            for allocation_id, by_rank in allocation_gpu_uuids.items()
        },
    }
    if any(
        count != EXPECTED_RANKS
        for count in consistency["gpu_uuid_count_by_allocation"].values()
    ):
        raise ValueError("an allocation does not contain eight distinct GPUs")
    return measurements, consistency


def deterministic_bootstrap_ci(
    values: list[float],
    label: str,
    *,
    statistic: str = "median",
) -> list[float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("bootstrap values must be finite and non-empty")
    seed_material = (
        f"{label}|{PAIR_BOOTSTRAP_RESAMPLES}|"
        f"{PAIR_BOOTSTRAP_CONFIDENCE}"
    )
    seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    source = np.asarray(values, dtype=np.float64)
    indices = rng.integers(
        0, len(source), size=(PAIR_BOOTSTRAP_RESAMPLES, len(source))
    )
    sampled = source[indices]
    if statistic == "median":
        estimates = np.median(sampled, axis=1)
    elif statistic == "mean":
        estimates = np.mean(sampled, axis=1)
    else:
        raise ValueError(f"unsupported bootstrap statistic: {statistic}")
    alpha = 1.0 - PAIR_BOOTSTRAP_CONFIDENCE
    low, high = np.quantile(
        estimates,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    return [float(low), float(high)]


def distribution_summary(values: list[float], label: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    q1, q3 = np.quantile(array, [0.25, 0.75], method="linear")
    median = float(np.median(array))
    return {
        "allocation_count": len(values),
        "values_ms": values,
        "median_ms": median,
        "mean_ms": float(np.mean(array)),
        "sample_stdev_ms": (
            float(statistics.stdev(values)) if len(values) > 1 else 0.0
        ),
        "min_ms": float(np.min(array)),
        "max_ms": float(np.max(array)),
        "range_ms": float(np.max(array) - np.min(array)),
        "mad_ms": float(np.median(np.abs(array - median))),
        "iqr_ms": float(q3 - q1),
        "bootstrap_median_ci95_ms": deterministic_bootstrap_ci(values, label),
    }


def practical_classification(delta: float, ci: list[float]) -> dict[str, str]:
    tolerance = PRACTICAL_TOLERANCE_MS
    if delta < -tolerance:
        point = "first_practically_faster"
    elif delta > tolerance:
        point = "second_practically_faster"
    else:
        point = "practically_equivalent"
    if ci[1] < -tolerance:
        stable = "first_stably_faster"
    elif ci[0] > tolerance:
        stable = "second_stably_faster"
    elif ci[0] >= -tolerance and ci[1] <= tolerance:
        stable = "stable_practical_equivalence"
    else:
        stable = "uncertain_at_practical_tolerance"
    return {"point_classification": point, "ci_classification": stable}


def comparative_analysis(
    manifest: dict[str, Any],
    measurements: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_ids = sorted(
        candidate["candidate_id"] for candidate in manifest["candidates"]
    )
    allocation_ids = [
        block["allocation_id"] for block in manifest["schedule"]
    ]
    by_key = {
        (row["allocation_id"], row["candidate_id"]): row
        for row in measurements
    }
    between = []
    aggregate_medians = {}
    for candidate_id in candidate_ids:
        medians = [
            by_key[(allocation_id, candidate_id)]["median_ms"]
            for allocation_id in allocation_ids
        ]
        p95s = [
            by_key[(allocation_id, candidate_id)]["p95_ms"]
            for allocation_id in allocation_ids
        ]
        median_summary = distribution_summary(
            medians, f"{manifest['manifest_id']}|{candidate_id}|median"
        )
        aggregate_medians[candidate_id] = median_summary["median_ms"]
        between.append(
            {
                "candidate_id": candidate_id,
                "median_latency": median_summary,
                "p95_latency": distribution_summary(
                    p95s, f"{manifest['manifest_id']}|{candidate_id}|p95"
                ),
            }
        )

    pairs = []
    stable_claims = []
    for first_index, first in enumerate(candidate_ids):
        for second in candidate_ids[first_index + 1 :]:
            median_differences = [
                by_key[(allocation_id, first)]["median_ms"]
                - by_key[(allocation_id, second)]["median_ms"]
                for allocation_id in allocation_ids
            ]
            p95_differences = [
                by_key[(allocation_id, first)]["p95_ms"]
                - by_key[(allocation_id, second)]["p95_ms"]
                for allocation_id in allocation_ids
            ]
            median_delta = float(np.median(median_differences))
            p95_delta = float(np.median(p95_differences))
            median_ci = deterministic_bootstrap_ci(
                median_differences,
                f"{manifest['manifest_id']}|paired-median|{first}|{second}",
            )
            p95_ci = deterministic_bootstrap_ci(
                p95_differences,
                f"{manifest['manifest_id']}|paired-p95|{first}|{second}",
            )
            classification = practical_classification(median_delta, median_ci)
            pair = {
                "first_candidate_id": first,
                "second_candidate_id": second,
                "delta_convention": "first_minus_second; negative means first faster",
                "allocation_ids": allocation_ids,
                "paired_median_differences_ms": median_differences,
                "median_paired_difference_ms": median_delta,
                "median_paired_bootstrap_ci95_ms": median_ci,
                "paired_p95_differences_ms": p95_differences,
                "median_paired_p95_difference_ms": p95_delta,
                "p95_paired_bootstrap_ci95_ms": p95_ci,
                "practical_tolerance_ms": PRACTICAL_TOLERANCE_MS,
                **classification,
            }
            pairs.append(pair)
            if classification["ci_classification"] == "first_stably_faster":
                stable_claims.append(
                    {
                        "faster_candidate_id": first,
                        "slower_candidate_id": second,
                        "basis": "paired median CI entirely below -0.75 ms",
                    }
                )
            elif classification["ci_classification"] == "second_stably_faster":
                stable_claims.append(
                    {
                        "faster_candidate_id": second,
                        "slower_candidate_id": first,
                        "basis": "paired median CI entirely above +0.75 ms",
                    }
                )

    descriptive_order = sorted(
        candidate_ids, key=lambda item: (aggregate_medians[item], item)
    )
    stable_edges = {
        (item["faster_candidate_id"], item["slower_candidate_id"])
        for item in stable_claims
    }
    all_ordered_pairs_stable = all(
        (descriptive_order[first_index], descriptive_order[second_index])
        in stable_edges
        for first_index in range(len(descriptive_order))
        for second_index in range(first_index + 1, len(descriptive_order))
    )
    adjacent_stability = [
        {
            "faster_candidate_id": descriptive_order[index],
            "slower_candidate_id": descriptive_order[index + 1],
            "stable": (
                descriptive_order[index],
                descriptive_order[index + 1],
            )
            in stable_edges,
        }
        for index in range(len(descriptive_order) - 1)
    ]
    return {
        "practical_tolerance_ms": PRACTICAL_TOLERANCE_MS,
        "paired_bootstrap": {
            "unit": "allocation",
            "resamples": PAIR_BOOTSTRAP_RESAMPLES,
            "confidence_level": PAIR_BOOTSTRAP_CONFIDENCE,
            "statistic": "median paired difference",
            "seed_rule": "first 64 bits of SHA256 of immutable pair label",
        },
        "between_allocation_statistics": between,
        "all_pairwise_comparisons": pairs,
        "descriptive_latency_order": descriptive_order,
        "descriptive_order_is_a_stable_total_order": all_ordered_pairs_stable,
        "adjacent_order_stability": adjacent_stability,
        "stable_ordering_claims": stable_claims,
        "ordering_claim_policy": (
            "A directional claim is emitted only when the paired-bootstrap "
            "CI lies wholly beyond the +/-0.75 ms practical-equivalence band."
        ),
    }


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest, manifest_sha256 = load_manifest(manifest_path)
    submissions, submissions_sha256 = load_submissions(
        args.submissions.resolve(), manifest
    )
    base_report: dict[str, Any] = {
        "schema_version": 1,
        "checked_at_utc": utc_timestamp(),
        "manifest_path": str(manifest_path),
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_sha256,
        "submissions_path": str(args.submissions.resolve()),
        "submissions_sha256": submissions_sha256,
        "feeds_selection": False,
        "practical_tolerance_ms": PRACTICAL_TOLERANCE_MS,
    }
    if submissions is None:
        report = {
            **base_report,
            "overall_status": "not_submitted",
            "all_jobs_complete": False,
            "jobs": [],
            "note": "No matched-latency submission ledger exists yet.",
        }
        atomic_json(args.report.resolve(), report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 0

    loaded_keys = load_env_file(args.secrets_env.resolve())
    jobs, database = inspect_jobs(
        manifest, submissions, args.sdk_state.resolve()
    )
    report = {
        **base_report,
        "secrets_env_path": str(args.secrets_env.resolve()),
        "loaded_secret_keys": loaded_keys,
        "secret_values_recorded": False,
        "sdk_state_path": str(args.sdk_state.resolve()),
        "sdk_database_path": str(database),
        "jobs": jobs,
        "all_jobs_complete": all(job["complete"] for job in jobs),
    }
    if not report["all_jobs_complete"]:
        failed = any(
            job["sdk_status"] in {"Error", "Canceled"}
            or (
                job["slurm_state"]
                not in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}
                and not job["complete"]
            )
            for job in jobs
        )
        report["overall_status"] = (
            "failed_or_unverifiable" if failed else "pending"
        )
        report["note"] = (
            "Aggregation requires all six SDK statuses Complete and all six "
            "SLURM allocation rows COMPLETED with exit code 0:0."
        )
        atomic_json(args.report.resolve(), report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 0

    bundles = {
        job["allocation_id"]: fetch_allocation_bundle(job) for job in jobs
    }
    measurements, consistency = aggregate_bundles(manifest, jobs, bundles)
    report.update(
        {
            "overall_status": "complete",
            "artifact_consistency": consistency,
            "per_allocation_candidate_measurements": measurements,
            "analysis": comparative_analysis(manifest, measurements),
            "result_policy": (
                "Validation-only matched-latency evidence; never feeds "
                "historical or final AutoML selection."
            ),
        }
    )
    atomic_json(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
