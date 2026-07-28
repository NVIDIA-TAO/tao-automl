#!/usr/bin/env python3

"""Run the reviewed DINO multi-objective validation on a shared archive.

Three reproducible Bayesian searches generate independent ten-candidate
sub-archives. Every candidate is trained for the same budget, evaluated for
mAP50, and benchmarked with the same eight-replica latency protocol. The final
accuracy, latency, and multi-objective winners are selected algorithmically
from the union of all successful measured candidates.
"""

from __future__ import annotations

import argparse
import base64
import copy
import csv
from dataclasses import asdict
import fcntl
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

import yaml

from tao_automl.latency_stats import (
    LatencyProtocol,
    LatencyValidityThresholds,
    aggregate_synchronized_latency,
)
from tao_automl.objectives import parse_objective_config
from tao_automl.runner import AutoMLRunner
from tao_sdk.platforms.slurm import SlurmSDK
from tao_sdk.script_runner import build_entrypoint


EXPERIMENT_DIR = Path(__file__).resolve().parent
SKILL_DIR = Path(
    "/localhome/local-rarunachalam/tao-skills-external/"
    "skills/models/tao-train-dino"
)
EVALUATE_TEMPLATE = SKILL_DIR / "references/spec_template_evaluate.yaml"
BENCHMARK_SCRIPT = EXPERIMENT_DIR / "dino_latency_benchmark.py"

IMAGE_URI = "nvcr.io/nvidia/tao/tao-toolkit:7.0.1-pyt"
SQSH_PATH = (
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
    "nvcr.io_nvidia_tao_tao-toolkit_7.0.1-pyt.sqsh"
)
PARTITION = "polar3"
ACCOUNT = "edgeai_tao-ptm_image-foundation-model-clip"
DATASET_URI = (
    "s3://nvcf-storage-handling/data/"
    "tao_od_synthetic_full_dino_coco/"
)
DATA_ROOT = (
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
    "tao_od_synthetic_full_dino_coco"
)
PTM = (
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/ptm/"
    "pretrained_dino_coco/dino_resnet_50_trainable_v1.0/"
    "dino_resnet50_ep12.pth"
)
REPOSITORIES = {
    "tao_automl": Path("/localhome/local-rarunachalam/tao-automl"),
    "tao_sdk": Path("/localhome/local-rarunachalam/tao-sdk"),
    "tao_skills": Path("/localhome/local-rarunachalam/tao-skills-external"),
}

GPU_COUNT = 8
TRAIN_BATCH_SIZE_PER_GPU = 4
LATENCY_BATCH_SIZE_PER_GPU = 1
TRAIN_EPOCHS = 10
TRAIN_SEED = 1234
SEARCH_SEEDS = (314159, 271828, 161803)
RECOMMENDATIONS_PER_SEED = 10
ACCURACY_RETENTION_FRACTION = 0.98
BASELINE_MAP50_INFORMATIONAL = 0.007808934173321529
TERMINAL = {"Complete", "Error", "Canceled"}

LATENCY_WARMUPS = 50
LATENCY_ITERATIONS = 100
LATENCY_ROUNDS = 5
LATENCY_PRELOADED_BATCHES = 16
LATENCY_BOOTSTRAP_RESAMPLES = 5000
LATENCY_BOOTSTRAP_SEED = 424242
LATENCY_BENCHMARK_SEED = 20260727
LATENCY_PROTOCOL = LatencyProtocol(
    warmup_iterations=LATENCY_WARMUPS,
    timed_iterations=LATENCY_ITERATIONS,
    repeated_rounds=LATENCY_ROUNDS,
    tail_percentile=95.0,
    bootstrap_resamples=LATENCY_BOOTSTRAP_RESAMPLES,
    bootstrap_confidence_level=0.95,
    bootstrap_seed=LATENCY_BOOTSTRAP_SEED,
    expected_devices=tuple(str(index) for index in range(GPU_COUNT)),
    validity_thresholds=LatencyValidityThresholds(
        max_robust_cv=0.10,
        max_round_median_range_fraction=0.05,
        max_absolute_round_drift_fraction=0.05,
        max_device_median_range_fraction=0.05,
        max_bootstrap_ci_width_fraction=0.03,
    ),
)

SEARCH_PARAMETERS = (
    "model.num_queries",
    "train.optim.lr",
    "train.optim.weight_decay",
)
SEARCH_RANGES = {
    "model.num_queries": {"valid_min": 300, "valid_max": 900},
    "train.optim.lr": {"valid_min": 1.0e-5, "valid_max": 5.0e-4},
    "train.optim.weight_decay": {
        "valid_min": 1.0e-5,
        "valid_max": 1.0e-3,
    },
}
SPEC_OVERRIDES = {
    "dataset.train_data_sources[0].image_dir": (
        f"{DATA_ROOT}/train/images/images"
    ),
    "dataset.train_data_sources[0].json_file": (
        f"{DATA_ROOT}/train/annotations.json"
    ),
    "dataset.val_data_sources[0].image_dir": f"{DATA_ROOT}/val/images/images",
    "dataset.val_data_sources[0].json_file": (
        f"{DATA_ROOT}/val/annotations.json"
    ),
    "dataset.num_classes": 5,
    "dataset.eval_class_ids": [1, 2, 3, 4],
    "dataset.batch_size": TRAIN_BATCH_SIZE_PER_GPU,
    "model.backbone": "resnet_50",
    "model.num_queries": 900,
    "model.num_select": 300,
    "train.pretrained_model_path": PTM,
    "train.num_gpus": GPU_COUNT,
    "train.gpu_ids": list(range(GPU_COUNT)),
    "train.num_nodes": 1,
    "train.num_epochs": TRAIN_EPOCHS,
    "train.validation_interval": 1,
    "train.seed": TRAIN_SEED,
    "train.precision": "fp32",
    "train.distributed_strategy": "ddp",
    "train.cudnn.benchmark": False,
    "train.cudnn.deterministic": True,
    "wandb.enable": False,
}
EVALUATE_INPUTS = {
    "evaluate.checkpoint": {"type": "file"},
    "dataset.test_data_sources.image_dir": {"type": "file"},
    "dataset.test_data_sources.json_file": {"type": "file"},
}
EVALUATE_OUTPUTS = {"results_dir": {"type": "folder"}}
MAP50_PATTERNS = (
    re.compile(
        r"(?:Validation|Test)\s+mAP50\s*[:=]\s*"
        r"([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)"
    ),
    re.compile(
        r"\btest_mAP50\b[^0-9+\-]*"
        r"([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)"
    ),
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Evaluate and benchmark the compatible ResNet50 PTM on one "
            "eight-GPU SQSH job per phase, then freeze the hardware contract."
        ),
    )
    mode.add_argument(
        "--combine-only",
        action="store_true",
        help="Recompute final selections from completed candidate records.",
    )
    return parser.parse_args()


def append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **payload,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()


def repository_identity(path: Path) -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()

    return {
        "path": str(path),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
    }


def dino_metric_extractor(logs: str, metric_name: str) -> float | None:
    if metric_name != "mAP50":
        return None
    matches: list[float] = []
    for pattern in MAP50_PATTERNS:
        matches.extend(float(value) for value in pattern.findall(logs))
    return matches[-1] if matches else None


def local_lustre_path(uri: str) -> str:
    if uri.startswith("lustre://"):
        path = uri.removeprefix("lustre://")
        return path if path.startswith("/") else f"/{path}"
    if uri.startswith("/"):
        return uri
    raise ValueError(f"Expected a Lustre result URI, got {uri!r}")


def ssh_target() -> str:
    host = os.environ["SLURM_HOSTNAME"].split(",", 1)[0].strip()
    user = os.environ["SLURM_USER"].strip()
    return f"{user}@{host}"


def remote_output(command: str, timeout: int = 180) -> str:
    ssh_command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
    ]
    key_path = os.environ.get("SSH_KEY_PATH")
    if key_path:
        ssh_command.extend(["-i", key_path])
    ssh_command.extend([ssh_target(), command])
    completed = subprocess.run(
        ssh_command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.stdout


def find_terminal_checkpoint(sdk: SlurmSDK, train_job_id: str) -> str:
    result_root = local_lustre_path(sdk.get_job_results_dir(train_job_id))
    if Path(result_root).name != train_job_id:
        raise ValueError(
            f"Training result root does not end in job ID {train_job_id}: "
            f"{result_root}"
        )
    command = (
        f"find {shlex.quote(result_root)} -type f "
        "\\( -name '*.pth' -o -name '*.ckpt' \\) "
        "-printf '%T@ %p\\n' | sort -nr | head -1"
    )
    output = remote_output(command).strip()
    if not output or " " not in output:
        raise FileNotFoundError(
            f"No terminal checkpoint found for training job {train_job_id}"
        )
    checkpoint = output.split(" ", 1)[1].strip()
    if not checkpoint.startswith(f"{result_root.rstrip('/')}/"):
        raise ValueError(
            f"Resolved checkpoint escaped training result root: {checkpoint}"
        )
    return checkpoint


def read_status_map50(sdk: SlurmSDK, eval_job_id: str) -> float | None:
    result_root = local_lustre_path(sdk.get_job_results_dir(eval_job_id))
    status_path = f"{result_root}/results_dir/evaluate/status.json"
    output = remote_output(
        f"test -f {shlex.quote(status_path)} && "
        f"tail -80 {shlex.quote(status_path)}"
    )
    values: list[float] = []
    for line in output.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        kpi = record.get("kpi")
        if not isinstance(kpi, dict):
            continue
        value = kpi.get("test_mAP50", kpi.get("val_mAP50"))
        if value is not None:
            values.append(float(value))
    return values[-1] if values else None


def evaluation_specs(
    checkpoint: str,
    num_queries: int,
    *,
    num_classes: int = 5,
) -> dict[str, Any]:
    specs = copy.deepcopy(yaml.safe_load(EVALUATE_TEMPLATE.read_text()))
    specs["wandb"]["enable"] = False
    specs["model"]["backbone"] = "resnet_50"
    specs["model"]["num_queries"] = int(num_queries)
    specs["model"]["num_select"] = min(300, int(num_queries))
    specs["dataset"]["num_classes"] = int(num_classes)
    specs["dataset"]["eval_class_ids"] = [1, 2, 3, 4]
    specs["dataset"]["batch_size"] = TRAIN_BATCH_SIZE_PER_GPU
    specs["dataset"]["workers"] = 8
    specs["dataset"]["augmentation"]["test_random_resize"] = 800
    specs["dataset"]["augmentation"]["random_resize_max_size"] = 1333
    specs["dataset"]["augmentation"]["fixed_padding"] = True
    specs["dataset"]["test_data_sources"]["image_dir"] = (
        f"{DATA_ROOT}/val/images/images"
    )
    specs["dataset"]["test_data_sources"]["json_file"] = (
        f"{DATA_ROOT}/val/annotations.json"
    )
    specs["evaluate"]["batch_size"] = TRAIN_BATCH_SIZE_PER_GPU
    specs["evaluate"]["num_gpus"] = GPU_COUNT
    specs["evaluate"]["gpu_ids"] = list(range(GPU_COUNT))
    specs["evaluate"]["num_nodes"] = 1
    specs["evaluate"]["checkpoint"] = checkpoint
    return specs


def wait_for_job(
    sdk: SlurmSDK,
    job_id: str,
    *,
    event_path: Path,
    phase: str,
    seed: int,
    rec_id: int,
) -> str:
    last_status = None
    while True:
        status = sdk.get_job_status(job_id).status
        if status != last_status:
            append_event(
                event_path,
                {
                    "event": "job_status",
                    "phase": phase,
                    "seed": seed,
                    "rec_id": rec_id,
                    "job_id": job_id,
                    "status": status,
                },
            )
            print(
                f"JOB_STATUS phase={phase} seed={seed} rec={rec_id} "
                f"job={job_id} status={status}",
                flush=True,
            )
            last_status = status
        if status in TERMINAL:
            return status
        time.sleep(10)


def launch_accuracy_evaluation(
    sdk: SlurmSDK,
    checkpoint: str,
    num_queries: int,
    *,
    event_path: Path,
    seed: int,
    rec_id: int,
    num_classes: int = 5,
) -> tuple[float, dict[str, Any]]:
    entrypoint = build_entrypoint(
        command="dino evaluate -e {config_path}",
        specs=evaluation_specs(
            checkpoint,
            num_queries,
            num_classes=num_classes,
        ),
        inputs=EVALUATE_INPUTS,
        outputs=EVALUATE_OUTPUTS,
        config_format="yaml",
        upload_excludes=["inputs/"],
    )
    job = sdk.create_job(
        image=SQSH_PATH,
        command=entrypoint["command"],
        gpu_count=GPU_COUNT,
        num_nodes=1,
        partition=PARTITION,
        account=ACCOUNT,
    )
    runtime = sdk._handler.get_job_runtime_identity(job.id)
    status = wait_for_job(
        sdk,
        job.id,
        event_path=event_path,
        phase="accuracy_evaluation",
        seed=seed,
        rec_id=rec_id,
    )
    logs = sdk.get_job_logs(job.id, tail=5000)
    if status != "Complete":
        raise RuntimeError(
            f"accuracy evaluation job {job.id} ended as {status}: "
            f"{logs[-4000:]}"
        )
    map50 = read_status_map50(sdk, job.id)
    if map50 is None:
        matches = []
        for pattern in MAP50_PATTERNS:
            matches.extend(float(value) for value in pattern.findall(logs))
        map50 = matches[-1] if matches else None
    if map50 is None:
        raise RuntimeError(f"accuracy evaluation job {job.id} emitted no mAP50")
    return float(map50), {
        "job_id": job.id,
        "slurm_job_id": runtime.get("slurm_job_id", ""),
        "result_root": local_lustre_path(sdk.get_job_results_dir(job.id)),
        "status": status,
    }


def benchmark_command(checkpoint: str) -> str:
    source_b64 = base64.b64encode(BENCHMARK_SCRIPT.read_bytes()).decode("ascii")
    install_script = (
        "import base64;"
        "open('/tmp/dino_latency_benchmark.py','wb').write("
        f"base64.b64decode('{source_b64}'))"
    )
    return " ".join(
        [
            "python -c",
            shlex.quote(install_script),
            "&&",
            "torchrun",
            "--standalone",
            f"--nproc_per_node={GPU_COUNT}",
            "/tmp/dino_latency_benchmark.py",
            "--config",
            "{config_path}",
            "--checkpoint",
            shlex.quote(checkpoint),
            "--output-root",
            '"$TAO_RESULTS_ROOT"',
            "--warmup-iterations",
            str(LATENCY_WARMUPS),
            "--timed-iterations",
            str(LATENCY_ITERATIONS),
            "--rounds",
            str(LATENCY_ROUNDS),
            "--preloaded-batches",
            str(LATENCY_PRELOADED_BATCHES),
            "--seed",
            str(LATENCY_BENCHMARK_SEED),
        ]
    )


def read_latency_rank_records(
    sdk: SlurmSDK,
    benchmark_job_id: str,
) -> list[dict[str, Any]]:
    result_root = local_lustre_path(sdk.get_job_results_dir(benchmark_job_id))
    latency_dir = f"{result_root}/latency"
    reader = (
        "import glob,json,sys;"
        "paths=sorted(glob.glob(sys.argv[1]+'/rank_*.json'));"
        "print(json.dumps([json.load(open(path)) for path in paths]))"
    )
    output = remote_output(
        f"python3 -c {shlex.quote(reader)} {shlex.quote(latency_dir)}",
        timeout=300,
    )
    records = json.loads(output)
    if len(records) != GPU_COUNT:
        raise RuntimeError(
            f"expected {GPU_COUNT} latency rank records, found {len(records)}"
        )
    if {int(record["rank"]) for record in records} != set(range(GPU_COUNT)):
        raise RuntimeError("latency rank records are incomplete or duplicated")
    return records


def enforce_hardware_contract(records: list[dict[str, Any]]) -> dict[str, Any]:
    signatures = {
        (
            record["hardware"]["gpu_name"],
            record["hardware"]["compute_capability"],
            int(record["hardware"]["total_memory_bytes"]),
            record["runtime"]["torch"],
            record["runtime"]["cuda"],
            record["runtime"]["cudnn"],
        )
        for record in records
    }
    if len(signatures) != 1:
        raise RuntimeError(
            f"benchmark ranks do not share one hardware/runtime signature: {signatures}"
        )
    signature = next(iter(signatures))
    contract = {
        "gpu_name": signature[0],
        "compute_capability": signature[1],
        "total_memory_bytes": signature[2],
        "torch": signature[3],
        "cuda": signature[4],
        "cudnn": signature[5],
        "world_size": GPU_COUNT,
        "sqsh_path": SQSH_PATH,
    }

    contract_path = EXPERIMENT_DIR / "hardware_contract.json"
    lock_path = EXPERIMENT_DIR / ".hardware_contract.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        if contract_path.exists():
            expected = json.loads(contract_path.read_text())
            if expected != contract:
                raise RuntimeError(
                    "benchmark hardware/runtime differs from the frozen "
                    f"experiment contract: expected={expected}, actual={contract}"
                )
        else:
            atomic_json(contract_path, contract)
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    return contract


def enforce_input_identity(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Require all eight replicas to benchmark the exact same input workload."""

    metadata = [record.get("benchmark_inputs") for record in records]
    if any(not isinstance(item, dict) for item in metadata):
        raise RuntimeError(
            "latency rank record is missing benchmark input identity metadata"
        )
    canonical = {
        json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)
        for item in metadata
    }
    if len(canonical) != 1:
        identities = [
            {
                "rank": record["rank"],
                "identity_sha256": record["benchmark_inputs"].get(
                    "identity_sha256"
                ),
            }
            for record in records
        ]
        raise RuntimeError(
            "latency benchmark ranks used different input workloads: "
            f"{identities}"
        )
    return copy.deepcopy(metadata[0])


def launch_latency_benchmark(
    sdk: SlurmSDK,
    checkpoint: str,
    num_queries: int,
    *,
    event_path: Path,
    seed: int,
    rec_id: int,
    num_classes: int = 5,
) -> tuple[dict[str, float], dict[str, Any]]:
    specs = evaluation_specs(
        checkpoint,
        num_queries,
        num_classes=num_classes,
    )
    specs["dataset"]["batch_size"] = LATENCY_BATCH_SIZE_PER_GPU
    specs["evaluate"]["batch_size"] = LATENCY_BATCH_SIZE_PER_GPU
    entrypoint = build_entrypoint(
        command=benchmark_command(checkpoint),
        specs=specs,
        inputs=EVALUATE_INPUTS,
        outputs={},
        config_format="yaml",
        upload_excludes=["inputs/"],
    )
    job = sdk.create_job(
        image=SQSH_PATH,
        command=entrypoint["command"],
        gpu_count=GPU_COUNT,
        num_nodes=1,
        partition=PARTITION,
        account=ACCOUNT,
    )
    runtime = sdk._handler.get_job_runtime_identity(job.id)
    status = wait_for_job(
        sdk,
        job.id,
        event_path=event_path,
        phase="latency_benchmark",
        seed=seed,
        rec_id=rec_id,
    )
    logs = sdk.get_job_logs(job.id, tail=5000)
    if status != "Complete" or "TAO_AUTOML_LATENCY_COMPLETE" not in logs:
        raise RuntimeError(
            f"latency benchmark job {job.id} did not complete cleanly "
            f"(status={status}): {logs[-6000:]}"
        )

    rank_records = read_latency_rank_records(sdk, job.id)
    hardware_contract = enforce_hardware_contract(rank_records)
    benchmark_inputs = enforce_input_identity(rank_records)
    samples = {
        round_index: {
            str(record["rank"]): record["samples_ms"][round_index]
            for record in rank_records
        }
        for round_index in range(LATENCY_ROUNDS)
    }
    statistics = aggregate_synchronized_latency(samples, LATENCY_PROTOCOL)
    statistics_dict = asdict(statistics)
    if not statistics.is_valid:
        raise RuntimeError(
            "latency measurement failed frozen quality thresholds: "
            + ",".join(statistics.invalid_reasons)
        )
    flat_metrics = {
        "latency_ms": statistics.median_ms,
        "latency_p95_ms": statistics.tail_latency_ms,
        "latency_mad_ms": statistics.mad_ms,
        "latency_iqr_ms": statistics.iqr_ms,
        "latency_robust_cv": statistics.robust_cv,
        "latency_ci95_low": statistics.bootstrap_median_ci_ms[0],
        "latency_ci95_high": statistics.bootstrap_median_ci_ms[1],
        "latency_bootstrap_ci_width_ms": statistics.bootstrap_ci_width_ms,
        "latency_round_drift_fraction": statistics.round_drift_fraction,
        "latency_device_range_fraction": (
            statistics.device_median_range_fraction
        ),
        "latency_synchronized_median_ms": (
            statistics.synchronized_median_ms
        ),
        "latency_synchronized_p95_ms": (
            statistics.synchronized_tail_latency_ms
        ),
    }
    return flat_metrics, {
        "job_id": job.id,
        "slurm_job_id": runtime.get("slurm_job_id", ""),
        "result_root": local_lustre_path(sdk.get_job_results_dir(job.id)),
        "raw_samples_dir": (
            f"{local_lustre_path(sdk.get_job_results_dir(job.id))}/latency"
        ),
        "status": status,
        "statistics": statistics_dict,
        "hardware_contract": hardware_contract,
        "benchmark_inputs": benchmark_inputs,
        "rank_input_identities": [
            {
                "rank": record["rank"],
                "identity_sha256": record["benchmark_inputs"][
                    "identity_sha256"
                ],
            }
            for record in rank_records
        ],
        "rank_hardware": [
            {
                "rank": record["rank"],
                "hardware": record["hardware"],
                "runtime": record["runtime"],
            }
            for record in rank_records
        ],
    }


def configure_slurm_environment() -> None:
    # The image argument is already an absolute, verified SQSH path. Disabling
    # conversion prevents registry fallback and makes the sbatch image identity
    # fail closed while Pyxis still receives the .sqsh path directly.
    os.environ["SLURM_USE_SQSH"] = "false"
    os.environ["SLURM_PARTITION"] = PARTITION
    os.environ["SLURM_ACCOUNT"] = ACCOUNT


def run_smoke() -> dict[str, Any]:
    smoke_dir = EXPERIMENT_DIR / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    event_path = smoke_dir / "events.jsonl"
    result_path = smoke_dir / "result.json"
    configure_slurm_environment()
    sdk = SlurmSDK(
        poll_interval=10,
        state_file=smoke_dir / "slurm_state.json",
    )
    try:
        map50, accuracy_job = launch_accuracy_evaluation(
            sdk,
            PTM,
            900,
            event_path=event_path,
            seed=TRAIN_SEED,
            rec_id=-1,
            num_classes=91,
        )
        latency_metrics, latency_job = launch_latency_benchmark(
            sdk,
            PTM,
            900,
            event_path=event_path,
            seed=TRAIN_SEED,
            rec_id=-1,
            num_classes=91,
        )
        result = {
            "status": "success",
            "purpose": (
                "compatible DINO ResNet50 PTM baseline and end-to-end "
                "eight-GPU SQSH launch validation"
            ),
            "dataset": DATASET_URI,
            "checkpoint": PTM,
            "checkpoint_classifier_classes": 91,
            "mAP50": map50,
            "latency_metrics": latency_metrics,
            "accuracy_evaluation": accuracy_job,
            "latency_benchmark": latency_job,
            "latency_protocol": asdict(LATENCY_PROTOCOL),
        }
    except BaseException as exc:
        result = {
            "status": "failure",
            "error": f"{type(exc).__name__}: {exc}",
        }
        atomic_json(result_path, result)
        raise
    atomic_json(result_path, result)
    print("SMOKE_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return result


def automl_settings(seed: int) -> dict[str, Any]:
    return {
        "algorithm": "bayesian",
        "automl_max_recommendations": RECOMMENDATIONS_PER_SEED,
        "automl_max_concurrent": 1,
        "session_id": f"dino_moo_review_seed_{seed}",
        "experiment_id": f"dino_moo_review_seed_{seed}",
        "random_seed": seed,
        "selection_mode": "multi_objective",
        "objectives": [
            {
                "metric": "mAP50",
                "direction": "maximize",
                "weight": 1.0,
            },
            {
                "metric": "latency_ms",
                "direction": "minimize",
                "weight": 1.0,
            },
        ],
        "accuracy_constraint": {
            "type": "relative",
            "retained_fraction": ACCURACY_RETENTION_FRACTION,
            "reference": "accuracy_winner",
        },
        "objective_normalization": "pareto_front",
        "augmentation_rho": 1.0e-6,
        "accuracy_tolerance": 1.0e-12,
        "latency_tolerance": 0.0,
        "selection_score_tolerance": 1.0e-12,
        "latency_ci_low_metric": "latency_ci95_low",
        "latency_ci_high_metric": "latency_ci95_high",
        "require_eval_fn_success": True,
        "run_baseline": True,
        "baseline_metric": BASELINE_MAP50_INFORMATIONAL,
        "run_final_evaluation": True,
        "reuse_best_metric_for_final_evaluation": True,
        "automl_delete_intermediate_ckpt": True,
        "automl_checkpoint_retention_strategy": "terminal",
    }


def run_seed(seed: int) -> None:
    seed_dir = EXPERIMENT_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    event_path = seed_dir / "events.jsonl"
    evaluations_path = seed_dir / "candidate_evaluations.json"
    result_path = seed_dir / "result.json"
    evaluations: list[dict[str, Any]] = []

    configure_slurm_environment()

    sdk = SlurmSDK(
        poll_interval=10,
        state_file=seed_dir / "slurm_state.json",
    )
    runner = AutoMLRunner(
        sdk=sdk,
        skill_dir=SKILL_DIR,
        action="train",
        poll_interval=10,
    )

    def evaluate_candidate(rec: Any, train_job_id: str):
        checkpoint = find_terminal_checkpoint(sdk, train_job_id)
        num_queries = int(rec.specs.get("model.num_queries", 900))
        record = {
            "candidate_id": f"seed_{seed}_rec_{rec.id}",
            "search_seed": seed,
            "training_seed": TRAIN_SEED,
            "rec_id": int(rec.id),
            "specs": dict(rec.specs),
            "train_job_id": train_job_id,
            "checkpoint": checkpoint,
            "num_queries": num_queries,
            "mAP50": None,
            "latency": None,
            "status": "evaluating",
        }
        evaluations.append(record)
        atomic_json(evaluations_path, {"evaluations": evaluations})
        try:
            map50, accuracy_job = launch_accuracy_evaluation(
                sdk,
                checkpoint,
                num_queries,
                event_path=event_path,
                seed=seed,
                rec_id=int(rec.id),
            )
            record["mAP50"] = map50
            record["accuracy_evaluation"] = accuracy_job
            atomic_json(evaluations_path, {"evaluations": evaluations})

            latency_metrics, latency_job = launch_latency_benchmark(
                sdk,
                checkpoint,
                num_queries,
                event_path=event_path,
                seed=seed,
                rec_id=int(rec.id),
            )
            record["latency"] = latency_job
            record["objective_values"] = {
                "mAP50": map50,
                **latency_metrics,
            }
            record["status"] = "success"
            atomic_json(evaluations_path, {"evaluations": evaluations})
            append_event(
                event_path,
                {
                    "event": "candidate_measurement_complete",
                    **record,
                },
            )
            return dict(record["objective_values"])
        except BaseException as exc:
            record["status"] = "failure"
            record["failure_reason"] = f"{type(exc).__name__}: {exc}"
            atomic_json(evaluations_path, {"evaluations": evaluations})
            append_event(
                event_path,
                {
                    "event": "candidate_measurement_failure",
                    **record,
                },
            )
            raise

    def on_recommendation(rec: Any) -> None:
        append_event(
            event_path,
            {
                "event": "recommendation",
                "seed": seed,
                "rec_id": int(rec.id),
                "specs": rec.specs,
            },
        )

    def on_result(rec: Any, metric: Any, status: str) -> None:
        append_event(
            event_path,
            {
                "event": "result",
                "seed": seed,
                "rec_id": int(rec.id),
                "train_job_id": rec.job_id,
                "metric": metric,
                "status": status,
            },
        )

    try:
        result = runner.run(
            train_dataset_uri=DATASET_URI,
            eval_dataset_uri=DATASET_URI,
            base_checkpoint=PTM,
            workspace_id=f"dino-moo-review-{seed}",
            image=SQSH_PATH,
            automl_settings=automl_settings(seed),
            automl_hyperparameters=list(SEARCH_PARAMETERS),
            custom_param_ranges=SEARCH_RANGES,
            workspace_path=str(seed_dir / "workspace"),
            spec_overrides=SPEC_OVERRIDES,
            metric_extractor=dino_metric_extractor,
            eval_fn=evaluate_candidate,
            on_recommendation=on_recommendation,
            on_result=on_result,
            gpu_count=GPU_COUNT,
            num_nodes=1,
            partition=PARTITION,
            account=ACCOUNT,
        )
    except BaseException as exc:
        atomic_json(
            result_path,
            {
                "seed": seed,
                "status": "failure",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    atomic_json(
        result_path,
        {
            "seed": seed,
            "status": "success",
            "result": result,
        },
    )


def union_candidate_records() -> list[dict[str, Any]]:
    records = []
    for seed in SEARCH_SEEDS:
        path = EXPERIMENT_DIR / f"seed_{seed}" / "candidate_evaluations.json"
        payload = json.loads(path.read_text())
        for record in payload["evaluations"]:
            if record.get("status") == "success":
                records.append(record)
    return records


def select_union_archive() -> dict[str, Any]:
    records = union_candidate_records()
    candidates = [
        SimpleNamespace(
            id=record["candidate_id"],
            specs=record["specs"],
            status="success",
            objective_values=record["objective_values"],
        )
        for record in records
    ]
    objective_config = parse_objective_config(automl_settings(SEARCH_SEEDS[0]))
    analysis = objective_config.analyze_archive(candidates)
    result = analysis.to_dict()
    result["search"] = {
        "seeds": list(SEARCH_SEEDS),
        "recommendations_per_seed": RECOMMENDATIONS_PER_SEED,
        "successful_candidates": len(records),
        "candidate_generation": (
            "three seeded Bayesian sub-archives; final selectors use their union"
        ),
    }
    result["candidate_records"] = {
        record["candidate_id"]: record for record in records
    }
    atomic_json(EXPERIMENT_DIR / "combined_selection.json", result)
    write_candidate_csv(result)
    return result


def write_candidate_csv(result: dict[str, Any]) -> None:
    records = result["candidate_records"]
    audits = {
        item["candidate_id"]: item
        for item in result["candidates"]
    }
    path = EXPERIMENT_DIR / "full_candidate_table.csv"
    columns = [
        "candidate_id",
        "search_seed",
        "training_seed",
        "rec_id",
        "model.num_queries",
        "train.optim.lr",
        "train.optim.weight_decay",
        "mAP50",
        "latency_ms",
        "latency_p95_ms",
        "latency_mad_ms",
        "latency_iqr_ms",
        "latency_robust_cv",
        "latency_ci95_low",
        "latency_ci95_high",
        "accuracy_feasible",
        "pareto_rank",
        "feasible_pareto_rank",
        "dominated_by",
        "feasible_dominated_by",
        "normalized_accuracy_objective",
        "normalized_latency_objective",
        "multi_objective_compromise_score",
        "accuracy_winner",
        "latency_winner",
        "multi_objective_winner",
        "train_job_id",
        "accuracy_eval_job_id",
        "latency_job_id",
        "checkpoint",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for candidate_id in sorted(records):
            record = records[candidate_id]
            values = record["objective_values"]
            audit = audits[candidate_id]
            winner = audit["winner"]
            writer.writerow(
                {
                    "candidate_id": candidate_id,
                    "search_seed": record["search_seed"],
                    "training_seed": record["training_seed"],
                    "rec_id": record["rec_id"],
                    "model.num_queries": record["specs"].get(
                        "model.num_queries"
                    ),
                    "train.optim.lr": record["specs"].get("train.optim.lr"),
                    "train.optim.weight_decay": record["specs"].get(
                        "train.optim.weight_decay"
                    ),
                    "mAP50": values["mAP50"],
                    "latency_ms": values["latency_ms"],
                    "latency_p95_ms": values["latency_p95_ms"],
                    "latency_mad_ms": values["latency_mad_ms"],
                    "latency_iqr_ms": values["latency_iqr_ms"],
                    "latency_robust_cv": values["latency_robust_cv"],
                    "latency_ci95_low": values["latency_ci95_low"],
                    "latency_ci95_high": values["latency_ci95_high"],
                    "accuracy_feasible": audit["accuracy_feasible"],
                    "pareto_rank": audit["pareto_rank"],
                    "feasible_pareto_rank": audit["feasible_pareto_rank"],
                    "dominated_by": ";".join(audit["dominated_by"]),
                    "feasible_dominated_by": ";".join(
                        audit["feasible_dominated_by"]
                    ),
                    "normalized_accuracy_objective": audit[
                        "normalized_accuracy_objective"
                    ],
                    "normalized_latency_objective": audit[
                        "normalized_latency_objective"
                    ],
                    "multi_objective_compromise_score": audit[
                        "multi_objective_compromise_score"
                    ],
                    "accuracy_winner": winner["accuracy"],
                    "latency_winner": winner["latency"],
                    "multi_objective_winner": winner["multi_objective"],
                    "train_job_id": record["train_job_id"],
                    "accuracy_eval_job_id": record["accuracy_evaluation"][
                        "job_id"
                    ],
                    "latency_job_id": record["latency"]["job_id"],
                    "checkpoint": record["checkpoint"],
                }
            )


def write_launch_manifest() -> None:
    launch_manifest = {
        "created_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "repositories": {
            name: repository_identity(path)
            for name, path in REPOSITORIES.items()
        },
        "source_dataset": DATASET_URI,
        "staged_dataset": DATA_ROOT,
        "dataset_inventory": {
            "train_images": 1414,
            "train_annotations": 8395,
            "train_annotation_sha256": (
                "7401a1245dc0b691c40f9f53cf4f46f9"
                "b96a3e0bc3dcfd357de038074acc1994"
            ),
            "validation_images": 353,
            "validation_annotations": 2186,
            "validation_annotation_sha256": (
                "9b715b689e9a17588805faad26ed9459"
                "7886d28ac687438dcb778de433f997af"
            ),
            "category_ids": [1, 2, 3, 4],
        },
        "model": "DINO ResNet50",
        "ptm": PTM,
        "image_uri": IMAGE_URI,
        "sqsh_path": SQSH_PATH,
        "direct_sqsh": True,
        "partition": PARTITION,
        "account": ACCOUNT,
        "gpu_count_per_job": GPU_COUNT,
        "num_nodes": 1,
        "search_seeds": list(SEARCH_SEEDS),
        "training_seed": TRAIN_SEED,
        "recommendations_per_seed": RECOMMENDATIONS_PER_SEED,
        "total_candidate_budget": (
            len(SEARCH_SEEDS) * RECOMMENDATIONS_PER_SEED
        ),
        "training_epochs": TRAIN_EPOCHS,
        "training_runtime": {
            "precision": "fp32",
            "distributed_strategy": "ddp",
            "batch_size_per_gpu": TRAIN_BATCH_SIZE_PER_GPU,
            "global_batch_size": TRAIN_BATCH_SIZE_PER_GPU * GPU_COUNT,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        },
        "accuracy_evaluation": {
            "metric": "mAP50",
            "batch_size_per_gpu": TRAIN_BATCH_SIZE_PER_GPU,
            "precision": "fp32",
            "test_random_resize": 800,
            "random_resize_max_size": 1333,
            "fixed_padding": True,
            "gpu_count": GPU_COUNT,
        },
        "smoke_baseline": {
            "checkpoint": PTM,
            "checkpoint_classifier_classes": 91,
            "evaluation_class_ids": [1, 2, 3, 4],
            "purpose": (
                "raw COCO PTM compatibility and launch validation only; "
                "never used as the retained-accuracy reference"
            ),
        },
        "search_parameters": list(SEARCH_PARAMETERS),
        "search_ranges": SEARCH_RANGES,
        "spec_overrides": SPEC_OVERRIDES,
        "selection": automl_settings(SEARCH_SEEDS[0]),
        "latency_protocol": asdict(LATENCY_PROTOCOL),
        "latency_benchmark": {
            "batch_size_per_gpu": LATENCY_BATCH_SIZE_PER_GPU,
            "preloaded_batches": LATENCY_PRELOADED_BATCHES,
            "benchmark_seed": LATENCY_BENCHMARK_SEED,
            "precision": "fp32",
            "tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "timed_scope": "model_forward_plus_dino_gpu_postprocess",
            "excluded_scope": [
                "checkpoint_load",
                "disk_io",
                "decode_resize_normalize",
                "host_to_device_transfer",
                "coco_accumulation",
                "distributed_gather",
            ],
            "synchronization": (
                "cuda_sync_each_sample_and_nccl_barrier"
            ),
            "primary_estimator": "median_of_device_round_medians",
            "raw_samples_per_candidate": (
                GPU_COUNT * LATENCY_ROUNDS * LATENCY_ITERATIONS
            ),
        },
        "shared_archive": {
            "generation": (
                "three seeded sequential Bayesian sub-archives"
            ),
            "selection_population": (
                "union of every successful measured candidate"
            ),
            "all_modes_receive_identical_candidates": True,
        },
    }
    atomic_json(EXPERIMENT_DIR / "launch_manifest.json", launch_manifest)


def require_successful_smoke() -> None:
    result_path = EXPERIMENT_DIR / "smoke" / "result.json"
    if not result_path.exists():
        raise RuntimeError(
            "full run is blocked until the reviewed smoke succeeds; run "
            f"{Path(__file__).name} --smoke first"
        )
    result = json.loads(result_path.read_text())
    if result.get("status") != "success":
        raise RuntimeError(
            "full run is blocked because the reviewed smoke did not succeed: "
            f"{result}"
        )
    if not (EXPERIMENT_DIR / "hardware_contract.json").exists():
        raise RuntimeError(
            "full run is blocked because the smoke did not freeze a hardware "
            "contract"
        )


def main() -> int:
    args = parse_args()
    write_launch_manifest()
    if args.smoke:
        run_smoke()
        return 0
    if args.combine_only:
        combined = select_union_archive()
        print(
            "COMBINED_SELECTION "
            + json.dumps(combined["selections"], sort_keys=True),
            flush=True,
        )
        return 0
    require_successful_smoke()

    context = mp.get_context("spawn")
    processes = {
        seed: context.Process(
            target=run_seed,
            args=(seed,),
            name=f"dino-moo-seed-{seed}",
        )
        for seed in SEARCH_SEEDS
    }
    for process in processes.values():
        process.start()

    def forward_signal(signum: int, _frame: object) -> None:
        for process in processes.values():
            if process.is_alive() and process.pid:
                os.kill(process.pid, signum)

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)

    exit_codes: dict[int, int | None] = {}
    while processes:
        for seed, process in list(processes.items()):
            process.join(timeout=1)
            if not process.is_alive():
                exit_codes[seed] = process.exitcode
                processes.pop(seed)
                print(
                    f"SEED_PROCESS_EXIT seed={seed} "
                    f"exitcode={process.exitcode}",
                    flush=True,
                )
        if processes:
            time.sleep(1)
    atomic_json(
        EXPERIMENT_DIR / "process_status.json",
        {"exit_codes": exit_codes},
    )
    if not all(code == 0 for code in exit_codes.values()):
        return 1
    combined = select_union_archive()
    print(
        "COMBINED_SELECTION "
        + json.dumps(combined["selections"], sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s",
    )
    raise SystemExit(main())
