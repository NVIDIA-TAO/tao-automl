#!/usr/bin/env python3

"""Run three independent full-VOC2007 DINO AutoML jobs on SLURM.

Dry-run is the default and never constructs an SDK.  ``--launch`` starts one
independent controller for accuracy, constrained latency, and multi-objective
search.  Each controller proposes its own candidates and each candidate is a
real eight-GPU train, evaluate, and stabilized-latency sequence.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import multiprocessing as mp
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping

import yaml

try:
    from .manifest_generator import (
        AGENT_FLAGS,
        MODES,
        SEARCH_PARAMETERS,
        SELECTION_FLAGS,
        load_manifest,
        manifest_sha256,
        sha256_file,
    )
except ImportError:  # pragma: no cover - direct script execution
    from manifest_generator import (  # type: ignore[no-redef]
        AGENT_FLAGS,
        MODES,
        SEARCH_PARAMETERS,
        SELECTION_FLAGS,
        load_manifest,
        manifest_sha256,
        sha256_file,
    )


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "campaign.v1.json"
DEFAULT_RUNTIME = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/dino_three_mode"
)
ENV_PATH = Path("/localhome/local-rarunachalam/.tao/config.env")
TERMINAL_JOB_STATUSES = frozenset({"Complete", "Error", "Canceled"})
SUCCESS_RECOMMENDATION_STATUSES = frozenset({"success", "done"})
MAP50_PATTERNS = (
    re.compile(
        r"(?:Validation|Test)\s+mAP50\s*[:=]\s*"
        r"([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)"
    ),
    re.compile(
        r"\b(?:test|val)_mAP50\b[^0-9+\-]*"
        r"([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)"
    ),
)


class CampaignExecutionError(RuntimeError):
    """The sealed direct campaign cannot safely continue."""


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_env_file(path: Path = ENV_PATH) -> tuple[str, ...]:
    """Load required launch variables without returning or logging values."""
    if not path.is_file():
        raise CampaignExecutionError(f"secrets file is unavailable: {path}")
    names = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if (
            not name
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
        ):
            raise CampaignExecutionError("secrets file contains an invalid key")
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        os.environ[name] = value
        names.append(name)
    required = {"SLURM_HOSTNAME", "SLURM_USER"}
    missing = sorted(name for name in required if not os.environ.get(name))
    if missing:
        raise CampaignExecutionError(
            "secrets file did not define required keys: " + ", ".join(missing)
        )
    return tuple(sorted(set(names)))


def _ssh_command(command: str) -> list[str]:
    host = os.environ["SLURM_HOSTNAME"].split(",", 1)[0].strip()
    user = os.environ["SLURM_USER"].strip()
    result = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
    ]
    key_path = os.environ.get("SSH_KEY_PATH")
    if key_path:
        result.extend(["-i", key_path])
    result.extend([f"{user}@{host}", command])
    return result


def remote_output(command: str, *, timeout: int = 900) -> str:
    return subprocess.run(
        _ssh_command(command),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    ).stdout


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def verify_local_launch_contract(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind launch HEAD and the exact clean SDK/skills/package sources."""
    source = manifest["source"]
    runtime = manifest["runtime"]
    repository = Path(source["repository"])
    launch_head = _git(repository, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "merge-base",
            "--is-ancestor",
            source["commit"],
            launch_head,
        ],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if ancestor.returncode != 0:
        raise CampaignExecutionError(
            "sealed manifest source commit is not an ancestor of launch HEAD"
        )
    if _git(repository, "status", "--porcelain"):
        raise CampaignExecutionError("launch repository is not clean")
    if sha256_file(source["wheel_path"]) != source["wheel_sha256"]:
        raise CampaignExecutionError("production wheel identity changed")
    revisions = {}
    for label, path_key, revision_key in (
        ("sdk", "sdk_dir", "sdk_revision"),
        ("skills", "skill_dir", "skill_revision"),
    ):
        path = Path(runtime[path_key])
        revision = _git(path, "rev-parse", "HEAD")
        if revision != runtime[revision_key]:
            raise CampaignExecutionError(f"{label} revision changed")
        if _git(path, "status", "--porcelain"):
            raise CampaignExecutionError(f"{label} source is not clean")
        revisions[label] = {
            "path": str(path),
            "commit": revision,
            "clean": True,
        }
    return {
        "manifest_source_commit": source["commit"],
        "launch_head": launch_head,
        "manifest_source_is_ancestor": True,
        "wheel_path": source["wheel_path"],
        "wheel_sha256": source["wheel_sha256"],
        "wheel_source_commit": source["wheel_source_commit"],
        "sources": revisions,
    }


def configure_slurm_runtime(manifest: Mapping[str, Any]) -> None:
    """Apply the frozen clean-SDK environment before importing ``tao_sdk``."""
    runtime = manifest["runtime"]
    sdk_dir = runtime["sdk_dir"]
    sys.path = [sdk_dir, *[item for item in sys.path if item != sdk_dir]]
    existing = os.environ.get("PYTHONPATH", "")
    components = [
        sdk_dir,
        *[item for item in existing.split(os.pathsep) if item and item != sdk_dir],
    ]
    os.environ["PYTHONPATH"] = os.pathsep.join(components)
    os.environ.update(
        {
            "SLURM_USE_SQSH": "false",
            "SLURM_USE_REQUEUE": "true",
            "SLURM_TIME_HOURS": "4.0",
            "SLURM_TIMEOUT_HOURS": "3.8",
            "SLURM_MAX_GPUS_PER_NODE": "8",
            "SLURM_PARTITION": runtime["partition"],
            "SLURM_ACCOUNT": runtime["account"],
            "SLURM_BASE_RESULTS_DIR": runtime["base_results_dir"],
            "SLURM_CONTAINER_MOUNTS": runtime["container_mounts"],
        }
    )


def verify_remote_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Verify immutable remote artifacts before constructing SlurmSDK."""
    runtime = manifest["runtime"]
    dataset = manifest["dataset"]
    files = {
        "sqsh": (
            runtime["sqsh_path"],
            runtime["sqsh_sha256"],
            runtime["sqsh_size_bytes"],
        ),
        "train_annotation": (
            dataset["train_annotation"],
            dataset["train_annotation_sha256"],
            dataset["train_annotation_size_bytes"],
        ),
        "validation_annotation": (
            dataset["validation_annotation"],
            dataset["validation_annotation_sha256"],
            dataset["validation_annotation_size_bytes"],
        ),
    }
    for ptm in manifest["ptms"]:
        files[f"ptm:{ptm['id']}"] = (
            ptm["slurm_path"],
            ptm["runtime_artifact"]["sha256"],
            ptm["runtime_artifact"]["size_bytes"],
        )
    checks = {}
    for label, (path, expected_sha, expected_size) in files.items():
        probe = remote_output(
            " ".join(
                [
                    "test -f",
                    shlex.quote(path),
                    "&&",
                    "stat -c '%s'",
                    shlex.quote(path),
                    "&&",
                    "sha256sum",
                    shlex.quote(path),
                ]
            ),
            timeout=7200 if label == "sqsh" else 1800,
        ).strip().splitlines()
        if len(probe) != 2:
            raise CampaignExecutionError(
                f"remote artifact probe was incomplete for {label}"
            )
        size = int(probe[0])
        observed_sha = probe[1].split()[0]
        if size != expected_size or observed_sha != expected_sha:
            raise CampaignExecutionError(
                f"remote artifact identity mismatch for {label}"
            )
        checks[label] = {
            "path": path,
            "size_bytes": size,
            "sha256": observed_sha,
        }
    image_probe = (
        "import hashlib,json,pathlib,sys;"
        "root=pathlib.Path(sys.argv[1]);"
        "items=sorted((p for p in root.iterdir() if p.is_file()),"
        "key=lambda p:p.name);"
        "outer=hashlib.sha256();total=0;"
        "exec(\"for p in items:\\n"
        " h=hashlib.sha256(p.read_bytes()).hexdigest()\\n"
        " outer.update(f'{h}  {p.name}\\\\n'.encode())\\n"
        " total+=p.stat().st_size\");"
        "print(json.dumps({'algorithm':"
        "'sha256_of_sorted_sha256sum_basename_lines',"
        "'sha256':outer.hexdigest(),'file_count':len(items),"
        "'total_bytes':total},sort_keys=True))"
    )
    image_tree = json.loads(
        remote_output(
            f"python3 -c {shlex.quote(image_probe)} "
            f"{shlex.quote(dataset['train_image_dir'])}",
            timeout=3600,
        ).strip()
    )
    if image_tree != dataset["image_tree"]:
        raise CampaignExecutionError(
            "remote VOC2007 image-tree identity does not match the manifest"
        )
    checks["voc_images"] = {
        "path": dataset["train_image_dir"],
        **image_tree,
    }
    return checks


def _mode_record(manifest: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
    matches = [item for item in manifest["modes"] if item["mode"] == mode]
    if len(matches) != 1:
        raise CampaignExecutionError(f"manifest has no unique {mode} mode")
    return matches[0]


def mode_settings(
    manifest: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Translate one sealed mode contract to production AutoML settings."""
    record = _mode_record(manifest, mode)
    objective = record["objective"]
    search = manifest["search"]
    settings = {
        "algorithm": "bayesian",
        "automl_max_recommendations": search["candidate_budget_per_mode"],
        "automl_max_concurrent": 1,
        "campaign_id": manifest["campaign_id"],
        "job_id": record["job_id"],
        "session_id": record["session_id"],
        "experiment_id": record["observation_namespace"],
        "random_seed": search["search_seed"],
        "objectives": [
            {
                "metric": item["metric"],
                "direction": item["direction"],
            }
            for item in objective["objectives"]
        ],
        "selection_mode": mode,
        "accuracy_metric": "mAP50",
        "latency_metric": "latency_ms",
        "multi_objective_min_accuracy": None,
        "objective_acquisition": {
            "calibration_points": search["calibration_points"],
            "augmentation_rho": 1.0e-6,
        },
        "objective_normalization": "pareto_front",
        "augmentation_rho": 1.0e-6,
        "accuracy_tolerance": 1.0e-12,
        "latency_tolerance": search["latency_practical_tolerance_ms"],
        "selection_score_tolerance": 1.0e-12,
        "latency_ci_low_metric": "latency_ci95_low_ms",
        "latency_ci_high_metric": "latency_ci95_high_ms",
        "run_baseline": False,
        "run_final_evaluation": False,
        "require_eval_fn_success": True,
        "automl_delete_intermediate_ckpt": False,
        "automl_checkpoint_retention_strategy": "terminal",
    }
    if mode == "latency":
        settings["latency_accuracy_retention"] = {
            "type": "relative",
            "retained_fraction": search["latency_accuracy_retention"],
            "reference": "accuracy_winner",
        }
    return settings


def custom_ranges(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for parameter in SEARCH_PARAMETERS:
        domain = manifest["search"]["space"][parameter]
        if domain["type"] == "categorical":
            result[parameter] = {"valid_options": list(domain["values"])}
        else:
            result[parameter] = {
                "valid_min": domain["minimum"],
                "valid_max": domain["maximum"],
            }
    return result


def spec_overrides(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return identical full-dataset/full-training overrides for every mode."""
    if len(manifest["ptms"]) != 1:
        raise CampaignExecutionError(
            "the narrow direct launcher supports exactly one registry-supported "
            "PTM; use the typed hierarchical runtime when the supported "
            "inventory grows"
        )
    ptm = manifest["ptms"][0]
    dataset = manifest["dataset"]
    runtime = manifest["runtime"]
    search = manifest["search"]
    overrides = copy.deepcopy(ptm["default_spec_overrides"])
    overrides.update(
        {
            "dataset.train_data_sources[0].image_dir": (
                dataset["train_image_dir"]
            ),
            "dataset.train_data_sources[0].json_file": (
                dataset["train_annotation"]
            ),
            "dataset.val_data_sources[0].image_dir": (
                dataset["validation_image_dir"]
            ),
            "dataset.val_data_sources[0].json_file": (
                dataset["validation_annotation"]
            ),
            "dataset.num_classes": dataset["num_classes"],
            "dataset.eval_class_ids": dataset["eval_class_ids"],
            "dataset.batch_size": runtime["train_batch_size_per_gpu"],
            "dataset.workers": 8,
            "model.num_select": 300,
            "train.num_gpus": 8,
            "train.gpu_ids": list(range(8)),
            "train.num_nodes": 1,
            "train.num_epochs": search["training_epochs"],
            "train.validation_interval": 1,
            "train.checkpoint_interval": search["training_epochs"],
            "train.checkpoint_interval_unit": "epoch",
            "train.seed": search["training_seed"],
            "train.precision": "fp32",
            "train.distributed_strategy": "ddp",
            "train.is_dry_run": False,
            "train.cudnn.benchmark": False,
            "train.cudnn.deterministic": True,
            "wandb.enable": False,
        }
    )
    return overrides


def launch_plan(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Render the exact direct execution plan without importing an SDK."""
    return {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "execution_kind": "direct_full_search",
        "cpu_runs": 0,
        "smoke_runs": 0,
        "smoke_or_cpu_preflight_skipped_by_user": True,
        "shared_archive": False,
        "mode_jobs": [
            {
                "mode": mode,
                "job_id": _mode_record(manifest, mode)["job_id"],
                "workspace_namespace": _mode_record(manifest, mode)[
                    "observation_namespace"
                ],
                "candidate_budget": manifest["search"][
                    "candidate_budget_per_mode"
                ],
                "objective": _mode_record(manifest, mode)["objective"],
                "settings": mode_settings(manifest, mode),
                "initial_observation_ids": [],
            }
            for mode in MODES
        ],
        "per_candidate": [
            "eight_gpu_full_voc2007_train",
            "eight_gpu_voc2007_validation",
            "eight_replica_stabilized_selection_time_latency",
        ],
        "parallel_mode_controllers": 3,
        "gpus_per_slurm_job": 8,
        "slurm": {
            "sqsh_path": manifest["runtime"]["sqsh_path"],
            "slurm_use_sqsh": False,
            "slurm_use_requeue": True,
            "max_job_retries": manifest["runtime"]["max_job_retries"],
            "time_hours": manifest["runtime"]["time_hours"],
            "timeout_hours": manifest["runtime"]["timeout_hours"],
            "base_results_dir": manifest["runtime"]["base_results_dir"],
            "container_mounts": manifest["runtime"]["container_mounts"],
            "hardware_contract": manifest["runtime"]["hardware_contract"],
        },
        "ptm_ids": [item["id"] for item in manifest["ptms"]],
        "selection_isolation_flags": copy.deepcopy(
            manifest["selection_isolation_flags"]
        ),
        "agent_intervention_flags": copy.deepcopy(
            manifest["agent_intervention_flags"]
        ),
    }


def _set_dotted(target: dict[str, Any], path: str, replacement: Any) -> None:
    cursor: Any = target
    parts = path.split(".")
    for raw in parts[:-1]:
        match = re.fullmatch(r"([A-Za-z0-9_]+)\[(\d+)\]", raw)
        if match:
            cursor = cursor[match.group(1)][int(match.group(2))]
        else:
            cursor = cursor[raw]
    final = parts[-1]
    match = re.fullmatch(r"([A-Za-z0-9_]+)\[(\d+)\]", final)
    if match:
        cursor[match.group(1)][int(match.group(2))] = copy.deepcopy(replacement)
    else:
        cursor[final] = copy.deepcopy(replacement)


def _merge_spec(
    base: Mapping[str, Any],
    updates: Mapping[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in updates.items():
        if "." in key or "[" in key:
            _set_dotted(result, key, value)
        elif isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge_spec(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _evaluation_spec(
    manifest: Mapping[str, Any],
    recommendation_specs: Mapping[str, Any],
    checkpoint: str,
) -> dict[str, Any]:
    template = (
        Path(manifest["runtime"]["skill_dir"])
        / "references/spec_template_evaluate.yaml"
    )
    spec = yaml.safe_load(template.read_text(encoding="utf-8"))
    spec = _merge_spec(spec, spec_overrides(manifest))
    spec = _merge_spec(spec, recommendation_specs)
    dataset = manifest["dataset"]
    spec["dataset"]["test_data_sources"] = {
        "image_dir": dataset["validation_image_dir"],
        "json_file": dataset["validation_annotation"],
    }
    spec["dataset"]["batch_size"] = manifest["runtime"][
        "evaluation_batch_size_per_gpu"
    ]
    spec["evaluate"]["batch_size"] = manifest["runtime"][
        "evaluation_batch_size_per_gpu"
    ]
    spec["evaluate"]["num_gpus"] = 8
    spec["evaluate"]["gpu_ids"] = list(range(8))
    spec["evaluate"]["num_nodes"] = 1
    spec["evaluate"]["checkpoint"] = checkpoint
    spec["wandb"]["enable"] = False
    queries = int(spec["model"]["num_queries"])
    spec["model"]["num_select"] = min(int(spec["model"]["num_select"]), queries)
    return spec


def _local_lustre_path(uri: str) -> str:
    if uri.startswith("lustre://"):
        path = uri.removeprefix("lustre://")
        return path if path.startswith("/") else f"/{path}"
    if uri.startswith("/"):
        return uri
    raise CampaignExecutionError(f"expected Lustre results URI, got {uri!r}")


def _wait_for_job(
    sdk: Any,
    job_id: str,
    *,
    events: Path,
    phase: str,
    mode: str,
    candidate_id: str,
) -> str:
    previous = None
    while True:
        status = sdk.get_job_status(job_id).status
        if status != previous:
            append_jsonl(
                events,
                {
                    "event": "slurm_job_status",
                    "phase": phase,
                    "mode": mode,
                    "candidate_id": candidate_id,
                    "tao_job_id": job_id,
                    "status": status,
                },
            )
            previous = status
        if status in TERMINAL_JOB_STATUSES:
            return status
        time.sleep(10)


def _terminal_checkpoint(sdk: Any, job_id: str) -> str:
    root = _local_lustre_path(sdk.get_job_results_dir(job_id))
    output = remote_output(
        f"find {shlex.quote(root)} -type f "
        "\\( -name '*.pth' -o -name '*.ckpt' \\) "
        "-printf '%T@ %p\\n' | sort -nr | head -1"
    ).strip()
    if " " not in output:
        raise CampaignExecutionError(
            f"training job {job_id} produced no terminal checkpoint"
        )
    checkpoint = output.split(" ", 1)[1]
    if not checkpoint.startswith(root.rstrip("/") + "/"):
        raise CampaignExecutionError("terminal checkpoint escaped its result root")
    return checkpoint


def _map50_from_status(sdk: Any, job_id: str) -> float | None:
    root = _local_lustre_path(sdk.get_job_results_dir(job_id))
    status_path = f"{root}/results_dir/evaluate/status.json"
    output = remote_output(
        f"(test -f {shlex.quote(status_path)} && "
        f"tail -100 {shlex.quote(status_path)}) || true"
    )
    values = []
    for line in output.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        kpi = record.get("kpi")
        if isinstance(kpi, Mapping):
            value = kpi.get("test_mAP50", kpi.get("val_mAP50"))
            if value is not None:
                values.append(float(value))
    return values[-1] if values else None


def _launch_evaluation(
    sdk: Any,
    manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    events: Path,
    mode: str,
    candidate_id: str,
    existing_job: Mapping[str, Any] | None = None,
    on_submitted: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[float, dict[str, Any]]:
    from tao_sdk.script_runner import build_entrypoint

    action = yaml.safe_load(
        (
            Path(manifest["runtime"]["skill_dir"])
            / "references/skill_info.yaml"
        ).read_text(encoding="utf-8")
    )["actions"]["evaluate"]
    entrypoint = build_entrypoint(
        command=action["command"],
        specs=spec,
        inputs=action["inputs"],
        outputs=action["outputs"],
        config_format=action.get("config_format", "yaml"),
        upload_excludes=action.get("upload_excludes", []),
    )
    expected = {
        "spec_sha256": manifest_sha256(spec),
        "command_sha256": text_sha256(entrypoint["command"]),
    }
    runtime = manifest["runtime"]
    if existing_job is not None:
        for key, expected_value in expected.items():
            if existing_job.get(key) != expected_value:
                raise CampaignExecutionError(
                    f"persisted evaluation child {key} changed"
                )
        job_id = existing_job.get("tao_job_id")
        if not isinstance(job_id, str) or not job_id:
            raise CampaignExecutionError(
                "persisted evaluation child lacks a TAO job ID"
            )
        submitted = {**dict(existing_job), **expected}
    else:
        job = sdk.create_job(
            image=runtime["sqsh_path"],
            command=entrypoint["command"],
            gpu_count=8,
            num_nodes=1,
            partition=runtime["partition"],
            account=runtime["account"],
        )
        job_id = job.id
        submitted = {
            "status": "submitted",
            "submitted_at_utc": utc_timestamp(),
            "tao_job_id": job_id,
            **expected,
        }
    if on_submitted is not None:
        on_submitted(copy.deepcopy(submitted))
    status = _wait_for_job(
        sdk,
        job_id,
        events=events,
        phase="evaluation",
        mode=mode,
        candidate_id=candidate_id,
    )
    final_evidence = {
        **submitted,
        "status": status,
        "terminal_at_utc": utc_timestamp(),
    }
    if on_submitted is not None:
        on_submitted(copy.deepcopy(final_evidence))
    logs = sdk.get_job_logs(job_id, tail=5000)
    if status != "Complete":
        raise CampaignExecutionError(
            f"evaluation job {job_id} ended as {status}: {logs[-3000:]}"
        )
    metric = _map50_from_status(sdk, job_id)
    if metric is None:
        values = [
            float(value)
            for pattern in MAP50_PATTERNS
            for value in pattern.findall(logs)
        ]
        metric = values[-1] if values else None
    if metric is None or not 0.0 <= metric <= 1.0:
        raise CampaignExecutionError(
            f"evaluation job {job_id} emitted no valid mAP50"
        )
    return metric, {
        **final_evidence,
        "result_root": _local_lustre_path(sdk.get_job_results_dir(job_id)),
    }


def _install_payload(manifest: Mapping[str, Any]) -> str:
    source_root = Path(manifest["source"]["repository"]) / "src/tao_automl"
    files = {
        "tao_automl/__init__.py": b"",
        "tao_automl/latency_stats.py": (
            source_root / "latency_stats.py"
        ).read_bytes(),
        "tao_automl/latency_benchmark.py": (
            source_root / "latency_benchmark.py"
        ).read_bytes(),
        "dino_latency_worker.py": (
            HERE / "dino_latency_worker.py"
        ).read_bytes(),
        "contract.json": json.dumps(
            _latency_contract_document(manifest),
            sort_keys=True,
        ).encode("utf-8"),
        "input_descriptor.json": json.dumps(
            manifest["latency_protocol"]["input_descriptor"],
            sort_keys=True,
        ).encode("utf-8"),
    }
    encoded = {
        name: base64.b64encode(content).decode("ascii")
        for name, content in files.items()
    }
    script = (
        "import base64,json,pathlib;"
        "root=pathlib.Path('/tmp/dino_campaign_runtime');"
        f"files=json.loads({json.dumps(json.dumps(encoded))});"
        "[(root/name).parent.mkdir(parents=True,exist_ok=True) "
        "for name in files];"
        "[(root/name).write_bytes(base64.b64decode(data)) "
        "for name,data in files.items()]"
    )
    # The SDK runner resolves only ``{config_path}`` at container start with
    # ``str.format``.  The embedded JSON mapping also contains braces, so make
    # those braces literal for that later formatting pass.  Keep the actual
    # config placeholder outside this payload unescaped in ``_launch_latency``.
    command = f"python -c {shlex.quote(script)}"
    return command.replace("{", "{{").replace("}", "}}")


def _latency_contract_document(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = manifest["latency_protocol"]
    return {
        "schema_version": 1,
        "warmup_iterations": protocol["warmup_iterations"],
        "timed_iterations": protocol["timed_iterations"],
        "repeated_rounds": protocol["repeated_rounds"],
        "tail_percentile": protocol["tail_percentile"],
        "bootstrap_resamples": protocol["bootstrap_resamples"],
        "bootstrap_confidence_level": protocol[
            "bootstrap_confidence_level"
        ],
        "bootstrap_seed": protocol["bootstrap_seed"],
        "batch_size_per_replica": protocol["batch_size_per_replica"],
        "precision": protocol["precision"],
        "timed_scope": protocol["timed_scope"],
        "input_sha256": protocol["input_sha256"],
        "runtime_sha256": protocol["runtime_sha256"],
        "expected_replicas": protocol["expected_replicas"],
        "measurement_role": protocol["measurement_role"],
        "synchronization": protocol["synchronization"],
        "validity_thresholds": protocol["validity_thresholds"],
    }


def _launch_latency(
    sdk: Any,
    manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
    checkpoint: str,
    candidate_fingerprint: str,
    *,
    events: Path,
    mode: str,
    candidate_id: str,
    existing_job: Mapping[str, Any] | None = None,
    on_submitted: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    from tao_automl.latency_benchmark import combine_replica_records
    from tao_sdk.script_runner import build_entrypoint

    benchmark_spec = copy.deepcopy(dict(spec))
    benchmark_spec["dataset"]["batch_size"] = 1
    benchmark_spec["evaluate"]["batch_size"] = 1
    installer = _install_payload(manifest)
    command = " ".join(
        [
            installer,
            "&&",
            "torchrun",
            "--standalone",
            "--nproc_per_node=8",
            "/tmp/dino_campaign_runtime/dino_latency_worker.py",
            "--config",
            "{config_path}",
            "--checkpoint",
            shlex.quote(checkpoint),
            "--contract",
            "/tmp/dino_campaign_runtime/contract.json",
            "--input-descriptor",
            "/tmp/dino_campaign_runtime/input_descriptor.json",
            "--candidate-fingerprint",
            shlex.quote(candidate_fingerprint),
            "--runtime-modules-root",
            "/tmp/dino_campaign_runtime",
            "--output-root",
            '"$TAO_RESULTS_ROOT/$TAO_JOB_ID/latency"',
        ]
    )
    action = yaml.safe_load(
        (
            Path(manifest["runtime"]["skill_dir"])
            / "references/skill_info.yaml"
        ).read_text(encoding="utf-8")
    )["actions"]["evaluate"]
    entrypoint = build_entrypoint(
        command=command,
        specs=benchmark_spec,
        inputs=action["inputs"],
        outputs={},
        config_format="yaml",
        upload_excludes=action.get("upload_excludes", []),
    )
    expected = {
        "spec_sha256": manifest_sha256(benchmark_spec),
        "command_sha256": text_sha256(entrypoint["command"]),
        "candidate_fingerprint": candidate_fingerprint,
        "contract_sha256": manifest_sha256(
            _latency_contract_document(manifest)
        ),
    }
    runtime = manifest["runtime"]
    if existing_job is not None:
        for key, expected_value in expected.items():
            if existing_job.get(key) != expected_value:
                raise CampaignExecutionError(
                    f"persisted latency child {key} changed"
                )
        job_id = existing_job.get("tao_job_id")
        if not isinstance(job_id, str) or not job_id:
            raise CampaignExecutionError(
                "persisted latency child lacks a TAO job ID"
            )
        submitted = {**dict(existing_job), **expected}
    else:
        job = sdk.create_job(
            image=runtime["sqsh_path"],
            command=entrypoint["command"],
            gpu_count=8,
            num_nodes=1,
            partition=runtime["partition"],
            account=runtime["account"],
        )
        job_id = job.id
        submitted = {
            "status": "submitted",
            "submitted_at_utc": utc_timestamp(),
            "tao_job_id": job_id,
            **expected,
        }
    if on_submitted is not None:
        on_submitted(copy.deepcopy(submitted))
    status = _wait_for_job(
        sdk,
        job_id,
        events=events,
        phase="selection_time_latency",
        mode=mode,
        candidate_id=candidate_id,
    )
    final_evidence = {
        **submitted,
        "status": status,
        "terminal_at_utc": utc_timestamp(),
    }
    if on_submitted is not None:
        on_submitted(copy.deepcopy(final_evidence))
    logs = sdk.get_job_logs(job_id, tail=5000)
    if (
        status != "Complete"
        or "TAO_AUTOML_DINO_LATENCY_COMPLETE" not in logs
    ):
        raise CampaignExecutionError(
            f"latency job {job_id} ended as {status}: {logs[-3000:]}"
        )
    root = _local_lustre_path(sdk.get_job_results_dir(job_id))
    reader = (
        "import glob,json,sys;"
        "paths=sorted(glob.glob(sys.argv[1]+'/rank_*.json'));"
        "print(json.dumps([json.load(open(path)) for path in paths]))"
    )
    records = json.loads(
        remote_output(
            f"python3 -c {shlex.quote(reader)} "
            f"{shlex.quote(root + '/latency')}"
        )
    )
    if len(records) != manifest["latency_protocol"]["expected_replicas"]:
        raise CampaignExecutionError(
            f"latency job {job_id} produced {len(records)}/"
            f"{manifest['latency_protocol']['expected_replicas']} "
            "job-scoped replica records"
        )
    record_job_ids = {
        item.get("tao_job_id")
        for item in records
        if isinstance(item, Mapping)
    }
    if record_job_ids != {job_id}:
        raise CampaignExecutionError(
            f"latency replica records are not isolated to TAO job {job_id}"
        )
    input_hashes = set()
    rank_runtime_evidence = []
    for item in records:
        evidence = (
            item.get("input_evidence")
            if isinstance(item, Mapping)
            else None
        )
        if not isinstance(evidence, Mapping):
            raise CampaignExecutionError(
                "latency replica omitted its input evidence"
            )
        evidence_payload = dict(evidence)
        evidence_sha = evidence_payload.pop("sha256", None)
        if evidence_sha != manifest_sha256(evidence_payload):
            raise CampaignExecutionError(
                "latency replica input-evidence integrity failed"
            )
        input_hashes.add(evidence_sha)
        runtime_evidence = item.get("rank_runtime_evidence")
        if not isinstance(runtime_evidence, Mapping):
            raise CampaignExecutionError(
                "latency replica omitted hardware/runtime evidence"
            )
        for key, expected in manifest["runtime"][
            "hardware_contract"
        ].items():
            if runtime_evidence.get(key) != expected:
                raise CampaignExecutionError(
                    f"latency replica hardware changed: {key}"
                )
        rank_runtime_evidence.append(dict(runtime_evidence))
    if len(input_hashes) != 1 or None in input_hashes:
        raise CampaignExecutionError(
            "latency replicas did not use one identical preprocessed input set"
        )
    aggregate = combine_replica_records(records)
    aggregate_evidence = {
        "schema_version": 1,
        "aggregate": aggregate,
        "input_evidence_sha256": next(iter(input_hashes)),
        "rank_runtime_evidence": sorted(
            rank_runtime_evidence,
            key=lambda item: item["local_rank"],
        ),
    }
    aggregate_evidence["evidence_sha256"] = manifest_sha256(
        aggregate_evidence
    )
    statistics = aggregate["statistics"]
    if (
        not statistics["is_valid"]
        or statistics["raw_sample_count_total"] != 4000
    ):
        raise CampaignExecutionError(
            "selection-time latency failed its frozen quality gate"
        )
    low, high = statistics["bootstrap_median_ci_ms"]
    metrics = {
        "latency_ms": float(statistics["median_ms"]),
        "latency_p95_ms": float(statistics["p95_ms"]),
        "latency_ci95_low_ms": float(low),
        "latency_ci95_high_ms": float(high),
    }
    return metrics, {
        **final_evidence,
        "result_root": root,
        "aggregate_evidence": aggregate_evidence,
    }


def _metric_extractor(logs: str, metric_name: str) -> float | None:
    if metric_name != "mAP50":
        return None
    values = [
        float(value)
        for pattern in MAP50_PATTERNS
        for value in pattern.findall(logs)
    ]
    return values[-1] if values else None


def _run_mode(
    manifest_path: str,
    runtime_root: str,
    mode: str,
    resume: bool,
) -> None:
    manifest = load_manifest(manifest_path)
    mode_dir = Path(runtime_root) / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    events = mode_dir / "events.jsonl"
    candidates_path = mode_dir / "candidate_evidence.json"
    candidates: dict[str, Any] = {}
    if resume and candidates_path.is_file():
        candidates = json.loads(
            candidates_path.read_text(encoding="utf-8")
        )["candidates"]

    configure_slurm_runtime(manifest)
    from tao_automl.runner import AutoMLRunner
    from tao_sdk.platforms.slurm import SlurmSDK

    import tao_sdk

    sdk_source = Path(tao_sdk.__file__).resolve()
    if not sdk_source.is_relative_to(
        Path(manifest["runtime"]["sdk_dir"]).resolve()
    ):
        raise CampaignExecutionError(
            f"tao_sdk imported from unsealed source: {sdk_source}"
        )

    sdk = SlurmSDK(
        poll_interval=10,
        state_file=mode_dir / "slurm_state.json",
    )
    runner = AutoMLRunner(
        sdk=sdk,
        skill_dir=Path(manifest["runtime"]["skill_dir"]),
        action="train",
        poll_interval=10,
    )

    def persist() -> None:
        atomic_json(
            candidates_path,
            {
                "schema_version": 1,
                "manifest_sha256": manifest["manifest_sha256"],
                "mode": mode,
                "candidates": candidates,
            },
        )

    def on_recommendation(rec: Any) -> None:
        candidate_id = f"{mode}_rec_{rec.id}"
        record = candidates.setdefault(candidate_id, {})
        immutable = {
            "candidate_id": candidate_id,
            "rec_id": str(rec.id),
            "specs": copy.deepcopy(rec.specs),
            "recommendation_audit": copy.deepcopy(
                rec.recommendation_audit
            ),
            "agent_intervention_flags": {
                name: False for name in AGENT_FLAGS
            },
        }
        for key, value in immutable.items():
            if key in record and record[key] != value:
                raise CampaignExecutionError(
                    f"resumed recommendation changed {candidate_id} {key}"
                )
            record[key] = value
        record.setdefault("status", "recommended")
        persist()

    def evaluate_candidate(rec: Any, train_job_id: str) -> dict[str, float]:
        candidate_id = f"{mode}_rec_{rec.id}"
        existing = candidates.setdefault(candidate_id, {})
        cached_objectives = existing.get("objective_values")
        if (
            existing.get("status") == "success"
            and isinstance(cached_objectives, Mapping)
        ):
            return {
                str(key): float(value)
                for key, value in cached_objectives.items()
            }
        checkpoint = _terminal_checkpoint(sdk, train_job_id)
        specification = _evaluation_spec(
            manifest,
            rec.specs,
            checkpoint,
        )
        from tao_automl.selection import canonical_spec_fingerprint

        fingerprint = canonical_spec_fingerprint(rec.specs)
        audit_fingerprint = rec.recommendation_audit.get(
            "candidate_fingerprint"
        )
        if audit_fingerprint != fingerprint:
            raise CampaignExecutionError(
                f"{candidate_id} recommendation fingerprint changed"
            )
        checkpoint_identity_sha256 = manifest_sha256(
            {
                "train_job_id": train_job_id,
                "checkpoint_path": checkpoint,
            }
        )
        candidates[candidate_id].update(
            {
                "status": "evaluating",
                "train_job_id": train_job_id,
                "checkpoint": checkpoint,
                "candidate_fingerprint": fingerprint,
                "checkpoint_identity_sha256": checkpoint_identity_sha256,
            }
        )
        persist()
        cached_evaluation = candidates[candidate_id].get(
            "accuracy_evaluation"
        )
        map50, evaluation = _launch_evaluation(
            sdk,
            manifest,
            specification,
            events=events,
            mode=mode,
            candidate_id=candidate_id,
            existing_job=(
                cached_evaluation
                if isinstance(cached_evaluation, Mapping)
                else None
            ),
            on_submitted=lambda evidence: (
                candidates[candidate_id].update(
                    {"accuracy_evaluation": evidence}
                ),
                persist(),
            ),
        )
        candidates[candidate_id].update(
            {
                "mAP50": map50,
                "accuracy_evaluation": evaluation,
            }
        )
        persist()
        cached_latency = candidates[candidate_id].get(
            "selection_time_latency"
        )
        latency, latency_job = _launch_latency(
            sdk,
            manifest,
            specification,
            checkpoint,
            fingerprint,
            events=events,
            mode=mode,
            candidate_id=candidate_id,
            existing_job=(
                cached_latency
                if isinstance(cached_latency, Mapping)
                else None
            ),
            on_submitted=lambda evidence: (
                candidates[candidate_id].update(
                    {"selection_time_latency": evidence}
                ),
                persist(),
            ),
        )
        objectives = {"mAP50": map50, **latency}
        candidates[candidate_id].update(
            {
                "status": "success",
                "objective_values": objectives,
                "accuracy_evaluation": evaluation,
                "selection_time_latency": latency_job,
                "measurement_role": "selection_time",
                "selection_time_measurements_feed_selection": True,
                "matched_validation_selection_isolation_flags": {
                    name: False for name in SELECTION_FLAGS
                },
            }
        )
        persist()
        return objectives

    def on_result(rec: Any, metric: Any, status: str) -> None:
        candidate_id = f"{mode}_rec_{rec.id}"
        record = candidates.setdefault(candidate_id, {})
        record["automl_status"] = status
        record["reported_metric"] = metric
        recommendation_job_id = getattr(rec, "job_id", None)
        if recommendation_job_id:
            record["train_job_id"] = recommendation_job_id
        if str(status).lower() not in SUCCESS_RECOMMENDATION_STATUSES:
            record["status"] = "terminal_failure"
            record["failure_reason"] = getattr(rec, "failure_reason", None)
        persist()

    result = runner.run(
        train_dataset_uri=manifest["dataset"]["slurm_root"],
        eval_dataset_uri=manifest["dataset"]["slurm_root"],
        base_checkpoint=manifest["ptms"][0]["slurm_path"],
        base_checkpoint_target=manifest["ptms"][0]["checkpoint_target"],
        workspace_id=f"{manifest['campaign_id']}-{mode}",
        image=manifest["runtime"]["sqsh_path"],
        automl_settings=mode_settings(manifest, mode),
        automl_hyperparameters=list(SEARCH_PARAMETERS),
        custom_param_ranges=custom_ranges(manifest),
        workspace_path=str(mode_dir / "workspace"),
        spec_overrides=spec_overrides(manifest),
        metric_extractor=_metric_extractor,
        eval_fn=evaluate_candidate,
        on_recommendation=on_recommendation,
        on_result=on_result,
        resume=resume,
        gpu_count=8,
        num_nodes=1,
        partition=manifest["runtime"]["partition"],
        account=manifest["runtime"]["account"],
    )
    atomic_json(
        mode_dir / "result.json",
        {
            "schema_version": 1,
            "manifest_sha256": manifest["manifest_sha256"],
            "mode": mode,
            "status": "success",
            "result": result,
        },
    )


def launch_all_modes(
    manifest_path: Path,
    runtime_root: Path,
    *,
    resume: bool,
) -> dict[str, int | None]:
    context = mp.get_context("spawn")
    processes = {
        mode: context.Process(
            target=_run_mode,
            args=(str(manifest_path), str(runtime_root), mode, resume),
            name=f"dino-direct-{mode}",
        )
        for mode in MODES
    }
    for process in processes.values():
        process.start()

    def forward(signum: int, _frame: object) -> None:
        for process in processes.values():
            if process.is_alive() and process.pid:
                os.kill(process.pid, signum)

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    exit_codes: dict[str, int | None] = {}
    while processes:
        for mode, process in list(processes.items()):
            process.join(timeout=1)
            if not process.is_alive():
                exit_codes[mode] = process.exitcode
                processes.pop(mode)
        if processes:
            time.sleep(1)
    atomic_json(runtime_root / "mode_process_status.json", exit_codes)
    return exit_codes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--acknowledge-direct-full-dataset",
        action="store_true",
        help="Required with --launch; this is not a smoke or CPU run.",
    )
    parser.add_argument("--env-file", type=Path, default=ENV_PATH)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    plan = launch_plan(manifest)
    runtime_root = args.runtime_root.resolve()
    if not args.launch:
        runtime_root.mkdir(parents=True, exist_ok=True)
        atomic_json(runtime_root / "launch_plan.json", plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if not args.acknowledge_direct_full_dataset:
        raise CampaignExecutionError(
            "--launch requires --acknowledge-direct-full-dataset"
        )
    runtime_exists = runtime_root.exists()
    if args.resume and not runtime_root.is_dir():
        raise CampaignExecutionError("resume runtime root does not exist")
    if not args.resume and runtime_exists:
        raise CampaignExecutionError(
            "fresh launch refuses a pre-existing runtime root; use --resume "
            "only for the same sealed campaign"
        )
    if not args.resume:
        runtime_root.mkdir(parents=True)
    atomic_json(runtime_root / "launch_plan.json", plan)
    local = verify_local_launch_contract(manifest)
    loaded_names = load_env_file(args.env_file)
    configure_slurm_runtime(manifest)
    remote = verify_remote_contract(manifest)
    atomic_json(
        runtime_root / "submission_provenance.json",
        {
            "schema_version": 1,
            "manifest_sha256": manifest["manifest_sha256"],
            "loaded_secret_keys": list(loaded_names),
            "secret_values_recorded": False,
            "local_contract": local,
            "remote_contract": remote,
            "slurm_use_sqsh": False,
            "slurm_use_requeue": True,
            "slurm_retry_cap": manifest["runtime"]["max_job_retries"],
            "slurm_time_hours": manifest["runtime"]["time_hours"],
            "slurm_timeout_hours": manifest["runtime"]["timeout_hours"],
            "cpu_runs": 0,
            "smoke_runs": 0,
        },
    )
    exit_codes = launch_all_modes(
        manifest_path,
        runtime_root,
        resume=args.resume,
    )
    if set(exit_codes) != set(MODES) or any(
        value != 0 for value in exit_codes.values()
    ):
        raise CampaignExecutionError(
            f"one or more mode controllers failed: {exit_codes}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
