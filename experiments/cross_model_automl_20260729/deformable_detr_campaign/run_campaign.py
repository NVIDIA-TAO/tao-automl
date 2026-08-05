#!/usr/bin/env python3

"""Launch two parallel full-VOC2007 Deformable DETR qualifications.

The only executable path requires both ``--launch`` and
``--acknowledge-direct-full-dataset``.  Each logical workflow trains one
official PTM for ten full epochs on one eight-A100 node, validates every epoch,
then submits a standalone evaluation using the terminal checkpoint.  There is
no CPU, smoke, mini-step, local-model, replacement, or fallback path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from tao_automl.ptm_registry import load_ptm_registry

try:
    from .manifest_generator import (
        AGENT_FLAGS,
        DEFAULT_OUTPUT,
        canonical_sha,
        load_manifest,
        sha256_file,
    )
except ImportError:  # pragma: no cover - direct script execution
    from manifest_generator import (  # type: ignore[no-redef]
        AGENT_FLAGS,
        DEFAULT_OUTPUT,
        canonical_sha,
        load_manifest,
        sha256_file,
    )


HERE = Path(__file__).resolve().parent
ENV_PATH = Path("/localhome/local-rarunachalam/.tao/config.env")
DEFAULT_RUNTIME_ROOT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/deformable_detr_qualification"
)
TERMINAL_JOB_STATUSES = frozenset({"Complete", "Error", "Canceled"})
SAFE_ENV_KEYS = frozenset({"SLURM_HOSTNAME", "SLURM_USER", "SSH_KEY_PATH"})
SENSITIVE_ENV_KEYS = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "CLOUD_METADATA",
        "HF_TOKEN",
        "NGC_KEY",
        "S3_BUCKET_NAME",
        "S3_ENDPOINT_URL",
        "SECRET_KEY",
        "ACCESS_KEY",
    }
)


class CampaignExecutionError(RuntimeError):
    """The sealed direct qualification cannot safely continue."""


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(dict(value), sort_keys=True, allow_nan=False) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def load_launch_environment(path: Path = ENV_PATH) -> tuple[str, ...]:
    """Load only SSH/SLURM routing keys; PTMs and data are already on Lustre."""
    if not path.is_file():
        raise CampaignExecutionError(f"launch environment is unavailable: {path}")
    loaded: list[str] = []
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
        if name not in SAFE_ENV_KEYS:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[name] = value
        loaded.append(name)
    missing = sorted(
        name for name in ("SLURM_HOSTNAME", "SLURM_USER")
        if not os.environ.get(name)
    )
    if missing:
        raise CampaignExecutionError(
            "launch environment lacks required keys: " + ", ".join(missing)
        )
    for name in SENSITIVE_ENV_KEYS:
        os.environ.pop(name, None)
    return tuple(sorted(set(loaded)))


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
    if os.environ.get("SSH_KEY_PATH"):
        result.extend(["-i", os.environ["SSH_KEY_PATH"]])
    return [*result, f"{user}@{host}", command]


def remote_output(command: str, *, timeout: int = 900) -> str:
    return subprocess.run(
        _ssh_command(command),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    ).stdout


def verify_local_launch_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    source = manifest["source"]
    runtime = manifest["runtime"]
    repository = Path(source["repository"])
    head = _git(repository, "rev-parse", "HEAD")
    if subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "merge-base",
            "--is-ancestor",
            source["commit"],
            head,
        ],
        check=False,
        capture_output=True,
        timeout=30,
    ).returncode != 0:
        raise CampaignExecutionError("sealed source is not an ancestor of launch HEAD")
    if _git(repository, "status", "--porcelain"):
        raise CampaignExecutionError("launch repository must be clean")
    integrity = manifest["integrity"]
    required = {
        HERE / "campaign.inputs.v1.json": integrity["inputs_sha256"],
        HERE / "manifest_generator.py": integrity["manifest_generator_sha256"],
        HERE / "run_campaign.py": integrity["launcher_sha256"],
    }
    required.update(
        {
            Path(item["checkpoint_spec"]["path"]): item[
                "checkpoint_spec"
            ]["sha256"]
            for item in manifest["ptms"]
        }
    )
    for path, expected in required.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise CampaignExecutionError(f"sealed launcher input changed: {path}")
    sources = {}
    for label, path_key, revision_key in (
        ("skills", "skill_dir", "skill_revision"),
        ("sdk", "sdk_dir", "sdk_revision"),
    ):
        path = Path(runtime[path_key])
        revision = _git(path, "rev-parse", "HEAD")
        if revision != runtime[revision_key] or _git(path, "status", "--porcelain"):
            raise CampaignExecutionError(f"sealed {label} checkout changed")
        sources[label] = {"path": str(path), "commit": revision, "clean": True}
    registry_path = (
        repository / "src" / "tao_automl" / "data" / "ptm_registry.v1.json"
    )
    registry = load_ptm_registry(registry_path)
    if registry.document_sha256 != integrity["ptm_registry_sha256"]:
        raise CampaignExecutionError(
            "sealed PTM registry document identity changed"
        )
    sources["ptm_registry"] = {
        "path": str(registry_path),
        "document_sha256": registry.document_sha256,
        "file_sha256": sha256_file(registry_path),
    }
    return {
        "source_commit": source["commit"],
        "launch_head": head,
        "source_is_ancestor": True,
        "required_files": [
            {"path": str(path), "sha256": digest}
            for path, digest in required.items()
        ],
        "sources": sources,
    }


def configure_slurm_runtime(manifest: Mapping[str, Any]) -> None:
    runtime = manifest["runtime"]
    sdk_dir = runtime["sdk_dir"]
    sys.path = [sdk_dir, *[item for item in sys.path if item != sdk_dir]]
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [
            sdk_dir,
            *[
                item for item in existing.split(os.pathsep)
                if item and item != sdk_dir
            ],
        ]
    )
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
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )


def verify_remote_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
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
            ptm["artifact"]["slurm_path"],
            ptm["artifact"]["sha256"],
            ptm["artifact"]["size_bytes"],
        )
    verified: dict[str, Any] = {}
    for label, (path, expected_sha, expected_size) in files.items():
        lines = remote_output(
            " ".join(
                [
                    "test -f",
                    shlex.quote(path),
                    "&& stat -c '%s'",
                    shlex.quote(path),
                    "&& sha256sum",
                    shlex.quote(path),
                ]
            ),
            timeout=7200 if label == "sqsh" else 1800,
        ).strip().splitlines()
        if len(lines) != 2:
            raise CampaignExecutionError(f"remote probe incomplete for {label}")
        size = int(lines[0])
        digest = lines[1].split()[0]
        if size != expected_size or digest != expected_sha:
            raise CampaignExecutionError(f"remote identity mismatch for {label}")
        verified[label] = {
            "path": path,
            "sha256": digest,
            "size_bytes": size,
        }

    tree_script = (
        "import hashlib,json,pathlib,sys;"
        "root=pathlib.Path(sys.argv[1]);"
        "items=sorted((p for p in root.iterdir() if p.is_file()),key=lambda p:p.name);"
        "outer=hashlib.sha256();total=0;"
        "exec(\"for p in items:\\n"
        " h=hashlib.sha256(p.read_bytes()).hexdigest()\\n"
        " outer.update(f'{h}  {p.name}\\\\n'.encode())\\n"
        " total+=p.stat().st_size\");"
        "print(json.dumps({'algorithm':'sha256_of_sorted_sha256sum_basename_lines',"
        "'sha256':outer.hexdigest(),'file_count':len(items),'total_bytes':total},"
        "sort_keys=True))"
    )
    image_tree = json.loads(
        remote_output(
            f"python3 -c {shlex.quote(tree_script)} "
            f"{shlex.quote(dataset['train_image_dir'])}",
            timeout=3600,
        ).strip()
    )
    if image_tree != dataset["image_tree"]:
        raise CampaignExecutionError("remote VOC2007 image-tree identity mismatch")
    verified["voc_images"] = {
        "path": dataset["train_image_dir"],
        **image_tree,
    }
    return verified


def _merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _ptm_by_workflow(
    manifest: Mapping[str, Any], workflow_id: str
) -> Mapping[str, Any]:
    matches = [item for item in manifest["ptms"] if item["workflow_id"] == workflow_id]
    if len(matches) != 1:
        raise CampaignExecutionError(f"no unique PTM for {workflow_id}")
    return matches[0]


def build_train_spec(
    manifest: Mapping[str, Any], workflow_id: str
) -> dict[str, Any]:
    ptm = _ptm_by_workflow(manifest, workflow_id)
    template = (
        Path(manifest["runtime"]["skill_dir"])
        / "references/spec_template_train.yaml"
    )
    spec = yaml.safe_load(template.read_text(encoding="utf-8"))
    checkpoint_spec = yaml.safe_load(
        Path(ptm["checkpoint_spec"]["path"]).read_text(encoding="utf-8")
    )
    spec = _merge(spec, checkpoint_spec)
    spec = _merge(spec, ptm["default_spec_overrides"])
    dataset = manifest["dataset"]
    qualification = manifest["qualification"]
    spec = _merge(
        spec,
        {
            "results_dir": "",
            "wandb": {"enable": False},
            "dataset": {
                "train_data_sources": [
                    {
                        "image_dir": dataset["train_image_dir"],
                        "json_file": dataset["train_annotation"],
                    }
                ],
                "val_data_sources": [
                    {
                        "image_dir": dataset["validation_image_dir"],
                        "json_file": dataset["validation_annotation"],
                    }
                ],
                "batch_size": qualification["train_batch_size_per_gpu"],
                "num_classes": dataset["num_classes"],
                "eval_class_ids": dataset["eval_class_ids"],
            },
            "train": {
                "num_gpus": 8,
                "gpu_ids": list(range(8)),
                "num_nodes": 1,
                "seed": qualification["seed"],
                "num_epochs": qualification["training_epochs"],
                "checkpoint_interval": qualification["checkpoint_interval"],
                "checkpoint_interval_unit": "epoch",
                "validation_interval": qualification["validation_interval"],
                "pretrained_model_path": ptm["artifact"]["slurm_path"],
                "resume_training_checkpoint_path": "",
                "results_dir": "",
                "is_dry_run": False,
                "precision": qualification["precision"],
                "distributed_strategy": "ddp",
                "cudnn": {"benchmark": False, "deterministic": True},
                "optim": {"lr_steps": [8], "lr_step_size": 8},
            },
        },
    )
    spec["model"]["num_select"] = min(
        int(spec["model"]["num_select"]),
        int(spec["model"]["num_queries"]),
    )
    return spec


def build_evaluation_spec(
    manifest: Mapping[str, Any],
    workflow_id: str,
    checkpoint: str,
) -> dict[str, Any]:
    ptm = _ptm_by_workflow(manifest, workflow_id)
    template = (
        Path(manifest["runtime"]["skill_dir"])
        / "references/spec_template_evaluate.yaml"
    )
    spec = yaml.safe_load(template.read_text(encoding="utf-8"))
    checkpoint_spec = yaml.safe_load(
        Path(ptm["checkpoint_spec"]["path"]).read_text(encoding="utf-8")
    )
    spec = _merge(spec, checkpoint_spec)
    spec = _merge(spec, ptm["default_spec_overrides"])
    dataset = manifest["dataset"]
    batch_size = manifest["qualification"]["evaluation_batch_size_per_gpu"]
    spec = _merge(
        spec,
        {
            "results_dir": "",
            "wandb": {"enable": False},
            "dataset": {
                "test_data_sources": {
                    "image_dir": dataset["validation_image_dir"],
                    "json_file": dataset["validation_annotation"],
                },
                "batch_size": batch_size,
                "num_classes": dataset["num_classes"],
                "eval_class_ids": dataset["eval_class_ids"],
            },
            "evaluate": {
                "num_gpus": 8,
                "gpu_ids": list(range(8)),
                "num_nodes": 1,
                "checkpoint": checkpoint,
                "trt_engine": "",
                "results_dir": "",
                "batch_size": batch_size,
            },
        },
    )
    spec["model"]["num_select"] = min(
        int(spec["model"]["num_select"]),
        int(spec["model"]["num_queries"]),
    )
    return spec


def _action(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    metadata = yaml.safe_load(
        (
            Path(manifest["runtime"]["skill_dir"])
            / "references/skill_info.yaml"
        ).read_text(encoding="utf-8")
    )
    return metadata["actions"][name]


def _gpu_guard(command: str) -> str:
    """Fail closed inside the allocation unless all eight GPUs are A100-80GB."""
    return " ".join(
        [
            "set -eu;",
            "gpu_names=\"$(nvidia-smi --query-gpu=name --format=csv,noheader)\";",
            "gpu_caps=\"$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)\";",
            "gpu_mem=\"$(nvidia-smi --query-gpu=memory.total "
            "--format=csv,noheader,nounits)\";",
            "test \"$(printf '%s\\n' \"$gpu_names\" | sed '/^$/d' | wc -l)\" -eq 8;",
            "test \"$(printf '%s\\n' \"$gpu_names\" | sort -u)\" = "
            "'NVIDIA A100-SXM4-80GB';",
            "test \"$(printf '%s\\n' \"$gpu_caps\" | sort -u)\" = '8.0';",
            "test \"$(printf '%s\\n' \"$gpu_mem\" | sort -u)\" = '81920';",
            command,
        ]
    )


def _job_entrypoint(
    manifest: Mapping[str, Any],
    action_name: str,
    spec: Mapping[str, Any],
) -> tuple[str, str]:
    from tao_sdk.script_runner import build_entrypoint

    action = _action(manifest, action_name)
    entrypoint = build_entrypoint(
        command=_gpu_guard(action["command"]),
        specs=spec,
        inputs=action["inputs"],
        outputs=action["outputs"],
        config_format=action["config_format"],
        upload_excludes=action.get("upload_excludes", []),
    )
    return entrypoint["command"], hashlib.sha256(
        entrypoint["command"].encode("utf-8")
    ).hexdigest()


def _wait_for_job(
    sdk: Any,
    job_id: str,
    *,
    events: Path,
    workflow_id: str,
    phase: str,
) -> str:
    previous = None
    while True:
        status = sdk.get_job_status(job_id).status
        if status != previous:
            append_jsonl(
                events,
                {
                    "at_utc": utc_timestamp(),
                    "event": "job_status",
                    "workflow_id": workflow_id,
                    "phase": phase,
                    "tao_job_id": job_id,
                    "status": status,
                },
            )
            previous = status
        if status in TERMINAL_JOB_STATUSES:
            return status
        time.sleep(10)


def _local_lustre_path(uri: str) -> str:
    if uri.startswith("lustre://"):
        path = uri.removeprefix("lustre://")
        return path if path.startswith("/") else f"/{path}"
    if uri.startswith("/"):
        return uri
    raise CampaignExecutionError(f"expected Lustre result URI, got {uri!r}")


def _terminal_checkpoint(
    sdk: Any,
    job_id: str,
    *,
    training_epochs: int,
) -> dict[str, Any]:
    root = _local_lustre_path(sdk.get_job_results_dir(job_id))
    terminal_epoch_index = training_epochs - 1
    pattern = f"model_epoch_{terminal_epoch_index:03d}_step_*.pth"
    paths = [
        line
        for line in remote_output(
            f"find {shlex.quote(root)} -type f "
            f"-name {shlex.quote(pattern)} -print"
        ).splitlines()
        if line
    ]
    if len(paths) != 1:
        raise CampaignExecutionError(
            f"training job {job_id} emitted {len(paths)} exact "
            f"{pattern!r} terminal checkpoints"
        )
    checkpoint = paths[0]
    if not checkpoint.startswith(root.rstrip("/") + "/"):
        raise CampaignExecutionError("checkpoint escaped its result root")
    identity = remote_output(
        " ".join(
            [
                "stat -c '%s'",
                shlex.quote(checkpoint),
                "&& sha256sum",
                shlex.quote(checkpoint),
            ]
        ),
        timeout=1800,
    ).strip().splitlines()
    if len(identity) != 2:
        raise CampaignExecutionError(
            "terminal checkpoint identity probe was incomplete"
        )
    size = int(identity[0])
    digest = identity[1].split()[0]
    if size <= 0 or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise CampaignExecutionError(
            "terminal checkpoint identity probe was invalid"
        )
    return {
        "path": checkpoint,
        "sha256": digest,
        "size_bytes": size,
        "training_epochs": training_epochs,
        "terminal_epoch_index": terminal_epoch_index,
    }


def _status_records(
    sdk: Any,
    job_id: str,
    *,
    action: str,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Read one exact TAO action status file and preserve its identity."""
    root = _local_lustre_path(sdk.get_job_results_dir(job_id))
    path = f"{root.rstrip('/')}/results_dir/{action}/status.json"
    text = remote_output(
        f"test -f {shlex.quote(path)} && cat {shlex.quote(path)}"
    )
    identity = remote_output(
        " ".join(
            [
                "stat -c '%s'",
                shlex.quote(path),
                "&& sha256sum",
                shlex.quote(path),
            ]
        )
    ).strip().splitlines()
    if len(identity) != 2:
        raise CampaignExecutionError(
            f"{action} status identity probe was incomplete"
        )
    try:
        size = int(identity[0])
    except ValueError as exc:
        raise CampaignExecutionError(
            f"{action} status identity size was invalid"
        ) from exc
    digest = identity[1].split()[0]
    if size <= 0 or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise CampaignExecutionError(
            f"{action} status identity was invalid"
        )

    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CampaignExecutionError(
                f"{action} status line {line_number} is not valid JSON"
            ) from exc
        if not isinstance(record, Mapping):
            raise CampaignExecutionError(
                f"{action} status line {line_number} is not a JSON object"
            )
        records.append(record)
    if not records:
        raise CampaignExecutionError(f"{action} status file is empty")
    return records, {
        "path": path,
        "sha256": digest,
        "size_bytes": size,
        "record_count": len(records),
    }


def _metric(
    kpi: Mapping[str, Any],
    name: str,
    *,
    context: str,
) -> float:
    try:
        value = float(kpi[name])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise CampaignExecutionError(
            f"{context} did not emit finite {name}"
        ) from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise CampaignExecutionError(
            f"{context} emitted invalid {name}"
        )
    return value


def _training_status_evidence(
    sdk: Any,
    job_id: str,
    *,
    expected_validation_records: int,
) -> dict[str, Any]:
    records, identity = _status_records(
        sdk,
        job_id,
        action="train",
    )
    validation_metrics: list[dict[str, float]] = []
    for record in records:
        if record.get("message") != "Eval metrics generated.":
            continue
        kpi = record.get("kpi")
        if not isinstance(kpi, Mapping):
            raise CampaignExecutionError(
                "in-training validation record lacks a KPI object"
            )
        validation_metrics.append(
            {
                "mAP": _metric(
                    kpi,
                    "val_mAP",
                    context="in-training validation",
                ),
                "mAP50": _metric(
                    kpi,
                    "val_mAP50",
                    context="in-training validation",
                ),
            }
        )
    if len(validation_metrics) != expected_validation_records:
        raise CampaignExecutionError(
            f"training job {job_id} emitted {len(validation_metrics)} "
            "in-training validation records; expected "
            f"{expected_validation_records}"
        )
    if not any(
        record.get("message") == "Train finished successfully."
        for record in records
    ):
        raise CampaignExecutionError(
            f"training job {job_id} lacks terminal TAO train success"
        )
    return {
        **identity,
        "validation_record_count": len(validation_metrics),
        "validation_metrics": validation_metrics,
        "terminal_success_message": "Train finished successfully.",
        "terminal_success": True,
    }


def _evaluation_status_evidence(
    sdk: Any,
    job_id: str,
) -> dict[str, Any]:
    records, identity = _status_records(
        sdk,
        job_id,
        action="evaluate",
    )
    metric_records = [
        record
        for record in records
        if record.get("message") == "Test metrics generated."
    ]
    if len(metric_records) != 1:
        raise CampaignExecutionError(
            f"standalone evaluation job {job_id} emitted "
            f"{len(metric_records)} exact test metric records; expected 1"
        )
    kpi = metric_records[0].get("kpi")
    if not isinstance(kpi, Mapping):
        raise CampaignExecutionError(
            "standalone evaluation metric record lacks a KPI object"
        )
    metrics = {
        "mAP": _metric(kpi, "test_mAP", context="standalone evaluation"),
        "mAP50": _metric(
            kpi,
            "test_mAP50",
            context="standalone evaluation",
        ),
    }
    if not any(
        record.get("message") == "Evaluate finished successfully."
        for record in records
    ):
        raise CampaignExecutionError(
            f"standalone evaluation job {job_id} lacks terminal TAO "
            "evaluation success"
        )
    return {
        "metrics": metrics,
        "status_evidence": {
            **identity,
            "test_metric_record_count": 1,
            "metrics": metrics,
            "terminal_success_message": "Evaluate finished successfully.",
            "terminal_success": True,
        },
    }


def _evaluation_metrics(sdk: Any, job_id: str) -> dict[str, float]:
    """Compatibility wrapper returning metrics from exact evaluation evidence."""
    return _evaluation_status_evidence(sdk, job_id)["metrics"]


def _initial_workflow(
    manifest: Mapping[str, Any], workflow_id: str
) -> dict[str, Any]:
    ptm = _ptm_by_workflow(manifest, workflow_id)
    return {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "workflow_id": workflow_id,
        "ptm_id": ptm["id"],
        "ptm_sha256": ptm["artifact"]["sha256"],
        "ptm_path": ptm["artifact"]["slurm_path"],
        "status": "initialized",
        "terminal": False,
        "agent_intervention_flags": {name: False for name in AGENT_FLAGS},
        "jobs": {},
    }


def _submit_job(
    sdk: Any,
    manifest: Mapping[str, Any],
    command: str,
) -> Any:
    runtime = manifest["runtime"]
    return sdk.create_job(
        image=runtime["sqsh_path"],
        command=command,
        gpu_count=8,
        num_nodes=1,
        partition=runtime["partition"],
        account=runtime["account"],
    )


def _run_workflow(
    manifest_path: str,
    runtime_root: str,
    workflow_id: str,
) -> None:
    manifest = load_manifest(manifest_path)
    workflow_dir = Path(runtime_root) / workflow_id
    workflow_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = workflow_dir / "workflow_completion.json"
    events = workflow_dir / "events.jsonl"
    evidence = _initial_workflow(manifest, workflow_id)
    atomic_json(evidence_path, evidence)
    configure_slurm_runtime(manifest)

    try:
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
            state_file=workflow_dir / "slurm_state.json",
        )

        train_spec = build_train_spec(manifest, workflow_id)
        train_command, train_command_sha = _job_entrypoint(
            manifest, "train", train_spec
        )
        train_job = _submit_job(sdk, manifest, train_command)
        evidence["status"] = "training"
        evidence["jobs"]["train"] = {
            "tao_job_id": train_job.id,
            "status": "submitted",
            "submitted_at_utc": utc_timestamp(),
            "spec_sha256": canonical_sha(train_spec),
            "command_sha256": train_command_sha,
            "full_dataset": True,
            "training_epochs": 10,
            "validation_interval": 1,
            "nodes": 1,
            "gpus": 8,
        }
        atomic_json(evidence_path, evidence)
        train_status = _wait_for_job(
            sdk,
            train_job.id,
            events=events,
            workflow_id=workflow_id,
            phase="train",
        )
        evidence["jobs"]["train"]["status"] = train_status
        evidence["jobs"]["train"]["terminal_at_utc"] = utc_timestamp()
        evidence["jobs"]["train"]["result_root"] = _local_lustre_path(
            sdk.get_job_results_dir(train_job.id)
        )
        atomic_json(evidence_path, evidence)
        if train_status != "Complete":
            evidence["jobs"]["train"]["failure_analysis"] = (
                sdk.get_failure_analysis(train_job.id)
            )
            raise CampaignExecutionError(
                f"training job ended with terminal status {train_status}"
            )

        evidence["jobs"]["train"]["status_evidence"] = (
            _training_status_evidence(
                sdk,
                train_job.id,
                expected_validation_records=manifest["qualification"][
                    "training_epochs"
                ],
            )
        )
        checkpoint = _terminal_checkpoint(
            sdk,
            train_job.id,
            training_epochs=manifest["qualification"]["training_epochs"],
        )
        evidence["jobs"]["train"]["terminal_checkpoint"] = checkpoint
        evaluation_spec = build_evaluation_spec(
            manifest, workflow_id, checkpoint["path"]
        )
        evaluation_command, evaluation_command_sha = _job_entrypoint(
            manifest, "evaluate", evaluation_spec
        )
        evaluation_job = _submit_job(sdk, manifest, evaluation_command)
        evidence["status"] = "standalone_evaluation"
        evidence["jobs"]["evaluation"] = {
            "tao_job_id": evaluation_job.id,
            "status": "submitted",
            "submitted_at_utc": utc_timestamp(),
            "spec_sha256": canonical_sha(evaluation_spec),
            "command_sha256": evaluation_command_sha,
            "full_validation_split": True,
            "nodes": 1,
            "gpus": 8,
            "checkpoint": checkpoint["path"],
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_size_bytes": checkpoint["size_bytes"],
        }
        atomic_json(evidence_path, evidence)
        evaluation_status = _wait_for_job(
            sdk,
            evaluation_job.id,
            events=events,
            workflow_id=workflow_id,
            phase="standalone_evaluation",
        )
        evidence["jobs"]["evaluation"]["status"] = evaluation_status
        evidence["jobs"]["evaluation"]["terminal_at_utc"] = utc_timestamp()
        evidence["jobs"]["evaluation"]["result_root"] = _local_lustre_path(
            sdk.get_job_results_dir(evaluation_job.id)
        )
        if evaluation_status != "Complete":
            evidence["jobs"]["evaluation"]["failure_analysis"] = (
                sdk.get_failure_analysis(evaluation_job.id)
            )
            raise CampaignExecutionError(
                "standalone evaluation job ended with terminal status "
                f"{evaluation_status}"
            )
        evaluation_evidence = _evaluation_status_evidence(
            sdk,
            evaluation_job.id,
        )
        evidence["jobs"]["evaluation"]["status_evidence"] = (
            evaluation_evidence["status_evidence"]
        )
        evidence["metrics"] = evaluation_evidence["metrics"]
        evidence["status"] = "success"
        evidence["terminal"] = True
        evidence["terminal_at_utc"] = utc_timestamp()
        evidence["failure_preserved"] = False
        atomic_json(evidence_path, evidence)
    except BaseException as exc:
        evidence["status"] = "terminal_failure"
        evidence["terminal"] = True
        evidence["terminal_at_utc"] = utc_timestamp()
        evidence["failure_preserved"] = True
        evidence["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "replacement_submitted": False,
        }
        atomic_json(evidence_path, evidence)
        raise


def _workflow_process(
    manifest_path: str,
    runtime_root: str,
    workflow_id: str,
) -> None:
    try:
        _run_workflow(manifest_path, runtime_root, workflow_id)
    except BaseException:
        # The workflow artifact already preserves the terminal failure.  The
        # non-zero child exit lets the parent classify the aggregate result.
        raise


def build_completion(
    manifest: Mapping[str, Any],
    runtime_root: Path,
    workflow_ids: tuple[str, ...],
    exit_codes: Mapping[str, int | None],
) -> dict[str, Any]:
    workflows = []
    for workflow_id in workflow_ids:
        path = runtime_root / workflow_id / "workflow_completion.json"
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
        else:
            record = {
                **_initial_workflow(manifest, workflow_id),
                "status": "terminal_failure",
                "terminal": True,
                "failure_preserved": True,
                "failure": {
                    "type": "MissingWorkflowArtifact",
                    "message": "worker exited without a terminal artifact",
                    "replacement_submitted": False,
                },
            }
            atomic_json(path, record)
        if record.get("status") not in {"success", "terminal_failure"}:
            record["status"] = "terminal_failure"
            record["terminal"] = True
            record["failure_preserved"] = True
            record["failure"] = {
                "type": "IncompleteWorkflow",
                "message": "worker exited before recording a terminal outcome",
                "replacement_submitted": False,
            }
            atomic_json(path, record)
        record["process_exit_code"] = exit_codes.get(workflow_id)
        workflows.append(record)
    success_count = sum(item["status"] == "success" for item in workflows)
    payload = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "model": "deformable_detr",
        "manifest_sha256": manifest["manifest_sha256"],
        "terminal": True,
        "status": "success" if success_count == 2 else "terminal_with_failures",
        "terminal_at_utc": utc_timestamp(),
        "logical_workflows_submitted": 2,
        "successful_workflows": success_count,
        "failed_workflows": 2 - success_count,
        "workflows_started_in_parallel": True,
        "cpu_runs": 0,
        "smoke_runs": 0,
        "ministep_runs": 0,
        "local_model_runs": 0,
        "failures_preserved": True,
        "replacement_workflows_submitted": False,
        "outcomes": {
            item["workflow_id"]: item["status"] for item in workflows
        },
        "workflows": workflows,
    }
    payload["completion_sha256"] = canonical_sha(payload)
    return payload


def validate_completion(
    value: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    completion = copy.deepcopy(dict(value))
    expected = completion.pop("completion_sha256", None)
    if expected != canonical_sha(completion):
        raise CampaignExecutionError("completion artifact integrity failed")
    if (
        completion.get("schema_version") != 1
        or completion.get("campaign_id") != manifest["campaign_id"]
        or completion.get("model") != "deformable_detr"
        or completion.get("manifest_sha256") != manifest["manifest_sha256"]
        or completion.get("terminal") is not True
        or completion.get("status")
        not in {"success", "terminal_with_failures"}
    ):
        raise CampaignExecutionError("completion artifact contract failed")
    outcomes = completion.get("outcomes")
    if not isinstance(outcomes, Mapping) or set(outcomes) != {
        item["workflow_id"] for item in manifest["ptms"]
    }:
        raise CampaignExecutionError("completion workflow outcomes are incomplete")
    if any(
        outcome not in {"success", "terminal_failure"}
        for outcome in outcomes.values()
    ):
        raise CampaignExecutionError("completion contains a non-terminal outcome")

    expected_ptms = {
        item["workflow_id"]: item
        for item in manifest["ptms"]
    }
    workflows = completion.get("workflows")
    if (
        not isinstance(workflows, list)
        or len(workflows) != len(expected_ptms)
        or any(not isinstance(record, Mapping) for record in workflows)
    ):
        raise CampaignExecutionError(
            "completion workflow records are incomplete"
        )
    workflow_records = {
        record.get("workflow_id"): record
        for record in workflows
    }
    if (
        len(workflow_records) != len(workflows)
        or set(workflow_records) != set(expected_ptms)
    ):
        raise CampaignExecutionError(
            "completion workflow records are not unique and complete"
        )

    success_count = 0
    for workflow_id, record in workflow_records.items():
        status = record.get("status")
        exit_code = record.get("process_exit_code")
        if (
            record.get("schema_version") != 1
            or record.get("campaign_id") != manifest["campaign_id"]
            or record.get("manifest_sha256") != manifest["manifest_sha256"]
            or record.get("ptm_id") != expected_ptms[workflow_id]["id"]
            or record.get("ptm_sha256")
            != expected_ptms[workflow_id]["artifact"]["sha256"]
            or record.get("terminal") is not True
            or status != outcomes[workflow_id]
        ):
            raise CampaignExecutionError(
                f"completion workflow {workflow_id} is inconsistent"
            )
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise CampaignExecutionError(
                f"completion workflow {workflow_id} lacks an integer exit code"
            )
        if (status == "success") != (exit_code == 0):
            raise CampaignExecutionError(
                f"completion workflow {workflow_id} status and exit code disagree"
            )
        flags = record.get("agent_intervention_flags")
        if (
            not isinstance(flags, Mapping)
            or set(flags) != set(AGENT_FLAGS)
            or any(value is not False for value in flags.values())
        ):
            raise CampaignExecutionError(
                f"completion workflow {workflow_id} agent flags are invalid"
            )

        if status == "success":
            success_count += 1
            metrics = record.get("metrics")
            if not isinstance(metrics, Mapping):
                raise CampaignExecutionError(
                    f"completion workflow {workflow_id} lacks metrics"
                )
            _metric(
                metrics,
                "mAP",
                context=f"completion workflow {workflow_id}",
            )
            _metric(
                metrics,
                "mAP50",
                context=f"completion workflow {workflow_id}",
            )
            jobs = record.get("jobs")
            if not isinstance(jobs, Mapping):
                raise CampaignExecutionError(
                    f"completion workflow {workflow_id} lacks jobs"
                )
            train = jobs.get("train")
            evaluation = jobs.get("evaluation")
            if (
                not isinstance(train, Mapping)
                or train.get("status") != "Complete"
                or not isinstance(evaluation, Mapping)
                or evaluation.get("status") != "Complete"
            ):
                raise CampaignExecutionError(
                    f"completion workflow {workflow_id} jobs are incomplete"
                )
            train_status = train.get("status_evidence")
            evaluation_status = evaluation.get("status_evidence")
            if (
                not isinstance(train_status, Mapping)
                or train_status.get("validation_record_count")
                != manifest["qualification"]["training_epochs"]
                or train_status.get("terminal_success_message")
                != "Train finished successfully."
                or train_status.get("terminal_success") is not True
                or not isinstance(evaluation_status, Mapping)
                or evaluation_status.get("test_metric_record_count") != 1
                or evaluation_status.get("terminal_success_message")
                != "Evaluate finished successfully."
                or evaluation_status.get("terminal_success") is not True
            ):
                raise CampaignExecutionError(
                    f"completion workflow {workflow_id} status evidence "
                    "is incomplete"
                )
            for action, job, action_status in (
                ("train", train, train_status),
                ("evaluate", evaluation, evaluation_status),
            ):
                result_root = job.get("result_root")
                status_path = action_status.get("path")
                size = action_status.get("size_bytes")
                digest = action_status.get("sha256")
                record_count = action_status.get("record_count")
                if (
                    not isinstance(result_root, str)
                    or status_path
                    != (
                        f"{result_root.rstrip('/')}/results_dir/"
                        f"{action}/status.json"
                    )
                    or isinstance(size, bool)
                    or not isinstance(size, int)
                    or size <= 0
                    or not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                    or isinstance(record_count, bool)
                    or not isinstance(record_count, int)
                    or record_count <= 0
                ):
                    raise CampaignExecutionError(
                        f"completion workflow {workflow_id} {action} "
                        "status identity is invalid"
                    )
            validation_metrics = train_status.get("validation_metrics")
            if (
                not isinstance(validation_metrics, list)
                or len(validation_metrics)
                != manifest["qualification"]["training_epochs"]
            ):
                raise CampaignExecutionError(
                    f"completion workflow {workflow_id} training metrics "
                    "are incomplete"
                )
            for item in validation_metrics:
                if not isinstance(item, Mapping):
                    raise CampaignExecutionError(
                        f"completion workflow {workflow_id} training metric "
                        "is invalid"
                    )
                _metric(
                    item,
                    "mAP",
                    context=f"completion workflow {workflow_id} training",
                )
                _metric(
                    item,
                    "mAP50",
                    context=f"completion workflow {workflow_id} training",
                )
            evaluation_metrics = evaluation_status.get("metrics")
            if not isinstance(evaluation_metrics, Mapping):
                raise CampaignExecutionError(
                    f"completion workflow {workflow_id} evaluation status "
                    "lacks metrics"
                )
            if {
                "mAP": _metric(
                    evaluation_metrics,
                    "mAP",
                    context=(
                        f"completion workflow {workflow_id} evaluation status"
                    ),
                ),
                "mAP50": _metric(
                    evaluation_metrics,
                    "mAP50",
                    context=(
                        f"completion workflow {workflow_id} evaluation status"
                    ),
                ),
            } != {
                "mAP": float(metrics["mAP"]),
                "mAP50": float(metrics["mAP50"]),
            }:
                raise CampaignExecutionError(
                    f"completion workflow {workflow_id} metrics disagree "
                    "with evaluation status"
                )
        elif (
            record.get("failure_preserved") is not True
            or not isinstance(record.get("failure"), Mapping)
        ):
            raise CampaignExecutionError(
                f"completion workflow {workflow_id} failure was not preserved"
            )

    failed_count = len(workflow_records) - success_count
    expected_status = (
        "success" if success_count == len(workflow_records)
        else "terminal_with_failures"
    )
    if (
        completion.get("status") != expected_status
        or completion.get("logical_workflows_submitted")
        != len(workflow_records)
        or completion.get("successful_workflows") != success_count
        or completion.get("failed_workflows") != failed_count
        or completion.get("workflows_started_in_parallel") is not True
        or completion.get("cpu_runs") != 0
        or completion.get("smoke_runs") != 0
        or completion.get("ministep_runs") != 0
        or completion.get("local_model_runs") != 0
        or completion.get("failures_preserved") is not True
        or completion.get("replacement_workflows_submitted") is not False
    ):
        raise CampaignExecutionError(
            "completion aggregate counts or execution contract are inconsistent"
        )
    completion["completion_sha256"] = expected
    return completion


def launch(
    manifest_path: Path,
    runtime_root: Path,
    completion_artifact: Path,
    env_file: Path = ENV_PATH,
) -> int:
    manifest = load_manifest(manifest_path)
    if runtime_root.exists() and any(runtime_root.iterdir()):
        raise CampaignExecutionError(
            f"runtime root is not empty; refusing duplicate workflows: {runtime_root}"
        )
    runtime_root.mkdir(parents=True, exist_ok=True)
    load_launch_environment(env_file)
    local = verify_local_launch_contract(manifest)
    remote = verify_remote_contract(manifest)
    atomic_json(
        runtime_root / "launch_plan.json",
        {
            "schema_version": 1,
            "campaign_id": manifest["campaign_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "launched_at_utc": utc_timestamp(),
            "direct_full_dataset_acknowledged": True,
            "workflow_ids": [item["workflow_id"] for item in manifest["ptms"]],
            "parallel": True,
            "local_provenance": local,
            "remote_provenance": remote,
        },
    )
    workflow_ids = tuple(item["workflow_id"] for item in manifest["ptms"])
    if len(workflow_ids) != 2 or len(set(workflow_ids)) != 2:
        raise CampaignExecutionError("sealed workflow cohort is not exactly two")
    context = mp.get_context("spawn")
    processes = {
        workflow_id: context.Process(
            target=_workflow_process,
            args=(str(manifest_path), str(runtime_root), workflow_id),
            name=f"deformable-detr-{workflow_id}",
        )
        for workflow_id in workflow_ids
    }
    for process in processes.values():
        process.start()
    for process in processes.values():
        process.join()
    completion = build_completion(
        manifest,
        runtime_root,
        workflow_ids,
        {name: process.exitcode for name, process in processes.items()},
    )
    atomic_json(completion_artifact, completion)
    persisted = json.loads(completion_artifact.read_text(encoding="utf-8"))
    validated = validate_completion(persisted, manifest)
    if validated["status"] == "success":
        return 0
    return 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--completion-artifact", type=Path)
    parser.add_argument("--env-file", type=Path, default=ENV_PATH)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument(
        "--acknowledge-direct-full-dataset",
        action="store_true",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    completion = (
        args.completion_artifact
        if args.completion_artifact is not None
        else args.runtime_root / manifest["completion_contract"][
            "terminal_artifact_name"
        ]
    )
    if not args.launch:
        print(
            json.dumps(
                {
                    "campaign_id": manifest["campaign_id"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "launch": False,
                    "logical_workflows": 2,
                    "parallel": True,
                    "training_epochs": 10,
                    "gpus_per_workflow": 8,
                    "cpu_runs": 0,
                    "smoke_runs": 0,
                    "required_launch_flags": [
                        "--launch",
                        "--acknowledge-direct-full-dataset",
                    ],
                    "completion_artifact": str(completion),
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.acknowledge_direct_full_dataset:
        raise CampaignExecutionError(
            "--launch requires --acknowledge-direct-full-dataset"
        )
    return launch(
        args.manifest,
        args.runtime_root,
        completion,
        args.env_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())
