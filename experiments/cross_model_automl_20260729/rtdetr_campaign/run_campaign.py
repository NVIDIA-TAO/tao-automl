#!/usr/bin/env python3

"""Launch four parallel full-dataset RT-DETR PTM qualifications.

The executable path requires both ``--launch`` and
``--acknowledge-direct-full-dataset``. Each workflow trains one immutable
registry PTM for ten complete epochs on one eight-A100 node, validates every
epoch, and then evaluates the exact terminal checkpoint. There is no CPU,
smoke, mini-step, local-model, replacement, or container-conversion path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from tao_automl.ptm_registry import load_ptm_registry

try:
    from . import manifest_generator
except ImportError:  # pragma: no cover - direct script execution
    import manifest_generator  # type: ignore[no-redef]

try:
    from experiments.cross_model_automl_20260729.deformable_detr_campaign import (
        run_campaign as workflow_support,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    repository = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repository))
    from experiments.cross_model_automl_20260729.deformable_detr_campaign import (
        run_campaign as workflow_support,
    )


HERE = Path(__file__).resolve().parent
ENV_PATH = Path("/localhome/local-rarunachalam/.tao/config.env")
DEFAULT_RUNTIME_ROOT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/rtdetr_qualification"
)
CampaignExecutionError = workflow_support.CampaignExecutionError
atomic_json = workflow_support.atomic_json
append_jsonl = workflow_support.append_jsonl
utc_timestamp = workflow_support.utc_timestamp
load_launch_environment = workflow_support.load_launch_environment
remote_output = workflow_support.remote_output


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _verified_local_file(
    path: str | Path,
    expected_sha256: str,
    expected_size: int,
    *,
    label: str,
) -> dict[str, Any]:
    value = Path(path)
    if not value.is_file():
        raise CampaignExecutionError(f"{label} is unavailable: {value}")
    size = value.stat().st_size
    digest = manifest_generator.sha256_file(value)
    if size != expected_size or digest != expected_sha256:
        raise CampaignExecutionError(f"{label} identity mismatch")
    return {
        "path": str(value),
        "sha256": digest,
        "size_bytes": size,
    }


def verify_local_launch_contract(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
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
        raise CampaignExecutionError(
            "sealed source is not an ancestor of launch HEAD"
        )
    if _git(repository, "status", "--porcelain"):
        raise CampaignExecutionError("launch repository must be clean")

    integrity = manifest["integrity"]
    required = {
        HERE / "campaign.inputs.v1.json": integrity["inputs_sha256"],
        HERE / "manifest_generator.py": integrity["manifest_generator_sha256"],
        HERE / "run_campaign.py": integrity["launcher_sha256"],
        HERE / "resume_evaluation.py": integrity[
            "resume_launcher_sha256"
        ],
        (
            HERE.parent
            / "deformable_detr_campaign"
            / "run_campaign.py"
        ): integrity["workflow_support_sha256"],
        manifest_generator.PRIOR_MANIFEST: integrity[
            "prior_manifest_file_sha256"
        ],
        Path(manifest["dataset"]["manifest"]["path"]): manifest["dataset"][
            "manifest"
        ]["sha256"],
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
        if (
            not isinstance(expected, str)
            or not path.is_file()
            or manifest_generator.sha256_file(path) != expected
        ):
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
        raise CampaignExecutionError("sealed PTM registry identity changed")
    local_ptms = {
        item["id"]: _verified_local_file(
            item["artifact"]["local_source_path"],
            item["artifact"]["sha256"],
            item["artifact"]["size_bytes"],
            label=f"local PTM {item['id']}",
        )
        for item in manifest["ptms"]
    }
    return {
        "source_commit": source["commit"],
        "launch_head": head,
        "source_is_ancestor": True,
        "required_files": [
            {"path": str(path), "sha256": digest}
            for path, digest in required.items()
        ],
        "sources": {
            **sources,
            "ptm_registry": {
                "path": str(registry_path),
                "document_sha256": registry.document_sha256,
                "file_sha256": manifest_generator.sha256_file(registry_path),
            },
        },
        "local_ptms": local_ptms,
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
                item
                for item in existing.split(os.pathsep)
                if item and item != sdk_dir
            ],
        ]
    )
    os.environ.update(
        {
            # The image argument is already an immutable .sqsh path. Enabling
            # SDK conversion here would incorrectly schedule a CPU conversion.
            "SLURM_USE_SQSH": "false",
            "SLURM_USE_REQUEUE": "true",
            "SLURM_TIME_HOURS": str(runtime["time_hours"]),
            "SLURM_TIMEOUT_HOURS": str(runtime["timeout_hours"]),
            "SLURM_MAX_GPUS_PER_NODE": "8",
            "SLURM_PARTITION": runtime["partition"],
            "SLURM_ACCOUNT": runtime["account"],
            "SLURM_BASE_RESULTS_DIR": runtime["base_results_dir"],
            "SLURM_CONTAINER_MOUNTS": runtime["container_mounts"],
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )


def _remote_identity(
    path: str,
    expected_sha256: str,
    expected_size: int,
    *,
    label: str,
    timeout: int,
) -> dict[str, Any]:
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
        timeout=timeout,
    ).strip().splitlines()
    if len(lines) != 2:
        raise CampaignExecutionError(f"remote probe incomplete for {label}")
    size = int(lines[0])
    digest = lines[1].split()[0]
    if size != expected_size or digest != expected_sha256:
        raise CampaignExecutionError(f"remote identity mismatch for {label}")
    return {"path": path, "sha256": digest, "size_bytes": size}


def _remote_image_tree(path: str) -> dict[str, Any]:
    script = (
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
    return json.loads(
        remote_output(
            f"python3 -c {shlex.quote(script)} {shlex.quote(path)}",
            timeout=3600,
        ).strip()
    )


def verify_remote_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    runtime = manifest["runtime"]
    dataset = manifest["dataset"]
    verified = {
        "sqsh": _remote_identity(
            runtime["sqsh_path"],
            runtime["sqsh_sha256"],
            runtime["sqsh_size_bytes"],
            label="sqsh",
            timeout=7200,
        )
    }
    for split_name, split in dataset["splits"].items():
        verified[f"{split_name}_annotation"] = _remote_identity(
            split["annotation"],
            split["annotation_sha256"],
            split["annotation_size_bytes"],
            label=f"{split_name} annotation",
            timeout=1800,
        )
        actual_tree = _remote_image_tree(split["image_dir"])
        if actual_tree != split["image_tree"]:
            raise CampaignExecutionError(
                f"remote {split_name} image-tree identity mismatch"
            )
        verified[f"{split_name}_images"] = {
            "path": split["image_dir"],
            **actual_tree,
        }
    for ptm in manifest["ptms"]:
        verified[f"ptm:{ptm['id']}"] = _remote_identity(
            ptm["artifact"]["slurm_path"],
            ptm["artifact"]["sha256"],
            ptm["artifact"]["size_bytes"],
            label=f"PTM {ptm['id']}",
            timeout=1800,
        )
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
    manifest: Mapping[str, Any],
    workflow_id: str,
) -> Mapping[str, Any]:
    matches = [
        item for item in manifest["ptms"] if item["workflow_id"] == workflow_id
    ]
    if len(matches) != 1:
        raise CampaignExecutionError(f"no unique PTM for {workflow_id}")
    return matches[0]


def _base_spec(
    manifest: Mapping[str, Any],
    workflow_id: str,
    action: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    ptm = _ptm_by_workflow(manifest, workflow_id)
    template = (
        Path(manifest["runtime"]["skill_dir"])
        / f"references/spec_template_{action}.yaml"
    )
    spec = yaml.safe_load(template.read_text(encoding="utf-8"))
    checkpoint_spec = yaml.safe_load(
        Path(ptm["checkpoint_spec"]["path"]).read_text(encoding="utf-8")
    )
    spec = _merge(spec, checkpoint_spec)
    spec = _merge(spec, ptm["default_spec_overrides"])
    return spec, ptm


def _augmentation(ptm: Mapping[str, Any]) -> dict[str, Any]:
    contract = ptm["input_contract"]
    height = int(contract["height"])
    width = int(contract["width"])
    return {
        "train_spatial_size": [height, width],
        "eval_spatial_size": [height, width],
        "preserve_aspect_ratio": bool(
            contract["preprocessing"]["preserve_aspect_ratio"]
        ),
    }


def build_train_spec(
    manifest: Mapping[str, Any],
    workflow_id: str,
) -> dict[str, Any]:
    spec, ptm = _base_spec(manifest, workflow_id, "train")
    dataset = manifest["dataset"]
    train = dataset["splits"]["train"]
    validation = dataset["splits"]["validation"]
    qualification = manifest["qualification"]
    spec = _merge(
        spec,
        {
            "results_dir": "",
            "wandb": {"enable": False},
            "dataset": {
                "train_data_sources": [
                    {
                        "image_dir": train["image_dir"],
                        "json_file": train["annotation"],
                    }
                ],
                # RT-DETR explicitly requires a mapping, not a one-item list.
                "val_data_sources": {
                    "image_dir": validation["image_dir"],
                    "json_file": validation["annotation"],
                },
                "batch_size": qualification["train_batch_size_per_gpu"],
                "num_classes": dataset["num_classes"],
                "eval_class_ids": dataset["eval_class_ids"],
                "remap_mscoco_category": False,
                "augmentation": _augmentation(ptm),
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
    spec, ptm = _base_spec(manifest, workflow_id, "evaluate")
    dataset = manifest["dataset"]
    validation = dataset["splits"]["validation"]
    batch_size = manifest["qualification"]["evaluation_batch_size_per_gpu"]
    spec = _merge(
        spec,
        {
            "results_dir": "",
            "wandb": {"enable": False},
            "dataset": {
                "test_data_sources": {
                    "image_dir": validation["image_dir"],
                    "json_file": validation["annotation"],
                },
                "batch_size": batch_size,
                "num_classes": dataset["num_classes"],
                "eval_class_ids": dataset["eval_class_ids"],
                "remap_mscoco_category": False,
                "augmentation": _augmentation(ptm),
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


def _job_entrypoint(
    manifest: Mapping[str, Any],
    action_name: str,
    spec: Mapping[str, Any],
) -> tuple[str, str]:
    from tao_sdk.script_runner import build_entrypoint

    metadata = yaml.safe_load(
        (
            Path(manifest["runtime"]["skill_dir"])
            / "references/skill_info.yaml"
        ).read_text(encoding="utf-8")
    )
    action = metadata["actions"][action_name]
    command = workflow_support._gpu_guard(action["command"])
    entrypoint = build_entrypoint(
        command=command,
        specs=spec,
        inputs=action["inputs"],
        outputs=action["outputs"],
        config_format=action["config_format"],
        upload_excludes=action.get("upload_excludes", []),
    )
    return entrypoint["command"], hashlib.sha256(
        entrypoint["command"].encode("utf-8")
    ).hexdigest()


def _initial_workflow(
    manifest: Mapping[str, Any],
    workflow_id: str,
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
        "agent_intervention_flags": {
            name: False for name in manifest_generator.AGENT_FLAGS
        },
        "jobs": {},
    }


def _submit_job(sdk: Any, manifest: Mapping[str, Any], command: str) -> Any:
    runtime = manifest["runtime"]
    return sdk.create_job(
        image=runtime["sqsh_path"],
        command=command,
        gpu_count=8,
        num_nodes=1,
        partition=runtime["partition"],
        account=runtime["account"],
    )


def _terminal_checkpoint(
    sdk: Any,
    job_id: str,
    *,
    training_epochs: int,
) -> dict[str, Any]:
    """Resolve RT-DETR's exact terminal checkpoint, failing on ambiguity.

    RT-DETR emits ``model_epoch_NNN.pth``. It does not use the
    ``model_epoch_NNN_step_*.pth`` convention used by Deformable DETR.
    Candidate enumeration order is never used to choose a checkpoint.
    """
    root = workflow_support._local_lustre_path(
        sdk.get_job_results_dir(job_id)
    )
    terminal_epoch_index = training_epochs - 1
    filename = f"model_epoch_{terminal_epoch_index:03d}.pth"
    train_dir = f"{root.rstrip('/')}/results_dir/train"
    expected_path = f"{train_dir}/{filename}"
    paths = sorted(
        {
            line.strip()
            for line in remote_output(
                f"find {shlex.quote(train_dir)} -maxdepth 1 -type f "
                f"-name {shlex.quote(filename)} -print"
            ).splitlines()
            if line.strip()
        }
    )
    if len(paths) != 1:
        rendered = ", ".join(paths) if paths else "<none>"
        raise CampaignExecutionError(
            f"RT-DETR training job {job_id} emitted {len(paths)} exact "
            f"{filename!r} terminal checkpoints; matches={rendered}"
        )
    checkpoint = paths[0]
    if checkpoint != expected_path:
        raise CampaignExecutionError(
            "RT-DETR terminal checkpoint did not resolve to its exact "
            f"train output path: {checkpoint}"
        )
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
            "RT-DETR terminal checkpoint identity probe was incomplete"
        )
    try:
        size = int(identity[0])
    except ValueError as exc:
        raise CampaignExecutionError(
            "RT-DETR terminal checkpoint size was invalid"
        ) from exc
    digest = identity[1].split()[0]
    if size <= 0 or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise CampaignExecutionError(
            "RT-DETR terminal checkpoint identity was invalid"
        )
    return {
        "path": checkpoint,
        "sha256": digest,
        "size_bytes": size,
        "training_epochs": training_epochs,
        "terminal_epoch_index": terminal_epoch_index,
        "filename": filename,
        "naming_contract": "rtdetr_model_epoch_without_step_suffix",
        "ambiguity_policy": "fail_closed",
    }


def _run_workflow(
    manifest_path: str,
    runtime_root: str,
    workflow_id: str,
) -> None:
    manifest = manifest_generator.load_manifest(manifest_path)
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
            "spec_sha256": manifest_generator.canonical_sha(train_spec),
            "command_sha256": train_command_sha,
            "full_dataset": True,
            "training_epochs": 10,
            "validation_interval": 1,
            "nodes": 1,
            "gpus": 8,
        }
        atomic_json(evidence_path, evidence)
        train_status = workflow_support._wait_for_job(
            sdk,
            train_job.id,
            events=events,
            workflow_id=workflow_id,
            phase="train",
        )
        evidence["jobs"]["train"]["status"] = train_status
        evidence["jobs"]["train"]["terminal_at_utc"] = utc_timestamp()
        evidence["jobs"]["train"]["result_root"] = (
            workflow_support._local_lustre_path(
                sdk.get_job_results_dir(train_job.id)
            )
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
            workflow_support._training_status_evidence(
                sdk,
                train_job.id,
                expected_validation_records=10,
            )
        )
        checkpoint = _terminal_checkpoint(
            sdk,
            train_job.id,
            training_epochs=10,
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
            "spec_sha256": manifest_generator.canonical_sha(evaluation_spec),
            "command_sha256": evaluation_command_sha,
            "full_validation_split": True,
            "nodes": 1,
            "gpus": 8,
            "checkpoint": checkpoint["path"],
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_size_bytes": checkpoint["size_bytes"],
        }
        atomic_json(evidence_path, evidence)
        evaluation_status = workflow_support._wait_for_job(
            sdk,
            evaluation_job.id,
            events=events,
            workflow_id=workflow_id,
            phase="standalone_evaluation",
        )
        evidence["jobs"]["evaluation"]["status"] = evaluation_status
        evidence["jobs"]["evaluation"]["terminal_at_utc"] = utc_timestamp()
        evidence["jobs"]["evaluation"]["result_root"] = (
            workflow_support._local_lustre_path(
                sdk.get_job_results_dir(evaluation_job.id)
            )
        )
        if evaluation_status != "Complete":
            evidence["jobs"]["evaluation"]["failure_analysis"] = (
                sdk.get_failure_analysis(evaluation_job.id)
            )
            raise CampaignExecutionError(
                "standalone evaluation job ended with terminal status "
                f"{evaluation_status}"
            )
        evaluation_evidence = workflow_support._evaluation_status_evidence(
            sdk, evaluation_job.id
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
    _run_workflow(manifest_path, runtime_root, workflow_id)


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
            record.update(
                {
                    "status": "terminal_failure",
                    "terminal": True,
                    "failure_preserved": True,
                    "failure": {
                        "type": "IncompleteWorkflow",
                        "message": (
                            "worker exited before recording a terminal outcome"
                        ),
                        "replacement_submitted": False,
                    },
                }
            )
            atomic_json(path, record)
        record["process_exit_code"] = exit_codes.get(workflow_id)
        workflows.append(record)
    success_count = sum(item["status"] == "success" for item in workflows)
    count = len(workflow_ids)
    payload = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "model": "rtdetr",
        "manifest_sha256": manifest["manifest_sha256"],
        "terminal": True,
        "status": "success" if success_count == count else "terminal_with_failures",
        "terminal_at_utc": utc_timestamp(),
        "logical_workflows_submitted": count,
        "successful_workflows": success_count,
        "failed_workflows": count - success_count,
        "workflows_started_in_parallel": True,
        "completion_generated_automatically": True,
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
    payload["completion_sha256"] = manifest_generator.canonical_sha(payload)
    return payload


def _metric(value: Any, name: str) -> float:
    return workflow_support._metric(value, name, context="completion")


def validate_completion(
    value: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    completion = copy.deepcopy(dict(value))
    expected_hash = completion.pop("completion_sha256", None)
    if expected_hash != manifest_generator.canonical_sha(completion):
        raise CampaignExecutionError("completion artifact integrity failed")
    expected_ptms = {
        item["workflow_id"]: item for item in manifest["ptms"]
    }
    outcomes = completion.get("outcomes")
    workflows = completion.get("workflows")
    if (
        completion.get("schema_version") != 1
        or completion.get("campaign_id") != manifest["campaign_id"]
        or completion.get("model") != "rtdetr"
        or completion.get("manifest_sha256") != manifest["manifest_sha256"]
        or completion.get("terminal") is not True
        or completion.get("completion_generated_automatically") is not True
        or not isinstance(outcomes, Mapping)
        or set(outcomes) != set(expected_ptms)
        or not isinstance(workflows, list)
        or len(workflows) != len(expected_ptms)
    ):
        raise CampaignExecutionError("completion artifact contract failed")
    records = {
        record.get("workflow_id"): record
        for record in workflows
        if isinstance(record, Mapping)
    }
    if len(records) != len(workflows) or set(records) != set(expected_ptms):
        raise CampaignExecutionError(
            "completion workflow records are not unique and complete"
        )

    success_count = 0
    for workflow_id, record in records.items():
        status = record.get("status")
        exit_code = record.get("process_exit_code")
        ptm = expected_ptms[workflow_id]
        if (
            record.get("campaign_id") != manifest["campaign_id"]
            or record.get("manifest_sha256") != manifest["manifest_sha256"]
            or record.get("ptm_id") != ptm["id"]
            or record.get("ptm_sha256") != ptm["artifact"]["sha256"]
            or record.get("terminal") is not True
            or status != outcomes[workflow_id]
            or status not in {"success", "terminal_failure"}
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
            or set(flags) != set(manifest_generator.AGENT_FLAGS)
            or any(flag is not False for flag in flags.values())
        ):
            raise CampaignExecutionError(
                f"completion workflow {workflow_id} agent flags are invalid"
            )
        if status == "success":
            success_count += 1
            metrics = record.get("metrics")
            jobs = record.get("jobs")
            if not isinstance(metrics, Mapping) or not isinstance(jobs, Mapping):
                raise CampaignExecutionError(
                    f"completion workflow {workflow_id} lacks evidence"
                )
            _metric(metrics, "mAP")
            _metric(metrics, "mAP50")
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
                or train_status.get("validation_record_count") != 10
                or train_status.get("terminal_success") is not True
                or not isinstance(evaluation_status, Mapping)
                or evaluation_status.get("test_metric_record_count") != 1
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
                digest = action_status.get("sha256")
                if (
                    not isinstance(result_root, str)
                    or status_path
                    != (
                        f"{result_root.rstrip('/')}/results_dir/"
                        f"{action}/status.json"
                    )
                    or not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                ):
                    raise CampaignExecutionError(
                        f"completion workflow {workflow_id} {action} "
                        "status identity is invalid"
                    )
        elif (
            record.get("failure_preserved") is not True
            or not isinstance(record.get("failure"), Mapping)
        ):
            raise CampaignExecutionError(
                f"completion workflow {workflow_id} failure was not preserved"
            )

    count = len(expected_ptms)
    failed_count = count - success_count
    expected_status = (
        "success" if success_count == count else "terminal_with_failures"
    )
    if (
        completion.get("status") != expected_status
        or completion.get("logical_workflows_submitted") != count
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
    completion["completion_sha256"] = expected_hash
    return completion


def launch(
    manifest_path: Path,
    runtime_root: Path,
    completion_artifact: Path,
    env_file: Path = ENV_PATH,
) -> int:
    manifest = manifest_generator.load_manifest(manifest_path)
    if runtime_root.exists() and any(runtime_root.iterdir()):
        raise CampaignExecutionError(
            f"runtime root is not empty; refusing duplicate workflows: {runtime_root}"
        )
    runtime_root.mkdir(parents=True, exist_ok=True)
    load_launch_environment(env_file)
    local = verify_local_launch_contract(manifest)
    remote = verify_remote_contract(manifest)
    workflow_ids = tuple(item["workflow_id"] for item in manifest["ptms"])
    if (
        len(workflow_ids) != len(manifest_generator.EXPECTED_PTMS)
        or len(set(workflow_ids)) != len(workflow_ids)
    ):
        raise CampaignExecutionError("sealed workflow cohort is not exactly four")
    atomic_json(
        runtime_root / "launch_plan.json",
        {
            "schema_version": 1,
            "campaign_id": manifest["campaign_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "launched_at_utc": utc_timestamp(),
            "direct_full_dataset_acknowledged": True,
            "workflow_ids": list(workflow_ids),
            "parallel": True,
            "local_provenance": local,
            "remote_provenance": remote,
        },
    )
    context = mp.get_context("spawn")
    processes = {
        workflow_id: context.Process(
            target=_workflow_process,
            args=(str(manifest_path), str(runtime_root), workflow_id),
            name=f"rtdetr-{workflow_id}",
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
    return 0 if validated["status"] == "success" else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=manifest_generator.DEFAULT_OUTPUT,
    )
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
    manifest = manifest_generator.load_manifest(args.manifest)
    completion = (
        args.completion_artifact
        if args.completion_artifact is not None
        else args.runtime_root
        / manifest["completion_contract"]["terminal_artifact_name"]
    )
    if not args.launch:
        print(
            json.dumps(
                {
                    "campaign_id": manifest["campaign_id"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "launch": False,
                    "logical_workflows": 4,
                    "parallel": True,
                    "training_epochs": 10,
                    "gpus_per_workflow": 8,
                    "cpu_runs": 0,
                    "smoke_runs": 0,
                    "ministep_runs": 0,
                    "required_launch_flags": [
                        "--launch",
                        "--acknowledge-direct-full-dataset",
                    ],
                    "completion_artifact": str(completion),
                    "completion_generated_automatically": True,
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
