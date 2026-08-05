#!/usr/bin/env python3

"""Seal the four-workflow direct RT-DETR qualification campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tao_automl.ptm_registry import canonical_sha256, load_ptm_registry


HERE = Path(__file__).resolve().parent
DEFAULT_INPUTS = HERE / "campaign.inputs.v1.json"
DEFAULT_OUTPUT = HERE / "campaign.v1.json"
PRIOR_MANIFEST = HERE / "campaign.pre_checkpoint_fix.v1.json"
EXPECTED_PTMS = (
    "rtdetr.trafficcam.resnet50.trainable.v2.0",
    "rtdetr.trafficcam.resnet18.trainable.v2.0",
    "rtdetr.warehouse.resnet50.trainable.v1.0.2",
    "rtdetr.warehouse.efficientvit_l2.trainable.v1.0",
)
WORKFLOW_IDS = {
    "rtdetr.trafficcam.resnet50.trainable.v2.0": "trafficcam_resnet50",
    "rtdetr.trafficcam.resnet18.trainable.v2.0": "trafficcam_resnet18",
    "rtdetr.warehouse.resnet50.trainable.v1.0.2": "warehouse_resnet50",
    "rtdetr.warehouse.efficientvit_l2.trainable.v1.0": (
        "warehouse_efficientvit_l2"
    ),
}
AGENT_FLAGS = (
    "agent_selected_candidate",
    "agent_injected_candidate",
    "agent_modified_search_space_after_results",
    "agent_changed_seed_after_results",
    "agent_changed_budget_after_results",
    "agent_changed_threshold_after_results",
    "agent_changed_ptm_after_results",
    "agent_overrode_winner",
)
HARDWARE_CONTRACT = {
    "gpu_name": "NVIDIA A100-SXM4-80GB",
    "gpu_count": 8,
    "compute_capability": "8.0",
    "nvidia_smi_memory_mib": 81920,
}
QUALIFICATION_CONTRACT = {
    "checkpoint_interval": 10,
    "evaluation_batch_size_per_gpu": 4,
    "precision": "fp32",
    "seed": 1234,
    "train_batch_size_per_gpu": 4,
    "training_epochs": 10,
    "validation_interval": 1,
}
EXECUTION_CONTRACT = {
    "kind": "direct_full_qualification",
    "cpu_runs": 0,
    "smoke_runs": 0,
    "ministep_runs": 0,
    "local_model_runs": 0,
    "qualification_workflows": 4,
    "parallel_workflows": True,
    "full_training": True,
    "full_in_training_validation": True,
    "standalone_evaluation": True,
    "requires_direct_full_dataset_acknowledgement": True,
    "submission_ready_after_artifact_preflight": True,
}
RTDETR_CHECKPOINT_CONTRACT = {
    "directory": "results_dir/train",
    "terminal_filename_template": "model_epoch_{epoch_index:03d}.pth",
    "enumeration_order_is_selection_input": False,
    "ambiguous_match_policy": "fail_closed",
}


class ManifestError(ValueError):
    """The qualification manifest is incomplete or inconsistent."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{name} must be a non-empty string")
    return value.strip()


def _absolute(value: Any, name: str, *, suffix: str | None = None) -> str:
    path = Path(_text(value, name))
    if not path.is_absolute():
        raise ManifestError(f"{name} must be an absolute path")
    if suffix is not None and not str(path).endswith(suffix):
        raise ManifestError(f"{name} must end in {suffix!r}")
    return str(path)


def _sha(value: Any, name: str) -> str:
    value = _text(value, name)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ManifestError(f"{name} must be lowercase SHA-256 hex")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ManifestError(f"{name} must be an integer >= 1")
    return value


def _git_oid(value: Any, name: str) -> str:
    value = _text(value, name)
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ManifestError(f"{name} must be a full Git object ID")
    return value


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _registry_records() -> tuple[Any, dict[str, Mapping[str, Any]]]:
    registry = load_ptm_registry()
    model = registry.to_dict()["models"]["rtdetr"]
    records = {
        item["id"]: copy.deepcopy(item) for item in model["checkpoints"]
    }
    return registry, records


def _dataset_record(inputs: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(_absolute(inputs["manifest_path"], "dataset.manifest_path"))
    expected_sha = _sha(
        inputs["manifest_sha256"], "dataset.manifest_sha256"
    )
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ManifestError("shared synthetic dataset manifest identity mismatch")
    document = json.loads(path.read_text(encoding="utf-8"))
    try:
        train = document["splits"]["train"]
        validation = document["splits"]["validation"]
    except (KeyError, TypeError) as exc:
        raise ManifestError("dataset manifest split structure is invalid") from exc
    expected = {
        "dataset_id": "tao_od_synthetic_full_dino_coco",
        "task": "object_detection",
        "num_classes_with_background": 5,
        "eval_class_ids": [1, 2, 3, 4],
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise ManifestError(f"dataset manifest {key} changed")
    identities = {
        "train": {
            "image_dir": train["images"]["path"],
            "image_tree": train["images"]["identity"],
            "annotation": train["annotation"]["path"],
            "annotation_sha256": train["annotation"]["sha256"],
            "annotation_size_bytes": train["annotation"]["size_bytes"],
            "image_count": train["annotation"]["image_count"],
            "annotation_count": train["annotation"]["annotation_count"],
        },
        "validation": {
            "image_dir": validation["images"]["path"],
            "image_tree": validation["images"]["identity"],
            "annotation": validation["annotation"]["path"],
            "annotation_sha256": validation["annotation"]["sha256"],
            "annotation_size_bytes": validation["annotation"]["size_bytes"],
            "image_count": validation["annotation"]["image_count"],
            "annotation_count": validation["annotation"]["annotation_count"],
        },
    }
    if (
        identities["train"]["image_count"] != 1414
        or identities["validation"]["image_count"] != 353
        or identities["train"]["annotation_sha256"]
        != "7401a1245dc0b691c40f9f53cf4f46f9b96a3e0bc3dcfd357de038074acc1994"
        or identities["validation"]["annotation_sha256"]
        != "9b715b689e9a17588805faad26ed94597886d28ac687438dcb778de433f997af"
    ):
        raise ManifestError("shared synthetic split identity changed")
    return {
        "id": document["dataset_id"],
        "source_uri": document["source"]["original_uri"],
        "staged_lustre_root": document["source"]["staged_lustre_root"],
        "manifest": {
            "path": str(path),
            "sha256": expected_sha,
            "size_bytes": path.stat().st_size,
        },
        "num_classes": 5,
        "eval_class_ids": [1, 2, 3, 4],
        "remap_mscoco_category": False,
        "splits": identities,
    }


def build_manifest(inputs: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(inputs, Mapping):
        raise ManifestError("campaign inputs must be an object")
    source = inputs.get("source")
    runtime = inputs.get("runtime")
    dataset_inputs = inputs.get("dataset")
    qualification = inputs.get("qualification")
    if not all(
        isinstance(value, Mapping)
        for value in (source, runtime, dataset_inputs, qualification)
    ):
        raise ManifestError(
            "source, runtime, dataset, and qualification must be objects"
        )

    repository = Path(_absolute(source["repository"], "source.repository"))
    source_commit = _git_oid(source["commit"], "source.commit")
    _git(repository, "cat-file", "-e", f"{source_commit}^{{commit}}")
    head = _git(repository, "rev-parse", "HEAD")
    if subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "merge-base",
            "--is-ancestor",
            source_commit,
            head,
        ],
        check=False,
        capture_output=True,
        timeout=30,
    ).returncode != 0:
        raise ManifestError("source.commit must be an ancestor of repository HEAD")

    skill_dir = Path(_absolute(runtime["skill_dir"], "runtime.skill_dir"))
    sdk_dir = Path(_absolute(runtime["sdk_dir"], "runtime.sdk_dir"))
    skill_revision = _git_oid(
        runtime["skill_revision"], "runtime.skill_revision"
    )
    sdk_revision = _git_oid(runtime["sdk_revision"], "runtime.sdk_revision")
    if _git(skill_dir, "rev-parse", "HEAD") != skill_revision:
        raise ManifestError("runtime skill revision mismatch")
    if _git(sdk_dir, "rev-parse", "HEAD") != sdk_revision:
        raise ManifestError("runtime SDK revision mismatch")
    if _git(skill_dir, "status", "--porcelain"):
        raise ManifestError("runtime skill checkout must be clean")
    if _git(sdk_dir, "status", "--porcelain"):
        raise ManifestError("runtime SDK checkout must be clean")
    if dict(qualification) != QUALIFICATION_CONTRACT:
        raise ManifestError("qualification budget or runtime profile changed")

    registry, records = _registry_records()
    supplied = inputs.get("ptm_runtime")
    if not isinstance(supplied, Mapping) or tuple(supplied) != EXPECTED_PTMS:
        raise ManifestError(
            "ptm_runtime must contain the four registry PTMs in frozen order"
        )
    ptms = []
    for checkpoint_id in EXPECTED_PTMS:
        record = records.get(checkpoint_id)
        if not isinstance(record, Mapping):
            raise ManifestError(f"registry checkpoint missing: {checkpoint_id}")
        source_info = record.get("source", {})
        if (
            record.get("status") != "unverified"
            or source_info.get("provider") != "ngc"
            or source_info.get("registry") != "nvidia/tao"
            or source_info.get("official") is not True
        ):
            raise ManifestError(
                f"PTM registry qualification state changed: {checkpoint_id}"
            )
        supplied_record = supplied[checkpoint_id]
        if not isinstance(supplied_record, Mapping):
            raise ManifestError(f"invalid PTM runtime record: {checkpoint_id}")
        artifact_sha = _sha(
            supplied_record["artifact_sha256"],
            f"{checkpoint_id}.artifact_sha256",
        )
        artifact_size = _positive_int(
            supplied_record["artifact_size_bytes"],
            f"{checkpoint_id}.artifact_size_bytes",
        )
        if (
            artifact_sha != record.get("sha256")
            or artifact_size != record["expected_size_bytes"]
        ):
            raise ManifestError(
                f"PTM runtime identity disagrees with registry: {checkpoint_id}"
            )
        checkpoint_spec = record["checkpoint_spec_file"]
        spec_path = repository / "src/tao_automl" / checkpoint_spec["path"]
        spec_sha = _sha(
            checkpoint_spec["sha256"],
            f"{checkpoint_id}.checkpoint_spec_file.sha256",
        )
        if not spec_path.is_file() or sha256_file(spec_path) != spec_sha:
            raise ManifestError(f"checkpoint spec identity mismatch: {checkpoint_id}")
        ptms.append(
            {
                "id": checkpoint_id,
                "workflow_id": WORKFLOW_IDS[checkpoint_id],
                "registry_status_before_qualification": record["status"],
                "registry_record_sha256": canonical_sha256(record),
                "source_identity": source_info["immutable_identity"],
                "source": copy.deepcopy(source_info),
                "backbone": record["backbone"],
                "checkpoint_target": record["checkpoint_target"],
                "input_contract": copy.deepcopy(record["input_contract"]),
                "artifact": {
                    "sha256": artifact_sha,
                    "size_bytes": artifact_size,
                    "local_source_path": _absolute(
                        supplied_record["local_source_path"],
                        f"{checkpoint_id}.local_source_path",
                        suffix=".pth",
                    ),
                    "slurm_path": _absolute(
                        supplied_record["slurm_path"],
                        f"{checkpoint_id}.slurm_path",
                        suffix=".pth",
                    ),
                    "availability_required_at_launch": True,
                },
                "checkpoint_spec": {
                    "path": str(spec_path),
                    "sha256": spec_sha,
                },
                "default_spec_overrides": copy.deepcopy(
                    record["default_spec_overrides"]
                ),
                "agent_intervention_flags": {
                    name: False for name in AGENT_FLAGS
                },
            }
        )

    if not PRIOR_MANIFEST.is_file():
        raise ManifestError("pre-fix campaign manifest is unavailable")
    prior_manifest = json.loads(PRIOR_MANIFEST.read_text(encoding="utf-8"))
    prior_payload = copy.deepcopy(prior_manifest)
    prior_manifest_sha = prior_payload.pop("manifest_sha256", None)
    if (
        prior_manifest_sha != canonical_sha(prior_payload)
        or prior_manifest_sha
        != "a0f6a0d5aa54a9c2dcbdf70a87c1138f708965b11c8e7060b83f7aaabc5be141"
        or prior_manifest.get("campaign_id") != inputs["campaign_id"]
        or tuple(item["id"] for item in prior_manifest.get("ptms", []))
        != EXPECTED_PTMS
    ):
        raise ManifestError("pre-fix campaign manifest identity changed")

    launcher = HERE / "run_campaign.py"
    resume_launcher = HERE / "resume_evaluation.py"
    if not resume_launcher.is_file():
        raise ManifestError("evaluation-only resume launcher is unavailable")
    workflow_support = (
        HERE.parent / "deformable_detr_campaign" / "run_campaign.py"
    )
    if not workflow_support.is_file():
        raise ManifestError("shared direct-workflow support is unavailable")
    manifest = {
        "schema_version": 1,
        "campaign_id": _text(inputs["campaign_id"], "campaign_id"),
        "model": "rtdetr",
        "task": "object_detection",
        "cpu_runs": 0,
        "smoke_runs": 0,
        "execution": copy.deepcopy(EXECUTION_CONTRACT),
        "source": {
            "repository": str(repository),
            "commit": source_commit,
            "launch_head_policy": "clean_descendant",
        },
        "runtime": {
            "platform": "slurm",
            "skill_dir": str(skill_dir),
            "skill_revision": skill_revision,
            "sdk_dir": str(sdk_dir),
            "sdk_revision": sdk_revision,
            "tao_version": _text(runtime["tao_version"], "runtime.tao_version"),
            "image_reference": _text(
                runtime["image_reference"], "runtime.image_reference"
            ),
            "sqsh_path": _absolute(
                runtime["sqsh_path"], "runtime.sqsh_path", suffix=".sqsh"
            ),
            "sqsh_sha256": _sha(
                runtime["sqsh_sha256"], "runtime.sqsh_sha256"
            ),
            "sqsh_size_bytes": _positive_int(
                runtime["sqsh_size_bytes"], "runtime.sqsh_size_bytes"
            ),
            "sqsh_direct_path": True,
            "slurm_image_conversion": False,
            "partition": _text(runtime["partition"], "runtime.partition"),
            "account": _text(runtime["account"], "runtime.account"),
            "base_results_dir": _absolute(
                runtime["base_results_dir"], "runtime.base_results_dir"
            ),
            "container_mounts": _absolute(
                runtime["container_mounts"], "runtime.container_mounts"
            ),
            "nodes": 1,
            "tasks_per_node": 1,
            "gpus_per_node": 8,
            "distributed_workers_per_node": 8,
            "hardware_contract": copy.deepcopy(HARDWARE_CONTRACT),
            "time_hours": 4.0,
            "timeout_hours": 3.8,
            "slurm_use_sqsh_conversion": False,
            "slurm_use_requeue": True,
            "max_infrastructure_retries": 10,
        },
        "dataset": _dataset_record(dataset_inputs),
        "qualification": dict(qualification),
        "ptms": ptms,
        "checkpoint_resolution": copy.deepcopy(RTDETR_CHECKPOINT_CONTRACT),
        "resume_contract": {
            "prior_manifest": {
                "path": str(PRIOR_MANIFEST),
                "file_sha256": sha256_file(PRIOR_MANIFEST),
                "manifest_sha256": prior_manifest_sha,
            },
            "eligible_prior_status": "terminal_failure",
            "eligible_train_status": "Complete",
            "eligible_failure_type": "CampaignExecutionError",
            "eligible_failure_message_regex": (
                "^training job [0-9a-f-]{36} emitted 0 exact "
                "'model_epoch_009_step_\\*\\.pth' terminal checkpoints$"
            ),
            "reuse_completed_training_job": True,
            "training_job_resubmission": False,
            "prior_workflow_artifact_immutable": True,
            "prior_completion_artifact_immutable": True,
            "resume_workflow_artifact_name": (
                "workflow_resume_completion.json"
            ),
            "resume_completion_artifact_name": "completion.resume.json",
            "resume_launcher": {
                "path": str(resume_launcher),
                "sha256": sha256_file(resume_launcher),
            },
        },
        "failure_policy": {
            "preserve_terminal_failures": True,
            "replace_failed_workflow": False,
            "manual_ptm_substitution": False,
            "maximum_logical_workflows": 4,
            "maximum_train_jobs": 4,
            "maximum_evaluation_jobs": 4,
        },
        "completion_contract": {
            "terminal_artifact_name": "completion.json",
            "terminal_on_partial_or_total_failure": True,
            "workflow_artifact_name": "workflow_completion.json",
            "automatic_after_all_workflows_terminal": True,
        },
        "integrity": {
            "inputs_sha256": sha256_file(DEFAULT_INPUTS),
            "manifest_generator_sha256": sha256_file(Path(__file__)),
            "launcher_sha256": (
                sha256_file(launcher) if launcher.is_file() else None
            ),
            "resume_launcher_sha256": sha256_file(resume_launcher),
            "workflow_support_sha256": sha256_file(workflow_support),
            "prior_manifest_file_sha256": sha256_file(PRIOR_MANIFEST),
            "ptm_registry_sha256": registry.document_sha256,
        },
    }
    return manifest


def seal_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(manifest))
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = canonical_sha(payload)
    return payload


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(manifest))
    expected = value.pop("manifest_sha256", None)
    if expected != canonical_sha(value):
        raise ManifestError("manifest canonical integrity verification failed")
    if (
        manifest.get("model") != "rtdetr"
        or manifest.get("cpu_runs") != 0
        or manifest.get("smoke_runs") != 0
        or manifest.get("execution") != EXECUTION_CONTRACT
        or manifest.get("qualification") != QUALIFICATION_CONTRACT
        or manifest.get("checkpoint_resolution")
        != RTDETR_CHECKPOINT_CONTRACT
    ):
        raise ManifestError("direct-full RT-DETR execution contract changed")
    if tuple(item["id"] for item in manifest["ptms"]) != EXPECTED_PTMS:
        raise ManifestError("the sealed four-PTM cohort changed")
    runtime = manifest["runtime"]
    if (
        runtime["nodes"] != 1
        or runtime["tasks_per_node"] != 1
        or runtime["gpus_per_node"] != 8
        or runtime["hardware_contract"] != HARDWARE_CONTRACT
        or runtime["sqsh_direct_path"] is not True
        or runtime["slurm_image_conversion"] is not False
    ):
        raise ManifestError("one-node/eight-A100 pinned-SQSH contract changed")
    dataset = manifest["dataset"]
    if (
        dataset["num_classes"] != 5
        or dataset["eval_class_ids"] != [1, 2, 3, 4]
        or dataset["remap_mscoco_category"] is not False
    ):
        raise ManifestError("synthetic COCO class/remap contract changed")
    resume = manifest.get("resume_contract")
    if (
        not isinstance(resume, Mapping)
        or resume.get("reuse_completed_training_job") is not True
        or resume.get("training_job_resubmission") is not False
        or resume.get("prior_workflow_artifact_immutable") is not True
        or resume.get("prior_completion_artifact_immutable") is not True
        or resume.get("prior_manifest", {}).get("manifest_sha256")
        != "a0f6a0d5aa54a9c2dcbdf70a87c1138f708965b11c8e7060b83f7aaabc5be141"
    ):
        raise ManifestError("completed-training resume contract changed")
    return copy.deepcopy(dict(manifest))


def load_manifest(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    return validate_manifest(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        manifest = load_manifest(args.output)
        print(manifest["manifest_sha256"])
        return 0
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    manifest = seal_manifest(build_manifest(inputs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
