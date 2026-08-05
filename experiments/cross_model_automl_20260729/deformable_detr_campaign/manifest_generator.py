#!/usr/bin/env python3

"""Seal the two-workflow direct Deformable DETR qualification campaign."""

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
EXPECTED_PTMS = (
    "deformable_detr.coco.gcvit_tiny.trainable.v1.0",
    "deformable_detr.coco.resnet50.trainable.v1.0",
)
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


def _verified_file(
    path_value: Any,
    expected_sha: Any,
    expected_size: Any,
    *,
    name: str,
) -> dict[str, Any]:
    path = Path(_absolute(path_value, f"{name}.path"))
    digest = _sha(expected_sha, f"{name}.sha256")
    size = _positive_int(expected_size, f"{name}.size_bytes")
    if not path.is_file():
        raise ManifestError(f"{name} is unavailable: {path}")
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise ManifestError(f"{name} local identity mismatch")
    return {"path": str(path), "sha256": digest, "size_bytes": size}


def _registry_records() -> tuple[Any, dict[str, Mapping[str, Any]]]:
    registry = load_ptm_registry()
    model = registry.to_dict()["models"]["deformable_detr"]
    records = {
        item["id"]: copy.deepcopy(item) for item in model["checkpoints"]
    }
    return registry, records


def build_manifest(inputs: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(inputs, Mapping):
        raise ManifestError("campaign inputs must be an object")
    source = inputs.get("source")
    runtime = inputs.get("runtime")
    dataset = inputs.get("dataset")
    qualification = inputs.get("qualification")
    if not all(
        isinstance(value, Mapping)
        for value in (source, runtime, dataset, qualification)
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

    manifest_file = _verified_file(
        dataset["manifest_path"],
        dataset["manifest_sha256"],
        Path(dataset["manifest_path"]).stat().st_size,
        name="dataset manifest",
    )
    integrity_file = _verified_file(
        dataset["integrity_path"],
        dataset["integrity_sha256"],
        Path(dataset["integrity_path"]).stat().st_size,
        name="dataset integrity",
    )
    if qualification != {
        "checkpoint_interval": 10,
        "evaluation_batch_size_per_gpu": 4,
        "precision": "fp32",
        "seed": 1234,
        "train_batch_size_per_gpu": 4,
        "training_epochs": 10,
        "validation_interval": 1,
    }:
        raise ManifestError("qualification budget or runtime profile changed")

    registry, records = _registry_records()
    supplied = inputs.get("ptm_runtime")
    if not isinstance(supplied, Mapping) or set(supplied) != set(EXPECTED_PTMS):
        raise ManifestError("ptm_runtime must contain exactly R50 and GCViT-Tiny")
    ptms = []
    for checkpoint_id in EXPECTED_PTMS:
        record = records.get(checkpoint_id)
        if not isinstance(record, Mapping):
            raise ManifestError(f"registry checkpoint missing: {checkpoint_id}")
        source_info = record.get("source", {})
        if (
            source_info.get("provider") != "ngc"
            or source_info.get("registry") != "nvidia/tao"
            or source_info.get("official") is not True
        ):
            raise ManifestError(f"PTM is not an official NGC record: {checkpoint_id}")
        supplied_record = supplied[checkpoint_id]
        if not isinstance(supplied_record, Mapping):
            raise ManifestError(f"invalid PTM runtime record: {checkpoint_id}")
        local = _verified_file(
            supplied_record["local_source_path"],
            supplied_record["artifact_sha256"],
            supplied_record["artifact_size_bytes"],
            name=f"PTM {checkpoint_id}",
        )
        if local["size_bytes"] != record["expected_size_bytes"]:
            raise ManifestError(f"registry size mismatch: {checkpoint_id}")
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
                "workflow_id": (
                    "gcvit_tiny" if record["backbone"] == "gc_vit_tiny" else "resnet50"
                ),
                "registry_status_before_qualification": record["status"],
                "registry_record_sha256": canonical_sha256(record),
                "source_identity": source_info["immutable_identity"],
                "backbone": record["backbone"],
                "checkpoint_target": record["checkpoint_target"],
                "artifact": {
                    "sha256": local["sha256"],
                    "size_bytes": local["size_bytes"],
                    "slurm_path": _absolute(
                        supplied_record["slurm_path"],
                        f"{checkpoint_id}.slurm_path",
                        suffix=".pth",
                    ),
                    "local_source_path": local["path"],
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

    slurm_root = _absolute(dataset["slurm_root"], "dataset.slurm_root")
    launcher = HERE / "run_campaign.py"
    manifest = {
        "schema_version": 1,
        "campaign_id": _text(inputs["campaign_id"], "campaign_id"),
        "model": "deformable_detr",
        "task": "object_detection",
        "cpu_runs": 0,
        "smoke_runs": 0,
        "execution": {
            "kind": "direct_full_qualification",
            "cpu_runs": 0,
            "smoke_runs": 0,
            "ministep_runs": 0,
            "local_model_runs": 0,
            "qualification_workflows": 2,
            "parallel_workflows": True,
            "full_training": True,
            "full_in_training_validation": True,
            "standalone_evaluation": True,
            "requires_direct_full_dataset_acknowledgement": True,
            "submission_ready": True,
        },
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
            "slurm_use_sqsh": False,
            "slurm_use_requeue": True,
            "max_infrastructure_retries": 10,
        },
        "dataset": {
            "id": "pascal_voc_2007_full_detection",
            "manifest": manifest_file,
            "integrity": integrity_file,
            "slurm_root": slurm_root,
            "train_image_dir": f"{slurm_root}/VOCdevkit/VOC2007/JPEGImages",
            "validation_image_dir": f"{slurm_root}/VOCdevkit/VOC2007/JPEGImages",
            "train_annotation": (
                f"{slurm_root}/coco/annotations/instances_train2007.json"
            ),
            "train_annotation_sha256": _sha(
                dataset["train_annotation_sha256"],
                "dataset.train_annotation_sha256",
            ),
            "train_annotation_size_bytes": _positive_int(
                dataset["train_annotation_size_bytes"],
                "dataset.train_annotation_size_bytes",
            ),
            "validation_annotation": (
                f"{slurm_root}/coco/annotations/instances_val2007.json"
            ),
            "validation_annotation_sha256": _sha(
                dataset["validation_annotation_sha256"],
                "dataset.validation_annotation_sha256",
            ),
            "validation_annotation_size_bytes": _positive_int(
                dataset["validation_annotation_size_bytes"],
                "dataset.validation_annotation_size_bytes",
            ),
            "image_tree": {
                "algorithm": _text(
                    dataset["image_tree_algorithm"],
                    "dataset.image_tree_algorithm",
                ),
                "sha256": _sha(
                    dataset["image_tree_sha256"], "dataset.image_tree_sha256"
                ),
                "file_count": _positive_int(
                    dataset["image_count"], "dataset.image_count"
                ),
                "total_bytes": _positive_int(
                    dataset["image_total_bytes"], "dataset.image_total_bytes"
                ),
            },
            "num_classes": 21,
            "eval_class_ids": list(range(1, 21)),
        },
        "qualification": dict(qualification),
        "ptms": ptms,
        "failure_policy": {
            "preserve_terminal_failures": True,
            "replace_failed_workflow": False,
            "manual_ptm_substitution": False,
            "maximum_logical_workflows": 2,
            "maximum_train_jobs": 2,
            "maximum_evaluation_jobs": 2,
        },
        "completion_contract": {
            "terminal_artifact_name": "completion.json",
            "terminal_on_partial_or_total_failure": True,
            "workflow_artifact_name": "workflow_completion.json",
        },
        "integrity": {
            "inputs_sha256": sha256_file(DEFAULT_INPUTS),
            "manifest_generator_sha256": sha256_file(Path(__file__)),
            "launcher_sha256": sha256_file(launcher),
            "ptm_registry_sha256": registry.document_sha256,
        },
    }
    return manifest


def seal_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(manifest))
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = canonical_sha(payload)
    return payload


def load_manifest(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = manifest.get("manifest_sha256")
    payload = copy.deepcopy(manifest)
    payload.pop("manifest_sha256", None)
    if expected != canonical_sha(payload):
        raise ManifestError("manifest canonical integrity verification failed")
    if manifest["execution"] != {
        "kind": "direct_full_qualification",
        "cpu_runs": 0,
        "smoke_runs": 0,
        "ministep_runs": 0,
        "local_model_runs": 0,
        "qualification_workflows": 2,
        "parallel_workflows": True,
        "full_training": True,
        "full_in_training_validation": True,
        "standalone_evaluation": True,
        "requires_direct_full_dataset_acknowledgement": True,
        "submission_ready": True,
    }:
        raise ManifestError("direct-full execution contract changed")
    if (
        manifest.get("model") != "deformable_detr"
        or manifest.get("cpu_runs") != 0
        or manifest.get("smoke_runs") != 0
    ):
        raise ManifestError("top-level model/CPU/smoke contract changed")
    if tuple(item["id"] for item in manifest["ptms"]) != EXPECTED_PTMS:
        raise ManifestError("the sealed two-PTM cohort changed")
    if manifest["qualification"]["training_epochs"] != 10:
        raise ManifestError("qualification must remain ten epochs")
    runtime = manifest["runtime"]
    if (
        runtime["nodes"] != 1
        or runtime["tasks_per_node"] != 1
        or runtime["gpus_per_node"] != 8
        or runtime["hardware_contract"] != HARDWARE_CONTRACT
    ):
        raise ManifestError("the one-node/eight-A100 contract changed")
    return manifest


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
