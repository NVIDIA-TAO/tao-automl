#!/usr/bin/env python3

"""Seal the Mask2Former/COCO2017 campaign after immutable prerequisites exist."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from tao_automl.ptm_registry import canonical_sha256

from . import campaign_contract, runtime_overlay


HERE = Path(__file__).resolve().parent
DEFAULT_REPOSITORY = Path("/localhome/local-rarunachalam/tao-automl")
DEFAULT_WHEEL = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/wheels/35972c1/"
    "nvidia_tao_automl-0.1.0-py3-none-any.whl"
)
DEFAULT_SDK = Path(
    "/localhome/local-rarunachalam/.tao/worktrees/"
    "tao-sdk-bounded-self-requeue"
)
DEFAULT_SKILLS = Path(
    "/localhome/local-rarunachalam/.tao/worktrees/"
    "tao-skills-release-7.1.0"
)
DEFAULT_DATASET_MANIFEST = Path(
    "/localhome/local-rarunachalam/.tao/datasets/"
    "cross_model_automl_20260729/manifests/"
    "coco2017_instance_panoptic_v1.FILE_MANIFEST.sha256"
)
DEFAULT_STAGE_MANIFEST = (
    DEFAULT_REPOSITORY
    / "experiments/cross_model_automl_20260729/"
    "segmentation_datasets/dataset_stage_manifest.v1.json"
)
DEFAULT_QUALIFICATION = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask2former_coco2017_ptm_qualification_v3_replay_v1/completion.json"
)
DEFAULT_QUALIFICATION_CONTRACT = Path(
    campaign_contract.FROZEN_V3_QUALIFICATION_CONTRACT["path"]
)
# The already sealed PTM bytes are intentionally reused. Qualification/runtime
# v3 changes only slice-safe checkpointing and same-job continuation.
DEFAULT_PTM_STAGE_MANIFEST = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask2former_coco2017_ptm_qualification_v1/ptm_stage_manifest.json"
)
EXPECTED_DATASET_FILE_MANIFEST_SHA256 = (
    "10566a60498de9998154f44a34445a488c9f030e09f2a7346d20a4a1c55f804e"
)
EXPECTED_STAGE_MANIFEST_SHA256 = (
    "437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d"
)
EXPECTED_WHEEL_SHA256 = (
    "304824dc95ee0ef763ae72f8872e79e593613a36a17e20dbf50b3a561892b381"
)
WHEEL_BUILD_COMMIT = "35972c1bc63e64901c40b0de5be95cc14c19ec80"
EXPECTED_SDK_COMMIT = "1a981d79af40d156735f3d89b98495e7818d0891"
EXPECTED_SKILLS_COMMIT = "2e9c1b25f3c7cb1ae444c75652e36c47eace8229"


class ManifestGenerationError(RuntimeError):
    """The campaign cannot be sealed from the supplied artifacts."""


def _lower_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ManifestGenerationError(f"{name} must be lowercase SHA-256")
    return value


def qualification_evidence_record(
    path: str | Path,
    qualification_contract: str | Path,
) -> dict[str, Any]:
    """Bind terminal v3 evidence and the exact immutable v3 contract."""
    evidence_path = Path(path).resolve()
    contract_path = Path(qualification_contract).resolve()
    frozen = campaign_contract.FROZEN_V3_QUALIFICATION_CONTRACT
    original_evidence = (
        str(evidence_path) == frozen["qualification_evidence_path"]
    )
    if not evidence_path.is_file():
        raise ManifestGenerationError(
            "terminal Mask2Former v3 qualification evidence is unavailable"
        )
    if (
        str(contract_path) != frozen["path"]
        or not contract_path.is_file()
        or campaign_contract.sha256_file(contract_path)
        != frozen["file_sha256"]
    ):
        raise ManifestGenerationError(
            "immutable Mask2Former v3 qualification contract changed"
        )
    try:
        source = json.loads(contract_path.read_text(encoding="utf-8"))
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestGenerationError(
            "Mask2Former v3 qualification JSON is invalid"
        ) from exc
    source_payload = copy.deepcopy(source)
    source_internal = source_payload.pop("contract_sha256", None)
    if (
        source_internal != canonical_sha256(source_payload)
        or source_internal != frozen["contract_sha256"]
        or source.get("runtime", {}).get("source_commit")
        != frozen["source_commit"]
        or source.get("runtime", {}).get("wheel_sha256")
        != frozen["wheel_sha256"]
        or source.get("runtime", {}).get("sdk_commit")
        != frozen["sdk_commit"]
        or source.get("runtime", {}).get("skills_commit")
        != frozen["skills_commit"]
        or source.get("runtime", {}).get("qualification_evidence_path")
        != frozen["qualification_evidence_path"]
        or source.get("runtime", {}).get("ptm_stage_manifest_path")
        != frozen["ptm_stage_manifest_path"]
        or source.get("runtime", {}).get("ptm_stage_manifest_sha256")
        != frozen["ptm_stage_manifest_sha256"]
        or source.get("runtime", {}).get("ptm_stage_content_sha256")
        != frozen["ptm_stage_content_sha256"]
        or source.get("runtime", {}).get("tao_pytorch_overlay")
        != frozen["runtime_overlay"]
        or source.get("runtime", {}).get("walltime_policy")
        != frozen["walltime_policy"]
        or source.get("ptm_inventory", {}).get("registry_version")
        != frozen["registry_version"]
        or source.get("ptm_inventory", {}).get("registry_sha256")
        != frozen["registry_sha256"]
        or source.get("launcher_integrity", {}).get(
            "qualification_campaign_sha256"
        )
        != frozen["qualification_campaign_sha256"]
    ):
        raise ManifestGenerationError(
            "Mask2Former v3 qualification contract identity changed"
        )
    snapshot_records = {
        record["id"]: record["registry_record_sha256"]
        for record in source["ptm_inventory"]["records"]
    }
    expected_records = {
        record["id"]: record["registry_record_sha256"]
        for record in campaign_contract.mask2former_registry_snapshot()[
            "records"
        ]
    }
    if snapshot_records != expected_records:
        raise ManifestGenerationError(
            "Mask2Former v3 qualification record identities changed"
        )

    evidence_payload = copy.deepcopy(evidence)
    evidence_internal = evidence_payload.pop("evidence_sha256", None)
    workflows = evidence.get("workflows")
    expected_ids = tuple(sorted(snapshot_records))
    derivation: dict[str, Any] | None = None
    if original_evidence:
        expected_campaign_id = frozen["qualification_campaign_id"]
        expected_revision = "qualification_runtime_v3"
    else:
        replay = evidence.get("evidence_replay")
        parent = campaign_contract.FROZEN_V3_FAILURE_EVIDENCE
        if (
            not isinstance(replay, dict)
            or replay.get("kind")
            != "immutable_status_metric_deduplication_replay_v1"
            or replay.get("parent_path") != parent["path"]
            or replay.get("parent_file_sha256") != parent["file_sha256"]
            or replay.get("parent_evidence_sha256")
            != parent["evidence_sha256"]
            or replay.get("retraining_jobs_submitted") != 0
            or replay.get("evaluation_jobs_submitted") != 0
            or replay.get("selection_invoked") is not False
            or replay.get("original_evidence_overwritten") is not False
            or not Path(parent["path"]).is_file()
            or campaign_contract.sha256_file(parent["path"])
            != parent["file_sha256"]
        ):
            raise ManifestGenerationError(
                "Mask2Former qualification replay derivation is invalid"
            )
        parent_document = json.loads(
            Path(parent["path"]).read_text(encoding="utf-8")
        )
        if parent_document.get("evidence_sha256") != parent["evidence_sha256"]:
            raise ManifestGenerationError(
                "sealed Mask2Former parent evidence changed"
            )
        expected_campaign_id = (
            "mask2former-coco2017-direct-full-qualification-"
            "v3-replay-v1-20260801"
        )
        expected_revision = "qualification_runtime_v3_evidence_replay_v1"
        derivation = copy.deepcopy(replay)
    if (
        evidence_internal != canonical_sha256(evidence_payload)
        or evidence.get("schema_version") != 1
        or evidence.get("campaign_id") != expected_campaign_id
        or evidence.get("contract_revision") != expected_revision
        or evidence.get("model") != "mask2former"
        or evidence.get("task") != "instance_segmentation"
        or evidence.get("primary_metric") != "segm_val_mAP"
        or evidence.get("standalone_reported_metric") != "segm_test_mAP"
        or evidence.get("qualification_contract_sha256")
        != frozen["contract_sha256"]
        or evidence.get("qualification_campaign_sha256")
        != frozen["qualification_campaign_sha256"]
        or evidence.get("ptm_stage_manifest_path")
        != frozen["ptm_stage_manifest_path"]
        or evidence.get("ptm_stage_manifest_sha256")
        != frozen["ptm_stage_manifest_sha256"]
        or evidence.get("registry_sha256") != frozen["registry_sha256"]
        or evidence.get("sqsh_sha256")
        != campaign_contract.FROZEN_SQSH["sha256"]
        or evidence.get("tao_pytorch_overlay")
        != frozen["runtime_overlay"]
        or evidence.get("walltime_policy") != frozen["walltime_policy"]
        or evidence.get("cpu_model_runs") != 0
        or evidence.get("smoke_model_runs") != 0
        or evidence.get("mini_step_runs") != 0
        or evidence.get("replacement_workflows_submitted") is not False
        or not isinstance(workflows, list)
        or len(workflows) != len(expected_ids)
        or tuple(
            sorted(
                item.get("checkpoint_id")
                for item in workflows
                if isinstance(item, dict)
            )
        )
        != expected_ids
        or any(
            not isinstance(item, dict)
            or item.get("terminal") is not True
            or item.get("status") not in {"success", "failure"}
            or canonical_sha256(
                {
                    key: value
                    for key, value in item.items()
                    if key != "workflow_sha256"
                }
            )
            != item.get("workflow_sha256")
            for item in workflows
        )
    ):
        raise ManifestGenerationError(
            "terminal Mask2Former v3 qualification evidence is invalid"
        )
    _lower_sha256(evidence_internal, "qualification evidence SHA-256")
    return {
        "schema_version": 2,
        "kind": "direct_full_gpu_qualification_runtime_local_v2",
        "enabled": True,
        "scope": "campaign_local_in_memory_projection",
        "model": "mask2former",
        "task": "instance_segmentation",
        "tao_version": "7.1.0",
        "container_sha256": campaign_contract.FROZEN_SQSH["sha256"],
        "base_registry_version": frozen["registry_version"],
        "base_registry_sha256": frozen["registry_sha256"],
        "base_record_sha256_by_checkpoint_id": snapshot_records,
        "qualification_path": str(evidence_path),
        "qualification_file_sha256": campaign_contract.sha256_file(
            evidence_path
        ),
        "qualification_evidence_sha256": evidence_internal,
        "qualification_contract_path": str(contract_path),
        "qualification_contract_file_sha256": frozen["file_sha256"],
        "qualification_contract_sha256": frozen["contract_sha256"],
        "qualification_source_commit": frozen["source_commit"],
        "qualification_source_wheel_sha256": frozen["wheel_sha256"],
        "qualification_source_sdk_commit": frozen["sdk_commit"],
        "qualification_source_skills_commit": frozen["skills_commit"],
        "qualification_campaign_sha256": frozen[
            "qualification_campaign_sha256"
        ],
        "qualification_campaign_id": expected_campaign_id,
        "qualification_contract_revision": expected_revision,
        "qualification_derivation": derivation,
        "ptm_stage_manifest_path": frozen["ptm_stage_manifest_path"],
        "ptm_stage_manifest_sha256": frozen[
            "ptm_stage_manifest_sha256"
        ],
        "ptm_stage_content_sha256": frozen["ptm_stage_content_sha256"],
        "qualification_runtime_overlay": copy.deepcopy(
            frozen["runtime_overlay"]
        ),
        "qualification_walltime_policy": copy.deepcopy(
            frozen["walltime_policy"]
        ),
        "repository_registry_mutation_allowed": False,
        "projection_persisted_as_global_registry": False,
        "failed_arm_promotion_allowed": False,
        "unsupported_arm_promotion_allowed": False,
        "agent_override_allowed": False,
    }


def wait_for_terminal_qualification(
    path: str | Path,
    qualification_contract: str | Path,
    *,
    status_path: str | Path,
    poll_seconds: float = 30.0,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Wait only while v3 completion is absent; reject invalid final data."""
    from . import run_campaign

    evidence_path = Path(path).resolve()
    status = Path(status_path).resolve()
    started = time.monotonic()
    while not evidence_path.is_file():
        run_campaign.atomic_json(
            status,
            {
                "schema_version": 1,
                "automatic_successor": True,
                "state": "waiting_for_terminal_v3_completion",
                "qualification_path": str(evidence_path),
                "model_jobs_launched": False,
                "checked_at_utc": run_campaign.utc_timestamp(),
            },
        )
        if (
            timeout_seconds is not None
            and time.monotonic() - started >= timeout_seconds
        ):
            raise TimeoutError(
                "automatic Mask2Former successor timed out waiting for v3"
            )
        time.sleep(poll_seconds)
    try:
        record = qualification_evidence_record(
            evidence_path,
            qualification_contract,
        )
    except Exception as exc:
        run_campaign.atomic_json(
            status,
            {
                "schema_version": 1,
                "automatic_successor": True,
                "state": "terminal_v3_evidence_rejected",
                "qualification_path": str(evidence_path),
                "model_jobs_launched": False,
                "checked_at_utc": run_campaign.utc_timestamp(),
                "reason": str(exc),
            },
        )
        raise
    run_campaign.atomic_json(
        status,
        {
            "schema_version": 1,
            "automatic_successor": True,
            "state": "terminal_v3_evidence_accepted",
            "qualification_path": str(evidence_path),
            "qualification_file_sha256": record[
                "qualification_file_sha256"
            ],
            "qualification_evidence_sha256": record[
                "qualification_evidence_sha256"
            ],
            "model_jobs_launched": False,
            "checked_at_utc": run_campaign.utc_timestamp(),
        },
    )
    return record


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def dataset_record(
    manifest: str | Path,
    stage_manifest: str | Path = DEFAULT_STAGE_MANIFEST,
) -> dict[str, Any]:
    path = Path(manifest).resolve()
    if (
        not path.is_file()
        or campaign_contract.sha256_file(path)
        != EXPECTED_DATASET_FILE_MANIFEST_SHA256
    ):
        raise ManifestGenerationError(
            "canonical COCO2017 file manifest is unavailable or changed"
        )
    stage_path = Path(stage_manifest).resolve()
    if (
        not stage_path.is_file()
        or campaign_contract.sha256_file(stage_path)
        != EXPECTED_STAGE_MANIFEST_SHA256
    ):
        raise ManifestGenerationError(
            "final segmentation dataset stage manifest is unavailable or changed"
        )
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    execution = stage.get("execution_contract", {})
    validation = stage.get("validation", {})
    coco = stage.get("datasets", {}).get("coco2017", {})
    file_manifest = coco.get("file_manifest", {})
    source_archives = coco.get("source_archives", {})
    if (
        stage.get("schema_version") != 1
        or execution.get("data_only") is not True
        or execution.get("model_invoked") is not False
        or execution.get("cpu_model_smoke_run") is not False
        or execution.get("gpu_model_smoke_run") is not False
        or execution.get("training_run") is not False
        or execution.get("evaluation_run") is not False
        or execution.get("latency_benchmark_run") is not False
        or execution.get("slurm_job_submitted") is not False
        or validation.get("coco2017_subreport_content_sha256")
        != "deced9d6766344fe6fc69cd9de3bcff2cba456a14b3391d07bcedb74c250909e"
        or coco.get("dataset_id")
        != "coco2017_instance_panoptic"
        or coco.get("lustre_root")
        != (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
            "cross_model_automl_20260729/"
            "coco2017_instance_panoptic_v1"
        )
        or coco.get("task_scope", {}).get("mask2former")
        != "instance_segmentation"
        or coco.get("splits", {}).get("train_images") != 118287
        or coco.get("splits", {}).get("val_images") != 5000
        or coco.get("splits", {}).get("train_instance_annotations") != 860001
        or coco.get("splits", {}).get("val_instance_annotations") != 36781
        or coco.get("categories", {}).get("instance_things") != 80
        or coco.get("tao_assets", {}).get("instance_label_map", {}).get(
            "categories"
        )
        != 80
        or coco.get("tao_assets", {}).get("instance_label_map", {}).get(
            "sha256"
        )
        != "67f15c4dd7d52aa73025da8307dec17e907f13db6d5d82332a670f73da68c306"
        or source_archives.get("all_archive_integrity_checks_passed") is not True
        or source_archives.get("annotations_trainval2017.zip", {}).get("sha256")
        != "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
        or source_archives.get(
            "panoptic_annotations_trainval2017.zip", {}
        ).get("sha256")
        != "c05f76d2129b6b561eb70efe16e7006df62f73fb92889132d373b9d90e31a370"
        or source_archives.get("train2017.zip", {}).get("sha256")
        != "69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929"
        or source_archives.get("val2017.zip", {}).get("sha256")
        != "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05"
        or file_manifest.get("entries") != 246593
        or file_manifest.get("sha256")
        != EXPECTED_DATASET_FILE_MANIFEST_SHA256
        or file_manifest.get("remote_sha256sum_check") != "passed"
        or file_manifest.get("remote_file_set_check") != "passed"
        or coco.get("remote_read_only") is not True
        or coco.get("remote_writable_entries_after_lock") != 0
        or stage.get("transfer_provenance", {}).get(
            "remote_bytes_verified_against_local_manifest"
        )
        is not True
    ):
        raise ManifestGenerationError(
            "final COCO2017 stage provenance does not pass the frozen contract"
        )
    return {
        "id": "coco2017_full_instance_segmentation",
        "official_sources": {
            name: item["url"] for name, item in source_archives.items()
            if isinstance(item, dict) and "url" in item
        },
        "license": "Creative Commons Attribution 4.0 for COCO annotations",
        "prepared_root": (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
            "cross_model_automl_20260729/"
            "coco2017_instance_panoptic_v1"
        ),
        "train_image_count": 118287,
        "validation_image_count": 5000,
        "train_instance_annotations": 860001,
        "validation_instance_annotations": 36781,
        "num_classes": 80,
        "train_instance_json_sha256": (
            "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d"
        ),
        "validation_instance_json_sha256": (
            "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
        ),
        "label_map_sha256": (
            "67f15c4dd7d52aa73025da8307dec17e907f13db6d5d82332a670f73da68c306"
        ),
        "official_archive_sha256": {
            name: item["sha256"] for name, item in source_archives.items()
            if isinstance(item, dict) and "sha256" in item
        },
        "content_sha256": (
            "deced9d6766344fe6fc69cd9de3bcff2cba456a14b3391d07bcedb74c250909e"
        ),
        "manifest_path": str(path),
        "manifest_sha256": EXPECTED_DATASET_FILE_MANIFEST_SHA256,
        "file_manifest_entry_count": 246593,
        "remote_sha256sum_check": "passed_all_246593",
        "stage_manifest_path": str(stage_path),
        "stage_manifest_lustre_path": (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
            "cross_model_automl_20260729/coco2017_instance_panoptic_v1/"
            "dataset_stage_manifest.v1.json"
        ),
        "stage_manifest_sha256": EXPECTED_STAGE_MANIFEST_SHA256,
        "remote_file_manifest_path": file_manifest["lustre_path"],
        "remote_read_only": True,
        "remote_writable_entries_after_lock": 0,
    }


def ptm_stage_record(path: str | Path) -> dict[str, Any]:
    """Validate and bind the immutable official Mask2Former PTM stage."""
    stage_path = Path(path).resolve()
    if not stage_path.is_file():
        raise ManifestGenerationError(
            f"PTM stage manifest is unavailable: {stage_path}"
        )
    document = json.loads(stage_path.read_text(encoding="utf-8"))
    supplied = document.get("manifest_sha256")
    payload = copy.deepcopy(document)
    payload.pop("manifest_sha256", None)
    snapshot = campaign_contract.mask2former_registry_snapshot()
    records = document.get("checkpoints")
    if (
        supplied != canonical_sha256(payload)
        or document.get("schema_version") != 1
        or document.get("model") != "mask2former"
        or not isinstance(document.get("registry_sha256"), str)
        or len(document["registry_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in document["registry_sha256"]
        )
        or document.get("stage_complete") is not True
        or document.get("remote_read_only") is not True
        or document.get("cpu_model_runs") != 0
        or document.get("gpu_model_runs") != 0
        or document.get("smoke_model_runs") != 0
        or document.get("mini_step_runs") != 0
        or document.get("scheduler_jobs_submitted") != 0
        or not isinstance(records, list)
    ):
        raise ManifestGenerationError(
            "official Mask2Former PTM stage is incomplete or changed"
        )
    by_id = {
        item.get("id"): item
        for item in records
        if isinstance(item, dict)
    }
    expected = {
        item["id"]: item for item in snapshot["records"]
    }
    if set(by_id) != set(expected) or len(records) != len(expected):
        raise ManifestGenerationError(
            "PTM stage must contain every and only official Mask2Former arm"
        )
    for checkpoint_id, registry_record in expected.items():
        item = by_id[checkpoint_id]
        digest = item.get("sha256")
        if (
            not isinstance(item.get("path"), str)
            or not item["path"].startswith("/lustre/")
            or isinstance(item.get("size_bytes"), bool)
            or item.get("size_bytes")
            != registry_record["expected_size_bytes"]
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or item.get("immutable_source_identity")
            != registry_record["source"]["immutable_identity"]
            or (
                registry_record.get("sha256") is not None
                and digest != registry_record["sha256"]
            )
            or item.get("remote_read_only") is not True
        ):
            raise ManifestGenerationError(
                f"official PTM stage record changed: {checkpoint_id}"
            )
    return {
        "path": str(stage_path),
        "sha256": campaign_contract.sha256_file(stage_path),
        "manifest_sha256": supplied,
        "checkpoint_ids": sorted(by_id),
        "checkpoints": copy.deepcopy(records),
    }


def _runtime(
    *,
    repository: Path,
    wheel: Path,
    sdk: Path,
    skills: Path,
    qualification: Path,
    qualification_contract: Path,
    ptm_stage_manifest: Path,
) -> dict[str, Any]:
    if not wheel.is_file() or (
        campaign_contract.sha256_file(wheel) != EXPECTED_WHEEL_SHA256
    ):
        raise ManifestGenerationError("production AutoML wheel changed")
    if _git(sdk, "rev-parse", "HEAD") != EXPECTED_SDK_COMMIT:
        raise ManifestGenerationError("TAO SDK commit changed")
    if _git(skills, "rev-parse", "HEAD") != EXPECTED_SKILLS_COMMIT:
        raise ManifestGenerationError("TAO skills commit changed")
    if _git(repository, "status", "--porcelain"):
        raise ManifestGenerationError(
            "AutoML source must be clean before campaign sealing"
        )
    ptm_stage = ptm_stage_record(ptm_stage_manifest)
    head = _git(repository, "rev-parse", "HEAD")
    runtime_local_eligibility = qualification_evidence_record(
        qualification,
        qualification_contract,
    )
    runtime_local_eligibility.update(
        {
            "eligibility_source_commit": head,
            "wheel_sha256": EXPECTED_WHEEL_SHA256,
            "sdk_commit": EXPECTED_SDK_COMMIT,
            "skills_commit": EXPECTED_SKILLS_COMMIT,
        }
    )
    return {
        "repository": str(repository.resolve()),
        "source_commit": head,
        "source_dirty": False,
        "wheel_path": str(wheel.resolve()),
        "wheel_sha256": EXPECTED_WHEEL_SHA256,
        "wheel_build_commit": WHEEL_BUILD_COMMIT,
        "sdk_dir": str(sdk.resolve()),
        "sdk_commit": EXPECTED_SDK_COMMIT,
        "skills_repository": str(skills.resolve()),
        "skills_commit": EXPECTED_SKILLS_COMMIT,
        "skill_dir": str(
            (
                skills
                / "skills/models/tao-train-mask2former"
            ).resolve()
        ),
        "qualification_evidence_path": str(qualification.resolve()),
        "runtime_local_eligibility": runtime_local_eligibility,
        "ptm_stage_manifest_path": ptm_stage["path"],
        "ptm_stage_manifest_sha256": ptm_stage["sha256"],
        "ptm_stage_content_sha256": ptm_stage["manifest_sha256"],
        "tao_pytorch_overlay": runtime_overlay.successor_contract_record(),
        "partition": campaign_contract.FROZEN_SLURM_PARTITION,
        "account": "edgeai_tao-ptm_image-foundation-model-clip",
        "base_results_dir": (
            "/lustre/fsw/portfolios/edgeai/projects/"
            "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam"
        ),
        "container_mounts": "/lustre",
        "time_hours": campaign_contract.FROZEN_SLURM_TIME_HOURS,
        "timeout_hours": campaign_contract.FROZEN_SLURM_TIMEOUT_HOURS,
        "use_requeue": campaign_contract.FROZEN_SLURM_USE_REQUEUE,
        "walltime_policy": copy.deepcopy(
            campaign_contract.SUCCESSOR_WALLTIME_POLICY
        ),
        "max_job_retries": campaign_contract.FROZEN_SLURM_RETRY_CAP,
        "hardware_contract": copy.deepcopy(
            campaign_contract.FROZEN_HARDWARE
        ),
    }


def build_contract(
    *,
    repository: str | Path = DEFAULT_REPOSITORY,
    wheel: str | Path = DEFAULT_WHEEL,
    sdk: str | Path = DEFAULT_SDK,
    skills: str | Path = DEFAULT_SKILLS,
    dataset_manifest: str | Path = DEFAULT_DATASET_MANIFEST,
    stage_manifest: str | Path = DEFAULT_STAGE_MANIFEST,
    qualification: str | Path = DEFAULT_QUALIFICATION,
    qualification_contract: str | Path = DEFAULT_QUALIFICATION_CONTRACT,
    ptm_stage_manifest: str | Path = DEFAULT_PTM_STAGE_MANIFEST,
) -> dict[str, Any]:
    repository_path = Path(repository).resolve()
    value = campaign_contract.build_preregistered_contract(
        campaign_id=(
            "mask2former-coco2017-objective-aware-three-mode-v5-20260801"
        ),
        dataset=dataset_record(dataset_manifest, stage_manifest),
        skill_dir=(
            Path(skills).resolve()
            / "skills/models/tao-train-mask2former"
        ),
        runtime=_runtime(
            repository=repository_path,
            wheel=Path(wheel).resolve(),
            sdk=Path(sdk).resolve(),
            skills=Path(skills).resolve(),
            qualification=Path(qualification),
            qualification_contract=Path(qualification_contract),
            ptm_stage_manifest=Path(ptm_stage_manifest),
        ),
    )
    value.pop("contract_sha256")
    value["launcher_integrity"] = {
        "campaign_contract_sha256": campaign_contract.sha256_file(
            HERE / "campaign_contract.py"
        ),
        "qualification_gate_sha256": campaign_contract.sha256_file(
            HERE / "qualification_gate.py"
        ),
        "qualification_campaign_sha256": campaign_contract.sha256_file(
            HERE / "qualification_campaign.py"
        ),
        "run_campaign_sha256": campaign_contract.sha256_file(
            HERE / "run_campaign.py"
        ),
        "mask2former_latency_worker_sha256": (
            campaign_contract.sha256_file(
                HERE / "mask2former_latency_worker.py"
            )
        ),
        "runtime_overlay_sha256": campaign_contract.sha256_file(
            HERE / "runtime_overlay.py"
        ),
        "checkpoint_resume_sha256": campaign_contract.sha256_file(
            HERE / "checkpoint_resume.py"
        ),
        "manifest_generator_sha256": campaign_contract.sha256_file(
            HERE / "manifest_generator.py"
        ),
    }
    value["contract_sha256"] = canonical_sha256(value)
    return campaign_contract.validate_contract(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument("--wheel", type=Path, default=DEFAULT_WHEEL)
    parser.add_argument("--sdk", type=Path, default=DEFAULT_SDK)
    parser.add_argument("--skills", type=Path, default=DEFAULT_SKILLS)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument(
        "--stage-manifest",
        type=Path,
        default=DEFAULT_STAGE_MANIFEST,
    )
    parser.add_argument(
        "--qualification", type=Path, default=DEFAULT_QUALIFICATION
    )
    parser.add_argument(
        "--qualification-contract",
        type=Path,
        default=DEFAULT_QUALIFICATION_CONTRACT,
    )
    parser.add_argument(
        "--ptm-stage-manifest",
        type=Path,
        default=DEFAULT_PTM_STAGE_MANIFEST,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--automatic-trigger", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path(
            "/localhome/local-rarunachalam/.tao/artifacts/"
            "cross_model_automl_20260729/"
            "mask2former_coco2017_three_mode_v5"
        ),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("/localhome/local-rarunachalam/.tao/config.env"),
    )
    args = parser.parse_args(argv)
    if args.launch and not args.automatic_trigger:
        raise ManifestGenerationError(
            "automatic successor launch requires --automatic-trigger"
        )
    if args.automatic_trigger:
        wait_for_terminal_qualification(
            args.qualification,
            args.qualification_contract,
            status_path=(
                args.runtime_root / "automatic_successor_status.json"
            ),
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
    contract = build_contract(
        repository=args.repository,
        wheel=args.wheel,
        sdk=args.sdk,
        skills=args.skills,
        dataset_manifest=args.dataset_manifest,
        stage_manifest=args.stage_manifest,
        qualification=args.qualification,
        qualification_contract=args.qualification_contract,
        ptm_stage_manifest=args.ptm_stage_manifest,
    )
    from . import run_campaign

    if args.output.is_file():
        try:
            existing = campaign_contract.validate_contract(
                json.loads(args.output.read_text(encoding="utf-8"))
            )
        except Exception as exc:
            raise ManifestGenerationError(
                "existing successor contract is invalid"
            ) from exc
        if existing != contract:
            raise ManifestGenerationError(
                "existing successor contract differs; refusing overwrite"
            )
    else:
        run_campaign.atomic_json(args.output, contract)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "contract_sha256": contract["contract_sha256"],
                "launch_authorized": False,
                "reason": (
                    "the exact terminal v3 evidence and its campaign-local "
                    "registry projection are evaluated by the automatic trigger"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not args.launch:
        return 0
    runner_arguments = [
        "--contract",
        str(args.output.resolve()),
        "--runtime-root",
        str(args.runtime_root.resolve()),
        "--env-file",
        str(args.env_file.resolve()),
        "--automatic-trigger",
        "--launch",
        "--poll-seconds",
        str(args.poll_seconds),
    ]
    if args.resume:
        runner_arguments.append("--resume")
    return run_campaign.main(runner_arguments)


if __name__ == "__main__":
    raise SystemExit(main())
