#!/usr/bin/env python3

"""Seal the OneFormer/full-COCO2017 campaign from immutable inputs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any

from tao_automl.ptm_registry import canonical_sha256, load_ptm_registry

from . import campaign_contract, ptm_stage


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
    "/localhome/local-rarunachalam/.tao/worktrees/tao-skills-release-7.1.0"
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
    "oneformer_coco2017_ptm_qualification_v4/completion.json"
)
DEFAULT_QUALIFICATION_CONTRACT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "oneformer_coco2017_ptm_qualification_v4/qualification.v4.json"
)
DEFAULT_PTM_STAGE_MANIFEST = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "oneformer_coco2017_ptm_qualification_v1/ptm_stage_manifest.json"
)
DEFAULT_RUNTIME_OVERLAY = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "oneformer-runtime-product-fixes-1752ec2c/"
    "oneformer-runtime-overlay.v2.tar"
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
WHEEL_BUILD_COMMIT = (
    "35972c1bc63e64901c40b0de5be95cc14c19ec80"
)
EXPECTED_WHEEL_REGISTRY_FILE_SHA256 = (
    "c28831814cabcd676909260ab347c6f756294089371be61e941e2de69400a725"
)
EXPECTED_WHEEL_ONEFORMER_REGISTRY_SHA256 = (
    "3872bec8c0e58f79cd2f941d18bcc1bcb5660ed90c9a4500f8ab5cf3004bde2a"
)
QUALIFICATION_SDK_COMMIT = "1a981d79af40d156735f3d89b98495e7818d0891"
EXPECTED_SDK_COMMIT = "98c1144fd57b28f38ab5b7b41c113fac6e5e670a"
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


def _successor_qualification_evidence_record(
    evidence_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    """Bind the selective v4 recovery and its preserved v3 predecessor."""
    try:
        source = json.loads(contract_path.read_text(encoding="utf-8"))
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestGenerationError(
            "OneFormer v4 qualification JSON is invalid"
        ) from exc
    source_payload = copy.deepcopy(source)
    source_internal = source_payload.pop("contract_sha256", None)
    policy = source.get("qualification_policy", {})
    runtime = source.get("runtime", {})
    predecessor = policy.get("predecessor_evidence", {})
    expected_recovery = [
        "oneformer.its.commercial.dinat_large.trainable"
    ]
    expected_reused = {
        "oneformer.ade20k.research.swin_large.trainable.v1.0",
        "oneformer.coco.research.swin_large.trainable",
        "oneformer.its.commercial.swin_large.trainable.v1.0",
    }
    if (
        source_internal != canonical_sha256(source_payload)
        or source.get("campaign_id")
        != "oneformer-coco2017-direct-full-ptm-qualification-v4-20260801"
        or source.get("model") != "oneformer"
        or source.get("task") != "panoptic_segmentation"
        or source.get("sqsh") != campaign_contract.FROZEN_SQSH
        or policy.get("version") != 4
        or policy.get("qualification_campaign_id")
        != source.get("campaign_id")
        or policy.get("qualification_evidence_path")
        != str(evidence_path)
        or runtime.get("qualification_evidence_path")
        != str(evidence_path)
        or runtime.get("source_commit")
        != "35972c1bc63e64901c40b0de5be95cc14c19ec80"
        or runtime.get("wheel_sha256") != EXPECTED_WHEEL_SHA256
        or runtime.get("sdk_commit") != QUALIFICATION_SDK_COMMIT
        or runtime.get("skills_commit") != EXPECTED_SKILLS_COMMIT
        or policy.get("checkpoint_resume_policy")
        != campaign_contract.CHECKPOINT_RESUME_POLICY
        or policy.get("recovery_checkpoint_ids") != expected_recovery
        or set(policy.get("reused_checkpoint_ids", [])) != expected_reused
        or predecessor.get("failed_workflow_preserved") is not True
        or predecessor.get("recovery_checkpoint_ids") != expected_recovery
        or set(predecessor.get("reused_success_checkpoint_ids", []))
        != expected_reused
    ):
        raise ManifestGenerationError(
            "OneFormer v4 qualification contract identity changed"
        )
    snapshot = campaign_contract.oneformer_registry_snapshot()
    snapshot_records = {
        record["id"]: record["registry_record_sha256"]
        for record in source["ptm_inventory"]["records"]
    }
    expected_records = {
        record["id"]: record["registry_record_sha256"]
        for record in snapshot["records"]
    }
    if snapshot_records != expected_records:
        raise ManifestGenerationError(
            "OneFormer v4 qualification PTM identities changed"
        )
    evidence_payload = copy.deepcopy(evidence)
    evidence_internal = evidence_payload.pop("evidence_sha256", None)
    workflows = evidence.get("workflows")
    if (
        evidence_internal != canonical_sha256(evidence_payload)
        or evidence.get("campaign_id") != source.get("campaign_id")
        or evidence.get("qualification_contract_sha256") != source_internal
        or evidence.get("qualification_campaign_sha256")
        != source.get("launcher_integrity", {}).get(
            "qualification_campaign_sha256"
        )
        or evidence.get("registry_sha256") != snapshot["registry_sha256"]
        or evidence.get("sqsh_sha256")
        != campaign_contract.FROZEN_SQSH["sha256"]
        or evidence.get("runtime_overlay_sha256")
        != campaign_contract.FROZEN_RUNTIME_OVERLAY["archive_sha256"]
        or evidence.get("ptm_stage_manifest_sha256")
        != runtime.get("ptm_stage_manifest_sha256")
        or evidence.get("ptm_stage_content_sha256")
        != runtime.get("ptm_stage_content_sha256")
        or evidence.get("replacement_workflows_submitted") is not True
        or evidence.get("replacement_workflow_count") != 1
        or evidence.get("recovery_checkpoint_ids") != expected_recovery
        or set(evidence.get("reused_predecessor_checkpoint_ids", []))
        != expected_reused
        or evidence.get("predecessor_evidence") != predecessor
        or evidence.get("checkpoint_resume_policy")
        != campaign_contract.CHECKPOINT_RESUME_POLICY
        or any(evidence.get(name) != 0 for name in (
            "cpu_model_runs", "smoke_model_runs", "mini_step_runs"
        ))
        or not isinstance(workflows, list)
        or len(workflows) != 4
        or {item.get("checkpoint_id") for item in workflows}
        != set(expected_records)
        or any(
            not isinstance(item, dict)
            or item.get("terminal") is not True
            or item.get("status") not in {"success", "failure"}
            or canonical_sha256(
                {key: value for key, value in item.items()
                 if key != "workflow_sha256"}
            ) != item.get("workflow_sha256")
            for item in workflows
        )
    ):
        raise ManifestGenerationError(
            "terminal OneFormer v4 qualification evidence is invalid"
        )
    for name, value in (
        ("qualification evidence SHA-256", evidence_internal),
        ("qualification contract SHA-256", source_internal),
    ):
        _lower_sha256(value, name)
    return {
        "schema_version": 2,
        "kind": "direct_full_gpu_qualification_runtime_local_v2",
        "enabled": True,
        "scope": "campaign_local_in_memory_projection",
        "model": "oneformer",
        "task": "panoptic_segmentation",
        "tao_version": "7.1.0",
        "container_sha256": campaign_contract.FROZEN_SQSH["sha256"],
        "base_registry_version": snapshot["registry_version"],
        "base_registry_sha256": snapshot["registry_sha256"],
        "base_record_sha256_by_checkpoint_id": snapshot_records,
        "qualification_path": str(evidence_path),
        "qualification_file_sha256": campaign_contract.sha256_file(
            evidence_path
        ),
        "qualification_evidence_sha256": evidence_internal,
        "qualification_contract_path": str(contract_path),
        "qualification_contract_file_sha256": (
            campaign_contract.sha256_file(contract_path)
        ),
        "qualification_contract_sha256": source_internal,
        "qualification_source_commit": runtime["source_commit"],
        "qualification_source_wheel_sha256": runtime["wheel_sha256"],
        "qualification_source_sdk_commit": runtime["sdk_commit"],
        "qualification_source_skills_commit": runtime["skills_commit"],
        "qualification_campaign_sha256": source["launcher_integrity"][
            "qualification_campaign_sha256"
        ],
        "qualification_campaign_id": source["campaign_id"],
        "ptm_stage_manifest_path": runtime["ptm_stage_manifest_path"],
        "ptm_stage_manifest_sha256": runtime[
            "ptm_stage_manifest_sha256"
        ],
        "ptm_stage_content_sha256": runtime["ptm_stage_content_sha256"],
        "qualification_successor_version": 4,
        "replacement_workflows_submitted": True,
        "replacement_workflow_count": 1,
        "recovery_checkpoint_ids": expected_recovery,
        "reused_predecessor_checkpoint_ids": sorted(expected_reused),
        "predecessor_evidence": copy.deepcopy(predecessor),
        "repository_registry_mutation_allowed": False,
        "projection_persisted_as_global_registry": False,
        "failed_arm_promotion_allowed": False,
        "unsupported_arm_promotion_allowed": False,
        "agent_override_allowed": False,
    }


def qualification_evidence_record(
    path: str | Path,
    qualification_contract: str | Path,
) -> dict[str, Any]:
    """Bind terminal v3 evidence and its exact immutable input contract."""
    evidence_path = Path(path).resolve()
    contract_path = Path(qualification_contract).resolve()
    frozen = campaign_contract.FROZEN_V3_QUALIFICATION_CONTRACT
    if str(contract_path) != frozen["path"]:
        if not evidence_path.is_file() or not contract_path.is_file():
            raise ManifestGenerationError(
                "terminal OneFormer v4 qualification evidence is unavailable"
            )
        return _successor_qualification_evidence_record(
            evidence_path, contract_path
        )
    if (
        str(evidence_path) != frozen["qualification_evidence_path"]
        or not evidence_path.is_file()
    ):
        raise ManifestGenerationError(
            "terminal OneFormer v3 qualification evidence is unavailable"
        )
    if (
        str(contract_path) != frozen["path"]
        or not contract_path.is_file()
        or campaign_contract.sha256_file(contract_path)
        != frozen["file_sha256"]
    ):
        raise ManifestGenerationError(
            "immutable OneFormer v3 qualification contract changed"
        )
    try:
        source_contract = json.loads(
            contract_path.read_text(encoding="utf-8")
        )
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestGenerationError(
            "OneFormer v3 qualification JSON is invalid"
        ) from exc
    source_payload = copy.deepcopy(source_contract)
    source_internal = source_payload.pop("contract_sha256", None)
    if (
        source_internal != canonical_sha256(source_payload)
        or source_internal != frozen["contract_sha256"]
        or source_contract.get("runtime", {}).get("source_commit")
        != frozen["source_commit"]
        or source_contract.get("runtime", {}).get("wheel_sha256")
        != frozen["wheel_sha256"]
        or source_contract.get("runtime", {}).get("sdk_commit")
        != frozen["sdk_commit"]
        or source_contract.get("runtime", {}).get("skills_commit")
        != frozen["skills_commit"]
        or source_contract.get("ptm_inventory", {}).get("registry_version")
        != frozen["registry_version"]
        or source_contract.get("ptm_inventory", {}).get("registry_sha256")
        != frozen["registry_sha256"]
        or source_contract.get("launcher_integrity", {}).get(
            "qualification_campaign_sha256"
        )
        != frozen["qualification_campaign_sha256"]
        or source_contract.get("qualification_policy", {}).get(
            "qualification_evidence_path"
        )
        != frozen["qualification_evidence_path"]
        or source_contract.get("qualification_policy", {}).get(
            "ptm_stage_manifest_path"
        )
        != frozen["ptm_stage_manifest_path"]
        or source_contract.get("qualification_policy", {}).get(
            "ptm_stage_manifest_sha256"
        )
        != frozen["ptm_stage_manifest_sha256"]
        or source_contract.get("qualification_policy", {}).get(
            "ptm_stage_content_sha256"
        )
        != frozen["ptm_stage_content_sha256"]
    ):
        raise ManifestGenerationError(
            "OneFormer v3 qualification contract identity changed"
        )
    snapshot_records = {
        record["id"]: record["registry_record_sha256"]
        for record in source_contract["ptm_inventory"]["records"]
    }
    expected_records = {
        record["id"]: record["registry_record_sha256"]
        for record in campaign_contract.oneformer_registry_snapshot()[
            "records"
        ]
    }
    if snapshot_records != expected_records:
        raise ManifestGenerationError(
            "OneFormer v3 qualification record identities changed"
        )

    evidence_payload = copy.deepcopy(evidence)
    evidence_internal = evidence_payload.pop("evidence_sha256", None)
    workflows = evidence.get("workflows")
    expected_ids = tuple(sorted(snapshot_records))
    if (
        evidence_internal != canonical_sha256(evidence_payload)
        or evidence.get("schema_version") != 1
        or evidence.get("campaign_id") != frozen["qualification_campaign_id"]
        or evidence.get("model") != "oneformer"
        or evidence.get("task") != "panoptic_segmentation"
        or evidence.get("metric") != "PQ"
        or evidence.get("qualification_contract_sha256")
        != frozen["contract_sha256"]
        or evidence.get("qualification_campaign_sha256")
        != frozen["qualification_campaign_sha256"]
        or evidence.get("registry_sha256") != frozen["registry_sha256"]
        or evidence.get("sqsh_sha256")
        != campaign_contract.FROZEN_SQSH["sha256"]
        or evidence.get("runtime_overlay_sha256")
        != campaign_contract.FROZEN_RUNTIME_OVERLAY["archive_sha256"]
        or evidence.get("runtime_overlay_source_commit")
        != campaign_contract.FROZEN_RUNTIME_OVERLAY["source_commit"]
        or evidence.get("ptm_stage_manifest_sha256")
        != frozen["ptm_stage_manifest_sha256"]
        or evidence.get("ptm_stage_content_sha256")
        != frozen["ptm_stage_content_sha256"]
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
            "terminal OneFormer v3 qualification evidence is invalid"
        )
    _lower_sha256(evidence_internal, "qualification evidence SHA-256")
    return {
        "schema_version": 2,
        "kind": "direct_full_gpu_qualification_runtime_local_v2",
        "enabled": True,
        "scope": "campaign_local_in_memory_projection",
        "model": "oneformer",
        "task": "panoptic_segmentation",
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
        "qualification_campaign_id": frozen["qualification_campaign_id"],
        "ptm_stage_manifest_path": frozen["ptm_stage_manifest_path"],
        "ptm_stage_manifest_sha256": frozen[
            "ptm_stage_manifest_sha256"
        ],
        "ptm_stage_content_sha256": frozen["ptm_stage_content_sha256"],
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
    """Wait only while v3 completion is absent; invalid evidence is final."""
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
                "automatic OneFormer successor timed out waiting for v3"
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


def runtime_overlay_record(path: str | Path) -> dict[str, Any]:
    """Validate the reviewed TAO PyTorch overlay without installing it."""
    archive = Path(path).resolve()
    frozen = campaign_contract.FROZEN_RUNTIME_OVERLAY
    if (
        not archive.is_file()
        or archive.stat().st_size != frozen["archive_size_bytes"]
        or campaign_contract.sha256_file(archive) != frozen["archive_sha256"]
    ):
        raise ManifestGenerationError(
            "reviewed OneFormer runtime-overlay archive is unavailable or changed"
        )
    manifest_member = (
        f"{frozen['archive_root']}/MANIFEST.json"
    )
    installer_member = (
        f"{frozen['archive_root']}/install_overlay.py"
    )
    try:
        with tarfile.open(archive, "r") as bundle:
            members = {member.name: member for member in bundle.getmembers()}
            manifest_bytes = bundle.extractfile(members[manifest_member]).read()
            installer_bytes = bundle.extractfile(members[installer_member]).read()
            manifest = json.loads(manifest_bytes)
    except (KeyError, OSError, tarfile.TarError, ValueError) as exc:
        raise ManifestGenerationError(
            "reviewed OneFormer runtime-overlay archive is invalid"
        ) from exc
    if (
        hashlib.sha256(manifest_bytes).hexdigest()
        != frozen["manifest_sha256"]
        or hashlib.sha256(installer_bytes).hexdigest()
        != frozen["installer_sha256"]
        or manifest.get("schema_version")
        != frozen["manifest_schema_version"]
        or manifest.get("artifact_type") != frozen["artifact_type"]
        or manifest.get("scope") != frozen["scope"]
        or manifest.get("source", {}).get("commit")
        != frozen["source_commit"]
        or manifest.get("source", {}).get("base_commit")
        != frozen["base_commit"]
        or manifest.get("container", {}).get("sha256")
        != campaign_contract.FROZEN_SQSH["sha256"]
        or manifest.get("container", {}).get("site_packages")
        != frozen["base_site_packages"]
        or manifest.get("runtime_contract", {}).get(
            "panoptic_primary_metric"
        )
        != "PQ"
        or manifest.get("runtime_contract", {}).get("base_audit_root")
        != frozen["base_site_packages"]
        or manifest.get("runtime_contract", {}).get("overlay_output_root")
        != "ephemeral_pythonpath_site_packages"
        or len(manifest.get("files", ())) != frozen["file_count"]
    ):
        raise ManifestGenerationError(
            "reviewed OneFormer runtime-overlay manifest changed"
        )
    return {
        "local_archive_path": str(archive),
        "archive_sha256": frozen["archive_sha256"],
        "archive_size_bytes": frozen["archive_size_bytes"],
        "manifest_sha256": frozen["manifest_sha256"],
        "installer_sha256": frozen["installer_sha256"],
        "source_commit": frozen["source_commit"],
        "file_count": frozen["file_count"],
    }


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
            "final segmentation stage manifest is unavailable or changed"
        )
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    execution = stage.get("execution_contract", {})
    validation = stage.get("validation", {})
    coco = stage.get("datasets", {}).get("coco2017", {})
    file_manifest = coco.get("file_manifest", {})
    splits = coco.get("splits", {})
    categories = coco.get("categories", {})
    assets = coco.get("tao_assets", {})
    sources = coco.get("source_archives", {})
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
        or coco.get("dataset_id") != "coco2017_instance_panoptic"
        or coco.get("lustre_root")
        != (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
            "cross_model_automl_20260729/coco2017_instance_panoptic_v1"
        )
        or splits.get("train_images") != 118287
        or splits.get("val_images") != 5000
        or splits.get("train_panoptic_pngs") != 118287
        or splits.get("val_panoptic_pngs") != 5000
        or splits.get("train_panoptic_segments") != 1329984
        or splits.get("val_panoptic_segments") != 56728
        or categories.get("panoptic_total") != 133
        or assets.get("panoptic_label_map", {}).get("sha256")
        != "4b28b3773f0f8e63d836dc20da77276633da72178453458b79e32be8e892ce56"
        or assets.get("instance_label_map", {}).get("sha256")
        != "67f15c4dd7d52aa73025da8307dec17e907f13db6d5d82332a670f73da68c306"
        or sources.get("all_archive_integrity_checks_passed") is not True
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
    root = coco["lustre_root"]
    return {
        "id": "coco_2017_full_instance_panoptic",
        "official_source": "https://cocodataset.org/",
        "license": (
            "COCO annotations CC BY 4.0; source images retain individual licenses"
        ),
        "root": root,
        "train_image_count": 118287,
        "validation_image_count": 5000,
        "train_panoptic_png_count": 118287,
        "validation_panoptic_png_count": 5000,
        "train_panoptic_segment_count": 1329984,
        "validation_panoptic_segment_count": 56728,
        "panoptic_category_count": 133,
        "instance_category_count": 80,
        "panoptic_label_map_sha256": (
            "4b28b3773f0f8e63d836dc20da77276633da72178453458b79e32be8e892ce56"
        ),
        "instance_label_map_sha256": (
            "67f15c4dd7d52aa73025da8307dec17e907f13db6d5d82332a670f73da68c306"
        ),
        "train_panoptic_json_sha256": (
            "560a90a275c65b089d4944fbd8d44d04c57d2e36bf7f66597f367cc4a42bfbbb"
        ),
        "validation_panoptic_json_sha256": (
            "454873a8a01114246066ac841750eb742df3b5e42ce927ef38b49690084ec75a"
        ),
        "content_sha256": (
            "deced9d6766344fe6fc69cd9de3bcff2cba456a14b3391d07bcedb74c250909e"
        ),
        "manifest_path": str(path),
        "manifest_sha256": EXPECTED_DATASET_FILE_MANIFEST_SHA256,
        "file_manifest_entry_count": 246593,
        "remote_sha256sum_check": "passed_all_246593",
        "stage_manifest_path": str(stage_path),
        "stage_manifest_lustre_path": (
            f"{root}/dataset_stage_manifest.v1.json"
        ),
        "stage_manifest_sha256": EXPECTED_STAGE_MANIFEST_SHA256,
        "remote_file_manifest_path": file_manifest["lustre_path"],
        "remote_read_only": True,
        "remote_writable_entries_after_lock": 0,
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
    runtime_overlay: Path,
) -> dict[str, Any]:
    if (
        not wheel.is_file()
        or campaign_contract.sha256_file(wheel) != EXPECTED_WHEEL_SHA256
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
    head = _git(repository, "rev-parse", "HEAD")
    registry_path = repository / "src/tao_automl/data/ptm_registry.v1.json"
    try:
        source_registry = json.loads(registry_path.read_text(encoding="utf-8"))
        oneformer_sha = canonical_sha256(
            source_registry["models"]["oneformer"]
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ManifestGenerationError(
            "campaign source lacks the wheel's OneFormer registry"
        ) from exc
    if oneformer_sha != EXPECTED_WHEEL_ONEFORMER_REGISTRY_SHA256:
        raise ManifestGenerationError(
            "campaign source does not match the wheel's OneFormer inventory"
        )
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
    overlay = runtime_overlay_record(runtime_overlay)
    if (
        not ptm_stage_manifest.is_file()
        or ptm_stage_manifest.is_symlink()
        or ptm_stage_manifest.stat().st_mode & 0o222
    ):
        raise ManifestGenerationError(
            "immutable OneFormer PTM stage manifest is unavailable"
        )
    stage_document = json.loads(
        ptm_stage_manifest.read_text(encoding="utf-8")
    )
    stage_root = stage_document.get("publication", {}).get(
        "canonical_root"
    )
    try:
        stage_document = ptm_stage.validate_stage_manifest(
            stage_document,
            registry=load_ptm_registry(),
            canonical_root=stage_root,
        )
    except Exception as exc:
        raise ManifestGenerationError(
            "OneFormer PTM stage manifest is invalid"
        ) from exc
    return {
        "repository": str(repository.resolve()),
        "source_commit": head,
        "source_dirty": False,
        "wheel_path": str(wheel.resolve()),
        "wheel_sha256": EXPECTED_WHEEL_SHA256,
        "wheel_build_commit": WHEEL_BUILD_COMMIT,
        "wheel_registry_file_sha256": (
            EXPECTED_WHEEL_REGISTRY_FILE_SHA256
        ),
        "wheel_oneformer_registry_sha256": (
            EXPECTED_WHEEL_ONEFORMER_REGISTRY_SHA256
        ),
        "sdk_dir": str(sdk.resolve()),
        "sdk_commit": EXPECTED_SDK_COMMIT,
        "skills_repository": str(skills.resolve()),
        "skills_commit": EXPECTED_SKILLS_COMMIT,
        "skill_dir": str(
            (skills / "skills/models/tao-train-oneformer").resolve()
        ),
        "qualification_evidence_path": str(qualification.resolve()),
        "runtime_local_eligibility": runtime_local_eligibility,
        "ptm_stage_manifest_path": str(ptm_stage_manifest.resolve()),
        "ptm_stage_manifest_sha256": campaign_contract.sha256_file(
            ptm_stage_manifest
        ),
        "ptm_stage_content_sha256": stage_document["manifest_sha256"],
        "runtime_overlay_local_archive_path": overlay[
            "local_archive_path"
        ],
        "runtime_overlay_local_identity": overlay,
        "partition": "polar3",
        "account": "edgeai_tao-ptm_image-foundation-model-clip",
        "base_results_dir": (
            "/lustre/fsw/portfolios/edgeai/projects/"
            "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam"
        ),
        "container_mounts": "/lustre",
        "time_hours": 4.0,
        "timeout_hours": 3.8,
        "max_job_retries": campaign_contract.FROZEN_SLURM_RETRY_CAP,
        "hardware_contract": copy.deepcopy(campaign_contract.FROZEN_HARDWARE),
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
    runtime_overlay: str | Path = DEFAULT_RUNTIME_OVERLAY,
) -> dict[str, Any]:
    repository_path = Path(repository).resolve()
    value = campaign_contract.build_preregistered_contract(
        campaign_id=(
            "oneformer-coco2017-objective-aware-three-mode-v5-20260801"
        ),
        dataset=dataset_record(dataset_manifest, stage_manifest),
        skill_dir=(
            Path(skills).resolve()
            / "skills/models/tao-train-oneformer"
        ),
        runtime=_runtime(
            repository=repository_path,
            wheel=Path(wheel).resolve(),
            sdk=Path(sdk).resolve(),
            skills=Path(skills).resolve(),
            qualification=Path(qualification),
            qualification_contract=Path(qualification_contract),
            ptm_stage_manifest=Path(ptm_stage_manifest),
            runtime_overlay=Path(runtime_overlay),
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
        "oneformer_latency_worker_sha256": campaign_contract.sha256_file(
            HERE / "oneformer_latency_worker.py"
        ),
        "checkpoint_resume_sha256": campaign_contract.sha256_file(
            HERE.parent / "checkpoint_resume.py"
        ),
        "static_sqsh_audit_sha256": campaign_contract.sha256_file(
            HERE / "static_sqsh_audit.v1.json"
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
        "--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST
    )
    parser.add_argument(
        "--stage-manifest", type=Path, default=DEFAULT_STAGE_MANIFEST
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
    parser.add_argument(
        "--runtime-overlay",
        type=Path,
        default=DEFAULT_RUNTIME_OVERLAY,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--automatic-trigger",
        action="store_true",
        help=(
            "Wait for terminal immutable v4 evidence, seal v5, and continue "
            "without another confirmation step."
        ),
    )
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
            "oneformer_coco2017_three_mode_v5"
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
        runtime_overlay=args.runtime_overlay,
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
                    "the sealed overlay remediates the immutable base-SQSH "
                    "findings; the automatic trigger validates the exact v4 "
                    "completion and its campaign-local in-memory PTM registry "
                    "projection before launching"
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
