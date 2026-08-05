from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from tao_automl.ptm_registry import canonical_sha256, load_ptm_registry

from . import (
    campaign_contract,
    manifest_generator,
    qualification_campaign,
    qualification_gate,
    run_campaign,
)
from .qualification_gate import (
    QualificationGateError,
    QualificationLoadEvidence,
    audit_qualification,
)


HERE = Path(__file__).resolve().parent
SKILL_DIR = (
    manifest_generator.DEFAULT_SKILLS
    / "skills/models/tao-train-segformer"
)
DATASET_STAGE_MANIFEST = manifest_generator.DEFAULT_STAGE_MANIFEST
if not DATASET_STAGE_MANIFEST.is_file():
    # The dataset-staging branch is an explicit integration dependency. This
    # fallback keeps this isolated worktree testable until both commits land.
    DATASET_STAGE_MANIFEST = Path(
        "/localhome/local-rarunachalam/.tao/worktrees/"
        "tao-automl-segmentation-datasets/experiments/"
        "cross_model_automl_20260729/segmentation_datasets/"
        "dataset_stage_manifest.v1.json"
    )


def _dataset() -> dict:
    value = manifest_generator.dataset_record(
        manifest_generator.DEFAULT_DATASET_MANIFEST,
        DATASET_STAGE_MANIFEST,
    )
    return value


def _runtime(tmp_path: Path) -> dict:
    return {
        "repository": str(Path(__file__).resolve().parents[3]),
        "source_commit": "c" * 40,
        "source_dirty": False,
        "wheel_path": str(manifest_generator.DEFAULT_WHEEL),
        "wheel_sha256": manifest_generator.EXPECTED_WHEEL_SHA256,
        "sdk_dir": str(manifest_generator.DEFAULT_SDK),
        "sdk_commit": manifest_generator.EXPECTED_SDK_COMMIT,
        "skills_repository": str(manifest_generator.DEFAULT_SKILLS),
        "skills_commit": manifest_generator.EXPECTED_SKILLS_COMMIT,
        "skill_dir": str(SKILL_DIR),
        "qualification_evidence_path": str(
            tmp_path / "qualification.json"
        ),
        "ptm_stage_manifest_path": str(tmp_path / "ptms.json"),
        "partition": campaign_contract.FROZEN_SLURM_PARTITION,
        "account": "edgeai_tao-ptm_image-foundation-model-clip",
        "base_results_dir": "/lustre/fsw/portfolios/edgeai/users/rarunachalam",
        "container_mounts": "/lustre",
        "time_hours": campaign_contract.FROZEN_SLURM_TIME_HOURS,
        "timeout_hours": campaign_contract.FROZEN_SLURM_TIMEOUT_HOURS,
        "max_job_retries": 10,
        "hardware_contract": copy.deepcopy(
            campaign_contract.FROZEN_HARDWARE
        ),
    }


@pytest.fixture
def contract(tmp_path: Path) -> dict:
    value = campaign_contract.build_preregistered_contract(
        campaign_id="segformer-test",
        dataset=_dataset(),
        skill_dir=str(SKILL_DIR),
        runtime=_runtime(tmp_path),
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
        "segformer_latency_worker_sha256": (
            campaign_contract.sha256_file(
                HERE / "segformer_latency_worker.py"
            )
        )
    }
    value["contract_sha256"] = canonical_sha256(value)
    return campaign_contract.validate_contract(value)


def _workflow(
    checkpoint_id: str,
    *,
    success: bool,
    metric: float = 0.4,
) -> dict:
    record = load_ptm_registry().checkpoint(checkpoint_id)
    plan = qualification_campaign._v4_phase_recovery_records()[checkpoint_id]
    phase_policy = (
        campaign_contract.FROZEN_QUALIFICATION_PHASE_RECOVERY_POLICY
    )
    if not success:
        value = {
            "schema_version": 2,
            "qualification_revision": (
                campaign_contract.QUALIFICATION_REVISION
            ),
            "checkpoint_id": checkpoint_id,
            "status": "failure",
            "terminal": True,
            "failure_preserved": True,
            "failure_code": "direct_full_run_failed",
            "failure_reason": "frozen test failure",
            "recipe_fidelity": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_FIDELITY
            ),
            "runtime_overlay": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
            ),
            "infrastructure_retry_policy": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY
            ),
            "phase_recovery_policy": copy.deepcopy(phase_policy),
            "execution_plan": copy.deepcopy(plan),
        }
        value["workflow_sha256"] = canonical_sha256(value)
        return value
    reused = plan["mode"] == "reuse_sealed_v4_terminal_train"
    source_path = (
        plan["source_checkpoint"]["path"]
        if reused
        else f"/lustre/ptms/{checkpoint_id}.pth"
    )
    load_payload = {
        "checkpoint": source_path,
        "component": (
            "model"
            if record["checkpoint_target"] == "train.pretrained_model_path"
            else "backbone"
        ),
        "loaded_keyset_sha256": "c" * 64,
        "loaded_tensor_count": 365,
        "missing_tensor_count": 4,
        "non_tensor_count": 0,
        "schema_version": 1,
        "shape_mismatched_tensor_count": 4,
        "unmatched_tensor_count": 2,
    }
    pretrained_load = (
        copy.deepcopy(plan["pretrained_load"])
        if reused
        else {
            **load_payload,
            "status_record_occurrences": 1,
            "report_sha256": canonical_sha256(load_payload),
        }
    )

    def completed_job(phase: str) -> dict:
        policy = campaign_contract.FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY
        command_sha256 = ("d" if phase == "train" else "e") * 64
        job_id = f"{phase}-job-id"
        result_root = f"/lustre/results/{checkpoint_id}/{phase}"
        submitted_at = "2026-08-01T00:00:00Z"
        terminal_at = "2026-08-01T00:01:00Z"
        attempt = {
            "job_attempt": 1,
            "tao_job_id": job_id,
            "status": "Complete",
            "submitted_at_utc": submitted_at,
            "terminal_at_utc": terminal_at,
            "result_root": result_root,
            "submission": {
                "attempt_count": 1,
                "retry_count": 0,
                "transient_failures": [],
                "stable_job_identity_obtained": True,
                "policy_sha256": canonical_sha256(policy),
            },
            "command_sha256": command_sha256,
            "infrastructure_failure_evidence": {
                "classification": "terminal_status_not_retryable",
                "retry_eligible": False,
                "terminal_status": "Complete",
            },
            "infrastructure_retry_submitted": False,
        }
        return {
            "execution_mode": (
                "run_fresh_full_train"
                if phase == "train"
                else "new_standalone_evaluation"
            ),
            "new_job_submitted": True,
            "runtime_overlay_required": True,
            "command_sha256": command_sha256,
            "tao_job_id": job_id,
            "status": "Complete",
            "submitted_at_utc": submitted_at,
            "terminal_at_utc": terminal_at,
            "result_root": result_root,
            "job_attempt": 1,
            "attempts": [attempt],
            "infrastructure_retry_count": 0,
            "infrastructure_retry_policy_sha256": canonical_sha256(policy),
            "maximum_job_attempts": policy[
                "maximum_job_attempts_per_phase"
            ],
            "successful_job_replacement_allowed": False,
        }

    if reused:
        train_status_evidence = copy.deepcopy(
            plan["validation_status_evidence"]
        )
        train_status_evidence["pretrained_load"] = copy.deepcopy(
            pretrained_load
        )
        terminal_checkpoint = copy.deepcopy(plan["terminal_checkpoint"])
        predecessor_job = copy.deepcopy(plan["train_job"])
        train_job = {
            "execution_mode": "reuse_sealed_v4_terminal_train",
            "new_job_submitted": False,
            "successful_train_reexecution": False,
            "runtime_overlay_required": True,
            "runtime_overlay": copy.deepcopy(
                plan["predecessor_runtime_overlay"]
            ),
            "predecessor_campaign_id": phase_policy[
                "predecessor_campaign_id"
            ],
            "predecessor_completion_whole_file_sha256": phase_policy[
                "predecessor_completion_whole_file_sha256"
            ],
            "predecessor_load_audit_whole_file_sha256": phase_policy[
                "predecessor_load_audit_whole_file_sha256"
            ],
            "v4_workflow_sha256": plan["v4_workflow_sha256"],
            "v4_load_audit_row_sha256": plan[
                "v4_load_audit_row_sha256"
            ],
            "tao_job_id": predecessor_job["tao_job_id"],
            "tao_job_id_origin": "sealed_predecessor_v4",
            "status": "Complete",
            "result_root": predecessor_job["result_root"],
            "command_sha256": predecessor_job["command_sha256"],
            "spec_sha256": predecessor_job["spec_sha256"],
            "predecessor_train_job": predecessor_job,
            "status_evidence": copy.deepcopy(train_status_evidence),
            "terminal_checkpoint": copy.deepcopy(terminal_checkpoint),
        }
        train_recipe = copy.deepcopy(plan["predecessor_recipe_fidelity"])
        train_overlay = copy.deepcopy(plan["predecessor_runtime_overlay"])
        train_revision = plan["predecessor_qualification_revision"]
        train_metric = train_status_evidence["val_miou"]
    else:
        train_status_evidence = {"pretrained_load": pretrained_load}
        terminal_checkpoint = {
            "path": (
                "/lustre/results/"
                f"{checkpoint_id}/model_epoch_049_step_09150.pth"
            ),
            "size_bytes": 123,
            "sha256": "b" * 64,
            "training_epochs": 50,
            "terminal_epoch_index": 49,
            "naming_contract": "model_epoch_049_step_numeric",
            "ambiguity_policy": "fail_closed",
        }
        train_job = completed_job("train")
        train_recipe = copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_FIDELITY
        )
        train_overlay = copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
        )
        train_revision = campaign_contract.QUALIFICATION_REVISION
        train_metric = metric

    evaluation_job = completed_job("evaluate")
    evaluation_job["checkpoint"] = copy.deepcopy(terminal_checkpoint)
    value = {
        "schema_version": 2,
        "qualification_revision": campaign_contract.QUALIFICATION_REVISION,
        "checkpoint_id": checkpoint_id,
        "status": "success",
        "terminal": True,
        "failure_preserved": False,
        "source_checkpoint": {
            "path": source_path,
            "size_bytes": record["expected_size_bytes"],
            "sha256": (
                plan["source_checkpoint"]["sha256"]
                if reused
                else "a" * 64
            ),
        },
        "train": {
            "status": "Complete",
            "execution_mode": plan["mode"],
            "source_qualification_revision": train_revision,
            "full_dataset": True,
            "training_epochs": (
                campaign_contract.FROZEN_QUALIFICATION_TRAINING_EPOCHS
            ),
            "validation_interval": 1,
            "validation_record_count": (
                campaign_contract.FROZEN_QUALIFICATION_TRAINING_EPOCHS
            ),
            "recipe_fidelity": train_recipe,
            "runtime_overlay": train_overlay,
            "job": train_job,
            "nodes": 1,
            "gpus": 8,
            "val_miou": train_metric,
            "terminal_checkpoint": terminal_checkpoint,
            "status_evidence": train_status_evidence,
        },
        "evaluation": {
            "status": "Complete",
            "full_validation_split": True,
            "runtime_overlay": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
            ),
            "job": evaluation_job,
            "nodes": 1,
            "gpus": 8,
            "test_miou": metric,
        },
        "agent_intervention_flags": {
            name: False for name in campaign_contract.AGENT_FLAGS
        },
        "recipe_fidelity": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_FIDELITY
        ),
        "runtime_overlay": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
        ),
        "infrastructure_retry_policy": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY
        ),
        "phase_recovery_policy": copy.deepcopy(phase_policy),
        "execution_plan": copy.deepcopy(plan),
    }
    value["workflow_sha256"] = canonical_sha256(value)
    return value


def _qualification_document(success_id: str | None = None) -> dict:
    snapshot = campaign_contract.segformer_registry_snapshot()
    workflows = [
        _workflow(
            record["id"],
            success=record["id"] == success_id,
        )
        for record in snapshot["records"]
    ]
    successful = sum(item["status"] == "success" for item in workflows)
    value = {
        "schema_version": 2,
        "qualification_revision": campaign_contract.QUALIFICATION_REVISION,
        "campaign_id": campaign_contract.QUALIFICATION_CAMPAIGN_ID,
        "model": "segformer",
        "task": "semantic_segmentation",
        "registry_sha256": snapshot["registry_sha256"],
        "sqsh_sha256": campaign_contract.FROZEN_SQSH["sha256"],
        "recipe_fidelity": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_FIDELITY
        ),
        "runtime_overlay": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
        ),
        "infrastructure_retry_policy": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY
        ),
        "phase_recovery_policy": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_PHASE_RECOVERY_POLICY
        ),
        "prior_revision_evidence": copy.deepcopy(
            campaign_contract.FROZEN_PRIOR_QUALIFICATION_EVIDENCE
        ),
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "status": (
            "success"
            if successful == len(workflows)
            else "terminal_with_failures"
        ),
        "terminal": True,
        "successful_workflows": successful,
        "failed_workflows": len(workflows) - successful,
        "all_official_arms_attempted": True,
        "failure_records_preserved": True,
        "replacement_workflows_submitted": False,
        "agent_intervention_flags": {
            name: False for name in campaign_contract.AGENT_FLAGS
        },
        "workflows": workflows,
    }
    value["evidence_sha256"] = canonical_sha256(value)
    return value


def _refresh_qualification_summary(document: dict) -> None:
    successful = sum(
        item["status"] == "success" for item in document["workflows"]
    )
    document.update(
        {
            "status": (
                "success"
                if successful == len(document["workflows"])
                else "terminal_with_failures"
            ),
            "terminal": True,
            "successful_workflows": successful,
            "failed_workflows": len(document["workflows"]) - successful,
            "all_official_arms_attempted": True,
            "failure_records_preserved": True,
            "replacement_workflows_submitted": False,
            "agent_intervention_flags": {
                name: False for name in campaign_contract.AGENT_FLAGS
            },
        }
    )


def _seal_runtime_local_qualification(
    contract: dict,
    tmp_path: Path,
    success_ids: tuple[str, ...],
) -> tuple[dict, Path]:
    document = _qualification_document()
    by_id = {
        item["checkpoint_id"]: index
        for index, item in enumerate(document["workflows"])
    }
    for checkpoint_id in success_ids:
        document["workflows"][by_id[checkpoint_id]] = _workflow(
            checkpoint_id,
            success=True,
        )
    _refresh_qualification_summary(document)
    document["automl_contract_sha256"] = (
        campaign_contract.FROZEN_V5_QUALIFICATION_CONTRACT[
            "contract_sha256"
        ]
    )
    document["qualification_controller_sha256"] = contract[
        "launcher_integrity"
    ]["qualification_campaign_sha256"]
    document["evidence_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in document.items()
            if key != "evidence_sha256"
        }
    )
    qualification_path = tmp_path / "qualification.json"
    qualification_path.write_text(json.dumps(document), encoding="utf-8")

    snapshot = campaign_contract.segformer_registry_snapshot()
    policy = {
        "schema_version": 1,
        "kind": campaign_contract.RUNTIME_LOCAL_ELIGIBILITY_KIND,
        "enabled": True,
        "scope": "campaign_local_in_memory_projection",
        "model": "segformer",
        "task": "semantic_segmentation",
        "tao_version": "7.1.0",
        "container_sha256": campaign_contract.FROZEN_SQSH["sha256"],
        "base_registry_version": snapshot["registry_version"],
        "base_registry_sha256": snapshot["registry_sha256"],
        "qualification_evidence_path": str(qualification_path),
        "qualification_file_sha256": campaign_contract.sha256_file(
            qualification_path
        ),
        "qualification_evidence_sha256": document["evidence_sha256"],
        "qualification_contract_sha256": (
            campaign_contract.FROZEN_V5_QUALIFICATION_CONTRACT[
                "contract_sha256"
            ]
        ),
        "qualification_controller_sha256": contract[
            "launcher_integrity"
        ]["qualification_campaign_sha256"],
        "eligibility_gate_sha256": contract["launcher_integrity"][
            "qualification_gate_sha256"
        ],
        "runtime_resolver_sha256": campaign_contract.sha256_file(
            Path(contract["runtime"]["repository"])
            / "src/tao_automl/ptm_runtime.py"
        ),
        "eligibility_source_commit": contract["runtime"]["source_commit"],
        "wheel_sha256": contract["runtime"]["wheel_sha256"],
        "sdk_commit": contract["runtime"]["sdk_commit"],
        "skills_commit": contract["runtime"]["skills_commit"],
        "license_policy": "complete_existing_registry_metadata_only",
        "checkpoint_spec_file": copy.deepcopy(
            campaign_contract.FROZEN_RUNTIME_LOCAL_CHECKPOINT_SPEC_FILE
        ),
        "repository_registry_mutation_allowed": False,
        "missing_license_normalization_allowed": False,
        "failed_arm_promotion_allowed": False,
        "unsupported_arm_promotion_allowed": False,
        "agent_override_allowed": False,
    }
    sealed = copy.deepcopy(contract)
    sealed.pop("contract_sha256")
    sealed["runtime"]["automatic_successor_contract_path"] = (
        campaign_contract.FROZEN_V6_SUCCESSOR_CONTRACT_PATH
    )
    sealed["runtime"]["automatic_successor_runtime_root"] = (
        campaign_contract.FROZEN_V6_SUCCESSOR_RUNTIME_ROOT
    )
    sealed["runtime"]["runtime_local_eligibility"] = copy.deepcopy(policy)
    sealed["qualification_policy"]["runtime_local_eligibility"] = (
        copy.deepcopy(policy)
    )
    sealed["contract_sha256"] = canonical_sha256(sealed)
    return campaign_contract.validate_contract(sealed), qualification_path


def _fake_qualification_stage(contract: dict) -> dict:
    rows = []
    registry = load_ptm_registry()
    recovery_records = qualification_campaign._v4_phase_recovery_records()
    for record_summary in campaign_contract.segformer_registry_snapshot()[
        "records"
    ]:
        record = registry.checkpoint(record_summary["id"])
        recovery = recovery_records[record["id"]]
        checkpoint_path = (
            recovery["source_checkpoint"]["path"]
            if recovery["mode"] == "reuse_sealed_v4_terminal_train"
            else (
                "/lustre/segformer-qualification/ptms/"
                f"{record['id']}/{record['source']['member']}"
            )
        )
        specifications = qualification_campaign.qualification_specs(
            contract,
            record,
            checkpoint_path,
        )
        specs = {}
        for action, document in specifications.items():
            content = yaml.safe_dump(document, sort_keys=True).encode()
            digest = hashlib.sha256(content).hexdigest()
            specs[action] = {
                "action": action,
                "document": document,
                "document_sha256": canonical_sha256(document),
                "raw_yaml_sha256": digest,
                "size_bytes": len(content),
                "base_template": {
                    "path": str(
                        SKILL_DIR
                        / "references"
                        / f"spec_template_{action}.yaml"
                    ),
                    "sha256": campaign_contract.sha256_file(
                        SKILL_DIR
                        / "references"
                        / f"spec_template_{action}.yaml"
                    ),
                },
                "local_path": f"/tmp/{record['id']}-{action}.yaml",
                "lustre": {
                    "path": (
                        "/lustre/segformer-qualification/specs/"
                        f"{record['id']}/{action}.yaml"
                    ),
                    "size_bytes": len(content),
                    "sha256": digest,
                    "mode": "444",
                    "cache_hit": False,
                },
            }
        observed_sha = (
            recovery["source_checkpoint"]["sha256"]
            if recovery["mode"] == "reuse_sealed_v4_terminal_train"
            else hashlib.sha256(record["id"].encode()).hexdigest()
        )
        rows.append(
            {
                "checkpoint_id": record["id"],
                "workflow_id": qualification_campaign._workflow_id(
                    record["id"]
                ),
                "registry_status_at_stage": record["status"],
                "registry_record_sha256": canonical_sha256(record),
                "registry_core_identity": (
                    qualification_campaign.registry_core_identity(record)
                ),
                "registry_core_identity_sha256": canonical_sha256(
                    qualification_campaign.registry_core_identity(record)
                ),
                "source": copy.deepcopy(record["source"]),
                "checkpoint_target": record["checkpoint_target"],
                "backbone": record["backbone"],
                "expected_size_bytes": record["expected_size_bytes"],
                "registered_sha256": record.get("sha256"),
                "observed_sha256": observed_sha,
                "verification_mode": (
                    "immutable_identity_observed_sha256"
                ),
                "source_identity_sha256": canonical_sha256(
                    record["source"]
                ),
                "access_probe": {
                    "ok": True,
                    "code": "accessible",
                    "remote_size_bytes": record[
                        "expected_size_bytes"
                    ],
                },
                "checkpoint_specific_source_spec": {
                    "available": False,
                    "registry_field_present": (
                        "checkpoint_spec_file" in record
                    ),
                    "reason": (
                        "The official SegFormer registry record publishes no "
                        "checkpoint-specific YAML; the staged specs are "
                        "generated from the sealed TAO templates, frozen VOC "
                        "profile, and exact registry checkpoint target."
                    ),
                },
                "checkpoint": {
                    "path": checkpoint_path,
                    "size_bytes": record["expected_size_bytes"],
                    "sha256": observed_sha,
                    "mode": "444",
                    "cache_hit": False,
                },
                "specs": specs,
                "execution_plan": copy.deepcopy(recovery),
            }
        )
    value = {
        "schema_version": 2,
        "qualification_revision": campaign_contract.QUALIFICATION_REVISION,
        "campaign_id": qualification_campaign.QUALIFICATION_CAMPAIGN_ID,
        "automl_contract_sha256": contract["contract_sha256"],
        "created_at_utc": "2026-07-31T00:00:00Z",
        "model": "segformer",
        "task": "semantic_segmentation",
        "registry_sha256": contract["ptm_inventory"]["registry_sha256"],
        "registry_version": contract["ptm_inventory"]["registry_version"],
        "source_policy": (
            "all_13_official_registry_arms_without_manual_exclusion"
        ),
        "dataset": {
            "prepared_root": contract["dataset"]["prepared_root"],
            "content_sha256": contract["dataset"]["content_sha256"],
            "stage_manifest_sha256": contract["dataset"][
                "stage_manifest_sha256"
            ],
            "train_pairs": 1464,
            "validation_pairs": 1449,
        },
        "runtime": {
            "sqsh_path": contract["sqsh"]["path"],
            "sqsh_sha256": contract["sqsh"]["sha256"],
            "sdk_commit": contract["runtime"]["sdk_commit"],
            "skills_commit": contract["runtime"]["skills_commit"],
            "source_commit": contract["runtime"]["source_commit"],
            "partition": contract["runtime"]["partition"],
            "time_hours": contract["runtime"]["time_hours"],
            "timeout_hours": contract["runtime"]["timeout_hours"],
            "nodes_per_workflow": 1,
            "gpus_per_workflow": 8,
            "required_gpu": copy.deepcopy(
                campaign_contract.FROZEN_HARDWARE
            ),
            "runtime_overlay": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
            ),
            "infrastructure_retry_policy": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY
            ),
            "phase_recovery_policy": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_PHASE_RECOVERY_POLICY
            ),
        },
        "recipe_fidelity": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_FIDELITY
        ),
        "prior_revision_evidence": copy.deepcopy(
            campaign_contract.FROZEN_PRIOR_QUALIFICATION_EVIDENCE
        ),
        "ptms": rows,
        "execution": {
            "operation": (
                "data_only_download_checksum_spec_generation_and_"
                "lustre_publication"
            ),
            "cpu_model_runs": 0,
            "gpu_model_runs": 0,
            "smoke_model_runs": 0,
            "mini_step_runs": 0,
            "checkpoint_loads": 0,
            "scheduler_jobs_submitted": 0,
            "fallback_ptms_used": 0,
            "manually_excluded_ptms": 0,
        },
        "agent_intervention_flags": {
            name: False for name in campaign_contract.AGENT_FLAGS
        },
    }
    value["stage_manifest_sha256"] = canonical_sha256(value)
    return value


def test_exact_tao_identifier_action_and_primary_metric():
    info = json.loads(
        json.dumps(
            __import__("yaml").safe_load(
                (SKILL_DIR / "references/skill_info.yaml").read_text()
            )
        )
    )
    assert info["network_arch"] == "segformer"
    assert info["actions"]["train"]["command"] == (
        "segformer train -e {config_path}"
    )
    assert campaign_contract.mode_settings("x", "accuracy")[
        "accuracy_metric"
    ] == "val_miou"


def test_only_common_train_parameters_are_searched():
    evidence = campaign_contract.validate_packaged_train_schema(SKILL_DIR)
    assert tuple(evidence["explicit_search_parameters"]) == (
        "train.optim.lr",
        "train.optim.weight_decay",
    )
    assert "dataset.segment.img_size" not in campaign_contract.SEARCH_PARAMETERS
    assert "model.backbone.type" not in campaign_contract.SEARCH_PARAMETERS
    assert campaign_contract.SEARCH_SPACE == {
        "train.optim.lr": {
            "type": "float",
            "minimum": 2e-5,
            "maximum": 6e-4,
            "scale": "linear",
        },
        "train.optim.weight_decay": {
            "type": "float",
            "minimum": 1e-4,
            "maximum": 0.1,
            "scale": "linear",
        },
    }


def test_complete_voc2012_record_and_loss_preserving_palette():
    dataset = _dataset()
    assert dataset["train_image_count"] == dataset["train_mask_count"] == 1464
    assert (
        dataset["validation_image_count"]
        == dataset["validation_mask_count"]
        == 1449
    )
    assert dataset["file_manifest_entry_count"] == 5827
    assert dataset["manifest_sha256"] == (
        "051ab20215b8e6976763ac82a3db20a68264759edef3d62fd0c8553c501123ff"
    )
    assert dataset["content_sha256"] == (
        "815b5d01b625238b449c4bca828bf96107b367f0f4d5d8a31d2f97c6161a5de0"
    )
    assert dataset["stage_manifest_sha256"] == (
        "437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d"
    )
    assert dataset["remote_read_only"] is True
    assert dataset["remote_writable_entries_after_lock"] == 0
    palette = campaign_contract.voc_palette()
    assert [item["label_id"] for item in palette] == [*range(21), 255]
    assert all(item["rgb"] == [item["label_id"]] for item in palette)


def test_search_profile_remains_frozen_at_v1_fidelity():
    profile = campaign_contract.profile_overrides(
        _dataset()["prepared_root"]
    )
    segment = profile["dataset"]["segment"]
    train = profile["train"]
    assert segment["dataset"] == "SFDataset"
    assert segment["num_classes"] == 21
    assert segment["label_transform"] == "None"
    assert segment["img_size"] == 512
    assert train["num_gpus"] == 8
    assert train["gpu_ids"] == list(range(8))
    assert train["num_nodes"] == 1
    assert train["num_epochs"] == 10
    assert train["use_distributed_sampler"] is False
    assert train["validation_interval"] == 1
    assert train["tensorboard"]["enabled"] is False


def test_qualification_v5_uses_official_multiclass_fidelity_uniformly():
    profile = campaign_contract.qualification_profile_overrides(
        _dataset()["prepared_root"]
    )
    segment = profile["dataset"]["segment"]
    train = profile["train"]
    fidelity = campaign_contract.FROZEN_QUALIFICATION_FIDELITY
    assert fidelity["source_recipe"].endswith(
        "segformer/experiment_specs/experiment_multi-class.yaml"
    )
    assert fidelity["source_recipe_sha256"] == (
        "210b6b6c4952289e3dbc1f025b3f0b8f17a073702290cb565796ed6c6ea36b21"
    )
    assert train["num_epochs"] == 50
    assert train["checkpoint_interval"] == 50
    assert train["validation_interval"] == 1
    assert train["optim"] == {
        "optim": "adamw",
        "lr": 1.0e-4,
        "weight_decay": 5.0e-4,
    }
    assert segment["augmentation"]["random_color"]["enable"] is False
    assert segment["augmentation"]["with_random_blur"] is False
    assert train["use_distributed_sampler"] is True


def test_qualification_v5_paths_preserve_frozen_v1_v2_v3_v4_evidence():
    v1 = campaign_contract.FROZEN_V1_QUALIFICATION_EVIDENCE
    v2 = campaign_contract.FROZEN_V2_QUALIFICATION_EVIDENCE
    v3 = campaign_contract.FROZEN_V3_QUALIFICATION_EVIDENCE
    v4 = campaign_contract.FROZEN_V4_QUALIFICATION_EVIDENCE
    prior = campaign_contract.FROZEN_PRIOR_QUALIFICATION_EVIDENCE
    assert prior == [v1, v2, v3, v4]
    assert v1["campaign_id"].endswith("-v1")
    assert v1["status"] == "terminal_with_failures"
    assert v1["successful_workflows"] == 0
    assert v1["failed_workflows"] == 13
    assert v1["preserve_immutable"] is True
    assert v1["reuse_for_v2"] is False
    assert "/segformer_voc2012_ptm_qualification_v1/" in (
        v1["completion_path"]
    )
    assert v2["campaign_id"].endswith("-v2")
    assert v2["status"] == "terminal_with_failures"
    assert v2["successful_workflows"] == 0
    assert v2["failed_workflows"] == 13
    assert v2["controller_failure_workflows"] == 12
    assert v2["runtime_failure_workflows"] == 1
    assert v2["preserve_immutable"] is True
    assert v2["reuse_for_v3"] is False
    assert campaign_contract.sha256_file(v2["completion_path"]) == (
        v2["completion_whole_file_sha256"]
    )
    assert campaign_contract.sha256_file(
        v2["ptm_stage_manifest_path"]
    ) == v2["ptm_stage_manifest_whole_file_sha256"]
    assert campaign_contract.sha256_file(
        v2["launch_preflight_path"]
    ) == v2["launch_preflight_whole_file_sha256"]
    assert campaign_contract.sha256_file(
        v2["automatic_handoff_path"]
    ) == v2["automatic_handoff_whole_file_sha256"]
    assert v3["campaign_id"].endswith("-v3")
    assert v3["status"] == "terminal_with_failures"
    assert v3["successful_workflows"] == 0
    assert v3["failed_workflows"] == 13
    assert v3["controller_template_failure_workflows"] == 13
    assert v3["preserve_immutable"] is True
    assert v3["reuse_for_v4"] is False
    for path_key, sha_key in (
        ("contract_path", "contract_whole_file_sha256"),
        ("completion_path", "completion_whole_file_sha256"),
        (
            "ptm_stage_manifest_path",
            "ptm_stage_manifest_whole_file_sha256",
        ),
        ("launch_preflight_path", "launch_preflight_whole_file_sha256"),
        (
            "automatic_handoff_path",
            "automatic_handoff_whole_file_sha256",
        ),
    ):
        assert campaign_contract.sha256_file(v3[path_key]) == v3[sha_key]
    assert v4["campaign_id"].endswith("-v4")
    assert v4["positive_load_train_workflows"] == 4
    assert v4["backbone_prefix_load_failure_workflows"] == 9
    for path_key, sha_key in (
        ("contract_path", "contract_whole_file_sha256"),
        ("completion_path", "completion_whole_file_sha256"),
        (
            "ptm_stage_manifest_path",
            "ptm_stage_manifest_whole_file_sha256",
        ),
        ("ptm_load_audit_path", "ptm_load_audit_whole_file_sha256"),
    ):
        assert campaign_contract.sha256_file(v4[path_key]) == v4[sha_key]
    assert qualification_campaign.QUALIFICATION_CAMPAIGN_ID.endswith("-v5")
    assert qualification_campaign.DEFAULT_CONTRACT.name == "campaign.v5.json"
    assert run_campaign.DEFAULT_CONTRACT.name == "campaign.v5.json"
    assert "qualification_v5" in str(
        qualification_campaign.DEFAULT_RUNTIME_ROOT
    )
    assert "qualification_v5" in str(
        qualification_campaign.DEFAULT_LOCAL_CACHE
    )
    assert "qualification_v5" in str(
        qualification_campaign.DEFAULT_LUSTRE_INPUT_ROOT
    )
    assert manifest_generator.DEFAULT_QUALIFICATION != Path(
        v3["completion_path"]
    )
    assert manifest_generator.DEFAULT_PTM_STAGE_MANIFEST != Path(
        v3["ptm_stage_manifest_path"]
    )


def test_qualification_v5_binds_combined_runtime_overlay(contract):
    overlay = campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
    assert overlay["combined_commit"] == (
        "2681dea4c876b759f8a0446491b3619e6120b531"
    )
    assert overlay["source_commit"] == overlay["combined_commit"]
    assert overlay["archive_sha256"] == (
        "a7d5316816710b258c52001f979a22723c88fca5101a05ca3a48838ce81d1ee4"
    )
    assert overlay["required_actions"] == ["train", "evaluate"]
    assert overlay["file_count"] == 5
    assert "positive_pretrained_load_receipt" in overlay["remediates"]
    policy = contract["qualification_policy"]
    assert policy["revision"] == 5
    assert policy["campaign_id"].endswith("-v5")
    assert policy["training_epochs"] == 50
    assert policy["recipe_fidelity"] == (
        campaign_contract.FROZEN_QUALIFICATION_FIDELITY
    )
    assert policy["runtime_overlay"] == overlay
    assert policy["phase_recovery_policy"] == (
        campaign_contract.FROZEN_QUALIFICATION_PHASE_RECOVERY_POLICY
    )
    assert policy["prior_revision_evidence"] == (
        campaign_contract.FROZEN_PRIOR_QUALIFICATION_EVIDENCE
    )
    assert contract["search"]["training_epochs"] == 10


def test_voc_metric_sanity_is_separate_from_product_selection(contract):
    gate = contract["validation_sanity_gate"]
    assert gate["metric"] == "val_miou"
    assert gate["minimum"] == 0.10
    assert gate["role"] == (
        "experiment_correctness_gate_not_product_selection"
    )
    assert gate["low_finite_metric_automatically_accepted"] is False


def test_all_official_ptms_are_hierarchical_arms():
    snapshot = campaign_contract.segformer_registry_snapshot()
    assert snapshot["record_count"] == 13
    assert len({item["id"] for item in snapshot["records"]}) == 13
    assert all(item["source"]["official"] is True for item in snapshot["records"])
    assert all(
        item["checkpoint_target"]
        in {
            "train.pretrained_model_path",
            "model.backbone.pretrained_backbone_path",
        }
        for item in snapshot["records"]
    )
    assert set(snapshot["supported_ids"]).isdisjoint(
        snapshot["unverified_ids"]
    )


def test_budget_covers_two_calibration_points_per_official_arm():
    arm_count = campaign_contract.segformer_registry_snapshot()[
        "record_count"
    ]
    assert campaign_contract.FROZEN_CANDIDATE_BUDGET >= (
        2 * arm_count + 4
    )


def test_mode_objectives_are_independent(contract):
    modes = {
        item["mode"]: item["settings"] for item in contract["modes"]
    }
    assert modes["accuracy"]["selection_mode"] == "accuracy"
    assert "latency_accuracy_retention" not in modes["accuracy"]
    assert modes["latency"]["latency_accuracy_retention"] == {
        "type": "relative",
        "retained_fraction": 0.9,
        "reference": "accuracy_winner",
    }
    assert modes["latency"]["objective_acquisition"][
        "calibration_points"
    ] == 2
    assert modes["multi_objective"]["selection_mode"] == "multi_objective"
    assert "latency_accuracy_retention" not in modes["multi_objective"]
    assert modes["multi_objective"]["multi_objective_min_accuracy"] is None
    assert all(item["observation_sharing"] is False for item in contract["modes"])


def test_contract_is_pinned_sqsh_eight_gpu_and_zero_local_model_runs(contract):
    assert contract["sqsh"] == campaign_contract.FROZEN_SQSH
    assert contract["execution"]["container_mode"] == "pinned_sqsh"
    assert contract["execution"]["nodes_per_child"] == 1
    assert contract["execution"]["gpus_per_child"] == 8
    assert contract["execution"]["cpu_runs"] == 0
    assert contract["execution"]["smoke_runs"] == 0
    assert contract["execution"]["local_model_runs"] == 0
    assert contract["qualification_policy"]["cpu_model_runs"] == 0
    assert contract["qualification_policy"]["smoke_model_runs"] == 0
    assert contract["qualification_policy"]["mini_step_runs"] == 0


def test_latency_protocol_is_4000_real_validation_samples(contract):
    protocol = contract["latency_protocol"]
    assert protocol["warmup_iterations"] == 50
    assert protocol["repeated_rounds"] == 5
    assert protocol["timed_iterations"] == 100
    assert protocol["expected_replicas"] == 8
    assert protocol["raw_samples_per_candidate"] == 4000
    source = (HERE / "segformer_latency_worker.py").read_text()
    assert "SFDataModule" in source
    assert "test_dataloader" in source
    assert "model(preloaded[" in source
    assert "torch.randn" not in source
    assert "torch.rand(" not in source


def test_model_imports_and_execution_live_only_below_worker_main():
    tree = ast.parse(
        (HERE / "segformer_latency_worker.py").read_text(encoding="utf-8")
    )
    module_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        not (
            isinstance(node, ast.Import)
            and any(alias.name == "torch" for alias in node.names)
        )
        for node in module_imports
    )
    assert all(
        not (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("nvidia_tao_pytorch")
        )
        for node in module_imports
    )


def test_unverified_full_run_success_cannot_bypass_registry(tmp_path: Path):
    snapshot = campaign_contract.segformer_registry_snapshot()
    success_id = snapshot["records"][0]["id"]
    path = tmp_path / "qualification.json"
    path.write_text(
        json.dumps(_qualification_document(success_id)),
        encoding="utf-8",
    )
    decision = audit_qualification(path)
    if load_ptm_registry().checkpoint(success_id)["status"] == "unverified":
        assert success_id not in decision.checkpoint_ids
        assert any(
            item["checkpoint_id"] == success_id
            and item["code"] == "registry_not_supported"
            for item in decision.blockers
        )
        with pytest.raises(QualificationGateError):
            QualificationLoadEvidence(decision)


def test_sealed_runtime_local_projection_admits_only_exact_success_with_license(
    contract,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        qualification_gate,
        "_validate_sealed_v5_predecessor",
        lambda **_kwargs: None,
    )
    snapshot = campaign_contract.segformer_registry_snapshot()
    city_id = next(
        item["id"]
        for item in snapshot["records"]
        if item["checkpoint_target"] == "train.pretrained_model_path"
    )
    backbone_id = next(
        item["id"]
        for item in snapshot["records"]
        if item["checkpoint_target"]
        == "model.backbone.pretrained_backbone_path"
    )
    sealed, qualification_path = _seal_runtime_local_qualification(
        contract,
        tmp_path,
        (city_id, backbone_id),
    )
    base = load_ptm_registry()
    base_sha = base.document_sha256

    decision = audit_qualification(
        qualification_path,
        expected_contract=sealed,
    )

    assert decision.runtime_ready is True
    assert decision.checkpoint_ids == (city_id,)
    assert decision.blockers == ()
    assert decision.runtime_registry.checkpoint(city_id)["status"] == (
        "supported"
    )
    assert decision.runtime_registry.checkpoint(backbone_id)["status"] == (
        "unverified"
    )
    incomplete = next(
        item
        for item in decision.exclusions
        if item["checkpoint_id"] == backbone_id
    )
    assert incomplete["code"] == "runtime_metadata_incomplete"
    assert "will not invent or normalize a license" in incomplete["reason"]
    eligibility = decision.runtime_eligibility
    assert eligibility["qualified_checkpoint_ids"] == [city_id]
    assert eligibility[
        "runtime_metadata_incomplete_checkpoint_ids"
    ] == [backbone_id]
    assert eligibility["repository_registry_mutated"] is False
    assert eligibility["missing_license_normalization_allowed"] is False
    assert eligibility["failed_arms_preserved"] is True
    assert load_ptm_registry().document_sha256 == base_sha
    assert load_ptm_registry().checkpoint(city_id)["status"] == "unverified"


def test_runtime_local_projection_fails_closed_on_evidence_hash_change(
    contract,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        qualification_gate,
        "_validate_sealed_v5_predecessor",
        lambda **_kwargs: None,
    )
    city_id = next(
        item["id"]
        for item in campaign_contract.segformer_registry_snapshot()["records"]
        if item["checkpoint_target"] == "train.pretrained_model_path"
    )
    sealed, qualification_path = _seal_runtime_local_qualification(
        contract,
        tmp_path,
        (city_id,),
    )
    changed = copy.deepcopy(sealed)
    changed.pop("contract_sha256")
    for location in (
        changed["runtime"],
        changed["qualification_policy"],
    ):
        location["runtime_local_eligibility"][
            "qualification_file_sha256"
        ] = "f" * 64
    changed["contract_sha256"] = canonical_sha256(changed)
    changed = campaign_contract.validate_contract(changed)

    with pytest.raises(
        QualificationGateError,
        match="exact base registry and qualification evidence",
    ):
        audit_qualification(
            qualification_path,
            expected_contract=changed,
        )


def test_runtime_local_projection_rejects_coherent_predecessor_hash_change(
    contract,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint_id = campaign_contract.segformer_registry_snapshot()[
        "records"
    ][0]["id"]
    sealed, qualification_path = _seal_runtime_local_qualification(
        contract,
        tmp_path,
        (checkpoint_id,),
    )
    document = json.loads(qualification_path.read_text(encoding="utf-8"))
    document["automl_contract_sha256"] = "f" * 64
    document.pop("evidence_sha256")
    document["evidence_sha256"] = canonical_sha256(document)
    qualification_path.write_text(json.dumps(document), encoding="utf-8")
    changed = copy.deepcopy(sealed)
    changed.pop("contract_sha256")
    for location in (changed["runtime"], changed["qualification_policy"]):
        policy = location["runtime_local_eligibility"]
        policy["qualification_contract_sha256"] = "f" * 64
        policy["qualification_evidence_sha256"] = document["evidence_sha256"]
        policy["qualification_file_sha256"] = campaign_contract.sha256_file(
            qualification_path
        )
    changed["contract_sha256"] = canonical_sha256(changed)
    monkeypatch.setattr(
        qualification_gate,
        "_project_runtime_registry",
        lambda **_kwargs: pytest.fail("tampered predecessor reached projection"),
    )

    with pytest.raises(
        QualificationGateError,
        match="expected successor campaign contract is invalid",
    ):
        audit_qualification(qualification_path, expected_contract=changed)


def test_prior_evidence_is_preserved_but_cannot_satisfy_v4_gate(
    tmp_path: Path,
):
    document = _qualification_document()
    document["schema_version"] = 1
    document["qualification_revision"] = 1
    document["campaign_id"] = (
        campaign_contract.FROZEN_V1_QUALIFICATION_EVIDENCE["campaign_id"]
    )
    document["evidence_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in document.items()
            if key != "evidence_sha256"
        }
    )
    path = tmp_path / "qualification.v1.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(
        QualificationGateError,
        match="campaign identity or execution policy changed",
    ):
        audit_qualification(path)

    document = _qualification_document()
    document["qualification_revision"] = 3
    document["campaign_id"] = (
        campaign_contract.FROZEN_V3_QUALIFICATION_EVIDENCE["campaign_id"]
    )
    document["evidence_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in document.items()
            if key != "evidence_sha256"
        }
    )
    path = tmp_path / "qualification.v3.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(
        QualificationGateError,
        match="campaign identity or execution policy changed",
    ):
        audit_qualification(path)

    document = _qualification_document()
    document["qualification_revision"] = 2
    document["campaign_id"] = (
        campaign_contract.FROZEN_V2_QUALIFICATION_EVIDENCE["campaign_id"]
    )
    document["evidence_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in document.items()
            if key != "evidence_sha256"
        }
    )
    path = tmp_path / "qualification.v2.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(
        QualificationGateError,
        match="campaign identity or execution policy changed",
    ):
        audit_qualification(path)


def test_terminal_ptm_failures_are_preserved_as_exclusions(tmp_path: Path):
    path = tmp_path / "qualification.json"
    path.write_text(
        json.dumps(_qualification_document()),
        encoding="utf-8",
    )
    decision = audit_qualification(path)
    assert len(decision.exclusions) == 13
    assert all(
        item["code"] == "direct_full_run_failed"
        for item in decision.exclusions
    )
    assert any(
        item["code"] == "no_runtime_qualified_ptm"
        for item in decision.blockers
    )


def test_low_finite_miou_does_not_pass_ptm_qualification(tmp_path: Path):
    document = _qualification_document()
    checkpoint_id = document["workflows"][0]["checkpoint_id"]
    document["workflows"][0] = _workflow(
        checkpoint_id,
        success=True,
        metric=0.09,
    )
    _refresh_qualification_summary(document)
    document["evidence_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in document.items()
            if key != "evidence_sha256"
        }
    )
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    decision = audit_qualification(path)
    assert any(
        item["checkpoint_id"] == checkpoint_id
        and item["code"] == "invalid_success_evidence"
        and "0.10 mIoU" in item["reason"]
        for item in decision.blockers
    )


def _write_gate_cell(
    root: Path,
    contract_sha256: str,
    mode: str,
    passed: bool,
) -> None:
    run_campaign.atomic_json(
        root / "first_candidate_gate" / f"{mode}.json",
        {
            "schema_version": 1,
            "contract_sha256": contract_sha256,
            "mode": mode,
            "candidate_id": f"{mode}_rec_0",
            "passed": passed,
        },
    )


def test_first_candidate_gate_releases_automatically_only_after_all_pass(
    tmp_path: Path,
):
    processes = {
        mode: SimpleNamespace(is_alive=lambda: True)
        for mode in campaign_contract.MODES
    }
    for mode in campaign_contract.MODES[:2]:
        _write_gate_cell(tmp_path, "a" * 64, mode, True)
    assert (
        run_campaign._release_first_candidate_gate(
            tmp_path, processes, "a" * 64
        )
        is None
    )
    _write_gate_cell(
        tmp_path, "a" * 64, campaign_contract.MODES[2], True
    )
    release = run_campaign._release_first_candidate_gate(
        tmp_path, processes, "a" * 64
    )
    assert release["release_remaining_budget"] is True
    assert release["generated_automatically"] is True


def test_first_candidate_gate_fails_closed_on_any_failed_mode(
    tmp_path: Path,
):
    processes = {
        mode: SimpleNamespace(is_alive=lambda: True)
        for mode in campaign_contract.MODES
    }
    for index, mode in enumerate(campaign_contract.MODES):
        _write_gate_cell(tmp_path, "b" * 64, mode, index != 1)
    release = run_campaign._release_first_candidate_gate(
        tmp_path, processes, "b" * 64
    )
    assert release["release_remaining_budget"] is False


def test_launch_plan_is_automatic_and_does_not_launch(contract):
    plan = run_campaign.launch_plan(
        contract,
        ready=False,
        blockers=[{"code": "ptm_qualification_not_ready"}],
    )
    assert plan["launch_authorized"] is False
    assert plan["automatic_trigger"] is True
    assert plan["cpu_or_smoke_model_jobs"] == 0
    assert plan["first_candidate_gate"]["automatic_release"] is True
    assert plan["first_candidate_gate"][
        "remaining_candidates_per_mode"
    ] == 29
    assert plan["resources_per_child"]["gpus"] == 8


def _automatic_successor_contract(
    *,
    contract_path: Path = manifest_generator.DEFAULT_SUCCESSOR_CONTRACT,
    runtime_root: Path = manifest_generator.DEFAULT_SUCCESSOR_RUNTIME_ROOT,
) -> dict:
    source = _automatic_source_seal()
    return {
        "contract_sha256": "c" * 64,
        "runtime": {
            "source_commit": source["source_commit"],
            "wheel_sha256": source["wheel_sha256"],
            "automatic_successor_contract_path": str(contract_path.resolve()),
            "automatic_successor_runtime_root": str(runtime_root.resolve()),
        },
        "launcher_integrity": {
            name: source[name]
            for name in (
                "campaign_contract_sha256",
                "qualification_gate_sha256",
                "manifest_generator_sha256",
                "run_campaign_sha256",
            )
        },
        "qualification_policy": {
            "runtime_local_eligibility": {
                "qualification_file_sha256": "d" * 64,
                "qualification_evidence_sha256": "e" * 64,
            }
        },
    }


def _automatic_source_seal() -> dict:
    return {
        "source_commit": "f" * 40,
        "wheel_sha256": "a" * 64,
        "campaign_contract_sha256": "1" * 64,
        "qualification_gate_sha256": "2" * 64,
        "manifest_generator_sha256": "3" * 64,
        "run_campaign_sha256": "4" * 64,
    }


def _bind_test_automatic_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path]:
    output = tmp_path / "campaign.v6.json"
    runtime_root = tmp_path / "runtime"
    monkeypatch.setattr(
        manifest_generator,
        "DEFAULT_SUCCESSOR_CONTRACT",
        output,
    )
    monkeypatch.setattr(
        manifest_generator,
        "DEFAULT_SUCCESSOR_RUNTIME_ROOT",
        runtime_root,
    )
    return output, runtime_root


def test_automatic_successor_waits_without_early_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    completion = tmp_path / "completion.json"
    status = tmp_path / "runtime/automatic_successor_status.json"
    observed_states = []

    def finish_after_first_wait(_seconds: float) -> None:
        observed_states.append(json.loads(status.read_text())["state"])
        completion.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(manifest_generator.time, "sleep", finish_after_first_wait)
    monkeypatch.setattr(
        manifest_generator,
        "qualification_evidence_record",
        lambda *_args: {
            "qualification_file_sha256": "a" * 64,
            "qualification_evidence_sha256": "b" * 64,
        },
    )
    record = manifest_generator.wait_for_terminal_qualification(
        completion,
        tmp_path / "campaign.v5.json",
        status_path=status,
        poll_seconds=0,
    )

    assert observed_states == ["waiting_for_terminal_v5_completion"]
    assert record["qualification_evidence_sha256"] == "b" * 64
    final = json.loads(status.read_text(encoding="utf-8"))
    assert final["state"] == "terminal_v5_evidence_accepted"
    assert final["successor_contract_sealed"] is False
    assert final["model_jobs_launched"] is False


def test_present_invalid_completion_fails_immediately_without_polling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    completion = tmp_path / "completion.json"
    completion.write_text("{}", encoding="utf-8")
    status = tmp_path / "runtime/status.json"
    monkeypatch.setattr(
        manifest_generator,
        "qualification_evidence_record",
        lambda *_args: (_ for _ in ()).throw(
            manifest_generator.ManifestGenerationError("invalid completion")
        ),
    )
    monkeypatch.setattr(
        manifest_generator.time,
        "sleep",
        lambda _seconds: pytest.fail("invalid present evidence was polled"),
    )
    with pytest.raises(
        manifest_generator.ManifestGenerationError,
        match="invalid completion",
    ):
        manifest_generator.wait_for_terminal_qualification(
            completion,
            tmp_path / "campaign.v5.json",
            status_path=status,
            poll_seconds=0,
        )
    rejected = json.loads(status.read_text(encoding="utf-8"))
    assert rejected["state"] == "terminal_v5_evidence_rejected"
    assert rejected["model_jobs_launched"] is False


def test_sealed_successor_retries_transient_remote_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class Decision:
        def to_dict(self):
            return {"runtime_ready": True}

    decision = Decision()
    outcomes = iter(
        (
            (False, [{"code": "dataset_not_ready"}], None),
            (True, [], decision),
        )
    )
    sleeps = []
    monkeypatch.setattr(run_campaign, "launch_readiness", lambda _contract: next(outcomes))
    monkeypatch.setattr(run_campaign, "atomic_json", lambda *_args: None)
    monkeypatch.setattr(run_campaign.time, "sleep", sleeps.append)
    contract = {
        "contract_sha256": "a" * 64,
        "qualification_policy": {"runtime_local_eligibility": {}},
    }

    assert run_campaign.wait_for_launch_authorization(
        contract,
        runtime_root=tmp_path,
        poll_seconds=0.25,
    ) is decision
    assert sleeps == [0.25]


def test_sealed_successor_rejects_tampered_terminal_evidence_without_polling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        run_campaign,
        "launch_readiness",
        lambda _contract: (
            False,
            [{"code": "ptm_qualification_not_ready"}],
            None,
        ),
    )
    monkeypatch.setattr(run_campaign, "atomic_json", lambda *_args: None)
    monkeypatch.setattr(
        run_campaign.time,
        "sleep",
        lambda _seconds: pytest.fail("tampered terminal evidence was polled"),
    )
    contract = {
        "contract_sha256": "a" * 64,
        "qualification_policy": {"runtime_local_eligibility": {}},
    }

    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="sealed immutable terminal qualification evidence",
    ):
        run_campaign.wait_for_launch_authorization(
            contract,
            runtime_root=tmp_path,
            poll_seconds=0,
        )


def test_zero_success_terminal_completion_is_rejected():
    completion = {"terminal": True, "successful_workflows": 0}
    completion["evidence_sha256"] = canonical_sha256(completion)
    with pytest.raises(
        manifest_generator.ManifestGenerationError,
        match="zero successes",
    ):
        manifest_generator._terminal_completion_sha(completion)


def test_automatic_successor_orders_wait_seal_then_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    order = []
    output, runtime_root = _bind_test_automatic_paths(monkeypatch, tmp_path)
    contract = _automatic_successor_contract(
        contract_path=output,
        runtime_root=runtime_root,
    )
    monkeypatch.setattr(
        manifest_generator,
        "_source_seal_identity",
        lambda *_args: order.append("bind") or _automatic_source_seal(),
    )
    monkeypatch.setattr(
        manifest_generator,
        "wait_for_terminal_qualification",
        lambda *_args, **_kwargs: order.append("wait"),
    )
    monkeypatch.setattr(
        manifest_generator,
        "build_contract",
        lambda **_kwargs: order.append("build") or contract,
    )
    monkeypatch.setattr(
        manifest_generator,
        "seal_contract_no_overwrite",
        lambda *_args: order.append("seal") or True,
    )

    def launch(**kwargs):
        order.append("launch")
        assert kwargs["resume"] is False
        assert kwargs["runtime_root"] == runtime_root.resolve()
        return 0

    monkeypatch.setattr(manifest_generator, "launch_successor_once", launch)
    assert manifest_generator.main(
        [
            "--qualification",
            str(tmp_path / "completion.json"),
            "--qualification-contract",
            str(tmp_path / "campaign.v5.json"),
            "--output",
            str(output),
            "--runtime-root",
            str(runtime_root),
            "--automatic-trigger",
            "--launch",
        ]
    ) == 0
    assert order == ["bind", "wait", "bind", "build", "seal", "launch"]


def test_automatic_successor_wait_failure_prevents_seal_and_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output, runtime_root = _bind_test_automatic_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        manifest_generator,
        "_source_seal_identity",
        lambda *_args: _automatic_source_seal(),
    )
    monkeypatch.setattr(
        manifest_generator,
        "wait_for_terminal_qualification",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            manifest_generator.ManifestGenerationError("invalid terminal v5")
        ),
    )
    monkeypatch.setattr(
        manifest_generator,
        "build_contract",
        lambda **_kwargs: pytest.fail("built before the terminal gate"),
    )
    monkeypatch.setattr(
        manifest_generator,
        "launch_successor_once",
        lambda **_kwargs: pytest.fail("launched before the terminal gate"),
    )
    with pytest.raises(
        manifest_generator.ManifestGenerationError,
        match="invalid terminal v5",
    ):
        manifest_generator.main(
            [
                "--output",
                str(output),
                "--runtime-root",
                str(runtime_root),
                "--automatic-trigger",
                "--launch",
            ]
        )


def test_watcher_source_change_blocks_before_contract_or_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output, runtime_root = _bind_test_automatic_paths(monkeypatch, tmp_path)
    first = _automatic_source_seal()
    second = {**first, "source_commit": "0" * 40}
    identities = iter((first, second))
    monkeypatch.setattr(
        manifest_generator,
        "_source_seal_identity",
        lambda *_args: next(identities),
    )
    monkeypatch.setattr(
        manifest_generator,
        "wait_for_terminal_qualification",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        manifest_generator,
        "build_contract",
        lambda **_kwargs: pytest.fail("built after source changed"),
    )
    with pytest.raises(
        manifest_generator.ManifestGenerationError,
        match="source changed while waiting",
    ):
        manifest_generator.main(
            [
                "--output",
                str(output),
                "--runtime-root",
                str(runtime_root),
                "--automatic-trigger",
                "--launch",
            ]
        )
    assert not output.exists()
    assert not (
        runtime_root / "automatic_successor_launch_claim.json"
    ).exists()


def test_successor_contract_seal_is_atomic_and_never_overwrites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    contract = {"contract_sha256": "a" * 64, "value": 1}
    output = tmp_path / "campaign.v6.json"
    monkeypatch.setattr(
        campaign_contract,
        "validate_contract",
        lambda value: value,
    )

    assert manifest_generator.seal_contract_no_overwrite(output, contract)
    original = output.read_bytes()
    assert output.stat().st_mode & 0o222 == 0
    assert not manifest_generator.seal_contract_no_overwrite(output, contract)
    output.chmod(0o644)
    with pytest.raises(
        manifest_generator.ManifestGenerationError,
        match="refusing overwrite",
    ):
        manifest_generator.seal_contract_no_overwrite(output, contract)
    output.chmod(0o444)
    with pytest.raises(
        manifest_generator.ManifestGenerationError,
        match="refusing overwrite",
    ):
        manifest_generator.seal_contract_no_overwrite(
            output,
            {"contract_sha256": "b" * 64, "value": 2},
        )
    assert output.read_bytes() == original


def test_concurrent_identical_writable_seal_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "claim.json"

    def publish_writable_then_race(source: Path, destination: Path) -> None:
        destination.write_bytes(Path(source).read_bytes())
        destination.chmod(0o644)
        raise FileExistsError

    monkeypatch.setattr(manifest_generator.os, "link", publish_writable_then_race)
    with pytest.raises(
        manifest_generator.ManifestGenerationError,
        match="concurrent seal differs",
    ):
        manifest_generator._seal_json_no_overwrite(output, {"value": 1})
    assert output.stat().st_mode & 0o222


def test_automatic_paths_reject_before_any_side_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output, runtime_root = _bind_test_automatic_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        manifest_generator,
        "_source_seal_identity",
        lambda *_args: pytest.fail("source checked before path binding"),
    )
    cases = (
        (tmp_path / "alternate.json", runtime_root),
        (output, tmp_path / "alternate-runtime"),
    )
    for candidate_output, candidate_root in cases:
        with pytest.raises(
            manifest_generator.ManifestGenerationError,
            match="exact sealed v6 contract and fresh runtime paths",
        ):
            manifest_generator.main(
                [
                    "--output",
                    str(candidate_output),
                    "--runtime-root",
                    str(candidate_root),
                    "--automatic-trigger",
                    "--launch",
                ]
            )
        assert not candidate_output.exists()
        assert not candidate_root.exists()


def test_campaign_runner_rejects_alternate_successor_root_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    contract_path = tmp_path / "campaign.v6.json"
    expected_root = tmp_path / "runtime"
    alternate_root = tmp_path / "alternate-runtime"
    contract = _automatic_successor_contract(
        contract_path=contract_path,
        runtime_root=expected_root,
    )
    monkeypatch.setattr(run_campaign, "load_contract", lambda _path: contract)
    monkeypatch.setattr(
        run_campaign,
        "load_env_file",
        lambda _path: pytest.fail("environment loaded before path binding"),
    )
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="differs from its sealed path",
    ):
        run_campaign.main(
            [
                "--contract",
                str(contract_path),
                "--runtime-root",
                str(alternate_root),
                "--automatic-trigger",
            ]
        )
    assert not alternate_root.exists()


def test_successor_launch_claim_prevents_duplicate_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    def successful_run(arguments):
        calls.append(arguments)
        (tmp_path / "runtime/mode_process_status.json").write_text(
            json.dumps({mode: 0 for mode in campaign_contract.MODES}),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(
        run_campaign,
        "main",
        successful_run,
    )
    contract_path = tmp_path / "campaign.v6.json"
    runtime_root = tmp_path / "runtime"
    arguments = {
        "contract": _automatic_successor_contract(
            contract_path=contract_path,
            runtime_root=runtime_root,
        ),
        "contract_path": contract_path,
        "runtime_root": runtime_root,
        "env_file": tmp_path / "config.env",
        "poll_seconds": 1.0,
        "resume": False,
    }
    assert manifest_generator.launch_successor_once(**arguments) == 0
    assert manifest_generator.launch_successor_once(**arguments) == 0
    assert len(calls) == 1
    assert "--automatic-trigger" in calls[0]
    assert "--launch" in calls[0]
    assert "--resume" not in calls[0]
    result = runtime_root / "automatic_successor_launch_result.json"
    assert result.stat().st_mode & 0o222 == 0


def test_partial_launch_claim_requires_explicit_supported_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []
    monkeypatch.setattr(
        run_campaign,
        "main",
        lambda arguments: calls.append(arguments) or 1,
    )
    contract_path = tmp_path / "campaign.v6.json"
    runtime_root = tmp_path / "runtime"
    contract = _automatic_successor_contract(
        contract_path=contract_path,
        runtime_root=runtime_root,
    )
    contract["search"] = {"candidate_budget_per_mode": 30}
    before = copy.deepcopy(contract)
    arguments = {
        "contract": contract,
        "contract_path": contract_path,
        "runtime_root": runtime_root,
        "env_file": tmp_path / "config.env",
        "poll_seconds": 1.0,
        "resume": False,
    }
    assert manifest_generator.launch_successor_once(**arguments) == 1
    with pytest.raises(
        manifest_generator.ManifestGenerationError,
        match="use --resume",
    ):
        manifest_generator.launch_successor_once(**arguments)
    assert len(calls) == 1
    arguments["resume"] = True
    assert manifest_generator.launch_successor_once(**arguments) == 1
    assert len(calls) == 2
    assert "--resume" in calls[-1]
    assert contract == before


def test_resume_does_not_resubmit_completed_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []
    monkeypatch.setattr(
        run_campaign,
        "main",
        lambda arguments: calls.append(arguments) or 1,
    )
    contract_path = tmp_path / "campaign.v6.json"
    runtime_root = tmp_path / "runtime"
    arguments = {
        "contract": _automatic_successor_contract(
            contract_path=contract_path,
            runtime_root=runtime_root,
        ),
        "contract_path": contract_path,
        "runtime_root": runtime_root,
        "env_file": tmp_path / "config.env",
        "poll_seconds": 1.0,
        "resume": False,
    }
    assert manifest_generator.launch_successor_once(**arguments) == 1
    (runtime_root / "mode_process_status.json").write_text(
        json.dumps({mode: 0 for mode in campaign_contract.MODES}),
        encoding="utf-8",
    )
    arguments["resume"] = True
    assert manifest_generator.launch_successor_once(**arguments) == 0
    result = runtime_root / "automatic_successor_launch_result.json"
    assert result.is_file()
    assert result.stat().st_mode & 0o222 == 0
    arguments["resume"] = False
    assert manifest_generator.launch_successor_once(**arguments) == 0
    assert len(calls) == 1


def test_launch_lock_prevents_concurrent_submission(
    tmp_path: Path,
):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    lock_path = runtime / "automatic_successor_launch.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        manifest_generator.fcntl.flock(
            lock.fileno(),
            manifest_generator.fcntl.LOCK_EX
            | manifest_generator.fcntl.LOCK_NB,
        )
        with pytest.raises(
            manifest_generator.ManifestGenerationError,
            match="already active",
        ):
            manifest_generator.launch_successor_once(
                contract=_automatic_successor_contract(
                    contract_path=tmp_path / "campaign.v6.json",
                    runtime_root=runtime,
                ),
                contract_path=tmp_path / "campaign.v6.json",
                runtime_root=runtime,
                env_file=tmp_path / "config.env",
                poll_seconds=1.0,
                resume=False,
            )
    assert not (runtime / "automatic_successor_launch_claim.json").exists()


def test_local_seal_is_revalidated_before_launch(
    contract,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = contract["runtime"]

    def fake_git(repository: Path, *arguments: str) -> str:
        if arguments == ("status", "--porcelain"):
            return ""
        if Path(repository).resolve() == Path(runtime["sdk_dir"]).resolve():
            return runtime["sdk_commit"]
        if (
            Path(repository).resolve()
            == Path(runtime["skills_repository"]).resolve()
        ):
            return runtime["skills_commit"]
        return runtime["source_commit"]

    monkeypatch.setattr(run_campaign, "_git", fake_git)
    evidence = run_campaign.verify_local_contract(contract)
    assert evidence["source_commit"] == runtime["source_commit"]
    assert evidence["artifacts"]["wheel"]["sha256"] == (
        manifest_generator.EXPECTED_WHEEL_SHA256
    )

    changed = copy.deepcopy(contract)
    changed["runtime"]["wheel_sha256"] = "0" * 64
    with pytest.raises(run_campaign.CampaignExecutionError):
        run_campaign.verify_local_contract(changed)


def test_archive_order_cannot_enter_campaign_search_contract(contract):
    modes = contract["modes"]
    assert [item["mode"] for item in modes] == [
        "accuracy",
        "latency",
        "multi_objective",
    ]
    assert all(item["initial_observation_ids"] == [] for item in modes)
    assert len(
        {item["observation_namespace"] for item in modes}
    ) == 3
    assert contract["execution"]["shared_archive"] is False


def test_contract_integrity_rejects_mutation(contract):
    changed = copy.deepcopy(contract)
    changed["execution"]["gpus_per_child"] = 1
    with pytest.raises(campaign_contract.CampaignContractError):
        campaign_contract.validate_contract(changed)

    changed = copy.deepcopy(contract)
    changed["qualification_policy"]["recipe_fidelity"][
        "learning_rate"
    ] = 2.0e-4
    changed["contract_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in changed.items()
            if key != "contract_sha256"
        }
    )
    with pytest.raises(
        campaign_contract.CampaignContractError,
        match="qualification v5 fidelity or provenance changed",
    ):
        campaign_contract.validate_contract(changed)


def test_qualification_plan_contains_every_official_arm_without_fallback(
    contract,
):
    plan = qualification_campaign.qualification_plan(contract)
    assert plan["workflow_count"] == 13
    assert plan["schema_version"] == 2
    assert plan["qualification_revision"] == 5
    assert plan["workflow"] == (
        "selective_v4_terminal_train_reuse_or_fresh_full_voc2012_"
        "50_epoch_train_then_new_standalone_full_validation"
    )
    assert plan["new_full_train_job_count"] == 9
    assert plan["reused_terminal_train_phase_count"] == 4
    assert plan["new_standalone_evaluation_job_count"] == 13
    assert plan["recipe_fidelity"] == (
        campaign_contract.FROZEN_QUALIFICATION_FIDELITY
    )
    assert plan["runtime_overlay"] == (
        campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
    )
    assert plan["checkpoint_ids"] == [
        item["id"]
        for item in campaign_contract.segformer_registry_snapshot()[
            "records"
        ]
    ]
    assert plan["all_workflows_independent"] is True
    assert plan[
        "all_workflows_submitted_without_result_driven_exclusion"
    ] is True
    assert plan["terminal_failures_preserved"] is True
    assert plan["replacement_workflows_submitted"] is False
    assert plan["resources_per_job"] == {
        "nodes": 1,
        "gpus": 8,
        "gpu": "NVIDIA A100-SXM4-80GB",
        "partition": "polar3",
        "time_hours": 4.0,
        "container": campaign_contract.FROZEN_SQSH["path"],
    }
    assert plan["cpu_model_runs"] == 0
    assert plan["smoke_model_runs"] == 0
    assert plan["mini_step_runs"] == 0


def test_qualification_specs_bind_only_the_registered_checkpoint_target(
    contract,
):
    stage = qualification_campaign.validate_stage_manifest(
        _fake_qualification_stage(contract),
        contract=contract,
    )
    target_counts = {
        "train.pretrained_model_path": 0,
        "model.backbone.pretrained_backbone_path": 0,
    }
    for row in stage["ptms"]:
        target_counts[row["checkpoint_target"]] += 1
        checkpoint = row["checkpoint"]["path"]
        train = row["specs"]["train"]["document"]
        train_ptm = train["train"]["pretrained_model_path"]
        backbone_ptm = train["model"]["backbone"][
            "pretrained_backbone_path"
        ]
        if row["checkpoint_target"] == "train.pretrained_model_path":
            assert train_ptm == checkpoint
            assert backbone_ptm == ""
        else:
            assert train_ptm == ""
            assert backbone_ptm == checkpoint
        assert train["train"]["num_epochs"] == 50
        assert train["train"]["checkpoint_interval"] == 50
        assert train["train"]["validation_interval"] == 1
        assert train["train"]["num_gpus"] == 8
        assert train["train"]["optim"]["lr"] == 1.0e-4
        assert train["train"]["optim"]["weight_decay"] == 5.0e-4
        assert train["train"]["use_distributed_sampler"] is True
        assert train["dataset"]["segment"]["augmentation"][
            "random_color"
        ]["enable"] is False
        assert train["dataset"]["segment"]["augmentation"][
            "with_random_blur"
        ] is False
        assert train["dataset"]["segment"]["root_dir"] == (
            contract["dataset"]["prepared_root"]
        )
        evaluate = row["specs"]["evaluate"]["document"]
        assert evaluate["evaluate"]["checkpoint"] == (
            qualification_campaign.EVALUATION_CHECKPOINT_SENTINEL
        )
        assert evaluate["train"]["num_epochs"] == 50
        assert evaluate["train"]["optim"]["lr"] == 1.0e-4
        assert evaluate["train"]["optim"]["weight_decay"] == 5.0e-4
        assert evaluate["train"]["use_distributed_sampler"] is True
        assert evaluate["dataset"]["segment"]["augmentation"][
            "random_color"
        ]["enable"] is False
        assert evaluate["dataset"]["segment"]["augmentation"][
            "with_random_blur"
        ] is False
    assert target_counts == {
        "train.pretrained_model_path": 4,
        "model.backbone.pretrained_backbone_path": 9,
    }


def test_ptm_stage_manifest_rejects_missing_or_writable_inputs(contract):
    stage = _fake_qualification_stage(contract)
    qualification_campaign.validate_stage_manifest(stage, contract=contract)

    missing = copy.deepcopy(stage)
    missing["ptms"].pop()
    missing["stage_manifest_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in missing.items()
            if key != "stage_manifest_sha256"
        }
    )
    with pytest.raises(
        qualification_campaign.CampaignExecutionError,
        match="campaign contract changed",
    ):
        qualification_campaign.validate_stage_manifest(
            missing,
            contract=contract,
        )

    writable = copy.deepcopy(stage)
    writable["ptms"][0]["checkpoint"]["mode"] = "644"
    writable["stage_manifest_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in writable.items()
            if key != "stage_manifest_sha256"
        }
    )
    with pytest.raises(
        qualification_campaign.CampaignExecutionError,
        match="checkpoint identity differs",
    ):
        qualification_campaign.validate_stage_manifest(
            writable,
            contract=contract,
        )


def test_completion_exactly_round_trips_through_qualification_gate(
    contract,
    tmp_path: Path,
):
    stage = _fake_qualification_stage(contract)
    Path(
        contract["qualification_policy"]["ptm_stage_manifest_path"]
    ).write_text(json.dumps(stage), encoding="utf-8")
    success_id = stage["ptms"][0]["checkpoint_id"]
    for row in stage["ptms"]:
        path = (
            tmp_path
            / "workflows"
            / row["workflow_id"]
            / "workflow_completion.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        workflow = _workflow(
            row["checkpoint_id"],
            success=row["checkpoint_id"] == success_id,
        )
        workflow["source_checkpoint"] = copy.deepcopy(
            row["checkpoint"]
        )
        if (
            workflow["status"] == "success"
            and workflow["train"]["status_evidence"][
                "pretrained_load"
            ]["schema_version"]
            == 1
        ):
            report = workflow["train"]["status_evidence"][
                "pretrained_load"
            ]
            report["checkpoint"] = row["checkpoint"]["path"]
            report["report_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in report.items()
                    if key
                    not in {"status_record_occurrences", "report_sha256"}
                }
            )
        workflow["workflow_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in workflow.items()
                if key != "workflow_sha256"
            }
        )
        path.write_text(json.dumps(workflow), encoding="utf-8")
    completion = qualification_campaign.build_completion(
        contract=contract,
        stage=stage,
        runtime_root=tmp_path,
        exit_codes={
            row["checkpoint_id"]: 0 for row in stage["ptms"]
        },
    )
    output = tmp_path / "qualification.json"
    output.write_text(json.dumps(completion), encoding="utf-8")
    decision = audit_qualification(output)
    assert completion["all_official_arms_attempted"] is True
    assert len(completion["workflows"]) == 13
    assert all(
        item["terminal"] is True for item in completion["workflows"]
    )
    assert not any(
        item["code"] == "invalid_success_evidence"
        for item in decision.blockers
    )
    if load_ptm_registry().checkpoint(success_id)["status"] == "unverified":
        assert any(
            item["checkpoint_id"] == success_id
            and item["code"] == "registry_not_supported"
            for item in decision.blockers
        )
    assert len(decision.exclusions) == 12


def test_qualification_handoff_is_automatic_but_never_promotes_registry(
    contract,
    tmp_path: Path,
):
    stage = _fake_qualification_stage(contract)
    Path(
        contract["qualification_policy"]["ptm_stage_manifest_path"]
    ).write_text(json.dumps(stage), encoding="utf-8")
    for row in stage["ptms"]:
        path = (
            tmp_path
            / "workflows"
            / row["workflow_id"]
            / "workflow_completion.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        workflow = _workflow(row["checkpoint_id"], success=False)
        workflow["source_checkpoint"] = copy.deepcopy(
            row["checkpoint"]
        )
        if (
            workflow["status"] == "success"
            and workflow["train"]["status_evidence"][
                "pretrained_load"
            ]["schema_version"]
            == 1
        ):
            report = workflow["train"]["status_evidence"][
                "pretrained_load"
            ]
            report["checkpoint"] = row["checkpoint"]["path"]
            report["report_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in report.items()
                    if key
                    not in {"status_record_occurrences", "report_sha256"}
                }
            )
        workflow["workflow_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in workflow.items()
                if key != "workflow_sha256"
            }
        )
        path.write_text(json.dumps(workflow), encoding="utf-8")
    completion = qualification_campaign.build_completion(
        contract=contract,
        stage=stage,
        runtime_root=tmp_path,
        exit_codes={
            row["checkpoint_id"]: 1 for row in stage["ptms"]
        },
    )
    qualification_path = tmp_path / "qualification.json"
    qualification_path.write_text(
        json.dumps(completion),
        encoding="utf-8",
    )
    handoff = qualification_campaign.build_handoff(
        contract=contract,
        completion=completion,
        qualification_path=qualification_path,
    )
    assert handoff["automatic"] is True
    assert handoff["manual_confirmation_required"] is False
    assert handoff["registry_mutated"] is False
    assert handoff["registry_bypass_allowed"] is False
    assert handoff["fallback_ptm_selected"] is False
    assert handoff["failed_workflow_replaced"] is False
    assert handoff["status"] == "terminal_no_successful_ptm"


def test_independent_status_promotion_preserves_pre_promotion_evidence(
    contract,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    stage = _fake_qualification_stage(contract)
    Path(
        contract["qualification_policy"]["ptm_stage_manifest_path"]
    ).write_text(json.dumps(stage), encoding="utf-8")
    success_id = stage["ptms"][0]["checkpoint_id"]
    for row in stage["ptms"]:
        path = (
            tmp_path
            / "workflows"
            / row["workflow_id"]
            / "workflow_completion.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        workflow = _workflow(
            row["checkpoint_id"],
            success=row["checkpoint_id"] == success_id,
        )
        workflow["source_checkpoint"] = copy.deepcopy(
            row["checkpoint"]
        )
        if (
            workflow["status"] == "success"
            and workflow["train"]["status_evidence"][
                "pretrained_load"
            ]["schema_version"]
            == 1
        ):
            report = workflow["train"]["status_evidence"][
                "pretrained_load"
            ]
            report["checkpoint"] = row["checkpoint"]["path"]
            report["report_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in report.items()
                    if key
                    not in {"status_record_occurrences", "report_sha256"}
                }
            )
        workflow["workflow_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in workflow.items()
                if key != "workflow_sha256"
            }
        )
        path.write_text(json.dumps(workflow), encoding="utf-8")
    completion = qualification_campaign.build_completion(
        contract=contract,
        stage=stage,
        runtime_root=tmp_path,
        exit_codes={
            row["checkpoint_id"]: 0 for row in stage["ptms"]
        },
    )
    output = tmp_path / "qualification.json"
    output.write_text(json.dumps(completion), encoding="utf-8")

    promoted_document = load_ptm_registry().to_dict()
    for record in promoted_document["models"]["segformer"][
        "checkpoints"
    ]:
        if record["id"] == success_id:
            record["status"] = "supported"
            record["status_reason"] = "independent full-run review passed"
            record["sha256"] = stage["ptms"][0]["checkpoint"]["sha256"]
            record["compatible_tao_versions"] = ["==7.1.0"]
            record["validation"] = {
                "status": "validated",
                "tao_version": "7.1.0-rc-245",
                "evidence": str(output),
            }

    class PromotedRegistry:
        registry_version = "test-promoted"
        document_sha256 = canonical_sha256(promoted_document)

        def to_dict(self):
            return copy.deepcopy(promoted_document)

        def checkpoint(self, checkpoint_id):
            for model in promoted_document["models"].values():
                for record in model["checkpoints"]:
                    if record["id"] == checkpoint_id:
                        return copy.deepcopy(record)
            raise KeyError(checkpoint_id)

    promoted = PromotedRegistry()
    monkeypatch.setattr(
        qualification_gate,
        "load_ptm_registry",
        lambda: promoted,
    )
    monkeypatch.setattr(
        campaign_contract,
        "load_ptm_registry",
        lambda: promoted,
    )
    decision = qualification_gate.audit_qualification(output)
    assert decision.checkpoint_ids == (success_id,)
    assert decision.blockers == ()
    assert len(decision.exclusions) == 12


def test_direct_qualification_submission_is_pinned_one_node_eight_gpu(
    contract,
):
    calls = []

    class FakeSDK:
        def create_job(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(id="job")

    job, submission = qualification_campaign._submit_job(
        FakeSDK(), contract, "command"
    )
    assert job.id == "job"
    assert submission == {
        "attempt_count": 1,
        "retry_count": 0,
        "transient_failures": [],
        "stable_job_identity_obtained": True,
        "policy_sha256": canonical_sha256(
            campaign_contract.FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY
        ),
    }
    assert calls == [
        {
            "image": campaign_contract.FROZEN_SQSH["path"],
            "command": "command",
            "gpu_count": 8,
            "num_nodes": 1,
            "partition": "polar3",
            "account": "edgeai_tao-ptm_image-foundation-model-clip",
        }
    ]
    guard = qualification_campaign._gpu_guard(
        "segformer train -e {config_path}"
    )
    assert "NVIDIA A100-SXM4-80GB" in guard
    assert "wc -l)\" -eq 8" in guard
    assert "export MASTER_ADDR=127.0.0.1" in guard
    assert "15000 + SLURM_JOB_ID % 10000" in guard
    assert "s.bind" in guard
    assert "segformer train -e {config_path}" in guard
    rendered = guard.format(config_path="/tmp/spec.yaml")
    assert "case \"$SLURM_JOB_ID\"" in rendered
    assert "segformer train -e /tmp/spec.yaml" in rendered


def test_qualification_submission_retries_only_exact_stable_identity_error(
    contract,
    monkeypatch: pytest.MonkeyPatch,
):
    message = campaign_contract.FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY[
        "retryable_submission_message"
    ]
    calls = []
    sleeps = []

    class FakeSDK:
        def create_job(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError(message)
            return SimpleNamespace(id="stable-job")

    monkeypatch.setattr(
        qualification_campaign.time,
        "sleep",
        lambda seconds: sleeps.append(seconds),
    )
    job, evidence = qualification_campaign._submit_job(
        FakeSDK(), contract, "command"
    )

    assert job.id == "stable-job"
    assert len(calls) == 2
    assert sleeps == [10]
    assert evidence["attempt_count"] == 2
    assert evidence["retry_count"] == 1
    assert evidence["transient_failures"] == [
        {
            "attempt": 1,
            "exception_type": "RuntimeError",
            "message": message,
            "classification": "pre_submission_stable_identity_unavailable",
        }
    ]


def test_qualification_submission_retry_is_bounded_and_fail_closed(
    contract,
    monkeypatch: pytest.MonkeyPatch,
):
    message = campaign_contract.FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY[
        "retryable_submission_message"
    ]
    calls = []
    monkeypatch.setattr(
        qualification_campaign.time,
        "sleep",
        lambda _seconds: None,
    )

    class AlwaysUnstable:
        def create_job(self, **_kwargs):
            calls.append(True)
            raise RuntimeError(message)

    with pytest.raises(RuntimeError, match="stable identity"):
        qualification_campaign._submit_job(
            AlwaysUnstable(), contract, "command"
        )
    assert len(calls) == 2

    calls.clear()

    class UnrelatedFailure:
        def create_job(self, **_kwargs):
            calls.append(True)
            raise RuntimeError("unrelated scheduler failure")

    with pytest.raises(RuntimeError, match="unrelated scheduler failure"):
        qualification_campaign._submit_job(
            UnrelatedFailure(), contract, "command"
        )
    assert len(calls) == 1


def test_terminal_infrastructure_retry_requires_exact_owned_marker(contract):
    policy = campaign_contract.FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY

    class FakeSDK:
        def __init__(self, logs):
            self.logs = logs

        def get_job_logs(self, _job_id, tail=None):
            assert tail == 500
            return self.logs

        def get_failure_analysis(self, _job_id):
            return {
                "reason": "infrastructure_failure_pattern",
                "pattern": "CUDA driver.*insufficient",
                "match": policy["sdk_failure_analysis_match"],
                "retriable": True,
            }

    exact = qualification_campaign._terminal_infrastructure_retry_evidence(
        FakeSDK(policy["node_preflight_failure_marker"] + "\n"),
        contract,
        "job",
        "Error",
    )
    assert exact["retry_eligible"] is True
    assert exact["classification"] == (
        "pre_import_cuda_driver_runtime_incompatible"
    )

    for logs in (
        "CUDA driver version is insufficient\n",
        policy["node_preflight_failure_marker"] * 2,
        "model raised CUDA initialization error\n",
    ):
        rejected = (
            qualification_campaign._terminal_infrastructure_retry_evidence(
                FakeSDK(logs), contract, "job", "Error"
            )
        )
        assert rejected["retry_eligible"] is False


def test_phase_retry_preserves_failed_attempt_and_never_replaces_success(
    contract,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    policy = campaign_contract.FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY
    created = []

    class FakeSDK:
        def create_job(self, **_kwargs):
            job = SimpleNamespace(id=f"job-{len(created) + 1}")
            created.append(job)
            return job

        def get_job_results_dir(self, job_id):
            return f"/lustre/results/{job_id}"

        def get_job_logs(self, job_id, tail=None):
            assert tail == 500
            if job_id == "job-1":
                return policy["node_preflight_failure_marker"] + "\n"
            return ""

        def get_failure_analysis(self, job_id):
            if job_id != "job-1":
                return None
            return {
                "reason": "infrastructure_failure_pattern",
                "pattern": "CUDA driver.*insufficient",
                "match": policy["sdk_failure_analysis_match"],
                "retriable": True,
            }

    statuses = iter(("Error", "Complete"))
    monkeypatch.setattr(
        qualification_campaign,
        "_wait_for_job",
        lambda *_args, **_kwargs: next(statuses),
    )
    monkeypatch.setattr(
        qualification_campaign.time,
        "sleep",
        lambda _seconds: None,
    )
    evidence = {"jobs": {}}
    output = tmp_path / "workflow.json"
    events = tmp_path / "events.jsonl"
    job, status = qualification_campaign._run_qualification_job(
        FakeSDK(),
        contract,
        "sealed-command",
        evidence=evidence,
        evidence_path=output,
        events=events,
        checkpoint_id="segformer.test",
        phase="standalone_evaluation",
        job_key="evaluate",
        job_metadata={"command_sha256": "a" * 64},
    )

    assert job.id == "job-2"
    assert status == "Complete"
    assert len(created) == 2
    record = evidence["jobs"]["evaluate"]
    assert record["infrastructure_retry_count"] == 1
    assert [item["status"] for item in record["attempts"]] == [
        "Error",
        "Complete",
    ]
    assert record["attempts"][0]["infrastructure_retry_submitted"] is True
    assert record["attempts"][1]["infrastructure_retry_submitted"] is False


def test_qualification_gpu_guard_exports_usable_allocation_port(
    tmp_path: Path,
):
    nvidia_smi = tmp_path / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *query-gpu=name*) value='NVIDIA A100-SXM4-80GB' ;;\n"
            "  *query-gpu=compute_cap*) value='8.0' ;;\n"
            "  *query-gpu=memory.total*) value='81920' ;;\n"
            "  *query-gpu=driver_version*) value='580.65.06' ;;\n"
        "  *) exit 2 ;;\n"
        "esac\n"
        "i=0; while [ \"$i\" -lt 8 ]; do printf '%s\\n' \"$value\"; "
        "i=$((i + 1)); done\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    python3 = tmp_path / "python3"
    python3.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *cuDriverGetVersion*) exit 0 ;;\n"
        "esac\n"
        f"exec {shlex.quote(os.path.realpath(sys.executable))} \"$@\"\n",
        encoding="utf-8",
    )
    python3.chmod(0o755)
    selected_port = None
    for port in range(
        qualification_campaign.QUALIFICATION_MASTER_PORT_BASE,
        qualification_campaign.QUALIFICATION_MASTER_PORT_BASE
        + qualification_campaign.QUALIFICATION_MASTER_PORT_SPAN,
    ):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        selected_port = port
        break
    assert selected_port is not None
    job_id = str(
        selected_port
        - qualification_campaign.QUALIFICATION_MASTER_PORT_BASE
    )
    guard = qualification_campaign._gpu_guard(
        "printf 'rendezvous=%s:%s\\n' \"$MASTER_ADDR\" "
        "\"$MASTER_PORT\""
    )

    result = subprocess.run(
        ["bash", "-c", guard],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "SLURM_JOB_ID": job_id,
        },
    )

    assert result.stdout == (
        "SEGFORMER_INFRASTRUCTURE_PREFLIGHT_OK "
        "minimum_nvidia_driver_major=580 "
        "minimum_cuda_driver_api_version=13000\n"
        f"rendezvous=127.0.0.1:{selected_port}\n"
    )


def test_qualification_entrypoint_installs_exact_overlay_for_both_actions(
    contract,
):
    overlay = campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
    for action in ("train", "evaluate"):
        command = qualification_campaign._runtime_overlay_install_command(
            contract,
            action_name=action,
        )
        assert overlay["archive_path"] in command
        assert overlay["archive_sha256"] in command
        assert overlay["installer_path"] in command
        assert overlay["installer_sha256"] in command
        assert overlay["receipt_path"] in command
        assert "--expected-sha256" in command
        assert "test -s" in command
        resolved = (
            f"{command} && segformer {action} -e {{config_path}}"
        ).format(config_path="/tmp/spec.yaml")
        assert f"segformer {action} -e /tmp/spec.yaml" in resolved

    changed = copy.deepcopy(contract)
    changed["qualification_policy"]["runtime_overlay"][
        "archive_sha256"
    ] = "0" * 64
    with pytest.raises(
        qualification_campaign.CampaignExecutionError,
        match="runtime overlay is not authorized",
    ):
        qualification_campaign._runtime_overlay_install_command(
            changed,
            action_name="train",
        )


def test_training_status_evidence_counts_one_evaluation_record_per_epoch(
    monkeypatch: pytest.MonkeyPatch,
):
    records = []
    checkpoint_path = "/lustre/staged/segformer.ptm"
    load_report = {
        "checkpoint": checkpoint_path,
        "component": "backbone",
        "loaded_keyset_sha256": "d" * 64,
        "loaded_tensor_count": 365,
        "missing_tensor_count": 0,
        "non_tensor_count": 0,
        "schema_version": 1,
        "shape_mismatched_tensor_count": 0,
        "unmatched_tensor_count": 2,
    }
    records.append(
        {
            "message": (
                qualification_campaign.PRETRAINED_LOAD_REPORT_PREFIX
                + json.dumps(load_report, sort_keys=True, separators=(",", ":"))
            )
        }
    )
    records.append(copy.deepcopy(records[-1]))
    epochs = campaign_contract.FROZEN_QUALIFICATION_TRAINING_EPOCHS
    for epoch in range(epochs):
        metric = 0.10 + epoch / 100
        kpi = {"val_miou": metric}
        records.extend(
            [
                {
                    "message": "Eval metrics generated.",
                    "kpi": copy.deepcopy(kpi),
                },
                {
                    "message": "Training loop in progress",
                    "kpi": copy.deepcopy(kpi),
                },
            ]
        )
    records.append({"message": "Train finished successfully."})
    monkeypatch.setattr(
        qualification_campaign,
        "_status_records",
        lambda *_args, **_kwargs: (
            records,
            {"path": "/immutable/status.json", "record_count": len(records)},
        ),
    )

    evidence = qualification_campaign._training_status_evidence(
        object(),
        "job-id",
        expected_checkpoint_path=checkpoint_path,
        expected_component="backbone",
    )

    assert evidence["validation_record_count"] == epochs
    assert [row["val_miou"] for row in evidence["validation_metrics"]] == [
        0.10 + epoch / 100 for epoch in range(epochs)
    ]
    assert evidence["val_miou"] == pytest.approx(0.59)
    assert evidence["terminal_success"] is True
    assert evidence["pretrained_load"]["loaded_tensor_count"] == 365
    assert evidence["pretrained_load"]["status_record_occurrences"] == 2
    assert evidence["pretrained_load"]["report_sha256"] == canonical_sha256(
        load_report
    )


def test_evaluation_status_deduplicates_identical_semantic_kpi_snapshots(
    monkeypatch: pytest.MonkeyPatch,
):
    kpi = {"test_miou": 0.431, "test_loss": 1.25}
    records = [
        {
            "message": "Test metrics generated.",
            "kpi": copy.deepcopy(kpi),
        },
        {
            "message": "Evaluate finished successfully.",
            "kpi": copy.deepcopy(kpi),
        },
    ]
    monkeypatch.setattr(
        qualification_campaign,
        "_status_records",
        lambda *_args, **_kwargs: (
            records,
            {"path": "/immutable/status.json", "record_count": 2},
        ),
    )

    evidence = qualification_campaign._evaluation_status_evidence(
        object(), "job-id"
    )

    snapshot = {
        "reported_name": "test_miou",
        "test_miou": 0.431,
        "kpi": kpi,
    }
    assert evidence["test_metric_record_count"] == 2
    assert evidence["unique_test_metric_snapshot_count"] == 1
    assert evidence["duplicate_identical_metric_snapshots_allowed"] is True
    assert evidence["metric_snapshot_sha256"] == canonical_sha256(snapshot)
    assert evidence["test_miou"] == pytest.approx(0.431)


@pytest.mark.parametrize(
    "second_kpi",
    [
        {"test_miou": 0.432, "test_loss": 1.25},
        {"test_miou": 0.431, "test_loss": 1.24},
        {"val_miou": 0.431, "test_loss": 1.25},
    ],
)
def test_evaluation_status_rejects_conflicting_kpi_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    second_kpi: dict,
):
    records = [
        {
            "message": "Test metrics generated.",
            "kpi": {"test_miou": 0.431, "test_loss": 1.25},
        },
        {
            "message": "Evaluate finished successfully.",
            "kpi": second_kpi,
        },
    ]
    monkeypatch.setattr(
        qualification_campaign,
        "_status_records",
        lambda *_args, **_kwargs: (records, {"record_count": 2}),
    )

    with pytest.raises(
        qualification_campaign.CampaignExecutionError,
        match="2 unique semantic KPI snapshots",
    ):
        qualification_campaign._evaluation_status_evidence(
            object(), "job-id"
        )


def test_training_status_evidence_rejects_missing_epoch_evaluation_record(
    monkeypatch: pytest.MonkeyPatch,
):
    records = [
        {
            "message": "Eval metrics generated.",
            "kpi": {"val_miou": 0.2},
        }
        for _ in range(
            campaign_contract.FROZEN_QUALIFICATION_TRAINING_EPOCHS - 1
        )
    ]
    records.extend(
        [
            {
                "message": "Training loop in progress",
                "kpi": {"val_miou": 0.2},
            },
            {"message": "Train finished successfully."},
        ]
    )
    monkeypatch.setattr(
        qualification_campaign,
        "_status_records",
        lambda *_args, **_kwargs: (records, {"record_count": len(records)}),
    )

    with pytest.raises(
        qualification_campaign.CampaignExecutionError,
        match="emitted 49 val_miou records; expected 50",
    ):
        qualification_campaign._training_status_evidence(
            object(),
            "job-id",
            expected_checkpoint_path="/lustre/staged/segformer.ptm",
            expected_component="backbone",
        )


@pytest.mark.parametrize(
    ("report_change", "expected"),
    [
        (None, "exactly one unique positive"),
        ({"loaded_tensor_count": 0}, "does not prove a positive load"),
        ({"checkpoint": "/lustre/other.ptm"}, "does not prove a positive load"),
        ({"component": "model"}, "does not prove a positive load"),
    ],
)
def test_training_status_evidence_requires_exact_positive_load_receipt(
    monkeypatch: pytest.MonkeyPatch,
    report_change: dict | None,
    expected: str,
):
    checkpoint_path = "/lustre/staged/segformer.ptm"
    report = {
        "checkpoint": checkpoint_path,
        "component": "backbone",
        "loaded_keyset_sha256": "d" * 64,
        "loaded_tensor_count": 365,
        "missing_tensor_count": 0,
        "non_tensor_count": 0,
        "schema_version": 1,
        "shape_mismatched_tensor_count": 0,
        "unmatched_tensor_count": 2,
    }
    records = [
        {
            "message": "Eval metrics generated.",
            "kpi": {"val_miou": 0.4},
        }
        for _ in range(
            campaign_contract.FROZEN_QUALIFICATION_TRAINING_EPOCHS
        )
    ]
    if report_change is not None:
        report.update(report_change)
        records.insert(
            0,
            {
                "message": (
                    qualification_campaign.PRETRAINED_LOAD_REPORT_PREFIX
                    + json.dumps(report, separators=(",", ":"))
                )
            },
        )
    records.append({"message": "Train finished successfully."})
    monkeypatch.setattr(
        qualification_campaign,
        "_status_records",
        lambda *_args, **_kwargs: (records, {"record_count": len(records)}),
    )

    with pytest.raises(
        qualification_campaign.CampaignExecutionError,
        match=expected,
    ):
        qualification_campaign._training_status_evidence(
            object(),
            "job-id",
            expected_checkpoint_path=checkpoint_path,
            expected_component="backbone",
        )


def test_qualification_gate_rejects_finite_metrics_without_positive_load():
    checkpoint_id = next(
        item["id"]
        for item in campaign_contract.segformer_registry_snapshot()[
            "records"
        ]
        if item["checkpoint_target"]
        == "model.backbone.pretrained_backbone_path"
    )
    workflow = _workflow(checkpoint_id, success=True, metric=0.7)
    report = workflow["train"]["status_evidence"]["pretrained_load"]
    report["loaded_tensor_count"] = 0
    report["report_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in report.items()
            if key not in {"status_record_occurrences", "report_sha256"}
        }
    )
    workflow["workflow_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in workflow.items()
            if key != "workflow_sha256"
        }
    )
    record = load_ptm_registry().checkpoint(checkpoint_id)

    with pytest.raises(
        QualificationGateError,
        match="does not prove a nonzero exact-component load",
    ):
        qualification_gate._successful_workflow(
            workflow,
            checkpoint_id=checkpoint_id,
            registry_record=record,
        )


def test_qualification_terminal_checkpoint_uses_epoch_49_not_search_epoch_9(
    monkeypatch: pytest.MonkeyPatch,
):
    commands = []

    class FakeSDK:
        def get_job_results_dir(self, job_id):
            assert job_id == "train-job"
            return "/lustre/results/train-job"

    def fake_remote_output(command):
        commands.append(command)
        return json.dumps(
            {
                "path": (
                    "/lustre/results/train-job/results_dir/train/"
                    "model_epoch_049_step_09150.pth"
                ),
                "filename": "model_epoch_049_step_09150.pth",
                "size_bytes": 123,
                "sha256": "a" * 64,
            }
        )

    monkeypatch.setattr(
        qualification_campaign,
        "remote_output",
        fake_remote_output,
    )

    evidence = qualification_campaign._qualification_terminal_checkpoint(
        FakeSDK(),
        "train-job",
    )

    assert len(commands) == 1
    assert "model_epoch_049_step_*.pth" in commands[0]
    assert "model_epoch_009" not in commands[0]
    assert evidence["terminal_epoch_index"] == 49
    assert evidence["training_epochs"] == 50
    assert evidence["naming_contract"] == (
        "model_epoch_049_step_numeric"
    )


def test_qualification_terminal_checkpoint_rejects_search_epoch_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeSDK:
        def get_job_results_dir(self, _job_id):
            return "/lustre/results/train-job"

    monkeypatch.setattr(
        qualification_campaign,
        "remote_output",
        lambda _command: json.dumps(
            {
                "path": (
                    "/lustre/results/train-job/results_dir/train/"
                    "model_epoch_009_step_01830.pth"
                ),
                "filename": "model_epoch_009_step_01830.pth",
                "size_bytes": 123,
                "sha256": "a" * 64,
            }
        ),
    )

    with pytest.raises(
        qualification_campaign.CampaignExecutionError,
        match="checkpoint identity is invalid",
    ):
        qualification_campaign._qualification_terminal_checkpoint(
            FakeSDK(),
            "train-job",
        )


def test_qualification_gate_rejects_epoch_9_terminal_checkpoint():
    checkpoint_id = next(
        item["id"]
        for item in campaign_contract.segformer_registry_snapshot()[
            "records"
        ]
        if item["checkpoint_target"]
        == "model.backbone.pretrained_backbone_path"
    )
    workflow = _workflow(checkpoint_id, success=True)
    terminal = workflow["train"]["terminal_checkpoint"]
    terminal.update(
        {
            "path": (
                "/lustre/results/"
                f"{checkpoint_id}/model_epoch_009_step_01830.pth"
            ),
            "training_epochs": 10,
            "terminal_epoch_index": 9,
            "naming_contract": "model_epoch_009_step_numeric",
        }
    )
    workflow["evaluation"]["job"]["checkpoint"] = copy.deepcopy(terminal)
    workflow["workflow_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in workflow.items()
            if key != "workflow_sha256"
        }
    )
    record = load_ptm_registry().checkpoint(checkpoint_id)
    record["sha256"] = "a" * 64

    with pytest.raises(
        QualificationGateError,
        match="terminal checkpoint contract changed",
    ):
        qualification_gate._successful_workflow(
            workflow,
            checkpoint_id=checkpoint_id,
            registry_record=record,
        )


def test_qualification_slurm_preflight_is_read_only_and_job_free(
    contract,
    monkeypatch: pytest.MonkeyPatch,
):
    configured = []
    commands = []
    monkeypatch.setattr(
        run_campaign,
        "configure_slurm_runtime",
        lambda value: configured.append(value["contract_sha256"]),
    )

    def fake_remote_output(command, **_kwargs):
        commands.append(command)
        return "READY\n"

    monkeypatch.setattr(
        qualification_campaign,
        "remote_output",
        fake_remote_output,
    )
    evidence = qualification_campaign.verify_slurm_preflight(contract)
    assert configured == [contract["contract_sha256"]]
    assert len(commands) == 1
    assert "sbatch squeue sacct srun" in commands[0]
    assert "MaxTime=04:00:00" in commands[0]
    assert campaign_contract.FROZEN_SQSH["path"] in commands[0]
    overlay = campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
    assert overlay["archive_path"] in commands[0]
    assert overlay["archive_sha256"] in commands[0]
    assert overlay["installer_path"] in commands[0]
    assert overlay["installer_sha256"] in commands[0]
    assert evidence["status"] == "ready"
    assert evidence["partition"] == "polar3"
    assert evidence["scheduler_jobs_submitted"] == 0
    assert evidence["qualification_runtime_overlay"] == overlay
    assert evidence["sdk_source"].startswith(
        contract["runtime"]["sdk_dir"]
    )


def test_qualification_controller_has_no_local_model_or_smoke_path():
    source = (HERE / "qualification_campaign.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "torch" not in imported
    assert "nvidia_tao_pytorch" not in imported
    assert "torch.load" not in source
    assert "load_smoke" not in source
    assert "mini_step" in source
    assert "scheduler_jobs_submitted\": 0" in source


def test_v5_phase_recovery_partition_and_plan_hashes_are_exact():
    records = qualification_campaign._v4_phase_recovery_records()
    policy = campaign_contract.FROZEN_QUALIFICATION_PHASE_RECOVERY_POLICY
    reused = {
        checkpoint_id
        for checkpoint_id, plan in records.items()
        if plan["mode"] == "reuse_sealed_v4_terminal_train"
    }
    fresh = set(records) - reused
    assert reused == set(
        campaign_contract.FROZEN_V4_REUSABLE_TRAIN_CHECKPOINT_IDS
    )
    assert fresh == set(campaign_contract.FROZEN_V5_FRESH_TRAIN_CHECKPOINT_IDS)
    assert len(reused) == 4
    assert len(fresh) == 9
    assert {
        checkpoint_id: canonical_sha256(plan)
        for checkpoint_id, plan in records.items()
    } == policy["execution_plan_sha256_by_checkpoint_id"]


def test_v5_stage_rejects_resigned_execution_plan_tamper(contract):
    stage = _fake_qualification_stage(contract)
    stage["ptms"][0]["execution_plan"]["new_train_job_required"] = True
    stage["stage_manifest_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in stage.items()
            if key != "stage_manifest_sha256"
        }
    )
    with pytest.raises(
        qualification_campaign.CampaignExecutionError,
        match="checkpoint identity differs",
    ):
        qualification_campaign.validate_stage_manifest(stage, contract=contract)


def test_v5_reused_train_evidence_proves_no_new_job(
    contract,
    monkeypatch: pytest.MonkeyPatch,
):
    row = next(
        item
        for item in _fake_qualification_stage(contract)["ptms"]
        if item["execution_plan"]["mode"]
        == "reuse_sealed_v4_terminal_train"
    )
    plan = row["execution_plan"]
    identities = {
        item["path"]: item
        for item in (
            row["checkpoint"],
            plan["terminal_checkpoint"],
            plan["validation_status_evidence"],
        )
    }

    def fake_identity(path):
        expected = identities[path]
        return {
            "path": path,
            "size_bytes": expected["size_bytes"],
            "sha256": expected["sha256"],
            "mode": expected.get("mode", "644"),
        }

    monkeypatch.setattr(
        qualification_campaign,
        "_remote_identity",
        fake_identity,
    )
    status, checkpoint, job = (
        qualification_campaign._reused_train_phase_evidence(row)
    )
    assert job["new_job_submitted"] is False
    assert job["successful_train_reexecution"] is False
    assert job["tao_job_id"] == plan["train_job"]["tao_job_id"]
    assert job["runtime_overlay"] == plan["predecessor_runtime_overlay"]
    assert checkpoint == plan["terminal_checkpoint"]
    assert status["pretrained_load"] == plan["pretrained_load"]


def test_v5_worker_submits_nine_trains_and_thirteen_new_evaluations(
    contract,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    stage = _fake_qualification_stage(contract)
    stage_path = tmp_path / "stage.json"
    stage_path.write_text(json.dumps(stage), encoding="utf-8")
    monkeypatch.setattr(run_campaign, "load_contract", lambda _path: contract)
    monkeypatch.setattr(
        run_campaign,
        "configure_slurm_runtime",
        lambda _contract: None,
    )

    tao_sdk = ModuleType("tao_sdk")
    tao_sdk.__path__ = []
    tao_sdk.__file__ = str(
        Path(contract["runtime"]["sdk_dir"]) / "tao_sdk/__init__.py"
    )
    platforms = ModuleType("tao_sdk.platforms")
    platforms.__path__ = []
    slurm = ModuleType("tao_sdk.platforms.slurm")

    class FakeSDK:
        def __init__(self, **_kwargs):
            pass

    slurm.SlurmSDK = FakeSDK
    monkeypatch.setitem(sys.modules, "tao_sdk", tao_sdk)
    monkeypatch.setitem(sys.modules, "tao_sdk.platforms", platforms)
    monkeypatch.setitem(sys.modules, "tao_sdk.platforms.slurm", slurm)
    monkeypatch.setattr(
        qualification_campaign,
        "_entrypoint",
        lambda _contract, action, _spec: (action, action[0] * 64),
    )
    submissions = []

    def fake_run_job(
        _sdk,
        _contract,
        _command,
        *,
        evidence,
        checkpoint_id,
        phase,
        job_key,
        job_metadata,
        **_kwargs,
    ):
        submissions.append((checkpoint_id, phase, job_key))
        evidence["jobs"][job_key] = {
            **copy.deepcopy(dict(job_metadata)),
            "status": "Complete",
        }
        return SimpleNamespace(id=f"{checkpoint_id}-{job_key}"), "Complete"

    monkeypatch.setattr(
        qualification_campaign,
        "_run_qualification_job",
        fake_run_job,
    )

    def fake_reuse(row):
        plan = row["execution_plan"]
        status = copy.deepcopy(plan["validation_status_evidence"])
        status["pretrained_load"] = copy.deepcopy(plan["pretrained_load"])
        return status, copy.deepcopy(plan["terminal_checkpoint"]), {
            "status": "Complete",
            "new_job_submitted": False,
            "tao_job_id": plan["train_job"]["tao_job_id"],
        }

    monkeypatch.setattr(
        qualification_campaign,
        "_reused_train_phase_evidence",
        fake_reuse,
    )
    monkeypatch.setattr(
        qualification_campaign,
        "_training_status_evidence",
        lambda *_args, **_kwargs: {
            "validation_record_count": 50,
            "val_miou": 0.5,
            "pretrained_load": {"loaded_tensor_count": 1},
        },
    )
    monkeypatch.setattr(
        qualification_campaign,
        "_qualification_terminal_checkpoint",
        lambda *_args, **_kwargs: {
            "path": "/lustre/results/model_epoch_049_step_1.pth",
            "size_bytes": 1,
            "sha256": "a" * 64,
            "training_epochs": 50,
            "terminal_epoch_index": 49,
            "naming_contract": "model_epoch_049_step_numeric",
            "ambiguity_policy": "fail_closed",
        },
    )
    monkeypatch.setattr(
        qualification_campaign,
        "_evaluation_status_evidence",
        lambda *_args, **_kwargs: {"test_miou": 0.5},
    )
    for row in stage["ptms"]:
        qualification_campaign._run_workflow(
            "contract.json",
            str(stage_path),
            str(tmp_path / "runtime"),
            row["checkpoint_id"],
        )
    train_ids = {
        checkpoint_id
        for checkpoint_id, _phase, job_key in submissions
        if job_key == "train"
    }
    evaluation_ids = {
        checkpoint_id
        for checkpoint_id, _phase, job_key in submissions
        if job_key == "evaluate"
    }
    assert train_ids == set(
        campaign_contract.FROZEN_V5_FRESH_TRAIN_CHECKPOINT_IDS
    )
    assert train_ids.isdisjoint(
        campaign_contract.FROZEN_V4_REUSABLE_TRAIN_CHECKPOINT_IDS
    )
    assert evaluation_ids == set(row["checkpoint_id"] for row in stage["ptms"])
    assert len([item for item in submissions if item[2] == "train"]) == 9
    assert len([item for item in submissions if item[2] == "evaluate"]) == 13


def test_v5_launch_claim_forbids_reentry_and_existing_workflow_state(
    tmp_path: Path,
):
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    claim = qualification_campaign._claim_qualification_launch(
        runtime_root,
        contract_sha256="a" * 64,
        stage_manifest_sha256="b" * 64,
    )
    marker = Path(claim["path"])
    sealed = json.loads(marker.read_text(encoding="utf-8"))
    assert sealed["successful_train_reexecution_allowed"] is False
    supplied = sealed.pop("claim_sha256")
    assert supplied == canonical_sha256(sealed)
    with pytest.raises(
        qualification_campaign.CampaignExecutionError,
        match="already claimed",
    ):
        qualification_campaign._claim_qualification_launch(
            runtime_root,
            contract_sha256="a" * 64,
            stage_manifest_sha256="b" * 64,
        )

    dirty_root = tmp_path / "dirty-runtime"
    workflow = dirty_root / "workflows" / "existing"
    workflow.mkdir(parents=True)
    with pytest.raises(
        qualification_campaign.CampaignExecutionError,
        match="workflow state already exists",
    ):
        qualification_campaign._claim_qualification_launch(
            dirty_root,
            contract_sha256="a" * 64,
            stage_manifest_sha256="b" * 64,
        )


def test_v5_launch_rejects_unsealed_runtime_root_before_any_work(
    contract,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(run_campaign, "load_contract", lambda _path: contract)
    with pytest.raises(
        qualification_campaign.CampaignExecutionError,
        match="runtime root differs from the sealed contract",
    ):
        qualification_campaign.launch(
            contract_path=tmp_path / "contract.json",
            stage_path=tmp_path / "stage.json",
            runtime_root=tmp_path / "alternate-root",
        )
