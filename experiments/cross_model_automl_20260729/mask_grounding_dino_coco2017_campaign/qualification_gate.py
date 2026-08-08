#!/usr/bin/env python3

"""Direct-full-run qualification gate for Mask Grounding DINO PTM arms.

The user explicitly disallowed CPU/model smokes and mini-steps.  This adapter
therefore accepts only stronger evidence from real one-node/eight-GPU,
full-COCO2017 training plus standalone evaluation.  Evidence never mutates the
repository PTM registry.  Ordinary callers still require a supported record;
the sealed campaign can additionally authorize an exact, versioned in-memory
registry projection bound to the immutable direct-full-run evidence.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tao_automl.ptm_preflight import (
    CheckpointLoadSmokeRequest,
    CheckpointLoadSmokeResult,
)
from tao_automl.ptm_registry import (
    PTMRegistry,
    canonical_sha256,
    load_ptm_registry,
)

from .campaign_contract import (
    AGENT_FLAGS,
    CHECKPOINT_RESUME_POLICY,
    FROZEN_DDP_STRATEGY_RESOLUTION,
    FROZEN_SQSH,
    FROZEN_TRAINING_EPOCHS,
    FROZEN_VALIDATION_SANITY_MIN_MASK_AP,
    mask_grounding_dino_registry_snapshot,
    sha256_file,
)


class QualificationGateError(RuntimeError):
    """Qualification evidence cannot authorize the campaign."""


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise QualificationGateError(f"{name} must be lowercase SHA-256")
    return value


def _metric(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise QualificationGateError(f"{name} must be finite in [0, 1]")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QualificationGateError(
            f"{name} must be finite in [0, 1]"
        ) from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise QualificationGateError(f"{name} must be finite in [0, 1]")
    return number


@dataclass(frozen=True)
class QualifiedPTM:
    checkpoint_id: str
    source_checkpoint_path: str
    source_checkpoint_sha256: str
    source_checkpoint_size_bytes: int
    terminal_checkpoint_path: str
    terminal_checkpoint_sha256: str
    terminal_checkpoint_size_bytes: int
    val_mask_ap: float | None
    standalone_mask_ap: float
    workflow_sha256: str
    metric_evidence_kind: str = "in_epoch_and_standalone"

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "source_checkpoint_path": self.source_checkpoint_path,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_checkpoint_size_bytes": self.source_checkpoint_size_bytes,
            "terminal_checkpoint_path": self.terminal_checkpoint_path,
            "terminal_checkpoint_sha256": self.terminal_checkpoint_sha256,
            "terminal_checkpoint_size_bytes": self.terminal_checkpoint_size_bytes,
            "val_mask_ap": self.val_mask_ap,
            "standalone_mask_ap": self.standalone_mask_ap,
            "workflow_sha256": self.workflow_sha256,
            "metric_evidence_kind": self.metric_evidence_kind,
        }


@dataclass(frozen=True)
class QualificationDecision:
    evidence_path: str
    evidence_sha256: str
    qualification_campaign_id: str
    qualified: tuple[QualifiedPTM, ...]
    exclusions: tuple[Mapping[str, Any], ...]
    blockers: tuple[Mapping[str, Any], ...]
    runtime_eligibility: Mapping[str, Any]
    decision_sha256: str
    runtime_registry: PTMRegistry = field(repr=False, compare=False)

    @property
    def runtime_ready(self) -> bool:
        return bool(self.qualified) and not self.blockers

    @property
    def checkpoint_ids(self) -> tuple[str, ...]:
        return tuple(item.checkpoint_id for item in self.qualified)

    def stable_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "gate": self.runtime_eligibility["kind"],
            "evidence_path": self.evidence_path,
            "evidence_sha256": self.evidence_sha256,
            "qualification_campaign_id": self.qualification_campaign_id,
            "qualified": [item.to_dict() for item in self.qualified],
            "exclusions": [copy.deepcopy(dict(item)) for item in self.exclusions],
            "blockers": [copy.deepcopy(dict(item)) for item in self.blockers],
            "runtime_eligibility": copy.deepcopy(
                dict(self.runtime_eligibility)
            ),
            "runtime_ready": self.runtime_ready,
            "registry_bypass_allowed": False,
            "repository_registry_mutated": False,
            "cpu_or_smoke_model_job_launched": False,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.stable_dict()
        value["decision_sha256"] = self.decision_sha256
        return value

    def assert_runtime_ready(self) -> None:
        if self.runtime_ready:
            return
        codes = ", ".join(
            f"{item.get('checkpoint_id', 'campaign')}:{item['code']}"
            for item in self.blockers
        )
        raise QualificationGateError(
            "Mask Grounding DINO AutoML is fail-closed: "
            f"{codes or 'no runtime-qualified PTM'}. Direct GPU evidence "
            "cannot mutate or bypass the repository PTM registry."
        )


def _artifact(
    value: Any,
    *,
    name: str,
    expected_size: int | None = None,
) -> tuple[str, str, int]:
    if not isinstance(value, Mapping):
        raise QualificationGateError(f"{name} artifact is unavailable")
    path = value.get("path")
    size = value.get("size_bytes")
    digest = _sha(value.get("sha256"), f"{name}.sha256")
    if (
        not isinstance(path, str)
        or not path.startswith("/lustre/")
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
        or (expected_size is not None and size != expected_size)
    ):
        raise QualificationGateError(f"{name} artifact identity is invalid")
    return path, digest, size


def _successful_workflow(
    workflow: Mapping[str, Any],
    *,
    checkpoint_id: str,
    registry_record: Mapping[str, Any],
) -> QualifiedPTM:
    if (
        workflow.get("checkpoint_id") != checkpoint_id
        or workflow.get("status") != "success"
        or workflow.get("terminal") is not True
        or workflow.get("failure_preserved") is not False
    ):
        raise QualificationGateError(
            f"{checkpoint_id} did not finish qualification successfully"
        )
    flags = workflow.get("agent_intervention_flags")
    if (
        not isinstance(flags, Mapping)
        or set(flags) != set(AGENT_FLAGS)
        or any(value is not False for value in flags.values())
    ):
        raise QualificationGateError(
            f"{checkpoint_id} agent-intervention flags are invalid"
        )
    source_path, source_sha, source_size = _artifact(
        workflow.get("source_checkpoint"),
        name=f"{checkpoint_id}.source_checkpoint",
        expected_size=int(registry_record["expected_size_bytes"]),
    )
    train = workflow.get("train")
    evaluation = workflow.get("evaluation")
    if (
        not isinstance(train, Mapping)
        or train.get("status") != "Complete"
        or train.get("full_dataset") is not True
        or train.get("training_epochs") != FROZEN_TRAINING_EPOCHS
        or train.get("validation_interval") != 1
        or train.get("validation_record_count") != FROZEN_TRAINING_EPOCHS
        or train.get("nodes") != 1
        or train.get("gpus") != 8
        or train.get("distributed_strategy_resolution")
        != FROZEN_DDP_STRATEGY_RESOLUTION
        or not isinstance(evaluation, Mapping)
        or evaluation.get("status") != "Complete"
        or evaluation.get("full_validation_split") is not True
        or evaluation.get("nodes") != 1
        or evaluation.get("gpus") != 8
    ):
        raise QualificationGateError(
            f"{checkpoint_id} full train/evaluation contract is incomplete"
        )
    checkpoint_path, checkpoint_sha, checkpoint_size = _artifact(
        train.get("terminal_checkpoint"),
        name=f"{checkpoint_id}.terminal_checkpoint",
    )
    val_mask_ap = _metric(
        train.get("segm_val_mAP50_95"),
        f"{checkpoint_id}.segm_val_mAP50_95",
    )
    standalone_mask_ap = _metric(
        evaluation.get("segm_val_mAP50_95"),
        f"{checkpoint_id}.standalone.segm_val_mAP50_95",
    )
    if (
        val_mask_ap < FROZEN_VALIDATION_SANITY_MIN_MASK_AP
        or standalone_mask_ap < FROZEN_VALIDATION_SANITY_MIN_MASK_AP
    ):
        raise QualificationGateError(
            f"{checkpoint_id} is below the preregistered 0.05 COCO mask AP50-95 "
            "experiment sanity gate"
        )
    payload = copy.deepcopy(dict(workflow))
    supplied = payload.pop("workflow_sha256", None)
    expected = canonical_sha256(payload)
    if supplied != expected:
        raise QualificationGateError(
            f"{checkpoint_id} workflow integrity failed"
        )
    return QualifiedPTM(
        checkpoint_id=checkpoint_id,
        source_checkpoint_path=source_path,
        source_checkpoint_sha256=source_sha,
        source_checkpoint_size_bytes=source_size,
        terminal_checkpoint_path=checkpoint_path,
        terminal_checkpoint_sha256=checkpoint_sha,
        terminal_checkpoint_size_bytes=checkpoint_size,
        val_mask_ap=val_mask_ap,
        standalone_mask_ap=standalone_mask_ap,
        workflow_sha256=expected,
    )


def _validated_json_with_digest(
    path: Path,
    *,
    digest_field: str,
    name: str,
) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QualificationGateError(f"{name} is unavailable or invalid") from exc
    payload = copy.deepcopy(document)
    supplied = payload.pop(digest_field, None)
    if supplied != canonical_sha256(payload):
        raise QualificationGateError(f"{name} integrity failed")
    return document


def _recovered_workflow(
    *,
    checkpoint_id: str,
    training_workflow: Mapping[str, Any],
    recovery_workflow: Mapping[str, Any],
    registry_record: Mapping[str, Any],
    overlay_sha256: str,
) -> QualifiedPTM:
    """Combine immutable v3 training with its corrected v5 evaluation."""
    training_payload = copy.deepcopy(dict(training_workflow))
    training_sha = training_payload.pop("workflow_sha256", None)
    recovery_payload = copy.deepcopy(dict(recovery_workflow))
    recovery_sha = recovery_payload.pop("workflow_sha256", None)
    diagnostics = training_workflow.get("diagnostics", {})
    train = diagnostics.get("train_job", {})
    failed_evaluation = diagnostics.get("evaluation_job", {})
    evaluation = recovery_workflow.get("evaluation_job", {})
    source_flags = diagnostics.get("agent_intervention_flags", {})
    recovery_flags = recovery_workflow.get("agent_intervention_flags", {})
    if (
        training_sha != canonical_sha256(training_payload)
        or recovery_sha != canonical_sha256(recovery_payload)
        or training_workflow.get("checkpoint_id") != checkpoint_id
        or training_workflow.get("status") != "failure"
        or training_workflow.get("failure_code")
        != "task_correct_metric_missing"
        or training_workflow.get("terminal") is not True
        or training_workflow.get("failure_preserved") is not True
        or train.get("status") != "Complete"
        or train.get("nodes") != 1
        or train.get("gpus") != 8
        or failed_evaluation.get("status") != "Complete"
        or failed_evaluation.get("nodes") != 1
        or failed_evaluation.get("gpus") != 8
        or source_flags != {name: False for name in AGENT_FLAGS}
        or recovery_workflow.get("checkpoint_id") != checkpoint_id
        or recovery_workflow.get("status") != "success"
        or recovery_workflow.get("training_reused") is not True
        or recovery_workflow.get("training_jobs_submitted") != 0
        or recovery_workflow.get("metric_sanity_gate_passed") is not True
        or recovery_workflow.get("source_train_job_id") != train.get("tao_job_id")
        or recovery_workflow.get("failed_evaluation_job_id")
        != failed_evaluation.get("tao_job_id")
        or recovery_flags != {name: False for name in AGENT_FLAGS}
        or evaluation.get("status") != "Complete"
        or evaluation.get("nodes") != 1
        or evaluation.get("gpus") != 8
        or evaluation.get("overlay_sha256") != overlay_sha256
    ):
        raise QualificationGateError(
            f"{checkpoint_id} v3/v5 qualification chain is invalid"
        )
    source_path, source_sha, source_size = _artifact(
        diagnostics.get("source_checkpoint"),
        name=f"{checkpoint_id}.source_checkpoint",
        expected_size=int(registry_record["expected_size_bytes"]),
    )
    checkpoint_path, checkpoint_sha, checkpoint_size = _artifact(
        train.get("terminal_checkpoint"),
        name=f"{checkpoint_id}.terminal_checkpoint",
    )
    if recovery_workflow.get("terminal_checkpoint") != train.get(
        "terminal_checkpoint"
    ):
        raise QualificationGateError(
            f"{checkpoint_id} v5 evaluation changed the v3 terminal checkpoint"
        )
    standalone_mask_ap = _metric(
        recovery_workflow.get("segm_val_mAP50_95"),
        f"{checkpoint_id}.recovered.segm_val_mAP50_95",
    )
    if standalone_mask_ap < FROZEN_VALIDATION_SANITY_MIN_MASK_AP:
        raise QualificationGateError(
            f"{checkpoint_id} is below the preregistered 0.05 COCO mask "
            "AP50-95 experiment sanity gate"
        )
    return QualifiedPTM(
        checkpoint_id=checkpoint_id,
        source_checkpoint_path=source_path,
        source_checkpoint_sha256=source_sha,
        source_checkpoint_size_bytes=source_size,
        terminal_checkpoint_path=checkpoint_path,
        terminal_checkpoint_sha256=checkpoint_sha,
        terminal_checkpoint_size_bytes=checkpoint_size,
        val_mask_ap=None,
        standalone_mask_ap=standalone_mask_ap,
        workflow_sha256=canonical_sha256(
            {
                "training_workflow_sha256": training_sha,
                "recovery_workflow_sha256": recovery_sha,
            }
        ),
        metric_evidence_kind="v3_training_plus_v5_standalone_recovery",
    )


def _runtime_local_policy(
    expected_contract: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if expected_contract is None:
        return None
    value = expected_contract.get("qualification_policy", {}).get(
        "runtime_local_eligibility"
    )
    if not isinstance(value, Mapping) or value.get("enabled") is not True:
        return None
    required_false = (
        "repository_registry_mutation_allowed",
        "failed_arm_promotion_allowed",
        "unsupported_arm_promotion_allowed",
        "agent_override_allowed",
    )
    if (
        value.get("schema_version") != 2
        or value.get("kind")
        != "direct_full_gpu_qualification_runtime_local_v2"
        or value.get("scope") != "campaign_local_in_memory_projection"
        or value.get("model") != "mask_grounding_dino"
        or value.get("task")
        != "category_prompted_grounded_instance_segmentation"
        or value.get("tao_version") != "7.1.0"
        or value.get("container_sha256") != FROZEN_SQSH["sha256"]
        or any(value.get(name) is not False for name in required_false)
    ):
        raise QualificationGateError(
            "sealed runtime-local PTM eligibility policy is invalid"
        )
    for name in (
        "base_registry_sha256",
        "qualification_file_sha256",
        "qualification_evidence_sha256",
        "qualification_contract_sha256",
        "qualification_campaign_sha256",
        "wheel_sha256",
    ):
        _sha(value.get(name), f"runtime_local_eligibility.{name}")
    for name in ("eligibility_source_commit", "sdk_commit", "skills_commit"):
        commit = value.get(name)
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise QualificationGateError(
                f"runtime_local_eligibility.{name} must be a Git commit"
            )
    if value.get("qualification_successor_version") == 5:
        for name in (
            "qualification_contract_file_sha256",
            "training_qualification_file_sha256",
            "training_qualification_evidence_sha256",
            "training_qualification_contract_file_sha256",
            "training_qualification_contract_sha256",
            "ptm_stage_manifest_sha256",
            "ptm_stage_content_sha256",
            "qualification_source_wheel_sha256",
            "metric_recovery_overlay_sha256",
        ):
            _sha(value.get(name), f"runtime_local_eligibility.{name}")
        for name in (
            "qualification_contract_path",
            "training_qualification_path",
            "training_qualification_contract_path",
            "ptm_stage_manifest_path",
        ):
            path = value.get(name)
            if not isinstance(path, str) or not Path(path).is_absolute():
                raise QualificationGateError(
                    f"runtime_local_eligibility.{name} must be absolute"
                )
        for name in (
            "qualification_source_commit",
            "qualification_source_sdk_commit",
            "qualification_source_skills_commit",
            "metric_recovery_source_commit",
        ):
            commit = value.get(name)
            if (
                not isinstance(commit, str)
                or len(commit) != 40
                or any(
                    character not in "0123456789abcdef"
                    for character in commit
                )
            ):
                raise QualificationGateError(
                    f"runtime_local_eligibility.{name} must be a Git commit"
                )
        if (
            value.get("evaluation_recovery_jobs_submitted") != 4
            or value.get("training_jobs_submitted") != 0
            or value.get("replacement_workflows_submitted") is not True
            or value.get("replacement_workflow_count") != 4
            or value.get("checkpoint_resume_policy")
            != CHECKPOINT_RESUME_POLICY
            or not isinstance(
                value.get("predecessor_failure_evidence"), Mapping
            )
        ):
            raise QualificationGateError(
                "sealed v5 evaluation-recovery policy is invalid"
            )
    return value


def _project_runtime_registry(
    *,
    base_registry: PTMRegistry,
    successful: tuple[QualifiedPTM, ...],
    evidence_path: Path,
    evidence_sha256: str,
    policy: Mapping[str, Any] | None,
) -> tuple[PTMRegistry, dict[str, Any]]:
    """Create a validated in-memory projection for exact successful arms.

    The repository registry object and file remain unchanged.  Without the
    explicit schema-v2 campaign policy, only already-supported records are
    runtime eligible, preserving the ordinary production trust boundary.
    """
    base_sha = base_registry.document_sha256
    base_version = base_registry.registry_version
    base_model_records = base_registry.to_dict()["models"][
        "mask_grounding_dino"
    ]["checkpoints"]
    base_record_hashes = {
        record["id"]: canonical_sha256(record)
        for record in sorted(base_model_records, key=lambda value: value["id"])
    }
    if policy is None:
        return base_registry, {
            "schema_version": 1,
            "kind": "repository_supported_registry",
            "scope": "repository_registry",
            "base_registry_version": base_version,
            "base_registry_sha256": base_sha,
            "projected_registry_version": base_version,
            "projected_registry_sha256": base_sha,
            "qualified_checkpoint_ids": [
                item.checkpoint_id for item in successful
            ],
            "base_record_sha256_by_checkpoint_id": base_record_hashes,
            "transformations": [],
            "repository_registry_mutated": False,
            "failed_arms_preserved": True,
        }

    if (
        policy["base_registry_sha256"] != base_sha
        or policy.get("base_registry_version") != base_version
        or policy["qualification_file_sha256"]
        != sha256_file(evidence_path)
        or policy["qualification_evidence_sha256"] != evidence_sha256
    ):
        raise QualificationGateError(
            "runtime-local eligibility is not bound to the exact base "
            "registry and qualification evidence"
        )

    document = base_registry.to_dict()
    document["registry_version"] = (
        f"{base_version}+mask-grounding-dino-runtime-local-v2"
    )
    records = {
        record["id"]: record
        for record in document["models"]["mask_grounding_dino"][
            "checkpoints"
        ]
    }
    transformations = []
    for item in sorted(successful, key=lambda value: value.checkpoint_id):
        record = records[item.checkpoint_id]
        original = copy.deepcopy(record)
        if record["status"] == "unsupported":
            raise QualificationGateError(
                f"{item.checkpoint_id} is explicitly unsupported and cannot "
                "enter a runtime-local projection"
            )
        if record.get("sha256") not in (
            None,
            item.source_checkpoint_sha256,
        ):
            raise QualificationGateError(
                f"{item.checkpoint_id} registry/source checksum mismatch"
            )
        record["status"] = "supported"
        record.pop("status_reason", None)
        record["sha256"] = item.source_checkpoint_sha256
        record["validation"] = {
            "status": "validated",
            "tao_version": policy["tao_version"],
            "container_identity": (
                "sqsh-sha256:" + policy["container_sha256"]
            ),
            "evidence": (
                f"{evidence_path}#evidence_sha256={evidence_sha256};"
                f"workflow_sha256={item.workflow_sha256}"
            ),
        }
        transformations.append(
            {
                "checkpoint_id": item.checkpoint_id,
                "action": (
                    "retain_supported_identity"
                    if original["status"] == "supported"
                    else "qualify_exact_unverified_identity"
                ),
                "base_status": original["status"],
                "projected_status": "supported",
                "base_record_sha256": canonical_sha256(original),
                "projected_record_sha256": canonical_sha256(record),
                "source_checkpoint_sha256": item.source_checkpoint_sha256,
                "workflow_sha256": item.workflow_sha256,
            }
        )

    projected = PTMRegistry(document)
    if base_registry.document_sha256 != base_sha:
        raise QualificationGateError(
            "base repository registry changed during projection"
        )
    eligibility = {
        **copy.deepcopy(dict(policy)),
        "schema_version": 2,
        "kind": "direct_full_gpu_qualification_runtime_local_v2",
        "scope": "campaign_local_in_memory_projection",
        "model": "mask_grounding_dino",
        "task": "category_prompted_grounded_instance_segmentation",
        "tao_version": policy["tao_version"],
        "container_sha256": policy["container_sha256"],
        "qualification_path": str(evidence_path),
        "qualification_file_sha256": sha256_file(evidence_path),
        "qualification_evidence_sha256": evidence_sha256,
        "qualification_contract_sha256": policy[
            "qualification_contract_sha256"
        ],
        "qualification_campaign_sha256": policy[
            "qualification_campaign_sha256"
        ],
        "eligibility_source_commit": policy["eligibility_source_commit"],
        "wheel_sha256": policy["wheel_sha256"],
        "sdk_commit": policy["sdk_commit"],
        "skills_commit": policy["skills_commit"],
        "base_registry_version": base_version,
        "base_registry_sha256": base_sha,
        "projected_registry_version": projected.registry_version,
        "projected_registry_sha256": projected.document_sha256,
        "qualified_checkpoint_ids": [
            item.checkpoint_id
            for item in sorted(successful, key=lambda value: value.checkpoint_id)
        ],
        "base_record_sha256_by_checkpoint_id": base_record_hashes,
        "unchanged_checkpoint_ids": sorted(
            set(base_record_hashes)
            - {item.checkpoint_id for item in successful}
        ),
        "transformations": transformations,
        "repository_registry_mutated": False,
        "projection_persisted_as_global_registry": False,
        "failed_arms_preserved": True,
        "failed_arm_promotion_allowed": False,
        "unsupported_arm_promotion_allowed": False,
        "agent_override_allowed": False,
        "agent_intervention_flags": {
            name: False for name in AGENT_FLAGS
        },
    }
    eligibility["eligibility_sha256"] = canonical_sha256(eligibility)
    return projected, eligibility


def _audit_v5_recovery(
    *,
    evidence_path: Path,
    document: Mapping[str, Any],
    supplied_sha: str,
    expected_contract: Mapping[str, Any],
    policy: Mapping[str, Any],
    registry: PTMRegistry,
    snapshot: Mapping[str, Any],
) -> QualificationDecision:
    """Audit evaluator-only recovery without weakening v3 training provenance."""
    contract_path = Path(policy["qualification_contract_path"])
    training_path = Path(policy["training_qualification_path"])
    training_contract_path = Path(
        policy["training_qualification_contract_path"]
    )
    stage_path = Path(policy["ptm_stage_manifest_path"])
    if (
        policy["qualification_file_sha256"] != sha256_file(evidence_path)
        or policy["qualification_evidence_sha256"] != supplied_sha
        or expected_contract.get("runtime", {}).get(
            "qualification_contract_path"
        )
        != str(contract_path)
        or expected_contract.get("runtime", {}).get(
            "qualification_contract_file_sha256"
        )
        != policy["qualification_contract_file_sha256"]
    ):
        raise QualificationGateError(
            "v5 recovery is not bound to the sealed campaign contract"
        )
    for path, expected_sha, name in (
        (
            contract_path,
            policy["qualification_contract_file_sha256"],
            "v5 qualification contract",
        ),
        (
            training_path,
            policy["training_qualification_file_sha256"],
            "v3 training completion",
        ),
        (
            training_contract_path,
            policy["training_qualification_contract_file_sha256"],
            "v3 training contract",
        ),
        (
            stage_path,
            policy["ptm_stage_manifest_sha256"],
            "PTM stage manifest",
        ),
    ):
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise QualificationGateError(f"sealed {name} changed")
    recovery_contract = _validated_json_with_digest(
        contract_path,
        digest_field="contract_sha256",
        name="v5 qualification contract",
    )
    training = _validated_json_with_digest(
        training_path,
        digest_field="evidence_sha256",
        name="v3 training completion",
    )
    training_contract = _validated_json_with_digest(
        training_contract_path,
        digest_field="contract_sha256",
        name="v3 training contract",
    )
    predecessor = recovery_contract.get("predecessor", {})
    recovery_execution = recovery_contract.get("execution", {})
    training_policy = training_contract.get("qualification_policy", {})
    if (
        recovery_contract.get("contract_sha256")
        != policy["qualification_contract_sha256"]
        or recovery_contract.get("campaign_id")
        != policy["qualification_campaign_id"]
        or recovery_contract.get("model") != "mask_grounding_dino"
        or recovery_contract.get("task")
        != "category_prompted_grounded_instance_segmentation"
        or recovery_contract.get("primary_metric") != "segm_val_mAP50_95"
        or recovery_contract.get("sqsh") != FROZEN_SQSH
        or recovery_contract.get("overlay", {}).get("archive_sha256")
        != policy["metric_recovery_overlay_sha256"]
        or recovery_contract.get("overlay", {}).get("source_commit")
        != policy["metric_recovery_source_commit"]
        or recovery_execution.get("scope")
        != "standalone_full_validation_only"
        or recovery_execution.get("training_jobs_submitted") != 0
        or recovery_execution.get("evaluation_jobs_expected") != 4
        or recovery_execution.get("nodes_per_job") != 1
        or recovery_execution.get("gpus_per_job") != 8
        or recovery_execution.get("cpu_model_runs") != 0
        or recovery_execution.get("smoke_model_runs") != 0
        or recovery_execution.get("mini_step_runs") != 0
        or recovery_execution.get("selection_invoked") is not False
        or recovery_execution.get("validation_measurements_feed_selection")
        is not False
        or any(recovery_contract.get("agent_intervention_flags", {}).values())
        or predecessor.get("completion_path") != str(training_path)
        or predecessor.get("completion_file_sha256")
        != policy["training_qualification_file_sha256"]
        or predecessor.get("evidence_sha256")
        != policy["training_qualification_evidence_sha256"]
        or predecessor.get("contract_path") != str(training_contract_path)
        or predecessor.get("contract_file_sha256")
        != policy["training_qualification_contract_file_sha256"]
        or predecessor.get("contract_sha256")
        != policy["training_qualification_contract_sha256"]
        or training.get("evidence_sha256")
        != policy["training_qualification_evidence_sha256"]
        or training.get("qualification_contract_sha256")
        != policy["training_qualification_contract_sha256"]
        or training.get("qualification_campaign_sha256")
        != policy["qualification_campaign_sha256"]
        or training.get("registry_sha256") != policy["base_registry_sha256"]
        or training.get("ptm_stage_manifest_path") != str(stage_path)
        or training.get("ptm_stage_manifest_sha256")
        != policy["ptm_stage_manifest_sha256"]
        or training.get("replacement_workflows_submitted") is not True
        or training.get("replacement_workflow_count") != 4
        or training.get("checkpoint_resume_policy")
        != CHECKPOINT_RESUME_POLICY
        or training.get("predecessor_failure_evidence")
        != policy["predecessor_failure_evidence"]
        or training_contract.get("contract_sha256")
        != policy["training_qualification_contract_sha256"]
        or training_contract.get("campaign_id") != training.get("campaign_id")
        or training_contract.get("sqsh") != FROZEN_SQSH
        or training_policy.get("full_dataset") is not True
        or training_policy.get("training_epochs") != FROZEN_TRAINING_EPOCHS
        or training_policy.get("standalone_evaluation") is not True
        or training_policy.get("nodes_per_job") != 1
        or training_policy.get("gpus_per_job") != 8
        or training_policy.get("checkpoint_resume_policy")
        != CHECKPOINT_RESUME_POLICY
    ):
        raise QualificationGateError(
            "v5 recovery or its v3 training provenance changed"
        )
    if (
        document.get("campaign_id") != recovery_contract.get("campaign_id")
        or document.get("contract_sha256")
        != recovery_contract.get("contract_sha256")
        or document.get("model") != "mask_grounding_dino"
        or document.get("task")
        != "category_prompted_grounded_instance_segmentation"
        or document.get("primary_metric") != "segm_val_mAP50_95"
        or document.get("overlay") != recovery_contract.get("overlay")
        or document.get("predecessor") != predecessor
        or document.get("training_jobs_submitted") != 0
        or document.get("evaluation_jobs_submitted") != 4
        or document.get("evaluations_submitted_concurrently") is not True
        or any(document.get(name) != 0 for name in (
            "cpu_model_runs", "smoke_model_runs", "mini_step_runs"
        ))
        or document.get("selection_invoked") is not False
        or document.get("validation_measurements_feed_selection") is not False
        or document.get("agent_intervention_flags")
        != {name: False for name in AGENT_FLAGS}
    ):
        raise QualificationGateError(
            "v5 recovery completion execution policy changed"
        )
    try:
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QualificationGateError("sealed PTM stage is invalid") from exc
    stage_by_id = {
        item.get("id"): item
        for item in stage.get("checkpoints", [])
        if isinstance(item, Mapping)
    }
    training_by_id = {
        item.get("checkpoint_id"): item
        for item in training.get("workflows", [])
        if isinstance(item, Mapping)
    }
    recovery_by_id = {
        item.get("checkpoint_id"): item
        for item in document.get("workflows", [])
        if isinstance(item, Mapping)
    }
    expected_ids = tuple(item["id"] for item in snapshot["records"])
    if (
        set(stage_by_id) != set(expected_ids)
        or set(training_by_id) != set(expected_ids)
        or set(recovery_by_id) != set(expected_ids)
        or len(training.get("workflows", [])) != len(expected_ids)
        or len(document.get("workflows", [])) != len(expected_ids)
    ):
        raise QualificationGateError(
            "v5 recovery must preserve exactly one workflow per official PTM"
        )
    qualified: list[QualifiedPTM] = []
    blockers: list[dict[str, Any]] = []
    for checkpoint_id in expected_ids:
        record = registry.checkpoint(checkpoint_id)
        try:
            item = _recovered_workflow(
                checkpoint_id=checkpoint_id,
                training_workflow=training_by_id[checkpoint_id],
                recovery_workflow=recovery_by_id[checkpoint_id],
                registry_record=record,
                overlay_sha256=policy["metric_recovery_overlay_sha256"],
            )
        except QualificationGateError as exc:
            blockers.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "code": "invalid_v5_recovery_evidence",
                    "reason": str(exc),
                }
            )
            continue
        staged = stage_by_id[checkpoint_id]
        if (
            staged.get("path") != item.source_checkpoint_path
            or staged.get("size_bytes") != item.source_checkpoint_size_bytes
            or staged.get("sha256") != item.source_checkpoint_sha256
        ):
            blockers.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "code": "qualification_source_not_in_sealed_stage",
                    "reason": "v3 source checkpoint differs from sealed PTM stage",
                }
            )
        elif record.get("status") == "unsupported":
            blockers.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "code": "registry_explicitly_unsupported",
                    "reason": "unsupported PTMs cannot enter the projection",
                }
            )
        elif record.get("sha256") not in (
            None,
            item.source_checkpoint_sha256,
        ):
            blockers.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "code": "qualified_checkpoint_registry_sha_mismatch",
                    "reason": "repository and sealed source checksums differ",
                }
            )
        else:
            qualified.append(item)
    if not qualified:
        blockers.append(
            {
                "checkpoint_id": None,
                "code": "no_runtime_qualified_ptm",
                "reason": "no v3/v5 qualification chain passed",
            }
        )
    runtime_registry, runtime_eligibility = _project_runtime_registry(
        base_registry=registry,
        successful=tuple(qualified),
        evidence_path=evidence_path,
        evidence_sha256=supplied_sha,
        policy=policy,
    )
    payload = {
        "evidence_path": str(evidence_path),
        "evidence_sha256": supplied_sha,
        "qualification_campaign_id": document.get("campaign_id"),
        "qualified": [item.to_dict() for item in qualified],
        "exclusions": [],
        "blockers": blockers,
        "runtime_eligibility": runtime_eligibility,
    }
    return QualificationDecision(
        evidence_path=str(evidence_path),
        evidence_sha256=supplied_sha,
        qualification_campaign_id=str(document.get("campaign_id", "")),
        qualified=tuple(qualified),
        exclusions=(),
        blockers=tuple(blockers),
        runtime_eligibility=runtime_eligibility,
        decision_sha256=canonical_sha256(payload),
        runtime_registry=runtime_registry,
    )


def audit_qualification(
    path: str | Path,
    *,
    expected_contract: Mapping[str, Any] | None = None,
) -> QualificationDecision:
    evidence_path = Path(path).resolve()
    if not evidence_path.is_file():
        raise QualificationGateError(
            f"qualification evidence is unavailable: {evidence_path}"
        )
    document = json.loads(evidence_path.read_text(encoding="utf-8"))
    supplied_sha = document.get("evidence_sha256")
    payload = copy.deepcopy(document)
    payload.pop("evidence_sha256", None)
    if supplied_sha != canonical_sha256(payload):
        raise QualificationGateError("qualification evidence integrity failed")
    policy = _runtime_local_policy(expected_contract)
    registry = load_ptm_registry()
    snapshot = mask_grounding_dino_registry_snapshot()
    if policy is not None and policy.get("qualification_successor_version") == 5:
        if expected_contract is None:
            raise QualificationGateError(
                "v5 recovery requires a sealed campaign contract"
            )
        return _audit_v5_recovery(
            evidence_path=evidence_path,
            document=document,
            supplied_sha=supplied_sha,
            expected_contract=expected_contract,
            policy=policy,
            registry=registry,
            snapshot=snapshot,
        )
    evidence_registry_sha = document.get("registry_sha256")
    for name in (
        "qualification_contract_sha256",
        "qualification_campaign_sha256",
        "ptm_stage_manifest_sha256",
    ):
        _sha(document.get(name), name)
    stage_path = document.get("ptm_stage_manifest_path")
    if (
        not isinstance(stage_path, str)
        or not Path(stage_path).is_absolute()
    ):
        raise QualificationGateError(
            "ptm_stage_manifest_path must be absolute"
        )
    sealed_stage_by_id: dict[str, Mapping[str, Any]] = {}
    if expected_contract is not None:
        runtime = expected_contract.get("runtime", {})
        launchers = expected_contract.get("launcher_integrity", {})
        expected_qualification_campaign_sha = (
            policy["qualification_campaign_sha256"]
            if policy is not None
            else launchers.get("qualification_campaign_sha256")
        )
        if (
            document["qualification_campaign_sha256"]
            != expected_qualification_campaign_sha
            or document["ptm_stage_manifest_path"]
            != runtime.get("ptm_stage_manifest_path")
            or document["ptm_stage_manifest_sha256"]
            != runtime.get("ptm_stage_manifest_sha256")
            or document.get("distributed_strategy_resolution")
            != expected_contract.get("qualification_policy", {}).get(
                "distributed_strategy_resolution"
            )
            or document.get("predecessor_failure_evidence")
            != expected_contract.get("qualification_policy", {}).get(
                "predecessor_failure_evidence"
            )
            or (
                policy is not None
                and (
                    document["qualification_contract_sha256"]
                    != policy["qualification_contract_sha256"]
                    or document["registry_sha256"]
                    != policy["base_registry_sha256"]
                    or registry.registry_version
                    != policy["base_registry_version"]
                    or document.get(
                        "replacement_workflows_submitted", False
                    )
                    is not policy.get(
                        "replacement_workflows_submitted", False
                    )
                    or document.get("replacement_workflow_count", 0)
                    != policy.get("replacement_workflow_count", 0)
                    or document.get("checkpoint_resume_policy")
                    != policy.get("checkpoint_resume_policy")
                )
            )
        ):
            raise QualificationGateError(
                "qualification launcher or PTM stage differs from the "
                "sealed final campaign"
            )
        local_stage = Path(stage_path)
        if (
            not local_stage.is_file()
            or sha256_file(local_stage)
            != document["ptm_stage_manifest_sha256"]
        ):
            raise QualificationGateError(
                "sealed PTM stage manifest is unavailable or changed"
            )
        stage_document = json.loads(
            local_stage.read_text(encoding="utf-8")
        )
        stage_records = stage_document.get("checkpoints")
        if not isinstance(stage_records, list):
            raise QualificationGateError(
                "sealed PTM stage records are unavailable"
            )
        sealed_stage_by_id = {
            str(item.get("id")): item
            for item in stage_records
            if isinstance(item, Mapping)
        }
    if (
        document.get("schema_version") != 1
        or document.get("model") != "mask_grounding_dino"
        or document.get("task") != "category_prompted_grounded_instance_segmentation"
        or document.get("primary_metric") != "segm_val_mAP50_95"
        or document.get("VG_overall_iou_accepted_as_mask_ap") is not False
        or not isinstance(evidence_registry_sha, str)
        or len(evidence_registry_sha) != 64
        or any(
            character not in "0123456789abcdef"
            for character in evidence_registry_sha
        )
        or document.get("sqsh_sha256") != FROZEN_SQSH["sha256"]
        or document.get("cpu_model_runs") != 0
        or document.get("smoke_model_runs") != 0
        or document.get("mini_step_runs") != 0
        or not (
            (
                document.get("replacement_workflows_submitted", False)
                is False
                and document.get("replacement_workflow_count", 0) == 0
                and document.get("checkpoint_resume_policy") is None
            )
            or (
                document.get("replacement_workflows_submitted") is True
                and document.get("replacement_workflow_count") == 4
                and document.get("checkpoint_resume_policy")
                == CHECKPOINT_RESUME_POLICY
            )
        )
        or document.get("distributed_strategy_resolution")
        != FROZEN_DDP_STRATEGY_RESOLUTION
        or not isinstance(
            document.get("predecessor_failure_evidence"), Mapping
        )
        or document["predecessor_failure_evidence"].get(
            "all_terminal_failures_preserved"
        )
        is not True
        or document["predecessor_failure_evidence"].get(
            "replacement_submitted"
        )
        is not False
    ):
        raise QualificationGateError(
            "qualification campaign identity or execution policy changed"
        )
    workflows = document.get("workflows")
    if not isinstance(workflows, list):
        raise QualificationGateError("qualification workflows are unavailable")
    by_id = {
        item.get("checkpoint_id"): item
        for item in workflows
        if isinstance(item, Mapping)
    }
    expected_ids = tuple(
        item["id"] for item in snapshot["records"]
    )
    if set(by_id) != set(expected_ids) or len(workflows) != len(expected_ids):
        raise QualificationGateError(
            "qualification must preserve exactly one workflow per official PTM"
        )

    qualified: list[QualifiedPTM] = []
    exclusions: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for checkpoint_id in expected_ids:
        workflow = by_id[checkpoint_id]
        record = registry.checkpoint(checkpoint_id)
        status = workflow.get("status")
        if status == "success":
            try:
                item = _successful_workflow(
                    workflow,
                    checkpoint_id=checkpoint_id,
                    registry_record=record,
                )
            except QualificationGateError as exc:
                blockers.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "code": "invalid_success_evidence",
                        "reason": str(exc),
                    }
                )
                continue
            current_sha = record.get("sha256")
            if (
                current_sha is not None
                and current_sha != item.source_checkpoint_sha256
            ):
                blockers.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "code": "qualified_checkpoint_registry_sha_mismatch",
                        "reason": (
                            "The promoted registry checksum differs from "
                            "the direct-full-run source checkpoint"
                        ),
                    }
                )
                continue
            if expected_contract is not None:
                staged = sealed_stage_by_id.get(checkpoint_id)
                if (
                    not isinstance(staged, Mapping)
                    or staged.get("path")
                    != item.source_checkpoint_path
                    or staged.get("size_bytes")
                    != item.source_checkpoint_size_bytes
                    or staged.get("sha256")
                    != item.source_checkpoint_sha256
                ):
                    blockers.append(
                        {
                            "checkpoint_id": checkpoint_id,
                            "code": "qualification_source_not_in_sealed_stage",
                            "reason": (
                                "Direct-full-run source checkpoint differs "
                                "from the sealed immutable PTM stage"
                            ),
                        }
                    )
                    continue
            if record.get("status") == "unsupported":
                blockers.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "code": "registry_explicitly_unsupported",
                        "reason": (
                            "An explicitly unsupported repository record "
                            "cannot be promoted by campaign-local evidence"
                        ),
                    }
                )
            elif record.get("status") != "supported" and policy is None:
                blockers.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "code": "registry_not_supported",
                        "reason": (
                            "Direct full-run success exists, but neither "
                            "repository support nor an explicit sealed "
                            "runtime-local eligibility policy is available"
                        ),
                    }
                )
            else:
                qualified.append(item)
            continue
        if (
            status != "failure"
            or workflow.get("terminal") is not True
            or workflow.get("failure_preserved") is not True
            or not isinstance(workflow.get("failure_reason"), str)
            or not workflow["failure_reason"].strip()
        ):
            blockers.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "code": "terminal_record_missing",
                    "reason": (
                        "Every unsuccessful official PTM must retain a "
                        "structured terminal failure"
                    ),
                }
            )
            continue
        exclusion = {
            "checkpoint_id": checkpoint_id,
            "code": workflow.get("failure_code", "direct_full_run_failed"),
            "reason": workflow["failure_reason"],
            "workflow_sha256": workflow.get("workflow_sha256"),
        }
        if record.get("status") == "supported":
            blockers.append(
                {
                    **exclusion,
                    "code": "supported_registry_record_failed_direct_run",
                }
            )
        else:
            exclusions.append(exclusion)

    if not qualified:
        blockers.append(
            {
                "checkpoint_id": None,
                "code": "no_runtime_qualified_ptm",
                "reason": (
                    "No exact PTM has successful direct full-run evidence and "
                    "an authorized runtime eligibility path"
                ),
            }
        )
    runtime_registry, runtime_eligibility = _project_runtime_registry(
        base_registry=registry,
        successful=tuple(qualified),
        evidence_path=evidence_path,
        evidence_sha256=supplied_sha,
        policy=policy,
    )
    decision_payload = {
        "evidence_path": str(evidence_path),
        "evidence_sha256": supplied_sha,
        "qualification_campaign_id": document.get("campaign_id"),
        "qualified": [item.to_dict() for item in qualified],
        "exclusions": exclusions,
        "blockers": blockers,
        "runtime_eligibility": runtime_eligibility,
    }
    return QualificationDecision(
        evidence_path=str(evidence_path),
        evidence_sha256=supplied_sha,
        qualification_campaign_id=str(document.get("campaign_id", "")),
        qualified=tuple(qualified),
        exclusions=tuple(exclusions),
        blockers=tuple(blockers),
        runtime_eligibility=runtime_eligibility,
        decision_sha256=canonical_sha256(decision_payload),
        runtime_registry=runtime_registry,
    )


class QualificationLoadEvidence:
    """Production-preflight callback backed by real full GPU workflows."""

    def __init__(self, decision: QualificationDecision):
        decision.assert_runtime_ready()
        self._decision = decision
        self._records = {
            item.checkpoint_id: item for item in decision.qualified
        }

    def __call__(
        self,
        request: CheckpointLoadSmokeRequest,
    ) -> CheckpointLoadSmokeResult:
        record = self._records.get(request.checkpoint_id)
        if record is None:
            return CheckpointLoadSmokeResult(
                ok=False,
                code="qualification_evidence_missing",
                reason="No completed direct full-run qualification exists",
            )
        registry_record = self._decision.runtime_registry.checkpoint(
            request.checkpoint_id
        )
        observed_size = request.checkpoint_path.stat().st_size
        observed_sha = sha256_file(request.checkpoint_path)
        if (
            registry_record.get("status") != "supported"
            or observed_size != record.source_checkpoint_size_bytes
            or observed_sha != record.source_checkpoint_sha256
        ):
            return CheckpointLoadSmokeResult(
                ok=False,
                code="qualification_artifact_or_registry_mismatch",
                reason=(
                    "Live artifact or bound runtime-registry identity differs "
                    "from direct full-run evidence"
                ),
            )
        return CheckpointLoadSmokeResult(
            ok=True,
            code="direct_full_train_eval_qualification_reused",
            reason=(
                "Exact checkpoint passed full-dataset one-node/eight-GPU "
                "training, validation, terminal reload, and standalone eval"
            ),
            details={
                "cpu_or_smoke_model_job_launched": False,
                "qualification_evidence_sha256": (
                    self._decision.evidence_sha256
                ),
                "runtime_eligibility_sha256": self._decision.runtime_eligibility.get(
                    "eligibility_sha256"
                ),
                "projected_registry_sha256": (
                    self._decision.runtime_registry.document_sha256
                ),
                "workflow_sha256": record.workflow_sha256,
                "qualified_val_mask_ap": record.val_mask_ap,
                "qualified_standalone_mask_ap": (
                    record.standalone_mask_ap
                ),
            },
        )


__all__ = [
    "QualificationDecision",
    "QualificationGateError",
    "QualificationLoadEvidence",
    "QualifiedPTM",
    "audit_qualification",
]
