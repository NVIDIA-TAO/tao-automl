#!/usr/bin/env python3

"""Direct-full-run qualification gate for OneFormer PTM arms.

The user explicitly disallowed CPU/model smokes and mini-steps.  This adapter
therefore accepts only stronger evidence from real one-node/eight-GPU,
full-COCO2017 training plus standalone evaluation.  Evidence never mutates or
bypasses or mutates the repository PTM registry. The versioned successor may
project an exact successful v3 arm to ``supported`` only inside a validated,
campaign-local in-memory registry bound to the sealed completion evidence.
"""

from __future__ import annotations

import copy
import json
import math
import re
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

from . import campaign_contract, ptm_stage
from .campaign_contract import (
    AGENT_FLAGS,
    FROZEN_RUNTIME_OVERLAY,
    FROZEN_SQSH,
    FROZEN_TRAINING_EPOCHS,
    FROZEN_VALIDATION_SANITY_MIN_PQ,
    oneformer_registry_snapshot,
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
    val_pq: float
    test_pq: float
    workflow_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "source_checkpoint_path": self.source_checkpoint_path,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_checkpoint_size_bytes": self.source_checkpoint_size_bytes,
            "terminal_checkpoint_path": self.terminal_checkpoint_path,
            "terminal_checkpoint_sha256": self.terminal_checkpoint_sha256,
            "terminal_checkpoint_size_bytes": self.terminal_checkpoint_size_bytes,
            "val_pq": self.val_pq,
            "test_pq": self.test_pq,
            "workflow_sha256": self.workflow_sha256,
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
            "OneFormer AutoML is fail-closed: "
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


def _validate_workflow_audit(
    workflow: Mapping[str, Any],
    *,
    checkpoint_id: str,
) -> str:
    flags = workflow.get("agent_intervention_flags")
    if (
        not isinstance(flags, Mapping)
        or set(flags) != set(AGENT_FLAGS)
        or any(value is not False for value in flags.values())
    ):
        raise QualificationGateError(
            f"{checkpoint_id} agent-intervention flags are invalid"
        )
    payload = copy.deepcopy(dict(workflow))
    supplied = payload.pop("workflow_sha256", None)
    expected = canonical_sha256(payload)
    if supplied != expected:
        raise QualificationGateError(
            f"{checkpoint_id} workflow integrity failed"
        )
    return expected


def _validate_overlay_receipt(
    value: Any,
    *,
    checkpoint_id: str,
    phase: str,
) -> dict[str, Any]:
    """Validate the immutable receipt captured by one overlaid model job."""
    if not isinstance(value, Mapping):
        raise QualificationGateError(
            f"{checkpoint_id} {phase} runtime-overlay receipt is unavailable"
        )
    receipt = copy.deepcopy(dict(value))
    path = receipt.get("path")
    digest = receipt.get("sha256")
    actions = receipt.get("actions")
    if (
        receipt.get("schema_version")
        != FROZEN_RUNTIME_OVERLAY["receipt_schema_version"]
        or receipt.get("overlay_source_commit")
        != FROZEN_RUNTIME_OVERLAY["source_commit"]
        or receipt.get("container_expected_sha256") != FROZEN_SQSH["sha256"]
        or receipt.get("dry_run") is not False
        or not isinstance(receipt.get("site_packages"), str)
        or not receipt["site_packages"].startswith(
            "/tmp/oneformer-runtime-overlay."
        )
        or not receipt["site_packages"].endswith(
            FROZEN_RUNTIME_OVERLAY["runtime_site_packages_suffix"]
        )
        or receipt.get("base_site_packages")
        != FROZEN_RUNTIME_OVERLAY["base_site_packages"]
        or not isinstance(path, str)
        or not path.startswith("/lustre/")
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or not isinstance(actions, list)
        or len(actions) != FROZEN_RUNTIME_OVERLAY["file_count"]
        or any(
            not isinstance(item, Mapping)
            or item.get("action")
            not in {"replace_base", "already_installed", "install_new"}
            or (
                item.get("base_sha256") is not None
                and re.fullmatch(
                    r"[0-9a-f]{64}", str(item.get("base_sha256"))
                )
                is None
            )
            or not isinstance(item.get("path"), str)
            or re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256")))
            is None
            for item in actions
        )
        or len({item["path"] for item in actions}) != len(actions)
    ):
        raise QualificationGateError(
            f"{checkpoint_id} {phase} runtime-overlay receipt is invalid"
        )
    return receipt


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
    workflow_sha256 = _validate_workflow_audit(
        workflow,
        checkpoint_id=checkpoint_id,
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
        or not isinstance(evaluation, Mapping)
        or evaluation.get("status") != "Complete"
        or evaluation.get("full_validation_split") is not True
        or evaluation.get("nodes") != 1
        or evaluation.get("gpus") != 8
    ):
        raise QualificationGateError(
            f"{checkpoint_id} full train/evaluation contract is incomplete"
        )
    _validate_overlay_receipt(
        train.get("runtime_overlay_receipt"),
        checkpoint_id=checkpoint_id,
        phase="train",
    )
    _validate_overlay_receipt(
        evaluation.get("runtime_overlay_receipt"),
        checkpoint_id=checkpoint_id,
        phase="evaluation",
    )
    checkpoint_path, checkpoint_sha, checkpoint_size = _artifact(
        train.get("terminal_checkpoint"),
        name=f"{checkpoint_id}.terminal_checkpoint",
    )
    val_pq = _metric(train.get("PQ"), f"{checkpoint_id}.PQ")
    test_pq = _metric(
        evaluation.get("test_PQ"),
        f"{checkpoint_id}.test_PQ",
    )
    if (
        val_pq < FROZEN_VALIDATION_SANITY_MIN_PQ
        or test_pq < FROZEN_VALIDATION_SANITY_MIN_PQ
    ):
        raise QualificationGateError(
            f"{checkpoint_id} is below the preregistered 0.01 PQ "
            "experiment sanity gate"
        )
    return QualifiedPTM(
        checkpoint_id=checkpoint_id,
        source_checkpoint_path=source_path,
        source_checkpoint_sha256=source_sha,
        source_checkpoint_size_bytes=source_size,
        terminal_checkpoint_path=checkpoint_path,
        terminal_checkpoint_sha256=checkpoint_sha,
        terminal_checkpoint_size_bytes=checkpoint_size,
        val_pq=val_pq,
        test_pq=test_pq,
        workflow_sha256=workflow_sha256,
    )


def _runtime_local_policy(
    expected_contract: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if expected_contract is None:
        return None
    runtime = expected_contract.get("runtime")
    qualification = expected_contract.get("qualification_policy")
    if not isinstance(runtime, Mapping) or not isinstance(
        qualification, Mapping
    ):
        raise QualificationGateError(
            "sealed runtime-local PTM eligibility policy is unavailable"
        )
    value = runtime.get("runtime_local_eligibility")
    if value != qualification.get("runtime_local_eligibility"):
        raise QualificationGateError(
            "runtime-local PTM eligibility policy differs across contract layers"
        )
    try:
        return campaign_contract.validate_runtime_local_eligibility(
            value,
            runtime=runtime,
            snapshot=oneformer_registry_snapshot(),
        )
    except campaign_contract.CampaignContractError as exc:
        raise QualificationGateError(str(exc)) from exc


def _repository_runtime_eligibility(
    registry: PTMRegistry,
    successful: tuple[QualifiedPTM, ...],
) -> dict[str, Any]:
    records = registry.to_dict()["models"]["oneformer"]["checkpoints"]
    return {
        "schema_version": 1,
        "kind": "repository_supported_registry",
        "scope": "repository_registry",
        "base_registry_version": registry.registry_version,
        "base_registry_sha256": registry.document_sha256,
        "projected_registry_version": registry.registry_version,
        "projected_registry_sha256": registry.document_sha256,
        "qualified_checkpoint_ids": [
            item.checkpoint_id for item in successful
        ],
        "base_record_sha256_by_checkpoint_id": {
            record["id"]: canonical_sha256(record)
            for record in sorted(records, key=lambda item: item["id"])
        },
        "transformations": [],
        "repository_registry_mutated": False,
        "failed_arms_preserved": True,
    }


def _project_runtime_registry(
    *,
    base_registry: PTMRegistry,
    successful: tuple[QualifiedPTM, ...],
    evidence_path: Path,
    evidence_sha256: str,
    policy: Mapping[str, Any] | None,
) -> tuple[PTMRegistry, dict[str, Any]]:
    """Project only exact successful v3 arms into an in-memory registry."""
    if policy is None:
        return (
            base_registry,
            _repository_runtime_eligibility(base_registry, successful),
        )
    base_document = base_registry.to_dict()
    base_model = base_document["models"]["oneformer"]
    base_records = {
        record["id"]: record
        for record in sorted(
            base_model["checkpoints"], key=lambda item: item["id"]
        )
    }
    base_record_hashes = {
        checkpoint_id: canonical_sha256(record)
        for checkpoint_id, record in base_records.items()
    }
    if (
        policy["base_registry_version"] != base_registry.registry_version
        or policy["base_registry_sha256"] != base_registry.document_sha256
        or policy["base_record_sha256_by_checkpoint_id"]
        != base_record_hashes
        or policy["qualification_path"] != str(evidence_path)
        or policy["qualification_file_sha256"]
        != sha256_file(evidence_path)
        or policy["qualification_evidence_sha256"] != evidence_sha256
    ):
        raise QualificationGateError(
            "runtime-local eligibility is not bound to the exact v3 "
            "completion, base registry, and registry records"
        )

    document = copy.deepcopy(base_document)
    document["registry_version"] = (
        f"{base_registry.registry_version}+oneformer-runtime-local-v2"
    )
    projected_records = {
        record["id"]: record
        for record in document["models"]["oneformer"]["checkpoints"]
    }
    transformations = []
    for item in sorted(successful, key=lambda value: value.checkpoint_id):
        record = projected_records[item.checkpoint_id]
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
                "source_checkpoint_sha256": (
                    item.source_checkpoint_sha256
                ),
                "workflow_sha256": item.workflow_sha256,
            }
        )
    projected = PTMRegistry(document)
    if base_registry.document_sha256 != policy["base_registry_sha256"]:
        raise QualificationGateError(
            "base repository registry changed during projection"
        )
    successful_ids = {
        item.checkpoint_id for item in successful
    }
    eligibility = {
        **copy.deepcopy(dict(policy)),
        "projected_registry_version": projected.registry_version,
        "projected_registry_sha256": projected.document_sha256,
        "qualified_checkpoint_ids": sorted(successful_ids),
        "unchanged_checkpoint_ids": sorted(
            set(base_record_hashes) - successful_ids
        ),
        "transformations": transformations,
        "repository_registry_mutated": False,
        "projection_persisted_as_global_registry": False,
        "failed_arms_preserved": True,
        "agent_intervention_flags": {
            name: False for name in AGENT_FLAGS
        },
    }
    eligibility["eligibility_sha256"] = canonical_sha256(eligibility)
    return projected, eligibility


def audit_qualification(
    path: str | Path,
    *,
    expected_contract: Mapping[str, Any] | None = None,
) -> QualificationDecision:
    if expected_contract is not None:
        try:
            expected_contract = campaign_contract.validate_contract(
                expected_contract
            )
        except campaign_contract.CampaignContractError as exc:
            raise QualificationGateError(
                "sealed successor contract is invalid"
            ) from exc
    policy = _runtime_local_policy(expected_contract)
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
    snapshot = oneformer_registry_snapshot()
    registry = load_ptm_registry()
    sealed_stage_by_id: dict[str, Mapping[str, Any]] = {}
    if (
        document.get("schema_version") != 1
        or document.get("model") != "oneformer"
        or document.get("task") != "panoptic_segmentation"
        or document.get("metric") != "PQ"
        or document.get("metric_semantics")
        != "panoptic_quality_from_native_coco_panoptic_annotations"
        or document.get("pq_emitted") is not True
        or document.get("pq_claim_authorized") is not True
        or document.get("registry_sha256") != snapshot["registry_sha256"]
        or document.get("sqsh_sha256") != FROZEN_SQSH["sha256"]
        or document.get("runtime_overlay_sha256")
        != FROZEN_RUNTIME_OVERLAY["archive_sha256"]
        or document.get("runtime_overlay_source_commit")
        != FROZEN_RUNTIME_OVERLAY["source_commit"]
        or document.get("cpu_model_runs") != 0
        or document.get("smoke_model_runs") != 0
        or document.get("mini_step_runs") != 0
    ):
        raise QualificationGateError(
            "qualification campaign identity or execution policy changed"
        )
    if policy is not None and (
        document.get("campaign_id") != policy["qualification_campaign_id"]
        or document.get("qualification_contract_sha256")
        != policy["qualification_contract_sha256"]
        or document.get("qualification_campaign_sha256")
        != policy["qualification_campaign_sha256"]
        or document.get("ptm_stage_manifest_sha256")
        != policy["ptm_stage_manifest_sha256"]
        or document.get("ptm_stage_content_sha256")
        != policy["ptm_stage_content_sha256"]
        or document.get("registry_sha256")
        != policy["base_registry_sha256"]
        or registry.registry_version != policy["base_registry_version"]
        or document.get("replacement_workflows_submitted")
        is not policy.get("replacement_workflows_submitted", False)
        or document.get("replacement_workflow_count", 0)
        != policy.get("replacement_workflow_count", 0)
        or document.get("recovery_checkpoint_ids", [])
        != policy.get("recovery_checkpoint_ids", [])
        or document.get("reused_predecessor_checkpoint_ids", [])
        != policy.get("reused_predecessor_checkpoint_ids", [])
    ):
        raise QualificationGateError(
            "qualification evidence differs from the sealed v3 contract"
        )
    if policy is not None:
        stage_path = Path(policy["ptm_stage_manifest_path"])
        if (
            not stage_path.is_file()
            or sha256_file(stage_path)
            != policy["ptm_stage_manifest_sha256"]
        ):
            raise QualificationGateError(
                "sealed v3 PTM stage manifest is unavailable or changed"
            )
        try:
            stage_document = json.loads(
                stage_path.read_text(encoding="utf-8")
            )
            stage_document = ptm_stage.validate_stage_manifest(
                stage_document,
                registry=registry,
                canonical_root=stage_document.get("publication", {}).get(
                    "canonical_root"
                ),
            )
        except Exception as exc:
            raise QualificationGateError(
                "sealed v3 PTM stage manifest is invalid"
            ) from exc
        if (
            stage_document.get("manifest_sha256")
            != policy["ptm_stage_content_sha256"]
        ):
            raise QualificationGateError(
                "sealed v3 PTM stage content identity changed"
            )
        sealed_stage_by_id = {
            item["id"]: item for item in stage_document["checkpoints"]
        }
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
            if record.get("status") == "unsupported":
                exclusions.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "code": "registry_explicitly_unsupported",
                        "reason": (
                            "The repository explicitly marks this PTM "
                            "unsupported; direct-run evidence cannot promote it"
                        ),
                        "workflow_sha256": item.workflow_sha256,
                        "base_record_sha256": canonical_sha256(record),
                    }
                )
            elif policy is None and record.get("status") != "supported":
                blockers.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "code": "registry_not_supported",
                        "reason": (
                            "Direct full-run success exists, but the exact "
                            "repository record is not independently supported"
                        ),
                    }
                )
            else:
                staged = sealed_stage_by_id.get(checkpoint_id)
                if policy is not None and (
                    not isinstance(staged, Mapping)
                    or staged.get("path") != item.source_checkpoint_path
                    or staged.get("sha256")
                    != item.source_checkpoint_sha256
                    or staged.get("size_bytes")
                    != item.source_checkpoint_size_bytes
                    or staged.get("registry_record_sha256")
                    != policy["base_record_sha256_by_checkpoint_id"][
                        checkpoint_id
                    ]
                ):
                    blockers.append(
                        {
                            "checkpoint_id": checkpoint_id,
                            "code": "qualification_source_not_in_sealed_stage",
                            "reason": (
                                "Direct-run source differs from the exact "
                                "sealed v3 PTM stage record"
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
        try:
            workflow_sha256 = _validate_workflow_audit(
                workflow,
                checkpoint_id=checkpoint_id,
            )
        except QualificationGateError as exc:
            blockers.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "code": "invalid_failure_evidence",
                    "reason": str(exc),
                }
            )
            continue
        exclusion = {
            "checkpoint_id": checkpoint_id,
            "code": workflow.get("failure_code", "direct_full_run_failed"),
            "reason": workflow["failure_reason"],
            "workflow_sha256": workflow_sha256,
        }
        if policy is None and record.get("status") == "supported":
            blockers.append(
                {
                    **exclusion,
                    "code": "supported_registry_record_failed_direct_run",
                }
            )
        else:
            exclusions.append(
                {
                    **exclusion,
                    "base_record_sha256": canonical_sha256(record),
                }
            )

    if not qualified:
        blockers.append(
            {
                "checkpoint_id": None,
                "code": "no_runtime_qualified_ptm",
                "reason": "No exact PTM is runtime eligible",
            }
        )
    projected_registry, runtime_eligibility = _project_runtime_registry(
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
        runtime_registry=projected_registry,
    )


class QualificationLoadEvidence:
    """Production-preflight callback backed by real full GPU workflows."""

    def __init__(self, decision: QualificationDecision):
        decision.assert_runtime_ready()
        self._decision = decision
        self._records = {
            item.checkpoint_id: item for item in decision.qualified
        }
        self._registry = decision.runtime_registry

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
        registry_record = self._registry.checkpoint(request.checkpoint_id)
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
                    "Live artifact or supported registry identity differs "
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
                "projected_registry_sha256": (
                    self._registry.document_sha256
                ),
                "workflow_sha256": record.workflow_sha256,
                "qualified_val_pq": record.val_pq,
                "qualified_test_pq": record.test_pq,
            },
        )


__all__ = [
    "QualificationDecision",
    "QualificationGateError",
    "QualificationLoadEvidence",
    "QualifiedPTM",
    "audit_qualification",
]
