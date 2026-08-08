#!/usr/bin/env python3

"""Fail-closed qualification-to-runtime gate for the RT-DETR campaign.

The four direct RT-DETR workflows are stronger than a synthetic load smoke:
each PTM must finish ten distributed training epochs, ten validations, a
terminal checkpoint, and standalone evaluation.  This module turns that
immutable completion evidence into a load-verification callback for the live
production PTM preflight.  It never changes registry status.  Runtime remains
blocked until the repository-owned registry records are independently promoted
to ``supported`` and a live typed runtime preflight accepts them.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tao_automl.ptm_preflight import (
    CheckpointLoadSmokeRequest,
    CheckpointLoadSmokeResult,
)
from tao_automl.ptm_registry import canonical_sha256, load_ptm_registry


EXPECTED_PTMS = (
    "rtdetr.trafficcam.resnet18.trainable.v2.0",
    "rtdetr.trafficcam.resnet50.trainable.v2.0",
    "rtdetr.warehouse.efficientvit_l2.trainable.v1.0",
    "rtdetr.warehouse.resnet50.trainable.v1.0.2",
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
DEFAULT_QUALIFICATION_ROOT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/rtdetr_qualification_20260730"
)
DEFAULT_COMPLETION = DEFAULT_QUALIFICATION_ROOT / "completion.resume.json"


class QualificationGateError(RuntimeError):
    """Qualification evidence cannot authorize the AutoML successor."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise QualificationGateError(f"{name} must be lowercase SHA-256")
    return value


@dataclass(frozen=True)
class QualifiedPTM:
    """One fully qualified PTM with immutable evidence."""

    checkpoint_id: str
    workflow_id: str
    source_checkpoint_sha256: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    checkpoint_path: str
    map50: float
    map_value: float
    workflow_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "workflow_id": self.workflow_id,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "checkpoint_path": self.checkpoint_path,
            "mAP50": self.map50,
            "mAP": self.map_value,
            "workflow_sha256": self.workflow_sha256,
        }


@dataclass(frozen=True)
class QualificationDecision:
    """Immutable result of the qualification and registry gates."""

    evidence_path: str
    evidence_sha256: str
    qualification_campaign_id: str
    qualified: tuple[QualifiedPTM, ...]
    blockers: tuple[Mapping[str, Any], ...]
    decision_sha256: str

    @property
    def runtime_ready(self) -> bool:
        return not self.blockers and bool(self.qualified)

    @property
    def checkpoint_ids(self) -> tuple[str, ...]:
        return tuple(item.checkpoint_id for item in self.qualified)

    def stable_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "gate": "rtdetr_full_qualification_then_supported_registry_v1",
            "evidence_path": self.evidence_path,
            "evidence_sha256": self.evidence_sha256,
            "qualification_campaign_id": self.qualification_campaign_id,
            "qualified": [item.to_dict() for item in self.qualified],
            "blockers": [copy.deepcopy(dict(item)) for item in self.blockers],
            "runtime_ready": self.runtime_ready,
            "promotion_is_dynamic": False,
            "registry_bypass_allowed": False,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.stable_dict()
        value["decision_sha256"] = self.decision_sha256
        return value

    def assert_runtime_ready(self) -> None:
        if self.runtime_ready:
            return
        codes = ", ".join(
            f"{item['checkpoint_id']}:{item['code']}"
            for item in self.blockers
        )
        raise QualificationGateError(
            "RT-DETR AutoML successor is fail-closed: "
            f"{codes or 'no qualified PTMs'}. Full qualification evidence "
            "cannot mutate or bypass the repository PTM registry."
        )


def _qualified_workflow(
    workflow: Mapping[str, Any],
    *,
    expected_ptm: str,
) -> QualifiedPTM:
    if (
        workflow.get("ptm_id") != expected_ptm
        or workflow.get("status") != "success"
        or workflow.get("terminal") is not True
        or workflow.get("failure_preserved") is not False
    ):
        raise QualificationGateError(
            f"{expected_ptm} did not finish qualification successfully"
        )
    resume = workflow.get("resume")
    if (
        not isinstance(resume, Mapping)
        or resume.get("completed_training_job_reused") is not True
        or resume.get("training_job_submitted") is not False
        or resume.get("selection_or_candidate_change") is not False
        or resume.get("prior_workflow_artifact_modified") is not False
        or resume.get("checkpoint_resolved_after_fix") is not True
    ):
        raise QualificationGateError(
            f"{expected_ptm} evaluation-only resume provenance is invalid"
        )
    flags = workflow.get("agent_intervention_flags")
    if (
        not isinstance(flags, Mapping)
        or set(flags) != set(AGENT_FLAGS)
        or any(value is not False for value in flags.values())
    ):
        raise QualificationGateError(
            f"{expected_ptm} agent-intervention flags are invalid"
        )
    jobs = workflow.get("jobs")
    if not isinstance(jobs, Mapping):
        raise QualificationGateError(f"{expected_ptm} jobs are unavailable")
    train = jobs.get("train")
    evaluation = jobs.get("evaluation")
    if (
        not isinstance(train, Mapping)
        or train.get("status") != "Complete"
        or train.get("full_dataset") is not True
        or train.get("training_epochs") != 10
        or train.get("validation_interval") != 1
        or train.get("nodes") != 1
        or train.get("gpus") != 8
        or not isinstance(evaluation, Mapping)
        or evaluation.get("status") != "Complete"
        or evaluation.get("full_validation_split") is not True
        or evaluation.get("nodes") != 1
        or evaluation.get("gpus") != 8
    ):
        raise QualificationGateError(
            f"{expected_ptm} full train/evaluation contract is incomplete"
        )
    train_status = train.get("status_evidence")
    eval_status = evaluation.get("status_evidence")
    if (
        not isinstance(train_status, Mapping)
        or train_status.get("terminal_success") is not True
        or train_status.get("validation_record_count") != 10
        or not isinstance(eval_status, Mapping)
        or eval_status.get("terminal_success") is not True
        or eval_status.get("test_metric_record_count") != 1
    ):
        raise QualificationGateError(
            f"{expected_ptm} metric/status evidence is incomplete"
        )
    checkpoint = train.get("terminal_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise QualificationGateError(
            f"{expected_ptm} terminal checkpoint evidence is unavailable"
        )
    checkpoint_sha = _sha(
        checkpoint.get("sha256"),
        f"{expected_ptm}.checkpoint.sha256",
    )
    checkpoint_size = checkpoint.get("size_bytes")
    checkpoint_path = checkpoint.get("path")
    if (
        isinstance(checkpoint_size, bool)
        or not isinstance(checkpoint_size, int)
        or checkpoint_size < 1
        or not isinstance(checkpoint_path, str)
        or not checkpoint_path.startswith("/lustre/")
    ):
        raise QualificationGateError(
            f"{expected_ptm} terminal checkpoint identity is invalid"
        )
    metrics = workflow.get("metrics")
    if not isinstance(metrics, Mapping):
        raise QualificationGateError(
            f"{expected_ptm} standalone metrics are unavailable"
        )
    stable = copy.deepcopy(dict(workflow))
    source_checkpoint_sha = _sha(
        workflow.get("ptm_sha256"),
        f"{expected_ptm}.source_checkpoint.sha256",
    )
    return QualifiedPTM(
        checkpoint_id=expected_ptm,
        workflow_id=str(workflow["workflow_id"]),
        source_checkpoint_sha256=source_checkpoint_sha,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_size_bytes=checkpoint_size,
        checkpoint_path=checkpoint_path,
        map50=_metric(metrics.get("mAP50"), f"{expected_ptm}.mAP50"),
        map_value=_metric(metrics.get("mAP"), f"{expected_ptm}.mAP"),
        workflow_sha256=canonical_sha256(stable),
    )


def audit_qualification(
    completion_path: str | Path = DEFAULT_COMPLETION,
    *,
    expected_manifest_sha256: str | None = None,
) -> QualificationDecision:
    """Validate completion and require evidence-backed registry promotion."""
    path = Path(completion_path)
    if not path.is_file():
        raise QualificationGateError(
            f"qualification completion is unavailable: {path}"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    integrity_payload = copy.deepcopy(document)
    completion_sha256 = integrity_payload.pop("completion_sha256", None)
    if completion_sha256 != canonical_sha256(integrity_payload):
        raise QualificationGateError(
            "qualification completion integrity verification failed"
        )
    if (
        expected_manifest_sha256 is not None
        and document.get("manifest_sha256") != expected_manifest_sha256
    ):
        raise QualificationGateError(
            "qualification completion references a different sealed manifest"
        )
    if (
        document.get("schema_version") != 1
        or document.get("model") != "rtdetr"
        or document.get("terminal") is not True
        or document.get("status") != "success"
        or document.get("logical_workflows_submitted") != 4
        or document.get("successful_workflows") != 4
        or document.get("failed_workflows") != 0
        or document.get("replacement_workflows_submitted") is not False
        or document.get("completion_generated_automatically") is not True
        or document.get("resume_completed_training") is not True
        or document.get("completed_training_jobs_reused") != 4
        or document.get("training_jobs_submitted") != 0
        or document.get("prior_completion_artifact_modified") is not False
        or document.get("failures_preserved") is not True
    ):
        raise QualificationGateError(
            "four-PTM RT-DETR qualification did not finish successfully"
        )
    workflows = document.get("workflows")
    if not isinstance(workflows, list):
        raise QualificationGateError("qualification workflows are unavailable")
    by_ptm = {
        item.get("ptm_id"): item
        for item in workflows
        if isinstance(item, Mapping)
    }
    if set(by_ptm) != set(EXPECTED_PTMS):
        raise QualificationGateError(
            "qualification evidence must contain exactly the four frozen PTMs"
        )
    qualified = tuple(
        _qualified_workflow(by_ptm[checkpoint_id], expected_ptm=checkpoint_id)
        for checkpoint_id in EXPECTED_PTMS
    )
    registry = load_ptm_registry()
    blockers: list[dict[str, Any]] = []
    for checkpoint_id in EXPECTED_PTMS:
        record = registry.checkpoint(checkpoint_id)
        if record.get("status") != "supported":
            blockers.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "stage": "registry_runtime_eligibility",
                    "code": "registry_status_not_supported",
                    "observed_status": record.get("status"),
                    "required_status": "supported",
                    "reason": (
                        "full qualification is evidence for review, not "
                        "authority to mutate or bypass the registry"
                    ),
                }
            )
    stable = {
        "schema_version": 1,
        "gate": "rtdetr_full_qualification_then_supported_registry_v1",
        "evidence_path": str(path),
        "evidence_sha256": sha256_file(path),
        "qualification_campaign_id": str(document["campaign_id"]),
        "qualified": [item.to_dict() for item in qualified],
        "blockers": blockers,
        "runtime_ready": not blockers,
        "promotion_is_dynamic": False,
        "registry_bypass_allowed": False,
    }
    return QualificationDecision(
        evidence_path=str(path),
        evidence_sha256=stable["evidence_sha256"],
        qualification_campaign_id=stable["qualification_campaign_id"],
        qualified=qualified,
        blockers=tuple(blockers),
        decision_sha256=canonical_sha256(stable),
    )


class QualificationLoadEvidence:
    """Production-preflight callback backed by the completed GPU workflows."""

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
                reason="No completed full qualification exists for this PTM",
            )
        observed_sha = sha256_file(request.checkpoint_path)
        observed_size = request.checkpoint_path.stat().st_size
        registry_record = load_ptm_registry().checkpoint(
            request.checkpoint_id
        )
        if (
            registry_record.get("status") != "supported"
            or observed_sha != record.source_checkpoint_sha256
            or observed_size
            != int(registry_record.get("expected_size_bytes", -1))
        ):
            return CheckpointLoadSmokeResult(
                ok=False,
                code="qualification_artifact_or_registry_mismatch",
                reason=(
                    "The live artifact or supported registry identity differs "
                    "from completed qualification evidence"
                ),
                details={
                    "qualification_evidence_sha256": (
                        self._decision.evidence_sha256
                    ),
                    "workflow_sha256": record.workflow_sha256,
                },
            )
        return CheckpointLoadSmokeResult(
            ok=True,
            code="full_train_eval_qualification_reused",
            reason=(
                "Exact checkpoint passed ten-epoch eight-GPU training, ten "
                "validations, terminal reload, and standalone evaluation"
            ),
            details={
                "cpu_or_smoke_model_job_launched": False,
                "qualification_evidence_sha256": (
                    self._decision.evidence_sha256
                ),
                "workflow_sha256": record.workflow_sha256,
                "qualified_mAP50": record.map50,
                "qualified_mAP": record.map_value,
            },
        )


__all__ = [
    "DEFAULT_COMPLETION",
    "EXPECTED_PTMS",
    "QualificationDecision",
    "QualificationGateError",
    "QualificationLoadEvidence",
    "QualifiedPTM",
    "audit_qualification",
    "sha256_file",
]
