#!/usr/bin/env python3

"""Fail-closed bridge from shared-data qualification to PTM runtime eligibility.

The production hierarchical PTM runtime accepts only a live, typed runtime
preflight over registry records whose status is ``supported``.  Historical
training evidence is useful qualification provenance, but must never be
silently converted into runtime eligibility.  This adapter validates that
evidence and reports the exact remaining eligibility blockers.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tao_automl.ptm_registry import canonical_sha256, load_ptm_registry


EXPECTED_PTMS = (
    "deformable_detr.coco.gcvit_tiny.trainable.v1.0",
    "deformable_detr.coco.resnet50.trainable.v1.0",
)
DEFAULT_QUALIFICATION_COMPLETION = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "deformable_detr_synthetic_qualification_20260730/completion.json"
)
EXPECTED_QUALIFICATION_MANIFEST_SHA256 = (
    "59aa20e07b9a6dc28c9afece57a508add4627a4b7bcdf23bc07beeb0722aa9a9"
)
EXPECTED_QUALIFICATION_COMPLETION_SHA256 = (
    "9a3e397a41c1ae576a633d4ee7aaf41bc1a5408da86768fe7ecfe94d4c6ab622"
)


class QualificationEvidenceError(RuntimeError):
    """Qualification evidence is absent, corrupt, or runtime-ineligible."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_metric(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise QualificationEvidenceError(f"{name} must be finite in [0, 1]")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QualificationEvidenceError(
            f"{name} must be finite in [0, 1]"
        ) from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise QualificationEvidenceError(f"{name} must be finite in [0, 1]")
    return number


@dataclass(frozen=True)
class QualificationDecision:
    """Audited qualification evidence and its runtime-eligibility decision."""

    evidence_path: str
    evidence_sha256: str | None
    qualification_campaign_id: str | None
    records: tuple[Mapping[str, Any], ...]
    blockers: tuple[Mapping[str, Any], ...]
    decision_sha256: str

    @property
    def runtime_ready(self) -> bool:
        return not self.blockers

    def stable_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "adapter": "fail_closed_qualification_evidence_v1",
            "evidence_path": self.evidence_path,
            "evidence_sha256": self.evidence_sha256,
            "expected_manifest_sha256": (
                EXPECTED_QUALIFICATION_MANIFEST_SHA256
            ),
            "qualification_campaign_id": self.qualification_campaign_id,
            "records": [dict(item) for item in self.records],
            "blockers": [dict(item) for item in self.blockers],
            "runtime_ready": self.runtime_ready,
            "eligibility_policy": (
                "qualification evidence is provenance only; production "
                "runtime requires status=supported and live typed preflight"
            ),
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
        raise QualificationEvidenceError(
            "Deformable DETR PTM runtime remains fail-closed: "
            f"{codes}. Promote only after the repository PTM review updates "
            "the records to status=supported, then rerun a live typed runtime "
            "preflight; serialized qualification evidence is never executable."
        )


def _workflow_record(
    workflow: Mapping[str, Any],
    *,
    expected_ptm: str,
) -> dict[str, Any]:
    if workflow.get("ptm_id") != expected_ptm:
        raise QualificationEvidenceError(
            f"qualification PTM mismatch for {expected_ptm}"
        )
    qualified_input_sha = workflow.get("ptm_sha256")
    if (
        not isinstance(qualified_input_sha, str)
        or len(qualified_input_sha) != 64
        or any(
            character not in "0123456789abcdef"
            for character in qualified_input_sha
        )
    ):
        raise QualificationEvidenceError(
            f"{expected_ptm} input checkpoint identity is invalid"
        )
    train = workflow.get("jobs", {}).get("train")
    if not isinstance(train, Mapping) or train.get("status") != "Complete":
        raise QualificationEvidenceError(
            f"{expected_ptm} has no completed full training evidence"
        )
    if (
        train.get("full_dataset") is not True
        or train.get("gpus") != 8
        or train.get("nodes") != 1
        or train.get("training_epochs") != 10
        or train.get("validation_interval") != 1
    ):
        raise QualificationEvidenceError(
            f"{expected_ptm} qualification resource/fidelity contract changed"
        )
    status = train.get("status_evidence")
    if (
        not isinstance(status, Mapping)
        or status.get("terminal_success") is not True
        or status.get("validation_record_count") != 10
    ):
        raise QualificationEvidenceError(
            f"{expected_ptm} training status evidence is incomplete"
        )
    metrics = status.get("validation_metrics")
    if not isinstance(metrics, list) or len(metrics) != 10:
        raise QualificationEvidenceError(
            f"{expected_ptm} validation history is incomplete"
        )
    final_map50 = _finite_metric(
        metrics[-1].get("mAP50"),
        f"{expected_ptm}.final_mAP50",
    )
    checkpoint = train.get("terminal_checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or not isinstance(checkpoint.get("path"), str)
        or len(str(checkpoint.get("sha256", ""))) != 64
        or checkpoint.get("size_bytes", 0) < 1
    ):
        raise QualificationEvidenceError(
            f"{expected_ptm} terminal checkpoint evidence is incomplete"
        )
    evaluation = workflow.get("jobs", {}).get("evaluation")
    standalone_evaluation = bool(
        isinstance(evaluation, Mapping)
        and evaluation.get("status") == "Complete"
        and evaluation.get("status_evidence", {}).get("terminal_success")
        is True
    )
    if workflow.get("status") != "success" or not standalone_evaluation:
        raise QualificationEvidenceError(
            f"{expected_ptm} did not complete standalone qualification"
        )
    evaluation_metrics = evaluation["status_evidence"].get("metrics", {})
    evaluation_map50 = _finite_metric(
        evaluation_metrics.get("mAP50"),
        f"{expected_ptm}.standalone_mAP50",
    )
    if abs(evaluation_map50 - final_map50) > 1.0e-12:
        raise QualificationEvidenceError(
            f"{expected_ptm} train/evaluate mAP50 evidence disagrees"
        )
    return {
        "checkpoint_id": expected_ptm,
        "qualified_input_checkpoint_sha256": qualified_input_sha,
        "full_train_load_proven": True,
        "ten_epoch_training_passed": True,
        "ten_in_training_validations_passed": True,
        "standalone_evaluation_passed": standalone_evaluation,
        "standalone_evaluation_mAP50": evaluation_map50,
        "standalone_evaluation_status_sha256": evaluation[
            "status_evidence"
        ]["sha256"],
        "final_in_training_mAP50": final_map50,
        "terminal_checkpoint_sha256": checkpoint["sha256"],
        "terminal_checkpoint_size_bytes": checkpoint["size_bytes"],
        "qualification_workflow_status": workflow.get("status"),
        "qualification_workflow_failure_preserved": bool(
            workflow.get("failure_preserved")
        ),
    }


def evidence_load_callback(decision: QualificationDecision):
    """Return a load callback backed only by stronger full-training evidence.

    This callback is safe only inside a normal production ``runtime`` preflight.
    The preflight must still resolve a ``supported`` registry inventory, probe
    NGC access, verify/download the exact member, validate the YAML, and verify
    the downloaded checkpoint hash.  This callback merely avoids repeating a
    weaker model-load smoke after a ten-epoch eight-GPU load/train/validation
    qualification already proved that same immutable input checkpoint.
    """
    from tao_automl.ptm_preflight import CheckpointLoadSmokeResult

    records = {
        item["checkpoint_id"]: item
        for item in decision.records
    }

    def callback(request: Any) -> CheckpointLoadSmokeResult:
        record = records.get(request.checkpoint_id)
        if record is None:
            return CheckpointLoadSmokeResult(
                ok=False,
                code="qualification_evidence_missing",
                reason="No frozen full-training evidence exists for this checkpoint",
            )
        observed_sha = sha256_file(request.checkpoint_path)
        expected_sha = record.get("qualified_input_checkpoint_sha256")
        if observed_sha != expected_sha:
            return CheckpointLoadSmokeResult(
                ok=False,
                code="qualification_checkpoint_identity_mismatch",
                reason=(
                    "The runtime checkpoint differs from the input checkpoint "
                    "used by the frozen full-training qualification"
                ),
                details={
                    "observed_checkpoint_sha256": observed_sha,
                    "qualified_checkpoint_sha256": expected_sha,
                },
            )
        if (
            record.get("full_train_load_proven") is not True
            or record.get("ten_epoch_training_passed") is not True
            or record.get("ten_in_training_validations_passed") is not True
        ):
            return CheckpointLoadSmokeResult(
                ok=False,
                code="qualification_training_evidence_incomplete",
                reason="Frozen full-training qualification evidence is incomplete",
            )
        return CheckpointLoadSmokeResult(
            ok=True,
            code="qualified_full_training_load",
            reason=(
                "The identical checkpoint completed ten epochs of eight-GPU "
                "training and ten in-training validations"
            ),
            details={
                "qualification_decision_sha256": decision.decision_sha256,
                "qualification_campaign_id": (
                    decision.qualification_campaign_id
                ),
                "checkpoint_id": request.checkpoint_id,
                "checkpoint_sha256": observed_sha,
                "training_epochs": 10,
                "validation_records": 10,
            },
        )

    return callback


def audit_qualification_evidence(
    evidence_path: str | Path = DEFAULT_QUALIFICATION_COMPLETION,
) -> QualificationDecision:
    """Validate frozen evidence and report why typed runtime is or is not ready."""
    path = Path(evidence_path)
    if not path.is_file():
        stable = {
            "schema_version": 1,
            "adapter": "fail_closed_qualification_evidence_v1",
            "evidence_path": str(path),
            "evidence_sha256": None,
            "expected_manifest_sha256": (
                EXPECTED_QUALIFICATION_MANIFEST_SHA256
            ),
            "qualification_campaign_id": None,
            "records": [],
            "blockers": [
                {
                    "checkpoint_id": "deformable_detr",
                    "stage": "shared_dataset_qualification",
                    "code": "qualification_completion_pending",
                    "reason": (
                        "The shared-synthetic-data qualification has not "
                        "emitted its terminal completion artifact"
                    ),
                }
            ],
            "runtime_ready": False,
            "eligibility_policy": (
                "qualification evidence is provenance only; production "
                "runtime requires status=supported and live typed preflight"
            ),
        }
        decision = QualificationDecision(
            evidence_path=str(path),
            evidence_sha256=None,
            qualification_campaign_id=None,
            records=(),
            blockers=tuple(stable["blockers"]),
            decision_sha256=canonical_sha256(stable),
        )
        if canonical_sha256(decision.stable_dict()) != decision.decision_sha256:
            raise QualificationEvidenceError(
                "pending qualification decision integrity check failed"
            )
        return decision
    document = json.loads(path.read_text(encoding="utf-8"))
    observed_completion_sha256 = sha256_file(path)
    if observed_completion_sha256 != EXPECTED_QUALIFICATION_COMPLETION_SHA256:
        raise QualificationEvidenceError(
            "shared-dataset qualification completion identity changed"
        )
    if (
        document.get("schema_version") != 1
        or document.get("model") != "deformable_detr"
        or document.get("terminal") is not True
        or document.get("status") != "success"
        or document.get("manifest_sha256")
        != EXPECTED_QUALIFICATION_MANIFEST_SHA256
        or document.get("logical_workflows_submitted") != 2
        or document.get("successful_workflows") != 2
        or document.get("failed_workflows") != 0
        or document.get("replacement_workflows_submitted") is not False
    ):
        raise QualificationEvidenceError(
            "qualification completion contract is invalid"
        )
    workflows = document.get("workflows")
    if not isinstance(workflows, list):
        raise QualificationEvidenceError(
            "qualification workflows are unavailable"
        )
    by_ptm = {
        item.get("ptm_id"): item
        for item in workflows
        if isinstance(item, Mapping)
    }
    if set(by_ptm) != set(EXPECTED_PTMS):
        raise QualificationEvidenceError(
            "qualification evidence does not contain exactly the two PTM arms"
        )
    records = tuple(
        _workflow_record(by_ptm[checkpoint_id], expected_ptm=checkpoint_id)
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
                        "qualification evidence cannot promote or bypass the "
                        "repository-owned PTM registry"
                    ),
                }
            )
    stable = {
        "schema_version": 1,
        "adapter": "fail_closed_qualification_evidence_v1",
        "evidence_path": str(path),
        "evidence_sha256": observed_completion_sha256,
        "expected_manifest_sha256": EXPECTED_QUALIFICATION_MANIFEST_SHA256,
        "qualification_campaign_id": document["campaign_id"],
        "records": [dict(item) for item in records],
        "blockers": blockers,
        "runtime_ready": not blockers,
        "eligibility_policy": (
            "qualification evidence is provenance only; production runtime "
            "requires status=supported and live typed preflight"
        ),
    }
    decision = QualificationDecision(
        evidence_path=str(path),
        evidence_sha256=stable["evidence_sha256"],
        qualification_campaign_id=document["campaign_id"],
        records=records,
        blockers=tuple(blockers),
        decision_sha256=canonical_sha256(stable),
    )
    if canonical_sha256(decision.stable_dict()) != decision.decision_sha256:
        raise QualificationEvidenceError(
            "qualification decision integrity check failed"
        )
    return decision


__all__ = [
    "DEFAULT_QUALIFICATION_COMPLETION",
    "EXPECTED_PTMS",
    "EXPECTED_QUALIFICATION_COMPLETION_SHA256",
    "EXPECTED_QUALIFICATION_MANIFEST_SHA256",
    "QualificationDecision",
    "QualificationEvidenceError",
    "audit_qualification_evidence",
    "evidence_load_callback",
    "sha256_file",
]
