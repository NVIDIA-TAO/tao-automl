#!/usr/bin/env python3

"""Typed bridge from Grounding DINO qualification to pilot PTM eligibility.

The qualification completion is provenance, not executable runtime state.
This module validates the terminal train/evaluate evidence, structurally
excludes failed PTMs, and still requires each successful checkpoint to be
``supported`` by the repository registry before production runtime preflight.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tao_automl.ptm_registry import (
    PTMRegistry,
    canonical_sha256,
    load_ptm_registry,
)

try:
    from .contract import AGENT_FLAGS, PreparationError, read_json
    from .future_contract import validate_future_contract
except ImportError:  # pragma: no cover - direct script execution
    from contract import AGENT_FLAGS, PreparationError, read_json
    from future_contract import validate_future_contract


HERE = Path(__file__).resolve().parent


class PilotQualificationError(RuntimeError):
    """Qualification evidence is absent, corrupt, or runtime-ineligible."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_metric(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise PilotQualificationError(f"{name} must be finite in [0, 1]")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PilotQualificationError(
            f"{name} must be finite in [0, 1]"
        ) from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise PilotQualificationError(f"{name} must be finite in [0, 1]")
    return number


def _canonical_document(
    document: Mapping[str, Any],
    *,
    field: str,
    label: str,
) -> str:
    payload = copy.deepcopy(dict(document))
    observed = payload.pop(field, None)
    if not isinstance(observed, str) or observed != canonical_sha256(payload):
        raise PilotQualificationError(f"{label} canonical identity is invalid")
    return observed


def _all_agent_flags_false(document: Mapping[str, Any], label: str) -> None:
    flags = document.get("agent_intervention_flags")
    if (
        not isinstance(flags, Mapping)
        or set(flags) != set(AGENT_FLAGS)
        or any(value is not False for value in flags.values())
    ):
        raise PilotQualificationError(
            f"{label} agent-intervention audit is invalid"
        )


def _successful_workflow_record(
    workflow: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    workflow_id = expected["workflow_id"]
    checkpoint_id = expected["ptm_id"]
    staged = expected["staged_checkpoint"]
    if (
        workflow.get("workflow_id") != workflow_id
        or workflow.get("ptm_id") != checkpoint_id
        or workflow.get("ptm_sha256") != staged["sha256"]
        or workflow.get("status") != "success"
        or workflow.get("terminal") is not True
        or workflow.get("failure_preserved") is not False
    ):
        raise PilotQualificationError(
            f"{workflow_id} successful workflow identity/status is invalid"
        )
    _all_agent_flags_false(workflow, workflow_id)
    jobs = workflow.get("jobs")
    if not isinstance(jobs, Mapping):
        raise PilotQualificationError(f"{workflow_id} jobs are absent")
    train = jobs.get("train")
    if (
        not isinstance(train, Mapping)
        or train.get("status") != "Complete"
        or train.get("nodes") != 1
        or train.get("gpus") != 8
        or train.get("training_epochs") != 10
        or train.get("spec_sha256") != expected["train"]["spec_sha256"]
    ):
        raise PilotQualificationError(
            f"{workflow_id} full training contract is invalid"
        )
    status = train.get("status_evidence")
    if (
        not isinstance(status, Mapping)
        or status.get("terminal_success") is not True
        or status.get("validation_record_count") != 10
    ):
        raise PilotQualificationError(
            f"{workflow_id} training status evidence is incomplete"
        )
    validation = status.get("validation_metrics")
    if not isinstance(validation, list) or len(validation) != 10:
        raise PilotQualificationError(
            f"{workflow_id} validation history is incomplete"
        )
    validation_map50 = [
        _finite_metric(
            item.get("mAP50") if isinstance(item, Mapping) else None,
            f"{workflow_id}.validation[{index}].mAP50",
        )
        for index, item in enumerate(validation)
    ]
    checkpoint = train.get("terminal_checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or not isinstance(checkpoint.get("path"), str)
        or checkpoint.get("training_epochs") != 10
        or checkpoint.get("terminal_epoch_index") != 9
        or not isinstance(checkpoint.get("sha256"), str)
        or len(checkpoint["sha256"]) != 64
        or checkpoint.get("size_bytes", 0) < 1
    ):
        raise PilotQualificationError(
            f"{workflow_id} terminal checkpoint evidence is incomplete"
        )
    evaluation = jobs.get("evaluate")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("status") != "Complete"
        or evaluation.get("nodes") != 1
        or evaluation.get("gpus") != 8
        or evaluation.get("checkpoint") != checkpoint
    ):
        raise PilotQualificationError(
            f"{workflow_id} standalone evaluation contract is invalid"
        )
    evaluation_status = evaluation.get("status_evidence")
    if (
        not isinstance(evaluation_status, Mapping)
        or evaluation_status.get("terminal_success") is not True
    ):
        raise PilotQualificationError(
            f"{workflow_id} standalone evaluation evidence is incomplete"
        )
    evaluation_metrics = evaluation_status.get("metrics")
    if not isinstance(evaluation_metrics, Mapping):
        raise PilotQualificationError(
            f"{workflow_id} standalone evaluation metrics are absent"
        )
    standalone_map50 = _finite_metric(
        evaluation_metrics.get("mAP50"),
        f"{workflow_id}.standalone.mAP50",
    )
    recorded_metrics = workflow.get("metrics")
    if (
        not isinstance(recorded_metrics, Mapping)
        or recorded_metrics.get("training_validation") != validation
        or recorded_metrics.get("standalone") != evaluation_metrics
    ):
        raise PilotQualificationError(
            f"{workflow_id} metric provenance is inconsistent"
        )
    return {
        "checkpoint_id": checkpoint_id,
        "workflow_id": workflow_id,
        "qualified_input_checkpoint": {
            "path": staged["path"],
            "sha256": staged["sha256"],
            "size_bytes": staged["size_bytes"],
        },
        "full_training_passed": True,
        "training_epochs": 10,
        "validation_records": 10,
        "final_validation_mAP50": validation_map50[-1],
        "standalone_evaluation_passed": True,
        "standalone_evaluation_mAP50": standalone_map50,
        "standalone_minus_final_validation_mAP50": (
            standalone_map50 - validation_map50[-1]
        ),
        "terminal_checkpoint": copy.deepcopy(dict(checkpoint)),
        "qualification_workflow_sha256": canonical_sha256(workflow),
    }


def _failed_workflow_record(
    workflow: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    workflow_id = expected["workflow_id"]
    if (
        workflow.get("workflow_id") != workflow_id
        or workflow.get("ptm_id") != expected["ptm_id"]
        or workflow.get("status") != "terminal_failure"
        or workflow.get("terminal") is not True
        or workflow.get("failure_preserved") is not True
        or workflow.get("failure", {}).get("replacement_submitted") is not False
    ):
        raise PilotQualificationError(
            f"{workflow_id} failure was not preserved correctly"
        )
    _all_agent_flags_false(workflow, workflow_id)
    return {
        "checkpoint_id": expected["ptm_id"],
        "workflow_id": workflow_id,
        "status": "terminal_failure",
        "failure_preserved": True,
        "replacement_submitted": False,
        "failure": copy.deepcopy(workflow.get("failure")),
        "qualification_workflow_sha256": canonical_sha256(workflow),
    }


@dataclass(frozen=True)
class PilotQualificationDecision:
    """Validated qualification population and runtime-eligibility decision."""

    contract_path: str
    contract_sha256: str
    completion_path: str
    completion_file_sha256: str | None
    completion_sha256: str | None
    handoff_path: str
    handoff_file_sha256: str | None
    handoff_sha256: str | None
    successful_records: tuple[Mapping[str, Any], ...]
    failed_records: tuple[Mapping[str, Any], ...]
    blockers: tuple[Mapping[str, Any], ...]
    registry_sha256: str
    decision_sha256: str

    @property
    def runtime_ready(self) -> bool:
        return bool(self.successful_records) and not self.blockers

    @property
    def checkpoint_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(item["checkpoint_id"] for item in self.successful_records)
        )

    def stable_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "adapter": "grounding_dino_full_qualification_v1",
            "contract_path": self.contract_path,
            "contract_sha256": self.contract_sha256,
            "completion_path": self.completion_path,
            "completion_file_sha256": self.completion_file_sha256,
            "completion_sha256": self.completion_sha256,
            "handoff_path": self.handoff_path,
            "handoff_file_sha256": self.handoff_file_sha256,
            "handoff_sha256": self.handoff_sha256,
            "successful_records": [
                copy.deepcopy(dict(item)) for item in self.successful_records
            ],
            "failed_records": [
                copy.deepcopy(dict(item)) for item in self.failed_records
            ],
            "qualified_checkpoint_ids": list(self.checkpoint_ids),
            "blockers": [
                copy.deepcopy(dict(item)) for item in self.blockers
            ],
            "runtime_ready": self.runtime_ready,
            "registry_sha256": self.registry_sha256,
            "eligibility_policy": (
                "only successful full qualifications become nonordinal PTM "
                "arms; repository status=supported and live typed runtime "
                "preflight remain mandatory"
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
            str(item.get("code", "unknown")) for item in self.blockers
        )
        raise PilotQualificationError(
            "Grounding DINO pilot remains fail-closed: " + codes
        )


def _decision(
    *,
    contract_path: Path,
    contract_sha256: str,
    completion_path: Path,
    completion_file_sha256: str | None,
    completion_sha256: str | None,
    handoff_path: Path,
    handoff_file_sha256: str | None,
    handoff_sha256: str | None,
    successful_records: list[Mapping[str, Any]],
    failed_records: list[Mapping[str, Any]],
    blockers: list[Mapping[str, Any]],
    registry: PTMRegistry,
) -> PilotQualificationDecision:
    provisional = PilotQualificationDecision(
        contract_path=str(contract_path),
        contract_sha256=contract_sha256,
        completion_path=str(completion_path),
        completion_file_sha256=completion_file_sha256,
        completion_sha256=completion_sha256,
        handoff_path=str(handoff_path),
        handoff_file_sha256=handoff_file_sha256,
        handoff_sha256=handoff_sha256,
        successful_records=tuple(successful_records),
        failed_records=tuple(failed_records),
        blockers=tuple(blockers),
        registry_sha256=registry.document_sha256,
        decision_sha256="",
    )
    decision = PilotQualificationDecision(
        **{
            **provisional.__dict__,
            "decision_sha256": canonical_sha256(provisional.stable_dict()),
        }
    )
    if canonical_sha256(decision.stable_dict()) != decision.decision_sha256:
        raise PilotQualificationError(
            "qualification decision integrity check failed"
        )
    return decision


def audit_pilot_qualification(
    inputs: Mapping[str, Any],
    *,
    experiment_dir: str | Path = HERE,
    registry: PTMRegistry | None = None,
) -> PilotQualificationDecision:
    """Audit the live qualification handoff without restoring runtime objects."""
    root = Path(experiment_dir).resolve()
    qualification = inputs.get("qualification")
    if not isinstance(qualification, Mapping):
        raise PilotQualificationError("qualification input is missing")
    contract_path = (root / qualification["contract_file"]).resolve()
    if not contract_path.is_file():
        raise PilotQualificationError(
            f"qualification contract is unavailable: {contract_path}"
        )
    contract = read_json(contract_path)
    validate_future_contract(contract)
    expected_contract_sha = qualification["expected_contract_sha256"]
    if contract.get("contract_sha256") != expected_contract_sha:
        raise PilotQualificationError(
            "qualification contract identity changed"
        )
    runtime_root = Path(qualification["runtime_root"])
    if not runtime_root.is_absolute():
        raise PilotQualificationError(
            "qualification runtime root must be absolute"
        )
    completion_path = runtime_root / "qualification_completion.json"
    handoff_path = runtime_root / "pilot_handoff.json"
    registry = registry or load_ptm_registry()
    if not completion_path.is_file() or not handoff_path.is_file():
        missing = [
            str(path)
            for path in (completion_path, handoff_path)
            if not path.is_file()
        ]
        return _decision(
            contract_path=contract_path,
            contract_sha256=expected_contract_sha,
            completion_path=completion_path,
            completion_file_sha256=None,
            completion_sha256=None,
            handoff_path=handoff_path,
            handoff_file_sha256=None,
            handoff_sha256=None,
            successful_records=[],
            failed_records=[],
            blockers=[
                {
                    "code": "qualification_handoff_pending",
                    "missing_paths": missing,
                    "reason": (
                        "Terminal qualification and automatic pilot handoff "
                        "must both exist before any pilot SDK is constructed"
                    ),
                }
            ],
            registry=registry,
        )

    completion = read_json(completion_path)
    handoff = read_json(handoff_path)
    completion_sha = _canonical_document(
        completion,
        field="completion_sha256",
        label="qualification completion",
    )
    handoff_sha = _canonical_document(
        handoff,
        field="handoff_sha256",
        label="pilot handoff",
    )
    _all_agent_flags_false(completion, "qualification completion")
    _all_agent_flags_false(handoff, "pilot handoff")
    if (
        completion.get("schema_version") != 1
        or completion.get("campaign_id") != contract["campaign_id"]
        or completion.get("contract_sha256") != expected_contract_sha
        or completion.get("model") != "grounding_dino"
        or completion.get("terminal") is not True
        or completion.get("failures_preserved") is not True
        or completion.get("replacement_workflows_submitted") is not False
        or completion.get("minimum_supported_ptms_for_pilot") != 1
        or completion.get("pilot_handoff_ready") is not True
    ):
        raise PilotQualificationError(
            "qualification completion contract is invalid"
        )
    if (
        handoff.get("schema_version") != 1
        or handoff.get("campaign_id") != contract["campaign_id"]
        or handoff.get("contract_sha256") != expected_contract_sha
        or handoff.get("qualification_completion_sha256") != completion_sha
        or handoff.get("automatic") is not True
        or handoff.get("manual_confirmation_required") is not False
        or handoff.get("pilot_modes")
        != ["accuracy", "latency", "multi_objective"]
        or handoff.get("status")
        != "ready_for_algorithm_generated_mode_pilots"
        or handoff.get("selection_or_recommendation_performed") is not False
    ):
        raise PilotQualificationError("pilot handoff contract is invalid")
    expected_jobs = {
        item["workflow_id"]: item
        for item in contract["qualification"]["jobs"]
    }
    workflows = completion.get("workflows")
    if not isinstance(workflows, list):
        raise PilotQualificationError(
            "qualification workflows are unavailable"
        )
    observed = {
        item.get("workflow_id"): item
        for item in workflows
        if isinstance(item, Mapping)
    }
    if set(observed) != set(expected_jobs):
        raise PilotQualificationError(
            "qualification workflow population changed"
        )
    successful: list[Mapping[str, Any]] = []
    failed: list[Mapping[str, Any]] = []
    for workflow_id in sorted(expected_jobs):
        workflow = observed[workflow_id]
        if workflow.get("status") == "success":
            successful.append(
                _successful_workflow_record(
                    workflow,
                    expected_jobs[workflow_id],
                )
            )
        else:
            failed.append(
                _failed_workflow_record(
                    workflow,
                    expected_jobs[workflow_id],
                )
            )
    if (
        len(successful) != completion.get("successful_workflows")
        or len(failed) != completion.get("failed_workflows")
        or len(successful) < 1
        or completion.get("status")
        not in {"success", "terminal_with_failures"}
    ):
        raise PilotQualificationError(
            "qualification completion counts/status are inconsistent"
        )
    blockers: list[Mapping[str, Any]] = []
    for item in successful:
        try:
            record = registry.checkpoint(item["checkpoint_id"])
        except KeyError as exc:
            raise PilotQualificationError(
                f"qualified PTM is absent from registry: "
                f"{item['checkpoint_id']}"
            ) from exc
        if (
            record.get("model_family") != "grounding_dino"
            or record.get("status") != "supported"
        ):
            blockers.append(
                {
                    "checkpoint_id": item["checkpoint_id"],
                    "code": "registry_status_not_supported",
                    "observed_status": record.get("status"),
                    "required_status": "supported",
                    "reason": (
                        "Full qualification cannot silently bypass the "
                        "repository-owned runtime eligibility review"
                    ),
                }
            )
    return _decision(
        contract_path=contract_path,
        contract_sha256=expected_contract_sha,
        completion_path=completion_path,
        completion_file_sha256=sha256_file(completion_path),
        completion_sha256=completion_sha,
        handoff_path=handoff_path,
        handoff_file_sha256=sha256_file(handoff_path),
        handoff_sha256=handoff_sha,
        successful_records=successful,
        failed_records=failed,
        blockers=blockers,
        registry=registry,
    )


def evidence_load_callback(decision: PilotQualificationDecision):
    """Use stronger full-training evidence inside live runtime preflight only."""
    from tao_automl.ptm_preflight import CheckpointLoadSmokeResult

    qualified = {
        item["checkpoint_id"]: item for item in decision.successful_records
    }

    def callback(request: Any) -> CheckpointLoadSmokeResult:
        record = qualified.get(request.checkpoint_id)
        if record is None:
            return CheckpointLoadSmokeResult(
                ok=False,
                code="qualification_evidence_missing",
                reason=(
                    "No successful full qualification exists for this "
                    "checkpoint"
                ),
            )
        observed_sha = sha256_file(request.checkpoint_path)
        expected = record["qualified_input_checkpoint"]
        if (
            observed_sha != expected["sha256"]
            or request.checkpoint_path.stat().st_size
            != expected["size_bytes"]
        ):
            return CheckpointLoadSmokeResult(
                ok=False,
                code="qualification_checkpoint_identity_mismatch",
                reason=(
                    "Runtime bytes differ from the checkpoint used by the "
                    "frozen full qualification"
                ),
            )
        return CheckpointLoadSmokeResult(
            ok=True,
            code="qualified_full_training_load",
            reason=(
                "The identical checkpoint completed ten epochs of one-node "
                "eight-GPU training, validation, and standalone evaluation"
            ),
            details={
                "qualification_decision_sha256": decision.decision_sha256,
                "checkpoint_id": request.checkpoint_id,
                "checkpoint_sha256": observed_sha,
                "training_epochs": 10,
                "validation_records": 10,
                "cpu_model_runs": 0,
                "smoke_or_ministep_runs": 0,
            },
        )

    return callback


__all__ = [
    "PilotQualificationDecision",
    "PilotQualificationError",
    "audit_pilot_qualification",
    "evidence_load_callback",
    "sha256_file",
]
