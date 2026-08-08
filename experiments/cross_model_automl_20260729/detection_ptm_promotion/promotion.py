# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derive candidate PTM registries from sealed full-model qualifications.

This utility has no model execution or live-registry mutation path. It
validates the complete sealed Deformable DETR and/or RT-DETR qualification
population, promotes exactly the successful checkpoint records in an emitted
candidate document, and records every failure without replacing it.

RT-DETR qualification is intentionally read from ``completion.resume.json``:
the resume artifact proves standalone evaluation while reusing the immutable
ten-epoch training jobs from the preserved initial terminal-failure record.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tao_automl.ptm_registry import PTMRegistry, canonical_sha256


TAO_VERSION = "7.1.0-rc-245"
TAO_COMPATIBILITY = "==7.1.0"
SUPPORTED_MODELS = frozenset({"deformable_detr", "rtdetr"})
INTERVENTION_FLAGS = (
    "agent_selected_candidate",
    "agent_injected_candidate",
    "agent_modified_search_space_after_results",
    "agent_changed_seed_after_results",
    "agent_changed_budget_after_results",
    "agent_changed_threshold_after_results",
    "agent_changed_ptm_after_results",
    "agent_overrode_winner",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DetectionPTMPromotionError(ValueError):
    """Qualification evidence is inconsistent, incomplete, or drifted."""


@dataclass(frozen=True)
class QualificationEvidence:
    """One model's sealed manifest and terminal completion."""

    model: str
    manifest: Mapping[str, Any]
    completion: Mapping[str, Any]
    manifest_path: str
    completion_path: str
    manifest_file_sha256: str | None = None
    completion_file_sha256: str | None = None


@dataclass(frozen=True)
class QualificationDecision:
    """Exact pass/failure partition derived from one sealed population."""

    model: str
    campaign_id: str
    manifest_sha256: str
    completion_sha256: str
    manifest_path: str
    completion_path: str
    manifest_file_sha256: str | None
    completion_file_sha256: str | None
    evaluated_checkpoint_ids: tuple[str, ...]
    promoted_checkpoint_ids: tuple[str, ...]
    failed_checkpoint_ids: tuple[str, ...]
    workflow_records: tuple[Mapping[str, Any], ...]
    runtime_provenance: Mapping[str, Any]
    recovery_provenance: Mapping[str, Any]
    failure_records: tuple[Mapping[str, Any], ...]
    default_ptm: str | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            ensure_ascii=False,
            allow_nan=False,
        )
        + ("\n" if indent is not None else "")
    ).encode("utf-8")


def _create_only(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DetectionPTMPromotionError(
            f"cannot load JSON object {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise DetectionPTMPromotionError(f"{path} must contain a JSON object")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DetectionPTMPromotionError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _finite_metric(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise DetectionPTMPromotionError(f"{label} must be finite in [0, 1]")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DetectionPTMPromotionError(
            f"{label} must be finite in [0, 1]"
        ) from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise DetectionPTMPromotionError(f"{label} must be finite in [0, 1]")
    return number


def _verify_self_hash(
    document: Mapping[str, Any],
    field: str,
    label: str,
) -> str:
    payload = copy.deepcopy(dict(document))
    recorded = _sha256(payload.pop(field, None), f"{label}.{field}")
    if canonical_sha256(payload) != recorded:
        raise DetectionPTMPromotionError(f"{label} integrity failed")
    return recorded


def _require_false_flags(value: Any, label: str) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(INTERVENTION_FLAGS)
        or any(flag is not False for flag in value.values())
    ):
        raise DetectionPTMPromotionError(
            f"{label} agent intervention flags are invalid"
        )


def _registry_records(
    base_registry: PTMRegistry,
    model: str,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    document = base_registry.to_dict()
    model_config = document.get("models", {}).get(model)
    if not isinstance(model_config, dict):
        raise DetectionPTMPromotionError(
            f"base registry has no {model!r} model"
        )
    checkpoints = model_config.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise DetectionPTMPromotionError(
            f"base registry {model!r} checkpoint population is empty"
        )
    records = {
        item.get("id"): item
        for item in checkpoints
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    if len(records) != len(checkpoints):
        raise DetectionPTMPromotionError(
            f"base registry {model!r} checkpoint identities are not unique"
        )
    return model_config, records


def _validate_runtime(manifest: Mapping[str, Any]) -> dict[str, Any]:
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        raise DetectionPTMPromotionError(
            "qualification manifest runtime contract is missing"
        )
    if (
        runtime.get("tao_version") != TAO_VERSION
        or runtime.get("platform") != "slurm"
        or runtime.get("nodes") != 1
        or runtime.get("gpus_per_node") != 8
        or runtime.get("distributed_workers_per_node") != 8
    ):
        raise DetectionPTMPromotionError(
            "qualification manifest TAO/SLURM resource contract drifted"
        )
    image = runtime.get("image_reference")
    sqsh_path = runtime.get("sqsh_path")
    sqsh_sha = _sha256(
        runtime.get("sqsh_sha256"),
        "qualification manifest runtime.sqsh_sha256",
    )
    sqsh_size = runtime.get("sqsh_size_bytes")
    if (
        not isinstance(image, str)
        or not image
        or not isinstance(sqsh_path, str)
        or not sqsh_path.startswith("/lustre/")
        or isinstance(sqsh_size, bool)
        or not isinstance(sqsh_size, int)
        or sqsh_size <= 0
    ):
        raise DetectionPTMPromotionError(
            "qualification manifest container/SQSH identity is invalid"
        )
    return {
        "tao_version": TAO_VERSION,
        "tao_compatibility": TAO_COMPATIBILITY,
        "image_reference": image,
        "sqsh_path": sqsh_path,
        "sqsh_sha256": sqsh_sha,
        "sqsh_size_bytes": sqsh_size,
        "nodes": 1,
        "gpus_per_node": 8,
        "distributed_workers_per_node": 8,
    }


def _validate_manifest(
    base_registry: PTMRegistry,
    evidence: QualificationEvidence,
) -> tuple[
    str,
    tuple[Mapping[str, Any], ...],
    dict[str, Mapping[str, Any]],
    dict[str, Any],
    str | None,
]:
    model = evidence.model
    manifest = evidence.manifest
    if model not in SUPPORTED_MODELS:
        raise DetectionPTMPromotionError(
            f"unsupported promotion model {model!r}"
        )
    manifest_sha = _verify_self_hash(
        manifest, "manifest_sha256", f"{model} manifest"
    )
    campaign_id = manifest.get("campaign_id")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("model") != model
        or manifest.get("task") != "object_detection"
        or not isinstance(campaign_id, str)
        or not campaign_id
    ):
        raise DetectionPTMPromotionError(
            f"{model} manifest identity is invalid"
        )
    integrity = manifest.get("integrity")
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("ptm_registry_sha256")
        != base_registry.document_sha256
    ):
        raise DetectionPTMPromotionError(
            f"{model} manifest is not bound to the supplied base registry"
        )

    model_config, records = _registry_records(base_registry, model)
    ptms = manifest.get("ptms")
    if not isinstance(ptms, list) or not ptms:
        raise DetectionPTMPromotionError(
            f"{model} manifest PTM population is empty"
        )
    ids = tuple(
        item.get("id")
        for item in ptms
        if isinstance(item, Mapping)
    )
    workflow_ids = tuple(
        item.get("workflow_id")
        for item in ptms
        if isinstance(item, Mapping)
    )
    registry_ids = tuple(records)
    if (
        len(ids) != len(ptms)
        or any(not isinstance(item, str) or not item for item in ids)
        or len(set(ids)) != len(ids)
        or len(workflow_ids) != len(ptms)
        or any(
            not isinstance(item, str) or not item
            for item in workflow_ids
        )
        or len(set(workflow_ids)) != len(workflow_ids)
        or set(ids) != set(registry_ids)
    ):
        raise DetectionPTMPromotionError(
            f"{model} manifest PTM population drifted from the registry"
        )

    for item in ptms:
        checkpoint_id = item["id"]
        record = records[checkpoint_id]
        _require_false_flags(
            item.get("agent_intervention_flags"),
            f"{model} manifest {checkpoint_id}",
        )
        if (
            record.get("status") != "unverified"
            or item.get("registry_status_before_qualification")
            != record.get("status")
            or item.get("registry_record_sha256")
            != canonical_sha256(record)
        ):
            raise DetectionPTMPromotionError(
                f"{checkpoint_id} registry status/record identity drifted"
            )
        source_identity = record.get("source", {}).get(
            "immutable_identity"
        )
        artifact = item.get("artifact")
        if (
            item.get("source_identity") != source_identity
            or not isinstance(artifact, Mapping)
            or artifact.get("size_bytes")
            != record.get("expected_size_bytes")
        ):
            raise DetectionPTMPromotionError(
                f"{checkpoint_id} source or artifact size identity drifted"
            )
        artifact_sha = _sha256(
            artifact.get("sha256"),
            f"{checkpoint_id} artifact.sha256",
        )
        registry_sha = record.get("sha256")
        if registry_sha is not None and artifact_sha != registry_sha:
            raise DetectionPTMPromotionError(
                f"{checkpoint_id} checkpoint hash drifted"
            )
        checkpoint_spec = item.get("checkpoint_spec")
        registered_spec = record.get("checkpoint_spec_file")
        checkpoint_spec_path = (
            checkpoint_spec.get("path")
            if isinstance(checkpoint_spec, Mapping)
            else None
        )
        registered_spec_path = (
            registered_spec.get("path")
            if isinstance(registered_spec, Mapping)
            else None
        )
        if (
            not isinstance(checkpoint_spec, Mapping)
            or not isinstance(registered_spec, Mapping)
            or checkpoint_spec.get("sha256")
            != registered_spec.get("sha256")
            or not isinstance(checkpoint_spec_path, str)
            or not isinstance(registered_spec_path, str)
            or not checkpoint_spec_path.endswith(registered_spec_path)
            or item.get("checkpoint_target")
            != record.get("checkpoint_target")
            or item.get("default_spec_overrides")
            != record.get("default_spec_overrides")
            or item.get("backbone") != record.get("backbone")
        ):
            raise DetectionPTMPromotionError(
                f"{checkpoint_id} spec identity drifted"
            )
        if (
            "input_contract" in item
            and item.get("input_contract") != record.get("input_contract")
        ):
            raise DetectionPTMPromotionError(
                f"{checkpoint_id} input contract drifted"
            )
        if "source" in item and item.get("source") != record.get("source"):
            raise DetectionPTMPromotionError(
                f"{checkpoint_id} source metadata drifted"
            )
    return (
        manifest_sha,
        tuple(ptms),
        records,
        _validate_runtime(manifest),
        model_config.get("default_ptm"),
    )


def _validate_status_identity(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DetectionPTMPromotionError(f"{label} is missing")
    path = value.get("path")
    if (
        value.get("terminal_success") is not True
        or not isinstance(path, str)
        or not path.startswith("/lustre/")
    ):
        raise DetectionPTMPromotionError(
            f"{label} terminal provenance is invalid"
        )
    _sha256(value.get("sha256"), f"{label}.sha256")
    size = value.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise DetectionPTMPromotionError(
            f"{label}.size_bytes must be positive"
        )
    return value


def _validate_success(
    workflow: Mapping[str, Any],
    *,
    model: str,
    checkpoint_id: str,
    resume: bool,
) -> dict[str, Any]:
    jobs = workflow.get("jobs")
    train = jobs.get("train") if isinstance(jobs, Mapping) else None
    evaluation = (
        jobs.get("evaluation") if isinstance(jobs, Mapping) else None
    )
    if not isinstance(train, Mapping) or not isinstance(evaluation, Mapping):
        raise DetectionPTMPromotionError(
            f"{checkpoint_id} lacks train/evaluation evidence"
        )
    if (
        train.get("status") != "Complete"
        or train.get("full_dataset") is not True
        or train.get("gpus") != 8
        or train.get("nodes") != 1
        or train.get("training_epochs") != 10
        or train.get("validation_interval") != 1
        or evaluation.get("status") != "Complete"
        or evaluation.get("full_validation_split") is not True
        or evaluation.get("gpus") != 8
        or evaluation.get("nodes") != 1
    ):
        raise DetectionPTMPromotionError(
            f"{checkpoint_id} full 10-epoch eight-GPU contract failed"
        )
    train_status = _validate_status_identity(
        train.get("status_evidence"),
        f"{checkpoint_id} train status evidence",
    )
    evaluation_status = _validate_status_identity(
        evaluation.get("status_evidence"),
        f"{checkpoint_id} evaluation status evidence",
    )
    validation_metrics = train_status.get("validation_metrics")
    if (
        train_status.get("validation_record_count") != 10
        or not isinstance(validation_metrics, list)
        or len(validation_metrics) != 10
    ):
        raise DetectionPTMPromotionError(
            f"{checkpoint_id} ten-epoch validation evidence is incomplete"
        )
    for index, metrics in enumerate(validation_metrics):
        if not isinstance(metrics, Mapping):
            raise DetectionPTMPromotionError(
                f"{checkpoint_id} validation metric {index} is invalid"
            )
        _finite_metric(
            metrics.get("mAP"),
            f"{checkpoint_id} validation[{index}].mAP",
        )
        _finite_metric(
            metrics.get("mAP50"),
            f"{checkpoint_id} validation[{index}].mAP50",
        )
    if evaluation_status.get("test_metric_record_count") != 1:
        raise DetectionPTMPromotionError(
            f"{checkpoint_id} standalone evaluation evidence is incomplete"
        )
    result_metrics = workflow.get("metrics")
    status_metrics = evaluation_status.get("metrics")
    if (
        not isinstance(result_metrics, Mapping)
        or not isinstance(status_metrics, Mapping)
    ):
        raise DetectionPTMPromotionError(
            f"{checkpoint_id} standalone metrics are missing"
        )
    for metric_name in ("mAP", "mAP50"):
        result = _finite_metric(
            result_metrics.get(metric_name),
            f"{checkpoint_id}.{metric_name}",
        )
        status_result = _finite_metric(
            status_metrics.get(metric_name),
            f"{checkpoint_id}.evaluation.{metric_name}",
        )
        if result != status_result:
            raise DetectionPTMPromotionError(
                f"{checkpoint_id} standalone metric evidence disagrees"
            )

    checkpoint = train.get("terminal_checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or not isinstance(checkpoint.get("path"), str)
        or not checkpoint["path"].startswith("/lustre/")
        or checkpoint.get("size_bytes") != evaluation.get(
            "checkpoint_size_bytes"
        )
        or checkpoint.get("path") != evaluation.get("checkpoint")
    ):
        raise DetectionPTMPromotionError(
            f"{checkpoint_id} terminal/evaluation checkpoint identity drifted"
        )
    checkpoint_sha = _sha256(
        checkpoint.get("sha256"),
        f"{checkpoint_id} terminal checkpoint sha256",
    )
    if checkpoint_sha != evaluation.get("checkpoint_sha256"):
        raise DetectionPTMPromotionError(
            f"{checkpoint_id} evaluation checkpoint hash drifted"
        )
    if (
        checkpoint.get("training_epochs") != 10
        or checkpoint.get("terminal_epoch_index") != 9
    ):
        raise DetectionPTMPromotionError(
            f"{checkpoint_id} terminal checkpoint fidelity drifted"
        )

    if resume:
        resume_record = workflow.get("resume")
        if (
            not isinstance(resume_record, Mapping)
            or resume_record.get("completed_training_job_reused") is not True
            or resume_record.get("training_job_submitted") is not False
            or resume_record.get("selection_or_candidate_change") is not False
            or resume_record.get("prior_workflow_artifact_modified") is not False
            or resume_record.get("checkpoint_resolved_after_fix") is not True
            or train.get("reused_for_resume") is not True
            or evaluation.get("resume_only") is not True
        ):
            raise DetectionPTMPromotionError(
                f"{checkpoint_id} RT-DETR resume isolation contract failed"
            )
    return {
        "checkpoint_id": checkpoint_id,
        "qualified_input_checkpoint_sha256": workflow["ptm_sha256"],
        "terminal_checkpoint_sha256": checkpoint_sha,
        "standalone_evaluation_status_sha256": evaluation_status["sha256"],
        "mAP": result_metrics["mAP"],
        "mAP50": result_metrics["mAP50"],
        "train_job_id": train.get("tao_job_id"),
        "evaluation_job_id": evaluation.get("tao_job_id"),
        "training_reused": resume,
        "training_jobs_submitted_by_resume": 0 if resume else None,
    }


def derive_qualification_decision(
    base_registry: PTMRegistry,
    evidence: QualificationEvidence,
) -> QualificationDecision:
    """Validate evidence and derive the exact successful PTM population."""
    (
        manifest_sha,
        manifest_ptms,
        _,
        runtime,
        default_ptm,
    ) = _validate_manifest(base_registry, evidence)
    model = evidence.model
    manifest = evidence.manifest
    completion = evidence.completion
    completion_sha = _verify_self_hash(
        completion, "completion_sha256", f"{model} completion"
    )
    campaign_id = manifest["campaign_id"]
    expected_workflows = tuple(
        item["workflow_id"] for item in manifest_ptms
    )
    expected_by_workflow = {
        item["workflow_id"]: item for item in manifest_ptms
    }
    outcomes = completion.get("outcomes")
    workflows = completion.get("workflows")
    if (
        completion.get("schema_version") != 1
        or completion.get("campaign_id") != campaign_id
        or completion.get("model") != model
        or completion.get("manifest_sha256") != manifest_sha
        or completion.get("terminal") is not True
        or not isinstance(outcomes, Mapping)
        # JSON object member order is not semantic, and the sealed writer
        # canonicalizes mapping keys. The workflow evidence list below
        # remains strictly ordered.
        or set(outcomes) != set(expected_workflows)
        or not isinstance(workflows, list)
        or tuple(
            item.get("workflow_id")
            for item in workflows
            if isinstance(item, Mapping)
        )
        != expected_workflows
    ):
        raise DetectionPTMPromotionError(
            f"{model} completion population/identity drifted"
        )
    if (
        completion.get("logical_workflows_submitted")
        != len(expected_workflows)
        or completion.get("cpu_runs") != 0
        or completion.get("smoke_runs") != 0
        or completion.get("ministep_runs") != 0
        or completion.get("local_model_runs") != 0
        or completion.get("failures_preserved") is not True
        or completion.get("replacement_workflows_submitted") is not False
    ):
        raise DetectionPTMPromotionError(
            f"{model} completion execution/isolation contract drifted"
        )

    resume = model == "rtdetr"
    if resume:
        resume_contract = manifest.get("resume_contract")
        prior = completion.get("prior_completion")
        prior_path = (
            prior.get("path") if isinstance(prior, Mapping) else None
        )
        if (
            not isinstance(resume_contract, Mapping)
            or resume_contract.get("resume_completion_artifact_name")
            != "completion.resume.json"
            or resume_contract.get("reuse_completed_training_job") is not True
            or resume_contract.get("training_job_resubmission") is not False
            or resume_contract.get("prior_completion_artifact_immutable")
            is not True
            or resume_contract.get("prior_workflow_artifact_immutable")
            is not True
            or completion.get("completion_generated_automatically") is not True
            or completion.get("resume_completed_training") is not True
            or completion.get("completed_training_jobs_reused")
            != len(expected_workflows)
            or completion.get("training_jobs_submitted") != 0
            or completion.get("prior_completion_artifact_modified") is not False
            or not isinstance(prior, Mapping)
            or not isinstance(prior_path, str)
            or not prior_path.endswith("/completion.json")
            or prior.get("manifest_sha256")
            != resume_contract.get("prior_manifest", {}).get(
                "manifest_sha256"
            )
            or prior.get("completion_sha256") is None
        ):
            raise DetectionPTMPromotionError(
                "RT-DETR completion is not the sealed evaluation-only resume"
            )
        _sha256(
            prior.get("completion_sha256"),
            "RT-DETR prior completion sha256",
        )
        _sha256(
            prior.get("file_sha256"),
            "RT-DETR prior completion file sha256",
        )

    passed: list[str] = []
    failed: list[str] = []
    validated_records: list[Mapping[str, Any]] = []
    failure_records: list[Mapping[str, Any]] = []
    for workflow in workflows:
        workflow_id = workflow["workflow_id"]
        expected = expected_by_workflow[workflow_id]
        checkpoint_id = expected["id"]
        status = workflow.get("status")
        exit_code = workflow.get("process_exit_code")
        _require_false_flags(
            workflow.get("agent_intervention_flags"),
            f"{checkpoint_id} completion",
        )
        if (
            workflow.get("campaign_id") != campaign_id
            or workflow.get("manifest_sha256") != manifest_sha
            or workflow.get("ptm_id") != checkpoint_id
            or workflow.get("ptm_sha256")
            != expected["artifact"]["sha256"]
            or workflow.get("ptm_path")
            != expected["artifact"]["slurm_path"]
            or workflow.get("terminal") is not True
            or status != outcomes[workflow_id]
            or status not in {"success", "terminal_failure"}
            or isinstance(exit_code, bool)
            or not isinstance(exit_code, int)
            or ((status == "success") != (exit_code == 0))
        ):
            raise DetectionPTMPromotionError(
            f"{checkpoint_id} completion identity/status drifted"
        )
        if status == "success":
            if (
                workflow.get("failure_preserved") is not False
                or "failure" in workflow
            ):
                raise DetectionPTMPromotionError(
                    f"{checkpoint_id} success/failure or PTM path drifted"
                )
            passed.append(checkpoint_id)
            validated_records.append(
                _validate_success(
                    workflow,
                    model=model,
                    checkpoint_id=checkpoint_id,
                    resume=resume,
                )
            )
        else:
            failure = workflow.get("failure")
            if (
                workflow.get("failure_preserved") is not True
                or not isinstance(failure, Mapping)
                or failure.get("replacement_submitted") is not False
            ):
                raise DetectionPTMPromotionError(
                    f"{checkpoint_id} terminal failure was not preserved"
                )
            failed.append(checkpoint_id)
            failure_records.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "workflow_id": workflow_id,
                    "failure_type": failure.get("type"),
                    "failure_message": failure.get("message"),
                    "replacement_submitted": False,
                }
            )

    count = len(expected_workflows)
    expected_status = (
        "success" if len(passed) == count else "terminal_with_failures"
    )
    if (
        completion.get("status") != expected_status
        or completion.get("successful_workflows") != len(passed)
        or completion.get("failed_workflows") != len(failed)
        or completion.get("workflows_started_in_parallel") is not True
    ):
        raise DetectionPTMPromotionError(
            f"{model} aggregate completion accounting drifted"
        )

    return QualificationDecision(
        model=model,
        campaign_id=campaign_id,
        manifest_sha256=manifest_sha,
        completion_sha256=completion_sha,
        manifest_path=evidence.manifest_path,
        completion_path=evidence.completion_path,
        manifest_file_sha256=evidence.manifest_file_sha256,
        completion_file_sha256=evidence.completion_file_sha256,
        evaluated_checkpoint_ids=tuple(item["id"] for item in manifest_ptms),
        promoted_checkpoint_ids=tuple(passed),
        failed_checkpoint_ids=tuple(failed),
        workflow_records=tuple(validated_records),
        runtime_provenance=runtime,
        recovery_provenance=(
            {
                "evaluation_only_resume": True,
                "completed_training_jobs_reused": completion[
                    "completed_training_jobs_reused"
                ],
                "training_jobs_submitted": 0,
                "prior_completion": copy.deepcopy(
                    dict(completion["prior_completion"])
                ),
                "prior_completion_artifact_modified": False,
                "initial_terminal_failures_preserved": True,
            }
            if resume
            else {}
        ),
        failure_records=tuple(failure_records),
        default_ptm=default_ptm,
    )


def _validation_evidence(decision: QualificationDecision) -> str:
    return (
        f"{decision.completion_path}"
        f"#completion_sha256={decision.completion_sha256};"
        f"manifest_sha256={decision.manifest_sha256}"
    )


def _container_identity(decision: QualificationDecision) -> str:
    runtime = decision.runtime_provenance
    return (
        f"{runtime['image_reference']}"
        f"@sqsh-sha256:{runtime['sqsh_sha256']}"
    )


def _rtdetr_input_contract_projection(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Project frozen PTM input metadata into RT-DETR runtime defaults."""
    contract = record.get("input_contract")
    preprocessing = (
        contract.get("preprocessing")
        if isinstance(contract, Mapping)
        else None
    )
    height = contract.get("height") if isinstance(contract, Mapping) else None
    width = contract.get("width") if isinstance(contract, Mapping) else None
    preserve = (
        preprocessing.get("preserve_aspect_ratio")
        if isinstance(preprocessing, Mapping)
        else None
    )
    if (
        isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
        or isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or not isinstance(preserve, bool)
    ):
        raise DetectionPTMPromotionError(
            f"{record.get('id')} cannot derive its RT-DETR input projection"
        )
    return {
        "dataset": {
            "augmentation": {
                "train_spatial_size": [height, width],
                "eval_spatial_size": [height, width],
                "preserve_aspect_ratio": preserve,
            }
        }
    }


def _apply_rtdetr_input_contract_projection(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Add the preregistered projection without overwriting a conflict."""
    projection = _rtdetr_input_contract_projection(record)
    overrides = record.get("default_spec_overrides")
    if not isinstance(overrides, dict):
        raise DetectionPTMPromotionError(
            f"{record.get('id')} default_spec_overrides is invalid"
        )
    expected_augmentation = projection["dataset"]["augmentation"]
    existing_dataset = overrides.get("dataset")
    if existing_dataset is not None and not isinstance(existing_dataset, dict):
        raise DetectionPTMPromotionError(
            f"{record.get('id')} dataset override conflicts with projection"
        )
    existing_augmentation = (
        existing_dataset.get("augmentation")
        if isinstance(existing_dataset, dict)
        else None
    )
    if (
        existing_augmentation is not None
        and existing_augmentation != expected_augmentation
    ):
        raise DetectionPTMPromotionError(
            f"{record.get('id')} augmentation override conflicts with "
            "its frozen input contract"
        )
    overrides.setdefault("dataset", {})["augmentation"] = copy.deepcopy(
        expected_augmentation
    )
    return projection


def build_candidate_registry(
    *,
    base_registry: PTMRegistry,
    qualifications: Sequence[QualificationEvidence],
    registry_version: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a candidate registry without mutating the supplied registry."""
    if not isinstance(registry_version, str) or not registry_version.strip():
        raise DetectionPTMPromotionError(
            "registry_version must be a non-empty string"
        )
    if not qualifications:
        raise DetectionPTMPromotionError(
            "at least one qualification is required"
        )
    decisions = tuple(
        derive_qualification_decision(base_registry, evidence)
        for evidence in qualifications
    )
    if len({item.model for item in decisions}) != len(decisions):
        raise DetectionPTMPromotionError(
            "each model may appear in at most one qualification bundle"
        )

    base_document = base_registry.to_dict()
    document = copy.deepcopy(base_document)
    document["registry_version"] = registry_version.strip()
    promoted: list[str] = []
    failures: list[str] = []
    compatibility_projections: dict[str, dict[str, Any]] = {}
    records_by_model = {
        model: {
            item["id"]: item
            for item in config["checkpoints"]
        }
        for model, config in document["models"].items()
    }
    for decision in decisions:
        model_records = records_by_model[decision.model]
        manifest_ptms = {
            item["id"]: item
            for item in next(
                evidence.manifest["ptms"]
                for evidence in qualifications
                if evidence.model == decision.model
            )
        }
        for checkpoint_id in decision.promoted_checkpoint_ids:
            record = model_records[checkpoint_id]
            record["status"] = "supported"
            record.pop("status_reason", None)
            record["sha256"] = manifest_ptms[checkpoint_id]["artifact"][
                "sha256"
            ]
            record["compatible_tao_versions"] = [TAO_COMPATIBILITY]
            record["validation"] = {
                "status": "validated",
                "tao_version": TAO_VERSION,
                "container_identity": _container_identity(decision),
                "evidence": _validation_evidence(decision),
            }
            if decision.model == "rtdetr":
                compatibility_projections[checkpoint_id] = (
                    _apply_rtdetr_input_contract_projection(record)
                )
            promoted.append(checkpoint_id)
        failures.extend(decision.failed_checkpoint_ids)

    for decision in decisions:
        if (
            base_document["models"][decision.model]["default_ptm"]
            != document["models"][decision.model]["default_ptm"]
            or decision.default_ptm
            != document["models"][decision.model]["default_ptm"]
        ):
            raise DetectionPTMPromotionError(
                f"{decision.model} default PTM drifted during promotion"
            )
    candidate = PTMRegistry(document)
    candidate_document = candidate.to_dict()
    audit_models = []
    for decision in decisions:
        audit_models.append(
            {
                "model": decision.model,
                "campaign_id": decision.campaign_id,
                "manifest_path": decision.manifest_path,
                "manifest_sha256": decision.manifest_sha256,
                "manifest_file_sha256": decision.manifest_file_sha256,
                "completion_path": decision.completion_path,
                "completion_sha256": decision.completion_sha256,
                "completion_file_sha256": decision.completion_file_sha256,
                "evaluated_checkpoint_ids": list(
                    decision.evaluated_checkpoint_ids
                ),
                "promoted_checkpoint_ids": list(
                    decision.promoted_checkpoint_ids
                ),
                "failed_checkpoint_ids": list(
                    decision.failed_checkpoint_ids
                ),
                "default_ptm_before": decision.default_ptm,
                "default_ptm_after": candidate_document["models"][
                    decision.model
                ]["default_ptm"],
                "workflow_records": [
                    dict(item) for item in decision.workflow_records
                ],
                "failure_records": [
                    dict(item) for item in decision.failure_records
                ],
                "runtime_provenance": dict(
                    decision.runtime_provenance
                ),
                "recovery_provenance": copy.deepcopy(
                    dict(decision.recovery_provenance)
                ),
                "compatibility_projections": {
                    checkpoint_id: copy.deepcopy(
                        compatibility_projections[checkpoint_id]
                    )
                    for checkpoint_id in decision.promoted_checkpoint_ids
                    if checkpoint_id in compatibility_projections
                },
            }
        )
    audit = {
        "schema_version": 1,
        "promotion_algorithm": (
            "exact_sealed_full_qualification_success_population_v1"
        ),
        "promotion_mode": "candidate_registry_only",
        "live_registry_modified": False,
        "runtime_preflight_invoked": False,
        "model_jobs_launched": False,
        "base_registry_sha256": base_registry.document_sha256,
        "candidate_registry_sha256": candidate.document_sha256,
        "registry_version": candidate.registry_version,
        "models": audit_models,
        "promoted_checkpoint_ids": sorted(promoted),
        "failed_checkpoint_ids": sorted(failures),
        "promotion_ready": bool(promoted),
        "intervention_flags": {
            flag: False for flag in INTERVENTION_FLAGS
        },
    }
    return candidate_document, audit


def _parse_qualifications(
    values: Sequence[Sequence[str]],
) -> tuple[QualificationEvidence, ...]:
    results = []
    seen: set[str] = set()
    for raw_model, raw_manifest, raw_completion in values:
        model = raw_model.strip()
        if model in seen:
            raise DetectionPTMPromotionError(
                f"duplicate qualification model {model!r}"
            )
        seen.add(model)
        manifest_path = Path(raw_manifest).expanduser().resolve()
        completion_path = Path(raw_completion).expanduser().resolve()
        if model == "rtdetr" and completion_path.name != "completion.resume.json":
            raise DetectionPTMPromotionError(
                "RT-DETR promotion requires completion.resume.json"
            )
        results.append(
            QualificationEvidence(
                model=model,
                manifest=_load_object(manifest_path),
                completion=_load_object(completion_path),
                manifest_path=str(manifest_path),
                completion_path=str(completion_path),
                manifest_file_sha256=_sha256_file(manifest_path),
                completion_file_sha256=_sha256_file(completion_path),
            )
        )
    return tuple(results)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Emit an evidence-derived candidate detection PTM registry. "
            "The live registry is never modified."
        )
    )
    parser.add_argument("--base-registry", required=True)
    parser.add_argument(
        "--qualification",
        action="append",
        nargs=3,
        metavar=("MODEL", "MANIFEST", "COMPLETION"),
        required=True,
    )
    parser.add_argument("--registry-version", required=True)
    parser.add_argument("--output-registry", required=True)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args(argv)

    base_path = Path(args.base_registry).expanduser().resolve()
    output_path = Path(args.output_registry).expanduser().resolve()
    audit_path = Path(args.audit).expanduser().resolve()
    if output_path == base_path or audit_path == base_path:
        raise DetectionPTMPromotionError(
            "candidate outputs may not overwrite the live/base registry"
        )
    if output_path.exists() or audit_path.exists():
        raise FileExistsError("promotion outputs are create-only")
    base_registry = PTMRegistry(_load_object(base_path))
    candidate, audit = build_candidate_registry(
        base_registry=base_registry,
        qualifications=_parse_qualifications(args.qualification),
        registry_version=args.registry_version,
    )
    registry_bytes = _canonical_json_bytes(candidate, indent=2)
    output_file_sha = hashlib.sha256(registry_bytes).hexdigest()
    audit = {
        **audit,
        "base_registry_path": str(base_path),
        "base_registry_file_sha256": _sha256_file(base_path),
        "candidate_registry_path": str(output_path),
        "candidate_registry_file_sha256": output_file_sha,
    }
    audit = {**audit, "audit_sha256": canonical_sha256(audit)}
    _create_only(output_path, registry_bytes)
    _create_only(audit_path, _canonical_json_bytes(audit, indent=2))
    print(
        _canonical_json_bytes(
            {
                "audit_sha256": audit["audit_sha256"],
                "candidate_registry_file_sha256": output_file_sha,
                "candidate_registry_sha256": audit[
                    "candidate_registry_sha256"
                ],
                "failed_checkpoint_ids": audit["failed_checkpoint_ids"],
                "live_registry_modified": False,
                "promoted_checkpoint_ids": audit[
                    "promoted_checkpoint_ids"
                ],
            }
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
