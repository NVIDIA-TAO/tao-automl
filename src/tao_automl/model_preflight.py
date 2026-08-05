# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content-addressed local model-preflight orchestration.

This module defines the production contract for proving that a TAO model is
ready to enter a GPU campaign.  It intentionally performs no downloads and no
model execution.  A model integration supplies an adapter which executes each
physical step and returns narrowly scoped evidence.  The orchestrator validates
that evidence, derives the task-aware metric-sanity decision, and seals every
record in an ordered hash chain.

The full-epoch obligation applies only to the registered default PTM.  Every
eligible PTM has a separate, lower-cost load/train/validation/inference smoke
obligation.  Keeping those gates distinct prevents an aggregate success bit
from hiding incomplete PTM coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from tao_automl.metric_sanity import (
    MetricEvidence,
    UnknownMetricPolicyError,
    evaluate_metric_sanity,
)


MODEL_PREFLIGHT_SCHEMA_VERSION = 1

CANONICAL_MODEL_TASKS = MappingProxyType({
    "dino": "object_detection",
    "deformable_detr": "object_detection",
    "rtdetr": "object_detection",
    "grounding_dino": "referring_expression_box_grounding",
    "segformer": "semantic_segmentation",
    "oneformer": "panoptic_segmentation",
    "mask2former": "instance_segmentation",
    "mask_grounding_dino": "referring_expression_segmentation",
})
CANONICAL_MODEL_IDS = tuple(CANONICAL_MODEL_TASKS)

MODEL_PREFLIGHT_STAGES = (
    "dataset_validation",
    "default_ptm_load",
    "eligible_ptm_smoke",
    "default_model_full_epoch",
    "in_epoch_validation",
    "standalone_evaluation",
    "metric_sanity",
    "checkpoint_save_reload",
    "latency_instrumentation",
    "output_artifact_validation",
    "interrupted_resume_replay",
)
_DERIVED_STAGES = frozenset({"metric_sanity"})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]*$")
_SAFE_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,79}$")
_SAFE_EXCEPTION_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,79}$")


class ModelPreflightValidationError(ValueError):
    """Raised when inputs, evidence, or a sealed report violate the contract."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__(
            "Invalid model preflight contract:\n"
            + "\n".join(f"- {error}" for error in self.errors)
        )


class ModelPreflightResumeError(RuntimeError):
    """Raised before resume when a prior report is not an intact prefix."""


def canonical_sha256(value: Any) -> str:
    """Return a deterministic hash for finite JSON-compatible content."""
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ModelPreflightValidationError(
            ("value must be finite canonical JSON",)
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _canonical_roundtrip(value: Any) -> Any:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ModelPreflightValidationError(
            ("value must be finite canonical JSON",)
        ) from exc
    return json.loads(encoded)


def _require_identifier(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or _IDENTIFIER_RE.fullmatch(value) is None
    ):
        raise ModelPreflightValidationError(
            (f"{name} must be a non-empty portable identifier",)
        )
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ModelPreflightValidationError(
            (f"{name} must be a lowercase 64-character SHA-256",)
        )
    return value


def _require_bool(value: Any, name: str, *, expected: bool | None = None) -> bool:
    if not isinstance(value, bool):
        raise ModelPreflightValidationError((f"{name} must be boolean",))
    if expected is not None and value is not expected:
        raise ModelPreflightValidationError(
            (f"{name} must be {str(expected).lower()}",)
        )
    return value


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ModelPreflightValidationError(
            (f"{name} must be an integer >= {minimum}",)
        )
    return value


def _require_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ModelPreflightValidationError(
            (f"{name} must be a finite number",)
        )
    number = float(value)
    if minimum is not None and number < minimum:
        raise ModelPreflightValidationError(
            (f"{name} must be >= {minimum}",)
        )
    return number


def _require_exact_keys(
    value: Any,
    *,
    name: str,
    keys: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelPreflightValidationError((f"{name} must be an object",))
    observed = set(value)
    expected = set(keys)
    errors = [
        *(f"{name}.{key} is required" for key in sorted(expected - observed)),
        *(
            f"{name}.{key} is not a recognized field"
            for key in sorted(observed - expected)
        ),
    ]
    if errors:
        raise ModelPreflightValidationError(errors)
    canonical = _canonical_roundtrip(dict(value))
    assert isinstance(canonical, dict)
    return canonical


@dataclass(frozen=True)
class PreflightPTMIdentity:
    """Immutable identity for one eligible pretrained checkpoint."""

    id: str
    checkpoint_sha256: str
    registry_record_sha256: str
    ptm_preflight_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_identifier(self.id, "ptm.id"))
        for field in (
            "checkpoint_sha256",
            "registry_record_sha256",
            "ptm_preflight_sha256",
        ):
            object.__setattr__(
                self,
                field,
                _require_sha256(getattr(self, field), f"ptm.{field}"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "registry_record_sha256": self.registry_record_sha256,
            "ptm_preflight_sha256": self.ptm_preflight_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PreflightPTMIdentity":
        item = _require_exact_keys(
            value,
            name="eligible_ptm",
            keys=(
                "id",
                "checkpoint_sha256",
                "registry_record_sha256",
                "ptm_preflight_sha256",
            ),
        )
        return cls(**item)


@dataclass(frozen=True)
class ModelPreflightInputs:
    """Frozen identities and contracts for a local model preflight."""

    preflight_id: str
    model_id: str
    task: str
    tao_version: str
    source_commit: str
    package_sha256: str
    container_sha256: str
    dataset_id: str
    dataset_manifest_sha256: str
    annotation_contract_sha256: str
    train_split_sha256: str
    validation_split_sha256: str
    default_ptm_id: str
    eligible_ptms: tuple[PreflightPTMIdentity, ...]
    merged_spec_sha256: str
    metric_name: str
    latency_protocol_sha256: str
    latency_input_sha256: str
    latency_timed_scope: str
    output_contract_sha256: str
    seed: int
    local_gpu_count: int = 1
    schema_version: int = MODEL_PREFLIGHT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_PREFLIGHT_SCHEMA_VERSION:
            raise ModelPreflightValidationError(
                (
                    "schema_version must equal "
                    f"{MODEL_PREFLIGHT_SCHEMA_VERSION}",
                )
            )
        for field in ("preflight_id", "dataset_id", "default_ptm_id"):
            object.__setattr__(
                self,
                field,
                _require_identifier(getattr(self, field), field),
            )
        model = _require_identifier(self.model_id, "model_id")
        if model not in CANONICAL_MODEL_TASKS:
            raise ModelPreflightValidationError(
                (
                    "model_id must be one of "
                    + ", ".join(CANONICAL_MODEL_IDS),
                )
            )
        object.__setattr__(self, "model_id", model)
        if self.task != CANONICAL_MODEL_TASKS[model]:
            raise ModelPreflightValidationError(
                (
                    f"task must be {CANONICAL_MODEL_TASKS[model]!r} "
                    f"for model {model!r}",
                )
            )
        for field in ("tao_version", "metric_name", "latency_timed_scope"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ModelPreflightValidationError(
                    (f"{field} must be a non-empty string",)
                )
        if (
            not isinstance(self.source_commit, str)
            or _COMMIT_RE.fullmatch(self.source_commit) is None
        ):
            raise ModelPreflightValidationError(
                ("source_commit must be a full lowercase Git object ID",)
            )
        for field in (
            "package_sha256",
            "container_sha256",
            "dataset_manifest_sha256",
            "annotation_contract_sha256",
            "train_split_sha256",
            "validation_split_sha256",
            "merged_spec_sha256",
            "latency_protocol_sha256",
            "latency_input_sha256",
            "output_contract_sha256",
        ):
            object.__setattr__(
                self,
                field,
                _require_sha256(getattr(self, field), field),
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ModelPreflightValidationError(
                ("seed must be an integer >= 0",)
            )
        if self.local_gpu_count != 1 or isinstance(self.local_gpu_count, bool):
            raise ModelPreflightValidationError(
                ("local_gpu_count must be exactly 1",)
            )
        ptms = tuple(self.eligible_ptms)
        if not ptms or not all(isinstance(item, PreflightPTMIdentity) for item in ptms):
            raise ModelPreflightValidationError(
                ("eligible_ptms must contain at least one PTM identity",)
            )
        identifiers = [item.id for item in ptms]
        if len(identifiers) != len(set(identifiers)):
            raise ModelPreflightValidationError(
                ("eligible_ptms must not contain duplicate IDs",)
            )
        if self.default_ptm_id not in identifiers:
            raise ModelPreflightValidationError(
                ("default_ptm_id must identify an eligible PTM",)
            )
        object.__setattr__(
            self,
            "eligible_ptms",
            tuple(sorted(ptms, key=lambda item: item.id)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "preflight_id": self.preflight_id,
            "model_id": self.model_id,
            "task": self.task,
            "tao_version": self.tao_version,
            "source_commit": self.source_commit,
            "package_sha256": self.package_sha256,
            "container_sha256": self.container_sha256,
            "dataset_id": self.dataset_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "annotation_contract_sha256": self.annotation_contract_sha256,
            "train_split_sha256": self.train_split_sha256,
            "validation_split_sha256": self.validation_split_sha256,
            "default_ptm_id": self.default_ptm_id,
            "eligible_ptms": [item.to_dict() for item in self.eligible_ptms],
            "merged_spec_sha256": self.merged_spec_sha256,
            "metric_name": self.metric_name,
            "latency_protocol_sha256": self.latency_protocol_sha256,
            "latency_input_sha256": self.latency_input_sha256,
            "latency_timed_scope": self.latency_timed_scope,
            "output_contract_sha256": self.output_contract_sha256,
            "seed": self.seed,
            "local_gpu_count": self.local_gpu_count,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelPreflightInputs":
        item = _require_exact_keys(
            value,
            name="inputs",
            keys=(
                "schema_version",
                "preflight_id",
                "model_id",
                "task",
                "tao_version",
                "source_commit",
                "package_sha256",
                "container_sha256",
                "dataset_id",
                "dataset_manifest_sha256",
                "annotation_contract_sha256",
                "train_split_sha256",
                "validation_split_sha256",
                "default_ptm_id",
                "eligible_ptms",
                "merged_spec_sha256",
                "metric_name",
                "latency_protocol_sha256",
                "latency_input_sha256",
                "latency_timed_scope",
                "output_contract_sha256",
                "seed",
                "local_gpu_count",
            ),
        )
        ptm_values = item.pop("eligible_ptms")
        if not isinstance(ptm_values, list):
            raise ModelPreflightValidationError(
                ("inputs.eligible_ptms must be a list",)
            )
        item["eligible_ptms"] = tuple(
            PreflightPTMIdentity.from_dict(ptm) for ptm in ptm_values
        )
        return cls(**item)


@dataclass(frozen=True)
class ModelPreflightStepRequest:
    """Immutable request presented to a model-specific execution adapter."""

    inputs: ModelPreflightInputs
    stage: str
    stage_index: int
    prior_record_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.stage not in MODEL_PREFLIGHT_STAGES:
            raise ModelPreflightValidationError(
                (f"unexpected preflight stage {self.stage!r}",)
            )
        if MODEL_PREFLIGHT_STAGES[self.stage_index] != self.stage:
            raise ModelPreflightValidationError(
                ("stage_index does not match the required stage ordering",)
            )
        if len(self.prior_record_sha256s) != self.stage_index:
            raise ModelPreflightValidationError(
                ("prior record chain length does not match stage_index",)
            )
        for index, digest in enumerate(self.prior_record_sha256s):
            _require_sha256(digest, f"prior_record_sha256s[{index}]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_PREFLIGHT_SCHEMA_VERSION,
            "inputs_sha256": self.inputs.canonical_sha256,
            "stage": self.stage,
            "stage_index": self.stage_index,
            "prior_record_sha256s": list(self.prior_record_sha256s),
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class ModelPreflightStepResult:
    """Adapter response; successful evidence is validated per stage."""

    stage: str
    passed: bool
    evidence: Mapping[str, Any]
    code: str = "ok"

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str):
            raise TypeError("stage must be a string")
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be boolean")
        if not isinstance(self.code, str) or _SAFE_CODE_RE.fullmatch(self.code) is None:
            raise ValueError("code must be a safe lowercase identifier")
        canonical = _canonical_roundtrip(dict(self.evidence))
        object.__setattr__(self, "evidence", MappingProxyType(canonical))

    @classmethod
    def success(
        cls,
        stage: str,
        evidence: Mapping[str, Any],
    ) -> "ModelPreflightStepResult":
        return cls(stage=stage, passed=True, evidence=evidence, code="ok")

    @classmethod
    def failure(
        cls,
        stage: str,
        code: str,
    ) -> "ModelPreflightStepResult":
        return cls(stage=stage, passed=False, evidence={}, code=code)


class ModelPreflightAdapter(Protocol):
    """Execution adapter used by :func:`run_model_preflight`."""

    def __call__(
        self,
        request: ModelPreflightStepRequest,
    ) -> ModelPreflightStepResult:
        """Execute exactly one requested preflight step."""


def _ptm_lookup(inputs: ModelPreflightInputs) -> dict[str, PreflightPTMIdentity]:
    return {item.id: item for item in inputs.eligible_ptms}


def _dataset_evidence(
    evidence: Mapping[str, Any],
    inputs: ModelPreflightInputs,
) -> dict[str, Any]:
    item = _require_exact_keys(
        evidence,
        name="dataset_validation.evidence",
        keys=(
            "dataset_id",
            "manifest_sha256",
            "annotation_contract_sha256",
            "annotations_valid",
            "train_split_sha256",
            "validation_split_sha256",
            "train_samples",
            "validation_samples",
        ),
    )
    expected = {
        "dataset_id": inputs.dataset_id,
        "manifest_sha256": inputs.dataset_manifest_sha256,
        "annotation_contract_sha256": inputs.annotation_contract_sha256,
        "train_split_sha256": inputs.train_split_sha256,
        "validation_split_sha256": inputs.validation_split_sha256,
    }
    for field, value in expected.items():
        if item[field] != value:
            raise ModelPreflightValidationError(
                (f"dataset_validation.evidence.{field} does not match inputs",)
            )
    _require_bool(
        item["annotations_valid"],
        "dataset_validation.evidence.annotations_valid",
        expected=True,
    )
    for field in ("train_samples", "validation_samples"):
        _require_int(
            item[field],
            f"dataset_validation.evidence.{field}",
            minimum=1,
        )
    return item


def _default_ptm_evidence(
    evidence: Mapping[str, Any],
    inputs: ModelPreflightInputs,
) -> dict[str, Any]:
    item = _require_exact_keys(
        evidence,
        name="default_ptm_load.evidence",
        keys=(
            "ptm_id",
            "checkpoint_sha256",
            "loaded",
            "input_contract_verified",
            "spec_merge_verified",
        ),
    )
    ptm = _ptm_lookup(inputs)[inputs.default_ptm_id]
    if item["ptm_id"] != ptm.id or item["checkpoint_sha256"] != ptm.checkpoint_sha256:
        raise ModelPreflightValidationError(
            ("default_ptm_load evidence does not identify the frozen default PTM",)
        )
    for field in ("loaded", "input_contract_verified", "spec_merge_verified"):
        _require_bool(
            item[field],
            f"default_ptm_load.evidence.{field}",
            expected=True,
        )
    return item


def _eligible_ptm_evidence(
    evidence: Mapping[str, Any],
    inputs: ModelPreflightInputs,
) -> dict[str, Any]:
    item = _require_exact_keys(
        evidence,
        name="eligible_ptm_smoke.evidence",
        keys=("ptms",),
    )
    values = item["ptms"]
    if not isinstance(values, list):
        raise ModelPreflightValidationError(
            ("eligible_ptm_smoke.evidence.ptms must be a list",)
        )
    expected = _ptm_lookup(inputs)
    observed: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(values):
        record = _require_exact_keys(
            value,
            name=f"eligible_ptm_smoke.evidence.ptms[{index}]",
            keys=(
                "ptm_id",
                "checkpoint_sha256",
                "loaded",
                "train_step_passed",
                "validation_step_passed",
                "inference_step_passed",
            ),
        )
        ptm_id = record["ptm_id"]
        if not isinstance(ptm_id, str) or ptm_id not in expected:
            raise ModelPreflightValidationError(
                ("eligible_ptm_smoke contains an unexpected PTM",)
            )
        if ptm_id in observed:
            raise ModelPreflightValidationError(
                ("eligible_ptm_smoke contains duplicate PTM evidence",)
            )
        if record["checkpoint_sha256"] != expected[ptm_id].checkpoint_sha256:
            raise ModelPreflightValidationError(
                (f"eligible_ptm_smoke checkpoint hash mismatch for {ptm_id}",)
            )
        for field in (
            "loaded",
            "train_step_passed",
            "validation_step_passed",
            "inference_step_passed",
        ):
            _require_bool(
                record[field],
                f"eligible_ptm_smoke.evidence.{ptm_id}.{field}",
                expected=True,
            )
        observed[ptm_id] = record
    missing = sorted(set(expected) - set(observed))
    if missing:
        raise ModelPreflightValidationError(
            (
                "eligible_ptm_smoke is missing required PTMs: "
                + ", ".join(missing),
            )
        )
    item["ptms"] = [observed[key] for key in sorted(observed)]
    return item


def _full_epoch_evidence(
    evidence: Mapping[str, Any],
    inputs: ModelPreflightInputs,
) -> dict[str, Any]:
    item = _require_exact_keys(
        evidence,
        name="default_model_full_epoch.evidence",
        keys=(
            "ptm_id",
            "single_gpu",
            "completed",
            "completed_epochs",
            "training_batches",
            "distinct_training_steps",
            "final_checkpoint_sha256",
        ),
    )
    if item["ptm_id"] != inputs.default_ptm_id:
        raise ModelPreflightValidationError(
            ("default_model_full_epoch must use the frozen default PTM",)
        )
    for field in ("single_gpu", "completed"):
        _require_bool(
            item[field],
            f"default_model_full_epoch.evidence.{field}",
            expected=True,
        )
    _require_int(
        item["completed_epochs"],
        "default_model_full_epoch.evidence.completed_epochs",
        minimum=1,
    )
    for field in ("training_batches", "distinct_training_steps"):
        _require_int(
            item[field],
            f"default_model_full_epoch.evidence.{field}",
            minimum=1,
        )
    _require_sha256(
        item["final_checkpoint_sha256"],
        "default_model_full_epoch.evidence.final_checkpoint_sha256",
    )
    return item


def _metric_evidence(
    stage: str,
    evidence: Mapping[str, Any],
    inputs: ModelPreflightInputs,
) -> dict[str, Any]:
    keys = [
        "metric_name",
        "metric_value",
        "completed_evaluations",
        "passed",
    ]
    if stage == "standalone_evaluation":
        keys.append("runtime_metric_contract_verified")
    item = _require_exact_keys(
        evidence,
        name=f"{stage}.evidence",
        keys=keys,
    )
    if item["metric_name"] != inputs.metric_name:
        raise ModelPreflightValidationError(
            (f"{stage}.evidence.metric_name does not match inputs",)
        )
    item["metric_value"] = _require_float(
        item["metric_value"],
        f"{stage}.evidence.metric_value",
    )
    _require_int(
        item["completed_evaluations"],
        f"{stage}.evidence.completed_evaluations",
        minimum=1,
    )
    _require_bool(
        item["passed"],
        f"{stage}.evidence.passed",
        expected=True,
    )
    if stage == "standalone_evaluation":
        _require_bool(
            item["runtime_metric_contract_verified"],
            f"{stage}.evidence.runtime_metric_contract_verified",
            expected=True,
        )
    return item


def _checkpoint_evidence(
    evidence: Mapping[str, Any],
    inputs: ModelPreflightInputs,
) -> dict[str, Any]:
    item = _require_exact_keys(
        evidence,
        name="checkpoint_save_reload.evidence",
        keys=(
            "ptm_id",
            "saved",
            "reloaded",
            "saved_checkpoint_sha256",
            "reloaded_checkpoint_sha256",
        ),
    )
    if item["ptm_id"] != inputs.default_ptm_id:
        raise ModelPreflightValidationError(
            ("checkpoint_save_reload must use the frozen default PTM",)
        )
    for field in ("saved", "reloaded"):
        _require_bool(
            item[field],
            f"checkpoint_save_reload.evidence.{field}",
            expected=True,
        )
    saved = _require_sha256(
        item["saved_checkpoint_sha256"],
        "checkpoint_save_reload.evidence.saved_checkpoint_sha256",
    )
    reloaded = _require_sha256(
        item["reloaded_checkpoint_sha256"],
        "checkpoint_save_reload.evidence.reloaded_checkpoint_sha256",
    )
    if saved != reloaded:
        raise ModelPreflightValidationError(
            ("checkpoint save and reload hashes must match",)
        )
    return item


def _latency_evidence(
    evidence: Mapping[str, Any],
    inputs: ModelPreflightInputs,
) -> dict[str, Any]:
    item = _require_exact_keys(
        evidence,
        name="latency_instrumentation.evidence",
        keys=(
            "protocol_sha256",
            "input_sha256",
            "timed_scope",
            "single_gpu",
            "warmup_iterations",
            "timed_iterations",
            "rounds",
            "synchronized",
            "median_ms",
            "p95_ms",
            "mad_ms",
            "iqr_ms",
            "robust_cv",
            "round_drift_ms",
            "device_spread_ms",
            "quality_gates_passed",
        ),
    )
    expected = {
        "protocol_sha256": inputs.latency_protocol_sha256,
        "input_sha256": inputs.latency_input_sha256,
        "timed_scope": inputs.latency_timed_scope,
    }
    for field, value in expected.items():
        if item[field] != value:
            raise ModelPreflightValidationError(
                (f"latency_instrumentation.evidence.{field} does not match inputs",)
            )
    for field in ("single_gpu", "synchronized", "quality_gates_passed"):
        _require_bool(
            item[field],
            f"latency_instrumentation.evidence.{field}",
            expected=True,
        )
    for field in ("warmup_iterations", "timed_iterations"):
        _require_int(
            item[field],
            f"latency_instrumentation.evidence.{field}",
            minimum=1,
        )
    _require_int(
        item["rounds"],
        "latency_instrumentation.evidence.rounds",
        minimum=2,
    )
    for field in (
        "median_ms",
        "p95_ms",
        "mad_ms",
        "iqr_ms",
        "robust_cv",
        "round_drift_ms",
        "device_spread_ms",
    ):
        item[field] = _require_float(
            item[field],
            f"latency_instrumentation.evidence.{field}",
            minimum=0.0,
        )
    if item["median_ms"] <= 0.0:
        raise ModelPreflightValidationError(
            ("latency_instrumentation.evidence.median_ms must be > 0",)
        )
    if item["p95_ms"] < item["median_ms"]:
        raise ModelPreflightValidationError(
            ("latency p95 must be greater than or equal to median",)
        )
    return item


def _output_evidence(
    evidence: Mapping[str, Any],
    inputs: ModelPreflightInputs,
) -> dict[str, Any]:
    item = _require_exact_keys(
        evidence,
        name="output_artifact_validation.evidence",
        keys=(
            "contract_sha256",
            "artifacts",
            "missing_artifact_ids",
            "valid",
        ),
    )
    if item["contract_sha256"] != inputs.output_contract_sha256:
        raise ModelPreflightValidationError(
            ("output artifact contract hash does not match inputs",)
        )
    _require_bool(
        item["valid"],
        "output_artifact_validation.evidence.valid",
        expected=True,
    )
    if item["missing_artifact_ids"] != []:
        raise ModelPreflightValidationError(
            ("output_artifact_validation must not have missing artifacts",)
        )
    values = item["artifacts"]
    if not isinstance(values, list) or not values:
        raise ModelPreflightValidationError(
            ("output_artifact_validation.evidence.artifacts must be non-empty",)
        )
    observed: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(values):
        record = _require_exact_keys(
            value,
            name=f"output_artifact_validation.evidence.artifacts[{index}]",
            keys=("artifact_id", "sha256", "size_bytes"),
        )
        artifact_id = _require_identifier(
            record["artifact_id"],
            f"output_artifact_validation.evidence.artifacts[{index}].artifact_id",
        )
        if artifact_id in observed:
            raise ModelPreflightValidationError(
                ("output artifact IDs must be unique",)
            )
        _require_sha256(
            record["sha256"],
            f"output_artifact_validation.evidence.artifacts[{index}].sha256",
        )
        _require_int(
            record["size_bytes"],
            f"output_artifact_validation.evidence.artifacts[{index}].size_bytes",
            minimum=1,
        )
        observed[artifact_id] = record
    item["artifacts"] = [observed[key] for key in sorted(observed)]
    return item


def _resume_replay_evidence(
    evidence: Mapping[str, Any],
    _inputs: ModelPreflightInputs,
) -> dict[str, Any]:
    item = _require_exact_keys(
        evidence,
        name="interrupted_resume_replay.evidence",
        keys=(
            "interrupted",
            "state_saved",
            "state_sha256",
            "resumed",
            "replay_deterministic",
            "expected_next_request_sha256",
            "actual_next_request_sha256",
            "no_duplicate_trials",
            "no_lost_trials",
        ),
    )
    for field in (
        "interrupted",
        "state_saved",
        "resumed",
        "replay_deterministic",
        "no_duplicate_trials",
        "no_lost_trials",
    ):
        _require_bool(
            item[field],
            f"interrupted_resume_replay.evidence.{field}",
            expected=True,
        )
    _require_sha256(
        item["state_sha256"],
        "interrupted_resume_replay.evidence.state_sha256",
    )
    expected = _require_sha256(
        item["expected_next_request_sha256"],
        "interrupted_resume_replay.evidence.expected_next_request_sha256",
    )
    actual = _require_sha256(
        item["actual_next_request_sha256"],
        "interrupted_resume_replay.evidence.actual_next_request_sha256",
    )
    if expected != actual:
        raise ModelPreflightValidationError(
            ("interrupted resume replay request hashes must match",)
        )
    return item


_EVIDENCE_VALIDATORS: Mapping[
    str,
    Callable[[Mapping[str, Any], ModelPreflightInputs], dict[str, Any]],
] = MappingProxyType({
    "dataset_validation": _dataset_evidence,
    "default_ptm_load": _default_ptm_evidence,
    "eligible_ptm_smoke": _eligible_ptm_evidence,
    "default_model_full_epoch": _full_epoch_evidence,
    "in_epoch_validation": lambda evidence, inputs: _metric_evidence(
        "in_epoch_validation", evidence, inputs
    ),
    "standalone_evaluation": lambda evidence, inputs: _metric_evidence(
        "standalone_evaluation", evidence, inputs
    ),
    "checkpoint_save_reload": _checkpoint_evidence,
    "latency_instrumentation": _latency_evidence,
    "output_artifact_validation": _output_evidence,
    "interrupted_resume_replay": _resume_replay_evidence,
})


def _reason(code: str, stage: str) -> str:
    fixed = {
        "ok": "Required preflight evidence passed.",
        "adapter_exception": "The execution adapter raised a typed exception.",
        "adapter_stage_mismatch": "The adapter returned evidence for a different stage.",
        "adapter_rejected": "The execution adapter reported a required step failure.",
        "invalid_stage_evidence": "The adapter evidence failed the stage contract.",
        "metric_sanity_failed": "The task-aware metric sanity gate did not pass.",
        "unknown_metric_policy": "No verified task-aware metric contract is available.",
    }
    return fixed.get(code, f"Required stage {stage!r} did not pass.")


def _validate_failure_evidence(
    *,
    stage: str,
    code: str,
    reason: Any,
    evidence: Any,
) -> None:
    if code not in {
        "adapter_exception",
        "adapter_stage_mismatch",
        "adapter_rejected",
        "invalid_stage_evidence",
    }:
        raise ModelPreflightValidationError(
            (f"failed physical stage {stage!r} has an unsupported code",)
        )
    if reason != _reason(code, stage):
        raise ModelPreflightValidationError(
            (f"failed physical stage {stage!r} has a noncanonical reason",)
        )
    if code == "adapter_exception":
        item = _require_exact_keys(
            evidence,
            name=f"{stage}.failure_evidence",
            keys=("exception_type",),
        )
        if (
            not isinstance(item["exception_type"], str)
            or _SAFE_EXCEPTION_RE.fullmatch(item["exception_type"]) is None
        ):
            raise ModelPreflightValidationError(
                (f"{stage}.failure_evidence.exception_type is unsafe",)
            )
    elif code == "adapter_stage_mismatch":
        item = _require_exact_keys(
            evidence,
            name=f"{stage}.failure_evidence",
            keys=("adapter_code",),
        )
        if item["adapter_code"] != "stage_mismatch":
            raise ModelPreflightValidationError(
                (f"{stage}.failure_evidence.adapter_code is invalid",)
            )
    elif code == "adapter_rejected":
        item = _require_exact_keys(
            evidence,
            name=f"{stage}.failure_evidence",
            keys=("adapter_code",),
        )
        if (
            not isinstance(item["adapter_code"], str)
            or _SAFE_CODE_RE.fullmatch(item["adapter_code"]) is None
        ):
            raise ModelPreflightValidationError(
                (f"{stage}.failure_evidence.adapter_code is unsafe",)
            )
    else:
        item = _require_exact_keys(
            evidence,
            name=f"{stage}.failure_evidence",
            keys=("adapter_evidence_sha256",),
        )
        _require_sha256(
            item["adapter_evidence_sha256"],
            f"{stage}.failure_evidence.adapter_evidence_sha256",
        )


def _record(
    *,
    request: ModelPreflightStepRequest,
    passed: bool,
    code: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    canonical_evidence = _canonical_roundtrip(dict(evidence))
    evidence_sha = canonical_sha256(canonical_evidence)
    previous = (
        request.prior_record_sha256s[-1]
        if request.prior_record_sha256s
        else None
    )
    value = {
        "stage": request.stage,
        "stage_index": request.stage_index,
        "request_sha256": request.canonical_sha256,
        "status": "passed" if passed else "failed",
        "code": code,
        "reason": _reason(code, request.stage),
        "evidence": canonical_evidence,
        "evidence_sha256": evidence_sha,
        "previous_record_sha256": previous,
    }
    value["record_sha256"] = canonical_sha256(value)
    return value


def _adapter_record(
    request: ModelPreflightStepRequest,
    adapter: ModelPreflightAdapter,
) -> dict[str, Any]:
    try:
        result = adapter(request)
    except Exception as exc:  # adapter boundary is deliberately fail-closed
        exception_type = type(exc).__name__
        if _SAFE_EXCEPTION_RE.fullmatch(exception_type) is None:
            exception_type = "Exception"
        return _record(
            request=request,
            passed=False,
            code="adapter_exception",
            evidence={"exception_type": exception_type},
        )
    if not isinstance(result, ModelPreflightStepResult):
        return _record(
            request=request,
            passed=False,
            code="adapter_rejected",
            evidence={"adapter_code": "invalid_result_type"},
        )
    if result.stage != request.stage:
        return _record(
            request=request,
            passed=False,
            code="adapter_stage_mismatch",
            evidence={"adapter_code": "stage_mismatch"},
        )
    if not result.passed:
        return _record(
            request=request,
            passed=False,
            code="adapter_rejected",
            evidence={"adapter_code": result.code},
        )
    try:
        evidence = _EVIDENCE_VALIDATORS[request.stage](
            result.evidence,
            request.inputs,
        )
    except ModelPreflightValidationError:
        # The untrusted evidence is intentionally not copied into the report.
        return _record(
            request=request,
            passed=False,
            code="invalid_stage_evidence",
            evidence={
                "adapter_evidence_sha256": canonical_sha256(dict(result.evidence)),
            },
        )
    return _record(
        request=request,
        passed=True,
        code="ok",
        evidence=evidence,
    )


def _prior_stage_evidence(
    records: Sequence[Mapping[str, Any]],
    stage: str,
) -> Mapping[str, Any]:
    for record in records:
        if record["stage"] == stage:
            evidence = record["evidence"]
            assert isinstance(evidence, Mapping)
            return evidence
    raise ModelPreflightValidationError(
        (f"metric_sanity requires prior stage {stage!r}",)
    )


def _metric_sanity_record(
    request: ModelPreflightStepRequest,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    dataset = _prior_stage_evidence(records, "dataset_validation")
    epoch = _prior_stage_evidence(records, "default_model_full_epoch")
    in_epoch = _prior_stage_evidence(records, "in_epoch_validation")
    standalone = _prior_stage_evidence(records, "standalone_evaluation")
    metric_evidence = MetricEvidence(
        completed_evaluations=(
            int(in_epoch["completed_evaluations"])
            + int(standalone["completed_evaluations"])
        ),
        distinct_training_steps=epoch["distinct_training_steps"],
        annotation_contract_verified=dataset["annotations_valid"],
        standalone_evaluation_passed=standalone["passed"],
        runtime_metric_contract_verified=standalone[
            "runtime_metric_contract_verified"
        ],
        first_metric_value=in_epoch["metric_value"],
        best_metric_value=max(
            float(in_epoch["metric_value"]),
            float(standalone["metric_value"]),
        ),
    )
    try:
        decision = evaluate_metric_sanity(
            request.inputs.model_id,
            request.inputs.metric_name,
            standalone["metric_value"],
            evidence=metric_evidence,
        )
    except UnknownMetricPolicyError as exc:
        return _record(
            request=request,
            passed=False,
            code="unknown_metric_policy",
            evidence={
                "error_code": exc.code
                if _SAFE_CODE_RE.fullmatch(exc.code)
                else "unknown_metric_policy",
                "model_id": request.inputs.model_id,
                "metric_name": request.inputs.metric_name,
            },
        )
    evidence = decision.to_dict()
    return _record(
        request=request,
        passed=decision.passed,
        code="ok" if decision.passed else "metric_sanity_failed",
        evidence=evidence,
    )


_READINESS_STAGE_FIELDS = (
    ("dataset_prepared", "dataset_validation"),
    ("default_ptm_loaded", "default_ptm_load"),
    ("all_ptms_smoke_tested", "eligible_ptm_smoke"),
    ("default_one_epoch_passed", "default_model_full_epoch"),
    ("in_epoch_validation_passed", "in_epoch_validation"),
    ("standalone_evaluation_passed", "standalone_evaluation"),
    ("metric_valid", "metric_sanity"),
    ("checkpoint_save_reload_passed", "checkpoint_save_reload"),
    ("latency_valid", "latency_instrumentation"),
    ("output_artifacts_valid", "output_artifact_validation"),
    ("resume_replay_valid", "interrupted_resume_replay"),
)


def _readiness(records: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    passed = {
        str(record["stage"])
        for record in records
        if record["status"] == "passed"
    }
    value = {
        field: stage in passed
        for field, stage in _READINESS_STAGE_FIELDS
    }
    value["slurm_ready"] = all(value.values())
    return value


def _provenance(inputs: ModelPreflightInputs) -> dict[str, Any]:
    value = {
        "source_commit": inputs.source_commit,
        "package_sha256": inputs.package_sha256,
        "container_sha256": inputs.container_sha256,
        "dataset_manifest_sha256": inputs.dataset_manifest_sha256,
        "annotation_contract_sha256": inputs.annotation_contract_sha256,
        "merged_spec_sha256": inputs.merged_spec_sha256,
        "latency_protocol_sha256": inputs.latency_protocol_sha256,
        "output_contract_sha256": inputs.output_contract_sha256,
        "ptms": [
            {
                "id": ptm.id,
                "checkpoint_sha256": ptm.checkpoint_sha256,
                "registry_record_sha256": ptm.registry_record_sha256,
                "ptm_preflight_sha256": ptm.ptm_preflight_sha256,
            }
            for ptm in inputs.eligible_ptms
        ],
    }
    value["provenance_sha256"] = canonical_sha256(value)
    return value


def _seal_report(
    *,
    inputs: ModelPreflightInputs,
    records: Sequence[Mapping[str, Any]],
    completion_state: str,
) -> dict[str, Any]:
    canonical_records = _canonical_roundtrip(list(records))
    readiness = _readiness(canonical_records)
    failure = None
    if completion_state == "failed":
        failed = canonical_records[-1]
        failure = {
            "stage": failed["stage"],
            "code": failed["code"],
            "reason": failed["reason"],
        }
    value = {
        "schema_version": MODEL_PREFLIGHT_SCHEMA_VERSION,
        "inputs": inputs.to_dict(),
        "inputs_sha256": inputs.canonical_sha256,
        "provenance": _provenance(inputs),
        "required_stage_order": list(MODEL_PREFLIGHT_STAGES),
        "records": canonical_records,
        "completion_state": completion_state,
        "failure": failure,
        "readiness": readiness,
        "slurm_ready": readiness["slurm_ready"],
    }
    value["report_sha256"] = canonical_sha256(value)
    return value


def validate_model_preflight_report(
    report: Mapping[str, Any],
    *,
    expected_inputs: ModelPreflightInputs | None = None,
) -> dict[str, Any]:
    """Validate hashes, exact stage prefix, failure state, and readiness."""
    item = _require_exact_keys(
        report,
        name="report",
        keys=(
            "schema_version",
            "inputs",
            "inputs_sha256",
            "provenance",
            "required_stage_order",
            "records",
            "completion_state",
            "failure",
            "readiness",
            "slurm_ready",
            "report_sha256",
        ),
    )
    supplied_report_sha = item.pop("report_sha256")
    _require_sha256(supplied_report_sha, "report.report_sha256")
    if canonical_sha256(item) != supplied_report_sha:
        raise ModelPreflightValidationError(
            ("report.report_sha256 does not match report content",)
        )
    item["report_sha256"] = supplied_report_sha

    inputs = ModelPreflightInputs.from_dict(item["inputs"])
    if item["inputs_sha256"] != inputs.canonical_sha256:
        raise ModelPreflightValidationError(
            ("report.inputs_sha256 does not match inputs",)
        )
    if (
        expected_inputs is not None
        and inputs.canonical_sha256 != expected_inputs.canonical_sha256
    ):
        raise ModelPreflightResumeError(
            "resume inputs do not match the sealed preflight inputs"
        )
    if item["schema_version"] != MODEL_PREFLIGHT_SCHEMA_VERSION:
        raise ModelPreflightValidationError(
            ("report.schema_version is unsupported",)
        )
    if item["required_stage_order"] != list(MODEL_PREFLIGHT_STAGES):
        raise ModelPreflightValidationError(
            ("report.required_stage_order does not match the production contract",)
        )
    if item["provenance"] != _provenance(inputs):
        raise ModelPreflightValidationError(
            ("report.provenance does not match sealed inputs",)
        )

    records = item["records"]
    if not isinstance(records, list):
        raise ModelPreflightValidationError(("report.records must be a list",))
    if len(records) > len(MODEL_PREFLIGHT_STAGES):
        raise ModelPreflightValidationError(
            ("report contains unexpected or duplicate stages",)
        )
    record_hashes: list[str] = []
    validated_records: list[dict[str, Any]] = []
    saw_failure = False
    for index, record_value in enumerate(records):
        record = _require_exact_keys(
            record_value,
            name=f"report.records[{index}]",
            keys=(
                "stage",
                "stage_index",
                "request_sha256",
                "status",
                "code",
                "reason",
                "evidence",
                "evidence_sha256",
                "previous_record_sha256",
                "record_sha256",
            ),
        )
        if record["stage"] != MODEL_PREFLIGHT_STAGES[index]:
            raise ModelPreflightValidationError(
                (
                    "report stages must be an exact prefix in production order; "
                    f"expected {MODEL_PREFLIGHT_STAGES[index]!r} at index {index}",
                )
            )
        if record["stage_index"] != index:
            raise ModelPreflightValidationError(
                (f"report.records[{index}].stage_index is invalid",)
            )
        request = ModelPreflightStepRequest(
            inputs=inputs,
            stage=record["stage"],
            stage_index=index,
            prior_record_sha256s=tuple(record_hashes),
        )
        if record["request_sha256"] != request.canonical_sha256:
            raise ModelPreflightValidationError(
                (f"report.records[{index}].request_sha256 is invalid",)
            )
        previous = record_hashes[-1] if record_hashes else None
        if record["previous_record_sha256"] != previous:
            raise ModelPreflightValidationError(
                (f"report.records[{index}] breaks the record hash chain",)
            )
        if record["evidence_sha256"] != canonical_sha256(record["evidence"]):
            raise ModelPreflightValidationError(
                (f"report.records[{index}].evidence_sha256 is invalid",)
            )
        supplied = record.pop("record_sha256")
        if canonical_sha256(record) != supplied:
            raise ModelPreflightValidationError(
                (f"report.records[{index}].record_sha256 is invalid",)
            )
        record["record_sha256"] = supplied
        record_hashes.append(supplied)
        if record["status"] not in ("passed", "failed"):
            raise ModelPreflightValidationError(
                (f"report.records[{index}].status is invalid",)
            )
        if (
            not isinstance(record["code"], str)
            or _SAFE_CODE_RE.fullmatch(record["code"]) is None
        ):
            raise ModelPreflightValidationError(
                (f"report.records[{index}].code is unsafe",)
            )
        if saw_failure:
            raise ModelPreflightValidationError(
                ("no stage may follow a failed stage",)
            )
        saw_failure = record["status"] == "failed"
        if record["stage"] == "metric_sanity":
            expected_metric_record = _metric_sanity_record(
                request,
                validated_records,
            )
            if record != expected_metric_record:
                raise ModelPreflightValidationError(
                    (
                        "report metric_sanity record is not the derived "
                        "task-aware decision",
                    )
                )
        elif record["status"] == "passed":
            if record["code"] != "ok" or record["reason"] != _reason(
                "ok",
                record["stage"],
            ):
                raise ModelPreflightValidationError(
                    (
                        f"passed stage {record['stage']!r} must use the "
                        "canonical success code and reason",
                    )
                )
            expected_evidence = _EVIDENCE_VALIDATORS[record["stage"]](
                record["evidence"],
                inputs,
            )
            if record["evidence"] != expected_evidence:
                raise ModelPreflightValidationError(
                    (
                        f"report evidence for stage {record['stage']!r} "
                        "is not canonically normalized",
                    )
                )
            if record["stage"] == "checkpoint_save_reload":
                epoch = _prior_stage_evidence(
                    validated_records,
                    "default_model_full_epoch",
                )
                if (
                    record["evidence"]["saved_checkpoint_sha256"]
                    != epoch["final_checkpoint_sha256"]
                ):
                    raise ModelPreflightValidationError(
                        (
                            "report checkpoint evidence does not match the "
                            "full-epoch checkpoint",
                        )
                    )
        else:
            _validate_failure_evidence(
                stage=record["stage"],
                code=record["code"],
                reason=record["reason"],
                evidence=record["evidence"],
            )
        validated_records.append(record)

    state = item["completion_state"]
    if state not in ("completed", "failed", "interrupted"):
        raise ModelPreflightValidationError(
            ("report.completion_state is invalid",)
        )
    if state == "completed":
        if len(records) != len(MODEL_PREFLIGHT_STAGES):
            raise ModelPreflightValidationError(
                ("a completed report must contain every required stage",)
            )
        if any(record["status"] != "passed" for record in records):
            raise ModelPreflightValidationError(
                ("a completed report cannot contain a failed stage",)
            )
        if item["failure"] is not None:
            raise ModelPreflightValidationError(
                ("a completed report cannot contain failure metadata",)
            )
    elif state == "failed":
        if not records or records[-1]["status"] != "failed":
            raise ModelPreflightValidationError(
                ("a failed report must end with a failed stage",)
            )
        expected_failure = {
            "stage": records[-1]["stage"],
            "code": records[-1]["code"],
            "reason": records[-1]["reason"],
        }
        if item["failure"] != expected_failure:
            raise ModelPreflightValidationError(
                ("report.failure does not match the failed stage",)
            )
    else:
        if not records or len(records) >= len(MODEL_PREFLIGHT_STAGES):
            raise ModelPreflightValidationError(
                ("an interrupted report must contain a non-empty strict prefix",)
            )
        if any(record["status"] != "passed" for record in records):
            raise ModelPreflightValidationError(
                ("an interrupted report may contain only passed stages",)
            )
        if item["failure"] is not None:
            raise ModelPreflightValidationError(
                ("an interrupted report cannot contain failure metadata",)
            )

    expected_readiness = _readiness(records)
    if item["readiness"] != expected_readiness:
        raise ModelPreflightValidationError(
            ("report.readiness is not derived from passed stage evidence",)
        )
    if item["slurm_ready"] is not expected_readiness["slurm_ready"]:
        raise ModelPreflightValidationError(
            ("report.slurm_ready is not derived from all required gates",)
        )
    if item["slurm_ready"] and state != "completed":
        raise ModelPreflightValidationError(
            ("only a completed report may be SLURM-ready",)
        )
    return _canonical_roundtrip(item)


def run_model_preflight(
    inputs: ModelPreflightInputs,
    adapter: ModelPreflightAdapter,
    *,
    resume_report: Mapping[str, Any] | None = None,
    stop_after_stage: str | None = None,
) -> dict[str, Any]:
    """Run or resume the deterministic local-preflight stage contract.

    ``stop_after_stage`` models a controlled scheduler interruption.  The
    returned report remains a validated content-addressed prefix and can be
    supplied through ``resume_report``.  A failed report is terminal and may
    not be resumed; a caller must start a new preflight identity after fixing
    the underlying issue.
    """
    if not isinstance(inputs, ModelPreflightInputs):
        raise TypeError("inputs must be ModelPreflightInputs")
    if stop_after_stage is not None and stop_after_stage not in MODEL_PREFLIGHT_STAGES:
        raise ModelPreflightValidationError(
            ("stop_after_stage is not a recognized preflight stage",)
        )

    records: list[dict[str, Any]] = []
    if resume_report is not None:
        prior = validate_model_preflight_report(
            resume_report,
            expected_inputs=inputs,
        )
        if prior["completion_state"] == "failed":
            raise ModelPreflightResumeError(
                "a terminal failed preflight report cannot be resumed"
            )
        if prior["completion_state"] == "completed":
            return prior
        records.extend(prior["records"])

    start = len(records)
    for index in range(start, len(MODEL_PREFLIGHT_STAGES)):
        stage = MODEL_PREFLIGHT_STAGES[index]
        request = ModelPreflightStepRequest(
            inputs=inputs,
            stage=stage,
            stage_index=index,
            prior_record_sha256s=tuple(
                record["record_sha256"] for record in records
            ),
        )
        if stage in _DERIVED_STAGES:
            record = _metric_sanity_record(request, records)
        else:
            record = _adapter_record(request, adapter)
        if (
            stage == "checkpoint_save_reload"
            and record["status"] == "passed"
        ):
            epoch = _prior_stage_evidence(
                records,
                "default_model_full_epoch",
            )
            if (
                record["evidence"]["saved_checkpoint_sha256"]
                != epoch["final_checkpoint_sha256"]
            ):
                record = _record(
                    request=request,
                    passed=False,
                    code="invalid_stage_evidence",
                    evidence={
                        "adapter_evidence_sha256": record[
                            "evidence_sha256"
                        ],
                    },
                )
        records.append(record)
        if record["status"] == "failed":
            report = _seal_report(
                inputs=inputs,
                records=records,
                completion_state="failed",
            )
            return validate_model_preflight_report(
                report,
                expected_inputs=inputs,
            )
        if stop_after_stage == stage and index < len(MODEL_PREFLIGHT_STAGES) - 1:
            report = _seal_report(
                inputs=inputs,
                records=records,
                completion_state="interrupted",
            )
            return validate_model_preflight_report(
                report,
                expected_inputs=inputs,
            )

    report = _seal_report(
        inputs=inputs,
        records=records,
        completion_state="completed",
    )
    return validate_model_preflight_report(report, expected_inputs=inputs)


__all__ = [
    "CANONICAL_MODEL_IDS",
    "CANONICAL_MODEL_TASKS",
    "MODEL_PREFLIGHT_SCHEMA_VERSION",
    "MODEL_PREFLIGHT_STAGES",
    "ModelPreflightAdapter",
    "ModelPreflightInputs",
    "ModelPreflightResumeError",
    "ModelPreflightStepRequest",
    "ModelPreflightStepResult",
    "ModelPreflightValidationError",
    "PreflightPTMIdentity",
    "canonical_sha256",
    "run_model_preflight",
    "validate_model_preflight_report",
]
