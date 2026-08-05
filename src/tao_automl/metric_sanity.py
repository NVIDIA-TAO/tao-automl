# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-aware validation sanity gates for AutoML metric evidence.

This module answers one narrow question: is a reported model-quality metric
credible enough for an experiment to continue? It deliberately does *not*
decide whether a candidate satisfies latency mode's retained-accuracy policy.
That product decision remains archive-relative and belongs to the selector.

Policies are model-, task-, metric-, and scale-specific. Unknown or
repository-unverified metric contracts fail closed; there is no universal
``metric >= 0.1`` rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from numbers import Integral
from types import MappingProxyType
from typing import Any, Iterable


def _is_boolean(value: Any) -> bool:
    """Reject Python and NumPy booleans without coercing them to 0/1."""
    return isinstance(value, bool) or (
        type(value).__module__ == "numpy"
        and type(value).__name__ in {"bool", "bool_"}
    )


def _canonical_sha256(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _finite_float(value: Any) -> float | None:
    if _is_boolean(value) or isinstance(value, (str, bytes)):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def _normalized_name(value: Any) -> str:
    return str(value).strip().lower()


@dataclass(frozen=True)
class EvidencePolicy:
    """Minimum experiment evidence required by a sanity gate."""

    minimum_completed_evaluations: int = 1
    minimum_distinct_training_steps: int = 1
    require_annotation_contract: bool = True
    require_standalone_evaluation: bool = True
    require_runtime_metric_contract: bool = True
    require_observed_improvement: bool = False
    minimum_improvement: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "minimum_completed_evaluations",
            "minimum_distinct_training_steps",
        ):
            value = getattr(self, name)
            if _is_boolean(value) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, int(value))
        for name in (
            "require_annotation_contract",
            "require_standalone_evaluation",
            "require_runtime_metric_contract",
            "require_observed_improvement",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        improvement = _finite_float(self.minimum_improvement)
        if improvement is None or improvement < 0.0:
            raise ValueError("minimum_improvement must be finite and >= 0")
        if not self.require_observed_improvement and improvement != 0.0:
            raise ValueError(
                "minimum_improvement requires "
                "require_observed_improvement=True"
            )
        object.__setattr__(self, "minimum_improvement", improvement)

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_completed_evaluations": (
                self.minimum_completed_evaluations
            ),
            "minimum_distinct_training_steps": (
                self.minimum_distinct_training_steps
            ),
            "require_annotation_contract": self.require_annotation_contract,
            "require_standalone_evaluation": (
                self.require_standalone_evaluation
            ),
            "require_runtime_metric_contract": (
                self.require_runtime_metric_contract
            ),
            "require_observed_improvement": (
                self.require_observed_improvement
            ),
            "minimum_improvement": self.minimum_improvement,
        }


@dataclass(frozen=True)
class MetricEvidence:
    """Evidence supplied by a model/dataset preflight or campaign."""

    completed_evaluations: Any
    distinct_training_steps: Any
    annotation_contract_verified: Any
    standalone_evaluation_passed: Any
    runtime_metric_contract_verified: Any
    first_metric_value: Any = None
    best_metric_value: Any = None


@dataclass(frozen=True)
class MetricSanityOverride:
    """Preregistered experiment-specific narrowing of a metric policy."""

    minimum_value: Any = None
    maximum_value: Any = None
    evidence_policy: EvidencePolicy | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_value": self.minimum_value,
            "maximum_value": self.maximum_value,
            "evidence_policy": (
                self.evidence_policy.to_dict()
                if self.evidence_policy is not None
                else None
            ),
        }


@dataclass(frozen=True)
class StructuredReason:
    """Machine-readable reason emitted by a sanity decision."""

    code: str
    message: str
    details: tuple[tuple[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class MetricSanityPolicy:
    """One repository-verified model/task/metric contract."""

    policy_id: str
    model: str
    model_aliases: tuple[str, ...]
    task: str
    metric: str
    metric_aliases: tuple[str, ...]
    direction: str
    scale: str
    valid_minimum: float | None
    valid_maximum: float | None
    availability: str
    availability_reason: str
    evidence_policy: EvidencePolicy
    source_evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.direction != "maximize":
            raise ValueError("quality metric policies must currently maximize")
        if self.scale not in {"fraction", "percent", "unverified"}:
            raise ValueError("metric scale must be fraction, percent, or unverified")
        if self.availability not in {"supported", "blocked"}:
            raise ValueError("metric availability must be supported or blocked")
        if not isinstance(self.evidence_policy, EvidencePolicy):
            raise TypeError("evidence_policy must be EvidencePolicy")
        if not self.policy_id or not self.model or not self.metric:
            raise ValueError("policy_id, model, and metric must be non-empty")
        if not self.task or not self.availability_reason:
            raise ValueError("task and availability_reason must be non-empty")
        if not self.source_evidence or not all(
            isinstance(item, str) and item.strip()
            for item in self.source_evidence
        ):
            raise ValueError(
                "metric policies require non-empty repository source evidence"
            )
        if self.availability == "supported":
            if self.scale == "unverified":
                raise ValueError(
                    "supported metric policies require a verified scale"
                )
            minimum = _finite_float(self.valid_minimum)
            maximum = _finite_float(self.valid_maximum)
            if minimum is None or maximum is None or minimum > maximum:
                raise ValueError(
                    "supported metric policies require finite ordered bounds"
                )
            scale_minimum, scale_maximum = {
                "fraction": (0.0, 1.0),
                "percent": (0.0, 100.0),
            }[self.scale]
            if minimum < scale_minimum or maximum > scale_maximum:
                raise ValueError(
                    f"{self.scale} policy bounds must remain within "
                    f"[{scale_minimum}, {scale_maximum}]"
                )
            object.__setattr__(self, "valid_minimum", minimum)
            object.__setattr__(self, "valid_maximum", maximum)

    def to_dict(self) -> dict[str, Any]:
        """Return canonical, stable policy metadata."""
        return {
            "policy_id": self.policy_id,
            "model": self.model,
            "model_aliases": sorted(set(self.model_aliases)),
            "task": self.task,
            "metric": self.metric,
            "metric_aliases": sorted(set(self.metric_aliases)),
            "direction": self.direction,
            "scale": self.scale,
            "valid_minimum": self.valid_minimum,
            "valid_maximum": self.valid_maximum,
            "availability": self.availability,
            "availability_reason": self.availability_reason,
            "evidence_policy": self.evidence_policy.to_dict(),
            "source_evidence": sorted(set(self.source_evidence)),
        }

    @property
    def canonical_sha256(self) -> str:
        """Return an immutable identity for this exact policy contract."""
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class MetricSanityDecision:
    """Structured result of the validation-only sanity gate."""

    passed: bool
    model: str
    task: str
    metric: str
    metric_value: float | None
    reported_value_repr: str
    direction: str
    scale: str
    policy_availability: str
    effective_minimum: float | None
    effective_maximum: float | None
    evidence_policy: EvidencePolicy
    policy_id: str
    policy_sha256: str
    effective_policy_sha256: str
    registry_sha256: str
    gate_type: str
    product_latency_feasibility: str
    reasons: tuple[StructuredReason, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "model": self.model,
            "task": self.task,
            "metric": self.metric,
            "metric_value": self.metric_value,
            "reported_value_repr": self.reported_value_repr,
            "direction": self.direction,
            "scale": self.scale,
            "policy_availability": self.policy_availability,
            "effective_minimum": self.effective_minimum,
            "effective_maximum": self.effective_maximum,
            "evidence_policy": self.evidence_policy.to_dict(),
            "policy_id": self.policy_id,
            "policy_sha256": self.policy_sha256,
            "effective_policy_sha256": self.effective_policy_sha256,
            "registry_sha256": self.registry_sha256,
            "gate_type": self.gate_type,
            "product_latency_feasibility": (
                self.product_latency_feasibility
            ),
            "reasons": [reason.to_dict() for reason in self.reasons],
        }


class UnknownMetricPolicyError(ValueError):
    """Raised when a model/metric contract has not been registered."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        model: str,
        metric: str,
        available: Iterable[str] = (),
    ):
        super().__init__(message)
        self.code = code
        self.model = model
        self.metric = metric
        self.available = tuple(sorted(available))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "model": self.model,
            "metric": self.metric,
            "available": list(self.available),
        }


class MetricSanityPolicyRegistry:
    """Immutable task-aware policy lookup and evaluation surface."""

    schema_version = 1

    def __init__(self, policies: Iterable[MetricSanityPolicy]):
        ordered = tuple(sorted(policies, key=lambda item: item.policy_id))
        if not ordered:
            raise ValueError("at least one metric sanity policy is required")
        by_model: dict[str, list[MetricSanityPolicy]] = {}
        seen_policy_ids = set()
        for policy in ordered:
            if policy.policy_id in seen_policy_ids:
                raise ValueError(
                    f"duplicate metric policy ID {policy.policy_id!r}"
                )
            seen_policy_ids.add(policy.policy_id)
            model_names = {
                _normalized_name(policy.model),
                *(_normalized_name(alias) for alias in policy.model_aliases),
            }
            for model_name in model_names:
                by_model.setdefault(model_name, []).append(policy)

        lookup = {}
        for model_name, model_policies in by_model.items():
            for policy in model_policies:
                metric_names = {
                    _normalized_name(policy.metric),
                    *(
                        _normalized_name(alias)
                        for alias in policy.metric_aliases
                    ),
                }
                for metric_name in metric_names:
                    key = (model_name, metric_name)
                    existing = lookup.get(key)
                    if existing is not None and existing != policy:
                        raise ValueError(
                            "ambiguous metric policy alias "
                            f"{model_name!r}/{metric_name!r}"
                        )
                    lookup[key] = policy
        self._policies = ordered
        self._by_model = MappingProxyType({
            name: tuple(items) for name, items in by_model.items()
        })
        self._lookup = MappingProxyType(lookup)

    @property
    def policies(self) -> tuple[MetricSanityPolicy, ...]:
        return self._policies

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policies": [policy.to_dict() for policy in self._policies],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
        )

    @property
    def canonical_sha256(self) -> str:
        """Return an order-independent identity for the complete registry."""
        return _canonical_sha256(self.to_dict())

    def resolve(self, model: str, metric: str) -> MetricSanityPolicy:
        """Resolve an exact model/metric contract or fail closed."""
        model_name = _normalized_name(model)
        metric_name = _normalized_name(metric)
        model_policies = self._by_model.get(model_name)
        if model_policies is None:
            canonical_models = sorted(
                {policy.model for policy in self._policies}
            )
            raise UnknownMetricPolicyError(
                code="unknown_model_policy",
                message=(
                    f"No metric sanity policies are registered for model "
                    f"{model!r}"
                ),
                model=str(model),
                metric=str(metric),
                available=canonical_models,
            )
        policy = self._lookup.get((model_name, metric_name))
        if policy is None:
            available_metrics = sorted(
                {item.metric for item in model_policies}
            )
            raise UnknownMetricPolicyError(
                code="unknown_metric_policy",
                message=(
                    f"No metric sanity policy is registered for model "
                    f"{model!r} and metric {metric!r}"
                ),
                model=str(model),
                metric=str(metric),
                available=available_metrics,
            )
        return policy

    def evaluate(
        self,
        model: str,
        metric: str,
        value: Any,
        *,
        evidence: MetricEvidence | None,
        override: MetricSanityOverride | None = None,
    ) -> MetricSanityDecision:
        """Evaluate a validation sanity gate without product selection."""
        policy = self.resolve(model, metric)
        effective = _effective_policy(policy, override)
        reasons = []
        normalized_value = _finite_float(value)

        if policy.availability != "supported":
            reasons.append(
                StructuredReason(
                    code="metric_policy_blocked",
                    message=policy.availability_reason,
                    details=(
                        ("policy_id", policy.policy_id),
                        ("scale", policy.scale),
                    ),
                )
            )
        elif _is_boolean(value):
            reasons.append(
                StructuredReason(
                    code="metric_value_boolean",
                    message="Boolean metric values are invalid",
                )
            )
        elif normalized_value is None:
            reasons.append(
                StructuredReason(
                    code="metric_value_not_finite",
                    message="Metric value must be numeric and finite",
                )
            )
        else:
            if normalized_value < effective["valid_minimum"]:
                reasons.append(
                    StructuredReason(
                        code="metric_below_valid_range",
                        message="Metric value is below the configured range",
                        details=(
                            ("minimum", effective["valid_minimum"]),
                            ("actual", normalized_value),
                            ("scale", policy.scale),
                        ),
                    )
                )
            if normalized_value > effective["valid_maximum"]:
                reasons.append(
                    StructuredReason(
                        code="metric_above_valid_range",
                        message="Metric value is above the configured range",
                        details=(
                            ("maximum", effective["valid_maximum"]),
                            ("actual", normalized_value),
                            ("scale", policy.scale),
                        ),
                    )
                )

        if policy.availability == "supported":
            reasons.extend(
                _evaluate_evidence(
                    evidence,
                    effective["evidence_policy"],
                    valid_minimum=effective["valid_minimum"],
                    valid_maximum=effective["valid_maximum"],
                )
            )

        effective_sha256 = _canonical_sha256({
            "base_policy_sha256": policy.canonical_sha256,
            "valid_minimum": effective["valid_minimum"],
            "valid_maximum": effective["valid_maximum"],
            "evidence_policy": effective["evidence_policy"].to_dict(),
        })
        return MetricSanityDecision(
            passed=not reasons,
            model=policy.model,
            task=policy.task,
            metric=policy.metric,
            metric_value=normalized_value,
            reported_value_repr=repr(value),
            direction=policy.direction,
            scale=policy.scale,
            policy_availability=policy.availability,
            effective_minimum=effective["valid_minimum"],
            effective_maximum=effective["valid_maximum"],
            evidence_policy=effective["evidence_policy"],
            policy_id=policy.policy_id,
            policy_sha256=policy.canonical_sha256,
            effective_policy_sha256=effective_sha256,
            registry_sha256=self.canonical_sha256,
            gate_type="validation_sanity_gate",
            product_latency_feasibility="not_evaluated",
            reasons=tuple(reasons),
        )


def _effective_policy(
    policy: MetricSanityPolicy,
    override: MetricSanityOverride | None,
) -> dict[str, Any]:
    if override is not None and not isinstance(override, MetricSanityOverride):
        raise TypeError("metric sanity override must be MetricSanityOverride")
    minimum = policy.valid_minimum
    maximum = policy.valid_maximum
    evidence_policy = policy.evidence_policy
    if override is not None and policy.availability == "supported":
        if override.minimum_value is not None:
            minimum = _finite_float(override.minimum_value)
            if minimum is None:
                raise ValueError("override minimum_value must be finite")
        if override.maximum_value is not None:
            maximum = _finite_float(override.maximum_value)
            if maximum is None:
                raise ValueError("override maximum_value must be finite")
        if minimum < policy.valid_minimum or maximum > policy.valid_maximum:
            raise ValueError(
                "metric sanity overrides may narrow but cannot expand the "
                "repository-verified metric scale"
            )
        if minimum > maximum:
            raise ValueError(
                "metric sanity override minimum cannot exceed maximum"
            )
        if override.evidence_policy is not None:
            if not isinstance(override.evidence_policy, EvidencePolicy):
                raise TypeError(
                    "override evidence_policy must be EvidencePolicy"
                )
            evidence_policy = override.evidence_policy
    return {
        "valid_minimum": minimum,
        "valid_maximum": maximum,
        "evidence_policy": evidence_policy,
    }


def _nonnegative_int(value: Any) -> int | None:
    if _is_boolean(value) or not isinstance(value, Integral) or value < 0:
        return None
    return int(value)


def _evaluate_evidence(
    evidence: MetricEvidence | None,
    policy: EvidencePolicy,
    *,
    valid_minimum: float,
    valid_maximum: float,
) -> list[StructuredReason]:
    reasons = []
    if evidence is None:
        return [
            StructuredReason(
                code="metric_evidence_missing",
                message=(
                    "Metric sanity requires explicit annotation, runtime, "
                    "training, and standalone-evaluation evidence"
                ),
            )
        ]
    if not isinstance(evidence, MetricEvidence):
        raise TypeError("metric evidence must be MetricEvidence")

    completed = _nonnegative_int(evidence.completed_evaluations)
    if completed is None:
        reasons.append(
            StructuredReason(
                code="completed_evaluations_invalid",
                message="completed_evaluations must be a non-negative integer",
            )
        )
    elif completed < policy.minimum_completed_evaluations:
        reasons.append(
            StructuredReason(
                code="completed_evaluations_insufficient",
                message="Too few completed evaluations for the sanity policy",
                details=(
                    ("required", policy.minimum_completed_evaluations),
                    ("actual", completed),
                ),
            )
        )

    distinct_steps = _nonnegative_int(evidence.distinct_training_steps)
    if distinct_steps is None:
        reasons.append(
            StructuredReason(
                code="distinct_training_steps_invalid",
                message=(
                    "distinct_training_steps must be a non-negative integer"
                ),
            )
        )
    elif distinct_steps < policy.minimum_distinct_training_steps:
        reasons.append(
            StructuredReason(
                code="distinct_training_steps_insufficient",
                message="Too few distinct training steps for the sanity policy",
                details=(
                    ("required", policy.minimum_distinct_training_steps),
                    ("actual", distinct_steps),
                ),
            )
        )

    boolean_requirements = (
        (
            policy.require_annotation_contract,
            evidence.annotation_contract_verified,
            "annotation_contract_not_verified",
            "Dataset annotation and label contract is not verified",
        ),
        (
            policy.require_standalone_evaluation,
            evidence.standalone_evaluation_passed,
            "standalone_evaluation_not_passed",
            "Standalone evaluation has not passed",
        ),
        (
            policy.require_runtime_metric_contract,
            evidence.runtime_metric_contract_verified,
            "runtime_metric_contract_not_verified",
            "Runtime metric name and scale have not been verified",
        ),
    )
    for required, actual, code, message in boolean_requirements:
        if required and actual is not True:
            reasons.append(StructuredReason(code=code, message=message))

    if policy.require_observed_improvement:
        first = _finite_float(evidence.first_metric_value)
        best = _finite_float(evidence.best_metric_value)
        if first is None or best is None:
            reasons.append(
                StructuredReason(
                    code="learning_evidence_missing",
                    message=(
                        "Finite first and best metric values are required to "
                        "verify learning"
                    ),
                )
            )
        elif (
            first < valid_minimum
            or first > valid_maximum
            or best < valid_minimum
            or best > valid_maximum
        ):
            reasons.append(
                StructuredReason(
                    code="learning_evidence_out_of_range",
                    message=(
                        "First and best learning metrics must be within the "
                        "effective metric range"
                    ),
                    details=(
                        ("first", first),
                        ("best", best),
                        ("minimum", valid_minimum),
                        ("maximum", valid_maximum),
                    ),
                )
            )
        else:
            improvement = best - first
            minimum = policy.minimum_improvement
            passed = (
                improvement > 0.0
                if minimum == 0.0
                else improvement >= minimum
            )
            if not passed:
                reasons.append(
                    StructuredReason(
                        code="minimum_learning_not_observed",
                        message=(
                            "Observed metric improvement does not satisfy the "
                            "preregistered learning policy"
                        ),
                        details=(
                            ("first", first),
                            ("best", best),
                            ("improvement", improvement),
                            ("required", minimum),
                        ),
                    )
                )
    return reasons


_DEFAULT_EVIDENCE = EvidencePolicy()

_DETECTION_SOURCE = (
    "tao-pytorch:nvidia_tao_pytorch/cv/{model}/model/"
    "{module}: COCOeval bbox.stats[0:2] are logged directly as val_mAP/"
    "val_mAP50 on the COCO fraction scale",
)


def _detection_policies(
    *,
    model: str,
    model_aliases: tuple[str, ...],
    source_model: str,
    source_module: str,
) -> tuple[MetricSanityPolicy, ...]:
    source = tuple(
        item.format(model=source_model, module=source_module)
        for item in _DETECTION_SOURCE
    )
    common = {
        "model": model,
        "model_aliases": model_aliases,
        "task": "object_detection",
        "direction": "maximize",
        "scale": "fraction",
        "valid_minimum": 0.0,
        "valid_maximum": 1.0,
        "availability": "supported",
        "availability_reason": (
            "TAO logs the unscaled COCO evaluator statistic directly"
        ),
        "evidence_policy": _DEFAULT_EVIDENCE,
        "source_evidence": source,
    }
    return (
        MetricSanityPolicy(
            policy_id=f"{model}.coco_bbox_ap",
            metric="val_mAP",
            metric_aliases=("mAP", "bbox_val_mAP", "coco_bbox_ap"),
            **common,
        ),
        MetricSanityPolicy(
            policy_id=f"{model}.coco_bbox_ap50",
            metric="val_mAP50",
            metric_aliases=("mAP50", "bbox_val_mAP50", "coco_bbox_ap50"),
            **common,
        ),
    )


def _default_policies() -> tuple[MetricSanityPolicy, ...]:
    policies = []
    policies.extend(
        _detection_policies(
            model="dino",
            model_aliases=(),
            source_model="dino",
            source_module="pl_dino_model.py",
        )
    )
    policies.extend(
        _detection_policies(
            model="deformable_detr",
            model_aliases=("deformable-detr", "ddetr"),
            source_model="deformable_detr",
            source_module="pl_dd_model.py",
        )
    )
    policies.extend(
        _detection_policies(
            model="rtdetr",
            model_aliases=("rt-detr", "rt_detr"),
            source_model="rtdetr",
            source_module="pl_rtdetr_model.py",
        )
    )
    policies.extend(
        _detection_policies(
            model="grounding_dino",
            model_aliases=("grounding-dino",),
            source_model="grounding_dino",
            source_module="pl_gdino_model.py",
        )
    )
    policies.extend((
        MetricSanityPolicy(
            policy_id="grounding_dino.referring_box_pr50",
            model="grounding_dino",
            model_aliases=("grounding-dino",),
            task="referring_expression_box_grounding",
            metric="val_Pr@0.5",
            metric_aliases=("Pr@0.5", "pr50"),
            direction="maximize",
            scale="unverified",
            valid_minimum=None,
            valid_maximum=None,
            availability="blocked",
            availability_reason=(
                "The inspected Grounding DINO validation loader constructs "
                "CocoDetection and emits COCO bbox AP; it does not implement "
                "a phrase-grounding Pr@0.5 evaluator or verified output scale."
            ),
            evidence_policy=_DEFAULT_EVIDENCE,
            source_evidence=(
                "tao-pytorch:nvidia_tao_pytorch/cv/grounding_dino/"
                "dataloader/pl_odvg_data_module.py: validation uses "
                "CocoDetection",
                "tao-pytorch:nvidia_tao_pytorch/cv/grounding_dino/model/"
                "pl_gdino_model.py: validation emits val_mAP and val_mAP50",
            ),
        ),
        MetricSanityPolicy(
            policy_id="segformer.semantic_miou",
            model="segformer",
            model_aliases=(),
            task="semantic_segmentation",
            metric="val_miou",
            metric_aliases=("mIoU", "val_mIoU", "miou"),
            direction="maximize",
            scale="fraction",
            valid_minimum=0.0,
            valid_maximum=1.0,
            availability="supported",
            availability_reason=(
                "TAO computes intersection/union ratios and logs their "
                "unscaled mean"
            ),
            evidence_policy=_DEFAULT_EVIDENCE,
            source_evidence=(
                "tao-pytorch:nvidia_tao_pytorch/cv/segformer/utils/"
                "iou_metric.py: miou=np.nanmean(intersection/union)",
                "tao-pytorch:nvidia_tao_pytorch/cv/segformer/model/"
                "segformer_pl_model.py: status KPI is val_miou",
            ),
        ),
        MetricSanityPolicy(
            policy_id="oneformer.panoptic_quality",
            model="oneformer",
            model_aliases=(),
            task="panoptic_segmentation",
            metric="PQ",
            metric_aliases=("val_PQ", "panoptic_quality"),
            direction="maximize",
            scale="unverified",
            valid_minimum=None,
            valid_maximum=None,
            availability="blocked",
            availability_reason=(
                "The inspected OneFormer validation path emits semantic mIoU "
                "and pixel accuracy, not panoptic PQ. PQ scale and runtime "
                "metric identity are therefore unverified."
            ),
            evidence_policy=_DEFAULT_EVIDENCE,
            source_evidence=(
                "tao-pytorch:nvidia_tao_pytorch/cv/oneformer/model/"
                "pl_oneformer.py: validation status contains mIoU/ACC",
            ),
        ),
        MetricSanityPolicy(
            policy_id="mask2former.coco_mask_ap",
            model="mask2former",
            model_aliases=("mask2-former",),
            task="instance_segmentation",
            metric="segm_val_mAP",
            metric_aliases=("mask_AP", "mask_mAP", "coco_mask_ap"),
            direction="maximize",
            scale="unverified",
            valid_minimum=None,
            valid_maximum=None,
            availability="blocked",
            availability_reason=(
                "The inspected Mask2Former validation path emits semantic "
                "mIoU and pixel accuracy, not COCO mask AP. Its requested "
                "mask-AP output contract is not implemented."
            ),
            evidence_policy=_DEFAULT_EVIDENCE,
            source_evidence=(
                "tao-pytorch:nvidia_tao_pytorch/cv/mask2former/model/"
                "pl_model.py: validation status contains mIoU/ACC_all",
            ),
        ),
        MetricSanityPolicy(
            policy_id="mask_grounding_dino.coco_mask_ap50_95",
            model="mask_grounding_dino",
            model_aliases=("mask-grounding-dino",),
            task="category_prompted_grounded_instance_segmentation",
            metric="segm_val_mAP50_95",
            metric_aliases=(
                "[segm] val_mAP@50-95",
                "coco_mask_ap",
                "mask_AP",
                "segm_val_mAP",
            ),
            direction="maximize",
            scale="fraction",
            valid_minimum=0.0,
            valid_maximum=1.0,
            availability="supported",
            availability_reason=(
                "The OD evaluation path runs the COCO evaluator for both bbox "
                "and segm IoU types and reports the unscaled COCO AP50-95 "
                "statistic for segmentation"
            ),
            evidence_policy=_DEFAULT_EVIDENCE,
            source_evidence=(
                "tao-pytorch:nvidia_tao_pytorch/cv/mask_grounding_dino/"
                "model/pl_gdino_model.py: OD validation constructs "
                "OD_Evaluator with iou_types=['bbox', 'segm']",
                "tao-pytorch:nvidia_tao_pytorch/cv/mask_grounding_dino/"
                "model/pl_gdino_model.py: OD status keys include "
                "[segm] val_mAP@50-95",
                "tao-pytorch:nvidia_tao_pytorch/cv/grounding_dino/utils/"
                "coco_eval.py: mAP@50-95 is the unscaled COCO summary value",
            ),
        ),
        MetricSanityPolicy(
            policy_id="mask_grounding_dino.overall_iou",
            model="mask_grounding_dino",
            model_aliases=("mask-grounding-dino",),
            task="referring_expression_segmentation",
            metric="val_overall_IoU",
            metric_aliases=("overall_IoU", "overall_iou"),
            direction="maximize",
            scale="percent",
            valid_minimum=0.0,
            valid_maximum=100.0,
            availability="supported",
            availability_reason=(
                "The VG evaluator explicitly multiplies aggregate "
                "intersection-over-union by 100 before reporting overall_IoU"
            ),
            evidence_policy=_DEFAULT_EVIDENCE,
            source_evidence=(
                "tao-pytorch:nvidia_tao_pytorch/cv/mask_grounding_dino/"
                "utils/evaluator.py: overall_IoU = 100 * sum(intersection) / "
                "sum(union)",
                "tao-pytorch:nvidia_tao_pytorch/cv/mask_grounding_dino/"
                "model/pl_gdino_model.py: VG results are status-logged with "
                "a val_ prefix",
            ),
        ),
        MetricSanityPolicy(
            policy_id="mask_grounding_dino.legacy_ciou",
            model="mask_grounding_dino",
            model_aliases=("mask-grounding-dino",),
            task="referring_expression_segmentation",
            metric="val_cIoU",
            metric_aliases=("cIoU", "test_cIoU"),
            direction="maximize",
            scale="unverified",
            valid_minimum=None,
            valid_maximum=None,
            availability="blocked",
            availability_reason=(
                "val_cIoU remains in legacy metric enums/docstrings, but the "
                "inspected evaluator implements and emits overall_IoU instead. "
                "The names must not be treated as interchangeable."
            ),
            evidence_policy=_DEFAULT_EVIDENCE,
            source_evidence=(
                "tao-automl:src/tao_automl/schema/enum_constants.py: "
                "val_cIoU legacy enum",
                "tao-pytorch:nvidia_tao_pytorch/cv/mask_grounding_dino/"
                "utils/evaluator.py: implemented key is overall_IoU",
            ),
        ),
    ))
    return tuple(policies)


@lru_cache(maxsize=1)
def default_metric_sanity_registry() -> MetricSanityPolicyRegistry:
    """Return the immutable repository-owned default policy registry."""
    return MetricSanityPolicyRegistry(_default_policies())


def evaluate_metric_sanity(
    model: str,
    metric: str,
    value: Any,
    *,
    evidence: MetricEvidence | None,
    override: MetricSanityOverride | None = None,
    registry: MetricSanityPolicyRegistry | None = None,
) -> MetricSanityDecision:
    """Convenience wrapper around the default task-aware registry."""
    active_registry = registry or default_metric_sanity_registry()
    return active_registry.evaluate(
        model,
        metric,
        value,
        evidence=evidence,
        override=override,
    )
