# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for task-aware validation metric sanity policies."""

from dataclasses import replace
import json
import math
import re

import numpy as np
import pytest

from tao_automl.metric_sanity import (
    EvidencePolicy,
    MetricEvidence,
    MetricSanityOverride,
    MetricSanityPolicyRegistry,
    UnknownMetricPolicyError,
    default_metric_sanity_registry,
    evaluate_metric_sanity,
)


def _evidence(**changes):
    values = {
        "completed_evaluations": 1,
        "distinct_training_steps": 1,
        "annotation_contract_verified": True,
        "standalone_evaluation_passed": True,
        "runtime_metric_contract_verified": True,
        "first_metric_value": None,
        "best_metric_value": None,
    }
    values.update(changes)
    return MetricEvidence(**values)


def _reason_codes(decision):
    return {reason.code for reason in decision.reasons}


@pytest.mark.parametrize(
    ("model", "metric", "task", "scale", "availability"),
    [
        ("dino", "val_mAP", "object_detection", "fraction", "supported"),
        (
            "deformable_detr",
            "val_mAP50",
            "object_detection",
            "fraction",
            "supported",
        ),
        ("rtdetr", "val_mAP", "object_detection", "fraction", "supported"),
        (
            "grounding_dino",
            "val_mAP50",
            "object_detection",
            "fraction",
            "supported",
        ),
        (
            "grounding_dino",
            "val_Pr@0.5",
            "referring_expression_box_grounding",
            "unverified",
            "blocked",
        ),
        (
            "segformer",
            "val_miou",
            "semantic_segmentation",
            "fraction",
            "supported",
        ),
        (
            "oneformer",
            "PQ",
            "panoptic_segmentation",
            "unverified",
            "blocked",
        ),
        (
            "mask2former",
            "segm_val_mAP",
            "instance_segmentation",
            "unverified",
            "blocked",
        ),
        (
            "mask_grounding_dino",
            "segm_val_mAP50_95",
            "category_prompted_grounded_instance_segmentation",
            "fraction",
            "supported",
        ),
        (
            "mask_grounding_dino",
            "val_overall_IoU",
            "referring_expression_segmentation",
            "percent",
            "supported",
        ),
        (
            "mask_grounding_dino",
            "val_cIoU",
            "referring_expression_segmentation",
            "unverified",
            "blocked",
        ),
    ],
)
def test_registry_covers_required_model_task_metric_contracts(
    model,
    metric,
    task,
    scale,
    availability,
):
    policy = default_metric_sanity_registry().resolve(model, metric)

    assert policy.task == task
    assert policy.scale == scale
    assert policy.availability == availability
    assert policy.source_evidence


@pytest.mark.parametrize(
    "model",
    ["dino", "deformable_detr", "rtdetr", "grounding_dino"],
)
@pytest.mark.parametrize("metric", ["val_mAP", "val_mAP50"])
@pytest.mark.parametrize("value", [0.0, 0.007, 0.5, 1.0])
def test_detection_coco_ap_fraction_range_has_no_universal_point_one_floor(
    model,
    metric,
    value,
):
    decision = evaluate_metric_sanity(
        model,
        metric,
        value,
        evidence=_evidence(),
    )

    assert decision.passed
    assert decision.metric_value == value


@pytest.mark.parametrize(
    "model",
    ["dino", "deformable_detr", "rtdetr", "grounding_dino"],
)
@pytest.mark.parametrize(
    ("value", "expected_code"),
    [
        (-0.00001, "metric_below_valid_range"),
        (1.00001, "metric_above_valid_range"),
    ],
)
def test_detection_coco_ap_rejects_values_outside_fraction_scale(
    model,
    value,
    expected_code,
):
    decision = evaluate_metric_sanity(
        model,
        "val_mAP",
        value,
        evidence=_evidence(),
    )

    assert not decision.passed
    assert expected_code in _reason_codes(decision)


@pytest.mark.parametrize("value", [0.0, 0.001, 1.0])
def test_segformer_miou_uses_fraction_scale(value):
    decision = evaluate_metric_sanity(
        "segformer",
        "val_miou",
        value,
        evidence=_evidence(),
    )

    assert decision.passed


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_segformer_miou_rejects_out_of_range_values(value):
    decision = evaluate_metric_sanity(
        "segformer",
        "val_miou",
        value,
        evidence=_evidence(),
    )

    assert not decision.passed


@pytest.mark.parametrize("value", [0.0, 50.0, 100.0])
def test_mask_grounding_dino_overall_iou_uses_verified_percent_scale(value):
    decision = evaluate_metric_sanity(
        "mask_grounding_dino",
        "val_overall_IoU",
        value,
        evidence=_evidence(),
    )

    assert decision.passed
    assert (
        default_metric_sanity_registry()
        .resolve("mask_grounding_dino", "val_overall_IoU")
        .scale
        == "percent"
    )


@pytest.mark.parametrize("value", [0.0, 0.001, 0.5, 1.0])
def test_mask_grounding_dino_coco_mask_ap50_95_uses_fraction_scale(value):
    decision = evaluate_metric_sanity(
        "mask_grounding_dino",
        "segm_val_mAP50_95",
        value,
        evidence=_evidence(),
    )

    assert decision.passed
    assert decision.metric_value == value
    assert decision.task == (
        "category_prompted_grounded_instance_segmentation"
    )


@pytest.mark.parametrize("value", [-0.00001, 1.00001])
def test_mask_grounding_dino_coco_mask_ap50_95_rejects_out_of_range(value):
    decision = evaluate_metric_sanity(
        "mask-grounding-dino",
        "[segm] val_mAP@50-95",
        value,
        evidence=_evidence(),
    )

    assert not decision.passed


def test_mask_grounding_dino_od_mask_ap_and_vg_iou_are_distinct():
    registry = default_metric_sanity_registry()

    mask_ap = registry.resolve("mask_grounding_dino", "coco_mask_ap")
    vg_iou = registry.resolve("mask_grounding_dino", "overall_IoU")

    assert mask_ap.policy_id != vg_iou.policy_id
    assert mask_ap.task == (
        "category_prompted_grounded_instance_segmentation"
    )
    assert mask_ap.scale == "fraction"
    assert vg_iou.task == "referring_expression_segmentation"
    assert vg_iou.scale == "percent"


@pytest.mark.parametrize(
    ("value", "expected_code"),
    [
        (-0.001, "metric_below_valid_range"),
        (100.001, "metric_above_valid_range"),
    ],
)
def test_mask_grounding_dino_overall_iou_rejects_out_of_percent_range(
    value,
    expected_code,
):
    decision = evaluate_metric_sanity(
        "mask_grounding_dino",
        "val_overall_IoU",
        value,
        evidence=_evidence(),
    )

    assert not decision.passed
    assert expected_code in _reason_codes(decision)


@pytest.mark.parametrize("value", [True, False, np.bool_(True)])
def test_metric_values_reject_booleans_without_numeric_coercion(value):
    decision = evaluate_metric_sanity(
        "dino",
        "val_mAP50",
        value,
        evidence=_evidence(),
    )

    assert not decision.passed
    assert decision.metric_value is None
    assert _reason_codes(decision) == {"metric_value_boolean"}


@pytest.mark.parametrize(
    "value",
    [
        None,
        "0.5",
        "not-a-number",
        float("nan"),
        float("inf"),
        float("-inf"),
        np.float64(np.nan),
    ],
)
def test_metric_values_must_be_numeric_and_finite(value):
    decision = evaluate_metric_sanity(
        "dino",
        "val_mAP50",
        value,
        evidence=_evidence(),
    )

    assert not decision.passed
    assert decision.metric_value is None
    assert "metric_value_not_finite" in _reason_codes(decision)


@pytest.mark.parametrize(
    ("model", "metric"),
    [
        ("grounding_dino", "val_Pr@0.5"),
        ("oneformer", "PQ"),
        ("mask2former", "segm_val_mAP"),
        ("mask_grounding_dino", "val_cIoU"),
    ],
)
def test_unverified_runtime_metric_contracts_fail_closed(model, metric):
    decision = evaluate_metric_sanity(
        model,
        metric,
        0.5,
        evidence=_evidence(),
    )

    assert not decision.passed
    assert _reason_codes(decision) == {"metric_policy_blocked"}
    reason = decision.reasons[0].to_dict()
    assert reason["details"]["scale"] == "unverified"
    assert reason["details"]["policy_id"] == decision.policy_id


def test_grounding_dino_detection_and_referring_metrics_remain_distinct():
    registry = default_metric_sanity_registry()

    detection = registry.resolve("grounding-dino", "mAP50")
    referring = registry.resolve("grounding_dino", "Pr@0.5")

    assert detection.task == "object_detection"
    assert detection.availability == "supported"
    assert detection.scale == "fraction"
    assert referring.task == "referring_expression_box_grounding"
    assert referring.availability == "blocked"
    assert referring.scale == "unverified"
    assert detection.policy_id != referring.policy_id


def test_mask_grounding_dino_legacy_ciou_is_not_overall_iou_alias():
    registry = default_metric_sanity_registry()

    overall = registry.resolve("mask_grounding_dino", "overall_IoU")
    legacy = registry.resolve("mask_grounding_dino", "cIoU")

    assert overall.policy_id != legacy.policy_id
    assert overall.availability == "supported"
    assert legacy.availability == "blocked"


@pytest.mark.parametrize(
    ("model", "metric", "expected_code"),
    [
        ("unknown_model", "val_mAP", "unknown_model_policy"),
        ("dino", "unknown_metric", "unknown_metric_policy"),
    ],
)
def test_unknown_policy_fails_closed_with_structured_error(
    model,
    metric,
    expected_code,
):
    with pytest.raises(UnknownMetricPolicyError) as exc_info:
        evaluate_metric_sanity(
            model,
            metric,
            0.5,
            evidence=_evidence(),
        )

    details = exc_info.value.to_dict()
    assert details["code"] == expected_code
    assert details["model"] == model
    assert details["metric"] == metric
    assert details["available"]


def test_missing_evidence_fails_sanity_gate():
    decision = evaluate_metric_sanity(
        "dino",
        "val_mAP",
        0.5,
        evidence=None,
    )

    assert not decision.passed
    assert _reason_codes(decision) == {"metric_evidence_missing"}


def test_incomplete_evidence_emits_all_relevant_structured_reasons():
    decision = evaluate_metric_sanity(
        "dino",
        "val_mAP",
        0.5,
        evidence=_evidence(
            completed_evaluations=0,
            distinct_training_steps=0,
            annotation_contract_verified=False,
            standalone_evaluation_passed=False,
            runtime_metric_contract_verified=False,
        ),
    )

    assert not decision.passed
    assert _reason_codes(decision) == {
        "completed_evaluations_insufficient",
        "distinct_training_steps_insufficient",
        "annotation_contract_not_verified",
        "standalone_evaluation_not_passed",
        "runtime_metric_contract_not_verified",
    }


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("completed_evaluations", True, "completed_evaluations_invalid"),
        (
            "completed_evaluations",
            np.bool_(True),
            "completed_evaluations_invalid",
        ),
        ("completed_evaluations", 1.0, "completed_evaluations_invalid"),
        ("distinct_training_steps", -1, "distinct_training_steps_invalid"),
    ],
)
def test_evidence_counts_reject_boolean_noninteger_or_negative_values(
    field,
    value,
    expected_code,
):
    decision = evaluate_metric_sanity(
        "dino",
        "val_mAP",
        0.5,
        evidence=_evidence(**{field: value}),
    )

    assert not decision.passed
    assert expected_code in _reason_codes(decision)


def test_numpy_integer_evidence_counts_are_accepted():
    decision = evaluate_metric_sanity(
        "dino",
        "val_mAP",
        0.5,
        evidence=_evidence(
            completed_evaluations=np.int64(1),
            distinct_training_steps=np.int64(1),
        ),
    )

    assert decision.passed


def test_preregistered_minimum_learning_policy_passes_with_sufficient_delta():
    override = MetricSanityOverride(
        evidence_policy=EvidencePolicy(
            minimum_completed_evaluations=2,
            minimum_distinct_training_steps=10,
            require_observed_improvement=True,
            minimum_improvement=0.05,
        )
    )
    decision = evaluate_metric_sanity(
        "segformer",
        "val_miou",
        0.3,
        evidence=_evidence(
            completed_evaluations=2,
            distinct_training_steps=10,
            first_metric_value=0.1,
            best_metric_value=0.16,
        ),
        override=override,
    )

    assert decision.passed


@pytest.mark.parametrize(
    ("first", "best", "expected_code"),
    [
        (None, 0.2, "learning_evidence_missing"),
        (0.1, float("nan"), "learning_evidence_missing"),
        (0.1, 1.1, "learning_evidence_out_of_range"),
        (0.1, 0.149, "minimum_learning_not_observed"),
        (0.2, 0.1, "minimum_learning_not_observed"),
    ],
)
def test_preregistered_minimum_learning_policy_fails_closed(
    first,
    best,
    expected_code,
):
    override = MetricSanityOverride(
        evidence_policy=EvidencePolicy(
            require_observed_improvement=True,
            minimum_improvement=0.05,
        )
    )
    decision = evaluate_metric_sanity(
        "segformer",
        "val_miou",
        0.3,
        evidence=_evidence(
            first_metric_value=first,
            best_metric_value=best,
        ),
        override=override,
    )

    assert not decision.passed
    assert expected_code in _reason_codes(decision)


def test_zero_minimum_learning_requires_strict_positive_improvement():
    override = MetricSanityOverride(
        evidence_policy=EvidencePolicy(
            require_observed_improvement=True,
            minimum_improvement=0.0,
        )
    )

    unchanged = evaluate_metric_sanity(
        "dino",
        "val_mAP",
        0.2,
        evidence=_evidence(
            first_metric_value=0.2,
            best_metric_value=0.2,
        ),
        override=override,
    )
    improved = evaluate_metric_sanity(
        "dino",
        "val_mAP",
        0.2,
        evidence=_evidence(
            first_metric_value=0.1,
            best_metric_value=0.10001,
        ),
        override=override,
    )

    assert "minimum_learning_not_observed" in _reason_codes(unchanged)
    assert improved.passed


def test_override_can_preregister_dataset_floor_without_product_feasibility():
    override = MetricSanityOverride(minimum_value=0.02)

    below = evaluate_metric_sanity(
        "dino",
        "val_mAP",
        0.007,
        evidence=_evidence(),
        override=override,
    )
    above = evaluate_metric_sanity(
        "dino",
        "val_mAP",
        0.02,
        evidence=_evidence(),
        override=override,
    )

    assert not below.passed
    assert "metric_below_valid_range" in _reason_codes(below)
    assert above.passed
    assert above.gate_type == "validation_sanity_gate"
    assert above.product_latency_feasibility == "not_evaluated"


@pytest.mark.parametrize(
    "override",
    [
        MetricSanityOverride(minimum_value=-0.1),
        MetricSanityOverride(maximum_value=1.1),
        MetricSanityOverride(minimum_value=0.8, maximum_value=0.7),
        MetricSanityOverride(minimum_value=True),
        MetricSanityOverride(maximum_value=float("nan")),
    ],
)
def test_override_cannot_expand_scale_or_use_invalid_bounds(override):
    with pytest.raises(ValueError):
        evaluate_metric_sanity(
            "dino",
            "val_mAP",
            0.5,
            evidence=_evidence(),
            override=override,
        )


def test_registry_hash_is_canonical_order_independent_and_json_safe():
    registry = default_metric_sanity_registry()
    reversed_registry = MetricSanityPolicyRegistry(reversed(registry.policies))

    assert registry.canonical_sha256 == reversed_registry.canonical_sha256
    assert re.fullmatch(r"[0-9a-f]{64}", registry.canonical_sha256)
    assert json.loads(registry.to_json()) == registry.to_dict()


def test_registry_hash_changes_when_policy_contract_changes():
    registry = default_metric_sanity_registry()
    changed = list(registry.policies)
    changed[0] = replace(
        changed[0],
        availability_reason="Different repository evidence contract",
    )

    changed_registry = MetricSanityPolicyRegistry(changed)

    assert changed_registry.canonical_sha256 != registry.canonical_sha256


def test_effective_policy_hash_is_stable_and_override_sensitive():
    evidence = _evidence()
    first = evaluate_metric_sanity(
        "dino",
        "val_mAP",
        0.5,
        evidence=evidence,
        override=MetricSanityOverride(minimum_value=0.1),
    )
    identical = evaluate_metric_sanity(
        "dino",
        "val_mAP",
        0.5,
        evidence=evidence,
        override=MetricSanityOverride(minimum_value=0.1),
    )
    different = evaluate_metric_sanity(
        "dino",
        "val_mAP",
        0.5,
        evidence=evidence,
        override=MetricSanityOverride(minimum_value=0.2),
    )

    assert first.effective_policy_sha256 == identical.effective_policy_sha256
    assert first.effective_policy_sha256 != different.effective_policy_sha256
    assert re.fullmatch(r"[0-9a-f]{64}", first.effective_policy_sha256)


def test_decision_serialization_is_structured_and_json_safe():
    decision = evaluate_metric_sanity(
        "dino",
        "val_mAP50",
        2.0,
        evidence=_evidence(),
    )
    payload = decision.to_dict()

    assert not payload["passed"]
    assert payload["gate_type"] == "validation_sanity_gate"
    assert payload["product_latency_feasibility"] == "not_evaluated"
    assert payload["reasons"][0]["code"] == "metric_above_valid_range"
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload


@pytest.mark.parametrize(
    ("field", "value", "expected_exception"),
    [
        ("minimum_completed_evaluations", True, TypeError),
        ("minimum_distinct_training_steps", -1, ValueError),
        ("minimum_improvement", True, ValueError),
        ("minimum_improvement", math.inf, ValueError),
        ("minimum_improvement", 0.1, ValueError),
    ],
)
def test_evidence_policy_configuration_rejects_invalid_values(
    field,
    value,
    expected_exception,
):
    with pytest.raises(expected_exception):
        EvidencePolicy(**{field: value})
