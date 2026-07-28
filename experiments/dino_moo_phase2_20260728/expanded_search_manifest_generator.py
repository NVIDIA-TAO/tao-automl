#!/usr/bin/env python3

"""Derive the immutable expanded DINO search manifest from frozen evidence.

This script intentionally has no launch path.  It validates a complete
sensitivity result and applies the preregistered, direction-agnostic
architecture-axis policy.  Accuracy-retention annotations are preserved for
audit but never participate in axis inclusion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_POLICY = HERE / "expanded_search_derivation_policy.v1.json"
DEFAULT_OUTPUT = HERE / "expanded_search_manifest.v1.json"
HEX = frozenset("0123456789abcdef")


class ContractError(ValueError):
    """Raised when immutable input evidence violates the frozen contract."""


def canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ContractError(f"value is not canonical JSON: {error}") from error
    return text.encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX for character in value)
    ):
        raise ContractError(f"{label} must be a lowercase SHA256 digest")
    return value


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON constant is forbidden: {value}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return value


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ContractError(
            f"{label} mismatch: expected {expected!r}, got {actual!r}"
        )


def require_finite_positive(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ContractError(f"{label} must be a finite positive number")
    return float(value)


def require_finite_interval(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ContractError(f"{label} must contain exactly two values")
    lower, upper = value
    if (
        isinstance(lower, bool)
        or isinstance(upper, bool)
        or not isinstance(lower, (int, float))
        or not isinstance(upper, (int, float))
        or not math.isfinite(float(lower))
        or not math.isfinite(float(upper))
        or float(lower) > float(upper)
    ):
        raise ContractError(f"{label} must be an ordered finite interval")
    return float(lower), float(upper)


def nested_value(mapping: dict[str, Any], dotted_path: str) -> Any:
    current: Any = mapping
    for component in dotted_path.split("."):
        if not isinstance(current, dict) or component not in current:
            raise ContractError(f"missing required field {dotted_path}")
        current = current[component]
    return current


def validate_false_audit_flags(value: Any, path: str = "result") -> None:
    """Fail if any nested manual/selection-feed audit flag is true."""

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            lowered = key.lower()
            guarded = (
                "manual" in lowered
                or key in {"feeds_final_selection", "winner_selected"}
            )
            if guarded and isinstance(child, bool) and child:
                raise ContractError(f"forbidden true audit flag: {child_path}")
            validate_false_audit_flags(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_false_audit_flags(child, f"{path}[{index}]")


def validate_policy(policy: dict[str, Any]) -> None:
    require_equal(policy.get("schema_version"), 1, "policy schema_version")
    require_equal(
        policy.get("policy_id"),
        "dino_expanded_search_derivation_20260728_v1",
        "policy_id",
    )
    require_equal(
        policy.get("status"),
        "preregistered_before_sensitivity_latency_results",
        "policy status",
    )
    require_equal(
        policy.get("manual_override_permitted"),
        False,
        "manual_override_permitted",
    )
    require_equal(
        policy.get("result_dependent_policy_changes_permitted"),
        False,
        "result_dependent_policy_changes_permitted",
    )
    design = policy["search_design"]
    require_equal(design["search_seeds"], [314159, 271828, 161803], "search seeds")
    require_equal(design["training_seed"], 1234, "training seed")
    require_equal(design["recommendations_per_seed"], 20, "per-seed budget")
    require_equal(design["total_candidate_budget"], 60, "total budget")
    require_equal(
        design["total_candidate_budget"],
        len(design["search_seeds"]) * design["recommendations_per_seed"],
        "budget arithmetic",
    )
    selection = policy["selection_contract"]
    require_equal(
        selection["latency_mode"]["latency_accuracy_retention"],
        {
            "type": "relative",
            "retained_fraction": 0.98,
            "reference": "accuracy_winner",
        },
        "latency accuracy policy",
    )
    require_equal(
        selection["multi_objective_mode"]["multi_objective_min_accuracy"],
        None,
        "multi-objective minimum accuracy",
    )
    require_equal(
        selection["multi_objective_mode"]["weights"],
        {"accuracy_regret": 1.0, "latency_regret": 1.0},
        "multi-objective weights",
    )
    require_equal(
        selection["multi_objective_mode"]["selector"],
        "normalized_augmented_chebyshev",
        "multi-objective selector",
    )
    require_equal(
        policy["sensitivity_evidence_contract"]["source_manifest"]["sha256"],
        "c569f858f4513139292d7189ab5e57f897b8794fdbe5b2dcafc45b0efcd663aa",
        "pinned sensitivity manifest SHA256",
    )
    axis_policy = policy["architecture_axis_policy"]
    axes = axis_policy["axes"]
    expected_order = axis_policy["axis_order"]
    require_equal([axis["path"] for axis in axes], expected_order, "axis order")
    require_equal(
        [
            (
                axis["path"],
                axis["reference"],
                axis["preregistered_levels"],
                axis["search_domain"],
            )
            for axis in axes
        ],
        [
            (
                "model.num_queries",
                594,
                [300, 450, 594, 750, 900],
                {
                    "representation": "integer_range",
                    "valid_min": 300,
                    "valid_max": 900,
                },
            ),
            (
                "model.enc_layers",
                6,
                [3, 4, 5, 6],
                {
                    "representation": "ordered_integer_levels",
                    "valid_options": [3, 4, 5, 6],
                    "valid_min": 3,
                    "valid_max": 6,
                },
            ),
            (
                "model.dec_layers",
                6,
                [3, 4, 5, 6],
                {
                    "representation": "ordered_integer_levels",
                    "valid_options": [3, 4, 5, 6],
                    "valid_min": 3,
                    "valid_max": 6,
                },
            ),
            (
                "model.num_select",
                300,
                [50, 100, 200, 300],
                {
                    "representation": "ordered_integer_levels",
                    "valid_options": [50, 100, 200, 300],
                },
            ),
        ],
        "frozen architecture axis contracts",
    )
    if len(set(expected_order)) != len(expected_order):
        raise ContractError("architecture axis paths must be unique")
    for axis in axes:
        levels = axis["preregistered_levels"]
        reference = axis["reference"]
        if reference not in levels or len(levels) != len(set(levels)):
            raise ContractError(
                f"{axis['path']} has an invalid reference or duplicate levels"
            )
        expected_non_reference = {str(level) for level in levels if level != reference}
        require_equal(
            set(axis["non_reference_profile_ids"]),
            expected_non_reference,
            f"{axis['path']} non-reference level keys",
        )
        if len(set(axis["non_reference_profile_ids"].values())) != len(
            axis["non_reference_profile_ids"]
        ):
            raise ContractError(f"{axis['path']} profile IDs must be unique")
    training_parameters = policy["always_included_training_parameters"]
    require_equal(
        [parameter["path"] for parameter in training_parameters],
        ["train.optim.lr", "train.optim.weight_decay"],
        "always-included training parameters",
    )
    require_equal(
        [parameter["search_domain"] for parameter in training_parameters],
        [
            {
                "representation": "continuous",
                "valid_min": 1.0e-5,
                "valid_max": 5.0e-4,
            },
            {
                "representation": "continuous",
                "valid_min": 1.0e-5,
                "valid_max": 1.0e-3,
            },
        ],
        "always-included training parameter ranges",
    )
    validate_false_audit_flags(policy, "policy")


def expected_decision_identity(
    policy: dict[str, Any],
) -> dict[tuple[str, int], str]:
    expected: dict[tuple[str, int], str] = {}
    for axis in policy["architecture_axis_policy"]["axes"]:
        for level_text, profile_id in axis["non_reference_profile_ids"].items():
            key = (axis["path"], int(level_text))
            if key in expected:
                raise ContractError(f"duplicate policy decision identity {key}")
            expected[key] = profile_id
    return expected


def normalize_decisions(
    policy: dict[str, Any],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    evidence = policy["sensitivity_evidence_contract"]
    collection_name = evidence["decision_collection"]
    decisions = result.get(collection_name)
    if not isinstance(decisions, list):
        raise ContractError(f"{collection_name} must be an array")
    expected = expected_decision_identity(policy)
    actual: dict[tuple[str, int], dict[str, Any]] = {}
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise ContractError(f"{collection_name}[{index}] must be an object")
        axis = decision.get("axis")
        level = decision.get("level")
        if (
            not isinstance(axis, str)
            or isinstance(level, bool)
            or not isinstance(level, int)
        ):
            raise ContractError(
                f"{collection_name}[{index}] has invalid axis or level"
            )
        key = (axis, level)
        if key not in expected:
            raise ContractError(f"unknown sensitivity axis/level {key}")
        if key in actual:
            raise ContractError(f"duplicate sensitivity axis/level {key}")
        require_equal(
            decision.get("profile_id"),
            expected[key],
            f"{key} profile_id",
        )
        for field in (
            evidence["support_field"],
            evidence["qualification_field"],
            evidence["accuracy_annotation_field"],
        ):
            if not isinstance(decision.get(field), bool):
                raise ContractError(f"{key} {field} must be boolean")
        if decision[evidence["qualification_field"]] and not decision[
            evidence["support_field"]
        ]:
            raise ContractError(
                f"{key} cannot qualify without complete valid repeat support"
            )
        require_equal(
            decision[evidence["support_field"]],
            True,
            f"{key} complete support for a complete result",
        )
        require_equal(
            decision.get("feeds_final_selection"),
            False,
            f"{key} feeds_final_selection",
        )
        require_equal(
            decision.get("winner_selected"),
            False,
            f"{key} winner_selected",
        )
        decision_floor = require_finite_positive(
            decision.get("effective_noise_floor_ms"),
            f"{key} effective_noise_floor_ms",
        )
        result_floor = require_finite_positive(
            nested_value(result, evidence["noise_floor_path"]),
            evidence["noise_floor_path"],
        )
        require_equal(
            decision_floor,
            result_floor,
            f"{key} effective noise floor identity",
        )
        ci_low, ci_high = require_finite_interval(
            decision.get("hierarchical_paired_effect_ci95_ms"),
            f"{key} hierarchical_paired_effect_ci95_ms",
        )
        expected_qualified = (
            ci_high < -result_floor or ci_low > result_floor
        )
        expected_direction = (
            "faster"
            if ci_high < -result_floor
            else (
                "slower"
                if ci_low > result_floor
                else "uncertain_or_within_practical_band"
            )
        )
        require_equal(
            decision[evidence["qualification_field"]],
            expected_qualified,
            f"{key} direction-agnostic CI qualification",
        )
        require_equal(
            decision.get("effect_direction"),
            expected_direction,
            f"{key} effect direction",
        )
        if "future_shared_multi_objective_eligible" in decision:
            require_equal(
                decision["future_shared_multi_objective_eligible"],
                expected_qualified,
                f"{key} future shared multi-objective eligibility",
            )
        if "latency_reduction_qualified" in decision:
            require_equal(
                decision["latency_reduction_qualified"],
                ci_high < -result_floor,
                f"{key} latency reduction qualification",
            )
        actual[key] = copy.deepcopy(decision)
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise ContractError(f"missing sensitivity decisions: {missing}")
    if len(actual) != len(expected):
        raise ContractError("sensitivity decision cardinality mismatch")
    return [actual[key] for key in sorted(actual)]


def derive_architecture_axes(
    policy: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply only the frozen direction-agnostic latency qualification rule."""

    evidence = policy["sensitivity_evidence_contract"]
    qualification_field = evidence["qualification_field"]
    support_field = evidence["support_field"]
    decision_by_axis: dict[str, list[dict[str, Any]]] = {}
    for decision in decisions:
        decision_by_axis.setdefault(decision["axis"], []).append(decision)
    derived: list[dict[str, Any]] = []
    for axis in policy["architecture_axis_policy"]["axes"]:
        axis_decisions = decision_by_axis.get(axis["path"], [])
        qualified = sorted(
            decision["level"]
            for decision in axis_decisions
            if decision[support_field] and decision[qualification_field]
        )
        if not qualified:
            continue
        # The search domain and all tested levels come from policy, not from the
        # qualified-value hull.  Accuracy annotations are intentionally unread.
        derived.append(
            {
                "path": axis["path"],
                "reference": axis["reference"],
                "qualified_non_reference_levels": qualified,
                "qualification_basis": (
                    "at_least_one_direction_agnostic_latency_effect_qualified"
                ),
                "preregistered_levels": copy.deepcopy(
                    axis["preregistered_levels"]
                ),
                "search_domain": copy.deepcopy(axis["search_domain"]),
                **(
                    {
                        "compatibility_constraint": axis[
                            "compatibility_constraint"
                        ]
                    }
                    if "compatibility_constraint" in axis
                    else {}
                ),
            }
        )
    if not derived:
        raise ContractError(
            "no architecture axis has a qualified non-reference latency effect; "
            "expanded search is blocked"
        )
    return derived


def validate_source_identity(
    policy: dict[str, Any],
    result: dict[str, Any],
    base: Path,
) -> dict[str, Any]:
    evidence = policy["sensitivity_evidence_contract"]
    source_policy = evidence["source_manifest"]
    source_path = (base / source_policy["path"]).resolve()
    source = load_json(source_path)
    source_sha256 = sha256_file(source_path)
    require_equal(
        source_sha256,
        source_policy["sha256"],
        "pinned sensitivity manifest SHA256",
    )
    require_equal(source.get("schema_version"), 1, "sensitivity manifest schema")
    require_equal(source.get("manifest_id"), evidence["manifest_id"], "manifest ID")
    require_equal(
        source.get("status"),
        source_policy["required_status"],
        "sensitivity manifest status",
    )
    require_equal(
        source.get("feeds_final_selection"),
        False,
        "sensitivity manifest feeds_final_selection",
    )
    require_equal(
        source.get("manual_promotion_permitted"),
        False,
        "sensitivity manifest manual_promotion_permitted",
    )
    require_equal(
        result.get("manifest_sha256"),
        source_sha256,
        "result sensitivity manifest SHA256",
    )
    frozen = source["frozen_inputs"]
    require_equal(
        frozen["one_factor_manifest_sha256"],
        source_policy["one_factor_manifest_sha256"],
        "one-factor manifest pinned SHA256",
    )
    require_equal(
        frozen["checkpoint_artifact_id"],
        source_policy["checkpoint_artifact_id"],
        "checkpoint artifact pinned ID",
    )
    require_equal(
        frozen["checkpoint_artifact_sha256"],
        source_policy["checkpoint_artifact_sha256"],
        "checkpoint artifact pinned SHA256",
    )
    require_equal(
        frozen["accuracy_artifact_id"],
        source_policy["accuracy_artifact_id"],
        "accuracy artifact pinned ID",
    )
    one_path = (base / source_policy["one_factor_manifest_path"]).resolve()
    checkpoint_path = (base / source_policy["checkpoint_artifact_path"]).resolve()
    accuracy_path = (base / source_policy["accuracy_artifact_path"]).resolve()
    require_equal(
        sha256_file(one_path),
        source_policy["one_factor_manifest_sha256"],
        "one-factor manifest local SHA256",
    )
    require_equal(
        sha256_file(checkpoint_path),
        source_policy["checkpoint_artifact_sha256"],
        "checkpoint artifact local SHA256",
    )
    checkpoint = load_json(checkpoint_path)
    require_equal(
        checkpoint.get("schema_version"), 1, "checkpoint artifact schema"
    )
    require_equal(
        checkpoint.get("artifact_id"),
        source_policy["checkpoint_artifact_id"],
        "checkpoint artifact local ID",
    )
    require_equal(
        checkpoint.get("status"), "complete", "checkpoint artifact status"
    )
    require_equal(
        checkpoint.get("feeds_final_selection"),
        False,
        "checkpoint artifact feeds_final_selection",
    )
    require_equal(
        checkpoint.get("manual_selection_permitted"),
        False,
        "checkpoint artifact manual_selection_permitted",
    )
    require_equal(
        checkpoint.get("winner_selected"),
        False,
        "checkpoint artifact winner_selected",
    )
    require_equal(
        checkpoint.get("source", {}).get("manifest_sha256"),
        source_policy["one_factor_manifest_sha256"],
        "checkpoint artifact one-factor provenance",
    )
    require_equal(
        checkpoint.get("source", {}).get("frozen_plan_sha256"),
        frozen["one_factor_plan_sha256"],
        "checkpoint artifact plan provenance",
    )
    accuracy_sha256 = sha256_file(accuracy_path)
    require_sha256(accuracy_sha256, "accuracy artifact local SHA256")
    accuracy = load_json(accuracy_path)
    require_equal(accuracy.get("schema_version"), 1, "accuracy artifact schema")
    require_equal(
        accuracy.get("artifact_id"),
        source_policy["accuracy_artifact_id"],
        "accuracy artifact local ID",
    )
    require_equal(
        accuracy.get("status"), "complete", "accuracy artifact status"
    )
    require_equal(
        accuracy.get("feeds_final_selection"),
        False,
        "accuracy artifact feeds_final_selection",
    )
    require_equal(
        accuracy.get("manual_selection_permitted"),
        False,
        "accuracy artifact manual_selection_permitted",
    )
    require_equal(
        accuracy.get("winner_selected"),
        False,
        "accuracy artifact winner_selected",
    )
    require_equal(
        accuracy.get("selection", {}).get("performed"),
        False,
        "accuracy artifact selection performed",
    )
    require_equal(
        accuracy.get("source", {}).get("manifest_sha256"),
        source_policy["one_factor_manifest_sha256"],
        "accuracy artifact one-factor provenance",
    )
    require_equal(
        accuracy.get("source", {}).get("frozen_training_plan_sha256"),
        frozen["one_factor_plan_sha256"],
        "accuracy artifact training-plan provenance",
    )
    require_equal(
        accuracy.get("source", {}).get("checkpoint_artifact_sha256"),
        source_policy["checkpoint_artifact_sha256"],
        "accuracy artifact checkpoint provenance",
    )
    require_equal(
        result.get("checkpoint_artifact_id"),
        source_policy["checkpoint_artifact_id"],
        "result checkpoint artifact ID",
    )
    require_equal(
        result.get("checkpoint_artifact_sha256"),
        source_policy["checkpoint_artifact_sha256"],
        "result checkpoint artifact SHA256",
    )
    require_equal(
        result.get("accuracy_artifact_id"),
        source_policy["accuracy_artifact_id"],
        "result accuracy artifact ID",
    )
    require_equal(
        result.get("accuracy_artifact_sha256"),
        accuracy_sha256,
        "result accuracy artifact SHA256",
    )
    require_equal(
        result.get("schedule_sha256"),
        source["design"]["schedule_sha256"],
        "result schedule SHA256",
    )
    one = load_json(one_path)
    require_equal(
        result.get("one_factor_manifest_path"),
        str(one_path),
        "result one-factor manifest path",
    )
    identity = policy["frozen_identity"]
    require_equal(one["scope"], policy["scope"], "DINO scope identity")
    require_equal(
        one["runtime_contract"]["sqsh_path"],
        identity["runtime"]["sqsh_path"],
        "SQSH path identity",
    )
    require_equal(
        one["runtime_contract"]["sqsh_sha256"],
        identity["runtime"]["sqsh_sha256"],
        "SQSH SHA256 identity",
    )
    require_equal(
        one["runtime_contract"]["pretrained_model_path"],
        identity["runtime"]["pretrained_model_path"],
        "PTM path identity",
    )
    require_equal(
        one["runtime_contract"]["pretrained_model_sha256"],
        identity["runtime"]["pretrained_model_sha256"],
        "PTM SHA256 identity",
    )
    require_equal(
        one["dataset_contract"]["source_uri"],
        identity["dataset"]["source_uri"],
        "dataset URI identity",
    )
    require_equal(
        one["dataset_contract"]["train_annotation_sha256"],
        identity["dataset"]["train_annotation_sha256"],
        "training annotation identity",
    )
    require_equal(
        one["dataset_contract"]["validation_annotation_sha256"],
        identity["dataset"]["validation_annotation_sha256"],
        "validation annotation identity",
    )
    for key in (
        "staged_root",
        "train_image_dir",
        "train_annotation",
        "validation_image_dir",
        "validation_annotation",
        "num_classes",
        "eval_class_ids",
    ):
        require_equal(
            one["dataset_contract"][key],
            identity["dataset"][key],
            f"dataset {key} identity",
        )
    one_slurm = one["runtime_contract"]["slurm"]
    for one_key, identity_key in (
        ("partition", "partition"),
        ("account", "account"),
        ("num_nodes", "num_nodes"),
        ("gpu_count_per_node", "gpu_count_per_node"),
        ("required_gpu_model", "required_gpu_model"),
        ("required_gpu_memory_bytes", "required_gpu_memory_bytes"),
        ("required_compute_capability", "required_compute_capability"),
    ):
        require_equal(
            one_slurm[one_key],
            identity["runtime"][identity_key],
            f"runtime {identity_key} identity",
        )
    require_equal(
        one["runtime_contract"]["software"]["precision"],
        identity["runtime"]["precision"],
        "runtime precision identity",
    )
    require_equal(
        one["runtime_contract"]["software"]["distributed_strategy"],
        identity["runtime"]["distributed_strategy"],
        "runtime distributed strategy identity",
    )
    require_equal(
        one["runtime_contract"]["software"]["tf32"],
        identity["runtime"]["tf32"],
        "runtime TF32 identity",
    )
    for key in ("torch", "cuda", "cudnn"):
        require_equal(
            one["runtime_contract"]["software"][key],
            identity["runtime"][key],
            f"runtime {key} identity",
        )
    for one_key, identity_key in (
        ("train_epochs", "train_epochs"),
        ("checkpoint_interval_epochs", "checkpoint_interval_epochs"),
        ("validation_interval_epochs", "validation_interval_epochs"),
        ("train_batch_size_per_gpu", "batch_size_per_gpu"),
        ("train_num_gpus", "num_gpus"),
        ("train_num_nodes", "num_nodes"),
        ("activation_checkpoint", "activation_checkpoint"),
        ("cudnn_benchmark", "cudnn_benchmark"),
        ("cudnn_deterministic", "cudnn_deterministic"),
        ("wandb_enabled", "wandb_enabled"),
        ("optimizer", "optimizer"),
        ("lr_backbone", "lr_backbone"),
        ("lr_linear_proj_mult", "lr_linear_proj_mult"),
        ("lr_scheduler", "lr_scheduler"),
        ("augmentation_source", "augmentation_source"),
    ):
        require_equal(
            one["controlled_constants"][one_key],
            identity["training_controls"][identity_key],
            f"training control {identity_key}",
        )
    require_equal(
        one["controlled_constants"]["train_num_gpus"]
        * one["controlled_constants"]["train_batch_size_per_gpu"],
        identity["training_controls"]["global_batch_size"],
        "training global batch size",
    )
    expected_axis_contract = {
        axis["path"]: (axis["reference"], axis["preregistered_levels"])
        for axis in policy["architecture_axis_policy"]["axes"]
    }
    actual_axis_contract = {
        axis["path"]: (axis["reference"], axis["levels"])
        for axis in one["design"]["axes"]
    }
    require_equal(
        actual_axis_contract,
        expected_axis_contract,
        "one-factor architecture axes",
    )
    validate_false_audit_flags(source, "sensitivity_manifest")
    validate_false_audit_flags(checkpoint, "checkpoint_artifact")
    validate_false_audit_flags(accuracy, "accuracy_artifact")
    return {
        "sensitivity_manifest_path": str(source_path),
        "sensitivity_manifest_sha256": source_sha256,
        "one_factor_manifest_path": str(one_path),
        "one_factor_manifest_sha256": source_policy[
            "one_factor_manifest_sha256"
        ],
        "checkpoint_artifact_path": str(checkpoint_path),
        "checkpoint_artifact_id": source_policy["checkpoint_artifact_id"],
        "checkpoint_artifact_sha256": source_policy[
            "checkpoint_artifact_sha256"
        ],
        "accuracy_artifact_path": str(accuracy_path),
        "accuracy_artifact_id": source_policy["accuracy_artifact_id"],
        "accuracy_artifact_sha256": accuracy_sha256,
        "schedule_sha256": source["design"]["schedule_sha256"],
        "reference_model_spec": copy.deepcopy(one["reference"]["model"]),
        "reference_optimizer": copy.deepcopy(one["reference"]["optimizer"]),
    }


def validate_sensitivity_result(
    policy: dict[str, Any],
    result: dict[str, Any],
    *,
    result_path: Path,
    supplied_sha256: str,
    source_base: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    supplied_sha256 = require_sha256(
        supplied_sha256, "supplied sensitivity result SHA256"
    )
    require_equal(
        sha256_file(result_path),
        supplied_sha256,
        "sensitivity result whole-file SHA256",
    )
    evidence = policy["sensitivity_evidence_contract"]
    require_equal(
        result.get("schema_version"),
        evidence["result_schema_version"],
        "sensitivity result schema_version",
    )
    require_equal(
        result.get("manifest_id"),
        evidence["manifest_id"],
        "sensitivity result manifest_id",
    )
    require_equal(
        result.get("status"),
        evidence["required_status"],
        "sensitivity result status",
    )
    require_equal(
        result.get("blockers"),
        evidence["required_blockers"],
        "sensitivity result blockers",
    )
    for key, expected in evidence["required_flags"].items():
        require_equal(result.get(key), expected, f"sensitivity result {key}")
    validate_false_audit_flags(result)
    internal_digest = require_sha256(
        result.get("report_sha256"), "sensitivity result report_sha256"
    )
    digest_payload = copy.deepcopy(result)
    del digest_payload["report_sha256"]
    require_equal(
        sha256_value(digest_payload),
        internal_digest,
        "sensitivity result internal report_sha256",
    )
    source = validate_source_identity(
        policy,
        result,
        (source_base if source_base is not None else result_path.parent).resolve(),
    )
    decisions = normalize_decisions(policy, result)
    tolerance = require_finite_positive(
        nested_value(result, evidence["noise_floor_path"]),
        evidence["noise_floor_path"],
    )
    return decisions, source, tolerance


def build_manifest(
    policy: dict[str, Any],
    decisions: list[dict[str, Any]],
    *,
    sensitivity_result_path: Path,
    sensitivity_result_sha256: str,
    source_identity: dict[str, Any],
    latency_tolerance: float,
    policy_path: Path,
    policy_sha256: str,
    generator_path: Path,
    generator_sha256: str,
    sensitivity_report_sha256: str,
) -> dict[str, Any]:
    derived_axes = derive_architecture_axes(policy, decisions)
    training_parameters = copy.deepcopy(
        policy["always_included_training_parameters"]
    )
    search_parameters = [axis["path"] for axis in derived_axes] + [
        parameter["path"] for parameter in training_parameters
    ]
    search_domains = {
        axis["path"]: copy.deepcopy(axis["search_domain"])
        for axis in derived_axes
    }
    search_domains.update(
        {
            parameter["path"]: copy.deepcopy(parameter["search_domain"])
            for parameter in training_parameters
        }
    )
    evidence = policy["sensitivity_evidence_contract"]
    normalized_audit = [
        {
            "profile_id": decision["profile_id"],
            "axis": decision["axis"],
            "level": decision["level"],
            "support_validity_repeatability_gate": decision[
                evidence["support_field"]
            ],
            "latency_effect_qualified": decision[
                evidence["qualification_field"]
            ],
            "latency_mode_98pct_suitable": decision[
                evidence["accuracy_annotation_field"]
            ],
            "effect_direction": decision["effect_direction"],
            "hierarchical_paired_effect_ci95_ms": copy.deepcopy(
                decision["hierarchical_paired_effect_ci95_ms"]
            ),
            "effective_noise_floor_ms": decision[
                "effective_noise_floor_ms"
            ],
        }
        for decision in sorted(
            decisions,
            key=lambda item: (
                policy["architecture_axis_policy"]["axis_order"].index(
                    item["axis"]
                ),
                item["level"],
                item["profile_id"],
            ),
        )
    ]
    selection = copy.deepcopy(policy["selection_contract"])
    selection["latency_tolerance"]["value_ms"] = latency_tolerance
    manifest = {
        "schema_version": 1,
        "manifest_id": "dino_expanded_search_20260728_v1",
        "status": policy["output_contract"]["status"],
        "scope": copy.deepcopy(policy["scope"]),
        "feeds_final_selection": True,
        "manual_override_permitted": False,
        "algorithm_only_selection_required": True,
        "derivation": {
            "policy_id": policy["policy_id"],
            "policy_path": str(policy_path.resolve()),
            "policy_sha256": policy_sha256,
            "generator_path": str(generator_path.resolve()),
            "generator_sha256": generator_sha256,
            "sensitivity_result_path": str(sensitivity_result_path.resolve()),
            "sensitivity_result_sha256": sensitivity_result_sha256,
            "sensitivity_report_sha256": sensitivity_report_sha256,
            "source_identity": copy.deepcopy(source_identity),
            "rule": (
                "An axis is included iff at least one non-reference level has "
                "complete support and direction-agnostic latency-effect "
                "qualification; the full frozen domain is then used."
            ),
            "accuracy_retention_used_for_axis_derivation": False,
            "qualified_value_hull_used": False,
            "manual_override_used": False,
            "decision_audit": normalized_audit,
        },
        "search_space": {
            "architecture_axes": derived_axes,
            "always_included_training_parameters": training_parameters,
            "search_parameters": search_parameters,
            "search_domains": search_domains,
            "compatibility_constraints": [
                "model.num_queries >= model.num_select"
            ],
            "reference_model_spec": copy.deepcopy(
                source_identity["reference_model_spec"]
            ),
            "reference_optimizer": copy.deepcopy(
                source_identity["reference_optimizer"]
            ),
        },
        "search_design": copy.deepcopy(policy["search_design"]),
        "selection": selection,
        "frozen_identity": copy.deepcopy(policy["frozen_identity"]),
    }
    manifest["manifest_sha256"] = sha256_value(manifest)
    return manifest


def atomic_write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise ContractError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    if pending.exists():
        raise ContractError(f"pending output already exists: {pending}")
    try:
        pending.write_text(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        pending.replace(path)
    finally:
        if pending.exists():
            pending.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--sensitivity-result", type=Path, required=True)
    parser.add_argument("--sensitivity-result-sha256", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy_path = args.policy.resolve()
    result_path = args.sensitivity_result.resolve()
    output_path = args.output.resolve()
    policy = load_json(policy_path)
    validate_policy(policy)
    expected_filename = policy["output_contract"]["filename"]
    if output_path.name != expected_filename:
        raise ContractError(
            f"output basename must be exactly {expected_filename!r}"
        )
    result = load_json(result_path)
    decisions, source_identity, latency_tolerance = (
        validate_sensitivity_result(
            policy,
            result,
            result_path=result_path,
            supplied_sha256=args.sensitivity_result_sha256,
            source_base=policy_path.parent,
        )
    )
    generator_path = Path(__file__).resolve()
    manifest = build_manifest(
        policy,
        decisions,
        sensitivity_result_path=result_path,
        sensitivity_result_sha256=args.sensitivity_result_sha256,
        source_identity=source_identity,
        latency_tolerance=latency_tolerance,
        policy_path=policy_path,
        policy_sha256=sha256_file(policy_path),
        generator_path=generator_path,
        generator_sha256=sha256_file(generator_path),
        sensitivity_report_sha256=result["report_sha256"],
    )
    atomic_write_new(output_path, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest_id": manifest["manifest_id"],
                "manifest_sha256": manifest["manifest_sha256"],
                "architecture_axes": [
                    axis["path"]
                    for axis in manifest["search_space"]["architecture_axes"]
                ],
                "latency_tolerance_ms": latency_tolerance,
                "output": str(output_path),
                "launched": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        raise SystemExit(2)
