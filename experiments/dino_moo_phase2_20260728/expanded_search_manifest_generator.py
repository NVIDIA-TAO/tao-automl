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
import re
import subprocess
import sys
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_POLICY = HERE / "expanded_search_derivation_policy.v1.json"
DEFAULT_OUTPUT = HERE / "expanded_search_manifest.v2.json"
HEX = frozenset("0123456789abcdef")
EXPECTED_ANALYSIS_ERRATUM_CONTRACT_SHA256 = (
    "609bc9863a7e3289fe5f374b935f9da8422860eb00c62ea3d4bab00846d2fd7f"
)
EXPECTED_POST_FRONT_CONTRACT_SHA256 = (
    "aba3a961bf50caf15803f271b59d7ffbd091414816d14f3deb793452f75ec281"
)
EXPECTED_SENSITIVITY_RESULT_SHA256 = (
    "33aea1c13ece0ce632587abd16ed6020ecc88c63220f89891a5f30183322eaea"
)
EXPECTED_SENSITIVITY_REPORT_SHA256 = (
    "40a8bccb6e43b8238c2cf6b47eaf3253e735d82fd160212d12915b3137a3fa79"
)


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


def validate_analysis_erratum_contract(policy: dict[str, Any]) -> None:
    evidence = policy["sensitivity_evidence_contract"]
    require_equal(
        {
            key: evidence.get(key)
            for key in (
                "result_path",
                "result_sha256",
                "result_report_sha256",
            )
        },
        {
            "result_path": "sensitivity_latency_analysis.v2.json",
            "result_sha256": EXPECTED_SENSITIVITY_RESULT_SHA256,
            "result_report_sha256": EXPECTED_SENSITIVITY_REPORT_SHA256,
        },
        "approved sensitivity analysis result",
    )
    contract = evidence["analysis_erratum"]
    require_equal(
        sha256_value(contract),
        EXPECTED_ANALYSIS_ERRATUM_CONTRACT_SHA256,
        "analysis erratum preregistration contract",
    )
    require_equal(
        {
            key: contract.get(key)
            for key in (
                "path",
                "sha256",
                "schema_version",
                "erratum_id",
                "status",
                "scope",
                "reason_code",
            )
        },
        {
            "path": "sensitivity_latency_analysis_erratum.v1.json",
            "sha256": (
                "8e19287bf2ffd674f62b21cdaf11e000"
                "b0eae1ed8af9d0ada1238491588993f2"
            ),
            "schema_version": 1,
            "erratum_id": (
                "dino_sensitivity_latency_analysis_erratum_20260728_v1"
            ),
            "status": "approved_analysis_only",
            "scope": (
                "aggregation_validation_evidence_access_and_"
                "descendant_commit_only"
            ),
            "reason_code": (
                "allocation_torch_version_used_full_string_"
                "instead_of_declared_base_release"
            ),
        },
        "analysis erratum identity",
    )
    require_equal(
        contract["required_flags"],
        {
            "measurement_generation_unchanged": True,
            "qualification_policy_unchanged": True,
            "objective_values_altered": False,
            "winner_selected": False,
            "feeds_final_selection": False,
            "manual_promotion_permitted": False,
        },
        "analysis erratum flags",
    )
    require_equal(
        contract["measurement_contract"],
        {
            "manifest_id": "dino_sensitivity_latency_20260728_v2",
            "manifest_path": "sensitivity_latency_manifest.v2.json",
            "manifest_sha256": (
                "aedc117414b2691c1a70b73fa4e9e0ac"
                "123cb4d20dfd9d25dfe2d4aa490d7655"
            ),
            "submission_ledger_path": (
                "runtime/sensitivity_latency_v2/block_submissions.json"
            ),
            "submission_ledger_sha256": (
                "b1c170c0d4697463d171cbeca3e4adcbd"
                "34cc1cb7429c236f48b58c46c3b6d54"
            ),
            "launch_automl_branch": (
                "rarunachalam/pre-platform-sdk-removal-20260714"
            ),
            "launch_automl_commit": (
                "cb62ef447704b95980b17aa82604992564b4e71f"
            ),
        },
        "analysis erratum measurement contract",
    )
    require_equal(
        contract["source_files"],
        {
            "original_aggregator_path": "sensitivity_latency_aggregate.py",
            "original_aggregator_sha256": (
                "5f5aebd4274c746ec9674f28f978af5d"
                "228d98c6ba0af8d76cff8b1742dab967"
            ),
            "corrected_aggregator_path": (
                "sensitivity_latency_aggregate_erratum.py"
            ),
            "corrected_aggregator_sha256": (
                "9209e748093e0555fe5cba339327a821"
                "6744ec9ca6b9dae276c7041703a409c6"
            ),
            "validation_test_path": (
                "test_sensitivity_latency_analysis_erratum.py"
            ),
            "validation_test_sha256": (
                "e9bfe695b0142e8a944d412fa78a2235"
                "e8bd960c5c5ad0320ddd006e20460f59"
            ),
        },
        "analysis erratum source files",
    )


def validate_post_front_contract(policy: dict[str, Any]) -> None:
    contract = policy["post_front_matched_validation"]
    require_equal(
        sha256_value(contract),
        EXPECTED_POST_FRONT_CONTRACT_SHA256,
        "post-front matched-validation contract",
    )
    require_equal(
        contract["allocation_design"]["allocation_count"],
        6,
        "post-front allocation count",
    )
    require_equal(
        contract["allocation_design"]["gpus_per_node"],
        8,
        "post-front GPUs",
    )
    require_equal(
        contract["ordering"]["algorithm"],
        "balanced_williams_rows_v1",
        "post-front ordering",
    )
    require_equal(
        {
            key: contract["latency_protocol"][key]
            for key in (
                "warmup_iterations",
                "timed_iterations_per_round",
                "repeated_rounds",
            )
        },
        {
            "warmup_iterations": 50,
            "timed_iterations_per_round": 100,
            "repeated_rounds": 5,
        },
        "post-front latency protocol",
    )
    require_equal(
        contract["paired_analysis"]["bootstrap_resamples"],
        10000,
        "post-front paired bootstrap resamples",
    )
    require_equal(
        contract["paired_analysis"]["bootstrap_confidence_level"],
        0.95,
        "post-front paired bootstrap confidence",
    )
    require_equal(
        contract["selection_isolation"],
        {
            "measurements_feed_reselection": False,
            "winner_reselection_permitted": False,
            "original_selection_time_measurements_replaced": False,
            "algorithm_selected_candidate_overridden": False,
            "allowed_use": "stability analysis and hypothesis verdict only",
        },
        "post-front selection isolation",
    )


def validate_runtime_supersession(policy: dict[str, Any]) -> None:
    contract = policy["runtime_supersession"]
    require_equal(
        contract["only_permitted_changes"],
        [
            "manifest identity and supersession metadata",
            (
                "derivation policy, manifest generator, and corrected runner "
                "provenance SHA256 values"
            ),
            "strict finite numeric metric parser contract",
            "new runtime path",
            "manifest-bound fresh and resume runtime-state provenance guard",
        ],
        "expanded v2 only-permitted changes",
    )
    require_equal(
        {
            key: contract.get(key)
            for key in (
                "erratum_path",
                "erratum_sha256",
                "erratum_internal_sha256",
                "erratum_id",
                "erratum_status",
                "source_runtime_path",
                "source_runtime_status",
                "source_runtime_resume_permitted",
                "source_runtime_reuse_permitted",
                "valid_objective_observation_count",
                "selection_performed",
            )
        },
        {
            "erratum_path": "expanded_search_runtime_erratum.v1.json",
            "erratum_sha256": (
                "a89b5816b45e1df9c6286c25ccbe8314"
                "daee53843decc4400882ec33f10ffa17"
            ),
            "erratum_internal_sha256": (
                "b61196e5b76153fa71f4e73c58e4bc6f"
                "58eeba1f8fed3783482ba8c3156e5954"
            ),
            "erratum_id": (
                "dino_expanded_search_runtime_erratum_20260728_v1"
            ),
            "erratum_status": (
                "approved_preselection_runtime_failure_audit"
            ),
            "source_runtime_path": "runtime/expanded_search",
            "source_runtime_status": (
                "failed_preselection_superseded_preserved_read_only"
            ),
            "source_runtime_resume_permitted": False,
            "source_runtime_reuse_permitted": False,
            "valid_objective_observation_count": 0,
            "selection_performed": False,
        },
        "expanded runtime supersession identity",
    )
    require_equal(
        contract["source_manifest"],
        {
            "path": "expanded_search_manifest.v1.json",
            "manifest_id": "dino_expanded_search_20260728_v1",
            "whole_file_sha256": (
                "57e331686b8896989263a39f72edb6954"
                "3fc58833f20a1e6e698c31f34d2e8be"
            ),
            "internal_manifest_sha256": (
                "39fb50997a39ff4bfaa8036cd6222127f"
                "3ce8dda25a704ac188eab4dd6b75b82"
            ),
        },
        "expanded v1 source manifest",
    )
    require_equal(
        contract["target_manifest"],
        {
            "filename": "expanded_search_manifest.v2.json",
            "manifest_id": "dino_expanded_search_20260728_v2",
            "runtime_path": "runtime/expanded_search_v2",
            "overwrite_permitted": False,
        },
        "expanded v2 target manifest",
    )
    for key in (
        "search_space_changed",
        "search_algorithm_changed",
        "search_seeds_changed",
        "training_seed_changed",
        "candidate_budget_changed",
        "training_budget_changed",
        "selection_configuration_changed",
        "latency_protocol_changed",
        "dataset_changed",
        "runtime_hardware_changed",
    ):
        require_equal(contract.get(key), False, f"runtime erratum {key}")
    require_equal(
        contract["strict_metric_parser"],
        {
            "function": "parse_finite_numeric_metric",
            "accept_native_int_or_float": True,
            "accept_strict_json_number_string": True,
            "reject_bool": True,
            "reject_empty_or_whitespace": True,
            "reject_nan_or_infinity": True,
            "reject_junk": True,
            "finite_result_required": True,
        },
        "strict metric parser contract",
    )
    erratum_path = (HERE / contract["erratum_path"]).resolve()
    require_equal(
        sha256_file(erratum_path),
        contract["erratum_sha256"],
        "expanded runtime erratum whole-file SHA256",
    )
    erratum = load_json(erratum_path)
    claimed = erratum.get("audit_sha256")
    require_equal(
        claimed,
        contract["erratum_internal_sha256"],
        "expanded runtime erratum internal SHA256",
    )
    unhashed = copy.deepcopy(erratum)
    del unhashed["audit_sha256"]
    require_equal(
        sha256_value(unhashed),
        claimed,
        "expanded runtime erratum canonical digest",
    )
    require_equal(
        erratum.get("erratum_id"),
        contract["erratum_id"],
        "expanded runtime erratum ID",
    )
    require_equal(
        erratum.get("status"),
        contract["erratum_status"],
        "expanded runtime erratum status",
    )
    require_equal(
        erratum["bayesian_state_audit"]["valid_objective_observation_count"],
        0,
        "v1 valid objective observations",
    )
    require_equal(
        erratum["bayesian_state_audit"]["all_seed_brain_y_arrays_empty"],
        True,
        "v1 empty Bayesian observation arrays",
    )
    require_equal(
        erratum["bayesian_state_audit"]["selection_invoked"],
        False,
        "v1 selection invocation",
    )
    source_manifest = (HERE / contract["source_manifest"]["path"]).resolve()
    require_equal(
        sha256_file(source_manifest),
        contract["source_manifest"]["whole_file_sha256"],
        "expanded v1 whole-file SHA256",
    )
    source = load_json(source_manifest)
    require_equal(
        source.get("manifest_id"),
        contract["source_manifest"]["manifest_id"],
        "expanded v1 manifest ID",
    )
    require_equal(
        source.get("manifest_sha256"),
        contract["source_manifest"]["internal_manifest_sha256"],
        "expanded v1 internal SHA256",
    )


def validate_policy(policy: dict[str, Any]) -> None:
    require_equal(policy.get("schema_version"), 1, "policy schema_version")
    require_equal(
        policy.get("policy_id"),
        "dino_expanded_search_derivation_20260728_v1",
        "policy_id",
    )
    require_equal(policy.get("policy_revision"), 3, "policy revision")
    require_equal(
        policy.get("status"),
        "amended_after_v1_preselection_runtime_failure_before_v2_launch",
        "policy status",
    )
    require_equal(
        policy.get("amendment"),
        {
            "reason": (
                "Preserve the failed v1 pre-selection runtime and generate "
                "a clean v2 manifest whose only operational change is "
                "strict parsing of finite numeric metrics emitted as "
                "JSON-number strings."
            ),
            "original_derivation_rule_changed": False,
            "search_domains_changed": False,
            "selection_rule_changed": False,
            "expanded_manifest_existed_at_amendment": True,
            "expanded_job_launched_at_amendment": True,
            "v1_valid_objective_observation_count": 0,
            "v1_selection_performed": False,
            "v1_runtime_reuse_permitted": False,
        },
        "policy amendment audit",
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
    evidence = policy["sensitivity_evidence_contract"]
    require_equal(
        evidence["manifest_id"],
        "dino_sensitivity_latency_20260728_v2",
        "pinned sensitivity manifest ID",
    )
    source_manifest = evidence["source_manifest"]
    require_equal(
        source_manifest["path"],
        "sensitivity_latency_manifest.v2.json",
        "pinned sensitivity manifest path",
    )
    require_equal(
        source_manifest["sha256"],
        "aedc117414b2691c1a70b73fa4e9e0ac123cb4d20dfd9d25dfe2d4aa490d7655",
        "pinned sensitivity manifest SHA256",
    )
    require_equal(
        source_manifest["supersedes"],
        {
            "manifest_id": "dino_sensitivity_latency_20260728_v1",
            "manifest_sha256": (
                "c569f858f4513139292d7189ab5e57f8"
                "97b8794fdbe5b2dcafc45b0efcd663aa"
            ),
            "disposition": "preflight_failed_no_latency_measurements",
        },
        "pinned sensitivity manifest supersession",
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
    validate_analysis_erratum_contract(policy)
    validate_post_front_contract(policy)
    validate_runtime_supersession(policy)
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


def validate_analysis_erratum_source(
    policy: dict[str, Any],
    result: dict[str, Any],
    base: Path,
    sensitivity_manifest: dict[str, Any],
) -> dict[str, Any]:
    contract = policy["sensitivity_evidence_contract"]["analysis_erratum"]
    erratum_path = (base / contract["path"]).resolve()
    erratum = load_json(erratum_path)
    require_equal(
        sha256_file(erratum_path),
        contract["sha256"],
        "approved analysis erratum whole-file SHA256",
    )
    for key in (
        "schema_version",
        "erratum_id",
        "status",
        "scope",
        "reason_code",
    ):
        require_equal(
            erratum.get(key),
            contract[key],
            f"approved analysis erratum {key}",
        )
    for key, expected in contract["required_flags"].items():
        require_equal(
            erratum.get(key),
            expected,
            f"approved analysis erratum {key}",
        )
    require_equal(
        erratum.get("measurement_contract"),
        contract["measurement_contract"],
        "approved analysis erratum measurement contract",
    )
    require_equal(
        erratum.get("unchanged_policy_pins"),
        contract["unchanged_policy_pins"],
        "approved analysis erratum policy pins",
    )
    require_equal(
        erratum.get("correction"),
        contract["correction"],
        "approved analysis erratum correction",
    )
    fingerprints = contract["contract_fingerprints"]
    fingerprint_sources = {
        "measurement_contract_sha256": erratum["measurement_contract"],
        "source_pins_sha256": erratum["source_pins"],
        "unchanged_policy_pins_sha256": erratum["unchanged_policy_pins"],
        "correction_sha256": erratum["correction"],
        "evidence_acquisition_policy_sha256": erratum[
            "evidence_acquisition_policy"
        ],
        "sdk_state_inspection_policy_sha256": erratum[
            "sdk_state_inspection_policy"
        ],
        "analysis_commit_correction_sha256": erratum[
            "analysis_commit_correction"
        ],
        "analysis_guards_sha256": erratum["analysis_guards"],
    }
    for key, value in fingerprint_sources.items():
        require_equal(
            sha256_value(value),
            fingerprints[key],
            f"approved analysis erratum {key}",
        )

    measurement_sources = sensitivity_manifest["runtime_contract"][
        "source_code_sha256"
    ]
    require_equal(
        erratum["source_pins"]["measurement_generation"],
        {
            "launcher": measurement_sources["launcher"],
            "block_runner": measurement_sources["block_runner"],
            "common": measurement_sources["common"],
            "original_aggregator": measurement_sources["aggregator"],
            "latency_stats": measurement_sources["latency_stats"],
        },
        "approved erratum measurement-generation sources",
    )
    source_files = contract["source_files"]
    for path_key, digest_key in (
        ("original_aggregator_path", "original_aggregator_sha256"),
        ("corrected_aggregator_path", "corrected_aggregator_sha256"),
        ("validation_test_path", "validation_test_sha256"),
    ):
        source_path = (base / source_files[path_key]).resolve()
        require_equal(
            sha256_file(source_path),
            source_files[digest_key],
            f"approved analysis source {path_key}",
        )
    require_equal(
        erratum["source_pins"]["original_aggregator_sha256"],
        source_files["original_aggregator_sha256"],
        "erratum original aggregator pin",
    )
    require_equal(
        erratum["source_pins"]["corrected_aggregator_sha256"],
        source_files["corrected_aggregator_sha256"],
        "erratum corrected aggregator pin",
    )
    measurement = contract["measurement_contract"]
    measurement_path = (base / measurement["manifest_path"]).resolve()
    ledger_path = (base / measurement["submission_ledger_path"]).resolve()
    require_equal(
        sha256_file(measurement_path),
        measurement["manifest_sha256"],
        "erratum measurement manifest source",
    )
    require_equal(
        sha256_file(ledger_path),
        measurement["submission_ledger_sha256"],
        "erratum submission ledger source",
    )

    identity = result.get(contract["result_binding"]["result_field"])
    if not isinstance(identity, dict):
        raise ContractError(
            "sensitivity result lacks approved analysis_erratum identity"
        )
    expected_identity = {
        "erratum_id": contract["erratum_id"],
        "erratum_path": str(erratum_path),
        "erratum_sha256": contract["sha256"],
        "reason_code": contract["reason_code"],
        "measurement_manifest_id": measurement["manifest_id"],
        "measurement_manifest_path": str(measurement_path),
        "measurement_manifest_sha256": measurement["manifest_sha256"],
        "submission_ledger_path": str(ledger_path),
        "submission_ledger_sha256": measurement[
            "submission_ledger_sha256"
        ],
        "original_aggregator_sha256": source_files[
            "original_aggregator_sha256"
        ],
        "corrected_aggregator_sha256": source_files[
            "corrected_aggregator_sha256"
        ],
        "measurement_policy_sha256": contract["unchanged_policy_pins"][
            "measurement_policy_sha256"
        ],
        "qualification_policy_sha256": contract["unchanged_policy_pins"][
            "qualification_policy_sha256"
        ],
        "evidence_acquisition_policy_sha256": fingerprints[
            "evidence_acquisition_policy_sha256"
        ],
        "sdk_state_inspection_policy_sha256": fingerprints[
            "sdk_state_inspection_policy_sha256"
        ],
        "measurement_generation_unchanged": True,
        "qualification_policy_unchanged": True,
        "objective_values_altered": False,
        "raw_runtime_string_preserved": True,
        "correction": contract["correction"],
        "analysis_commit_correction": erratum[
            "analysis_commit_correction"
        ],
    }
    require_equal(
        identity,
        expected_identity,
        "sensitivity result approved analysis_erratum identity",
    )
    require_equal(
        result.get("analysis_source_checks"),
        {
            "original_aggregator": source_files[
                "original_aggregator_sha256"
            ],
            "corrected_aggregator": source_files[
                "corrected_aggregator_sha256"
            ],
            "analysis_erratum": contract["sha256"],
        },
        "sensitivity result analysis sources",
    )
    return {
        "path": str(erratum_path),
        "sha256": contract["sha256"],
        "erratum_id": contract["erratum_id"],
        "corrected_aggregator_sha256": source_files[
            "corrected_aggregator_sha256"
        ],
        "submission_ledger_path": str(ledger_path),
        "submission_ledger_sha256": measurement[
            "submission_ledger_sha256"
        ],
        "contract_sha256": EXPECTED_ANALYSIS_ERRATUM_CONTRACT_SHA256,
    }


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
        {
            key: source.get("supersedes", {}).get(key)
            for key in ("manifest_id", "manifest_sha256", "disposition")
        },
        source_policy["supersedes"],
        "sensitivity manifest supersession",
    )
    erratum_identity = validate_analysis_erratum_source(
        policy,
        result,
        base,
        source,
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
        "analysis_erratum": erratum_identity,
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
    evidence = policy["sensitivity_evidence_contract"]
    base = (
        source_base if source_base is not None else result_path.parent
    ).resolve()
    require_equal(
        result_path.resolve(),
        (base / evidence["result_path"]).resolve(),
        "approved sensitivity result path",
    )
    require_equal(
        supplied_sha256,
        evidence["result_sha256"],
        "approved sensitivity result whole-file SHA256",
    )
    require_equal(
        sha256_file(result_path),
        supplied_sha256,
        "sensitivity result whole-file SHA256",
    )
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
    require_equal(
        internal_digest,
        evidence["result_report_sha256"],
        "approved sensitivity result report_sha256",
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
        base,
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
    runner_path: Path,
    runner_sha256: str,
    corrected_runner_commit_identity: dict[str, str],
    sensitivity_report_sha256: str,
) -> dict[str, Any]:
    require_sha256(runner_sha256, "expanded runner source SHA256")
    if runner_path.name != "expanded_search_runner.py":
        raise ContractError("expanded runner source has an unexpected basename")
    validate_corrected_runner_commit_identity(
        corrected_runner_commit_identity,
        runner_path=runner_path,
        runner_sha256=runner_sha256,
    )
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
    supersession = policy["runtime_supersession"]
    target = supersession["target_manifest"]
    manifest = {
        "schema_version": 1,
        "manifest_id": target["manifest_id"],
        "status": policy["output_contract"]["status"],
        "scope": copy.deepcopy(policy["scope"]),
        "feeds_final_selection": True,
        "manual_override_permitted": False,
        "algorithm_only_selection_required": True,
        "supersedes": {
            "manifest_id": supersession["source_manifest"]["manifest_id"],
            "manifest_path": str(
                (HERE / supersession["source_manifest"]["path"]).resolve()
            ),
            "manifest_whole_file_sha256": supersession["source_manifest"][
                "whole_file_sha256"
            ],
            "manifest_internal_sha256": supersession["source_manifest"][
                "internal_manifest_sha256"
            ],
            "runtime_path": str(
                (HERE / supersession["source_runtime_path"]).resolve()
            ),
            "disposition": supersession["source_runtime_status"],
            "resume_permitted": False,
            "runtime_reuse_permitted": False,
        },
        "runtime_supersession": {
            "erratum_path": str(
                (HERE / supersession["erratum_path"]).resolve()
            ),
            "erratum_sha256": supersession["erratum_sha256"],
            "erratum_internal_sha256": supersession[
                "erratum_internal_sha256"
            ],
            "erratum_id": supersession["erratum_id"],
            "target_runtime_path": str(
                (HERE / target["runtime_path"]).resolve()
            ),
            "strict_metric_parser": copy.deepcopy(
                supersession["strict_metric_parser"]
            ),
            "valid_objective_observations_reused": 0,
            "v1_runtime_reused": False,
            "corrected_runner_commit_identity": copy.deepcopy(
                corrected_runner_commit_identity
            ),
        },
        "derivation": {
            "policy_id": policy["policy_id"],
            "policy_path": str(policy_path.resolve()),
            "policy_sha256": policy_sha256,
            "generator_path": str(generator_path.resolve()),
            "generator_sha256": generator_sha256,
            "runner_path": str(runner_path.resolve()),
            "runner_sha256": runner_sha256,
            "sensitivity_result_path": str(sensitivity_result_path.resolve()),
            "sensitivity_result_sha256": sensitivity_result_sha256,
            "sensitivity_report_sha256": sensitivity_report_sha256,
            "source_identity": copy.deepcopy(source_identity),
            "analysis_erratum_contract_sha256": (
                EXPECTED_ANALYSIS_ERRATUM_CONTRACT_SHA256
            ),
            "post_front_contract_sha256": (
                EXPECTED_POST_FRONT_CONTRACT_SHA256
            ),
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
        "post_front_matched_validation": copy.deepcopy(
            policy["post_front_matched_validation"]
        ),
        "frozen_identity": copy.deepcopy(policy["frozen_identity"]),
    }
    manifest["manifest_sha256"] = sha256_value(manifest)
    return manifest


def validate_corrected_runner_commit_identity(
    identity: Any,
    *,
    runner_path: Path,
    runner_sha256: str,
) -> None:
    """Validate the immutable git identity of the corrected runner source."""

    if not isinstance(identity, dict):
        raise ContractError("corrected runner commit identity must be an object")
    require_equal(
        set(identity),
        {
            "repository",
            "relative_path",
            "head_commit",
            "git_blob",
            "sha256",
        },
        "corrected runner commit identity keys",
    )
    repository = identity.get("repository")
    relative_path = identity.get("relative_path")
    if (
        not isinstance(repository, str)
        or not Path(repository).is_absolute()
        or not isinstance(relative_path, str)
        or Path(relative_path).is_absolute()
        or relative_path != Path(relative_path).as_posix()
        or ".." in Path(relative_path).parts
    ):
        raise ContractError("corrected runner repository/path identity is invalid")
    require_equal(
        (Path(repository) / relative_path).resolve(),
        runner_path.resolve(),
        "corrected runner committed path",
    )
    for key in ("head_commit", "git_blob"):
        value = identity.get(key)
        if (
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{40}", value) is None
        ):
            raise ContractError(
                f"corrected runner {key} must be a lowercase 40-hex git object"
            )
    require_equal(
        require_sha256(identity.get("sha256"), "corrected runner SHA256"),
        runner_sha256,
        "corrected runner committed SHA256",
    )


def validate_v2_behavioral_identity(
    manifest: dict[str, Any],
    policy: dict[str, Any],
) -> None:
    """Prove v2 changes no search, selection, data, or runtime behavior."""

    supersession = policy["runtime_supersession"]
    v1_path = (HERE / supersession["source_manifest"]["path"]).resolve()
    v1 = load_json(v1_path)
    require_equal(
        set(manifest),
        set(v1) | {"supersedes", "runtime_supersession"},
        "v2 top-level manifest keys",
    )
    for key in ("schema_version", "status"):
        require_equal(
            manifest.get(key),
            v1.get(key),
            f"v2/v1 unchanged manifest {key}",
        )
    require_equal(
        manifest.get("manifest_id"),
        supersession["target_manifest"]["manifest_id"],
        "v2 target manifest ID",
    )
    require_equal(
        manifest.get("supersedes"),
        {
            "manifest_id": supersession["source_manifest"]["manifest_id"],
            "manifest_path": str(v1_path),
            "manifest_whole_file_sha256": supersession["source_manifest"][
                "whole_file_sha256"
            ],
            "manifest_internal_sha256": supersession["source_manifest"][
                "internal_manifest_sha256"
            ],
            "runtime_path": str(
                (HERE / supersession["source_runtime_path"]).resolve()
            ),
            "disposition": supersession["source_runtime_status"],
            "resume_permitted": False,
            "runtime_reuse_permitted": False,
        },
        "v2 superseded v1 identity",
    )
    for key in (
        "scope",
        "feeds_final_selection",
        "manual_override_permitted",
        "algorithm_only_selection_required",
        "search_space",
        "search_design",
        "selection",
        "post_front_matched_validation",
        "frozen_identity",
    ):
        require_equal(
            manifest.get(key),
            v1.get(key),
            f"v2/v1 unchanged behavioral contract {key}",
        )
    for key in (
        "policy_id",
        "policy_path",
        "generator_path",
        "runner_path",
        "sensitivity_result_path",
        "rule",
        "accuracy_retention_used_for_axis_derivation",
        "qualified_value_hull_used",
        "manual_override_used",
        "decision_audit",
        "source_identity",
        "sensitivity_result_sha256",
        "sensitivity_report_sha256",
        "analysis_erratum_contract_sha256",
        "post_front_contract_sha256",
    ):
        require_equal(
            manifest["derivation"].get(key),
            v1["derivation"].get(key),
            f"v2/v1 unchanged derivation {key}",
        )
    require_equal(
        manifest["runtime_supersession"][
            "valid_objective_observations_reused"
        ],
        0,
        "v1 objective observation reuse",
    )
    require_equal(
        manifest["runtime_supersession"]["v1_runtime_reused"],
        False,
        "v1 runtime reuse",
    )
    require_equal(
        {
            key: manifest["runtime_supersession"].get(key)
            for key in (
                "erratum_path",
                "erratum_sha256",
                "erratum_internal_sha256",
                "erratum_id",
                "target_runtime_path",
                "strict_metric_parser",
            )
        },
        {
            "erratum_path": str(
                (HERE / supersession["erratum_path"]).resolve()
            ),
            "erratum_sha256": supersession["erratum_sha256"],
            "erratum_internal_sha256": supersession[
                "erratum_internal_sha256"
            ],
            "erratum_id": supersession["erratum_id"],
            "target_runtime_path": str(
                (
                    HERE
                    / supersession["target_manifest"]["runtime_path"]
                ).resolve()
            ),
            "strict_metric_parser": supersession["strict_metric_parser"],
        },
        "v2 runtime supersession contract",
    )
    require_equal(
        set(manifest["runtime_supersession"]),
        {
            "erratum_path",
            "erratum_sha256",
            "erratum_internal_sha256",
            "erratum_id",
            "target_runtime_path",
            "strict_metric_parser",
            "valid_objective_observations_reused",
            "v1_runtime_reused",
            "corrected_runner_commit_identity",
        },
        "v2 runtime supersession keys",
    )
    validate_corrected_runner_commit_identity(
        manifest["runtime_supersession"].get(
            "corrected_runner_commit_identity"
        ),
        runner_path=Path(manifest["derivation"]["runner_path"]),
        runner_sha256=manifest["derivation"]["runner_sha256"],
    )


def require_corrected_runner_committed(
    policy: dict[str, Any],
    runner_path: Path,
) -> dict[str, str]:
    repo = Path(
        policy["frozen_identity"]["source_repositories"]["tao_automl"][
            "path"
        ]
    ).resolve()
    try:
        relative = runner_path.resolve().relative_to(repo)
    except ValueError as error:
        raise ContractError("expanded runner escaped TAO AutoML repository") from error
    relative_text = relative.as_posix()
    tracked = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "ls-files",
            "--error-unmatch",
            "--",
            relative_text,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        raise ContractError(
            "v2 generation requires corrected runner tracked in git"
        )
    for cached in (False, True):
        command = ["git", "-C", str(repo), "diff", "--quiet"]
        if cached:
            command.append("--cached")
        command.extend(["--", relative_text])
        if subprocess.run(command, check=False).returncode != 0:
            raise ContractError(
                "v2 generation requires corrected runner committed and clean"
            )
    head_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    head_blob = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "rev-parse",
            f"{head_commit}:{relative_text}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    current_blob = subprocess.run(
        ["git", "-C", str(repo), "hash-object", str(runner_path.resolve())],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require_equal(current_blob, head_blob, "corrected runner HEAD blob")
    identity = {
        "repository": str(repo),
        "relative_path": relative_text,
        "head_commit": head_commit,
        "git_blob": head_blob,
        "sha256": sha256_file(runner_path),
    }
    require_equal(
        subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        head_commit,
        "corrected runner HEAD stability",
    )
    return identity


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
    runner_path = (generator_path.parent / "expanded_search_runner.py").resolve()
    if not runner_path.is_file():
        raise ContractError(f"expanded-search runner is missing: {runner_path}")
    corrected_runner_commit_identity = require_corrected_runner_committed(
        policy,
        runner_path,
    )
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
        runner_path=runner_path,
        runner_sha256=sha256_file(runner_path),
        corrected_runner_commit_identity=corrected_runner_commit_identity,
        sensitivity_report_sha256=result["report_sha256"],
    )
    validate_v2_behavioral_identity(manifest, policy)
    require_equal(
        require_corrected_runner_committed(policy, runner_path),
        corrected_runner_commit_identity,
        "corrected runner identity before immutable manifest write",
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
