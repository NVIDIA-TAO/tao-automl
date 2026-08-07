"""Static contract tests for the pre-data phase-two protocol erratum.

The tests read only committed/local source artifacts.  They never inspect the
mutable expanded-search runtime, submit work, or consume post-front results.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
ERRATUM_PATH = HERE / "phase2_protocol_erratum.v1.json"
EXPANDED_MANIFEST_PATH = HERE / "expanded_search_manifest.v2.json"
EXPECTED_ERRATUM_SHA256 = (
    "95bba65099027459a50b5e74e43a4ab32c56057e534e70aa7f85bdc9246a7d13"
)
EXPECTED_EXPANDED_MANIFEST_SHA256 = (
    "9ac29e1aa07167a040d217fdab2d3cfdea0baad690dc95a70f2fe6715908793a"
)
EXPECTED_EXPANDED_MANIFEST_INTERNAL_SHA256 = (
    "910744ae2fead7e4e2e9a53fc672baef1ac43307e3979671b2b876fff422de96"
)
EXPECTED_POST_FRONT_CONTRACT_SHA256 = (
    "aba3a961bf50caf15803f271b59d7ffbd091414816d14f3deb793452f75ec281"
)
EXPECTED_SELECTOR_COMMIT = "83d9d7ecc783724f674cb954f9fbb6c91ea8b0eb"
EXPECTED_SELECTOR_BLOB = "3533fd3e1751f9ffdb03abe1cb58b8739ba4bd7f"
EXPECTED_SELECTOR_SHA256 = (
    "7e787a18bca05464e0043367aee4f2c8cff3d93aef7f9e92aaf88c47d255a532"
)
TOLERANCE_MS = 0.73553775


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def git_text(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def selector_functions() -> dict[str, str]:
    pinned_source = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY),
            "show",
            f"{EXPECTED_SELECTOR_COMMIT}:src/tao_automl/selection.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tree = ast.parse(pinned_source)
    return {
        node.name: ast.unparse(node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_erratum_identity_scope_and_whole_file_are_exact() -> None:
    erratum = load_json(ERRATUM_PATH)
    assert sha256_file(ERRATUM_PATH) == EXPECTED_ERRATUM_SHA256
    assert set(erratum) == {
        "schema_version",
        "erratum_id",
        "status",
        "issued_at_utc",
        "scope",
        "source_pins",
        "issuance_state",
        "corrections",
        "invariants",
        "post_front_enforcement",
    }
    assert erratum["schema_version"] == 1
    assert erratum["erratum_id"] == "dino_phase2_protocol_erratum_20260728_v1"
    assert (
        erratum["status"]
        == "issued_before_expanded_selection_and_post_front_measurement"
    )
    assert erratum["issued_at_utc"] == "2026-07-28T06:36:41Z"
    assert erratum["scope"] == {
        "model_family": "DINO ResNet50",
        "dataset_uri": (
            "s3://nvcf-storage-handling/data/"
            "tao_od_synthetic_full_dino_coco/"
        ),
        "other_models_permitted": False,
        "other_datasets_permitted": False,
        "expanded_manifest_interpretation_only": True,
        "post_front_analysis_policy_only": True,
    }


def test_source_pins_and_frozen_expanded_manifest_bytes_are_exact() -> None:
    erratum = load_json(ERRATUM_PATH)
    pins = erratum["source_pins"]
    manifest_pin = pins["expanded_manifest"]
    assert manifest_pin == {
        "path": "expanded_search_manifest.v2.json",
        "whole_file_sha256": EXPECTED_EXPANDED_MANIFEST_SHA256,
        "internal_manifest_sha256": (
            EXPECTED_EXPANDED_MANIFEST_INTERNAL_SHA256
        ),
    }
    assert sha256_file(EXPANDED_MANIFEST_PATH) == EXPECTED_EXPANDED_MANIFEST_SHA256
    manifest = load_json(EXPANDED_MANIFEST_PATH)
    assert manifest["manifest_sha256"] == EXPECTED_EXPANDED_MANIFEST_INTERNAL_SHA256
    unhashed_manifest = dict(manifest)
    del unhashed_manifest["manifest_sha256"]
    assert (
        canonical_sha256(unhashed_manifest)
        == EXPECTED_EXPANDED_MANIFEST_INTERNAL_SHA256
    )

    post_front = manifest["post_front_matched_validation"]
    assert canonical_sha256(post_front) == EXPECTED_POST_FRONT_CONTRACT_SHA256
    assert pins["frozen_post_front_contract"] == {
        "json_pointer": "/post_front_matched_validation",
        "canonical_sha256": EXPECTED_POST_FRONT_CONTRACT_SHA256,
        "allocation_count": 6,
        "paired_bootstrap_resamples": 10_000,
        "paired_bootstrap_confidence_level": 0.95,
        "paired_bootstrap_seed": 20_260_728,
        "practical_tolerance_ms": TOLERANCE_MS,
    }

    selector_pin = pins["production_selector"]
    assert selector_pin == {
        "repository_path": str(REPOSITORY),
        "commit": EXPECTED_SELECTOR_COMMIT,
        "relative_path": "src/tao_automl/selection.py",
        "git_blob": EXPECTED_SELECTOR_BLOB,
        "sha256": EXPECTED_SELECTOR_SHA256,
        "authority": "executed production behavior",
    }
    assert (
        git_text(
            "rev-parse",
            f"{EXPECTED_SELECTOR_COMMIT}:src/tao_automl/selection.py",
        )
        == EXPECTED_SELECTOR_BLOB
    )
    pinned_source = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY),
            "show",
            f"{EXPECTED_SELECTOR_COMMIT}:src/tao_automl/selection.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    assert sha256_bytes(pinned_source) == EXPECTED_SELECTOR_SHA256


def test_issuance_cutoff_is_exactly_before_union_and_post_front_data() -> None:
    state = load_json(ERRATUM_PATH)["issuance_state"]
    assert state == {
        "expanded_runtime_path": "runtime/expanded_search_v2",
        "successful_candidate_count": 15,
        "successful_candidate_count_by_search_seed": {
            "161803": 5,
            "271828": 5,
            "314159": 5,
        },
        "expected_candidate_count": 60,
        "expanded_archive_complete": False,
        "candidate_objective_values_already_existed": True,
        "completed_union_selection_existed": False,
        "final_global_pareto_front_known_or_used": False,
        "absent_expanded_outputs": [
            "runtime/expanded_search_v2/expanded_combined_selection.json",
            "runtime/expanded_search_v2/expanded_candidate_table.json",
            "runtime/expanded_search_v2/expanded_candidate_table.csv",
            "runtime/expanded_search_v2/expanded_integrity_audit.json",
            "runtime/expanded_search_v2/expanded_completion.json",
        ],
        "post_front": {
            "manifest_path": "post_front_matched_manifest.v1.json",
            "manifest_existed": False,
            "tao_job_count": 0,
            "slurm_allocation_count": 0,
            "candidate_measurement_count": 0,
            "pairwise_comparison_count": 0,
        },
    }


def test_frozen_old_prose_is_preserved_as_erratum_input() -> None:
    erratum = load_json(ERRATUM_PATH)
    manifest = load_json(EXPANDED_MANIFEST_PATH)
    affected = erratum["corrections"]["selection_tie_break_documentation"][
        "affected_frozen_prose"
    ]
    assert affected == [
        {
            "json_pointer": "/selection/accuracy_mode/tie_break",
            "value": "lower stable latency, then deterministic candidate key",
        },
        {
            "json_pointer": "/selection/multi_objective_mode/tie_break",
            "value": (
                "lower max normalized regret, then lower sum normalized "
                "regret, then higher accuracy, then lower latency, then "
                "deterministic candidate key"
            ),
        },
    ]
    assert (
        manifest["selection"]["accuracy_mode"]["tie_break"]
        == affected[0]["value"]
    )
    assert (
        manifest["selection"]["multi_objective_mode"]["tie_break"]
        == affected[1]["value"]
    )


def test_erratum_matches_actual_pinned_selection_formulas_and_keys() -> None:
    behavior = load_json(ERRATUM_PATH)["corrections"][
        "selection_tie_break_documentation"
    ]["authoritative_behavior"]
    canonical_key = [
        "canonical_spec_fingerprint_ascending",
        "candidate_id_ascending",
    ]
    assert behavior["canonical_audit_input_order"] == canonical_key
    assert behavior["accuracy_mode"] == {
        "primary": "maximum finite accuracy",
        "accuracy_tie_predicate": (
            "best_accuracy - candidate_accuracy <= accuracy_tolerance"
        ),
        "accuracy_tolerance": 1e-12,
        "latency_anchor": (
            "minimum raw latency within the accuracy-tied cohort"
        ),
        "latency_tie_predicate": (
            "candidate_latency - minimum_tied_latency <= latency_tolerance"
        ),
        "latency_tolerance_ms": TOLERANCE_MS,
        "final_key": canonical_key,
    }
    assert behavior["latency_mode"] == {
        "eligibility": (
            "configured accuracy-winner-relative retention constraint"
        ),
        "anchor_key": [
            "raw_latency_ascending",
            "canonical_spec_fingerprint_ascending",
        ],
        "anchor_exact_key_ties_inherit_canonical_audit_input_order": True,
        "tie_predicate": (
            "candidate_latency - anchor_latency <= latency_tolerance OR "
            "candidate_and_anchor_latency_confidence_intervals_overlap"
        ),
        "latency_tolerance_ms": TOLERANCE_MS,
        "final_key": [
            "accuracy_descending",
            "canonical_spec_fingerprint_ascending",
            "candidate_id_ascending",
        ],
    }
    multi = behavior["multi_objective_mode"]
    assert multi["duplicate_objective_representative_key"] == canonical_key
    assert multi["compromise_score_formula"] == (
        "max(weight_accuracy * normalized_accuracy_regret, weight_latency * "
        "normalized_latency_regret) + augmentation_rho * (weight_accuracy * "
        "normalized_accuracy_regret + weight_latency * "
        "normalized_latency_regret)"
    )
    assert multi["ideal_distance_formula"] == (
        "hypot(weight_accuracy * normalized_accuracy_regret, weight_latency "
        "* normalized_latency_regret)"
    )
    assert multi["balance_gap_formula"] == (
        "abs(weight_accuracy * normalized_accuracy_regret - weight_latency "
        "* normalized_latency_regret)"
    )
    assert multi["final_key"] == [
        "normalized_accuracy_regret_ascending",
        "canonical_spec_fingerprint_ascending",
        "candidate_id_ascending",
    ]

    functions = selector_functions()
    assert (
        "return sorted(audits, key=lambda item: "
        "(item.fingerprint, item.candidate_id))"
        in functions["_build_audits"]
    )
    assert (
        "best_accuracy - float(item.accuracy) <= "
        "config.accuracy_tolerance"
        in functions["_choose_accuracy"]
    )
    assert (
        "float(item.latency) - best_latency <= config.latency_tolerance"
        in functions["_choose_accuracy"]
    )
    assert (
        "return min(latency_tied, key=lambda item: "
        "(item.fingerprint, item.candidate_id))"
        in functions["_choose_accuracy"]
    )
    assert (
        "fastest = min(feasible, key=lambda item: "
        "(float(item.latency), item.fingerprint))"
        in functions["_choose_latency"]
    )
    assert (
        "or _latency_intervals_overlap(item, fastest)"
        in functions["_choose_latency"]
    )
    assert (
        "key=lambda item: (-float(item.accuracy), item.fingerprint, "
        "item.candidate_id)"
        in functions["_choose_latency"]
    )
    assert (
        "ordered = sorted(aliases, key=lambda item: "
        "(item.fingerprint, item.candidate_id))"
        in functions["_deduplicate_objective_points"]
    )
    scores = functions["_set_normalized_scores"]
    assert (
        "max(weighted_accuracy, weighted_latency) + "
        "config.augmentation_rho * (weighted_accuracy + weighted_latency)"
        in scores
    )
    assert "math.hypot(weighted_accuracy, weighted_latency)" in scores
    assert "abs(weighted_accuracy - weighted_latency)" in scores
    compromise = functions["_choose_compromise"]
    for fragment in (
        "float(item.compromise_score) - best_score <= config.score_tolerance",
        "float(item.ideal_distance) - best_distance <= config.score_tolerance",
        "float(item.balance_gap) - best_gap <= config.score_tolerance",
        "key=lambda item: (float(item.normalized_accuracy_regret), "
        "item.fingerprint, item.candidate_id)",
    ):
        assert fragment in compromise


def test_original_and_effective_paired_policies_are_both_exact() -> None:
    policy = load_json(ERRATUM_PATH)["corrections"][
        "post_front_paired_classification"
    ]
    assert policy["endpoints"] == ["median_ms", "p95_ms"]
    assert policy["both_policy_branches_must_be_emitted"] is True
    original = policy["original_preregistered_bootstrap_classification"]
    assert original == {
        "status": "preserved_and_reported",
        "bootstrap": (
            "paired percentile bootstrap resampling the six allocation-level "
            "differences with replacement"
        ),
        "stable_difference_rule": (
            "Claim a relative latency direction only when the paired 95 "
            "percent CI lies wholly below negative practical tolerance or "
            "wholly above positive practical tolerance."
        ),
        "equivalence_rule": (
            "Report practical equivalence when the complete paired 95 "
            "percent CI lies within the plus-or-minus practical tolerance "
            "band; otherwise report uncertainty."
        ),
        "first_stably_faster_condition": (
            "bootstrap_ci_high < -practical_tolerance_ms"
        ),
        "second_stably_faster_condition": (
            "bootstrap_ci_low > practical_tolerance_ms"
        ),
        "practically_equivalent_condition": (
            "bootstrap_ci_low >= -practical_tolerance_ms AND "
            "bootstrap_ci_high <= practical_tolerance_ms"
        ),
        "otherwise": "uncertain",
        "point_classification_preserved": True,
    }
    effective = policy["effective_erratum_directional_classification"]
    assert effective == {
        "status": "controls_directional_and_ordering_claims",
        "bootstrap_role": "descriptive_only",
        "test": (
            "one-sided exact paired sign-flip permutation test after shifting "
            "each paired difference by the claimed practical-tolerance "
            "boundary"
        ),
        "allocation_count": 6,
        "permutation_count": 64,
        "alpha": 0.05,
        "practical_tolerance_ms": TOLERANCE_MS,
        "first_stably_faster_condition": (
            "p_value_one_sided <= alpha AND every one of the six paired "
            "differences is strictly below -practical_tolerance_ms"
        ),
        "second_stably_faster_condition": (
            "p_value_one_sided <= alpha AND every one of the six paired "
            "differences is strictly above practical_tolerance_ms"
        ),
        "otherwise": "no_stable_directional_claim",
        "no_stable_directional_claim_implies_equivalence": False,
        "median_and_p95_classified_independently": True,
        "headline_stable_ordering_endpoint": "median_ms",
        "scope": "pairwise_only",
        "multiplicity_adjustment": "none",
        "simultaneous_total_order_inference_permitted": False,
    }


def test_nonselection_invariants_and_enforcement_are_fail_closed() -> None:
    erratum = load_json(ERRATUM_PATH)
    assert erratum["invariants"] == {
        "expanded_manifest_mutation_permitted": False,
        "expanded_search_runner_change_required": False,
        "search_space_changed": False,
        "search_budget_changed": False,
        "training_or_evaluation_changed": False,
        "selection_configuration_changed": False,
        "selection_implementation_changed": False,
        "archive_objective_value_changed": False,
        "selection_time_measurement_replacement_permitted": False,
        "post_front_measurements_feed_reselection": False,
        "post_front_winner_reselection_permitted": False,
        "manual_candidate_addition_or_removal_permitted": False,
        "manual_winner_override_permitted": False,
    }
    correction = erratum["corrections"][
        "selection_tie_break_documentation"
    ]
    for key in (
        "selection_execution_changed",
        "candidate_objectives_changed",
        "pareto_ranks_changed",
        "algorithm_winner_changed_or_overridden",
        "manual_winner_override_permitted",
    ):
        assert correction[key] is False
    assert erratum["post_front_enforcement"] == {
        "erratum_whole_file_sha256_must_be_pinned": True,
        "manifest_generation_must_validate_exact_erratum": True,
        "launch_must_validate_exact_erratum": True,
        "aggregation_must_validate_exact_erratum": True,
        "erratum_absence_or_drift_blocks_execution": True,
        "original_bootstrap_classification_must_be_reported": True,
        "effective_directional_classification_must_be_reported": True,
        "effective_directional_classification_controls_stable_ordering_claims": (
            True
        ),
    }
