from __future__ import annotations

import copy
from itertools import combinations
import sys
from pathlib import Path
from typing import Any

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import latency_feasible_matched_aggregator as aggregator  # noqa: E402
import latency_feasible_matched_block_runner as block_runner  # noqa: E402


SHA = "a" * 64
COHORT = [
    "seed_161803_rec_14",
    "seed_271828_rec_16",
    "seed_271828_rec_18",
    "seed_314159_rec_12",
]
ACCURACY = {
    "seed_161803_rec_14": 0.6503731411565659,
    "seed_271828_rec_16": 0.6544218576499151,
    "seed_271828_rec_18": 0.6554138278683255,
    "seed_314159_rec_12": 0.6517250365478822,
}
LATENCY = {
    "seed_161803_rec_14": 66.5894095,
    "seed_271828_rec_16": 66.82186425,
    "seed_271828_rec_18": 66.23099475,
    "seed_314159_rec_12": 66.685121,
}


def _manifest() -> dict[str, Any]:
    return {
        "candidate_derivation": {
            "candidate_ids": copy.deepcopy(COHORT),
            "selector_replay_proof": {
                "selector_settings": {
                    "accuracy_metric": "mAP50",
                    "latency_metric": "latency_ms",
                    "accuracy_tolerance": 1e-12,
                    "latency_accuracy_retention": {
                        "type": "relative",
                        "retained_fraction": 0.98,
                        "reference": "accuracy_winner",
                    },
                    "multi_objective_min_accuracy": None,
                }
            },
        },
        "paired_analysis": {
            "practical_tolerance_ms": 0.73553775,
        },
        "selection_snapshot": {
            "preserved_unchanged": True,
            "selections": {
                "accuracy": {
                    "winner_id": "seed_271828_rec_18",
                },
                "latency": {
                    "winner_id": "seed_271828_rec_18",
                    "latency_tied_candidate_ids": copy.deepcopy(COHORT),
                },
                "multi_objective": {
                    "winner_id": "seed_271828_rec_19",
                },
            },
        },
        "candidates": [
            {
                "candidate_id": candidate_id,
                "selection_time_objective_values": {
                    "mAP50": ACCURACY[candidate_id],
                    "latency_ms": LATENCY[candidate_id],
                },
            }
            for candidate_id in COHORT
        ],
    }


def _pairs(*, equivalent: bool) -> list[dict[str, Any]]:
    return [
        {
            "first_candidate_id": first,
            "second_candidate_id": second,
            "median_effective_classification": {
                "descriptive_practical_equivalence": equivalent,
            },
            "p95_effective_classification": {
                "descriptive_practical_equivalence": equivalent,
            },
        }
        for first, second in combinations(sorted(COHORT), 2)
    ]


def _matched_aggregates() -> dict[str, dict[str, float]]:
    return {
        candidate_id: {
            "median_ms": latency,
            "p95_ms": latency + 0.5,
        }
        for candidate_id, latency in LATENCY.items()
    }


def _edge(faster: str, slower: str, endpoint: str) -> dict[str, Any]:
    return {
        "faster_candidate_id": faster,
        "slower_candidate_id": slower,
        "endpoint": endpoint,
        "scope": "pairwise_only",
        "simultaneous_order_inference_permitted": False,
    }


def _plan() -> dict[str, Any]:
    candidates = [
        {
            "candidate_id": candidate_id,
            "position": position,
            "run_label": (
                f"latency_feasible_allocation_00_p{position:03d}_"
                f"{candidate_id}"
            ),
            "checkpoint_path": f"/checkpoints/{candidate_id}.pth",
            "config_relative_path": f"configs/{candidate_id}.yaml",
            "checkpoint_sha256": SHA,
            "resolved_model_spec_sha256": SHA,
            "candidate_table_record_sha256": SHA,
            "config_sha256": SHA,
        }
        for position, candidate_id in enumerate(COHORT)
    ]
    plan = {
        "schema_version": 1,
        "manifest_id": "dino_latency_feasible_matched_20260728_v1",
        "manifest_sha256": SHA,
        "schedule_sha256": SHA,
        "benchmark_sha256": SHA,
        "latency_stats_sha256": SHA,
        "block_runner_sha256": SHA,
        "allocation_index": 0,
        "allocation_id": "latency_feasible_allocation_00",
        "design_row_index": 0,
        "gpu_count": 8,
        "num_nodes": 1,
        "feeds_final_selection": False,
        "feeds_reselection": False,
        "manual_candidate_addition_or_removal_permitted": False,
        "winner_override_permitted": False,
        "selection_time_objective_replacement_permitted": False,
        "selector_invoked_on_matched_measurements": False,
        "selection_time_objectives_replaced": False,
        "measurements_feed_selection": False,
        "measurements_feed_reselection": False,
        "algorithm_selected_candidate_overridden": False,
        "output_contract": {
            "root_expression": "$TAO_RESULTS_ROOT/$TAO_JOB_ID",
            "sdk_job_scoped": True,
            "relative_layout": (
                "dino_moo_phase2_20260728/latency_feasible_matched/"
                "<manifest_id>/<allocation_id>"
            ),
        },
        "candidate_count": 4,
        "candidates": candidates,
        "latency_protocol": {
            "warmup_iterations": 50,
            "timed_iterations": 100,
            "repeated_rounds": 5,
            "preloaded_batches": 16,
            "batch_size_per_gpu": 1,
            "precision": "fp32",
            "tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "benchmark_seed": 20260727,
            "tail_percentile": 95,
            "bootstrap_resamples": 5000,
            "bootstrap_confidence_level": 0.95,
            "bootstrap_seed": 424242,
            "synchronization": (
                "cuda_sync_each_sample_and_nccl_barrier"
            ),
            "timed_scope": "model_forward_plus_dino_gpu_postprocess",
            "fixed_preprocessed_shapes": {
                "model_input": [1, 4, 800, 1333],
                "image_tensor": [1, 3, 800, 1333],
                "padding_mask": [1, 1, 800, 1333],
            },
        },
    }
    plan["block_plan_sha256"] = block_runner.sha256_value(plan)
    return plan


def test_block_plan_freezes_exact_protocol_cohort_and_isolation() -> None:
    plan = _plan()
    block_runner.validate_plan(plan)
    assert plan["candidate_count"] == 4
    assert (
        plan["latency_protocol"]["warmup_iterations"],
        plan["latency_protocol"]["repeated_rounds"],
        plan["latency_protocol"]["timed_iterations"],
        plan["gpu_count"],
    ) == (50, 5, 100, 8)
    assert 5 * 100 * 8 == 4000
    for key in aggregator.SELECTION_ISOLATION_FLAGS:
        assert plan[key] is False


def test_block_plan_rejects_missing_isolation_flag() -> None:
    plan = _plan()
    del plan["measurements_feed_reselection"]
    unhashed = dict(plan)
    del unhashed["block_plan_sha256"]
    plan["block_plan_sha256"] = block_runner.sha256_value(unhashed)
    with pytest.raises(ValueError, match="measurements_feed_reselection"):
        block_runner.validate_plan(plan)


def test_effective_direction_requires_exact_test_and_all_six_margin() -> None:
    tolerance = 0.73553775
    evidence = aggregator.directional_pairwise_evidence(
        [-0.8, -0.9, -1.0, -1.1, -1.2, -1.3],
        tolerance=tolerance,
        confidence=0.95,
    )
    assert evidence["first_faster_test"]["permutation_count"] == 64
    assert evidence["first_faster_test"]["p_value_one_sided"] == 1 / 64
    assert evidence["all_six_beyond_negative_tolerance"] is True
    assert evidence["directional_claim"] == "first_stably_faster"

    mixed = aggregator.directional_pairwise_evidence(
        [-0.8, -0.9, -1.0, -1.1, -1.2, -0.7],
        tolerance=tolerance,
        confidence=0.95,
    )
    assert mixed["all_six_beyond_negative_tolerance"] is False
    assert mixed["directional_claim"] == "no_stable_directional_claim"


def test_no_direction_does_not_imply_equivalence() -> None:
    evidence = {
        "directional_claim": "no_stable_directional_claim",
    }
    unresolved = aggregator.endpoint_pairwise_classification(
        evidence,
        "crosses_a_practical_tolerance_boundary",
    )
    assert unresolved == {
        "effective_directional_classification": (
            "no_stable_directional_claim"
        ),
        "descriptive_practical_equivalence": False,
        "combined_interpretation": "unresolved",
        "failure_to_establish_direction_implies_equivalence": False,
    }
    equivalent = aggregator.endpoint_pairwise_classification(
        evidence,
        "entirely_within_practical_tolerance",
    )
    assert equivalent["descriptive_practical_equivalence"] is True
    assert "descriptively_practically_equivalent" in equivalent[
        "combined_interpretation"
    ]


def test_outcome_a_requires_rec18_to_beat_every_other_candidate() -> None:
    edges = [
        _edge("seed_271828_rec_18", candidate_id, "median_ms")
        for candidate_id in COHORT
        if candidate_id != "seed_271828_rec_18"
    ]
    evidence = aggregator.feasible_cohort_validation_evidence(
        _manifest(),
        _pairs(equivalent=False),
        edges,
        [],
        _matched_aggregates(),
    )
    assert evidence["outcome"] == "A_rec18_stably_fastest"
    assert evidence[
        "higher_accuracy_tie_break_needed_for_practical_result"
    ] is False


def test_outcome_b_requires_all_median_and_p95_intervals_inside_band() -> None:
    evidence = aggregator.feasible_cohort_validation_evidence(
        _manifest(),
        _pairs(equivalent=True),
        [],
        [],
        _matched_aggregates(),
    )
    assert evidence["outcome"] == (
        "B_all_feasible_candidates_practically_equivalent"
    )
    assert evidence[
        "higher_accuracy_tie_break_needed_for_practical_result"
    ] is True

    mixed = _pairs(equivalent=True)
    mixed[0]["p95_effective_classification"][
        "descriptive_practical_equivalence"
    ] = False
    unresolved = aggregator.feasible_cohort_validation_evidence(
        _manifest(),
        mixed,
        [],
        [],
        _matched_aggregates(),
    )
    assert unresolved["outcome"] == "D_mixed_or_unresolved"


def test_outcome_c_never_overrides_frozen_selection() -> None:
    evidence = aggregator.feasible_cohort_validation_evidence(
        _manifest(),
        _pairs(equivalent=False),
        [
            _edge(
                "seed_161803_rec_14",
                "seed_271828_rec_18",
                "median_ms",
            )
        ],
        [],
        _matched_aggregates(),
    )
    assert evidence["outcome"] == (
        "C_another_feasible_candidate_stably_faster"
    )
    assert evidence["frozen_winner_preserved"] is True
    assert evidence["algorithm_change_required"] is False
    for key in aggregator.SELECTION_ISOLATION_FLAGS:
        assert evidence[key] is False


def test_final_report_emits_exact_matched_measurement_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    manifest["manifest_id"] = (
        "dino_latency_feasible_matched_20260728_v1"
    )
    manifest["schedule"] = {"schedule_sha256": SHA}
    ledger = {
        "ledger_revision": 1,
        "superseded_submissions": [],
        "submission_recovery_events": [],
        "source_checks": {},
    }
    monkeypatch.setattr(
        aggregator,
        "comparative_analysis",
        lambda _manifest, _measurements: {"validated": True},
    )
    report = aggregator.build_final_report(
        manifest,
        SHA,
        ledger,
        SHA,
        [],
        [],
        {"complete_block_contract": "pass"},
    )
    isolation = report["selection_isolation"]
    for key in aggregator.SELECTION_ISOLATION_FLAGS:
        assert isolation[key] is False
    assert "selector_invoked_on_postfront_measurements" not in isolation
    assert report["original_selection_snapshot"] == manifest[
        "selection_snapshot"
    ]
