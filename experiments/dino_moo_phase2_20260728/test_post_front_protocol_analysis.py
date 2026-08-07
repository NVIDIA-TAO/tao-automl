"""Focused tests for the post-front dual-policy and verdict-only analysis."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import post_front_matched_aggregator as aggregator  # noqa: E402
import post_front_matched_manifest_generator as generator  # noqa: E402


ERRATUM_PATH = (HERE / "phase2_protocol_erratum.v1.json").resolve()
ERRATUM = json.loads(ERRATUM_PATH.read_text(encoding="utf-8"))
ERRATUM_CORRECTION = ERRATUM["corrections"][
    "post_front_paired_classification"
]
SHA_A = "a" * 64
TOLERANCE = 0.73553775
ACCURACY_ID = "seed_314159_rec_0"
MULTI_ID = "seed_271828_rec_0"
LATENCY_ID = "seed_161803_rec_0"


def _archive_candidate_ids() -> list[str]:
    return sorted(
        f"seed_{seed}_rec_{rec_id}"
        for seed in (314159, 271828, 161803)
        for rec_id in range(20)
    )


def _archive_snapshot() -> dict[str, Any]:
    candidate_ids = _archive_candidate_ids()
    return {
        "search_seeds": [314159, 271828, 161803],
        "recommendations_per_seed": 20,
        "candidate_count": 60,
        "terminal_candidate_count": 60,
        "successful_candidate_count": 60,
        "failed_candidate_count": 0,
        "manual_candidate_injection_used": False,
        "canonical_order": "ascending UTF-8 candidate_id",
        "candidate_ids": candidate_ids,
        "candidate_ids_sha256": generator.sha256_value(candidate_ids),
        "full_record_union_sha256": SHA_A,
        "candidate_table_projection_sha256": SHA_A,
        "expanded_combined_selection_sha256": SHA_A,
        "expanded_candidate_table_sha256": SHA_A,
        "expanded_candidate_table_csv_sha256": SHA_A,
        "expanded_integrity_audit_sha256": SHA_A,
        "seed_archives": [
            {
                "path": f"/frozen/seed_{seed}/seed_archive.v1.json",
                "whole_file_sha256": SHA_A,
                "internal_archive_sha256": SHA_A,
                "search_seed": seed,
                "record_count": 20,
                "terminal_record_count": 20,
                "successful_record_count": 20,
                "failed_record_count": 0,
                "candidate_ids_sha256": SHA_A,
                "full_records_sha256": SHA_A,
            }
            for seed in (314159, 271828, 161803)
        ],
    }


def _candidate(
    candidate_id: str,
    *,
    accuracy: float,
    latency_ms: float,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "selection_time_objective_values": {
            "mAP50": accuracy,
            "latency_ms": latency_ms,
        },
    }


def _manifest(
    *,
    multi_winner_id: str = MULTI_ID,
    geometric_distinct: bool = True,
) -> dict[str, Any]:
    candidate_ids = sorted([ACCURACY_ID, MULTI_ID, LATENCY_ID])
    candidates = [
        _candidate(ACCURACY_ID, accuracy=0.90, latency_ms=12.0),
        _candidate(MULTI_ID, accuracy=0.86, latency_ms=8.0),
        _candidate(LATENCY_ID, accuracy=0.83, latency_ms=4.0),
    ]
    paired = {
        "bootstrap_resamples": 10_000,
        "bootstrap_confidence_level": 0.95,
        "bootstrap_seed": 20_260_728,
        "practical_tolerance_ms": TOLERANCE,
        "policy_erratum_id": aggregator.EXPECTED_PROTOCOL_ERRATUM_ID,
        "policy_erratum_sha256": (
            aggregator.EXPECTED_PROTOCOL_ERRATUM_SHA256
        ),
        "both_policy_branches_must_be_emitted": True,
        "original_preregistered_bootstrap_classification": copy.deepcopy(
            ERRATUM_CORRECTION[
                "original_preregistered_bootstrap_classification"
            ]
        ),
        "effective_erratum_directional_classification": copy.deepcopy(
            ERRATUM_CORRECTION[
                "effective_erratum_directional_classification"
            ]
        ),
    }
    return {
        "source_artifacts": {
            "phase2_protocol_erratum": {
                "path": str(ERRATUM_PATH),
                "sha256": aggregator.EXPECTED_PROTOCOL_ERRATUM_SHA256,
                "erratum_id": aggregator.EXPECTED_PROTOCOL_ERRATUM_ID,
                "issued_at_utc": aggregator.EXPECTED_PROTOCOL_ERRATUM_ISSUED_AT,
            }
        },
        "paired_analysis": paired,
        "expanded_archive_snapshot": _archive_snapshot(),
        "candidate_derivation": {
            "candidate_ids": candidate_ids,
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
        "selection_snapshot": {
            "selections": {
                "accuracy": {
                    "winner_id": ACCURACY_ID,
                    "distinct_compromise": None,
                },
                "latency": {
                    "winner_id": LATENCY_ID,
                    "distinct_compromise": None,
                },
                "multi_objective": {
                    "winner_id": multi_winner_id,
                    "distinct_compromise": geometric_distinct,
                },
            }
        },
        "candidates": candidates,
        "schedule": {
            "allocations": [
                {
                    "allocation_id": f"allocation_{index:02d}",
                    "allocation_index": index,
                }
                for index in range(6)
            ]
        },
    }


def _measurements(
    manifest: dict[str, Any],
    *,
    median_by_candidate: dict[str, list[float]] | None = None,
    p95_by_candidate: dict[str, list[float]] | None = None,
) -> list[dict[str, Any]]:
    median_by_candidate = median_by_candidate or {
        ACCURACY_ID: [12.0] * 6,
        MULTI_ID: [8.0] * 6,
        LATENCY_ID: [4.0] * 6,
    }
    p95_by_candidate = p95_by_candidate or {
        ACCURACY_ID: [13.0] * 6,
        MULTI_ID: [9.0] * 6,
        LATENCY_ID: [5.0] * 6,
    }
    measurements = []
    for allocation in manifest["schedule"]["allocations"]:
        index = allocation["allocation_index"]
        for candidate_id in manifest["candidate_derivation"][
            "candidate_ids"
        ]:
            measurements.append(
                {
                    "allocation_id": allocation["allocation_id"],
                    "candidate_id": candidate_id,
                    "median_ms": median_by_candidate[candidate_id][index],
                    "p95_ms": p95_by_candidate[candidate_id][index],
                }
            )
    return measurements


@pytest.mark.parametrize(
    ("ci", "expected"),
    [
        ([-2.0, -TOLERANCE - 1e-9], "first_stably_faster"),
        ([TOLERANCE + 1e-9, 2.0], "second_stably_faster"),
        ([-TOLERANCE, TOLERANCE], "practically_equivalent"),
        ([-TOLERANCE - 1e-9, TOLERANCE], "uncertain"),
        ([-2.0, -TOLERANCE], "uncertain"),
        ([TOLERANCE, 2.0], "uncertain"),
    ],
)
def test_original_bootstrap_policy_has_exact_inclusive_boundaries(
    ci: list[float],
    expected: str,
) -> None:
    assert aggregator.original_bootstrap_claim(ci, TOLERANCE) == expected


def test_counterexample_preserves_original_claim_but_blocks_effective_edge() -> None:
    values = [-2.0, -1.9, -1.8, -1.7, -1.6, 0.0]
    ci = aggregator.paired_bootstrap_ci(
        values,
        resamples=10_000,
        confidence=0.95,
        seed=20_260_728,
    )
    assert (
        aggregator.original_bootstrap_claim(ci, TOLERANCE)
        == "first_stably_faster"
    )
    effective = aggregator.directional_pairwise_evidence(
        values,
        tolerance=TOLERANCE,
        confidence=0.95,
    )
    assert effective["first_faster_exact_test_passes"] is True
    assert effective["all_six_beyond_negative_tolerance"] is False
    assert effective["directional_claim"] == "no_stable_directional_claim"


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([-2.0] * 6, "first_stably_faster"),
        ([2.0] * 6, "second_stably_faster"),
        ([0.0] * 6, "no_stable_directional_claim"),
    ],
)
def test_effective_policy_unanimous_cases(
    values: list[float],
    expected: str,
) -> None:
    evidence = aggregator.directional_pairwise_evidence(
        values,
        tolerance=TOLERANCE,
        confidence=0.95,
    )
    assert evidence["directional_claim"] == expected
    if expected == "no_stable_directional_claim":
        assert "equivalent" not in evidence["directional_claim"]


def test_dual_outputs_keep_p95_separate_and_bind_legacy_alias() -> None:
    manifest = _manifest()
    p95 = {
        ACCURACY_ID: [10.0] * 6,
        MULTI_ID: [10.0] * 6,
        LATENCY_ID: [10.0] * 6,
    }
    analysis = aggregator.comparative_analysis(
        manifest,
        _measurements(manifest, p95_by_candidate=p95),
    )
    original = analysis[
        "original_preregistered_bootstrap_ordering_claims"
    ]
    effective = analysis["effective_stable_ordering_claims"]
    assert all(
        claim["classification"] == "practically_equivalent"
        for claim in original["p95_ms"]
    )
    assert effective["median_ms"]
    assert effective["p95_ms"] == []
    assert effective["no_stable_directional_claim_implies_equivalence"] is False
    assert analysis["stable_ordering_claims"] is effective["median_ms"]
    assert analysis["stable_ordering_claims_alias"] == {
        "target": "effective_stable_ordering_claims.median_ms",
        "exact_value_alias": True,
        "value_sha256": generator.sha256_value(effective["median_ms"]),
    }


def test_analysis_is_invariant_to_candidate_measurement_and_schedule_order() -> None:
    manifest = _manifest()
    measurements = _measurements(manifest)
    reference = aggregator.comparative_analysis(manifest, measurements)

    reordered = copy.deepcopy(manifest)
    reordered["candidate_derivation"]["candidate_ids"].reverse()
    reordered["candidates"].reverse()
    reordered["schedule"]["allocations"].reverse()
    observed = aggregator.comparative_analysis(
        reordered,
        list(reversed(measurements)),
    )
    assert observed == reference


@pytest.mark.parametrize(
    "missing_path",
    [
        ("paired_analysis", "original_preregistered_bootstrap_classification"),
        ("paired_analysis", "effective_erratum_directional_classification"),
        ("source_artifacts", "phase2_protocol_erratum"),
        ("expanded_archive_snapshot",),
    ],
)
def test_analysis_fails_closed_when_protocol_evidence_is_absent(
    missing_path: tuple[str, ...],
) -> None:
    manifest = _manifest()
    parent: dict[str, Any] = manifest
    for key in missing_path[:-1]:
        parent = parent[key]
    del parent[missing_path[-1]]
    with pytest.raises(aggregator.ContractError):
        aggregator.comparative_analysis(
            manifest,
            _measurements(_manifest()),
        )


def test_geometric_true_can_still_fail_actual_mode_hypothesis() -> None:
    manifest = _manifest(
        multi_winner_id=LATENCY_ID,
        geometric_distinct=True,
    )
    analysis = aggregator.comparative_analysis(
        manifest,
        _measurements(manifest),
    )
    evidence = analysis["hypothesis_distinctness_evidence"]
    assert evidence["selector_geometric_distinct_compromise"]["value"] is True
    assert (
        evidence["identity_distinctness"][
            "different_from_constrained_latency_winner"
        ]
        is False
    )
    assert evidence["actual_three_mode_intermediate_supported"] is False


def test_true_three_mode_case_reports_matched_deltas_without_reselection() -> None:
    manifest = _manifest()
    analysis = aggregator.comparative_analysis(
        manifest,
        _measurements(manifest),
    )
    evidence = analysis["hypothesis_distinctness_evidence"]
    assert evidence["actual_three_mode_intermediate_supported"] is True
    assert evidence["feeds_selection"] is False
    assert evidence["feeds_reselection"] is False
    assert evidence["selector_invoked_on_matched_measurements"] is False
    matched = evidence["matched_aggregate_latency"]
    assert matched["by_mode"] == {
        "accuracy": {"median_ms": 12.0, "p95_ms": 13.0},
        "latency": {"median_ms": 4.0, "p95_ms": 5.0},
        "multi_objective": {"median_ms": 8.0, "p95_ms": 9.0},
    }
    assert (
        matched["multi_objective_median_delta_from_accuracy_winner_ms"]
        == -4.0
    )
    assert (
        matched[
            "multi_objective_median_delta_from_"
            "constrained_latency_winner_ms"
        ]
        == 4.0
    )
    assert matched["multi_objective_vs_accuracy"][
        "effective_median_expected_direction_edge_exists"
    ]
    assert matched["multi_objective_vs_constrained_latency"][
        "effective_median_expected_direction_edge_exists"
    ]
    assert analysis["protocol_erratum"]["sha256"] == (
        aggregator.EXPECTED_PROTOCOL_ERRATUM_SHA256
    )
    assert analysis["expanded_archive_snapshot"] == (
        manifest["expanded_archive_snapshot"]
    )
