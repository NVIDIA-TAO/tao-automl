#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the two-candidate matched latency aggregation path."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "latency_90_policy_matched_aggregator.py"
SPEC = importlib.util.spec_from_file_location(
    "latency_90_policy_matched_aggregator_test_module",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
AGGREGATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGGREGATOR)
MATCHED = AGGREGATOR.MATCHED


def synthetic_measurements(
    *,
    first_medians: list[float],
    second_medians: list[float],
    first_p95s: list[float],
    second_p95s: list[float],
) -> tuple[dict, list[dict]]:
    projection = MATCHED.build_projection()
    manifest = MATCHED.compatibility_manifest(projection)
    first, second = projection["candidate_derivation"]["candidate_ids"]
    cells = []
    for index, allocation in enumerate(
        manifest["schedule"]["allocations"]
    ):
        for candidate_id, median, p95 in (
            (first, first_medians[index], first_p95s[index]),
            (second, second_medians[index], second_p95s[index]),
        ):
            cells.append({
                "allocation_id": allocation["allocation_id"],
                "allocation_index": allocation["allocation_index"],
                "candidate_id": candidate_id,
                "median_ms": median,
                "p95_ms": p95,
            })
    return manifest, cells


def test_pairwise_analysis_emits_exact_and_descriptive_branches():
    manifest, cells = synthetic_measurements(
        first_medians=[57.0, 57.1, 57.2, 57.0, 57.1, 57.2],
        second_medians=[57.1, 57.2, 57.3, 57.1, 57.2, 57.3],
        first_p95s=[57.4, 57.5, 57.6, 57.4, 57.5, 57.6],
        second_p95s=[57.5, 57.6, 57.7, 57.5, 57.6, 57.7],
    )
    analysis = AGGREGATOR.comparative_analysis(manifest, cells)
    assert analysis["directional_rule"] == {
        "exact_one_sided_tolerance_shifted_sign_flip_required": True,
        "all_six_differences_beyond_tolerance_required": True,
        "alpha": 0.050000000000000044,
        "absence_of_direction_implies_equivalence": False,
        "simultaneous_total_order_claimed": False,
    }
    for endpoint in analysis["pairwise_endpoints"].values():
        assert len(endpoint["paired_differences_ms"]) == 6
        assert endpoint["descriptive_practical_equivalence"] is True
        assert endpoint["effective_classification"] == (
            "no_stable_direction_descriptive_practical_equivalence"
        )
        exact = endpoint["exact_tolerance_shifted_sign_flip"]
        assert exact["directional_claim"] == (
            "no_stable_directional_claim"
        )
        assert exact["first_faster_test"]["permutation_count"] == 64
        assert exact["second_faster_test"]["permutation_count"] == 64


def test_all_six_threshold_rule_establishes_stable_direction():
    manifest, cells = synthetic_measurements(
        first_medians=[55.0] * 6,
        second_medians=[57.0] * 6,
        first_p95s=[55.5] * 6,
        second_p95s=[57.5] * 6,
    )
    analysis = AGGREGATOR.comparative_analysis(manifest, cells)
    for endpoint in analysis["pairwise_endpoints"].values():
        assert endpoint["all_six_beyond_negative_tolerance"] is True
        assert endpoint["effective_classification"] == (
            "first_stably_faster"
        )
        exact = endpoint["exact_tolerance_shifted_sign_flip"]
        assert exact["first_faster_exact_test_passes"] is True
        assert exact["first_faster_test"]["p_value_one_sided"] == 0.015625


def test_quality_projection_includes_all_required_diagnostics():
    projection = MATCHED.build_projection()
    manifest = MATCHED.compatibility_manifest(projection)
    measurement = {
        "median_ms": 57.0,
        "p95_ms": 57.5,
        "mad_ms": 0.1,
        "iqr_ms": 0.2,
        "robust_cv": 0.003,
        "round_median_range_ms": 0.3,
        "round_drift_ms": -0.1,
        "device_median_range_ms": 0.4,
        "bootstrap_median_ci95_ms": [56.9, 57.1],
        "bootstrap_p95_ci95_ms": [57.3, 57.7],
        "raw_sample_count_total": 4000,
        "is_valid": True,
        "invalid_reasons": [],
    }
    result = AGGREGATOR.quality_gate_projection(
        measurement,
        manifest,
    )
    assert result["all_quality_gates_pass"] is True
    assert set(result["gates"]) == {
        "robust_cv",
        "round_median_range",
        "absolute_round_drift",
        "device_median_range",
        "median_bootstrap_ci_width",
        "raw_sample_count",
        "proven_aggregator_validity",
    }
    assert result["values"]["p95_bootstrap_ci_width_ms"] == (
        0.4000000000000057
    )


def test_complete_cell_table_rejects_partial_matrix():
    projection = MATCHED.build_projection()
    manifest = MATCHED.compatibility_manifest(projection)
    try:
        AGGREGATOR.complete_cell_table(manifest, [])
    except AGGREGATOR.AggregationError as error:
        assert "complete 12-cell matrix" in str(error)
    else:
        raise AssertionError("partial matched matrix was accepted")


def test_aggregator_never_invokes_selector_or_feeds_reselection():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "analyze_archive" not in source
    assert "tao_automl.selection" not in source
    assert MATCHED.SELECTION_ISOLATION == {
        "selector_invoked_on_matched_measurements": False,
        "selection_time_objectives_replaced": False,
        "measurements_feed_selection": False,
        "measurements_feed_reselection": False,
        "algorithm_selected_candidate_overridden": False,
    }
