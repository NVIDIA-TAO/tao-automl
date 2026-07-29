# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for constrained Pareto archive selection."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tao_automl.objectives import parse_objective_config
from tao_automl.selection import (
    AccuracyConstraint,
    SelectionConfig,
    analyze_archive,
)


@dataclass
class Candidate:
    id: str
    accuracy: object
    latency: object
    specs: dict | None = None
    status: str = "success"
    extra: dict | None = None

    @property
    def objective_values(self):
        values = {"accuracy": self.accuracy, "latency": self.latency}
        values.update(self.extra or {})
        return values

    def __post_init__(self):
        if self.specs is None:
            self.specs = {"candidate": self.id}


def config(
    *,
    mode="multi_objective",
    constraint=None,
    multi_objective_min_accuracy=None,
    accuracy_tolerance=0.0,
    latency_tolerance=0.0,
):
    return SelectionConfig(
        mode=mode,
        accuracy_metric="accuracy",
        latency_metric="latency",
        latency_accuracy_retention=constraint
        or AccuracyConstraint(kind="absolute", value=1_000_000.0),
        multi_objective_min_accuracy=multi_objective_min_accuracy,
        accuracy_tolerance=accuracy_tolerance,
        latency_tolerance=latency_tolerance,
    )


def audits_by_id(analysis):
    return {item.candidate_id: item for item in analysis.audits}


def test_strictly_dominated_candidate_is_excluded_from_front():
    analysis = analyze_archive(
        [Candidate("A", 0.80, 20.0), Candidate("B", 0.90, 10.0)],
        config(),
    )
    audits = audits_by_id(analysis)

    assert audits["B"].pareto_rank == 0
    assert audits["A"].pareto_rank == 1
    assert audits["A"].dominated_by == ("B",)
    assert analysis.multi_objective.winner_id == "B"


def test_three_point_front_selects_balanced_chebyshev_compromise():
    analysis = analyze_archive(
        [
            Candidate("A", 0.92, 20.0),
            Candidate("M", 0.89, 14.0),
            Candidate("L", 0.86, 10.0),
        ],
        config(),
    )
    audits = audits_by_id(analysis)

    assert {item.pareto_rank for item in audits.values()} == {0}
    assert audits["A"].normalized_accuracy_regret == pytest.approx(0.0)
    assert audits["A"].normalized_latency_regret == pytest.approx(1.0)
    assert audits["M"].normalized_accuracy_regret == pytest.approx(0.5)
    assert audits["M"].normalized_latency_regret == pytest.approx(0.4)
    assert audits["L"].normalized_accuracy_regret == pytest.approx(1.0)
    assert audits["L"].normalized_latency_regret == pytest.approx(0.0)
    assert audits["M"].compromise_score == pytest.approx(0.25000045)
    assert analysis.multi_objective.winner_id == "M"
    assert analysis.multi_objective.distinct_compromise is True
    assert analysis.multi_objective.fallback_used is False


def test_latency_winner_dominates_fake_middle_and_triggers_fallback():
    analysis = analyze_archive(
        [
            Candidate("A", 0.92, 20.0),
            Candidate("L", 0.89, 10.0),
            Candidate("D", 0.88, 11.0),
        ],
        config(),
    )
    audits = audits_by_id(analysis)

    assert audits["D"].pareto_rank == 1
    assert "L" in audits["D"].dominated_by
    assert analysis.multi_objective.winner_id in {"A", "L"}
    assert analysis.multi_objective.distinct_compromise is False
    assert analysis.multi_objective.fallback_used is True
    assert analysis.multi_objective.reason.startswith(
        "No distinct Pareto compromise exists under the configured "
        "multi-objective eligibility policy"
    )


def test_duplicate_objective_points_have_deterministic_representative():
    first = Candidate("first", 0.90, 10.0, specs={"z": 1})
    second = Candidate("second", 0.90, 10.0, specs={"a": 1})

    forward = analyze_archive([first, second], config())
    reverse = analyze_archive([second, first], config())
    forward_audits = audits_by_id(forward)
    reverse_audits = audits_by_id(reverse)

    assert forward.multi_objective.winner_id == reverse.multi_objective.winner_id
    representatives = {
        item.duplicate_representative for item in forward_audits.values()
    }
    assert len(representatives) == 1
    assert {
        item.duplicate_aliases for item in forward_audits.values()
    } == {("second", "first")} or {
        item.duplicate_aliases for item in forward_audits.values()
    } == {("first", "second")}
    assert {
        item.duplicate_representative for item in reverse_audits.values()
    } == representatives


def test_candidate_ordering_does_not_change_analysis_or_winner():
    candidates = [
        Candidate("A", 0.92, 20.0),
        Candidate("M", 0.89, 14.0),
        Candidate("L", 0.86, 10.0),
        Candidate("D", 0.84, 16.0),
    ]
    baseline = analyze_archive(candidates, config())
    baseline_audits = audits_by_id(baseline)

    for permutation in itertools.permutations(candidates):
        actual = analyze_archive(permutation, config())
        actual_audits = audits_by_id(actual)
        assert actual.multi_objective.winner_id == baseline.multi_objective.winner_id
        assert {
            key: (item.pareto_rank, item.dominated_by)
            for key, item in actual_audits.items()
        } == {
            key: (item.pareto_rank, item.dominated_by)
            for key, item in baseline_audits.items()
        }


def test_positive_affine_scale_changes_preserve_normalized_decision():
    original = [
        Candidate("A", 0.92, 20.0),
        Candidate("M", 0.89, 14.0),
        Candidate("L", 0.86, 10.0),
    ]
    transformed = [
        Candidate(item.id, 100.0 * item.accuracy + 7.0, 1000.0 * item.latency + 3.0)
        for item in original
    ]

    before = analyze_archive(original, config())
    after = analyze_archive(transformed, config())
    assert before.multi_objective.winner_id == after.multi_objective.winner_id
    for candidate_id in ("A", "M", "L"):
        left = audits_by_id(before)[candidate_id]
        right = audits_by_id(after)[candidate_id]
        assert left.normalized_accuracy_regret == pytest.approx(
            right.normalized_accuracy_regret
        )
        assert left.normalized_latency_regret == pytest.approx(
            right.normalized_latency_regret
        )


def test_dominated_finite_outlier_cannot_distort_front_normalization():
    front = [
        Candidate("A", 0.92, 20.0),
        Candidate("M", 0.89, 14.0),
        Candidate("L", 0.86, 10.0),
    ]
    baseline = analyze_archive(front, config())
    with_outlier = analyze_archive(
        [
            *front,
            Candidate("dominated_outlier", 0.10, 1.0e12),
        ],
        config(),
    )
    audits = audits_by_id(with_outlier)

    assert with_outlier.multi_objective.winner_id == (
        baseline.multi_objective.winner_id
    )
    assert with_outlier.normalization_bounds == baseline.normalization_bounds
    assert audits["dominated_outlier"].pareto_rank > 0
    assert audits["dominated_outlier"].dominated_by
    assert all(
        math.isfinite(value)
        for value in (
            audits["dominated_outlier"].normalized_accuracy_regret,
            audits["dominated_outlier"].normalized_latency_regret,
            audits["dominated_outlier"].compromise_score,
        )
    )


@pytest.mark.parametrize("count", [1, 3])
def test_identical_objectives_are_finite_and_do_not_divide_by_zero(count):
    analysis = analyze_archive(
        [Candidate(str(index), 0.90, 10.0) for index in range(count)],
        config(),
    )
    for audit in analysis.audits:
        assert audit.normalized_accuracy_regret == 0.0
        assert audit.normalized_latency_regret == 0.0
        assert math.isfinite(audit.compromise_score)
    assert analysis.normalization_bounds["accuracy"]["inactive"] is True
    assert analysis.normalization_bounds["latency"]["inactive"] is True
    assert analysis.multi_objective.winner_id is not None


@pytest.mark.parametrize(
    ("candidate", "reason_prefix"),
    [
        ({"id": "missing_accuracy", "status": "success",
          "specs": {}, "objective_values": {"latency": 10.0}},
         "missing_metric:accuracy"),
        ({"id": "missing_latency", "status": "success",
          "specs": {}, "objective_values": {"accuracy": 0.9}},
         "missing_metric:latency"),
        (Candidate("nan", float("nan"), 10.0), "invalid_metric:accuracy"),
        (Candidate("inf", 0.9, float("inf")), "invalid_metric:latency"),
        (Candidate("zero_latency", 0.9, 0.0), "invalid_metric:latency"),
        (Candidate("negative_latency", 0.9, -1.0), "invalid_metric:latency"),
        (Candidate("bool", True, 10.0), "invalid_metric:accuracy"),
        (Candidate("failed", 0.9, 10.0, status="failure"), "non_success_status"),
    ],
)
def test_missing_invalid_and_failed_measurements_are_rejected(
    candidate,
    reason_prefix,
):
    analysis = analyze_archive(
        [candidate, Candidate("valid", 0.8, 12.0)],
        config(),
    )
    invalid = next(item for item in analysis.audits if item.candidate_id != "valid")
    assert invalid.valid is False
    assert invalid.invalid_reason.startswith(reason_prefix)
    assert analysis.multi_objective.winner_id == "valid"


def test_all_invalid_archive_returns_no_winner():
    analysis = analyze_archive(
        [Candidate("failed", 0.9, 10.0, status="failure")],
        config(),
    )
    assert analysis.accuracy.status == "no_valid_candidates"
    assert analysis.latency.winner_id is None
    assert analysis.multi_objective.winner_id is None


def test_two_point_front_reports_documented_no_intermediate_fallback():
    analysis = analyze_archive(
        [Candidate("A", 0.92, 20.0), Candidate("L", 0.86, 10.0)],
        config(),
    )
    assert analysis.multi_objective.status == "selected"
    assert analysis.multi_objective.distinct_compromise is False
    assert analysis.multi_objective.fallback_used is True
    assert analysis.multi_objective.winner_id == "A"


def test_extreme_winner_is_not_mislabeled_as_distinct_when_middle_exists():
    analysis = analyze_archive(
        [
            Candidate(
                "accuracy_extreme",
                0.92,
                19.0,
                extra={
                    "latency_ci95_low": 18.0,
                    "latency_ci95_high": 20.0,
                },
            ),
            Candidate(
                "statistical_middle",
                0.92,
                20.0,
                extra={
                    "latency_ci95_low": 18.5,
                    "latency_ci95_high": 20.5,
                },
            ),
            Candidate("latency_extreme", 0.86, 10.0),
        ],
        config(),
    )

    assert analysis.multi_objective.winner_id == "accuracy_extreme"
    assert analysis.multi_objective.distinct_compromise is False
    assert analysis.multi_objective.fallback_used is False
    assert "not reported as a distinct compromise" in (
        analysis.multi_objective.reason
    )


def test_compromise_selector_never_returns_a_dominated_point():
    analysis = analyze_archive(
        [
            Candidate("dominated", 0.80, 20.0),
            Candidate("winner", 0.90, 10.0),
            Candidate("other", 0.95, 30.0),
        ],
        config(),
        accuracy_weight=1.0,
        latency_weight=1.0,
    )
    winner = audits_by_id(analysis)[analysis.multi_objective.winner_id]
    assert winner.pareto_rank == 0
    assert winner.dominated_by == ()


def test_accuracy_maximize_and_latency_minimize_directionality():
    analysis = analyze_archive(
        [
            Candidate("better_both", 0.9, 10.0),
            Candidate("worse_both", 0.8, 20.0),
        ],
        config(),
    )
    audits = audits_by_id(analysis)
    assert audits["better_both"].pareto_rank == 0
    assert audits["worse_both"].dominated_by == ("better_both",)
    assert audits["better_both"].normalized_accuracy_regret == 0.0
    assert audits["better_both"].normalized_latency_regret == 0.0


def test_mode_tie_breaking_is_deterministic_and_noise_aware():
    candidates = [
        Candidate(
            "higher_accuracy",
            0.900,
            10.03,
            extra={"latency_ci95_low": 9.97, "latency_ci95_high": 10.08},
        ),
        Candidate(
            "faster_median",
            0.895,
            10.00,
            extra={"latency_ci95_low": 9.96, "latency_ci95_high": 10.04},
        ),
    ]
    selection_config = config(
        accuracy_tolerance=0.01,
        latency_tolerance=0.05,
    )

    forward = analyze_archive(candidates, selection_config)
    reverse = analyze_archive(reversed(candidates), selection_config)
    assert forward.accuracy.winner_id == "higher_accuracy"
    assert forward.latency.winner_id == "higher_accuracy"
    assert forward.latency.latency_tied_candidate_ids == (
        "faster_median",
        "higher_accuracy",
    )
    assert forward.accuracy.winner_id == reverse.accuracy.winner_id
    assert forward.latency.winner_id == reverse.latency.winner_id


def test_latency_practical_tolerance_forms_tie_without_confidence_intervals():
    analysis = analyze_archive(
        [
            Candidate("raw_minimum", 0.88, 10.00),
            Candidate("higher_accuracy", 0.91, 10.50),
        ],
        config(latency_tolerance=0.60),
    )

    assert analysis.latency.latency_tied_candidate_ids == (
        "higher_accuracy",
        "raw_minimum",
    )
    assert analysis.latency.winner_id == "higher_accuracy"
    assert analysis.latency.reason == (
        "Highest-accuracy member of the equivalent-fastest cohort satisfying "
        "the accuracy-winner-relative constraint; deterministic specification "
        "fingerprint and candidate ID resolve remaining ties."
    )


def test_latency_direct_raw_fastest_reason_is_explicit():
    analysis = analyze_archive(
        [
            Candidate("accuracy_winner", 1.00, 20.0),
            Candidate("raw_fastest", 0.91, 10.0),
            Candidate("slower", 0.95, 12.0),
        ],
        config(
            constraint=AccuracyConstraint(kind="relative", value=0.90),
            latency_tolerance=0.50,
        ),
    )

    assert analysis.latency.winner_id == "raw_fastest"
    assert analysis.latency.latency_tied_candidate_ids == ("raw_fastest",)
    assert analysis.latency.reason == (
        "Lowest stabilized latency candidate satisfying the "
        "accuracy-winner-relative constraint; no equivalent-fastest "
        "tie-break was required."
    )


def test_latency_confidence_interval_overlap_cannot_expand_practical_cohort():
    analysis = analyze_archive(
        [
            Candidate(
                "raw_minimum",
                0.88,
                10.00,
                extra={
                    "latency_ci95_low": 9.90,
                    "latency_ci95_high": 10.20,
                },
            ),
            Candidate(
                "higher_accuracy",
                0.91,
                10.50,
                extra={
                    "latency_ci95_low": 10.10,
                    "latency_ci95_high": 10.60,
                },
            ),
        ],
        config(latency_tolerance=0.40),
    )

    assert analysis.latency.latency_tied_candidate_ids == ("raw_minimum",)
    assert analysis.latency.winner_id == "raw_minimum"
    assert analysis.latency.reason == (
        "Lowest stabilized latency candidate satisfying the "
        "accuracy-winner-relative constraint; no equivalent-fastest "
        "tie-break was required."
    )


def test_latency_tie_cohort_is_anchored_at_raw_minimum_not_chained():
    analysis = analyze_archive(
        [
            Candidate("raw_minimum", 0.88, 10.00),
            Candidate("within_anchor_tolerance", 0.90, 10.55),
            Candidate("only_within_chained_tolerance", 0.99, 11.10),
        ],
        config(latency_tolerance=0.60),
    )

    assert analysis.latency.latency_tied_candidate_ids == (
        "raw_minimum",
        "within_anchor_tolerance",
    )
    assert analysis.latency.winner_id == "within_anchor_tolerance"


def test_latency_exact_tie_uses_candidate_id_after_equal_fingerprint():
    shared_specs = {"model": {"depth": 6}}
    candidates = [
        Candidate("candidate_b", 0.90, 10.0, specs=shared_specs),
        Candidate("candidate_a", 0.90, 10.0, specs=shared_specs),
    ]

    forward = analyze_archive(candidates, config())
    reverse = analyze_archive(reversed(candidates), config())

    assert forward.latency.latency_tied_candidate_ids == (
        "candidate_a",
        "candidate_b",
    )
    assert forward.latency.winner_id == "candidate_a"
    assert reverse.latency.winner_id == "candidate_a"


def test_multi_objective_weights_do_not_influence_latency_selection():
    candidates = [
        Candidate("accuracy", 0.95, 20.0),
        Candidate("latency", 0.90, 10.0),
        Candidate("middle", 0.93, 14.0),
    ]
    selection_config = config(
        constraint=AccuracyConstraint(kind="relative", value=0.90),
    )

    accuracy_heavy = analyze_archive(
        candidates,
        selection_config,
        accuracy_weight=100.0,
        latency_weight=1.0,
    )
    latency_heavy = analyze_archive(
        reversed(candidates),
        selection_config,
        accuracy_weight=1.0,
        latency_weight=100.0,
    )

    assert accuracy_heavy.latency.winner_id == "latency"
    assert latency_heavy.latency.winner_id == "latency"
    assert (
        accuracy_heavy.latency.latency_tied_candidate_ids
        == latency_heavy.latency.latency_tied_candidate_ids
    )


def test_latency_90_percent_retention_is_relative_to_accuracy_winner():
    candidates = [
        Candidate("accuracy_winner", 0.66, 20.0),
        Candidate("on_relative_floor", 0.594, 12.0),
        Candidate("below_relative_floor", 0.593999, 9.0),
        Candidate("feasible_fastest", 0.60, 10.0),
    ]
    analysis = analyze_archive(
        candidates,
        config(
            constraint=AccuracyConstraint(kind="relative", value=0.90),
            accuracy_tolerance=1e-12,
        ),
    )
    audits = audits_by_id(analysis)

    assert analysis.accuracy.winner_id == "accuracy_winner"
    assert analysis.accuracy_reference_candidate_id == "accuracy_winner"
    assert analysis.accuracy_reference_value == pytest.approx(0.66)
    assert analysis.accuracy_threshold == pytest.approx(0.594)
    assert {
        candidate_id
        for candidate_id, audit in audits.items()
        if audit.accuracy_feasible
    } == {
        "accuracy_winner",
        "on_relative_floor",
        "feasible_fastest",
    }
    assert audits["below_relative_floor"].accuracy_feasible is False
    assert analysis.latency.winner_id == "feasible_fastest"


def test_accuracy_tie_break_never_reaches_outside_fastest_latency_cohort():
    analysis = analyze_archive(
        [
            Candidate("raw_fastest", 0.90, 10.0),
            Candidate("equivalent_higher_accuracy", 0.92, 10.5),
            Candidate("meaningfully_slower_best_accuracy", 1.00, 10.8),
        ],
        config(
            constraint=AccuracyConstraint(kind="relative", value=0.90),
            latency_tolerance=0.60,
        ),
    )

    assert analysis.latency.latency_tied_candidate_ids == (
        "equivalent_higher_accuracy",
        "raw_fastest",
    )
    assert analysis.latency.winner_id == "equivalent_higher_accuracy"


def test_overlapping_latency_intervals_do_not_make_slower_median_no_worse():
    analysis = analyze_archive(
        [
            Candidate(
                "higher_accuracy_slower_median",
                0.90,
                10.03,
                extra={
                    "latency_ci95_low": 9.97,
                    "latency_ci95_high": 10.08,
                },
            ),
            Candidate(
                "lower_accuracy_faster_median",
                0.89,
                10.00,
                extra={
                    "latency_ci95_low": 9.96,
                    "latency_ci95_high": 10.04,
                },
            ),
        ],
        config(),
    )
    audits = audits_by_id(analysis)

    assert audits["higher_accuracy_slower_median"].pareto_rank == 0
    assert audits["lower_accuracy_faster_median"].pareto_rank == 0
    assert all(not item.dominated_by for item in audits.values())


def test_latency_confidence_intervals_control_strict_dominance_only():
    overlapping = analyze_archive(
        [
            Candidate(
                "faster",
                0.90,
                10.00,
                extra={
                    "latency_ci95_low": 9.95,
                    "latency_ci95_high": 10.05,
                },
            ),
            Candidate(
                "slower",
                0.90,
                10.02,
                extra={
                    "latency_ci95_low": 9.97,
                    "latency_ci95_high": 10.07,
                },
            ),
        ],
        config(),
    )
    separated = analyze_archive(
        [
            Candidate(
                "faster",
                0.90,
                10.00,
                extra={
                    "latency_ci95_low": 9.98,
                    "latency_ci95_high": 10.01,
                },
            ),
            Candidate(
                "slower",
                0.90,
                10.10,
                extra={
                    "latency_ci95_low": 10.08,
                    "latency_ci95_high": 10.12,
                },
            ),
        ],
        config(),
    )

    overlapping_audits = audits_by_id(overlapping)
    separated_audits = audits_by_id(separated)
    assert overlapping_audits["slower"].pareto_rank == 0
    assert overlapping_audits["slower"].dominated_by == ()
    assert separated_audits["slower"].pareto_rank == 1
    assert separated_audits["slower"].dominated_by == ("faster",)


@pytest.mark.parametrize(
    "extra",
    [
        {"latency_ci95_low": 9.9},
        {"latency_ci95_high": 10.1},
        {"latency_ci95_low": 10.1, "latency_ci95_high": 10.2},
        {"latency_ci95_low": float("nan"), "latency_ci95_high": 10.1},
    ],
)
def test_incomplete_or_invalid_latency_confidence_intervals_are_rejected(extra):
    analysis = analyze_archive(
        [
            Candidate("invalid_ci", 0.90, 10.0, extra=extra),
            Candidate("valid", 0.80, 12.0),
        ],
        config(),
    )
    audit = audits_by_id(analysis)["invalid_ci"]

    assert audit.valid is False
    assert audit.invalid_reason is not None
    assert analysis.multi_objective.winner_id == "valid"


def test_latency_is_constrained_relative_to_accuracy_winner():
    analysis = analyze_archive(
        [
            Candidate("accuracy", 0.90, 20.0),
            Candidate("feasible_fast", 0.86, 11.0),
            Candidate("literal_fastest", 0.85, 9.0),
            Candidate("feasible_slow", 0.87, 12.0),
        ],
        config(
            constraint=AccuracyConstraint(kind="relative", value=0.95),
        ),
    )
    audits = audits_by_id(analysis)
    assert analysis.accuracy_reference_candidate_id == "accuracy"
    assert analysis.accuracy_threshold == pytest.approx(0.855)
    assert audits["feasible_fast"].accuracy_feasible is True
    assert audits["literal_fastest"].accuracy_feasible is False
    assert analysis.latency.winner_id == "feasible_fast"


def test_latency_retention_does_not_filter_multi_objective_front():
    analysis = analyze_archive(
        [
            Candidate("accuracy", 0.95, 20.0),
            Candidate("middle", 0.85, 14.0),
            Candidate("latency", 0.75, 10.0),
        ],
        config(
            constraint=AccuracyConstraint(kind="relative", value=0.90),
        ),
    )
    audits = audits_by_id(analysis)

    assert analysis.latency_accuracy_threshold == pytest.approx(0.855)
    assert analysis.latency.winner_id == "accuracy"
    assert analysis.multi_objective_accuracy_reference_candidate_id is None
    assert analysis.multi_objective_accuracy_reference_value is None
    assert analysis.multi_objective_accuracy_threshold is None
    assert audits["middle"].accuracy_feasible is False
    assert audits["middle"].multi_objective_accuracy_feasible is True
    assert audits["middle"].feasible_pareto_rank == 0
    assert analysis.multi_objective.winner_id == "middle"
    assert analysis.multi_objective.distinct_compromise is True


@pytest.mark.parametrize(
    "multi_objective_policy",
    [
        0.90,
        {"type": "absolute", "value": 0.90},
    ],
)
def test_optional_multi_objective_min_accuracy_filters_only_compromise_mode(
    multi_objective_policy,
):
    analysis = analyze_archive(
        [
            Candidate("accuracy", 0.95, 20.0),
            Candidate("middle", 0.85, 14.0),
            Candidate("latency", 0.75, 10.0),
        ],
        config(
            constraint=AccuracyConstraint(kind="relative", value=0.90),
            multi_objective_min_accuracy=multi_objective_policy,
        ),
    )
    audits = audits_by_id(analysis)

    assert analysis.config.multi_objective_min_accuracy.kind == "absolute"
    assert analysis.config.multi_objective_min_accuracy.value == pytest.approx(
        0.90
    )
    assert analysis.latency.winner_id == "accuracy"
    assert audits["middle"].accuracy_feasible is False
    assert audits["middle"].multi_objective_accuracy_feasible is False
    assert audits["middle"].pareto_rank == 0
    assert audits["middle"].feasible_pareto_rank is None
    assert analysis.multi_objective_accuracy_reference_candidate_id is None
    assert analysis.multi_objective_accuracy_reference_value is None
    assert analysis.multi_objective_accuracy_threshold == pytest.approx(0.90)
    assert analysis.multi_objective.winner_id == "accuracy"
    assert analysis.multi_objective.fallback_used is True
    assert analysis.multi_objective.reason.startswith(
        "No distinct Pareto compromise exists under the configured "
        "multi-objective eligibility policy"
    )


def test_relative_multi_objective_floor_is_resolved_from_accuracy_winner():
    analysis = analyze_archive(
        [
            Candidate("accuracy", 0.95, 22.0),
            Candidate("middle", 0.90, 15.0),
            Candidate("latency", 0.86, 10.0),
        ],
        config(
            constraint=AccuracyConstraint(kind="relative", value=0.98),
            multi_objective_min_accuracy={
                "type": "relative",
                "value": 0.90,
            },
        ),
    )

    assert analysis.latency_accuracy_threshold == pytest.approx(0.931)
    assert analysis.latency.winner_id == "accuracy"
    assert analysis.multi_objective_accuracy_reference_candidate_id == (
        "accuracy"
    )
    assert analysis.multi_objective_accuracy_reference_value == pytest.approx(
        0.95
    )
    assert analysis.multi_objective_accuracy_threshold == pytest.approx(0.855)
    assert analysis.multi_objective.winner_id == "middle"
    serialized = analysis.to_dict()["algorithm"]
    assert serialized["multi_objective_accuracy_reference_candidate_id"] == (
        "accuracy"
    )
    assert serialized["multi_objective_accuracy_reference_value"] == (
        pytest.approx(0.95)
    )
    assert serialized["multi_objective_accuracy_threshold"] == pytest.approx(
        0.855
    )
    assert serialized["configuration"]["multi_objective_min_accuracy"] == {
        "type": "relative",
        "value": 0.90,
        "reference": "accuracy_winner",
    }


@pytest.mark.parametrize(
    ("retained_fraction", "expected_eligible", "expected_winner"),
    [
        (0.90, {"accuracy", "middle", "latency"}, "middle"),
        (0.95, {"accuracy"}, "accuracy"),
        (0.98, {"accuracy"}, "accuracy"),
    ],
)
def test_relative_multi_objective_sensitivity_resolves_without_fitted_floor(
    retained_fraction,
    expected_eligible,
    expected_winner,
):
    analysis = analyze_archive(
        [
            Candidate("accuracy", 0.95, 22.0),
            Candidate("middle", 0.90, 15.0),
            Candidate("latency", 0.86, 10.0),
        ],
        config(
            multi_objective_min_accuracy={
                "type": "relative",
                "value": retained_fraction,
                "reference": "accuracy_winner",
            },
        ),
    )

    assert analysis.multi_objective_accuracy_reference_candidate_id == (
        "accuracy"
    )
    assert analysis.multi_objective_accuracy_reference_value == pytest.approx(
        0.95
    )
    assert analysis.multi_objective_accuracy_threshold == pytest.approx(
        0.95 * retained_fraction
    )
    assert {
        audit.candidate_id
        for audit in analysis.audits
        if audit.multi_objective_accuracy_feasible
    } == expected_eligible
    assert analysis.multi_objective.winner_id == expected_winner


def test_optional_multi_objective_floor_can_report_no_eligible_candidate():
    analysis = analyze_archive(
        [Candidate("best", 0.90, 10.0)],
        config(multi_objective_min_accuracy=0.91),
    )
    audit = audits_by_id(analysis)["best"]

    assert analysis.accuracy.winner_id == "best"
    assert analysis.latency.winner_id == "best"
    assert audit.multi_objective_accuracy_feasible is False
    assert analysis.multi_objective.status == (
        "no_multi_objective_accuracy_feasible_candidates"
    )
    assert analysis.multi_objective.winner_id is None
    assert "minimum accuracy 0.91" in analysis.multi_objective.reason


def test_multi_objective_dominance_is_independent_of_latency_retention():
    analysis = analyze_archive(
        [
            Candidate("accuracy", 0.92, 20.0),
            Candidate("latency", 0.84, 10.0),
            Candidate("dominated", 0.83, 11.0),
        ],
        config(
            constraint=AccuracyConstraint(kind="relative", value=0.95),
        ),
    )
    audits = audits_by_id(analysis)

    assert analysis.latency.winner_id == "accuracy"
    assert audits["latency"].accuracy_feasible is False
    assert audits["latency"].multi_objective_accuracy_feasible is True
    assert audits["dominated"].feasible_pareto_rank == 1
    assert audits["dominated"].feasible_dominated_by == ("latency",)
    assert analysis.multi_objective.winner_id != "dominated"


def test_separate_constraints_are_order_invariant():
    candidates = [
        Candidate("accuracy", 0.95, 22.0),
        Candidate("middle", 0.88, 15.0),
        Candidate("latency", 0.84, 10.0),
        Candidate("dominated", 0.83, 11.0),
    ]
    selection_config = config(
        constraint=AccuracyConstraint(kind="relative", value=0.90),
        multi_objective_min_accuracy=0.82,
    )

    forward = analyze_archive(candidates, selection_config)
    reverse = analyze_archive(reversed(candidates), selection_config)
    assert (
        forward.accuracy.winner_id,
        forward.latency.winner_id,
        forward.multi_objective.winner_id,
    ) == (
        reverse.accuracy.winner_id,
        reverse.latency.winner_id,
        reverse.multi_objective.winner_id,
    )
    assert {
        item.candidate_id: (
            item.accuracy_feasible,
            item.multi_objective_accuracy_feasible,
            item.feasible_pareto_rank,
            item.feasible_dominated_by,
        )
        for item in forward.audits
    } == {
        item.candidate_id: (
            item.accuracy_feasible,
            item.multi_objective_accuracy_feasible,
            item.feasible_pareto_rank,
            item.feasible_dominated_by,
        )
        for item in reverse.audits
    }


def test_objective_config_parses_separate_constraint_settings():
    objective_config = parse_objective_config({
        "objectives": [
            {"metric": "accuracy", "direction": "maximize"},
            {"metric": "latency", "direction": "minimize"},
        ],
        "latency_accuracy_retention": {
            "type": "absolute",
            "max_absolute_degradation": 0.03,
        },
        "multi_objective_min_accuracy": {
            "type": "relative",
            "value": 0.90,
        },
    })
    selection = objective_config.selection_config

    assert selection is not None
    assert selection.latency_accuracy_retention.kind == "absolute"
    assert selection.latency_accuracy_retention.value == pytest.approx(0.03)
    assert selection.multi_objective_min_accuracy.kind == "relative"
    assert selection.multi_objective_min_accuracy.value == pytest.approx(0.90)
    assert selection.multi_objective_min_accuracy.reference == "accuracy_winner"
    serialized = selection.to_dict()
    assert serialized["latency_accuracy_retention"]["type"] == "absolute"
    assert serialized["multi_objective_min_accuracy"] == {
        "type": "relative",
        "value": 0.90,
        "reference": "accuracy_winner",
    }
    assert "accuracy_constraint" not in serialized


def test_legacy_accuracy_constraint_maps_to_latency_only():
    objective_config = parse_objective_config({
        "objectives": [
            {"metric": "accuracy", "direction": "maximize"},
            {"metric": "latency", "direction": "minimize"},
        ],
        "accuracy_retention_fraction": 0.90,
    })
    selection = objective_config.selection_config

    assert selection is not None
    assert selection.latency_accuracy_retention.value == pytest.approx(0.90)
    assert selection.multi_objective_min_accuracy is None


def test_new_and_legacy_latency_constraints_cannot_conflict():
    with pytest.raises(ValueError, match="not both"):
        parse_objective_config({
            "objectives": [
                {"metric": "accuracy", "direction": "maximize"},
                {"metric": "latency", "direction": "minimize"},
            ],
            "latency_accuracy_retention": 0.95,
            "accuracy_retention_fraction": 0.90,
        })


@pytest.mark.parametrize(
    "settings",
    [
        {
            "accuracy_constraint": {"retained_fraction": 0.98},
            "accuracy_retention_fraction": 0.90,
        },
        {
            "accuracy_constraint": {"max_absolute_degradation": 0.02},
            "max_accuracy_degradation": 0.03,
        },
        {
            "latency_accuracy_retention": {
                "type": "relative",
                "value": 0.90,
                "retained_fraction": 0.95,
            },
        },
        {
            "latency_accuracy_retention": {
                "retained_fraction": 0.90,
                "min_retained_fraction": 0.95,
            },
        },
    ],
)
def test_conflicting_latency_constraint_representations_are_rejected(settings):
    with pytest.raises(ValueError, match="only one"):
        parse_objective_config({
            "objectives": [
                {"metric": "accuracy", "direction": "maximize"},
                {"metric": "latency", "direction": "minimize"},
            ],
            **settings,
        })


@pytest.mark.parametrize(
    ("value", "exception", "message"),
    [
        (0.0, ValueError, r"must be in \(0, 1\]"),
        (-0.1, ValueError, r"must be in \(0, 1\]"),
        (1.000001, ValueError, r"must be in \(0, 1\]"),
        (float("nan"), ValueError, "finite"),
        (float("inf"), ValueError, "finite"),
        (True, TypeError, "number or dictionary"),
    ],
)
def test_latency_retention_configuration_rejects_invalid_values(
    value,
    exception,
    message,
):
    with pytest.raises(exception, match=message):
        parse_objective_config({
            "objectives": [
                {"metric": "accuracy", "direction": "maximize"},
                {"metric": "latency", "direction": "minimize"},
            ],
            "latency_accuracy_retention": value,
        })


def test_explicit_90_percent_profile_does_not_change_global_default():
    objectives = [
        {"metric": "accuracy", "direction": "maximize"},
        {"metric": "latency", "direction": "minimize"},
    ]
    default_selection = parse_objective_config({
        "objectives": objectives,
    }).selection_config
    profile_selection = parse_objective_config({
        "objectives": objectives,
        "latency_accuracy_retention": {
            "type": "relative",
            "retained_fraction": 0.90,
            "reference": "accuracy_winner",
        },
        "multi_objective_min_accuracy": None,
    }).selection_config

    assert default_selection is not None
    assert profile_selection is not None
    assert default_selection.latency_accuracy_retention.value == pytest.approx(
        0.98
    )
    assert profile_selection.latency_accuracy_retention.value == pytest.approx(
        0.90
    )
    assert profile_selection.multi_objective_min_accuracy is None


def test_absolute_accuracy_degradation_rule_and_inclusive_floor():
    analysis = analyze_archive(
        [
            Candidate("accuracy", 0.90, 20.0),
            Candidate("on_floor", 0.88, 11.0),
            Candidate("below", 0.879999, 9.0),
        ],
        config(
            constraint=AccuracyConstraint(kind="absolute", value=0.02),
        ),
    )
    audits = audits_by_id(analysis)
    assert analysis.accuracy_threshold == pytest.approx(0.88)
    assert audits["on_floor"].accuracy_feasible is True
    assert audits["below"].accuracy_feasible is False
    assert analysis.latency.winner_id == "on_floor"


def test_external_latency_reference_does_not_block_multi_objective_selection():
    analysis = analyze_archive(
        [Candidate("best_observed", 0.90, 10.0)],
        config(
            constraint=AccuracyConstraint(
                kind="relative",
                value=1.0,
                reference_value=0.95,
                reference_candidate_id="external_accuracy_anchor",
            ),
        ),
    )
    assert analysis.accuracy.winner_id == "best_observed"
    assert analysis.latency.status == "no_accuracy_feasible_candidates"
    assert analysis.latency.winner_id is None
    assert analysis.multi_objective.status == "selected"
    assert analysis.multi_objective.winner_id == "best_observed"


@settings(max_examples=100, deadline=None, derandomize=True, database=None)
@given(
    st.lists(
        st.tuples(
            st.integers(min_value=-1000, max_value=1000),
            st.integers(min_value=1, max_value=1000),
        ),
        min_size=1,
        max_size=20,
    )
)
def test_property_winner_is_nondominated_permutation_and_scale_invariant(points):
    candidates = [
        Candidate(str(index), accuracy, latency)
        for index, (accuracy, latency) in enumerate(points)
    ]
    selection_config = config()
    analysis = analyze_archive(candidates, selection_config)
    reversed_analysis = analyze_archive(reversed(candidates), selection_config)
    transformed = [
        Candidate(
            item.id,
            10.0 * float(item.accuracy) + 7.0,
            1000.0 * float(item.latency) + 3.0,
        )
        for item in candidates
    ]
    transformed_analysis = analyze_archive(transformed, selection_config)

    assert analysis.multi_objective.winner_id is not None
    assert (
        analysis.multi_objective.winner_id
        == reversed_analysis.multi_objective.winner_id
        == transformed_analysis.multi_objective.winner_id
    )
    winner = audits_by_id(analysis)[analysis.multi_objective.winner_id]
    assert winner.feasible_pareto_rank == 0
    assert winner.feasible_dominated_by == ()
    assert math.isfinite(winner.compromise_score)
