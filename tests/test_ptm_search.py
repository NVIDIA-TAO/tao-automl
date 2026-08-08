# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hierarchical, non-ordinal PTM-arm search tests."""

import copy

import pytest

from tao_automl.ptm_search import (
    HierarchicalPTMPolicy,
    HierarchicalPTMScheduler,
    PTMArm,
    PTMArmObservation,
)
from tao_automl.recommendation_audit import (
    ALGORITHMIC_CAMPAIGN_FLAGS,
    canonical_audit_sha256,
)


def _arm(name, marker):
    return PTMArm(
        checkpoint_id=name,
        conditional_search_space_sha256=marker * 64,
        preflight_provenance_sha256=chr(ord(marker) + 1) * 64,
        input_contract_sha256=chr(ord(marker) + 2) * 64,
    )


ARMS = (
    _arm("dino.resnet50", "1"),
    _arm("dino.fan_small", "4"),
    _arm("dino.nvdinov2", "7"),
)


def _observation(
    identifier,
    arm,
    accuracy,
    latency,
    *,
    status="success",
    fidelity=1.0,
):
    return PTMArmObservation(
        candidate_id=str(identifier),
        checkpoint_id=arm,
        status=status,
        accuracy=accuracy,
        latency=latency,
        fidelity=fidelity,
    )


def test_initial_design_is_balanced_and_arm_input_order_invariant():
    policy = HierarchicalPTMPolicy(
        mode="accuracy",
        initial_issues_per_arm=2,
    )
    first = HierarchicalPTMScheduler(ARMS, policy, random_seed=314159)
    second = HierarchicalPTMScheduler(
        tuple(reversed(ARMS)),
        policy,
        random_seed=314159,
    )

    sequence_a = [first.choose_arm([]).checkpoint_id for _ in range(6)]
    sequence_b = [second.choose_arm([]).checkpoint_id for _ in range(6)]

    assert sequence_a == sequence_b
    assert {arm: sequence_a.count(arm) for arm in sequence_a} == {
        arm.checkpoint_id: 2 for arm in ARMS
    }


def test_latency_outer_policy_rejects_degenerate_fast_arm():
    policy = HierarchicalPTMPolicy(
        mode="latency",
        initial_issues_per_arm=0,
        invalid_recovery_issues_per_arm=0,
        exploration_strength=0.0,
        latency_accuracy_retention=0.90,
    )
    scheduler = HierarchicalPTMScheduler(ARMS, policy, random_seed=1)
    observations = [
        _observation("accurate", "dino.resnet50", 0.90, 20.0),
        _observation("viable-fast", "dino.fan_small", 0.85, 10.0),
        _observation("degenerate", "dino.nvdinov2", 0.20, 1.0),
    ]

    decision = scheduler.choose_arm(observations)

    assert decision.accuracy_reference == pytest.approx(0.90)
    assert decision.accuracy_threshold == pytest.approx(0.81)
    assert decision.checkpoint_id == "dino.fan_small"
    assert (
        decision.arm_scores["dino.nvdinov2"]
        < decision.arm_scores["dino.resnet50"]
    )


def test_latency_outer_policy_uses_tolerance_boundary_order_invariant():
    policy = HierarchicalPTMPolicy(
        mode="latency",
        initial_issues_per_arm=0,
        invalid_recovery_issues_per_arm=0,
        exploration_strength=0.0,
        latency_accuracy_retention=0.90,
        accuracy_tolerance=0.02,
    )
    observations = [
        _observation("accurate", "dino.resnet50", 0.90, 20.0),
        # Raw threshold is 0.81; effective boundary is 0.79.
        _observation("relaxed-fast", "dino.fan_small", 0.80, 10.0),
        _observation("infeasible", "dino.nvdinov2", 0.78, 1.0),
    ]
    forward = HierarchicalPTMScheduler(ARMS, policy, random_seed=11)
    reverse = HierarchicalPTMScheduler(
        tuple(reversed(ARMS)),
        policy,
        random_seed=11,
    )

    first = forward.choose_arm(observations)
    second = reverse.choose_arm(list(reversed(observations)))

    assert first.checkpoint_id == second.checkpoint_id == "dino.fan_small"
    assert first.accuracy_threshold == pytest.approx(0.79)
    assert first.arm_scores == second.arm_scores
    assert first.decision_sha256 == second.decision_sha256


def test_latency_without_positive_reference_remains_in_quality_discovery():
    arms = ARMS[:2]
    scheduler = HierarchicalPTMScheduler(
        arms,
        HierarchicalPTMPolicy(
            mode="latency",
            initial_issues_per_arm=0,
            invalid_recovery_issues_per_arm=0,
            exploration_strength=0.0,
        ),
        random_seed=2,
    )
    decision = scheduler.choose_arm(
        [
            _observation("zero", "dino.resnet50", 0.0, 1.0),
            _observation("negative", "dino.fan_small", -1.0, 0.5),
        ]
    )

    assert decision.accuracy_reference == 0.0
    assert decision.accuracy_threshold is None
    assert decision.checkpoint_id == "dino.resnet50"


def test_failed_initial_arm_uses_only_preregistered_recovery_allowance():
    arms = ARMS[:2]
    scheduler = HierarchicalPTMScheduler(
        arms,
        HierarchicalPTMPolicy(
            mode="accuracy",
            initial_issues_per_arm=1,
            invalid_recovery_issues_per_arm=1,
            exploration_strength=0.0,
        ),
        random_seed=7,
    )
    first = scheduler.choose_arm([])
    second = scheduler.choose_arm([])
    assert {first.checkpoint_id, second.checkpoint_id} == {
        arm.checkpoint_id for arm in arms
    }

    failed = [
        _observation(
            "failure",
            first.checkpoint_id,
            None,
            None,
            status="failure",
        ),
        _observation("valid", second.checkpoint_id, 0.7, 10.0),
    ]
    recovery = scheduler.choose_arm(failed)
    assert recovery.checkpoint_id == first.checkpoint_id
    assert recovery.stage == "preregistered_invalid_recovery"

    # The failed record is preserved, but the arm cannot consume an unlimited
    # series of replacement recommendations.
    next_decision = scheduler.choose_arm(failed)
    assert next_decision.stage == "mode_aware_outer_allocation"


def test_partial_valid_calibration_uses_bounded_recovery_before_modelling():
    arms = ARMS[:2]
    scheduler = HierarchicalPTMScheduler(
        arms,
        HierarchicalPTMPolicy(
            mode="accuracy",
            initial_issues_per_arm=2,
            invalid_recovery_issues_per_arm=1,
            exploration_strength=0.0,
        ),
        random_seed=13,
    )
    initial = [scheduler.choose_arm([]).checkpoint_id for _ in range(4)]
    assert {arm: initial.count(arm) for arm in initial} == {
        item.checkpoint_id: 2 for item in arms
    }

    under_calibrated = arms[0].checkpoint_id
    calibrated = arms[1].checkpoint_id
    observations = [
        _observation("valid-a", under_calibrated, 0.6, 11.0),
        _observation(
            "failed-a",
            under_calibrated,
            None,
            None,
            status="failure",
        ),
        _observation("valid-b-1", calibrated, 0.7, 10.0),
        _observation("valid-b-2", calibrated, 0.8, 9.0),
    ]

    recovery = scheduler.choose_arm(observations)
    assert recovery.checkpoint_id == under_calibrated
    assert recovery.stage == "preregistered_invalid_recovery"
    assert recovery.valid_observation_counts[under_calibrated] == 1

    # Failed recommendations remain counted as issued. Once the single frozen
    # recovery allowance is consumed, an unchanged archive cannot trigger an
    # unbounded sequence of replacements.
    exhausted = scheduler.choose_arm(observations)
    assert exhausted.stage == "mode_aware_outer_allocation"
    assert scheduler.issued_counts[under_calibrated] == 3


def test_resume_replays_next_arm_exactly():
    policy = HierarchicalPTMPolicy(mode="multi_objective")
    uninterrupted = HierarchicalPTMScheduler(
        ARMS,
        policy,
        random_seed=271828,
    )
    for _ in range(4):
        uninterrupted.choose_arm([])
    state = copy.deepcopy(uninterrupted.state_dict())

    restored = HierarchicalPTMScheduler.from_state_dict(
        arms=tuple(reversed(ARMS)),
        policy=policy,
        random_seed=271828,
        state=state,
    )

    assert restored.state_dict() == state
    assert restored.choose_arm([]).to_dict() == uninterrupted.choose_arm([]).to_dict()


def test_resume_accepts_legacy_implicit_default_accuracy_tolerance():
    policy = HierarchicalPTMPolicy(mode="multi_objective")
    scheduler = HierarchicalPTMScheduler(
        ARMS,
        policy,
        random_seed=271828,
    )
    scheduler.choose_arm([])
    state = copy.deepcopy(scheduler.state_dict())
    state["signature"]["policy"].pop("accuracy_tolerance")
    state["signature_sha256"] = canonical_audit_sha256(state["signature"])

    restored = HierarchicalPTMScheduler.from_state_dict(
        arms=ARMS,
        policy=policy,
        random_seed=271828,
        state=state,
    )

    assert restored.policy.accuracy_tolerance == pytest.approx(1e-12)
    assert restored.issued_counts == scheduler.issued_counts


def test_resume_rejects_legacy_state_for_nondefault_accuracy_tolerance():
    policy = HierarchicalPTMPolicy(
        mode="latency",
        accuracy_tolerance=0.01,
    )
    scheduler = HierarchicalPTMScheduler(
        ARMS,
        policy,
        random_seed=271828,
    )
    state = copy.deepcopy(scheduler.state_dict())
    state["signature"]["policy"].pop("accuracy_tolerance")
    state["signature_sha256"] = canonical_audit_sha256(state["signature"])

    with pytest.raises(ValueError, match="different arm inventory, policy"):
        HierarchicalPTMScheduler.from_state_dict(
            arms=ARMS,
            policy=policy,
            random_seed=271828,
            state=state,
        )


def test_first_model_based_moo_arm_decision_uses_balanced_parego_weight():
    arms = ARMS[:2]
    scheduler = HierarchicalPTMScheduler(
        arms,
        HierarchicalPTMPolicy(
            mode="multi_objective",
            initial_issues_per_arm=1,
            invalid_recovery_issues_per_arm=0,
            exploration_strength=0.0,
        ),
        random_seed=9,
    )
    scheduler.choose_arm([])
    scheduler.choose_arm([])
    decision = scheduler.choose_arm(
        [
            _observation("a", arms[0].checkpoint_id, 0.9, 20.0),
            _observation("b", arms[1].checkpoint_id, 0.7, 10.0),
        ]
    )

    assert decision.model_based_decision_index == 0
    assert decision.parego["iteration"] == 0
    assert decision.parego["weights"] == {
        "accuracy": 0.5,
        "latency": 0.5,
    }


def test_mixed_fidelity_is_rejected_without_a_frozen_comparison_level():
    scheduler = HierarchicalPTMScheduler(
        ARMS[:2],
        HierarchicalPTMPolicy(
            mode="accuracy",
            initial_issues_per_arm=0,
            invalid_recovery_issues_per_arm=0,
        ),
        random_seed=3,
    )
    with pytest.raises(ValueError, match="mixed fidelities"):
        scheduler.choose_arm(
            [
                _observation("low", "dino.resnet50", 0.5, 10.0, fidelity=3),
                _observation("high", "dino.fan_small", 0.6, 12.0, fidelity=12),
            ]
        )


def test_required_fidelity_filters_lower_budget_observations():
    scheduler = HierarchicalPTMScheduler(
        ARMS[:2],
        HierarchicalPTMPolicy(
            mode="accuracy",
            initial_issues_per_arm=0,
            invalid_recovery_issues_per_arm=0,
            exploration_strength=0.0,
            required_fidelity=12.0,
        ),
        random_seed=3,
    )
    decision = scheduler.choose_arm(
        [
            _observation("low", "dino.resnet50", 0.99, 10.0, fidelity=3),
            _observation("full-a", "dino.resnet50", 0.5, 10.0, fidelity=12),
            _observation("full-b", "dino.fan_small", 0.6, 12.0, fidelity=12),
        ]
    )
    assert decision.checkpoint_id == "dino.fan_small"


def test_unknown_arms_and_duplicate_candidates_fail_closed():
    scheduler = HierarchicalPTMScheduler(
        ARMS[:2],
        HierarchicalPTMPolicy(mode="accuracy", initial_issues_per_arm=0),
        random_seed=4,
    )
    with pytest.raises(ValueError, match="unknown arm"):
        scheduler.choose_arm(
            [_observation("x", "unknown.ptm", 0.5, 10.0)]
        )
    duplicate = _observation("same", "dino.resnet50", 0.5, 10.0)
    with pytest.raises(ValueError, match="duplicate candidate"):
        scheduler.choose_arm([duplicate, duplicate])
    with pytest.raises(ValueError, match="duplicate global candidate"):
        scheduler.choose_arm(
            [
                duplicate,
                _observation("same", "dino.fan_small", 0.6, 11.0),
            ]
        )


def test_outer_decision_is_content_addressed_and_agent_free():
    scheduler = HierarchicalPTMScheduler(
        ARMS[:1],
        HierarchicalPTMPolicy(mode="multi_objective"),
        random_seed=5,
    )
    decision = scheduler.choose_arm([])
    payload = decision.to_dict()

    assert payload["algorithmic_campaign_flags"] == ALGORITHMIC_CAMPAIGN_FLAGS
    expected = payload.pop("decision_sha256")
    assert expected == canonical_audit_sha256(payload)
