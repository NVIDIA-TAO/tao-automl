# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Production contract tests for the hierarchical PTM Bayesian wrapper."""

from __future__ import annotations

import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tao_automl.brain.bayesian import Bayesian
from tao_automl.brain.hierarchical_ptm import HierarchicalPTMBrain
from tao_automl.controller.controller import Controller
from tao_automl.objectives import parse_objective_config
from tao_automl.ptm_search import (
    HierarchicalPTMPolicy,
    HierarchicalPTMScheduler,
    PTMArm,
)
from tao_automl.recommendation_audit import (
    ALGORITHMIC_CAMPAIGN_FLAGS,
    build_recommendation_audit,
    canonical_audit_sha256,
    visible_history_snapshot,
)
from tao_automl.state.state_store import StateStore as FileStateStore
from tao_automl.types import Recommendation


def _parameter(name, *, minimum=0.0, maximum=1.0):
    return {
        "parameter": name,
        "value_type": "float",
        "default_value": (minimum + maximum) / 2,
        "valid_min": minimum,
        "valid_max": maximum,
        "valid_options": [],
        "option_weights": None,
        "math_cond": None,
        "parent_param": None,
        "depends_on": None,
    }


class _StateStore:
    def __init__(self):
        self.brain_state = None
        self.save_count = 0
        self.controller_state = None
        self.best_state = None

    def get_job_specs(self, _job_id):
        return {"train": {"num_epochs": 1}}

    def get_custom_param_ranges(self, _experiment_id):
        return {}

    def get_brain_info(self, _job_id):
        return copy.deepcopy(self.brain_state)

    def save_brain_info(self, _job_id, state):
        self.brain_state = copy.deepcopy(state)
        self.save_count += 1

    def save_controller_info(self, _job_id, state):
        self.controller_state = copy.deepcopy(state)

    def get_controller_info(self, _job_id):
        return copy.deepcopy(self.controller_state)

    def save_best_rec_info(self, _job_id, rec_number, rec_data):
        self.best_state = {
            "rec_number": rec_number,
            "rec_data": copy.deepcopy(rec_data),
        }

    def lock(self):
        return nullcontext()


ARMS = (
    PTMArm(
        checkpoint_id="dino.a",
        conditional_search_space_sha256="1" * 64,
        preflight_provenance_sha256="2" * 64,
        input_contract_sha256="3" * 64,
    ),
    PTMArm(
        checkpoint_id="dino.b",
        conditional_search_space_sha256="4" * 64,
        preflight_provenance_sha256="5" * 64,
        input_contract_sha256="6" * 64,
    ),
)


def _objective_config(mode):
    return parse_objective_config(
        {
            "objectives": [
                {"metric": "mAP50", "direction": "maximize"},
                {"metric": "latency_ms", "direction": "minimize"},
            ],
            "selection_mode": mode,
            "accuracy_metric": "mAP50",
            "latency_metric": "latency_ms",
            "latency_accuracy_retention": 0.9,
        }
    )


def _make_brain(
    *,
    store=None,
    mode="multi_objective",
    resume=False,
    arms=ARMS,
    policy_kwargs=None,
    candidate_overrides=None,
    checkpoint_targets=None,
    parameters=None,
):
    store = store or _StateStore()
    context = SimpleNamespace(
        id="hierarchical-session",
        handler_id="hierarchical-session",
        random_seed=271828,
    )
    config = _objective_config(mode)
    configured_parameters = parameters or {
        "dino.a": [_parameter("model.width", minimum=0.25, maximum=1.0)],
        "dino.b": [_parameter("model.depth", minimum=1.0, maximum=6.0)],
    }
    inner = {
        arm.checkpoint_id: Bayesian(
            context,
            store,
            "dino",
            configured_parameters[arm.checkpoint_id],
            metric="multi_objective_score",
            direction="maximize",
            objective_config=config,
            acquisition_settings={
                "calibration_points": 2,
                "xi": 0.01,
                "augmentation_rho": 1e-6,
            },
        )
        for arm in arms
    }
    scheduler = HierarchicalPTMScheduler(
        tuple(reversed(arms)),
        HierarchicalPTMPolicy(
            mode=mode,
            initial_issues_per_arm=1,
            invalid_recovery_issues_per_arm=1,
            exploration_strength=0.0,
            **(policy_kwargs or {}),
        ),
        random_seed=314159,
    )
    overrides = candidate_overrides or {
        "dino.a": {
            "train.pretrained_model_path": "/resolved/a.pth",
            "model": {"width": 0.125, "backbone": "a"},
        },
        "dino.b": {
            "train": {"pretrained_model_path": "/resolved/b.pth"},
            "model": {"depth": 8.0, "backbone": "b"},
        },
    }
    targets = checkpoint_targets or {
        arm.checkpoint_id: "train.pretrained_model_path" for arm in arms
    }
    kwargs = {
        "context": context,
        "state_store": store,
        "scheduler": scheduler,
        "inner_brains": inner,
        "candidate_overrides": overrides,
        "checkpoint_targets": targets,
        "accuracy_metric": "mAP50",
        "latency_metric": "latency_ms",
    }
    if resume:
        return HierarchicalPTMBrain.load_state(**kwargs), store, config
    return HierarchicalPTMBrain(**kwargs), store, config


def _issued_recommendation(
    wrapper,
    config,
    history,
    identifier,
    *,
    accuracy,
    latency,
    status="success",
):
    raw = wrapper.generate_recommendations(history)
    assert len(raw) == 1
    proposal = wrapper.consume_last_recommendation_audits()
    assert len(proposal) == 1
    recommendation = Recommendation(identifier, raw[0], "multi_objective_score")
    recommendation.recommendation_audit = build_recommendation_audit(
        candidate_id=identifier,
        specs=raw[0],
        algorithm="bayesian",
        search_seed=wrapper.random_seed,
        search_space=wrapper.parameters,
        custom_ranges=wrapper.custom_ranges,
        objective_config=config,
        visible_history=visible_history_snapshot(history),
        acquisition={
            "proposal": proposal[0],
            "algorithm_capability": wrapper.algorithm_capability,
            "objective_mode_capability": wrapper.objective_mode_capability,
        },
    )
    recommendation.objective_values = {
        "mAP50": accuracy,
        "latency_ms": latency,
    }
    recommendation.result = accuracy
    recommendation.objective_score = accuracy
    recommendation.status = status
    return recommendation, proposal[0]


def test_ptm_identity_is_outer_categorical_and_override_merge_is_candidate_last():
    wrapper, _, _ = _make_brain()

    recommendation = wrapper.generate_recommendations([])[0]
    audit = wrapper.consume_last_recommendation_audits()[0]
    arm_id = audit["ptm"]["arm_id"]
    target = audit["ptm"]["checkpoint_target"]

    assert audit["checkpoint_identity_is_ordinal"] is False
    assert audit["algorithmic_campaign_flags"] == ALGORITHMIC_CAMPAIGN_FLAGS
    assert audit["ptm"]["outer_decision"]["checkpoint_id"] == arm_id
    assert audit["ptm"]["inner_acquisition"]["normalized_suggestion"]
    assert audit["ptm"]["conditional_search_space_sha256"] == (
        wrapper._arms[arm_id].conditional_search_space_sha256
    )
    assert target == "train.pretrained_model_path"
    assert recommendation["train"]["pretrained_model_path"].endswith(
        f"{arm_id[-1]}.pth"
    )
    searched_name = wrapper.inner_brains[arm_id].parameters[0]["parameter"]
    top, leaf = searched_name.split(".")
    assert recommendation[top][leaf] != (
        wrapper._candidate_overrides[arm_id][top][leaf]
    )
    assert len(wrapper.inner_brains[arm_id].Xs[-1]) == 1
    assert all(
        parameter["parameter"] != target
        for parameter in wrapper.inner_brains[arm_id].parameters
    )
    assert audit["emitted_specs_sha256"] == canonical_audit_sha256(
        recommendation
    )


def test_history_is_partitioned_only_by_signed_arm_audit():
    wrapper, _, config = _make_brain()
    history = []
    first, first_audit = _issued_recommendation(
        wrapper,
        config,
        history,
        0,
        accuracy=0.7,
        latency=12.0,
    )
    history.append(first)
    second, second_audit = _issued_recommendation(
        wrapper,
        config,
        history,
        1,
        accuracy=0.6,
        latency=9.0,
    )
    history.append(second)
    assert first_audit["ptm"]["arm_id"] != second_audit["ptm"]["arm_id"]

    # Runtime spec mutation cannot reassign an observation to another arm.
    first.specs["train"]["pretrained_model_path"] = (
        second.specs["train"]["pretrained_model_path"]
    )
    _, third_audit = _issued_recommendation(
        wrapper,
        config,
        history,
        2,
        accuracy=0.65,
        latency=10.0,
    )
    selected_arm = third_audit["ptm"]["arm_id"]
    expected_ids = [
        str(item.id)
        for item in history
        if item.recommendation_audit["acquisition"]["proposal"]["ptm"][
            "arm_id"
        ]
        == selected_arm
    ]
    assert third_audit["ptm"]["inner_acquisition"][
        "observation_summary"
    ]["candidate_ids"] == expected_ids


def test_tampered_or_missing_arm_audit_fails_closed():
    wrapper, _, config = _make_brain()
    recommendation, _ = _issued_recommendation(
        wrapper,
        config,
        [],
        0,
        accuracy=0.7,
        latency=10.0,
    )
    ptm_audit = recommendation.recommendation_audit["acquisition"]["proposal"][
        "ptm"
    ]
    ptm_audit["arm_id"] = (
        "dino.a" if ptm_audit["arm_id"] == "dino.b" else "dino.b"
    )
    with pytest.raises(ValueError, match="integrity verification failed"):
        wrapper.generate_recommendations([recommendation])

    recommendation.recommendation_audit = {}
    with pytest.raises(ValueError, match="schema version"):
        wrapper.generate_recommendations([recommendation])


def test_checkpoint_target_cannot_be_a_search_variable_or_generated_value():
    parameters = {
        "dino.a": [_parameter("train.pretrained_model_path")],
        "dino.b": [_parameter("model.depth")],
    }
    with pytest.raises(ValueError, match="overlaps searchable parameter"):
        _make_brain(parameters=parameters)


def test_exact_resume_replays_next_arm_and_inner_acquisition():
    wrapper, store, config = _make_brain()
    history = []
    for identifier, values in enumerate(((0.7, 12.0), (0.6, 9.0))):
        recommendation, _ = _issued_recommendation(
            wrapper,
            config,
            history,
            identifier,
            accuracy=values[0],
            latency=values[1],
        )
        history.append(recommendation)
    wrapper.save_state()

    restored, _, _ = _make_brain(store=store, resume=True)
    uninterrupted_raw = wrapper.generate_recommendations(history)
    uninterrupted_audit = wrapper.consume_last_recommendation_audits()
    restored_raw = restored.generate_recommendations(list(reversed(history)))
    restored_audit = restored.consume_last_recommendation_audits()

    assert restored_raw == uninterrupted_raw
    assert restored_audit == uninterrupted_audit
    assert restored.scheduler.state_dict() == wrapper.scheduler.state_dict()


def test_unbounded_schema_metadata_persists_as_strict_json_and_resumes(
    tmp_path,
):
    parameter = _parameter(
        "model.num_select",
        minimum=1.0,
        maximum=100.0,
    )
    parameter["valid_max"] = float("inf")
    parameters = {
        "dino.a": [parameter],
        "dino.b": [_parameter("model.depth", minimum=1.0, maximum=6.0)],
    }
    store = FileStateStore(str(tmp_path))
    store.save_job_specs(
        "hierarchical-session",
        {"train": {"num_epochs": 1}},
    )
    wrapper, _, _ = _make_brain(
        store=store,
        parameters=parameters,
    )

    wrapper.save_state()

    persisted = store._read_json(
        "brain",
        "hierarchical-session.json",
    )
    tagged_maximum = persisted["signature"]["arms"]["dino.a"]["inner"][
        "parameters"
    ][0]["valid_max"]
    assert tagged_maximum == {
        "__automl_nonfinite__": "positive_infinity"
    }
    json.dumps(persisted, allow_nan=False)

    restored, _, _ = _make_brain(
        store=store,
        parameters=parameters,
        resume=True,
    )
    assert restored.signature == wrapper.signature
    assert restored.signature_sha256 == wrapper.signature_sha256


def test_wrapper_resume_accepts_legacy_implicit_accuracy_tolerance():
    wrapper, store, _ = _make_brain()
    wrapper.save_state()
    state = store.brain_state
    state["signature"]["scheduler"]["policy"].pop("accuracy_tolerance")
    state["signature_sha256"] = canonical_audit_sha256(state["signature"])
    scheduler_state = state["scheduler_state"]
    scheduler_state["signature"]["policy"].pop("accuracy_tolerance")
    scheduler_state["signature_sha256"] = canonical_audit_sha256(
        scheduler_state["signature"]
    )
    state["scheduler_state_sha256"] = canonical_audit_sha256(
        scheduler_state
    )
    payload = copy.deepcopy(state)
    payload.pop("state_sha256")
    state["state_sha256"] = canonical_audit_sha256(payload)

    restored, _, _ = _make_brain(store=store, resume=True)

    assert restored.scheduler.policy.accuracy_tolerance == pytest.approx(
        1e-12
    )
    assert restored.scheduler.state_dict() == wrapper.scheduler.state_dict()


def test_resume_rejects_state_tamper_and_configuration_mismatch():
    wrapper, store, _ = _make_brain()
    wrapper.generate_recommendations([])
    wrapper.save_state()

    store.brain_state["scheduler_state"]["decision_index"] = 999
    with pytest.raises(ValueError, match="state integrity"):
        _make_brain(store=store, resume=True)

    wrapper, store, _ = _make_brain()
    wrapper.generate_recommendations([])
    wrapper.save_state()
    changed = copy.deepcopy(
        {
            "dino.a": {
                "train.pretrained_model_path": "/resolved/changed-a.pth",
            },
            "dino.b": {
                "train.pretrained_model_path": "/resolved/b.pth",
            },
        }
    )
    with pytest.raises(ValueError, match="different configuration"):
        _make_brain(
            store=store,
            resume=True,
            candidate_overrides=changed,
        )


def test_save_state_snapshots_scheduler_and_every_inner_brain():
    wrapper, store, _ = _make_brain()
    wrapper.generate_recommendations([])
    for inner in wrapper.inner_brains.values():
        inner.save_state = MagicMock(wraps=inner.save_state)

    wrapper.save_state()

    assert all(inner.save_state.call_count == 1 for inner in wrapper.inner_brains.values())
    assert set(store.brain_state["inner_states"]) == {"dino.a", "dino.b"}
    assert store.brain_state["scheduler_state"]["decision_index"] == 1
    payload = copy.deepcopy(store.brain_state)
    expected = payload.pop("state_sha256")
    assert expected == canonical_audit_sha256(payload)


def test_failed_trials_are_preserved_and_recovery_is_bounded():
    wrapper, _, config = _make_brain(mode="accuracy")
    history = []
    failed, first_audit = _issued_recommendation(
        wrapper,
        config,
        history,
        0,
        accuracy=None,
        latency=None,
        status="failure",
    )
    history.append(failed)
    successful, second_audit = _issued_recommendation(
        wrapper,
        config,
        history,
        1,
        accuracy=0.7,
        latency=10.0,
    )
    history.append(successful)
    assert first_audit["ptm"]["arm_id"] != second_audit["ptm"]["arm_id"]

    recovery, recovery_audit = _issued_recommendation(
        wrapper,
        config,
        history,
        2,
        accuracy=None,
        latency=None,
        status="failure",
    )
    history.append(recovery)
    assert recovery_audit["stage"] == "preregistered_invalid_recovery"
    assert recovery_audit["ptm"]["arm_id"] == first_audit["ptm"]["arm_id"]

    _, next_audit = _issued_recommendation(
        wrapper,
        config,
        history,
        3,
        accuracy=0.65,
        latency=11.0,
    )
    assert next_audit["stage"] == "mode_aware_outer_allocation"
    assert len(history) == 3
    assert [item.status for item in history].count("failure") == 2


def test_global_history_order_does_not_change_recommendation_or_audit():
    first, _, config = _make_brain()
    second, _, _ = _make_brain()
    history_a = []
    history_b = []
    for identifier, values in enumerate(((0.7, 12.0), (0.6, 9.0))):
        rec_a, audit_a = _issued_recommendation(
            first,
            config,
            history_a,
            identifier,
            accuracy=values[0],
            latency=values[1],
        )
        rec_b, audit_b = _issued_recommendation(
            second,
            config,
            list(reversed(history_b)),
            identifier,
            accuracy=values[0],
            latency=values[1],
        )
        assert rec_a.specs == rec_b.specs
        assert audit_a == audit_b
        history_a.append(rec_a)
        history_b.append(rec_b)

    raw_a = first.generate_recommendations(history_a)
    raw_b = second.generate_recommendations(list(reversed(history_b)))
    assert raw_a == raw_b
    assert (
        first.consume_last_recommendation_audits()
        == second.consume_last_recommendation_audits()
    )


def test_pending_global_candidate_blocks_without_consuming_outer_allocation():
    wrapper, _, config = _make_brain()
    recommendation, _ = _issued_recommendation(
        wrapper,
        config,
        [],
        0,
        accuracy=0.0,
        latency=0.0,
        status="pending",
    )
    state = wrapper.scheduler.state_dict()

    assert wrapper.generate_recommendations([recommendation]) == []
    assert wrapper.scheduler.state_dict() == state
    assert wrapper.consume_last_recommendation_audits() == []


def test_empty_inner_emission_rolls_back_scheduler_issue():
    wrapper, _, _ = _make_brain()
    selected = HierarchicalPTMScheduler(
        ARMS,
        wrapper.scheduler.policy,
        random_seed=wrapper.scheduler.random_seed,
    ).choose_arm([]).checkpoint_id
    wrapper.inner_brains[selected].generate_recommendations = MagicMock(
        return_value=[]
    )
    before = wrapper.scheduler.state_dict()

    assert wrapper.generate_recommendations([]) == []
    assert wrapper.scheduler.state_dict() == before


def test_controller_metadata_describes_conditional_native_search():
    wrapper, _, _ = _make_brain()

    assert wrapper.algorithm_capability["inner_algorithm"] == "bayesian"
    assert wrapper.algorithm_capability[
        "checkpoint_identity_representation"
    ] == "categorical_outer_arm_not_surrogate_dimension"
    assert wrapper.objective_mode_capability["objective_aware"] is True
    assert wrapper.objective_mode_capability["sees_raw_objectives"] is True
    assert set(wrapper.custom_ranges) == {"dino.a", "dino.b"}
    assert [item["ptm_arm_id"] for item in wrapper.parameters] == [
        "dino.a",
        "dino.b",
    ]


def test_controller_persists_complete_wrapper_on_issuance_and_result():
    wrapper, store, config = _make_brain()
    controller = Controller(
        brain=wrapper,
        context=wrapper.context,
        state_store=store,
        settings=SimpleNamespace(automl_max_recommendations=4),
        metric="multi_objective_score",
        algorithm="bayesian",
        objective_config=config,
    )

    recommendations = controller.next_recommendation()

    assert len(recommendations) == 1
    assert store.save_count == 1
    assert set(store.brain_state["inner_states"]) == {"dino.a", "dino.b"}
    proposal = recommendations[0].recommendation_audit["acquisition"]["proposal"]
    assert proposal["ptm"]["arm_id"] in {"dino.a", "dino.b"}
    assert store.controller_state[0]["recommendation_audit"][
        "audit_sha256"
    ]

    controller.report_result(
        recommendations[0].id,
        {"mAP50": 0.7, "latency_ms": 10.0},
    )

    assert store.save_count == 2
    assert store.brain_state["scheduler_state"]["decision_index"] == 1
    assert store.controller_state[0]["status"] == "success"


def test_interrupted_hierarchical_issuance_fails_closed_on_resume(
    tmp_path,
    monkeypatch,
):
    store = FileStateStore(str(tmp_path))
    store.save_job_specs(
        "hierarchical-session",
        {"train": {"num_epochs": 1}},
    )
    wrapper, _, config = _make_brain(store=store)
    controller = Controller(
        brain=wrapper,
        context=wrapper.context,
        state_store=store,
        settings=SimpleNamespace(automl_max_recommendations=2),
        metric="multi_objective_score",
        algorithm="bayesian",
        objective_config=config,
    )
    first = controller.next_recommendation()[0]
    controller.report_result(
        first.id,
        {"mAP50": 0.70, "latency_ms": 10.0},
    )

    def fail_controller_write(_job_id, _state):
        raise RuntimeError("injected hierarchical controller write failure")

    monkeypatch.setattr(store, "save_controller_info", fail_controller_write)
    with pytest.raises(
        RuntimeError,
        match="injected hierarchical controller write failure",
    ):
        controller.next_recommendation()

    transaction = store.get_state_transaction(wrapper.context.id)
    assert transaction["status"] == "pending"
    persisted_brain = store._read_json(
        "brain",
        f"{wrapper.context.id}.json",
    )
    assert persisted_brain["scheduler_state"]["decision_index"] == 2
    assert len(
        store._read_json(
            "controller",
            f"{wrapper.context.id}.json",
        )
    ) == 1
    with pytest.raises(
        RuntimeError,
        match="Incomplete AutoML state transaction",
    ):
        _make_brain(store=store, resume=True)
