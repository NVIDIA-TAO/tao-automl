# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Issuance-time recommendation audit contract tests."""

from types import SimpleNamespace

import numpy as np
import pytest

from tao_automl.brain.bayesian import Bayesian
from tao_automl.controller.controller import Controller
from tao_automl.objectives import parse_objective_config
from tao_automl.recommendation_audit import (
    ALGORITHMIC_CAMPAIGN_FLAGS,
    algorithmic_campaign_flags,
    build_recommendation_audit,
    canonical_audit_sha256,
    validate_recommendation_audit,
)
from tao_automl.state.state_store import StateStore
from tao_automl.types import AutoMLContext


class _AuditedBrain:
    def __init__(self):
        self.random_seed = 271828
        self.parameters = [
            {
                "parameter": "model.depth",
                "value_type": "int",
                "valid_min": np.int64(3),
                "valid_max": np.int64(6),
                "depends_on": np.nan,
            }
        ]
        self.custom_ranges = {
            "model.depth": {"valid_min": np.int64(3), "valid_max": 6}
        }
        self.count = 0
        self.save_calls = 0
        self.acquisition_audit = {
            "method": "deterministic_test_acquisition",
            "observation_count": 0,
        }

    def generate_recommendations(self, history):
        if self.count >= 2:
            return []
        self.acquisition_audit = {
            "method": "deterministic_test_acquisition",
            "observation_count": sum(
                rec.status in {"success", "done"} for rec in history
            ),
            "design_index": self.count,
        }
        self.count += 1
        return [{"model.depth": 2 + self.count}]

    def save_state(self):
        self.save_calls += 1


class _FailControllerWriteStateStore(StateStore):
    def __init__(self, workspace_path):
        super().__init__(workspace_path)
        self.fail_controller_write = False

    def save_controller_info(self, job_id, recs):
        if self.fail_controller_write:
            raise RuntimeError("injected controller persistence failure")
        super().save_controller_info(job_id, recs)


def _controller(tmp_path, brain=None):
    brain = brain or _AuditedBrain()
    context = AutoMLContext(
        id="recommendation-audit",
        network="dino",
        random_seed=271828,
    )
    return Controller(
        brain=brain,
        context=context,
        state_store=StateStore(str(tmp_path)),
        settings=SimpleNamespace(automl_max_recommendations=2),
        metric="accuracy",
        algorithm="bayesian",
    )


def test_recommendation_audit_records_frozen_algorithmic_inputs(tmp_path):
    controller = _controller(tmp_path)
    first = controller.next_recommendation()[0]
    audit = first.recommendation_audit

    assert audit["candidate_id"] == "0"
    assert audit["search_algorithm"] == "bayesian"
    assert audit["search_seed"] == 271828
    assert audit["generated_parameter_values"] == {"model.depth": 3}
    assert audit["history_visible_to_algorithm"] == []
    assert audit["previous_successful_observations"] == []
    assert audit["acquisition"] == {
        "proposal": {
            "method": "deterministic_test_acquisition",
            "observation_count": 0,
            "design_index": 0,
        },
        "algorithm_capability": None,
        "objective_mode_capability": None,
    }
    assert audit["algorithmic_campaign_flags"] == ALGORITHMIC_CAMPAIGN_FLAGS
    assert audit["selection_time_measurements_only"] is True
    assert audit["search_space"][0]["depends_on"] == {
        "__automl_nonfinite__": "nan"
    }
    assert audit["search_space_sha256"] == canonical_audit_sha256(
        audit["search_space"]
    )
    expected_hash = audit.pop("audit_sha256")
    assert expected_hash == canonical_audit_sha256(audit)
    audit["audit_sha256"] = expected_hash


def test_issuance_persists_brain_state_before_returning_recommendation(tmp_path):
    brain = _AuditedBrain()
    controller = _controller(tmp_path, brain=brain)
    controller.next_recommendation()
    assert brain.save_calls == 1


def test_next_audit_contains_only_observations_visible_at_issuance(tmp_path):
    controller = _controller(tmp_path)
    first = controller.next_recommendation()[0]
    controller.report_result(first.id, 0.73)
    second = controller.next_recommendation()[0]

    visible = second.recommendation_audit["history_visible_to_algorithm"]
    assert len(visible) == 1
    assert visible[0]["candidate_id"] == "0"
    assert visible[0]["status"] == "success"
    assert visible[0]["objective_values"] == {"accuracy": 0.73}
    assert second.recommendation_audit["previous_successful_observations"] == visible
    assert (
        second.recommendation_audit["acquisition"]["proposal"][
            "observation_count"
        ]
        == 1
    )

    # Later results cannot rewrite the frozen issuance record.
    frozen_hash = second.recommendation_audit["audit_sha256"]
    controller.report_result(second.id, 0.81)
    assert second.recommendation_audit["audit_sha256"] == frozen_hash


def test_recommendation_audit_survives_controller_resume(tmp_path):
    controller = _controller(tmp_path)
    recommendation = controller.next_recommendation()[0]
    expected = recommendation.recommendation_audit

    loaded = Controller.load_state(
        brain=_AuditedBrain(),
        context=controller.context,
        state_store=controller.state_store,
        settings=controller.settings,
        metric=controller.metric,
        algorithm=controller.algorithm,
    )
    assert loaded.history[0].recommendation_audit == expected
    assert loaded.get_status()["recommendations"][0]["recommendation_audit"] == expected


def test_mutated_recommendation_audit_cannot_be_persisted(tmp_path):
    controller = _controller(tmp_path)
    recommendation = controller.next_recommendation()[0]
    recommendation.recommendation_audit["algorithmic_campaign_flags"][
        "agent_injected_candidate"
    ] = True

    with pytest.raises(ValueError, match="integrity verification"):
        controller.save_state()


def test_public_campaign_flags_cannot_be_mutated():
    with pytest.raises(TypeError):
        ALGORITHMIC_CAMPAIGN_FLAGS["agent_injected_candidate"] = True


@pytest.mark.parametrize("replacement", [True, 0, np.bool_(False), None])
def test_rehashed_intervention_flag_tampering_is_rejected(replacement):
    record = build_recommendation_audit(
        candidate_id="candidate",
        specs={"model.depth": 3},
        algorithm="bayesian",
        search_seed=271828,
        search_space=[],
        custom_ranges={},
        objective_config=None,
        visible_history=[],
        acquisition={},
    )
    record["algorithmic_campaign_flags"]["agent_injected_candidate"] = replacement
    payload = dict(record)
    payload.pop("audit_sha256")
    record["audit_sha256"] = canonical_audit_sha256(payload)

    with pytest.raises(ValueError, match="intervention flags"):
        validate_recommendation_audit(record)


def test_rehashed_campaign_flag_schema_tampering_is_rejected():
    record = build_recommendation_audit(
        candidate_id="candidate",
        specs={"model.depth": 3},
        algorithm="bayesian",
        search_seed=271828,
        search_space=[],
        custom_ranges={},
        objective_config=None,
        visible_history=[],
        acquisition={},
    )
    record["algorithmic_campaign_flags"]["new_flag"] = False
    payload = dict(record)
    payload.pop("audit_sha256")
    record["audit_sha256"] = canonical_audit_sha256(payload)

    with pytest.raises(ValueError, match="flag schema"):
        validate_recommendation_audit(record)


def test_campaign_flag_factory_returns_independent_false_records():
    first = algorithmic_campaign_flags()
    second = algorithmic_campaign_flags()
    first["agent_injected_candidate"] = True
    assert second == ALGORITHMIC_CAMPAIGN_FLAGS


def test_interrupted_native_bayesian_issuance_fails_closed_on_resume(tmp_path):
    context = AutoMLContext(
        id="split-native-bayesian",
        network="dino",
        random_seed=271828,
    )
    store = _FailControllerWriteStateStore(str(tmp_path))
    store.save_job_specs(context.id, {"train": {"num_epochs": 1}})
    objective_config = parse_objective_config({
        "objectives": [
            {"metric": "mAP50", "direction": "maximize"},
            {"metric": "latency_ms", "direction": "minimize"},
        ],
        "selection_mode": "latency",
        "accuracy_metric": "mAP50",
        "latency_metric": "latency_ms",
        "latency_accuracy_retention": 0.90,
    })
    parameters = [{
        "parameter": "model.depth",
        "value_type": "int",
        "default_value": 4,
        "valid_min": 3,
        "valid_max": 6,
        "valid_options": [],
        "option_weights": None,
        "math_cond": None,
        "parent_param": None,
        "depends_on": None,
    }]
    brain = Bayesian(
        context,
        store,
        "dino",
        parameters,
        metric="multi_objective_score",
        direction="maximize",
        objective_config=objective_config,
        acquisition_settings={"calibration_points": 2},
    )
    controller = Controller(
        brain=brain,
        context=context,
        state_store=store,
        settings=SimpleNamespace(automl_max_recommendations=2),
        metric="multi_objective_score",
        algorithm="bayesian",
        objective_config=objective_config,
    )
    first = controller.next_recommendation()[0]
    controller.report_result(
        first.id,
        {"mAP50": 0.70, "latency_ms": 10.0},
    )
    store.fail_controller_write = True

    with pytest.raises(
        RuntimeError,
        match="injected controller persistence failure",
    ):
        controller.next_recommendation()

    transaction = store.get_state_transaction(context.id)
    assert transaction["status"] == "pending"
    assert transaction["operation"] == "recommendation_issuance"
    assert store._read_json("brain", f"{context.id}.json")[
        "recommendation_count"
    ] == 2
    assert len(
        store._read_json("controller", f"{context.id}.json")
    ) == 1
    with pytest.raises(
        RuntimeError,
        match="Incomplete AutoML state transaction",
    ):
        Bayesian.load_state(
            context,
            store,
            "dino",
            parameters,
            metric="multi_objective_score",
            direction="maximize",
            objective_config=objective_config,
            acquisition_settings={"calibration_points": 2},
        )


def test_committed_component_changed_outside_transaction_is_rejected(tmp_path):
    controller = _controller(tmp_path)
    controller.next_recommendation()
    persisted = controller.state_store._read_json(
        "controller",
        f"{controller.context.id}.json",
    )
    persisted[0]["status"] = "success"
    controller.state_store._write_json(
        persisted,
        "controller",
        f"{controller.context.id}.json",
    )

    with pytest.raises(
        RuntimeError,
        match="integrity mismatch for 'controller'",
    ):
        Controller.load_state(
            brain=_AuditedBrain(),
            context=controller.context,
            state_store=controller.state_store,
            settings=controller.settings,
            metric=controller.metric,
            algorithm=controller.algorithm,
        )
