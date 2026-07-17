# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for HybridBrain strategy guardrails."""

from tao_automl.brain.hybrid_controller import HybridBrain, HybridStrategist
from tao_automl.state.state_store import StateStore
from tao_automl.types import AutoMLContext, JobStates, Recommendation


def _params(names):
    return [{"parameter": name, "value_type": "float"} for name in names]


def test_hybrid_first_phase_keeps_cosmos_lora_coverage():
    available = _params([
        "train.epoch",
        "train.train_batch_per_replica",
        "train.optm_lr",
        "train.optm_weight_decay",
        "train.optm_warmup_epochs",
        "policy.lora.r",
        "policy.lora.lora_alpha",
        "policy.lora.lora_dropout",
    ])
    strategist = HybridStrategist(llm_client=object())

    plan = strategist._validate_plan(
        {
            "action": "sweep",
            "algorithm": "bfbo",
            "parameters": [
                "train.optm_lr",
                "train.optm_weight_decay",
                "policy.lora.r",
                "policy.lora.lora_alpha",
            ],
            "trials": 10,
        },
        available,
    )

    assert plan["parameters"] == [param["parameter"] for param in available]
    assert plan["guardrail_added_parameters"] == [
        "train.epoch",
        "train.train_batch_per_replica",
        "train.optm_warmup_epochs",
        "policy.lora.lora_dropout",
    ]


def test_hybrid_first_phase_reserves_refinement_budget(tmp_path, monkeypatch):
    class StaticStrategist:
        def __init__(self):
            self.completed_phases = []
            self.full_history = []

        def plan_next_phase(self, **_kwargs):
            return {
                "action": "sweep",
                "algorithm": "bayesian",
                "parameters": ["train.optm_lr"],
                "trials": 10,
                "algorithm_params": {},
            }

    class BurstBrain:
        def generate_recommendations(self, _history):
            return [{"train.optm_lr": i} for i in range(10)]

    def fake_create_brain(**_kwargs):
        return BurstBrain()

    monkeypatch.setattr(
        "tao_automl.brain.factory.BrainFactory.create_brain",
        fake_create_brain,
    )

    store = StateStore(str(tmp_path))
    ctx = AutoMLContext(id="hybrid-budget", network="cosmos-rl", metric="val/avg_loss")
    brain = HybridBrain(
        context=ctx,
        state_store=store,
        network="cosmos-rl",
        parameters=_params(["train.optm_lr"]),
        metric="val/avg_loss",
        max_experiments=10,
    )
    brain.strategist = StaticStrategist()

    recs = brain.generate_recommendations([])

    assert len(recs) == 8
    assert brain.current_plan["trials"] == 8
    assert brain.current_plan["reserved_refinement_budget"] == 2


def test_hybrid_two_trial_budget_reserves_evidence_based_second_phase(tmp_path, monkeypatch):
    class StaticStrategist:
        def __init__(self):
            self.completed_phases = []
            self.full_history = []

        def plan_next_phase(self, **_kwargs):
            return {
                "action": "sweep",
                "algorithm": "bayesian",
                "parameters": ["train.optm_lr"],
                "trials": 10,
                "algorithm_params": {},
            }

    class BurstBrain:
        def generate_recommendations(self, _history):
            return [{"train.optm_lr": i} for i in range(10)]

    monkeypatch.setattr(
        "tao_automl.brain.factory.BrainFactory.create_brain",
        lambda **_kwargs: BurstBrain(),
    )

    store = StateStore(str(tmp_path))
    ctx = AutoMLContext(id="hybrid-two-trial", network="cosmos-rl", metric="val/avg_loss")
    brain = HybridBrain(
        context=ctx,
        state_store=store,
        network="cosmos-rl",
        parameters=_params(["train.optm_lr"]),
        metric="val/avg_loss",
        max_experiments=2,
    )
    brain.strategist = StaticStrategist()

    recs = brain.generate_recommendations([])

    assert len(recs) == 1
    assert brain.current_plan["trials"] == 1
    assert brain.current_plan["reserved_refinement_budget"] == 1


def test_hybrid_persists_llm_usage_and_phase_decisions():
    class Usage:
        @staticmethod
        def to_dict():
            return {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "num_calls": 1,
                "total_latency_ms": 12.3,
                "errors": 0,
            }

    class Client:
        usage = Usage()

    strategist = HybridStrategist(llm_client=Client())
    strategist.record_phase_results(
        phase_plan={"action": "sweep", "algorithm": "bayesian"},
        results=[
            {"rec_id": 0, "metric": 0.5, "status": "success"},
            {"rec_id": 1, "metric": 0.7, "status": "success"},
        ],
        best_metric=0.7,
        reverse_sort=True,
    )

    state = strategist.to_dict()

    assert state["llm_usage"]["num_calls"] == 1
    assert [item["decision"] for item in state["full_history"]] == ["discard", "keep"]
    assert state["completed_phases"][0]["top_results"][0]["decision"] == "keep"


def test_hybrid_records_phase_when_budget_exhausted(tmp_path):
    store = StateStore(str(tmp_path))
    ctx = AutoMLContext(id="hybrid-record", network="cosmos-rl", metric="val/avg_loss")
    brain = HybridBrain(
        context=ctx,
        state_store=store,
        network="cosmos-rl",
        parameters=_params(["train.optm_lr"]),
        metric="val/avg_loss",
        max_experiments=2,
    )
    brain.current_plan = {
        "action": "sweep",
        "algorithm": "bayesian",
        "parameters": ["train.optm_lr"],
        "trials": 2,
    }

    rec0 = Recommendation(0, {"train.optm_lr": 0.001}, "val/avg_loss")
    rec0.update_result(0.9)
    rec0.update_status(JobStates.success)
    rec1 = Recommendation(1, {"train.optm_lr": 0.0001}, "val/avg_loss")
    rec1.update_result(1.2)
    rec1.update_status(JobStates.failure)

    assert brain.generate_recommendations([rec0, rec1]) == []

    phase = brain.strategist.completed_phases[0]
    assert phase["num_experiments"] == 2
    assert phase["num_success"] == 1
    assert phase["num_failure"] == 1
    assert phase["best_metric"] == 0.9
    assert brain.strategist.full_history[0]["rec_id"] == 0
    assert brain.done()


def test_hybrid_range_narrowing_disabled_by_default():
    available = [
        {
            "parameter": "train.train_batch_per_replica",
            "value_type": "ordered_int",
            "valid_options": [8, 16, 32],
        }
    ]
    strategist = HybridStrategist(llm_client=object())
    strategist.completed_phases = [{"phase_number": 1}]

    plan = strategist._validate_plan(
        {
            "action": "sweep",
            "algorithm": "bayesian",
            "parameters": ["train.train_batch_per_replica"],
            "trials": 4,
            "parameter_overrides": {
                "train.train_batch_per_replica": {"valid_options": [8]},
            },
        },
        available,
    )

    assert "parameter_overrides" not in plan


def test_hybrid_validates_llm_option_narrowing_from_bounds():
    available = [
        {
            "parameter": "train.train_batch_per_replica",
            "value_type": "ordered_int",
            "valid_options": [8, 16, 32],
        }
    ]
    strategist = HybridStrategist(
        llm_client=object(),
        enable_range_narrowing=True,
    )
    strategist.completed_phases = [{"phase_number": 1}]

    plan = strategist._validate_plan(
        {
            "action": "sweep",
            "algorithm": "bayesian",
            "parameters": ["train.train_batch_per_replica"],
            "trials": 4,
            "parameter_overrides": {
                "train.train_batch_per_replica": {"valid_max": 8},
            },
        },
        available,
    )

    assert plan["parameter_overrides"] == {
        "train.train_batch_per_replica": {"valid_options": [8]},
    }


def test_hybrid_applies_llm_phase_overrides_to_sub_brain_ranges(tmp_path, monkeypatch):
    captured = {}

    class DummyBrain:
        num_epochs_per_experiment = 0

    def fake_create_brain(**kwargs):
        captured["parameters"] = kwargs["parameters"]
        captured["custom_ranges"] = kwargs["state_store"].get_custom_param_ranges(
            kwargs["context"].handler_id
        )
        return DummyBrain()

    monkeypatch.setattr(
        "tao_automl.brain.factory.BrainFactory.create_brain",
        fake_create_brain,
    )

    store = StateStore(str(tmp_path))
    ctx = AutoMLContext(id="hybrid-narrow", network="cosmos-rl", metric="val/avg_loss")
    store.save_custom_param_ranges(ctx.handler_id, {
        "train.train_batch_per_replica": {
            "value_type": "ordered_int",
            "valid_options": [8, 16, 32],
        }
    })
    brain = HybridBrain(
        context=ctx,
        state_store=store,
        network="cosmos-rl",
        parameters=[
            {
                "parameter": "train.train_batch_per_replica",
                "value_type": "ordered_int",
                "valid_options": [8, 16, 32],
            }
        ],
        metric="val/avg_loss",
        max_experiments=8,
        enable_llm_range_narrowing=True,
    )

    brain._create_sub_brain({
        "action": "sweep",
        "algorithm": "bayesian",
        "parameters": ["train.train_batch_per_replica"],
        "algorithm_params": {},
        "parameter_overrides": {
            "train.train_batch_per_replica": {"valid_options": [8]},
        },
    })

    assert captured["custom_ranges"]["train.train_batch_per_replica"]["valid_options"] == [8]
    assert captured["parameters"][0]["valid_options"] == [8]
    assert brain.base_custom_ranges["train.train_batch_per_replica"]["valid_options"] == [8, 16, 32]


def test_hybrid_uses_distinct_sub_brain_seed_context_per_phase(tmp_path, monkeypatch):
    context_ids = []

    class DummyBrain:
        num_epochs_per_experiment = 0

    def fake_create_brain(**kwargs):
        context_ids.append(kwargs["context"].id)
        return DummyBrain()

    monkeypatch.setattr(
        "tao_automl.brain.factory.BrainFactory.create_brain",
        fake_create_brain,
    )

    store = StateStore(str(tmp_path))
    ctx = AutoMLContext(id="hybrid-seeds", network="action-recognition", metric="accuracy")
    brain = HybridBrain(
        context=ctx,
        state_store=store,
        network="action-recognition",
        parameters=_params(["train.optim.lr"]),
        metric="accuracy",
        max_experiments=2,
    )
    plan = {
        "action": "sweep",
        "algorithm": "bayesian",
        "parameters": ["train.optim.lr"],
        "algorithm_params": {},
    }

    brain._create_sub_brain(plan)
    brain.strategist.completed_phases.append({"phase_number": 1})
    brain._create_sub_brain(plan)

    assert context_ids == [
        "hybrid-seeds-hybrid-phase-1",
        "hybrid-seeds-hybrid-phase-2",
    ]
