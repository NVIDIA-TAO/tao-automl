# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for reflective text evolution and diagnostic feedback."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _make_autoresearch_brain(evolvable_text_parameters=None):
    from tao_automl.brain.autoresearch_controller import AutoresearchBrain

    state_store = MagicMock()
    state_store.get_custom_param_ranges.return_value = {}
    parameters = [
        {
            "parameter": "dataset.system_prompt",
            "value_type": "categorical",
            "default_value": "seed prompt",
            "valid_options": ["seed prompt", "second seed"],
        },
        {
            "parameter": "generation.temperature",
            "value_type": "float",
            "default_value": 0.0,
            "valid_min": 0.0,
            "valid_max": 0.4,
        },
    ]
    return AutoresearchBrain(
        context=SimpleNamespace(handler_id="experiment", id="session"),
        state_store=state_store,
        network="cosmos-rl",
        parameters=parameters,
        evolvable_text_parameters=evolvable_text_parameters,
    )


def test_autoresearch_accepts_generated_text_for_designated_parameter():
    brain = _make_autoresearch_brain(["dataset.system_prompt"])
    generated_prompt = "Verify the event using visible temporal evidence."

    validated = brain._validate_and_clamp({
        "dataset.system_prompt": generated_prompt,
        "generation.temperature": 0.9,
    })

    assert validated == {
        "dataset.system_prompt": generated_prompt,
        "generation.temperature": 0.4,
    }
    prompt_parameter = brain._parameters_with_custom_ranges()[0]
    assert prompt_parameter["evolvable_text"] is True
    assert prompt_parameter["valid_options"] == ["seed prompt", "second seed"]


def test_autoresearch_keeps_categorical_text_bounded_without_opt_in():
    brain = _make_autoresearch_brain()

    assert brain._validate_and_clamp({
        "dataset.system_prompt": "an arbitrary generated prompt",
    }) == {}


def test_autoresearch_rejects_numeric_evolvable_text_parameter():
    with pytest.raises(TypeError, match="string-valued"):
        _make_autoresearch_brain(["generation.temperature"])


def test_autoresearch_prompt_includes_actionable_evaluation_feedback():
    from tao_automl.brain.prompts.autoresearch_prompts import build_autoresearch_prompt

    messages = build_autoresearch_prompt(
        spec_schema={},
        current_best_spec={"dataset": {"system_prompt": "seed prompt"}},
        experiment_history=[{
            "status": "success",
            "metric": 0.7,
            "modifications": {"dataset.system_prompt": "seed prompt"},
            "feedback": {
                "failures": [{
                    "query": "Did the forklift yield before entering the aisle?",
                    "generated_output": "No",
                    "comment": "The response ignored temporal order.",
                }],
            },
        }],
        network="cosmos-rl",
        metric_name="accuracy",
        metric_direction="maximize",
        parameters=[{
            "parameter": "dataset.system_prompt",
            "value_type": "categorical",
            "valid_options": ["seed prompt"],
            "evolvable_text": True,
        }],
    )

    prompt = messages[1]["content"]
    assert "free-form evolvable text" in prompt
    assert "seed examples" in prompt
    assert "Evaluation feedback" in prompt
    assert "ignored temporal order" in prompt


def test_experiment_feedback_round_trips_through_tracker_state():
    from tao_automl.brain.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(metric_direction="maximize")
    tracker.record_experiment(
        spec={"dataset": {"system_prompt": "prompt"}},
        modifications={"dataset.system_prompt": "prompt"},
        metric=0.75,
        status="success",
        feedback={"failures": [{"query": "Was the aisle blocked?", "feedback": "Missed occlusion."}]},
    )

    restored = ExperimentTracker.from_dict(tracker.to_dict())
    expected = {
        "failures": [{"query": "Was the aisle blocked?", "feedback": "Missed occlusion."}]
    }
    assert restored.history[0].feedback == expected
    assert restored.get_history_for_llm()[0]["feedback"] == expected


def test_algorithm_params_parse_evolvable_text_parameters():
    from tao_automl.brain.factory import AlgorithmParams

    params = AlgorithmParams.from_dict({
        "evolvable_text_parameters": "dataset.system_prompt, task.prompt",
    })

    assert params.evolvable_text_parameters == [
        "dataset.system_prompt",
        "task.prompt",
    ]


def test_direct_script_schema_allows_generated_text_only_for_opted_in_parameter(tmp_path):
    from tao_automl.runner import _validate_specs_against_schema

    schema = {
        "type": "object",
        "properties": {
            "dataset": {
                "type": "object",
                "properties": {
                    "system_prompt": {
                        "type": "categorical",
                        "enum": ["seed prompt"],
                    },
                },
            },
        },
    }
    generated = {"dataset": {"system_prompt": "new reflected prompt"}}

    with pytest.raises(ValueError, match="not in enum"):
        _validate_specs_against_schema(
            generated, schema, tmp_path / "schema.json", require_all=True
        )

    _validate_specs_against_schema(
        generated,
        schema,
        tmp_path / "schema.json",
        require_all=True,
        evolvable_text_parameters={"dataset.system_prompt"},
    )
