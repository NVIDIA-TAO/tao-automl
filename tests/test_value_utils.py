# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for canonical AutoML values and strict state persistence."""

import json
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tao_automl.controller.controller import Controller
from tao_automl.objectives import parse_objective_config
from tao_automl.state.state_store import StateStore
from tao_automl.types import AutoMLContext, Recommendation, ResumeRecommendation
from tao_automl.utils.value_utils import normalize_json_value


class _Mode(Enum):
    FAST = "fast"


class _StaticBrain:
    def __init__(self, recommendation):
        self.recommendation = recommendation

    def generate_recommendations(self, history):
        return [self.recommendation]

    def save_state(self):
        return None


def _controller(tmp_path, recommendation):
    context = AutoMLContext(id="value-test", network="external")
    return Controller(
        brain=_StaticBrain(recommendation),
        context=context,
        state_store=StateStore(str(tmp_path)),
        settings=SimpleNamespace(automl_max_recommendations=1),
        metric="accuracy",
        algorithm="bayesian",
    )


def _numpy_specs(tmp_path):
    return {
        "float": np.float64(0.25),
        "int": np.int64(7),
        "bool": np.bool_(True),
        "string": np.str_("value"),
        "array": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "nested": {
            "floats": np.array([0.5, 0.75], dtype=np.float64),
            "tuple": (np.int32(9), Path(tmp_path) / "dataset"),
            "mode": _Mode.FAST,
        },
    }


def _assert_canonical_specs(specs, tmp_path):
    assert type(specs["float"]) is float
    assert type(specs["int"]) is int
    assert type(specs["bool"]) is bool
    assert type(specs["string"]) is str
    assert specs["array"] == [[1, 2], [3, 4]]
    assert all(type(value) is int for row in specs["array"] for value in row)
    assert specs["nested"] == {
        "floats": [0.5, 0.75],
        "tuple": [9, str(Path(tmp_path) / "dataset")],
        "mode": "fast",
    }


def test_normal_recommendation_is_normalized_before_persisting(tmp_path):
    controller = _controller(tmp_path, _numpy_specs(tmp_path))

    recommendation = controller.next_recommendation()[0]
    persisted = controller.state_store.get_controller_info(
        controller.context.id
    )[0]["specs"]

    _assert_canonical_specs(recommendation.specs, tmp_path)
    _assert_canonical_specs(persisted, tmp_path)
    assert recommendation.specs == persisted

    loaded = Controller.load_state(
        brain=_StaticBrain({}),
        context=controller.context,
        state_store=controller.state_store,
        settings=controller.settings,
        metric=controller.metric,
        algorithm=controller.algorithm,
    )

    assert len(loaded.history) == 1
    _assert_canonical_specs(loaded.history[0].specs, tmp_path)
    assert loaded.history[0].specs == recommendation.specs


def test_resume_recommendation_is_normalized_before_creation(tmp_path):
    raw = ResumeRecommendation(
        identity=0,
        specs=_numpy_specs(tmp_path),
        job_id="previous-job",
    )
    controller = _controller(tmp_path, raw)

    recommendation = controller.next_recommendation()[0]
    persisted = controller.state_store.get_controller_info(
        controller.context.id
    )[0]["specs"]

    _assert_canonical_specs(recommendation.specs, tmp_path)
    _assert_canonical_specs(persisted, tmp_path)
    assert recommendation.specs == persisted
    assert recommendation.resume_from_job_id == "previous-job"


def test_state_store_writes_strict_json_after_normalizing(tmp_path):
    store = StateStore(str(tmp_path))
    store.save_job_specs("job", _numpy_specs(tmp_path))

    state_path = tmp_path / ".automl" / "specs" / "job.json"
    payload = json.loads(
        state_path.read_text(encoding="utf-8"),
        parse_constant=lambda value: pytest.fail(
            f"non-standard JSON constant persisted: {value}"
        ),
    )

    _assert_canonical_specs(payload, tmp_path)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -float("inf"), np.float64("nan")],
    ids=["nan", "positive-infinity", "negative-infinity", "numpy-nan"],
)
def test_normalizer_rejects_nonfinite_numbers(value):
    with pytest.raises(ValueError, match=r"\$\.metric: numeric values must be finite"):
        normalize_json_value({"metric": value})


@pytest.mark.parametrize(
    "value",
    [object(), {1, 2}, 1 + 2j, b"bytes"],
    ids=["object", "set", "complex", "bytes"],
)
def test_normalizer_rejects_unsupported_values(value):
    with pytest.raises(TypeError, match=r"\$\.value: unsupported value type"):
        normalize_json_value({"value": value})


def test_normalizer_rejects_non_string_mapping_keys():
    with pytest.raises(TypeError, match="mapping keys must be strings"):
        normalize_json_value({1: "value"})


def test_normalizer_converts_numpy_extended_float_without_recursing():
    normalized = normalize_json_value(np.longdouble("0.125"))

    assert normalized == pytest.approx(0.125)
    assert type(normalized) is float


@pytest.mark.parametrize("value", [True, np.bool_(True), [0.5], np.array([0.5])])
def test_recommendation_result_rejects_bool_and_non_scalar_values(value):
    recommendation = Recommendation(0, {}, "accuracy")

    with pytest.raises(TypeError, match="finite numeric scalar"):
        recommendation.update_result(value)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -float("inf"), np.float64("nan")],
)
def test_recommendation_result_rejects_nonfinite_values(value):
    recommendation = Recommendation(0, {}, "accuracy")

    with pytest.raises(ValueError, match="numeric values must be finite"):
        recommendation.update_result(value)


def test_recommendation_result_rejects_integer_too_large_for_float():
    recommendation = Recommendation(0, {}, "accuracy")

    with pytest.raises(ValueError, match="representable as a finite float"):
        recommendation.update_result(10**400)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.float64(0.25), 0.25),
        (np.float32(0.5), 0.5),
        (np.int64(3), 3.0),
        (np.array(0.75), 0.75),
    ],
)
def test_recommendation_result_accepts_finite_numpy_scalars(value, expected):
    recommendation = Recommendation(0, {}, "accuracy")

    recommendation.update_result(value)

    assert recommendation.result == pytest.approx(expected)
    assert type(recommendation.result) is float


def test_recommendation_result_refreshes_primary_objective_value():
    recommendation = Recommendation(0, {}, "accuracy")

    recommendation.update_result(0.25)
    recommendation.update_result(0.75)

    assert recommendation.primary_metric_value() == pytest.approx(0.75)
    assert recommendation.objective_values == {"accuracy": 0.75}


def test_recommendation_objectives_normalize_numpy_values():
    recommendation = Recommendation(0, {}, "accuracy")

    recommendation.update_objectives(
        {"accuracy": np.float64(0.8), "latency": np.int64(12)},
        np.float64(0.68),
    )

    assert recommendation.objective_values == {
        "accuracy": 0.8,
        "latency": 12.0,
    }
    assert recommendation.objective_score == pytest.approx(0.68)


@pytest.mark.parametrize(
    "value",
    [True, "0.8", float("nan"), float("inf"), -float("inf")],
)
def test_objective_config_rejects_invalid_metric_values(value):
    config = parse_objective_config({
        "metric": "accuracy",
        "multi_objective": True,
        "latency_metric": "latency",
    })

    with pytest.raises((TypeError, ValueError)):
        config.coerce_values({"accuracy": value, "latency": 12.0})


@pytest.mark.parametrize(
    "field,value",
    [
        ("weight", True),
        ("weight", float("nan")),
        ("weight", -1.0),
        ("scale", True),
        ("scale", float("inf")),
        ("scale", 0.0),
        ("scale", -1.0),
    ],
)
def test_objective_config_rejects_invalid_weight_and_scale(field, value):
    with pytest.raises((TypeError, ValueError)):
        parse_objective_config({
            "objectives": [
                {"metric": "accuracy", field: value},
            ],
        })


def test_explicit_objective_list_uses_first_metric_as_primary():
    config = parse_objective_config({
        "objectives": [
            {"metric": "accuracy", "direction": "maximize"},
            {"metric": "latency", "direction": "minimize"},
        ],
    })

    assert config.primary_metric == "accuracy"
    assert config.metric_names == ["accuracy", "latency"]
