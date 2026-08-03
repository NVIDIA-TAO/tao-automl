# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AutoML search-space filtering."""

from unittest.mock import patch

import pytest

from tao_automl import AutoML
from tao_automl.schema.generate_schema import generate_schema
from tao_automl.search_space.params import generate_hyperparams_to_search


def _external_model_schema():
    return {
        "type": "object",
        "default": {
            "model": {
                "n_estimators": 10,
                "max_depth": 3,
            },
        },
        "properties": {
            "model": {
                "type": "object",
                "properties": {
                    "n_estimators": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 2,
                        "maximum": 20,
                        "automl_enabled": True,
                    },
                    "max_depth": {
                        "type": "integer",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 6,
                        "automl_enabled": True,
                    },
                },
            },
        },
    }


def test_cosmos_default_search_excludes_fps_when_nframes_is_active():
    specs = generate_schema("cosmos-rl", "train")["default"]

    _, param_names = generate_hyperparams_to_search(
        network="cosmos-rl",
        action="train",
        train_specs=specs,
        automl_hyperparameters=[],
    )

    assert "nframes" in specs["custom"]["vision"]
    assert "custom.vision.fps" not in param_names


def test_cosmos_explicit_fps_search_is_still_allowed():
    specs = generate_schema("cosmos-rl", "train")["default"]

    _, param_names = generate_hyperparams_to_search(
        network="cosmos-rl",
        action="train",
        train_specs=specs,
        automl_hyperparameters=["custom.vision.fps"],
    )

    assert "custom.vision.fps" in param_names


def test_external_schema_supports_network_without_builtin_config():
    schema = _external_model_schema()
    train_specs = schema["default"]

    with patch(
        "tao_automl.search_space.params.generate_schema",
        side_effect=AssertionError("built-in schema generation must be skipped"),
    ):
        records, param_names = generate_hyperparams_to_search(
            network="public_random_forest",
            action="train",
            train_specs=train_specs,
            automl_hyperparameters=[],
            schema=schema,
        )

    assert param_names == ["model.n_estimators", "model.max_depth"]
    assert {record["parameter"] for record in records} == set(param_names)
    assert records[0]["valid_min"] == 2
    assert records[0]["valid_max"] == 20


def test_automl_tunes_arbitrary_network_from_external_schema(tmp_path):
    schema = _external_model_schema()

    with patch(
        "tao_automl.search_space.params.generate_schema",
        side_effect=AssertionError("built-in schema generation must be skipped"),
    ):
        automl = AutoML(
            workspace=str(tmp_path),
            network="public_random_forest",
            train_specs=schema["default"],
            settings={
                "algorithm": "bayesian",
                "metric": "accuracy",
                "automl_max_recommendations": 1,
            },
            search_schema=schema,
        )
        recommendations = automl.next_recommendation()

    assert len(recommendations) == 1
    specs = recommendations[0].specs
    assert set(specs) == {"model.n_estimators", "model.max_depth"}
    assert 2 <= specs["model.n_estimators"] <= 20
    assert 1 <= specs["model.max_depth"] <= 6


def test_external_schema_preserves_standard_scalar_enums():
    schema = {
        "type": "object",
        "default": {"mode": "fast", "depth": 2},
        "properties": {
            "mode": {
                "type": "string",
                "default": "fast",
                "enum": ["fast", "accurate"],
                "automl_enabled": True,
            },
            "depth": {
                "type": "integer",
                "default": 2,
                "enum": [2, 4, 8],
                "automl_enabled": True,
            },
        },
    }

    records, names = generate_hyperparams_to_search(
        network="external_enum_model",
        action="train",
        train_specs=schema["default"],
        automl_hyperparameters=[],
        schema=schema,
    )

    by_name = {record["parameter"]: record for record in records}
    assert names == ["mode", "depth"]
    assert by_name["mode"]["value_type"] == "categorical"
    assert by_name["mode"]["valid_options"] == ["fast", "accurate"]
    assert by_name["depth"]["value_type"] == "ordered_int"
    assert by_name["depth"]["valid_options"] == [2, 4, 8]


def test_external_schema_uses_metadata_from_concrete_anyof_branch():
    schema = {
        "type": "object",
        "default": {"width": 3},
        "properties": {
            "width": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "integer",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 5,
                        "automl_enabled": True,
                    },
                ],
            },
        },
    }

    records, names = generate_hyperparams_to_search(
        network="external_optional_model",
        action="train",
        train_specs=schema["default"],
        automl_hyperparameters=[],
        schema=schema,
    )

    assert names == ["width"]
    assert records[0]["value_type"] == "integer"
    assert records[0]["default_value"] == 3
    assert records[0]["valid_min"] == 1
    assert records[0]["valid_max"] == 5


def test_automl_rejects_empty_external_search_space(tmp_path):
    schema = {
        "type": "object",
        "default": {"name": "fixed"},
        "properties": {
            "name": {"type": "string", "default": "fixed"},
        },
    }

    with pytest.raises(ValueError, match="No searchable parameters"):
        AutoML(
            workspace=str(tmp_path),
            network="external_fixed_model",
            train_specs=schema["default"],
            settings={"algorithm": "bayesian", "metric": "accuracy"},
            search_schema=schema,
        )
