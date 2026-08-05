# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for fail-closed algorithm/objective capability routing."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tao_automl.brain.algorithm_capabilities import (
    build_algorithm_capability_registry,
)
from tao_automl.brain.factory import AlgorithmParams, BrainFactory
from tao_automl.objectives import parse_objective_config


_NUMERICAL_FALLBACKS = {
    "bfbo",
    "hyperband",
    "bohb",
    "asha",
    "pbt",
    "dehb",
    "hyperband_es",
}
_SCALAR_ONLY_AGENTIC = {"llm", "hybrid", "autoresearch"}


def _selector_config(mode):
    return parse_objective_config({
        "objectives": [
            {"metric": "mAP50", "direction": "maximize"},
            {"metric": "latency_ms", "direction": "minimize"},
        ],
        "selection_mode": mode,
        "accuracy_metric": "mAP50",
        "latency_metric": "latency_ms",
    })


def _factory_kwargs(objective_config):
    return {
        "context": SimpleNamespace(id="capability", handler_id="capability"),
        "state_store": object(),
        "network": "fake",
        "parameters": [],
        "params": AlgorithmParams(),
        "metric": "multi_objective_score",
        "objective_config": objective_config,
    }


def test_registry_is_derived_from_every_brain_factory_algorithm():
    definitions = BrainFactory.algorithm_definitions()
    registry = BrainFactory.objective_capabilities()

    assert registry.algorithms == tuple(definitions)
    assert {
        entry["algorithm"]
        for entry in BrainFactory.objective_capability_matrix()["algorithms"]
    } == set(definitions)


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("bayesian", "bayesian"),
        ("b", "bayesian"),
        ("h", "hyperband"),
        ("hes", "hyperband_es"),
    ],
)
def test_registry_resolves_factory_aliases(alias, canonical):
    assert (
        BrainFactory.objective_capabilities().resolve(alias).algorithm
        == canonical
    )


def test_bayesian_is_the_native_accuracy_latency_multi_objective_path():
    bayesian = BrainFactory.objective_capabilities().resolve("bayesian")

    for mode in ("accuracy", "latency", "multi_objective"):
        capability = bayesian.for_mode(mode)
        assert capability.supported
        assert capability.support_level == "native"
        assert capability.sees_raw_objectives
        assert capability.objective_aware
        assert not capability.consumes_archive_acquisition_score


@pytest.mark.parametrize("algorithm", sorted(_NUMERICAL_FALLBACKS))
@pytest.mark.parametrize("mode", ["latency", "multi_objective"])
def test_numerical_algorithms_are_honest_scalarized_fallbacks(
    algorithm,
    mode,
):
    capability = (
        BrainFactory.objective_capabilities()
        .resolve(algorithm)
        .for_mode(mode)
    )

    assert capability.supported
    assert capability.support_level == "scalarized_fallback"
    assert not capability.sees_raw_objectives
    assert capability.consumes_archive_acquisition_score
    assert not capability.objective_aware


@pytest.mark.parametrize("algorithm", sorted(_SCALAR_ONLY_AGENTIC))
@pytest.mark.parametrize("mode", ["latency", "multi_objective"])
def test_scalar_only_agentic_algorithms_are_unsupported(algorithm, mode):
    capability = (
        BrainFactory.objective_capabilities()
        .resolve(algorithm)
        .for_mode(mode)
    )

    assert not capability.supported
    assert capability.support_level == "unsupported"
    assert not capability.sees_raw_objectives


def test_accuracy_mode_is_supported_for_every_factory_algorithm():
    registry = BrainFactory.objective_capabilities()

    assert all(
        registry.resolve(algorithm).for_mode("accuracy").supported
        for algorithm in registry.algorithms
    )


def test_matrix_serialization_is_deterministic_and_json_safe():
    registry = BrainFactory.objective_capabilities()

    first = registry.to_json()
    second = BrainFactory.objective_capability_matrix_json()

    assert first == second
    assert json.loads(first) == BrainFactory.objective_capability_matrix()
    assert json.loads(first)["schema_version"] == 1


def test_factory_rejects_unsupported_mode_before_constructing_brain():
    with patch("tao_automl.brain.factory.LLMBrain") as constructor:
        with pytest.raises(
            ValueError,
            match=(
                "does not support objective mode 'latency'.*"
                "only a scalar result"
            ),
        ):
            BrainFactory.create_brain(
                algorithm="llm",
                **_factory_kwargs(_selector_config("latency")),
            )

    constructor.assert_not_called()


def test_factory_labels_scalarized_fallback_on_constructed_brain(caplog):
    created = SimpleNamespace()
    with patch(
        "tao_automl.brain.factory.ASHA",
        return_value=created,
    ):
        brain = BrainFactory.create_brain(
            algorithm="asha",
            **_factory_kwargs(_selector_config("multi_objective")),
        )

    assert brain is created
    assert brain.algorithm_capability["algorithm"] == "asha"
    assert brain.objective_mode_capability["support_level"] == (
        "scalarized_fallback"
    )
    assert "does not model the raw objectives independently" in caplog.text


def test_factory_rejects_bayesian_acquisition_settings_for_fallback_algorithm():
    kwargs = _factory_kwargs(_selector_config("latency"))
    kwargs["acquisition_settings"] = {"calibration_points": 4}

    with patch("tao_automl.brain.factory.ASHA") as constructor:
        with pytest.raises(
            ValueError,
            match=(
                "objective_acquisition settings are supported only by.*"
                "'asha' would ignore them"
            ),
        ):
            BrainFactory.create_brain(
                algorithm="asha",
                **kwargs,
            )

    constructor.assert_not_called()


def test_factory_preserves_legacy_single_objective_agentic_search():
    created = SimpleNamespace()
    with patch(
        "tao_automl.brain.factory.LLMBrain",
        return_value=created,
    ):
        brain = BrainFactory.create_brain(
            algorithm="llm",
            **_factory_kwargs(
                parse_objective_config({"metric": "mAP50"})
            ),
        )

    assert brain.objective_mode_capability["mode"] == "single_objective"
    assert brain.objective_mode_capability["supported"]


def test_unknown_algorithm_fails_with_registered_aliases():
    with pytest.raises(
        ValueError,
        match="is not registered.*Supported names and aliases",
    ):
        BrainFactory.create_brain(
            algorithm="future_search",
            **_factory_kwargs(parse_objective_config({"metric": "mAP50"})),
        )


def test_registry_builder_fails_when_factory_algorithm_is_unclassified():
    definitions = BrainFactory.algorithm_definitions()
    definitions["future_search"] = {
        "aliases": ("future_search",),
        "implementation": "FutureSearch",
    }

    with pytest.raises(
        RuntimeError,
        match="unclassified BrainFactory algorithm.*future_search",
    ):
        build_algorithm_capability_registry(definitions)


def test_generic_multi_objective_agentic_search_is_rejected():
    generic_multi = parse_objective_config({
        "objectives": [
            {"metric": "accuracy", "direction": "maximize"},
            {"metric": "energy", "direction": "minimize"},
            {"metric": "memory", "direction": "minimize"},
        ]
    })

    with pytest.raises(
        ValueError,
        match="generic_multi_objective",
    ):
        BrainFactory.create_brain(
            algorithm="autoresearch",
            **_factory_kwargs(generic_multi),
        )
