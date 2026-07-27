# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for Bayesian objective ingestion and reproducibility."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tao_automl.brain.base import (
    OBSERVATION_UTILITY_VERSION,
    _stable_context_seed,
)
from tao_automl.brain.bayesian import Bayesian
from tao_automl.brain.bfbo import BFBO
from tao_automl.brain.factory import AlgorithmParams, BrainFactory


_PARAMETERS = [{
    "parameter": "model.width",
    "value_type": "float",
    "default_value": 0.5,
    "valid_min": 0.0,
    "valid_max": 1.0,
    "valid_options": [],
    "option_weights": None,
    "math_cond": None,
    "parent_param": None,
    "depends_on": None,
}]


class _StateStore:
    def __init__(self, brain_state=None):
        self.brain_state = brain_state
        self.saved_state = None

    def get_job_specs(self, _job_id):
        return {"train": {"num_epochs": 1}}

    def get_custom_param_ranges(self, _experiment_id):
        return {}

    def get_brain_info(self, _job_id):
        return self.brain_state

    def save_brain_info(self, _job_id, state):
        self.saved_state = state


def _context(identifier="objective-test", random_seed=None):
    values = {
        "id": identifier,
        "handler_id": identifier,
    }
    if random_seed is not None:
        values["random_seed"] = random_seed
    return SimpleNamespace(**values)


def _recommendation(
    *,
    identifier=0,
    status="success",
    result=1.0,
    objective_values=None,
):
    return SimpleNamespace(
        id=identifier,
        status=status,
        result=result,
        objective_values=objective_values or {},
    )


@pytest.mark.parametrize("brain_class", [Bayesian, BFBO])
def test_minimize_metric_is_oriented_only_for_acquisition(brain_class):
    brain = brain_class(
        _context(),
        _StateStore(),
        "fake",
        _PARAMETERS,
        metric="latency_ms",
        direction="minimize",
    )
    recommendation = _recommendation(
        result=5.25,
        objective_values={"latency_ms": 5.25},
    )

    assert brain._observation_utility(recommendation) == pytest.approx(-5.25)
    assert recommendation.result == pytest.approx(5.25)


@pytest.mark.parametrize("brain_class", [Bayesian, BFBO])
@pytest.mark.parametrize(
    "recommendation",
    [
        _recommendation(status="failure", result=0.0),
        _recommendation(status="error", result=0.0),
        _recommendation(status="success", result=float("nan")),
        _recommendation(status="success", result=float("inf")),
    ],
)
def test_invalid_or_failed_observations_are_not_fitted(
    brain_class,
    recommendation,
):
    brain = brain_class(
        _context(),
        _StateStore(),
        "fake",
        _PARAMETERS,
        metric="latency_ms",
        direction="minimize",
    )
    brain.Xs = [np.array([0.25])]
    brain.update_gp = MagicMock()
    if isinstance(brain, Bayesian):
        brain.optimize_ei = MagicMock(return_value=np.array([0.5]))
    else:
        brain.optimize_ucb = MagicMock(return_value=np.array([0.5]))

    generated = brain.generate_recommendations([recommendation])

    assert len(generated) == 1
    assert brain.ys == []
    assert len(brain.Xs) == 1
    brain.update_gp.assert_not_called()


@pytest.mark.parametrize("brain_class", [Bayesian, BFBO])
def test_incomplete_multi_objective_observation_is_not_fitted(brain_class):
    brain = brain_class(
        _context(),
        _StateStore(),
        "fake",
        _PARAMETERS,
        metric="multi_objective_score",
        direction="maximize",
    )
    brain.Xs = [np.array([0.25])]
    brain.update_gp = MagicMock()
    recommendation = _recommendation(
        result=-0.5,
        objective_values={"mAP50": 0.6},
    )

    brain.generate_recommendations([recommendation])

    assert brain.ys == []
    brain.update_gp.assert_not_called()


@pytest.mark.parametrize(
    ("brain_class", "optimizer_name"),
    [(Bayesian, "optimize_ei"), (BFBO, "optimize_ucb")],
)
def test_successful_minimize_result_fits_oriented_utility(
    brain_class,
    optimizer_name,
):
    brain = brain_class(
        _context(),
        _StateStore(),
        "fake",
        _PARAMETERS,
        metric="latency_ms",
        direction="minimize",
    )
    brain.Xs = [np.array([0.25])]
    brain.update_gp = MagicMock()
    setattr(brain, optimizer_name, MagicMock(return_value=np.array([0.5])))
    recommendation = _recommendation(
        result=4.5,
        objective_values={"latency_ms": 4.5},
    )

    brain.generate_recommendations([recommendation])

    assert brain.ys == pytest.approx([-4.5])
    assert recommendation.result == pytest.approx(4.5)
    brain.update_gp.assert_called_once_with()


@pytest.mark.parametrize(
    ("brain_class", "optimizer_name"),
    [(Bayesian, "optimize_ei"), (BFBO, "optimize_ucb")],
)
def test_archive_rebuild_replaces_changed_earlier_utility(
    brain_class,
    optimizer_name,
):
    brain = brain_class(
        _context(),
        _StateStore(),
        "fake",
        _PARAMETERS,
        metric="multi_objective_score",
        direction="maximize",
    )
    brain.Xs = [np.array([0.1])]
    brain.update_gp = MagicMock()
    setattr(
        brain,
        optimizer_name,
        MagicMock(side_effect=[np.array([0.2]), np.array([0.3])]),
    )
    first = _recommendation(
        identifier=0,
        result=-0.1,
        objective_values={"mAP50": 0.6, "latency_ms": 5.0},
    )

    brain.generate_recommendations([first])
    assert brain.ys == pytest.approx([-0.1])

    # Archive-relative normalization recomputes the first scalar score when the
    # second measurement arrives. The next GP fit must use -0.7, not stale -0.1.
    first.result = -0.7
    second = _recommendation(
        identifier=1,
        result=-0.3,
        objective_values={"mAP50": 0.59, "latency_ms": 4.2},
    )
    brain.generate_recommendations([first, second])

    assert brain.ys == pytest.approx([-0.7, -0.3])
    assert [point.tolist() for point in brain.Xs] == [[0.1], [0.2], [0.3]]
    assert brain.update_gp.call_count == 2


@pytest.mark.parametrize(
    ("brain_class", "optimizer_name"),
    [(Bayesian, "optimize_ei"), (BFBO, "optimize_ucb")],
)
def test_failure_discards_pending_x_and_rebuilds_prior_utilities(
    brain_class,
    optimizer_name,
):
    brain = brain_class(
        _context(),
        _StateStore(),
        "fake",
        _PARAMETERS,
        metric="multi_objective_score",
        direction="maximize",
    )
    brain.Xs = [np.array([0.1]), np.array([0.2])]
    brain.ys = [-0.1]
    brain.update_gp = MagicMock()
    setattr(brain, optimizer_name, MagicMock(return_value=np.array([0.3])))
    first = _recommendation(
        identifier=0,
        result=-0.7,
        objective_values={"mAP50": 0.6, "latency_ms": 5.0},
    )
    failed = _recommendation(identifier=1, status="failure", result=0.0)

    brain.generate_recommendations([first, failed])

    assert brain.ys == pytest.approx([-0.7])
    assert [point.tolist() for point in brain.Xs] == [[0.1], [0.3]]
    brain.update_gp.assert_called_once_with()


@pytest.mark.parametrize(
    ("algorithm", "expected_class"),
    [("bayesian", "Bayesian"), ("bfbo", "BFBO")],
)
def test_factory_passes_metric_and_inferred_direction(
    algorithm,
    expected_class,
):
    context = _context()
    store = _StateStore()
    sentinel = object()
    target = f"tao_automl.brain.factory.{expected_class}"

    with patch(target, return_value=sentinel) as constructor:
        created = BrainFactory.create_brain(
            algorithm=algorithm,
            context=context,
            state_store=store,
            network="fake",
            parameters=_PARAMETERS,
            params=AlgorithmParams(),
            metric="latency_ms",
        )

    assert created is sentinel
    assert constructor.call_args.kwargs["metric"] == "latency_ms"
    assert constructor.call_args.kwargs["direction"] == "minimize"


def test_stable_context_seed_is_sha_derived_and_process_independent():
    context = _context("stable-session")
    expected = int.from_bytes(
        hashlib.sha256(b"stable-session").digest()[:8],
        byteorder="big",
        signed=False,
    ) % (2**32)

    assert _stable_context_seed(context) == expected
    assert _stable_context_seed(_context("stable-session")) == expected


def test_explicit_context_random_seed_is_honored():
    assert _stable_context_seed(_context(random_seed=12345)) == 12345


@pytest.mark.parametrize("brain_class", [Bayesian, BFBO])
def test_explicit_seed_controls_gp_optimizer(brain_class):
    brain = brain_class(
        _context(random_seed=12345),
        _StateStore(),
        "fake",
        _PARAMETERS,
    )

    assert brain.random_seed == 12345
    assert brain.gp.random_state == 12345


@pytest.mark.parametrize("bad_seed", [True, -1, 2**32, 1.5, "not-an-int"])
def test_invalid_explicit_context_random_seed_is_rejected(bad_seed):
    with pytest.raises((TypeError, ValueError)):
        _stable_context_seed(_context(random_seed=bad_seed))


@pytest.mark.parametrize("brain_class", [Bayesian, BFBO])
def test_same_context_produces_same_initial_recommendation(brain_class):
    first = brain_class(
        _context("repeatable"),
        _StateStore(),
        "fake",
        _PARAMETERS,
        metric="mAP50",
        direction="maximize",
    )
    first_recommendation = first.generate_recommendations([])

    second = brain_class(
        _context("repeatable"),
        _StateStore(),
        "fake",
        _PARAMETERS,
        metric="mAP50",
        direction="maximize",
    )
    second_recommendation = second.generate_recommendations([])

    assert first_recommendation == second_recommendation


@pytest.mark.parametrize("brain_class", [Bayesian, BFBO])
def test_legacy_minimize_state_is_oriented_and_nonfinite_pairs_are_dropped(
    brain_class,
):
    state = {
        "Xs": [[0.1], [0.2], [0.3]],
        "ys": [5.0, float("nan")],
    }
    store = _StateStore(brain_state=state)
    fit_target = (
        "tao_automl.brain.bayesian.GaussianProcessRegressor.fit"
        if brain_class is Bayesian
        else "tao_automl.brain.bfbo.GaussianProcessRegressor.fit"
    )

    with patch(fit_target, return_value=None):
        brain = brain_class.load_state(
            _context(),
            store,
            "fake",
            _PARAMETERS,
            metric="latency_ms",
            direction="minimize",
        )

    assert brain.ys == pytest.approx([-5.0])
    assert [point.tolist() for point in brain.Xs] == [[0.1], [0.3]]


@pytest.mark.parametrize(
    ("brain_class", "optimizer_name"),
    [(Bayesian, "optimize_ei"), (BFBO, "optimize_ucb")],
)
def test_resumed_brain_rebuilds_stale_archive_utilities(
    brain_class,
    optimizer_name,
):
    state = {
        "Xs": [[0.1], [0.2]],
        "ys": [-0.1],
        "metric": "multi_objective_score",
        "metric_direction": "maximize",
        "observation_utility_version": OBSERVATION_UTILITY_VERSION,
    }
    store = _StateStore(brain_state=state)
    fit_target = (
        "tao_automl.brain.bayesian.GaussianProcessRegressor.fit"
        if brain_class is Bayesian
        else "tao_automl.brain.bfbo.GaussianProcessRegressor.fit"
    )
    with patch(fit_target, return_value=None):
        brain = brain_class.load_state(
            _context(),
            store,
            "fake",
            _PARAMETERS,
            metric="multi_objective_score",
            direction="maximize",
        )

    brain.update_gp = MagicMock()
    setattr(brain, optimizer_name, MagicMock(return_value=np.array([0.3])))
    history = [
        _recommendation(
            identifier=0,
            result=-0.7,
            objective_values={"mAP50": 0.6, "latency_ms": 5.0},
        ),
        _recommendation(
            identifier=1,
            result=-0.3,
            objective_values={"mAP50": 0.59, "latency_ms": 4.2},
        ),
    ]

    brain.generate_recommendations(history)

    assert brain.ys == pytest.approx([-0.7, -0.3])
    assert [point.tolist() for point in brain.Xs] == [[0.1], [0.2], [0.3]]
    brain.update_gp.assert_called_once_with()


@pytest.mark.parametrize("brain_class", [Bayesian, BFBO])
def test_state_records_oriented_utility_contract(brain_class):
    store = _StateStore()
    brain = brain_class(
        _context(random_seed=7),
        store,
        "fake",
        _PARAMETERS,
        metric="latency_ms",
        direction="minimize",
    )
    brain.Xs = [np.array([0.25])]
    brain.ys = [-4.5]

    brain.save_state()

    assert store.saved_state["metric"] == "latency_ms"
    assert store.saved_state["metric_direction"] == "minimize"
    assert (
        store.saved_state["observation_utility_version"]
        == OBSERVATION_UTILITY_VERSION
    )
    assert store.saved_state["random_seed"] == 7
