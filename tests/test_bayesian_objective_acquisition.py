# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Production routing tests for mode-aware Bayesian acquisition."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tao_automl import AutoML
from tao_automl.brain.bayesian import Bayesian
from tao_automl.brain.factory import AlgorithmParams, BrainFactory
from tao_automl.objectives import parse_objective_config


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
    def __init__(self, brain_state=None, *, job_specs=None, custom_ranges=None):
        self.brain_state = brain_state
        self.saved_state = None
        self.job_specs = job_specs or {"train": {"num_epochs": 1}}
        self.custom_ranges = custom_ranges or {}

    def get_job_specs(self, _job_id):
        return copy.deepcopy(self.job_specs)

    def get_custom_param_ranges(self, _experiment_id):
        return copy.deepcopy(self.custom_ranges)

    def get_brain_info(self, _job_id):
        return self.brain_state

    def save_brain_info(self, _job_id, state):
        self.saved_state = copy.deepcopy(state)


def _context(identifier="native-acquisition", seed=314159):
    return SimpleNamespace(
        id=identifier,
        handler_id=identifier,
        random_seed=seed,
    )


def _objective_config(
    mode,
    *,
    reverse=False,
    retention=0.9,
    accuracy_tolerance=1e-12,
):
    objectives = [
        {"metric": "mAP50", "direction": "maximize"},
        {"metric": "latency", "direction": "minimize"},
    ]
    if reverse:
        objectives.reverse()
    return parse_objective_config({
        "objectives": objectives,
        "selection_mode": mode,
        "accuracy_metric": "mAP50",
        "latency_metric": "latency",
        "latency_accuracy_retention": retention,
        "accuracy_tolerance": accuracy_tolerance,
    })


def _brain(
    mode,
    *,
    calibration_points=2,
    reverse=False,
    store=None,
    accuracy_tolerance=1e-12,
):
    return Bayesian(
        _context(),
        store or _StateStore(),
        "fake",
        _PARAMETERS,
        metric="multi_objective_score",
        direction="maximize",
        objective_config=_objective_config(
            mode,
            reverse=reverse,
            accuracy_tolerance=accuracy_tolerance,
        ),
        acquisition_settings={
            "calibration_points": calibration_points,
            "xi": 0.02,
            "augmentation_rho": 1e-5,
        },
    )


def _observation(identifier, accuracy, latency, *, result=999.0):
    return SimpleNamespace(
        id=identifier,
        status="success",
        result=result,
        objective_values={
            "mAP50": accuracy,
            "latency": latency,
        },
    )


def test_factory_routes_objective_config_and_acquisition_settings_to_bayesian():
    config = _objective_config("latency")
    settings = {"calibration_points": 6}
    sentinel = object()
    with patch(
        "tao_automl.brain.factory.Bayesian",
        return_value=sentinel,
    ) as constructor:
        result = BrainFactory.create_brain(
            algorithm="bayesian",
            context=_context(),
            state_store=_StateStore(),
            network="fake",
            parameters=_PARAMETERS,
            params=AlgorithmParams(),
            metric="multi_objective_score",
            objective_config=config,
            acquisition_settings=settings,
        )

    assert result is sentinel
    assert constructor.call_args.kwargs["objective_config"] is config
    assert constructor.call_args.kwargs["acquisition_settings"] is settings


def test_automl_routes_mode_and_acquisition_profile_to_bayesian(tmp_path):
    automl = AutoML(
        workspace=str(tmp_path),
        network="cosmos-rl",
        train_specs={"train": {"epoch": 2, "optm_lr": 1e-6}},
        settings={
            "algorithm": "bayesian",
            "objectives": [
                {"metric": "val_accuracy", "direction": "maximize"},
                {"metric": "latency_ms", "direction": "minimize"},
            ],
            "selection_mode": "latency",
            "accuracy_metric": "val_accuracy",
            "latency_metric": "latency_ms",
            "objective_acquisition": {
                "calibration_points": 6,
                "xi": 0.03,
                "augmentation_rho": 2e-6,
            },
        },
        automl_hyperparameters=["train.optm_lr"],
        custom_param_ranges={
            "train.optm_lr": {
                "valid_min": 5e-7,
                "valid_max": 2e-6,
            }
        },
    )

    brain = automl._controller.brain
    assert brain.objective_acquisition_mode == "latency"
    assert brain.acquisition_settings == {
        "calibration_points": 6,
        "xi": 0.03,
        "augmentation_rho": 2e-6,
    }


def test_accuracy_acquisition_fits_raw_accuracy_not_selector_score():
    brain = _brain("accuracy")
    brain.generate_recommendations([])
    first = _observation(0, 0.61, 11.0, result=-12345.0)
    brain.generate_recommendations([first])
    second = _observation(1, 0.62, 10.0, result=12345.0)
    history = [first, second]
    brain.accuracy_gp.fit = MagicMock()
    brain._maximize_expected_improvement = MagicMock(
        return_value=np.asarray([0.75])
    )

    brain.generate_recommendations(history)

    fitted = brain.accuracy_gp.fit.call_args.args[1]
    assert fitted.tolist() == pytest.approx([0.61, 0.62])
    assert brain.ys == pytest.approx([0.61, 0.62])
    assert brain.acquisition_audit["active_method"] == (
        "accuracy_expected_improvement"
    )
    assert brain.acquisition_audit["selector_score_used"] is False


def test_all_modes_share_initial_design_then_route_to_distinct_acquisitions():
    brains = {
        mode: _brain(mode, calibration_points=3)
        for mode in ("accuracy", "latency", "multi_objective")
    }
    histories = {mode: [] for mode in brains}
    points = {mode: [] for mode in brains}
    values = (
        (0.55, 12.0),
        (0.60, 10.0),
        (0.58, 8.0),
    )
    for index, (accuracy, latency) in enumerate(values):
        for mode, brain in brains.items():
            brain.generate_recommendations(histories[mode])
            points[mode].append(brain.Xs[-1].copy())
            histories[mode].append(
                _observation(index, accuracy, latency)
            )

    assert np.asarray(points["accuracy"]) == pytest.approx(
        np.asarray(points["latency"])
    )
    assert np.asarray(points["accuracy"]) == pytest.approx(
        np.asarray(points["multi_objective"])
    )

    brains["accuracy"].accuracy_gp.fit = MagicMock()
    brains["accuracy"]._maximize_expected_improvement = MagicMock(
        return_value=np.asarray([0.1])
    )
    brains["latency"].accuracy_gp.fit = MagicMock()
    brains["latency"].latency_gp.fit = MagicMock()
    brains["latency"]._optimize_acquisition = MagicMock(
        return_value=np.asarray([0.2])
    )
    brains["multi_objective"].gp.fit = MagicMock()
    brains["multi_objective"]._maximize_expected_improvement = MagicMock(
        return_value=np.asarray([0.3])
    )

    for mode, brain in brains.items():
        brain.generate_recommendations(histories[mode])

    assert {
        mode: brain.acquisition_audit["active_method"]
        for mode, brain in brains.items()
    } == {
        "accuracy": "accuracy_expected_improvement",
        "latency": "constrained_latency_expected_improvement",
        "multi_objective": "parego_expected_improvement",
    }


def test_latency_acquisition_uses_independent_models_and_observed_reference():
    brain = _brain("latency", calibration_points=2)
    brain.generate_recommendations([])
    first = _observation(0, 0.70, 10.0, result=777.0)
    brain.generate_recommendations([first])
    second = _observation(1, 0.65, 8.0, result=-777.0)
    brain.accuracy_gp.fit = MagicMock()
    brain.latency_gp.fit = MagicMock()
    brain._optimize_acquisition = MagicMock(return_value=np.asarray([0.5]))

    brain.generate_recommendations([first, second])

    assert brain.accuracy_gp.fit.call_args.args[1].tolist() == pytest.approx(
        [0.70, 0.65]
    )
    assert brain.latency_gp.fit.call_args.args[1].tolist() == pytest.approx(
        [10.0, 8.0]
    )
    assert brain.ys == pytest.approx([-10.0, -8.0])
    audit = brain.acquisition_audit
    assert audit["active_method"] == "constrained_latency_expected_improvement"
    assert audit["accuracy_reference"] == pytest.approx(0.70)
    assert audit["accuracy_threshold"] == pytest.approx(0.63)
    assert audit["feasible_latency_incumbent"] == pytest.approx(8.0)
    assert audit["optimization_direction"] == {
        "accuracy": "constraint_maximize",
        "latency": "minimize",
    }


def test_latency_acquisition_uses_tolerance_adjusted_feasibility_boundary():
    brain = _brain(
        "latency",
        calibration_points=2,
        accuracy_tolerance=0.01,
    )
    brain.generate_recommendations([])
    first = _observation(0, 0.70, 10.0)
    brain.generate_recommendations([first])
    # Raw retained threshold is 0.63. This point is feasible only through
    # the configured 0.01 tolerance and must therefore be the incumbent.
    second = _observation(1, 0.625, 8.0)
    captured = {}

    def optimize(acquisition):
        acquisition(np.asarray([0.5]))
        return np.asarray([0.5])

    brain._optimize_acquisition = optimize
    with patch(
        "tao_automl.brain.bayesian.constrained_latency_ei",
        return_value=np.asarray([1.0]),
    ) as constrained:
        brain.generate_recommendations([first, second])
        captured.update(constrained.call_args.kwargs)

    assert captured["accuracy_threshold"] == pytest.approx(0.62)
    assert captured["feasible_latency_incumbent"] == pytest.approx(8.0)
    assert brain.acquisition_audit["accuracy_threshold"] == pytest.approx(
        0.63
    )
    assert brain.acquisition_audit[
        "accuracy_feasibility_boundary"
    ] == pytest.approx(0.62)


def test_accuracy_acquisition_uses_accuracy_without_latency_and_replays_resume():
    store = _StateStore()
    uninterrupted = _brain(
        "accuracy",
        calibration_points=2,
        store=store,
    )
    uninterrupted.generate_recommendations([])
    first = SimpleNamespace(
        id=0,
        status="success",
        result=0.60,
        objective_values={"mAP50": 0.60},
    )
    uninterrupted.generate_recommendations([first])
    uninterrupted.save_state()
    frozen_state = copy.deepcopy(store.saved_state)
    second = SimpleNamespace(
        id=1,
        status="success",
        result=0.70,
        objective_values={"mAP50": 0.70, "latency": -1.0},
    )

    uninterrupted_next = uninterrupted.generate_recommendations(
        [first, second]
    )
    uninterrupted_audit = uninterrupted.consume_last_recommendation_audits()[
        -1
    ]
    resumed = Bayesian.load_state(
        _context(),
        _StateStore(brain_state=frozen_state),
        "fake",
        _PARAMETERS,
        metric="multi_objective_score",
        direction="maximize",
        objective_config=_objective_config("accuracy"),
        acquisition_settings={
            "calibration_points": 2,
            "xi": 0.02,
            "augmentation_rho": 1e-5,
        },
    )
    resumed_next = resumed.generate_recommendations([first, second])
    resumed_audit = resumed.consume_last_recommendation_audits()[-1]

    assert resumed_next == uninterrupted_next
    assert resumed.ys == pytest.approx([0.60, 0.70])
    assert resumed_audit == uninterrupted_audit
    assert resumed_audit["objectives"] == [
        {"metric": "mAP50", "direction": "maximize"},
    ]
    assert resumed_audit["decision_state"]["observations"] == [
        {"candidate_id": "0", "accuracy": 0.60},
        {"candidate_id": "1", "accuracy": 0.70},
    ]


@pytest.mark.parametrize("mode", ["latency", "multi_objective"])
def test_latency_dependent_modes_reject_incomplete_or_nonpositive_pairs(mode):
    brain = _brain(mode, calibration_points=2)
    first_point = brain.generate_recommendations([])
    incomplete = SimpleNamespace(
        id=0,
        status="success",
        result=0.70,
        objective_values={"mAP50": 0.70},
    )
    second_point = brain.generate_recommendations([incomplete])

    assert second_point != first_point
    assert brain.acquisition_audit["observation_count"] == 0
    assert brain.acquisition_audit["calibration_design_index"] == 1


def test_multi_objective_acquisition_varies_parego_weights_deterministically():
    brain = _brain("multi_objective", calibration_points=2)
    brain.generate_recommendations([])
    first = _observation(0, 0.70, 12.0, result=1000.0)
    brain.generate_recommendations([first])
    second = _observation(1, 0.60, 8.0, result=-1000.0)
    brain.gp.fit = MagicMock()
    brain._maximize_expected_improvement = MagicMock(
        side_effect=[np.asarray([0.4]), np.asarray([0.6])]
    )

    brain.generate_recommendations([first, second])
    first_weights = brain.acquisition_audit["parego"]["weights"]
    third = _observation(2, 0.66, 10.0, result=999999.0)
    brain.generate_recommendations([first, second, third])
    second_weights = brain.acquisition_audit["parego"]["weights"]

    assert first_weights == {"accuracy": 0.5, "latency": 0.5}
    assert second_weights == {"accuracy": 0.25, "latency": 0.75}
    assert brain.acquisition_audit["selector_score_used"] is False
    assert brain.acquisition_audit["optimization_direction"] == {
        "accuracy": "maximize",
        "latency": "minimize",
    }


@pytest.mark.parametrize("mode", ["accuracy", "latency", "multi_objective"])
def test_objective_declaration_order_does_not_change_native_routing(mode):
    normal = _brain(mode, reverse=False)
    reversed_config = _brain(mode, reverse=True)

    assert normal.objective_acquisition_mode == mode
    assert reversed_config.objective_acquisition_mode == mode
    assert normal.accuracy_metric == reversed_config.accuracy_metric == "mAP50"
    assert normal.latency_metric == reversed_config.latency_metric == "latency"
    assert normal.acquisition_audit["method"] == (
        reversed_config.acquisition_audit["method"]
    )


def test_low_discrepancy_calibration_replays_across_resume():
    store = _StateStore()
    uninterrupted = _brain(
        "latency",
        calibration_points=4,
        store=store,
    )
    uninterrupted.generate_recommendations([])
    first = _observation(0, 0.55, 12.0)
    uninterrupted.generate_recommendations([first])
    uninterrupted.save_state()

    resumed_store = _StateStore(brain_state=store.saved_state)
    resumed = Bayesian.load_state(
        _context(),
        resumed_store,
        "fake",
        _PARAMETERS,
        metric="multi_objective_score",
        direction="maximize",
        objective_config=_objective_config("latency"),
        acquisition_settings={
            "calibration_points": 4,
            "xi": 0.02,
            "augmentation_rho": 1e-5,
        },
    )
    second = _observation(1, 0.60, 10.0)

    uninterrupted_next = uninterrupted.generate_recommendations([first, second])
    resumed_next = resumed.generate_recommendations([first, second])

    assert resumed_next == uninterrupted_next
    assert resumed.acquisition_audit["calibration_design"] == "scrambled_halton"
    assert resumed.acquisition_audit["calibration_design_seed"] == 314159
    assert resumed.acquisition_audit["calibration_design_index"] == 2
    assert resumed.acquisition_audit["recommendations_issued"] == 3


def test_failed_calibration_trial_advances_design_instead_of_repeating_point():
    brain = _brain("latency", calibration_points=4)
    first = brain.generate_recommendations([])
    first_point = brain.Xs[-1].copy()
    failed = SimpleNamespace(
        id=0,
        status="failure",
        result=0.0,
        objective_values={},
    )

    second = brain.generate_recommendations([failed])

    assert second != first
    assert brain.acquisition_audit["calibration_design_index"] == 1
    assert not np.array_equal(brain.Xs[-1], first_point)


@pytest.mark.parametrize("mode", ["accuracy", "latency", "multi_objective"])
def test_model_based_recommendation_replays_exactly_after_resume(mode):
    store = _StateStore()
    uninterrupted = _brain(
        mode,
        calibration_points=2,
        store=store,
    )
    uninterrupted.generate_recommendations([])
    first = _observation(0, 0.60, 12.0)
    uninterrupted.generate_recommendations([first])
    second = _observation(1, 0.58, 8.0)
    uninterrupted.generate_recommendations([first, second])
    uninterrupted.save_state()
    frozen_state = copy.deepcopy(store.saved_state)

    third = _observation(2, 0.59, 9.0)
    uninterrupted_next = uninterrupted.generate_recommendations(
        [first, second, third]
    )
    uninterrupted_audit = uninterrupted.acquisition_audit

    resumed = Bayesian.load_state(
        _context(),
        _StateStore(brain_state=frozen_state),
        "fake",
        _PARAMETERS,
        metric="multi_objective_score",
        direction="maximize",
        objective_config=_objective_config(mode),
        acquisition_settings={
            "calibration_points": 2,
            "xi": 0.02,
            "augmentation_rho": 1e-5,
        },
    )
    resumed_next = resumed.generate_recommendations([first, second, third])

    assert resumed_next == uninterrupted_next
    assert resumed.Xs[-1] == pytest.approx(uninterrupted.Xs[-1])
    assert resumed.acquisition_audit["active_method"] == (
        uninterrupted_audit["active_method"]
    )
    assert resumed.acquisition_audit["acquisition_index"] == (
        uninterrupted_audit["acquisition_index"]
    )
    assert resumed.consume_last_recommendation_audits()[-1][
        "rng_state_sha256"
    ] == uninterrupted.consume_last_recommendation_audits()[-1][
        "rng_state_sha256"
    ]


def test_resume_rejects_changed_objective_acquisition_configuration():
    store = _StateStore()
    brain = _brain("latency", calibration_points=4, store=store)
    brain.generate_recommendations([])
    brain.save_state()
    resumed_store = _StateStore(brain_state=store.saved_state)

    with pytest.raises(
        ValueError,
        match="different objective acquisition configuration",
    ):
        Bayesian.load_state(
            _context(),
            resumed_store,
            "fake",
            _PARAMETERS,
            metric="multi_objective_score",
            direction="maximize",
            objective_config=_objective_config("multi_objective"),
            acquisition_settings={
                "calibration_points": 4,
                "xi": 0.02,
                "augmentation_rho": 1e-5,
            },
        )


@pytest.mark.parametrize(
    ("context", "network", "parameters", "expected_field"),
    [
        (_context(seed=999), "fake", _PARAMETERS, "random_seed"),
        (_context(), "different-network", _PARAMETERS, "network"),
        (
            _context(),
            "fake",
            [{**_PARAMETERS[0], "valid_max": 2.0}],
            "search_space_sha256",
        ),
    ],
)
def test_resume_rejects_changed_seed_network_or_complete_search_space(
    context,
    network,
    parameters,
    expected_field,
):
    store = _StateStore()
    brain = _brain("latency", calibration_points=4, store=store)
    brain.generate_recommendations([])
    brain.save_state()

    with pytest.raises(ValueError, match=expected_field):
        Bayesian.load_state(
            context,
            _StateStore(brain_state=store.saved_state),
            network,
            parameters,
            metric="multi_objective_score",
            direction="maximize",
            objective_config=_objective_config("latency"),
            acquisition_settings={
                "calibration_points": 4,
                "xi": 0.02,
                "augmentation_rho": 1e-5,
            },
        )


def test_resume_rejects_changed_train_spec_and_custom_ranges():
    original_ranges = {"model.width": {"valid_min": 0.1, "valid_max": 0.9}}
    store = _StateStore(custom_ranges=original_ranges)
    brain = Bayesian(
        _context(),
        store,
        "fake",
        _PARAMETERS,
        metric="multi_objective_score",
        direction="maximize",
        objective_config=_objective_config("latency"),
        acquisition_settings={"calibration_points": 2},
    )
    brain.generate_recommendations([])
    brain.save_state()

    with pytest.raises(ValueError, match="train_spec_sha256"):
        Bayesian.load_state(
            _context(),
            _StateStore(
                brain_state=store.saved_state,
                job_specs={"train": {"num_epochs": 2}},
                custom_ranges=original_ranges,
            ),
            "fake",
            _PARAMETERS,
            metric="multi_objective_score",
            direction="maximize",
            objective_config=_objective_config("latency"),
            acquisition_settings={"calibration_points": 2},
        )

    with pytest.raises(ValueError, match="custom_ranges_sha256"):
        Bayesian.load_state(
            _context(),
            _StateStore(
                brain_state=store.saved_state,
                custom_ranges={
                    "model.width": {"valid_min": 0.2, "valid_max": 0.8}
                },
            ),
            "fake",
            _PARAMETERS,
            metric="multi_objective_score",
            direction="maximize",
            objective_config=_objective_config("latency"),
            acquisition_settings={"calibration_points": 2},
        )


def test_resume_rejects_legacy_bayesian_state_without_complete_signature():
    store = _StateStore()
    brain = _brain("accuracy", store=store)
    brain.save_state()
    legacy_state = copy.deepcopy(store.saved_state)
    legacy_state.pop("objective_acquisition_signature")

    with pytest.raises(ValueError, match="complete.*compatibility signature"):
        Bayesian.load_state(
            _context(),
            _StateStore(brain_state=legacy_state),
            "fake",
            _PARAMETERS,
            metric="multi_objective_score",
            direction="maximize",
            objective_config=_objective_config("accuracy"),
            acquisition_settings={
                "calibration_points": 2,
                "xi": 0.02,
                "augmentation_rho": 1e-5,
            },
        )


def test_automl_resume_validates_spec_before_any_overwrite(tmp_path):
    settings = {
        "algorithm": "bayesian",
        "session_id": "resume-construction-order",
        "random_seed": 17,
        "metric": "accuracy",
        "automl_max_recommendations": 2,
    }
    train_specs = {"train": {"epoch": 2, "optm_lr": 1e-6}}
    initial = AutoML(
        workspace=str(tmp_path),
        network="cosmos-rl",
        train_specs=train_specs,
        settings=settings,
        automl_hyperparameters=["train.optm_lr"],
    )
    assert initial.next_recommendation()

    with pytest.raises(ValueError, match="different training specification"):
        AutoML(
            workspace=str(tmp_path),
            network="cosmos-rl",
            train_specs={"train": {"epoch": 3, "optm_lr": 1e-6}},
            settings=settings,
            automl_hyperparameters=["train.optm_lr"],
            resume=True,
        )

    assert initial._state_store.get_job_specs(settings["session_id"]) == train_specs


def test_automl_resume_discovers_only_persisted_session_and_rejects_seed_change(
    tmp_path,
):
    settings = {
        "algorithm": "bayesian",
        "session_id": "discoverable-session",
        "random_seed": 23,
        "metric": "accuracy",
        "automl_max_recommendations": 2,
    }
    train_specs = {"train": {"epoch": 2, "optm_lr": 1e-6}}
    initial = AutoML(
        workspace=str(tmp_path),
        network="cosmos-rl",
        train_specs=train_specs,
        settings=settings,
        automl_hyperparameters=["train.optm_lr"],
    )
    assert initial.next_recommendation()

    discovered_settings = {**settings}
    discovered_settings.pop("session_id")
    resumed = AutoML(
        workspace=str(tmp_path),
        network="cosmos-rl",
        train_specs=train_specs,
        settings=discovered_settings,
        automl_hyperparameters=["train.optm_lr"],
        resume=True,
    )
    assert resumed._context.id == "discoverable-session"

    with pytest.raises(ValueError, match="random_seed"):
        AutoML(
            workspace=str(tmp_path),
            network="cosmos-rl",
            train_specs=train_specs,
            settings={**discovered_settings, "random_seed": 24},
            automl_hyperparameters=["train.optm_lr"],
            resume=True,
        )


def test_automl_resume_rejects_changed_schema_and_ranges_without_overwrite(
    tmp_path,
):
    schema = {
        "type": "object",
        "default": {"width": 0.5},
        "properties": {
            "width": {
                "type": "number",
                "default": 0.5,
                "minimum": 0.0,
                "maximum": 1.0,
                "automl_enabled": True,
            }
        },
    }
    settings = {
        "algorithm": "bayesian",
        "session_id": "schema-bound-session",
        "random_seed": 29,
        "metric": "accuracy",
        "automl_max_recommendations": 2,
    }
    ranges = {"width": {"valid_min": 0.1, "valid_max": 0.9}}
    initial = AutoML(
        workspace=str(tmp_path),
        network="external-network",
        train_specs={"width": 0.5},
        settings=settings,
        automl_hyperparameters=["width"],
        custom_param_ranges=ranges,
        search_schema=schema,
    )
    assert initial.next_recommendation()

    changed_schema = copy.deepcopy(schema)
    changed_schema["properties"]["width"]["maximum"] = 2.0
    with pytest.raises(ValueError, match="parameter_schema_sha256"):
        AutoML(
            workspace=str(tmp_path),
            network="external-network",
            train_specs={"width": 0.5},
            settings=settings,
            automl_hyperparameters=["width"],
            custom_param_ranges=ranges,
            search_schema=changed_schema,
            resume=True,
        )

    changed_ranges = {"width": {"valid_min": 0.2, "valid_max": 0.8}}
    with pytest.raises(ValueError, match="different custom parameter ranges"):
        AutoML(
            workspace=str(tmp_path),
            network="external-network",
            train_specs={"width": 0.5},
            settings=settings,
            automl_hyperparameters=["width"],
            custom_param_ranges=changed_ranges,
            search_schema=schema,
            resume=True,
        )
    assert initial._state_store.get_custom_param_ranges(
        settings["session_id"]
    ) == ranges


def test_acquisition_audit_is_not_mutable_by_callers():
    brain = _brain("latency")
    exposed = brain.acquisition_audit
    exposed["mode"] = "corrupted"

    assert brain.acquisition_audit["mode"] == "latency"


def test_recommendation_audits_are_frozen_and_consumed_in_emission_order():
    brain = _brain("latency", calibration_points=4)
    brain.generate_recommendations([])
    first = _observation(0, 0.55, 12.0)
    brain.generate_recommendations([first])

    audits = brain.consume_last_recommendation_audits()

    assert [item["recommendation_index"] for item in audits] == [0, 1]
    assert [item["calibration_or_acquisition_index"] for item in audits] == [
        0,
        1,
    ]
    assert all(item["acquisition_mode"] == "latency" for item in audits)
    assert all(
        item["acquisition_function"]
        == "deterministic_low_discrepancy_design"
        for item in audits
    )
    assert audits[1]["observation_summary"] == {
        "count": 1,
        "candidate_ids": ["0"],
        "objective_bounds": {
            "mAP50": {"minimum": 0.55, "maximum": 0.55},
            "latency": {"minimum": 12.0, "maximum": 12.0},
        },
    }
    assert len(audits[0]["rng_state_sha256"]) == 64
    assert brain.consume_last_recommendation_audits() == []


def test_latency_proposal_freezes_exact_decision_state_across_resume():
    store = _StateStore()
    brain = _brain("latency", calibration_points=2, store=store)
    brain.generate_recommendations([])
    brain.consume_last_recommendation_audits()
    first = _observation(0, 0.70, 10.0)
    brain.generate_recommendations([first])
    brain.consume_last_recommendation_audits()
    second = _observation(1, 0.65, 8.0)
    brain._optimize_acquisition = MagicMock(return_value=np.asarray([0.5]))

    brain.generate_recommendations([first, second])
    brain.save_state()

    resumed_store = _StateStore(brain_state=store.saved_state)
    resumed = Bayesian.load_state(
        _context(),
        resumed_store,
        "fake",
        _PARAMETERS,
        metric="multi_objective_score",
        direction="maximize",
        objective_config=_objective_config("latency"),
        acquisition_settings={
            "calibration_points": 2,
            "xi": 0.02,
            "augmentation_rho": 1e-5,
        },
    )
    audits = resumed.consume_last_recommendation_audits()

    assert len(audits) == 1
    state = audits[0]["decision_state"]
    assert state["active_method"] == (
        "constrained_latency_expected_improvement"
    )
    assert state["accuracy_reference"] == pytest.approx(0.70)
    assert state["accuracy_threshold"] == pytest.approx(0.63)
    assert state["feasible_latency_incumbent"] == pytest.approx(8.0)
    assert state["feasible_observation_count"] == 2
    assert state["optimization_direction"] == {
        "accuracy": "constraint_maximize",
        "latency": "minimize",
    }


def test_multi_objective_proposal_freezes_parego_geometry():
    brain = _brain("multi_objective", calibration_points=2)
    brain.generate_recommendations([])
    brain.consume_last_recommendation_audits()
    first = _observation(0, 0.70, 12.0)
    brain.generate_recommendations([first])
    brain.consume_last_recommendation_audits()
    second = _observation(1, 0.60, 8.0)
    brain._maximize_expected_improvement = MagicMock(
        return_value=np.asarray([0.4])
    )

    brain.generate_recommendations([first, second])
    audit = brain.consume_last_recommendation_audits()[0]

    assert audit["decision_state"]["parego"] == {
        "method": "parego_augmented_chebyshev",
        "iteration": 0,
        "weights": {"accuracy": 0.5, "latency": 0.5},
        "normalization_bounds": {
            "accuracy_min": pytest.approx(0.60),
            "accuracy_max": pytest.approx(0.70),
            "latency_min": pytest.approx(8.0),
            "latency_max": pytest.approx(12.0),
            "accuracy_nadir_source": "pareto_front",
            "latency_nadir_source": "pareto_front",
        },
        "augmentation_rho": pytest.approx(1e-5),
    }


@pytest.mark.parametrize(
    "settings",
    [
        {"calibration_points": True},
        {"calibration_points": 1},
        {"xi": -0.1},
        {"augmentation_rho": float("nan")},
        {"unknown": 1},
    ],
)
def test_invalid_objective_acquisition_settings_are_rejected(settings):
    with pytest.raises((TypeError, ValueError)):
        Bayesian(
            _context(),
            _StateStore(),
            "fake",
            _PARAMETERS,
            metric="multi_objective_score",
            objective_config=_objective_config("latency"),
            acquisition_settings=settings,
        )
