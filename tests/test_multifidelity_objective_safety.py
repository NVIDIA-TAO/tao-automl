# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for objective direction and safe trial promotion."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from tao_automl import AutoML
from tao_automl.brain.base import AutoMLAlgorithmBase
from tao_automl.brain.bohb import BOHB
from tao_automl.brain.dehb import DEHB
from tao_automl.brain.hyperband import HyperBand
from tao_automl.brain.hyperband_es import HyperBandES


_SEARCH_RANGES = {
    "train.optm_lr": {"valid_min": 5e-7, "valid_max": 2e-6},
}
_TRAIN_SPEC = {"train": {"epoch": 10, "optm_lr": 1e-6}}


def _algorithm_settings(algorithm: str) -> dict:
    settings = {"algorithm": algorithm, "metric": "latency_ms"}
    if algorithm in {"hyperband", "bohb", "dehb", "hyperband_es"}:
        settings.update({
            "automl_max_epochs": 2,
            "automl_reduction_factor": 2,
            "epoch_multiplier": 1,
        })
    if algorithm == "asha":
        settings.update({
            "automl_max_epochs": 2,
            "automl_reduction_factor": 2,
            "epoch_multiplier": 1,
            "automl_max_concurrent": 2,
            "automl_max_trials": 2,
            "automl_min_top_configs": 1,
        })
    if algorithm == "pbt":
        settings.update({
            "automl_population_size": 2,
            "automl_max_generations": 2,
            "automl_eval_interval": 1,
        })
    return settings


@pytest.mark.parametrize(
    "algorithm",
    [
        "hyperband",
        "bohb",
        "asha",
        "dehb",
        "pbt",
        "hyperband_es",
        "llm",
        "hybrid",
        "autoresearch",
    ],
)
def test_non_bayesian_algorithms_use_generic_latency_direction(
    tmp_path,
    algorithm,
):
    automl = AutoML(
        workspace=str(tmp_path / algorithm),
        network="cosmos-rl",
        train_specs=_TRAIN_SPEC,
        settings=_algorithm_settings(algorithm),
        automl_hyperparameters=["train.optm_lr"],
        custom_param_ranges=_SEARCH_RANGES,
    )

    assert automl._controller.brain.reverse_sort is False


@pytest.mark.parametrize(
    ("status", "result", "expected"),
    [
        ("success", -0.25, -0.25),
        ("done", 0.0, 0.0),
        ("failure", 0.0, None),
        ("error", 0.0, None),
        ("canceled", 0.0, None),
        ("pending", -1.0, None),
        ("success", float("nan"), None),
        ("success", float("inf"), None),
        ("success", True, None),
    ],
)
def test_completed_observation_gate(status, result, expected):
    value = AutoMLAlgorithmBase._completed_observation_value(
        SimpleNamespace(status=status, result=result)
    )

    assert value == expected


def _multi_objective_settings(algorithm: str) -> dict:
    settings = {
        "algorithm": algorithm,
        "objectives": [
            {"metric": "val_accuracy", "direction": "maximize"},
            {"metric": "latency_ms", "direction": "minimize"},
        ],
        "selection_mode": "multi_objective",
        "accuracy_metric": "val_accuracy",
        "latency_metric": "latency_ms",
    }
    if algorithm in {"hyperband", "bohb", "dehb"}:
        settings.update({
            "automl_max_epochs": 4,
            "automl_reduction_factor": 2,
            "epoch_multiplier": 1,
        })
    if algorithm == "pbt":
        settings.update({
            "automl_population_size": 3,
            "automl_max_generations": 2,
            "automl_eval_interval": 1,
        })
    return settings


def _report_negative_moo_archive(automl, recommendations):
    recommendations = sorted(recommendations, key=lambda rec: rec.id)
    observations = (
        {"val_accuracy": 0.90, "latency_ms": 10.0},
        {"val_accuracy": 0.80, "latency_ms": 5.0},
        {"val_accuracy": 0.70, "latency_ms": 12.0},
    )
    for rec, values in zip(recommendations[:3], observations):
        rec.assign_job_id(f"job-{rec.id}")
        automl.report_result(rec.id, values, status="success")
    failed = recommendations[-1]
    failed.assign_job_id(f"job-{failed.id}")
    automl.report_result(failed.id, 0.0, status="failure")
    return recommendations


@pytest.mark.parametrize("algorithm", ["hyperband", "bohb", "dehb"])
def test_failed_zero_never_beats_negative_valid_moo_during_promotion(
    tmp_path,
    algorithm,
):
    automl = AutoML(
        workspace=str(tmp_path / algorithm),
        network="cosmos-rl",
        train_specs=_TRAIN_SPEC,
        settings=_multi_objective_settings(algorithm),
        automl_hyperparameters=["train.optm_lr"],
        custom_param_ranges=_SEARCH_RANGES,
    )
    first_rung = automl.next_recommendation()
    assert len(first_rung) == 4
    recommendations = _report_negative_moo_archive(automl, first_rung)

    assert recommendations[-1].result == 0.0
    assert all(rec.result < 0.0 for rec in recommendations[:3])

    promoted = automl.next_recommendation()

    assert {rec.id for rec in promoted} == {0, 1}
    assert recommendations[-1].id not in {rec.id for rec in promoted}


def test_pbt_failed_zero_is_not_an_exploitation_source_for_negative_moo(
    tmp_path,
):
    automl = AutoML(
        workspace=str(tmp_path / "pbt"),
        network="cosmos-rl",
        train_specs=_TRAIN_SPEC,
        settings=_multi_objective_settings("pbt"),
        automl_hyperparameters=["train.optm_lr"],
        custom_param_ranges=_SEARCH_RANGES,
    )
    population = sorted(automl.next_recommendation(), key=lambda rec: rec.id)
    assert len(population) == 3

    population[0].assign_job_id("job-0")
    population[1].assign_job_id("job-1")
    population[2].assign_job_id("job-2")
    automl.report_result(
        population[0].id,
        {"val_accuracy": 0.90, "latency_ms": 10.0},
        status="success",
    )
    automl.report_result(
        population[1].id,
        {"val_accuracy": 0.80, "latency_ms": 5.0},
        status="success",
    )
    automl.report_result(population[2].id, 0.0, status="failure")

    assert population[2].result == 0.0
    assert population[0].result < 0.0
    assert population[1].result < 0.0

    resumed = {
        rec.id: rec
        for rec in automl.next_recommendation()
    }

    assert set(resumed) == {0, 1, 2}
    assert resumed[2].resume_from_job_id in {"job-0", "job-1"}
    assert all(rec.resume_from_job_id != "job-2" for rec in resumed.values())


def test_pbt_all_failed_population_stops_without_exploitation(tmp_path):
    automl = AutoML(
        workspace=str(tmp_path / "pbt-all-failed"),
        network="cosmos-rl",
        train_specs=_TRAIN_SPEC,
        settings=_algorithm_settings("pbt"),
        automl_hyperparameters=["train.optm_lr"],
        custom_param_ranges=_SEARCH_RANGES,
    )
    population = automl.next_recommendation()
    for rec in population:
        rec.assign_job_id(f"job-{rec.id}")
        automl.report_result(rec.id, 0.0, status="failure")

    assert automl.next_recommendation() == []
    assert automl.is_complete()
    assert automl.get_best() is None


class _BrainStateStore:
    def __init__(self, state=None):
        self.state = copy.deepcopy(state)
        self.spec = {"train": {"epoch": 4}}

    def get_job_specs(self, _context_id):
        return copy.deepcopy(self.spec)

    def save_job_specs(self, _context_id, spec):
        self.spec = copy.deepcopy(spec)

    def get_custom_param_ranges(self, _handler_id):
        return {}

    def save_brain_info(self, _context_id, state):
        self.state = copy.deepcopy(state)

    def get_brain_info(self, _context_id):
        return copy.deepcopy(self.state)


def _promotion_history():
    results = (0.90, 0.80, 0.70, 0.0)
    statuses = ("success", "success", "success", "failure")
    return [
        SimpleNamespace(
            id=index,
            status=statuses[index],
            result=results[index],
            specs={"model.width": 0.1 * (index + 1)},
            job_id=f"job-{index}",
        )
        for index in range(4)
    ]


@pytest.mark.parametrize("brain_class", [HyperBand, HyperBandES, BOHB, DEHB])
def test_failure_reduced_promotion_schedule_replays_after_resume(brain_class):
    context = SimpleNamespace(
        id=f"{brain_class.__name__}-resume",
        handler_id=f"{brain_class.__name__}-resume",
        random_seed=42,
    )
    store = _BrainStateStore()
    constructor_args = (
        context,
        store,
        "fake",
        [],
        4,
        2,
        1,
    )
    brain = brain_class(*constructor_args, metric="val_accuracy")
    brain.bracket = "0"
    brain.sh_iter = 1
    brain.expt_iter = 0
    history = _promotion_history()

    first = brain._generate_one_recommendation(history)
    assert first.id == 0
    assert brain.ni["0"][1] == 2
    brain.save_state()
    frozen_state = copy.deepcopy(store.state)

    uninterrupted_second = brain._generate_one_recommendation(history)

    resumed_store = _BrainStateStore(state=frozen_state)
    resumed = brain_class.load_state(
        context,
        resumed_store,
        "fake",
        [],
        4,
        2,
        1,
        metric="val_accuracy",
    )
    resumed_second = resumed._generate_one_recommendation(history)

    assert resumed.ni == brain.ni
    assert resumed_second.id == uninterrupted_second.id == 1
    assert resumed_second.specs == uninterrupted_second.specs
    assert resumed_second.resume_from_job_id == (
        uninterrupted_second.resume_from_job_id
    )
    assert resumed.expt_iter == brain.expt_iter == 2


def test_hyperband_es_resume_rejects_changed_early_stop_policy():
    context = SimpleNamespace(
        id="hyperband-es-policy",
        handler_id="hyperband-es-policy",
        random_seed=42,
    )
    store = _BrainStateStore()
    brain = HyperBandES(
        context,
        store,
        "fake",
        [],
        4,
        2,
        1,
        early_stop_threshold=0.8,
        min_early_stop_epochs=3,
        metric="val_accuracy",
    )
    brain.save_state()

    with pytest.raises(ValueError, match="different early-stopping"):
        HyperBandES.load_state(
            context,
            store,
            "fake",
            [],
            4,
            2,
            1,
            early_stop_threshold=0.9,
            min_early_stop_epochs=3,
            metric="val_accuracy",
        )
