"""Regression tests for BOHB observation encoding."""

from types import SimpleNamespace

import numpy as np

from tao_automl.brain.bohb import BOHB
from tao_automl.types import JobStates


def _brain():
    brain = BOHB.__new__(BOHB)
    brain.custom_ranges = {}
    brain.parent_params = {}
    return brain


def test_list_observations_do_not_collapse_to_one_value():
    brain = _brain()
    parameter = {"parameter": "train.optim.lr_steps", "value_type": "list"}

    first = brain._normalize_observation_value(parameter, [1, 1, 1])
    second = brain._normalize_observation_value(parameter, [1, 1, 1, 1])

    assert first != second
    assert first == brain._normalize_observation_value(parameter, [1, 1, 1])


def test_categorical_observations_use_option_position():
    brain = _brain()
    parameter = {
        "parameter": "model.backbone",
        "value_type": "categorical",
        "valid_options": ["small", "medium", "large"],
    }

    assert brain._normalize_observation_value(parameter, "small") == 0.0
    assert brain._normalize_observation_value(parameter, "medium") == 0.5
    assert brain._normalize_observation_value(parameter, "large") == 1.0


def test_integer_observations_use_numeric_range():
    brain = _brain()
    parameter = {
        "parameter": "train.batch_size",
        "value_type": "int",
        "valid_min": 8,
        "valid_max": 24,
    }

    assert brain._normalize_observation_value(parameter, 8) == 0.0
    assert brain._normalize_observation_value(parameter, 16) == 0.5
    assert brain._normalize_observation_value(parameter, 24) == 1.0


def test_kde_sampling_still_resamples_and_clips():
    class FakeKDE:
        def resample(self, n_samples):
            assert n_samples == 2
            return np.array([[-0.2, 1.3], [0.25, 0.75]])

    samples = _brain()._sample_from_kde(FakeKDE(), 2)

    np.testing.assert_array_equal(samples, [[0.0, 0.25], [1.0, 0.75]])


def test_warmup_batch_retries_duplicate_configuration():
    brain = _brain()
    brain.parameters = [{"parameter": "steps", "value_type": "list"}]
    brain.expt_iter = 2
    generated = iter([{"steps": [1, 1, 1, 1]}, {"steps": [1, 1, 1]}])
    brain._generate_one_recommendation = lambda history: next(generated)

    rec = brain._generate_unique_recommendation([], [{"steps": [1, 1, 1, 1]}])

    assert rec == {"steps": [1, 1, 1]}
    assert brain.expt_iter == 1


def test_successful_zero_metrics_are_recorded_as_observations():
    brain = _brain()
    brain.parameters = [{"parameter": "queries", "value_type": "int", "valid_min": 10, "valid_max": 50}]
    brain.observations = []
    history = [
        SimpleNamespace(status=JobStates.success, result=0.0, specs={"queries": 20}),
        SimpleNamespace(status=JobStates.success, result=0.0, specs={"queries": 40}),
    ]

    brain._update_observations(history)

    assert len(brain.observations) == 2
    assert [observation[1] for observation in brain.observations] == [0.0, 0.0]
