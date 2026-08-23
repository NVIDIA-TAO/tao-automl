# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failed trials must not contaminate the Bayesian GP observations."""

import numpy as np
import pytest

from tao_automl import AutoML


def _schema():
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


def _make_automl(tmp_path, algorithm="bayesian", resume=False):
    schema = _schema()
    return AutoML(
        workspace=str(tmp_path),
        network="external_failure_model",
        train_specs=schema["default"],
        settings={
            "algorithm": algorithm,
            "metric": "accuracy",
            "automl_max_recommendations": 5,
            "session_id": f"resume-{algorithm}",
            "experiment_id": f"resume-{algorithm}",
        },
        search_schema=schema,
        resume=resume,
    )


@pytest.mark.parametrize("algorithm", ["bayesian", "bfbo"])
def test_pending_design_point_survives_controller_resume(tmp_path, algorithm):
    """A recovered result must retain the X issued before controller death."""
    automl = _make_automl(tmp_path, algorithm=algorithm)
    rec = automl.next_recommendation()[0]
    issued_x = np.array(automl._controller.brain.Xs[0])

    resumed = _make_automl(tmp_path, algorithm=algorithm, resume=True)
    resumed_brain = resumed._controller.brain
    assert len(resumed_brain.Xs) == 1
    np.testing.assert_allclose(resumed_brain.Xs[0], issued_x)

    resumed.report_result(rec.id, 0.75, status="success")
    next_rec = resumed.next_recommendation()

    assert next_rec
    assert resumed_brain.ys == [0.75]
    assert len(resumed_brain.Xs) == 2


def test_failed_trial_is_not_recorded_as_gp_observation(tmp_path):
    """An infra-failed job reports a synthetic 0.0; the GP must never see it."""
    automl = _make_automl(tmp_path)
    brain = automl._controller.brain

    rec1 = automl.next_recommendation()[0]
    assert len(brain.Xs) == 1 and brain.ys == []

    automl.report_result(rec1.id, 0.0, status="failure")
    recs = automl.next_recommendation()

    assert recs, "search must continue after a failed trial"
    # The failed design point was dropped, not observed as 0.0: the only
    # pending Xs entry belongs to the new recommendation.
    assert brain.ys == []
    assert len(brain.Xs) == len(brain.ys) + 1


def test_search_alternates_failure_and_success(tmp_path):
    automl = _make_automl(tmp_path)
    brain = automl._controller.brain

    rec1 = automl.next_recommendation()[0]
    automl.report_result(rec1.id, 0.0, status="failure")

    rec2 = automl.next_recommendation()[0]
    automl.report_result(rec2.id, 0.75, status="success")

    rec3 = automl.next_recommendation()
    assert rec3, "search must continue after a success following a failure"
    # Only the successful trial is an observation; rec3's point is pending.
    assert brain.ys == [0.75]
    assert len(brain.Xs) == len(brain.ys) + 1
