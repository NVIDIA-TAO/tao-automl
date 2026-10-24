# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resume binds to durable controller and optimizer state before any trial recovery."""
import json

import numpy as np
import pytest

from tao_automl import AutoML
from test_bayesian_failure_handling import _schema


def make_search(workspace, *, resume=False, session_id=None, algorithm="bayesian"):
    schema = _schema()
    settings = {"algorithm": algorithm, "metric": "accuracy",
                "automl_max_recommendations": 4}
    if session_id is not None:
        settings["session_id"] = session_id
    return AutoML(workspace=str(workspace), network="external_failure_model",
                  train_specs=schema["default"], settings=settings,
                  search_schema=schema, resume=resume)


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("algorithm", ["bayesian", "bfbo"])
def test_resume_recovers_pending_recommendation_and_optimizer(tmp_path, explicit, algorithm):
    original = make_search(tmp_path, algorithm=algorithm)
    rec = original.next_recommendation()[0]
    identity = original._context.id
    point = original._controller.brain.Xs[0].copy()
    resumed = make_search(tmp_path, resume=True, algorithm=algorithm,
                          session_id=identity if explicit else None)
    assert resumed._context.id == identity
    assert [r.specs for r in resumed.get_history()] == [rec.specs]
    np.testing.assert_array_equal(resumed._controller.brain.Xs, [point])
    resumed.report_result(rec.id, 0.87, status="success")
    assert resumed.next_recommendation()[0].id == 1
    assert resumed._controller.brain.ys == [0.87]


@pytest.mark.parametrize("case", ["missing", "wrong", "ambiguous", "corrupt", "shape"])
def test_invalid_resume_fails_without_creating_or_overwriting_state(tmp_path, case):
    if case != "missing":
        make_search(tmp_path, session_id="original").next_recommendation()
    controllers = tmp_path / ".automl/controller"
    if case == "ambiguous":
        (controllers / "second.json").write_text("[]")
    if case == "corrupt":
        (controllers / "original.json").write_text("{truncated")
    if case == "shape":
        (controllers / "original.json").write_text("{}")
    before = {p.relative_to(tmp_path): p.read_bytes()
              for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="[Rr]esume|controller"):
        make_search(tmp_path, resume=True, session_id="wrong" if case == "wrong" else None)
    after = {p.relative_to(tmp_path): p.read_bytes()
             for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before


def test_explicit_identity_selects_existing_ambiguous_workspace(tmp_path):
    make_search(tmp_path, session_id="first").next_recommendation()
    make_search(tmp_path, session_id="second").next_recommendation()
    resumed = make_search(tmp_path, resume=True, session_id="second")
    assert resumed._context.id == "second"
    assert len(resumed.get_history()) == 1


@pytest.mark.parametrize("identity", ["", "../escape", "white space", 42])
def test_invalid_identity_cannot_escape_state_namespace(tmp_path, identity):
    with pytest.raises(ValueError, match="session_id"):
        make_search(tmp_path, session_id=identity)
    assert not (tmp_path / ".automl").exists()


def test_runner_rejects_missing_resume_before_baseline_or_backend_actions(tmp_path):
    from unittest.mock import MagicMock
    from tao_automl.runner import AutoMLRunner
    from test_runner import _write_fake_skill

    sdk, baseline = MagicMock(), MagicMock()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    workspace = tmp_path / "missing-run"
    with pytest.raises(ValueError, match="Resume requires"):
        runner.run(workspace_path=str(workspace), resume=True, baseline_fn=baseline,
                   automl_settings={"algorithm": "bayesian", "metric": "accuracy"})
    baseline.assert_not_called()
    sdk.create_job.assert_not_called()
    sdk.cancel_job.assert_not_called()
    assert not workspace.exists()
