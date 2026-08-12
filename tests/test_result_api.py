# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Result-API ergonomics: Mapping-style Recommendation, the stable run()
schema, the format_result renderer, and the explicit callback_error status."""

from unittest.mock import MagicMock

from tao_automl import format_result
from tao_automl.types import JobStates, Recommendation

from test_runner import _write_fake_skill


def test_dict_style_callback_semantics():
    """Callbacks treat recs as the spec dict; every dict idiom must work."""
    rec = Recommendation(3, {"train.optim.lr": 0.001, "model.dropout": 0.1}, "mAP50")

    assert rec.get("train.optim.lr", 1e-4) == 0.001
    assert rec.get("missing.key", 1e-4) == 1e-4
    assert rec.get("missing.key") is None
    assert rec["train.optim.lr"] == 0.001
    assert dict(rec) == {"train.optim.lr": 0.001, "model.dropout": 0.1}
    assert "model.dropout" in rec
    assert len(rec) == 2
    assert Recommendation(0, {}, "m"), "empty-spec rec must stay truthy"


def test_attribute_style_access_unchanged():
    rec = Recommendation(7, {"a": 1}, "accuracy")
    assert rec.id == 7
    assert rec.specs == {"a": 1}
    assert rec.status == JobStates.pending
    assert rec.result == 0.0
    rec.assign_job_id("job-7")
    assert rec.job_id == "job-7"
    assert list(rec.items()) == [("a", 1)]


def _fake_run(tmp_path, monkeypatch, final_eval_fn):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)

    class FakeAutoML:
        def __init__(self, *args, **kwargs):
            self.rec = Recommendation(0, {"train.num_epochs": 2}, "accuracy")
            self.complete = False

        def is_complete(self):
            return self.complete

        def next_recommendation(self):
            return [self.rec]

        def report_result(self, rec_id, metric_value, best_epoch=None, status="success"):
            self.rec.update_result(metric_value)
            self.rec.update_status(status)
            self.complete = True

        def get_best(self):
            return self.rec if self.rec.status == JobStates.success else None

        def get_progress(self):
            return {
                "completed": int(self.complete),
                "total": 1,
                "best_metric": self.rec.result,
                "best_rec_id": self.rec.id,
                "algorithm": "bayesian",
            }

        def get_history(self):
            return [self.rec]

    def fake_run_one_job(self, *args, **kwargs):
        kwargs["rec"].assign_job_id("train-job-0")
        return 0.62, "success"

    monkeypatch.setattr("tao_automl.AutoML", FakeAutoML)
    monkeypatch.setattr(AutoMLRunner, "_run_one_job", fake_run_one_job)

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    return runner.run(
        image="nvcr.io/test:1",
        automl_settings={
            "algorithm": "bayesian",
            "metric": "accuracy",
            "direction": "maximize",
            "automl_max_recommendations": 1,
            "run_final_evaluation": True,
        },
        baseline_fn=lambda specs: 0.5,
        final_eval_fn=final_eval_fn,
        workspace_path=str(tmp_path / "workspace"),
    )


def test_dict_style_final_eval_fn_succeeds(tmp_path, monkeypatch):
    """The exact live-failure shape: a callback that does rec.get(key, default)
    and dict(rec) on the best rec must run, not crash with TypeError."""
    def final_eval_fn(best_rec, train_job_id):
        assert best_rec.get("train.num_epochs", 10) == 2
        assert best_rec.get("not.a.key", "fallback") == "fallback"
        assert dict(best_rec) == best_rec.specs
        return 0.64

    result = _fake_run(tmp_path, monkeypatch, final_eval_fn)
    assert result["final_evaluation"]["status"] == "measured"
    assert result["final_evaluation"]["metric_value"] == 0.64


def test_crashing_final_eval_fn_reports_callback_error(tmp_path, monkeypatch):
    def final_eval_fn(best_rec, train_job_id):
        raise TypeError("Recommendation.get() takes 2 positional arguments but 3 were given")

    result = _fake_run(tmp_path, monkeypatch, final_eval_fn)
    final = result["final_evaluation"]
    assert final["status"] == "callback_error"
    assert "2 positional arguments" in final["failure_reason"]
    assert final["source"] == "final_eval_fn"
    assert result["best"]["metric_value"] == 0.62, "run result must survive"


def test_format_result_golden():
    result = {
        "best": {
            "rec_id": 4,
            "specs": {"train.optim.lr": 0.0003},
            "metric_value": 0.91,
            "objective_score": 0.91,
            "objective_values": {"mAP50": 0.91},
            "adjustments": [],
        },
        "progress": {
            "completed": 3, "total": 3, "best_metric": 0.91,
            "best_rec_id": 4, "algorithm": "bayesian",
        },
        "baseline": {
            "enabled": True, "metric_name": "mAP50", "metric_value": 0.85,
            "status": "measured",
            "comparison_to_best": {
                "delta": 0.06, "improved": True, "direction": "maximize",
            },
        },
        "final_evaluation": {
            "enabled": True, "metric_name": "mAP50", "metric_value": None,
            "status": "callback_error",
            "failure_reason": (
                "Recommendation.get() takes 2 positional arguments "
                "but 3 were given"
            ),
            "source": "final_eval_fn",
            "comparison_to_baseline": None,
        },
        "history": [
            {"rec_id": 0, "metric": 0.8, "objective_score": 0.8,
             "objective_values": {"mAP50": 0.8}, "status": "success",
             "failure_reason": None, "adjustments": []},
            {"rec_id": 1, "metric": 0.0, "objective_score": 0.0,
             "objective_values": {}, "status": "failure",
             "failure_reason": "job_creation_failed: boom", "adjustments": []},
            {"rec_id": 4, "metric": 0.91, "objective_score": 0.91,
             "objective_values": {"mAP50": 0.91}, "status": "success",
             "failure_reason": None, "adjustments": []},
        ],
    }

    expected = "\n".join([
        "AutoML result",
        "  algorithm=bayesian completed=3/3",
        "",
        "Best: rec 4 — mAP50=0.91",
        "  train.optim.lr = 0.0003",
        "",
        "History:",
        "   rec  status     mAP50",
        "     0  success    0.8",
        "     1  failure    0  (job_creation_failed: boom)",
        "     4  success    0.91",
        "",
        "Baseline: mAP50=0.85",
        "  best vs baseline: improved by 0.06 (maximize)",
        "Final evaluation: callback_error",
        "  final_eval_fn raised: Recommendation.get() takes 2 positional "
        "arguments but 3 were given",
    ])
    assert format_result(result) == expected


def test_run_docstring_schema_matches_result(tmp_path, monkeypatch):
    """The literal result dict and the documented stable schema must not
    drift: every actual key must be declared in run()'s docstring."""
    from tao_automl.runner import AutoMLRunner

    result = _fake_run(tmp_path, monkeypatch, final_eval_fn=lambda r, j: 0.64)
    doc = AutoMLRunner.run.__doc__

    assert set(result.keys()) == {
        "best", "progress", "baseline", "final_evaluation", "history",
        "algorithm_state",
    }
    assert set(result["best"].keys()) == {
        "rec_id", "job_id", "specs", "metric_value", "objective_score",
        "objective_values", "adjustments",
    }
    assert set(result["history"][0].keys()) == {
        "rec_id", "job_id", "metric", "objective_score", "objective_values",
        "status", "failure_reason", "adjustments",
    }
    for key in ("enabled", "metric_name", "metric_value", "status",
                "comparison_to_best"):
        assert key in result["baseline"]
    for key in ("enabled", "metric_name", "metric_value", "status",
                "comparison_to_baseline"):
        assert key in result["final_evaluation"]

    documented = (
        set(result.keys()) |
        set(result["best"].keys()) |
        set(result["history"][0].keys()) |
        {"enabled", "metric_name", "comparison_to_best",
         "comparison_to_baseline", "record_path", "source",
         "callback_error", "pareto_front", "completed", "total",
         "best_metric", "best_rec_id", "algorithm"}
    )
    for key in documented:
        assert f"``{key}``" in doc, f"run() docstring is missing ``{key}``"
