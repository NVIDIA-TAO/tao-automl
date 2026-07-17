# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for persistent LLM recommendation decisions."""

from tao_automl.brain.llm_brain import LLMBrain
from tao_automl.types import JobStates, Recommendation


def _completed_rec(identifier, metric, config):
    rec = Recommendation(identifier, config, "accuracy")
    rec.update_result(metric)
    rec.update_status(JobStates.success)
    return rec


def test_llm_records_final_result_and_keep_discard_decisions():
    brain = LLMBrain.__new__(LLMBrain)
    brain.experiment_history = []
    brain.best_config = None
    brain.best_metric = None
    brain.reverse_sort = True

    rec0 = _completed_rec(0, 50.0, {"train.optim.lr": 5e-4})
    brain.on_recommendation_result(rec0, [rec0])

    rec1 = _completed_rec(1, 0.0, {"train.optim.lr": 1e-4})
    brain.on_recommendation_result(rec1, [rec0, rec1])

    assert [entry["rec_id"] for entry in brain.experiment_history] == [0, 1]
    assert [entry["decision"] for entry in brain.experiment_history] == ["keep", "discard"]
    assert brain.best_metric == 50.0
    assert brain.best_config == {"train.optim.lr": 5e-4}
