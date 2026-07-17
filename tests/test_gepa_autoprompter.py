# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the TAO-owned GEPA batch Auto-Prompter."""

from types import SimpleNamespace

import pytest

from tao_automl.gepa_autoprompter import (
    GEPAutoPrompter,
    GEPAReflectionLM,
    TAOActionBatchRunner,
    TAOGEPAAdapter,
)


def test_reflection_lm_adapts_tao_client_and_system_prompt():
    calls = []

    class Client:
        def chat(self, messages, json_mode=False):
            calls.append((messages, json_mode))
            return SimpleNamespace(ok=True, content="proposal", error=None)

    lm = GEPAReflectionLM(Client(), system_prompt="Stay general.")

    assert lm("Improve this prompt") == "proposal"
    assert calls == [([
        {"role": "system", "content": "Stay general."},
        {"role": "user", "content": "Improve this prompt"},
    ], False)]


def test_reflection_lm_raises_on_tao_client_failure():
    client = SimpleNamespace(chat=lambda messages, json_mode=False: SimpleNamespace(
        ok=False, content="", error="endpoint unavailable"
    ))

    with pytest.raises(RuntimeError, match="endpoint unavailable"):
        GEPAReflectionLM(client)("prompt")


def test_reflection_lm_preserves_multimodal_content_parts():
    calls = []

    class Client:
        def chat(self, messages, json_mode=False):
            calls.append((messages, json_mode))
            return SimpleNamespace(ok=True, content="proposal", error=None)

    visual_prompt = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Review the failure."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
        ],
    }]
    lm = GEPAReflectionLM(Client(), system_prompt="Stay general.")

    assert lm(visual_prompt) == "proposal"
    assert calls == [([{"role": "system", "content": "Stay general."}, *visual_prompt], False)]


def test_action_batch_runner_applies_dotted_candidate_without_mutating_base_specs():
    base = {
        "dataset": {"system_prompt": "seed"},
        "vision": {"nframes": 8},
    }
    calls = []

    def evaluate_action(specs, items):
        calls.append((specs, items))
        return [f"answer-{item['id']}" for item in items]

    runner = TAOActionBatchRunner(
        base,
        evaluate_action,
        value_coercers={"vision.nframes": int},
    )
    outputs = runner.run_batch(
        {"dataset.system_prompt": "reflected", "vision.nframes": "16"},
        [{"id": "a"}, {"id": "b"}],
    )

    assert outputs == ["answer-a", "answer-b"]
    assert calls[0][0] == {
        "dataset": {"system_prompt": "reflected"},
        "vision": {"nframes": 16},
    }
    assert base == {
        "dataset": {"system_prompt": "seed"},
        "vision": {"nframes": 8},
    }


def test_adapter_batches_once_and_builds_leak_free_reflection_records():
    class Runner:
        def __init__(self):
            self.calls = []

        def run_batch(self, candidate, items):
            self.calls.append((candidate, items))
            return ["Yes", "No"]

    runner = Runner()

    def metric(output, gold):
        return (
            float(output == gold),
            {
                "comment": "Re-check temporal order.",
                "video_id": "private-video",
                "gold": gold,
            },
            None,
        )

    adapter = TAOGEPAAdapter(
        runner,
        metric,
        fixed_candidate={"vision.nframes": "8"},
    )
    items = [
        {"id": "private-a", "video": "/secret/a.mp4", "query": "Did it stop?", "gold": "Yes"},
        {"id": "private-b", "video": "/secret/b.mp4", "query": "Did it turn?", "gold": "Yes"},
    ]
    evaluated = adapter.evaluate(items, {"dataset.system_prompt": "prompt"}, capture_traces=True)
    records = adapter.make_reflective_dataset(
        {"dataset.system_prompt": "prompt"},
        evaluated,
        ["dataset.system_prompt"],
    )["dataset.system_prompt"]

    assert len(runner.calls) == 1
    assert runner.calls[0][0] == {
        "vision.nframes": "8",
        "dataset.system_prompt": "prompt",
    }
    assert evaluated.scores == [1.0, 0.0]
    assert records[0]["Inputs"] == {"query": "Did it stop?"}
    serialized = repr(records)
    assert "private-a" not in serialized
    assert "private-video" not in serialized
    assert "/secret" not in serialized
    assert "'gold'" not in serialized


def test_adapter_attaches_visual_evidence_only_to_failed_vision_components():
    evidence_calls = []

    def evidence(item, candidate):
        evidence_calls.append((item["id"], dict(candidate)))
        return {"t=1.0s": "frame-one", "t=2.0s": "frame-two"}

    adapter = TAOGEPAAdapter(
        SimpleNamespace(run_batch=lambda candidate, items: ["A", "B"]),
        lambda output, gold: (float(output == gold), "generic feedback", None),
        fixed_candidate={"vision.nframes": "8"},
        reflection_evidence_fn=evidence,
        vision_components=["dataset.system_prompt"],
    )
    items = [
        {"id": "private-a", "video": "/secret/a.mp4", "query": "First?", "gold": "A"},
        {"id": "private-b", "video": "/secret/b.mp4", "query": "Second?", "gold": "A"},
    ]
    evaluated = adapter.evaluate(
        items,
        {"dataset.system_prompt": "prompt"},
        capture_traces=True,
    )
    records = adapter.make_reflective_dataset(
        {"dataset.system_prompt": "prompt"},
        evaluated,
        ["dataset.system_prompt", "text.summary_prompt"],
    )

    assert evidence_calls == [(
        "private-b",
        {"vision.nframes": "8", "dataset.system_prompt": "prompt"},
    )]
    assert "Visual Evidence" not in records["dataset.system_prompt"][0]
    assert records["dataset.system_prompt"][1]["Visual Evidence"] == {
        "t=1.0s": "frame-one",
        "t=2.0s": "frame-two",
    }
    assert "Visual Evidence" not in records["text.summary_prompt"][1]
    serialized = repr(records)
    assert "private-b" not in serialized
    assert "/secret" not in serialized
    assert "'gold'" not in serialized


def test_joint_adapter_uses_bounded_history_aware_config_proposals():
    adapter = TAOGEPAAdapter(
        SimpleNamespace(run_batch=lambda candidate, items: [candidate["vision.nframes"]]),
        lambda output, gold: (float(output == gold), "generic feedback", None),
        config_choices={"vision.nframes": [4, 8, 16]},
    )
    item = {"query": "How many frames?", "gold": "8"}

    adapter.evaluate([item], {"system_prompt": "seed", "vision.nframes": "8"})
    first = adapter.propose_new_texts(
        {"system_prompt": "seed", "vision.nframes": "8"},
        {"vision.nframes": []},
        ["vision.nframes"],
    )
    adapter.evaluate([item], {"system_prompt": "seed", "vision.nframes": first["vision.nframes"]})
    second = adapter.propose_new_texts(
        {"system_prompt": "seed", "vision.nframes": first["vision.nframes"]},
        {"vision.nframes": []},
        ["vision.nframes"],
    )

    assert first == {"vision.nframes": "4"}
    assert second == {"vision.nframes": "16"}


def test_joint_adapter_does_not_extract_visual_evidence_for_config_component():
    evidence_calls = []
    adapter = TAOGEPAAdapter(
        SimpleNamespace(run_batch=lambda candidate, items: ["B"]),
        lambda output, gold: (0.0, "generic feedback", None),
        reflection_evidence_fn=lambda item, candidate: evidence_calls.append(item) or {"t=0": "frame"},
        vision_components=["system_prompt"],
        config_choices={"vision.nframes": [4, 8, 16]},
    )
    item = {"id": "private", "query": "Question?", "gold": "A"}
    evaluated = adapter.evaluate(
        [item],
        {"system_prompt": "seed", "vision.nframes": "8"},
        capture_traces=True,
    )

    records = adapter.make_reflective_dataset(
        {"system_prompt": "seed", "vision.nframes": "8"},
        evaluated,
        ["vision.nframes"],
    )

    assert evidence_calls == []
    assert "Visual Evidence" not in records["vision.nframes"][0]


def test_joint_adapter_validates_seed_config_choices():
    adapter = TAOGEPAAdapter(
        SimpleNamespace(run_batch=lambda candidate, items: []),
        lambda output, gold: 0.0,
        config_choices={"vision.nframes": [4, 8, 16]},
    )

    with pytest.raises(ValueError, match="missing config component"):
        adapter.validate_seed_candidate({"system_prompt": "seed"})
    with pytest.raises(ValueError, match="is not in"):
        adapter.validate_seed_candidate({"system_prompt": "seed", "vision.nframes": 32})


def test_adapter_rejects_unaligned_action_outputs():
    runner = SimpleNamespace(run_batch=lambda candidate, items: ["only one"])
    adapter = TAOGEPAAdapter(runner, lambda output, gold: (1.0, "ok", None))

    with pytest.raises(ValueError, match="1 outputs for 2 input items"):
        adapter.evaluate(
            [{"query": "a", "gold": "a"}, {"query": "b", "gold": "b"}],
            {"dataset.system_prompt": "prompt"},
        )


def test_aggregate_metric_reranks_gepa_proxy_winner_and_scores_test(monkeypatch):
    class Runner:
        def __init__(self):
            self.calls = []

        def run_batch(self, candidate, items):
            self.calls.append((dict(candidate), [item["id"] for item in items]))
            prompt = candidate["dataset.system_prompt"]
            return [prompt for _ in items]

    runner = Runner()
    adapter = TAOGEPAAdapter(
        runner,
        lambda output, gold: (float(output == gold), "generic failure", None),
        fixed_candidate={"vision.nframes": "8"},
    )
    candidates = [
        {"dataset.system_prompt": "proxy-winner"},
        {"dataset.system_prompt": "macro-winner"},
    ]
    fake_result = SimpleNamespace(
        candidates=candidates,
        val_aggregate_scores=[0.9, 0.8],
        best_idx=0,
    )

    def fake_optimize(**kwargs):
        assert kwargs["max_metric_calls"] == 20
        assert kwargs["cache_evaluation"] is True
        kwargs["adapter"].evaluate(
            kwargs["trainset"], kwargs["seed_candidate"], capture_traces=True
        )
        return fake_result

    monkeypatch.setitem(__import__("sys").modules, "gepa", SimpleNamespace(optimize=fake_optimize))

    def aggregate(outputs, golds):
        del golds
        score = 0.8 if outputs[0] == "macro-winner" else 0.6
        return {"macro_f1": score, "accuracy": score + 0.05}

    prompter = GEPAutoPrompter(
        adapter,
        reflection_lm=object(),
        aggregate_metric_fn=aggregate,
        aggregate_metric_key="macro_f1",
        seed=42,
    )
    train = [{"id": "train", "query": "train", "gold": "seed"}]
    val = [{"id": "val", "query": "val", "gold": "Yes"}]
    test = [{"id": "test", "query": "test", "gold": "Yes"}]
    result = prompter.optimize(
        {"dataset.system_prompt": "seed"},
        train,
        val,
        budget=20,
        testset=test,
    )

    assert result.gepa_candidate_index == 0
    assert result.selected_candidate_index == 1
    assert result.best_candidate == {"dataset.system_prompt": "macro-winner"}
    assert result.best_full_candidate == {
        "vision.nframes": "8",
        "dataset.system_prompt": "macro-winner",
    }
    assert result.validation_score == 0.8
    assert result.test_score == 0.8
    assert result.test_metrics["macro_f1"] == 0.8
    assert result.test_metrics["accuracy"] == pytest.approx(0.85)
    assert len(result.candidate_validation_metrics) == 2
    assert runner.calls[-1][1] == ["test"]


def test_aggregate_metric_must_expose_requested_key(monkeypatch):
    fake_result = SimpleNamespace(
        candidates=[{"dataset.system_prompt": "seed"}],
        val_aggregate_scores=[1.0],
        best_idx=0,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "gepa",
        SimpleNamespace(optimize=lambda **kwargs: fake_result),
    )
    runner = SimpleNamespace(run_batch=lambda candidate, items: ["Yes"] * len(items))
    adapter = TAOGEPAAdapter(runner, lambda output, gold: (1.0, "ok", None))
    prompter = GEPAutoPrompter(
        adapter,
        reflection_lm=object(),
        aggregate_metric_fn=lambda outputs, golds: {"accuracy": 1.0},
        aggregate_metric_key="macro_f1",
    )

    with pytest.raises(KeyError, match="macro_f1"):
        prompter.optimize(
            {"dataset.system_prompt": "seed"},
            [{"query": "train", "gold": "Yes"}],
            [{"query": "val", "gold": "Yes"}],
            budget=1,
        )
