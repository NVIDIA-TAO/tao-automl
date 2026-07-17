from types import SimpleNamespace

from tao_automl.brain.autoresearch_controller import AutoresearchBrain
from tao_automl.brain.experiment_tracker import ExperimentTracker


class _Verifier:
    def verify_result(self, **_kwargs):
        return SimpleNamespace(should_count=True)


def _brain(llm_response):
    brain = AutoresearchBrain.__new__(AutoresearchBrain)
    brain.parameters = [{"parameter": "train.optim.lr"}]
    brain.metric = "accuracy"
    brain.network = "action_recognition"
    brain.tracker = ExperimentTracker(metric_direction="maximize")
    brain.tracker.best_metric = 10.0
    brain.verifier = _Verifier()
    brain.llm_client = SimpleNamespace(
        chat=lambda *_args, **_kwargs: SimpleNamespace(
            ok=True,
            json_content=llm_response,
        )
    )
    return brain


def test_keep_reasoning_compares_against_pre_update_best(monkeypatch):
    brain = _brain({"decision": "keep", "reasoning": "Measured improvement."})
    observed = {}

    def prompt(**kwargs):
        observed.update(kwargs)
        return []

    monkeypatch.setattr(
        "tao_automl.brain.autoresearch_controller.build_keep_discard_prompt",
        prompt,
    )

    entry = brain.record_result(
        spec={"train.optim.lr": 0.001},
        metric_value=50.0,
        status="success",
        job_id="job-0",
    )

    assert entry.decision == "keep"
    assert entry.reasoning == "Measured improvement."
    assert observed["best_result"] == {"metric": 10.0}
    assert brain.tracker.best_metric == 50.0


def test_inconsistent_llm_decision_cannot_contradict_tracker():
    brain = _brain({"decision": "discard", "reasoning": "Discard it."})

    entry = brain.record_result(
        spec={"train.optim.lr": 0.001},
        metric_value=50.0,
        status="success",
        job_id="job-0",
    )

    assert entry.decision == "keep"
    assert entry.reasoning.startswith("Keep based on the measured accuracy")
