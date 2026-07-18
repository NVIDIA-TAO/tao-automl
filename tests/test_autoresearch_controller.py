from types import SimpleNamespace

from tao_automl.brain.autoresearch_controller import AutoresearchBrain
from tao_automl.brain.experiment_tracker import ExperimentTracker
from tao_automl.brain.llm_client import first_json_object


class _Verifier:
    def verify_result(self, **_kwargs):
        return SimpleNamespace(should_count=True)


def test_first_json_object_normalizes_supported_shapes():
    expected = {"decision": "keep"}

    assert first_json_object(expected) == expected
    assert first_json_object([expected]) == expected
    assert first_json_object([1, expected]) == expected
    assert first_json_object([1, 2]) is None
    assert first_json_object("not-json-object") is None


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


def test_missing_llm_json_still_persists_decision_reasoning():
    brain = _brain(None)

    entry = brain.record_result(
        spec={"train.optim.lr": 0.001},
        metric_value=50.0,
        status="success",
        job_id="job-0",
    )

    assert entry.decision == "keep"
    assert entry.reasoning == (
        "Keep based on the measured accuracy relative to the previous best (10.0)."
    )


def test_list_wrapped_proposal_is_normalized(monkeypatch):
    proposal = {
        "modifications": {"train.optim.lr": 0.002},
        "reasoning": "Try a larger learning rate.",
    }
    brain = _brain([proposal])
    brain.spec_schema = {}
    brain.research_program = None
    brain.external_knowledge = None
    brain.custom_ranges = {}
    monkeypatch.setattr(
        "tao_automl.brain.autoresearch_controller.build_autoresearch_prompt",
        lambda **_kwargs: [],
    )

    assert brain._propose_modification() == proposal


def test_list_wrapped_keep_discard_response_is_normalized():
    brain = _brain([{"decision": "keep", "reasoning": "Measured improvement."}])

    entry = brain.record_result(
        spec={"train.optim.lr": 0.001},
        metric_value=50.0,
        status="success",
        job_id="job-0",
    )

    assert entry.decision == "keep"
    assert entry.reasoning == "Measured improvement."


def test_first_minimized_result_is_described_as_initial_best(monkeypatch):
    brain = _brain(None)
    brain.tracker.best_metric = None
    brain.tracker.metric_direction = "minimize"
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
        metric_value=11.75,
        status="success",
        job_id="job-0",
    )

    assert entry.decision == "keep"
    assert entry.reasoning == (
        "Keep as the first measured accuracy result; there is no previous best."
    )
    assert observed["best_result"] == {"metric": None}
