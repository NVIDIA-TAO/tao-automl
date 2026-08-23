from pathlib import Path

import pytest

from tao_automl.brain.factory import AlgorithmParams
from tao_automl.controller.controller import Controller
from tao_automl.state.state_store import StateStore
from tao_automl.types import AutoMLContext, Recommendation


class _Brain:
    def __init__(self):
        self.parameters = [
            {
                "parameter": "train.optim.lr",
                "value_type": "float",
                "valid_min": 0.0,
                "valid_max": 1.0,
                "default_value": 0.5,
            }
        ]
        self.custom_ranges = {
            "train.optim.lr": {"valid_min": 0.1, "valid_max": 0.9}
        }

    def save_state(self):
        pass


class _Analyzer:
    def __init__(self, analysis):
        self.analysis = analysis
        self._last_analysis_count = 0
        self._analyses = []
        self._applied_narrowings = []
        self.parameters = None

    def should_analyze(self, completed_count):
        return completed_count >= 1

    def analyze(self, **kwargs):
        self.parameters = kwargs["parameters"]
        if self.analysis is not None:
            self._last_analysis_count = len(kwargs["experiments"])
            self._analyses.append(self.analysis)
        return self.analysis

    def get_validated_range_narrowings(self, parameters):
        return {"train.optim.lr": {"valid_min": 0.2, "valid_max": 0.4}}

    def get_all_analyses(self):
        return list(self._analyses)

    def format_for_metadata(self):
        return {"total_analyses": len(self._analyses)}


def _controller(tmp_path: Path, *, strict=True):
    settings = AlgorithmParams(
        llm_endpoint="https://example.invalid/v1",
        llm_model="test-model",
        llm_api_key="secret",
        llm_analyzer_enabled=True,
        llm_analyzer_interval=1,
        llm_analyzer_narrow_ranges=True,
        llm_analyzer_strict=strict,
    )
    context = AutoMLContext(
        id="test-job",
        network="segformer",
        workspace_path=str(tmp_path),
        metric="val_miou",
        handler_id="test-experiment",
    )
    store = StateStore(str(tmp_path))
    brain = _Brain()
    controller = Controller(
        brain=brain,
        context=context,
        state_store=store,
        settings=settings,
        metric="val_miou",
        algorithm="bayesian",
    )
    rec = Recommendation(0, {"train.optim.lr": 0.3}, "val_miou")
    controller.history.append(rec)
    return controller, brain, store


def test_generic_analyzer_applies_and_persists_narrowing(tmp_path):
    controller, brain, store = _controller(tmp_path)
    analyzer = _Analyzer({"summary": "narrow it"})
    controller._llm_analyzer = analyzer

    controller.report_result(0, 0.95)

    assert analyzer.parameters[0]["valid_min"] == 0.1
    assert analyzer.parameters[0]["valid_max"] == 0.9
    assert brain.custom_ranges["train.optim.lr"] == {
        "valid_min": 0.2,
        "valid_max": 0.4,
    }
    assert store.get_custom_param_ranges("test-experiment") == brain.custom_ranges
    assert store.get_llm_analyzer_info("test-job")["last_analysis_count"] == 1


def test_strict_generic_analyzer_stops_on_llm_failure(tmp_path):
    controller, _, _ = _controller(tmp_path, strict=True)
    controller._llm_analyzer = _Analyzer(None)

    with pytest.raises(RuntimeError, match="refusing to launch an unguided"):
        controller.report_result(0, 0.95)


def test_range_narrowing_alias_enables_classical_analyzer(monkeypatch):
    monkeypatch.delenv("AUTOML_LLM_ANALYZER_ENABLED", raising=False)
    params = AlgorithmParams.from_dict(
        {
            "enable_llm_range_narrowing": True,
            "llm_analysis_interval": 1,
        }
    )
    assert params.llm_analyzer_enabled is True
    assert params.llm_analyzer_narrow_ranges is True
    assert params.llm_analyzer_interval == 1


def test_classical_analyzer_requires_all_llm_credentials(tmp_path):
    settings = AlgorithmParams(
        llm_endpoint="https://example.invalid/v1",
        llm_model="test-model",
        llm_api_key="",
        llm_analyzer_enabled=True,
    )
    context = AutoMLContext(
        id="test-job",
        network="segformer",
        workspace_path=str(tmp_path),
        metric="val_miou",
        handler_id="test-experiment",
    )
    with pytest.raises(ValueError, match="llm_api_key"):
        Controller(
            brain=_Brain(),
            context=context,
            state_store=StateStore(str(tmp_path)),
            settings=settings,
            metric="val_miou",
            algorithm="bfbo",
        )
