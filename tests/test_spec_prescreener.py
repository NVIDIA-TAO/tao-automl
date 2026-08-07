from types import SimpleNamespace

from tao_automl.brain.spec_prescreener import SpecPrescreener


def _prescreener(json_content):
    client = SimpleNamespace(
        chat=lambda *_args, **_kwargs: SimpleNamespace(
            ok=True,
            json_content=json_content,
            error=None,
        )
    )
    return SpecPrescreener(llm_client=client)


def _candidates():
    return [{"x": 1}, {"x": 2}, {"x": 3}]


def test_prescreen_normalizes_list_wrapped_object():
    result = _prescreener([
        {
            "recommended_to_run": [2],
            "confidence": "high",
            "reasoning": "Candidate 2 is best.",
        }
    ]).prescreen(
        candidates=_candidates(),
        network="grounding_dino",
        metric_name="test_mAP50",
        metric_direction="maximize",
        max_to_run=1,
    )

    assert result == [{"x": 2}]


def test_prescreen_unsupported_list_falls_back_to_all_candidates():
    candidates = _candidates()
    result = _prescreener([1, 2, 3]).prescreen(
        candidates=candidates,
        network="grounding_dino",
        metric_name="test_mAP50",
        metric_direction="maximize",
        max_to_run=1,
    )

    assert result == candidates
