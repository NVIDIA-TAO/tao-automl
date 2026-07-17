# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TAO-owned GEPA Auto-Prompter for batched non-train actions.

The target boundary is a single ``run_batch(candidate, items)`` call. A caller
can therefore launch one TAO evaluate action per candidate while GEPA still
receives aligned per-example scores and leak-free reflection records.

GEPA evolves text with a decomposable per-item objective. When an official
benchmark metric is set-level (for example VANTAGE Macro-F1), ``GEPAutoPrompter``
reruns/rereads every accepted candidate on validation and selects the winner by
that aggregate metric before touching the test set.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from tao_automl.brain.llm_client import LLMClient
from tao_automl.reflection import sanitize_reflective_feedback

try:
    from gepa.core.adapter import EvaluationBatch
except Exception:  # pragma: no cover - exercised in environments without the optional extra
    @dataclass
    class EvaluationBatch:
        outputs: list[Any]
        scores: list[float]
        trajectories: list[Any] | None = None
        objective_scores: list[dict[str, float]] | None = None


MetricFn = Callable[[Any, Any], Any]
AggregateMetricFn = Callable[[Sequence[Any], Sequence[Any]], float | Mapping[str, Any]]
ReflectionEvidenceFn = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, Any] | None,
]


def _set_dotted_value(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if not dotted_key or any(not part for part in parts):
        raise ValueError(f"Invalid dotted candidate key: {dotted_key!r}")
    node = target
    for part in parts[:-1]:
        current = node.get(part)
        if current is None:
            current = {}
            node[part] = current
        if not isinstance(current, dict):
            raise TypeError(
                f"Candidate key {dotted_key!r} crosses non-object spec field {part!r}"
            )
        node = current
    node[parts[-1]] = value


class TAOActionBatchRunner:
    """Map a candidate into a TAO action spec and execute one batch callback.

    ``evaluate_action`` owns platform execution and artifact parsing. Its
    contract is ``evaluate_action(specs, items) -> outputs`` with one output in
    the same order as each input item.
    """

    def __init__(
        self,
        base_specs: Mapping[str, Any],
        evaluate_action: Callable[[dict[str, Any], list[dict[str, Any]]], Sequence[Any]],
        *,
        value_coercers: Mapping[str, Callable[[Any], Any]] | None = None,
    ):
        self.base_specs = copy.deepcopy(dict(base_specs))
        self.evaluate_action = evaluate_action
        self.value_coercers = dict(value_coercers or {})

    def run_batch(self, candidate: Mapping[str, Any], items: Sequence[dict[str, Any]]):
        specs = copy.deepcopy(self.base_specs)
        for key, raw_value in candidate.items():
            coerce = self.value_coercers.get(key)
            value = coerce(raw_value) if coerce is not None else raw_value
            _set_dotted_value(specs, str(key), value)
        return list(self.evaluate_action(specs, list(items)))


class GEPAReflectionLM:
    """Adapt TAO's OpenAI-compatible ``LLMClient`` to GEPA's LM protocol."""

    def __init__(self, client: LLMClient, *, system_prompt: str | None = None):
        self.client = client
        self.system_prompt = system_prompt

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [dict(message) for message in prompt]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        response = self.client.chat(messages, json_mode=False)
        if not response.ok:
            raise RuntimeError(f"GEPA reflection LLM failed: {response.error}")
        return response.content


class TAOGEPAAdapter:
    """GEPA adapter that evaluates one TAO action batch per candidate."""

    propose_new_texts = None

    def __init__(
        self,
        runner,
        metric_fn: MetricFn,
        *,
        fixed_candidate: Mapping[str, Any] | None = None,
        metric_context_fn: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
        cache_outputs: bool = True,
        reflection_evidence_fn: ReflectionEvidenceFn | None = None,
        vision_components: Sequence[str] | None = None,
    ):
        self.runner = runner
        self.metric_fn = metric_fn
        self.fixed_candidate = dict(fixed_candidate or {})
        self.metric_context_fn = metric_context_fn
        self.cache_outputs = cache_outputs
        self.reflection_evidence_fn = reflection_evidence_fn
        self.vision_components = (
            {str(component) for component in vision_components}
            if vision_components is not None
            else None
        )
        self._output_cache: dict[str, list[Any]] = {}

    def full_candidate(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        return {**self.fixed_candidate, **dict(candidate)}

    @staticmethod
    def _cache_key(candidate: Mapping[str, Any], items: Sequence[dict[str, Any]]) -> str:
        payload = {"candidate": candidate, "items": list(items)}
        encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def run_outputs(
        self,
        candidate: Mapping[str, Any],
        items: Sequence[dict[str, Any]],
    ) -> list[Any]:
        batch = list(items)
        full = self.full_candidate(candidate)
        key = self._cache_key(full, batch)
        if self.cache_outputs and key in self._output_cache:
            return list(self._output_cache[key])

        run_batch = getattr(self.runner, "run_batch", None)
        if not callable(run_batch):
            raise TypeError("TAO Auto-Prompter runner must implement run_batch(candidate, items)")
        outputs = list(run_batch(full, batch))
        if len(outputs) != len(batch):
            raise ValueError(
                f"run_batch returned {len(outputs)} outputs for {len(batch)} input items"
            )
        if self.cache_outputs:
            self._output_cache[key] = list(outputs)
        return outputs

    @staticmethod
    def _metric_result(value: Any) -> tuple[float, Any, dict[str, float] | None]:
        if isinstance(value, tuple):
            if len(value) == 3:
                score, feedback, objectives = value
            elif len(value) == 2:
                score, feedback = value
                objectives = None
            else:
                raise TypeError("metric_fn tuple result must contain two or three values")
        else:
            score, feedback, objectives = value, "", None
        score = float(score)
        if not math.isfinite(score):
            raise ValueError(f"metric_fn returned a non-finite score: {score!r}")
        if objectives is not None and not isinstance(objectives, dict):
            raise TypeError("metric_fn objective scores must be a dictionary or None")
        return score, sanitize_reflective_feedback(feedback), objectives

    def evaluate(
        self,
        batch: Sequence[dict[str, Any]],
        candidate: Mapping[str, Any],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        items = list(batch)
        raw_outputs = self.run_outputs(candidate, items)
        outputs: list[Any] = []
        scores: list[float] = []
        objectives: list[dict[str, float] | None] = []
        trajectories = [] if capture_traces else None

        for item, output in zip(items, raw_outputs):
            try:
                context = dict(self.metric_context_fn(item)) if self.metric_context_fn else {}
                score, feedback, objective = self._metric_result(
                    self.metric_fn(output, item.get("gold"), **context)
                )
            except Exception as exc:
                score, feedback, objective = 0.0, f"evaluation failed: {exc}", None
            outputs.append(output)
            scores.append(score)
            objectives.append(objective)
            if trajectories is not None:
                trajectories.append({
                    "item": item,
                    "query": item.get("query") or "(video analysis task)",
                    "output": output,
                    "feedback": feedback,
                    "score": score,
                })

        any_objectives = any(objective is not None for objective in objectives)
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objectives if any_objectives else None,
        )

    def make_reflective_dataset(
        self,
        candidate: Mapping[str, Any],
        eval_batch: EvaluationBatch,
        components_to_update: Sequence[str],
    ) -> dict[str, list[dict[str, Any]]]:
        full_candidate = self.full_candidate(candidate)
        records = {component: [] for component in components_to_update}
        for trajectory in eval_batch.trajectories or []:
            feedback = trajectory.get("feedback", "")
            if not isinstance(feedback, str):
                feedback = json.dumps(feedback, sort_keys=True, default=str)
            record = {
                "Inputs": {"query": trajectory.get("query")},
                "Generated Outputs": str(trajectory.get("output", ""))[:800],
                "Feedback": feedback,
            }
            evidence = None
            if (
                self.reflection_evidence_fn is not None
                and float(trajectory.get("score", 0.0)) < 1.0
            ):
                evidence = self.reflection_evidence_fn(
                    trajectory.get("item", {}),
                    full_candidate,
                )
                if evidence is not None and not isinstance(evidence, Mapping):
                    raise TypeError("reflection_evidence_fn must return a mapping or None")
            for component in components_to_update:
                component_record = copy.deepcopy(record)
                if evidence and (
                    self.vision_components is None
                    or component in self.vision_components
                ):
                    component_record["Visual Evidence"] = copy.deepcopy(dict(evidence))
                records[component].append(component_record)
        return records


@dataclass
class AutoPrompterResult:
    best_candidate: dict[str, Any]
    best_full_candidate: dict[str, Any]
    selected_candidate_index: int
    gepa_candidate_index: int
    validation_score: float
    validation_metrics: dict[str, Any]
    candidate_validation_metrics: list[dict[str, Any]]
    test_score: float | None
    test_metrics: dict[str, Any] | None
    gepa_result: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_candidate": self.best_candidate,
            "best_full_candidate": self.best_full_candidate,
            "selected_candidate_index": self.selected_candidate_index,
            "gepa_candidate_index": self.gepa_candidate_index,
            "validation_score": self.validation_score,
            "validation_metrics": self.validation_metrics,
            "candidate_validation_metrics": self.candidate_validation_metrics,
            "test_score": self.test_score,
            "test_metrics": self.test_metrics,
        }


class GEPAutoPrompter:
    """Run GEPA and select its final candidate with the official set metric."""

    def __init__(
        self,
        adapter: TAOGEPAAdapter,
        *,
        reflection_lm,
        aggregate_metric_fn: AggregateMetricFn | None = None,
        aggregate_metric_key: str = "macro_f1",
        **gepa_kwargs,
    ):
        self.adapter = adapter
        self.reflection_lm = reflection_lm
        self.aggregate_metric_fn = aggregate_metric_fn
        self.aggregate_metric_key = aggregate_metric_key
        self.gepa_kwargs = dict(gepa_kwargs)

    def _aggregate(self, outputs: Sequence[Any], items: Sequence[dict[str, Any]]):
        if self.aggregate_metric_fn is None:
            raise RuntimeError("aggregate_metric_fn is required for aggregate candidate reranking")
        raw = self.aggregate_metric_fn(outputs, [item.get("gold") for item in items])
        if isinstance(raw, Mapping):
            metrics = dict(raw)
            if self.aggregate_metric_key not in metrics:
                raise KeyError(
                    f"Aggregate metric result has no {self.aggregate_metric_key!r} field"
                )
            score = float(metrics[self.aggregate_metric_key])
        else:
            score = float(raw)
            metrics = {self.aggregate_metric_key: score}
        if not math.isfinite(score):
            raise ValueError(f"Aggregate metric returned a non-finite score: {score!r}")
        return score, metrics

    def optimize(
        self,
        seed_candidate: Mapping[str, str],
        trainset: Sequence[dict[str, Any]],
        valset: Sequence[dict[str, Any]],
        *,
        budget: int,
        testset: Sequence[dict[str, Any]] | None = None,
    ) -> AutoPrompterResult:
        try:
            import gepa
        except ImportError as exc:  # pragma: no cover - depends on optional environment
            raise RuntimeError(
                "GEPA Auto-Prompter requires the 'autoprompter' package extra"
            ) from exc

        train_items, val_items = list(trainset), list(valset)
        if not train_items or not val_items:
            raise ValueError("GEPA Auto-Prompter requires non-empty train and validation sets")
        if budget <= 0:
            raise ValueError("GEPA Auto-Prompter budget must be positive")

        kwargs = dict(self.gepa_kwargs)
        for reserved in ("seed_candidate", "trainset", "valset", "adapter", "reflection_lm", "max_metric_calls"):
            if reserved in kwargs:
                raise TypeError(f"GEPA option {reserved!r} is managed by GEPAutoPrompter")
        kwargs.setdefault("cache_evaluation", True)
        result = gepa.optimize(
            seed_candidate=dict(seed_candidate),
            trainset=train_items,
            valset=val_items,
            adapter=self.adapter,
            reflection_lm=self.reflection_lm,
            max_metric_calls=budget,
            **kwargs,
        )

        candidates = [dict(candidate) for candidate in result.candidates]
        if not candidates:
            raise RuntimeError("GEPA returned no candidates")
        gepa_index = int(result.best_idx)

        candidate_metrics = []
        if self.aggregate_metric_fn is None:
            selected_index = gepa_index
            validation_score = float(result.val_aggregate_scores[selected_index])
            validation_metrics = {"gepa_proxy": validation_score}
        else:
            for index, candidate in enumerate(candidates):
                outputs = self.adapter.run_outputs(candidate, val_items)
                score, metrics = self._aggregate(outputs, val_items)
                candidate_metrics.append({
                    "candidate_index": index,
                    "score": score,
                    "metrics": metrics,
                    "gepa_proxy": float(result.val_aggregate_scores[index]),
                })
            selected = max(
                candidate_metrics,
                key=lambda row: (row["score"], row["gepa_proxy"], -row["candidate_index"]),
            )
            selected_index = int(selected["candidate_index"])
            validation_score = float(selected["score"])
            validation_metrics = dict(selected["metrics"])

        best_candidate = candidates[selected_index]
        test_score = None
        test_metrics = None
        if testset is not None:
            test_items = list(testset)
            if not test_items:
                raise ValueError("testset must be non-empty when provided")
            if self.aggregate_metric_fn is None:
                raise RuntimeError("aggregate_metric_fn is required to score the test set")
            test_outputs = self.adapter.run_outputs(best_candidate, test_items)
            test_score, test_metrics = self._aggregate(test_outputs, test_items)

        return AutoPrompterResult(
            best_candidate=best_candidate,
            best_full_candidate=self.adapter.full_candidate(best_candidate),
            selected_candidate_index=selected_index,
            gepa_candidate_index=gepa_index,
            validation_score=validation_score,
            validation_metrics=validation_metrics,
            candidate_validation_metrics=candidate_metrics,
            test_score=test_score,
            test_metrics=test_metrics,
            gepa_result=result,
        )
