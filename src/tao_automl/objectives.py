# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Objective parsing, scalarization, and Pareto utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from tao_automl.utils.value_utils import normalize_finite_number, normalize_json_value


_MINIMIZE_TOKENS = (
    "loss",
    "latency",
    "runtime",
    "duration",
    "time",
    "cost",
    "flops",
    "params",
    "memory",
    "energy",
    "wer",
    "cer",
    "perplexity",
    "ppl",
    "error",
)


def implicit_direction(metric_name: str) -> str:
    """Infer a default direction from a metric name."""
    normalized = (metric_name or "").lower()
    return "minimize" if any(token in normalized for token in _MINIMIZE_TOKENS) else "maximize"


def normalize_direction(direction: str | None, metric_name: str) -> str:
    """Validate an explicit direction or infer one from the metric name."""
    if direction is None:
        return implicit_direction(metric_name)
    normalized = str(direction).strip().lower()
    if normalized not in ("minimize", "maximize"):
        raise ValueError(
            f"Objective direction must be 'minimize' or 'maximize', got {direction!r}"
        )
    return normalized


def is_latency_metric(metric_name: str) -> bool:
    """Return whether a metric name denotes latency-like deployment time."""
    normalized = (metric_name or "").lower()
    return "latency" in normalized or normalized in {"runtime", "duration", "time_ms"}


@dataclass(frozen=True)
class ObjectiveSpec:
    """One optimization objective.

    ``scale`` lets callers put objectives with different units on comparable
    magnitudes before weighted scalarization. For example, latency in
    milliseconds can use ``scale=100``.
    """

    metric: str
    direction: str
    weight: float = 1.0
    scale: float = 1.0

    @classmethod
    def from_raw(cls, raw: str | dict[str, Any], fallback_direction: str | None = None) -> "ObjectiveSpec":
        if isinstance(raw, str):
            metric = raw
            direction = normalize_direction(fallback_direction, metric)
            return cls(metric=metric, direction=direction)

        if not isinstance(raw, dict):
            raise TypeError(f"Objective entries must be strings or dictionaries, got {type(raw)}")

        metric = raw.get("metric") or raw.get("name")
        if not metric:
            raise ValueError("Objective dictionary must include 'metric' or 'name'")

        direction = normalize_direction(raw.get("direction", fallback_direction), metric)
        weight = normalize_finite_number(
            raw.get("weight", 1.0),
            path=f"objective[{metric}].weight",
        )
        scale = normalize_finite_number(
            raw.get("scale", 1.0),
            path=f"objective[{metric}].scale",
        )
        if scale <= 0.0:
            raise ValueError(f"Objective {metric!r} must have scale > 0")
        if weight < 0.0:
            raise ValueError(f"Objective {metric!r} has negative weight={weight}")
        return cls(metric=str(metric), direction=direction, weight=weight, scale=scale)


class ObjectiveConfig:
    """Collection of objectives plus scoring/Pareto helpers."""

    def __init__(self, objectives: Iterable[ObjectiveSpec], primary_metric: str | None = None):
        specs = tuple(objectives)
        if not specs:
            raise ValueError("At least one objective is required")
        seen = set()
        unique = []
        for spec in specs:
            if spec.metric in seen:
                continue
            seen.add(spec.metric)
            unique.append(spec)
        self.objectives = tuple(unique)
        self.primary_metric = primary_metric or self.objectives[0].metric

    @property
    def is_multi_objective(self) -> bool:
        return len(self.objectives) > 1

    @property
    def metric_names(self) -> list[str]:
        return [spec.metric for spec in self.objectives]

    @property
    def primary_direction(self) -> str:
        return self.objectives[0].direction

    @property
    def brain_metric(self) -> str:
        if self.is_multi_objective:
            return "multi_objective_score"
        if self.primary_direction != implicit_direction(self.primary_metric):
            return "objective_loss" if self.primary_direction == "minimize" else "objective_score"
        return self.primary_metric

    @property
    def score_direction(self) -> str:
        if self.is_multi_objective:
            return "maximize"
        return self.primary_direction

    def coerce_values(self, metric_value: float | int | dict[str, Any]) -> dict[str, float]:
        """Convert a reported scalar/dict into raw objective values."""
        if isinstance(metric_value, dict):
            raw = normalize_json_value(
                metric_value,
                path="objective_values",
            )
            if self.primary_metric not in raw:
                for alias in ("metric", "metric_value", "value", "score"):
                    if alias in raw:
                        raw[self.primary_metric] = raw[alias]
                        break
            values = {}
            for spec in self.objectives:
                if spec.metric not in raw:
                    continue
                values[spec.metric] = normalize_finite_number(
                    raw[spec.metric],
                    path=f"objective_values.{spec.metric}",
                )
            return values

        value = normalize_finite_number(
            metric_value,
            path=f"objective_values.{self.primary_metric}",
        )
        return {self.primary_metric: value}

    def validate_complete(self, values: dict[str, float]) -> None:
        missing = [spec.metric for spec in self.objectives if spec.metric not in values]
        if missing:
            raise ValueError(
                "Missing objective value(s): "
                + ", ".join(missing)
                + ". Report a dict with all objective metrics for multi-objective runs."
            )

    def scalarize(self, values: dict[str, float]) -> float:
        """Return the scalar score used by existing search algorithms.

        For single-objective runs this is the raw metric, preserving legacy
        behavior. For multi-objective runs every objective is oriented so
        higher is better, scaled, weighted, and summed.
        """
        self.validate_complete(values)
        if not self.is_multi_objective:
            return float(values[self.primary_metric])

        score = 0.0
        for spec in self.objectives:
            value = float(values[spec.metric]) / spec.scale
            contribution = value if spec.direction == "maximize" else -value
            score += spec.weight * contribution
        if not math.isfinite(score):
            raise ValueError("Objective score must be finite")
        return float(score)

    def is_better_score(self, left: float, right: float) -> bool:
        if self.score_direction == "minimize":
            return left < right
        return left > right

    def dominates(self, left: dict[str, float], right: dict[str, float]) -> bool:
        """Return True if ``left`` Pareto-dominates ``right``."""
        self.validate_complete(left)
        self.validate_complete(right)
        better_or_equal_all = True
        strictly_better_any = False
        for spec in self.objectives:
            a = left[spec.metric]
            b = right[spec.metric]
            if spec.direction == "maximize":
                if a < b:
                    better_or_equal_all = False
                    break
                if a > b:
                    strictly_better_any = True
            else:
                if a > b:
                    better_or_equal_all = False
                    break
                if a < b:
                    strictly_better_any = True
        return better_or_equal_all and strictly_better_any

    def pareto_front(self, recommendations: Iterable[Any]) -> list[Any]:
        """Return non-dominated successful recommendations."""
        candidates = [
            rec for rec in recommendations
            if getattr(rec, "objective_values", None)
            and all(spec.metric in rec.objective_values for spec in self.objectives)
        ]
        front = []
        for rec in candidates:
            dominated = False
            for other in candidates:
                if other is rec:
                    continue
                if self.dominates(other.objective_values, rec.objective_values):
                    dominated = True
                    break
            if not dominated:
                front.append(rec)
        return sorted(front, key=lambda rec: rec.id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_metric": self.primary_metric,
            "objectives": [
                {
                    "metric": spec.metric,
                    "direction": spec.direction,
                    "weight": spec.weight,
                    "scale": spec.scale,
                }
                for spec in self.objectives
            ],
            "score_direction": self.score_direction,
        }


def parse_objective_config(settings: dict[str, Any] | None) -> ObjectiveConfig:
    """Parse legacy ``metric`` settings or the new multi-objective settings."""
    settings = settings or {}
    metric = settings.get("metric", "loss")

    raw_objectives = settings.get("objectives")
    if raw_objectives:
        if not isinstance(raw_objectives, list):
            raise TypeError("automl_settings['objectives'] must be a list")
        objectives = [ObjectiveSpec.from_raw(item) for item in raw_objectives]
    else:
        objectives = [
            ObjectiveSpec.from_raw({
                "metric": metric,
                "direction": settings.get("direction"),
                "weight": settings.get("metric_weight", 1.0),
                "scale": settings.get("metric_scale", 1.0),
            })
        ]

    include_latency = (
        settings.get("multi_objective")
        or settings.get("latency_objective")
        or settings.get("include_latency")
    )
    if include_latency:
        latency_metric = settings.get("latency_metric", "latency")
        if latency_metric not in {spec.metric for spec in objectives}:
            objectives.append(
                ObjectiveSpec.from_raw({
                    "metric": latency_metric,
                    "direction": settings.get("latency_direction", "minimize"),
                    "weight": settings.get("latency_weight", 1.0),
                    "scale": settings.get("latency_scale", 1.0),
                })
            )

    return ObjectiveConfig(objectives, primary_metric=objectives[0].metric)
