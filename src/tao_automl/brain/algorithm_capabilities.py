# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed algorithm capabilities for objective-aware AutoML search.

Final archive selection and search acquisition are different capabilities.
This registry describes what each production brain actually consumes while it
generates or promotes candidates:

* ``native`` implementations model the raw configured objective values;
* ``scalarized_fallback`` implementations consume the controller's
  archive-derived, maximize-oriented acquisition score;
* ``unsupported`` implementations do not receive enough raw objective context
  to make the requested mode-specific search claim.

The algorithm names and aliases are supplied by :class:`BrainFactory`, keeping
the factory as the source of truth. Registry construction fails if a factory
algorithm has not been explicitly classified.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping


_MODES = (
    "single_objective",
    "accuracy",
    "latency",
    "multi_objective",
    "generic_multi_objective",
)

_NATIVE_MULTI_OBJECTIVE_ALGORITHMS = frozenset({"bayesian"})
_SCALARIZED_NUMERICAL_ALGORITHMS = frozenset({
    "bfbo",
    "hyperband",
    "bohb",
    "asha",
    "pbt",
    "dehb",
    "hyperband_es",
})
_SCALAR_ONLY_AGENTIC_ALGORITHMS = frozenset({
    "llm",
    "hybrid",
    "autoresearch",
})


@dataclass(frozen=True)
class ObjectiveModeCapability:
    """One algorithm's production acquisition behavior for one mode."""

    mode: str
    supported: bool
    support_level: str
    acquisition_strategy: str
    sees_raw_objectives: bool
    consumes_archive_acquisition_score: bool
    objective_aware: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-safe capability metadata."""
        return {
            "mode": self.mode,
            "supported": self.supported,
            "support_level": self.support_level,
            "acquisition_strategy": self.acquisition_strategy,
            "sees_raw_objectives": self.sees_raw_objectives,
            "consumes_archive_acquisition_score": (
                self.consumes_archive_acquisition_score
            ),
            "objective_aware": self.objective_aware,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AlgorithmCapability:
    """Capabilities for one canonical BrainFactory algorithm."""

    algorithm: str
    aliases: tuple[str, ...]
    implementation: str
    modes: tuple[ObjectiveModeCapability, ...]

    def for_mode(self, mode: str) -> ObjectiveModeCapability:
        """Resolve a mode capability or fail closed."""
        for capability in self.modes:
            if capability.mode == mode:
                return capability
        raise ValueError(
            f"Algorithm {self.algorithm!r} has no declared capability for "
            f"objective mode {mode!r}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-safe algorithm metadata."""
        return {
            "algorithm": self.algorithm,
            "aliases": list(self.aliases),
            "implementation": self.implementation,
            "modes": {
                capability.mode: capability.to_dict()
                for capability in self.modes
            },
        }


class AlgorithmCapabilityRegistry:
    """Immutable lookup and serialization surface for algorithm capabilities."""

    schema_version = 1

    def __init__(self, capabilities: tuple[AlgorithmCapability, ...]):
        self._capabilities = capabilities
        by_alias = {}
        for capability in capabilities:
            for alias in capability.aliases:
                normalized = str(alias).strip().lower()
                if normalized in by_alias:
                    raise ValueError(
                        f"Duplicate AutoML algorithm alias {normalized!r}"
                    )
                by_alias[normalized] = capability
        self._by_alias = by_alias

    @property
    def algorithms(self) -> tuple[str, ...]:
        """Return canonical algorithms in BrainFactory declaration order."""
        return tuple(item.algorithm for item in self._capabilities)

    def resolve(self, algorithm: str) -> AlgorithmCapability:
        """Resolve an algorithm name or alias with a clear supported list."""
        normalized = str(algorithm).strip().lower()
        capability = self._by_alias.get(normalized)
        if capability is None:
            supported = ", ".join(sorted(self._by_alias))
            raise ValueError(
                f"AutoML algorithm {algorithm!r} is not registered. "
                f"Supported names and aliases: {supported}"
            )
        return capability

    def validate(
        self,
        algorithm: str,
        objective_config=None,
    ) -> tuple[AlgorithmCapability, ObjectiveModeCapability]:
        """Validate one requested algorithm/objective pairing."""
        capability = self.resolve(algorithm)
        mode = objective_mode_from_config(objective_config)
        mode_capability = capability.for_mode(mode)
        if not mode_capability.supported:
            raise ValueError(
                f"AutoML algorithm {algorithm!r} does not support objective "
                f"mode {mode!r}: {mode_capability.reason}"
            )
        return capability, mode_capability

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete compatibility matrix."""
        return {
            "schema_version": self.schema_version,
            "algorithms": [
                capability.to_dict()
                for capability in self._capabilities
            ],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the compatibility matrix as stable JSON."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
        )


def objective_mode_from_config(objective_config) -> str:
    """Resolve the acquisition capability mode from an ObjectiveConfig."""
    if objective_config is None or not objective_config.is_multi_objective:
        return "single_objective"
    if objective_config.has_archive_selector:
        return objective_config.selection_config.mode
    return "generic_multi_objective"


def _single_objective_capability() -> ObjectiveModeCapability:
    return ObjectiveModeCapability(
        mode="single_objective",
        supported=True,
        support_level="native",
        acquisition_strategy="native_scalar_objective",
        sees_raw_objectives=False,
        consumes_archive_acquisition_score=False,
        objective_aware=True,
        reason=(
            "The brain directly consumes the configured scalar objective with "
            "its declared optimization direction."
        ),
    )


def _native_mode_capability(mode: str) -> ObjectiveModeCapability:
    strategies = {
        "accuracy": "raw_accuracy_expected_improvement",
        "latency": "raw_constrained_latency_expected_improvement",
        "multi_objective": "raw_parego_expected_improvement",
    }
    return ObjectiveModeCapability(
        mode=mode,
        supported=True,
        support_level="native",
        acquisition_strategy=strategies[mode],
        sees_raw_objectives=True,
        consumes_archive_acquisition_score=False,
        objective_aware=True,
        reason=(
            "The Bayesian brain models the raw accuracy and latency objective "
            "values using the mode-specific production acquisition path."
        ),
    )


def _accuracy_scalar_capability() -> ObjectiveModeCapability:
    return ObjectiveModeCapability(
        mode="accuracy",
        supported=True,
        support_level="scalarized_fallback",
        acquisition_strategy="archive_accuracy_score_feedback",
        sees_raw_objectives=False,
        consumes_archive_acquisition_score=True,
        objective_aware=True,
        reason=(
            "Accuracy-mode archive scoring is the raw valid accuracy metric, "
            "so scalar feedback preserves the requested single objective."
        ),
    )


def _scalarized_mode_capability(mode: str) -> ObjectiveModeCapability:
    strategy = (
        "constrained_archive_score_feedback"
        if mode == "latency"
        else "pareto_archive_score_feedback"
    )
    return ObjectiveModeCapability(
        mode=mode,
        supported=True,
        support_level="scalarized_fallback",
        acquisition_strategy=strategy,
        sees_raw_objectives=False,
        consumes_archive_acquisition_score=True,
        objective_aware=False,
        reason=(
            "The brain receives the controller's normalized archive-derived "
            "acquisition score for search, ranking, or promotion. It does not "
            "model accuracy and latency as independent raw objectives."
        ),
    )


def _generic_scalarized_capability() -> ObjectiveModeCapability:
    return ObjectiveModeCapability(
        mode="generic_multi_objective",
        supported=True,
        support_level="scalarized_fallback",
        acquisition_strategy="configured_weighted_scalar_feedback",
        sees_raw_objectives=False,
        consumes_archive_acquisition_score=False,
        objective_aware=False,
        reason=(
            "Generic multi-objective sessions retain the legacy configured "
            "weighted scalarization rather than native Pareto acquisition."
        ),
    )


def _unsupported_raw_multi_objective_capability(
    mode: str,
) -> ObjectiveModeCapability:
    return ObjectiveModeCapability(
        mode=mode,
        supported=False,
        support_level="unsupported",
        acquisition_strategy="none",
        sees_raw_objectives=False,
        consumes_archive_acquisition_score=False,
        objective_aware=False,
        reason=(
            "This agentic search path receives only a scalar result in its "
            "prompt/history and cannot independently reason over both raw "
            "accuracy and latency objectives."
        ),
    )


def build_algorithm_capability_registry(
    algorithm_definitions: Mapping[str, Mapping[str, Any]],
) -> AlgorithmCapabilityRegistry:
    """Build a fail-closed registry from BrainFactory algorithm definitions."""
    configured = set(algorithm_definitions)
    classified = (
        _NATIVE_MULTI_OBJECTIVE_ALGORITHMS
        | _SCALARIZED_NUMERICAL_ALGORITHMS
        | _SCALAR_ONLY_AGENTIC_ALGORITHMS
    )
    missing = sorted(configured - classified)
    stale = sorted(classified - configured)
    if missing or stale:
        details = []
        if missing:
            details.append(
                "unclassified BrainFactory algorithm(s): " + ", ".join(missing)
            )
        if stale:
            details.append(
                "capability entries without a BrainFactory algorithm: "
                + ", ".join(stale)
            )
        raise RuntimeError(
            "Objective capability registry is incomplete: "
            + "; ".join(details)
        )

    capabilities = []
    for algorithm, definition in algorithm_definitions.items():
        aliases = tuple(definition["aliases"])
        implementation = str(definition["implementation"])
        if algorithm in _NATIVE_MULTI_OBJECTIVE_ALGORITHMS:
            modes = (
                _single_objective_capability(),
                _native_mode_capability("accuracy"),
                _native_mode_capability("latency"),
                _native_mode_capability("multi_objective"),
                _generic_scalarized_capability(),
            )
        elif algorithm in _SCALARIZED_NUMERICAL_ALGORITHMS:
            modes = (
                _single_objective_capability(),
                _accuracy_scalar_capability(),
                _scalarized_mode_capability("latency"),
                _scalarized_mode_capability("multi_objective"),
                _generic_scalarized_capability(),
            )
        else:
            modes = (
                _single_objective_capability(),
                _accuracy_scalar_capability(),
                _unsupported_raw_multi_objective_capability("latency"),
                _unsupported_raw_multi_objective_capability(
                    "multi_objective"
                ),
                _unsupported_raw_multi_objective_capability(
                    "generic_multi_objective"
                ),
            )
        if tuple(item.mode for item in modes) != _MODES:
            raise RuntimeError(
                f"Objective capability modes are incomplete for {algorithm!r}"
            )
        capabilities.append(
            AlgorithmCapability(
                algorithm=algorithm,
                aliases=aliases,
                implementation=implementation,
                modes=modes,
            )
        )
    return AlgorithmCapabilityRegistry(tuple(capabilities))
