# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic constrained and Pareto-aware candidate selection.

The search algorithm's acquisition score and the final deployment decision are
different concerns.  This module owns the latter.  It operates on the complete
measured archive, rejects invalid observations, derives an accuracy-winner
relative feasibility constraint for latency mode, performs non-dominated
sorting under an independent optional multi-objective accuracy policy, and
selects a scale-independent compromise from that policy's rank-zero front.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from tao_automl.utils.value_utils import normalize_finite_number, normalize_json_value


_SUCCESS_STATUSES = frozenset({"success", "done"})
_NO_DISTINCT_COMPROMISE = (
    "No distinct Pareto compromise exists under the configured "
    "multi-objective eligibility policy."
)


def _candidate_value(candidate: Any, name: str, default: Any = None) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get(name, default)
    return getattr(candidate, name, default)


def _candidate_values(candidate: Any) -> Mapping[str, Any]:
    values = _candidate_value(candidate, "objective_values", {})
    return values if isinstance(values, Mapping) else {}


def _stable_identifier(candidate: Any) -> str:
    identifier = _candidate_value(candidate, "id", None)
    if identifier is None:
        identifier = _candidate_value(candidate, "candidate_id", "")
    return str(identifier)


def canonical_spec_fingerprint(specs: Mapping[str, Any] | None) -> str:
    """Return a stable SHA-256 fingerprint for a candidate parameter mapping."""
    normalized = normalize_json_value(specs or {}, path="candidate.specs")
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class AccuracyConstraint:
    """Accuracy-winner-relative retention constraint for latency mode.

    ``relative`` requires ``accuracy >= reference * value``.
    ``absolute`` requires ``accuracy >= reference - value``.
    """

    kind: str = "relative"
    value: float = 0.98
    reference: str = "accuracy_winner"
    reference_value: float | None = None
    reference_candidate_id: str | None = None

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        object.__setattr__(self, "kind", kind)
        if kind not in {"relative", "absolute"}:
            raise ValueError(
                "accuracy constraint type must be 'relative' or 'absolute'"
            )
        value = normalize_finite_number(
            self.value,
            path="selection.latency_accuracy_retention.value",
        )
        if kind == "relative" and not 0.0 < value <= 1.0:
            raise ValueError("relative retained-accuracy fraction must be in (0, 1]")
        if kind == "absolute" and value < 0.0:
            raise ValueError("absolute maximum accuracy degradation must be >= 0")
        object.__setattr__(self, "value", value)
        if self.reference != "accuracy_winner":
            raise ValueError("accuracy constraint reference must be 'accuracy_winner'")
        if self.reference_value is not None:
            object.__setattr__(
                self,
                "reference_value",
                normalize_finite_number(
                    self.reference_value,
                    path="selection.latency_accuracy_retention.reference_value",
                ),
            )

    def threshold(self, reference_accuracy: float) -> float:
        if self.kind == "relative":
            return float(reference_accuracy * self.value)
        return float(reference_accuracy - self.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "value": self.value,
            "reference": self.reference,
            "reference_value": self.reference_value,
            "reference_candidate_id": self.reference_candidate_id,
        }


@dataclass(frozen=True)
class MultiObjectiveAccuracyPolicy:
    """Optional minimum-accuracy policy for multi-objective eligibility.

    ``absolute`` treats ``value`` as the accuracy metric floor itself.
    ``relative`` resolves the floor as ``accuracy_winner * value``.
    """

    kind: str
    value: float
    reference: str | None = None

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        object.__setattr__(self, "kind", kind)
        if kind not in {"absolute", "relative"}:
            raise ValueError(
                "multi-objective minimum-accuracy type must be 'absolute' "
                "or 'relative'"
            )
        value = normalize_finite_number(
            self.value,
            path="selection.multi_objective_min_accuracy.value",
        )
        if kind == "relative":
            if not 0.0 < value <= 1.0:
                raise ValueError(
                    "relative multi-objective retained-accuracy fraction "
                    "must be in (0, 1]"
                )
            reference = self.reference or "accuracy_winner"
            if reference != "accuracy_winner":
                raise ValueError(
                    "relative multi-objective minimum accuracy must reference "
                    "'accuracy_winner'"
                )
        else:
            if self.reference not in (None, ""):
                raise ValueError(
                    "absolute multi-objective minimum accuracy cannot specify "
                    "a reference"
                )
            reference = None
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "reference", reference)

    @classmethod
    def from_raw(
        cls,
        raw: MultiObjectiveAccuracyPolicy | Mapping[str, Any] | float | int,
    ) -> MultiObjectiveAccuracyPolicy:
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return cls(kind="absolute", value=raw)
        if not isinstance(raw, Mapping):
            raise TypeError(
                "multi_objective_min_accuracy must be a number, mapping, or None"
            )
        kind = raw.get("type", "absolute")
        if "value" not in raw:
            raise ValueError(
                "multi_objective_min_accuracy mapping must include 'value'"
            )
        return cls(
            kind=str(kind),
            value=raw["value"],
            reference=raw.get("reference"),
        )

    def threshold(self, accuracy_winner: float) -> float:
        if self.kind == "relative":
            return float(accuracy_winner * self.value)
        return self.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "value": self.value,
            "reference": self.reference,
        }


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for deterministic archive selection."""

    mode: str = "multi_objective"
    accuracy_metric: str = "accuracy"
    latency_metric: str = "latency"
    latency_accuracy_retention: AccuracyConstraint | None = None
    multi_objective_min_accuracy: (
        MultiObjectiveAccuracyPolicy | Mapping[str, Any] | float | int | None
    ) = None
    # Deprecated constructor alias retained for API compatibility. It applies
    # only to latency mode and never constrains multi-objective selection.
    accuracy_constraint: AccuracyConstraint | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    accuracy_tolerance: float = 1e-12
    latency_tolerance: float = 0.0
    score_tolerance: float = 1e-12
    augmentation_rho: float = 1e-6
    normalization: str = "pareto_front"
    latency_ci_low_metric: str = "latency_ci95_low"
    latency_ci_high_metric: str = "latency_ci95_high"

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode == "multi":
            mode = "multi_objective"
        if mode not in {"accuracy", "latency", "multi_objective"}:
            raise ValueError(
                "selection mode must be 'accuracy', 'latency', or 'multi_objective'"
            )
        object.__setattr__(self, "mode", mode)
        if not self.accuracy_metric or not self.latency_metric:
            raise ValueError("accuracy_metric and latency_metric must be non-empty")
        if self.accuracy_metric == self.latency_metric:
            raise ValueError("accuracy_metric and latency_metric must be different")
        retention = self.latency_accuracy_retention
        legacy_retention = self.accuracy_constraint
        if (
            retention is not None
            and legacy_retention is not None
            and retention != legacy_retention
        ):
            raise ValueError(
                "Configure latency_accuracy_retention or the deprecated "
                "accuracy_constraint alias, not both"
            )
        if retention is None:
            retention = (
                legacy_retention
                if legacy_retention is not None
                else AccuracyConstraint()
            )
        if not isinstance(retention, AccuracyConstraint):
            raise TypeError(
                "latency_accuracy_retention must be an AccuracyConstraint"
            )
        object.__setattr__(self, "latency_accuracy_retention", retention)
        object.__setattr__(self, "accuracy_constraint", retention)
        if self.multi_objective_min_accuracy is not None:
            object.__setattr__(
                self,
                "multi_objective_min_accuracy",
                MultiObjectiveAccuracyPolicy.from_raw(
                    self.multi_objective_min_accuracy
                ),
            )
        for name in ("accuracy_tolerance", "latency_tolerance", "score_tolerance"):
            value = normalize_finite_number(
                getattr(self, name),
                path=f"selection.{name}",
            )
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0")
            object.__setattr__(self, name, value)
        rho = normalize_finite_number(
            self.augmentation_rho,
            path="selection.augmentation_rho",
        )
        if rho <= 0.0:
            raise ValueError("augmentation_rho must be > 0")
        object.__setattr__(self, "augmentation_rho", rho)
        normalization = str(self.normalization).strip().lower()
        if normalization != "pareto_front":
            raise ValueError("only 'pareto_front' normalization is currently supported")
        object.__setattr__(self, "normalization", normalization)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "accuracy_metric": self.accuracy_metric,
            "latency_metric": self.latency_metric,
            "latency_accuracy_retention": (
                self.latency_accuracy_retention.to_dict()
            ),
            "multi_objective_min_accuracy": (
                self.multi_objective_min_accuracy.to_dict()
                if self.multi_objective_min_accuracy is not None
                else None
            ),
            "accuracy_tolerance": self.accuracy_tolerance,
            "latency_tolerance": self.latency_tolerance,
            "score_tolerance": self.score_tolerance,
            "augmentation_rho": self.augmentation_rho,
            "normalization": self.normalization,
            "latency_ci_low_metric": self.latency_ci_low_metric,
            "latency_ci_high_metric": self.latency_ci_high_metric,
        }


@dataclass
class CandidateAudit:
    """Derived selection evidence for one archive candidate."""

    candidate: Any
    candidate_id: str
    fingerprint: str
    valid: bool = False
    invalid_reason: str | None = None
    accuracy: float | None = None
    latency: float | None = None
    latency_ci95_low: float | None = None
    latency_ci95_high: float | None = None
    # Compatibility name for latency-mode retention feasibility.
    accuracy_feasible: bool = False
    multi_objective_accuracy_feasible: bool = False
    pareto_rank: int | None = None
    dominated_by: tuple[str, ...] = ()
    feasible_pareto_rank: int | None = None
    feasible_dominated_by: tuple[str, ...] = ()
    duplicate_representative: str | None = None
    duplicate_aliases: tuple[str, ...] = ()
    normalized_accuracy_regret: float | None = None
    normalized_latency_regret: float | None = None
    compromise_score: float | None = None
    ideal_distance: float | None = None
    balance_gap: float | None = None
    acquisition_score: float | None = None
    accuracy_winner: bool = False
    latency_winner: bool = False
    multi_objective_winner: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "fingerprint": self.fingerprint,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "accuracy": self.accuracy,
            "latency": self.latency,
            "latency_ci95_low": self.latency_ci95_low,
            "latency_ci95_high": self.latency_ci95_high,
            "accuracy_feasible": self.accuracy_feasible,
            "latency_accuracy_feasible": self.accuracy_feasible,
            "multi_objective_accuracy_feasible": (
                self.multi_objective_accuracy_feasible
            ),
            "pareto_rank": self.pareto_rank,
            "dominated_by": list(self.dominated_by),
            "feasible_pareto_rank": self.feasible_pareto_rank,
            "feasible_dominated_by": list(self.feasible_dominated_by),
            "multi_objective_pareto_rank": self.feasible_pareto_rank,
            "multi_objective_dominated_by": list(self.feasible_dominated_by),
            "duplicate_representative": self.duplicate_representative,
            "duplicate_aliases": list(self.duplicate_aliases),
            "normalized_accuracy_objective": self.normalized_accuracy_regret,
            "normalized_latency_objective": self.normalized_latency_regret,
            "multi_objective_compromise_score": self.compromise_score,
            "ideal_distance": self.ideal_distance,
            "balance_gap": self.balance_gap,
            "acquisition_score": self.acquisition_score,
            "tie_breaking_values": {
                "accuracy_mode": {
                    "accuracy": self.accuracy,
                    "latency": self.latency,
                    "fingerprint": self.fingerprint,
                    "candidate_id": self.candidate_id,
                },
                "latency_mode": {
                    "latency": self.latency,
                    "latency_ci95_low": self.latency_ci95_low,
                    "latency_ci95_high": self.latency_ci95_high,
                    "accuracy": self.accuracy,
                    "fingerprint": self.fingerprint,
                    "candidate_id": self.candidate_id,
                },
                "multi_objective_mode": {
                    "compromise_score": self.compromise_score,
                    "ideal_distance": self.ideal_distance,
                    "balance_gap": self.balance_gap,
                    "normalized_accuracy_objective": (
                        self.normalized_accuracy_regret
                    ),
                    "fingerprint": self.fingerprint,
                    "candidate_id": self.candidate_id,
                },
            },
            "winner": {
                "accuracy": self.accuracy_winner,
                "latency": self.latency_winner,
                "multi_objective": self.multi_objective_winner,
            },
        }


@dataclass(frozen=True)
class ModeSelection:
    mode: str
    status: str
    winner_id: str | None
    reason: str
    distinct_compromise: bool | None = None
    fallback_used: bool = False
    latency_tied_candidate_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "status": self.status,
            "winner_id": self.winner_id,
            "reason": self.reason,
            "distinct_compromise": self.distinct_compromise,
            "fallback_used": self.fallback_used,
            "latency_tied_candidate_ids": list(self.latency_tied_candidate_ids),
        }


@dataclass
class SelectionAnalysis:
    """Complete, reproducible analysis of a shared measured archive."""

    config: SelectionConfig
    audits: list[CandidateAudit]
    accuracy: ModeSelection
    latency: ModeSelection
    multi_objective: ModeSelection
    accuracy_reference_candidate_id: str | None
    accuracy_reference_value: float | None
    accuracy_threshold: float | None
    multi_objective_accuracy_reference_candidate_id: str | None
    multi_objective_accuracy_reference_value: float | None
    multi_objective_accuracy_threshold: float | None
    normalization_bounds: dict[str, dict[str, float | bool]]
    objective_weights: dict[str, float]

    @property
    def latency_accuracy_reference_candidate_id(self) -> str | None:
        """Return the accuracy anchor used only by latency-mode retention."""
        return self.accuracy_reference_candidate_id

    @property
    def latency_accuracy_reference_value(self) -> float | None:
        """Return the latency-mode retained-accuracy reference value."""
        return self.accuracy_reference_value

    @property
    def latency_accuracy_threshold(self) -> float | None:
        """Return the threshold used only to choose the latency winner."""
        return self.accuracy_threshold

    def selection_for(self, mode: str | None = None) -> ModeSelection:
        resolved = (mode or self.config.mode).lower()
        if resolved == "multi":
            resolved = "multi_objective"
        return getattr(self, resolved)

    def winner(self, mode: str | None = None) -> Any | None:
        winner_id = self.selection_for(mode).winner_id
        if winner_id is None:
            return None
        for audit in self.audits:
            if audit.candidate_id == winner_id:
                return audit.candidate
        return None

    def audit_for(self, candidate_id: Any) -> CandidateAudit | None:
        key = str(candidate_id)
        return next((item for item in self.audits if item.candidate_id == key), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": {
                "selector": "feasible_pareto_augmented_chebyshev",
                "configuration": self.config.to_dict(),
                "objective_weights": dict(self.objective_weights),
                "normalization_bounds": self.normalization_bounds,
                "accuracy_reference_candidate_id": self.accuracy_reference_candidate_id,
                "accuracy_reference_value": self.accuracy_reference_value,
                "accuracy_threshold": self.accuracy_threshold,
                "latency_accuracy_reference_candidate_id": (
                    self.latency_accuracy_reference_candidate_id
                ),
                "latency_accuracy_reference_value": (
                    self.latency_accuracy_reference_value
                ),
                "latency_accuracy_threshold": self.latency_accuracy_threshold,
                "multi_objective_accuracy_reference_candidate_id": (
                    self.multi_objective_accuracy_reference_candidate_id
                ),
                "multi_objective_accuracy_reference_value": (
                    self.multi_objective_accuracy_reference_value
                ),
                "multi_objective_accuracy_threshold": (
                    self.multi_objective_accuracy_threshold
                ),
            },
            "selections": {
                "accuracy": self.accuracy.to_dict(),
                "latency": self.latency.to_dict(),
                "multi_objective": self.multi_objective.to_dict(),
            },
            "candidates": [audit.to_dict() for audit in self.audits],
        }


def _extract_finite(
    values: Mapping[str, Any],
    metric: str,
    *,
    required: bool = True,
) -> tuple[float | None, str | None]:
    if metric not in values:
        return (None, f"missing_metric:{metric}") if required else (None, None)
    try:
        value = normalize_finite_number(
            values[metric],
            path=f"candidate.objective_values.{metric}",
        )
    except (TypeError, ValueError) as exc:
        return None, f"invalid_metric:{metric}:{exc}"
    return value, None


def _build_audits(
    candidates: Iterable[Any],
    config: SelectionConfig,
) -> list[CandidateAudit]:
    audits: list[CandidateAudit] = []
    seen_ids: set[str] = set()
    for candidate in candidates:
        candidate_id = _stable_identifier(candidate)
        if candidate_id in seen_ids:
            raise ValueError(f"duplicate candidate identifier: {candidate_id!r}")
        seen_ids.add(candidate_id)
        try:
            fingerprint = canonical_spec_fingerprint(
                _candidate_value(candidate, "specs", {})
            )
        except (TypeError, ValueError) as exc:
            audits.append(
                CandidateAudit(
                    candidate=candidate,
                    candidate_id=candidate_id,
                    fingerprint="",
                    invalid_reason=f"invalid_specs:{exc}",
                )
            )
            continue
        audit = CandidateAudit(
            candidate=candidate,
            candidate_id=candidate_id,
            fingerprint=fingerprint,
        )
        status = _candidate_value(candidate, "status", "success")
        if status not in _SUCCESS_STATUSES:
            audit.invalid_reason = f"non_success_status:{status}"
            audits.append(audit)
            continue
        values = _candidate_values(candidate)
        accuracy, error = _extract_finite(values, config.accuracy_metric)
        if error:
            audit.invalid_reason = error
            audits.append(audit)
            continue
        latency, error = _extract_finite(values, config.latency_metric)
        if error:
            audit.invalid_reason = error
            audits.append(audit)
            continue
        if latency is None or latency <= 0.0:
            audit.invalid_reason = (
                f"invalid_metric:{config.latency_metric}:must_be_positive"
            )
            audits.append(audit)
            continue
        ci_low, error_low = _extract_finite(
            values,
            config.latency_ci_low_metric,
            required=False,
        )
        ci_high, error_high = _extract_finite(
            values,
            config.latency_ci_high_metric,
            required=False,
        )
        if error_low or error_high:
            audit.invalid_reason = error_low or error_high
            audits.append(audit)
            continue
        if (ci_low is None) != (ci_high is None):
            audit.invalid_reason = "incomplete_latency_confidence_interval"
            audits.append(audit)
            continue
        if ci_low is not None and ci_low <= 0.0:
            audit.invalid_reason = "invalid_latency_confidence_interval"
            audits.append(audit)
            continue
        if ci_low is not None and not ci_low <= latency <= ci_high:
            audit.invalid_reason = "latency_confidence_interval_excludes_median"
            audits.append(audit)
            continue
        audit.valid = True
        audit.accuracy = accuracy
        audit.latency = latency
        audit.latency_ci95_low = ci_low
        audit.latency_ci95_high = ci_high
        audits.append(audit)
    return sorted(audits, key=lambda item: (item.fingerprint, item.candidate_id))


def _dominates(
    left: CandidateAudit,
    right: CandidateAudit,
    *,
    accuracy_tolerance: float,
    latency_tolerance: float,
) -> bool:
    assert left.accuracy is not None and left.latency is not None
    assert right.accuracy is not None and right.latency is not None
    # "No worse" always follows the observed objective directions exactly.
    # Tolerances and confidence intervals decide whether an improvement is
    # strict; they never permit a numerically worse point to dominate.  This
    # keeps the dominance relation acyclic and the returned rank-zero point
    # Pareto-safe even when statistical equivalence is configured.
    accuracy_no_worse = left.accuracy >= right.accuracy
    accuracy_strict = left.accuracy > right.accuracy + accuracy_tolerance
    latency_no_worse = left.latency <= right.latency
    if (
        left.latency_ci95_low is not None
        and left.latency_ci95_high is not None
        and right.latency_ci95_low is not None
        and right.latency_ci95_high is not None
    ):
        # A strict latency improvement requires non-overlap in the favorable
        # direction. Overlap therefore cannot create a latency-only dominance
        # claim, but a point that is numerically slower is never "no worse".
        latency_strict = (
            left.latency_ci95_high
            < right.latency_ci95_low - latency_tolerance
        )
    else:
        latency_strict = left.latency < right.latency - latency_tolerance
    return accuracy_no_worse and latency_no_worse and (
        accuracy_strict or latency_strict
    )


def _nondominated_sort(
    candidates: Sequence[CandidateAudit],
    config: SelectionConfig,
) -> tuple[dict[str, int], dict[str, tuple[str, ...]]]:
    if not candidates:
        return {}, {}
    dominates: dict[str, set[str]] = {
        item.candidate_id: set() for item in candidates
    }
    dominated_by: dict[str, set[str]] = {
        item.candidate_id: set() for item in candidates
    }
    for left in candidates:
        for right in candidates:
            if left is right:
                continue
            if _dominates(
                left,
                right,
                accuracy_tolerance=config.accuracy_tolerance,
                latency_tolerance=config.latency_tolerance,
            ):
                dominates[left.candidate_id].add(right.candidate_id)
                dominated_by[right.candidate_id].add(left.candidate_id)

    remaining = {item.candidate_id for item in candidates}
    ranks: dict[str, int] = {}
    rank = 0
    while remaining:
        front = sorted(
            candidate_id
            for candidate_id in remaining
            if not (dominated_by[candidate_id] & remaining)
        )
        if not front:
            raise RuntimeError(
                "dominance relation produced a cycle; refusing to assign a "
                "Pareto rank or select a potentially dominated candidate"
            )
        for candidate_id in front:
            ranks[candidate_id] = rank
            remaining.remove(candidate_id)
        rank += 1
    return (
        ranks,
        {
            candidate_id: tuple(sorted(parent_ids))
            for candidate_id, parent_ids in dominated_by.items()
        },
    )


def _choose_accuracy(
    valid: Sequence[CandidateAudit],
    config: SelectionConfig,
) -> CandidateAudit | None:
    if not valid:
        return None
    best_accuracy = max(item.accuracy for item in valid if item.accuracy is not None)
    tied = [
        item
        for item in valid
        if best_accuracy - float(item.accuracy) <= config.accuracy_tolerance
    ]
    best_latency = min(item.latency for item in tied if item.latency is not None)
    latency_tied = [
        item
        for item in tied
        if float(item.latency) - best_latency <= config.latency_tolerance
    ]
    return min(latency_tied, key=lambda item: (item.fingerprint, item.candidate_id))


def _choose_latency(
    feasible: Sequence[CandidateAudit],
    config: SelectionConfig,
) -> tuple[CandidateAudit | None, tuple[str, ...]]:
    if not feasible:
        return None, ()
    fastest = min(
        feasible,
        key=lambda item: (
            float(item.latency),
            item.fingerprint,
            item.candidate_id,
        ),
    )
    # Practical tolerance is a hard cohort boundary. Confidence intervals are
    # still used by Pareto dominance and reported uncertainty, but overlap
    # cannot authorize a higher-accuracy candidate whose observed latency
    # disadvantage exceeds the configured practical threshold.
    tied = [
        item
        for item in feasible
        if (
            float(item.latency) - float(fastest.latency)
            <= config.latency_tolerance
        )
    ]
    winner = min(
        tied,
        key=lambda item: (
            -float(item.accuracy),
            item.fingerprint,
            item.candidate_id,
        ),
    )
    return winner, tuple(sorted(item.candidate_id for item in tied))


def _deduplicate_objective_points(
    candidates: Sequence[CandidateAudit],
) -> list[CandidateAudit]:
    groups: dict[tuple[float, float], list[CandidateAudit]] = {}
    for item in candidates:
        assert item.accuracy is not None and item.latency is not None
        groups.setdefault((item.accuracy, item.latency), []).append(item)
    representatives: list[CandidateAudit] = []
    for aliases in groups.values():
        ordered = sorted(
            aliases,
            key=lambda item: (item.fingerprint, item.candidate_id),
        )
        representative = ordered[0]
        alias_ids = tuple(item.candidate_id for item in ordered)
        for item in ordered:
            item.duplicate_representative = representative.candidate_id
            item.duplicate_aliases = alias_ids
        representatives.append(representative)
    return sorted(
        representatives,
        key=lambda item: (item.fingerprint, item.candidate_id),
    )


def _normalization_bounds(
    front: Sequence[CandidateAudit],
    config: SelectionConfig,
) -> dict[str, dict[str, float | bool]]:
    accuracies = [float(item.accuracy) for item in front]
    latencies = [float(item.latency) for item in front]
    if not front:
        return {}
    accuracy_min = min(accuracies)
    accuracy_max = max(accuracies)
    latency_min = min(latencies)
    latency_max = max(latencies)
    return {
        config.accuracy_metric: {
            "ideal": accuracy_max,
            "nadir": accuracy_min,
            "range": accuracy_max - accuracy_min,
            "inactive": (
                accuracy_max - accuracy_min <= config.accuracy_tolerance
            ),
            "direction": "maximize",
        },
        config.latency_metric: {
            "ideal": latency_min,
            "nadir": latency_max,
            "range": latency_max - latency_min,
            "inactive": latency_max - latency_min <= config.latency_tolerance,
            "direction": "minimize",
        },
    }


def _set_normalized_scores(
    audits: Sequence[CandidateAudit],
    bounds: Mapping[str, Mapping[str, Any]],
    config: SelectionConfig,
    weights: Mapping[str, float],
) -> None:
    if not bounds:
        return
    accuracy_bounds = bounds[config.accuracy_metric]
    latency_bounds = bounds[config.latency_metric]
    for item in audits:
        if not item.valid:
            continue
        assert item.accuracy is not None and item.latency is not None
        if accuracy_bounds["inactive"]:
            accuracy_regret = 0.0
        else:
            accuracy_regret = (
                float(accuracy_bounds["ideal"]) - item.accuracy
            ) / float(accuracy_bounds["range"])
        if latency_bounds["inactive"]:
            latency_regret = 0.0
        else:
            latency_regret = (
                item.latency - float(latency_bounds["ideal"])
            ) / float(latency_bounds["range"])
        weighted_accuracy = weights[config.accuracy_metric] * accuracy_regret
        weighted_latency = weights[config.latency_metric] * latency_regret
        item.normalized_accuracy_regret = float(accuracy_regret)
        item.normalized_latency_regret = float(latency_regret)
        item.compromise_score = float(
            max(weighted_accuracy, weighted_latency)
            + config.augmentation_rho * (weighted_accuracy + weighted_latency)
        )
        item.ideal_distance = float(
            math.hypot(weighted_accuracy, weighted_latency)
        )
        item.balance_gap = float(abs(weighted_accuracy - weighted_latency))
        rank = item.feasible_pareto_rank
        if rank is not None:
            item.acquisition_score = float(-(rank + item.compromise_score))


def _choose_compromise(
    front: Sequence[CandidateAudit],
    config: SelectionConfig,
) -> CandidateAudit | None:
    if not front:
        return None
    best_score = min(float(item.compromise_score) for item in front)
    score_tied = [
        item
        for item in front
        if float(item.compromise_score) - best_score <= config.score_tolerance
    ]
    best_distance = min(float(item.ideal_distance) for item in score_tied)
    distance_tied = [
        item
        for item in score_tied
        if float(item.ideal_distance) - best_distance <= config.score_tolerance
    ]
    best_gap = min(float(item.balance_gap) for item in distance_tied)
    balance_tied = [
        item
        for item in distance_tied
        if float(item.balance_gap) - best_gap <= config.score_tolerance
    ]
    # Prefer the accuracy-safe extreme only when every balance criterion is
    # tolerance-equivalent, then use the canonical specification fingerprint.
    return min(
        balance_tied,
        key=lambda item: (
            float(item.normalized_accuracy_regret),
            item.fingerprint,
            item.candidate_id,
        ),
    )


def analyze_archive(
    candidates: Iterable[Any],
    config: SelectionConfig,
    *,
    accuracy_weight: float = 1.0,
    latency_weight: float = 1.0,
) -> SelectionAnalysis:
    """Analyze one shared measured archive and select all three mode winners.

    The returned object contains every value used by the decision.  No candidate
    is selected outside this function.
    """
    raw_accuracy_weight = normalize_finite_number(
        accuracy_weight,
        path="selection.accuracy_weight",
    )
    raw_latency_weight = normalize_finite_number(
        latency_weight,
        path="selection.latency_weight",
    )
    if raw_accuracy_weight <= 0.0 or raw_latency_weight <= 0.0:
        raise ValueError("multi-objective preference weights must be > 0")
    total_weight = raw_accuracy_weight + raw_latency_weight
    weights = {
        config.accuracy_metric: raw_accuracy_weight / total_weight,
        config.latency_metric: raw_latency_weight / total_weight,
    }

    audits = _build_audits(candidates, config)
    valid = [item for item in audits if item.valid]
    global_ranks, global_dominated_by = _nondominated_sort(valid, config)
    for item in valid:
        item.pareto_rank = global_ranks[item.candidate_id]
        item.dominated_by = global_dominated_by[item.candidate_id]

    accuracy_winner = _choose_accuracy(valid, config)
    if accuracy_winner is None:
        empty = ModeSelection(
            mode="accuracy",
            status="no_valid_candidates",
            winner_id=None,
            reason="No candidate has complete finite accuracy and latency measurements.",
        )
        return SelectionAnalysis(
            config=config,
            audits=audits,
            accuracy=empty,
            latency=ModeSelection(
                mode="latency",
                status="no_valid_candidates",
                winner_id=None,
                reason=empty.reason,
            ),
            multi_objective=ModeSelection(
                mode="multi_objective",
                status="no_valid_candidates",
                winner_id=None,
                reason=empty.reason,
                distinct_compromise=False,
            ),
            accuracy_reference_candidate_id=None,
            accuracy_reference_value=None,
            accuracy_threshold=None,
            multi_objective_accuracy_reference_candidate_id=None,
            multi_objective_accuracy_reference_value=None,
            multi_objective_accuracy_threshold=None,
            normalization_bounds={},
            objective_weights=weights,
        )

    accuracy_winner.accuracy_winner = True
    retention = config.latency_accuracy_retention
    assert retention is not None
    reference_value = (
        retention.reference_value
        if retention.reference_value is not None
        else float(accuracy_winner.accuracy)
    )
    reference_candidate_id = (
        retention.reference_candidate_id
        if retention.reference_value is not None
        else accuracy_winner.candidate_id
    )
    threshold = retention.threshold(reference_value)
    latency_feasible = []
    for item in valid:
        assert item.accuracy is not None
        item.accuracy_feasible = (
            item.accuracy >= threshold - config.accuracy_tolerance
        )
        if item.accuracy_feasible:
            latency_feasible.append(item)

    latency_winner, latency_ties = _choose_latency(latency_feasible, config)
    if latency_winner is None:
        latency_selection = ModeSelection(
            mode="latency",
            status="no_accuracy_feasible_candidates",
            winner_id=None,
            reason=(
                "No candidate satisfies the configured accuracy-retention "
                f"threshold {threshold:.12g}."
            ),
        )
    else:
        latency_winner.latency_winner = True
        if len(latency_ties) == 1:
            latency_reason = (
                "Lowest stabilized latency candidate satisfying the "
                "accuracy-winner-relative constraint; no equivalent-fastest "
                "tie-break was required."
            )
        else:
            latency_reason = (
                "Highest-accuracy member of the equivalent-fastest cohort "
                "satisfying the accuracy-winner-relative constraint; "
                "deterministic specification fingerprint and candidate ID "
                "resolve remaining ties."
            )
        latency_selection = ModeSelection(
            mode="latency",
            status="selected",
            winner_id=latency_winner.candidate_id,
            reason=latency_reason,
            latency_tied_candidate_ids=latency_ties,
        )

    multi_objective_feasible = []
    multi_objective_policy = config.multi_objective_min_accuracy
    if multi_objective_policy is None:
        multi_objective_floor = None
        multi_objective_reference_value = None
        multi_objective_reference_candidate_id = None
    else:
        assert isinstance(
            multi_objective_policy,
            MultiObjectiveAccuracyPolicy,
        )
        if multi_objective_policy.kind == "relative":
            multi_objective_reference_value = float(accuracy_winner.accuracy)
            multi_objective_reference_candidate_id = (
                accuracy_winner.candidate_id
            )
        else:
            multi_objective_reference_value = None
            multi_objective_reference_candidate_id = None
        multi_objective_floor = multi_objective_policy.threshold(
            float(accuracy_winner.accuracy)
        )
    for item in valid:
        assert item.accuracy is not None
        item.multi_objective_accuracy_feasible = (
            multi_objective_floor is None
            or item.accuracy
            >= multi_objective_floor - config.accuracy_tolerance
        )
        if item.multi_objective_accuracy_feasible:
            multi_objective_feasible.append(item)

    feasible_ranks, feasible_dominated_by = _nondominated_sort(
        multi_objective_feasible,
        config,
    )
    for item in multi_objective_feasible:
        item.feasible_pareto_rank = feasible_ranks[item.candidate_id]
        item.feasible_dominated_by = feasible_dominated_by[item.candidate_id]

    feasible_front_all = [
        item
        for item in multi_objective_feasible
        if item.feasible_pareto_rank == 0
    ]
    feasible_front = _deduplicate_objective_points(feasible_front_all)
    bounds = _normalization_bounds(feasible_front, config)
    _set_normalized_scores(audits, bounds, config, weights)
    compromise_winner = _choose_compromise(feasible_front, config)
    multi_objective_latency_extreme, _ = _choose_latency(
        multi_objective_feasible,
        config,
    )

    distinct_candidates: list[CandidateAudit] = []
    extremes = [
        item
        for item in (accuracy_winner, multi_objective_latency_extreme)
        if item is not None
    ]
    for item in feasible_front:
        equivalent_to_extreme = any(
            abs(float(item.accuracy) - float(extreme.accuracy))
            <= config.accuracy_tolerance
            and abs(float(item.latency) - float(extreme.latency))
            <= config.latency_tolerance
            for extreme in extremes
        )
        if not equivalent_to_extreme:
            distinct_candidates.append(item)
    has_distinct_candidate = bool(distinct_candidates)
    winner_is_distinct = (
        compromise_winner is not None
        and any(
            item.candidate_id == compromise_winner.candidate_id
            for item in distinct_candidates
        )
    )

    if compromise_winner is None:
        if multi_objective_floor is None:
            reason = (
                "No valid candidate is available for Pareto compromise "
                "selection."
            )
        else:
            reason = (
                "No candidate satisfies the optional multi-objective minimum "
                f"accuracy {multi_objective_floor:.12g}."
            )
        multi_selection = ModeSelection(
            mode="multi_objective",
            status="no_multi_objective_accuracy_feasible_candidates",
            winner_id=None,
            reason=reason,
            distinct_compromise=False,
        )
    else:
        compromise_winner.multi_objective_winner = True
        if winner_is_distinct:
            reason = (
                "Minimum front-normalized augmented-Chebyshev regret on the "
                "multi-objective-eligible Pareto front selected a distinct "
                "compromise."
            )
        elif not has_distinct_candidate:
            reason = (
                _NO_DISTINCT_COMPROMISE
                + " The deterministic augmented-Chebyshev fallback selected "
                "an extreme point."
            )
        else:
            reason = (
                "Minimum front-normalized augmented-Chebyshev regret selected "
                "an extreme point even though distinct multi-objective-eligible "
                "Pareto points exist; this result is not reported as a distinct "
                "compromise."
            )
        multi_selection = ModeSelection(
            mode="multi_objective",
            status="selected",
            winner_id=compromise_winner.candidate_id,
            reason=reason,
            distinct_compromise=winner_is_distinct,
            fallback_used=not has_distinct_candidate,
        )

    # Acquisition utilities are normalized, finite, and always maximize.
    # They are intentionally separate from the final deployment score.
    if config.mode == "accuracy":
        for item in valid:
            item.acquisition_score = float(item.accuracy)
    elif config.mode == "latency":
        if latency_feasible:
            feasible_latencies = [
                float(item.latency) for item in latency_feasible
            ]
            latency_min = min(feasible_latencies)
            latency_range = max(feasible_latencies) - latency_min
            for item in latency_feasible:
                item.acquisition_score = (
                    0.0
                    if latency_range <= config.latency_tolerance
                    else 1.0 - (float(item.latency) - latency_min) / latency_range
                )
        violation_scale = max(abs(reference_value), config.accuracy_tolerance, 1e-12)
        for item in valid:
            if not item.accuracy_feasible:
                item.acquisition_score = float(
                    -1.0 - (threshold - float(item.accuracy)) / violation_scale
                )
    elif multi_objective_floor is not None:
        infeasible_base = -(len(valid) + 1.0)
        violation_scale = max(
            abs(multi_objective_floor),
            abs(float(accuracy_winner.accuracy)),
            config.accuracy_tolerance,
            1e-12,
        )
        for item in valid:
            if not item.multi_objective_accuracy_feasible:
                item.acquisition_score = float(
                    infeasible_base
                    - (
                        multi_objective_floor - float(item.accuracy)
                    )
                    / violation_scale
                )

    return SelectionAnalysis(
        config=config,
        audits=audits,
        accuracy=ModeSelection(
            mode="accuracy",
            status="selected",
            winner_id=accuracy_winner.candidate_id,
            reason=(
                "Highest valid accuracy; accuracy-equivalent ties use lower "
                "stabilized latency then canonical configuration hash."
            ),
        ),
        latency=latency_selection,
        multi_objective=multi_selection,
        accuracy_reference_candidate_id=reference_candidate_id,
        accuracy_reference_value=reference_value,
        accuracy_threshold=threshold,
        multi_objective_accuracy_reference_candidate_id=(
            multi_objective_reference_candidate_id
        ),
        multi_objective_accuracy_reference_value=(
            multi_objective_reference_value
        ),
        multi_objective_accuracy_threshold=multi_objective_floor,
        normalization_bounds=bounds,
        objective_weights=weights,
    )


__all__ = [
    "AccuracyConstraint",
    "CandidateAudit",
    "ModeSelection",
    "MultiObjectiveAccuracyPolicy",
    "SelectionAnalysis",
    "SelectionConfig",
    "analyze_archive",
    "canonical_spec_fingerprint",
]
