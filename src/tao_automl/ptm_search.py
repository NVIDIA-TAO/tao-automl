# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic hierarchical PTM-arm allocation.

Checkpoint identity is categorical, not ordinal.  This module therefore keeps
one conditional inner search per preflight-qualified checkpoint and provides a
mode-aware outer allocator.  It never encodes checkpoint IDs as distances in a
Gaussian-process input.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from tao_automl.brain.objective_acquisition import parego_utilities
from tao_automl.recommendation_audit import algorithmic_campaign_flags
from tao_automl.selection import accuracy_feasibility_boundary


# Version 2 defines calibration in terms of valid observations while retaining
# a separately bounded count of issued recovery trials. States written under
# the issue-count-only v1 semantics cannot be replayed safely.
PTM_SEARCH_SCHEMA_VERSION = 2
PTM_SEARCH_MODES = frozenset({"accuracy", "latency", "multi_objective"})


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class PTMArm:
    """One preflight-qualified checkpoint and its conditional inner space."""

    checkpoint_id: str
    conditional_search_space_sha256: str
    preflight_provenance_sha256: str
    input_contract_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id:
            raise ValueError("PTM arm checkpoint_id must be non-empty")
        for field_name in (
            "conditional_search_space_sha256",
            "preflight_provenance_sha256",
            "input_contract_sha256",
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{field_name} must be lowercase SHA-256 hex")

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "conditional_search_space_sha256": (
                self.conditional_search_space_sha256
            ),
            "preflight_provenance_sha256": self.preflight_provenance_sha256,
            "input_contract_sha256": self.input_contract_sha256,
        }


@dataclass(frozen=True)
class PTMArmObservation:
    """One terminal inner-search result visible to the outer allocator."""

    candidate_id: str
    checkpoint_id: str
    status: str
    accuracy: float | None
    latency: float | None
    fidelity: float | None = None

    @property
    def valid(self) -> bool:
        accuracy = _finite(self.accuracy)
        latency = _finite(self.latency)
        fidelity = _finite(self.fidelity) if self.fidelity is not None else 1.0
        return (
            self.status in {"success", "done"}
            and accuracy is not None
            and latency is not None
            and latency > 0.0
            and fidelity is not None
            and fidelity > 0.0
        )


@dataclass(frozen=True)
class HierarchicalPTMPolicy:
    """Frozen outer-search policy shared by all PTM arms."""

    mode: str
    initial_issues_per_arm: int = 2
    invalid_recovery_issues_per_arm: int = 1
    exploration_strength: float = 0.15
    latency_accuracy_retention: float = 0.90
    accuracy_tolerance: float = 1e-12
    required_fidelity: float | None = None
    fidelity_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.mode not in PTM_SEARCH_MODES:
            raise ValueError(
                f"PTM search mode must be one of {sorted(PTM_SEARCH_MODES)}"
            )
        for name in (
            "initial_issues_per_arm",
            "invalid_recovery_issues_per_arm",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        exploration = _finite(self.exploration_strength)
        if exploration is None or exploration < 0.0:
            raise ValueError("exploration_strength must be finite and >= 0")
        retention = _finite(self.latency_accuracy_retention)
        if retention is None or not 0.0 < retention <= 1.0:
            raise ValueError(
                "latency_accuracy_retention must be finite and in (0, 1]"
            )
        accuracy_tolerance = _finite(self.accuracy_tolerance)
        if accuracy_tolerance is None or accuracy_tolerance < 0.0:
            raise ValueError("accuracy_tolerance must be finite and >= 0")
        if self.required_fidelity is not None:
            required_fidelity = _finite(self.required_fidelity)
            if required_fidelity is None or required_fidelity <= 0.0:
                raise ValueError("required_fidelity must be finite and > 0")
        fidelity_tolerance = _finite(self.fidelity_tolerance)
        if fidelity_tolerance is None or fidelity_tolerance < 0.0:
            raise ValueError("fidelity_tolerance must be finite and >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "initial_issues_per_arm": self.initial_issues_per_arm,
            "invalid_recovery_issues_per_arm": (
                self.invalid_recovery_issues_per_arm
            ),
            "exploration_strength": self.exploration_strength,
            "latency_accuracy_retention": self.latency_accuracy_retention,
            "accuracy_tolerance": self.accuracy_tolerance,
            "required_fidelity": self.required_fidelity,
            "fidelity_tolerance": self.fidelity_tolerance,
        }


@dataclass(frozen=True)
class PTMArmDecision:
    """Content-addressed outer-arm decision made before candidate generation."""

    decision_index: int
    model_based_decision_index: int
    checkpoint_id: str
    stage: str
    reason: str
    arm_scores: Mapping[str, float | None]
    issued_counts: Mapping[str, int]
    valid_observation_counts: Mapping[str, int]
    accuracy_reference: float | None
    accuracy_threshold: float | None
    parego: Mapping[str, Any] | None
    scheduler_signature_sha256: str
    decision_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PTM_SEARCH_SCHEMA_VERSION,
            "decision_index": self.decision_index,
            "model_based_decision_index": self.model_based_decision_index,
            "checkpoint_id": self.checkpoint_id,
            "stage": self.stage,
            "reason": self.reason,
            "arm_scores": dict(self.arm_scores),
            "issued_counts": dict(self.issued_counts),
            "valid_observation_counts": dict(self.valid_observation_counts),
            "accuracy_reference": self.accuracy_reference,
            "accuracy_threshold": self.accuracy_threshold,
            "parego": copy.deepcopy(self.parego),
            "scheduler_signature_sha256": self.scheduler_signature_sha256,
            "algorithmic_campaign_flags": algorithmic_campaign_flags(),
            "decision_sha256": self.decision_sha256,
        }


class HierarchicalPTMScheduler:
    """Select categorical PTM arms without imposing an ordinal geometry."""

    def __init__(
        self,
        arms: Sequence[PTMArm],
        policy: HierarchicalPTMPolicy,
        *,
        random_seed: int,
    ):
        ordered = tuple(sorted(arms, key=lambda arm: arm.checkpoint_id))
        if not ordered:
            raise ValueError("Hierarchical PTM search requires at least one arm")
        if len({arm.checkpoint_id for arm in ordered}) != len(ordered):
            raise ValueError("PTM arm checkpoint IDs must be unique")
        if (
            isinstance(random_seed, bool)
            or not isinstance(random_seed, int)
            or not 0 <= random_seed < 2**32
        ):
            raise ValueError("random_seed must be an integer in [0, 2**32)")
        self.arms = ordered
        self.policy = policy
        self.random_seed = random_seed
        self.issued_counts = {arm.checkpoint_id: 0 for arm in ordered}
        self.decision_index = 0
        self.model_based_decision_index = 0

    @property
    def signature(self) -> dict[str, Any]:
        return {
            "schema_version": PTM_SEARCH_SCHEMA_VERSION,
            "arms": [arm.to_dict() for arm in self.arms],
            "policy": self.policy.to_dict(),
            "random_seed": self.random_seed,
        }

    @property
    def signature_sha256(self) -> str:
        return _canonical_sha256(self.signature)

    def state_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "signature_sha256": self.signature_sha256,
            "issued_counts": dict(self.issued_counts),
            "decision_index": self.decision_index,
            "model_based_decision_index": self.model_based_decision_index,
        }

    @classmethod
    def from_state_dict(
        cls,
        *,
        arms: Sequence[PTMArm],
        policy: HierarchicalPTMPolicy,
        random_seed: int,
        state: Mapping[str, Any],
    ) -> "HierarchicalPTMScheduler":
        scheduler = cls(arms, policy, random_seed=random_seed)
        stored_signature = state.get("signature")
        if not isinstance(stored_signature, Mapping):
            raise ValueError("Persisted PTM scheduler signature is invalid")
        if state.get("signature_sha256") != _canonical_sha256(
            stored_signature
        ):
            raise ValueError("Hierarchical PTM search signature hash is corrupt")
        normalized_signature = copy.deepcopy(dict(stored_signature))
        stored_policy = normalized_signature.get("policy")
        if isinstance(stored_policy, Mapping):
            stored_policy = copy.deepcopy(dict(stored_policy))
            # Schema-v1 states written before accuracy tolerance was bound to
            # the outer policy used the product default implicitly.
            stored_policy.setdefault("accuracy_tolerance", 1e-12)
            normalized_signature["policy"] = stored_policy
        if normalized_signature != scheduler.signature:
            raise ValueError(
                "Cannot resume hierarchical PTM search with a different "
                "arm inventory, policy, or seed"
            )
        raw_counts = state.get("issued_counts")
        if not isinstance(raw_counts, Mapping) or set(raw_counts) != set(
            scheduler.issued_counts
        ):
            raise ValueError("Persisted PTM issued counts do not match arm inventory")
        counts = {}
        for checkpoint_id, value in raw_counts.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Persisted PTM issued count is invalid")
            counts[checkpoint_id] = value
        decision_index = state.get("decision_index")
        if (
            isinstance(decision_index, bool)
            or not isinstance(decision_index, int)
            or decision_index < 0
            or sum(counts.values()) != decision_index
        ):
            raise ValueError("Persisted PTM decision index is invalid")
        model_based_decision_index = state.get(
            "model_based_decision_index",
            0,
        )
        if (
            isinstance(model_based_decision_index, bool)
            or not isinstance(model_based_decision_index, int)
            or not 0 <= model_based_decision_index <= decision_index
        ):
            raise ValueError(
                "Persisted PTM model-based decision index is invalid"
            )
        scheduler.issued_counts = counts
        scheduler.decision_index = decision_index
        scheduler.model_based_decision_index = model_based_decision_index
        return scheduler

    def _seeded_tie_key(self, checkpoint_id: str) -> tuple[str, str]:
        digest = hashlib.sha256(
            f"{self.random_seed}:{checkpoint_id}".encode("utf-8")
        ).hexdigest()
        return digest, checkpoint_id

    def _balanced_choice(self, checkpoint_ids: Iterable[str]) -> str:
        candidates = tuple(checkpoint_ids)
        return min(
            candidates,
            key=lambda checkpoint_id: (
                self.issued_counts[checkpoint_id],
                self._seeded_tie_key(checkpoint_id),
            ),
        )

    def _valid_observations(
        self,
        observations: Iterable[PTMArmObservation],
    ) -> list[PTMArmObservation]:
        arm_ids = set(self.issued_counts)
        observations = list(observations)
        unknown = sorted(
            {
                observation.checkpoint_id
                for observation in observations
                if observation.checkpoint_id not in arm_ids
            }
        )
        if unknown:
            raise ValueError(
                "PTM observations reference unknown arm(s): "
                + ", ".join(unknown)
            )
        valid = [
            observation
            for observation in observations
            if observation.valid
        ]
        candidate_ids = [item.candidate_id for item in valid]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(
                "PTM observations contain duplicate candidate IDs; duplicate "
                "global candidate IDs are not permitted across PTM arms"
            )

        required_fidelity = self.policy.required_fidelity
        if required_fidelity is not None:
            return [
                item
                for item in valid
                if abs(
                    float(item.fidelity if item.fidelity is not None else 1.0)
                    - required_fidelity
                )
                <= self.policy.fidelity_tolerance
            ]

        fidelities = {
            float(item.fidelity if item.fidelity is not None else 1.0)
            for item in valid
        }
        if len(fidelities) > 1:
            raise ValueError(
                "Hierarchical PTM allocation cannot compare mixed fidelities "
                "without policy.required_fidelity"
            )
        return valid

    @staticmethod
    def _accuracy_normalization(
        observations: Sequence[PTMArmObservation],
    ) -> tuple[float, float]:
        values = [float(item.accuracy) for item in observations]
        return min(values), max(values)

    @staticmethod
    def _normalized_accuracy(
        value: float,
        bounds: tuple[float, float],
    ) -> float:
        minimum, maximum = bounds
        return 0.5 if maximum <= minimum else (value - minimum) / (maximum - minimum)

    def _model_based_scores(
        self,
        valid: Sequence[PTMArmObservation],
    ) -> tuple[dict[str, float | None], float | None, float | None, dict | None]:
        grouped = {
            arm.checkpoint_id: [
                item for item in valid if item.checkpoint_id == arm.checkpoint_id
            ]
            for arm in self.arms
        }
        counts = {
            checkpoint_id: len(items)
            for checkpoint_id, items in grouped.items()
        }
        total = max(1, sum(counts.values()))
        exploration = {
            checkpoint_id: self.policy.exploration_strength
            * math.sqrt(math.log(total + 1.0) / (count + 1.0))
            for checkpoint_id, count in counts.items()
        }
        if not valid:
            return (
                {
                    checkpoint_id: exploration[checkpoint_id]
                    for checkpoint_id in grouped
                },
                None,
                None,
                None,
            )

        accuracy_bounds = self._accuracy_normalization(valid)
        reference = max(float(item.accuracy) for item in valid)
        policy_threshold = (
            reference * self.policy.latency_accuracy_retention
        )
        threshold = accuracy_feasibility_boundary(
            policy_threshold,
            self.policy.accuracy_tolerance,
        )
        scores: dict[str, float | None] = {}
        parego = None

        if self.policy.mode == "accuracy":
            for checkpoint_id, items in grouped.items():
                if not items:
                    scores[checkpoint_id] = -1.0 + exploration[checkpoint_id]
                    continue
                best = max(float(item.accuracy) for item in items)
                scores[checkpoint_id] = (
                    self._normalized_accuracy(best, accuracy_bounds)
                    + exploration[checkpoint_id]
                )
            return scores, reference, None, None

        if self.policy.mode == "latency":
            if reference <= 0.0:
                for checkpoint_id, items in grouped.items():
                    if not items:
                        base = -1.0
                    else:
                        best_accuracy = max(
                            float(item.accuracy) for item in items
                        )
                        base = self._normalized_accuracy(
                            best_accuracy,
                            accuracy_bounds,
                        )
                    scores[checkpoint_id] = (
                        base + exploration[checkpoint_id]
                    )
                return scores, reference, None, None
            feasible = [
                item for item in valid if float(item.accuracy) >= threshold
            ]
            feasible_latencies = [float(item.latency) for item in feasible]
            latency_min = min(feasible_latencies) if feasible_latencies else None
            latency_max = max(feasible_latencies) if feasible_latencies else None
            scale = max(abs(reference), 1e-12)
            for checkpoint_id, items in grouped.items():
                arm_feasible = [
                    item for item in items if float(item.accuracy) >= threshold
                ]
                if arm_feasible:
                    latency = min(float(item.latency) for item in arm_feasible)
                    if latency_max is None or latency_min is None or latency_max <= latency_min:
                        base = 1.0
                    else:
                        base = 1.0 - (latency - latency_min) / (
                            latency_max - latency_min
                        )
                elif items:
                    best_accuracy = max(float(item.accuracy) for item in items)
                    base = -1.0 - max(0.0, threshold - best_accuracy) / scale
                else:
                    base = -2.0
                scores[checkpoint_id] = base + exploration[checkpoint_id]
            return scores, reference, threshold, None

        accuracies = [float(item.accuracy) for item in valid]
        latencies = [float(item.latency) for item in valid]
        utilities, parego = parego_utilities(
            accuracies,
            latencies,
            iteration=self.model_based_decision_index,
        )
        utility_by_candidate = {
            (item.checkpoint_id, item.candidate_id): float(utility)
            for item, utility in zip(valid, utilities)
        }
        for checkpoint_id, items in grouped.items():
            base = (
                max(
                    utility_by_candidate[
                        (item.checkpoint_id, item.candidate_id)
                    ]
                    for item in items
                )
                if items
                else -2.0
            )
            scores[checkpoint_id] = base + exploration[checkpoint_id]
        return scores, reference, None, parego

    def choose_arm(
        self,
        observations: Iterable[PTMArmObservation],
    ) -> PTMArmDecision:
        """Choose and atomically count the next PTM arm."""
        observations = tuple(observations)
        valid = self._valid_observations(observations)
        valid_counts = {
            arm.checkpoint_id: sum(
                item.checkpoint_id == arm.checkpoint_id for item in valid
            )
            for arm in self.arms
        }

        calibration = [
            arm.checkpoint_id
            for arm in self.arms
            if self.issued_counts[arm.checkpoint_id]
            < self.policy.initial_issues_per_arm
        ]
        if calibration:
            selected = self._balanced_choice(calibration)
            stage = "balanced_initial_design"
            reason = (
                "Deterministic equal-issue PTM calibration; no observed result "
                "or checkpoint ordering preference selected this arm."
            )
            scores = {arm.checkpoint_id: None for arm in self.arms}
            reference = threshold = parego = None
        else:
            recovery = [
                arm.checkpoint_id
                for arm in self.arms
                if valid_counts[arm.checkpoint_id]
                < self.policy.initial_issues_per_arm
                and self.issued_counts[arm.checkpoint_id]
                < (
                    self.policy.initial_issues_per_arm
                    + self.policy.invalid_recovery_issues_per_arm
                )
            ]
            if recovery:
                selected = self._balanced_choice(recovery)
                stage = "preregistered_invalid_recovery"
                reason = (
                    "The arm has fewer complete valid calibration observations "
                    "than required and remains inside its preregistered bounded "
                    "recovery allowance."
                )
                scores = {arm.checkpoint_id: None for arm in self.arms}
                reference = threshold = parego = None
            else:
                (
                    scores,
                    reference,
                    threshold,
                    parego,
                ) = self._model_based_scores(valid)
                selected = min(
                    scores,
                    key=lambda checkpoint_id: (
                        -float(scores[checkpoint_id]),
                        self._seeded_tie_key(checkpoint_id),
                    ),
                )
                stage = "mode_aware_outer_allocation"
                reason = (
                    "Highest mode-aware arm utility including the frozen "
                    "count-based exploration bonus; deterministic seeded hash "
                    "resolved any exact tie."
                )

        decision_payload = {
            "schema_version": PTM_SEARCH_SCHEMA_VERSION,
            "decision_index": self.decision_index,
            "model_based_decision_index": self.model_based_decision_index,
            "checkpoint_id": selected,
            "stage": stage,
            "reason": reason,
            "arm_scores": scores,
            "issued_counts": dict(self.issued_counts),
            "valid_observation_counts": valid_counts,
            "accuracy_reference": reference,
            "accuracy_threshold": threshold,
            "parego": parego,
            "scheduler_signature_sha256": self.signature_sha256,
            "algorithmic_campaign_flags": algorithmic_campaign_flags(),
        }
        decision_hash = _canonical_sha256(decision_payload)
        decision = PTMArmDecision(
            decision_index=self.decision_index,
            model_based_decision_index=self.model_based_decision_index,
            checkpoint_id=selected,
            stage=stage,
            reason=reason,
            arm_scores=copy.deepcopy(scores),
            issued_counts=copy.deepcopy(self.issued_counts),
            valid_observation_counts=valid_counts,
            accuracy_reference=reference,
            accuracy_threshold=threshold,
            parego=copy.deepcopy(parego),
            scheduler_signature_sha256=self.signature_sha256,
            decision_sha256=decision_hash,
        )
        self.issued_counts[selected] += 1
        self.decision_index += 1
        if stage == "mode_aware_outer_allocation":
            self.model_based_decision_index += 1
        return decision


__all__ = [
    "HierarchicalPTMPolicy",
    "HierarchicalPTMScheduler",
    "PTMArm",
    "PTMArmDecision",
    "PTMArmObservation",
    "PTM_SEARCH_MODES",
    "PTM_SEARCH_SCHEMA_VERSION",
]
