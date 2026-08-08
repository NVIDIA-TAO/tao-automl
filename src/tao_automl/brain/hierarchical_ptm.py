# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hierarchical, non-ordinal pretrained-model Bayesian search.

Checkpoint identity is an outer categorical arm.  Each qualified checkpoint
owns an independent native :class:`~tao_automl.brain.bayesian.Bayesian`
surrogate over its conditional parameter space.  The outer scheduler chooses
an arm and only that arm's history is visible to its inner surrogate.

The wrapper deliberately does not encode checkpoint identity as a number in a
Gaussian process.  It also keeps its scheduler and every inner Bayesian state
in one content-addressed persistence record so a resume cannot silently mix an
old arm inventory, search space, preflight result, or acquisition policy.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

from tao_automl.brain.bayesian import Bayesian
from tao_automl.ptm_registry import merge_ptm_spec_precedence
from tao_automl.ptm_search import (
    HierarchicalPTMScheduler,
    PTMArm,
    PTMArmObservation,
)
from tao_automl.recommendation_audit import (
    algorithmic_campaign_flags,
    audit_json_value,
    canonical_audit_sha256,
    validate_algorithmic_campaign_flags,
    validate_recommendation_audit,
)


HIERARCHICAL_PTM_BRAIN_SCHEMA_VERSION = 1
_TERMINAL_STATUSES = frozenset(
    {"success", "done", "failure", "error", "canceled"}
)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return normalized if math.isfinite(normalized) else None


def _path_overlap(left: str, right: str) -> bool:
    """Return whether two dotted/indexed spec paths overlap."""
    return (
        left == right
        or left.startswith(f"{right}.")
        or left.startswith(f"{right}[")
        or right.startswith(f"{left}.")
        or right.startswith(f"{left}[")
    )


def _stable_candidate_key(candidate_id: str) -> tuple[int, int | str, str]:
    """Sort numeric controller IDs numerically and all other IDs lexically."""
    try:
        numeric = int(candidate_id)
    except (TypeError, ValueError, OverflowError):
        return (1, str(candidate_id), str(candidate_id))
    return (0, numeric, str(candidate_id))


class _EmbeddedBrainStateStore:
    """Keep one inner brain's state inside the wrapper persistence record.

    Non-brain operations are delegated to the real state store.  This prevents
    multiple inner Bayesian brains sharing a controller context ID from
    overwriting one another's ``brain/<context>.json`` file.
    """

    def __init__(
        self,
        delegate: Any,
        brain_state: Mapping[str, Any] | None = None,
        *,
        job_spec: Mapping[str, Any],
        custom_ranges: Mapping[str, Any],
    ):
        self._delegate = delegate
        self._brain_state = (
            copy.deepcopy(dict(brain_state)) if brain_state is not None else None
        )
        self._job_spec = copy.deepcopy(dict(job_spec))
        self._custom_ranges = copy.deepcopy(dict(custom_ranges))

    def get_brain_info(self, _job_id: str) -> dict[str, Any] | None:
        return copy.deepcopy(self._brain_state)

    def save_brain_info(self, _job_id: str, state: Mapping[str, Any]) -> None:
        self._brain_state = copy.deepcopy(dict(state))

    def get_job_specs(self, _job_id: str) -> dict[str, Any]:
        """Return this arm's immutable PTM-effective base specification."""
        return copy.deepcopy(self._job_spec)

    def get_custom_param_ranges(self, _experiment_id: str) -> dict[str, Any]:
        """Return this arm's immutable conditional custom ranges."""
        return copy.deepcopy(self._custom_ranges)

    def snapshot(self) -> dict[str, Any]:
        if self._brain_state is None:
            raise ValueError("Inner Bayesian brain did not persist any state")
        return copy.deepcopy(self._brain_state)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class HierarchicalPTMBrain:
    """Controller-compatible wrapper for categorical PTM Bayesian search.

    Args:
        context: Controller context shared by the wrapper and inner brains.
        state_store: Outer controller state store.
        scheduler: Frozen hierarchical PTM scheduler.
        inner_brains: Exactly one newly constructed native Bayesian brain per
            scheduler checkpoint arm, keyed by the arm's stable checkpoint ID.
        candidate_overrides: Lower-precedence per-arm candidate spec values.
            Each arm must set its configured checkpoint target here.
        checkpoint_targets: Dotted checkpoint spec path for each arm.
        accuracy_metric: Raw quality metric supplied in objective values.
        latency_metric: Raw stabilized latency metric supplied in objective
            values.
        fidelity_metric: Optional objective/diagnostic key used by a scheduler
            with a frozen required fidelity.
        resume: Restore and verify the wrapper state already in ``state_store``.

    The regular :class:`~tao_automl.controller.controller.Controller` invokes
    :meth:`save_state` after recommendation issuance and result reporting.
    This method snapshots the scheduler and *all* inner brains on either path.
    """

    def __init__(
        self,
        *,
        context: Any,
        state_store: Any,
        scheduler: HierarchicalPTMScheduler,
        inner_brains: Mapping[str, Bayesian],
        candidate_overrides: Mapping[str, Mapping[str, Any]],
        checkpoint_targets: Mapping[str, str],
        accuracy_metric: str,
        latency_metric: str,
        fidelity_metric: str | None = None,
        resume: bool = False,
    ):
        if not isinstance(scheduler, HierarchicalPTMScheduler):
            raise TypeError("scheduler must be a HierarchicalPTMScheduler")
        self.context = context
        self.state_store = state_store
        self.scheduler = scheduler
        self.random_seed = scheduler.random_seed
        self.network = None
        self.accuracy_metric = self._nonempty_string(
            accuracy_metric, "accuracy_metric"
        )
        self.latency_metric = self._nonempty_string(
            latency_metric, "latency_metric"
        )
        if self.accuracy_metric == self.latency_metric:
            raise ValueError("accuracy_metric and latency_metric must differ")
        self.fidelity_metric = (
            self._nonempty_string(fidelity_metric, "fidelity_metric")
            if fidelity_metric is not None
            else None
        )

        arm_ids = tuple(arm.checkpoint_id for arm in scheduler.arms)
        if set(inner_brains) != set(arm_ids):
            raise ValueError(
                "inner_brains must contain exactly one brain per PTM arm"
            )
        if len({id(brain) for brain in inner_brains.values()}) != len(arm_ids):
            raise ValueError("Every PTM arm must own a distinct inner brain")
        if set(candidate_overrides) != set(arm_ids):
            raise ValueError(
                "candidate_overrides must contain exactly one mapping per PTM arm"
            )
        if set(checkpoint_targets) != set(arm_ids):
            raise ValueError(
                "checkpoint_targets must contain exactly one path per PTM arm"
            )

        self._arms = {
            arm.checkpoint_id: arm for arm in scheduler.arms
        }
        self._candidate_overrides: dict[str, dict[str, Any]] = {}
        self._checkpoint_targets: dict[str, str] = {}
        self._candidate_override_hashes: dict[str, str] = {}
        self._fresh_inner_brains: dict[str, Bayesian] = {}

        expected_context_id = str(getattr(context, "id", ""))
        for arm_id in arm_ids:
            brain = inner_brains[arm_id]
            if not isinstance(brain, Bayesian):
                raise TypeError(
                    "Hierarchical PTM arms require native Bayesian brains; "
                    f"{arm_id!r} received {type(brain).__name__}"
                )
            if str(getattr(brain.context, "id", "")) != expected_context_id:
                raise ValueError(
                    f"Inner brain {arm_id!r} uses a different controller context"
                )
            target = self._nonempty_string(
                checkpoint_targets[arm_id],
                f"checkpoint_targets[{arm_id!r}]",
            )
            raw_overrides = candidate_overrides[arm_id]
            if not isinstance(raw_overrides, Mapping):
                raise TypeError(
                    f"candidate_overrides[{arm_id!r}] must be a mapping"
                )
            # The precedence utility validates ambiguous dotted/nested paths and
            # produces a deterministic nested representation.
            overrides = merge_ptm_spec_precedence(
                model_defaults=raw_overrides
            ).spec
            target_value = self._get_path(overrides, target)
            if not isinstance(target_value, str) or not target_value.strip():
                raise ValueError(
                    f"PTM arm {arm_id!r} must set checkpoint target "
                    f"{target!r} to a non-empty resolved artifact"
                )
            parameter_names = self._parameter_names(brain.parameters, arm_id)
            conflicting = sorted(
                name for name in parameter_names if _path_overlap(name, target)
            )
            if conflicting:
                raise ValueError(
                    f"PTM arm {arm_id!r} checkpoint target {target!r} "
                    "overlaps searchable parameter(s): "
                    + ", ".join(conflicting)
                )
            self._candidate_overrides[arm_id] = overrides
            self._checkpoint_targets[arm_id] = target
            self._candidate_override_hashes[arm_id] = (
                canonical_audit_sha256(overrides)
            )
            self._fresh_inner_brains[arm_id] = brain

        # Controller-facing search-space metadata is explicitly conditional.
        # PTM identity is absent from every numeric inner parameter vector.
        self.conditional_parameters = {
            arm_id: copy.deepcopy(self._fresh_inner_brains[arm_id].parameters)
            for arm_id in arm_ids
        }
        self.parameters = [
            {
                "ptm_arm_id": arm_id,
                "checkpoint_identity_representation": (
                    "categorical_outer_arm_not_surrogate_dimension"
                ),
                "conditional_search_space_sha256": (
                    self._arms[arm_id].conditional_search_space_sha256
                ),
                "parameters": copy.deepcopy(self.conditional_parameters[arm_id]),
            }
            for arm_id in arm_ids
        ]
        self.custom_ranges = {
            arm_id: copy.deepcopy(
                getattr(self._fresh_inner_brains[arm_id], "custom_ranges", {})
                or {}
            )
            for arm_id in arm_ids
        }
        self.algorithm_capability = {
            "algorithm": "hierarchical_ptm_bayesian",
            "implementation": (
                "tao_automl.brain.hierarchical_ptm.HierarchicalPTMBrain"
            ),
            "inner_algorithm": "bayesian",
            "checkpoint_identity_representation": (
                "categorical_outer_arm_not_surrogate_dimension"
            ),
            "one_conditional_surrogate_per_arm": True,
            "resume_deterministic": True,
        }
        self.objective_mode_capability = {
            "mode": scheduler.policy.mode,
            "supported": True,
            "support_level": "native_hierarchical",
            "acquisition_strategy": (
                "mode_aware_outer_ptm_allocation_with_native_per_arm_bayesian"
            ),
            "sees_raw_objectives": True,
            "consumes_archive_acquisition_score": False,
            "objective_aware": True,
        }
        self._pending_recommendation_audits: list[dict[str, Any]] = []
        self._last_acquisition_audit: dict[str, Any] = {
            "schema_version": HIERARCHICAL_PTM_BRAIN_SCHEMA_VERSION,
            "method": "hierarchical_ptm_native_bayesian",
            "stage": "initialized",
            "mode": scheduler.policy.mode,
        }

        self._signature = self._build_signature()
        self._signature_sha256 = canonical_audit_sha256(self._signature)
        self._inner_stores: dict[str, _EmbeddedBrainStateStore] = {}
        if resume:
            self._restore_state()
        else:
            self.inner_brains = {}
            for arm_id in arm_ids:
                brain = self._fresh_inner_brains[arm_id]
                store = _EmbeddedBrainStateStore(
                    self.state_store,
                    job_spec=brain.default_train_spec,
                    custom_ranges=brain.custom_ranges,
                )
                brain.state_store = store
                self._inner_stores[arm_id] = store
                self.inner_brains[arm_id] = brain

    @staticmethod
    def _nonempty_string(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _parameter_names(
        parameters: Sequence[Mapping[str, Any]],
        arm_id: str,
    ) -> tuple[str, ...]:
        if not isinstance(parameters, Sequence) or isinstance(
            parameters, (str, bytes)
        ):
            raise TypeError(f"Inner brain {arm_id!r} parameters must be a sequence")
        names = []
        for index, parameter in enumerate(parameters):
            if not isinstance(parameter, Mapping):
                raise TypeError(
                    f"Inner brain {arm_id!r} parameter {index} must be a mapping"
                )
            name = parameter.get("parameter")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"Inner brain {arm_id!r} parameter {index} has no valid name"
                )
            names.append(name.strip())
        if len(set(names)) != len(names):
            raise ValueError(
                f"Inner brain {arm_id!r} contains duplicate parameter names"
            )
        return tuple(names)

    @staticmethod
    def _get_path(spec: Mapping[str, Any], path: str) -> Any:
        current: Any = spec
        for token in path.split("."):
            if not isinstance(current, Mapping) or token not in current:
                return None
            current = current[token]
        return current

    def _inner_configuration(self, arm_id: str, brain: Bayesian) -> dict[str, Any]:
        objective_config = getattr(brain, "objective_config", None)
        return {
            "class": f"{type(brain).__module__}.{type(brain).__qualname__}",
            "network": getattr(brain, "network", None),
            "parameters": copy.deepcopy(brain.parameters),
            "custom_ranges": copy.deepcopy(
                getattr(brain, "custom_ranges", {}) or {}
            ),
            "random_seed": getattr(brain, "random_seed", None),
            "metric": getattr(brain, "metric", None),
            "metric_direction": getattr(brain, "metric_direction", None),
            "objective_config": (
                objective_config.to_dict()
                if objective_config is not None
                else None
            ),
            "objective_acquisition_signature": (
                brain._acquisition_signature()
            ),
            "checkpoint_target": self._checkpoint_targets[arm_id],
            "candidate_overrides_sha256": (
                self._candidate_override_hashes[arm_id]
            ),
        }

    def _build_signature(self) -> dict[str, Any]:
        # TAO schema metadata may use non-finite sentinels for an unbounded
        # declared range (for example ``valid_max=inf``).  These are search
        # space identity metadata, not measured objective values.  Persist
        # their explicit audit tags so the state remains strict JSON without
        # weakening finite-value validation at recommendation/result
        # boundaries.  Operational parameter records remain unchanged.
        return audit_json_value(
            {
                "schema_version": HIERARCHICAL_PTM_BRAIN_SCHEMA_VERSION,
                "context_id": str(getattr(self.context, "id", "")),
                "scheduler": copy.deepcopy(self.scheduler.signature),
                "accuracy_metric": self.accuracy_metric,
                "latency_metric": self.latency_metric,
                "fidelity_metric": self.fidelity_metric,
                "arms": {
                    arm_id: {
                        "arm": self._arms[arm_id].to_dict(),
                        "arm_sha256": canonical_audit_sha256(
                            self._arms[arm_id].to_dict()
                        ),
                        "inner": self._inner_configuration(
                            arm_id, self._fresh_inner_brains[arm_id]
                        ),
                    }
                    for arm_id in sorted(self._arms)
                },
            }
        )

    @property
    def signature(self) -> dict[str, Any]:
        return copy.deepcopy(self._signature)

    @property
    def signature_sha256(self) -> str:
        return self._signature_sha256

    @property
    def acquisition_audit(self) -> dict[str, Any]:
        return copy.deepcopy(self._last_acquisition_audit)

    def consume_last_recommendation_audits(self) -> list[dict[str, Any]]:
        """Return and clear exactly the wrapper audits not yet consumed."""
        audits = copy.deepcopy(self._pending_recommendation_audits)
        self._pending_recommendation_audits.clear()
        return audits

    @staticmethod
    def _validate_combined_audit(audit: Any) -> None:
        if not isinstance(audit, Mapping):
            raise ValueError("Hierarchical PTM acquisition audit must be a mapping")
        expected = audit.get("combined_acquisition_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(
                "Hierarchical PTM acquisition audit hash is missing or invalid"
            )
        payload = copy.deepcopy(dict(audit))
        payload.pop("combined_acquisition_sha256", None)
        if canonical_audit_sha256(payload) != expected:
            raise ValueError(
                "Hierarchical PTM acquisition audit integrity verification failed"
            )
        validate_algorithmic_campaign_flags(
            audit.get("algorithmic_campaign_flags")
        )

    def _history_arm_id(self, recommendation: Any) -> tuple[str, str]:
        record = getattr(recommendation, "recommendation_audit", None)
        validate_recommendation_audit(record)
        acquisition = record.get("acquisition")
        if not isinstance(acquisition, Mapping):
            raise ValueError("Recommendation acquisition audit is missing")
        proposal = acquisition.get("proposal")
        self._validate_combined_audit(proposal)
        ptm = proposal.get("ptm")
        if not isinstance(ptm, Mapping):
            raise ValueError("Recommendation has no immutable PTM arm audit")
        arm_id = ptm.get("arm_id")
        if arm_id not in self._arms:
            raise ValueError(
                f"Recommendation references unknown PTM arm {arm_id!r}"
            )
        arm = self._arms[arm_id]
        expected_values = {
            "arm_sha256": canonical_audit_sha256(arm.to_dict()),
            "preflight_provenance_sha256": (
                arm.preflight_provenance_sha256
            ),
            "conditional_search_space_sha256": (
                arm.conditional_search_space_sha256
            ),
        }
        for name, expected in expected_values.items():
            if ptm.get(name) != expected:
                raise ValueError(
                    f"Recommendation PTM audit {name} does not match arm "
                    f"{arm_id!r}"
                )
        generated_specs = record.get("generated_parameter_values")
        if proposal.get("emitted_specs_sha256") != canonical_audit_sha256(
            generated_specs
        ):
            raise ValueError(
                "Recommendation PTM audit does not match emitted specifications"
            )
        candidate_id = str(record.get("candidate_id"))
        return arm_id, candidate_id

    def _partition_history(
        self,
        history: Sequence[Any],
    ) -> tuple[dict[str, list[Any]], list[PTMArmObservation]]:
        partitions = {arm_id: [] for arm_id in self._arms}
        indexed = []
        seen_candidate_ids = set()
        for recommendation in history:
            arm_id, candidate_id = self._history_arm_id(recommendation)
            if candidate_id in seen_candidate_ids:
                raise ValueError(
                    f"Hierarchical PTM history duplicates candidate {candidate_id!r}"
                )
            seen_candidate_ids.add(candidate_id)
            indexed.append((candidate_id, arm_id, recommendation))
        indexed.sort(key=lambda item: _stable_candidate_key(item[0]))

        observations = []
        for candidate_id, arm_id, recommendation in indexed:
            partitions[arm_id].append(recommendation)
            values = getattr(recommendation, "objective_values", None)
            values = values if isinstance(values, Mapping) else {}
            fidelity = (
                _finite(values.get(self.fidelity_metric))
                if self.fidelity_metric is not None
                else 1.0
            )
            observations.append(
                PTMArmObservation(
                    candidate_id=candidate_id,
                    checkpoint_id=arm_id,
                    status=str(
                        getattr(recommendation, "status", "")
                    ).lower(),
                    accuracy=_finite(values.get(self.accuracy_metric)),
                    latency=_finite(values.get(self.latency_metric)),
                    fidelity=fidelity,
                )
            )
        observations.sort(
            key=lambda item: (
                item.checkpoint_id,
                _stable_candidate_key(item.candidate_id),
            )
        )
        return partitions, observations

    def _merge_recommendation(
        self,
        arm_id: str,
        generated: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(generated, Mapping):
            raise TypeError("Inner Bayesian recommendation must be a mapping")
        declared = set(
            self._parameter_names(self.inner_brains[arm_id].parameters, arm_id)
        )
        unexpected = sorted(set(generated) - declared)
        missing = sorted(declared - set(generated))
        if unexpected or missing:
            details = []
            if unexpected:
                details.append("unexpected: " + ", ".join(unexpected))
            if missing:
                details.append("missing: " + ", ".join(missing))
            raise ValueError(
                f"Inner Bayesian recommendation for {arm_id!r} does not match "
                "its declared conditional space (" + "; ".join(details) + ")"
            )
        target = self._checkpoint_targets[arm_id]
        conflicting = sorted(
            name for name in generated if _path_overlap(name, target)
        )
        if conflicting:
            raise ValueError(
                f"Generated parameter(s) overlap checkpoint target {target!r}: "
                + ", ".join(conflicting)
            )
        return merge_ptm_spec_precedence(
            model_defaults=self._candidate_overrides[arm_id],
            candidate_overrides=generated,
        ).spec

    def generate_recommendations(self, history: Sequence[Any]) -> list[dict[str, Any]]:
        """Choose one PTM arm and ask only its native Bayesian brain.

        The controller's global history is partitioned solely by the
        content-addressed arm ID stored in each recommendation's immutable
        acquisition audit.  Checkpoint paths or candidate enumeration order
        are never used to infer arm identity.
        """
        partitions, observations = self._partition_history(history)
        if any(
            str(getattr(item, "status", "")).lower() not in _TERMINAL_STATUSES
            for items in partitions.values()
            for item in items
        ):
            return []

        scheduler_state_before = self.scheduler.state_dict()
        decision = self.scheduler.choose_arm(observations)
        arm_id = decision.checkpoint_id
        inner = self.inner_brains[arm_id]
        raw = inner.generate_recommendations(partitions[arm_id])
        if not raw:
            # ``choose_arm`` atomically increments its issuance counters.  An
            # empty inner result did not issue a candidate, so restore the
            # exact scheduler state rather than consuming an allocation.
            self.scheduler = HierarchicalPTMScheduler.from_state_dict(
                arms=tuple(self._arms.values()),
                policy=self.scheduler.policy,
                random_seed=self.scheduler.random_seed,
                state=scheduler_state_before,
            )
            return []
        if len(raw) != 1:
            raise ValueError(
                "A hierarchical native Bayesian arm must emit exactly one "
                f"recommendation, got {len(raw)}"
            )

        consume = getattr(inner, "consume_last_recommendation_audits", None)
        if not callable(consume):
            raise TypeError(
                "Inner Bayesian brain does not expose per-recommendation audits"
            )
        inner_audits = consume()
        if len(inner_audits) != 1:
            raise ValueError(
                "Inner Bayesian recommendation/audit count mismatch: "
                f"1 recommendation, {len(inner_audits)} audit(s)"
            )
        merged = self._merge_recommendation(arm_id, raw[0])
        arm = self._arms[arm_id]
        combined = {
            "schema_version": HIERARCHICAL_PTM_BRAIN_SCHEMA_VERSION,
            "method": "hierarchical_ptm_native_bayesian",
            "mode": self.scheduler.policy.mode,
            "stage": decision.stage,
            "checkpoint_identity_is_ordinal": False,
            "algorithmic_campaign_flags": algorithmic_campaign_flags(),
            "ptm": {
                "arm_id": arm_id,
                "arm_sha256": canonical_audit_sha256(arm.to_dict()),
                "preflight_provenance_sha256": (
                    arm.preflight_provenance_sha256
                ),
                "conditional_search_space_sha256": (
                    arm.conditional_search_space_sha256
                ),
                "input_contract_sha256": arm.input_contract_sha256,
                "candidate_overrides_sha256": (
                    self._candidate_override_hashes[arm_id]
                ),
                "checkpoint_target": self._checkpoint_targets[arm_id],
                "outer_decision": decision.to_dict(),
                "inner_acquisition": copy.deepcopy(inner_audits[0]),
            },
            "emitted_specs_sha256": canonical_audit_sha256(merged),
        }
        combined["combined_acquisition_sha256"] = canonical_audit_sha256(
            combined
        )
        self._validate_combined_audit(combined)
        self._pending_recommendation_audits.append(copy.deepcopy(combined))
        self._last_acquisition_audit = copy.deepcopy(combined)
        return [merged]

    def _state_payload(self) -> dict[str, Any]:
        inner_states = {}
        inner_state_hashes = {}
        for arm_id in sorted(self.inner_brains):
            self.inner_brains[arm_id].save_state()
            state = self._inner_stores[arm_id].snapshot()
            inner_states[arm_id] = state
            inner_state_hashes[arm_id] = canonical_audit_sha256(state)
        scheduler_state = self.scheduler.state_dict()
        return {
            "schema_version": HIERARCHICAL_PTM_BRAIN_SCHEMA_VERSION,
            "signature": copy.deepcopy(self._signature),
            "signature_sha256": self._signature_sha256,
            "scheduler_state": scheduler_state,
            "scheduler_state_sha256": canonical_audit_sha256(
                scheduler_state
            ),
            "inner_states": inner_states,
            "inner_state_sha256": inner_state_hashes,
            "pending_recommendation_audits": copy.deepcopy(
                self._pending_recommendation_audits
            ),
            "last_acquisition_audit": copy.deepcopy(
                self._last_acquisition_audit
            ),
        }

    def save_state(self) -> None:
        """Persist one integrity-protected scheduler/all-inner snapshot."""
        payload = self._state_payload()
        payload["state_sha256"] = canonical_audit_sha256(payload)
        self.state_store.save_brain_info(self.context.id, payload)

    def _restore_state(self) -> None:
        raw_state = self.state_store.get_brain_info(self.context.id)
        if not isinstance(raw_state, Mapping):
            raise ValueError("No hierarchical PTM brain state exists to resume")
        state = copy.deepcopy(dict(raw_state))
        if state.get("schema_version") != HIERARCHICAL_PTM_BRAIN_SCHEMA_VERSION:
            raise ValueError("Hierarchical PTM brain state schema is unsupported")
        expected_state_hash = state.pop("state_sha256", None)
        if (
            not isinstance(expected_state_hash, str)
            or canonical_audit_sha256(state) != expected_state_hash
        ):
            raise ValueError("Hierarchical PTM brain state integrity check failed")
        stored_signature = state.get("signature")
        stored_signature_sha256 = state.get("signature_sha256")
        if (
            not isinstance(stored_signature, Mapping)
            or canonical_audit_sha256(stored_signature)
            != stored_signature_sha256
        ):
            raise ValueError(
                "Persisted hierarchical PTM brain signature is corrupt"
            )
        normalized_signature = copy.deepcopy(dict(stored_signature))
        stored_scheduler = normalized_signature.get("scheduler")
        if isinstance(stored_scheduler, Mapping):
            stored_scheduler = copy.deepcopy(dict(stored_scheduler))
            stored_policy = stored_scheduler.get("policy")
            if isinstance(stored_policy, Mapping):
                stored_policy = copy.deepcopy(dict(stored_policy))
                # Schema-v1 states written before the outer tolerance was
                # explicit used the product default implicitly.
                stored_policy.setdefault("accuracy_tolerance", 1e-12)
                stored_scheduler["policy"] = stored_policy
            normalized_signature["scheduler"] = stored_scheduler
        if normalized_signature != self._signature:
            raise ValueError(
                "Cannot resume hierarchical PTM brain with a different "
                "configuration, arm inventory, preflight, or search space"
            )
        scheduler_state = state.get("scheduler_state")
        if (
            not isinstance(scheduler_state, Mapping)
            or canonical_audit_sha256(scheduler_state)
            != state.get("scheduler_state_sha256")
        ):
            raise ValueError(
                "Persisted hierarchical PTM scheduler state is corrupt"
            )
        self.scheduler = HierarchicalPTMScheduler.from_state_dict(
            arms=tuple(self._arms.values()),
            policy=self.scheduler.policy,
            random_seed=self.scheduler.random_seed,
            state=scheduler_state,
        )

        inner_states = state.get("inner_states")
        inner_hashes = state.get("inner_state_sha256")
        if (
            not isinstance(inner_states, Mapping)
            or not isinstance(inner_hashes, Mapping)
            or set(inner_states) != set(self._arms)
            or set(inner_hashes) != set(self._arms)
        ):
            raise ValueError(
                "Persisted inner Bayesian state inventory does not match PTM arms"
            )
        self.inner_brains = {}
        for arm_id in sorted(self._arms):
            inner_state = inner_states[arm_id]
            if (
                not isinstance(inner_state, Mapping)
                or canonical_audit_sha256(inner_state)
                != inner_hashes[arm_id]
            ):
                raise ValueError(
                    f"Persisted inner Bayesian state for {arm_id!r} is corrupt"
                )
            fresh = self._fresh_inner_brains[arm_id]
            store = _EmbeddedBrainStateStore(
                self.state_store,
                brain_state=inner_state,
                job_spec=fresh.default_train_spec,
                custom_ranges=fresh.custom_ranges,
            )
            restored = Bayesian.load_state(
                context=fresh.context,
                state_store=store,
                network=fresh.network,
                parameters=fresh.parameters,
                metric=fresh.metric,
                direction=fresh.metric_direction,
                objective_config=fresh.objective_config,
                acquisition_settings=fresh.acquisition_settings,
            )
            for name in ("algorithm_capability", "objective_mode_capability"):
                if hasattr(fresh, name):
                    setattr(restored, name, copy.deepcopy(getattr(fresh, name)))
            self._inner_stores[arm_id] = store
            self.inner_brains[arm_id] = restored

        pending = state.get("pending_recommendation_audits", [])
        if not isinstance(pending, list):
            raise ValueError("Persisted PTM acquisition audit queue is invalid")
        for audit in pending:
            self._validate_combined_audit(audit)
        self._pending_recommendation_audits = copy.deepcopy(pending)
        last = state.get("last_acquisition_audit")
        if not isinstance(last, Mapping):
            raise ValueError("Persisted PTM last acquisition audit is invalid")
        if last.get("stage") != "initialized":
            self._validate_combined_audit(last)
        self._last_acquisition_audit = copy.deepcopy(dict(last))

    @classmethod
    def load_state(cls, **kwargs: Any) -> "HierarchicalPTMBrain":
        """Construct a wrapper and fail closed while restoring its state."""
        if "resume" in kwargs:
            raise TypeError("load_state determines resume=True")
        return cls(resume=True, **kwargs)


__all__ = [
    "HIERARCHICAL_PTM_BRAIN_SCHEMA_VERSION",
    "HierarchicalPTMBrain",
]
