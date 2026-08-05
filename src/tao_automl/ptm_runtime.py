# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed runtime construction from verified pretrained-model preflight.

The runtime boundary is intentionally split in two:

``resolve_ptm_runtime_inventory``
    Validates a live, typed runtime preflight report, applies the objective
    mode's PTM inventory policy, and produces one content-addressed
    PTM-effective base specification per selected checkpoint.

``build_hierarchical_ptm_runtime``
    Adds the conditional parameter records/ranges, constructs one native
    Bayesian brain per PTM arm, and wraps them with the non-ordinal
    :class:`tao_automl.brain.hierarchical_ptm.HierarchicalPTMBrain`.

The typed report is required because ``ValidatedCheckpointSpec.document`` is
deliberately excluded from serialized preflight evidence.  Reconstructing that
live validated document from an arbitrary JSON report would weaken the trust
boundary, so this module does not offer a JSON-report loader.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tao_automl.brain.bayesian import Bayesian
from tao_automl.brain.hierarchical_ptm import HierarchicalPTMBrain
from tao_automl.ptm_preflight import (
    PTMPreflightReport,
    PreparedPTM,
)
from tao_automl.ptm_registry import (
    PTMCompatibilityResult,
    PTMRegistry,
    canonical_sha256,
    load_ptm_registry,
    merge_ptm_spec_precedence,
    sha256_file,
)
from tao_automl.ptm_search import (
    HierarchicalPTMPolicy,
    HierarchicalPTMScheduler,
    PTMArm,
)
from tao_automl.recommendation_audit import (
    algorithmic_campaign_flags,
    audit_json_value,
    canonical_audit_sha256,
)


PTM_RUNTIME_SCHEMA_VERSION = 1
_OBJECTIVE_MODES = frozenset({"accuracy", "latency", "multi_objective"})
_ACCURACY_PTM_POLICIES = frozenset({"default", "user", "all"})
_SCHEDULER_OPTION_KEYS = frozenset(
    {
        "initial_issues_per_arm",
        "invalid_recovery_issues_per_arm",
        "exploration_strength",
        "required_fidelity",
        "fidelity_tolerance",
    }
)
_MISSING = object()


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def canonical_ptm_algorithm(algorithm: Any) -> str:
    """Return the canonical native Bayesian name accepted by PTM search."""
    normalized = _nonempty_string(algorithm, "algorithm").lower()
    if normalized not in {"bayesian", "b"}:
        raise ValueError(
            "Hierarchical PTM runtime supports only native "
            "algorithm='bayesian' (alias 'b')"
        )
    return "bayesian"


def _validate_algorithm(algorithm: Any) -> str:
    """Backward-compatible internal alias for algorithm canonicalization."""
    return canonical_ptm_algorithm(algorithm)


def _objective_mode(objective_config: Any) -> str:
    selection = getattr(objective_config, "selection_config", None)
    mode = getattr(selection, "mode", None)
    if mode not in _OBJECTIVE_MODES:
        raise ValueError(
            "PTM runtime requires an accuracy/latency multi-objective "
            "ObjectiveConfig with archive selection enabled"
        )
    if not getattr(objective_config, "is_multi_objective", False):
        raise ValueError(
            "PTM runtime objective-aware search requires both raw objectives"
        )
    return mode


def _artifact_is_current(artifact: Any, label: str) -> None:
    path = getattr(artifact, "path", None)
    if not isinstance(path, Path) or not path.is_file():
        raise ValueError(f"{label} verified artifact is no longer available")
    size = path.stat().st_size
    if size != getattr(artifact, "size_bytes", None):
        raise ValueError(f"{label} verified artifact size changed after preflight")
    if sha256_file(path) != getattr(artifact, "sha256", None):
        raise ValueError(
            f"{label} verified artifact checksum changed after preflight"
        )


def _prepared_provenance_payload(
    prepared: PreparedPTM,
    *,
    purpose: str,
) -> dict[str, Any]:
    payload = {
        "checkpoint_id": prepared.checkpoint_id,
        "purpose": purpose,
        "registry_status": prepared.registry_status,
        "runtime_eligible": prepared.runtime_eligible,
        "checkpoint": prepared.checkpoint.stable_dict(),
        "checkpoint_spec_artifact": (
            prepared.checkpoint_spec_artifact.stable_dict()
        ),
        "checkpoint_spec": prepared.checkpoint_spec.stable_dict(),
        "access_probe": prepared.access_probe.to_dict(),
        "load_smoke": prepared.load_smoke.stable_dict(),
        "registry_record_sha256": prepared.registry_record_sha256,
    }
    if prepared.source_checkpoint is not None:
        payload["source_checkpoint"] = prepared.source_checkpoint.stable_dict()
    if prepared.artifact_adaptation is not None:
        payload["artifact_adaptation"] = (
            prepared.artifact_adaptation.stable_dict()
        )
    return payload


def _validate_runtime_report(
    report: PTMPreflightReport,
    *,
    model: str,
    registry: PTMRegistry | None = None,
) -> Any:
    if not isinstance(report, PTMPreflightReport):
        raise TypeError(
            "report must be a live typed PTMPreflightReport; JSON reports "
            "cannot restore the validated checkpoint-spec document"
        )
    if report.purpose != "runtime":
        raise ValueError("PTM runtime builder rejects non-runtime preflight reports")
    if report.validation_statuses != ("supported",):
        raise ValueError(
            "Runtime preflight must contain only supported validation status"
        )
    if report.model != model:
        raise ValueError(
            f"Preflight model {report.model!r} does not match runtime model "
            f"{model!r}"
        )
    if not isinstance(report.inventory, PTMCompatibilityResult):
        raise ValueError(
            "Runtime preflight inventory must be PTMCompatibilityResult"
        )
    if (
        report.inventory.model != report.model
        or report.inventory.task != report.task
        or report.inventory.tao_version != report.tao_version
    ):
        raise ValueError("Runtime preflight inventory metadata is inconsistent")
    if not report.credential_probe.ok:
        raise ValueError("Runtime preflight did not pass its credential gate")
    if canonical_sha256(report.stable_dict()) != report.report_sha256:
        raise ValueError("PTM preflight report integrity verification failed")
    prepared_ids = tuple(item.checkpoint_id for item in report.prepared)
    if prepared_ids != tuple(sorted(prepared_ids)):
        raise ValueError("PTM preflight prepared inventory is not canonical")
    if len(set(prepared_ids)) != len(prepared_ids):
        raise ValueError("PTM preflight contains duplicate prepared checkpoints")
    compatible_ids = set(report.inventory.eligible_checkpoint_ids)
    if any(checkpoint_id not in compatible_ids for checkpoint_id in prepared_ids):
        raise ValueError(
            "PTM preflight prepared a checkpoint outside its runtime inventory"
        )

    if registry is None:
        registry = load_ptm_registry()
    elif not isinstance(registry, PTMRegistry):
        raise TypeError("registry must be a PTMRegistry when provided")
    if (
        report.registry_version != registry.registry_version
        or report.registry_sha256 != registry.document_sha256
    ):
        raise ValueError(
            "PTM preflight registry identity does not match the bound "
            "runtime registry"
        )
    for prepared in report.prepared:
        if (
            not prepared.runtime_eligible
            or prepared.registry_status != "supported"
            or not prepared.access_probe.ok
            or not prepared.load_smoke.ok
        ):
            raise ValueError(
                f"Prepared checkpoint {prepared.checkpoint_id!r} is not "
                "runtime eligible"
            )
        record = registry.checkpoint(prepared.checkpoint_id)
        if canonical_sha256(record) != prepared.registry_record_sha256:
            raise ValueError(
                f"Registry record changed after preflight for "
                f"{prepared.checkpoint_id!r}"
            )
        if record.get("model_family") != model or record.get("status") != "supported":
            raise ValueError(
                f"Prepared checkpoint {prepared.checkpoint_id!r} is not a "
                f"supported {model!r} record"
            )
        if canonical_sha256(prepared.checkpoint_spec.document) != (
            prepared.checkpoint_spec.document_sha256
        ):
            raise ValueError(
                f"Validated checkpoint spec changed after preflight for "
                f"{prepared.checkpoint_id!r}"
            )
        expected_provenance = canonical_sha256(
            _prepared_provenance_payload(prepared, purpose=report.purpose)
        )
        if expected_provenance != prepared.provenance_sha256:
            raise ValueError(
                f"Prepared checkpoint provenance is corrupt for "
                f"{prepared.checkpoint_id!r}"
            )
        _artifact_is_current(
            prepared.checkpoint,
            f"{prepared.checkpoint_id} checkpoint",
        )
        _artifact_is_current(
            prepared.checkpoint_spec_artifact,
            f"{prepared.checkpoint_id} checkpoint spec",
        )
    return registry


def _normalize_target_map(
    checkpoint_targets: str | Mapping[str, str] | None,
    checkpoint_ids: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    if checkpoint_targets is None:
        supplied = {
            checkpoint_id: records[checkpoint_id]["checkpoint_target"]
            for checkpoint_id in checkpoint_ids
        }
    elif isinstance(checkpoint_targets, str):
        target = _nonempty_string(checkpoint_targets, "checkpoint_targets")
        supplied = {checkpoint_id: target for checkpoint_id in checkpoint_ids}
    elif isinstance(checkpoint_targets, Mapping):
        if set(checkpoint_targets) != set(checkpoint_ids):
            raise ValueError(
                "checkpoint_targets must contain exactly the selected PTM IDs"
            )
        supplied = {
            checkpoint_id: _nonempty_string(
                checkpoint_targets[checkpoint_id],
                f"checkpoint_targets[{checkpoint_id!r}]",
            )
            for checkpoint_id in checkpoint_ids
        }
    else:
        raise TypeError("checkpoint_targets must be a string, mapping, or None")
    for checkpoint_id, target in supplied.items():
        registered = records[checkpoint_id]["checkpoint_target"]
        if target != registered:
            raise ValueError(
                f"Checkpoint target {target!r} for {checkpoint_id!r} does not "
                f"match registered target {registered!r}"
            )
    return supplied


def _get_path(spec: Mapping[str, Any], path: str, default: Any = _MISSING) -> Any:
    current: Any = spec
    for token in path.split("."):
        if not isinstance(current, Mapping) or token not in current:
            return default
        current = current[token]
    return current


def _normalize_layer(layer: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if layer is None:
        return {}
    if not isinstance(layer, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return merge_ptm_spec_precedence(model_defaults=layer).spec


def _arm_seed(random_seed: int, checkpoint_id: str) -> int:
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or not 0 <= random_seed < 2**32
    ):
        raise ValueError("random_seed must be an integer in [0, 2**32)")
    digest = hashlib.sha256(
        f"hierarchical-ptm:{random_seed}:{checkpoint_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


@dataclass(frozen=True)
class ResolvedPTMRuntimeArm:
    """One verified PTM and its effective base spec before generated values."""

    checkpoint_id: str
    checkpoint_target: str
    checkpoint_path: str
    effective_base_spec: Mapping[str, Any] = field(repr=False, compare=False)
    report_sha256: str
    registry_sha256: str
    registry_record_sha256: str
    preflight_provenance_sha256: str
    checkpoint_artifact_sha256: str
    checkpoint_spec_artifact_sha256: str
    checkpoint_spec_document_sha256: str
    input_contract_sha256: str
    ptm_layer_sha256: str
    effective_base_spec_sha256: str

    def stable_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_target": self.checkpoint_target,
            "checkpoint_path": self.checkpoint_path,
            "effective_base_spec": copy.deepcopy(dict(self.effective_base_spec)),
            "report_sha256": self.report_sha256,
            "registry_sha256": self.registry_sha256,
            "registry_record_sha256": self.registry_record_sha256,
            "preflight_provenance_sha256": (
                self.preflight_provenance_sha256
            ),
            "checkpoint_artifact_sha256": self.checkpoint_artifact_sha256,
            "checkpoint_spec_artifact_sha256": (
                self.checkpoint_spec_artifact_sha256
            ),
            "checkpoint_spec_document_sha256": (
                self.checkpoint_spec_document_sha256
            ),
            "input_contract_sha256": self.input_contract_sha256,
            "ptm_layer_sha256": self.ptm_layer_sha256,
            "effective_base_spec_sha256": self.effective_base_spec_sha256,
        }


@dataclass(frozen=True)
class ResolvedPTMRuntimeInventory:
    """Content-addressed first-stage runtime inventory."""

    report: PTMPreflightReport = field(repr=False, compare=False)
    algorithm: str
    mode: str
    model: str
    task: str
    tao_version: str
    ptm_policy: str
    user_checkpoint_id: str | None
    objective_config_sha256: str
    base_layers_sha256: Mapping[str, str]
    arms: tuple[ResolvedPTMRuntimeArm, ...]
    inventory_sha256: str
    runtime_registry: PTMRegistry | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def checkpoint_ids(self) -> tuple[str, ...]:
        return tuple(arm.checkpoint_id for arm in self.arms)

    def stable_dict(self) -> dict[str, Any]:
        spec_merge_precedence = [
            "model_defaults",
            "checkpoint_spec_document",
            "registry_default_spec_overrides",
            "automl_profile_overrides",
        ]
        if "per_checkpoint_profile_overrides" in self.base_layers_sha256:
            spec_merge_precedence.append(
                "per_checkpoint_profile_overrides"
            )
        spec_merge_precedence.extend(
            [
                "user_overrides",
                "verified_checkpoint_artifact_identity",
                "generated_candidate_values",
            ]
        )
        return {
            "schema_version": PTM_RUNTIME_SCHEMA_VERSION,
            "stage": "resolved_runtime_inventory",
            "algorithm": self.algorithm,
            "mode": self.mode,
            "model": self.model,
            "task": self.task,
            "tao_version": self.tao_version,
            "ptm_policy": self.ptm_policy,
            "user_checkpoint_id": self.user_checkpoint_id,
            "objective_config_sha256": self.objective_config_sha256,
            "base_layers_sha256": dict(self.base_layers_sha256),
            "report_sha256": self.report.report_sha256,
            "registry_sha256": self.report.registry_sha256,
            "spec_merge_precedence": spec_merge_precedence,
            "arms": [arm.stable_dict() for arm in self.arms],
            "algorithmic_campaign_flags": algorithmic_campaign_flags(),
        }

    def validate(self) -> None:
        if canonical_audit_sha256(self.stable_dict()) != self.inventory_sha256:
            raise ValueError("Resolved PTM runtime inventory integrity check failed")

    def to_dict(self) -> dict[str, Any]:
        value = self.stable_dict()
        value["inventory_sha256"] = self.inventory_sha256
        return value


def resolve_ptm_runtime_inventory(
    *,
    report: PTMPreflightReport,
    objective_config: Any,
    base_model_defaults: Mapping[str, Any],
    profile_overrides: Mapping[str, Any] | None = None,
    user_overrides: Mapping[str, Any] | None = None,
    checkpoint_targets: str | Mapping[str, str] | None = None,
    ptm_policy: str | None = None,
    user_checkpoint_id: str | None = None,
    execution_checkpoint_artifacts: (
        Mapping[str, Mapping[str, Any]] | None
    ) = None,
    per_checkpoint_profile_overrides: (
        Mapping[str, Mapping[str, Any]] | None
    ) = None,
    registry: PTMRegistry | None = None,
    model: str,
    algorithm: str = "bayesian",
) -> ResolvedPTMRuntimeInventory:
    """Resolve policy-compliant PTMs and effective pre-candidate base specs.

    ``execution_checkpoint_artifacts`` is an optional content-preserving path
    projection for remote runtimes such as SLURM.  Live PTM preflight still
    verifies the local cached bytes.  A caller may bind those exact bytes to a
    shared-filesystem execution path only by providing the same SHA-256 and
    byte size for every selected arm.  The runtime inventory records the
    projected path and the already verified content identity, so this cannot
    substitute a different checkpoint or bypass preflight.

    ``per_checkpoint_profile_overrides`` binds profile values that follow a
    checkpoint's registered runtime contract, such as its input resolution.
    It must contain exactly the selected checkpoint IDs.  These values are
    merged after the shared profile and before user overrides, preserving the
    documented precedence while keeping heterogeneous PTM arms comparable.

    ``registry`` may bind an explicit, already validated in-memory registry
    projection to the typed preflight report.  This is intentionally opt-in:
    omitting it preserves the packaged-registry trust boundary.  Supplying it
    never changes the repository registry; the report and every prepared arm
    must instead match the explicit registry's version, document digest, and
    canonical record digests exactly.
    """
    algorithm = _validate_algorithm(algorithm)
    model = _nonempty_string(model, "model")
    mode = _objective_mode(objective_config)
    registry = _validate_runtime_report(
        report,
        model=model,
        registry=registry,
    )
    if not report.prepared:
        raise ValueError("Runtime preflight has no prepared PTMs")
    prepared_by_id = {
        item.checkpoint_id: item for item in report.prepared
    }
    prepared_ids = tuple(sorted(prepared_by_id))

    if mode == "accuracy":
        policy = "default" if ptm_policy is None else str(ptm_policy).lower()
        if policy not in _ACCURACY_PTM_POLICIES:
            raise ValueError(
                "Accuracy PTM policy must be 'default', 'user', or 'all'"
            )
        if policy == "default":
            if user_checkpoint_id is not None:
                raise ValueError(
                    "user_checkpoint_id is valid only with ptm_policy='user'"
                )
            selected_ids = (report.inventory.default_checkpoint_id,)
        elif policy == "user":
            requested = _nonempty_string(
                user_checkpoint_id, "user_checkpoint_id"
            )
            selected_ids = (requested,)
        else:
            if user_checkpoint_id is not None:
                raise ValueError(
                    "user_checkpoint_id is valid only with ptm_policy='user'"
                )
            selected_ids = prepared_ids
    else:
        policy = "all" if ptm_policy is None else str(ptm_policy).lower()
        if policy != "all":
            raise ValueError(
                f"{mode} mode requires the complete prepared PTM inventory; "
                "narrower PTM policies are not allowed"
            )
        if user_checkpoint_id is not None:
            raise ValueError(
                f"{mode} mode cannot select one user_checkpoint_id"
            )
        selected_ids = prepared_ids

    if not selected_ids or selected_ids[0] is None:
        raise ValueError("Selected PTM policy did not resolve a checkpoint")
    missing = sorted(set(selected_ids) - set(prepared_ids))
    if missing:
        raise ValueError(
            "Selected checkpoint(s) did not pass runtime preflight: "
            + ", ".join(missing)
        )
    selected_ids = tuple(sorted(selected_ids))
    if execution_checkpoint_artifacts is None:
        execution_artifacts: dict[str, Mapping[str, Any]] = {}
    elif not isinstance(execution_checkpoint_artifacts, Mapping):
        raise TypeError(
            "execution_checkpoint_artifacts must be a mapping or None"
        )
    else:
        if set(execution_checkpoint_artifacts) != set(selected_ids):
            raise ValueError(
                "execution_checkpoint_artifacts must contain exactly the "
                "selected PTM checkpoint IDs"
            )
        execution_artifacts = {
            checkpoint_id: execution_checkpoint_artifacts[checkpoint_id]
            for checkpoint_id in selected_ids
        }
    checkpoint_profiles_supplied = (
        per_checkpoint_profile_overrides is not None
    )
    if per_checkpoint_profile_overrides is None:
        checkpoint_profiles: dict[str, Mapping[str, Any]] = {
            checkpoint_id: {} for checkpoint_id in selected_ids
        }
    elif not isinstance(per_checkpoint_profile_overrides, Mapping):
        raise TypeError(
            "per_checkpoint_profile_overrides must be a mapping or None"
        )
    else:
        if set(per_checkpoint_profile_overrides) != set(selected_ids):
            raise ValueError(
                "per_checkpoint_profile_overrides must contain exactly the "
                "selected PTM checkpoint IDs"
            )
        checkpoint_profiles = {}
        for checkpoint_id in selected_ids:
            raw_profile = per_checkpoint_profile_overrides[checkpoint_id]
            if not isinstance(raw_profile, Mapping):
                raise TypeError(
                    "per_checkpoint_profile_overrides values must be mappings"
                )
            checkpoint_profiles[checkpoint_id] = _normalize_layer(
                raw_profile,
                f"per_checkpoint_profile_overrides[{checkpoint_id!r}]",
            )
    all_prepared_records = {
        checkpoint_id: registry.checkpoint(checkpoint_id)
        for checkpoint_id in prepared_ids
    }
    records = {
        checkpoint_id: registry.checkpoint(checkpoint_id)
        for checkpoint_id in selected_ids
    }
    targets = _normalize_target_map(
        checkpoint_targets,
        selected_ids,
        records,
    )
    defaults = _normalize_layer(base_model_defaults, "base_model_defaults")
    profile = _normalize_layer(profile_overrides, "profile_overrides")
    user = _normalize_layer(user_overrides, "user_overrides")
    registered_runtime_targets = {
        record["checkpoint_target"]
        for record in all_prepared_records.values()
    }
    conflicts = [
        target
        for target in registered_runtime_targets
        if _get_path(user, target) is not _MISSING
    ]
    if conflicts:
        raise ValueError(
            "user_overrides cannot assign registry-resolved checkpoint "
            "target(s): " + ", ".join(sorted(set(conflicts)))
        )

    arms = []
    for checkpoint_id in selected_ids:
        prepared = prepared_by_id[checkpoint_id]
        record = records[checkpoint_id]
        execution_path = str(prepared.checkpoint.path)
        projected = execution_artifacts.get(checkpoint_id)
        if projected is not None:
            if not isinstance(projected, Mapping):
                raise TypeError(
                    "execution_checkpoint_artifacts values must be mappings"
                )
            if set(projected) != {"path", "sha256", "size_bytes"}:
                raise ValueError(
                    "each execution checkpoint artifact must contain exactly "
                    "path, sha256, and size_bytes"
                )
            raw_path = projected.get("path")
            if (
                not isinstance(raw_path, str)
                or not raw_path.strip()
                or not Path(raw_path).is_absolute()
            ):
                raise ValueError(
                    "execution checkpoint path must be a non-empty absolute "
                    "shared-filesystem path"
                )
            if (
                projected.get("sha256") != prepared.checkpoint.sha256
                or projected.get("size_bytes")
                != prepared.checkpoint.size_bytes
            ):
                raise ValueError(
                    f"execution checkpoint content identity does not match "
                    f"live preflight for {checkpoint_id!r}"
                )
            execution_path = raw_path.strip()
        # The official checkpoint document is the lower part of the PTM
        # layer. Repository-owned normalized overrides resolve any conflict.
        ptm_layer = merge_ptm_spec_precedence(
            model_defaults=prepared.checkpoint_spec.document,
            candidate_overrides=record["default_spec_overrides"],
        ).spec
        effective_profile = merge_ptm_spec_precedence(
            model_defaults=profile,
            candidate_overrides=checkpoint_profiles[checkpoint_id],
        ).spec
        target = targets[checkpoint_id]
        effective = merge_ptm_spec_precedence(
            model_defaults=defaults,
            ptm_overrides=ptm_layer,
            automl_profile_overrides=effective_profile,
            user_overrides=user,
            # The verified artifact identity is injected after user values.
            # Generated values are applied one level later by the wrapper.
            candidate_overrides={target: execution_path},
        ).spec
        input_contract_hash = canonical_sha256(record["input_contract"])
        ptm_layer_hash = canonical_sha256(ptm_layer)
        effective_hash = canonical_sha256(effective)
        arms.append(
            ResolvedPTMRuntimeArm(
                checkpoint_id=checkpoint_id,
                checkpoint_target=target,
                checkpoint_path=execution_path,
                effective_base_spec=effective,
                report_sha256=report.report_sha256,
                registry_sha256=report.registry_sha256,
                registry_record_sha256=prepared.registry_record_sha256,
                preflight_provenance_sha256=prepared.provenance_sha256,
                checkpoint_artifact_sha256=prepared.checkpoint.sha256,
                checkpoint_spec_artifact_sha256=(
                    prepared.checkpoint_spec_artifact.sha256
                ),
                checkpoint_spec_document_sha256=(
                    prepared.checkpoint_spec.document_sha256
                ),
                input_contract_sha256=input_contract_hash,
                ptm_layer_sha256=ptm_layer_hash,
                effective_base_spec_sha256=effective_hash,
            )
        )
    layers = {
        "model_defaults": canonical_sha256(defaults),
        "automl_profile_overrides": canonical_sha256(profile),
        "user_overrides": canonical_sha256(user),
    }
    if checkpoint_profiles_supplied:
        layers["per_checkpoint_profile_overrides"] = canonical_sha256(
            checkpoint_profiles
        )
    provisional = ResolvedPTMRuntimeInventory(
        report=report,
        algorithm=algorithm,
        mode=mode,
        model=model,
        task=report.task,
        tao_version=report.tao_version,
        ptm_policy=policy,
        user_checkpoint_id=user_checkpoint_id,
        objective_config_sha256=canonical_audit_sha256(
            objective_config.to_dict()
        ),
        base_layers_sha256=layers,
        arms=tuple(arms),
        inventory_sha256="",
        runtime_registry=registry,
    )
    resolved = ResolvedPTMRuntimeInventory(
        **{
            **provisional.__dict__,
            "inventory_sha256": canonical_audit_sha256(
                provisional.stable_dict()
            ),
        }
    )
    resolved.validate()
    return resolved


class _ArmConstructionStateStore:
    """Expose a PTM-effective spec/range only to one inner constructor."""

    def __init__(
        self,
        delegate: Any,
        *,
        job_spec: Mapping[str, Any],
        custom_ranges: Mapping[str, Any],
    ):
        self._delegate = delegate
        self._job_spec = copy.deepcopy(dict(job_spec))
        self._custom_ranges = copy.deepcopy(dict(custom_ranges))
        self._brain_state = None

    def get_job_specs(self, _job_id: str) -> dict[str, Any]:
        return copy.deepcopy(self._job_spec)

    def get_custom_param_ranges(self, _handler_id: str) -> dict[str, Any]:
        return copy.deepcopy(self._custom_ranges)

    def get_brain_info(self, _job_id: str) -> Any:
        return copy.deepcopy(self._brain_state)

    def save_brain_info(self, _job_id: str, state: Any) -> None:
        self._brain_state = copy.deepcopy(state)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


def _context_for_arm(context: Any, seed: int) -> Any:
    arm_context = copy.copy(context)
    try:
        setattr(arm_context, "random_seed", seed)
    except (AttributeError, TypeError) as exc:
        raise TypeError(
            "context must support a per-arm random_seed copy"
        ) from exc
    return arm_context


def _normalize_conditional_mapping(
    value: Mapping[str, Any],
    checkpoint_ids: Sequence[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping keyed by checkpoint ID")
    if set(value) != set(checkpoint_ids):
        raise ValueError(
            f"{name} must contain exactly the resolved PTM checkpoint IDs"
        )
    return {
        checkpoint_id: copy.deepcopy(value[checkpoint_id])
        for checkpoint_id in checkpoint_ids
    }


def _validate_parameter_records(
    records: Any,
    *,
    checkpoint_id: str,
    checkpoint_target: str,
) -> list[dict[str, Any]]:
    if (
        not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
        or not records
    ):
        raise ValueError(
            f"conditional_parameters[{checkpoint_id!r}] must be a non-empty "
            "sequence"
        )
    result = []
    names = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(
                f"conditional parameter {checkpoint_id!r}[{index}] must be "
                "a mapping"
            )
        normalized = copy.deepcopy(dict(record))
        name = normalized.get("parameter")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"conditional parameter {checkpoint_id!r}[{index}] has no "
                "valid parameter name"
            )
        name = name.strip()
        if (
            name == checkpoint_target
            or name.startswith(f"{checkpoint_target}.")
            or name.startswith(f"{checkpoint_target}[")
            or checkpoint_target.startswith(f"{name}.")
            or checkpoint_target.startswith(f"{name}[")
        ):
            raise ValueError(
                f"Checkpoint target {checkpoint_target!r} overlaps searchable "
                f"parameter {name!r} for {checkpoint_id!r}"
            )
        names.append(name)
        result.append(normalized)
    if len(set(names)) != len(names):
        raise ValueError(
            f"conditional_parameters[{checkpoint_id!r}] contains duplicate "
            "parameter names"
        )
    return result


@dataclass(frozen=True)
class PTMRuntimeBuild:
    """Built controller brain plus its immutable construction manifest."""

    brain: HierarchicalPTMBrain = field(repr=False, compare=False)
    resolved_inventory: ResolvedPTMRuntimeInventory = field(
        repr=False,
        compare=False,
    )
    arms: tuple[PTMArm, ...]
    per_arm_seeds: Mapping[str, int]
    manifest: Mapping[str, Any]
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        value = copy.deepcopy(dict(self.manifest))
        value["manifest_sha256"] = self.manifest_sha256
        return value


def build_hierarchical_ptm_runtime(
    *,
    resolved_inventory: ResolvedPTMRuntimeInventory,
    objective_config: Any,
    conditional_parameters: Mapping[str, Sequence[Mapping[str, Any]]],
    conditional_ranges: Mapping[str, Mapping[str, Any]],
    context: Any,
    state_store: Any,
    random_seed: int,
    acquisition_settings: Mapping[str, Any] | None = None,
    scheduler_options: Mapping[str, Any] | None = None,
    fidelity_metric: str | None = None,
    algorithm: str = "bayesian",
    resume: bool = False,
) -> PTMRuntimeBuild:
    """Build deterministic per-arm native Bayesian brains and their wrapper."""
    algorithm = _validate_algorithm(algorithm)
    if not isinstance(resolved_inventory, ResolvedPTMRuntimeInventory):
        raise TypeError(
            "resolved_inventory must come from resolve_ptm_runtime_inventory"
        )
    resolved_inventory.validate()
    if resolved_inventory.algorithm != algorithm:
        raise ValueError("Resolved inventory algorithm does not match build")
    mode = _objective_mode(objective_config)
    if mode != resolved_inventory.mode:
        raise ValueError("Objective mode changed after PTM inventory resolution")
    objective_hash = canonical_audit_sha256(objective_config.to_dict())
    if objective_hash != resolved_inventory.objective_config_sha256:
        raise ValueError(
            "Objective configuration changed after PTM inventory resolution"
        )
    # Revalidate the live report and repository registry at the second trust
    # boundary; resolved mappings are mutable Python objects despite frozen
    # container dataclasses.
    _validate_runtime_report(
        resolved_inventory.report,
        model=resolved_inventory.model,
        registry=resolved_inventory.runtime_registry,
    )

    checkpoint_ids = resolved_inventory.checkpoint_ids
    raw_parameters = _normalize_conditional_mapping(
        conditional_parameters,
        checkpoint_ids,
        "conditional_parameters",
    )
    raw_ranges = _normalize_conditional_mapping(
        conditional_ranges,
        checkpoint_ids,
        "conditional_ranges",
    )
    if scheduler_options is None:
        scheduler_values = {}
    elif isinstance(scheduler_options, Mapping):
        scheduler_values = dict(scheduler_options)
    else:
        raise TypeError("scheduler_options must be a mapping")
    unknown_scheduler = sorted(set(scheduler_values) - _SCHEDULER_OPTION_KEYS)
    if unknown_scheduler:
        raise ValueError(
            "Unsupported hierarchical scheduler option(s): "
            + ", ".join(unknown_scheduler)
        )
    selection = objective_config.selection_config
    constraint = selection.latency_accuracy_retention
    if constraint is not None and constraint.kind == "relative":
        retention = constraint.value
    elif mode == "latency":
        if constraint is None or constraint.kind != "relative":
            raise ValueError(
                "Hierarchical constrained-latency allocation currently "
                "requires a relative retained-accuracy policy"
            )
    else:
        # The scheduler field is unused by accuracy and multi-objective outer
        # allocation. One is a neutral value and must not be interpreted as a
        # multi-objective accuracy floor.
        retention = 1.0
    arms = []
    inner_brains = {}
    candidate_overrides = {}
    checkpoint_targets = {}
    per_arm_seeds = {}
    arm_manifests = []
    resolved_by_id = {
        arm.checkpoint_id: arm for arm in resolved_inventory.arms
    }
    for checkpoint_id in checkpoint_ids:
        resolved_arm = resolved_by_id[checkpoint_id]
        parameters = _validate_parameter_records(
            raw_parameters[checkpoint_id],
            checkpoint_id=checkpoint_id,
            checkpoint_target=resolved_arm.checkpoint_target,
        )
        ranges = raw_ranges[checkpoint_id]
        if not isinstance(ranges, Mapping):
            raise TypeError(
                f"conditional_ranges[{checkpoint_id!r}] must be a mapping"
            )
        ranges = copy.deepcopy(dict(ranges))
        parameter_names = {item["parameter"] for item in parameters}
        unknown_ranges = sorted(set(ranges) - parameter_names)
        if unknown_ranges:
            raise ValueError(
                f"conditional_ranges[{checkpoint_id!r}] references unknown "
                "parameter(s): " + ", ".join(unknown_ranges)
            )
        search_space_payload = {
            "parameters": audit_json_value(parameters),
            "custom_ranges": audit_json_value(ranges),
        }
        search_space_hash = canonical_audit_sha256(search_space_payload)
        arm = PTMArm(
            checkpoint_id=checkpoint_id,
            conditional_search_space_sha256=search_space_hash,
            preflight_provenance_sha256=(
                resolved_arm.preflight_provenance_sha256
            ),
            input_contract_sha256=resolved_arm.input_contract_sha256,
        )
        arm_seed = _arm_seed(random_seed, checkpoint_id)
        construction_store = _ArmConstructionStateStore(
            state_store,
            job_spec=resolved_arm.effective_base_spec,
            custom_ranges=ranges,
        )
        inner = Bayesian(
            _context_for_arm(context, arm_seed),
            construction_store,
            resolved_inventory.model,
            parameters,
            metric=objective_config.brain_metric,
            direction=objective_config.score_direction,
            objective_config=objective_config,
            acquisition_settings=(
                copy.deepcopy(dict(acquisition_settings))
                if acquisition_settings is not None
                else None
            ),
        )
        # Assert the per-arm constructor saw the intended inputs before the
        # wrapper swaps in its embedded persistence adapter.
        if canonical_sha256(inner.default_train_spec) != (
            resolved_arm.effective_base_spec_sha256
        ):
            raise RuntimeError(
                f"Inner Bayesian {checkpoint_id!r} did not receive its "
                "PTM-effective base specification"
            )
        if canonical_audit_sha256(inner.custom_ranges) != (
            canonical_audit_sha256(ranges)
        ):
            raise RuntimeError(
                f"Inner Bayesian {checkpoint_id!r} did not receive its "
                "conditional custom ranges"
            )
        arms.append(arm)
        inner_brains[checkpoint_id] = inner
        candidate_overrides[checkpoint_id] = copy.deepcopy(
            dict(resolved_arm.effective_base_spec)
        )
        checkpoint_targets[checkpoint_id] = resolved_arm.checkpoint_target
        per_arm_seeds[checkpoint_id] = arm_seed
        arm_manifests.append(
            {
                **resolved_arm.stable_dict(),
                "ptm_arm_sha256": canonical_audit_sha256(arm.to_dict()),
                "conditional_parameters_sha256": canonical_audit_sha256(
                    parameters
                ),
                "conditional_ranges_sha256": canonical_audit_sha256(ranges),
                "conditional_search_space_sha256": search_space_hash,
                "per_arm_seed": arm_seed,
                "constructor_base_spec_sha256": canonical_sha256(
                    inner.default_train_spec
                ),
                "constructor_custom_ranges_sha256": (
                    canonical_audit_sha256(inner.custom_ranges)
                ),
                "inner_calibration_points": inner.calibration_points,
            }
        )

    required_inner_calibration = max(
        inner.calibration_points for inner in inner_brains.values()
    )
    configured_initial_issues = scheduler_values.get(
        "initial_issues_per_arm"
    )
    if configured_initial_issues is None:
        scheduler_values["initial_issues_per_arm"] = (
            required_inner_calibration
        )
    elif configured_initial_issues < required_inner_calibration:
        raise ValueError(
            "initial_issues_per_arm cannot be lower than the largest inner "
            "Bayesian calibration requirement: "
            f"{configured_initial_issues} < {required_inner_calibration}"
        )
    policy = HierarchicalPTMPolicy(
        mode=mode,
        latency_accuracy_retention=retention,
        accuracy_tolerance=selection.accuracy_tolerance,
        **scheduler_values,
    )

    scheduler = HierarchicalPTMScheduler(
        tuple(arms),
        policy,
        random_seed=random_seed,
    )
    wrapper_kwargs = {
        "context": context,
        "state_store": state_store,
        "scheduler": scheduler,
        "inner_brains": inner_brains,
        "candidate_overrides": candidate_overrides,
        "checkpoint_targets": checkpoint_targets,
        "accuracy_metric": selection.accuracy_metric,
        "latency_metric": selection.latency_metric,
        "fidelity_metric": fidelity_metric,
    }
    brain = (
        HierarchicalPTMBrain.load_state(**wrapper_kwargs)
        if resume
        else HierarchicalPTMBrain(**wrapper_kwargs)
    )
    # Bayesian.load_state reconstructs through the wrapper's embedded state
    # adapter. Reassert the already signature-bound per-arm constructor inputs
    # because the shared outer store intentionally does not own PTM-conditional
    # defaults or ranges.
    for checkpoint_id in checkpoint_ids:
        restored_inner = brain.inner_brains[checkpoint_id]
        restored_inner.default_train_spec = copy.deepcopy(
            dict(resolved_by_id[checkpoint_id].effective_base_spec)
        )
        restored_inner.default_train_spec_flattened = {}
        restored_inner.custom_ranges = copy.deepcopy(
            dict(raw_ranges[checkpoint_id])
        )
    manifest = {
        "schema_version": PTM_RUNTIME_SCHEMA_VERSION,
        "stage": "built_hierarchical_ptm_runtime",
        "algorithm": algorithm,
        "mode": mode,
        "model": resolved_inventory.model,
        "task": resolved_inventory.task,
        "tao_version": resolved_inventory.tao_version,
        "random_seed": random_seed,
        "resolved_inventory_sha256": resolved_inventory.inventory_sha256,
        "objective_config_sha256": objective_hash,
        "scheduler_signature_sha256": scheduler.signature_sha256,
        "wrapper_signature_sha256": brain.signature_sha256,
        "scheduler_policy": policy.to_dict(),
        "required_inner_calibration_issues_per_arm": (
            required_inner_calibration
        ),
        "latency_retention_active": mode == "latency",
        "acquisition_settings": audit_json_value(
            acquisition_settings or {}
        ),
        "fidelity_metric": fidelity_metric,
        "arms": arm_manifests,
        "algorithmic_campaign_flags": algorithmic_campaign_flags(),
    }
    manifest_hash = canonical_audit_sha256(manifest)
    return PTMRuntimeBuild(
        brain=brain,
        resolved_inventory=resolved_inventory,
        arms=tuple(arms),
        per_arm_seeds=per_arm_seeds,
        manifest=manifest,
        manifest_sha256=manifest_hash,
    )


__all__ = [
    "PTM_RUNTIME_SCHEMA_VERSION",
    "PTMRuntimeBuild",
    "ResolvedPTMRuntimeArm",
    "ResolvedPTMRuntimeInventory",
    "build_hierarchical_ptm_runtime",
    "canonical_ptm_algorithm",
    "resolve_ptm_runtime_inventory",
]
