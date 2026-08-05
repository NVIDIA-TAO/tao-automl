# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable contracts for staged, independent SLURM AutoML campaigns.

This module validates and seals experiment intent.  It does not submit jobs,
resolve containers, download artifacts, or mutate scheduler state.

The contract deliberately keeps shared fairness inputs at campaign scope and
stores only mode-specific acquisition state in each of the three independent
job records.  A derived per-mode manifest binds that job to the sealed shared
inputs and can be persisted beside scheduler artifacts for safe resume.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from tao_automl.ptm_registry import canonical_sha256


CAMPAIGN_MANIFEST_SCHEMA_VERSION = 1
CAMPAIGN_MODES = ("accuracy", "latency", "multi_objective")
CAMPAIGN_STAGES = (
    "single_candidate_gate",
    "pilot_batch",
    "full_search",
    "matched_validation",
)
MAX_RETRIES_PER_TRIAL = 5

AGENT_INTERVENTION_FLAGS = (
    "agent_selected_candidate",
    "agent_injected_candidate",
    "agent_modified_search_space_after_results",
    "agent_changed_seed_after_results",
    "agent_changed_budget_after_results",
    "agent_changed_threshold_after_results",
    "agent_changed_ptm_after_results",
    "agent_overrode_winner",
)
SELECTION_ISOLATION_FLAGS = (
    "selector_invoked_on_matched_measurements",
    "selection_time_objectives_replaced",
    "measurements_feed_selection",
    "measurements_feed_reselection",
    "algorithm_selected_candidate_overridden",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]*$")
_MODE_INDEX = {mode: index for index, mode in enumerate(CAMPAIGN_MODES)}
_STRICT_MODE_ACQUISITIONS = {
    "accuracy": "expected_improvement",
    "latency": "constrained_expected_improvement",
    "multi_objective": "parego_expected_improvement",
}

_SECTION_KEYS = (
    "source",
    "package",
    "container",
    "runtime",
    "ptms",
    "ptm_search",
    "dataset",
    "algorithm",
    "search_space",
    "budget",
    "fidelity",
    "resources",
    "workload",
    "retry_policy",
    "stages",
    "cancellation",
    "failed_trial_policy",
    "agent_intervention_flags",
    "selection_isolation_flags",
    "modes",
)


class CampaignManifestValidationError(ValueError):
    """Raised when a campaign document violates the immutable contract."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__(
            "Invalid campaign manifest:\n"
            + "\n".join(f"- {error}" for error in self.errors)
        )


class CampaignResumeMismatchError(RuntimeError):
    """Raised before resume when persisted intent differs from current intent."""

    def __init__(
        self,
        *,
        expected_sha256: str,
        observed_sha256: str | None,
        scope: str,
        reason: str,
    ):
        self.expected_sha256 = expected_sha256
        self.observed_sha256 = observed_sha256
        self.scope = scope
        self.reason = reason
        super().__init__(
            f"{scope} resume manifest mismatch: {reason}; "
            f"expected_sha256={expected_sha256}, "
            f"observed_sha256={observed_sha256 or '<unavailable>'}"
        )


@dataclass(frozen=True)
class CampaignFairnessAudit:
    """Deterministic evidence for cross-mode independence and fairness."""

    passed_checks: tuple[str, ...]
    violations: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "passed_checks": list(self.passed_checks),
            "violations": list(self.violations),
        }


def _canonical_roundtrip(value: Any) -> Any:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CampaignManifestValidationError(
            (f"manifest must be finite canonical JSON: {exc}",)
        ) from exc
    return json.loads(encoded)


def _normalized_document(document: Mapping[str, Any]) -> dict[str, Any]:
    value = _canonical_roundtrip(document)
    if not isinstance(value, dict):
        raise CampaignManifestValidationError(("manifest root must be an object",))

    ptms = value.get("ptms")
    if isinstance(ptms, list) and all(isinstance(item, dict) for item in ptms):
        ptms.sort(key=lambda item: str(item.get("id", "")))
    ptm_search = value.get("ptm_search")
    if isinstance(ptm_search, dict):
        arms = ptm_search.get("arms")
        if isinstance(arms, list) and all(isinstance(item, dict) for item in arms):
            arms.sort(key=lambda item: str(item.get("checkpoint_id", "")))
    modes = value.get("modes")
    if isinstance(modes, list) and all(isinstance(item, dict) for item in modes):
        modes.sort(
            key=lambda item: _MODE_INDEX.get(str(item.get("mode")), len(_MODE_INDEX))
        )
        for item in modes:
            identifiers = item.get("allowed_ptm_ids")
            if isinstance(identifiers, list):
                identifiers.sort(key=str)
    retry = value.get("retry_policy")
    if isinstance(retry, dict) and isinstance(
        retry.get("retryable_failure_codes"), list
    ):
        retry["retryable_failure_codes"].sort(key=str)
    cancellation = value.get("cancellation")
    if isinstance(cancellation, dict) and isinstance(
        cancellation.get("criteria"), list
    ):
        cancellation["criteria"].sort(key=str)
    stages = value.get("stages")
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            for field in ("entry_criteria", "exit_criteria"):
                if isinstance(stage.get(field), list):
                    stage[field].sort(key=str)
    return value


def _require_keys(
    value: Any,
    *,
    path: str,
    required: Sequence[str],
    errors: list[str],
    optional: Sequence[str] = (),
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{path} must be an object")
        return None
    keys = set(value)
    missing = sorted(set(required) - keys)
    unknown = sorted(keys - set(required) - set(optional))
    for key in missing:
        errors.append(f"{path}.{key} is required")
    for key in unknown:
        errors.append(f"{path}.{key} is not a recognized field")
    return value


def _nonempty_string(
    value: Any,
    *,
    path: str,
    errors: list[str],
    identifier: bool = False,
) -> str | None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path} must be a non-empty string")
        return None
    if identifier and _IDENTIFIER_RE.fullmatch(value) is None:
        errors.append(f"{path} contains unsupported identifier characters")
        return None
    return value


def _sha256(value: Any, *, path: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        errors.append(f"{path} must be a lowercase 64-character SHA-256")
        return None
    return value


def _integer(
    value: Any,
    *,
    path: str,
    errors: list[str],
    minimum: int = 0,
) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        errors.append(f"{path} must be an integer >= {minimum}")
        return None
    return value


def _finite_number(
    value: Any,
    *,
    path: str,
    errors: list[str],
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        errors.append(f"{path} must be a finite number")
        return None
    number = float(value)
    if minimum is not None and number < minimum:
        errors.append(f"{path} must be >= {minimum}")
    if maximum is not None and number > maximum:
        errors.append(f"{path} must be <= {maximum}")
    return number


def _boolean(
    value: Any,
    *,
    path: str,
    errors: list[str],
    expected: bool | None = None,
) -> bool | None:
    if not isinstance(value, bool):
        errors.append(f"{path} must be boolean")
        return None
    if expected is not None and value is not expected:
        errors.append(f"{path} must be {str(expected).lower()}")
    return value


def _string_list(
    value: Any,
    *,
    path: str,
    errors: list[str],
    allow_empty: bool = False,
) -> tuple[str, ...] | None:
    if (
        not isinstance(value, list)
        or (not allow_empty and not value)
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        qualifier = "possibly empty" if allow_empty else "non-empty"
        errors.append(f"{path} must be a {qualifier} string list")
        return None
    if len(value) != len(set(value)):
        errors.append(f"{path} must not contain duplicates")
    return tuple(value)


def _safe_identity_location(
    value: Any,
    *,
    path: str,
    errors: list[str],
) -> None:
    text = _nonempty_string(value, path=path, errors=errors)
    if text is None:
        return
    if "?" in text or "#" in text:
        errors.append(
            f"{path} must be an immutable credential-free identity without "
            "query or fragment"
        )
        return
    if "://" not in text:
        return
    parsed = urlsplit(text)
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        errors.append(
            f"{path} must be an immutable credential-free identity without "
            "query or fragment"
        )


def _validate_source(document: Mapping[str, Any], errors: list[str]) -> None:
    value = _require_keys(
        document.get("source"),
        path="source",
        required=("commit", "dirty_tree_policy", "dirty", "diff_sha256"),
        errors=errors,
    )
    if value is None:
        return
    commit = value.get("commit")
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        errors.append("source.commit must be a full lowercase Git object ID")
    policy = value.get("dirty_tree_policy")
    if policy not in ("reject", "allow_with_diff_hash"):
        errors.append(
            "source.dirty_tree_policy must be 'reject' or "
            "'allow_with_diff_hash'"
        )
    dirty = _boolean(value.get("dirty"), path="source.dirty", errors=errors)
    diff_hash = value.get("diff_sha256")
    if dirty is True:
        if policy == "reject":
            errors.append("source.dirty cannot be true when dirty-tree policy rejects it")
        _sha256(diff_hash, path="source.diff_sha256", errors=errors)
    elif dirty is False and diff_hash is not None:
        errors.append("source.diff_sha256 must be null for a clean tree")


def _validate_package(document: Mapping[str, Any], errors: list[str]) -> None:
    value = _require_keys(
        document.get("package"),
        path="package",
        required=("distribution", "version", "wheel_sha256"),
        errors=errors,
    )
    if value is None:
        return
    _nonempty_string(
        value.get("distribution"),
        path="package.distribution",
        errors=errors,
        identifier=True,
    )
    _nonempty_string(value.get("version"), path="package.version", errors=errors)
    _sha256(value.get("wheel_sha256"), path="package.wheel_sha256", errors=errors)


def _validate_container(document: Mapping[str, Any], errors: list[str]) -> None:
    value = _require_keys(
        document.get("container"),
        path="container",
        required=(
            "runtime",
            "sqsh_uri",
            "sqsh_sha256",
            "container_digest",
        ),
        errors=errors,
    )
    if value is None:
        return
    if value.get("runtime") != "enroot":
        errors.append("container.runtime must be 'enroot' for the SQSH contract")
    _safe_identity_location(
        value.get("sqsh_uri"),
        path="container.sqsh_uri",
        errors=errors,
    )
    sqsh_uri = value.get("sqsh_uri")
    if isinstance(sqsh_uri, str) and not urlsplit(sqsh_uri).path.endswith(".sqsh"):
        errors.append("container.sqsh_uri must identify an exact .sqsh image")
    _sha256(value.get("sqsh_sha256"), path="container.sqsh_sha256", errors=errors)
    digest = value.get("container_digest")
    if (
        not isinstance(digest, str)
        or not digest.startswith("sha256:")
        or _SHA256_RE.fullmatch(digest[7:]) is None
    ):
        errors.append(
            "container.container_digest must be an exact lowercase sha256 digest"
        )


def _validate_runtime(document: Mapping[str, Any], errors: list[str]) -> None:
    value = _require_keys(
        document.get("runtime"),
        path="runtime",
        required=(
            "tao_version",
            "precision",
            "train_batch_size_per_gpu",
            "eval_batch_size_per_gpu",
            "latency_batch_size_per_gpu",
            "latency_protocol_sha256",
            "latency_input_sha256",
            "latency_timed_scope",
        ),
        errors=errors,
    )
    if value is None:
        return
    _nonempty_string(
        value.get("tao_version"),
        path="runtime.tao_version",
        errors=errors,
    )
    _nonempty_string(
        value.get("precision"),
        path="runtime.precision",
        errors=errors,
    )
    for field in (
        "train_batch_size_per_gpu",
        "eval_batch_size_per_gpu",
        "latency_batch_size_per_gpu",
    ):
        _integer(
            value.get(field),
            path=f"runtime.{field}",
            errors=errors,
            minimum=1,
        )
    for field in ("latency_protocol_sha256", "latency_input_sha256"):
        _sha256(value.get(field), path=f"runtime.{field}", errors=errors)
    _nonempty_string(
        value.get("latency_timed_scope"),
        path="runtime.latency_timed_scope",
        errors=errors,
    )


def _validate_ptms(document: Mapping[str, Any], errors: list[str]) -> tuple[str, ...]:
    values = document.get("ptms")
    if not isinstance(values, list) or not values:
        errors.append("ptms must be a non-empty list")
        return ()
    identifiers: list[str] = []
    for index, item in enumerate(values):
        path = f"ptms[{index}]"
        record = _require_keys(
            item,
            path=path,
            required=(
                "id",
                "artifact_sha256",
                "registry_record_sha256",
                "preflight_report_sha256",
            ),
            errors=errors,
        )
        if record is None:
            continue
        identifier = _nonempty_string(
            record.get("id"),
            path=f"{path}.id",
            errors=errors,
            identifier=True,
        )
        if identifier is not None:
            identifiers.append(identifier)
        for field in (
            "artifact_sha256",
            "registry_record_sha256",
            "preflight_report_sha256",
        ):
            _sha256(record.get(field), path=f"{path}.{field}", errors=errors)
    if len(identifiers) != len(set(identifiers)):
        errors.append("ptms IDs must be unique")
    return tuple(sorted(identifiers))


def _validate_ptm_search(
    document: Mapping[str, Any],
    errors: list[str],
    *,
    ptm_ids: tuple[str, ...],
) -> str | None:
    value = _require_keys(
        document.get("ptm_search"),
        path="ptm_search",
        required=("representation", "default_ptm_id", "arms"),
        errors=errors,
    )
    if value is None:
        return None
    if value.get("representation") != "hierarchical_nonordinal_arms":
        errors.append(
            "ptm_search.representation must be "
            "'hierarchical_nonordinal_arms'"
        )
    default_ptm = _nonempty_string(
        value.get("default_ptm_id"),
        path="ptm_search.default_ptm_id",
        errors=errors,
        identifier=True,
    )
    if default_ptm is not None and default_ptm not in ptm_ids:
        errors.append("ptm_search.default_ptm_id must be in the PTM inventory")

    arms = value.get("arms")
    arm_ids: list[str] = []
    if not isinstance(arms, list) or not arms:
        errors.append("ptm_search.arms must be a non-empty list")
    else:
        for index, item in enumerate(arms):
            path = f"ptm_search.arms[{index}]"
            arm = _require_keys(
                item,
                path=path,
                required=(
                    "checkpoint_id",
                    "conditional_search_space_sha256",
                    "preflight_provenance_sha256",
                    "input_contract_sha256",
                ),
                errors=errors,
            )
            if arm is None:
                continue
            checkpoint_id = _nonempty_string(
                arm.get("checkpoint_id"),
                path=f"{path}.checkpoint_id",
                errors=errors,
                identifier=True,
            )
            if checkpoint_id is not None:
                arm_ids.append(checkpoint_id)
            for field in (
                "conditional_search_space_sha256",
                "preflight_provenance_sha256",
                "input_contract_sha256",
            ):
                _sha256(arm.get(field), path=f"{path}.{field}", errors=errors)
    if tuple(sorted(arm_ids)) != ptm_ids:
        errors.append(
            "ptm_search.arms must contain every qualified PTM exactly once"
        )
    return default_ptm


def _validate_dataset(document: Mapping[str, Any], errors: list[str]) -> None:
    value = _require_keys(
        document.get("dataset"),
        path="dataset",
        required=(
            "id",
            "source",
            "manifest_sha256",
            "conversion_sha256",
            "splits",
        ),
        errors=errors,
    )
    if value is None:
        return
    _nonempty_string(
        value.get("id"),
        path="dataset.id",
        errors=errors,
        identifier=True,
    )
    _safe_identity_location(value.get("source"), path="dataset.source", errors=errors)
    _sha256(
        value.get("manifest_sha256"),
        path="dataset.manifest_sha256",
        errors=errors,
    )
    _sha256(
        value.get("conversion_sha256"),
        path="dataset.conversion_sha256",
        errors=errors,
    )
    splits = value.get("splits")
    if not isinstance(splits, Mapping):
        errors.append("dataset.splits must be an object")
        return
    for required_split in ("train", "validation"):
        if required_split not in splits:
            errors.append(f"dataset.splits.{required_split} is required")
    for split, digest in sorted(splits.items(), key=lambda item: str(item[0])):
        if not isinstance(split, str) or not split:
            errors.append("dataset.splits keys must be non-empty strings")
            continue
        _sha256(digest, path=f"dataset.splits.{split}", errors=errors)


def _validate_algorithm(document: Mapping[str, Any], errors: list[str]) -> None:
    value = _require_keys(
        document.get("algorithm"),
        path="algorithm",
        required=(
            "name",
            "implementation_version",
            "acquisition_version",
            "deterministic_replay",
        ),
        errors=errors,
    )
    if value is None:
        return
    for field in ("name", "implementation_version", "acquisition_version"):
        _nonempty_string(value.get(field), path=f"algorithm.{field}", errors=errors)
    _boolean(
        value.get("deterministic_replay"),
        path="algorithm.deterministic_replay",
        errors=errors,
        expected=True,
    )
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        return
    # Import lazily so the pure manifest module does not eagerly initialize
    # every production brain. A strict campaign may make product-level
    # objective-aware claims only when each of its three modes is implemented
    # as objective-aware acquisition, not a scalar archive-feedback fallback.
    from tao_automl.brain.factory import BrainFactory

    try:
        capability = BrainFactory.objective_capabilities().resolve(name)
    except (RuntimeError, ValueError) as exc:
        errors.append(f"algorithm.name is not a registered AutoML algorithm: {exc}")
        return
    for mode in CAMPAIGN_MODES:
        mode_capability = capability.for_mode(mode)
        if not mode_capability.supported or not mode_capability.objective_aware:
            errors.append(
                f"algorithm {capability.algorithm!r} is not objective-aware "
                f"for strict campaign mode {mode!r}: "
                f"{mode_capability.support_level}"
            )


def _validate_search_space(
    document: Mapping[str, Any],
    errors: list[str],
    *,
    ptm_ids: tuple[str, ...],
) -> str | None:
    value = _require_keys(
        document.get("search_space"),
        path="search_space",
        required=("parameters", "sha256"),
        errors=errors,
    )
    if value is None:
        return None
    parameters = value.get("parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        errors.append("search_space.parameters must be a non-empty object")
        return None
    forbidden_names = {
        "ptm_id",
        "checkpoint_id",
        "model.ptm_id",
        "train.pretrained_model_path",
    }
    for name, specification in parameters.items():
        if str(name).lower() in forbidden_names:
            errors.append(
                "search_space.parameters must not encode PTM identity as an "
                "inner ordinal/categorical parameter; use ptm_search arms"
            )
        if isinstance(specification, Mapping):
            values = specification.get("values")
            if isinstance(values, list) and set(ptm_ids).intersection(
                item for item in values if isinstance(item, str)
            ):
                errors.append(
                    f"search_space.parameters.{name} embeds PTM IDs; PTM "
                    "selection must use hierarchical non-ordinal arms"
                )
    try:
        expected = canonical_sha256(parameters)
    except ValueError as exc:
        errors.append(f"search_space.parameters is not canonical JSON: {exc}")
        return None
    observed = _sha256(value.get("sha256"), path="search_space.sha256", errors=errors)
    if observed is not None and observed != expected:
        errors.append(
            "search_space.sha256 does not match the canonical parameter domain"
        )
    return expected


def _validate_budget(document: Mapping[str, Any], errors: list[str]) -> int | None:
    value = _require_keys(
        document.get("budget"),
        path="budget",
        required=(
            "max_candidates_per_mode",
            "max_concurrent_candidates_per_mode",
            "max_wallclock_minutes_per_mode",
            "max_terminal_failures_per_mode",
        ),
        errors=errors,
    )
    if value is None:
        return None
    maximum = _integer(
        value.get("max_candidates_per_mode"),
        path="budget.max_candidates_per_mode",
        errors=errors,
        minimum=3,
    )
    concurrent = _integer(
        value.get("max_concurrent_candidates_per_mode"),
        path="budget.max_concurrent_candidates_per_mode",
        errors=errors,
        minimum=1,
    )
    _integer(
        value.get("max_wallclock_minutes_per_mode"),
        path="budget.max_wallclock_minutes_per_mode",
        errors=errors,
        minimum=1,
    )
    failures = _integer(
        value.get("max_terminal_failures_per_mode"),
        path="budget.max_terminal_failures_per_mode",
        errors=errors,
        minimum=0,
    )
    if maximum is not None and concurrent is not None and concurrent > maximum:
        errors.append(
            "budget.max_concurrent_candidates_per_mode cannot exceed "
            "max_candidates_per_mode"
        )
    if maximum is not None and failures is not None and failures > maximum:
        errors.append(
            "budget.max_terminal_failures_per_mode cannot exceed "
            "max_candidates_per_mode"
        )
    return maximum


def _validate_fidelity(document: Mapping[str, Any], errors: list[str]) -> None:
    value = _require_keys(
        document.get("fidelity"),
        path="fidelity",
        required=(
            "unit",
            "rungs",
            "final_validation_budget",
            "evaluation_interval",
            "checkpoint_interval",
            "policy",
        ),
        errors=errors,
    )
    if value is None:
        return
    if value.get("unit") not in ("epochs", "steps"):
        errors.append("fidelity.unit must be 'epochs' or 'steps'")
    _nonempty_string(value.get("policy"), path="fidelity.policy", errors=errors)
    rungs = value.get("rungs")
    rung_values: list[int] = []
    if not isinstance(rungs, list) or not rungs:
        errors.append("fidelity.rungs must be a non-empty integer list")
    else:
        for index, rung in enumerate(rungs):
            parsed = _integer(
                rung,
                path=f"fidelity.rungs[{index}]",
                errors=errors,
                minimum=1,
            )
            if parsed is not None:
                rung_values.append(parsed)
        if rung_values != sorted(set(rung_values)):
            errors.append("fidelity.rungs must be strictly increasing")
    final_budget = _integer(
        value.get("final_validation_budget"),
        path="fidelity.final_validation_budget",
        errors=errors,
        minimum=1,
    )
    evaluation = _integer(
        value.get("evaluation_interval"),
        path="fidelity.evaluation_interval",
        errors=errors,
        minimum=1,
    )
    checkpoint = _integer(
        value.get("checkpoint_interval"),
        path="fidelity.checkpoint_interval",
        errors=errors,
        minimum=1,
    )
    if final_budget is not None:
        if rung_values and final_budget < rung_values[-1]:
            errors.append(
                "fidelity.final_validation_budget cannot be below the last rung"
            )
        if evaluation is not None and evaluation > final_budget:
            errors.append(
                "fidelity.evaluation_interval cannot exceed final validation budget"
            )
        if checkpoint is not None and checkpoint > final_budget:
            errors.append(
                "fidelity.checkpoint_interval cannot exceed final validation budget"
            )


def _validate_resources(document: Mapping[str, Any], errors: list[str]) -> None:
    value = _require_keys(
        document.get("resources"),
        path="resources",
        required=(
            "platform",
            "nodes",
            "gpus_per_node",
            "tasks_per_node",
            "distributed_workers_per_node",
            "gpu_type",
            "cpus_per_task",
            "memory_gib_per_node",
            "time_limit_minutes",
            "partition",
            "exclusive_node",
        ),
        errors=errors,
    )
    if value is None:
        return
    if value.get("platform") != "slurm":
        errors.append("resources.platform must be 'slurm'")
    if (
        not isinstance(value.get("nodes"), int)
        or isinstance(value.get("nodes"), bool)
        or value.get("nodes") != 1
    ):
        errors.append("resources.nodes must equal one")
    if (
        not isinstance(value.get("gpus_per_node"), int)
        or isinstance(value.get("gpus_per_node"), bool)
        or value.get("gpus_per_node") != 8
    ):
        errors.append("resources.gpus_per_node must equal eight")
    if (
        not isinstance(value.get("tasks_per_node"), int)
        or isinstance(value.get("tasks_per_node"), bool)
        or value.get("tasks_per_node") != 1
    ):
        errors.append("resources.tasks_per_node must equal one")
    if (
        not isinstance(value.get("distributed_workers_per_node"), int)
        or isinstance(value.get("distributed_workers_per_node"), bool)
        or value.get("distributed_workers_per_node") != 8
    ):
        errors.append("resources.distributed_workers_per_node must equal eight")
    _nonempty_string(value.get("gpu_type"), path="resources.gpu_type", errors=errors)
    for field in (
        "cpus_per_task",
        "memory_gib_per_node",
        "time_limit_minutes",
    ):
        _integer(value.get(field), path=f"resources.{field}", errors=errors, minimum=1)
    _nonempty_string(value.get("partition"), path="resources.partition", errors=errors)
    _boolean(
        value.get("exclusive_node"),
        path="resources.exclusive_node",
        errors=errors,
        expected=True,
    )


_REQUIRED_STAGE_CRITERIA = {
    "single_candidate_gate": {
        "entry": {
            "local_model_preflight_passed",
            "dataset_preflight_passed",
            "ptm_preflight_passed",
            "wheel_contents_verified",
        },
        "exit": {
            "one_candidate_per_mode_succeeded",
            "artifact_contract_passed",
        },
    },
    "pilot_batch": {
        "entry": {"single_candidate_gate_passed"},
        "exit": {"pilot_artifacts_passed", "metric_sanity_passed"},
    },
    "full_search": {
        "entry": {"pilot_batch_passed"},
        "exit": {"search_archives_sealed", "selection_winners_frozen"},
    },
    "matched_validation": {
        "entry": {"selection_winners_frozen"},
        "exit": {"matched_validation_artifacts_sealed"},
    },
}


def _validate_stages(
    document: Mapping[str, Any],
    errors: list[str],
    max_candidates_per_mode: int | None,
) -> tuple[int | None, int | None, int | None]:
    stages = document.get("stages")
    if not isinstance(stages, list) or len(stages) != len(CAMPAIGN_STAGES):
        errors.append(
            f"stages must contain exactly {list(CAMPAIGN_STAGES)}"
        )
        return None, None, None
    expected_jobs: dict[str, int] = {}
    observed_names: list[str] = []
    for index, stage in enumerate(stages):
        path = f"stages[{index}]"
        value = _require_keys(
            stage,
            path=path,
            required=(
                "name",
                "order",
                "expected_jobs",
                "entry_criteria",
                "exit_criteria",
                "on_failure",
            ),
            errors=errors,
        )
        if value is None:
            continue
        name = value.get("name")
        if isinstance(name, str):
            observed_names.append(name)
        order = _integer(
            value.get("order"),
            path=f"{path}.order",
            errors=errors,
            minimum=1,
        )
        if order is not None and order != index + 1:
            errors.append(f"{path}.order must equal {index + 1}")
        jobs = _integer(
            value.get("expected_jobs"),
            path=f"{path}.expected_jobs",
            errors=errors,
            minimum=1,
        )
        if isinstance(name, str) and jobs is not None:
            expected_jobs[name] = jobs
        entry = _string_list(
            value.get("entry_criteria"),
            path=f"{path}.entry_criteria",
            errors=errors,
        )
        exit_criteria = _string_list(
            value.get("exit_criteria"),
            path=f"{path}.exit_criteria",
            errors=errors,
        )
        if name in _REQUIRED_STAGE_CRITERIA:
            requirements = _REQUIRED_STAGE_CRITERIA[name]
            if entry is not None:
                missing = sorted(requirements["entry"] - set(entry))
                if missing:
                    errors.append(
                        f"{path}.entry_criteria is missing required gates {missing}"
                    )
            if exit_criteria is not None:
                missing = sorted(requirements["exit"] - set(exit_criteria))
                if missing:
                    errors.append(
                        f"{path}.exit_criteria is missing required gates {missing}"
                    )
        if value.get("on_failure") != "halt_before_next_stage":
            errors.append(f"{path}.on_failure must be 'halt_before_next_stage'")
    if tuple(observed_names) != CAMPAIGN_STAGES:
        errors.append(f"stages must be ordered as {list(CAMPAIGN_STAGES)}")

    single_jobs = expected_jobs.get("single_candidate_gate")
    pilot_jobs = expected_jobs.get("pilot_batch")
    full_jobs = expected_jobs.get("full_search")
    matched_jobs = expected_jobs.get("matched_validation")
    if single_jobs is not None and single_jobs != len(CAMPAIGN_MODES):
        errors.append("single_candidate_gate must schedule one job per mode")
    for stage_name, jobs in (
        ("pilot_batch", pilot_jobs),
        ("full_search", full_jobs),
    ):
        if jobs is not None and jobs % len(CAMPAIGN_MODES) != 0:
            errors.append(
                f"{stage_name} expected_jobs must be balanced across three modes"
            )
    candidate_jobs = (
        single_jobs + pilot_jobs + full_jobs
        if None not in (single_jobs, pilot_jobs, full_jobs)
        else None
    )
    if (
        candidate_jobs is not None
        and max_candidates_per_mode is not None
        and candidate_jobs != len(CAMPAIGN_MODES) * max_candidates_per_mode
    ):
        errors.append(
            "candidate-stage expected_jobs must equal three times "
            "budget.max_candidates_per_mode"
        )
    total_jobs = (
        candidate_jobs + matched_jobs
        if candidate_jobs is not None and matched_jobs is not None
        else None
    )
    return candidate_jobs, matched_jobs, total_jobs


def _validate_workload(
    document: Mapping[str, Any],
    errors: list[str],
    *,
    candidate_jobs: int | None,
    matched_jobs: int | None,
    total_jobs: int | None,
) -> None:
    value = _require_keys(
        document.get("workload"),
        path="workload",
        required=(
            "expected_candidate_jobs",
            "expected_matched_validation_jobs",
            "expected_total_jobs",
            "estimated_storage_bytes",
        ),
        errors=errors,
    )
    if value is None:
        return
    observed_candidate = _integer(
        value.get("expected_candidate_jobs"),
        path="workload.expected_candidate_jobs",
        errors=errors,
        minimum=1,
    )
    observed_matched = _integer(
        value.get("expected_matched_validation_jobs"),
        path="workload.expected_matched_validation_jobs",
        errors=errors,
        minimum=1,
    )
    observed_total = _integer(
        value.get("expected_total_jobs"),
        path="workload.expected_total_jobs",
        errors=errors,
        minimum=1,
    )
    _integer(
        value.get("estimated_storage_bytes"),
        path="workload.estimated_storage_bytes",
        errors=errors,
        minimum=1,
    )
    for observed, expected, path in (
        (
            observed_candidate,
            candidate_jobs,
            "workload.expected_candidate_jobs",
        ),
        (
            observed_matched,
            matched_jobs,
            "workload.expected_matched_validation_jobs",
        ),
        (observed_total, total_jobs, "workload.expected_total_jobs"),
    ):
        if observed is not None and expected is not None and observed != expected:
            errors.append(f"{path} does not match the staged job count")


def _validate_retry_and_failure_policy(
    document: Mapping[str, Any],
    errors: list[str],
) -> None:
    retry = _require_keys(
        document.get("retry_policy"),
        path="retry_policy",
        required=(
            "max_retries_per_trial",
            "retryable_failure_codes",
            "preserve_failed_trials",
            "replacement_policy",
        ),
        errors=errors,
    )
    if retry is not None:
        retries = _integer(
            retry.get("max_retries_per_trial"),
            path="retry_policy.max_retries_per_trial",
            errors=errors,
            minimum=0,
        )
        if retries is not None and retries > MAX_RETRIES_PER_TRIAL:
            errors.append(
                "retry_policy.max_retries_per_trial exceeds the product bound "
                f"of {MAX_RETRIES_PER_TRIAL}"
            )
        _string_list(
            retry.get("retryable_failure_codes"),
            path="retry_policy.retryable_failure_codes",
            errors=errors,
            allow_empty=True,
        )
        _boolean(
            retry.get("preserve_failed_trials"),
            path="retry_policy.preserve_failed_trials",
            errors=errors,
            expected=True,
        )
        if retry.get("replacement_policy") != "never_silent":
            errors.append(
                "retry_policy.replacement_policy must be 'never_silent'"
            )

    failed = _require_keys(
        document.get("failed_trial_policy"),
        path="failed_trial_policy",
        required=(
            "preserve_records",
            "preserve_terminal_recommendation",
            "count_toward_candidate_budget",
            "silent_replacement",
            "terminal_status",
        ),
        errors=errors,
    )
    if failed is not None:
        for field in (
            "preserve_records",
            "preserve_terminal_recommendation",
            "count_toward_candidate_budget",
        ):
            _boolean(
                failed.get(field),
                path=f"failed_trial_policy.{field}",
                errors=errors,
                expected=True,
            )
        _boolean(
            failed.get("silent_replacement"),
            path="failed_trial_policy.silent_replacement",
            errors=errors,
            expected=False,
        )
        if failed.get("terminal_status") != "failed":
            errors.append("failed_trial_policy.terminal_status must be 'failed'")


def _validate_cancellation(document: Mapping[str, Any], errors: list[str]) -> None:
    value = _require_keys(
        document.get("cancellation"),
        path="cancellation",
        required=("criteria", "action", "preserve_records"),
        errors=errors,
    )
    if value is None:
        return
    criteria = _string_list(
        value.get("criteria"),
        path="cancellation.criteria",
        errors=errors,
    )
    required = {
        "artifact_integrity_failure",
        "preflight_gate_failure",
        "metric_sanity_failure",
        "failure_budget_exceeded",
        "storage_budget_exceeded",
        "wallclock_budget_exceeded",
    }
    if criteria is not None:
        missing = sorted(required - set(criteria))
        if missing:
            errors.append(f"cancellation.criteria is missing {missing}")
    if value.get("action") != "cancel_pending_and_halt":
        errors.append("cancellation.action must be 'cancel_pending_and_halt'")
    _boolean(
        value.get("preserve_records"),
        path="cancellation.preserve_records",
        errors=errors,
        expected=True,
    )


def _validate_false_flags(
    document: Mapping[str, Any],
    *,
    field: str,
    expected_names: Sequence[str],
    errors: list[str],
) -> None:
    value = _require_keys(
        document.get(field),
        path=field,
        required=expected_names,
        errors=errors,
    )
    if value is None:
        return
    for name in expected_names:
        _boolean(
            value.get(name),
            path=f"{field}.{name}",
            errors=errors,
            expected=False,
        )


def _validate_metric(
    value: Any,
    *,
    path: str,
    errors: list[str],
) -> tuple[str, str, str] | None:
    metric = _require_keys(
        value,
        path=path,
        required=("name", "role", "direction"),
        errors=errors,
    )
    if metric is None:
        return None
    name = _nonempty_string(metric.get("name"), path=f"{path}.name", errors=errors)
    role = metric.get("role")
    direction = metric.get("direction")
    if role not in ("accuracy", "latency"):
        errors.append(f"{path}.role must be 'accuracy' or 'latency'")
    expected_direction = {"accuracy": "maximize", "latency": "minimize"}.get(role)
    if direction != expected_direction:
        errors.append(
            f"{path}.direction must be {expected_direction!r} for role {role!r}"
        )
    if name is None or role not in ("accuracy", "latency"):
        return None
    return name, role, direction


def _validate_objective(
    value: Any,
    *,
    mode: str,
    path: str,
    errors: list[str],
) -> tuple[str | None, str | None]:
    objective = _require_keys(
        value,
        path=path,
        required=(
            "mode",
            "primary_role",
            "metrics",
            "quality_constraint",
            "acquisition",
            "selection_policy",
        ),
        errors=errors,
    )
    if objective is None:
        return None, None
    if objective.get("mode") != mode:
        errors.append(f"{path}.mode must equal {mode!r}")
    _nonempty_string(
        objective.get("acquisition"),
        path=f"{path}.acquisition",
        errors=errors,
    )
    expected_acquisition = _STRICT_MODE_ACQUISITIONS[mode]
    if objective.get("acquisition") != expected_acquisition:
        errors.append(
            f"{path}.acquisition must be objective-aware and equal "
            f"{expected_acquisition!r} for strict {mode!r} campaigns"
        )
    _nonempty_string(
        objective.get("selection_policy"),
        path=f"{path}.selection_policy",
        errors=errors,
    )

    metrics = objective.get("metrics")
    parsed: list[tuple[str, str, str]] = []
    if not isinstance(metrics, list) or not metrics:
        errors.append(f"{path}.metrics must be a non-empty list")
    else:
        for index, metric in enumerate(metrics):
            result = _validate_metric(
                metric,
                path=f"{path}.metrics[{index}]",
                errors=errors,
            )
            if result is not None:
                parsed.append(result)
    roles = [item[1] for item in parsed]
    if len(roles) != len(set(roles)):
        errors.append(f"{path}.metrics must contain each objective role once")

    expected_roles = {
        "accuracy": {"accuracy"},
        "latency": {"accuracy", "latency"},
        "multi_objective": {"accuracy", "latency"},
    }.get(mode, set())
    if set(roles) != expected_roles:
        errors.append(
            f"{path}.metrics roles must equal {sorted(expected_roles)} for {mode}"
        )
    expected_primary = {
        "accuracy": "accuracy",
        "latency": "latency",
        "multi_objective": "pareto",
    }.get(mode)
    if objective.get("primary_role") != expected_primary:
        errors.append(f"{path}.primary_role must be {expected_primary!r}")

    constraint = objective.get("quality_constraint")
    if mode == "accuracy":
        if constraint is not None:
            errors.append(f"{path}.quality_constraint must be null in accuracy mode")
    elif mode == "latency":
        quality = _require_keys(
            constraint,
            path=f"{path}.quality_constraint",
            required=(
                "type",
                "retained_fraction",
                "reference",
                "reference_updates",
                "terminal_reference",
            ),
            errors=errors,
        )
        if quality is not None:
            if quality.get("type") != "relative_retention":
                errors.append(
                    f"{path}.quality_constraint.type must be 'relative_retention'"
                )
            retained = _finite_number(
                quality.get("retained_fraction"),
                path=f"{path}.quality_constraint.retained_fraction",
                errors=errors,
                minimum=0.0,
                maximum=1.0,
            )
            if retained is not None and retained <= 0:
                errors.append(
                    f"{path}.quality_constraint.retained_fraction must be > 0"
                )
            if quality.get("reference") != "best_observed_within_job":
                errors.append(
                    f"{path}.quality_constraint.reference must be "
                    "'best_observed_within_job'"
                )
            if quality.get("reference_updates") != "monotonic":
                errors.append(
                    f"{path}.quality_constraint.reference_updates must be "
                    "'monotonic'"
                )
            if (
                quality.get("terminal_reference")
                != "terminal_archive_accuracy_winner"
            ):
                errors.append(
                    f"{path}.quality_constraint.terminal_reference must bind "
                    "to the terminal archive accuracy winner"
                )
    elif mode == "multi_objective" and constraint is not None:
        quality = _require_keys(
            constraint,
            path=f"{path}.quality_constraint",
            required=("type", "value", "source"),
            errors=errors,
        )
        if quality is not None:
            if quality.get("type") != "absolute_minimum":
                errors.append(
                    f"{path}.quality_constraint.type must be 'absolute_minimum'"
                )
            _finite_number(
                quality.get("value"),
                path=f"{path}.quality_constraint.value",
                errors=errors,
            )
            if quality.get("source") != "multi_objective_explicit":
                errors.append(
                    f"{path}.quality_constraint.source must be "
                    "'multi_objective_explicit'"
                )

    accuracy_name = next(
        (name for name, role, _ in parsed if role == "accuracy"),
        None,
    )
    latency_name = next(
        (name for name, role, _ in parsed if role == "latency"),
        None,
    )
    return accuracy_name, latency_name


def _fairness_audit(
    document: Mapping[str, Any],
    *,
    ptm_ids: tuple[str, ...],
    default_ptm_id: str | None,
    search_space_sha256: str | None,
) -> CampaignFairnessAudit:
    passed: list[str] = []
    violations: list[str] = []
    modes = document.get("modes")
    if not isinstance(modes, list):
        return CampaignFairnessAudit((), ("modes is not a list",))

    mode_names = tuple(
        item.get("mode") for item in modes if isinstance(item, Mapping)
    )
    if mode_names == CAMPAIGN_MODES:
        passed.append("exact_three_mode_jobs")
    else:
        violations.append(
            f"mode jobs must be exactly {list(CAMPAIGN_MODES)}"
        )

    jobs = [
        item.get("job_id") for item in modes if isinstance(item, Mapping)
    ]
    if (
        len(jobs) == len(CAMPAIGN_MODES)
        and all(isinstance(job, str) for job in jobs)
        and len(set(jobs)) == len(jobs)
    ):
        passed.append("unique_mode_job_ids")
    else:
        violations.append("mode job IDs must be unique")

    namespaces = [
        item.get("observation_namespace")
        for item in modes
        if isinstance(item, Mapping)
    ]
    if (
        len(namespaces) == len(CAMPAIGN_MODES)
        and all(isinstance(namespace, str) for namespace in namespaces)
        and len(set(namespaces)) == len(namespaces)
    ):
        passed.append("unique_observation_namespaces")
    else:
        violations.append("observation namespaces must be unique")

    if all(
        isinstance(item, Mapping)
        and item.get("observation_sharing") is False
        and item.get("initial_observation_ids") == []
        for item in modes
    ):
        passed.append("no_cross_mode_observation_sharing")
    else:
        violations.append(
            "every mode must disable sharing and start without observations"
        )

    seeds = [
        item.get("seed") for item in modes if isinstance(item, Mapping)
    ]
    if (
        len(seeds) == len(CAMPAIGN_MODES)
        and all(
            isinstance(seed, int) and not isinstance(seed, bool)
            for seed in seeds
        )
        and len(set(seeds)) == 1
    ):
        passed.append("same_preregistered_seed")
    else:
        violations.append("all modes must use the same preregistered seed")

    expected_ptms = list(ptm_ids)
    mode_records = {
        item.get("mode"): item
        for item in modes
        if isinstance(item, Mapping) and item.get("mode") in CAMPAIGN_MODES
    }
    latency_and_moo = (
        mode_records.get("latency"),
        mode_records.get("multi_objective"),
    )
    if all(
        item is not None
        and item.get("ptm_policy") == "all_qualified"
        and item.get("allowed_ptm_ids") == expected_ptms
        for item in latency_and_moo
    ):
        passed.append("latency_and_moo_full_ptm_inventory")
    else:
        violations.append(
            "latency and multi-objective modes must bind all qualified PTMs"
        )

    accuracy = mode_records.get("accuracy")
    accuracy_policy_ok = False
    if accuracy is not None:
        policy = accuracy.get("ptm_policy")
        allowed = accuracy.get("allowed_ptm_ids")
        if policy == "registered_default":
            accuracy_policy_ok = (
                default_ptm_id is not None and allowed == [default_ptm_id]
            )
        elif policy == "user_provided":
            accuracy_policy_ok = (
                isinstance(allowed, list)
                and len(allowed) == 1
                and allowed[0] in ptm_ids
            )
        elif policy == "all_qualified_explicit":
            accuracy_policy_ok = allowed == expected_ptms
    if accuracy_policy_ok:
        if accuracy.get("ptm_policy") == "all_qualified_explicit":
            passed.append("same_preflight_ptm_inventory")
        else:
            passed.append("documented_accuracy_ptm_policy_exception")
    else:
        violations.append(
            "accuracy PTM inventory does not match its explicit PTM policy"
        )

    if search_space_sha256 is not None and all(
        isinstance(item, Mapping)
        and item.get("search_space_sha256") == search_space_sha256
        for item in modes
    ):
        passed.append("same_search_space")
    else:
        violations.append("all modes must bind the same search-space hash")

    passed.extend(
        (
            "same_dataset_by_campaign_scope",
            "same_budget_by_campaign_scope",
            "same_fidelity_by_campaign_scope",
            "same_resources_by_campaign_scope",
            "same_runtime_by_campaign_scope",
            "same_algorithm_by_campaign_scope",
        )
    )
    return CampaignFairnessAudit(tuple(passed), tuple(violations))


def _validate_modes(
    document: Mapping[str, Any],
    errors: list[str],
    *,
    ptm_ids: tuple[str, ...],
    default_ptm_id: str | None,
    search_space_sha256: str | None,
) -> CampaignFairnessAudit:
    modes = document.get("modes")
    if not isinstance(modes, list):
        errors.append("modes must be a list")
        return CampaignFairnessAudit((), ("modes is not a list",))
    accuracy_metrics: dict[str, str | None] = {}
    latency_metrics: dict[str, str | None] = {}
    for index, item in enumerate(modes):
        path = f"modes[{index}]"
        value = _require_keys(
            item,
            path=path,
            required=(
                "mode",
                "job_id",
                "seed",
                "observation_namespace",
                "observation_sharing",
                "initial_observation_ids",
                "ptm_policy",
                "allowed_ptm_ids",
                "search_space_sha256",
                "objective",
                "objective_sha256",
            ),
            errors=errors,
        )
        if value is None:
            continue
        mode = value.get("mode")
        if mode not in CAMPAIGN_MODES:
            errors.append(f"{path}.mode must be one of {list(CAMPAIGN_MODES)}")
            continue
        _nonempty_string(
            value.get("job_id"),
            path=f"{path}.job_id",
            errors=errors,
            identifier=True,
        )
        _integer(value.get("seed"), path=f"{path}.seed", errors=errors, minimum=0)
        _nonempty_string(
            value.get("observation_namespace"),
            path=f"{path}.observation_namespace",
            errors=errors,
            identifier=True,
        )
        _boolean(
            value.get("observation_sharing"),
            path=f"{path}.observation_sharing",
            errors=errors,
            expected=False,
        )
        initial = value.get("initial_observation_ids")
        if initial != []:
            errors.append(
                f"{path}.initial_observation_ids must be empty for an "
                "independent job"
            )
        allowed = _string_list(
            value.get("allowed_ptm_ids"),
            path=f"{path}.allowed_ptm_ids",
            errors=errors,
        )
        policy = value.get("ptm_policy")
        if mode == "accuracy":
            if policy not in (
                "registered_default",
                "user_provided",
                "all_qualified_explicit",
            ):
                errors.append(
                    f"{path}.ptm_policy must be 'registered_default', "
                    "'user_provided', or 'all_qualified_explicit'"
                )
            elif allowed is not None:
                if policy == "registered_default" and (
                    default_ptm_id is None or allowed != (default_ptm_id,)
                ):
                    errors.append(
                        f"{path}.allowed_ptm_ids must contain only the "
                        "registered default PTM"
                    )
                elif policy == "user_provided" and (
                    len(allowed) != 1 or allowed[0] not in ptm_ids
                ):
                    errors.append(
                        f"{path}.allowed_ptm_ids must contain exactly one "
                        "qualified user-provided PTM"
                    )
                elif (
                    policy == "all_qualified_explicit"
                    and tuple(sorted(allowed)) != ptm_ids
                ):
                    errors.append(
                        f"{path}.allowed_ptm_ids must contain every qualified "
                        "PTM when multi-PTM accuracy search is explicit"
                    )
        else:
            if policy != "all_qualified":
                errors.append(f"{path}.ptm_policy must be 'all_qualified'")
            if allowed is not None and tuple(sorted(allowed)) != ptm_ids:
                errors.append(
                    f"{path}.allowed_ptm_ids must equal the full PTM inventory"
                )
        observed_space_hash = _sha256(
            value.get("search_space_sha256"),
            path=f"{path}.search_space_sha256",
            errors=errors,
        )
        if (
            observed_space_hash is not None
            and search_space_sha256 is not None
            and observed_space_hash != search_space_sha256
        ):
            errors.append(f"{path}.search_space_sha256 does not match campaign")
        objective = value.get("objective")
        accuracy_name, latency_name = _validate_objective(
            objective,
            mode=mode,
            path=f"{path}.objective",
            errors=errors,
        )
        accuracy_metrics[mode] = accuracy_name
        latency_metrics[mode] = latency_name
        try:
            objective_hash = canonical_sha256(objective)
        except ValueError as exc:
            errors.append(f"{path}.objective is not canonical JSON: {exc}")
        else:
            observed_objective_hash = _sha256(
                value.get("objective_sha256"),
                path=f"{path}.objective_sha256",
                errors=errors,
            )
            if (
                observed_objective_hash is not None
                and observed_objective_hash != objective_hash
            ):
                errors.append(
                    f"{path}.objective_sha256 does not match objective config"
                )

    accuracy_names = {
        name for name in accuracy_metrics.values() if name is not None
    }
    if len(accuracy_names) > 1:
        errors.append("all modes must use the same primary accuracy metric")
    latency_names = {
        name for name in latency_metrics.values() if name is not None
    }
    if len(latency_names) > 1:
        errors.append("latency and multi-objective modes must use the same latency metric")

    audit = _fairness_audit(
        document,
        ptm_ids=ptm_ids,
        default_ptm_id=default_ptm_id,
        search_space_sha256=search_space_sha256,
    )
    errors.extend(f"fairness: {violation}" for violation in audit.violations)
    return audit


def validate_campaign_manifest(document: Any) -> CampaignFairnessAudit:
    """Validate one unsealed canonical campaign document."""
    if not isinstance(document, Mapping):
        raise CampaignManifestValidationError(("manifest root must be an object",))
    document = _normalized_document(document)
    errors: list[str] = []
    required = (
        "schema_version",
        "campaign_id",
        "model",
        "task",
        *_SECTION_KEYS,
    )
    _require_keys(
        document,
        path="manifest",
        required=required,
        errors=errors,
    )
    schema_version = document.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != CAMPAIGN_MANIFEST_SCHEMA_VERSION
    ):
        errors.append(
            "schema_version must equal "
            f"{CAMPAIGN_MANIFEST_SCHEMA_VERSION}"
        )
    for field in ("campaign_id", "model", "task"):
        _nonempty_string(
            document.get(field),
            path=field,
            errors=errors,
            identifier=True,
        )

    _validate_source(document, errors)
    _validate_package(document, errors)
    _validate_container(document, errors)
    _validate_runtime(document, errors)
    ptm_ids = _validate_ptms(document, errors)
    default_ptm_id = _validate_ptm_search(
        document,
        errors,
        ptm_ids=ptm_ids,
    )
    _validate_dataset(document, errors)
    _validate_algorithm(document, errors)
    space_hash = _validate_search_space(document, errors, ptm_ids=ptm_ids)
    max_candidates = _validate_budget(document, errors)
    _validate_fidelity(document, errors)
    _validate_resources(document, errors)
    candidate_jobs, matched_jobs, total_jobs = _validate_stages(
        document,
        errors,
        max_candidates,
    )
    _validate_workload(
        document,
        errors,
        candidate_jobs=candidate_jobs,
        matched_jobs=matched_jobs,
        total_jobs=total_jobs,
    )
    _validate_retry_and_failure_policy(document, errors)
    _validate_cancellation(document, errors)
    _validate_false_flags(
        document,
        field="agent_intervention_flags",
        expected_names=AGENT_INTERVENTION_FLAGS,
        errors=errors,
    )
    _validate_false_flags(
        document,
        field="selection_isolation_flags",
        expected_names=SELECTION_ISOLATION_FLAGS,
        errors=errors,
    )
    audit = _validate_modes(
        document,
        errors,
        ptm_ids=ptm_ids,
        default_ptm_id=default_ptm_id,
        search_space_sha256=space_hash,
    )
    if errors:
        raise CampaignManifestValidationError(errors)
    return audit


class CampaignManifest:
    """Immutable canonical campaign intent with deterministic derived manifests."""

    __slots__ = ("_canonical_json", "_manifest_sha256", "_fairness_audit")

    def __init__(
        self,
        document: Mapping[str, Any],
        *,
        require_seal: bool = False,
    ):
        if not isinstance(document, Mapping):
            raise CampaignManifestValidationError(
                ("manifest root must be an object",)
            )
        raw = dict(document)
        supplied_hash = raw.pop("manifest_sha256", None)
        normalized = _normalized_document(raw)
        audit = validate_campaign_manifest(normalized)
        digest = canonical_sha256(normalized)
        if require_seal and supplied_hash is None:
            raise CampaignManifestValidationError(
                ("manifest_sha256 is required when loading a sealed manifest",)
            )
        if supplied_hash is not None:
            if (
                not isinstance(supplied_hash, str)
                or _SHA256_RE.fullmatch(supplied_hash) is None
            ):
                raise CampaignManifestValidationError(
                    ("manifest_sha256 must be a lowercase 64-character SHA-256",)
                )
            if supplied_hash != digest:
                raise CampaignManifestValidationError(
                    (
                        "manifest_sha256 does not match the canonical campaign "
                        "configuration",
                    )
                )
        self._canonical_json = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        self._manifest_sha256 = digest
        self._fairness_audit = audit

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    @property
    def fairness_audit(self) -> CampaignFairnessAudit:
        return self._fairness_audit

    def stable_dict(self) -> dict[str, Any]:
        """Return a defensive copy of the hash-bearing campaign payload."""
        return json.loads(self._canonical_json)

    def to_dict(self) -> dict[str, Any]:
        """Return the sealed manifest."""
        value = self.stable_dict()
        value["manifest_sha256"] = self.manifest_sha256
        return value

    def mode_manifest(self, mode: str) -> dict[str, Any]:
        """Derive one independently resumable mode manifest."""
        if mode not in CAMPAIGN_MODES:
            raise KeyError(f"Unknown campaign mode: {mode!r}")
        campaign = self.stable_dict()
        mode_job = next(
            item for item in campaign.pop("modes") if item["mode"] == mode
        )
        payload = {
            **campaign,
            "parent_campaign_manifest_sha256": self.manifest_sha256,
            "mode_job": mode_job,
        }
        digest = canonical_sha256(payload)
        payload["mode_manifest_sha256"] = digest
        return payload

    def assert_resume_compatible(
        self,
        persisted_manifest: Mapping[str, Any] | "CampaignManifest",
    ) -> None:
        """Reject campaign resume before execution when any input changed."""
        try:
            persisted = (
                persisted_manifest
                if isinstance(persisted_manifest, CampaignManifest)
                else CampaignManifest(persisted_manifest, require_seal=True)
            )
        except CampaignManifestValidationError as exc:
            observed = (
                persisted_manifest.get("manifest_sha256")
                if isinstance(persisted_manifest, Mapping)
                else None
            )
            raise CampaignResumeMismatchError(
                expected_sha256=self.manifest_sha256,
                observed_sha256=observed if isinstance(observed, str) else None,
                scope="campaign",
                reason="persisted manifest is invalid or unsealed",
            ) from exc
        if persisted.manifest_sha256 != self.manifest_sha256:
            raise CampaignResumeMismatchError(
                expected_sha256=self.manifest_sha256,
                observed_sha256=persisted.manifest_sha256,
                scope="campaign",
                reason="configuration identity changed",
            )

    def assert_mode_resume_compatible(
        self,
        mode: str,
        persisted_mode_manifest: Mapping[str, Any],
    ) -> None:
        """Reject one mode's resume if its derived binding changed."""
        expected = self.mode_manifest(mode)
        if not isinstance(persisted_mode_manifest, Mapping):
            raise CampaignResumeMismatchError(
                expected_sha256=expected["mode_manifest_sha256"],
                observed_sha256=None,
                scope=f"mode:{mode}",
                reason="persisted mode manifest is not an object",
            )
        observed_hash = persisted_mode_manifest.get("mode_manifest_sha256")
        candidate = dict(persisted_mode_manifest)
        candidate.pop("mode_manifest_sha256", None)
        try:
            computed = canonical_sha256(_canonical_roundtrip(candidate))
        except (CampaignManifestValidationError, ValueError) as exc:
            raise CampaignResumeMismatchError(
                expected_sha256=expected["mode_manifest_sha256"],
                observed_sha256=(
                    observed_hash if isinstance(observed_hash, str) else None
                ),
                scope=f"mode:{mode}",
                reason="persisted mode manifest is not canonical JSON",
            ) from exc
        if (
            observed_hash != computed
            or computed != expected["mode_manifest_sha256"]
        ):
            raise CampaignResumeMismatchError(
                expected_sha256=expected["mode_manifest_sha256"],
                observed_sha256=(
                    observed_hash if isinstance(observed_hash, str) else computed
                ),
                scope=f"mode:{mode}",
                reason="mode or shared configuration identity changed",
            )

    def __repr__(self) -> str:
        return (
            "CampaignManifest("
            f"campaign_id={self.stable_dict()['campaign_id']!r}, "
            f"sha256={self.manifest_sha256!r})"
        )


def create_campaign_manifest(document: Mapping[str, Any]) -> CampaignManifest:
    """Validate, normalize, and seal a new campaign manifest."""
    if not isinstance(document, Mapping):
        raise CampaignManifestValidationError(("manifest root must be an object",))
    if "manifest_sha256" in document:
        raise CampaignManifestValidationError(
            ("new campaign input must not predeclare manifest_sha256",)
        )
    return CampaignManifest(document)


def load_campaign_manifest(document: Mapping[str, Any]) -> CampaignManifest:
    """Load a sealed campaign manifest and verify its canonical hash."""
    return CampaignManifest(document, require_seal=True)


__all__ = [
    "AGENT_INTERVENTION_FLAGS",
    "CAMPAIGN_MANIFEST_SCHEMA_VERSION",
    "CAMPAIGN_MODES",
    "CAMPAIGN_STAGES",
    "CampaignFairnessAudit",
    "CampaignManifest",
    "CampaignManifestValidationError",
    "CampaignResumeMismatchError",
    "MAX_RETRIES_PER_TRIAL",
    "SELECTION_ISOLATION_FLAGS",
    "create_campaign_manifest",
    "load_campaign_manifest",
    "validate_campaign_manifest",
]
