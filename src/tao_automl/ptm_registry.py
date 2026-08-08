# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned pretrained-model registry primitives.

The registry is deliberately repository-owned.  It records immutable model
artifact identities and compatibility metadata. Authenticated resolution,
download, integrity enforcement, and load qualification are implemented by
the separate ``tao_automl.ptm_preflight`` boundary.

This module does not download artifacts.  It provides:

* strict validation for records advertised as ``supported``;
* structured compatibility and exclusion results;
* deterministic specification merge precedence;
* local SHA-256 calculation and verification hooks.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version


PTM_REGISTRY_SCHEMA_VERSION = 1
PTM_REGISTRY_RESOURCE = "data/ptm_registry.v1.json"
PTM_REGISTRY_SCHEMA_RESOURCE = "data/ptm_registry.schema.v1.json"
PTM_STATUS_VALUES = frozenset(
    {"supported", "unverified", "unsupported", "deprecated"}
)
PTM_QUALIFICATION_STATUS_VALUES = frozenset({
    "supported",
    "unverified",
    "unsupported",
})
SPEC_PRECEDENCE = (
    "model_defaults",
    "ptm_overrides",
    "automl_profile_overrides",
    "user_overrides",
    "candidate_overrides",
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_CHECKPOINT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_PATH_TOKEN_RE = re.compile(r"([^.\[\]]+)|\[(\d+)\]")
_DOTTED_PATH_RE = re.compile(
    r"^[^.\[\]]+(?:\[\d+\])*(?:\.[^.\[\]]+(?:\[\d+\])*)*$"
)
_VERSION_PREFIX_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)")
_SPECIFIER_PREFIXES = ("~=", "==", "!=", "<=", ">=", "<", ">")

_ROOT_FIELDS = frozenset({"schema_version", "registry_version", "models"})
_MODEL_FIELDS = frozenset({"default_ptm", "checkpoints"})
_CHECKPOINT_FIELDS = frozenset({
    "artifact_adapters",
    "id",
    "status",
    "status_reason",
    "source",
    "sha256",
    "expected_size_bytes",
    "compatible_tao_versions",
    "model_family",
    "architecture",
    "backbone",
    "checkpoint_target",
    "input_contract",
    "default_spec_overrides",
    "checkpoint_spec_file",
    "task_compatibility",
    "license",
    "deprecation",
    "validation",
})
_SOURCE_FIELDS = frozenset({
    "provider",
    "registry",
    "resource",
    "version",
    "member",
    "official",
    "immutable_identity",
})
_INPUT_CONTRACT_FIELDS = frozenset({
    "channels",
    "height",
    "width",
    "preprocessing",
})
_CHECKPOINT_SPEC_FIELDS = frozenset({
    "source",
    "path",
    "member",
    "expected_size_bytes",
    "sha256",
    "immutable_identity",
    "provenance",
})
_PROVENANCE_FIELDS = frozenset({"source", "evidence"})
_LICENSE_FIELDS = frozenset({"name", "url", "access_requirements"})
_DEPRECATION_FIELDS = frozenset({
    "is_deprecated",
    "reason",
    "replacement_id",
})
_VALIDATION_FIELDS = frozenset({
    "status",
    "tao_version",
    "container_identity",
    "evidence",
})
_ARTIFACT_ADAPTER_FIELDS = frozenset({
    "id",
    "adapter_type",
    "compatible_tao_versions",
    "recipe",
    "output",
    "provenance",
})
_ARTIFACT_ADAPTER_RECIPE_FIELDS = frozenset({
    "retain_top_level_keys",
    "add_top_level_metadata",
    "tensor_container_key",
    "require_exact_tensor_key_set",
    "require_exact_tensor_values",
})
_ARTIFACT_ADAPTER_OUTPUT_FIELDS = frozenset({
    "member",
    "expected_size_bytes",
    "sha256",
})
_ARTIFACT_ADAPTER_TYPES = frozenset({
    "checkpoint_metadata_projection_v1",
})


class PTMRegistryValidationError(ValueError):
    """Raised when a registry document violates the versioned contract."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__(
            "Invalid PTM registry:\n" + "\n".join(f"- {error}" for error in self.errors)
        )


class PTMArtifactAdapterResolutionError(ValueError):
    """Raised when version-scoped adapter metadata is ambiguous."""


@dataclass(frozen=True)
class PTMExclusion:
    """Why one checkpoint was excluded from an execution inventory."""

    checkpoint_id: str
    status: str
    codes: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "status": self.status,
            "codes": list(self.codes),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class PTMCompatibilityResult:
    """Deterministic supported/excluded inventory for one model and task."""

    model: str
    tao_version: str
    task: str
    model_found: bool
    eligible_checkpoint_ids: tuple[str, ...]
    excluded: tuple[PTMExclusion, ...]
    default_checkpoint_id: str | None
    reasons: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether at least one supported compatible checkpoint is available."""
        return bool(self.eligible_checkpoint_ids)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "model": self.model,
            "tao_version": self.tao_version,
            "task": self.task,
            "model_found": self.model_found,
            "eligible_checkpoint_ids": list(self.eligible_checkpoint_ids),
            "excluded": [item.to_dict() for item in self.excluded],
            "default_checkpoint_id": self.default_checkpoint_id,
            "reasons": list(self.reasons),
            "ok": self.ok,
        }


@dataclass(frozen=True)
class PTMQualificationResult:
    """Explicit non-runtime inventory for checkpoint qualification."""

    model: str
    tao_version: str
    task: str
    validation_statuses: tuple[str, ...]
    model_found: bool
    candidate_checkpoint_ids: tuple[str, ...]
    excluded: tuple[PTMExclusion, ...]
    reasons: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether at least one record has enough metadata for qualification."""
        return bool(self.candidate_checkpoint_ids)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable, explicitly non-runtime representation."""
        return {
            "model": self.model,
            "tao_version": self.tao_version,
            "task": self.task,
            "validation_statuses": list(self.validation_statuses),
            "model_found": self.model_found,
            "candidate_checkpoint_ids": list(self.candidate_checkpoint_ids),
            "excluded": [item.to_dict() for item in self.excluded],
            "reasons": list(self.reasons),
            "runtime_eligible": False,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class ChecksumVerification:
    """Structured result from verifying one local artifact."""

    path: str
    expected_sha256: str | None
    actual_sha256: str | None
    ok: bool
    code: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "path": self.path,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
            "ok": self.ok,
            "code": self.code,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RepositorySidecarVerification:
    """Checksum evidence for one registry-owned checkpoint spec sidecar."""

    checkpoint_id: str
    resource_path: str
    checksum: ChecksumVerification

    @property
    def ok(self) -> bool:
        return self.checksum.ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "resource_path": self.resource_path,
            "ok": self.ok,
            "checksum": self.checksum.to_dict(),
        }


@dataclass(frozen=True)
class SpecOverwrite:
    """One leaf value replaced by a higher-precedence spec layer."""

    path: str
    previous_layer: str
    replacement_layer: str


@dataclass(frozen=True)
class SpecMergeResult:
    """Merged specification plus deterministic provenance."""

    spec: dict[str, Any]
    precedence: tuple[str, ...]
    layer_sha256: tuple[tuple[str, str], ...]
    final_sha256: str
    overwritten: tuple[SpecOverwrite, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "spec": copy.deepcopy(self.spec),
            "precedence": list(self.precedence),
            "layer_sha256": dict(self.layer_sha256),
            "final_sha256": self.final_sha256,
            "overwritten": [
                {
                    "path": item.path,
                    "previous_layer": item.previous_layer,
                    "replacement_layer": item.replacement_layer,
                }
                for item in self.overwritten
            ],
        }


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_string_list(value: Any, *, allow_empty: bool = False) -> bool:
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(_is_nonempty_string(item) for item in value)
    )


def _reject_unknown_fields(
    value: Any,
    allowed: frozenset[str],
    path: str,
    errors: list[str],
) -> None:
    """Reject misspelled contract fields in this versioned registry."""
    if not isinstance(value, Mapping):
        return
    unknown = sorted(set(value) - allowed, key=str)
    if unknown:
        errors.append(f"{path} contains unsupported field(s): {unknown}")


def _validate_sha256(value: Any, path: str, errors: list[str]) -> None:
    if value is not None and (
        not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
    ):
        errors.append(f"{path} must be a 64-character hexadecimal SHA-256 digest")


def _validate_expected_size(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(f"{path} must be a positive integer")


def _specifier_set(value: str) -> SpecifierSet:
    text = value.strip()
    if not text.startswith(_SPECIFIER_PREFIXES):
        text = f"=={text}"
    return SpecifierSet(text)


def _normalized_version(value: str) -> Version:
    match = _VERSION_PREFIX_RE.match(str(value))
    if not match:
        raise InvalidVersion(str(value))
    return Version(match.group(1))


def _validate_source(source: Any, path: str, errors: list[str]) -> None:
    if not isinstance(source, Mapping):
        errors.append(f"{path} must be an object")
        return
    _reject_unknown_fields(source, _SOURCE_FIELDS, path, errors)
    for field in ("provider", "registry", "resource", "version", "member"):
        if not _is_nonempty_string(source.get(field)):
            errors.append(f"{path}.{field} must be a non-empty string")
    if source.get("official") is not True:
        errors.append(f"{path}.official must be true for a supported checkpoint")
    if str(source.get("version", "")).strip().lower() == "latest":
        errors.append(f"{path}.version must be exact; 'latest' is not immutable")
    immutable = source.get("immutable_identity")
    if immutable is not None and not _is_nonempty_string(immutable):
        errors.append(f"{path}.immutable_identity must be a non-empty string")


def _validate_checkpoint_spec_file(
    value: Any,
    path: str,
    errors: list[str],
) -> None:
    if not isinstance(value, Mapping):
        errors.append(
            f"{path} must describe the checkpoint-specific YAML/spec artifact"
        )
        return
    _reject_unknown_fields(value, _CHECKPOINT_SPEC_FIELDS, path, errors)
    source = value.get("source", "checkpoint_source")
    if source == "repository":
        sidecar_path = value.get("path")
        if not _is_nonempty_string(sidecar_path):
            errors.append(f"{path}.path must be a non-empty packaged resource path")
        else:
            pure_path = PurePosixPath(sidecar_path)
            if (
                pure_path.is_absolute()
                or ".." in pure_path.parts
                or not pure_path.parts
                or pure_path.parts[0] != "data"
            ):
                errors.append(
                    f"{path}.path must be a safe package-relative path under data/"
                )
        _validate_sha256(value.get("sha256"), f"{path}.sha256", errors)
        if value.get("sha256") is None:
            errors.append(f"{path}.sha256 is required for a repository sidecar")
        provenance = value.get("provenance")
        if not isinstance(provenance, Mapping):
            errors.append(f"{path}.provenance must be an object")
        else:
            _reject_unknown_fields(
                provenance,
                _PROVENANCE_FIELDS,
                f"{path}.provenance",
                errors,
            )
            for field in ("source", "evidence"):
                if not _is_nonempty_string(provenance.get(field)):
                    errors.append(
                        f"{path}.provenance.{field} must be a non-empty string"
                    )
        return

    if not _is_nonempty_string(value.get("member")):
        errors.append(f"{path}.member must be a non-empty string")
    _validate_expected_size(
        value.get("expected_size_bytes"),
        f"{path}.expected_size_bytes",
        errors,
    )
    if source != "checkpoint_source" and not isinstance(source, Mapping):
        errors.append(
            f"{path}.source must be 'checkpoint_source', 'repository', "
            "or a complete source object"
        )
    elif isinstance(source, Mapping):
        _validate_source(source, f"{path}.source", errors)
    _validate_sha256(value.get("sha256"), f"{path}.sha256", errors)
    immutable = value.get("immutable_identity")
    if immutable is not None and not _is_nonempty_string(immutable):
        errors.append(f"{path}.immutable_identity must be a non-empty string")
    if value.get("sha256") is None and not _is_nonempty_string(immutable):
        errors.append(
            f"{path} requires sha256 or immutable_identity"
        )


def _validate_artifact_adapters(
    value: Any,
    path: str,
    errors: list[str],
) -> None:
    """Validate declarative, version-scoped checkpoint adaptation recipes.

    Registry records intentionally contain data only. Execution is delegated
    to an injected callback after the official input artifact is verified.
    """
    if not isinstance(value, list) or not value:
        errors.append(f"{path} must be a non-empty list")
        return

    seen_ids: set[str] = set()
    for index, adapter in enumerate(value):
        adapter_path = f"{path}[{index}]"
        if not isinstance(adapter, Mapping):
            errors.append(f"{adapter_path} must be an object")
            continue
        _reject_unknown_fields(
            adapter,
            _ARTIFACT_ADAPTER_FIELDS,
            adapter_path,
            errors,
        )

        adapter_id = adapter.get("id")
        if (
            not isinstance(adapter_id, str)
            or _CHECKPOINT_ID_RE.fullmatch(adapter_id) is None
        ):
            errors.append(
                f"{adapter_path}.id must be a stable lowercase adapter identifier"
            )
        elif adapter_id in seen_ids:
            errors.append(f"{adapter_path}.id duplicates adapter {adapter_id!r}")
        else:
            seen_ids.add(adapter_id)

        adapter_type = adapter.get("adapter_type")
        if adapter_type not in _ARTIFACT_ADAPTER_TYPES:
            errors.append(
                f"{adapter_path}.adapter_type must be one of "
                f"{sorted(_ARTIFACT_ADAPTER_TYPES)}"
            )

        versions = adapter.get("compatible_tao_versions")
        if not _is_string_list(versions):
            errors.append(
                f"{adapter_path}.compatible_tao_versions must be a "
                "non-empty string list"
            )
        else:
            for version_index, specifier in enumerate(versions):
                try:
                    _specifier_set(specifier)
                except InvalidSpecifier:
                    errors.append(
                        f"{adapter_path}.compatible_tao_versions"
                        f"[{version_index}] is not a valid version specifier"
                    )

        recipe = adapter.get("recipe")
        if not isinstance(recipe, Mapping):
            errors.append(f"{adapter_path}.recipe must be an object")
        else:
            recipe_path = f"{adapter_path}.recipe"
            _reject_unknown_fields(
                recipe,
                _ARTIFACT_ADAPTER_RECIPE_FIELDS,
                recipe_path,
                errors,
            )
            retained = recipe.get("retain_top_level_keys")
            if not _is_string_list(retained):
                errors.append(
                    f"{recipe_path}.retain_top_level_keys must be a "
                    "non-empty string list"
                )
                retained_keys: set[str] = set()
            else:
                retained_keys = set(retained)
                if len(retained_keys) != len(retained):
                    errors.append(
                        f"{recipe_path}.retain_top_level_keys must be unique"
                    )

            metadata = recipe.get("add_top_level_metadata")
            if not isinstance(metadata, Mapping) or not metadata:
                errors.append(
                    f"{recipe_path}.add_top_level_metadata must be a "
                    "non-empty object"
                )
                metadata_keys: set[str] = set()
            else:
                metadata_keys = set()
                for key, metadata_value in metadata.items():
                    if not _is_nonempty_string(key):
                        errors.append(
                            f"{recipe_path}.add_top_level_metadata keys must "
                            "be non-empty strings"
                        )
                        continue
                    metadata_keys.add(key)
                    if metadata_value is not None and not isinstance(
                        metadata_value,
                        (str, int, float, bool),
                    ):
                        errors.append(
                            f"{recipe_path}.add_top_level_metadata.{key} "
                            "must be a JSON scalar"
                        )
                    elif isinstance(metadata_value, float) and not math.isfinite(
                        metadata_value
                    ):
                        errors.append(
                            f"{recipe_path}.add_top_level_metadata.{key} "
                            "must be finite"
                        )
            overlap = sorted(retained_keys & metadata_keys)
            if overlap:
                errors.append(
                    f"{recipe_path} cannot retain and replace the same "
                    f"top-level key(s): {overlap}"
                )

            tensor_key = recipe.get("tensor_container_key")
            if not _is_nonempty_string(tensor_key):
                errors.append(
                    f"{recipe_path}.tensor_container_key must be a "
                    "non-empty string"
                )
            elif tensor_key not in retained_keys:
                errors.append(
                    f"{recipe_path}.tensor_container_key must be retained"
                )
            for field in (
                "require_exact_tensor_key_set",
                "require_exact_tensor_values",
            ):
                if recipe.get(field) is not True:
                    errors.append(f"{recipe_path}.{field} must be true")

        output = adapter.get("output")
        if not isinstance(output, Mapping):
            errors.append(f"{adapter_path}.output must be an object")
        else:
            output_path = f"{adapter_path}.output"
            _reject_unknown_fields(
                output,
                _ARTIFACT_ADAPTER_OUTPUT_FIELDS,
                output_path,
                errors,
            )
            member = output.get("member")
            if not _is_nonempty_string(member):
                errors.append(f"{output_path}.member must be a non-empty string")
            else:
                pure_member = PurePosixPath(member)
                if (
                    pure_member.is_absolute()
                    or "\\" in member
                    or any(part in ("", ".", "..") for part in member.split("/"))
                ):
                    errors.append(
                        f"{output_path}.member must be a safe relative member path"
                    )
            _validate_expected_size(
                output.get("expected_size_bytes"),
                f"{output_path}.expected_size_bytes",
                errors,
            )
            _validate_sha256(
                output.get("sha256"),
                f"{output_path}.sha256",
                errors,
            )
            if output.get("sha256") is None:
                errors.append(f"{output_path}.sha256 is required")

        provenance = adapter.get("provenance")
        if not isinstance(provenance, Mapping):
            errors.append(f"{adapter_path}.provenance must be an object")
        else:
            provenance_path = f"{adapter_path}.provenance"
            _reject_unknown_fields(
                provenance,
                _PROVENANCE_FIELDS,
                provenance_path,
                errors,
            )
            for field in ("source", "evidence"):
                if not _is_nonempty_string(provenance.get(field)):
                    errors.append(
                        f"{provenance_path}.{field} must be a non-empty string"
                    )


def _validate_supported_record(
    record: Mapping[str, Any],
    model: str,
    path: str,
    errors: list[str],
) -> None:
    required_strings = (
        "model_family",
        "architecture",
        "backbone",
        "checkpoint_target",
    )
    for field in required_strings:
        if not _is_nonempty_string(record.get(field)):
            errors.append(f"{path}.{field} must be a non-empty string")
    if record.get("model_family") != model:
        errors.append(f"{path}.model_family must equal registry model key {model!r}")

    source = record.get("source")
    _validate_source(source, f"{path}.source", errors)
    _validate_sha256(record.get("sha256"), f"{path}.sha256", errors)
    _validate_expected_size(
        record.get("expected_size_bytes"),
        f"{path}.expected_size_bytes",
        errors,
    )
    immutable = source.get("immutable_identity") if isinstance(source, Mapping) else None
    if record.get("sha256") is None and not _is_nonempty_string(immutable):
        errors.append(
            f"{path} requires sha256 or source.immutable_identity"
        )

    compat = record.get("compatible_tao_versions")
    if not _is_string_list(compat):
        errors.append(
            f"{path}.compatible_tao_versions must be a non-empty string list"
        )
    else:
        for index, specifier in enumerate(compat):
            try:
                _specifier_set(specifier)
            except InvalidSpecifier:
                errors.append(
                    f"{path}.compatible_tao_versions[{index}] is not a valid version specifier"
                )

    tasks = record.get("task_compatibility")
    if not _is_string_list(tasks):
        errors.append(
            f"{path}.task_compatibility must be a non-empty string list"
        )

    input_contract = record.get("input_contract")
    if not isinstance(input_contract, Mapping):
        errors.append(f"{path}.input_contract must be an object")
    else:
        _reject_unknown_fields(
            input_contract,
            _INPUT_CONTRACT_FIELDS,
            f"{path}.input_contract",
            errors,
        )
        channels = input_contract.get("channels")
        if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
            errors.append(f"{path}.input_contract.channels must be a positive integer")
        for dimension in ("height", "width"):
            value = input_contract.get(dimension)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                errors.append(
                    f"{path}.input_contract.{dimension} must be null or a positive integer"
                )

    if not isinstance(record.get("default_spec_overrides"), Mapping):
        errors.append(f"{path}.default_spec_overrides must be an object")
    _validate_checkpoint_spec_file(
        record.get("checkpoint_spec_file"),
        f"{path}.checkpoint_spec_file",
        errors,
    )

    license_info = record.get("license")
    if not isinstance(license_info, Mapping):
        errors.append(f"{path}.license must be an object")
    else:
        _reject_unknown_fields(
            license_info,
            _LICENSE_FIELDS,
            f"{path}.license",
            errors,
        )
        if not _is_nonempty_string(license_info.get("name")):
            errors.append(f"{path}.license.name must be a non-empty string")
        if not _is_string_list(
            license_info.get("access_requirements"), allow_empty=True
        ):
            errors.append(
                f"{path}.license.access_requirements must be a string list"
            )

    deprecation = record.get("deprecation")
    if not isinstance(deprecation, Mapping):
        errors.append(f"{path}.deprecation must be an object")
    else:
        _reject_unknown_fields(
            deprecation,
            _DEPRECATION_FIELDS,
            f"{path}.deprecation",
            errors,
        )
        if deprecation.get("is_deprecated") is not False:
            errors.append(
                f"{path}.deprecation.is_deprecated must be false when status is supported"
            )

    validation = record.get("validation")
    if not isinstance(validation, Mapping):
        errors.append(f"{path}.validation must be an object")
    else:
        _reject_unknown_fields(
            validation,
            _VALIDATION_FIELDS,
            f"{path}.validation",
            errors,
        )
        if validation.get("status") != "validated":
            errors.append(f"{path}.validation.status must be 'validated'")
        for field in ("tao_version", "container_identity", "evidence"):
            if not _is_nonempty_string(validation.get(field)):
                errors.append(
                    f"{path}.validation.{field} must be a non-empty string"
                )


def _qualification_metadata_errors(
    record: Mapping[str, Any],
    *,
    model: str,
    tao_version: Version,
) -> tuple[str, ...]:
    """Apply the executable-record contract without claiming validation.

    Qualification intentionally targets the caller's TAO version, even when a
    not-yet-supported record has no compatibility claim. Existing compatibility
    specifiers are still syntax-checked, but do not prevent revalidation on a
    different TAO release.
    """
    candidate = copy.deepcopy(dict(record))
    candidate.setdefault(
        "compatible_tao_versions",
        [f"=={tao_version}"],
    )
    candidate["validation"] = {
        "status": "validated",
        "tao_version": str(tao_version),
        "container_identity": "qualification-only",
        "evidence": "qualification-only",
    }
    errors: list[str] = []
    _validate_supported_record(
        candidate,
        model,
        f"qualification.{record.get('id', '<unknown>')}",
        errors,
    )
    return tuple(errors)


def _reject_record_nested_unknown_fields(
    record: Mapping[str, Any],
    path: str,
    errors: list[str],
) -> None:
    """Close nested registry objects even for partial qualification records."""
    for field, allowed in (
        ("source", _SOURCE_FIELDS),
        ("input_contract", _INPUT_CONTRACT_FIELDS),
        ("checkpoint_spec_file", _CHECKPOINT_SPEC_FIELDS),
        ("license", _LICENSE_FIELDS),
        ("deprecation", _DEPRECATION_FIELDS),
        ("validation", _VALIDATION_FIELDS),
    ):
        value = record.get(field)
        if isinstance(value, Mapping):
            _reject_unknown_fields(value, allowed, f"{path}.{field}", errors)
    spec_file = record.get("checkpoint_spec_file")
    if isinstance(spec_file, Mapping):
        provenance = spec_file.get("provenance")
        if isinstance(provenance, Mapping):
            _reject_unknown_fields(
                provenance,
                _PROVENANCE_FIELDS,
                f"{path}.checkpoint_spec_file.provenance",
                errors,
            )
    adapters = record.get("artifact_adapters")
    if isinstance(adapters, list):
        for index, adapter in enumerate(adapters):
            if not isinstance(adapter, Mapping):
                continue
            adapter_path = f"{path}.artifact_adapters[{index}]"
            _reject_unknown_fields(
                adapter,
                _ARTIFACT_ADAPTER_FIELDS,
                adapter_path,
                errors,
            )
            recipe = adapter.get("recipe")
            if isinstance(recipe, Mapping):
                _reject_unknown_fields(
                    recipe,
                    _ARTIFACT_ADAPTER_RECIPE_FIELDS,
                    f"{adapter_path}.recipe",
                    errors,
                )
            output = adapter.get("output")
            if isinstance(output, Mapping):
                _reject_unknown_fields(
                    output,
                    _ARTIFACT_ADAPTER_OUTPUT_FIELDS,
                    f"{adapter_path}.output",
                    errors,
                )
            provenance = adapter.get("provenance")
            if isinstance(provenance, Mapping):
                _reject_unknown_fields(
                    provenance,
                    _PROVENANCE_FIELDS,
                    f"{adapter_path}.provenance",
                    errors,
                )


def validate_ptm_registry(document: Any) -> None:
    """Validate one schema-version-1 PTM registry document.

    Non-supported records may intentionally be partial, but they must be
    explicit about their status and carry a reason.  A record advertised as
    supported must contain the full immutable execution contract.
    """
    errors: list[str] = []
    if not isinstance(document, Mapping):
        raise PTMRegistryValidationError(("registry root must be an object",))

    _reject_unknown_fields(document, _ROOT_FIELDS, "registry", errors)
    if document.get("schema_version") != PTM_REGISTRY_SCHEMA_VERSION:
        errors.append(
            f"schema_version must equal {PTM_REGISTRY_SCHEMA_VERSION}"
        )
    if not _is_nonempty_string(document.get("registry_version")):
        errors.append("registry_version must be a non-empty string")
    models = document.get("models")
    if not isinstance(models, Mapping):
        errors.append("models must be an object")
        models = {}

    seen_ids: dict[str, str] = {}
    for model in sorted(models, key=str):
        model_path = f"models.{model}"
        if not isinstance(model, str) or _MODEL_ID_RE.fullmatch(model) is None:
            errors.append(
                f"{model_path} model key must use lowercase letters, digits, '_' or '-'"
            )
        config = models[model]
        if not isinstance(config, Mapping):
            errors.append(f"{model_path} must be an object")
            continue
        _reject_unknown_fields(config, _MODEL_FIELDS, model_path, errors)
        if "default_ptm" not in config:
            errors.append(f"{model_path}.default_ptm is required")
        if "checkpoints" not in config:
            errors.append(f"{model_path}.checkpoints is required")
        checkpoints = config.get("checkpoints")
        if not isinstance(checkpoints, list):
            errors.append(f"{model_path}.checkpoints must be a list")
            checkpoints = []

        local_records: dict[str, Mapping[str, Any]] = {}
        for index, record in enumerate(checkpoints):
            path = f"{model_path}.checkpoints[{index}]"
            if not isinstance(record, Mapping):
                errors.append(f"{path} must be an object")
                continue
            _reject_unknown_fields(record, _CHECKPOINT_FIELDS, path, errors)
            _reject_record_nested_unknown_fields(record, path, errors)
            if "artifact_adapters" in record:
                _validate_artifact_adapters(
                    record["artifact_adapters"],
                    f"{path}.artifact_adapters",
                    errors,
                )
            checkpoint_id = record.get("id")
            if (
                not isinstance(checkpoint_id, str)
                or _CHECKPOINT_ID_RE.fullmatch(checkpoint_id) is None
            ):
                errors.append(
                    f"{path}.id must be a stable lowercase checkpoint identifier"
                )
                continue
            if checkpoint_id in seen_ids:
                errors.append(
                    f"{path}.id duplicates {seen_ids[checkpoint_id]}"
                )
            else:
                seen_ids[checkpoint_id] = f"{path}.id"
            local_records[checkpoint_id] = record

            status = record.get("status")
            if status not in PTM_STATUS_VALUES:
                errors.append(
                    f"{path}.status must be one of {sorted(PTM_STATUS_VALUES)}"
                )
                continue
            if status == "supported":
                _validate_supported_record(record, model, path, errors)
            elif not _is_nonempty_string(record.get("status_reason")):
                errors.append(
                    f"{path}.status_reason is required when status is {status!r}"
                )

        default_ptm = config.get("default_ptm")
        if default_ptm is not None:
            if not _is_nonempty_string(default_ptm):
                errors.append(
                    f"{model_path}.default_ptm must be null or a checkpoint ID"
                )
            elif default_ptm not in local_records:
                errors.append(
                    f"{model_path}.default_ptm references unknown checkpoint {default_ptm!r}"
                )
            elif local_records[default_ptm].get("status") != "supported":
                errors.append(
                    f"{model_path}.default_ptm must reference a supported checkpoint"
                )

    if errors:
        raise PTMRegistryValidationError(errors)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonically JSON serializable: {exc}") from exc
    return encoded.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return a stable SHA-256 over canonical JSON."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a local file's SHA-256 without loading it into memory."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _invalid_checksum_result(
    path: str,
    expected_sha256: Any,
) -> ChecksumVerification | None:
    if expected_sha256 is None:
        return ChecksumVerification(
            path,
            None,
            None,
            False,
            "checksum_unavailable",
            "No expected SHA-256 was registered for this artifact",
        )
    if not isinstance(expected_sha256, str) or _SHA256_RE.fullmatch(expected_sha256) is None:
        return ChecksumVerification(
            path,
            str(expected_sha256),
            None,
            False,
            "invalid_expected_checksum",
            "Expected SHA-256 must be a 64-character hexadecimal digest",
        )
    return None


def verify_file_sha256(
    path: str | Path,
    expected_sha256: str | None,
) -> ChecksumVerification:
    """Verify a local artifact and return a structured, non-throwing result."""
    normalized_path = str(Path(path))
    invalid = _invalid_checksum_result(normalized_path, expected_sha256)
    if invalid is not None:
        return invalid
    artifact = Path(path)
    if not artifact.is_file():
        return ChecksumVerification(
            normalized_path,
            expected_sha256.lower(),
            None,
            False,
            "artifact_missing",
            "Artifact is missing or is not a regular file",
        )
    try:
        actual = sha256_file(artifact)
    except OSError as exc:
        return ChecksumVerification(
            normalized_path,
            expected_sha256.lower(),
            None,
            False,
            "artifact_unreadable",
            f"Artifact could not be read ({type(exc).__name__})",
        )
    expected = expected_sha256.lower()
    ok = actual == expected
    return ChecksumVerification(
        normalized_path,
        expected,
        actual,
        ok,
        "verified" if ok else "checksum_mismatch",
        "SHA-256 verified" if ok else "Artifact SHA-256 does not match registry",
    )


def verify_packaged_resource_sha256(
    resource_path: str,
    expected_sha256: str | None,
) -> ChecksumVerification:
    """Verify a package resource without assuming the wheel is unpacked."""
    normalized_path = f"package:tao_automl/{resource_path}"
    invalid = _invalid_checksum_result(normalized_path, expected_sha256)
    if invalid is not None:
        return invalid
    if not _is_nonempty_string(resource_path):
        return ChecksumVerification(
            normalized_path,
            expected_sha256.lower(),
            None,
            False,
            "invalid_resource_path",
            "Packaged resource must be a safe path under data/",
        )
    pure_path = PurePosixPath(resource_path)
    if (
        pure_path.is_absolute()
        or ".." in pure_path.parts
        or not pure_path.parts
        or pure_path.parts[0] != "data"
    ):
        return ChecksumVerification(
            normalized_path,
            expected_sha256.lower(),
            None,
            False,
            "invalid_resource_path",
            "Packaged resource must be a safe path under data/",
        )
    resource = resources.files("tao_automl").joinpath(*pure_path.parts)
    if not resource.is_file():
        return ChecksumVerification(
            normalized_path,
            expected_sha256.lower(),
            None,
            False,
            "artifact_missing",
            "Packaged artifact is missing or is not a regular file",
        )
    digest = hashlib.sha256()
    try:
        with resource.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        return ChecksumVerification(
            normalized_path,
            expected_sha256.lower(),
            None,
            False,
            "artifact_unreadable",
            f"Packaged artifact could not be read ({type(exc).__name__})",
        )
    actual = digest.hexdigest()
    expected = expected_sha256.lower()
    ok = actual == expected
    return ChecksumVerification(
        normalized_path,
        expected,
        actual,
        ok,
        "verified" if ok else "checksum_mismatch",
        "SHA-256 verified" if ok else "Artifact SHA-256 does not match registry",
    )


def _path_tokens(path: str) -> list[str | int]:
    if not isinstance(path, str) or _DOTTED_PATH_RE.fullmatch(path) is None:
        raise ValueError(f"invalid dotted/indexed spec path: {path!r}")
    tokens: list[str | int] = []
    for match in _PATH_TOKEN_RE.finditer(path):
        tokens.append(match.group(1) if match.group(1) is not None else int(match.group(2)))
    return tokens


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    tokens = _path_tokens(path)
    cursor: Any = target
    for index, token in enumerate(tokens[:-1]):
        next_token = tokens[index + 1]
        if isinstance(token, str):
            if not isinstance(cursor, dict):
                raise ValueError(f"spec path {path!r} traverses a non-object")
            if token not in cursor or cursor[token] is None:
                cursor[token] = [] if isinstance(next_token, int) else {}
            cursor = cursor[token]
        else:
            if not isinstance(cursor, list):
                raise ValueError(f"spec path {path!r} traverses a non-list")
            while len(cursor) <= token:
                cursor.append(None)
            if cursor[token] is None:
                cursor[token] = [] if isinstance(next_token, int) else {}
            cursor = cursor[token]

    leaf = tokens[-1]
    if isinstance(leaf, str):
        if not isinstance(cursor, dict):
            raise ValueError(f"spec path {path!r} has an object key under a non-object")
        cursor[leaf] = copy.deepcopy(value)
    else:
        if not isinstance(cursor, list):
            raise ValueError(f"spec path {path!r} has a list index under a non-list")
        while len(cursor) <= leaf:
            cursor.append(None)
        cursor[leaf] = copy.deepcopy(value)


def _leaf_paths(value: Any, prefix: str = "") -> tuple[str, ...]:
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key in sorted(value, key=str):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_leaf_paths(value[key], next_prefix))
        return tuple(paths) or ((prefix,) if prefix else ())
    if isinstance(value, list):
        paths = []
        for index, item in enumerate(value):
            next_prefix = f"{prefix}[{index}]"
            paths.extend(_leaf_paths(item, next_prefix))
        return tuple(paths) or ((prefix,) if prefix else ())
    return (prefix,)


def _declared_layer_paths(
    mapping: Mapping[str, Any],
    prefix: str = "",
) -> tuple[str, ...]:
    paths: list[str] = []
    for key in sorted(mapping, key=str):
        if not isinstance(key, str) or not key:
            raise ValueError("spec keys must be non-empty strings")
        path = f"{prefix}.{key}" if prefix else key
        value = mapping[key]
        if isinstance(value, Mapping) and "." not in key and "[" not in key:
            nested = _declared_layer_paths(value, path)
            paths.extend(nested or (path,))
        else:
            paths.extend(_leaf_paths(value, path))
    return tuple(paths)


def _validate_unambiguous_layer(
    layer: Mapping[str, Any],
    layer_name: str,
) -> None:
    paths = _declared_layer_paths(layer)
    seen: set[str] = set()
    for path in paths:
        if path in seen:
            raise ValueError(
                f"{layer_name} assigns spec path {path!r} more than once"
            )
        for prior in seen:
            if (
                path.startswith(f"{prior}.")
                or path.startswith(f"{prior}[")
                or prior.startswith(f"{path}.")
                or prior.startswith(f"{path}[")
            ):
                raise ValueError(
                    f"{layer_name} contains ambiguous overlapping paths "
                    f"{prior!r} and {path!r}"
                )
        seen.add(path)


def merge_ptm_spec_precedence(
    *,
    model_defaults: Mapping[str, Any],
    ptm_overrides: Mapping[str, Any] | None = None,
    automl_profile_overrides: Mapping[str, Any] | None = None,
    user_overrides: Mapping[str, Any] | None = None,
    candidate_overrides: Mapping[str, Any] | None = None,
) -> SpecMergeResult:
    """Merge specification layers using the frozen product precedence.

    Precedence, lowest to highest:

    ``model defaults < PTM overrides < AutoML profile < user < candidate``.

    Both nested mappings and dotted/indexed TAO spec keys are accepted.
    Inputs are never mutated.
    """
    layers = {
        "model_defaults": model_defaults,
        "ptm_overrides": ptm_overrides or {},
        "automl_profile_overrides": automl_profile_overrides or {},
        "user_overrides": user_overrides or {},
        "candidate_overrides": candidate_overrides or {},
    }
    for name, layer in layers.items():
        if not isinstance(layer, Mapping):
            raise TypeError(f"{name} must be a mapping")
        _validate_unambiguous_layer(layer, name)

    merged: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    overwritten: list[SpecOverwrite] = []

    def merge_mapping(mapping: Mapping[str, Any], layer_name: str, prefix: str = "") -> None:
        for key in sorted(mapping, key=str):
            if not isinstance(key, str) or not key:
                raise ValueError(f"{layer_name} spec keys must be non-empty strings")
            value = mapping[key]
            path = f"{prefix}.{key}" if prefix else key
            if "." in key or "[" in key:
                affected_paths = _leaf_paths(value, path)
                for affected in affected_paths:
                    if affected in provenance:
                        overwritten.append(
                            SpecOverwrite(affected, provenance[affected], layer_name)
                        )
                    provenance[affected] = layer_name
                _set_path(merged, path, value)
                continue

            current = merged.get(key) if not prefix else None
            if not prefix and isinstance(current, dict) and isinstance(value, Mapping):
                merge_mapping(value, layer_name, key)
                continue
            if prefix:
                existing: Any = merged
                for token in prefix.split("."):
                    existing = existing[token]
                if isinstance(existing.get(key), dict) and isinstance(value, Mapping):
                    merge_mapping(value, layer_name, path)
                    continue

            affected_paths = _leaf_paths(value, path)
            for affected in affected_paths:
                if affected in provenance:
                    overwritten.append(
                        SpecOverwrite(affected, provenance[affected], layer_name)
                    )
                provenance[affected] = layer_name
            _set_path(merged, path, value)

    layer_hashes = []
    for layer_name in SPEC_PRECEDENCE:
        layer = layers[layer_name]
        layer_hashes.append((layer_name, canonical_sha256(layer)))
        merge_mapping(layer, layer_name)

    return SpecMergeResult(
        spec=merged,
        precedence=SPEC_PRECEDENCE,
        layer_sha256=tuple(layer_hashes),
        final_sha256=canonical_sha256(merged),
        overwritten=tuple(overwritten),
    )


class PTMRegistry:
    """Validated immutable-view wrapper around a registry document."""

    def __init__(self, document: Mapping[str, Any]):
        validate_ptm_registry(document)
        self._document = copy.deepcopy(dict(document))
        self._records: dict[str, dict[str, Any]] = {}
        for config in self._document["models"].values():
            for record in config["checkpoints"]:
                self._records[record["id"]] = record

    @property
    def schema_version(self) -> int:
        return self._document["schema_version"]

    @property
    def registry_version(self) -> str:
        return self._document["registry_version"]

    @property
    def document_sha256(self) -> str:
        return canonical_sha256(self._document)

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(sorted(self._document["models"]))

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive copy of the validated registry."""
        return copy.deepcopy(self._document)

    def checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        """Return a defensive copy of one record."""
        try:
            return copy.deepcopy(self._records[checkpoint_id])
        except KeyError as exc:
            raise KeyError(f"Unknown PTM checkpoint ID: {checkpoint_id}") from exc

    def artifact_adapter(
        self,
        checkpoint_id: str,
        *,
        tao_version: str,
    ) -> dict[str, Any] | None:
        """Resolve at most one declarative adapter for a target TAO version.

        Adapter metadata does not alter compatibility or qualification. It is
        consulted only after an inventory has independently admitted the
        checkpoint.
        """
        record = self.checkpoint(checkpoint_id)
        try:
            normalized_tao_version = _normalized_version(tao_version)
        except InvalidVersion as exc:
            raise ValueError(f"Invalid TAO version: {tao_version!r}") from exc
        matches = [
            adapter
            for adapter in record.get("artifact_adapters", ())
            if any(
                normalized_tao_version in _specifier_set(specifier)
                for specifier in adapter["compatible_tao_versions"]
            )
        ]
        if len(matches) > 1:
            raise PTMArtifactAdapterResolutionError(
                f"Checkpoint {checkpoint_id!r} has {len(matches)} artifact "
                f"adapters matching TAO {normalized_tao_version}"
            )
        return copy.deepcopy(matches[0]) if matches else None

    def compatibility(
        self,
        model: str,
        *,
        tao_version: str,
        task: str,
    ) -> PTMCompatibilityResult:
        """Resolve supported checkpoints and structured exclusions."""
        model_config = self._document["models"].get(model)
        if model_config is None:
            return PTMCompatibilityResult(
                model=model,
                tao_version=str(tao_version),
                task=task,
                model_found=False,
                eligible_checkpoint_ids=(),
                excluded=(),
                default_checkpoint_id=None,
                reasons=(f"Model {model!r} is not present in the PTM registry",),
            )
        try:
            normalized_tao_version = _normalized_version(tao_version)
        except InvalidVersion as exc:
            raise ValueError(f"Invalid TAO version: {tao_version!r}") from exc
        if not _is_nonempty_string(task):
            raise ValueError("task must be a non-empty string")

        eligible: list[str] = []
        excluded: list[PTMExclusion] = []
        for record in sorted(model_config["checkpoints"], key=lambda item: item["id"]):
            checkpoint_id = record["id"]
            status = record["status"]
            codes: list[str] = []
            reasons: list[str] = []
            if status != "supported":
                codes.append(f"status_{status}")
                reasons.append(record["status_reason"])
            else:
                version_specs = record["compatible_tao_versions"]
                if not any(
                    normalized_tao_version in _specifier_set(specifier)
                    for specifier in version_specs
                ):
                    codes.append("tao_version_incompatible")
                    reasons.append(
                        f"TAO {normalized_tao_version} does not satisfy "
                        f"{version_specs}"
                    )
                if task not in record["task_compatibility"]:
                    codes.append("task_incompatible")
                    reasons.append(
                        f"Task {task!r} is not in {record['task_compatibility']}"
                    )
            if codes:
                excluded.append(
                    PTMExclusion(
                        checkpoint_id=checkpoint_id,
                        status=status,
                        codes=tuple(codes),
                        reasons=tuple(reasons),
                    )
                )
            else:
                eligible.append(checkpoint_id)

        default_ptm = model_config.get("default_ptm")
        return PTMCompatibilityResult(
            model=model,
            tao_version=str(tao_version),
            task=task,
            model_found=True,
            eligible_checkpoint_ids=tuple(eligible),
            excluded=tuple(excluded),
            default_checkpoint_id=default_ptm if default_ptm in eligible else None,
        )

    def qualification(
        self,
        model: str,
        *,
        tao_version: str,
        task: str,
        validation_statuses: Sequence[str],
    ) -> PTMQualificationResult:
        """Resolve rich non-supported records for explicit qualification only.

        This API does not change ``compatibility`` or runtime eligibility.
        Callers must opt in to the statuses they intend to qualify. Supported
        records are admitted only when the target TAO version is outside their
        current compatibility claim, allowing explicit requalification on a
        new release. Already-compatible supported records continue through the
        normal runtime preflight. Deprecated records remain excluded.
        """
        if isinstance(validation_statuses, (str, bytes)):
            raise ValueError("validation_statuses must be a sequence of statuses")
        try:
            status_values = tuple(validation_statuses)
        except TypeError as exc:
            raise ValueError(
                "validation_statuses must be a sequence of statuses"
            ) from exc
        if not all(isinstance(status, str) for status in status_values):
            raise ValueError("validation_statuses entries must be strings")
        requested = tuple(sorted(set(status_values)))
        if not requested:
            raise ValueError("validation_statuses must not be empty")
        invalid = set(requested) - PTM_QUALIFICATION_STATUS_VALUES
        if invalid:
            raise ValueError(
                "validation_statuses may contain only "
                f"{sorted(PTM_QUALIFICATION_STATUS_VALUES)}; got {sorted(invalid)}"
            )
        try:
            normalized_tao_version = _normalized_version(tao_version)
        except InvalidVersion as exc:
            raise ValueError(f"Invalid TAO version: {tao_version!r}") from exc
        if not _is_nonempty_string(task):
            raise ValueError("task must be a non-empty string")

        model_config = self._document["models"].get(model)
        if model_config is None:
            return PTMQualificationResult(
                model=model,
                tao_version=str(tao_version),
                task=task,
                validation_statuses=requested,
                model_found=False,
                candidate_checkpoint_ids=(),
                excluded=(),
                reasons=(f"Model {model!r} is not present in the PTM registry",),
            )

        candidates: list[str] = []
        excluded: list[PTMExclusion] = []
        for record in sorted(model_config["checkpoints"], key=lambda item: item["id"]):
            checkpoint_id = record["id"]
            status = record["status"]
            codes: list[str] = []
            reasons: list[str] = []
            if status not in requested:
                codes.append("status_not_requested_for_qualification")
                reasons.append(
                    f"Status {status!r} is not in requested qualification "
                    f"statuses {list(requested)}"
                )
            else:
                version_specs = record.get("compatible_tao_versions", ())
                already_compatible = (
                    status == "supported"
                    and isinstance(version_specs, list)
                    and any(
                        normalized_tao_version in _specifier_set(specifier)
                        for specifier in version_specs
                    )
                )
                if already_compatible:
                    codes.append("already_runtime_compatible")
                    reasons.append(
                        f"Supported checkpoint already declares TAO "
                        f"{normalized_tao_version} compatibility; use runtime "
                        "preflight"
                    )
                else:
                    metadata_errors = _qualification_metadata_errors(
                        record,
                        model=model,
                        tao_version=normalized_tao_version,
                    )
                    if metadata_errors:
                        codes.append("qualification_metadata_incomplete")
                        reasons.extend(metadata_errors)
                    elif task not in record["task_compatibility"]:
                        codes.append("task_incompatible")
                        reasons.append(
                            f"Task {task!r} is not in "
                            f"{record['task_compatibility']}"
                        )
            if codes:
                excluded.append(
                    PTMExclusion(
                        checkpoint_id=checkpoint_id,
                        status=status,
                        codes=tuple(codes),
                        reasons=tuple(reasons),
                    )
                )
            else:
                candidates.append(checkpoint_id)

        return PTMQualificationResult(
            model=model,
            tao_version=str(tao_version),
            task=task,
            validation_statuses=requested,
            model_found=True,
            candidate_checkpoint_ids=tuple(candidates),
            excluded=tuple(excluded),
        )

    def verify_checkpoint_files(
        self,
        checkpoint_id: str,
        *,
        checkpoint_path: str | Path,
        checkpoint_spec_path: str | Path | None = None,
    ) -> tuple[ChecksumVerification, ...]:
        """Verify downloaded checkpoint and optional checkpoint-spec artifacts."""
        record = self.checkpoint(checkpoint_id)
        results = [
            verify_file_sha256(checkpoint_path, record.get("sha256"))
        ]
        spec_record = record.get("checkpoint_spec_file")
        if checkpoint_spec_path is not None:
            expected = (
                spec_record.get("sha256")
                if isinstance(spec_record, Mapping)
                else None
            )
            results.append(verify_file_sha256(checkpoint_spec_path, expected))
        elif (
            isinstance(spec_record, Mapping)
            and spec_record.get("source") == "repository"
        ):
            results.append(
                verify_packaged_resource_sha256(
                    spec_record.get("path", ""),
                    spec_record.get("sha256"),
                )
            )
        return tuple(results)

    def repository_sidecar_verifications(
        self,
    ) -> tuple[RepositorySidecarVerification, ...]:
        """Verify every supported repository-owned spec sidecar."""
        results: list[RepositorySidecarVerification] = []
        for checkpoint_id in sorted(self._records):
            record = self._records[checkpoint_id]
            if record.get("status") != "supported":
                continue
            sidecar = record.get("checkpoint_spec_file")
            if (
                not isinstance(sidecar, Mapping)
                or sidecar.get("source") != "repository"
            ):
                continue
            resource_path = sidecar.get("path", "")
            results.append(
                RepositorySidecarVerification(
                    checkpoint_id=checkpoint_id,
                    resource_path=resource_path,
                    checksum=verify_packaged_resource_sha256(
                        resource_path,
                        sidecar.get("sha256"),
                    ),
                )
            )
        return tuple(results)

    def require_repository_sidecars(self) -> None:
        """Fail closed if a registered package sidecar is absent or altered."""
        failures = [
            result
            for result in self.repository_sidecar_verifications()
            if not result.ok
        ]
        if failures:
            raise PTMRegistryValidationError(
                tuple(
                    f"checkpoint {failure.checkpoint_id!r} repository sidecar "
                    f"{failure.resource_path!r}: {failure.checksum.reason}"
                    for failure in failures
                )
            )


def _load_json_resource(resource_name: str) -> dict[str, Any]:
    package_root = resources.files("tao_automl")
    resource = package_root.joinpath(*resource_name.split("/"))
    with resource.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise PTMRegistryValidationError(
            (f"packaged resource {resource_name} must contain a JSON object",)
        )
    return value


def load_ptm_registry(
    path: str | Path | None = None,
    *,
    verify_repository_sidecars: bool = True,
) -> PTMRegistry:
    """Load the packaged or explicit registry and verify packaged sidecars."""
    if path is None:
        document = _load_json_resource(PTM_REGISTRY_RESOURCE)
    else:
        with Path(path).open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    registry = PTMRegistry(document)
    if verify_repository_sidecars:
        registry.require_repository_sidecars()
    return registry


def load_ptm_registry_schema() -> dict[str, Any]:
    """Load the packaged machine-readable schema for external tooling."""
    return _load_json_resource(PTM_REGISTRY_SCHEMA_RESOURCE)


__all__ = [
    "ChecksumVerification",
    "PTMCompatibilityResult",
    "PTMArtifactAdapterResolutionError",
    "PTMExclusion",
    "PTMQualificationResult",
    "PTM_QUALIFICATION_STATUS_VALUES",
    "PTMRegistry",
    "PTMRegistryValidationError",
    "PTM_REGISTRY_RESOURCE",
    "PTM_REGISTRY_SCHEMA_RESOURCE",
    "PTM_REGISTRY_SCHEMA_VERSION",
    "SPEC_PRECEDENCE",
    "RepositorySidecarVerification",
    "SpecMergeResult",
    "SpecOverwrite",
    "canonical_sha256",
    "load_ptm_registry",
    "load_ptm_registry_schema",
    "merge_ptm_spec_precedence",
    "sha256_file",
    "validate_ptm_registry",
    "verify_file_sha256",
    "verify_packaged_resource_sha256",
]
