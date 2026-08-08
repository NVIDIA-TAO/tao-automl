# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create-only/resumable DINO PTM qualification driver.

The driver wires the repository registry, authenticated exact-member NGC
transport, atomic cache, DINO metadata projection, and an injected TAO 7.1
Docker load-smoke callback. Artifacts intentionally contain no timestamps,
absolute paths, credentials, callback exception text, or validation-time
mutation of registry eligibility.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import tao_automl.ptm_preflight as ptm_preflight_module
import tao_automl.ptm_registry as ptm_registry_module
from tao_automl.ptm_preflight import (
    AtomicArtifactCache,
    CheckpointLoadSmokeCallback,
    CheckpointLoadSmokeRequest,
    CheckpointLoadSmokeResult,
    DEFAULT_NGC_API_BASE_URL,
    NGCCredential,
    NGCHTTPSClient,
    PTMCheckpointPreflight,
)
from tao_automl.ptm_registry import (
    PTMRegistry,
    canonical_sha256,
    load_ptm_registry,
    sha256_file,
)

try:
    from .dino_checkpoint_adapter import (
        DINOCheckpointMetadataProjectionCallback,
    )
except ImportError:  # Direct execution/test import from this directory.
    from dino_checkpoint_adapter import DINOCheckpointMetadataProjectionCallback

try:
    from .tao71_docker_load_smoke import (
        PINNED_TAO71_DOCKER_IMAGE,
        TAO71DINOCheckpointLoadSmoke,
    )
except ImportError:  # Direct execution/test import from this directory.
    from tao71_docker_load_smoke import (
        PINNED_TAO71_DOCKER_IMAGE,
        TAO71DINOCheckpointLoadSmoke,
    )


QUALIFICATION_MANIFEST_SCHEMA_VERSION = 1
QUALIFICATION_COMPLETION_SCHEMA_VERSION = 1
QUALIFICATION_MANIFEST_FILENAME = "qualification_manifest.v1.json"
QUALIFICATION_COMPLETION_FILENAME = "qualification_completion.v1.json"
DEFAULT_TAO_VERSION = "7.1.0-rc-245"
DEFAULT_VALIDATION_STATUSES = ("supported", "unverified")
PINNED_TAO71_CONTAINER_IDENTITY = (
    "sha256:949c0ea8ace09ac91951be4169353cf214daaa3ede7db9eed94070b020361667"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTAINER_IDENTITY_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TAO71_RE = re.compile(r"^7\.1(?:\.0)?(?:$|[-+].*)")


class DINOQualificationError(RuntimeError):
    """Fail-closed qualification manifest, resume, or evidence error."""


@dataclass(frozen=True)
class DINOQualificationConfiguration:
    """Frozen non-secret DINO qualification inputs."""

    tao_version: str = DEFAULT_TAO_VERSION
    task: str = "object_detection"
    validation_statuses: tuple[str, ...] = DEFAULT_VALIDATION_STATUSES
    container_identity: str = PINNED_TAO71_CONTAINER_IDENTITY
    ngc_api_base_url: str = DEFAULT_NGC_API_BASE_URL
    checkpoint_ids: tuple[str, ...] | None = None
    upstream_completion_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tao_version, str) or (
            _TAO71_RE.fullmatch(self.tao_version) is None
        ):
            raise ValueError("tao_version must identify the pinned TAO 7.1 line")
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task must be a non-empty string")
        statuses = self.validation_statuses
        if isinstance(statuses, (str, bytes)) or not statuses:
            raise ValueError("validation_statuses must be a non-empty tuple")
        if tuple(sorted(set(statuses))) != tuple(statuses):
            raise ValueError(
                "validation_statuses must be unique and lexicographically sorted"
            )
        if _CONTAINER_IDENTITY_RE.fullmatch(self.container_identity) is None:
            raise ValueError(
                "container_identity must be an exact SHA-256 image identity"
            )
        if (
            not isinstance(self.ngc_api_base_url, str)
            or not self.ngc_api_base_url.startswith("https://")
        ):
            raise ValueError("ngc_api_base_url must be HTTPS")
        if self.checkpoint_ids is not None and (
            not self.checkpoint_ids
            or any(
                not isinstance(item, str) or not item.strip()
                for item in self.checkpoint_ids
            )
            or tuple(sorted(set(self.checkpoint_ids))) != self.checkpoint_ids
        ):
            raise ValueError(
                "checkpoint_ids must be non-empty, unique, and sorted"
            )
        if self.upstream_completion_sha256 is not None and (
            not isinstance(self.upstream_completion_sha256, str)
            or _SHA256_RE.fullmatch(self.upstream_completion_sha256) is None
        ):
            raise ValueError(
                "upstream_completion_sha256 must be a SHA-256 digest"
            )


class TAO71DockerLoadSmokeContract:
    """Validate evidence returned by an injected Docker load-smoke callback."""

    def __init__(
        self,
        delegate: CheckpointLoadSmokeCallback,
        *,
        tao_version: str,
        container_identity: str,
    ):
        if not callable(delegate):
            raise TypeError("Docker load-smoke delegate must be callable")
        self.delegate = delegate
        self.tao_version = tao_version
        self.container_identity = container_identity

    def __call__(
        self,
        request: CheckpointLoadSmokeRequest,
    ) -> CheckpointLoadSmokeResult:
        result = self.delegate(request)
        if not isinstance(result, CheckpointLoadSmokeResult) or not result.ok:
            return result
        try:
            actual_size = request.checkpoint_path.stat().st_size
            actual_sha = sha256_file(request.checkpoint_path)
        except OSError as exc:
            return CheckpointLoadSmokeResult(
                False,
                "tao71_docker_load_smoke_artifact_unreadable",
                "Load-smoke checkpoint evidence could not be verified",
                {"exception_type": type(exc).__name__},
            )
        expected = {
            "contract_version": 1,
            "execution_backend": "docker",
            "container_identity": self.container_identity,
            "tao_version": self.tao_version,
            "checkpoint_sha256": actual_sha,
            "checkpoint_size_bytes": actual_size,
            "checkpoint_loaded": True,
            "state_dict_compatible": True,
        }
        mismatched = sorted(
            key
            for key, value in expected.items()
            if result.details.get(key) != value
        )
        if mismatched:
            return CheckpointLoadSmokeResult(
                False,
                "invalid_tao71_docker_load_smoke_evidence",
                "Injected TAO 7.1 Docker load smoke returned invalid evidence",
                {"missing_or_mismatched_fields": mismatched},
            )
        return result


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_only_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically create a canonical JSON artifact without replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _canonical_bytes(value) + b"\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.partial-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _read_canonical_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise DINOQualificationError(
            f"Qualification artifact {path.name!r} is unreadable or invalid"
        ) from exc
    if not isinstance(value, dict):
        raise DINOQualificationError(
            f"Qualification artifact {path.name!r} must contain an object"
        )
    if raw != _canonical_bytes(value) + b"\n":
        raise DINOQualificationError(
            f"Qualification artifact {path.name!r} is not canonical"
        )
    return value


def _freeze_or_verify(
    path: Path,
    value: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    expected = _canonical_bytes(value) + b"\n"
    if resume:
        if not path.is_file() or path.is_symlink():
            raise DINOQualificationError(
                f"Resume requires frozen artifact {path.name!r}"
            )
        if path.read_bytes() != expected:
            raise DINOQualificationError(
                f"Resume rejected drift in frozen artifact {path.name!r}"
            )
        return
    _create_only_json(path, value)


def _adapter_inventory(
    registry: PTMRegistry,
    *,
    tao_version: str,
) -> list[dict[str, Any]]:
    document = registry.to_dict()
    records = document["models"]["dino"]["checkpoints"]
    inventory = []
    for record in sorted(records, key=lambda item: item["id"]):
        adapter = registry.artifact_adapter(
            record["id"],
            tao_version=tao_version,
        )
        item = {
            "checkpoint_id": record["id"],
            "registry_status": record["status"],
            "registry_record_sha256": canonical_sha256(record),
            "checkpoint_target": record.get("checkpoint_target"),
            "backbone": record.get("backbone"),
            "source": {
                "provider": record.get("source", {}).get("provider"),
                "registry": record.get("source", {}).get("registry"),
                "resource": record.get("source", {}).get("resource"),
                "version": record.get("source", {}).get("version"),
                "member": record.get("source", {}).get("member"),
                "expected_size_bytes": record.get("expected_size_bytes"),
                "sha256": record.get("sha256"),
            },
            "artifact_adapter": None,
        }
        if adapter is not None:
            item["artifact_adapter"] = {
                "id": adapter["id"],
                "adapter_type": adapter["adapter_type"],
                "adapter_sha256": canonical_sha256(adapter),
                "recipe_sha256": canonical_sha256(adapter["recipe"]),
                "output": copy.deepcopy(adapter["output"]),
            }
        inventory.append(item)
    return inventory


def _source_file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    metadata = resolved.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise DINOQualificationError(
            "Qualification implementation source must be a regular file"
        )
    return {
        "sha256": sha256_file(resolved),
        "size_bytes": metadata.st_size,
    }


def build_dino_qualification_manifest(
    registry: PTMRegistry,
    configuration: DINOQualificationConfiguration,
    *,
    artifact_adapter: Any | None = None,
    docker_load_smoke: CheckpointLoadSmokeCallback | None = None,
) -> dict[str, Any]:
    """Build the deterministic non-secret manifest frozen before execution."""
    if not isinstance(registry, PTMRegistry):
        raise TypeError("registry must be a PTMRegistry")
    active_adapter = (
        artifact_adapter
        if artifact_adapter is not None
        else DINOCheckpointMetadataProjectionCallback()
    )
    identity_callback = getattr(active_adapter, "manifest_identity", None)
    if callable(identity_callback):
        adapter_execution = identity_callback()
    else:
        adapter_execution = {
            "callback": "injected",
            "callback_type": (
                f"{type(active_adapter).__module__}."
                f"{type(active_adapter).__qualname__}"
            ),
        }
    active_smoke = (
        docker_load_smoke
        if docker_load_smoke is not None
        else TAO71DINOCheckpointLoadSmoke(
            tao_version=configuration.tao_version,
            container_identity=configuration.container_identity,
        )
    )
    smoke_identity_callback = getattr(
        active_smoke,
        "manifest_identity",
        None,
    )
    if callable(smoke_identity_callback):
        smoke_execution = smoke_identity_callback()
    else:
        smoke_execution = {
            "callback": "injected",
            "callback_type": (
                f"{type(active_smoke).__module__}."
                f"{type(active_smoke).__qualname__}"
            ),
        }
    body = {
        "schema_version": QUALIFICATION_MANIFEST_SCHEMA_VERSION,
        "purpose": "dino_ptm_tao71_qualification",
        "model": "dino",
        "task": configuration.task,
        "tao_version": configuration.tao_version,
        "validation_statuses": list(configuration.validation_statuses),
        "checkpoint_ids": (
            list(configuration.checkpoint_ids)
            if configuration.checkpoint_ids is not None
            else None
        ),
        "upstream_completion_sha256": (
            configuration.upstream_completion_sha256
        ),
        "registry_version": registry.registry_version,
        "registry_sha256": registry.document_sha256,
        "implementation_source": {
            "binding": "source_file_sha256",
            "qualification_driver": _source_file_identity(Path(__file__)),
            "ptm_preflight": _source_file_identity(
                Path(ptm_preflight_module.__file__)
            ),
            "ptm_registry": _source_file_identity(
                Path(ptm_registry_module.__file__)
            ),
        },
        "adapter_execution": adapter_execution,
        "load_smoke_contract": {
            **smoke_execution,
            "execution_backend": "docker",
            "container_identity": configuration.container_identity,
            "checkpoint_loaded": True,
            "state_dict_compatible": True,
        },
        "artifact_policy": {
            "create_only": True,
            "canonical_json": True,
            "resume_requires_byte_identity": True,
            "resume_reverifies_cached_artifacts": True,
            "runtime_eligibility_mutation": False,
        },
        "checkpoint_inventory": _adapter_inventory(
            registry,
            tao_version=configuration.tao_version,
        ),
    }
    return {
        **body,
        "manifest_sha256": canonical_sha256(body),
    }


def _completion_document(
    *,
    manifest: Mapping[str, Any],
    report: Any,
) -> dict[str, Any]:
    report_document = report.stable_dict()
    inventory = report_document.get("inventory", {})
    resolved = inventory.get("candidate_checkpoint_ids", [])
    requested = manifest.get("checkpoint_ids")
    evaluated = list(resolved if requested is None else requested)
    prepared_ids = [
        item["checkpoint_id"] for item in report_document.get("prepared", [])
        if item.get("checkpoint_id") in set(evaluated)
    ]
    excluded_ids = [
        item["checkpoint_id"] for item in report_document.get("exclusions", [])
        if item.get("checkpoint_id") in set(evaluated)
    ]
    if (
        len(evaluated) != len(set(evaluated))
        or len(prepared_ids) != len(set(prepared_ids))
        or len(excluded_ids) != len(set(excluded_ids))
        or set(prepared_ids) & set(excluded_ids)
        or set(prepared_ids) | set(excluded_ids) != set(evaluated)
    ):
        raise DINOQualificationError(
            "Qualification report does not account exactly once for every "
            "frozen candidate"
        )
    accounting = {
        "evaluated_checkpoint_ids": sorted(evaluated),
        "prepared_checkpoint_ids": sorted(prepared_ids),
        "excluded_checkpoint_ids": sorted(excluded_ids),
        "complete": True,
    }
    body = {
        "schema_version": QUALIFICATION_COMPLETION_SCHEMA_VERSION,
        "purpose": "dino_ptm_tao71_qualification",
        "manifest_sha256": manifest["manifest_sha256"],
        "upstream_completion_sha256": manifest.get(
            "upstream_completion_sha256"
        ),
        "production_report_sha256": report.report_sha256,
        "report": report_document,
        "candidate_accounting": accounting,
        "qualification_only": True,
        "runtime_eligibility_mutated": False,
        "selection_invoked": False,
        "agent_selected_checkpoint": False,
    }
    return {
        **body,
        "completion_sha256": canonical_sha256(body),
    }


def _validate_manifest_document(value: Mapping[str, Any]) -> None:
    body = dict(value)
    claimed = body.pop("manifest_sha256", None)
    if not isinstance(claimed, str) or _SHA256_RE.fullmatch(claimed) is None:
        raise DINOQualificationError("Frozen manifest SHA-256 is invalid")
    if canonical_sha256(body) != claimed:
        raise DINOQualificationError("Frozen manifest SHA-256 does not match")


def _validate_completion_document(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> None:
    body = dict(value)
    claimed = body.pop("completion_sha256", None)
    if not isinstance(claimed, str) or _SHA256_RE.fullmatch(claimed) is None:
        raise DINOQualificationError("Completion SHA-256 is invalid")
    if canonical_sha256(body) != claimed:
        raise DINOQualificationError("Completion SHA-256 does not match")
    if value.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise DINOQualificationError(
            "Completion does not reference the frozen manifest"
        )
    if value.get("upstream_completion_sha256") != manifest.get(
        "upstream_completion_sha256"
    ):
        raise DINOQualificationError(
            "Completion upstream qualification binding does not match"
        )
    report = value.get("report")
    if not isinstance(report, Mapping):
        raise DINOQualificationError("Completion report is invalid")
    if canonical_sha256(report) != value.get("production_report_sha256"):
        raise DINOQualificationError(
            "Completion production report SHA-256 does not match"
        )
    accounting = value.get("candidate_accounting")
    if (
        not isinstance(accounting, Mapping)
        or set(accounting) != {
            "evaluated_checkpoint_ids",
            "prepared_checkpoint_ids",
            "excluded_checkpoint_ids",
            "complete",
        }
        or accounting.get("complete") is not True
    ):
        raise DINOQualificationError(
            "Completion candidate accounting is invalid"
        )
    evaluated = accounting["evaluated_checkpoint_ids"]
    prepared_ids = accounting["prepared_checkpoint_ids"]
    excluded_ids = accounting["excluded_checkpoint_ids"]
    if any(
        not isinstance(items, list)
        or items != sorted(set(items))
        or any(not isinstance(item, str) or not item for item in items)
        for items in (evaluated, prepared_ids, excluded_ids)
    ):
        raise DINOQualificationError(
            "Completion candidate accounting IDs are invalid"
        )
    expected_evaluated = manifest.get("checkpoint_ids")
    if expected_evaluated is None:
        expected_evaluated = report.get("inventory", {}).get(
            "candidate_checkpoint_ids"
        )
    observed_prepared = sorted(
        item.get("checkpoint_id")
        for item in report.get("prepared", [])
        if item.get("checkpoint_id") in set(evaluated)
    )
    observed_excluded = sorted(
        item.get("checkpoint_id")
        for item in report.get("exclusions", [])
        if item.get("checkpoint_id") in set(evaluated)
    )
    if (
        evaluated != sorted(expected_evaluated or [])
        or prepared_ids != observed_prepared
        or excluded_ids != observed_excluded
        or set(prepared_ids) & set(excluded_ids)
        or set(prepared_ids) | set(excluded_ids) != set(evaluated)
    ):
        raise DINOQualificationError(
            "Completion candidate accounting does not match its report"
        )
    if (
        value.get("qualification_only") is not True
        or value.get("runtime_eligibility_mutated") is not False
        or value.get("selection_invoked") is not False
        or value.get("agent_selected_checkpoint") is not False
    ):
        raise DINOQualificationError(
            "Completion qualification-isolation flags are invalid"
        )


def _safe_cached_path(cache_root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise DINOQualificationError("Cached artifact relative path is invalid")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise DINOQualificationError("Cached artifact relative path is unsafe")
    candidate = cache_root.joinpath(*pure.parts)
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        raise DINOQualificationError(
            f"Cached qualification artifact {relative!r} is missing"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise DINOQualificationError(
            f"Cached qualification artifact {relative!r} is not regular"
        )
    return candidate


def _verify_resume_cache(
    completion: Mapping[str, Any],
    *,
    cache_root: Path,
) -> None:
    report = completion["report"]
    prepared = report.get("prepared", [])
    if not isinstance(prepared, list):
        raise DINOQualificationError("Completion prepared inventory is invalid")
    for item in prepared:
        if not isinstance(item, Mapping):
            raise DINOQualificationError(
                "Completion prepared inventory contains an invalid record"
            )
        artifacts = ["checkpoint", "checkpoint_spec_artifact"]
        if item.get("source_checkpoint") is not None:
            artifacts.append("source_checkpoint")
        for field_name in artifacts:
            artifact = item.get(field_name)
            if not isinstance(artifact, Mapping):
                raise DINOQualificationError(
                    f"Completion {field_name!r} evidence is invalid"
                )
            path = _safe_cached_path(
                cache_root,
                artifact.get("cache_relative_path"),
            )
            observed_size = path.stat().st_size
            observed_sha = sha256_file(path)
            if (
                observed_size != artifact.get("size_bytes")
                or observed_sha != artifact.get("sha256")
            ):
                raise DINOQualificationError(
                    f"Cached qualification artifact {field_name!r} "
                    "does not match frozen evidence"
                )


def load_verified_qualification_completion(
    *,
    output_dir: str | Path,
    cache_dir: str | Path,
) -> dict[str, Any]:
    """Load immutable qualification evidence and reverify cached artifacts."""
    output_root = Path(output_dir).expanduser().resolve()
    cache_root = Path(cache_dir).expanduser().resolve()
    manifest = _read_canonical_json(
        output_root / QUALIFICATION_MANIFEST_FILENAME
    )
    completion = _read_canonical_json(
        output_root / QUALIFICATION_COMPLETION_FILENAME
    )
    _validate_manifest_document(manifest)
    _validate_completion_document(completion, manifest=manifest)
    _verify_resume_cache(completion, cache_root=cache_root)
    return copy.deepcopy(completion)


def run_dino_ptm_qualification(
    *,
    output_dir: str | Path,
    cache_dir: str | Path,
    docker_load_smoke: CheckpointLoadSmokeCallback | None = None,
    configuration: DINOQualificationConfiguration | None = None,
    registry: PTMRegistry | None = None,
    registry_path: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    http_session: Any | None = None,
    artifact_adapter: Any | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run qualification, or verify and return a completed frozen resume.

    A completed resume performs no credential lookup, NGC request,
    transformation, Docker load smoke, or artifact rewrite.
    """
    if not isinstance(resume, bool):
        raise TypeError("resume must be bool")
    config = configuration or DINOQualificationConfiguration()
    if registry is not None and registry_path is not None:
        raise ValueError("Specify registry or registry_path, not both")
    active_registry = registry or load_ptm_registry(registry_path)
    output_root = Path(output_dir).expanduser().resolve()
    cache_root = Path(cache_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    active_adapter = (
        artifact_adapter
        if artifact_adapter is not None
        else DINOCheckpointMetadataProjectionCallback()
    )
    active_smoke = (
        docker_load_smoke
        if docker_load_smoke is not None
        else TAO71DINOCheckpointLoadSmoke(
            tao_version=config.tao_version,
            container_identity=config.container_identity,
        )
    )
    manifest = build_dino_qualification_manifest(
        active_registry,
        config,
        artifact_adapter=active_adapter,
        docker_load_smoke=active_smoke,
    )
    manifest_path = output_root / QUALIFICATION_MANIFEST_FILENAME
    completion_path = output_root / QUALIFICATION_COMPLETION_FILENAME
    _freeze_or_verify(manifest_path, manifest, resume=resume)
    _validate_manifest_document(manifest)

    if resume and completion_path.exists():
        if completion_path.is_symlink() or not completion_path.is_file():
            raise DINOQualificationError(
                "Resume completion artifact must be a regular file"
            )
        completion = _read_canonical_json(completion_path)
        _validate_completion_document(completion, manifest=manifest)
        _verify_resume_cache(completion, cache_root=cache_root)
        return copy.deepcopy(completion)

    credential = NGCCredential.from_environment(environment)
    ngc_client = NGCHTTPSClient(
        credential,
        session=http_session,
        api_base_url=config.ngc_api_base_url,
    )
    validated_smoke = TAO71DockerLoadSmokeContract(
        active_smoke,
        tao_version=config.tao_version,
        container_identity=config.container_identity,
    )
    preflight = PTMCheckpointPreflight(
        registry=active_registry,
        cache=AtomicArtifactCache(cache_root),
        ngc_client=ngc_client,
        load_smoke=validated_smoke,
        artifact_adapter=(
            active_adapter
        ),
    )
    report = preflight.run_qualification(
        model="dino",
        task=config.task,
        tao_version=config.tao_version,
        validation_statuses=config.validation_statuses,
        checkpoint_ids=config.checkpoint_ids,
    )
    completion = _completion_document(manifest=manifest, report=report)
    _validate_completion_document(completion, manifest=manifest)
    _create_only_json(completion_path, completion)
    return copy.deepcopy(completion)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify registered DINO PTMs with the pinned TAO 7.1 Docker "
            "adapter and checkpoint-load smoke. NGC_KEY is read only from "
            "the environment."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Create-only qualification evidence directory",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Atomic verified-checkpoint cache directory",
    )
    parser.add_argument(
        "--registry-path",
        help="Optional repository-owned PTM registry JSON path",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and return an existing byte-identical completion",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the non-secret production qualification CLI."""
    arguments = _parser().parse_args(argv)
    completion = run_dino_ptm_qualification(
        output_dir=arguments.output_dir,
        cache_dir=arguments.cache_dir,
        registry_path=arguments.registry_path,
        resume=arguments.resume,
    )
    prepared = completion["report"].get("prepared", [])
    exclusions = completion["report"].get("exclusions", [])
    summary = {
        "completion_sha256": completion["completion_sha256"],
        "manifest_sha256": completion["manifest_sha256"],
        "prepared_checkpoint_ids": [
            item["checkpoint_id"] for item in prepared
        ],
        "excluded_checkpoint_ids": [
            item["checkpoint_id"] for item in exclusions
        ],
    }
    print(_canonical_bytes(summary).decode("utf-8"))
    return 0 if prepared else 1


__all__ = [
    "DEFAULT_TAO_VERSION",
    "DEFAULT_VALIDATION_STATUSES",
    "DINOQualificationConfiguration",
    "DINOQualificationError",
    "PINNED_TAO71_CONTAINER_IDENTITY",
    "PINNED_TAO71_DOCKER_IMAGE",
    "QUALIFICATION_COMPLETION_FILENAME",
    "QUALIFICATION_MANIFEST_FILENAME",
    "TAO71DockerLoadSmokeContract",
    "build_dino_qualification_manifest",
    "load_verified_qualification_completion",
    "main",
    "run_dino_ptm_qualification",
]


if __name__ == "__main__":
    raise SystemExit(main())
