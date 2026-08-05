# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local-Docker execution boundary for the frozen DINO preflight plan.

The plan owns model/data/PTM semantics.  This module only translates its
absolute host inputs through declared Docker binds, invokes the checked-in
DINO action contracts through ``build_entrypoint``, and turns exact TAO
artifacts into the evidence expected by ``DINOModelPreflightAdapter``.

Credentials are environment-only.  Configuration never contains credential
values, job logs are not copied into evidence, the image must already exist
locally by digest, and ``DOCKER_PULL_POLICY=never`` is forced while constructing
the production ``DockerSDK``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Protocol

import yaml

import tao_automl.latency_benchmark as _latency_benchmark_module
import tao_automl.latency_stats as _latency_stats_module
from tao_automl.latency_benchmark import (
    ReplicaIdentity,
    run_replica_benchmark,
)

try:
    from .dino_preflight import (
        DINOPreflightCommand,
        DINOPreflightCommandPlan,
        DINOPreflightContractError,
        DINOPreflightExecutionResult,
        run_dino_local_preflight,
    )
except ImportError:  # pragma: no cover - direct script execution
    from dino_preflight import (  # type: ignore[no-redef]
        DINOPreflightCommand,
        DINOPreflightCommandPlan,
        DINOPreflightContractError,
        DINOPreflightExecutionResult,
        run_dino_local_preflight,
    )


LOCAL_EXECUTOR_SCHEMA_VERSION = 1
_DIGEST_IMAGE_RE = re.compile(
    r"^(?P<base>[^@\s]+)@sha256:(?P<digest>[0-9a-f]{64})$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SAFE_CODE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CHECKPOINT_RE = re.compile(
    r"^model_epoch_(?P<epoch>\d+)_step_(?P<step>\d+)\.(?:pth|tlt)$"
)
_SECRET_FIELD_FRAGMENTS = (
    "secret",
    "password",
    "token",
    "credential",
    "access_key",
    "ngc_key",
)
_TERMINAL_JOB_STATES = frozenset({"Complete", "Error", "Canceled"})
_PTM_SPEC_KEYS = (
    "train.pretrained_model_path",
    "train.resume_training_checkpoint_path",
    "model.pretrained_backbone_path",
)
_RESULT_IDS = (
    "full_epoch_checkpoint",
    "in_epoch_validation_metrics",
    "standalone_evaluation_metrics",
    "latency_aggregate",
    "resume_replay_state",
)


class DINOLocalExecutionError(RuntimeError):
    """Safe, classified local execution failure."""

    def __init__(self, code: str, message: str):
        if not isinstance(code, str) or _SAFE_CODE_RE.fullmatch(code) is None:
            raise ValueError("execution error code must be a safe identifier")
        super().__init__(message)
        self.code = code


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_copy(value: Any) -> Any:
    def thaw(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): thaw(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [thaw(child) for child in item]
        return copy.deepcopy(item)

    return json.loads(
        json.dumps(
            thaw(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )


def _require_absolute(path: str | Path, name: str) -> Path:
    result = Path(path)
    if not result.is_absolute():
        raise DINOLocalExecutionError(
            "invalid_configuration",
            f"{name} must be an absolute path",
        )
    return result.resolve(strict=False)


def _repository_without_tag(image: str) -> str:
    last_slash = image.rfind("/")
    last_colon = image.rfind(":")
    return image[:last_colon] if last_colon > last_slash else image


@dataclass(frozen=True, slots=True)
class DockerBind:
    """One explicit local-host Docker bind."""

    host_path: Path
    container_path: str
    read_only: bool

    def __post_init__(self) -> None:
        host = _require_absolute(self.host_path, "mount.host_path")
        if host in {Path("/"), Path("/localhome")}:
            raise DINOLocalExecutionError(
                "unsafe_mount",
                "broad host mounts are not permitted",
            )
        container = PurePosixPath(self.container_path)
        if (
            not self.container_path.startswith("/")
            or str(container) in {"/", "/etc", "/root", "/var/run"}
        ):
            raise DINOLocalExecutionError(
                "unsafe_mount",
                "mount.container_path must be a scoped absolute path",
            )
        if not isinstance(self.read_only, bool):
            raise DINOLocalExecutionError(
                "invalid_configuration",
                "mount.read_only must be boolean",
            )
        object.__setattr__(self, "host_path", host)
        object.__setattr__(self, "container_path", str(container))

    def to_sdk_dict(self) -> dict[str, Any]:
        return {
            "host_path": str(self.host_path),
            "container_path": self.container_path,
            "read_only": self.read_only,
        }


@dataclass(frozen=True, slots=True)
class DINOLocalExecutorConfig:
    """Non-secret, external configuration for one frozen plan."""

    plan_sha256: str
    image: str
    results_root: Path
    mounts: tuple[DockerBind, ...]
    required_environment: tuple[str, ...] = ()
    poll_interval_seconds: float = 5.0
    max_polls: int = 720
    shm_size: str = "16g"
    container_user: str | None = None
    schema_version: int = LOCAL_EXECUTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LOCAL_EXECUTOR_SCHEMA_VERSION:
            raise DINOLocalExecutionError(
                "invalid_configuration",
                "unsupported local executor schema version",
            )
        if not isinstance(self.plan_sha256, str) or not _SHA256_RE.fullmatch(
            self.plan_sha256
        ):
            raise DINOLocalExecutionError(
                "invalid_configuration",
                "plan_sha256 must be lowercase SHA-256 hex",
            )
        if _DIGEST_IMAGE_RE.fullmatch(self.image) is None:
            raise DINOLocalExecutionError(
                "unpinned_image",
                "image must be an exact name@sha256:<digest> reference",
            )
        root = _require_absolute(self.results_root, "results_root")
        object.__setattr__(self, "results_root", root)
        mounts = tuple(self.mounts)
        if not mounts or not all(isinstance(item, DockerBind) for item in mounts):
            raise DINOLocalExecutionError(
                "invalid_configuration",
                "mounts must contain typed Docker binds",
            )
        if len({item.container_path for item in mounts}) != len(mounts):
            raise DINOLocalExecutionError(
                "invalid_configuration",
                "mount container paths must be unique",
            )
        result_mounts = [
            item for item in mounts if item.container_path.rstrip("/") == "/results"
        ]
        if (
            len(result_mounts) != 1
            or result_mounts[0].read_only
            or result_mounts[0].host_path != root
        ):
            raise DINOLocalExecutionError(
                "invalid_configuration",
                "results_root must be the single writable /results bind",
            )
        object.__setattr__(self, "mounts", mounts)
        required = tuple(self.required_environment)
        if (
            len(set(required)) != len(required)
            or any(_ENV_NAME_RE.fullmatch(item) is None for item in required)
        ):
            raise DINOLocalExecutionError(
                "invalid_configuration",
                "required_environment must contain unique environment names",
            )
        object.__setattr__(self, "required_environment", required)
        if (
            isinstance(self.poll_interval_seconds, bool)
            or not isinstance(self.poll_interval_seconds, (int, float))
            or not math.isfinite(float(self.poll_interval_seconds))
            or float(self.poll_interval_seconds) <= 0
        ):
            raise DINOLocalExecutionError(
                "invalid_configuration",
                "poll_interval_seconds must be finite and > 0",
            )
        if (
            isinstance(self.max_polls, bool)
            or not isinstance(self.max_polls, int)
            or self.max_polls < 1
        ):
            raise DINOLocalExecutionError(
                "invalid_configuration",
                "max_polls must be an integer >= 1",
            )
        if not isinstance(self.shm_size, str) or not self.shm_size.strip():
            raise DINOLocalExecutionError(
                "invalid_configuration",
                "shm_size must be non-empty",
            )
        if self.container_user is not None and (
            not isinstance(self.container_user, str)
            or re.fullmatch(r"[1-9]\d*:\d+", self.container_user) is None
        ):
            raise DINOLocalExecutionError(
                "invalid_configuration",
                "container_user must be a non-root numeric UID:GID",
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DINOLocalExecutorConfig":
        if not isinstance(value, Mapping):
            raise DINOLocalExecutionError(
                "invalid_configuration",
                "executor configuration must be a mapping",
            )
        allowed = {
            "schema_version",
            "plan_sha256",
            "image",
            "results_root",
            "mounts",
            "required_environment",
            "poll_interval_seconds",
            "max_polls",
            "shm_size",
            "container_user",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            fragments = [
                item
                for item in unknown
                if any(fragment in item.lower() for fragment in _SECRET_FIELD_FRAGMENTS)
            ]
            code = "inline_secret_forbidden" if fragments else "invalid_configuration"
            raise DINOLocalExecutionError(
                code,
                "executor configuration contains unsupported fields",
            )
        raw_mounts = value.get("mounts")
        if not isinstance(raw_mounts, Sequence) or isinstance(
            raw_mounts, (str, bytes)
        ):
            raise DINOLocalExecutionError(
                "invalid_configuration",
                "mounts must be a list",
            )
        mounts = []
        for item in raw_mounts:
            if not isinstance(item, Mapping) or set(item) != {
                "host_path",
                "container_path",
                "read_only",
            }:
                raise DINOLocalExecutionError(
                    "invalid_configuration",
                    "each mount must declare host_path, container_path, read_only",
                )
            mounts.append(DockerBind(**dict(item)))
        kwargs = dict(value)
        kwargs["mounts"] = tuple(mounts)
        kwargs["results_root"] = Path(kwargs["results_root"])
        kwargs["required_environment"] = tuple(
            kwargs.get("required_environment", ())
        )
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: str | Path) -> "DINOLocalExecutorConfig":
        source = _require_absolute(path, "config")
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        return cls.from_mapping(raw)

    def public_dict(self) -> dict[str, Any]:
        """Return the complete non-secret launch contract."""
        return {
            "schema_version": self.schema_version,
            "plan_sha256": self.plan_sha256,
            "image": self.image,
            "results_root": str(self.results_root),
            "mounts": [item.to_sdk_dict() for item in self.mounts],
            "required_environment": list(self.required_environment),
            "poll_interval_seconds": float(self.poll_interval_seconds),
            "max_polls": self.max_polls,
            "shm_size": self.shm_size,
            "container_user": self.container_user,
            "gpu_count": 1,
            "docker_pull_policy": "never",
        }


@dataclass(frozen=True, slots=True)
class LatencyRuntime:
    """Prepared, untimed DINO model-forward callback."""

    step: Callable[[int, int], Any] = field(repr=False)
    synchronize: Callable[[], None] = field(repr=False)
    hardware_sha256: str
    clock_ns: Callable[[], int] = field(
        default=time.perf_counter_ns,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not all(
            callable(item)
            for item in (self.step, self.synchronize, self.clock_ns)
        ):
            raise TypeError("latency runtime callbacks must be callable")
        if (
            not isinstance(self.hardware_sha256, str)
            or _SHA256_RE.fullmatch(self.hardware_sha256) is None
        ):
            raise ValueError("hardware_sha256 must be lowercase SHA-256 hex")


@dataclass(frozen=True, slots=True)
class ContainerLatencyRuntime:
    """Request the packaged DINO model-forward worker in the pinned image.

    The marker deliberately has no caller-controlled command text.  The
    executor owns the fixed worker command, SDK-written YAML, mounts, image,
    one-GPU allocation, and output validation.
    """

    worker_contract_version: int = 1

    def __post_init__(self) -> None:
        if self.worker_contract_version != 1:
            raise ValueError("unsupported DINO latency worker contract")


class LatencyRuntimeFactory(Protocol):
    def __call__(
        self,
        *,
        plan: DINOPreflightCommandPlan,
        command: DINOPreflightCommand,
        checkpoint_path: Path,
        inference_spec: Mapping[str, Any],
    ) -> LatencyRuntime | ContainerLatencyRuntime:
        ...


class ResumeReplayRunner(Protocol):
    def __call__(
        self,
        *,
        plan: DINOPreflightCommandPlan,
        command: DINOPreflightCommand,
        state_path: Path,
    ) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True, slots=True)
class DINOLocalExecutorHooks:
    latency_runtime_factory: LatencyRuntimeFactory
    resume_replay_runner: ResumeReplayRunner

    def __post_init__(self) -> None:
        if not callable(self.latency_runtime_factory):
            raise TypeError("latency_runtime_factory must be callable")
        if not callable(self.resume_replay_runner):
            raise TypeError("resume_replay_runner must be callable")


@dataclass(frozen=True, slots=True)
class _ActionOutcome:
    action: str
    job_id: str
    job_root: Path
    status_path: Path
    records: tuple[Mapping[str, Any], ...]
    metric: float | None = None
    checkpoint_path: Path | None = None
    completed_epochs: int | None = None
    training_steps: int | None = None


@dataclass(frozen=True, slots=True)
class _Artifact:
    artifact_id: str
    path: Path
    sha256: str
    size_bytes: int

    @classmethod
    def from_file(cls, artifact_id: str, path: Path) -> "_Artifact":
        if not path.is_file():
            raise DINOLocalExecutionError(
                "missing_artifact",
                f"required artifact {artifact_id!r} is missing",
            )
        return cls(
            artifact_id=artifact_id,
            path=path,
            sha256=_sha256_file(path),
            size_bytes=path.stat().st_size,
        )

    def evidence(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def _get_nested(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        match = re.fullmatch(r"([A-Za-z_]\w*)(?:\[(\d+)\])?", part)
        if match is None or not isinstance(current, Mapping):
            return None
        current = current.get(match.group(1))
        if match.group(2) is not None:
            if not isinstance(current, list):
                return None
            index = int(match.group(2))
            if index >= len(current):
                return None
            current = current[index]
    return current


def _set_nested(value: dict[str, Any], dotted: str, replacement: Any) -> None:
    current: Any = value
    parts = dotted.split(".")
    for part in parts[:-1]:
        match = re.fullmatch(r"([A-Za-z_]\w*)(?:\[(\d+)\])?", part)
        if match is None or not isinstance(current, dict):
            raise DINOLocalExecutionError(
                "invalid_action_spec",
                f"invalid DINO spec path {dotted!r}",
            )
        current = current[match.group(1)]
        if match.group(2) is not None:
            current = current[int(match.group(2))]
    last = re.fullmatch(r"([A-Za-z_]\w*)(?:\[(\d+)\])?", parts[-1])
    if last is None or not isinstance(current, dict):
        raise DINOLocalExecutionError(
            "invalid_action_spec",
            f"invalid DINO spec path {dotted!r}",
        )
    if last.group(2) is None:
        current[last.group(1)] = replacement
    else:
        current[last.group(1)][int(last.group(2))] = replacement


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise DINOLocalExecutionError(
                "artifact_drift",
                f"immutable artifact drift at {path.name!r}",
            )
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(content)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != content:
            raise DINOLocalExecutionError(
                "artifact_drift",
                f"immutable artifact drift at {path.name!r}",
            )
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_immutable(path: Path, value: Any) -> None:
    _write_immutable(path, _canonical_bytes(value))


def _read_status_jsonl(path: Path, action: str) -> tuple[Mapping[str, Any], ...]:
    if not path.is_file():
        raise DINOLocalExecutionError(
            "missing_status_artifact",
            f"{action} status artifact is missing",
        )
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DINOLocalExecutionError(
                "invalid_status_artifact",
                f"{action} status artifact has invalid JSON at line {line_number}",
            ) from exc
        if not isinstance(value, dict):
            raise DINOLocalExecutionError(
                "invalid_status_artifact",
                f"{action} status record must be an object",
            )
        records.append(MappingProxyType(value))
    expected = f"{action.capitalize()} finished successfully."
    if not records or not any(item.get("message") == expected for item in records):
        raise DINOLocalExecutionError(
            "action_status_incomplete",
            f"{action} did not emit its exact success record",
        )
    if any(item.get("status") == "FAILURE" for item in records):
        raise DINOLocalExecutionError(
            "action_status_failed",
            f"{action} status artifact contains a failure record",
        )
    return tuple(records)


def _metric_from_records(
    records: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> float:
    values = []
    for record in records:
        kpi = record.get("kpi")
        if not isinstance(kpi, Mapping):
            continue
        for key in keys:
            if key not in kpi:
                continue
            raw = kpi[key]
            if isinstance(raw, bool):
                continue
            try:
                metric = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(metric):
                values.append(metric)
    if not values:
        raise DINOLocalExecutionError(
            "missing_metric",
            "DINO status artifact does not contain a finite mAP50 metric",
        )
    result = values[-1]
    if not 0.0 <= result <= 1.0:
        raise DINOLocalExecutionError(
            "invalid_metric",
            "DINO mAP50 metric is outside [0, 1]",
        )
    return result


class DINOLocalDockerExecutor:
    """Concrete action executor for a live, typed DINO preflight plan."""

    def __init__(
        self,
        *,
        plan: DINOPreflightCommandPlan,
        config: DINOLocalExecutorConfig,
        hooks: DINOLocalExecutorHooks,
        sdk: Any | None = None,
        entrypoint_builder: Callable[..., Mapping[str, Any]] | None = None,
        process_runner: Callable[..., Any] = subprocess.run,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if not isinstance(plan, DINOPreflightCommandPlan):
            raise TypeError("plan must be DINOPreflightCommandPlan")
        if not isinstance(config, DINOLocalExecutorConfig):
            raise TypeError("config must be DINOLocalExecutorConfig")
        if not isinstance(hooks, DINOLocalExecutorHooks):
            raise TypeError("hooks must be DINOLocalExecutorHooks")
        plan.validate()
        if config.plan_sha256 != plan.plan_sha256:
            raise DINOLocalExecutionError(
                "plan_mismatch",
                "external executor configuration targets a different plan",
            )
        image_match = _DIGEST_IMAGE_RE.fullmatch(config.image)
        assert image_match is not None
        runtime_image = plan.settings.runtime_image_contract.runtime_image
        if (
            config.image != runtime_image
            or image_match.group("digest") != plan.settings.container_sha256
        ):
            raise DINOLocalExecutionError(
                "image_contract_mismatch",
                "pinned image does not match the reviewed TAO 7.1 runtime "
                "mapping and digest",
            )
        missing_env = [
            name for name in config.required_environment if not os.environ.get(name)
        ]
        if missing_env:
            raise DINOLocalExecutionError(
                "missing_environment",
                "one or more required environment variables are unavailable",
            )
        self.plan = plan
        self.config = config
        self.hooks = hooks
        self._sdk = sdk
        self._entrypoint_builder = entrypoint_builder
        self._process_runner = process_runner
        self._sleeper = sleeper
        self._prepared = False
        self._results: dict[str, DINOPreflightExecutionResult] = {}
        self._outcomes: dict[str, _ActionOutcome] = {}
        self._artifacts: dict[str, _Artifact] = {}
        self._audit_root = (
            config.results_root
            / ".dino-preflight"
            / plan.plan_sha256
        )
        self._inline_paths: dict[str, Path] = {}

    def _create_sdk(self) -> Any:
        previous = os.environ.get("DOCKER_PULL_POLICY")
        os.environ["DOCKER_PULL_POLICY"] = "never"
        try:
            from tao_sdk.platforms.docker import DockerSDK

            state_file = self._audit_root / "docker_sdk_state.json"
            return DockerSDK(
                poll_interval=max(1, int(self.config.poll_interval_seconds)),
                state_file=state_file,
            )
        finally:
            if previous is None:
                os.environ.pop("DOCKER_PULL_POLICY", None)
            else:
                os.environ["DOCKER_PULL_POLICY"] = previous

    def _builder(self) -> Callable[..., Mapping[str, Any]]:
        if self._entrypoint_builder is None:
            from tao_sdk.script_runner import build_entrypoint

            self._entrypoint_builder = build_entrypoint
        return self._entrypoint_builder

    def _inspect_image(self) -> None:
        result = self._process_runner(
            ["docker", "image", "inspect", self.config.image],
            check=False,
            capture_output=True,
            text=True,
        )
        if getattr(result, "returncode", 1) != 0:
            raise DINOLocalExecutionError(
                "local_image_missing",
                "pinned image is not present on the local Docker daemon",
            )
        try:
            values = json.loads(result.stdout)
        except (AttributeError, json.JSONDecodeError) as exc:
            raise DINOLocalExecutionError(
                "invalid_image_inspection",
                "Docker returned invalid local image inspection data",
            ) from exc
        if not isinstance(values, list) or len(values) != 1:
            raise DINOLocalExecutionError(
                "invalid_image_inspection",
                "Docker image inspection must resolve exactly one local image",
            )
        image_match = _DIGEST_IMAGE_RE.fullmatch(self.config.image)
        assert image_match is not None
        expected = (
            f"{_repository_without_tag(image_match.group('base'))}"
            f"@sha256:{image_match.group('digest')}"
        )
        digests = values[0].get("RepoDigests", [])
        if not isinstance(digests, list) or expected not in digests:
            raise DINOLocalExecutionError(
                "image_digest_mismatch",
                "local image does not expose the frozen repository digest",
            )

    def _container_path(self, host_path: Path) -> str:
        host = host_path.resolve(strict=False)
        matches = []
        for mount in self.config.mounts:
            try:
                relative = host.relative_to(mount.host_path)
            except ValueError:
                continue
            matches.append((len(mount.host_path.parts), mount, relative))
        if not matches:
            raise DINOLocalExecutionError(
                "undeclared_mount",
                f"input path {host.name!r} is outside the declared Docker binds",
            )
        _, mount, relative = max(matches, key=lambda item: item[0])
        return str(PurePosixPath(mount.container_path) / PurePosixPath(relative.as_posix()))

    def _copy_immutable(self, source: Path, destination: Path) -> None:
        if not source.is_file():
            raise DINOLocalExecutionError(
                "missing_input_artifact",
                f"input image {source.name!r} is missing",
            )
        content_hash = _sha256_file(source)
        if destination.exists():
            if (
                not destination.is_file()
                or _sha256_file(destination) != content_hash
            ):
                raise DINOLocalExecutionError(
                    "artifact_drift",
                    f"materialized input {destination.name!r} changed",
                )
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}"
        )
        shutil.copyfile(source, temporary)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if _sha256_file(destination) != content_hash:
                raise DINOLocalExecutionError(
                    "artifact_drift",
                    f"materialized input {destination.name!r} changed",
                )
        finally:
            temporary.unlink(missing_ok=True)

    def _materialize_inline_artifacts(self) -> None:
        artifacts = self.plan.to_dict()["inline_artifacts"]
        inputs = self._audit_root / "inputs"
        classmap = inputs / "label_map.txt"
        classmap_record = artifacts["voc_label_map"]
        content = classmap_record["content"].encode("utf-8")
        if hashlib.sha256(content).hexdigest() != classmap_record["sha256"]:
            raise DINOLocalExecutionError(
                "plan_integrity_failure",
                "inline class map checksum is invalid",
            )
        _write_immutable(classmap, content)

        subset_record = artifacts["voc_inference_subset"]
        source_root = Path(subset_record["source_image_root"])
        full_subset = inputs / "inference_subset"
        smoke_subset = inputs / "smoke_inference_subset"
        for index, entry in enumerate(subset_record["entries"]):
            source = source_root / entry["file_name"]
            self._copy_immutable(source, full_subset / entry["file_name"])
            if index == 0:
                self._copy_immutable(source, smoke_subset / entry["file_name"])

        validation = json.loads(
            self.plan.voc_integrity.validation_annotation_path.read_text(
                encoding="utf-8"
            )
        )
        ordered_images = sorted(
            validation["images"],
            key=lambda item: (item["id"], item["file_name"]),
        )
        selected = ordered_images[: self.plan.settings.batch_size]
        selected_ids = {item["id"] for item in selected}
        mini_validation = {
            key: copy.deepcopy(value)
            for key, value in validation.items()
            if key not in {"images", "annotations"}
        }
        mini_validation["images"] = selected
        mini_validation["annotations"] = [
            item
            for item in validation["annotations"]
            if item["image_id"] in selected_ids
        ]
        mini_path = inputs / "smoke_validation.json"
        _write_json_immutable(mini_path, mini_validation)

        resume_path = self._audit_root / "resume_replay_state.json"
        resume_command = self.plan.commands_for_stage(
            "interrupted_resume_replay"
        )[0]
        _write_json_immutable(
            resume_path,
            {
                "schema_version": 1,
                "plan_sha256": self.plan.plan_sha256,
                "command_sha256": resume_command.sha256,
                "workspace_identity_sha256": resume_command.metadata[
                    "workspace_identity_sha256"
                ],
                "state": "preregistered_for_interrupted_resume_replay",
            },
        )
        self._artifacts["resume_replay_state"] = _Artifact.from_file(
            "resume_replay_state", resume_path
        )
        smoke_worker_source = Path(__file__).with_name(
            "dino_ptm_smoke_worker.py"
        )
        smoke_worker = self._audit_root / "dino_ptm_smoke_worker.py"
        if not smoke_worker_source.is_file():
            raise DINOLocalExecutionError(
                "missing_smoke_worker",
                "packaged DINO backbone smoke worker is missing",
            )
        _write_immutable(smoke_worker, smoke_worker_source.read_bytes())
        latency_worker_source = Path(__file__).with_name(
            "dino_latency_worker.py"
        )
        latency_worker = self._audit_root / "dino_latency_worker.py"
        if not latency_worker_source.is_file():
            raise DINOLocalExecutionError(
                "missing_latency_worker",
                "packaged DINO latency worker is missing",
            )
        _write_immutable(latency_worker, latency_worker_source.read_bytes())
        runtime_modules_root = self._audit_root / "runtime_modules"
        package_root = runtime_modules_root / "tao_automl"
        _write_immutable(package_root / "__init__.py", b"")
        for module, destination in (
            (_latency_benchmark_module, "latency_benchmark.py"),
            (_latency_stats_module, "latency_stats.py"),
        ):
            source_path = Path(module.__file__).resolve(strict=True)
            _write_immutable(package_root / destination, source_path.read_bytes())
        latency_contract_path = inputs / "latency_contract.json"
        _write_json_immutable(
            latency_contract_path,
            self.plan.latency_contract.to_dict(),
        )
        latency_descriptor_path = inputs / "latency_input_descriptor.json"
        _write_json_immutable(
            latency_descriptor_path,
            _json_copy(self.plan.settings.latency_input_descriptor),
        )
        self._inline_paths = {
            "artifact://voc2007/label_map.txt": classmap,
            "artifact://voc2007/inference_subset": full_subset,
            "smoke_inference_subset": smoke_subset,
            "smoke_validation": mini_path,
            "resume_replay_state": resume_path,
            "backbone_smoke_worker": smoke_worker,
            "latency_worker": latency_worker,
            "latency_runtime_modules": runtime_modules_root,
            "latency_contract": latency_contract_path,
            "latency_input_descriptor": latency_descriptor_path,
        }

    def _prepare(self) -> None:
        if self._prepared:
            return
        self.plan.validate()
        self.plan.voc_integrity.validate_current_files()
        self.config.results_root.mkdir(parents=True, exist_ok=True)
        self._inspect_image()
        self._materialize_inline_artifacts()
        if self._sdk is None:
            self._sdk = self._create_sdk()
        self._prepared = True

    def _resolve_input_value(
        self,
        value: Any,
        bindings: Mapping[str, Path],
    ) -> Any:
        if isinstance(value, str):
            if value in bindings:
                value = str(bindings[value])
            if value.startswith(("artifact://", "runtime://")):
                raise DINOLocalExecutionError(
                    "unresolved_artifact",
                    "action spec contains an unresolved local artifact token",
                )
            if value.startswith("/"):
                return self._container_path(Path(value))
            return value
        if isinstance(value, list):
            return [
                self._resolve_input_value(item, bindings)
                for item in value
            ]
        if isinstance(value, Mapping):
            return {
                str(key): self._resolve_input_value(item, bindings)
                for key, item in value.items()
            }
        return value

    def _resolved_spec(
        self,
        *,
        action: str,
        spec: Mapping[str, Any],
        bindings: Mapping[str, Path],
        smoke: bool,
    ) -> dict[str, Any]:
        result = self._resolve_input_value(_json_copy(spec), bindings)
        if smoke and action == "evaluate":
            _set_nested(
                result,
                "dataset.test_data_sources.json_file",
                self._container_path(self._inline_paths["smoke_validation"]),
            )
        if smoke and action == "inference":
            _set_nested(
                result,
                "dataset.infer_data_sources.image_dir",
                [
                    self._container_path(
                        self._inline_paths["smoke_inference_subset"]
                    )
                ],
            )
        return result

    def _poll(self, job_id: str) -> None:
        for attempt in range(self.config.max_polls):
            status = self._sdk.get_job_status(job_id)
            value = getattr(status, "status", None)
            if value in _TERMINAL_JOB_STATES:
                if value != "Complete":
                    raise DINOLocalExecutionError(
                        "tao_action_failed",
                        f"TAO Docker job {job_id} reached terminal state {value}",
                    )
                return
            if attempt + 1 < self.config.max_polls:
                self._sleeper(float(self.config.poll_interval_seconds))
        raise DINOLocalExecutionError(
            "tao_action_timeout",
            f"TAO Docker job {job_id} exceeded the frozen polling budget",
        )

    def _parse_action(
        self,
        *,
        action: str,
        job_id: str,
        job_root: Path,
        require_metric: bool,
        require_checkpoint: bool,
    ) -> _ActionOutcome:
        status = job_root / "results_dir" / action / "status.json"
        records = _read_status_jsonl(status, action)
        metric = None
        if require_metric:
            keys = (
                ("val_mAP50",)
                if action == "train"
                else ("test_mAP50", "val_mAP50")
            )
            metric = _metric_from_records(records, keys)
        checkpoint = None
        epochs = None
        steps = None
        if require_checkpoint:
            progress = [
                item
                for item in records
                if isinstance(item.get("epoch"), int)
                and isinstance(item.get("step"), int)
            ]
            if not progress:
                raise DINOLocalExecutionError(
                    "missing_training_progress",
                    "full-epoch train status has no progress record",
                )
            final = max(progress, key=lambda item: (item["epoch"], item["step"]))
            epochs = int(final["epoch"]) + 1
            steps = int(final["step"])
            matches = []
            for candidate in (job_root / "results_dir" / "train").glob(
                "model_epoch_*_step_*.*"
            ):
                match = _CHECKPOINT_RE.fullmatch(candidate.name)
                if match is not None:
                    matches.append(
                        (
                            int(match.group("epoch")),
                            int(match.group("step")),
                            candidate,
                        )
                    )
            expected = [
                item
                for item in matches
                if item[:2] == (int(final["epoch"]), steps)
            ]
            if len(expected) != 1 or expected[0][2].is_symlink():
                raise DINOLocalExecutionError(
                    "ambiguous_checkpoint",
                    "full-epoch train did not emit one exact final checkpoint",
                )
            checkpoint = expected[0][2].resolve(strict=True)
        return _ActionOutcome(
            action=action,
            job_id=job_id,
            job_root=job_root,
            status_path=status,
            records=records,
            metric=metric,
            checkpoint_path=checkpoint,
            completed_epochs=epochs,
            training_steps=steps,
        )

    def _run_action(
        self,
        *,
        owner: DINOPreflightCommand,
        action: str,
        spec: Mapping[str, Any],
        bindings: Mapping[str, Path] | None = None,
        smoke: bool = False,
        dry_train: bool = False,
        require_metric: bool = False,
        require_checkpoint: bool = False,
    ) -> _ActionOutcome:
        contract = _json_copy(self.plan.skill_contract.actions[action])
        if contract.get("mode") != "config" or contract.get(
            "config_format"
        ) != "yaml":
            raise DINOLocalExecutionError(
                "skill_contract_drift",
                f"DINO {action} must remain a YAML config action",
            )
        token_bindings = dict(self._inline_paths)
        token_bindings.update(bindings or {})
        resolved = self._resolved_spec(
            action=action,
            spec=spec,
            bindings=token_bindings,
            smoke=smoke,
        )
        if dry_train:
            resolved.setdefault("train", {})["is_dry_run"] = True
        entrypoint = self._builder()(
            command=contract["command"],
            specs=resolved,
            inputs=contract["inputs"],
            outputs=contract["outputs"],
            config_format=contract["config_format"],
            upload_excludes=contract.get("upload_excludes", []),
        )
        if not isinstance(entrypoint, Mapping) or not isinstance(
            entrypoint.get("command"), str
        ):
            raise DINOLocalExecutionError(
                "invalid_entrypoint",
                "TAO SDK build_entrypoint returned an invalid command",
            )
        job = self._sdk.create_job(
            image=self.config.image,
            command=entrypoint["command"],
            gpu_count=1,
            env_vars={
                "TAO_PREFLIGHT_COMMAND_ID": owner.command_id,
                "TAO_PREFLIGHT_ACTION": action,
            },
            mounts=[item.to_sdk_dict() for item in self.config.mounts],
            shm_size=self.config.shm_size,
            run_as_user=True,
            container_user=self.config.container_user,
        )
        job_id = getattr(job, "id", None)
        if not isinstance(job_id, str) or not job_id:
            raise DINOLocalExecutionError(
                "invalid_job_handle",
                "DockerSDK returned an invalid job handle",
            )
        self._poll(job_id)
        result_value = self._sdk.get_job_results_dir(job_id)
        if not result_value:
            result_value = getattr(job, "results_dir", "")
        job_root = _require_absolute(result_value, "job results")
        return self._parse_action(
            action=action,
            job_id=job_id,
            job_root=job_root,
            require_metric=require_metric,
            require_checkpoint=require_checkpoint,
        )

    def _ptm(self, ptm_id: str) -> Any:
        return next(
            item
            for item in self.plan.model_preflight_inputs.eligible_ptms
            if item.id == ptm_id
        )

    def _dataset_result(
        self, command: DINOPreflightCommand
    ) -> DINOPreflightExecutionResult:
        voc = self.plan.voc_integrity
        inputs = self.plan.model_preflight_inputs
        return DINOPreflightExecutionResult(
            command_id=command.command_id,
            passed=True,
            evidence={
                "dataset_id": inputs.dataset_id,
                "manifest_sha256": inputs.dataset_manifest_sha256,
                "annotation_contract_sha256": inputs.annotation_contract_sha256,
                "annotations_valid": True,
                "train_split_sha256": inputs.train_split_sha256,
                "validation_split_sha256": inputs.validation_split_sha256,
                "train_samples": voc.train_samples,
                "validation_samples": voc.validation_samples,
            },
        )

    def _default_load(
        self, command: DINOPreflightCommand
    ) -> DINOPreflightExecutionResult:
        outcome = self._run_action(
            owner=command,
            action="train",
            spec=command.specs_by_action["train"],
            dry_train=True,
        )
        self._outcomes[command.command_id] = outcome
        ptm = self._ptm(command.ptm_id)
        return DINOPreflightExecutionResult(
            command_id=command.command_id,
            passed=True,
            evidence={
                "ptm_id": ptm.id,
                "checkpoint_sha256": ptm.checkpoint_sha256,
                "loaded": True,
                "input_contract_verified": True,
                "spec_merge_verified": True,
            },
        )

    def _ptm_smoke(
        self, command: DINOPreflightCommand
    ) -> DINOPreflightExecutionResult:
        if (
            command.metadata["checkpoint_target"]
            == "model.pretrained_backbone_path"
        ):
            return self._backbone_ptm_smoke(command)
        if command.metadata["checkpoint_target"] != "train.pretrained_model_path":
            raise DINOLocalExecutionError(
                "invalid_checkpoint_target",
                "DINO PTM uses an unsupported checkpoint target",
            )
        checkpoint = Path(command.metadata["checkpoint_path"])
        runtime_token = command.metadata["initialized_model_binding"]
        bindings = {runtime_token: checkpoint}
        train = self._run_action(
            owner=command,
            action="train",
            spec=command.specs_by_action["train"],
            dry_train=True,
            smoke=True,
        )
        self._run_action(
            owner=command,
            action="evaluate",
            spec=command.specs_by_action["evaluate"],
            bindings=bindings,
            smoke=True,
        )
        self._run_action(
            owner=command,
            action="inference",
            spec=command.specs_by_action["inference"],
            bindings=bindings,
            smoke=True,
        )
        self._outcomes[command.command_id] = train
        ptm = self._ptm(command.ptm_id)
        return DINOPreflightExecutionResult(
            command_id=command.command_id,
            passed=True,
            evidence={
                "ptm_id": ptm.id,
                "checkpoint_sha256": ptm.checkpoint_sha256,
                "loaded": True,
                "train_step_passed": True,
                "validation_step_passed": True,
                "inference_step_passed": True,
            },
        )

    def _backbone_ptm_smoke(
        self, command: DINOPreflightCommand
    ) -> DINOPreflightExecutionResult:
        """Run the mounted one-process TAO adapter for a backbone-only PTM."""
        contract = _json_copy(self.plan.skill_contract.actions["train"])
        spec = self._resolved_spec(
            action="train",
            spec=command.specs_by_action["train"],
            bindings=self._inline_paths,
            smoke=True,
        )
        worker = self._container_path(
            self._inline_paths["backbone_smoke_worker"]
        )
        checkpoint_sha256 = command.metadata["checkpoint_sha256"]
        custom_command = " ".join(
            (
                "python",
                shlex.quote(worker),
                "--config",
                "{config_path}",
                "--ptm-id",
                shlex.quote(command.ptm_id),
                "--checkpoint-sha256",
                shlex.quote(checkpoint_sha256),
            )
        )
        entrypoint = self._builder()(
            command=custom_command,
            specs=spec,
            inputs=contract["inputs"],
            outputs=contract["outputs"],
            config_format=contract["config_format"],
            upload_excludes=contract.get("upload_excludes", []),
        )
        if not isinstance(entrypoint, Mapping) or not isinstance(
            entrypoint.get("command"), str
        ):
            raise DINOLocalExecutionError(
                "invalid_entrypoint",
                "TAO SDK build_entrypoint returned an invalid smoke command",
            )
        job = self._sdk.create_job(
            image=self.config.image,
            command=entrypoint["command"],
            gpu_count=1,
            env_vars={
                "TAO_PREFLIGHT_COMMAND_ID": command.command_id,
                "TAO_PREFLIGHT_ACTION": "backbone_ptm_smoke",
            },
            mounts=[item.to_sdk_dict() for item in self.config.mounts],
            shm_size=self.config.shm_size,
            run_as_user=True,
            container_user=self.config.container_user,
        )
        job_id = getattr(job, "id", None)
        if not isinstance(job_id, str) or not job_id:
            raise DINOLocalExecutionError(
                "invalid_job_handle",
                "DockerSDK returned an invalid backbone-smoke job handle",
            )
        self._poll(job_id)
        result_value = self._sdk.get_job_results_dir(job_id) or getattr(
            job, "results_dir", ""
        )
        job_root = _require_absolute(result_value, "job results")
        evidence_path = (
            job_root / "results_dir" / "ptm_smoke_evidence.json"
        )
        if not evidence_path.is_file():
            raise DINOLocalExecutionError(
                "missing_smoke_evidence",
                "backbone PTM smoke did not emit structured evidence",
            )
        try:
            raw = json.loads(evidence_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DINOLocalExecutionError(
                "invalid_smoke_evidence",
                "backbone PTM smoke evidence is invalid JSON",
            ) from exc
        expected_keys = {
            "schema_version",
            "ptm_id",
            "checkpoint_sha256",
            "checkpoint_target",
            "device",
            "loaded",
            "real_data",
            "train",
            "validation",
            "inference",
        }
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise DINOLocalExecutionError(
                "invalid_smoke_evidence",
                "backbone PTM smoke evidence has an unexpected shape",
            )
        if (
            raw["schema_version"] != 1
            or raw["ptm_id"] != command.ptm_id
            or raw["checkpoint_sha256"] != checkpoint_sha256
            or raw["checkpoint_target"]
            != "model.pretrained_backbone_path"
            or raw["device"] != "cuda:0"
            or raw["loaded"] is not True
            or raw["real_data"] is not True
        ):
            raise DINOLocalExecutionError(
                "invalid_smoke_evidence",
                "backbone PTM smoke identity or runtime evidence mismatched",
            )
        for stage in ("train", "validation"):
            item = raw[stage]
            if (
                not isinstance(item, dict)
                or set(item) != {"batches", "finite", "loss"}
                or item["batches"] != 1
                or item["finite"] is not True
                or isinstance(item["loss"], bool)
                or not isinstance(item["loss"], (int, float))
                or not math.isfinite(float(item["loss"]))
            ):
                raise DINOLocalExecutionError(
                    "invalid_smoke_evidence",
                    f"backbone PTM {stage} evidence is not finite one-batch evidence",
                )
        inference = raw["inference"]
        if (
            not isinstance(inference, dict)
            or set(inference)
            != {"batches", "finite", "output_tensor_count"}
            or inference["batches"] != 1
            or inference["finite"] is not True
            or isinstance(inference["output_tensor_count"], bool)
            or not isinstance(inference["output_tensor_count"], int)
            or inference["output_tensor_count"] < 1
        ):
            raise DINOLocalExecutionError(
                "invalid_smoke_evidence",
                "backbone PTM inference evidence is not finite one-batch evidence",
            )
        self._outcomes[command.command_id] = _ActionOutcome(
            action="backbone_ptm_smoke",
            job_id=job_id,
            job_root=job_root,
            status_path=evidence_path,
            records=(MappingProxyType(raw),),
        )
        ptm = self._ptm(command.ptm_id)
        return DINOPreflightExecutionResult(
            command_id=command.command_id,
            passed=True,
            evidence={
                "ptm_id": ptm.id,
                "checkpoint_sha256": ptm.checkpoint_sha256,
                "loaded": True,
                "train_step_passed": True,
                "validation_step_passed": True,
                "inference_step_passed": True,
            },
        )

    def _full_epoch(
        self, command: DINOPreflightCommand
    ) -> DINOPreflightExecutionResult:
        outcome = self._run_action(
            owner=command,
            action="train",
            spec=command.specs_by_action["train"],
            require_metric=True,
            require_checkpoint=True,
        )
        assert outcome.checkpoint_path is not None
        assert outcome.completed_epochs is not None
        assert outcome.training_steps is not None
        if outcome.completed_epochs != command.metadata["complete_epochs"]:
            raise DINOLocalExecutionError(
                "training_fidelity_mismatch",
                "TAO train did not complete the frozen one-epoch fidelity",
            )
        artifact = _Artifact.from_file(
            "full_epoch_checkpoint", outcome.checkpoint_path
        )
        self._artifacts[artifact.artifact_id] = artifact
        self._outcomes[command.command_id] = outcome
        return DINOPreflightExecutionResult(
            command_id=command.command_id,
            passed=True,
            evidence={
                "ptm_id": command.ptm_id,
                "single_gpu": True,
                "completed": True,
                "completed_epochs": outcome.completed_epochs,
                "training_batches": outcome.training_steps,
                "distinct_training_steps": outcome.training_steps,
                "final_checkpoint_sha256": artifact.sha256,
            },
        )

    def _in_epoch_validation(
        self, command: DINOPreflightCommand
    ) -> DINOPreflightExecutionResult:
        outcome = self._outcomes["default_model_full_epoch"]
        assert outcome.metric is not None
        path = self._audit_root / "in_epoch_validation_metrics.json"
        evidence = {
            "metric_name": self.plan.model_preflight_inputs.metric_name,
            "metric_value": outcome.metric,
            "completed_evaluations": 1,
            "passed": True,
        }
        _write_json_immutable(path, evidence)
        self._artifacts["in_epoch_validation_metrics"] = _Artifact.from_file(
            "in_epoch_validation_metrics", path
        )
        return DINOPreflightExecutionResult(
            command_id=command.command_id,
            passed=True,
            evidence=evidence,
        )

    def _standalone_evaluation(
        self, command: DINOPreflightCommand
    ) -> DINOPreflightExecutionResult:
        checkpoint = self._artifacts["full_epoch_checkpoint"].path
        outcome = self._run_action(
            owner=command,
            action="evaluate",
            spec=command.specs_by_action["evaluate"],
            bindings={command.metadata["checkpoint_binding"]: checkpoint},
            require_metric=True,
        )
        assert outcome.metric is not None
        self._outcomes[command.command_id] = outcome
        evidence = {
            "metric_name": self.plan.model_preflight_inputs.metric_name,
            "metric_value": outcome.metric,
            "completed_evaluations": 1,
            "passed": True,
            "runtime_metric_contract_verified": True,
        }
        path = self._audit_root / "standalone_evaluation_metrics.json"
        _write_json_immutable(path, evidence)
        self._artifacts[
            "standalone_evaluation_metrics"
        ] = _Artifact.from_file("standalone_evaluation_metrics", path)
        return DINOPreflightExecutionResult(
            command_id=command.command_id,
            passed=True,
            evidence=evidence,
        )

    def _checkpoint_reload(
        self, command: DINOPreflightCommand
    ) -> DINOPreflightExecutionResult:
        checkpoint = self._artifacts["full_epoch_checkpoint"]
        inference_command = self.plan.commands_for_stage(
            "latency_instrumentation"
        )[0]
        self._run_action(
            owner=command,
            action="inference",
            spec=inference_command.specs_by_action["inference"],
            bindings={command.metadata["checkpoint_binding"]: checkpoint.path},
            smoke=True,
        )
        if _sha256_file(checkpoint.path) != checkpoint.sha256:
            raise DINOLocalExecutionError(
                "checkpoint_drift",
                "checkpoint changed during reload verification",
            )
        return DINOPreflightExecutionResult(
            command_id=command.command_id,
            passed=True,
            evidence={
                "ptm_id": command.ptm_id,
                "saved": True,
                "reloaded": True,
                "saved_checkpoint_sha256": checkpoint.sha256,
                "reloaded_checkpoint_sha256": checkpoint.sha256,
            },
        )

    def _latency(
        self, command: DINOPreflightCommand
    ) -> DINOPreflightExecutionResult:
        checkpoint = self._artifacts["full_epoch_checkpoint"].path
        inference_spec = _json_copy(command.specs_by_action["inference"])
        _set_nested(
            inference_spec,
            "inference.checkpoint",
            str(checkpoint),
        )
        runtime = self.hooks.latency_runtime_factory(
            plan=self.plan,
            command=command,
            checkpoint_path=checkpoint,
            inference_spec=inference_spec,
        )
        if isinstance(runtime, ContainerLatencyRuntime):
            record = self._run_container_latency_worker(
                command=command,
                checkpoint_path=checkpoint,
                inference_spec=inference_spec,
            )
        elif isinstance(runtime, LatencyRuntime):
            record = run_replica_benchmark(
                contract=self.plan.latency_contract,
                identity=ReplicaIdentity(
                    rank=0,
                    world_size=1,
                    device_id="cuda:0",
                    hardware_sha256=runtime.hardware_sha256,
                ),
                candidate_fingerprint=command.metadata[
                    "candidate_fingerprint"
                ],
                step=runtime.step,
                synchronize=runtime.synchronize,
                clock_ns=runtime.clock_ns,
            )
        else:
            raise DINOLocalExecutionError(
                "invalid_latency_runtime",
                "latency runtime factory returned the wrong type",
            )
        path = self._audit_root / "latency_replica_0.json"
        _write_json_immutable(path, record)
        self._artifacts["latency_aggregate"] = _Artifact.from_file(
            "latency_aggregate", path
        )
        return DINOPreflightExecutionResult(
            command_id=command.command_id,
            passed=True,
            evidence={"replica_records": [record]},
        )

    def _run_container_latency_worker(
        self,
        *,
        command: DINOPreflightCommand,
        checkpoint_path: Path,
        inference_spec: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Run the fixed model-forward worker through the existing DockerSDK."""
        contract = _json_copy(self.plan.skill_contract.actions["inference"])
        worker = self._container_path(self._inline_paths["latency_worker"])
        checkpoint = self._container_path(checkpoint_path)
        contract_path = self._container_path(
            self._inline_paths["latency_contract"]
        )
        descriptor = self._container_path(
            self._inline_paths["latency_input_descriptor"]
        )
        modules = self._container_path(
            self._inline_paths["latency_runtime_modules"]
        )
        output_host = self._audit_root / "latency_worker_output.json"
        output_container = self._container_path(output_host)
        raw_command = " ".join(
            (
                "python3",
                shlex.quote(worker),
                "--config",
                "{config_path}",
                "--checkpoint",
                shlex.quote(checkpoint),
                "--contract",
                shlex.quote(contract_path),
                "--input-descriptor",
                shlex.quote(descriptor),
                "--candidate-fingerprint",
                shlex.quote(command.metadata["candidate_fingerprint"]),
                "--runtime-modules-root",
                shlex.quote(modules),
            )
        )
        entrypoint = self._builder()(
            command=raw_command,
            specs=_json_copy(inference_spec),
            inputs=contract["inputs"],
            outputs=contract["outputs"],
            config_format="yaml",
            upload_excludes=contract.get("upload_excludes", []),
        )
        if not isinstance(entrypoint, Mapping) or not isinstance(
            entrypoint.get("command"), str
        ):
            raise DINOLocalExecutionError(
                "invalid_entrypoint",
                "TAO SDK build_entrypoint returned an invalid latency command",
            )
        job = self._sdk.create_job(
            image=self.config.image,
            command=entrypoint["command"],
            gpu_count=1,
            env_vars={
                "TAO_PREFLIGHT_COMMAND_ID": command.command_id,
                "TAO_PREFLIGHT_ACTION": "dino_model_forward_latency",
                "TAO_DINO_LATENCY_OUTPUT": output_container,
            },
            mounts=[item.to_sdk_dict() for item in self.config.mounts],
            shm_size=self.config.shm_size,
            run_as_user=True,
            container_user=self.config.container_user,
        )
        job_id = getattr(job, "id", None)
        if not isinstance(job_id, str) or not job_id:
            raise DINOLocalExecutionError(
                "invalid_job_handle",
                "DockerSDK returned an invalid latency job handle",
            )
        self._poll(job_id)
        if not output_host.is_file() or output_host.is_symlink():
            raise DINOLocalExecutionError(
                "missing_latency_record",
                "DINO latency worker did not emit its raw replica record",
            )
        try:
            record = json.loads(output_host.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DINOLocalExecutionError(
                "invalid_latency_record",
                "DINO latency worker emitted invalid JSON",
            ) from exc
        try:
            aggregate = self.plan.latency_contract
            from tao_automl.latency_benchmark import combine_replica_records

            combined = combine_replica_records([record])
        except (TypeError, ValueError) as exc:
            raise DINOLocalExecutionError(
                "invalid_latency_record",
                "DINO latency worker record failed the production contract",
            ) from exc
        if (
            combined["contract_sha256"] != aggregate.sha256
            or combined["candidate_fingerprint"]
            != command.metadata["candidate_fingerprint"]
        ):
            raise DINOLocalExecutionError(
                "invalid_latency_record",
                "DINO latency worker record identity mismatched the plan",
            )
        return record

    def _output_validation(
        self, command: DINOPreflightCommand
    ) -> DINOPreflightExecutionResult:
        missing = [item for item in _RESULT_IDS if item not in self._artifacts]
        if missing:
            raise DINOLocalExecutionError(
                "missing_artifact",
                "one or more required preflight artifacts are missing",
            )
        for artifact in self._artifacts.values():
            if (
                not artifact.path.is_file()
                or artifact.path.stat().st_size != artifact.size_bytes
                or _sha256_file(artifact.path) != artifact.sha256
            ):
                raise DINOLocalExecutionError(
                    "artifact_drift",
                    "a preflight artifact changed after creation",
                )
        return DINOPreflightExecutionResult(
            command_id=command.command_id,
            passed=True,
            evidence={
                "contract_sha256": (
                    self.plan.model_preflight_inputs.output_contract_sha256
                ),
                "artifacts": [
                    self._artifacts[item].evidence() for item in _RESULT_IDS
                ],
                "missing_artifact_ids": [],
                "valid": True,
            },
        )

    def _resume_replay(
        self, command: DINOPreflightCommand
    ) -> DINOPreflightExecutionResult:
        state = self._inline_paths["resume_replay_state"]
        evidence = _json_copy(
            self.hooks.resume_replay_runner(
                plan=self.plan,
                command=command,
                state_path=state,
            )
        )
        if evidence.get("state_sha256") != _sha256_file(state):
            raise DINOLocalExecutionError(
                "resume_state_mismatch",
                "resume replay did not use the frozen state artifact",
            )
        return DINOPreflightExecutionResult(
            command_id=command.command_id,
            passed=True,
            evidence=evidence,
        )

    def _dispatch(
        self, command: DINOPreflightCommand
    ) -> DINOPreflightExecutionResult:
        handlers = {
            "dataset_validation": self._dataset_result,
            "default_ptm_load": self._default_load,
            "eligible_ptm_smoke": self._ptm_smoke,
            "default_model_full_epoch": self._full_epoch,
            "in_epoch_validation": self._in_epoch_validation,
            "standalone_evaluation": self._standalone_evaluation,
            "checkpoint_save_reload": self._checkpoint_reload,
            "latency_instrumentation": self._latency,
            "output_artifact_validation": self._output_validation,
            "interrupted_resume_replay": self._resume_replay,
        }
        handler = handlers.get(command.stage)
        if handler is None:
            raise DINOLocalExecutionError(
                "unsupported_stage",
                f"unsupported DINO preflight stage {command.stage!r}",
            )
        return handler(command)

    def __call__(
        self, command: DINOPreflightCommand
    ) -> DINOPreflightExecutionResult:
        if not isinstance(command, DINOPreflightCommand):
            raise TypeError("command must be DINOPreflightCommand")
        expected = next(
            (
                item
                for item in self.plan.commands
                if item.command_id == command.command_id
            ),
            None,
        )
        if expected is None or expected.sha256 != command.sha256:
            raise DINOPreflightContractError(
                "executor command is outside the frozen DINO plan"
            )
        cached = self._results.get(command.command_id)
        if cached is not None:
            return cached
        try:
            self._prepare()
            missing = [
                item
                for item in command.depends_on
                if item not in self._results or not self._results[item].passed
            ]
            if missing:
                raise DINOLocalExecutionError(
                    "dependency_not_complete",
                    "a frozen command dependency has not completed",
                )
            result = self._dispatch(command)
        except DINOLocalExecutionError as exc:
            result = DINOPreflightExecutionResult(
                command_id=command.command_id,
                passed=False,
                code=exc.code,
            )
        self._results[command.command_id] = result
        return result


def _load_factory(reference: str) -> Callable[..., Any]:
    if not isinstance(reference, str) or reference.count(":") != 1:
        raise DINOLocalExecutionError(
            "invalid_factory",
            "factory must use module:callable syntax",
        )
    module_name, attribute = reference.split(":", 1)
    if not module_name or not attribute:
        raise DINOLocalExecutionError(
            "invalid_factory",
            "factory must use module:callable syntax",
        )
    factory = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(factory):
        raise DINOLocalExecutionError(
            "invalid_factory",
            "configured factory is not callable",
        )
    return factory


def _write_report(path: Path, report: Mapping[str, Any], *, resume: bool) -> None:
    content = _canonical_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if resume and path.read_bytes() == content:
            return
        raise FileExistsError(path)
    _write_immutable(path, content)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute a frozen DINO local preflight through DockerSDK."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan-factory", required=True)
    parser.add_argument(
        "--hooks-factory",
        default="dino_local_factories:build_default_hooks",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resume-report", type=Path)
    parser.add_argument("--stop-after-stage")
    args = parser.parse_args(argv)

    try:
        config = DINOLocalExecutorConfig.from_file(args.config)
        plan = _load_factory(args.plan_factory)()
        if not isinstance(plan, DINOPreflightCommandPlan):
            raise DINOLocalExecutionError(
                "invalid_factory",
                "plan factory did not return DINOPreflightCommandPlan",
            )
        hooks = _load_factory(args.hooks_factory)(plan, config)
        if not isinstance(hooks, DINOLocalExecutorHooks):
            raise DINOLocalExecutionError(
                "invalid_factory",
                "hooks factory did not return DINOLocalExecutorHooks",
            )
        resume = None
        if args.resume_report is not None:
            resume = json.loads(
                _require_absolute(
                    args.resume_report, "resume_report"
                ).read_text(encoding="utf-8")
            )
        executor = DINOLocalDockerExecutor(
            plan=plan,
            config=config,
            hooks=hooks,
        )
        report = run_dino_local_preflight(
            plan=plan,
            executor=executor,
            resume_report=resume,
            stop_after_stage=args.stop_after_stage,
        )
        report_path = _require_absolute(args.report, "report")
        _write_report(report_path, report, resume=resume is not None)
        print(
            json.dumps(
                {
                    "completion_state": report["completion_state"],
                    "plan_sha256": plan.plan_sha256,
                    "report": str(report_path),
                },
                sort_keys=True,
            )
        )
        return 0 if report["completion_state"] in {"completed", "interrupted"} else 1
    except (
        DINOLocalExecutionError,
        DINOPreflightContractError,
        FileExistsError,
        OSError,
        ValueError,
        TypeError,
    ) as exc:
        code = getattr(exc, "code", "local_preflight_failed")
        print(
            json.dumps(
                {
                    "completion_state": "failed",
                    "code": code,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DINOLocalDockerExecutor",
    "DINOLocalExecutionError",
    "DINOLocalExecutorConfig",
    "DINOLocalExecutorHooks",
    "DockerBind",
    "LatencyRuntime",
    "main",
]
