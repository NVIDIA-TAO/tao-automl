# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted DINO checkpoint metadata projection for TAO 7.1 qualification.

The production callback never deserializes the official checkpoint itself.
It first verifies the file against the size and SHA-256 supplied by
``PTMCheckpointPreflight``, then delegates the transformation to an execution
backend. The campaign default is an exact-digest TAO 7.1 Docker backend with
``--pull=never``; host PyTorch is available only as an explicitly injected
backend and is lazily imported.
"""

from __future__ import annotations

import argparse
import base64
import copy
import gc
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol


PINNED_TAO71_DOCKER_IMAGE = (
    "nvcr.io/nvstaging/tao/tao-toolkit-pyt@"
    "sha256:949c0ea8ace09ac91951be4169353cf214daaa3ede7db9eed94070b020361667"
)
TENSOR_HASH_ALGORITHM = "sha256_sorted_key_dtype_shape_raw_bytes_v1"
DINO_METADATA_PROJECTION_RECIPE = {
    "retain_top_level_keys": ["state_dict"],
    "add_top_level_metadata": {"tao_model": "dino"},
    "tensor_container_key": "state_dict",
    "require_exact_tensor_key_set": True,
    "require_exact_tensor_values": True,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST_RE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")


class DINOProjectionFailure(RuntimeError):
    """Stable adapter failure suitable for a structured preflight exclusion."""

    def __init__(
        self,
        code: str,
        reason: str,
        details: Mapping[str, Any] | None = None,
    ):
        self.code = code
        self.reason = reason
        self.details = dict(details or {})
        super().__init__(reason)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DINOProjectionFailure(
            "invalid_adapter_output_member",
            "Registered adapter output member must be a non-empty string",
        )
    text = value.strip()
    if (
        "\\" in text
        or text.startswith("/")
        or any(part in ("", ".", "..") for part in text.split("/"))
    ):
        raise DINOProjectionFailure(
            "invalid_adapter_output_member",
            "Registered adapter output member must be a safe relative path",
        )
    return PurePosixPath(text).as_posix()


def _verify_input(
    path: Path,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
) -> None:
    """Reverify the production evidence before any deserialization."""
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise DINOProjectionFailure(
            "adapter_input_unreadable",
            "Verified checkpoint input could not be inspected",
            {"exception_type": type(exc).__name__},
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise DINOProjectionFailure(
            "adapter_input_not_regular",
            "Verified checkpoint input must be a regular non-symlink file",
        )
    if metadata.st_size != expected_size_bytes:
        raise DINOProjectionFailure(
            "adapter_input_size_mismatch",
            "Checkpoint changed after production input verification",
            {
                "expected_size_bytes": expected_size_bytes,
                "observed_size_bytes": metadata.st_size,
            },
        )
    try:
        digest = _sha256_file(path)
    except OSError as exc:
        raise DINOProjectionFailure(
            "adapter_input_unreadable",
            "Verified checkpoint input could not be hashed",
            {"exception_type": type(exc).__name__},
        ) from exc
    if digest != expected_sha256:
        raise DINOProjectionFailure(
            "adapter_input_checksum_mismatch",
            "Checkpoint changed after production input verification",
            {
                "expected_sha256": expected_sha256,
                "observed_sha256": digest,
            },
        )


def _frame(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


@dataclass(frozen=True)
class TensorDigestEvidence:
    """Deterministic tensor-key and tensor-byte evidence."""

    input_tensor_count: int
    output_tensor_count: int
    input_tensor_keys_sha256: str
    output_tensor_keys_sha256: str
    input_tensor_values_sha256: str
    output_tensor_values_sha256: str

    @property
    def exact(self) -> bool:
        return (
            self.input_tensor_count == self.output_tensor_count
            and self.input_tensor_keys_sha256
            == self.output_tensor_keys_sha256
            and self.input_tensor_values_sha256
            == self.output_tensor_values_sha256
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hash_algorithm": TENSOR_HASH_ALGORITHM,
            "input_tensor_count": self.input_tensor_count,
            "output_tensor_count": self.output_tensor_count,
            "input_tensor_keys_sha256": self.input_tensor_keys_sha256,
            "output_tensor_keys_sha256": self.output_tensor_keys_sha256,
            "input_tensor_values_sha256": self.input_tensor_values_sha256,
            "output_tensor_values_sha256": self.output_tensor_values_sha256,
            "exact": self.exact,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TensorDigestEvidence":
        if value.get("hash_algorithm") != TENSOR_HASH_ALGORITHM:
            raise DINOProjectionFailure(
                "invalid_tensor_evidence",
                "Projection backend returned an unsupported tensor hash algorithm",
            )
        try:
            evidence = cls(
                input_tensor_count=value["input_tensor_count"],
                output_tensor_count=value["output_tensor_count"],
                input_tensor_keys_sha256=value["input_tensor_keys_sha256"],
                output_tensor_keys_sha256=value["output_tensor_keys_sha256"],
                input_tensor_values_sha256=value["input_tensor_values_sha256"],
                output_tensor_values_sha256=value["output_tensor_values_sha256"],
            )
        except (KeyError, TypeError) as exc:
            raise DINOProjectionFailure(
                "invalid_tensor_evidence",
                "Projection backend returned incomplete tensor evidence",
            ) from exc
        for count in (evidence.input_tensor_count, evidence.output_tensor_count):
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise DINOProjectionFailure(
                    "invalid_tensor_evidence",
                    "Projection tensor counts must be positive integers",
                )
        for digest in (
            evidence.input_tensor_keys_sha256,
            evidence.output_tensor_keys_sha256,
            evidence.input_tensor_values_sha256,
            evidence.output_tensor_values_sha256,
        ):
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise DINOProjectionFailure(
                    "invalid_tensor_evidence",
                    "Projection tensor evidence must contain SHA-256 digests",
                )
        return evidence


@dataclass(frozen=True)
class ProjectionBackendRequest:
    input_path: Path
    output_path: Path
    output_member: str
    input_sha256: str
    input_size_bytes: int
    recipe: Mapping[str, Any] = field(repr=False)


@dataclass(frozen=True)
class ProjectionBackendResult:
    tensor_evidence: TensorDigestEvidence
    details: Mapping[str, Any] = field(default_factory=dict)


class ProjectionBackend(Protocol):
    def transform(
        self,
        request: ProjectionBackendRequest,
    ) -> ProjectionBackendResult:
        ...


def _tensor_raw_bytes(tensor: Any, torch_module: Any) -> bytes:
    if getattr(tensor, "layout", torch_module.strided) != torch_module.strided:
        raise DINOProjectionFailure(
            "unsupported_state_dict_tensor",
            "Only dense strided state_dict tensors can be projected",
        )
    if bool(getattr(tensor, "is_quantized", False)):
        raise DINOProjectionFailure(
            "unsupported_state_dict_tensor",
            "Quantized state_dict tensors are not supported by this recipe",
        )
    try:
        value = tensor.detach().cpu().contiguous()
        # PyTorch 2.6+ rejects a dtype-changing ``view`` directly on a 0-D
        # tensor (for example BatchNorm ``num_batches_tracked``). Flatten the
        # logical tensor first so scalar and non-scalar state entries follow
        # the same byte-exact hashing path.
        byte_view = value.reshape(-1).view(torch_module.uint8).reshape(-1)
        return byte_view.numpy().tobytes(order="C")
    except Exception as exc:
        raise DINOProjectionFailure(
            "tensor_byte_hash_failed",
            "A state_dict tensor could not be converted to canonical raw bytes",
            {"exception_type": type(exc).__name__},
        ) from exc


def _state_dict_hashes(
    state_dict: Any,
    torch_module: Any,
) -> tuple[int, str, str]:
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise DINOProjectionFailure(
            "invalid_state_dict",
            "Checkpoint state_dict must be a non-empty mapping",
        )
    keys = list(state_dict)
    if not all(isinstance(key, str) and key for key in keys):
        raise DINOProjectionFailure(
            "invalid_state_dict",
            "Checkpoint state_dict keys must be non-empty strings",
        )
    sorted_keys = sorted(keys)
    key_digest = hashlib.sha256()
    value_digest = hashlib.sha256()
    for key in sorted_keys:
        tensor = state_dict[key]
        if not torch_module.is_tensor(tensor):
            raise DINOProjectionFailure(
                "invalid_state_dict_value",
                "Every state_dict value must be a tensor",
                {"key_sha256": hashlib.sha256(key.encode()).hexdigest()},
            )
        key_bytes = key.encode("utf-8")
        _frame(key_digest, key_bytes)
        _frame(value_digest, key_bytes)
        _frame(value_digest, str(tensor.dtype).encode("utf-8"))
        shape = [int(item) for item in tensor.shape]
        _frame(value_digest, _canonical_bytes(shape))
        _frame(value_digest, _tensor_raw_bytes(tensor, torch_module))
    return len(sorted_keys), key_digest.hexdigest(), value_digest.hexdigest()


def _safe_torch_load(torch_module: Any, path: Path) -> Mapping[str, Any]:
    try:
        value = torch_module.load(
            str(path),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise DINOProjectionFailure(
            "safe_checkpoint_load_failed",
            "Checkpoint could not be loaded with weights_only=True on CPU",
            {"exception_type": type(exc).__name__},
        ) from exc
    if not isinstance(value, Mapping):
        raise DINOProjectionFailure(
            "invalid_checkpoint_root",
            "Checkpoint root must be a mapping",
        )
    return value


class HostTorchProjectionBackend:
    """Explicit in-process backend; PyTorch is lazily imported when needed."""

    def __init__(self, torch_module: Any | None = None):
        self._injected_torch = torch_module

    def manifest_identity(self) -> dict[str, Any]:
        return {
            "backend": "host_torch",
            "explicit_opt_in": True,
            "torch_module_injected": self._injected_torch is not None,
            "deserialization": {
                "map_location": "cpu",
                "weights_only": True,
            },
        }

    def _torch(self) -> Any:
        if self._injected_torch is not None:
            return self._injected_torch
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DINOProjectionFailure(
                "torch_runtime_unavailable",
                "PyTorch is unavailable in the projection execution environment",
            ) from exc
        return torch

    def transform(
        self,
        request: ProjectionBackendRequest,
    ) -> ProjectionBackendResult:
        _verify_input(
            request.input_path,
            expected_size_bytes=request.input_size_bytes,
            expected_sha256=request.input_sha256,
        )
        torch_module = self._torch()
        checkpoint = _safe_torch_load(torch_module, request.input_path)
        state_dict = checkpoint.get("state_dict")
        input_count, input_keys, input_values = _state_dict_hashes(
            state_dict,
            torch_module,
        )
        projected = {
            "state_dict": state_dict,
            "tao_model": "dino",
        }
        output_member = _safe_member(request.output_member)

        try:
            with tempfile.TemporaryDirectory(
                dir=request.output_path.parent,
                prefix=".dino-metadata-projection-",
            ) as temporary_dir:
                staged_path = Path(temporary_dir) / output_member
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    torch_module.save(projected, str(staged_path))
                except Exception as exc:
                    raise DINOProjectionFailure(
                        "checkpoint_save_failed",
                        "Projected checkpoint could not be serialized",
                        {"exception_type": type(exc).__name__},
                    ) from exc

                del projected
                del state_dict
                del checkpoint
                gc.collect()

                reloaded = _safe_torch_load(torch_module, staged_path)
                if set(reloaded) != {"state_dict", "tao_model"}:
                    raise DINOProjectionFailure(
                        "projected_metadata_mismatch",
                        "Projected checkpoint has unexpected top-level metadata",
                    )
                if reloaded.get("tao_model") != "dino":
                    raise DINOProjectionFailure(
                        "projected_metadata_mismatch",
                        "Projected checkpoint tao_model metadata is invalid",
                    )
                output_count, output_keys, output_values = _state_dict_hashes(
                    reloaded["state_dict"],
                    torch_module,
                )
                evidence = TensorDigestEvidence(
                    input_tensor_count=input_count,
                    output_tensor_count=output_count,
                    input_tensor_keys_sha256=input_keys,
                    output_tensor_keys_sha256=output_keys,
                    input_tensor_values_sha256=input_values,
                    output_tensor_values_sha256=output_values,
                )
                if not evidence.exact:
                    raise DINOProjectionFailure(
                        "tensor_preservation_mismatch",
                        "Projected checkpoint changed tensor keys or values",
                        evidence.to_dict(),
                    )
                shutil.copyfile(staged_path, request.output_path)
        except DINOProjectionFailure:
            raise
        except OSError as exc:
            raise DINOProjectionFailure(
                "projection_io_failed",
                "Projection output could not be staged",
                {"exception_type": type(exc).__name__},
            ) from exc

        return ProjectionBackendResult(
            tensor_evidence=evidence,
            details={
                "execution_backend": "host_torch",
                "safe_load": {
                    "map_location": "cpu",
                    "weights_only": True,
                },
                "serialization_member": output_member,
            },
        )


@dataclass(frozen=True)
class DockerRunResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class DockerCommandRunner(Protocol):
    def run(
        self,
        argv: tuple[str, ...],
        *,
        timeout_seconds: float,
    ) -> DockerRunResult:
        ...


class SubprocessDockerCommandRunner:
    """No-shell Docker runner; command output is never exposed in evidence."""

    def run(
        self,
        argv: tuple[str, ...],
        *,
        timeout_seconds: float,
    ) -> DockerRunResult:
        try:
            completed = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except Exception as exc:
            raise DINOProjectionFailure(
                "docker_projection_launch_failed",
                "Pinned Docker projection could not be launched",
                {"exception_type": type(exc).__name__},
            ) from exc
        return DockerRunResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


class DockerTorchProjectionBackend:
    """Exact-digest, no-network TAO 7.1 projection execution backend."""

    def __init__(
        self,
        *,
        image: str = PINNED_TAO71_DOCKER_IMAGE,
        runner: DockerCommandRunner | None = None,
        timeout_seconds: float = 3600.0,
    ):
        if not isinstance(image, str) or _IMAGE_DIGEST_RE.fullmatch(image) is None:
            raise ValueError("Docker projection image must use an exact SHA-256 digest")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        self.image = image
        self.runner = runner or SubprocessDockerCommandRunner()
        self.timeout_seconds = float(timeout_seconds)
        worker_path = Path(__file__).resolve()
        self.worker_source_bytes = worker_path.read_bytes()
        self.worker_source_sha256 = hashlib.sha256(
            self.worker_source_bytes
        ).hexdigest()
        self.worker_source_size_bytes = len(self.worker_source_bytes)

    def manifest_identity(self) -> dict[str, Any]:
        return {
            "backend": "tao71_docker",
            "worker_source_sha256": self.worker_source_sha256,
            "worker_source_size_bytes": self.worker_source_size_bytes,
            "container_image": self.image,
            "pull_policy": "never",
            "network": "none",
            "input_mount": "single_file_read_only",
            "output_mount": "isolated_directory_read_write",
            "deserialization": {
                "map_location": "cpu",
                "weights_only": True,
                "after_production_sha256_and_size_verification": True,
            },
        }

    @staticmethod
    def _mount(source: Path, target: str, *, readonly: bool = False) -> str:
        source_text = str(source.resolve())
        if "," in source_text:
            raise DINOProjectionFailure(
                "unsupported_docker_mount_path",
                "Docker bind source paths must not contain commas",
            )
        value = f"type=bind,src={source_text},dst={target}"
        return f"{value},readonly" if readonly else value

    def transform(
        self,
        request: ProjectionBackendRequest,
    ) -> ProjectionBackendResult:
        _verify_input(
            request.input_path,
            expected_size_bytes=request.input_size_bytes,
            expected_sha256=request.input_sha256,
        )
        output_member = _safe_member(request.output_member)
        recipe_b64 = base64.urlsafe_b64encode(
            _canonical_bytes(request.recipe)
        ).decode("ascii")

        with tempfile.TemporaryDirectory(
            dir=request.output_path.parent,
            prefix=".dino-docker-projection-",
        ) as temporary_dir:
            temporary_root = Path(temporary_dir)
            host_input_root = temporary_root / "input"
            host_output_root = temporary_root / "output"
            host_input_root.mkdir()
            host_output_root.mkdir()
            module_path = host_input_root / "dino_checkpoint_adapter.py"
            module_path.write_bytes(self.worker_source_bytes)
            os.chmod(module_path, 0o400)
            container_output = Path("/output/adapted-output.pth")
            container_evidence = Path("/output/evidence.json")
            argv = (
                "docker",
                "run",
                "--rm",
                "--pull=never",
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--pids-limit=256",
                f"--user={os.getuid()}:{os.getgid()}",
                "--mount",
                self._mount(
                    request.input_path,
                    "/input/checkpoint.pth",
                    readonly=True,
                ),
                "--mount",
                self._mount(host_output_root, "/output"),
                "--mount",
                self._mount(
                    module_path,
                    "/opt/dino_checkpoint_adapter.py",
                    readonly=True,
                ),
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,noexec",
                self.image,
                "python",
                "/opt/dino_checkpoint_adapter.py",
                "_transform",
                "--input",
                "/input/checkpoint.pth",
                "--input-size",
                str(request.input_size_bytes),
                "--input-sha256",
                request.input_sha256,
                "--output",
                str(container_output),
                "--output-member",
                output_member,
                "--recipe-b64",
                recipe_b64,
                "--evidence",
                str(container_evidence),
            )
            result = self.runner.run(
                argv,
                timeout_seconds=self.timeout_seconds,
            )
            if not isinstance(result, DockerRunResult):
                raise DINOProjectionFailure(
                    "invalid_docker_runner_result",
                    "Docker command runner returned an invalid result",
                )
            if result.returncode != 0:
                raise DINOProjectionFailure(
                    "docker_projection_failed",
                    "Pinned TAO 7.1 Docker projection returned a nonzero status",
                    {"returncode": result.returncode},
                )
            staged_output = host_output_root / container_output.name
            evidence_path = host_output_root / container_evidence.name
            if (
                staged_output.is_symlink()
                or evidence_path.is_symlink()
                or not staged_output.is_file()
                or not evidence_path.is_file()
            ):
                raise DINOProjectionFailure(
                    "docker_projection_output_missing",
                    "Pinned Docker projection did not produce regular artifacts",
                )
            try:
                evidence_document = json.loads(
                    evidence_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError, TypeError) as exc:
                raise DINOProjectionFailure(
                    "invalid_docker_projection_evidence",
                    "Pinned Docker projection evidence is invalid",
                    {"exception_type": type(exc).__name__},
                ) from exc
            if not isinstance(evidence_document, Mapping):
                raise DINOProjectionFailure(
                    "invalid_docker_projection_evidence",
                    "Pinned Docker projection evidence must be an object",
                )
            tensor_evidence = TensorDigestEvidence.from_dict(
                evidence_document.get("tensor_evidence", {})
            )
            if not tensor_evidence.exact:
                raise DINOProjectionFailure(
                    "tensor_preservation_mismatch",
                    "Pinned Docker projection changed tensor keys or values",
                    tensor_evidence.to_dict(),
                )
            shutil.copyfile(staged_output, request.output_path)
            return ProjectionBackendResult(
                tensor_evidence=tensor_evidence,
                details={
                    "execution_backend": "tao71_docker",
                    "container_image": self.image,
                    "docker_pull_policy": "never",
                    "docker_network": "none",
                    "safe_load": {
                        "map_location": "cpu",
                        "weights_only": True,
                    },
                    "serialization_member": output_member,
                },
            )


class DINOCheckpointMetadataProjectionCallback:
    """Production preflight callback for the one registered DINO recipe."""

    def __init__(self, backend: ProjectionBackend | None = None):
        self.backend = backend or DockerTorchProjectionBackend()

    def manifest_identity(self) -> dict[str, Any]:
        identity = getattr(self.backend, "manifest_identity", None)
        if callable(identity):
            backend = identity()
        else:
            backend = {
                "backend": "injected",
                "backend_type": (
                    f"{type(self.backend).__module__}."
                    f"{type(self.backend).__qualname__}"
                ),
            }
        return {
            "callback": "dino_checkpoint_metadata_projection_v1",
            "worker_source_sha256": _sha256_file(Path(__file__).resolve()),
            "worker_source_size_bytes": Path(__file__).resolve().stat().st_size,
            "recipe_sha256": _canonical_sha256(
                DINO_METADATA_PROJECTION_RECIPE
            ),
            "backend": backend,
        }

    @staticmethod
    def _registered_adapter(request: Any) -> Mapping[str, Any]:
        matches = [
            adapter
            for adapter in request.registry_record.get("artifact_adapters", ())
            if adapter.get("id") == request.adapter_id
        ]
        if len(matches) != 1:
            raise DINOProjectionFailure(
                "adapter_registry_binding_mismatch",
                "Adapter request does not bind exactly one registry record",
            )
        adapter = matches[0]
        if (
            adapter.get("adapter_type") != request.adapter_type
            or _canonical_sha256(adapter) != request.adapter_sha256
        ):
            raise DINOProjectionFailure(
                "adapter_registry_binding_mismatch",
                "Adapter request does not match the registered adapter identity",
            )
        return adapter

    def __call__(self, request: Any) -> Any:
        from tao_automl.ptm_preflight import (
            ArtifactAdapterCallbackResult,
            TensorPreservationEvidence,
        )

        try:
            if copy.deepcopy(dict(request.recipe)) != (
                DINO_METADATA_PROJECTION_RECIPE
            ):
                raise DINOProjectionFailure(
                    "unsupported_dino_projection_recipe",
                    "DINO adapter supports only the registered metadata projection",
                )
            if _canonical_sha256(request.recipe) != request.recipe_sha256:
                raise DINOProjectionFailure(
                    "adapter_recipe_binding_mismatch",
                    "Adapter recipe does not match its production request hash",
                )
            if not isinstance(request.input_sha256, str) or (
                _SHA256_RE.fullmatch(request.input_sha256) is None
            ):
                raise DINOProjectionFailure(
                    "invalid_adapter_input_digest",
                    "Adapter input SHA-256 is invalid",
                )
            _verify_input(
                Path(request.input_path),
                expected_size_bytes=request.input_size_bytes,
                expected_sha256=request.input_sha256,
            )
            adapter = self._registered_adapter(request)
            if adapter["recipe"] != DINO_METADATA_PROJECTION_RECIPE:
                raise DINOProjectionFailure(
                    "unsupported_dino_projection_recipe",
                    "Registered adapter recipe is not the supported DINO projection",
                )
            output_record = adapter["output"]
            backend_result = self.backend.transform(
                ProjectionBackendRequest(
                    input_path=Path(request.input_path),
                    output_path=Path(request.output_path),
                    output_member=output_record["member"],
                    input_sha256=request.input_sha256,
                    input_size_bytes=request.input_size_bytes,
                    recipe=copy.deepcopy(dict(request.recipe)),
                )
            )
            if not isinstance(backend_result, ProjectionBackendResult):
                raise DINOProjectionFailure(
                    "invalid_projection_backend_result",
                    "Projection backend returned an invalid result",
                )
            evidence = backend_result.tensor_evidence
            if not evidence.exact:
                raise DINOProjectionFailure(
                    "tensor_preservation_mismatch",
                    "Projected checkpoint changed tensor keys or values",
                    evidence.to_dict(),
                )
            output_path = Path(request.output_path)
            if output_path.is_symlink() or not output_path.is_file():
                raise DINOProjectionFailure(
                    "adapted_output_missing",
                    "Projection backend did not produce a regular output file",
                )
            observed_size = output_path.stat().st_size
            observed_sha = _sha256_file(output_path)
            expected_size = output_record["expected_size_bytes"]
            expected_sha = output_record["sha256"].lower()
            if observed_size != expected_size:
                raise DINOProjectionFailure(
                    "adapted_output_size_mismatch",
                    "Deterministic projection did not reproduce registered size",
                    {
                        "expected_size_bytes": expected_size,
                        "observed_size_bytes": observed_size,
                    },
                )
            if observed_sha != expected_sha:
                raise DINOProjectionFailure(
                    "adapted_output_checksum_mismatch",
                    "Deterministic projection did not reproduce registered SHA-256",
                    {
                        "expected_sha256": expected_sha,
                        "observed_sha256": observed_sha,
                    },
                )
            production_evidence = TensorPreservationEvidence(
                hash_algorithm=TENSOR_HASH_ALGORITHM,
                input_tensor_count=evidence.input_tensor_count,
                output_tensor_count=evidence.output_tensor_count,
                input_tensor_keys_sha256=evidence.input_tensor_keys_sha256,
                output_tensor_keys_sha256=evidence.output_tensor_keys_sha256,
                input_tensor_values_sha256=evidence.input_tensor_values_sha256,
                output_tensor_values_sha256=evidence.output_tensor_values_sha256,
            )
            details = dict(backend_result.details)
            details.update(
                {
                    "recipe_sha256": request.recipe_sha256,
                    "output_member": output_record["member"],
                    "output_size_bytes": observed_size,
                    "output_sha256": observed_sha,
                    "tensor_evidence_exact": evidence.exact,
                }
            )
            return ArtifactAdapterCallbackResult(
                True,
                "dino_metadata_projection_verified",
                "DINO checkpoint metadata projection verified",
                production_evidence,
                details,
            )
        except DINOProjectionFailure as exc:
            return ArtifactAdapterCallbackResult(
                False,
                exc.code,
                exc.reason,
                None,
                exc.details,
            )
        except Exception as exc:
            return ArtifactAdapterCallbackResult(
                False,
                "unexpected_dino_projection_error",
                "Unexpected DINO projection failure; inspect protected logs",
                None,
                {"exception_type": type(exc).__name__},
            )


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    content = _canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _transform_cli(arguments: argparse.Namespace) -> int:
    try:
        recipe_raw = base64.urlsafe_b64decode(
            arguments.recipe_b64.encode("ascii")
        )
        recipe = json.loads(recipe_raw.decode("utf-8"))
    except Exception as exc:
        raise SystemExit(
            f"invalid recipe payload ({type(exc).__name__})"
        ) from exc
    if recipe != DINO_METADATA_PROJECTION_RECIPE:
        raise SystemExit("unsupported DINO projection recipe")
    result = HostTorchProjectionBackend().transform(
        ProjectionBackendRequest(
            input_path=Path(arguments.input),
            output_path=Path(arguments.output),
            output_member=arguments.output_member,
            input_sha256=arguments.input_sha256,
            input_size_bytes=arguments.input_size,
            recipe=recipe,
        )
    )
    _write_json_create_only(
        Path(arguments.evidence),
        {
            "schema_version": 1,
            "tensor_evidence": result.tensor_evidence.to_dict(),
            "details": dict(result.details),
        },
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    transform = subparsers.add_parser("_transform")
    transform.add_argument("--input", required=True)
    transform.add_argument("--input-size", type=int, required=True)
    transform.add_argument("--input-sha256", required=True)
    transform.add_argument("--output", required=True)
    transform.add_argument("--output-member", required=True)
    transform.add_argument("--recipe-b64", required=True)
    transform.add_argument("--evidence", required=True)
    return parser


if __name__ == "__main__":
    parsed = _parser().parse_args()
    if parsed.command == "_transform":
        raise SystemExit(_transform_cli(parsed))
