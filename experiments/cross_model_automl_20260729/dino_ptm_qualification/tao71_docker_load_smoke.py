# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concrete exact-digest TAO 7.1 Docker load smoke for DINO PTMs.

The module is also its own in-container worker. Top-level imports are standard
library only so the single file can be mounted read-only into the pinned TAO
image without installing the AutoML checkout in that image.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol


PINNED_TAO71_DOCKER_IMAGE = (
    "nvcr.io/nvstaging/tao/tao-toolkit-pyt@"
    "sha256:949c0ea8ace09ac91951be4169353cf214daaa3ede7db9eed94070b020361667"
)
PINNED_TAO71_CONTAINER_IDENTITY = (
    "sha256:949c0ea8ace09ac91951be4169353cf214daaa3ede7db9eed94070b020361667"
)
DEFAULT_TAO_VERSION = "7.1.0-rc-245"
LOAD_SMOKE_SCHEMA_VERSION = 1
LOAD_COVERAGE_POLICY_VERSION = 1
FULL_DETECTOR_CHECKPOINT_TARGET = "train.pretrained_model_path"
BACKBONE_CHECKPOINT_TARGET = "model.pretrained_backbone_path"
SUPPORTED_CHECKPOINT_TARGETS = frozenset(
    {
        FULL_DETECTOR_CHECKPOINT_TARGET,
        BACKBONE_CHECKPOINT_TARGET,
    }
)

# A registered full detector is expected to initialize nearly the entire
# architecture. A registered backbone may omit small task heads, but must
# initialize most target parameters by volume and at least half of target
# tensor entries. These are qualification-safety gates, not AutoML objectives.
FULL_MODEL_MIN_TARGET_TENSOR_FRACTION = 0.90
FULL_MODEL_MIN_TARGET_NUMEL_FRACTION = 0.90
BACKBONE_MIN_TARGET_TENSOR_FRACTION = 0.50
BACKBONE_MIN_TARGET_NUMEL_FRACTION = 0.90

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST_RE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_TAO71_RE = re.compile(r"^7\.1(?:\.0)?(?:$|[-+].*)")

# PyTorch's restricted weights-only unpickler rejects argparse.Namespace
# unless it is explicitly allowlisted. One official NVImageNet checkpoint
# stores training arguments at the root alongside its tensor state. The
# exception is bound to both the stable registry ID and the verified official
# artifact digest; all other globals and artifacts remain rejected.
_CHECKPOINT_SAFE_GLOBALS = {
    "dino.backbone.nvimagenet.resnet50": {
        "checkpoint_sha256": (
            "49b0df2b517a28760e17158c9ad78371"
            "c1f833d6ad257f117ff81356743060b7"
        ),
        "allowed_globals": ("argparse.Namespace",),
    },
}


def coverage_policy(checkpoint_target: str) -> dict[str, Any]:
    """Return the immutable qualification coverage gate for one target."""
    if checkpoint_target == FULL_DETECTOR_CHECKPOINT_TARGET:
        minimum_tensor_fraction = FULL_MODEL_MIN_TARGET_TENSOR_FRACTION
        minimum_numel_fraction = FULL_MODEL_MIN_TARGET_NUMEL_FRACTION
        load_scope = "full_detector"
    elif checkpoint_target == BACKBONE_CHECKPOINT_TARGET:
        minimum_tensor_fraction = BACKBONE_MIN_TARGET_TENSOR_FRACTION
        minimum_numel_fraction = BACKBONE_MIN_TARGET_NUMEL_FRACTION
        load_scope = "backbone"
    else:
        raise TAO71LoadSmokeFailure(
            "unsupported_checkpoint_target",
            "DINO checkpoint target is not supported by TAO 7.1 load smoke",
            {"checkpoint_target": checkpoint_target},
        )
    return {
        "policy_version": LOAD_COVERAGE_POLICY_VERSION,
        "load_scope": load_scope,
        "minimum_target_tensor_fraction": minimum_tensor_fraction,
        "minimum_target_numel_fraction": minimum_numel_fraction,
        "require_all_shape_compatible_values_loaded": True,
        "require_finite_matched_tensors": True,
    }


class TAO71LoadSmokeFailure(RuntimeError):
    """Stable, secret-free Docker load-smoke failure."""

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


def _checkpoint_safe_global_names(
    checkpoint_id: str,
    checkpoint_sha256: str,
) -> tuple[str, ...]:
    policy = _CHECKPOINT_SAFE_GLOBALS.get(checkpoint_id)
    if policy is None:
        return ()
    if checkpoint_sha256 != policy["checkpoint_sha256"]:
        raise TAO71LoadSmokeFailure(
            "safe_global_checkpoint_identity_mismatch",
            "Checkpoint-specific safe globals require the registered artifact digest",
            {"checkpoint_id": checkpoint_id},
        )
    return tuple(policy["allowed_globals"])


def _validated_checkpoint_safe_global_names(
    checkpoint_id: str,
    checkpoint_sha256: str,
    provided: tuple[str, ...],
) -> tuple[str, ...]:
    expected = _checkpoint_safe_global_names(
        checkpoint_id,
        checkpoint_sha256,
    )
    if provided != expected:
        raise ValueError(
            "checkpoint-specific safe-global policy disagrees with artifact identity"
        )
    return expected


@contextlib.contextmanager
def _weights_only_safe_globals(
    torch_module: Any,
    names: tuple[str, ...],
):
    supported = {"argparse.Namespace": argparse.Namespace}
    if any(name not in supported for name in names):
        raise ValueError("unsupported weights-only safe global")
    values = [supported[name] for name in names]
    if values:
        with torch_module.serialization.safe_globals(values):
            yield
    else:
        yield


def _registered_load_target(request: Any) -> tuple[str, str]:
    record = request.registry_record
    if not isinstance(record, Mapping):
        raise TAO71LoadSmokeFailure(
            "invalid_load_smoke_registry_record",
            "Load-smoke registry record must be an object",
        )
    if record.get("id") != request.checkpoint_id:
        raise TAO71LoadSmokeFailure(
            "load_smoke_registry_binding_mismatch",
            "Load-smoke request does not match its registry record",
        )
    if record.get("model_family") != "dino":
        raise TAO71LoadSmokeFailure(
            "load_smoke_registry_binding_mismatch",
            "Load-smoke registry record is not a DINO checkpoint",
        )
    checkpoint_target = record.get("checkpoint_target")
    if checkpoint_target not in SUPPORTED_CHECKPOINT_TARGETS:
        raise TAO71LoadSmokeFailure(
            "unsupported_checkpoint_target",
            "DINO checkpoint target is not supported by TAO 7.1 load smoke",
            {"checkpoint_target": checkpoint_target},
        )
    backbone = record.get("backbone")
    if not isinstance(backbone, str) or not backbone.strip():
        raise TAO71LoadSmokeFailure(
            "invalid_registered_backbone",
            "DINO registry record must identify a non-empty backbone",
        )
    default_model = record.get("default_spec_overrides", {}).get("model", {})
    if (
        not isinstance(default_model, Mapping)
        or default_model.get("backbone") != backbone
    ):
        raise TAO71LoadSmokeFailure(
            "load_smoke_registry_binding_mismatch",
            "Registered DINO backbone and default overrides disagree",
        )
    return checkpoint_target, backbone


def _verify_registry_artifact_identity(
    request: Any,
    *,
    checkpoint_target: str,
    observed_size_bytes: int,
    observed_sha256: str,
) -> None:
    """Rebind the effective checkpoint to registered immutable evidence."""
    record = request.registry_record
    adapters = record.get("artifact_adapters", ())
    outputs = [
        adapter.get("output")
        for adapter in adapters
        if isinstance(adapter, Mapping)
        and isinstance(adapter.get("output"), Mapping)
    ]
    if checkpoint_target == FULL_DETECTOR_CHECKPOINT_TARGET and outputs:
        registered = {
            (output.get("expected_size_bytes"), output.get("sha256"))
            for output in outputs
        }
        if (observed_size_bytes, observed_sha256) not in registered:
            raise TAO71LoadSmokeFailure(
                "load_smoke_registry_artifact_mismatch",
                "Effective DINO checkpoint does not match a registered "
                "adapted artifact identity",
            )
        return
    expected_size = record.get("expected_size_bytes")
    expected_sha = record.get("sha256")
    if observed_size_bytes != expected_size:
        raise TAO71LoadSmokeFailure(
            "load_smoke_registry_artifact_mismatch",
            "Effective DINO checkpoint size does not match the registry",
        )
    if expected_sha is not None and observed_sha256 != expected_sha:
        raise TAO71LoadSmokeFailure(
            "load_smoke_registry_artifact_mismatch",
            "Effective DINO checkpoint SHA-256 does not match the registry",
        )


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


def _verify_regular_file(
    path: Path,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TAO71LoadSmokeFailure(
            "load_smoke_input_unreadable",
            "Load-smoke checkpoint could not be inspected",
            {"exception_type": type(exc).__name__},
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise TAO71LoadSmokeFailure(
            "load_smoke_input_not_regular",
            "Load-smoke checkpoint must be a regular non-symlink file",
        )
    if metadata.st_size != expected_size_bytes:
        raise TAO71LoadSmokeFailure(
            "load_smoke_input_size_mismatch",
            "Load-smoke checkpoint size changed after preflight",
            {
                "expected_size_bytes": expected_size_bytes,
                "observed_size_bytes": metadata.st_size,
            },
        )
    observed_sha = _sha256_file(path)
    if observed_sha != expected_sha256:
        raise TAO71LoadSmokeFailure(
            "load_smoke_input_checksum_mismatch",
            "Load-smoke checkpoint SHA-256 changed after preflight",
            {
                "expected_sha256": expected_sha256,
                "observed_sha256": observed_sha,
            },
        )


def _snapshot_regular_file(path: Path) -> tuple[int, str]:
    """Return a stable regular-file identity without following symlinks."""
    try:
        before = path.lstat()
    except OSError as exc:
        raise TAO71LoadSmokeFailure(
            "load_smoke_input_unreadable",
            "Load-smoke checkpoint could not be inspected",
            {"exception_type": type(exc).__name__},
        ) from exc
    if not stat.S_ISREG(before.st_mode):
        raise TAO71LoadSmokeFailure(
            "load_smoke_input_not_regular",
            "Load-smoke checkpoint must be a regular non-symlink file",
        )
    observed_sha = _sha256_file(path)
    try:
        after = path.lstat()
    except OSError as exc:
        raise TAO71LoadSmokeFailure(
            "load_smoke_input_changed",
            "Load-smoke checkpoint changed while it was being verified",
            {"exception_type": type(exc).__name__},
        ) from exc
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
        raise TAO71LoadSmokeFailure(
            "load_smoke_input_changed",
            "Load-smoke checkpoint changed while it was being verified",
        )
    return before.st_size, observed_sha


def _deep_merge(
    lower: Mapping[str, Any],
    higher: Mapping[str, Any],
) -> dict[str, Any]:
    merged = {
        key: (
            _deep_merge(value, {})
            if isinstance(value, Mapping)
            else json.loads(json.dumps(value))
        )
        for key, value in lower.items()
    }
    for key, value in higher.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = (
                _deep_merge(value, {})
                if isinstance(value, Mapping)
                else json.loads(json.dumps(value))
            )
    return merged


def merged_checkpoint_overrides(request: Any) -> dict[str, Any]:
    """Merge sidecar then registered defaults; runtime path stays container-side."""
    if not isinstance(request.checkpoint_spec, Mapping):
        raise TAO71LoadSmokeFailure(
            "invalid_checkpoint_sidecar",
            "Checkpoint sidecar must be a mapping",
        )
    if not isinstance(request.default_spec_overrides, Mapping):
        raise TAO71LoadSmokeFailure(
            "invalid_checkpoint_defaults",
            "Registered checkpoint defaults must be a mapping",
        )
    merged = _deep_merge(
        request.checkpoint_spec,
        request.default_spec_overrides,
    )
    train = merged.setdefault("train", {})
    if not isinstance(train, dict):
        raise TAO71LoadSmokeFailure(
            "invalid_checkpoint_overrides",
            "Merged train overrides must be an object",
        )
    train.pop("pretrained_model_path", None)
    model = merged.setdefault("model", {})
    if not isinstance(model, dict):
        raise TAO71LoadSmokeFailure(
            "invalid_checkpoint_overrides",
            "Merged model overrides must be an object",
        )
    model["pretrained_backbone_path"] = None
    return merged


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
    """Execute an argv-only Docker command; never expose captured output."""

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
            raise TAO71LoadSmokeFailure(
                "tao71_load_smoke_launch_failed",
                "Pinned TAO 7.1 Docker load smoke could not be launched",
                {"exception_type": type(exc).__name__},
            ) from exc
        return DockerRunResult(
            completed.returncode,
            completed.stdout,
            completed.stderr,
        )


class TAO71DINOCheckpointLoadSmoke:
    """Concrete production ``CheckpointLoadSmokeCallback`` implementation."""

    def __init__(
        self,
        *,
        image: str = PINNED_TAO71_DOCKER_IMAGE,
        tao_version: str = DEFAULT_TAO_VERSION,
        container_identity: str = PINNED_TAO71_CONTAINER_IDENTITY,
        runner: DockerCommandRunner | None = None,
        timeout_seconds: float = 1800.0,
    ):
        if _IMAGE_DIGEST_RE.fullmatch(image) is None:
            raise ValueError("image must use an exact SHA-256 digest")
        if _TAO71_RE.fullmatch(tao_version) is None:
            raise ValueError("tao_version must identify the pinned TAO 7.1 line")
        expected_identity = image.rsplit("@", 1)[1]
        if container_identity != expected_identity:
            raise ValueError(
                "container_identity must equal the pinned image digest"
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        self.image = image
        self.tao_version = tao_version
        self.container_identity = container_identity
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
            "callback": "tao71_dino_checkpoint_load_smoke_v1",
            "worker_source_sha256": self.worker_source_sha256,
            "worker_source_size_bytes": self.worker_source_size_bytes,
            "execution_backend": "docker",
            "container_image": self.image,
            "container_identity": self.container_identity,
            "tao_version": self.tao_version,
            "pull_policy": "never",
            "network": "none",
            "checkpoint_mount": "single_file_read_only",
            "merged_overrides_mount": "single_file_read_only",
            "output_mount": "isolated_directory_read_write",
            "device": "cpu",
            "safe_load": {
                "map_location": "cpu",
                "weights_only": True,
                "checkpoint_specific_allowed_globals": {
                    checkpoint_id: {
                        "checkpoint_sha256": policy["checkpoint_sha256"],
                        "allowed_globals": list(policy["allowed_globals"]),
                    }
                    for checkpoint_id, policy in sorted(
                        _CHECKPOINT_SAFE_GLOBALS.items()
                    )
                },
            },
            "target_routing": sorted(SUPPORTED_CHECKPOINT_TARGETS),
            "coverage_policies": {
                "full_detector": coverage_policy(
                    FULL_DETECTOR_CHECKPOINT_TARGET
                ),
                "backbone": coverage_policy(BACKBONE_CHECKPOINT_TARGET),
            },
        }

    @staticmethod
    def _mount(source: Path, target: str, *, readonly: bool = False) -> str:
        source_text = str(source.resolve())
        if "," in source_text:
            raise TAO71LoadSmokeFailure(
                "unsupported_docker_mount_path",
                "Docker bind source paths must not contain commas",
            )
        value = f"type=bind,src={source_text},dst={target}"
        return f"{value},readonly" if readonly else value

    @staticmethod
    def _failure_result(exc: TAO71LoadSmokeFailure) -> Any:
        from tao_automl.ptm_preflight import CheckpointLoadSmokeResult

        return CheckpointLoadSmokeResult(
            False,
            exc.code,
            exc.reason,
            exc.details,
        )

    def __call__(self, request: Any) -> Any:
        from tao_automl.ptm_preflight import CheckpointLoadSmokeResult

        try:
            if request.model != "dino" or request.task != "object_detection":
                raise TAO71LoadSmokeFailure(
                    "unsupported_load_smoke_target",
                    "TAO 7.1 DINO load smoke requires object_detection",
                )
            if request.tao_version != self.tao_version:
                raise TAO71LoadSmokeFailure(
                    "load_smoke_tao_version_mismatch",
                    "Load-smoke request does not match the pinned TAO version",
                )
            checkpoint_target, backbone = _registered_load_target(request)
            active_coverage_policy = coverage_policy(checkpoint_target)
            checkpoint_path = Path(request.checkpoint_path)
            expected_size, expected_sha = _snapshot_regular_file(
                checkpoint_path
            )
            _verify_registry_artifact_identity(
                request,
                checkpoint_target=checkpoint_target,
                observed_size_bytes=expected_size,
                observed_sha256=expected_sha,
            )
            _verify_regular_file(
                checkpoint_path,
                expected_size_bytes=expected_size,
                expected_sha256=expected_sha,
            )
            safe_global_names = _checkpoint_safe_global_names(
                request.checkpoint_id,
                expected_sha,
            )
            overrides = merged_checkpoint_overrides(request)
            overrides_sha = _canonical_sha256(overrides)

            with tempfile.TemporaryDirectory(
                dir=checkpoint_path.parent,
                prefix=".tao71-dino-load-smoke-",
            ) as temporary_dir:
                root = Path(temporary_dir)
                input_root = root / "input"
                output_root = root / "output"
                input_root.mkdir()
                output_root.mkdir()
                overrides_path = input_root / "merged-overrides.json"
                # The mounted file is itself integrity-checked in the
                # container against ``overrides_sha``. Keep its bytes exactly
                # equal to the canonical payload used to derive that digest.
                overrides_path.write_bytes(_canonical_bytes(overrides))
                os.chmod(overrides_path, 0o600)
                worker_path = input_root / "tao71_dino_load_smoke.py"
                worker_path.write_bytes(self.worker_source_bytes)
                os.chmod(worker_path, 0o400)
                evidence_path = output_root / "load-smoke-evidence.json"
                safe_global_argv = tuple(
                    item
                    for name in safe_global_names
                    for item in ("--safe-global", name)
                )
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
                    "--env=PYTHONDONTWRITEBYTECODE=1",
                    "--env=PYTHONHASHSEED=0",
                    # The host UID is intentionally not required to exist in
                    # the image's passwd database. Torch's cache bootstrap
                    # calls getpass.getuser(), so provide a non-secret,
                    # deterministic identity through its documented env path.
                    "--env=USER=tao-automl",
                    "--env=LOGNAME=tao-automl",
                    "--env=NVIDIA_VISIBLE_DEVICES=void",
                    "--env=CUDA_VISIBLE_DEVICES=",
                    "--env=HOME=/tmp",
                    "--env=XDG_CACHE_HOME=/tmp",
                    "--workdir=/tmp",
                    "--mount",
                    self._mount(
                        checkpoint_path,
                        "/input/checkpoint.pth",
                        readonly=True,
                    ),
                    "--mount",
                    self._mount(
                        overrides_path,
                        "/input/merged-overrides.json",
                        readonly=True,
                    ),
                    "--mount",
                    self._mount(output_root, "/output"),
                    "--mount",
                    self._mount(
                        worker_path,
                        "/opt/tao71_dino_load_smoke.py",
                        readonly=True,
                    ),
                    "--tmpfs",
                    "/tmp:rw,nosuid,nodev,noexec",
                    self.image,
                    "python",
                    "/opt/tao71_dino_load_smoke.py",
                    "_load",
                    "--checkpoint",
                    "/input/checkpoint.pth",
                    "--checkpoint-size",
                    str(expected_size),
                    "--checkpoint-sha256",
                    expected_sha,
                    "--overrides",
                    "/input/merged-overrides.json",
                    "--overrides-sha256",
                    overrides_sha,
                    "--checkpoint-id",
                    request.checkpoint_id,
                    "--checkpoint-target",
                    checkpoint_target,
                    "--backbone",
                    backbone,
                    "--tao-version",
                    self.tao_version,
                    "--container-identity",
                    self.container_identity,
                    *safe_global_argv,
                    "--evidence",
                    "/output/load-smoke-evidence.json",
                )
                run_result = self.runner.run(
                    argv,
                    timeout_seconds=self.timeout_seconds,
                )
                if not isinstance(run_result, DockerRunResult):
                    raise TAO71LoadSmokeFailure(
                        "invalid_load_smoke_runner_result",
                        "Docker load-smoke runner returned an invalid result",
                    )
                if run_result.returncode != 0:
                    raise TAO71LoadSmokeFailure(
                        "tao71_docker_load_smoke_failed",
                        "Pinned TAO 7.1 Docker load smoke returned nonzero",
                        {"returncode": run_result.returncode},
                    )
                if (
                    evidence_path.is_symlink()
                    or not evidence_path.is_file()
                ):
                    raise TAO71LoadSmokeFailure(
                        "tao71_load_smoke_evidence_missing",
                        "Pinned TAO 7.1 load smoke produced no regular evidence",
                    )
                try:
                    evidence = json.loads(
                        evidence_path.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeError, ValueError, TypeError) as exc:
                    raise TAO71LoadSmokeFailure(
                        "invalid_tao71_load_smoke_evidence",
                        "Pinned TAO 7.1 load-smoke evidence is invalid",
                        {"exception_type": type(exc).__name__},
                    ) from exc
                if not isinstance(evidence, Mapping):
                    raise TAO71LoadSmokeFailure(
                        "invalid_tao71_load_smoke_evidence",
                        "Pinned TAO 7.1 load-smoke evidence must be an object",
                    )

            expected = {
                "schema_version": LOAD_SMOKE_SCHEMA_VERSION,
                "contract_version": 1,
                "checkpoint_id": request.checkpoint_id,
                "checkpoint_target": checkpoint_target,
                "backbone": backbone,
                "checkpoint_sha256": expected_sha,
                "checkpoint_size_bytes": expected_size,
                "execution_backend": "docker",
                "container_identity": self.container_identity,
                "tao_version": self.tao_version,
                "device": "cpu",
                "weights_only": True,
                "weights_only_allowed_globals": list(safe_global_names),
                "merged_overrides_sha256": overrides_sha,
                "coverage_policy": active_coverage_policy,
                "tao_load_path_executed": True,
                "tao_load_path": (
                    "train_shape_aware_full_detector"
                    if checkpoint_target == FULL_DETECTOR_CHECKPOINT_TARGET
                    else "model_pretrained_backbone_path"
                ),
                "safe_path_load_count": (
                    0
                    if checkpoint_target == FULL_DETECTOR_CHECKPOINT_TARGET
                    else 1
                ),
            }
            mismatched = sorted(
                key
                for key, value in expected.items()
                if evidence.get(key) != value
            )
            if mismatched:
                raise TAO71LoadSmokeFailure(
                    "invalid_tao71_load_smoke_evidence",
                    "Pinned TAO 7.1 load-smoke evidence failed verification",
                    {"missing_or_mismatched_fields": mismatched},
                )
            count_fields = (
                "source_tensor_count",
                "target_tensor_count",
                "matched_tensor_count",
                "matched_numel",
                "target_numel",
                "shape_mismatch_count",
                "missing_target_count",
                "unexpected_source_count",
                "loaded_value_match_count",
                "loaded_value_match_numel",
            )
            invalid_counts = [
                key
                for key in count_fields
                if isinstance(evidence.get(key), bool)
                or not isinstance(evidence.get(key), int)
                or evidence.get(key) < 0
            ]
            if (
                invalid_counts
                or evidence["source_tensor_count"] <= 0
                or evidence["target_tensor_count"] <= 0
                or evidence["matched_tensor_count"] <= 0
                or evidence["target_numel"] <= 0
                or evidence["matched_numel"] <= 0
                or evidence["matched_numel"] > evidence["target_numel"]
                or evidence["loaded_value_match_count"]
                != evidence["matched_tensor_count"]
                or evidence["loaded_value_match_numel"]
                != evidence["matched_numel"]
            ):
                raise TAO71LoadSmokeFailure(
                    "invalid_tao71_load_smoke_evidence",
                    "Pinned TAO 7.1 load-smoke tensor counts are invalid",
                    {"invalid_count_fields": sorted(invalid_counts)},
                )
            fraction_fields = (
                "matched_target_tensor_fraction",
                "matched_target_numel_fraction",
            )
            invalid_fractions = [
                key
                for key in fraction_fields
                if isinstance(evidence.get(key), bool)
                or not isinstance(evidence.get(key), (int, float))
                or not math.isfinite(float(evidence.get(key)))
                or not 0.0 <= float(evidence.get(key)) <= 1.0
            ]
            if invalid_fractions:
                raise TAO71LoadSmokeFailure(
                    "invalid_tao71_load_smoke_evidence",
                    "Pinned TAO 7.1 load-smoke coverage is invalid",
                    {"invalid_fraction_fields": sorted(invalid_fractions)},
                )
            for key in (
                "adapted_state_keys_sha256",
                "matched_state_keys_sha256",
                "missing_target_keys_sha256",
                "shape_mismatch_keys_sha256",
                "unexpected_source_keys_sha256",
                "loaded_value_match_keys_sha256",
            ):
                value = evidence.get(key)
                if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                    raise TAO71LoadSmokeFailure(
                        "invalid_tao71_load_smoke_evidence",
                        "Pinned TAO 7.1 load-smoke key evidence is invalid",
                        {"invalid_digest_field": key},
                    )
            expected_tensor_fraction = (
                evidence["matched_tensor_count"]
                / evidence["target_tensor_count"]
            )
            expected_numel_fraction = (
                evidence["matched_numel"] / evidence["target_numel"]
            )
            if (
                not math.isclose(
                    float(evidence["matched_target_tensor_fraction"]),
                    expected_tensor_fraction,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    float(evidence["matched_target_numel_fraction"]),
                    expected_numel_fraction,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                raise TAO71LoadSmokeFailure(
                    "invalid_tao71_load_smoke_evidence",
                    "Pinned TAO 7.1 load-smoke coverage is inconsistent",
                )
            coverage_passed = _coverage_passes(
                evidence,
                active_coverage_policy,
            )
            if (
                evidence.get("checkpoint_loaded") is not coverage_passed
                or evidence.get("state_dict_compatible") is not coverage_passed
            ):
                raise TAO71LoadSmokeFailure(
                    "invalid_tao71_load_smoke_evidence",
                    "Pinned TAO 7.1 compatibility decision is inconsistent",
                )
            if not coverage_passed:
                raise TAO71LoadSmokeFailure(
                    "insufficient_tao71_checkpoint_coverage",
                    "TAO 7.1 loaded the checkpoint but compatible coverage "
                    "did not satisfy the frozen qualification policy",
                    {
                        "checkpoint_target": checkpoint_target,
                        "matched_tensor_count": (
                            evidence["matched_tensor_count"]
                        ),
                        "target_tensor_count": (
                            evidence["target_tensor_count"]
                        ),
                        "matched_target_tensor_fraction": (
                            evidence["matched_target_tensor_fraction"]
                        ),
                        "matched_target_numel_fraction": (
                            evidence["matched_target_numel_fraction"]
                        ),
                        "coverage_policy": active_coverage_policy,
                    },
                )
            return CheckpointLoadSmokeResult(
                True,
                "tao71_dino_checkpoint_loaded",
                "Pinned TAO 7.1 DINO checkpoint load smoke passed",
                dict(evidence),
            )
        except TAO71LoadSmokeFailure as exc:
            return self._failure_result(exc)
        except Exception as exc:
            return CheckpointLoadSmokeResult(
                False,
                "unexpected_tao71_load_smoke_error",
                "Unexpected TAO 7.1 load-smoke failure; inspect protected logs",
                {"exception_type": type(exc).__name__},
            )


def _keys_sha256(keys: Any) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        encoded = str(key).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _state_compatibility(
    *,
    torch: Any,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    loaded_target: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure shape coverage and prove that compatible values were loaded."""
    if not source or not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in source.items()
    ):
        raise ValueError("adapted checkpoint state_dict is not tensor-only")
    if not target or not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in target.items()
    ):
        raise ValueError("target model state_dict is not tensor-only")
    if set(target) != set(loaded_target):
        raise ValueError("loaded target state_dict key set changed")

    compatible: dict[str, Any] = {}
    matched_keys = []
    shape_mismatch = []
    missing_target = []
    loaded_value_match = []
    loaded_value_match_numel = 0
    matched_numel = 0
    target_numel = 0
    for key in sorted(target):
        target_tensor = target[key]
        target_numel += int(target_tensor.numel())
        source_tensor = source.get(key)
        if source_tensor is None:
            missing_target.append(key)
            continue
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            shape_mismatch.append(key)
            continue
        if not bool(torch.isfinite(source_tensor).all().item()):
            raise ValueError("shape-compatible checkpoint tensor is not finite")
        compatible[key] = source_tensor
        matched_keys.append(key)
        matched_numel += int(target_tensor.numel())
        if bool(torch.equal(source_tensor, loaded_target[key])):
            loaded_value_match.append(key)
            loaded_value_match_numel += int(target_tensor.numel())
    unexpected_source = sorted(set(source) - set(target))
    target_tensor_count = len(target)
    return (
        {
            "source_tensor_count": len(source),
            "target_tensor_count": target_tensor_count,
            "matched_tensor_count": len(matched_keys),
            "matched_numel": matched_numel,
            "target_numel": target_numel,
            "matched_target_tensor_fraction": (
                len(matched_keys) / target_tensor_count
            ),
            "matched_target_numel_fraction": matched_numel / target_numel,
            "shape_mismatch_count": len(shape_mismatch),
            "missing_target_count": len(missing_target),
            "unexpected_source_count": len(unexpected_source),
            "loaded_value_match_count": len(loaded_value_match),
            "loaded_value_match_numel": loaded_value_match_numel,
            "adapted_state_keys_sha256": _keys_sha256(source),
            "matched_state_keys_sha256": _keys_sha256(matched_keys),
            "missing_target_keys_sha256": _keys_sha256(missing_target),
            "shape_mismatch_keys_sha256": _keys_sha256(shape_mismatch),
            "unexpected_source_keys_sha256": _keys_sha256(
                unexpected_source
            ),
            "loaded_value_match_keys_sha256": _keys_sha256(
                loaded_value_match
            ),
        },
        compatible,
    )


def _coverage_passes(
    evidence: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> bool:
    return (
        evidence["matched_tensor_count"] > 0
        and evidence["matched_numel"] > 0
        and evidence["matched_target_tensor_fraction"]
        >= policy["minimum_target_tensor_fraction"]
        and evidence["matched_target_numel_fraction"]
        >= policy["minimum_target_numel_fraction"]
        and evidence["loaded_value_match_count"]
        == evidence["matched_tensor_count"]
        and evidence["loaded_value_match_numel"] == evidence["matched_numel"]
    )


def _in_container_load(
    *,
    checkpoint_path: Path,
    checkpoint_size_bytes: int,
    checkpoint_sha256: str,
    overrides_path: Path,
    overrides_sha256: str,
    checkpoint_id: str,
    checkpoint_target: str,
    backbone: str,
    tao_version: str,
    container_identity: str,
    safe_global_names: tuple[str, ...],
) -> dict[str, Any]:
    """Execute TAO's DINO PTM adapter and state-loading path on CPU."""
    _verify_regular_file(
        checkpoint_path,
        expected_size_bytes=checkpoint_size_bytes,
        expected_sha256=checkpoint_sha256,
    )
    overrides_metadata = overrides_path.lstat()
    _verify_regular_file(
        overrides_path,
        expected_size_bytes=overrides_metadata.st_size,
        expected_sha256=overrides_sha256,
    )

    import torch
    from omegaconf import OmegaConf

    from nvidia_tao_pytorch.config.dino.default_config import ExperimentConfig
    from nvidia_tao_pytorch.core.utils.ptm_utils import load_pretrained_weights
    from nvidia_tao_pytorch.cv.dino.model.pl_dino_model import DINOPlModel
    from nvidia_tao_pytorch.cv.dino.model.utils import dino_parser, ptm_adapter

    policy = coverage_policy(checkpoint_target)
    safe_global_names = _validated_checkpoint_safe_global_names(
        checkpoint_id,
        checkpoint_sha256,
        safe_global_names,
    )
    if not isinstance(backbone, str) or not backbone:
        raise ValueError("registered backbone is empty")
    overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
    base = OmegaConf.structured(ExperimentConfig())
    config = OmegaConf.merge(base, OmegaConf.create(overrides))
    if str(config.model.backbone) != backbone:
        raise ValueError("effective DINO backbone disagrees with registry")

    with _weights_only_safe_globals(torch, safe_global_names):
        raw_checkpoint = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=True,
        )
    if not isinstance(raw_checkpoint, Mapping):
        raise ValueError("checkpoint root is not a mapping")

    if checkpoint_target == FULL_DETECTOR_CHECKPOINT_TARGET:
        if raw_checkpoint.get("tao_model") != "dino":
            raise ValueError("full checkpoint tao_model metadata is not dino")
        if not isinstance(raw_checkpoint.get("state_dict"), Mapping):
            raise ValueError("full checkpoint state_dict is not a mapping")
        source = load_pretrained_weights(
            raw_checkpoint,
            parser=dino_parser,
            ptm_adapter=ptm_adapter,
        )
        config.train.pretrained_model_path = str(checkpoint_path)
        config.model.pretrained_backbone_path = None
        model = DINOPlModel(config)
        target = model.model.state_dict()
        placeholder_evidence, compatible = _state_compatibility(
            torch=torch,
            source=source,
            target=target,
            loaded_target=target,
        )
        del placeholder_evidence
        merged_state = dict(target)
        merged_state.update(compatible)
        load_result = model.model.load_state_dict(
            merged_state,
            strict=False,
        )
        if load_result.missing_keys or load_result.unexpected_keys:
            raise ValueError("full DINO state load returned incompatible keys")
        loaded_target = model.model.state_dict()
        state_evidence, _ = _state_compatibility(
            torch=torch,
            source=source,
            target=target,
            loaded_target=loaded_target,
        )
        load_path = "train_shape_aware_full_detector"
        safe_path_load_count = 0
    elif checkpoint_target == BACKBONE_CHECKPOINT_TARGET:
        source = load_pretrained_weights(
            raw_checkpoint,
            parser=dino_parser,
            ptm_adapter=ptm_adapter,
        )
        config.train.pretrained_model_path = None
        config.model.pretrained_backbone_path = str(checkpoint_path)
        original_torch_load = torch.load
        safe_path_load_count = 0

        def safe_torch_load(path: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal safe_path_load_count
            if Path(path).resolve() != checkpoint_path.resolve():
                raise ValueError("unexpected checkpoint path load")
            kwargs["map_location"] = "cpu"
            kwargs["weights_only"] = True
            safe_path_load_count += 1
            return original_torch_load(path, *args, **kwargs)

        torch.load = safe_torch_load
        try:
            with _weights_only_safe_globals(torch, safe_global_names):
                model = DINOPlModel(config)
        finally:
            torch.load = original_torch_load
        if safe_path_load_count != 1:
            raise ValueError("TAO backbone path was not loaded exactly once")
        loaded_target = model.model.model.backbone[0].body.state_dict()
        state_evidence, _ = _state_compatibility(
            torch=torch,
            source=source,
            target=loaded_target,
            loaded_target=loaded_target,
        )
        load_path = "model_pretrained_backbone_path"
    else:
        raise ValueError("unsupported checkpoint target")

    checkpoint_loaded = _coverage_passes(state_evidence, policy)
    return {
        "schema_version": LOAD_SMOKE_SCHEMA_VERSION,
        "contract_version": 1,
        "checkpoint_id": checkpoint_id,
        "checkpoint_target": checkpoint_target,
        "backbone": backbone,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "checkpoint_loaded": checkpoint_loaded,
        "state_dict_compatible": checkpoint_loaded,
        "execution_backend": "docker",
        "container_identity": container_identity,
        "tao_version": tao_version,
        "device": "cpu",
        "weights_only": True,
        "weights_only_allowed_globals": list(safe_global_names),
        "merged_overrides_sha256": overrides_sha256,
        "coverage_policy": policy,
        "tao_load_path": load_path,
        "tao_load_path_executed": True,
        "safe_path_load_count": safe_path_load_count,
        **state_evidence,
    }


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _load_cli(arguments: argparse.Namespace) -> int:
    evidence = _in_container_load(
        checkpoint_path=Path(arguments.checkpoint),
        checkpoint_size_bytes=arguments.checkpoint_size,
        checkpoint_sha256=arguments.checkpoint_sha256,
        overrides_path=Path(arguments.overrides),
        overrides_sha256=arguments.overrides_sha256,
        checkpoint_id=arguments.checkpoint_id,
        checkpoint_target=arguments.checkpoint_target,
        backbone=arguments.backbone,
        tao_version=arguments.tao_version,
        container_identity=arguments.container_identity,
        safe_global_names=tuple(arguments.safe_global),
    )
    _write_json_create_only(Path(arguments.evidence), evidence)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    load = subparsers.add_parser("_load")
    load.add_argument("--checkpoint", required=True)
    load.add_argument("--checkpoint-size", type=int, required=True)
    load.add_argument("--checkpoint-sha256", required=True)
    load.add_argument("--overrides", required=True)
    load.add_argument("--overrides-sha256", required=True)
    load.add_argument("--checkpoint-id", required=True)
    load.add_argument(
        "--checkpoint-target",
        choices=sorted(SUPPORTED_CHECKPOINT_TARGETS),
        required=True,
    )
    load.add_argument("--backbone", required=True)
    load.add_argument("--tao-version", required=True)
    load.add_argument("--container-identity", required=True)
    load.add_argument(
        "--safe-global",
        action="append",
        default=[],
        choices=sorted(
            {
                name
                for policy in _CHECKPOINT_SAFE_GLOBALS.values()
                for name in policy["allowed_globals"]
            }
        ),
    )
    load.add_argument("--evidence", required=True)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    if arguments.command == "_load":
        raise SystemExit(_load_cli(arguments))
