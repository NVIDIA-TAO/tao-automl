#!/usr/bin/env python3

"""Data-only staging for the official OneFormer PTM inventory.

The physical publication root may be a locally mounted SSHFS view of remote
Lustre.  The canonical publication root is the path visible to TAO jobs on the
cluster.  Only canonical paths enter the immutable manifest.

``--stage`` is the only action that accesses NGC.  ``--check-stage`` performs
local filesystem identity checks through the physical publication root.  This
module imports no model or scheduler implementation and submits no job.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from tao_automl.ptm_preflight import (
    AtomicArtifactCache,
    NGCCredential,
    NGCHTTPSClient,
    PTMPreflightConfigurationError,
    VerifiedArtifact,
)
from tao_automl.ptm_registry import (
    PTMRegistry,
    canonical_sha256,
    load_ptm_registry,
)

from . import campaign_contract


DEFAULT_ENV_FILE = Path("/localhome/local-rarunachalam/.tao/config.env")
DEFAULT_CACHE_ROOT = Path(
    "/localhome/local-rarunachalam/.tao/cache/"
    "cross_model_automl_20260729/oneformer_ptms_v1"
)
DEFAULT_CANONICAL_PUBLICATION_ROOT = PurePosixPath(
    "/lustre/fsw/portfolios/edgeai/projects/"
    "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/ptms/"
    "cross_model_automl_20260729/oneformer_v1"
)
DEFAULT_PHYSICAL_PUBLICATION_ROOT = Path(
    str(DEFAULT_CANONICAL_PUBLICATION_ROOT)
)
DEFAULT_MANIFEST = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "oneformer_coco2017_ptm_qualification_v1/"
    "ptm_stage_manifest.json"
)
REMOTE_MANIFEST_NAME = "ptm_stage_manifest.json"
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_MEMBER_COMPONENT = re.compile(r"^[A-Za-z0-9._+=-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PTMStageError(RuntimeError):
    """The data-only OneFormer PTM stage cannot be completed safely."""


@dataclass(frozen=True)
class PublicationPath:
    """One path represented in both the physical and canonical namespaces."""

    physical: Path
    canonical: str


@dataclass(frozen=True)
class PublishResult:
    """One verified immutable publication."""

    physical_path: Path
    canonical_path: str
    size_bytes: int
    sha256: str
    reused: bool


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _read_ngc_credential(path: str | Path) -> NGCCredential:
    """Read only ``NGC_KEY`` from the configured secrets file."""
    env_path = Path(path).expanduser().resolve()
    if not env_path.is_file() or env_path.is_symlink():
        raise PTMStageError(f"secrets file is unavailable: {env_path}")
    ngc_key: str | None = None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if _ENV_NAME.fullmatch(name) is None:
            raise PTMStageError("secrets file contains an invalid key")
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        if name == "NGC_KEY":
            ngc_key = value
    try:
        return NGCCredential.from_environment({"NGC_KEY": ngc_key or ""})
    except PTMPreflightConfigurationError as exc:
        raise PTMStageError(
            "secrets file does not contain a usable NGC_KEY"
        ) from exc


def _official_records(registry: PTMRegistry) -> tuple[dict[str, Any], ...]:
    """Return the exact campaign inventory or fail before network access."""
    if not isinstance(registry, PTMRegistry):
        raise TypeError("registry must be a PTMRegistry")
    snapshot = campaign_contract.oneformer_registry_snapshot()
    try:
        model = registry.to_dict()["models"]["oneformer"]
    except KeyError as exc:
        raise PTMStageError(
            "OneFormer is absent from the PTM registry"
        ) from exc
    records = tuple(
        sorted(
            (copy.deepcopy(item) for item in model["checkpoints"]),
            key=lambda item: item["id"],
        )
    )
    expected = tuple(
        sorted(snapshot["records"], key=lambda item: item["id"])
    )
    if (
        registry.document_sha256 != snapshot["registry_sha256"]
        or registry.registry_version != snapshot["registry_version"]
        or model.get("default_ptm") != snapshot["default_ptm"]
        or tuple(item["id"] for item in records)
        != tuple(item["id"] for item in expected)
        or len(records) != 4
    ):
        raise PTMStageError(
            "registry does not match the exact frozen OneFormer inventory"
        )
    for record, frozen in zip(records, expected, strict=True):
        source = record.get("source", {})
        sidecar = record.get("checkpoint_spec_file", {})
        immutable = (
            f"ngc://{source.get('registry')}/{source.get('resource')}:"
            f"{source.get('version')}#{source.get('member')}"
        )
        if (
            canonical_sha256(record) != frozen["registry_record_sha256"]
            or record.get("status") not in {"unverified", "supported"}
            or record.get("model_family") != "oneformer"
            or record.get("architecture") != "oneformer"
            or record.get("checkpoint_target") != "train.pretrained_model"
            or record.get("compatible_tao_versions") != ["==7.1.0"]
            or "panoptic_segmentation"
            not in record.get("task_compatibility", ())
            or source.get("provider") != "ngc"
            or source.get("registry") != "nvidia/tao"
            or source.get("official") is not True
            or source.get("immutable_identity") != immutable
            or isinstance(record.get("expected_size_bytes"), bool)
            or not isinstance(record.get("expected_size_bytes"), int)
            or record["expected_size_bytes"] <= 0
            or _SHA256.fullmatch(str(record.get("sha256", ""))) is None
            or sidecar.get("source") != "repository"
            or not str(sidecar.get("path", "")).startswith(
                "data/ptm_specs/oneformer/"
            )
            or _SHA256.fullmatch(str(sidecar.get("sha256", ""))) is None
        ):
            raise PTMStageError(
                f"registry identity is incomplete: {record.get('id')}"
            )
    return records


def _canonical_root(value: str | PurePosixPath) -> PurePosixPath:
    raw = str(value)
    path = PurePosixPath(raw)
    if (
        not path.is_absolute()
        or not raw.startswith("/lustre/")
        or path == PurePosixPath("/lustre")
        or "\\" in raw
        or any(part in {"", ".", ".."} for part in raw.split("/")[1:])
    ):
        raise PTMStageError(
            "canonical publication root must be a dedicated /lustre path"
        )
    return path


def _member_parts(member: str) -> tuple[str, ...]:
    if (
        not isinstance(member, str)
        or not member
        or member.startswith("/")
        or "\\" in member
    ):
        raise PTMStageError("checkpoint member is not a safe relative path")
    parts = tuple(member.split("/"))
    if any(
        part in {"", ".", ".."}
        or _SAFE_MEMBER_COMPONENT.fullmatch(part) is None
        for part in parts
    ):
        raise PTMStageError("checkpoint member is not a safe relative path")
    return parts


class PublicationRoot:
    """Create-or-verify publisher with physical/canonical path separation."""

    def __init__(
        self,
        physical_root: str | Path,
        canonical_root: str | PurePosixPath,
        *,
        enforce_physical_lustre_prefix: bool = False,
    ):
        physical = Path(physical_root).expanduser()
        if not physical.is_absolute():
            raise PTMStageError("physical publication root must be absolute")
        self.physical_root = physical.resolve()
        self.canonical_root = _canonical_root(canonical_root)
        if (
            self.physical_root == Path("/")
            or self.physical_root == Path("/lustre")
            or (
                enforce_physical_lustre_prefix
                and not str(self.physical_root).startswith("/lustre/")
            )
        ):
            raise PTMStageError(
                "physical publication root must be a dedicated path"
            )
        if self.physical_root.exists() and (
            self.physical_root.is_symlink()
            or not self.physical_root.is_dir()
        ):
            raise PTMStageError(
                "physical publication root is not a regular directory"
            )

    def _path(self, relative_parts: Sequence[str]) -> PublicationPath:
        physical = self.physical_root.joinpath(*relative_parts).resolve()
        if not physical.is_relative_to(self.physical_root):
            raise PTMStageError("publication path escaped its physical root")
        canonical = str(self.canonical_root.joinpath(*relative_parts))
        return PublicationPath(physical=physical, canonical=canonical)

    def checkpoint_path(
        self,
        checkpoint_id: str,
        member: str,
    ) -> PublicationPath:
        if (
            not isinstance(checkpoint_id, str)
            or _SAFE_ID.fullmatch(checkpoint_id) is None
        ):
            raise PTMStageError("checkpoint ID is unsafe")
        return self._path((checkpoint_id, *_member_parts(member)))

    @property
    def manifest_path(self) -> PublicationPath:
        return self._path((REMOTE_MANIFEST_NAME,))

    def physical_for_canonical(self, canonical: str) -> Path:
        raw = PurePosixPath(canonical)
        try:
            relative = raw.relative_to(self.canonical_root)
        except ValueError as exc:
            raise PTMStageError(
                "canonical path is outside the publication root"
            ) from exc
        if not relative.parts:
            raise PTMStageError("canonical path resolves to the broad root")
        return self._path(relative.parts).physical

    @staticmethod
    def _existing(
        path: PublicationPath,
        *,
        expected_size: int,
        expected_sha256: str,
    ) -> PublishResult | None:
        physical = path.physical
        if not physical.exists():
            return None
        if physical.is_symlink() or not physical.is_file():
            raise PTMStageError(
                f"publication destination is unsafe: {physical}"
            )
        size = physical.stat().st_size
        digest = _sha256_file(physical)
        if size != expected_size or digest != expected_sha256:
            raise PTMStageError(
                f"existing publication identity differs: {physical}"
            )
        if physical.stat().st_mode & 0o222:
            raise PTMStageError(
                f"existing publication artifact is writable: {physical}"
            )
        return PublishResult(
            physical_path=physical,
            canonical_path=path.canonical,
            size_bytes=size,
            sha256=digest,
            reused=True,
        )

    @staticmethod
    def _directory_ancestors(path: Path, root: Path) -> set[Path]:
        result = {root.resolve()}
        current = path.parent.resolve()
        while current != root.resolve():
            if not current.is_relative_to(root.resolve()):
                raise PTMStageError("publication directory escaped its root")
            result.add(current)
            current = current.parent
        return result

    def assert_recoverable_layout(
        self,
        checkpoint_paths: Sequence[PublicationPath],
    ) -> None:
        """Reject symlinks and unrelated content before NGC access."""
        if not self.physical_root.exists():
            return
        allowed_files = {
            *(path.physical.resolve() for path in checkpoint_paths),
            self.manifest_path.physical.resolve(),
        }
        allowed_directories = {self.physical_root.resolve()}
        for path in checkpoint_paths:
            allowed_directories.update(
                self._directory_ancestors(
                    path.physical,
                    self.physical_root,
                )
            )
        for path in self.physical_root.rglob("*"):
            resolved = path.resolve()
            if path.is_symlink():
                raise PTMStageError(
                    f"publication contains a symlink: {path}"
                )
            if path.is_file() and resolved not in allowed_files:
                raise PTMStageError(
                    f"publication contains an unexpected file: {path}"
                )
            if path.is_dir() and resolved not in allowed_directories:
                raise PTMStageError(
                    f"publication contains an unexpected directory: {path}"
                )
        if self.manifest_path.physical.exists() and any(
            not path.physical.exists() for path in checkpoint_paths
        ):
            raise PTMStageError(
                "completed stage manifest exists with missing checkpoints"
            )

    def _publish_chunks(
        self,
        *,
        destination: PublicationPath,
        chunks: Any,
        expected_size: int,
        expected_sha256: str,
    ) -> PublishResult:
        existing = self._existing(
            destination,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        if existing is not None:
            return existing
        destination.physical.parent.mkdir(parents=True, exist_ok=True)
        if (
            destination.physical.parent.is_symlink()
            or not destination.physical.parent.is_dir()
        ):
            raise PTMStageError(
                "publication destination parent is unsafe: "
                f"{destination.physical.parent}"
            )
        temporary: Path | None = None
        digest = hashlib.sha256()
        size = 0
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.physical.parent,
                prefix=".partial-ptm-",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                for chunk in chunks:
                    if not isinstance(chunk, (bytes, bytearray)):
                        raise PTMStageError(
                            "publication stream yielded non-byte content"
                        )
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > expected_size:
                        raise PTMStageError(
                            "publication stream exceeds verified size"
                        )
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if size != expected_size or digest.hexdigest() != expected_sha256:
                raise PTMStageError(
                    "publication stream differs from cache evidence"
                )
            os.chmod(temporary, 0o444)
            os.replace(temporary, destination.physical)
            temporary = None
            _fsync_directory(destination.physical.parent)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
        result = self._existing(
            destination,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        if result is None:  # pragma: no cover - defensive
            raise PTMStageError("atomic publication disappeared")
        return PublishResult(
            physical_path=result.physical_path,
            canonical_path=result.canonical_path,
            size_bytes=result.size_bytes,
            sha256=result.sha256,
            reused=False,
        )

    def publish_artifact(
        self,
        artifact: VerifiedArtifact,
        *,
        destination: PublicationPath,
    ) -> PublishResult:
        def chunks() -> Any:
            with artifact.path.open("rb") as stream:
                yield from iter(lambda: stream.read(1024 * 1024), b"")

        return self._publish_chunks(
            destination=destination,
            chunks=chunks(),
            expected_size=artifact.size_bytes,
            expected_sha256=artifact.sha256,
        )

    def publish_manifest(self, content: bytes) -> PublishResult:
        return self._publish_chunks(
            destination=self.manifest_path,
            chunks=iter((content,)),
            expected_size=len(content),
            expected_sha256=hashlib.sha256(content).hexdigest(),
        )

    def _verify_exact_layout(
        self,
        expected_files: Mapping[PublicationPath, tuple[int, str]],
        *,
        require_read_only_directories: bool,
    ) -> None:
        observed_files = {
            path.resolve()
            for path in self.physical_root.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        expected = {
            path.physical.resolve() for path in expected_files
        }
        if observed_files != expected:
            raise PTMStageError(
                "publication file set differs from frozen inventory"
            )
        directories = {self.physical_root}
        for path in expected_files:
            result = self._existing(
                path,
                expected_size=expected_files[path][0],
                expected_sha256=expected_files[path][1],
            )
            if result is None:  # pragma: no cover - defensive
                raise PTMStageError(
                    f"publication artifact disappeared: {path.physical}"
                )
            directories.update(
                self._directory_ancestors(
                    path.physical,
                    self.physical_root,
                )
            )
        if require_read_only_directories and any(
            directory.stat().st_mode & 0o222
            for directory in directories
        ):
            raise PTMStageError(
                "a finalized publication directory remains writable"
            )

    def finalize(
        self,
        expected_files: Mapping[PublicationPath, tuple[int, str]],
    ) -> None:
        """Verify the exact file set and remove every directory write bit."""
        self._verify_exact_layout(
            expected_files,
            require_read_only_directories=False,
        )
        directories = {self.physical_root}
        for path in expected_files:
            directories.update(
                self._directory_ancestors(
                    path.physical,
                    self.physical_root,
                )
            )
        for directory in sorted(
            directories,
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            if directory.is_symlink() or not directory.is_dir():
                raise PTMStageError(
                    f"publication directory is unsafe: {directory}"
                )
            os.chmod(directory, 0o555)
        self._verify_exact_layout(
            expected_files,
            require_read_only_directories=True,
        )


def _write_manifest_create_or_verify(path: Path, content: bytes) -> bool:
    """Create immutable local evidence or verify byte-identical reuse."""
    target = path.expanduser().resolve()
    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise PTMStageError("local stage manifest path is unsafe")
        if target.read_bytes() != content:
            raise PTMStageError(
                "existing local stage manifest differs; overwrite refused"
            )
        if target.stat().st_mode & 0o222:
            raise PTMStageError(
                "existing local stage manifest is writable"
            )
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=".partial-manifest-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, target)
        temporary = None
        _fsync_directory(target.parent)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return False


def _manifest_payload(
    *,
    registry: PTMRegistry,
    publisher: PublicationRoot,
    checkpoints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    snapshot = campaign_contract.oneformer_registry_snapshot()
    return {
        "schema_version": 1,
        "model": "oneformer",
        "task": "panoptic_segmentation",
        "registry_version": snapshot["registry_version"],
        "registry_sha256": registry.document_sha256,
        "default_ptm": snapshot["default_ptm"],
        "publication": {
            "canonical_root": str(publisher.canonical_root),
            "manifest_path": publisher.manifest_path.canonical,
            "physical_root_recorded": False,
            "path_contract": (
                "canonical_paths_for_cluster_runtime;"
                "physical_root_is_stage_and_verification_only"
            ),
        },
        "stage_complete": True,
        "remote_read_only": True,
        "cpu_model_runs": 0,
        "gpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "scheduler_jobs_submitted": 0,
        "checkpoints": [copy.deepcopy(dict(item)) for item in checkpoints],
    }


def validate_stage_manifest(
    document: Mapping[str, Any],
    *,
    registry: PTMRegistry,
    canonical_root: str | PurePosixPath,
) -> dict[str, Any]:
    """Validate the complete deterministic stage schema and registry binding."""
    value = copy.deepcopy(dict(document))
    supplied = value.pop("manifest_sha256", None)
    if supplied != canonical_sha256(value):
        raise PTMStageError("PTM stage manifest integrity failed")
    records = _official_records(registry)
    root = _canonical_root(canonical_root)
    publication = value.get("publication")
    rows = value.get("checkpoints")
    if (
        value.get("schema_version") != 1
        or value.get("model") != "oneformer"
        or value.get("task") != "panoptic_segmentation"
        or value.get("registry_version") != registry.registry_version
        or value.get("registry_sha256") != registry.document_sha256
        or value.get("default_ptm")
        != campaign_contract.oneformer_registry_snapshot()["default_ptm"]
        or not isinstance(publication, Mapping)
        or publication
        != {
            "canonical_root": str(root),
            "manifest_path": str(root / REMOTE_MANIFEST_NAME),
            "physical_root_recorded": False,
            "path_contract": (
                "canonical_paths_for_cluster_runtime;"
                "physical_root_is_stage_and_verification_only"
            ),
        }
        or value.get("stage_complete") is not True
        or value.get("remote_read_only") is not True
        or value.get("cpu_model_runs") != 0
        or value.get("gpu_model_runs") != 0
        or value.get("smoke_model_runs") != 0
        or value.get("mini_step_runs") != 0
        or value.get("scheduler_jobs_submitted") != 0
        or not isinstance(rows, list)
        or tuple(item.get("id") for item in rows)
        != tuple(record["id"] for record in records)
    ):
        raise PTMStageError("OneFormer PTM stage contract changed")
    for row, record in zip(rows, records, strict=True):
        source = record["source"]
        sidecar = record["checkpoint_spec_file"]
        expected_path = str(
            root.joinpath(record["id"], *_member_parts(source["member"]))
        )
        if (
            set(row)
            != {
                "id",
                "path",
                "size_bytes",
                "sha256",
                "mode",
                "immutable_source_identity",
                "source_identity_sha256",
                "verification_mode",
                "registry_record_sha256",
                "checkpoint_target",
                "architecture",
                "backbone",
                "checkpoint_spec_file",
                "remote_read_only",
            }
            or row.get("id") != record["id"]
            or row.get("path") != expected_path
            or row.get("size_bytes") != record["expected_size_bytes"]
            or row.get("sha256") != record["sha256"]
            or row.get("mode") != "444"
            or row.get("immutable_source_identity")
            != source["immutable_identity"]
            or row.get("source_identity_sha256")
            != canonical_sha256(
                {
                    "provider": "ngc",
                    "registry": source["registry"],
                    "resource": source["resource"],
                    "version": source["version"],
                    "member": source["member"],
                    "immutable_identity": source["immutable_identity"],
                }
            )
            or row.get("verification_mode") != "registered_sha256"
            or row.get("registry_record_sha256")
            != canonical_sha256(record)
            or row.get("checkpoint_target")
            != record["checkpoint_target"]
            or row.get("architecture") != record["architecture"]
            or row.get("backbone") != record["backbone"]
            or row.get("checkpoint_spec_file") != sidecar
            or row.get("remote_read_only") is not True
        ):
            raise PTMStageError(
                f"staged PTM identity changed: {row.get('id')}"
            )
    value["manifest_sha256"] = supplied
    return value


def stage_official_ptms(
    *,
    registry: PTMRegistry,
    cache: AtomicArtifactCache,
    ngc_client: NGCHTTPSClient,
    publisher: PublicationRoot,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Fetch, verify, and publish every official arm without model work."""
    if not isinstance(cache, AtomicArtifactCache):
        raise TypeError("cache must be an AtomicArtifactCache")
    if not isinstance(ngc_client, NGCHTTPSClient):
        raise TypeError("ngc_client must be an NGCHTTPSClient")
    local_manifest = Path(manifest_path).expanduser().resolve()
    if local_manifest.is_relative_to(publisher.physical_root):
        raise PTMStageError(
            "local evidence manifest must be outside the publication root"
        )
    records = _official_records(registry)
    destinations = tuple(
        publisher.checkpoint_path(
            item["id"],
            item["source"]["member"],
        )
        for item in records
    )
    publisher.assert_recoverable_layout(destinations)

    checkpoints = []
    cache_hits: dict[str, bool] = {}
    publication_reuse: dict[str, bool] = {}
    for record, destination in zip(records, destinations, strict=True):
        checkpoint_id = record["id"]
        source = record["source"]
        reference = ngc_client.resolve_member(source)
        expected_identity = {
            "provider": "ngc",
            "registry": source["registry"],
            "resource": source["resource"],
            "version": source["version"],
            "member": source["member"],
            "immutable_identity": source["immutable_identity"],
        }
        if (
            reference.identity != expected_identity
            or reference.immutable_identity
            != source["immutable_identity"]
        ):
            raise PTMStageError(
                f"resolved immutable member identity changed: "
                f"{checkpoint_id}"
            )
        probe = ngc_client.probe_member(reference)
        if not probe.ok:
            raise PTMStageError(
                f"NGC access failed for {checkpoint_id}: {probe.code}"
            )
        expected_size = record["expected_size_bytes"]
        if (
            probe.remote_size_bytes is not None
            and probe.remote_size_bytes != expected_size
        ):
            raise PTMStageError(
                f"NGC member size changed for {checkpoint_id}"
            )
        artifact = cache.fetch_ngc_member(
            checkpoint_id=checkpoint_id,
            reference=reference,
            expected_size_bytes=expected_size,
            expected_sha256=record["sha256"],
            client=ngc_client,
        )
        if (
            artifact.size_bytes != expected_size
            or artifact.sha256 != record["sha256"]
            or artifact.source_identity_sha256
            != reference.identity_sha256
            or artifact.verification_mode != "registered_sha256"
        ):
            raise PTMStageError(
                f"verified cache evidence changed: {checkpoint_id}"
            )
        published = publisher.publish_artifact(
            artifact,
            destination=destination,
        )
        cache_hits[checkpoint_id] = artifact.cache_hit
        publication_reuse[checkpoint_id] = published.reused
        checkpoints.append(
            {
                "id": checkpoint_id,
                "path": published.canonical_path,
                "size_bytes": published.size_bytes,
                "sha256": published.sha256,
                "mode": "444",
                "immutable_source_identity": source[
                    "immutable_identity"
                ],
                "source_identity_sha256": (
                    artifact.source_identity_sha256
                ),
                "verification_mode": artifact.verification_mode,
                "registry_record_sha256": canonical_sha256(record),
                "checkpoint_target": record["checkpoint_target"],
                "architecture": record["architecture"],
                "backbone": record["backbone"],
                "checkpoint_spec_file": copy.deepcopy(
                    record["checkpoint_spec_file"]
                ),
                "remote_read_only": True,
            }
        )

    payload = _manifest_payload(
        registry=registry,
        publisher=publisher,
        checkpoints=checkpoints,
    )
    document = {
        **payload,
        "manifest_sha256": canonical_sha256(payload),
    }
    validate_stage_manifest(
        document,
        registry=registry,
        canonical_root=publisher.canonical_root,
    )
    content = _json_bytes(document)
    remote_manifest = publisher.publish_manifest(content)
    expected_files = {
        destination: (
            checkpoint["size_bytes"],
            checkpoint["sha256"],
        )
        for destination, checkpoint in zip(
            destinations,
            checkpoints,
            strict=True,
        )
    }
    expected_files[publisher.manifest_path] = (
        remote_manifest.size_bytes,
        remote_manifest.sha256,
    )
    publisher.finalize(expected_files)
    local_reused = _write_manifest_create_or_verify(
        local_manifest,
        content,
    )
    return {
        "schema_version": 1,
        "manifest_path": str(local_manifest),
        "manifest_file_sha256": hashlib.sha256(content).hexdigest(),
        "manifest_sha256": document["manifest_sha256"],
        "physical_publication_root": str(publisher.physical_root),
        "canonical_publication_root": str(publisher.canonical_root),
        "canonical_remote_manifest_path": remote_manifest.canonical_path,
        "checkpoint_ids": [item["id"] for item in checkpoints],
        "cache_hits": cache_hits,
        "publication_reuse": publication_reuse,
        "local_manifest_reused": local_reused,
        "execution": {
            "data_only": True,
            "model_invoked": False,
            "cpu_model_runs": 0,
            "gpu_model_runs": 0,
            "smoke_model_runs": 0,
            "mini_step_runs": 0,
            "slurm_jobs_submitted": 0,
            "scheduler_client_constructed": False,
        },
    }


def verify_staged_ptms(
    *,
    registry: PTMRegistry,
    publisher: PublicationRoot,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Re-hash a stage through its physical mount without NGC or model work."""
    local_manifest = Path(manifest_path).expanduser().resolve()
    if (
        not local_manifest.is_file()
        or local_manifest.is_symlink()
        or local_manifest.stat().st_mode & 0o222
    ):
        raise PTMStageError(
            "local immutable stage manifest is unavailable"
        )
    content = local_manifest.read_bytes()
    try:
        document = json.loads(content)
    except (UnicodeDecodeError, ValueError) as exc:
        raise PTMStageError("local stage manifest is invalid JSON") from exc
    stage = validate_stage_manifest(
        document,
        registry=registry,
        canonical_root=publisher.canonical_root,
    )
    records = _official_records(registry)
    checkpoint_paths = tuple(
        publisher.checkpoint_path(
            item["id"],
            item["source"]["member"],
        )
        for item in records
    )
    publisher.assert_recoverable_layout(checkpoint_paths)
    expected_files = {
        destination: (row["size_bytes"], row["sha256"])
        for destination, row in zip(
            checkpoint_paths,
            stage["checkpoints"],
            strict=True,
        )
    }
    expected_files[publisher.manifest_path] = (
        len(content),
        hashlib.sha256(content).hexdigest(),
    )
    publisher._verify_exact_layout(
        expected_files,
        require_read_only_directories=True,
    )
    if publisher.manifest_path.physical.read_bytes() != content:
        raise PTMStageError(
            "physical and local stage manifests differ"
        )
    return {
        "schema_version": 1,
        "manifest_sha256": stage["manifest_sha256"],
        "manifest_file_sha256": hashlib.sha256(content).hexdigest(),
        "physical_publication_root": str(publisher.physical_root),
        "canonical_publication_root": str(publisher.canonical_root),
        "checkpoint_ids": [item["id"] for item in stage["checkpoints"]],
        "all_artifacts_verified": True,
        "all_artifacts_read_only": True,
        "execution": {
            "data_only": True,
            "model_invoked": False,
            "network_accessed": False,
            "cpu_model_runs": 0,
            "gpu_model_runs": 0,
            "smoke_model_runs": 0,
            "mini_step_runs": 0,
            "slurm_jobs_submitted": 0,
            "scheduler_client_constructed": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--stage", action="store_true")
    action.add_argument("--check-stage", action="store_true")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--physical-publication-root",
        type=Path,
        default=DEFAULT_PHYSICAL_PUBLICATION_ROOT,
    )
    parser.add_argument(
        "--canonical-publication-root",
        default=str(DEFAULT_CANONICAL_PUBLICATION_ROOT),
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--registry", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    registry = load_ptm_registry(arguments.registry)
    publisher = PublicationRoot(
        arguments.physical_publication_root,
        arguments.canonical_publication_root,
    )
    if arguments.stage:
        summary = stage_official_ptms(
            registry=registry,
            cache=AtomicArtifactCache(arguments.cache_root),
            ngc_client=NGCHTTPSClient(
                _read_ngc_credential(arguments.env_file)
            ),
            publisher=publisher,
            manifest_path=arguments.manifest,
        )
    else:
        summary = verify_staged_ptms(
            registry=registry,
            publisher=publisher,
            manifest_path=arguments.manifest,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
