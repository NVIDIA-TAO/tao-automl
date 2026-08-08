#!/usr/bin/env python3

"""Data-only staging for the four official Mask Grounding DINO PTMs.

Run this CLI where the destination ``/lustre`` filesystem is directly mounted,
or map the canonical ``/lustre`` publication root through an active SSHFS
mount. It uses the production NGC HTTPS client and atomic verified cache, then
atomically publishes immutable checkpoint bytes. It imports no model or
scheduler implementation and submits no job.
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
from typing import Any, Callable

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


EXPECTED_CHECKPOINT_IDS = (
    "mask_grounding_dino.commercial.swin_tiny.trainable.v1.0",
    "mask_grounding_dino.commercial.swin_tiny.trainable.v2.0",
    "mask_grounding_dino.commercial.swin_tiny.trainable.v2.1",
    "mask_grounding_dino.research.swin_tiny.trainable.v2.0",
)
DEFAULT_ENV_FILE = Path("/localhome/local-rarunachalam/.tao/config.env")
DEFAULT_CACHE_ROOT = Path(
    "/localhome/local-rarunachalam/.tao/cache/"
    "cross_model_automl_20260729/mask_grounding_dino_ptms_v1"
)
DEFAULT_LUSTRE_ROOT = Path(
    "/lustre/fsw/portfolios/edgeai/projects/"
    "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/ptms/"
    "cross_model_automl_20260729/mask_grounding_dino_v1"
)
DEFAULT_MANIFEST = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask_grounding_dino_coco2017_ptm_qualification_v1/"
    "ptm_stage_manifest.json"
)
REMOTE_MANIFEST_NAME = "ptm_stage_manifest.json"
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PTMStageError(RuntimeError):
    """The data-only PTM stage cannot be completed safely."""


@dataclass(frozen=True)
class PublishResult:
    """One verified read-only Lustre publication."""

    path: Path
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
    try:
        model = registry.to_dict()["models"]["mask_grounding_dino"]
    except KeyError as exc:
        raise PTMStageError(
            "Mask Grounding DINO is absent from the PTM registry"
        ) from exc
    records = tuple(
        sorted(
            (copy.deepcopy(item) for item in model["checkpoints"]),
            key=lambda item: item["id"],
        )
    )
    if tuple(item["id"] for item in records) != EXPECTED_CHECKPOINT_IDS:
        raise PTMStageError(
            "registry must contain exactly the four official frozen "
            "Mask Grounding DINO checkpoints"
        )
    for record in records:
        source = record.get("source", {})
        immutable = (
            f"ngc://{source.get('registry')}/{source.get('resource')}:"
            f"{source.get('version')}#{source.get('member')}"
        )
        digest = record.get("sha256")
        size = record.get("expected_size_bytes")
        sidecar = record.get("checkpoint_spec_file", {})
        if (
            record.get("status") not in {"unverified", "supported"}
            or record.get("model_family") != "mask_grounding_dino"
            or record.get("architecture") != "mask_grounding_dino"
            or record.get("backbone") != "swin_tiny_224_1k"
            or record.get("checkpoint_target")
            != "train.pretrained_model_path"
            or record.get("compatible_tao_versions") != ["==7.1.0"]
            or "grounded_instance_segmentation"
            not in record.get("task_compatibility", ())
            or source.get("provider") != "ngc"
            or source.get("registry") != "nvidia/tao"
            or source.get("official") is not True
            or source.get("immutable_identity") != immutable
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or (
                digest is not None
                and (
                    not isinstance(digest, str)
                    or _SHA256.fullmatch(digest) is None
                )
            )
            or sidecar.get("source") != "repository"
            or _SHA256.fullmatch(str(sidecar.get("sha256"))) is None
        ):
            raise PTMStageError(
                f"registry identity is incomplete: {record.get('id')}"
            )
    return records


class LustreStagePublisher:
    """Atomic create-or-verify publication into one dedicated stage root."""

    def __init__(
        self,
        root: str | Path,
        *,
        canonical_root: str | PurePosixPath | None = None,
        physical_lustre_mount: str | Path | None = None,
        enforce_lustre_prefix: bool = True,
        mount_verifier: Callable[[str | Path], bool] = os.path.ismount,
    ):
        root_input = Path(root).expanduser()
        if root_input.is_symlink():
            raise PTMStageError("PTM stage root cannot be a symlink")
        self.root = root_input.resolve()
        canonical_value = (
            PurePosixPath(str(self.root))
            if canonical_root is None
            else PurePosixPath(str(canonical_root))
        )
        if (
            not canonical_value.is_absolute()
            or ".." in canonical_value.parts
            or canonical_value == PurePosixPath("/lustre")
            or (
                enforce_lustre_prefix
                and not canonical_value.is_relative_to(
                    PurePosixPath("/lustre")
                )
            )
        ):
            raise PTMStageError(
                "canonical PTM root must be a dedicated /lustre path"
            )
        self.canonical_root = canonical_value
        if physical_lustre_mount is not None:
            mount_input = Path(physical_lustre_mount).expanduser()
            if mount_input.is_symlink():
                raise PTMStageError(
                    "physical Lustre root is not an active safe mount"
                )
            mount = mount_input.resolve()
            if (
                mount == Path("/")
                or not mount.is_dir()
                or not mount_verifier(mount)
            ):
                raise PTMStageError(
                    "physical Lustre root is not an active safe mount"
                )
            if not self.canonical_root.is_relative_to(
                PurePosixPath("/lustre")
            ):
                raise PTMStageError(
                    "mapped canonical root must be below /lustre"
                )
            relative = self.canonical_root.relative_to(
                PurePosixPath("/lustre")
            )
            expected_physical = (
                mount.joinpath(*relative.parts).resolve()
            )
            if (
                self.root != expected_physical
                or not self.root.is_relative_to(mount)
                or self.root == mount
            ):
                raise PTMStageError(
                    "physical and canonical publication roots do not "
                    "correspond"
                )
            self.physical_lustre_mount: Path | None = mount
        else:
            if (
                canonical_root is not None
                and self.root != Path(str(self.canonical_root)).resolve()
            ):
                raise PTMStageError(
                    "a distinct canonical root requires its physical "
                    "Lustre mount"
                )
            if (
                enforce_lustre_prefix
                and not str(self.root).startswith("/lustre/")
            ):
                raise PTMStageError(
                    "direct PTM publication root must be below /lustre"
                )
            self.physical_lustre_mount = None
        if self.root.exists() and (
            self.root.is_symlink() or not self.root.is_dir()
        ):
            raise PTMStageError("PTM stage root is not a regular directory")

    @classmethod
    def from_publication_roots(
        cls,
        canonical_root: str | PurePosixPath,
        *,
        physical_lustre_mount: str | Path | None = None,
        mount_verifier: Callable[[str | Path], bool] = os.path.ismount,
    ) -> "LustreStagePublisher":
        """Map canonical ``/lustre`` identity onto an optional mount root."""
        canonical = PurePosixPath(str(canonical_root))
        if physical_lustre_mount is None:
            return cls(str(canonical))
        if (
            not canonical.is_absolute()
            or ".." in canonical.parts
            or not canonical.is_relative_to(PurePosixPath("/lustre"))
            or canonical == PurePosixPath("/lustre")
        ):
            raise PTMStageError(
                "canonical PTM root must be a dedicated /lustre path"
            )
        mount = Path(physical_lustre_mount).expanduser()
        relative = canonical.relative_to(PurePosixPath("/lustre"))
        physical = mount.joinpath(*relative.parts)
        return cls(
            physical,
            canonical_root=canonical,
            physical_lustre_mount=mount,
            mount_verifier=mount_verifier,
        )

    def canonical_path(self, physical_path: str | Path) -> PurePosixPath:
        path = Path(physical_path).expanduser().resolve()
        try:
            relative = path.relative_to(self.root)
        except ValueError as exc:
            raise PTMStageError(
                "physical artifact is outside its publication root"
            ) from exc
        return self.canonical_root.joinpath(*relative.parts)

    def physical_path(self, canonical_path: str | PurePosixPath) -> Path:
        path = PurePosixPath(str(canonical_path))
        try:
            relative = path.relative_to(self.canonical_root)
        except ValueError as exc:
            raise PTMStageError(
                "canonical artifact is outside its publication root"
            ) from exc
        physical = self.root.joinpath(*relative.parts).resolve()
        if not physical.is_relative_to(self.root):
            raise PTMStageError("canonical artifact escaped its stage")
        return physical

    def checkpoint_path(
        self,
        checkpoint_id: str,
        member: str,
    ) -> Path:
        member_name = PurePosixPath(member).name
        if (
            _SAFE_COMPONENT.fullmatch(checkpoint_id) is None
            or _SAFE_COMPONENT.fullmatch(member_name) is None
        ):
            raise PTMStageError("checkpoint ID or member name is unsafe")
        destination = (self.root / checkpoint_id / member_name).resolve()
        if not destination.is_relative_to(self.root):
            raise PTMStageError("checkpoint destination escaped its stage")
        return destination

    @property
    def manifest_path(self) -> Path:
        return self.root / REMOTE_MANIFEST_NAME

    @property
    def canonical_manifest_path(self) -> PurePosixPath:
        return self.canonical_path(self.manifest_path)

    @staticmethod
    def _existing(
        path: Path,
        *,
        expected_size: int,
        expected_sha256: str,
    ) -> PublishResult | None:
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise PTMStageError(f"stage destination is unsafe: {path}")
        size = path.stat().st_size
        digest = _sha256_file(path)
        if size != expected_size or digest != expected_sha256:
            raise PTMStageError(
                f"existing stage artifact identity differs: {path}"
            )
        if path.stat().st_mode & 0o222:
            raise PTMStageError(
                f"existing stage artifact is writable: {path}"
            )
        return PublishResult(path, size, digest, True)

    def assert_recoverable_layout(
        self,
        checkpoint_paths: Sequence[Path],
    ) -> None:
        """Reject symlinks and unrelated files before network access."""
        if not self.root.exists():
            return
        allowed_files = {
            *(path.resolve() for path in checkpoint_paths),
            self.manifest_path.resolve(),
        }
        allowed_directories = {
            self.root.resolve(),
            *(path.parent.resolve() for path in checkpoint_paths),
        }
        for path in self.root.rglob("*"):
            resolved = path.resolve()
            if path.is_symlink():
                raise PTMStageError(f"stage contains a symlink: {path}")
            if path.is_file() and resolved not in allowed_files:
                raise PTMStageError(
                    f"stage contains an unexpected file: {path}"
                )
            if path.is_dir() and resolved not in allowed_directories:
                raise PTMStageError(
                    f"stage contains an unexpected directory: {path}"
                )
        if self.manifest_path.exists() and any(
            not path.exists() for path in checkpoint_paths
        ):
            raise PTMStageError(
                "completed stage manifest exists with missing checkpoints"
            )

    def _publish_chunks(
        self,
        *,
        destination: Path,
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
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (
            destination.parent.is_symlink()
            or not destination.parent.is_dir()
        ):
            raise PTMStageError(
                f"stage destination parent is unsafe: {destination.parent}"
            )
        temporary: Path | None = None
        digest = hashlib.sha256()
        size = 0
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
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
                            "publication stream exceeds the verified size"
                        )
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if size != expected_size or digest.hexdigest() != expected_sha256:
                raise PTMStageError(
                    "publication stream differs from verified cache evidence"
                )
            os.chmod(temporary, 0o444)
            os.replace(temporary, destination)
            temporary = None
            _fsync_directory(destination.parent)
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
            raise PTMStageError("atomic stage publication disappeared")
        return PublishResult(
            result.path,
            result.size_bytes,
            result.sha256,
            False,
        )

    def publish_artifact(
        self,
        artifact: VerifiedArtifact,
        *,
        destination: Path,
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

    def finalize(
        self,
        expected_files: Mapping[Path, tuple[int, str]],
    ) -> None:
        """Verify the exact file set and remove every write bit."""
        observed_files = {
            path.resolve()
            for path in self.root.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        expected = {path.resolve() for path in expected_files}
        if observed_files != expected:
            raise PTMStageError(
                "PTM stage file set differs from the frozen inventory"
            )
        for path, (size, digest) in expected_files.items():
            result = self._existing(
                path,
                expected_size=size,
                expected_sha256=digest,
            )
            if result is None:  # pragma: no cover - defensive
                raise PTMStageError(f"stage artifact disappeared: {path}")
        directories = sorted(
            (
                path
                for path in {self.root, *(p.parent for p in expected_files)}
                if path.exists()
            ),
            key=lambda item: len(item.parts),
            reverse=True,
        )
        for directory in directories:
            if directory.is_symlink() or not directory.is_dir():
                raise PTMStageError(
                    f"stage directory is unsafe: {directory}"
                )
            os.chmod(directory, 0o555)
        if any(path.stat().st_mode & 0o222 for path in expected_files):
            raise PTMStageError("a finalized PTM artifact remains writable")
        if any(path.stat().st_mode & 0o222 for path in directories):
            raise PTMStageError("a finalized PTM directory remains writable")


def _write_manifest_create_or_verify(
    path: Path,
    content: bytes,
) -> bool:
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
            raise PTMStageError("existing local stage manifest is writable")
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


def stage_official_ptms(
    *,
    registry: PTMRegistry,
    cache: AtomicArtifactCache,
    ngc_client: NGCHTTPSClient,
    publisher: LustreStagePublisher,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Download, verify, and publish every official arm without model work."""
    if not isinstance(registry, PTMRegistry):
        raise TypeError("registry must be a PTMRegistry")
    if not isinstance(cache, AtomicArtifactCache):
        raise TypeError("cache must be an AtomicArtifactCache")
    if not isinstance(ngc_client, NGCHTTPSClient):
        raise TypeError("ngc_client must be an NGCHTTPSClient")
    records = _official_records(registry)
    destinations = tuple(
        publisher.checkpoint_path(item["id"], item["source"]["member"])
        for item in records
    )
    publisher.assert_recoverable_layout(destinations)

    checkpoints = []
    cache_hits: dict[str, bool] = {}
    published_reuse: dict[str, bool] = {}
    for record, destination in zip(records, destinations, strict=True):
        checkpoint_id = record["id"]
        source = record["source"]
        reference = ngc_client.resolve_member(source)
        if (
            reference.identity != {
                "provider": "ngc",
                "registry": source["registry"],
                "resource": source["resource"],
                "version": source["version"],
                "member": source["member"],
                "immutable_identity": source["immutable_identity"],
            }
            or reference.immutable_identity != source["immutable_identity"]
        ):
            raise PTMStageError(
                f"resolved immutable member identity changed: {checkpoint_id}"
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
            expected_sha256=record.get("sha256"),
            client=ngc_client,
        )
        if (
            artifact.size_bytes != expected_size
            or artifact.source_identity_sha256 != reference.identity_sha256
            or (
                record.get("sha256") is not None
                and artifact.sha256 != record["sha256"]
            )
        ):
            raise PTMStageError(
                f"verified cache evidence changed: {checkpoint_id}"
            )
        published = publisher.publish_artifact(
            artifact,
            destination=destination,
        )
        cache_hits[checkpoint_id] = artifact.cache_hit
        published_reuse[checkpoint_id] = published.reused
        checkpoints.append(
            {
                "id": checkpoint_id,
                "path": str(publisher.canonical_path(published.path)),
                "size_bytes": published.size_bytes,
                "sha256": published.sha256,
                "immutable_source_identity": source[
                    "immutable_identity"
                ],
                "remote_read_only": True,
            }
        )

    payload = {
        "schema_version": 1,
        "model": "mask_grounding_dino",
        "registry_sha256": registry.document_sha256,
        "stage_complete": True,
        "remote_read_only": True,
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "checkpoints": checkpoints,
    }
    document = {
        **payload,
        "manifest_sha256": canonical_sha256(payload),
    }
    content = _json_bytes(document)
    remote_manifest = publisher.publish_manifest(content)
    expected_files = {
        destination: (item["size_bytes"], item["sha256"])
        for destination, item in zip(
            destinations,
            checkpoints,
            strict=True,
        )
    }
    expected_files[remote_manifest.path] = (
        remote_manifest.size_bytes,
        remote_manifest.sha256,
    )
    publisher.finalize(expected_files)
    local_reused = _write_manifest_create_or_verify(
        Path(manifest_path),
        content,
    )
    return {
        "schema_version": 1,
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "manifest_file_sha256": hashlib.sha256(content).hexdigest(),
        "manifest_sha256": document["manifest_sha256"],
        "remote_manifest_path": str(
            publisher.canonical_path(remote_manifest.path)
        ),
        "physical_manifest_path": str(remote_manifest.path),
        "checkpoint_ids": [item["id"] for item in checkpoints],
        "cache_hits": cache_hits,
        "published_reuse": published_reuse,
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--lustre-root",
        type=Path,
        default=DEFAULT_LUSTRE_ROOT,
    )
    parser.add_argument(
        "--physical-lustre-mount",
        type=Path,
        help=(
            "Optional active SSHFS mount of remote /lustre. The physical "
            "stage path is derived from --lustre-root and cannot be supplied "
            "independently."
        ),
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--registry", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    registry = load_ptm_registry(arguments.registry)
    summary = stage_official_ptms(
        registry=registry,
        cache=AtomicArtifactCache(arguments.cache_root),
        ngc_client=NGCHTTPSClient(
            _read_ngc_credential(arguments.env_file)
        ),
        publisher=LustreStagePublisher.from_publication_roots(
            arguments.lustre_root,
            physical_lustre_mount=arguments.physical_lustre_mount,
        ),
        manifest_path=arguments.manifest,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
