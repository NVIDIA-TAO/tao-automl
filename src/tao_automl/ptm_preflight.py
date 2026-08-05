# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Production checkpoint preflight for repository-owned PTM records.

The preflight is intentionally independent of ``AutoMLRunner``.  It resolves a
model/task/TAO inventory, probes and downloads exact registry members over
authenticated HTTPS, verifies immutable artifact evidence, validates the
checkpoint-specific spec, and delegates the actual framework load to an
explicit caller-provided smoke callback.

Transport credentials, signed redirect URLs, and arbitrary exception text are
never included in results, cache metadata, or provenance hashes. Callback
results are an explicit structured diagnostic contract and must be secret-free.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Protocol, Sequence
from urllib.parse import quote, urlsplit

import requests
import yaml
from yaml.constructor import ConstructorError

from tao_automl.ptm_registry import (
    PTMArtifactAdapterResolutionError,
    PTMCompatibilityResult,
    PTMQualificationResult,
    PTMRegistry,
    canonical_sha256,
    verify_packaged_resource_sha256,
)


PREFLIGHT_REPORT_SCHEMA_VERSION = 1
DEFAULT_NGC_API_BASE_URL = "https://api.ngc.nvidia.com"
_CONTENT_RANGE_TOTAL_RE = re.compile(r"/(\d+)$")
_SAFE_CACHE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class PTMPreflightConfigurationError(ValueError):
    """Raised for invalid preflight configuration rather than PTM exclusion."""


class NGCReferenceError(ValueError):
    """Raised when an NGC source cannot identify one exact member."""


class NGCTransportError(RuntimeError):
    """Safe transport failure with no credential material."""

    def __init__(
        self,
        code: str,
        reason: str,
        *,
        status_code: int | None = None,
    ):
        self.code = code
        self.reason = reason
        self.status_code = status_code
        super().__init__(reason)


class ArtifactCacheError(RuntimeError):
    """Artifact verification or atomic-cache failure."""

    def __init__(self, code: str, reason: str, details: Mapping[str, Any] | None = None):
        self.code = code
        self.reason = reason
        self.details = dict(details or {})
        super().__init__(reason)


class CheckpointSpecError(RuntimeError):
    """Checkpoint-specific YAML/spec validation failure."""

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


class ArtifactAdapterError(RuntimeError):
    """Checkpoint adaptation failure with stable, secret-free diagnostics."""

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


class NGCCredential:
    """Opaque bearer credential whose representation is always redacted."""

    __slots__ = ("_token",)

    def __init__(self, token: str):
        if not isinstance(token, str) or not token.strip():
            raise PTMPreflightConfigurationError(
                "NGC credential must be a non-empty string"
            )
        self._token = token.strip()

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        variable: str = "NGC_KEY",
    ) -> "NGCCredential":
        """Resolve a credential without copying it into diagnostics."""
        values = os.environ if environment is None else environment
        token = values.get(variable)
        if not token:
            raise PTMPreflightConfigurationError(
                f"Required NGC credential variable {variable!r} is not set"
            )
        return cls(token)

    def authorization_header(self) -> str:
        """Return the HTTPS Authorization value for the transport only."""
        return f"Bearer {self._token}"

    def __repr__(self) -> str:
        return "NGCCredential(<redacted>)"


@dataclass(frozen=True)
class CredentialProbeResult:
    """Whether an HTTPS client has credential material configured."""

    ok: bool
    code: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "code": self.code, "reason": self.reason}


@dataclass(frozen=True)
class NGCMemberReference:
    """One exact member of one exact NGC model version."""

    registry: str
    resource: str
    version: str
    member: str
    url: str
    immutable_identity: str | None = None

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "provider": "ngc",
            "registry": self.registry,
            "resource": self.resource,
            "version": self.version,
            "member": self.member,
            "immutable_identity": self.immutable_identity,
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.identity)


@dataclass(frozen=True)
class AccessProbeResult:
    """Credential/access and remote member metadata probe."""

    ok: bool
    code: str
    reason: str
    status_code: int | None
    remote_size_bytes: int | None
    etag: str | None
    exact_member_url: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "code": self.code,
            "reason": self.reason,
            "status_code": self.status_code,
            "remote_size_bytes": self.remote_size_bytes,
            "etag": self.etag,
            "exact_member_url": self.exact_member_url,
        }


def _safe_member(member: Any) -> str:
    if not isinstance(member, str) or not member.strip():
        raise NGCReferenceError("NGC member must be a non-empty string")
    text = member.strip()
    raw_parts = text.split("/")
    if (
        "\\" in text
        or text.startswith("/")
        or any(part in ("", ".", "..") for part in raw_parts)
    ):
        raise NGCReferenceError("NGC member must be a safe relative member path")
    normalized = PurePosixPath(text)
    return normalized.as_posix()


def _header(headers: Mapping[str, Any], name: str) -> str | None:
    requested = name.lower()
    for key, value in headers.items():
        if str(key).lower() == requested:
            return str(value)
    return None


def _remote_size(headers: Mapping[str, Any]) -> int | None:
    content_range = _header(headers, "Content-Range")
    if content_range:
        match = _CONTENT_RANGE_TOTAL_RE.search(content_range)
        if match:
            return int(match.group(1))
    content_length = _header(headers, "Content-Length")
    if content_length:
        try:
            size = int(content_length)
        except (TypeError, ValueError):
            return None
        return size if size >= 0 else None
    return None


class NGCHTTPSClient:
    """Minimal authenticated HTTPS client for exact NGC registry members."""

    def __init__(
        self,
        credential: NGCCredential | None,
        *,
        session: Any | None = None,
        api_base_url: str = DEFAULT_NGC_API_BASE_URL,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 120.0,
    ):
        base = str(api_base_url).rstrip("/")
        parsed_base = urlsplit(base)
        if (
            parsed_base.scheme != "https"
            or not parsed_base.netloc
            or parsed_base.username is not None
            or parsed_base.password is not None
            or parsed_base.query
            or parsed_base.fragment
        ):
            raise PTMPreflightConfigurationError(
                "NGC API base URL must be credential-free HTTPS without "
                "query or fragment"
            )
        if connect_timeout_seconds <= 0 or read_timeout_seconds <= 0:
            raise PTMPreflightConfigurationError(
                "HTTP timeouts must be positive"
            )
        self._credential = credential
        self._session = session if session is not None else requests.Session()
        self._api_base_url = base
        self._timeout = (connect_timeout_seconds, read_timeout_seconds)

    def credential_probe(self) -> CredentialProbeResult:
        if self._credential is None:
            return CredentialProbeResult(
                False,
                "credential_missing",
                "NGC credential is not configured",
            )
        return CredentialProbeResult(
            True,
            "credential_available",
            "NGC credential is configured",
        )

    def resolve_member(
        self,
        source: Mapping[str, Any],
        *,
        member: str | None = None,
        immutable_identity: str | None = None,
    ) -> NGCMemberReference:
        """Build the exact authenticated endpoint for one registry member."""
        if source.get("provider") != "ngc":
            raise NGCReferenceError(
                f"Unsupported checkpoint provider {source.get('provider')!r}"
            )
        registry = source.get("registry")
        resource = source.get("resource")
        version = source.get("version")
        if not isinstance(registry, str):
            raise NGCReferenceError("NGC registry must be '<org>/<team>'")
        registry_parts = registry.strip("/").split("/")
        if (
            len(registry_parts) != 2
            or not all(part and part not in (".", "..") for part in registry_parts)
        ):
            raise NGCReferenceError("NGC registry must be exactly '<org>/<team>'")
        if (
            not isinstance(resource, str)
            or not resource.strip()
            or "/" in resource
            or "\\" in resource
            or resource.strip() in (".", "..")
        ):
            raise NGCReferenceError("NGC resource must be one exact model name")
        if (
            not isinstance(version, str)
            or not version.strip()
            or version.strip().lower() == "latest"
            or "/" in version
            or "\\" in version
            or version.strip() in (".", "..")
        ):
            raise NGCReferenceError("NGC version must be one exact immutable version")
        exact_member = _safe_member(member if member is not None else source.get("member"))
        org, team = registry_parts
        encoded_member = "/".join(
            quote(part, safe="") for part in PurePosixPath(exact_member).parts
        )
        # NGC's exact-member alias is intentionally used here instead of a
        # version archive.  Authenticated HEAD on this route returns member
        # metadata; full GETs may redirect to signed object storage, whose URL
        # must never be copied into diagnostics.
        url = (
            f"{self._api_base_url}/v2/models/"
            f"{quote(org, safe='')}/{quote(team, safe='')}/"
            f"{quote(resource.strip(), safe='')}/versions/"
            f"{quote(version.strip(), safe='')}/files/{encoded_member}"
        )
        return NGCMemberReference(
            registry=f"{org}/{team}",
            resource=resource.strip(),
            version=version.strip(),
            member=exact_member,
            url=url,
            immutable_identity=(
                immutable_identity
                if immutable_identity is not None
                else source.get("immutable_identity")
            ),
        )

    def _headers(self, *, range_probe: bool = False) -> dict[str, str]:
        if self._credential is None:
            raise NGCTransportError(
                "credential_missing",
                "NGC credential is not configured",
            )
        headers = {
            "Authorization": self._credential.authorization_header(),
            "Accept-Encoding": "identity",
            "User-Agent": "nvidia-tao-automl-ptm-preflight/1",
        }
        if range_probe:
            headers["Range"] = "bytes=0-0"
        return headers

    @staticmethod
    def safe_transport_reason(exc: BaseException) -> str:
        """Return a fixed diagnostic that cannot leak signed URLs or secrets."""
        return (
            "NGC HTTPS request failed "
            f"({type(exc).__name__}); inspect protected transport logs"
        )

    @staticmethod
    def _status_failure(
        status_code: int,
        reference: NGCMemberReference,
    ) -> AccessProbeResult:
        if status_code in (401, 403):
            code = "access_denied"
            reason = "NGC credentials do not grant access to the exact member"
        elif status_code == 404:
            code = "member_not_found"
            reason = "The exact NGC model member was not found"
        else:
            code = "http_error"
            reason = f"NGC member probe returned HTTP {status_code}"
        return AccessProbeResult(
            False,
            code,
            reason,
            status_code,
            None,
            None,
            reference.url,
        )

    def probe_member(self, reference: NGCMemberReference) -> AccessProbeResult:
        """Probe access and size without accepting an imprecise version archive."""
        credential = self.credential_probe()
        if not credential.ok:
            return AccessProbeResult(
                False,
                credential.code,
                credential.reason,
                None,
                None,
                None,
                reference.url,
            )
        response = None
        try:
            response = self._session.head(
                reference.url,
                headers=self._headers(),
                timeout=self._timeout,
                allow_redirects=True,
            )
            status_code = int(response.status_code)
            if status_code == 405:
                response.close()
                response = self._session.get(
                    reference.url,
                    headers=self._headers(range_probe=True),
                    timeout=self._timeout,
                    allow_redirects=True,
                    stream=True,
                )
                status_code = int(response.status_code)
            if status_code not in (200, 206):
                return self._status_failure(status_code, reference)
            return AccessProbeResult(
                True,
                "accessible",
                "Exact NGC member is accessible",
                status_code,
                _remote_size(response.headers),
                _header(response.headers, "ETag"),
                reference.url,
            )
        except Exception as exc:
            return AccessProbeResult(
                False,
                "network_error",
                self.safe_transport_reason(exc),
                None,
                None,
                None,
                reference.url,
            )
        finally:
            if response is not None:
                response.close()

    def open_member_download(self, reference: NGCMemberReference) -> Any:
        """Open a full streamed response; the caller must close it."""
        try:
            response = self._session.get(
                reference.url,
                headers=self._headers(),
                timeout=self._timeout,
                allow_redirects=True,
                stream=True,
            )
        except NGCTransportError:
            raise
        except Exception as exc:
            raise NGCTransportError(
                "network_error",
                self.safe_transport_reason(exc),
            ) from exc
        status_code = int(response.status_code)
        if status_code != 200:
            response.close()
            if status_code in (401, 403):
                code = "access_denied"
                reason = "NGC credentials do not grant download access"
            elif status_code == 404:
                code = "member_not_found"
                reason = "The exact NGC model member was not found"
            else:
                code = "http_error"
                reason = f"NGC member download returned HTTP {status_code}"
            raise NGCTransportError(code, reason, status_code=status_code)
        return response


@dataclass(frozen=True)
class VerifiedArtifact:
    """One fully verified artifact accepted into the atomic cache."""

    path: Path
    cache_relative_path: str
    size_bytes: int
    sha256: str
    expected_sha256: str | None
    verification_mode: str
    cache_hit: bool
    source_identity_sha256: str

    def stable_dict(self) -> dict[str, Any]:
        return {
            "cache_relative_path": self.cache_relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "expected_sha256": self.expected_sha256,
            "verification_mode": self.verification_mode,
            "source_identity_sha256": self.source_identity_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.stable_dict()
        value.update({"path": str(self.path), "cache_hit": self.cache_hit})
        return value


_TENSOR_EVIDENCE_ALGORITHM = "sha256_sorted_key_dtype_shape_raw_bytes_v1"


@dataclass(frozen=True)
class TensorPreservationEvidence:
    """Callback evidence that checkpoint tensors were copied byte-for-byte."""

    hash_algorithm: str
    input_tensor_count: int
    output_tensor_count: int
    input_tensor_keys_sha256: str
    output_tensor_keys_sha256: str
    input_tensor_values_sha256: str
    output_tensor_values_sha256: str

    def __post_init__(self) -> None:
        if self.hash_algorithm != _TENSOR_EVIDENCE_ALGORITHM:
            raise ValueError(
                "Tensor preservation hash_algorithm must equal "
                f"{_TENSOR_EVIDENCE_ALGORITHM!r}"
            )
        for field_name in ("input_tensor_count", "output_tensor_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        for field_name in (
            "input_tensor_keys_sha256",
            "output_tensor_keys_sha256",
            "input_tensor_values_sha256",
            "output_tensor_values_sha256",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a SHA-256 digest")

    @property
    def exact(self) -> bool:
        return (
            self.input_tensor_count == self.output_tensor_count
            and self.input_tensor_keys_sha256.lower()
            == self.output_tensor_keys_sha256.lower()
            and self.input_tensor_values_sha256.lower()
            == self.output_tensor_values_sha256.lower()
        )

    def stable_dict(self) -> dict[str, Any]:
        return {
            "hash_algorithm": self.hash_algorithm,
            "input_tensor_count": self.input_tensor_count,
            "output_tensor_count": self.output_tensor_count,
            "input_tensor_keys_sha256": self.input_tensor_keys_sha256.lower(),
            "output_tensor_keys_sha256": self.output_tensor_keys_sha256.lower(),
            "input_tensor_values_sha256": self.input_tensor_values_sha256.lower(),
            "output_tensor_values_sha256": self.output_tensor_values_sha256.lower(),
            "exact": self.exact,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TensorPreservationEvidence":
        return cls(
            hash_algorithm=value["hash_algorithm"],
            input_tensor_count=value["input_tensor_count"],
            output_tensor_count=value["output_tensor_count"],
            input_tensor_keys_sha256=value["input_tensor_keys_sha256"],
            output_tensor_keys_sha256=value["output_tensor_keys_sha256"],
            input_tensor_values_sha256=value["input_tensor_values_sha256"],
            output_tensor_values_sha256=value["output_tensor_values_sha256"],
        )


@dataclass(frozen=True)
class ArtifactAdapterRequest:
    """Verified-input contract passed to an injected model-owned adapter."""

    checkpoint_id: str
    model: str
    task: str
    tao_version: str
    input_path: Path
    output_path: Path
    input_sha256: str
    input_size_bytes: int
    adapter_id: str
    adapter_type: str
    adapter_sha256: str
    recipe_sha256: str
    recipe: Mapping[str, Any] = field(repr=False)
    registry_record: Mapping[str, Any] = field(repr=False)


@dataclass(frozen=True)
class ArtifactAdapterCallbackResult:
    """Secret-free result and tensor-preservation proof from an adapter."""

    ok: bool
    code: str
    reason: str
    tensor_preservation: TensorPreservationEvidence | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool):
            raise TypeError("ArtifactAdapterCallbackResult.ok must be bool")
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("ArtifactAdapterCallbackResult.code must be non-empty")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError(
                "ArtifactAdapterCallbackResult.reason must be non-empty"
            )
        if not isinstance(self.details, Mapping):
            raise TypeError(
                "ArtifactAdapterCallbackResult.details must be a mapping"
            )
        canonical_sha256(self.details)
        if self.ok and not isinstance(
            self.tensor_preservation,
            TensorPreservationEvidence,
        ):
            raise ValueError(
                "Successful artifact adaptation requires tensor preservation evidence"
            )

    @property
    def details_sha256(self) -> str:
        return canonical_sha256(self.details)


class ArtifactAdapterCallback(Protocol):
    """Trusted model-owned implementation of a declarative adapter recipe."""

    def __call__(
        self,
        request: ArtifactAdapterRequest,
    ) -> ArtifactAdapterCallbackResult:
        ...


@dataclass(frozen=True)
class ArtifactAdaptationEvidence:
    """Immutable provenance binding recipe, input, output, and tensor proof."""

    adapter_id: str
    adapter_type: str
    adapter_sha256: str
    recipe_sha256: str
    input_sha256: str
    input_size_bytes: int
    output_sha256: str
    output_size_bytes: int
    tensor_preservation: TensorPreservationEvidence
    callback_details_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "adapter_sha256",
            "recipe_sha256",
            "input_sha256",
            "output_sha256",
            "callback_details_sha256",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a SHA-256 digest")
        for field_name in ("input_size_bytes", "output_size_bytes"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if not self.tensor_preservation.exact:
            raise ValueError("Tensor preservation evidence must be exact")

    def stable_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_type": self.adapter_type,
            "adapter_sha256": self.adapter_sha256.lower(),
            "recipe_sha256": self.recipe_sha256.lower(),
            "input_sha256": self.input_sha256.lower(),
            "input_size_bytes": self.input_size_bytes,
            "output_sha256": self.output_sha256.lower(),
            "output_size_bytes": self.output_size_bytes,
            "tensor_preservation": self.tensor_preservation.stable_dict(),
            "callback_details_sha256": self.callback_details_sha256.lower(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactAdaptationEvidence":
        return cls(
            adapter_id=value["adapter_id"],
            adapter_type=value["adapter_type"],
            adapter_sha256=value["adapter_sha256"],
            recipe_sha256=value["recipe_sha256"],
            input_sha256=value["input_sha256"],
            input_size_bytes=value["input_size_bytes"],
            output_sha256=value["output_sha256"],
            output_size_bytes=value["output_size_bytes"],
            tensor_preservation=TensorPreservationEvidence.from_dict(
                value["tensor_preservation"]
            ),
            callback_details_sha256=value["callback_details_sha256"],
        )


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_cache_component(value: str) -> str:
    sanitized = _SAFE_CACHE_COMPONENT_RE.sub("_", value).strip("._")
    if not sanitized:
        raise ArtifactCacheError("invalid_cache_key", "Artifact cache key is empty")
    return sanitized


class AtomicArtifactCache:
    """Content-verified cache that never exposes a partial final file."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(
        self,
        checkpoint_id: str,
        source_identity: Mapping[str, Any],
        member: str,
    ) -> tuple[Path, Path, Path, str, str]:
        identity_sha = canonical_sha256(source_identity)
        checkpoint_component = _safe_cache_component(checkpoint_id)
        member_name = _safe_cache_component(PurePosixPath(member).name)
        relative = (
            Path(checkpoint_component) / identity_sha[:24] / member_name
        )
        final_path = self.root / relative
        metadata_path = final_path.with_name(f"{final_path.name}.metadata.json")
        lock_path = final_path.with_name(f".{final_path.name}.lock")
        return final_path, metadata_path, lock_path, relative.as_posix(), identity_sha

    @staticmethod
    def _digest_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _read_metadata(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        return value if isinstance(value, dict) else None

    def _existing(
        self,
        final_path: Path,
        metadata_path: Path,
        relative_path: str,
        source_identity_sha: str,
        expected_size_bytes: int,
        expected_sha256: str | None,
    ) -> VerifiedArtifact | None:
        if not final_path.is_file():
            return None
        size = final_path.stat().st_size
        if size != expected_size_bytes:
            return None
        digest = self._digest_file(final_path)
        expected = expected_sha256.lower() if expected_sha256 else None
        if expected is not None:
            if digest != expected:
                return None
            mode = "registered_sha256"
        else:
            metadata = self._read_metadata(metadata_path)
            if (
                metadata is None
                or metadata.get("source_identity_sha256") != source_identity_sha
                or metadata.get("sha256") != digest
                or metadata.get("size_bytes") != size
            ):
                return None
            mode = "immutable_identity_observed_sha256"
        return VerifiedArtifact(
            path=final_path,
            cache_relative_path=relative_path,
            size_bytes=size,
            sha256=digest,
            expected_sha256=expected,
            verification_mode=mode,
            cache_hit=True,
            source_identity_sha256=source_identity_sha,
        )

    @staticmethod
    def _remove_invalid(final_path: Path, metadata_path: Path) -> None:
        for path in (final_path, metadata_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _write_metadata_atomic(path: Path, value: Mapping[str, Any]) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=".partial-metadata-",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(
                    value,
                    handle,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
            _fsync_directory(path.parent)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

    def _accept_stream(
        self,
        *,
        checkpoint_id: str,
        source_identity: Mapping[str, Any],
        member: str,
        expected_size_bytes: int,
        expected_sha256: str | None,
        chunks: Iterator[bytes],
    ) -> VerifiedArtifact:
        if (
            isinstance(expected_size_bytes, bool)
            or not isinstance(expected_size_bytes, int)
            or expected_size_bytes <= 0
        ):
            raise ArtifactCacheError(
                "invalid_expected_size",
                "Expected artifact size must be a positive integer",
            )
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str)
            or _SHA256_RE.fullmatch(expected_sha256) is None
        ):
            raise ArtifactCacheError(
                "invalid_expected_checksum",
                "Expected SHA-256 must be a 64-character hexadecimal digest",
            )
        (
            final_path,
            metadata_path,
            lock_path,
            relative_path,
            source_identity_sha,
        ) = self._paths(checkpoint_id, source_identity, member)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        expected = expected_sha256.lower() if expected_sha256 else None

        with _exclusive_lock(lock_path):
            existing = self._existing(
                final_path,
                metadata_path,
                relative_path,
                source_identity_sha,
                expected_size_bytes,
                expected,
            )
            if existing is not None:
                return existing
            self._remove_invalid(final_path, metadata_path)

            temporary: Path | None = None
            digest = hashlib.sha256()
            byte_count = 0
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=final_path.parent,
                    prefix=".partial-artifact-",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    for chunk in chunks:
                        if not isinstance(chunk, (bytes, bytearray)):
                            raise ArtifactCacheError(
                                "invalid_download_chunk",
                                "Artifact stream yielded a non-byte chunk",
                            )
                        if not chunk:
                            continue
                        byte_count += len(chunk)
                        if byte_count > expected_size_bytes:
                            raise ArtifactCacheError(
                                "size_mismatch",
                                "Downloaded artifact exceeds its registered size",
                                {
                                    "expected_size_bytes": expected_size_bytes,
                                    "observed_size_bytes": byte_count,
                                },
                            )
                        digest.update(chunk)
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())

                if byte_count != expected_size_bytes:
                    raise ArtifactCacheError(
                        "size_mismatch",
                        "Downloaded artifact size does not match the registry",
                        {
                            "expected_size_bytes": expected_size_bytes,
                            "observed_size_bytes": byte_count,
                        },
                    )
                actual_sha = digest.hexdigest()
                if expected is not None and actual_sha != expected:
                    raise ArtifactCacheError(
                        "checksum_mismatch",
                        "Downloaded artifact SHA-256 does not match the registry",
                        {
                            "expected_sha256": expected,
                            "observed_sha256": actual_sha,
                        },
                    )

                os.chmod(temporary, 0o600)
                os.replace(temporary, final_path)
                temporary = None
                _fsync_directory(final_path.parent)
                metadata = {
                    "schema_version": 1,
                    "source_identity_sha256": source_identity_sha,
                    "size_bytes": byte_count,
                    "sha256": actual_sha,
                }
                self._write_metadata_atomic(metadata_path, metadata)
                return VerifiedArtifact(
                    path=final_path,
                    cache_relative_path=relative_path,
                    size_bytes=byte_count,
                    sha256=actual_sha,
                    expected_sha256=expected,
                    verification_mode=(
                        "registered_sha256"
                        if expected is not None
                        else "immutable_identity_observed_sha256"
                    ),
                    cache_hit=False,
                    source_identity_sha256=source_identity_sha,
                )
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink()
                    except FileNotFoundError:
                        pass

    def fetch_ngc_member(
        self,
        *,
        checkpoint_id: str,
        reference: NGCMemberReference,
        expected_size_bytes: int,
        expected_sha256: str | None,
        client: NGCHTTPSClient,
    ) -> VerifiedArtifact:
        """Download and atomically accept one exact NGC member."""
        def chunks() -> Iterator[bytes]:
            response = client.open_member_download(reference)
            response_size = _remote_size(response.headers)
            try:
                if (
                    response_size is not None
                    and response_size != expected_size_bytes
                ):
                    raise ArtifactCacheError(
                        "size_mismatch",
                        "NGC download response size does not match the registry",
                        {
                            "expected_size_bytes": expected_size_bytes,
                            "response_size_bytes": response_size,
                        },
                    )
                try:
                    yield from response.iter_content(chunk_size=1024 * 1024)
                except (ArtifactCacheError, NGCTransportError):
                    raise
                except Exception as exc:
                    raise NGCTransportError(
                        "network_error",
                        client.safe_transport_reason(exc),
                    ) from exc
            finally:
                response.close()

        return self._accept_stream(
            checkpoint_id=checkpoint_id,
            source_identity=reference.identity,
            member=reference.member,
            expected_size_bytes=expected_size_bytes,
            expected_sha256=expected_sha256,
            chunks=chunks(),
        )

    def store_bytes(
        self,
        *,
        checkpoint_id: str,
        source_identity: Mapping[str, Any],
        member: str,
        content: bytes,
        expected_sha256: str,
    ) -> VerifiedArtifact:
        """Atomically materialize an already verified package resource."""
        return self._accept_stream(
            checkpoint_id=checkpoint_id,
            source_identity=source_identity,
            member=member,
            expected_size_bytes=len(content),
            expected_sha256=expected_sha256,
            chunks=iter((content,)),
        )

    def adapt_verified_artifact(
        self,
        *,
        checkpoint_id: str,
        model: str,
        task: str,
        tao_version: str,
        input_artifact: VerifiedArtifact,
        registry_record: Mapping[str, Any],
        adapter: Mapping[str, Any],
        callback: ArtifactAdapterCallback,
    ) -> tuple[VerifiedArtifact, ArtifactAdaptationEvidence]:
        """Apply a registered recipe and atomically cache its verified output.

        The registry never supplies executable code. The caller injects the
        model-owned callback, which is invoked only after the official input
        bytes have been reverified against their preflight evidence.
        """
        if not callable(callback):
            raise ArtifactAdapterError(
                "artifact_adapter_missing",
                "A registered artifact adapter requires an injected callback",
            )
        recipe = adapter["recipe"]
        output = adapter["output"]
        adapter_sha = canonical_sha256(adapter)
        recipe_sha = canonical_sha256(recipe)
        expected_output_sha = str(output["sha256"]).lower()
        expected_output_size = output["expected_size_bytes"]
        source_identity = {
            "schema_version": 1,
            "kind": "verified_checkpoint_artifact_adaptation",
            "checkpoint_id": checkpoint_id,
            "input_sha256": input_artifact.sha256.lower(),
            "input_size_bytes": input_artifact.size_bytes,
            "input_source_identity_sha256": (
                input_artifact.source_identity_sha256
            ),
            "adapter_id": adapter["id"],
            "adapter_type": adapter["adapter_type"],
            "adapter_sha256": adapter_sha,
            "recipe_sha256": recipe_sha,
            "output_sha256": expected_output_sha,
            "output_size_bytes": expected_output_size,
        }
        (
            final_path,
            metadata_path,
            lock_path,
            relative_path,
            source_identity_sha,
        ) = self._paths(
            f"{checkpoint_id}.adapted",
            source_identity,
            output["member"],
        )
        final_path.parent.mkdir(parents=True, exist_ok=True)

        with _exclusive_lock(lock_path):
            if (
                input_artifact.path.is_symlink()
                or not input_artifact.path.is_file()
            ):
                raise ArtifactAdapterError(
                    "artifact_adapter_input_missing",
                    "Verified adapter input is missing or is not a regular file",
                )
            try:
                input_size = input_artifact.path.stat().st_size
                input_sha = self._digest_file(input_artifact.path)
            except OSError as exc:
                raise ArtifactAdapterError(
                    "artifact_adapter_input_unreadable",
                    "Verified adapter input could not be reread",
                    {"exception_type": type(exc).__name__},
                ) from exc
            if (
                input_size != input_artifact.size_bytes
                or input_sha != input_artifact.sha256.lower()
            ):
                raise ArtifactAdapterError(
                    "artifact_adapter_input_mismatch",
                    "Adapter input no longer matches its verified artifact evidence",
                    {
                        "expected_size_bytes": input_artifact.size_bytes,
                        "observed_size_bytes": input_size,
                        "expected_sha256": input_artifact.sha256.lower(),
                        "observed_sha256": input_sha,
                    },
                )

            existing = self._existing(
                final_path,
                metadata_path,
                relative_path,
                source_identity_sha,
                expected_output_size,
                expected_output_sha,
            )
            if existing is not None:
                metadata = self._read_metadata(metadata_path)
                try:
                    evidence = ArtifactAdaptationEvidence.from_dict(
                        metadata["artifact_adaptation"]
                    )
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                    AttributeError,
                ):
                    evidence = None
                if (
                    evidence is not None
                    and metadata.get("source_identity_sha256")
                    == source_identity_sha
                    and evidence.adapter_id == adapter["id"]
                    and evidence.adapter_type == adapter["adapter_type"]
                    and evidence.adapter_sha256 == adapter_sha
                    and evidence.recipe_sha256 == recipe_sha
                    and evidence.input_sha256 == input_sha
                    and evidence.input_size_bytes == input_size
                    and evidence.output_sha256 == existing.sha256
                    and evidence.output_size_bytes == existing.size_bytes
                ):
                    return existing, evidence

            self._remove_invalid(final_path, metadata_path)
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=final_path.parent,
                    prefix=".partial-adapted-artifact-",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)

                request = ArtifactAdapterRequest(
                    checkpoint_id=checkpoint_id,
                    model=model,
                    task=task,
                    tao_version=tao_version,
                    input_path=input_artifact.path,
                    output_path=temporary,
                    input_sha256=input_sha,
                    input_size_bytes=input_size,
                    adapter_id=adapter["id"],
                    adapter_type=adapter["adapter_type"],
                    adapter_sha256=adapter_sha,
                    recipe_sha256=recipe_sha,
                    recipe=copy.deepcopy(dict(recipe)),
                    registry_record=copy.deepcopy(dict(registry_record)),
                )
                try:
                    result = callback(request)
                except Exception as exc:
                    raise ArtifactAdapterError(
                        "artifact_adapter_exception",
                        "Artifact adapter callback raised an exception",
                        {"exception_type": type(exc).__name__},
                    ) from exc
                if not isinstance(result, ArtifactAdapterCallbackResult):
                    raise ArtifactAdapterError(
                        "invalid_artifact_adapter_result",
                        "Artifact adapter callback must return "
                        "ArtifactAdapterCallbackResult",
                    )
                if not result.ok:
                    raise ArtifactAdapterError(
                        result.code,
                        result.reason,
                        result.details,
                    )
                tensor_evidence = result.tensor_preservation
                if tensor_evidence is None or not tensor_evidence.exact:
                    raise ArtifactAdapterError(
                        "tensor_preservation_mismatch",
                        "Artifact adapter did not preserve tensor keys and values",
                        (
                            tensor_evidence.stable_dict()
                            if tensor_evidence is not None
                            else {}
                        ),
                    )
                if temporary.is_symlink() or not temporary.is_file():
                    raise ArtifactAdapterError(
                        "adapted_output_missing",
                        "Artifact adapter did not produce a regular output file",
                    )
                output_size = temporary.stat().st_size
                output_sha = self._digest_file(temporary)
                if output_size != expected_output_size:
                    raise ArtifactAdapterError(
                        "adapted_output_size_mismatch",
                        "Adapted output size does not match the registered wrapper",
                        {
                            "expected_size_bytes": expected_output_size,
                            "observed_size_bytes": output_size,
                        },
                    )
                if output_sha != expected_output_sha:
                    raise ArtifactAdapterError(
                        "adapted_output_checksum_mismatch",
                        "Adapted output SHA-256 does not match the registered wrapper",
                        {
                            "expected_sha256": expected_output_sha,
                            "observed_sha256": output_sha,
                        },
                    )

                evidence = ArtifactAdaptationEvidence(
                    adapter_id=adapter["id"],
                    adapter_type=adapter["adapter_type"],
                    adapter_sha256=adapter_sha,
                    recipe_sha256=recipe_sha,
                    input_sha256=input_sha,
                    input_size_bytes=input_size,
                    output_sha256=output_sha,
                    output_size_bytes=output_size,
                    tensor_preservation=tensor_evidence,
                    callback_details_sha256=result.details_sha256,
                )
                os.chmod(temporary, 0o600)
                os.replace(temporary, final_path)
                temporary = None
                _fsync_directory(final_path.parent)
                self._write_metadata_atomic(
                    metadata_path,
                    {
                        "schema_version": 1,
                        "source_identity_sha256": source_identity_sha,
                        "size_bytes": output_size,
                        "sha256": output_sha,
                        "artifact_adaptation": evidence.stable_dict(),
                    },
                )
                return (
                    VerifiedArtifact(
                        path=final_path,
                        cache_relative_path=relative_path,
                        size_bytes=output_size,
                        sha256=output_sha,
                        expected_sha256=expected_output_sha,
                        verification_mode="registered_sha256",
                        cache_hit=False,
                        source_identity_sha256=source_identity_sha,
                    ),
                    evidence,
                )
            except ArtifactAdapterError:
                raise
            except OSError as exc:
                raise ArtifactAdapterError(
                    "artifact_adapter_io_error",
                    "Artifact adapter output could not be verified or cached",
                    {"exception_type": type(exc).__name__},
                ) from exc
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink()
                    except FileNotFoundError:
                        pass


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _validate_spec_tree(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise CheckpointSpecError(
                    "invalid_checkpoint_spec",
                    f"Checkpoint spec key at {path} must be a non-empty string",
                )
            _validate_spec_tree(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_spec_tree(item, f"{path}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise CheckpointSpecError(
            "invalid_checkpoint_spec",
            f"Checkpoint spec value at {path} is not finite",
        )
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise CheckpointSpecError(
            "invalid_checkpoint_spec",
            f"Checkpoint spec value at {path} has unsupported type",
        )


@dataclass(frozen=True)
class ValidatedCheckpointSpec:
    """Parsed and validated checkpoint-specific configuration."""

    path: Path
    cache_relative_path: str
    artifact_sha256: str
    document_sha256: str
    top_level_keys: tuple[str, ...]
    document: Mapping[str, Any] = field(repr=False, compare=False)

    def stable_dict(self) -> dict[str, Any]:
        return {
            "cache_relative_path": self.cache_relative_path,
            "artifact_sha256": self.artifact_sha256,
            "document_sha256": self.document_sha256,
            "top_level_keys": list(self.top_level_keys),
        }


def validate_checkpoint_spec(artifact: VerifiedArtifact) -> ValidatedCheckpointSpec:
    """Load one YAML/JSON spec safely and require a mapping root."""
    try:
        with artifact.path.open("r", encoding="utf-8") as handle:
            document = yaml.load(handle, Loader=_UniqueKeySafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CheckpointSpecError(
            "invalid_checkpoint_spec",
            "Checkpoint spec could not be parsed safely",
            {"exception_type": type(exc).__name__},
        ) from exc
    if not isinstance(document, Mapping) or not document:
        raise CheckpointSpecError(
            "invalid_checkpoint_spec",
            "Checkpoint spec must contain a non-empty mapping",
        )
    _validate_spec_tree(document)
    normalized = copy.deepcopy(dict(document))
    return ValidatedCheckpointSpec(
        path=artifact.path,
        cache_relative_path=artifact.cache_relative_path,
        artifact_sha256=artifact.sha256,
        document_sha256=canonical_sha256(normalized),
        top_level_keys=tuple(sorted(normalized)),
        document=normalized,
    )


@dataclass(frozen=True)
class CheckpointLoadSmokeRequest:
    """Secret-free contract passed to the model-specific load smoke callback."""

    checkpoint_id: str
    model: str
    task: str
    tao_version: str
    checkpoint_path: Path
    checkpoint_spec_path: Path
    checkpoint_spec: Mapping[str, Any] = field(repr=False)
    default_spec_overrides: Mapping[str, Any] = field(repr=False)
    registry_record: Mapping[str, Any] = field(repr=False)


@dataclass(frozen=True)
class CheckpointLoadSmokeResult:
    """Required secret-free result from a real model/checkpoint load attempt."""

    ok: bool
    code: str
    reason: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool):
            raise TypeError("CheckpointLoadSmokeResult.ok must be bool")
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("CheckpointLoadSmokeResult.code must be non-empty")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("CheckpointLoadSmokeResult.reason must be non-empty")
        if not isinstance(self.details, Mapping):
            raise TypeError("CheckpointLoadSmokeResult.details must be a mapping")
        canonical_sha256(self.details)

    @property
    def details_sha256(self) -> str:
        return canonical_sha256(self.details)

    def stable_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "code": self.code,
            "reason": self.reason,
            "details": copy.deepcopy(dict(self.details)),
            "details_sha256": self.details_sha256,
        }


class CheckpointLoadSmokeCallback(Protocol):
    """Callable that actually loads one checkpoint with its effective spec."""

    def __call__(
        self,
        request: CheckpointLoadSmokeRequest,
    ) -> CheckpointLoadSmokeResult:
        ...


@dataclass(frozen=True)
class PTMPreflightExclusion:
    """One registry-compatible PTM rejected by a later preflight stage."""

    checkpoint_id: str
    stage: str
    code: str
    reason: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def stable_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "stage": self.stage,
            "code": self.code,
            "reason": self.reason,
            "details": copy.deepcopy(dict(self.details)),
        }


@dataclass(frozen=True)
class PreparedPTM:
    """One checkpoint that passed access, integrity, spec, and load smoke."""

    checkpoint_id: str
    registry_status: str
    runtime_eligible: bool
    checkpoint: VerifiedArtifact
    checkpoint_spec_artifact: VerifiedArtifact
    checkpoint_spec: ValidatedCheckpointSpec
    access_probe: AccessProbeResult
    load_smoke: CheckpointLoadSmokeResult
    registry_record_sha256: str
    provenance_sha256: str
    source_checkpoint: VerifiedArtifact | None = None
    artifact_adaptation: ArtifactAdaptationEvidence | None = None

    def stable_dict(self) -> dict[str, Any]:
        value = {
            "checkpoint_id": self.checkpoint_id,
            "registry_status": self.registry_status,
            "runtime_eligible": self.runtime_eligible,
            "checkpoint": self.checkpoint.stable_dict(),
            "checkpoint_spec_artifact": self.checkpoint_spec_artifact.stable_dict(),
            "checkpoint_spec": self.checkpoint_spec.stable_dict(),
            "access_probe": self.access_probe.to_dict(),
            "load_smoke": self.load_smoke.stable_dict(),
            "registry_record_sha256": self.registry_record_sha256,
            "provenance_sha256": self.provenance_sha256,
        }
        if self.source_checkpoint is not None:
            value["source_checkpoint"] = self.source_checkpoint.stable_dict()
        if self.artifact_adaptation is not None:
            value["artifact_adaptation"] = (
                self.artifact_adaptation.stable_dict()
            )
        return value

    def to_dict(self) -> dict[str, Any]:
        value = self.stable_dict()
        value["checkpoint"] = self.checkpoint.to_dict()
        value["checkpoint_spec_artifact"] = self.checkpoint_spec_artifact.to_dict()
        if self.source_checkpoint is not None:
            value["source_checkpoint"] = self.source_checkpoint.to_dict()
        return value


@dataclass(frozen=True)
class PTMPreflightReport:
    """Deterministic preflight inventory and provenance evidence."""

    purpose: str
    validation_statuses: tuple[str, ...]
    model: str
    task: str
    tao_version: str
    registry_version: str
    registry_sha256: str
    inventory: PTMCompatibilityResult | PTMQualificationResult
    credential_probe: CredentialProbeResult
    prepared: tuple[PreparedPTM, ...]
    exclusions: tuple[PTMPreflightExclusion, ...]
    report_sha256: str

    @property
    def ok(self) -> bool:
        return bool(self.prepared)

    def stable_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PREFLIGHT_REPORT_SCHEMA_VERSION,
            "purpose": self.purpose,
            "validation_statuses": list(self.validation_statuses),
            "model": self.model,
            "task": self.task,
            "tao_version": self.tao_version,
            "registry_version": self.registry_version,
            "registry_sha256": self.registry_sha256,
            "inventory": self.inventory.to_dict(),
            "credential_probe": self.credential_probe.to_dict(),
            "prepared": [item.stable_dict() for item in self.prepared],
            "exclusions": [item.stable_dict() for item in self.exclusions],
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.stable_dict()
        value["prepared"] = [item.to_dict() for item in self.prepared]
        value["report_sha256"] = self.report_sha256
        value["ok"] = self.ok
        return value


class PTMCheckpointPreflight:
    """Resolve and preflight every compatible checkpoint deterministically."""

    def __init__(
        self,
        *,
        registry: PTMRegistry,
        cache: AtomicArtifactCache,
        ngc_client: NGCHTTPSClient,
        load_smoke: CheckpointLoadSmokeCallback,
        artifact_adapter: ArtifactAdapterCallback | None = None,
    ):
        if not isinstance(registry, PTMRegistry):
            raise TypeError("registry must be a PTMRegistry")
        if not isinstance(cache, AtomicArtifactCache):
            raise TypeError("cache must be an AtomicArtifactCache")
        if not isinstance(ngc_client, NGCHTTPSClient):
            raise TypeError("ngc_client must be an NGCHTTPSClient")
        if not callable(load_smoke):
            raise PTMPreflightConfigurationError(
                "load_smoke callback is required"
            )
        self.registry = registry
        self.cache = cache
        self.ngc_client = ngc_client
        self.load_smoke = load_smoke
        self.artifact_adapter = artifact_adapter

    @staticmethod
    def _registry_exclusions(
        inventory: PTMCompatibilityResult | PTMQualificationResult,
        *,
        purpose: str,
    ) -> list[PTMPreflightExclusion]:
        exclusions = []
        for item in inventory.excluded:
            exclusions.append(
                PTMPreflightExclusion(
                    checkpoint_id=item.checkpoint_id,
                    stage=(
                        "registry_compatibility"
                        if purpose == "runtime"
                        else "registry_qualification"
                    ),
                    code="+".join(item.codes),
                    reason="; ".join(item.reasons),
                    details={"status": item.status, "codes": list(item.codes)},
                )
            )
        return exclusions

    @staticmethod
    def _probe_size_or_raise(
        probe: AccessProbeResult,
        expected_size_bytes: int,
    ) -> None:
        if not probe.ok:
            raise NGCTransportError(
                probe.code,
                probe.reason,
                status_code=probe.status_code,
            )
        if (
            probe.remote_size_bytes is not None
            and probe.remote_size_bytes != expected_size_bytes
        ):
            raise ArtifactCacheError(
                "remote_size_mismatch",
                "NGC probe size does not match the registry",
                {
                    "expected_size_bytes": expected_size_bytes,
                    "remote_size_bytes": probe.remote_size_bytes,
                },
            )

    def _prepare_ngc_artifact(
        self,
        *,
        checkpoint_id: str,
        source: Mapping[str, Any],
        member: str,
        immutable_identity: str | None,
        expected_size_bytes: int,
        expected_sha256: str | None,
    ) -> tuple[VerifiedArtifact, AccessProbeResult]:
        reference = self.ngc_client.resolve_member(
            source,
            member=member,
            immutable_identity=immutable_identity,
        )
        probe = self.ngc_client.probe_member(reference)
        self._probe_size_or_raise(probe, expected_size_bytes)
        artifact = self.cache.fetch_ngc_member(
            checkpoint_id=checkpoint_id,
            reference=reference,
            expected_size_bytes=expected_size_bytes,
            expected_sha256=expected_sha256,
            client=self.ngc_client,
        )
        return artifact, probe

    def _prepare_checkpoint(
        self,
        record: Mapping[str, Any],
    ) -> tuple[VerifiedArtifact, AccessProbeResult]:
        source = record["source"]
        return self._prepare_ngc_artifact(
            checkpoint_id=record["id"],
            source=source,
            member=source["member"],
            immutable_identity=source.get("immutable_identity"),
            expected_size_bytes=record["expected_size_bytes"],
            expected_sha256=record.get("sha256"),
        )

    def _prepare_repository_spec(
        self,
        record: Mapping[str, Any],
        spec_record: Mapping[str, Any],
    ) -> VerifiedArtifact:
        verification = verify_packaged_resource_sha256(
            spec_record["path"],
            spec_record["sha256"],
        )
        if not verification.ok:
            raise ArtifactCacheError(
                verification.code,
                verification.reason,
                verification.to_dict(),
            )
        resource_path = PurePosixPath(spec_record["path"])
        resource = resources.files("tao_automl").joinpath(*resource_path.parts)
        content = resource.read_bytes()
        return self.cache.store_bytes(
            checkpoint_id=f"{record['id']}.spec",
            source_identity={
                "provider": "repository",
                "resource_path": spec_record["path"],
                "sha256": spec_record["sha256"].lower(),
                "provenance": spec_record["provenance"],
            },
            member=resource_path.name,
            content=content,
            expected_sha256=spec_record["sha256"],
        )

    def _prepare_checkpoint_spec(
        self,
        record: Mapping[str, Any],
    ) -> VerifiedArtifact:
        spec_record = record["checkpoint_spec_file"]
        if spec_record.get("source") == "repository":
            return self._prepare_repository_spec(record, spec_record)

        source_config = spec_record.get("source", "checkpoint_source")
        if source_config == "checkpoint_source":
            source = record["source"]
        elif isinstance(source_config, Mapping):
            source = source_config
        else:
            raise CheckpointSpecError(
                "invalid_checkpoint_spec_source",
                "Checkpoint spec source is not supported",
            )
        artifact, _ = self._prepare_ngc_artifact(
            checkpoint_id=f"{record['id']}.spec",
            source=source,
            member=spec_record["member"],
            immutable_identity=spec_record.get("immutable_identity"),
            expected_size_bytes=spec_record["expected_size_bytes"],
            expected_sha256=spec_record.get("sha256"),
        )
        return artifact

    def _run_load_smoke(
        self,
        *,
        record: Mapping[str, Any],
        model: str,
        task: str,
        tao_version: str,
        checkpoint: VerifiedArtifact,
        checkpoint_spec: ValidatedCheckpointSpec,
    ) -> CheckpointLoadSmokeResult:
        request = CheckpointLoadSmokeRequest(
            checkpoint_id=record["id"],
            model=model,
            task=task,
            tao_version=tao_version,
            checkpoint_path=checkpoint.path,
            checkpoint_spec_path=checkpoint_spec.path,
            checkpoint_spec=copy.deepcopy(dict(checkpoint_spec.document)),
            default_spec_overrides=copy.deepcopy(
                dict(record["default_spec_overrides"])
            ),
            registry_record=copy.deepcopy(dict(record)),
        )
        try:
            result = self.load_smoke(request)
        except Exception as exc:
            raise CheckpointSpecError(
                "load_smoke_exception",
                "Checkpoint load smoke callback raised an exception",
                {"exception_type": type(exc).__name__},
            ) from exc
        if not isinstance(result, CheckpointLoadSmokeResult):
            raise CheckpointSpecError(
                "invalid_load_smoke_result",
                "Load smoke callback must return CheckpointLoadSmokeResult",
            )
        if not result.ok:
            raise CheckpointSpecError(result.code, result.reason, result.details)
        return result

    @staticmethod
    def _operational_exclusion(
        checkpoint_id: str,
        stage: str,
        exc: BaseException,
    ) -> PTMPreflightExclusion:
        if isinstance(exc, NGCTransportError):
            details = (
                {"status_code": exc.status_code}
                if exc.status_code is not None
                else {}
            )
            return PTMPreflightExclusion(
                checkpoint_id, stage, exc.code, exc.reason, details
            )
        if isinstance(exc, ArtifactCacheError):
            return PTMPreflightExclusion(
                checkpoint_id, stage, exc.code, exc.reason, exc.details
            )
        if isinstance(exc, CheckpointSpecError):
            return PTMPreflightExclusion(
                checkpoint_id,
                stage,
                exc.code,
                str(exc),
                exc.details,
            )
        if isinstance(exc, ArtifactAdapterError):
            return PTMPreflightExclusion(
                checkpoint_id,
                stage,
                exc.code,
                exc.reason,
                exc.details,
            )
        if isinstance(exc, PTMArtifactAdapterResolutionError):
            return PTMPreflightExclusion(
                checkpoint_id,
                stage,
                "artifact_adapter_ambiguous",
                "Multiple registered artifact adapters match the target TAO version",
            )
        if isinstance(exc, NGCReferenceError):
            return PTMPreflightExclusion(
                checkpoint_id,
                stage,
                "invalid_ngc_reference",
                str(exc),
            )
        return PTMPreflightExclusion(
            checkpoint_id,
            stage,
            "unexpected_preflight_error",
            "Unexpected preflight exception; inspect protected logs",
            {"exception_type": type(exc).__name__},
        )

    def _run_inventory(
        self,
        *,
        model: str,
        task: str,
        tao_version: str,
        purpose: str,
        validation_statuses: tuple[str, ...],
        inventory: PTMCompatibilityResult | PTMQualificationResult,
        checkpoint_ids: tuple[str, ...],
    ) -> PTMPreflightReport:
        """Execute one already-resolved inventory through the preflight gates."""
        credential_probe = self.ngc_client.credential_probe()
        exclusions = self._registry_exclusions(inventory, purpose=purpose)
        prepared: list[PreparedPTM] = []

        for checkpoint_id in checkpoint_ids:
            record = self.registry.checkpoint(checkpoint_id)
            stage = "checkpoint_access"
            try:
                checkpoint, access_probe = self._prepare_checkpoint(record)
                source_checkpoint: VerifiedArtifact | None = None
                adaptation_evidence: ArtifactAdaptationEvidence | None = None
                stage = "artifact_adaptation"
                adapter = self.registry.artifact_adapter(
                    checkpoint_id,
                    tao_version=tao_version,
                )
                if adapter is not None:
                    if self.artifact_adapter is None:
                        raise ArtifactAdapterError(
                            "artifact_adapter_missing",
                            "A registered artifact adapter requires an "
                            "injected callback",
                            {
                                "adapter_id": adapter["id"],
                                "adapter_type": adapter["adapter_type"],
                            },
                        )
                    source_checkpoint = checkpoint
                    checkpoint, adaptation_evidence = (
                        self.cache.adapt_verified_artifact(
                            checkpoint_id=checkpoint_id,
                            model=model,
                            task=task,
                            tao_version=tao_version,
                            input_artifact=source_checkpoint,
                            registry_record=record,
                            adapter=adapter,
                            callback=self.artifact_adapter,
                        )
                    )
                stage = "checkpoint_spec"
                spec_artifact = self._prepare_checkpoint_spec(record)
                validated_spec = validate_checkpoint_spec(spec_artifact)
                stage = "load_smoke"
                smoke = self._run_load_smoke(
                    record=record,
                    model=model,
                    task=task,
                    tao_version=tao_version,
                    checkpoint=checkpoint,
                    checkpoint_spec=validated_spec,
                )
                record_sha = canonical_sha256(record)
                provenance_payload = {
                    "checkpoint_id": checkpoint_id,
                    "purpose": purpose,
                    "registry_status": record["status"],
                    "runtime_eligible": (
                        purpose == "runtime" and record["status"] == "supported"
                    ),
                    "checkpoint": checkpoint.stable_dict(),
                    "checkpoint_spec_artifact": spec_artifact.stable_dict(),
                    "checkpoint_spec": validated_spec.stable_dict(),
                    "access_probe": access_probe.to_dict(),
                    "load_smoke": smoke.stable_dict(),
                    "registry_record_sha256": record_sha,
                }
                if source_checkpoint is not None:
                    provenance_payload["source_checkpoint"] = (
                        source_checkpoint.stable_dict()
                    )
                if adaptation_evidence is not None:
                    provenance_payload["artifact_adaptation"] = (
                        adaptation_evidence.stable_dict()
                    )
                prepared.append(
                    PreparedPTM(
                        checkpoint_id=checkpoint_id,
                        registry_status=record["status"],
                        runtime_eligible=(
                            purpose == "runtime"
                            and record["status"] == "supported"
                        ),
                        checkpoint=checkpoint,
                        checkpoint_spec_artifact=spec_artifact,
                        checkpoint_spec=validated_spec,
                        access_probe=access_probe,
                        load_smoke=smoke,
                        registry_record_sha256=record_sha,
                        provenance_sha256=canonical_sha256(provenance_payload),
                        source_checkpoint=source_checkpoint,
                        artifact_adaptation=adaptation_evidence,
                    )
                )
            except Exception as exc:
                exclusions.append(
                    self._operational_exclusion(checkpoint_id, stage, exc)
                )

        prepared_tuple = tuple(sorted(prepared, key=lambda item: item.checkpoint_id))
        exclusions_tuple = tuple(
            sorted(
                exclusions,
                key=lambda item: (
                    item.checkpoint_id,
                    item.stage,
                    item.code,
                ),
            )
        )
        stable_payload = {
            "schema_version": PREFLIGHT_REPORT_SCHEMA_VERSION,
            "purpose": purpose,
            "validation_statuses": list(validation_statuses),
            "model": model,
            "task": task,
            "tao_version": tao_version,
            "registry_version": self.registry.registry_version,
            "registry_sha256": self.registry.document_sha256,
            "inventory": inventory.to_dict(),
            "credential_probe": credential_probe.to_dict(),
            "prepared": [item.stable_dict() for item in prepared_tuple],
            "exclusions": [item.stable_dict() for item in exclusions_tuple],
        }
        return PTMPreflightReport(
            purpose=purpose,
            validation_statuses=validation_statuses,
            model=model,
            task=task,
            tao_version=tao_version,
            registry_version=self.registry.registry_version,
            registry_sha256=self.registry.document_sha256,
            inventory=inventory,
            credential_probe=credential_probe,
            prepared=prepared_tuple,
            exclusions=exclusions_tuple,
            report_sha256=canonical_sha256(stable_payload),
        )

    def run(
        self,
        *,
        model: str,
        task: str,
        tao_version: str,
    ) -> PTMPreflightReport:
        """Preflight only supported, compatible runtime checkpoints."""
        inventory = self.registry.compatibility(
            model,
            tao_version=tao_version,
            task=task,
        )
        return self._run_inventory(
            model=model,
            task=task,
            tao_version=tao_version,
            purpose="runtime",
            validation_statuses=("supported",),
            inventory=inventory,
            checkpoint_ids=inventory.eligible_checkpoint_ids,
        )

    def run_qualification(
        self,
        *,
        model: str,
        task: str,
        tao_version: str,
        validation_statuses: Sequence[str],
        checkpoint_ids: Sequence[str] | None = None,
    ) -> PTMPreflightReport:
        """Explicitly preflight records for target-release qualification.

        This includes unverified/unsupported records and supported records
        whose current compatibility range excludes the target TAO version.
        A successful result is evidence for a future registry update; it never
        changes status or makes the prepared checkpoint runtime eligible.
        """
        inventory = self.registry.qualification(
            model,
            tao_version=tao_version,
            task=task,
            validation_statuses=validation_statuses,
        )
        candidates = inventory.candidate_checkpoint_ids
        if checkpoint_ids is not None:
            if isinstance(checkpoint_ids, (str, bytes)):
                raise PTMPreflightConfigurationError(
                    "checkpoint_ids must be a sequence of checkpoint IDs"
                )
            try:
                requested_values = tuple(checkpoint_ids)
            except TypeError as exc:
                raise PTMPreflightConfigurationError(
                    "checkpoint_ids must be a sequence of checkpoint IDs"
                ) from exc
            if (
                not requested_values
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in requested_values
                )
                or tuple(sorted(set(requested_values))) != requested_values
            ):
                raise PTMPreflightConfigurationError(
                    "checkpoint_ids must be non-empty, unique, and sorted"
                )
            unknown = sorted(set(requested_values) - set(candidates))
            if unknown:
                raise PTMPreflightConfigurationError(
                    "checkpoint_ids contains records outside the resolved "
                    f"qualification population: {unknown}"
                )
            candidates = requested_values
        return self._run_inventory(
            model=model,
            task=task,
            tao_version=tao_version,
            purpose="qualification",
            validation_statuses=inventory.validation_statuses,
            inventory=inventory,
            checkpoint_ids=candidates,
        )


__all__ = [
    "AccessProbeResult",
    "ArtifactAdaptationEvidence",
    "ArtifactAdapterCallback",
    "ArtifactAdapterCallbackResult",
    "ArtifactAdapterError",
    "ArtifactAdapterRequest",
    "ArtifactCacheError",
    "AtomicArtifactCache",
    "CheckpointLoadSmokeCallback",
    "CheckpointLoadSmokeRequest",
    "CheckpointLoadSmokeResult",
    "CheckpointSpecError",
    "CredentialProbeResult",
    "DEFAULT_NGC_API_BASE_URL",
    "NGCCredential",
    "NGCHTTPSClient",
    "NGCMemberReference",
    "NGCReferenceError",
    "NGCTransportError",
    "PREFLIGHT_REPORT_SCHEMA_VERSION",
    "PTMCheckpointPreflight",
    "PTMPreflightConfigurationError",
    "PTMPreflightExclusion",
    "PTMPreflightReport",
    "PreparedPTM",
    "TensorPreservationEvidence",
    "ValidatedCheckpointSpec",
    "VerifiedArtifact",
    "validate_checkpoint_spec",
]
