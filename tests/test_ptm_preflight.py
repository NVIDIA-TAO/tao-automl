# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mocked-HTTPS tests for production PTM checkpoint preflight."""

from __future__ import annotations

import hashlib
import json
from importlib import resources

import pytest

from tao_automl.ptm_preflight import (
    ArtifactAdapterCallbackResult,
    AtomicArtifactCache,
    CheckpointLoadSmokeResult,
    NGCCredential,
    NGCHTTPSClient,
    NGCReferenceError,
    PTMCheckpointPreflight,
    PTMPreflightConfigurationError,
    TensorPreservationEvidence,
)
from tao_automl.ptm_registry import PTMRegistry


CHECKPOINT_BYTES = b"authoritative-checkpoint-bytes"
SPEC_BYTES = b"model:\n  backbone: resnet_50\n  num_queries: 900\n"
ADAPTED_CHECKPOINT_BYTES = b"tao71-metadata-wrapper-checkpoint-bytes"
SECRET = "test-ngc-secret-that-must-never-appear"
TENSOR_KEYS_SHA256 = hashlib.sha256(b"model.weight").hexdigest()
TENSOR_VALUES_SHA256 = hashlib.sha256(b"float32:[1]:value").hexdigest()


def _tensor_evidence(
    *,
    output_keys_sha256=TENSOR_KEYS_SHA256,
    output_values_sha256=TENSOR_VALUES_SHA256,
):
    return TensorPreservationEvidence(
        hash_algorithm="sha256_sorted_key_dtype_shape_raw_bytes_v1",
        input_tensor_count=1,
        output_tensor_count=1,
        input_tensor_keys_sha256=TENSOR_KEYS_SHA256,
        output_tensor_keys_sha256=output_keys_sha256,
        input_tensor_values_sha256=TENSOR_VALUES_SHA256,
        output_tensor_values_sha256=output_values_sha256,
    )


def _record(
    checkpoint_id="dino.resnet50.v1",
    *,
    checkpoint_bytes=CHECKPOINT_BYTES,
    spec_bytes=SPEC_BYTES,
):
    return {
        "id": checkpoint_id,
        "status": "supported",
        "source": {
            "provider": "ngc",
            "registry": "nvidia/tao",
            "resource": "pretrained_dino_coco",
            "version": "v1.0",
            "member": "checkpoints/dino_resnet50_ep12.pth",
            "official": True,
            "immutable_identity": (
                "ngc://nvidia/tao/pretrained_dino_coco:v1.0"
                "#checkpoints/dino_resnet50_ep12.pth"
            ),
        },
        "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "expected_size_bytes": len(checkpoint_bytes),
        "compatible_tao_versions": [">=7.0,<7.2"],
        "model_family": "dino",
        "architecture": "dino",
        "backbone": "resnet_50",
        "checkpoint_target": "train.pretrained_model_path",
        "input_contract": {"channels": 3, "height": None, "width": None},
        "default_spec_overrides": {
            "model": {"backbone": "resnet_50", "num_queries": 900}
        },
        "checkpoint_spec_file": {
            "source": "checkpoint_source",
            "member": "specs/train.yaml",
            "expected_size_bytes": len(spec_bytes),
            "sha256": hashlib.sha256(spec_bytes).hexdigest(),
        },
        "task_compatibility": ["object_detection"],
        "license": {
            "name": "NVIDIA TAO Model License",
            "access_requirements": ["NGC account"],
        },
        "deprecation": {"is_deprecated": False},
        "validation": {
            "status": "validated",
            "tao_version": "7.0.1",
            "container_identity": "sha256:" + "1" * 64,
            "evidence": "verified-preflight-record.json",
        },
    }


def _record_with_adapter(
    checkpoint_id="dino.resnet50.v1",
    *,
    output_bytes=ADAPTED_CHECKPOINT_BYTES,
):
    record = _record(checkpoint_id)
    record["artifact_adapters"] = [
        {
            "id": "dino.tao71.metadata_wrapper.v1",
            "adapter_type": "checkpoint_metadata_projection_v1",
            "compatible_tao_versions": ["==7.1.0"],
            "recipe": {
                "retain_top_level_keys": ["state_dict"],
                "add_top_level_metadata": {"tao_model": "dino"},
                "tensor_container_key": "state_dict",
                "require_exact_tensor_key_set": True,
                "require_exact_tensor_values": True,
            },
            "output": {
                "member": "tao71_dino_resnet50_ep12.pth",
                "expected_size_bytes": len(output_bytes),
                "sha256": hashlib.sha256(output_bytes).hexdigest(),
            },
            "provenance": {
                "source": "Preserved qualification fixture",
                "evidence": "tests/test_ptm_preflight.py",
            },
        }
    ]
    return record


def _registry(*records, default_ptm="dino.resnet50.v1"):
    return PTMRegistry(
        {
            "schema_version": 1,
            "registry_version": "preflight-test-v1",
            "models": {
                "dino": {
                    "default_ptm": default_ptm,
                    "checkpoints": list(records),
                }
            },
        }
    )


class FakeResponse:
    def __init__(
        self,
        status_code,
        content=b"",
        headers=None,
        iter_exception=None,
    ):
        self.status_code = status_code
        self.content = content
        self.headers = dict(headers or {})
        self.iter_exception = iter_exception
        self.closed = False

    def iter_content(self, chunk_size):
        del chunk_size
        if self.iter_exception is not None:
            raise self.iter_exception
        for offset in range(0, len(self.content), 5):
            yield self.content[offset:offset + 5]

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, payloads):
        self.payloads = dict(payloads)
        self.head_status = {}
        self.get_status = {}
        self.head_size = {}
        self.download_content = {}
        self.download_size_header = {}
        self.download_iter_exception = {}
        self.exceptions = {}
        self.calls = []

    def _exception(self, method, url):
        error = self.exceptions.get((method, url))
        if error is not None:
            raise error

    def head(self, url, **kwargs):
        self.calls.append(("HEAD", url, kwargs))
        self._exception("HEAD", url)
        payload = self.payloads.get(url, b"")
        status = self.head_status.get(url, 200 if url in self.payloads else 404)
        size = self.head_size.get(url, len(payload))
        return FakeResponse(
            status,
            headers={"Content-Length": str(size), "ETag": f'"{size:x}"'},
        )

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        self._exception("GET", url)
        payload = self.payloads.get(url, b"")
        headers = kwargs.get("headers", {})
        if headers.get("Range") == "bytes=0-0":
            status = self.get_status.get((url, "range"), 206)
            return FakeResponse(
                status,
                content=payload[:1],
                headers={
                    "Content-Length": "1",
                    "Content-Range": f"bytes 0-0/{len(payload)}",
                },
            )
        status = self.get_status.get((url, "download"), 200)
        content = self.download_content.get(url, payload)
        size = self.download_size_header.get(url, len(content))
        return FakeResponse(
            status,
            content=content,
            headers={"Content-Length": str(size)},
            iter_exception=self.download_iter_exception.get(url),
        )

    @property
    def download_calls(self):
        return [
            call
            for call in self.calls
            if call[0] == "GET"
            and call[2].get("headers", {}).get("Range") is None
        ]


def _client_and_urls(record, session=None, credential=True):
    client = NGCHTTPSClient(
        NGCCredential(SECRET) if credential else None,
        session=session,
        api_base_url="https://ngc.example.test",
    )
    checkpoint = client.resolve_member(record["source"])
    spec_record = record["checkpoint_spec_file"]
    spec_url = None
    if spec_record.get("source") != "repository":
        spec_url = client.resolve_member(
            record["source"],
            member=spec_record["member"],
        ).url
    return client, checkpoint.url, spec_url


def _successful_smoke(request):
    assert request.checkpoint_path.read_bytes() == CHECKPOINT_BYTES
    assert request.checkpoint_spec["model"]["backbone"] == "resnet_50"
    assert request.default_spec_overrides["model"]["num_queries"] == 900
    assert request.registry_record["id"] == request.checkpoint_id
    return CheckpointLoadSmokeResult(
        True,
        "loaded",
        "Checkpoint loaded successfully",
        {"framework": "mock-tao", "finite_parameters": True},
    )


def _preflight(
    tmp_path,
    record,
    session,
    smoke=_successful_smoke,
    credential=True,
    artifact_adapter=None,
):
    client, checkpoint_url, spec_url = _client_and_urls(
        record,
        session=session,
        credential=credential,
    )
    runner = PTMCheckpointPreflight(
        registry=_registry(
            record,
            default_ptm=(
                record["id"] if record["status"] == "supported" else None
            ),
        ),
        cache=AtomicArtifactCache(tmp_path / "cache"),
        ngc_client=client,
        load_smoke=smoke,
        artifact_adapter=artifact_adapter,
    )
    return runner, checkpoint_url, spec_url


def test_exact_member_preflight_is_atomic_secret_free_and_deterministic(tmp_path):
    record = _record()
    client, checkpoint_url, spec_url = _client_and_urls(record)
    assert checkpoint_url == (
        "https://ngc.example.test/v2/models/nvidia/tao/"
        "pretrained_dino_coco/versions/v1.0/files/"
        "checkpoints/dino_resnet50_ep12.pth"
    )
    assert spec_url.endswith("/versions/v1.0/files/specs/train.yaml")
    session = FakeSession(
        {checkpoint_url: CHECKPOINT_BYTES, spec_url: SPEC_BYTES}
    )
    runner, _, _ = _preflight(tmp_path, record, session)

    first = runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.0.1-pyt",
    )
    assert first.ok
    assert not first.exclusions
    assert len(first.prepared) == 1
    prepared = first.prepared[0]
    assert prepared.checkpoint.sha256 == hashlib.sha256(
        CHECKPOINT_BYTES
    ).hexdigest()
    assert prepared.checkpoint_spec.document["model"]["num_queries"] == 900
    assert len(prepared.provenance_sha256) == 64
    assert len(first.report_sha256) == 64
    assert first.report_sha256 == hashlib.sha256(
        json.dumps(
            first.stable_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert len(session.download_calls) == 2

    for method, _, kwargs in session.calls:
        assert kwargs["headers"]["Authorization"] == f"Bearer {SECRET}"
        assert kwargs["headers"]["Accept-Encoding"] == "identity"
        assert method in {"HEAD", "GET"}
    serialized = json.dumps(first.to_dict(), sort_keys=True)
    assert SECRET not in serialized
    assert SECRET not in repr(NGCCredential(SECRET))
    assert not list((tmp_path / "cache").rglob(".partial-*"))

    second = runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.0.1-pyt",
    )
    assert second.report_sha256 == first.report_sha256
    assert (
        second.prepared[0].provenance_sha256
        == first.prepared[0].provenance_sha256
    )
    assert second.prepared[0].checkpoint.cache_hit
    assert second.prepared[0].checkpoint_spec_artifact.cache_hit
    assert len(session.download_calls) == 2


def test_registered_adapter_is_atomic_cached_and_reaches_load_smoke(tmp_path):
    record = _record_with_adapter()
    client, checkpoint_url, spec_url = _client_and_urls(record)
    del client
    session = FakeSession(
        {checkpoint_url: CHECKPOINT_BYTES, spec_url: SPEC_BYTES}
    )
    requests = []
    smoke_paths = []

    def adapter(request):
        requests.append(request)
        assert request.input_path.read_bytes() == CHECKPOINT_BYTES
        assert request.input_sha256 == hashlib.sha256(
            CHECKPOINT_BYTES
        ).hexdigest()
        assert request.input_size_bytes == len(CHECKPOINT_BYTES)
        assert request.recipe == {
            "retain_top_level_keys": ["state_dict"],
            "add_top_level_metadata": {"tao_model": "dino"},
            "tensor_container_key": "state_dict",
            "require_exact_tensor_key_set": True,
            "require_exact_tensor_values": True,
        }
        request.output_path.write_bytes(ADAPTED_CHECKPOINT_BYTES)
        return ArtifactAdapterCallbackResult(
            True,
            "adapted",
            "Checkpoint metadata wrapper created",
            _tensor_evidence(),
            {"framework": "mock-torch"},
        )

    def smoke(request):
        smoke_paths.append(request.checkpoint_path)
        assert request.checkpoint_path.read_bytes() == ADAPTED_CHECKPOINT_BYTES
        return CheckpointLoadSmokeResult(
            True,
            "loaded",
            "Adapted checkpoint reached load smoke",
        )

    runner, _, _ = _preflight(
        tmp_path,
        record,
        session,
        smoke=smoke,
        artifact_adapter=adapter,
    )
    first = runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.1.0-rc245",
    )
    assert first.ok
    assert len(requests) == 1
    prepared = first.prepared[0]
    assert prepared.source_checkpoint is not None
    assert prepared.source_checkpoint.sha256 == hashlib.sha256(
        CHECKPOINT_BYTES
    ).hexdigest()
    assert prepared.checkpoint.sha256 == hashlib.sha256(
        ADAPTED_CHECKPOINT_BYTES
    ).hexdigest()
    assert prepared.checkpoint.path == smoke_paths[0]
    assert prepared.artifact_adaptation is not None
    evidence = prepared.artifact_adaptation
    assert evidence.adapter_sha256 == requests[0].adapter_sha256
    assert evidence.recipe_sha256 == requests[0].recipe_sha256
    assert evidence.input_sha256 == prepared.source_checkpoint.sha256
    assert evidence.input_size_bytes == len(CHECKPOINT_BYTES)
    assert evidence.output_sha256 == prepared.checkpoint.sha256
    assert evidence.output_size_bytes == len(ADAPTED_CHECKPOINT_BYTES)
    assert evidence.tensor_preservation.exact
    assert not list((tmp_path / "cache").rglob(".partial-*"))
    assert not list(
        (tmp_path / "cache").rglob(".partial-adapted-artifact-*")
    )

    second = runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.1.0-rc245",
    )
    assert second.ok
    assert len(requests) == 1
    assert second.prepared[0].checkpoint.cache_hit
    assert second.prepared[0].artifact_adaptation == evidence
    assert second.report_sha256 == first.report_sha256


def test_registered_adapter_missing_or_mismatched_is_structural_exclusion(
    tmp_path,
):
    record = _record_with_adapter()
    client, checkpoint_url, spec_url = _client_and_urls(record)
    del client

    missing_session = FakeSession(
        {checkpoint_url: CHECKPOINT_BYTES, spec_url: SPEC_BYTES}
    )
    smoke_called = False

    def smoke(_request):
        nonlocal smoke_called
        smoke_called = True
        return CheckpointLoadSmokeResult(True, "loaded", "loaded")

    missing_runner, _, _ = _preflight(
        tmp_path / "missing",
        record,
        missing_session,
        smoke=smoke,
    )
    missing = missing_runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.1.0",
    )
    assert not missing.ok
    assert missing.exclusions[0].stage == "artifact_adaptation"
    assert missing.exclusions[0].code == "artifact_adapter_missing"
    assert not smoke_called

    mismatch_session = FakeSession(
        {checkpoint_url: CHECKPOINT_BYTES, spec_url: SPEC_BYTES}
    )

    def mismatched_output(request):
        request.output_path.write_bytes(b"x" * len(ADAPTED_CHECKPOINT_BYTES))
        return ArtifactAdapterCallbackResult(
            True,
            "adapted",
            "Fixture output produced",
            _tensor_evidence(),
        )

    mismatch_runner, _, _ = _preflight(
        tmp_path / "output-mismatch",
        record,
        mismatch_session,
        smoke=smoke,
        artifact_adapter=mismatched_output,
    )
    mismatch = mismatch_runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.1.0",
    )
    assert not mismatch.ok
    assert mismatch.exclusions[0].stage == "artifact_adaptation"
    assert mismatch.exclusions[0].code == "adapted_output_checksum_mismatch"
    assert not list(
        (tmp_path / "output-mismatch").rglob(
            ".partial-adapted-artifact-*"
        )
    )

    tensor_session = FakeSession(
        {checkpoint_url: CHECKPOINT_BYTES, spec_url: SPEC_BYTES}
    )

    def mismatched_tensor_keys(request):
        request.output_path.write_bytes(ADAPTED_CHECKPOINT_BYTES)
        return ArtifactAdapterCallbackResult(
            True,
            "adapted",
            "Fixture output produced",
            _tensor_evidence(output_keys_sha256="f" * 64),
        )

    tensor_runner, _, _ = _preflight(
        tmp_path / "tensor-mismatch",
        record,
        tensor_session,
        smoke=smoke,
        artifact_adapter=mismatched_tensor_keys,
    )
    tensor_mismatch = tensor_runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.1.0",
    )
    assert not tensor_mismatch.ok
    assert tensor_mismatch.exclusions[0].code == (
        "tensor_preservation_mismatch"
    )
    assert not smoke_called


def test_adapted_qualification_remains_runtime_ineligible(tmp_path):
    record = _record_with_adapter("dino.pending")
    record["status"] = "unverified"
    record["status_reason"] = "TAO 7.1 load qualification is pending"
    record.pop("validation")
    record.pop("compatible_tao_versions")
    client, checkpoint_url, spec_url = _client_and_urls(record)
    del client
    session = FakeSession(
        {checkpoint_url: CHECKPOINT_BYTES, spec_url: SPEC_BYTES}
    )

    def adapter(request):
        request.output_path.write_bytes(ADAPTED_CHECKPOINT_BYTES)
        return ArtifactAdapterCallbackResult(
            True,
            "adapted",
            "Checkpoint metadata wrapper created",
            _tensor_evidence(),
        )

    def smoke(request):
        assert request.checkpoint_path.read_bytes() == ADAPTED_CHECKPOINT_BYTES
        return CheckpointLoadSmokeResult(True, "loaded", "loaded")

    runner, _, _ = _preflight(
        tmp_path,
        record,
        session,
        smoke=smoke,
        artifact_adapter=adapter,
    )
    report = runner.run_qualification(
        model="dino",
        task="object_detection",
        tao_version="7.1.0",
        validation_statuses=("unverified",),
    )
    assert report.ok
    assert report.prepared[0].runtime_eligible is False
    assert report.prepared[0].artifact_adaptation is not None
    assert report.to_dict()["prepared"][0]["runtime_eligible"] is False


def test_credential_probe_and_access_denial_are_structured(tmp_path):
    record = _record()
    session = FakeSession({})
    runner, checkpoint_url, _ = _preflight(
        tmp_path,
        record,
        session,
        credential=False,
    )
    report = runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.0.1",
    )
    assert not report.ok
    assert report.credential_probe.code == "credential_missing"
    assert report.exclusions[0].code == "credential_missing"
    assert not session.calls

    client, checkpoint_url, spec_url = _client_and_urls(record)
    del client
    denied_session = FakeSession(
        {checkpoint_url: CHECKPOINT_BYTES, spec_url: SPEC_BYTES}
    )
    denied_session.head_status[checkpoint_url] = 403
    denied_runner, _, _ = _preflight(tmp_path / "denied", record, denied_session)
    denied = denied_runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.0.1",
    )
    assert denied.exclusions[0].code == "access_denied"
    assert denied.exclusions[0].details == {"status_code": 403}


def test_head_405_uses_one_byte_access_probe(tmp_path):
    record = _record()
    client, checkpoint_url, spec_url = _client_and_urls(record)
    del client
    session = FakeSession(
        {checkpoint_url: CHECKPOINT_BYTES, spec_url: SPEC_BYTES}
    )
    session.head_status[checkpoint_url] = 405
    runner, _, _ = _preflight(tmp_path, record, session)
    report = runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.0.1",
    )
    assert report.ok
    range_calls = [
        call
        for call in session.calls
        if call[0] == "GET"
        and call[2]["headers"].get("Range") == "bytes=0-0"
    ]
    assert len(range_calls) == 1
    assert report.prepared[0].access_probe.remote_size_bytes == len(
        CHECKPOINT_BYTES
    )


def test_remote_size_mismatch_excludes_before_download(tmp_path):
    record = _record()
    client, checkpoint_url, spec_url = _client_and_urls(record)
    del client
    session = FakeSession(
        {checkpoint_url: CHECKPOINT_BYTES, spec_url: SPEC_BYTES}
    )
    session.head_size[checkpoint_url] = len(CHECKPOINT_BYTES) + 1
    runner, _, _ = _preflight(tmp_path, record, session)
    report = runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.0.1",
    )
    exclusion = report.exclusions[0]
    assert exclusion.stage == "checkpoint_access"
    assert exclusion.code == "remote_size_mismatch"
    assert not session.download_calls


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("truncated", "size_mismatch"),
        ("checksum", "checksum_mismatch"),
    ],
)
def test_corrupt_download_never_becomes_a_cache_artifact(
    tmp_path,
    mutation,
    expected_code,
):
    record = _record()
    client, checkpoint_url, spec_url = _client_and_urls(record)
    del client
    session = FakeSession(
        {checkpoint_url: CHECKPOINT_BYTES, spec_url: SPEC_BYTES}
    )
    if mutation == "truncated":
        session.download_content[checkpoint_url] = CHECKPOINT_BYTES[:-3]
        session.download_size_header[checkpoint_url] = len(CHECKPOINT_BYTES)
    else:
        record["sha256"] = "0" * 64
    runner, _, _ = _preflight(tmp_path, record, session)
    report = runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.0.1",
    )
    assert report.exclusions[0].code == expected_code
    cache_files = [
        path
        for path in (tmp_path / "cache").rglob("*")
        if path.is_file()
        and not path.name.endswith(".lock")
        and not path.name.endswith(".metadata.json")
    ]
    assert cache_files == []
    assert not list((tmp_path / "cache").rglob(".partial-*"))


def test_immutable_identity_computes_and_reuses_observed_digest(tmp_path):
    record = _record()
    record.pop("sha256")
    client, checkpoint_url, spec_url = _client_and_urls(record)
    del client
    session = FakeSession(
        {checkpoint_url: CHECKPOINT_BYTES, spec_url: SPEC_BYTES}
    )
    runner, _, _ = _preflight(tmp_path, record, session)
    first = runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.0.1",
    )
    artifact = first.prepared[0].checkpoint
    assert artifact.verification_mode == "immutable_identity_observed_sha256"
    assert artifact.expected_sha256 is None
    assert artifact.sha256 == hashlib.sha256(CHECKPOINT_BYTES).hexdigest()

    second = runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.0.1",
    )
    assert second.prepared[0].checkpoint.cache_hit
    assert len(session.download_calls) == 2
    assert second.report_sha256 == first.report_sha256


def test_corrupted_immutable_cache_is_never_reused(tmp_path):
    record = _record()
    record.pop("sha256")
    client, checkpoint_url, spec_url = _client_and_urls(record)
    del client
    session = FakeSession(
        {checkpoint_url: CHECKPOINT_BYTES, spec_url: SPEC_BYTES}
    )
    runner, _, _ = _preflight(tmp_path, record, session)
    first = runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.0.1",
    )
    checkpoint = first.prepared[0].checkpoint.path
    checkpoint.write_bytes(b"x" * len(CHECKPOINT_BYTES))

    second = runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.0.1",
    )
    assert second.ok
    assert checkpoint.read_bytes() == CHECKPOINT_BYTES
    assert not second.prepared[0].checkpoint.cache_hit
    assert len(session.download_calls) == 3
    assert not list((tmp_path / "cache").rglob(".partial-*"))


def test_repository_sidecar_is_verified_materialized_and_parsed(tmp_path):
    record = _record()
    packaged = resources.files("tao_automl").joinpath(
        "data", "ptm_registry.v1.json"
    )
    packaged_bytes = packaged.read_bytes()
    record["checkpoint_spec_file"] = {
        "source": "repository",
        "path": "data/ptm_registry.v1.json",
        "sha256": hashlib.sha256(packaged_bytes).hexdigest(),
        "provenance": {
            "source": "repository test fixture",
            "evidence": "tests/test_ptm_preflight.py",
        },
    }
    client, checkpoint_url, _ = _client_and_urls(_record())
    del client
    session = FakeSession({checkpoint_url: CHECKPOINT_BYTES})

    def repository_spec_smoke(request):
        assert request.checkpoint_spec["schema_version"] == 1
        return CheckpointLoadSmokeResult(
            True,
            "loaded",
            "Repository sidecar reached the load contract",
        )

    runner, _, _ = _preflight(
        tmp_path,
        record,
        session,
        smoke=repository_spec_smoke,
    )
    report = runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.0.1",
    )
    assert report.ok
    spec = report.prepared[0].checkpoint_spec
    assert spec.document["schema_version"] == 1
    assert spec.path.is_file()
    assert len(session.download_calls) == 1


def test_invalid_checkpoint_yaml_is_excluded_before_load_smoke(tmp_path):
    invalid_spec = b"model:\n  backbone: resnet_50\n  backbone: duplicate\n"
    record = _record(spec_bytes=invalid_spec)
    client, checkpoint_url, spec_url = _client_and_urls(record)
    del client
    session = FakeSession(
        {checkpoint_url: CHECKPOINT_BYTES, spec_url: invalid_spec}
    )
    smoke_called = False

    def smoke(_request):
        nonlocal smoke_called
        smoke_called = True
        return CheckpointLoadSmokeResult(True, "loaded", "loaded")

    runner, _, _ = _preflight(tmp_path, record, session, smoke=smoke)
    report = runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.0.1",
    )
    assert report.exclusions[0].stage == "checkpoint_spec"
    assert report.exclusions[0].code == "invalid_checkpoint_spec"
    assert report.exclusions[0].reason == (
        "Checkpoint spec could not be parsed safely"
    )
    assert report.exclusions[0].details == {
        "exception_type": "ConstructorError"
    }
    assert not smoke_called


def test_load_smoke_contract_is_required_and_failure_is_preserved(tmp_path):
    record = _record()
    client, checkpoint_url, spec_url = _client_and_urls(record)
    session = FakeSession(
        {checkpoint_url: CHECKPOINT_BYTES, spec_url: SPEC_BYTES}
    )
    with pytest.raises(PTMPreflightConfigurationError, match="required"):
        PTMCheckpointPreflight(
            registry=_registry(record),
            cache=AtomicArtifactCache(tmp_path / "missing-smoke"),
            ngc_client=client,
            load_smoke=None,
        )

    def failed_smoke(_request):
        return CheckpointLoadSmokeResult(
            False,
            "checkpoint_load_failed",
            "State dictionary is incompatible",
            {"missing_keys": 3},
        )

    runner, _, _ = _preflight(
        tmp_path / "failure",
        record,
        session,
        smoke=failed_smoke,
    )
    report = runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.0.1",
    )
    exclusion = report.exclusions[0]
    assert exclusion.stage == "load_smoke"
    assert exclusion.code == "checkpoint_load_failed"
    assert exclusion.details == {"missing_keys": 3}


def test_registry_inventory_exclusions_are_preserved_without_http(tmp_path):
    unverified = {
        "id": "dino.pending",
        "status": "unverified",
        "status_reason": "Authoritative digest is pending",
    }
    registry = _registry(unverified, default_ptm=None)
    session = FakeSession({})
    client = NGCHTTPSClient(
        NGCCredential(SECRET),
        session=session,
        api_base_url="https://ngc.example.test",
    )
    runner = PTMCheckpointPreflight(
        registry=registry,
        cache=AtomicArtifactCache(tmp_path / "cache"),
        ngc_client=client,
        load_smoke=_successful_smoke,
    )
    report = runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.0.1",
    )
    assert not report.ok
    assert report.exclusions[0].stage == "registry_compatibility"
    assert report.exclusions[0].code == "status_unverified"
    assert not session.calls


def test_explicit_qualification_runs_same_gates_but_never_enables_runtime(
    tmp_path,
):
    record = _record("dino.pending")
    record["status"] = "unverified"
    record["status_reason"] = "Load qualification is pending"
    record.pop("validation")
    record.pop("compatible_tao_versions")
    client, checkpoint_url, spec_url = _client_and_urls(record)
    del client
    session = FakeSession(
        {checkpoint_url: CHECKPOINT_BYTES, spec_url: SPEC_BYTES}
    )
    runner, _, _ = _preflight(tmp_path, record, session)

    runtime = runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.0.1",
    )
    assert runtime.purpose == "runtime"
    assert runtime.validation_statuses == ("supported",)
    assert not runtime.ok
    assert not session.calls

    qualification = runner.run_qualification(
        model="dino",
        task="object_detection",
        tao_version="7.0.1",
        validation_statuses=("unverified",),
    )
    assert qualification.ok
    assert qualification.purpose == "qualification"
    assert qualification.validation_statuses == ("unverified",)
    assert qualification.inventory.candidate_checkpoint_ids == (
        "dino.pending",
    )
    prepared = qualification.prepared[0]
    assert prepared.registry_status == "unverified"
    assert prepared.runtime_eligible is False
    serialized = qualification.to_dict()
    assert serialized["prepared"][0]["runtime_eligible"] is False


def test_explicit_qualification_can_freeze_a_resolved_candidate_subset(
    tmp_path,
):
    record = _record("dino.pending")
    record["status"] = "unverified"
    record["status_reason"] = "Load qualification is pending"
    record.pop("validation")
    record.pop("compatible_tao_versions")
    client, checkpoint_url, spec_url = _client_and_urls(record)
    del client
    runner, _, _ = _preflight(
        tmp_path,
        record,
        FakeSession({
            checkpoint_url: CHECKPOINT_BYTES,
            spec_url: SPEC_BYTES,
        }),
    )

    report = runner.run_qualification(
        model="dino",
        task="object_detection",
        tao_version="7.1.0",
        validation_statuses=("unverified",),
        checkpoint_ids=("dino.pending",),
    )

    assert [item.checkpoint_id for item in report.prepared] == [
        "dino.pending"
    ]
    with pytest.raises(
        PTMPreflightConfigurationError,
        match="outside the resolved qualification population",
    ):
        runner.run_qualification(
            model="dino",
            task="object_detection",
            tao_version="7.1.0",
            validation_statuses=("unverified",),
            checkpoint_ids=("dino.agent-injected",),
        )


def test_transport_exception_does_not_report_secret_or_signed_url(tmp_path):
    record = _record()
    client, checkpoint_url, spec_url = _client_and_urls(record)
    del client
    session = FakeSession(
        {checkpoint_url: CHECKPOINT_BYTES, spec_url: SPEC_BYTES}
    )
    signed_url = (
        "https://storage.example.test/member?"
        "X-Amz-Credential=private&X-Amz-Signature=signed-secret"
    )
    session.exceptions[("HEAD", checkpoint_url)] = RuntimeError(
        f"transport leaked {SECRET} at {signed_url}"
    )
    runner, _, _ = _preflight(tmp_path, record, session)
    report = runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.0.1",
    )
    serialized = json.dumps(report.to_dict())
    assert SECRET not in serialized
    assert signed_url not in serialized
    assert "X-Amz-Signature" not in serialized
    assert report.exclusions[0].reason == (
        "NGC HTTPS request failed (RuntimeError); inspect protected transport logs"
    )


def test_stream_and_load_smoke_exceptions_have_fixed_safe_diagnostics(tmp_path):
    record = _record()
    client, checkpoint_url, spec_url = _client_and_urls(record)
    del client
    signed_url = (
        "https://storage.example.test/member?"
        "X-Amz-Credential=private&X-Amz-Signature=signed-secret"
    )
    session = FakeSession(
        {checkpoint_url: CHECKPOINT_BYTES, spec_url: SPEC_BYTES}
    )
    session.download_iter_exception[checkpoint_url] = RuntimeError(
        f"{SECRET} {signed_url}"
    )
    runner, _, _ = _preflight(tmp_path / "stream", record, session)
    stream_report = runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.0.1",
    )
    stream_serialized = json.dumps(stream_report.to_dict())
    assert SECRET not in stream_serialized
    assert "X-Amz-Signature" not in stream_serialized
    assert stream_report.exclusions[0].code == "network_error"
    assert stream_report.exclusions[0].reason == (
        "NGC HTTPS request failed (RuntimeError); inspect protected transport logs"
    )

    clean_session = FakeSession(
        {checkpoint_url: CHECKPOINT_BYTES, spec_url: SPEC_BYTES}
    )

    def raising_smoke(_request):
        raise RuntimeError(f"{SECRET} {signed_url}")

    smoke_runner, _, _ = _preflight(
        tmp_path / "smoke",
        record,
        clean_session,
        smoke=raising_smoke,
    )
    smoke_report = smoke_runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.0.1",
    )
    smoke_serialized = json.dumps(smoke_report.to_dict())
    assert SECRET not in smoke_serialized
    assert "X-Amz-Signature" not in smoke_serialized
    assert smoke_report.exclusions[0].code == "load_smoke_exception"
    assert smoke_report.exclusions[0].reason == (
        "Checkpoint load smoke callback raised an exception"
    )
    assert smoke_report.exclusions[0].details == {
        "exception_type": "RuntimeError"
    }


def test_unexpected_exception_exclusion_never_serializes_exception_text(
    tmp_path,
    monkeypatch,
):
    record = _record()
    session = FakeSession({})
    runner, _, _ = _preflight(tmp_path, record, session)
    signed_secret = (
        f"{SECRET} https://storage.example.test/a?X-Amz-Signature=secret"
    )

    def unexpected(_record):
        raise RuntimeError(signed_secret)

    monkeypatch.setattr(runner, "_prepare_checkpoint", unexpected)
    report = runner.run(
        model="dino",
        task="object_detection",
        tao_version="7.0.1",
    )
    serialized = json.dumps(report.to_dict())
    assert SECRET not in serialized
    assert "X-Amz-Signature" not in serialized
    assert report.exclusions[0].reason == (
        "Unexpected preflight exception; inspect protected logs"
    )
    assert report.exclusions[0].details == {
        "exception_type": "RuntimeError"
    }


def test_exact_member_reference_rejects_ambiguous_or_unsafe_inputs():
    credential = NGCCredential.from_environment({"NGC_KEY": SECRET})
    client = NGCHTTPSClient(
        credential,
        session=FakeSession({}),
        api_base_url="https://ngc.example.test",
    )
    source = _record()["source"]
    for member in (
        "../whole-version.zip",
        "specs//train.yaml",
        "specs/./train.yaml",
        r"specs\train.yaml",
    ):
        unsafe = dict(source)
        unsafe["member"] = member
        with pytest.raises(NGCReferenceError, match="safe relative"):
            client.resolve_member(unsafe)

    ambiguous = dict(source)
    ambiguous["registry"] = "nvidia"
    with pytest.raises(NGCReferenceError, match="<org>/<team>"):
        client.resolve_member(ambiguous)

    for base_url in (
        "http://ngc.example.test",
        "https://user:password@ngc.example.test",
        "https://ngc.example.test?X-Amz-Signature=secret",
    ):
        with pytest.raises(PTMPreflightConfigurationError, match="HTTPS"):
            NGCHTTPSClient(
                credential,
                session=FakeSession({}),
                api_base_url=base_url,
            )
