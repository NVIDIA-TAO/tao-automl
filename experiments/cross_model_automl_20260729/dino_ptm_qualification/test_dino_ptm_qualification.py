# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixture-only tests for DINO TAO 7.1 PTM qualification."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tao_automl.ptm_preflight import (
    ArtifactAdapterRequest,
    CheckpointLoadSmokeRequest,
    CheckpointLoadSmokeResult,
)
from tao_automl.ptm_registry import (
    PTMRegistry,
    canonical_sha256,
    load_ptm_registry,
    sha256_file,
)

from dino_checkpoint_adapter import (
    DINOCheckpointMetadataProjectionCallback,
    DINOProjectionFailure,
    DINO_METADATA_PROJECTION_RECIPE,
    DockerRunResult,
    DockerTorchProjectionBackend,
    HostTorchProjectionBackend,
    PINNED_TAO71_DOCKER_IMAGE,
    ProjectionBackendRequest,
    TENSOR_HASH_ALGORITHM,
    _tensor_raw_bytes,
)
from qualification_driver import (
    DINOQualificationConfiguration,
    DINOQualificationError,
    PINNED_TAO71_CONTAINER_IDENTITY,
    QUALIFICATION_COMPLETION_FILENAME,
    QUALIFICATION_MANIFEST_FILENAME,
    load_verified_qualification_completion,
    run_dino_ptm_qualification,
)
from tao71_docker_load_smoke import (
    BACKBONE_CHECKPOINT_TARGET,
    DockerRunResult as LoadSmokeDockerRunResult,
    FULL_DETECTOR_CHECKPOINT_TARGET,
    TAO71DINOCheckpointLoadSmoke,
    TAO71LoadSmokeFailure,
    _CHECKPOINT_SAFE_GLOBALS,
    _checkpoint_safe_global_names,
    _validated_checkpoint_safe_global_names,
    coverage_policy,
)


SECRET = "fixture-ngc-secret-must-not-appear"
SPEC_BYTES = b"model:\n  backbone: resnet_50\n"


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


class FakeNumpy:
    def __init__(self, content):
        self.content = content

    def tobytes(self, order):
        assert order == "C"
        return self.content


class FakeByteTensor:
    def __init__(self, content):
        self.content = content

    def reshape(self, value):
        assert value == -1
        return self

    def numpy(self):
        return FakeNumpy(self.content)


class FakeTensor:
    def __init__(self, content, *, dtype="torch.float32", shape=None):
        self.content = bytes(content)
        self.dtype = dtype
        self.shape = tuple(shape if shape is not None else (len(content),))
        self.layout = FakeTorch.strided
        self.is_quantized = False

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def reshape(self, value):
        assert value == -1
        return FakeTensor(
            self.content,
            dtype=self.dtype,
            shape=(1,) if self.shape == () else self.shape,
        )

    def view(self, dtype):
        assert dtype is FakeTorch.uint8
        if self.shape == ():
            raise RuntimeError("0-D dtype-changing view is unsupported")
        return FakeByteTensor(self.content)


def _encode_document(value):
    def encode(item):
        if isinstance(item, FakeTensor):
            return {
                "__tensor__": True,
                "dtype": item.dtype,
                "shape": list(item.shape),
                "content_hex": item.content.hex(),
            }
        if isinstance(item, dict):
            return {key: encode(nested) for key, nested in item.items()}
        return item

    return _canonical(encode(value))


def _decode_document(content):
    def decode(item):
        if isinstance(item, dict) and item.get("__tensor__") is True:
            return FakeTensor(
                bytes.fromhex(item["content_hex"]),
                dtype=item["dtype"],
                shape=item["shape"],
            )
        if isinstance(item, dict):
            return {key: decode(nested) for key, nested in item.items()}
        return item

    return decode(json.loads(content))


class FakeTorch:
    strided = object()
    uint8 = object()

    def __init__(self, *, mutate_reloaded=None):
        self.load_calls = []
        self.save_calls = []
        self.mutate_reloaded = mutate_reloaded

    @staticmethod
    def is_tensor(value):
        return isinstance(value, FakeTensor)

    def load(self, path, **kwargs):
        self.load_calls.append((Path(path), dict(kwargs)))
        value = _decode_document(Path(path).read_bytes())
        if len(self.load_calls) > 1 and self.mutate_reloaded is not None:
            self.mutate_reloaded(value)
        return value

    def save(self, value, path):
        self.save_calls.append(Path(path))
        Path(path).write_bytes(_encode_document(value))


def _fixture_documents():
    state_dict = {
        "model.backbone.weight": FakeTensor(b"\x01\x02", shape=(2,)),
        "model.head.bias": FakeTensor(b"\x03", shape=(1,)),
    }
    source = {
        "state_dict": state_dict,
        "epoch": 12,
        "optimizer_states": [{"ignored": True}],
    }
    projected = {"state_dict": state_dict, "tao_model": "dino"}
    return source, projected


def test_scalar_tensor_bytes_are_hashed_after_logical_flattening():
    scalar = FakeTensor(
        b"\x01\x00\x00\x00\x00\x00\x00\x00",
        dtype="torch.int64",
        shape=(),
    )

    assert _tensor_raw_bytes(scalar, FakeTorch) == scalar.content


def test_serializer_qualification_evidence_matches_registry_and_worker():
    repository_root = Path(__file__).resolve().parents[3]
    evidence_path = Path(__file__).with_name(
        "serializer_qualification.v1.json"
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    registry_path = (
        repository_root / "src/tao_automl/data/ptm_registry.v1.json"
    )
    worker_path = Path(__file__).with_name("dino_checkpoint_adapter.py")

    assert evidence["diagnostic_only"] is True
    assert evidence["runtime_eligibility_mutated"] is False
    assert evidence["selection_invoked"] is False
    assert evidence["agent_selected_checkpoint"] is False
    assert evidence["container"]["image"] == PINNED_TAO71_DOCKER_IMAGE
    assert evidence["container"]["identity"] == (
        PINNED_TAO71_CONTAINER_IDENTITY
    )
    assert evidence["implementation"]["worker_sha256"] == sha256_file(
        worker_path
    )
    assert evidence["implementation"]["worker_size_bytes"] == (
        worker_path.stat().st_size
    )
    frozen_registry_version = evidence["implementation"]["registry_version"]
    frozen_registry_sha256 = evidence["implementation"][
        "registry_raw_sha256"
    ]
    current_registry_sha256 = sha256_file(registry_path)
    assert evidence["implementation"]["recipe_sha256"] == canonical_sha256(
        DINO_METADATA_PROJECTION_RECIPE
    )
    assert evidence["implementation"]["safe_load"] == {
        "map_location": "cpu",
        "weights_only": True,
    }
    assert evidence["implementation"]["tensor_hash_algorithm"] == (
        TENSOR_HASH_ALGORITHM
    )

    registry = load_ptm_registry(registry_path).to_dict()
    current_registry_version = registry["registry_version"]
    if current_registry_version == frozen_registry_version:
        assert current_registry_sha256 == frozen_registry_sha256
    else:
        # The qualification evidence intentionally binds the exact registry
        # bytes that existed when the serializer was run.  A later registry
        # release may promote unrelated, independently qualified PTMs without
        # invalidating that immutable evidence.  In that case, require a
        # strictly newer registry and verify the complete DINO adapter records
        # semantically below instead of pretending the historical raw file is
        # the current file.
        assert tuple(map(int, current_registry_version.split("."))) > tuple(
            map(int, frozen_registry_version.split("."))
        )
        assert current_registry_sha256 != frozen_registry_sha256
        assert len(frozen_registry_sha256) == 64
        int(frozen_registry_sha256, 16)
    adapter_records = {
        record["id"]: record
        for record in registry["models"]["dino"]["checkpoints"]
        if any(
            adapter["id"] == "dino.tao71.metadata_wrapper.v1"
            for adapter in record.get("artifact_adapters", ())
        )
    }
    observed = {
        record["checkpoint_id"]: record for record in evidence["records"]
    }
    assert set(observed) == set(adapter_records)
    for checkpoint_id, record in observed.items():
        registered = adapter_records[checkpoint_id]
        adapter = next(
            item
            for item in registered["artifact_adapters"]
            if item["id"] == "dino.tao71.metadata_wrapper.v1"
        )
        assert record["source"] == {
            "member": registered["source"]["member"],
            "size_bytes": registered["expected_size_bytes"],
            "sha256": registered["sha256"],
        }
        assert record["output"] == {
            "member": adapter["output"]["member"],
            "size_bytes": adapter["output"]["expected_size_bytes"],
            "sha256": adapter["output"]["sha256"],
        }
        assert len(record["runs"]) == 2
        assert {
            run["output_sha256"] for run in record["runs"]
        } == {record["output"]["sha256"]}
        assert len(
            {run["evidence_sha256"] for run in record["runs"]}
        ) == 1
        tensor = record["tensor_evidence"]
        assert tensor["exact"] is True
        assert tensor["input_tensor_count"] == tensor["output_tensor_count"]
        assert (
            tensor["input_tensor_keys_sha256"]
            == tensor["output_tensor_keys_sha256"]
        )
        assert (
            tensor["input_tensor_values_sha256"]
            == tensor["output_tensor_values_sha256"]
        )
        assert record["two_run_byte_identity"] is True


def _adapter_record(input_bytes, output_bytes, *, status="supported"):
    record = {
        "id": "dino.fixture.resnet50",
        "status": status,
        "source": {
            "provider": "ngc",
            "registry": "nvidia/tao",
            "resource": "pretrained_dino_coco",
            "version": "fixture_v1",
            "member": "checkpoint.pth",
            "official": True,
            "immutable_identity": (
                "ngc://nvidia/tao/pretrained_dino_coco:"
                "fixture_v1#checkpoint.pth"
            ),
        },
        "sha256": hashlib.sha256(input_bytes).hexdigest(),
        "expected_size_bytes": len(input_bytes),
        "model_family": "dino",
        "architecture": "dino",
        "backbone": "resnet_50",
        "checkpoint_target": "train.pretrained_model_path",
        "input_contract": {"channels": 3, "height": 544, "width": 960},
        "default_spec_overrides": {"model": {"backbone": "resnet_50"}},
        "checkpoint_spec_file": {
            "source": "checkpoint_source",
            "member": "spec.yaml",
            "expected_size_bytes": len(SPEC_BYTES),
            "sha256": hashlib.sha256(SPEC_BYTES).hexdigest(),
        },
        "task_compatibility": ["object_detection"],
        "license": {
            "name": "fixture-only",
            "access_requirements": ["fixture credential"],
        },
        "deprecation": {"is_deprecated": False},
        "artifact_adapters": [
            {
                "id": "dino.tao71.metadata_wrapper.v1",
                "adapter_type": "checkpoint_metadata_projection_v1",
                "compatible_tao_versions": ["==7.1.0"],
                "recipe": copy.deepcopy(DINO_METADATA_PROJECTION_RECIPE),
                "output": {
                    "member": "tao71_fixture_checkpoint.pth",
                    "expected_size_bytes": len(output_bytes),
                    "sha256": hashlib.sha256(output_bytes).hexdigest(),
                },
                "provenance": {
                    "source": "fixture",
                    "evidence": "test_dino_ptm_qualification.py",
                },
            }
        ],
    }
    if status == "supported":
        record["compatible_tao_versions"] = ["==7.0.1"]
        record["validation"] = {
            "status": "validated",
            "tao_version": "7.0.1",
            "container_identity": "sha256:" + "1" * 64,
            "evidence": "fixture.json",
        }
    else:
        record["status_reason"] = "TAO 7.1 qualification is pending"
    return record


def _adapter_request(tmp_path, *, record=None, input_bytes=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source, projected = _fixture_documents()
    source_bytes = input_bytes or _encode_document(source)
    output_bytes = _encode_document(projected)
    active_record = record or _adapter_record(source_bytes, output_bytes)
    input_path = tmp_path / "official-checkpoint.pth"
    input_path.write_bytes(source_bytes)
    adapter = active_record["artifact_adapters"][0]
    request = ArtifactAdapterRequest(
        checkpoint_id=active_record["id"],
        model="dino",
        task="object_detection",
        tao_version="7.1.0",
        input_path=input_path,
        output_path=tmp_path / "production-partial-output",
        input_sha256=hashlib.sha256(source_bytes).hexdigest(),
        input_size_bytes=len(source_bytes),
        adapter_id=adapter["id"],
        adapter_type=adapter["adapter_type"],
        adapter_sha256=canonical_sha256(adapter),
        recipe_sha256=canonical_sha256(adapter["recipe"]),
        recipe=copy.deepcopy(adapter["recipe"]),
        registry_record=copy.deepcopy(active_record),
    )
    return request, source_bytes, output_bytes


def test_host_projection_is_safe_exact_and_reproducible(tmp_path):
    request, _, expected_output = _adapter_request(tmp_path)
    fake_torch = FakeTorch()
    callback = DINOCheckpointMetadataProjectionCallback(
        HostTorchProjectionBackend(fake_torch)
    )

    result = callback(request)

    assert result.ok
    assert result.code == "dino_metadata_projection_verified"
    assert request.output_path.read_bytes() == expected_output
    assert result.tensor_preservation.exact
    assert result.tensor_preservation.hash_algorithm == TENSOR_HASH_ALGORITHM
    assert len(fake_torch.load_calls) == 2
    assert all(
        kwargs == {"map_location": "cpu", "weights_only": True}
        for _, kwargs in fake_torch.load_calls
    )
    assert fake_torch.save_calls[0].name == (
        "tao71_fixture_checkpoint.pth"
    )
    reloaded = _decode_document(request.output_path.read_bytes())
    assert set(reloaded) == {"state_dict", "tao_model"}
    assert reloaded["tao_model"] == "dino"
    assert result.details["output_sha256"] == hashlib.sha256(
        expected_output
    ).hexdigest()

    second_request, _, _ = _adapter_request(tmp_path / "second")
    second_request.input_path.parent.mkdir(parents=True, exist_ok=True)
    second = DINOCheckpointMetadataProjectionCallback(
        HostTorchProjectionBackend(FakeTorch())
    )(second_request)
    assert second.ok
    assert second_request.output_path.read_bytes() == expected_output


def test_unverified_or_drifted_input_is_never_deserialized(tmp_path):
    request, source_bytes, _ = _adapter_request(tmp_path)
    fake_torch = FakeTorch()
    request.input_path.write_bytes(b"x" * len(source_bytes))
    result = DINOCheckpointMetadataProjectionCallback(
        HostTorchProjectionBackend(fake_torch)
    )(request)
    assert not result.ok
    assert result.code == "adapter_input_checksum_mismatch"
    assert fake_torch.load_calls == []
    assert not request.output_path.exists()

    target = tmp_path / "real-input"
    target.write_bytes(source_bytes)
    symlink_request = copy.copy(request)
    object.__setattr__(symlink_request, "input_path", tmp_path / "link-input")
    symlink_request.input_path.symlink_to(target)
    symlink_result = DINOCheckpointMetadataProjectionCallback(
        HostTorchProjectionBackend(fake_torch)
    )(symlink_request)
    assert not symlink_result.ok
    assert symlink_result.code == "adapter_input_not_regular"
    assert fake_torch.load_calls == []


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (
            lambda request: object.__setattr__(
                request,
                "recipe",
                {**request.recipe, "add_top_level_metadata": {"tao_model": "x"}},
            ),
            "unsupported_dino_projection_recipe",
        ),
        (
            lambda request: object.__setattr__(
                request,
                "adapter_sha256",
                "f" * 64,
            ),
            "adapter_registry_binding_mismatch",
        ),
    ],
)
def test_recipe_and_registry_binding_fail_closed(
    tmp_path,
    mutation,
    expected_code,
):
    request, _, _ = _adapter_request(tmp_path)
    fake_torch = FakeTorch()
    mutation(request)
    result = DINOCheckpointMetadataProjectionCallback(
        HostTorchProjectionBackend(fake_torch)
    )(request)
    assert not result.ok
    assert result.code == expected_code
    assert fake_torch.load_calls == []


def test_tensor_mutation_and_registered_output_mismatch_are_structured(
    tmp_path,
):
    request, _, _ = _adapter_request(tmp_path / "tensor")
    request.input_path.parent.mkdir(parents=True, exist_ok=True)

    def mutate(value):
        value["state_dict"]["model.head.bias"] = FakeTensor(b"\xff")

    tensor_result = DINOCheckpointMetadataProjectionCallback(
        HostTorchProjectionBackend(FakeTorch(mutate_reloaded=mutate))
    )(request)
    assert not tensor_result.ok
    assert tensor_result.code == "tensor_preservation_mismatch"

    checksum_request, source_bytes, expected_output = _adapter_request(
        tmp_path / "checksum"
    )
    checksum_request.input_path.parent.mkdir(parents=True, exist_ok=True)
    bad_record = _adapter_record(source_bytes, expected_output)
    bad_record["artifact_adapters"][0]["output"]["sha256"] = "0" * 64
    adapter = bad_record["artifact_adapters"][0]
    object.__setattr__(checksum_request, "registry_record", bad_record)
    object.__setattr__(
        checksum_request,
        "adapter_sha256",
        canonical_sha256(adapter),
    )
    checksum_result = DINOCheckpointMetadataProjectionCallback(
        HostTorchProjectionBackend(FakeTorch())
    )(checksum_request)
    assert not checksum_result.ok
    assert checksum_result.code == "adapted_output_checksum_mismatch"
    assert checksum_result.details == {
        "expected_sha256": "0" * 64,
        "observed_sha256": hashlib.sha256(expected_output).hexdigest(),
    }


class RecordingDockerRunner:
    def __init__(self, output_bytes=b"docker-projected", *, returncode=0):
        self.output_bytes = output_bytes
        self.returncode = returncode
        self.calls = []

    def run(self, argv, *, timeout_seconds):
        self.calls.append((argv, timeout_seconds))
        if self.returncode == 0:
            mounts = [
                argv[index + 1]
                for index, value in enumerate(argv)
                if value == "--mount"
            ]
            output_mount = next(
                value for value in mounts if "dst=/output" in value
            )
            source = output_mount.split("src=", 1)[1].split(",dst=", 1)[0]
            root = Path(source)
            (root / "adapted-output.pth").write_bytes(self.output_bytes)
            digest = hashlib.sha256(b"tensor-proof").hexdigest()
            evidence = {
                "schema_version": 1,
                "tensor_evidence": {
                    "hash_algorithm": TENSOR_HASH_ALGORITHM,
                    "input_tensor_count": 1,
                    "output_tensor_count": 1,
                    "input_tensor_keys_sha256": digest,
                    "output_tensor_keys_sha256": digest,
                    "input_tensor_values_sha256": digest,
                    "output_tensor_values_sha256": digest,
                    "exact": True,
                },
            }
            (root / "evidence.json").write_bytes(_canonical(evidence) + b"\n")
        return DockerRunResult(self.returncode, "ignored", "secret ignored")


def test_pinned_docker_backend_is_no_pull_no_network_and_mount_scoped(tmp_path):
    source = tmp_path / "official.pth"
    source.write_bytes(b"verified-source")
    output = tmp_path / "production-output"
    runner = RecordingDockerRunner()
    backend = DockerTorchProjectionBackend(runner=runner, timeout_seconds=42)

    result = backend.transform(
        ProjectionBackendRequest(
            input_path=source,
            output_path=output,
            output_member="tao71_fixture.pth",
            input_sha256=sha256_file(source),
            input_size_bytes=source.stat().st_size,
            recipe=DINO_METADATA_PROJECTION_RECIPE,
        )
    )

    assert output.read_bytes() == b"docker-projected"
    assert result.tensor_evidence.exact
    argv, timeout = runner.calls[0]
    assert timeout == 42
    assert PINNED_TAO71_DOCKER_IMAGE in argv
    assert "--pull=never" in argv
    assert "--network=none" in argv
    assert "--read-only" in argv
    assert "--cap-drop=ALL" in argv
    mounts = [
        argv[index + 1]
        for index, value in enumerate(argv)
        if value == "--mount"
    ]
    assert any("dst=/input/checkpoint.pth,readonly" in item for item in mounts)
    assert any("dst=/output" in item and "readonly" not in item for item in mounts)
    assert any(
        "dst=/opt/dino_checkpoint_adapter.py,readonly" in item
        for item in mounts
    )
    assert SECRET not in " ".join(argv)

    failed = DockerTorchProjectionBackend(
        runner=RecordingDockerRunner(returncode=17)
    )
    with pytest.raises(DINOProjectionFailure, match="nonzero"):
        failed.transform(
            ProjectionBackendRequest(
                input_path=source,
                output_path=tmp_path / "failed",
                output_member="tao71_fixture.pth",
                input_sha256=sha256_file(source),
                input_size_bytes=source.stat().st_size,
                recipe=DINO_METADATA_PROJECTION_RECIPE,
            )
        )


class RecordingTAO71LoadSmokeRunner:
    def __init__(self, *, mode="success"):
        self.mode = mode
        self.calls = []
        self.merged_overrides = None

    @staticmethod
    def _argument(argv, name):
        return argv[argv.index(name) + 1]

    @staticmethod
    def _mount_source(argv, destination):
        mounts = [
            argv[index + 1]
            for index, value in enumerate(argv)
            if value == "--mount"
        ]
        mount = next(
            item for item in mounts if f"dst={destination}" in item
        )
        return Path(mount.split("src=", 1)[1].split(",dst=", 1)[0])

    def run(self, argv, *, timeout_seconds):
        self.calls.append((argv, timeout_seconds))
        if self.mode == "nonzero":
            return LoadSmokeDockerRunResult(19, "ignored", "protected")

        overrides_path = self._mount_source(
            argv,
            "/input/merged-overrides.json",
        )
        assert hashlib.sha256(overrides_path.read_bytes()).hexdigest() == (
            self._argument(argv, "--overrides-sha256")
        )
        self.merged_overrides = json.loads(overrides_path.read_text())
        output_root = self._mount_source(argv, "/output")
        evidence_path = output_root / "load-smoke-evidence.json"
        if self.mode == "malformed":
            evidence_path.write_bytes(b"{")
            return LoadSmokeDockerRunResult(0)

        digest = hashlib.sha256(b"load-smoke-keys").hexdigest()
        checkpoint_target = self._argument(
            argv,
            "--checkpoint-target",
        )
        if self.mode == "insufficient":
            matched_tensor_count = 1
            matched_numel = 1
            checkpoint_loaded = False
        else:
            matched_tensor_count = 10
            matched_numel = 100
            checkpoint_loaded = self.mode != "mismatched"
        evidence = {
            "schema_version": 1,
            "contract_version": 1,
            "checkpoint_id": self._argument(argv, "--checkpoint-id"),
            "checkpoint_target": checkpoint_target,
            "backbone": self._argument(argv, "--backbone"),
            "checkpoint_sha256": self._argument(
                argv,
                "--checkpoint-sha256",
            ),
            "checkpoint_size_bytes": int(
                self._argument(argv, "--checkpoint-size")
            ),
            "checkpoint_loaded": checkpoint_loaded,
            "state_dict_compatible": checkpoint_loaded,
            "execution_backend": "docker",
            "container_identity": self._argument(
                argv,
                "--container-identity",
            ),
            "tao_version": self._argument(argv, "--tao-version"),
            "device": "cpu",
            "weights_only": True,
            "weights_only_allowed_globals": [
                argv[index + 1]
                for index, value in enumerate(argv)
                if value == "--safe-global"
            ],
            "merged_overrides_sha256": self._argument(
                argv,
                "--overrides-sha256",
            ),
            "coverage_policy": coverage_policy(checkpoint_target),
            "tao_load_path": (
                "train_shape_aware_full_detector"
                if checkpoint_target == FULL_DETECTOR_CHECKPOINT_TARGET
                else "model_pretrained_backbone_path"
            ),
            "tao_load_path_executed": True,
            "safe_path_load_count": (
                0
                if checkpoint_target == FULL_DETECTOR_CHECKPOINT_TARGET
                else 1
            ),
            "source_tensor_count": 10,
            "target_tensor_count": 10,
            "matched_tensor_count": matched_tensor_count,
            "matched_numel": matched_numel,
            "target_numel": 100,
            "matched_target_tensor_fraction": (
                matched_tensor_count / 10
            ),
            "matched_target_numel_fraction": matched_numel / 100,
            "shape_mismatch_count": 0,
            "missing_target_count": 10 - matched_tensor_count,
            "unexpected_source_count": 0,
            "loaded_value_match_count": matched_tensor_count,
            "loaded_value_match_numel": matched_numel,
            "adapted_state_keys_sha256": digest,
            "matched_state_keys_sha256": digest,
            "missing_target_keys_sha256": digest,
            "shape_mismatch_keys_sha256": digest,
            "unexpected_source_keys_sha256": digest,
            "loaded_value_match_keys_sha256": digest,
        }
        if self.mode == "wrong_safe_global":
            evidence["weights_only_allowed_globals"] = []
        evidence_path.write_bytes(_canonical(evidence) + b"\n")
        return LoadSmokeDockerRunResult(0, "ignored", "protected")


def _load_smoke_request(tmp_path):
    source, projected = _fixture_documents()
    source_bytes = _encode_document(source)
    projected_bytes = _encode_document(projected)
    record = _adapter_record(
        source_bytes,
        projected_bytes,
        status="unverified",
    )
    checkpoint = tmp_path / "adapted-checkpoint.pth"
    checkpoint.write_bytes(projected_bytes)
    sidecar = tmp_path / "checkpoint-spec.yaml"
    sidecar.write_bytes(SPEC_BYTES)
    return CheckpointLoadSmokeRequest(
        checkpoint_id=record["id"],
        model="dino",
        task="object_detection",
        tao_version="7.1.0-rc-245",
        checkpoint_path=checkpoint,
        checkpoint_spec_path=sidecar,
        checkpoint_spec={
            "model": {
                "backbone": "sidecar-backbone",
                "num_queries": 300,
                "pretrained_backbone_path": "/host/backbone.pth",
            },
            "train": {
                "num_epochs": 12,
                "pretrained_model_path": "/host/checkpoint.pth",
            },
        },
        default_spec_overrides={
            "model": {"backbone": "resnet_50"},
            "train": {"num_epochs": 36},
        },
        registry_record=record,
    )


def test_concrete_tao71_load_smoke_is_pinned_safe_and_spec_aware(tmp_path):
    request = _load_smoke_request(tmp_path)
    runner = RecordingTAO71LoadSmokeRunner()
    callback = TAO71DINOCheckpointLoadSmoke(
        runner=runner,
        timeout_seconds=77,
    )

    result = callback(request)

    assert result.ok
    assert result.code == "tao71_dino_checkpoint_loaded"
    assert result.details["matched_tensor_count"] == 10
    assert result.details["checkpoint_target"] == (
        FULL_DETECTOR_CHECKPOINT_TARGET
    )
    argv, timeout = runner.calls[0]
    assert timeout == 77
    assert argv[0:3] == ("docker", "run", "--rm")
    assert PINNED_TAO71_DOCKER_IMAGE in argv
    assert "--pull=never" in argv
    assert "--network=none" in argv
    assert "--read-only" in argv
    assert "--cap-drop=ALL" in argv
    assert "--security-opt=no-new-privileges" in argv
    assert "--env=USER=tao-automl" in argv
    assert "--env=LOGNAME=tao-automl" in argv
    assert "--env=NVIDIA_VISIBLE_DEVICES=void" in argv
    assert "--env=CUDA_VISIBLE_DEVICES=" in argv
    assert "--gpus" not in argv
    assert (
        RecordingTAO71LoadSmokeRunner._argument(
            argv,
            "--checkpoint-target",
        )
        == FULL_DETECTOR_CHECKPOINT_TARGET
    )
    assert RecordingTAO71LoadSmokeRunner._argument(
        argv,
        "--backbone",
    ) == "resnet_50"
    mounts = [
        argv[index + 1]
        for index, value in enumerate(argv)
        if value == "--mount"
    ]
    assert any("dst=/input/checkpoint.pth,readonly" in item for item in mounts)
    assert any(
        "dst=/input/merged-overrides.json,readonly" in item
        for item in mounts
    )
    assert any("dst=/output" in item and "readonly" not in item for item in mounts)
    assert any(
        "dst=/opt/tao71_dino_load_smoke.py,readonly" in item
        for item in mounts
    )
    assert SECRET not in " ".join(argv)
    assert "NGC_KEY" not in " ".join(argv)
    assert runner.merged_overrides == {
        "model": {
            "backbone": "resnet_50",
            "num_queries": 300,
            "pretrained_backbone_path": None,
        },
        "train": {"num_epochs": 36},
    }


def test_concrete_tao71_load_smoke_routes_backbone_target(tmp_path):
    request = _load_smoke_request(tmp_path)
    record = copy.deepcopy(request.registry_record)
    record["checkpoint_target"] = BACKBONE_CHECKPOINT_TARGET
    record.pop("artifact_adapters")
    record["expected_size_bytes"] = request.checkpoint_path.stat().st_size
    record["sha256"] = sha256_file(request.checkpoint_path)
    object.__setattr__(request, "registry_record", record)
    runner = RecordingTAO71LoadSmokeRunner()

    result = TAO71DINOCheckpointLoadSmoke(runner=runner)(request)

    assert result.ok
    assert result.details["checkpoint_target"] == BACKBONE_CHECKPOINT_TARGET
    assert result.details["backbone"] == "resnet_50"
    assert result.details["coverage_policy"]["load_scope"] == "backbone"
    assert result.details["tao_load_path"] == (
        "model_pretrained_backbone_path"
    )
    assert result.details["safe_path_load_count"] == 1
    argv, _ = runner.calls[0]
    assert RecordingTAO71LoadSmokeRunner._argument(
        argv,
        "--checkpoint-target",
    ) == BACKBONE_CHECKPOINT_TARGET
    assert RecordingTAO71LoadSmokeRunner._argument(
        argv,
        "--backbone",
    ) == "resnet_50"


def test_checkpoint_safe_global_is_bound_to_exact_registry_identity():
    checkpoint_id = "dino.backbone.nvimagenet.resnet50"
    checkpoint_sha256 = (
        "49b0df2b517a28760e17158c9ad78371"
        "c1f833d6ad257f117ff81356743060b7"
    )

    assert _checkpoint_safe_global_names(
        checkpoint_id,
        checkpoint_sha256,
    ) == ("argparse.Namespace",)
    assert _checkpoint_safe_global_names(
        "dino.backbone.imagenet.fan_hybrid_small",
        "0" * 64,
    ) == ()
    with pytest.raises(
        TAO71LoadSmokeFailure,
        match="registered artifact digest",
    ):
        _checkpoint_safe_global_names(checkpoint_id, "0" * 64)


def test_checkpoint_safe_global_policy_matches_packaged_registry():
    records = {
        record["id"]: record
        for record in load_ptm_registry().to_dict()["models"]["dino"][
            "checkpoints"
        ]
    }

    assert set(_CHECKPOINT_SAFE_GLOBALS) == {
        "dino.backbone.nvimagenet.resnet50"
    }
    for checkpoint_id, policy in _CHECKPOINT_SAFE_GLOBALS.items():
        assert policy["checkpoint_sha256"] == records[checkpoint_id]["sha256"]


def test_safe_global_plumbing_and_echo_are_fail_closed(tmp_path, monkeypatch):
    import tao71_docker_load_smoke

    request = _load_smoke_request(tmp_path)
    checkpoint_sha256 = sha256_file(request.checkpoint_path)
    monkeypatch.setitem(
        tao71_docker_load_smoke._CHECKPOINT_SAFE_GLOBALS,
        request.checkpoint_id,
        {
            "checkpoint_sha256": checkpoint_sha256,
            "allowed_globals": ("argparse.Namespace",),
        },
    )
    runner = RecordingTAO71LoadSmokeRunner()

    result = TAO71DINOCheckpointLoadSmoke(runner=runner)(request)

    assert result.ok
    argv, _ = runner.calls[0]
    safe_global_positions = [
        index for index, value in enumerate(argv) if value == "--safe-global"
    ]
    assert len(safe_global_positions) == 1
    assert argv[safe_global_positions[0] + 1] == "argparse.Namespace"
    assert result.details["weights_only_allowed_globals"] == [
        "argparse.Namespace"
    ]

    rejected = TAO71DINOCheckpointLoadSmoke(
        runner=RecordingTAO71LoadSmokeRunner(mode="wrong_safe_global")
    )(request)
    assert not rejected.ok
    assert rejected.code == "invalid_tao71_load_smoke_evidence"

    with pytest.raises(ValueError, match="disagrees with artifact identity"):
        _validated_checkpoint_safe_global_names(
            request.checkpoint_id,
            checkpoint_sha256,
            (),
        )
    with pytest.raises(ValueError, match="disagrees with artifact identity"):
        _validated_checkpoint_safe_global_names(
            request.checkpoint_id,
            checkpoint_sha256,
            ("argparse.Namespace", "argparse.Namespace"),
        )


def test_concrete_tao71_load_smoke_rejects_unregistered_target(tmp_path):
    request = _load_smoke_request(tmp_path)
    record = copy.deepcopy(request.registry_record)
    record["checkpoint_target"] = "agent.selected.path"
    object.__setattr__(request, "registry_record", record)
    runner = RecordingTAO71LoadSmokeRunner()

    result = TAO71DINOCheckpointLoadSmoke(runner=runner)(request)

    assert not result.ok
    assert result.code == "unsupported_checkpoint_target"
    assert runner.calls == []


def test_concrete_tao71_load_smoke_rebinds_registered_artifact(tmp_path):
    request = _load_smoke_request(tmp_path)
    content = request.checkpoint_path.read_bytes()
    request.checkpoint_path.write_bytes(
        bytes([content[0] ^ 1]) + content[1:]
    )
    runner = RecordingTAO71LoadSmokeRunner()

    result = TAO71DINOCheckpointLoadSmoke(runner=runner)(request)

    assert not result.ok
    assert result.code == "load_smoke_registry_artifact_mismatch"
    assert runner.calls == []


@pytest.mark.parametrize(
    ("mode", "expected_code"),
    [
        ("nonzero", "tao71_docker_load_smoke_failed"),
        ("malformed", "invalid_tao71_load_smoke_evidence"),
        ("mismatched", "invalid_tao71_load_smoke_evidence"),
        ("insufficient", "insufficient_tao71_checkpoint_coverage"),
    ],
)
def test_concrete_tao71_load_smoke_failures_are_structured(
    tmp_path,
    mode,
    expected_code,
):
    request = _load_smoke_request(tmp_path)
    result = TAO71DINOCheckpointLoadSmoke(
        runner=RecordingTAO71LoadSmokeRunner(mode=mode)
    )(request)
    assert not result.ok
    assert result.code == expected_code
    assert SECRET not in result.reason
    assert SECRET not in json.dumps(result.details)


def test_concrete_tao71_load_smoke_rejects_symlink_before_docker(tmp_path):
    request = _load_smoke_request(tmp_path)
    target = request.checkpoint_path
    link = tmp_path / "checkpoint-link.pth"
    link.symlink_to(target)
    object.__setattr__(request, "checkpoint_path", link)
    runner = RecordingTAO71LoadSmokeRunner()

    result = TAO71DINOCheckpointLoadSmoke(runner=runner)(request)

    assert not result.ok
    assert result.code == "load_smoke_input_not_regular"
    assert runner.calls == []


class FakeResponse:
    def __init__(self, status_code, content=b"", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = dict(headers or {})
        self.closed = False

    def iter_content(self, chunk_size):
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset:offset + chunk_size]

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, payloads):
        self.payloads = dict(payloads)
        self.calls = []

    def head(self, url, **kwargs):
        self.calls.append(("HEAD", url, kwargs))
        payload = self.payloads.get(url)
        return FakeResponse(
            200 if payload is not None else 404,
            headers={"Content-Length": str(len(payload or b"")), "ETag": '"x"'},
        )

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        payload = self.payloads.get(url)
        if payload is None:
            return FakeResponse(404)
        if kwargs.get("headers", {}).get("Range") == "bytes=0-0":
            return FakeResponse(
                206,
                payload[:1],
                {
                    "Content-Length": "1",
                    "Content-Range": f"bytes 0-0/{len(payload)}",
                },
            )
        return FakeResponse(
            200,
            payload,
            {"Content-Length": str(len(payload))},
        )


class FakeDockerLoadSmoke:
    def __init__(self, configuration, *, invalid_field=None):
        self.configuration = configuration
        self.invalid_field = invalid_field
        self.calls = []

    def __call__(self, request):
        self.calls.append(request)
        details = {
            "contract_version": 1,
            "execution_backend": "docker",
            "container_identity": self.configuration.container_identity,
            "tao_version": self.configuration.tao_version,
            "checkpoint_sha256": sha256_file(request.checkpoint_path),
            "checkpoint_size_bytes": request.checkpoint_path.stat().st_size,
            "checkpoint_loaded": True,
            "state_dict_compatible": True,
        }
        if self.invalid_field is not None:
            details[self.invalid_field] = "invalid"
        return CheckpointLoadSmokeResult(
            True,
            "tao71_docker_loaded",
            "Pinned Docker load smoke passed",
            details,
        )


def _driver_fixture():
    source, projected = _fixture_documents()
    input_bytes = _encode_document(source)
    output_bytes = _encode_document(projected)
    record = _adapter_record(input_bytes, output_bytes, status="unverified")
    registry = PTMRegistry(
        {
            "schema_version": 1,
            "registry_version": "fixture-v1",
            "models": {
                "dino": {
                    "default_ptm": None,
                    "checkpoints": [record],
                }
            },
        }
    )
    base = "https://ngc.example.test"
    prefix = (
        f"{base}/v2/models/nvidia/tao/pretrained_dino_coco/"
        "versions/fixture_v1/files/"
    )
    session = FakeSession(
        {
            prefix + "checkpoint.pth": input_bytes,
            prefix + "spec.yaml": SPEC_BYTES,
        }
    )
    configuration = DINOQualificationConfiguration(
        ngc_api_base_url=base,
        validation_statuses=("unverified",),
    )
    adapter = DINOCheckpointMetadataProjectionCallback(
        HostTorchProjectionBackend(FakeTorch())
    )
    smoke = FakeDockerLoadSmoke(configuration)
    return registry, session, configuration, adapter, smoke


def test_driver_emits_deterministic_secret_free_create_only_artifacts(
    tmp_path,
):
    registry, session, configuration, adapter, smoke = _driver_fixture()
    output_dir = tmp_path / "evidence"
    cache_dir = tmp_path / "cache"
    completion = run_dino_ptm_qualification(
        output_dir=output_dir,
        cache_dir=cache_dir,
        docker_load_smoke=smoke,
        configuration=configuration,
        registry=registry,
        environment={"NGC_KEY": SECRET},
        http_session=session,
        artifact_adapter=adapter,
    )

    assert completion["qualification_only"] is True
    assert completion["runtime_eligibility_mutated"] is False
    assert completion["selection_invoked"] is False
    prepared = completion["report"]["prepared"]
    assert len(prepared) == 1
    assert prepared[0]["runtime_eligible"] is False
    assert prepared[0]["artifact_adaptation"]["tensor_preservation"][
        "exact"
    ] is True
    assert len(smoke.calls) == 1
    assert smoke.calls[0].checkpoint_path.read_bytes() == (
        _encode_document(_fixture_documents()[1])
    )
    manifest_path = output_dir / QUALIFICATION_MANIFEST_FILENAME
    completion_path = output_dir / QUALIFICATION_COMPLETION_FILENAME
    assert manifest_path.is_file()
    assert completion_path.is_file()
    serialized = manifest_path.read_text() + completion_path.read_text()
    assert SECRET not in serialized
    assert "/localhome/" not in serialized
    assert json.loads(manifest_path.read_text())["adapter_execution"][
        "backend"
    ]["backend"] == "host_torch"

    with pytest.raises(FileExistsError):
        run_dino_ptm_qualification(
            output_dir=output_dir,
            cache_dir=cache_dir,
            docker_load_smoke=smoke,
            configuration=configuration,
            registry=registry,
            environment={"NGC_KEY": SECRET},
            http_session=session,
            artifact_adapter=adapter,
            resume=False,
        )

    original_call_count = len(session.calls)
    original_smoke_count = len(smoke.calls)
    resumed = run_dino_ptm_qualification(
        output_dir=output_dir,
        cache_dir=cache_dir,
        docker_load_smoke=smoke,
        configuration=configuration,
        registry=registry,
        environment={},
        http_session=None,
        artifact_adapter=adapter,
        resume=True,
    )
    assert resumed == completion
    assert len(session.calls) == original_call_count
    assert len(smoke.calls) == original_smoke_count

    registry_2, session_2, config_2, adapter_2, smoke_2 = _driver_fixture()
    completion_2 = run_dino_ptm_qualification(
        output_dir=tmp_path / "evidence-2",
        cache_dir=tmp_path / "cache-2",
        docker_load_smoke=smoke_2,
        configuration=config_2,
        registry=registry_2,
        environment={"NGC_KEY": SECRET},
        http_session=session_2,
        artifact_adapter=adapter_2,
    )
    assert completion_2 == completion


def test_driver_freezes_upstream_cpu_population_and_reloads_evidence(
    tmp_path,
):
    registry, session, configuration, adapter, _ = _driver_fixture()
    upstream_sha = "c" * 64
    configuration = DINOQualificationConfiguration(
        ngc_api_base_url=configuration.ngc_api_base_url,
        validation_statuses=("unverified",),
        checkpoint_ids=("dino.fixture.resnet50",),
        upstream_completion_sha256=upstream_sha,
    )
    smoke = FakeDockerLoadSmoke(configuration)
    output_dir = tmp_path / "evidence"
    cache_dir = tmp_path / "cache"

    completion = run_dino_ptm_qualification(
        output_dir=output_dir,
        cache_dir=cache_dir,
        docker_load_smoke=smoke,
        configuration=configuration,
        registry=registry,
        environment={"NGC_KEY": SECRET},
        http_session=session,
        artifact_adapter=adapter,
    )

    manifest = json.loads(
        (output_dir / QUALIFICATION_MANIFEST_FILENAME).read_text()
    )
    assert manifest["checkpoint_ids"] == ["dino.fixture.resnet50"]
    assert manifest["upstream_completion_sha256"] == upstream_sha
    assert load_verified_qualification_completion(
        output_dir=output_dir,
        cache_dir=cache_dir,
    ) == completion


@pytest.mark.parametrize(
    "kwargs",
    [
        {"checkpoint_ids": ()},
        {"checkpoint_ids": ("z", "a")},
        {"upstream_completion_sha256": "not-a-digest"},
    ],
)
def test_qualification_configuration_rejects_unfrozen_upstream_inputs(kwargs):
    with pytest.raises(ValueError):
        DINOQualificationConfiguration(**kwargs)


def test_default_driver_excludes_unsupported_without_checkpoint_io(tmp_path):
    registry, session, _, adapter, _ = _driver_fixture()
    document = registry.to_dict()
    document["models"]["dino"]["checkpoints"].append(
        {
            "id": "dino.fixture.gcvit",
            "status": "unsupported",
            "status_reason": (
                "TAO 7.1 raises NotImplementedError for the GCViT backbone"
            ),
        }
    )
    registry = PTMRegistry(document)
    configuration = DINOQualificationConfiguration(
        ngc_api_base_url="https://ngc.example.test"
    )
    smoke = FakeDockerLoadSmoke(configuration)

    completion = run_dino_ptm_qualification(
        output_dir=tmp_path / "evidence",
        cache_dir=tmp_path / "cache",
        docker_load_smoke=smoke,
        configuration=configuration,
        registry=registry,
        environment={"NGC_KEY": SECRET},
        http_session=session,
        artifact_adapter=adapter,
    )

    assert [item["checkpoint_id"] for item in completion["report"]["prepared"]] == [
        "dino.fixture.resnet50"
    ]
    exclusion = next(
        item
        for item in completion["report"]["exclusions"]
        if item["checkpoint_id"] == "dino.fixture.gcvit"
    )
    assert exclusion["stage"] == "registry_qualification"
    assert exclusion["code"] == "status_not_requested_for_qualification"
    assert all("gcvit" not in url for _, url, _ in session.calls)


def test_resume_rejects_completion_or_cache_tampering(tmp_path):
    registry, session, configuration, adapter, smoke = _driver_fixture()
    output_dir = tmp_path / "evidence"
    cache_dir = tmp_path / "cache"
    completion = run_dino_ptm_qualification(
        output_dir=output_dir,
        cache_dir=cache_dir,
        docker_load_smoke=smoke,
        configuration=configuration,
        registry=registry,
        environment={"NGC_KEY": SECRET},
        http_session=session,
        artifact_adapter=adapter,
    )
    checkpoint = completion["report"]["prepared"][0]["checkpoint"]
    checkpoint_path = cache_dir / checkpoint["cache_relative_path"]
    checkpoint_path.write_bytes(b"x" * checkpoint["size_bytes"])
    with pytest.raises(DINOQualificationError, match="frozen evidence"):
        run_dino_ptm_qualification(
            output_dir=output_dir,
            cache_dir=cache_dir,
            docker_load_smoke=smoke,
            configuration=configuration,
            registry=registry,
            environment={},
            artifact_adapter=adapter,
            resume=True,
        )

    completion_path = output_dir / QUALIFICATION_COMPLETION_FILENAME
    document = json.loads(completion_path.read_text())
    document["selection_invoked"] = True
    completion_path.write_bytes(_canonical(document) + b"\n")
    with pytest.raises(DINOQualificationError, match="SHA-256"):
        run_dino_ptm_qualification(
            output_dir=output_dir,
            cache_dir=cache_dir,
            docker_load_smoke=smoke,
            configuration=configuration,
            registry=registry,
            environment={},
            artifact_adapter=adapter,
            resume=True,
        )


def test_invalid_docker_load_smoke_evidence_is_preserved_as_exclusion(
    tmp_path,
):
    registry, session, configuration, adapter, _ = _driver_fixture()
    bad_smoke = FakeDockerLoadSmoke(
        configuration,
        invalid_field="container_identity",
    )
    completion = run_dino_ptm_qualification(
        output_dir=tmp_path / "evidence",
        cache_dir=tmp_path / "cache",
        docker_load_smoke=bad_smoke,
        configuration=configuration,
        registry=registry,
        environment={"NGC_KEY": SECRET},
        http_session=session,
        artifact_adapter=adapter,
    )
    assert completion["report"]["prepared"] == []
    exclusion = completion["report"]["exclusions"][0]
    assert exclusion["stage"] == "load_smoke"
    assert exclusion["code"] == "invalid_tao71_docker_load_smoke_evidence"
    assert exclusion["details"] == {
        "missing_or_mismatched_fields": ["container_identity"]
    }


def test_default_campaign_adapter_identity_is_pinned_docker(tmp_path):
    source, projected = _fixture_documents()
    record = _adapter_record(
        _encode_document(source),
        _encode_document(projected),
        status="unverified",
    )
    registry = PTMRegistry(
        {
            "schema_version": 1,
            "registry_version": "fixture-v1",
            "models": {
                "dino": {"default_ptm": None, "checkpoints": [record]}
            },
        }
    )
    import dino_checkpoint_adapter
    import qualification_driver
    import tao71_docker_load_smoke
    from qualification_driver import (
        build_dino_qualification_manifest,
    )

    manifest = build_dino_qualification_manifest(
        registry,
        DINOQualificationConfiguration(validation_statuses=("unverified",)),
    )
    backend = manifest["adapter_execution"]["backend"]
    assert backend["backend"] == "tao71_docker"
    assert backend["container_image"] == PINNED_TAO71_DOCKER_IMAGE
    assert backend["pull_policy"] == "never"
    assert backend["network"] == "none"
    assert manifest["adapter_execution"]["worker_source_sha256"] == (
        sha256_file(Path(dino_checkpoint_adapter.__file__))
    )
    assert backend["worker_source_sha256"] == (
        manifest["adapter_execution"]["worker_source_sha256"]
    )
    assert manifest["load_smoke_contract"]["container_identity"] == (
        PINNED_TAO71_CONTAINER_IDENTITY
    )
    assert manifest["load_smoke_contract"]["callback"] == (
        "tao71_dino_checkpoint_load_smoke_v1"
    )
    assert manifest["load_smoke_contract"]["container_image"] == (
        PINNED_TAO71_DOCKER_IMAGE
    )
    assert manifest["load_smoke_contract"]["safe_load"] == {
        "map_location": "cpu",
        "weights_only": True,
        "checkpoint_specific_allowed_globals": {
            "dino.backbone.nvimagenet.resnet50": {
                "checkpoint_sha256": (
                    "49b0df2b517a28760e17158c9ad78371"
                    "c1f833d6ad257f117ff81356743060b7"
                ),
                "allowed_globals": ["argparse.Namespace"],
            }
        },
    }
    assert manifest["load_smoke_contract"]["target_routing"] == [
        BACKBONE_CHECKPOINT_TARGET,
        FULL_DETECTOR_CHECKPOINT_TARGET,
    ]
    assert manifest["checkpoint_inventory"][0]["checkpoint_target"] == (
        FULL_DETECTOR_CHECKPOINT_TARGET
    )
    assert manifest["checkpoint_inventory"][0]["backbone"] == "resnet_50"
    assert manifest["load_smoke_contract"]["worker_source_sha256"] == (
        sha256_file(Path(tao71_docker_load_smoke.__file__))
    )
    implementation = manifest["implementation_source"]
    assert implementation["binding"] == "source_file_sha256"
    assert implementation["qualification_driver"]["sha256"] == (
        sha256_file(Path(qualification_driver.__file__))
    )
    assert set(implementation) == {
        "binding",
        "qualification_driver",
        "ptm_preflight",
        "ptm_registry",
    }


def test_qualification_cli_has_no_secret_argument_and_routes_defaults(
    monkeypatch,
    capsys,
):
    import qualification_driver

    help_text = qualification_driver._parser().format_help()
    assert "--ngc-key" not in help_text.lower()
    assert "--password" not in help_text.lower()
    observed = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {
            "completion_sha256": "c" * 64,
            "manifest_sha256": "m" * 64,
            "report": {
                "prepared": [{"checkpoint_id": "dino.fixture.prepared"}],
                "exclusions": [{"checkpoint_id": "dino.fixture.excluded"}],
            },
        }

    monkeypatch.setattr(
        qualification_driver,
        "run_dino_ptm_qualification",
        fake_run,
    )
    result = qualification_driver.main(
        [
            "--output-dir",
            "qualification-output",
            "--cache-dir",
            "qualification-cache",
            "--registry-path",
            "registry.json",
            "--resume",
        ]
    )

    assert result == 0
    assert observed == {
        "output_dir": "qualification-output",
        "cache_dir": "qualification-cache",
        "registry_path": "registry.json",
        "resume": True,
    }
    assert json.loads(capsys.readouterr().out) == {
        "completion_sha256": "c" * 64,
        "excluded_checkpoint_ids": ["dino.fixture.excluded"],
        "manifest_sha256": "m" * 64,
        "prepared_checkpoint_ids": ["dino.fixture.prepared"],
    }
