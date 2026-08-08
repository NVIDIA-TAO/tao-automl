# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the repository-owned pretrained-model registry."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from tao_automl.ptm_registry import (
    PTMArtifactAdapterResolutionError,
    PTMRegistry,
    PTMRegistryValidationError,
    SPEC_PRECEDENCE,
    canonical_sha256,
    load_ptm_registry,
    load_ptm_registry_schema,
    merge_ptm_spec_precedence,
    sha256_file,
    validate_ptm_registry,
    verify_file_sha256,
    verify_packaged_resource_sha256,
)


def _artifact_adapter(
    *,
    versions=None,
    adapter_id="dino.tao71.metadata_wrapper.v1",
):
    return {
        "id": adapter_id,
        "adapter_type": "checkpoint_metadata_projection_v1",
        "compatible_tao_versions": versions or ["==7.1.0"],
        "recipe": {
            "retain_top_level_keys": ["state_dict"],
            "add_top_level_metadata": {"tao_model": "dino"},
            "tensor_container_key": "state_dict",
            "require_exact_tensor_key_set": True,
            "require_exact_tensor_values": True,
        },
        "output": {
            "member": "tao71_checkpoint.pth",
            "expected_size_bytes": 123,
            "sha256": "a" * 64,
        },
        "provenance": {
            "source": "Preserved qualification fixture",
            "evidence": "tests/test_ptm_registry.py",
        },
    }


def _supported_checkpoint(checkpoint_id="dino.resnet50.v1"):
    checkpoint_bytes = b"authoritative checkpoint fixture"
    spec_bytes = b"model:\n  backbone: resnet_50\n"
    return {
        "id": checkpoint_id,
        "status": "supported",
        "source": {
            "provider": "ngc",
            "registry": "nvidia/tao",
            "resource": "pretrained_dino_coco",
            "version": "v1.0",
            "member": "dino_resnet50_ep12.pth",
            "official": True,
            "immutable_identity": (
                "ngc://nvidia/tao/pretrained_dino_coco:v1.0"
                "#dino_resnet50_ep12.pth"
            ),
        },
        "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "expected_size_bytes": len(checkpoint_bytes),
        "compatible_tao_versions": [">=7.0,<7.2"],
        "model_family": "dino",
        "architecture": "dino",
        "backbone": "resnet_50",
        "checkpoint_target": "train.pretrained_model_path",
        "input_contract": {
            "channels": 3,
            "height": None,
            "width": None,
            "preprocessing": {"color_space": "rgb"},
        },
        "default_spec_overrides": {
            "model": {"backbone": "resnet_50", "num_queries": 900}
        },
        "checkpoint_spec_file": {
            "source": "checkpoint_source",
            "member": "experiment_spec.yaml",
            "expected_size_bytes": len(spec_bytes),
            "sha256": hashlib.sha256(spec_bytes).hexdigest(),
        },
        "task_compatibility": ["object_detection"],
        "license": {
            "name": "NVIDIA TAO Model License",
            "url": "https://example.invalid/license",
            "access_requirements": ["NGC account"],
        },
        "deprecation": {"is_deprecated": False},
        "validation": {
            "status": "validated",
            "tao_version": "7.0.1",
            "container_identity": "sha256:" + "1" * 64,
            "evidence": "tests/fixtures/dino_resnet50_preflight.json",
        },
    }


def _registry(*records, default_ptm=None):
    return {
        "schema_version": 1,
        "registry_version": "test-v1",
        "models": {
            "dino": {
                "default_ptm": default_ptm,
                "checkpoints": list(records),
            }
        },
    }


def test_packaged_dino_registry_and_schema_load():
    registry = load_ptm_registry()
    assert registry.schema_version == 1
    assert registry.registry_version == "1.5.0"
    assert "dino" in registry.models
    assert registry.models == tuple(sorted(registry.models))
    assert len(registry.document_sha256) == 64

    schema = load_ptm_registry_schema()
    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["properties"]["schema_version"]["const"] == 1
    assert (
        schema["$defs"]["artifact_adapter"]["properties"]["adapter_type"]["const"]
        == "checkpoint_metadata_projection_v1"
    )


@pytest.mark.parametrize(
    ("mutate", "expected_path"),
    [
        (lambda doc: doc.update({"schema_verzion": 1}), "registry"),
        (
            lambda doc: doc["models"]["dino"].update({"default_checkpoint": "x"}),
            "models.dino",
        ),
        (
            lambda doc: doc["models"]["dino"]["checkpoints"][0].update(
                {"expected_bytes": 1}
            ),
            "checkpoints",
        ),
        (
            lambda doc: doc["models"]["dino"]["checkpoints"][0]["source"].update(
                {"registry_path": "nvidia/tao"}
            ),
            ".source",
        ),
        (
            lambda doc: doc["models"]["dino"]["checkpoints"][0][
                "checkpoint_spec_file"
            ].update({"checksum": "0" * 64}),
            ".checkpoint_spec_file",
        ),
    ],
)
def test_registry_rejects_unknown_contract_fields(mutate, expected_path):
    document = _registry(
        _supported_checkpoint(),
        default_ptm="dino.resnet50.v1",
    )
    mutate(document)
    with pytest.raises(PTMRegistryValidationError, match=expected_path):
        validate_ptm_registry(document)


def test_packaged_dino_inventory_matches_frozen_official_metadata():
    registry = load_ptm_registry()
    document = registry.to_dict()
    dino = document["models"]["dino"]
    assert dino["default_ptm"] == "dino.coco.resnet50.trainable.v1.0"

    records = {record["id"]: record for record in dino["checkpoints"]}
    expected = {
        "dino.coco.resnet50.trainable.v1.0": {
            "status": "supported",
            "resource": "pretrained_dino_coco",
            "version": "dino_resnet_50_trainable_v1.0",
            "member": "dino_resnet50_ep12.pth",
            "size": 568767395,
            "sha256": (
                "7a391fb84a18714b60258becdb512594ec54faff5dccbf17ca53c5d902137512"
            ),
        },
        "dino.coco.gcvit_tiny.trainable.v1.0": {
            "status": "unsupported",
            "resource": "pretrained_dino_coco",
            "version": "dino_gc_vit_tiny_trainable_v1.0",
            "member": "dino_gcvit_tiny0_ep12.pth",
            "size": 577043347,
            "sha256": (
                "6322af1a26eb025139bc7bfe32591e38fe2998c998e376bab7ca644261d2bfbe"
            ),
        },
        "dino.coco.fan_small.trainable.v1.0": {
            "status": "unverified",
            "resource": "pretrained_dino_coco",
            "version": "dino_fan_small_trainable_v1.0",
            "member": "dino_fan_small_ep12.pth",
            "size": 580862106,
            "sha256": (
                "df3e4e07d411f3d61c882ee9e61d5cb7cef613d1e4748d75b9b86e3dd1c83185"
            ),
        },
        "dino.coco.fan_large.trainable.v1.0": {
            "status": "unverified",
            "resource": "pretrained_dino_coco",
            "version": "dino_fan_large_trainable_v1.0",
            "member": "dino_fan_large_imagenet22k_36ep.pth",
            "size": 1197490926,
            "sha256": (
                "8e9f8d865a315a40d4854cc2846317a03ca1f87a40a9ddde6fa7631828f73cb8"
            ),
        },
        "dino.coco.nvdinov2_large.trainable.v1.0": {
            "status": "unverified",
            "resource": "dino_with_fm_backbone",
            "version": "trainable_v1.0",
            "member": "dino_nvdinov2_518_1536_coco_e36.pth",
            "size": 4232490433,
            "sha256": (
                "013e8e6e6a0a913ac56cf1a581f9d1dd7abe6cb47a57664a21290d0f44866c78"
            ),
        },
    }
    assert len(records) == 31
    assert set(expected).issubset(records)

    for checkpoint_id, values in expected.items():
        record = records[checkpoint_id]
        source = record["source"]
        assert record["status"] == values["status"]
        assert source["provider"] == "ngc"
        assert source["registry"] == "nvidia/tao"
        assert source["official"] is True
        assert source["resource"] == values["resource"]
        assert source["version"] == values["version"]
        assert source["member"] == values["member"]
        assert source["version"].lower() != "latest"
        assert record["expected_size_bytes"] == values["size"]
        assert record.get("sha256") == values["sha256"]


def test_packaged_dino_registry_exactly_covers_official_trainable_inventory():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "cross_model_automl_20260729"
        / "dino_ptm_inventory"
        / "source_inventory.v1.json"
    )
    source_inventory = json.loads(source_path.read_text(encoding="utf-8"))
    official = {
        (
            resource["resource"],
            record["version"],
            record["member"],
        ): record
        for resource in source_inventory["resources"]
        for record in resource["records"]
    }
    official_trainable = {
        key: record
        for key, record in official.items()
        if record["automl_training_artifact"]
    }
    official_deployable = {
        key: record
        for key, record in official.items()
        if not record["automl_training_artifact"]
    }

    registry = load_ptm_registry().to_dict()
    dino = registry["models"]["dino"]
    records = dino["checkpoints"]
    packaged = {
        (
            record["source"]["resource"],
            record["source"]["version"],
            record["source"]["member"],
        ): record
        for record in records
    }

    assert dino["default_ptm"] == "dino.coco.resnet50.trainable.v1.0"
    assert len(official) == 35
    assert len(official_trainable) == len(packaged) == 31
    assert len(official_deployable) == 4
    assert set(packaged) == set(official_trainable)
    assert set(packaged).isdisjoint(official_deployable)
    assert all(
        record["artifact_role"] == "full_detector_deployable"
        for record in official_deployable.values()
    )
    qualification_backbone_corrections = {
        (
            "dino_with_fm_backbone",
            "trainable_v1.0",
            "dino_nvdinov2_518_1536_coco_e36.pth",
        ): "vit_large_nvdinov2",
    }

    for identity, source in official_trainable.items():
        record = packaged[identity]
        assert record["source"]["immutable_identity"] == (
            source["immutable_identity"]
        )
        assert record["expected_size_bytes"] == source["size_bytes"]
        expected_backbone = qualification_backbone_corrections.get(
            identity,
            source["runtime_backbone"],
        )
        assert record["backbone"] == expected_backbone
        if identity in qualification_backbone_corrections:
            assert source["runtime_backbone"] == "vit_large_dinov2"
        assert record["checkpoint_target"] == source["checkpoint_target"]
        assert record["input_contract"]["channels"] == 3
        assert record["input_contract"]["height"] == (
            source["resolution"]["height"]
        )
        assert record["input_contract"]["width"] == (
            source["resolution"]["width"]
        )


def test_dino_backbone_records_preserve_qualification_runtime_boundary():
    records = load_ptm_registry().to_dict()["models"]["dino"]["checkpoints"]
    backbones = [
        record
        for record in records
        if record["source"]["resource"]
        in {"pretrained_dino_imagenet", "pretrained_dino_nvimagenet"}
    ]
    assert len(backbones) == 26

    gc_vit = [
        record for record in backbones if record["backbone"].startswith("gc_vit")
    ]
    qualification_candidates = [
        record
        for record in backbones
        if not record["backbone"].startswith("gc_vit")
    ]
    assert len(gc_vit) == 12
    assert len(qualification_candidates) == 14

    source_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "cross_model_automl_20260729"
        / "dino_ptm_inventory"
        / "source_inventory.v1.json"
    )
    source_inventory = json.loads(source_path.read_text(encoding="utf-8"))
    source_gc_vit = [
        record
        for resource in source_inventory["resources"]
        for record in resource["records"]
        if record["runtime_backbone"].startswith("gc_vit")
    ]
    registry_gc_vit = [
        record for record in records if record["backbone"].startswith("gc_vit")
    ]
    assert len(source_gc_vit) == 14
    assert len(registry_gc_vit) == 13
    assert all(record["status"] == "unsupported" for record in registry_gc_vit)
    deployable_gc_vit = [
        record
        for record in source_gc_vit
        if not record["automl_training_artifact"]
    ]
    assert len(deployable_gc_vit) == 1
    assert deployable_gc_vit[0]["runtime_status"] == (
        "excluded_static_tao71_gcvit"
    )

    for record in gc_vit:
        assert record["status"] == "unsupported"
        assert (
            "0dfaf824025b69e66c0c239ec6e11e64f6b2eecdcd42b41f9226e14f1348568e"
            in record["status_reason"]
        )
        assert "no GCViT import, dictionary, or construction branch" in (
            record["status_reason"]
        )
    for record in qualification_candidates:
        assert record["status"] == "unverified"
        if record["id"] == "dino.backbone.nvimagenet.resnet50":
            assert record["sha256"] == (
                "49b0df2b517a28760e17158c9ad78371"
                "c1f833d6ad257f117ff81356743060b7"
            )
        else:
            assert "sha256" not in record
        assert "not runtime eligible" in record["status_reason"]

    compatibility = load_ptm_registry().compatibility(
        "dino",
        tao_version="7.0.1-pyt",
        task="object_detection",
    )
    excluded = {
        item.checkpoint_id: item for item in compatibility.excluded
    }
    for record in backbones:
        expected_code = (
            "status_unsupported"
            if record["status"] == "unsupported"
            else "status_unverified"
        )
        assert excluded[record["id"]].codes == (expected_code,)


def test_packaged_dino_tao71_adapters_match_pinned_serializer_evidence():
    records = {
        record["id"]: record
        for record in load_ptm_registry().to_dict()["models"]["dino"][
            "checkpoints"
        ]
    }
    expected = {
        "dino.coco.resnet50.trainable.v1.0": (
            "tao71_dino_resnet50_ep12.pth",
            195109331,
            "678064a0706ec778edb17583be78e9a138afac1c48832ba419b8c774ac7d5756",
        ),
        "dino.coco.fan_small.trainable.v1.0": (
            "tao71_dino_fan_small_ep12.pth",
            193716107,
            "0a9e5ebfba383bbba8084db72a595bac2be512742998a2b4c0168b4300f3b580",
        ),
        "dino.coco.fan_large.trainable.v1.0": (
            "tao71_dino_fan_large_imagenet22k_36ep.pth",
            399422743,
            "149b670a4ca0cb701bdd32c69244593f2d6c699fd0d8b1851a9ad385434c7303",
        ),
        "dino.coco.nvdinov2_large.trainable.v1.0": (
            "tao71_dino_nvdinov2_518_1536_coco_e36.pth",
            1410846731,
            "d7bacddff9393d5f37ecca67686467bce2cd77d95b26c6a01908a90dbc6b6333",
        ),
    }
    for checkpoint_id, output_values in expected.items():
        adapters = records[checkpoint_id]["artifact_adapters"]
        assert len(adapters) == 1
        adapter = adapters[0]
        assert adapter["compatible_tao_versions"] == ["==7.1.0"]
        assert adapter["recipe"] == {
            "retain_top_level_keys": ["state_dict"],
            "add_top_level_metadata": {"tao_model": "dino"},
            "tensor_container_key": "state_dict",
            "require_exact_tensor_key_set": True,
            "require_exact_tensor_values": True,
        }
        assert (
            adapter["output"]["member"],
            adapter["output"]["expected_size_bytes"],
            adapter["output"]["sha256"],
        ) == output_values

    assert "artifact_adapters" not in records[
        "dino.coco.gcvit_tiny.trainable.v1.0"
    ]
    registry = load_ptm_registry()
    assert registry.artifact_adapter(
        "dino.coco.resnet50.trainable.v1.0",
        tao_version="7.0.1-pyt",
    ) is None
    assert registry.artifact_adapter(
        "dino.coco.resnet50.trainable.v1.0",
        tao_version="7.1.0-rc245",
    )["output"]["sha256"] == expected[
        "dino.coco.resnet50.trainable.v1.0"
    ][2]


def test_artifact_adapter_contract_is_declarative_strict_and_unambiguous():
    record = _supported_checkpoint()
    record["artifact_adapters"] = [_artifact_adapter()]
    registry = PTMRegistry(_registry(record, default_ptm=record["id"]))
    resolved = registry.artifact_adapter(record["id"], tao_version="7.1.0")
    assert resolved["recipe"]["add_top_level_metadata"] == {
        "tao_model": "dino"
    }
    assert registry.artifact_adapter(record["id"], tao_version="7.0.1") is None

    executable = copy.deepcopy(record)
    executable["artifact_adapters"][0]["command"] = "python adapter.py"
    with pytest.raises(PTMRegistryValidationError, match="unsupported field"):
        validate_ptm_registry(
            _registry(executable, default_ptm=executable["id"])
        )

    mismatch = copy.deepcopy(record)
    mismatch["artifact_adapters"][0]["recipe"][
        "tensor_container_key"
    ] = "weights"
    with pytest.raises(PTMRegistryValidationError, match="must be retained"):
        validate_ptm_registry(
            _registry(mismatch, default_ptm=mismatch["id"])
        )

    ambiguous = copy.deepcopy(record)
    ambiguous["artifact_adapters"].append(
        _artifact_adapter(
            versions=[">=7.1,<7.2"],
            adapter_id="dino.tao71.metadata_wrapper.v2",
        )
    )
    ambiguous_registry = PTMRegistry(
        _registry(ambiguous, default_ptm=ambiguous["id"])
    )
    with pytest.raises(
        PTMArtifactAdapterResolutionError,
        match="2 artifact adapters",
    ):
        ambiguous_registry.artifact_adapter(
            ambiguous["id"],
            tao_version="7.1.0",
        )


def test_packaged_dino_compatibility_is_fail_closed_for_tao_701():
    result = load_ptm_registry().compatibility(
        "dino",
        tao_version="7.0.1-pyt",
        task="object_detection",
    )
    assert result.eligible_checkpoint_ids == (
        "dino.coco.resnet50.trainable.v1.0",
    )
    assert result.default_checkpoint_id == (
        "dino.coco.resnet50.trainable.v1.0"
    )

    exclusions = {item.checkpoint_id: item for item in result.excluded}
    assert exclusions["dino.coco.gcvit_tiny.trainable.v1.0"].codes == (
        "status_unsupported",
    )
    assert "no GCViT import, dictionary, or construction branch" in (
        exclusions["dino.coco.gcvit_tiny.trainable.v1.0"].reasons[0]
    )
    for checkpoint_id in (
        "dino.coco.fan_small.trainable.v1.0",
        "dino.coco.fan_large.trainable.v1.0",
    ):
        assert exclusions[checkpoint_id].codes == ("status_unverified",)
        assert "explicit target-release qualification" in (
            exclusions[checkpoint_id].reasons[0]
        )
    nvdino = exclusions["dino.coco.nvdinov2_large.trainable.v1.0"]
    assert nvdino.codes == ("status_unverified",)
    assert "load smoke test" in nvdino.reasons[0]


def test_packaged_unverified_records_are_available_only_to_qualification():
    registry = load_ptm_registry()
    qualification = registry.qualification(
        "dino",
        tao_version="7.0.1-pyt",
        task="object_detection",
        validation_statuses=("unverified",),
    )
    records = registry.to_dict()["models"]["dino"]["checkpoints"]
    expected = tuple(sorted(
        record["id"] for record in records if record["status"] == "unverified"
    ))
    assert len(expected) == 17
    assert qualification.candidate_checkpoint_ids == expected
    assert all(
        record["id"] not in qualification.candidate_checkpoint_ids
        for record in records
        if record["status"] in {"supported", "unsupported"}
    )
    assert qualification.to_dict()["runtime_eligible"] is False

    target_release = registry.qualification(
        "dino",
        tao_version="7.1.0",
        task="object_detection",
        validation_statuses=("supported", "unverified"),
    )
    assert len(target_release.candidate_checkpoint_ids) == 18
    assert "dino.coco.resnet50.trainable.v1.0" in (
        target_release.candidate_checkpoint_ids
    )
    assert all(
        record["id"] not in target_release.candidate_checkpoint_ids
        for record in records
        if record["status"] == "unsupported"
    )
    assert target_release.to_dict()["runtime_eligible"] is False


def test_all_packaged_dino_sidecars_match_registered_overrides_and_hashes():
    from importlib import resources

    registry = load_ptm_registry()
    records = registry.to_dict()["models"]["dino"]["checkpoints"]
    for record in records:
        sidecar = record["checkpoint_spec_file"]
        assert sidecar["source"] == "repository"
        verification = verify_packaged_resource_sha256(
            sidecar["path"],
            sidecar["sha256"],
        )
        assert verification.ok, record["id"]

        resource = resources.files("tao_automl").joinpath(
            *sidecar["path"].split("/")
        )
        sidecar_spec = yaml.safe_load(resource.read_text(encoding="utf-8"))
        assert sidecar_spec == record["default_spec_overrides"], record["id"]


def test_only_validated_resnet_record_is_advertised_supported():
    records = load_ptm_registry().to_dict()["models"]["dino"]["checkpoints"]
    supported = [record for record in records if record["status"] == "supported"]
    assert [record["id"] for record in supported] == [
        "dino.coco.resnet50.trainable.v1.0"
    ]
    record = supported[0]
    assert record["compatible_tao_versions"] == ["==7.0.1"]
    assert record["validation"] == {
        "status": "validated",
        "tao_version": "7.0.1",
        "container_identity": (
            "sha256:"
            "88ba75e3a8eb9524fc0dbf026f2ea5da2c68696ae8d918b0afde5e0384ca641e"
        ),
        "evidence": (
            "experiments/dino_moo_phase2_20260728/"
            "phase2_validation_report.md"
        ),
    }


def test_packaged_cross_model_inventory_is_exact_and_fail_closed():
    registry = load_ptm_registry()
    document = registry.to_dict()
    expected = {
        "deformable_detr": (
            (
                "ddetr_resnet_50_trainable_v1.0",
                "dd_resnet50_ep50.pth",
                492568963,
            ),
            (
                "ddetr_gc_vit_tiny_trainable_v1.0",
                "dd_gcvit_tiny_ep50.pth",
                497618156,
            ),
        ),
        "rtdetr": (
            (
                "trainable_resnet50_v2.0",
                "resnet50_trafficcamnet_rtdetr.pth",
                511956488,
            ),
            (
                "trainable_resnet18_v2.0",
                "resnet18_trafficcamnet_rtdetr.pth",
                357560178,
            ),
            (
                "trainable_rn50_v1.0.2",
                "rtdetr_warehouse_v1.0.2.pth",
                514392577,
            ),
            (
                "trainable_efficientvit_l2_v1.0",
                "rtdetr_warehouse_v1.0.pth",
                813085917,
            ),
        ),
        "grounding_dino": (
            (
                "grounding_dino_swin_tiny_commercial_trainable_v1.1",
                "grounding_dino_swin_tiny_commercial_trainable.pth",
                2070860191,
            ),
            (
                "grounding_dino_swin_tiny_commercial_trainable_v1.0",
                "grounding_dino_swin_tiny_commercial_trainable.pth",
                2070704394,
            ),
        ),
        "segformer": (
            (
                "trainable_fan_tiny_hybrid_v1.0",
                "cityscapes_fan_tiny_hybrid_224.pth",
                123189663,
            ),
            (
                "trainable_fan_small_hybrid_v1.0",
                "cityscapes_fan_small_hybrid_224.pth",
                348336617,
            ),
            (
                "trainable_fan_base_hybrid_v1.0",
                "cityscapes_fan_base_hybrid_224.pth",
                654187968,
            ),
            (
                "trainable_fan_large_hybrid_v1.0",
                "cityscapes_fan_large_hybrid_224.pth",
                958222841,
            ),
            ("fan_hybrid_tiny", "fan_hybrid_tiny.pth", 30032227),
            ("fan_hybrid_small", "fan_hybrid_small.pth", 104729466),
            (
                "fan_hybrid_base_in22k",
                "fan_hybrid_base_in22k.pth",
                234955105,
            ),
            (
                "fan_hybrid_base_in22k_1k",
                "fan_hybrid_base_in22k_1k.pth",
                202329016,
            ),
            (
                "fan_hybrid_base_in22k_1k_384",
                "fan_hybrid_base_in22k_1k_384.pth",
                202331436,
            ),
            (
                "fan_hybrid_large_in22k",
                "fan_hybrid_large_in22k.pth",
                342977114,
            ),
            (
                "fan_hybrid_large_in22k_384",
                "fan_hybrid_large_in22k_384.pth",
                308027006,
            ),
            (
                "fan_hybrid_large_in22k_1k",
                "fan_hybrid_large_in22k_1k.pth",
                308026197,
            ),
            (
                "fan_hybrid_large_in22k_1k_384",
                "fan_hybrid_large_in22k_1k_384.pth",
                308029433,
            ),
        ),
        "oneformer": (
            (
                "oneformer_ade_pretrained_research_trainable_v1.0",
                "model_epoch_003_step_02528.pth",
                2840732008,
            ),
            (
                "oneformer_coco_pretrained_research",
                "oneformer_pretrained_research.pth",
                2840372264,
            ),
            (
                "oneformer_pretrained_commercial_dinat_its",
                "oneformer_pretrained_commercial_dinat_its.pth",
                2887215443,
            ),
            (
                "oneformer_its_swinl_commercial_trainable_v1.0",
                "its_miou=82.pth",
                2834018088,
            ),
        ),
        "mask2former": (
            (
                "mask2former_swint_trainable_v1.0",
                "mask2former_swint.pth",
                569716712,
            ),
        ),
        "mask_grounding_dino": (
            (
                "mask_grounding_dino_swin_tiny_commercial_trainable_v2.1",
                "model_epoch_029_step_20642.pth",
                2216696690,
            ),
            (
                "mask_grounding_dino_swin_tiny_commercial_trainable_v2.0",
                "model_epoch_005_step_09942.pth",
                2216696690,
            ),
            (
                "mask_grounding_dino_swin_tiny_commercial_trainable_v1.0",
                "model_epoch_049.pth",
                718739024,
            ),
            (
                "mask_grounding_dino_swin_tiny_research_trainable_v2.0",
                "model_epoch_021_step_35970.pth",
                2216677507,
            ),
        ),
    }
    assert set(document["models"]) == {"dino", *expected}
    supported_models = {
        "deformable_detr",
        "rtdetr",
        "grounding_dino",
    }

    for model, exact_members in expected.items():
        config = document["models"][model]
        assert config["default_ptm"] is None
        assert all(
            record["status"]
            == ("supported" if model in supported_models else "unverified")
            for record in config["checkpoints"]
        )
        actual = {
            (
                record["source"]["version"],
                record["source"]["member"],
                record["expected_size_bytes"],
            )
            for record in config["checkpoints"]
        }
        assert actual == set(exact_members), model


def test_cross_model_ngc_checksums_are_authoritative_hex_when_available():
    registry = load_ptm_registry().to_dict()
    cross_models = set(registry["models"]) - {"dino"}
    checksums = {
        record["id"]: record["sha256"]
        for model in cross_models
        for record in registry["models"][model]["checkpoints"]
        if "sha256" in record
    }
    assert checksums == {
        "deformable_detr.coco.resnet50.trainable.v1.0": (
            "ddb80bd87fe5882c7b86e6402d9a7b91be874505330140b0d19a42b095fa7b3f"
        ),
        "deformable_detr.coco.gcvit_tiny.trainable.v1.0": (
            "519937c15c6282a9628a8abb616d0f14f1363ea4992048c9357e8ae2f7fb29ba"
        ),
        "rtdetr.trafficcam.resnet50.trainable.v2.0": (
            "9e21450a1eac2012ab713dc103e1655eb438a73a125e2f23b0a2ba8c0583ea6a"
        ),
        "rtdetr.trafficcam.resnet18.trainable.v2.0": (
            "f7db8cfc7b0e36bdd190cbfeebed91f71460cdb347553e0de401c06c7a9fc1dd"
        ),
        "rtdetr.warehouse.resnet50.trainable.v1.0.2": (
            "745f42d0378f915f27c00baca86d788de995475af990aac370340c235bee6377"
        ),
        "rtdetr.warehouse.efficientvit_l2.trainable.v1.0": (
            "18c21b12478855e11d7f6012a4386f515465bd47fc013ea452ee76ca3fd7ecdb"
        ),
        "grounding_dino.commercial.swin_tiny.trainable.v1.1": (
            "8ea7e089e174e72a7fe57ff63cdba5e1e4994b159e41cf72122a7e0d841beaa6"
        ),
        "grounding_dino.commercial.swin_tiny.trainable.v1.0": (
            "20c3ea116d1b841063aa5efffdd386b3d85a1c35f2d702d3c95150ef1efead73"
        ),
        "oneformer.ade20k.research.swin_large.trainable.v1.0": (
            "bd727f429eba64978afdf87fadb98a801a0574b45fe033faf3514b4045e561f4"
        ),
        "oneformer.coco.research.swin_large.trainable": (
            "1627287f51067e8511973b914d94b071414e16b4830ecd9efe5ad78b97207c6c"
        ),
        "oneformer.its.commercial.dinat_large.trainable": (
            "6bbaacc876c686bcbeca1d87fc096bd59c60b567ef361f0a42fb2dacd8c75e1c"
        ),
        "oneformer.its.commercial.swin_large.trainable.v1.0": (
            "59a06b8e80fca90f5392bb9d3a4fdd51b45c6f06f617e6749949781fa0fa3443"
        ),
        "mask_grounding_dino.commercial.swin_tiny.trainable.v2.1": (
            "f9f9ef7a3d0b96e5653c5bd9d61b7c89a92a25c99cfad53e89d56472d830139b"
        ),
        "mask_grounding_dino.commercial.swin_tiny.trainable.v2.0": (
            "aaa6894df881a8959723270c507f24054618cd5321b5b9de470d1f33c74af9a5"
        ),
        "mask_grounding_dino.research.swin_tiny.trainable.v2.0": (
            "80d94e276468dcb003b701c77d3524ec2e99a2d44d1c4d32068f5ea5ee6cdf16"
        ),
    }
    assert all(
        len(value) == 64
        and value == value.lower()
        and set(value) <= set("0123456789abcdef")
        for value in checksums.values()
    )


def test_oneformer_records_package_exact_path_free_architecture_specs():
    registry = load_ptm_registry().to_dict()
    records = registry["models"]["oneformer"]["checkpoints"]
    expected = {
        "oneformer.ade20k.research.swin_large.trainable.v1.0": (
            "da8997de338775ade30865ab3d500f1a968432a56ad979ac9302d90c742bf2ce",
            "D2SwinTransformer",
            250,
        ),
        "oneformer.coco.research.swin_large.trainable": (
            "5d85e7be1a37690151e05195153c2f5eacd8e2027bf6f401a8846244aef28f3d",
            "D2SwinTransformer",
            150,
        ),
        "oneformer.its.commercial.dinat_large.trainable": (
            "362abebb68337d55b918d2a70fc91fa8f816640b6aae62ebd1cd9c5a24281211",
            "D2DiNAT",
            150,
        ),
        "oneformer.its.commercial.swin_large.trainable.v1.0": (
            "444746e8c4b0a2f5ac3386e3945509272249b38c7993d498ec209ecb211ab851",
            "D2SwinTransformer",
            150,
        ),
    }
    assert {record["id"] for record in records} == set(expected)
    package_root = Path(__file__).parents[1] / "src/tao_automl"
    for record in records:
        digest, backbone, queries = expected[record["id"]]
        assert record["status"] == "unverified"
        assert record["compatible_tao_versions"] == ["==7.1.0"]
        assert record["default_spec_overrides"]["model"]["backbone"]["name"] == backbone
        assert (
            record["default_spec_overrides"]["model"]["one_former"][
                "num_object_queries"
            ]
            == queries
        )
        spec = record["checkpoint_spec_file"]
        assert spec["source"] == "repository"
        assert spec["sha256"] == digest
        path = package_root / spec["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_cross_model_runtime_resolution_uses_qualified_status():
    registry = load_ptm_registry()
    tasks = {
        "deformable_detr": "object_detection",
        "rtdetr": "object_detection",
        "grounding_dino": "grounded_object_detection",
        "segformer": "semantic_segmentation",
        "oneformer": "panoptic_segmentation",
        "mask2former": "instance_segmentation",
        "mask_grounding_dino": "grounded_instance_segmentation",
    }
    for model, task in tasks.items():
        result = registry.compatibility(
            model,
            tao_version="7.1.0",
            task=task,
        )
        assert result.default_checkpoint_id is None
        if model in {"deformable_detr", "rtdetr", "grounding_dino"}:
            assert result.ok, model
            assert len(result.eligible_checkpoint_ids) == {
                "deformable_detr": 2,
                "rtdetr": 4,
                "grounding_dino": 2,
            }[model]
            assert result.excluded == ()
        else:
            assert not result.ok, model
            assert result.eligible_checkpoint_ids == ()
            assert all(
                exclusion.codes == ("status_unverified",)
                for exclusion in result.excluded
            )


def test_cross_model_repository_sidecars_match_registered_path_free_overrides():
    from importlib import resources

    registry = load_ptm_registry().to_dict()
    records = [
        record
        for model, config in registry["models"].items()
        if model != "dino"
        for record in config["checkpoints"]
        if "checkpoint_spec_file" in record
    ]
    assert len(records) == 17
    for record in records:
        sidecar = record["checkpoint_spec_file"]
        verification = verify_packaged_resource_sha256(
            sidecar["path"],
            sidecar["sha256"],
        )
        assert verification.ok, record["id"]
        resource = resources.files("tao_automl").joinpath(
            *sidecar["path"].split("/")
        )
        sidecar_spec = yaml.safe_load(resource.read_text(encoding="utf-8"))
        overrides = record["default_spec_overrides"]
        if record["model_family"] == "rtdetr":
            assert sidecar_spec == {"model": overrides["model"]}, record["id"]
            augmentation = overrides["dataset"]["augmentation"]
            assert augmentation["train_spatial_size"] == [
                record["input_contract"]["height"],
                record["input_contract"]["width"],
            ]
            assert augmentation["eval_spatial_size"] == [
                record["input_contract"]["height"],
                record["input_contract"]["width"],
            ]
            assert (
                augmentation["preserve_aspect_ratio"]
                == record["input_contract"]["preprocessing"][
                    "preserve_aspect_ratio"
                ]
            )
        else:
            assert sidecar_spec == overrides, record["id"]
        serialized = resource.read_text(encoding="utf-8")
        assert "/lustre/" not in serialized
        assert "/datasets/" not in serialized
        assert "/checkpoints/" not in serialized


def test_rich_cross_model_records_follow_qualification_state():
    registry = load_ptm_registry()
    expected_counts = {
        "mask2former": ("instance_segmentation", 1),
    }
    for model, (task, expected_count) in expected_counts.items():
        result = registry.qualification(
            model,
            tao_version="7.1.0",
            task=task,
            validation_statuses=("unverified",),
        )
        assert len(result.candidate_checkpoint_ids) == expected_count, model
        assert result.to_dict()["runtime_eligible"] is False
    for model, expected_count in {"deformable_detr": 2, "rtdetr": 4}.items():
        result = registry.compatibility(
            model,
            tao_version="7.1.0",
            task="object_detection",
        )
        assert len(result.eligible_checkpoint_ids) == expected_count, model
        assert result.ok


def test_complete_supported_record_is_valid():
    document = _registry(
        _supported_checkpoint(),
        default_ptm="dino.resnet50.v1",
    )
    validate_ptm_registry(document)
    registry = PTMRegistry(document)
    assert registry.checkpoint("dino.resnet50.v1")["backbone"] == "resnet_50"


def test_supported_record_accepts_checksum_verified_repository_spec_sidecar():
    record = _supported_checkpoint()
    record["checkpoint_spec_file"] = {
        "source": "repository",
        "path": "data/ptm_specs/dino/dino.resnet50.v1.yaml",
        "sha256": "a" * 64,
        "provenance": {
            "source": "TAO 7.0.1 DINO experiment defaults",
            "evidence": "registry-provenance/dino.resnet50.v1.json",
        },
    }
    validate_ptm_registry(
        _registry(record, default_ptm="dino.resnet50.v1")
    )


def test_repository_spec_sidecar_is_fail_closed():
    record = _supported_checkpoint()
    record["checkpoint_spec_file"] = {
        "source": "repository",
        "path": "../outside.yaml",
        "provenance": {},
    }
    with pytest.raises(PTMRegistryValidationError) as error:
        validate_ptm_registry(
            _registry(record, default_ptm="dino.resnet50.v1")
        )
    message = str(error.value)
    assert "safe package-relative path under data/" in message
    assert "sha256 is required" in message
    assert "provenance.source" in message
    assert "provenance.evidence" in message


def test_registry_loader_fails_closed_on_missing_repository_sidecar(tmp_path):
    record = _supported_checkpoint()
    record["checkpoint_spec_file"] = {
        "source": "repository",
        "path": "data/ptm_specs/dino/missing.yaml",
        "sha256": "a" * 64,
        "provenance": {
            "source": "TAO defaults",
            "evidence": "registry-provenance/missing.json",
        },
    }
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(_registry(record, default_ptm=record["id"])),
        encoding="utf-8",
    )
    with pytest.raises(
        PTMRegistryValidationError,
        match="repository sidecar.*missing",
    ):
        load_ptm_registry(path)

    structurally_valid = load_ptm_registry(
        path,
        verify_repository_sidecars=False,
    )
    verification = structurally_valid.repository_sidecar_verifications()[0]
    assert not verification.ok
    assert verification.checksum.code == "artifact_missing"


def test_supported_record_rejects_missing_authoritative_fields():
    record = {
        "id": "dino.incomplete",
        "status": "supported",
        "model_family": "dino",
    }
    with pytest.raises(PTMRegistryValidationError) as error:
        validate_ptm_registry(_registry(record, default_ptm="dino.incomplete"))

    message = str(error.value)
    assert "source must be an object" in message
    assert "compatible_tao_versions" in message
    assert "checkpoint_spec_file" in message
    assert "validation must be an object" in message


@pytest.mark.parametrize("status", ["unverified", "unsupported", "deprecated"])
def test_non_supported_records_require_explicit_reason(status):
    record = {"id": f"dino.{status}", "status": status}
    with pytest.raises(PTMRegistryValidationError, match="status_reason"):
        validate_ptm_registry(_registry(record))

    record["status_reason"] = "Official immutable metadata has not been verified"
    validate_ptm_registry(_registry(record))


def test_default_must_reference_local_supported_record():
    unsupported = {
        "id": "dino.legacy",
        "status": "unsupported",
        "status_reason": "Checkpoint does not load with this TAO release",
    }
    with pytest.raises(
        PTMRegistryValidationError,
        match="default_ptm must reference a supported checkpoint",
    ):
        validate_ptm_registry(_registry(unsupported, default_ptm="dino.legacy"))


def test_checkpoint_ids_are_globally_unique():
    first = _supported_checkpoint("shared.id")
    document = _registry(first, default_ptm="shared.id")
    document["models"]["segformer"] = {
        "default_ptm": None,
        "checkpoints": [
            {
                "id": "shared.id",
                "status": "unverified",
                "status_reason": "Pending authoritative metadata",
            }
        ],
    }
    with pytest.raises(PTMRegistryValidationError, match="duplicates"):
        validate_ptm_registry(document)


def test_compatibility_returns_eligible_and_structured_exclusions():
    supported = _supported_checkpoint()
    wrong_task = _supported_checkpoint("dino.grounding-only")
    wrong_task["task_compatibility"] = ["grounding"]
    unverified = {
        "id": "dino.pending",
        "status": "unverified",
        "status_reason": "Checkpoint checksum is not yet verified",
    }
    registry = PTMRegistry(
        _registry(
            supported,
            wrong_task,
            unverified,
            default_ptm=supported["id"],
        )
    )

    result = registry.compatibility(
        "dino",
        tao_version="7.0.1-pyt",
        task="object_detection",
    )
    assert result.ok
    assert result.eligible_checkpoint_ids == ("dino.resnet50.v1",)
    assert result.default_checkpoint_id == "dino.resnet50.v1"
    exclusions = {item.checkpoint_id: item for item in result.excluded}
    assert exclusions["dino.grounding-only"].codes == ("task_incompatible",)
    assert exclusions["dino.pending"].codes == ("status_unverified",)

    incompatible = registry.compatibility(
        "dino",
        tao_version="8.0.0",
        task="object_detection",
    )
    version_exclusion = {
        item.checkpoint_id: item for item in incompatible.excluded
    }["dino.resnet50.v1"]
    assert version_exclusion.codes == ("tao_version_incompatible",)
    assert incompatible.default_checkpoint_id is None


def test_qualification_is_explicit_strict_and_never_runtime_eligible():
    supported = _supported_checkpoint()
    pending = _supported_checkpoint("dino.pending")
    pending["status"] = "unverified"
    pending["status_reason"] = "Load validation is pending"
    pending.pop("validation")
    pending.pop("compatible_tao_versions")
    incomplete = {
        "id": "dino.incomplete",
        "status": "unverified",
        "status_reason": "Discovery metadata only",
    }
    registry = PTMRegistry(
        _registry(
            supported,
            pending,
            incomplete,
            default_ptm=supported["id"],
        )
    )

    runtime = registry.compatibility(
        "dino",
        tao_version="7.0.1",
        task="object_detection",
    )
    assert runtime.eligible_checkpoint_ids == ("dino.resnet50.v1",)
    assert "dino.pending" not in runtime.eligible_checkpoint_ids

    qualification = registry.qualification(
        "dino",
        tao_version="7.0.1-pyt",
        task="object_detection",
        validation_statuses=("unverified",),
    )
    assert qualification.candidate_checkpoint_ids == ("dino.pending",)
    assert qualification.to_dict()["runtime_eligible"] is False
    exclusions = {
        item.checkpoint_id: item for item in qualification.excluded
    }
    assert exclusions["dino.incomplete"].codes == (
        "qualification_metadata_incomplete",
    )
    assert any(
        "source must be an object" in reason
        for reason in exclusions["dino.incomplete"].reasons
    )
    assert exclusions["dino.resnet50.v1"].codes == (
        "status_not_requested_for_qualification",
    )


def test_qualification_can_revalidate_unsupported_on_a_new_tao_version():
    legacy = _supported_checkpoint("dino.legacy")
    legacy["status"] = "unsupported"
    legacy["status_reason"] = "Failed its prior TAO-version smoke test"
    legacy.pop("validation")
    legacy["compatible_tao_versions"] = ["==7.0.1"]
    registry = PTMRegistry(_registry(legacy))

    qualification = registry.qualification(
        "dino",
        tao_version="8.0.0",
        task="object_detection",
        validation_statuses=("unsupported",),
    )
    assert qualification.candidate_checkpoint_ids == ("dino.legacy",)


def test_qualification_can_revalidate_supported_checkpoint_for_new_version():
    checkpoint = _supported_checkpoint("dino.release-7")
    checkpoint["compatible_tao_versions"] = ["==7.0.1"]
    registry = PTMRegistry(
        _registry(checkpoint, default_ptm=checkpoint["id"])
    )

    qualification = registry.qualification(
        "dino",
        tao_version="7.1.0rc245",
        task="object_detection",
        validation_statuses=("supported",),
    )
    assert qualification.candidate_checkpoint_ids == ("dino.release-7",)
    assert qualification.to_dict()["runtime_eligible"] is False

    already_compatible = registry.qualification(
        "dino",
        tao_version="7.0.1",
        task="object_detection",
        validation_statuses=("supported",),
    )
    assert already_compatible.candidate_checkpoint_ids == ()
    assert already_compatible.excluded[0].codes == (
        "already_runtime_compatible",
    )


@pytest.mark.parametrize(
    "statuses",
    [
        (),
        ("deprecated",),
        "unverified",
        None,
        (True,),
    ],
)
def test_qualification_rejects_implicit_or_runtime_statuses(statuses):
    registry = PTMRegistry(_registry(_supported_checkpoint()))
    with pytest.raises(ValueError, match="validation_statuses"):
        registry.qualification(
            "dino",
            tao_version="7.0.1",
            task="object_detection",
            validation_statuses=statuses,
        )


def test_missing_model_is_a_structured_result():
    result = load_ptm_registry().compatibility(
        "unknown_model",
        tao_version="7.0.1",
        task="object_detection",
    )
    assert not result.ok
    assert not result.model_found
    assert result.reasons == (
        "Model 'unknown_model' is not present in the PTM registry",
    )


def test_registry_returns_defensive_copies():
    document = _registry(
        _supported_checkpoint(),
        default_ptm="dino.resnet50.v1",
    )
    registry = PTMRegistry(document)
    document["models"]["dino"]["checkpoints"][0]["backbone"] = "mutated"
    record = registry.checkpoint("dino.resnet50.v1")
    record["backbone"] = "also-mutated"
    assert registry.checkpoint("dino.resnet50.v1")["backbone"] == "resnet_50"


def test_spec_precedence_is_deterministic_and_does_not_mutate_inputs():
    defaults = {
        "model": {"backbone": "resnet_50", "num_queries": 300},
        "train": {"optim": {"lr": 1e-4, "weight_decay": 1e-4}},
        "dataset": {"sources": [{"name": "default"}]},
    }
    defaults_before = copy.deepcopy(defaults)
    result = merge_ptm_spec_precedence(
        model_defaults=defaults,
        ptm_overrides={"model": {"num_queries": 900}},
        automl_profile_overrides={"train.optim.lr": 2e-4},
        user_overrides={"dataset.sources[0].name": "user"},
        candidate_overrides={
            "train": {"optim": {"lr": 3e-4}},
            "model.num_queries": 600,
        },
    )

    assert defaults == defaults_before
    assert result.precedence == SPEC_PRECEDENCE
    assert result.spec["model"]["num_queries"] == 600
    assert result.spec["train"]["optim"] == {
        "lr": 3e-4,
        "weight_decay": 1e-4,
    }
    assert result.spec["dataset"]["sources"][0]["name"] == "user"
    assert result.final_sha256 == canonical_sha256(result.spec)
    assert any(
        item.path == "train.optim.lr"
        and item.replacement_layer == "candidate_overrides"
        for item in result.overwritten
    )

    reordered = merge_ptm_spec_precedence(
        model_defaults={
            "dataset": {"sources": [{"name": "default"}]},
            "train": {"optim": {"weight_decay": 1e-4, "lr": 1e-4}},
            "model": {"num_queries": 300, "backbone": "resnet_50"},
        },
        ptm_overrides={"model": {"num_queries": 900}},
        automl_profile_overrides={"train.optim.lr": 2e-4},
        user_overrides={"dataset.sources[0].name": "user"},
        candidate_overrides={
            "model.num_queries": 600,
            "train": {"optim": {"lr": 3e-4}},
        },
    )
    assert reordered.final_sha256 == result.final_sha256
    assert reordered.layer_sha256 == result.layer_sha256


def test_spec_precedence_rejects_ambiguous_paths_inside_one_layer():
    with pytest.raises(ValueError, match="assigns spec path .* more than once"):
        merge_ptm_spec_precedence(
            model_defaults={"model": {"num_queries": 300}},
            user_overrides={
                "model": {"num_queries": 600},
                "model.num_queries": 900,
            },
        )

    with pytest.raises(ValueError, match="invalid dotted/indexed spec path"):
        merge_ptm_spec_precedence(
            model_defaults={"model": {"num_queries": 300}},
            user_overrides={"model..num_queries": 900},
        )


def test_canonical_sha_rejects_nan():
    with pytest.raises(ValueError, match="not canonically JSON serializable"):
        canonical_sha256({"metric": float("nan")})


def test_checksum_verification_hooks(tmp_path):
    artifact = tmp_path / "checkpoint.pth"
    artifact.write_bytes(b"checkpoint")
    expected = hashlib.sha256(b"checkpoint").hexdigest()

    assert sha256_file(artifact) == expected
    verified = verify_file_sha256(artifact, expected.upper())
    assert verified.ok
    assert verified.code == "verified"
    assert verified.actual_sha256 == expected

    mismatch = verify_file_sha256(artifact, "0" * 64)
    assert not mismatch.ok
    assert mismatch.code == "checksum_mismatch"

    missing = verify_file_sha256(tmp_path / "missing.pth", expected)
    assert not missing.ok
    assert missing.code == "artifact_missing"

    unavailable = verify_file_sha256(artifact, None)
    assert not unavailable.ok
    assert unavailable.code == "checksum_unavailable"

    invalid = verify_file_sha256(artifact, "not-a-digest")
    assert not invalid.ok
    assert invalid.code == "invalid_expected_checksum"


def test_checksum_read_failure_does_not_expose_raw_exception(monkeypatch, tmp_path):
    import tao_automl.ptm_registry as registry_module

    artifact = tmp_path / "checkpoint.pth"
    artifact.write_bytes(b"checkpoint")
    expected = hashlib.sha256(b"checkpoint").hexdigest()

    def fail_read(_path):
        raise OSError("secret mount name and signed-url query")

    monkeypatch.setattr(registry_module, "sha256_file", fail_read)
    result = verify_file_sha256(artifact, expected)
    assert result.code == "artifact_unreadable"
    assert "secret mount" not in result.reason
    assert "signed-url" not in result.reason


def test_packaged_resource_checksum_hook():
    from importlib import resources

    resource = resources.files("tao_automl").joinpath(
        "data", "ptm_registry.v1.json"
    )
    digest = hashlib.sha256(resource.read_bytes()).hexdigest()
    verified = verify_packaged_resource_sha256(
        "data/ptm_registry.v1.json",
        digest,
    )
    assert verified.ok
    assert verified.code == "verified"

    unsafe = verify_packaged_resource_sha256("../outside.yaml", digest)
    assert not unsafe.ok
    assert unsafe.code == "invalid_resource_path"


def test_registry_checkpoint_and_spec_verification(tmp_path):
    checkpoint_bytes = b"authoritative checkpoint fixture"
    spec_bytes = b"model:\n  backbone: resnet_50\n"
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint_spec = tmp_path / "experiment_spec.yaml"
    checkpoint.write_bytes(checkpoint_bytes)
    checkpoint_spec.write_bytes(spec_bytes)
    registry = PTMRegistry(
        _registry(
            _supported_checkpoint(),
            default_ptm="dino.resnet50.v1",
        )
    )

    results = registry.verify_checkpoint_files(
        "dino.resnet50.v1",
        checkpoint_path=checkpoint,
        checkpoint_spec_path=checkpoint_spec,
    )
    assert len(results) == 2
    assert all(result.ok for result in results)


def test_explicit_registry_file_load(tmp_path):
    path = tmp_path / "registry.json"
    document = _registry(
        _supported_checkpoint(),
        default_ptm="dino.resnet50.v1",
    )
    path.write_text(json.dumps(document), encoding="utf-8")
    assert load_ptm_registry(path).models == ("dino",)
