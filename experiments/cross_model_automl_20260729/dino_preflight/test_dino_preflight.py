# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from tao_automl.latency_benchmark import (
    ReplicaIdentity,
    run_replica_benchmark,
)
from tao_automl.model_preflight import canonical_sha256
from tao_automl.ptm_preflight import (
    AccessProbeResult,
    CheckpointLoadSmokeResult,
    CredentialProbeResult,
    PTMPreflightReport,
    PreparedPTM,
    ValidatedCheckpointSpec,
    VerifiedArtifact,
)
from tao_automl.ptm_registry import (
    PTMCompatibilityResult,
    canonical_sha256 as ptm_sha256,
)
from tao_automl.ptm_runtime import (
    ResolvedPTMRuntimeArm,
    ResolvedPTMRuntimeInventory,
)
from tao_automl.recommendation_audit import canonical_audit_sha256

from dino_preflight import (
    DINOPreflightCommandPlan,
    DINOPreflightContractError,
    DINOPreflightExecutionResult,
    DINOPreflightSettings,
    DINORuntimeImageContract,
    VOCRealDataIntegrityEvidence,
    build_dino_preflight_plan,
    collect_voc_real_data_integrity,
    freeze_dino_preflight_plan,
    load_dino_skill_contract,
    run_dino_local_preflight,
)


def _sha(label: str | bytes) -> str:
    data = label if isinstance(label, bytes) else label.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    root = tmp_path / "tao-train-dino"
    references = root / "references"
    schemas = root / "schemas"
    references.mkdir(parents=True)
    schemas.mkdir()
    train_inputs = {
        "dataset.train_data_sources[0].image_dir": {"type": "file"},
        "dataset.train_data_sources[0].json_file": {"type": "file"},
        "dataset.val_data_sources[0].image_dir": {"type": "file"},
        "dataset.val_data_sources[0].json_file": {"type": "file"},
        "train.pretrained_model_path": {"type": "file", "optional": True},
    }
    evaluate_inputs = {
        "evaluate.checkpoint": {"type": "file"},
        "dataset.test_data_sources.image_dir": {"type": "file"},
        "dataset.test_data_sources.json_file": {"type": "file"},
    }
    inference_inputs = {
        "inference.checkpoint": {"type": "file"},
        "dataset.infer_data_sources.image_dir": {"type": "file"},
        "dataset.infer_data_sources.classmap": {"type": "file"},
    }
    skill_info = {
        "name": "tao-train-dino",
        "network_arch": "dino",
        "automl_enabled": True,
        "container_image": (
            "nvcr.io/nvstaging/tao/tao-toolkit-pyt:"
            "7.1.0-rc-245-multiarch"
        ),
        "data_format": "coco",
        "actions": {
            "train": {
                "command": "dino train -e {config_path}",
                "config_format": "yaml",
                "mode": "config",
                "inputs": train_inputs,
                "outputs": {"results_dir": {"type": "folder"}},
                "upload_excludes": ["inputs/"],
            },
            "evaluate": {
                "command": "dino evaluate -e {config_path}",
                "config_format": "yaml",
                "mode": "config",
                "inputs": evaluate_inputs,
                "outputs": {"results_dir": {"type": "folder"}},
                "upload_excludes": ["inputs/"],
            },
            "inference": {
                "command": "dino inference -e {config_path}",
                "config_format": "yaml",
                "mode": "config",
                "inputs": inference_inputs,
                "outputs": {"results_dir": {"type": "folder"}},
                "upload_excludes": ["inputs/"],
            },
        },
    }
    (references / "skill_info.yaml").write_text(
        yaml.safe_dump(skill_info, sort_keys=True),
        encoding="utf-8",
    )
    common = {
        "wandb": {"enable": True},
        "model": {
            "backbone": "resnet_50",
            "num_queries": 300,
            "dropout_ratio": 0.0,
        },
        "dataset": {
            "train_data_sources": [{"image_dir": "", "json_file": ""}],
            "val_data_sources": [{"image_dir": "", "json_file": ""}],
            "test_data_sources": {"image_dir": "", "json_file": ""},
            "infer_data_sources": {"image_dir": [""], "classmap": ""},
            "num_classes": 91,
            "eval_class_ids": [1],
            "batch_size": 4,
        },
        "train": {
            "num_gpus": 1,
            "gpu_ids": [0],
            "num_nodes": 1,
            "seed": 1234,
            "num_epochs": 10,
            "checkpoint_interval": 1,
            "validation_interval": 1,
            "precision": "fp32",
            "pretrained_model_path": "",
        },
    }
    train_template = copy.deepcopy(common)
    evaluate_template = copy.deepcopy(common)
    evaluate_template["evaluate"] = {
        "checkpoint": "???",
        "num_gpus": 1,
        "gpu_ids": [0],
        "num_nodes": 1,
        "batch_size": -1,
    }
    inference_template = copy.deepcopy(common)
    inference_template["inference"] = {
        "checkpoint": "???",
        "num_gpus": 1,
        "gpu_ids": [0],
        "num_nodes": 1,
        "batch_size": -1,
    }
    for name, value in (
        ("train", train_template),
        ("evaluate", evaluate_template),
        ("inference", inference_template),
    ):
        (references / f"spec_template_{name}.yaml").write_text(
            yaml.safe_dump(value, sort_keys=True),
            encoding="utf-8",
        )
        schema = {
            "type": "object",
            "properties": {
                "dataset": {
                    "type": "object",
                    "properties": {
                        "num_classes": {"type": "int", "minimum": 1}
                    },
                },
                "train": {
                    "type": "object",
                    "properties": {
                        "num_gpus": {"type": "int", "minimum": 1}
                    },
                },
            },
        }
        _write_json(schemas / f"{name}.schema.json", schema)
    return root


@pytest.fixture
def voc_evidence(tmp_path: Path) -> VOCRealDataIntegrityEvidence:
    manifest_path = tmp_path / "voc_gate" / "manifest.v1.json"
    dataset_root = tmp_path / "voc_prepared"
    image_root = dataset_root / "VOCdevkit/VOC2007/JPEGImages"
    image_root.mkdir(parents=True)
    categories = [f"class_{index:02d}" for index in range(1, 21)]
    category_records = [
        {"id": index, "name": name}
        for index, name in enumerate(categories, start=1)
    ]
    train_images = [
        {"id": 11, "file_name": "000011.jpg"},
        {"id": 12, "file_name": "000012.jpg"},
    ]
    validation_images = [
        {"id": 21, "file_name": "000021.jpg"},
        {"id": 22, "file_name": "000022.jpg"},
        {"id": 23, "file_name": "000023.jpg"},
    ]
    for image in train_images + validation_images:
        (image_root / image["file_name"]).write_bytes(
            f"fixture:{image['id']}".encode()
        )
    train_document = {
        "images": train_images,
        "annotations": [
            {"id": 1, "image_id": 11, "category_id": 1},
            {"id": 2, "image_id": 12, "category_id": 20},
        ],
        "categories": category_records,
    }
    validation_document = {
        "images": validation_images,
        "annotations": [
            {"id": 1, "image_id": 21, "category_id": 2},
            {"id": 2, "image_id": 22, "category_id": 19},
            {"id": 3, "image_id": 23, "category_id": 20},
        ],
        "categories": category_records,
    }
    train_path = dataset_root / "coco/annotations/instances_train2007.json"
    validation_path = (
        dataset_root / "coco/annotations/instances_val2007.json"
    )
    _write_json(train_path, train_document)
    _write_json(validation_path, validation_document)
    manifest = {
        "schema_version": 1,
        "dataset_id": "pascal_voc_2007_full_detection",
        "categories": categories,
        "conversion_contract": {
            "output_annotations": {
                "train": "coco/annotations/instances_train2007.json",
                "val": "coco/annotations/instances_val2007.json",
            }
        },
    }
    _write_json(manifest_path, manifest)
    invariants = {
        "all_images_preserved": True,
        "all_objects_preserved": True,
        "all_categories_mapped": True,
        "all_bboxes_reversible": True,
        "all_difficult_flags_preserved": True,
        "jpeg_dimensions_verified": True,
        "source_inventory_exact": True,
        "train_val_disjoint": True,
        "trainval_test_disjoint": True,
    }
    validation = {
        "splits": {
            "train": {"images": len(train_images)},
            "val": {"images": len(validation_images)},
        },
        "invariants": invariants,
    }
    manifest_sha = _sha(manifest_path.read_bytes())
    integrity_path = dataset_root / "integrity.v1.json"
    integrity = {
        "manifest_sha256": manifest_sha,
        "dataset_terms_acknowledged": True,
        "source_tree": {
            "algorithm": "sha256",
            "digest": _sha("source-tree"),
        },
        "outputs": [
            {
                "split": "train",
                "path": train_path.relative_to(dataset_root).as_posix(),
                "size_bytes": train_path.stat().st_size,
                "sha256": _sha(train_path.read_bytes()),
            },
            {
                "split": "val",
                "path": validation_path.relative_to(dataset_root).as_posix(),
                "size_bytes": validation_path.stat().st_size,
                "sha256": _sha(validation_path.read_bytes()),
            },
        ],
        "validation": validation,
    }
    _write_json(integrity_path, integrity)
    report = {
        "status": "valid",
        "dataset_root": str(dataset_root.resolve()),
        "manifest_sha256": manifest_sha,
        "integrity_sha256": _sha(integrity_path.read_bytes()),
        "validation": validation,
    }

    def validator(**kwargs):
        assert kwargs == {
            "manifest_path": manifest_path.resolve(),
            "dataset_root": dataset_root.resolve(),
        }
        return copy.deepcopy(report)

    return collect_voc_real_data_integrity(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        validator=validator,
    )


def _artifact(
    path: Path,
    data: bytes,
    checkpoint_id: str,
    kind: str,
) -> VerifiedArtifact:
    path.write_bytes(data)
    return VerifiedArtifact(
        path=path,
        cache_relative_path=f"{checkpoint_id}/{path.name}",
        size_bytes=len(data),
        sha256=_sha(data),
        expected_sha256=_sha(data),
        verification_mode="sha256",
        cache_hit=False,
        source_identity_sha256=_sha(f"{checkpoint_id}:{kind}"),
    )


@pytest.fixture
def resolved_inventory(tmp_path: Path) -> ResolvedPTMRuntimeInventory:
    checkpoint_ids = ("dino.fixture.a", "dino.fixture.b")
    compatibility = PTMCompatibilityResult(
        model="dino",
        tao_version="7.1.0",
        task="object_detection",
        model_found=True,
        eligible_checkpoint_ids=checkpoint_ids,
        excluded=(),
        default_checkpoint_id=checkpoint_ids[0],
    )
    prepared = []
    arms = []
    for index, checkpoint_id in enumerate(checkpoint_ids):
        checkpoint = _artifact(
            tmp_path / f"checkpoint_{index}.pth",
            f"checkpoint-{index}".encode(),
            checkpoint_id,
            "checkpoint",
        )
        spec_document = {
            "model": {
                "backbone": "resnet_50",
                "num_queries": 300 + index,
                "dropout_ratio": 0.0,
            },
            "train": {
                "num_epochs": 10,
                "pretrained_model_path": "",
            },
        }
        spec_bytes = yaml.safe_dump(spec_document).encode()
        spec_artifact = _artifact(
            tmp_path / f"checkpoint_{index}.yaml",
            spec_bytes,
            checkpoint_id,
            "spec",
        )
        validated = ValidatedCheckpointSpec(
            path=spec_artifact.path,
            cache_relative_path=spec_artifact.cache_relative_path,
            artifact_sha256=spec_artifact.sha256,
            document_sha256=ptm_sha256(spec_document),
            top_level_keys=("model", "train"),
            document=copy.deepcopy(spec_document),
        )
        access = AccessProbeResult(
            ok=True,
            code="accessible",
            reason="fixture",
            status_code=200,
            remote_size_bytes=checkpoint.size_bytes,
            etag=f'"fixture-{index}"',
            exact_member_url=f"https://example.invalid/{checkpoint_id}",
        )
        smoke = CheckpointLoadSmokeResult(
            ok=True,
            code="load_passed",
            reason="fixture load passed",
            details={"checkpoint_id": checkpoint_id},
        )
        registry_record_sha = _sha(f"registry:{checkpoint_id}")
        provenance_payload = {
            "checkpoint_id": checkpoint_id,
            "purpose": "runtime",
            "registry_status": "supported",
            "runtime_eligible": True,
            "checkpoint": checkpoint.stable_dict(),
            "checkpoint_spec_artifact": spec_artifact.stable_dict(),
            "checkpoint_spec": validated.stable_dict(),
            "access_probe": access.to_dict(),
            "load_smoke": smoke.stable_dict(),
            "registry_record_sha256": registry_record_sha,
        }
        item = PreparedPTM(
            checkpoint_id=checkpoint_id,
            registry_status="supported",
            runtime_eligible=True,
            checkpoint=checkpoint,
            checkpoint_spec_artifact=spec_artifact,
            checkpoint_spec=validated,
            access_probe=access,
            load_smoke=smoke,
            registry_record_sha256=registry_record_sha,
            provenance_sha256=ptm_sha256(provenance_payload),
        )
        prepared.append(item)

    provisional_report = PTMPreflightReport(
        purpose="runtime",
        validation_statuses=("supported",),
        model="dino",
        task="object_detection",
        tao_version="7.1.0",
        registry_version="fixture-v1",
        registry_sha256=_sha("registry-document"),
        inventory=compatibility,
        credential_probe=CredentialProbeResult(
            ok=True,
            code="credential_available",
            reason="fixture credential",
        ),
        prepared=tuple(prepared),
        exclusions=(),
        report_sha256="",
    )
    report = replace(
        provisional_report,
        report_sha256=ptm_sha256(provisional_report.stable_dict()),
    )
    for index, item in enumerate(prepared):
        checkpoint_target = (
            "train.pretrained_model_path"
            if index == 0
            else "model.pretrained_backbone_path"
        )
        effective_spec = {
            "wandb": {"enable": True},
            "model": {
                "backbone": "resnet_50",
                "num_queries": 300 + index,
                "dropout_ratio": 0.0,
                "pretrained_backbone_path": (
                    str(item.checkpoint.path) if index == 1 else ""
                ),
            },
            "dataset": {
                "train_data_sources": [{"image_dir": "", "json_file": ""}],
                "val_data_sources": [{"image_dir": "", "json_file": ""}],
                "num_classes": 91,
                "batch_size": 4,
            },
            "train": {
                "pretrained_model_path": (
                    str(item.checkpoint.path) if index == 0 else ""
                ),
                "num_gpus": 1,
                "gpu_ids": [0],
                "num_nodes": 1,
                "num_epochs": 10,
                "checkpoint_interval": 1,
                "validation_interval": 1,
                "precision": "fp32",
            },
        }
        arms.append(
            ResolvedPTMRuntimeArm(
                checkpoint_id=item.checkpoint_id,
                checkpoint_target=checkpoint_target,
                checkpoint_path=str(item.checkpoint.path),
                effective_base_spec=effective_spec,
                report_sha256=report.report_sha256,
                registry_sha256=report.registry_sha256,
                registry_record_sha256=item.registry_record_sha256,
                preflight_provenance_sha256=item.provenance_sha256,
                checkpoint_artifact_sha256=item.checkpoint.sha256,
                checkpoint_spec_artifact_sha256=(
                    item.checkpoint_spec_artifact.sha256
                ),
                checkpoint_spec_document_sha256=(
                    item.checkpoint_spec.document_sha256
                ),
                input_contract_sha256=_sha(
                    f"input-contract:{item.checkpoint_id}"
                ),
                ptm_layer_sha256=_sha(f"ptm-layer:{item.checkpoint_id}"),
                effective_base_spec_sha256=ptm_sha256(effective_spec),
            )
        )
    provisional = ResolvedPTMRuntimeInventory(
        report=report,
        algorithm="bayesian",
        mode="multi_objective",
        model="dino",
        task="object_detection",
        tao_version="7.1.0",
        ptm_policy="all",
        user_checkpoint_id=None,
        objective_config_sha256=_sha("objective"),
        base_layers_sha256={
            "model_defaults": _sha("defaults"),
            "automl_profile_overrides": _sha("profile"),
            "user_overrides": _sha("user"),
        },
        arms=tuple(arms),
        inventory_sha256="",
    )
    return replace(
        provisional,
        inventory_sha256=canonical_audit_sha256(
            provisional.stable_dict()
        ),
    )


@pytest.fixture
def settings(skill_dir: Path) -> DINOPreflightSettings:
    skill = load_dino_skill_contract(skill_dir)
    image_digest = _sha("sqsh")
    return DINOPreflightSettings(
        preflight_id="dino.voc2007.local.v1",
        tao_version="7.1.0",
        source_commit="a" * 40,
        package_sha256=_sha("wheel"),
        container_sha256=image_digest,
        runtime_sha256=_sha("runtime"),
        runtime_image_contract=DINORuntimeImageContract(
            source_skill_revision="2" * 40,
            compatible_skill_revision="2" * 40,
            source_skill_image=skill.container_image,
            compatible_skill_image=skill.container_image,
            source_skill_contract_sha256=skill.sha256,
            compatible_skill_contract_sha256=skill.sha256,
            runtime_image=(
                "nvcr.io/nvstaging/tao/tao-toolkit-pyt"
                f"@sha256:{image_digest}"
            ),
            tao_schema_compatibility_sha256=_sha("tao71-schema"),
            tao_source_compatibility_sha256=_sha("tao71-source"),
        ),
        latency_input_descriptor={
            "shape": [1, 3, 544, 960],
            "dtype": "float32",
            "content": "seeded_fixture_tensor",
        },
        seed=271828,
        batch_size=1,
    )


@pytest.fixture
def plan(
    voc_evidence,
    resolved_inventory,
    skill_dir,
    settings,
) -> DINOPreflightCommandPlan:
    return build_dino_preflight_plan(
        voc_integrity=voc_evidence,
        resolved_ptm_inventory=resolved_inventory,
        skill_dir=skill_dir,
        settings=settings,
    )


class FixtureExecutor:
    def __init__(self, plan: DINOPreflightCommandPlan):
        self.plan = plan
        self.calls = []

    def __call__(self, command):
        self.calls.append(command.command_id)
        inputs = self.plan.model_preflight_inputs
        default = next(
            item
            for item in inputs.eligible_ptms
            if item.id == inputs.default_ptm_id
        )
        if command.stage == "dataset_validation":
            voc = self.plan.voc_integrity
            evidence = {
                "dataset_id": inputs.dataset_id,
                "manifest_sha256": inputs.dataset_manifest_sha256,
                "annotation_contract_sha256": (
                    inputs.annotation_contract_sha256
                ),
                "annotations_valid": True,
                "train_split_sha256": inputs.train_split_sha256,
                "validation_split_sha256": inputs.validation_split_sha256,
                "train_samples": voc.train_samples,
                "validation_samples": voc.validation_samples,
            }
        elif command.stage == "default_ptm_load":
            evidence = {
                "ptm_id": default.id,
                "checkpoint_sha256": default.checkpoint_sha256,
                "loaded": True,
                "input_contract_verified": True,
                "spec_merge_verified": True,
            }
        elif command.stage == "eligible_ptm_smoke":
            ptm = next(
                item
                for item in inputs.eligible_ptms
                if item.id == command.ptm_id
            )
            evidence = {
                "ptm_id": ptm.id,
                "checkpoint_sha256": ptm.checkpoint_sha256,
                "loaded": True,
                "train_step_passed": True,
                "validation_step_passed": True,
                "inference_step_passed": True,
            }
        elif command.stage == "default_model_full_epoch":
            evidence = {
                "ptm_id": default.id,
                "single_gpu": True,
                "completed": True,
                "completed_epochs": 1,
                "training_batches": 2,
                "distinct_training_steps": 2,
                "final_checkpoint_sha256": _sha("epoch-checkpoint"),
            }
        elif command.stage == "in_epoch_validation":
            evidence = {
                "metric_name": inputs.metric_name,
                "metric_value": 0.42,
                "completed_evaluations": 1,
                "passed": True,
            }
        elif command.stage == "standalone_evaluation":
            evidence = {
                "metric_name": inputs.metric_name,
                "metric_value": 0.44,
                "completed_evaluations": 1,
                "passed": True,
                "runtime_metric_contract_verified": True,
            }
        elif command.stage == "checkpoint_save_reload":
            evidence = {
                "ptm_id": default.id,
                "saved": True,
                "reloaded": True,
                "saved_checkpoint_sha256": _sha("epoch-checkpoint"),
                "reloaded_checkpoint_sha256": _sha("epoch-checkpoint"),
            }
        elif command.stage == "latency_instrumentation":
            tick = 0

            def clock():
                nonlocal tick
                tick += 1_000_000
                return tick

            record = run_replica_benchmark(
                contract=self.plan.latency_contract,
                identity=ReplicaIdentity(
                    rank=0,
                    world_size=1,
                    device_id="cuda:0",
                    hardware_sha256=_sha("fixture-gpu"),
                ),
                candidate_fingerprint=command.metadata[
                    "candidate_fingerprint"
                ],
                step=lambda _round, _iteration: None,
                synchronize=lambda: None,
                clock_ns=clock,
            )
            evidence = {"replica_records": [record]}
        elif command.stage == "output_artifact_validation":
            evidence = {
                "contract_sha256": inputs.output_contract_sha256,
                "artifacts": [
                    {
                        "artifact_id": artifact_id,
                        "sha256": _sha(artifact_id),
                        "size_bytes": index + 1,
                    }
                    for index, artifact_id in enumerate(
                        command.metadata["output_contract"][
                            "required_artifact_ids"
                        ]
                    )
                ],
                "missing_artifact_ids": [],
                "valid": True,
            }
        elif command.stage == "interrupted_resume_replay":
            next_request = _sha("next-request")
            evidence = {
                "interrupted": True,
                "state_saved": True,
                "state_sha256": _sha("resume-state"),
                "resumed": True,
                "replay_deterministic": True,
                "expected_next_request_sha256": next_request,
                "actual_next_request_sha256": next_request,
                "no_duplicate_trials": True,
                "no_lost_trials": True,
            }
        else:
            raise AssertionError(command.stage)
        return DINOPreflightExecutionResult(
            command_id=command.command_id,
            passed=True,
            evidence=evidence,
        )


def test_plan_freezes_complete_dino_skill_data_ptm_and_latency_contract(plan):
    plan.validate()
    assert plan.model_preflight_inputs.metric_name == "val_mAP50"
    assert plan.model_preflight_inputs.local_gpu_count == 1
    assert plan.voc_integrity.dataset_num_classes == 21
    assert len(plan.eligible_ptm_ids) == 2

    smoke_commands = plan.commands_for_stage("eligible_ptm_smoke")
    assert tuple(item.ptm_id for item in smoke_commands) == (
        "dino.fixture.a",
        "dino.fixture.b",
    )
    for command in smoke_commands:
        train = command.specs_by_action["train"]
        assert [
            dict(item)
            for item in train["dataset"]["train_data_sources"]
        ] == [{
            "image_dir": str(plan.voc_integrity.image_root),
            "json_file": str(plan.voc_integrity.train_annotation_path),
        }]
        assert [
            dict(item)
            for item in train["dataset"]["val_data_sources"]
        ] == [{
            "image_dir": str(plan.voc_integrity.image_root),
            "json_file": str(plan.voc_integrity.validation_annotation_path),
        }]
        assert train["dataset"]["num_classes"] == 21
        assert train["train"]["num_gpus"] == 1
        assert train["train"]["gpu_ids"] == (0,)
        assert train["train"]["num_epochs"] == 1
        assert command.metadata["step_limits"] == {
            "train_batches": 1,
            "validation_batches": 1,
            "inference_batches": 1,
        }
        assert command.specs_by_action["inference"]["dataset"][
            "infer_data_sources"
        ] == {
            "image_dir": ("artifact://voc2007/inference_subset",),
            "classmap": "artifact://voc2007/label_map.txt",
        }
    assert smoke_commands[0].metadata["checkpoint_target"] == (
        "train.pretrained_model_path"
    )
    assert smoke_commands[0].specs_by_action["train"]["train"][
        "pretrained_model_path"
    ].endswith("checkpoint_0.pth")
    assert smoke_commands[1].metadata["checkpoint_target"] == (
        "model.pretrained_backbone_path"
    )
    assert smoke_commands[1].specs_by_action["train"]["model"][
        "pretrained_backbone_path"
    ].endswith("checkpoint_1.pth")
    for command in smoke_commands:
        initialized = (
            "runtime://eligible_ptm_smoke/"
            f"{command.ptm_id}/initialized_model"
        )
        assert command.metadata["initialized_model_binding"] == initialized
        assert command.specs_by_action["evaluate"]["evaluate"][
            "checkpoint"
        ] == initialized
        assert command.specs_by_action["inference"]["inference"][
            "checkpoint"
        ] == initialized

    full_epoch = plan.commands_for_stage("default_model_full_epoch")[0]
    assert full_epoch.ptm_id == plan.default_ptm_id
    assert full_epoch.metadata["complete_epochs"] == 1
    assert full_epoch.metadata["single_gpu"] is True
    standalone = plan.commands_for_stage("standalone_evaluation")[0]
    assert standalone.specs_by_action["evaluate"]["evaluate"][
        "checkpoint"
    ] == "artifact://default_model_full_epoch/final_checkpoint"

    assert plan.latency_contract.warmup_iterations == 50
    assert plan.latency_contract.timed_iterations == 100
    assert plan.latency_contract.repeated_rounds == 5
    assert plan.latency_contract.expected_replicas == 1


def test_runtime_image_mapping_exposes_tag_digest_divergence_and_rejects_stale_skill(
    plan,
    voc_evidence,
    resolved_inventory,
    skill_dir,
    settings,
):
    runtime = plan.settings.runtime_image_contract
    assert plan.skill_contract.container_image == (
        "nvcr.io/nvstaging/tao/tao-toolkit-pyt:"
        "7.1.0-rc-245-multiarch"
    )
    assert runtime.runtime_image == (
        "nvcr.io/nvstaging/tao/tao-toolkit-pyt"
        f"@sha256:{settings.container_sha256}"
    )
    assert runtime.source_skill_image != runtime.runtime_image
    stale = replace(
        runtime,
        source_skill_image="nvcr.io/nvidia/tao/tao-toolkit:7.0.1-pyt",
    )
    with pytest.raises(
        DINOPreflightContractError,
        match="runtime image mapping",
    ):
        build_dino_preflight_plan(
            voc_integrity=voc_evidence,
            resolved_ptm_inventory=resolved_inventory,
            skill_dir=skill_dir,
            settings=replace(settings, runtime_image_contract=stale),
        )
    latency = plan.commands_for_stage("latency_instrumentation")[0]
    assert latency.metadata["single_gpu_cli"] == {
        "world_size": 1,
        "rank": 0,
        "local_rank": 0,
        "device_id": "cuda:0",
        "gpu_ids": (0,),
        "world_size_from_plan": True,
        "legacy_world_size_8_permitted": False,
    }


def test_plan_freezes_deterministic_subset_and_exact_twenty_line_classmap(plan):
    artifacts = plan.to_dict()["inline_artifacts"]
    classmap = artifacts["voc_label_map"]
    subset = artifacts["voc_inference_subset"]
    assert classmap["line_count"] == 20
    assert len(classmap["content"].splitlines()) == 20
    assert classmap["content"].endswith("\n")
    assert subset["selection_rule"] == "lowest_coco_image_id_first"
    assert [item["image_id"] for item in subset["entries"]] == [21, 22, 23]
    assert subset["sha256"] == plan.voc_integrity.inference_subset_sha256


def test_injected_executor_completes_production_preflight_without_launch(plan):
    executor = FixtureExecutor(plan)
    report = run_dino_local_preflight(plan=plan, executor=executor)

    assert report["completion_state"] == "completed"
    assert report["slurm_ready"] is True
    assert report["readiness"]["all_ptms_smoke_tested"] is True
    assert report["readiness"]["default_one_epoch_passed"] is True
    assert report["readiness"]["standalone_evaluation_passed"] is True
    assert report["readiness"]["checkpoint_save_reload_passed"] is True
    assert report["readiness"]["latency_valid"] is True
    assert report["readiness"]["resume_replay_valid"] is True
    assert "metric_sanity" not in executor.calls
    assert executor.calls.count("eligible_ptm_smoke/dino.fixture.a") == 1
    assert executor.calls.count("eligible_ptm_smoke/dino.fixture.b") == 1
    latency = next(
        item
        for item in report["records"]
        if item["stage"] == "latency_instrumentation"
    )
    assert latency["evidence"]["warmup_iterations"] == 50
    assert latency["evidence"]["timed_iterations"] == 100
    assert latency["evidence"]["rounds"] == 5
    assert latency["evidence"]["quality_gates_passed"] is True


def test_interrupted_outer_preflight_resumes_without_reexecuting_prefix(plan):
    first = FixtureExecutor(plan)
    interrupted = run_dino_local_preflight(
        plan=plan,
        executor=first,
        stop_after_stage="default_model_full_epoch",
    )
    assert interrupted["completion_state"] == "interrupted"
    prefix_calls = list(first.calls)

    resumed_executor = FixtureExecutor(plan)
    resumed = run_dino_local_preflight(
        plan=plan,
        executor=resumed_executor,
        resume_report=interrupted,
    )
    uninterrupted = run_dino_local_preflight(
        plan=plan,
        executor=FixtureExecutor(plan),
    )
    assert resumed == uninterrupted
    assert resumed_executor.calls[0] == "in_epoch_validation"
    assert not set(prefix_calls).intersection(resumed_executor.calls)


def test_freeze_is_create_only_and_resume_requires_identical_bytes(
    plan,
    tmp_path,
):
    path = (tmp_path / "plan.v1.json").resolve()
    assert freeze_dino_preflight_plan(path, plan) == plan.plan_sha256
    original = path.read_bytes()
    assert freeze_dino_preflight_plan(
        path,
        plan,
        resume=True,
    ) == plan.plan_sha256
    assert path.read_bytes() == original
    with pytest.raises(FileExistsError):
        freeze_dino_preflight_plan(path, plan)
    path.write_bytes(original + b" ")
    with pytest.raises(
        DINOPreflightContractError,
        match="byte-identical",
    ):
        freeze_dino_preflight_plan(path, plan, resume=True)


def test_plan_refuses_untyped_or_incomplete_prerequisites(
    voc_evidence,
    resolved_inventory,
    skill_dir,
    settings,
):
    with pytest.raises(
        DINOPreflightContractError,
        match="typed VOC",
    ):
        build_dino_preflight_plan(
            voc_integrity=None,
            resolved_ptm_inventory=resolved_inventory,
            skill_dir=skill_dir,
            settings=settings,
        )
    with pytest.raises(
        DINOPreflightContractError,
        match="live typed",
    ):
        build_dino_preflight_plan(
            voc_integrity=voc_evidence,
            resolved_ptm_inventory=resolved_inventory.to_dict(),
            skill_dir=skill_dir,
            settings=settings,
        )

    incomplete = replace(
        resolved_inventory,
        arms=resolved_inventory.arms[:1],
        inventory_sha256="",
    )
    incomplete = replace(
        incomplete,
        inventory_sha256=canonical_audit_sha256(
            incomplete.stable_dict()
        ),
    )
    with pytest.raises(
        DINOPreflightContractError,
        match="cover every prepared PTM",
    ):
        build_dino_preflight_plan(
            voc_integrity=voc_evidence,
            resolved_ptm_inventory=incomplete,
            skill_dir=skill_dir,
            settings=settings,
        )


def test_plan_refuses_voc_or_ptm_artifact_drift(
    voc_evidence,
    resolved_inventory,
    skill_dir,
    settings,
):
    voc_evidence.train_annotation_path.write_text("{}", encoding="utf-8")
    with pytest.raises(
        DINOPreflightContractError,
        match="changed after integrity",
    ):
        build_dino_preflight_plan(
            voc_integrity=voc_evidence,
            resolved_ptm_inventory=resolved_inventory,
            skill_dir=skill_dir,
            settings=settings,
        )


def test_skill_contract_refuses_missing_explicit_validation_source(
    skill_dir,
):
    path = skill_dir / "references/skill_info.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    del value["actions"]["train"]["inputs"][
        "dataset.val_data_sources[0].json_file"
    ]
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(
        DINOPreflightContractError,
        match="train action contract drifted",
    ):
        load_dino_skill_contract(skill_dir)


def test_latency_executor_must_return_raw_replica_records(plan):
    base = FixtureExecutor(plan)

    def executor(command):
        if command.stage == "latency_instrumentation":
            return DINOPreflightExecutionResult(
                command_id=command.command_id,
                passed=True,
                evidence={"median_ms": 1.0},
            )
        return base(command)

    report = run_dino_local_preflight(plan=plan, executor=executor)
    assert report["completion_state"] == "failed"
    assert report["failure"]["stage"] == "latency_instrumentation"
    assert report["failure"]["code"] == "adapter_exception"


def test_voc_evidence_rejects_non_twenty_category_contract(voc_evidence):
    with pytest.raises(
        DINOPreflightContractError,
        match="IDs 1 through 20",
    ):
        replace(
            voc_evidence,
            categories=voc_evidence.categories[:-1],
            annotation_contract_sha256=canonical_sha256(
                {
                    "dataset_id": voc_evidence.dataset_id,
                    "categories": [
                        {"id": category_id, "name": name}
                        for category_id, name in voc_evidence.categories[:-1]
                    ],
                    "invariants": dict(voc_evidence.invariants),
                }
            ),
        )
