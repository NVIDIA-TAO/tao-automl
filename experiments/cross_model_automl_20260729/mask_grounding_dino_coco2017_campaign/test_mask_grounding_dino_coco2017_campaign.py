from __future__ import annotations

import ast
import copy
import json
import sqlite3
import threading
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tao_automl.ptm_registry import canonical_sha256, load_ptm_registry
from tao_automl.selection import canonical_spec_fingerprint

from . import (
    campaign_contract,
    manifest_generator,
    qualification_campaign,
    run_campaign,
)
from .qualification_gate import (
    QualificationGateError,
    QualificationLoadEvidence,
    audit_qualification,
)


HERE = Path(__file__).resolve().parent
SKILL_DIR = (
    manifest_generator.DEFAULT_SKILLS
    / "skills/models/tao-train-mask-grounding-dino"
)
DATASET_STAGE_MANIFEST = manifest_generator.DEFAULT_STAGE_MANIFEST
if not DATASET_STAGE_MANIFEST.is_file():
    # Dataset staging is an explicit integration dependency. This keeps the
    # isolated campaign worktree testable before the staging commit lands.
    DATASET_STAGE_MANIFEST = Path(
        "/localhome/local-rarunachalam/.tao/worktrees/"
        "tao-automl-segmentation-datasets/experiments/"
        "cross_model_automl_20260729/segmentation_datasets/"
        "dataset_stage_manifest.v1.json"
    )


@lru_cache(maxsize=1)
def _dataset_cached() -> dict:
    return manifest_generator.dataset_record(
        manifest_generator.DEFAULT_DATASET_MANIFEST,
        DATASET_STAGE_MANIFEST,
    )


def _dataset() -> dict:
    return copy.deepcopy(_dataset_cached())


def _runtime(tmp_path: Path) -> dict:
    registry = campaign_contract.mask_grounding_dino_registry_snapshot()
    return {
        "repository": str(Path(__file__).resolve().parents[3]),
        "source_commit": "c" * 40,
        "source_dirty": False,
        "wheel_path": str(manifest_generator.DEFAULT_WHEEL),
        "wheel_sha256": manifest_generator.EXPECTED_WHEEL_SHA256,
        "sdk_dir": str(manifest_generator.DEFAULT_SDK),
        "sdk_commit": manifest_generator.EXPECTED_SDK_COMMIT,
        "skills_repository": str(manifest_generator.DEFAULT_SKILLS),
        "skills_commit": manifest_generator.EXPECTED_SKILLS_COMMIT,
        "skill_dir": str(SKILL_DIR),
        "qualification_evidence_path": str(
            tmp_path / "qualification.json"
        ),
        "runtime_local_eligibility": {
            "schema_version": 2,
            "kind": "direct_full_gpu_qualification_runtime_local_v2",
            "enabled": True,
            "scope": "campaign_local_in_memory_projection",
            "model": "mask_grounding_dino",
            "task": "category_prompted_grounded_instance_segmentation",
            "tao_version": "7.1.0",
            "container_sha256": campaign_contract.FROZEN_SQSH["sha256"],
            "base_registry_version": registry["registry_version"],
            "base_registry_sha256": registry["registry_sha256"],
            "qualification_file_sha256": "1" * 64,
            "qualification_evidence_sha256": "2" * 64,
            "qualification_contract_sha256": "3" * 64,
            "qualification_campaign_sha256": "4" * 64,
            "eligibility_source_commit": "c" * 40,
            "wheel_sha256": manifest_generator.EXPECTED_WHEEL_SHA256,
            "sdk_commit": manifest_generator.EXPECTED_SDK_COMMIT,
            "skills_commit": manifest_generator.EXPECTED_SKILLS_COMMIT,
            "repository_registry_mutation_allowed": False,
            "failed_arm_promotion_allowed": False,
            "unsupported_arm_promotion_allowed": False,
            "agent_override_allowed": False,
        },
        "predecessor_failure_evidence": {
            "path": str(tmp_path / "qualification_v1.json"),
            "sha256": "9" * 64,
            "campaign_id": "mask-grounding-dino-qualification-v1-test",
            "workflow_count": 4,
            "all_terminal_failures_preserved": True,
            "replacement_submitted": False,
        },
        "ptm_stage_manifest_path": str(tmp_path / "ptms.json"),
        "ptm_stage_manifest_sha256": "e" * 64,
        "ptm_stage_content_sha256": "f" * 64,
        "partition": "polar3",
        "account": "edgeai_tao-ptm_image-foundation-model-clip",
        "base_results_dir": (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam"
        ),
        "container_mounts": "/lustre",
        "time_hours": 4.0,
        "timeout_hours": 3.8,
        "max_job_retries": campaign_contract.FROZEN_SLURM_RETRY_CAP,
        "hardware_contract": copy.deepcopy(
            campaign_contract.FROZEN_HARDWARE
        ),
    }


@pytest.fixture
def contract(tmp_path: Path) -> dict:
    value = campaign_contract.build_preregistered_contract(
        campaign_id="mask_grounding_dino-test",
        dataset=_dataset(),
        skill_dir=str(SKILL_DIR),
        runtime=_runtime(tmp_path),
    )
    value.pop("contract_sha256")
    value["launcher_integrity"] = {
        "ddp_strategy_audit_sha256": campaign_contract.sha256_file(
            HERE / "ddp_strategy_audit.v2.json"
        ),
        "campaign_contract_sha256": campaign_contract.sha256_file(
            HERE / "campaign_contract.py"
        ),
        "qualification_gate_sha256": campaign_contract.sha256_file(
            HERE / "qualification_gate.py"
        ),
        "qualification_campaign_sha256": campaign_contract.sha256_file(
            HERE / "qualification_campaign.py"
        ),
        "run_campaign_sha256": campaign_contract.sha256_file(
            HERE / "run_campaign.py"
        ),
        "mask_grounding_dino_latency_worker_sha256": (
            campaign_contract.sha256_file(
                HERE / "mask_grounding_dino_latency_worker.py"
            )
        ),
    }
    value["contract_sha256"] = canonical_sha256(value)
    return campaign_contract.validate_contract(value)


def _workflow(
    checkpoint_id: str,
    *,
    success: bool,
    metric: float = 0.20,
    include_mask_ap: bool = True,
) -> dict:
    record = load_ptm_registry().checkpoint(checkpoint_id)
    if not success:
        value = {
            "checkpoint_id": checkpoint_id,
            "status": "failure",
            "terminal": True,
            "failure_preserved": True,
            "failure_code": "direct_full_run_failed",
            "failure_reason": "frozen test failure",
        }
        value["workflow_sha256"] = canonical_sha256(value)
        return value
    train_metric = (
        {"segm_val_mAP50_95": metric}
        if include_mask_ap
        else {"mIoU": metric}
    )
    eval_metric = (
        {"segm_val_mAP50_95": metric}
        if include_mask_ap
        else {"mIoU": metric}
    )
    value = {
        "checkpoint_id": checkpoint_id,
        "status": "success",
        "terminal": True,
        "failure_preserved": False,
        "source_checkpoint": {
            "path": f"/lustre/ptms/{checkpoint_id}.pth",
            "size_bytes": record["expected_size_bytes"],
            "sha256": "a" * 64,
        },
        "train": {
            "status": "Complete",
            "full_dataset": True,
            "training_epochs": (
                campaign_contract.FROZEN_TRAINING_EPOCHS
            ),
            "validation_interval": 1,
            "validation_record_count": (
                campaign_contract.FROZEN_TRAINING_EPOCHS
            ),
            "nodes": 1,
            "gpus": 8,
            "distributed_strategy_resolution": copy.deepcopy(
                campaign_contract.FROZEN_DDP_STRATEGY_RESOLUTION
            ),
            **train_metric,
            "terminal_checkpoint": {
                "path": f"/lustre/results/{checkpoint_id}.pth",
                "size_bytes": 123,
                "sha256": "b" * 64,
            },
        },
        "evaluation": {
            "status": "Complete",
            "full_validation_split": True,
            "nodes": 1,
            "gpus": 8,
            **eval_metric,
        },
        "agent_intervention_flags": {
            name: False for name in campaign_contract.AGENT_FLAGS
        },
    }
    value["workflow_sha256"] = canonical_sha256(value)
    return value


def _qualification_document(
    success_id: str | None = None,
    *,
    metric: float = 0.20,
    include_mask_ap: bool = True,
) -> dict:
    snapshot = campaign_contract.mask_grounding_dino_registry_snapshot()
    workflows = [
        _workflow(
            record["id"],
            success=record["id"] == success_id,
            metric=metric,
            include_mask_ap=include_mask_ap,
        )
        for record in snapshot["records"]
    ]
    value = {
        "schema_version": 1,
        "campaign_id": "mask_grounding_dino-direct-full-qualification-test",
        "model": "mask_grounding_dino",
        "task": "category_prompted_grounded_instance_segmentation",
        "primary_metric": "segm_val_mAP50_95",
        "VG_overall_iou_accepted_as_mask_ap": False,
        "qualification_contract_sha256": "c" * 64,
        "qualification_campaign_sha256": (
            campaign_contract.sha256_file(
                HERE / "qualification_campaign.py"
            )
        ),
        "ptm_stage_manifest_path": "/tmp/frozen-ptm-stage.json",
        "ptm_stage_manifest_sha256": "d" * 64,
        "registry_sha256": snapshot["registry_sha256"],
        "sqsh_sha256": campaign_contract.FROZEN_SQSH["sha256"],
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "distributed_strategy_resolution": copy.deepcopy(
            campaign_contract.FROZEN_DDP_STRATEGY_RESOLUTION
        ),
        "predecessor_failure_evidence": {
            "path": "/tmp/qualification_v1.json",
            "sha256": "9" * 64,
            "campaign_id": "mask-grounding-dino-qualification-v1-test",
            "workflow_count": 4,
            "all_terminal_failures_preserved": True,
            "replacement_submitted": False,
        },
        "workflows": workflows,
    }
    value["evidence_sha256"] = canonical_sha256(value)
    return value


def _seal_runtime_local_qualification(
    contract: dict,
    tmp_path: Path,
    checkpoint_id: str,
) -> tuple[dict, Path]:
    snapshot = campaign_contract.mask_grounding_dino_registry_snapshot()
    stage_path = tmp_path / "ptms.json"
    stage_path.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "id": item["id"],
                        "path": f"/lustre/ptms/{item['id']}.pth",
                        "size_bytes": item["expected_size_bytes"],
                        "sha256": (
                            "a" * 64
                            if item["id"] == checkpoint_id
                            else (item.get("sha256") or "e" * 64)
                        ),
                    }
                    for item in snapshot["records"]
                ]
            }
        ),
        encoding="utf-8",
    )
    stage_sha = campaign_contract.sha256_file(stage_path)
    document = _qualification_document(checkpoint_id)
    document["ptm_stage_manifest_path"] = str(stage_path)
    document["ptm_stage_manifest_sha256"] = stage_sha
    document["predecessor_failure_evidence"] = copy.deepcopy(
        contract["runtime"]["predecessor_failure_evidence"]
    )
    document["evidence_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in document.items()
            if key != "evidence_sha256"
        }
    )
    qualification_path = tmp_path / "qualification.json"
    qualification_path.write_text(json.dumps(document), encoding="utf-8")

    sealed = copy.deepcopy(contract)
    sealed.pop("contract_sha256")
    sealed["runtime"]["ptm_stage_manifest_path"] = str(stage_path)
    sealed["runtime"]["ptm_stage_manifest_sha256"] = stage_sha
    sealed["qualification_policy"]["ptm_stage_manifest_path"] = str(
        stage_path
    )
    policy = copy.deepcopy(
        sealed["runtime"]["runtime_local_eligibility"]
    )
    policy.update(
        {
            "base_registry_version": snapshot["registry_version"],
            "base_registry_sha256": snapshot["registry_sha256"],
            "qualification_file_sha256": (
                campaign_contract.sha256_file(qualification_path)
            ),
            "qualification_evidence_sha256": document[
                "evidence_sha256"
            ],
            "qualification_contract_sha256": document[
                "qualification_contract_sha256"
            ],
            "qualification_campaign_sha256": document[
                "qualification_campaign_sha256"
            ],
        }
    )
    sealed["runtime"]["runtime_local_eligibility"] = copy.deepcopy(policy)
    sealed["qualification_policy"]["runtime_local_eligibility"] = (
        copy.deepcopy(policy)
    )
    sealed["contract_sha256"] = canonical_sha256(sealed)
    return campaign_contract.validate_contract(sealed), qualification_path


def test_exact_tao_identifier_actions_and_task_correct_metric():
    info = yaml.safe_load(
        (SKILL_DIR / "references/skill_info.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert info["network_arch"] == "mask_grounding_dino"
    assert info["actions"]["train"]["command"] == (
        "mask_grounding_dino train -e {config_path}"
    )
    assert info["actions"]["evaluate"]["command"] == (
        "mask_grounding_dino evaluate -e {config_path}"
    )
    settings = campaign_contract.mode_settings("x", "accuracy")
    assert settings["accuracy_metric"] == "segm_val_mAP50_95"
    assert (
        run_campaign._metric_extractor(
            "mIoU: 0.71", "segm_val_mAP50_95"
        )
        is None
    )
    assert (
        run_campaign._metric_extractor(
            "[segm] val_mAP@50-95: 0.321", "segm_val_mAP50_95"
        )
        == pytest.approx(0.321)
    )


def test_search_parameters_are_packaged_train_parameters():
    evidence = campaign_contract.validate_packaged_train_schema(SKILL_DIR)
    assert tuple(evidence["explicit_search_parameters"]) == (
        "model.num_select",
        "train.optim.lr",
        "train.optim.lr_backbone",
        "train.optim.weight_decay",
    )
    assert campaign_contract.SEARCH_SPACE["model.num_select"] == {
        "type": "integer",
        "values": [50, 100, 200, 300],
    }
    assert evidence["fixed_architecture_depths"] == {
        "model.enc_layers": 6,
        "model.dec_layers": 6,
    }
    assert "model.enc_layers" not in campaign_contract.SEARCH_SPACE
    assert "model.dec_layers" not in campaign_contract.SEARCH_SPACE


def test_complete_coco2017_instance_dataset_is_frozen():
    dataset = _dataset()
    assert dataset["prepared_root"].endswith(
        "/coco2017_instance_panoptic_v1"
    )
    assert dataset["train_image_count"] == 118287
    assert dataset["validation_image_count"] == 5000
    assert dataset["train_instance_annotations"] == 860001
    assert dataset["validation_instance_annotations"] == 36781
    assert dataset["num_classes"] == 80
    assert dataset["file_manifest_entry_count"] == 246593
    assert dataset["manifest_sha256"] == (
        "10566a60498de9998154f44a34445a488c9f030e09f2a7346d20a4a1c55f804e"
    )
    assert dataset["stage_manifest_sha256"] == (
        "437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d"
    )
    assert dataset["remote_read_only"] is True
    assert dataset["remote_writable_entries_after_lock"] == 0
    assert dataset["train_odvg_projected_images"] == 117266
    assert dataset["train_odvg_projected_annotations"] == 860001
    assert dataset["train_odvg_masks_preserved"] == 860001
    assert dataset["train_odvg_jsonl_sha256"] == (
        "d5deb4f5cfe027786fb1ceb52632ad6d3ef027e95e434525ba715d6841fb2921"
    )
    assert dataset["train_odvg_label_map_sha256"] == (
        "02075d96f6bf06d061f9329b4775dc7c3bb5ac140c77bc5c0e465d305c46d6c1"
    )
    assert dataset["contiguous_validation_json_path"] == (
        campaign_contract.FROZEN_CONTIGUOUS_VALIDATION_JSON
    )
    assert dataset["contiguous_validation_json_sha256"] == (
        campaign_contract.FROZEN_CONTIGUOUS_VALIDATION_SHA256
    )
    assert dataset["contiguous_validation_manifest_sha256"] == (
        campaign_contract.FROZEN_CONTIGUOUS_MANIFEST_SHA256
    )
    assert dataset["contiguous_validation_category_ids"] == list(range(80))
    assert dataset["contiguous_validation_remote_read_only"] is True


def test_profile_is_instance_coco_eight_gpu_not_smoke():
    root = _dataset()["prepared_root"]
    profile = campaign_contract.profile_overrides(root)
    model = profile["model"]
    assert model["has_mask"] is True
    assert model["enc_layers"] == 6
    assert model["dec_layers"] == 6
    assert model["num_select"] == 300
    assert model["text_encoder_type"] == (
        campaign_contract.FROZEN_TEXT_ENCODER_ROOT
    )
    dataset = profile["dataset"]
    train_source = dataset["train_data_sources"][0]
    assert train_source["image_dir"] == f"{root}/images/train2017"
    assert train_source["json_file"].endswith(
        "/tao/mask_grounding_dino/train/instances_train2017_odvg.jsonl"
    )
    assert train_source["label_map"].endswith(
        "/tao/mask_grounding_dino/train/"
        "instances_train2017_odvg_labelmap.json"
    )
    for split in ("val_data_sources", "test_data_sources"):
        source = dataset[split]
        assert source["image_dir"] == f"{root}/images/val2017"
        assert source["data_type"] == "OD"
        assert source["json_file"] == (
            campaign_contract.FROZEN_CONTIGUOUS_VALIDATION_JSON
        )
    assert dataset["batch_size"] == 4
    assert dataset["max_labels"] == 80
    assert dataset["eval_class_ids"] == list(range(80))
    assert dataset["has_mask"] is True
    train = profile["train"]
    assert train["num_gpus"] == 8
    assert train["gpu_ids"] == list(range(8))
    assert train["num_nodes"] == 1
    assert train["num_epochs"] == 3
    assert train["checkpoint_interval"] == 1
    assert train["checkpoint_interval_unit"] == "epoch"
    assert train["resume_training_checkpoint_path"] == ""
    assert train["validation_interval"] == 1
    assert train["distributed_strategy"] == "ddp"
    assert train["activation_checkpoint"] is False
    assert train["precision"] == "fp32"
    assert campaign_contract.CHECKPOINT_RESUME_POLICY[
        "post_requeue_missing_checkpoint_behavior"
    ] == "fail_closed"
    evaluate = profile["evaluate"]
    assert evaluate["num_gpus"] == 8
    assert evaluate["gpu_ids"] == list(range(8))


def test_v2_uses_tao_supported_unused_parameter_strategy_resolution(
    contract,
):
    resolution = contract["qualification_policy"][
        "distributed_strategy_resolution"
    ]
    assert resolution == campaign_contract.FROZEN_DDP_STRATEGY_RESOLUTION
    assert resolution == {
        "tao_config_value": "ddp",
        "activation_checkpoint": False,
        "resolved_lightning_strategy": "ddp_find_unused_parameters_true",
        "direct_alias_is_valid_tao_config_value": False,
        "resolution_source": (
            "pinned_mask_grounding_dino_train_launcher_branch"
        ),
    }
    assert contract["qualification_policy"]["version"] == 2


def test_v2_strategy_audit_preserves_exact_v1_failure_evidence():
    audit_path = HERE / "ddp_strategy_audit.v2.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    predecessor = Path(
        audit["v1_preserved"]["completion_path"]
    )
    assert predecessor.is_file()
    assert campaign_contract.sha256_file(predecessor) == (
        audit["v1_preserved"]["completion_sha256"]
    )
    assert audit["v1_preserved"]["slurm_job_ids"] == [
        "31243535",
        "31243536",
        "31243537",
        "31243538",
    ]
    assert audit["v2_change_scope"] == {
        "effective_distributed_strategy_changed": True,
        "ptm_changed": False,
        "dataset_changed": False,
        "search_space_changed": False,
        "training_epochs_changed": False,
        "objective_changed": False,
        "seed_changed": False,
        "candidate_injected": False,
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
    }


def test_strategy_or_predecessor_mutation_is_fail_closed(contract):
    changed = copy.deepcopy(contract)
    changed.pop("contract_sha256")
    changed["qualification_policy"]["distributed_strategy_resolution"][
        "resolved_lightning_strategy"
    ] = "ddp"
    changed["contract_sha256"] = canonical_sha256(changed)
    with pytest.raises(
        campaign_contract.CampaignContractError,
        match="campaign execution policy changed",
    ):
        campaign_contract.validate_contract(changed)

    changed = copy.deepcopy(contract)
    changed.pop("contract_sha256")
    changed["runtime"]["predecessor_failure_evidence"][
        "replacement_submitted"
    ] = True
    changed["qualification_policy"]["predecessor_failure_evidence"][
        "replacement_submitted"
    ] = True
    changed["contract_sha256"] = canonical_sha256(changed)
    with pytest.raises(
        campaign_contract.CampaignContractError,
        match="preserved v1 qualification evidence contract changed",
    ):
        campaign_contract.validate_contract(changed)


def test_mask_ap_sanity_is_separate_from_product_selection(contract):
    assert contract["task"] == "category_prompted_grounded_instance_segmentation"
    metric = contract["metric_contract"]
    assert metric["required"] == "segm_val_mAP50_95"
    assert metric["VG_overall_iou_is_not_an_alias"] is True
    assert metric["known_repository_state"] == (
        "statically_implemented_runtime_qualification_required"
    )
    gate = contract["validation_sanity_gate"]
    assert gate["metric"] == "segm_val_mAP50_95"
    assert gate["minimum"] == 0.05
    assert gate["role"] == (
        "experiment_correctness_gate_not_product_selection"
    )


def test_official_ptm_registry_is_exact_and_hierarchical():
    snapshot = campaign_contract.mask_grounding_dino_registry_snapshot()
    expected = {
        "mask_grounding_dino.commercial.swin_tiny.trainable.v2.1",
        "mask_grounding_dino.commercial.swin_tiny.trainable.v2.0",
        "mask_grounding_dino.commercial.swin_tiny.trainable.v1.0",
        "mask_grounding_dino.research.swin_tiny.trainable.v2.0",
    }
    assert snapshot["record_count"] == 4
    assert {item["id"] for item in snapshot["records"]} == expected
    for record in snapshot["records"]:
        assert record["source"]["official"] is True
        assert record["checkpoint_target"] == "train.pretrained_model_path"
        assert record["compatible_tao_versions"] == ["==7.1.0"]
        assert record["default_spec_overrides"]["model"]["enc_layers"] == 6
        assert record["default_spec_overrides"]["model"]["dec_layers"] == 6
        assert record["checkpoint_spec_file"]["source"] == "repository"
        assert record["checkpoint_spec_file"]["path"].endswith(".yaml")
        assert len(record["checkpoint_spec_file"]["sha256"]) == 64
    assert snapshot["supported_ids"] == []
    assert set(snapshot["unverified_ids"]) == expected


def test_mode_acquisitions_and_constraints_are_independent(contract):
    modes = {
        item["mode"]: item for item in contract["modes"]
    }
    assert modes["accuracy"]["objective"]["acquisition"] == (
        "expected_improvement"
    )
    assert "latency_accuracy_retention" not in modes["accuracy"]["settings"]
    assert modes["latency"]["objective"]["acquisition"] == (
        "constrained_expected_improvement"
    )
    assert modes["latency"]["settings"][
        "latency_accuracy_retention"
    ] == {
        "type": "relative",
        "retained_fraction": 0.90,
        "reference": "accuracy_winner",
    }
    assert modes["multi_objective"]["objective"]["acquisition"] == (
        "parego_expected_improvement"
    )
    assert "latency_accuracy_retention" not in (
        modes["multi_objective"]["settings"]
    )
    assert (
        modes["multi_objective"]["settings"][
            "multi_objective_min_accuracy"
        ]
        is None
    )
    assert all(
        item["observation_sharing"] is False
        and item["initial_observation_ids"] == []
        for item in modes.values()
    )


def test_contract_is_pinned_sqsh_eight_gpu_and_zero_local_runs(contract):
    assert contract["sqsh"] == campaign_contract.FROZEN_SQSH
    assert contract["execution"]["container_mode"] == "pinned_sqsh"
    assert contract["execution"]["nodes_per_child"] == 1
    assert contract["execution"]["gpus_per_child"] == 8
    assert contract["execution"]["cpu_runs"] == 0
    assert contract["execution"]["smoke_runs"] == 0
    assert contract["execution"]["local_model_runs"] == 0
    assert contract["qualification_policy"]["cpu_model_runs"] == 0
    assert contract["qualification_policy"]["smoke_model_runs"] == 0
    assert contract["qualification_policy"]["mini_step_runs"] == 0
    assert all(
        value is False
        for value in contract["agent_intervention_flags"].values()
    )
    assert all(
        value is False
        for value in contract["selection_isolation_flags"].values()
    )


def test_latency_protocol_is_4000_real_coco_validation_samples(contract):
    protocol = contract["latency_protocol"]
    assert protocol["warmup_iterations"] == 50
    assert protocol["repeated_rounds"] == 5
    assert protocol["timed_iterations"] == 100
    assert protocol["expected_replicas"] == 8
    assert protocol["raw_samples_per_candidate"] == 4000
    assert "instance_mask_serialization" in protocol["excluded_scope"]
    assert "gpu_mask_postprocess" in protocol["timed_scope"]
    source = (HERE / "mask_grounding_dino_latency_worker.py").read_text(
        encoding="utf-8"
    )
    assert "ODVGDataModule" in source
    assert "MaskGDINOPlModel" in source
    assert "box_processors" in source
    assert "outputs = model(" in source
    assert "COCODataset" not in source
    assert "torch.randn" not in source
    assert "torch.rand(" not in source


def test_model_imports_live_only_below_latency_worker_main():
    tree = ast.parse(
        (HERE / "mask_grounding_dino_latency_worker.py").read_text(
            encoding="utf-8"
        )
    )
    module_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        not (
            isinstance(node, ast.Import)
            and any(alias.name == "torch" for alias in node.names)
        )
        for node in module_imports
    )
    assert all(
        not (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("nvidia_tao_pytorch")
        )
        for node in module_imports
    )


def test_VG_overall_iou_cannot_qualify_category_prompted_grounded_instance_segmentation(tmp_path: Path):
    checkpoint_id = campaign_contract.mask_grounding_dino_registry_snapshot()[
        "records"
    ][0]["id"]
    document = _qualification_document(
        checkpoint_id,
        include_mask_ap=False,
    )
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    decision = audit_qualification(path)
    assert checkpoint_id not in decision.checkpoint_ids
    assert any(
        item["checkpoint_id"] == checkpoint_id
        and item["code"] == "invalid_success_evidence"
        and "segm_val_mAP50_95" in item["reason"]
        for item in decision.blockers
    )


def test_low_finite_mask_ap_does_not_pass_qualification(tmp_path: Path):
    checkpoint_id = campaign_contract.mask_grounding_dino_registry_snapshot()[
        "records"
    ][0]["id"]
    document = _qualification_document(checkpoint_id, metric=0.049)
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    decision = audit_qualification(path)
    assert any(
        item["checkpoint_id"] == checkpoint_id
        and item["code"] == "invalid_success_evidence"
        and "0.05 COCO mask AP50-95" in item["reason"]
        for item in decision.blockers
    )


def test_unverified_full_run_success_cannot_bypass_registry(tmp_path: Path):
    checkpoint_id = campaign_contract.mask_grounding_dino_registry_snapshot()[
        "records"
    ][0]["id"]
    path = tmp_path / "qualification.json"
    path.write_text(
        json.dumps(_qualification_document(checkpoint_id)),
        encoding="utf-8",
    )
    decision = audit_qualification(path)
    assert checkpoint_id not in decision.checkpoint_ids
    assert any(
        item["checkpoint_id"] == checkpoint_id
        and item["code"] == "registry_not_supported"
        for item in decision.blockers
    )
    with pytest.raises(QualificationGateError):
        QualificationLoadEvidence(decision)


def test_qualification_can_precede_registry_promotion_without_bypass(
    tmp_path: Path,
):
    checkpoint_id = campaign_contract.mask_grounding_dino_registry_snapshot()[
        "records"
    ][0]["id"]
    document = _qualification_document(checkpoint_id)
    # Qualification evidence is naturally produced against the pre-promotion
    # registry. Its immutable checkpoint/workflow identity remains usable
    # after a separate reviewed promotion, but current status is still
    # enforced and therefore blocks in this unverified repository.
    document["registry_sha256"] = "1" * 64
    document["evidence_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in document.items()
            if key != "evidence_sha256"
        }
    )
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    decision = audit_qualification(path)
    assert any(
        item["checkpoint_id"] == checkpoint_id
        and item["code"] == "registry_not_supported"
        for item in decision.blockers
    )


def test_sealed_runtime_local_eligibility_projects_only_exact_success(
    contract,
    tmp_path: Path,
):
    checkpoint_id = (
        "mask_grounding_dino.commercial.swin_tiny.trainable.v1.0"
    )
    sealed, qualification_path = _seal_runtime_local_qualification(
        contract,
        tmp_path,
        checkpoint_id,
    )
    base = load_ptm_registry()
    assert base.checkpoint(checkpoint_id)["status"] == "unverified"

    decision = audit_qualification(
        qualification_path,
        expected_contract=sealed,
    )

    assert decision.runtime_ready is True
    assert decision.checkpoint_ids == (checkpoint_id,)
    assert len(decision.exclusions) == 3
    assert decision.blockers == ()
    assert decision.runtime_registry.checkpoint(checkpoint_id)["status"] == (
        "supported"
    )
    assert load_ptm_registry().checkpoint(checkpoint_id)["status"] == (
        "unverified"
    )
    eligibility = decision.runtime_eligibility
    assert eligibility["schema_version"] == 2
    assert eligibility["scope"] == "campaign_local_in_memory_projection"
    assert eligibility["repository_registry_mutated"] is False
    assert eligibility["failed_arms_preserved"] is True
    assert eligibility["qualified_checkpoint_ids"] == [checkpoint_id]
    transformation = eligibility["transformations"][0]
    assert transformation["checkpoint_id"] == checkpoint_id
    assert transformation["action"] == "qualify_exact_unverified_identity"
    assert transformation["base_status"] == "unverified"
    assert transformation["projected_status"] == "supported"
    assert transformation["base_record_sha256"] == eligibility[
        "base_record_sha256_by_checkpoint_id"
    ][checkpoint_id]
    assert set(eligibility["unchanged_checkpoint_ids"]) == (
        set(eligibility["base_record_sha256_by_checkpoint_id"])
        - {checkpoint_id}
    )
    assert decision.runtime_registry.document_sha256 == eligibility[
        "projected_registry_sha256"
    ]


def test_runtime_local_eligibility_fails_closed_on_evidence_hash_change(
    contract,
    tmp_path: Path,
):
    checkpoint_id = (
        "mask_grounding_dino.commercial.swin_tiny.trainable.v1.0"
    )
    sealed, qualification_path = _seal_runtime_local_qualification(
        contract,
        tmp_path,
        checkpoint_id,
    )
    changed = copy.deepcopy(sealed)
    changed.pop("contract_sha256")
    for location in (
        changed["runtime"],
        changed["qualification_policy"],
    ):
        location["runtime_local_eligibility"][
            "qualification_file_sha256"
        ] = "f" * 64
    changed["contract_sha256"] = canonical_sha256(changed)
    changed = campaign_contract.validate_contract(changed)

    with pytest.raises(
        QualificationGateError,
        match="exact base registry and qualification evidence",
    ):
        audit_qualification(
            qualification_path,
            expected_contract=changed,
        )


def test_direct_full_qualification_plan_is_plan_only(contract):
    plan = qualification_campaign.qualification_plan(contract)
    assert plan["official_checkpoint_ids"] == [
        item["id"] for item in contract["ptm_inventory"]["records"]
    ]
    assert plan["workflow_count"] == 4
    assert plan["concurrent_workflow_count"] == 4
    assert plan["independent_sdk_state_per_workflow"] is True
    assert plan["training_epochs"] == 3
    assert plan["nodes_per_job"] == 1
    assert plan["gpus_per_job"] == 8
    assert plan["scheduler_client_constructed"] is False
    assert plan["jobs_submitted"] == 0
    assert plan["cpu_model_runs"] == 0
    assert plan["smoke_model_runs"] == 0
    assert plan["mini_step_runs"] == 0
    assert plan["replacement_workflows_allowed"] is False


def test_v3_qualification_records_exact_four_arm_recovery(contract):
    changed = copy.deepcopy(contract)
    changed["qualification_policy"].update(
        {
            "version": 3,
            "replacement_scope": "all_four_v2_timeout_loops",
            "checkpoint_resume_policy": copy.deepcopy(
                campaign_contract.CHECKPOINT_RESUME_POLICY
            ),
        }
    )
    plan = qualification_campaign.qualification_plan(changed)
    assert plan["replacement_workflows_allowed"] is True
    assert plan["replacement_scope"] == "all_four_v2_timeout_loops"
    workflows = [
        {"checkpoint_id": item["id"], "status": "success"}
        for item in changed["ptm_inventory"]["records"]
    ]
    completion = qualification_campaign.build_completion(changed, workflows)
    assert completion["replacement_workflows_submitted"] is True
    assert completion["replacement_workflow_count"] == 4


def test_direct_full_qualifications_run_all_four_arms_concurrently(
    contract,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint_ids = sorted(
        record["id"] for record in contract["ptm_inventory"]["records"]
    )
    staged = {
        checkpoint_id: {
            "path": f"/lustre/ptms/{checkpoint_id}.pth",
            "size_bytes": 1,
            "sha256": "a" * 64,
        }
        for checkpoint_id in checkpoint_ids
    }
    barrier = threading.Barrier(len(checkpoint_ids))
    lock = threading.Lock()
    state_paths: dict[str, Path] = {}

    def sdk_factory(checkpoint_id: str, state_file: Path):
        with lock:
            state_paths[checkpoint_id] = state_file
        return object()

    def fake_run_one(
        supplied_contract,
        sdk,
        checkpoint_id,
        source,
        runtime_root,
    ):
        assert supplied_contract is contract
        assert sdk is not None
        assert source is staged[checkpoint_id]
        assert runtime_root == tmp_path
        barrier.wait(timeout=5)
        return {
            "checkpoint_id": checkpoint_id,
            "status": "success",
        }

    monkeypatch.setattr(
        qualification_campaign,
        "_run_one",
        fake_run_one,
    )
    workflows = qualification_campaign._run_qualifications_concurrently(
        contract,
        staged,
        tmp_path,
        sdk_factory,
    )
    assert [item["checkpoint_id"] for item in workflows] == checkpoint_ids
    assert set(state_paths) == set(checkpoint_ids)
    assert len(set(state_paths.values())) == len(checkpoint_ids)
    assert all(
        path.name == "slurm_state.json"
        and path.parent.name == checkpoint_id.replace("/", "_")
        for checkpoint_id, path in state_paths.items()
    )


def test_concurrent_qualification_sdk_failure_is_terminal(
    contract,
    tmp_path: Path,
):
    checkpoint_id = contract["ptm_inventory"]["records"][0]["id"]

    def fail_sdk(checkpoint_id: str, state_file: Path):
        del checkpoint_id, state_file
        raise RuntimeError("frozen SDK construction failure")

    workflows = qualification_campaign._run_qualifications_concurrently(
        contract,
        {
            checkpoint_id: {
                "path": f"/lustre/ptms/{checkpoint_id}.pth",
                "size_bytes": 1,
                "sha256": "a" * 64,
            }
        },
        tmp_path,
        fail_sdk,
    )
    assert len(workflows) == 1
    failure = workflows[0]
    assert failure["checkpoint_id"] == checkpoint_id
    assert failure["status"] == "failure"
    assert failure["terminal"] is True
    assert failure["failure_preserved"] is True
    assert failure["replacement_submitted"] is False
    assert failure["failure_code"] == "direct_full_workflow_exception"


def test_direct_qualification_spec_precedence_preserves_coco_profile(
    contract,
):
    checkpoint_id = contract["ptm_inventory"]["records"][0]["id"]
    train, evaluate = qualification_campaign._qualification_specs(
        contract,
        checkpoint_id,
        "/lustre/ptms/mask_grounding_dino.pth",
    )
    for specification in (train, evaluate):
        assert specification["model"]["enc_layers"] == 6
        assert specification["model"]["dec_layers"] == 6
        assert specification["model"]["has_mask"] is True
        assert specification["model"]["text_encoder_type"] == (
            campaign_contract.FROZEN_TEXT_ENCODER_ROOT
        )
        assert specification["dataset"]["val_data_sources"]["data_type"] == (
            "OD"
        )
        assert specification["dataset"]["val_data_sources"]["json_file"] == (
            campaign_contract.FROZEN_CONTIGUOUS_VALIDATION_JSON
        )
        assert specification["dataset"]["eval_class_ids"] == list(range(80))
    assert train["train"]["pretrained_model_path"] == (
        "/lustre/ptms/mask_grounding_dino.pth"
    )


def test_ptm_stage_is_exact_content_addressed_inventory(
    contract,
    tmp_path: Path,
):
    records = contract["ptm_inventory"]["records"]
    stage = {
        "schema_version": 1,
        "model": "mask_grounding_dino",
        "registry_sha256": contract["ptm_inventory"]["registry_sha256"],
        "stage_complete": True,
        "remote_read_only": True,
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "checkpoints": [
            {
                "id": record["id"],
                "path": f"/lustre/ptms/{record['id']}.pth",
                "size_bytes": record["expected_size_bytes"],
                "sha256": record["sha256"] or "d" * 64,
                "immutable_source_identity": record["source"][
                    "immutable_identity"
                ],
                "remote_read_only": True,
            }
            for record in records
        ],
    }
    stage["manifest_sha256"] = canonical_sha256(stage)
    path = tmp_path / "ptm_stage.json"
    path.write_text(json.dumps(stage), encoding="utf-8")
    frozen_contract = copy.deepcopy(contract)
    frozen_contract["runtime"]["ptm_stage_manifest_sha256"] = (
        campaign_contract.sha256_file(path)
    )
    loaded = qualification_campaign.load_ptm_stage(
        path, frozen_contract, verify_remote=False
    )
    assert set(loaded) == {record["id"] for record in records}
    for record in records:
        assert loaded[record["id"]]["sha256"] == (
            record["sha256"] or "d" * 64
        )
    sealed = manifest_generator.ptm_stage_record(path)
    assert sealed["sha256"] == campaign_contract.sha256_file(path)
    assert sealed["manifest_sha256"] == stage["manifest_sha256"]
    assert sealed["checkpoint_ids"] == sorted(
        record["id"] for record in records
    )

    changed = copy.deepcopy(stage)
    changed["checkpoints"][0]["size_bytes"] -= 1
    changed["manifest_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in changed.items()
            if key != "manifest_sha256"
        }
    )
    path.write_text(json.dumps(changed), encoding="utf-8")
    frozen_contract["runtime"]["ptm_stage_manifest_sha256"] = (
        campaign_contract.sha256_file(path)
    )
    with pytest.raises(run_campaign.CampaignExecutionError):
        qualification_campaign.load_ptm_stage(
            path, frozen_contract, verify_remote=False
        )


def test_qualification_missing_mask_ap_is_terminal_and_not_replaced():
    failure = qualification_campaign._failure_workflow(
        "mask_grounding_dino.commercial.swin_tiny.trainable.v2.1",
        "task-correct segm_val_mAP50_95 missing; observed mIoU only",
        code="task_correct_metric_missing",
        diagnostics={"mIoU": [0.7]},
    )
    assert failure["status"] == "failure"
    assert failure["terminal"] is True
    assert failure["failure_preserved"] is True
    assert failure["replacement_submitted"] is False
    payload = copy.deepcopy(failure)
    supplied = payload.pop("workflow_sha256")
    assert supplied == canonical_sha256(payload)


def test_terminal_ptm_failure_is_preserved_as_exclusion(tmp_path: Path):
    path = tmp_path / "qualification.json"
    path.write_text(
        json.dumps(_qualification_document()),
        encoding="utf-8",
    )
    decision = audit_qualification(path)
    assert len(decision.exclusions) == 4
    assert all(
        item["code"] == "direct_full_run_failed"
        for item in decision.exclusions
    )
    assert any(
        item["code"] == "no_runtime_qualified_ptm"
        for item in decision.blockers
    )


def _write_gate_cell(
    root: Path,
    contract_sha256: str,
    mode: str,
    passed: bool,
) -> None:
    run_campaign.atomic_json(
        root / "first_candidate_gate" / f"{mode}.json",
        {
            "schema_version": 1,
            "contract_sha256": contract_sha256,
            "mode": mode,
            "candidate_id": f"{mode}_rec_0",
            "passed": passed,
        },
    )


def test_first_candidate_gate_releases_automatically_only_after_all_pass(
    tmp_path: Path,
):
    processes = {
        mode: SimpleNamespace(is_alive=lambda: True)
        for mode in campaign_contract.MODES
    }
    for mode in campaign_contract.MODES[:2]:
        _write_gate_cell(tmp_path, "a" * 64, mode, True)
    assert (
        run_campaign._release_first_candidate_gate(
            tmp_path, processes, "a" * 64
        )
        is None
    )
    _write_gate_cell(
        tmp_path, "a" * 64, campaign_contract.MODES[2], True
    )
    release = run_campaign._release_first_candidate_gate(
        tmp_path, processes, "a" * 64
    )
    assert release["release_remaining_budget"] is True
    assert release["generated_automatically"] is True


def test_first_candidate_gate_fails_closed_on_any_failed_mode(
    tmp_path: Path,
):
    processes = {
        mode: SimpleNamespace(is_alive=lambda: True)
        for mode in campaign_contract.MODES
    }
    for index, mode in enumerate(campaign_contract.MODES):
        _write_gate_cell(tmp_path, "b" * 64, mode, index != 1)
    release = run_campaign._release_first_candidate_gate(
        tmp_path, processes, "b" * 64
    )
    assert release["release_remaining_budget"] is False


def test_successor_gate_preserves_predecessor_release(tmp_path: Path):
    predecessor_sha = "b" * 64
    successor_sha = "c" * 64
    predecessor_release = {
        "schema_version": 1,
        "contract_sha256": predecessor_sha,
        "release_remaining_budget": False,
        "modes": list(campaign_contract.MODES),
        "reason": "preserved predecessor failure",
    }
    run_campaign.atomic_json(
        tmp_path / "first_candidate_gate" / "release.json",
        predecessor_release,
    )
    successor_dir = run_campaign._first_candidate_gate_dir(
        tmp_path, successor_sha
    )
    assert successor_dir == (
        tmp_path / f"first_candidate_gate_{successor_sha[:16]}"
    )
    assert json.loads(
        (tmp_path / "first_candidate_gate" / "release.json").read_text()
    ) == predecessor_release


def test_remaining_budget_gate_blocks_second_recommendation_after_rejection(
    tmp_path: Path,
):
    contract_sha256 = "d" * 64
    gate = run_campaign._first_candidate_gate_dir(
        tmp_path, contract_sha256
    )
    run_campaign.atomic_json(
        gate / "release.json",
        {
            "schema_version": 1,
            "contract_sha256": contract_sha256,
            "release_remaining_budget": False,
            "modes": list(campaign_contract.MODES),
            "reason": "one or more real first candidates failed",
        },
    )
    run_campaign._require_remaining_budget_release(
        runtime_root=tmp_path,
        contract_sha256=contract_sha256,
        recommendation_id=0,
    )
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="blocked by first-candidate gate",
    ):
        run_campaign._require_remaining_budget_release(
            runtime_root=tmp_path,
            contract_sha256=contract_sha256,
            recommendation_id=1,
        )


def test_remaining_budget_gate_allows_second_recommendation_only_after_release(
    tmp_path: Path,
):
    contract_sha256 = "e" * 64
    gate = run_campaign._first_candidate_gate_dir(
        tmp_path, contract_sha256
    )
    run_campaign.atomic_json(
        gate / "release.json",
        {
            "schema_version": 1,
            "contract_sha256": contract_sha256,
            "release_remaining_budget": True,
            "modes": list(campaign_contract.MODES),
            "reason": "all real first candidates passed",
        },
    )
    run_campaign._require_remaining_budget_release(
        runtime_root=tmp_path,
        contract_sha256=contract_sha256,
        recommendation_id="1",
    )


def test_latency_worker_receives_frozen_offline_text_environment(contract):
    assert run_campaign._offline_text_environment(contract) == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def _first_candidate_reuse_fixture(
    tmp_path: Path,
    contract: dict,
) -> dict:
    predecessor = copy.deepcopy(contract)
    predecessor["campaign_id"] = (
        "mask_grounding_dino-coco2017-objective-aware-three-mode-v5-20260803"
    )
    predecessor["runtime"]["source_commit"] = "a" * 40
    predecessor.pop("contract_sha256", None)
    predecessor["contract_sha256"] = canonical_sha256(predecessor)
    contract_path = tmp_path / "campaign.v7.json"
    contract_path.write_text(json.dumps(predecessor), encoding="utf-8")
    root = tmp_path / "runtime"
    for mode in campaign_contract.MODES:
        mode_root = root / mode
        mode_root.mkdir(parents=True)
        job_id = f"{mode}-train-job"
        results_dir = f"lustre:///lustre/results/{job_id}"
        specs = {"train": {"optim": {"lr": 0.0002}}}
        candidate = {
            "candidate_id": f"{mode}_rec_0",
            "rec_id": "0",
            "status": "terminal_failure",
            "automl_status": "failure",
            "failure_reason": "required_eval_fn_failed:latency job failed",
            "checkpoint_id": "checkpoint-arm",
            "specs": specs,
            "candidate_fingerprint": canonical_spec_fingerprint(specs),
            "recommendation_audit": {"audit_sha256": "b" * 64},
            "train_job_id": job_id,
            "terminal_checkpoint": {
                "path": (
                    f"/lustre/results/{job_id}/results_dir/train/"
                    "model_epoch_002_step_00100.pth"
                ),
                "filename": "model_epoch_002_step_00100.pth",
                "size_bytes": 10,
                "sha256": "c" * 64,
            },
        }
        evidence_path = mode_root / "candidate_evidence.json"
        evidence_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "contract_sha256": predecessor["contract_sha256"],
                    "mode": mode,
                    "candidates": {
                        f"{mode}_rec_0": candidate,
                        f"{mode}_rec_1": {
                            "status": "terminal_failure"
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        state_db = mode_root / "slurm_state.db"
        with sqlite3.connect(state_db) as connection:
            connection.execute(
                "CREATE TABLE jobs (job_id TEXT, status TEXT, results_dir TEXT)"
            )
            connection.execute(
                "INSERT INTO jobs VALUES (?, ?, ?)",
                (job_id, "Complete", results_dir),
            )
    return manifest_generator.first_candidate_training_reuse_record(
        contract_path, root
    )


def test_first_candidate_reuse_is_exact_completed_training_only(
    tmp_path: Path,
    contract,
):
    reuse = _first_candidate_reuse_fixture(tmp_path, contract)
    assert set(reuse["modes"]) == set(campaign_contract.MODES)
    assert reuse["training_relaunch_allowed"] is False
    assert reuse["objective_reuse_allowed"] is False
    assert reuse["evaluation_reuse_allowed"] is False
    assert reuse["latency_reuse_allowed"] is False
    assert reuse["new_training_jobs_submitted"] == 0
    for mode, record in reuse["modes"].items():
        assert record["candidate_id"] == f"{mode}_rec_0"
        assert record["discarded_non_observations"] == 1


def test_contract_accepts_fresh_first_candidate_training_reuse(
    tmp_path: Path,
    contract,
):
    value = copy.deepcopy(contract)
    value["runtime"]["first_candidate_training_reuse"] = (
        _first_candidate_reuse_fixture(tmp_path, contract)
    )
    value.pop("contract_sha256")
    value["contract_sha256"] = canonical_sha256(value)
    assert campaign_contract.validate_contract(value) == value


def test_first_candidate_training_reuse_sdk_is_one_shot():
    class FakeSDK:
        def __init__(self, jobs=None):
            self.jobs = jobs or {}
            self.created = []

        def get_job(self, job_id):
            return self.jobs.get(job_id)

        def create_job(self, *args, **kwargs):
            self.created.append((args, kwargs))
            return SimpleNamespace(id="new-job", backend_job_id="new-backend")

    specs = {"model": {"num_select": 100}}
    fingerprint = canonical_spec_fingerprint(specs)
    record = {
        "candidate_fingerprint": fingerprint,
        "specs_sha256": canonical_sha256(specs),
        "checkpoint_id": "checkpoint-arm",
        "source_train_job_id": "source-job",
        "source_results_dir": "lustre:///lustre/results/source-job",
        "source_state_db_sha256": "a" * 64,
        "source_candidate_evidence_sha256": "b" * 64,
        "terminal_checkpoint": {
            "path": "/lustre/results/source-job/model_epoch_002_step_1.pth",
            "sha256": "c" * 64,
            "size_bytes": 1,
        },
    }
    source = FakeSDK(
        {
            "source-job": {
                "status": "Complete",
                "results_dir": record["source_results_dir"],
                "backend_job_id": "source-backend",
            }
        }
    )
    delegate = FakeSDK()
    sdk = run_campaign.FirstCandidateTrainingReuseSDK(
        delegate, source, record
    )
    recommendation = SimpleNamespace(
        id=0,
        specs=specs,
        recommendation_audit={
            "acquisition": {
                "proposal": {
                    "ptm": {"arm_id": "checkpoint-arm"}
                }
            }
        },
    )
    sdk.arm_first_candidate_training_reuse(recommendation)
    reused = sdk.create_job(image="sqsh", command="train")
    fresh = sdk.create_job(image="sqsh", command="train")
    assert reused.id == "source-job"
    assert fresh.id == "new-job"
    assert len(delegate.created) == 1
    assert sdk.training_reuse_evidence("source-job")[
        "new_training_job_submitted"
    ] is False


def test_launch_plan_is_automatic_and_does_not_launch(contract):
    plan = run_campaign.launch_plan(
        contract,
        ready=False,
        blockers=[{"code": "ptm_qualification_not_ready"}],
    )
    assert plan["launch_authorized"] is False
    assert plan["automatic_trigger"] is True
    assert plan["cpu_or_smoke_model_jobs"] == 0
    assert plan["first_candidate_gate"]["automatic_release"] is True
    assert plan["first_candidate_gate"][
        "remaining_candidates_per_mode"
    ] == 23
    assert plan["resources_per_child"]["gpus"] == 8


def test_validation_descriptor_is_bound_to_coco_records(
    contract,
    monkeypatch: pytest.MonkeyPatch,
):
    records = [
        {
            "id": index + 1,
            "file_name": f"{index + 1:012d}.jpg",
            "width": 640,
            "height": 480,
            "size_bytes": 100 + index,
            "sha256": f"{index + 1:064x}",
        }
        for index in range(16)
    ]
    categories = [f"class_{index:02d}" for index in range(80)]
    observed_command = ""

    def fake_remote(command: str) -> str:
        nonlocal observed_command
        observed_command = command
        return json.dumps(
            {"images": records, "categories": categories}
        )

    monkeypatch.setattr(run_campaign, "remote_output", fake_remote)
    descriptor = run_campaign.validation_input_descriptor(contract)
    assert "instances_val2017_remapped.json" in observed_command
    assert descriptor["images"] == records
    assert descriptor["category_prompts"] == categories
    assert descriptor["source_annotation_sha256"] == (
        campaign_contract.FROZEN_CONTIGUOUS_VALIDATION_SHA256
    )
    assert descriptor["preloaded_batches"] == 16
    assert "image_size" not in descriptor


def test_evaluation_spec_is_full_coco_instance_and_eight_gpu(contract):
    specification = run_campaign.evaluation_spec(
        contract,
        {"model": {"num_select": 100}},
        "/lustre/checkpoints/final.pth",
    )
    assert specification["model"]["num_select"] == 100
    assert specification["model"]["enc_layers"] == 6
    assert specification["model"]["dec_layers"] == 6
    assert specification["model"]["has_mask"] is True
    assert specification["dataset"]["test_data_sources"]["data_type"] == "OD"
    assert specification["dataset"]["test_data_sources"]["json_file"] == (
        campaign_contract.FROZEN_CONTIGUOUS_VALIDATION_JSON
    )
    assert specification["dataset"]["batch_size"] == 1
    assert specification["evaluate"]["num_gpus"] == 8
    assert specification["evaluate"]["gpu_ids"] == list(range(8))
    assert specification["evaluate"]["checkpoint"] == (
        "/lustre/checkpoints/final.pth"
    )


def test_archive_order_cannot_enter_independent_campaign_jobs(contract):
    modes = contract["modes"]
    assert [item["mode"] for item in modes] == [
        "accuracy",
        "latency",
        "multi_objective",
    ]
    assert len(
        {item["observation_namespace"] for item in modes}
    ) == 3
    assert all(item["initial_observation_ids"] == [] for item in modes)
    assert contract["execution"]["shared_archive"] is False


def test_contract_integrity_rejects_policy_mutation(contract):
    changed = copy.deepcopy(contract)
    changed["execution"]["gpus_per_child"] = 1
    with pytest.raises(campaign_contract.CampaignContractError):
        campaign_contract.validate_contract(changed)

    changed = copy.deepcopy(contract)
    changed["metric_contract"]["VG_overall_iou_is_not_an_alias"] = False
    with pytest.raises(campaign_contract.CampaignContractError):
        campaign_contract.validate_contract(changed)

    changed = copy.deepcopy(contract)
    changed["modes"][2]["objective"]["acquisition"] = (
        "expected_improvement"
    )
    changed.pop("contract_sha256")
    changed["contract_sha256"] = canonical_sha256(changed)
    with pytest.raises(campaign_contract.CampaignContractError):
        campaign_contract.validate_contract(changed)
