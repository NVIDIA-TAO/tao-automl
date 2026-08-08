from __future__ import annotations

import ast
import copy
import json
import shlex
import subprocess
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tao_automl.ptm_registry import canonical_sha256, load_ptm_registry

from . import (
    campaign_contract,
    checkpoint_resume,
    manifest_generator,
    qualification_campaign,
    run_campaign,
    runtime_overlay,
)
from .qualification_gate import (
    QualificationGateError,
    QualificationLoadEvidence,
    audit_qualification,
)


HERE = Path(__file__).resolve().parent
SKILL_DIR = (
    manifest_generator.DEFAULT_SKILLS
    / "skills/models/tao-train-mask2former"
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
    snapshot = campaign_contract.mask2former_registry_snapshot()
    frozen = campaign_contract.FROZEN_V3_QUALIFICATION_CONTRACT
    value = {
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
        "qualification_evidence_path": frozen[
            "qualification_evidence_path"
        ],
        "ptm_stage_manifest_path": frozen["ptm_stage_manifest_path"],
        "ptm_stage_manifest_sha256": frozen[
            "ptm_stage_manifest_sha256"
        ],
        "ptm_stage_content_sha256": frozen[
            "ptm_stage_content_sha256"
        ],
        "tao_pytorch_overlay": runtime_overlay.successor_contract_record(),
        "partition": campaign_contract.FROZEN_SLURM_PARTITION,
        "account": "edgeai_tao-ptm_image-foundation-model-clip",
        "base_results_dir": (
            "/lustre/fsw/portfolios/edgeai/projects/"
            "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam"
        ),
        "container_mounts": "/lustre",
        "time_hours": campaign_contract.FROZEN_SLURM_TIME_HOURS,
        "timeout_hours": campaign_contract.FROZEN_SLURM_TIMEOUT_HOURS,
        "use_requeue": campaign_contract.FROZEN_SLURM_USE_REQUEUE,
        "walltime_policy": copy.deepcopy(
            campaign_contract.SUCCESSOR_WALLTIME_POLICY
        ),
        "max_job_retries": campaign_contract.FROZEN_SLURM_RETRY_CAP,
        "hardware_contract": copy.deepcopy(
            campaign_contract.FROZEN_HARDWARE
        ),
    }
    value["runtime_local_eligibility"] = {
        "schema_version": 2,
        "kind": "direct_full_gpu_qualification_runtime_local_v2",
        "enabled": True,
        "scope": "campaign_local_in_memory_projection",
        "model": "mask2former",
        "task": "instance_segmentation",
        "tao_version": "7.1.0",
        "container_sha256": campaign_contract.FROZEN_SQSH["sha256"],
        "base_registry_version": snapshot["registry_version"],
        "base_registry_sha256": snapshot["registry_sha256"],
        "base_record_sha256_by_checkpoint_id": {
            record["id"]: record["registry_record_sha256"]
            for record in snapshot["records"]
        },
        "qualification_path": frozen["qualification_evidence_path"],
        "qualification_file_sha256": "1" * 64,
        "qualification_evidence_sha256": "2" * 64,
        "qualification_contract_path": frozen["path"],
        "qualification_contract_file_sha256": frozen["file_sha256"],
        "qualification_contract_sha256": frozen["contract_sha256"],
        "qualification_source_commit": frozen["source_commit"],
        "qualification_source_wheel_sha256": frozen["wheel_sha256"],
        "qualification_source_sdk_commit": frozen["sdk_commit"],
        "qualification_source_skills_commit": frozen["skills_commit"],
        "qualification_campaign_sha256": frozen[
            "qualification_campaign_sha256"
        ],
        "qualification_campaign_id": frozen["qualification_campaign_id"],
        "ptm_stage_manifest_path": frozen["ptm_stage_manifest_path"],
        "ptm_stage_manifest_sha256": frozen[
            "ptm_stage_manifest_sha256"
        ],
        "ptm_stage_content_sha256": frozen["ptm_stage_content_sha256"],
        "qualification_runtime_overlay": copy.deepcopy(
            frozen["runtime_overlay"]
        ),
        "qualification_walltime_policy": copy.deepcopy(
            frozen["walltime_policy"]
        ),
        "eligibility_source_commit": value["source_commit"],
        "wheel_sha256": value["wheel_sha256"],
        "sdk_commit": value["sdk_commit"],
        "skills_commit": value["skills_commit"],
        "repository_registry_mutation_allowed": False,
        "projection_persisted_as_global_registry": False,
        "failed_arm_promotion_allowed": False,
        "unsupported_arm_promotion_allowed": False,
        "agent_override_allowed": False,
    }
    return value


def _contract(tmp_path: Path) -> dict:
    value = campaign_contract.build_preregistered_contract(
        campaign_id="mask2former-test",
        dataset=_dataset(),
        skill_dir=str(SKILL_DIR),
        runtime=_runtime(tmp_path),
    )
    value.pop("contract_sha256")
    value["launcher_integrity"] = {
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
        "mask2former_latency_worker_sha256": (
            campaign_contract.sha256_file(
                HERE / "mask2former_latency_worker.py"
            )
        ),
        "runtime_overlay_sha256": campaign_contract.sha256_file(
            HERE / "runtime_overlay.py"
        ),
        "checkpoint_resume_sha256": campaign_contract.sha256_file(
            HERE / "checkpoint_resume.py"
        ),
    }
    value["contract_sha256"] = canonical_sha256(value)
    return campaign_contract.validate_contract(value)


@pytest.fixture
def contract(tmp_path: Path) -> dict:
    return _contract(tmp_path)


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
            "agent_intervention_flags": {
                name: False for name in campaign_contract.AGENT_FLAGS
            },
        }
        value["workflow_sha256"] = canonical_sha256(value)
        return value
    train_metric = (
        {"segm_val_mAP": metric}
        if include_mask_ap
        else {"mIoU": metric}
    )
    eval_metric = (
        {
            "segm_test_mAP": metric,
            "segm_test_mAP50": metric,
            "objective_binding": {
                "reported_metric": "segm_test_mAP",
                "canonical_metric": "segm_val_mAP",
                "value": metric,
            },
        }
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
    snapshot = campaign_contract.mask2former_registry_snapshot()
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
        "campaign_id": "mask2former-direct-full-qualification-test",
        "contract_revision": "qualification_runtime_v3",
        "model": "mask2former",
        "task": "instance_segmentation",
        "primary_metric": "segm_val_mAP",
        "standalone_reported_metric": "segm_test_mAP",
        "standalone_objective_binding": {
            "reported_metric": "segm_test_mAP",
            "canonical_metric": "segm_val_mAP",
        },
        "semantic_miou_accepted_as_mask_ap": False,
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
        "tao_pytorch_overlay": runtime_overlay.contract_record(),
        "walltime_policy": copy.deepcopy(
            campaign_contract.FROZEN_WALLTIME_POLICY
        ),
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "workflows": workflows,
    }
    value["evidence_sha256"] = canonical_sha256(value)
    return value


def _seal_terminal_v3_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    success: bool,
) -> tuple[dict, Path]:
    evidence_path = tmp_path / "completion.json"
    monkeypatch.setitem(
        campaign_contract.FROZEN_V3_QUALIFICATION_CONTRACT,
        "qualification_evidence_path",
        str(evidence_path),
    )
    sealed = _contract(tmp_path)
    checkpoint_id = sealed["ptm_inventory"]["records"][0]["id"]
    workflow = _workflow(checkpoint_id, success=success)
    if success:
        stage = json.loads(
            Path(
                campaign_contract.FROZEN_V3_QUALIFICATION_CONTRACT[
                    "ptm_stage_manifest_path"
                ]
            ).read_text(encoding="utf-8")
        )
        source = stage["checkpoints"][0]
        workflow["source_checkpoint"] = {
            "path": source["path"],
            "size_bytes": source["size_bytes"],
            "sha256": source["sha256"],
        }
        workflow.pop("workflow_sha256")
        workflow["workflow_sha256"] = canonical_sha256(workflow)
    policy = sealed["runtime"]["runtime_local_eligibility"]
    evidence = {
        "schema_version": 1,
        "campaign_id": policy["qualification_campaign_id"],
        "contract_revision": "qualification_runtime_v3",
        "model": "mask2former",
        "task": "instance_segmentation",
        "primary_metric": "segm_val_mAP",
        "standalone_reported_metric": "segm_test_mAP",
        "standalone_objective_binding": {
            "reported_metric": "segm_test_mAP",
            "canonical_metric": "segm_val_mAP",
        },
        "semantic_miou_accepted_as_mask_ap": False,
        "qualification_contract_sha256": policy[
            "qualification_contract_sha256"
        ],
        "qualification_campaign_sha256": policy[
            "qualification_campaign_sha256"
        ],
        "ptm_stage_manifest_path": policy["ptm_stage_manifest_path"],
        "ptm_stage_manifest_sha256": policy[
            "ptm_stage_manifest_sha256"
        ],
        "registry_sha256": policy["base_registry_sha256"],
        "sqsh_sha256": campaign_contract.FROZEN_SQSH["sha256"],
        "tao_pytorch_overlay": copy.deepcopy(
            policy["qualification_runtime_overlay"]
        ),
        "walltime_policy": copy.deepcopy(
            policy["qualification_walltime_policy"]
        ),
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "replacement_workflows_submitted": False,
        "workflows": [workflow],
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    updated_policy = copy.deepcopy(policy)
    updated_policy["qualification_file_sha256"] = (
        campaign_contract.sha256_file(evidence_path)
    )
    updated_policy["qualification_evidence_sha256"] = evidence[
        "evidence_sha256"
    ]
    sealed.pop("contract_sha256")
    sealed["runtime"]["runtime_local_eligibility"] = copy.deepcopy(
        updated_policy
    )
    sealed["qualification_policy"]["runtime_local_eligibility"] = (
        copy.deepcopy(updated_policy)
    )
    sealed["contract_sha256"] = canonical_sha256(sealed)
    return campaign_contract.validate_contract(sealed), evidence_path


def test_exact_tao_identifier_actions_and_task_correct_metric():
    info = yaml.safe_load(
        (SKILL_DIR / "references/skill_info.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert info["network_arch"] == "mask2former"
    assert info["actions"]["train"]["command"] == (
        "mask2former train -e {config_path}"
    )
    assert info["actions"]["evaluate"]["command"] == (
        "mask2former evaluate -e {config_path}"
    )
    settings = campaign_contract.mode_settings("x", "accuracy")
    assert settings["accuracy_metric"] == "segm_val_mAP"
    assert run_campaign._metric_extractor("mIoU: 0.71", "segm_val_mAP") is None
    assert (
        run_campaign._metric_extractor(
            "segm_val_mAP: 0.321", "segm_val_mAP"
        )
        == pytest.approx(0.321)
    )


def test_standalone_metric_is_bound_without_mislabeling(
    monkeypatch: pytest.MonkeyPatch,
):
    status = "\n".join([
        json.dumps({
            "kpi": {
                "segm_test_mAP": "0.42",
                "segm_test_mAP50": "0.61",
                "segm_val_mAP": "0.99",
            }
        }),
    ])
    monkeypatch.setattr(run_campaign, "remote_output", lambda _: status)
    sdk = SimpleNamespace(
        get_job_results_dir=lambda _: "lustre:///lustre/results/job"
    )
    assert run_campaign._status_metric(
        sdk,
        "job",
        action="evaluate",
        names=("segm_test_mAP",),
    ) == pytest.approx(0.42)
    assert qualification_campaign._status_values(
        sdk,
        "job",
        action="evaluate",
        names=("segm_test_mAP50",),
    ) == [pytest.approx(0.61)]
    assert qualification_campaign._status_values(
        sdk,
        "job",
        action="evaluate",
        names=("segm_val_mAP",),
    ) == [pytest.approx(0.99)]


def test_qualification_epoch_metrics_deduplicate_generic_and_rank_records(
    monkeypatch: pytest.MonkeyPatch,
):
    records = []
    for epoch, value in enumerate((0.20, 0.25, 0.31)):
        kpi = {"segm_val_mAP": value}
        records.extend(
            [
                {"message": "Eval metrics generated.", "kpi": kpi},
                {
                    "epoch": epoch,
                    "step": (epoch + 1) * 100,
                    "rank": 0,
                    "kpi": kpi,
                },
                {
                    "epoch": epoch,
                    "step": (epoch + 1) * 100,
                    "rank": 1,
                    "kpi": kpi,
                },
            ]
        )
    monkeypatch.setattr(
        run_campaign,
        "remote_output",
        lambda _: "\n".join(json.dumps(item) for item in records),
    )
    sdk = SimpleNamespace(
        get_job_results_dir=lambda _: "lustre:///lustre/results/job"
    )
    assert qualification_campaign._status_epoch_values(
        sdk,
        "job",
        action="train",
        names=("segm_val_mAP",),
    ) == [pytest.approx(0.20), pytest.approx(0.25), pytest.approx(0.31)]


def test_qualification_epoch_metric_conflicts_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    records = [
        {"epoch": 0, "step": 100, "kpi": {"segm_val_mAP": 0.20}},
        {"epoch": 0, "step": 100, "kpi": {"segm_val_mAP": 0.21}},
    ]
    monkeypatch.setattr(
        run_campaign,
        "remote_output",
        lambda _: "\n".join(json.dumps(item) for item in records),
    )
    sdk = SimpleNamespace(
        get_job_results_dir=lambda _: "lustre:///lustre/results/job"
    )
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="conflicting task metric values",
    ):
        qualification_campaign._status_epoch_values(
            sdk,
            "job",
            action="train",
            names=("segm_val_mAP",),
        )


def test_search_parameters_are_packaged_train_parameters():
    evidence = campaign_contract.validate_packaged_train_schema(SKILL_DIR)
    assert tuple(evidence["explicit_search_parameters"]) == (
        "model.mask_former.num_object_queries",
        "model.mask_former.dec_layers",
        "dataset.augmentation.test_min_size",
        "train.optim.lr",
        "train.optim.weight_decay",
    )
    assert (
        campaign_contract.SEARCH_SPACE[
            "model.mask_former.num_object_queries"
        ]
        == {"type": "integer", "minimum": 50, "maximum": 200}
    )
    assert campaign_contract.SEARCH_SPACE[
        "model.mask_former.dec_layers"
    ] == {"type": "integer", "minimum": 4, "maximum": 10}
    assert campaign_contract.SEARCH_SPACE[
        "dataset.augmentation.test_min_size"
    ] == {"type": "integer", "minimum": 480, "maximum": 800}


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


def test_profile_is_instance_coco_eight_gpu_not_smoke():
    root = _dataset()["prepared_root"]
    profile = campaign_contract.profile_overrides(root)
    assert profile["model"]["mode"] == "instance"
    assert profile["model"]["sem_seg_head"]["num_classes"] == 80
    assert profile["dataset"]["contiguous_id"] is True
    assert profile["dataset"]["label_map"] == (
        f"{root}/tao/label_map_instance.json"
    )
    for split, name in (
        ("train", "instances_train2017.json"),
        ("val", "instances_val2017.json"),
        ("test", "instances_val2017.json"),
    ):
        assert profile["dataset"][split]["type"] == "coco"
        assert profile["dataset"][split]["instance_json"].endswith(name)
        assert profile["dataset"][split]["panoptic_json"] == ""
        assert profile["dataset"][split]["batch_size"] == 1
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
    assert train["precision"] == "fp32"


def test_mask_ap_sanity_is_separate_from_product_selection(contract):
    assert contract["task"] == "instance_segmentation"
    metric = contract["metric_contract"]
    assert metric["required"] == "segm_val_mAP"
    assert metric["validation_reported_metric"] == "segm_val_mAP"
    assert metric["standalone_reported_metric"] == "segm_test_mAP"
    assert metric["standalone_reported_metric50"] == "segm_test_mAP50"
    assert metric["standalone_canonical_objective"] == "segm_val_mAP"
    assert metric["semantic_miou_is_not_an_alias"] is True
    assert metric["known_repository_state"] == (
        "runtime_fix_available_pending_gpu_qualification"
    )
    gate = contract["validation_sanity_gate"]
    assert gate["metric"] == "segm_val_mAP"
    assert gate["minimum"] == 0.05
    assert gate["role"] == (
        "experiment_correctness_gate_not_product_selection"
    )


def test_official_ptm_registry_is_exact_and_hierarchical():
    snapshot = campaign_contract.mask2former_registry_snapshot()
    assert snapshot["record_count"] == 1
    assert [item["id"] for item in snapshot["records"]] == [
        "mask2former.coco.swin_tiny.trainable.v1.0"
    ]
    record = snapshot["records"][0]
    assert record["source"]["official"] is True
    assert record["source"]["version"] == (
        "mask2former_swint_trainable_v1.0"
    )
    assert record["checkpoint_target"] == "train.pretrained_model_path"
    assert snapshot["supported_ids"] == []
    assert snapshot["unverified_ids"] == [record["id"]]


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


def test_v4_requeue_resume_is_bounded_without_scientific_budget_change(
    contract,
):
    runtime = contract["runtime"]
    policy = runtime["walltime_policy"]
    assert runtime["partition"] == "polar3"
    assert runtime["time_hours"] == 4.0
    assert runtime["timeout_hours"] == 3.8
    assert runtime["use_requeue"] is True
    assert policy == campaign_contract.SUCCESSOR_WALLTIME_POLICY
    assert policy["contract_revision"] == "automl_runtime_v4"
    assert policy["observed_full_epoch_minutes_approx"] == 90.0
    assert policy["observed_minimum_training_hours_approx"] == 4.5
    assert policy["training_epochs"] == 3
    assert policy["slurm_self_requeue"] is True
    assert policy["checkpoint_interval_epochs"] == 1
    assert policy["checkpoint_resume_policy"] == (
        "same_job_exact_epoch_step_max_with_history_v2"
    )
    assert policy["timeout_requeue_cap_environment"] == (
        "SLURM_MAX_JOB_RETRIES"
    )
    assert policy["max_timeout_requeues"] == 10
    assert policy["first_post_requeue_decision_recorded"] is True
    assert policy["resume_history_overwrite_allowed"] is False
    assert policy["training_budget_changed"] is False
    assert policy["search_space_changed"] is False
    assert policy["candidate_budget_changed"] is False
    assert policy["retry_policy_changed"] is False
    assert policy["v1_runtime_evidence_preserved"] is True
    assert policy["v2_runtime_evidence_preserved"] is True
    assert contract["search"]["training_epochs"] == 3
    assert contract["search"]["candidate_budget_per_mode"] == 20
    assert contract["search"]["space"] == campaign_contract.SEARCH_SPACE


def test_v5_runtime_paths_preserve_v1_v2_v3_and_replay_evidence():
    assert str(qualification_campaign.DEFAULT_RUNTIME_ROOT).endswith(
        "mask2former_coco2017_ptm_qualification_v3"
    )
    assert str(qualification_campaign.DEFAULT_STAGE_MANIFEST).endswith(
        "mask2former_coco2017_ptm_qualification_v1/"
        "ptm_stage_manifest.json"
    )
    assert str(run_campaign.DEFAULT_RUNTIME_ROOT).endswith(
        "mask2former_coco2017_three_mode_v5"
    )
    assert run_campaign.DEFAULT_CONTRACT.name == "campaign.v5.json"
    assert str(manifest_generator.DEFAULT_QUALIFICATION).endswith(
        "mask2former_coco2017_ptm_qualification_v3_replay_v1/"
        "completion.json"
    )
    assert qualification_campaign.DEFAULT_CONTRACT == Path(
        campaign_contract.FROZEN_V3_QUALIFICATION_CONTRACT["path"]
    )


def test_v4_pins_reviewed_bounded_requeue_sdk_and_product_wheel():
    assert manifest_generator.EXPECTED_SDK_COMMIT == (
        "1a981d79af40d156735f3d89b98495e7818d0891"
    )
    assert (
        subprocess.run(
            [
                "git",
                "-C",
                str(manifest_generator.DEFAULT_SDK),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == manifest_generator.EXPECTED_SDK_COMMIT
    )
    assert campaign_contract.sha256_file(manifest_generator.DEFAULT_WHEEL) == (
        manifest_generator.EXPECTED_WHEEL_SHA256
    )


def test_automatic_successor_waits_without_sealing_or_launching(
    tmp_path: Path,
):
    missing = tmp_path / "completion.json"
    status_path = tmp_path / "automatic_successor_status.json"
    with pytest.raises(TimeoutError, match="waiting for v3"):
        manifest_generator.wait_for_terminal_qualification(
            missing,
            campaign_contract.FROZEN_V3_QUALIFICATION_CONTRACT["path"],
            status_path=status_path,
            poll_seconds=0,
            timeout_seconds=0,
        )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["state"] == "waiting_for_terminal_v3_completion"
    assert status["model_jobs_launched"] is False
    assert not (tmp_path / "campaign.v4.json").exists()


def test_v4_slurm_environment_enables_bounded_four_hour_self_requeue(
    contract,
    monkeypatch: pytest.MonkeyPatch,
):
    for name in (
        "PYTHONPATH",
        "SLURM_USE_SQSH",
        "SLURM_USE_REQUEUE",
        "SLURM_TIME_HOURS",
        "SLURM_TIMEOUT_HOURS",
        "SLURM_MAX_GPUS_PER_NODE",
        "SLURM_PARTITION",
        "SLURM_ACCOUNT",
        "SLURM_BASE_RESULTS_DIR",
        "SLURM_CONTAINER_MOUNTS",
        "SLURM_MAX_JOB_RETRIES",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
    ):
        monkeypatch.setenv(name, run_campaign.os.environ.get(name, ""))
    original_path = list(run_campaign.sys.path)
    try:
        run_campaign.configure_slurm_runtime(contract)
        assert run_campaign.os.environ["SLURM_PARTITION"] == "polar3"
        assert run_campaign.os.environ["SLURM_TIME_HOURS"] == "4.0"
        assert run_campaign.os.environ["SLURM_TIMEOUT_HOURS"] == "3.8"
        assert run_campaign.os.environ["SLURM_USE_REQUEUE"] == "true"
        assert run_campaign.os.environ["SLURM_MAX_JOB_RETRIES"] == "10"
    finally:
        run_campaign.sys.path[:] = original_path


def test_latency_protocol_is_4000_real_coco_validation_samples(contract):
    protocol = contract["latency_protocol"]
    assert protocol["warmup_iterations"] == 50
    assert protocol["repeated_rounds"] == 5
    assert protocol["timed_iterations"] == 100
    assert protocol["expected_replicas"] == 8
    assert protocol["raw_samples_per_candidate"] == 4000
    assert "instance_postprocessing" in protocol["excluded_scope"]
    source = (HERE / "mask2former_latency_worker.py").read_text(
        encoding="utf-8"
    )
    assert "COCODataset" in source
    assert "Mask2formerPlModule" in source
    assert "model(preloaded[" in source
    assert "torch.randn" not in source
    assert "torch.rand(" not in source


def test_model_imports_live_only_below_latency_worker_main():
    tree = ast.parse(
        (HERE / "mask2former_latency_worker.py").read_text(
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


def test_semantic_miou_cannot_qualify_instance_segmentation(tmp_path: Path):
    checkpoint_id = campaign_contract.mask2former_registry_snapshot()[
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
        and "segm_val_mAP" in item["reason"]
        for item in decision.blockers
    )


def test_mislabeled_standalone_val_metric_cannot_qualify(tmp_path: Path):
    checkpoint_id = campaign_contract.mask2former_registry_snapshot()[
        "records"
    ][0]["id"]
    document = _qualification_document(checkpoint_id)
    evaluation = document["workflows"][0]["evaluation"]
    evaluation.pop("segm_test_mAP")
    evaluation["segm_val_mAP"] = 0.20
    document["workflows"][0]["workflow_sha256"] = canonical_sha256({
        key: value
        for key, value in document["workflows"][0].items()
        if key != "workflow_sha256"
    })
    document["evidence_sha256"] = canonical_sha256({
        key: value
        for key, value in document.items()
        if key != "evidence_sha256"
    })
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    decision = audit_qualification(path)
    assert any(
        item["checkpoint_id"] == checkpoint_id
        and item["code"] == "invalid_success_evidence"
        and "segm_test_mAP" in item["reason"]
        for item in decision.blockers
    )


def test_low_finite_mask_ap_does_not_pass_qualification(tmp_path: Path):
    checkpoint_id = campaign_contract.mask2former_registry_snapshot()[
        "records"
    ][0]["id"]
    document = _qualification_document(checkpoint_id, metric=0.049)
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    decision = audit_qualification(path)
    assert any(
        item["checkpoint_id"] == checkpoint_id
        and item["code"] == "invalid_success_evidence"
        and "0.05 COCO mask AP" in item["reason"]
        for item in decision.blockers
    )


def test_unverified_full_run_success_cannot_bypass_registry(tmp_path: Path):
    checkpoint_id = campaign_contract.mask2former_registry_snapshot()[
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
    checkpoint_id = campaign_contract.mask2former_registry_snapshot()[
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


def test_direct_full_qualification_plan_is_plan_only():
    frozen_contract = qualification_campaign.load_frozen_v3_contract(
        qualification_campaign.DEFAULT_CONTRACT
    )
    plan = qualification_campaign.qualification_plan(frozen_contract)
    assert plan["campaign_id"] == (
        "mask2former-coco2017-direct-full-qualification-v3-20260801"
    )
    assert plan["contract_revision"] == "qualification_runtime_v3"
    assert plan["official_checkpoint_ids"] == [
        "mask2former.coco.swin_tiny.trainable.v1.0"
    ]
    assert plan["workflow_count"] == 1
    assert plan["primary_metric"] == "segm_val_mAP"
    assert plan["standalone_reported_metric"] == "segm_test_mAP"
    assert plan["standalone_objective_binding"] == {
        "reported_metric": "segm_test_mAP",
        "canonical_metric": "segm_val_mAP",
    }
    assert plan["training_epochs"] == 3
    assert plan["nodes_per_job"] == 1
    assert plan["gpus_per_job"] == 8
    assert plan["walltime_policy"] == (
        campaign_contract.FROZEN_WALLTIME_POLICY
    )
    assert plan["checkpoint_interval_epochs"] == 1
    assert plan["checkpoint_resume_policy"] == (
        "same_job_exact_epoch_step_max_v1"
    )
    assert plan["slurm_self_requeue"] is True
    assert plan["scheduler_client_constructed"] is False
    assert plan["jobs_submitted"] == 0
    assert plan["cpu_model_runs"] == 0
    assert plan["smoke_model_runs"] == 0
    assert plan["mini_step_runs"] == 0
    assert plan["replacement_workflows_allowed"] is False


def test_v3_completion_records_exact_requeue_resume_contract():
    frozen_contract = qualification_campaign.load_frozen_v3_contract(
        qualification_campaign.DEFAULT_CONTRACT
    )
    completion = qualification_campaign.build_completion(frozen_contract, [])
    assert completion["campaign_id"] == (
        "mask2former-coco2017-direct-full-qualification-v3-20260801"
    )
    assert completion["contract_revision"] == "qualification_runtime_v3"
    assert completion["qualification_contract_sha256"] == (
        frozen_contract["contract_sha256"]
    )
    assert completion["walltime_policy"] == (
        campaign_contract.FROZEN_WALLTIME_POLICY
    )
    payload = copy.deepcopy(completion)
    observed = payload.pop("evidence_sha256")
    assert observed == canonical_sha256(payload)


def test_v1_or_v2_evidence_cannot_satisfy_v3_contract(
    tmp_path: Path,
):
    frozen = qualification_campaign.load_frozen_v3_contract(
        qualification_campaign.DEFAULT_CONTRACT
    )
    assert frozen["runtime"]["walltime_policy"]["contract_revision"] == (
        "qualification_runtime_v3"
    )
    for old_revision in (
        "qualification_runtime_v1",
        "qualification_runtime_v2",
    ):
        changed = copy.deepcopy(frozen)
        changed["runtime"]["walltime_policy"][
            "contract_revision"
        ] = old_revision
        changed.pop("contract_sha256")
        changed["contract_sha256"] = canonical_sha256(changed)
        path = tmp_path / f"campaign-{old_revision}.json"
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(
            run_campaign.CampaignExecutionError,
            match="immutable Mask2Former v3 qualification contract changed",
        ):
            qualification_campaign.load_frozen_v3_contract(path)


def test_direct_qualification_spec_precedence_preserves_coco_profile(
    contract,
):
    checkpoint_id = contract["ptm_inventory"]["records"][0]["id"]
    train, evaluate = qualification_campaign._qualification_specs(
        contract,
        checkpoint_id,
        "/lustre/ptms/mask2former.pth",
    )
    for specification in (train, evaluate):
        assert specification["model"]["mode"] == "instance"
        # The PTM YAML says one class; the explicit AutoML COCO profile has
        # higher precedence and correctly supplies the official 80 classes.
        assert specification["model"]["sem_seg_head"]["num_classes"] == 80
        assert specification["dataset"]["contiguous_id"] is True
        assert specification["dataset"]["label_map"].endswith(
            "/tao/label_map_instance.json"
        )
    assert train["train"]["pretrained_model_path"] == (
        "/lustre/ptms/mask2former.pth"
    )


def test_ptm_stage_is_exact_content_addressed_inventory(
    contract,
    tmp_path: Path,
):
    record = contract["ptm_inventory"]["records"][0]
    stage = {
        "schema_version": 1,
        "model": "mask2former",
        "registry_sha256": contract["ptm_inventory"]["registry_sha256"],
        "stage_complete": True,
        "remote_read_only": True,
        "cpu_model_runs": 0,
        "gpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "scheduler_jobs_submitted": 0,
        "checkpoints": [
            {
                "id": record["id"],
                "path": "/lustre/ptms/mask2former.pth",
                "size_bytes": record["expected_size_bytes"],
                "sha256": "d" * 64,
                "immutable_source_identity": record["source"][
                    "immutable_identity"
                ],
                "remote_read_only": True,
            }
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
    assert loaded[record["id"]]["sha256"] == "d" * 64
    sealed = manifest_generator.ptm_stage_record(path)
    assert sealed["sha256"] == campaign_contract.sha256_file(path)
    assert sealed["manifest_sha256"] == stage["manifest_sha256"]
    assert sealed["checkpoint_ids"] == [record["id"]]

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
        "mask2former.coco.swin_tiny.trainable.v1.0",
        "task-correct segm_val_mAP missing; observed mIoU only",
        code="task_correct_metric_missing",
        diagnostics={"mIoU": [0.7]},
    )
    assert failure["status"] == "failure"
    assert failure["terminal"] is True
    assert failure["failure_preserved"] is True
    assert failure["replacement_submitted"] is False
    assert all(
        value is False
        for value in failure["agent_intervention_flags"].values()
    )
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
    assert len(decision.exclusions) == 1
    assert decision.exclusions[0]["code"] == "direct_full_run_failed"
    assert any(
        item["code"] == "no_runtime_qualified_ptm"
        for item in decision.blockers
    )


def test_runtime_local_projection_admits_exact_success_without_registry_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    sealed, evidence_path = _seal_terminal_v3_evidence(
        tmp_path,
        monkeypatch,
        success=True,
    )
    repository_before = load_ptm_registry()
    base_sha = repository_before.document_sha256
    checkpoint_id = sealed["ptm_inventory"]["records"][0]["id"]

    decision = audit_qualification(
        evidence_path,
        expected_contract=sealed,
    )

    assert decision.runtime_ready is True
    assert decision.checkpoint_ids == (checkpoint_id,)
    assert decision.blockers == ()
    assert decision.exclusions == ()
    assert decision.runtime_registry.checkpoint(checkpoint_id)["status"] == (
        "supported"
    )
    assert decision.runtime_eligibility["qualified_checkpoint_ids"] == [
        checkpoint_id
    ]
    transformations = decision.runtime_eligibility["transformations"]
    assert len(transformations) == 1
    assert transformations[0]["checkpoint_id"] == checkpoint_id
    assert transformations[0]["action"] == (
        "qualify_exact_unverified_identity"
    )
    assert decision.runtime_eligibility["repository_registry_mutated"] is False
    assert (
        decision.runtime_eligibility["projection_persisted_as_global_registry"]
        is False
    )
    repository_after = load_ptm_registry()
    assert repository_after.document_sha256 == base_sha
    assert repository_after.checkpoint(checkpoint_id)["status"] == "unverified"

    exact = SimpleNamespace(
        ok=True,
        prepared=(SimpleNamespace(checkpoint_id=checkpoint_id),),
        exclusions=(),
    )
    run_campaign._validate_live_preflight_cohort(exact, decision)
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="exact qualified and excluded PTM cohorts",
    ):
        run_campaign._validate_live_preflight_cohort(
            SimpleNamespace(ok=True, prepared=(), exclusions=()),
            decision,
        )


def test_terminal_zero_success_projection_stops_automatic_trigger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    sealed, evidence_path = _seal_terminal_v3_evidence(
        tmp_path,
        monkeypatch,
        success=False,
    )
    decision = audit_qualification(
        evidence_path,
        expected_contract=sealed,
    )
    assert decision.runtime_ready is False
    assert decision.checkpoint_ids == ()
    assert len(decision.exclusions) == 1
    assert decision.runtime_eligibility["transformations"] == []
    assert any(
        item["code"] == "no_runtime_qualified_ptm"
        for item in decision.blockers
    )

    monkeypatch.setattr(
        run_campaign,
        "launch_readiness",
        lambda _contract: (
            False,
            [{"code": "ptm_qualification_not_ready", "reason": "final"}],
            decision,
        ),
    )
    root = tmp_path / "automatic_gate"
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="immutable terminal qualification evidence",
    ):
        run_campaign.wait_for_launch_authorization(
            sealed,
            runtime_root=root,
            poll_seconds=0,
        )
    status = json.loads(
        (root / "automatic_trigger_status.json").read_text(encoding="utf-8")
    )
    assert status["terminal"] is True
    assert status["model_jobs_launched"] is False


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
    ] == 19
    assert plan["resources_per_child"]["gpus"] == 8


def test_validation_descriptor_is_bound_to_coco_records(
    contract,
    monkeypatch: pytest.MonkeyPatch,
):
    records = [
        {
            "image_id": index + 1,
            "name": f"{index + 1:012d}.jpg",
            "size_bytes": 100 + index,
            "sha256": f"{index + 1:064x}",
        }
        for index in range(16)
    ]
    observed_command = ""

    def fake_remote(command: str) -> str:
        nonlocal observed_command
        observed_command = command
        return json.dumps(records)

    monkeypatch.setattr(run_campaign, "remote_output", fake_remote)
    descriptor = run_campaign.validation_input_descriptor(contract)
    assert "instances_val2017.json" in observed_command
    assert descriptor["validation_files"] == records
    assert descriptor["preloaded_batches"] == 16
    assert "image_size" not in descriptor


def test_evaluation_spec_is_full_coco_instance_and_eight_gpu(contract):
    specification = run_campaign.evaluation_spec(
        contract,
        {"model": {"mask_former": {"dec_layers": 6}}},
        "/lustre/checkpoints/final.pth",
    )
    assert specification["model"]["mode"] == "instance"
    assert specification["model"]["sem_seg_head"]["num_classes"] == 80
    assert specification["dataset"]["val"]["type"] == "coco"
    assert specification["dataset"]["val"]["instance_json"].endswith(
        "instances_val2017.json"
    )
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
    changed["runtime"]["time_hours"] = 8.0
    changed["runtime"]["timeout_hours"] = 7.8
    changed.pop("contract_sha256")
    changed["contract_sha256"] = canonical_sha256(changed)
    with pytest.raises(
        campaign_contract.CampaignContractError,
        match="bounded v4 requeue/resume policy",
    ):
        campaign_contract.validate_contract(changed)

    changed = copy.deepcopy(contract)
    changed["metric_contract"]["semantic_miou_is_not_an_alias"] = False
    with pytest.raises(campaign_contract.CampaignContractError):
        campaign_contract.validate_contract(changed)

    changed = copy.deepcopy(contract)
    changed["qualification_policy"]["standalone_objective_binding"][
        "reported_metric"
    ] = "segm_val_mAP"
    changed.pop("contract_sha256")
    changed["contract_sha256"] = canonical_sha256(changed)
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


def test_runtime_overlay_contract_is_sealed_and_pythonpath_only(contract):
    overlay = contract["runtime"]["tao_pytorch_overlay"]
    assert overlay == runtime_overlay.successor_contract_record()
    assert overlay["directory"].startswith(
        "/lustre/fsw/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/"
    )
    assert overlay["source_commit"] == (
        "c2e86fe1646ebe89fc280083797dcc544ce88322"
    )
    assert overlay["archive"]["sha256"] == (
        "c395474592d557e0179066c1f99d5cb8f352e10e501621d57043782440dea8c2"
    )
    assert overlay["injection"]["mechanism"] == "PYTHONPATH"
    assert overlay["injection"]["installed_package_mutated"] is False

    changed = copy.deepcopy(contract)
    changed["runtime"]["tao_pytorch_overlay"]["archive"]["sha256"] = "0" * 64
    changed.pop("contract_sha256")
    changed["contract_sha256"] = canonical_sha256(changed)
    with pytest.raises(runtime_overlay.RuntimeOverlayError):
        campaign_contract.validate_contract(changed)


def test_runtime_overlay_wrap_is_fail_closed_and_precedes_action(contract):
    overlay = contract["runtime"]["tao_pytorch_overlay"]
    command = runtime_overlay.wrap_command(
        "mask2former train -e {config_path}",
        overlay,
    )
    assert overlay["installer"]["path"] in command
    assert overlay["installer"]["sha256"] in command
    assert overlay["archive"]["path"] in command
    assert overlay["archive"]["sha256"] in command
    assert overlay["source_commit"] in command
    assert (
        f"export PYTHONPATH={overlay['injection']['pythonpath_root']}:"
        "\"$(printenv PYTHONPATH || true)\""
    ) in command
    assert command.index(overlay["installer"]["path"]) < command.index(
        "mask2former train"
    )
    changed = copy.deepcopy(overlay)
    changed["source_commit"] = "0" * 40
    with pytest.raises(runtime_overlay.RuntimeOverlayError):
        runtime_overlay.wrap_command("mask2former train", changed)


def test_checkpoint_resume_no_checkpoint_leaves_yaml_blank_and_audits(
    tmp_path: Path,
):
    results_dir = tmp_path / "tao-job" / "results_dir"
    spec_path = tmp_path / "tao-job" / "spec.yaml"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(
        yaml.safe_dump(
            {
                "results_dir": str(results_dir),
                "train": {
                    "resume_training_checkpoint_path": (
                        "/unrelated/stale-checkpoint.pth"
                    )
                },
            }
        ),
        encoding="utf-8",
    )

    decision = checkpoint_resume.inject_resume_checkpoint(spec_path)

    updated = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    assert updated["train"]["resume_training_checkpoint_path"] == ""
    assert decision["resume_enabled"] is False
    assert decision["selected_checkpoint"] is None
    assert decision["eligible_checkpoint_count"] == 0
    audit = json.loads(
        (spec_path.parent / checkpoint_resume.DECISION_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    supplied = audit.pop("decision_sha256")
    assert supplied == canonical_sha256(audit)


def test_checkpoint_resume_selects_latest_exact_numeric_independent_of_order(
    tmp_path: Path,
):
    train_dir = tmp_path / "tao-job" / "results_dir" / "train"
    train_dir.mkdir(parents=True)
    paths = [
        train_dir / "model_epoch_001_step_29572.pth",
        train_dir / "model_epoch_002_step_00003.pth",
        train_dir / "model_epoch_002_step_00002.pth",
    ]
    for path in paths:
        path.write_bytes(path.name.encode("utf-8"))

    forward, forward_count = checkpoint_resume.select_latest_checkpoint(
        train_dir,
        entries=paths,
    )
    reverse, reverse_count = checkpoint_resume.select_latest_checkpoint(
        train_dir,
        entries=reversed(paths),
    )

    assert forward == reverse
    assert forward_count == reverse_count == 3
    assert forward is not None
    assert forward["epoch"] == 2
    assert forward["step"] == 3
    assert forward["filename"] == "model_epoch_002_step_00003.pth"


def test_checkpoint_resume_rejects_a_different_tao_job_results_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    expected_root = tmp_path / "results"
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "results_dir": str(
                    expected_root / "other-job" / "results_dir"
                ),
                "train": {"resume_training_checkpoint_path": ""},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TAO_RESULTS_ROOT", str(expected_root))
    monkeypatch.setenv("TAO_JOB_ID", "current-job")

    with pytest.raises(
        checkpoint_resume.CheckpointResumeError,
        match="not this TAO job's output",
    ):
        checkpoint_resume.inject_resume_checkpoint(spec_path)


def test_checkpoint_resume_ignores_symlink_malformed_and_unrelated(
    tmp_path: Path,
):
    train_dir = tmp_path / "tao-job" / "results_dir" / "train"
    train_dir.mkdir(parents=True)
    valid = train_dir / "model_epoch_001_step_00100.pth"
    valid.write_bytes(b"valid")
    malformed = train_dir / "mask2former_model_latest.pth"
    malformed.write_bytes(b"malformed")
    zero_length = train_dir / "model_epoch_009_step_99999.pth"
    zero_length.touch()
    symlink = train_dir / "model_epoch_010_step_99999.pth"
    symlink.symlink_to(valid.name)
    unrelated = tmp_path / "other" / "model_epoch_011_step_99999.pth"
    unrelated.parent.mkdir()
    unrelated.write_bytes(b"unrelated")

    selected, count = checkpoint_resume.select_latest_checkpoint(
        train_dir,
        entries=[symlink, malformed, unrelated, zero_length, valid],
    )

    assert count == 1
    assert selected is not None
    assert selected["path"] == str(valid)


def test_checkpoint_resume_injects_latest_exact_path_into_yaml(
    tmp_path: Path,
):
    results_dir = tmp_path / "tao-job" / "results_dir"
    train_dir = results_dir / "train"
    train_dir.mkdir(parents=True)
    first = train_dir / "model_epoch_000_step_14786.pth"
    latest = train_dir / "model_epoch_001_step_29572.pth"
    first.write_bytes(b"first")
    latest.write_bytes(b"latest")
    spec_path = tmp_path / "tao-job" / "spec.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "results_dir": str(results_dir),
                "train": {"resume_training_checkpoint_path": ""},
            }
        ),
        encoding="utf-8",
    )

    decision = checkpoint_resume.inject_resume_checkpoint(spec_path)

    updated = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    assert updated["train"]["resume_training_checkpoint_path"] == str(latest)
    assert decision["resume_enabled"] is True
    assert decision["trusted_own_checkpoint_load"] is True
    assert decision["selected_checkpoint"]["epoch"] == 1
    assert decision["selected_checkpoint"]["step"] == 29572


def test_checkpoint_resume_post_requeue_without_checkpoint_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    results_dir = tmp_path / "tao-job" / "results_dir"
    spec_path = tmp_path / "tao-job" / "spec.yaml"
    spec_path.parent.mkdir(parents=True)
    original = {
        "results_dir": str(results_dir),
        "train": {"resume_training_checkpoint_path": "/stale/value.pth"},
    }
    spec_path.write_text(yaml.safe_dump(original), encoding="utf-8")
    monkeypatch.setenv("SLURM_JOB_ID", "9988")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")

    with pytest.raises(
        checkpoint_resume.CheckpointResumeError,
        match="post-requeue.*no eligible.*epoch/step checkpoint",
    ):
        checkpoint_resume.inject_resume_checkpoint(spec_path)

    assert yaml.safe_load(spec_path.read_text(encoding="utf-8")) == original
    assert not (
        results_dir / checkpoint_resume.DECISION_HISTORY_DIRECTORY
    ).exists()


@pytest.mark.parametrize("value", ["", "-1", "1.0", "x", "+1"])
def test_checkpoint_resume_rejects_invalid_slurm_restart_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: str,
):
    results_dir = tmp_path / "tao-job" / "results_dir"
    spec_path = tmp_path / "tao-job" / "spec.yaml"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(
        yaml.safe_dump(
            {
                "results_dir": str(results_dir),
                "train": {"resume_training_checkpoint_path": ""},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SLURM_RESTART_COUNT", value)
    with pytest.raises(
        checkpoint_resume.CheckpointResumeError,
        match="SLURM_RESTART_COUNT",
    ):
        checkpoint_resume.inject_resume_checkpoint(spec_path)


def test_checkpoint_resume_history_is_immutable_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    results_dir = tmp_path / "tao-job" / "results_dir"
    train_dir = results_dir / "train"
    train_dir.mkdir(parents=True)
    checkpoint = train_dir / "model_epoch_002_step_00420.pth"
    checkpoint.write_bytes(b"checkpoint")
    spec_path = tmp_path / "tao-job" / "spec.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "results_dir": str(results_dir),
                "train": {"resume_training_checkpoint_path": ""},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SLURM_JOB_ID", "778899")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "2")

    first = checkpoint_resume.inject_resume_checkpoint(spec_path)
    history_path = Path(first["history_path"])
    before = history_path.read_bytes()
    second = checkpoint_resume.inject_resume_checkpoint(spec_path)

    assert first == second
    assert history_path.read_bytes() == before
    assert history_path.stat().st_mode & 0o222 == 0
    assert first["selected_checkpoint"]["epoch"] == 2
    assert first["selected_checkpoint"]["step"] == 420


def test_checkpoint_resume_wraps_command_before_train_and_trusts_only_resume():
    original = "mask2former train -e {config_path}"
    wrapped = checkpoint_resume.wrap_train_command(original)

    assert wrapped.endswith(original)
    assert wrapped.count("{config_path}") == 2
    assert "MASK2FORMER_RESUME_STATE" in wrapped
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1" in wrapped
    assert "fresh) unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" in wrapped
    assert wrapped.index("MASK2FORMER_RESUME_STATE") < wrapped.index(
        "mask2former train"
    )


def test_checkpoint_resume_wrapped_command_executes_fresh_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    results_dir = tmp_path / "tao-job" / "results_dir"
    spec_path = tmp_path / "tao-job" / "spec.yaml"
    marker = tmp_path / "trusted-env.txt"
    spec_path.parent.mkdir(parents=True)

    def write_spec() -> None:
        spec_path.write_text(
            yaml.safe_dump(
                {
                    "results_dir": str(results_dir),
                    "train": {"resume_training_checkpoint_path": ""},
                }
            ),
            encoding="utf-8",
        )

    probe = (
        "import os,sys;"
        "open(sys.argv[2],'w').write("
        "os.environ.get('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD',''))"
    )
    original = " ".join(
        [
            "python3 -c",
            shlex.quote(probe),
            "{config_path}",
            shlex.quote(str(marker)),
        ]
    )
    wrapped = checkpoint_resume.wrap_train_command(original)

    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    write_spec()
    subprocess.run(
        ["bash", "-c", wrapped.format(config_path=shlex.quote(str(spec_path)))],
        check=True,
        timeout=30,
    )
    assert marker.read_text(encoding="utf-8") == ""

    train_dir = results_dir / "train"
    train_dir.mkdir(parents=True)
    latest = train_dir / "model_epoch_001_step_29572.pth"
    latest.write_bytes(b"checkpoint")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    write_spec()
    subprocess.run(
        ["bash", "-c", wrapped.format(config_path=shlex.quote(str(spec_path)))],
        check=True,
        timeout=30,
    )
    assert marker.read_text(encoding="utf-8") == "1"
    updated = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    assert updated["train"]["resume_training_checkpoint_path"] == str(
        latest
    )
    history = sorted(
        (results_dir / checkpoint_resume.DECISION_HISTORY_DIRECTORY).glob(
            "*.json"
        )
    )
    assert [path.name for path in history] == [
        "slurm_job_12345_restart_0000.json",
        "slurm_job_12345_restart_0001.json",
    ]
    first, second = [
        json.loads(path.read_text(encoding="utf-8")) for path in history
    ]
    assert first["resume_enabled"] is False
    assert second["resume_enabled"] is True
    assert second["selected_checkpoint"]["epoch"] == 1
    assert second["selected_checkpoint"]["step"] == 29572
    assert second["post_requeue_slice"] is True


def test_direct_qualification_train_command_uses_resume_wrapper(
    contract,
    monkeypatch: pytest.MonkeyPatch,
):
    import tao_sdk.script_runner

    captured: dict[str, object] = {}

    def fake_build_entrypoint(**kwargs):
        captured.update(kwargs)
        return {"command": kwargs["command"]}

    monkeypatch.setattr(
        tao_sdk.script_runner,
        "build_entrypoint",
        fake_build_entrypoint,
    )
    command, digest = qualification_campaign._entrypoint(
        contract,
        "train",
        {"results_dir": "", "train": {}},
    )

    assert digest == run_campaign.text_sha256(command)
    assert "MASK2FORMER_RESUME_STATE" in command
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1" in command
    assert command.index("MASK2FORMER_RESUME_STATE") < command.index(
        "mask2former train"
    )


def test_runtime_overlay_remote_identity_and_readonly_are_launch_gates(
    contract,
    monkeypatch: pytest.MonkeyPatch,
):
    overlay = contract["runtime"]["tao_pytorch_overlay"]

    def identity(path: str) -> dict:
        if path == overlay["archive"]["path"]:
            return {
                "path": path,
                "size_bytes": overlay["archive"]["size_bytes"],
                "sha256": overlay["archive"]["sha256"],
            }
        return {
            "path": path,
            "size_bytes": overlay["installer"]["size_bytes"],
            "sha256": overlay["installer"]["sha256"],
        }

    monkeypatch.setattr(run_campaign, "_remote_file_identity", identity)
    monkeypatch.setattr(run_campaign, "remote_output", lambda command: "")
    evidence = run_campaign.verify_runtime_overlay_remote(contract)
    assert evidence["remote_read_only"] is True
    assert evidence["injection"]["mechanism"] == "PYTHONPATH"

    monkeypatch.setattr(
        run_campaign,
        "remote_output",
        lambda command: f"{overlay['directory']}/writable.py\n",
    )
    with pytest.raises(run_campaign.CampaignExecutionError):
        run_campaign.verify_runtime_overlay_remote(contract)

    def mismatched(path: str) -> dict:
        value = identity(path)
        if path == overlay["installer"]["path"]:
            value["sha256"] = "0" * 64
        return value

    monkeypatch.setattr(run_campaign, "_remote_file_identity", mismatched)
    monkeypatch.setattr(run_campaign, "remote_output", lambda command: "")
    with pytest.raises(run_campaign.CampaignExecutionError):
        run_campaign.verify_runtime_overlay_remote(contract)


def test_runner_training_command_uses_sealed_overlay_and_resume(contract):
    original = "mask2former train -e {config_path}"
    runner = SimpleNamespace(
        skill_ctx=SimpleNamespace(action_cfg={"command": original})
    )
    digest = run_campaign.configure_runner_runtime_overlay(runner, contract)
    wrapped = runner.skill_ctx.action_cfg["command"]
    assert digest == run_campaign.text_sha256(wrapped)
    assert contract["runtime"]["tao_pytorch_overlay"]["archive"][
        "sha256"
    ] in wrapped
    assert "MASK2FORMER_RESUME_STATE" in wrapped
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1" in wrapped
    assert wrapped.endswith(original)


def test_data_only_stage_manifest_has_no_model_or_scheduler_execution():
    record = qualification_campaign._records()[0]
    stage = {
        "schema_version": 1,
        "model": "mask2former",
        "registry_sha256": (
            campaign_contract.mask2former_registry_snapshot()[
                "registry_sha256"
            ]
        ),
        "created_at_utc": "2026-07-31T00:00:00Z",
        "stage_complete": True,
        "remote_read_only": True,
        "cpu_model_runs": 0,
        "gpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "scheduler_jobs_submitted": 0,
        "checkpoints": [
            {
                "id": record["id"],
                "path": "/lustre/ptms/mask2former_swint.pth",
                "size_bytes": record["expected_size_bytes"],
                "sha256": "d" * 64,
                "mode": "444",
                "immutable_source_identity": record["source"][
                    "immutable_identity"
                ],
                "remote_read_only": True,
            }
        ],
    }
    stage["manifest_sha256"] = canonical_sha256(stage)
    validated = qualification_campaign.validate_stage_document(stage)
    assert validated["gpu_model_runs"] == 0
    assert validated["scheduler_jobs_submitted"] == 0
