"""Contract tests for the OneFormer/full-COCO2017 campaign."""

from __future__ import annotations

import copy
import json
import shlex
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tao_automl.ptm_registry import canonical_sha256, load_ptm_registry

from . import (
    campaign_contract,
    manifest_generator,
    qualification_campaign,
    run_campaign,
)
from . import qualification_gate
from .qualification_gate import (
    QualificationGateError,
    audit_qualification,
)


SKILLS = Path(
    "/localhome/local-rarunachalam/.tao/worktrees/tao-skills-release-7.1.0"
)
SKILL_DIR = SKILLS / "skills/models/tao-train-oneformer"
DATASET_ROOT = (
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
    "cross_model_automl_20260729/coco2017_instance_panoptic_v1"
)


def dataset_record() -> dict:
    return {
        "id": "coco_2017_full_instance_panoptic",
        "official_source": "https://cocodataset.org/",
        "license": "test",
        "root": DATASET_ROOT,
        "train_image_count": 118287,
        "validation_image_count": 5000,
        "train_panoptic_png_count": 118287,
        "validation_panoptic_png_count": 5000,
        "train_panoptic_segment_count": 1329984,
        "validation_panoptic_segment_count": 56728,
        "panoptic_category_count": 133,
        "instance_category_count": 80,
        "panoptic_label_map_sha256": (
            "4b28b3773f0f8e63d836dc20da77276633da72178453458b79e32be8e892ce56"
        ),
        "instance_label_map_sha256": (
            "67f15c4dd7d52aa73025da8307dec17e907f13db6d5d82332a670f73da68c306"
        ),
        "train_panoptic_json_sha256": (
            "560a90a275c65b089d4944fbd8d44d04c57d2e36bf7f66597f367cc4a42bfbbb"
        ),
        "validation_panoptic_json_sha256": (
            "454873a8a01114246066ac841750eb742df3b5e42ce927ef38b49690084ec75a"
        ),
        "content_sha256": (
            "deced9d6766344fe6fc69cd9de3bcff2cba456a14b3391d07bcedb74c250909e"
        ),
        "manifest_path": "/tmp/coco.FILE_MANIFEST.sha256",
        "manifest_sha256": (
            "10566a60498de9998154f44a34445a488c9f030e09f2a7346d20a4a1c55f804e"
        ),
        "file_manifest_entry_count": 246593,
        "remote_sha256sum_check": "passed_all_246593",
        "stage_manifest_path": "/tmp/dataset_stage_manifest.v1.json",
        "stage_manifest_lustre_path": (
            f"{DATASET_ROOT}/dataset_stage_manifest.v1.json"
        ),
        "stage_manifest_sha256": (
            "437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d"
        ),
        "remote_file_manifest_path": (
            f"{DATASET_ROOT}/coco2017_instance_panoptic_v1.FILE_MANIFEST.sha256"
        ),
        "remote_read_only": True,
        "remote_writable_entries_after_lock": 0,
    }


def runtime() -> dict:
    snapshot = campaign_contract.oneformer_registry_snapshot()
    value = {
        "repository": "/localhome/local-rarunachalam/tao-automl",
        "source_commit": "a" * 40,
        "source_dirty": False,
        "wheel_path": "/tmp/automl.whl",
        "wheel_sha256": "b" * 64,
        "sdk_dir": "/tmp/sdk",
        "sdk_commit": "c" * 40,
        "skills_repository": str(SKILLS),
        "skills_commit": "d" * 40,
        "skill_dir": str(SKILL_DIR),
        "qualification_evidence_path": "/tmp/qualification.json",
        "ptm_stage_manifest_path": "/tmp/ptm_stage.json",
        "ptm_stage_manifest_sha256": "e" * 64,
        "ptm_stage_content_sha256": "f" * 64,
        "runtime_overlay_local_archive_path": (
            str(manifest_generator.DEFAULT_RUNTIME_OVERLAY)
        ),
        "runtime_overlay_local_identity": {
            "archive_sha256": (
                campaign_contract.FROZEN_RUNTIME_OVERLAY["archive_sha256"]
            )
        },
        "partition": "polar3",
        "account": "account",
        "base_results_dir": "/lustre/results",
        "container_mounts": "/lustre",
        "time_hours": 4.0,
        "timeout_hours": 3.8,
        "max_job_retries": 10,
        "hardware_contract": copy.deepcopy(campaign_contract.FROZEN_HARDWARE),
    }
    frozen = campaign_contract.FROZEN_V3_QUALIFICATION_CONTRACT
    value["qualification_evidence_path"] = frozen[
        "qualification_evidence_path"
    ]
    value["ptm_stage_manifest_path"] = frozen[
        "ptm_stage_manifest_path"
    ]
    value["ptm_stage_manifest_sha256"] = frozen[
        "ptm_stage_manifest_sha256"
    ]
    value["ptm_stage_content_sha256"] = frozen[
        "ptm_stage_content_sha256"
    ]
    value["runtime_local_eligibility"] = {
        "schema_version": 2,
        "kind": "direct_full_gpu_qualification_runtime_local_v2",
        "enabled": True,
        "scope": "campaign_local_in_memory_projection",
        "model": "oneformer",
        "task": "panoptic_segmentation",
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
        "qualification_campaign_id": frozen[
            "qualification_campaign_id"
        ],
        "ptm_stage_manifest_path": frozen["ptm_stage_manifest_path"],
        "ptm_stage_manifest_sha256": frozen["ptm_stage_manifest_sha256"],
        "ptm_stage_content_sha256": frozen["ptm_stage_content_sha256"],
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


def contract() -> dict:
    return campaign_contract.build_preregistered_contract(
        campaign_id="oneformer-test",
        dataset=dataset_record(),
        skill_dir=str(SKILL_DIR),
        runtime=runtime(),
    )


def _qualification_receipt(token: str) -> dict:
    overlay = campaign_contract.FROZEN_RUNTIME_OVERLAY
    return {
        "schema_version": overlay["receipt_schema_version"],
        "overlay_source_commit": overlay["source_commit"],
        "container_expected_sha256": campaign_contract.FROZEN_SQSH["sha256"],
        "base_site_packages": overlay["base_site_packages"],
        "site_packages": (
            f"/tmp/oneformer-runtime-overlay.{token}/site-packages"
        ),
        "dry_run": False,
        "path": f"/lustre/results/{token}/runtime_overlay/receipt.json",
        "sha256": token[0] * 64,
        "actions": [
            {
                "path": f"nvidia_tao_pytorch/file_{index}.py",
                "action": "replace_base",
                "base_sha256": "b" * 64,
                "sha256": f"{index:064x}",
            }
            for index in range(overlay["file_count"])
        ],
    }


def _successful_workflow(checkpoint_id: str) -> dict:
    stage = json.loads(
        Path(
            campaign_contract.FROZEN_V3_QUALIFICATION_CONTRACT[
                "ptm_stage_manifest_path"
            ]
        ).read_text(encoding="utf-8")
    )
    source = next(
        item for item in stage["checkpoints"] if item["id"] == checkpoint_id
    )
    value = {
        "checkpoint_id": checkpoint_id,
        "status": "success",
        "terminal": True,
        "failure_preserved": False,
        "source_checkpoint": {
            "path": source["path"],
            "size_bytes": source["size_bytes"],
            "sha256": source["sha256"],
        },
        "train": {
            "status": "Complete",
            "full_dataset": True,
            "training_epochs": 1,
            "validation_interval": 1,
            "validation_record_count": 1,
            "nodes": 1,
            "gpus": 8,
            "PQ": 0.125,
            "runtime_overlay_receipt": _qualification_receipt("a1"),
            "terminal_checkpoint": {
                "path": f"/lustre/results/{checkpoint_id}/terminal.pth",
                "size_bytes": 123,
                "sha256": "c" * 64,
            },
        },
        "evaluation": {
            "status": "Complete",
            "full_validation_split": True,
            "nodes": 1,
            "gpus": 8,
            "test_PQ": 0.125,
            "runtime_overlay_receipt": _qualification_receipt("d1"),
        },
        "agent_intervention_flags": {
            name: False for name in campaign_contract.AGENT_FLAGS
        },
    }
    value["workflow_sha256"] = canonical_sha256(value)
    return value


def _failed_workflow(checkpoint_id: str) -> dict:
    value = {
        "checkpoint_id": checkpoint_id,
        "status": "failure",
        "terminal": True,
        "failure_preserved": True,
        "failure_code": "direct_full_training_failed",
        "failure_reason": "frozen unit-test failure",
        "replacement_submitted": False,
        "diagnostics": {},
        "agent_intervention_flags": {
            name: False for name in campaign_contract.AGENT_FLAGS
        },
    }
    value["workflow_sha256"] = canonical_sha256(value)
    return value


def _seal_terminal_v3_evidence(
    tmp_path: Path,
    monkeypatch,
    *,
    successful_ids: set[str],
) -> tuple[dict, Path]:
    evidence_path = tmp_path / "completion.json"
    monkeypatch.setitem(
        campaign_contract.FROZEN_V3_QUALIFICATION_CONTRACT,
        "qualification_evidence_path",
        str(evidence_path),
    )
    sealed = contract()
    snapshot = campaign_contract.oneformer_registry_snapshot()
    workflows = [
        (
            _successful_workflow(record["id"])
            if record["id"] in successful_ids
            else _failed_workflow(record["id"])
        )
        for record in snapshot["records"]
    ]
    policy = sealed["runtime"]["runtime_local_eligibility"]
    evidence = {
        "schema_version": 1,
        "campaign_id": policy["qualification_campaign_id"],
        "model": "oneformer",
        "task": "panoptic_segmentation",
        "metric": "PQ",
        "metric_semantics": (
            "panoptic_quality_from_native_coco_panoptic_annotations"
        ),
        "pq_emitted": True,
        "pq_claim_authorized": True,
        "qualification_contract_sha256": policy[
            "qualification_contract_sha256"
        ],
        "qualification_campaign_sha256": policy[
            "qualification_campaign_sha256"
        ],
        "ptm_stage_manifest_sha256": policy[
            "ptm_stage_manifest_sha256"
        ],
        "ptm_stage_content_sha256": policy["ptm_stage_content_sha256"],
        "registry_sha256": policy["base_registry_sha256"],
        "sqsh_sha256": campaign_contract.FROZEN_SQSH["sha256"],
        "runtime_overlay_sha256": (
            campaign_contract.FROZEN_RUNTIME_OVERLAY["archive_sha256"]
        ),
        "runtime_overlay_source_commit": (
            campaign_contract.FROZEN_RUNTIME_OVERLAY["source_commit"]
        ),
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "replacement_workflows_submitted": False,
        "workflows": workflows,
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


def test_registry_snapshots_all_four_official_nonordinal_arms():
    snapshot = campaign_contract.oneformer_registry_snapshot()
    assert snapshot["record_count"] == 4
    assert snapshot["supported_ids"] == []
    assert len(snapshot["unverified_ids"]) == 4
    assert all(
        record["source"]["official"] is True
        and record["compatible_tao_versions"] == ["==7.1.0"]
        and record["checkpoint_spec_file"]["source"] == "repository"
        for record in snapshot["records"]
    )


def test_direct_qualification_plan_is_four_concurrent_full_gpu_workflows():
    plan = qualification_campaign.qualification_plan(contract())
    assert plan["workflow_count"] == 4
    assert plan["all_workflows_independent"] is True
    assert plan["all_workflows_concurrent"] is True
    assert plan["full_dataset"] is True
    assert plan["training_epochs"] == 1
    assert plan["standalone_full_validation"] is True
    assert plan["resources_per_job"]["nodes"] == 1
    assert plan["resources_per_job"]["gpus"] == 8
    assert plan["cpu_model_runs"] == 0
    assert plan["smoke_model_runs"] == 0
    assert plan["mini_step_runs"] == 0
    assert plan["replacement_workflows_allowed"] is False


def test_direct_qualification_completion_matches_gate_identity():
    value = contract()
    value["launcher_integrity"] = {
        "qualification_campaign_sha256": "1" * 64,
    }
    completion = qualification_campaign.build_completion(value, [])
    assert completion["model"] == "oneformer"
    assert completion["task"] == "panoptic_segmentation"
    assert completion["metric"] == "PQ"
    assert completion["pq_emitted"] is True
    assert completion["pq_claim_authorized"] is True
    assert completion["cpu_model_runs"] == 0
    assert completion["smoke_model_runs"] == 0
    assert completion["mini_step_runs"] == 0
    payload = copy.deepcopy(completion)
    supplied = payload.pop("evidence_sha256")
    assert supplied == canonical_sha256(payload)


def test_v4_qualification_plan_and_completion_recover_only_failed_arm():
    value = contract()
    checkpoint_ids = [
        record["id"] for record in value["ptm_inventory"]["records"]
    ]
    recovery = "oneformer.its.commercial.dinat_large.trainable"
    reused = sorted(set(checkpoint_ids) - {recovery})
    value["qualification_policy"].update(
        {
            "version": 4,
            "qualification_campaign_id": (
                "oneformer-coco2017-direct-full-ptm-qualification-v4-20260801"
            ),
            "recovery_checkpoint_ids": [recovery],
            "reused_checkpoint_ids": reused,
            "checkpoint_resume_policy": copy.deepcopy(
                campaign_contract.CHECKPOINT_RESUME_POLICY
            ),
            "predecessor_evidence": {"file_sha256": "a" * 64},
        }
    )
    value["launcher_integrity"] = {
        "qualification_campaign_sha256": "1" * 64,
    }
    plan = qualification_campaign.qualification_plan(value)
    assert plan["checkpoint_ids"] == [recovery]
    assert plan["workflow_count"] == 1
    assert plan["reused_predecessor_workflow_count"] == 3
    assert plan["replacement_workflows_allowed"] is True

    completion = qualification_campaign.build_completion(value, [])
    assert completion["replacement_workflows_submitted"] is True
    assert completion["replacement_workflow_count"] == 1
    assert completion["reused_predecessor_workflow_count"] == 3
    assert completion["recovery_checkpoint_ids"] == [recovery]


def test_packaged_schema_owns_every_frozen_search_parameter():
    evidence = campaign_contract.validate_packaged_train_schema(SKILL_DIR)
    assert evidence["explicit_search_parameters"] == list(
        campaign_contract.SEARCH_PARAMETERS
    )
    assert evidence["non_train_fields_excluded"] is True


def test_profile_uses_native_panoptic_contract_and_correct_label_map():
    profile = campaign_contract.profile_overrides(DATASET_ROOT)
    dataset = profile["dataset"]
    assert dataset["train"]["annotations"].endswith(
        "/annotations/panoptic_train2017.json"
    )
    assert dataset["val"]["panoptic"].endswith(
        "/annotations/panoptic_val2017"
    )
    assert dataset["label_map"].endswith("/tao/label_map_panoptic.json")
    assert "label_map_instance" not in json.dumps(profile)
    assert dataset["contiguous_id"] is True
    assert dataset["task_prob_train"] == {
        "semantic": 0.0,
        "instance": 0.0,
        "panoptic": 1.0,
    }
    assert profile["model"]["sem_seg_head"]["num_classes"] == 133
    assert profile["train"]["num_gpus"] == 8
    assert profile["train"]["num_nodes"] == 1
    assert profile["train"]["precision"] == "32"
    assert profile["train"]["checkpoint_interval"] == 100
    assert profile["train"]["checkpoint_interval_unit"] == "step"
    assert profile["train"]["resume_training_checkpoint_path"] == ""
    assert campaign_contract.CHECKPOINT_RESUME_POLICY[
        "post_requeue_missing_checkpoint_behavior"
    ] == "fail_closed"
    assert profile["evaluate"]["task"] == "panoptic"


def test_campaign_is_three_independent_objective_aware_jobs():
    value = contract()
    assert value["model"] == value["network_arch"] == "oneformer"
    assert value["task"] == "panoptic_segmentation"
    assert value["execution"]["independent_mode_jobs"] is True
    assert value["execution"]["shared_archive"] is False
    assert [item["mode"] for item in value["modes"]] == list(
        campaign_contract.MODES
    )
    acquisitions = [
        item["objective"]["acquisition"] for item in value["modes"]
    ]
    assert acquisitions == [
        "expected_improvement",
        "constrained_expected_improvement",
        "parego_expected_improvement",
    ]
    namespaces = {
        item["observation_namespace"] for item in value["modes"]
    }
    assert len(namespaces) == 3


def test_latency_retention_does_not_leak_into_multi_objective():
    latency = campaign_contract.mode_settings("campaign", "latency")
    multi = campaign_contract.mode_settings("campaign", "multi_objective")
    assert latency["latency_accuracy_retention"] == {
        "type": "relative",
        "retained_fraction": 0.90,
        "reference": "accuracy_winner",
    }
    assert "latency_accuracy_retention" not in multi
    assert multi["multi_objective_min_accuracy"] is None


def test_metric_semantics_use_task_correct_globally_reduced_pq():
    value = contract()
    assert value["primary_accuracy_metric"] == "PQ"
    assert value["metric_semantics"] == {
        "observed_metric": "panoptic_quality",
        "metric_scale": "unit_interval",
        "source": "native_coco_panoptic_annotations",
        "pq_emitted_by_overlaid_train_evaluate_path": True,
        "pq_claim_authorized": True,
        "semantic_miou_used_as_panoptic_objective": False,
        "distributed_reduction": (
            "global_additive_sufficient_statistics_before_metric"
        ),
    }
    assert all(
        item["settings"]["accuracy_metric"] == "PQ"
        for item in value["modes"]
    )


def test_frozen_pilot_fidelity_and_no_model_smoke_contract():
    value = contract()
    assert value["training_fidelity"]["epochs"] == 1
    assert value["training_fidelity"]["kind"].startswith("one_complete")
    assert value["qualification_policy"]["full_dataset"] is True
    assert value["qualification_policy"]["cpu_model_runs"] == 0
    assert value["qualification_policy"]["smoke_model_runs"] == 0
    assert value["qualification_policy"]["mini_step_runs"] == 0
    assert value["execution"]["cpu_runs"] == 0
    assert value["execution"]["smoke_runs"] == 0


def test_latency_protocol_is_eight_replica_four_thousand_sample_protocol():
    protocol = campaign_contract.LATENCY_PROTOCOL
    assert protocol["warmup_iterations"] == 50
    assert protocol["timed_iterations"] == 100
    assert protocol["repeated_rounds"] == 5
    assert protocol["expected_replicas"] == 8
    assert protocol["raw_samples_per_candidate"] == 4000
    assert protocol["timed_scope"] == "oneformer_model_forward"
    assert protocol["measurement_role"] == "selection_time"


def test_contract_integrity_and_agent_flags_are_fail_closed():
    value = contract()
    assert campaign_contract.validate_contract(value) == value
    assert not any(value["agent_intervention_flags"].values())
    assert not any(value["selection_isolation_flags"].values())
    mutated = copy.deepcopy(value)
    mutated["agent_intervention_flags"]["agent_overrode_winner"] = True
    mutated.pop("contract_sha256")
    mutated["contract_sha256"] = canonical_sha256(mutated)
    with pytest.raises(campaign_contract.CampaignContractError):
        campaign_contract.validate_contract(mutated)
    mutated = copy.deepcopy(value)
    mutated["modes"][2]["settings"]["accuracy_metric"] = "mIoU"
    mutated.pop("contract_sha256")
    mutated["contract_sha256"] = canonical_sha256(mutated)
    with pytest.raises(campaign_contract.CampaignContractError):
        campaign_contract.validate_contract(mutated)


def test_invalid_retention_values_are_rejected():
    for value in (True, 0, -0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(campaign_contract.CampaignContractError):
            campaign_contract._finite_fraction(value, "retention")


def test_manifest_constants_bind_final_coco_stage_and_new_wheel():
    assert (
        manifest_generator.EXPECTED_DATASET_FILE_MANIFEST_SHA256
        == dataset_record()["manifest_sha256"]
    )
    assert (
        manifest_generator.EXPECTED_STAGE_MANIFEST_SHA256
        == dataset_record()["stage_manifest_sha256"]
    )
    assert (
        manifest_generator.WHEEL_BUILD_COMMIT
        == "35972c1bc63e64901c40b0de5be95cc14c19ec80"
    )
    assert manifest_generator.DEFAULT_WHEEL.is_file()
    assert (
        campaign_contract.sha256_file(manifest_generator.DEFAULT_WHEEL)
        == manifest_generator.EXPECTED_WHEEL_SHA256
    )
    with zipfile.ZipFile(manifest_generator.DEFAULT_WHEEL) as archive:
        runtime_source = archive.read("tao_automl/ptm_runtime.py")
    assert b"runtime_registry: PTMRegistry | None" in runtime_source
    assert b"registry=resolved_inventory.runtime_registry" in runtime_source
    overlay = manifest_generator.runtime_overlay_record(
        manifest_generator.DEFAULT_RUNTIME_OVERLAY
    )
    assert overlay["archive_sha256"] == (
        campaign_contract.FROZEN_RUNTIME_OVERLAY["archive_sha256"]
    )
    assert overlay["source_commit"] == (
        campaign_contract.FROZEN_RUNTIME_OVERLAY["source_commit"]
    )


def test_successor_binds_exact_live_v3_contract_and_will_not_seal_early(
    tmp_path,
    monkeypatch,
):
    frozen = campaign_contract.FROZEN_V3_QUALIFICATION_CONTRACT
    source_path = Path(frozen["path"])
    assert campaign_contract.sha256_file(source_path) == frozen[
        "file_sha256"
    ]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_payload = copy.deepcopy(source)
    assert source_payload.pop("contract_sha256") == frozen[
        "contract_sha256"
    ] == canonical_sha256(source_payload)
    assert source["runtime"]["source_commit"] == frozen["source_commit"]
    assert source["ptm_inventory"]["registry_sha256"] == frozen[
        "registry_sha256"
    ]

    missing = tmp_path / "not-terminal" / "completion.json"
    monkeypatch.setitem(
        frozen,
        "qualification_evidence_path",
        str(missing),
    )
    with pytest.raises(
        manifest_generator.ManifestGenerationError,
        match="terminal OneFormer v3 qualification evidence is unavailable",
    ):
        manifest_generator.qualification_evidence_record(
            missing,
            source_path,
        )


def test_successor_and_frozen_qualification_cli_defaults_are_decoupled(
    capsys,
):
    assert run_campaign.DEFAULT_CONTRACT.name == "campaign.v5.json"
    assert run_campaign.DEFAULT_RUNTIME_ROOT.name.endswith("three_mode_v5")
    assert qualification_campaign.DEFAULT_CONTRACT.name == (
        "qualification.v4.json"
    )
    assert qualification_campaign.DEFAULT_RUNTIME_ROOT.name.endswith(
        "ptm_qualification_v4"
    )
    assert qualification_campaign.main(
        [
            "--contract",
            campaign_contract.FROZEN_V3_QUALIFICATION_CONTRACT["path"],
            "--runtime-root",
            str(
                Path(
                    campaign_contract.FROZEN_V3_QUALIFICATION_CONTRACT[
                        "qualification_evidence_path"
                    ]
                ).parent
            ),
        ]
    ) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["campaign_id"] == (
        qualification_campaign.QUALIFICATION_CAMPAIGN_ID
    )


def test_committed_stage_and_complete_file_manifest_build_exact_dataset_record():
    record = manifest_generator.dataset_record(
        manifest_generator.DEFAULT_DATASET_MANIFEST,
        Path(__file__).parents[1]
        / "segmentation_datasets/dataset_stage_manifest.v1.json",
    )
    assert record["root"] == DATASET_ROOT
    assert record["train_image_count"] == 118287
    assert record["validation_image_count"] == 5000
    assert record["panoptic_category_count"] == 133
    assert record["file_manifest_entry_count"] == 246593
    assert record["remote_read_only"] is True


def test_all_failed_qualification_evidence_is_preserved_and_blocks(tmp_path):
    snapshot = campaign_contract.oneformer_registry_snapshot()
    workflows = []
    for record in snapshot["records"]:
        item = {
            "checkpoint_id": record["id"],
            "status": "failure",
            "terminal": True,
            "failure_preserved": True,
            "failure_code": "direct_full_run_failed",
            "failure_reason": "frozen test failure",
            "agent_intervention_flags": {
                name: False for name in campaign_contract.AGENT_FLAGS
            },
        }
        item["workflow_sha256"] = canonical_sha256(item)
        workflows.append(item)
    evidence = {
        "schema_version": 1,
        "campaign_id": "qualification-test",
        "model": "oneformer",
        "task": "panoptic_segmentation",
        "metric": "PQ",
        "metric_semantics": (
            "panoptic_quality_from_native_coco_panoptic_annotations"
        ),
        "pq_emitted": True,
        "pq_claim_authorized": True,
        "registry_sha256": snapshot["registry_sha256"],
        "sqsh_sha256": campaign_contract.FROZEN_SQSH["sha256"],
        "runtime_overlay_sha256": (
            campaign_contract.FROZEN_RUNTIME_OVERLAY["archive_sha256"]
        ),
        "runtime_overlay_source_commit": (
            campaign_contract.FROZEN_RUNTIME_OVERLAY["source_commit"]
        ),
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "workflows": workflows,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    path = tmp_path / "completion.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    decision = audit_qualification(path)
    assert decision.runtime_ready is False
    assert len(decision.exclusions) == 4
    assert any(
        blocker["code"] == "no_runtime_qualified_ptm"
        for blocker in decision.blockers
    )
    with pytest.raises(QualificationGateError):
        decision.assert_runtime_ready()


def test_tampered_failed_workflow_is_blocked_instead_of_excluded(tmp_path):
    snapshot = campaign_contract.oneformer_registry_snapshot()
    workflows = []
    for record in snapshot["records"]:
        item = {
            "checkpoint_id": record["id"],
            "status": "failure",
            "terminal": True,
            "failure_preserved": True,
            "failure_code": "direct_full_run_failed",
            "failure_reason": "frozen test failure",
            "agent_intervention_flags": {
                name: False for name in campaign_contract.AGENT_FLAGS
            },
        }
        item["workflow_sha256"] = canonical_sha256(item)
        workflows.append(item)
    workflows[0]["failure_reason"] = "tampered after sealing"
    evidence = {
        "schema_version": 1,
        "campaign_id": "qualification-tamper-test",
        "model": "oneformer",
        "task": "panoptic_segmentation",
        "metric": "PQ",
        "metric_semantics": (
            "panoptic_quality_from_native_coco_panoptic_annotations"
        ),
        "pq_emitted": True,
        "pq_claim_authorized": True,
        "registry_sha256": snapshot["registry_sha256"],
        "sqsh_sha256": campaign_contract.FROZEN_SQSH["sha256"],
        "runtime_overlay_sha256": (
            campaign_contract.FROZEN_RUNTIME_OVERLAY["archive_sha256"]
        ),
        "runtime_overlay_source_commit": (
            campaign_contract.FROZEN_RUNTIME_OVERLAY["source_commit"]
        ),
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "workflows": workflows,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    path = tmp_path / "completion.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    decision = audit_qualification(path)
    assert len(decision.exclusions) == 3
    assert any(
        blocker["code"] == "invalid_failure_evidence"
        for blocker in decision.blockers
    )


def test_runtime_local_projection_admits_only_success_and_preserves_failures(
    tmp_path,
    monkeypatch,
):
    selected = campaign_contract.oneformer_registry_snapshot()["records"][0][
        "id"
    ]
    sealed, evidence_path = _seal_terminal_v3_evidence(
        tmp_path,
        monkeypatch,
        successful_ids={selected},
    )
    repository_before = load_ptm_registry()
    before_sha = repository_before.document_sha256

    decision = audit_qualification(
        evidence_path,
        expected_contract=sealed,
    )

    watcher_status = tmp_path / "automatic_successor_status.json"
    monkeypatch.setattr(
        manifest_generator,
        "qualification_evidence_record",
        lambda *_args: {
            "qualification_file_sha256": campaign_contract.sha256_file(
                evidence_path
            ),
            "qualification_evidence_sha256": decision.evidence_sha256,
        },
    )
    watched = manifest_generator.wait_for_terminal_qualification(
        evidence_path,
        campaign_contract.FROZEN_V3_QUALIFICATION_CONTRACT["path"],
        status_path=watcher_status,
        poll_seconds=0,
    )
    assert watched["qualification_evidence_sha256"] == (
        decision.evidence_sha256
    )
    assert json.loads(watcher_status.read_text(encoding="utf-8"))[
        "state"
    ] == "terminal_v3_evidence_accepted"

    assert decision.runtime_ready is True
    assert decision.checkpoint_ids == (selected,)
    assert len(decision.exclusions) == 3
    assert decision.blockers == ()
    assert all(
        item["code"] == "direct_full_training_failed"
        for item in decision.exclusions
    )
    assert decision.runtime_registry.checkpoint(selected)["status"] == (
        "supported"
    )
    assert decision.runtime_registry.compatibility(
        "oneformer",
        tao_version="7.1.0",
        task="panoptic_segmentation",
    ).eligible_checkpoint_ids == (selected,)
    eligibility = decision.runtime_eligibility
    assert eligibility["qualified_checkpoint_ids"] == [selected]
    assert eligibility["repository_registry_mutated"] is False
    assert eligibility["projection_persisted_as_global_registry"] is False
    assert eligibility["failed_arms_preserved"] is True
    assert eligibility["transformations"][0]["checkpoint_id"] == selected
    assert eligibility["transformations"][0]["action"] == (
        "qualify_exact_unverified_identity"
    )
    assert set(eligibility["unchanged_checkpoint_ids"]) == (
        set(eligibility["base_record_sha256_by_checkpoint_id"])
        - {selected}
    )
    repository_after = load_ptm_registry()
    assert repository_after.document_sha256 == before_sha
    assert repository_after.checkpoint(selected)["status"] == "unverified"

    report = SimpleNamespace(
        ok=True,
        prepared=(SimpleNamespace(checkpoint_id=selected),),
        exclusions=tuple(
            SimpleNamespace(checkpoint_id=item["checkpoint_id"])
            for item in decision.exclusions
        ),
    )
    run_campaign._validate_live_preflight_cohort(report, decision)

    missing_exclusion = SimpleNamespace(
        ok=True,
        prepared=report.prepared,
        exclusions=report.exclusions[:-1],
    )
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="exact qualified and excluded PTM cohorts",
    ):
        run_campaign._validate_live_preflight_cohort(
            missing_exclusion,
            decision,
        )


def test_runtime_local_projection_with_zero_successes_fails_closed(
    tmp_path,
    monkeypatch,
):
    sealed, evidence_path = _seal_terminal_v3_evidence(
        tmp_path,
        monkeypatch,
        successful_ids=set(),
    )

    decision = audit_qualification(
        evidence_path,
        expected_contract=sealed,
    )

    assert decision.runtime_ready is False
    assert decision.checkpoint_ids == ()
    assert len(decision.exclusions) == 4
    assert decision.runtime_registry.compatibility(
        "oneformer",
        tao_version="7.1.0",
        task="panoptic_segmentation",
    ).eligible_checkpoint_ids == ()
    assert decision.runtime_eligibility["transformations"] == []
    assert any(
        blocker["code"] == "no_runtime_qualified_ptm"
        for blocker in decision.blockers
    )
    with pytest.raises(QualificationGateError):
        decision.assert_runtime_ready()

    monkeypatch.setattr(
        run_campaign,
        "launch_readiness",
        lambda _contract: (
            False,
            [{"code": "ptm_qualification_not_ready", "reason": "final"}],
            decision,
        ),
    )
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="immutable terminal qualification evidence",
    ):
        run_campaign.wait_for_launch_authorization(
            sealed,
            runtime_root=tmp_path / "automatic_gate",
            poll_seconds=0,
        )
    status = json.loads(
        (tmp_path / "automatic_gate/automatic_trigger_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["terminal"] is True
    assert status["model_jobs_launched"] is False


def test_runtime_local_projection_rejects_changed_completion_hash(
    tmp_path,
    monkeypatch,
):
    selected = campaign_contract.oneformer_registry_snapshot()["records"][0][
        "id"
    ]
    sealed, evidence_path = _seal_terminal_v3_evidence(
        tmp_path,
        monkeypatch,
        successful_ids={selected},
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
        match="exact v3 completion",
    ):
        audit_qualification(
            evidence_path,
            expected_contract=changed,
        )


def test_missing_qualification_never_launches_a_model(tmp_path):
    with pytest.raises(QualificationGateError):
        audit_qualification(tmp_path / "missing.json")


def test_qualification_receipt_requires_exact_overlay_and_all_actions():
    receipt = {
        "schema_version": 2,
        "overlay_source_commit": (
            campaign_contract.FROZEN_RUNTIME_OVERLAY["source_commit"]
        ),
        "container_expected_sha256": campaign_contract.FROZEN_SQSH["sha256"],
        "base_site_packages": (
            campaign_contract.FROZEN_RUNTIME_OVERLAY["base_site_packages"]
        ),
        "site_packages": (
            "/tmp/oneformer-runtime-overlay.abc123/site-packages"
        ),
        "dry_run": False,
        "path": "/lustre/results/job/runtime_overlay/receipt.json",
        "sha256": "a" * 64,
        "actions": [
            {
                "path": f"nvidia_tao_pytorch/file_{index}.py",
                "action": "replace_base",
                "base_sha256": "b" * 64,
                "sha256": f"{index:064x}",
            }
            for index in range(
                campaign_contract.FROZEN_RUNTIME_OVERLAY["file_count"]
            )
        ],
    }
    assert qualification_gate._validate_overlay_receipt(
        receipt,
        checkpoint_id="checkpoint",
        phase="train",
    ) == receipt
    receipt["actions"].pop()
    with pytest.raises(QualificationGateError):
        qualification_gate._validate_overlay_receipt(
            receipt,
            checkpoint_id="checkpoint",
            phase="train",
        )


def test_static_sqsh_findings_are_remediated_only_by_exact_overlay():
    blockers = run_campaign.static_sqsh_runtime_blockers(contract())
    assert blockers == []
    mutated = contract()
    mutated["runtime_overlay"] = copy.deepcopy(
        mutated["runtime_overlay"]
    )
    mutated["runtime_overlay"]["source_commit"] = "0" * 40
    blockers = run_campaign.static_sqsh_runtime_blockers(mutated)
    assert len(blockers) == 1
    assert blockers[0]["code"] == "static_oneformer_runtime_blocker"


def test_metric_extractor_accepts_only_exact_panoptic_metric():
    logs = "PQ: 0.125\nPQ=0.375\ntest_PQ=0.625\n"
    assert run_campaign._metric_extractor(logs, "PQ") == pytest.approx(
        0.375
    )
    assert run_campaign._metric_extractor(logs, "mIoU") is None


def test_runtime_overlay_prefix_is_applied_to_every_container_job():
    class DummySDK:
        def __init__(self):
            self.calls = []

        def create_job(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return type("Job", (), {"id": "job-1"})()

    raw = DummySDK()
    wrapped = run_campaign.RuntimeOverlaySDK(raw, contract())
    job = wrapped.create_job(image="image", command="tao model train")
    assert job.id == "job-1"
    command = raw.calls[0][1]["command"]
    tokens = shlex.split(command)
    assert tokens[:2] == ["bash", "-lc"]
    assert len(tokens) == 3
    in_container_payload = tokens[2]
    overlay = campaign_contract.FROZEN_RUNTIME_OVERLAY
    assert in_container_payload.endswith("&& (\ntao model train\n)")
    assert overlay["archive_path"] in in_container_payload
    assert overlay["archive_sha256"] in in_container_payload
    assert "install_overlay.py" in in_container_payload
    assert (
        f"--base-site-packages {overlay['base_site_packages']}"
        in in_container_payload
    )
    assert '--site-packages "$overlay_site"' in in_container_payload
    assert 'export PYTHONPATH="$overlay_site' in in_container_payload
    assert "runtime_overlay/receipt.json" in in_container_payload
    command_evidence = wrapped.command_evidence(job.id)
    assert command_evidence["runtime_overlay_applied"] is True
    assert command_evidence["command_sha256"] == run_campaign.text_sha256(
        command
    )


def test_runtime_overlay_positional_command_is_one_in_container_shell():
    class DummySDK:
        def __init__(self):
            self.calls = []

        def create_job(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return type("Job", (), {"id": "job-positional"})()

    raw = DummySDK()
    wrapped = run_campaign.RuntimeOverlaySDK(raw, contract())
    wrapped.create_job("image", "/bin/bash /lustre/entrypoint.sh")
    arguments = raw.calls[0][0]
    tokens = shlex.split(arguments[1])
    assert tokens[:2] == ["bash", "-lc"]
    assert tokens[2].endswith(
        "&& (\n/bin/bash /lustre/entrypoint.sh\n)"
    )


def test_runtime_overlay_failure_cannot_be_swallowed_by_entrypoint_or_true(
    tmp_path,
):
    marker = tmp_path / "entrypoint-ran"
    entrypoint = "\n".join(
        [
            "false || true",
            f"printf reached > {shlex.quote(str(marker))}",
        ]
    )
    payload = run_campaign._overlay_then_command("false", entrypoint)
    completed = subprocess.run(
        ["bash", "-lc", payload],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert not marker.exists()


def test_evaluation_spec_forces_panoptic_task():
    specification = run_campaign.evaluation_spec(
        contract(),
        {},
        "/lustre/checkpoint.pth",
    )
    assert specification["evaluate"]["task"] == "panoptic"


def test_runner_source_preserves_objective_aware_and_automatic_gates():
    source = Path(run_campaign.__file__).read_text(encoding="utf-8")
    assert "ptm_aware_runtime=True" in source
    assert "resolved_ptm_inventory=inventory" in source
    assert "expected_contract=contract" in source
    assert "registry=decision.runtime_registry" in source
    assert "first_candidate_gate" in source
    assert "automatic_trigger" in source
    assert "gpu_count=8" in source
    assert "num_nodes=1" in source
    assert "TAO_AUTOML_ONEFORMER_LATENCY_COMPLETE" in source
    assert "same_job_max_epoch_step" in source
    assert "model_epoch_([0-9]+)_step_([0-9]+)" in source
    assert "sdk = RuntimeOverlaySDK(" in source
    assert 'names=(\"test_PQ\", \"PQ\")' in source


def test_custom_ranges_equal_frozen_search_space():
    ranges = campaign_contract.custom_ranges()
    assert set(ranges) == set(campaign_contract.SEARCH_PARAMETERS)
    for name, record in ranges.items():
        assert record["valid_min"] == campaign_contract.SEARCH_SPACE[name][
            "minimum"
        ]
        assert record["valid_max"] == campaign_contract.SEARCH_SPACE[name][
            "maximum"
        ]


def test_registry_remains_unverified_until_real_direct_runs():
    model = load_ptm_registry().to_dict()["models"]["oneformer"]
    assert model["default_ptm"] is None
    assert {record["status"] for record in model["checkpoints"]} == {
        "unverified"
    }


def test_shared_checkpoint_resume_selects_exact_max_independent_of_order(
    tmp_path: Path,
):
    helper = qualification_campaign.checkpoint_resume
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    paths = [
        train_dir / "model_epoch_000_step_00100.pth",
        train_dir / "model_epoch_000_step_00200.pth",
        train_dir / "model_epoch_001_step_00001.pth",
    ]
    for path in paths:
        path.write_bytes(b"checkpoint")
    forward, count = helper.select_latest_checkpoint(
        train_dir, entries=paths
    )
    reverse, reverse_count = helper.select_latest_checkpoint(
        train_dir, entries=reversed(paths)
    )
    assert count == reverse_count == 3
    assert forward == reverse
    assert forward["filename"] == "model_epoch_001_step_00001.pth"


def test_shared_checkpoint_resume_injects_same_job_path_and_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    helper = qualification_campaign.checkpoint_resume
    runtime_root = tmp_path / "runtime"
    results_dir = runtime_root / "job-id" / "results_dir"
    train_dir = results_dir / "train"
    train_dir.mkdir(parents=True)
    checkpoint = train_dir / "model_epoch_000_step_00200.pth"
    checkpoint.write_bytes(b"checkpoint")
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "results_dir": str(results_dir),
                "train": {"resume_training_checkpoint_path": ""},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TAO_RESULTS_ROOT", str(runtime_root))
    monkeypatch.setenv("TAO_JOB_ID", "job-id")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    decision = helper.inject_resume_checkpoint(
        spec,
        model_slug="oneformer",
        decision_filename="decision.json",
        history_directory="history",
    )
    loaded = yaml.safe_load(spec.read_text(encoding="utf-8"))
    assert loaded["train"]["resume_training_checkpoint_path"] == str(
        checkpoint
    )
    assert decision["resume_enabled"] is True
    assert Path(decision["history_path"]).is_file()


def test_shared_checkpoint_resume_fails_closed_after_requeue_without_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    helper = qualification_campaign.checkpoint_resume
    results_dir = tmp_path / "job" / "results_dir"
    (results_dir / "train").mkdir(parents=True)
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "results_dir": str(results_dir),
                "train": {"resume_training_checkpoint_path": ""},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    with pytest.raises(
        helper.CheckpointResumeError,
        match="no eligible same-job checkpoint",
    ):
        helper.inject_resume_checkpoint(
            spec,
            model_slug="oneformer",
            decision_filename="decision.json",
            history_directory="history",
        )


def test_terminal_checkpoint_probe_selects_latest_saved_numeric_step(
    tmp_path: Path,
):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "model_epoch_000_step_00100.pth").write_bytes(b"first")
    latest = train_dir / "model_epoch_000_step_14700.pth"
    latest.write_bytes(b"latest")
    (train_dir / "unrelated.pth").write_bytes(b"ignored")
    (train_dir / "model_epoch_000_step_14786.pth").symlink_to(latest)

    result = subprocess.run(
        [
            "python3",
            "-c",
            run_campaign._terminal_checkpoint_probe_script(),
            str(train_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    evidence = json.loads(result.stdout)

    assert evidence["filename"] == "model_epoch_000_step_14700.pth"
    assert evidence["epoch"] == 0
    assert evidence["step"] == 14700
    assert evidence["eligible_checkpoint_count"] == 2


def test_terminal_checkpoint_probe_rejects_equal_numeric_maximum(
    tmp_path: Path,
):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "model_epoch_000_step_14700.pth").write_bytes(b"first")
    (train_dir / "model_epoch_00_step_014700.pth").write_bytes(b"duplicate")

    result = subprocess.run(
        [
            "python3",
            "-c",
            run_campaign._terminal_checkpoint_probe_script(),
            str(train_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0


def test_latency_payload_is_compressed_below_safe_argument_budget():
    repository = Path(run_campaign.__file__).resolve().parents[3]
    contract = {
        "runtime": {"repository": str(repository)},
        "latency_protocol": copy.deepcopy(
            campaign_contract.LATENCY_PROTOCOL
        ),
        "launcher_integrity": {
            "oneformer_latency_worker_sha256": (
                campaign_contract.sha256_file(
                    Path(run_campaign.__file__).with_name(
                        "oneformer_latency_worker.py"
                    )
                )
            )
        },
        "sqsh": {"sha256": "a" * 64},
    }
    descriptor = {
        "schema_version": 1,
        "validation_files": [
            {"name": f"image-{index}.jpg", "sha256": "b" * 64}
            for index in range(16)
        ],
    }

    command, latency_contract = run_campaign._payload_command(
        contract,
        descriptor,
    )

    assert len(command.encode("utf-8")) < 64 * 1024
    assert "zlib.decompress" in command
    assert latency_contract["expected_replicas"] == 8
