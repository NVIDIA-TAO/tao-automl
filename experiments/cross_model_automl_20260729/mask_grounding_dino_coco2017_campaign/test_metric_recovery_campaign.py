"""Contract tests for evaluator-only Mask Grounding DINO recovery."""

import copy
import json
from pathlib import Path

import pytest

from tao_automl.ptm_registry import canonical_sha256

from . import (
    campaign_contract,
    manifest_generator,
    metric_recovery_campaign as recovery,
)
from .qualification_gate import QualificationGateError, audit_qualification


def test_frozen_predecessor_yields_exact_four_checkpoint_cohort():
    completion = json.loads(
        recovery.DEFAULT_PREDECESSOR_COMPLETION.read_text(encoding="utf-8")
    )

    records = recovery._checkpoint_records(completion)

    assert len(records) == 4
    assert [record["checkpoint_id"] for record in records] == sorted(
        record["checkpoint_id"] for record in records
    )
    assert all(
        record["terminal_checkpoint"]["terminal_epoch_index"] == 2
        and record["terminal_checkpoint"]["training_epochs"] == 3
        for record in records
    )


def test_overlay_command_is_fail_closed_and_does_not_mutate_base():
    command = recovery.overlay_install_command(
        {"overlay": recovery.OVERLAY}
    )

    assert "sha256sum" in command
    assert recovery.OVERLAY["archive_sha256"] in command
    assert "cp -as" in command
    assert "install_overlay.py" in command
    assert "export PYTHONPATH=" in command
    assert (
        f'{recovery.OVERLAY["base_site_packages"]}/nvidia_tao_pytorch/.'
        in command
    )


def test_workflow_submits_evaluation_only_with_frozen_checkpoint(
    monkeypatch, tmp_path
):
    checkpoint = {
        "path": "/lustre/frozen/model_epoch_002_step_10992.pth",
        "sha256": "a" * 64,
        "size_bytes": 123,
        "terminal_epoch_index": 2,
        "training_epochs": 3,
    }
    record = {
        "checkpoint_id": "mask_grounding_dino.test",
        "source_train_job_id": "train-job",
        "failed_evaluation_job_id": "old-eval-job",
        "terminal_checkpoint": checkpoint,
    }
    contract = {
        "overlay": recovery.OVERLAY,
        "runtime": {},
    }
    called = []

    monkeypatch.setattr(
        recovery.qualification_campaign,
        "_qualification_specs",
        lambda *_: ({"train": {}}, {"evaluate": {"checkpoint": ""}}),
    )

    def entrypoint(_contract, action, specification):
        called.append((action, specification["evaluate"]["checkpoint"]))
        return "tao model evaluate -e {config_path}", "b" * 64

    monkeypatch.setattr(recovery.qualification_campaign, "_entrypoint", entrypoint)

    class Job:
        id = "new-eval-job"

    submitted = []

    def submit(_sdk, _contract, command):
        submitted.append(command)
        return Job()

    monkeypatch.setattr(recovery.qualification_campaign, "_submit", submit)
    monkeypatch.setattr(recovery.run_campaign, "_wait_for_job", lambda *_a, **_k: "Complete")
    monkeypatch.setattr(
        recovery.qualification_campaign,
        "_status_values",
        lambda _sdk, _job, *, action, names: (
            [0.11] if names[0].startswith("[segm]") else [0.22]
        ),
    )

    result = recovery._run_one(contract, record, tmp_path, object())

    assert called == [("evaluate", checkpoint["path"])]
    assert len(submitted) == 1
    assert "install_overlay.py" in submitted[0]
    assert " evaluate " in submitted[0]
    assert " train " not in submitted[0]
    assert result["status"] == "success"
    assert result["training_jobs_submitted"] == 0
    assert result["segm_val_mAP50_95"] == 0.11
    assert result["bbox_val_mAP50_95"] == 0.22


def test_sdk_state_directory_exists_before_database_construction(tmp_path):
    observed = {}

    class FakeSDK:
        def __init__(self, *, poll_interval, state_file):
            observed["parent_exists"] = state_file.parent.is_dir()
            observed["poll_interval"] = poll_interval
            observed["state_file"] = state_file

    workflow_dir = tmp_path / "previously-missing" / "workflow"
    sdk = recovery._sdk_for_workflow(FakeSDK, workflow_dir)

    assert isinstance(sdk, FakeSDK)
    assert observed == {
        "parent_exists": True,
        "poll_interval": 10,
        "state_file": workflow_dir / "slurm_state.json",
    }


def test_v5_recovery_is_bound_to_v3_training_and_projects_all_ptms(
    monkeypatch,
):
    real_git = manifest_generator._git

    def clean_git(repository, *arguments):
        if arguments == ("status", "--porcelain"):
            return ""
        return real_git(repository, *arguments)

    monkeypatch.setattr(manifest_generator, "_git", clean_git)
    monkeypatch.setattr(recovery.run_campaign, "_git", clean_git)
    contract = manifest_generator.build_contract()
    local = recovery.run_campaign.verify_local_contract(contract)
    decision = audit_qualification(
        manifest_generator.DEFAULT_QUALIFICATION,
        expected_contract=contract,
    )

    assert decision.runtime_ready is True
    assert local["artifacts"]["predecessor_qualification"]["sha256"] == (
        contract["runtime"]["predecessor_failure_evidence"][
            "file_sha256"
        ]
    )
    assert len(decision.qualified) == 4
    assert decision.blockers == ()
    assert all(
        item.val_mask_ap is None
        and item.standalone_mask_ap >= 0.05
        and item.metric_evidence_kind
        == "v3_training_plus_v5_standalone_recovery"
        for item in decision.qualified
    )
    assert decision.runtime_eligibility[
        "qualification_successor_version"
    ] == 5
    assert decision.runtime_eligibility["training_jobs_submitted"] == 0
    assert decision.runtime_eligibility[
        "evaluation_recovery_jobs_submitted"
    ] == 4
    assert contract["runtime"]["evaluation_overlay"] == recovery.OVERLAY
    command = recovery.run_campaign.evaluator_overlay_install_command(
        contract
    )
    assert recovery.OVERLAY["archive_path"] in command
    assert recovery.OVERLAY["archive_sha256"] in command
    assert "runtime_overlay/receipt.json" in command
    assert "export PYTHONPATH=" in command


def test_selection_evaluator_rejects_missing_or_changed_overlay(monkeypatch):
    real_git = manifest_generator._git

    def clean_git(repository, *arguments):
        if arguments == ("status", "--porcelain"):
            return ""
        return real_git(repository, *arguments)

    monkeypatch.setattr(manifest_generator, "_git", clean_git)
    contract = manifest_generator.build_contract()
    missing = copy.deepcopy(contract)
    missing["runtime"].pop("evaluation_overlay")
    with pytest.raises(
        recovery.run_campaign.CampaignExecutionError,
        match="requires the sealed evaluator overlay",
    ):
        recovery.run_campaign.evaluator_overlay_install_command(missing)

    changed = copy.deepcopy(contract)
    changed.pop("contract_sha256")
    changed["runtime"]["evaluation_overlay"]["archive_sha256"] = "0" * 64
    changed["contract_sha256"] = canonical_sha256(changed)
    with pytest.raises(
        campaign_contract.CampaignContractError,
        match="v5 evaluation-recovery eligibility contract changed",
    ):
        campaign_contract.validate_contract(changed)


def test_evaluator_overlay_successor_seals_exact_predecessor(monkeypatch):
    predecessor = Path(
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "mask_grounding_dino_coco2017_three_mode_v5/campaign.v5.json"
    )
    real_git = manifest_generator._git

    def clean_git(repository, *arguments):
        if arguments == ("status", "--porcelain"):
            return ""
        return real_git(repository, *arguments)

    monkeypatch.setattr(manifest_generator, "_git", clean_git)
    contract = manifest_generator.build_contract(
        resume_predecessor_contract=predecessor
    )
    record = contract["runtime"]["resume_predecessor_contract"]
    predecessor_document = json.loads(predecessor.read_text(encoding="utf-8"))
    assert record["contract_sha256"] == predecessor_document[
        "contract_sha256"
    ]
    assert record["training_job_reuse_required"] is True
    assert record["training_relaunch_allowed"] is False
    assert record["recommendation_change_allowed"] is False
    assert record["objective_policy_change_allowed"] is False
    assert contract["runtime"]["runtime_local_eligibility"] == (
        predecessor_document["runtime"]["runtime_local_eligibility"]
    )
    assert contract["qualification_policy"]["runtime_local_eligibility"] == (
        predecessor_document["runtime"]["runtime_local_eligibility"]
    )


def test_v5_recovery_rejects_changed_metric(tmp_path):
    evidence = json.loads(
        manifest_generator.DEFAULT_QUALIFICATION.read_text(encoding="utf-8")
    )
    evidence["workflows"][0]["segm_val_mAP50_95"] = 0.99
    changed = tmp_path / "completion.json"
    changed.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(
        manifest_generator.ManifestGenerationError,
        match="identity changed",
    ):
        manifest_generator.qualification_evidence_record(
            changed,
            manifest_generator.DEFAULT_QUALIFICATION_CONTRACT,
        )


def test_v5_contract_rejects_changed_training_provenance(monkeypatch):
    real_git = manifest_generator._git

    def clean_git(repository, *arguments):
        if arguments == ("status", "--porcelain"):
            return ""
        return real_git(repository, *arguments)

    monkeypatch.setattr(manifest_generator, "_git", clean_git)
    contract = manifest_generator.build_contract()
    changed = copy.deepcopy(contract)
    changed.pop("contract_sha256")
    changed["runtime"]["runtime_local_eligibility"][
        "training_qualification_file_sha256"
    ] = "0" * 64
    changed["qualification_policy"]["runtime_local_eligibility"] = copy.deepcopy(
        changed["runtime"]["runtime_local_eligibility"]
    )
    changed["contract_sha256"] = canonical_sha256(changed)

    with pytest.raises(
        QualificationGateError,
        match="training completion changed",
    ):
        audit_qualification(
            manifest_generator.DEFAULT_QUALIFICATION,
            expected_contract=changed,
        )
