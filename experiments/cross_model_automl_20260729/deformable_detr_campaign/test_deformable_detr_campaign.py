"""Pure contract tests for the direct full Deformable DETR qualification."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from . import manifest_generator
from . import run_campaign


HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "campaign.v1.json"


@pytest.fixture()
def manifest():
    return manifest_generator.load_manifest(MANIFEST_PATH)


def _successful_workflow(manifest, workflow_id):
    record = run_campaign._initial_workflow(manifest, workflow_id)
    record.update(
        {
            "status": "success",
            "terminal": True,
            "failure_preserved": False,
            "metrics": {"mAP": 0.31, "mAP50": 0.52},
            "jobs": {
                "train": {
                    "status": "Complete",
                    "result_root": f"/lustre/results/{workflow_id}/train-job",
                    "status_evidence": {
                        "path": (
                            f"/lustre/results/{workflow_id}/train-job/"
                            "results_dir/train/status.json"
                        ),
                        "sha256": "a" * 64,
                        "size_bytes": 123,
                        "record_count": 11,
                        "validation_record_count": 10,
                        "validation_metrics": [
                            {"mAP": 0.2, "mAP50": 0.4}
                            for _ in range(10)
                        ],
                        "terminal_success_message": (
                            "Train finished successfully."
                        ),
                        "terminal_success": True,
                    },
                },
                "evaluation": {
                    "status": "Complete",
                    "result_root": (
                        f"/lustre/results/{workflow_id}/evaluation-job"
                    ),
                    "status_evidence": {
                        "path": (
                            f"/lustre/results/{workflow_id}/evaluation-job/"
                            "results_dir/evaluate/status.json"
                        ),
                        "sha256": "b" * 64,
                        "size_bytes": 123,
                        "record_count": 2,
                        "test_metric_record_count": 1,
                        "metrics": {"mAP": 0.31, "mAP50": 0.52},
                        "terminal_success_message": (
                            "Evaluate finished successfully."
                        ),
                        "terminal_success": True,
                    },
                },
            },
        }
    )
    return record


def _self_hash_completion(value):
    value.pop("completion_sha256", None)
    value["completion_sha256"] = manifest_generator.canonical_sha(value)
    return value


def test_manifest_is_exactly_two_direct_full_gpu_workflows(manifest):
    assert manifest["model"] == "deformable_detr"
    assert manifest["cpu_runs"] == 0
    assert manifest["smoke_runs"] == 0
    assert manifest["execution"] == {
        "kind": "direct_full_qualification",
        "cpu_runs": 0,
        "smoke_runs": 0,
        "ministep_runs": 0,
        "local_model_runs": 0,
        "qualification_workflows": 2,
        "parallel_workflows": True,
        "full_training": True,
        "full_in_training_validation": True,
        "standalone_evaluation": True,
        "requires_direct_full_dataset_acknowledgement": True,
        "submission_ready": True,
    }
    assert manifest["runtime"]["nodes"] == 1
    assert manifest["runtime"]["tasks_per_node"] == 1
    assert manifest["runtime"]["gpus_per_node"] == 8
    assert manifest["runtime"]["hardware_contract"]["gpu_name"] == (
        "NVIDIA A100-SXM4-80GB"
    )
    assert manifest["qualification"]["training_epochs"] == 10
    assert manifest["qualification"]["validation_interval"] == 1
    assert [item["workflow_id"] for item in manifest["ptms"]] == [
        "gcvit_tiny",
        "resnet50",
    ]


@pytest.mark.parametrize(
    ("workflow_id", "backbone"),
    [("resnet50", "resnet_50"), ("gcvit_tiny", "gc_vit_tiny")],
)
def test_train_spec_is_full_voc_ten_epoch_ddp(
    manifest, workflow_id, backbone
):
    spec = run_campaign.build_train_spec(manifest, workflow_id)
    assert spec["model"]["backbone"] == backbone
    assert spec["dataset"]["num_classes"] == 21
    assert spec["dataset"]["eval_class_ids"] == list(range(1, 21))
    assert "instances_train2007.json" in (
        spec["dataset"]["train_data_sources"][0]["json_file"]
    )
    assert "instances_val2007.json" in (
        spec["dataset"]["val_data_sources"][0]["json_file"]
    )
    assert spec["train"]["num_epochs"] == 10
    assert spec["train"]["validation_interval"] == 1
    assert spec["train"]["num_nodes"] == 1
    assert spec["train"]["num_gpus"] == 8
    assert spec["train"]["gpu_ids"] == list(range(8))
    assert spec["train"]["distributed_strategy"] == "ddp"
    assert spec["train"]["is_dry_run"] is False


@pytest.mark.parametrize("workflow_id", ["resnet50", "gcvit_tiny"])
def test_standalone_evaluation_carries_architecture_and_full_split(
    manifest, workflow_id
):
    checkpoint = f"/lustre/results/{workflow_id}/model_epoch_009_step_100.pth"
    train = run_campaign.build_train_spec(manifest, workflow_id)
    evaluate = run_campaign.build_evaluation_spec(
        manifest, workflow_id, checkpoint
    )
    assert evaluate["model"] == train["model"]
    assert evaluate["evaluate"]["checkpoint"] == checkpoint
    assert evaluate["evaluate"]["num_gpus"] == 8
    assert evaluate["evaluate"]["gpu_ids"] == list(range(8))
    assert evaluate["dataset"]["test_data_sources"]["json_file"].endswith(
        "instances_val2007.json"
    )
    assert evaluate["dataset"]["eval_class_ids"] == list(range(1, 21))


def test_launch_requires_explicit_direct_full_acknowledgement(manifest):
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="acknowledge-direct-full-dataset",
    ):
        run_campaign.main(["--manifest", str(MANIFEST_PATH), "--launch"])


def test_explicit_env_file_is_parsed_and_loads_only_slurm_routing(
    tmp_path, monkeypatch
):
    env_file = tmp_path / "config.env"
    env_file.write_text(
        "SLURM_HOSTNAME=login.example\n"
        "SLURM_USER=campaign-user\n"
        "SSH_KEY_PATH=/tmp/campaign-key\n"
        "NGC_KEY=must-not-be-loaded\n",
        encoding="utf-8",
    )
    args = run_campaign.parse_args(["--env-file", str(env_file)])
    assert args.env_file == env_file
    monkeypatch.delenv("NGC_KEY", raising=False)
    loaded = run_campaign.load_launch_environment(args.env_file)
    assert loaded == ("SLURM_HOSTNAME", "SLURM_USER", "SSH_KEY_PATH")
    assert "NGC_KEY" not in __import__("os").environ


def test_terminal_completion_is_hashed_and_covers_both_outcomes(
    manifest, tmp_path
):
    workflow_ids = tuple(item["workflow_id"] for item in manifest["ptms"])
    for workflow_id in workflow_ids:
        record = _successful_workflow(manifest, workflow_id)
        run_campaign.atomic_json(
            tmp_path / workflow_id / "workflow_completion.json",
            record,
        )
    completion = run_campaign.build_completion(
        manifest,
        tmp_path,
        workflow_ids,
        {workflow_id: 0 for workflow_id in workflow_ids},
    )
    assert completion["status"] == "success"
    assert completion["terminal"] is True
    assert completion["model"] == "deformable_detr"
    assert completion["outcomes"] == {
        "gcvit_tiny": "success",
        "resnet50": "success",
    }
    assert run_campaign.validate_completion(completion, manifest) == completion
    tampered = copy.deepcopy(completion)
    tampered["outcomes"]["resnet50"] = "terminal_failure"
    _self_hash_completion(tampered)
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="inconsistent",
    ):
        run_campaign.validate_completion(tampered, manifest)


def test_self_hashed_completion_rejects_status_exit_and_metric_disagreement(
    manifest, tmp_path
):
    workflow_ids = tuple(item["workflow_id"] for item in manifest["ptms"])
    for workflow_id in workflow_ids:
        run_campaign.atomic_json(
            tmp_path / workflow_id / "workflow_completion.json",
            _successful_workflow(manifest, workflow_id),
        )
    completion = run_campaign.build_completion(
        manifest,
        tmp_path,
        workflow_ids,
        {workflow_id: 0 for workflow_id in workflow_ids},
    )

    wrong_exit = copy.deepcopy(completion)
    wrong_exit["workflows"][0]["process_exit_code"] = 1
    _self_hash_completion(wrong_exit)
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="status and exit code disagree",
    ):
        run_campaign.validate_completion(wrong_exit, manifest)

    invalid_metric = copy.deepcopy(completion)
    invalid_metric["workflows"][0]["metrics"]["mAP50"] = 1.01
    _self_hash_completion(invalid_metric)
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="invalid mAP50",
    ):
        run_campaign.validate_completion(invalid_metric, manifest)


def test_terminal_checkpoint_is_exact_epoch_and_content_addressed(monkeypatch):
    class _SDK:
        @staticmethod
        def get_job_results_dir(_job_id):
            return "lustre:///lustre/results/job"

    calls = []

    def remote(command, *, timeout=900):
        calls.append((command, timeout))
        if command.startswith("find "):
            return (
                "/lustre/results/job/results_dir/train/"
                "model_epoch_009_step_00100.pth\n"
            )
        return "123456\n" + "a" * 64 + "  checkpoint.pth\n"

    monkeypatch.setattr(run_campaign, "remote_output", remote)
    identity = run_campaign._terminal_checkpoint(
        _SDK(),
        "job-id",
        training_epochs=10,
    )
    assert identity == {
        "path": (
            "/lustre/results/job/results_dir/train/"
            "model_epoch_009_step_00100.pth"
        ),
        "sha256": "a" * 64,
        "size_bytes": 123456,
        "training_epochs": 10,
        "terminal_epoch_index": 9,
    }
    assert "model_epoch_009_step_*.pth" in calls[0][0]
    assert "*.ckpt" not in calls[0][0]


def test_terminal_checkpoint_rejects_ambiguous_epoch(monkeypatch):
    class _SDK:
        @staticmethod
        def get_job_results_dir(_job_id):
            return "/lustre/results/job"

    monkeypatch.setattr(
        run_campaign,
        "remote_output",
        lambda *_args, **_kwargs: (
            "/lustre/results/job/model_epoch_009_step_1.pth\n"
            "/lustre/results/job/model_epoch_009_step_2.pth\n"
        ),
    )
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="emitted 2 exact",
    ):
        run_campaign._terminal_checkpoint(
            _SDK(),
            "job-id",
            training_epochs=10,
        )


def test_training_status_requires_ten_validation_records_and_terminal_success(
    monkeypatch,
):
    class _SDK:
        @staticmethod
        def get_job_results_dir(_job_id):
            return "/lustre/results/train-job"

    records = [
        {
            "message": "Eval metrics generated.",
            "kpi": {
                "val_mAP": 0.1 + index / 100,
                "val_mAP50": 0.2 + index / 100,
            },
        }
        for index in range(10)
    ]
    records.append({"message": "Train finished successfully."})
    text = "".join(json.dumps(record) + "\n" for record in records)
    calls = []

    def remote(command, **_kwargs):
        calls.append(command)
        if " cat " in f" {command} ":
            return text
        return f"{len(text)}\n" + "b" * 64 + "  status.json\n"

    monkeypatch.setattr(run_campaign, "remote_output", remote)
    evidence = run_campaign._training_status_evidence(
        _SDK(),
        "job-id",
        expected_validation_records=10,
    )
    assert evidence["validation_record_count"] == 10
    assert evidence["terminal_success"] is True
    assert evidence["path"] == (
        "/lustre/results/train-job/results_dir/train/status.json"
    )
    assert evidence["sha256"] == "b" * 64
    assert all("find " not in command for command in calls)


@pytest.mark.parametrize(
    ("validation_count", "terminal", "error"),
    [
        (9, True, "emitted 9 in-training validation records"),
        (10, False, "lacks terminal TAO train success"),
    ],
)
def test_training_status_rejects_incomplete_evidence(
    monkeypatch,
    validation_count,
    terminal,
    error,
):
    class _SDK:
        @staticmethod
        def get_job_results_dir(_job_id):
            return "lustre:///lustre/results/train-job"

    records = [
        {
            "message": "Eval metrics generated.",
            "kpi": {"val_mAP": 0.31, "val_mAP50": 0.52},
        }
        for _ in range(validation_count)
    ]
    if terminal:
        records.append({"message": "Train finished successfully."})
    text = "".join(json.dumps(record) + "\n" for record in records)

    def remote(command, **_kwargs):
        if " cat " in f" {command} ":
            return text
        return f"{len(text)}\n" + "c" * 64 + "  status.json\n"

    monkeypatch.setattr(run_campaign, "remote_output", remote)
    with pytest.raises(run_campaign.CampaignExecutionError, match=error):
        run_campaign._training_status_evidence(
            _SDK(),
            "job-id",
            expected_validation_records=10,
        )


def test_standalone_evaluation_requires_exact_status_and_terminal_success(
    monkeypatch,
):
    class _SDK:
        @staticmethod
        def get_job_results_dir(_job_id):
            return "/lustre/results/evaluate-job"

    records = (
        '{"message":"Test metrics generated.",'
        '"kpi":{"test_mAP":0.31,"test_mAP50":0.52}}\n'
        '{"message":"Evaluate finished successfully."}\n'
    )
    calls = []

    def remote(command, **_kwargs):
        calls.append(command)
        if " cat " in f" {command} ":
            return records
        return f"{len(records)}\n" + "d" * 64 + "  status.json\n"

    monkeypatch.setattr(run_campaign, "remote_output", remote)
    evidence = run_campaign._evaluation_status_evidence(_SDK(), "job-id")
    assert evidence["metrics"] == {
        "mAP": 0.31,
        "mAP50": 0.52,
    }
    assert evidence["status_evidence"]["path"] == (
        "/lustre/results/evaluate-job/results_dir/evaluate/status.json"
    )
    assert evidence["status_evidence"]["terminal_success"] is True
    assert all("find " not in command for command in calls)


def test_standalone_evaluation_rejects_missing_metric_or_terminal(monkeypatch):
    class _SDK:
        @staticmethod
        def get_job_results_dir(_job_id):
            return "/lustre/results/evaluate-job"

    records = (
        '{"message":"Test metrics generated.",'
        '"kpi":{"test_mAP50":0.52}}\n'
        '{"message":"Evaluate finished successfully."}\n'
    )

    def remote(command, **_kwargs):
        if " cat " in f" {command} ":
            return records
        return f"{len(records)}\n" + "e" * 64 + "  status.json\n"

    monkeypatch.setattr(run_campaign, "remote_output", remote)
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="finite test_mAP",
    ):
        run_campaign._evaluation_metrics(_SDK(), "job-id")

    no_terminal = (
        '{"message":"Test metrics generated.",'
        '"kpi":{"test_mAP":0.31,"test_mAP50":0.52}}\n'
    )

    def remote_without_terminal(command, **_kwargs):
        if " cat " in f" {command} ":
            return no_terminal
        return f"{len(no_terminal)}\n" + "f" * 64 + "  status.json\n"

    monkeypatch.setattr(
        run_campaign,
        "remote_output",
        remote_without_terminal,
    )
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="lacks terminal TAO evaluation success",
    ):
        run_campaign._evaluation_status_evidence(_SDK(), "job-id")


def test_dry_run_is_a_plan_only(manifest, capsys):
    assert run_campaign.main(["--manifest", str(MANIFEST_PATH)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["launch"] is False
    assert plan["logical_workflows"] == 2
    assert plan["parallel"] is True
    assert plan["cpu_runs"] == 0
    assert plan["smoke_runs"] == 0
