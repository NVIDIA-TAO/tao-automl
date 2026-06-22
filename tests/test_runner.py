# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the migrated AutoMLRunner.

Validates:
- SkillContext loads skill_info.yaml and spec_template_<action>.yaml.
- AutoMLRunner.__init__ takes (sdk, skill_dir, action) — no SkillBank.
- _make_sdk returns the right per-platform SDK class; rejects bad names.
- _run_one_job calls build_entrypoint with action_cfg fields and
  sdk.create_job with the resulting command + platform_kwargs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# SkillContext
# ---------------------------------------------------------------------------

def _write_fake_skill(tmp_path: Path, action: str = "train") -> Path:
    """Create a minimal skill bank layout for tests."""
    skill_dir = tmp_path / "models" / "fake-net"
    refs = skill_dir / "references"
    refs.mkdir(parents=True)
    (refs / "skill_info.yaml").write_text(
        "network_arch: fake-net\n"
        "container_image: nvcr.io/nvidia/tao/fake:0.1\n"
        "data_format: coco\n"
        "actions:\n"
        f"  {action}:\n"
        "    command: fake train -e {config_path}\n"
        "    config_format: yaml\n"
        "    inputs:\n"
        "      dataset.train_data_sources[0].image_dir:\n"
        "        type: file\n"
        "    outputs:\n"
        "      results_dir:\n"
        "        type: folder\n"
        "    upload_excludes: ['inputs/']\n"
    )
    (refs / f"spec_template_{action}.yaml").write_text(
        "train:\n"
        "  num_epochs: 12\n"
        "  optim:\n"
        "    lr: 2.0e-4\n"
        "dataset:\n"
        "  num_classes: 80\n"
    )
    return skill_dir


def test_skill_context_loads_skill_info_and_template(tmp_path):
    from tao_automl.runner import SkillContext
    skill_dir = _write_fake_skill(tmp_path)
    ctx = SkillContext(skill_dir=skill_dir, action="train")
    assert ctx.network_arch == "fake-net"
    assert ctx.action_cfg["command"] == "fake train -e {config_path}"
    assert ctx.action_cfg["config_format"] == "yaml"
    assert ctx.default_specs["train"]["num_epochs"] == 12
    assert ctx.default_specs["dataset"]["num_classes"] == 80


def test_skill_context_action_container_image_overrides_model_image(tmp_path):
    from tao_automl.runner import SkillContext
    skill_dir = _write_fake_skill(tmp_path, action="dataset_convert")
    info_path = skill_dir / "references/skill_info.yaml"
    info_path.write_text(
        info_path.read_text().replace(
            "    command: fake train -e {config_path}\n",
            "    container_image: nvcr.io/nvidia/tao/fake-ds:0.1\n"
            "    command: fake convert -e {config_path}\n",
        )
    )
    ctx = SkillContext(skill_dir=skill_dir, action="dataset_convert")
    assert ctx.container_image == "nvcr.io/nvidia/tao/fake-ds:0.1"


def test_skill_context_no_template_yields_empty_specs(tmp_path):
    """Models without a spec_template_<action>.yaml get default_specs={}.
    Caller is responsible for constructing the spec from skill SKILL.md."""
    from tao_automl.runner import SkillContext
    skill_dir = _write_fake_skill(tmp_path)
    (skill_dir / "references/spec_template_train.yaml").unlink()
    ctx = SkillContext(skill_dir=skill_dir, action="train")
    assert ctx.default_specs == {}


def test_skill_context_missing_action_raises(tmp_path):
    from tao_automl.runner import SkillContext
    skill_dir = _write_fake_skill(tmp_path, action="train")
    with pytest.raises(KeyError, match="evaluate"):
        SkillContext(skill_dir=skill_dir, action="evaluate")


def test_skill_context_missing_skill_info_raises(tmp_path):
    from tao_automl.runner import SkillContext
    with pytest.raises(FileNotFoundError, match="skill_info.yaml"):
        SkillContext(skill_dir=tmp_path / "nonexistent", action="train")


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def test_extract_metric_allows_val_prefix_for_sparse4d_map():
    from tao_automl.runner import _extract_metric_from_logs
    logs = "Calculating metrics...\nmAP: 0.0000\nNDS: 0.0000\nExecution status: PASS\n"
    assert _extract_metric_from_logs(logs, "val_mAP") == 0.0


def test_extract_metric_reads_sparse4d_status_kpi_alias(tmp_path):
    from tao_automl.runner import _extract_metric_from_status_file

    status_path = tmp_path / "status.json"
    status_path.write_text(
        '{"status": "RUNNING", "kpi": {"img_bbox_NuScenes/mAP": 0.125}}\n'
    )

    assert _extract_metric_from_status_file(status_path, "val_mAP") == 0.125


def test_execution_status_can_ignore_fatal_cleanup_patterns():
    from tao_automl.runner import _check_execution_status

    logs = "Saved best score to best_score.json\nRendezvousConnectionError\n"

    assert _check_execution_status(logs) == "FAIL"
    assert _check_execution_status(logs, include_fatal_patterns=False) is None


def test_execution_status_detects_nccl_watchdog_as_hard_failure():
    from tao_automl.runner import _check_execution_status, _has_hard_failure_pattern

    logs = (
        "Watchdog caught collective operation timeout: "
        "WorkNCCL(SeqNum=33271, OpType=ALLREDUCE)\n"
    )

    assert _check_execution_status(logs) == "FAIL"
    assert _has_hard_failure_pattern(logs)


def test_extract_metric_reads_cosmos_best_score_json(tmp_path):
    from tao_automl.runner import _extract_metric_from_local_results

    best_score = (
        tmp_path / "results" / "job-1" / "train_output_dir" / "best"
        / "best_score.json"
    )
    best_score.parent.mkdir(parents=True)
    best_score.write_text(
        '{"best_score": 0.8927091135965706, "metric": "val_loss"}\n'
    )

    metric = _extract_metric_from_local_results(
        "job-1",
        "val/avg_loss",
        {"mounts": [{"host_path": str(tmp_path / "results"),
                     "container_path": "/results"}]},
    )

    assert metric == pytest.approx(0.8927091135965706)


def test_llm_config_accepts_provider_aliases():
    from tao_automl.brain.llm_client import LLMConfig

    config = LLMConfig.from_params({
        "base_url": "https://inference-api.nvidia.com",
        "model": "gcp/google/gemini-3.1-pro-preview",
        "api_key": "secret",
    })

    assert config.endpoint == "https://inference-api.nvidia.com"
    assert config.model == "gcp/google/gemini-3.1-pro-preview"
    assert config.api_key == "secret"


def test_algorithm_params_pass_provider_aliases_to_llm_client():
    from tao_automl.brain.factory import AlgorithmParams

    params = AlgorithmParams.from_dict({
        "base_url": "https://inference-api.nvidia.com",
        "model": "gcp/google/gemini-3.1-pro-preview",
        "api_key": "secret",
    })

    assert params.get_llm_params() == {
        "llm_endpoint": "https://inference-api.nvidia.com",
        "llm_model": "gcp/google/gemini-3.1-pro-preview",
        "llm_api_key": "secret",
    }


def test_algorithm_params_parse_hybrid_range_narrowing_flag():
    from tao_automl.brain.factory import AlgorithmParams

    assert AlgorithmParams.from_dict({
        "hybrid_enable_llm_range_narrowing": "true",
    }).hybrid_enable_llm_range_narrowing
    assert not AlgorithmParams.from_dict({}).hybrid_enable_llm_range_narrowing


def test_promoted_metric_missing_checkpoint_carries_forward_prior_metric(
    tmp_path, monkeypatch
):
    from tao_automl.runner import AutoMLRunner
    from tao_automl.types import JobStates, Recommendation

    skill_dir = _write_fake_skill(tmp_path)
    results_root = tmp_path / "results"
    for job_id in ("parent-job", "child-job"):
        ckpt_dir = results_root / job_id / "results_dir" / "train"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "model_latest.pth").write_text("checkpoint")

    class FakeAutoML:
        def __init__(self, *args, **kwargs):
            self.rec = Recommendation(0, {"train.num_epochs": 2}, "val_mAP")
            self.rec.resume_from_job_id = "parent-job"
            self.rec.result = 0.42
            self.complete = False

        def is_complete(self):
            return self.complete

        def next_recommendation(self):
            return [self.rec]

        def report_result(self, rec_id, metric_value, best_epoch=None, status="success"):
            self.rec.update_result(metric_value)
            self.rec.update_status(status)
            self.complete = True

        def get_best(self):
            return self.rec if self.rec.status == JobStates.success else None

        def get_progress(self):
            return {"completed": int(self.complete), "best_metric": self.rec.result}

        def get_history(self):
            return [self.rec]

    def fake_run_one_job(self, *args, **kwargs):
        kwargs["rec"].assign_job_id("child-job")
        return None, "metric_missing"

    monkeypatch.setattr("tao_automl.AutoML", FakeAutoML)
    monkeypatch.setattr(AutoMLRunner, "_run_one_job", fake_run_one_job)

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    result = runner.run(
        image="nvcr.io/test:1",
        automl_settings={
            "algorithm": "dehb",
            "metric": "val_mAP",
            "direction": "maximize",
        },
        automl_hyperparameters=["train.optim.lr"],
        workspace_path=str(tmp_path / "workspace"),
        mounts=[{"host_path": str(results_root), "container_path": "/results"}],
    )

    assert result["best"]["metric_value"] == 0.42
    assert result["history"][0]["status"] == JobStates.success


def test_run_reports_baseline_metric_and_comparison(tmp_path, monkeypatch):
    from tao_automl.runner import AutoMLRunner
    from tao_automl.types import JobStates, Recommendation

    skill_dir = _write_fake_skill(tmp_path)

    class FakeAutoML:
        def __init__(self, *args, **kwargs):
            self.rec = Recommendation(0, {"train.num_epochs": 2}, "accuracy")
            self.complete = False

        def is_complete(self):
            return self.complete

        def next_recommendation(self):
            return [self.rec]

        def report_result(self, rec_id, metric_value, best_epoch=None, status="success"):
            self.rec.update_result(metric_value)
            self.rec.update_status(status)
            self.complete = True

        def get_best(self):
            return self.rec if self.rec.status == JobStates.success else None

        def get_progress(self):
            return {"completed": int(self.complete), "best_metric": self.rec.result}

        def get_history(self):
            return [self.rec]

    def fake_run_one_job(self, *args, **kwargs):
        return 0.62, "success"

    monkeypatch.setattr("tao_automl.AutoML", FakeAutoML)
    monkeypatch.setattr(AutoMLRunner, "_run_one_job", fake_run_one_job)

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    result = runner.run(
        image="nvcr.io/test:1",
        automl_settings={
            "algorithm": "bayesian",
            "metric": "accuracy",
            "direction": "maximize",
            "automl_max_recommendations": 1,
        },
        baseline_fn=lambda specs: 0.5,
        workspace_path=str(tmp_path / "workspace"),
    )

    assert result["baseline"]["status"] == "measured"
    assert result["baseline"]["metric_value"] == 0.5
    assert result["baseline"]["comparison_to_best"]["delta"] == pytest.approx(0.12)
    assert result["baseline"]["comparison_to_best"]["improved"] is True


def test_run_reports_runner_owned_final_evaluation(tmp_path, monkeypatch):
    from tao_automl.runner import AutoMLRunner
    from tao_automl.types import JobStates, Recommendation

    skill_dir = _write_fake_skill(tmp_path)
    final_eval_calls = []
    final_record_path = tmp_path / "workspace" / "evaluations" / "best_automl.json"

    class FakeAutoML:
        def __init__(self, *args, **kwargs):
            self.rec = Recommendation(0, {"train.num_epochs": 2}, "accuracy")
            self.complete = False

        def is_complete(self):
            return self.complete

        def next_recommendation(self):
            return [self.rec]

        def report_result(self, rec_id, metric_value, best_epoch=None, status="success"):
            self.rec.update_result(metric_value)
            self.rec.update_status(status)
            self.complete = True

        def get_best(self):
            return self.rec if self.rec.status == JobStates.success else None

        def get_progress(self):
            return {"completed": int(self.complete), "best_metric": self.rec.result}

        def get_history(self):
            return [self.rec]

    def fake_run_one_job(self, *args, **kwargs):
        kwargs["rec"].assign_job_id("train-job-0")
        return 0.62, "success"

    def final_eval_fn(best_rec, train_job_id):
        final_eval_calls.append((best_rec.id, train_job_id))
        return {"metric_value": 0.64, "record_path": str(final_record_path)}

    monkeypatch.setattr("tao_automl.AutoML", FakeAutoML)
    monkeypatch.setattr(AutoMLRunner, "_run_one_job", fake_run_one_job)

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    result = runner.run(
        image="nvcr.io/test:1",
        automl_settings={
            "algorithm": "bayesian",
            "metric": "accuracy",
            "direction": "maximize",
            "automl_max_recommendations": 1,
            "run_final_evaluation": True,
        },
        baseline_fn=lambda specs: 0.5,
        final_eval_fn=final_eval_fn,
        workspace_path=str(tmp_path / "workspace"),
    )

    assert final_eval_calls == [(0, "train-job-0")]
    assert result["best"]["metric_value"] == 0.62
    assert result["final_evaluation"]["status"] == "measured"
    assert result["final_evaluation"]["source"] == "final_eval_fn"
    assert result["final_evaluation"]["metric_value"] == 0.64
    assert result["final_evaluation"]["record_path"] == str(final_record_path)
    assert result["final_evaluation"]["comparison_to_baseline"]["delta"] == pytest.approx(0.14)
    assert result["final_evaluation"]["comparison_to_baseline"]["improved"] is True


def test_effective_batch_is_capped_before_launch(tmp_path, monkeypatch):
    from tao_automl.runner import AutoMLRunner
    from tao_automl.types import JobStates, Recommendation

    skill_dir = _write_fake_skill(tmp_path)
    captured_specs = {}

    class FakeAutoML:
        def __init__(self, *args, **kwargs):
            self.rec = Recommendation(
                0,
                {
                    "train.train_batch_per_replica": 8,
                    "policy.parallelism.dp_shard_size": 8,
                },
                "accuracy",
            )
            self.complete = False

        def is_complete(self):
            return self.complete

        def next_recommendation(self):
            return [self.rec]

        def report_result(self, rec_id, metric_value, best_epoch=None, status="success"):
            self.rec.update_result(metric_value)
            self.rec.update_status(status)
            self.complete = True

        def get_best(self):
            return self.rec if self.rec.status == JobStates.success else None

        def get_progress(self):
            return {"completed": int(self.complete), "best_metric": self.rec.result}

        def get_history(self):
            return [self.rec]

    def fake_run_one_job(self, *args, **kwargs):
        captured_specs.update(kwargs["specs"])
        return 0.6, "success"

    monkeypatch.setattr("tao_automl.AutoML", FakeAutoML)
    monkeypatch.setattr(AutoMLRunner, "_run_one_job", fake_run_one_job)

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    result = runner.run(
        image="nvcr.io/test:1",
        automl_settings={
            "algorithm": "bayesian",
            "metric": "accuracy",
            "direction": "maximize",
            "automl_max_recommendations": 1,
            "train_sample_count": 31,
        },
        workspace_path=str(tmp_path / "workspace"),
    )

    assert captured_specs["train"]["train_batch_per_replica"] == 3
    assert result["history"][0]["adjustments"][0]["type"] == "effective_batch_cap"
    assert result["history"][0]["adjustments"][0]["from"] == 8
    assert result["history"][0]["adjustments"][0]["to"] == 3


def test_effective_batch_reports_invalid_when_no_rank_has_samples():
    from tao_automl.runner import _maybe_cap_effective_batch
    from tao_automl.types import Recommendation

    rec = Recommendation(
        0,
        {
            "train.train_batch_per_replica": 8,
            "policy.parallelism.dp_shard_size": 8,
        },
        "loss",
    )
    specs = {
        "train": {"train_batch_per_replica": 8},
        "policy": {"parallelism": {"dp_shard_size": 8}},
    }

    reason = _maybe_cap_effective_batch(
        specs, rec, {"train_sample_count": 2}, {}
    )

    assert "invalid_configuration" in reason
    assert rec.failure_reason == reason
    assert specs["train"]["train_batch_per_replica"] == 8


def test_validate_skill_runtime_probes_schema_import_path(tmp_path):
    from tao_automl.runner import validate_skill_runtime

    skill_dir = _write_fake_skill(tmp_path)
    info_path = skill_dir / "references/skill_info.yaml"
    info_path.write_text(
        info_path.read_text().replace("network_arch: fake-net", "network_arch: cosmos-rl")
    )

    result = validate_skill_runtime(skill_dir, action="train")

    assert result["network_arch"] == "cosmos-rl"
    assert result["action"] == "train"
    assert result["parameter_count"] > 0


# ---------------------------------------------------------------------------
# _make_sdk — platform selection
# ---------------------------------------------------------------------------

def test_make_sdk_rejects_unknown_platform():
    from tao_automl.runner import _make_sdk
    with pytest.raises(ValueError, match="Unknown platform"):
        _make_sdk("aws-batch")


def test_make_sdk_lists_all_5_platforms_in_error():
    from tao_automl.runner import _make_sdk, _PLATFORMS
    assert set(_PLATFORMS) == {"lepton", "slurm", "kubernetes", "docker", "brev"}
    try:
        _make_sdk("nope")
    except ValueError as e:
        for p in _PLATFORMS:
            assert p in str(e)


# ---------------------------------------------------------------------------
# AutoMLRunner — submission path uses build_entrypoint + new create_job shape
# ---------------------------------------------------------------------------

def test_run_one_job_calls_build_entrypoint_with_action_cfg(tmp_path):
    """_run_one_job should pass the action's command/inputs/outputs/config_format/
    upload_excludes to build_entrypoint, and the resulting command string to
    sdk.create_job. No old kwargs (specs=, script_runner=, network_arch=)."""
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    fake_sdk = MagicMock()
    fake_sdk.create_job.return_value = MagicMock(id="job-xyz", backend_job_id="be-xyz")

    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    rec = MagicMock(id=1)

    fake_ep = {"command": "BAKED_HEREDOC_COMMAND", "args_template": ""}

    # Patch build_entrypoint at its import site (inside _run_one_job).
    with patch("tao_sdk.script_runner.build_entrypoint", return_value=fake_ep) as build:
        # Avoid the polling loop — make get_job_status return Complete immediately.
        fake_sdk.get_job_status.return_value = MagicMock(status="Complete")
        fake_sdk.get_job_logs.return_value = "loss: 0.5\n"
        runner._poll_interval = 0  # tight loop
        runner._run_one_job(
            image="nvcr.io/test:1",
            action_cfg=runner.skill_ctx.action_cfg,
            specs={"train": {"num_epochs": 10}},
            rec=rec, metric_name="loss",
            workspace_path=str(tmp_path),
            platform_kwargs={"gpu_count": 4, "num_nodes": 1},
        )

    # build_entrypoint received the schema fields from skill_info
    build.assert_called_once()
    kwargs = build.call_args.kwargs
    assert kwargs["command"] == "fake train -e {config_path}"
    assert kwargs["config_format"] == "yaml"
    assert "dataset.train_data_sources[0].image_dir" in kwargs["inputs"]
    assert "results_dir" in kwargs["outputs"]
    assert kwargs["upload_excludes"] == ["inputs/"]
    assert kwargs["specs"]["train"]["num_epochs"] == 10

    # create_job got the baked command + platform kwargs (no old shape)
    create_kwargs = fake_sdk.create_job.call_args.kwargs
    assert create_kwargs["command"] == "BAKED_HEREDOC_COMMAND"
    assert create_kwargs["image"] == "nvcr.io/test:1"
    assert create_kwargs["gpu_count"] == 4
    assert create_kwargs["num_nodes"] == 1
    # Old kwargs should NOT be present
    for legacy in ("specs", "script_runner", "network_arch", "data_format",
                   "backend_details", "workspace_id", "train_dataset_uri"):
        assert legacy not in create_kwargs, f"legacy kwarg {legacy!r} leaked"


def test_run_one_job_allows_completed_metric_with_cleanup_rendezvous(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    fake_sdk = MagicMock()
    fake_sdk.create_job.return_value = MagicMock(id="job-xyz", backend_job_id="be-xyz")
    fake_sdk.get_job_status.return_value = MagicMock(status="Complete")
    fake_sdk.get_job_logs.return_value = (
        "[cosmos] Validation rank 0: avg_loss=0.951012, samples=6\n"
        "Saved best score to best_score.json: 0.9510115849549504\n"
        "torch.distributed.elastic.rendezvous.api.RendezvousConnectionError\n"
        "torch.distributed.DistNetworkError: Failed to recv, got 0 bytes.\n"
    )

    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    runner._poll_interval = 0
    rec = MagicMock(id=7)

    with patch(
        "tao_sdk.script_runner.build_entrypoint",
        return_value={"command": "BAKED_HEREDOC_COMMAND", "args_template": ""},
    ):
        metric, status = runner._run_one_job(
            image="nvcr.io/test:1",
            action_cfg=runner.skill_ctx.action_cfg,
            specs={"train": {"num_epochs": 1}},
            rec=rec,
            metric_name="val/avg_loss",
            workspace_path=str(tmp_path),
            platform_kwargs={},
        )

    assert metric == pytest.approx(0.951012)
    assert status == "success"
    fake_sdk.cancel_job.assert_not_called()


def test_run_one_job_cancels_hard_failure_and_recovers_remote_best_score(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    fake_sdk = MagicMock()
    fake_sdk.create_job.return_value = MagicMock(id="job-hard", backend_job_id="be-hard")
    fake_sdk.get_job_logs.return_value = (
        "Watchdog caught collective operation timeout: "
        "WorkNCCL(SeqNum=33271, OpType=ALLREDUCE)\n"
    )
    fake_sdk.read_job_result_file.return_value = (
        '{"best_score": 0.8927091135965706, "metric": "val_loss"}\n'
    )

    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    runner._poll_interval = 0
    rec = MagicMock(id=8)

    with patch(
        "tao_sdk.script_runner.build_entrypoint",
        return_value={"command": "BAKED_HEREDOC_COMMAND", "args_template": ""},
    ):
        metric, status = runner._run_one_job(
            image="nvcr.io/test:1",
            action_cfg=runner.skill_ctx.action_cfg,
            specs={"train": {"num_epochs": 10}},
            rec=rec,
            metric_name="val/avg_loss",
            workspace_path=str(tmp_path),
            platform_kwargs={},
        )

    assert metric == pytest.approx(0.8927091135965706)
    assert status == "failure"
    fake_sdk.cancel_job.assert_called_once_with("job-hard")


def test_run_one_job_preserves_metric_when_slurm_reports_canceled(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    fake_sdk = MagicMock()
    fake_sdk.create_job.return_value = MagicMock(id="job-canceled", backend_job_id="be")
    fake_sdk.get_job_status.return_value = MagicMock(status="Canceled")
    fake_sdk.get_job_logs.return_value = (
        "[SFT] Validation loss: 0.751 for train step 8/10, epoch 4\n"
    )

    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    runner._poll_interval = 0
    rec = MagicMock(id=9)

    with patch(
        "tao_sdk.script_runner.build_entrypoint",
        return_value={"command": "BAKED_HEREDOC_COMMAND", "args_template": ""},
    ):
        metric, status = runner._run_one_job(
            image="nvcr.io/test:1",
            action_cfg=runner.skill_ctx.action_cfg,
            specs={"train": {"num_epochs": 10}},
            rec=rec,
            metric_name="val_loss",
            workspace_path=str(tmp_path),
            platform_kwargs={},
        )

    assert metric == pytest.approx(0.751)
    assert status == "failure"


def test_runner_init_replaces_skillbank_with_skillcontext(tmp_path):
    """AutoMLRunner.__init__ no longer takes (sdk, poll_interval) only —
    skill_dir + action are now required."""
    from tao_automl.runner import AutoMLRunner, SkillContext
    skill_dir = _write_fake_skill(tmp_path)
    fake_sdk = MagicMock()
    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    assert isinstance(runner.skill_ctx, SkillContext)
    assert runner.skill_ctx.network_arch == "fake-net"
    # Old API would've worked without skill_dir; new API is explicit.
    with pytest.raises(TypeError):
        AutoMLRunner(sdk=fake_sdk)  # missing skill_dir


# ---------------------------------------------------------------------------
# Override merging
# ---------------------------------------------------------------------------

def test_merge_specs_deep_merges_dotted_keys():
    from tao_automl.runner import AutoMLRunner
    base = {"train": {"num_epochs": 12, "optim": {"lr": 2.0e-4}}}
    overrides = {"train.optim.lr": 5.0e-5, "train.num_gpus": 8}
    merged = AutoMLRunner._merge_specs(base, overrides)
    assert merged["train"]["num_epochs"] == 12       # untouched
    assert merged["train"]["optim"]["lr"] == 5.0e-5  # overridden
    assert merged["train"]["num_gpus"] == 8           # added


def test_merge_specs_does_not_mutate_base():
    from tao_automl.runner import AutoMLRunner
    base = {"train": {"num_epochs": 12}}
    AutoMLRunner._merge_specs(base, {"train.num_epochs": 5})
    assert base["train"]["num_epochs"] == 12  # base stayed pristine


# ---------------------------------------------------------------------------
# Resume checkpoint handoff
# ---------------------------------------------------------------------------

def test_apply_resume_checkpoint_sets_training_checkpoint_path(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    template = skill_dir / "references/spec_template_train.yaml"
    template.write_text(
        "train:\n"
        "  num_epochs: 12\n"
        "  resume_training_checkpoint_path: ''\n"
        "dataset:\n"
        "  num_classes: 80\n"
    )

    results_root = tmp_path / "results"
    checkpoint = (
        results_root / "parent-job" / "results_dir" / "train" / "model_epoch_001.pth"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint")
    latest = checkpoint.parent / "classifier_model_latest.pth"
    latest.write_text("latest")

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    rec = MagicMock(id=2, resume_from_job_id="parent-job", resume_from_epoch=1)

    specs = {"train": {"resume_training_checkpoint_path": ""}}
    updated = runner._apply_resume_checkpoint(
        specs,
        rec,
        {"mounts": [{"host_path": str(results_root), "container_path": "/results"}]},
    )

    assert (
        updated["train"]["resume_training_checkpoint_path"]
        == "/results/parent-job/results_dir/train/model_epoch_001.pth"
    )
    assert rec.resume_checkpoint_path.endswith("model_epoch_001.pth")


def test_apply_resume_checkpoint_does_not_use_latest_for_requested_epoch(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    (skill_dir / "references/spec_template_train.yaml").write_text(
        "train:\n"
        "  resume_training_checkpoint_path: ''\n"
    )

    results_root = tmp_path / "results"
    checkpoint_dir = results_root / "parent-job" / "results_dir" / "train"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model_latest.pth").write_text("latest")

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    rec = MagicMock(id=2, resume_from_job_id="parent-job", resume_from_epoch=1)

    updated = runner._apply_resume_checkpoint(
        {"train": {"resume_training_checkpoint_path": ""}},
        rec,
        {"mounts": [{"host_path": str(results_root), "container_path": "/results"}]},
    )

    assert updated["train"]["resume_training_checkpoint_path"] == ""
    assert rec.resume_checkpoint_missing is True
    assert rec.resume_checkpoint_path is None


def test_apply_resume_checkpoint_sets_cosmos_resume_to_checkpoint_dir(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    template = skill_dir / "references/spec_template_train.yaml"
    template.write_text("train:\n  resume: false\n  epoch: 2\n")

    results_root = tmp_path / "results"
    checkpoint_dir = (
        results_root / "parent-job" / "train_output_dir" / "run1"
        / "checkpoints" / "epoch_1"
    )
    (checkpoint_dir / "policy").mkdir(parents=True)
    (checkpoint_dir / "policy" / "model_rank_0.pth").write_text("checkpoint")
    safetensor_dir = (
        results_root / "parent-job" / "train_output_dir" / "run1"
        / "safetensors" / "epoch_1"
    )
    safetensor_dir.mkdir(parents=True)
    (safetensor_dir / "adapter_model.safetensors").write_text("adapter")

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    rec = MagicMock(id=3, resume_from_job_id="parent-job", resume_from_epoch=1)

    updated = runner._apply_resume_checkpoint(
        {"train": {"resume": False, "epoch": 2}},
        rec,
        {"mounts": [{"host_path": str(results_root), "container_path": "/results"}]},
    )

    assert (
        updated["train"]["resume"]
        == "/results/parent-job/train_output_dir/run1/checkpoints/epoch_1"
    )


def test_apply_resume_environment_enables_trusted_checkpoint_resume(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    rec = MagicMock(
        id=4,
        resume_from_job_id="parent-job",
        resume_checkpoint_path="/results/parent-job/train/checkpoint.pth",
    )

    updated = runner._apply_resume_environment(
        {"env_vars": {"WANDB_MODE": "disabled"}},
        rec,
    )

    assert updated["env_vars"]["WANDB_MODE"] == "disabled"
    assert updated["env_vars"]["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"


def test_apply_resume_environment_does_not_mutate_non_resume_kwargs(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    rec = MagicMock(id=5, resume_from_job_id=None)
    platform_kwargs = {"env_vars": {"WANDB_MODE": "disabled"}}

    assert runner._apply_resume_environment(platform_kwargs, rec) is platform_kwargs


def test_apply_resume_environment_ignores_missing_checkpoint_path(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    rec = MagicMock(
        id=6,
        resume_from_job_id="parent-job",
        resume_checkpoint_path=None,
    )
    platform_kwargs = {"env_vars": {"WANDB_MODE": "disabled"}}

    assert runner._apply_resume_environment(platform_kwargs, rec) is platform_kwargs
