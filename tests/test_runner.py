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
