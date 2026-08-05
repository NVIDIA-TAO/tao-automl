# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner trust-boundary tests for live PTM-aware AutoML inventories."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tao_automl.objectives import parse_objective_config
from tao_automl.ptm_runtime import ResolvedPTMRuntimeInventory
from tao_automl.recommendation_audit import canonical_audit_sha256
from tao_automl.runner import AutoMLRunner, _persist_ptm_runtime_manifest
from tao_automl.types import Recommendation


def _write_skill(tmp_path: Path) -> Path:
    skill = tmp_path / "models" / "fake-net"
    references = skill / "references"
    references.mkdir(parents=True)
    (references / "skill_info.yaml").write_text(
        "network_arch: fake-net\n"
        "container_image: nvcr.io/nvidia/tao/fake:0.1\n"
        "data_sources:\n"
        "  train:\n"
        "    dataset.train_path:\n"
        "      source: train_datasets\n"
        "      path: images\n"
        "actions:\n"
        "  train:\n"
        "    command: fake train -e {config_path}\n"
        "    config_format: yaml\n"
        "    inputs: {}\n"
        "    outputs: {}\n",
        encoding="utf-8",
    )
    (references / "spec_template_train.yaml").write_text(
        "train:\n"
        "  num_epochs: 1\n"
        "model:\n"
        "  width: 1\n",
        encoding="utf-8",
    )
    return skill


def _settings(mode: str = "latency") -> dict:
    return {
        "algorithm": "bayesian",
        "objectives": [
            {"metric": "accuracy", "direction": "maximize"},
            {"metric": "latency", "direction": "minimize"},
        ],
        "selection_mode": mode,
        "latency_accuracy_retention": 0.90,
        "run_baseline": False,
        "run_final_evaluation": False,
        "automl_delete_intermediate_ckpt": False,
    }


def _inventory(
    monkeypatch,
    settings: dict,
    *,
    model: str = "fake-net",
    algorithm: str = "bayesian",
    mode: str | None = None,
    prepared_ids: tuple[str, ...] = ("ptm-a",),
    selected_ids: tuple[str, ...] | None = None,
    ptm_policy: str = "all",
    effective_spec: dict | None = None,
) -> ResolvedPTMRuntimeInventory:
    objective_config = parse_objective_config(settings)
    selected_ids = prepared_ids if selected_ids is None else selected_ids
    report = SimpleNamespace(
        prepared=tuple(
            SimpleNamespace(checkpoint_id=checkpoint_id)
            for checkpoint_id in prepared_ids
        ),
        report_sha256="report-sha",
        registry_sha256="registry-sha",
    )
    inventory = ResolvedPTMRuntimeInventory(
        report=report,
        algorithm=algorithm,
        mode=mode or settings["selection_mode"],
        model=model,
        task="object_detection",
        tao_version="7.0.1",
        ptm_policy=ptm_policy,
        user_checkpoint_id=None,
        objective_config_sha256=canonical_audit_sha256(
            objective_config.to_dict()
        ),
        base_layers_sha256={},
        arms=tuple(
            SimpleNamespace(
                checkpoint_id=checkpoint_id,
                checkpoint_target="train.pretrained_model_path",
                effective_base_spec=effective_spec or {
                    "train": {
                        "num_epochs": 1,
                        "pretrained_model_path": (
                            f"/prepared/{checkpoint_id}.pth"
                        ),
                    },
                    "model": {"width": 1},
                },
            )
            for checkpoint_id in selected_ids
        ),
        inventory_sha256="test-inventory-sha",
    )
    # The runner must invoke the live type's integrity method. Full inventory
    # construction and tamper tests live in test_ptm_runtime.py; this focused
    # seam uses a minimal typed object so no checkpoint or registry I/O occurs.
    validation_calls = []
    monkeypatch.setattr(
        ResolvedPTMRuntimeInventory,
        "validate",
        lambda self: validation_calls.append(self),
    )
    object.__setattr__(inventory, "_validation_calls", validation_calls)
    return inventory


class _NoLaunchSDK:
    def __init__(self):
        self.launches = 0

    def create_job(self, *args, **kwargs):
        self.launches += 1
        raise AssertionError("platform job must not launch")


class _CompleteAutoML:
    manifest = {
        "schema_version": 1,
        "stage": "built_hierarchical_ptm_runtime",
        "arms": [{"checkpoint_id": "ptm-a"}],
    }

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.ptm_runtime_manifest = dict(self.manifest)
        self.ptm_runtime_manifest_sha256 = canonical_audit_sha256(
            self.ptm_runtime_manifest
        )
        self.rec = Recommendation(0, {"model.width": 1}, "accuracy")
        self.rec.update_objectives(
            {"accuracy": 0.75, "latency": 10.0},
            0.0,
        )
        self.rec.update_status("success")

    def is_complete(self):
        return True

    def get_best(self):
        return self.rec

    def get_progress(self):
        return {"completed": 1, "best_metric": 0.75}

    def get_history(self):
        return [self.rec]

    def get_status(self):
        return {
            "recommendations": [],
            "pareto_front": [],
            "selection_analysis": {},
        }


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("missing", "requires a live typed"),
        ("serialized", "live ResolvedPTMRuntimeInventory"),
        ("model", "model .* does not match"),
        ("algorithm", "algorithm does not match"),
        ("mode", "mode does not match"),
        ("incomplete", "complete prepared checkpoint inventory"),
    ],
)
def test_ptm_runtime_rejects_invalid_inventory_before_baseline_or_launch(
    tmp_path, monkeypatch, case, expected
):
    settings = _settings()
    inventory = _inventory(monkeypatch, settings)
    if case == "missing":
        inventory = None
    elif case == "serialized":
        inventory = {"mode": "latency"}
    elif case == "model":
        inventory = replace(inventory, model="rtdetr")
    elif case == "algorithm":
        inventory = replace(inventory, algorithm="random")
    elif case == "mode":
        inventory = replace(inventory, mode="accuracy")
    elif case == "incomplete":
        inventory = _inventory(
            monkeypatch,
            settings,
            prepared_ids=("ptm-a", "ptm-b"),
            selected_ids=("ptm-a",),
        )

    sdk = _NoLaunchSDK()
    baseline_calls = []
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_skill(tmp_path))
    with pytest.raises((TypeError, ValueError), match=expected):
        runner.run(
            automl_settings=settings,
            ptm_aware_runtime=True,
            resolved_ptm_inventory=inventory,
            baseline_fn=lambda specs: baseline_calls.append(specs),
            workspace_path=str(tmp_path / "workspace"),
            resume=True,
        )

    assert baseline_calls == []
    assert sdk.launches == 0


def test_ptm_runtime_inventory_is_passed_unchanged_and_manifest_is_persisted(
    tmp_path, monkeypatch
):
    settings = _settings()
    inventory = _inventory(monkeypatch, settings)
    constructed = []

    class CapturingAutoML(_CompleteAutoML):
        def __init__(self, *args, **kwargs):
            constructed.append(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("tao_automl.AutoML", CapturingAutoML)
    sdk = _NoLaunchSDK()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    expected_hash = canonical_audit_sha256(_CompleteAutoML.manifest)
    (workspace / "ptm_runtime_manifest.json").write_text(
        json.dumps({
            "manifest": _CompleteAutoML.manifest,
            "manifest_sha256": expected_hash,
        }),
        encoding="utf-8",
    )
    result = AutoMLRunner(
        sdk=sdk,
        skill_dir=_write_skill(tmp_path),
    ).run(
        automl_settings=settings,
        ptm_aware_runtime=True,
        resolved_ptm_inventory=inventory,
        workspace_path=str(workspace),
        resume=True,
    )

    assert len(constructed) == 1
    assert constructed[0]["resolved_ptm_inventory"] is inventory
    assert inventory._validation_calls == [inventory]
    assert sdk.launches == 0
    assert result["ptm_runtime"]["manifest"] == _CompleteAutoML.manifest
    assert result["ptm_runtime"]["manifest_sha256"] == expected_hash
    record = json.loads(
        (workspace / "ptm_runtime_manifest.json").read_text(encoding="utf-8")
    )
    assert record == {
        "manifest": _CompleteAutoML.manifest,
        "manifest_sha256": expected_hash,
    }


def test_ptm_runtime_inventory_argument_enables_runtime_unless_explicitly_disabled(
    tmp_path, monkeypatch
):
    settings = _settings()
    inventory = _inventory(monkeypatch, settings)
    monkeypatch.setattr("tao_automl.AutoML", _CompleteAutoML)

    result = AutoMLRunner(
        sdk=_NoLaunchSDK(),
        skill_dir=_write_skill(tmp_path),
    ).run(
        automl_settings=settings,
        resolved_ptm_inventory=inventory,
        workspace_path=str(tmp_path / "workspace"),
    )
    assert result["ptm_runtime"]["manifest_sha256"]

    with pytest.raises(ValueError, match="while ptm_aware_runtime is false"):
        AutoMLRunner(
            sdk=_NoLaunchSDK(),
            skill_dir=_write_skill(tmp_path / "disabled"),
        ).run(
            automl_settings=settings,
            ptm_aware_runtime=False,
            resolved_ptm_inventory=inventory,
            workspace_path=str(tmp_path / "disabled-workspace"),
            resume=True,
        )


def test_ptm_runtime_manifest_with_secret_is_rejected_before_baseline(
    tmp_path, monkeypatch
):
    settings = _settings()
    inventory = _inventory(monkeypatch, settings)
    baseline_calls = []

    class SecretManifestAutoML(_CompleteAutoML):
        manifest = {"configuration": {"api_key": "must-not-be-written"}}

    monkeypatch.setattr("tao_automl.AutoML", SecretManifestAutoML)
    (tmp_path / "workspace").mkdir()
    with pytest.raises(ValueError, match="secret-bearing field"):
        AutoMLRunner(
            sdk=_NoLaunchSDK(),
            skill_dir=_write_skill(tmp_path),
        ).run(
            automl_settings=settings,
            ptm_aware_runtime=True,
            resolved_ptm_inventory=inventory,
            baseline_fn=lambda specs: baseline_calls.append(specs),
            workspace_path=str(tmp_path / "workspace"),
            resume=True,
        )
    assert baseline_calls == []


@pytest.mark.parametrize(
    ("runner_kwargs", "expected"),
    [
        (
            {"base_checkpoint": "/other/checkpoint.pth"},
            "exclusively owns checkpoint identity",
        ),
        (
            {"base_checkpoint_target": "train.pretrained_model_path"},
            "exclusively owns checkpoint identity",
        ),
        (
            {"spec_overrides": {"model.width": 2}},
            "was not frozen into PTM arm",
        ),
        (
            {"train_dataset_uri": "s3://dataset/v1"},
            "Runner dataset binding .* was not frozen",
        ),
        (
            {
                "spec_overrides": {
                    "train.pretrained_model_path": "/other/checkpoint.pth"
                }
            },
            "exclusively owns checkpoint target",
        ),
    ],
)
def test_ptm_runtime_rejects_unfrozen_runner_inputs_before_baseline(
    tmp_path, monkeypatch, runner_kwargs, expected
):
    settings = _settings()
    inventory = _inventory(monkeypatch, settings)
    baseline_calls = []
    sdk = _NoLaunchSDK()
    (tmp_path / "workspace").mkdir()
    with pytest.raises(ValueError, match=expected):
        AutoMLRunner(
            sdk=sdk,
            skill_dir=_write_skill(tmp_path),
        ).run(
            automl_settings=settings,
            ptm_aware_runtime=True,
            resolved_ptm_inventory=inventory,
            baseline_fn=lambda specs: baseline_calls.append(specs),
            workspace_path=str(tmp_path / "workspace"),
            resume=True,
            **runner_kwargs,
        )
    assert baseline_calls == []
    assert sdk.launches == 0


def test_ptm_runtime_accepts_runner_values_frozen_into_every_arm(
    tmp_path, monkeypatch
):
    settings = _settings()
    effective_spec = {
        "train": {
            "num_epochs": 1,
            "pretrained_model_path": "/prepared/ptm-a.pth",
        },
        "model": {"width": 2},
        "dataset": {"train_path": "s3://dataset/v1/images"},
    }
    inventory = _inventory(
        monkeypatch,
        settings,
        effective_spec=effective_spec,
    )
    monkeypatch.setattr("tao_automl.AutoML", _CompleteAutoML)
    result = AutoMLRunner(
        sdk=_NoLaunchSDK(),
        skill_dir=_write_skill(tmp_path),
    ).run(
        automl_settings=settings,
        ptm_aware_runtime=True,
        resolved_ptm_inventory=inventory,
        spec_overrides={"model.width": 2},
        train_dataset_uri="s3://dataset/v1",
        workspace_path=str(tmp_path / "workspace"),
    )
    assert result["ptm_runtime"]["manifest_sha256"]


def test_ptm_runtime_manifest_resume_and_fresh_write_are_drift_safe(tmp_path):
    automl = _CompleteAutoML()
    workspace = tmp_path / "workspace"

    with pytest.raises(RuntimeError, match="resume requires"):
        _persist_ptm_runtime_manifest(
            automl=automl,
            workspace_path=str(workspace),
            resume=True,
        )

    _persist_ptm_runtime_manifest(
        automl=automl,
        workspace_path=str(workspace),
        resume=False,
    )
    record_path = workspace / "ptm_runtime_manifest.json"
    original = record_path.read_bytes()
    _persist_ptm_runtime_manifest(
        automl=automl,
        workspace_path=str(workspace),
        resume=True,
    )
    assert record_path.read_bytes() == original

    record_path.write_text(
        json.dumps({"manifest": {"stage": "other"}, "manifest_sha256": "bad"}),
        encoding="utf-8",
    )
    conflicting = record_path.read_bytes()
    with pytest.raises(RuntimeError, match="conflicts"):
        _persist_ptm_runtime_manifest(
            automl=automl,
            workspace_path=str(workspace),
            resume=False,
        )
    assert record_path.read_bytes() == conflicting


def test_legacy_runner_does_not_pass_ptm_keyword_or_emit_manifest(
    tmp_path, monkeypatch
):
    constructed = []

    class LegacyAutoML(_CompleteAutoML):
        def __init__(self, *args, **kwargs):
            assert "resolved_ptm_inventory" not in kwargs
            constructed.append(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("tao_automl.AutoML", LegacyAutoML)
    (tmp_path / "workspace").mkdir()
    result = AutoMLRunner(
        sdk=_NoLaunchSDK(),
        skill_dir=_write_skill(tmp_path),
    ).run(
        automl_settings=_settings(),
        workspace_path=str(tmp_path / "workspace"),
        resume=True,
    )
    assert len(constructed) == 1
    assert "ptm_runtime" not in result
    assert not (tmp_path / "workspace" / "ptm_runtime_manifest.json").exists()
