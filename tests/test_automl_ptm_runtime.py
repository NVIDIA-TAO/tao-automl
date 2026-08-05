# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused AutoML wiring tests for already-resolved PTM runtime inventories."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tao_automl import AutoML, query_status
from tao_automl.brain.factory import BrainFactory
from tao_automl.objectives import parse_objective_config
from tao_automl.ptm_runtime import (
    ResolvedPTMRuntimeArm,
    ResolvedPTMRuntimeInventory,
)
from tao_automl.recommendation_audit import canonical_audit_sha256


def _settings(mode="multi_objective", algorithm="bayesian"):
    return {
        "algorithm": algorithm,
        "session_id": "automl-ptm-wiring",
        "random_seed": 271828,
        "objectives": [
            {"metric": "mAP50", "direction": "maximize"},
            {"metric": "latency_ms", "direction": "minimize"},
        ],
        "selection_mode": mode,
        "accuracy_metric": "mAP50",
        "latency_metric": "latency_ms",
        "latency_accuracy_retention": 0.9,
    }


def _arm(checkpoint_id, marker, spec):
    return ResolvedPTMRuntimeArm(
        checkpoint_id=checkpoint_id,
        checkpoint_target="train.pretrained_model_path",
        checkpoint_path=f"/verified/{checkpoint_id}.pth",
        effective_base_spec=spec,
        report_sha256="1" * 64,
        registry_sha256="2" * 64,
        registry_record_sha256=marker * 64,
        preflight_provenance_sha256=chr(ord(marker) + 1) * 64,
        checkpoint_artifact_sha256=chr(ord(marker) + 2) * 64,
        checkpoint_spec_artifact_sha256=chr(ord(marker) + 3) * 64,
        checkpoint_spec_document_sha256=chr(ord(marker) + 4) * 64,
        input_contract_sha256=chr(ord(marker) + 5) * 64,
        ptm_layer_sha256=chr(ord(marker) + 6) * 64,
        effective_base_spec_sha256=canonical_audit_sha256(spec),
    )


def _inventory(settings, *, model="dino", arms=None):
    config = parse_objective_config(settings)
    arm_values = tuple(
        arms
        if arms is not None
        else (
            _arm(
                "dino.a",
                "1",
                {
                    "model": {"arm": "a", "width": "small"},
                    "train": {"pretrained_model_path": "/verified/dino.a.pth"},
                },
            ),
            _arm(
                "dino.b",
                "8",
                {
                    "model": {"arm": "b", "depth": 3},
                    "train": {"pretrained_model_path": "/verified/dino.b.pth"},
                },
            ),
        )
    )
    report = SimpleNamespace(
        report_sha256="1" * 64,
        registry_sha256="2" * 64,
    )
    provisional = ResolvedPTMRuntimeInventory(
        report=report,
        algorithm="bayesian",
        mode=config.selection_config.mode,
        model=model,
        task="object_detection",
        tao_version="7.1.0",
        ptm_policy="all" if len(arm_values) != 1 else "default",
        user_checkpoint_id=None,
        objective_config_sha256=canonical_audit_sha256(config.to_dict()),
        base_layers_sha256={
            "model_defaults": "3" * 64,
            "automl_profile_overrides": "4" * 64,
            "user_overrides": "5" * 64,
        },
        arms=arm_values,
        inventory_sha256="",
    )
    return ResolvedPTMRuntimeInventory(
        **{
            **provisional.__dict__,
            "inventory_sha256": canonical_audit_sha256(
                provisional.stable_dict()
            ),
        }
    )


def _parameter(name, *, options=None):
    return {
        "parameter": name,
        "value_type": "categorical" if options else "float",
        "default_value": options[0] if options else 0.5,
        "valid_min": 0.0,
        "valid_max": 1.0,
        "valid_options": options or [],
        "option_weights": None,
        "math_cond": None,
        "parent_param": None,
        "depends_on": None,
    }


def _conditional_generator(**kwargs):
    arm = kwargs["train_specs"]["model"]["arm"]
    if arm == "a":
        return (
            [_parameter("model.width", options=["small", "large"])],
            ["model.width"],
        )
    return ([_parameter("model.depth")], ["model.depth"])


def _patch_runtime(monkeypatch):
    import tao_automl.ptm_runtime as runtime_module
    import tao_automl.search_space.params as params_module

    captured = {}
    brain = SimpleNamespace(name="hierarchical-ptm")

    def build(**kwargs):
        captured.update(kwargs)
        manifest = {
            "schema_version": 1,
            "arms": sorted(kwargs["conditional_parameters"]),
        }
        return SimpleNamespace(
            brain=brain,
            manifest=manifest,
            manifest_sha256=canonical_audit_sha256(manifest),
        )

    monkeypatch.setattr(
        params_module,
        "generate_hyperparams_to_search",
        _conditional_generator,
    )
    monkeypatch.setattr(
        runtime_module,
        "build_hierarchical_ptm_runtime",
        build,
    )
    return captured, brain


def test_resolved_inventory_builds_conditional_wrapper_and_union(
    tmp_path,
    monkeypatch,
):
    settings = _settings()
    inventory = _inventory(settings)
    captured, brain = _patch_runtime(monkeypatch)
    direct_factory = MagicMock()
    monkeypatch.setattr(BrainFactory, "create_brain", direct_factory)

    automl = AutoML(
        workspace=str(tmp_path),
        network="dino",
        train_specs={"model": {"arm": "global"}},
        settings=settings,
        automl_hyperparameters=["model.width", "model.depth"],
        custom_param_ranges={
            "model.width": {
                "valid_options": ["large", "invalid"],
            },
            "model.depth": {
                "valid_min": 2.0,
                "valid_max": 5.0,
            },
        },
        resolved_ptm_inventory=inventory,
    )

    direct_factory.assert_not_called()
    assert automl._controller.brain is brain
    assert automl._controller.parameter_names == [
        "model.depth",
        "model.width",
    ]
    assert set(captured["conditional_parameters"]) == {"dino.a", "dino.b"}
    assert captured["conditional_parameters"]["dino.a"][0]["parameter"] == (
        "model.width"
    )
    assert captured["conditional_parameters"]["dino.b"][0]["parameter"] == (
        "model.depth"
    )
    assert captured["conditional_ranges"] == {
        "dino.a": {
            "model.width": {
                "valid_options": ["large"],
            }
        },
        "dino.b": {
            "model.depth": {
                "valid_min": 2.0,
                "valid_max": 5.0,
            }
        },
    }
    assert captured["random_seed"] == 271828
    assert captured["resume"] is False
    assert captured["resolved_inventory"] is inventory

    manifest = automl.ptm_runtime_manifest
    assert manifest == {"schema_version": 1, "arms": ["dino.a", "dino.b"]}
    assert automl.ptm_runtime_manifest_sha256 == canonical_audit_sha256(
        manifest
    )
    manifest["arms"].append("tampered")
    assert automl.ptm_runtime_manifest["arms"] == ["dino.a", "dino.b"]
    with pytest.raises(AttributeError):
        automl.ptm_runtime_manifest = {}


def test_resolved_inventory_accepts_bayesian_alias_and_passes_canonical_name(
    tmp_path,
    monkeypatch,
):
    settings = _settings(algorithm="b")
    inventory = _inventory(settings)
    captured, brain = _patch_runtime(monkeypatch)

    automl = AutoML(
        workspace=str(tmp_path),
        network="dino",
        train_specs={"model": {"arm": "global"}},
        settings=settings,
        resolved_ptm_inventory=inventory,
    )

    assert automl._controller.brain is brain
    assert captured["algorithm"] == "bayesian"


def test_single_arm_accuracy_still_uses_hierarchical_wrapper(
    tmp_path,
    monkeypatch,
):
    settings = _settings("accuracy")
    inventory = _inventory(
        settings,
        arms=(
            _arm(
                "dino.a",
                "1",
                {
                    "model": {"arm": "a", "width": "small"},
                    "train": {"pretrained_model_path": "/verified/dino.a.pth"},
                },
            ),
        ),
    )
    captured, brain = _patch_runtime(monkeypatch)
    direct_factory = MagicMock()
    monkeypatch.setattr(BrainFactory, "create_brain", direct_factory)

    automl = AutoML(
        workspace=str(tmp_path),
        network="dino",
        train_specs={"model": {"arm": "global"}},
        settings=settings,
        resolved_ptm_inventory=inventory,
    )

    direct_factory.assert_not_called()
    assert automl._controller.brain is brain
    assert tuple(captured["conditional_parameters"]) == ("dino.a",)
    assert automl.ptm_runtime_manifest["arms"] == ["dino.a"]


def test_absent_inventory_preserves_direct_brain_factory_path(
    tmp_path,
    monkeypatch,
):
    import tao_automl.search_space.params as params_module

    parameter = _parameter("model.width")
    monkeypatch.setattr(
        params_module,
        "generate_hyperparams_to_search",
        lambda **_kwargs: ([parameter], ["model.width"]),
    )
    direct_brain = SimpleNamespace(name="direct")
    direct_factory = MagicMock(return_value=direct_brain)
    monkeypatch.setattr(BrainFactory, "create_brain", direct_factory)

    automl = AutoML(
        workspace=str(tmp_path),
        network="dino",
        train_specs={"model": {"width": 0.5}},
        settings=_settings("accuracy"),
    )

    direct_factory.assert_called_once()
    assert automl._controller.brain is direct_brain
    assert automl.ptm_runtime_manifest is None
    assert automl.ptm_runtime_manifest_sha256 is None


@pytest.mark.parametrize(
    ("settings", "inventory_factory", "message"),
    [
        (
            _settings(algorithm="random"),
            lambda settings: _inventory(settings),
            "algorithm='bayesian'",
        ),
        (
            _settings(),
            lambda settings: _inventory(settings, model="rtdetr"),
            "does not match AutoML network",
        ),
        (
            _settings("latency"),
            lambda _settings_value: _inventory(_settings("accuracy")),
            "objective configuration does not match",
        ),
        (
            _settings(),
            lambda settings: _inventory(settings, arms=()),
            "no conditional arms",
        ),
    ],
)
def test_resolved_inventory_mismatches_fail_closed(
    tmp_path,
    monkeypatch,
    settings,
    inventory_factory,
    message,
):
    inventory = inventory_factory(settings)
    _patch_runtime(monkeypatch)

    with pytest.raises(ValueError, match=message):
        AutoML(
            workspace=str(tmp_path),
            network="dino",
            train_specs={"model": {"arm": "global"}},
            settings=settings,
            resolved_ptm_inventory=inventory,
        )


def test_empty_conditional_arm_fails_before_runtime_build(
    tmp_path,
    monkeypatch,
):
    import tao_automl.ptm_runtime as runtime_module
    import tao_automl.search_space.params as params_module

    settings = _settings()
    inventory = _inventory(settings)
    build = MagicMock()
    monkeypatch.setattr(
        runtime_module,
        "build_hierarchical_ptm_runtime",
        build,
    )
    monkeypatch.setattr(
        params_module,
        "generate_hyperparams_to_search",
        lambda **kwargs: (
            ([], [])
            if kwargs["train_specs"]["model"]["arm"] == "b"
            else (
                [_parameter("model.width")],
                ["model.width"],
            )
        ),
    )

    with pytest.raises(ValueError, match="dino.b.*no searchable"):
        AutoML(
            workspace=str(tmp_path),
            network="dino",
            train_specs={"model": {"arm": "global"}},
            settings=settings,
            resolved_ptm_inventory=inventory,
        )
    build.assert_not_called()


def test_custom_range_absent_from_every_conditional_arm_fails_closed(
    tmp_path,
    monkeypatch,
):
    import tao_automl.ptm_runtime as runtime_module

    settings = _settings()
    inventory = _inventory(settings)
    _patch_runtime(monkeypatch)
    build = MagicMock()
    monkeypatch.setattr(
        runtime_module,
        "build_hierarchical_ptm_runtime",
        build,
    )

    with pytest.raises(ValueError, match="absent from every resolved PTM"):
        AutoML(
            workspace=str(tmp_path),
            network="dino",
            train_specs={"model": {"arm": "global"}},
            settings=settings,
            custom_param_ranges={
                "model.unknown": {
                    "valid_min": 0.0,
                    "valid_max": 1.0,
                }
            },
            resolved_ptm_inventory=inventory,
        )
    build.assert_not_called()


def test_resolved_inventory_is_revalidated_before_controller_creation(
    tmp_path,
    monkeypatch,
):
    from tao_automl.controller.controller import Controller

    settings = _settings()
    inventory = _inventory(settings)
    inventory.arms[0].effective_base_spec["model"]["width"] = "tampered"
    _patch_runtime(monkeypatch)
    controller_init = MagicMock(
        side_effect=AssertionError("Controller must not be constructed")
    )
    monkeypatch.setattr(Controller, "__init__", controller_init)

    with pytest.raises(ValueError, match="inventory integrity"):
        AutoML(
            workspace=str(tmp_path),
            network="dino",
            train_specs={"model": {"arm": "global"}},
            settings=settings,
            resolved_ptm_inventory=inventory,
        )
    controller_init.assert_not_called()


def test_resume_routes_built_wrapper_through_controller_load_state(
    tmp_path,
    monkeypatch,
):
    from tao_automl.controller.controller import Controller

    settings = _settings("accuracy")
    inventory = _inventory(
        settings,
        arms=(
            _arm(
                "dino.a",
                "1",
                {
                    "model": {"arm": "a", "width": "small"},
                    "train": {"pretrained_model_path": "/verified/dino.a.pth"},
                },
            ),
        ),
    )
    captured, brain = _patch_runtime(monkeypatch)
    restored_controller = SimpleNamespace(brain=brain)
    load_state = MagicMock(return_value=restored_controller)
    monkeypatch.setattr(Controller, "load_state", load_state)
    from tao_automl.state.state_store import StateStore

    StateStore(str(tmp_path)).save_job_specs(
        settings["session_id"],
        {"model": {"arm": "global"}},
    )

    automl = AutoML(
        workspace=str(tmp_path),
        network="dino",
        train_specs={"model": {"arm": "global"}},
        settings=settings,
        resolved_ptm_inventory=inventory,
        resume=True,
    )

    assert captured["resume"] is True
    load_state.assert_called_once()
    assert load_state.call_args.kwargs["brain"] is brain
    assert automl._controller is restored_controller


def test_query_status_verifies_and_exposes_ptm_runtime_manifest(tmp_path):
    workspace = tmp_path / "workspace"
    controller_dir = workspace / ".automl" / "controller"
    controller_dir.mkdir(parents=True)
    (controller_dir / "runtime-session.json").write_text(
        "[]",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "stage": "built_hierarchical_ptm_runtime",
        "arms": [{"checkpoint_id": "dino.a"}],
    }
    record = {
        "manifest": manifest,
        "manifest_sha256": canonical_audit_sha256(manifest),
    }
    manifest_path = workspace / "ptm_runtime_manifest.json"
    manifest_path.write_text(
        json.dumps(record),
        encoding="utf-8",
    )

    status = query_status(str(workspace))
    assert status["ptm_runtime"] == record
    status["ptm_runtime"]["manifest"]["arms"].append(
        {"checkpoint_id": "tampered"}
    )
    assert query_status(str(workspace))["ptm_runtime"] == record

    record["manifest_sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(record),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="integrity verification"):
        query_status(str(workspace))
