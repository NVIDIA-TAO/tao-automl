"""Contract tests for the qualification-driven Grounding DINO pilot."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tao_automl.ptm_registry import canonical_sha256

from .contract import AGENT_FLAGS, MODES, PreparationError, read_json
from .pilot_campaign import (
    AutomaticFirstCandidateGate,
    PilotExecutionError,
    build_evaluation_spec,
    custom_ranges,
    frozen_intent_signature,
    latency_worker_command,
    latency_worker_contract,
    mode_settings,
    nested_spec_overrides,
    spec_overrides,
    wait_for_launch_authorization,
)
from . import pilot_campaign
from .pilot_manifest import (
    SEARCH_PARAMETERS,
    build_manifest,
    seal_manifest,
    validate_manifest,
)
from .pilot_qualification import (
    PilotQualificationError,
    audit_pilot_qualification,
    evidence_load_callback,
)


HERE = Path(__file__).resolve().parent


class _SupportedRegistry:
    document_sha256 = "f" * 64

    def checkpoint(self, checkpoint_id):
        return {
            "id": checkpoint_id,
            "model_family": "grounding_dino",
            "status": "supported",
        }


class _UnverifiedRegistry:
    document_sha256 = "e" * 64

    def checkpoint(self, checkpoint_id):
        return {
            "id": checkpoint_id,
            "model_family": "grounding_dino",
            "status": "unverified",
        }


def _workflow(
    contract,
    expected,
    *,
    success=True,
    standalone_mAP50_delta=0.0,
):
    flags = {name: False for name in AGENT_FLAGS}
    checkpoint = {
        "path": (
            f"/lustre/results/{expected['workflow_id']}/"
            "model_epoch_009_step_00420.pth"
        ),
        "sha256": ("a" if expected["workflow_id"].endswith("0") else "b")
        * 64,
        "size_bytes": 2048,
        "training_epochs": 10,
        "terminal_epoch_index": 9,
    }
    if not success:
        return {
            "schema_version": 1,
            "campaign_id": contract["campaign_id"],
            "contract_sha256": contract["contract_sha256"],
            "workflow_id": expected["workflow_id"],
            "ptm_id": expected["ptm_id"],
            "ptm_sha256": expected["staged_checkpoint"]["sha256"],
            "status": "terminal_failure",
            "terminal": True,
            "jobs": {},
            "failure_preserved": True,
            "failure": {
                "type": "CampaignExecutionError",
                "message": "frozen failure",
                "replacement_submitted": False,
            },
            "agent_intervention_flags": flags,
        }
    validation = [
        {"mAP": 0.1 + index / 1000, "mAP50": 0.2 + index / 1000}
        for index in range(10)
    ]
    standalone = {
        "mAP": validation[-1]["mAP"],
        "mAP50": validation[-1]["mAP50"] + standalone_mAP50_delta,
    }
    return {
        "schema_version": 1,
        "campaign_id": contract["campaign_id"],
        "contract_sha256": contract["contract_sha256"],
        "workflow_id": expected["workflow_id"],
        "ptm_id": expected["ptm_id"],
        "ptm_sha256": expected["staged_checkpoint"]["sha256"],
        "status": "success",
        "terminal": True,
        "jobs": {
            "train": {
                "status": "Complete",
                "nodes": 1,
                "gpus": 8,
                "training_epochs": 10,
                "spec_sha256": expected["train"]["spec_sha256"],
                "status_evidence": {
                    "terminal_success": True,
                    "validation_record_count": 10,
                    "validation_metrics": validation,
                },
                "terminal_checkpoint": checkpoint,
            },
            "evaluate": {
                "status": "Complete",
                "nodes": 1,
                "gpus": 8,
                "checkpoint": checkpoint,
                "status_evidence": {
                    "terminal_success": True,
                    "metrics": standalone,
                },
            },
        },
        "metrics": {
            "training_validation": validation,
            "standalone": standalone,
        },
        "failure_preserved": False,
        "agent_intervention_flags": flags,
    }


def _write_qualification(
    tmp_path,
    *,
    failed_indices=(),
    standalone_mAP50_delta=0.0,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    contract_path = HERE / "successor.runtime.contract.v2.json"
    contract = read_json(contract_path)
    workflows = [
        _workflow(
            contract,
            expected,
            success=index not in failed_indices,
            standalone_mAP50_delta=standalone_mAP50_delta,
        )
        for index, expected in enumerate(contract["qualification"]["jobs"])
    ]
    successful = sum(item["status"] == "success" for item in workflows)
    completion = {
        "schema_version": 1,
        "campaign_id": contract["campaign_id"],
        "contract_sha256": contract["contract_sha256"],
        "model": "grounding_dino",
        "terminal": True,
        "status": (
            "success"
            if successful == len(workflows)
            else "terminal_with_failures"
        ),
        "successful_workflows": successful,
        "failed_workflows": len(workflows) - successful,
        "minimum_supported_ptms_for_pilot": 1,
        "pilot_handoff_ready": successful >= 1,
        "failures_preserved": True,
        "replacement_workflows_submitted": False,
        "workflows": workflows,
        "agent_intervention_flags": {
            name: False for name in AGENT_FLAGS
        },
    }
    completion["completion_sha256"] = canonical_sha256(completion)
    completion_path = tmp_path / "qualification_completion.json"
    completion_path.write_text(
        __import__("json").dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    handoff = {
        "schema_version": 1,
        "campaign_id": contract["campaign_id"],
        "contract_sha256": contract["contract_sha256"],
        "qualification_completion_sha256": completion[
            "completion_sha256"
        ],
        "automatic": True,
        "manual_confirmation_required": False,
        "pilot_modes": list(MODES),
        "status": "ready_for_algorithm_generated_mode_pilots",
        "selection_or_recommendation_performed": False,
        "agent_intervention_flags": {
            name: False for name in AGENT_FLAGS
        },
    }
    handoff["handoff_sha256"] = canonical_sha256(handoff)
    (tmp_path / "pilot_handoff.json").write_text(
        __import__("json").dumps(handoff, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inputs = read_json(HERE / "pilot.inputs.v1.json")
    inputs["qualification"] = {
        "contract_file": str(contract_path),
        "expected_contract_sha256": contract["contract_sha256"],
        "runtime_root": str(tmp_path),
    }
    return inputs, contract


def _source_state(inputs):
    return {
        "repository": inputs["source"]["repository"],
        "commit": inputs["source"]["minimum_ancestor_commit"],
        "minimum_ancestor_commit": inputs["source"][
            "minimum_ancestor_commit"
        ],
        "minimum_ancestor_satisfied": True,
        "clean": True,
    }


def _ready_manifest(tmp_path, *, failed_indices=()):
    inputs, _ = _write_qualification(
        tmp_path,
        failed_indices=failed_indices,
    )
    decision = audit_pilot_qualification(
        inputs,
        experiment_dir=HERE,
        registry=_SupportedRegistry(),
    )
    manifest = seal_manifest(
        build_manifest(
            inputs,
            experiment_dir=HERE,
            qualification_decision=decision,
            observed_source_state=_source_state(inputs),
        )
    )
    return inputs, decision, manifest


def test_pending_handoff_is_fail_closed(tmp_path):
    inputs = read_json(HERE / "pilot.inputs.v1.json")
    inputs["qualification"]["runtime_root"] = str(tmp_path)
    decision = audit_pilot_qualification(
        inputs,
        registry=_SupportedRegistry(),
    )
    assert decision.runtime_ready is False
    assert decision.successful_records == ()
    assert decision.blockers[0]["code"] == "qualification_handoff_pending"


def test_only_successful_qualified_ptms_become_arms(tmp_path):
    inputs, decision, manifest = _ready_manifest(
        tmp_path,
        failed_indices=(0,),
    )
    del inputs
    expected = (
        "grounding_dino.commercial.swin_tiny.trainable.v1.1",
    )
    assert decision.checkpoint_ids == expected
    assert len(decision.failed_records) == 1
    assert [item["id"] for item in manifest["ptms"]] == list(expected)
    assert all(
        item["allowed_ptm_ids"] == list(expected)
        for item in manifest["modes"]
    )
    assert manifest["execution"]["submission_ready"] is True


def test_successful_qualification_cannot_bypass_registry_review(tmp_path):
    inputs, _ = _write_qualification(tmp_path)
    decision = audit_pilot_qualification(
        inputs,
        experiment_dir=HERE,
        registry=_UnverifiedRegistry(),
    )
    assert decision.runtime_ready is False
    assert len(decision.successful_records) == 2
    assert {
        item["code"] for item in decision.blockers
    } == {"registry_status_not_supported"}
    with pytest.raises(PilotQualificationError, match="fail-closed"):
        decision.assert_runtime_ready()


def test_qualification_reports_standalone_metric_delta_without_fitted_tolerance(
    tmp_path,
):
    inputs, _ = _write_qualification(
        tmp_path,
        standalone_mAP50_delta=0.00125,
    )
    decision = audit_pilot_qualification(
        inputs,
        experiment_dir=HERE,
        registry=_SupportedRegistry(),
    )
    assert decision.runtime_ready is True
    assert all(
        item["standalone_evaluation_mAP50"]
        == pytest.approx(item["final_validation_mAP50"] + 0.00125)
        for item in decision.successful_records
    )
    assert all(
        item["standalone_minus_final_validation_mAP50"]
        == pytest.approx(0.00125)
        for item in decision.successful_records
    )


def test_qualification_agent_intervention_is_rejected(tmp_path):
    inputs, _ = _write_qualification(tmp_path)
    path = tmp_path / "qualification_completion.json"
    completion = read_json(path)
    completion["agent_intervention_flags"]["agent_selected_candidate"] = True
    completion.pop("completion_sha256")
    completion["completion_sha256"] = canonical_sha256(completion)
    path.write_text(
        __import__("json").dumps(completion, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        PilotQualificationError,
        match="agent-intervention",
    ):
        audit_pilot_qualification(
            inputs,
            registry=_SupportedRegistry(),
        )


def test_full_qualification_callback_checks_exact_input_bytes(
    tmp_path,
    monkeypatch,
):
    inputs, decision, _ = _ready_manifest(tmp_path / "evidence")
    del inputs
    record = decision.successful_records[0]
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"qualified")
    qualified = record["qualified_input_checkpoint"]
    monkeypatch.setitem(
        qualified,
        "sha256",
        hashlib_sha256(b"qualified"),
    )
    monkeypatch.setitem(qualified, "size_bytes", len(b"qualified"))
    callback = evidence_load_callback(decision)
    result = callback(
        SimpleNamespace(
            checkpoint_id=record["checkpoint_id"],
            checkpoint_path=checkpoint,
        )
    )
    assert result.ok is True
    checkpoint.write_bytes(b"changed")
    assert callback(
        SimpleNamespace(
            checkpoint_id=record["checkpoint_id"],
            checkpoint_path=checkpoint,
        )
    ).ok is False


def hashlib_sha256(value):
    import hashlib

    return hashlib.sha256(value).hexdigest()


def test_three_modes_route_independent_objective_acquisition(tmp_path):
    _, _, manifest = _ready_manifest(tmp_path)
    settings = {
        mode: mode_settings(manifest, mode) for mode in MODES
    }
    assert [
        settings[mode]["selection_mode"] for mode in MODES
    ] == list(MODES)
    assert [
        manifest["modes"][index]["objective"]["acquisition"]
        for index in range(3)
    ] == [
        "expected_improvement",
        "constrained_expected_improvement",
        "parego_expected_improvement",
    ]
    assert settings["latency"]["latency_accuracy_retention"] == {
        "type": "relative",
        "retained_fraction": 0.9,
        "reference": "accuracy_winner",
    }
    assert "latency_accuracy_retention" not in settings["multi_objective"]
    assert len(
        {
            item["observation_namespace"] for item in manifest["modes"]
        }
    ) == 3
    assert all(item["initial_observation_ids"] == [] for item in manifest["modes"])


def test_search_space_is_frozen_schema_enabled_and_nonordinal(tmp_path):
    _, _, manifest = _ready_manifest(tmp_path)
    assert tuple(manifest["search"]["parameters"]) == SEARCH_PARAMETERS
    assert manifest["ptm_representation"] == "hierarchical_nonordinal_arms"
    ranges = custom_ranges(manifest)
    assert ranges["model.enc_layers"]["valid_options"] == [3, 4, 5, 6]
    assert ranges["model.dec_layers"]["valid_options"] == [3, 4, 5, 6]
    assert ranges["model.num_select"]["valid_options"] == [100, 200, 300]
    assert ranges["train.optim.lr"] == {
        "valid_min": 1e-05,
        "valid_max": 0.0005,
    }
    assert manifest["search"]["candidate_budget_per_mode"] == 20
    assert manifest["search"]["training_epochs"] == 10


def test_grounding_data_text_and_eight_gpu_overrides_are_exact(tmp_path):
    _, _, manifest = _ready_manifest(tmp_path)
    dotted = spec_overrides(manifest)
    assert dotted["dataset.train_data_sources[0].label_map"].endswith(
        "annotations_odvg_labelmap.json"
    )
    assert dotted["dataset.val_data_sources.json_file"].endswith(
        "annotations_remapped.json"
    )
    assert dotted["dataset.eval_class_ids"] == [0, 1, 2, 3]
    assert dotted["model.text_encoder_type"].startswith("/lustre/")
    assert dotted["train.num_gpus"] == 8
    assert dotted["train.gpu_ids"] == list(range(8))
    assert dotted["train.num_epochs"] == 10
    assert dotted["train.is_dry_run"] is False
    assert manifest["runtime"]["sqsh_direct_path"] is True
    assert manifest["runtime"]["slurm_use_sqsh_conversion"] is False
    assert manifest["runtime"]["sqsh_path"].endswith(".sqsh")
    nested = nested_spec_overrides(manifest)
    assert len(nested["dataset"]["train_data_sources"]) == 1
    assert nested["dataset"]["val_data_sources"]["json_file"].endswith(
        "annotations_remapped.json"
    )


def test_evaluation_carries_candidate_architecture_and_exact_checkpoint(
    tmp_path,
):
    _, _, manifest = _ready_manifest(tmp_path)
    recommendation = {
        "model": {
            "enc_layers": 3,
            "dec_layers": 4,
            "num_queries": 900,
            "num_select": 200,
        },
        "train": {
            "optim": {
                "lr": 1e-4,
                "lr_backbone": 1e-5,
                "weight_decay": 1e-4,
            }
        },
    }
    checkpoint = "/lustre/results/model_epoch_009_step_00420.pth"
    spec = build_evaluation_spec(manifest, recommendation, checkpoint)
    assert spec["model"]["enc_layers"] == 3
    assert spec["model"]["dec_layers"] == 4
    assert spec["model"]["num_select"] == 200
    assert spec["evaluate"]["checkpoint"] == checkpoint
    assert spec["evaluate"]["num_gpus"] == 8
    assert spec["dataset"]["test_data_sources"]["json_file"].endswith(
        "annotations_remapped.json"
    )


def test_latency_protocol_is_stabilized_and_all_eight_replicas_count(
    tmp_path,
):
    _, _, manifest = _ready_manifest(tmp_path)
    protocol = manifest["latency_protocol"]
    assert protocol["warmup_iterations"] == 50
    assert protocol["timed_iterations"] == 100
    assert protocol["repeated_rounds"] == 5
    assert protocol["expected_replicas"] == 8
    assert protocol["raw_samples_per_candidate"] == 4000
    assert "text_tokenization_excluded" in protocol["timed_scope"]
    contract = latency_worker_contract(manifest)
    assert contract["measurement_role"] == "selection_time"
    assert contract["expected_replicas"] == 8
    command = latency_worker_command(
        checkpoint="/lustre/candidate.pth",
        candidate_fingerprint="f" * 64,
    )
    assert "torchrun --standalone --nproc_per_node=8" in command
    assert "grounding_dino_latency_worker.py" in command


def test_candidate_zero_barrier_releases_remaining_budget_automatically(
    tmp_path,
):
    _, _, manifest = _ready_manifest(tmp_path / "qualification")
    gate = AutomaticFirstCandidateGate(tmp_path / "gate", manifest)
    for mode in MODES:
        gate.record(
            mode,
            candidate_id=f"{mode}_rec_0",
            passed=True,
            evidence_sha256=mode.ljust(64, "0"),
            reason="all frozen candidate gates passed",
        )
    release = gate.wait_for_release()
    assert release["released"] is True
    assert release["automatic"] is True
    assert release["remaining_candidates_per_mode"] == 19
    assert (tmp_path / "gate" / "automatic_release.json").is_file()


def test_candidate_zero_failure_halts_before_rec_one(tmp_path):
    _, _, manifest = _ready_manifest(tmp_path / "qualification")
    gate = AutomaticFirstCandidateGate(tmp_path / "gate", manifest)
    gate.record(
        "accuracy",
        candidate_id="accuracy_rec_0",
        passed=False,
        evidence_sha256=None,
        reason="frozen failure",
    )
    with pytest.raises(PilotExecutionError, match="accuracy"):
        gate.wait_for_release()
    assert not (tmp_path / "gate" / "automatic_release.json").exists()


def test_dynamic_qualification_population_does_not_change_frozen_intent(
    tmp_path,
):
    _, _, all_ptms = _ready_manifest(tmp_path / "all")
    _, _, one_ptm = _ready_manifest(
        tmp_path / "one",
        failed_indices=(0,),
    )
    assert all_ptms["ptms"] != one_ptm["ptms"]
    assert frozen_intent_signature(all_ptms) == frozen_intent_signature(
        one_ptm
    )


def test_automatic_trigger_waits_then_releases_without_confirmation(
    tmp_path,
    monkeypatch,
):
    _, _, ready = _ready_manifest(tmp_path / "ready")
    pending = copy.deepcopy(ready)
    pending["ptms"] = []
    pending["qualification_evidence"]["successful_records"] = []
    pending["qualification_evidence"]["qualified_checkpoint_ids"] = []
    pending["qualification_evidence"]["runtime_ready"] = False
    pending["qualification_evidence"]["blockers"] = [
        {"code": "qualification_handoff_pending"}
    ]
    pending["execution"]["submission_ready"] = False
    for mode in pending["modes"]:
        mode["allowed_ptm_ids"] = []
    decision = pending["qualification_evidence"]
    decision.pop("decision_sha256")
    decision["decision_sha256"] = canonical_sha256(decision)
    pending.pop("manifest_sha256")
    pending = seal_manifest(pending)
    values = iter((pending, ready))
    monkeypatch.setattr(
        pilot_campaign,
        "assert_launchable",
        lambda *args, **kwargs: None,
    )
    release = wait_for_launch_authorization(
        pending,
        inputs=read_json(HERE / "pilot.inputs.v1.json"),
        experiment_dir=HERE,
        runtime_root=tmp_path / "runtime",
        poll_seconds=1,
        timeout_seconds=10,
        manifest_builder=lambda: next(values),
        sleeper=lambda _seconds: None,
        monotonic=iter((0.0, 0.0, 1.0, 1.0)).__next__,
    )
    assert release == ready
    status = read_json(tmp_path / "runtime" / "automatic_trigger_status.json")
    assert status["status"] == "ready"
    assert status["slurm_jobs_submitted"] is False
    assert read_json(tmp_path / "runtime" / "launch_manifest.json") == ready


def test_manifest_rejects_manual_intervention(tmp_path):
    _, _, manifest = _ready_manifest(tmp_path)
    changed = copy.deepcopy(manifest)
    changed["agent_intervention_flags"]["agent_overrode_winner"] = True
    changed.pop("manifest_sha256")
    changed["manifest_sha256"] = canonical_sha256(changed)
    with pytest.raises(PreparationError, match="manual intervention"):
        validate_manifest(changed)


def test_manifest_rejects_qualification_decision_tamper(tmp_path):
    _, _, manifest = _ready_manifest(tmp_path)
    changed = copy.deepcopy(manifest)
    changed["qualification_evidence"]["successful_records"][0][
        "standalone_evaluation_mAP50"
    ] += 0.01
    changed.pop("manifest_sha256")
    changed["manifest_sha256"] = canonical_sha256(changed)
    with pytest.raises(
        PreparationError,
        match="qualification decision canonical",
    ):
        validate_manifest(changed)
