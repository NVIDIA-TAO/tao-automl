"""Contract tests for the RT-DETR synthetic three-mode campaign."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tao_automl.ptm_preflight import CheckpointLoadSmokeRequest
from tao_automl.recommendation_audit import build_recommendation_audit
from tao_automl.ptm_registry import canonical_sha256

from . import campaign_contract
from . import qualification_gate
from . import run_campaign


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _Registry:
    def __init__(self, *, supported: bool, source_sha: str, source_size: int):
        self._status = "supported" if supported else "unverified"
        self._source_sha = source_sha
        self._source_size = source_size

    def checkpoint(self, checkpoint_id):
        assert checkpoint_id in qualification_gate.EXPECTED_PTMS
        return {
            "id": checkpoint_id,
            "status": self._status,
            "sha256": self._source_sha,
            "expected_size_bytes": self._source_size,
        }


def _workflow(checkpoint_id: str, source_sha: str) -> dict:
    workflow_id = checkpoint_id.replace(".", "_")
    terminal = f"terminal-{checkpoint_id}".encode()
    return {
        "schema_version": 1,
        "campaign_id": "rtdetr-qualification",
        "manifest_sha256": "a" * 64,
        "workflow_id": workflow_id,
        "ptm_id": checkpoint_id,
        "ptm_sha256": source_sha,
        "status": "success",
        "terminal": True,
        "failure_preserved": False,
        "resume": {
            "completed_training_job_reused": True,
            "training_job_submitted": False,
            "selection_or_candidate_change": False,
            "prior_workflow_artifact_modified": False,
            "checkpoint_resolved_after_fix": True,
        },
        "agent_intervention_flags": {
            name: False for name in qualification_gate.AGENT_FLAGS
        },
        "jobs": {
            "train": {
                "status": "Complete",
                "full_dataset": True,
                "training_epochs": 10,
                "validation_interval": 1,
                "nodes": 1,
                "gpus": 8,
                "status_evidence": {
                    "terminal_success": True,
                    "validation_record_count": 10,
                },
                "terminal_checkpoint": {
                    "path": f"/lustre/results/{workflow_id}/model.pth",
                    "sha256": _sha(terminal),
                    "size_bytes": len(terminal),
                },
            },
            "evaluation": {
                "status": "Complete",
                "full_validation_split": True,
                "nodes": 1,
                "gpus": 8,
                "status_evidence": {
                    "terminal_success": True,
                    "test_metric_record_count": 1,
                },
            },
        },
        "metrics": {"mAP": 0.4, "mAP50": 0.6},
    }


def _completion(path: Path, source_sha: str) -> Path:
    workflows = [
        _workflow(checkpoint_id, source_sha)
        for checkpoint_id in qualification_gate.EXPECTED_PTMS
    ]
    payload = {
        "schema_version": 1,
        "campaign_id": "rtdetr-qualification",
        "model": "rtdetr",
        "manifest_sha256": "a" * 64,
        "terminal": True,
        "status": "success",
        "logical_workflows_submitted": 4,
        "successful_workflows": 4,
        "failed_workflows": 0,
        "completion_generated_automatically": True,
        "resume_completed_training": True,
        "completed_training_jobs_reused": 4,
        "training_jobs_submitted": 0,
        "prior_completion_artifact_modified": False,
        "failures_preserved": True,
        "replacement_workflows_submitted": False,
        "workflows": workflows,
    }
    payload["completion_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_qualification_gate_blocks_unpromoted_registry(
    tmp_path,
    monkeypatch,
):
    source = b"official-checkpoint"
    completion = _completion(tmp_path / "completion.json", _sha(source))
    monkeypatch.setattr(
        qualification_gate,
        "load_ptm_registry",
        lambda: _Registry(
            supported=False,
            source_sha=_sha(source),
            source_size=len(source),
        ),
    )

    decision = qualification_gate.audit_qualification(
        completion,
        expected_manifest_sha256="a" * 64,
    )

    assert not decision.runtime_ready
    assert len(decision.qualified) == 4
    assert {
        item["code"] for item in decision.blockers
    } == {"registry_status_not_supported"}
    with pytest.raises(
        qualification_gate.QualificationGateError,
        match="fail-closed",
    ):
        decision.assert_runtime_ready()


def test_qualified_full_gpu_evidence_replaces_model_smoke_without_bypass(
    tmp_path,
    monkeypatch,
):
    source = b"official-checkpoint"
    checkpoint = tmp_path / "official.pth"
    checkpoint.write_bytes(source)
    registry = _Registry(
        supported=True,
        source_sha=_sha(source),
        source_size=len(source),
    )
    monkeypatch.setattr(
        qualification_gate, "load_ptm_registry", lambda: registry
    )
    completion = _completion(tmp_path / "completion.json", _sha(source))
    decision = qualification_gate.audit_qualification(
        completion,
        expected_manifest_sha256="a" * 64,
    )
    callback = qualification_gate.QualificationLoadEvidence(decision)
    checkpoint_id = qualification_gate.EXPECTED_PTMS[0]
    request = CheckpointLoadSmokeRequest(
        checkpoint_id=checkpoint_id,
        model="rtdetr",
        task="object_detection",
        tao_version="7.1.0",
        checkpoint_path=checkpoint,
        checkpoint_spec_path=tmp_path / "spec.yaml",
        checkpoint_spec={"model": {}},
        default_spec_overrides={"model": {}},
        registry_record=registry.checkpoint(checkpoint_id),
    )

    result = callback(request)

    assert result.ok
    assert result.code == "full_train_eval_qualification_reused"
    assert result.details["cpu_or_smoke_model_job_launched"] is False


def test_completion_integrity_and_manifest_identity_fail_closed(
    tmp_path,
    monkeypatch,
):
    source = b"official-checkpoint"
    path = _completion(tmp_path / "completion.json", _sha(source))
    monkeypatch.setattr(
        qualification_gate,
        "load_ptm_registry",
        lambda: _Registry(
            supported=True,
            source_sha=_sha(source),
            source_size=len(source),
        ),
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    document["successful_workflows"] = 3
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(
        qualification_gate.QualificationGateError,
        match="integrity",
    ):
        qualification_gate.audit_qualification(
            path, expected_manifest_sha256="a" * 64
        )

    _completion(path, _sha(source))
    with pytest.raises(
        qualification_gate.QualificationGateError,
        match="different sealed manifest",
    ):
        qualification_gate.audit_qualification(
            path, expected_manifest_sha256="b" * 64
        )


def test_mode_policies_are_independent_and_objective_aware():
    accuracy = campaign_contract.mode_settings("campaign", "accuracy")
    latency = campaign_contract.mode_settings("campaign", "latency")
    moo = campaign_contract.mode_settings("campaign", "multi_objective")

    assert "latency_accuracy_retention" not in accuracy
    assert latency["latency_accuracy_retention"] == {
        "type": "relative",
        "retained_fraction": 0.9,
        "reference": "accuracy_winner",
    }
    assert "latency_accuracy_retention" not in moo
    assert moo["multi_objective_min_accuracy"] is None
    assert (
        campaign_contract.mode_objective("accuracy")["acquisition"]
        == "expected_improvement"
    )
    assert (
        campaign_contract.mode_objective("latency")["acquisition"]
        == "constrained_expected_improvement"
    )
    assert (
        campaign_contract.mode_objective("multi_objective")["acquisition"]
        == "parego_expected_improvement"
    )
    assert len(
        {
            accuracy["experiment_id"],
            latency["experiment_id"],
            moo["experiment_id"],
        }
    ) == 3


def test_search_space_is_frozen_dependent_and_train_only():
    assert campaign_contract.FROZEN_CANDIDATE_BUDGET == 20
    assert campaign_contract.FROZEN_TRAINING_EPOCHS == 10
    assert campaign_contract.SEARCH_PARAMETERS == (
        "model.enc_layers",
        "model.dec_layers",
        "model.num_queries",
        "model.num_select",
        "train.optim.lr",
        "train.optim.weight_decay",
    )
    select = campaign_contract.SEARCH_SPACE["model.num_select"]
    assert select["depends_on"] == "model.num_queries"
    assert select["constraint"] == "value <= model.num_queries"
    assert campaign_contract.SEARCH_SPACE["model.num_queries"][
        "values"
    ] == [100, 200, 300]
    assert campaign_contract.SEARCH_SPACE["model.num_select"][
        "values"
    ] == [50, 100, 200, 300]
    assert campaign_contract.custom_ranges() == {
        name: {
            "valid_min": domain["minimum"],
            "valid_max": domain["maximum"],
            **(
                {"valid_options": domain["values"]}
                if "values" in domain
                else {}
            ),
            **(
                {"depends_on": "model.num_queries"}
                if name == "model.num_select"
                else {}
            ),
        }
        for name, domain in campaign_contract.SEARCH_SPACE.items()
    }
    assert all(
        not name.startswith(("quantize", "gen_trt_engine"))
        for name in campaign_contract.SEARCH_PARAMETERS
    )


def test_bayesian_integer_adapter_honors_frozen_discrete_options():
    from tao_automl.brain.bayesian import Bayesian

    brain = object.__new__(Bayesian)
    brain.custom_ranges = {
        "model.num_queries": campaign_contract.custom_ranges()[
            "model.num_queries"
        ]
    }
    brain.parent_params = {}
    brain.network = None
    brain.default_train_spec = {}
    parameter = {
        "parameter": "model.num_queries",
        "value_type": "int",
        "default_value": 300,
        "valid_min": 100,
        "valid_max": 300,
        "valid_options": [],
        "parent_param": "TRUE",
        "math_cond": None,
    }

    observed = [
        brain.generate_automl_param_rec_value(
            copy.deepcopy(parameter), suggestion
        )
        for suggestion in (0.0, 0.34, 0.67, 0.999999)
    ]

    assert observed == [100, 200, 300, 300]
    assert set(observed) <= {100, 200, 300}


def test_packaged_schema_has_frozen_train_parameters_and_no_quantize_leak():
    skill = Path(
        "/localhome/local-rarunachalam/.tao/worktrees/"
        "tao-skills-release-7.1.0/skills/models/tao-train-rtdetr"
    )
    evidence = campaign_contract.validate_packaged_train_schema(skill)

    assert evidence["non_train_fields_excluded"] is True
    assert evidence["leaked_quantize_fields"] == []
    assert evidence["explicit_search_parameters"] == list(
        campaign_contract.SEARCH_PARAMETERS
    )


def test_first_candidate_gate_releases_only_complete_three_mode_cohort(
    tmp_path,
):
    processes = {
        mode: SimpleNamespace(is_alive=lambda: True)
        for mode in campaign_contract.MODES
    }
    gate = tmp_path / "first_candidate_gate"
    gate.mkdir()
    for mode in campaign_contract.MODES:
        (gate / f"{mode}.json").write_text(
            json.dumps(
                {
                    "contract_sha256": "c" * 64,
                    "mode": mode,
                    "passed": True,
                }
            ),
            encoding="utf-8",
        )

    release = run_campaign._release_first_candidate_gate(
        tmp_path, processes, "c" * 64
    )

    assert release["release_remaining_budget"] is True
    assert release["generated_automatically"] is True
    assert release["modes"] == list(campaign_contract.MODES)


def test_first_candidate_gate_fails_closed_on_any_failed_mode(tmp_path):
    processes = {
        mode: SimpleNamespace(is_alive=lambda: True)
        for mode in campaign_contract.MODES
    }
    gate = tmp_path / "first_candidate_gate"
    gate.mkdir()
    for mode in campaign_contract.MODES:
        (gate / f"{mode}.json").write_text(
            json.dumps(
                {
                    "contract_sha256": "c" * 64,
                    "mode": mode,
                    "passed": mode != "latency",
                }
            ),
            encoding="utf-8",
        )

    release = run_campaign._release_first_candidate_gate(
        tmp_path, processes, "c" * 64
    )

    assert release["release_remaining_budget"] is False


def test_latency_protocol_is_exact_eight_replica_stabilized_contract():
    protocol = campaign_contract.LATENCY_PROTOCOL
    assert protocol["warmup_iterations"] == 50
    assert protocol["timed_iterations"] == 100
    assert protocol["repeated_rounds"] == 5
    assert protocol["expected_replicas"] == 8
    assert protocol["raw_samples_per_candidate"] == 4000
    assert (
        protocol["timed_scope"]
        == "model_forward_plus_rtdetr_gpu_postprocess"
    )


def test_execution_checkpoint_projection_is_part_of_launcher_source():
    source = (
        Path(run_campaign.__file__).read_text(encoding="utf-8")
    )
    assert "execution_checkpoint_artifacts=" in source
    assert "ptm_aware_runtime=True" in source
    assert "resolved_ptm_inventory=resolved_inventory" in source
    assert "gpu_count=8" in source
    assert "num_nodes=1" in source
    assert "verify_live_runtime_preflight(" in source


def test_ptm_profiles_preserve_qualified_input_resolution():
    manifest = json.loads(
        (
            Path(run_campaign.DEFAULT_QUALIFICATION_MANIFEST)
        ).read_text(encoding="utf-8")
    )
    profiles = run_campaign._per_checkpoint_profiles(
        manifest,
        qualification_gate.EXPECTED_PTMS,
    )

    assert profiles[
        "rtdetr.trafficcam.resnet18.trainable.v2.0"
    ]["dataset"]["augmentation"] == {
        "train_spatial_size": [544, 960],
        "eval_spatial_size": [544, 960],
        "preserve_aspect_ratio": False,
    }
    assert profiles[
        "rtdetr.warehouse.resnet50.trainable.v1.0.2"
    ]["dataset"]["augmentation"] == {
        "train_spatial_size": [640, 640],
        "eval_spatial_size": [640, 640],
        "preserve_aspect_ratio": False,
    }


def test_automl_checkpoint_adapter_uses_exact_rtdetr_resolver(monkeypatch):
    expected = {
        "path": "/lustre/results/results_dir/train/model_epoch_009.pth",
        "sha256": "a" * 64,
        "size_bytes": 42,
        "training_epochs": 10,
        "terminal_epoch_index": 9,
        "filename": "model_epoch_009.pth",
        "naming_contract": "rtdetr_model_epoch_without_step_suffix",
        "ambiguity_policy": "fail_closed",
    }
    calls = []

    def resolve(sdk, job_id, *, training_epochs):
        calls.append((sdk, job_id, training_epochs))
        return expected

    monkeypatch.setattr(
        run_campaign.qualification_runner,
        "_terminal_checkpoint",
        resolve,
    )
    sdk = object()

    assert run_campaign._terminal_checkpoint(sdk, "job-id") == expected
    assert calls == [(sdk, "job-id", 10)]


def _recommendation(
    *,
    specs: dict | None = None,
    candidate_id: str = "0",
) -> SimpleNamespace:
    values = specs or {
        "model": {"enc_layers": 1},
        "train": {"pretrained_model_path": "/lustre/ptm/model.pth"},
    }
    objective = SimpleNamespace(
        to_dict=lambda: {
            "selection": {"mode": "accuracy"},
            "objectives": [],
        }
    )
    audit = build_recommendation_audit(
        candidate_id=candidate_id,
        specs=values,
        algorithm="bayesian",
        search_seed=271828,
        search_space=[],
        custom_ranges={},
        objective_config=objective,
        visible_history=[],
        acquisition={
            "proposal": {
                "ptm": {
                    "arm_id": qualification_gate.EXPECTED_PTMS[0],
                }
            }
        },
    )
    return SimpleNamespace(
        id=candidate_id,
        specs=values,
        recommendation_audit=audit,
    )


def test_resume_recommendation_identity_is_preserved_and_drift_rejected():
    candidates: dict[str, object] = {}
    original = run_campaign._immutable_recommendation_record(
        _recommendation(),
        "accuracy",
    )
    run_campaign._preserve_or_add_recommendation(candidates, original)
    candidates["accuracy_rec_0"]["status"] = "success"
    candidates["accuracy_rec_0"]["objective_values"] = {"mAP50": 0.5}

    run_campaign._preserve_or_add_recommendation(
        candidates,
        run_campaign._immutable_recommendation_record(
            _recommendation(),
            "accuracy",
        ),
    )
    assert candidates["accuracy_rec_0"]["status"] == "success"
    assert candidates["accuracy_rec_0"]["objective_values"] == {
        "mAP50": 0.5
    }

    changed = _recommendation(
        specs={
            "model": {"enc_layers": 2},
            "train": {
                "pretrained_model_path": "/lustre/ptm/model.pth"
            },
        }
    )
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="resumed recommendation changed",
    ):
        run_campaign._preserve_or_add_recommendation(
            candidates,
            run_campaign._immutable_recommendation_record(
                changed,
                "accuracy",
            ),
        )


def test_automatic_wait_rejects_immutable_terminal_failure(tmp_path):
    completion = tmp_path / "completion.resume.json"
    completion.write_text(
        json.dumps(
            {
                "terminal": True,
                "status": "terminal_with_failures",
            }
        ),
        encoding="utf-8",
    )
    status = tmp_path / "gate.json"

    with pytest.raises(
        qualification_gate.QualificationGateError,
        match="immutable resumed qualification completion is terminal",
    ):
        run_campaign.wait_for_successful_qualification(
            completion,
            expected_manifest_sha256="a" * 64,
            poll_seconds=0.001,
            status_path=status,
        )

    recorded = json.loads(status.read_text(encoding="utf-8"))
    assert recorded["status"] == "blocked"
    assert recorded["successor_launched"] is False


def test_automatic_wait_returns_only_after_supported_registry(
    tmp_path,
    monkeypatch,
):
    source = b"official-checkpoint"
    completion = _completion(
        tmp_path / "completion.resume.json",
        _sha(source),
    )
    monkeypatch.setattr(
        qualification_gate,
        "load_ptm_registry",
        lambda: _Registry(
            supported=True,
            source_sha=_sha(source),
            source_size=len(source),
        ),
    )
    status = tmp_path / "gate.json"

    decision = run_campaign.wait_for_successful_qualification(
        completion,
        expected_manifest_sha256="a" * 64,
        poll_seconds=0.001,
        status_path=status,
    )

    assert decision.runtime_ready
    assert json.loads(status.read_text(encoding="utf-8"))["status"] == "ready"
