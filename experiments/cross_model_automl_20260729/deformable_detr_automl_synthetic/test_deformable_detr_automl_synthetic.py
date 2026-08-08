"""Contract tests for the Deformable DETR shared-dataset AutoML campaign."""

from __future__ import annotations

import copy
import concurrent.futures
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from tao_automl.ptm_registry import canonical_sha256

from . import manifest_generator as generator
from . import qualification_evidence
from . import run_campaign
from . import deformable_detr_latency_worker as latency_worker


HERE = Path(__file__).resolve().parent


@dataclass
class _EvaluateDefaults:
    input_width: int | None = None
    input_height: int | None = None


@dataclass
class _ExportDefaults:
    format: str = "onnx"


@dataclass
class _ExperimentDefaults:
    evaluate: _EvaluateDefaults = field(default_factory=_EvaluateDefaults)
    export: _ExportDefaults = field(default_factory=_ExportDefaults)


@pytest.fixture(scope="module")
def manifest():
    return generator.seal_manifest(generator.build_manifest())


def test_campaign_is_direct_three_mode_and_preserves_frozen_budget(manifest):
    assert manifest["execution"] == {
        "kind": "objective_aware_three_mode_search",
        "cpu_runs": 0,
        "smoke_runs": 0,
        "local_model_runs": 0,
        "shared_archive": False,
        "independent_mode_jobs": True,
        "submission_ready": True,
        "blocked_before_sdk_construction": False,
    }
    assert manifest["search"]["candidate_budget_per_mode"] == 20
    assert manifest["search"]["training_epochs"] == 10
    assert manifest["search"]["search_seed"] == 271828
    assert manifest["search"]["training_seed"] == 1234
    assert manifest["search"]["space"] == generator.SEARCH_SPACE
    assert manifest["search"]["space_sha256"] == canonical_sha256(
        generator.SEARCH_SPACE
    )


def test_uses_shared_synthetic_dataset_and_exact_two_ptm_arms(manifest):
    dataset = manifest["dataset"]
    assert dataset["dataset_id"] == "tao_od_synthetic_full_dino_coco"
    assert dataset["source"]["staged_lustre_root"] == (
        "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
        "tao_od_synthetic_full_dino_coco"
    )
    assert dataset["splits"]["train"]["annotation"]["image_count"] == 1414
    assert dataset["splits"]["validation"]["annotation"]["image_count"] == 353
    assert [item["id"] for item in manifest["ptms"]] == list(
        qualification_evidence.EXPECTED_PTMS
    )
    assert all(
        item["official_source"].startswith("ngc://nvidia/tao/")
        for item in manifest["ptms"]
    )


def test_runtime_uses_packaged_skill_schema_and_pinned_sqsh(manifest):
    runtime = manifest["runtime"]
    assert runtime["slurm_use_sqsh"] is True
    assert runtime["sqsh_path"].endswith(".sqsh")
    assert runtime["sqsh_sha256"] == (
        "e36640f9ae7a03bc80828cf7de93bd6bdbbb0fecf509a71a243be0ab5b497fc2"
    )
    assert runtime["nodes_per_child"] == 1
    assert runtime["gpus_per_child"] == 8
    assert runtime["distributed_workers_per_child"] == 8
    assert len(runtime["train_schema_sha256"]) == 64
    assert len(runtime["train_template_sha256"]) == 64
    assert len(runtime["evaluate_template_sha256"]) == 64
    assert len(runtime["export_template_sha256"]) == 64


def test_mode_acquisitions_are_separate_and_objective_aware(manifest):
    settings = {
        mode: run_campaign.mode_settings(manifest, mode)
        for mode in generator.MODES
    }
    records = {
        item["mode"]: item["objective"]
        for item in manifest["modes"]
    }
    assert records["accuracy"]["acquisition"] == "expected_improvement"
    assert records["latency"]["acquisition"] == (
        "constrained_expected_improvement"
    )
    assert records["multi_objective"]["acquisition"] == (
        "parego_expected_improvement"
    )
    assert "latency_accuracy_retention" not in settings["accuracy"]
    assert settings["latency"]["latency_accuracy_retention"] == {
        "type": "relative",
        "retained_fraction": 0.90,
        "reference": "accuracy_winner",
    }
    assert "latency_accuracy_retention" not in settings["multi_objective"]
    assert settings["multi_objective"]["multi_objective_min_accuracy"] is None
    assert len(
        {item["observation_namespace"] for item in manifest["modes"]}
    ) == 3
    assert all(
        item["initial_observation_ids"] == []
        and item["observation_sharing"] is False
        for item in manifest["modes"]
    )


def test_custom_ranges_preserve_discrete_architecture_values(manifest):
    ranges = run_campaign.custom_ranges(manifest)
    assert ranges["model.enc_layers"] == {
        "valid_options": [3, 4, 5, 6]
    }
    assert ranges["model.dec_layers"] == {
        "valid_options": [3, 4, 5, 6]
    }
    assert ranges["model.num_queries"] == {
        "valid_options": [100, 200, 300]
    }
    assert ranges["train.optim.lr"] == {
        "valid_min": 1.0e-5,
        "valid_max": 5.0e-4,
    }


def test_spec_profile_is_full_dataset_ten_epoch_eight_gpu(manifest):
    overrides = run_campaign.spec_overrides(manifest)
    assert overrides["train.num_epochs"] == 10
    assert overrides["train.num_gpus"] == 8
    assert overrides["train.gpu_ids"] == list(range(8))
    assert overrides["train.num_nodes"] == 1
    assert overrides["dataset.num_classes"] == 5
    assert overrides["dataset.eval_class_ids"] == [1, 2, 3, 4]
    assert overrides["dataset.train_data_sources[0].image_dir"].endswith(
        "/train/images/images"
    )
    assert overrides["dataset.val_data_sources[0].image_dir"].endswith(
        "/val/images/images"
    )


def test_runtime_inventory_starts_from_exact_skill_train_template(manifest):
    defaults = run_campaign.skill_base_model_defaults(manifest)
    for parameter in generator.SEARCH_PARAMETERS:
        value = defaults
        for component in parameter.split("."):
            value = value[component]
        assert isinstance(value, (int, float))

    tampered = copy.deepcopy(manifest)
    tampered["runtime"]["train_template_sha256"] = "0" * 64
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="changed after campaign sealing",
    ):
        run_campaign.skill_base_model_defaults(tampered)


def test_evaluation_adapter_carries_candidate_architecture(manifest):
    result = run_campaign.build_evaluation_spec(
        manifest,
        {
            "model": {
                "enc_layers": 3,
                "dec_layers": 4,
                "num_queries": 100,
            }
        },
        "/lustre/results/candidate.pth",
    )
    assert result["model"]["enc_layers"] == 3
    assert result["model"]["dec_layers"] == 4
    assert result["model"]["num_queries"] == 100
    assert result["model"]["num_select"] == 100
    assert result["evaluate"]["checkpoint"] == (
        "/lustre/results/candidate.pth"
    )
    assert result["evaluate"]["num_gpus"] == 8
    assert result["export"]["format"] == "onnx"
    assert result["dataset"]["test_data_sources"]["json_file"].endswith(
        "/val/annotations.json"
    )

    tampered = copy.deepcopy(manifest)
    tampered["runtime"]["export_template_sha256"] = "0" * 64
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="spec_template_export.yaml changed after campaign sealing",
    ):
        run_campaign.build_evaluation_spec(
            tampered,
            {"model": {"num_queries": 100}},
            "/lustre/results/candidate.pth",
        )


def test_latency_input_manifest_records_actual_annotation_order(manifest):
    source = json.loads(
        (HERE / "latency_input.v1.json").read_text(encoding="utf-8")
    )
    expected_files = [
        "000246.jpg",
        "001170.jpg",
        "001602.jpg",
        "000291.jpg",
        "001372.jpg",
        "000078.jpg",
        "001548.jpg",
        "000681.jpg",
        "000865.jpg",
        "000454.jpg",
        "000482.jpg",
        "000864.jpg",
        "000858.jpg",
        "001213.jpg",
        "000643.jpg",
        "001398.jpg",
    ]
    assert [item["id"] for item in source["images"]] == list(range(16))
    assert [item["file_name"] for item in source["images"]] == expected_files
    descriptor = manifest["latency_protocol"]["input_descriptor"]
    assert descriptor["images"] == source["images"]
    assert descriptor["validation_image_ids"] == list(range(16))
    assert manifest["latency_protocol"]["input_descriptor_sha256"] == (
        canonical_sha256(descriptor)
    )


def test_stabilized_latency_contract_and_worker_are_deformable_detr_specific(
    manifest,
):
    protocol = manifest["latency_protocol"]
    assert protocol["warmup_iterations"] == 50
    assert protocol["timed_iterations"] == 100
    assert protocol["repeated_rounds"] == 5
    assert protocol["expected_replicas"] == 8
    assert protocol["raw_samples_per_candidate"] == 4000
    assert protocol["timed_scope"] == (
        "model_forward_plus_deformable_detr_gpu_postprocess"
    )
    worker = (
        HERE / "deformable_detr_latency_worker.py"
    ).read_text(encoding="utf-8")
    assert "DeformableDETRModel" in worker
    assert "DINOPlModel" not in worker
    assert "--nproc_per_node=8" in run_campaign.latency_worker_command(
        checkpoint="/lustre/model.pth",
        candidate_fingerprint="a" * 64,
    )


def test_latency_worker_materializes_complete_structured_model_defaults():
    config = latency_worker._materialize_experiment_config(
        {"evaluate": {"input_width": 960}},
        omega_conf=OmegaConf,
        experiment_config_type=_ExperimentDefaults,
    )

    assert config.evaluate.input_width == 960
    assert config.evaluate.input_height is None
    assert config.export.format == "onnx"


def test_first_candidate_gate_releases_automatically(manifest, tmp_path):
    gate = run_campaign.AutomaticFirstCandidateGate(
        tmp_path,
        manifest,
        poll_seconds=0.001,
        timeout_seconds=1,
    )
    for mode in generator.MODES:
        gate.record(
            mode,
            candidate_id=f"{mode}-rec-0",
            passed=True,
            evidence_sha256="a" * 64,
            reason="all frozen candidate gates passed",
        )
    release = gate.wait_for_release()
    assert release["released"] is True
    assert release["automatic"] is True
    assert release["remaining_candidates_per_mode"] == 19


def test_first_candidate_gate_release_is_safe_for_three_waiters(
    manifest,
    tmp_path,
):
    gate = run_campaign.AutomaticFirstCandidateGate(
        tmp_path,
        manifest,
        poll_seconds=0.001,
        timeout_seconds=1,
    )
    for mode in generator.MODES:
        gate.record(
            mode,
            candidate_id=f"{mode}-rec-0",
            passed=True,
            evidence_sha256="a" * 64,
            reason="all frozen candidate gates passed",
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        releases = list(pool.map(lambda _index: gate.wait_for_release(), range(3)))
    assert len(releases) == 3
    assert all(item == releases[0] for item in releases)


def test_first_candidate_failure_halts_before_release(manifest, tmp_path):
    gate = run_campaign.AutomaticFirstCandidateGate(
        tmp_path,
        manifest,
        poll_seconds=0.001,
        timeout_seconds=1,
    )
    gate.record(
        "accuracy",
        candidate_id="accuracy-rec-0",
        passed=False,
        evidence_sha256=None,
        reason="evaluation failed",
    )
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="first-candidate gate failed",
    ):
        gate.wait_for_release()
    assert not (tmp_path / "automatic_release.json").exists()


def test_candidate_adapter_raises_after_recording_first_candidate_failure(
    manifest,
    tmp_path,
):
    gate = run_campaign.AutomaticFirstCandidateGate(
        tmp_path / "first_candidate_gate",
        manifest,
        poll_seconds=0.001,
        timeout_seconds=1,
    )
    evaluator = run_campaign.DeformableDETRCandidateEvaluator(
        sdk=object(),
        manifest=manifest,
        mode="latency",
        runtime_root=tmp_path,
        gate=gate,
    )
    recommendation = SimpleNamespace(
        id=0,
        failure_reason="required_eval_fn_failed:latency worker failed",
    )

    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="latency_rec_0 failed the automatic first-candidate gate",
    ):
        evaluator.on_result(recommendation, None, "failure")

    gate_record = json.loads(
        (
            tmp_path
            / "first_candidate_gate"
            / "latency.json"
        ).read_text()
    )
    assert gate_record["passed"] is False
    assert not (
        tmp_path / "first_candidate_gate" / "automatic_release.json"
    ).exists()


def test_candidate_adapter_blocks_second_candidate_before_job_launch(
    manifest,
    tmp_path,
):
    gate = run_campaign.AutomaticFirstCandidateGate(
        tmp_path / "first_candidate_gate",
        manifest,
        poll_seconds=0.001,
        timeout_seconds=1,
    )
    gate.record(
        "latency",
        candidate_id="latency_rec_0",
        passed=False,
        evidence_sha256=None,
        reason="required latency worker failed",
    )
    evaluator = run_campaign.DeformableDETRCandidateEvaluator(
        sdk=object(),
        manifest=manifest,
        mode="latency",
        runtime_root=tmp_path,
        gate=gate,
    )

    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="first-candidate gate failed: latency",
    ):
        evaluator.on_recommendation(SimpleNamespace(id=1))
    assert not evaluator.evidence_path.exists()


def test_qualification_adapter_gates_on_shared_dataset_campaign(manifest):
    evidence = manifest["qualification_evidence"]
    assert evidence["evidence_path"] == str(
        qualification_evidence.DEFAULT_QUALIFICATION_COMPLETION
    )
    assert evidence["expected_manifest_sha256"] == (
        qualification_evidence.EXPECTED_QUALIFICATION_MANIFEST_SHA256
    )
    assert evidence["runtime_ready"] is True
    assert evidence["blockers"] == []


def test_launch_fails_before_sdk_for_a_synthetic_blocked_projection(manifest):
    blocked = _blocked_projection(manifest)
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="fail-closed|changed after sealing",
    ):
        run_campaign.assert_launchable(blocked)
    run_campaign.assert_launchable(manifest)


def test_run_mode_rejects_non_typed_ptm_inventory_before_sdk_use(
    manifest,
    tmp_path,
):
    with pytest.raises(TypeError, match="live typed production object"):
        run_campaign.run_mode(
            manifest=manifest,
            mode="accuracy",
            sdk=object(),
            resolved_ptm_inventory={"serialized": "forbidden"},
            workspace_path=tmp_path,
            eval_latency_fn=lambda _rec, _job: {},
        )


def test_launch_plan_exposes_automatic_gate_and_exact_blocker(manifest):
    plan = run_campaign.launch_plan(manifest)
    assert plan["submission_ready"] is True
    assert plan["qualification_blockers"] == []
    assert plan["first_candidate_gate"]["automatic_release"] is True
    assert plan["first_candidate_gate"]["remaining_candidates_per_mode"] == 19
    assert plan["per_candidate_children"] == [
        "eight_gpu_ten_epoch_training",
        "eight_gpu_standalone_evaluation",
        "eight_replica_stabilized_latency",
    ]


def test_manifest_tampering_fails_closed(manifest):
    tampered = copy.deepcopy(manifest)
    tampered["search"]["candidate_budget_per_mode"] = 21
    with pytest.raises(generator.ManifestError):
        generator.validate_manifest(tampered)

    tampered = copy.deepcopy(manifest)
    tampered["execution"]["submission_ready"] = False
    tampered.pop("manifest_sha256")
    with pytest.raises(generator.ManifestError):
        generator.validate_manifest(tampered, require_seal=False)


def test_agent_and_selection_isolation_flags_are_false(manifest):
    assert all(not value for value in manifest["agent_intervention_flags"].values())
    assert all(not value for value in manifest["selection_isolation_flags"].values())


def _blocked_projection(manifest):
    value = copy.deepcopy(manifest)
    value.pop("manifest_sha256")
    value["qualification_evidence"]["blockers"] = [
        {
            "checkpoint_id": ptm["id"],
            "code": "registry_status_not_supported",
            "observed_status": "unverified",
            "reason": "synthetic automatic-trigger test blocker",
            "required_status": "supported",
            "stage": "registry_runtime_eligibility",
        }
        for ptm in value["ptms"]
    ]
    value["qualification_evidence"]["runtime_ready"] = False
    value["qualification_evidence"]["decision_sha256"] = "b" * 64
    value["execution"]["submission_ready"] = False
    value["execution"]["blocked_before_sdk_construction"] = True
    for ptm in value["ptms"]:
        ptm["registry_status"] = "unverified"
        ptm["registry_record_sha256"] = "c" * 64
    return generator.seal_manifest(value)


def test_automatic_trigger_waits_then_releases_same_frozen_campaign(
    manifest,
    tmp_path,
):
    blocked = _blocked_projection(manifest)
    documents = iter((blocked, manifest))
    observed = run_campaign.wait_for_launch_authorization(
        blocked,
        runtime_root=tmp_path,
        poll_seconds=0.001,
        timeout_seconds=1,
        manifest_builder=lambda: next(documents),
        readiness_check=lambda _value: None,
        sleeper=lambda _seconds: None,
        monotonic=lambda: 0.0,
    )
    assert observed == manifest
    assert run_campaign.frozen_campaign_signature(observed) == (
        run_campaign.frozen_campaign_signature(blocked)
    )
    assert json.loads(
        (tmp_path / "automatic_trigger_status.json").read_text()
    )["status"] == "ready"
    assert json.loads(
        (tmp_path / "launch_manifest.json").read_text()
    ) == manifest


def test_automatic_trigger_rejects_preregistered_drift(manifest, tmp_path):
    blocked = _blocked_projection(manifest)
    drifted = copy.deepcopy(manifest)
    drifted.pop("manifest_sha256")
    drifted["runtime"]["time_hours"] = 5.0
    drifted = generator.seal_manifest(drifted)
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="campaign drift",
    ):
        run_campaign.wait_for_launch_authorization(
            blocked,
            runtime_root=tmp_path,
            poll_seconds=0.001,
            timeout_seconds=1,
            manifest_builder=lambda: drifted,
            readiness_check=lambda _value: None,
            sleeper=lambda _seconds: None,
            monotonic=lambda: 0.0,
        )


def test_runtime_execution_projection_uses_sealed_lustre_identity(manifest):
    prepared = tuple(
        SimpleNamespace(
            checkpoint_id=item["id"],
            checkpoint=SimpleNamespace(
                sha256=item["artifact"]["sha256"],
                size_bytes=item["artifact"]["size_bytes"],
            ),
        )
        for item in manifest["ptms"]
    )
    report = SimpleNamespace(prepared=prepared)
    projected = run_campaign.execution_checkpoint_artifacts(
        manifest,
        report,
    )
    assert projected == {
        item["id"]: {
            "path": item["artifact"]["slurm_path"],
            "sha256": item["artifact"]["sha256"],
            "size_bytes": item["artifact"]["size_bytes"],
        }
        for item in manifest["ptms"]
    }
    corrupt = copy.deepcopy(manifest)
    corrupt["ptms"][0]["artifact"]["sha256"] = "0" * 64
    with pytest.raises(
        run_campaign.CampaignExecutionError,
        match="does not preserve live preflight identity",
    ):
        run_campaign.execution_checkpoint_artifacts(corrupt, report)


def test_exact_terminal_checkpoint_adapter_is_epoch_bound(monkeypatch):
    qualification = run_campaign._experiment_run_campaign(
        "deformable_detr_campaign"
    )

    calls = []

    def exact(_sdk, job_id, *, training_epochs):
        calls.append((job_id, training_epochs))
        return {
            "path": "/lustre/results/model_epoch_009_step_123.pth",
            "sha256": "a" * 64,
            "size_bytes": 123,
            "training_epochs": 10,
            "terminal_epoch_index": 9,
        }

    monkeypatch.setattr(qualification, "_terminal_checkpoint", exact)
    result = run_campaign.terminal_checkpoint_identity(
        object(),
        "job-123",
        training_epochs=10,
    )
    assert calls == [("job-123", 10)]
    assert result["path"].endswith("model_epoch_009_step_123.pth")
    assert "mtime" not in json.dumps(result)


def test_three_mode_controller_is_concurrent_and_inventory_typed_at_boundary(
    manifest,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(run_campaign, "assert_launchable", lambda _value: None)
    barrier = threading.Barrier(3)
    observed = []

    class Inventory:
        def __init__(self, mode):
            self.mode = mode

        def to_dict(self):
            return {"mode": self.mode, "typed": True}

    class Report:
        def to_dict(self):
            return {"typed": True}

    def inventory_builder(_manifest, *, mode, report):
        assert isinstance(report, Report)
        return Inventory(mode)

    def sdk_factory(mode, state_file):
        assert state_file.parent.name == mode
        return SimpleNamespace(mode=mode)

    def mode_runner(**kwargs):
        barrier.wait(timeout=1)
        observed.append(
            (
                kwargs["mode"],
                kwargs["sdk"].mode,
                kwargs["resolved_ptm_inventory"].mode,
            )
        )
        return {"mode": kwargs["mode"], "status": "success"}

    completion = run_campaign.launch_mode_controllers(
        manifest=manifest,
        report=Report(),
        runtime_root=tmp_path,
        sdk_factory=sdk_factory,
        inventory_builder=inventory_builder,
        mode_runner=mode_runner,
    )
    assert completion["status"] == "success"
    assert sorted(observed) == [
        ("accuracy", "accuracy", "accuracy"),
        ("latency", "latency", "latency"),
        (
            "multi_objective",
            "multi_objective",
            "multi_objective",
        ),
    ]
