from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from tao_automl.recommendation_audit import (
    algorithmic_campaign_flags,
    build_recommendation_audit,
    canonical_audit_sha256,
)
from tao_automl.selection import analyze_archive

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from automatic_successor import (  # noqa: E402
    AutomaticSuccessorError,
    EXPECTED_MODEL_BASED_METHOD,
    EXPECTED_OPTIMIZATION_DIRECTION,
    MODES,
    SELECTION_TIME_ISOLATION,
    _history_snapshot,
    _process_identity,
    _selection_config,
    _validate_candidate_history_alignment,
    canonical_sha256,
    routing_identity_from_environment_file,
    sha256_file,
    trigger_successor,
    validate_completed_dino,
    validate_successor_completion,
    validate_successor_descriptor,
    watch_and_trigger,
)
from dino_campaign.manifest_generator import load_manifest  # noqa: E402


MANIFEST_PATH = HERE / "dino_campaign" / "campaign.v1.json"
MANIFEST = load_manifest(MANIFEST_PATH)


class _Objectives:
    def __init__(self, mode: str):
        self.mode = mode

    def to_dict(self):
        return {
            "mode": self.mode,
            "objectives": [
                {"metric": "mAP50", "direction": "maximize"},
                {"metric": "latency_ms", "direction": "minimize"},
            ],
        }


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _successor_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    marker = tmp_path / "successor-runtime" / "completion.json"
    script = tmp_path / "successor.py"
    script.write_text(
        "import argparse, hashlib, json, pathlib, time\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--manifest', required=True)\n"
        "p.add_argument('--runtime-root', required=True)\n"
        "p.add_argument('--completion-artifact', required=True)\n"
        "p.add_argument('--env-file', required=True)\n"
        "p.add_argument('--launch', action='store_true')\n"
        "p.add_argument('--acknowledge-direct-full-dataset', action='store_true')\n"
        "p.add_argument('--skip-completion', action='store_true')\n"
        "a=p.parse_args()\n"
        "if a.skip_completion: raise SystemExit(0)\n"
        "m=json.loads(pathlib.Path(a.manifest).read_text())\n"
        "ids={'gcvit_tiny':'deformable-detr-gcvit',"
        "'resnet50':'deformable-detr-resnet50'}\n"
        "w=[{'workflow_id':n,'ptm_id':ids[n],'status':'success',"
        "'terminal':True,'process_exit_code':0,'failure_preserved':False,"
        "'metrics':{'mAP':0.4,'mAP50':0.6}} for n in ids]\n"
        "c={'schema_version':1,'campaign_id':m['campaign_id'],"
        "'model':'deformable_detr','manifest_sha256':m['manifest_sha256'],"
        "'terminal':True,'status':'success','logical_workflows_submitted':2,"
        "'successful_workflows':2,'failed_workflows':0,"
        "'workflows_started_in_parallel':True,'cpu_runs':0,'smoke_runs':0,"
        "'ministep_runs':0,'local_model_runs':0,'failures_preserved':True,"
        "'replacement_workflows_submitted':False,"
        "'outcomes':{'gcvit_tiny':'success','resnet50':'success'},"
        "'workflows':w}\n"
        "raw=json.dumps(c,sort_keys=True,separators=(',',':'),"
        "ensure_ascii=True,allow_nan=False).encode()\n"
        "c['completion_sha256']=hashlib.sha256(raw).hexdigest()\n"
        "out=pathlib.Path(a.completion_artifact);out.parent.mkdir("
        "parents=True,exist_ok=True)\n"
        "out.write_text(json.dumps(c,sort_keys=True)+'\\n',encoding='utf-8')\n"
        "time.sleep(0.1)\n",
        encoding="utf-8",
    )
    generator = tmp_path / "manifest_generator.py"
    generator.write_text("# sealed test manifest generator\n", encoding="utf-8")
    campaign_manifest = tmp_path / "deformable_detr_campaign.v1.json"
    manifest_payload = {
        "schema_version": 1,
        "campaign_id": "deformable-detr-test-successor",
        "model": "deformable_detr",
        "task": "object_detection",
        "execution": {
            "kind": "direct_full_qualification",
            "cpu_runs": 0,
            "smoke_runs": 0,
            "ministep_runs": 0,
            "local_model_runs": 0,
            "full_training": True,
            "standalone_evaluation": True,
        },
        "runtime": {
            "nodes": 1,
            "tasks_per_node": 1,
            "gpus_per_node": 8,
            "sqsh_path": "/lustre/test.sqsh",
            "sqsh_sha256": "a" * 64,
        },
        "ptms": [
            {
                "workflow_id": "gcvit_tiny",
                "id": "deformable-detr-gcvit",
            },
            {
                "workflow_id": "resnet50",
                "id": "deformable-detr-resnet50",
            },
        ],
        "integrity": {
            "launcher_sha256": sha256_file(script),
            "manifest_generator_sha256": sha256_file(generator),
        },
    }
    _write_json(
        campaign_manifest,
        {
            **manifest_payload,
            "manifest_sha256": canonical_sha256(manifest_payload),
        },
    )
    return script, marker, campaign_manifest


def _descriptor(
    tmp_path: Path,
    runtime_root: Path,
    *,
    script: Path,
    marker: Path,
    campaign_manifest: Path,
) -> tuple[Path, dict]:
    process_identity = _process_identity(os.getpid())
    assert process_identity is not None
    executable = Path(sys.executable).resolve()
    successor_runtime = marker.parent
    generator = tmp_path / "manifest_generator.py"
    environment_file = tmp_path / "config.env"
    environment_file.write_text(
        "SLURM_HOSTNAME=test.invalid\nSLURM_USER=test\n",
        encoding="utf-8",
    )
    dino_validator = HERE / "dino_campaign" / "manifest_generator.py"
    payload = {
        "schema_version": 1,
        "predecessor": {
            "campaign_id": MANIFEST["campaign_id"],
            "manifest_path": str(MANIFEST_PATH),
            "manifest_file_sha256": sha256_file(MANIFEST_PATH),
            "manifest_sha256": MANIFEST["manifest_sha256"],
            "manifest_validator_path": str(dino_validator),
            "manifest_validator_sha256": sha256_file(dino_validator),
            "runtime_root": str(runtime_root),
            "required_modes": list(MODES),
            "controller_process": process_identity,
        },
        "successor": {
            "name": "direct-full-deformable-detr",
            "campaign_id": "deformable-detr-test-successor",
            "model": "deformable_detr",
            "execution_kind": "direct_full_qualification",
            "cpu_runs": 0,
            "smoke_runs": 0,
            "manifest_path": str(campaign_manifest),
            "manifest_file_sha256": sha256_file(campaign_manifest),
            "manifest_generator_path": str(generator),
            "launcher_path": str(script),
            "runtime_root": str(successor_runtime),
            "working_directory": str(tmp_path),
            "command": [
                str(executable),
                str(script),
                "--manifest",
                str(campaign_manifest),
                "--runtime-root",
                str(successor_runtime),
                "--completion-artifact",
                str(marker),
                "--env-file",
                str(environment_file),
                "--launch",
                "--acknowledge-direct-full-dataset",
            ],
            "environment": {
                "HOME": str(tmp_path),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": str(tmp_path),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            "environment_file": str(environment_file),
            "routing": routing_identity_from_environment_file(
                environment_file
            ),
            "completion_artifact": str(marker),
            "required_files": [
                {
                    "path": str(executable),
                    "sha256": sha256_file(executable),
                },
                {
                    "path": str(script),
                    "sha256": sha256_file(script),
                },
                {
                    "path": str(generator),
                    "sha256": sha256_file(generator),
                },
                {
                    "path": str(campaign_manifest),
                    "sha256": sha256_file(campaign_manifest),
                },
            ],
        },
    }
    descriptor = {
        **payload,
        "descriptor_sha256": canonical_sha256(payload),
    }
    path = tmp_path / "successor.v1.json"
    _write_json(path, descriptor)
    return path, descriptor


def _objective_values(rec_id: int) -> dict[str, float]:
    latency = 50.0 + rec_id
    return {
        "mAP50": 0.40 + 0.02 * rec_id,
        "latency_ms": latency,
        "latency_p95_ms": latency + 1.0,
        "latency_ci95_low_ms": latency - 0.1,
        "latency_ci95_high_ms": latency + 0.1,
    }


def _visible_history(mode: str, rec_id: int) -> list[dict]:
    values = []
    for previous in range(rec_id):
        specs = {"model.enc_layers": 3 + previous % 4}
        values.append(
            {
                "candidate_id": str(previous),
                "candidate_fingerprint": (
                    __import__(
                        "tao_automl.selection",
                        fromlist=["canonical_spec_fingerprint"],
                    ).canonical_spec_fingerprint(specs)
                ),
                "status": "success",
                "result": _objective_values(previous)["mAP50"],
                "objective_score": None,
                "objective_values": _objective_values(previous),
                "failure_reason": None,
            }
        )
    return values


def _recommendation_audit(mode: str, rec_id: int) -> dict:
    specs = {"model.enc_layers": 3 + rec_id % 4}
    model_based = rec_id >= MANIFEST["search"]["calibration_points"]
    method = (
        EXPECTED_MODEL_BASED_METHOD[mode]
        if model_based
        else "deterministic_low_discrepancy_design"
    )
    stage = "model_based" if model_based else "calibration"
    decision = {
        "stage": stage,
        "active_method": method,
        "mode": mode,
        "uses_raw_objectives": True,
        "selector_score_used": False,
        "observation_count": rec_id,
    }
    if model_based:
        decision["optimization_direction"] = (
            EXPECTED_OPTIMIZATION_DIRECTION[mode]
        )
    return build_recommendation_audit(
        candidate_id=rec_id,
        specs=specs,
        algorithm="bayesian",
        search_seed=MANIFEST["search"]["search_seed"],
        search_space=[{"parameter": "model.enc_layers"}],
        custom_ranges={
            "model.enc_layers": {"valid_min": 3, "valid_max": 6}
        },
        objective_config=_Objectives(mode),
        visible_history=_visible_history(mode, rec_id),
        acquisition={
            "proposal": {
                "stage": stage,
                "acquisition_mode": mode,
                "decision_state": decision,
            }
        },
    )


def _latency_evidence(fingerprint: str, rec_id: int) -> dict:
    protocol = MANIFEST["latency_protocol"]
    contract = {
        "schema_version": 1,
        "warmup_iterations": protocol["warmup_iterations"],
        "timed_iterations": protocol["timed_iterations"],
        "repeated_rounds": protocol["repeated_rounds"],
        "tail_percentile": protocol["tail_percentile"],
        "bootstrap_resamples": protocol["bootstrap_resamples"],
        "bootstrap_confidence_level": protocol[
            "bootstrap_confidence_level"
        ],
        "bootstrap_seed": protocol["bootstrap_seed"],
        "batch_size_per_replica": protocol["batch_size_per_replica"],
        "precision": protocol["precision"],
        "timed_scope": protocol["timed_scope"],
        "input_sha256": protocol["input_sha256"],
        "runtime_sha256": protocol["runtime_sha256"],
        "expected_replicas": protocol["expected_replicas"],
        "measurement_role": protocol["measurement_role"],
        "synchronization": protocol["synchronization"],
        "validity_thresholds": protocol["validity_thresholds"],
    }
    runtime_contract = {
        **MANIFEST["runtime"]["hardware_contract"],
        "python": "3.12.0",
        "torch": "2.8.0",
        "cuda": "12.8",
        "cudnn": 91002,
        "tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }
    latency = _objective_values(rec_id)["latency_ms"]
    statistics = {
        "median_ms": latency,
        "p95_ms": latency + 1.0,
        "mad_ms": 0.05,
        "iqr_ms": 0.10,
        "robust_cv": 0.001,
        "bootstrap_median_ci_ms": [latency - 0.1, latency + 0.1],
        "round_median_range_ms": 0.1,
        "round_drift_ms": 0.01,
        "device_median_range_ms": 0.1,
        "raw_sample_count_total": 4000,
        "samples_per_device": 500,
        "is_valid": True,
        "invalid_reasons": [],
    }
    aggregate_payload = {
        "schema_version": 1,
        "contract": contract,
        "contract_sha256": canonical_sha256(contract),
        "candidate_fingerprint": fingerprint,
        "hardware_sha256": canonical_sha256(runtime_contract),
        "replica_record_sha256": [
            f"{rank + 1:064x}" for rank in range(8)
        ],
        "statistics": statistics,
        "selection_isolation": SELECTION_TIME_ISOLATION,
    }
    aggregate = {
        **aggregate_payload,
        "aggregate_sha256": canonical_sha256(aggregate_payload),
    }
    outer_payload = {
        "schema_version": 1,
        "aggregate": aggregate,
        "input_evidence_sha256": "b" * 64,
        "rank_runtime_evidence": [
            {
                "hostname": f"node-{rank}",
                "local_rank": rank,
                "nvidia_smi": "redacted-test-evidence",
                **runtime_contract,
            }
            for rank in range(8)
        ],
    }
    return {
        **outer_payload,
        "evidence_sha256": canonical_sha256(outer_payload),
    }


def _candidate(mode: str, rec_id: int) -> dict:
    from tao_automl.selection import canonical_spec_fingerprint

    specs = {"model.enc_layers": 3 + rec_id % 4}
    fingerprint = canonical_spec_fingerprint(specs)
    return {
        "candidate_id": f"{mode}_rec_{rec_id}",
        "rec_id": str(rec_id),
        "specs": specs,
        "recommendation_audit": _recommendation_audit(mode, rec_id),
        "agent_intervention_flags": algorithmic_campaign_flags(),
        "status": "success",
        "train_job_id": f"{mode}-train-{rec_id}",
        "candidate_fingerprint": fingerprint,
        "objective_values": _objective_values(rec_id),
        "selection_time_latency": {
            "aggregate_evidence": _latency_evidence(
                fingerprint,
                rec_id,
            )
        },
        "matched_validation_selection_isolation_flags": dict(
            MANIFEST["selection_isolation_flags"]
        ),
    }


def _completed_runtime(runtime_root: Path, descriptor: dict) -> None:
    budget = MANIFEST["search"]["candidate_budget_per_mode"]
    _write_json(
        runtime_root / "mode_process_status.json",
        {mode: 0 for mode in MODES},
    )
    for mode in MODES:
        candidates = {
            f"{mode}_rec_{rec_id}": _candidate(mode, rec_id)
            for rec_id in range(budget)
        }
        history = [
            {
                "rec_id": rec_id,
                "specs": candidates[f"{mode}_rec_{rec_id}"]["specs"],
                "job_id": f"{mode}-train-{rec_id}",
                "metric": _objective_values(rec_id)["mAP50"],
                "objective_score": None,
                "objective_values": _objective_values(rec_id),
                "status": "success",
                "failure_reason": None,
                "adjustments": [],
                "selection_audit": None,
            }
            for rec_id in range(budget)
        ]
        archive = [
            {
                "id": item["rec_id"],
                "specs": item["specs"],
                "status": item["status"],
                "objective_values": item["objective_values"],
            }
            for item in history
        ]
        analysis = analyze_archive(
            archive,
            _selection_config(MANIFEST, mode),
        ).to_dict()
        winner_id = int(analysis["selections"][mode]["winner_id"])
        result = {
            "best": {
                "rec_id": winner_id,
                "specs": history[winner_id]["specs"],
                "metric_value": history[winner_id]["metric"],
                "objective_score": None,
                "objective_values": history[winner_id]["objective_values"],
                "adjustments": [],
            },
            "progress": {
                "completed": budget,
                "total": budget,
            },
            "history": history,
            "pareto_front": [],
            "selection_analysis": analysis,
        }
        _write_json(
            runtime_root / mode / "result.json",
            {
                "schema_version": 1,
                "manifest_sha256": descriptor["predecessor"][
                    "manifest_sha256"
                ],
                "mode": mode,
                "status": "success",
                "result": result,
            },
        )
        _write_json(
            runtime_root / mode / "candidate_evidence.json",
            {
                "schema_version": 1,
                "manifest_sha256": descriptor["predecessor"][
                    "manifest_sha256"
                ],
                "mode": mode,
                "candidates": candidates,
            },
        )


def _fixture(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    script, marker, campaign_manifest = _successor_files(tmp_path)
    descriptor_path, descriptor = _descriptor(
        tmp_path,
        runtime_root,
        script=script,
        marker=marker,
        campaign_manifest=campaign_manifest,
    )
    return (
        runtime_root,
        script,
        marker,
        campaign_manifest,
        descriptor_path,
        descriptor,
    )


def _failure_history(*, failure_reason: str = "required_eval_fn_failed:test"):
    specs = {"model.enc_layers": 4}
    return {
        "rec_id": 0,
        "specs": specs,
        "job_id": "failed-job",
        "metric": 0,
        "objective_score": 0,
        "objective_values": {"mAP50": 0},
        "status": "failure",
        "failure_reason": failure_reason,
    }


def test_terminal_failure_aligns_with_runner_failure_sentinel():
    history = _failure_history()
    record = {
        "status": "terminal_failure",
        "specs": history["specs"],
        "train_job_id": history["job_id"],
        "failure_reason": history["failure_reason"],
        "automl_status": "failure",
        "reported_metric": None,
    }

    assert (
        _validate_candidate_history_alignment(
            record,
            history,
            candidate_path="accuracy.candidates.accuracy_rec_0",
        )
        is False
    )
    assert _history_snapshot(history) == {
        "candidate_id": "0",
        "candidate_fingerprint": (
            __import__(
                "tao_automl.selection",
                fromlist=["canonical_spec_fingerprint"],
            ).canonical_spec_fingerprint(history["specs"])
        ),
        "status": "failure",
        "objective_values": {"mAP50": 0},
        "failure_reason": history["failure_reason"],
    }


def test_pre_submission_cancellation_is_narrowly_recovered():
    history = _failure_history(failure_reason="job_canceled")
    record = {
        "agent_intervention_flags": algorithmic_campaign_flags(),
        "candidate_id": "accuracy_rec_0",
        "rec_id": "0",
        "recommendation_audit": {},
        "specs": history["specs"],
        "status": "recommended",
    }

    assert (
        _validate_candidate_history_alignment(
            record,
            history,
            candidate_path="accuracy.candidates.accuracy_rec_0",
        )
        is True
    )


def test_recommended_non_cancellation_failure_is_rejected():
    history = _failure_history()
    record = {
        "agent_intervention_flags": algorithmic_campaign_flags(),
        "candidate_id": "accuracy_rec_0",
        "rec_id": "0",
        "recommendation_audit": {},
        "specs": history["specs"],
        "status": "recommended",
    }

    with pytest.raises(
        AutomaticSuccessorError,
        match="not a narrowly recoverable cancellation",
    ):
        _validate_candidate_history_alignment(
            record,
            history,
            candidate_path="accuracy.candidates.accuracy_rec_0",
        )


def test_replay_uses_explicit_retention_only_for_latency_mode():
    assert (
        _selection_config(
            MANIFEST,
            "latency",
        ).latency_accuracy_retention.value
        == 0.90
    )
    assert (
        _selection_config(
            MANIFEST,
            "accuracy",
        ).latency_accuracy_retention.value
        == 0.98
    )
    multi_objective = _selection_config(MANIFEST, "multi_objective")
    assert multi_objective.latency_accuracy_retention.value == 0.98
    assert multi_objective.multi_objective_min_accuracy is None


def test_completed_gate_triggers_successor_exactly_once(tmp_path):
    (
        runtime_root,
        _script,
        marker,
        _campaign_manifest,
        descriptor_path,
        descriptor,
    ) = _fixture(tmp_path)
    _completed_runtime(runtime_root, descriptor)

    report = validate_completed_dino(descriptor, runtime_root)
    assert report["status"] == "passed"
    assert set(report["modes"]) == set(MODES)
    assert all(
        item["model_based_candidate_ids"]
        == [str(index) for index in range(8, 20)]
        for item in report["modes"].values()
    )

    assert (
        watch_and_trigger(
            descriptor_path,
            runtime_root,
            poll_seconds=0.001,
        )
        == 0
    )
    completion = json.loads(marker.read_text())
    assert completion["status"] == "success"
    assert completion["outcomes"] == {
        "gcvit_tiny": "success",
        "resnet50": "success",
    }
    state = json.loads(
        (
            runtime_root
            / "automatic_successor"
            / "automatic_successor_state.json"
        ).read_text()
    )
    assert state["status"] == "successor_completed"
    decision = json.loads(
        (
            runtime_root
            / "automatic_successor"
            / "gate_decision.json"
        ).read_text()
    )
    assert decision["successor_submitted"] is True

    with pytest.raises(
        AutomaticSuccessorError,
        match="already has terminal or running state",
    ):
        trigger_successor(
            descriptor,
            state_dir=runtime_root / "automatic_successor",
            gate_report=report,
        )


def test_waiting_gate_does_not_launch_successor(tmp_path):
    (
        runtime_root,
        _script,
        marker,
        _campaign_manifest,
        descriptor_path,
        _descriptor_value,
    ) = _fixture(tmp_path)
    runtime_root.mkdir()

    assert (
        watch_and_trigger(
            descriptor_path,
            runtime_root,
            poll_seconds=0.001,
            once=True,
        )
        == 3
    )
    assert not marker.exists()


def test_dead_controller_blocks_before_terminal_marker(tmp_path):
    (
        runtime_root,
        _script,
        marker,
        _campaign_manifest,
        descriptor_path,
        descriptor,
    ) = _fixture(tmp_path)
    runtime_root.mkdir()
    descriptor["predecessor"]["controller_process"]["start_time_ticks"] += 1
    payload = {
        key: value
        for key, value in descriptor.items()
        if key != "descriptor_sha256"
    }
    descriptor["descriptor_sha256"] = canonical_sha256(payload)
    _write_json(descriptor_path, descriptor)

    with pytest.raises(
        AutomaticSuccessorError,
        match="controller exited or its sealed process identity changed",
    ):
        watch_and_trigger(
            descriptor_path,
            runtime_root,
            poll_seconds=0.001,
            once=True,
        )
    assert not marker.exists()


def test_selector_policy_tampering_blocks_without_launch(tmp_path):
    (
        runtime_root,
        _script,
        marker,
        _campaign_manifest,
        descriptor_path,
        descriptor,
    ) = _fixture(tmp_path)
    _completed_runtime(runtime_root, descriptor)
    result_path = runtime_root / "latency" / "result.json"
    result = json.loads(result_path.read_text())
    result["result"]["selection_analysis"]["selections"]["latency"][
        "winner_id"
    ] = "19"
    result["result"]["best"]["rec_id"] = 19
    _write_json(result_path, result)

    with pytest.raises(
        AutomaticSuccessorError,
        match="selector evidence differs from production replay",
    ):
        watch_and_trigger(
            descriptor_path,
            runtime_root,
            poll_seconds=0.001,
        )
    assert not marker.exists()


def test_latency_provenance_tampering_blocks_without_launch(tmp_path):
    (
        runtime_root,
        _script,
        marker,
        _campaign_manifest,
        descriptor_path,
        descriptor,
    ) = _fixture(tmp_path)
    _completed_runtime(runtime_root, descriptor)
    evidence_path = runtime_root / "accuracy" / "candidate_evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["candidates"]["accuracy_rec_0"]["objective_values"][
        "latency_ms"
    ] += 1.0
    _write_json(evidence_path, evidence)

    with pytest.raises(
        AutomaticSuccessorError,
        match="differs from runner history",
    ):
        watch_and_trigger(
            descriptor_path,
            runtime_root,
            poll_seconds=0.001,
        )
    assert not marker.exists()


def test_post_calibration_fallback_blocks_without_launch(tmp_path):
    (
        runtime_root,
        _script,
        marker,
        _campaign_manifest,
        descriptor_path,
        descriptor,
    ) = _fixture(tmp_path)
    _completed_runtime(runtime_root, descriptor)
    evidence_path = runtime_root / "accuracy" / "candidate_evidence.json"
    evidence = json.loads(evidence_path.read_text())
    audit = evidence["candidates"]["accuracy_rec_8"][
        "recommendation_audit"
    ]
    proposal = audit["acquisition"]["proposal"]
    proposal["stage"] = "calibration"
    proposal["decision_state"]["stage"] = "calibration"
    payload = {
        key: value for key, value in audit.items() if key != "audit_sha256"
    }
    audit["audit_sha256"] = canonical_audit_sha256(payload)
    _write_json(evidence_path, evidence)

    with pytest.raises(
        AutomaticSuccessorError,
        match="reverted from objective-aware model-based acquisition",
    ):
        watch_and_trigger(
            descriptor_path,
            runtime_root,
            poll_seconds=0.001,
        )
    assert not marker.exists()


def test_tampered_successor_input_is_rejected(tmp_path):
    (
        runtime_root,
        script,
        marker,
        _campaign_manifest,
        descriptor_path,
        descriptor,
    ) = _fixture(tmp_path)
    _completed_runtime(runtime_root, descriptor)
    script.write_text("# changed after sealing\n", encoding="utf-8")

    with pytest.raises(
        AutomaticSuccessorError,
        match="successor input identity changed",
    ):
        watch_and_trigger(
            descriptor_path,
            runtime_root,
            poll_seconds=0.001,
        )
    assert not marker.exists()


def test_changed_slurm_routing_is_rejected(tmp_path):
    (
        runtime_root,
        _script,
        marker,
        _campaign_manifest,
        descriptor_path,
        descriptor,
    ) = _fixture(tmp_path)
    _completed_runtime(runtime_root, descriptor)
    environment_file = Path(descriptor["successor"]["environment_file"])
    environment_file.write_text(
        "SLURM_HOSTNAME=redirected.invalid\nSLURM_USER=test\n",
        encoding="utf-8",
    )

    with pytest.raises(
        AutomaticSuccessorError,
        match="routing identity changed after sealing",
    ):
        watch_and_trigger(
            descriptor_path,
            runtime_root,
            poll_seconds=0.001,
        )
    assert not marker.exists()


def test_successor_command_must_consume_sealed_manifest(tmp_path):
    (
        _runtime_root,
        _script,
        _marker,
        _campaign_manifest,
        descriptor_path,
        descriptor,
    ) = _fixture(tmp_path)
    command = descriptor["successor"]["command"]
    command[command.index("--manifest") + 1] = str(tmp_path / "other.json")
    payload = {
        key: value
        for key, value in descriptor.items()
        if key != "descriptor_sha256"
    }
    descriptor["descriptor_sha256"] = canonical_sha256(payload)
    _write_json(descriptor_path, descriptor)

    with pytest.raises(
        AutomaticSuccessorError,
        match="does not consume its sealed manifest",
    ):
        validate_successor_descriptor(descriptor_path)


def test_successor_descriptor_cannot_embed_credentials(tmp_path):
    (
        _runtime_root,
        _script,
        _marker,
        _campaign_manifest,
        descriptor_path,
        descriptor,
    ) = _fixture(tmp_path)
    descriptor["successor"]["environment"]["NGC_KEY"] = "not-allowed"
    payload = {
        key: value
        for key, value in descriptor.items()
        if key != "descriptor_sha256"
    }
    descriptor["descriptor_sha256"] = canonical_sha256(payload)
    _write_json(descriptor_path, descriptor)

    with pytest.raises(
        AutomaticSuccessorError,
        match="must not embed credential variables",
    ):
        validate_successor_descriptor(descriptor_path)


def test_zero_exit_contract_requires_fresh_valid_completion(tmp_path):
    (
        runtime_root,
        _script,
        marker,
        _campaign_manifest,
        descriptor_path,
        descriptor,
    ) = _fixture(tmp_path)
    _completed_runtime(runtime_root, descriptor)
    descriptor["successor"]["command"].append("--skip-completion")
    payload = {
        key: value
        for key, value in descriptor.items()
        if key != "descriptor_sha256"
    }
    descriptor["descriptor_sha256"] = canonical_sha256(payload)
    _write_json(descriptor_path, descriptor)

    with pytest.raises(
        AutomaticSuccessorError,
        match="without a regular completion artifact",
    ):
        watch_and_trigger(
            descriptor_path,
            runtime_root,
            poll_seconds=0.001,
        )
    assert not marker.exists()
    state = json.loads(
        (
            runtime_root
            / "automatic_successor"
            / "automatic_successor_state.json"
        ).read_text()
    )
    assert state["status"] == "successor_completion_invalid"


def test_fresh_trigger_rejects_preexisting_completion(tmp_path):
    (
        runtime_root,
        _script,
        marker,
        _campaign_manifest,
        descriptor_path,
        descriptor,
    ) = _fixture(tmp_path)
    _completed_runtime(runtime_root, descriptor)
    _write_json(marker, {"stale": True})

    with pytest.raises(
        AutomaticSuccessorError,
        match="refuses a pre-existing completion artifact",
    ):
        watch_and_trigger(
            descriptor_path,
            runtime_root,
            poll_seconds=0.001,
        )


def test_self_hashed_inconsistent_completion_is_rejected(tmp_path):
    (
        runtime_root,
        _script,
        marker,
        _campaign_manifest,
        descriptor_path,
        descriptor,
    ) = _fixture(tmp_path)
    _completed_runtime(runtime_root, descriptor)
    assert (
        watch_and_trigger(
            descriptor_path,
            runtime_root,
            poll_seconds=0.001,
        )
        == 0
    )
    completion = json.loads(marker.read_text())
    completion["successful_workflows"] = 1
    payload = {
        key: value
        for key, value in completion.items()
        if key != "completion_sha256"
    }
    completion["completion_sha256"] = canonical_sha256(payload)
    _write_json(marker, completion)

    with pytest.raises(
        AutomaticSuccessorError,
        match="counts or aggregate status are inconsistent",
    ):
        validate_successor_completion(descriptor)


def test_descriptor_is_content_addressed(tmp_path):
    (
        _runtime_root,
        _script,
        _marker,
        _campaign_manifest,
        descriptor_path,
        descriptor,
    ) = _fixture(tmp_path)
    descriptor["successor"]["name"] = "result-driven-change"
    _write_json(descriptor_path, descriptor)

    with pytest.raises(
        AutomaticSuccessorError,
        match="descriptor integrity verification failed",
    ):
        validate_successor_descriptor(descriptor_path)
