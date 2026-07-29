"""Hermetic fail-closed tests for the frozen rec6 recovery launcher."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "rec6_checkpoint_recovery.py"
SPEC = importlib.util.spec_from_file_location(
    "rec6_checkpoint_recovery_test_module",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
recovery = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = recovery
SPEC.loader.exec_module(recovery)


def test_contract_reconstructs_the_sealed_candidate_exactly() -> None:
    _runner, manifest, command, contract = recovery.build_contract()

    assert manifest["manifest_id"] == "dino_expanded_search_20260728_v2"
    assert recovery.sha256_file(recovery.ARCHIVE_PATH) == (
        recovery.EXPECTED_ARCHIVE_SHA256
    )
    assert contract["candidate_id"] == "seed_271828_rec_6"
    assert contract["reconstruction"]["candidate_specs"] == (
        recovery.EXPECTED_SPECS
    )
    assert contract["reconstruction"]["training_seed"] == 1234
    assert contract["reconstruction"]["train_epochs"] == 10
    assert contract["reconstruction"]["num_nodes"] == 1
    assert contract["reconstruction"]["gpus_per_node"] == 8
    assert contract["reconstruction"]["gpu_ids"] == list(range(8))
    assert contract["reconstruction"]["precision"] == "fp32"
    assert contract["reconstruction"]["distributed_strategy"] == "ddp"
    assert contract["reconstruction"]["command_sha256"] == (
        recovery.EXPECTED_COMMAND_SHA256
    )
    assert len(command.encode("utf-8")) == recovery.EXPECTED_COMMAND_SIZE_BYTES
    assert contract["historical_training"]["node"] == (
        "batch-block7-01453"
    )
    assert contract["acceptance"] == {
        "accuracy_or_latency_equivalence_not_assumed": True,
        "byte_identical_checkpoint_not_assumed": True,
        "configuration_exact_reconstruction": True,
        "recovery_is_validation_only": True,
    }
    assert set(contract["selection_isolation"].values()) == {False}
    assert (
        contract["retention_policy"]["historical_archive_replacement_permitted"]
        is False
    )


def test_preregistration_is_candidate_specific_and_selection_isolated() -> None:
    preregistration, digest = recovery.load_preregistration()

    assert len(digest) == 64
    assert preregistration["candidate_id"] == recovery.CANDIDATE_ID
    assert preregistration["selection_isolation"] == (
        recovery.SELECTION_ISOLATION
    )
    assert (
        preregistration["retention_and_identity_policy"][
            "byte_identical_checkpoint_assumed"
        ]
        is False
    )
    assert (
        preregistration["submission_policy"]["manual_candidate_injection"]
        is False
    )


def test_archive_record_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = json.loads(recovery.ARCHIVE_PATH.read_text(encoding="utf-8"))
    modified = copy.deepcopy(archive)
    modified["records"][recovery.CANDIDATE_ID]["specs"][
        "model.enc_layers"
    ] = 3
    modified_path = tmp_path / "seed_archive.v1.json"
    modified_path.write_text(
        json.dumps(modified, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(recovery, "ARCHIVE_PATH", modified_path)
    monkeypatch.setattr(
        recovery,
        "EXPECTED_ARCHIVE_SHA256",
        recovery.sha256_file(modified_path),
    )

    with pytest.raises(
        recovery.RecoveryError,
        match="candidate-record digest drift",
    ):
        recovery.build_contract()


def test_manifest_digest_drift_fails_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(recovery, "EXPECTED_RUNNER_SHA256", "0" * 64)

    with pytest.raises(
        recovery.RecoveryError,
        match="expanded_search_runner.py digest drift",
    ):
        recovery.build_contract()


class FakeRemoteRunner:
    def __init__(self, result: str) -> None:
        self.result = result
        self.commands: list[tuple[str, int]] = []

    def remote_output(self, command: str, *, timeout: int) -> str:
        self.commands.append((command, timeout))
        return self.result


def test_missing_checkpoint_is_a_launch_precondition() -> None:
    runner = FakeRemoteRunner("MISSING\n")

    evidence = recovery.verify_historical_checkpoint_missing(runner)

    assert evidence == {
        "path": recovery.EXPECTED_HISTORICAL_CHECKPOINT_PATH,
        "observed": "missing",
        "verified": True,
    }
    assert runner.commands[0][1] == 120
    assert recovery.EXPECTED_HISTORICAL_CHECKPOINT_PATH in (
        runner.commands[0][0]
    )


@pytest.mark.parametrize("remote_result", ["PRESENT\n", "", "unexpected\n"])
def test_present_or_ambiguous_checkpoint_probe_fails_closed(
    remote_result: str,
) -> None:
    runner = FakeRemoteRunner(remote_result)

    with pytest.raises(
        recovery.RecoveryError,
        match="refusing duplicate recovery",
    ):
        recovery.verify_historical_checkpoint_missing(runner)


def test_atomic_writer_rejects_stale_pending_file(tmp_path: Path) -> None:
    output = tmp_path / "dry_run.json"
    pending = output.with_name(f".{output.name}.pending")
    pending.write_text("stale", encoding="utf-8")

    with pytest.raises(recovery.RecoveryError, match="stale pending artifact"):
        recovery.atomic_json(output, {"ok": True})


def complete_status() -> dict:
    checkpoint_path = (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/"
        "results/recovered-tao-job/results_dir/train/"
        "model_epoch_009_step_00440.pth"
    )
    artifact = lambda name: {
        "path": f"/remote/{name}",
        "sha256": "b" * 64,
        "size_bytes": 123,
    }
    return {
        "schema_version": 1,
        "candidate_id": recovery.CANDIDATE_ID,
        "tao_job_id": "recovered-tao-job",
        "slurm_job_id": "31010000",
        "sdk_status": "Complete",
        "sdk_message": "complete",
        "slurm_accounting": {
            "slurm_job_id": "31010000",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "node": "batch-block7-00001",
            "submit_time_utc": "2026-07-28T20:00:00Z",
            "start_time_utc": "2026-07-28T20:00:10Z",
            "end_time_utc": "2026-07-28T20:07:00Z",
            "complete": True,
        },
        "result_root": (
            "/lustre/fs11/portfolios/edgeai/projects/"
            "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/"
            "results/recovered-tao-job"
        ),
        "complete": True,
        "terminal": True,
        "checkpoint": {
            "path": checkpoint_path,
            "sha256": "a" * 64,
            "size_bytes": recovery.EXPECTED_HISTORICAL_CHECKPOINT_SIZE,
            "epoch": 9,
        },
        "launch_artifacts": {
            "entrypoint": artifact("entrypoint"),
            "sbatch": artifact("sbatch"),
            "remote_specs": artifact("remote_specs"),
        },
        "historical_checkpoint_missing": {
            "path": recovery.EXPECTED_HISTORICAL_CHECKPOINT_PATH,
            "observed": "missing",
            "verified": True,
        },
        "selection_isolation": recovery.SELECTION_ISOLATION,
        "inspected_at_utc": "2026-07-28T20:08:00Z",
    }


def test_final_evidence_is_consumable_by_matched_launcher(
    tmp_path: Path,
) -> None:
    _runner, _manifest, _command, contract = recovery.build_contract()
    evidence = recovery.build_recovery_evidence(
        complete_status(),
        contract,
    )
    core = {
        key: value
        for key, value in evidence.items()
        if key != "artifact_integrity"
    }
    assert evidence["artifact_integrity"]["canonical_payload_sha256"] == (
        recovery.sha256_value(core)
    )
    assert evidence["checkpoint_identity"][
        "byte_identical_checkpoint_assumed"
    ] is False
    assert evidence["checkpoint_identity"][
        "historical_sha256_match"
    ] is False
    assert evidence["recovery_provenance"]["node"] == (
        "batch-block7-00001"
    )

    matched_path = HERE / "latency_90_policy_matched_launcher.py"
    matched_spec = importlib.util.spec_from_file_location(
        "latency_90_policy_matched_for_recovery_test",
        matched_path,
    )
    assert matched_spec is not None and matched_spec.loader is not None
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    matched = importlib.util.module_from_spec(matched_spec)
    matched_spec.loader.exec_module(matched)
    projection = json.loads(
        (
            HERE
            / "latency_90_policy"
            / "matched"
            / "execution_projection.v1.json"
        ).read_text(encoding="utf-8")
    )
    evidence_path = tmp_path / "recovery_evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    loaded, binding = matched.load_recovery_evidence(
        projection,
        evidence_path,
        matched.file_sha256(evidence_path),
    )

    assert loaded == evidence
    assert binding["internal_sha256"] == (
        evidence["artifact_integrity"]["canonical_payload_sha256"]
    )


def test_incomplete_or_failed_recovery_cannot_finalize() -> None:
    _runner, _manifest, _command, contract = recovery.build_contract()
    status = complete_status()
    status["complete"] = False
    with pytest.raises(recovery.RecoveryError, match="not SDK-and-SLURM"):
        recovery.build_recovery_evidence(status, contract)

    status = complete_status()
    status["slurm_accounting"]["exit_code"] = "1:0"
    with pytest.raises(
        recovery.RecoveryError,
        match="SLURM completion evidence is invalid",
    ):
        recovery.build_recovery_evidence(status, contract)


def test_existing_evidence_must_match_live_checkpoint() -> None:
    _runner, _manifest, _command, contract = recovery.build_contract()
    status = complete_status()
    evidence = recovery.build_recovery_evidence(status, contract)
    recovery.validate_existing_evidence(evidence, status)

    status["checkpoint"]["sha256"] = "c" * 64
    with pytest.raises(
        recovery.RecoveryError,
        match="differs from live provenance",
    ):
        recovery.validate_existing_evidence(evidence, status)


def test_scheduler_accounting_requires_exact_successful_top_level_row() -> None:
    runner = FakeRemoteRunner(
        "31010000|COMPLETED|0:0|batch-block7-00001|"
        "2026-07-28T20:00:00|2026-07-28T20:00:10|"
        "2026-07-28T20:07:00\n"
    )

    record = recovery.scheduler_accounting(runner, "31010000")

    assert record["complete"] is True
    assert record["node"] == "batch-block7-00001"
    assert record["submit_time_utc"] == "2026-07-28T20:00:00Z"
    assert record["end_time_utc"] == "2026-07-28T20:07:00Z"


def test_remote_file_identity_is_fail_closed() -> None:
    path = "/remote/model_epoch_009_step_00440.pth"
    runner = FakeRemoteRunner(f"475869698\n{'a' * 64}  {path}\n")
    assert recovery.remote_file_identity(runner, path) == {
        "path": path,
        "sha256": "a" * 64,
        "size_bytes": 475869698,
    }

    wrong = FakeRemoteRunner(f"1\n{'a' * 64}  /different\n")
    with pytest.raises(
        recovery.RecoveryError,
        match="different path",
    ):
        recovery.remote_file_identity(wrong, path)
