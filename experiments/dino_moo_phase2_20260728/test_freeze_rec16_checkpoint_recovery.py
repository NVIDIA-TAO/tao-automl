"""Focused fail-closed tests for rec16 checkpoint-recovery freezing."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "freeze_rec16_checkpoint_recovery.py"
SPEC = importlib.util.spec_from_file_location(
    "freeze_rec16_checkpoint_recovery_test_module",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
freezer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = freezer
SPEC.loader.exec_module(freezer)


class FakeProbe:
    """Exact in-memory projection of the frozen remote evidence."""

    def __init__(self) -> None:
        self.scheduler = {}
        historical = freezer.HISTORICAL
        self.scheduler[historical["slurm_job_id"]] = {
            "slurm_job_id": historical["slurm_job_id"],
            "state": "COMPLETED",
            "exit_code": "0:0",
            "node": historical["node"],
            "submit_time_utc": historical["submit_time_utc"],
            "start_time_utc": historical["start_time_utc"],
            "end_time_utc": historical["end_time_utc"],
        }
        for contract in freezer.ATTEMPT_CONTRACTS:
            self.scheduler[contract["slurm_job_id"]] = {
                key: contract[key]
                for key in (
                    "slurm_job_id",
                    "state",
                    "exit_code",
                    "node",
                    "submit_time_utc",
                    "start_time_utc",
                    "end_time_utc",
                )
            }
        self.pending = {
            "slurm_job_id": freezer.SUPPLEMENTARY["slurm_job_id"],
            "state": "PENDING",
            "exit_code": "0:0",
            "node": "None assigned",
            "submit_time_utc": freezer.SUPPLEMENTARY["submit_time_utc"],
            "reason": "(Resources)",
            "squeue_node": "",
        }
        self.files = {}
        for source in (
            freezer.HISTORICAL,
            *freezer.ATTEMPT_CONTRACTS,
            freezer.SUPPLEMENTARY,
        ):
            for label in ("entrypoint", "sbatch", "remote_specs"):
                identity = source[label]
                self.files[identity["path"]] = copy.deepcopy(identity)
        for contract in freezer.ATTEMPT_CONTRACTS:
            identity = contract["checkpoint"]
            self.files[identity["path"]] = copy.deepcopy(identity)
        self.historical_absent = True
        self.node_line_present = True

    def scheduler_record(self, slurm_job_id: str) -> dict[str, str]:
        return copy.deepcopy(self.scheduler[slurm_job_id])

    def pending_record(self, slurm_job_id: str) -> dict[str, str]:
        assert slurm_job_id == freezer.SUPPLEMENTARY["slurm_job_id"]
        return copy.deepcopy(self.pending)

    def file_identity(self, path: str) -> dict[str, object]:
        return copy.deepcopy(self.files[path])

    def path_absent(self, path: str) -> bool:
        assert path == freezer.HISTORICAL["checkpoint"]["path"]
        return self.historical_absent

    def file_contains_line(self, path: str, line: str) -> bool:
        assert path == freezer.SUPPLEMENTARY["sbatch"]["path"]
        assert line == (
            "#SBATCH --nodelist="
            + freezer.SUPPLEMENTARY["expected_node"]
        )
        return self.node_line_present


def build_fixture(probe: FakeProbe | None = None) -> dict:
    return freezer.build_evidence(
        probe or FakeProbe(),
        {"fixture_source_identity": True},
        frozen_at_utc="2026-07-28T19:30:00Z",
    )


def test_local_source_identity_reconstructs_frozen_spec() -> None:
    source = freezer.validate_local_source_identity()

    assert source["candidate_specs"] == freezer.EXPECTED_CANDIDATE_SPECS
    assert source["train_spec_sha256"] == freezer.EXPECTED_TRAIN_SPEC_SHA256
    assert source["model_spec_sha256"] == freezer.EXPECTED_MODEL_SPEC_SHA256
    assert source["command_sha256"] == freezer.EXPECTED_COMMAND_SHA256
    assert (
        source["seed_archive"]["whole_file_sha256"]
        == freezer.EXPECTED_SEED_ARCHIVE_SHA256
    )


def test_build_evidence_selects_earliest_attempt_without_values() -> None:
    evidence = build_fixture()

    assert [item["submission_index"] for item in evidence["recovery_attempts"]] == [
        0,
        1,
    ]
    assert (
        evidence["selected_recovery"]["tao_job_id"]
        == freezer.ATTEMPT_CONTRACTS[0]["tao_job_id"]
    )
    assert evidence["selection_policy"]["value_independent"] is True
    assert evidence["selection_policy"]["checkpoint_hash_used"] is False
    assert evidence["selection_policy"]["objective_value_used"] is False
    assert (
        evidence["selected_recovery"]["historical_checkpoint_sha256_match"]
        is False
    )
    assert (
        evidence["selected_recovery"]["configuration_exact_not_byte_identical"]
        is True
    )
    assert len(evidence["selection_isolation"]) == 5
    assert set(evidence["selection_isolation"].values()) == {False}
    unhashed = copy.deepcopy(evidence)
    claimed = unhashed.pop("evidence_sha256")
    assert freezer.sha256_value(unhashed) == claimed


def test_selection_key_uses_numeric_slurm_id_then_tao_id() -> None:
    timestamp = "2026-07-28T12:00:00Z"
    attempts = [
        {
            "submit_time_utc": timestamp,
            "slurm_job_id": "10",
            "tao_job_id": "a",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "exact_config": True,
        },
        {
            "submit_time_utc": timestamp,
            "slurm_job_id": "9",
            "tao_job_id": "z",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "exact_config": True,
        },
    ]

    assert freezer.choose_recovery(attempts)["slurm_job_id"] == "9"

    attempts[0]["slurm_job_id"] = "9"
    attempts[0]["tao_job_id"] = "a"
    attempts[1]["tao_job_id"] = "z"
    assert freezer.choose_recovery(attempts)["tao_job_id"] == "a"


def test_checkpoint_hash_drift_fails_closed() -> None:
    probe = FakeProbe()
    checkpoint_path = freezer.ATTEMPT_CONTRACTS[0]["checkpoint"]["path"]
    probe.files[checkpoint_path]["sha256"] = "0" * 64

    with pytest.raises(freezer.FreezeError, match="checkpoint file identity"):
        build_fixture(probe)


def test_historical_checkpoint_must_remain_missing() -> None:
    probe = FakeProbe()
    probe.historical_absent = False

    with pytest.raises(
        freezer.FreezeError,
        match="historical checkpoint unexpectedly exists",
    ):
        build_fixture(probe)


def test_supplementary_replay_must_be_pending_and_node_pinned() -> None:
    probe = FakeProbe()
    probe.pending["state"] = "RUNNING"
    with pytest.raises(
        freezer.FreezeError,
        match="supplementary exact-node scheduler identity",
    ):
        build_fixture(probe)

    probe = FakeProbe()
    probe.pending["reason"] = "(Priority)"
    evidence = build_fixture(probe)
    assert (
        evidence["supplementary_exact_node_replay"]["pending_reason"]
        == "(Priority)"
    )

    probe = FakeProbe()
    probe.node_line_present = False
    with pytest.raises(
        freezer.FreezeError,
        match="lacks its exact node pin",
    ):
        build_fixture(probe)


def test_evidence_validation_rejects_selection_isolation_drift() -> None:
    evidence = build_fixture()
    evidence["selection_isolation"]["measurements_feed_selection"] = True
    evidence["evidence_sha256"] = freezer.sha256_value(
        {
            key: value
            for key, value in evidence.items()
            if key != "evidence_sha256"
        }
    )

    with pytest.raises(
        freezer.FreezeError,
        match="selection-isolation flags",
    ):
        freezer.validate_evidence(evidence)


def test_immutable_writer_refuses_overwrite(tmp_path: Path) -> None:
    evidence = build_fixture()
    output = tmp_path / "recovery.json"

    freezer.write_new_evidence(output, evidence)
    assert json.loads(output.read_text(encoding="utf-8")) == evidence

    with pytest.raises(freezer.FreezeError, match="refusing to overwrite"):
        freezer.write_new_evidence(output, evidence)
