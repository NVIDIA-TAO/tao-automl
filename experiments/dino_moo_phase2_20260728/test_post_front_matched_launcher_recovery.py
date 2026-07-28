"""Hermetic crash-recovery and security tests for the matched-front launcher.

The tests in this module intentionally use only temporary runtime directories
and an in-memory fake of ``tao_sdk.platforms.slurm.SlurmSDK``.  They never
generate a real manifest, contact a remote host, submit a scheduler job, or
read the user's secrets file.
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import post_front_matched_launcher as launcher  # noqa: E402
import post_front_matched_manifest_generator as generator  # noqa: E402


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
MANIFEST_FILE_SHA256 = "d" * 64
BASE_SOURCE_CHECKS = {"launcher_source": {"sha256": SHA_A}}
CONTAINMENT = {
    "base_mode": "0700",
    "secret_values_recorded": False,
    "evidence_sha256": SHA_C,
}


def _manifest(runtime_dir: Path) -> dict[str, Any]:
    candidate_ids = ["candidate_a", "candidate_b"]
    schedule = generator.build_schedule(candidate_ids)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_id": "synthetic_post_front_recovery_v1",
        "candidate_derivation": {
            "candidate_ids": candidate_ids,
            "candidate_set_sha256": generator.sha256_value(candidate_ids),
        },
        "schedule": schedule,
        "runtime": {
            "local_runtime_path": str(runtime_dir.resolve()),
            "sdk_path": "/synthetic/tao-sdk",
            "sqsh_path": "/synthetic/dino.sqsh",
            "partition": "synthetic_partition",
            "account": "synthetic_account",
            "slurm_time_hours": 4.0,
            "slurm_timeout_hours": 3.8,
        },
        "incomplete_allocation_policy": (
            "Exclude the entire allocation and rerun the complete front "
            "under a new TAO job ID; never combine a partial block."
        ),
        "source_artifacts": {
            "post_front_tools": {
                launcher.AGGREGATOR.name: {
                    "path": str(launcher.AGGREGATOR),
                    "sha256": SHA_A,
                    "git_blob": "0" * 40,
                    "head_git_blob": "0" * 40,
                }
            }
        },
    }
    manifest["manifest_sha256"] = generator.sha256_value(manifest)
    return manifest


def _commands(
    manifest: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    result: list[tuple[str, dict[str, Any]]] = []
    for allocation in manifest["schedule"]["allocations"]:
        command = f"synthetic-command-{allocation['allocation_index']}"
        summary = {
            "allocation_id": allocation["allocation_id"],
            "allocation_index": allocation["allocation_index"],
            "design_row_index": allocation["design_row_index"],
            "candidate_order": copy.deepcopy(
                allocation["candidate_order"]
            ),
            "candidate_count": len(allocation["candidate_order"]),
            "block_plan_sha256": SHA_A,
            "command_sha256": launcher.sha256_bytes(
                command.encode("utf-8")
            ),
            "staging_bundle_sha256": SHA_A,
            "staging_bundle_json_sha256": SHA_B,
            "staging_file_sha256": {
                "configs/candidate_a.yaml": SHA_A,
                "configs/candidate_b.yaml": SHA_B,
            },
        }
        result.append((command, summary))
    return result


def _write_dry_run(
    runtime_dir: Path,
    manifest: dict[str, Any],
) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "dry_run.json").write_text(
        json.dumps(
            {
                "status": "dry_run_validated_not_launched",
                "manifest": {
                    "whole_file_sha256": MANIFEST_FILE_SHA256,
                },
                "schedule_sha256": manifest["schedule"]["schedule_sha256"],
                "candidate_ids": manifest["candidate_derivation"][
                    "candidate_ids"
                ],
                "submission_ready": True,
            }
        ),
        encoding="utf-8",
    )


def _write_launch_binding(
    runtime_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    contract = launcher.launch_contract_payload(
        manifest,
        MANIFEST_FILE_SHA256,
        runtime_dir,
        BASE_SOURCE_CHECKS,
    )
    path = runtime_dir / launcher.LAUNCH_CONTRACT_NAME
    launcher.atomic_json(path, contract)
    return {
        **copy.deepcopy(BASE_SOURCE_CHECKS),
        "launch_contract": {
            "path": str(path),
            "whole_file_sha256": generator.sha256_file(path),
            "internal_sha256": contract["contract_sha256"],
        },
    }


def _record(
    summary: dict[str, Any],
    *,
    tao_job_id: str,
    slurm_job_id: str,
) -> dict[str, Any]:
    return {
        **copy.deepcopy(summary),
        "tao_job_id": tao_job_id,
        "slurm_job_id": slurm_job_id,
        "retry_count": 0,
        "failed_slurm_job_ids": [],
        "launch_uncertain": False,
        "sdk_results_uri": f"lustre:///results/{tao_job_id}",
        "remote_entrypoint_sha256": SHA_C,
        "remote_sdk_containment": copy.deepcopy(CONTAINMENT),
        "feeds_final_selection": False,
        "feeds_reselection": False,
    }


class _FakeState:
    """Durable in-memory SDK state shared by successive SDK instances."""

    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        self.identities: dict[str, dict[str, Any]] = {}
        self.remote_files: dict[str, str] = {}
        self.commands: dict[str, str] = {}
        self.create_attempt = 0
        self.fail_before_attempts: set[int] = set()
        self.fail_after_attempts: set[int] = set()
        self.events: list[tuple[str, str]] = []

    @staticmethod
    def entrypoint(command: str) -> str:
        return f"#!/bin/sh\n{command}\n"

    def add_job(
        self,
        *,
        command: str,
        job_id: str,
        slurm_job_id: str,
        image: str = "/synthetic/dino.sqsh",
        status: str = "Running",
        launch_uncertain: bool = False,
        submission_attempted: bool = True,
        entrypoint_content: str | None = None,
    ) -> None:
        remote_entrypoint = f"/private/entrypoints/{job_id}.sh"
        self.entries[job_id] = {
            "job_id": job_id,
            "backend_type": "slurm",
            "image": image,
            "results_dir": f"lustre:///results/{job_id}",
            "status": status,
        }
        self.identities[job_id] = {
            "slurm_job_id": slurm_job_id,
            "failed_slurm_job_ids": [],
            "retry_count": 0,
            "submission_attempted": submission_attempted,
            "launch_uncertain": launch_uncertain,
            "launch_token": f"token-{job_id}",
            "pre_launch_slurm_job_id": "",
            "remote_entrypoint": remote_entrypoint,
        }
        self.remote_files[remote_entrypoint] = (
            self.entrypoint(command)
            if entrypoint_content is None
            else entrypoint_content
        )
        self.commands[job_id] = command


def _install_fake_sdk(
    monkeypatch: pytest.MonkeyPatch,
    state: _FakeState,
) -> None:
    class FakeSlurmSDK:
        def __init__(self, **_: Any) -> None:
            self._state = state
            self._handler = self
            self._monitor = SimpleNamespace(stop=lambda: None)
            self._store = SimpleNamespace(close=lambda: None)

        def list_jobs(self) -> list[dict[str, Any]]:
            return [
                {
                    "backend_type": entry["backend_type"],
                    "job_id": job_id,
                }
                for job_id, entry in self._state.entries.items()
            ]

        def create_job(self, **kwargs: Any) -> SimpleNamespace:
            self._state.create_attempt += 1
            attempt = self._state.create_attempt
            command = str(kwargs["command"])
            self._state.events.append(("create_attempt", command))
            if attempt in self._state.fail_before_attempts:
                self._state.fail_before_attempts.remove(attempt)
                raise RuntimeError(f"synthetic failure before create {attempt}")
            job_id = f"created-job-{attempt:02d}"
            self._state.add_job(
                command=command,
                job_id=job_id,
                slurm_job_id=str(9000 + attempt),
                image=str(kwargs["image"]),
            )
            self._state.events.append(("created", job_id))
            if attempt in self._state.fail_after_attempts:
                self._state.fail_after_attempts.remove(attempt)
                raise RuntimeError(f"synthetic failure after create {attempt}")
            return SimpleNamespace(id=job_id)

        def _load_job_from_store(self, job_id: str) -> SimpleNamespace | None:
            if job_id not in self._state.entries:
                return None
            return SimpleNamespace(id=job_id)

        def get_job_runtime_identity(self, job_id: str) -> dict[str, Any]:
            return copy.deepcopy(self._state.identities[job_id])

        def _read_remote_log_file(self, path: str) -> str | None:
            return self._state.remote_files.get(path)

        def _build_entrypoint_script(self, command: str) -> str:
            return self._state.entrypoint(command)

        def get_job_results_dir(self, job_id: str) -> str:
            return str(self._state.entries[job_id]["results_dir"])

        def get_job(self, job_id: str) -> dict[str, Any] | None:
            entry = self._state.entries.get(job_id)
            return copy.deepcopy(entry) if entry is not None else None

        def _backend_identity_matches(
            self,
            job_id: str,
            *,
            entry: dict[str, Any],
        ) -> bool:
            return entry.get("job_id") == job_id

        def get_tao_job_status(self, job_id: str, **_: Any) -> str:
            return str(self._state.entries[job_id]["status"])

        @staticmethod
        def normalize_status(status: str) -> str:
            return status

        def get_job_message(self, job_id: str) -> str:
            return f"synthetic status for {job_id}"

        def cancel_job(self, job_id: str) -> bool:
            self._state.events.append(("cancel", job_id))
            self._state.entries[job_id]["status"] = "Canceled"
            self._state.identities[job_id]["launch_uncertain"] = False
            self._state.identities[job_id]["slurm_job_id"] = ""
            return True

    root = types.ModuleType("tao_sdk")
    platforms = types.ModuleType("tao_sdk.platforms")
    slurm = types.ModuleType("tao_sdk.platforms.slurm")
    slurm.SlurmSDK = FakeSlurmSDK
    root.platforms = platforms
    platforms.slurm = slurm
    monkeypatch.setitem(sys.modules, "tao_sdk", root)
    monkeypatch.setitem(sys.modules, "tao_sdk.platforms", platforms)
    monkeypatch.setitem(sys.modules, "tao_sdk.platforms.slurm", slurm)
    monkeypatch.setattr(
        launcher,
        "enforce_remote_sdk_containment",
        lambda: copy.deepcopy(CONTAINMENT),
    )


def _crash_initial_launch(
    *,
    runtime_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: _FakeState,
    failure_attempt: int,
    after_create: bool,
) -> tuple[
    dict[str, Any],
    list[tuple[str, dict[str, Any]]],
]:
    manifest = _manifest(runtime_dir)
    commands = _commands(manifest)
    _write_dry_run(runtime_dir, manifest)
    _install_fake_sdk(monkeypatch, state)
    target = (
        state.fail_after_attempts
        if after_create
        else state.fail_before_attempts
    )
    target.add(failure_attempt)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        launcher.submit_all(
            manifest=manifest,
            manifest_file_sha256=MANIFEST_FILE_SHA256,
            commands=commands,
            runtime_dir=runtime_dir,
            source_checks=BASE_SOURCE_CHECKS,
        )
    return manifest, commands


def _resume(
    *,
    runtime_dir: Path,
    manifest: dict[str, Any],
    commands: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    ledger = runtime_dir / "block_submissions.json"
    return launcher.resume_incomplete_submission(
        manifest=manifest,
        manifest_file_sha256=MANIFEST_FILE_SHA256,
        commands=commands,
        runtime_dir=runtime_dir,
        source_checks=BASE_SOURCE_CHECKS,
        supplied_ledger_sha256=generator.sha256_file(ledger),
    )


def _incomplete_ledger_fixture(
    runtime_dir: Path,
) -> tuple[
    dict[str, Any],
    list[tuple[str, dict[str, Any]]],
    Path,
    dict[str, Any],
]:
    manifest = _manifest(runtime_dir)
    commands = _commands(manifest)
    source_checks = _write_launch_binding(runtime_dir, manifest)
    first_summary = commands[0][1]
    first = _record(
        first_summary,
        tao_job_id="existing-job-00",
        slurm_job_id="1000",
    )
    pending = {
        "allocation_id": commands[1][1]["allocation_id"],
        "command_sha256": commands[1][1]["command_sha256"],
        "state": "intent_recorded_before_create_job",
        "sdk_job_ids_before": ["existing-job-00"],
        "remote_sdk_containment_sha256": SHA_C,
    }
    ledger = launcher.submission_ledger_payload(
        manifest,
        MANIFEST_FILE_SHA256,
        [first],
        status="submitting_incomplete",
        source_checks=source_checks,
        pending_submission=pending,
    )
    path = runtime_dir / "block_submissions.json"
    launcher.atomic_json(path, ledger)
    return manifest, commands, path, source_checks


def test_exclusive_nonblocking_lock_rejects_concurrent_operation(
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "runtime"
    lock_path = (
        runtime_dir.parent / f".{runtime_dir.name}.submission.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch(mode=0o600)
    os.chmod(lock_path, 0o600)
    called = False

    @launcher.exclusive_submission_operation
    def mutation(*, runtime_dir: Path) -> None:
        nonlocal called
        called = True

    with lock_path.open("a+b") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            launcher.ContractError,
            match="another post-front submission operation",
        ):
            mutation(runtime_dir=runtime_dir)
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
    assert called is False


def test_submission_lock_refuses_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "runtime"
    lock_path = (
        runtime_dir.parent / f".{runtime_dir.name}.submission.lock"
    )
    target = tmp_path / "unrelated"
    target.write_bytes(b"preserve-me")
    os.chmod(target, 0o644)
    lock_path.symlink_to(target)

    @launcher.exclusive_submission_operation
    def mutation(*, runtime_dir: Path) -> None:
        raise AssertionError("unsafe lock must prevent mutation")

    with pytest.raises(
        launcher.ContractError,
        match="cannot be opened safely",
    ):
        mutation(runtime_dir=runtime_dir)
    assert target.read_bytes() == b"preserve-me"
    assert (target.stat().st_mode & 0o777) == 0o644


def test_env_file_accepts_exact_0600_and_rejects_ambient_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "config.env"
    path.write_text("SYNTHETIC_TOKEN='file-value'\n", encoding="utf-8")
    os.chmod(path, 0o600)
    monkeypatch.delenv("SYNTHETIC_TOKEN", raising=False)
    assert launcher.load_env_file(path) == ["SYNTHETIC_TOKEN"]
    assert os.environ["SYNTHETIC_TOKEN"] == "file-value"

    conflict = tmp_path / "conflict.env"
    conflict.write_text("SYNTHETIC_TOKEN=other-value\n", encoding="utf-8")
    os.chmod(conflict, 0o600)
    with pytest.raises(ValueError, match="ambient environment conflicts"):
        launcher.load_env_file(conflict)


@pytest.mark.parametrize("unsafe_kind", ["symlink", "group_bits", "owner"])
def test_env_file_rejects_unsafe_identity_or_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_kind: str,
) -> None:
    target = tmp_path / "target.env"
    target.write_text("SYNTHETIC_ONLY=value\n", encoding="utf-8")
    os.chmod(target, 0o600)
    path = target
    if unsafe_kind == "symlink":
        path = tmp_path / "linked.env"
        path.symlink_to(target)
    elif unsafe_kind == "group_bits":
        os.chmod(path, 0o640)
    else:
        monkeypatch.setattr(launcher.os, "geteuid", lambda: os.getuid() + 1)
    with pytest.raises(PermissionError, match="owner-owned, non-symlink"):
        launcher.load_env_file(path)


def test_incomplete_ledger_exact_snapshot_loads_and_tampering_fails(
    tmp_path: Path,
) -> None:
    manifest, commands, path, source_checks = _incomplete_ledger_fixture(
        tmp_path / "runtime"
    )
    loaded, whole = launcher.load_incomplete_ledger_for_resume(
        path=path,
        supplied_sha256=generator.sha256_file(path),
        manifest=manifest,
        manifest_file_sha256=MANIFEST_FILE_SHA256,
        commands=commands,
        expected_source_checks=BASE_SOURCE_CHECKS,
    )
    assert loaded["pending_submission"]["sdk_job_ids_before"] == [
        "existing-job-00"
    ]
    assert whole == generator.sha256_file(path)

    semantic_tamper = copy.deepcopy(loaded)
    semantic_tamper["pending_submission"]["sdk_job_ids_before"] = [
        "z-job",
        "a-job",
    ]
    semantic_tamper.pop("ledger_sha256")
    semantic_tamper["ledger_sha256"] = generator.sha256_value(
        semantic_tamper
    )
    launcher.atomic_json(path, semantic_tamper)
    with pytest.raises(
        launcher.ContractError,
        match="resume SDK job snapshot is invalid",
    ):
        launcher.load_incomplete_ledger_for_resume(
            path=path,
            supplied_sha256=generator.sha256_file(path),
            manifest=manifest,
            manifest_file_sha256=MANIFEST_FILE_SHA256,
            commands=commands,
            expected_source_checks=BASE_SOURCE_CHECKS,
        )

    digest_tamper = copy.deepcopy(loaded)
    digest_tamper["manifest_id"] = "tampered-without-rehash"
    launcher.atomic_json(path, digest_tamper)
    with pytest.raises(
        generator.ContractError,
        match="canonical digest mismatch",
    ):
        launcher.load_incomplete_ledger_for_resume(
            path=path,
            supplied_sha256=generator.sha256_file(path),
            manifest=manifest,
            manifest_file_sha256=MANIFEST_FILE_SHA256,
            commands=commands,
            expected_source_checks=source_checks,
        )


@pytest.mark.parametrize("completed_before_crash", [0, 2, 5])
def test_failure_after_k_submissions_resumes_without_duplicate_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    completed_before_crash: int,
) -> None:
    runtime_dir = tmp_path / f"runtime-{completed_before_crash}"
    state = _FakeState()
    manifest, commands = _crash_initial_launch(
        runtime_dir=runtime_dir,
        monkeypatch=monkeypatch,
        state=state,
        failure_attempt=completed_before_crash + 1,
        after_create=False,
    )
    incomplete = generator.load_json(
        runtime_dir / "block_submissions.json"
    )
    assert incomplete["status"] == "submitting_incomplete"
    assert incomplete["allocation_count"] == completed_before_crash
    assert incomplete["pending_submission"]["allocation_id"] == commands[
        completed_before_crash
    ][1]["allocation_id"]
    completed = _resume(
        runtime_dir=runtime_dir,
        manifest=manifest,
        commands=commands,
    )
    assert len(completed) == 6
    assert len(state.entries) == 6
    assert len({item["tao_job_id"] for item in completed}) == 6
    assert [
        state.commands[item["tao_job_id"]] for item in completed
    ] == [command for command, _ in commands]
    final = generator.load_json(runtime_dir / "block_submissions.json")
    assert final["status"] == "complete"
    assert final["pending_submission"] is None


def test_resume_recovers_one_definitely_created_job_without_resubmitting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_dir = tmp_path / "runtime"
    state = _FakeState()
    manifest, commands = _crash_initial_launch(
        runtime_dir=runtime_dir,
        monkeypatch=monkeypatch,
        state=state,
        failure_attempt=3,
        after_create=True,
    )
    recovered_id = "created-job-03"
    assert state.commands[recovered_id] == commands[2][0]
    completed = _resume(
        runtime_dir=runtime_dir,
        manifest=manifest,
        commands=commands,
    )
    assert completed[2]["tao_job_id"] == recovered_id
    assert len(state.entries) == 6
    assert list(state.commands.values()).count(commands[2][0]) == 1


def test_resume_blocks_ambiguous_multiple_job_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_dir = tmp_path / "runtime"
    state = _FakeState()
    manifest, commands = _crash_initial_launch(
        runtime_dir=runtime_dir,
        monkeypatch=monkeypatch,
        state=state,
        failure_attempt=3,
        after_create=False,
    )
    state.add_job(
        command=commands[2][0],
        job_id="unbound-one",
        slurm_job_id="8001",
    )
    state.add_job(
        command=commands[2][0],
        job_id="unbound-two",
        slurm_job_id="8002",
    )
    attempts_before = state.create_attempt
    with pytest.raises(
        launcher.ContractError,
        match="ambiguous pending submission created multiple SDK jobs",
    ):
        _resume(
            runtime_dir=runtime_dir,
            manifest=manifest,
            commands=commands,
        )
    assert state.create_attempt == attempts_before


def test_resume_blocks_uncertain_created_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_dir = tmp_path / "runtime"
    state = _FakeState()
    manifest, commands = _crash_initial_launch(
        runtime_dir=runtime_dir,
        monkeypatch=monkeypatch,
        state=state,
        failure_attempt=3,
        after_create=True,
    )
    state.identities["created-job-03"]["launch_uncertain"] = True
    attempts_before = state.create_attempt
    with pytest.raises(
        launcher.ContractError,
        match="pending SDK launch remains uncertain",
    ):
        _resume(
            runtime_dir=runtime_dir,
            manifest=manifest,
            commands=commands,
        )
    assert state.create_attempt == attempts_before


def test_resume_blocks_remote_entrypoint_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_dir = tmp_path / "runtime"
    state = _FakeState()
    manifest, commands = _crash_initial_launch(
        runtime_dir=runtime_dir,
        monkeypatch=monkeypatch,
        state=state,
        failure_attempt=3,
        after_create=True,
    )
    path = state.identities["created-job-03"]["remote_entrypoint"]
    state.remote_files[path] = "#!/bin/sh\nwrong-command\n"
    with pytest.raises(
        launcher.ContractError,
        match="remote command does not match intent",
    ):
        _resume(
            runtime_dir=runtime_dir,
            manifest=manifest,
            commands=commands,
        )


def test_pre_scheduler_job_is_terminalized_before_retry_and_history_is_known(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_dir = tmp_path / "runtime"
    state = _FakeState()
    manifest, commands = _crash_initial_launch(
        runtime_dir=runtime_dir,
        monkeypatch=monkeypatch,
        state=state,
        failure_attempt=1,
        after_create=True,
    )
    abandoned_id = "created-job-01"
    state.identities[abandoned_id]["slurm_job_id"] = ""
    state.identities[abandoned_id]["submission_attempted"] = False

    # The first resume must cancel/record the staged row, submit its retry,
    # and then crash while recording the next row's intent.
    state.fail_before_attempts.add(3)
    with pytest.raises(RuntimeError, match="failure before create 3"):
        _resume(
            runtime_dir=runtime_dir,
            manifest=manifest,
            commands=commands,
        )
    intermediate = generator.load_json(
        runtime_dir / "block_submissions.json"
    )
    assert intermediate["allocation_count"] == 1
    assert len(intermediate["submission_recovery_events"]) == 1
    recovery = intermediate["submission_recovery_events"][0]
    assert {
        key: recovery[key]
        for key in (
            "event_index",
            "allocation_id",
            "command_sha256",
            "reason",
            "tao_job_id",
            "slurm_job_id",
            "sdk_status",
            "submission_attempted",
            "launch_uncertain",
            "partial_measurements_reused",
            "feeds_final_selection",
            "feeds_reselection",
        )
    } == {
        "event_index": 0,
        "allocation_id": commands[0][1]["allocation_id"],
        "command_sha256": commands[0][1]["command_sha256"],
        "reason": "proven_pre_scheduler_submission_abandoned",
        "tao_job_id": abandoned_id,
        "slurm_job_id": "",
        "sdk_status": "Canceled",
        "submission_attempted": False,
        "launch_uncertain": False,
        "partial_measurements_reused": False,
        "feeds_final_selection": False,
        "feeds_reselection": False,
    }
    assert recovery["reconciliation"]["decision"] == (
        "pre_scheduler_job_terminalized_then_resubmit"
    )
    cancel_index = state.events.index(("cancel", abandoned_id))
    retry_index = state.events.index(("create_attempt", commands[0][0]), 1)
    assert cancel_index < retry_index

    # The second resume must include the terminal recovery-event ID in its
    # exact durable-state set.  Omitting history would classify it as unbound.
    completed = _resume(
        runtime_dir=runtime_dir,
        manifest=manifest,
        commands=commands,
    )
    assert len(completed) == 6
    assert len(state.entries) == 7
    assert state.entries[abandoned_id]["status"] == "Canceled"
    final = generator.load_json(runtime_dir / "block_submissions.json")
    assert final["status"] == "complete"
    assert final["submission_recovery_events"][0]["tao_job_id"] == abandoned_id


def _complete_replacement_fixture(
    *,
    runtime_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    dict[str, Any],
    list[tuple[str, dict[str, Any]]],
    _FakeState,
    Path,
    list[dict[str, Any]],
]:
    manifest = _manifest(runtime_dir)
    commands = _commands(manifest)
    source_checks = _write_launch_binding(runtime_dir, manifest)
    state = _FakeState()
    submissions: list[dict[str, Any]] = []
    for index, (command, summary) in enumerate(commands):
        job_id = f"prior-job-{index:02d}"
        slurm_id = str(7000 + index)
        state.add_job(
            command=command,
            job_id=job_id,
            slurm_job_id=slurm_id,
            status="Error" if index == 0 else "Running",
        )
        submissions.append(
            _record(
                summary,
                tao_job_id=job_id,
                slurm_job_id=slurm_id,
            )
        )
    _install_fake_sdk(monkeypatch, state)
    ledger_path = runtime_dir / "block_submissions.json"
    launcher.atomic_json(
        ledger_path,
        launcher.submission_ledger_payload(
            manifest,
            MANIFEST_FILE_SHA256,
            submissions,
            status="complete",
            source_checks=source_checks,
        ),
    )
    return manifest, commands, state, ledger_path, submissions


def _replacement_intent(
    *,
    runtime_dir: Path,
    manifest: dict[str, Any],
    commands: list[tuple[str, dict[str, Any]]],
    submissions: list[dict[str, Any]],
    ledger_whole_sha256: str,
) -> dict[str, Any]:
    parent_ledger = generator.load_json(
        runtime_dir / "block_submissions.json"
    )
    intent = {
        "schema_version": 1,
        "intent_id": "dino_post_front_complete_allocation_replacement_v1",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": MANIFEST_FILE_SHA256,
        "allocation_id": commands[0][1]["allocation_id"],
        "command_sha256": commands[0][1]["command_sha256"],
        "ledger_revision": 2,
        "parent_ledger_whole_file_sha256": ledger_whole_sha256,
        "parent_ledger_internal_sha256": parent_ledger["ledger_sha256"],
        "prior_tao_job_id": submissions[0]["tao_job_id"],
        "prior_slurm_job_id": submissions[0]["slurm_job_id"],
        "prior_sdk_status": "Error",
        "replacement_basis": "sdk_terminal_failure",
        "invalidation_evidence": None,
        "partial_measurements_reused": False,
        "feeds_final_selection": False,
        "feeds_reselection": False,
    }
    intent["intent_sha256"] = generator.sha256_value(intent)
    return intent


def test_exact_orphan_replacement_intent_is_replayed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_dir = tmp_path / "runtime"
    (
        manifest,
        commands,
        state,
        ledger_path,
        submissions,
    ) = _complete_replacement_fixture(
        runtime_dir=runtime_dir,
        monkeypatch=monkeypatch,
    )
    parent_sha = generator.sha256_file(ledger_path)
    intent = _replacement_intent(
        runtime_dir=runtime_dir,
        manifest=manifest,
        commands=commands,
        submissions=submissions,
        ledger_whole_sha256=parent_sha,
    )
    intent_path = runtime_dir / "replacement_intent.r002.json"
    launcher.atomic_create_json(intent_path, intent)
    before_intent_bytes = intent_path.read_bytes()
    replacement = launcher.replacement_submission(
        manifest=manifest,
        manifest_file_sha256=MANIFEST_FILE_SHA256,
        commands=commands,
        runtime_dir=runtime_dir,
        source_checks=BASE_SOURCE_CHECKS,
        allocation_id=commands[0][1]["allocation_id"],
        supplied_ledger_sha256=parent_sha,
    )
    assert intent_path.read_bytes() == before_intent_bytes
    assert replacement["tao_job_id"] == "created-job-01"
    assert state.create_attempt == 1
    final = generator.load_json(ledger_path)
    assert final["status"] == "complete"
    assert final["ledger_revision"] == 2
    assert final["superseded_submissions"][0][
        "replacement_intent"
    ]["internal_sha256"] == intent["intent_sha256"]


def test_tampered_orphan_replacement_intent_blocks_before_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_dir = tmp_path / "runtime"
    (
        manifest,
        commands,
        state,
        ledger_path,
        submissions,
    ) = _complete_replacement_fixture(
        runtime_dir=runtime_dir,
        monkeypatch=monkeypatch,
    )
    parent_sha = generator.sha256_file(ledger_path)
    intent = _replacement_intent(
        runtime_dir=runtime_dir,
        manifest=manifest,
        commands=commands,
        submissions=submissions,
        ledger_whole_sha256=parent_sha,
    )
    intent["feeds_reselection"] = True
    intent.pop("intent_sha256")
    intent["intent_sha256"] = generator.sha256_value(intent)
    launcher.atomic_create_json(
        runtime_dir / "replacement_intent.r002.json",
        intent,
    )
    with pytest.raises(
        launcher.ContractError,
        match="orphan replacement intent replay",
    ):
        launcher.replacement_submission(
            manifest=manifest,
            manifest_file_sha256=MANIFEST_FILE_SHA256,
            commands=commands,
            runtime_dir=runtime_dir,
            source_checks=BASE_SOURCE_CHECKS,
            allocation_id=commands[0][1]["allocation_id"],
            supplied_ledger_sha256=parent_sha,
        )
    assert state.create_attempt == 0
