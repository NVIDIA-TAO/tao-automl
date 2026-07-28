"""Hermetic tests for Complete-but-invalid matched-allocation recovery."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
from typing import Any

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import post_front_matched_aggregator as aggregator  # noqa: E402
import post_front_matched_launcher as launcher  # noqa: E402
import post_front_matched_manifest_generator as generator  # noqa: E402
import test_post_front_matched_launcher_recovery as recovery  # noqa: E402


def _complete_jobs(
    submissions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "allocation_id": item["allocation_id"],
            "tao_job_id": item["tao_job_id"],
            "slurm_job_id": str(item["slurm_job_id"]),
            "sdk_status": "Complete",
            "slurm_state": "COMPLETED",
            "slurm_exit_code": "0:0",
            "complete": True,
        }
        for item in submissions
    ]


def _exact_invalidation(
    *,
    runtime_dir: Path,
    manifest: dict[str, Any],
    ledger_path: Path,
    submissions: list[dict[str, Any]],
    allocation_id: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    ledger = generator.load_json(ledger_path)
    payload = aggregator.build_invalidation_evidence(
        manifest=manifest,
        manifest_file_sha256=recovery.MANIFEST_FILE_SHA256,
        ledger=ledger,
        ledger_file_sha256=generator.sha256_file(ledger_path),
        ledger_path=ledger_path,
        jobs=_complete_jobs(submissions),
        allocation_ids=[allocation_id],
        failure=aggregator.deterministic_failure_record(
            stage="semantic_aggregation",
            error=aggregator.ContractError(
                f"{allocation_id}: synthetic rank record is invalid"
            ),
        ),
        available_artifacts=[
            {
                "kind": "allocation_result",
                "allocation_id": allocation_id,
                "candidate_id": "",
                "rank": "",
                "path": "/synthetic/allocation_result.json",
                "sha256": recovery.SHA_A,
            },
            {
                "kind": "rank_record",
                "allocation_id": allocation_id,
                "candidate_id": "candidate_a",
                "rank": "0",
                "path": "/synthetic/rank_0.json",
                "sha256": recovery.SHA_B,
            },
        ],
        artifact_probe={
            "status": "not_required_bundle_hashes_already_fetched",
            "allocation_ids": [allocation_id],
            "available_artifact_count": 2,
        },
    )
    reference = aggregator.write_invalidation_evidence(
        manifest=manifest,
        payload=payload,
    )
    return payload, reference


def _complete_fixture(
    *,
    runtime_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    dict[str, Any],
    list[tuple[str, dict[str, Any]]],
    recovery._FakeState,
    Path,
    list[dict[str, Any]],
]:
    fixture = recovery._complete_replacement_fixture(
        runtime_dir=runtime_dir,
        monkeypatch=monkeypatch,
    )
    manifest, commands, state, _, submissions = fixture
    state.entries[submissions[0]["tao_job_id"]]["status"] = "Complete"
    return fixture


def _write_valid_analysis(
    *,
    runtime_dir: Path,
    manifest: dict[str, Any],
    ledger_path: Path,
) -> None:
    ledger = generator.load_json(ledger_path)
    report = {
        "schema_version": 1,
        "status": "complete",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": recovery.MANIFEST_FILE_SHA256,
        "submission_ledger_sha256": generator.sha256_file(ledger_path),
        "submission_ledger_revision": ledger["ledger_revision"],
    }
    report["report_sha256"] = generator.sha256_value(report)
    launcher.atomic_create_json(
        runtime_dir / "post_front_matched_analysis.json",
        report,
    )


def test_complete_valid_job_cannot_be_replaced_without_invalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_dir = tmp_path / "runtime"
    manifest, commands, state, ledger_path, _ = _complete_fixture(
        runtime_dir=runtime_dir,
        monkeypatch=monkeypatch,
    )
    _write_valid_analysis(
        runtime_dir=runtime_dir,
        manifest=manifest,
        ledger_path=ledger_path,
    )
    with pytest.raises(
        launcher.ContractError,
        match="valid immutable analysis",
    ):
        launcher.replacement_submission(
            manifest=manifest,
            manifest_file_sha256=recovery.MANIFEST_FILE_SHA256,
            commands=commands,
            runtime_dir=runtime_dir,
            source_checks=recovery.BASE_SOURCE_CHECKS,
            allocation_id=commands[0][1]["allocation_id"],
            supplied_ledger_sha256=generator.sha256_file(ledger_path),
        )
    assert state.create_attempt == 0


def test_complete_job_without_analysis_still_requires_exact_invalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_dir = tmp_path / "runtime"
    manifest, commands, state, ledger_path, _ = _complete_fixture(
        runtime_dir=runtime_dir,
        monkeypatch=monkeypatch,
    )
    with pytest.raises(
        launcher.ContractError,
        match="requires invalidation evidence",
    ):
        launcher.replacement_submission(
            manifest=manifest,
            manifest_file_sha256=recovery.MANIFEST_FILE_SHA256,
            commands=commands,
            runtime_dir=runtime_dir,
            source_checks=recovery.BASE_SOURCE_CHECKS,
            allocation_id=commands[0][1]["allocation_id"],
            supplied_ledger_sha256=generator.sha256_file(ledger_path),
        )
    assert state.create_attempt == 0


def test_complete_invalid_exact_evidence_reruns_entire_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_dir = tmp_path / "runtime"
    manifest, commands, state, ledger_path, submissions = _complete_fixture(
        runtime_dir=runtime_dir,
        monkeypatch=monkeypatch,
    )
    allocation_id = submissions[0]["allocation_id"]
    payload, reference = _exact_invalidation(
        runtime_dir=runtime_dir,
        manifest=manifest,
        ledger_path=ledger_path,
        submissions=submissions,
        allocation_id=allocation_id,
    )
    replacement = launcher.replacement_submission(
        manifest=manifest,
        manifest_file_sha256=recovery.MANIFEST_FILE_SHA256,
        commands=commands,
        runtime_dir=runtime_dir,
        source_checks=recovery.BASE_SOURCE_CHECKS,
        allocation_id=allocation_id,
        supplied_ledger_sha256=generator.sha256_file(ledger_path),
        invalidation_evidence=reference,
    )
    assert replacement["tao_job_id"] != submissions[0]["tao_job_id"]
    assert replacement["candidate_order"] == submissions[0]["candidate_order"]
    assert replacement["candidate_count"] == submissions[0]["candidate_count"]
    assert state.create_attempt == 1
    revised = generator.load_json(ledger_path)
    supersession = revised["superseded_submissions"][0]
    assert supersession["prior_sdk_status"] == "Complete"
    assert supersession["reason"] == (
        "complete_but_semantically_invalid_allocation"
    )
    assert supersession["invalidation_evidence"] == reference
    assert supersession["partial_measurements_reused"] is False
    assert payload["full_allocation_discarded"] is True
    assert payload["available_artifact_count"] == 2
    assert payload["partial_measurements_used_for_analysis"] is False
    assert revised["feeds_final_selection"] is False
    assert revised["feeds_reselection"] is False


@pytest.mark.parametrize(
    "tamper",
    ["file", "stale_ledger", "wrong_allocation"],
)
def test_complete_invalidation_tamper_staleness_and_binding_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    runtime_dir = tmp_path / tamper / "runtime"
    manifest, commands, state, ledger_path, submissions = _complete_fixture(
        runtime_dir=runtime_dir,
        monkeypatch=monkeypatch,
    )
    allocation_id = submissions[0]["allocation_id"]
    payload, reference = _exact_invalidation(
        runtime_dir=runtime_dir,
        manifest=manifest,
        ledger_path=ledger_path,
        submissions=submissions,
        allocation_id=allocation_id,
    )
    target_allocation = allocation_id
    if tamper == "file":
        evidence_path = Path(reference["path"])
        changed = copy.deepcopy(payload)
        changed["failure"]["message"] = "tampered"
        launcher.atomic_json(evidence_path, changed)
    elif tamper == "stale_ledger":
        changed = copy.deepcopy(payload)
        changed["complete_ledger"]["whole_file_sha256"] = "e" * 64
        changed.pop("invalidation_sha256")
        changed["invalidation_sha256"] = generator.sha256_value(changed)
        evidence_path = Path(reference["path"])
        launcher.atomic_json(evidence_path, changed)
        reference = {
            "path": str(evidence_path),
            "whole_file_sha256": generator.sha256_file(evidence_path),
            "internal_sha256": changed["invalidation_sha256"],
        }
    else:
        target_allocation = submissions[1]["allocation_id"]
        state.entries[submissions[1]["tao_job_id"]]["status"] = "Complete"
    with pytest.raises(
        (launcher.ContractError, generator.ContractError),
    ):
        launcher.replacement_submission(
            manifest=manifest,
            manifest_file_sha256=recovery.MANIFEST_FILE_SHA256,
            commands=commands,
            runtime_dir=runtime_dir,
            source_checks=recovery.BASE_SOURCE_CHECKS,
            allocation_id=target_allocation,
            supplied_ledger_sha256=generator.sha256_file(ledger_path),
            invalidation_evidence=reference,
        )
    assert state.create_attempt == 0


@pytest.mark.parametrize(
    "allocation_count",
    [0, 2],
)
def test_unknown_or_multiple_semantic_failure_blocks_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allocation_count: int,
) -> None:
    runtime_dir = tmp_path / f"blocked-{allocation_count}" / "runtime"
    manifest, commands, state, ledger_path, submissions = _complete_fixture(
        runtime_dir=runtime_dir,
        monkeypatch=monkeypatch,
    )
    for submission in submissions:
        state.entries[submission["tao_job_id"]]["status"] = "Complete"
    allocation_ids = [
        item["allocation_id"]
        for item in submissions[:allocation_count]
    ]
    ledger = generator.load_json(ledger_path)
    payload = aggregator.build_invalidation_evidence(
        manifest=manifest,
        manifest_file_sha256=recovery.MANIFEST_FILE_SHA256,
        ledger=ledger,
        ledger_file_sha256=generator.sha256_file(ledger_path),
        ledger_path=ledger_path,
        jobs=_complete_jobs(submissions),
        allocation_ids=allocation_ids,
        failure=aggregator.deterministic_failure_record(
            stage="semantic_aggregation",
            error=aggregator.ContractError("synthetic global mismatch"),
        ),
        available_artifacts=[],
        artifact_probe={
            "status": "not_required_bundle_hashes_already_fetched",
            "allocation_ids": allocation_ids,
            "available_artifact_count": 0,
        },
    )
    reference = aggregator.write_invalidation_evidence(
        manifest=manifest,
        payload=payload,
    )
    assert payload["replacement_permitted"] is False
    assert payload["attribution"]["replacement_blocked"] is True
    assert payload["attribution"]["block_reason"] in {
        "failure_could_not_be_attributed_to_any_allocation",
        "failure_implicates_multiple_allocations",
    }
    with pytest.raises(
        launcher.ContractError,
        match="invalidation-evidence path|status mismatch",
    ):
        launcher.replacement_submission(
            manifest=manifest,
            manifest_file_sha256=recovery.MANIFEST_FILE_SHA256,
            commands=commands,
            runtime_dir=runtime_dir,
            source_checks=recovery.BASE_SOURCE_CHECKS,
            allocation_id=submissions[0]["allocation_id"],
            supplied_ledger_sha256=generator.sha256_file(ledger_path),
            invalidation_evidence=reference,
        )
    assert state.create_attempt == 0


def test_semantic_failure_attribution_is_exact_or_explicitly_ambiguous() -> None:
    manifest = recovery._manifest(Path("/tmp/synthetic-runtime"))
    first = manifest["schedule"]["allocations"][0]["allocation_id"]
    second = manifest["schedule"]["allocations"][1]["allocation_id"]
    assert aggregator.semantic_failure_allocation_ids(
        aggregator.ContractError(f"{first}: invalid rank record"),
        manifest,
    ) == [first]
    assert aggregator.semantic_failure_allocation_ids(
        aggregator.ContractError(f"{first} disagrees with {second}"),
        manifest,
    ) == sorted([first, second])
    assert (
        aggregator.semantic_failure_allocation_ids(
            aggregator.ContractError("global runtime mismatch"),
            manifest,
        )
        == []
    )
