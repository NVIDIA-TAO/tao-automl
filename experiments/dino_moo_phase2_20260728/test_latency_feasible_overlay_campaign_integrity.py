"""Read-only and mutation contracts for the overlay campaign verifier."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import latency_feasible_overlay_campaign_integrity as audit_module  # noqa: E402


HASHES = {
    "manifest_whole_file_sha256": (
        "f83be40f9e00bcd5fc62959bf5a732327353a91ca12ee2efc118325ded2d0db4"
    ),
    "manifest_internal_sha256": (
        "32c268f11742c26bdc47c61272f73a6d9f651317606e9102e75585bc7d2100e9"
    ),
    "overlay_whole_file_sha256": (
        "d054df9923a717759786628177814d5d531d7af3bf136da7361a4882b83410aa"
    ),
    "overlay_internal_sha256": (
        "e2e6446e3dca7f52e7b2bac682c26bc5b711067b0a18dae71cfbb24f5a44402c"
    ),
    "recovery_evidence_whole_file_sha256": (
        "923284e51f9e41f6954a1615f683ea6221b62fce0f490c3d5252b7c0d67f3e56"
    ),
    "recovery_evidence_internal_sha256": (
        "01715fecc26c3f2b5085650f3bd17f62a502dfc6a85560f11cfa968d08ac9456"
    ),
    "launch_contract_whole_file_sha256": (
        "a9fdc853d5847b9a945a012c5c281877a0e8d70c5bec09b5caea31382639b84d"
    ),
    "launch_contract_internal_sha256": (
        "9a29cad7a9863b7a5b9c961d62a0761a2462422468499948ff840667a608d94f"
    ),
    "ledger_whole_file_sha256": (
        "a5768627797d91f8c9ab88d5893bcf607705b83ce30a1ceef9d5bc20ce39fd81"
    ),
    "ledger_internal_sha256": (
        "0c0a69ca6203a88fac9f04aaf62bc7a7ed2644b18307c1664243af2d389ad168"
    ),
    "analysis_whole_file_sha256": (
        "30505ee6fd1635ac4edba01f95136f527a2df9e54257675a1fb505bd0b422efd"
    ),
    "analysis_internal_sha256": (
        "8ff21a404357e59949dfde7c9889710e854b0e5721ef53f39f439e71478c7351"
    ),
}


def live_arguments() -> dict:
    return {
        "manifest_path": audit_module.DEFAULT_MANIFEST,
        "overlay_path": audit_module.DEFAULT_OVERLAY,
        "recovery_evidence_path": audit_module.DEFAULT_RECOVERY_EVIDENCE,
        "launch_contract_path": audit_module.DEFAULT_LAUNCH_CONTRACT,
        "ledger_path": audit_module.DEFAULT_LEDGER,
        **{
            key: value
            for key, value in HASHES.items()
            if not key.startswith("analysis_")
        },
    }


def rehash(value: dict, field: str) -> None:
    value.pop(field, None)
    value[field] = audit_module.manifest_generator.sha256_value(value)


def write_json(path: Path, value: dict) -> str:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit_module.manifest_generator.sha256_file(path)


def test_live_campaign_golden_reconstructs_every_submitted_byte() -> None:
    audit = audit_module.build_audit(**live_arguments())

    assert audit["status"] == "pass"
    assert audit["cohort_and_schedule"]["exact_frozen_6x4_design"] is True
    assert audit["execution_reconstruction"][
        "all_command_and_bundle_hashes_match"
    ] is True
    assert [
        item["block_plan_sha256"]
        for item in audit["execution_reconstruction"]["allocations"]
    ] == [
        "cbcbefcc0c3389d419655325d43fc66f8757ea5b8bef9d15554f8c3a46752595",
        "882d2887be60f031f1d578e954035611afa76825685fb658cf88077867185e13",
        "7bc5b80e7bc82d7eacde430bf2fade5224276bf8b069b143613ccfe6627dce36",
        "e59d0cd7e1ff4f90e7f950d19da2d4554c50a44e7f7698ef35a8c6cce975b8e6",
        "ac242d01754e4ce27915d429023e20ee30836337c02aaf060852f213c161e7f5",
        "e83d820c372146a3148a07444ec71972e21c30a83af6d6ee6284681edaee798e",
    ]
    assert set(audit["selection_isolation"]) == set(
        audit_module.SELECTION_ISOLATION_FLAGS
    )
    assert set(audit["selection_isolation"].values()) == {False}
    assert audit["post_submission_disposition"] == {
        "launch_artifacts_modified": False,
        "measurements_reused": True,
        "rerun_required": False,
    }
    unhashed = copy.deepcopy(audit)
    claimed = unhashed.pop("audit_sha256")
    assert (
        audit_module.manifest_generator.sha256_value(unhashed) == claimed
    )
    assert claimed == (
        "0253f5ccf422c5d26ce683e10e58bf199590bd6b78613d7d8243ef3847894425"
    )


def test_overlay_and_recovery_shape_mutations_fail_closed() -> None:
    overlay = audit_module.manifest_generator.load_json(
        audit_module.DEFAULT_OVERLAY
    )
    overlay["unexpected"] = False
    with pytest.raises(
        audit_module.IntegrityError,
        match="checkpoint overlay keys",
    ):
        audit_module.validate_exact_overlay_shape(overlay)

    overlay = audit_module.manifest_generator.load_json(
        audit_module.DEFAULT_OVERLAY
    )
    overlay["scope"]["dataset_uri"] = "s3://wrong/"
    with pytest.raises(
        audit_module.IntegrityError,
        match="checkpoint overlay scope",
    ):
        audit_module.validate_exact_overlay_shape(overlay)

    overlay = audit_module.manifest_generator.load_json(
        audit_module.DEFAULT_OVERLAY
    )
    overlay["selection_isolation"]["unexpected"] = False
    with pytest.raises(
        audit_module.IntegrityError,
        match="selection isolation keys",
    ):
        audit_module.validate_exact_overlay_shape(overlay)

    recovery = audit_module.manifest_generator.load_json(
        audit_module.DEFAULT_RECOVERY_EVIDENCE
    )
    del recovery["source_identity"]["candidate_specs"]["model.enc_layers"]
    with pytest.raises(
        audit_module.IntegrityError,
        match="recovery candidate spec keys",
    ):
        audit_module.validate_exact_recovery_shape(recovery)

    recovery = audit_module.manifest_generator.load_json(
        audit_module.DEFAULT_RECOVERY_EVIDENCE
    )
    recovery["recovery_attempts"][0]["checkpoint"]["sha256"] = "f" * 64
    rehash(recovery, "evidence_sha256")
    with pytest.raises(
        audit_module.checkpoint_overlay.OverlayError,
        match="checkpoint SHA256",
    ):
        audit_module.checkpoint_overlay.validate_recovery_evidence(
            recovery,
            "e" * 64,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "candidate_order",
        "block_plan",
        "command",
        "staging_bundle",
        "staged_plan",
        "job_identity",
        "selection_isolation",
    ),
)
def test_rehashed_ledger_mutations_are_rejected(
    tmp_path: Path,
    mutation: str,
) -> None:
    ledger = audit_module.manifest_generator.load_json(
        audit_module.DEFAULT_LEDGER
    )
    submission = ledger["submissions"][0]
    if mutation == "candidate_order":
        submission["candidate_order"][0:2] = reversed(
            submission["candidate_order"][0:2]
        )
    elif mutation == "block_plan":
        submission["block_plan_sha256"] = "f" * 64
    elif mutation == "command":
        submission["command_sha256"] = "f" * 64
    elif mutation == "staging_bundle":
        submission["staging_bundle_sha256"] = "f" * 64
    elif mutation == "staged_plan":
        key = "plans/latency_feasible_allocation_00.json"
        submission["staging_file_sha256"][key] = "f" * 64
    elif mutation == "job_identity":
        submission["tao_job_id"] = "ffffffff-ffff-ffff-ffff-ffffffffffff"
    elif mutation == "selection_isolation":
        ledger["measurements_feed_selection"] = True
    else:  # pragma: no cover - the parameter list is closed above.
        raise AssertionError(mutation)
    rehash(ledger, "ledger_sha256")
    path = tmp_path / "block_submissions.json"
    whole = write_json(path, ledger)
    arguments = live_arguments()
    arguments.update(
        {
            "ledger_path": path,
            "ledger_whole_file_sha256": whole,
            "ledger_internal_sha256": ledger["ledger_sha256"],
        }
    )

    with pytest.raises(
        audit_module.IntegrityError,
    ):
        audit_module.build_audit(**arguments)


def test_verifier_does_not_invoke_selector_or_source_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("selector/source replay was invoked")

    monkeypatch.setattr(
        audit_module.manifest_generator,
        "validate_completed_archive",
        forbidden,
    )
    monkeypatch.setattr(
        audit_module.launcher,
        "validate_final_source_evidence",
        forbidden,
    )

    audit = audit_module.build_audit(**live_arguments())
    assert audit["verifier_operations"]["selector_called"] is False
    assert audit["verifier_operations"]["analyze_archive_called"] is False


def test_optional_final_analysis_binds_all_plans_and_checkpoints(
    tmp_path: Path,
) -> None:
    base = audit_module.build_audit(**live_arguments())
    ledger = audit_module.manifest_generator.load_json(
        audit_module.DEFAULT_LEDGER
    )
    manifest = audit_module.manifest_generator.load_json(
        audit_module.DEFAULT_MANIFEST
    )
    overlay = audit_module.manifest_generator.load_json(
        audit_module.DEFAULT_OVERLAY
    )
    effective = audit_module.checkpoint_overlay.execution_manifest(
        manifest,
        overlay,
    )
    effective_candidates = {
        item["candidate_id"]: item for item in effective["candidates"]
    }
    submissions = {
        item["allocation_id"]: item for item in ledger["submissions"]
    }
    jobs = []
    measurements = []
    for allocation in manifest["schedule"]["allocations"]:
        submission = submissions[allocation["allocation_id"]]
        jobs.append(
            {
                key: copy.deepcopy(submission[key])
                for key in (
                    "allocation_id",
                    "allocation_index",
                    "design_row_index",
                    "candidate_order",
                    "block_plan_sha256",
                    "tao_job_id",
                    "slurm_job_id",
                )
            }
        )
        for position, candidate_id in enumerate(
            allocation["candidate_order"]
        ):
            measurements.append(
                {
                    "allocation_id": allocation["allocation_id"],
                    "candidate_id": candidate_id,
                    "position": position,
                    "tao_job_id": submission["tao_job_id"],
                    "slurm_job_id": submission["slurm_job_id"],
                    "checkpoint_sha256": effective_candidates[candidate_id][
                        "checkpoint"
                    ]["sha256"],
                }
            )
    report = {
        "schema_version": 1,
        "status": "complete",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": HASHES["manifest_whole_file_sha256"],
        "submission_ledger_sha256": HASHES[
            "ledger_whole_file_sha256"
        ],
        "schedule_sha256": manifest["schedule"]["schedule_sha256"],
        "candidate_ids": manifest["candidate_derivation"]["candidate_ids"],
        "source_checks": ledger["source_checks"],
        "jobs": jobs,
        "per_allocation_candidate_measurements": measurements,
        "checkpoint_overlay": {
            **audit_module.expected_overlay_source_checks(
                overlay,
                audit_module.DEFAULT_OVERLAY,
                HASHES["overlay_whole_file_sha256"],
            ),
            "parent_manifest_candidate_records_preserved": True,
            "parent_manifest_selection_snapshot_preserved": True,
            "execution_projection_substitution_count": 1,
            "accuracy_or_selection_evidence_from_recovered_checkpoint": False,
        },
        "parent_manifest_candidate_checkpoint_evidence": {
            item["candidate_id"]: item["checkpoint"]
            for item in manifest["candidates"]
        },
        "selection_isolation": {
            key: False for key in audit_module.SELECTION_ISOLATION_FLAGS
        },
    }
    rehash(report, "report_sha256")
    path = tmp_path / "latency_feasible_matched_analysis.json"
    whole = write_json(path, report)
    arguments = live_arguments()
    arguments.update(
        {
            "analysis_path": path,
            "analysis_whole_file_sha256": whole,
            "analysis_internal_sha256": report["report_sha256"],
        }
    )

    audit = audit_module.build_audit(**arguments)
    assert audit["matched_analysis"]["status"] == "validated_complete"
    assert audit["matched_analysis"]["job_count"] == 6
    assert audit["matched_analysis"]["measurement_cell_count"] == 24
    assert base["matched_analysis"]["status"] == "not_supplied"


def test_real_final_analysis_matches_emitted_immutable_audit() -> None:
    arguments = live_arguments()
    arguments.update(
        {
            "analysis_path": audit_module.DEFAULT_ANALYSIS,
            "analysis_whole_file_sha256": HASHES[
                "analysis_whole_file_sha256"
            ],
            "analysis_internal_sha256": HASHES[
                "analysis_internal_sha256"
            ],
        }
    )
    generated = audit_module.build_audit(**arguments)
    emitted = audit_module.manifest_generator.load_json(
        audit_module.DEFAULT_OUTPUT
    )

    assert generated == emitted
    assert generated["matched_analysis"]["status"] == "validated_complete"
    assert generated["matched_analysis"][
        "all_result_plan_bindings_match"
    ] is True
    assert generated["matched_analysis"][
        "all_effective_checkpoint_bindings_match"
    ] is True
    assert generated["audit_sha256"] == (
        "03083c5c1f604c3e47eee2644e1ef932266610bec41cf19a2dbf78b5ae483ac4"
    )
    assert audit_module.manifest_generator.sha256_file(
        audit_module.DEFAULT_OUTPUT
    ) == "03784ce116783c12d6373ad4e5842974deae75898b37af08cf672354c1611e7b"
