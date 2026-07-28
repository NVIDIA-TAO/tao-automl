"""Fail-closed tests for the phase-2 erratum and expanded archive authority."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import post_front_matched_launcher as launcher  # noqa: E402
import post_front_matched_manifest_generator as generator  # noqa: E402
import test_post_front_matched_tools as shared  # noqa: E402
import expanded_search_runner as expanded_runner  # noqa: E402


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rehash_archive(archive: dict[str, Any]) -> None:
    archive.pop("archive_sha256", None)
    archive["archive_sha256"] = generator.sha256_value(archive)


def _candidate_record(seed: int, rec_id: int) -> dict[str, Any]:
    candidate_id = f"seed_{seed}_rec_{rec_id}"
    model = {
        "model": {
            "backbone": "resnet_50",
            "enc_layers": 3 + rec_id % 4,
            "dec_layers": 3 + (rec_id // 4) % 4,
        },
        "candidate_id": candidate_id,
    }
    return {
        "candidate_id": candidate_id,
        "search_seed": seed,
        "training_seed": generator.EXPECTED_TRAINING_SEED,
        "rec_id": rec_id,
        "status": "success",
        "specs": {
            "model.enc_layers": model["model"]["enc_layers"],
            "model.dec_layers": model["model"]["dec_layers"],
        },
        "resolved_model_spec": model,
        "resolved_model_spec_sha256": generator.sha256_value(model),
        "checkpoint": {
            "path": f"/lustre/checkpoints/{candidate_id}.pth",
            "sha256": SHA_C,
        },
        "train_job_id": f"train-{candidate_id}",
        "training_runtime": {"status": "Complete"},
        "accuracy_evaluation": {"status": "Complete"},
        "selection_time_latency": {"status": "Complete"},
        "objective_values": {
            "mAP50": 0.5 + rec_id / 1000.0,
            "latency_ms": 50.0 + rec_id,
            "latency_p95_ms": 51.0 + rec_id,
            "latency_ci95_low": 49.9 + rec_id,
            "latency_ci95_high": 50.1 + rec_id,
        },
        "failure_reason": None,
        "search_manifest_file_sha256": (
            generator.EXPECTED_EXPANDED_MANIFEST_FILE_SHA256
        ),
        "manual_candidate_injection_used": False,
    }


def _archive_fixture(tmp_path: Path) -> dict[str, Any]:
    paths: list[Path] = []
    records_by_id: dict[str, dict[str, Any]] = {}
    for seed in generator.EXPECTED_SEARCH_SEEDS:
        records = {
            f"seed_{seed}_rec_{rec_id}": _candidate_record(seed, rec_id)
            for rec_id in range(
                generator.EXPECTED_RECOMMENDATIONS_PER_SEED
            )
        }
        archive = {
            "schema_version": 1,
            "status": "complete",
            "created_at_utc": "2026-07-28T12:00:00Z",
            "manifest_file_sha256": (
                generator.EXPECTED_EXPANDED_MANIFEST_FILE_SHA256
            ),
            "search_seed": seed,
            "recommendations": (
                generator.EXPECTED_RECOMMENDATIONS_PER_SEED
            ),
            "manual_candidate_injection_used": False,
            "records": records,
            "automl_result_sha256": SHA_A,
        }
        _rehash_archive(archive)
        path = tmp_path / f"seed_{seed}" / "seed_archive.v1.json"
        _write_json(path, archive)
        paths.append(path)
        records_by_id.update(copy.deepcopy(records))

    audits = [
        {
            "candidate_id": candidate_id,
            "valid": True,
            "pareto_rank": 0,
            "dominated_by": [],
        }
        for candidate_id in sorted(records_by_id)
    ]
    audit_by_id = {
        item["candidate_id"]: item for item in audits
    }
    rows = [
        generator.candidate_table_projection(
            records_by_id[candidate_id],
            audit_by_id[candidate_id],
        )
        for candidate_id in sorted(records_by_id)
    ]
    combined = {"candidates": audits}
    table = {
        "schema_version": 1,
        "candidate_count": generator.EXPECTED_TOTAL_CANDIDATES,
        "successful_count": generator.EXPECTED_TOTAL_CANDIDATES,
        "manual_candidate_injection_used": False,
        "rows": rows,
    }
    csv_path = tmp_path / "expanded_candidate_table.csv"
    csv_path.write_bytes(
        generator.candidate_table_csv_bytes(
            records_by_id,
            audit_by_id,
        )
    )
    return {
        "seed_archive_paths": paths,
        "expanded_manifest_sha256": (
            generator.EXPECTED_EXPANDED_MANIFEST_FILE_SHA256
        ),
        "combined": combined,
        "combined_sha256": generator.sha256_value(combined),
        "table": table,
        "table_sha256": generator.sha256_value(table),
        "candidate_table_csv_path": csv_path,
        "candidate_table_csv_sha256": generator.sha256_file(csv_path),
        "integrity_sha256": SHA_B,
    }


def _validate(fixture: dict[str, Any]) -> dict[str, Any]:
    return generator.validate_expanded_archive_authority(**fixture)


def test_exact_protocol_erratum_and_both_policy_branches_are_accepted() -> None:
    erratum, whole, policy = (
        generator.load_and_validate_protocol_erratum(
            generator.DEFAULT_PROTOCOL_ERRATUM,
            generator.EXPECTED_PROTOCOL_ERRATUM_FILE_SHA256,
        )
    )
    assert whole == generator.EXPECTED_PROTOCOL_ERRATUM_FILE_SHA256
    assert (
        generator.sha256_value(erratum)
        == generator.EXPECTED_PROTOCOL_ERRATUM_CANONICAL_SHA256
    )
    assert set(policy) == {
        "original_preregistered_bootstrap_classification",
        "effective_erratum_directional_classification",
        "both_policy_branches_must_be_emitted",
    }
    assert policy["both_policy_branches_must_be_emitted"] is True


def test_protocol_erratum_missing_or_tampered_fails_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(generator.ContractError, match="cannot load"):
        generator.load_and_validate_protocol_erratum(
            tmp_path / "missing.json",
            generator.EXPECTED_PROTOCOL_ERRATUM_FILE_SHA256,
        )

    erratum = generator.load_json(generator.DEFAULT_PROTOCOL_ERRATUM)
    erratum["invariants"]["manual_winner_override_permitted"] = True
    path = tmp_path / "tampered.json"
    _write_json(path, erratum)
    with pytest.raises(
        generator.ContractError,
        match="whole-file SHA256",
    ):
        generator.load_and_validate_protocol_erratum(
            path,
            generator.sha256_file(path),
        )


def test_generator_and_launcher_cli_require_exact_erratum_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator_arguments = [
        "generator",
        "--expanded-manifest-sha256",
        SHA_A,
        "--combined-selection-sha256",
        SHA_A,
        "--candidate-table-sha256",
        SHA_A,
        "--integrity-audit-sha256",
        SHA_A,
    ]
    monkeypatch.setattr(sys, "argv", generator_arguments)
    with pytest.raises(SystemExit):
        generator.parse_args()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *generator_arguments,
            "--protocol-erratum",
            str(generator.DEFAULT_PROTOCOL_ERRATUM),
            "--protocol-erratum-sha256",
            generator.EXPECTED_PROTOCOL_ERRATUM_FILE_SHA256,
        ],
    )
    parsed_generator = generator.parse_args()
    assert (
        parsed_generator.protocol_erratum_sha256
        == generator.EXPECTED_PROTOCOL_ERRATUM_FILE_SHA256
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["launcher", "--manifest-file-sha256", SHA_A],
    )
    with pytest.raises(SystemExit):
        launcher.parse_args()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launcher",
            "--manifest-file-sha256",
            SHA_A,
            "--protocol-erratum",
            str(generator.DEFAULT_PROTOCOL_ERRATUM),
            "--protocol-erratum-sha256",
            generator.EXPECTED_PROTOCOL_ERRATUM_FILE_SHA256,
        ],
    )
    parsed_launcher = launcher.parse_args()
    assert (
        parsed_launcher.protocol_erratum_sha256
        == generator.EXPECTED_PROTOCOL_ERRATUM_FILE_SHA256
    )


def test_three_seed_archive_authority_binds_exact_full_union(
    tmp_path: Path,
) -> None:
    fixture = _archive_fixture(tmp_path)
    snapshot = _validate(fixture)
    assert snapshot["search_seeds"] == list(
        generator.EXPECTED_SEARCH_SEEDS
    )
    assert [item["search_seed"] for item in snapshot["seed_archives"]] == list(
        generator.EXPECTED_SEARCH_SEEDS
    )
    assert snapshot["candidate_count"] == 60
    assert snapshot["terminal_candidate_count"] == 60
    assert snapshot["successful_candidate_count"] == 60
    assert snapshot["failed_candidate_count"] == 0
    assert len(snapshot["candidate_ids"]) == 60
    assert snapshot["candidate_ids"] == sorted(snapshot["candidate_ids"])
    assert snapshot["candidate_ids_sha256"] == generator.sha256_value(
        snapshot["candidate_ids"]
    )
    assert all(
        item["record_count"] == 20
        and item["terminal_record_count"] == 20
        and len(item["whole_file_sha256"]) == 64
        and len(item["internal_archive_sha256"]) == 64
        for item in snapshot["seed_archives"]
    )


def test_csv_reconstruction_matches_the_pinned_runner_writer(
    tmp_path: Path,
) -> None:
    fixture = _archive_fixture(tmp_path / "fixture")
    records = []
    for path in fixture["seed_archive_paths"]:
        archive = generator.load_json(path)
        records.extend(copy.deepcopy(list(archive["records"].values())))
    output_dir = tmp_path / "runner_output"
    output_dir.mkdir()
    artifacts = expanded_runner.write_candidate_artifacts(
        {
            "search_space": {
                "search_parameters": list(
                    generator.EXPECTED_SEARCH_PARAMETERS
                )
            }
        },
        list(reversed(records)),
        fixture["combined"],
        output_dir,
    )
    runner_csv = Path(artifacts["candidate_table_csv"]).read_bytes()
    assert runner_csv == fixture["candidate_table_csv_path"].read_bytes()
    assert runner_csv == generator.candidate_table_csv_bytes(
        {
            record["candidate_id"]: record
            for record in records
        },
        {
            audit["candidate_id"]: audit
            for audit in fixture["combined"]["candidates"]
        },
    )


def test_seed_archive_input_order_does_not_change_snapshot(
    tmp_path: Path,
) -> None:
    fixture = _archive_fixture(tmp_path)
    reference = _validate(fixture)
    reordered = copy.deepcopy(fixture)
    reordered["seed_archive_paths"] = list(
        reversed(fixture["seed_archive_paths"])
    )
    assert _validate(reordered) == reference


def test_missing_seed_archive_file_is_rejected(tmp_path: Path) -> None:
    fixture = _archive_fixture(tmp_path)
    fixture["seed_archive_paths"][1].unlink()
    with pytest.raises(generator.ContractError, match="cannot load"):
        _validate(fixture)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_seed_archive_missing_or_extra_record_is_rejected(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _archive_fixture(tmp_path)
    path = fixture["seed_archive_paths"][0]
    archive = generator.load_json(path)
    if mutation == "missing":
        archive["records"].pop(next(iter(archive["records"])))
    else:
        archive["records"]["unexpected"] = copy.deepcopy(
            next(iter(archive["records"].values()))
        )
    _rehash_archive(archive)
    _write_json(path, archive)
    with pytest.raises(generator.ContractError, match="candidate IDs"):
        _validate(fixture)


def test_seed_archive_duplicate_path_or_seed_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _archive_fixture(tmp_path)
    duplicate_path = copy.deepcopy(fixture)
    duplicate_path["seed_archive_paths"] = [
        fixture["seed_archive_paths"][0],
        fixture["seed_archive_paths"][0],
        fixture["seed_archive_paths"][2],
    ]
    with pytest.raises(generator.ContractError, match="paths must be unique"):
        _validate(duplicate_path)

    source = fixture["seed_archive_paths"][0]
    duplicate = (
        tmp_path
        / "duplicate"
        / source.parent.name
        / "seed_archive.v1.json"
    )
    _write_json(duplicate, generator.load_json(source))
    duplicate_seed = copy.deepcopy(fixture)
    duplicate_seed["seed_archive_paths"] = [
        fixture["seed_archive_paths"][0],
        duplicate,
        fixture["seed_archive_paths"][2],
    ]
    with pytest.raises(generator.ContractError, match="duplicate seed"):
        _validate(duplicate_seed)


def test_seed_archive_wrong_seed_and_nonterminal_record_are_rejected(
    tmp_path: Path,
) -> None:
    fixture = _archive_fixture(tmp_path)
    path = fixture["seed_archive_paths"][0]
    archive = generator.load_json(path)
    archive["search_seed"] = 999
    _rehash_archive(archive)
    _write_json(path, archive)
    with pytest.raises(generator.ContractError, match="unexpected search seed"):
        _validate(fixture)

    fixture = _archive_fixture(tmp_path / "nonterminal")
    path = fixture["seed_archive_paths"][0]
    archive = generator.load_json(path)
    record = next(iter(archive["records"].values()))
    record["status"] = "evaluating"
    _rehash_archive(archive)
    _write_json(path, archive)
    with pytest.raises(generator.ContractError, match="not terminal"):
        _validate(fixture)


@pytest.mark.parametrize(
    ("target", "value", "message"),
    [
        ("manifest_file_sha256", SHA_A, "expanded-manifest"),
        ("recommendations", 19, "recommendation count"),
        ("manual_candidate_injection_used", True, "manual candidate injection"),
        ("status", "incomplete", "status"),
    ],
)
def test_seed_archive_identity_contract_is_exact(
    tmp_path: Path,
    target: str,
    value: Any,
    message: str,
) -> None:
    fixture = _archive_fixture(tmp_path)
    path = fixture["seed_archive_paths"][0]
    archive = generator.load_json(path)
    archive[target] = value
    _rehash_archive(archive)
    _write_json(path, archive)
    with pytest.raises(generator.ContractError, match=message):
        _validate(fixture)


def test_record_manual_injection_or_manifest_drift_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _archive_fixture(tmp_path)
    path = fixture["seed_archive_paths"][0]
    archive = generator.load_json(path)
    record = next(iter(archive["records"].values()))
    record["manual_candidate_injection_used"] = True
    _rehash_archive(archive)
    _write_json(path, archive)
    with pytest.raises(generator.ContractError, match="manual candidate"):
        _validate(fixture)

    fixture = _archive_fixture(tmp_path / "record_manifest")
    path = fixture["seed_archive_paths"][0]
    archive = generator.load_json(path)
    record = next(iter(archive["records"].values()))
    record["search_manifest_file_sha256"] = SHA_A
    _rehash_archive(archive)
    _write_json(path, archive)
    with pytest.raises(generator.ContractError, match="manifest binding"):
        _validate(fixture)


def test_seed_archive_internal_and_whole_hashes_are_independent_authorities(
    tmp_path: Path,
) -> None:
    fixture = _archive_fixture(tmp_path)
    path = fixture["seed_archive_paths"][0]
    original_whole = generator.sha256_file(path)
    archive = generator.load_json(path)
    first = next(iter(archive["records"].values()))
    first["objective_values"]["mAP50"] += 0.01
    _write_json(path, archive)
    assert generator.sha256_file(path) != original_whole
    with pytest.raises(generator.ContractError, match="canonical digest"):
        _validate(fixture)
    with pytest.raises(generator.ContractError, match="whole-file SHA256"):
        generator.load_exact_json(path, original_whole, "seed archive")


def test_candidate_table_projection_drift_and_duplicate_rows_are_rejected(
    tmp_path: Path,
) -> None:
    fixture = _archive_fixture(tmp_path)
    drift = copy.deepcopy(fixture)
    drift["table"]["rows"][0]["objective_values"]["mAP50"] += 0.1
    drift["table_sha256"] = generator.sha256_value(drift["table"])
    with pytest.raises(generator.ContractError, match="projection"):
        _validate(drift)

    duplicate = copy.deepcopy(fixture)
    duplicate["table"]["rows"][-1] = copy.deepcopy(
        duplicate["table"]["rows"][0]
    )
    duplicate["table_sha256"] = generator.sha256_value(duplicate["table"])
    with pytest.raises(generator.ContractError, match="duplicate"):
        _validate(duplicate)


def test_candidate_table_row_order_does_not_change_canonical_union(
    tmp_path: Path,
) -> None:
    fixture = _archive_fixture(tmp_path)
    reference = _validate(fixture)
    reordered = copy.deepcopy(fixture)
    reordered["table"]["rows"].reverse()
    assert _validate(reordered)["full_record_union_sha256"] == reference[
        "full_record_union_sha256"
    ]
    assert _validate(reordered)[
        "candidate_table_projection_sha256"
    ] == reference["candidate_table_projection_sha256"]


def test_csv_hash_drift_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _archive_fixture(tmp_path)
    fixture["candidate_table_csv_path"].write_text(
        "candidate_id\ntampered\n",
        encoding="utf-8",
    )
    with pytest.raises(generator.ContractError, match="CSV whole-file"):
        _validate(fixture)


def test_rehashed_semantically_drifted_csv_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _archive_fixture(tmp_path)
    path = fixture["candidate_table_csv_path"]
    payload = path.read_text(encoding="utf-8").replace(
        "seed_314159_rec_0",
        "tampered_candidate",
        1,
    )
    path.write_text(payload, encoding="utf-8")
    fixture["candidate_table_csv_sha256"] = generator.sha256_file(path)
    with pytest.raises(
        generator.ContractError,
        match="CSV exact archive projection",
    ):
        _validate(fixture)


@pytest.mark.parametrize(
    ("snapshot_key", "source_key", "message"),
    [
        (
            "expanded_combined_selection_sha256",
            "expanded_combined_selection",
            "combined selection",
        ),
        (
            "expanded_candidate_table_sha256",
            "expanded_candidate_table",
            "candidate table JSON",
        ),
        (
            "expanded_candidate_table_csv_sha256",
            "expanded_candidate_table_csv",
            "candidate table CSV",
        ),
        (
            "expanded_integrity_audit_sha256",
            "expanded_integrity_audit",
            "integrity audit",
        ),
    ],
)
def test_top_level_output_hash_drift_is_rejected(
    snapshot_key: str,
    source_key: str,
    message: str,
) -> None:
    manifest = shared._minimal_manifest(["candidate_a", "candidate_b"])
    manifest["source_artifacts"][source_key]["sha256"] = (
        SHA_B
        if manifest["expanded_archive_snapshot"][snapshot_key] != SHA_B
        else SHA_C
    )
    with pytest.raises(
        launcher.ContractError,
        match=message,
    ):
        launcher.validate_archive_snapshot_contract(manifest)


@pytest.mark.parametrize(
    "branch",
    [
        "original_preregistered_bootstrap_classification",
        "effective_erratum_directional_classification",
    ],
)
def test_paired_policy_branches_are_exact_and_independent(
    branch: str,
) -> None:
    erratum, _, policy = generator.load_and_validate_protocol_erratum(
        generator.DEFAULT_PROTOCOL_ERRATUM,
        generator.EXPECTED_PROTOCOL_ERRATUM_FILE_SHA256,
    )
    manifest = shared._minimal_manifest(
        ["candidate_a", "candidate_b", "candidate_c"]
    )
    launcher.validate_paired_policy_binding(manifest, policy)
    assert (
        manifest["paired_analysis"][branch]
        == erratum["corrections"]["post_front_paired_classification"][branch]
    )
    tampered = copy.deepcopy(manifest)
    tampered["paired_analysis"][branch]["status"] = "tampered"
    with pytest.raises(
        launcher.ContractError,
        match="paired-analysis policy",
    ):
        launcher.validate_paired_policy_binding(tampered, policy)


def test_archive_snapshot_binds_seed_internal_hash_and_csv_hash() -> None:
    manifest = shared._minimal_manifest(["candidate_a", "candidate_b"])
    launcher.validate_archive_snapshot_contract(manifest)
    tampered_seed = copy.deepcopy(manifest)
    tampered_seed["source_artifacts"]["expanded_seed_archives"][0][
        "internal_sha256"
    ] = SHA_A
    with pytest.raises(
        launcher.ContractError,
        match="seed-archive source/snapshot",
    ):
        launcher.validate_archive_snapshot_contract(tampered_seed)
    tampered_csv = copy.deepcopy(manifest)
    tampered_csv["source_artifacts"]["expanded_candidate_table_csv"][
        "sha256"
    ] = SHA_A
    with pytest.raises(
        launcher.ContractError,
        match="candidate table CSV",
    ):
        launcher.validate_archive_snapshot_contract(tampered_csv)
