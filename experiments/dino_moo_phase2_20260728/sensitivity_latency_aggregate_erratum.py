#!/usr/bin/env python3

"""Analysis-only erratum for DINO sensitivity-latency v2 aggregation.

The v2 measurement manifest and submission ledger are immutable.  The
measurement-time aggregator named by that manifest is also retained byte for
byte.  This entrypoint reuses its provenance, scheduling, statistics, and
qualification utilities while correcting only the allocation-level PyTorch
version comparison to use the v2-declared major.minor.patch policy.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shlex
import sqlite3
import stat
import subprocess
import sys
import tempfile
from typing import Any

import sensitivity_latency_aggregate as original
from sensitivity_latency_common import (
    build_profiles,
    build_schedule,
    load_accuracy_artifact,
    load_checkpoint_artifact,
    load_contract,
    sha256_file,
    sha256_value,
)


HERE = Path(__file__).resolve().parent
DEFAULT_ERRATUM = HERE / "sensitivity_latency_analysis_erratum.v1.json"
CORRECTED_SOURCE_PATH = Path(__file__).resolve()
ORIGINAL_AGGREGATOR_PATH = HERE / "sensitivity_latency_aggregate.py"
EXPECTED_LAUNCH_SOURCE_KEYS = {
    "aggregator_sha256",
    "automl_branch",
    "automl_commit",
    "automl_required_ancestor_commit",
    "benchmark_sha256",
    "block_runner_sha256",
    "common_sha256",
    "evaluate_template_sha256",
    "latency_stats_sha256",
    "launcher_sha256",
    "sdk_branch",
    "sdk_commit",
    "submission_source_state",
}
EXPECTED_EVIDENCE_ACQUISITION_POLICY = {
    "mode": "read_only_remote_exact_inventory_snapshot",
    "remote_transport": "ssh_batch_mode_plus_rsync_files_from",
    "remote_results_anchor": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/"
        "rarunachalam/results"
    ),
    "local_snapshot_parent": "runtime/sensitivity_latency_v2",
    "path_derivation": (
        "trusted ledger SDK job root plus regenerated allocation plan; "
        "no discovery globbing"
    ),
    "expected_allocation_result_count": 9,
    "expected_rank_result_count": 1008,
    "expected_file_count": 1017,
    "remote_file_type": "regular_non_symlink",
    "remote_hash_stability": "lstat_identity_equal_before_and_after_hash",
    "digest_algorithm": "sha256",
    "transfer_scope": "missing_expected_files_only",
    "existing_file_policy": "accept_only_when_remote_and_local_sha256_match",
    "overwrite_policy": "forbidden",
    "extra_file_policy": "reject",
    "missing_file_policy": "reject_after_transfer",
    "duplicate_path_policy": "reject",
    "embedded_remote_path_policy": "validate_without_rewriting",
    "remote_write_permitted": False,
}
EXPECTED_SDK_STATE_INSPECTION_POLICY = {
    "mode": "read_only_durable_sqlite_plus_terminal_sacct",
    "json_sidecar_role": "optional_non_authoritative_monitor_cache",
    "database_access": "sqlite_uri_mode_ro_query_only",
    "database_snapshot_required_stable": True,
    "required_database_job_set": "exact_immutable_ledger_tao_job_ids",
    "accepted_durable_statuses": [
        "Complete",
        "Pending",
        "Running",
        "Paused",
    ],
    "stale_nonterminal_acceptance_gate": (
        "exact_job_runtime_artifact_identity_and_sacct_COMPLETED_0_0"
    ),
    "rejected_durable_statuses": [
        "Error",
        "Canceled",
        "Canceling",
        "Unknown",
    ],
    "scheduler_evidence": "exact_pinned_partition_account_one_node_eight_gpu",
    "sdk_database_mutation_permitted": False,
    "sdk_monitor_or_retry_permitted": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-erratum", type=Path, default=DEFAULT_ERRATUM)
    parser.add_argument("--analysis-erratum-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-artifact", type=Path, required=True)
    parser.add_argument("--checkpoint-artifact-sha256", required=True)
    parser.add_argument("--accuracy-artifact", type=Path, required=True)
    parser.add_argument("--accuracy-artifact-sha256", required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--submission-ledger-sha256", required=True)
    parser.add_argument("--sdk-state", type=Path, required=True)
    parser.add_argument("--evidence-snapshot", type=Path, required=True)
    parser.add_argument("--secrets-env", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def measurement_policy_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the complete measurement-generating policy subset."""

    return {
        key: manifest[key]
        for key in (
            "frozen_inputs",
            "design",
            "runtime_contract",
            "latency_protocol",
            "evaluation_config_contract",
            "checkpoint_reuse_contract",
            "submission_policy",
        )
    }


def qualification_policy_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the complete analysis/selection-policy subset."""

    return {
        "feeds_final_selection": manifest["feeds_final_selection"],
        "manual_promotion_permitted": manifest[
            "manual_promotion_permitted"
        ],
        "aggregation": manifest["aggregation"],
    }


def validate_analysis_erratum(
    erratum_path: Path,
    expected_erratum_sha256: str,
    manifest_path: Path,
    submission_ledger_path: Path,
    expected_submission_ledger_sha256: str,
    *,
    corrected_source_path: Path = CORRECTED_SOURCE_PATH,
    original_aggregator_path: Path = ORIGINAL_AGGREGATOR_PATH,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the separate erratum without creating a hash cycle.

    The caller supplies the immutable erratum-file digest.  The erratum pins
    this corrected source digest, while the source contains no erratum digest,
    avoiding a source/contract self-reference.
    """

    erratum_path = erratum_path.resolve()
    manifest_path = manifest_path.resolve()
    submission_ledger_path = submission_ledger_path.resolve()
    erratum, actual_erratum_sha256 = original.read_hashed_json(
        erratum_path, "analysis erratum"
    )
    if actual_erratum_sha256 != expected_erratum_sha256:
        raise RuntimeError("analysis erratum digest mismatch")
    if (
        erratum.get("schema_version") != 1
        or erratum.get("erratum_id")
        != "dino_sensitivity_latency_analysis_erratum_20260728_v1"
        or erratum.get("status") != "approved_analysis_only"
        or erratum.get("scope")
        != "aggregation_validation_evidence_access_and_descendant_commit_only"
        or erratum.get("measurement_generation_unchanged") is not True
        or erratum.get("qualification_policy_unchanged") is not True
        or erratum.get("objective_values_altered") is not False
        or erratum.get("winner_selected") is not False
        or erratum.get("feeds_final_selection") is not False
        or erratum.get("manual_promotion_permitted") is not False
        or erratum.get("reason_code")
        != "allocation_torch_version_used_full_string_instead_of_declared_base_release"
        or erratum.get("reason_codes")
        != [
            (
                "allocation_torch_version_used_full_string_instead_of_"
                "declared_base_release"
            ),
            "analysis_head_descends_from_immutable_launch_commit",
            "remote_results_not_locally_mounted",
            "unwatched_sdk_jobs_have_no_sidecar_and_stale_pending_rows",
        ]
    ):
        raise ValueError("analysis erratum identity or policy mismatch")

    measurement = erratum.get("measurement_contract")
    if not isinstance(measurement, dict):
        raise ValueError("analysis erratum measurement contract missing")
    pinned_manifest_path = (
        erratum_path.parent / str(measurement.get("manifest_path", ""))
    ).resolve()
    pinned_ledger_path = (
        erratum_path.parent
        / str(measurement.get("submission_ledger_path", ""))
    ).resolve()
    if pinned_manifest_path != manifest_path:
        raise ValueError("analysis erratum manifest path mismatch")
    if pinned_ledger_path != submission_ledger_path:
        raise ValueError("analysis erratum submission ledger path mismatch")

    manifest, actual_manifest_sha256 = original.read_hashed_json(
        manifest_path, "measurement manifest"
    )
    if (
        manifest.get("manifest_id")
        != "dino_sensitivity_latency_20260728_v2"
        or measurement.get("manifest_id") != manifest.get("manifest_id")
        or measurement.get("manifest_sha256") != actual_manifest_sha256
        or actual_manifest_sha256
        != "aedc117414b2691c1a70b73fa4e9e0ac123cb4d20dfd9d25dfe2d4aa490d7655"
    ):
        raise ValueError("analysis erratum measurement manifest mismatch")

    actual_ledger_sha256 = sha256_file(submission_ledger_path)
    if (
        expected_submission_ledger_sha256
        != measurement.get("submission_ledger_sha256")
        or actual_ledger_sha256 != expected_submission_ledger_sha256
        or actual_ledger_sha256
        != "b1c170c0d4697463d171cbeca3e4adcbd34cc1cb7429c236f48b58c46c3b6d54"
        or measurement.get("launch_automl_branch")
        != "rarunachalam/pre-platform-sdk-removal-20260714"
        or measurement.get("launch_automl_commit")
        != "cb62ef447704b95980b17aa82604992564b4e71f"
    ):
        raise ValueError("analysis erratum submission ledger mismatch")

    source_pins = erratum.get("source_pins")
    if not isinstance(source_pins, dict):
        raise ValueError("analysis erratum source pins missing")
    measurement_sources = manifest["runtime_contract"]["source_code_sha256"]
    expected_measurement_sources = {
        "launcher": measurement_sources["launcher"],
        "block_runner": measurement_sources["block_runner"],
        "common": measurement_sources["common"],
        "original_aggregator": measurement_sources["aggregator"],
        "latency_stats": measurement_sources["latency_stats"],
    }
    if source_pins.get("measurement_generation") != expected_measurement_sources:
        raise ValueError("analysis erratum measurement source identity mismatch")
    actual_original_sha256 = sha256_file(original_aggregator_path.resolve())
    if (
        actual_original_sha256
        != "5f5aebd4274c746ec9674f28f978af5d228d98c6ba0af8d76cff8b1742dab967"
        or source_pins.get("original_aggregator_sha256")
        != actual_original_sha256
    ):
        raise ValueError("analysis erratum original aggregator source mismatch")
    actual_corrected_sha256 = sha256_file(corrected_source_path.resolve())
    if source_pins.get("corrected_aggregator_sha256") != actual_corrected_sha256:
        raise ValueError("analysis erratum corrected aggregator source mismatch")

    policy_pins = erratum.get("unchanged_policy_pins")
    if not isinstance(policy_pins, dict):
        raise ValueError("analysis erratum policy pins missing")
    measurement_policy_sha256 = sha256_value(
        measurement_policy_payload(manifest)
    )
    qualification_policy_sha256 = sha256_value(
        qualification_policy_payload(manifest)
    )
    if (
        policy_pins.get("measurement_policy_sha256")
        != measurement_policy_sha256
        or policy_pins.get("qualification_policy_sha256")
        != qualification_policy_sha256
    ):
        raise ValueError("analysis erratum policy fingerprint mismatch")

    correction = erratum.get("correction")
    if (
        not isinstance(correction, dict)
        or correction.get("field") != "allocation.hardware.runtime.torch"
        or correction.get("old_comparison") != "full_string_exact"
        or correction.get("new_comparison") != "major_minor_patch"
        or correction.get("declared_by")
        != "measurement_manifest.runtime_contract.torch_version_match"
        or correction.get("raw_runtime_string_preserved") is not True
        or correction.get("measurement_values_recomputed") is not False
    ):
        raise ValueError("analysis erratum correction scope mismatch")
    acquisition_policy = erratum.get("evidence_acquisition_policy")
    if acquisition_policy != EXPECTED_EVIDENCE_ACQUISITION_POLICY:
        raise ValueError("analysis erratum evidence acquisition policy mismatch")
    sdk_state_policy = erratum.get("sdk_state_inspection_policy")
    if sdk_state_policy != EXPECTED_SDK_STATE_INSPECTION_POLICY:
        raise ValueError("analysis erratum SDK state inspection policy mismatch")
    commit_correction = erratum.get("analysis_commit_correction")
    if (
        not isinstance(commit_correction, dict)
        or commit_correction.get("launch_identity_source")
        != "immutable_submission_ledger.source_checks"
        or commit_correction.get("analysis_identity_source")
        != "git_HEAD_and_current_validated_source_checks"
        or commit_correction.get("branch_policy") != "exact_same_branch"
        or commit_correction.get("commit_policy")
        != "launch_commit_is_ancestor_of_analysis_commit"
        or commit_correction.get("measurement_source_hash_policy")
        != "exact_immutable_manifest_values"
        or commit_correction.get("command_plan_policy")
        != "regenerate_current_and_reconcile_against_immutable_ledger"
        or commit_correction.get("source_hash_weakening_permitted") is not False
    ):
        raise ValueError("analysis erratum commit correction mismatch")

    identity = {
        "erratum_id": erratum["erratum_id"],
        "erratum_path": str(erratum_path),
        "erratum_sha256": actual_erratum_sha256,
        "reason_code": erratum["reason_code"],
        "measurement_manifest_id": manifest["manifest_id"],
        "measurement_manifest_path": str(manifest_path),
        "measurement_manifest_sha256": actual_manifest_sha256,
        "submission_ledger_path": str(submission_ledger_path),
        "submission_ledger_sha256": actual_ledger_sha256,
        "original_aggregator_sha256": actual_original_sha256,
        "corrected_aggregator_sha256": actual_corrected_sha256,
        "measurement_policy_sha256": measurement_policy_sha256,
        "qualification_policy_sha256": qualification_policy_sha256,
        "evidence_acquisition_policy_sha256": sha256_value(
            acquisition_policy
        ),
        "sdk_state_inspection_policy_sha256": sha256_value(
            sdk_state_policy
        ),
        "measurement_generation_unchanged": True,
        "qualification_policy_unchanged": True,
        "objective_values_altered": False,
        "raw_runtime_string_preserved": True,
    }
    return erratum, manifest, identity


def git_output(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def git_is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(
            "git ancestry verification failed: "
            f"{completed.stderr.strip()}"
        )
    return completed.returncode == 0


def expected_fixed_launch_source_checks(
    contract: dict[str, Any],
) -> dict[str, str]:
    frozen = contract["frozen_inputs"]
    runtime = contract["runtime_contract"]
    sources = runtime["source_code_sha256"]
    return {
        "aggregator_sha256": sources["aggregator"],
        "automl_branch": runtime["automl_branch"],
        "automl_required_ancestor_commit": runtime[
            "automl_required_ancestor_commit"
        ],
        "benchmark_sha256": frozen["benchmark_sha256"],
        "block_runner_sha256": sources["block_runner"],
        "common_sha256": sources["common"],
        "evaluate_template_sha256": frozen["evaluate_template_sha256"],
        "latency_stats_sha256": sources["latency_stats"],
        "launcher_sha256": sources["launcher"],
        "sdk_branch": runtime["sdk_branch"],
        "sdk_commit": runtime["sdk_commit"],
        "submission_source_state": "tracked_and_clean",
    }


def validate_launch_source_checks(
    recorded: Any,
    current: dict[str, str],
    contract: dict[str, Any],
) -> tuple[str, str]:
    """Validate immutable launch sources and current analysis sources."""

    if not isinstance(recorded, dict) or set(recorded) != (
        EXPECTED_LAUNCH_SOURCE_KEYS
    ):
        raise ValueError("submission ledger launch source key set drift")
    fixed = expected_fixed_launch_source_checks(contract)
    for key, expected in fixed.items():
        if recorded.get(key) != expected:
            raise ValueError(f"submission ledger launch source drift: {key}")
    launch_commit = recorded.get("automl_commit")
    if (
        not isinstance(launch_commit, str)
        or len(launch_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in launch_commit
        )
    ):
        raise ValueError("submission ledger launch commit is invalid")

    expected_current_keys = EXPECTED_LAUNCH_SOURCE_KEYS - {
        "submission_source_state"
    }
    if set(current) != expected_current_keys:
        raise ValueError("current analysis source key set drift")
    for key, expected in fixed.items():
        if key == "submission_source_state":
            continue
        if current.get(key) != expected:
            raise ValueError(f"current analysis source drift: {key}")
    analysis_commit = current.get("automl_commit")
    if (
        not isinstance(analysis_commit, str)
        or len(analysis_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in analysis_commit
        )
    ):
        raise ValueError("current analysis commit is invalid")
    return launch_commit, analysis_commit


def validate_launch_analysis_provenance(
    ledger_path: Path,
    expected_ledger_sha256: str,
    contract: dict[str, Any],
    current_source_checks: dict[str, str],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Prove that analysis HEAD descends from the exact launch commit."""

    ledger, actual_sha256 = original.read_hashed_json(
        ledger_path.resolve(), "submission ledger ancestry proof"
    )
    if (
        actual_sha256 != expected_ledger_sha256
        or actual_sha256
        != "b1c170c0d4697463d171cbeca3e4adcbd34cc1cb7429c236f48b58c46c3b6d54"
    ):
        raise RuntimeError("immutable submission ledger digest mismatch")
    recorded = ledger.get("source_checks")
    launch_commit, analysis_commit = validate_launch_source_checks(
        recorded, current_source_checks, contract
    )
    repo = Path(contract["runtime_contract"]["automl_path"]).resolve()
    actual_head = git_output(repo, "rev-parse", "HEAD")
    actual_branch = git_output(repo, "branch", "--show-current")
    required_branch = contract["runtime_contract"]["automl_branch"]
    if (
        actual_head != analysis_commit
        or actual_branch != required_branch
        or recorded["automl_branch"] != required_branch
        or current_source_checks["automl_branch"] != required_branch
    ):
        raise ValueError("launch/analysis branch or HEAD identity drift")
    required_ancestor = contract["runtime_contract"][
        "automl_required_ancestor_commit"
    ]
    if not git_is_ancestor(repo, required_ancestor, launch_commit):
        raise ValueError(
            "launch commit does not contain the manifest-required ancestor"
        )
    if not git_is_ancestor(repo, launch_commit, analysis_commit):
        raise ValueError(
            "analysis commit is not a descendant of the launch commit"
        )
    merge_base = git_output(
        repo, "merge-base", launch_commit, analysis_commit
    )
    if merge_base != launch_commit:
        raise ValueError("launch/analysis merge-base proof mismatch")
    distance_text = git_output(
        repo, "rev-list", "--count", f"{launch_commit}..{analysis_commit}"
    )
    if not distance_text.isdigit():
        raise ValueError("launch/analysis commit distance is invalid")
    proof = {
        "policy": "launch_commit_is_ancestor_of_analysis_commit",
        "automl_branch": required_branch,
        "launch_commit": launch_commit,
        "analysis_commit": analysis_commit,
        "manifest_required_ancestor_commit": required_ancestor,
        "manifest_required_ancestor_of_launch": True,
        "launch_commit_ancestor_of_analysis": True,
        "merge_base": merge_base,
        "commit_distance": int(distance_text),
        "launch_source_checks": dict(recorded),
        "analysis_source_checks": dict(current_source_checks),
        "measurement_source_hashes_exact": True,
        "branch_exact": True,
        "ledger_sha256": actual_sha256,
    }
    return dict(recorded), proof


def inspect_optional_sdk_sidecar(
    state_path: Path,
    ledger_job_ids: set[str],
) -> dict[str, Any]:
    """Classify the non-authoritative JobMonitor JSON compatibility cache."""

    if not state_path.exists():
        return {
            "path": str(state_path),
            "status": "absent_expected_no_watched_jobs",
            "present": False,
            "authoritative": False,
            "sha256": None,
            "active_job_count": 0,
        }
    if not state_path.is_file() or state_path.is_symlink():
        raise ValueError("SDK monitor sidecar is not a regular file")
    raw = state_path.read_bytes()
    payload = original.strict_json_bytes(raw, "SDK monitor sidecar")
    active = payload.get("active_jobs") if isinstance(payload, dict) else None
    if not isinstance(active, list):
        raise ValueError("SDK monitor sidecar active_jobs is invalid")
    ids = [item.get("id") for item in active if isinstance(item, dict)]
    if (
        len(ids) != len(active)
        or any(not isinstance(job_id, str) for job_id in ids)
        or len(ids) != len(set(ids))
        or not set(ids).issubset(ledger_job_ids)
    ):
        raise ValueError("SDK monitor sidecar job identity drift")
    return {
        "path": str(state_path),
        "status": (
            "present_empty_non_authoritative"
            if not active
            else "present_stale_active_non_authoritative"
        ),
        "present": True,
        "authoritative": False,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "active_job_count": len(active),
        "active_job_ids": ids,
        "recorded_status_by_job_id": {
            item["id"]: item.get("status") for item in active
        },
    }


def validate_terminal_scheduler_evidence(
    scheduler: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    runtime = contract["runtime_contract"]
    gpu_match = re.search(
        r"(?:^|,)(?:gres/)?gpu(?:=[^,:]+)?[:=](\d+)(?:,|$)",
        str(scheduler.get("alloc_tres", "")),
    )
    if (
        scheduler.get("state") != "COMPLETED"
        or scheduler.get("exit_code") != "0:0"
        or scheduler.get("partition") != runtime["partition"]
        or scheduler.get("account") != runtime["account"]
        or scheduler.get("node_count") != 1
        or not isinstance(scheduler.get("expanded_nodes"), list)
        or len(scheduler["expanded_nodes"]) != 1
        or gpu_match is None
        or int(gpu_match.group(1)) != original.EXPECTED_RANKS
    ):
        raise ValueError("scheduler evidence is not exact COMPLETED/0:0")


def inspect_sdk_jobs_read_only(
    contract: dict[str, Any],
    ledger: dict[str, Any],
    state_path: Path,
    accounting: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Inspect immutable SDK rows without monitor construction or mutation."""

    state_path = state_path.resolve()
    database = original.sdk_db_path(state_path)
    if not database.is_file() or database.is_symlink():
        raise FileNotFoundError(
            f"SDK durable state database missing: {database}"
        )
    ledger_by_job_id = {
        item["tao_job_id"]: item for item in ledger["submissions"]
    }
    if len(ledger_by_job_id) != original.EXPECTED_ALLOCATIONS:
        raise ValueError("immutable ledger SDK job set is not exactly nine")
    sidecar = inspect_optional_sdk_sidecar(
        state_path, set(ledger_by_job_id)
    )
    before_snapshot = original.sqlite_snapshot_sha256(database)
    before_file_sha256 = sha256_file(database)
    connection = sqlite3.connect(
        f"file:{database}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only = ON")
        rows = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM jobs ORDER BY job_id"
            ).fetchall()
        ]
    finally:
        connection.close()
    by_job_id = {row.get("job_id"): row for row in rows}
    if (
        len(by_job_id) != len(rows)
        or set(by_job_id) != set(ledger_by_job_id)
    ):
        raise ValueError(
            "SDK durable database job set differs from immutable ledger"
        )

    runtime_contract = contract["runtime_contract"]
    accepted_statuses = set(
        EXPECTED_SDK_STATE_INSPECTION_POLICY["accepted_durable_statuses"]
    )
    rejected_statuses = set(
        EXPECTED_SDK_STATE_INSPECTION_POLICY["rejected_durable_statuses"]
    )
    jobs = []
    status_counts: dict[str, int] = defaultdict(int)
    for submission in ledger["submissions"]:
        allocation_id = submission["allocation_id"]
        tao_id = submission["tao_job_id"]
        slurm_id = str(submission["slurm_job_id"])
        row = by_job_id[tao_id]
        status = row.get("status")
        status_counts[str(status)] += 1
        if status in rejected_statuses or status not in accepted_statuses:
            raise ValueError(
                f"{allocation_id}: rejected durable SDK status {status!r}"
            )
        specs_value = row.get("specs")
        if not isinstance(specs_value, str):
            raise ValueError(f"{allocation_id}: SDK specs are not serialized")
        specs = original.strict_json_bytes(
            specs_value.encode(), f"{allocation_id} SDK specs"
        )
        runtime = specs.get("_slurm_runtime", {})
        artifacts = specs.get("_tao_artifacts", {})
        scheduler = accounting.get(slurm_id)
        if scheduler is None:
            raise ValueError(f"{allocation_id}: scheduler evidence is absent")
        validate_terminal_scheduler_evidence(scheduler, contract)
        sdk_uri = row.get("results_dir")
        if (
            row.get("backend_type") != "slurm"
            or row.get("image") != runtime_contract["sqsh_path"]
            or str(runtime.get("slurm_job_id", "")) != slurm_id
            or int(runtime.get("retry_count", -1)) != 0
            or runtime.get("failed_slurm_job_ids") != []
            or runtime.get("launch_uncertain") is not False
            or runtime.get("job_name") != row.get("backend_job_id")
            or runtime.get("partition") != runtime_contract["partition"]
            or runtime.get("account") != runtime_contract["account"]
            or runtime.get("image") != runtime_contract["sqsh_path"]
            or artifacts.get("kind") != "lustre"
            or artifacts.get("root") != sdk_uri
            or sdk_uri != submission["sdk_results_uri"]
            or scheduler["job_name"] != row.get("backend_job_id")
        ):
            raise ValueError(
                f"{allocation_id}: durable SDK identity/state mismatch"
            )
        result_root = original.local_lustre_path(str(sdk_uri))
        if (
            result_root.name != tao_id
            or result_root.parent.name != "results"
        ):
            raise ValueError(
                f"{allocation_id}: SDK result root is not job-scoped"
            )
        stale = status != "Complete"
        jobs.append(
            {
                "allocation_id": allocation_id,
                "tao_job_id": tao_id,
                "slurm_job_id": slurm_id,
                "sdk_status": status,
                "sdk_status_stale_nonterminal": stale,
                "sdk_status_interpretation": (
                    "stale_nonterminal_accepted_by_exact_terminal_scheduler_evidence"
                    if stale
                    else "durable_terminal_complete"
                ),
                "effective_status": "Complete",
                "effective_status_source": "sacct_COMPLETED_exit_0_0",
                "sdk_results_uri": sdk_uri,
                "sdk_job_scoped_result_root": str(result_root),
                "sdk_backend_job_id": row["backend_job_id"],
                "sdk_runtime_revision": runtime.get("revision"),
                "sdk_retry_count": runtime.get("retry_count"),
                "sdk_failed_slurm_job_ids": runtime.get(
                    "failed_slurm_job_ids", []
                ),
                "sdk_launch_uncertain": runtime.get("launch_uncertain"),
                "scheduler": scheduler,
                "complete": True,
            }
        )
    after_snapshot = original.sqlite_snapshot_sha256(database)
    after_file_sha256 = sha256_file(database)
    if (
        after_snapshot != before_snapshot
        or after_file_sha256 != before_file_sha256
    ):
        raise RuntimeError("read-only SDK inspection mutated durable state")
    return jobs, {
        "state_path": str(state_path),
        "json_sidecar": sidecar,
        "database_path": str(database),
        "database_file_sha256": before_file_sha256,
        "consistent_sqlite_snapshot_sha256": before_snapshot,
        "read_only_snapshot_stable": True,
        "database_open_mode": "sqlite_uri_mode_ro_query_only",
        "job_count": len(rows),
        "durable_status_counts": dict(sorted(status_counts.items())),
        "stale_nonterminal_accepted_count": sum(
            item["sdk_status_stale_nonterminal"] for item in jobs
        ),
        "effective_complete_count": len(jobs),
        "scheduler_gate": "COMPLETED/0:0",
        "monitor_constructed": False,
        "retry_permitted": False,
        "database_mutation_permitted": False,
    }


@dataclass(frozen=True)
class EvidenceSnapshot:
    """Exact local mirror of immutable, remotely hosted result evidence."""

    root: Path
    remote_anchor: str
    by_remote_path: dict[str, Path]
    report: dict[str, Any]

    def local_path(self, remote_path: Path | str) -> Path:
        key = str(remote_path)
        local = self.by_remote_path.get(key)
        if local is None:
            raise ValueError(f"remote evidence path was not preregistered: {key}")
        return local


def validate_remote_path(path: str, anchor: str) -> str:
    """Return an anchor-relative path after strict POSIX normalization."""

    remote = PurePosixPath(path)
    root = PurePosixPath(anchor)
    if (
        not remote.is_absolute()
        or not root.is_absolute()
        or "\n" in path
        or "\x00" in path
        or str(remote) != path
    ):
        raise ValueError(f"non-canonical remote evidence path: {path!r}")
    try:
        relative = remote.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"remote evidence path is outside the pinned anchor: {path}"
        ) from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"invalid remote evidence relative path: {path}")
    return str(relative)


def build_expected_evidence_files(
    contract: dict[str, Any],
    schedule: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    plans: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive every expected file from trusted identities without discovery."""

    policy = EXPECTED_EVIDENCE_ACQUISITION_POLICY
    anchor = policy["remote_results_anchor"]
    blocks = {block["allocation_id"]: block for block in schedule}
    expected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for job in jobs:
        allocation_id = job["allocation_id"]
        block = blocks.get(allocation_id)
        plan = plans.get(allocation_id)
        if block is None or plan is None:
            raise ValueError(f"{allocation_id}: missing schedule or plan")
        result_path = original.result_path_for_job(job, contract, block)

        def add(
            remote_path: Path,
            *,
            kind: str,
            profile_id: str | None = None,
            rank: int | None = None,
        ) -> None:
            key = str(remote_path)
            if key in seen:
                raise ValueError(f"duplicate expected remote path: {key}")
            seen.add(key)
            expected.append(
                {
                    "remote_path": key,
                    "relative_path": validate_remote_path(key, anchor),
                    "kind": kind,
                    "allocation_id": allocation_id,
                    "tao_job_id": job["tao_job_id"],
                    "profile_id": profile_id,
                    "rank": rank,
                }
            )

        add(result_path, kind="allocation_result")
        plan_profiles = plan.get("profiles")
        if (
            not isinstance(plan_profiles, list)
            or len(plan_profiles) != original.EXPECTED_PROFILES
            or [item.get("profile_id") for item in plan_profiles]
            != block["profile_order"]
        ):
            raise ValueError(f"{allocation_id}: regenerated plan order drift")
        for item in plan_profiles:
            profile_id = item["profile_id"]
            run_label = item["run_label"]
            remote_raw_dir = (
                result_path.parent
                / "profiles"
                / run_label
                / job["tao_job_id"]
                / "latency"
            )
            for rank in range(original.EXPECTED_RANKS):
                add(
                    remote_raw_dir / f"rank_{rank}.json",
                    kind="rank_result",
                    profile_id=profile_id,
                    rank=rank,
                )
    kind_counts = {
        kind: sum(item["kind"] == kind for item in expected)
        for kind in ("allocation_result", "rank_result")
    }
    if (
        len(expected) != policy["expected_file_count"]
        or kind_counts["allocation_result"]
        != policy["expected_allocation_result_count"]
        or kind_counts["rank_result"]
        != policy["expected_rank_result_count"]
        or len(seen) != len(expected)
    ):
        raise RuntimeError("derived remote evidence dimensions drifted")
    return sorted(expected, key=lambda item: item["remote_path"])


REMOTE_INVENTORY_SCRIPT = r"""
import hashlib
import json
import os
import stat
import sys

request = json.load(sys.stdin)
paths = request.get("paths")
if request.get("schema_version") != 1 or not isinstance(paths, list):
    raise SystemExit("invalid inventory request")
if len(paths) != len(set(paths)):
    raise SystemExit("duplicate inventory request path")
files = []
for path in paths:
    if not isinstance(path, str) or not path.startswith("/"):
        raise SystemExit("invalid inventory path")
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise SystemExit("inventory path is not a regular non-symlink file")
    before = (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after_info = os.lstat(path)
    after = (
        after_info.st_dev,
        after_info.st_ino,
        after_info.st_mode,
        after_info.st_size,
        after_info.st_mtime_ns,
        after_info.st_ctime_ns,
    )
    if before != after:
        raise SystemExit("inventory path changed while hashing")
    files.append({
        "path": path,
        "size_bytes": info.st_size,
        "sha256": digest.hexdigest(),
        "file_type": "regular_non_symlink",
        "stat_fingerprint": {
            "device": info.st_dev,
            "inode": info.st_ino,
            "mode": info.st_mode,
            "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns,
        },
    })
json.dump({"schema_version": 1, "files": files}, sys.stdout, sort_keys=True)
"""


def ssh_base_command() -> list[str]:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
    ]
    key = os.environ.get("SSH_KEY_PATH")
    if key:
        command.extend(["-i", key])
    return command


def fetch_remote_inventory(
    expected: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Hash the exact expected file list remotely over read-only SSH."""

    paths = [item["remote_path"] for item in expected]
    request = {"schema_version": 1, "paths": paths}
    expected_paths = set(paths)
    request_bytes = json.dumps(
        request, sort_keys=True, separators=(",", ":")
    ).encode()
    remote_command = "python3 -c " + shlex.quote(REMOTE_INVENTORY_SCRIPT)
    command = [
        *ssh_base_command(),
        original.ssh_target(),
        remote_command,
    ]
    completed = subprocess.run(
        command,
        input=request_bytes,
        check=True,
        capture_output=True,
        timeout=3600,
    )
    response = original.strict_json_bytes(
        completed.stdout, "remote evidence inventory"
    )
    files = response.get("files") if isinstance(response, dict) else None
    if (
        not isinstance(response, dict)
        or response.get("schema_version") != 1
        or not isinstance(files, list)
        or len(files) != len(paths)
    ):
        raise ValueError("remote evidence inventory response mismatch")
    by_path: dict[str, dict[str, Any]] = {}
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("invalid remote evidence inventory entry")
        path = item.get("path")
        digest = item.get("sha256")
        size = item.get("size_bytes")
        fingerprint = item.get("stat_fingerprint")
        if (
            path in by_path
            or path not in expected_paths
            or item.get("file_type") != "regular_non_symlink"
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(fingerprint, dict)
            or set(fingerprint)
            != {"device", "inode", "mode", "mtime_ns", "ctime_ns"}
            or any(
                not isinstance(value, int) for value in fingerprint.values()
            )
        ):
            raise ValueError("invalid or duplicate remote inventory entry")
        by_path[path] = item
    if set(by_path) != expected_paths:
        raise ValueError("remote evidence inventory missing or extra paths")
    return by_path, {
        "transport": "ssh_batch_mode",
        "remote_target": original.ssh_target(),
        "remote_inventory_script_sha256": hashlib.sha256(
            REMOTE_INVENTORY_SCRIPT.encode()
        ).hexdigest(),
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "response_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "remote_stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "requested_file_count": len(paths),
    }


def scan_local_snapshot(
    snapshot_root: Path,
    expected_relative_paths: set[str],
) -> set[str]:
    """Reject symlinks, unexpected files, and unexpected directories."""

    if snapshot_root.is_symlink():
        raise ValueError("evidence snapshot root must not be a symlink")
    if not snapshot_root.exists():
        return set()
    if not snapshot_root.is_dir():
        raise ValueError("evidence snapshot root must be a directory")
    expected_directories = {""}
    for relative in expected_relative_paths:
        parent = PurePosixPath(relative).parent
        while str(parent) != ".":
            expected_directories.add(str(parent))
            parent = parent.parent
    found: set[str] = set()
    for current, directories, files in os.walk(
        snapshot_root, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        current_relative = current_path.relative_to(snapshot_root)
        for directory in directories:
            child = current_path / directory
            relative = (current_relative / directory).as_posix()
            if child.is_symlink() or relative not in expected_directories:
                raise ValueError(
                    f"unexpected snapshot directory: {relative}"
                )
        for filename in files:
            child = current_path / filename
            relative = (current_relative / filename).as_posix()
            mode = child.lstat().st_mode
            if (
                stat.S_ISLNK(mode)
                or not stat.S_ISREG(mode)
                or relative not in expected_relative_paths
            ):
                raise ValueError(f"unexpected snapshot file: {relative}")
            found.add(relative)
    return found


def rsync_missing_files(
    relative_paths: list[str],
    remote_anchor: str,
    destination: Path,
) -> dict[str, Any]:
    """Fetch only exact missing files into an isolated local staging tree."""

    if not relative_paths:
        return {
            "executed": False,
            "requested_file_count": 0,
            "stdout_sha256": None,
            "stderr_sha256": None,
        }
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="dino_sensitivity_files_",
        suffix=".txt",
        delete=False,
    ) as stream:
        files_from = Path(stream.name)
        for relative in relative_paths:
            stream.write(relative + "\n")
    try:
        rsh_parts = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
        ]
        key = os.environ.get("SSH_KEY_PATH")
        if key:
            rsh_parts.extend(["-i", key])
        command = [
            "rsync",
            "--archive",
            "--recursive",
            "--protect-args",
            "--ignore-existing",
            f"--files-from={files_from}",
            "--rsh",
            shlex.join(rsh_parts),
            f"{original.ssh_target()}:{remote_anchor}/",
            f"{destination}/",
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=3600,
        )
        return {
            "executed": True,
            "requested_file_count": len(relative_paths),
            "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        }
    finally:
        files_from.unlink(missing_ok=True)


def acquire_evidence_snapshot(
    expected: list[dict[str, Any]],
    snapshot_root: Path,
) -> EvidenceSnapshot:
    """Acquire and verify an exact, resume-safe local evidence mirror."""

    policy = EXPECTED_EVIDENCE_ACQUISITION_POLICY
    anchor = policy["remote_results_anchor"]
    snapshot_root = snapshot_root.resolve()
    required_parent = (HERE / policy["local_snapshot_parent"]).resolve()
    try:
        snapshot_root.relative_to(required_parent)
    except ValueError as error:
        raise ValueError(
            "evidence snapshot must be scoped beneath the pinned runtime "
            f"directory: {required_parent}"
        ) from error
    if snapshot_root == required_parent:
        raise ValueError(
            "evidence snapshot must be a dedicated child directory"
        )
    relative_to_expected = {
        item["relative_path"]: item for item in expected
    }
    if len(relative_to_expected) != len(expected):
        raise ValueError("duplicate expected snapshot relative path")
    remote_inventory, remote_audit = fetch_remote_inventory(expected)

    snapshot_root.mkdir(parents=True, exist_ok=True)
    existing = scan_local_snapshot(
        snapshot_root, set(relative_to_expected)
    )
    for relative in sorted(existing):
        local_path = snapshot_root / Path(*PurePosixPath(relative).parts)
        remote_path = relative_to_expected[relative]["remote_path"]
        if sha256_file(local_path) != remote_inventory[remote_path]["sha256"]:
            raise RuntimeError(
                "refusing to overwrite non-identical existing snapshot file: "
                f"{relative}"
            )
    missing = sorted(set(relative_to_expected) - existing)

    transfer_audit = {
        "executed": False,
        "requested_file_count": 0,
        "stdout_sha256": None,
        "stderr_sha256": None,
    }
    if missing:
        snapshot_parent = snapshot_root.parent
        snapshot_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{snapshot_root.name}.incoming-",
            dir=snapshot_parent,
        ) as staging_name:
            staging = Path(staging_name)
            transfer_audit = rsync_missing_files(missing, anchor, staging)
            staged_found = scan_local_snapshot(staging, set(missing))
            if staged_found != set(missing):
                raise RuntimeError(
                    "remote transfer did not produce every missing file"
                )
            for relative in missing:
                staged_path = staging / Path(
                    *PurePosixPath(relative).parts
                )
                remote_path = relative_to_expected[relative]["remote_path"]
                expected_digest = remote_inventory[remote_path]["sha256"]
                if (
                    staged_path.stat().st_size
                    != remote_inventory[remote_path]["size_bytes"]
                    or sha256_file(staged_path) != expected_digest
                ):
                    raise RuntimeError(
                        f"transferred evidence digest mismatch: {relative}"
                    )
            for relative in missing:
                staged_path = staging / Path(
                    *PurePosixPath(relative).parts
                )
                local_path = snapshot_root / Path(
                    *PurePosixPath(relative).parts
                )
                local_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(staged_path, local_path)
                except FileExistsError:
                    remote_path = relative_to_expected[relative][
                        "remote_path"
                    ]
                    if (
                        not local_path.is_file()
                        or local_path.is_symlink()
                        or sha256_file(local_path)
                        != remote_inventory[remote_path]["sha256"]
                    ):
                        raise RuntimeError(
                            "snapshot path appeared with non-identical bytes: "
                            f"{relative}"
                        )

    complete = scan_local_snapshot(
        snapshot_root, set(relative_to_expected)
    )
    if complete != set(relative_to_expected):
        raise RuntimeError("local evidence snapshot is incomplete")
    inventory = []
    by_remote_path: dict[str, Path] = {}
    for item in expected:
        remote_path = item["remote_path"]
        local_path = snapshot_root / Path(
            *PurePosixPath(item["relative_path"]).parts
        )
        local_digest = sha256_file(local_path)
        remote = remote_inventory[remote_path]
        if (
            local_path.stat().st_size != remote["size_bytes"]
            or local_digest != remote["sha256"]
        ):
            raise RuntimeError(
                f"remote/local evidence mismatch: {item['relative_path']}"
            )
        local_path.chmod(local_path.stat().st_mode & ~0o222)
        local_mode = stat.S_IMODE(local_path.stat().st_mode)
        by_remote_path[remote_path] = local_path
        inventory.append(
            {
                **item,
                "local_path": str(local_path),
                "size_bytes": remote["size_bytes"],
                "remote_sha256": remote["sha256"],
                "remote_stat_fingerprint": remote.get("stat_fingerprint"),
                "local_sha256": local_digest,
                "digest_equal": True,
                "local_mode_octal": oct(local_mode),
                "local_write_bits_cleared": local_mode & 0o222 == 0,
            }
        )
    report = {
        "mode": policy["mode"],
        "remote_write_permitted": False,
        "snapshot_root": str(snapshot_root),
        "remote_anchor": anchor,
        "expected_file_count": len(expected),
        "existing_identical_file_count": len(existing),
        "transferred_file_count": len(missing),
        "remote_inventory": remote_audit,
        "transfer": transfer_audit,
        "inventory": inventory,
        "inventory_sha256": sha256_value(inventory),
        "complete": True,
        "missing_files": [],
        "extra_files": [],
        "duplicate_files": [],
        "overwrite_performed": False,
    }
    return EvidenceSnapshot(
        root=snapshot_root,
        remote_anchor=anchor,
        by_remote_path=by_remote_path,
        report=report,
    )


def validate_runtime_identity(
    runtime: Any,
    expected: dict[str, Any],
    *,
    label: str,
) -> tuple[Any, Any, Any]:
    """Validate a runtime under the explicit base-release policy."""

    if not isinstance(runtime, dict):
        raise ValueError(f"{label} runtime missing")
    if expected.get("torch_version_match") != "major_minor_patch":
        raise ValueError(f"{label} torch version policy mismatch")
    actual_torch = runtime.get("torch")
    if (
        original.major_minor_patch(actual_torch, f"{label} torch version")
        != expected["required_torch"]
        or runtime.get("cuda") != expected["required_cuda"]
        or runtime.get("cudnn") != expected["required_cudnn"]
    ):
        raise ValueError(f"{label} runtime mismatch")
    return actual_torch, runtime.get("cuda"), runtime.get("cudnn")


def validate_allocation_hardware(
    hardware: Any,
    expected: dict[str, Any],
) -> tuple[Any, Any, Any]:
    """Validate all eight devices and the allocation-level runtime."""

    if not isinstance(hardware, dict):
        raise ValueError("allocation hardware missing")
    devices = hardware.get("devices")
    if (
        not isinstance(devices, list)
        or len(devices) != original.EXPECTED_RANKS
        or [device.get("index") for device in devices]
        != list(range(original.EXPECTED_RANKS))
        or any(
            device.get("name") != expected["required_gpu_name"]
            or device.get("compute_capability")
            != expected["required_compute_capability"]
            or device.get("total_memory_bytes")
            != expected["required_total_memory_bytes"]
            for device in devices
        )
    ):
        raise ValueError("allocation hardware mismatch")
    return validate_runtime_identity(
        hardware.get("runtime"), expected, label="allocation"
    )


def validate_rank_runtime(
    record: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[Any, Any, Any]:
    """Apply the same explicit runtime policy at rank level."""

    return validate_runtime_identity(
        record.get("runtime"),
        contract["runtime_contract"],
        label="rank",
    )


def aggregate_job_results(
    contract: dict[str, Any],
    profiles: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    artifact_entries: dict[tuple[int, str], dict[str, Any]],
    accuracy_entries: dict[tuple[int, str], dict[str, Any]],
    jobs: list[dict[str, Any]],
    expected_plans: dict[str, dict[str, Any]],
    manifest_sha256: str,
    checkpoint_artifact_sha256: str,
    evidence_snapshot: EvidenceSnapshot | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Aggregate immutable v2 outputs with corrected allocation validation."""

    protocol = original.latency_protocol(contract)
    blocks = {block["allocation_id"]: block for block in schedule}
    profile_by_id = {profile["profile_id"]: profile for profile in profiles}
    input_identities: dict[int, str] = {}
    runtime_signature: tuple[Any, ...] | None = None
    allocation_runtime_signature: tuple[Any, ...] | None = None
    gpu_uuid_by_allocation_rank: dict[str, dict[int, str]] = defaultdict(dict)
    measurements: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []

    for job in jobs:
        allocation_id = job["allocation_id"]
        block = blocks[allocation_id]
        plan = expected_plans[allocation_id]
        remote_result_path = original.result_path_for_job(
            job, contract, block
        )
        result_path = (
            evidence_snapshot.local_path(remote_result_path)
            if evidence_snapshot is not None
            else remote_result_path
        )
        result, result_sha256 = original.read_hashed_json(
            result_path, f"{allocation_id} result"
        )
        expected_result_values = {
            "schema_version": 1,
            "status": "success",
            "manifest_id": contract["manifest_id"],
            "manifest_sha256": manifest_sha256,
            "checkpoint_artifact_sha256": checkpoint_artifact_sha256,
            "schedule_sha256": contract["design"]["schedule_sha256"],
            "allocation_id": allocation_id,
            "seed": block["seed"],
            "repeat_index": block["repeat_index"],
            "williams_row_index": block["williams_row_index"],
            "block_plan_sha256": plan["block_plan_sha256"],
            "tao_job_id": job["tao_job_id"],
            "sdk_job_scoped_result_root": job[
                "sdk_job_scoped_result_root"
            ],
            "feeds_final_selection": False,
            "manual_promotion_permitted": False,
        }
        for key, expected in expected_result_values.items():
            if result.get(key) != expected:
                raise ValueError(f"{allocation_id}: result {key} drift")
        hostname = result.get("hostname")
        if (
            not isinstance(hostname, str)
            or hostname != job["scheduler"]["expanded_nodes"][0]
        ):
            raise ValueError(f"{allocation_id}: scheduler hostname mismatch")
        result_output = result.get("output_contract", {})
        if (
            result_output.get("root_env") != "TAO_RESULTS_ROOT"
            or result_output.get("job_scope_env") != "TAO_JOB_ID"
            or result_output.get("root")
            != job["sdk_job_scoped_result_root"]
        ):
            raise ValueError(f"{allocation_id}: result output contract mismatch")
        allocation_signature = validate_allocation_hardware(
            result.get("hardware"), contract["runtime_contract"]
        )
        if allocation_runtime_signature is None:
            allocation_runtime_signature = allocation_signature
        elif allocation_runtime_signature != allocation_signature:
            raise ValueError("allocation runtime signature drift")

        runs = result.get("profile_runs")
        if (
            not isinstance(runs, list)
            or len(runs) != original.EXPECTED_PROFILES
            or [run.get("profile_id") for run in runs]
            != block["profile_order"]
            or [run.get("position") for run in runs]
            != list(range(original.EXPECTED_PROFILES))
            or any(
                run.get("status") != "success"
                or run.get("exit_code") != 0
                or run.get("seed") != block["seed"]
                for run in runs
            )
        ):
            raise ValueError(f"{allocation_id}: incomplete/reordered block")
        expected_config_digests = {
            item["profile_id"]: item["config_sha256"]
            for item in plan["profiles"]
        }
        if result.get("verified_config_sha256") != expected_config_digests:
            raise ValueError(f"{allocation_id}: verified config digest drift")

        remote_result_root = Path(job["sdk_job_scoped_result_root"])
        block_measurements = []
        raw_file_count = 0
        raw_digest_by_profile: dict[str, dict[str, str]] = {}
        for run in runs:
            profile_id = run["profile_id"]
            expected_profile = plan["profiles"][run["position"]]
            artifact = artifact_entries[(block["seed"], profile_id)]
            if (
                expected_profile["profile_id"] != profile_id
                or run.get("run_label") != expected_profile["run_label"]
                or run.get("config_sha256")
                != expected_profile["config_sha256"]
                or run.get("checkpoint_path")
                != artifact["checkpoint_path"]
                or run.get("checkpoint_sha256")
                != artifact["checkpoint_sha256"]
                or run.get("checkpoint_source_profile_id")
                != artifact["checkpoint_source_profile_id"]
                or run.get("resolved_model_spec_sha256")
                != profile_by_id[profile_id]["resolved_model_spec_sha256"]
            ):
                raise ValueError(
                    f"{allocation_id}/{profile_id}: run provenance mismatch"
                )
            remote_raw_dir = (
                remote_result_path.parent
                / "profiles"
                / run["run_label"]
                / job["tao_job_id"]
                / "latency"
            )
            if run.get("raw_samples_dir") != str(remote_raw_dir):
                raise ValueError(
                    f"{allocation_id}/{profile_id}: raw path drift"
                )
            samples = {
                round_index: {}
                for round_index in range(protocol.repeated_rounds)
            }
            rank_digests = {}
            for rank in range(original.EXPECTED_RANKS):
                remote_rank_path = remote_raw_dir / f"rank_{rank}.json"
                rank_path = (
                    evidence_snapshot.local_path(remote_rank_path)
                    if evidence_snapshot is not None
                    else remote_rank_path
                )
                record, digest = original.read_hashed_json(
                    rank_path,
                    f"{allocation_id}/{profile_id}/rank_{rank}",
                )
                rank_digests[str(rank)] = digest
                raw_file_count += 1
                validate_rank_runtime(record, contract)
                identity, signature, gpu_uuid = original.validate_rank_record(
                    record,
                    rank=rank,
                    contract=contract,
                    checkpoint_path=artifact["checkpoint_path"],
                    config_path=expected_profile["config_path"],
                    allocation_hostname=hostname,
                )
                prior_input = input_identities.setdefault(rank, identity)
                if identity != prior_input:
                    raise ValueError("benchmark input identity drift")
                if runtime_signature is None:
                    runtime_signature = signature
                elif runtime_signature != signature:
                    raise ValueError("runtime signature drift")
                prior_uuid = gpu_uuid_by_allocation_rank[
                    allocation_id
                ].setdefault(rank, gpu_uuid)
                if prior_uuid != gpu_uuid:
                    raise ValueError("GPU rank mapping changed within allocation")
                rank_samples = record.get("samples_ms")
                if (
                    not isinstance(rank_samples, list)
                    or len(rank_samples) != protocol.repeated_rounds
                ):
                    raise ValueError("rank sample round count mismatch")
                for round_index, values in enumerate(rank_samples):
                    samples[round_index][str(rank)] = values
            stats = original.aggregate_synchronized_latency(samples, protocol)
            if not stats.is_valid:
                raise ValueError(
                    f"{allocation_id}/{profile_id}: {stats.validity_reason}"
                )
            raw_digest_by_profile[profile_id] = rank_digests
            accuracy = accuracy_entries[(block["seed"], profile_id)]
            block_measurements.append(
                {
                    "allocation_id": allocation_id,
                    "tao_job_id": job["tao_job_id"],
                    "slurm_job_id": job["slurm_job_id"],
                    "node_list": job["scheduler"]["node_list"],
                    "hostname": hostname,
                    "seed": block["seed"],
                    "repeat_index": block["repeat_index"],
                    "williams_row_index": block["williams_row_index"],
                    "profile_id": profile_id,
                    "axis": profile_by_id[profile_id]["axis"],
                    "level": profile_by_id[profile_id]["level"],
                    "position": run["position"],
                    "map50": float(accuracy["mAP50"]),
                    "checkpoint_sha256": artifact["checkpoint_sha256"],
                    "resolved_model_spec_sha256": artifact[
                        "resolved_model_spec_sha256"
                    ],
                    "config_sha256": expected_profile["config_sha256"],
                    "raw_rank_file_sha256_by_rank": rank_digests,
                    **original.stats_record(stats),
                }
            )
        if raw_file_count != (
            original.EXPECTED_PROFILES * original.EXPECTED_RANKS
        ):
            raise RuntimeError(f"{allocation_id}: raw rank file count drift")
        if (
            len(set(gpu_uuid_by_allocation_rank[allocation_id].values()))
            != original.EXPECTED_RANKS
        ):
            raise ValueError(
                f"{allocation_id}: GPUs are not eight distinct UUIDs"
            )
        measurements.extend(block_measurements)
        evidence.append(
            {
                "allocation_id": allocation_id,
                "tao_job_id": job["tao_job_id"],
                "slurm_job_id": job["slurm_job_id"],
                "node_list": job["scheduler"]["node_list"],
                "hostname": hostname,
                "sdk_job_scoped_result_root": str(remote_result_root),
                "remote_result_path": str(remote_result_path),
                "local_result_path": str(result_path),
                "result_sha256": result_sha256,
                "block_plan_sha256": plan["block_plan_sha256"],
                "allocation_runtime": {
                    "torch": allocation_signature[0],
                    "torch_major_minor_patch": original.major_minor_patch(
                        allocation_signature[0],
                        "allocation torch version",
                    ),
                    "cuda": allocation_signature[1],
                    "cudnn": allocation_signature[2],
                },
                "allocation_runtime_comparison": "major_minor_patch",
                "raw_rank_file_count": raw_file_count,
                "raw_rank_file_sha256_by_profile": raw_digest_by_profile,
                "gpu_uuid_sha256_by_rank": {
                    str(rank): hashlib.sha256(uuid.encode()).hexdigest()
                    for rank, uuid in sorted(
                        gpu_uuid_by_allocation_rank[allocation_id].items()
                    )
                },
            }
        )
    if len(measurements) != (
        original.EXPECTED_ALLOCATIONS * original.EXPECTED_PROFILES
    ):
        raise RuntimeError("expected exactly 126 valid measurements")
    consistency = {
        "hardware_contract": "pass",
        "runtime_contract": "pass",
        "allocation_torch_version_comparison": "major_minor_patch",
        "allocation_runtime_signature": list(
            allocation_runtime_signature or ()
        ),
        "raw_allocation_runtime_string_preserved": True,
        "protocol_contract": "pass",
        "benchmark_input_identity": "pass",
        "rank_files_per_profile": original.EXPECTED_RANKS,
        "runtime_signature": list(runtime_signature or ()),
        "benchmark_input_identity_sha256_by_rank": {
            str(rank): digest
            for rank, digest in sorted(input_identities.items())
        },
        "distinct_gpu_count_by_allocation": {
            allocation_id: len(set(by_rank.values()))
            for allocation_id, by_rank in sorted(
                gpu_uuid_by_allocation_rank.items()
            )
        },
        "node_frequency": dict(
            sorted(
                {
                    node: sum(item["hostname"] == node for item in evidence)
                    for node in {item["hostname"] for item in evidence}
                }.items()
            )
        ),
        "distinct_tao_job_count": len(
            {item["tao_job_id"] for item in evidence}
        ),
        "distinct_slurm_allocation_count": len(
            {item["slurm_job_id"] for item in evidence}
        ),
        "position_balance": {
            profile["profile_id"]: sorted(
                row["position"]
                for row in measurements
                if row["profile_id"] == profile["profile_id"]
            )
            for profile in profiles
        },
    }
    return measurements, evidence, consistency


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    erratum, _raw_manifest, erratum_identity = validate_analysis_erratum(
        args.analysis_erratum,
        args.analysis_erratum_sha256,
        manifest_path,
        args.submission_ledger,
        args.submission_ledger_sha256,
    )
    contract, one, one_path = load_contract(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    profiles = build_profiles(one)
    schedule = build_schedule(contract, profiles)
    if (
        len(profiles) != original.EXPECTED_PROFILES
        or len(schedule) != original.EXPECTED_ALLOCATIONS
    ):
        raise RuntimeError("frozen sensitivity design dimensions drifted")
    schedule_sha256 = sha256_value(schedule)
    checkpoint_artifact, artifact_entries = load_checkpoint_artifact(
        args.checkpoint_artifact.resolve(),
        args.checkpoint_artifact_sha256,
        contract,
        one,
        profiles,
    )
    accuracy_artifact, accuracy_entries = load_accuracy_artifact(
        args.accuracy_artifact.resolve(),
        args.accuracy_artifact_sha256,
        args.checkpoint_artifact_sha256,
        contract,
        one,
        profiles,
        artifact_entries,
    )
    plans, summaries, source_checks = original.regenerate_execution(
        manifest_path,
        contract,
        one,
        profiles,
        schedule,
        artifact_entries,
        manifest_sha256,
        args.checkpoint_artifact_sha256,
    )
    launch_source_checks, commit_provenance = (
        validate_launch_analysis_provenance(
            args.submission_ledger.resolve(),
            args.submission_ledger_sha256,
            contract,
            source_checks,
        )
    )
    ledger, ledger_sha256 = original.load_submission_ledger(
        args.submission_ledger.resolve(),
        args.submission_ledger_sha256,
        contract,
        manifest_sha256,
        args.checkpoint_artifact_sha256,
        schedule_sha256,
        schedule,
        summaries,
        launch_source_checks,
    )
    secrets_path = (
        args.secrets_env.resolve()
        if args.secrets_env
        else Path(contract["runtime_contract"]["secrets_env_path"])
    )
    loaded_keys = original.load_env_file(secrets_path)
    accounting = original.slurm_accounting(
        ledger["submissions"], contract
    )
    jobs, sdk_provenance = inspect_sdk_jobs_read_only(
        contract,
        ledger,
        args.sdk_state.resolve(),
        accounting,
    )
    expected_evidence = build_expected_evidence_files(
        contract,
        schedule,
        jobs,
        plans,
    )
    evidence_snapshot = acquire_evidence_snapshot(
        expected_evidence,
        args.evidence_snapshot,
    )
    measurements, allocation_evidence, consistency = aggregate_job_results(
        contract,
        profiles,
        schedule,
        artifact_entries,
        accuracy_entries,
        jobs,
        plans,
        manifest_sha256,
        args.checkpoint_artifact_sha256,
        evidence_snapshot,
    )
    analysis = original.qualification_analysis(
        contract,
        one,
        profiles,
        schedule,
        accuracy_entries,
        measurements,
    )
    if (
        erratum_identity["measurement_manifest_sha256"] != manifest_sha256
        or erratum_identity["submission_ledger_sha256"] != ledger_sha256
    ):
        raise RuntimeError("validated erratum identity changed during analysis")
    erratum_identity = {
        **erratum_identity,
        "correction": erratum["correction"],
        "analysis_commit_correction": erratum[
            "analysis_commit_correction"
        ],
    }
    report = {
        "schema_version": 1,
        "status": "complete",
        "blockers": [],
        "checked_at_utc": original.utc_timestamp(),
        "analysis_erratum": erratum_identity,
        "manifest_id": contract["manifest_id"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "one_factor_manifest_path": str(one_path),
        "checkpoint_artifact_id": checkpoint_artifact["artifact_id"],
        "checkpoint_artifact_sha256": args.checkpoint_artifact_sha256,
        "accuracy_artifact_id": accuracy_artifact["artifact_id"],
        "accuracy_artifact_sha256": args.accuracy_artifact_sha256,
        "submission_ledger_path": str(args.submission_ledger.resolve()),
        "submission_ledger_sha256": ledger_sha256,
        "schedule_sha256": schedule_sha256,
        "measurement_generation_source_checks": launch_source_checks,
        "current_analysis_measurement_source_checks": source_checks,
        "launch_analysis_commit_provenance": commit_provenance,
        "analysis_source_checks": {
            "original_aggregator": erratum_identity[
                "original_aggregator_sha256"
            ],
            "corrected_aggregator": erratum_identity[
                "corrected_aggregator_sha256"
            ],
            "analysis_erratum": erratum_identity["erratum_sha256"],
        },
        "secrets_env_path": str(secrets_path),
        "loaded_secret_keys": loaded_keys,
        "secret_values_recorded": False,
        "sdk_state_provenance": sdk_provenance,
        "remote_evidence_snapshot": evidence_snapshot.report,
        "jobs": jobs,
        "allocation_evidence": allocation_evidence,
        "artifact_consistency": consistency,
        "allocation_measurements": measurements,
        **analysis,
        "winner_selected": False,
        "feeds_final_selection": False,
        "manual_promotion_permitted": False,
        "result_policy": (
            "Sensitivity qualification only. The analysis erratum changes "
            "only allocation-level PyTorch base-release validation; it does "
            "not change samples, statistics, qualification policy, objective "
            "values, or any winner."
        ),
    }
    report["report_sha256"] = sha256_value(report)
    original.atomic_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "analysis_erratum_id": erratum["erratum_id"],
                "analysis_erratum_sha256": erratum_identity[
                    "erratum_sha256"
                ],
                "allocation_measurement_count": len(measurements),
                "remote_evidence_file_count": evidence_snapshot.report[
                    "expected_file_count"
                ],
                "remote_evidence_inventory_sha256": (
                    evidence_snapshot.report["inventory_sha256"]
                ),
                "latency_effect_qualified_profiles": report[
                    "latency_effect_qualified_profiles"
                ],
                "latency_reduction_qualified_profiles": report[
                    "latency_reduction_qualified_profiles"
                ],
                "latency_mode_98pct_suitable_profiles": report[
                    "latency_mode_98pct_suitable_profiles"
                ],
                "objective_values_altered": False,
                "winner_selected": False,
                "feeds_final_selection": False,
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
