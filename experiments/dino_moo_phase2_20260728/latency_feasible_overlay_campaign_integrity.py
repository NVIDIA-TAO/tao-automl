#!/usr/bin/env python3

"""Read-only integrity verification for the submitted rec16-overlay campaign.

This verifier is intentionally separate from the frozen manifest, overlay,
launch, block-runner, and aggregation paths.  It performs no scheduler, SDK,
network, selector, measurement, or selection operation.  Its only write is a
new, self-hashed audit JSON after every supplied artifact and every
reconstructed launch byte has passed validation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import yaml

import latency_feasible_checkpoint_overlay as checkpoint_overlay
import latency_feasible_matched_launcher as launcher
import latency_feasible_matched_manifest_generator as manifest_generator


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "latency_feasible_matched_manifest.v1.json"
DEFAULT_OVERLAY = HERE / "latency_feasible_rec16_checkpoint_overlay.v2.json"
DEFAULT_RECOVERY_EVIDENCE = HERE / "rec16_checkpoint_recovery_evidence.v1.json"
DEFAULT_RUNTIME = HERE / "runtime" / "latency_feasible_matched"
DEFAULT_LAUNCH_CONTRACT = DEFAULT_RUNTIME / "launch_contract.v1.json"
DEFAULT_LEDGER = DEFAULT_RUNTIME / "block_submissions.json"
DEFAULT_ANALYSIS = DEFAULT_RUNTIME / "latency_feasible_matched_analysis.json"
DEFAULT_OUTPUT = DEFAULT_RUNTIME / "overlay_campaign_integrity_audit.v1.json"

AUDIT_ID = "dino_latency_feasible_overlay_campaign_integrity_20260728_v1"
EXPECTED_CANDIDATE_IDS = (
    "seed_161803_rec_14",
    "seed_271828_rec_16",
    "seed_271828_rec_18",
    "seed_314159_rec_12",
)
EXPECTED_SCHEDULE = (
    (
        "latency_feasible_allocation_00",
        0,
        0,
        (
            "seed_161803_rec_14",
            "seed_271828_rec_16",
            "seed_314159_rec_12",
            "seed_271828_rec_18",
        ),
    ),
    (
        "latency_feasible_allocation_01",
        1,
        0,
        (
            "seed_161803_rec_14",
            "seed_271828_rec_16",
            "seed_314159_rec_12",
            "seed_271828_rec_18",
        ),
    ),
    (
        "latency_feasible_allocation_02",
        2,
        1,
        (
            "seed_271828_rec_16",
            "seed_271828_rec_18",
            "seed_161803_rec_14",
            "seed_314159_rec_12",
        ),
    ),
    (
        "latency_feasible_allocation_03",
        3,
        2,
        (
            "seed_271828_rec_18",
            "seed_314159_rec_12",
            "seed_271828_rec_16",
            "seed_161803_rec_14",
        ),
    ),
    (
        "latency_feasible_allocation_04",
        4,
        2,
        (
            "seed_271828_rec_18",
            "seed_314159_rec_12",
            "seed_271828_rec_16",
            "seed_161803_rec_14",
        ),
    ),
    (
        "latency_feasible_allocation_05",
        5,
        3,
        (
            "seed_314159_rec_12",
            "seed_161803_rec_14",
            "seed_271828_rec_18",
            "seed_271828_rec_16",
        ),
    ),
)
EXPECTED_SUBMISSION_IDENTITIES = {
    "latency_feasible_allocation_00": (
        "32862ad8-e015-46a1-892d-e399ae71a4a5",
        "31004302",
    ),
    "latency_feasible_allocation_01": (
        "ee1f5f87-ef4a-4e4a-b13c-fdbf3a90d792",
        "31004303",
    ),
    "latency_feasible_allocation_02": (
        "eecf47a7-aed4-4d5b-8921-7e50f644a5a6",
        "31004306",
    ),
    "latency_feasible_allocation_03": (
        "a789904e-a852-406b-b392-5b2c733dda3c",
        "31004308",
    ),
    "latency_feasible_allocation_04": (
        "2e7d214e-b5a3-44d5-b402-88a25db016d5",
        "31004316",
    ),
    "latency_feasible_allocation_05": (
        "2d88ee5f-1fd0-4644-b446-fa75c205e8a2",
        "31004322",
    ),
}
SELECTION_ISOLATION_FLAGS = (
    "selector_invoked_on_matched_measurements",
    "selection_time_objectives_replaced",
    "measurements_feed_selection",
    "measurements_feed_reselection",
    "algorithm_selected_candidate_overridden",
)
OVERLAY_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "overlay_id",
        "status",
        "scope",
        "parent_manifest",
        "recovery_evidence",
        "substitution",
        "invariants",
        "selection_isolation",
        "execution_tools",
        "overlay_sha256",
    }
)
RECOVERY_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "evidence_id",
        "status",
        "frozen_at_utc",
        "candidate_id",
        "historical_checkpoint",
        "source_identity",
        "recovery_attempts",
        "selection_policy",
        "selected_recovery",
        "supplementary_exact_node_replay",
        "selection_isolation",
        "evidence_sha256",
    }
)
RECOVERY_SOURCE_KEYS = frozenset(
    {
        "expanded_manifest",
        "seed_archive",
        "candidate_record_sha256",
        "candidate_specs",
        "train_spec_sha256",
        "model_spec_sha256",
        "command_sha256",
        "training_seed",
        "completed_recovery_launcher",
        "supplementary_recovery_launcher",
    }
)
RECOVERY_ATTEMPT_KEYS = frozenset(
    {
        "submission_index",
        "tao_job_id",
        "slurm_job_id",
        "submit_time_utc",
        "start_time_utc",
        "end_time_utc",
        "state",
        "exit_code",
        "node",
        "exact_config",
        "checkpoint_origin",
        "train_spec_sha256",
        "model_spec_sha256",
        "command_sha256",
        "checkpoint",
        "historical_checkpoint_sha256_match",
        "entrypoint",
        "remote_specs",
        "sbatch",
    }
)
RECOVERY_SELECTED_KEYS = frozenset(
    {
        "policy_key",
        "submission_index",
        "tao_job_id",
        "slurm_job_id",
        "checkpoint",
        "checkpoint_origin",
        "exact_config",
        "historical_checkpoint_sha256_match",
        "byte_identical_to_historical",
        "configuration_exact_not_byte_identical",
        "validation_only",
    }
)
RECOVERY_POLICY_KEYS = frozenset(
    {
        "policy_key",
        "eligible_attempt_predicate",
        "sort_key",
        "ascending",
        "value_independent",
        "checkpoint_hash_used",
        "checkpoint_size_used",
        "objective_value_used",
        "selected_submission_index",
    }
)
RECOVERY_SUPPLEMENTARY_KEYS = frozenset(
    {
        "tao_job_id",
        "slurm_job_id",
        "submit_time_utc",
        "state",
        "pending_reason",
        "expected_node",
        "exact_config",
        "included_in_selection_policy",
        "selected_recovery_can_change",
        "non_gating",
        "source",
        "entrypoint",
        "remote_specs",
        "sbatch",
    }
)
FILE_IDENTITY_KEYS = frozenset({"path", "sha256", "size_bytes"})
CHECKPOINT_IDENTITY_KEYS = frozenset(
    {"epoch", "path", "sha256", "size_bytes"}
)
SOURCE_FILE_IDENTITY_KEYS = frozenset(
    {"path", "sha256", "git_commit"}
)
STAGED_SUMMARY_KEYS = (
    "allocation_id",
    "allocation_index",
    "design_row_index",
    "candidate_order",
    "candidate_count",
    "block_plan_sha256",
    "command_sha256",
    "command_bytes",
    "staging_bundle_compression",
    "staging_bundle_sha256",
    "staging_bundle_json_sha256",
    "staging_file_sha256",
    "staging_chunk_count",
    "max_staging_chunk_bytes",
    "max_runtime_argument_bytes",
    "staging_root_policy",
    "output_root_expression",
)


class IntegrityError(ValueError):
    """Raised when the sealed campaign cannot be reconstructed exactly."""


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise IntegrityError(
            f"{label} mismatch: expected {expected!r}, got {actual!r}"
        )


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise IntegrityError(f"{label} must be an object")
    return value


def require_exact_keys(
    value: Any,
    expected: Iterable[str],
    label: str,
) -> dict[str, Any]:
    mapping = require_mapping(value, label)
    require_equal(set(mapping), set(expected), f"{label} keys")
    return mapping


def require_false_isolation(
    value: Any,
    label: str,
    *,
    exact: bool,
) -> dict[str, bool]:
    mapping = require_mapping(value, label)
    if exact:
        require_equal(
            set(mapping),
            set(SELECTION_ISOLATION_FLAGS),
            f"{label} keys",
        )
    result = {}
    for key in SELECTION_ISOLATION_FLAGS:
        require_equal(mapping.get(key), False, f"{label} {key}")
        result[key] = False
    return result


def artifact_binding(
    path: Path,
    whole_file_sha256: str,
    internal_sha256: str,
    internal_field: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    resolved = path.resolve()
    value, actual = manifest_generator.load_exact_json(
        resolved,
        whole_file_sha256,
        label,
    )
    actual_internal = manifest_generator.validate_internal_digest(
        value,
        internal_field,
        label,
    )
    expected_internal = manifest_generator.require_sha256(
        internal_sha256,
        f"{label} supplied internal SHA256",
    )
    require_equal(
        actual_internal,
        expected_internal,
        f"{label} internal SHA256",
    )
    return value, {
        "path": str(resolved),
        "whole_file_sha256": actual,
        "internal_sha256": actual_internal,
    }


def validate_exact_recovery_shape(evidence: dict[str, Any]) -> None:
    require_exact_keys(
        evidence,
        RECOVERY_TOP_LEVEL_KEYS,
        "recovery evidence",
    )
    historical = require_exact_keys(
        evidence["historical_checkpoint"],
        {
            "checkpoint",
            "entrypoint",
            "remote_specs",
            "sbatch",
            "tao_job_id",
            "slurm_job_id",
            "node",
            "remote_checkpoint_status",
            "historical_identity_preserved",
            "replacement_of_historical_bytes_permitted",
        },
        "historical recovery provenance",
    )
    require_exact_keys(
        historical["checkpoint"],
        CHECKPOINT_IDENTITY_KEYS,
        "historical checkpoint identity",
    )
    for key in ("entrypoint", "remote_specs", "sbatch"):
        require_exact_keys(
            historical[key],
            FILE_IDENTITY_KEYS,
            f"historical {key} identity",
        )
    source = require_exact_keys(
        evidence["source_identity"],
        RECOVERY_SOURCE_KEYS,
        "recovery source identity",
    )
    require_exact_keys(
        source["expanded_manifest"],
        {"path", "sha256"},
        "recovery expanded-manifest identity",
    )
    require_exact_keys(
        source["seed_archive"],
        {"path", "whole_file_sha256", "internal_sha256"},
        "recovery seed-archive identity",
    )
    require_exact_keys(
        source["candidate_specs"],
        {
            "model.enc_layers",
            "model.dec_layers",
            "train.optim.lr",
            "train.optim.weight_decay",
        },
        "recovery candidate spec",
    )
    require_equal(
        source["candidate_specs"],
        {
            "model.dec_layers": 3,
            "model.enc_layers": 6,
            "train.optim.lr": 0.0003007572504594793,
            "train.optim.weight_decay": 1.1000000000000001e-05,
        },
        "recovery candidate spec values",
    )
    for key in (
        "completed_recovery_launcher",
        "supplementary_recovery_launcher",
    ):
        require_exact_keys(
            source[key],
            SOURCE_FILE_IDENTITY_KEYS,
            f"recovery {key}",
        )
    attempts = evidence["recovery_attempts"]
    if not isinstance(attempts, list) or len(attempts) != 2:
        raise IntegrityError("recovery attempts must contain exactly two rows")
    for index, attempt in enumerate(attempts):
        attempt = require_exact_keys(
            attempt,
            RECOVERY_ATTEMPT_KEYS,
            f"recovery attempt {index}",
        )
        require_exact_keys(
            attempt["checkpoint"],
            CHECKPOINT_IDENTITY_KEYS,
            f"recovery attempt {index} checkpoint",
        )
        for key in ("entrypoint", "remote_specs", "sbatch"):
            require_exact_keys(
                attempt[key],
                FILE_IDENTITY_KEYS,
                f"recovery attempt {index} {key}",
            )
    selected = require_exact_keys(
        evidence["selected_recovery"],
        RECOVERY_SELECTED_KEYS,
        "selected recovery",
    )
    require_exact_keys(
        selected["checkpoint"],
        CHECKPOINT_IDENTITY_KEYS,
        "selected recovery checkpoint",
    )
    require_exact_keys(
        evidence["selection_policy"],
        RECOVERY_POLICY_KEYS,
        "recovery selection policy",
    )
    supplementary = require_exact_keys(
        evidence["supplementary_exact_node_replay"],
        RECOVERY_SUPPLEMENTARY_KEYS,
        "supplementary exact-node replay",
    )
    require_exact_keys(
        supplementary["source"],
        SOURCE_FILE_IDENTITY_KEYS,
        "supplementary replay source",
    )
    for key in ("entrypoint", "remote_specs", "sbatch"):
        require_exact_keys(
            supplementary[key],
            FILE_IDENTITY_KEYS,
            f"supplementary replay {key}",
        )
    require_false_isolation(
        evidence["selection_isolation"],
        "recovery selection isolation",
        exact=True,
    )


def validate_exact_overlay_shape(overlay: dict[str, Any]) -> None:
    require_exact_keys(overlay, OVERLAY_TOP_LEVEL_KEYS, "checkpoint overlay")
    require_equal(
        overlay["scope"],
        {
            "model_family": "DINO ResNet50",
            "dataset_uri": (
                "s3://nvcf-storage-handling/data/"
                "tao_od_synthetic_full_dino_coco/"
            ),
            "candidate_id": checkpoint_overlay.RECOVERED_CANDIDATE_ID,
            "use": "validation_only_latency_measurement_surrogate",
        },
        "checkpoint overlay scope",
    )
    require_false_isolation(
        overlay["selection_isolation"],
        "checkpoint overlay selection isolation",
        exact=True,
    )


def schedule_projection(manifest: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        (
            item["allocation_id"],
            item["allocation_index"],
            item["design_row_index"],
            tuple(item["candidate_order"]),
        )
        for item in manifest["schedule"]["allocations"]
    )


def expected_overlay_source_checks(
    overlay: dict[str, Any],
    overlay_path: Path,
    overlay_whole_file_sha256: str,
) -> dict[str, Any]:
    return checkpoint_overlay.overlay_source_checks(
        overlay,
        overlay_path.resolve(),
        overlay_whole_file_sha256,
    )


def validate_source_overlay_bindings(
    source_checks: Any,
    expected: dict[str, Any],
    label: str,
) -> None:
    source_checks = require_mapping(source_checks, label)
    for key, expected_value in expected.items():
        require_equal(
            source_checks.get(key),
            expected_value,
            f"{label} {key}",
        )


def validate_launch_contract(
    contract: dict[str, Any],
    binding: dict[str, str],
    manifest: dict[str, Any],
    manifest_binding: dict[str, str],
    overlay_checks: dict[str, Any],
) -> None:
    require_false_isolation(contract, "launch contract", exact=False)
    for key in (
        "feeds_final_selection",
        "feeds_reselection",
        "selection_time_objective_replacement_permitted",
        "manual_winner_override_permitted",
    ):
        require_equal(contract.get(key), False, f"launch contract {key}")
    validate_source_overlay_bindings(
        contract.get("source_checks"),
        overlay_checks,
        "launch-contract source checks",
    )
    expected = launcher.launch_contract_payload(
        manifest,
        manifest_binding["whole_file_sha256"],
        Path(binding["path"]).parent,
        contract["source_checks"],
    )
    require_equal(contract, expected, "reconstructed launch contract")


def validate_ledger(
    ledger: dict[str, Any],
    ledger_binding: dict[str, str],
    contract: dict[str, Any],
    contract_binding: dict[str, str],
    manifest: dict[str, Any],
    manifest_binding: dict[str, str],
    overlay_checks: dict[str, Any],
) -> None:
    require_equal(ledger.get("status"), "complete", "submission-ledger status")
    require_equal(ledger.get("ledger_revision"), 1, "submission-ledger revision")
    require_equal(
        ledger.get("pending_submission"),
        None,
        "submission-ledger pending submission",
    )
    require_equal(
        ledger.get("superseded_submissions"),
        [],
        "submission-ledger superseded submissions",
    )
    require_equal(
        ledger.get("submission_recovery_events"),
        [],
        "submission-ledger recovery events",
    )
    require_false_isolation(ledger, "submission ledger", exact=False)
    source_checks = require_mapping(
        ledger.get("source_checks"),
        "submission-ledger source checks",
    )
    validate_source_overlay_bindings(
        source_checks,
        overlay_checks,
        "submission-ledger source checks",
    )
    expected_contract_binding = {
        "path": contract_binding["path"],
        "whole_file_sha256": contract_binding["whole_file_sha256"],
        "internal_sha256": contract_binding["internal_sha256"],
    }
    require_equal(
        source_checks.get("launch_contract"),
        expected_contract_binding,
        "submission-ledger launch-contract binding",
    )
    ledger_without_contract = copy.deepcopy(source_checks)
    del ledger_without_contract["launch_contract"]
    require_equal(
        ledger_without_contract,
        contract["source_checks"],
        "launch-contract/ledger source checks",
    )
    submissions = ledger.get("submissions")
    if not isinstance(submissions, list) or len(submissions) != 6:
        raise IntegrityError("submission ledger must contain exactly six jobs")
    require_equal(
        len({item.get("tao_job_id") for item in submissions}),
        6,
        "unique TAO job count",
    )
    require_equal(
        len({item.get("slurm_job_id") for item in submissions}),
        6,
        "unique SLURM job count",
    )
    for submission in submissions:
        allocation_id = submission.get("allocation_id")
        if allocation_id not in EXPECTED_SUBMISSION_IDENTITIES:
            raise IntegrityError(
                f"unexpected submitted allocation identity: {allocation_id!r}"
            )
        require_equal(
            (
                submission.get("tao_job_id"),
                submission.get("slurm_job_id"),
            ),
            EXPECTED_SUBMISSION_IDENTITIES[allocation_id],
            f"{allocation_id} submitted job identities",
        )
    expected = launcher.submission_ledger_payload(
        manifest,
        manifest_binding["whole_file_sha256"],
        submissions,
        status=ledger["status"],
        source_checks=source_checks,
        ledger_revision=ledger["ledger_revision"],
        superseded_submissions=ledger["superseded_submissions"],
        submission_recovery_events=ledger["submission_recovery_events"],
        parent_ledger_sha256=ledger["parent_ledger_sha256"],
        pending_submission=ledger["pending_submission"],
    )
    require_equal(ledger, expected, "reconstructed submission ledger")
    require_equal(
        manifest_generator.sha256_file(Path(ledger_binding["path"])),
        ledger_binding["whole_file_sha256"],
        "post-reconstruction submission-ledger SHA256",
    )


def reconstruct_execution(
    manifest: dict[str, Any],
    manifest_binding: dict[str, str],
    overlay: dict[str, Any],
    overlay_binding: dict[str, str],
    ledger: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    effective = checkpoint_overlay.execution_manifest(manifest, overlay)
    configs = launcher.generate_configs(effective)
    require_equal(
        tuple(configs),
        EXPECTED_CANDIDATE_IDS,
        "execution-config candidate order",
    )
    config_evidence = {}
    effective_candidates = {
        item["candidate_id"]: item for item in effective["candidates"]
    }
    for candidate_id, payload in configs.items():
        config = yaml.safe_load(payload)
        expected_checkpoint = effective_candidates[candidate_id]["checkpoint"]
        require_equal(
            config["evaluate"]["checkpoint"],
            expected_checkpoint["path"],
            f"{candidate_id} execution-config checkpoint",
        )
        config_evidence[candidate_id] = {
            "sha256": launcher.sha256_bytes(payload),
            "size_bytes": len(payload),
            "checkpoint_path": config["evaluate"]["checkpoint"],
            "checkpoint_sha256": expected_checkpoint["sha256"],
        }

    submissions = {
        item["allocation_id"]: item for item in ledger["submissions"]
    }
    allocation_evidence = []
    plans: dict[str, dict[str, Any]] = {}
    for allocation in manifest["schedule"]["allocations"]:
        allocation_id = allocation["allocation_id"]
        submission = submissions.get(allocation_id)
        if not isinstance(submission, dict):
            raise IntegrityError(
                f"{allocation_id}: submission record is missing"
            )
        base_plan = launcher.build_block_plan(
            effective,
            manifest_binding["whole_file_sha256"],
            allocation,
            configs,
        )
        plan = checkpoint_overlay.augment_block_plan(
            base_plan,
            overlay,
            overlay_binding["whole_file_sha256"],
        )
        plans[allocation_id] = plan
        require_false_isolation(
            plan,
            f"{allocation_id} block plan",
            exact=False,
        )
        rec16 = [
            item
            for item in plan["candidates"]
            if item["candidate_id"]
            == checkpoint_overlay.RECOVERED_CANDIDATE_ID
        ]
        if len(rec16) != 1:
            raise IntegrityError(
                f"{allocation_id}: block plan must contain one rec16"
            )
        rec16_overlay = require_mapping(
            rec16[0].get("checkpoint_overlay"),
            f"{allocation_id} rec16 checkpoint overlay",
        )
        require_equal(
            rec16_overlay["historical_checkpoint"],
            overlay["substitution"]["historical_checkpoint"],
            f"{allocation_id} historical checkpoint binding",
        )
        require_equal(
            rec16_overlay["effective_checkpoint"],
            overlay["substitution"]["effective_checkpoint"],
            f"{allocation_id} effective checkpoint binding",
        )
        require_equal(
            plan["checkpoint_overlay"]["selection_isolation"],
            overlay["selection_isolation"],
            f"{allocation_id} overlay isolation binding",
        )

        command, summary = launcher.staged_command(
            manifest,
            allocation,
            plan,
            configs,
        )
        for key in STAGED_SUMMARY_KEYS:
            require_equal(
                submission.get(key),
                summary[key],
                f"{allocation_id} submitted {key}",
            )
        pretty_plan = (
            json.dumps(plan, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        plan_relative = f"plans/{allocation_id}.json"
        require_equal(
            summary["staging_file_sha256"][plan_relative],
            launcher.sha256_bytes(pretty_plan),
            f"{allocation_id} staged pretty-plan SHA256",
        )
        require_equal(
            summary["command_sha256"],
            launcher.sha256_bytes(command.encode("utf-8")),
            f"{allocation_id} rendered-command SHA256",
        )
        allocation_evidence.append(
            {
                "allocation_id": allocation_id,
                "allocation_index": allocation["allocation_index"],
                "design_row_index": allocation["design_row_index"],
                "candidate_order": copy.deepcopy(
                    allocation["candidate_order"]
                ),
                "block_plan_sha256": plan["block_plan_sha256"],
                "pretty_plan_sha256": launcher.sha256_bytes(pretty_plan),
                "pretty_plan_bytes": len(pretty_plan),
                "command_sha256": summary["command_sha256"],
                "command_bytes": summary["command_bytes"],
                "staging_bundle_sha256": summary["staging_bundle_sha256"],
                "staging_bundle_json_sha256": summary[
                    "staging_bundle_json_sha256"
                ],
                "staging_file_sha256": copy.deepcopy(
                    summary["staging_file_sha256"]
                ),
                "tao_job_id": submission["tao_job_id"],
                "slurm_job_id": submission["slurm_job_id"],
                "all_reconstructed_hashes_match_submission": True,
            }
        )
    require_equal(
        set(submissions),
        {item[0] for item in EXPECTED_SCHEDULE},
        "submission allocation set",
    )
    return {
        "config_count": len(configs),
        "configs": config_evidence,
        "allocation_count": len(allocation_evidence),
        "allocations": allocation_evidence,
        "all_execution_configs_reconstructed": True,
        "all_augmented_plans_reconstructed": True,
        "all_pretty_plan_bytes_reconstructed": True,
        "all_command_and_bundle_hashes_match": True,
    }, effective, plans


def validate_analysis(
    *,
    path: Path,
    whole_file_sha256: str,
    internal_sha256: str,
    manifest: dict[str, Any],
    manifest_binding: dict[str, str],
    overlay: dict[str, Any],
    overlay_binding: dict[str, str],
    ledger: dict[str, Any],
    ledger_binding: dict[str, str],
    effective: dict[str, Any],
    plans: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    report, binding = artifact_binding(
        path,
        whole_file_sha256,
        internal_sha256,
        "report_sha256",
        "matched analysis",
    )
    for key, expected in (
        ("schema_version", 1),
        ("status", "complete"),
        ("manifest_id", manifest["manifest_id"]),
        ("manifest_sha256", manifest_binding["whole_file_sha256"]),
        ("submission_ledger_sha256", ledger_binding["whole_file_sha256"]),
        ("schedule_sha256", manifest["schedule"]["schedule_sha256"]),
        (
            "candidate_ids",
            manifest["candidate_derivation"]["candidate_ids"],
        ),
    ):
        require_equal(report.get(key), expected, f"matched analysis {key}")
    require_equal(
        report.get("source_checks"),
        ledger["source_checks"],
        "matched analysis source checks",
    )
    require_false_isolation(
        report.get("selection_isolation"),
        "matched analysis selection isolation",
        exact=False,
    )
    expected_checkpoint_overlay = {
        **expected_overlay_source_checks(
            overlay,
            Path(overlay_binding["path"]),
            overlay_binding["whole_file_sha256"],
        ),
        "parent_manifest_candidate_records_preserved": True,
        "parent_manifest_selection_snapshot_preserved": True,
        "execution_projection_substitution_count": 1,
        "accuracy_or_selection_evidence_from_recovered_checkpoint": False,
    }
    require_equal(
        report.get("checkpoint_overlay"),
        expected_checkpoint_overlay,
        "matched analysis checkpoint-overlay evidence",
    )
    require_equal(
        report.get("parent_manifest_candidate_checkpoint_evidence"),
        {
            item["candidate_id"]: item["checkpoint"]
            for item in manifest["candidates"]
        },
        "matched analysis parent-checkpoint evidence",
    )

    submissions = {
        item["allocation_id"]: item for item in ledger["submissions"]
    }
    jobs = report.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 6:
        raise IntegrityError("matched analysis must contain exactly six jobs")
    require_equal(
        [item.get("allocation_id") for item in jobs],
        [item[0] for item in EXPECTED_SCHEDULE],
        "matched analysis job order",
    )
    for job in jobs:
        allocation_id = job["allocation_id"]
        submission = submissions[allocation_id]
        for key in (
            "allocation_index",
            "design_row_index",
            "candidate_order",
            "block_plan_sha256",
            "tao_job_id",
            "slurm_job_id",
        ):
            require_equal(
                job.get(key),
                submission[key],
                f"{allocation_id} analysis-job {key}",
            )
        require_equal(
            job["block_plan_sha256"],
            plans[allocation_id]["block_plan_sha256"],
            f"{allocation_id} analysis reconstructed block plan",
        )

    measurements = report.get("per_allocation_candidate_measurements")
    if not isinstance(measurements, list) or len(measurements) != 24:
        raise IntegrityError(
            "matched analysis must contain exactly 24 measurement cells"
        )
    expected_cells = {
        (allocation_id, candidate_id)
        for allocation_id, _, _, _ in EXPECTED_SCHEDULE
        for candidate_id in EXPECTED_CANDIDATE_IDS
    }
    observed_cells = {
        (item.get("allocation_id"), item.get("candidate_id"))
        for item in measurements
    }
    require_equal(
        len(observed_cells),
        24,
        "matched analysis unique measurement-cell count",
    )
    require_equal(
        observed_cells,
        expected_cells,
        "matched analysis 6x4 measurement cells",
    )
    effective_candidates = {
        item["candidate_id"]: item for item in effective["candidates"]
    }
    schedule = {
        item["allocation_id"]: item
        for item in manifest["schedule"]["allocations"]
    }
    for measurement in measurements:
        allocation_id = measurement["allocation_id"]
        candidate_id = measurement["candidate_id"]
        require_equal(
            measurement.get("checkpoint_sha256"),
            effective_candidates[candidate_id]["checkpoint"]["sha256"],
            f"{allocation_id}/{candidate_id} effective checkpoint",
        )
        require_equal(
            measurement.get("position"),
            schedule[allocation_id]["candidate_order"].index(candidate_id),
            f"{allocation_id}/{candidate_id} schedule position",
        )
        require_equal(
            measurement.get("tao_job_id"),
            submissions[allocation_id]["tao_job_id"],
            f"{allocation_id}/{candidate_id} TAO job",
        )
        require_equal(
            measurement.get("slurm_job_id"),
            submissions[allocation_id]["slurm_job_id"],
            f"{allocation_id}/{candidate_id} SLURM job",
        )
    return {
        **binding,
        "status": "validated_complete",
        "job_count": 6,
        "measurement_cell_count": 24,
        "all_result_plan_bindings_match": True,
        "all_effective_checkpoint_bindings_match": True,
    }, binding


def build_audit(
    *,
    manifest_path: Path,
    manifest_whole_file_sha256: str,
    manifest_internal_sha256: str,
    overlay_path: Path,
    overlay_whole_file_sha256: str,
    overlay_internal_sha256: str,
    recovery_evidence_path: Path,
    recovery_evidence_whole_file_sha256: str,
    recovery_evidence_internal_sha256: str,
    launch_contract_path: Path,
    launch_contract_whole_file_sha256: str,
    launch_contract_internal_sha256: str,
    ledger_path: Path,
    ledger_whole_file_sha256: str,
    ledger_internal_sha256: str,
    analysis_path: Path | None = None,
    analysis_whole_file_sha256: str | None = None,
    analysis_internal_sha256: str | None = None,
) -> dict[str, Any]:
    analysis_arguments = (
        analysis_path,
        analysis_whole_file_sha256,
        analysis_internal_sha256,
    )
    if any(item is not None for item in analysis_arguments) and not all(
        item is not None for item in analysis_arguments
    ):
        raise IntegrityError(
            "analysis path, whole-file SHA256, and internal SHA256 "
            "must be supplied together"
        )

    artifact_paths = {
        "manifest": manifest_path.resolve(),
        "overlay": overlay_path.resolve(),
        "recovery_evidence": recovery_evidence_path.resolve(),
        "launch_contract": launch_contract_path.resolve(),
        "submission_ledger": ledger_path.resolve(),
    }
    if analysis_path is not None:
        artifact_paths["matched_analysis"] = analysis_path.resolve()
    before = {
        key: manifest_generator.sha256_file(path)
        for key, path in artifact_paths.items()
    }

    manifest, manifest_binding = artifact_binding(
        manifest_path,
        manifest_whole_file_sha256,
        manifest_internal_sha256,
        "manifest_sha256",
        "latency-feasible manifest",
    )
    launcher.validate_manifest_contract(manifest)
    require_equal(
        tuple(manifest["candidate_derivation"]["candidate_ids"]),
        EXPECTED_CANDIDATE_IDS,
        "frozen feasible candidate IDs",
    )
    require_equal(
        schedule_projection(manifest),
        EXPECTED_SCHEDULE,
        "frozen six-allocation schedule",
    )

    recovery, recovery_binding = artifact_binding(
        recovery_evidence_path,
        recovery_evidence_whole_file_sha256,
        recovery_evidence_internal_sha256,
        "evidence_sha256",
        "rec16 recovery evidence",
    )
    validate_exact_recovery_shape(recovery)
    selected_recovery = checkpoint_overlay.validate_recovery_evidence(
        recovery,
        recovery_binding["whole_file_sha256"],
    )

    overlay, overlay_binding = artifact_binding(
        overlay_path,
        overlay_whole_file_sha256,
        overlay_internal_sha256,
        "overlay_sha256",
        "rec16 checkpoint overlay",
    )
    validate_exact_overlay_shape(overlay)
    require_equal(
        overlay["recovery_evidence"],
        {
            "path": recovery_binding["path"],
            "whole_file_sha256": recovery_binding["whole_file_sha256"],
            "internal_sha256": recovery_binding["internal_sha256"],
            "evidence_id": recovery["evidence_id"],
        },
        "overlay/CLI recovery-evidence binding",
    )
    checkpoint_overlay.validate_overlay(
        overlay,
        overlay_binding["whole_file_sha256"],
        manifest,
        Path(manifest_binding["path"]),
        manifest_binding["whole_file_sha256"],
    )
    require_equal(
        overlay["substitution"]["effective_checkpoint"],
        selected_recovery["checkpoint"],
        "overlay selected-recovery checkpoint",
    )

    overlay_checks = expected_overlay_source_checks(
        overlay,
        Path(overlay_binding["path"]),
        overlay_binding["whole_file_sha256"],
    )
    contract, contract_binding = artifact_binding(
        launch_contract_path,
        launch_contract_whole_file_sha256,
        launch_contract_internal_sha256,
        "contract_sha256",
        "latency-feasible launch contract",
    )
    validate_launch_contract(
        contract,
        contract_binding,
        manifest,
        manifest_binding,
        overlay_checks,
    )
    ledger, ledger_binding = artifact_binding(
        ledger_path,
        ledger_whole_file_sha256,
        ledger_internal_sha256,
        "ledger_sha256",
        "latency-feasible submission ledger",
    )
    validate_ledger(
        ledger,
        ledger_binding,
        contract,
        contract_binding,
        manifest,
        manifest_binding,
        overlay_checks,
    )
    execution, effective, plans = reconstruct_execution(
        manifest,
        manifest_binding,
        overlay,
        overlay_binding,
        ledger,
    )

    analysis_evidence: dict[str, Any] = {
        "status": "not_supplied",
        "reason": (
            "final matched analysis is optional until aggregation completes"
        ),
        "result_plan_bindings_checked": False,
        "effective_checkpoint_bindings_checked": False,
    }
    if analysis_path is not None:
        analysis_evidence, _ = validate_analysis(
            path=analysis_path,
            whole_file_sha256=str(analysis_whole_file_sha256),
            internal_sha256=str(analysis_internal_sha256),
            manifest=manifest,
            manifest_binding=manifest_binding,
            overlay=overlay,
            overlay_binding=overlay_binding,
            ledger=ledger,
            ledger_binding=ledger_binding,
            effective=effective,
            plans=plans,
        )

    after = {
        key: manifest_generator.sha256_file(path)
        for key, path in artifact_paths.items()
    }
    require_equal(after, before, "read-only input-artifact digests")
    audit = {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "status": "pass",
        "scope": {
            "model_family": "DINO ResNet50",
            "dataset_uri": (
                "s3://nvcf-storage-handling/data/"
                "tao_od_synthetic_full_dino_coco/"
            ),
            "campaign": "98%-accuracy-feasible matched latency",
            "validation_only": True,
        },
        "artifacts": {
            "manifest": manifest_binding,
            "overlay": overlay_binding,
            "recovery_evidence": recovery_binding,
            "launch_contract": contract_binding,
            "submission_ledger": ledger_binding,
        },
        "cohort_and_schedule": {
            "candidate_count": 4,
            "candidate_ids": list(EXPECTED_CANDIDATE_IDS),
            "candidate_set_sha256": manifest["candidate_derivation"][
                "candidate_set_sha256"
            ],
            "allocation_count": 6,
            "measurement_cell_count": 24,
            "schedule_sha256": manifest["schedule"]["schedule_sha256"],
            "schedule": [
                {
                    "allocation_id": allocation_id,
                    "allocation_index": allocation_index,
                    "design_row_index": design_row_index,
                    "candidate_order": list(candidate_order),
                }
                for (
                    allocation_id,
                    allocation_index,
                    design_row_index,
                    candidate_order,
                ) in EXPECTED_SCHEDULE
            ],
            "exact_frozen_6x4_design": True,
        },
        "recovery_provenance": {
            "evidence_id": recovery["evidence_id"],
            "candidate_id": recovery["candidate_id"],
            "attempt_count": len(recovery["recovery_attempts"]),
            "attempt_identities": [
                {
                    "submission_index": item["submission_index"],
                    "tao_job_id": item["tao_job_id"],
                    "slurm_job_id": item["slurm_job_id"],
                    "node": item["node"],
                    "checkpoint_sha256": item["checkpoint"]["sha256"],
                }
                for item in recovery["recovery_attempts"]
            ],
            "selection_policy": recovery["selection_policy"]["policy_key"],
            "selected_submission_index": selected_recovery[
                "submission_index"
            ],
            "selected_tao_job_id": selected_recovery["tao_job_id"],
            "selected_slurm_job_id": selected_recovery["slurm_job_id"],
            "historical_checkpoint_sha256": overlay["substitution"][
                "historical_checkpoint"
            ]["sha256"],
            "effective_checkpoint_sha256": overlay["substitution"][
                "effective_checkpoint"
            ]["sha256"],
            "exact_shape_and_identities_validated": True,
        },
        "overlay_contract": {
            "overlay_id": overlay["overlay_id"],
            "candidate_id": overlay["scope"]["candidate_id"],
            "substitution_count": overlay["invariants"][
                "substitution_count"
            ],
            "historical_byte_match": overlay["substitution"][
                "historical_byte_match"
            ],
            "top_level_shape_exact": True,
            "scope_exact": True,
            "isolation_shape_exact": True,
        },
        "launch_contract": {
            "reconstructed_exactly": True,
            "source_overlay_bindings_exact": True,
        },
        "submission_ledger": {
            "reconstructed_exactly": True,
            "status": ledger["status"],
            "ledger_revision": ledger["ledger_revision"],
            "submission_count": len(ledger["submissions"]),
            "submission_recovery_event_count": len(
                ledger["submission_recovery_events"]
            ),
            "source_overlay_bindings_exact": True,
        },
        "execution_reconstruction": execution,
        "matched_analysis": analysis_evidence,
        "selection_isolation": {
            key: False for key in SELECTION_ISOLATION_FLAGS
        },
        "verifier_operations": {
            "selector_called": False,
            "analyze_archive_called": False,
            "scheduler_or_sdk_called": False,
            "network_called": False,
            "measurement_called": False,
            "selection_or_reselection_called": False,
            "input_artifact_hashes_before": before,
            "input_artifact_hashes_after": after,
        },
        "post_submission_disposition": {
            "launch_artifacts_modified": False,
            "measurements_reused": True,
            "rerun_required": False,
        },
    }
    audit["audit_sha256"] = manifest_generator.sha256_value(audit)
    return audit


def write_new(path: Path, audit: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    except FileExistsError as error:
        raise IntegrityError(
            f"refusing to overwrite immutable integrity audit: {path}"
        ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest-whole-file-sha256", required=True)
    parser.add_argument("--manifest-internal-sha256", required=True)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--overlay-whole-file-sha256", required=True)
    parser.add_argument("--overlay-internal-sha256", required=True)
    parser.add_argument(
        "--recovery-evidence",
        type=Path,
        default=DEFAULT_RECOVERY_EVIDENCE,
    )
    parser.add_argument(
        "--recovery-evidence-whole-file-sha256",
        required=True,
    )
    parser.add_argument("--recovery-evidence-internal-sha256", required=True)
    parser.add_argument(
        "--launch-contract",
        type=Path,
        default=DEFAULT_LAUNCH_CONTRACT,
    )
    parser.add_argument("--launch-contract-whole-file-sha256", required=True)
    parser.add_argument("--launch-contract-internal-sha256", required=True)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--ledger-whole-file-sha256", required=True)
    parser.add_argument("--ledger-internal-sha256", required=True)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--analysis-whole-file-sha256")
    parser.add_argument("--analysis-internal-sha256")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if DEFAULT_ANALYSIS.is_file() and args.analysis is None:
        raise IntegrityError(
            "final matched analysis exists; bind it with all three "
            "--analysis arguments"
        )
    audit = build_audit(
        manifest_path=args.manifest,
        manifest_whole_file_sha256=args.manifest_whole_file_sha256,
        manifest_internal_sha256=args.manifest_internal_sha256,
        overlay_path=args.overlay,
        overlay_whole_file_sha256=args.overlay_whole_file_sha256,
        overlay_internal_sha256=args.overlay_internal_sha256,
        recovery_evidence_path=args.recovery_evidence,
        recovery_evidence_whole_file_sha256=(
            args.recovery_evidence_whole_file_sha256
        ),
        recovery_evidence_internal_sha256=(
            args.recovery_evidence_internal_sha256
        ),
        launch_contract_path=args.launch_contract,
        launch_contract_whole_file_sha256=(
            args.launch_contract_whole_file_sha256
        ),
        launch_contract_internal_sha256=args.launch_contract_internal_sha256,
        ledger_path=args.ledger,
        ledger_whole_file_sha256=args.ledger_whole_file_sha256,
        ledger_internal_sha256=args.ledger_internal_sha256,
        analysis_path=args.analysis,
        analysis_whole_file_sha256=args.analysis_whole_file_sha256,
        analysis_internal_sha256=args.analysis_internal_sha256,
    )
    write_new(args.output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
