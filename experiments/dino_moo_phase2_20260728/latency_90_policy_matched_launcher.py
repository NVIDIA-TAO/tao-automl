#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Project, dry-run, or launch the 90%-policy matched latency campaign.

Candidate membership comes only from the frozen archive replay.  Rendering,
secure staging, block execution, SQSH use, and SlurmSDK submission delegate to
the already validated post-front matched campaign machinery.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import post_front_matched_launcher as POST_FRONT  # noqa: E402


CAMPAIGN_ID = "dino_latency_90pct_matched_20260728_v1"
COMPATIBILITY_MANIFEST_ID = "dino_expanded_post_front_matched_20260728_v1"
EXPECTED_ACKNOWLEDGEMENT = (
    "USER_AUTHORIZED_DINO_LATENCY_90_MATCHED_6X8GPU_VALIDATION_20260728"
)
REPLAY_PATH = HERE / "latency_90_policy" / "archive_replay.v1.json"
PROFILE_PATH = HERE / "dino_latency_90_policy_profile.v1.json"
CANDIDATE_TABLE_PATH = (
    HERE
    / "runtime"
    / "expanded_search_v2"
    / "expanded_candidate_table.json"
)
SEALED_SELECTION_PATH = (
    HERE
    / "runtime"
    / "expanded_search_v2"
    / "expanded_combined_selection.json"
)
BASE_MANIFEST_PATH = HERE / "post_front_matched_manifest.v1.json"
DEFAULT_PROJECTION_PATH = (
    HERE / "latency_90_policy" / "matched" / "execution_projection.v1.json"
)
DEFAULT_RECOVERY_EVIDENCE_PATH = (
    HERE
    / "latency_90_policy"
    / "matched"
    / "rec6_checkpoint_recovery_evidence.v1.json"
)
DEFAULT_RUNTIME_DIR = (
    HERE / "runtime" / "latency_90_policy" / "matched"
)
RECOVERY_EVIDENCE_ID = "dino_latency_90_checkpoint_recovery_20260728_v1"
# A read-only remote provenance check found this exact frozen checkpoint
# deleted. Candidate identity still comes exclusively from the replay scope;
# this digest only marks the resulting recovery requirement.
RECOVERY_REQUIRED_ORIGINAL_CHECKPOINT_SHA256 = (
    "0338c35be50bbad6189d38e8f9007856a60e87a0861c8a6ff5d0bf85cd6df6c5"
)

EXPECTED_FROZEN_SHA256 = {
    REPLAY_PATH: (
        "7441053780c8b400a239a71f5e50b0b37813f8968a1284567fccd3dc857d33a9"
    ),
    PROFILE_PATH: (
        "f6e56ff8d61c91654a13c9759d7cc63f371ed66a9e958bb90ae56cba5112739e"
    ),
    CANDIDATE_TABLE_PATH: (
        "5ba323d05d9ec8e3703e636f8b5e2975cc620eeec10df75ec6e792318dc2df03"
    ),
    SEALED_SELECTION_PATH: (
        "78ab9d2fa83cc3abe9057d137c0b88f120158b6ad77268482d2c18f5a1533af1"
    ),
    BASE_MANIFEST_PATH: (
        "d468d5d26f607b115c7c1732966f0ac98664fd232ce83abfa6becc0ce062b7b6"
    ),
}
SELECTION_ISOLATION = {
    "selector_invoked_on_matched_measurements": False,
    "selection_time_objectives_replaced": False,
    "measurements_feed_selection": False,
    "measurements_feed_reselection": False,
    "algorithm_selected_candidate_overridden": False,
}


class ProjectionError(RuntimeError):
    """Raised when the matched execution projection loses provenance."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProjectionError(f"{path} must contain a JSON object")
    return value


def source_binding(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, str]:
    resolved = path.resolve()
    observed = file_sha256(resolved)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ProjectionError(
            f"Frozen source hash mismatch for {resolved}: "
            f"expected {expected_sha256}, observed {observed}"
        )
    try:
        display_path = str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "sha256": observed,
    }


def validate_internal_artifact(
    artifact: dict[str, Any],
    *,
    label: str,
) -> str:
    integrity = artifact.get("artifact_integrity")
    if not isinstance(integrity, dict):
        raise ProjectionError(f"{label} lacks artifact integrity")
    claimed = integrity.get("canonical_payload_sha256")
    core = {
        key: value
        for key, value in artifact.items()
        if key != "artifact_integrity"
    }
    observed = canonical_sha256(core)
    if claimed != observed:
        raise ProjectionError(
            f"{label} internal digest mismatch: {claimed} != {observed}"
        )
    return observed


def load_sources() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    for path, expected in EXPECTED_FROZEN_SHA256.items():
        source_binding(path, expected_sha256=expected)
    replay = load_json(REPLAY_PATH)
    replay_internal = validate_internal_artifact(
        replay,
        label="90%-policy archive replay",
    )
    if replay_internal != replay["artifact_integrity"][
        "canonical_payload_sha256"
    ]:
        raise ProjectionError("Replay internal identity drift")
    profile = load_json(PROFILE_PATH)
    table = load_json(CANDIDATE_TABLE_PATH)
    base_manifest = load_json(BASE_MANIFEST_PATH)
    return replay, profile, table, base_manifest


def validate_replay_policy(
    replay: dict[str, Any],
    profile: dict[str, Any],
) -> list[str]:
    if replay.get("selection_isolation") != SELECTION_ISOLATION:
        raise ProjectionError("Replay selection-isolation flags drifted")
    settings = profile.get("automl_settings", {})
    if (
        settings.get("latency_accuracy_retention", {}).get(
            "retained_fraction"
        )
        != 0.90
        or settings.get("latency_tolerance") != 0.73553775
        or settings.get("multi_objective_min_accuracy") is not None
    ):
        raise ProjectionError("90%-policy profile settings drifted")
    if replay.get("policy", {}).get(
        "latency_accuracy_retention"
    ) != 0.90:
        raise ProjectionError("Replay retention is not 90%")
    if replay.get("policy", {}).get(
        "multi_objective_min_accuracy"
    ) is not None:
        raise ProjectionError("Replay unexpectedly constrains multi-objective")
    scope = replay.get("latency_tied_cohort_audit", {}).get(
        "matched_validation_scope", {}
    )
    candidate_ids = scope.get("candidate_ids")
    if (
        not isinstance(candidate_ids, list)
        or not candidate_ids
        or any(not isinstance(item, str) or not item for item in candidate_ids)
        or len(set(candidate_ids)) != len(candidate_ids)
    ):
        raise ProjectionError("Replay matched scope is invalid")
    expected_scope = sorted(
        set(scope.get("equivalent_fastest_candidate_ids", []))
        | set(scope.get(
            "additional_uncertainty_plausible_candidate_ids",
            [],
        ))
    )
    if sorted(candidate_ids) != expected_scope:
        raise ProjectionError(
            "Replay matched scope is not its cohort plus plausible outsiders"
        )
    return sorted(candidate_ids)


def derive_candidates(
    replay: dict[str, Any],
    profile: dict[str, Any],
    table: dict[str, Any],
) -> list[dict[str, Any]]:
    candidate_ids = validate_replay_policy(replay, profile)
    rows = table.get("rows")
    if not isinstance(rows, list) or len(rows) != 60:
        raise ProjectionError("Frozen expanded table must contain 60 rows")
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ProjectionError("Candidate table row is not an object")
        candidate_id = row.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in rows_by_id
        ):
            raise ProjectionError("Candidate table IDs are invalid or duplicate")
        rows_by_id[candidate_id] = row

    records = []
    for candidate_id in candidate_ids:
        row = rows_by_id.get(candidate_id)
        if row is None:
            raise ProjectionError(
                f"Replay-derived candidate lacks frozen row: {candidate_id}"
            )
        if row.get("status") != "success":
            raise ProjectionError(
                f"Replay-derived candidate is not successful: {candidate_id}"
            )
        model = row.get("resolved_model_spec")
        model_sha256 = row.get("resolved_model_spec_sha256")
        if (
            not isinstance(model, dict)
            or canonical_sha256(model) != model_sha256
        ):
            raise ProjectionError(
                f"Resolved model digest mismatch: {candidate_id}"
            )
        checkpoint = row.get("checkpoint")
        if (
            not isinstance(checkpoint, dict)
            or not Path(str(checkpoint.get("path", ""))).is_absolute()
            or not isinstance(checkpoint.get("sha256"), str)
            or len(checkpoint["sha256"]) != 64
        ):
            raise ProjectionError(
                f"Checkpoint binding is invalid: {candidate_id}"
            )
        records.append({
            "candidate_id": candidate_id,
            "candidate_table_record_sha256": canonical_sha256(row),
            "selection_audit_sha256": canonical_sha256(
                row["selection_audit"]
            ),
            "search_seed": row.get("search_seed"),
            "training_seed": row.get("training_seed"),
            "rec_id": row.get("rec_id"),
            "train_job_id": row.get("train_job_id"),
            "specs": copy.deepcopy(row.get("specs")),
            "selection_time_objective_values": copy.deepcopy(
                row.get("objective_values")
            ),
            "resolved_model_spec": copy.deepcopy(model),
            "resolved_model_spec_sha256": model_sha256,
            "checkpoint": {
                "path": checkpoint["path"],
                "sha256": checkpoint["sha256"],
            },
        })
    if [item["candidate_id"] for item in records] != candidate_ids:
        raise ProjectionError("Candidate records lost canonical replay order")
    return records


def all_source_bindings(
    base_manifest: dict[str, Any],
) -> dict[str, dict[str, str]]:
    bindings = {
        "archive_replay": source_binding(
            REPLAY_PATH,
            expected_sha256=EXPECTED_FROZEN_SHA256[REPLAY_PATH],
        ),
        "policy_profile": source_binding(
            PROFILE_PATH,
            expected_sha256=EXPECTED_FROZEN_SHA256[PROFILE_PATH],
        ),
        "expanded_candidate_table": source_binding(
            CANDIDATE_TABLE_PATH,
            expected_sha256=EXPECTED_FROZEN_SHA256[CANDIDATE_TABLE_PATH],
        ),
        "sealed_combined_selection": source_binding(
            SEALED_SELECTION_PATH,
            expected_sha256=EXPECTED_FROZEN_SHA256[SEALED_SELECTION_PATH],
        ),
        "base_post_front_manifest": source_binding(
            BASE_MANIFEST_PATH,
            expected_sha256=EXPECTED_FROZEN_SHA256[BASE_MANIFEST_PATH],
        ),
        "execution_projection_launcher": source_binding(Path(__file__)),
        "post_front_matched_launcher": source_binding(
            HERE / "post_front_matched_launcher.py"
        ),
        "post_front_matched_block_runner": source_binding(
            HERE / "post_front_matched_block_runner.py"
        ),
        "post_front_matched_manifest_generator": source_binding(
            HERE / "post_front_matched_manifest_generator.py"
        ),
        "latency_90_policy_matched_aggregator": source_binding(
            HERE / "latency_90_policy_matched_aggregator.py"
        ),
    }
    sources = base_manifest["source_artifacts"]
    for key in (
        "expanded_runner",
        "dino_latency_benchmark",
        "latency_stats",
        "dino_evaluate_template",
    ):
        source = sources[key]
        bindings[key] = source_binding(
            Path(source["path"]),
            expected_sha256=source["sha256"],
        )
    return dict(sorted(bindings.items()))


def build_projection_core() -> dict[str, Any]:
    replay, profile, table, base_manifest = load_sources()
    candidates = derive_candidates(replay, profile, table)
    candidate_ids = [item["candidate_id"] for item in candidates]
    schedule = POST_FRONT.manifest_generator.build_schedule(candidate_ids)
    if len(schedule["allocations"]) != 6:
        raise ProjectionError("Matched schedule must have six allocations")
    bindings = all_source_bindings(base_manifest)
    replay_binding = bindings["archive_replay"]
    replay_binding["internal_sha256"] = replay["artifact_integrity"][
        "canonical_payload_sha256"
    ]
    profile_binding = bindings["policy_profile"]
    profile_binding["canonical_sha256"] = canonical_sha256(profile)
    runtime = copy.deepcopy(base_manifest["runtime"])
    runtime["local_runtime_path"] = str(DEFAULT_RUNTIME_DIR.resolve())
    recovery_candidates = [
        item
        for item in candidates
        if item["checkpoint"]["sha256"]
        == RECOVERY_REQUIRED_ORIGINAL_CHECKPOINT_SHA256
    ]
    if len(recovery_candidates) != 1:
        raise ProjectionError(
            "Expected exactly one replay-derived missing checkpoint"
        )
    recovery_candidate = recovery_candidates[0]
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "status": "immutable_execution_projection_not_launched",
        "purpose": (
            "Validation-only matched latency comparison for the complete "
            "production-selector-derived 90%-retention equivalent-fastest "
            "cohort."
        ),
        "compatibility_contract": {
            "manifest_id": COMPATIBILITY_MANIFEST_ID,
            "allocation_id_pattern": "post_front_allocation_<00..05>",
            "output_layout": runtime["output_contract"],
            "reason": (
                "The proven post-front block runner validates this legacy "
                "identity and layout; TAO job-scoped roots prevent collision. "
                "The distinct campaign_id is bound in every plan."
            ),
        },
        "candidate_derivation": {
            "source": (
                "archive_replay.v1.json matched_validation_scope.candidate_ids"
            ),
            "manual_candidate_addition_or_removal_used": False,
            "candidate_ids": candidate_ids,
            "candidate_count": len(candidates),
            "candidate_set_sha256": canonical_sha256(candidate_ids),
            "records_joined_by": "candidate_id",
            "records_source": (
                "sealed expanded_candidate_table.json complete 60-row archive"
            ),
        },
        "candidates": candidates,
        "checkpoint_availability": {
            "direct_remote_provenance_check_complete": True,
            "exact_frozen_checkpoint_available_candidate_ids": [
                item["candidate_id"]
                for item in candidates
                if item["candidate_id"]
                != recovery_candidate["candidate_id"]
            ],
            "identity_preserving_recovery_required_candidate_ids": [
                recovery_candidate["candidate_id"]
            ],
            "requirements": [{
                "candidate_id": recovery_candidate["candidate_id"],
                "historical_checkpoint": copy.deepcopy(
                    recovery_candidate["checkpoint"]
                ),
                "candidate_table_record_sha256": recovery_candidate[
                    "candidate_table_record_sha256"
                ],
                "resolved_model_spec_sha256": recovery_candidate[
                    "resolved_model_spec_sha256"
                ],
                "specs_sha256": canonical_sha256(
                    recovery_candidate["specs"]
                ),
                "search_seed": recovery_candidate["search_seed"],
                "training_seed": recovery_candidate["training_seed"],
                "rec_id": recovery_candidate["rec_id"],
                "status": "exact_historical_checkpoint_deleted",
                "required_resolution": (
                    "identity-preserving recovery with immutable provenance"
                ),
                "architecture_proxy_permitted": False,
                "candidate_substitution_permitted": False,
            }],
            "launch_blocked_until_recovery_evidence": True,
            "recovery_evidence_id": RECOVERY_EVIDENCE_ID,
            "default_recovery_evidence_path": str(
                DEFAULT_RECOVERY_EVIDENCE_PATH.relative_to(REPO_ROOT)
            ),
        },
        "schedule": schedule,
        "latency_protocol": copy.deepcopy(
            base_manifest["latency_protocol"]
        ),
        "runtime": runtime,
        "source_bindings": bindings,
        "selection_isolation": dict(SELECTION_ISOLATION),
        "machinery_reuse": {
            "schedule_builder": (
                "post_front_matched_manifest_generator.build_schedule"
            ),
            "config_builder": "post_front_matched_launcher.generate_configs",
            "plan_builder": "post_front_matched_launcher.build_block_plan",
            "secure_staging": "post_front_matched_launcher.staged_command",
            "block_runner": "post_front_matched_block_runner.py",
            "submission": "post_front_matched_launcher.submit_all",
            "aggregation": (
                "latency_90_policy_matched_aggregator reusing "
                "post_front_matched_aggregator.aggregate_bundles and "
                "paired inference utilities"
            ),
            "sdk_api": "tao_sdk.platforms.slurm.SlurmSDK.create_job",
            "prebuilt_sqsh": True,
        },
        "launch_policy": {
            "allocation_count": 6,
            "nodes_per_allocation": 1,
            "gpus_per_allocation": 8,
            "launches_concurrently_without_waiting": True,
            "launch_performed_by_projection_generation": False,
            "required_acknowledgement": EXPECTED_ACKNOWLEDGEMENT,
        },
    }


def build_projection() -> dict[str, Any]:
    core = build_projection_core()
    return {
        **core,
        "artifact_integrity": {
            "hash_algorithm": "sha256",
            "canonical_payload_sha256": canonical_sha256(core),
            "hash_excludes": ["artifact_integrity"],
        },
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def load_projection(
    path: Path,
    supplied_sha256: str,
) -> tuple[dict[str, Any], str]:
    whole_file_sha256 = file_sha256(path)
    if whole_file_sha256 != supplied_sha256:
        raise ProjectionError(
            "Execution projection whole-file SHA256 does not match CLI"
        )
    projection = load_json(path)
    validate_internal_artifact(
        projection,
        label="90%-policy matched execution projection",
    )
    if projection != build_projection():
        raise ProjectionError(
            "Execution projection differs from exact source reconstruction"
        )
    return projection, whole_file_sha256


def compatibility_manifest(
    projection: dict[str, Any],
    recovery_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = load_json(BASE_MANIFEST_PATH)
    manifest = copy.deepcopy(base)
    candidate_ids = projection["candidate_derivation"]["candidate_ids"]
    manifest.update({
        "campaign_id": projection["campaign_id"],
        "manifest_id": COMPATIBILITY_MANIFEST_ID,
        "manifest_sha256": projection["artifact_integrity"][
            "canonical_payload_sha256"
        ],
        "status": "immutable_ready_to_launch",
        "candidate_derivation": {
            "source": projection["candidate_derivation"]["source"],
            "candidate_ids": candidate_ids,
            "candidate_count": len(candidate_ids),
            "candidate_set_sha256": projection["candidate_derivation"][
                "candidate_set_sha256"
            ],
            "manual_filtering_used": False,
            "manual_candidate_addition_or_removal_used": False,
        },
        "candidates": copy.deepcopy(projection["candidates"]),
        "schedule": copy.deepcopy(projection["schedule"]),
        "latency_protocol": copy.deepcopy(projection["latency_protocol"]),
        "runtime": copy.deepcopy(projection["runtime"]),
        "selection_isolation": dict(SELECTION_ISOLATION),
        "feeds_final_selection": False,
        "feeds_reselection": False,
        "manual_candidate_addition_or_removal_permitted": False,
        "manual_winner_override_permitted": False,
        "selection_time_objective_replacement_permitted": False,
    })
    if recovery_evidence is not None:
        recovery_candidate_id = recovery_evidence["candidate_id"]
        candidate = next(
            item
            for item in manifest["candidates"]
            if item["candidate_id"] == recovery_candidate_id
        )
        candidate["historical_checkpoint"] = copy.deepcopy(
            candidate["checkpoint"]
        )
        candidate["checkpoint"] = copy.deepcopy(
            recovery_evidence["recovered_checkpoint"]
        )
        candidate["checkpoint_recovery_provenance"] = {
            "evidence_id": recovery_evidence["evidence_id"],
            "evidence_internal_sha256": recovery_evidence[
                "artifact_integrity"
            ]["canonical_payload_sha256"],
            "identity_preserving": True,
            "architecture_proxy_used": False,
        }
    return manifest


def load_recovery_evidence(
    projection: dict[str, Any],
    path: Path,
    supplied_sha256: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    observed_sha256 = file_sha256(path)
    if observed_sha256 != supplied_sha256:
        raise ProjectionError(
            "Checkpoint-recovery evidence whole-file SHA256 mismatch"
        )
    evidence = load_json(path)
    internal_sha256 = validate_internal_artifact(
        evidence,
        label="checkpoint-recovery evidence",
    )
    requirements = projection["checkpoint_availability"]["requirements"]
    if len(requirements) != 1:
        raise ProjectionError("Recovery requirement cardinality drift")
    requirement = requirements[0]
    expected = {
        "schema_version": 1,
        "evidence_id": RECOVERY_EVIDENCE_ID,
        "status": "identity_preserving_recovery_complete",
        "campaign_id": projection["campaign_id"],
        "candidate_id": requirement["candidate_id"],
        "historical_checkpoint": requirement["historical_checkpoint"],
        "candidate_table_record_sha256": requirement[
            "candidate_table_record_sha256"
        ],
        "resolved_model_spec_sha256": requirement[
            "resolved_model_spec_sha256"
        ],
        "specs_sha256": requirement["specs_sha256"],
        "search_seed": requirement["search_seed"],
        "training_seed": requirement["training_seed"],
        "rec_id": requirement["rec_id"],
        "exact_candidate_configuration_preserved": True,
        "architecture_proxy_used": False,
        "manual_candidate_substitution_used": False,
        "result_driven_parameter_change_used": False,
        "measurements_feed_selection": False,
        "measurements_feed_reselection": False,
        "algorithm_selected_candidate_overridden": False,
    }
    for key, value in expected.items():
        if evidence.get(key) != value:
            raise ProjectionError(
                f"Checkpoint-recovery evidence drift: {key}"
            )
    recovered = evidence.get("recovered_checkpoint")
    if (
        not isinstance(recovered, dict)
        or not Path(str(recovered.get("path", ""))).is_absolute()
        or not isinstance(recovered.get("sha256"), str)
        or len(recovered["sha256"]) != 64
    ):
        raise ProjectionError("Recovered checkpoint binding is invalid")
    return evidence, {
        "path": str(path.resolve()),
        "whole_file_sha256": observed_sha256,
        "internal_sha256": internal_sha256,
    }


def build_plans_and_commands(
    projection: dict[str, Any],
    projection_whole_sha256: str,
    recovery_evidence: dict[str, Any] | None = None,
    recovery_binding: dict[str, str] | None = None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[tuple[str, dict[str, Any]]],
]:
    if (recovery_evidence is None) != (recovery_binding is None):
        raise ProjectionError(
            "Recovery evidence and binding must be supplied together"
        )
    manifest = compatibility_manifest(projection, recovery_evidence)
    configs = POST_FRONT.generate_configs(manifest)
    plans = []
    rendered = []
    for allocation in manifest["schedule"]["allocations"]:
        plan = POST_FRONT.build_block_plan(
            manifest,
            projection_whole_sha256,
            allocation,
            configs,
        )
        del plan["block_plan_sha256"]
        plan.update({
            "campaign_id": projection["campaign_id"],
            "compatibility_manifest_id": COMPATIBILITY_MANIFEST_ID,
            "execution_projection": {
                "path": str(DEFAULT_PROJECTION_PATH.relative_to(REPO_ROOT)),
                "whole_file_sha256": projection_whole_sha256,
                "internal_sha256": projection["artifact_integrity"][
                    "canonical_payload_sha256"
                ],
            },
            "source_bindings": copy.deepcopy(
                projection["source_bindings"]
            ),
            "selection_isolation": dict(SELECTION_ISOLATION),
            "checkpoint_recovery": {
                "required_candidate_ids": projection[
                    "checkpoint_availability"
                ]["identity_preserving_recovery_required_candidate_ids"],
                "resolved": recovery_evidence is not None,
                "evidence": copy.deepcopy(recovery_binding),
                "architecture_proxy_used": False,
                "measurements_feed_selection": False,
                "measurements_feed_reselection": False,
            },
        })
        plan["block_plan_sha256"] = canonical_sha256(plan)
        plans.append(plan)
        rendered.append(
            POST_FRONT.staged_command(
                manifest,
                allocation,
                plan,
                configs,
            )
        )
    POST_FRONT.validate_submission_commands(manifest, rendered)
    return manifest, plans, rendered


def verify_source_bindings(
    projection: dict[str, Any],
) -> dict[str, Any]:
    checks = {}
    for name, binding in projection["source_bindings"].items():
        path = Path(binding["path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        observed = file_sha256(path)
        expected = binding["sha256"]
        if observed != expected:
            raise ProjectionError(
                f"Projection source changed: {name}: {observed} != {expected}"
            )
        checks[name] = {
            "path": binding["path"],
            "expected_sha256": expected,
            "observed_sha256": observed,
            "match": True,
        }
    return checks


def validate_campaign_source_state(
    projection_path: Path,
) -> dict[str, Any]:
    paths = [
        Path(__file__).resolve(),
        projection_path.resolve(),
        REPLAY_PATH.resolve(),
        PROFILE_PATH.resolve(),
    ]
    relative = [str(path.relative_to(REPO_ROOT)) for path in paths]
    tracked = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "ls-files",
            "--error-unmatch",
            "--",
            *relative,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        raise ProjectionError(
            "Launch requires the campaign sources to be tracked"
        )
    for cached in (False, True):
        command = ["git", "-C", str(REPO_ROOT), "diff", "--quiet"]
        if cached:
            command.append("--cached")
        command.extend(["--", *relative])
        if subprocess.run(command, check=False).returncode != 0:
            raise ProjectionError(
                "Launch requires committed clean campaign sources"
            )
    return {
        "tracked_committed_clean": True,
        "source_count": len(paths),
    }


def dry_run_report(
    *,
    projection_path: Path,
    projection_whole_sha256: str,
    projection: dict[str, Any],
    manifest: dict[str, Any],
    plans: list[dict[str, Any]],
    rendered: list[tuple[str, dict[str, Any]]],
    source_checks: dict[str, Any],
    remote_checks: dict[str, Any] | None,
    launch_source_checks: dict[str, Any] | None,
    loaded_secret_keys: list[str],
    requested_operation: str,
    recovery_binding: dict[str, str] | None,
) -> dict[str, Any]:
    blockers = []
    if remote_checks is None:
        blockers.append("remote artifact verification not requested")
    elif remote_checks.get("all_verified") is not True:
        blockers.append("remote artifact verification failed")
    if launch_source_checks is None:
        blockers.append("committed launch source verification not requested")
    recovery_required = projection["checkpoint_availability"][
        "launch_blocked_until_recovery_evidence"
    ]
    if recovery_required and recovery_binding is None:
        blockers.append(
            "identity-preserving checkpoint recovery evidence required"
        )
    return {
        "schema_version": 1,
        "status": "dry_run_validated_not_launched",
        "campaign_id": projection["campaign_id"],
        "compatibility_manifest_id": manifest["manifest_id"],
        "manifest": {
            "path": str(projection_path.resolve()),
            "whole_file_sha256": projection_whole_sha256,
            "internal_sha256": projection["artifact_integrity"][
                "canonical_payload_sha256"
            ],
        },
        "candidate_ids": projection["candidate_derivation"][
            "candidate_ids"
        ],
        "candidate_count": len(projection["candidates"]),
        "schedule_sha256": projection["schedule"]["schedule_sha256"],
        "allocations": [summary for _, summary in rendered],
        "block_plan_bindings": [
            {
                "allocation_id": plan["allocation_id"],
                "block_plan_sha256": plan["block_plan_sha256"],
                "campaign_id": plan["campaign_id"],
                "execution_projection": plan["execution_projection"],
                "selection_isolation": plan["selection_isolation"],
                "source_bindings_sha256": canonical_sha256(
                    plan["source_bindings"]
                ),
            }
            for plan in plans
        ],
        "source_checks": {
            "campaign_id": projection["campaign_id"],
            "projection_source_bindings": source_checks,
            "checkpoint_recovery_evidence": copy.deepcopy(
                recovery_binding
            ),
            "launch_source_state": launch_source_checks,
            "machinery_reuse": projection["machinery_reuse"],
        },
        "remote_checks": remote_checks,
        "loaded_secret_keys": loaded_secret_keys,
        "secret_values_recorded": False,
        "submission_ready": not blockers,
        "blockers": blockers,
        "feeds_final_selection": False,
        "feeds_reselection": False,
        "selection_time_objectives_replaced": False,
        "selection_isolation": dict(SELECTION_ISOLATION),
        "checkpoint_recovery": {
            "required": recovery_required,
            "resolved": recovery_binding is not None,
            "evidence": copy.deepcopy(recovery_binding),
            "architecture_proxy_used": False,
            "candidate_substitution_used": False,
        },
        "requested_operation": requested_operation,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--write-projection",
        action="store_true",
        help="Rebuild the deterministic checked-in execution projection.",
    )
    mode.add_argument(
        "--launch",
        action="store_true",
        help="Submit six one-node/eight-A100 jobs without waiting.",
    )
    parser.add_argument(
        "--projection",
        type=Path,
        default=DEFAULT_PROJECTION_PATH,
    )
    parser.add_argument(
        "--projection-sha256",
        help="Required exact whole-file digest except with --write-projection.",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=DEFAULT_RUNTIME_DIR,
    )
    parser.add_argument("--verify-remote", action="store_true")
    parser.add_argument(
        "--checkpoint-recovery-evidence",
        type=Path,
        help=(
            "Identity-preserving recovery provenance for the deleted "
            "checkpoint; defaults to the campaign recovery artifact when it "
            "exists."
        ),
    )
    parser.add_argument(
        "--checkpoint-recovery-evidence-sha256",
        help="Exact whole-file SHA256 of --checkpoint-recovery-evidence.",
    )
    parser.add_argument("--acknowledgement", default="")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    projection_path = args.projection.resolve()
    if args.write_projection:
        if args.launch or args.verify_remote:
            raise ProjectionError(
                "Projection generation cannot verify or launch jobs"
            )
        projection = build_projection()
        write_json(projection_path, projection)
        print(json.dumps({
            "status": "projection_written_not_launched",
            "campaign_id": CAMPAIGN_ID,
            "path": str(projection_path),
            "whole_file_sha256": file_sha256(projection_path),
            "internal_sha256": projection["artifact_integrity"][
                "canonical_payload_sha256"
            ],
        }, sort_keys=True))
        return 0
    if args.projection_sha256 is None:
        raise ProjectionError("--projection-sha256 is required")

    projection, whole_file_sha256 = load_projection(
        projection_path,
        args.projection_sha256,
    )
    runtime_dir = args.runtime_dir.resolve()
    if runtime_dir != DEFAULT_RUNTIME_DIR.resolve():
        raise ProjectionError("Runtime directory differs from frozen projection")
    effective_recovery_path = args.checkpoint_recovery_evidence
    if (
        effective_recovery_path is None
        and DEFAULT_RECOVERY_EVIDENCE_PATH.is_file()
    ):
        effective_recovery_path = DEFAULT_RECOVERY_EVIDENCE_PATH
    recovery_arguments = (
        effective_recovery_path,
        args.checkpoint_recovery_evidence_sha256,
    )
    if any(item is not None for item in recovery_arguments) and not all(
        item is not None for item in recovery_arguments
    ):
        if (
            effective_recovery_path == DEFAULT_RECOVERY_EVIDENCE_PATH
            and args.checkpoint_recovery_evidence_sha256 is None
        ):
            args.checkpoint_recovery_evidence_sha256 = file_sha256(
                effective_recovery_path
            )
        else:
            raise ProjectionError(
                "Both checkpoint-recovery evidence arguments are required"
            )
    recovery_evidence = None
    recovery_binding = None
    if effective_recovery_path is not None:
        recovery_evidence, recovery_binding = load_recovery_evidence(
            projection,
            effective_recovery_path.resolve(),
            str(args.checkpoint_recovery_evidence_sha256),
        )
    manifest, plans, rendered = build_plans_and_commands(
        projection,
        whole_file_sha256,
        recovery_evidence,
        recovery_binding,
    )
    source_checks = verify_source_bindings(projection)
    loaded_keys: list[str] = []
    remote_checks = None
    launch_source_checks = None
    if (args.verify_remote or args.launch) and recovery_binding is not None:
        loaded_keys = POST_FRONT.load_env_file(
            Path(manifest["runtime"]["secrets_env_path"])
        )
        remote_checks = POST_FRONT.verify_remote(manifest)
        if remote_checks.get("all_verified") is True:
            launch_source_checks = {
                "base_campaign": POST_FRONT.validate_launch_source_state(
                    manifest
                ),
                "latency_90_campaign": validate_campaign_source_state(
                    projection_path
                ),
            }
    requested_operation = "launch" if args.launch else "dry_run"
    report = dry_run_report(
        projection_path=projection_path,
        projection_whole_sha256=whole_file_sha256,
        projection=projection,
        manifest=manifest,
        plans=plans,
        rendered=rendered,
        source_checks=source_checks,
        remote_checks=remote_checks,
        launch_source_checks=launch_source_checks,
        loaded_secret_keys=loaded_keys,
        requested_operation=requested_operation,
        recovery_binding=recovery_binding,
    )
    runtime_dir.mkdir(parents=True, exist_ok=True)
    report_path = runtime_dir / "dry_run.json"
    POST_FRONT.atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not args.launch:
        return 0
    if args.acknowledgement != EXPECTED_ACKNOWLEDGEMENT:
        raise ProjectionError("Launch acknowledgement does not match")
    if not report["submission_ready"]:
        raise ProjectionError("Launch preflight is not submission-ready")
    submissions = POST_FRONT.submit_all(
        manifest=manifest,
        manifest_file_sha256=whole_file_sha256,
        commands=rendered,
        runtime_dir=runtime_dir,
        source_checks=report["source_checks"],
    )
    print(json.dumps({
        "status": "six_allocations_submitted_without_waiting",
        "campaign_id": CAMPAIGN_ID,
        "submission_count": len(submissions),
        "submissions": submissions,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
