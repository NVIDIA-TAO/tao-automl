#!/usr/bin/env python3

"""Seal and validate the rec16 validation-only checkpoint overlay.

The parent four-candidate manifest and its four pinned executables remain
byte-identical.  This module binds one exact-configuration recovery artifact
to the missing historical rec16 checkpoint for latency measurement only.
Neither the frozen archive nor any selection-time objective is changed.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import latency_feasible_matched_manifest_generator as manifest_generator
import latency_feasible_matched_launcher as parent_launcher


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
DEFAULT_PARENT_MANIFEST = HERE / "latency_feasible_matched_manifest.v1.json"
DEFAULT_RECOVERY_EVIDENCE = (
    HERE / "rec16_checkpoint_recovery_evidence.v1.json"
)
DEFAULT_OUTPUT = HERE / "latency_feasible_rec16_checkpoint_overlay.v2.json"

OVERLAY_ID = "dino_latency_feasible_rec16_checkpoint_overlay_20260728_v2"
RECOVERY_EVIDENCE_ID = "dino_rec16_checkpoint_recovery_20260728_v1"
PARENT_MANIFEST_ID = "dino_latency_feasible_matched_20260728_v1"
RECOVERED_CANDIDATE_ID = "seed_271828_rec_16"
EXPECTED_PARENT_FILE_SHA256 = (
    "f83be40f9e00bcd5fc62959bf5a732327353a91ca12ee2efc118325ded2d0db4"
)
EXPECTED_PARENT_INTERNAL_SHA256 = (
    "32c268f11742c26bdc47c61272f73a6d9f651317606e9102e75585bc7d2100e9"
)
EXPECTED_CANDIDATE_SET_SHA256 = (
    "143a2cf63b3b6d700cc2cd124f15a12ef6bdf58b85c2a6af5fdc4bcbfc73bce1"
)
EXPECTED_SCHEDULE_SHA256 = (
    "19cb88f9a061ab1dc46eb7aa1177fe9a93ae148524850d502d99af98089c4f9d"
)
EXPECTED_SELECTION_SNAPSHOT_SHA256 = (
    "bba9d8463db5e889ed8342d468df277c195e61ddd3cc231f009c711384900736"
)
EXPECTED_OBJECTIVE_PROJECTION_SHA256 = (
    "cc05fe80a9c9d78e0038e39c5dd232d8f76ca049beb196a4725ce66674f6d29e"
)
EXPECTED_WINNER_PROJECTION_SHA256 = (
    "25718917435aa71426256a65673a195d948eb1072b81ddf9480392b3b4a72bc6"
)
EXPECTED_HISTORICAL_CHECKPOINT = {
    "epoch": 9,
    "path": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/"
        "results/92d8f699-a780-4229-94ba-3520806d75da/results_dir/"
        "train/model_epoch_009_step_00440.pth"
    ),
    "sha256": (
        "4b5ff50181ff919a2796cdd54027fff92"
        "eb57c908701a34408d29136d5565b4d"
    ),
    "size_bytes": 506_687_042,
}
EXPECTED_RECOVERY_ATTEMPTS = (
    {
        "submission_index": 0,
        "tao_job_id": "7b585a7b-a291-4473-8bda-8e2b542e3982",
        "slurm_job_id": "31002892",
        "submit_time_utc": "2026-07-28T12:06:17Z",
        "node": "batch-block7-00556",
        "checkpoint_sha256": (
            "931bc787eb7b9b1752bd7613558a2e0f"
            "1d26ae7cc5a983d8f4333ef59abbd304"
        ),
    },
    {
        "submission_index": 1,
        "tao_job_id": "1dd7f4ab-843d-4425-8e41-248386ac9a6b",
        "slurm_job_id": "31002901",
        "submit_time_utc": "2026-07-28T12:07:04Z",
        "node": "batch-block7-02873",
        "checkpoint_sha256": (
            "eed8e5f05ec24dd1d62ee4e68ed9c441"
            "94f32c24180764c1e0285f76b0d53a35"
        ),
    },
)
EXPECTED_TRAIN_SPEC_SHA256 = (
    "6f04eab6794cbf8bd707a966ab85b149d7bc24ea4ae238025bb6f3193fca9bf1"
)
EXPECTED_MODEL_SPEC_SHA256 = (
    "bc18216f670d96963ab795be8d6b845f576f4eed17a2482516da91d27eb6248d"
)
EXPECTED_COMMAND_SHA256 = (
    "78174949b50d9a4cf619725a04f844e5f190bf4565716ea1eff2770ec21dd257"
)
EXPECTED_SELECTION_FLAGS = (
    "selector_invoked_on_matched_measurements",
    "selection_time_objectives_replaced",
    "measurements_feed_selection",
    "measurements_feed_reselection",
    "algorithm_selected_candidate_overridden",
)
OVERLAY_TOOL_FILENAMES = (
    "latency_feasible_checkpoint_overlay.py",
    "latency_feasible_matched_launcher_v2.py",
    "latency_feasible_matched_aggregator_v2.py",
)


class OverlayError(ValueError):
    """Raised when recovery or overlay evidence violates the frozen contract."""


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise OverlayError(
            f"{label} mismatch: expected {expected!r}, got {actual!r}"
        )


def require_false_flags(value: dict[str, Any], label: str) -> None:
    if not isinstance(value, dict):
        raise OverlayError(f"{label} must be an object")
    for key in EXPECTED_SELECTION_FLAGS:
        require_equal(value.get(key), False, f"{label} {key}")


def require_absolute_path(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not Path(value).is_absolute()
    ):
        raise OverlayError(f"{label} must be a non-empty absolute path")
    return value


def require_utc_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise OverlayError(f"{label} must be a timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise OverlayError(f"{label} is not ISO-8601") from error
    if parsed.tzinfo is not None and parsed.utcoffset().total_seconds() != 0:
        raise OverlayError(f"{label} must be UTC")
    return value


def objective_projection(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        candidate["candidate_id"]: copy.deepcopy(
            candidate["selection_time_objective_values"]
        )
        for candidate in manifest["candidates"]
    }


def winner_projection(manifest: dict[str, Any]) -> dict[str, str]:
    return {
        mode: selection["winner_id"]
        for mode, selection in manifest["selection_snapshot"][
            "selections"
        ].items()
    }


def parent_binding(
    manifest: dict[str, Any],
    path: Path,
    whole_file_sha256: str,
) -> dict[str, Any]:
    require_equal(
        whole_file_sha256,
        EXPECTED_PARENT_FILE_SHA256,
        "parent manifest whole-file SHA256",
    )
    require_equal(
        manifest.get("manifest_id"),
        PARENT_MANIFEST_ID,
        "parent manifest ID",
    )
    require_equal(
        manifest.get("manifest_sha256"),
        EXPECTED_PARENT_INTERNAL_SHA256,
        "parent manifest internal SHA256",
    )
    require_equal(
        manifest["candidate_derivation"]["candidate_set_sha256"],
        EXPECTED_CANDIDATE_SET_SHA256,
        "parent candidate-set SHA256",
    )
    require_equal(
        manifest["schedule"]["schedule_sha256"],
        EXPECTED_SCHEDULE_SHA256,
        "parent schedule SHA256",
    )
    selection_sha256 = manifest_generator.sha256_value(
        manifest["selection_snapshot"]
    )
    objectives_sha256 = manifest_generator.sha256_value(
        objective_projection(manifest)
    )
    winners_sha256 = manifest_generator.sha256_value(
        winner_projection(manifest)
    )
    require_equal(
        selection_sha256,
        EXPECTED_SELECTION_SNAPSHOT_SHA256,
        "parent selection-snapshot SHA256",
    )
    require_equal(
        objectives_sha256,
        EXPECTED_OBJECTIVE_PROJECTION_SHA256,
        "parent objective-projection SHA256",
    )
    require_equal(
        winners_sha256,
        EXPECTED_WINNER_PROJECTION_SHA256,
        "parent winner-projection SHA256",
    )
    return {
        "path": str(path.resolve()),
        "whole_file_sha256": whole_file_sha256,
        "internal_sha256": manifest["manifest_sha256"],
        "manifest_id": manifest["manifest_id"],
        "candidate_set_sha256": manifest["candidate_derivation"][
            "candidate_set_sha256"
        ],
        "schedule_sha256": manifest["schedule"]["schedule_sha256"],
        "selection_snapshot_sha256": selection_sha256,
        "candidate_audits_sha256": manifest["selection_snapshot"][
            "candidate_audits_sha256"
        ],
        "objective_projection_sha256": objectives_sha256,
        "winner_projection_sha256": winners_sha256,
    }


def _attempt_selection_key(attempt: dict[str, Any]) -> tuple[str, int, str]:
    submit = require_utc_timestamp(
        attempt.get("submit_time_utc"),
        "recovery attempt submit time",
    )
    slurm = str(attempt.get("slurm_job_id", ""))
    if not slurm.isdigit():
        raise OverlayError("recovery SLURM job ID must be numeric")
    tao = attempt.get("tao_job_id")
    if not isinstance(tao, str) or not tao:
        raise OverlayError("recovery TAO job ID is missing")
    return submit, int(slurm), tao


def validate_recovery_evidence(
    evidence: dict[str, Any],
    whole_file_sha256: str,
) -> dict[str, Any]:
    manifest_generator.require_sha256(
        whole_file_sha256,
        "recovery-evidence whole-file SHA256",
    )
    manifest_generator.validate_internal_digest(
        evidence,
        "evidence_sha256",
        "rec16 recovery evidence",
    )
    require_equal(evidence.get("schema_version"), 1, "recovery schema")
    require_equal(
        evidence.get("evidence_id"),
        RECOVERY_EVIDENCE_ID,
        "recovery evidence ID",
    )
    require_equal(
        evidence.get("status"),
        "complete",
        "recovery evidence status",
    )
    require_equal(
        evidence.get("candidate_id"),
        RECOVERED_CANDIDATE_ID,
        "recovery candidate ID",
    )
    require_utc_timestamp(evidence.get("frozen_at_utc"), "recovery freeze time")
    historical = evidence.get("historical_checkpoint")
    if not isinstance(historical, dict):
        raise OverlayError("historical rec16 checkpoint evidence is missing")
    require_equal(
        historical.get("checkpoint"),
        EXPECTED_HISTORICAL_CHECKPOINT,
        "historical rec16 checkpoint",
    )
    require_equal(
        {
            "remote_checkpoint_status": historical.get(
                "remote_checkpoint_status"
            ),
            "historical_identity_preserved": historical.get(
                "historical_identity_preserved"
            ),
            "replacement_of_historical_bytes_permitted": historical.get(
                "replacement_of_historical_bytes_permitted"
            ),
        },
        {
            "remote_checkpoint_status": "missing",
            "historical_identity_preserved": True,
            "replacement_of_historical_bytes_permitted": False,
        },
        "historical rec16 checkpoint status",
    )
    source = evidence.get("source_identity")
    if not isinstance(source, dict):
        raise OverlayError("recovery source identity is missing")
    require_equal(
        source.get("expanded_manifest", {}).get("sha256"),
        manifest_generator.EXPECTED_EXPANDED_MANIFEST_FILE_SHA256,
        "recovery expanded-manifest SHA256",
    )
    require_equal(
        source.get("seed_archive", {}).get("whole_file_sha256"),
        "a42a989ea27940ea9ae481212a75216c7f23f01602b0c260b6750c9fdb709c9e",
        "recovery seed-archive SHA256",
    )
    require_equal(
        source.get("train_spec_sha256"),
        EXPECTED_TRAIN_SPEC_SHA256,
        "recovery train-spec SHA256",
    )
    require_equal(
        source.get("model_spec_sha256"),
        EXPECTED_MODEL_SPEC_SHA256,
        "recovery model-spec SHA256",
    )
    require_equal(
        source.get("command_sha256"),
        EXPECTED_COMMAND_SHA256,
        "recovery command SHA256",
    )
    require_equal(source.get("training_seed"), 1234, "recovery training seed")
    require_equal(
        source.get("candidate_record_sha256"),
        "7b7b7f05beafe5fde34b86e0a3f3e21a48a04e38ddfb46a7ad435e25d2a0760c",
        "recovery candidate-record SHA256",
    )

    attempts = evidence.get("recovery_attempts")
    if not isinstance(attempts, list) or len(attempts) != 2:
        raise OverlayError(
            "recovery evidence must contain exactly two completed "
            "scheduler-assigned attempts"
        )
    if attempts != sorted(attempts, key=_attempt_selection_key):
        raise OverlayError("recovery attempts must be in deterministic key order")
    if len({item.get("tao_job_id") for item in attempts}) != 2 or len(
        {str(item.get("slurm_job_id")) for item in attempts}
    ) != 2:
        raise OverlayError("recovery job identities must be unique")
    for attempt, expected in zip(attempts, EXPECTED_RECOVERY_ATTEMPTS):
        for key in (
            "submission_index",
            "tao_job_id",
            "slurm_job_id",
            "submit_time_utc",
            "node",
        ):
            require_equal(
                attempt.get(key),
                expected[key],
                f"recovery attempt {expected['submission_index']} {key}",
            )
        require_equal(attempt.get("exact_config"), True, "exact recovery config")
        require_equal(attempt.get("state"), "COMPLETED", "recovery SLURM state")
        require_equal(attempt.get("exit_code"), "0:0", "recovery exit code")
        require_equal(
            attempt.get("train_spec_sha256"),
            EXPECTED_TRAIN_SPEC_SHA256,
            "attempt train-spec SHA256",
        )
        require_equal(
            attempt.get("model_spec_sha256"),
            EXPECTED_MODEL_SPEC_SHA256,
            "attempt model-spec SHA256",
        )
        require_equal(
            attempt.get("command_sha256"),
            EXPECTED_COMMAND_SHA256,
            "attempt command SHA256",
        )
        checkpoint = attempt.get("checkpoint")
        if not isinstance(checkpoint, dict):
            raise OverlayError("recovery attempt checkpoint is missing")
        require_absolute_path(
            checkpoint.get("path"),
            "recovery attempt checkpoint path",
        )
        require_equal(
            checkpoint.get("sha256"),
            expected["checkpoint_sha256"],
            "recovery attempt checkpoint SHA256",
        )
        require_equal(
            checkpoint.get("size_bytes"),
            EXPECTED_HISTORICAL_CHECKPOINT["size_bytes"],
            "recovery checkpoint size",
        )
        require_equal(checkpoint.get("epoch"), 9, "recovery checkpoint epoch")
        require_equal(
            attempt.get("historical_checkpoint_sha256_match"),
            False,
            "recovery historical-byte-match flag",
        )
        for key in ("start_time_utc", "end_time_utc"):
            require_utc_timestamp(
                attempt.get(key),
                f"recovery attempt {key}",
            )

    policy = evidence.get("selection_policy")
    if not isinstance(policy, dict):
        raise OverlayError("recovery selection policy is missing")
    require_equal(
        policy.get("policy_key"),
        "earliest_submitted_exact_config_recovery_v1",
        "recovery selection-policy key",
    )
    require_equal(
        {
            "eligible_attempt_predicate": policy.get(
                "eligible_attempt_predicate"
            ),
            "sort_key": policy.get("sort_key"),
            "ascending": policy.get("ascending"),
            "value_independent": policy.get("value_independent"),
            "checkpoint_hash_used": policy.get("checkpoint_hash_used"),
            "checkpoint_size_used": policy.get("checkpoint_size_used"),
            "objective_value_used": policy.get("objective_value_used"),
            "selected_submission_index": policy.get(
                "selected_submission_index"
            ),
        },
        {
            "eligible_attempt_predicate": (
                "exact_config is true and state is COMPLETED and "
                "exit_code is 0:0"
            ),
            "sort_key": [
                "submit_time_utc",
                "numeric_slurm_job_id",
                "tao_job_id",
            ],
            "ascending": True,
            "value_independent": True,
            "checkpoint_hash_used": False,
            "checkpoint_size_used": False,
            "objective_value_used": False,
            "selected_submission_index": 0,
        },
        "recovery selection policy",
    )
    selected = evidence.get("selected_recovery")
    if not isinstance(selected, dict):
        raise OverlayError("selected recovery is missing")
    require_equal(
        selected.get("policy_key"),
        "earliest_submitted_exact_config_recovery_v1",
        "selected recovery policy key",
    )
    first = attempts[0]
    require_equal(
        {
            "submission_index": selected.get("submission_index"),
            "tao_job_id": selected.get("tao_job_id"),
            "slurm_job_id": selected.get("slurm_job_id"),
            "checkpoint": selected.get("checkpoint"),
        },
        {
            "submission_index": first["submission_index"],
            "tao_job_id": first["tao_job_id"],
            "slurm_job_id": first["slurm_job_id"],
            "checkpoint": first["checkpoint"],
        },
        "deterministically selected recovery",
    )
    require_equal(selected.get("validation_only"), True, "validation-only recovery")
    require_equal(
        selected.get("configuration_exact_not_byte_identical"),
        True,
        "configuration-exact recovery identity",
    )
    require_equal(
        selected["checkpoint"]["sha256"]
        == EXPECTED_HISTORICAL_CHECKPOINT["sha256"],
        False,
        "recovery checkpoint must not be mislabeled historical bytes",
    )
    supplementary = evidence.get("supplementary_exact_node_replay")
    if not isinstance(supplementary, dict):
        raise OverlayError("supplementary exact-node replay is missing")
    require_equal(
        {
            "tao_job_id": supplementary.get("tao_job_id"),
            "slurm_job_id": supplementary.get("slurm_job_id"),
            "allowed_node": supplementary.get("expected_node"),
            "eligible_to_displace_selected_recovery": supplementary.get(
                "selected_recovery_can_change"
            ),
            "gates_overlay": supplementary.get("non_gating") is False,
        },
        {
            "tao_job_id": "bc087e0c-e006-4a31-aa7b-228cb7340dbe",
            "slurm_job_id": "31003516",
            "allowed_node": "batch-block7-02877",
            "eligible_to_displace_selected_recovery": False,
            "gates_overlay": False,
        },
        "supplementary exact-node replay",
    )
    require_equal(
        supplementary.get("included_in_selection_policy"),
        False,
        "supplementary selection-policy membership",
    )
    require_equal(
        supplementary.get("state"),
        "PENDING",
        "supplementary exact-node state",
    )
    require_false_flags(evidence.get("selection_isolation"), "recovery isolation")
    return selected


def tool_sources() -> dict[str, Any]:
    sources = {}
    for filename in OVERLAY_TOOL_FILENAMES:
        sources[filename] = manifest_generator.clean_head_source_provenance(
            REPOSITORY,
            HERE / filename,
        )
    return sources


def build_overlay(
    *,
    parent_manifest: dict[str, Any],
    parent_manifest_path: Path,
    parent_manifest_sha256: str,
    recovery_evidence: dict[str, Any],
    recovery_evidence_path: Path,
    recovery_evidence_sha256: str,
    execution_tools: dict[str, Any],
) -> dict[str, Any]:
    selected = validate_recovery_evidence(
        recovery_evidence,
        recovery_evidence_sha256,
    )
    parent = parent_binding(
        parent_manifest,
        parent_manifest_path,
        parent_manifest_sha256,
    )
    rec16 = next(
        (
            item
            for item in parent_manifest["candidates"]
            if item["candidate_id"] == RECOVERED_CANDIDATE_ID
        ),
        None,
    )
    if rec16 is None:
        raise OverlayError("parent manifest lacks frozen rec16 candidate")
    require_equal(
        rec16["checkpoint"],
        {
            "path": EXPECTED_HISTORICAL_CHECKPOINT["path"],
            "sha256": EXPECTED_HISTORICAL_CHECKPOINT["sha256"],
        },
        "parent rec16 checkpoint",
    )
    effective_checkpoint = copy.deepcopy(selected["checkpoint"])
    overlay = {
        "schema_version": 2,
        "overlay_id": OVERLAY_ID,
        "status": "immutable_ready_to_launch",
        "scope": {
            "model_family": "DINO ResNet50",
            "dataset_uri": (
                "s3://nvcf-storage-handling/data/"
                "tao_od_synthetic_full_dino_coco/"
            ),
            "candidate_id": RECOVERED_CANDIDATE_ID,
            "use": "validation_only_latency_measurement_surrogate",
        },
        "parent_manifest": parent,
        "recovery_evidence": {
            "path": str(recovery_evidence_path.resolve()),
            "whole_file_sha256": recovery_evidence_sha256,
            "internal_sha256": recovery_evidence["evidence_sha256"],
            "evidence_id": recovery_evidence["evidence_id"],
        },
        "substitution": {
            "candidate_id": RECOVERED_CANDIDATE_ID,
            "role": "latency_measurement_surrogate_only",
            "checkpoint_origin": "exact_config_retrain",
            "historical_checkpoint": copy.deepcopy(
                EXPECTED_HISTORICAL_CHECKPOINT
            ),
            "effective_checkpoint": effective_checkpoint,
            "selected_recovery_submission_index": selected[
                "submission_index"
            ],
            "selected_recovery_tao_job_id": selected["tao_job_id"],
            "selected_recovery_slurm_job_id": selected["slurm_job_id"],
            "historical_byte_match": False,
            "accuracy_evidence_replaced": False,
            "selection_time_latency_replaced": False,
        },
        "invariants": {
            "substitution_count": 1,
            "candidate_set_changed": False,
            "candidate_order_changed": False,
            "schedule_changed": False,
            "selection_snapshot_changed": False,
            "candidate_audits_changed": False,
            "selection_time_objectives_changed": False,
            "winner_identities_changed": False,
            "resolved_model_spec_changed": False,
            "frozen_archive_mutated": False,
            "original_checkpoint_still_recorded": True,
            "recovered_checkpoint_is_historical_artifact": False,
        },
        "selection_isolation": {
            "selector_invoked_on_matched_measurements": False,
            "selection_time_objectives_replaced": False,
            "measurements_feed_selection": False,
            "measurements_feed_reselection": False,
            "algorithm_selected_candidate_overridden": False,
        },
        "execution_tools": copy.deepcopy(execution_tools),
    }
    overlay["overlay_sha256"] = manifest_generator.sha256_value(overlay)
    return overlay


def validate_overlay(
    overlay: dict[str, Any],
    whole_file_sha256: str,
    parent_manifest: dict[str, Any],
    parent_manifest_path: Path,
    parent_manifest_sha256: str,
) -> None:
    manifest_generator.require_sha256(
        whole_file_sha256,
        "checkpoint-overlay whole-file SHA256",
    )
    manifest_generator.validate_internal_digest(
        overlay,
        "overlay_sha256",
        "rec16 checkpoint overlay",
    )
    require_equal(overlay.get("schema_version"), 2, "overlay schema")
    require_equal(overlay.get("overlay_id"), OVERLAY_ID, "overlay ID")
    require_equal(
        overlay.get("status"),
        "immutable_ready_to_launch",
        "overlay status",
    )
    require_equal(
        overlay.get("parent_manifest"),
        parent_binding(
            parent_manifest,
            parent_manifest_path,
            parent_manifest_sha256,
        ),
        "overlay parent binding",
    )
    recovery_source = overlay.get("recovery_evidence")
    if not isinstance(recovery_source, dict):
        raise OverlayError("overlay recovery source is missing")
    recovery_path = Path(
        require_absolute_path(
            recovery_source.get("path"),
            "overlay recovery-evidence path",
        )
    ).resolve()
    recovery_sha256 = manifest_generator.require_sha256(
        recovery_source.get("whole_file_sha256"),
        "overlay recovery-evidence SHA256",
    )
    recovery, actual = manifest_generator.load_exact_json(
        recovery_path,
        recovery_sha256,
        "rec16 recovery evidence",
    )
    selected = validate_recovery_evidence(recovery, actual)
    require_equal(
        recovery_source,
        {
            "path": str(recovery_path),
            "whole_file_sha256": actual,
            "internal_sha256": recovery["evidence_sha256"],
            "evidence_id": recovery["evidence_id"],
        },
        "overlay recovery-evidence binding",
    )
    require_equal(
        overlay.get("substitution"),
        {
            "candidate_id": RECOVERED_CANDIDATE_ID,
            "role": "latency_measurement_surrogate_only",
            "checkpoint_origin": "exact_config_retrain",
            "historical_checkpoint": copy.deepcopy(
                EXPECTED_HISTORICAL_CHECKPOINT
            ),
            "effective_checkpoint": copy.deepcopy(selected["checkpoint"]),
            "selected_recovery_submission_index": selected[
                "submission_index"
            ],
            "selected_recovery_tao_job_id": selected["tao_job_id"],
            "selected_recovery_slurm_job_id": selected["slurm_job_id"],
            "historical_byte_match": False,
            "accuracy_evidence_replaced": False,
            "selection_time_latency_replaced": False,
        },
        "overlay substitution",
    )
    require_equal(
        overlay.get("invariants"),
        {
            "substitution_count": 1,
            "candidate_set_changed": False,
            "candidate_order_changed": False,
            "schedule_changed": False,
            "selection_snapshot_changed": False,
            "candidate_audits_changed": False,
            "selection_time_objectives_changed": False,
            "winner_identities_changed": False,
            "resolved_model_spec_changed": False,
            "frozen_archive_mutated": False,
            "original_checkpoint_still_recorded": True,
            "recovered_checkpoint_is_historical_artifact": False,
        },
        "overlay invariants",
    )
    require_false_flags(overlay.get("selection_isolation"), "overlay isolation")
    expected_tools = overlay.get("execution_tools")
    if not isinstance(expected_tools, dict):
        raise OverlayError("overlay execution-tool evidence is missing")
    require_equal(
        set(expected_tools),
        set(OVERLAY_TOOL_FILENAMES),
        "overlay execution-tool set",
    )
    require_equal(tool_sources(), expected_tools, "overlay execution tools")


def load_overlay(
    path: Path,
    supplied_sha256: str,
    parent_manifest: dict[str, Any],
    parent_manifest_path: Path,
    parent_manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    overlay, actual = manifest_generator.load_exact_json(
        path.resolve(),
        supplied_sha256,
        "rec16 checkpoint overlay",
    )
    validate_overlay(
        overlay,
        actual,
        parent_manifest,
        parent_manifest_path,
        parent_manifest_sha256,
    )
    return overlay, actual


def execution_manifest(
    parent_manifest: dict[str, Any],
    overlay: dict[str, Any],
) -> dict[str, Any]:
    """Return a measurement-only projection with one checkpoint substitution."""

    effective = copy.deepcopy(parent_manifest)
    original_candidates = copy.deepcopy(parent_manifest["candidates"])
    candidates = effective["candidates"]
    matching = [
        item
        for item in candidates
        if item["candidate_id"] == RECOVERED_CANDIDATE_ID
    ]
    if len(matching) != 1:
        raise OverlayError("execution projection requires exactly one rec16")
    substitution = overlay["substitution"]
    require_equal(
        matching[0]["checkpoint"],
        {
            "path": substitution["historical_checkpoint"]["path"],
            "sha256": substitution["historical_checkpoint"]["sha256"],
        },
        "execution projection historical checkpoint",
    )
    matching[0]["checkpoint"] = {
        "path": substitution["effective_checkpoint"]["path"],
        "sha256": substitution["effective_checkpoint"]["sha256"],
    }
    expected = copy.deepcopy(original_candidates)
    expected_rec16 = next(
        item
        for item in expected
        if item["candidate_id"] == RECOVERED_CANDIDATE_ID
    )
    expected_rec16["checkpoint"] = copy.deepcopy(matching[0]["checkpoint"])
    require_equal(candidates, expected, "single-checkpoint execution projection")
    for key in (
        "candidate_derivation",
        "selection_snapshot",
        "schedule",
        "latency_protocol",
        "paired_analysis",
        "selection_isolation",
    ):
        require_equal(
            effective[key],
            parent_manifest[key],
            f"execution projection {key}",
        )
    return effective


def augment_block_plan(
    plan: dict[str, Any],
    overlay: dict[str, Any],
    overlay_whole_file_sha256: str,
) -> dict[str, Any]:
    """Bind original/effective checkpoint identity into a v1-compatible plan."""

    augmented = copy.deepcopy(plan)
    claimed = augmented.pop("block_plan_sha256", None)
    manifest_generator.require_sha256(claimed, "base block-plan SHA256")
    substitution = overlay["substitution"]
    matches = [
        item
        for item in augmented["candidates"]
        if item["candidate_id"] == RECOVERED_CANDIDATE_ID
    ]
    if len(matches) != 1:
        raise OverlayError("block plan must contain exactly one rec16")
    candidate = matches[0]
    require_equal(
        {
            "path": candidate["checkpoint_path"],
            "sha256": candidate["checkpoint_sha256"],
        },
        {
            "path": substitution["effective_checkpoint"]["path"],
            "sha256": substitution["effective_checkpoint"]["sha256"],
        },
        "effective block-plan checkpoint",
    )
    candidate["checkpoint_overlay"] = {
        "role": substitution["role"],
        "checkpoint_origin": substitution["checkpoint_origin"],
        "historical_checkpoint": copy.deepcopy(
            substitution["historical_checkpoint"]
        ),
        "effective_checkpoint": copy.deepcopy(
            substitution["effective_checkpoint"]
        ),
        "overlay_id": overlay["overlay_id"],
        "overlay_whole_file_sha256": overlay_whole_file_sha256,
        "overlay_internal_sha256": overlay["overlay_sha256"],
    }
    augmented["checkpoint_overlay"] = {
        "overlay_id": overlay["overlay_id"],
        "whole_file_sha256": overlay_whole_file_sha256,
        "internal_sha256": overlay["overlay_sha256"],
        "candidate_id": RECOVERED_CANDIDATE_ID,
        "substitution_count": 1,
        "selection_isolation": copy.deepcopy(overlay["selection_isolation"]),
    }
    augmented["block_plan_sha256"] = manifest_generator.sha256_value(augmented)
    return augmented


def overlay_source_checks(
    overlay: dict[str, Any],
    overlay_path: Path,
    overlay_whole_file_sha256: str,
) -> dict[str, Any]:
    return {
        "checkpoint_overlay": {
            "path": str(overlay_path.resolve()),
            "whole_file_sha256": overlay_whole_file_sha256,
            "internal_sha256": overlay["overlay_sha256"],
            "overlay_id": overlay["overlay_id"],
        },
        "checkpoint_recovery_evidence": copy.deepcopy(
            overlay["recovery_evidence"]
        ),
        "checkpoint_overlay_execution_tools": {
            name: source["sha256"]
            for name, source in sorted(overlay["execution_tools"].items())
        },
        "effective_checkpoint": copy.deepcopy(
            overlay["substitution"]["effective_checkpoint"]
        ),
        "historical_checkpoint": copy.deepcopy(
            overlay["substitution"]["historical_checkpoint"]
        ),
        "historical_byte_match": False,
        "measurement_role": "latency_measurement_surrogate_only",
        "selection_isolation": copy.deepcopy(overlay["selection_isolation"]),
    }


def write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    except FileExistsError as error:
        raise OverlayError(f"refusing to overwrite immutable overlay: {path}") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, default=DEFAULT_PARENT_MANIFEST)
    parser.add_argument("--parent-manifest-sha256", required=True)
    parser.add_argument(
        "--recovery-evidence",
        type=Path,
        default=DEFAULT_RECOVERY_EVIDENCE,
    )
    parser.add_argument("--recovery-evidence-sha256", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parent_path = args.parent_manifest.resolve()
    parent, parent_sha256 = parent_launcher.load_manifest(
        parent_path,
        args.parent_manifest_sha256,
    )
    recovery_path = args.recovery_evidence.resolve()
    recovery, recovery_sha256 = manifest_generator.load_exact_json(
        recovery_path,
        args.recovery_evidence_sha256,
        "rec16 recovery evidence",
    )
    overlay = build_overlay(
        parent_manifest=parent,
        parent_manifest_path=parent_path,
        parent_manifest_sha256=parent_sha256,
        recovery_evidence=recovery,
        recovery_evidence_path=recovery_path,
        recovery_evidence_sha256=recovery_sha256,
        execution_tools=tool_sources(),
    )
    write_new(args.output.resolve(), overlay)
    print(
        json.dumps(
            {
                "status": "overlay_created_not_launched",
                "path": str(args.output.resolve()),
                "whole_file_sha256": manifest_generator.sha256_file(
                    args.output.resolve()
                ),
                "internal_sha256": overlay["overlay_sha256"],
                "candidate_id": RECOVERED_CANDIDATE_ID,
                "selection_isolation": overlay["selection_isolation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
