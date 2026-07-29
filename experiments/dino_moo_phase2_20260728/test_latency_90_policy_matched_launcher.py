#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the 90%-policy matched execution projection."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "latency_90_policy_matched_launcher.py"
PROJECTION_PATH = (
    HERE / "latency_90_policy" / "matched" / "execution_projection.v1.json"
)
SPEC = importlib.util.spec_from_file_location(
    "latency_90_policy_matched_launcher",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MATCHED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATCHED)

BLOCK_SPEC = importlib.util.spec_from_file_location(
    "post_front_matched_block_runner_for_90_policy_test",
    HERE / "post_front_matched_block_runner.py",
)
assert BLOCK_SPEC is not None and BLOCK_SPEC.loader is not None
BLOCK = importlib.util.module_from_spec(BLOCK_SPEC)
BLOCK_SPEC.loader.exec_module(BLOCK)


def checked_in_projection() -> dict:
    value = json.loads(PROJECTION_PATH.read_text(encoding="utf-8"))
    MATCHED.validate_internal_artifact(
        value,
        label="checked-in execution projection",
    )
    return value


def test_candidate_ids_are_derived_only_from_replay_scope():
    replay, profile, table, _ = MATCHED.load_sources()
    candidates = MATCHED.derive_candidates(replay, profile, table)
    replay_scope = replay["latency_tied_cohort_audit"][
        "matched_validation_scope"
    ]["candidate_ids"]
    assert [item["candidate_id"] for item in candidates] == sorted(
        replay_scope
    )
    assert [item["candidate_id"] for item in candidates] == [
        "seed_271828_rec_19",
        "seed_271828_rec_6",
    ]
    rows = {row["candidate_id"]: row for row in table["rows"]}
    for candidate in candidates:
        row = rows[candidate["candidate_id"]]
        assert candidate["candidate_table_record_sha256"] == (
            MATCHED.canonical_sha256(row)
        )
        assert candidate["checkpoint"]["path"] == row["checkpoint"]["path"]
        assert candidate["checkpoint"]["sha256"] == row["checkpoint"]["sha256"]


def test_missing_replay_scope_record_is_rejected():
    replay, profile, table, _ = MATCHED.load_sources()
    damaged = copy.deepcopy(table)
    replay_ids = set(replay["latency_tied_cohort_audit"][
        "matched_validation_scope"
    ]["candidate_ids"])
    damaged["rows"] = [
        row
        for row in damaged["rows"]
        if row["candidate_id"] not in replay_ids
    ]
    with pytest.raises(
        MATCHED.ProjectionError,
        match="must contain 60 rows",
    ):
        MATCHED.derive_candidates(replay, profile, damaged)


def test_six_allocation_schedule_balances_position_and_adjacency():
    projection = MATCHED.build_projection()
    schedule = projection["schedule"]
    candidate_ids = projection["candidate_derivation"]["candidate_ids"]
    assert len(schedule["allocations"]) == 6
    assert all(
        sorted(item["candidate_order"]) == candidate_ids
        for item in schedule["allocations"]
    )
    assert schedule["audit"]["position_counts"] == {
        candidate_id: {"0": 3, "1": 3}
        for candidate_id in candidate_ids
    }
    assert schedule["audit"]["ordered_immediate_adjacency_counts"] == [
        {
            "first_candidate_id": candidate_ids[0],
            "second_candidate_id": candidate_ids[1],
            "count": 3,
        },
        {
            "first_candidate_id": candidate_ids[1],
            "second_candidate_id": candidate_ids[0],
            "count": 3,
        },
    ]


def test_checked_in_projection_exactly_reconstructs_and_binds_sources():
    existing = checked_in_projection()
    assert existing == MATCHED.build_projection()
    assert existing["campaign_id"] != (
        existing["compatibility_contract"]["manifest_id"]
    )
    assert existing["selection_isolation"] == MATCHED.SELECTION_ISOLATION
    assert not any(existing["selection_isolation"].values())
    availability = existing["checkpoint_availability"]
    assert availability[
        "identity_preserving_recovery_required_candidate_ids"
    ] == ["seed_271828_rec_6"]
    assert availability["launch_blocked_until_recovery_evidence"] is True
    assert availability["requirements"][0][
        "architecture_proxy_permitted"
    ] is False
    for required in (
        "archive_replay",
        "policy_profile",
        "expanded_candidate_table",
        "execution_projection_launcher",
        "post_front_matched_launcher",
        "post_front_matched_block_runner",
        "dino_latency_benchmark",
        "latency_stats",
    ):
        assert required in existing["source_bindings"]


def test_every_plan_passes_existing_block_runner_and_binds_campaign():
    projection = checked_in_projection()
    whole_sha256 = MATCHED.file_sha256(PROJECTION_PATH)
    manifest, plans, rendered = MATCHED.build_plans_and_commands(
        projection,
        whole_sha256,
    )
    assert manifest["manifest_id"] == MATCHED.COMPATIBILITY_MANIFEST_ID
    assert len(plans) == len(rendered) == 6
    source_sha256 = MATCHED.canonical_sha256(
        projection["source_bindings"]
    )
    for plan, (_, summary) in zip(plans, rendered):
        BLOCK.validate_plan(plan)
        assert plan["campaign_id"] == MATCHED.CAMPAIGN_ID
        assert plan["selection_isolation"] == MATCHED.SELECTION_ISOLATION
        assert not any(plan["selection_isolation"].values())
        assert MATCHED.canonical_sha256(plan["source_bindings"]) == (
            source_sha256
        )
        assert plan["execution_projection"]["whole_file_sha256"] == (
            whole_sha256
        )
        assert summary["block_plan_sha256"] == plan["block_plan_sha256"]


def test_rendering_reuses_post_front_staging_and_sqsh_contract():
    projection = checked_in_projection()
    whole_sha256 = MATCHED.file_sha256(PROJECTION_PATH)
    manifest, _, rendered = MATCHED.build_plans_and_commands(
        projection,
        whole_sha256,
    )
    assert manifest["runtime"]["image_is_prebuilt_sqsh"] is True
    assert manifest["runtime"]["sdk_sqsh_conversion_enabled"] is False
    assert manifest["runtime"]["sqsh_path"].endswith(".sqsh")
    assert all(
        "post_front_matched_block_runner.py" in command
        and "$TAO_RESULTS_ROOT/$TAO_JOB_ID" in command
        for command, _ in rendered
    )
    assert manifest["runtime"]["local_runtime_path"] == str(
        MATCHED.DEFAULT_RUNTIME_DIR.resolve()
    )


def test_dry_run_report_remains_validation_only():
    projection = checked_in_projection()
    whole_sha256 = MATCHED.file_sha256(PROJECTION_PATH)
    manifest, plans, rendered = MATCHED.build_plans_and_commands(
        projection,
        whole_sha256,
    )
    report = MATCHED.dry_run_report(
        projection_path=PROJECTION_PATH,
        projection_whole_sha256=whole_sha256,
        projection=projection,
        manifest=manifest,
        plans=plans,
        rendered=rendered,
        source_checks=MATCHED.verify_source_bindings(projection),
        remote_checks=None,
        launch_source_checks=None,
        loaded_secret_keys=[],
        requested_operation="dry_run",
        recovery_binding=None,
    )
    assert report["submission_ready"] is False
    assert report["selection_isolation"] == MATCHED.SELECTION_ISOLATION
    assert report["requested_operation"] == "dry_run"
    assert report["secret_values_recorded"] is False
    assert len(report["allocations"]) == 6
    assert report["checkpoint_recovery"]["required"] is True
    assert report["checkpoint_recovery"]["resolved"] is False
    assert (
        "identity-preserving checkpoint recovery evidence required"
        in report["blockers"]
    )


def test_wrapper_does_not_encode_candidate_ids():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "seed_271828_rec_19" not in source
    assert "seed_271828_rec_6" not in source


def test_wrong_projection_digest_is_rejected():
    with pytest.raises(
        MATCHED.ProjectionError,
        match="whole-file SHA256",
    ):
        MATCHED.load_projection(PROJECTION_PATH, "0" * 64)


def test_recovery_evidence_rejects_architecture_proxy(tmp_path):
    projection = checked_in_projection()
    requirement = projection["checkpoint_availability"]["requirements"][0]
    core = {
        "schema_version": 1,
        "evidence_id": MATCHED.RECOVERY_EVIDENCE_ID,
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
        "recovered_checkpoint": {
            "path": "/absolute/recovered/checkpoint.pth",
            "sha256": "a" * 64,
        },
        "exact_candidate_configuration_preserved": True,
        "architecture_proxy_used": True,
        "manual_candidate_substitution_used": False,
        "result_driven_parameter_change_used": False,
        "measurements_feed_selection": False,
        "measurements_feed_reselection": False,
        "algorithm_selected_candidate_overridden": False,
    }
    artifact = {
        **core,
        "artifact_integrity": {
            "canonical_payload_sha256": MATCHED.canonical_sha256(core),
            "hash_algorithm": "sha256",
            "hash_excludes": ["artifact_integrity"],
        },
    }
    path = tmp_path / "recovery.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(
        MATCHED.ProjectionError,
        match="architecture_proxy_used",
    ):
        MATCHED.load_recovery_evidence(
            projection,
            path,
            MATCHED.file_sha256(path),
        )
