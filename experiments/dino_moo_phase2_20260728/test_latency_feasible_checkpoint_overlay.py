"""Contracts for the validation-only rec16 checkpoint overlay."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import freeze_rec16_checkpoint_recovery as freezer  # noqa: E402
import latency_feasible_checkpoint_overlay as overlay_module  # noqa: E402
import latency_feasible_matched_launcher as parent_launcher  # noqa: E402
from test_freeze_rec16_checkpoint_recovery import FakeProbe  # noqa: E402


def recovery_evidence() -> dict:
    return freezer.build_evidence(
        FakeProbe(),
        freezer.validate_local_source_identity(),
        frozen_at_utc="2026-07-28T19:30:00Z",
    )


def parent_manifest() -> tuple[dict, str]:
    path = overlay_module.DEFAULT_PARENT_MANIFEST
    return parent_launcher.load_manifest(
        path,
        overlay_module.EXPECTED_PARENT_FILE_SHA256,
    )


def fake_tool_sources() -> dict:
    return {
        name: {
            "path": str(HERE / name),
            "relative_path": (
                "experiments/dino_moo_phase2_20260728/" + name
            ),
            "sha256": "a" * 64,
            "git_blob": "b" * 40,
            "head_git_blob": "b" * 40,
            "tracked": True,
            "committed": True,
            "clean_against_head": True,
        }
        for name in overlay_module.OVERLAY_TOOL_FILENAMES
    }


def built_overlay() -> tuple[dict, dict]:
    parent, parent_sha256 = parent_manifest()
    evidence = recovery_evidence()
    overlay = overlay_module.build_overlay(
        parent_manifest=parent,
        parent_manifest_path=overlay_module.DEFAULT_PARENT_MANIFEST,
        parent_manifest_sha256=parent_sha256,
        recovery_evidence=evidence,
        recovery_evidence_path=overlay_module.DEFAULT_RECOVERY_EVIDENCE,
        recovery_evidence_sha256="c" * 64,
        execution_tools=fake_tool_sources(),
    )
    return parent, overlay


def rehash_evidence(evidence: dict) -> None:
    unhashed = copy.deepcopy(evidence)
    unhashed.pop("evidence_sha256", None)
    evidence["evidence_sha256"] = overlay_module.manifest_generator.sha256_value(
        unhashed
    )


def test_parent_v1_sources_and_manifest_remain_byte_identical() -> None:
    parent, parent_sha256 = parent_manifest()

    assert parent_sha256 == overlay_module.EXPECTED_PARENT_FILE_SHA256
    assert (
        parent["manifest_sha256"]
        == overlay_module.EXPECTED_PARENT_INTERNAL_SHA256
    )
    expected_source_hashes = {
        "latency_feasible_matched_manifest_generator.py": (
            "3c83f3ea4644efefb9a425facbda358753de9abf78b293405c60799c60abdcb6"
        ),
        "latency_feasible_matched_launcher.py": (
            "b6a9f13476e9287d7b7040f8d289e8210706cedc6746429ba1e89410ed9b0025"
        ),
        "latency_feasible_matched_block_runner.py": (
            "f2ef5f7125a55db408ba27f67d5e81d279c669a24206038c78748f2676fc14bb"
        ),
        "latency_feasible_matched_aggregator.py": (
            "254b887d829f7c853510cf34707e782cbbeeea9049777482662f8c9fd2d578fa"
        ),
    }
    assert {
        name: source["sha256"]
        for name, source in parent["source_artifacts"][
            "post_front_tools"
        ].items()
    } == expected_source_hashes


def test_execution_projection_changes_only_rec16_checkpoint() -> None:
    parent, overlay = built_overlay()
    effective = overlay_module.execution_manifest(parent, overlay)
    expected = copy.deepcopy(parent)
    rec16 = next(
        item
        for item in expected["candidates"]
        if item["candidate_id"] == overlay_module.RECOVERED_CANDIDATE_ID
    )
    rec16["checkpoint"] = {
        "path": overlay["substitution"]["effective_checkpoint"]["path"],
        "sha256": overlay["substitution"]["effective_checkpoint"]["sha256"],
    }

    assert effective == expected
    assert effective["selection_snapshot"] == parent["selection_snapshot"]
    assert effective["schedule"] == parent["schedule"]
    assert (
        overlay_module.objective_projection(effective)
        == overlay_module.objective_projection(parent)
    )
    assert (
        overlay_module.winner_projection(effective)
        == overlay_module.winner_projection(parent)
    )


def test_recovery_attempt_order_and_selected_attempt_are_fail_closed() -> None:
    evidence = recovery_evidence()
    evidence["recovery_attempts"].reverse()
    rehash_evidence(evidence)
    with pytest.raises(
        overlay_module.OverlayError,
        match="deterministic key order",
    ):
        overlay_module.validate_recovery_evidence(evidence, "d" * 64)

    evidence = recovery_evidence()
    evidence["selected_recovery"] = copy.deepcopy(
        evidence["selected_recovery"]
    )
    evidence["selected_recovery"]["tao_job_id"] = (
        evidence["recovery_attempts"][1]["tao_job_id"]
    )
    rehash_evidence(evidence)
    with pytest.raises(
        overlay_module.OverlayError,
        match="deterministically selected recovery",
    ):
        overlay_module.validate_recovery_evidence(evidence, "d" * 64)


def test_recovery_selection_cannot_consult_hash_objective_or_latency() -> None:
    evidence = recovery_evidence()
    evidence["selection_policy"]["checkpoint_hash_used"] = True
    rehash_evidence(evidence)
    with pytest.raises(
        overlay_module.OverlayError,
        match="recovery selection policy",
    ):
        overlay_module.validate_recovery_evidence(evidence, "d" * 64)

    evidence = recovery_evidence()
    evidence["selection_isolation"]["measurements_feed_selection"] = True
    rehash_evidence(evidence)
    with pytest.raises(
        overlay_module.OverlayError,
        match="recovery isolation measurements_feed_selection",
    ):
        overlay_module.validate_recovery_evidence(evidence, "d" * 64)


def test_parent_archive_objective_or_winner_drift_is_rejected() -> None:
    parent, parent_sha256 = parent_manifest()
    changed = copy.deepcopy(parent)
    changed["selection_snapshot"]["selections"]["latency"]["winner_id"] = (
        "seed_271828_rec_16"
    )
    with pytest.raises(
        overlay_module.OverlayError,
        match="selection-snapshot SHA256",
    ):
        overlay_module.parent_binding(
            changed,
            overlay_module.DEFAULT_PARENT_MANIFEST,
            parent_sha256,
        )

    changed = copy.deepcopy(parent)
    changed["candidates"][0]["selection_time_objective_values"][
        "latency_ms"
    ] += 1
    with pytest.raises(
        overlay_module.OverlayError,
        match="objective-projection SHA256",
    ):
        overlay_module.parent_binding(
            changed,
            overlay_module.DEFAULT_PARENT_MANIFEST,
            parent_sha256,
        )


def test_block_plan_binds_original_and_effective_checkpoint() -> None:
    parent, overlay = built_overlay()
    effective = overlay_module.execution_manifest(parent, overlay)
    allocation = parent["schedule"]["allocations"][0]
    configs = {
        item["candidate_id"]: item["candidate_id"].encode()
        for item in effective["candidates"]
    }
    plan = parent_launcher.build_block_plan(
        effective,
        overlay_module.EXPECTED_PARENT_FILE_SHA256,
        allocation,
        configs,
    )
    augmented = overlay_module.augment_block_plan(
        plan,
        overlay,
        "e" * 64,
    )
    rec16 = next(
        item
        for item in augmented["candidates"]
        if item["candidate_id"] == overlay_module.RECOVERED_CANDIDATE_ID
    )

    assert rec16["checkpoint_overlay"]["historical_checkpoint"] == (
        overlay_module.EXPECTED_HISTORICAL_CHECKPOINT
    )
    assert rec16["checkpoint_overlay"]["effective_checkpoint"] == (
        overlay["substitution"]["effective_checkpoint"]
    )
    unhashed = copy.deepcopy(augmented)
    claimed = unhashed.pop("block_plan_sha256")
    assert overlay_module.manifest_generator.sha256_value(unhashed) == claimed


def test_no_fallback_to_second_attempt_if_earliest_is_invalid() -> None:
    evidence = recovery_evidence()
    evidence["recovery_attempts"][0]["state"] = "FAILED"
    rehash_evidence(evidence)
    with pytest.raises(
        overlay_module.OverlayError,
        match="recovery SLURM state",
    ):
        overlay_module.validate_recovery_evidence(evidence, "f" * 64)


def test_overlay_has_exact_five_false_selection_isolation_flags() -> None:
    _, overlay = built_overlay()

    assert set(overlay["selection_isolation"]) == set(
        overlay_module.EXPECTED_SELECTION_FLAGS
    )
    assert set(overlay["selection_isolation"].values()) == {False}
    assert overlay["invariants"]["substitution_count"] == 1
    assert overlay["invariants"]["winner_identities_changed"] is False
