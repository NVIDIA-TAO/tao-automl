#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the sealed 90%-retention DINO archive replay."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SCRIPT_PATH = HERE / "replay_latency_90_policy.py"
ARTIFACT_PATH = HERE / "latency_90_policy" / "archive_replay.v1.json"
SPEC = importlib.util.spec_from_file_location(
    "replay_latency_90_policy",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


def test_replay_script_does_not_encode_expected_candidate_ids():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "seed_271828_rec_18" not in source
    assert "seed_271828_rec_19" not in source
    assert "seed_271828_rec_6" not in source


def test_production_replay_derives_relative_threshold_and_complete_population():
    payload = REPLAY.build_payload()
    resolved = payload["resolved_policy"]
    assert resolved["minimum_accuracy_mAP50"] == pytest.approx(
        0.90 * resolved["accuracy_winner_mAP50"],
        rel=0.0,
        abs=0.0,
    )
    assert resolved["minimum_accuracy_mAP50"] == pytest.approx(
        0.5898724450814929
    )
    assert resolved["feasible_candidate_count"] == 17
    expected_ids = {
        "seed_161803_rec_11",
        "seed_161803_rec_14",
        "seed_271828_rec_1",
        "seed_271828_rec_6",
        "seed_271828_rec_8",
        "seed_271828_rec_12",
        "seed_271828_rec_13",
        "seed_271828_rec_16",
        "seed_271828_rec_18",
        "seed_271828_rec_19",
        "seed_314159_rec_3",
        "seed_314159_rec_5",
        "seed_314159_rec_7",
        "seed_314159_rec_11",
        "seed_314159_rec_12",
        "seed_314159_rec_17",
        "seed_314159_rec_18",
    }
    assert set(resolved["feasible_candidate_ids"]) == expected_ids
    assert all(
        item["mAP50"] >= resolved["minimum_accuracy_mAP50"] - 1e-12
        for item in payload[
            "complete_90_percent_feasible_candidate_table"
        ]
    )


def test_latency_winner_and_cohort_are_entirely_selector_derived():
    payload = REPLAY.build_payload()
    audit = payload["latency_tied_cohort_audit"]
    assert audit["raw_minimum_latency_candidate_id"] == "seed_271828_rec_19"
    assert audit["equivalent_fastest_candidate_ids"] == [
        "seed_271828_rec_19",
        "seed_271828_rec_6",
    ]
    assert audit["selected_latency_candidate_id"] == "seed_271828_rec_19"
    assert payload["selections"]["latency"]["winner_id"] == (
        audit["selected_latency_candidate_id"]
    )
    assert audit["accuracy_tie_break_invoked"] is True
    assert (
        audit["higher_accuracy_resolved_equivalent_fastest_cohort"] is True
    )
    assert (
        audit["fingerprint_or_candidate_id_tie_break_required"] is False
    )
    assert audit["production_selection_reason"] == (
        "Highest-accuracy member of the equivalent-fastest cohort satisfying "
        "the accuracy-winner-relative constraint; deterministic specification "
        "fingerprint and candidate ID resolve remaining ties."
    )
    assert audit["policy_interpretation"] == (
        "Latency mode selected the highest-accuracy member of the "
        "equivalent-fastest cohort satisfying 90% retained accuracy."
    )


def test_matched_scope_is_cohort_only_when_no_outsider_is_plausible():
    scope = REPLAY.build_payload()["latency_tied_cohort_audit"][
        "matched_validation_scope"
    ]
    assert scope["candidate_ids"] == [
        "seed_271828_rec_19",
        "seed_271828_rec_6",
    ]
    assert scope["additional_uncertainty_plausible_candidate_ids"] == []
    assert scope["nearest_excluded_candidate"]["candidate_id"] == (
        "seed_271828_rec_12"
    )
    assert (
        scope["nearest_excluded_candidate"][
            "median_delta_from_raw_minimum_ms"
        ]
        > REPLAY.LATENCY_TOLERANCE_MS
    )


def test_multi_objective_and_accuracy_modes_remain_independent():
    payload = REPLAY.build_payload()
    independence = payload["mode_independence"]
    assert independence[
        "accuracy_selection_unchanged_from_sealed_archive"
    ] is True
    assert independence[
        "multi_objective_selection_unchanged_from_sealed_archive"
    ] is True
    assert independence["multi_objective_min_accuracy"] is None
    assert independence[
        "multi_objective_inherited_latency_threshold"
    ] is False
    assert (
        independence["sealed_accuracy_selection_sha256"]
        == independence["replayed_accuracy_selection_sha256"]
    )
    assert (
        independence["sealed_multi_objective_selection_sha256"]
        == independence["replayed_multi_objective_selection_sha256"]
    )


def test_all_orders_have_identical_complete_selector_output():
    order = REPLAY.build_payload()["archive_order_invariance"]
    assert order["passed"] is True
    assert order["complete_selector_output_identical"] is True
    assert order["ordering_count"] == 3 + len(REPLAY.PERMUTATION_SEEDS)
    assert len({
        value["selector_output_sha256"]
        for value in order["orders"].values()
    }) == 1


def test_frozen_hash_mismatch_is_rejected(tmp_path, monkeypatch):
    changed = tmp_path / "expanded_candidate_table.json"
    changed.write_bytes(REPLAY.CANDIDATE_TABLE_PATH.read_bytes() + b" ")
    monkeypatch.setattr(REPLAY, "CANDIDATE_TABLE_PATH", changed)
    with pytest.raises(REPLAY.ReplayContractError, match="hash mismatch"):
        REPLAY.build_payload()


def test_selection_isolation_flags_are_all_false():
    isolation = REPLAY.build_payload()["selection_isolation"]
    assert isolation == {
        "selector_invoked_on_matched_measurements": False,
        "selection_time_objectives_replaced": False,
        "measurements_feed_selection": False,
        "measurements_feed_reselection": False,
        "algorithm_selected_candidate_overridden": False,
    }
    assert not any(isolation.values())


def test_checked_in_artifact_matches_fresh_replay_and_hash():
    existing = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    REPLAY.verify_payload(existing)
    assert existing == REPLAY.build_payload()
    damaged = copy.deepcopy(existing)
    damaged["policy"]["latency_accuracy_retention"] = 0.91
    with pytest.raises(
        REPLAY.ReplayContractError,
        match="canonical payload hash",
    ):
        REPLAY.verify_payload(damaged)
