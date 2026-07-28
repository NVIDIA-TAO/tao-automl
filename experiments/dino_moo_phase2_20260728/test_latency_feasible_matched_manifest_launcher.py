"""Focused contracts for the frozen latency-feasible matched harness."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for path in (ROOT / "src", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import latency_feasible_matched_launcher as launcher  # noqa: E402
import latency_feasible_matched_manifest_generator as generator  # noqa: E402


def frozen_context() -> dict:
    expanded_path = HERE / "expanded_search_manifest.v2.json"
    combined_path = (
        HERE / "runtime" / "expanded_search_v2"
        / "expanded_combined_selection.json"
    )
    table_path = (
        HERE / "runtime" / "expanded_search_v2"
        / "expanded_candidate_table.json"
    )
    integrity_path = (
        HERE / "runtime" / "expanded_search_v2"
        / "expanded_integrity_audit.json"
    )
    expanded = generator.load_json(expanded_path)
    combined = generator.load_json(combined_path)
    table = generator.load_json(table_path)
    integrity = generator.load_json(integrity_path)
    replay = generator.validate_completed_archive(
        combined,
        table,
        expanded_manifest=expanded,
        expanded_manifest_path=expanded_path,
        expanded_manifest_sha256=generator.sha256_file(expanded_path),
        integrity=integrity,
    )
    candidates = generator.derive_candidate_records(
        combined,
        table,
        replay,
    )
    old_manifest = generator.load_json(
        HERE / "post_front_matched_manifest.v1.json"
    )
    protocol_erratum = generator.load_json(
        HERE / "phase2_protocol_erratum.v1.json"
    )
    sensitivity = generator.load_json(
        Path(
            old_manifest["source_artifacts"]["sensitivity_manifest"]["path"]
        )
    )
    manifest = generator.build_manifest(
        expanded_manifest=expanded,
        candidates=candidates,
        sources=old_manifest["source_artifacts"],
        sensitivity_manifest=sensitivity,
        combined=combined,
        selector_replay=replay,
        protocol_erratum=protocol_erratum,
        archive_snapshot=old_manifest["expanded_archive_snapshot"],
    )
    return {
        "combined": combined,
        "replay": replay,
        "candidates": candidates,
        "manifest": manifest,
    }


def test_frozen_replay_derives_exact_complete_feasible_cohort() -> None:
    context = frozen_context()
    candidates = context["candidates"]
    candidate_ids = [item["candidate_id"] for item in candidates]

    assert candidate_ids == list(generator.EXPECTED_FEASIBLE_CANDIDATE_IDS)
    assert (
        generator.sha256_value(candidate_ids)
        == generator.EXPECTED_FEASIBLE_CANDIDATE_SET_SHA256
    )
    assert [item["global_pareto_rank"] for item in candidates] == [1, 1, 0, 1]
    assert [item["global_dominated_by"] for item in candidates] == [
        ["seed_271828_rec_18"],
        ["seed_271828_rec_18"],
        [],
        ["seed_271828_rec_18"],
    ]
    assert all(item["latency_accuracy_feasible"] for item in candidates)
    assert (
        context["combined"]["selections"]["latency"][
            "latency_tied_candidate_ids"
        ]
        == candidate_ids
    )


def test_feasible_derivation_rejects_policy_or_population_drift() -> None:
    context = frozen_context()
    analysis = copy.deepcopy(context["replay"]["analysis"])
    next(
        item
        for item in analysis["candidates"]
        if item["candidate_id"] == "seed_271828_rec_16"
    )["latency_accuracy_feasible"] = False

    with pytest.raises(
        generator.ContractError,
        match="frozen 98%-feasible candidate IDs",
    ):
        generator.latency_feasible_audits(analysis)

    analysis = copy.deepcopy(context["replay"]["analysis"])
    analysis["algorithm"]["configuration"][
        "latency_accuracy_retention"
    ]["value"] = 0.95
    with pytest.raises(
        generator.ContractError,
        match="accuracy-retention configuration",
    ):
        generator.latency_feasible_audits(analysis)


def test_manifest_schedule_and_launcher_contract_are_isolated() -> None:
    manifest = frozen_context()["manifest"]
    launcher.validate_manifest_contract(manifest)

    assert manifest["manifest_id"] == (
        "dino_latency_feasible_matched_20260728_v1"
    )
    assert [item["allocation_id"] for item in manifest["schedule"]["allocations"]] == [
        f"latency_feasible_allocation_{index:02d}" for index in range(6)
    ]
    assert manifest["runtime"]["local_runtime_path"] == str(
        launcher.DEFAULT_RUNTIME.resolve()
    )
    assert manifest["runtime"]["output_contract"]["relative_layout"] == (
        "dino_moo_phase2_20260728/latency_feasible_matched/"
        "<manifest_id>/<allocation_id>"
    )
    for key in (
        "selector_invoked_on_matched_measurements",
        "selection_time_objectives_replaced",
        "measurements_feed_selection",
        "measurements_feed_reselection",
        "algorithm_selected_candidate_overridden",
    ):
        assert manifest["selection_isolation"][key] is False

    configs = {
        item["candidate_id"]: item["candidate_id"].encode()
        for item in manifest["candidates"]
    }
    manifest["source_artifacts"]["post_front_tools"][
        launcher.BLOCK_RUNNER.name
    ] = copy.deepcopy(
        manifest["source_artifacts"]["post_front_tools"][
            "post_front_matched_block_runner.py"
        ]
    )
    plan = launcher.build_block_plan(
        manifest,
        "a" * 64,
        manifest["schedule"]["allocations"][0],
        configs,
    )
    for key in (
        "selector_invoked_on_matched_measurements",
        "selection_time_objectives_replaced",
        "measurements_feed_selection",
        "measurements_feed_reselection",
        "algorithm_selected_candidate_overridden",
    ):
        assert plan[key] is False


def test_new_harness_sources_are_the_only_pinned_tool_set() -> None:
    assert generator.TOOL_FILENAMES == (
        "latency_feasible_matched_manifest_generator.py",
        "latency_feasible_matched_launcher.py",
        "latency_feasible_matched_block_runner.py",
        "latency_feasible_matched_aggregator.py",
    )
    assert launcher.BLOCK_RUNNER.name in generator.TOOL_FILENAMES
    assert launcher.AGGREGATOR.name in generator.TOOL_FILENAMES
    assert launcher.EXPECTED_ACKNOWLEDGEMENT == (
        "USER_AUTHORIZED_DINO_LATENCY_FEASIBLE_6X8GPU_VALIDATION_20260728"
    )
