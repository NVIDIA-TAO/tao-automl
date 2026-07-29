#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay the sealed 60-candidate DINO archive under 90% latency retention.

This evidence generator delegates every mode winner and tied-cohort decision
to the production selector.  It does not benchmark candidates, rerun search,
or replace any frozen objective.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

from tao_automl.selection import (
    AccuracyConstraint,
    SelectionAnalysis,
    SelectionConfig,
    analyze_archive,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
CANDIDATE_TABLE_PATH = (
    EXPERIMENT_DIR
    / "runtime"
    / "expanded_search_v2"
    / "expanded_candidate_table.json"
)
SEALED_SELECTION_PATH = (
    EXPERIMENT_DIR
    / "runtime"
    / "expanded_search_v2"
    / "expanded_combined_selection.json"
)
SELECTOR_SOURCE_PATH = REPO_ROOT / "src" / "tao_automl" / "selection.py"
DEFAULT_OUTPUT_PATH = (
    EXPERIMENT_DIR / "latency_90_policy" / "archive_replay.v1.json"
)

EXPECTED_INPUT_SHA256 = {
    "expanded_candidate_table.json": (
        "5ba323d05d9ec8e3703e636f8b5e2975cc620eeec10df75ec6e792318dc2df03"
    ),
    "expanded_combined_selection.json": (
        "78ab9d2fa83cc3abe9057d137c0b88f120158b6ad77268482d2c18f5a1533af1"
    ),
}
RETENTION = 0.90
LATENCY_TOLERANCE_MS = 0.73553775
PERMUTATION_SEEDS = (161803, 271828, 314159, 20260728, 90, 9001)
ISOLATION_FLAGS = {
    "selector_invoked_on_matched_measurements": False,
    "selection_time_objectives_replaced": False,
    "measurements_feed_selection": False,
    "measurements_feed_reselection": False,
    "algorithm_selected_candidate_overridden": False,
}


class ReplayContractError(RuntimeError):
    """Raised when frozen replay evidence violates its preregistered contract."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _validate_input_hash(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    observed = _file_sha256(path)
    if observed != expected_sha256:
        raise ReplayContractError(
            f"Frozen input hash mismatch for {path}: "
            f"expected {expected_sha256}, observed {observed}"
        )
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "expected_sha256": expected_sha256,
        "observed_sha256": observed,
        "match": True,
    }


def _load_frozen_archive() -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    input_integrity = {
        "candidate_table": _validate_input_hash(
            CANDIDATE_TABLE_PATH,
            expected_sha256=EXPECTED_INPUT_SHA256[
                "expanded_candidate_table.json"
            ],
        ),
        "sealed_selection": _validate_input_hash(
            SEALED_SELECTION_PATH,
            expected_sha256=EXPECTED_INPUT_SHA256[
                "expanded_combined_selection.json"
            ],
        ),
    }
    table = _load_json(CANDIDATE_TABLE_PATH)
    sealed = _load_json(SEALED_SELECTION_PATH)
    rows = table.get("rows")
    if not isinstance(rows, list) or len(rows) != 60:
        raise ReplayContractError(
            "Frozen expanded candidate table must contain exactly 60 rows"
        )
    candidate_ids = [str(row.get("candidate_id", "")) for row in rows]
    if len(set(candidate_ids)) != 60 or any(not item for item in candidate_ids):
        raise ReplayContractError(
            "Frozen expanded archive must contain 60 unique non-empty IDs"
        )
    if any(row.get("status") != "success" for row in rows):
        raise ReplayContractError("All frozen archive rows must be successful")
    if table.get("manual_candidate_injection_used") is not False:
        raise ReplayContractError("Frozen archive reports manual injection")

    sealed_audits = {
        item["candidate_id"]: item for item in sealed.get("candidates", [])
    }
    if set(sealed_audits) != set(candidate_ids):
        raise ReplayContractError(
            "Candidate table and sealed selection contain different IDs"
        )
    for row in rows:
        candidate_id = row["candidate_id"]
        objectives = row["objective_values"]
        audit = sealed_audits[candidate_id]
        expected_values = {
            "accuracy": objectives["mAP50"],
            "latency": objectives["latency_ms"],
            "latency_ci95_low": objectives["latency_ci95_low"],
            "latency_ci95_high": objectives["latency_ci95_high"],
        }
        observed_values = {
            key: audit[key] for key in expected_values
        }
        if observed_values != expected_values:
            raise ReplayContractError(
                f"Frozen selector audit differs from candidate row {candidate_id}"
            )

    input_integrity.update({
        "candidate_count": len(rows),
        "successful_candidate_count": len(rows),
        "unique_candidate_id_count": len(set(candidate_ids)),
        "candidate_id_set_sha256": _canonical_sha256(sorted(candidate_ids)),
        "candidate_objectives_match_sealed_selection": True,
        "manual_candidate_injection_used": False,
    })
    return copy.deepcopy(rows), sealed, input_integrity


def _production_selector_identity() -> dict[str, str]:
    imported = Path(inspect.getsourcefile(analyze_archive) or "").resolve()
    expected = SELECTOR_SOURCE_PATH.resolve()
    if imported != expected:
        raise ReplayContractError(
            "Replay imported a selector outside this checkout: "
            f"{imported}, expected {expected}"
        )
    return {
        "module": "tao_automl.selection",
        "function": "analyze_archive",
        "source_path": str(expected.relative_to(REPO_ROOT)),
        "source_sha256": _file_sha256(expected),
    }


def _selection_config(
    sealed_selection: dict[str, Any],
) -> SelectionConfig:
    frozen = sealed_selection["algorithm"]["configuration"]
    if not math.isclose(
        float(frozen["latency_tolerance"]),
        LATENCY_TOLERANCE_MS,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ReplayContractError(
            "Sealed latency tolerance differs from the 90% replay contract"
        )
    return SelectionConfig(
        mode=str(frozen["mode"]),
        accuracy_metric=str(frozen["accuracy_metric"]),
        latency_metric=str(frozen["latency_metric"]),
        latency_accuracy_retention=AccuracyConstraint(
            kind="relative",
            value=RETENTION,
            reference="accuracy_winner",
        ),
        multi_objective_min_accuracy=None,
        accuracy_tolerance=float(frozen["accuracy_tolerance"]),
        latency_tolerance=LATENCY_TOLERANCE_MS,
        score_tolerance=float(frozen["score_tolerance"]),
        augmentation_rho=float(frozen["augmentation_rho"]),
        normalization=str(frozen["normalization"]),
        latency_ci_low_metric=str(frozen["latency_ci_low_metric"]),
        latency_ci_high_metric=str(frozen["latency_ci_high_metric"]),
    )


def _candidate_orders(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    orders = {
        "archive": list(rows),
        "reverse": list(reversed(rows)),
        "candidate_id": sorted(rows, key=lambda row: row["candidate_id"]),
    }
    for seed in PERMUTATION_SEEDS:
        shuffled = list(rows)
        random.Random(seed).shuffle(shuffled)
        orders[f"permutation_seed_{seed}"] = shuffled
    return orders


def _run_order_invariant_selector(
    rows: list[dict[str, Any]],
    config: SelectionConfig,
) -> tuple[SelectionAnalysis, dict[str, Any]]:
    orders = _candidate_orders(rows)
    analyses = {
        name: analyze_archive(order, config)
        for name, order in orders.items()
    }
    payloads = {
        name: analysis.to_dict() for name, analysis in analyses.items()
    }
    payload_hashes = {
        name: _canonical_sha256(payload)
        for name, payload in payloads.items()
    }
    if len(set(payload_hashes.values())) != 1:
        raise ReplayContractError(
            "Production selector output changed with candidate order"
        )
    return analyses["archive"], {
        "passed": True,
        "complete_selector_output_identical": True,
        "ordering_count": len(orders),
        "orders": {
            name: {
                "candidate_id_order_sha256": _canonical_sha256(
                    [row["candidate_id"] for row in order]
                ),
                "selector_output_sha256": payload_hashes[name],
            }
            for name, order in orders.items()
        },
        "common_selector_output_sha256": next(iter(payload_hashes.values())),
    }


def _intervals_overlap(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    return (
        float(first["latency_ci95_low"])
        <= float(second["latency_ci95_high"])
        and float(second["latency_ci95_low"])
        <= float(first["latency_ci95_high"])
    )


def _feasible_table(
    analysis: SelectionAnalysis,
    source_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    serialized = analysis.to_dict()
    audits = {
        item["candidate_id"]: item for item in serialized["candidates"]
    }
    feasible = [
        item
        for item in audits.values()
        if item["valid"] and item["latency_accuracy_feasible"]
    ]
    if not feasible:
        raise ReplayContractError(
            "The frozen archive contains no 90%-accuracy-feasible candidate"
        )
    feasible.sort(
        key=lambda item: (
            float(item["latency"]),
            item["fingerprint"],
            item["candidate_id"],
        )
    )
    raw_anchor = feasible[0]
    cohort_ids = tuple(analysis.latency.latency_tied_candidate_ids)
    cohort_set = set(cohort_ids)
    recomputed_cohort = {
        item["candidate_id"]
        for item in feasible
        if (
            float(item["latency"]) - float(raw_anchor["latency"])
            <= analysis.config.latency_tolerance
        )
    }
    if recomputed_cohort != cohort_set:
        raise ReplayContractError(
            "Serialized latency cohort disagrees with production tie policy"
        )

    winner_id = analysis.latency.winner_id
    if winner_id is None or winner_id not in cohort_set:
        raise ReplayContractError("Latency winner is absent from its tied cohort")
    winner = audits[winner_id]
    expected_winner = min(
        (audits[candidate_id] for candidate_id in cohort_ids),
        key=lambda item: (
            -float(item["accuracy"]),
            item["fingerprint"],
            item["candidate_id"],
        ),
    )
    if winner_id != expected_winner["candidate_id"]:
        raise ReplayContractError(
            "Latency winner disagrees with documented cohort tie-break"
        )

    accuracy_tie_break_invoked = len(cohort_ids) > 1
    other_accuracies = [
        float(audits[candidate_id]["accuracy"])
        for candidate_id in cohort_ids
        if candidate_id != winner_id
    ]
    higher_accuracy_resolved = bool(other_accuracies) and (
        float(winner["accuracy"]) - max(other_accuracies)
        > analysis.config.accuracy_tolerance
    )
    accuracy_equivalent_winners = [
        item
        for item in feasible
        if item["candidate_id"] in cohort_set
        and abs(float(item["accuracy"]) - float(winner["accuracy"]))
        <= analysis.config.accuracy_tolerance
    ]
    fingerprint_or_id_required = len(accuracy_equivalent_winners) > 1

    table: list[dict[str, Any]] = []
    for rank, item in enumerate(feasible, start=1):
        source = source_by_id[item["candidate_id"]]
        table.append({
            "selection_time_latency_rank": rank,
            "candidate_id": item["candidate_id"],
            "specs": copy.deepcopy(source["specs"]),
            "mAP50": item["accuracy"],
            "accuracy_retained_fraction": (
                float(item["accuracy"])
                / float(analysis.accuracy_reference_value)
            ),
            "selection_time_median_latency_ms": item["latency"],
            "selection_time_p95_latency_ms": source["objective_values"].get(
                "latency_p95_ms"
            ),
            "selection_time_latency_ci95_low_ms": item["latency_ci95_low"],
            "selection_time_latency_ci95_high_ms": item["latency_ci95_high"],
            "latency_delta_from_raw_minimum_ms": (
                float(item["latency"]) - float(raw_anchor["latency"])
            ),
            "within_practical_tolerance_of_raw_minimum": (
                float(item["latency"]) - float(raw_anchor["latency"])
                <= analysis.config.latency_tolerance
            ),
            "confidence_interval_overlaps_raw_minimum": (
                _intervals_overlap(item, raw_anchor)
            ),
            "equivalent_fastest_cohort": item["candidate_id"] in cohort_set,
            "selected": item["candidate_id"] == winner_id,
            "fingerprint": item["fingerprint"],
        })

    additional_plausible = [
        item["candidate_id"]
        for item in feasible
        if item["candidate_id"] not in cohort_set
        and float(item["latency_ci95_low"])
        <= (
            float(raw_anchor["latency_ci95_high"])
            + analysis.config.latency_tolerance
        )
    ]
    matched_scope = sorted(cohort_set | set(additional_plausible))
    excluded = [
        item for item in feasible if item["candidate_id"] not in matched_scope
    ]
    nearest_excluded = excluded[0] if excluded else None
    cohort = {
        "raw_minimum_latency_candidate_id": raw_anchor["candidate_id"],
        "raw_minimum_selection_time_latency_ms": raw_anchor["latency"],
        "equivalent_fastest_candidate_ids": sorted(cohort_ids),
        "selected_latency_candidate_id": winner_id,
        "accuracy_tie_break_invoked": accuracy_tie_break_invoked,
        "higher_accuracy_resolved_equivalent_fastest_cohort": (
            higher_accuracy_resolved
        ),
        "fingerprint_or_candidate_id_tie_break_required": (
            fingerprint_or_id_required
        ),
        "production_selection_reason": analysis.latency.reason,
        "policy_interpretation": (
            "Latency mode selected the lowest-latency candidate satisfying "
            "90% retained accuracy."
            if len(cohort_ids) == 1
            else
            "Latency mode selected the highest-accuracy member of the "
            "equivalent-fastest cohort satisfying 90% retained accuracy."
        ),
        "matched_validation_scope": {
            "candidate_ids": matched_scope,
            "equivalent_fastest_candidate_ids": sorted(cohort_ids),
            "additional_uncertainty_plausible_candidate_ids": (
                sorted(additional_plausible)
            ),
            "plausibility_rule": (
                "For an excluded feasible candidate, its selection-time "
                "latency CI lower bound is no greater than the raw anchor CI "
                "upper bound plus the frozen practical tolerance."
            ),
            "nearest_excluded_candidate": (
                None
                if nearest_excluded is None
                else {
                    "candidate_id": nearest_excluded["candidate_id"],
                    "median_delta_from_raw_minimum_ms": (
                        float(nearest_excluded["latency"])
                        - float(raw_anchor["latency"])
                    ),
                    "ci_gap_from_raw_anchor_ms": (
                        float(nearest_excluded["latency_ci95_low"])
                        - float(raw_anchor["latency_ci95_high"])
                    ),
                }
            ),
        },
    }
    return table, cohort


def build_payload() -> dict[str, Any]:
    rows, sealed, input_integrity = _load_frozen_archive()
    config = _selection_config(sealed)
    analysis, order_invariance = _run_order_invariant_selector(rows, config)
    serialized = analysis.to_dict()
    source_by_id = {row["candidate_id"]: row for row in rows}
    feasible_table, cohort = _feasible_table(analysis, source_by_id)

    accuracy_reference = float(analysis.accuracy_reference_value)
    expected_threshold = accuracy_reference * RETENTION
    if analysis.accuracy_threshold != expected_threshold:
        raise ReplayContractError(
            "Production selector threshold is not retention times A*"
        )
    if analysis.multi_objective_accuracy_threshold is not None:
        raise ReplayContractError(
            "Multi-objective mode unexpectedly inherited latency retention"
        )
    sealed_accuracy = sealed["selections"]["accuracy"]
    sealed_multi_objective = sealed["selections"]["multi_objective"]
    if serialized["selections"]["accuracy"] != sealed_accuracy:
        raise ReplayContractError(
            "Changing latency retention changed accuracy-mode selection"
        )
    if serialized["selections"]["multi_objective"] != sealed_multi_objective:
        raise ReplayContractError(
            "Changing latency retention changed multi-objective selection"
        )

    core = {
        "schema_version": 1,
        "purpose": (
            "Offline production-selector replay of the sealed 60-candidate "
            "DINO archive with latency mode retaining 90% of the accuracy "
            "winner; no training, benchmarking, or reselection from matched "
            "measurements is performed."
        ),
        "dataset": (
            "s3://nvcf-storage-handling/data/"
            "tao_od_synthetic_full_dino_coco/"
        ),
        "model_family": "DINO",
        "generated_by": {
            "script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "script_sha256": _file_sha256(Path(__file__).resolve()),
            "production_selector": _production_selector_identity(),
        },
        "frozen_inputs": input_integrity,
        "policy": {
            "latency_accuracy_retention": RETENTION,
            "constraint_type": "relative_to_accuracy_winner",
            "latency_tolerance_ms": LATENCY_TOLERANCE_MS,
            "multi_objective_min_accuracy": None,
            "objective": (
                "minimize latency subject to accuracy >= "
                "0.90 * accuracy_mode_winner_accuracy"
            ),
        },
        "resolved_policy": {
            "accuracy_winner_candidate_id": analysis.accuracy.winner_id,
            "accuracy_winner_mAP50": analysis.accuracy_reference_value,
            "minimum_accuracy_mAP50": analysis.accuracy_threshold,
            "decimal_policy_expression": (
                "0.90 * accuracy_mode_winner_accuracy"
            ),
            "feasible_candidate_count": len(feasible_table),
            "feasible_candidate_ids": [
                item["candidate_id"] for item in feasible_table
            ],
        },
        "selections": serialized["selections"],
        "latency_tied_cohort_audit": cohort,
        "complete_90_percent_feasible_candidate_table": feasible_table,
        "mode_independence": {
            "accuracy_selection_unchanged_from_sealed_archive": True,
            "multi_objective_selection_unchanged_from_sealed_archive": True,
            "multi_objective_min_accuracy": None,
            "multi_objective_inherited_latency_threshold": False,
            "sealed_accuracy_selection_sha256": _canonical_sha256(
                sealed_accuracy
            ),
            "replayed_accuracy_selection_sha256": _canonical_sha256(
                serialized["selections"]["accuracy"]
            ),
            "sealed_multi_objective_selection_sha256": _canonical_sha256(
                sealed_multi_objective
            ),
            "replayed_multi_objective_selection_sha256": _canonical_sha256(
                serialized["selections"]["multi_objective"]
            ),
        },
        "archive_order_invariance": order_invariance,
        "selection_isolation": dict(ISOLATION_FLAGS),
        "selection_authority": {
            "winner_ids_are_production_selector_outputs": True,
            "expected_candidate_id_encoded_in_replay_logic": False,
            "manual_candidate_injection_used": False,
            "manual_winner_override_used": False,
        },
    }
    return {
        **core,
        "artifact_integrity": {
            "hash_algorithm": "sha256",
            "canonical_payload_sha256": _canonical_sha256(core),
            "hash_excludes": ["artifact_integrity"],
        },
    }


def verify_payload(payload: dict[str, Any]) -> None:
    integrity = payload.get("artifact_integrity")
    if not isinstance(integrity, dict):
        raise ReplayContractError("Artifact integrity block is missing")
    core = {
        key: value
        for key, value in payload.items()
        if key != "artifact_integrity"
    }
    observed = _canonical_sha256(core)
    if observed != integrity.get("canonical_payload_sha256"):
        raise ReplayContractError(
            "Artifact canonical payload hash does not match its content"
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Deterministic JSON artifact path",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that an existing output exactly matches a fresh replay",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = build_payload()
    verify_payload(payload)
    output = args.output.resolve()
    if args.check:
        existing = _load_json(output)
        verify_payload(existing)
        if existing != payload:
            raise ReplayContractError(
                f"Existing artifact differs from fresh replay: {output}"
            )
    else:
        _write_json(output, payload)
    print(json.dumps({
        "status": "verified" if args.check else "written",
        "path": str(output),
        "whole_file_sha256": _file_sha256(output),
        "canonical_payload_sha256": payload["artifact_integrity"][
            "canonical_payload_sha256"
        ],
        "selected_latency_candidate_id": payload["selections"]["latency"][
            "winner_id"
        ],
        "feasible_candidate_count": payload["resolved_policy"][
            "feasible_candidate_count"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
