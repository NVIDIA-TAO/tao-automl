#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate the two-candidate 90%-policy matched latency campaign.

Raw-result validation and per-cell stabilized statistics reuse the proven
post-front aggregator. This wrapper limits comparative analysis to the one
replay-derived pair and keeps every matched measurement outside selection.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import latency_90_policy_matched_launcher as MATCHED  # noqa: E402
import post_front_matched_aggregator as POST_FRONT  # noqa: E402


DEFAULT_RUNTIME_DIR = MATCHED.DEFAULT_RUNTIME_DIR
DEFAULT_LEDGER_PATH = DEFAULT_RUNTIME_DIR / "block_submissions.json"
DEFAULT_SDK_STATE_PATH = DEFAULT_RUNTIME_DIR / "slurm_state.json"
DEFAULT_OUTPUT_PATH = (
    DEFAULT_RUNTIME_DIR / "latency_90_policy_matched_analysis.json"
)


class AggregationError(RuntimeError):
    """Raised when the 90%-policy comparative-analysis contract is violated."""


def quality_gate_projection(
    measurement: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    thresholds = manifest["latency_protocol"]["validity_thresholds"]
    median = float(measurement["median_ms"])
    median_ci = measurement["bootstrap_median_ci95_ms"]
    p95_ci = measurement["bootstrap_p95_ci95_ms"]
    values = {
        "robust_cv": float(measurement["robust_cv"]),
        "round_median_range_fraction": (
            float(measurement["round_median_range_ms"]) / median
        ),
        "absolute_round_drift_fraction": (
            abs(float(measurement["round_drift_ms"])) / median
        ),
        "device_median_range_fraction": (
            float(measurement["device_median_range_ms"]) / median
        ),
        "median_bootstrap_ci_width_fraction": (
            (float(median_ci[1]) - float(median_ci[0])) / median
        ),
        "p95_bootstrap_ci_width_ms": (
            float(p95_ci[1]) - float(p95_ci[0])
        ),
    }
    gates = {
        "robust_cv": (
            values["robust_cv"] <= thresholds["max_robust_cv"]
        ),
        "round_median_range": (
            values["round_median_range_fraction"]
            <= thresholds["max_round_median_range_fraction"]
        ),
        "absolute_round_drift": (
            values["absolute_round_drift_fraction"]
            <= thresholds["max_absolute_round_drift_fraction"]
        ),
        "device_median_range": (
            values["device_median_range_fraction"]
            <= thresholds["max_device_median_range_fraction"]
        ),
        "median_bootstrap_ci_width": (
            values["median_bootstrap_ci_width_fraction"]
            <= thresholds["max_bootstrap_ci_width_fraction"]
        ),
        "raw_sample_count": (
            measurement["raw_sample_count_total"] == 4000
        ),
        "proven_aggregator_validity": measurement["is_valid"] is True,
    }
    return {
        "values": values,
        "thresholds": copy.deepcopy(thresholds),
        "gates": gates,
        "all_quality_gates_pass": all(gates.values()),
        "invalid_reasons": copy.deepcopy(
            measurement.get("invalid_reasons", [])
        ),
    }


def complete_cell_table(
    manifest: dict[str, Any],
    measurements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_ids = manifest["candidate_derivation"]["candidate_ids"]
    allocations = sorted(
        manifest["schedule"]["allocations"],
        key=lambda item: item["allocation_index"],
    )
    by_key = {
        (item["allocation_id"], item["candidate_id"]): item
        for item in measurements
    }
    expected = {
        (allocation["allocation_id"], candidate_id)
        for allocation in allocations
        for candidate_id in candidate_ids
    }
    if len(measurements) != 12 or set(by_key) != expected:
        raise AggregationError(
            "Matched aggregation requires the complete 12-cell matrix"
        )
    cells = []
    for allocation in allocations:
        for candidate_id in candidate_ids:
            item = by_key[(allocation["allocation_id"], candidate_id)]
            quality = quality_gate_projection(item, manifest)
            if not quality["all_quality_gates_pass"]:
                raise AggregationError(
                    f"Quality gates failed for "
                    f"{allocation['allocation_id']}/{candidate_id}"
                )
            cells.append({
                **copy.deepcopy(item),
                "quality_gate_audit": quality,
            })
    return cells


def endpoint_analysis(
    *,
    first_candidate_id: str,
    second_candidate_id: str,
    allocation_ids: list[str],
    by_key: dict[tuple[str, str], dict[str, Any]],
    metric: str,
    tolerance: float,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    differences = [
        float(by_key[(allocation_id, first_candidate_id)][metric])
        - float(by_key[(allocation_id, second_candidate_id)][metric])
        for allocation_id in allocation_ids
    ]
    paired_difference = float(statistics.median(differences))
    bootstrap_ci = POST_FRONT.paired_bootstrap_ci(
        differences,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    descriptive = POST_FRONT.classify_difference(
        paired_difference,
        bootstrap_ci,
        tolerance,
    )
    directional = POST_FRONT.directional_pairwise_evidence(
        differences,
        tolerance=tolerance,
        confidence=confidence,
    )
    descriptive_equivalence = (
        bootstrap_ci[0] >= -tolerance
        and bootstrap_ci[1] <= tolerance
    )
    if (
        directional["directional_claim"]
        != "no_stable_directional_claim"
    ):
        effective_classification = directional["directional_claim"]
    elif descriptive_equivalence:
        effective_classification = (
            "no_stable_direction_descriptive_practical_equivalence"
        )
    else:
        effective_classification = "no_stable_direction_unresolved"
    return {
        "metric": metric,
        "delta_convention": (
            "first minus second; negative means first is faster"
        ),
        "allocation_ids": allocation_ids,
        "paired_differences_ms": differences,
        "median_paired_difference_ms": paired_difference,
        "descriptive_paired_bootstrap_ci95_ms": bootstrap_ci,
        "bootstrap_role": (
            "descriptive only for the effective directional rule"
        ),
        "point_and_interval_classification": descriptive,
        "descriptive_practical_equivalence": descriptive_equivalence,
        "exact_tolerance_shifted_sign_flip": directional,
        "all_six_beyond_negative_tolerance": directional[
            "all_six_beyond_negative_tolerance"
        ],
        "all_six_beyond_positive_tolerance": directional[
            "all_six_beyond_positive_tolerance"
        ],
        "effective_classification": effective_classification,
        "failure_to_prove_direction_implies_equivalence": False,
    }


def comparative_analysis(
    manifest: dict[str, Any],
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_ids = sorted(
        manifest["candidate_derivation"]["candidate_ids"]
    )
    if len(candidate_ids) != 2:
        raise AggregationError(
            "90%-policy matched analysis requires exactly two candidates"
        )
    allocation_ids = [
        item["allocation_id"]
        for item in sorted(
            manifest["schedule"]["allocations"],
            key=lambda item: item["allocation_index"],
        )
    ]
    by_key = {
        (item["allocation_id"], item["candidate_id"]): item
        for item in cells
    }
    paired = manifest["paired_analysis"]
    tolerance = float(paired["practical_tolerance_ms"])
    resamples = int(paired["bootstrap_resamples"])
    confidence = float(paired["bootstrap_confidence_level"])
    seed = int(paired["bootstrap_seed"])
    between = []
    for candidate_id in candidate_ids:
        medians = [
            float(by_key[(allocation_id, candidate_id)]["median_ms"])
            for allocation_id in allocation_ids
        ]
        p95s = [
            float(by_key[(allocation_id, candidate_id)]["p95_ms"])
            for allocation_id in allocation_ids
        ]
        between.append({
            "candidate_id": candidate_id,
            "median_latency": POST_FRONT.distribution_summary(medians),
            "p95_latency": POST_FRONT.distribution_summary(p95s),
        })
    first, second = candidate_ids
    endpoints = {
        "median_ms": endpoint_analysis(
            first_candidate_id=first,
            second_candidate_id=second,
            allocation_ids=allocation_ids,
            by_key=by_key,
            metric="median_ms",
            tolerance=tolerance,
            resamples=resamples,
            confidence=confidence,
            seed=seed,
        ),
        "p95_ms": endpoint_analysis(
            first_candidate_id=first,
            second_candidate_id=second,
            allocation_ids=allocation_ids,
            by_key=by_key,
            metric="p95_ms",
            tolerance=tolerance,
            resamples=resamples,
            confidence=confidence,
            seed=seed,
        ),
    }
    return {
        "scope": "one allocation-matched candidate pair",
        "first_candidate_id": first,
        "second_candidate_id": second,
        "practical_tolerance_ms": tolerance,
        "paired_bootstrap": {
            "unit": "allocation",
            "resamples": resamples,
            "confidence_level": confidence,
            "seed": seed,
            "statistic": "median of six allocation-paired differences",
            "effective_role": "descriptive_only",
        },
        "directional_rule": {
            "exact_one_sided_tolerance_shifted_sign_flip_required": True,
            "all_six_differences_beyond_tolerance_required": True,
            "alpha": 1.0 - confidence,
            "absence_of_direction_implies_equivalence": False,
            "simultaneous_total_order_claimed": False,
        },
        "between_allocation_statistics": between,
        "pairwise_endpoints": endpoints,
    }


def expected_ledger_source_checks(
    projection: dict[str, Any],
    recovery_binding: dict[str, str],
) -> dict[str, Any]:
    return {
        "campaign_id": projection["campaign_id"],
        "projection_source_bindings": (
            MATCHED.verify_source_bindings(projection)
        ),
        "checkpoint_recovery_evidence": copy.deepcopy(recovery_binding),
        "machinery_reuse": copy.deepcopy(projection["machinery_reuse"]),
    }


def build_report(
    *,
    projection: dict[str, Any],
    projection_whole_sha256: str,
    recovery_binding: dict[str, str],
    manifest: dict[str, Any],
    ledger: dict[str, Any],
    ledger_whole_sha256: str,
    jobs: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    consistency: dict[str, Any],
    runtime_provenance: dict[str, Any],
) -> dict[str, Any]:
    core = {
        "schema_version": 1,
        "status": "complete",
        "campaign_id": projection["campaign_id"],
        "compatibility_manifest_id": manifest["manifest_id"],
        "execution_projection": {
            "path": str(MATCHED.DEFAULT_PROJECTION_PATH.resolve()),
            "whole_file_sha256": projection_whole_sha256,
            "internal_sha256": projection["artifact_integrity"][
                "canonical_payload_sha256"
            ],
        },
        "checkpoint_recovery_evidence": copy.deepcopy(recovery_binding),
        "submission_ledger": {
            "path": str(DEFAULT_LEDGER_PATH.resolve()),
            "whole_file_sha256": ledger_whole_sha256,
            "internal_sha256": ledger["ledger_sha256"],
            "revision": ledger["ledger_revision"],
        },
        "candidate_ids": projection["candidate_derivation"][
            "candidate_ids"
        ],
        "schedule_sha256": projection["schedule"]["schedule_sha256"],
        "jobs": copy.deepcopy(jobs),
        "artifact_consistency": copy.deepcopy(consistency),
        "aggregation_runtime_provenance": copy.deepcopy(
            runtime_provenance
        ),
        "complete_12_cell_latency_evidence": copy.deepcopy(cells),
        "analysis": comparative_analysis(manifest, cells),
        "selection_isolation": dict(MATCHED.SELECTION_ISOLATION),
        "allowed_use": (
            "validation of latency ordering and equivalence only; matched "
            "measurements cannot enter selection or reselection"
        ),
        "selector_invocation_count_on_matched_measurements": 0,
        "selection_time_archive_preserved": True,
    }
    return {
        **core,
        "artifact_integrity": {
            "hash_algorithm": "sha256",
            "canonical_payload_sha256": MATCHED.canonical_sha256(core),
            "hash_excludes": ["artifact_integrity"],
        },
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projection",
        type=Path,
        default=MATCHED.DEFAULT_PROJECTION_PATH,
    )
    parser.add_argument("--projection-sha256", required=True)
    parser.add_argument(
        "--checkpoint-recovery-evidence",
        type=Path,
        default=MATCHED.DEFAULT_RECOVERY_EVIDENCE_PATH,
    )
    parser.add_argument("--checkpoint-recovery-evidence-sha256")
    parser.add_argument(
        "--submission-ledger",
        type=Path,
        default=DEFAULT_LEDGER_PATH,
    )
    parser.add_argument("--submission-ledger-sha256", required=True)
    parser.add_argument(
        "--sdk-state",
        type=Path,
        default=DEFAULT_SDK_STATE_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    projection, projection_whole_sha256 = MATCHED.load_projection(
        args.projection.resolve(),
        args.projection_sha256,
    )
    recovery_path = args.checkpoint_recovery_evidence.resolve()
    recovery_sha256 = (
        MATCHED.file_sha256(recovery_path)
        if args.checkpoint_recovery_evidence_sha256 is None
        else args.checkpoint_recovery_evidence_sha256
    )
    recovery_evidence, recovery_binding = MATCHED.load_recovery_evidence(
        projection,
        recovery_path,
        recovery_sha256,
    )
    manifest = MATCHED.compatibility_manifest(
        projection,
        recovery_evidence,
    )
    runtime_dir = Path(manifest["runtime"]["local_runtime_path"]).resolve()
    if args.submission_ledger.resolve() != runtime_dir / (
        "block_submissions.json"
    ):
        raise AggregationError("Submission ledger path drift")
    if args.sdk_state.resolve() != runtime_dir / "slurm_state.json":
        raise AggregationError("SDK state path drift")
    if args.output.resolve() != DEFAULT_OUTPUT_PATH.resolve():
        raise AggregationError("Output path drift")
    expected_sources = expected_ledger_source_checks(
        projection,
        recovery_binding,
    )
    ledger, ledger_whole_sha256 = POST_FRONT.load_ledger(
        args.submission_ledger.resolve(),
        args.submission_ledger_sha256,
        manifest,
        projection_whole_sha256,
        expected_sources,
    )
    runtime_provenance = POST_FRONT.aggregation_runtime_provenance(
        manifest
    )
    POST_FRONT.launcher.load_env_file(
        Path(manifest["runtime"]["secrets_env_path"])
    )
    jobs, _ = POST_FRONT.inspect_jobs(
        manifest,
        ledger,
        args.sdk_state.resolve(),
    )
    if not all(item["complete"] for item in jobs):
        print(json.dumps({
            "schema_version": 1,
            "status": "pending_or_failed_no_partial_aggregation",
            "campaign_id": projection["campaign_id"],
            "jobs": jobs,
            "partial_measurements_used": False,
            "selection_isolation": dict(MATCHED.SELECTION_ISOLATION),
        }, indent=2, sort_keys=True))
        return 2
    bundles = {
        job["allocation_id"]: POST_FRONT.fetch_allocation_bundle(
            manifest,
            job,
        )
        for job in jobs
    }
    measurements, consistency = POST_FRONT.aggregate_bundles(
        manifest,
        projection_whole_sha256,
        jobs,
        bundles,
    )
    cells = complete_cell_table(manifest, measurements)
    report = build_report(
        projection=projection,
        projection_whole_sha256=projection_whole_sha256,
        recovery_binding=recovery_binding,
        manifest=manifest,
        ledger=ledger,
        ledger_whole_sha256=ledger_whole_sha256,
        jobs=jobs,
        cells=cells,
        consistency=consistency,
        runtime_provenance=runtime_provenance,
    )
    POST_FRONT.write_new_report(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
