# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministically replay the frozen Phase-1 DINO candidate archive.

This is an offline evidence generator. It never launches work and never
implements its own ranking rule: every winner, rank, normalized regret, and
tie value comes from ``tao_automl.selection.analyze_archive``.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Iterable

from tao_automl.selection import (
    AccuracyConstraint,
    SelectionAnalysis,
    SelectionConfig,
    analyze_archive,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE1_DIR = REPO_ROOT / "experiments" / "dino_moo_review_20260727"
DEFAULT_OUTPUT = Path(__file__).with_name("phase1_offline_replay.json")
SELECTOR_SOURCE = REPO_ROOT / "src" / "tao_automl" / "selection.py"

SOURCE_ARCHIVE_SHA256 = {
    "experiments/dino_moo_review_20260727/combined_selection.json": (
        "794c038c9506b805b57c7812355ec173e5ea0275b21587befaf8c91a78cbe2f7"
    ),
    "experiments/dino_moo_review_20260727/seed_161803/"
    "candidate_evaluations.json": (
        "9f8cc57d0939e6744f78057da5b284d639402d6a91d9fdd1d66a6fb65818e258"
    ),
    "experiments/dino_moo_review_20260727/seed_271828/"
    "candidate_evaluations.json": (
        "5e10721a3ee0016222c6c9a23054e4a2837d15a8182cd94a1e4941fe84b316f3"
    ),
    "experiments/dino_moo_review_20260727/seed_314159/"
    "candidate_evaluations.json": (
        "d7701574825b7d8df5acbe84fc9b87df27f1afce784e1c9322a8deec8f7507ff"
    ),
}

POLICIES: tuple[tuple[str, dict[str, Any] | None], ...] = (
    (
        "relative_98",
        {
            "type": "relative",
            "value": 0.98,
            "reference": "accuracy_winner",
        },
    ),
    (
        "relative_95",
        {
            "type": "relative",
            "value": 0.95,
            "reference": "accuracy_winner",
        },
    ),
    (
        "relative_90",
        {
            "type": "relative",
            "value": 0.90,
            "reference": "accuracy_winner",
        },
    ),
    ("none", None),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _validate_selector_source() -> dict[str, str]:
    imported_source = Path(inspect.getsourcefile(analyze_archive) or "").resolve()
    expected_source = SELECTOR_SOURCE.resolve()
    if imported_source != expected_source:
        raise RuntimeError(
            "Replay must use the production selector in this checkout; "
            f"imported {imported_source}, expected {expected_source}"
        )
    return {
        "module": "tao_automl.selection",
        "function": "analyze_archive",
        "source_path": str(SELECTOR_SOURCE.relative_to(REPO_ROOT)),
        "source_sha256": _sha256(SELECTOR_SOURCE),
    }


def _load_frozen_candidates() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observed_hashes: dict[str, str] = {}
    for relative_path, expected_hash in SOURCE_ARCHIVE_SHA256.items():
        source_path = REPO_ROOT / relative_path
        observed_hash = _sha256(source_path)
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"Frozen source hash mismatch for {relative_path}: "
                f"expected {expected_hash}, observed {observed_hash}"
            )
        observed_hashes[relative_path] = observed_hash

    candidates: list[dict[str, Any]] = []
    seed_counts: dict[str, int] = {}
    for seed in (161803, 271828, 314159):
        relative_path = (
            f"experiments/dino_moo_review_20260727/seed_{seed}/"
            "candidate_evaluations.json"
        )
        evaluations = _read_json(REPO_ROOT / relative_path)["evaluations"]
        if not isinstance(evaluations, list):
            raise TypeError(f"{relative_path} evaluations must be a list")
        candidates.extend(evaluations)
        seed_counts[str(seed)] = len(evaluations)

    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    if len(candidates) != 30 or len(set(candidate_ids)) != 30:
        raise RuntimeError(
            "Frozen Phase-1 archive must contain exactly 30 uniquely identified "
            f"candidates, observed {len(candidates)} records and "
            f"{len(set(candidate_ids))} unique IDs"
        )

    combined_path = PHASE1_DIR / "combined_selection.json"
    combined_records = _read_json(combined_path)["candidate_records"]
    if set(combined_records) != set(candidate_ids):
        raise RuntimeError(
            "Per-seed archives and combined_selection.json contain different "
            "candidate identifiers"
        )
    if any(
        combined_records[candidate["candidate_id"]] != candidate
        for candidate in candidates
    ):
        raise RuntimeError(
            "Per-seed candidate records differ from the frozen combined archive"
        )

    return candidates, {
        "expected_and_observed_sha256": {
            path: {
                "expected": SOURCE_ARCHIVE_SHA256[path],
                "observed": observed_hashes[path],
                "match": True,
            }
            for path in sorted(SOURCE_ARCHIVE_SHA256)
        },
        "candidate_count": len(candidates),
        "unique_candidate_id_count": len(set(candidate_ids)),
        "per_seed_candidate_counts": seed_counts,
        "per_seed_records_match_combined_archive": True,
        "candidate_id_set_sha256": _canonical_sha256(sorted(candidate_ids)),
    }


def _selection_config(
    multi_objective_policy: dict[str, Any] | None,
) -> SelectionConfig:
    return SelectionConfig(
        mode="multi_objective",
        accuracy_metric="mAP50",
        latency_metric="latency_ms",
        latency_accuracy_retention=AccuracyConstraint(
            kind="relative",
            value=0.98,
            reference="accuracy_winner",
        ),
        multi_objective_min_accuracy=multi_objective_policy,
        accuracy_tolerance=1e-12,
        latency_tolerance=0.0,
        score_tolerance=1e-12,
        augmentation_rho=1e-6,
        normalization="pareto_front",
        latency_ci_low_metric="latency_ci95_low",
        latency_ci_high_metric="latency_ci95_high",
    )


def _orders(
    candidates: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    keyed_permutation = sorted(
        candidates,
        key=lambda candidate: hashlib.sha256(
            (
                "phase1-order-invariance-20260728:"
                + str(candidate["candidate_id"])
            ).encode("utf-8")
        ).hexdigest(),
    )
    return {
        "source": list(candidates),
        "reverse": list(reversed(candidates)),
        "sha256_keyed_permutation": keyed_permutation,
    }


def _analyze_orders(
    candidates: list[dict[str, Any]],
    config: SelectionConfig,
    policy_id: str,
) -> tuple[SelectionAnalysis, dict[str, Any]]:
    candidate_orders = _orders(candidates)
    analyses = {
        order_name: analyze_archive(order_candidates, config)
        for order_name, order_candidates in candidate_orders.items()
    }
    serialized = {
        order_name: analysis.to_dict()
        for order_name, analysis in analyses.items()
    }
    reference = serialized["source"]
    if any(payload != reference for payload in serialized.values()):
        raise RuntimeError(
            f"Production selector is candidate-order dependent for {policy_id}"
        )
    return analyses["source"], {
        "passed": True,
        "orders": {
            order_name: {
                "candidate_id_order_sha256": _canonical_sha256(
                    [
                        candidate["candidate_id"]
                        for candidate in candidate_orders[order_name]
                    ]
                ),
                "analysis_sha256": _canonical_sha256(payload),
            }
            for order_name, payload in serialized.items()
        },
        "all_analysis_hashes_identical": (
            len({_canonical_sha256(payload) for payload in serialized.values()})
            == 1
        ),
    }


def _audit_snapshot(
    analysis: SelectionAnalysis,
    candidate_id: str | None,
) -> dict[str, Any] | None:
    if candidate_id is None:
        return None
    audit = analysis.audit_for(candidate_id)
    if audit is None:
        raise RuntimeError(f"Selected candidate {candidate_id!r} lacks an audit")
    return {
        "candidate_id": audit.candidate_id,
        "mAP50": audit.accuracy,
        "median_latency_ms": audit.latency,
        "global_pareto_rank": audit.pareto_rank,
        "multi_objective_pareto_rank": audit.feasible_pareto_rank,
    }


def _comparison_to_mode_winners(
    analysis: SelectionAnalysis,
) -> dict[str, Any]:
    accuracy_id = analysis.accuracy.winner_id
    latency_id = analysis.latency.winner_id
    selected_id = analysis.multi_objective.winner_id
    accuracy = analysis.audit_for(accuracy_id) if accuracy_id is not None else None
    latency = analysis.audit_for(latency_id) if latency_id is not None else None
    selected = analysis.audit_for(selected_id) if selected_id is not None else None

    deltas: dict[str, float | None] = {
        "mAP50_vs_accuracy_winner": None,
        "median_latency_ms_vs_accuracy_winner": None,
        "mAP50_vs_latency_winner": None,
        "median_latency_ms_vs_latency_winner": None,
    }
    if selected is not None and accuracy is not None:
        deltas["mAP50_vs_accuracy_winner"] = (
            float(selected.accuracy) - float(accuracy.accuracy)
        )
        deltas["median_latency_ms_vs_accuracy_winner"] = (
            float(selected.latency) - float(accuracy.latency)
        )
    if selected is not None and latency is not None:
        deltas["mAP50_vs_latency_winner"] = (
            float(selected.accuracy) - float(latency.accuracy)
        )
        deltas["median_latency_ms_vs_latency_winner"] = (
            float(selected.latency) - float(latency.latency)
        )

    selected_is_accuracy = selected_id is not None and selected_id == accuracy_id
    selected_is_latency = selected_id is not None and selected_id == latency_id
    if analysis.multi_objective.status != "selected":
        classification = "no_multi_objective_selection"
    elif analysis.multi_objective.distinct_compromise:
        classification = "distinct_compromise"
    elif selected_is_accuracy and selected_is_latency:
        classification = "shared_accuracy_and_latency_mode_extreme"
    elif selected_is_accuracy:
        classification = "accuracy_mode_extreme"
    elif selected_is_latency:
        classification = "latency_mode_extreme"
    else:
        classification = "production_selector_policy_front_extreme"

    return {
        "accuracy_winner": _audit_snapshot(analysis, accuracy_id),
        "latency_winner": _audit_snapshot(analysis, latency_id),
        "multi_objective_winner": _audit_snapshot(analysis, selected_id),
        "multi_objective_minus_mode_winner_deltas": deltas,
        "extreme_status": {
            "classification": classification,
            "production_selector_distinct_compromise": (
                analysis.multi_objective.distinct_compromise
            ),
            "production_selector_fallback_used": (
                analysis.multi_objective.fallback_used
            ),
            "production_selector_reports_extreme": (
                analysis.multi_objective.status == "selected"
                and analysis.multi_objective.distinct_compromise is False
            ),
            "selected_is_accuracy_mode_winner": selected_is_accuracy,
            "selected_is_latency_mode_winner": selected_is_latency,
            "selected_is_either_mode_winner": (
                selected_is_accuracy or selected_is_latency
            ),
        },
    }


def _candidate_evidence(
    analysis: SelectionAnalysis,
    source_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for audit in analysis.to_dict()["candidates"]:
        source = source_by_id[audit["candidate_id"]]
        evidence.append({
            **audit,
            "global_pareto_rank": audit["pareto_rank"],
            "global_dominated_by": audit["dominated_by"],
            "eligible_pareto_rank": audit["multi_objective_pareto_rank"],
            "eligible_dominated_by": audit["multi_objective_dominated_by"],
            "normalized_accuracy_regret": (
                audit["normalized_accuracy_objective"]
            ),
            "normalized_latency_regret": (
                audit["normalized_latency_objective"]
            ),
            "augmented_chebyshev_score": (
                audit["multi_objective_compromise_score"]
            ),
            "search_seed": source["search_seed"],
            "training_seed": source["training_seed"],
            "rec_id": source["rec_id"],
            "specs": source["specs"],
            "measured_objectives": source["objective_values"],
        })
    return evidence


def _policy_replay(
    policy_id: str,
    policy: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _selection_config(policy)
    analysis, order_validation = _analyze_orders(
        candidates,
        config,
        policy_id,
    )
    source_by_id = {
        str(candidate["candidate_id"]): candidate for candidate in candidates
    }
    eligible_ids = sorted(
        audit.candidate_id
        for audit in analysis.audits
        if audit.multi_objective_accuracy_feasible
    )
    global_front_ids = sorted(
        audit.candidate_id
        for audit in analysis.audits
        if audit.pareto_rank == 0
    )
    eligible_front_ids = sorted(
        audit.candidate_id
        for audit in analysis.audits
        if audit.feasible_pareto_rank == 0
    )
    winner_id = analysis.multi_objective.winner_id
    winner_audit = analysis.audit_for(winner_id) if winner_id is not None else None
    winner_is_eligible_rank_zero = (
        winner_audit is not None
        and winner_audit.multi_objective_accuracy_feasible
        and winner_audit.feasible_pareto_rank == 0
        and not winner_audit.feasible_dominated_by
    )
    if analysis.multi_objective.status == "selected" and not (
        winner_is_eligible_rank_zero
    ):
        raise RuntimeError(
            f"Production selector returned a non-front winner for {policy_id}"
        )

    serialized = analysis.to_dict()
    replay = {
        "policy_id": policy_id,
        "configured_multi_objective_min_accuracy": (
            serialized["algorithm"]["configuration"][
                "multi_objective_min_accuracy"
            ]
        ),
        "fixed_latency_accuracy_retention": (
            serialized["algorithm"]["configuration"][
                "latency_accuracy_retention"
            ]
        ),
        "resolved_multi_objective_accuracy_policy": {
            "reference_candidate_id": (
                analysis.multi_objective_accuracy_reference_candidate_id
            ),
            "reference_value": (
                analysis.multi_objective_accuracy_reference_value
            ),
            "threshold": analysis.multi_objective_accuracy_threshold,
        },
        "eligible_candidate_ids": eligible_ids,
        "global_pareto_front_candidate_ids": global_front_ids,
        "eligible_pareto_front_candidate_ids": eligible_front_ids,
        "selected_candidate_ids": {
            "accuracy": analysis.accuracy.winner_id,
            "latency": analysis.latency.winner_id,
            "multi_objective": analysis.multi_objective.winner_id,
        },
        "normalization_bounds": analysis.normalization_bounds,
        "selections": serialized["selections"],
        "comparison_to_accuracy_and_latency_winners": (
            _comparison_to_mode_winners(analysis)
        ),
        "candidates": _candidate_evidence(analysis, source_by_id),
    }
    validation = {
        "policy_id": policy_id,
        "candidate_ordering_invariance": order_validation,
        "multi_objective_winner_is_eligible_nondominated_rank_zero": (
            winner_is_eligible_rank_zero
        ),
        "latency_accuracy_retention_is_relative_98": (
            analysis.config.latency_accuracy_retention.kind == "relative"
            and analysis.config.latency_accuracy_retention.value == 0.98
        ),
    }
    return replay, validation


def build_payload() -> dict[str, Any]:
    selector = _validate_selector_source()
    candidates, source_validation = _load_frozen_candidates()
    replays: list[dict[str, Any]] = []
    policy_validations: list[dict[str, Any]] = []
    for policy_id, policy in POLICIES:
        replay, validation = _policy_replay(policy_id, policy, candidates)
        replays.append(replay)
        policy_validations.append(validation)

    if len({
        replay["selected_candidate_ids"]["accuracy"] for replay in replays
    }) != 1:
        raise RuntimeError("Accuracy winner changed across MO eligibility policies")
    if len({
        replay["selected_candidate_ids"]["latency"] for replay in replays
    }) != 1:
        raise RuntimeError("Latency winner changed across MO eligibility policies")

    return {
        "schema_version": 1,
        "purpose": (
            "Offline production-selector sensitivity replay over the frozen "
            "30-candidate Phase-1 DINO archive; no measurements are changed "
            "and no result is fed back into candidate generation."
        ),
        "dataset": (
            "s3://nvcf-storage-handling/data/"
            "tao_od_synthetic_full_dino_coco/"
        ),
        "model": "DINO ResNet50",
        "generated_by": {
            "script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "production_selector": selector,
            "selection_authority": (
                "All ranks, scores, tie values, and selected IDs are emitted "
                "by tao_automl.selection.analyze_archive."
            ),
        },
        "source_archive_validation": source_validation,
        "controlled_configuration": {
            "candidate_archive": "frozen_phase1_union",
            "candidate_count": 30,
            "latency_accuracy_retention": {
                "type": "relative",
                "value": 0.98,
                "reference": "accuracy_winner",
            },
            "multi_objective_policy_sequence": [
                policy_id for policy_id, _policy in POLICIES
            ],
            "accuracy_metric": "mAP50",
            "latency_metric": "latency_ms",
            "normalization": "pareto_front",
            "augmentation_rho": 1e-6,
            "accuracy_tolerance": 1e-12,
            "latency_tolerance": 0.0,
            "score_tolerance": 1e-12,
        },
        "validation": {
            "all_source_hashes_match": True,
            "all_30_per_seed_records_match_combined_archive": True,
            "all_policy_replays_are_candidate_order_invariant": all(
                item["candidate_ordering_invariance"]["passed"]
                and item["candidate_ordering_invariance"][
                    "all_analysis_hashes_identical"
                ]
                for item in policy_validations
            ),
            "accuracy_winner_constant_across_policies": True,
            "latency_winner_constant_across_policies": True,
            "policies": policy_validations,
        },
        "replays": replays,
    }


def _serialize(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Generated replay JSON (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that --output already exactly matches regenerated evidence.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    serialized = _serialize(build_payload())
    output = args.output.resolve()
    if args.check:
        if not output.is_file():
            raise SystemExit(f"Replay output does not exist: {output}")
        observed = output.read_text(encoding="utf-8")
        if observed != serialized:
            raise SystemExit(
                f"Replay output is stale or non-deterministic: {output}"
            )
        print(f"verified deterministic replay: {output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    print(f"wrote deterministic replay: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
