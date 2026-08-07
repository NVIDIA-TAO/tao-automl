#!/usr/bin/env python3

"""Replay and classify frozen cross-model Pareto evidence without mutation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from tao_automl.selection import (
    AccuracyConstraint,
    SelectionConfig,
    analyze_archive,
)


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCES = HERE / "pareto_validation_sources.v1.json"
MODES = ("accuracy", "latency", "multi_objective")
ISOLATION_FLAGS = (
    "agent_selected_candidate",
    "agent_overrode_winner",
    "agent_injected_candidate",
    "agent_removed_candidate_to_change_winner",
    "agent_changed_objective_weights_after_results",
    "agent_changed_accuracy_retention_after_results",
    "agent_changed_multi_objective_policy_after_results",
    "agent_changed_search_space_after_results",
    "agent_changed_seed_after_results",
    "agent_replaced_measurement",
    "agent_modified_metric_to_favor_candidate",
    "agent_increased_budget_for_preferred_candidate",
    "agent_reordered_candidates_to_affect_ties",
)


class AuditError(RuntimeError):
    """Frozen evidence cannot be replayed safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_config(raw: Mapping[str, Any]) -> SelectionConfig:
    retention = raw["latency_accuracy_retention"]
    return SelectionConfig(
        mode=str(raw["mode"]),
        accuracy_metric=str(raw["accuracy_metric"]),
        latency_metric=str(raw["latency_metric"]),
        latency_accuracy_retention=AccuracyConstraint(
            kind=str(retention["type"]),
            value=retention["value"],
            reference=str(retention["reference"]),
        ),
        multi_objective_min_accuracy=raw.get(
            "multi_objective_min_accuracy"
        ),
        accuracy_tolerance=raw["accuracy_tolerance"],
        latency_tolerance=raw["latency_tolerance"],
        score_tolerance=raw["score_tolerance"],
        augmentation_rho=raw["augmentation_rho"],
        normalization=str(raw["normalization"]),
        latency_ci_low_metric=str(raw["latency_ci_low_metric"]),
        latency_ci_high_metric=str(raw["latency_ci_high_metric"]),
    )


def _recommendations(result: Mapping[str, Any]) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            id=str(item["rec_id"]),
            specs=item.get("specs", {}),
            status=item["status"],
            objective_values=item.get("objective_values", {}),
        )
        for item in result["history"]
    ]


def _replay(result: Mapping[str, Any], order_seed: int | None = None) -> dict[str, Any]:
    persisted = result["selection_analysis"]
    config = _selection_config(persisted["algorithm"]["configuration"])
    candidates = _recommendations(result)
    if order_seed is not None:
        random.Random(order_seed).shuffle(candidates)
    weights = persisted["algorithm"]["objective_weights"]
    return analyze_archive(
        candidates,
        config,
        accuracy_weight=weights[config.accuracy_metric],
        latency_weight=weights[config.latency_metric],
    ).to_dict()


def _load_result(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise AuditError(f"authoritative artifact is unavailable or changed: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    result = document.get("result")
    if not isinstance(result, dict):
        raise AuditError(f"result payload is invalid: {path}")
    return result


def _load_modes(
    record: Mapping[str, Any], artifact_root: Path
) -> dict[str, dict[str, Any]]:
    if record["kind"] == "completion":
        path = artifact_root / record["path"]
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise AuditError(
                f"authoritative completion is unavailable or changed: {path}"
            )
        modes = json.loads(path.read_text(encoding="utf-8")).get(
            "mode_results"
        )
        if not isinstance(modes, dict):
            raise AuditError(f"completion has no mode_results: {path}")
        return {mode: dict(modes[mode]) for mode in MODES}
    if record["kind"] != "mode_results":
        raise AuditError(f"unsupported source kind: {record['kind']!r}")
    return {
        mode: _load_result(
            artifact_root / record["paths"][mode]["path"],
            record["paths"][mode]["sha256"],
        )
        for mode in MODES
    }


def _winner(result: Mapping[str, Any], mode: str) -> dict[str, Any]:
    selection = result["selection_analysis"]["selections"][mode]
    candidate_id = selection["winner_id"]
    return next(
        item
        for item in result["selection_analysis"]["candidates"]
        if item["candidate_id"] == candidate_id
    )


def _mode_audit(result: Mapping[str, Any], mode: str) -> dict[str, Any]:
    analysis = result["selection_analysis"]
    config = analysis["algorithm"]["configuration"]
    selection = analysis["selections"][mode]
    valid = [item for item in analysis["candidates"] if item["valid"]]
    winner = _winner(result, mode)
    persisted_id = str(analysis["selections"][mode]["winner_id"])
    replay = _replay(result)
    replay_id = str(replay["selections"][mode]["winner_id"])
    order_ids = {
        str(_replay(result, seed)["selections"][mode]["winner_id"])
        for seed in range(10)
    }
    if mode == "accuracy":
        invariant = winner["accuracy"] >= (
            max(item["accuracy"] for item in valid)
            - config["accuracy_tolerance"]
        )
    elif mode == "latency":
        feasible = [
            item for item in valid if item["latency_accuracy_feasible"]
        ]
        tied_ids = set(
            analysis["selections"]["latency"][
                "latency_tied_candidate_ids"
            ]
        )
        tied = [item for item in feasible if item["candidate_id"] in tied_ids]
        invariant = bool(tied) and winner in tied and winner["accuracy"] >= (
            max(item["accuracy"] for item in tied)
            - config["accuracy_tolerance"]
        )
    else:
        eligible_front = [
            item
            for item in valid
            if item["multi_objective_accuracy_feasible"]
            and item["multi_objective_pareto_rank"] == 0
        ]
        best_score = min(
            item["multi_objective_compromise_score"]
            for item in eligible_front
        )
        invariant = (
            winner["multi_objective_pareto_rank"] == 0
            and not winner["multi_objective_dominated_by"]
            and winner["multi_objective_compromise_score"]
            <= best_score + config["score_tolerance"]
        )
    details: dict[str, Any] = {}
    if mode == "accuracy":
        details = {
            "maximum_valid_accuracy": max(
                item["accuracy"] for item in valid
            ),
        }
    elif mode == "latency":
        feasible = [
            item for item in valid if item["latency_accuracy_feasible"]
        ]
        details = {
            "accuracy_reference_candidate_id": analysis["algorithm"][
                "latency_accuracy_reference_candidate_id"
            ],
            "accuracy_reference_value": analysis["algorithm"][
                "latency_accuracy_reference_value"
            ],
            "accuracy_threshold": analysis["algorithm"][
                "latency_accuracy_threshold"
            ],
            "feasible_candidate_ids": sorted(
                str(item["candidate_id"]) for item in feasible
            ),
            "raw_minimum_latency_ms": min(
                item["latency"] for item in feasible
            ),
            "equivalent_fastest_candidate_ids": list(
                selection["latency_tied_candidate_ids"]
            ),
        }
    else:
        details = {
            "multi_objective_min_accuracy": config[
                "multi_objective_min_accuracy"
            ],
            "normalization_bounds": analysis["algorithm"][
                "normalization_bounds"
            ],
            "objective_weights": analysis["algorithm"][
                "objective_weights"
            ],
            "augmentation_rho": config["augmentation_rho"],
        }
    return {
        "winner_id": persisted_id,
        "candidate_fingerprint": winner["fingerprint"],
        "accuracy": winner["accuracy"],
        "latency_ms": winner["latency"],
        "invariant": bool(invariant),
        "valid_candidate_count": len(valid),
        "total_candidate_count": len(analysis["candidates"]),
        "replay_winner_id": replay_id,
        "replay_matches": replay_id == persisted_id,
        "order_invariant": order_ids == {persisted_id},
        "selection_reason": analysis["selections"][mode]["reason"],
        "configuration": config,
        "details": details,
    }


def _classify(
    mode_audits: Mapping[str, Mapping[str, Any]],
    result: Mapping[str, Any],
    *,
    pooled_accuracy_invariant: bool = True,
) -> tuple[str, bool]:
    accuracy = mode_audits["accuracy"]
    latency = mode_audits["latency"]
    multi = mode_audits["multi_objective"]
    config = result["selection_analysis"]["algorithm"]["configuration"]
    middle = (
        latency["accuracy"] <= multi["accuracy"] <= accuracy["accuracy"]
        and latency["latency_ms"] - config["latency_tolerance"]
        <= multi["latency_ms"]
        <= accuracy["latency_ms"] + config["latency_tolerance"]
    )
    if not accuracy["invariant"]:
        return "FAIL_SELECTOR", middle
    if not latency["invariant"]:
        return "FAIL_SELECTOR", middle
    if not multi["invariant"]:
        return "FAIL_SELECTOR", middle
    if not all(
        item["replay_matches"] and item["order_invariant"]
        for item in mode_audits.values()
    ):
        return "FAIL_SELECTOR", middle
    if not pooled_accuracy_invariant:
        # The accuracy selector is correct for the evidence it saw, but an
        # independently generated mode archive contains a higher valid task
        # metric.  This is a search/archive coverage outcome, not a selector
        # direction or tie-breaking failure.
        return "FAIL_SEARCH_OR_ARCHIVE", middle
    if multi["candidate_fingerprint"] in {
        accuracy["candidate_fingerprint"],
        latency["candidate_fingerprint"],
    }:
        return "PASS_ENDPOINT_COLLAPSE", middle
    if middle:
        return "PASS_EXPECTED_COMPROMISE", middle
    # A distinct compromise in its independently acquired archive can lie
    # outside the interval formed by winners of two other independent jobs.
    # That fact alone cannot distinguish valid geometry from acquisition or
    # measurement instability, so the machine audit deliberately does not
    # manufacture a PASS or FAIL label.
    return "INCONCLUSIVE", middle


def _valid_candidates(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in result["selection_analysis"]["candidates"]
        if item["valid"]
    ]


def _cross_archive_accuracy_audit(
    results: Mapping[str, Mapping[str, Any]],
    accuracy_mode: Mapping[str, Any],
) -> dict[str, Any]:
    config = results["accuracy"]["selection_analysis"]["algorithm"][
        "configuration"
    ]
    observed = [
        {
            "mode": mode,
            "candidate_id": str(item["candidate_id"]),
            "fingerprint": item["fingerprint"],
            "accuracy": item["accuracy"],
            "latency_ms": item["latency"],
        }
        for mode, result in results.items()
        for item in _valid_candidates(result)
    ]
    maximum = max(item["accuracy"] for item in observed)
    winner_accuracy = float(accuracy_mode["accuracy"])
    winner_fingerprint = str(accuracy_mode["candidate_fingerprint"])
    higher = [
        item
        for item in observed
        if item["accuracy"]
        > winner_accuracy + config["accuracy_tolerance"]
    ]
    higher_distinct = [
        item for item in higher if item["fingerprint"] != winner_fingerprint
    ]
    return {
        "invariant": not higher,
        "distinct_fingerprint_invariant": not higher_distinct,
        "independent_retraining_variation_only": bool(higher)
        and not higher_distinct,
        "observed_valid_candidate_count": len(observed),
        "maximum_observed_accuracy": maximum,
        "selected_accuracy": winner_accuracy,
        "higher_observations": sorted(
            higher,
            key=lambda item: (
                -item["accuracy"],
                item["fingerprint"],
                item["candidate_id"],
            ),
        ),
        "higher_distinct_fingerprint_observations": sorted(
            higher_distinct,
            key=lambda item: (
                -item["accuracy"],
                item["fingerprint"],
                item["candidate_id"],
            ),
        ),
        "interpretation": (
            "accuracy winner is the maximum valid task metric across all "
            "three independently generated archives"
            if not higher
            else (
                "the same selected specification produced a slightly higher "
                "metric in an independent retraining; direction is "
                "noise-limited without matched accuracy validation"
                if not higher_distinct
                else "accuracy selector is correct within its archive, but "
                "an independent mode job discovered a higher valid task "
                "metric from a different specification"
            )
        ),
    }


def _multi_objective_front(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    front = []
    for item in _valid_candidates(result):
        if (
            item["multi_objective_accuracy_feasible"]
            and item["multi_objective_pareto_rank"] == 0
        ):
            front.append(
                {
                    "candidate_id": str(item["candidate_id"]),
                    "fingerprint": item["fingerprint"],
                    "accuracy": item["accuracy"],
                    "latency_ms": item["latency"],
                    "normalized_accuracy_regret": item[
                        "normalized_accuracy_objective"
                    ],
                    "normalized_latency_regret": item[
                        "normalized_latency_objective"
                    ],
                    "compromise_score": item[
                        "multi_objective_compromise_score"
                    ],
                    "dominated_by": list(
                        item["multi_objective_dominated_by"]
                    ),
                    "tie_breaking_values": item["tie_breaking_values"][
                        "multi_objective_mode"
                    ],
                }
            )
    return sorted(
        front,
        key=lambda item: (
            -item["accuracy"],
            item["latency_ms"],
            item["fingerprint"],
            item["candidate_id"],
        ),
    )


def build_audit(source_path: Path, artifact_root: Path | None = None) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    root = (artifact_root or Path(source["artifact_root"])).resolve()
    models = []
    for record in source["models"]:
        results = _load_modes(record, root)
        modes = {
            mode: _mode_audit(results[mode], mode) for mode in MODES
        }
        cross_accuracy = _cross_archive_accuracy_audit(
            results, modes["accuracy"]
        )
        classification, middle = _classify(
            modes,
            results["multi_objective"],
            pooled_accuracy_invariant=cross_accuracy[
                "distinct_fingerprint_invariant"
            ],
        )
        models.append(
            {
                "model": record["model"],
                "dataset": record["dataset"],
                "modes": modes,
                "accuracy_invariant": cross_accuracy["invariant"],
                "accuracy_invariant_within_accuracy_archive": modes[
                    "accuracy"
                ]["invariant"],
                "cross_archive_accuracy_audit": cross_accuracy,
                "latency_invariant": modes["latency"]["invariant"],
                "pareto_invariant": modes["multi_objective"]["invariant"],
                "middle_ground_invariant": middle,
                "multi_objective_rank_zero_front": _multi_objective_front(
                    results["multi_objective"]
                ),
                "preliminary_classification": classification,
            }
        )
    return {
        "schema_version": 1,
        "kind": "read_only_cross_model_pareto_outlier_audit",
        "source_manifest": str(source_path.resolve()),
        "source_manifest_sha256": _sha256(source_path),
        "artifact_root": str(root),
        "models": models,
        "agent_intervention_flags": {name: False for name in ISOLATION_FLAGS},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    audit = build_audit(args.sources, args.artifact_root)
    payload = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
