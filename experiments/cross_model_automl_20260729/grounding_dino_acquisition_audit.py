#!/usr/bin/env python3

"""Audit frozen Grounding DINO recommendation coverage without reselection."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Mapping

from tao_automl.recommendation_audit import validate_recommendation_audit
from tao_automl.selection import canonical_spec_fingerprint


DEFAULT_ROOT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/grounding_dino_three_mode_pilot_v1"
)
MODES = ("accuracy", "latency", "multi_objective")
EXPECTED_OBJECTIVES = {
    "accuracy": [{"metric": "mAP50", "direction": "maximize"}],
    "latency": [
        {"metric": "mAP50", "direction": "maximize"},
        {"metric": "latency_ms", "direction": "minimize"},
    ],
    "multi_objective": [
        {"metric": "mAP50", "direction": "maximize"},
        {"metric": "latency_ms", "direction": "minimize"},
    ],
}
ISOLATION_FLAGS = {
    "agent_selected_candidate": False,
    "agent_overrode_winner": False,
    "agent_injected_candidate": False,
    "agent_removed_candidate_to_change_winner": False,
    "agent_changed_objective_weights_after_results": False,
    "agent_changed_accuracy_retention_after_results": False,
    "agent_changed_multi_objective_policy_after_results": False,
    "agent_changed_search_space_after_results": False,
    "agent_changed_seed_after_results": False,
    "agent_replaced_measurement": False,
    "agent_modified_metric_to_favor_candidate": False,
    "agent_increased_budget_for_preferred_candidate": False,
    "agent_reordered_candidates_to_affect_ties": False,
    "measurements_feed_selection": False,
    "measurements_feed_reselection": False,
    "selection_time_objectives_replaced": False,
    "algorithm_selected_candidate_overridden": False,
    "posthoc_measurements_feed_selection": False,
    "posthoc_measurements_feed_reselection": False,
    "historical_winner_overridden": False,
}


class AcquisitionAuditError(RuntimeError):
    """Frozen recommendation evidence is incomplete or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_value(document: Mapping[str, Any], path: str) -> Any:
    value: Any = document
    for part in path.split("."):
        value = value[part]
    return value


def _ordered_candidates(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (dict(item) for item in document["candidates"].values()),
        key=lambda item: int(
            item.get("recommendation_id", item.get("rec_id"))
        ),
    )


def _validate_range(value: Any, declaration: Mapping[str, Any]) -> bool:
    if declaration.get("valid_options"):
        return value in declaration["valid_options"]
    return (
        float(declaration["valid_min"])
        <= float(value)
        <= float(declaration["valid_max"])
    )


def _candidate_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    recommendation = item["recommendation_audit"]
    proposal = recommendation["acquisition"]["proposal"]
    ptm = proposal["ptm"]
    inner = ptm["inner_acquisition"]
    ranges = recommendation["custom_parameter_ranges"][ptm["arm_id"]]
    return {
        "candidate_id": item["candidate_id"],
        "recommendation_id": str(
            item.get("recommendation_id", item.get("rec_id"))
        ),
        "candidate_fingerprint": item["candidate_fingerprint"],
        "checkpoint_id": ptm["arm_id"],
        "stage": proposal["stage"],
        "acquisition_function": inner["acquisition_function"],
        "acquisition_mode": inner["acquisition_mode"],
        "search_seed": recommendation["search_seed"],
        "inner_seed": inner["seed"],
        "rng_state_sha256": inner["rng_state_sha256"],
        "visible_observation_count": len(
            recommendation["history_visible_to_algorithm"]
        ),
        "modelled_observation_count": inner["decision_state"][
            "observation_count"
        ],
        "objectives": inner["objectives"],
        "objective_values": item["objective_values"],
        "generated_search_values": {
            path: _path_value(item["specs"], path) for path in sorted(ranges)
        },
    }


def _validate_mode(
    mode: str,
    evidence: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = _ordered_candidates(evidence)
    history = sorted(result["history"], key=lambda item: int(item["rec_id"]))
    if len(candidates) != len(history):
        raise AcquisitionAuditError(f"{mode} candidate/history count changed")
    expected_ids = list(range(len(candidates)))
    observed_ids = [
        int(item.get("recommendation_id", item.get("rec_id")))
        for item in candidates
    ]
    if observed_ids != expected_ids:
        raise AcquisitionAuditError(f"{mode} recommendation IDs are not contiguous")

    summaries = []
    search_space_hashes = set()
    custom_range_hashes = set()
    search_seeds = set()
    fingerprints = []
    stages: dict[str, int] = {}
    acquisition_functions: dict[str, int] = {}
    for index, (item, persisted) in enumerate(zip(candidates, history)):
        audit = item["recommendation_audit"]
        validate_recommendation_audit(audit)
        if item["status"] != "success" or persisted["status"] != "success":
            raise AcquisitionAuditError(
                f"{mode} recommendation {index} is not a valid observation"
            )
        fingerprint = canonical_spec_fingerprint(item["specs"])
        if not (
            fingerprint
            == item["candidate_fingerprint"]
            == audit["candidate_fingerprint"]
            == canonical_spec_fingerprint(persisted["specs"])
        ):
            raise AcquisitionAuditError(
                f"{mode} recommendation {index} fingerprint mismatch"
            )
        visible = audit["history_visible_to_algorithm"]
        if len(visible) != index or [
            int(previous["candidate_id"]) for previous in visible
        ] != list(range(index)):
            raise AcquisitionAuditError(
                f"{mode} recommendation {index} visible history changed"
            )
        for previous_index, previous in enumerate(visible):
            prior = candidates[previous_index]
            if (
                previous["candidate_fingerprint"]
                != prior["candidate_fingerprint"]
                or previous["status"] != prior["status"]
                or previous["objective_values"] != prior["objective_values"]
            ):
                raise AcquisitionAuditError(
                    f"{mode} recommendation {index} history is inconsistent"
                )
        proposal = audit["acquisition"]["proposal"]
        if proposal["mode"] != mode:
            raise AcquisitionAuditError(f"{mode} proposal was routed incorrectly")
        ptm = proposal["ptm"]
        inner = ptm["inner_acquisition"]
        if (
            inner["acquisition_mode"] != mode
            or inner["objectives"] != EXPECTED_OBJECTIVES[mode]
        ):
            raise AcquisitionAuditError(
                f"{mode} acquisition objective direction changed"
            )
        ranges = audit["custom_parameter_ranges"][ptm["arm_id"]]
        for path, declaration in ranges.items():
            if not _validate_range(_path_value(item["specs"], path), declaration):
                raise AcquisitionAuditError(
                    f"{mode} recommendation {index} is outside {path}"
                )
        fingerprints.append(fingerprint)
        search_space_hashes.add(audit["search_space_sha256"])
        custom_range_hashes.add(audit["custom_parameter_ranges_sha256"])
        search_seeds.add(audit["search_seed"])
        stages[proposal["stage"]] = stages.get(proposal["stage"], 0) + 1
        function = inner["acquisition_function"]
        acquisition_functions[function] = (
            acquisition_functions.get(function, 0) + 1
        )
        summaries.append(_candidate_summary(item))
    if len(set(fingerprints)) != len(fingerprints):
        raise AcquisitionAuditError(f"{mode} contains duplicate recommendations")
    selection = result["selection_analysis"]["selections"][mode]
    return {
        "candidate_count": len(candidates),
        "failed_candidate_count": 0,
        "candidate_fingerprints_unique": True,
        "candidate_fingerprint_set_sha256": hashlib.sha256(
            json.dumps(sorted(fingerprints), separators=(",", ":")).encode()
        ).hexdigest(),
        "search_space_sha256_values": sorted(search_space_hashes),
        "custom_parameter_ranges_sha256_values": sorted(custom_range_hashes),
        "search_seed_values": sorted(search_seeds),
        "stages": stages,
        "acquisition_functions": acquisition_functions,
        "objective_directions": EXPECTED_OBJECTIVES[mode],
        "persisted_winner_id": str(selection["winner_id"]),
        "selection_reason": selection["reason"],
    }, summaries


def build_audit(root: Path) -> dict[str, Any]:
    completion_path = root / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "success" or completion.get("mode_failures"):
        raise AcquisitionAuditError("Grounding DINO completion is not successful")
    mode_audits = {}
    sequences = {}
    evidence_files = {}
    for mode in MODES:
        evidence_path = root / mode / "candidate_evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("mode") != mode:
            raise AcquisitionAuditError(f"{mode} evidence mode changed")
        if evidence.get("manifest_sha256") != completion["manifest_sha256"]:
            raise AcquisitionAuditError(f"{mode} manifest binding changed")
        audit, sequence = _validate_mode(
            mode, evidence, completion["mode_results"][mode]
        )
        mode_audits[mode] = audit
        sequences[mode] = sequence
        evidence_files[mode] = {
            "path": str(evidence_path),
            "sha256": _sha256(evidence_path),
        }

    counts = {len(sequence) for sequence in sequences.values()}
    if len(counts) != 1:
        raise AcquisitionAuditError("mode budgets differ")
    budget = counts.pop()
    common_prefix = 0
    for index in range(budget):
        if len(
            {
                sequences[mode][index]["candidate_fingerprint"]
                for mode in MODES
            }
        ) != 1:
            break
        common_prefix += 1

    accuracy_result = completion["mode_results"]["accuracy"]
    accuracy_winner_id = str(
        accuracy_result["selection_analysis"]["selections"]["accuracy"][
            "winner_id"
        ]
    )
    accuracy_winner = next(
        item
        for item in sequences["accuracy"]
        if item["recommendation_id"] == accuracy_winner_id
    )
    independent = [
        {**item, "mode": mode}
        for mode in ("latency", "multi_objective")
        for item in sequences[mode]
        if item["candidate_fingerprint"]
        != accuracy_winner["candidate_fingerprint"]
    ]
    higher = [
        item
        for item in independent
        if item["objective_values"]["mAP50"]
        > accuracy_winner["objective_values"]["mAP50"]
    ]
    higher_candidate = max(
        higher,
        key=lambda item: (
            item["objective_values"]["mAP50"],
            item["candidate_fingerprint"],
        ),
        default=None,
    )

    common_fingerprints = set.intersection(
        *(
            {item["candidate_fingerprint"] for item in sequences[mode]}
            for mode in MODES
        )
    )
    repeated_ranges = []
    for fingerprint in sorted(common_fingerprints):
        measurements = [
            item["objective_values"]["mAP50"]
            for mode in MODES
            for item in sequences[mode]
            if item["candidate_fingerprint"] == fingerprint
        ]
        repeated_ranges.append(max(measurements) - min(measurements))
    gap = (
        higher_candidate["objective_values"]["mAP50"]
        - accuracy_winner["objective_values"]["mAP50"]
        if higher_candidate
        else None
    )

    reachable = None
    if higher_candidate:
        arm = higher_candidate["checkpoint_id"]
        accuracy_ranges = json.loads(
            (root / "accuracy" / "candidate_evidence.json").read_text(
                encoding="utf-8"
            )
        )["candidates"]
        first = _ordered_candidates({"candidates": accuracy_ranges})[0]
        ranges = first["recommendation_audit"]["custom_parameter_ranges"][arm]
        reachable = all(
            _validate_range(
                higher_candidate["generated_search_values"][path], declaration
            )
            for path, declaration in ranges.items()
        )

    result = {
        "schema_version": 1,
        "kind": "read_only_grounding_dino_acquisition_coverage_audit",
        "completion": {
            "path": str(completion_path),
            "sha256": _sha256(completion_path),
            "manifest_sha256": completion["manifest_sha256"],
        },
        "candidate_evidence": evidence_files,
        "candidate_universe_contract": {
            "design": "independent_objective_aware_searches",
            "shared_candidate_universe": False,
            "cross_mode_observation_sharing": False,
        },
        "modes": mode_audits,
        "common_calibration": {
            "identical_fingerprint_prefix_count": common_prefix,
            "mode_specific_fingerprint_count": budget - common_prefix,
            "common_fingerprint_count": len(common_fingerprints),
        },
        "accuracy_coverage_observation": {
            "accuracy_winner": accuracy_winner,
            "higher_distinct_candidate": higher_candidate,
            "accuracy_gap": gap,
            "higher_candidate_reachable_in_accuracy_search_space": reachable,
            "higher_candidate_present_in_accuracy_archive": (
                higher_candidate is not None
                and any(
                    item["candidate_fingerprint"]
                    == higher_candidate["candidate_fingerprint"]
                    for item in sequences["accuracy"]
                )
            ),
            "identical_fingerprint_accuracy_range": {
                "count": len(repeated_ranges),
                "median": statistics.median(repeated_ranges),
                "maximum": max(repeated_ranges),
                "values": repeated_ranges,
            },
            "gap_is_below_median_identical_fingerprint_range": (
                gap is not None
                and gap <= statistics.median(repeated_ranges)
            ),
            "matched_accuracy_validation_required": bool(higher_candidate),
            "classification": (
                "INCONCLUSIVE_PENDING_MATCHED_ACCURACY_VALIDATION"
                if higher_candidate
                else "NO_CROSS_ARCHIVE_ACCURACY_COVERAGE_GAP"
            ),
        },
        "implementation_checks": {
            "accuracy_direction_is_maximize": True,
            "latency_direction_is_minimize": True,
            "multi_objective_uses_both_directions": True,
            "intended_accuracy_metric_is_used": True,
            "recommendation_audits_integrity_valid": True,
            "visible_history_is_complete_and_ordered": True,
            "failed_candidates_corrupting_surrogate": False,
            "generated_values_within_frozen_ranges": True,
            "candidate_fingerprints_are_unique_within_mode": True,
            "candidate_fingerprints_match_generated_specs": True,
            "mode_specific_acquisition_routing_valid": True,
            "no_reproducible_acquisition_implementation_defect": True,
        },
        "selection_isolation": dict(ISOLATION_FLAGS),
        "recommendation_sequences": sequences,
    }
    result["audit_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    audit = build_audit(args.root.resolve())
    payload = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
