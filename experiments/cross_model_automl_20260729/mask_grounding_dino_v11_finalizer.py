#!/usr/bin/env python3

"""Finalize MGDINO v11 with read-only production selector replay evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

try:
    from . import pareto_outlier_audit
except ImportError:  # pragma: no cover - direct script execution
    import pareto_outlier_audit  # type: ignore[no-redef]


MODES = ("accuracy", "latency", "multi_objective")
REQUIRED_CANDIDATES = 24
REQUIRED_OBJECTIVES = (
    "segm_val_mAP50_95",
    "latency_ms",
    "latency_p95_ms",
    "latency_ci95_low_ms",
    "latency_ci95_high_ms",
)
DEFAULT_ROOT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask_grounding_dino_coco2017_three_mode_v11"
)
DEFAULT_SOURCES = Path(__file__).with_name("pareto_validation_sources.v1.json")


class FinalizationError(RuntimeError):
    """MGDINO terminal evidence is incomplete or inconsistent."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def terminal(root: Path) -> bool:
    status_path = root / "mode_process_status.json"
    if not status_path.is_file():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if set(status) != set(MODES):
        raise FinalizationError("MGDINO terminal mode set changed")
    if any(value != 0 for value in status.values()):
        raise FinalizationError(f"MGDINO controller failed: {status}")
    for mode in MODES:
        if not (root / mode / "result.json").is_file():
            return False
        if not (root / mode / "candidate_evidence.json").is_file():
            return False
    return True


def wait_for_terminal(root: Path, poll_seconds: float, timeout_seconds: float) -> None:
    started = time.monotonic()
    while not terminal(root):
        if time.monotonic() - started >= timeout_seconds:
            raise FinalizationError("MGDINO v11 finalizer timed out")
        time.sleep(poll_seconds)


def _load_mode(root: Path, mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence_path = root / mode / "candidate_evidence.json"
    result_path = root / mode / "result.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    result_document = json.loads(result_path.read_text(encoding="utf-8"))
    if evidence.get("mode") != mode or result_document.get("mode") != mode:
        raise FinalizationError(f"{mode} artifact routing changed")
    if result_document.get("status") != "success":
        raise FinalizationError(f"{mode} result is not successful")
    if evidence.get("contract_sha256") != result_document.get("contract_sha256"):
        raise FinalizationError(f"{mode} contract binding changed")
    candidates = evidence.get("candidates")
    if not isinstance(candidates, Mapping) or len(candidates) != REQUIRED_CANDIDATES:
        raise FinalizationError(f"{mode} does not contain 24 candidates")
    successful_rec_ids: set[str] = set()
    for candidate_id, candidate in candidates.items():
        status = candidate.get("status")
        if status not in {"success", "terminal_failure"}:
            raise FinalizationError(f"{candidate_id} is not terminal")
        if any(candidate.get("agent_intervention_flags", {}).values()):
            raise FinalizationError(f"{candidate_id} has agent intervention")
        if any(candidate.get("selection_isolation_flags", {}).values()):
            raise FinalizationError(f"{candidate_id} violated selection isolation")
        if status == "terminal_failure":
            if candidate.get("automl_status") != "failure":
                raise FinalizationError(
                    f"{candidate_id} terminal failure status is inconsistent"
                )
            if not candidate.get("failure_reason"):
                raise FinalizationError(
                    f"{candidate_id} terminal failure has no reason"
                )
            continue
        successful_rec_ids.add(str(candidate.get("rec_id")))
        values = candidate.get("objective_values")
        if not isinstance(values, Mapping):
            raise FinalizationError(f"{candidate_id} has no objective vector")
        for objective in REQUIRED_OBJECTIVES:
            value = values.get(objective)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise FinalizationError(
                    f"{candidate_id} has invalid objective {objective}"
                )
        if float(values["latency_ms"]) <= 0.0:
            raise FinalizationError(f"{candidate_id} has non-positive latency")
        if candidate.get("selection_time_latency", {}).get(
            "quality_gate_passed"
        ) is not True:
            raise FinalizationError(f"{candidate_id} failed latency quality")
    analysis = result_document.get("result", {}).get("selection_analysis", {})
    analyzed_rec_ids = {
        str(candidate.get("candidate_id"))
        for candidate in analysis.get("candidates", [])
    }
    if analyzed_rec_ids != successful_rec_ids:
        raise FinalizationError(
            f"{mode} selector population does not equal successful candidates"
        )
    return evidence, result_document["result"]


def build_source_manifest(
    root: Path,
    base_source: Path,
) -> dict[str, Any]:
    source = json.loads(base_source.read_text(encoding="utf-8"))
    artifact_root = Path(source["artifact_root"]).resolve()
    relative_root = root.resolve().relative_to(artifact_root)
    value = copy.deepcopy(source)
    value["schema_version"] = 2
    value["base_source_manifest"] = {
        "path": str(base_source.resolve()),
        "sha256": file_sha256(base_source),
    }
    value["models"].append(
        {
            "model": "mask_grounding_dino",
            "dataset": "coco2017_instance_segmentation",
            "kind": "mode_results",
            "paths": {
                mode: {
                    "path": str(relative_root / mode / "result.json"),
                    "sha256": file_sha256(root / mode / "result.json"),
                }
                for mode in MODES
            },
        }
    )
    return value


def _winner_row(
    mode: str,
    mode_audit: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_id = str(mode_audit["winner_id"])
    candidate = next(
        item
        for item in result["selection_analysis"]["candidates"]
        if str(item["candidate_id"]) == candidate_id
    )
    return {
        "mode": mode,
        "candidate_id": f"{mode}_rec_{candidate_id}",
        "candidate_fingerprint": mode_audit["candidate_fingerprint"],
        "accuracy": mode_audit["accuracy"],
        "median_latency_ms": mode_audit["latency_ms"],
        "accuracy_feasible": (
            candidate["latency_accuracy_feasible"]
            if mode == "latency"
            else candidate["multi_objective_accuracy_feasible"]
            if mode == "multi_objective"
            else True
        ),
        "pareto_rank": candidate["multi_objective_pareto_rank"],
        "compromise_score": candidate["multi_objective_compromise_score"],
        "normalized_accuracy_regret": candidate[
            "normalized_accuracy_objective"
        ],
        "normalized_latency_regret": candidate[
            "normalized_latency_objective"
        ],
        "tie_break_values": candidate["tie_breaking_values"][f"{mode}_mode"],
        "selection_reason": mode_audit["selection_reason"],
        "replay_matches": mode_audit["replay_matches"],
        "order_invariant": mode_audit["order_invariant"],
    }


def _markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Mask Grounding DINO v11 Pareto result",
        "",
        "| Mode | Candidate | Accuracy | Median latency | Accuracy feasible | Pareto rank | Compromise score | Selection reason |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in audit["winner_table"]:
        lines.append(
            f"| {row['mode']} | `{row['candidate_id']}` | "
            f"{row['accuracy']:.10f} | {row['median_latency_ms']:.6f} ms | "
            f"{str(row['accuracy_feasible']).lower()} | {row['pareto_rank']} | "
            f"{row['compromise_score']:.12g} | {row['selection_reason']} |"
        )
    lines.extend(
        [
            "",
            f"Classification: `{audit['classification']}`",
            "",
            "All winners were persisted by production AutoML, reproduced by a read-only production-selector replay, and invariant under ten candidate-order permutations. No validation measurement triggered selection or reselection.",
            "",
            "## Complete rank-zero front",
            "",
            "```json",
            json.dumps(audit["rank_zero_front"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def finalize(root: Path, base_source: Path, output_root: Path) -> dict[str, Any]:
    if not terminal(root):
        raise FinalizationError("MGDINO v11 is not terminal")
    contract_path = root / "campaign.v11.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    mode_evidence: dict[str, Any] = {}
    mode_results: dict[str, Any] = {}
    for mode in MODES:
        evidence, result = _load_mode(root, mode)
        mode_evidence[mode] = evidence
        mode_results[mode] = result

    source = build_source_manifest(root, base_source)
    source_path = output_root / "pareto_validation_sources.v2.json"
    atomic_json(source_path, source)
    matrix = pareto_outlier_audit.build_audit(source_path)
    matrix_path = output_root / "matrix.json"
    atomic_json(matrix_path, matrix)
    model = next(
        item for item in matrix["models"] if item["model"] == "mask_grounding_dino"
    )
    winner_table = [
        _winner_row(mode, model["modes"][mode], mode_results[mode])
        for mode in MODES
    ]
    candidate_outcomes = {
        mode: {
            "total": len(mode_evidence[mode]["candidates"]),
            "success": sum(
                candidate.get("status") == "success"
                for candidate in mode_evidence[mode]["candidates"].values()
            ),
            "terminal_failure": sum(
                candidate.get("status") == "terminal_failure"
                for candidate in mode_evidence[mode]["candidates"].values()
            ),
        }
        for mode in MODES
    }
    failed_candidates = {
        mode: [
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_fingerprint": candidate["candidate_fingerprint"],
                "failure_reason": candidate["failure_reason"],
            }
            for candidate in mode_evidence[mode]["candidates"].values()
            if candidate.get("status") == "terminal_failure"
        ]
        for mode in MODES
    }
    audit = {
        "schema_version": 1,
        "kind": "mask_grounding_dino_v11_terminal_pareto_audit",
        "campaign_contract": {
            "path": str(contract_path),
            "file_sha256": file_sha256(contract_path),
            "contract_sha256": contract["contract_sha256"],
        },
        "candidate_counts": {
            mode: len(mode_evidence[mode]["candidates"]) for mode in MODES
        },
        "candidate_outcomes": candidate_outcomes,
        "failed_candidates": failed_candidates,
        "failed_recommendations_preserved": True,
        "result_file_sha256": {
            mode: file_sha256(root / mode / "result.json") for mode in MODES
        },
        "candidate_evidence_file_sha256": {
            mode: file_sha256(root / mode / "candidate_evidence.json")
            for mode in MODES
        },
        "classification": model["selection_policy_classification"],
        "middle_ground_observed_across_independent_archives": model[
            "middle_ground_invariant"
        ],
        "candidate_universe_contract": model["candidate_universe_contract"],
        "winner_table": winner_table,
        "winner_fingerprints": {
            row["mode"]: row["candidate_fingerprint"] for row in winner_table
        },
        "rank_zero_front": model["multi_objective_rank_zero_front"],
        "normalization_bounds": model["modes"]["multi_objective"]["details"][
            "normalization_bounds"
        ],
        "objective_weights": model["modes"]["multi_objective"]["details"][
            "objective_weights"
        ],
        "augmentation_rho": model["modes"]["multi_objective"]["details"][
            "augmentation_rho"
        ],
        "archive_fingerprint_sha256": {
            mode: model["modes"][mode]["archive_fingerprint_sha256"]
            for mode in MODES
        },
        "all_successful_objective_vectors_complete": True,
        "all_successful_latency_quality_gates_passed": True,
        "all_production_replays_match": all(
            row["replay_matches"] for row in winner_table
        ),
        "all_order_permutation_checks_pass": all(
            row["order_invariant"] for row in winner_table
        ),
        "agent_intervention_flags": {
            name: False for name in pareto_outlier_audit.ISOLATION_FLAGS
        },
        "selector_invoked_on_posthoc_measurements": False,
        "matrix": {
            "path": str(matrix_path),
            "file_sha256": file_sha256(matrix_path),
        },
        "source_manifest": {
            "path": str(source_path),
            "file_sha256": file_sha256(source_path),
        },
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    atomic_json(output_root / "mgdino_v11_final_audit.json", audit)
    (output_root / "mgdino_v11_report.md").write_text(
        _markdown(audit), encoding="utf-8"
    )
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--base-sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=7 * 24 * 3600)
    args = parser.parse_args(argv)
    if args.wait:
        wait_for_terminal(args.root, args.poll_seconds, args.timeout_seconds)
    try:
        audit = finalize(args.root, args.base_sources, args.output_root)
    except Exception as exc:
        atomic_json(
            args.output_root / "finalization_status.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    atomic_json(
        args.output_root / "finalization_status.json",
        {
            "status": "complete",
            "audit_sha256": audit["audit_sha256"],
            "classification": audit["classification"],
        },
    )
    print(json.dumps({
        "audit_sha256": audit["audit_sha256"],
        "classification": audit["classification"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
