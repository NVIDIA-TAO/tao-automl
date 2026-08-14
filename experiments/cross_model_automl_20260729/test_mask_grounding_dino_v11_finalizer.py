from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))

import mask_grounding_dino_v11_finalizer as finalizer  # noqa: E402


def _successful_candidate(rec_id: int) -> dict:
    return {
        "candidate_id": f"accuracy_rec_{rec_id}",
        "rec_id": str(rec_id),
        "status": "success",
        "automl_status": "success",
        "objective_values": {
            "segm_val_mAP50_95": 0.5,
            "latency_ms": 10.0,
            "latency_p95_ms": 11.0,
            "latency_ci95_low_ms": 9.5,
            "latency_ci95_high_ms": 10.5,
        },
        "selection_time_latency": {"quality_gate_passed": True},
        "agent_intervention_flags": {},
        "selection_isolation_flags": {},
    }


def test_terminal_requires_all_successful_mode_processes(tmp_path: Path) -> None:
    assert finalizer.terminal(tmp_path) is False
    (tmp_path / "mode_process_status.json").write_text(
        json.dumps({mode: 0 for mode in finalizer.MODES})
    )
    for mode in finalizer.MODES:
        mode_root = tmp_path / mode
        mode_root.mkdir()
        (mode_root / "result.json").write_text("{}")
        (mode_root / "candidate_evidence.json").write_text("{}")
    assert finalizer.terminal(tmp_path) is True


def test_terminal_rejects_failed_mode(tmp_path: Path) -> None:
    (tmp_path / "mode_process_status.json").write_text(
        json.dumps({"accuracy": 0, "latency": 1, "multi_objective": 0})
    )
    with pytest.raises(finalizer.FinalizationError, match="controller failed"):
        finalizer.terminal(tmp_path)


def test_load_mode_preserves_terminal_failure_and_excludes_it_from_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(finalizer, "REQUIRED_CANDIDATES", 2)
    mode_root = tmp_path / "accuracy"
    mode_root.mkdir()
    successful = _successful_candidate(0)
    failed = {
        "candidate_id": "accuracy_rec_1",
        "rec_id": "1",
        "status": "terminal_failure",
        "automl_status": "failure",
        "failure_reason": "job_canceled",
        "candidate_fingerprint": "failed-fingerprint",
        "agent_intervention_flags": {},
        "selection_isolation_flags": {},
    }
    evidence = {
        "mode": "accuracy",
        "contract_sha256": "contract",
        "candidates": {"accuracy_rec_0": successful, "accuracy_rec_1": failed},
    }
    result = {
        "mode": "accuracy",
        "status": "success",
        "contract_sha256": "contract",
        "result": {
            "selection_analysis": {"candidates": [{"candidate_id": "0"}]}
        },
    }
    (mode_root / "candidate_evidence.json").write_text(json.dumps(evidence))
    (mode_root / "result.json").write_text(json.dumps(result))

    loaded_evidence, _ = finalizer._load_mode(tmp_path, "accuracy")

    assert loaded_evidence["candidates"]["accuracy_rec_1"] == failed


def test_load_mode_rejects_failed_candidate_in_selector_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(finalizer, "REQUIRED_CANDIDATES", 2)
    mode_root = tmp_path / "accuracy"
    mode_root.mkdir()
    successful = _successful_candidate(0)
    failed = {
        "candidate_id": "accuracy_rec_1",
        "rec_id": "1",
        "status": "terminal_failure",
        "automl_status": "failure",
        "failure_reason": "job_canceled",
        "agent_intervention_flags": {},
        "selection_isolation_flags": {},
    }
    evidence = {
        "mode": "accuracy",
        "contract_sha256": "contract",
        "candidates": {"accuracy_rec_0": successful, "accuracy_rec_1": failed},
    }
    result = {
        "mode": "accuracy",
        "status": "success",
        "contract_sha256": "contract",
        "result": {
            "selection_analysis": {
                "candidates": [{"candidate_id": "0"}, {"candidate_id": "1"}]
            }
        },
    }
    (mode_root / "candidate_evidence.json").write_text(json.dumps(evidence))
    (mode_root / "result.json").write_text(json.dumps(result))

    with pytest.raises(finalizer.FinalizationError, match="selector population"):
        finalizer._load_mode(tmp_path, "accuracy")


def test_winner_row_preserves_production_geometry() -> None:
    result = {
        "selection_analysis": {
            "candidates": [
                {
                    "candidate_id": "2",
                    "latency_accuracy_feasible": True,
                    "multi_objective_accuracy_feasible": True,
                    "multi_objective_pareto_rank": 0,
                    "multi_objective_compromise_score": 0.2,
                    "normalized_accuracy_objective": 0.3,
                    "normalized_latency_objective": 0.4,
                    "tie_breaking_values": {
                        "multi_objective_mode": [0.2, "f", "2"]
                    },
                }
            ]
        }
    }
    mode = {
        "winner_id": "2",
        "candidate_fingerprint": "f",
        "accuracy": 0.7,
        "latency_ms": 10.0,
        "selection_reason": "production reason",
        "replay_matches": True,
        "order_invariant": True,
    }
    row = finalizer._winner_row("multi_objective", mode, result)
    assert row["candidate_id"] == "multi_objective_rec_2"
    assert row["pareto_rank"] == 0
    assert row["compromise_score"] == 0.2
    assert row["selection_reason"] == "production reason"
