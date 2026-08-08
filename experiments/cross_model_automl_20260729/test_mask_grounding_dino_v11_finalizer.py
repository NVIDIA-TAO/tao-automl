from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))

import mask_grounding_dino_v11_finalizer as finalizer  # noqa: E402


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
