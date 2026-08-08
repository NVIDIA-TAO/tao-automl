from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))

import grounding_dino_matched_accuracy_validation as matched  # noqa: E402


def _contract() -> dict:
    labels = (
        "accuracy_archive_winner",
        "higher_distinct_external_candidate",
    )
    schedule = []
    for repeat in range(6):
        order = labels if repeat % 2 == 0 else tuple(reversed(labels))
        for position, label in enumerate(order):
            schedule.append(
                {
                    "cell_id": f"repeat_{repeat}_{label}",
                    "repeat": repeat,
                    "position": position,
                    "specification_label": label,
                }
            )
    value = {
        "design": {
            "repeat_count": 6,
            "schedule": schedule,
            "bootstrap_resamples": 1000,
            "bootstrap_seed": 7,
            "confidence_level": 0.95,
        },
        "specifications": {label: {} for label in labels},
        "validation_isolation": copy.deepcopy(matched.VALIDATION_FLAGS),
    }
    value["contract_sha256"] = matched.canonical_sha256(value)
    return value


def test_schedule_is_position_balanced() -> None:
    contract = _contract()
    schedule = contract["design"]["schedule"]
    for label in contract["specifications"]:
        assert sorted(
            cell["position"]
            for cell in schedule
            if cell["specification_label"] == label
        ) == [0, 0, 0, 1, 1, 1]


def test_validation_flags_prevent_selection_feedback() -> None:
    assert matched.VALIDATION_FLAGS["measurements_feed_selection"] is False
    assert matched.VALIDATION_FLAGS["measurements_feed_reselection"] is False
    assert matched.VALIDATION_FLAGS["historical_winner_overridden"] is False
    assert matched.VALIDATION_FLAGS["agent_selected_candidate"] is False


def test_analysis_classifies_stable_positive_direction() -> None:
    contract = _contract()
    state = {"cells": {}}
    for cell in contract["design"]["schedule"]:
        base = 0.70 + 0.001 * cell["repeat"]
        metric = (
            base + 0.02
            if cell["specification_label"]
            == "higher_distinct_external_candidate"
            else base
        )
        state["cells"][cell["cell_id"]] = {"mAP50": metric}
    result = matched.analyze_completed(contract, state)
    assert result["classification"] == "HIGHER_FINGERPRINT_REPRODUCIBLY_BETTER"
    assert result["selector_invoked"] is False
    assert result["median_paired_difference"] == pytest.approx(0.02)


def test_analysis_keeps_overlapping_direction_unresolved() -> None:
    contract = _contract()
    differences = [0.01, -0.01, 0.02, -0.02, 0.005, -0.005]
    state = {"cells": {}}
    for cell in contract["design"]["schedule"]:
        base = 0.70
        difference = differences[cell["repeat"]]
        metric = (
            base + difference
            if cell["specification_label"]
            == "higher_distinct_external_candidate"
            else base
        )
        state["cells"][cell["cell_id"]] = {"mAP50": metric}
    result = matched.analyze_completed(contract, state)
    assert result["classification"] == "DIRECTION_UNRESOLVED_OR_TRAINING_NOISE"


def test_contract_digest_tampering_is_rejected() -> None:
    contract = _contract()
    contract["design"]["repeat_count"] = 5
    with pytest.raises(matched.MatchedAccuracyError, match="digest changed"):
        matched.validate_contract(contract)


def test_mgdino_gate_uses_controller_terminal_artifacts(tmp_path: Path) -> None:
    campaign_path = tmp_path / "campaign.v11.json"
    campaign_path.write_text("{}")
    gate_contract = {
        "prerequisite_gate": {
            "root": str(tmp_path),
            "campaign_contract_path": str(campaign_path),
            "campaign_contract_file_sha256": matched.file_sha256(campaign_path),
            "required_candidates_per_mode": 24,
            "required_modes": ["accuracy", "latency", "multi_objective"],
        }
    }
    assert matched._mgdino_terminal(gate_contract) is False
    (tmp_path / "mode_process_status.json").write_text(
        json.dumps({
            "accuracy": 0,
            "latency": 0,
            "multi_objective": 0,
        })
    )
    for mode in gate_contract["prerequisite_gate"]["required_modes"]:
        mode_root = tmp_path / mode
        mode_root.mkdir()
        (mode_root / "candidate_evidence.json").write_text(
            json.dumps({
                "candidates": {str(index): {} for index in range(24)}
            })
        )
        (mode_root / "result.json").write_text(
            json.dumps({"mode": mode, "status": "success"})
        )
    assert matched._mgdino_terminal(gate_contract) is True
