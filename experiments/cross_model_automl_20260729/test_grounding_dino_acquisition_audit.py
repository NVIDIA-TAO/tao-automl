"""Tests for the read-only Grounding DINO acquisition audit."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))

import grounding_dino_acquisition_audit as audit  # noqa: E402


@pytest.mark.parametrize(
    ("value", "declaration", "expected"),
    [
        (3, {"valid_options": [3, 4, 5, 6]}, True),
        (7, {"valid_options": [3, 4, 5, 6]}, False),
        (0.5, {"valid_min": 0.1, "valid_max": 0.9}, True),
        (1.0, {"valid_min": 0.1, "valid_max": 0.9}, False),
    ],
)
def test_frozen_range_validation(value, declaration, expected):
    assert audit._validate_range(value, declaration) is expected


def test_expected_objective_directions_are_mode_specific():
    assert audit.EXPECTED_OBJECTIVES == {
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


def test_validation_measurements_are_isolated_from_selection():
    assert audit.ISOLATION_FLAGS
    assert set(audit.ISOLATION_FLAGS.values()) == {False}
    assert {
        "measurements_feed_selection",
        "measurements_feed_reselection",
        "selection_time_objectives_replaced",
        "algorithm_selected_candidate_overridden",
        "posthoc_measurements_feed_selection",
        "posthoc_measurements_feed_reselection",
        "historical_winner_overridden",
    } <= set(audit.ISOLATION_FLAGS)
