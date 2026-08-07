"""Regression tests for the frozen Pareto outlier audit."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

import pareto_outlier_audit  # noqa: E402


def _mode(fingerprint: str, accuracy: float, latency: float) -> dict:
    return {
        "candidate_fingerprint": fingerprint,
        "accuracy": accuracy,
        "latency_ms": latency,
        "invariant": True,
        "replay_matches": True,
        "order_invariant": True,
    }


def test_expected_cross_mode_compromise_is_classified():
    modes = {
        "accuracy": _mode("a", 0.8, 30.0),
        "latency": _mode("l", 0.7, 10.0),
        "multi_objective": _mode("m", 0.75, 20.0),
    }
    result = {
        "selection_analysis": {
            "algorithm": {"configuration": {"latency_tolerance": 0.5}}
        }
    }
    assert pareto_outlier_audit._classify(modes, result) == (
        "PASS_EXPECTED_COMPROMISE",
        True,
    )


def test_same_fingerprint_is_endpoint_collapse():
    modes = {
        "accuracy": _mode("same", 0.8, 10.0),
        "latency": _mode("l", 0.7, 10.0),
        "multi_objective": _mode("same", 0.79, 10.0),
    }
    result = {
        "selection_analysis": {
            "algorithm": {"configuration": {"latency_tolerance": 0.5}}
        }
    }
    assert pareto_outlier_audit._classify(modes, result)[0] == (
        "PASS_ENDPOINT_COLLAPSE"
    )


def test_cross_job_visual_order_does_not_manufacture_failure():
    modes = {
        "accuracy": _mode("a", 0.8, 30.0),
        "latency": _mode("l", 0.75, 20.0),
        "multi_objective": _mode("m", 0.70, 15.0),
    }
    result = {
        "selection_analysis": {
            "algorithm": {"configuration": {"latency_tolerance": 0.5}}
        }
    }
    assert pareto_outlier_audit._classify(modes, result) == (
        "INCONCLUSIVE",
        False,
    )


def test_any_selector_invariant_failure_is_a_failure():
    modes = {
        "accuracy": _mode("a", 0.8, 30.0),
        "latency": _mode("l", 0.7, 10.0),
        "multi_objective": _mode("m", 0.75, 20.0),
    }
    modes["multi_objective"]["invariant"] = False
    result = {
        "selection_analysis": {
            "algorithm": {"configuration": {"latency_tolerance": 0.5}}
        }
    }
    assert pareto_outlier_audit._classify(modes, result)[0] == "FAIL_SELECTOR"


def test_higher_accuracy_in_an_independent_archive_is_search_coverage_failure():
    modes = {
        "accuracy": _mode("a", 0.8, 30.0),
        "latency": _mode("l", 0.81, 20.0),
        "multi_objective": _mode("m", 0.75, 25.0),
    }
    result = {
        "selection_analysis": {
            "algorithm": {"configuration": {"latency_tolerance": 0.5}}
        }
    }
    assert pareto_outlier_audit._classify(
        modes, result, pooled_accuracy_invariant=False
    )[0] == "FAIL_SEARCH_OR_ARCHIVE"


def test_cross_archive_geometry_does_not_rewrite_independent_selection():
    modes = {
        "accuracy": _mode("a", 0.8, 30.0),
        "latency": _mode("l", 0.7, 10.0),
        "multi_objective": _mode("m", 0.75, 20.0),
    }
    result = {
        "selection_analysis": {
            "algorithm": {"configuration": {"latency_tolerance": 0.5}}
        }
    }
    assert pareto_outlier_audit._classify(modes, result)[0] == (
        "PASS_EXPECTED_COMPROMISE"
    )
