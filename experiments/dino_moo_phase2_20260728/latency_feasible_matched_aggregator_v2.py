#!/usr/bin/env python3

"""Aggregate matched latency with the immutable rec16 checkpoint overlay."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import Any

import latency_feasible_checkpoint_overlay as checkpoint_overlay
import latency_feasible_matched_aggregator as parent_aggregator
import latency_feasible_matched_launcher as parent_launcher
import latency_feasible_matched_launcher_v2 as launcher_v2


HERE = Path(__file__).resolve().parent
DEFAULT_OVERLAY = HERE / "latency_feasible_rec16_checkpoint_overlay.v2.json"


class WrapperError(RuntimeError):
    """Raised when aggregation is not bound to the exact checkpoint overlay."""


def parse_overlay_args(
    arguments: list[str],
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--checkpoint-overlay",
        type=Path,
        default=DEFAULT_OVERLAY,
    )
    parser.add_argument("--checkpoint-overlay-sha256", required=True)
    return parser.parse_known_args(arguments)


def install_aggregation_bindings(
    *,
    overlay_path: Path,
    overlay_sha256: str,
) -> dict[str, Any]:
    state = launcher_v2.install_overlay_bindings(
        overlay_path=overlay_path,
        overlay_sha256=overlay_sha256,
    )
    overlay_aware_load = parent_launcher.load_manifest
    overlay_aware_validate_sources = (
        parent_launcher.validate_final_source_evidence
    )
    original_build_final_report = parent_aggregator.build_final_report

    def ensure_loaded() -> tuple[dict[str, Any], dict[str, Any], str]:
        parent = state.get("parent_manifest")
        overlay = state.get("overlay")
        actual = state.get("overlay_whole_file_sha256")
        if (
            not isinstance(parent, dict)
            or not isinstance(overlay, dict)
            or not isinstance(actual, str)
        ):
            raise WrapperError("parent manifest and overlay are not loaded")
        return parent, overlay, actual

    def load_execution_manifest(
        path: Path,
        supplied_sha256: str,
    ) -> tuple[dict[str, Any], str]:
        parent, actual = overlay_aware_load(path, supplied_sha256)
        _, overlay, _ = ensure_loaded()
        return checkpoint_overlay.execution_manifest(parent, overlay), actual

    def validate_execution_sources(
        manifest: dict[str, Any],
        protocol_erratum_path: Path | None = None,
        protocol_erratum_sha256: str | None = None,
    ) -> dict[str, Any]:
        parent, overlay, _ = ensure_loaded()
        parent_launcher.require_equal(
            manifest,
            checkpoint_overlay.execution_manifest(parent, overlay),
            "aggregation execution projection",
        )
        return overlay_aware_validate_sources(
            parent,
            protocol_erratum_path=protocol_erratum_path,
            protocol_erratum_sha256=protocol_erratum_sha256,
        )

    def build_final_report(
        manifest: dict[str, Any],
        manifest_file_sha256: str,
        ledger: dict[str, Any],
        ledger_file_sha256: str,
        jobs: list[dict[str, Any]],
        measurements: list[dict[str, Any]],
        consistency: dict[str, Any],
        runtime_provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        parent, overlay, overlay_actual = ensure_loaded()
        parent_launcher.require_equal(
            manifest,
            checkpoint_overlay.execution_manifest(parent, overlay),
            "final-report execution projection",
        )
        report = original_build_final_report(
            manifest,
            manifest_file_sha256,
            ledger,
            ledger_file_sha256,
            jobs,
            measurements,
            consistency,
            runtime_provenance,
        )
        report.pop("report_sha256")
        report["checkpoint_overlay"] = {
            **checkpoint_overlay.overlay_source_checks(
                overlay,
                overlay_path,
                overlay_actual,
            ),
            "parent_manifest_candidate_records_preserved": True,
            "parent_manifest_selection_snapshot_preserved": True,
            "execution_projection_substitution_count": 1,
            "accuracy_or_selection_evidence_from_recovered_checkpoint": False,
        }
        report["parent_manifest_candidate_checkpoint_evidence"] = {
            candidate["candidate_id"]: copy.deepcopy(candidate["checkpoint"])
            for candidate in parent["candidates"]
        }
        report["selection_isolation"].update(
            {
                "selector_invoked_on_matched_measurements": False,
                "selection_time_objectives_replaced": False,
                "measurements_feed_selection": False,
                "measurements_feed_reselection": False,
                "algorithm_selected_candidate_overridden": False,
            }
        )
        report["report_sha256"] = (
            parent_aggregator.manifest_generator.sha256_value(report)
        )
        return report

    parent_launcher.load_manifest = load_execution_manifest
    parent_launcher.validate_final_source_evidence = (
        validate_execution_sources
    )
    parent_aggregator.build_final_report = build_final_report
    return state


def main() -> int:
    overlay_args, remaining = parse_overlay_args(sys.argv[1:])
    install_aggregation_bindings(
        overlay_path=overlay_args.checkpoint_overlay.resolve(),
        overlay_sha256=overlay_args.checkpoint_overlay_sha256,
    )
    sys.argv = [sys.argv[0], *remaining]
    return parent_aggregator.main()


if __name__ == "__main__":
    raise SystemExit(main())
