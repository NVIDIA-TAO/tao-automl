#!/usr/bin/env python3

"""Launch the frozen matched cohort with the rec16 checkpoint overlay.

All scheduling, SDK, SQSH, retry, and ledger behavior is delegated to the
byte-identical v1 launcher.  This wrapper changes only the rec16 checkpoint in
the execution projection and binds the immutable overlay into source checks
and every block-plan digest.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import latency_feasible_checkpoint_overlay as checkpoint_overlay
import latency_feasible_matched_launcher as parent_launcher


HERE = Path(__file__).resolve().parent
DEFAULT_OVERLAY = HERE / "latency_feasible_rec16_checkpoint_overlay.v2.json"


class WrapperError(RuntimeError):
    """Raised when the v2 wrapper cannot bind the immutable overlay."""


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


def install_overlay_bindings(
    *,
    overlay_path: Path,
    overlay_sha256: str,
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    original_load_manifest = parent_launcher.load_manifest
    original_validate_sources = parent_launcher.validate_final_source_evidence
    original_generate_configs = parent_launcher.generate_configs
    original_build_plan = parent_launcher.build_block_plan
    original_verify_remote = parent_launcher.verify_remote
    original_launch_sources = parent_launcher.validate_launch_source_state

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

    def load_manifest(
        path: Path,
        supplied_sha256: str,
    ) -> tuple[dict[str, Any], str]:
        parent, parent_actual = original_load_manifest(path, supplied_sha256)
        overlay, overlay_actual = checkpoint_overlay.load_overlay(
            overlay_path,
            overlay_sha256,
            parent,
            path.resolve(),
            parent_actual,
        )
        state.update(
            {
                "parent_manifest": parent,
                "parent_manifest_path": path.resolve(),
                "parent_manifest_sha256": parent_actual,
                "overlay": overlay,
                "overlay_whole_file_sha256": overlay_actual,
            }
        )
        return parent, parent_actual

    def validate_final_source_evidence(
        manifest: dict[str, Any],
        protocol_erratum_path: Path | None = None,
        protocol_erratum_sha256: str | None = None,
    ) -> dict[str, Any]:
        parent, overlay, overlay_actual = ensure_loaded()
        if manifest is not parent:
            parent_launcher.require_equal(
                manifest,
                parent,
                "launcher parent-manifest object",
            )
        checks = original_validate_sources(
            parent,
            protocol_erratum_path=protocol_erratum_path,
            protocol_erratum_sha256=protocol_erratum_sha256,
        )
        return {
            **checks,
            **checkpoint_overlay.overlay_source_checks(
                overlay,
                overlay_path,
                overlay_actual,
            ),
        }

    def generate_configs(manifest: dict[str, Any]) -> dict[str, bytes]:
        parent, overlay, _ = ensure_loaded()
        parent_launcher.require_equal(
            manifest,
            parent,
            "config-generation parent manifest",
        )
        return original_generate_configs(
            checkpoint_overlay.execution_manifest(parent, overlay)
        )

    def build_block_plan(
        manifest: dict[str, Any],
        whole_file_sha256: str,
        allocation: dict[str, Any],
        configs: dict[str, bytes],
    ) -> dict[str, Any]:
        parent, overlay, overlay_actual = ensure_loaded()
        parent_launcher.require_equal(
            manifest,
            parent,
            "block-plan parent manifest",
        )
        plan = original_build_plan(
            checkpoint_overlay.execution_manifest(parent, overlay),
            whole_file_sha256,
            allocation,
            configs,
        )
        return checkpoint_overlay.augment_block_plan(
            plan,
            overlay,
            overlay_actual,
        )

    def verify_remote(manifest: dict[str, Any]) -> dict[str, Any]:
        parent, overlay, _ = ensure_loaded()
        parent_launcher.require_equal(
            manifest,
            parent,
            "remote-verification parent manifest",
        )
        result = original_verify_remote(
            checkpoint_overlay.execution_manifest(parent, overlay)
        )
        result["checkpoint_overlay_applied"] = True
        result["historical_rec16_checkpoint_remote_presence_required"] = False
        return result

    def validate_launch_source_state(
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        parent, overlay, overlay_actual = ensure_loaded()
        parent_launcher.require_equal(
            manifest,
            parent,
            "launch-source parent manifest",
        )
        checks = original_launch_sources(parent)
        checks["checkpoint_overlay"] = (
            checkpoint_overlay.overlay_source_checks(
                overlay,
                overlay_path,
                overlay_actual,
            )
        )
        checks["overlay_wrapper_sources_committed_clean"] = True
        return checks

    parent_launcher.load_manifest = load_manifest
    parent_launcher.validate_final_source_evidence = (
        validate_final_source_evidence
    )
    parent_launcher.generate_configs = generate_configs
    parent_launcher.build_block_plan = build_block_plan
    parent_launcher.verify_remote = verify_remote
    parent_launcher.validate_launch_source_state = validate_launch_source_state
    return state


def main() -> int:
    overlay_args, remaining = parse_overlay_args(sys.argv[1:])
    install_overlay_bindings(
        overlay_path=overlay_args.checkpoint_overlay.resolve(),
        overlay_sha256=overlay_args.checkpoint_overlay_sha256,
    )
    sys.argv = [sys.argv[0], *remaining]
    return parent_launcher.main()


if __name__ == "__main__":
    raise SystemExit(main())
