# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the real-data TAO 7.1 DINO PTM train/validation qualification gate.

This is deliberately separate from the CPU checkpoint-compatibility report.
Both reports are qualification-only and neither mutates the repository PTM
registry. A checkpoint may be promoted to runtime-supported only after it
appears in the prepared inventory of both immutable reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:
    from .qualification_driver import (
        DINOQualificationConfiguration,
        load_verified_qualification_completion,
        run_dino_ptm_qualification,
    )
except ImportError:  # Direct execution from this directory.
    from qualification_driver import (
        DINOQualificationConfiguration,
        load_verified_qualification_completion,
        run_dino_ptm_qualification,
    )

try:
    from ..dino_preflight.dino_local_launch import (
        DINOStandardDryRunLoadSmoke,
        _verify_local_runtime_image,
    )
    from ..dino_preflight.dino_preflight import (
        collect_voc_real_data_integrity,
    )
except ImportError:  # Direct execution from this directory.
    import sys

    _PREFLIGHT_DIR = Path(__file__).resolve().parents[1] / "dino_preflight"
    if str(_PREFLIGHT_DIR) not in sys.path:
        sys.path.insert(0, str(_PREFLIGHT_DIR))
    from dino_local_launch import (  # type: ignore[no-redef]
        DINOStandardDryRunLoadSmoke,
        _verify_local_runtime_image,
    )
    from dino_preflight import (  # type: ignore[no-redef]
        collect_voc_real_data_integrity,
    )


def run_train_validation_qualification(args: argparse.Namespace) -> dict:
    """Execute or byte-verify the create-only GPU qualification report."""
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    runtime_results_dir = (
        Path(args.runtime_results_dir).expanduser().resolve()
    )
    upstream = load_verified_qualification_completion(
        output_dir=args.cpu_qualification_dir,
        cache_dir=args.cpu_cache_dir,
    )
    checkpoint_ids = tuple(sorted(
        item["checkpoint_id"]
        for item in upstream["report"]["prepared"]
    ))
    if not checkpoint_ids:
        raise RuntimeError(
            "CPU qualification produced no checkpoint eligible for the "
            "train/validation gate"
        )
    configuration = DINOQualificationConfiguration(
        checkpoint_ids=checkpoint_ids,
        upstream_completion_sha256=upstream["completion_sha256"],
    )
    if not args.resume:
        _verify_local_runtime_image()
        runtime_results_dir.mkdir(parents=True, exist_ok=True)
        smoke_sdk = None
        entrypoint_builder = None
    else:
        # The completed resume reconstructs only the callback identity needed
        # for byte comparison. The driver returns before invoking either
        # placeholder.
        smoke_sdk = object()
        entrypoint_builder = object()
    voc = collect_voc_real_data_integrity(
        manifest_path=args.voc_manifest,
        dataset_root=args.voc_root,
    )
    smoke = DINOStandardDryRunLoadSmoke(
        voc=voc,
        cache_root=cache_dir,
        results_root=runtime_results_dir,
        seed=args.seed,
        container_user=args.container_user,
        poll_interval_seconds=args.poll_interval_seconds,
        max_polls=args.max_polls,
        sdk=smoke_sdk,
        entrypoint_builder=entrypoint_builder,
    )

    return run_dino_ptm_qualification(
        output_dir=output_dir,
        cache_dir=cache_dir,
        docker_load_smoke=smoke,
        configuration=configuration,
        registry_path=args.registry_path,
        resume=args.resume,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one real train and validation batch for each registered DINO "
            "PTM in the exact TAO 7.1 image. NGC_KEY is read only from the "
            "environment by production preflight."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--runtime-results-dir", required=True)
    parser.add_argument("--cpu-qualification-dir", required=True)
    parser.add_argument("--cpu-cache-dir", required=True)
    parser.add_argument("--voc-manifest", required=True)
    parser.add_argument("--voc-root", required=True)
    parser.add_argument("--container-user", required=True)
    parser.add_argument("--registry-path")
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--max-polls", type=int, default=720)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    completion = run_train_validation_qualification(arguments)
    report = completion["report"]
    print(json.dumps({
        "completion_sha256": completion["completion_sha256"],
        "manifest_sha256": completion["manifest_sha256"],
        "prepared_checkpoint_ids": [
            item["checkpoint_id"] for item in report["prepared"]
        ],
        "excluded_checkpoint_ids": [
            item["checkpoint_id"] for item in report["exclusions"]
        ],
    }, sort_keys=True, separators=(",", ":")))
    return 0 if report["prepared"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
