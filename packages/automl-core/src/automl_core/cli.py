# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI shared by the generic command consumer and repository-native adapters."""

import argparse
import json
import subprocess
from pathlib import Path

from automl_core.contracts import Contract, Execution, Search
from automl_core.runner import Experiment, manage
from automl_core.workload import CommandWorkload, read_manifest


class Manifest(Contract):
    search: Search
    workload: CommandWorkload
    execution: Execution


def load(path, workspace):
    manifest = Manifest.model_validate(read_manifest(path))
    manifest.workload.cwd = (path.parent / manifest.workload.cwd).resolve()
    manifest.execution.mounts = [
        (path.parent / p).resolve() for p in manifest.execution.mounts
    ]
    return Experiment(manifest.search, manifest.workload, manifest.execution, workspace)


def main(loader=load):
    parser = argparse.ArgumentParser(
        description="Bounded AutoML: explicit workload, objective and execution"
    )
    parser.add_argument(
        "action", choices=["plan", "run", "resume", "status", "logs", "cancel"]
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--experiment-id", help="Manage an existing run without its source manifest"
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve the reviewed bounded execution or cancellation",
    )
    args = parser.parse_args()
    if args.action in {"run", "resume", "cancel"} and not args.approve:
        parser.error("Review plan and supply --approve for this side-effecting action")
    if args.action in {"plan", "run", "resume"} and (
        not args.manifest or args.experiment_id
    ):
        parser.error("plan/run/resume require --manifest, not --experiment-id")
    if bool(args.manifest) == bool(args.experiment_id):
        parser.error("Provide exactly one of --manifest or --experiment-id")
    try:
        experiment = (
            loader(args.manifest.resolve(), args.workspace) if args.manifest else None
        )
        if args.experiment_id:
            result = manage(
                args.workspace, args.experiment_id, cancel=args.action == "cancel"
            )
            if args.action == "logs":
                result = {trial["id"]: trial["log"] for trial in result["trials"]}
        elif args.action == "plan":
            result = experiment.plan()
        elif args.action in {"run", "resume"}:
            result = experiment.run(resume=args.action == "resume")
        elif args.action == "cancel":
            result = experiment.cancel()
        else:
            result = experiment.status()
            if args.action == "logs":
                result = {trial["id"]: trial["log"] for trial in result["trials"]}
        print(json.dumps(result, indent=2, allow_nan=False))
        if args.action in {"run", "resume"} and result["state"] == "ERROR":
            raise SystemExit(1)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        parser.exit(
            2,
            f"Execution preflight/submission failed ({type(exc).__name__}); inspect the local environment and job status.\n",
        )
    except (OSError, ValueError, KeyError) as exc:
        # Validation errors are useful; launch commands and secret env values are never logged here.
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
