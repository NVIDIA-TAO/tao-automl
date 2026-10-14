# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI shared by the generic command consumer and repository-native adapters."""

import argparse
import json
import subprocess
from pathlib import Path

from automl_core.contracts import Contract, Execution, Search
from automl_core.controller import Session, review
from automl_core.runner import Experiment, manage
from automl_core.workload import CommandWorkload, read_manifest


class Manifest(Contract):
    search: Search
    workload: CommandWorkload
    execution: Execution


def log_paths(result):
    if "batches" in result:
        return {
            batch["id"]: {
                trial["id"]: trial["log"]
                for trial in batch.get("native", {}).get("trials", [])
            }
            for batch in result["batches"]
        }
    return {trial["id"]: trial["log"] for trial in result["trials"]}


def load(path, workspace):
    manifest = Manifest.model_validate(read_manifest(path))
    manifest.workload.cwd = (path.parent / manifest.workload.cwd).resolve()
    manifest.execution.mounts = [
        (path.parent / p).resolve() for p in manifest.execution.mounts
    ]
    return Experiment(manifest.search, manifest.workload, manifest.execution, workspace)


def main(loader=load, *, prog=None, description=None, epilog=None):
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description
        or "Bounded AutoML: explicit workload, objective and execution",
        epilog=epilog,
    )
    parser.add_argument(
        "action",
        choices=["plan", "run", "resume", "migrate", "status", "logs", "cancel"],
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--from-wheel", type=Path, help="Original exact wheel for cold migration review"
    )
    parser.add_argument(
        "--acknowledge-cancel",
        action="store_true",
        help="Explicitly archive the old stop marker during migration",
    )
    parser.add_argument(
        "--experiment-id", help="Manage an existing run without its source manifest"
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--controller-max-allocations",
        type=int,
        help="Opt in to bounded SLURM CPU handoff; includes initial allocation",
    )
    parser.add_argument(
        "--controller-allocation-seconds",
        type=int,
        help="Reviewed maximum wall time for each CPU controller allocation",
    )
    parser.add_argument(
        "--controller-grace-seconds",
        type=int,
        default=None,
        help="Checkpoint/requeue allowance before CPU wall time (default: 300)",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve the reviewed bounded execution or cancellation",
    )
    args = parser.parse_args()
    if args.action in {"run", "resume", "cancel"} and not args.approve:
        parser.error("Review plan and supply --approve for this side-effecting action")
    if args.action in {"plan", "run", "resume", "migrate"} and (
        not args.manifest or args.experiment_id
    ):
        parser.error("plan/run/resume/migrate require --manifest, not --experiment-id")
    if (args.action == "migrate") != bool(args.from_wheel):
        parser.error("migrate requires --from-wheel; other actions do not accept it")
    if args.acknowledge_cancel and args.action != "migrate":
        parser.error("--acknowledge-cancel applies only to migrate")
    if bool(args.manifest) == bool(args.experiment_id):
        parser.error("Provide exactly one of --manifest or --experiment-id")
    if bool(args.controller_max_allocations is not None) != bool(
        args.controller_allocation_seconds is not None
    ):
        parser.error("Controller handoff needs both allocation count and seconds")
    if (
        args.controller_grace_seconds is not None
        and args.controller_max_allocations is None
    ):
        parser.error(
            "Controller grace requires an explicit allocation count and seconds"
        )
    if args.controller_max_allocations is not None and args.action not in {
        "plan",
        "run",
        "resume",
    }:
        parser.error("Controller allocation options apply only to plan/run/resume")
    try:
        experiment = (
            loader(args.manifest.resolve(), args.workspace) if args.manifest else None
        )
        policy = None
        if args.controller_max_allocations is not None:
            policy = review(
                args.controller_max_allocations,
                args.controller_allocation_seconds,
                args.controller_grace_seconds
                if args.controller_grace_seconds is not None
                else 300,
            )
            if experiment.execution.kind != "slurm":
                raise ValueError("Controller handoff requires SLURM execution")
            settings = (
                experiment.search
                if hasattr(experiment, "search")
                else experiment.settings
            )
            if (
                settings.llm
                and policy["grace_seconds"] < settings.llm.timeout_seconds + 60
            ):
                raise ValueError(
                    "Controller grace must cover an in-flight LLM request plus 60 seconds"
                )
        if args.experiment_id:
            result = manage(
                args.workspace, args.experiment_id, cancel=args.action == "cancel"
            )
            if args.action == "logs":
                result = log_paths(result)
        elif args.action == "migrate":
            from automl_core.migration import migrate

            result = migrate(
                experiment,
                args.from_wheel,
                approve=args.approve,
                acknowledge_cancel=args.acknowledge_cancel,
            )
        elif args.action == "plan":
            result = experiment.plan()
            if policy:
                result["controller_review"] = policy
        elif args.action in {"run", "resume"}:
            result = (Session(experiment, policy) if policy else experiment).run(
                resume=args.action == "resume"
            )
        elif args.action == "cancel":
            result = experiment.cancel()
        else:
            result = experiment.status()
            if args.action == "logs":
                result = log_paths(result)
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
