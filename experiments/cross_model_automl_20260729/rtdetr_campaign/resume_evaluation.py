#!/usr/bin/env python3

"""Resume only RT-DETR evaluations after the checkpoint-name correction.

The original training workflow and aggregate completion artifacts are
immutable inputs. This recovery path verifies each completed train job,
resolves the exact RT-DETR checkpoint, submits only standalone evaluation,
and writes separate resume artifacts.
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from . import manifest_generator
    from . import run_campaign
except ImportError:  # pragma: no cover - direct script execution
    import manifest_generator  # type: ignore[no-redef]
    import run_campaign  # type: ignore[no-redef]


DEFAULT_RUNTIME_ROOT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/rtdetr_qualification_20260730"
)
CampaignExecutionError = run_campaign.CampaignExecutionError


def _canonical_file_sha(path: Path) -> str:
    if not path.is_file():
        raise CampaignExecutionError(f"resume source is unavailable: {path}")
    return manifest_generator.sha256_file(path)


def validate_prior_completion(
    manifest: Mapping[str, Any],
    runtime_root: Path,
) -> dict[str, Any]:
    path = runtime_root / "completion.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = copy.deepcopy(value)
    expected = payload.pop("completion_sha256", None)
    if expected != manifest_generator.canonical_sha(payload):
        raise CampaignExecutionError(
            "prior aggregate completion integrity failed"
        )
    prior_sha = manifest["resume_contract"]["prior_manifest"][
        "manifest_sha256"
    ]
    expected_workflows = {
        item["workflow_id"] for item in manifest["ptms"]
    }
    if (
        value.get("campaign_id") != manifest["campaign_id"]
        or value.get("model") != "rtdetr"
        or value.get("manifest_sha256") != prior_sha
        or value.get("terminal") is not True
        or set(value.get("outcomes", {})) != expected_workflows
    ):
        raise CampaignExecutionError(
            "prior aggregate completion is not the sealed pre-fix campaign"
        )
    return {
        "path": str(path),
        "file_sha256": _canonical_file_sha(path),
        "completion_sha256": expected,
        "manifest_sha256": prior_sha,
    }


def validate_resume_source(
    manifest: Mapping[str, Any],
    runtime_root: Path,
    workflow_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = manifest["resume_contract"]
    path = runtime_root / workflow_id / "workflow_completion.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    ptm = run_campaign._ptm_by_workflow(manifest, workflow_id)
    failure = record.get("failure")
    train = record.get("jobs", {}).get("train")
    flags = record.get("agent_intervention_flags")
    if (
        record.get("campaign_id") != manifest["campaign_id"]
        or record.get("manifest_sha256")
        != contract["prior_manifest"]["manifest_sha256"]
        or record.get("workflow_id") != workflow_id
        or record.get("ptm_id") != ptm["id"]
        or record.get("ptm_sha256") != ptm["artifact"]["sha256"]
        or record.get("status") != contract["eligible_prior_status"]
        or record.get("terminal") is not True
        or record.get("failure_preserved") is not True
        or not isinstance(failure, Mapping)
        or failure.get("type") != contract["eligible_failure_type"]
        or re.fullmatch(
            contract["eligible_failure_message_regex"],
            str(failure.get("message", "")),
        )
        is None
        or not isinstance(train, Mapping)
        or train.get("status") != contract["eligible_train_status"]
        or "evaluation" in record.get("jobs", {})
        or not isinstance(flags, Mapping)
        or set(flags) != set(manifest_generator.AGENT_FLAGS)
        or any(value is not False for value in flags.values())
    ):
        raise CampaignExecutionError(
            f"{workflow_id} is not an eligible completed-training resume source"
        )
    status = train.get("status_evidence")
    if (
        not isinstance(status, Mapping)
        or status.get("validation_record_count") != 10
        or status.get("terminal_success") is not True
        or status.get("terminal_success_message")
        != "Train finished successfully."
    ):
        raise CampaignExecutionError(
            f"{workflow_id} lacks complete ten-epoch training evidence"
        )
    job_id = train.get("tao_job_id")
    result_root = train.get("result_root")
    if (
        not isinstance(job_id, str)
        or re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", job_id)
        is None
        or not isinstance(result_root, str)
        or not result_root.startswith("/lustre/")
    ):
        raise CampaignExecutionError(
            f"{workflow_id} has invalid completed-train provenance"
        )
    return record, {
        "path": str(path),
        "file_sha256": _canonical_file_sha(path),
        "prior_manifest_sha256": record["manifest_sha256"],
        "train_job_id": job_id,
        "train_result_root": result_root,
    }


def _initial_resume_record(
    manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    record = copy.deepcopy(dict(source))
    record["manifest_sha256"] = manifest["manifest_sha256"]
    record["status"] = "resume_initialized"
    record["terminal"] = False
    record.pop("terminal_at_utc", None)
    record.pop("failure", None)
    record.pop("failure_preserved", None)
    record["resume"] = {
        "source_workflow_artifact": copy.deepcopy(dict(source_identity)),
        "completed_training_job_reused": True,
        "training_job_submitted": False,
        "selection_or_candidate_change": False,
        "prior_workflow_artifact_modified": False,
    }
    record["jobs"]["train"]["reused_for_resume"] = True
    return record


def _run_resume_workflow(
    manifest_path: str,
    runtime_root_value: str,
    workflow_id: str,
) -> None:
    manifest = manifest_generator.load_manifest(manifest_path)
    runtime_root = Path(runtime_root_value)
    workflow_dir = runtime_root / workflow_id
    output_name = manifest["resume_contract"][
        "resume_workflow_artifact_name"
    ]
    output = workflow_dir / output_name
    events = workflow_dir / "events.resume.jsonl"
    source, source_identity = validate_resume_source(
        manifest, runtime_root, workflow_id
    )
    evidence = _initial_resume_record(
        manifest, source, source_identity
    )
    run_campaign.atomic_json(output, evidence)
    run_campaign.configure_slurm_runtime(manifest)

    try:
        from tao_sdk.platforms.slurm import SlurmSDK

        sdk = SlurmSDK(
            poll_interval=10,
            state_file=workflow_dir / "slurm_state.json",
        )
        train_job_id = source_identity["train_job_id"]
        status = sdk.get_job_status(train_job_id).status
        if status != "Complete":
            raise CampaignExecutionError(
                f"reused training job {train_job_id} is no longer Complete"
            )
        result_root = run_campaign.workflow_support._local_lustre_path(
            sdk.get_job_results_dir(train_job_id)
        )
        if result_root != source_identity["train_result_root"]:
            raise CampaignExecutionError(
                f"reused training job {train_job_id} result root changed"
            )
        remote_training = (
            run_campaign.workflow_support._training_status_evidence(
                sdk,
                train_job_id,
                expected_validation_records=10,
            )
        )
        if remote_training != source["jobs"]["train"]["status_evidence"]:
            raise CampaignExecutionError(
                f"reused training job {train_job_id} evidence changed"
            )
        checkpoint = run_campaign._terminal_checkpoint(
            sdk,
            train_job_id,
            training_epochs=10,
        )
        evidence["jobs"]["train"]["terminal_checkpoint"] = checkpoint
        evidence["resume"]["checkpoint_resolved_after_fix"] = True

        evaluation_spec = run_campaign.build_evaluation_spec(
            manifest, workflow_id, checkpoint["path"]
        )
        evaluation_command, evaluation_command_sha = (
            run_campaign._job_entrypoint(
                manifest, "evaluate", evaluation_spec
            )
        )
        evaluation_job = run_campaign._submit_job(
            sdk, manifest, evaluation_command
        )
        evidence["status"] = "standalone_evaluation"
        evidence["jobs"]["evaluation"] = {
            "tao_job_id": evaluation_job.id,
            "status": "submitted",
            "submitted_at_utc": run_campaign.utc_timestamp(),
            "spec_sha256": manifest_generator.canonical_sha(evaluation_spec),
            "command_sha256": evaluation_command_sha,
            "full_validation_split": True,
            "nodes": 1,
            "gpus": 8,
            "checkpoint": checkpoint["path"],
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_size_bytes": checkpoint["size_bytes"],
            "resume_only": True,
        }
        run_campaign.atomic_json(output, evidence)
        evaluation_status = run_campaign.workflow_support._wait_for_job(
            sdk,
            evaluation_job.id,
            events=events,
            workflow_id=workflow_id,
            phase="resumed_standalone_evaluation",
        )
        evaluation = evidence["jobs"]["evaluation"]
        evaluation["status"] = evaluation_status
        evaluation["terminal_at_utc"] = run_campaign.utc_timestamp()
        evaluation["result_root"] = (
            run_campaign.workflow_support._local_lustre_path(
                sdk.get_job_results_dir(evaluation_job.id)
            )
        )
        if evaluation_status != "Complete":
            evaluation["failure_analysis"] = sdk.get_failure_analysis(
                evaluation_job.id
            )
            raise CampaignExecutionError(
                "resumed standalone evaluation ended with terminal status "
                f"{evaluation_status}"
            )
        evaluation_evidence = (
            run_campaign.workflow_support._evaluation_status_evidence(
                sdk, evaluation_job.id
            )
        )
        evaluation["status_evidence"] = evaluation_evidence[
            "status_evidence"
        ]
        evidence["metrics"] = evaluation_evidence["metrics"]
        evidence["status"] = "success"
        evidence["terminal"] = True
        evidence["terminal_at_utc"] = run_campaign.utc_timestamp()
        evidence["failure_preserved"] = False
        run_campaign.atomic_json(output, evidence)
    except BaseException as exc:
        evidence["status"] = "terminal_failure"
        evidence["terminal"] = True
        evidence["terminal_at_utc"] = run_campaign.utc_timestamp()
        evidence["failure_preserved"] = True
        evidence["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "replacement_submitted": False,
            "training_job_resubmitted": False,
        }
        run_campaign.atomic_json(output, evidence)
        raise


def build_resume_completion(
    manifest: Mapping[str, Any],
    runtime_root: Path,
    workflow_ids: tuple[str, ...],
    exit_codes: Mapping[str, int | None],
    prior_completion: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_name = manifest["resume_contract"][
        "resume_workflow_artifact_name"
    ]
    workflows = []
    for workflow_id in workflow_ids:
        path = runtime_root / workflow_id / artifact_name
        if not path.is_file():
            raise CampaignExecutionError(
                f"resume worker omitted terminal artifact: {workflow_id}"
            )
        record = json.loads(path.read_text(encoding="utf-8"))
        record["process_exit_code"] = exit_codes.get(workflow_id)
        workflows.append(record)
    success_count = sum(item["status"] == "success" for item in workflows)
    count = len(workflows)
    payload = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "model": "rtdetr",
        "manifest_sha256": manifest["manifest_sha256"],
        "terminal": True,
        "status": "success" if success_count == count else "terminal_with_failures",
        "terminal_at_utc": run_campaign.utc_timestamp(),
        "logical_workflows_submitted": count,
        "successful_workflows": success_count,
        "failed_workflows": count - success_count,
        "workflows_started_in_parallel": True,
        "completion_generated_automatically": True,
        "resume_completed_training": True,
        "completed_training_jobs_reused": count,
        "training_jobs_submitted": 0,
        "prior_completion": copy.deepcopy(dict(prior_completion)),
        "prior_completion_artifact_modified": False,
        "cpu_runs": 0,
        "smoke_runs": 0,
        "ministep_runs": 0,
        "local_model_runs": 0,
        "failures_preserved": True,
        "replacement_workflows_submitted": False,
        "outcomes": {
            item["workflow_id"]: item["status"] for item in workflows
        },
        "workflows": workflows,
    }
    payload["completion_sha256"] = manifest_generator.canonical_sha(payload)
    return payload


def resume_evaluations(
    manifest_path: Path,
    runtime_root: Path,
    completion_artifact: Path,
    env_file: Path,
) -> int:
    manifest = manifest_generator.load_manifest(manifest_path)
    if not runtime_root.is_dir():
        raise CampaignExecutionError(
            f"resume runtime root is unavailable: {runtime_root}"
        )
    if completion_artifact.exists():
        raise CampaignExecutionError(
            f"resume completion already exists: {completion_artifact}"
        )
    workflow_ids = tuple(item["workflow_id"] for item in manifest["ptms"])
    for workflow_id in workflow_ids:
        output = (
            runtime_root
            / workflow_id
            / manifest["resume_contract"]["resume_workflow_artifact_name"]
        )
        if output.exists():
            raise CampaignExecutionError(
                f"resume artifact already exists: {output}"
            )

    run_campaign.load_launch_environment(env_file)
    local = run_campaign.verify_local_launch_contract(manifest)
    remote = run_campaign.verify_remote_contract(manifest)
    prior_completion = validate_prior_completion(manifest, runtime_root)
    sources = {}
    for workflow_id in workflow_ids:
        _, identity = validate_resume_source(
            manifest, runtime_root, workflow_id
        )
        sources[workflow_id] = identity
    run_campaign.atomic_json(
        runtime_root / "resume_plan.json",
        {
            "schema_version": 1,
            "campaign_id": manifest["campaign_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "resumed_at_utc": run_campaign.utc_timestamp(),
            "direct_full_dataset_acknowledged": True,
            "evaluation_only": True,
            "training_jobs_submitted": 0,
            "workflow_ids": list(workflow_ids),
            "local_provenance": local,
            "remote_provenance": remote,
            "prior_completion": prior_completion,
            "source_workflows": sources,
        },
    )

    context = mp.get_context("spawn")
    processes = {
        workflow_id: context.Process(
            target=_run_resume_workflow,
            args=(str(manifest_path), str(runtime_root), workflow_id),
            name=f"rtdetr-resume-{workflow_id}",
        )
        for workflow_id in workflow_ids
    }
    for process in processes.values():
        process.start()
    for process in processes.values():
        process.join()
    completion = build_resume_completion(
        manifest,
        runtime_root,
        workflow_ids,
        {name: process.exitcode for name, process in processes.items()},
        prior_completion,
    )
    run_campaign.atomic_json(completion_artifact, completion)
    validated = run_campaign.validate_completion(completion, manifest)
    if (
        validated.get("resume_completed_training") is not True
        or validated.get("training_jobs_submitted") != 0
    ):
        raise CampaignExecutionError("resume completion isolation failed")
    return 0 if validated["status"] == "success" else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=manifest_generator.DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=DEFAULT_RUNTIME_ROOT,
    )
    parser.add_argument("--completion-artifact", type=Path)
    parser.add_argument("--env-file", type=Path, default=run_campaign.ENV_PATH)
    parser.add_argument("--resume-evaluations", action="store_true")
    parser.add_argument(
        "--acknowledge-direct-full-dataset",
        action="store_true",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = manifest_generator.load_manifest(args.manifest)
    completion = (
        args.completion_artifact
        if args.completion_artifact is not None
        else args.runtime_root
        / manifest["resume_contract"]["resume_completion_artifact_name"]
    )
    if not args.resume_evaluations:
        print(
            json.dumps(
                {
                    "campaign_id": manifest["campaign_id"],
                    "resume_evaluations": False,
                    "evaluation_only": True,
                    "training_jobs_submitted": 0,
                    "workflows": len(manifest["ptms"]),
                    "completion_artifact": str(completion),
                    "required_flags": [
                        "--resume-evaluations",
                        "--acknowledge-direct-full-dataset",
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.acknowledge_direct_full_dataset:
        raise CampaignExecutionError(
            "--resume-evaluations requires "
            "--acknowledge-direct-full-dataset"
        )
    return resume_evaluations(
        args.manifest,
        args.runtime_root,
        completion,
        args.env_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())
