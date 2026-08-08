#!/usr/bin/env python3

"""Run real eight-GPU Grounding DINO qualification for all official PTMs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from tao_automl.ptm_registry import canonical_sha256

try:
    from .contract import AGENT_FLAGS, MODEL_ID, PreparationError, read_json
    from .future_contract import DEFAULT_OUTPUT, validate_future_contract
except ImportError:  # pragma: no cover - direct script execution
    from contract import AGENT_FLAGS, MODEL_ID, PreparationError, read_json
    from future_contract import DEFAULT_OUTPUT, validate_future_contract

from experiments.cross_model_automl_20260729.deformable_detr_campaign import (
    run_campaign as workflow_support,
)


HERE = Path(__file__).resolve().parent
DEFAULT_RUNTIME_ROOT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "grounding_dino_synthetic_structured_config_successor_v1"
)
ENV_PATH = Path("/localhome/local-rarunachalam/.tao/config.env")
CampaignExecutionError = workflow_support.CampaignExecutionError
atomic_json = workflow_support.atomic_json
utc_timestamp = workflow_support.utc_timestamp


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _workflow(
    contract: Mapping[str, Any],
    workflow_id: str,
) -> Mapping[str, Any]:
    matches = [
        item
        for item in contract["qualification"]["jobs"]
        if item["workflow_id"] == workflow_id
    ]
    if len(matches) != 1:
        raise CampaignExecutionError(f"no unique workflow {workflow_id!r}")
    return matches[0]


def configure_runtime(contract: Mapping[str, Any]) -> None:
    workflow_support.configure_slurm_runtime({"runtime": contract["runtime"]})
    for name, value in contract["runtime"]["offline_environment"].items():
        if name in {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}:
            os.environ[name] = str(value)


def _remote_file(
    path: str,
    *,
    expected_size: int,
    expected_sha256: str,
    timeout: int = 1800,
    require_nonwritable: bool = True,
) -> dict[str, Any]:
    lines = workflow_support.remote_output(
        " ".join(
            [
                "test -f",
                shlex.quote(path),
                "&& stat -c '%s %a'",
                shlex.quote(path),
                "&& sha256sum",
                shlex.quote(path),
            ]
        ),
        timeout=timeout,
    ).strip().splitlines()
    if len(lines) != 2:
        raise CampaignExecutionError(f"remote identity was incomplete: {path}")
    size_text, mode = lines[0].split()
    digest = lines[1].split()[0]
    if (
        int(size_text) != expected_size
        or digest != expected_sha256
        or (require_nonwritable and int(mode, 8) & 0o222)
    ):
        raise CampaignExecutionError(f"remote identity changed: {path}")
    return {
        "path": path,
        "size_bytes": expected_size,
        "sha256": digest,
        "mode": mode,
    }


def _remote_image_tree(path: str) -> dict[str, Any]:
    script = (
        "import hashlib,json,pathlib,sys;"
        "root=pathlib.Path(sys.argv[1]);"
        "items=sorted((p for p in root.iterdir() if p.is_file()),key=lambda p:p.name);"
        "outer=hashlib.sha256();total=0;"
        "exec(\"for p in items:\\n"
        " h=hashlib.sha256(p.read_bytes()).hexdigest()\\n"
        " outer.update(f'{h}  {p.name}\\\\n'.encode())\\n"
        " total+=p.stat().st_size\");"
        "print(json.dumps({'algorithm':'sha256_of_sorted_sha256sum_basename_lines',"
        "'sha256':outer.hexdigest(),'file_count':len(items),'total_bytes':total},"
        "sort_keys=True))"
    )
    return json.loads(
        workflow_support.remote_output(
            f"python3 -c {shlex.quote(script)} {shlex.quote(path)}",
            timeout=3600,
        ).strip()
    )


def verify_launch_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    validate_future_contract(contract)
    repository = Path(contract["source"]["repository"])
    minimum = contract["source"]["minimum_ancestor_commit"]
    head = _git(repository, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", minimum, head],
        check=False,
        capture_output=True,
        timeout=30,
    ).returncode == 0
    if not ancestor or _git(repository, "status", "--porcelain"):
        raise CampaignExecutionError(
            "Grounding DINO launch source must be a clean descendant"
        )
    runtime = contract["runtime"]
    sources = {}
    for label, path_key, revision_key in (
        ("sdk", "sdk_dir", "sdk_revision"),
        ("skills", "skill_dir", "skill_revision"),
    ):
        path = Path(runtime[path_key])
        revision = _git(path, "rev-parse", "HEAD")
        if revision != runtime[revision_key] or _git(path, "status", "--porcelain"):
            raise CampaignExecutionError(f"pinned {label} checkout changed")
        sources[label] = {"path": str(path), "commit": revision, "clean": True}

    verified_files = {
        "sqsh": _remote_file(
            runtime["sqsh_path"],
            expected_size=runtime["sqsh_size_bytes"],
            expected_sha256=runtime["sqsh_sha256"],
            timeout=7200,
            # This long-lived user-owned SQSH predates the campaign and is
            # mode 0644. Its immutable content identity, not an invented mode
            # requirement, is the runtime contract used by DINO/DDETR/RT.
            require_nonwritable=False,
        )
    }
    stage = read_json(contract["runtime_inputs"]["stage_record_path"])
    for item in stage["official_ptms"]:
        verified_files[f"ptm:{item['id']}"] = _remote_file(
            item["lustre"]["path"],
            expected_size=item["lustre"]["size_bytes"],
            expected_sha256=item["lustre"]["sha256"],
        )
    for item in stage["text_encoder"]["files"]:
        verified_files[f"bert:{item['path']}"] = _remote_file(
            item["lustre"]["path"],
            expected_size=item["lustre"]["size_bytes"],
            expected_sha256=item["lustre"]["sha256"],
        )
    dataset = contract["dataset"]
    source = read_json(dataset["source_manifest_path"])
    for split in ("train", "validation"):
        expected = source["splits"][split]["images"]["identity"]
        observed = _remote_image_tree(
            source["splits"][split]["images"]["path"]
        )
        if observed != expected:
            raise CampaignExecutionError(
                f"{split} image-tree identity changed"
            )
        verified_files[f"dataset:{split}_images"] = {
            "path": source["splits"][split]["images"]["path"],
            **observed,
        }
    for label, path in (
        ("train_odvg", dataset["paths"]["train_odvg_jsonl"]),
        ("train_label_map", dataset["paths"]["train_label_map"]),
        ("validation_coco", dataset["paths"]["validation_coco_contiguous"]),
    ):
        matching = [
            value
            for value in read_json(dataset["conversion_manifest_path"])[
                "canonical_outputs"
            ].values()
            if value["lustre_path"] == path
        ]
        if len(matching) != 1:
            raise CampaignExecutionError(f"no exact dataset identity for {label}")
        expected = matching[0]
        verified_files[f"dataset:{label}"] = _remote_file(
            path,
            expected_size=expected["size_bytes"],
            expected_sha256=expected["sha256"],
        )
    return {
        "source": {
            "minimum_ancestor_commit": minimum,
            "launch_head": head,
            "clean": True,
        },
        "pinned_sources": sources,
        "remote_files": verified_files,
    }


def _gpu_guard(command: str) -> str:
    return " ".join(
        [
            "set -eu;",
            "gpu_names=\"$(nvidia-smi --query-gpu=name --format=csv,noheader)\";",
            "test \"$(printf '%s\\n' \"$gpu_names\" | sed '/^$/d' | wc -l)\" -eq 8;",
            "test \"$(printf '%s\\n' \"$gpu_names\" | "
            "grep -Ec 'NVIDIA (A100|H100)')\" -eq 8;",
            "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1;",
            command,
        ]
    )


def _entrypoint(
    contract: Mapping[str, Any],
    action_name: str,
    specification: Mapping[str, Any],
) -> tuple[str, str]:
    from tao_sdk.script_runner import build_entrypoint

    metadata = yaml.safe_load(
        (
            Path(contract["runtime"]["skill_dir"])
            / "references"
            / "skill_info.yaml"
        ).read_text(encoding="utf-8")
    )
    action = metadata["actions"][action_name]
    entrypoint = build_entrypoint(
        command=_gpu_guard(action["command"]),
        specs=specification,
        inputs=action["inputs"],
        outputs=action["outputs"],
        config_format=action["config_format"],
        upload_excludes=action.get("upload_excludes", []),
    )
    command = entrypoint["command"]
    return command, hashlib.sha256(command.encode("utf-8")).hexdigest()


def _submit(
    sdk: Any,
    contract: Mapping[str, Any],
    command: str,
) -> Any:
    runtime = contract["runtime"]
    return sdk.create_job(
        image=runtime["sqsh_path"],
        command=command,
        gpu_count=8,
        num_nodes=1,
        partition=runtime["partition"],
        account=runtime["account"],
    )


def _initial_evidence(
    contract: Mapping[str, Any],
    workflow_id: str,
) -> dict[str, Any]:
    workflow = _workflow(contract, workflow_id)
    return {
        "schema_version": 1,
        "campaign_id": contract["campaign_id"],
        "contract_sha256": contract["contract_sha256"],
        "workflow_id": workflow_id,
        "ptm_id": workflow["ptm_id"],
        "ptm_sha256": workflow["staged_checkpoint"]["sha256"],
        "status": "initialized",
        "terminal": False,
        "jobs": {},
        "agent_intervention_flags": {name: False for name in AGENT_FLAGS},
    }


def _run_workflow(
    contract_path: str,
    runtime_root: str,
    workflow_id: str,
) -> None:
    contract = read_json(contract_path)
    validate_future_contract(contract)
    workflow = _workflow(contract, workflow_id)
    root = Path(runtime_root) / workflow_id
    root.mkdir(parents=True, exist_ok=True)
    evidence_path = root / "workflow_completion.json"
    events = root / "events.jsonl"
    evidence = _initial_evidence(contract, workflow_id)
    atomic_json(evidence_path, evidence)
    configure_runtime(contract)
    try:
        from tao_sdk.platforms.slurm import SlurmSDK
        import tao_sdk

        sdk_source = Path(tao_sdk.__file__).resolve()
        if not sdk_source.is_relative_to(
            Path(contract["runtime"]["sdk_dir"]).resolve()
        ):
            raise CampaignExecutionError(
                f"tao_sdk imported from unsealed source: {sdk_source}"
            )
        sdk = SlurmSDK(
            poll_interval=10,
            state_file=root / "slurm_state.json",
        )
        train_command, train_command_sha = _entrypoint(
            contract,
            "train",
            workflow["train"]["spec"],
        )
        train_job = _submit(sdk, contract, train_command)
        evidence["status"] = "training"
        evidence["jobs"]["train"] = {
            "tao_job_id": train_job.id,
            "status": "submitted",
            "submitted_at_utc": utc_timestamp(),
            "spec_sha256": workflow["train"]["spec_sha256"],
            "command_sha256": train_command_sha,
            "nodes": 1,
            "gpus": 8,
            "training_epochs": 10,
        }
        atomic_json(evidence_path, evidence)
        train_status = workflow_support._wait_for_job(
            sdk,
            train_job.id,
            events=events,
            workflow_id=workflow_id,
            phase="train",
        )
        evidence["jobs"]["train"]["status"] = train_status
        evidence["jobs"]["train"]["terminal_at_utc"] = utc_timestamp()
        if train_status != "Complete":
            evidence["jobs"]["train"]["failure_analysis"] = (
                sdk.get_failure_analysis(train_job.id)
            )
            raise CampaignExecutionError(
                f"training job ended with {train_status}"
            )
        evidence["jobs"]["train"]["status_evidence"] = (
            workflow_support._training_status_evidence(
                sdk,
                train_job.id,
                expected_validation_records=10,
            )
        )
        checkpoint = workflow_support._terminal_checkpoint(
            sdk,
            train_job.id,
            training_epochs=10,
        )
        evidence["jobs"]["train"]["terminal_checkpoint"] = checkpoint

        evaluation_spec = copy.deepcopy(workflow["evaluate"]["spec"])
        evaluation_spec["evaluate"]["checkpoint"] = checkpoint["path"]
        evaluation_command, evaluation_command_sha = _entrypoint(
            contract,
            "evaluate",
            evaluation_spec,
        )
        evaluation_job = _submit(sdk, contract, evaluation_command)
        evidence["status"] = "standalone_evaluation"
        evidence["jobs"]["evaluate"] = {
            "tao_job_id": evaluation_job.id,
            "status": "submitted",
            "submitted_at_utc": utc_timestamp(),
            "spec_sha256": canonical_sha256(evaluation_spec),
            "command_sha256": evaluation_command_sha,
            "checkpoint": checkpoint,
            "nodes": 1,
            "gpus": 8,
        }
        atomic_json(evidence_path, evidence)
        evaluate_status = workflow_support._wait_for_job(
            sdk,
            evaluation_job.id,
            events=events,
            workflow_id=workflow_id,
            phase="standalone_evaluation",
        )
        evidence["jobs"]["evaluate"]["status"] = evaluate_status
        evidence["jobs"]["evaluate"]["terminal_at_utc"] = utc_timestamp()
        if evaluate_status != "Complete":
            evidence["jobs"]["evaluate"]["failure_analysis"] = (
                sdk.get_failure_analysis(evaluation_job.id)
            )
            raise CampaignExecutionError(
                f"evaluation job ended with {evaluate_status}"
            )
        evaluation = workflow_support._evaluation_status_evidence(
            sdk,
            evaluation_job.id,
        )
        evidence["jobs"]["evaluate"]["status_evidence"] = evaluation[
            "status_evidence"
        ]
        evidence["metrics"] = {
            "training_validation": evidence["jobs"]["train"][
                "status_evidence"
            ]["validation_metrics"],
            "standalone": evaluation["metrics"],
        }
        evidence["status"] = "success"
        evidence["terminal"] = True
        evidence["terminal_at_utc"] = utc_timestamp()
        evidence["failure_preserved"] = False
        atomic_json(evidence_path, evidence)
    except BaseException as exc:
        evidence["status"] = "terminal_failure"
        evidence["terminal"] = True
        evidence["terminal_at_utc"] = utc_timestamp()
        evidence["failure_preserved"] = True
        evidence["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "replacement_submitted": False,
        }
        atomic_json(evidence_path, evidence)
        raise


def build_completion(
    contract: Mapping[str, Any],
    runtime_root: Path,
    workflow_ids: tuple[str, ...],
    exit_codes: Mapping[str, int | None],
) -> dict[str, Any]:
    workflows = []
    for workflow_id in workflow_ids:
        path = runtime_root / workflow_id / "workflow_completion.json"
        record = read_json(path) if path.is_file() else {
            **_initial_evidence(contract, workflow_id),
            "status": "terminal_failure",
            "terminal": True,
            "failure_preserved": True,
            "failure": {
                "type": "MissingWorkflowArtifact",
                "message": "worker exited without terminal evidence",
                "replacement_submitted": False,
            },
        }
        record["process_exit_code"] = exit_codes.get(workflow_id)
        workflows.append(record)
    successful = sum(item.get("status") == "success" for item in workflows)
    value = {
        "schema_version": 1,
        "campaign_id": contract["campaign_id"],
        "contract_sha256": contract["contract_sha256"],
        "model": MODEL_ID,
        "terminal": True,
        "status": (
            "success"
            if successful == len(workflow_ids)
            else "terminal_with_failures"
        ),
        "successful_workflows": successful,
        "failed_workflows": len(workflow_ids) - successful,
        "minimum_supported_ptms_for_pilot": 1,
        "pilot_handoff_ready": successful >= 1,
        "failures_preserved": True,
        "replacement_workflows_submitted": False,
        "workflows": workflows,
        "agent_intervention_flags": {name: False for name in AGENT_FLAGS},
    }
    value["completion_sha256"] = canonical_sha256(value)
    return value


def launch(
    *,
    contract_path: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    contract = read_json(contract_path)
    validate_future_contract(contract)
    runtime_root.mkdir(parents=True, exist_ok=True)
    workflow_support.load_launch_environment(ENV_PATH)
    configure_runtime(contract)
    preflight = verify_launch_contract(contract)
    atomic_json(
        runtime_root / "launch_preflight.json",
        {
            "schema_version": 1,
            "contract_sha256": contract["contract_sha256"],
            "verified": preflight,
            "sdk_constructed": False,
            "scheduler_jobs_submitted": 0,
        },
    )
    workflow_ids = tuple(
        item["workflow_id"] for item in contract["qualification"]["jobs"]
    )
    context = mp.get_context("spawn")
    processes = {
        workflow_id: context.Process(
            target=_run_workflow,
            args=(str(contract_path), str(runtime_root), workflow_id),
            name=f"grounding-dino-{workflow_id}",
        )
        for workflow_id in workflow_ids
    }
    for process in processes.values():
        process.start()
    for process in processes.values():
        process.join()
    completion = build_completion(
        contract,
        runtime_root,
        workflow_ids,
        {name: process.exitcode for name, process in processes.items()},
    )
    atomic_json(runtime_root / "qualification_completion.json", completion)
    if completion["pilot_handoff_ready"]:
        handoff = {
            "schema_version": 1,
            "campaign_id": contract["campaign_id"],
            "contract_sha256": contract["contract_sha256"],
            "qualification_completion_sha256": completion[
                "completion_sha256"
            ],
            "automatic": True,
            "manual_confirmation_required": False,
            "pilot_modes": ["accuracy", "latency", "multi_objective"],
            "status": "ready_for_algorithm_generated_mode_pilots",
            "selection_or_recommendation_performed": False,
            "agent_intervention_flags": {
                name: False for name in AGENT_FLAGS
            },
        }
        handoff["handoff_sha256"] = canonical_sha256(handoff)
        atomic_json(runtime_root / "pilot_handoff.json", handoff)
    return completion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--launch", action="store_true")
    arguments = parser.parse_args(argv)
    contract = read_json(arguments.contract)
    validate_future_contract(contract)
    if not arguments.launch:
        print(
            json.dumps(
                {
                    "contract_sha256": contract["contract_sha256"],
                    "launch": False,
                    "qualification_workflows": len(
                        contract["qualification"]["jobs"]
                    ),
                },
                sort_keys=True,
            )
        )
        return 0
    completion = launch(
        contract_path=arguments.contract.resolve(),
        runtime_root=arguments.runtime_root.resolve(),
    )
    print(json.dumps(completion, sort_keys=True))
    return 0 if completion["pilot_handoff_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
