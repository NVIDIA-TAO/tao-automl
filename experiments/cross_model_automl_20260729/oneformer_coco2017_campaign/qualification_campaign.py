#!/usr/bin/env python3

"""Run the frozen OneFormer PTM qualification directly on SLURM GPUs.

There is intentionally no CPU, smoke, or mini-step path.  ``--launch``
submits one independent full-COCO one-epoch train/evaluate workflow for every
checkpoint in the immutable data-only PTM stage.  Each job uses one node and
all eight pinned A100 GPUs in the reviewed SQSH/runtime overlay.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import re
import shlex
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from tao_automl.ptm_registry import (
    canonical_sha256,
    load_ptm_registry,
    merge_ptm_spec_precedence,
)

try:
    from experiments.cross_model_automl_20260729 import checkpoint_resume
except ModuleNotFoundError:  # pragma: no cover - pytest direct-path import
    import checkpoint_resume
from . import campaign_contract, ptm_stage, run_campaign


DEFAULT_CONTRACT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "oneformer_coco2017_ptm_qualification_v4/qualification.v4.json"
)
DEFAULT_RUNTIME_ROOT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "oneformer_coco2017_ptm_qualification_v4"
)
DEFAULT_STAGE_MANIFEST = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "oneformer_coco2017_ptm_qualification_v1/ptm_stage_manifest.json"
)
QUALIFICATION_CAMPAIGN_ID = (
    "oneformer-coco2017-direct-full-ptm-qualification-v3-20260801"
)
CampaignExecutionError = run_campaign.CampaignExecutionError
atomic_json = run_campaign.atomic_json
utc_timestamp = run_campaign.utc_timestamp


def load_frozen_v3_contract(path: str | Path) -> dict[str, Any]:
    """Load only the exact historical v3 qualification contract."""
    resolved = Path(path).resolve()
    frozen = campaign_contract.FROZEN_V3_QUALIFICATION_CONTRACT
    if (
        str(resolved) != frozen["path"]
        or not resolved.is_file()
        or campaign_contract.sha256_file(resolved) != frozen["file_sha256"]
    ):
        raise CampaignExecutionError(
            "immutable OneFormer v3 qualification contract changed"
        )
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CampaignExecutionError(
            "immutable OneFormer v3 qualification contract is invalid"
        ) from exc
    payload = copy.deepcopy(document)
    supplied = payload.pop("contract_sha256", None)
    if (
        supplied != frozen["contract_sha256"]
        or supplied != canonical_sha256(payload)
    ):
        raise CampaignExecutionError(
            "immutable OneFormer v3 qualification contract integrity failed"
        )
    return document


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def load_qualification_contract(path: str | Path) -> dict[str, Any]:
    """Load either the immutable v3 source or its selective v4 successor."""
    resolved = Path(path).resolve()
    frozen = campaign_contract.FROZEN_V3_QUALIFICATION_CONTRACT
    if str(resolved) == frozen["path"]:
        return load_frozen_v3_contract(resolved)
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CampaignExecutionError(
            "OneFormer qualification successor is unavailable or invalid"
        ) from exc
    payload = copy.deepcopy(document)
    supplied = payload.pop("contract_sha256", None)
    policy = document.get("qualification_policy", {})
    runtime = document.get("runtime", {})
    if (
        supplied != canonical_sha256(payload)
        or document.get("campaign_id")
        != "oneformer-coco2017-direct-full-ptm-qualification-v4-20260801"
        or document.get("model") != "oneformer"
        or document.get("task") != "panoptic_segmentation"
        or document.get("sqsh") != campaign_contract.FROZEN_SQSH
        or document.get("search", {}).get("space")
        != campaign_contract.SEARCH_SPACE
        or policy.get("version") != 4
        or policy.get("qualification_campaign_id")
        != document.get("campaign_id")
        or policy.get("checkpoint_resume_policy")
        != campaign_contract.CHECKPOINT_RESUME_POLICY
        or policy.get("runtime_local_eligibility") is not None
        or runtime.get("runtime_local_eligibility") is not None
        or policy.get("qualification_evidence_path")
        != runtime.get("qualification_evidence_path")
        or runtime.get("max_job_retries")
        != campaign_contract.FROZEN_SLURM_RETRY_CAP
        or policy.get("recovery_checkpoint_ids")
        != ["oneformer.its.commercial.dinat_large.trainable"]
        or set(policy.get("reused_checkpoint_ids", []))
        != {
            "oneformer.ade20k.research.swin_large.trainable.v1.0",
            "oneformer.coco.research.swin_large.trainable",
            "oneformer.its.commercial.swin_large.trainable.v1.0",
        }
        or any(document.get("agent_intervention_flags", {}).values())
    ):
        raise CampaignExecutionError(
            "OneFormer v4 qualification successor changed"
        )
    campaign_contract.validate_dataset_record(document["dataset"])
    return document


def verify_qualification_local_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the clean source and every local v4 launcher dependency."""
    runtime = contract["runtime"]
    repository = Path(runtime["repository"]).resolve()
    if (
        _git(repository, "rev-parse", "HEAD") != runtime["source_commit"]
        or _git(repository, "status", "--porcelain")
    ):
        raise CampaignExecutionError("sealed AutoML source changed")
    if (
        _git(Path(runtime["sdk_dir"]), "rev-parse", "HEAD")
        != runtime["sdk_commit"]
        or _git(Path(runtime["skills_repository"]), "rev-parse", "HEAD")
        != runtime["skills_commit"]
    ):
        raise CampaignExecutionError("sealed SDK or skills commit changed")
    here = Path(__file__).resolve().parent
    identities = {
        "wheel": (runtime["wheel_path"], runtime["wheel_sha256"]),
        "ptm_stage": (
            runtime["ptm_stage_manifest_path"],
            runtime["ptm_stage_manifest_sha256"],
        ),
        "predecessor": (
            runtime["qualification_predecessor"]["path"],
            runtime["qualification_predecessor"]["file_sha256"],
        ),
        "campaign_contract": (
            here / "campaign_contract.py",
            contract["launcher_integrity"]["campaign_contract_sha256"],
        ),
        "qualification_campaign": (
            Path(__file__),
            contract["launcher_integrity"]["qualification_campaign_sha256"],
        ),
        "qualification_successor": (
            here / "qualification_successor.py",
            contract["launcher_integrity"]["qualification_successor_sha256"],
        ),
        "run_campaign": (
            here / "run_campaign.py",
            contract["launcher_integrity"]["run_campaign_sha256"],
        ),
        "checkpoint_resume": (
            here.parent / "checkpoint_resume.py",
            contract["launcher_integrity"]["checkpoint_resume_sha256"],
        ),
    }
    evidence = {}
    for name, (path_value, expected_sha) in identities.items():
        artifact = Path(path_value).resolve()
        if (
            not artifact.is_file()
            or campaign_contract.sha256_file(artifact) != expected_sha
        ):
            raise CampaignExecutionError(
                f"sealed qualification artifact changed: {name}"
            )
        evidence[name] = {"path": str(artifact), "sha256": expected_sha}
    return {
        "source_commit": runtime["source_commit"],
        "sdk_commit": runtime["sdk_commit"],
        "skills_commit": runtime["skills_commit"],
        "artifacts": evidence,
    }


def _safe_component(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not result:
        raise CampaignExecutionError("empty qualification path component")
    return result


def qualification_plan(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the exact GPU-only qualification without submitting it."""
    policy = contract["qualification_policy"]
    checkpoint_ids = policy.get("recovery_checkpoint_ids") or [
        item["id"] for item in contract["ptm_inventory"]["records"]
    ]
    reused_ids = policy.get("reused_checkpoint_ids", [])
    return {
        "schema_version": 1,
        "campaign_id": policy.get(
            "qualification_campaign_id", QUALIFICATION_CAMPAIGN_ID
        ),
        "contract_sha256": contract["contract_sha256"],
        "model": "oneformer",
        "task": "panoptic_segmentation",
        "metric": "PQ",
        "checkpoint_ids": checkpoint_ids,
        "workflow_count": len(checkpoint_ids),
        "reused_predecessor_workflow_count": len(reused_ids),
        "reused_predecessor_checkpoint_ids": sorted(reused_ids),
        "all_workflows_independent": True,
        "all_workflows_concurrent": True,
        "workflow": "full_coco_one_epoch_train_then_standalone_evaluation",
        "full_dataset": True,
        "training_epochs": 1,
        "standalone_full_validation": True,
        "resources_per_job": {
            "nodes": 1,
            "gpus": 8,
            "gpu": campaign_contract.FROZEN_HARDWARE["gpu_name"],
            "partition": contract["runtime"]["partition"],
            "container": contract["sqsh"]["path"],
        },
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "recovery_scope": (
            "all_four_original_arms"
            if not reused_ids
            else "only_terminal_failed_arm_from_exact_v3_cohort"
        ),
        "replacement_workflows_allowed": bool(reused_ids),
        "scheduler_client_constructed": False,
        "jobs_submitted": 0,
        "agent_intervention_flags": {
            name: False for name in campaign_contract.AGENT_FLAGS
        },
    }


def load_ptm_stage(
    path: str | Path,
    contract: Mapping[str, Any],
    *,
    verify_remote: bool,
) -> dict[str, dict[str, Any]]:
    """Validate and optionally re-hash the exact four-arm PTM stage."""
    stage_path = Path(path).resolve()
    if not stage_path.is_file() or stage_path.is_symlink():
        raise CampaignExecutionError(
            f"PTM stage manifest is unavailable: {stage_path}"
        )
    raw_sha = campaign_contract.sha256_file(stage_path)
    runtime = contract["runtime"]
    if raw_sha != runtime.get("ptm_stage_manifest_sha256"):
        raise CampaignExecutionError("sealed PTM stage manifest bytes changed")
    document = json.loads(stage_path.read_text(encoding="utf-8"))
    root = document.get("publication", {}).get("canonical_root")
    stage = ptm_stage.validate_stage_manifest(
        document,
        registry=load_ptm_registry(),
        canonical_root=root,
    )
    if (
        stage.get("manifest_sha256")
        != runtime.get("ptm_stage_content_sha256")
        or stage_path.stat().st_mode & 0o222
    ):
        raise CampaignExecutionError("sealed PTM stage identity changed")
    expected_ids = [
        item["id"] for item in contract["ptm_inventory"]["records"]
    ]
    if [item["id"] for item in stage["checkpoints"]] != expected_ids:
        raise CampaignExecutionError(
            "PTM stage must contain every official OneFormer arm exactly once"
        )
    result: dict[str, dict[str, Any]] = {}
    for item in stage["checkpoints"]:
        source = {
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        if verify_remote:
            observed = run_campaign._remote_file_identity(item["path"])
            mode = run_campaign.remote_output(
                f"stat -c %a {shlex.quote(item['path'])}"
            ).strip()
            if (
                observed["size_bytes"] != item["size_bytes"]
                or observed["sha256"] != item["sha256"]
                or mode != "444"
            ):
                raise CampaignExecutionError(
                    f"remote staged PTM changed: {item['id']}"
                )
        result[item["id"]] = source
    if verify_remote:
        remote_manifest = stage["publication"]["manifest_path"]
        observed = run_campaign._remote_file_identity(remote_manifest)
        if observed["sha256"] != raw_sha:
            raise CampaignExecutionError(
                "remote and local PTM stage manifests differ"
            )
    return result


def _qualification_specs(
    contract: Mapping[str, Any],
    checkpoint_id: str,
    checkpoint_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = load_ptm_registry()
    record = registry.checkpoint(checkpoint_id)
    sidecar = record["checkpoint_spec_file"]
    sidecar_path = (
        Path(contract["runtime"]["repository"])
        / "src/tao_automl"
        / sidecar["path"]
    )
    if (
        not sidecar_path.is_file()
        or campaign_contract.sha256_file(sidecar_path) != sidecar["sha256"]
        or yaml.safe_load(sidecar_path.read_text(encoding="utf-8"))
        != record["default_spec_overrides"]
    ):
        raise CampaignExecutionError(
            f"checkpoint-specific spec changed: {checkpoint_id}"
        )
    skill_dir = Path(contract["runtime"]["skill_dir"])
    profile = campaign_contract.profile_overrides(contract["dataset"]["root"])
    train_defaults = yaml.safe_load(
        (skill_dir / "references/spec_template_train.yaml").read_text(
            encoding="utf-8"
        )
    )
    evaluate_defaults = yaml.safe_load(
        (skill_dir / "references/spec_template_evaluate.yaml").read_text(
            encoding="utf-8"
        )
    )
    train = merge_ptm_spec_precedence(
        model_defaults=train_defaults,
        ptm_overrides=record["default_spec_overrides"],
        automl_profile_overrides=profile,
    ).spec
    evaluate = merge_ptm_spec_precedence(
        model_defaults=evaluate_defaults,
        ptm_overrides=record["default_spec_overrides"],
        automl_profile_overrides=profile,
    ).spec
    if record["checkpoint_target"] != "train.pretrained_model":
        raise CampaignExecutionError(
            f"unsupported checkpoint target: {record['checkpoint_target']}"
        )
    train["train"]["pretrained_model"] = checkpoint_path
    train["results_dir"] = ""
    train["train"]["results_dir"] = ""
    evaluate["results_dir"] = ""
    evaluate["evaluate"]["results_dir"] = ""
    evaluate["evaluate"]["checkpoint"] = ""
    if (
        train["train"]["num_epochs"] != 1
        or train["train"]["validation_interval"] != 1
        or train["train"]["num_gpus"] != 8
        or train["train"]["gpu_ids"] != list(range(8))
        or train["train"]["num_nodes"] != 1
        or evaluate["evaluate"]["num_gpus"] != 8
        or evaluate["evaluate"]["gpu_ids"] != list(range(8))
        or evaluate["evaluate"]["num_nodes"] != 1
        or evaluate["evaluate"]["task"] != "panoptic"
    ):
        raise CampaignExecutionError(
            f"full GPU qualification spec changed: {checkpoint_id}"
        )
    return train, evaluate


def _gpu_guard(command: str) -> str:
    return " ".join(
        [
            "set -eu;",
            "names=\"$(nvidia-smi --query-gpu=name --format=csv,noheader)\";",
            "caps=\"$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)\";",
            "mem=\"$(nvidia-smi --query-gpu=memory.total "
            "--format=csv,noheader,nounits)\";",
            "test \"$(printf '%s\\n' \"$names\" | sed '/^$/d' | wc -l)\" -eq 8;",
            "test \"$(printf '%s\\n' \"$names\" | sort -u)\" = "
            "'NVIDIA A100-SXM4-80GB';",
            "test \"$(printf '%s\\n' \"$caps\" | sort -u)\" = '8.0';",
            "test \"$(printf '%s\\n' \"$mem\" | sort -u)\" = '81920';",
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
            / "references/skill_info.yaml"
        ).read_text(encoding="utf-8")
    )
    action = metadata["actions"][action_name]
    action_command = action["command"]
    if action_name == "train":
        action_command = checkpoint_resume.wrap_train_command(
            action_command,
            model_slug="oneformer",
            decision_filename="oneformer_checkpoint_resume_decision.json",
            history_directory="oneformer_checkpoint_resume_decisions",
            trust_checkpoint_on_fresh_start=True,
        )
    entrypoint = build_entrypoint(
        command=_gpu_guard(action_command),
        specs=copy.deepcopy(dict(specification)),
        inputs=action["inputs"],
        outputs=action["outputs"],
        config_format=action["config_format"],
        upload_excludes=action.get("upload_excludes", []),
    )
    command = entrypoint["command"]
    return command, run_campaign.text_sha256(command)


def _submit(sdk: Any, contract: Mapping[str, Any], command: str) -> Any:
    runtime = contract["runtime"]
    return sdk.create_job(
        image=contract["sqsh"]["path"],
        command=command,
        gpu_count=8,
        num_nodes=1,
        partition=runtime["partition"],
        account=runtime["account"],
        env_vars={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
    )


def _status_records(sdk: Any, job_id: str, action: str) -> list[dict[str, Any]]:
    root = run_campaign._local_lustre_path(sdk.get_job_results_dir(job_id))
    status_path = f"{root}/results_dir/{action}/status.json"
    output = run_campaign.remote_output(
        f"test -f {shlex.quote(status_path)} && cat {shlex.quote(status_path)}"
    )
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CampaignExecutionError(
                f"{action} status line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise CampaignExecutionError(f"{action} status record is invalid")
        records.append(value)
    if not records:
        raise CampaignExecutionError(f"{action} status evidence is empty")
    return records


def _metric(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise CampaignExecutionError(f"{name} must be finite in [0, 1]")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CampaignExecutionError(
            f"{name} must be finite in [0, 1]"
        ) from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise CampaignExecutionError(f"{name} must be finite in [0, 1]")
    return number


def _pq_evidence(sdk: Any, job_id: str, action: str) -> float:
    records = _status_records(sdk, job_id, action)
    expected_message = (
        "Eval metrics generated."
        if action == "train"
        else "Test metrics generated."
    )
    expected_key = "PQ" if action == "train" else "test_PQ"
    values = [
        _metric(record["kpi"][expected_key], f"{action} {expected_key}")
        for record in records
        if record.get("message") == expected_message
        and isinstance(record.get("kpi"), Mapping)
        and expected_key in record["kpi"]
    ]
    if len(values) != 1:
        raise CampaignExecutionError(
            f"{action} emitted {len(values)} task-correct PQ records; expected 1"
        )
    return values[0]


def _failure(
    checkpoint_id: str,
    reason: str,
    *,
    code: str,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "checkpoint_id": checkpoint_id,
        "status": "failure",
        "terminal": True,
        "failure_preserved": True,
        "failure_code": code,
        "failure_reason": reason,
        "replacement_submitted": False,
        "diagnostics": copy.deepcopy(dict(diagnostics)),
        "agent_intervention_flags": {
            name: False for name in campaign_contract.AGENT_FLAGS
        },
    }
    value["workflow_sha256"] = canonical_sha256(value)
    return value


def _execute_workflow(
    contract: Mapping[str, Any],
    source: Mapping[str, Any],
    checkpoint_id: str,
    workflow_dir: Path,
) -> dict[str, Any]:
    from tao_sdk.platforms.slurm import SlurmSDK

    base_sdk = SlurmSDK(
        poll_interval=10,
        state_file=workflow_dir / "slurm_state.json",
    )
    sdk = run_campaign.RuntimeOverlaySDK(
        base_sdk,
        contract,
        ledger_path=workflow_dir / "runtime_overlay_commands.json",
    )
    events = workflow_dir / "events.jsonl"
    train_spec, evaluate_spec = _qualification_specs(
        contract, checkpoint_id, source["path"]
    )
    diagnostics: dict[str, Any] = {
        "source_checkpoint": copy.deepcopy(dict(source)),
        "train_spec_sha256": canonical_sha256(train_spec),
        "evaluate_spec_sha256_before_checkpoint": canonical_sha256(
            evaluate_spec
        ),
        "checkpoint_resume_policy": copy.deepcopy(
            campaign_contract.CHECKPOINT_RESUME_POLICY
        ),
    }
    try:
        train_command, train_command_sha = _entrypoint(
            contract, "train", train_spec
        )
        train_job = _submit(sdk, contract, train_command)
        diagnostics["train_job"] = {
            "tao_job_id": train_job.id,
            "status": "submitted",
            "command_sha256": train_command_sha,
            "nodes": 1,
            "gpus": 8,
            **sdk.command_evidence(train_job.id),
        }
        atomic_json(workflow_dir / "workflow_progress.json", diagnostics)
        train_status = run_campaign._wait_for_job(
            sdk,
            train_job.id,
            events=events,
            phase="qualification_train",
            mode="qualification",
            candidate_id=checkpoint_id,
        )
        diagnostics["train_job"]["status"] = train_status
        if train_status != "Complete":
            diagnostics["train_job"]["failure_analysis"] = (
                sdk.get_failure_analysis(train_job.id)
            )
            return _failure(
                checkpoint_id,
                f"full training ended as {train_status}",
                code="direct_full_training_failed",
                diagnostics=diagnostics,
            )
        train_receipt = run_campaign._runtime_overlay_receipt(
            sdk, contract, train_job.id
        )
        terminal = run_campaign._terminal_checkpoint(sdk, train_job.id)
        val_pq = _pq_evidence(sdk, train_job.id, "train")
        diagnostics["train_job"].update(
            {
                "runtime_overlay_receipt": train_receipt,
                "terminal_checkpoint": terminal,
                "PQ": val_pq,
            }
        )

        evaluate_spec["evaluate"]["checkpoint"] = terminal["path"]
        evaluate_command, evaluate_command_sha = _entrypoint(
            contract, "evaluate", evaluate_spec
        )
        evaluate_job = _submit(sdk, contract, evaluate_command)
        diagnostics["evaluation_job"] = {
            "tao_job_id": evaluate_job.id,
            "status": "submitted",
            "spec_sha256": canonical_sha256(evaluate_spec),
            "command_sha256": evaluate_command_sha,
            "nodes": 1,
            "gpus": 8,
            **sdk.command_evidence(evaluate_job.id),
        }
        atomic_json(workflow_dir / "workflow_progress.json", diagnostics)
        evaluate_status = run_campaign._wait_for_job(
            sdk,
            evaluate_job.id,
            events=events,
            phase="qualification_evaluate",
            mode="qualification",
            candidate_id=checkpoint_id,
        )
        diagnostics["evaluation_job"]["status"] = evaluate_status
        if evaluate_status != "Complete":
            diagnostics["evaluation_job"]["failure_analysis"] = (
                sdk.get_failure_analysis(evaluate_job.id)
            )
            return _failure(
                checkpoint_id,
                f"standalone evaluation ended as {evaluate_status}",
                code="direct_full_evaluation_failed",
                diagnostics=diagnostics,
            )
        evaluate_receipt = run_campaign._runtime_overlay_receipt(
            sdk, contract, evaluate_job.id
        )
        test_pq = _pq_evidence(sdk, evaluate_job.id, "evaluate")
        diagnostics["evaluation_job"].update(
            {"runtime_overlay_receipt": evaluate_receipt, "test_PQ": test_pq}
        )
        value = {
            "checkpoint_id": checkpoint_id,
            "status": "success",
            "terminal": True,
            "failure_preserved": False,
            "source_checkpoint": copy.deepcopy(dict(source)),
            "train": {
                "status": "Complete",
                "full_dataset": True,
                "training_epochs": 1,
                "validation_interval": 1,
                "validation_record_count": 1,
                "nodes": 1,
                "gpus": 8,
                "PQ": val_pq,
                "runtime_overlay_receipt": train_receipt,
                "terminal_checkpoint": terminal,
                "tao_job_id": train_job.id,
            },
            "evaluation": {
                "status": "Complete",
                "full_validation_split": True,
                "nodes": 1,
                "gpus": 8,
                "test_PQ": test_pq,
                "runtime_overlay_receipt": evaluate_receipt,
                "tao_job_id": evaluate_job.id,
            },
            "diagnostics": diagnostics,
            "agent_intervention_flags": {
                name: False for name in campaign_contract.AGENT_FLAGS
            },
        }
        value["workflow_sha256"] = canonical_sha256(value)
        return value
    except BaseException as exc:
        return _failure(
            checkpoint_id,
            f"{type(exc).__name__}: {exc}",
            code="direct_full_workflow_exception",
            diagnostics=diagnostics,
        )


def _worker(
    contract_path: str,
    stage_path: str,
    runtime_root: str,
    checkpoint_id: str,
) -> None:
    root = Path(runtime_root)
    workflow_dir = root / "workflows" / _safe_component(checkpoint_id)
    workflow_dir.mkdir(parents=True, exist_ok=True)
    try:
        contract = load_qualification_contract(contract_path)
        run_campaign.configure_slurm_runtime(contract)
        staged = load_ptm_stage(stage_path, contract, verify_remote=False)
        workflow = _execute_workflow(
            contract, staged[checkpoint_id], checkpoint_id, workflow_dir
        )
    except BaseException as exc:
        workflow = _failure(
            checkpoint_id,
            f"worker: {type(exc).__name__}: {exc}",
            code="qualification_worker_failed",
            diagnostics={},
        )
    atomic_json(workflow_dir / "workflow_completion.json", workflow)


def _predecessor_successes(
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Load exact v3 successes reused by the selective v4 recovery."""
    policy = contract["qualification_policy"]
    reused_ids = tuple(policy.get("reused_checkpoint_ids", ()))
    if not reused_ids:
        return []
    record = policy.get("predecessor_evidence")
    if not isinstance(record, Mapping):
        raise CampaignExecutionError("v4 predecessor evidence is unavailable")
    path = Path(str(record.get("path", ""))).resolve()
    if (
        not path.is_file()
        or campaign_contract.sha256_file(path) != record.get("file_sha256")
    ):
        raise CampaignExecutionError("v3 predecessor evidence changed")
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = copy.deepcopy(document)
    supplied = payload.pop("evidence_sha256", None)
    if (
        supplied != canonical_sha256(payload)
        or supplied != record.get("evidence_sha256")
    ):
        raise CampaignExecutionError("v3 predecessor integrity failed")
    workflows = document.get("workflows")
    if not isinstance(workflows, list):
        raise CampaignExecutionError("v3 predecessor workflows are missing")
    by_id = {
        item.get("checkpoint_id"): item
        for item in workflows
        if isinstance(item, Mapping)
    }
    recovery_ids = set(policy.get("recovery_checkpoint_ids", ()))
    if (
        set(by_id) != set(reused_ids) | recovery_ids
        or any(by_id[item].get("status") != "success" for item in reused_ids)
        or any(
            by_id[item].get("status") != "failure"
            or by_id[item].get("terminal") is not True
            or by_id[item].get("failure_preserved") is not True
            for item in recovery_ids
        )
    ):
        raise CampaignExecutionError("v3 predecessor cohort changed")
    return [copy.deepcopy(dict(by_id[item])) for item in sorted(reused_ids)]


def build_completion(
    contract: Mapping[str, Any],
    workflows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    policy = contract["qualification_policy"]
    reused_ids = list(policy.get("reused_checkpoint_ids", []))
    recovery_ids = list(policy.get("recovery_checkpoint_ids", []))
    value = {
        "schema_version": 1,
        "campaign_id": policy.get(
            "qualification_campaign_id", QUALIFICATION_CAMPAIGN_ID
        ),
        "model": "oneformer",
        "task": "panoptic_segmentation",
        "metric": "PQ",
        "metric_semantics": (
            "panoptic_quality_from_native_coco_panoptic_annotations"
        ),
        "pq_emitted": True,
        "pq_claim_authorized": True,
        "qualification_contract_sha256": contract["contract_sha256"],
        "qualification_campaign_sha256": contract["launcher_integrity"][
            "qualification_campaign_sha256"
        ],
        "ptm_stage_manifest_sha256": contract["runtime"][
            "ptm_stage_manifest_sha256"
        ],
        "ptm_stage_content_sha256": contract["runtime"][
            "ptm_stage_content_sha256"
        ],
        "registry_sha256": contract["ptm_inventory"]["registry_sha256"],
        "sqsh_sha256": contract["sqsh"]["sha256"],
        "runtime_overlay_sha256": contract["runtime_overlay"][
            "archive_sha256"
        ],
        "runtime_overlay_source_commit": contract["runtime_overlay"][
            "source_commit"
        ],
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "replacement_workflows_submitted": bool(recovery_ids),
        "replacement_workflow_count": len(recovery_ids),
        "reused_predecessor_workflow_count": len(reused_ids),
        "reused_predecessor_checkpoint_ids": sorted(reused_ids),
        "recovery_checkpoint_ids": sorted(recovery_ids),
        "predecessor_evidence": copy.deepcopy(
            policy.get("predecessor_evidence")
        ),
        "workflows": [copy.deepcopy(dict(item)) for item in workflows],
    }
    resume_policy = policy.get("checkpoint_resume_policy")
    if resume_policy is not None:
        value["checkpoint_resume_policy"] = copy.deepcopy(resume_policy)
    value["evidence_sha256"] = canonical_sha256(value)
    return value


def launch(
    *,
    contract_path: Path,
    stage_path: Path,
    runtime_root: Path,
    env_path: Path,
) -> dict[str, Any]:
    contract = load_qualification_contract(contract_path)
    policy = contract["qualification_policy"]
    expected_stage = Path(
        contract["qualification_policy"]["ptm_stage_manifest_path"]
    ).resolve()
    expected_completion = Path(
        contract["qualification_policy"]["qualification_evidence_path"]
    ).resolve()
    if stage_path.resolve() != expected_stage:
        raise CampaignExecutionError(
            "PTM stage path differs from the sealed contract"
        )
    if expected_completion != (runtime_root / "completion.json").resolve():
        raise CampaignExecutionError(
            "runtime root differs from the sealed qualification path"
        )
    if expected_completion.exists():
        raise CampaignExecutionError(
            "qualification completion already exists; replacement is forbidden"
        )
    run_campaign.load_env_file(env_path)
    run_campaign.configure_slurm_runtime(contract)
    local = (
        verify_qualification_local_contract(contract)
        if policy.get("version") == 4
        else run_campaign.verify_local_contract(contract)
    )
    dataset = run_campaign._verify_dataset_remote(contract)
    sqsh = run_campaign._remote_file_identity(contract["sqsh"]["path"])
    if (
        sqsh["sha256"] != contract["sqsh"]["sha256"]
        or sqsh["size_bytes"] != contract["sqsh"]["size_bytes"]
    ):
        raise CampaignExecutionError("pinned SQSH identity changed")
    overlay = run_campaign._verify_runtime_overlay_remote(contract)
    staged = load_ptm_stage(stage_path, contract, verify_remote=True)
    runtime_root.mkdir(parents=True, exist_ok=True)
    recovery_ids = sorted(
        policy.get("recovery_checkpoint_ids") or staged
    )
    if not recovery_ids or not set(recovery_ids).issubset(staged):
        raise CampaignExecutionError(
            "sealed recovery checkpoint set is invalid"
        )
    predecessor_successes = _predecessor_successes(contract)
    if (
        policy.get("version") == 4
        and len(predecessor_successes) + len(recovery_ids) != len(staged)
    ):
        raise CampaignExecutionError(
            "v4 reused and recovery arms do not cover the exact PTM cohort"
        )
    atomic_json(
        runtime_root / "submission_provenance.json",
        {
            "schema_version": 1,
            "contract_sha256": contract["contract_sha256"],
            "secrets_loaded": True,
            "secret_names_recorded": False,
            "secret_values_recorded": False,
            "local_contract": local,
            "dataset": dataset,
            "sqsh": sqsh,
            "runtime_overlay": overlay,
            "staged_checkpoint_ids": sorted(staged),
            "submitted_checkpoint_ids": recovery_ids,
            "reused_predecessor_checkpoint_ids": sorted(
                item["checkpoint_id"] for item in predecessor_successes
            ),
            "nodes_per_job": 1,
            "gpus_per_job": 8,
            "cpu_model_runs": 0,
            "smoke_model_runs": 0,
            "mini_step_runs": 0,
            "submitted_at_utc": utc_timestamp(),
        },
    )

    context = mp.get_context("spawn")
    processes = {
        checkpoint_id: context.Process(
            target=_worker,
            args=(
                str(contract_path),
                str(stage_path),
                str(runtime_root),
                checkpoint_id,
            ),
            name=f"oneformer-{_safe_component(checkpoint_id)}",
        )
        for checkpoint_id in recovery_ids
    }
    if set(processes) != set(recovery_ids):
        raise CampaignExecutionError(
            "qualification workers differ from the sealed recovery set"
        )
    for process in processes.values():
        process.start()
    for process in processes.values():
        process.join()

    workflows = list(predecessor_successes)
    for checkpoint_id, process in processes.items():
        path = (
            runtime_root
            / "workflows"
            / _safe_component(checkpoint_id)
            / "workflow_completion.json"
        )
        if path.is_file():
            workflows.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            workflows.append(
                _failure(
                    checkpoint_id,
                    f"worker exited {process.exitcode} without evidence",
                    code="qualification_worker_evidence_missing",
                    diagnostics={"worker_exit_code": process.exitcode},
                )
            )
    by_id = {
        item.get("checkpoint_id"): item
        for item in workflows
        if isinstance(item, Mapping)
    }
    if set(by_id) != set(staged) or len(workflows) != len(staged):
        raise CampaignExecutionError(
            "completion does not preserve exactly one workflow per PTM"
        )
    completion = build_completion(
        contract,
        [by_id[checkpoint_id] for checkpoint_id in sorted(by_id)],
    )
    atomic_json(expected_completion, completion)
    return completion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT
    )
    parser.add_argument(
        "--stage-manifest", type=Path, default=DEFAULT_STAGE_MANIFEST
    )
    parser.add_argument("--env-file", type=Path, default=run_campaign.ENV_PATH)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args(argv)

    contract = load_qualification_contract(args.contract.resolve())
    if not args.launch:
        print(json.dumps(qualification_plan(contract), indent=2, sort_keys=True))
        return 0
    completion = launch(
        contract_path=args.contract.resolve(),
        stage_path=args.stage_manifest.resolve(),
        runtime_root=args.runtime_root.resolve(),
        env_path=args.env_file.resolve(),
    )
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0 if all(
        item.get("status") == "success" for item in completion["workflows"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
