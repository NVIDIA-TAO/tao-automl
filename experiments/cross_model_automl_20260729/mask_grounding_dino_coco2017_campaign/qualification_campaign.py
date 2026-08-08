#!/usr/bin/env python3

"""Run the checkpoint-resumable Mask Grounding DINO PTM qualification.

The default invocation is plan-only. ``--launch`` is the only path that
constructs a scheduler client or submits jobs. It runs no CPU/model smoke and
no mini-step: the official PTM receives a real three-epoch, one-node/eight-A100
full-dataset train followed by standalone full validation. Missing task-correct
COCO mask AP50-95 is retained as a terminal failure and never replaced by the
VG ``overall_IoU`` metric. The v3 successor preserves the exact v2 timeout
evidence and reruns the same arms with per-epoch same-job resume.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import math
import re
import shlex
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

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
from . import campaign_contract, run_campaign


DEFAULT_CONTRACT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask_grounding_dino_coco2017_ptm_qualification_v3/"
    "qualification.v3.json"
)
DEFAULT_RUNTIME_ROOT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask_grounding_dino_coco2017_ptm_qualification_v3"
)
ENV_PATH = run_campaign.ENV_PATH
CampaignExecutionError = run_campaign.CampaignExecutionError
atomic_json = run_campaign.atomic_json
utc_timestamp = run_campaign.utc_timestamp


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def load_qualification_contract(path: str | Path) -> dict[str, Any]:
    """Validate the v3 qualification-only successor contract."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = copy.deepcopy(document)
    supplied = payload.pop("contract_sha256", None)
    policy = document.get("qualification_policy", {})
    runtime = document.get("runtime", {})
    if (
        supplied != canonical_sha256(payload)
        or document.get("campaign_id")
        != "mask_grounding_dino-coco2017-direct-full-qualification-v3-20260801"
        or document.get("model") != "mask_grounding_dino"
        or document.get("task")
        != "category_prompted_grounded_instance_segmentation"
        or document.get("sqsh") != campaign_contract.FROZEN_SQSH
        or document.get("search", {}).get("space")
        != campaign_contract.SEARCH_SPACE
        or policy.get("version") != 3
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
        or any(document.get("agent_intervention_flags", {}).values())
    ):
        raise CampaignExecutionError(
            "Mask Grounding DINO v3 qualification contract changed"
        )
    campaign_contract.validate_dataset_record(document["dataset"])
    return document


def verify_qualification_local_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
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
            Path(__file__).with_name("campaign_contract.py"),
            contract["launcher_integrity"]["campaign_contract_sha256"],
        ),
        "qualification_campaign": (
            Path(__file__),
            contract["launcher_integrity"]["qualification_campaign_sha256"],
        ),
        "qualification_successor": (
            Path(__file__).with_name("qualification_successor.py"),
            contract["launcher_integrity"][
                "qualification_successor_sha256"
            ],
        ),
        "run_campaign": (
            Path(__file__).with_name("run_campaign.py"),
            contract["launcher_integrity"]["run_campaign_sha256"],
        ),
        "checkpoint_resume": (
            Path(__file__).parent.parent / "checkpoint_resume.py",
            contract["launcher_integrity"]["checkpoint_resume_sha256"],
        ),
    }
    evidence = {}
    for name, (path_value, expected) in identities.items():
        artifact = Path(path_value).resolve()
        if (
            not artifact.is_file()
            or campaign_contract.sha256_file(artifact) != expected
        ):
            raise CampaignExecutionError(
                f"sealed qualification artifact changed: {name}"
            )
        evidence[name] = {"path": str(artifact), "sha256": expected}
    return {
        "source_commit": runtime["source_commit"],
        "sdk_commit": runtime["sdk_commit"],
        "skills_commit": runtime["skills_commit"],
        "artifacts": evidence,
    }


def _lower_sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise CampaignExecutionError(f"{name} must be lowercase SHA-256")
    return value


def qualification_plan(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable direct-run plan without accessing a GPU/SLURM."""
    inventory = contract["ptm_inventory"]
    return {
        "schema_version": 1,
        "campaign_id": contract["qualification_policy"].get(
            "qualification_campaign_id",
            "mask_grounding_dino-coco2017-direct-full-qualification-v2-20260801",
        ),
        "contract_sha256": contract["contract_sha256"],
        "model": "mask_grounding_dino",
        "task": "category_prompted_grounded_instance_segmentation",
        "primary_metric": "segm_val_mAP50_95",
        "VG_overall_iou_accepted_as_mask_ap": False,
        "official_checkpoint_ids": [
            record["id"] for record in inventory["records"]
        ],
        "workflow_count": inventory["record_count"],
        "concurrent_workflow_count": inventory["record_count"],
        "independent_sdk_state_per_workflow": True,
        "full_dataset": True,
        "training_epochs": campaign_contract.FROZEN_TRAINING_EPOCHS,
        "standalone_full_validation": True,
        "nodes_per_job": 1,
        "gpus_per_job": 8,
        "hardware": copy.deepcopy(campaign_contract.FROZEN_HARDWARE),
        "sqsh": copy.deepcopy(campaign_contract.FROZEN_SQSH),
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "replacement_workflows_allowed": (
            contract["qualification_policy"].get("version") == 3
        ),
        "replacement_scope": contract["qualification_policy"].get(
            "replacement_scope"
        ),
        "registry_bypass_allowed": False,
        "distributed_strategy_resolution": copy.deepcopy(
            campaign_contract.FROZEN_DDP_STRATEGY_RESOLUTION
        ),
        "predecessor_failure_evidence": copy.deepcopy(
            contract["qualification_policy"]["predecessor_failure_evidence"]
        ),
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
    """Validate a content-addressed Lustre stage for every official arm."""
    stage_path = Path(path).resolve()
    if not stage_path.is_file():
        raise CampaignExecutionError(
            f"PTM stage manifest is unavailable: {stage_path}"
        )
    document = json.loads(stage_path.read_text(encoding="utf-8"))
    expected_file_sha = contract["runtime"].get(
        "ptm_stage_manifest_sha256"
    )
    if (
        not isinstance(expected_file_sha, str)
        or campaign_contract.sha256_file(stage_path) != expected_file_sha
    ):
        raise CampaignExecutionError(
            "sealed PTM stage manifest bytes changed"
        )
    supplied = document.get("manifest_sha256")
    payload = copy.deepcopy(document)
    payload.pop("manifest_sha256", None)
    if supplied != canonical_sha256(payload):
        raise CampaignExecutionError("PTM stage manifest integrity failed")
    expected = contract["ptm_inventory"]
    if (
        document.get("schema_version") != 1
        or document.get("model") != "mask_grounding_dino"
        or not isinstance(document.get("registry_sha256"), str)
        or re.fullmatch(
            r"[0-9a-f]{64}", document["registry_sha256"]
        )
        is None
        or document.get("stage_complete") is not True
        or document.get("remote_read_only") is not True
        or document.get("cpu_model_runs") != 0
        or document.get("smoke_model_runs") != 0
        or document.get("mini_step_runs") != 0
    ):
        raise CampaignExecutionError(
            "PTM stage identity or execution policy changed"
        )
    records = document.get("checkpoints")
    if not isinstance(records, list):
        raise CampaignExecutionError("PTM stage records are unavailable")
    by_id = {
        item.get("id"): item
        for item in records
        if isinstance(item, Mapping)
    }
    expected_by_id = {
        item["id"]: item for item in expected["records"]
    }
    if set(by_id) != set(expected_by_id) or len(records) != len(expected_by_id):
        raise CampaignExecutionError(
            "PTM stage must contain exactly every official Mask Grounding DINO arm"
        )
    result: dict[str, dict[str, Any]] = {}
    for checkpoint_id, registry_record in expected_by_id.items():
        item = by_id[checkpoint_id]
        path_value = item.get("path")
        size = item.get("size_bytes")
        digest = _lower_sha(
            item.get("sha256"), f"{checkpoint_id}.sha256"
        )
        if (
            not isinstance(path_value, str)
            or not path_value.startswith("/lustre/")
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size != registry_record["expected_size_bytes"]
            or item.get("immutable_source_identity")
            != registry_record["source"]["immutable_identity"]
            or (
                registry_record.get("sha256") is not None
                and digest != registry_record["sha256"]
            )
            or item.get("remote_read_only") is not True
        ):
            raise CampaignExecutionError(
                f"staged PTM identity changed: {checkpoint_id}"
            )
        if verify_remote:
            observed = run_campaign._remote_file_identity(path_value)
            if (
                observed["size_bytes"] != size
                or observed["sha256"] != digest
            ):
                raise CampaignExecutionError(
                    f"staged PTM bytes changed: {checkpoint_id}"
                )
            writable = run_campaign.remote_output(
                f"test ! -w {shlex.quote(path_value)} && echo readonly"
            ).strip()
            if writable != "readonly":
                raise CampaignExecutionError(
                    f"staged PTM is writable: {checkpoint_id}"
                )
        result[checkpoint_id] = {
            "path": path_value,
            "size_bytes": size,
            "sha256": digest,
        }
    return result


def _qualification_specs(
    contract: Mapping[str, Any],
    checkpoint_id: str,
    checkpoint_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = load_ptm_registry()
    record = registry.checkpoint(checkpoint_id)
    skill_dir = Path(contract["runtime"]["skill_dir"])
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
    profile = campaign_contract.profile_overrides(
        contract["dataset"]["prepared_root"]
    )
    train = merge_ptm_spec_precedence(
        model_defaults=train_defaults,
        ptm_overrides=record["default_spec_overrides"],
        automl_profile_overrides=profile,
    ).spec
    train["train"]["pretrained_model_path"] = checkpoint_path
    train["results_dir"] = ""
    train["train"]["results_dir"] = ""
    evaluate = merge_ptm_spec_precedence(
        model_defaults=evaluate_defaults,
        ptm_overrides=record["default_spec_overrides"],
        automl_profile_overrides=profile,
    ).spec
    evaluate["results_dir"] = ""
    evaluate["evaluate"]["checkpoint"] = ""
    evaluate["evaluate"]["results_dir"] = ""
    strategy = train["train"].get("distributed_strategy")
    activation_checkpoint = train["train"].get("activation_checkpoint")
    if (
        strategy != campaign_contract.FROZEN_TAO_DISTRIBUTED_STRATEGY
        or activation_checkpoint
        is not campaign_contract.FROZEN_ACTIVATION_CHECKPOINT
    ):
        raise CampaignExecutionError(
            "qualification DDP strategy differs from the sealed v2 policy"
        )
    return train, evaluate


def _gpu_guard(command: str) -> str:
    return " ".join(
        [
            "set -eu;",
            "gpu_names=\"$(nvidia-smi --query-gpu=name "
            "--format=csv,noheader)\";",
            "test \"$(printf '%s\\n' \"$gpu_names\" | "
            "sed '/^$/d' | wc -l)\" -eq 8;",
            "test \"$(printf '%s\\n' \"$gpu_names\" | "
            "grep -Fc 'NVIDIA A100-SXM4-80GB')\" -eq 8;",
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
            model_slug="mask_grounding_dino",
            decision_filename=(
                "mask_grounding_dino_checkpoint_resume_decision.json"
            ),
            history_directory=(
                "mask_grounding_dino_checkpoint_resume_decisions"
            ),
        )
    entrypoint = build_entrypoint(
        command=_gpu_guard(action_command),
        specs=specification,
        inputs=action["inputs"],
        outputs=action["outputs"],
        config_format=action["config_format"],
        upload_excludes=action.get("upload_excludes", []),
    )
    command = entrypoint["command"]
    return command, run_campaign.text_sha256(command)


def _submit(
    sdk: Any,
    contract: Mapping[str, Any],
    command: str,
) -> Any:
    runtime = contract["runtime"]
    return sdk.create_job(
        image=contract["sqsh"]["path"],
        command=command,
        gpu_count=8,
        num_nodes=1,
        partition=runtime["partition"],
        account=runtime["account"],
    )


def _status_values(
    sdk: Any,
    job_id: str,
    *,
    action: str,
    names: tuple[str, ...],
) -> list[float]:
    root = run_campaign._local_lustre_path(
        sdk.get_job_results_dir(job_id)
    )
    path = f"{root}/results_dir/{action}/status.json"
    output = run_campaign.remote_output(
        f"(test -f {shlex.quote(path)} && "
        f"cat {shlex.quote(path)}) || true"
    )
    values: list[float] = []
    for line in output.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        kpi = record.get("kpi")
        if not isinstance(kpi, Mapping):
            continue
        for name in names:
            try:
                value = float(kpi[name])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(value):
                values.append(value)
                break
    return values


def _failure_workflow(
    checkpoint_id: str,
    reason: str,
    *,
    code: str,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = {
        "checkpoint_id": checkpoint_id,
        "status": "failure",
        "terminal": True,
        "failure_preserved": True,
        "failure_code": code,
        "failure_reason": reason,
        "replacement_submitted": False,
        "diagnostics": copy.deepcopy(dict(diagnostics or {})),
        "agent_intervention_flags": {
            name: False for name in campaign_contract.AGENT_FLAGS
        },
    }
    value["workflow_sha256"] = canonical_sha256(value)
    return value


def _run_one(
    contract: Mapping[str, Any],
    sdk: Any,
    checkpoint_id: str,
    source: Mapping[str, Any],
    runtime_root: Path,
) -> dict[str, Any]:
    workflow_dir = runtime_root / checkpoint_id.replace("/", "_")
    workflow_dir.mkdir(parents=True, exist_ok=True)
    events = workflow_dir / "events.jsonl"
    train_spec, evaluate_spec = _qualification_specs(
        contract, checkpoint_id, str(source["path"])
    )
    diagnostics: dict[str, Any] = {
        "source_checkpoint": copy.deepcopy(dict(source)),
        "train_spec_sha256": canonical_sha256(train_spec),
        "distributed_strategy_resolution": copy.deepcopy(
            contract["qualification_policy"][
                "distributed_strategy_resolution"
            ]
        ),
        "checkpoint_resume_policy": copy.deepcopy(
            campaign_contract.CHECKPOINT_RESUME_POLICY
        ),
        "agent_intervention_flags": {
            name: False for name in campaign_contract.AGENT_FLAGS
        },
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
            return _failure_workflow(
                checkpoint_id,
                f"full training ended as {train_status}",
                code="direct_full_training_failed",
                diagnostics=diagnostics,
            )
        terminal = run_campaign._terminal_checkpoint(sdk, train_job.id)
        diagnostics["train_job"]["terminal_checkpoint"] = terminal
        mask_values = _status_values(
            sdk,
            train_job.id,
            action="train",
            names=("[segm] val_mAP@50-95",),
        )
        diagnostics["train_job"]["mask_ap_values"] = mask_values
        diagnostics["train_job"]["VG_overall_iou_diagnostic_values"] = (
            _status_values(
                sdk,
                train_job.id,
                action="train",
                names=("val_overall_IoU",),
            )
        )

        evaluate_spec["evaluate"]["checkpoint"] = terminal["path"]
        evaluate_command, evaluate_command_sha = _entrypoint(
            contract, "evaluate", evaluate_spec
        )
        evaluation_job = _submit(sdk, contract, evaluate_command)
        diagnostics["evaluation_job"] = {
            "tao_job_id": evaluation_job.id,
            "status": "submitted",
            "spec_sha256": canonical_sha256(evaluate_spec),
            "command_sha256": evaluate_command_sha,
            "nodes": 1,
            "gpus": 8,
        }
        atomic_json(workflow_dir / "workflow_progress.json", diagnostics)
        evaluate_status = run_campaign._wait_for_job(
            sdk,
            evaluation_job.id,
            events=events,
            phase="qualification_evaluate",
            mode="qualification",
            candidate_id=checkpoint_id,
        )
        diagnostics["evaluation_job"]["status"] = evaluate_status
        if evaluate_status != "Complete":
            diagnostics["evaluation_job"]["failure_analysis"] = (
                sdk.get_failure_analysis(evaluation_job.id)
            )
            return _failure_workflow(
                checkpoint_id,
                f"standalone evaluation ended as {evaluate_status}",
                code="direct_full_evaluation_failed",
                diagnostics=diagnostics,
            )
        standalone_mask_values = _status_values(
            sdk,
            evaluation_job.id,
            action="evaluate",
            names=("[segm] test_mAP@50-95",),
        )
        diagnostics["evaluation_job"]["mask_ap_values"] = (
            standalone_mask_values
        )
        diagnostics["evaluation_job"][
            "VG_overall_iou_diagnostic_values"
        ] = _status_values(
            sdk,
            evaluation_job.id,
            action="evaluate",
            names=("test_overall_IoU",),
        )
        if (
            len(mask_values)
            != campaign_contract.FROZEN_TRAINING_EPOCHS
            or not standalone_mask_values
        ):
            return _failure_workflow(
                checkpoint_id,
                "task-correct segm_val_mAP50_95 was not emitted by every "
                "in-epoch validation and standalone evaluation; VG "
                "overall_IoU diagnostics are not accepted",
                code="task_correct_metric_missing",
                diagnostics=diagnostics,
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
                "training_epochs": (
                    campaign_contract.FROZEN_TRAINING_EPOCHS
                ),
                "validation_interval": 1,
                "validation_record_count": len(mask_values),
                "nodes": 1,
                "gpus": 8,
                "distributed_strategy_resolution": copy.deepcopy(
                    contract["qualification_policy"][
                        "distributed_strategy_resolution"
                    ]
                ),
                "segm_val_mAP50_95": mask_values[-1],
                "terminal_checkpoint": terminal,
                "tao_job_id": train_job.id,
            },
            "evaluation": {
                "status": "Complete",
                "full_validation_split": True,
                "nodes": 1,
                "gpus": 8,
                "segm_val_mAP50_95": standalone_mask_values[-1],
                "tao_job_id": evaluation_job.id,
            },
            "diagnostics": diagnostics,
            "agent_intervention_flags": {
                name: False for name in campaign_contract.AGENT_FLAGS
            },
        }
        value["workflow_sha256"] = canonical_sha256(value)
        return value
    except BaseException as exc:
        return _failure_workflow(
            checkpoint_id,
            f"{type(exc).__name__}: {exc}",
            code="direct_full_workflow_exception",
            diagnostics=diagnostics,
        )


def build_completion(
    contract: Mapping[str, Any],
    workflows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "campaign_id": contract["qualification_policy"].get(
            "qualification_campaign_id",
            "mask_grounding_dino-coco2017-direct-full-qualification-v2-20260801",
        ),
        "model": "mask_grounding_dino",
        "task": "category_prompted_grounded_instance_segmentation",
        "primary_metric": "segm_val_mAP50_95",
        "VG_overall_iou_accepted_as_mask_ap": False,
        "qualification_contract_sha256": contract["contract_sha256"],
        "qualification_campaign_sha256": contract[
            "launcher_integrity"
        ]["qualification_campaign_sha256"],
        "ptm_stage_manifest_path": contract["runtime"][
            "ptm_stage_manifest_path"
        ],
        "ptm_stage_manifest_sha256": contract["runtime"][
            "ptm_stage_manifest_sha256"
        ],
        "registry_sha256": contract["ptm_inventory"]["registry_sha256"],
        "sqsh_sha256": contract["sqsh"]["sha256"],
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "replacement_workflows_submitted": (
            contract["qualification_policy"].get("version") == 3
        ),
        "replacement_workflow_count": (
            len(workflows)
            if contract["qualification_policy"].get("version") == 3
            else 0
        ),
        "distributed_strategy_resolution": copy.deepcopy(
            contract["qualification_policy"][
                "distributed_strategy_resolution"
            ]
        ),
        "predecessor_failure_evidence": copy.deepcopy(
            contract["qualification_policy"].get(
                "predecessor_failure_evidence"
            )
        ),
        "workflows": [copy.deepcopy(dict(item)) for item in workflows],
    }
    resume_policy = contract["qualification_policy"].get(
        "checkpoint_resume_policy"
    )
    if resume_policy is not None:
        value["checkpoint_resume_policy"] = copy.deepcopy(resume_policy)
    value["evidence_sha256"] = canonical_sha256(value)
    return value


def _run_qualifications_concurrently(
    contract: Mapping[str, Any],
    staged: Mapping[str, Mapping[str, Any]],
    runtime_root: Path,
    sdk_factory: Callable[[str, Path], Any],
) -> list[dict[str, Any]]:
    """Run every frozen PTM workflow concurrently with isolated SDK state."""
    checkpoint_ids = sorted(staged)
    if not checkpoint_ids:
        raise CampaignExecutionError(
            "no staged Mask Grounding DINO checkpoints are available"
        )

    def invoke(checkpoint_id: str) -> dict[str, Any]:
        workflow_dir = runtime_root / checkpoint_id.replace("/", "_")
        workflow_dir.mkdir(parents=True, exist_ok=True)
        try:
            sdk = sdk_factory(
                checkpoint_id,
                workflow_dir / "slurm_state.json",
            )
            return _run_one(
                contract,
                sdk,
                checkpoint_id,
                staged[checkpoint_id],
                runtime_root,
            )
        except BaseException as exc:
            return _failure_workflow(
                checkpoint_id,
                f"{type(exc).__name__}: {exc}",
                code="direct_full_workflow_exception",
                diagnostics={
                    "sdk_state_path": str(
                        workflow_dir / "slurm_state.json"
                    ),
                    "agent_intervention_flags": {
                        name: False
                        for name in campaign_contract.AGENT_FLAGS
                    },
                },
            )

    by_id: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(checkpoint_ids),
        thread_name_prefix="mask-grounding-dino-qualification",
    ) as pool:
        futures = {
            pool.submit(invoke, checkpoint_id): checkpoint_id
            for checkpoint_id in checkpoint_ids
        }
        for future in concurrent.futures.as_completed(futures):
            checkpoint_id = futures[future]
            by_id[checkpoint_id] = future.result()
    return [by_id[checkpoint_id] for checkpoint_id in checkpoint_ids]


def launch(
    *,
    contract_path: Path,
    runtime_root: Path,
    env_path: Path = ENV_PATH,
) -> dict[str, Any]:
    """Submit the one frozen direct-full workflow; no replacement is made."""
    contract = load_qualification_contract(contract_path)
    runtime_root.mkdir(parents=True, exist_ok=True)
    loaded_names = run_campaign.load_env_file(env_path)
    run_campaign.configure_slurm_runtime(contract)
    local = verify_qualification_local_contract(contract)
    dataset = run_campaign._verify_dataset_remote(contract)
    sqsh = run_campaign._remote_file_identity(contract["sqsh"]["path"])
    if sqsh["sha256"] != contract["sqsh"]["sha256"]:
        raise CampaignExecutionError("pinned SQSH identity changed")
    staged = load_ptm_stage(
        contract["qualification_policy"]["ptm_stage_manifest_path"],
        contract,
        verify_remote=True,
    )
    atomic_json(
        runtime_root / "submission_provenance.json",
        {
            "schema_version": 1,
            "contract_sha256": contract["contract_sha256"],
            "loaded_secret_keys": list(loaded_names),
            "secret_values_recorded": False,
            "local_contract": local,
            "dataset": dataset,
            "sqsh": sqsh,
            "ptm_stage_manifest_path": contract[
                "qualification_policy"
            ]["ptm_stage_manifest_path"],
            "ptm_stage_sha256": campaign_contract.sha256_file(
                contract["qualification_policy"]["ptm_stage_manifest_path"]
            ),
            "concurrent_workflow_count": len(staged),
            "independent_sdk_state_per_workflow": True,
            "nodes_per_job": 1,
            "gpus_per_job": 8,
            "cpu_model_runs": 0,
            "smoke_model_runs": 0,
            "mini_step_runs": 0,
            "submitted_at_utc": utc_timestamp(),
        },
    )

    from tao_sdk.platforms.slurm import SlurmSDK

    import tao_sdk

    sdk_source = Path(tao_sdk.__file__).resolve()
    if not sdk_source.is_relative_to(
        Path(contract["runtime"]["sdk_dir"]).resolve()
    ):
        raise CampaignExecutionError(
            f"tao_sdk imported from unsealed source: {sdk_source}"
        )
    def sdk_factory(checkpoint_id: str, state_file: Path) -> Any:
        del checkpoint_id
        return SlurmSDK(poll_interval=10, state_file=state_file)

    workflows = _run_qualifications_concurrently(
        contract,
        staged,
        runtime_root,
        sdk_factory,
    )
    completion = build_completion(contract, workflows)
    evidence_path = Path(
        contract["qualification_policy"]["qualification_evidence_path"]
    )
    if evidence_path.resolve() != (
        runtime_root / "completion.json"
    ).resolve():
        raise CampaignExecutionError(
            "runtime root does not match sealed qualification evidence path"
        )
    atomic_json(evidence_path, completion)
    return completion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT
    )
    parser.add_argument("--env-file", type=Path, default=ENV_PATH)
    parser.add_argument("--launch", action="store_true")
    arguments = parser.parse_args(argv)
    contract = load_qualification_contract(arguments.contract)
    plan = qualification_plan(contract)
    if not arguments.launch:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    completion = launch(
        contract_path=arguments.contract.resolve(),
        runtime_root=arguments.runtime_root.resolve(),
        env_path=arguments.env_file.resolve(),
    )
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0 if all(
        item.get("status") == "success"
        for item in completion["workflows"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
