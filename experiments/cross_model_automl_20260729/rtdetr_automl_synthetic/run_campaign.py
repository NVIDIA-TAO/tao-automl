#!/usr/bin/env python3

"""Run three independent objective-aware RT-DETR AutoML jobs on SLURM.

The launcher has no CPU/model-smoke execution path.  It consumes successful
four-PTM qualification evidence, requires repository registry promotion and a
live typed PTM preflight, then launches accuracy EI, constrained-latency EI,
and ParEGO searches in isolated workspaces.  Each candidate uses one node and
eight A100 GPUs for train, evaluate, and stabilized latency measurement.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from tao_automl.ptm_registry import canonical_sha256, load_ptm_registry
from tao_automl.recommendation_audit import validate_recommendation_audit
from tao_automl.selection import canonical_spec_fingerprint

try:
    from . import campaign_contract
    from .qualification_gate import (
        QualificationDecision,
        QualificationGateError,
        QualificationLoadEvidence,
        audit_qualification,
    )
except ImportError:  # pragma: no cover - direct execution
    import campaign_contract  # type: ignore[no-redef]
    from qualification_gate import (  # type: ignore[no-redef]
        QualificationDecision,
        QualificationGateError,
        QualificationLoadEvidence,
        audit_qualification,
    )

try:
    from experiments.cross_model_automl_20260729.dino_campaign import (
        run_campaign as workflow_support,
    )
    from experiments.cross_model_automl_20260729.rtdetr_campaign import (
        manifest_generator as qualification_manifest_generator,
    )
    from experiments.cross_model_automl_20260729.rtdetr_campaign import (
        run_campaign as qualification_runner,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution
    repository = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repository))
    from experiments.cross_model_automl_20260729.dino_campaign import (
        run_campaign as workflow_support,
    )
    from experiments.cross_model_automl_20260729.rtdetr_campaign import (
        manifest_generator as qualification_manifest_generator,
    )
    from experiments.cross_model_automl_20260729.rtdetr_campaign import (
        run_campaign as qualification_runner,
    )


HERE = Path(__file__).resolve().parent
ENV_PATH = Path("/localhome/local-rarunachalam/.tao/config.env")
DEFAULT_QUALIFICATION_MANIFEST = (
    HERE.parent / "rtdetr_campaign" / "campaign.v1.json"
)
DEFAULT_RUNTIME_ROOT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/rtdetr_three_mode_synthetic"
)
DEFAULT_CONTRACT = DEFAULT_RUNTIME_ROOT / "campaign.contract.json"
DEFAULT_COMPLETION = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/rtdetr_qualification_20260730/"
    "completion.resume.json"
)
TERMINAL_JOB_STATUSES = frozenset({"Complete", "Error", "Canceled"})
SUCCESS_RECOMMENDATION_STATUSES = frozenset({"success", "done"})
MAP50_PATTERNS = (
    re.compile(
        r"(?:Validation|Test)\s+mAP50\s*[:=]\s*"
        r"([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)"
    ),
    re.compile(
        r"\b(?:test|val)_mAP50\b[^0-9+\-]*"
        r"([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)"
    ),
)

CampaignExecutionError = workflow_support.CampaignExecutionError
atomic_json = workflow_support.atomic_json
append_jsonl = workflow_support.append_jsonl
load_env_file = workflow_support.load_env_file
remote_output = workflow_support.remote_output
utc_timestamp = workflow_support.utc_timestamp
text_sha256 = workflow_support.text_sha256
_local_lustre_path = workflow_support._local_lustre_path
_merge_spec = workflow_support._merge_spec


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def configure_slurm_runtime(contract: Mapping[str, Any]) -> None:
    runtime = contract["runtime"]
    sdk_dir = runtime["sdk_dir"]
    sys.path = [sdk_dir, *[item for item in sys.path if item != sdk_dir]]
    previous = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [
            sdk_dir,
            *[
                item
                for item in previous.split(os.pathsep)
                if item and item != sdk_dir
            ],
        ]
    )
    os.environ.update(
        {
            "SLURM_USE_SQSH": "false",
            "SLURM_USE_REQUEUE": "true",
            "SLURM_TIME_HOURS": str(runtime["time_hours"]),
            "SLURM_TIMEOUT_HOURS": str(runtime["timeout_hours"]),
            "SLURM_MAX_GPUS_PER_NODE": "8",
            "SLURM_PARTITION": runtime["partition"],
            "SLURM_ACCOUNT": runtime["account"],
            "SLURM_BASE_RESULTS_DIR": runtime["base_results_dir"],
            "SLURM_CONTAINER_MOUNTS": runtime["container_mounts"],
            "MAX_JOB_RETRIES": str(
                campaign_contract.FROZEN_SLURM_RETRY_CAP
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )


def seal_execution_contract(
    base: Mapping[str, Any],
    *,
    repository: str | Path,
    wheel_path: str | Path,
) -> dict[str, Any]:
    """Bind preregistered intent to one clean source and production wheel."""
    root = Path(repository).resolve()
    wheel = Path(wheel_path).resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise CampaignExecutionError("production AutoML wheel is unavailable")
    if _git(root, "status", "--porcelain"):
        raise CampaignExecutionError(
            "source repository must be clean before campaign sealing"
        )
    value = copy.deepcopy(dict(base))
    value.pop("contract_sha256", None)
    value["source"] = {
        "repository": str(root),
        "commit": _git(root, "rev-parse", "HEAD"),
        "dirty": False,
        "wheel_path": str(wheel),
        "wheel_sha256": campaign_contract.sha256_file(wheel),
    }
    value["launcher_integrity"] = {
        "campaign_contract_sha256": campaign_contract.sha256_file(
            HERE / "campaign_contract.py"
        ),
        "qualification_gate_sha256": campaign_contract.sha256_file(
            HERE / "qualification_gate.py"
        ),
        "run_campaign_sha256": campaign_contract.sha256_file(
            HERE / "run_campaign.py"
        ),
        "latency_worker_sha256": campaign_contract.sha256_file(
            HERE / "rtdetr_latency_worker.py"
        ),
    }
    value["contract_sha256"] = canonical_sha256(value)
    return value


def validate_execution_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(contract))
    observed = value.pop("contract_sha256", None)
    if observed != canonical_sha256(value):
        raise CampaignExecutionError("campaign contract integrity failed")
    if (
        value.get("model") != "rtdetr"
        or value.get("task") != "object_detection"
        or value.get("execution", {}).get("cpu_runs") != 0
        or value.get("execution", {}).get("smoke_runs") != 0
        or value.get("execution", {}).get("gpus_per_child") != 8
        or value.get("search", {}).get("candidate_budget_per_mode") != 20
        or value.get("search", {}).get("training_epochs") != 10
        or value.get("search", {}).get("space")
        != campaign_contract.SEARCH_SPACE
    ):
        raise CampaignExecutionError("campaign execution contract changed")
    if tuple(item["mode"] for item in value.get("modes", ())) != (
        campaign_contract.MODES
    ):
        raise CampaignExecutionError("three independent modes are required")
    expected_ptms = set(
        campaign_contract.EXPECTED_PTMS
        if hasattr(campaign_contract, "EXPECTED_PTMS")
        else ()
    )
    observed_ptms = {item.get("id") for item in value.get("ptms", ())}
    if expected_ptms and observed_ptms != expected_ptms:
        raise CampaignExecutionError("campaign PTM inventory changed")
    value["contract_sha256"] = observed
    return value


def _load_qualification_manifest(path: str | Path) -> dict[str, Any]:
    return qualification_manifest_generator.load_manifest(path)


def _execution_artifacts(
    qualification_manifest: Mapping[str, Any],
    checkpoint_ids: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    by_id = {
        item["id"]: item["artifact"]
        for item in qualification_manifest["ptms"]
    }
    if set(by_id) != set(checkpoint_ids):
        raise CampaignExecutionError(
            "qualification PTM inventory differs from runtime inventory"
        )
    return {
        checkpoint_id: {
            "path": by_id[checkpoint_id]["slurm_path"],
            "sha256": by_id[checkpoint_id]["sha256"],
            "size_bytes": by_id[checkpoint_id]["size_bytes"],
        }
        for checkpoint_id in checkpoint_ids
    }


def _per_checkpoint_profiles(
    qualification_manifest: Mapping[str, Any],
    checkpoint_ids: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Project each frozen registry input contract into its train/eval spec."""
    ptms = {
        item["id"]: item for item in qualification_manifest["ptms"]
    }
    if set(ptms) != set(checkpoint_ids):
        raise CampaignExecutionError(
            "qualification PTM inventory differs from runtime inventory"
        )
    profiles: dict[str, dict[str, Any]] = {}
    for checkpoint_id in checkpoint_ids:
        contract = ptms[checkpoint_id]["input_contract"]
        preprocessing = contract.get("preprocessing")
        height = contract.get("height")
        width = contract.get("width")
        if (
            contract.get("channels") != 3
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height < 1
            or isinstance(width, bool)
            or not isinstance(width, int)
            or width < 1
            or not isinstance(preprocessing, Mapping)
            or not isinstance(
                preprocessing.get("preserve_aspect_ratio"), bool
            )
        ):
            raise CampaignExecutionError(
                f"{checkpoint_id} has an invalid registered input contract"
            )
        profiles[checkpoint_id] = {
            "dataset": {
                "augmentation": {
                    "train_spatial_size": [height, width],
                    "eval_spatial_size": [height, width],
                    "preserve_aspect_ratio": preprocessing[
                        "preserve_aspect_ratio"
                    ],
                }
            }
        }
    return profiles


def build_live_runtime_inventory(
    *,
    contract: Mapping[str, Any],
    qualification_manifest: Mapping[str, Any],
    decision: QualificationDecision,
    mode: str,
    cache_root: str | Path,
) -> Any:
    """Build the live typed, content-preserving hierarchical PTM inventory."""
    from tao_automl.objectives import parse_objective_config
    from tao_automl.ptm_preflight import (
        AtomicArtifactCache,
        NGCCredential,
        NGCHTTPSClient,
        PTMCheckpointPreflight,
    )
    from tao_automl.ptm_runtime import resolve_ptm_runtime_inventory

    settings = campaign_contract.mode_settings(
        str(contract["campaign_id"]),
        mode,
    )
    objective = parse_objective_config(settings)
    registry = load_ptm_registry()
    preflight = PTMCheckpointPreflight(
        registry=registry,
        cache=AtomicArtifactCache(cache_root),
        ngc_client=NGCHTTPSClient(
            NGCCredential.from_environment()
        ),
        load_smoke=QualificationLoadEvidence(decision),
    )
    report = preflight.run(
        model="rtdetr",
        task="object_detection",
        tao_version="7.1.0",
    )
    if (
        not report.ok
        or tuple(item.checkpoint_id for item in report.prepared)
        != decision.checkpoint_ids
        or report.exclusions
    ):
        raise CampaignExecutionError(
            "live runtime preflight did not prepare exactly four PTM arms"
        )
    template = (
        Path(contract["runtime"]["skill_dir"])
        / "references/spec_template_train.yaml"
    )
    base_defaults = yaml.safe_load(template.read_text(encoding="utf-8"))
    profile = campaign_contract.profile_overrides(
        qualification_manifest
    )
    resolved = resolve_ptm_runtime_inventory(
        report=report,
        objective_config=objective,
        base_model_defaults=base_defaults,
        profile_overrides=profile,
        user_overrides=None,
        ptm_policy="all",
        model="rtdetr",
        algorithm="bayesian",
        execution_checkpoint_artifacts=_execution_artifacts(
            qualification_manifest,
            decision.checkpoint_ids,
        ),
        per_checkpoint_profile_overrides=_per_checkpoint_profiles(
            qualification_manifest,
            decision.checkpoint_ids,
        ),
    )
    if resolved.checkpoint_ids != decision.checkpoint_ids:
        raise CampaignExecutionError(
            "hierarchical runtime omitted a qualified PTM arm"
        )
    return resolved


def verify_live_runtime_preflight(
    *,
    contract: Mapping[str, Any],
    qualification_manifest: Mapping[str, Any],
    decision: QualificationDecision,
    cache_root: str | Path,
) -> dict[str, Any]:
    """Complete every live typed PTM/mode gate before starting mode workers."""
    modes: dict[str, Any] = {}
    for mode in campaign_contract.MODES:
        resolved = build_live_runtime_inventory(
            contract=contract,
            qualification_manifest=qualification_manifest,
            decision=decision,
            mode=mode,
            cache_root=cache_root,
        )
        resolved.validate()
        if (
            resolved.mode != mode
            or resolved.checkpoint_ids != decision.checkpoint_ids
        ):
            raise CampaignExecutionError(
                f"{mode} live runtime inventory changed"
            )
        modes[mode] = {
            "inventory_sha256": resolved.inventory_sha256,
            "preflight_report_sha256": resolved.report.report_sha256,
            "checkpoint_ids": list(resolved.checkpoint_ids),
            "arms": [
                {
                    "checkpoint_id": arm.checkpoint_id,
                    "checkpoint_path": arm.checkpoint_path,
                    "checkpoint_artifact_sha256": (
                        arm.checkpoint_artifact_sha256
                    ),
                    "input_contract_sha256": arm.input_contract_sha256,
                    "effective_base_spec_sha256": (
                        arm.effective_base_spec_sha256
                    ),
                    "train_spatial_size": (
                        arm.effective_base_spec["dataset"][
                            "augmentation"
                        ]["train_spatial_size"]
                    ),
                    "eval_spatial_size": (
                        arm.effective_base_spec["dataset"][
                            "augmentation"
                        ]["eval_spatial_size"]
                    ),
                }
                for arm in resolved.arms
            ],
        }
    record = {
        "schema_version": 1,
        "contract_sha256": contract["contract_sha256"],
        "qualification_evidence_sha256": decision.evidence_sha256,
        "status": "success",
        "model_jobs_launched": False,
        "cpu_or_smoke_model_jobs_launched": False,
        "modes": modes,
    }
    record["record_sha256"] = canonical_sha256(record)
    return record


def _wait_for_job(
    sdk: Any,
    job_id: str,
    *,
    events: Path,
    phase: str,
    mode: str,
    candidate_id: str,
) -> str:
    previous = None
    while True:
        status = sdk.get_job_status(job_id).status
        if status != previous:
            append_jsonl(
                events,
                {
                    "event": "slurm_job_status",
                    "phase": phase,
                    "mode": mode,
                    "candidate_id": candidate_id,
                    "tao_job_id": job_id,
                    "status": status,
                    "observed_at_utc": utc_timestamp(),
                },
            )
            previous = status
        if status in TERMINAL_JOB_STATUSES:
            return status
        time.sleep(10)


def _terminal_checkpoint(
    sdk: Any,
    job_id: str,
) -> dict[str, Any]:
    """Resolve the exact RT-DETR epoch-nine checkpoint fail-closed."""
    evidence = qualification_runner._terminal_checkpoint(
        sdk,
        job_id,
        training_epochs=campaign_contract.FROZEN_TRAINING_EPOCHS,
    )
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("training_epochs")
        != campaign_contract.FROZEN_TRAINING_EPOCHS
        or evidence.get("terminal_epoch_index")
        != campaign_contract.FROZEN_TRAINING_EPOCHS - 1
        or evidence.get("filename") != "model_epoch_009.pth"
        or evidence.get("naming_contract")
        != "rtdetr_model_epoch_without_step_suffix"
        or evidence.get("ambiguity_policy") != "fail_closed"
        or not isinstance(evidence.get("path"), str)
        or not evidence["path"].endswith(
            "/results_dir/train/model_epoch_009.pth"
        )
        or not isinstance(evidence.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", evidence["sha256"]) is None
        or isinstance(evidence.get("size_bytes"), bool)
        or not isinstance(evidence.get("size_bytes"), int)
        or evidence["size_bytes"] < 1
    ):
        raise CampaignExecutionError(
            "exact RT-DETR terminal checkpoint evidence is incomplete"
        )
    return copy.deepcopy(dict(evidence))


def _map50_from_status(sdk: Any, job_id: str) -> float | None:
    root = _local_lustre_path(sdk.get_job_results_dir(job_id))
    status_path = f"{root}/results_dir/evaluate/status.json"
    output = remote_output(
        f"(test -f {shlex.quote(status_path)} && "
        f"tail -100 {shlex.quote(status_path)}) || true"
    )
    values = []
    for line in output.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        kpi = record.get("kpi")
        if isinstance(kpi, Mapping):
            metric = kpi.get("test_mAP50", kpi.get("val_mAP50"))
            if metric is not None:
                values.append(float(metric))
    return values[-1] if values else None


def evaluation_spec(
    contract: Mapping[str, Any],
    qualification_manifest: Mapping[str, Any],
    recommendation_specs: Mapping[str, Any],
    checkpoint: str,
) -> dict[str, Any]:
    """Adapt a trained RT-DETR recommendation to standalone evaluation."""
    template = (
        Path(contract["runtime"]["skill_dir"])
        / "references/spec_template_evaluate.yaml"
    )
    spec = yaml.safe_load(template.read_text(encoding="utf-8"))
    spec = _merge_spec(
        spec,
        campaign_contract.profile_overrides(qualification_manifest),
    )
    spec = _merge_spec(spec, recommendation_specs)
    validation = qualification_manifest["dataset"]["splits"][
        "validation"
    ]
    spec["dataset"]["test_data_sources"] = {
        "image_dir": validation["image_dir"],
        "json_file": validation["annotation"],
    }
    spec["dataset"]["batch_size"] = 4
    spec["evaluate"].update(
        {
            "batch_size": 4,
            "num_gpus": 8,
            "gpu_ids": list(range(8)),
            "num_nodes": 1,
            "checkpoint": checkpoint,
            "trt_engine": "",
            "results_dir": "",
        }
    )
    spec["results_dir"] = ""
    spec["wandb"]["enable"] = False
    spec["model"]["num_select"] = min(
        int(spec["model"]["num_select"]),
        int(spec["model"]["num_queries"]),
    )
    return spec


def _launch_evaluation(
    sdk: Any,
    contract: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    events: Path,
    mode: str,
    candidate_id: str,
) -> tuple[float, dict[str, Any]]:
    from tao_sdk.script_runner import build_entrypoint

    action = yaml.safe_load(
        (
            Path(contract["runtime"]["skill_dir"])
            / "references/skill_info.yaml"
        ).read_text(encoding="utf-8")
    )["actions"]["evaluate"]
    entrypoint = build_entrypoint(
        command=action["command"],
        specs=spec,
        inputs=action["inputs"],
        outputs=action["outputs"],
        config_format=action.get("config_format", "yaml"),
        upload_excludes=action.get("upload_excludes", []),
    )
    runtime = contract["runtime"]
    job = sdk.create_job(
        image=runtime["sqsh_path"],
        command=entrypoint["command"],
        gpu_count=8,
        num_nodes=1,
        partition=runtime["partition"],
        account=runtime["account"],
    )
    evidence = {
        "tao_job_id": job.id,
        "status": "submitted",
        "submitted_at_utc": utc_timestamp(),
        "spec_sha256": canonical_sha256(spec),
        "command_sha256": text_sha256(entrypoint["command"]),
    }
    status = _wait_for_job(
        sdk,
        job.id,
        events=events,
        phase="standalone_evaluation",
        mode=mode,
        candidate_id=candidate_id,
    )
    evidence.update(
        {"status": status, "terminal_at_utc": utc_timestamp()}
    )
    logs = sdk.get_job_logs(job.id, tail=5000)
    if status != "Complete":
        raise CampaignExecutionError(
            f"evaluation job {job.id} ended as {status}: {logs[-3000:]}"
        )
    metric = _map50_from_status(sdk, job.id)
    if metric is None:
        values = [
            float(value)
            for pattern in MAP50_PATTERNS
            for value in pattern.findall(logs)
        ]
        metric = values[-1] if values else None
    if metric is None or not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
        raise CampaignExecutionError(
            f"evaluation job {job.id} emitted no valid mAP50"
        )
    evidence["result_root"] = _local_lustre_path(
        sdk.get_job_results_dir(job.id)
    )
    return metric, evidence


def _ptm_id(recommendation_audit: Mapping[str, Any]) -> str:
    try:
        checkpoint_id = recommendation_audit["acquisition"]["proposal"][
            "ptm"
        ]["arm_id"]
    except (KeyError, TypeError) as exc:
        raise CampaignExecutionError(
            "hierarchical recommendation omitted its signed PTM arm"
        ) from exc
    if checkpoint_id not in set(campaign_contract.EXPECTED_PTMS):
        raise CampaignExecutionError(
            f"recommendation emitted unknown PTM arm {checkpoint_id!r}"
        )
    return checkpoint_id


def _immutable_recommendation_record(
    recommendation: Any,
    mode: str,
) -> dict[str, Any]:
    """Validate and materialize one issuance-time recommendation identity."""
    candidate_id = f"{mode}_rec_{recommendation.id}"
    audit = copy.deepcopy(recommendation.recommendation_audit)
    try:
        validate_recommendation_audit(audit)
    except (TypeError, ValueError) as exc:
        raise CampaignExecutionError(
            f"{candidate_id} recommendation audit integrity failed"
        ) from exc
    fingerprint = canonical_spec_fingerprint(recommendation.specs)
    if (
        audit.get("candidate_id") != str(recommendation.id)
        or audit.get("candidate_fingerprint") != fingerprint
    ):
        raise CampaignExecutionError(
            f"{candidate_id} recommendation audit/spec identity changed"
        )
    return {
        "candidate_id": candidate_id,
        "rec_id": str(recommendation.id),
        "status": "recommended",
        "checkpoint_id": _ptm_id(audit),
        "specs": copy.deepcopy(recommendation.specs),
        "candidate_fingerprint": fingerprint,
        "recommendation_audit": audit,
        "agent_intervention_flags": {
            name: False for name in campaign_contract.AGENT_FLAGS
        },
    }


def _preserve_or_add_recommendation(
    candidates: dict[str, Any],
    record: Mapping[str, Any],
) -> None:
    """Fail closed if deterministic resume emits a changed recommendation."""
    candidate_id = str(record["candidate_id"])
    existing = candidates.get(candidate_id)
    if existing is not None and any(
        existing.get(key) != value
        for key, value in record.items()
        if key != "status"
    ):
        raise CampaignExecutionError(
            f"resumed recommendation changed: {candidate_id}"
        )
    stored = candidates.setdefault(candidate_id, {})
    for key, value in record.items():
        if key == "status":
            stored.setdefault(key, copy.deepcopy(value))
        else:
            stored[key] = copy.deepcopy(value)


def _validation_image_ids(annotation: str) -> list[int]:
    script = (
        "import json,sys;"
        "d=json.load(open(sys.argv[1]));"
        "print(json.dumps([x.get('id') for x in d.get('images',[])[:16]]))"
    )
    value = json.loads(
        remote_output(
            f"python3 -c {shlex.quote(script)} {shlex.quote(annotation)}"
        )
    )
    if (
        not isinstance(value, list)
        or len(value) != 16
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise CampaignExecutionError(
            "validation annotation lacks 16 deterministic image IDs"
        )
    return value


def latency_input_descriptor(
    contract: Mapping[str, Any],
    qualification_manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
    checkpoint_id: str,
) -> dict[str, Any]:
    """Bind benchmark preprocessing to the recommendation's PTM contract."""
    registry_record = load_ptm_registry().checkpoint(checkpoint_id)
    input_contract = registry_record["input_contract"]
    spatial = list(spec["dataset"]["augmentation"]["eval_spatial_size"])
    expected = [input_contract["height"], input_contract["width"]]
    if spatial != expected:
        raise CampaignExecutionError(
            f"{checkpoint_id} evaluation shape {spatial} differs from "
            f"registered input contract {expected}"
        )
    validation = qualification_manifest["dataset"]["splits"][
        "validation"
    ]
    shape = [1, 3, *spatial]
    return {
        "schema_version": 1,
        "checkpoint_id": checkpoint_id,
        "input_contract_sha256": canonical_sha256(input_contract),
        "shape_sequence": [shape for _ in range(16)],
        "validation_image_ids": _validation_image_ids(
            validation["annotation"]
        ),
        "dtype": "float32",
        "channels": 3,
        "content": "first_16_deterministic_preprocessed_validation_batches",
        "preloaded_batches": 16,
        "benchmark_seed": 20260727,
        "validation_annotation_sha256": validation["annotation_sha256"],
        "required_hardware": copy.deepcopy(
            campaign_contract.FROZEN_HARDWARE
        ),
    }


def _latency_contract(
    contract: Mapping[str, Any],
    input_sha256: str,
) -> dict[str, Any]:
    protocol = contract["latency_protocol"]
    runtime_identity = {
        "sqsh_sha256": contract["runtime"]["sqsh_sha256"],
        "latency_worker_sha256": contract["launcher_integrity"][
            "latency_worker_sha256"
        ],
        "precision": "fp32",
        "hardware": campaign_contract.FROZEN_HARDWARE,
    }
    return {
        "schema_version": 1,
        "warmup_iterations": protocol["warmup_iterations"],
        "timed_iterations": protocol["timed_iterations"],
        "repeated_rounds": protocol["repeated_rounds"],
        "tail_percentile": protocol["tail_percentile"],
        "bootstrap_resamples": protocol["bootstrap_resamples"],
        "bootstrap_confidence_level": protocol[
            "bootstrap_confidence_level"
        ],
        "bootstrap_seed": protocol["bootstrap_seed"],
        "batch_size_per_replica": protocol["batch_size_per_replica"],
        "precision": protocol["precision"],
        "timed_scope": protocol["timed_scope"],
        "input_sha256": input_sha256,
        "runtime_sha256": canonical_sha256(runtime_identity),
        "expected_replicas": protocol["expected_replicas"],
        "measurement_role": protocol["measurement_role"],
        "synchronization": protocol["synchronization"],
        "validity_thresholds": protocol["validity_thresholds"],
    }


def _payload_command(
    contract: Mapping[str, Any],
    descriptor: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    source_root = Path(contract["source"]["repository"]) / "src/tao_automl"
    latency_contract = _latency_contract(
        contract,
        canonical_sha256(descriptor),
    )
    files = {
        "tao_automl/__init__.py": b"",
        "tao_automl/latency_stats.py": (
            source_root / "latency_stats.py"
        ).read_bytes(),
        "tao_automl/latency_benchmark.py": (
            source_root / "latency_benchmark.py"
        ).read_bytes(),
        "rtdetr_latency_worker.py": (
            HERE / "rtdetr_latency_worker.py"
        ).read_bytes(),
        "contract.json": json.dumps(
            latency_contract, sort_keys=True
        ).encode("utf-8"),
        "input_descriptor.json": json.dumps(
            descriptor, sort_keys=True
        ).encode("utf-8"),
    }
    encoded = {
        name: base64.b64encode(content).decode("ascii")
        for name, content in files.items()
    }
    script = (
        "import base64,json,pathlib;"
        "root=pathlib.Path('/tmp/rtdetr_campaign_runtime');"
        f"files=json.loads({json.dumps(json.dumps(encoded))});"
        "[(root/name).parent.mkdir(parents=True,exist_ok=True) "
        "for name in files];"
        "[(root/name).write_bytes(base64.b64decode(data)) "
        "for name,data in files.items()]"
    )
    command = f"python -c {shlex.quote(script)}"
    return command.replace("{", "{{").replace("}", "}}"), latency_contract


def _launch_latency(
    sdk: Any,
    contract: Mapping[str, Any],
    qualification_manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
    checkpoint: str,
    checkpoint_id: str,
    fingerprint: str,
    *,
    events: Path,
    mode: str,
    candidate_id: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    from tao_automl.latency_benchmark import combine_replica_records
    from tao_sdk.script_runner import build_entrypoint

    descriptor = latency_input_descriptor(
        contract,
        qualification_manifest,
        spec,
        checkpoint_id,
    )
    installer, latency_contract = _payload_command(contract, descriptor)
    benchmark_spec = copy.deepcopy(dict(spec))
    benchmark_spec["dataset"]["batch_size"] = 1
    benchmark_spec["evaluate"]["batch_size"] = 1
    command = " ".join(
        [
            installer,
            "&& torchrun --standalone --nproc_per_node=8",
            "/tmp/rtdetr_campaign_runtime/rtdetr_latency_worker.py",
            "--config {config_path}",
            "--checkpoint",
            shlex.quote(checkpoint),
            "--contract /tmp/rtdetr_campaign_runtime/contract.json",
            "--input-descriptor "
            "/tmp/rtdetr_campaign_runtime/input_descriptor.json",
            "--candidate-fingerprint",
            shlex.quote(fingerprint),
            "--runtime-modules-root /tmp/rtdetr_campaign_runtime",
            "--output-root "
            '"$TAO_RESULTS_ROOT/$TAO_JOB_ID/latency"',
        ]
    )
    action = yaml.safe_load(
        (
            Path(contract["runtime"]["skill_dir"])
            / "references/skill_info.yaml"
        ).read_text(encoding="utf-8")
    )["actions"]["evaluate"]
    entrypoint = build_entrypoint(
        command=command,
        specs=benchmark_spec,
        inputs=action["inputs"],
        outputs={},
        config_format="yaml",
        upload_excludes=action.get("upload_excludes", []),
    )
    runtime = contract["runtime"]
    job = sdk.create_job(
        image=runtime["sqsh_path"],
        command=entrypoint["command"],
        gpu_count=8,
        num_nodes=1,
        partition=runtime["partition"],
        account=runtime["account"],
    )
    evidence = {
        "tao_job_id": job.id,
        "status": "submitted",
        "submitted_at_utc": utc_timestamp(),
        "spec_sha256": canonical_sha256(benchmark_spec),
        "command_sha256": text_sha256(entrypoint["command"]),
        "checkpoint_id": checkpoint_id,
        "candidate_fingerprint": fingerprint,
        "input_descriptor": descriptor,
        "input_sha256": canonical_sha256(descriptor),
        "contract_sha256": canonical_sha256(latency_contract),
    }
    status = _wait_for_job(
        sdk,
        job.id,
        events=events,
        phase="selection_time_latency",
        mode=mode,
        candidate_id=candidate_id,
    )
    evidence.update(
        {"status": status, "terminal_at_utc": utc_timestamp()}
    )
    logs = sdk.get_job_logs(job.id, tail=5000)
    if (
        status != "Complete"
        or "TAO_AUTOML_RTDETR_LATENCY_COMPLETE" not in logs
    ):
        raise CampaignExecutionError(
            f"latency job {job.id} ended as {status}: {logs[-3000:]}"
        )
    root = _local_lustre_path(sdk.get_job_results_dir(job.id))
    reader = (
        "import glob,json,sys;"
        "paths=sorted(glob.glob(sys.argv[1]+'/rank_*.json'));"
        "print(json.dumps([json.load(open(path)) for path in paths]))"
    )
    records = json.loads(
        remote_output(
            f"python3 -c {shlex.quote(reader)} "
            f"{shlex.quote(root + '/latency')}"
        )
    )
    if (
        len(records) != 8
        or {item.get("tao_job_id") for item in records} != {job.id}
    ):
        raise CampaignExecutionError(
            "latency evidence is not eight-replica job-isolated data"
        )
    aggregate = combine_replica_records(records)
    statistics = aggregate["statistics"]
    if (
        not statistics["is_valid"]
        or statistics["raw_sample_count_total"] != 4000
    ):
        raise CampaignExecutionError(
            "selection-time latency failed its frozen quality gate"
        )
    low, high = statistics["bootstrap_median_ci_ms"]
    metrics = {
        "latency_ms": float(statistics["median_ms"]),
        "latency_p95_ms": float(statistics["p95_ms"]),
        "latency_ci95_low_ms": float(low),
        "latency_ci95_high_ms": float(high),
    }
    evidence["result_root"] = root
    evidence["aggregate"] = aggregate
    evidence["quality_gate_passed"] = True
    return metrics, evidence


def _metric_extractor(logs: str, metric_name: str) -> float | None:
    if metric_name != "mAP50":
        return None
    values = [
        float(value)
        for pattern in MAP50_PATTERNS
        for value in pattern.findall(logs)
    ]
    return values[-1] if values else None


def _await_first_candidate_release(
    *,
    runtime_root: Path,
    mode: str,
    evidence: Mapping[str, Any],
) -> None:
    gate_dir = runtime_root / "first_candidate_gate"
    atomic_json(gate_dir / f"{mode}.json", evidence)
    release = gate_dir / "release.json"
    while not release.is_file():
        time.sleep(2)
    decision = json.loads(release.read_text(encoding="utf-8"))
    if (
        decision.get("release_remaining_budget") is not True
        or decision.get("modes") != list(campaign_contract.MODES)
    ):
        raise CampaignExecutionError(
            f"first-candidate gate rejected {mode}: "
            f"{decision.get('reason', 'unspecified')}"
        )


def _run_mode(
    contract_path: str,
    qualification_manifest_path: str,
    qualification_completion_path: str,
    runtime_root: str,
    mode: str,
    resume: bool,
) -> None:
    contract = validate_execution_contract(
        json.loads(Path(contract_path).read_text(encoding="utf-8"))
    )
    qualification_manifest = _load_qualification_manifest(
        qualification_manifest_path
    )
    decision = audit_qualification(
        qualification_completion_path,
        expected_manifest_sha256=qualification_manifest["manifest_sha256"],
    )
    decision.assert_runtime_ready()
    root = Path(runtime_root)
    mode_dir = root / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    events = mode_dir / "events.jsonl"
    evidence_path = mode_dir / "candidate_evidence.json"
    candidates: dict[str, Any] = {}
    if resume and evidence_path.is_file():
        evidence_document = json.loads(
            evidence_path.read_text(encoding="utf-8")
        )
        if (
            evidence_document.get("schema_version") != 1
            or evidence_document.get("contract_sha256")
            != contract["contract_sha256"]
            or evidence_document.get("mode") != mode
            or not isinstance(
                evidence_document.get("candidates"), Mapping
            )
        ):
            raise CampaignExecutionError(
                f"{mode} resume candidate evidence is incompatible"
            )
        candidates = copy.deepcopy(
            dict(evidence_document["candidates"])
        )

    configure_slurm_runtime(contract)
    from tao_automl.runner import AutoMLRunner
    from tao_sdk.platforms.slurm import SlurmSDK

    resolved_inventory = build_live_runtime_inventory(
        contract=contract,
        qualification_manifest=qualification_manifest,
        decision=decision,
        mode=mode,
        cache_root=(
            Path(runtime_root) / "verified_ptm_cache"
        ),
    )
    sdk = SlurmSDK(
        poll_interval=10,
        state_file=mode_dir / "slurm_state.json",
    )
    runner = AutoMLRunner(
        sdk=sdk,
        skill_dir=Path(contract["runtime"]["skill_dir"]),
        action="train",
        poll_interval=10,
    )

    def persist() -> None:
        atomic_json(
            evidence_path,
            {
                "schema_version": 1,
                "contract_sha256": contract["contract_sha256"],
                "mode": mode,
                "candidates": candidates,
            },
        )

    def on_recommendation(rec: Any) -> None:
        record = _immutable_recommendation_record(rec, mode)
        _preserve_or_add_recommendation(candidates, record)
        persist()

    def evaluate_candidate(rec: Any, train_job_id: str) -> dict[str, float]:
        candidate_id = f"{mode}_rec_{rec.id}"
        record = candidates[candidate_id]
        terminal_checkpoint = _terminal_checkpoint(sdk, train_job_id)
        checkpoint = terminal_checkpoint["path"]
        spec = evaluation_spec(
            contract,
            qualification_manifest,
            rec.specs,
            checkpoint,
        )
        record.update(
            {
                "status": "evaluating",
                "train_job_id": train_job_id,
                "checkpoint": checkpoint,
                "terminal_checkpoint": terminal_checkpoint,
            }
        )
        persist()
        map50, accuracy_job = _launch_evaluation(
            sdk,
            contract,
            spec,
            events=events,
            mode=mode,
            candidate_id=candidate_id,
        )
        latency, latency_job = _launch_latency(
            sdk,
            contract,
            qualification_manifest,
            spec,
            checkpoint,
            record["checkpoint_id"],
            record["candidate_fingerprint"],
            events=events,
            mode=mode,
            candidate_id=candidate_id,
        )
        objectives = {"mAP50": map50, **latency}
        record.update(
            {
                "status": "success",
                "objective_values": objectives,
                "accuracy_evaluation": accuracy_job,
                "selection_time_latency": latency_job,
                "measurement_role": "selection_time",
            }
        )
        persist()
        return objectives

    first_result_seen = (
        root / "first_candidate_gate" / f"{mode}.json"
    ).is_file()

    def on_result(rec: Any, metric: Any, status: str) -> None:
        nonlocal first_result_seen
        candidate_id = f"{mode}_rec_{rec.id}"
        record = candidates.setdefault(candidate_id, {})
        record["automl_status"] = status
        record["reported_metric"] = metric
        if str(status).lower() not in SUCCESS_RECOMMENDATION_STATUSES:
            record["status"] = "terminal_failure"
            record["failure_reason"] = getattr(rec, "failure_reason", None)
        persist()
        if not first_result_seen:
            first_result_seen = True
            objectives = record.get("objective_values")
            success = (
                str(status).lower() in SUCCESS_RECOMMENDATION_STATUSES
                and isinstance(objectives, Mapping)
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in objectives.values()
                )
                and record.get("selection_time_latency", {}).get(
                    "quality_gate_passed"
                )
                is True
            )
            _await_first_candidate_release(
                runtime_root=root,
                mode=mode,
                evidence={
                    "schema_version": 1,
                    "contract_sha256": contract["contract_sha256"],
                    "mode": mode,
                    "candidate_id": candidate_id,
                    "passed": success,
                    "objective_values": copy.deepcopy(objectives),
                    "accuracy_evaluation_status": record.get(
                        "accuracy_evaluation", {}
                    ).get("status"),
                    "latency_quality_gate_passed": record.get(
                        "selection_time_latency", {}
                    ).get("quality_gate_passed"),
                    "agent_intervention_flags": {
                        name: False
                        for name in campaign_contract.AGENT_FLAGS
                    },
                },
            )

    profile = campaign_contract.profile_overrides(
        qualification_manifest
    )
    result = runner.run(
        train_dataset_uri="",
        eval_dataset_uri="",
        workspace_id=f"{contract['campaign_id']}-{mode}",
        image=contract["runtime"]["sqsh_path"],
        automl_settings=campaign_contract.mode_settings(
            str(contract["campaign_id"]),
            mode,
        ),
        automl_hyperparameters=list(
            campaign_contract.SEARCH_PARAMETERS
        ),
        custom_param_ranges=campaign_contract.custom_ranges(),
        workspace_path=str(mode_dir / "workspace"),
        spec_overrides=profile,
        metric_extractor=_metric_extractor,
        eval_fn=evaluate_candidate,
        on_recommendation=on_recommendation,
        on_result=on_result,
        resume=resume,
        ptm_aware_runtime=True,
        resolved_ptm_inventory=resolved_inventory,
        gpu_count=8,
        num_nodes=1,
        partition=contract["runtime"]["partition"],
        account=contract["runtime"]["account"],
    )
    atomic_json(
        mode_dir / "result.json",
        {
            "schema_version": 1,
            "contract_sha256": contract["contract_sha256"],
            "mode": mode,
            "status": "success",
            "result": result,
        },
    )


def _release_first_candidate_gate(
    runtime_root: Path,
    processes: Mapping[str, mp.Process],
    contract_sha256: str,
) -> dict[str, Any] | None:
    gate_dir = runtime_root / "first_candidate_gate"
    release = gate_dir / "release.json"
    if release.is_file():
        return json.loads(release.read_text(encoding="utf-8"))
    cells = {}
    for mode in campaign_contract.MODES:
        path = gate_dir / f"{mode}.json"
        if path.is_file():
            cells[mode] = json.loads(path.read_text(encoding="utf-8"))
    dead_without_cell = [
        mode
        for mode, process in processes.items()
        if not process.is_alive() and mode not in cells
    ]
    if dead_without_cell:
        value = {
            "schema_version": 1,
            "contract_sha256": contract_sha256,
            "release_remaining_budget": False,
            "modes": list(campaign_contract.MODES),
            "reason": (
                "mode controller terminated before first-candidate evidence: "
                + ", ".join(sorted(dead_without_cell))
            ),
            "generated_automatically": True,
        }
        atomic_json(release, value)
        return value
    if set(cells) != set(campaign_contract.MODES):
        return None
    valid = all(
        cell.get("passed") is True
        and cell.get("contract_sha256") == contract_sha256
        and cell.get("mode") == mode
        for mode, cell in cells.items()
    )
    value = {
        "schema_version": 1,
        "contract_sha256": contract_sha256,
        "release_remaining_budget": valid,
        "modes": list(campaign_contract.MODES),
        "reason": (
            "all three real first candidates passed evaluation, latency, "
            "quality, and provenance gates"
            if valid
            else "one or more real first candidates failed a frozen gate"
        ),
        "first_candidates": cells,
        "generated_automatically": True,
        "generated_at_utc": utc_timestamp(),
    }
    atomic_json(release, value)
    return value


def launch_all_modes(
    contract_path: Path,
    qualification_manifest: Path,
    qualification_completion: Path,
    runtime_root: Path,
    *,
    resume: bool,
) -> dict[str, int | None]:
    contract = validate_execution_contract(
        json.loads(contract_path.read_text(encoding="utf-8"))
    )
    context = mp.get_context("spawn")
    processes = {
        mode: context.Process(
            target=_run_mode,
            args=(
                str(contract_path),
                str(qualification_manifest),
                str(qualification_completion),
                str(runtime_root),
                mode,
                resume,
            ),
            name=f"rtdetr-automl-{mode}",
        )
        for mode in campaign_contract.MODES
    }
    for process in processes.values():
        process.start()

    def forward(signum: int, _frame: object) -> None:
        for process in processes.values():
            if process.is_alive() and process.pid:
                os.kill(process.pid, signum)

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    remaining = dict(processes)
    exit_codes: dict[str, int | None] = {}
    while remaining:
        _release_first_candidate_gate(
            runtime_root,
            processes,
            contract["contract_sha256"],
        )
        for mode, process in list(remaining.items()):
            process.join(timeout=1)
            if not process.is_alive():
                exit_codes[mode] = process.exitcode
                remaining.pop(mode)
        if remaining:
            time.sleep(1)
    atomic_json(runtime_root / "mode_process_status.json", exit_codes)
    return exit_codes


def automatic_successor(
    *,
    qualification_manifest_path: Path,
    qualification_completion_path: Path,
    repository: Path,
    wheel_path: Path,
    runtime_root: Path,
    contract_path: Path,
    launch: bool,
) -> dict[str, Any]:
    """Evaluate the qualification gate and optionally launch immediately."""
    qualification_manifest = _load_qualification_manifest(
        qualification_manifest_path
    )
    decision = audit_qualification(
        qualification_completion_path,
        expected_manifest_sha256=qualification_manifest["manifest_sha256"],
    )
    gate_record = {
        "schema_version": 1,
        "qualification_completion": str(qualification_completion_path),
        "qualification_evidence_sha256": decision.evidence_sha256,
        "decision": decision.to_dict(),
        "launch_requested": launch,
        "evaluated_at_utc": utc_timestamp(),
    }
    if not decision.runtime_ready:
        gate_record.update(
            {
                "successor_launched": False,
                "status": "blocked",
                "reason": "qualification passed but registry promotion is pending",
            }
        )
        atomic_json(runtime_root / "automatic_successor_gate.json", gate_record)
        return gate_record
    base = campaign_contract.build_preregistered_contract(
        campaign_id="rtdetr-synthetic-objective-aware-three-mode-20260730",
        qualification_manifest=qualification_manifest,
        decision=decision,
    )
    sealed = seal_execution_contract(
        base,
        repository=repository,
        wheel_path=wheel_path,
    )
    atomic_json(contract_path, sealed)
    gate_record.update(
        {
            "status": "ready",
            "campaign_contract": str(contract_path),
            "campaign_contract_sha256": sealed["contract_sha256"],
            "successor_launched": False,
        }
    )
    atomic_json(runtime_root / "automatic_successor_gate.json", gate_record)
    if not launch:
        return gate_record
    loaded_names = load_env_file(ENV_PATH)
    configure_slurm_runtime(sealed)
    remote_contract = qualification_runner.verify_remote_contract(
        qualification_manifest
    )
    runtime_root.mkdir(parents=True, exist_ok=True)
    try:
        live_preflight = verify_live_runtime_preflight(
            contract=sealed,
            qualification_manifest=qualification_manifest,
            decision=decision,
            cache_root=runtime_root / "verified_ptm_cache",
        )
    except Exception as exc:
        gate_record.update(
            {
                "successor_launched": False,
                "status": "blocked",
                "reason": (
                    "live typed four-PTM runtime preflight failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )
        atomic_json(
            runtime_root / "automatic_successor_gate.json",
            gate_record,
        )
        raise
    atomic_json(
        runtime_root / "live_runtime_preflight.json",
        live_preflight,
    )
    atomic_json(
        runtime_root / "submission_provenance.json",
        {
            "schema_version": 1,
            "contract_sha256": sealed["contract_sha256"],
            "qualification_evidence_sha256": decision.evidence_sha256,
            "loaded_secret_keys": list(loaded_names),
            "secret_values_recorded": False,
            "sqsh_path": sealed["runtime"]["sqsh_path"],
            "sqsh_sha256": sealed["runtime"]["sqsh_sha256"],
            "remote_contract": remote_contract,
            "live_runtime_preflight_sha256": live_preflight[
                "record_sha256"
            ],
            "nodes_per_child": 1,
            "gpus_per_child": 8,
            "cpu_runs": 0,
            "smoke_runs": 0,
        },
    )
    gate_record["successor_launched"] = True
    gate_record["status"] = "launched"
    atomic_json(runtime_root / "automatic_successor_gate.json", gate_record)
    exit_codes = launch_all_modes(
        contract_path,
        qualification_manifest_path,
        qualification_completion_path,
        runtime_root,
        resume=False,
    )
    gate_record["mode_exit_codes"] = exit_codes
    gate_record["status"] = (
        "complete"
        if set(exit_codes) == set(campaign_contract.MODES)
        and all(code == 0 for code in exit_codes.values())
        else "terminal_with_failures"
    )
    atomic_json(runtime_root / "automatic_successor_gate.json", gate_record)
    return gate_record


def wait_for_successful_qualification(
    completion_path: Path,
    *,
    expected_manifest_sha256: str,
    poll_seconds: float = 10.0,
    status_path: Path | None = None,
) -> QualificationDecision:
    """Wait for exact resumed success and supported registry readiness.

    Missing and non-terminal artifacts may become ready.  An immutable
    terminal failure at the requested completion path fails immediately.
    Once qualification succeeds, repository registry promotion may arrive
    later and is polled without weakening the fail-closed runtime gate.
    """
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")

    def record(status: str, **details: Any) -> None:
        if status_path is None:
            return
        atomic_json(
            status_path,
            {
                "schema_version": 1,
                "status": status,
                "qualification_completion": str(completion_path),
                "successor_launched": False,
                "evaluated_at_utc": utc_timestamp(),
                **details,
            },
        )

    while True:
        if not completion_path.is_file():
            record(
                "waiting_qualification",
                reason="resumed qualification completion is unavailable",
            )
            time.sleep(poll_seconds)
            continue
        try:
            document = json.loads(
                completion_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            record(
                "waiting_qualification",
                reason="resumed qualification completion is not readable",
            )
            time.sleep(poll_seconds)
            continue
        if (
            document.get("terminal") is True
            and document.get("status") != "success"
        ):
            reason = (
                "immutable resumed qualification completion is terminal "
                f"with status {document.get('status')!r}"
            )
            record("blocked", reason=reason)
            raise QualificationGateError(reason)
        if document.get("status") != "success":
            record(
                "waiting_qualification",
                reason="resumed qualification is not terminal",
            )
            time.sleep(poll_seconds)
            continue
        if document.get("terminal") is not True:
            record(
                "waiting_qualification",
                reason="resumed qualification success is not terminal",
            )
            time.sleep(poll_seconds)
            continue
        try:
            decision = audit_qualification(
                completion_path,
                expected_manifest_sha256=expected_manifest_sha256,
            )
        except QualificationGateError as exc:
            record("blocked", reason=str(exc))
            raise
        if not decision.runtime_ready:
            codes = {item.get("code") for item in decision.blockers}
            if codes != {"registry_status_not_supported"}:
                decision.assert_runtime_ready()
            record(
                "waiting_registry_promotion",
                reason=(
                    "qualification passed; supported registry promotion "
                    "is pending"
                ),
                decision=decision.to_dict(),
            )
            time.sleep(poll_seconds)
            continue
        record(
            "ready",
            reason=(
                "resumed qualification and supported registry gates passed"
            ),
            decision=decision.to_dict(),
        )
        return decision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qualification-manifest",
        type=Path,
        default=DEFAULT_QUALIFICATION_MANIFEST,
    )
    parser.add_argument(
        "--qualification-completion",
        type=Path,
        default=DEFAULT_COMPLETION,
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path("/localhome/local-rarunachalam/tao-automl"),
    )
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=DEFAULT_RUNTIME_ROOT,
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument(
        "--wait-for-qualification",
        action="store_true",
        help=(
            "Wait for the exact resumed qualification completion before "
            "evaluating the automatic successor gate."
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    try:
        if args.wait_for_qualification:
            qualification_manifest = _load_qualification_manifest(
                args.qualification_manifest.resolve()
            )
            wait_for_successful_qualification(
                args.qualification_completion.resolve(),
                expected_manifest_sha256=(
                    qualification_manifest["manifest_sha256"]
                ),
                poll_seconds=args.poll_seconds,
                status_path=(
                    args.runtime_root.resolve()
                    / "automatic_successor_gate.json"
                ),
            )
        result = automatic_successor(
            qualification_manifest_path=args.qualification_manifest.resolve(),
            qualification_completion_path=(
                args.qualification_completion.resolve()
            ),
            repository=args.repository.resolve(),
            wheel_path=args.wheel.resolve(),
            runtime_root=args.runtime_root.resolve(),
            contract_path=args.contract.resolve(),
            launch=args.launch,
        )
    except QualificationGateError as exc:
        raise CampaignExecutionError(str(exc)) from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] not in {"terminal_with_failures"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
