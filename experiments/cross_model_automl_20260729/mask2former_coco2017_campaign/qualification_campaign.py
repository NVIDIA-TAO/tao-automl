#!/usr/bin/env python3

"""Stage and qualify the official Mask2Former PTM on full COCO.

``--stage`` is strictly data-only: it resolves the exact official NGC member,
downloads and checksums it, and publishes it read-only on Lustre. ``--launch``
is the only path that constructs a scheduler client or submits jobs. It runs
no CPU/model smoke and no mini-step: the official PTM receives a real
three-epoch, one-node/eight-A100 full-dataset train followed by standalone full
validation. Missing task-correct COCO mask AP is retained as a terminal
failure, never replaced by semantic mIoU and never retried with a different
PTM or specification.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shlex
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from tao_automl.ptm_preflight import (
    AtomicArtifactCache,
    NGCCredential,
    NGCHTTPSClient,
)
from tao_automl.ptm_registry import (
    canonical_sha256,
    load_ptm_registry,
    merge_ptm_spec_precedence,
)

from . import (
    campaign_contract,
    checkpoint_resume,
    run_campaign,
    runtime_overlay,
)


DEFAULT_CONTRACT = Path(
    campaign_contract.FROZEN_V3_QUALIFICATION_CONTRACT["path"]
)
DEFAULT_RUNTIME_ROOT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask2former_coco2017_ptm_qualification_v3"
)
# Runtime v3 reuses the immutable data-only v1 PTM stage. It never writes to
# the v1 runtime tree or republishes the checkpoint.
DEFAULT_STAGE_MANIFEST = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask2former_coco2017_ptm_qualification_v1/ptm_stage_manifest.json"
)
DEFAULT_LOCAL_CACHE = Path(
    "/localhome/local-rarunachalam/.tao/cache/"
    "mask2former_coco2017_ptm_qualification_v1"
)
DEFAULT_LUSTRE_INPUT_ROOT = Path(
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
    "cross_model_automl_20260729/"
    "mask2former_coco2017_ptm_qualification_v1/inputs"
)
QUALIFICATION_CAMPAIGN_ID = (
    "mask2former-coco2017-direct-full-qualification-v3-20260801"
)
ENV_PATH = run_campaign.ENV_PATH
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
            "immutable Mask2Former v3 qualification contract changed"
        )
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CampaignExecutionError(
            "immutable Mask2Former v3 qualification contract is invalid"
        ) from exc
    payload = copy.deepcopy(document)
    supplied = payload.pop("contract_sha256", None)
    if (
        supplied != frozen["contract_sha256"]
        or supplied != canonical_sha256(payload)
    ):
        raise CampaignExecutionError(
            "immutable Mask2Former v3 qualification contract integrity failed"
        )
    return document

VALIDATION_MASK_AP_METRIC = "segm_val_mAP"
STANDALONE_MASK_AP_METRIC = "segm_test_mAP"
STANDALONE_MASK_AP50_METRIC = "segm_test_mAP50"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _records() -> tuple[dict[str, Any], ...]:
    registry = load_ptm_registry()
    snapshot = campaign_contract.mask2former_registry_snapshot()
    return tuple(
        copy.deepcopy(registry.checkpoint(item["id"]))
        for item in snapshot["records"]
    )


def _safe_component(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not text:
        raise CampaignExecutionError("empty qualification path component")
    return text


def _ssh_options() -> list[str]:
    options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    key = os.environ.get("SSH_KEY_PATH")
    if key:
        options.extend(["-i", key])
    return options


def _ssh_target() -> str:
    hostname = os.environ["SLURM_HOSTNAME"].split(",", 1)[0].strip()
    user = os.environ["SLURM_USER"].strip()
    if not hostname or not user:
        raise CampaignExecutionError("SLURM SSH routing is incomplete")
    return f"{user}@{hostname}"


def _run(
    command: Sequence[str],
    *,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _remote_mode(path: str) -> str:
    mode = run_campaign.remote_output(
        f"stat -c %a {shlex.quote(path)}"
    ).strip()
    if re.fullmatch(r"[0-7]{3,4}", mode) is None:
        raise CampaignExecutionError(
            f"remote file mode is invalid: {path}"
        )
    return mode


def _publish_file(
    source: Path,
    destination: str,
    *,
    expected_size: int,
    expected_sha256: str,
) -> dict[str, Any]:
    """Publish one PTM atomically and leave its final bytes read-only."""
    if (
        not source.is_file()
        or source.stat().st_size != expected_size
        or campaign_contract.sha256_file(source) != expected_sha256
    ):
        raise CampaignExecutionError(
            f"local staged PTM identity changed: {source}"
        )
    try:
        existing = run_campaign._remote_file_identity(destination)
    except CampaignExecutionError:
        existing = None
    if existing is not None:
        mode = _remote_mode(destination)
        if (
            existing["size_bytes"] != expected_size
            or existing["sha256"] != expected_sha256
            or int(mode, 8) & 0o222
        ):
            raise CampaignExecutionError(
                f"existing immutable PTM differs: {destination}"
            )
        return {**existing, "mode": mode, "cache_hit": True}

    final_path = Path(destination)
    temporary = (
        final_path.parent
        / f".{final_path.name}.partial-{uuid.uuid4().hex}"
    )
    run_campaign.remote_output(
        f"mkdir -p {shlex.quote(str(final_path.parent))}"
    )
    _run(
        [
            "scp",
            *_ssh_options(),
            str(source),
            f"{_ssh_target()}:{temporary}",
        ],
        timeout=7200,
    )
    quoted_temporary = shlex.quote(str(temporary))
    quoted_final = shlex.quote(destination)
    run_campaign.remote_output(
        " && ".join(
            [
                f"test \"$(stat -c %s {quoted_temporary})\" = "
                f"{shlex.quote(str(expected_size))}",
                f"test \"$(sha256sum {quoted_temporary} | cut -d ' ' -f1)\" "
                f"= {shlex.quote(expected_sha256)}",
                f"chmod 0444 {quoted_temporary}",
                f"test ! -e {quoted_final}",
                f"mv {quoted_temporary} {quoted_final}",
            ]
        ),
        timeout=1800,
    )
    identity = run_campaign._remote_file_identity(destination)
    mode = _remote_mode(destination)
    if (
        identity["size_bytes"] != expected_size
        or identity["sha256"] != expected_sha256
        or int(mode, 8) & 0o222
    ):
        raise CampaignExecutionError(
            f"published PTM verification failed: {destination}"
        )
    return {**identity, "mode": mode, "cache_hit": False}


def validate_stage_document(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a data-only stage independently of the later campaign seal."""
    value = copy.deepcopy(dict(document))
    supplied = value.pop("manifest_sha256", None)
    if supplied != canonical_sha256(value):
        raise CampaignExecutionError("PTM stage manifest integrity failed")
    records = _records()
    rows = value.get("checkpoints")
    if (
        value.get("schema_version") != 1
        or value.get("model") != "mask2former"
        or _SHA256_RE.fullmatch(
            str(value.get("registry_sha256", ""))
        )
        is None
        or value.get("stage_complete") is not True
        or value.get("remote_read_only") is not True
        or value.get("cpu_model_runs") != 0
        or value.get("gpu_model_runs") != 0
        or value.get("smoke_model_runs") != 0
        or value.get("mini_step_runs") != 0
        or value.get("scheduler_jobs_submitted") != 0
        or not isinstance(rows, list)
        or tuple(item.get("id") for item in rows)
        != tuple(record["id"] for record in records)
    ):
        raise CampaignExecutionError(
            "Mask2Former PTM stage campaign contract changed"
        )
    by_id = {record["id"]: record for record in records}
    for row in rows:
        record = by_id[row["id"]]
        path = row.get("path")
        if (
            not isinstance(path, str)
            or not path.startswith("/lustre/")
            or row.get("size_bytes") != record["expected_size_bytes"]
            or _SHA256_RE.fullmatch(str(row.get("sha256", ""))) is None
            or row.get("immutable_source_identity")
            != record["source"]["immutable_identity"]
            or row.get("remote_read_only") is not True
            or int(str(row.get("mode", "0")), 8) & 0o222
            or (
                record.get("sha256") is not None
                and row["sha256"] != record["sha256"]
            )
        ):
            raise CampaignExecutionError(
                f"staged PTM identity changed: {row.get('id')}"
            )
    value["manifest_sha256"] = supplied
    return value


def verify_stage_remote(document: Mapping[str, Any]) -> dict[str, Any]:
    """Re-hash the data-only stage without running or loading a model."""
    stage = validate_stage_document(document)
    checked = []
    for row in stage["checkpoints"]:
        observed = run_campaign._remote_file_identity(row["path"])
        mode = _remote_mode(row["path"])
        if (
            observed["size_bytes"] != row["size_bytes"]
            or observed["sha256"] != row["sha256"]
            or int(mode, 8) & 0o222
        ):
            raise CampaignExecutionError(
                f"remote staged PTM changed: {row['id']}"
            )
        checked.append({**observed, "mode": mode, "id": row["id"]})
    return {
        "stage_manifest_sha256": stage["manifest_sha256"],
        "checked": checked,
        "all_read_only": True,
        "model_runs": 0,
        "scheduler_jobs_submitted": 0,
    }


def stage_runtime_inputs(
    *,
    local_cache_root: str | Path = DEFAULT_LOCAL_CACHE,
    lustre_input_root: str | Path = DEFAULT_LUSTRE_INPUT_ROOT,
) -> dict[str, Any]:
    """Download and publish every official Mask2Former PTM, data-only."""
    cache = AtomicArtifactCache(
        Path(local_cache_root).expanduser().resolve() / "ngc"
    )
    client = NGCHTTPSClient(NGCCredential.from_environment())
    lustre_root = Path(lustre_input_root)
    rows = []
    for record in _records():
        reference = client.resolve_member(record["source"])
        probe = client.probe_member(reference)
        if (
            probe.ok is not True
            or (
                probe.remote_size_bytes is not None
                and probe.remote_size_bytes
                != record["expected_size_bytes"]
            )
        ):
            raise CampaignExecutionError(
                f"exact NGC member preflight failed for "
                f"{record['id']}: {probe.code}"
            )
        checkpoint = cache.fetch_ngc_member(
            checkpoint_id=record["id"],
            reference=reference,
            expected_size_bytes=record["expected_size_bytes"],
            expected_sha256=record.get("sha256"),
            client=client,
        )
        destination = (
            lustre_root
            / "ptms"
            / _safe_component(record["id"])
            / record["source"]["member"]
        )
        remote = _publish_file(
            checkpoint.path,
            str(destination),
            expected_size=checkpoint.size_bytes,
            expected_sha256=checkpoint.sha256,
        )
        rows.append(
            {
                "id": record["id"],
                "path": remote["path"],
                "size_bytes": remote["size_bytes"],
                "sha256": remote["sha256"],
                "mode": remote["mode"],
                "immutable_source_identity": record["source"][
                    "immutable_identity"
                ],
                "verification_mode": checkpoint.verification_mode,
                "source_identity_sha256": (
                    checkpoint.source_identity_sha256
                ),
                "access_probe": probe.to_dict(),
                "remote_read_only": True,
            }
        )
    document = {
        "schema_version": 1,
        "model": "mask2former",
        "registry_sha256": campaign_contract.mask2former_registry_snapshot()[
            "registry_sha256"
        ],
        "created_at_utc": utc_timestamp(),
        "stage_complete": True,
        "remote_read_only": True,
        "cpu_model_runs": 0,
        "gpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "scheduler_jobs_submitted": 0,
        "checkpoints": rows,
    }
    document["manifest_sha256"] = canonical_sha256(document)
    return validate_stage_document(document)


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
        "campaign_id": QUALIFICATION_CAMPAIGN_ID,
        "contract_revision": "qualification_runtime_v3",
        "contract_sha256": contract["contract_sha256"],
        "model": "mask2former",
        "task": "instance_segmentation",
        "primary_metric": VALIDATION_MASK_AP_METRIC,
        "standalone_reported_metric": STANDALONE_MASK_AP_METRIC,
        "standalone_objective_binding": {
            "reported_metric": STANDALONE_MASK_AP_METRIC,
            "canonical_metric": VALIDATION_MASK_AP_METRIC,
        },
        "semantic_miou_accepted_as_mask_ap": False,
        "official_checkpoint_ids": [
            record["id"] for record in inventory["records"]
        ],
        "workflow_count": inventory["record_count"],
        "full_dataset": True,
        "training_epochs": campaign_contract.FROZEN_TRAINING_EPOCHS,
        "standalone_full_validation": True,
        "nodes_per_job": 1,
        "gpus_per_job": 8,
        "hardware": copy.deepcopy(campaign_contract.FROZEN_HARDWARE),
        "sqsh": copy.deepcopy(campaign_contract.FROZEN_SQSH),
        "walltime_policy": copy.deepcopy(
            contract["runtime"]["walltime_policy"]
        ),
        "checkpoint_interval_epochs": (
            campaign_contract.FROZEN_CHECKPOINT_INTERVAL_EPOCHS
        ),
        "checkpoint_resume_policy": (
            "same_job_exact_epoch_step_max_v1"
        ),
        "slurm_self_requeue": (
            campaign_contract.FROZEN_SLURM_USE_REQUEUE
        ),
        "tao_pytorch_overlay": copy.deepcopy(
            contract["runtime"]["tao_pytorch_overlay"]
        ),
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "replacement_workflows_allowed": False,
        "registry_bypass_allowed": False,
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
        or document.get("model") != "mask2former"
        or not isinstance(document.get("registry_sha256"), str)
        or re.fullmatch(
            r"[0-9a-f]{64}", document["registry_sha256"]
        )
        is None
        or document.get("stage_complete") is not True
        or document.get("remote_read_only") is not True
        or document.get("cpu_model_runs") != 0
        or document.get("gpu_model_runs") != 0
        or document.get("smoke_model_runs") != 0
        or document.get("mini_step_runs") != 0
        or document.get("scheduler_jobs_submitted") != 0
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
            "PTM stage must contain exactly every official Mask2Former arm"
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
    action_command = runtime_overlay.wrap_command(
        action["command"],
        contract["runtime"]["tao_pytorch_overlay"],
    )
    if action_name == "train":
        action_command = checkpoint_resume.wrap_train_command(
            action_command
        )
    entrypoint = build_entrypoint(
        command=_gpu_guard(
            action_command
        ),
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
    records = _status_records(sdk, job_id, action=action)
    values: list[float] = []
    for record in records:
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


def _status_records(
    sdk: Any,
    job_id: str,
    *,
    action: str,
) -> list[dict[str, Any]]:
    root = run_campaign._local_lustre_path(
        sdk.get_job_results_dir(job_id)
    )
    path = f"{root}/results_dir/{action}/status.json"
    output = run_campaign.remote_output(
        f"(test -f {shlex.quote(path)} && "
        f"cat {shlex.quote(path)}) || true"
    )
    records: list[dict[str, Any]] = []
    for line in output.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _status_epoch_values(
    sdk: Any,
    job_id: str,
    *,
    action: str,
    names: tuple[str, ...],
) -> list[float]:
    """Return one deterministic metric value per explicit training epoch.

    TAO emits the same validation KPI twice: a generic ``Eval metrics`` event
    and a training-progress event carrying ``epoch`` and ``step``.  DDP ranks
    may also repeat the structured event.  Only structured epoch records are
    eligible; exact repeats are collapsed by epoch, while conflicting finite
    values for the same epoch fail closed.
    """
    by_epoch: dict[int, list[tuple[int, float]]] = {}
    for record in _status_records(sdk, job_id, action=action):
        epoch = record.get("epoch")
        step = record.get("step")
        kpi = record.get("kpi")
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
            or isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
            or not isinstance(kpi, Mapping)
        ):
            continue
        for name in names:
            try:
                value = float(kpi[name])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(value):
                by_epoch.setdefault(epoch, []).append((step, value))
                break
    values: list[float] = []
    for epoch in sorted(by_epoch):
        records = by_epoch[epoch]
        distinct = {value for _, value in records}
        if len(distinct) != 1:
            raise CampaignExecutionError(
                f"conflicting task metric values for training epoch {epoch}"
            )
        values.append(max(records, key=lambda item: item[0])[1])
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
        "walltime_policy": copy.deepcopy(
            contract["runtime"]["walltime_policy"]
        ),
        "checkpoint_resume_policy": (
            "same_job_exact_epoch_step_max_v1"
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
        mask_values = _status_epoch_values(
            sdk,
            train_job.id,
            action="train",
            names=(VALIDATION_MASK_AP_METRIC,),
        )
        diagnostics["train_job"]["mask_ap_values"] = mask_values
        diagnostics["train_job"]["metric_deduplication"] = (
            "explicit_epoch_then_exact_rank_value_v1"
        )
        diagnostics["train_job"]["semantic_miou_diagnostic_values"] = (
            _status_values(
                sdk,
                train_job.id,
                action="train",
                names=("mIoU", "val_mIoU", "miou"),
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
            names=(STANDALONE_MASK_AP_METRIC,),
        )
        standalone_mask50_values = _status_values(
            sdk,
            evaluation_job.id,
            action="evaluate",
            names=(STANDALONE_MASK_AP50_METRIC,),
        )
        diagnostics["evaluation_job"]["mask_ap_values"] = (
            standalone_mask_values
        )
        diagnostics["evaluation_job"]["mask_ap50_values"] = (
            standalone_mask50_values
        )
        diagnostics["evaluation_job"]["reported_metric"] = (
            STANDALONE_MASK_AP_METRIC
        )
        diagnostics["evaluation_job"]["canonical_objective_metric"] = (
            VALIDATION_MASK_AP_METRIC
        )
        diagnostics["evaluation_job"][
            "semantic_miou_diagnostic_values"
        ] = _status_values(
            sdk,
            evaluation_job.id,
            action="evaluate",
            names=("mIoU", "val_mIoU", "miou"),
        )
        if (
            len(mask_values)
            != campaign_contract.FROZEN_TRAINING_EPOCHS
            or not standalone_mask_values
        ):
            return _failure_workflow(
                checkpoint_id,
                "task-correct segm_val_mAP was not emitted by every "
                "in-epoch validation or standalone evaluation did not emit "
                "segm_test_mAP; semantic mIoU diagnostics are not accepted",
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
                "segm_val_mAP": mask_values[-1],
                "terminal_checkpoint": terminal,
                "tao_job_id": train_job.id,
            },
            "evaluation": {
                "status": "Complete",
                "full_validation_split": True,
                "nodes": 1,
                "gpus": 8,
                STANDALONE_MASK_AP_METRIC: standalone_mask_values[-1],
                STANDALONE_MASK_AP50_METRIC: (
                    standalone_mask50_values[-1]
                    if standalone_mask50_values
                    else None
                ),
                "objective_binding": {
                    "reported_metric": STANDALONE_MASK_AP_METRIC,
                    "canonical_metric": VALIDATION_MASK_AP_METRIC,
                    "value": standalone_mask_values[-1],
                },
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
        "campaign_id": QUALIFICATION_CAMPAIGN_ID,
        "contract_revision": "qualification_runtime_v3",
        "model": "mask2former",
        "task": "instance_segmentation",
        "primary_metric": VALIDATION_MASK_AP_METRIC,
        "standalone_reported_metric": STANDALONE_MASK_AP_METRIC,
        "standalone_objective_binding": {
            "reported_metric": STANDALONE_MASK_AP_METRIC,
            "canonical_metric": VALIDATION_MASK_AP_METRIC,
        },
        "semantic_miou_accepted_as_mask_ap": False,
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
        "tao_pytorch_overlay": copy.deepcopy(
            contract["runtime"]["tao_pytorch_overlay"]
        ),
        "walltime_policy": copy.deepcopy(
            contract["runtime"]["walltime_policy"]
        ),
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "replacement_workflows_submitted": False,
        "workflows": [copy.deepcopy(dict(item)) for item in workflows],
    }
    value["evidence_sha256"] = canonical_sha256(value)
    return value


def launch(
    *,
    contract_path: Path,
    runtime_root: Path,
    env_path: Path = ENV_PATH,
) -> dict[str, Any]:
    """Submit the one frozen direct-full workflow; no replacement is made."""
    contract = load_frozen_v3_contract(contract_path)
    runtime_root.mkdir(parents=True, exist_ok=True)
    loaded_names = run_campaign.load_env_file(env_path)
    run_campaign.configure_slurm_runtime(contract)
    local = run_campaign.verify_local_contract(contract)
    dataset = run_campaign._verify_dataset_remote(contract)
    sqsh = run_campaign._remote_file_identity(contract["sqsh"]["path"])
    if sqsh["sha256"] != contract["sqsh"]["sha256"]:
        raise CampaignExecutionError("pinned SQSH identity changed")
    overlay = run_campaign.verify_runtime_overlay_remote(contract)
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
            "tao_pytorch_overlay": overlay,
            "ptm_stage_manifest_path": contract[
                "qualification_policy"
            ]["ptm_stage_manifest_path"],
            "ptm_stage_sha256": campaign_contract.sha256_file(
                contract["qualification_policy"]["ptm_stage_manifest_path"]
            ),
            "nodes_per_job": 1,
            "gpus_per_job": 8,
            "walltime_policy": copy.deepcopy(
                contract["runtime"]["walltime_policy"]
            ),
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
    sdk = SlurmSDK(
        poll_interval=10,
        state_file=runtime_root / "slurm_state.json",
    )
    workflows = [
        _run_one(contract, sdk, checkpoint_id, source, runtime_root)
        for checkpoint_id, source in sorted(staged.items())
    ]
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
    parser.add_argument(
        "--stage-manifest",
        type=Path,
        default=DEFAULT_STAGE_MANIFEST,
    )
    parser.add_argument(
        "--local-cache-root",
        type=Path,
        default=DEFAULT_LOCAL_CACHE,
    )
    parser.add_argument(
        "--lustre-input-root",
        type=Path,
        default=DEFAULT_LUSTRE_INPUT_ROOT,
    )
    parser.add_argument("--env-file", type=Path, default=ENV_PATH)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--stage", action="store_true")
    actions.add_argument("--check-stage", action="store_true")
    actions.add_argument("--launch", action="store_true")
    arguments = parser.parse_args(argv)

    if arguments.stage or arguments.check_stage:
        run_campaign.load_env_file(arguments.env_file.resolve())
        stage_path = arguments.stage_manifest.resolve()
        if stage_path.is_file():
            stage = validate_stage_document(
                json.loads(stage_path.read_text(encoding="utf-8"))
            )
        elif arguments.check_stage:
            raise CampaignExecutionError(
                f"PTM stage manifest is unavailable: {stage_path}"
            )
        else:
            stage = stage_runtime_inputs(
                local_cache_root=arguments.local_cache_root,
                lustre_input_root=arguments.lustre_input_root,
            )
            atomic_json(stage_path, stage)
        checked = verify_stage_remote(stage)
        print(
            json.dumps(
                {
                    "stage_manifest_path": str(stage_path),
                    "stage_manifest_sha256": stage["manifest_sha256"],
                    "ptm_count": len(stage["checkpoints"]),
                    "remote_verification": checked,
                    "model_runs": 0,
                    "scheduler_jobs_submitted": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    contract = load_frozen_v3_contract(arguments.contract)
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
