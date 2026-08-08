#!/usr/bin/env python3

"""Qualify every official SegFormer PTM with a direct full GPU workflow.

The controller has two explicit phases:

* ``--stage`` is data-only.  It downloads all 13 exact NGC members, verifies
  their immutable identities and sizes, generates the per-PTM train/evaluate
  specifications, and publishes every input read-only on Lustre.
* ``--launch`` verifies the sealed stage, reuses only the four exact v4 train
  phases with positive load evidence, runs fresh full-VOC2012 50-epoch trains
  for the other nine arms, and submits a new standalone evaluation for all 13.

There is no CPU/model smoke, mini-step, fallback checkpoint, replacement
workflow, successful-train reexecution, or manual PTM exclusion path.  Every
terminal outcome is preserved.
The completion document is the exact input consumed by ``qualification_gate``.
It cannot mutate the repository registry.  A separately sealed successor may
bind exact successful, positive-load workflows into a campaign-local runtime
view when the base record already contains complete license metadata.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import shlex
import subprocess
import tempfile
import time
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
from tao_automl.ptm_registry import canonical_sha256, load_ptm_registry

try:
    from . import campaign_contract, run_campaign
    from .qualification_gate import audit_qualification
except ImportError:  # pragma: no cover - direct script execution
    repository = Path(__file__).resolve().parents[3]
    import sys

    sys.path.insert(0, str(repository))
    from experiments.cross_model_automl_20260729.segformer_voc2012_campaign import (
        campaign_contract,
        run_campaign,
    )
    from experiments.cross_model_automl_20260729.segformer_voc2012_campaign import (
        qualification_gate,
    )

    audit_qualification = qualification_gate.audit_qualification


HERE = Path(__file__).resolve().parent
DEFAULT_CONTRACT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "segformer_voc2012_three_mode/campaign.v5.json"
)
DEFAULT_RUNTIME_ROOT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "segformer_voc2012_ptm_qualification_v5"
)
DEFAULT_STAGE_MANIFEST = DEFAULT_RUNTIME_ROOT / "ptm_stage_manifest.json"
DEFAULT_LOCAL_CACHE = Path(
    "/localhome/local-rarunachalam/.tao/cache/"
    "segformer_voc2012_ptm_qualification_v5"
)
DEFAULT_LUSTRE_INPUT_ROOT = Path(
    "/lustre/fsw/portfolios/edgeai/projects/"
    "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/"
    "cross_model_automl_20260729/"
    "segformer_voc2012_ptm_qualification_v5/inputs"
)
QUALIFICATION_CAMPAIGN_ID = campaign_contract.QUALIFICATION_CAMPAIGN_ID
EVALUATION_CHECKPOINT_SENTINEL = (
    "__TERMINAL_CHECKPOINT_FROM_THIS_WORKFLOW__"
)
# Keep the one-node Lightning rendezvous below the usual Linux ephemeral
# range and make it unique to the SLURM allocation.  Qualification v2 let
# Lightning choose a transient high port; one otherwise valid eight-GPU job
# failed when that port was claimed before rank zero could bind it.
QUALIFICATION_MASTER_PORT_BASE = 15000
QUALIFICATION_MASTER_PORT_SPAN = 10000
TERMINAL_JOB_STATUSES = frozenset({"Complete", "Error", "Canceled"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PRETRAINED_LOAD_REPORT_PREFIX = "SEGFORMER_PRETRAINED_LOAD_REPORT "
_PRETRAINED_LOAD_COUNT_FIELDS = (
    "loaded_tensor_count",
    "missing_tensor_count",
    "shape_mismatched_tensor_count",
    "unmatched_tensor_count",
    "non_tensor_count",
)

CampaignExecutionError = run_campaign.CampaignExecutionError
atomic_json = run_campaign.atomic_json
append_jsonl = run_campaign.append_jsonl
remote_output = run_campaign.remote_output
utc_timestamp = run_campaign.utc_timestamp


def _records() -> tuple[dict[str, Any], ...]:
    """Return the exact official inventory in deterministic ID order."""
    registry = load_ptm_registry()
    snapshot = campaign_contract.segformer_registry_snapshot()
    return tuple(
        copy.deepcopy(registry.checkpoint(item["id"]))
        for item in snapshot["records"]
    )


def registry_core_identity(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Identity that must survive independent registry status promotion."""
    keys = (
        "id",
        "source",
        "expected_size_bytes",
        "model_family",
        "architecture",
        "backbone",
        "checkpoint_target",
        "input_contract",
        "task_compatibility",
    )
    return {
        key: copy.deepcopy(record[key])
        for key in keys
    }


def _safe_component(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not text:
        raise CampaignExecutionError("empty qualification path component")
    return text


def _workflow_id(checkpoint_id: str) -> str:
    return f"qualify-{_safe_component(checkpoint_id)}"


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


def _remote_identity(
    path: str,
    *,
    timeout: int = 1800,
) -> dict[str, Any] | None:
    quoted = shlex.quote(path)
    output = remote_output(
        (
            f"if test -f {quoted}; then "
            f"stat -c '%s %a' {quoted}; sha256sum {quoted}; "
            "else printf 'ABSENT\\n'; fi"
        ),
        timeout=timeout,
    ).strip().splitlines()
    if output == ["ABSENT"]:
        return None
    if len(output) != 2:
        raise CampaignExecutionError(
            f"remote identity is incomplete: {path}"
        )
    size_text, mode = output[0].split()
    digest = output[1].split()[0]
    try:
        size = int(size_text)
    except ValueError as exc:
        raise CampaignExecutionError(
            f"remote size is invalid: {path}"
        ) from exc
    if (
        size < 1
        or _SHA256_RE.fullmatch(digest) is None
        or re.fullmatch(r"[0-7]{3,4}", mode) is None
    ):
        raise CampaignExecutionError(
            f"remote file identity is invalid: {path}"
        )
    return {
        "path": path,
        "size_bytes": size,
        "sha256": digest,
        "mode": mode,
    }


def verify_slurm_preflight(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Recheck the skill-required control plane without reserving GPUs."""
    runtime = contract["runtime"]
    if (
        runtime["partition"]
        != campaign_contract.FROZEN_SLURM_PARTITION
        or runtime["time_hours"]
        != campaign_contract.FROZEN_SLURM_TIME_HOURS
        or runtime["timeout_hours"]
        != campaign_contract.FROZEN_SLURM_TIMEOUT_HOURS
    ):
        raise CampaignExecutionError(
            "frozen SLURM resource policy changed"
        )
    qualification = contract["qualification_policy"]
    overlay = qualification.get("runtime_overlay")
    infrastructure_policy = qualification.get(
        "infrastructure_retry_policy"
    )
    phase_recovery_policy = qualification.get("phase_recovery_policy")
    if (
        qualification.get("revision")
        != campaign_contract.QUALIFICATION_REVISION
        or qualification.get("recipe_fidelity")
        != campaign_contract.FROZEN_QUALIFICATION_FIDELITY
        or overlay
        != campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
        or infrastructure_policy
        != campaign_contract.FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY
        or phase_recovery_policy
        != campaign_contract.FROZEN_QUALIFICATION_PHASE_RECOVERY_POLICY
    ):
        raise CampaignExecutionError(
            "qualification v5 fidelity or recovery policy changed"
        )
    run_campaign.configure_slurm_runtime(contract)
    sdk_dir = Path(runtime["sdk_dir"]).resolve()
    sdk_source = (sdk_dir / "tao_sdk" / "__init__.py").resolve()
    if not sdk_source.is_file() or not sdk_source.is_relative_to(sdk_dir):
        raise CampaignExecutionError(
            f"sealed tao_sdk package is unavailable: {sdk_source}"
        )
    partition = shlex.quote(runtime["partition"])
    sqsh = shlex.quote(contract["sqsh"]["path"])
    overlay_archive = shlex.quote(overlay["archive_path"])
    overlay_installer = shlex.quote(overlay["installer_path"])
    output = remote_output(
        "set -eu; "
        "for command in sbatch squeue sacct srun; do "
        "command -v \"$command\" >/dev/null; done; "
        f"test -r {sqsh}; "
        f"test -r {overlay_archive}; "
        f"test \"$(stat -c '%s' {overlay_archive})\" = "
        f"{overlay['archive_size_bytes']}; "
        f"test \"$(sha256sum {overlay_archive} | awk '{{print $1}}')\" = "
        f"{shlex.quote(overlay['archive_sha256'])}; "
        f"test -r {overlay_installer}; "
        f"test \"$(stat -c '%s' {overlay_installer})\" = "
        f"{overlay['installer_size_bytes']}; "
        f"test \"$(sha256sum {overlay_installer} | awk '{{print $1}}')\" = "
        f"{shlex.quote(overlay['installer_sha256'])}; "
        f"scontrol show partition {partition} -o "
        "| grep -q 'MaxTime=04:00:00'; "
        "printf 'READY\\n'"
    ).strip()
    if output != "READY":
        raise CampaignExecutionError(
            "SLURM control-plane preflight did not complete"
        )
    return {
        "status": "ready",
        "passwordless_ssh": True,
        "scheduler_commands": ["sbatch", "squeue", "sacct", "srun"],
        "partition": runtime["partition"],
        "partition_max_time": "04:00:00",
        "sdk_source": str(sdk_source),
        "sqsh_readable": True,
        "qualification_runtime_overlay": copy.deepcopy(overlay),
        "qualification_infrastructure_retry_policy": copy.deepcopy(
            infrastructure_policy
        ),
        "qualification_phase_recovery_policy": copy.deepcopy(
            phase_recovery_policy
        ),
        "scheduler_jobs_submitted": 0,
    }


def _publish_file(
    source: Path,
    destination: str,
    *,
    expected_size: int,
    expected_sha256: str,
) -> dict[str, Any]:
    """Publish one verified input without exposing a partial final file."""
    if (
        not source.is_file()
        or source.stat().st_size != expected_size
        or campaign_contract.sha256_file(source) != expected_sha256
    ):
        raise CampaignExecutionError(
            f"local staged input identity changed: {source}"
        )
    existing = _remote_identity(destination)
    if existing is not None:
        if (
            existing["size_bytes"] != expected_size
            or existing["sha256"] != expected_sha256
            or int(existing["mode"], 8) & 0o222
        ):
            raise CampaignExecutionError(
                f"existing immutable destination differs: {destination}"
            )
        existing["cache_hit"] = True
        return existing

    final_path = Path(destination)
    temporary = (
        final_path.parent
        / f".{final_path.name}.partial-{uuid.uuid4().hex}"
    )
    remote_output(
        f"mkdir -p {shlex.quote(str(final_path.parent))}",
        timeout=60,
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
    remote_output(
        " && ".join(
            [
                f"test \"$(stat -c '%s' {quoted_temporary})\" = "
                f"{shlex.quote(str(expected_size))}",
                f"test \"$(sha256sum {quoted_temporary} | "
                f"awk '{{print $1}}')\" = "
                f"{shlex.quote(expected_sha256)}",
                f"chmod a-w {quoted_temporary}",
                f"test ! -e {quoted_final}",
                f"mv {quoted_temporary} {quoted_final}",
            ]
        ),
        timeout=1800,
    )
    identity = _remote_identity(destination)
    if (
        identity is None
        or identity["size_bytes"] != expected_size
        or identity["sha256"] != expected_sha256
        or int(identity["mode"], 8) & 0o222
    ):
        raise CampaignExecutionError(
            f"published input verification failed: {destination}"
        )
    identity["cache_hit"] = False
    return identity


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.partial-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _nested_checkpoint_paths(
    specification: dict[str, Any],
    *,
    target: str,
    checkpoint_path: str,
) -> None:
    specification["train"]["pretrained_model_path"] = ""
    specification["model"]["backbone"][
        "pretrained_backbone_path"
    ] = ""
    if target == "train.pretrained_model_path":
        specification["train"]["pretrained_model_path"] = checkpoint_path
    elif target == "model.backbone.pretrained_backbone_path":
        specification["model"]["backbone"][
            "pretrained_backbone_path"
        ] = checkpoint_path
    else:
        raise CampaignExecutionError(
            f"unsupported SegFormer checkpoint target: {target}"
        )


def qualification_specs(
    contract: Mapping[str, Any],
    record: Mapping[str, Any],
    checkpoint_path: str,
) -> dict[str, dict[str, Any]]:
    """Build exact full-run train and standalone-evaluate specifications."""
    skill_dir = Path(contract["runtime"]["skill_dir"])
    train = yaml.safe_load(
        (
            skill_dir / "references/spec_template_train.yaml"
        ).read_text(encoding="utf-8")
    )
    evaluate = yaml.safe_load(
        (
            skill_dir / "references/spec_template_evaluate.yaml"
        ).read_text(encoding="utf-8")
    )
    profile = campaign_contract.qualification_profile_overrides(
        contract["dataset"]["prepared_root"]
    )
    train = run_campaign._merge_spec(train, profile)
    evaluate = run_campaign._merge_spec(evaluate, profile)
    for specification in (train, evaluate):
        specification["model"]["backbone"]["type"] = record["backbone"]
        _nested_checkpoint_paths(
            specification,
            target=record["checkpoint_target"],
            checkpoint_path=checkpoint_path,
        )
        specification["results_dir"] = ""
        specification["wandb"]["enable"] = False
    evaluate["evaluate"].update(
        {
            "num_gpus": 8,
            "gpu_ids": list(range(8)),
            "num_nodes": 1,
            "checkpoint": EVALUATION_CHECKPOINT_SENTINEL,
            "trt_engine": "",
            "results_dir": "",
            "batch_size": campaign_contract.FROZEN_BATCH_SIZE_PER_REPLICA,
            "vis_after_n_batches": 1000000,
        }
    )
    return {"train": train, "evaluate": evaluate}


def _spec_artifact(
    *,
    local_root: Path,
    lustre_root: Path,
    checkpoint_id: str,
    action: str,
    document: Mapping[str, Any],
    base_template_path: Path,
) -> dict[str, Any]:
    content = yaml.safe_dump(
        copy.deepcopy(dict(document)),
        sort_keys=True,
    ).encode("utf-8")
    relative = Path(_safe_component(checkpoint_id)) / f"{action}.yaml"
    local_path = local_root / relative
    _write_bytes_atomic(local_path, content)
    digest = hashlib.sha256(content).hexdigest()
    remote = _publish_file(
        local_path,
        str(lustre_root / relative),
        expected_size=len(content),
        expected_sha256=digest,
    )
    return {
        "action": action,
        "document": copy.deepcopy(dict(document)),
        "document_sha256": canonical_sha256(document),
        "raw_yaml_sha256": digest,
        "size_bytes": len(content),
        "base_template": {
            "path": str(base_template_path),
            "sha256": campaign_contract.sha256_file(base_template_path),
        },
        "local_path": str(local_path),
        "lustre": remote,
    }


def _sealed_predecessor_json(
    path: str,
    *,
    whole_file_sha256: str,
    internal_key: str,
    internal_sha256: str,
) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if (
        not resolved.is_file()
        or campaign_contract.sha256_file(resolved) != whole_file_sha256
    ):
        raise CampaignExecutionError(
            f"sealed predecessor file identity changed: {resolved}"
        )
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignExecutionError(
            f"sealed predecessor JSON is invalid: {resolved}"
        ) from exc
    if not isinstance(value, Mapping):
        raise CampaignExecutionError(
            f"sealed predecessor JSON root changed: {resolved}"
        )
    value = copy.deepcopy(dict(value))
    payload = copy.deepcopy(value)
    supplied = payload.pop(internal_key, None)
    if supplied != internal_sha256 or supplied != canonical_sha256(payload):
        raise CampaignExecutionError(
            f"sealed predecessor internal identity changed: {resolved}"
        )
    return value


def _v4_phase_recovery_records() -> dict[str, dict[str, Any]]:
    """Resolve exact v4 train phases eligible for bounded v5 reuse."""
    frozen = campaign_contract.FROZEN_V4_QUALIFICATION_EVIDENCE
    completion = _sealed_predecessor_json(
        frozen["completion_path"],
        whole_file_sha256=frozen["completion_whole_file_sha256"],
        internal_key="evidence_sha256",
        internal_sha256=frozen["evidence_sha256"],
    )
    stage = _sealed_predecessor_json(
        frozen["ptm_stage_manifest_path"],
        whole_file_sha256=frozen[
            "ptm_stage_manifest_whole_file_sha256"
        ],
        internal_key="stage_manifest_sha256",
        internal_sha256=frozen["ptm_stage_manifest_sha256"],
    )
    audit = _sealed_predecessor_json(
        frozen["ptm_load_audit_path"],
        whole_file_sha256=frozen["ptm_load_audit_whole_file_sha256"],
        internal_key="audit_sha256",
        internal_sha256=frozen["ptm_load_audit_sha256"],
    )
    if (
        completion.get("campaign_id") != frozen["campaign_id"]
        or completion.get("terminal") is not True
        or completion.get("successful_workflows") != 0
        or completion.get("failed_workflows") != 13
        or stage.get("campaign_id") != frozen["campaign_id"]
        or audit.get("schema_version") != 2
        or audit.get("qualification_campaign_id") != frozen["campaign_id"]
        or audit.get("positive_load_workflows") != 4
        or audit.get("ptm_load_failure_workflows") != 9
        or audit.get("all_missing_all_unexpected_backbone_prefix_workflows")
        != 9
    ):
        raise CampaignExecutionError(
            "sealed v4 phase-recovery evidence changed"
        )
    expected_ids = {record["id"] for record in _records()}

    def indexed(rows: Any, label: str) -> dict[str, Mapping[str, Any]]:
        if not isinstance(rows, list):
            raise CampaignExecutionError(f"v4 {label} rows are unavailable")
        result = {
            item.get("checkpoint_id"): item
            for item in rows
            if isinstance(item, Mapping)
        }
        if len(rows) != 13 or set(result) != expected_ids:
            raise CampaignExecutionError(f"v4 {label} rows are ambiguous")
        return result

    completions = indexed(completion.get("workflows"), "completion")
    stages = indexed(stage.get("ptms"), "stage")
    audits = indexed(audit.get("workflows"), "load-audit")
    reusable = set(
        campaign_contract.FROZEN_V4_REUSABLE_TRAIN_CHECKPOINT_IDS
    )
    fresh = set(campaign_contract.FROZEN_V5_FRESH_TRAIN_CHECKPOINT_IDS)
    if reusable & fresh or reusable | fresh != expected_ids:
        raise CampaignExecutionError("v5 phase-recovery partition changed")

    records = {}
    for checkpoint_id in sorted(expected_ids):
        workflow = completions[checkpoint_id]
        staged = stages[checkpoint_id]
        audited = audits[checkpoint_id]
        workflow_payload = copy.deepcopy(dict(workflow))
        workflow_sha256 = workflow_payload.pop("workflow_sha256", None)
        if (
            workflow_sha256 != canonical_sha256(workflow_payload)
            or workflow.get("source_checkpoint") != staged.get("checkpoint")
            or audited.get("source_checkpoint") != staged.get("checkpoint")
            or audited.get("workflow_sha256") != workflow_sha256
        ):
            raise CampaignExecutionError(
                f"v4 workflow/stage/audit identity changed: {checkpoint_id}"
            )
        audit_row_sha256 = canonical_sha256(audited)
        if checkpoint_id in fresh:
            if (
                audited.get("ptm_load_success") is not False
                or audited.get("load_evidence", {}).get("classification")
                != "all_missing_all_unexpected_backbone_prefix"
            ):
                raise CampaignExecutionError(
                    f"v5 fresh-train reason changed: {checkpoint_id}"
                )
            records[checkpoint_id] = {
                "mode": "run_fresh_full_train",
                "new_train_job_required": True,
                "new_standalone_evaluation_job_required": True,
                "v4_workflow_sha256": workflow_sha256,
                "v4_load_audit_row_sha256": audit_row_sha256,
                "v4_load_classification": (
                    "all_missing_all_unexpected_backbone_prefix"
                ),
            }
            continue

        train = workflow.get("jobs", {}).get("train")
        status_evidence = (
            train.get("status_evidence") if isinstance(train, Mapping) else None
        )
        terminal_checkpoint = (
            train.get("terminal_checkpoint")
            if isinstance(train, Mapping)
            else None
        )
        evaluate_job = workflow.get("jobs", {}).get("evaluate")
        load = audited.get("load_evidence", {})
        positive = load.get("unique_positive_observations")
        if (
            workflow.get("status") != "failure"
            or workflow.get("failure_phase") != "standalone_evaluation"
            or not isinstance(train, Mapping)
            or train.get("status") != "Complete"
            or train.get("spec_sha256")
            != staged.get("specs", {}).get("train", {}).get(
                "document_sha256"
            )
            or train.get("staged_spec_sha256")
            != staged.get("specs", {}).get("train", {}).get(
                "raw_yaml_sha256"
            )
            or train.get("nodes") != 1
            or train.get("gpus") != 8
            or train.get("runtime_overlay_required") is not True
            or not isinstance(status_evidence, Mapping)
            or status_evidence.get("validation_record_count")
            != campaign_contract.FROZEN_QUALIFICATION_TRAINING_EPOCHS
            or not isinstance(status_evidence.get("validation_metrics"), list)
            or len(status_evidence["validation_metrics"])
            != campaign_contract.FROZEN_QUALIFICATION_TRAINING_EPOCHS
            or status_evidence.get("terminal_success") is not True
            or status_evidence.get("terminal_success_message")
            != "Train finished successfully."
            or status_evidence.get("val_miou")
            != status_evidence["validation_metrics"][-1].get("val_miou")
            or not isinstance(terminal_checkpoint, Mapping)
            or terminal_checkpoint.get("training_epochs")
            != campaign_contract.FROZEN_QUALIFICATION_TRAINING_EPOCHS
            or terminal_checkpoint.get("terminal_epoch_index") != 49
            or terminal_checkpoint.get("naming_contract")
            != "model_epoch_049_step_numeric"
            or terminal_checkpoint.get("ambiguity_policy") != "fail_closed"
            or not str(terminal_checkpoint.get("path", "")).startswith(
                "/lustre/"
            )
            or not isinstance(terminal_checkpoint.get("size_bytes"), int)
            or terminal_checkpoint["size_bytes"] < 1
            or _SHA256_RE.fullmatch(
                str(terminal_checkpoint.get("sha256", ""))
            )
            is None
            or audited.get("ptm_load_success") is not True
            or load.get("classification") != "positive_compatible_tensor_load"
            or load.get("expected_component") != "model"
            or not isinstance(positive, list)
            or len(positive) != 1
            or positive[0].get("checkpoint")
            != staged.get("checkpoint", {}).get("path")
            or positive[0].get("loaded_tensor_count", 0) < 1
            or _SHA256_RE.fullmatch(
                str(positive[0].get("loaded_keyset_sha256", ""))
            )
            is None
            or not (
                evaluate_job is None
                or (
                    isinstance(evaluate_job, Mapping)
                    and evaluate_job.get("checkpoint")
                    == terminal_checkpoint
                )
            )
        ):
            raise CampaignExecutionError(
                f"v4 train phase is not reusable: {checkpoint_id}"
            )
        receipt = {
            "schema_version": 2,
            "evidence_kind": "sealed_v4_legacy_positive_load_audit",
            "checkpoint": staged["checkpoint"]["path"],
            "component": "model",
            "loaded_tensor_count": positive[0]["loaded_tensor_count"],
            "loaded_keyset_sha256": positive[0]["loaded_keyset_sha256"],
            "status_record_occurrences": load[
                "positive_observation_occurrences"
            ],
            "v4_load_audit_path": frozen["ptm_load_audit_path"],
            "v4_load_audit_whole_file_sha256": frozen[
                "ptm_load_audit_whole_file_sha256"
            ],
            "v4_load_audit_sha256": frozen["ptm_load_audit_sha256"],
            "v4_load_audit_row_sha256": audit_row_sha256,
            "v4_workflow_sha256": workflow_sha256,
            "v4_train_log_sha256": audited["train_log"]["sha256"],
        }
        receipt["report_sha256"] = canonical_sha256(receipt)
        records[checkpoint_id] = {
            "mode": "reuse_sealed_v4_terminal_train",
            "new_train_job_required": False,
            "new_standalone_evaluation_job_required": True,
            "predecessor_qualification_revision": workflow[
                "qualification_revision"
            ],
            "predecessor_recipe_fidelity": copy.deepcopy(
                workflow["recipe_fidelity"]
            ),
            "predecessor_runtime_overlay": copy.deepcopy(
                workflow["runtime_overlay"]
            ),
            "v4_workflow_sha256": workflow_sha256,
            "v4_load_audit_row_sha256": audit_row_sha256,
            "source_checkpoint": copy.deepcopy(staged["checkpoint"]),
            "train_job": {
                "tao_job_id": train["tao_job_id"],
                "status": "Complete",
                "command_sha256": train["command_sha256"],
                "spec_sha256": train["spec_sha256"],
                "result_root": train["result_root"],
            },
            "terminal_checkpoint": copy.deepcopy(dict(terminal_checkpoint)),
            "validation_status_evidence": copy.deepcopy(
                dict(status_evidence)
            ),
            "pretrained_load": receipt,
        }
    expected_plan_hashes = (
        campaign_contract.FROZEN_QUALIFICATION_PHASE_RECOVERY_POLICY[
            "execution_plan_sha256_by_checkpoint_id"
        ]
    )
    if (
        set(expected_plan_hashes) != expected_ids
        or any(
            canonical_sha256(records[checkpoint_id])
            != expected_plan_hashes[checkpoint_id]
            for checkpoint_id in expected_ids
        )
    ):
        raise CampaignExecutionError(
            "sealed v5 per-arm phase-recovery plan changed"
        )
    return records


def stage_runtime_inputs(
    *,
    contract: Mapping[str, Any],
    local_cache_root: str | Path = DEFAULT_LOCAL_CACHE,
    lustre_input_root: str | Path = DEFAULT_LUSTRE_INPUT_ROOT,
) -> dict[str, Any]:
    """Download, checksum, and publish all PTMs and generated specs."""
    run_campaign.verify_local_contract(contract)
    verify_slurm_preflight(contract)
    run_campaign._verify_dataset_remote(contract)
    sqsh = run_campaign._remote_file_identity(contract["sqsh"]["path"])
    if sqsh["sha256"] != contract["sqsh"]["sha256"]:
        raise CampaignExecutionError("pinned SQSH identity changed")

    cache_root = Path(local_cache_root).expanduser().resolve()
    lustre_root = Path(lustre_input_root)
    cache = AtomicArtifactCache(cache_root / "ngc")
    client = NGCHTTPSClient(NGCCredential.from_environment())
    skill_dir = Path(contract["runtime"]["skill_dir"])
    templates = {
        "train": skill_dir / "references/spec_template_train.yaml",
        "evaluate": skill_dir / "references/spec_template_evaluate.yaml",
    }
    recovery_records = _v4_phase_recovery_records()
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
        checkpoint_destination = (
            lustre_root
            / "ptms"
            / _safe_component(record["id"])
            / record["source"]["member"]
        )
        recovery = recovery_records[record["id"]]
        if recovery["mode"] == "reuse_sealed_v4_terminal_train":
            remote_checkpoint = copy.deepcopy(recovery["source_checkpoint"])
            observed_reused = _remote_identity(remote_checkpoint["path"])
            if (
                observed_reused is None
                or observed_reused["size_bytes"]
                != remote_checkpoint["size_bytes"]
                or observed_reused["sha256"] != remote_checkpoint["sha256"]
                or int(observed_reused["mode"], 8) & 0o222
            ):
                raise CampaignExecutionError(
                    f"reused v4 source checkpoint changed: {record['id']}"
                )
            remote_checkpoint = {**remote_checkpoint, **observed_reused}
            remote_checkpoint["cache_hit"] = checkpoint.cache_hit
        else:
            remote_checkpoint = _publish_file(
                checkpoint.path,
                str(checkpoint_destination),
                expected_size=checkpoint.size_bytes,
                expected_sha256=checkpoint.sha256,
            )
        specifications = qualification_specs(
            contract,
            record,
            remote_checkpoint["path"],
        )
        spec_rows = {
            action: _spec_artifact(
                local_root=cache_root / "specs",
                lustre_root=lustre_root / "specs",
                checkpoint_id=record["id"],
                action=action,
                document=specifications[action],
                base_template_path=templates[action],
            )
            for action in ("train", "evaluate")
        }
        rows.append(
            {
                "checkpoint_id": record["id"],
                "workflow_id": _workflow_id(record["id"]),
                "registry_status_at_stage": record["status"],
                "registry_record_sha256": canonical_sha256(record),
                "registry_core_identity": registry_core_identity(record),
                "registry_core_identity_sha256": canonical_sha256(
                    registry_core_identity(record)
                ),
                "source": copy.deepcopy(record["source"]),
                "checkpoint_target": record["checkpoint_target"],
                "backbone": record["backbone"],
                "expected_size_bytes": record["expected_size_bytes"],
                "registered_sha256": record.get("sha256"),
                "observed_sha256": checkpoint.sha256,
                "verification_mode": checkpoint.verification_mode,
                "source_identity_sha256": (
                    checkpoint.source_identity_sha256
                ),
                "access_probe": probe.to_dict(),
                "checkpoint_specific_source_spec": {
                    "available": False,
                    "registry_field_present": (
                        "checkpoint_spec_file" in record
                    ),
                    "reason": (
                        "The official SegFormer registry record publishes no "
                        "checkpoint-specific YAML; the staged specs are "
                        "generated from the sealed TAO templates, frozen VOC "
                        "profile, and exact registry checkpoint target."
                    ),
                },
                "checkpoint": remote_checkpoint,
                "specs": spec_rows,
                "execution_plan": copy.deepcopy(recovery),
            }
        )

    document = {
        "schema_version": 2,
        "qualification_revision": campaign_contract.QUALIFICATION_REVISION,
        "campaign_id": QUALIFICATION_CAMPAIGN_ID,
        "automl_contract_sha256": contract["contract_sha256"],
        "created_at_utc": utc_timestamp(),
        "model": "segformer",
        "task": "semantic_segmentation",
        "registry_sha256": contract["ptm_inventory"]["registry_sha256"],
        "registry_version": contract["ptm_inventory"]["registry_version"],
        "source_policy": (
            "all_13_official_registry_arms_without_manual_exclusion"
        ),
        "dataset": {
            "prepared_root": contract["dataset"]["prepared_root"],
            "content_sha256": contract["dataset"]["content_sha256"],
            "stage_manifest_sha256": contract["dataset"][
                "stage_manifest_sha256"
            ],
            "train_pairs": 1464,
            "validation_pairs": 1449,
        },
        "runtime": {
            "sqsh_path": contract["sqsh"]["path"],
            "sqsh_sha256": contract["sqsh"]["sha256"],
            "sdk_commit": contract["runtime"]["sdk_commit"],
            "skills_commit": contract["runtime"]["skills_commit"],
            "source_commit": contract["runtime"]["source_commit"],
            "partition": contract["runtime"]["partition"],
            "time_hours": contract["runtime"]["time_hours"],
            "timeout_hours": contract["runtime"]["timeout_hours"],
            "nodes_per_workflow": 1,
            "gpus_per_workflow": 8,
            "required_gpu": copy.deepcopy(
                campaign_contract.FROZEN_HARDWARE
            ),
            "runtime_overlay": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
            ),
            "infrastructure_retry_policy": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY
            ),
            "phase_recovery_policy": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_PHASE_RECOVERY_POLICY
            ),
        },
        "recipe_fidelity": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_FIDELITY
        ),
        "prior_revision_evidence": copy.deepcopy(
            campaign_contract.FROZEN_PRIOR_QUALIFICATION_EVIDENCE
        ),
        "ptms": rows,
        "execution": {
            "operation": (
                "data_only_download_checksum_spec_generation_and_"
                "lustre_publication"
            ),
            "cpu_model_runs": 0,
            "gpu_model_runs": 0,
            "smoke_model_runs": 0,
            "mini_step_runs": 0,
            "checkpoint_loads": 0,
            "scheduler_jobs_submitted": 0,
            "fallback_ptms_used": 0,
            "manually_excluded_ptms": 0,
        },
        "agent_intervention_flags": {
            name: False for name in campaign_contract.AGENT_FLAGS
        },
    }
    document["stage_manifest_sha256"] = canonical_sha256(document)
    validate_stage_manifest(document, contract=contract)
    return document


def validate_stage_manifest(
    document: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one sealed data-only PTM/spec staging record."""
    value = copy.deepcopy(dict(document))
    supplied = value.pop("stage_manifest_sha256", None)
    if supplied != canonical_sha256(value):
        raise CampaignExecutionError("PTM stage manifest integrity failed")
    expected_ids = tuple(record["id"] for record in _records())
    rows = value.get("ptms")
    if (
        value.get("schema_version") != 2
        or value.get("qualification_revision")
        != campaign_contract.QUALIFICATION_REVISION
        or value.get("campaign_id") != QUALIFICATION_CAMPAIGN_ID
        or value.get("automl_contract_sha256")
        != contract["contract_sha256"]
        or value.get("model") != "segformer"
        or value.get("task") != "semantic_segmentation"
        or value.get("registry_sha256")
        != contract["ptm_inventory"]["registry_sha256"]
        or value.get("source_policy")
        != "all_13_official_registry_arms_without_manual_exclusion"
        or value.get("dataset")
        != {
            "prepared_root": contract["dataset"]["prepared_root"],
            "content_sha256": contract["dataset"]["content_sha256"],
            "stage_manifest_sha256": contract["dataset"][
                "stage_manifest_sha256"
            ],
            "train_pairs": 1464,
            "validation_pairs": 1449,
        }
        or value.get("runtime")
        != {
            "sqsh_path": contract["sqsh"]["path"],
            "sqsh_sha256": contract["sqsh"]["sha256"],
            "sdk_commit": contract["runtime"]["sdk_commit"],
            "skills_commit": contract["runtime"]["skills_commit"],
            "source_commit": contract["runtime"]["source_commit"],
            "partition": contract["runtime"]["partition"],
            "time_hours": contract["runtime"]["time_hours"],
            "timeout_hours": contract["runtime"]["timeout_hours"],
            "nodes_per_workflow": 1,
            "gpus_per_workflow": 8,
            "required_gpu": campaign_contract.FROZEN_HARDWARE,
            "runtime_overlay": (
                campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
            ),
            "infrastructure_retry_policy": (
                campaign_contract.FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY
            ),
            "phase_recovery_policy": (
                campaign_contract.FROZEN_QUALIFICATION_PHASE_RECOVERY_POLICY
            ),
        }
        or value.get("recipe_fidelity")
        != campaign_contract.FROZEN_QUALIFICATION_FIDELITY
        or value.get("prior_revision_evidence")
        != campaign_contract.FROZEN_PRIOR_QUALIFICATION_EVIDENCE
        or not isinstance(rows, list)
        or tuple(item.get("checkpoint_id") for item in rows)
        != expected_ids
        or len({item.get("workflow_id") for item in rows}) != 13
        or value.get("execution")
        != {
            "operation": (
                "data_only_download_checksum_spec_generation_and_"
                "lustre_publication"
            ),
            "cpu_model_runs": 0,
            "gpu_model_runs": 0,
            "smoke_model_runs": 0,
            "mini_step_runs": 0,
            "checkpoint_loads": 0,
            "scheduler_jobs_submitted": 0,
            "fallback_ptms_used": 0,
            "manually_excluded_ptms": 0,
        }
        or any(value["agent_intervention_flags"].values())
    ):
        raise CampaignExecutionError("PTM stage campaign contract changed")

    records = {record["id"]: record for record in _records()}
    recovery_records = _v4_phase_recovery_records()
    expected_plan_hashes = (
        campaign_contract.FROZEN_QUALIFICATION_PHASE_RECOVERY_POLICY[
            "execution_plan_sha256_by_checkpoint_id"
        ]
    )
    for row in rows:
        checkpoint_id = row["checkpoint_id"]
        record = records[checkpoint_id]
        checkpoint = row.get("checkpoint", {})
        if (
            row.get("workflow_id") != _workflow_id(checkpoint_id)
            or row.get("registry_record_sha256")
            != canonical_sha256(record)
            or row.get("registry_core_identity")
            != registry_core_identity(record)
            or row.get("registry_core_identity_sha256")
            != canonical_sha256(registry_core_identity(record))
            or row.get("source") != record["source"]
            or row.get("checkpoint_target")
            != record["checkpoint_target"]
            or row.get("backbone") != record["backbone"]
            or row.get("expected_size_bytes")
            != record["expected_size_bytes"]
            or row.get("registered_sha256") != record.get("sha256")
            or row.get("verification_mode")
            not in {
                "registered_sha256",
                "immutable_identity_observed_sha256",
            }
            or _SHA256_RE.fullmatch(
                str(row.get("source_identity_sha256", ""))
            )
            is None
            or row.get("access_probe", {}).get("ok") is not True
            or row.get("access_probe", {}).get("remote_size_bytes")
            not in {None, record["expected_size_bytes"]}
            or row.get("checkpoint_specific_source_spec")
            != {
                "available": False,
                "registry_field_present": (
                    "checkpoint_spec_file" in record
                ),
                "reason": (
                    "The official SegFormer registry record publishes no "
                    "checkpoint-specific YAML; the staged specs are "
                    "generated from the sealed TAO templates, frozen VOC "
                    "profile, and exact registry checkpoint target."
                ),
            }
            or checkpoint.get("size_bytes")
            != record["expected_size_bytes"]
            or checkpoint.get("sha256") != row.get("observed_sha256")
            or not str(checkpoint.get("path", "")).startswith("/lustre/")
            or _SHA256_RE.fullmatch(
                str(checkpoint.get("sha256", ""))
            )
            is None
            or int(str(checkpoint.get("mode", "0")), 8) & 0o222
            or row.get("execution_plan") != recovery_records[checkpoint_id]
            or canonical_sha256(row.get("execution_plan"))
            != expected_plan_hashes[checkpoint_id]
        ):
            raise CampaignExecutionError(
                f"staged checkpoint identity differs: {checkpoint_id}"
            )
        if (
            record.get("sha256") is not None
            and checkpoint["sha256"] != record["sha256"]
        ):
            raise CampaignExecutionError(
                f"registered checksum differs: {checkpoint_id}"
            )
        specs = row.get("specs")
        if not isinstance(specs, Mapping) or set(specs) != {
            "train",
            "evaluate",
        }:
            raise CampaignExecutionError(
                f"staged specs are incomplete: {checkpoint_id}"
            )
        for action in ("train", "evaluate"):
            spec = specs[action]
            remote = spec.get("lustre", {})
            base_template = (
                Path(contract["runtime"]["skill_dir"])
                / "references"
                / f"spec_template_{action}.yaml"
            )
            if (
                spec.get("action") != action
                or spec.get("document_sha256")
                != canonical_sha256(spec.get("document"))
                or spec.get("raw_yaml_sha256")
                != remote.get("sha256")
                or spec.get("size_bytes") != remote.get("size_bytes")
                or spec.get("base_template")
                != {
                    "path": str(base_template),
                    "sha256": campaign_contract.sha256_file(
                        base_template
                    ),
                }
                or not str(remote.get("path", "")).startswith("/lustre/")
                or int(str(remote.get("mode", "0")), 8) & 0o222
            ):
                raise CampaignExecutionError(
                    f"staged {action} spec differs: {checkpoint_id}"
                )
        train = specs["train"]["document"]
        evaluate = specs["evaluate"]["document"]
        if (
            train["train"]["num_epochs"]
            != campaign_contract.FROZEN_QUALIFICATION_TRAINING_EPOCHS
            or train["train"]["checkpoint_interval"]
            != campaign_contract.FROZEN_QUALIFICATION_TRAINING_EPOCHS
            or train["train"]["validation_interval"]
            != campaign_contract.FROZEN_QUALIFICATION_FIDELITY[
                "validation_interval"
            ]
            or train["train"]["optim"]["optim"]
            != campaign_contract.FROZEN_QUALIFICATION_FIDELITY[
                "optimizer"
            ]
            or train["train"]["optim"]["lr"]
            != campaign_contract.FROZEN_QUALIFICATION_FIDELITY[
                "learning_rate"
            ]
            or train["train"]["optim"]["weight_decay"]
            != campaign_contract.FROZEN_QUALIFICATION_FIDELITY[
                "weight_decay"
            ]
            or train["dataset"]["segment"]["augmentation"][
                "random_color"
            ]["enable"]
            is not False
            or train["dataset"]["segment"]["augmentation"][
                "with_random_blur"
            ]
            is not False
            or train["train"]["use_distributed_sampler"] is not True
            or train["train"]["num_gpus"] != 8
            or train["train"]["gpu_ids"] != list(range(8))
            or train["train"]["num_nodes"] != 1
            or train["dataset"]["segment"]["root_dir"]
            != contract["dataset"]["prepared_root"]
            or train["model"]["backbone"]["type"] != record["backbone"]
            or evaluate["train"]["num_epochs"]
            != campaign_contract.FROZEN_QUALIFICATION_TRAINING_EPOCHS
            or evaluate["train"]["checkpoint_interval"]
            != campaign_contract.FROZEN_QUALIFICATION_TRAINING_EPOCHS
            or evaluate["train"]["validation_interval"]
            != campaign_contract.FROZEN_QUALIFICATION_FIDELITY[
                "validation_interval"
            ]
            or evaluate["train"]["optim"]["optim"]
            != campaign_contract.FROZEN_QUALIFICATION_FIDELITY[
                "optimizer"
            ]
            or evaluate["train"]["optim"]["lr"]
            != campaign_contract.FROZEN_QUALIFICATION_FIDELITY[
                "learning_rate"
            ]
            or evaluate["train"]["optim"]["weight_decay"]
            != campaign_contract.FROZEN_QUALIFICATION_FIDELITY[
                "weight_decay"
            ]
            or evaluate["dataset"]["segment"]["augmentation"][
                "random_color"
            ]["enable"]
            is not False
            or evaluate["dataset"]["segment"]["augmentation"][
                "with_random_blur"
            ]
            is not False
            or evaluate["train"]["use_distributed_sampler"] is not True
            or evaluate["evaluate"]["checkpoint"]
            != EVALUATION_CHECKPOINT_SENTINEL
            or evaluate["evaluate"]["num_gpus"] != 8
            or evaluate["evaluate"]["gpu_ids"] != list(range(8))
            or evaluate["evaluate"]["num_nodes"] != 1
        ):
            raise CampaignExecutionError(
                f"full-run spec contract differs: {checkpoint_id}"
            )
        checkpoint_path = checkpoint["path"]
        if (
            row["execution_plan"]["mode"]
            == "reuse_sealed_v4_terminal_train"
            and (
                checkpoint_path
                != row["execution_plan"]["source_checkpoint"]["path"]
                or checkpoint["size_bytes"]
                != row["execution_plan"]["source_checkpoint"]["size_bytes"]
                or checkpoint["sha256"]
                != row["execution_plan"]["source_checkpoint"]["sha256"]
            )
        ):
            raise CampaignExecutionError(
                f"reused source checkpoint differs: {checkpoint_id}"
            )
        expected_train_ptm = (
            checkpoint_path
            if record["checkpoint_target"]
            == "train.pretrained_model_path"
            else ""
        )
        expected_backbone_ptm = (
            checkpoint_path
            if record["checkpoint_target"]
            == "model.backbone.pretrained_backbone_path"
            else ""
        )
        if (
            train["train"]["pretrained_model_path"]
            != expected_train_ptm
            or train["model"]["backbone"][
                "pretrained_backbone_path"
            ]
            != expected_backbone_ptm
            or evaluate["train"]["pretrained_model_path"]
            != expected_train_ptm
            or evaluate["model"]["backbone"][
                "pretrained_backbone_path"
            ]
            != expected_backbone_ptm
        ):
            raise CampaignExecutionError(
                f"checkpoint target differs: {checkpoint_id}"
            )
    value["stage_manifest_sha256"] = supplied
    return value


def verify_stage_remote(
    stage: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-hash every staged PTM and spec before scheduler submission."""
    validate_stage_manifest(stage, contract=contract)
    checked = []
    for row in stage["ptms"]:
        artifacts = {
            "checkpoint": row["checkpoint"],
            "train_spec": row["specs"]["train"]["lustre"],
            "evaluate_spec": row["specs"]["evaluate"]["lustre"],
        }
        if (
            row["execution_plan"]["mode"]
            == "reuse_sealed_v4_terminal_train"
        ):
            artifacts["reused_terminal_checkpoint"] = row[
                "execution_plan"
            ]["terminal_checkpoint"]
        for label, expected in artifacts.items():
            observed = _remote_identity(expected["path"])
            if (
                observed is None
                or observed["size_bytes"] != expected["size_bytes"]
                or observed["sha256"] != expected["sha256"]
                or (
                    label != "reused_terminal_checkpoint"
                    and int(observed["mode"], 8) & 0o222
                )
            ):
                raise CampaignExecutionError(
                    f"remote {label} changed for {row['checkpoint_id']}"
                )
            checked.append(
                {
                    "checkpoint_id": row["checkpoint_id"],
                    "artifact": label,
                    **observed,
                }
            )
    return {
        "schema_version": 2,
        "qualification_revision": campaign_contract.QUALIFICATION_REVISION,
        "stage_manifest_sha256": stage["stage_manifest_sha256"],
        "checked_artifacts": checked,
        "checked_artifact_count": len(checked),
        "all_staged_artifacts_read_only": True,
        "reused_terminal_checkpoint_count": sum(
            item["artifact"] == "reused_terminal_checkpoint"
            for item in checked
        ),
        "reused_terminal_checkpoints_hash_verified": True,
        "reused_terminal_checkpoints_rehashed_in_worker_before_evaluation": (
            True
        ),
    }


def _action(contract: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    metadata = yaml.safe_load(
        (
            Path(contract["runtime"]["skill_dir"])
            / "references/skill_info.yaml"
        ).read_text(encoding="utf-8")
    )
    action = metadata["actions"][name]
    if (
        action.get("mode") != "config"
        or action.get("config_format") != "yaml"
    ):
        raise CampaignExecutionError(
            f"SegFormer {name} action metadata changed"
        )
    return action


def _infrastructure_policy(
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    policy = contract.get("qualification_policy", {}).get(
        "infrastructure_retry_policy"
    )
    if policy != campaign_contract.FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY:
        raise CampaignExecutionError(
            "qualification infrastructure retry policy changed"
        )
    return policy


def _gpu_guard(command: str) -> str:
    """Require the exact frozen one-node/eight-A100 runtime."""
    policy = campaign_contract.FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY
    failure_marker = shlex.quote(policy["node_preflight_failure_marker"])
    failure_exit = policy["node_preflight_failure_exit_code"]
    cuda_runtime_probe = policy["cuda_runtime_probe"]
    success_marker = policy["node_preflight_success_marker"]
    return " ".join(
        [
            "set -eu;",
            "case \"$SLURM_JOB_ID\" in "
            "(''|*[!0-9]*) exit 91;; esac;",
            "export MASTER_ADDR=127.0.0.1;",
            "export MASTER_PORT=\"$(("
            f"{QUALIFICATION_MASTER_PORT_BASE} + SLURM_JOB_ID % "
            f"{QUALIFICATION_MASTER_PORT_SPAN}"
            "))\";",
            "python3 -c 'import os,socket; s=socket.socket(); "
            "s.bind((os.environ[\"MASTER_ADDR\"], "
            "int(os.environ[\"MASTER_PORT\"]))); s.close()';",
            "gpu_names=\"$(nvidia-smi --query-gpu=name "
            "--format=csv,noheader)\";",
            "gpu_caps=\"$(nvidia-smi --query-gpu=compute_cap "
            "--format=csv,noheader)\";",
            "gpu_mem=\"$(nvidia-smi --query-gpu=memory.total "
            "--format=csv,noheader,nounits)\";",
            "if ! python3 -c 'import torch; "
            "assert torch.cuda.is_available(); "
            "assert torch.cuda.device_count() == 8; "
            "probe=torch.empty(1, device=\"cuda\"); "
            "probe.add_(1); torch.cuda.synchronize()'; then "
            f"printf '%s\\n' {failure_marker}; exit {failure_exit}; fi;",
            f"printf '%s\\n' '{success_marker} "
            f"cuda_runtime_probe={cuda_runtime_probe}';",
            "test \"$(printf '%s\\n' \"$gpu_names\" | "
            "sed '/^$/d' | wc -l)\" -eq 8;",
            "test \"$(printf '%s\\n' \"$gpu_names\" | sort -u)\" = "
            "'NVIDIA A100-SXM4-80GB';",
            "test \"$(printf '%s\\n' \"$gpu_caps\" | sort -u)\" = '8.0';",
            "test \"$(printf '%s\\n' \"$gpu_mem\" | sort -u)\" = '81920';",
            command,
        ]
    )


def _runtime_overlay_install_command(
    contract: Mapping[str, Any],
    *,
    action_name: str,
) -> str:
    """Return the fail-closed v5 overlay pre-entrypoint."""
    overlay = contract["qualification_policy"].get("runtime_overlay")
    if (
        overlay
        != campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
        or action_name not in overlay["required_actions"]
    ):
        raise CampaignExecutionError(
            "qualification runtime overlay is not authorized for action"
        )
    archive = shlex.quote(overlay["archive_path"])
    installer = shlex.quote(overlay["installer_path"])
    receipt = shlex.quote(overlay["receipt_path"])
    archive_sha = shlex.quote(overlay["archive_sha256"])
    installer_sha = shlex.quote(overlay["installer_sha256"])
    return " && ".join(
        [
            f"test \"$(stat -c '%s' {archive})\" = "
            f"{overlay['archive_size_bytes']}",
            f"test \"$(sha256sum {archive} | cut -d ' ' -f1)\" = "
            f"{archive_sha}",
            f"test \"$(stat -c '%s' {installer})\" = "
            f"{overlay['installer_size_bytes']}",
            f"test \"$(sha256sum {installer} | cut -d ' ' -f1)\" = "
            f"{installer_sha}",
            f"python {installer} --archive {archive} "
            f"--expected-sha256 {archive_sha} --receipt {receipt}",
            f"test -s {receipt}",
        ]
    )


def _entrypoint(
    contract: Mapping[str, Any],
    action_name: str,
    specification: Mapping[str, Any],
) -> tuple[str, str]:
    from tao_sdk.script_runner import build_entrypoint

    action = _action(contract, action_name)
    _infrastructure_policy(contract)
    overlay = _runtime_overlay_install_command(
        contract,
        action_name=action_name,
    )
    entrypoint = build_entrypoint(
        command=_gpu_guard(f"{overlay} && {action['command']}"),
        specs=copy.deepcopy(dict(specification)),
        inputs=action["inputs"],
        outputs=action["outputs"],
        config_format=action["config_format"],
        upload_excludes=action.get("upload_excludes", []),
    )
    command = entrypoint["command"]
    return command, hashlib.sha256(command.encode("utf-8")).hexdigest()


def _submit_job(
    sdk: Any,
    contract: Mapping[str, Any],
    command: str,
    *,
    events: Path | None = None,
    checkpoint_id: str | None = None,
    phase: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Submit with one sealed retry for the exact stable-identity defect."""
    runtime = contract["runtime"]
    policy = _infrastructure_policy(contract)
    failures = []
    maximum_attempts = policy["maximum_submission_attempts_per_job"]
    for attempt in range(1, maximum_attempts + 1):
        try:
            job = sdk.create_job(
                image=contract["sqsh"]["path"],
                command=command,
                gpu_count=8,
                num_nodes=1,
                partition=runtime["partition"],
                account=runtime["account"],
            )
        except RuntimeError as exc:
            exact_retryable = (
                type(exc).__name__
                == policy["retryable_submission_exception_type"]
                and str(exc) == policy["retryable_submission_message"]
            )
            if not exact_retryable:
                raise
            failure = {
                "attempt": attempt,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "classification": (
                    "pre_submission_stable_identity_unavailable"
                ),
            }
            failures.append(failure)
            exhausted = attempt == maximum_attempts
            if (
                events is not None
                and checkpoint_id is not None
                and phase is not None
            ):
                append_jsonl(
                    events,
                    {
                        "event": (
                            "qualification_submission_retry_exhausted"
                            if exhausted
                            else "qualification_submission_retry"
                        ),
                        "phase": phase,
                        "checkpoint_id": checkpoint_id,
                        "submission_attempt": attempt,
                        "classification": failure["classification"],
                        "observed_at_utc": utc_timestamp(),
                    },
                )
            if exhausted:
                raise
            time.sleep(policy["retry_delay_seconds"])
            continue
        return job, {
            "attempt_count": attempt,
            "retry_count": len(failures),
            "transient_failures": failures,
            "stable_job_identity_obtained": True,
            "policy_sha256": canonical_sha256(policy),
        }
    raise AssertionError("unreachable submission retry state")


def _wait_for_job(
    sdk: Any,
    job_id: str,
    *,
    events: Path,
    checkpoint_id: str,
    phase: str,
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
                    "checkpoint_id": checkpoint_id,
                    "tao_job_id": job_id,
                    "status": status,
                    "observed_at_utc": utc_timestamp(),
                },
            )
            previous = status
        if status in TERMINAL_JOB_STATUSES:
            return status
        time.sleep(10)


def _terminal_infrastructure_retry_evidence(
    sdk: Any,
    contract: Mapping[str, Any],
    job_id: str,
    status: str,
) -> dict[str, Any]:
    """Classify only the controller-owned CUDA preflight failure marker."""
    policy = _infrastructure_policy(contract)
    if status != policy["retryable_terminal_status"]:
        return {
            "classification": "terminal_status_not_retryable",
            "retry_eligible": False,
            "terminal_status": status,
        }
    try:
        logs = sdk.get_job_logs(job_id, tail=500)
        failure_analysis = sdk.get_failure_analysis(job_id)
    except Exception as exc:
        return {
            "classification": "infrastructure_evidence_unavailable",
            "retry_eligible": False,
            "terminal_status": status,
            "evidence_error_type": type(exc).__name__,
        }
    if not isinstance(logs, str):
        logs = ""
    marker = policy["node_preflight_failure_marker"]
    marker_occurrences = sum(
        line.strip() == marker for line in logs.splitlines()
    )
    analysis = (
        copy.deepcopy(dict(failure_analysis))
        if isinstance(failure_analysis, Mapping)
        else None
    )
    analysis_matches = bool(
        analysis
        and analysis.get("reason") == "infrastructure_failure_pattern"
        and analysis.get("retriable") is True
        and analysis.get("match") == policy["sdk_failure_analysis_match"]
    )
    retry_eligible = marker_occurrences == 1 and analysis_matches
    return {
        "classification": (
            "pre_import_cuda_driver_runtime_incompatible"
            if retry_eligible
            else "terminal_error_not_exact_infrastructure_signature"
        ),
        "retry_eligible": retry_eligible,
        "terminal_status": status,
        "controller_marker_occurrences": marker_occurrences,
        "log_size_bytes": len(logs.encode("utf-8")),
        "log_sha256": hashlib.sha256(logs.encode("utf-8")).hexdigest(),
        "sdk_failure_analysis": analysis,
    }


def _run_qualification_job(
    sdk: Any,
    contract: Mapping[str, Any],
    command: str,
    *,
    evidence: dict[str, Any],
    evidence_path: Path,
    events: Path,
    checkpoint_id: str,
    phase: str,
    job_key: str,
    job_metadata: Mapping[str, Any],
) -> tuple[Any, str]:
    """Run one phase with one exact-signature infrastructure retry."""
    policy = _infrastructure_policy(contract)
    policy_sha256 = canonical_sha256(policy)
    maximum_attempts = policy["maximum_job_attempts_per_phase"]
    attempts: list[dict[str, Any]] = []
    base = copy.deepcopy(dict(job_metadata))
    evidence["jobs"][job_key] = {
        **base,
        "status": "submitting",
        "attempts": [],
        "infrastructure_retry_policy_sha256": policy_sha256,
        "maximum_job_attempts": maximum_attempts,
        "successful_job_replacement_allowed": False,
    }
    atomic_json(evidence_path, evidence)

    for job_attempt in range(1, maximum_attempts + 1):
        job, submission = _submit_job(
            sdk,
            contract,
            command,
            events=events,
            checkpoint_id=checkpoint_id,
            phase=phase,
        )
        attempt = {
            "job_attempt": job_attempt,
            "tao_job_id": job.id,
            "status": "submitted",
            "submitted_at_utc": utc_timestamp(),
            "submission": submission,
            "command_sha256": base["command_sha256"],
        }
        attempts.append(attempt)
        evidence["jobs"][job_key] = {
            **base,
            "tao_job_id": job.id,
            "status": "submitted",
            "submitted_at_utc": attempt["submitted_at_utc"],
            "job_attempt": job_attempt,
            "attempts": copy.deepcopy(attempts),
            "infrastructure_retry_policy_sha256": policy_sha256,
            "maximum_job_attempts": maximum_attempts,
            "successful_job_replacement_allowed": False,
        }
        atomic_json(evidence_path, evidence)

        status = _wait_for_job(
            sdk,
            job.id,
            events=events,
            checkpoint_id=checkpoint_id,
            phase=phase,
        )
        attempt.update(
            {
                "status": status,
                "terminal_at_utc": utc_timestamp(),
                "result_root": run_campaign._local_lustre_path(
                    sdk.get_job_results_dir(job.id)
                ),
            }
        )
        infrastructure = _terminal_infrastructure_retry_evidence(
            sdk,
            contract,
            job.id,
            status,
        )
        attempt["infrastructure_failure_evidence"] = infrastructure
        retry_submitted = bool(
            infrastructure["retry_eligible"]
            and job_attempt < maximum_attempts
        )
        attempt["infrastructure_retry_submitted"] = retry_submitted
        evidence["jobs"][job_key] = {
            **base,
            "tao_job_id": job.id,
            "status": status,
            "submitted_at_utc": attempt["submitted_at_utc"],
            "terminal_at_utc": attempt["terminal_at_utc"],
            "result_root": attempt["result_root"],
            "job_attempt": job_attempt,
            "attempts": copy.deepcopy(attempts),
            "infrastructure_retry_count": sum(
                item["infrastructure_retry_submitted"] for item in attempts
            ),
            "infrastructure_retry_policy_sha256": policy_sha256,
            "maximum_job_attempts": maximum_attempts,
            "successful_job_replacement_allowed": False,
        }
        atomic_json(evidence_path, evidence)
        if status == "Complete":
            return job, status
        if not retry_submitted:
            return job, status
        append_jsonl(
            events,
            {
                "event": "qualification_infrastructure_phase_retry",
                "phase": phase,
                "checkpoint_id": checkpoint_id,
                "prior_tao_job_id": job.id,
                "prior_job_attempt": job_attempt,
                "next_job_attempt": job_attempt + 1,
                "classification": infrastructure["classification"],
                "observed_at_utc": utc_timestamp(),
            },
        )
        time.sleep(policy["retry_delay_seconds"])
    raise AssertionError("unreachable qualification job retry state")


def _status_records(
    sdk: Any,
    job_id: str,
    *,
    action: str,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    root = run_campaign._local_lustre_path(
        sdk.get_job_results_dir(job_id)
    )
    path = f"{root.rstrip('/')}/results_dir/{action}/status.json"
    text = remote_output(
        f"test -f {shlex.quote(path)} && cat {shlex.quote(path)}"
    )
    identity = _remote_identity(path)
    if identity is None:
        raise CampaignExecutionError(
            f"{action} status evidence is unavailable"
        )
    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CampaignExecutionError(
                f"{action} status line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(record, Mapping):
            raise CampaignExecutionError(
                f"{action} status line {line_number} is not an object"
            )
        records.append(record)
    if not records:
        raise CampaignExecutionError(f"{action} status file is empty")
    return records, {**identity, "record_count": len(records)}


def _metric(
    value: Any,
    *,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise CampaignExecutionError(f"{name} is not a finite metric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CampaignExecutionError(
            f"{name} is not a finite metric"
        ) from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise CampaignExecutionError(f"{name} is outside [0, 1]")
    return number


def _training_status_evidence(
    sdk: Any,
    job_id: str,
    *,
    expected_checkpoint_path: str,
    expected_component: str,
) -> dict[str, Any]:
    records, identity = _status_records(sdk, job_id, action="train")
    validation = []
    load_reports = []
    for record in records:
        message = record.get("message")
        if isinstance(message, str) and message.startswith(
            PRETRAINED_LOAD_REPORT_PREFIX
        ):
            payload_text = message[len(PRETRAINED_LOAD_REPORT_PREFIX):]
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError as exc:
                raise CampaignExecutionError(
                    "SegFormer pretrained-load report is invalid JSON"
                ) from exc
            if not isinstance(payload, Mapping):
                raise CampaignExecutionError(
                    "SegFormer pretrained-load report is not an object"
                )
            load_reports.append(copy.deepcopy(dict(payload)))
        kpi = record.get("kpi")
        # TAO writes the same validation snapshot twice per epoch: first when
        # evaluation generates it, then again with the training-loop progress
        # record. Only the evaluation record is an independent observation.
        if (
            record.get("message") != "Eval metrics generated."
            or not isinstance(kpi, Mapping)
            or "val_miou" not in kpi
        ):
            continue
        validation.append(
            {
                "record_index": len(validation),
                "val_miou": _metric(
                    kpi["val_miou"],
                    name="training val_miou",
                ),
            }
        )
    expected_epochs = (
        campaign_contract.FROZEN_QUALIFICATION_TRAINING_EPOCHS
    )
    if len(validation) != expected_epochs:
        raise CampaignExecutionError(
            f"training emitted {len(validation)} val_miou records; "
            f"expected {expected_epochs}"
        )
    if not any(
        record.get("message") == "Train finished successfully."
        for record in records
    ):
        raise CampaignExecutionError(
            "training status lacks the terminal TAO success record"
        )
    unique_reports = {
        canonical_sha256(report): report for report in load_reports
    }
    if len(unique_reports) != 1:
        raise CampaignExecutionError(
            "training must emit exactly one unique positive SegFormer "
            f"pretrained-load report; observed {len(unique_reports)}"
        )
    pretrained_load = next(iter(unique_reports.values()))
    if (
        set(pretrained_load)
        != {
            "schema_version",
            "checkpoint",
            "component",
            "loaded_keyset_sha256",
            *_PRETRAINED_LOAD_COUNT_FIELDS,
        }
        or pretrained_load.get("schema_version") != 1
        or pretrained_load.get("checkpoint") != expected_checkpoint_path
        or pretrained_load.get("component") != expected_component
        or _SHA256_RE.fullmatch(
            str(pretrained_load.get("loaded_keyset_sha256", ""))
        )
        is None
        or any(
            isinstance(pretrained_load.get(name), bool)
            or not isinstance(pretrained_load.get(name), int)
            or pretrained_load[name] < 0
            for name in _PRETRAINED_LOAD_COUNT_FIELDS
        )
        or pretrained_load["loaded_tensor_count"] < 1
    ):
        raise CampaignExecutionError(
            "SegFormer pretrained-load report does not prove a positive "
            "load from the exact staged checkpoint and component"
        )
    pretrained_load["status_record_occurrences"] = len(load_reports)
    pretrained_load["report_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in pretrained_load.items()
            if key not in {"status_record_occurrences", "report_sha256"}
        }
    )
    return {
        **identity,
        "validation_record_count": len(validation),
        "validation_metrics": validation,
        "val_miou": validation[-1]["val_miou"],
        "terminal_success": True,
        "terminal_success_message": "Train finished successfully.",
        "pretrained_load": pretrained_load,
    }


def _qualification_terminal_checkpoint(
    sdk: Any,
    job_id: str,
) -> dict[str, Any]:
    """Resolve the exact terminal checkpoint for the 50-epoch run."""
    epochs = campaign_contract.FROZEN_QUALIFICATION_TRAINING_EPOCHS
    terminal_epoch = epochs - 1
    epoch_token = f"{terminal_epoch:03d}"
    root = run_campaign._local_lustre_path(
        sdk.get_job_results_dir(job_id)
    )
    folder = f"{root.rstrip('/')}/results_dir/train"
    script = (
        "import glob,hashlib,json,pathlib,sys;"
        "paths=sorted(glob.glob(sys.argv[1]+'/'+sys.argv[2]));"
        "assert len(paths)==1,paths;"
        "p=pathlib.Path(paths[0]);h=hashlib.sha256();f=p.open('rb');"
        "[(h.update(c)) for c in iter(lambda:f.read(1048576),b'')];"
        "f.close();print(json.dumps({'path':str(p),'filename':p.name,"
        "'size_bytes':p.stat().st_size,'sha256':h.hexdigest()}))"
    )
    pattern = f"model_epoch_{epoch_token}_step_*.pth"
    try:
        evidence = json.loads(
            remote_output(
                f"python3 -c {shlex.quote(script)} "
                f"{shlex.quote(folder)} {shlex.quote(pattern)}"
            )
        )
    except Exception as exc:
        raise CampaignExecutionError(
            "exact terminal qualification checkpoint is unavailable "
            "or ambiguous"
        ) from exc
    if (
        not re.fullmatch(
            rf"model_epoch_{epoch_token}_step_[0-9]+[.]pth",
            str(evidence.get("filename", "")),
        )
        or not str(evidence.get("path", "")).startswith("/lustre/")
        or not isinstance(evidence.get("size_bytes"), int)
        or evidence["size_bytes"] < 1
        or not _SHA256_RE.fullmatch(str(evidence.get("sha256", "")))
    ):
        raise CampaignExecutionError(
            "terminal qualification checkpoint identity is invalid"
        )
    evidence.update(
        {
            "training_epochs": epochs,
            "terminal_epoch_index": terminal_epoch,
            "naming_contract": (
                f"model_epoch_{epoch_token}_step_numeric"
            ),
            "ambiguity_policy": "fail_closed",
        }
    )
    return evidence


def _evaluation_status_evidence(
    sdk: Any,
    job_id: str,
) -> dict[str, Any]:
    records, identity = _status_records(
        sdk,
        job_id,
        action="evaluate",
    )
    metric_occurrences = []
    for record in records:
        kpi = record.get("kpi")
        if not isinstance(kpi, Mapping):
            continue
        names = [
            name
            for name in ("test_miou", "val_miou", "mIoU")
            if name in kpi
        ]
        if not names:
            continue
        if len(names) != 1:
            raise CampaignExecutionError(
                "standalone evaluation KPI snapshot contains multiple "
                "accepted mIoU metric names"
            )
        name = names[0]
        snapshot = {
            "reported_name": name,
            "test_miou": _metric(
                kpi[name],
                name="standalone test_miou",
            ),
            "kpi": copy.deepcopy(dict(kpi)),
        }
        metric_occurrences.append(
            {
                **snapshot,
                "snapshot_sha256": canonical_sha256(snapshot),
            }
        )
    unique_metrics = {
        item["snapshot_sha256"]: item for item in metric_occurrences
    }
    if len(unique_metrics) != 1:
        raise CampaignExecutionError(
            "standalone evaluation emitted "
            f"{len(metric_occurrences)} mIoU records representing "
            f"{len(unique_metrics)} unique semantic KPI snapshots; "
            "expected exactly one unique snapshot"
        )
    if not any(
        record.get("message") == "Evaluate finished successfully."
        for record in records
    ):
        raise CampaignExecutionError(
            "evaluation status lacks the terminal TAO success record"
        )
    metric = next(iter(unique_metrics.values()))
    return {
        **identity,
        "test_metric_record_count": len(metric_occurrences),
        "unique_test_metric_snapshot_count": 1,
        "duplicate_identical_metric_snapshots_allowed": True,
        "metric_snapshot_sha256": metric["snapshot_sha256"],
        "reported_metric_name": metric["reported_name"],
        "test_miou": metric["test_miou"],
        "terminal_success": True,
        "terminal_success_message": "Evaluate finished successfully.",
    }


def _stage_by_id(
    stage: Mapping[str, Any],
    checkpoint_id: str,
) -> Mapping[str, Any]:
    matches = [
        item
        for item in stage["ptms"]
        if item["checkpoint_id"] == checkpoint_id
    ]
    if len(matches) != 1:
        raise CampaignExecutionError(
            f"no unique staged PTM row for {checkpoint_id}"
        )
    return matches[0]


def _finalize_workflow(record: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(record))
    value.pop("workflow_sha256", None)
    value["workflow_sha256"] = canonical_sha256(value)
    return value


def _reused_train_phase_evidence(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Re-verify and materialize one exact sealed v4 train phase."""
    plan = row.get("execution_plan")
    policy = campaign_contract.FROZEN_QUALIFICATION_PHASE_RECOVERY_POLICY
    checkpoint_id = row["checkpoint_id"]
    if (
        not isinstance(plan, Mapping)
        or plan.get("mode") != "reuse_sealed_v4_terminal_train"
        or canonical_sha256(plan)
        != policy["execution_plan_sha256_by_checkpoint_id"][checkpoint_id]
    ):
        raise CampaignExecutionError(
            f"sealed v4 train reuse plan changed: {checkpoint_id}"
        )
    for label, expected, require_read_only in (
        ("source checkpoint", row["checkpoint"], True),
        ("terminal checkpoint", plan["terminal_checkpoint"], False),
        (
            "training status evidence",
            plan["validation_status_evidence"],
            False,
        ),
    ):
        observed = _remote_identity(expected["path"])
        if (
            observed is None
            or observed["size_bytes"] != expected["size_bytes"]
            or observed["sha256"] != expected["sha256"]
            or (require_read_only and int(observed["mode"], 8) & 0o222)
        ):
            raise CampaignExecutionError(
                f"reused v4 {label} changed: {checkpoint_id}"
            )
    train_evidence = copy.deepcopy(
        dict(plan["validation_status_evidence"])
    )
    train_evidence["pretrained_load"] = copy.deepcopy(
        plan["pretrained_load"]
    )
    checkpoint = copy.deepcopy(dict(plan["terminal_checkpoint"]))
    predecessor_job = copy.deepcopy(dict(plan["train_job"]))
    job = {
        "execution_mode": "reuse_sealed_v4_terminal_train",
        "new_job_submitted": False,
        "successful_train_reexecution": False,
        "runtime_overlay_required": True,
        "runtime_overlay": copy.deepcopy(
            plan["predecessor_runtime_overlay"]
        ),
        "predecessor_campaign_id": policy["predecessor_campaign_id"],
        "predecessor_completion_whole_file_sha256": policy[
            "predecessor_completion_whole_file_sha256"
        ],
        "predecessor_load_audit_whole_file_sha256": policy[
            "predecessor_load_audit_whole_file_sha256"
        ],
        "v4_workflow_sha256": plan["v4_workflow_sha256"],
        "v4_load_audit_row_sha256": plan[
            "v4_load_audit_row_sha256"
        ],
        "tao_job_id": predecessor_job["tao_job_id"],
        "tao_job_id_origin": "sealed_predecessor_v4",
        "status": "Complete",
        "result_root": predecessor_job["result_root"],
        "command_sha256": predecessor_job["command_sha256"],
        "spec_sha256": predecessor_job["spec_sha256"],
        "predecessor_train_job": predecessor_job,
        "status_evidence": copy.deepcopy(train_evidence),
        "terminal_checkpoint": copy.deepcopy(checkpoint),
    }
    return train_evidence, checkpoint, job


def _run_workflow(
    contract_path: str,
    stage_path: str,
    runtime_root: str,
    checkpoint_id: str,
) -> None:
    contract = run_campaign.load_contract(contract_path)
    stage = validate_stage_manifest(
        json.loads(Path(stage_path).read_text(encoding="utf-8")),
        contract=contract,
    )
    row = _stage_by_id(stage, checkpoint_id)
    workflow_root = (
        Path(runtime_root) / "workflows" / row["workflow_id"]
    )
    workflow_root.mkdir(parents=True, exist_ok=True)
    evidence_path = workflow_root / "workflow_completion.json"
    events = workflow_root / "events.jsonl"
    evidence: dict[str, Any] = {
        "schema_version": 2,
        "qualification_revision": campaign_contract.QUALIFICATION_REVISION,
        "checkpoint_id": checkpoint_id,
        "status": "running",
        "terminal": False,
        "failure_preserved": False,
        "source_checkpoint": copy.deepcopy(row["checkpoint"]),
        "stage_manifest_sha256": stage["stage_manifest_sha256"],
        "recipe_fidelity": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_FIDELITY
        ),
        "runtime_overlay": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
        ),
        "infrastructure_retry_policy": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY
        ),
        "phase_recovery_policy": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_PHASE_RECOVERY_POLICY
        ),
        "execution_plan": copy.deepcopy(row["execution_plan"]),
        "jobs": {},
        "agent_intervention_flags": {
            name: False for name in campaign_contract.AGENT_FLAGS
        },
    }
    atomic_json(evidence_path, evidence)
    phase = "controller_initialization"
    run_campaign.configure_slurm_runtime(contract)
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
            state_file=workflow_root / "slurm_state.json",
        )

        execution_mode = row["execution_plan"]["mode"]
        if execution_mode == "reuse_sealed_v4_terminal_train":
            phase = "sealed_v4_train_reuse_validation"
            train_evidence, checkpoint, reused_job = (
                _reused_train_phase_evidence(row)
            )
            evidence["jobs"]["train"] = reused_job
            append_jsonl(
                events,
                {
                    "event": "sealed_v4_terminal_train_reused",
                    "checkpoint_id": checkpoint_id,
                    "predecessor_tao_job_id": reused_job["tao_job_id"],
                    "terminal_checkpoint": copy.deepcopy(checkpoint),
                    "observed_at_utc": utc_timestamp(),
                },
            )
        elif execution_mode == "run_fresh_full_train":
            phase = "train"
            train_spec = copy.deepcopy(row["specs"]["train"]["document"])
            if (
                canonical_sha256(train_spec)
                != row["specs"]["train"]["document_sha256"]
            ):
                raise CampaignExecutionError("staged train spec changed")
            train_command, train_command_sha = _entrypoint(
                contract,
                "train",
                train_spec,
            )
            train_job, train_status = _run_qualification_job(
                sdk,
                contract,
                train_command,
                evidence=evidence,
                evidence_path=evidence_path,
                events=events,
                checkpoint_id=checkpoint_id,
                phase=phase,
                job_key="train",
                job_metadata={
                    "execution_mode": execution_mode,
                    "new_job_submitted": True,
                    "spec_sha256": canonical_sha256(train_spec),
                    "staged_spec_sha256": row["specs"]["train"][
                        "raw_yaml_sha256"
                    ],
                    "command_sha256": train_command_sha,
                    "runtime_overlay_required": True,
                    "nodes": 1,
                    "gpus": 8,
                },
            )
            if train_status != "Complete":
                evidence["jobs"]["train"]["failure_analysis"] = (
                    sdk.get_failure_analysis(train_job.id)
                )
                raise CampaignExecutionError(
                    f"training job ended with {train_status}"
                )
            expected_load_component = (
                "model"
                if row["checkpoint_target"]
                == "train.pretrained_model_path"
                else "backbone"
            )
            train_evidence = _training_status_evidence(
                sdk,
                train_job.id,
                expected_checkpoint_path=row["checkpoint"]["path"],
                expected_component=expected_load_component,
            )
            checkpoint = _qualification_terminal_checkpoint(
                sdk,
                train_job.id,
            )
            evidence["jobs"]["train"].update(
                {
                    "status_evidence": train_evidence,
                    "terminal_checkpoint": checkpoint,
                }
            )
        else:
            raise CampaignExecutionError(
                f"unsupported phase-recovery mode: {execution_mode}"
            )
        atomic_json(evidence_path, evidence)

        phase = "standalone_evaluation"
        evaluation_spec = copy.deepcopy(
            row["specs"]["evaluate"]["document"]
        )
        if (
            canonical_sha256(evaluation_spec)
            != row["specs"]["evaluate"]["document_sha256"]
            or evaluation_spec["evaluate"]["checkpoint"]
            != EVALUATION_CHECKPOINT_SENTINEL
        ):
            raise CampaignExecutionError(
                "staged evaluation template changed"
            )
        evaluation_spec["evaluate"]["checkpoint"] = checkpoint["path"]
        evaluation_command, evaluation_command_sha = _entrypoint(
            contract,
            "evaluate",
            evaluation_spec,
        )
        evaluation_job, evaluation_status = _run_qualification_job(
            sdk,
            contract,
            evaluation_command,
            evidence=evidence,
            evidence_path=evidence_path,
            events=events,
            checkpoint_id=checkpoint_id,
            phase=phase,
            job_key="evaluate",
            job_metadata={
                "template_sha256": row["specs"]["evaluate"][
                    "document_sha256"
                ],
                "resolved_spec_sha256": canonical_sha256(evaluation_spec),
                "command_sha256": evaluation_command_sha,
                "runtime_overlay_required": True,
                "checkpoint": checkpoint,
                "execution_mode": "new_standalone_evaluation",
                "new_job_submitted": True,
                "nodes": 1,
                "gpus": 8,
            },
        )
        if evaluation_status != "Complete":
            evidence["jobs"]["evaluate"]["failure_analysis"] = (
                sdk.get_failure_analysis(evaluation_job.id)
            )
            raise CampaignExecutionError(
                f"evaluation job ended with {evaluation_status}"
            )
        evaluation_evidence = _evaluation_status_evidence(
            sdk,
            evaluation_job.id,
        )
        evidence["jobs"]["evaluate"][
            "status_evidence"
        ] = evaluation_evidence

        evidence.update(
            {
                "status": "success",
                "terminal": True,
                "failure_preserved": False,
                "terminal_at_utc": utc_timestamp(),
                "train": {
                    "status": "Complete",
                    "execution_mode": execution_mode,
                    "source_qualification_revision": (
                        row["execution_plan"].get(
                            "predecessor_qualification_revision"
                        )
                        if execution_mode
                        == "reuse_sealed_v4_terminal_train"
                        else campaign_contract.QUALIFICATION_REVISION
                    ),
                    "full_dataset": True,
                    "training_epochs": (
                        campaign_contract.FROZEN_QUALIFICATION_TRAINING_EPOCHS
                    ),
                    "validation_interval": 1,
                    "recipe_fidelity": copy.deepcopy(
                        row["execution_plan"].get(
                            "predecessor_recipe_fidelity",
                            campaign_contract.FROZEN_QUALIFICATION_FIDELITY,
                        )
                    ),
                    "runtime_overlay": copy.deepcopy(
                        row["execution_plan"].get(
                            "predecessor_runtime_overlay",
                            campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY,
                        )
                    ),
                    "validation_record_count": train_evidence[
                        "validation_record_count"
                    ],
                    "nodes": 1,
                    "gpus": 8,
                    "val_miou": train_evidence["val_miou"],
                    "terminal_checkpoint": checkpoint,
                    "status_evidence": train_evidence,
                    "job": copy.deepcopy(evidence["jobs"]["train"]),
                },
                "evaluation": {
                    "status": "Complete",
                    "full_validation_split": True,
                    "runtime_overlay": copy.deepcopy(
                        campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
                    ),
                    "nodes": 1,
                    "gpus": 8,
                    "test_miou": evaluation_evidence["test_miou"],
                    "status_evidence": evaluation_evidence,
                    "job": copy.deepcopy(evidence["jobs"]["evaluate"]),
                },
            }
        )
        atomic_json(evidence_path, _finalize_workflow(evidence))
    except BaseException as exc:
        failure = {
            "schema_version": 2,
            "qualification_revision": (
                campaign_contract.QUALIFICATION_REVISION
            ),
            "checkpoint_id": checkpoint_id,
            "status": "failure",
            "terminal": True,
            "failure_preserved": True,
            "failure_code": "direct_full_run_failed",
            "failure_reason": f"{phase}: {type(exc).__name__}: {exc}",
            "failure_phase": phase,
            "replacement_submitted": False,
            "source_checkpoint": copy.deepcopy(row["checkpoint"]),
            "stage_manifest_sha256": stage["stage_manifest_sha256"],
            "recipe_fidelity": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_FIDELITY
            ),
            "runtime_overlay": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
            ),
            "infrastructure_retry_policy": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY
            ),
            "phase_recovery_policy": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_PHASE_RECOVERY_POLICY
            ),
            "execution_plan": copy.deepcopy(row["execution_plan"]),
            "jobs": copy.deepcopy(evidence.get("jobs", {})),
            "terminal_at_utc": utc_timestamp(),
            "agent_intervention_flags": {
                name: False for name in campaign_contract.AGENT_FLAGS
            },
        }
        atomic_json(evidence_path, _finalize_workflow(failure))
        raise


def _missing_workflow(
    checkpoint_id: str,
    stage_sha256: str,
    execution_plan: Mapping[str, Any],
) -> dict[str, Any]:
    return _finalize_workflow(
        {
            "schema_version": 2,
            "qualification_revision": (
                campaign_contract.QUALIFICATION_REVISION
            ),
            "checkpoint_id": checkpoint_id,
            "status": "failure",
            "terminal": True,
            "failure_preserved": True,
            "failure_code": "workflow_artifact_missing",
            "failure_reason": (
                "worker exited without a terminal workflow record"
            ),
            "failure_phase": "controller",
            "replacement_submitted": False,
            "stage_manifest_sha256": stage_sha256,
            "recipe_fidelity": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_FIDELITY
            ),
            "runtime_overlay": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
            ),
            "infrastructure_retry_policy": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY
            ),
            "phase_recovery_policy": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_PHASE_RECOVERY_POLICY
            ),
            "execution_plan": copy.deepcopy(dict(execution_plan)),
            "terminal_at_utc": utc_timestamp(),
            "agent_intervention_flags": {
                name: False for name in campaign_contract.AGENT_FLAGS
            },
        }
    )


def build_completion(
    *,
    contract: Mapping[str, Any],
    stage: Mapping[str, Any],
    runtime_root: Path,
    exit_codes: Mapping[str, int | None],
) -> dict[str, Any]:
    """Build the exact immutable document consumed by qualification_gate."""
    workflows = []
    for row in stage["ptms"]:
        checkpoint_id = row["checkpoint_id"]
        path = (
            runtime_root
            / "workflows"
            / row["workflow_id"]
            / "workflow_completion.json"
        )
        workflow = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file()
            else _missing_workflow(
                checkpoint_id,
                stage["stage_manifest_sha256"],
                row["execution_plan"],
            )
        )
        workflow["worker_process_exit_code"] = exit_codes.get(
            checkpoint_id
        )
        workflows.append(_finalize_workflow(workflow))
    successful = sum(
        workflow["status"] == "success" for workflow in workflows
    )
    value = {
        "schema_version": 2,
        "qualification_revision": campaign_contract.QUALIFICATION_REVISION,
        "campaign_id": QUALIFICATION_CAMPAIGN_ID,
        "automl_contract_sha256": contract["contract_sha256"],
        "model": "segformer",
        "task": "semantic_segmentation",
        "registry_sha256": contract["ptm_inventory"]["registry_sha256"],
        "sqsh_sha256": contract["sqsh"]["sha256"],
        "ptm_stage_manifest_path": contract["qualification_policy"][
            "ptm_stage_manifest_path"
        ],
        "ptm_stage_manifest_sha256": stage["stage_manifest_sha256"],
        "source_commit": contract["runtime"]["source_commit"],
        "recipe_fidelity": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_FIDELITY
        ),
        "runtime_overlay": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
        ),
        "infrastructure_retry_policy": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY
        ),
        "phase_recovery_policy": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_PHASE_RECOVERY_POLICY
        ),
        "prior_revision_evidence": copy.deepcopy(
            campaign_contract.FROZEN_PRIOR_QUALIFICATION_EVIDENCE
        ),
        "qualification_controller_sha256": contract[
            "launcher_integrity"
        ]["qualification_campaign_sha256"],
        "status": (
            "success"
            if successful == len(workflows)
            else "terminal_with_failures"
        ),
        "terminal": True,
        "successful_workflows": successful,
        "failed_workflows": len(workflows) - successful,
        "all_official_arms_attempted": True,
        "failure_records_preserved": True,
        "replacement_workflows_submitted": False,
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "workflows": workflows,
        "agent_intervention_flags": {
            name: False for name in campaign_contract.AGENT_FLAGS
        },
        "completed_at_utc": utc_timestamp(),
    }
    value["evidence_sha256"] = canonical_sha256(value)
    return value


def build_handoff(
    *,
    contract: Mapping[str, Any],
    completion: Mapping[str, Any],
    qualification_path: Path,
) -> dict[str, Any]:
    """Create the automatic, non-promoting qualification handoff."""
    decision = audit_qualification(
        qualification_path,
        expected_contract=contract,
    )
    successful_ids = [
        item["checkpoint_id"]
        for item in completion["workflows"]
        if item["status"] == "success"
    ]
    failed_ids = [
        item["checkpoint_id"]
        for item in completion["workflows"]
        if item["status"] == "failure"
    ]
    value = {
        "schema_version": 2,
        "qualification_revision": campaign_contract.QUALIFICATION_REVISION,
        "campaign_id": QUALIFICATION_CAMPAIGN_ID,
        "automl_contract_sha256": contract["contract_sha256"],
        "qualification_evidence_path": str(qualification_path),
        "qualification_evidence_sha256": completion["evidence_sha256"],
        "automatic": True,
        "manual_confirmation_required": False,
        "registry_mutated": False,
        "registry_bypass_allowed": False,
        "successful_checkpoint_ids": successful_ids,
        "terminal_failure_checkpoint_ids": failed_ids,
        "runtime_ready_under_current_registry": decision.runtime_ready,
        "runtime_eligibility": copy.deepcopy(
            dict(decision.runtime_eligibility)
        ),
        "status": (
            "ready_for_three_mode_automatic_trigger"
            if decision.runtime_ready
            else (
                "ready_for_evidence_bound_successor_seal"
                if successful_ids
                else "terminal_no_successful_ptm"
            )
        ),
        "next_command_after_evidence_bound_successor_seal": (
            "python -m experiments.cross_model_automl_20260729."
            "segformer_voc2012_campaign.run_campaign "
            "--automatic-trigger --launch"
        ),
        "automatic_selector_or_recommendation_invoked": False,
        "fallback_ptm_selected": False,
        "failed_workflow_replaced": False,
        "agent_intervention_flags": {
            name: False for name in campaign_contract.AGENT_FLAGS
        },
    }
    value["handoff_sha256"] = canonical_sha256(value)
    return value


def qualification_plan(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "qualification_revision": campaign_contract.QUALIFICATION_REVISION,
        "campaign_id": QUALIFICATION_CAMPAIGN_ID,
        "automl_contract_sha256": contract["contract_sha256"],
        "model": "segformer",
        "dataset": contract["dataset"]["prepared_root"],
        "checkpoint_ids": [record["id"] for record in _records()],
        "workflow_count": 13,
        "workflow": (
            "selective_v4_terminal_train_reuse_or_fresh_full_voc2012_"
            "50_epoch_train_then_new_standalone_full_validation"
        ),
        "new_full_train_job_count": 9,
        "reused_terminal_train_phase_count": 4,
        "new_standalone_evaluation_job_count": 13,
        "recipe_fidelity": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_FIDELITY
        ),
        "runtime_overlay": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
        ),
        "phase_recovery_policy": copy.deepcopy(
            campaign_contract.FROZEN_QUALIFICATION_PHASE_RECOVERY_POLICY
        ),
        "prior_revision_evidence": copy.deepcopy(
            campaign_contract.FROZEN_PRIOR_QUALIFICATION_EVIDENCE
        ),
        "resources_per_job": {
            "nodes": 1,
            "gpus": 8,
            "gpu": campaign_contract.FROZEN_HARDWARE["gpu_name"],
            "partition": contract["runtime"]["partition"],
            "time_hours": contract["runtime"]["time_hours"],
            "container": contract["sqsh"]["path"],
        },
        "all_workflows_independent": True,
        "all_workflows_submitted_without_result_driven_exclusion": True,
        "terminal_failures_preserved": True,
        "replacement_workflows_submitted": False,
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "launch": False,
    }


def _claim_qualification_launch(
    runtime_root: Path,
    *,
    contract_sha256: str,
    stage_manifest_sha256: str,
) -> dict[str, Any]:
    """Atomically prevent controller re-entry and train-phase repetition."""
    workflows = runtime_root / "workflows"
    if workflows.exists() and any(workflows.iterdir()):
        raise CampaignExecutionError(
            "qualification workflow state already exists; launch re-entry "
            "is forbidden"
        )
    marker = runtime_root / "qualification_launch_started.json"
    payload = {
        "schema_version": 1,
        "qualification_revision": campaign_contract.QUALIFICATION_REVISION,
        "campaign_id": QUALIFICATION_CAMPAIGN_ID,
        "contract_sha256": contract_sha256,
        "stage_manifest_sha256": stage_manifest_sha256,
        "successful_train_reexecution_allowed": False,
        "claimed_at_utc": utc_timestamp(),
    }
    payload["claim_sha256"] = canonical_sha256(payload)
    data = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(
            marker,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o400,
        )
    except FileExistsError as exc:
        raise CampaignExecutionError(
            "qualification launch was already claimed; re-entry is forbidden"
        ) from exc
    try:
        if os.write(descriptor, data) != len(data):
            raise CampaignExecutionError(
                "qualification launch claim write was incomplete"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"path": str(marker), **payload}


def launch(
    *,
    contract_path: Path,
    stage_path: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    """Run the sealed 4-reuse/9-train/13-evaluate qualification plan."""
    contract = run_campaign.load_contract(contract_path)
    qualification_path = Path(
        contract["qualification_policy"]["qualification_evidence_path"]
    ).resolve()
    if runtime_root.resolve() != qualification_path.parent:
        raise CampaignExecutionError(
            "qualification runtime root differs from the sealed contract"
        )
    expected_stage = Path(
        contract["qualification_policy"]["ptm_stage_manifest_path"]
    ).resolve()
    if stage_path.resolve() != expected_stage:
        raise CampaignExecutionError(
            "qualification stage path differs from the sealed contract"
        )
    stage = validate_stage_manifest(
        json.loads(stage_path.read_text(encoding="utf-8")),
        contract=contract,
    )
    local = run_campaign.verify_local_contract(contract)
    platform = verify_slurm_preflight(contract)
    dataset = run_campaign._verify_dataset_remote(contract)
    remote_stage = verify_stage_remote(stage, contract=contract)
    sqsh = run_campaign._remote_file_identity(contract["sqsh"]["path"])
    if sqsh["sha256"] != contract["sqsh"]["sha256"]:
        raise CampaignExecutionError("pinned SQSH identity changed")
    runtime_root.mkdir(parents=True, exist_ok=True)
    launch_claim = _claim_qualification_launch(
        runtime_root,
        contract_sha256=contract["contract_sha256"],
        stage_manifest_sha256=stage["stage_manifest_sha256"],
    )
    atomic_json(
        runtime_root / "qualification_launch_preflight.json",
        {
            "schema_version": 2,
            "qualification_revision": (
                campaign_contract.QUALIFICATION_REVISION
            ),
            "contract_sha256": contract["contract_sha256"],
            "local": local,
            "platform": platform,
            "dataset": dataset,
            "remote_stage": remote_stage,
            "sqsh": sqsh,
            "launch_claim": launch_claim,
            "phase_recovery_policy": copy.deepcopy(
                campaign_contract.FROZEN_QUALIFICATION_PHASE_RECOVERY_POLICY
            ),
            "sdk_constructed": False,
            "scheduler_jobs_submitted": 0,
            "cpu_or_smoke_model_jobs_launched": False,
        },
    )

    context = mp.get_context("spawn")
    processes = {
        row["checkpoint_id"]: context.Process(
            target=_run_workflow,
            args=(
                str(contract_path),
                str(stage_path),
                str(runtime_root),
                row["checkpoint_id"],
            ),
            name=f"segformer-{row['workflow_id']}",
        )
        for row in stage["ptms"]
    }
    if len(processes) != 13:
        raise CampaignExecutionError(
            "qualification must create exactly 13 independent workers"
        )
    started: dict[str, mp.Process] = {}
    for checkpoint_id, process in processes.items():
        try:
            process.start()
            started[checkpoint_id] = process
        except BaseException as exc:
            row = _stage_by_id(stage, checkpoint_id)
            path = (
                runtime_root
                / "workflows"
                / row["workflow_id"]
                / "workflow_completion.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(
                path,
                _finalize_workflow(
                    {
                        "schema_version": 2,
                        "qualification_revision": (
                            campaign_contract.QUALIFICATION_REVISION
                        ),
                        "checkpoint_id": checkpoint_id,
                        "status": "failure",
                        "terminal": True,
                        "failure_preserved": True,
                        "failure_code": "worker_process_start_failed",
                        "failure_reason": (
                            "controller: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        "failure_phase": "controller",
                        "replacement_submitted": False,
                        "source_checkpoint": copy.deepcopy(
                            row["checkpoint"]
                        ),
                        "stage_manifest_sha256": stage[
                            "stage_manifest_sha256"
                        ],
                        "recipe_fidelity": copy.deepcopy(
                            campaign_contract.FROZEN_QUALIFICATION_FIDELITY
                        ),
                        "runtime_overlay": copy.deepcopy(
                            campaign_contract.FROZEN_QUALIFICATION_RUNTIME_OVERLAY
                        ),
                        "infrastructure_retry_policy": copy.deepcopy(
                            campaign_contract.FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY
                        ),
                        "phase_recovery_policy": copy.deepcopy(
                            campaign_contract.FROZEN_QUALIFICATION_PHASE_RECOVERY_POLICY
                        ),
                        "execution_plan": copy.deepcopy(
                            row["execution_plan"]
                        ),
                        "terminal_at_utc": utc_timestamp(),
                        "agent_intervention_flags": {
                            name: False
                            for name in campaign_contract.AGENT_FLAGS
                        },
                    }
                ),
            )
    for process in started.values():
        process.join()

    completion = build_completion(
        contract=contract,
        stage=stage,
        runtime_root=runtime_root,
        exit_codes={
            checkpoint_id: (
                started[checkpoint_id].exitcode
                if checkpoint_id in started
                else None
            )
            for checkpoint_id in processes
        },
    )
    atomic_json(qualification_path, completion)
    handoff = build_handoff(
        contract=contract,
        completion=completion,
        qualification_path=qualification_path,
    )
    atomic_json(runtime_root / "automatic_handoff.json", handoff)
    return completion


def _load_stage(
    path: Path,
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        raise CampaignExecutionError(
            f"PTM stage manifest is unavailable: {path}"
        )
    return validate_stage_manifest(
        json.loads(path.read_text(encoding="utf-8")),
        contract=contract,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=DEFAULT_RUNTIME_ROOT,
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
    parser.add_argument(
        "--env-file",
        type=Path,
        default=run_campaign.ENV_PATH,
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--stage", action="store_true")
    actions.add_argument("--check-stage", action="store_true")
    actions.add_argument("--launch", action="store_true")
    args = parser.parse_args(argv)

    contract_path = args.contract.resolve()
    contract = run_campaign.load_contract(contract_path)
    stage_path = args.stage_manifest.resolve()
    expected_stage = Path(
        contract["qualification_policy"]["ptm_stage_manifest_path"]
    ).resolve()
    if stage_path != expected_stage:
        raise CampaignExecutionError(
            "PTM stage manifest path differs from the sealed contract"
        )
    if not (args.stage or args.check_stage or args.launch):
        print(json.dumps(qualification_plan(contract), indent=2, sort_keys=True))
        return 0

    run_campaign.load_env_file(args.env_file)
    run_campaign.configure_slurm_runtime(contract)
    if args.stage:
        if stage_path.is_file():
            stage = _load_stage(stage_path, contract=contract)
            verify_stage_remote(stage, contract=contract)
        else:
            stage = stage_runtime_inputs(
                contract=contract,
                local_cache_root=args.local_cache_root,
                lustre_input_root=args.lustre_input_root,
            )
            atomic_json(stage_path, stage)
        print(
            json.dumps(
                {
                    "stage_manifest_path": str(stage_path),
                    "stage_manifest_sha256": stage[
                        "stage_manifest_sha256"
                    ],
                    "ptm_count": len(stage["ptms"]),
                    "model_runs": 0,
                    "scheduler_jobs_submitted": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.check_stage:
        stage = _load_stage(stage_path, contract=contract)
        result = verify_stage_remote(stage, contract=contract)
        print(json.dumps(result, sort_keys=True))
        return 0

    completion = launch(
        contract_path=contract_path,
        stage_path=stage_path,
        runtime_root=args.runtime_root.resolve(),
    )
    print(json.dumps(completion, sort_keys=True))
    return 0 if completion["successful_workflows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
