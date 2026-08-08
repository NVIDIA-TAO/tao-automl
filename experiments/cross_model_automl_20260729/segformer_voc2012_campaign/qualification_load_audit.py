#!/usr/bin/env python3

"""Seal a separate PTM-load audit for terminal SegFormer qualification v4.

Qualification v4 predates the structured positive-load receipt.  This audit
does not alter its workflows or reinterpret finite metrics as load evidence.
It binds the immutable v4 completion, stage, contract, train-job database row,
and complete train-log identity, then classifies the legacy loader messages.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
import shlex
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tao_automl.ptm_registry import canonical_sha256

try:
    from . import run_campaign
except ImportError:  # pragma: no cover - direct script execution
    import sys

    repository = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repository))
    from experiments.cross_model_automl_20260729.segformer_voc2012_campaign import (  # noqa: E501
        run_campaign,
    )


V4_CAMPAIGN_ID = "segformer-voc2012-direct-full-ptm-qualification-v4"
V4_QUALIFICATION_REVISION = 4
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_POSITIVE_RE = re.compile(
    r"Loaded (?P<count>[1-9][0-9]*) compatible SegFormer "
    r"(?:(?P<component>model|backbone) )?pretrained tensors from "
    r"(?P<path>/lustre/[^:\s]+): (?P<keys>\[[^\n]*\])"
)
_LEGACY_LOAD_RE = re.compile(
    r"Loaded pretrained weights from (?P<path>/lustre/[^\s()]+)"
)
_INCOMPATIBLE_RE = re.compile(
    r"_IncompatibleKeys\(missing_keys=(?P<missing>\[[^\n]*?\]), "
    r"unexpected_keys=(?P<unexpected>\[[^\n]*?\])\)"
)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_ALLOWED_CLASSIFIER_UNEXPECTED_KEYS = frozenset(
    {"head.fc.bias", "head.fc.weight"}
)


class QualificationLoadAuditError(RuntimeError):
    """The immutable v4 evidence cannot support a load audit."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sealed_json(
    path: str | Path,
    *,
    internal_key: str,
    expected_whole_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = Path(path).resolve()
    if (
        _SHA256_RE.fullmatch(expected_whole_sha256) is None
        or not resolved.is_file()
    ):
        raise QualificationLoadAuditError(
            f"sealed JSON input is unavailable or has invalid identity: {resolved}"
        )
    content = resolved.read_bytes()
    whole = _sha256_bytes(content)
    if whole != expected_whole_sha256:
        raise QualificationLoadAuditError(
            f"sealed JSON whole-file SHA changed: {resolved}"
        )
    try:
        document = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise QualificationLoadAuditError(
            f"sealed JSON is invalid: {resolved}"
        ) from exc
    if not isinstance(document, Mapping):
        raise QualificationLoadAuditError(
            f"sealed JSON root is not an object: {resolved}"
        )
    document = copy.deepcopy(dict(document))
    payload = copy.deepcopy(document)
    supplied = payload.pop(internal_key, None)
    if supplied != canonical_sha256(payload):
        raise QualificationLoadAuditError(
            f"sealed JSON internal integrity failed: {resolved}"
        )
    return document, {
        "path": str(resolved),
        "size_bytes": len(content),
        "whole_file_sha256": whole,
        "internal_sha256_field": internal_key,
        "internal_sha256": supplied,
    }


def extract_log_observations(text: str) -> dict[str, Any]:
    """Extract compact, non-secret load observations from a legacy log."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = _ANSI_RE.sub("", text)
    positive = []
    for match in _POSITIVE_RE.finditer(normalized):
        try:
            loaded_keys = ast.literal_eval(match.group("keys"))
        except (SyntaxError, ValueError) as exc:
            raise QualificationLoadAuditError(
                "legacy positive-load keyset is malformed"
            ) from exc
        count = int(match.group("count"))
        if (
            not isinstance(loaded_keys, list)
            or any(not isinstance(item, str) for item in loaded_keys)
            or len(loaded_keys) != count
            or len(set(loaded_keys)) != count
        ):
            raise QualificationLoadAuditError(
                "legacy positive-load count and keyset disagree"
            )
        positive.append(
            {
                "loaded_tensor_count": count,
                "loaded_keyset_sha256": _sha256_bytes(
                    "\n".join(sorted(loaded_keys)).encode("utf-8")
                ),
                "component": match.group("component"),
                "checkpoint": match.group("path"),
            }
        )
    legacy_paths = [
        match.group("path")
        for match in _LEGACY_LOAD_RE.finditer(normalized)
    ]
    incompatible = []
    for match in _INCOMPATIBLE_RE.finditer(normalized):
        try:
            missing = ast.literal_eval(match.group("missing"))
            unexpected = ast.literal_eval(match.group("unexpected"))
        except (SyntaxError, ValueError) as exc:
            raise QualificationLoadAuditError(
                "legacy _IncompatibleKeys observation is malformed"
            ) from exc
        if (
            not isinstance(missing, list)
            or not isinstance(unexpected, list)
            or any(not isinstance(item, str) for item in missing)
            or any(not isinstance(item, str) for item in unexpected)
        ):
            raise QualificationLoadAuditError(
                "legacy _IncompatibleKeys observation is not a string-key list"
            )
        unexpected_backbone = [
            item for item in unexpected if item.startswith("backbone.")
        ]
        non_backbone = [
            item for item in unexpected if not item.startswith("backbone.")
        ]
        incompatible.append(
            {
                "missing_tensor_count": len(missing),
                "unexpected_tensor_count": len(unexpected),
                "all_unexpected_backbone_prefix": bool(unexpected)
                and all(item.startswith("backbone.") for item in unexpected),
                "unexpected_backbone_prefix_count": len(
                    unexpected_backbone
                ),
                "allowlisted_classifier_unexpected_count": len(
                    non_backbone
                ),
                "allowlisted_classifier_keyset_sha256": _sha256_bytes(
                    "\n".join(sorted(non_backbone)).encode("utf-8")
                ),
                "all_loadable_unexpected_backbone_prefix": (
                    bool(unexpected_backbone)
                    and set(non_backbone).issubset(
                        _ALLOWED_CLASSIFIER_UNEXPECTED_KEYS
                    )
                ),
                "missing_keyset_sha256": _sha256_bytes(
                    "\n".join(sorted(missing)).encode("utf-8")
                ),
                "unexpected_keyset_sha256": _sha256_bytes(
                    "\n".join(sorted(unexpected)).encode("utf-8")
                ),
            }
        )
    return {
        "positive": positive,
        "legacy_load_paths": legacy_paths,
        "incompatible": incompatible,
    }


def _unique(values: list[Any]) -> list[Any]:
    return [
        value
        for _, value in sorted(
            {
                canonical_sha256(value): copy.deepcopy(value)
                for value in values
            }.items()
        )
    ]


def classify_log_observations(
    observations: Mapping[str, Any],
    *,
    checkpoint_path: str,
    checkpoint_target: str,
) -> dict[str, Any]:
    """Classify exact-path legacy evidence; ambiguity always fails closed."""
    expected_component = {
        "train.pretrained_model_path": "model",
        "model.backbone.pretrained_backbone_path": "backbone",
    }.get(checkpoint_target)
    if expected_component is None or not checkpoint_path.startswith("/lustre/"):
        raise QualificationLoadAuditError("checkpoint audit target is invalid")
    try:
        positive_all = list(observations["positive"])
        legacy_all = list(observations["legacy_load_paths"])
        incompatible_all = list(observations["incompatible"])
    except (KeyError, TypeError) as exc:
        raise QualificationLoadAuditError(
            "load observations are incomplete"
        ) from exc
    positive = _unique(positive_all)
    legacy_paths = sorted(set(legacy_all))
    incompatible = _unique(incompatible_all)
    matching_positive = [
        item
        for item in positive
        if item.get("checkpoint") == checkpoint_path
        and item.get("loaded_tensor_count", 0) > 0
        and (
            item.get("component") == expected_component
            or (
                expected_component == "model"
                and item.get("component") is None
            )
        )
    ]
    foreign_positive = [
        item for item in positive if item not in matching_positive
    ]
    evidence = {
        "expected_component": expected_component,
        "positive_observation_occurrences": len(positive_all),
        "unique_positive_observations": positive,
        "legacy_load_path_occurrences": len(legacy_all),
        "unique_legacy_load_paths": legacy_paths,
        "incompatible_observation_occurrences": len(incompatible_all),
        "unique_incompatible_observations": incompatible,
        "duplicate_identical_observations_allowed": True,
        "finite_metric_override_allowed": False,
    }
    if (
        len(matching_positive) == 1
        and not foreign_positive
        and not incompatible
    ):
        return {
            **evidence,
            "ptm_load_success": True,
            "classification": "positive_compatible_tensor_load",
            "loaded_tensor_count": matching_positive[0][
                "loaded_tensor_count"
            ],
        }
    all_missing_unexpected = bool(incompatible) and all(
        item.get("missing_tensor_count", 0) > 0
        and item.get("unexpected_tensor_count", 0) > 0
        and item.get("all_loadable_unexpected_backbone_prefix") is True
        for item in incompatible
    )
    if (
        expected_component == "backbone"
        and checkpoint_path in legacy_paths
        and not positive
        and all_missing_unexpected
        and len(incompatible) == 1
    ):
        return {
            **evidence,
            "ptm_load_success": False,
            "classification": (
                "all_missing_all_unexpected_backbone_prefix"
            ),
            "loaded_tensor_count": 0,
        }
    return {
        **evidence,
        "ptm_load_success": False,
        "classification": "load_evidence_missing_mismatched_or_ambiguous",
        "loaded_tensor_count": 0,
    }


def _train_log_locator(
    runtime_root: Path,
    *,
    workflow_id: str,
    tao_job_id: str,
) -> dict[str, str]:
    database = runtime_root / "workflows" / workflow_id / "slurm_state.db"
    if not database.is_file():
        raise QualificationLoadAuditError(
            f"train-job database is unavailable: {database}"
        )
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT job_id, status, specs FROM jobs WHERE job_id = ?",
            (tao_job_id,),
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != 1:
        raise QualificationLoadAuditError(
            f"no unique train-job row for {tao_job_id}"
        )
    job_id, status, specs_text = rows[0]
    try:
        runtime = json.loads(specs_text)["_slurm_runtime"]
        slurm_job_id = str(runtime["slurm_job_id"])
        job_name = str(runtime["job_name"])
        log_dir = str(runtime["log_dir"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QualificationLoadAuditError(
            f"train-job runtime metadata is invalid for {tao_job_id}"
        ) from exc
    if (
        job_id != tao_job_id
        or not slurm_job_id.isdigit()
        or not job_name.startswith("tao-job-")
        or not log_dir.startswith("/lustre/")
    ):
        raise QualificationLoadAuditError(
            f"train-job log identity is invalid for {tao_job_id}"
        )
    database_components = []
    for candidate in (database, Path(str(database) + "-wal")):
        if candidate.is_file():
            database_components.append(
                {
                    "path": str(candidate.resolve()),
                    "size_bytes": candidate.stat().st_size,
                    "sha256": _sha256_file(candidate),
                }
            )
    row_evidence = {
        "tao_job_id": tao_job_id,
        "slurm_job_id": slurm_job_id,
        "job_name": job_name,
        "database_job_status": str(status),
        "log_path": f"{log_dir.rstrip('/')}/{job_name}-{slurm_job_id}/main.out",
    }
    return {
        **row_evidence,
        "database_path": str(database.resolve()),
        "database_components": database_components,
        "database_query_record_sha256": canonical_sha256(row_evidence),
    }


def _remote_log_evidence(path: str) -> dict[str, Any]:
    """Hash the complete remote log and return compact parsed observations."""
    script = r'''
import ast,hashlib,json,pathlib,re,sys
p=pathlib.Path(sys.argv[1])
h=hashlib.sha256()
with p.open("rb") as f:
    for chunk in iter(lambda:f.read(1048576),b""):
        h.update(chunk)
text=p.read_text(errors="replace")
text=re.sub(r"\x1b\[[0-9;]*m","",text)
pos=[]
for m in re.finditer(r"Loaded ([1-9][0-9]*) compatible SegFormer (?:(model|backbone) )?pretrained tensors from (/lustre/[^:\s]+): (\[[^\n]*\])",text):
    keys=ast.literal_eval(m.group(4));count=int(m.group(1))
    assert isinstance(keys,list) and len(keys)==count and len(set(keys))==count
    assert all(isinstance(x,str) for x in keys)
    pos.append({"loaded_tensor_count":count,"loaded_keyset_sha256":hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest(),"component":m.group(2),"checkpoint":m.group(3)})
legacy=[m.group(1) for m in re.finditer(r"Loaded pretrained weights from (/lustre/[^\s()]+)",text)]
bad=[]
for m in re.finditer(r"_IncompatibleKeys\(missing_keys=(\[[^\n]*?\]), unexpected_keys=(\[[^\n]*?\])\)",text):
    missing=ast.literal_eval(m.group(1));unexpected=ast.literal_eval(m.group(2))
    assert isinstance(missing,list) and isinstance(unexpected,list)
    assert all(isinstance(x,str) for x in missing+unexpected)
    backbone=[x for x in unexpected if x.startswith("backbone.")]
    other=[x for x in unexpected if not x.startswith("backbone.")]
    bad.append({"missing_tensor_count":len(missing),"unexpected_tensor_count":len(unexpected),"all_unexpected_backbone_prefix":bool(unexpected) and all(x.startswith("backbone.") for x in unexpected),"unexpected_backbone_prefix_count":len(backbone),"allowlisted_classifier_unexpected_count":len(other),"allowlisted_classifier_keyset_sha256":hashlib.sha256("\n".join(sorted(other)).encode()).hexdigest(),"all_loadable_unexpected_backbone_prefix":bool(backbone) and set(other).issubset({"head.fc.bias","head.fc.weight"}),"missing_keyset_sha256":hashlib.sha256("\n".join(sorted(missing)).encode()).hexdigest(),"unexpected_keyset_sha256":hashlib.sha256("\n".join(sorted(unexpected)).encode()).hexdigest()})
print(json.dumps({"path":str(p),"size_bytes":p.stat().st_size,"sha256":h.hexdigest(),"observations":{"positive":pos,"legacy_load_paths":legacy,"incompatible":bad}},sort_keys=True,separators=(",",":")))
'''.strip()
    try:
        value = json.loads(
            run_campaign.remote_output(
                f"python3 -c {shlex.quote(script)} {shlex.quote(path)}",
                timeout=1800,
            )
        )
    except Exception as exc:
        raise QualificationLoadAuditError(
            f"remote train log is unavailable or invalid: {path}"
        ) from exc
    if (
        value.get("path") != path
        or not isinstance(value.get("size_bytes"), int)
        or value["size_bytes"] < 1
        or _SHA256_RE.fullmatch(str(value.get("sha256", ""))) is None
        or not isinstance(value.get("observations"), Mapping)
    ):
        raise QualificationLoadAuditError(
            f"remote train log identity is invalid: {path}"
        )
    return value


def build_v4_load_audit(
    *,
    contract_path: str | Path,
    contract_whole_sha256: str,
    completion_path: str | Path,
    completion_whole_sha256: str,
    stage_path: str | Path,
    stage_whole_sha256: str,
    runtime_root: str | Path,
) -> dict[str, Any]:
    """Build the audit only after all immutable v4 workflows are terminal."""
    contract, contract_identity = _sealed_json(
        contract_path,
        internal_key="contract_sha256",
        expected_whole_sha256=contract_whole_sha256,
    )
    completion, completion_identity = _sealed_json(
        completion_path,
        internal_key="evidence_sha256",
        expected_whole_sha256=completion_whole_sha256,
    )
    stage, stage_identity = _sealed_json(
        stage_path,
        internal_key="stage_manifest_sha256",
        expected_whole_sha256=stage_whole_sha256,
    )
    resolved_contract = Path(contract_path).resolve()
    resolved_completion = Path(completion_path).resolve()
    resolved_stage = Path(stage_path).resolve()
    rows = stage.get("ptms")
    workflows = completion.get("workflows")
    if (
        contract.get("qualification_policy", {}).get("revision")
        != V4_QUALIFICATION_REVISION
        or contract.get("qualification_policy", {}).get("campaign_id")
        != V4_CAMPAIGN_ID
        or stage.get("qualification_revision") != V4_QUALIFICATION_REVISION
        or stage.get("campaign_id") != V4_CAMPAIGN_ID
        or completion.get("qualification_revision")
        != V4_QUALIFICATION_REVISION
        or completion.get("campaign_id") != V4_CAMPAIGN_ID
        or stage.get("automl_contract_sha256")
        != contract.get("contract_sha256")
        or completion.get("automl_contract_sha256")
        != contract.get("contract_sha256")
        or completion.get("ptm_stage_manifest_sha256")
        != stage.get("stage_manifest_sha256")
        or Path(
            str(
                contract.get("qualification_policy", {}).get(
                    "qualification_evidence_path", ""
                )
            )
        ).resolve()
        != resolved_completion
        or Path(
            str(
                contract.get("qualification_policy", {}).get(
                    "ptm_stage_manifest_path", ""
                )
            )
        ).resolve()
        != resolved_stage
        or Path(str(completion.get("ptm_stage_manifest_path", ""))).resolve()
        != resolved_stage
        or contract_identity["path"] != str(resolved_contract)
        or completion.get("terminal") is not True
        or not isinstance(rows, list)
        or not isinstance(workflows, list)
        or len(rows) != 13
        or len(workflows) != 13
    ):
        raise QualificationLoadAuditError(
            "v4 contract, stage, or terminal completion identity changed"
        )
    stage_by_id = {
        row.get("checkpoint_id"): row
        for row in rows
        if isinstance(row, Mapping)
    }
    workflow_by_id = {
        row.get("checkpoint_id"): row
        for row in workflows
        if isinstance(row, Mapping)
    }
    if (
        len(stage_by_id) != 13
        or set(workflow_by_id) != set(stage_by_id)
        or any(item.get("terminal") is not True for item in workflows)
    ):
        raise QualificationLoadAuditError(
            "all 13 v4 workflows must be uniquely terminal before audit"
        )

    root = Path(runtime_root).resolve()
    audited = []
    for checkpoint_id in sorted(stage_by_id):
        staged = stage_by_id[checkpoint_id]
        workflow = workflow_by_id[checkpoint_id]
        workflow_payload = copy.deepcopy(dict(workflow))
        workflow_sha = workflow_payload.pop("workflow_sha256", None)
        if workflow_sha != canonical_sha256(workflow_payload):
            raise QualificationLoadAuditError(
                f"v4 workflow integrity failed: {checkpoint_id}"
            )
        checkpoint = staged.get("checkpoint", {})
        workflow_checkpoint = workflow.get("source_checkpoint")
        jobs = workflow.get("jobs", {})
        train = jobs.get("train") if isinstance(jobs, Mapping) else None
        if (
            not isinstance(checkpoint, Mapping)
            or not str(checkpoint.get("path", "")).startswith("/lustre/")
            or not isinstance(workflow_checkpoint, Mapping)
            or dict(workflow_checkpoint) != dict(checkpoint)
            or workflow.get("stage_manifest_sha256")
            != stage.get("stage_manifest_sha256")
            or not isinstance(train, Mapping)
            or not isinstance(train.get("tao_job_id"), str)
        ):
            raise QualificationLoadAuditError(
                f"v4 train identity is incomplete: {checkpoint_id}"
            )
        locator = _train_log_locator(
            root,
            workflow_id=str(staged["workflow_id"]),
            tao_job_id=train["tao_job_id"],
        )
        log = _remote_log_evidence(locator["log_path"])
        classification = classify_log_observations(
            log["observations"],
            checkpoint_path=str(checkpoint["path"]),
            checkpoint_target=str(staged["checkpoint_target"]),
        )
        workflow_success = workflow.get("status") == "success"
        audited.append(
            {
                "checkpoint_id": checkpoint_id,
                "checkpoint_target": staged["checkpoint_target"],
                "source_checkpoint": copy.deepcopy(dict(checkpoint)),
                "workflow_status": workflow.get("status"),
                "workflow_sha256": workflow_sha,
                "train_job": locator,
                "train_log": {
                    "path": log["path"],
                    "size_bytes": log["size_bytes"],
                    "sha256": log["sha256"],
                },
                "load_evidence": classification,
                "ptm_load_success": classification["ptm_load_success"],
                "load_qualified_workflow": (
                    workflow_success and classification["ptm_load_success"]
                ),
                "finite_metric_can_override_ptm_load_failure": False,
                "val_miou": workflow.get("train", {}).get("val_miou"),
                "test_miou": workflow.get("evaluation", {}).get("test_miou"),
            }
        )
    value = {
        "schema_version": 2,
        "audit_kind": "segformer_v4_pretrained_load_forensic_audit_v2",
        "qualification_revision": V4_QUALIFICATION_REVISION,
        "qualification_campaign_id": V4_CAMPAIGN_ID,
        "inputs": {
            "contract": contract_identity,
            "completion": completion_identity,
            "stage": stage_identity,
        },
        "audit_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "workflows": audited,
        "workflow_count": len(audited),
        "positive_load_workflows": sum(
            item["ptm_load_success"] for item in audited
        ),
        "ptm_load_failure_workflows": sum(
            not item["ptm_load_success"] for item in audited
        ),
        "all_missing_all_unexpected_backbone_prefix_workflows": sum(
            item["load_evidence"]["classification"]
            == "all_missing_all_unexpected_backbone_prefix"
            for item in audited
        ),
        "load_qualified_workflows": sum(
            item["load_qualified_workflow"] for item in audited
        ),
        "finite_metric_override_allowed": False,
        "original_v4_evidence_mutated": False,
        "scheduler_jobs_submitted": 0,
        "model_runs_executed": 0,
    }
    value["audit_sha256"] = canonical_sha256(value)
    return value


def _write_new_read_only(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise QualificationLoadAuditError(
            f"refusing to overwrite existing audit: {path}"
        )
    run_campaign.atomic_json(path, value)
    path.chmod(0o444)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-whole-sha256", required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--completion-whole-sha256", required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--stage-whole-sha256", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=run_campaign.ENV_PATH,
    )
    args = parser.parse_args(argv)
    # V4 is intentionally validated by this module's frozen v4 schema.  The
    # live campaign validator advances with v5 and must not reinterpret or
    # reject the immutable predecessor before its separate audit is sealed.
    contract, _ = _sealed_json(
        args.contract,
        internal_key="contract_sha256",
        expected_whole_sha256=args.contract_whole_sha256,
    )
    run_campaign.load_env_file(args.env_file)
    run_campaign.configure_slurm_runtime(contract)
    audit = build_v4_load_audit(
        contract_path=args.contract,
        contract_whole_sha256=args.contract_whole_sha256,
        completion_path=args.completion,
        completion_whole_sha256=args.completion_whole_sha256,
        stage_path=args.stage,
        stage_whole_sha256=args.stage_whole_sha256,
        runtime_root=args.runtime_root,
    )
    output = args.output.resolve()
    _write_new_read_only(output, audit)
    print(
        json.dumps(
            {
                "path": str(output),
                "audit_sha256": audit["audit_sha256"],
                "whole_file_sha256": _sha256_file(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
