#!/usr/bin/env python3

"""Replay Mask2Former qualification parsing over immutable v3 GPU output.

This command submits no scheduler or model jobs.  It preserves the sealed v3
completion byte-for-byte and derives a new evidence document by re-reading the
two completed TAO job status streams with the corrected explicit-epoch metric
deduplication policy.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from tao_automl.ptm_registry import canonical_sha256

from . import campaign_contract, manifest_generator, qualification_campaign
from . import run_campaign


DEFAULT_PARENT = qualification_campaign.DEFAULT_RUNTIME_ROOT / "completion.json"
DEFAULT_CONTRACT = qualification_campaign.DEFAULT_CONTRACT
DEFAULT_OUTPUT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask2former_coco2017_ptm_qualification_v3_replay_v1/completion.json"
)
REPLAY_CAMPAIGN_ID = (
    "mask2former-coco2017-direct-full-qualification-v3-replay-v1-20260801"
)


class QualificationReplayError(RuntimeError):
    """The immutable v3 evidence cannot be replayed safely."""


def _job_results_adapter(
    train_job_id: str,
    train_checkpoint_path: str,
) -> Any:
    checkpoint = Path(train_checkpoint_path)
    train_job_root = checkpoint.parents[2]
    if train_job_root.name != train_job_id:
        raise QualificationReplayError(
            "terminal checkpoint is not under its recorded TAO job ID"
        )
    results_root = train_job_root.parent
    return SimpleNamespace(
        get_job_results_dir=lambda job_id: (
            "lustre:///" + str(results_root / job_id).lstrip("/")
        )
    )


def replay_workflow(workflow: Mapping[str, Any]) -> dict[str, Any]:
    if (
        workflow.get("status") != "failure"
        or workflow.get("failure_code") != "task_correct_metric_missing"
        or workflow.get("terminal") is not True
        or workflow.get("failure_preserved") is not True
    ):
        raise QualificationReplayError(
            "parent workflow is not the preserved metric-counting failure"
        )
    diagnostics = workflow.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise QualificationReplayError("parent workflow diagnostics are absent")
    source = diagnostics.get("source_checkpoint")
    train = diagnostics.get("train_job")
    evaluation = diagnostics.get("evaluation_job")
    if not all(isinstance(item, Mapping) for item in (source, train, evaluation)):
        raise QualificationReplayError("parent GPU job identities are incomplete")
    if train.get("status") != "Complete" or evaluation.get("status") != "Complete":
        raise QualificationReplayError("parent GPU jobs are not both complete")
    terminal = train.get("terminal_checkpoint")
    if not isinstance(terminal, Mapping):
        raise QualificationReplayError("parent terminal checkpoint is absent")
    for item, name in ((source, "source checkpoint"), (terminal, "terminal checkpoint")):
        identity = run_campaign._remote_file_identity(str(item["path"]))
        if (
            identity["sha256"] != item.get("sha256")
            or identity["size_bytes"] != item.get("size_bytes")
        ):
            raise QualificationReplayError(f"{name} bytes changed")

    train_job_id = str(train["tao_job_id"])
    evaluation_job_id = str(evaluation["tao_job_id"])
    sdk = _job_results_adapter(train_job_id, str(terminal["path"]))
    mask_values = qualification_campaign._status_epoch_values(
        sdk,
        train_job_id,
        action="train",
        names=(qualification_campaign.VALIDATION_MASK_AP_METRIC,),
    )
    standalone = qualification_campaign._status_values(
        sdk,
        evaluation_job_id,
        action="evaluate",
        names=(qualification_campaign.STANDALONE_MASK_AP_METRIC,),
    )
    standalone50 = qualification_campaign._status_values(
        sdk,
        evaluation_job_id,
        action="evaluate",
        names=(qualification_campaign.STANDALONE_MASK_AP50_METRIC,),
    )
    if (
        len(mask_values) != campaign_contract.FROZEN_TRAINING_EPOCHS
        or not standalone
    ):
        raise QualificationReplayError(
            "corrected replay still lacks the task-correct metric contract"
        )
    flags = {name: False for name in campaign_contract.AGENT_FLAGS}
    replay_diagnostics = copy.deepcopy(dict(diagnostics))
    replay_diagnostics["metric_replay"] = {
        "policy": "explicit_epoch_then_exact_rank_value_v1",
        "parent_workflow_sha256": workflow.get("workflow_sha256"),
        "training_values": mask_values,
        "standalone_values": standalone,
        "standalone_ap50_values": standalone50,
        "scheduler_jobs_submitted": 0,
        "model_jobs_submitted": 0,
    }
    value = {
        "checkpoint_id": workflow["checkpoint_id"],
        "status": "success",
        "terminal": True,
        "failure_preserved": False,
        "source_checkpoint": copy.deepcopy(dict(source)),
        "train": {
            "status": "Complete",
            "full_dataset": True,
            "training_epochs": campaign_contract.FROZEN_TRAINING_EPOCHS,
            "validation_interval": 1,
            "validation_record_count": len(mask_values),
            "nodes": 1,
            "gpus": 8,
            "segm_val_mAP": mask_values[-1],
            "terminal_checkpoint": copy.deepcopy(dict(terminal)),
            "tao_job_id": train_job_id,
        },
        "evaluation": {
            "status": "Complete",
            "full_validation_split": True,
            "nodes": 1,
            "gpus": 8,
            qualification_campaign.STANDALONE_MASK_AP_METRIC: standalone[-1],
            qualification_campaign.STANDALONE_MASK_AP50_METRIC: (
                standalone50[-1] if standalone50 else None
            ),
            "objective_binding": {
                "reported_metric": qualification_campaign.STANDALONE_MASK_AP_METRIC,
                "canonical_metric": qualification_campaign.VALIDATION_MASK_AP_METRIC,
                "value": standalone[-1],
            },
            "tao_job_id": evaluation_job_id,
        },
        "diagnostics": replay_diagnostics,
        "agent_intervention_flags": flags,
    }
    value["workflow_sha256"] = canonical_sha256(value)
    return value


def replay(
    parent_path: Path,
    contract_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise QualificationReplayError(
            "replay output already exists; immutable evidence is not overwritten"
        )
    manifest_generator.qualification_evidence_record(
        parent_path, contract_path
    )
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    workflows = parent.get("workflows")
    if not isinstance(workflows, list) or len(workflows) != 1:
        raise QualificationReplayError("parent workflow cohort changed")
    corrected = replay_workflow(workflows[0])
    value = copy.deepcopy(parent)
    value.update(
        {
            "campaign_id": REPLAY_CAMPAIGN_ID,
            "contract_revision": "qualification_runtime_v3_evidence_replay_v1",
            "workflows": [corrected],
            "evidence_replay": {
                "kind": "immutable_status_metric_deduplication_replay_v1",
                "parent_path": str(parent_path.resolve()),
                "parent_file_sha256": campaign_contract.sha256_file(parent_path),
                "parent_evidence_sha256": parent["evidence_sha256"],
                "retraining_jobs_submitted": 0,
                "evaluation_jobs_submitted": 0,
                "selection_invoked": False,
                "original_evidence_overwritten": False,
            },
        }
    )
    value.pop("evidence_sha256", None)
    value["evidence_sha256"] = canonical_sha256(value)
    run_campaign.atomic_json(output_path, value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", type=Path, default=run_campaign.ENV_PATH)
    args = parser.parse_args(argv)
    run_campaign.load_env_file(args.env_file.resolve())
    value = replay(
        args.parent.resolve(), args.contract.resolve(), args.output.resolve()
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "evidence_sha256": value["evidence_sha256"],
                "workflow_status": value["workflows"][0]["status"],
                "scheduler_jobs_submitted": 0,
                "model_jobs_submitted": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
