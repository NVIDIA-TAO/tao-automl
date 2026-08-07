#!/usr/bin/env python3

"""Rerun full Mask Grounding DINO COCO evaluation with the fixed evaluator.

The four three-epoch checkpoints produced by qualification v3 are immutable and
are not retrained.  This successor submits one standalone full-validation job
per checkpoint, concurrently, on one node/eight A100s in the same pinned SQSH.
The source overlay is verified and installed into an ephemeral PYTHONPATH tree;
the package inside the SQSH is never modified.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Mapping

from tao_automl.ptm_registry import canonical_sha256

from . import campaign_contract, qualification_campaign, run_campaign


HERE = Path(__file__).resolve().parent
DEFAULT_PREDECESSOR_CONTRACT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask_grounding_dino_coco2017_ptm_qualification_v3/qualification.v3.json"
)
DEFAULT_PREDECESSOR_COMPLETION = DEFAULT_PREDECESSOR_CONTRACT.with_name(
    "completion.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask_grounding_dino_coco2017_ptm_qualification_v5"
)
DEFAULT_CONTRACT = DEFAULT_OUTPUT_ROOT / "qualification.v5.json"
DEFAULT_REPOSITORY = HERE.parents[2]
DEFAULT_TAO_PYTORCH = Path(
    "/localhome/local-rarunachalam/.tao/worktrees/"
    "tao-pytorch-mask-grounding-dino-coco-evaluator"
)
CAMPAIGN_ID = (
    "mask_grounding_dino-coco2017-coco-metric-recovery-v5-20260802"
)
OVERLAY = {
    "schema_version": 1,
    "source_repository": "tao-pytorch",
    "source_commit": "896bf2e3441bf609593e0d85ecb0d0454c9c8b71",
    "product_fix_commit": "385e53ebcacd7629cfed7d465e54262df3cfda8e",
    "merge_request": 669,
    "archive_path": (
        "/lustre/fsw/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/"
        "tao-pytorch-overlays/mask-grounding-dino-coco-evaluator/896bf2e3/"
        "mask-grounding-dino-coco-evaluator-overlay.tar"
    ),
    "archive_sha256": (
        "4e6541331d24c011b7364577a49f61acdc78a835368d5195360049bc6313b681"
    ),
    "archive_size_bytes": 40960,
    "archive_root": "mask-grounding-dino-coco-evaluator-overlay",
    "base_site_packages": "/usr/local/lib/python3.12/dist-packages",
    "installed_package_mutated": False,
}


class MetricRecoveryError(RuntimeError):
    """The evaluator recovery campaign contract or execution is invalid."""


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _validated_completion(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MetricRecoveryError(f"predecessor completion is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = copy.deepcopy(value)
    supplied = payload.pop("evidence_sha256", None)
    if supplied != canonical_sha256(payload):
        raise MetricRecoveryError("predecessor completion integrity failed")
    return value


def _checkpoint_records(completion: Mapping[str, Any]) -> list[dict[str, Any]]:
    workflows = completion.get("workflows")
    if not isinstance(workflows, list) or len(workflows) != 4:
        raise MetricRecoveryError("v3 must contain exactly four workflows")
    records = []
    for workflow in workflows:
        train = workflow.get("diagnostics", {}).get("train_job", {})
        evaluation = workflow.get("diagnostics", {}).get("evaluation_job", {})
        checkpoint = train.get("terminal_checkpoint")
        if (
            workflow.get("status") != "failure"
            or workflow.get("failure_code") != "task_correct_metric_missing"
            or train.get("status") != "Complete"
            or evaluation.get("status") != "Complete"
            or not isinstance(checkpoint, Mapping)
            or checkpoint.get("terminal_epoch_index") != 2
            or checkpoint.get("training_epochs") != 3
        ):
            raise MetricRecoveryError(
                "v3 is not the expected completed metric-only failure cohort"
            )
        records.append(
            {
                "checkpoint_id": workflow["checkpoint_id"],
                "terminal_checkpoint": copy.deepcopy(dict(checkpoint)),
                "source_train_job_id": train["tao_job_id"],
                "failed_evaluation_job_id": evaluation["tao_job_id"],
                "failed_workflow_sha256": workflow["workflow_sha256"],
            }
        )
    return sorted(records, key=lambda item: item["checkpoint_id"])


def build_contract(
    *,
    predecessor_contract: Path = DEFAULT_PREDECESSOR_CONTRACT,
    predecessor_completion: Path = DEFAULT_PREDECESSOR_COMPLETION,
    repository: Path = DEFAULT_REPOSITORY,
    tao_pytorch: Path = DEFAULT_TAO_PYTORCH,
) -> dict[str, Any]:
    """Seal evaluation-only recovery against the frozen v3 checkpoints."""
    v3_contract = qualification_campaign.load_qualification_contract(
        predecessor_contract
    )
    v3_completion = _validated_completion(predecessor_completion)
    if (
        v3_completion.get("campaign_id")
        != v3_contract["qualification_policy"]["qualification_campaign_id"]
        or v3_completion.get("qualification_contract_sha256")
        != v3_contract["contract_sha256"]
    ):
        raise MetricRecoveryError("v3 contract and completion do not match")
    if _git(repository, "status", "--porcelain"):
        raise MetricRecoveryError("AutoML repository must be clean")
    if _git(tao_pytorch, "status", "--porcelain"):
        raise MetricRecoveryError("TAO PyTorch evaluator worktree must be clean")
    if _git(tao_pytorch, "rev-parse", "HEAD") != OVERLAY["source_commit"]:
        raise MetricRecoveryError("TAO PyTorch evaluator source changed")

    runtime = copy.deepcopy(v3_contract["runtime"])
    runtime.update(
        {
            "repository": str(repository.resolve()),
            "source_commit": _git(repository, "rev-parse", "HEAD"),
            "source_dirty": False,
            "tao_pytorch_worktree": str(tao_pytorch.resolve()),
            "metric_recovery_evidence_path": str(
                (DEFAULT_OUTPUT_ROOT / "completion.json").resolve()
            ),
        }
    )
    value = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "model": "mask_grounding_dino",
        "task": "category_prompted_grounded_instance_segmentation",
        "primary_metric": "segm_val_mAP50_95",
        "dataset": copy.deepcopy(v3_contract["dataset"]),
        "sqsh": copy.deepcopy(v3_contract["sqsh"]),
        "runtime": runtime,
        "ptm_inventory": copy.deepcopy(v3_contract["ptm_inventory"]),
        "overlay": copy.deepcopy(OVERLAY),
        "predecessor": {
            "contract_path": str(predecessor_contract.resolve()),
            "contract_file_sha256": campaign_contract.sha256_file(
                predecessor_contract
            ),
            "contract_sha256": v3_contract["contract_sha256"],
            "completion_path": str(predecessor_completion.resolve()),
            "completion_file_sha256": campaign_contract.sha256_file(
                predecessor_completion
            ),
            "evidence_sha256": v3_completion["evidence_sha256"],
            "immutable": True,
        },
        "execution": {
            "scope": "standalone_full_validation_only",
            "training_jobs_submitted": 0,
            "checkpoint_recovery_jobs_submitted": 0,
            "evaluation_jobs_expected": 4,
            "evaluations_submitted_concurrently": True,
            "nodes_per_job": 1,
            "gpus_per_job": 8,
            "cpu_model_runs": 0,
            "smoke_model_runs": 0,
            "mini_step_runs": 0,
            "selection_invoked": False,
            "validation_measurements_feed_selection": False,
        },
        "checkpoints": _checkpoint_records(v3_completion),
        "agent_intervention_flags": {
            name: False for name in campaign_contract.AGENT_FLAGS
        },
    }
    value["contract_sha256"] = canonical_sha256(value)
    return value


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all frozen identities before scheduler construction."""
    document = copy.deepcopy(dict(value))
    payload = copy.deepcopy(document)
    supplied = payload.pop("contract_sha256", None)
    execution = document.get("execution", {})
    if (
        supplied != canonical_sha256(payload)
        or document.get("campaign_id") != CAMPAIGN_ID
        or document.get("model") != "mask_grounding_dino"
        or document.get("overlay") != OVERLAY
        or len(document.get("checkpoints", ())) != 4
        or execution.get("scope") != "standalone_full_validation_only"
        or execution.get("training_jobs_submitted") != 0
        or execution.get("evaluation_jobs_expected") != 4
        or execution.get("gpus_per_job") != 8
        or any(document.get("agent_intervention_flags", {}).values())
    ):
        raise MetricRecoveryError("metric-recovery contract changed")
    campaign_contract.validate_dataset_record(document["dataset"])
    return document


def load_contract(path: Path) -> dict[str, Any]:
    return validate_contract(json.loads(path.read_text(encoding="utf-8")))


def _verify_file(path: Path, expected_sha256: str) -> None:
    if (
        not path.is_file()
        or campaign_contract.sha256_file(path) != expected_sha256
    ):
        raise MetricRecoveryError(f"sealed file changed: {path}")


def verify_launch_inputs(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Verify local sources, frozen evidence, overlay, and checkpoints."""
    runtime = contract["runtime"]
    repository = Path(runtime["repository"])
    tao_pytorch = Path(runtime["tao_pytorch_worktree"])
    if (
        _git(repository, "status", "--porcelain")
        or _git(repository, "rev-parse", "HEAD") != runtime["source_commit"]
        or _git(tao_pytorch, "status", "--porcelain")
        or _git(tao_pytorch, "rev-parse", "HEAD")
        != contract["overlay"]["source_commit"]
    ):
        raise MetricRecoveryError("sealed source worktree changed")
    predecessor = contract["predecessor"]
    _verify_file(
        Path(predecessor["contract_path"]),
        predecessor["contract_file_sha256"],
    )
    _verify_file(
        Path(predecessor["completion_path"]),
        predecessor["completion_file_sha256"],
    )

    overlay = contract["overlay"]
    identity = run_campaign._remote_file_identity(overlay["archive_path"])
    if (
        identity["sha256"] != overlay["archive_sha256"]
        or identity["size_bytes"] != overlay["archive_size_bytes"]
    ):
        raise MetricRecoveryError("runtime overlay identity changed")
    readonly = run_campaign.remote_output(
        f"test ! -w {shlex.quote(overlay['archive_path'])} && echo readonly"
    ).strip()
    if readonly != "readonly":
        raise MetricRecoveryError("runtime overlay is writable")

    checkpoint_evidence = []
    for record in contract["checkpoints"]:
        checkpoint = record["terminal_checkpoint"]
        observed = run_campaign._remote_file_identity(checkpoint["path"])
        if (
            observed["sha256"] != checkpoint["sha256"]
            or observed["size_bytes"] != checkpoint["size_bytes"]
        ):
            raise MetricRecoveryError(
                f"terminal checkpoint changed: {record['checkpoint_id']}"
            )
        checkpoint_evidence.append(
            {
                "checkpoint_id": record["checkpoint_id"],
                "identity": observed,
            }
        )
    return {"overlay": identity, "checkpoints": checkpoint_evidence}


def overlay_install_command(contract: Mapping[str, Any]) -> str:
    """Return the fail-closed in-container overlay installation prefix."""
    overlay = contract["overlay"]
    archive = shlex.quote(overlay["archive_path"])
    digest = shlex.quote(overlay["archive_sha256"])
    base = shlex.quote(overlay["base_site_packages"])
    installer = shlex.quote(
        f"{overlay['archive_root']}/install_overlay.py"
    )
    return " ".join(
        [
            "mgdino_overlay_tmp=$(mktemp -d",
            "/tmp/mgdino-coco-evaluator.XXXXXX)",
            "&& test \"$(sha256sum",
            archive,
            "| awk '{print $1}')\" =",
            digest,
            "&& tar --extract --file",
            archive,
            "--directory \"$mgdino_overlay_tmp\"",
            "&& mgdino_overlay_site=\"$mgdino_overlay_tmp/site-packages\"",
            "&& mkdir -p \"$mgdino_overlay_site/nvidia_tao_pytorch\"",
            "&& cp -as",
            f"{base}/nvidia_tao_pytorch/.",
            "\"$mgdino_overlay_site/nvidia_tao_pytorch/\"",
            f"&& python \"$mgdino_overlay_tmp\"/{installer}",
            "--base-site-packages",
            base,
            "--site-packages \"$mgdino_overlay_site\"",
            "--receipt \"${TAO_RESULTS_ROOT:?}/${TAO_JOB_ID:?}/"
            "runtime_overlay/receipt.json\"",
            "&& export PYTHONPATH=\"$mgdino_overlay_site"
            "${PYTHONPATH:+:$PYTHONPATH}\"",
        ]
    )


def _run_one(
    contract: Mapping[str, Any],
    record: Mapping[str, Any],
    runtime_root: Path,
    sdk: Any,
) -> dict[str, Any]:
    checkpoint_id = record["checkpoint_id"]
    workflow_dir = runtime_root / checkpoint_id.replace("/", "_")
    workflow_dir.mkdir(parents=True, exist_ok=True)
    events = workflow_dir / "events.jsonl"
    _, evaluate_spec = qualification_campaign._qualification_specs(
        contract,
        checkpoint_id,
        record["terminal_checkpoint"]["path"],
    )
    evaluate_spec["evaluate"]["checkpoint"] = record[
        "terminal_checkpoint"
    ]["path"]
    base_command, base_command_sha256 = qualification_campaign._entrypoint(
        contract, "evaluate", evaluate_spec
    )
    command = f"{overlay_install_command(contract)} && (\n{base_command}\n)"
    job = qualification_campaign._submit(sdk, contract, command)
    progress = {
        "checkpoint_id": checkpoint_id,
        "source_train_job_id": record["source_train_job_id"],
        "failed_evaluation_job_id": record["failed_evaluation_job_id"],
        "terminal_checkpoint": copy.deepcopy(record["terminal_checkpoint"]),
        "evaluation_job": {
            "tao_job_id": job.id,
            "status": "submitted",
            "nodes": 1,
            "gpus": 8,
            "spec_sha256": canonical_sha256(evaluate_spec),
            "base_command_sha256": base_command_sha256,
            "wrapped_command_sha256": run_campaign.text_sha256(command),
            "overlay_sha256": contract["overlay"]["archive_sha256"],
        },
    }
    run_campaign.atomic_json(workflow_dir / "workflow_progress.json", progress)
    status = run_campaign._wait_for_job(
        sdk,
        job.id,
        events=events,
        phase="coco_metric_recovery_evaluate",
        mode="qualification",
        candidate_id=checkpoint_id,
    )
    progress["evaluation_job"]["status"] = status
    if status != "Complete":
        progress["status"] = "failure"
        progress["failure_code"] = "full_evaluation_failed"
        progress["failure_analysis"] = sdk.get_failure_analysis(job.id)
    else:
        segm = qualification_campaign._status_values(
            sdk,
            job.id,
            action="evaluate",
            names=("[segm] test_mAP@50-95",),
        )
        bbox = qualification_campaign._status_values(
            sdk,
            job.id,
            action="evaluate",
            names=("[bbox] test_mAP@50-95",),
        )
        progress["evaluation_job"]["segm_map50_95_values"] = segm
        progress["evaluation_job"]["bbox_map50_95_values"] = bbox
        progress["status"] = (
            "success"
            if segm
            and bbox
            and all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (*segm, *bbox))
            else "failure"
        )
        if progress["status"] == "success":
            progress["segm_val_mAP50_95"] = segm[-1]
            progress["bbox_val_mAP50_95"] = bbox[-1]
            progress["metric_sanity_gate_passed"] = (
                segm[-1]
                >= campaign_contract.FROZEN_VALIDATION_SANITY_MIN_MASK_AP
            )
        else:
            progress["failure_code"] = "task_correct_metric_missing"
    progress["training_reused"] = True
    progress["training_jobs_submitted"] = 0
    progress["agent_intervention_flags"] = {
        name: False for name in campaign_contract.AGENT_FLAGS
    }
    progress["workflow_sha256"] = canonical_sha256(progress)
    run_campaign.atomic_json(workflow_dir / "workflow_progress.json", progress)
    return progress


def _sdk_for_workflow(sdk_type: Any, workflow_dir: Path) -> Any:
    """Create isolated SDK state only after its directory is durable."""
    workflow_dir.mkdir(parents=True, exist_ok=True)
    return sdk_type(
        poll_interval=10,
        state_file=workflow_dir / "slurm_state.json",
    )


def launch(
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    runtime_root: Path = DEFAULT_OUTPUT_ROOT,
    env_path: Path = run_campaign.ENV_PATH,
) -> dict[str, Any]:
    """Verify and submit all four full evaluations concurrently."""
    contract = load_contract(contract_path)
    runtime_root.mkdir(parents=True, exist_ok=True)
    loaded = run_campaign.load_env_file(env_path)
    run_campaign.configure_slurm_runtime(contract)
    verified = verify_launch_inputs(contract)
    run_campaign.atomic_json(
        runtime_root / "submission_provenance.json",
        {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "contract_sha256": contract["contract_sha256"],
            "loaded_secret_keys": list(loaded),
            "secret_values_recorded": False,
            "verified_inputs": verified,
            "evaluation_jobs_expected": 4,
            "submitted_concurrently": True,
            "nodes_per_job": 1,
            "gpus_per_job": 8,
            "training_jobs_submitted": 0,
            "cpu_model_runs": 0,
            "smoke_model_runs": 0,
            "mini_step_runs": 0,
            "submitted_at_utc": run_campaign.utc_timestamp(),
        },
    )

    from tao_sdk.platforms.slurm import SlurmSDK

    def invoke(record: Mapping[str, Any]) -> dict[str, Any]:
        workflow_dir = runtime_root / record["checkpoint_id"].replace("/", "_")
        try:
            sdk = _sdk_for_workflow(SlurmSDK, workflow_dir)
            return _run_one(contract, record, runtime_root, sdk)
        except Exception as exc:  # preserve every frozen arm independently
            failure = {
                "checkpoint_id": record["checkpoint_id"],
                "status": "failure",
                "failure_code": "metric_recovery_workflow_exception",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "terminal_checkpoint": copy.deepcopy(
                    record["terminal_checkpoint"]
                ),
                "training_reused": True,
                "training_jobs_submitted": 0,
                "agent_intervention_flags": {
                    name: False for name in campaign_contract.AGENT_FLAGS
                },
            }
            failure["workflow_sha256"] = canonical_sha256(failure)
            run_campaign.atomic_json(
                workflow_dir / "workflow_progress.json", failure
            )
            return failure

    results = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=4,
        thread_name_prefix="mgdino-coco-metric-recovery",
    ) as pool:
        futures = {
            pool.submit(invoke, record): record["checkpoint_id"]
            for record in contract["checkpoints"]
        }
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()
    workflows = [results[key] for key in sorted(results)]
    completion = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "model": "mask_grounding_dino",
        "task": "category_prompted_grounded_instance_segmentation",
        "primary_metric": "segm_val_mAP50_95",
        "contract_sha256": contract["contract_sha256"],
        "overlay": copy.deepcopy(contract["overlay"]),
        "predecessor": copy.deepcopy(contract["predecessor"]),
        "training_jobs_submitted": 0,
        "evaluation_jobs_submitted": 4,
        "evaluations_submitted_concurrently": True,
        "cpu_model_runs": 0,
        "smoke_model_runs": 0,
        "mini_step_runs": 0,
        "selection_invoked": False,
        "validation_measurements_feed_selection": False,
        "workflows": workflows,
        "agent_intervention_flags": {
            name: False for name in campaign_contract.AGENT_FLAGS
        },
    }
    completion["evidence_sha256"] = canonical_sha256(completion)
    run_campaign.atomic_json(runtime_root / "completion.json", completion)
    return completion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--env-file", type=Path, default=run_campaign.ENV_PATH)
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args(argv)
    if args.seal:
        value = build_contract()
        if args.contract.exists():
            if json.loads(args.contract.read_text(encoding="utf-8")) != value:
                raise MetricRecoveryError(
                    "existing metric-recovery contract differs"
                )
        else:
            run_campaign.atomic_json(args.contract, value)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.launch:
        completion = launch(
            contract_path=args.contract.resolve(),
            runtime_root=args.runtime_root.resolve(),
            env_path=args.env_file.resolve(),
        )
        print(json.dumps(completion, indent=2, sort_keys=True))
        return 0 if all(item["status"] == "success" for item in completion["workflows"]) else 1
    print(json.dumps(load_contract(args.contract), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
