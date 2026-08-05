#!/usr/bin/env python3

"""Build the future-only Grounding DINO qualification handoff contract.

Unlike the preserved v1 successor artifact, this document is an armed runtime
contract.  It binds exact staged bytes and predecessor identities but does not
snapshot a transient predecessor status or claim that a job was submitted.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tao_automl.metric_sanity import default_metric_sanity_registry
from tao_automl.ptm_registry import canonical_sha256

try:
    from .contract import (
        AGENT_FLAGS,
        MODEL_ID,
        MODES,
        PreparationError,
        derive_official_ptms,
        read_json,
        sha256_file,
    )
    from .dataset_conversion import validate_conversion_manifest
    from .dataset_stage import validate_stage_record
    from .runtime_input_stage import validate_runtime_input_stage
    from .successor_contract import _evaluate_spec, _load_skill_contract, _train_spec
except ImportError:  # pragma: no cover - direct script execution
    from contract import (  # type: ignore[no-redef]
        AGENT_FLAGS,
        MODEL_ID,
        MODES,
        PreparationError,
        derive_official_ptms,
        read_json,
        sha256_file,
    )
    from dataset_conversion import validate_conversion_manifest
    from dataset_stage import validate_stage_record
    from runtime_input_stage import validate_runtime_input_stage
    from successor_contract import _evaluate_spec, _load_skill_contract, _train_spec


HERE = Path(__file__).resolve().parent
DEFAULT_INPUTS = HERE / "campaign.inputs.v3.json"
DEFAULT_STAGE = HERE / "runtime_inputs.stage.v1.json"
DEFAULT_OUTPUT = HERE / "successor.runtime.contract.v2.json"


def _qualification_jobs(
    *,
    inputs: Mapping[str, Any],
    stage: Mapping[str, Any],
    skill: Mapping[str, Any],
    dataset_paths: Mapping[str, str],
) -> list[dict[str, Any]]:
    stage_by_id = {item["id"]: item for item in stage["official_ptms"]}
    text_root = stage["text_encoder"]["lustre_root"]
    jobs = []
    for index, record in enumerate(derive_official_ptms()):
        staged = stage_by_id[record["id"]]
        checkpoint_path = staged["lustre"]["path"]
        train_spec = _train_spec(
            template=skill["train_template"],
            record=record,
            dataset_paths=dataset_paths,
            ptm_path=checkpoint_path,
            qualification=inputs["qualification"],
        )
        evaluate_spec = _evaluate_spec(
            template=skill["evaluate_template"],
            record=record,
            dataset_paths=dataset_paths,
            qualification=inputs["qualification"],
        )
        for spec in (train_spec, evaluate_spec):
            spec["model"]["text_encoder_type"] = text_root
        jobs.append(
            {
                "workflow_id": f"official_ptm_{index}",
                "ptm_id": record["id"],
                "registry_status_before_qualification": record["status"],
                "staged_checkpoint": {
                    "path": checkpoint_path,
                    "size_bytes": staged["lustre"]["size_bytes"],
                    "sha256": staged["lustre"]["sha256"],
                    "mode": staged["lustre"]["mode"],
                    "verification_mode": staged["verification_mode"],
                },
                "text_encoder_root": text_root,
                "train": {
                    "command": skill["skill_info"]["actions"]["train"]["command"],
                    "spec": train_spec,
                    "spec_sha256": canonical_sha256(train_spec),
                },
                "evaluate": {
                    "command": skill["skill_info"]["actions"]["evaluate"][
                        "command"
                    ],
                    "spec": evaluate_spec,
                    "spec_sha256": canonical_sha256(evaluate_spec),
                    "checkpoint_resolution": (
                        "one exact terminal model_epoch_009_step_*.pth under "
                        "the bound training result root"
                    ),
                    "test_metrics_feed_selection": False,
                },
                "resources": {
                    "nodes": 1,
                    "gpus_per_node": 8,
                    "distributed_workers": 8,
                    "time_hours": 4,
                    "timeout_hours": 3.8,
                },
            }
        )
    return jobs


def build_future_contract(
    *,
    experiment_dir: str | Path,
    inputs: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    here = Path(experiment_dir).resolve()
    validate_runtime_input_stage(stage, inputs=inputs)
    dataset_inputs = inputs["dataset"]
    conversion_path = (here / dataset_inputs["conversion_manifest_file"]).resolve()
    stage_path = (here / dataset_inputs["stage_record_file"]).resolve()
    source_manifest_path = (here / dataset_inputs["manifest_file"]).resolve()
    conversion = read_json(conversion_path)
    dataset_stage = read_json(stage_path)
    source_manifest = read_json(source_manifest_path)
    validate_conversion_manifest(conversion)
    validate_stage_record(dataset_stage)
    runtime = inputs["runtime"]
    skill = _load_skill_contract(Path(runtime["skill_dir"]))
    metric = default_metric_sanity_registry().resolve(MODEL_ID, "val_mAP50")
    if metric.availability != "supported" or metric.task != "object_detection":
        raise PreparationError("Grounding DINO val_mAP50 is not launchable")
    canonical_outputs = conversion["canonical_outputs"]
    dataset_paths = {
        "train_image_dir": source_manifest["splits"]["train"]["images"]["path"],
        "train_odvg_jsonl": canonical_outputs["train_odvg"]["lustre_path"],
        "train_label_map": canonical_outputs["train_label_map"]["lustre_path"],
        "validation_image_dir": source_manifest["splits"]["validation"]["images"][
            "path"
        ],
        "validation_coco_contiguous": canonical_outputs["validation_coco"][
            "lustre_path"
        ],
    }
    jobs = _qualification_jobs(
        inputs=inputs,
        stage=stage,
        skill=skill,
        dataset_paths=dataset_paths,
    )
    ddetr = copy.deepcopy(inputs["predecessor_gates"]["deformable_detr"])
    rtdetr = copy.deepcopy(inputs["predecessor_gates"]["rtdetr"])
    document = {
        "schema_version": 2,
        "campaign_id": inputs["campaign_id"],
        "model": {
            "id": MODEL_ID,
            "task": "category_prompted_open_vocabulary_detection",
        },
        "source": {
            "repository": inputs["source"]["repository"],
            "minimum_ancestor_commit": inputs["source"][
                "minimum_ancestor_commit"
            ],
        },
        "predecessor_release": {
            "deformable_detr": {
                **ddetr,
                "release_scope": (
                    "three_mode_first_candidate_gate_only; candidates 1-19 "
                    "are explicitly not a dependency"
                ),
            },
            "rtdetr": rtdetr,
        },
        "dataset": {
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "conversion_manifest_path": str(conversion_path),
            "conversion_manifest_sha256": sha256_file(conversion_path),
            "stage_record_path": str(stage_path),
            "stage_record_sha256": sha256_file(stage_path),
            "paths": dataset_paths,
            "train_image_identity": source_manifest["splits"]["train"]["images"][
                "identity"
            ],
            "validation_image_identity": source_manifest["splits"][
                "validation"
            ]["images"]["identity"],
        },
        "runtime_inputs": {
            "stage_record_path": str(
                (here / inputs["ptm_staging"]["stage_record_file"]).resolve()
            ),
            "stage_record_file_sha256": sha256_file(
                here / inputs["ptm_staging"]["stage_record_file"]
            ),
            "stage_record_semantic_sha256": stage["stage_record_sha256"],
            "official_ptm_count": len(stage["official_ptms"]),
            "bert_revision": stage["text_encoder"]["revision"],
            "bert_tree_sha256": stage["text_encoder"]["tree_sha256"],
        },
        "metric": {
            "selection_metric": "val_mAP50",
            "policy_id": metric.policy_id,
            "policy_sha256": metric.canonical_sha256,
            "standalone_metric": "test_mAP50",
            "standalone_metric_feeds_selection": False,
        },
        "runtime": {
            "sdk_dir": runtime["sdk_dir"],
            "sdk_revision": runtime["sdk_revision"],
            "skill_dir": runtime["skill_dir"],
            "skill_revision": runtime["skill_revision"],
            "sqsh_path": runtime["sqsh_path"],
            "sqsh_sha256": runtime["sqsh_sha256"],
            "sqsh_size_bytes": runtime["sqsh_size_bytes"],
            "partition": runtime["partition"],
            "account": runtime["account"],
            "base_results_dir": runtime["base_results_dir"],
            "container_mounts": runtime["container_mounts"],
            "tao_version": runtime["tao_version"],
            "offline_environment": copy.deepcopy(
                stage["text_encoder"]["offline_runtime"]
            ),
        },
        "qualification": {
            "jobs": jobs,
            "submit_in_parallel": True,
            "minimum_supported_ptms_for_pilot": 1,
            "failed_ptm_policy": "preserve_and_structurally_exclude",
            "replacement_ptm_submission": False,
            "first_model_execution": "real_one_node_eight_gpu_full_qualification",
            "cpu_model_runs": 0,
            "smoke_or_ministep_runs": 0,
        },
        "pilot_handoff": {
            "automatic": True,
            "manual_confirmation": False,
            "modes": list(MODES),
            "one_algorithm_generated_candidate_per_mode": True,
            "start_only_after_terminal_ptm_qualification": True,
            "remaining_budget_release": (
                "automatic after all three Grounding DINO candidate-0 gates pass"
            ),
        },
        "automatic_trigger": {
            "armed": True,
            "policy": "fail_closed",
            "poll_seconds": 30,
            "predecessor_waits_for_full_budget": False,
            "launch_command": [
                "/localhome/local-rarunachalam/.tao/venvs/"
                "dino-multiobjective-py314/bin/python",
                "-m",
                (
                    "experiments.cross_model_automl_20260729."
                    "grounding_dino_shared_detection.qualification_campaign"
                ),
                "--contract",
                str(DEFAULT_OUTPUT),
                "--launch",
            ],
        },
        "integrity": {
            "inputs_path": str(DEFAULT_INPUTS),
            "inputs_sha256": sha256_file(DEFAULT_INPUTS),
            "runtime_stage_path": str(DEFAULT_STAGE),
            "runtime_stage_sha256": sha256_file(DEFAULT_STAGE),
            "future_contract_generator_path": str(HERE / "future_contract.py"),
            "future_contract_generator_sha256": sha256_file(
                HERE / "future_contract.py"
            ),
            "runtime_input_stage_path": str(HERE / "runtime_input_stage.py"),
            "runtime_input_stage_sha256": sha256_file(
                HERE / "runtime_input_stage.py"
            ),
            "qualification_launcher_path": str(
                HERE / "qualification_campaign.py"
            ),
            "qualification_launcher_sha256": sha256_file(
                HERE / "qualification_campaign.py"
            ),
            "automatic_trigger_path": str(HERE / "automatic_trigger.py"),
            "automatic_trigger_sha256": sha256_file(
                HERE / "automatic_trigger.py"
            ),
        },
        "agent_intervention_flags": {name: False for name in AGENT_FLAGS},
        "execution": {
            "jobs_submitted": 0,
            "scheduler_mutation_performed": False,
            "model_execution_performed": False,
        },
    }
    document["contract_sha256"] = canonical_sha256(document)
    validate_future_contract(document)
    return document


def validate_future_contract(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != 2:
        raise PreparationError("future contract schema differs")
    if document.get("model", {}).get("id") != MODEL_ID:
        raise PreparationError("future contract model differs")
    source = document.get("source", {})
    if (
        not isinstance(source.get("repository"), str)
        or not Path(source["repository"]).is_absolute()
        or re.fullmatch(
            r"[0-9a-f]{40}",
            str(source.get("minimum_ancestor_commit", "")),
        )
        is None
    ):
        raise PreparationError("future contract source identity is invalid")
    dependency = document.get("predecessor_release", {}).get(
        "deformable_detr", {}
    )
    if dependency.get("static_campaign_manifest_sha256") != (
        "d70063f3fc6c4ed7c44d8c7d979e2dc3ffc27f576ddd13cf000648a2c2a26e83"
    ):
        raise PreparationError("future contract references a superseded DDETR")
    if dependency.get("artifact_path") != (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "deformable_detr_automl_synthetic_structured_config_fix_v1/"
        "first_candidate_gate/automatic_release.json"
    ):
        raise PreparationError("future contract DDETR release path differs")
    if dependency.get("runtime_launch_manifest_path") != (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "deformable_detr_automl_synthetic_structured_config_fix_v1/"
        "launch_manifest.json"
    ):
        raise PreparationError("future contract DDETR runtime manifest path differs")
    if dependency.get("static_campaign_manifest_path") != (
        "/localhome/local-rarunachalam/tao-automl/experiments/"
        "cross_model_automl_20260729/deformable_detr_automl_synthetic/"
        "campaign.v1.json"
    ):
        raise PreparationError("future contract DDETR static manifest path differs")
    if dependency.get("source_head") != (
        "8386f524502b1ae7e1a021a37ed8128e7a2fb719"
    ):
        raise PreparationError("future contract DDETR source head differs")
    if "candidates 1-19" not in dependency.get("release_scope", ""):
        raise PreparationError("future contract DDETR release scope is ambiguous")
    jobs = document.get("qualification", {}).get("jobs", [])
    if len(jobs) != 2:
        raise PreparationError("future contract must qualify both official PTMs")
    for job in jobs:
        if job["resources"]["nodes"] != 1 or job["resources"]["gpus_per_node"] != 8:
            raise PreparationError("qualification is not one-node/eight-GPU")
        if job["train"]["spec"]["train"]["is_dry_run"] is not False:
            raise PreparationError("qualification uses dry-run model execution")
        if job["train"]["spec"]["model"]["text_encoder_type"] != (
            job["text_encoder_root"]
        ):
            raise PreparationError("training does not use staged BERT")
        if job["evaluate"]["spec"]["model"]["text_encoder_type"] != (
            job["text_encoder_root"]
        ):
            raise PreparationError("evaluation does not use staged BERT")
    trigger = document.get("automatic_trigger", {})
    if trigger.get("armed") is not True:
        raise PreparationError("future trigger is not armed")
    if trigger.get("predecessor_waits_for_full_budget") is not False:
        raise PreparationError("future trigger waits for the wrong DDETR scope")
    integrity = document.get("integrity")
    if not isinstance(integrity, Mapping):
        raise PreparationError("future contract lacks launcher integrity")
    for field in (
        "inputs_sha256",
        "runtime_stage_sha256",
        "future_contract_generator_sha256",
        "runtime_input_stage_sha256",
        "qualification_launcher_sha256",
        "automatic_trigger_sha256",
    ):
        value = integrity.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise PreparationError(f"future contract {field} is invalid")
    if any(document.get("agent_intervention_flags", {}).values()):
        raise PreparationError("future contract contains agent intervention")
    if document.get("execution") != {
        "jobs_submitted": 0,
        "scheduler_mutation_performed": False,
        "model_execution_performed": False,
    }:
        raise PreparationError("future contract may not claim execution")
    payload = copy.deepcopy(dict(document))
    expected = payload.pop("contract_sha256", None)
    if expected != canonical_sha256(payload):
        raise PreparationError("future contract hash differs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    inputs = read_json(arguments.inputs)
    stage = read_json(arguments.stage)
    document = build_future_contract(
        experiment_dir=HERE,
        inputs=inputs,
        stage=stage,
    )
    if arguments.check_only:
        observed = read_json(arguments.output)
        validate_future_contract(observed)
        if observed != document:
            raise PreparationError("committed future contract is stale")
    else:
        arguments.output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "armed": document["automatic_trigger"]["armed"],
                "contract_sha256": document["contract_sha256"],
                "ddetr_static_manifest_sha256": document["predecessor_release"][
                    "deformable_detr"
                ]["static_campaign_manifest_sha256"],
                "jobs_submitted": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
