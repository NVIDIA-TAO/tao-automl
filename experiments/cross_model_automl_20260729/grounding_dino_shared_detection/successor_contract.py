#!/usr/bin/env python3

"""Seal the blocked Grounding DINO qualification and AutoML successor.

This module is intentionally non-launching.  It resolves complete train/eval
spec contracts from the Grounding DINO skill and repository PTM registry, then
evaluates the two predecessor first-candidate gates.  Missing evidence leaves
the successor blocked; there is no fallback workload.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

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
except ImportError:  # pragma: no cover - direct script execution
    from contract import (
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


HERE = Path(__file__).resolve().parent
DEFAULT_INPUTS = HERE / "campaign.inputs.v2.json"
DEFAULT_OUTPUT = HERE / "successor.contract.v1.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _deep_merge(target: dict[str, Any], update: Mapping[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _load_skill_contract(skill_dir: Path) -> dict[str, Any]:
    info = yaml.safe_load(
        (skill_dir / "references" / "skill_info.yaml").read_text(
            encoding="utf-8"
        )
    )
    if info.get("network_arch") != MODEL_ID:
        raise PreparationError("Grounding DINO skill model ID differs")
    train_action = info.get("actions", {}).get("train")
    evaluate_action = info.get("actions", {}).get("evaluate")
    if (
        not isinstance(train_action, Mapping)
        or train_action.get("mode") != "config"
        or not isinstance(evaluate_action, Mapping)
        or evaluate_action.get("mode") != "config"
    ):
        raise PreparationError("Grounding DINO train/evaluate must use config mode")
    train_template = skill_dir / "references" / "spec_template_train.yaml"
    evaluate_template = skill_dir / "references" / "spec_template_evaluate.yaml"
    return {
        "skill_info": info,
        "skill_info_path": skill_dir / "references" / "skill_info.yaml",
        "train_template": yaml.safe_load(train_template.read_text(encoding="utf-8")),
        "train_template_path": train_template,
        "evaluate_template": yaml.safe_load(
            evaluate_template.read_text(encoding="utf-8")
        ),
        "evaluate_template_path": evaluate_template,
    }


def _ptm_lustre_path(root: Path, record: Mapping[str, Any]) -> str:
    return str(root / record["id"] / record["source"]["member"])


def _train_spec(
    *,
    template: Mapping[str, Any],
    record: Mapping[str, Any],
    dataset_paths: Mapping[str, str],
    ptm_path: str,
    qualification: Mapping[str, Any],
) -> dict[str, Any]:
    spec = copy.deepcopy(dict(template))
    _deep_merge(spec, record["default_spec_overrides"])
    spec["wandb"]["enable"] = False
    spec["dataset"]["train_data_sources"] = [
        {
            "image_dir": dataset_paths["train_image_dir"],
            "json_file": dataset_paths["train_odvg_jsonl"],
            "label_map": dataset_paths["train_label_map"],
        }
    ]
    spec["dataset"]["val_data_sources"] = {
        "image_dir": dataset_paths["validation_image_dir"],
        "json_file": dataset_paths["validation_coco_contiguous"],
    }
    spec["dataset"]["eval_class_ids"] = [0, 1, 2, 3]
    spec["dataset"]["batch_size"] = qualification["train_batch_size_per_gpu"]
    spec["train"].update(
        {
            "num_gpus": 8,
            "gpu_ids": list(range(8)),
            "num_nodes": 1,
            "seed": qualification["seed"],
            "num_epochs": qualification["training_epochs"],
            "checkpoint_interval": qualification["checkpoint_interval"],
            "validation_interval": qualification["validation_interval"],
            "pretrained_model_path": ptm_path,
            "precision": qualification["precision"],
            "distributed_strategy": "ddp",
            "is_dry_run": False,
        }
    )
    return spec


def _evaluate_spec(
    *,
    template: Mapping[str, Any],
    record: Mapping[str, Any],
    dataset_paths: Mapping[str, str],
    qualification: Mapping[str, Any],
) -> dict[str, Any]:
    spec = copy.deepcopy(dict(template))
    _deep_merge(spec, record["default_spec_overrides"])
    spec["wandb"]["enable"] = False
    spec["dataset"]["test_data_sources"] = {
        "image_dir": dataset_paths["validation_image_dir"],
        "json_file": dataset_paths["validation_coco_contiguous"],
    }
    spec["dataset"]["eval_class_ids"] = [0, 1, 2, 3]
    spec["evaluate"].update(
        {
            "num_gpus": 8,
            "gpu_ids": list(range(8)),
            "num_nodes": 1,
            "batch_size": qualification["evaluation_batch_size_per_gpu"],
            "checkpoint": "resolve_from_exact_training_terminal_evidence",
        }
    )
    return spec


def _canonical_record_valid(
    record: Mapping[str, Any],
    *,
    hash_field: str,
) -> bool:
    if not isinstance(record, Mapping):
        return False
    payload = copy.deepcopy(dict(record))
    expected = payload.pop(hash_field, None)
    return (
        isinstance(expected, str)
        and _SHA256_RE.fullmatch(expected) is not None
        and expected == canonical_sha256(payload)
    )


def _evaluate_rtdetr_gate(configuration: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(configuration["artifact_path"])
    result: dict[str, Any] = {
        "model": "rtdetr",
        "artifact_path": str(path),
        "expected_contract_sha256": configuration["contract_sha256"],
        "passed": False,
        "blockers": [],
    }
    if not path.is_file():
        result["blockers"].append("required release artifact is absent")
        return result
    record = read_json(path)
    required = configuration["required_fields"]
    if record.get("contract_sha256") != configuration["contract_sha256"]:
        result["blockers"].append("contract_sha256 differs")
    if record.get("release_remaining_budget") is not required[
        "release_remaining_budget"
    ]:
        result["blockers"].append("release_remaining_budget differs")
    if record.get("modes") != required["modes"]:
        result["blockers"].append("modes differ")
    first = record.get("first_candidates")
    if not isinstance(first, Mapping):
        result["blockers"].append("first_candidates is absent")
    else:
        required_candidate = configuration[
            "required_first_candidate_fields"
        ]
        for mode in MODES:
            candidate = first.get(mode, {})
            for field, expected in required_candidate.items():
                if candidate.get(field) is not expected:
                    result["blockers"].append(
                        f"{mode} first candidate {field} differs"
                    )
    result["artifact_sha256"] = sha256_file(path)
    result["passed"] = not result["blockers"]
    return result


def _evaluate_ddetr_gate(configuration: Mapping[str, Any]) -> dict[str, Any]:
    path_value = configuration.get("artifact_path")
    result: dict[str, Any] = {
        "model": "deformable_detr",
        "artifact_path": path_value,
        "expected_manifest_sha256": configuration.get("manifest_sha256"),
        "passed": False,
        "blockers": [],
    }
    if not path_value or not configuration.get("manifest_sha256"):
        result["blockers"].append(
            "corrected fresh runtime path and manifest hash are not yet bound"
        )
        return result
    path = Path(path_value)
    if not path.is_file():
        result["blockers"].append("required automatic release artifact is absent")
        return result
    release = read_json(path)
    required = configuration["required_release_fields"]
    for field, expected in required.items():
        if release.get(field) != expected:
            result["blockers"].append(f"release field {field} differs")
    if release.get("manifest_sha256") != configuration["manifest_sha256"]:
        result["blockers"].append("release manifest_sha256 differs")
    if not _canonical_record_valid(release, hash_field="gate_record_sha256"):
        result["blockers"].append("release gate_record_sha256 is invalid")

    root = path.parents[1]
    for mode in MODES:
        gate_path = root / "first_candidate_gate" / f"{mode}.json"
        if not gate_path.is_file():
            result["blockers"].append(f"{mode} gate record is absent")
            continue
        gate = read_json(gate_path)
        if gate.get("manifest_sha256") != configuration["manifest_sha256"]:
            result["blockers"].append(f"{mode} gate manifest differs")
        if gate.get("candidate_index") != 0:
            result["blockers"].append(f"{mode} candidate index differs")
        if gate.get("candidate_id") != configuration[
            "required_candidate_id_template"
        ].format(mode=mode):
            result["blockers"].append(f"{mode} candidate ID differs")
        if gate.get("passed") is not True:
            result["blockers"].append(f"{mode} candidate gate did not pass")
        if (
            not isinstance(gate.get("evidence_sha256"), str)
            or _SHA256_RE.fullmatch(gate["evidence_sha256"]) is None
        ):
            result["blockers"].append(f"{mode} evidence hash is invalid")
        if gate.get("reason") != configuration["required_reason"]:
            result["blockers"].append(f"{mode} gate reason differs")
    result["artifact_sha256"] = sha256_file(path)
    result["passed"] = not result["blockers"]
    return result


def build_successor_contract(
    *,
    experiment_dir: str | Path,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    here = Path(experiment_dir).resolve()
    dataset = inputs["dataset"]
    conversion_path = (here / dataset["conversion_manifest_file"]).resolve()
    stage_path = (here / dataset["stage_record_file"]).resolve()
    conversion = read_json(conversion_path)
    stage = read_json(stage_path)
    validate_conversion_manifest(conversion)
    validate_stage_record(stage)

    metric = default_metric_sanity_registry().resolve(MODEL_ID, "val_mAP50")
    if metric.availability != "supported" or metric.task != "object_detection":
        raise PreparationError("Grounding DINO val_mAP50 policy is not supported")

    runtime = inputs["runtime"]
    skill = _load_skill_contract(Path(runtime["skill_dir"]))
    ptms = derive_official_ptms()
    ptm_root = Path(inputs["ptm_staging"]["checkpoint_root"])
    canonical_outputs = conversion["canonical_outputs"]
    source_manifest = read_json(
        (here / dataset["manifest_file"]).resolve()
    )
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
    qualification_jobs = []
    for record in ptms:
        checkpoint_path = _ptm_lustre_path(ptm_root, record)
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
        qualification_jobs.append(
            {
                "ptm_id": record["id"],
                "registry_status_before_qualification": record["status"],
                "source": record["source"],
                "expected_checkpoint_sha256": record.get("sha256"),
                "expected_checkpoint_size_bytes": record[
                    "expected_size_bytes"
                ],
                "staged_checkpoint_path": checkpoint_path,
                "checkpoint_staged_and_verified": False,
                "production_preflight": {
                    "api": "PTMCheckpointPreflight.run_qualification",
                    "model": MODEL_ID,
                    "task": "grounded_object_detection",
                    "tao_version": runtime["tao_version"],
                    "validation_statuses": ["unverified"],
                    "checkpoint_ids": [record["id"]],
                    "execution_location": "inside allocated eight-GPU job",
                    "cpu_load_smoke": False,
                },
                "train": {
                    "command": skill["skill_info"]["actions"]["train"]["command"],
                    "spec": train_spec,
                    "spec_sha256": canonical_sha256(train_spec),
                    "expected_validation_metric": "val_mAP50",
                    "required_status_message": "Eval metrics generated.",
                },
                "evaluate": {
                    "command": skill["skill_info"]["actions"]["evaluate"][
                        "command"
                    ],
                    "spec": evaluate_spec,
                    "spec_sha256": canonical_sha256(evaluate_spec),
                    "checkpoint_resolution": (
                        "exact terminal training checkpoint from bound status "
                        "and file hash evidence; never latest-file guessing"
                    ),
                    "expected_metric": "test_mAP50",
                    "required_metric_status_message": "Test metrics generated.",
                    "required_terminal_status_message": (
                        "Evaluate finished successfully."
                    ),
                    "test_metric_may_not_feed_automl_selection": True,
                },
                "resources": {
                    "platform": "slurm",
                    "nodes": 1,
                    "tasks_per_node": 1,
                    "gpus_per_node": 8,
                    "distributed_workers_per_node": 8,
                    "partition": runtime["partition"],
                    "account": runtime["account"],
                    "time_hours": 4,
                    "timeout_hours": 3.8,
                    "sqsh_path": runtime["sqsh_path"],
                    "sqsh_sha256": runtime["sqsh_sha256"],
                    "use_sqsh_conversion": False,
                },
            }
        )

    predecessor = {
        "rtdetr": _evaluate_rtdetr_gate(
            inputs["predecessor_gates"]["rtdetr"]
        ),
        "deformable_detr": _evaluate_ddetr_gate(
            inputs["predecessor_gates"]["deformable_detr"]
        ),
    }
    blockers = []
    for model, result in predecessor.items():
        if result["passed"] is not True:
            blockers.append(
                {
                    "code": f"{model}_first_candidate_gate_not_passed",
                    "details": result["blockers"],
                }
            )
    blockers.extend(
        [
            {
                "code": "official_ptm_checkpoints_not_staged",
                "ptm_ids": [record["id"] for record in ptms],
            },
            {
                "code": "bert_base_uncased_cache_not_sealed",
                "path": inputs["ptm_staging"]["bert_cache_root"],
            },
            {
                "code": "official_ptms_not_full_gpu_qualified",
                "ptm_ids": [record["id"] for record in ptms],
            },
        ]
    )

    mode_jobs = []
    for mode in MODES:
        acquisition = {
            "accuracy": "expected_improvement",
            "latency": "constrained_expected_improvement",
            "multi_objective": "parego_expected_improvement",
        }[mode]
        mode_jobs.append(
            {
                "mode": mode,
                "job_id": f"{inputs['campaign_id']}-{mode}",
                "independent_observation_namespace": (
                    f"{inputs['campaign_id']}-{mode}-observations"
                ),
                "observation_sharing": False,
                "algorithm": inputs["search"]["algorithm"],
                "search_seed": inputs["search"]["search_seed"],
                "acquisition": acquisition,
                "candidate_generation": "algorithm_only",
                "ptm_dimension": "qualified_official_ptm_hierarchical_arm",
                "first_candidate_gate": 1,
                "remaining_candidate_budget_after_gate": (
                    inputs["search"]["candidate_budget_per_mode"]
                    - inputs["search"]["first_candidate_gate_per_mode"]
                ),
                "objective": {
                    "accuracy": {
                        "metric": "val_mAP50",
                        "direction": "maximize",
                    },
                    "latency": (
                        None
                        if mode == "accuracy"
                        else {
                            "metric": "median_latency_ms",
                            "direction": "minimize",
                        }
                    ),
                    "latency_accuracy_retention": (
                        {
                            "type": "relative",
                            "retained_fraction": inputs["search"][
                                "latency_accuracy_retention"
                            ],
                            "reference": "best_observed_within_job",
                            "reference_updates": "monotonic",
                        }
                        if mode == "latency"
                        else None
                    ),
                    "multi_objective_min_accuracy": None,
                },
            }
        )

    document = {
        "schema_version": 1,
        "campaign_id": inputs["campaign_id"],
        "model": {
            "id": MODEL_ID,
            "task": "category_prompted_open_vocabulary_detection",
            "forbidden_claim": "referring_expression_box_grounding",
        },
        "dataset": {
            "conversion_manifest_path": str(conversion_path),
            "conversion_manifest_file_sha256": sha256_file(conversion_path),
            "conversion_manifest_semantic_sha256": conversion[
                "manifest_sha256"
            ],
            "stage_record_path": str(stage_path),
            "stage_record_file_sha256": sha256_file(stage_path),
            "stage_record_semantic_sha256": stage["stage_record_sha256"],
            "paths": dataset_paths,
            "annotation_lossless": True,
            "excluded_empty_train_images": conversion["semantic_validation"][
                "train"
            ]["excluded_empty_image_count"],
        },
        "metric_contract": {
            "policy_id": metric.policy_id,
            "policy_sha256": metric.canonical_sha256,
            "registry_sha256": default_metric_sanity_registry().canonical_sha256,
            "training_selection_metric": "val_mAP50",
            "standalone_qualification_metric": "test_mAP50",
            "standalone_test_metric_feeds_selection": False,
        },
        "skill_contract": {
            "skill_revision": runtime["skill_revision"],
            "skill_info_path": str(skill["skill_info_path"]),
            "skill_info_sha256": sha256_file(skill["skill_info_path"]),
            "train_template_path": str(skill["train_template_path"]),
            "train_template_sha256": sha256_file(skill["train_template_path"]),
            "evaluate_template_path": str(skill["evaluate_template_path"]),
            "evaluate_template_sha256": sha256_file(
                skill["evaluate_template_path"]
            ),
        },
        "predecessor_first_candidate_gates": predecessor,
        "ptm_inventory": {
            "derivation": "all repository records with source.official=true",
            "records": list(ptms),
            "manual_ptm_selection": False,
            "qualification_jobs": qualification_jobs,
            "jobs_may_submit_in_parallel_after_predecessor_release": True,
        },
        "automl_successor": {
            "mode_jobs": mode_jobs,
            "mode_pilots_may_submit_in_parallel_after_ptm_qualification": True,
            "automatic_remaining_budget_release_requires_all_mode_pilots": True,
            "candidate_budget_per_mode": inputs["search"][
                "candidate_budget_per_mode"
            ],
            "training_epochs": inputs["search"]["training_epochs"],
            "manual_candidate_injection": False,
            "manual_ptm_selection": False,
        },
        "automatic_trigger": {
            "policy": "fail_closed",
            "launch_authorized": False,
            "blockers": blockers,
            "trigger_sequence": [
                "validate_rtdetr_and_deformable_detr_first_candidate_releases",
                "stage_and_hash_all_official_ptms_and_bert_cache_before_gpu",
                "submit_official_ptm_qualification_jobs_in_parallel",
                "qualify_or_structurally_exclude_each_ptm_from_full_evidence",
                "submit_one_algorithm_generated_candidate_per_mode_in_parallel",
                "validate_all_three_first_candidate_gates",
                "automatically_release_remaining_frozen_budget",
            ],
        },
        "agent_intervention_flags": {name: False for name in AGENT_FLAGS},
        "execution": {
            "jobs_submitted": 0,
            "scheduler_mutation_performed": False,
            "model_execution_performed": False,
        },
    }
    document["contract_sha256"] = canonical_sha256(document)
    return document


def validate_successor_contract(document: Mapping[str, Any]) -> None:
    if document.get("model", {}).get("id") != MODEL_ID:
        raise PreparationError("successor model ID differs")
    jobs = document.get("ptm_inventory", {}).get("qualification_jobs", [])
    if len(jobs) != 2:
        raise PreparationError("successor must qualify both official PTMs")
    for job in jobs:
        resources = job.get("resources", {})
        if resources.get("nodes") != 1 or resources.get("gpus_per_node") != 8:
            raise PreparationError("qualification must use one eight-GPU node")
        if resources.get("use_sqsh_conversion") is not False:
            raise PreparationError("qualification must use pinned SQSH directly")
        if job["train"]["spec"]["train"]["is_dry_run"] is not False:
            raise PreparationError("qualification may not use dry-run mode")
        if job["evaluate"]["test_metric_may_not_feed_automl_selection"] is not True:
            raise PreparationError("test metrics must remain selection-isolated")
    modes = document.get("automl_successor", {}).get("mode_jobs", [])
    if tuple(item.get("mode") for item in modes) != MODES:
        raise PreparationError("successor mode ordering differs")
    if any(
        item["candidate_generation"] != "algorithm_only"
        for item in modes
    ):
        raise PreparationError("successor permits non-algorithmic candidates")
    if document.get("automatic_trigger", {}).get("launch_authorized") is not False:
        raise PreparationError("prepared successor must remain blocked")
    if any(document.get("agent_intervention_flags", {}).values()):
        raise PreparationError("agent intervention flags must remain false")
    if document.get("execution") != {
        "jobs_submitted": 0,
        "scheduler_mutation_performed": False,
        "model_execution_performed": False,
    }:
        raise PreparationError("successor contract may not claim execution")
    payload = copy.deepcopy(dict(document))
    expected = payload.pop("contract_sha256", None)
    if expected != canonical_sha256(payload):
        raise PreparationError("contract_sha256 does not match content")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    document = build_successor_contract(
        experiment_dir=HERE,
        inputs=read_json(arguments.inputs),
    )
    validate_successor_contract(document)
    if not arguments.check_only:
        arguments.output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "blocker_codes": [
                    item["code"]
                    for item in document["automatic_trigger"]["blockers"]
                ],
                "contract_sha256": document["contract_sha256"],
                "jobs_submitted": 0,
                "launch_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
