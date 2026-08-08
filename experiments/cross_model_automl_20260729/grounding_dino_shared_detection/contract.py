#!/usr/bin/env python3

"""Fail-closed preparation contract for Grounding DINO detection campaigns.

This module performs no model execution and exposes no scheduler submission
path.  It derives category prompts, official PTM inventory, search parameters,
and launch blockers from repository-owned sources.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tao_automl.metric_sanity import (
    UnknownMetricPolicyError,
    default_metric_sanity_registry,
)
from tao_automl.ptm_registry import canonical_sha256, load_ptm_registry


MODEL_ID = "grounding_dino"
MODEL_SKILL = "tao-train-grounding-dino"
MODES = ("accuracy", "latency", "multi_objective")
AGENT_FLAGS = (
    "agent_selected_candidate",
    "agent_injected_candidate",
    "agent_modified_search_space_after_results",
    "agent_changed_seed_after_results",
    "agent_changed_budget_after_results",
    "agent_changed_threshold_after_results",
    "agent_changed_ptm_after_results",
    "agent_overrode_winner",
)
SELECTION_FLAGS = (
    "selector_invoked_on_matched_measurements",
    "selection_time_objectives_replaced",
    "measurements_feed_selection",
    "measurements_feed_reselection",
    "algorithm_selected_candidate_overridden",
)


class PreparationError(ValueError):
    """A source contract is malformed or internally inconsistent."""


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repository: str | Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise PreparationError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PreparationError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise PreparationError(f"{name} must be finite")
    return number


def _finite_schema_projection(value: Any) -> Any:
    """Project generated-schema infinity sentinels into canonical JSON."""
    if isinstance(value, Mapping):
        return {
            str(key): _finite_schema_projection(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_finite_schema_projection(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "+infinity" if value > 0 else "-infinity"
    return value


def derive_category_contract(
    dataset_manifest: Mapping[str, Any],
    annotation_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the category-prompt mapping without inventing any phrase."""
    if dataset_manifest.get("annotation_format") != "COCO object detection":
        raise PreparationError("source dataset must be COCO object detection")
    categories = dataset_manifest.get("categories")
    if not isinstance(categories, list) or not categories:
        raise PreparationError("source dataset must have categories")

    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    normalized = []
    for index, category in enumerate(categories):
        if not isinstance(category, Mapping):
            raise PreparationError(f"categories[{index}] must be an object")
        source_id = category.get("id")
        name = category.get("name")
        if (
            isinstance(source_id, bool)
            or not isinstance(source_id, int)
            or source_id < 0
        ):
            raise PreparationError(f"categories[{index}].id must be >= 0")
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise PreparationError(
                f"categories[{index}].name must be a trimmed non-empty string"
            )
        if source_id in seen_ids:
            raise PreparationError("source category IDs must be unique")
        if name in seen_names:
            raise PreparationError("source category names must be unique")
        seen_ids.add(source_id)
        seen_names.add(name)
        normalized.append((source_id, name, copy.deepcopy(dict(category))))
    normalized.sort(key=lambda item: item[0])

    split_categories = []
    for split in ("train", "validation"):
        record = annotation_audit.get("splits", {}).get(split)
        if not isinstance(record, Mapping):
            raise PreparationError(f"audit split {split!r} is missing")
        split_value = sorted(
            (
                int(item["id"]),
                str(item["name"]),
            )
            for item in record.get("categories", [])
        )
        split_categories.append(split_value)
        if record.get("sha256") != (
            dataset_manifest["splits"][split]["annotation"]["sha256"]
        ):
            raise PreparationError(f"{split} audit hash differs from manifest")
    expected = [(source_id, name) for source_id, name, _ in normalized]
    if any(value != expected for value in split_categories):
        raise PreparationError("train/validation category identities differ")

    prompt_mapping = [
        {
            "source_category_id": source_id,
            "model_category_id": model_id,
            "prompt": name,
            "derivation": "exact_source_coco_category_name",
        }
        for model_id, (source_id, name, _) in enumerate(normalized)
    ]
    return {
        "source_category_count": len(normalized),
        "source_category_ids": [item[0] for item in normalized],
        "model_category_ids": list(range(len(normalized))),
        "prompt_mapping": prompt_mapping,
        "label_map": {
            str(item["model_category_id"]): item["prompt"]
            for item in prompt_mapping
        },
        "prompt_list": [item["prompt"] for item in prompt_mapping],
        "mapping_sha256": canonical_sha256(prompt_mapping),
        "manual_prompt_or_synonym_injection": False,
    }


def grounding_annotation_contract(
    annotation_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify detection-vs-phrase grounding from measured source fields."""
    splits = annotation_audit.get("splits", {})
    images_with_caption = sum(
        int(splits[name].get("images_with_caption", 0))
        for name in ("train", "validation")
    )
    annotations_with_tokens = sum(
        int(splits[name].get("annotations_with_tokens_positive", 0))
        for name in ("train", "validation")
    )
    phrase_ready = images_with_caption > 0 and annotations_with_tokens > 0
    return {
        "category_prompted_detection": {
            "supported_by_source": True,
            "training_contract": "TAO ODVG detection records plus label map",
            "validation_contract": "contiguous-ID COCO bbox evaluation",
            "metric": "val_mAP50",
        },
        "referring_expression_box_grounding": {
            "supported_by_source": phrase_ready,
            "required_image_field": "caption",
            "required_annotation_field": "tokens_positive",
            "images_with_caption": images_with_caption,
            "annotations_with_tokens_positive": annotations_with_tokens,
            "blocker": (
                None
                if phrase_ready
                else "Plain COCO category annotations contain neither "
                "per-image language expressions nor token-positive spans. "
                "Category names may drive category-prompted detection, but "
                "must not be represented as referring expressions."
            ),
        },
    }


def derive_official_ptms() -> tuple[dict[str, Any], ...]:
    """Return every official repository record for the exact model ID."""
    registry = load_ptm_registry()
    model = registry.to_dict()["models"].get(MODEL_ID)
    if not isinstance(model, Mapping):
        raise PreparationError(f"PTM registry has no model {MODEL_ID!r}")
    records = []
    for record in model.get("checkpoints", []):
        if record.get("source", {}).get("official") is not True:
            continue
        value = copy.deepcopy(record)
        value["registry_record_sha256"] = canonical_sha256(record)
        value["registry_sha256"] = registry.document_sha256
        records.append(value)
    records.sort(key=lambda item: item["id"])
    if not records:
        raise PreparationError("official Grounding DINO PTM inventory is empty")
    if len({item["id"] for item in records}) != len(records):
        raise PreparationError("official Grounding DINO PTM IDs are not unique")
    return tuple(records)


def _schema_node(schema: Mapping[str, Any], dotted_path: str) -> Mapping[str, Any]:
    node: Any = schema
    for component in dotted_path.split("."):
        node = node.get("properties", {}).get(component)
        if not isinstance(node, Mapping):
            raise PreparationError(
                f"schema parameter {dotted_path!r} cannot be resolved"
            )
    return node


def derive_schema_contract(skill_dir: str | Path) -> dict[str, Any]:
    """Derive model ID and all default search parameters from packaged files."""
    skill_path = Path(skill_dir)
    info = read_json(skill_path / "schemas" / "manifest.json")
    if "train" not in info.get("actions", {}):
        raise PreparationError("Grounding DINO train schema is not packaged")
    schema_path = skill_path / "schemas" / "train.schema.json"
    schema = read_json(schema_path)

    import yaml

    skill_info = yaml.safe_load(
        (skill_path / "references" / "skill_info.yaml").read_text(
            encoding="utf-8"
        )
    )
    if skill_info.get("network_arch") != MODEL_ID:
        raise PreparationError("Grounding DINO skill resolved the wrong model ID")
    if skill_info.get("automl_enabled") is not True:
        raise PreparationError("Grounding DINO skill is not AutoML enabled")

    names = schema.get("automl_default_parameters")
    if not isinstance(names, list) or not names:
        raise PreparationError("train schema has no default AutoML parameters")
    parameters = {}
    for name in names:
        if not isinstance(name, str):
            raise PreparationError("schema parameter names must be strings")
        node = _finite_schema_projection(
            copy.deepcopy(dict(_schema_node(schema, name)))
        )
        if node.get("automl_enabled") is not True:
            raise PreparationError(
                f"default parameter {name!r} is not AutoML enabled"
            )
        parameters[name] = node
    return {
        "model_id": skill_info["network_arch"],
        "model_skill": skill_path.name,
        "automl_enabled": True,
        "train_action_mode": skill_info["actions"]["train"]["mode"],
        "train_command": skill_info["actions"]["train"]["command"],
        "schema_path": str(schema_path),
        "schema_sha256": sha256_file(schema_path),
        "parameter_names": sorted(parameters),
        "parameters": {
            name: parameters[name] for name in sorted(parameters)
        },
        "parameters_sha256": canonical_sha256(parameters),
        "source": "packaged_train_schema_automl_default_parameters",
    }


def metric_contract() -> dict[str, Any]:
    """Report production policy support without silently inventing a policy."""
    registry = default_metric_sanity_registry()
    output: dict[str, Any] = {
        "registry_sha256": registry.canonical_sha256,
    }
    for metric in ("val_mAP50", "val_Pr@0.5"):
        try:
            policy = registry.resolve(MODEL_ID, metric)
        except UnknownMetricPolicyError as exc:
            output[metric] = {
                "availability": "unregistered",
                "reason": str(exc),
            }
        else:
            output[metric] = {
                "availability": policy.availability,
                "policy_id": policy.policy_id,
                "task": policy.task,
                "reason": policy.availability_reason,
            }
    return output


def _objective(mode: str, retained_fraction: float) -> dict[str, Any]:
    accuracy = {
        "name": "val_mAP50",
        "role": "accuracy",
        "direction": "maximize",
    }
    latency = {
        "name": "median_latency_ms",
        "role": "latency",
        "direction": "minimize",
    }
    if mode == "accuracy":
        return {
            "mode": mode,
            "acquisition": "expected_improvement",
            "metrics": [accuracy],
            "quality_constraint": None,
            "selection_policy": "highest_valid_accuracy",
        }
    if mode == "latency":
        return {
            "mode": mode,
            "acquisition": "constrained_expected_improvement",
            "metrics": [accuracy, latency],
            "quality_constraint": {
                "type": "relative_retention",
                "retained_fraction": retained_fraction,
                "reference": "best_observed_within_job",
                "reference_updates": "monotonic",
                "terminal_reference": "terminal_archive_accuracy_winner",
            },
            "selection_policy": "equivalent_fastest_accuracy_tiebreak",
        }
    if mode == "multi_objective":
        return {
            "mode": mode,
            "acquisition": "parego_expected_improvement",
            "metrics": [accuracy, latency],
            "quality_constraint": None,
            "selection_policy": "normalized_augmented_chebyshev",
        }
    raise PreparationError(f"unsupported objective mode: {mode}")


def build_preparation(
    *,
    experiment_dir: str | Path,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a canonical, non-launching preparation record."""
    here = Path(experiment_dir).resolve()
    dataset_config = inputs["dataset"]
    dataset_manifest_path = (here / dataset_config["manifest_file"]).resolve()
    audit_path = (here / dataset_config["audit_file"]).resolve()
    dataset_manifest = read_json(dataset_manifest_path)
    annotation_audit = read_json(audit_path)
    categories = derive_category_contract(dataset_manifest, annotation_audit)
    grounding = grounding_annotation_contract(annotation_audit)

    runtime = copy.deepcopy(inputs["runtime"])
    schema = derive_schema_contract(runtime["skill_dir"])
    ptms = derive_official_ptms()
    metrics = metric_contract()
    retained_fraction = _finite_number(
        inputs["search"]["latency_accuracy_retention"],
        "latency_accuracy_retention",
    )
    if not 0.0 < retained_fraction <= 1.0:
        raise PreparationError("latency_accuracy_retention must be in (0, 1]")

    converted_root = Path(dataset_config["converted_lustre_root"])
    converted_paths = {
        "train_image_dir": dataset_manifest["splits"]["train"]["images"]["path"],
        "train_odvg_jsonl": str(
            converted_root / "train" / "annotations_odvg.jsonl"
        ),
        "train_label_map": str(
            converted_root / "train" / "annotations_odvg_labelmap.json"
        ),
        "validation_image_dir": dataset_manifest["splits"]["validation"][
            "images"
        ]["path"],
        "validation_coco_contiguous": str(
            converted_root / "validation" / "annotations_remapped.json"
        ),
    }

    official_ids = [item["id"] for item in ptms]
    modes = []
    for mode in MODES:
        objective = _objective(mode, retained_fraction)
        modes.append(
            {
                "mode": mode,
                "job_id": f"{inputs['campaign_id']}-{mode}",
                "search_seed": inputs["search"]["search_seed"],
                "observation_namespace": (
                    f"{inputs['campaign_id']}-{mode}-observations"
                ),
                "observation_sharing": False,
                "initial_observation_ids": [],
                "ptm_policy": "all_qualified_explicit",
                "candidate_ptm_ids": official_ids,
                "qualified_ptm_ids_resolved_at_gate": True,
                "objective": objective,
                "objective_sha256": canonical_sha256(objective),
            }
        )

    qualification_jobs = []
    for record in ptms:
        source = record["source"]
        qualification_jobs.append(
            {
                "ptm_id": record["id"],
                "source_identity": source["immutable_identity"],
                "source_member": source["member"],
                "registry_status_before_qualification": record["status"],
                "checkpoint_target": record["checkpoint_target"],
                "default_spec_overrides": record["default_spec_overrides"],
                "resource": {
                    "platform": "slurm",
                    "nodes": 1,
                    "gpus_per_node": 8,
                    "tasks_per_node": 1,
                    "distributed_workers_per_node": 8,
                },
                "workflow": [
                    "full_10_epoch_train_with_validation_each_epoch",
                    "standalone_evaluate_from_exact_terminal_checkpoint",
                    "artifact_and_metric_contract_validation",
                    "evidence_based_registry_promotion_or_structured_exclusion",
                ],
            }
        )

    blockers = []
    if grounding["referring_expression_box_grounding"]["supported_by_source"] is not True:
        blockers.append(
            {
                "code": "referring_expression_annotation_contract_missing",
                "scope": "grounding_claim",
                "reason": grounding["referring_expression_box_grounding"][
                    "blocker"
                ],
            }
        )
    if metrics["val_mAP50"]["availability"] != "supported":
        blockers.append(
            {
                "code": "category_detection_metric_policy_not_supported",
                "scope": "objective_aware_campaign",
                "reason": metrics["val_mAP50"]["reason"],
            }
        )
    unverified = [
        record["id"] for record in ptms if record.get("status") != "supported"
    ]
    if unverified:
        blockers.append(
            {
                "code": "official_ptms_not_production_qualified",
                "scope": "objective_aware_campaign",
                "ptm_ids": unverified,
                "reason": (
                    "Only evidence-backed supported PTMs may enter production "
                    "hierarchical PTM search."
                ),
            }
        )
    blockers.append(
        {
            "code": "converted_dataset_artifacts_not_sealed",
            "scope": "direct_qualification",
                "required_paths": [
                    converted_paths["train_odvg_jsonl"],
                    converted_paths["train_label_map"],
                    converted_paths["validation_coco_contiguous"],
                ],
            "reason": (
                "The repository-owned preparation records conversion commands "
                "but does not assert that ungenerated files exist."
            ),
        }
    )

    source_repo = Path(inputs["source"]["repository"])
    source = {
        "repository": str(source_repo),
        "commit": _git(source_repo, "rev-parse", "HEAD"),
        "dirty": bool(_git(source_repo, "status", "--porcelain")),
        "launch_head_policy": "clean_descendant",
    }
    dataservices = copy.deepcopy(inputs["tao_dataservices"])
    dataservices["clean"] = not bool(
        _git(dataservices["repository"], "status", "--porcelain")
    )
    dataservices["actual_revision"] = _git(
        dataservices["repository"], "rev-parse", "HEAD"
    )

    preparation = {
        "schema_version": 1,
        "campaign_id": inputs["campaign_id"],
        "model": {
            "id": schema["model_id"],
            "skill": schema["model_skill"],
            "task": "category_prompted_open_vocabulary_detection",
            "forbidden_claim": "referring_expression_box_grounding",
        },
        "source": source,
        "runtime": {
            **runtime,
            "platform": "slurm",
            "nodes": 1,
            "gpus_per_node": 8,
            "tasks_per_node": 1,
            "distributed_workers_per_node": 8,
            "slurm_use_sqsh_conversion": False,
            "sqsh_direct_path": True,
            "container_mounts": "/lustre",
            "time_hours": 4,
            "timeout_hours": 3.8,
        },
        "dataset": {
            "source_manifest": str(dataset_manifest_path),
            "source_manifest_sha256": sha256_file(dataset_manifest_path),
            "source_annotation_audit": str(audit_path),
            "source_annotation_audit_sha256": sha256_file(audit_path),
            "source_task": dataset_manifest["task"],
            "category_contract": categories,
            "grounding_contract": grounding,
            "conversion": {
                "implementation": "nvidia-tao-dataservices",
                "source": dataservices,
                "use_all_categories": False,
                "train_command": [
                    "python",
                    "nvidia_tao_ds/annotations/scripts/convert.py",
                    (
                        "coco.ann_file="
                        + dataset_manifest["splits"]["train"]["annotation"][
                            "path"
                        ]
                    ),
                    "coco.use_all_categories=false",
                    "data.input_format=COCO",
                    "data.output_format=ODVG",
                    f"results_dir={converted_root / 'train'}",
                ],
                "validation_command": [
                    "python",
                    "nvidia_tao_ds/annotations/scripts/convert.py",
                    (
                        "coco.ann_file="
                        + dataset_manifest["splits"]["validation"]["annotation"][
                            "path"
                        ]
                    ),
                    "coco.use_all_categories=false",
                    "data.input_format=COCO",
                    "data.output_format=COCO",
                    f"results_dir={converted_root / 'validation'}",
                ],
                "empty_training_image_policy": (
                    "official_converter_excludes_and_audit_preserves_ids"
                ),
                "expected_excluded_empty_train_images": annotation_audit[
                    "splits"
                ]["train"]["empty_annotation_images"],
                "paths": converted_paths,
            },
        },
        "metric_contract": metrics,
        "official_ptm_inventory": {
            "derivation": (
                "all source.official=true records under "
                "ptm_registry.models.grounding_dino"
            ),
            "count": len(ptms),
            "records": list(ptms),
            "candidate_ids": official_ids,
            "manual_ptm_selection": False,
        },
        "direct_qualification": {
            "launches_prepared": len(qualification_jobs),
            "jobs": qualification_jobs,
            "training_epochs": inputs["qualification"]["training_epochs"],
            "validation_interval": inputs["qualification"][
                "validation_interval"
            ],
            "standalone_evaluation": True,
            "cpu_model_runs": False,
            "smoke_or_ministep_runs": False,
        },
        "automl": {
            "independent_mode_jobs": True,
            "shared_observations": False,
            "algorithm": inputs["search"]["algorithm"],
            "candidate_budget_per_mode": inputs["search"][
                "candidate_budget_per_mode"
            ],
            "training_epochs": inputs["search"]["training_epochs"],
            "search_schema": schema,
            "ptm_representation": "hierarchical_nonordinal_arms",
            "latency_protocol": {
                "warmup_iterations": 50,
                "timed_iterations": 100,
                "repeated_rounds": 5,
                "replicas": 8,
                "raw_samples_per_candidate": 4000,
                "batch_size_per_replica": 1,
                "precision": inputs["qualification"]["precision"],
                "practical_tolerance_ms": inputs["search"][
                    "latency_practical_tolerance_ms"
                ],
                "synchronization": (
                    "accelerator_sync_before_and_after_each_sample"
                ),
            },
            "modes": modes,
        },
        "automatic_gate": {
            "launch_authorized": False,
            "policy": "fail_closed",
            "blockers": blockers,
            "release_requirements": [
                "converted_dataset_integrity_sealed",
                "category_detection_metric_policy_supported",
                "at_least_one_official_ptm_evidence_qualified",
                "production_wheel_includes_required_runtime",
                "source_sdk_and_skill_revisions_clean_and_pinned",
                "one_candidate_per_mode_gate_then_automatic_release",
            ],
        },
        "agent_intervention_flags": {name: False for name in AGENT_FLAGS},
        "selection_isolation_flags": {
            name: False for name in SELECTION_FLAGS
        },
        "execution": {
            "jobs_submitted": 0,
            "scheduler_mutation_performed": False,
            "model_execution_performed": False,
        },
    }
    preparation["preparation_sha256"] = canonical_sha256(preparation)
    return preparation


def validate_preparation(document: Mapping[str, Any]) -> None:
    """Recheck the strongest safety properties of a prepared document."""
    if document.get("model", {}).get("id") != MODEL_ID:
        raise PreparationError("prepared model identifier changed")
    if tuple(
        mode.get("mode")
        for mode in document.get("automl", {}).get("modes", [])
    ) != MODES:
        raise PreparationError("prepared modes are not exact and ordered")
    resources = document.get("runtime", {})
    if resources.get("nodes") != 1 or resources.get("gpus_per_node") != 8:
        raise PreparationError("campaign must use one eight-GPU node")
    if resources.get("sqsh_direct_path") is not True:
        raise PreparationError("campaign must use the pinned SQSH directly")
    if document.get("automatic_gate", {}).get("launch_authorized") is not False:
        raise PreparationError("blocked preparation cannot authorize launch")
    if any(document.get("agent_intervention_flags", {}).values()):
        raise PreparationError("agent intervention flags must remain false")
    if any(document.get("selection_isolation_flags", {}).values()):
        raise PreparationError("selection-isolation flags must remain false")
    if document.get("execution") != {
        "jobs_submitted": 0,
        "scheduler_mutation_performed": False,
        "model_execution_performed": False,
    }:
        raise PreparationError("preparation may not claim model execution")

    expected = copy.deepcopy(dict(document))
    observed = expected.pop("preparation_sha256", None)
    if observed != canonical_sha256(expected):
        raise PreparationError("preparation_sha256 does not match content")
