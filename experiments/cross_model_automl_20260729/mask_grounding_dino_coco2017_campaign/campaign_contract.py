#!/usr/bin/env python3

"""Frozen Mask Grounding DINO/COCO 2017 three-mode campaign policy.

This is an experiment contract, not a launcher.  It binds the complete COCO
2017 instance-mask dataset to Mask Grounding DINO's ``data_type: OD`` path.
The text prompts are the 80 COCO category names; they are not referring
expressions.  Accuracy is the model's OD segmentation AP50-95 aggregate, never
``val_loss`` and never the unrelated VG ``overall_IoU`` metric.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tao_automl.ptm_registry import canonical_sha256, load_ptm_registry


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

# PTM identity is a hierarchical, non-ordinal outer arm.  ``num_select`` is
# the only packaged AutoML-enabled Mask Grounding DINO parameter in this
# contract that changes inference/post-processing work.  Optimizer parameters
# recover predictive quality within each checkpoint arm.  Encoder and decoder
# depth are deliberately absent: the mask head requires exactly six outputs.
SEARCH_PARAMETERS = (
    "model.num_select",
    "train.optim.lr",
    "train.optim.lr_backbone",
    "train.optim.weight_decay",
)
SEARCH_SPACE = {
    "model.num_select": {
        "type": "integer",
        "values": [50, 100, 200, 300],
    },
    "train.optim.lr": {
        "type": "float",
        "minimum": 1.0e-5,
        "maximum": 5.0e-4,
        "scale": "log",
    },
    "train.optim.lr_backbone": {
        "type": "float",
        "minimum": 1.0e-6,
        "maximum": 5.0e-5,
        "scale": "log",
    },
    "train.optim.weight_decay": {
        "type": "float",
        "minimum": 1.0e-5,
        "maximum": 1.0e-3,
        "scale": "log",
    },
}

FROZEN_CANDIDATE_BUDGET = 24
FROZEN_TRAINING_EPOCHS = 3
FROZEN_SEARCH_SEED = 271828
FROZEN_TRAINING_SEED = 1234
FROZEN_CALIBRATION_POINTS_PER_ARM = 2
FROZEN_INVALID_RECOVERY_ISSUES_PER_ARM = 1
FROZEN_LATENCY_RETENTION = 0.90
FROZEN_LATENCY_TOLERANCE_MS = 0.73553775
FROZEN_VALIDATION_SANITY_MIN_MASK_AP = 0.05
FROZEN_SLURM_RETRY_CAP = 10
FROZEN_BATCH_SIZE_PER_REPLICA = 4
FROZEN_QUALIFICATION_VERSION = 2
FROZEN_CHECKPOINT_INTERVAL_EPOCHS = 1
CHECKPOINT_RESUME_POLICY = {
    "kind": "same_job_exact_epoch_step_max_with_history_v1",
    "checkpoint_interval": FROZEN_CHECKPOINT_INTERVAL_EPOCHS,
    "checkpoint_interval_unit": "epoch",
    "resume_field": "train.resume_training_checkpoint_path",
    "same_job_only": True,
    "symlinks_eligible": False,
    "post_requeue_missing_checkpoint_behavior": "fail_closed",
    "selection_key": ["epoch", "step", "filename"],
}

# The pinned TAO 7.1 SQSH does not accept Lightning's strategy alias as the
# value of ``train.distributed_strategy``: that field is restricted to
# ``ddp``/``fsdp``.  Its Mask Grounding DINO launcher does, however, resolve
# the supported pair below to Lightning's unused-parameter-aware DDP strategy.
# V1 used activation checkpointing and therefore resolved to plain DDP, which
# failed on the first distributed batch.  V2 changes only that effective DDP
# behavior; data, PTMs, fidelity, objectives, seeds, and search space stay
# frozen.
FROZEN_TAO_DISTRIBUTED_STRATEGY = "ddp"
FROZEN_ACTIVATION_CHECKPOINT = False
FROZEN_LIGHTNING_DDP_STRATEGY = "ddp_find_unused_parameters_true"
FROZEN_DDP_STRATEGY_RESOLUTION = {
    "tao_config_value": FROZEN_TAO_DISTRIBUTED_STRATEGY,
    "activation_checkpoint": FROZEN_ACTIVATION_CHECKPOINT,
    "resolved_lightning_strategy": FROZEN_LIGHTNING_DDP_STRATEGY,
    "direct_alias_is_valid_tao_config_value": False,
    "resolution_source": (
        "pinned_mask_grounding_dino_train_launcher_branch"
    ),
}
FROZEN_CONTIGUOUS_VALIDATION_JSON = (
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
    "cross_model_automl_20260729/"
    "mask_grounding_dino_coco2017_od_v1/"
    "instances_val2017_remapped.json"
)
FROZEN_CONTIGUOUS_VALIDATION_SHA256 = (
    "9c9af9918e29292adfaa78a694d471e2be6d226e150300d9f4b22c2d77723ebc"
)
FROZEN_CONTIGUOUS_MANIFEST_SHA256 = (
    "3c2d09d20211017575a2c51a6797ef91f1939340d978a5d11d1d1edab1a30b2d"
)
FROZEN_TEXT_ENCODER_ROOT = (
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/ptms/"
    "huggingface/bert-base-uncased/"
    "86b5e0934494bd15c9632b12f734a8a67f723594"
)
FROZEN_TEXT_ENCODER_TREE_SHA256 = (
    "04cd5cc67804f4752df93e7c05dd51d904e82fc05d28794ddb03504cca689fb5"
)
FROZEN_HARDWARE = {
    "gpu_name": "NVIDIA A100-SXM4-80GB",
    "compute_capability": "8.0",
    "total_memory_bytes": 85174583296,
}
FROZEN_SQSH = {
    "path": (
        "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
        "nvcr.io_nvstaging_tao_tao-toolkit-pyt_7.1.0-rc-245-multiarch.sqsh"
    ),
    "sha256": (
        "e36640f9ae7a03bc80828cf7de93bd6bdbbb0fecf509a71a243be0ab5b497fc2"
    ),
    "image_reference": (
        "nvcr.io/nvstaging/tao/tao-toolkit-pyt:"
        "7.1.0-rc-245-multiarch"
    ),
}
LATENCY_PROTOCOL = {
    "warmup_iterations": 50,
    "timed_iterations": 100,
    "repeated_rounds": 5,
    "preloaded_batches": 16,
    "benchmark_seed": 20260727,
    "tail_percentile": 95.0,
    "bootstrap_resamples": 5000,
    "bootstrap_confidence_level": 0.95,
    "bootstrap_seed": 424242,
    "batch_size_per_replica": 1,
    "expected_replicas": 8,
    "precision": "fp32",
    "timed_scope": (
        "mask_grounding_dino_model_forward_plus_gpu_mask_postprocess;"
        "preprocessing_and_text_tokenization_excluded"
    ),
    "excluded_scope": [
        "checkpoint_load",
        "disk_io",
        "image_decode",
        "resize_normalize",
        "host_to_device_transfer",
        "category_prompt_tokenization",
        "instance_mask_serialization",
        "metric_accumulation",
        "distributed_gather",
    ],
    "synchronization": "accelerator_sync_before_and_after_each_sample",
    "replica_alignment": "nccl_barrier_before_each_timed_sample",
    "measurement_role": "selection_time",
    "raw_samples_per_candidate": 4000,
    "validity_thresholds": {
        "max_robust_cv": 0.10,
        "max_round_median_range_fraction": 0.05,
        "max_absolute_round_drift_fraction": 0.05,
        "max_device_median_range_fraction": 0.05,
        "max_bootstrap_ci_width_fraction": 0.03,
    },
}


class CampaignContractError(ValueError):
    """The Mask Grounding DINO campaign contract is inconsistent."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_fraction(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise CampaignContractError(f"{name} must be finite in (0, 1]")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CampaignContractError(
            f"{name} must be finite in (0, 1]"
        ) from exc
    if not math.isfinite(number) or not 0.0 < number <= 1.0:
        raise CampaignContractError(f"{name} must be finite in (0, 1]")
    return number


def mask_grounding_dino_registry_snapshot() -> dict[str, Any]:
    """Snapshot all official repository-owned Mask Grounding DINO PTMs."""
    registry = load_ptm_registry()
    model = registry.to_dict()["models"]["mask_grounding_dino"]
    records = []
    for record in model["checkpoints"]:
        if (
            record.get("source", {}).get("official") is not True
            or record.get("model_family") != "mask_grounding_dino"
            or "grounded_instance_segmentation"
            not in record.get("task_compatibility", ())
        ):
            raise CampaignContractError(
                "invalid official Mask Grounding DINO registry record: "
                f"{record.get('id')}"
            )
        records.append(
            {
                "id": record["id"],
                "status": record["status"],
                "status_reason": record.get("status_reason"),
                "source": copy.deepcopy(record["source"]),
                "sha256": record.get("sha256"),
                "expected_size_bytes": record["expected_size_bytes"],
                "checkpoint_target": record["checkpoint_target"],
                "architecture": record["architecture"],
                "backbone": record["backbone"],
                "compatible_tao_versions": copy.deepcopy(
                    record.get("compatible_tao_versions")
                ),
                "input_contract": copy.deepcopy(record["input_contract"]),
                "default_spec_overrides": copy.deepcopy(
                    record.get("default_spec_overrides", {})
                ),
                "checkpoint_spec_file": copy.deepcopy(
                    record.get("checkpoint_spec_file")
                ),
                "registry_record_sha256": canonical_sha256(record),
            }
        )
    records.sort(key=lambda item: item["id"])
    expected_ids = {
        "mask_grounding_dino.commercial.swin_tiny.trainable.v2.1",
        "mask_grounding_dino.commercial.swin_tiny.trainable.v2.0",
        "mask_grounding_dino.commercial.swin_tiny.trainable.v1.0",
        "mask_grounding_dino.research.swin_tiny.trainable.v2.0",
    }
    if {item["id"] for item in records} != expected_ids:
        raise CampaignContractError(
            "the frozen repository inventory must contain exactly the four "
            "official Mask Grounding DINO Swin-T checkpoints"
        )
    return {
        "registry_version": registry.registry_version,
        "registry_sha256": registry.document_sha256,
        "default_ptm": model["default_ptm"],
        "records": records,
        "record_count": len(records),
        "supported_ids": [
            item["id"] for item in records if item["status"] == "supported"
        ],
        "unverified_ids": [
            item["id"] for item in records if item["status"] == "unverified"
        ],
    }


def validate_packaged_train_schema(skill_dir: str | Path) -> dict[str, Any]:
    root = Path(skill_dir)
    info_path = root / "references/skill_info.yaml"
    schema_path = root / "schemas/train.schema.json"
    template_path = root / "references/spec_template_train.yaml"
    for path in (info_path, schema_path, template_path):
        if not path.is_file():
            raise CampaignContractError(
                f"missing packaged skill artifact: {path}"
            )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("x_tao_schema", {}).get("network_arch") != "mask_grounding_dino":
        raise CampaignContractError(
            "packaged train schema is not Mask Grounding DINO"
        )
    defaults = schema.get("automl_default_parameters")
    if not isinstance(defaults, list):
        raise CampaignContractError(
            "packaged train schema lacks AutoML default parameters"
        )
    missing = sorted(set(SEARCH_PARAMETERS) - set(defaults))
    if missing:
        raise CampaignContractError(
            "frozen search parameters are not AutoML enabled: "
            + ", ".join(missing)
        )
    model_properties = schema.get("properties", {}).get("model", {}).get(
        "properties", {}
    )
    for name in ("enc_layers", "dec_layers"):
        field = model_properties.get(name, {})
        if (
            field.get("automl_enabled") is not False
            or field.get("minimum") != 6
            or field.get("maximum") != 6
            or field.get("default") != 6
        ):
            raise CampaignContractError(
                f"Mask Grounding DINO {name} must remain fixed at six"
            )
    return {
        "skill_info_path": str(info_path),
        "skill_info_sha256": sha256_file(info_path),
        "schema_path": str(schema_path),
        "schema_sha256": sha256_file(schema_path),
        "template_path": str(template_path),
        "template_sha256": sha256_file(template_path),
        "explicit_search_parameters": list(SEARCH_PARAMETERS),
        "non_train_fields_excluded": True,
        "fixed_architecture_depths": {
            "model.enc_layers": 6,
            "model.dec_layers": 6,
        },
    }


def mode_objective(mode: str) -> dict[str, Any]:
    objectives = [
        {
            "metric": "segm_val_mAP50_95",
            "direction": "maximize",
            "role": "accuracy",
        },
        {
            "metric": "latency_ms",
            "direction": "minimize",
            "role": "latency",
        },
    ]
    if mode == "accuracy":
        return {
            "selection_mode": mode,
            "objectives": objectives,
            "acquisition": "expected_improvement",
            "latency_accuracy_retention": None,
            "multi_objective_min_accuracy": None,
            "selection_policy": "highest_valid_accuracy",
        }
    if mode == "latency":
        return {
            "selection_mode": mode,
            "objectives": objectives,
            "acquisition": "constrained_expected_improvement",
            "latency_accuracy_retention": {
                "type": "relative",
                "retained_fraction": FROZEN_LATENCY_RETENTION,
                "reference": "best_observed_within_job",
                "reference_updates": "monotonic",
                "terminal_reference": "terminal_archive_accuracy_winner",
            },
            "multi_objective_min_accuracy": None,
            "selection_policy": "equivalent_fastest_accuracy_tiebreak",
        }
    if mode == "multi_objective":
        return {
            "selection_mode": mode,
            "objectives": objectives,
            "acquisition": "parego_expected_improvement",
            "latency_accuracy_retention": None,
            "multi_objective_min_accuracy": None,
            "selection_policy": "normalized_augmented_chebyshev",
        }
    raise CampaignContractError(f"unsupported mode: {mode!r}")


def mode_settings(campaign_id: str, mode: str) -> dict[str, Any]:
    objective = mode_objective(mode)
    settings = {
        "algorithm": "bayesian",
        "automl_max_recommendations": FROZEN_CANDIDATE_BUDGET,
        "automl_max_concurrent": 1,
        "campaign_id": campaign_id,
        "job_id": f"{campaign_id}-{mode}",
        "session_id": f"{campaign_id}-{mode}",
        "experiment_id": f"{campaign_id}-{mode}-observations",
        "random_seed": FROZEN_SEARCH_SEED,
        "objectives": [
            {"metric": item["metric"], "direction": item["direction"]}
            for item in objective["objectives"]
        ],
        "selection_mode": mode,
        "accuracy_metric": "segm_val_mAP50_95",
        "latency_metric": "latency_ms",
        "objective_acquisition": {
            "calibration_points": FROZEN_CALIBRATION_POINTS_PER_ARM,
            "augmentation_rho": 1.0e-6,
        },
        "objective_normalization": "pareto_front",
        "augmentation_rho": 1.0e-6,
        "accuracy_tolerance": 1.0e-12,
        "latency_tolerance": FROZEN_LATENCY_TOLERANCE_MS,
        "selection_score_tolerance": 1.0e-12,
        "latency_ci_low_metric": "latency_ci95_low_ms",
        "latency_ci_high_metric": "latency_ci95_high_ms",
        "multi_objective_min_accuracy": None,
        "run_baseline": False,
        "run_final_evaluation": False,
        "require_eval_fn_success": True,
        "automl_delete_intermediate_ckpt": False,
        "automl_checkpoint_retention_strategy": "terminal",
    }
    if mode == "latency":
        # Acquisition self-calibrates monotonically from observations in this
        # independent job. Terminal selection is frozen against the accuracy
        # winner in the completed archive.
        settings["latency_accuracy_retention"] = {
            "type": "relative",
            "retained_fraction": FROZEN_LATENCY_RETENTION,
            "reference": "accuracy_winner",
        }
    return settings


def custom_ranges() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in SEARCH_PARAMETERS:
        domain = SEARCH_SPACE[name]
        if "values" in domain:
            result[name] = {"valid_options": copy.deepcopy(domain["values"])}
        else:
            result[name] = {
                "valid_min": domain["minimum"],
                "valid_max": domain["maximum"],
            }
    return result


def profile_overrides(
    dataset_root: str,
    *,
    contiguous_validation_json: str = FROZEN_CONTIGUOUS_VALIDATION_JSON,
    text_encoder_root: str = FROZEN_TEXT_ENCODER_ROOT,
) -> dict[str, Any]:
    """Return the identical full-COCO OD-mask profile for every job.

    Training uses the lossless official COCO-to-ODVG projection. Validation and
    standalone evaluation use the separately staged contiguous COCO derivative
    required by TAO's OD evaluator. Category names are prompts; this contract
    does not claim phrase-grounding coverage.
    """
    if not isinstance(dataset_root, str) or not dataset_root.startswith(
        "/lustre/"
    ):
        raise CampaignContractError(
            "dataset root must be an absolute Lustre path"
        )
    if (
        not isinstance(contiguous_validation_json, str)
        or not contiguous_validation_json.startswith("/lustre/")
        or not isinstance(text_encoder_root, str)
        or not text_encoder_root.startswith("/lustre/")
    ):
        raise CampaignContractError(
            "validation annotation and text encoder must be Lustre paths"
        )
    train_images = f"{dataset_root}/images/train2017"
    val_images = f"{dataset_root}/images/val2017"
    train_odvg = (
        f"{dataset_root}/tao/mask_grounding_dino/train/"
        "instances_train2017_odvg.jsonl"
    )
    train_label_map = (
        f"{dataset_root}/tao/mask_grounding_dino/train/"
        "instances_train2017_odvg_labelmap.json"
    )
    evaluation_source = {
        "image_dir": val_images,
        "json_file": contiguous_validation_json,
        "data_type": "OD",
    }
    return {
        "model_name": "mask_grounding_dino_coco2017_category_prompted_od",
        "results_dir": "",
        "wandb": {"enable": False},
        "model": {
            "backbone": "swin_tiny_224_1k",
            "num_queries": 900,
            "num_feature_levels": 4,
            "return_interm_indices": [1, 2, 3, 4],
            "num_select": 300,
            "hidden_dim": 256,
            "enc_layers": 6,
            "dec_layers": 6,
            "dim_feedforward": 2048,
            "text_encoder_type": text_encoder_root,
            "max_text_len": 256,
            "has_mask": True,
            "num_region_queries": 100,
            "loss_types": ["labels", "boxes", "masks"],
        },
        "dataset": {
            "train_data_sources": [
                {
                    "image_dir": train_images,
                    "json_file": train_odvg,
                    "label_map": train_label_map,
                }
            ],
            "val_data_sources": copy.deepcopy(evaluation_source),
            "test_data_sources": copy.deepcopy(evaluation_source),
            "batch_size": FROZEN_BATCH_SIZE_PER_REPLICA,
            "workers": 8,
            "pin_memory": True,
            "dataset_type": "serialized",
            "max_labels": 80,
            "eval_class_ids": list(range(80)),
            "has_mask": True,
            "augmentation": {
                "scales": [
                    480,
                    512,
                    544,
                    576,
                    608,
                    640,
                    672,
                    704,
                    736,
                    768,
                    800,
                ],
                "input_mean": [0.485, 0.456, 0.406],
                "input_std": [0.229, 0.224, 0.225],
                "train_random_resize": [400, 500, 600],
                "horizontal_flip_prob": 0.5,
                "train_random_crop_min": 384,
                "train_random_crop_max": 600,
                "random_resize_max_size": 1333,
                "test_random_resize": 800,
                "fixed_padding": True,
                "fixed_random_crop": 1024,
            },
        },
        "train": {
            "num_gpus": 8,
            "gpu_ids": list(range(8)),
            "num_nodes": 1,
            "seed": FROZEN_TRAINING_SEED,
            "num_epochs": FROZEN_TRAINING_EPOCHS,
            "checkpoint_interval": FROZEN_CHECKPOINT_INTERVAL_EPOCHS,
            "checkpoint_interval_unit": "epoch",
            "validation_interval": 1,
            "resume_training_checkpoint_path": "",
            "results_dir": "",
            "precision": "fp32",
            "distributed_strategy": FROZEN_TAO_DISTRIBUTED_STRATEGY,
            "activation_checkpoint": FROZEN_ACTIVATION_CHECKPOINT,
            "cudnn": {"benchmark": False, "deterministic": True},
            "optim": {
                "optimizer": "AdamW",
                "monitor_name": "val_loss",
                "lr": 2.0e-4,
                "lr_backbone": 2.0e-5,
                "lr_linear_proj_mult": 0.1,
                "momentum": 0.9,
                "weight_decay": 1.0e-4,
                "lr_scheduler": "MultiStep",
                "lr_steps": [2],
                "lr_step_size": 2,
                "lr_decay": 0.1,
            },
        },
        "evaluate": {
            "num_gpus": 8,
            "gpu_ids": list(range(8)),
            "num_nodes": 1,
            "checkpoint": "",
            "trt_engine": "",
            "results_dir": "",
            "batch_size": -1,
            "conf_threshold": 0.0,
            "ioi_threshold": 0.5,
            "nms_threshold": 0.2,
            "text_threshold": 0.3,
        },
    }


def validate_dataset_record(dataset: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "id": "coco2017_full_category_prompted_grounded_instance_segmentation",
        "train_image_count": 118287,
        "validation_image_count": 5000,
        "train_instance_annotations": 860001,
        "validation_instance_annotations": 36781,
        "num_classes": 80,
        "train_instance_json_sha256": (
            "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d"
        ),
        "validation_instance_json_sha256": (
            "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
        ),
        "label_map_sha256": (
            "67f15c4dd7d52aa73025da8307dec17e907f13db6d5d82332a670f73da68c306"
        ),
        "train_odvg_jsonl_sha256": (
            "d5deb4f5cfe027786fb1ceb52632ad6d3ef027e95e434525ba715d6841fb2921"
        ),
        "train_odvg_label_map_sha256": (
            "02075d96f6bf06d061f9329b4775dc7c3bb5ac140c77bc5c0e465d305c46d6c1"
        ),
        "train_odvg_projected_images": 117266,
        "train_odvg_projected_annotations": 860001,
        "train_odvg_masks_preserved": 860001,
        "contiguous_validation_json_sha256": (
            FROZEN_CONTIGUOUS_VALIDATION_SHA256
        ),
        "contiguous_validation_manifest_sha256": (
            FROZEN_CONTIGUOUS_MANIFEST_SHA256
        ),
        "contiguous_validation_image_count": 5000,
        "contiguous_validation_annotation_count": 36781,
        "contiguous_validation_category_ids": list(range(80)),
        "contiguous_validation_remote_read_only": True,
        "content_sha256": (
            "deced9d6766344fe6fc69cd9de3bcff2cba456a14b3391d07bcedb74c250909e"
        ),
        "manifest_sha256": (
            "10566a60498de9998154f44a34445a488c9f030e09f2a7346d20a4a1c55f804e"
        ),
        "file_manifest_entry_count": 246593,
        "stage_manifest_sha256": (
            "437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d"
        ),
        "remote_read_only": True,
        "remote_writable_entries_after_lock": 0,
    }
    for key, expected in required.items():
        if dataset.get(key) != expected:
            raise CampaignContractError(
                f"COCO 2017 dataset field {key!r} changed"
            )
    root = dataset.get("prepared_root")
    if not isinstance(root, str) or not root.startswith("/lustre/"):
        raise CampaignContractError(
            "COCO 2017 prepared_root must be on Lustre"
        )
    if (
        not isinstance(dataset.get("manifest_path"), str)
        or not Path(dataset["manifest_path"]).is_absolute()
        or not isinstance(dataset.get("stage_manifest_path"), str)
        or not Path(dataset["stage_manifest_path"]).is_absolute()
        or not isinstance(dataset.get("stage_manifest_lustre_path"), str)
        or not dataset["stage_manifest_lustre_path"].startswith("/lustre/")
        or not isinstance(dataset.get("remote_file_manifest_path"), str)
        or not dataset["remote_file_manifest_path"].startswith("/lustre/")
        or dataset.get("contiguous_validation_json_path")
        != FROZEN_CONTIGUOUS_VALIDATION_JSON
        or not isinstance(
            dataset.get("contiguous_validation_manifest_path"), str
        )
        or not dataset["contiguous_validation_manifest_path"].startswith(
            "/lustre/"
        )
    ):
        raise CampaignContractError(
            "COCO 2017 local and Lustre provenance paths are invalid"
        )
    return copy.deepcopy(dict(dataset))


def build_preregistered_contract(
    *,
    campaign_id: str,
    dataset: Mapping[str, Any],
    skill_dir: str,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Build immutable intent without granting launch authorization."""
    _finite_fraction(FROZEN_LATENCY_RETENTION, "latency retention")
    dataset_record = validate_dataset_record(dataset)
    schema = validate_packaged_train_schema(skill_dir)
    ptm_inventory = mask_grounding_dino_registry_snapshot()
    value = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "model": "mask_grounding_dino",
        "network_arch": "mask_grounding_dino",
        "task": "category_prompted_grounded_instance_segmentation",
        "primary_accuracy_metric": "segm_val_mAP50_95",
        "dataset": dataset_record,
        "runtime": copy.deepcopy(dict(runtime)),
        "sqsh": copy.deepcopy(FROZEN_SQSH),
        "schema": schema,
        "ptm_inventory": ptm_inventory,
        "text_encoder": {
            "provider": "huggingface",
            "repository": "google-bert/bert-base-uncased",
            "revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
            "lustre_root": FROZEN_TEXT_ENCODER_ROOT,
            "tree_sha256": FROZEN_TEXT_ENCODER_TREE_SHA256,
            "offline_environment": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
        },
        "metric_contract": {
            "required": "segm_val_mAP50_95",
            "direction": "maximize",
            "scale": "fraction",
            "task_correct": True,
            "VG_overall_iou_is_not_an_alias": True,
            "status_key": "[segm] val_mAP@50-95",
            "known_repository_state": (
                "statically_implemented_runtime_qualification_required"
            ),
            "failure_policy": (
                "fail_closed_without_finite_coco_mask_ap_from_in_epoch_and_"
                "standalone_evaluation"
            ),
        },
        "qualification_policy": {
            "version": FROZEN_QUALIFICATION_VERSION,
            "kind": (
                "direct_full_gpu_train_eval_then_evidence_bound_runtime_"
                "eligibility"
            ),
            "cpu_model_runs": 0,
            "smoke_model_runs": 0,
            "mini_step_runs": 0,
            "nodes_per_job": 1,
            "gpus_per_job": 8,
            "full_dataset": True,
            "training_epochs": FROZEN_TRAINING_EPOCHS,
            "standalone_evaluation": True,
            "required_metric": "segm_val_mAP50_95",
            "registry_bypass_allowed": False,
            "runtime_local_eligibility": copy.deepcopy(
                runtime["runtime_local_eligibility"]
            ),
            "qualification_evidence_path": runtime[
                "qualification_evidence_path"
            ],
            "ptm_stage_manifest_path": runtime["ptm_stage_manifest_path"],
            "distributed_strategy_resolution": copy.deepcopy(
                FROZEN_DDP_STRATEGY_RESOLUTION
            ),
            "predecessor_failure_evidence": copy.deepcopy(
                runtime["predecessor_failure_evidence"]
            ),
        },
        "execution": {
            "kind": "objective_aware_three_mode_search",
            "cpu_runs": 0,
            "smoke_runs": 0,
            "local_model_runs": 0,
            "independent_mode_jobs": True,
            "shared_archive": False,
            "first_candidate_gate": True,
            "automatic_remaining_budget_release": True,
            "automatic_trigger": True,
            "nodes_per_child": 1,
            "gpus_per_child": 8,
            "container_mode": "pinned_sqsh",
        },
        "search": {
            "algorithm": "bayesian",
            "implementation": (
                "hierarchical_ptm_objective_aware_bayesian_v1"
            ),
            "candidate_budget_per_mode": FROZEN_CANDIDATE_BUDGET,
            "search_seed": FROZEN_SEARCH_SEED,
            "training_seed": FROZEN_TRAINING_SEED,
            "training_epochs": FROZEN_TRAINING_EPOCHS,
            "calibration_points_per_arm": (
                FROZEN_CALIBRATION_POINTS_PER_ARM
            ),
            "invalid_recovery_issues_per_arm": (
                FROZEN_INVALID_RECOVERY_ISSUES_PER_ARM
            ),
            "parameters": list(SEARCH_PARAMETERS),
            "space": copy.deepcopy(SEARCH_SPACE),
            "space_sha256": canonical_sha256(SEARCH_SPACE),
            "latency_accuracy_retention": FROZEN_LATENCY_RETENTION,
            "latency_practical_tolerance_ms": (
                FROZEN_LATENCY_TOLERANCE_MS
            ),
            "ptm_representation": "hierarchical_nonordinal_arms",
            "single_arm_is_not_ordinal_encoding": True,
            "ptm_policy_by_mode": {
                "accuracy": "all_runtime_eligible",
                "latency": "all_runtime_eligible",
                "multi_objective": "all_runtime_eligible",
            },
        },
        "validation_sanity_gate": {
            "metric": "segm_val_mAP50_95",
            "minimum": FROZEN_VALIDATION_SANITY_MIN_MASK_AP,
            "role": "experiment_correctness_gate_not_product_selection",
            "rationale": (
                "For COCO 2017 80-class instance segmentation, mask AP50-95 below "
                "0.05 triggers metric/data/PTM/fidelity root-cause analysis."
            ),
            "low_finite_metric_automatically_accepted": False,
        },
        "latency_protocol": copy.deepcopy(LATENCY_PROTOCOL),
        "modes": [
            {
                "mode": mode,
                "observation_namespace": (
                    f"{campaign_id}-{mode}-observations"
                ),
                "observation_sharing": False,
                "initial_observation_ids": [],
                "objective": mode_objective(mode),
                "settings": mode_settings(campaign_id, mode),
            }
            for mode in MODES
        ],
        "agent_intervention_flags": {name: False for name in AGENT_FLAGS},
        "selection_isolation_flags": {
            name: False for name in SELECTION_FLAGS
        },
    }
    value["contract_sha256"] = canonical_sha256(value)
    return value


def validate_contract(document: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(document))
    observed = value.pop("contract_sha256", None)
    if observed != canonical_sha256(value):
        raise CampaignContractError("campaign contract integrity failed")
    campaign_id = value.get("campaign_id")
    expected_modes = [
        {
            "mode": mode,
            "observation_namespace": (
                f"{campaign_id}-{mode}-observations"
            ),
            "observation_sharing": False,
            "initial_observation_ids": [],
            "objective": mode_objective(mode),
            "settings": mode_settings(str(campaign_id), mode),
        }
        for mode in MODES
    ]
    if (
        not isinstance(campaign_id, str)
        or not campaign_id
        or value.get("model") != "mask_grounding_dino"
        or value.get("network_arch") != "mask_grounding_dino"
        or value.get("task") != "category_prompted_grounded_instance_segmentation"
        or value.get("primary_accuracy_metric") != "segm_val_mAP50_95"
        or value.get("execution", {}).get("cpu_runs") != 0
        or value.get("execution", {}).get("smoke_runs") != 0
        or value.get("execution", {}).get("gpus_per_child") != 8
        or value.get("execution", {}).get("container_mode")
        != "pinned_sqsh"
        or value.get("search", {}).get("space") != SEARCH_SPACE
        or value.get("modes") != expected_modes
        or value.get("metric_contract", {}).get(
            "VG_overall_iou_is_not_an_alias"
        )
        is not True
        or value.get("metric_contract", {}).get("status_key")
        != "[segm] val_mAP@50-95"
        or value.get("text_encoder", {}).get("lustre_root")
        != FROZEN_TEXT_ENCODER_ROOT
        or value.get("text_encoder", {}).get("tree_sha256")
        != FROZEN_TEXT_ENCODER_TREE_SHA256
        or value.get("qualification_policy", {}).get(
            "ptm_stage_manifest_path"
        )
        != value.get("runtime", {}).get("ptm_stage_manifest_path")
        or value.get("qualification_policy", {}).get("version")
        != FROZEN_QUALIFICATION_VERSION
        or value.get("qualification_policy", {}).get(
            "distributed_strategy_resolution"
        )
        != FROZEN_DDP_STRATEGY_RESOLUTION
        or value.get("qualification_policy", {}).get(
            "predecessor_failure_evidence"
        )
        != value.get("runtime", {}).get("predecessor_failure_evidence")
        or value.get("qualification_policy", {}).get(
            "runtime_local_eligibility"
        )
        != value.get("runtime", {}).get("runtime_local_eligibility")
    ):
        raise CampaignContractError("campaign execution policy changed")
    validate_dataset_record(value["dataset"])
    if value.get("sqsh") != FROZEN_SQSH:
        raise CampaignContractError("pinned SQSH identity changed")
    runtime = value.get("runtime", {})
    search = value.get("search", {})
    resume_predecessor = runtime.get("resume_predecessor_contract")
    if resume_predecessor is not None and (
        not isinstance(resume_predecessor, Mapping)
        or resume_predecessor.get("schema_version") != 1
        or resume_predecessor.get("kind")
        != "evaluator_overlay_only_successor"
        or not isinstance(resume_predecessor.get("path"), str)
        or not Path(resume_predecessor["path"]).is_absolute()
        or not _is_lower_sha256(resume_predecessor.get("file_sha256"))
        or not _is_lower_sha256(resume_predecessor.get("contract_sha256"))
        or resume_predecessor.get("campaign_id") != campaign_id
        or not isinstance(resume_predecessor.get("source_commit"), str)
        or len(resume_predecessor["source_commit"]) != 40
        or resume_predecessor.get("workspace_reuse_allowed") is not True
        or resume_predecessor.get("training_job_reuse_required") is not True
        or resume_predecessor.get("recommendation_change_allowed") is not False
        or resume_predecessor.get("training_relaunch_allowed") is not False
        or resume_predecessor.get("objective_policy_change_allowed") is not False
    ):
        raise CampaignContractError(
            "evaluator-overlay resume predecessor contract changed"
        )
    first_candidate_reuse = runtime.get("first_candidate_training_reuse")
    if first_candidate_reuse is not None:
        reuse_payload = copy.deepcopy(dict(first_candidate_reuse))
        reuse_sha256 = reuse_payload.pop("record_sha256", None)
        source_contract = first_candidate_reuse.get("source_contract", {})
        reuse_modes = first_candidate_reuse.get("modes")
        if (
            resume_predecessor is not None
            or reuse_sha256 != canonical_sha256(reuse_payload)
            or first_candidate_reuse.get("schema_version") != 1
            or first_candidate_reuse.get("kind")
            != "first_candidate_completed_training_reuse"
            or not isinstance(source_contract, Mapping)
            or not isinstance(source_contract.get("path"), str)
            or not Path(source_contract["path"]).is_absolute()
            or not _is_lower_sha256(source_contract.get("file_sha256"))
            or not _is_lower_sha256(source_contract.get("contract_sha256"))
            or not isinstance(source_contract.get("source_commit"), str)
            or len(source_contract["source_commit"]) != 40
            or not isinstance(
                first_candidate_reuse.get("source_runtime_root"), str
            )
            or not Path(
                first_candidate_reuse["source_runtime_root"]
            ).is_absolute()
            or set(reuse_modes or {}) != set(MODES)
            or first_candidate_reuse.get("fresh_controller_state_required")
            is not True
            or first_candidate_reuse.get("training_relaunch_allowed")
            is not False
            or first_candidate_reuse.get("objective_reuse_allowed") is not False
            or first_candidate_reuse.get("evaluation_reuse_allowed") is not False
            or first_candidate_reuse.get("latency_reuse_allowed") is not False
            or first_candidate_reuse.get("new_training_jobs_submitted") != 0
            or first_candidate_reuse.get("agent_selected_candidate") is not False
            or first_candidate_reuse.get("agent_overrode_observation") is not False
        ):
            raise CampaignContractError(
                "first-candidate completed-training reuse contract changed"
            )
        for mode in MODES:
            record = reuse_modes[mode]
            if not isinstance(record, Mapping):
                raise CampaignContractError(
                    f"{mode} first-candidate training reuse changed"
                )
            checkpoint = record.get("terminal_checkpoint", {})
            if (
                record.get("candidate_id") != f"{mode}_rec_0"
                or record.get("rec_id") != "0"
                or not _is_lower_sha256(record.get("candidate_fingerprint"))
                or not isinstance(record.get("checkpoint_id"), str)
                or not _is_lower_sha256(record.get("specs_sha256"))
                or not _is_lower_sha256(
                    record.get("recommendation_audit_sha256")
                )
                or not isinstance(record.get("source_train_job_id"), str)
                or not isinstance(record.get("source_results_dir"), str)
                or not record["source_results_dir"].startswith("lustre:///")
                or not isinstance(checkpoint, Mapping)
                or not isinstance(checkpoint.get("path"), str)
                or not checkpoint["path"].startswith("/lustre/")
                or not _is_lower_sha256(checkpoint.get("sha256"))
                or not isinstance(checkpoint.get("size_bytes"), int)
                or checkpoint["size_bytes"] < 1
                or not isinstance(record.get("source_state_file"), str)
                or not Path(record["source_state_file"]).is_absolute()
                or not isinstance(record.get("source_state_db"), str)
                or not Path(record["source_state_db"]).is_absolute()
                or not _is_lower_sha256(record.get("source_state_db_sha256"))
                or not isinstance(
                    record.get("source_candidate_evidence"), str
                )
                or not Path(record["source_candidate_evidence"]).is_absolute()
                or not _is_lower_sha256(
                    record.get("source_candidate_evidence_sha256")
                )
                or not isinstance(
                    record.get("discarded_non_observations"), int
                )
                or record["discarded_non_observations"] < 0
            ):
                raise CampaignContractError(
                    f"{mode} first-candidate training reuse changed"
                )
    predecessor = runtime.get("predecessor_failure_evidence", {})
    runtime_eligibility = runtime.get("runtime_local_eligibility", {})
    expected_eligibility_source_commit = (
        resume_predecessor["source_commit"]
        if isinstance(resume_predecessor, Mapping)
        else runtime.get("source_commit")
    )
    if (
        not isinstance(predecessor, Mapping)
        or not isinstance(predecessor.get("path"), str)
        or not Path(predecessor["path"]).is_absolute()
        or not isinstance(
            predecessor.get("file_sha256", predecessor.get("sha256")), str
        )
        or len(
            predecessor.get("file_sha256", predecessor.get("sha256", ""))
        )
        != 64
        or predecessor.get("workflow_count") != 4
        or predecessor.get("all_terminal_failures_preserved") is not True
        or predecessor.get("replacement_submitted") is not False
    ):
        raise CampaignContractError(
            "preserved v1 qualification evidence contract changed"
        )
    if (
        not isinstance(runtime_eligibility, Mapping)
        or runtime_eligibility.get("schema_version") != 2
        or runtime_eligibility.get("kind")
        != "direct_full_gpu_qualification_runtime_local_v2"
        or runtime_eligibility.get("enabled") is not True
        or runtime_eligibility.get("scope")
        != "campaign_local_in_memory_projection"
        or runtime_eligibility.get("model") != "mask_grounding_dino"
        or runtime_eligibility.get("task")
        != "category_prompted_grounded_instance_segmentation"
        or runtime_eligibility.get("tao_version") != "7.1.0"
        or runtime_eligibility.get("container_sha256")
        != FROZEN_SQSH["sha256"]
        or runtime_eligibility.get("base_registry_sha256")
        != value.get("ptm_inventory", {}).get("registry_sha256")
        or runtime_eligibility.get("base_registry_version")
        != value.get("ptm_inventory", {}).get("registry_version")
        or runtime_eligibility.get("eligibility_source_commit")
        != expected_eligibility_source_commit
        or runtime_eligibility.get("wheel_sha256")
        != runtime.get("wheel_sha256")
        or runtime_eligibility.get("sdk_commit")
        != runtime.get("sdk_commit")
        or runtime_eligibility.get("skills_commit")
        != runtime.get("skills_commit")
        or any(
            runtime_eligibility.get(name) is not False
            for name in (
                "repository_registry_mutation_allowed",
                "failed_arm_promotion_allowed",
                "unsupported_arm_promotion_allowed",
                "agent_override_allowed",
            )
        )
        or any(
            not isinstance(runtime_eligibility.get(name), str)
            or len(runtime_eligibility[name]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in runtime_eligibility[name]
            )
            for name in (
                "base_registry_sha256",
                "qualification_file_sha256",
                "qualification_evidence_sha256",
                "qualification_contract_sha256",
                "qualification_campaign_sha256",
                "wheel_sha256",
            )
        )
        or not isinstance(
            runtime_eligibility.get("eligibility_source_commit"), str
        )
        or len(runtime_eligibility["eligibility_source_commit"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in runtime_eligibility[
                "eligibility_source_commit"
            ]
        )
        or any(
            not isinstance(runtime_eligibility.get(name), str)
            or len(runtime_eligibility[name]) != 40
            or any(
                character not in "0123456789abcdef"
                for character in runtime_eligibility[name]
            )
            for name in ("sdk_commit", "skills_commit")
        )
    ):
        raise CampaignContractError(
            "runtime-local PTM eligibility contract changed"
        )
    if runtime_eligibility.get("qualification_successor_version") == 5:
        evaluation_overlay = runtime.get("evaluation_overlay", {})
        successor_sha_fields = (
            "qualification_contract_file_sha256",
            "training_qualification_file_sha256",
            "training_qualification_evidence_sha256",
            "training_qualification_contract_file_sha256",
            "training_qualification_contract_sha256",
            "ptm_stage_manifest_sha256",
            "ptm_stage_content_sha256",
            "qualification_source_wheel_sha256",
            "metric_recovery_overlay_sha256",
        )
        successor_commit_fields = (
            "qualification_source_commit",
            "qualification_source_sdk_commit",
            "qualification_source_skills_commit",
            "metric_recovery_source_commit",
        )
        if (
            runtime.get("qualification_contract_path")
            != runtime_eligibility.get("qualification_contract_path")
            or runtime.get("qualification_contract_file_sha256")
            != runtime_eligibility.get(
                "qualification_contract_file_sha256"
            )
            or runtime_eligibility.get("ptm_stage_manifest_path")
            != runtime.get("ptm_stage_manifest_path")
            or runtime_eligibility.get("ptm_stage_manifest_sha256")
            != runtime.get("ptm_stage_manifest_sha256")
            or runtime_eligibility.get("ptm_stage_content_sha256")
            != runtime.get("ptm_stage_content_sha256")
            or runtime_eligibility.get("predecessor_failure_evidence")
            != predecessor
            or runtime_eligibility.get("evaluation_recovery_jobs_submitted")
            != 4
            or runtime_eligibility.get("training_jobs_submitted") != 0
            or runtime_eligibility.get("replacement_workflows_submitted")
            is not True
            or runtime_eligibility.get("replacement_workflow_count") != 4
            or runtime_eligibility.get("checkpoint_resume_policy")
            != CHECKPOINT_RESUME_POLICY
            or any(
                not isinstance(runtime_eligibility.get(name), str)
                or not Path(runtime_eligibility[name]).is_absolute()
                for name in (
                    "qualification_contract_path",
                    "training_qualification_path",
                    "training_qualification_contract_path",
                    "ptm_stage_manifest_path",
                )
            )
            or any(
                not _is_lower_sha256(runtime_eligibility.get(name))
                for name in successor_sha_fields
            )
            or any(
                not isinstance(runtime_eligibility.get(name), str)
                or len(runtime_eligibility[name]) != 40
                or any(
                    character not in "0123456789abcdef"
                    for character in runtime_eligibility[name]
                )
                for name in successor_commit_fields
            )
            or not isinstance(evaluation_overlay, Mapping)
            or evaluation_overlay.get("schema_version") != 1
            or evaluation_overlay.get("source_repository") != "tao-pytorch"
            or evaluation_overlay.get("source_commit")
            != runtime_eligibility.get("metric_recovery_source_commit")
            or evaluation_overlay.get("archive_sha256")
            != runtime_eligibility.get("metric_recovery_overlay_sha256")
            or not isinstance(evaluation_overlay.get("archive_path"), str)
            or not evaluation_overlay["archive_path"].startswith("/lustre/")
            or not isinstance(
                evaluation_overlay.get("archive_size_bytes"), int
            )
            or isinstance(
                evaluation_overlay.get("archive_size_bytes"), bool
            )
            or evaluation_overlay["archive_size_bytes"] < 1
            or evaluation_overlay.get("archive_root")
            != "mask-grounding-dino-coco-evaluator-overlay"
            or evaluation_overlay.get("base_site_packages")
            != "/usr/local/lib/python3.12/dist-packages"
            or evaluation_overlay.get("installed_package_mutated") is not False
        ):
            raise CampaignContractError(
                "v5 evaluation-recovery eligibility contract changed"
            )
    if (
        value.get("ptm_inventory") != mask_grounding_dino_registry_snapshot()
        or value.get("latency_protocol") != LATENCY_PROTOCOL
        or search.get("algorithm") != "bayesian"
        or search.get("implementation")
        != "hierarchical_ptm_objective_aware_bayesian_v1"
        or search.get("candidate_budget_per_mode")
        != FROZEN_CANDIDATE_BUDGET
        or search.get("search_seed") != FROZEN_SEARCH_SEED
        or search.get("training_seed") != FROZEN_TRAINING_SEED
        or search.get("training_epochs") != FROZEN_TRAINING_EPOCHS
        or search.get("calibration_points_per_arm")
        != FROZEN_CALIBRATION_POINTS_PER_ARM
        or search.get("invalid_recovery_issues_per_arm")
        != FROZEN_INVALID_RECOVERY_ISSUES_PER_ARM
        or search.get("parameters") != list(SEARCH_PARAMETERS)
        or search.get("space_sha256") != canonical_sha256(SEARCH_SPACE)
        or search.get("latency_accuracy_retention")
        != FROZEN_LATENCY_RETENTION
        or search.get("latency_practical_tolerance_ms")
        != FROZEN_LATENCY_TOLERANCE_MS
        or search.get("ptm_representation")
        != "hierarchical_nonordinal_arms"
        or search.get("single_arm_is_not_ordinal_encoding") is not True
    ):
        raise CampaignContractError(
            "PTM inventory, objective-aware search, or latency policy changed"
        )
    for name in (
        "ptm_stage_manifest_sha256",
        "ptm_stage_content_sha256",
    ):
        digest = runtime.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
        ):
            raise CampaignContractError(
                f"runtime {name} must be lowercase SHA-256"
            )
    agent_flags = value.get("agent_intervention_flags")
    selection_flags = value.get("selection_isolation_flags")
    if (
        not isinstance(agent_flags, Mapping)
        or set(agent_flags) != set(AGENT_FLAGS)
        or any(item is not False for item in agent_flags.values())
    ):
        raise CampaignContractError(
            "agent intervention flags must remain false"
        )
    if (
        not isinstance(selection_flags, Mapping)
        or set(selection_flags) != set(SELECTION_FLAGS)
        or any(item is not False for item in selection_flags.values())
    ):
        raise CampaignContractError(
            "selection isolation flags must remain false"
        )
    value["contract_sha256"] = observed
    return value


__all__ = [
    "AGENT_FLAGS",
    "CampaignContractError",
    "FROZEN_BATCH_SIZE_PER_REPLICA",
    "FROZEN_ACTIVATION_CHECKPOINT",
    "FROZEN_CALIBRATION_POINTS_PER_ARM",
    "FROZEN_CANDIDATE_BUDGET",
    "FROZEN_CONTIGUOUS_MANIFEST_SHA256",
    "FROZEN_CONTIGUOUS_VALIDATION_JSON",
    "FROZEN_CONTIGUOUS_VALIDATION_SHA256",
    "FROZEN_HARDWARE",
    "FROZEN_LATENCY_RETENTION",
    "FROZEN_LATENCY_TOLERANCE_MS",
    "FROZEN_LIGHTNING_DDP_STRATEGY",
    "FROZEN_DDP_STRATEGY_RESOLUTION",
    "FROZEN_QUALIFICATION_VERSION",
    "FROZEN_SEARCH_SEED",
    "FROZEN_SLURM_RETRY_CAP",
    "FROZEN_SQSH",
    "FROZEN_TEXT_ENCODER_ROOT",
    "FROZEN_TEXT_ENCODER_TREE_SHA256",
    "FROZEN_TRAINING_EPOCHS",
    "FROZEN_TAO_DISTRIBUTED_STRATEGY",
    "FROZEN_VALIDATION_SANITY_MIN_MASK_AP",
    "LATENCY_PROTOCOL",
    "MODES",
    "SEARCH_PARAMETERS",
    "SEARCH_SPACE",
    "SELECTION_FLAGS",
    "build_preregistered_contract",
    "custom_ranges",
    "mask_grounding_dino_registry_snapshot",
    "mode_objective",
    "mode_settings",
    "profile_overrides",
    "sha256_file",
    "validate_contract",
    "validate_dataset_record",
    "validate_packaged_train_schema",
]
