#!/usr/bin/env python3

"""Frozen RT-DETR three-mode campaign contract.

This module contains the preregistered search and execution policy.  It is
deliberately independent of qualification metrics and selected winners.
Qualification decides only which of the four official checkpoint arms are
runtime eligible; it cannot change these ranges, seeds, budgets, or policies.
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

from .qualification_gate import (
    AGENT_FLAGS,
    EXPECTED_PTMS,
    QualificationDecision,
)


MODES = ("accuracy", "latency", "multi_objective")
SELECTION_FLAGS = (
    "selector_invoked_on_matched_measurements",
    "selection_time_objectives_replaced",
    "measurements_feed_selection",
    "measurements_feed_reselection",
    "algorithm_selected_candidate_overridden",
)
SEARCH_PARAMETERS = (
    "model.enc_layers",
    "model.dec_layers",
    "model.num_queries",
    "model.num_select",
    "train.optim.lr",
    "train.optim.weight_decay",
)
SEARCH_SPACE = {
    "model.enc_layers": {
        "type": "integer",
        "minimum": 1,
        "maximum": 3,
        "values": [1, 2, 3],
    },
    "model.dec_layers": {
        "type": "integer",
        "minimum": 3,
        "maximum": 6,
        "values": [3, 4, 5, 6],
    },
    "model.num_queries": {
        "type": "integer",
        "minimum": 100,
        "maximum": 300,
        "values": [100, 200, 300],
    },
    "model.num_select": {
        "type": "integer",
        "minimum": 50,
        "maximum": 300,
        "values": [50, 100, 200, 300],
        "depends_on": "model.num_queries",
        "constraint": "value <= model.num_queries",
        "enforcement": "registered_detr_network_constraint",
    },
    "train.optim.lr": {
        "type": "float",
        "minimum": 1.0e-5,
        "maximum": 5.0e-4,
    },
    "train.optim.weight_decay": {
        "type": "float",
        "minimum": 1.0e-6,
        "maximum": 1.0e-3,
    },
}
FROZEN_CANDIDATE_BUDGET = 20
FROZEN_TRAINING_EPOCHS = 10
FROZEN_SEARCH_SEED = 271828
FROZEN_TRAINING_SEED = 1234
FROZEN_CALIBRATION_POINTS_PER_ARM = 2
FROZEN_LATENCY_RETENTION = 0.90
FROZEN_LATENCY_TOLERANCE_MS = 0.73553775
FROZEN_SLURM_RETRY_CAP = 10
FROZEN_HARDWARE = {
    "gpu_name": "NVIDIA A100-SXM4-80GB",
    "compute_capability": "8.0",
    "total_memory_bytes": 85174583296,
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
    "timed_scope": "model_forward_plus_rtdetr_gpu_postprocess",
    "excluded_scope": [
        "checkpoint_load",
        "disk_io",
        "decode_resize_normalize",
        "host_to_device_transfer",
        "coco_accumulation",
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
    """The RT-DETR campaign contract is inconsistent."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def validate_packaged_train_schema(skill_dir: str | Path) -> dict[str, Any]:
    """Prove the explicit train search has no leaked non-train parameters."""
    root = Path(skill_dir)
    schema_path = root / "schemas/train.schema.json"
    template_path = root / "references/spec_template_train.yaml"
    if not schema_path.is_file() or not template_path.is_file():
        raise CampaignContractError(
            "packaged RT-DETR train schema/template are unavailable"
        )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(schema, Mapping):
        raise CampaignContractError("packaged train schema must be an object")
    defaults = schema.get("automl_default_parameters")
    disabled = schema.get("automl_disabled_parameters")
    if not isinstance(defaults, list) or not isinstance(disabled, list):
        raise CampaignContractError(
            "packaged train schema lacks AutoML parameter metadata"
        )
    leaked = sorted(
        name
        for name in SEARCH_PARAMETERS
        if name.startswith(
            (
                "quantize",
                "distill",
                "evaluate",
                "inference",
                "export",
                "gen_trt_engine",
            )
        )
    )
    if leaked:
        raise CampaignContractError(
            "RT-DETR search contains leaked non-train fields: "
            + ", ".join(leaked)
        )
    missing = sorted(set(SEARCH_PARAMETERS) - set(defaults))
    if missing:
        raise CampaignContractError(
            "packaged schema does not enable frozen parameters: "
            + ", ".join(missing)
        )
    return {
        "schema_path": str(schema_path),
        "schema_sha256": sha256_file(schema_path),
        "template_path": str(template_path),
        "template_sha256": sha256_file(template_path),
        "explicit_search_parameters": list(SEARCH_PARAMETERS),
        "non_train_fields_excluded": True,
        "leaked_quantize_fields": [],
    }


def mode_objective(mode: str) -> dict[str, Any]:
    objectives = [
        {"metric": "mAP50", "direction": "maximize", "role": "accuracy"},
        {"metric": "latency_ms", "direction": "minimize", "role": "latency"},
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
                "reference": "accuracy_winner",
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
    """Map one mode to production objective-aware AutoML settings."""
    _finite_fraction(FROZEN_LATENCY_RETENTION, "latency retention")
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
        "accuracy_metric": "mAP50",
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
        settings["latency_accuracy_retention"] = copy.deepcopy(
            objective["latency_accuracy_retention"]
        )
    return settings


def custom_ranges() -> dict[str, dict[str, Any]]:
    """Return bounded production ranges in the schema adapter format."""
    ranges: dict[str, dict[str, Any]] = {}
    for name in SEARCH_PARAMETERS:
        domain = SEARCH_SPACE[name]
        ranges[name] = {
            "valid_min": domain["minimum"],
            "valid_max": domain["maximum"],
        }
        if "values" in domain:
            ranges[name]["valid_options"] = list(domain["values"])
        if name == "model.num_select":
            ranges[name]["depends_on"] = "model.num_queries"
    return ranges


def profile_overrides(
    qualification_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Return identical dataset/fidelity/resource values for every PTM arm."""
    dataset = qualification_manifest["dataset"]
    train = dataset["splits"]["train"]
    validation = dataset["splits"]["validation"]
    if (
        dataset.get("id") != "tao_od_synthetic_full_dino_coco"
        or train.get("image_count") != 1414
        or validation.get("image_count") != 353
    ):
        raise CampaignContractError("shared synthetic dataset identity changed")
    return {
        "results_dir": "",
        "wandb": {"enable": False},
        "dataset": {
            "train_data_sources": [
                {
                    "image_dir": train["image_dir"],
                    "json_file": train["annotation"],
                }
            ],
            "val_data_sources": {
                "image_dir": validation["image_dir"],
                "json_file": validation["annotation"],
            },
            "num_classes": dataset["num_classes"],
            "eval_class_ids": list(dataset["eval_class_ids"]),
            "remap_mscoco_category": False,
            "batch_size": 4,
            "workers": 8,
        },
        "train": {
            "num_gpus": 8,
            "gpu_ids": list(range(8)),
            "num_nodes": 1,
            "seed": FROZEN_TRAINING_SEED,
            "num_epochs": FROZEN_TRAINING_EPOCHS,
            "checkpoint_interval": FROZEN_TRAINING_EPOCHS,
            "checkpoint_interval_unit": "epoch",
            "validation_interval": 1,
            "resume_training_checkpoint_path": "",
            "results_dir": "",
            "is_dry_run": False,
            "precision": "fp32",
            "distributed_strategy": "ddp",
            "cudnn": {"benchmark": False, "deterministic": True},
            "optim": {"lr_steps": [8], "lr_step_size": 8},
        },
    }


def build_preregistered_contract(
    *,
    campaign_id: str,
    qualification_manifest: Mapping[str, Any],
    decision: QualificationDecision,
) -> dict[str, Any]:
    """Build the frozen campaign intent after the qualification gate passes."""
    decision.assert_runtime_ready()
    if decision.checkpoint_ids != EXPECTED_PTMS:
        raise CampaignContractError(
            "all and only the four qualified PTMs must enter the campaign"
        )
    registry = load_ptm_registry()
    ptms = []
    for checkpoint_id in decision.checkpoint_ids:
        record = registry.checkpoint(checkpoint_id)
        if record.get("status") != "supported":
            raise CampaignContractError(
                f"PTM {checkpoint_id!r} is not runtime supported"
            )
        ptms.append(
            {
                "id": checkpoint_id,
                "registry_record_sha256": canonical_sha256(record),
                "checkpoint_target": record["checkpoint_target"],
                "input_contract": copy.deepcopy(record["input_contract"]),
                "input_contract_sha256": canonical_sha256(
                    record["input_contract"]
                ),
            }
        )
    schema = validate_packaged_train_schema(
        qualification_manifest["runtime"]["skill_dir"]
    )
    contract = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "model": "rtdetr",
        "task": "object_detection",
        "qualification_gate": decision.to_dict(),
        "execution": {
            "kind": "direct_full_search",
            "cpu_runs": 0,
            "smoke_runs": 0,
            "local_model_runs": 0,
            "independent_mode_jobs": True,
            "shared_archive": False,
            "first_candidate_gate": True,
            "automatic_remaining_budget_release": True,
            "nodes_per_child": 1,
            "gpus_per_child": 8,
        },
        "runtime": copy.deepcopy(qualification_manifest["runtime"]),
        "dataset": copy.deepcopy(qualification_manifest["dataset"]),
        "ptms": ptms,
        "schema": schema,
        "search": {
            "algorithm": "bayesian",
            "implementation": "hierarchical_ptm_objective_aware_bayesian_v1",
            "candidate_budget_per_mode": FROZEN_CANDIDATE_BUDGET,
            "search_seed": FROZEN_SEARCH_SEED,
            "training_seed": FROZEN_TRAINING_SEED,
            "training_epochs": FROZEN_TRAINING_EPOCHS,
            "calibration_points_per_arm": (
                FROZEN_CALIBRATION_POINTS_PER_ARM
            ),
            "parameters": list(SEARCH_PARAMETERS),
            "space": copy.deepcopy(SEARCH_SPACE),
            "space_sha256": canonical_sha256(SEARCH_SPACE),
            "latency_accuracy_retention": FROZEN_LATENCY_RETENTION,
            "latency_practical_tolerance_ms": (
                FROZEN_LATENCY_TOLERANCE_MS
            ),
            "ptm_representation": "hierarchical_nonordinal_arms",
            "accuracy_ptm_policy": "all",
            "latency_ptm_policy": "all",
            "multi_objective_ptm_policy": "all",
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
        "agent_intervention_flags": {
            name: False for name in AGENT_FLAGS
        },
        "selection_isolation_flags": {
            name: False for name in SELECTION_FLAGS
        },
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract


__all__ = [
    "CampaignContractError",
    "FROZEN_CANDIDATE_BUDGET",
    "FROZEN_CALIBRATION_POINTS_PER_ARM",
    "FROZEN_HARDWARE",
    "FROZEN_LATENCY_RETENTION",
    "FROZEN_LATENCY_TOLERANCE_MS",
    "FROZEN_SEARCH_SEED",
    "FROZEN_SLURM_RETRY_CAP",
    "FROZEN_TRAINING_EPOCHS",
    "EXPECTED_PTMS",
    "LATENCY_PROTOCOL",
    "MODES",
    "SEARCH_PARAMETERS",
    "SEARCH_SPACE",
    "SELECTION_FLAGS",
    "build_preregistered_contract",
    "canonical_bytes",
    "custom_ranges",
    "mode_objective",
    "mode_settings",
    "profile_overrides",
    "sha256_file",
    "validate_packaged_train_schema",
]
