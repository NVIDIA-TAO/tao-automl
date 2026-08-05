#!/usr/bin/env python3

"""Frozen Mask2Former/COCO 2017 three-mode campaign policy.

This is an experiment contract, not a launcher.  It deliberately keeps COCO
instance segmentation and COCO mask AP as the task/metric pair.  The current
TAO Mask2Former validation implementation is known to emit semantic mIoU
instead of mask AP, so the direct full-run qualification gate must fail closed
until the task-correct metric is actually available.
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

from . import runtime_overlay


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

# PTM identity is a hierarchical, non-ordinal outer arm.  The common inner
# search contains both accuracy-recovery parameters and supported parameters
# that materially change the inference graph/input cost.
SEARCH_PARAMETERS = (
    "model.mask_former.num_object_queries",
    "model.mask_former.dec_layers",
    "dataset.augmentation.test_min_size",
    "train.optim.lr",
    "train.optim.weight_decay",
)
SEARCH_SPACE = {
    "model.mask_former.num_object_queries": {
        "type": "integer",
        "minimum": 50,
        "maximum": 200,
    },
    "model.mask_former.dec_layers": {
        "type": "integer",
        "minimum": 4,
        "maximum": 10,
    },
    "dataset.augmentation.test_min_size": {
        "type": "integer",
        "minimum": 480,
        "maximum": 800,
    },
    "train.optim.lr": {
        "type": "float",
        "minimum": 2.0e-5,
        "maximum": 5.0e-4,
    },
    "train.optim.weight_decay": {
        "type": "float",
        "minimum": 1.0e-4,
        "maximum": 0.10,
    },
}

FROZEN_CANDIDATE_BUDGET = 20
FROZEN_TRAINING_EPOCHS = 3
FROZEN_SEARCH_SEED = 271828
FROZEN_TRAINING_SEED = 1234
FROZEN_CALIBRATION_POINTS_PER_ARM = 8
FROZEN_INVALID_RECOVERY_ISSUES_PER_ARM = 1
FROZEN_LATENCY_RETENTION = 0.90
FROZEN_LATENCY_TOLERANCE_MS = 0.73553775
FROZEN_VALIDATION_SANITY_MIN_MASK_AP = 0.05
FROZEN_SLURM_RETRY_CAP = 10
FROZEN_SLURM_PARTITION = "polar3"
FROZEN_SLURM_TIME_HOURS = 4.0
FROZEN_SLURM_TIMEOUT_HOURS = 3.8
FROZEN_SLURM_USE_REQUEUE = True
FROZEN_CHECKPOINT_INTERVAL_EPOCHS = 1
FROZEN_BATCH_SIZE_PER_REPLICA = 1
FROZEN_TEST_MAX_SIZE = 1333
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
FROZEN_WALLTIME_POLICY = {
    "contract_revision": "qualification_runtime_v3",
    "supersedes": "qualification_runtime_v2",
    "partition": FROZEN_SLURM_PARTITION,
    "observed_full_epoch_minutes_approx": 90.0,
    "training_epochs": FROZEN_TRAINING_EPOCHS,
    "observed_minimum_training_hours_approx": 4.5,
    "time_hours": FROZEN_SLURM_TIME_HOURS,
    "timeout_hours": FROZEN_SLURM_TIMEOUT_HOURS,
    "slurm_self_requeue": FROZEN_SLURM_USE_REQUEUE,
    "scheduler_timeout_headroom_minutes": 12.0,
    "checkpoint_interval_epochs": FROZEN_CHECKPOINT_INTERVAL_EPOCHS,
    "checkpoint_resume_policy": "same_job_exact_epoch_step_max_v1",
    "resume_field": "train.resume_training_checkpoint_path",
    "trusted_own_checkpoint_environment": (
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1"
    ),
    "resume_decision_record": (
        "mask2former_checkpoint_resume_decision.json"
    ),
    "applies_to": "qualification_and_automl_candidate_jobs",
    "training_budget_changed": False,
    "search_space_changed": False,
    "candidate_budget_changed": False,
    "retry_policy_changed": False,
    "v1_runtime_evidence_preserved": True,
    "v2_runtime_evidence_preserved": True,
    "v1_observed_failure": (
        "3.8-hour slices reached epoch 1 but checkpoint_interval=3 "
        "created no resumable checkpoint before self-requeue"
    ),
    "v2_observed_failure": (
        "8-hour request was rejected because polar3 has a 4-hour limit"
    ),
}
FROZEN_V3_QUALIFICATION_CONTRACT = {
    "path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "mask2former_coco2017_three_mode_v3/campaign.v3.json"
    ),
    "file_sha256": (
        "e14545fb0b8ff2d0553e2e04192280f0c381476387056edfa6578c3aad1022c0"
    ),
    "contract_sha256": (
        "5c13acc93ab1fa1e88511ff053a963a023b4d59ded17951fa3af86b0a9efa1e4"
    ),
    "source_commit": "67670fd9727bd06e99d0c651f249f8ba65c051a4",
    "wheel_sha256": (
        "3463187cb76ec3d07c64a21eaf34140e56bf251b46e56ce3c89c33728ee22784"
    ),
    "sdk_commit": "a2e50d0930c3e3785b4b39fa8c3da88b39ff89e5",
    "skills_commit": "2e9c1b25f3c7cb1ae444c75652e36c47eace8229",
    "registry_version": "1.5.0",
    "registry_sha256": (
        "8d40ebde0eec2b7c53f4c698285146c44056d3cc2560ce481cc57b6375b25f74"
    ),
    "qualification_campaign_sha256": (
        "465eb7f496a17696fb9eca43d59095d394ef7dcf59c6563bf268dbd8063a653a"
    ),
    "qualification_campaign_id": (
        "mask2former-coco2017-direct-full-qualification-v3-20260801"
    ),
    "qualification_evidence_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "mask2former_coco2017_ptm_qualification_v3/completion.json"
    ),
    "ptm_stage_manifest_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "mask2former_coco2017_ptm_qualification_v1/ptm_stage_manifest.json"
    ),
    "ptm_stage_manifest_sha256": (
        "3d51ad23d237b8472ebff629dc9ceb7909123c462683f899f4eabb6f4cc3166e"
    ),
    "ptm_stage_content_sha256": (
        "141d14f9b11e3cf81c087d7d05f4e054c885ced1be18554f9832e0cbc9b28bcc"
    ),
    "runtime_overlay": runtime_overlay.contract_record(),
    "walltime_policy": copy.deepcopy(FROZEN_WALLTIME_POLICY),
}
FROZEN_V3_FAILURE_EVIDENCE = {
    "path": FROZEN_V3_QUALIFICATION_CONTRACT["qualification_evidence_path"],
    "file_sha256": (
        "0bb59c56a5c7214ccfc2a2a817ce4c3808620b84a4649f4e0ebbd86c7ef0f41c"
    ),
    "evidence_sha256": (
        "ebc769f9bdb42f7cec35d7a8ea42d4d020e50e7ea520d0ec244e4f7cf92f0811"
    ),
}
SUCCESSOR_WALLTIME_POLICY = {
    **copy.deepcopy(FROZEN_WALLTIME_POLICY),
    "contract_revision": "automl_runtime_v4",
    "supersedes": "qualification_runtime_v3",
    "checkpoint_resume_policy": (
        "same_job_exact_epoch_step_max_with_history_v2"
    ),
    "resume_decision_history_directory": (
        "mask2former_checkpoint_resume_decisions"
    ),
    "resume_decision_history_pattern": (
        "slurm_job_{slurm_job_id}_restart_{restart_count:04d}.json"
    ),
    "slurm_restart_count_environment": "SLURM_RESTART_COUNT",
    "timeout_requeue_cap_environment": "SLURM_MAX_JOB_RETRIES",
    "max_timeout_requeues": FROZEN_SLURM_RETRY_CAP,
    "post_requeue_missing_checkpoint_behavior": "fail_closed",
    "first_post_requeue_decision_recorded": True,
    "resume_history_overwrite_allowed": False,
    "retry_policy_changed": False,
    "retry_policy_implementation_corrected": True,
    "qualification_runtime_v3_evidence_preserved": True,
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
    "timed_scope": "mask2former_model_forward",
    "excluded_scope": [
        "checkpoint_load",
        "disk_io",
        "image_decode",
        "resize_normalize",
        "host_to_device_transfer",
        "instance_postprocessing",
        "mask_serialization",
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
    """The Mask2Former campaign contract is inconsistent."""


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


def _is_lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_runtime_local_eligibility(
    value: Any,
    *,
    runtime: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact v3-evidence-bound in-memory registry policy."""
    if not isinstance(value, Mapping):
        raise CampaignContractError(
            "runtime-local PTM eligibility policy is unavailable"
        )
    policy = copy.deepcopy(dict(value))
    frozen = FROZEN_V3_QUALIFICATION_CONTRACT
    derivation = policy.get("qualification_derivation")
    if derivation is None:
        expected_path = frozen["qualification_evidence_path"]
        expected_campaign_id = frozen["qualification_campaign_id"]
        expected_revision = "qualification_runtime_v3"
    elif (
        isinstance(derivation, Mapping)
        and derivation.get("kind")
        == "immutable_status_metric_deduplication_replay_v1"
        and derivation.get("parent_path")
        == FROZEN_V3_FAILURE_EVIDENCE["path"]
        and derivation.get("parent_file_sha256")
        == FROZEN_V3_FAILURE_EVIDENCE["file_sha256"]
        and derivation.get("parent_evidence_sha256")
        == FROZEN_V3_FAILURE_EVIDENCE["evidence_sha256"]
        and derivation.get("retraining_jobs_submitted") == 0
        and derivation.get("evaluation_jobs_submitted") == 0
        and derivation.get("selection_invoked") is False
        and derivation.get("original_evidence_overwritten") is False
    ):
        expected_path = runtime.get("qualification_evidence_path")
        expected_campaign_id = (
            "mask2former-coco2017-direct-full-qualification-"
            "v3-replay-v1-20260801"
        )
        expected_revision = "qualification_runtime_v3_evidence_replay_v1"
    else:
        raise CampaignContractError(
            "runtime-local qualification derivation is invalid"
        )
    observed_revision = policy.get(
        "qualification_contract_revision",
        expected_revision if derivation is None else None,
    )
    expected_records = {
        record["id"]: record["registry_record_sha256"]
        for record in snapshot["records"]
    }
    required_false = (
        "repository_registry_mutation_allowed",
        "projection_persisted_as_global_registry",
        "failed_arm_promotion_allowed",
        "unsupported_arm_promotion_allowed",
        "agent_override_allowed",
    )
    if (
        policy.get("schema_version") != 2
        or policy.get("kind")
        != "direct_full_gpu_qualification_runtime_local_v2"
        or policy.get("enabled") is not True
        or policy.get("scope") != "campaign_local_in_memory_projection"
        or policy.get("model") != "mask2former"
        or policy.get("task") != "instance_segmentation"
        or policy.get("tao_version") != "7.1.0"
        or policy.get("container_sha256") != FROZEN_SQSH["sha256"]
        or policy.get("base_registry_version")
        != snapshot["registry_version"]
        or policy.get("base_registry_sha256")
        != snapshot["registry_sha256"]
        or policy.get("base_record_sha256_by_checkpoint_id")
        != expected_records
        or policy.get("qualification_path") != expected_path
        or policy.get("qualification_path")
        != runtime.get("qualification_evidence_path")
        or policy.get("qualification_contract_path") != frozen["path"]
        or policy.get("qualification_contract_file_sha256")
        != frozen["file_sha256"]
        or policy.get("qualification_contract_sha256")
        != frozen["contract_sha256"]
        or policy.get("qualification_source_commit")
        != frozen["source_commit"]
        or policy.get("qualification_source_wheel_sha256")
        != frozen["wheel_sha256"]
        or policy.get("qualification_source_sdk_commit")
        != frozen["sdk_commit"]
        or policy.get("qualification_source_skills_commit")
        != frozen["skills_commit"]
        or policy.get("qualification_campaign_sha256")
        != frozen["qualification_campaign_sha256"]
        or policy.get("qualification_campaign_id") != expected_campaign_id
        or observed_revision != expected_revision
        or policy.get("ptm_stage_manifest_path")
        != frozen["ptm_stage_manifest_path"]
        or policy.get("ptm_stage_manifest_sha256")
        != frozen["ptm_stage_manifest_sha256"]
        or policy.get("ptm_stage_content_sha256")
        != frozen["ptm_stage_content_sha256"]
        or policy.get("qualification_runtime_overlay")
        != frozen["runtime_overlay"]
        or policy.get("qualification_walltime_policy")
        != frozen["walltime_policy"]
        or policy.get("eligibility_source_commit")
        != runtime.get("source_commit")
        or policy.get("wheel_sha256") != runtime.get("wheel_sha256")
        or policy.get("sdk_commit") != runtime.get("sdk_commit")
        or policy.get("skills_commit") != runtime.get("skills_commit")
        or any(policy.get(name) is not False for name in required_false)
        or any(
            not _is_lower_sha256(policy.get(name))
            for name in (
                "base_registry_sha256",
                "qualification_file_sha256",
                "qualification_evidence_sha256",
                "qualification_contract_file_sha256",
                "qualification_contract_sha256",
                "qualification_source_wheel_sha256",
                "qualification_campaign_sha256",
                "ptm_stage_manifest_sha256",
                "ptm_stage_content_sha256",
                "wheel_sha256",
            )
        )
    ):
        raise CampaignContractError(
            "runtime-local PTM eligibility contract changed"
        )
    for name in (
        "qualification_source_commit",
        "qualification_source_sdk_commit",
        "qualification_source_skills_commit",
        "eligibility_source_commit",
        "sdk_commit",
        "skills_commit",
    ):
        commit = policy.get(name)
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(
                character not in "0123456789abcdef"
                for character in commit
            )
        ):
            raise CampaignContractError(
                f"runtime-local eligibility {name} is not a Git commit"
            )
    return policy


def mask2former_registry_snapshot() -> dict[str, Any]:
    """Snapshot all official repository-owned Mask2Former PTMs."""
    registry = load_ptm_registry()
    model = registry.to_dict()["models"]["mask2former"]
    records = []
    for record in model["checkpoints"]:
        if (
            record.get("source", {}).get("official") is not True
            or record.get("model_family") != "mask2former"
            or "instance_segmentation"
            not in record.get("task_compatibility", ())
        ):
            raise CampaignContractError(
                "invalid official Mask2Former registry record: "
                f"{record.get('id')}"
            )
        records.append(
            {
                "id": record["id"],
                "status": record["status"],
                "status_reason": record.get("status_reason"),
                "source": copy.deepcopy(record["source"]),
                "expected_size_bytes": record["expected_size_bytes"],
                "checkpoint_target": record["checkpoint_target"],
                "architecture": record["architecture"],
                "backbone": record["backbone"],
                "input_contract": copy.deepcopy(record["input_contract"]),
                "default_spec_overrides": copy.deepcopy(
                    record["default_spec_overrides"]
                ),
                "registry_record_sha256": canonical_sha256(record),
            }
        )
    records.sort(key=lambda item: item["id"])
    if (
        len(records) != 1
        or records[0]["id"]
        != "mask2former.coco.swin_tiny.trainable.v1.0"
    ):
        raise CampaignContractError(
            "the frozen repository inventory must contain the one official "
            "Mask2Former COCO Swin-T checkpoint"
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
    if schema.get("x_tao_schema", {}).get("network_arch") != "mask2former":
        raise CampaignContractError(
            "packaged train schema is not Mask2Former"
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
    return {
        "skill_info_path": str(info_path),
        "skill_info_sha256": sha256_file(info_path),
        "schema_path": str(schema_path),
        "schema_sha256": sha256_file(schema_path),
        "template_path": str(template_path),
        "template_sha256": sha256_file(template_path),
        "explicit_search_parameters": list(SEARCH_PARAMETERS),
        "non_train_fields_excluded": True,
    }


def mode_objective(mode: str) -> dict[str, Any]:
    objectives = [
        {
            "metric": "segm_val_mAP",
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
        "accuracy_metric": "segm_val_mAP",
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
    return {
        name: {
            "valid_min": SEARCH_SPACE[name]["minimum"],
            "valid_max": SEARCH_SPACE[name]["maximum"],
        }
        for name in SEARCH_PARAMETERS
    }


def profile_overrides(dataset_root: str) -> dict[str, Any]:
    """Return the identical full-COCO profile for every mode/PTM arm."""
    if not isinstance(dataset_root, str) or not dataset_root.startswith(
        "/lustre/"
    ):
        raise CampaignContractError(
            "dataset root must be an absolute Lustre path"
        )
    train_images = f"{dataset_root}/images/train2017"
    val_images = f"{dataset_root}/images/val2017"
    train_json = f"{dataset_root}/annotations/instances_train2017.json"
    val_json = f"{dataset_root}/annotations/instances_val2017.json"
    dataset_split = {
        "type": "coco",
        "name": "coco2017_instance",
        "panoptic_json": "",
        "instance_json": "",
        "img_dir": "",
        "panoptic_dir": "",
        "root_dir": "",
        "annot_file": "",
        "batch_size": FROZEN_BATCH_SIZE_PER_REPLICA,
        "num_workers": 8,
        "target_size": [],
    }
    train = copy.deepcopy(dataset_split)
    train.update({"instance_json": train_json, "img_dir": train_images})
    validation = copy.deepcopy(dataset_split)
    validation.update({"instance_json": val_json, "img_dir": val_images})
    return {
        "model_name": "mask2former_coco2017_instance",
        "results_dir": "",
        "wandb": {"enable": False},
        "model": {
            "mode": "instance",
            "sem_seg_head": {"num_classes": 80},
        },
        "dataset": {
            "train": train,
            "val": copy.deepcopy(validation),
            "test": copy.deepcopy(validation),
            "contiguous_id": True,
            "label_map": f"{dataset_root}/tao/label_map_instance.json",
            "augmentation": {
                "train_min_size": [640],
                "train_max_size": 1333,
                "train_crop_size": [640, 640],
                "test_min_size": 640,
                "test_max_size": FROZEN_TEST_MAX_SIZE,
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
            "distributed_strategy": "ddp",
            "activation_checkpoint": False,
            "use_distributed_sampler": False,
            "cudnn": {"benchmark": False, "deterministic": True},
            "optim": {
                "milestones": [2],
                "gamma": 0.1,
            },
        },
    }


def validate_dataset_record(dataset: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "id": "coco2017_full_instance_segmentation",
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
    ptm_inventory = mask2former_registry_snapshot()
    runtime_record = copy.deepcopy(dict(runtime))
    runtime_overlay.validate_contract_record(
        runtime_record.get("tao_pytorch_overlay", {})
    )
    if (
        runtime_record.get("tao_pytorch_overlay")
        != runtime_overlay.successor_contract_record()
    ):
        raise CampaignContractError(
            "successor must use the project-Lustre runtime overlay"
        )
    runtime_local_eligibility = validate_runtime_local_eligibility(
        runtime_record.get("runtime_local_eligibility"),
        runtime=runtime_record,
        snapshot=ptm_inventory,
    )
    if (
        runtime_record.get("partition") != FROZEN_SLURM_PARTITION
        or runtime_record.get("time_hours") != FROZEN_SLURM_TIME_HOURS
        or runtime_record.get("timeout_hours")
        != FROZEN_SLURM_TIMEOUT_HOURS
        or runtime_record.get("use_requeue")
        is not FROZEN_SLURM_USE_REQUEUE
        or runtime_record.get("walltime_policy")
        != SUCCESSOR_WALLTIME_POLICY
        or runtime_record.get("max_job_retries")
        != FROZEN_SLURM_RETRY_CAP
    ):
        raise CampaignContractError(
            "runtime must use the bounded v4 requeue/resume policy"
        )
    value = {
        "schema_version": 2,
        "campaign_id": campaign_id,
        "model": "mask2former",
        "network_arch": "mask2former",
        "task": "instance_segmentation",
        "primary_accuracy_metric": "segm_val_mAP",
        "dataset": dataset_record,
        "runtime": runtime_record,
        "sqsh": copy.deepcopy(FROZEN_SQSH),
        "schema": schema,
        "ptm_inventory": ptm_inventory,
        "metric_contract": {
            "required": "segm_val_mAP",
            "validation_reported_metric": "segm_val_mAP",
            "standalone_reported_metric": "segm_test_mAP",
            "standalone_reported_metric50": "segm_test_mAP50",
            "standalone_canonical_objective": "segm_val_mAP",
            "direction": "maximize",
            "scale": "fraction",
            "task_correct": True,
            "semantic_miou_is_not_an_alias": True,
            "known_repository_state": (
                "runtime_fix_available_pending_gpu_qualification"
            ),
            "failure_policy": "fail_closed_without_coco_mask_ap",
        },
        "qualification_policy": {
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
            "required_metric": "segm_val_mAP",
            "standalone_reported_metric": "segm_test_mAP",
            "standalone_objective_binding": {
                "reported_metric": "segm_test_mAP",
                "canonical_metric": "segm_val_mAP",
            },
            "registry_bypass_allowed": False,
            "runtime_local_eligibility": copy.deepcopy(
                runtime_local_eligibility
            ),
            "qualification_evidence_path": runtime[
                "qualification_evidence_path"
            ],
            "ptm_stage_manifest_path": runtime["ptm_stage_manifest_path"],
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
            "tao_pytorch_overlay_injection": "PYTHONPATH",
            "installed_tao_package_mutated": False,
            "slurm_self_requeue": FROZEN_SLURM_USE_REQUEUE,
            "checkpoint_interval_epochs": (
                FROZEN_CHECKPOINT_INTERVAL_EPOCHS
            ),
            "checkpoint_resume_policy": (
                "same_job_exact_epoch_step_max_with_history_v2"
            ),
            "timeout_requeue_cap": FROZEN_SLURM_RETRY_CAP,
            "timeout_requeue_cap_environment": "SLURM_MAX_JOB_RETRIES",
            "resume_decision_history": True,
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
                "accuracy": "all_runtime_supported",
                "latency": "all_runtime_supported",
                "multi_objective": "all_runtime_supported",
            },
        },
        "validation_sanity_gate": {
            "metric": "segm_val_mAP",
            "minimum": FROZEN_VALIDATION_SANITY_MIN_MASK_AP,
            "role": "experiment_correctness_gate_not_product_selection",
            "rationale": (
                "For COCO 2017 80-class instance segmentation, mask AP below "
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
        value.get("schema_version") != 2
        or not isinstance(campaign_id, str)
        or not campaign_id
        or value.get("model") != "mask2former"
        or value.get("network_arch") != "mask2former"
        or value.get("task") != "instance_segmentation"
        or value.get("primary_accuracy_metric") != "segm_val_mAP"
        or value.get("execution", {}).get("cpu_runs") != 0
        or value.get("execution", {}).get("smoke_runs") != 0
        or value.get("execution", {}).get("gpus_per_child") != 8
        or value.get("execution", {}).get("container_mode")
        != "pinned_sqsh"
        or value.get("execution", {}).get(
            "tao_pytorch_overlay_injection"
        )
        != "PYTHONPATH"
        or value.get("execution", {}).get(
            "installed_tao_package_mutated"
        )
        is not False
        or value.get("execution", {}).get("slurm_self_requeue")
        is not FROZEN_SLURM_USE_REQUEUE
        or value.get("execution", {}).get(
            "checkpoint_interval_epochs"
        )
        != FROZEN_CHECKPOINT_INTERVAL_EPOCHS
        or value.get("execution", {}).get(
            "checkpoint_resume_policy"
        )
        != "same_job_exact_epoch_step_max_with_history_v2"
        or value.get("execution", {}).get("timeout_requeue_cap")
        != FROZEN_SLURM_RETRY_CAP
        or value.get("execution", {}).get(
            "timeout_requeue_cap_environment"
        )
        != "SLURM_MAX_JOB_RETRIES"
        or value.get("execution", {}).get("resume_decision_history")
        is not True
        or value.get("search", {}).get("space") != SEARCH_SPACE
        or value.get("modes") != expected_modes
        or value.get("metric_contract", {}).get(
            "semantic_miou_is_not_an_alias"
        )
        is not True
        or value.get("metric_contract", {}).get(
            "validation_reported_metric"
        )
        != "segm_val_mAP"
        or value.get("metric_contract", {}).get(
            "standalone_reported_metric"
        )
        != "segm_test_mAP"
        or value.get("metric_contract", {}).get(
            "standalone_reported_metric50"
        )
        != "segm_test_mAP50"
        or value.get("metric_contract", {}).get(
            "standalone_canonical_objective"
        )
        != "segm_val_mAP"
        or value.get("metric_contract", {}).get(
            "known_repository_state"
        )
        != "runtime_fix_available_pending_gpu_qualification"
        or value.get("qualification_policy", {}).get(
            "standalone_objective_binding"
        )
        != {
            "reported_metric": "segm_test_mAP",
            "canonical_metric": "segm_val_mAP",
        }
        or value.get("qualification_policy", {}).get(
            "ptm_stage_manifest_path"
        )
        != value.get("runtime", {}).get("ptm_stage_manifest_path")
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
    runtime_overlay.validate_contract_record(
        runtime.get("tao_pytorch_overlay", {})
    )
    if (
        runtime.get("tao_pytorch_overlay")
        != runtime_overlay.successor_contract_record()
    ):
        raise CampaignContractError(
            "successor must use the project-Lustre runtime overlay"
        )
    snapshot = mask2former_registry_snapshot()
    validate_runtime_local_eligibility(
        runtime.get("runtime_local_eligibility"),
        runtime=runtime,
        snapshot=snapshot,
    )
    if (
        runtime.get("partition") != FROZEN_SLURM_PARTITION
        or runtime.get("time_hours") != FROZEN_SLURM_TIME_HOURS
        or runtime.get("timeout_hours") != FROZEN_SLURM_TIMEOUT_HOURS
        or runtime.get("use_requeue") is not FROZEN_SLURM_USE_REQUEUE
        or runtime.get("walltime_policy") != SUCCESSOR_WALLTIME_POLICY
        or runtime.get("max_job_retries") != FROZEN_SLURM_RETRY_CAP
    ):
        raise CampaignContractError(
            "runtime must use the bounded v4 requeue/resume policy"
        )
    search = value.get("search", {})
    if (
        value.get("ptm_inventory") != snapshot
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
    "FROZEN_CALIBRATION_POINTS_PER_ARM",
    "FROZEN_CANDIDATE_BUDGET",
    "FROZEN_CHECKPOINT_INTERVAL_EPOCHS",
    "FROZEN_HARDWARE",
    "FROZEN_LATENCY_RETENTION",
    "FROZEN_LATENCY_TOLERANCE_MS",
    "FROZEN_SEARCH_SEED",
    "FROZEN_SLURM_RETRY_CAP",
    "FROZEN_SLURM_PARTITION",
    "FROZEN_SLURM_TIME_HOURS",
    "FROZEN_SLURM_TIMEOUT_HOURS",
    "FROZEN_SLURM_USE_REQUEUE",
    "FROZEN_SQSH",
    "FROZEN_TEST_MAX_SIZE",
    "FROZEN_TRAINING_EPOCHS",
    "FROZEN_VALIDATION_SANITY_MIN_MASK_AP",
    "FROZEN_V3_QUALIFICATION_CONTRACT",
    "FROZEN_WALLTIME_POLICY",
    "SUCCESSOR_WALLTIME_POLICY",
    "LATENCY_PROTOCOL",
    "MODES",
    "SEARCH_PARAMETERS",
    "SEARCH_SPACE",
    "SELECTION_FLAGS",
    "build_preregistered_contract",
    "custom_ranges",
    "mask2former_registry_snapshot",
    "mode_objective",
    "mode_settings",
    "profile_overrides",
    "sha256_file",
    "validate_contract",
    "validate_dataset_record",
    "validate_packaged_train_schema",
    "validate_runtime_local_eligibility",
]
