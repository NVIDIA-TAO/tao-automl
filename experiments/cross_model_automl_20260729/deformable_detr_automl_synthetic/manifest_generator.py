#!/usr/bin/env python3

"""Build and verify the Deformable DETR objective-aware campaign manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tao_automl.ptm_registry import canonical_sha256, load_ptm_registry

try:
    from .qualification_evidence import (
        EXPECTED_PTMS,
        audit_qualification_evidence,
        sha256_file,
    )
except ImportError:  # pragma: no cover - direct script execution
    from qualification_evidence import (  # type: ignore[no-redef]
        EXPECTED_PTMS,
        audit_qualification_evidence,
        sha256_file,
    )


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
DATASET_MANIFEST = (
    HERE.parent
    / "datasets"
    / "tao_od_synthetic_full_dino_coco"
    / "manifest.v1.json"
)
DATASET_INTEGRITY = DATASET_MANIFEST.with_name("integrity.v1.json")
DEFAULT_OUTPUT = HERE / "campaign.v1.json"
LATENCY_INPUT_MANIFEST = HERE / "latency_input.v1.json"
WHEEL_PATH = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/wheel/1919228616b8/"
    "nvidia_tao_automl-0.1.0-py3-none-any.whl"
)
SKILL_DIR = Path(
    "/localhome/local-rarunachalam/.tao/worktrees/"
    "tao-skills-release-7.1.0/skills/models/tao-train-deformable-detr"
)
SDK_DIR = Path(
    "/localhome/local-rarunachalam/.tao/worktrees/tao-sdk-slurm-a2e50d0"
)
SQSH_PATH = (
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
    "nvcr.io_nvstaging_tao_tao-toolkit-pyt_7.1.0-rc-245-multiarch.sqsh"
)
SQSH_SHA256 = "e36640f9ae7a03bc80828cf7de93bd6bdbbb0fecf509a71a243be0ab5b497fc2"
SQSH_SIZE_BYTES = 28860358656
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
SEARCH_SPACE = {
    "model.enc_layers": {
        "type": "integer",
        "minimum": 3,
        "maximum": 6,
        "values": [3, 4, 5, 6],
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
    "train.optim.lr": {
        "type": "float",
        "minimum": 1.0e-5,
        "maximum": 5.0e-4,
        "scale": "log",
    },
    "train.optim.weight_decay": {
        "type": "float",
        "minimum": 1.0e-5,
        "maximum": 1.0e-3,
        "scale": "log",
    },
}
SEARCH_PARAMETERS = tuple(SEARCH_SPACE)
FROZEN_SEARCH_SEED = 271828
FROZEN_TRAINING_SEED = 1234
FROZEN_CANDIDATE_BUDGET = 20
FROZEN_TRAINING_EPOCHS = 10
FROZEN_CALIBRATION_POINTS = 8
FROZEN_LATENCY_RETENTION = 0.90
FROZEN_LATENCY_TOLERANCE_MS = 0.73553775
FROZEN_HARDWARE = {
    "gpu_name": "NVIDIA A100-SXM4-80GB",
    "compute_capability": "8.0",
    "total_memory_bytes": 85174583296,
}


class ManifestError(ValueError):
    """The campaign manifest is incomplete or internally inconsistent."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def manifest_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def frozen_campaign_signature(manifest: Mapping[str, Any]) -> str:
    """Hash preregistered intent while excluding runtime-readiness projections."""
    validate_manifest(manifest)
    value = copy.deepcopy(dict(manifest))
    value.pop("manifest_sha256", None)
    value["source"].pop("commit", None)
    value["execution"].pop("submission_ready", None)
    value["execution"].pop("blocked_before_sdk_construction", None)
    for ptm in value["ptms"]:
        ptm.pop("registry_status", None)
        ptm.pop("registry_record_sha256", None)
    decision = value["qualification_evidence"]
    decision.pop("blockers", None)
    decision.pop("runtime_ready", None)
    decision.pop("decision_sha256", None)
    return manifest_sha256(value)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _mode_objective(mode: str) -> dict[str, Any]:
    acquisition = {
        "accuracy": "expected_improvement",
        "latency": "constrained_expected_improvement",
        "multi_objective": "parego_expected_improvement",
    }[mode]
    selection_policy = {
        "accuracy": "highest_valid_accuracy",
        "latency": "equivalent_fastest_accuracy_tiebreak",
        "multi_objective": "normalized_augmented_chebyshev",
    }[mode]
    result: dict[str, Any] = {
        "mode": mode,
        "acquisition": acquisition,
        "selection_policy": selection_policy,
        "objectives": [
            {"metric": "mAP50", "role": "accuracy", "direction": "maximize"},
            {"metric": "latency_ms", "role": "latency", "direction": "minimize"},
        ],
        "multi_objective_min_accuracy": None,
    }
    if mode == "latency":
        result["latency_accuracy_retention"] = {
            "type": "relative",
            "retained_fraction": FROZEN_LATENCY_RETENTION,
            "reference": "best_observed_within_job",
            "reference_updates": "monotonic",
            "terminal_reference": "terminal_archive_accuracy_winner",
        }
    else:
        result["latency_accuracy_retention"] = None
    return result


def _latency_protocol(dataset: Mapping[str, Any]) -> dict[str, Any]:
    descriptor_source = json.loads(
        LATENCY_INPUT_MANIFEST.read_text(encoding="utf-8")
    )
    images = descriptor_source.get("images")
    if (
        descriptor_source.get("dataset_id") != dataset["dataset_id"]
        or descriptor_source.get("source_annotation_sha256")
        != dataset["splits"]["validation"]["annotation"]["sha256"]
        or not isinstance(images, list)
        or len(images) != 16
        or any(
            not isinstance(item.get("id"), int)
            or not isinstance(item.get("file_name"), str)
            or item.get("width", 0) < 1
            or item.get("height", 0) < 1
            for item in images
        )
    ):
        raise ManifestError("latency input manifest is inconsistent")
    descriptor = copy.deepcopy(descriptor_source)
    descriptor["manifest_path"] = str(LATENCY_INPUT_MANIFEST)
    descriptor["manifest_file_sha256"] = sha256_file(
        LATENCY_INPUT_MANIFEST
    )
    descriptor["validation_image_ids"] = [item["id"] for item in images]
    descriptor_sha256 = canonical_sha256(descriptor)
    return {
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
            "model_forward_plus_deformable_detr_gpu_postprocess"
        ),
        "excluded_scope": [
            "checkpoint_load",
            "disk_io",
            "decode_resize_normalize",
            "host_to_device_transfer",
            "coco_accumulation",
            "distributed_gather",
        ],
        "synchronization": (
            "accelerator_sync_before_and_after_each_sample"
        ),
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
        "input_descriptor": descriptor,
        "input_descriptor_sha256": descriptor_sha256,
    }


def _resolve_ptms() -> list[dict[str, Any]]:
    registry = load_ptm_registry()
    records = {
        item["id"]: item
        for item in registry.to_dict()["models"]["deformable_detr"][
            "checkpoints"
        ]
    }
    base = json.loads(
        (
            HERE.parent
            / "deformable_detr_synthetic_campaign"
            / "campaign.v1.json"
        ).read_text(encoding="utf-8")
    )
    runtime_by_id = {item["id"]: item for item in base["ptms"]}
    if set(runtime_by_id) != set(EXPECTED_PTMS):
        raise ManifestError("sealed PTM runtime inventory changed")
    result = []
    for checkpoint_id in EXPECTED_PTMS:
        record = records.get(checkpoint_id)
        runtime = runtime_by_id[checkpoint_id]
        if not isinstance(record, Mapping):
            raise ManifestError(f"registry record is missing: {checkpoint_id}")
        artifact = runtime["artifact"]
        local_path = Path(artifact["local_source_path"])
        if (
            not local_path.is_file()
            or local_path.stat().st_size != artifact["size_bytes"]
            or sha256_file(local_path) != artifact["sha256"]
        ):
            raise ManifestError(f"local PTM identity changed: {checkpoint_id}")
        spec_path = REPOSITORY / "src/tao_automl" / record[
            "checkpoint_spec_file"
        ]["path"]
        if (
            not spec_path.is_file()
            or sha256_file(spec_path)
            != record["checkpoint_spec_file"]["sha256"]
        ):
            raise ManifestError(f"PTM spec identity changed: {checkpoint_id}")
        result.append(
            {
                "id": checkpoint_id,
                "registry_status": record["status"],
                "registry_record_sha256": canonical_sha256(record),
                "official_source": record["source"]["immutable_identity"],
                "checkpoint_target": record["checkpoint_target"],
                "backbone": record["backbone"],
                "default_spec_overrides": copy.deepcopy(
                    record["default_spec_overrides"]
                ),
                "artifact": copy.deepcopy(artifact),
                "checkpoint_spec": {
                    "path": str(spec_path),
                    "sha256": sha256_file(spec_path),
                },
                "conditional_search_space": copy.deepcopy(SEARCH_SPACE),
            }
        )
    return result


def build_manifest() -> dict[str, Any]:
    dataset = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    integrity = json.loads(DATASET_INTEGRITY.read_text(encoding="utf-8"))
    decision = audit_qualification_evidence()
    skill_info = SKILL_DIR / "references/skill_info.yaml"
    schema = SKILL_DIR / "schemas/train.schema.json"
    template_train = SKILL_DIR / "references/spec_template_train.yaml"
    template_evaluate = SKILL_DIR / "references/spec_template_evaluate.yaml"
    template_export = SKILL_DIR / "references/spec_template_export.yaml"
    required_files = (
        WHEEL_PATH,
        skill_info,
        schema,
        template_train,
        template_evaluate,
        template_export,
    )
    for path in required_files:
        if not path.is_file():
            raise ManifestError(f"required campaign artifact is missing: {path}")
    ptms = _resolve_ptms()
    qualified_inputs = {
        item["checkpoint_id"]: item["qualified_input_checkpoint_sha256"]
        for item in decision.records
    }
    if (
        set(qualified_inputs) != {item["id"] for item in ptms}
        or any(
            item["artifact"]["sha256"] != qualified_inputs[item["id"]]
            for item in ptms
        )
    ):
        raise ManifestError(
            "sealed PTM artifacts differ from shared-data qualification inputs"
        )
    search_hash = canonical_sha256(SEARCH_SPACE)
    campaign_id = "deformable-detr-synthetic-objective-aware-v1"
    modes = []
    for mode in MODES:
        objective = _mode_objective(mode)
        modes.append(
            {
                "mode": mode,
                "job_id": f"{campaign_id}-{mode}",
                "session_id": f"{campaign_id}-{mode}",
                "observation_namespace": (
                    f"{campaign_id}-{mode}-observations"
                ),
                "observation_sharing": False,
                "initial_observation_ids": [],
                "search_seed": FROZEN_SEARCH_SEED,
                "ptm_policy": "all_qualified_explicit",
                "allowed_ptm_ids": list(EXPECTED_PTMS),
                "search_space_sha256": search_hash,
                "objective": objective,
                "objective_sha256": canonical_sha256(objective),
            }
        )
    document = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "model": "deformable_detr",
        "task": "object_detection",
        "execution": {
            "kind": "objective_aware_three_mode_search",
            "cpu_runs": 0,
            "smoke_runs": 0,
            "local_model_runs": 0,
            "shared_archive": False,
            "independent_mode_jobs": True,
            "submission_ready": decision.runtime_ready,
            "blocked_before_sdk_construction": not decision.runtime_ready,
        },
        "source": {
            "repository": str(REPOSITORY),
            "commit": _git(REPOSITORY, "rev-parse", "HEAD"),
            "launch_head_policy": "clean_descendant",
            "wheel_path": str(WHEEL_PATH),
            "wheel_sha256": sha256_file(WHEEL_PATH),
        },
        "runtime": {
            "platform": "slurm",
            "skill_dir": str(SKILL_DIR),
            "skill_revision": _git(SKILL_DIR, "rev-parse", "HEAD"),
            "sdk_dir": str(SDK_DIR),
            "sdk_revision": _git(SDK_DIR, "rev-parse", "HEAD"),
            "skill_info_sha256": sha256_file(skill_info),
            "train_schema_sha256": sha256_file(schema),
            "train_template_sha256": sha256_file(template_train),
            "evaluate_template_sha256": sha256_file(template_evaluate),
            "export_template_sha256": sha256_file(template_export),
            "image_reference": (
                "nvcr.io/nvstaging/tao/"
                "tao-toolkit-pyt:7.1.0-rc-245-multiarch"
            ),
            "sqsh_path": SQSH_PATH,
            "sqsh_sha256": SQSH_SHA256,
            "sqsh_size_bytes": SQSH_SIZE_BYTES,
            "slurm_use_sqsh": True,
            "nodes_per_child": 1,
            "gpus_per_child": 8,
            "distributed_workers_per_child": 8,
            "partition": "polar3",
            "account": "edgeai_tao-ptm_image-foundation-model-clip",
            "container_mounts": "/lustre",
            "base_results_dir": (
                "/lustre/fsw/portfolios/edgeai/users/rarunachalam"
            ),
            "time_hours": 4.0,
            "timeout_hours": 3.8,
            "max_infrastructure_retries": 3,
            "hardware_contract": copy.deepcopy(FROZEN_HARDWARE),
        },
        "launcher_integrity": {
            name: {
                "path": str(HERE / filename),
                "sha256": sha256_file(HERE / filename),
            }
            for name, filename in (
                ("manifest_generator", "manifest_generator.py"),
                ("qualification_evidence", "qualification_evidence.py"),
                ("campaign_controller", "run_campaign.py"),
                ("latency_worker", "deformable_detr_latency_worker.py"),
                ("latency_input", "latency_input.v1.json"),
            )
        },
        "dataset": {
            **copy.deepcopy(dataset),
            "manifest_path": str(DATASET_MANIFEST),
            "manifest_sha256": sha256_file(DATASET_MANIFEST),
            "integrity_path": str(DATASET_INTEGRITY),
            "integrity_sha256": sha256_file(DATASET_INTEGRITY),
            "integrity_document_sha256": canonical_sha256(integrity),
        },
        "ptms": ptms,
        "qualification_evidence": decision.to_dict(),
        "search": {
            "algorithm": "bayesian",
            "ptm_representation": "hierarchical_nonordinal_arms",
            "candidate_budget_per_mode": FROZEN_CANDIDATE_BUDGET,
            "training_epochs": FROZEN_TRAINING_EPOCHS,
            "search_seed": FROZEN_SEARCH_SEED,
            "training_seed": FROZEN_TRAINING_SEED,
            "calibration_points": FROZEN_CALIBRATION_POINTS,
            "parameters": list(SEARCH_PARAMETERS),
            "space": copy.deepcopy(SEARCH_SPACE),
            "space_sha256": search_hash,
            "latency_accuracy_retention": FROZEN_LATENCY_RETENTION,
            "latency_practical_tolerance_ms": (
                FROZEN_LATENCY_TOLERANCE_MS
            ),
            "manual_candidate_injection": False,
            "post_result_domain_changes": False,
        },
        "first_candidate_gate": {
            "candidate_index_per_mode": 0,
            "initial_jobs": 3,
            "parallel_across_modes": True,
            "required_mode_results": list(MODES),
            "success_criteria": [
                "training_complete",
                "standalone_evaluation_complete",
                "finite_mAP50",
                "stabilized_latency_quality_gate_passed",
                "recommendation_audit_complete",
                "artifact_provenance_complete",
            ],
            "automatic_release": True,
            "remaining_candidates_per_mode": 19,
            "release_action": (
                "continue_same_frozen_runner_without_confirmation"
            ),
            "failure_action": (
                "preserve_failure_and_halt_before_remaining_budget"
            ),
        },
        "latency_protocol": _latency_protocol(dataset),
        "modes": modes,
        "agent_intervention_flags": {name: False for name in AGENT_FLAGS},
        "selection_isolation_flags": {
            name: False for name in SELECTION_FLAGS
        },
        "failure_policy": {
            "preserve_failed_recommendations": True,
            "silent_replacement": False,
            "maximum_terminal_failures_per_mode": 3,
            "maximum_infrastructure_retries_per_child": 3,
            "manual_ptm_substitution": False,
        },
    }
    validate_manifest(document, require_seal=False)
    return document


def _require_finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ManifestError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ManifestError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ManifestError(f"{name} must be finite")
    return result


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    require_seal: bool = True,
) -> None:
    value = copy.deepcopy(dict(manifest))
    supplied_hash = value.pop("manifest_sha256", None)
    if require_seal and supplied_hash is None:
        raise ManifestError("manifest_sha256 is required")
    if supplied_hash is not None and supplied_hash != manifest_sha256(value):
        raise ManifestError("manifest_sha256 does not match the document")
    if value.get("schema_version") != 1:
        raise ManifestError("unsupported schema_version")
    if value.get("model") != "deformable_detr":
        raise ManifestError("model must be deformable_detr")
    execution = value.get("execution", {})
    if (
        execution.get("cpu_runs") != 0
        or execution.get("smoke_runs") != 0
        or execution.get("local_model_runs") != 0
        or execution.get("shared_archive") is not False
        or execution.get("independent_mode_jobs") is not True
    ):
        raise ManifestError("direct independent execution contract changed")
    search = value.get("search", {})
    if (
        search.get("candidate_budget_per_mode") != 20
        or search.get("training_epochs") != 10
        or search.get("parameters") != list(SEARCH_PARAMETERS)
        or search.get("space") != SEARCH_SPACE
        or search.get("space_sha256") != canonical_sha256(SEARCH_SPACE)
        or _require_finite(
            search.get("latency_accuracy_retention"),
            "latency_accuracy_retention",
        )
        != 0.90
    ):
        raise ManifestError("frozen search contract changed")
    runtime = value.get("runtime", {})
    if (
        runtime.get("slurm_use_sqsh") is not True
        or runtime.get("sqsh_path") != SQSH_PATH
        or runtime.get("nodes_per_child") != 1
        or runtime.get("gpus_per_child") != 8
        or any(
            not isinstance(runtime.get(field), str)
            or len(runtime[field]) != 64
            for field in (
                "train_template_sha256",
                "evaluate_template_sha256",
                "export_template_sha256",
            )
        )
    ):
        raise ManifestError("SLURM/SQSH resource contract changed")
    launcher_integrity = value.get("launcher_integrity")
    if (
        not isinstance(launcher_integrity, Mapping)
        or set(launcher_integrity)
        != {
            "manifest_generator",
            "qualification_evidence",
            "campaign_controller",
            "latency_worker",
            "latency_input",
        }
        or any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
            or len(item["sha256"]) != 64
            for item in launcher_integrity.values()
        )
    ):
        raise ManifestError("launcher integrity contract changed")
    ptms = value.get("ptms")
    if (
        not isinstance(ptms, list)
        or tuple(item.get("id") for item in ptms) != EXPECTED_PTMS
    ):
        raise ManifestError("PTM inventory changed")
    decision = value.get("qualification_evidence", {})
    blockers = decision.get("blockers")
    runtime_ready = decision.get("runtime_ready")
    if runtime_ready != (not blockers):
        raise ManifestError("qualification readiness contradicts blockers")
    if execution.get("submission_ready") != runtime_ready:
        raise ManifestError("submission readiness bypasses qualification gate")
    modes = value.get("modes")
    if (
        not isinstance(modes, list)
        or tuple(item.get("mode") for item in modes) != MODES
        or len({item.get("observation_namespace") for item in modes}) != 3
        or any(item.get("initial_observation_ids") != [] for item in modes)
        or any(item.get("observation_sharing") is not False for item in modes)
    ):
        raise ManifestError("independent mode contract changed")
    expected_acquisitions = {
        "accuracy": "expected_improvement",
        "latency": "constrained_expected_improvement",
        "multi_objective": "parego_expected_improvement",
    }
    for mode in modes:
        objective = mode.get("objective", {})
        if objective.get("acquisition") != expected_acquisitions[mode["mode"]]:
            raise ManifestError("objective-aware acquisition changed")
        if mode["mode"] != "latency" and objective.get(
            "latency_accuracy_retention"
        ) is not None:
            raise ManifestError("latency retention leaked into another mode")
        if mode.get("objective_sha256") != canonical_sha256(objective):
            raise ManifestError("objective hash changed")
    protocol = value.get("latency_protocol", {})
    if (
        protocol.get("warmup_iterations") != 50
        or protocol.get("timed_iterations") != 100
        or protocol.get("repeated_rounds") != 5
        or protocol.get("expected_replicas") != 8
        or protocol.get("raw_samples_per_candidate") != 4000
    ):
        raise ManifestError("stabilized latency protocol changed")
    gate = value.get("first_candidate_gate", {})
    if (
        gate.get("automatic_release") is not True
        or gate.get("remaining_candidates_per_mode") != 19
        or gate.get("failure_action")
        != "preserve_failure_and_halt_before_remaining_budget"
    ):
        raise ManifestError("automatic first-candidate gate changed")
    for field, names in (
        ("agent_intervention_flags", AGENT_FLAGS),
        ("selection_isolation_flags", SELECTION_FLAGS),
    ):
        flags = value.get(field)
        if flags != {name: False for name in names}:
            raise ManifestError(f"{field} must remain false")


def seal_manifest(document: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(document))
    value.pop("manifest_sha256", None)
    validate_manifest(value, require_seal=False)
    value["manifest_sha256"] = manifest_sha256(value)
    validate_manifest(value)
    return value


def load_manifest(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    expected = seal_manifest(build_manifest())
    if args.verify:
        observed = load_manifest(args.output)
        if frozen_campaign_signature(observed) != frozen_campaign_signature(
            expected
        ):
            raise SystemExit(
                "sealed campaign intent differs from current source inputs"
            )
        source_repository = Path(observed["source"]["repository"])
        subprocess.run(
            [
                "git",
                "-C",
                str(source_repository),
                "merge-base",
                "--is-ancestor",
                observed["source"]["commit"],
                expected["source"]["commit"],
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        print(observed["manifest_sha256"])
        return 0
    if args.output.exists():
        raise SystemExit(f"refusing to replace existing manifest: {args.output}")
    args.output.write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(expected["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
