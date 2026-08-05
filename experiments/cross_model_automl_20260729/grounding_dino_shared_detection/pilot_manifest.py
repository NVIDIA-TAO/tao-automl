#!/usr/bin/env python3

"""Build the portable, qualification-driven Grounding DINO pilot manifest."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml

from tao_automl.ptm_registry import canonical_sha256

try:
    from .contract import (
        AGENT_FLAGS,
        MODEL_ID,
        MODES,
        SELECTION_FLAGS,
        PreparationError,
        read_json,
        sha256_file,
    )
    from .dataset_conversion import validate_conversion_manifest
    from .pilot_qualification import (
        PilotQualificationDecision,
        audit_pilot_qualification,
    )
except ImportError:  # pragma: no cover - direct script execution
    from contract import (  # type: ignore[no-redef]
        AGENT_FLAGS,
        MODEL_ID,
        MODES,
        SELECTION_FLAGS,
        PreparationError,
        read_json,
        sha256_file,
    )
    from dataset_conversion import validate_conversion_manifest
    from pilot_qualification import (  # type: ignore[no-redef]
        PilotQualificationDecision,
        audit_pilot_qualification,
    )


HERE = Path(__file__).resolve().parent
DEFAULT_INPUTS = HERE / "pilot.inputs.v1.json"
DEFAULT_OUTPUT = HERE / "pilot.campaign.v1.json"
SEARCH_PARAMETERS = (
    "model.enc_layers",
    "model.dec_layers",
    "model.num_select",
    "train.optim.lr",
    "train.optim.lr_backbone",
    "train.optim.weight_decay",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def source_state(inputs: Mapping[str, Any]) -> dict[str, Any]:
    source = inputs["source"]
    repository = Path(source["repository"]).resolve()
    head = _git(repository, "rev-parse", "HEAD")
    minimum = source["minimum_ancestor_commit"]
    ancestor = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", minimum, head],
        check=False,
        capture_output=True,
        timeout=30,
    ).returncode == 0
    return {
        "repository": str(repository),
        "commit": head,
        "minimum_ancestor_commit": minimum,
        "minimum_ancestor_satisfied": ancestor,
        "clean": not bool(_git(repository, "status", "--porcelain")),
    }


def _mode_objective(
    mode: str,
    search: Mapping[str, Any],
) -> dict[str, Any]:
    common = [
        {"metric": "mAP50", "direction": "maximize", "role": "accuracy"},
        {"metric": "latency_ms", "direction": "minimize", "role": "latency"},
    ]
    if mode == "accuracy":
        return {
            "mode": mode,
            "acquisition": "expected_improvement",
            "objectives": common,
            "latency_accuracy_retention": None,
            "multi_objective_min_accuracy": None,
            "selection_policy": "highest_valid_accuracy",
        }
    if mode == "latency":
        return {
            "mode": mode,
            "acquisition": "constrained_expected_improvement",
            "objectives": common,
            "latency_accuracy_retention": {
                "type": "relative",
                "retained_fraction": search["latency_accuracy_retention"],
                "reference": "best_observed_within_job",
                "reference_updates": "monotonic",
                "terminal_reference": "terminal_archive_accuracy_winner",
            },
            "multi_objective_min_accuracy": None,
            "selection_policy": "equivalent_fastest_accuracy_tiebreak",
        }
    if mode == "multi_objective":
        return {
            "mode": mode,
            "acquisition": "parego_expected_improvement",
            "objectives": common,
            "latency_accuracy_retention": None,
            "multi_objective_min_accuracy": None,
            "selection_policy": "normalized_augmented_chebyshev",
        }
    raise PreparationError(f"unsupported pilot mode {mode!r}")


def _schema_node(schema: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    node: Any = schema
    for token in path.split("."):
        properties = node.get("properties") if isinstance(node, Mapping) else None
        if not isinstance(properties, Mapping) or token not in properties:
            raise PreparationError(
                f"search parameter {path!r} is absent from train schema"
            )
        node = properties[token]
    if not isinstance(node, Mapping):
        raise PreparationError(f"search parameter {path!r} is malformed")
    return node


def _validate_search_space(
    search: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> None:
    if tuple(search.get("parameters", ())) != SEARCH_PARAMETERS:
        raise PreparationError("pilot search parameters changed")
    space = search.get("space")
    if not isinstance(space, Mapping) or set(space) != set(SEARCH_PARAMETERS):
        raise PreparationError("pilot search domains are incomplete")
    for parameter in SEARCH_PARAMETERS:
        schema_node = _schema_node(schema, parameter)
        if schema_node.get("automl_enabled") is not True:
            raise PreparationError(
                f"{parameter} is not AutoML-enabled by the packaged schema"
            )
        domain = space[parameter]
        if not isinstance(domain, Mapping):
            raise PreparationError(f"{parameter} domain must be a mapping")
        if "values" in domain:
            values = domain["values"]
            if (
                not isinstance(values, list)
                or not values
                or len(set(values)) != len(values)
            ):
                raise PreparationError(
                    f"{parameter} discrete domain is invalid"
                )
            minimum = schema_node.get("minimum")
            maximum = schema_node.get("maximum")
            if minimum is not None and any(value < minimum for value in values):
                raise PreparationError(
                    f"{parameter} value is below schema minimum"
                )
            if maximum is not None and any(value > maximum for value in values):
                raise PreparationError(
                    f"{parameter} value is above schema maximum"
                )
        else:
            minimum = domain.get("minimum")
            maximum = domain.get("maximum")
            if (
                isinstance(minimum, bool)
                or isinstance(maximum, bool)
                or not isinstance(minimum, (int, float))
                or not isinstance(maximum, (int, float))
                or not math.isfinite(float(minimum))
                or not math.isfinite(float(maximum))
                or not 0 < minimum < maximum
            ):
                raise PreparationError(
                    f"{parameter} continuous domain is invalid"
                )
            if domain.get("scale") != "log":
                raise PreparationError(
                    f"{parameter} must retain its preregistered log scale"
                )
    if max(space["model.num_select"]["values"]) > 900:
        raise PreparationError(
            "model.num_select exceeds the qualified PTM num_queries contract"
        )


def _latency_protocol(
    inputs: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    descriptor_path: Path,
) -> dict[str, Any]:
    latency = inputs["latency"]
    replicas = inputs["runtime"]["gpus_per_child"]
    raw_samples = (
        replicas
        * latency["repeated_rounds"]
        * latency["timed_iterations"]
    )
    return {
        "warmup_iterations": latency["warmup_iterations"],
        "timed_iterations": latency["timed_iterations"],
        "repeated_rounds": latency["repeated_rounds"],
        "tail_percentile": latency["tail_percentile"],
        "bootstrap_resamples": latency["bootstrap_resamples"],
        "bootstrap_confidence_level": latency[
            "bootstrap_confidence_level"
        ],
        "bootstrap_seed": latency["bootstrap_seed"],
        "batch_size_per_replica": latency["batch_size_per_replica"],
        "precision": latency["precision"],
        "expected_replicas": replicas,
        "raw_samples_per_candidate": raw_samples,
        "preloaded_batches": descriptor["preloaded_batches"],
        "timed_scope": (
            "grounding_dino_model_forward_plus_gpu_postprocess; "
            "preprocessing_and_text_tokenization_excluded"
        ),
        "synchronization": "accelerator_sync_before_and_after_each_sample",
        "replica_alignment": "nccl_barrier_before_each_timed_sample",
        "measurement_role": "selection_time",
        "input_descriptor": copy.deepcopy(dict(descriptor)),
        "input_descriptor_path": str(descriptor_path),
        "input_descriptor_file_sha256": sha256_file(descriptor_path),
        "input_descriptor_sha256": canonical_sha256(descriptor),
        "validity_thresholds": {
            "max_robust_cv": 0.10,
            "max_absolute_round_drift_fraction": 0.05,
            "max_bootstrap_ci_width_fraction": 0.03,
            "max_device_median_range_fraction": 0.05,
            "max_round_median_range_fraction": 0.05,
        },
    }


def build_manifest(
    inputs: Mapping[str, Any],
    *,
    experiment_dir: str | Path = HERE,
    qualification_decision: PilotQualificationDecision | None = None,
    observed_source_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build current pilot intent; no SDK or model operation is performed."""
    if inputs.get("schema_version") != 1:
        raise PreparationError("pilot input schema differs")
    here = Path(experiment_dir).resolve()
    decision = qualification_decision or audit_pilot_qualification(
        inputs,
        experiment_dir=here,
    )
    source = (
        copy.deepcopy(dict(observed_source_state))
        if observed_source_state is not None
        else source_state(inputs)
    )
    dataset_inputs = inputs["dataset"]
    source_manifest_path = (
        here / dataset_inputs["source_manifest_file"]
    ).resolve()
    conversion_path = (
        here / dataset_inputs["conversion_manifest_file"]
    ).resolve()
    source_manifest = read_json(source_manifest_path)
    conversion = read_json(conversion_path)
    validate_conversion_manifest(conversion)
    outputs = conversion["canonical_outputs"]
    dataset = {
        "dataset_id": dataset_inputs["dataset_id"],
        "task": "category_prompted_open_vocabulary_detection",
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "conversion_manifest_path": str(conversion_path),
        "conversion_manifest_sha256": sha256_file(conversion_path),
        "train": {
            "image_dir": source_manifest["splits"]["train"]["images"]["path"],
            "image_identity": source_manifest["splits"]["train"]["images"][
                "identity"
            ],
            "json_file": outputs["train_odvg"]["lustre_path"],
            "json_sha256": outputs["train_odvg"]["sha256"],
            "label_map": outputs["train_label_map"]["lustre_path"],
            "label_map_sha256": outputs["train_label_map"]["sha256"],
        },
        "validation": {
            "image_dir": source_manifest["splits"]["validation"]["images"][
                "path"
            ],
            "image_identity": source_manifest["splits"]["validation"][
                "images"
            ]["identity"],
            "json_file": outputs["validation_coco"]["lustre_path"],
            "json_sha256": outputs["validation_coco"]["sha256"],
        },
        "eval_class_ids": [0, 1, 2, 3],
        "category_prompts": [
            "cone",
            "forklift",
            "cart",
            "fire_extinguisher",
        ],
    }
    runtime = copy.deepcopy(dict(inputs["runtime"]))
    qualification_contract = read_json(decision.contract_path)
    offline_environment = qualification_contract["runtime"][
        "offline_environment"
    ]
    runtime.update(
        {
            "offline_environment": copy.deepcopy(offline_environment),
            "text_encoder_root": offline_environment[
                "model_text_encoder_type"
            ],
            "text_encoder_revision": qualification_contract[
                "runtime_inputs"
            ]["bert_revision"],
            "text_encoder_tree_sha256": qualification_contract[
                "runtime_inputs"
            ]["bert_tree_sha256"],
        }
    )
    skill_dir = Path(runtime["skill_dir"])
    skill_info_path = skill_dir / "references" / "skill_info.yaml"
    train_schema_path = skill_dir / "schemas" / "train.schema.json"
    train_template_path = (
        skill_dir / "references" / "spec_template_train.yaml"
    )
    evaluate_template_path = (
        skill_dir / "references" / "spec_template_evaluate.yaml"
    )
    skill_info = yaml.safe_load(skill_info_path.read_text(encoding="utf-8"))
    schema = read_json(train_schema_path)
    if (
        skill_info.get("network_arch") != MODEL_ID
        or skill_info.get("actions", {}).get("train", {}).get("mode")
        != "config"
        or skill_info.get("actions", {}).get("evaluate", {}).get("mode")
        != "config"
    ):
        raise PreparationError("Grounding DINO skill action contract changed")
    _validate_search_space(inputs["search"], schema)
    runtime.update(
        {
            "platform": "slurm",
            "sqsh_direct_path": True,
            "slurm_use_sqsh_conversion": False,
            "distributed_workers_per_child": 8,
            "skill_info_sha256": sha256_file(skill_info_path),
            "train_schema_sha256": sha256_file(train_schema_path),
            "train_template_sha256": sha256_file(train_template_path),
            "evaluate_template_sha256": sha256_file(
                evaluate_template_path
            ),
        }
    )
    descriptor_path = (
        here / inputs["latency"]["input_descriptor_file"]
    ).resolve()
    descriptor = read_json(descriptor_path)
    qualified = {
        item["checkpoint_id"]: item
        for item in decision.successful_records
    }
    ptms = [
        {
            "id": checkpoint_id,
            "qualification": copy.deepcopy(dict(qualified[checkpoint_id])),
            "artifact": {
                "slurm_path": qualified[checkpoint_id][
                    "qualified_input_checkpoint"
                ]["path"],
                "sha256": qualified[checkpoint_id][
                    "qualified_input_checkpoint"
                ]["sha256"],
                "size_bytes": qualified[checkpoint_id][
                    "qualified_input_checkpoint"
                ]["size_bytes"],
            },
        }
        for checkpoint_id in sorted(qualified)
    ]
    search = copy.deepcopy(dict(inputs["search"]))
    search["space_sha256"] = canonical_sha256(search["space"])
    modes = []
    for mode in MODES:
        objective = _mode_objective(mode, search)
        modes.append(
            {
                "mode": mode,
                "job_id": f"{inputs['campaign_id']}-{mode}",
                "session_id": f"{inputs['campaign_id']}-{mode}",
                "observation_namespace": (
                    f"{inputs['campaign_id']}-{mode}-observations"
                ),
                "observation_sharing": False,
                "initial_observation_ids": [],
                "search_seed": search["search_seed"],
                "search_space_sha256": search["space_sha256"],
                "allowed_ptm_ids": [item["id"] for item in ptms],
                "ptm_policy": "all_successfully_qualified_explicit",
                "objective": objective,
                "objective_sha256": canonical_sha256(objective),
            }
        )
    integrity_paths = {
        "inputs": DEFAULT_INPUTS if here == HERE else here / "pilot.inputs.v1.json",
        "manifest_generator": HERE / "pilot_manifest.py",
        "qualification_adapter": HERE / "pilot_qualification.py",
        "controller": HERE / "pilot_campaign.py",
        "latency_worker": HERE / "grounding_dino_latency_worker.py",
    }
    integrity = {}
    for label, path in integrity_paths.items():
        integrity[f"{label}_path"] = str(path)
        integrity[f"{label}_sha256"] = (
            sha256_file(path) if path.is_file() else None
        )
    submission_ready = bool(
        decision.runtime_ready
        and source.get("minimum_ancestor_satisfied") is True
        and source.get("clean") is True
        and ptms
    )
    document = {
        "schema_version": 1,
        "campaign_id": inputs["campaign_id"],
        "model": {
            "id": MODEL_ID,
            "task": "category_prompted_open_vocabulary_detection",
        },
        "source": source,
        "input_profile_sha256": canonical_sha256(inputs),
        "dataset": dataset,
        "runtime": runtime,
        "search": search,
        "ptms": ptms,
        "ptm_representation": "hierarchical_nonordinal_arms",
        "qualification_evidence": decision.to_dict(),
        "modes": modes,
        "latency_protocol": _latency_protocol(
            inputs,
            descriptor,
            descriptor_path,
        ),
        "first_candidate_gate": {
            "candidate_index_per_mode": 0,
            "initial_jobs": 3,
            "parallel_across_modes": True,
            "required_mode_results": list(MODES),
            "automatic_release": True,
            "remaining_candidates_per_mode": 19,
            "release_action": (
                "continue_same_frozen_runner_without_confirmation"
            ),
            "failure_action": (
                "preserve_failure_and_halt_before_remaining_budget"
            ),
            "success_criteria": [
                "training_complete",
                "standalone_evaluation_complete",
                "finite_mAP50",
                "stabilized_latency_quality_gate_passed",
                "recommendation_audit_complete",
                "artifact_provenance_complete",
            ],
        },
        "execution": {
            "submission_ready": submission_ready,
            "sdk_constructed": False,
            "scheduler_jobs_submitted": 0,
            "cpu_model_runs": 0,
            "smoke_or_ministep_runs": 0,
        },
        "agent_intervention_flags": {
            name: False for name in AGENT_FLAGS
        },
        "selection_isolation_flags": {
            name: False for name in SELECTION_FLAGS
        },
        "failure_policy": {
            "preserve_failed_recommendations": True,
            "replacement_recommendations": False,
            "maximum_infrastructure_retries": runtime[
                "max_infrastructure_retries"
            ],
        },
        "integrity": integrity,
    }
    validate_manifest(document, sealed=False)
    return document


def validate_manifest(
    document: Mapping[str, Any],
    *,
    sealed: bool = True,
) -> None:
    if (
        document.get("schema_version") != 1
        or document.get("model", {}).get("id") != MODEL_ID
        or tuple(item.get("mode") for item in document.get("modes", ()))
        != MODES
    ):
        raise PreparationError("pilot manifest identity/modes changed")
    source = document.get("source", {})
    if (
        not isinstance(source.get("repository"), str)
        or not Path(source["repository"]).is_absolute()
        or _GIT_COMMIT_RE.fullmatch(str(source.get("commit", ""))) is None
        or _GIT_COMMIT_RE.fullmatch(
            str(source.get("minimum_ancestor_commit", ""))
        )
        is None
    ):
        raise PreparationError("pilot source identity is invalid")
    search = document.get("search", {})
    if (
        search.get("algorithm") != "bayesian"
        or search.get("candidate_budget_per_mode") != 20
        or search.get("training_epochs") != 10
        or search.get("calibration_points") != 8
        or search.get("latency_accuracy_retention") != 0.90
        or tuple(search.get("parameters", ())) != SEARCH_PARAMETERS
        or search.get("space_sha256")
        != canonical_sha256(search.get("space"))
    ):
        raise PreparationError("pilot search contract changed")
    runtime = document.get("runtime", {})
    if (
        runtime.get("platform") != "slurm"
        or runtime.get("nodes_per_child") != 1
        or runtime.get("gpus_per_child") != 8
        or runtime.get("distributed_workers_per_child") != 8
        or runtime.get("sqsh_direct_path") is not True
        or runtime.get("slurm_use_sqsh_conversion") is not False
        or not str(runtime.get("sqsh_path", "")).endswith(".sqsh")
        or _SHA256_RE.fullmatch(str(runtime.get("sqsh_sha256", ""))) is None
        or not str(runtime.get("text_encoder_root", "")).startswith(
            "/lustre/"
        )
        or _SHA256_RE.fullmatch(
            str(runtime.get("text_encoder_tree_sha256", ""))
        )
        is None
    ):
        raise PreparationError("pilot SLURM/SQSH contract changed")
    protocol = document.get("latency_protocol", {})
    if (
        protocol.get("warmup_iterations") != 50
        or protocol.get("timed_iterations") != 100
        or protocol.get("repeated_rounds") != 5
        or protocol.get("expected_replicas") != 8
        or protocol.get("raw_samples_per_candidate") != 4000
        or protocol.get("precision") != "fp32"
        or protocol.get("measurement_role") != "selection_time"
        or protocol.get("input_descriptor_sha256")
        != canonical_sha256(protocol.get("input_descriptor"))
    ):
        raise PreparationError("pilot latency protocol changed")
    ptms = document.get("ptms")
    if not isinstance(ptms, list):
        raise PreparationError("pilot PTM inventory is invalid")
    ptm_ids = [item.get("id") for item in ptms]
    if ptm_ids != sorted(set(ptm_ids)):
        raise PreparationError("pilot PTM inventory is not canonical")
    decision = document.get("qualification_evidence", {})
    decision_payload = copy.deepcopy(dict(decision))
    observed_decision_sha = decision_payload.pop("decision_sha256", None)
    if (
        observed_decision_sha is None
        or observed_decision_sha != canonical_sha256(decision_payload)
    ):
        raise PreparationError(
            "pilot qualification decision canonical hash differs"
        )
    if ptm_ids != decision.get("qualified_checkpoint_ids", []):
        raise PreparationError(
            "pilot PTM arms differ from successful qualification evidence"
        )
    for mode in document["modes"]:
        if (
            mode.get("allowed_ptm_ids") != ptm_ids
            or mode.get("observation_sharing") is not False
            or mode.get("initial_observation_ids") != []
        ):
            raise PreparationError(
                f"{mode.get('mode')} PTM/observation contract changed"
            )
    acquisitions = [
        item["objective"]["acquisition"] for item in document["modes"]
    ]
    if acquisitions != [
        "expected_improvement",
        "constrained_expected_improvement",
        "parego_expected_improvement",
    ]:
        raise PreparationError("pilot objective acquisition routing changed")
    if (
        document["modes"][2]["objective"][
            "latency_accuracy_retention"
        ]
        is not None
        or document["modes"][2]["objective"][
            "multi_objective_min_accuracy"
        ]
        is not None
    ):
        raise PreparationError(
            "multi-objective mode inherited a latency constraint"
        )
    ready = bool(
        decision.get("runtime_ready") is True
        and source.get("minimum_ancestor_satisfied") is True
        and source.get("clean") is True
        and ptms
    )
    if document.get("execution", {}).get("submission_ready") is not ready:
        raise PreparationError("pilot readiness is inconsistent")
    if (
        any(document.get("agent_intervention_flags", {}).values())
        or any(document.get("selection_isolation_flags", {}).values())
    ):
        raise PreparationError("pilot contains manual intervention")
    if sealed:
        payload = copy.deepcopy(dict(document))
        observed = payload.pop("manifest_sha256", None)
        if observed != canonical_sha256(payload):
            raise PreparationError("pilot manifest canonical hash differs")


def seal_manifest(document: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(document))
    value["manifest_sha256"] = canonical_sha256(value)
    validate_manifest(value)
    return value


def load_manifest(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    value = read_json(path)
    validate_manifest(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args(argv)
    inputs = read_json(arguments.inputs)
    document = seal_manifest(
        build_manifest(
            inputs,
            experiment_dir=arguments.inputs.resolve().parent,
        )
    )
    if arguments.check_only:
        observed = load_manifest(arguments.output)
        if observed != document:
            raise PreparationError("committed pilot manifest is stale")
    else:
        arguments.output.write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "manifest_sha256": document["manifest_sha256"],
                "submission_ready": document["execution"][
                    "submission_ready"
                ],
                "qualified_ptm_ids": [
                    item["id"] for item in document["ptms"]
                ],
                "scheduler_jobs_submitted": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
