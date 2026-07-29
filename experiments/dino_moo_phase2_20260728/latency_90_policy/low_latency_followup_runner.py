#!/usr/bin/env python3

"""Run the frozen DINO 90%-retention low-latency follow-up search.

The implementation deliberately reuses ``expanded_search_runner.py`` for
schema construction, DINO spec materialization, TAO SDK/SLURM submission,
held-out accuracy evaluation, stabilized latency measurement, durable resume,
and per-seed archive creation.  This wrapper changes only the preregistered
follow-up contract:

* three new SHA-derived Bayesian seeds generate 20 recommendations each;
* the optimizer receives no candidates from the sealed 60-candidate archive;
* every trial artifact/checkpoint is retained;
* latency selection uses 90% accuracy retention while multi-objective
  eligibility remains independent;
* final production selection runs over the sealed 60 candidates plus every
  valid newly generated candidate.

The default action is a read-only dry run.  Matched post-selection latency
measurements are outside this runner and can never replace selection-time
objectives.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping


HERE = Path(__file__).resolve().parent
PHASE2_DIR = HERE.parent
if str(PHASE2_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE2_DIR))

import expanded_search_runner as expanded  # noqa: E402


DEFAULT_MANIFEST = HERE / "low_latency_followup_manifest.v1.json"
DEFAULT_RUNTIME = PHASE2_DIR / "runtime" / "low_latency_followup_v1"
DEFAULT_REPORT = DEFAULT_RUNTIME / "dry_run.json"
SEALED_BASE_MANIFEST = PHASE2_DIR / "expanded_search_manifest.v2.json"
SEALED_ARCHIVE_TABLE = (
    PHASE2_DIR / "runtime" / "expanded_search_v2" / "expanded_candidate_table.json"
)
SEALED_COMBINED_SELECTION = (
    PHASE2_DIR / "runtime" / "expanded_search_v2" / "expanded_combined_selection.json"
)
EXPECTED_MANIFEST_ID = "dino_low_latency_followup_90pct_20260728_v1"
EXPECTED_SEEDS = (409976740, 1455024938, 1415367367)
EXPECTED_RECOMMENDATIONS_PER_SEED = 20
EXPECTED_NEW_CANDIDATE_BUDGET = 60
EXPECTED_EXISTING_CANDIDATE_COUNT = 60
EXPECTED_UNION_CANDIDATE_COUNT = 120
EXPECTED_RETENTION = 0.90
EXPECTED_OLD_ACCURACY_WINNER = "seed_271828_rec_18"
EXPECTED_OLD_ACCURACY = 0.6554138278683255
EXPECTED_OPPORTUNITY_FLOOR = 0.589872445081493
EXPECTED_LATENCY_TOLERANCE_MS = 0.73553775
EXPECTED_BASE_MANIFEST_SHA256 = (
    "9ac29e1aa07167a040d217fdab2d3cfdea0baad690dc95a70f2fe6715908793a"
)
EXPECTED_BASE_MANIFEST_INTERNAL_SHA256 = (
    "910744ae2fead7e4e2e9a53fc672baef1ac43307e3979671b2b876fff422de96"
)
EXPECTED_BASE_RUNNER_SHA256 = (
    "0eb1948d4fb887b9c3fe938d60865ebb4ef86ae00d9ca80aa0d42b465a073073"
)
EXPECTED_ARCHIVE_TABLE_SHA256 = (
    "5ba323d05d9ec8e3703e636f8b5e2975cc620eeec10df75ec6e792318dc2df03"
)
EXPECTED_COMBINED_SELECTION_SHA256 = (
    "78ab9d2fa83cc3abe9057d137c0b88f120158b6ad77268482d2c18f5a1533af1"
)
EXPECTED_ACKNOWLEDGEMENT = (
    "USER_AUTHORIZED_3X8GPU_SLURM_DINO_LOW_LATENCY_FOLLOWUP_20260728"
)
EXPECTED_SCOPE = {
    "model_family": "DINO ResNet50",
    "dataset_uri": (
        "s3://nvcf-storage-handling/data/"
        "tao_od_synthetic_full_dino_coco/"
    ),
    "other_models_permitted": False,
    "other_datasets_permitted": False,
}
EXPECTED_SEED_MATERIAL = tuple(
    f"dino-low-latency-followup-v1:{index}" for index in range(3)
)
EXPECTED_SEED_DIGESTS = (
    "186fbfa40c701027265e21568fb156c4e5e76f8b335cf68934558e9218c78aa5",
    "d6b9eb2a44d1c99ee3fa6f5f627bdd03e53092b6ecfc32d1c846c9180709cea2",
    "d45ccac7d0bdeb4df3bc0c72e698a3707dfc08f10a08bf514b35bef5b41cfb47",
)
ORIGINAL_LOAD_MANIFEST = expanded.load_manifest
ORIGINAL_SELECTION_SETTINGS = expanded.selection_settings


ContractError = expanded.ContractError


def derive_seed(material: str) -> tuple[int, str]:
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF, digest.hex()


def _validate_seed_contract(manifest: Mapping[str, Any]) -> None:
    records = manifest["search_design"]["seed_derivation"]["records"]
    expanded.require_equal(len(records), 3, "seed derivation record count")
    observed_seeds: list[int] = []
    for index, record in enumerate(records):
        material = EXPECTED_SEED_MATERIAL[index]
        seed, digest = derive_seed(material)
        expanded.require_equal(record["index"], index, "seed derivation index")
        expanded.require_equal(record["material_utf8"], material, "seed material")
        expanded.require_equal(
            record["sha256"], EXPECTED_SEED_DIGESTS[index], "seed digest"
        )
        expanded.require_equal(digest, record["sha256"], "derived seed digest")
        expanded.require_equal(seed, record["seed"], "derived seed value")
        observed_seeds.append(seed)
    expanded.require_equal(
        tuple(observed_seeds), EXPECTED_SEEDS, "SHA-derived search seeds"
    )


def validate_followup_manifest(manifest: Mapping[str, Any]) -> None:
    expanded.require_equal(manifest.get("schema_version"), 1, "schema version")
    expanded.require_equal(
        manifest.get("manifest_id"), EXPECTED_MANIFEST_ID, "manifest ID"
    )
    expanded.require_equal(
        manifest.get("status"), "preregistered_ready_to_launch", "status"
    )
    expanded.require_equal(manifest.get("scope"), EXPECTED_SCOPE, "DINO-only scope")
    expanded.require_equal(
        manifest.get("algorithm_only_generation_and_selection"), True,
        "algorithm-only requirement",
    )
    expanded.require_equal(
        manifest.get("manual_candidate_injection_permitted"), False,
        "manual candidate injection",
    )
    expanded.require_equal(
        manifest.get("winner_override_permitted"), False, "winner override"
    )

    source = manifest["sealed_source"]
    expanded.require_equal(
        source["expanded_search_manifest"],
        {
            "path": str(SEALED_BASE_MANIFEST),
            "whole_file_sha256": EXPECTED_BASE_MANIFEST_SHA256,
            "internal_manifest_sha256": EXPECTED_BASE_MANIFEST_INTERNAL_SHA256,
        },
        "sealed expanded-search manifest",
    )
    expanded.require_equal(
        source["expanded_search_runner"],
        {
            "path": str(PHASE2_DIR / "expanded_search_runner.py"),
            "sha256": EXPECTED_BASE_RUNNER_SHA256,
            "reuse": (
                "schema/spec/SLURM/evaluation/latency/resume/archive execution engine"
            ),
        },
        "sealed expanded-search runner",
    )
    expanded.require_equal(
        source["existing_archive"],
        {
            "candidate_count": EXPECTED_EXISTING_CANDIDATE_COUNT,
            "candidate_table_path": str(SEALED_ARCHIVE_TABLE),
            "candidate_table_sha256": EXPECTED_ARCHIVE_TABLE_SHA256,
            "combined_selection_path": str(SEALED_COMBINED_SELECTION),
            "combined_selection_sha256": EXPECTED_COMBINED_SELECTION_SHA256,
            "optimizer_prior_input": False,
            "final_selection_input": True,
        },
        "sealed archive binding",
    )
    wrapper = source["followup_runner"]
    expanded.require_equal(
        Path(wrapper["path"]).resolve(), Path(__file__).resolve(), "wrapper path"
    )
    expanded.require_sha256(wrapper["sha256"], "wrapper SHA256")

    design = manifest["search_design"]
    expanded.require_equal(design["algorithm"], "bayesian", "algorithm")
    expanded.require_equal(
        tuple(design["search_seeds"]), EXPECTED_SEEDS, "search seeds"
    )
    expanded.require_equal(
        design["recommendations_per_seed"],
        EXPECTED_RECOMMENDATIONS_PER_SEED,
        "recommendations per seed",
    )
    expanded.require_equal(
        design["new_candidate_budget"],
        EXPECTED_NEW_CANDIDATE_BUDGET,
        "new candidate budget",
    )
    expanded.require_equal(design["training_seed"], 1234, "training seed")
    expanded.require_equal(
        design["optimizer_generation_population"],
        "new_candidates_only",
        "optimizer population",
    )
    expanded.require_equal(
        design["optimizer_selection_mode"],
        "latency",
        "optimizer selection mode",
    )
    expanded.require_equal(
        design["final_selection_population"],
        "sealed_60_plus_all_valid_new_candidates",
        "final selection population",
    )
    expanded.require_equal(
        design["manual_candidate_injection_permitted"],
        False,
        "design manual injection",
    )
    expanded.require_equal(
        design["result_ordered_search_changes_permitted"],
        False,
        "result-ordered search changes",
    )
    _validate_seed_contract(manifest)

    domains = manifest["search_space"]["search_domains"]
    expanded.require_equal(
        manifest["search_space"]["search_parameters"],
        [
            "model.enc_layers",
            "model.dec_layers",
            "train.optim.lr",
            "train.optim.weight_decay",
        ],
        "search parameter order",
    )
    for path in ("model.enc_layers", "model.dec_layers"):
        expanded.require_equal(
            domains[path],
            {
                "representation": "ordered_integer_levels",
                "valid_min": 3,
                "valid_max": 6,
                "valid_options": [3, 4, 5, 6],
            },
            f"{path} domain",
        )
    expanded.require_equal(
        domains["train.optim.lr"],
        {
            "representation": "continuous",
            "valid_min": 1e-05,
            "valid_max": 0.0005,
        },
        "learning-rate domain",
    )
    expanded.require_equal(
        domains["train.optim.weight_decay"],
        {
            "representation": "continuous",
            "valid_min": 1e-05,
            "valid_max": 0.001,
        },
        "weight-decay domain",
    )

    runtime = manifest["runtime"]
    expanded.require_equal(runtime["platform"], "SLURM via TAO SDK", "platform")
    expanded.require_equal(runtime["container_format"], "SQSH", "container format")
    expanded.require_equal(runtime["num_nodes"], 1, "nodes per trial")
    expanded.require_equal(runtime["gpus_per_node"], 8, "GPUs per trial")
    expanded.require_equal(runtime["train_epochs"], 10, "train epochs")
    expanded.require_equal(runtime["precision"], "fp32", "precision")
    expanded.require_equal(
        Path(runtime["target_runtime_path"]).resolve(),
        DEFAULT_RUNTIME.resolve(),
        "runtime target",
    )
    expanded.require_equal(
        runtime["retain_every_trial_artifact"], True, "trial artifact retention"
    )
    expanded.require_equal(
        runtime["retain_every_produced_checkpoint"], True, "checkpoint retention"
    )

    selection = manifest["selection"]
    expanded.require_equal(
        selection["latency_accuracy_retention"],
        {
            "type": "relative",
            "retained_fraction": EXPECTED_RETENTION,
            "reference": "accuracy_winner",
        },
        "latency retention",
    )
    expanded.require_equal(
        selection["multi_objective_min_accuracy"],
        None,
        "multi-objective independence",
    )
    expanded.require_equal(
        selection["latency_tolerance_ms"],
        EXPECTED_LATENCY_TOLERANCE_MS,
        "latency tolerance",
    )
    opportunity = selection["fixed_opportunity_question"]
    expanded.require_equal(
        opportunity,
        {
            "reference_archive_accuracy_winner": EXPECTED_OLD_ACCURACY_WINNER,
            "reference_accuracy": EXPECTED_OLD_ACCURACY,
            "retained_fraction": EXPECTED_RETENTION,
            "minimum_mAP50": EXPECTED_OPPORTUNITY_FLOOR,
            "purpose": (
                "Determine whether an algorithm-generated lower-latency "
                "architecture reaches the frozen existing-archive floor."
            ),
            "does_not_replace_relative_policy_for_final_union_selection": True,
        },
        "fixed opportunity question",
    )
    expanded.require_equal(
        selection["final_union_threshold_rule"],
        (
            "Production latency mode recomputes 0.90 * the highest valid "
            "accuracy in the sealed-old-plus-new union."
        ),
        "final union threshold rule",
    )

    audit = manifest["frozen_prohibitions"]
    required_false = {
        "manual_candidate_injection",
        "post_result_threshold_change",
        "post_result_range_change",
        "post_result_seed_or_budget_change",
        "winner_override",
        "multi_objective_inherits_latency_retention",
        "matched_measurements_feed_selection",
        "matched_measurements_feed_reselection",
    }
    expanded.require_equal(set(audit), required_false, "prohibition fields")
    if any(audit.values()):
        raise ContractError("every frozen prohibition flag must remain false")


def load_followup_manifest(
    path: Path, *, supplied_file_sha256: str
) -> tuple[dict[str, Any], str]:
    supplied = expanded.require_sha256(
        supplied_file_sha256, "supplied follow-up manifest SHA256"
    )
    actual = expanded.sha256_file(path)
    expanded.require_equal(actual, supplied, "follow-up manifest whole-file SHA256")
    manifest = expanded.load_json(path)
    claimed = expanded.require_sha256(
        manifest.get("manifest_sha256"), "follow-up internal manifest SHA256"
    )
    unhashed = copy.deepcopy(manifest)
    del unhashed["manifest_sha256"]
    expanded.require_equal(
        expanded.sha256_value(unhashed), claimed, "follow-up internal manifest SHA256"
    )
    validate_followup_manifest(manifest)
    return manifest, actual


def load_sealed_base_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    source = manifest["sealed_source"]["expanded_search_manifest"]
    expanded.require_equal(
        expanded.sha256_file(SEALED_BASE_MANIFEST),
        source["whole_file_sha256"],
        "sealed base manifest whole-file SHA256",
    )
    base = expanded.load_json(SEALED_BASE_MANIFEST)
    expanded.require_equal(
        base["manifest_sha256"],
        source["internal_manifest_sha256"],
        "sealed base manifest internal SHA256",
    )
    unhashed = copy.deepcopy(base)
    del unhashed["manifest_sha256"]
    expanded.require_equal(
        expanded.sha256_value(unhashed),
        base["manifest_sha256"],
        "sealed base manifest canonical SHA256",
    )
    return base


def project_execution_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Project the compact follow-up contract onto the sealed execution schema."""

    projected = load_sealed_base_manifest(manifest)
    projected["manifest_id"] = manifest["manifest_id"]
    projected["manifest_sha256"] = manifest["manifest_sha256"]
    projected["status"] = manifest["status"]
    projected["scope"] = copy.deepcopy(manifest["scope"])
    projected["search_design"]["search_seeds"] = list(
        manifest["search_design"]["search_seeds"]
    )
    projected["search_design"]["recommendations_per_seed"] = (
        manifest["search_design"]["recommendations_per_seed"]
    )
    projected["search_design"]["total_candidate_budget"] = (
        manifest["search_design"]["new_candidate_budget"]
    )
    projected["search_design"]["budget_rule"] = (
        "Exactly 20 new-only Bayesian recommendations for each of three "
        "SHA-derived preregistered seeds."
    )
    projected["search_design"]["shared_archive"] = (
        "Optimizer generation uses only each new seeded subarchive; after all "
        "new measurements, final production selection receives the sealed 60 "
        "plus every valid new candidate."
    )
    projected["search_space"]["search_parameters"] = copy.deepcopy(
        manifest["search_space"]["search_parameters"]
    )
    projected["search_space"]["search_domains"] = copy.deepcopy(
        manifest["search_space"]["search_domains"]
    )
    by_path = {
        axis["path"]: axis for axis in projected["search_space"]["architecture_axes"]
    }
    for path in ("model.enc_layers", "model.dec_layers"):
        domain = manifest["search_space"]["search_domains"][path]
        by_path[path]["preregistered_levels"] = list(domain["valid_options"])
        by_path[path]["search_domain"] = copy.deepcopy(domain)
    projected["selection"]["latency_mode"]["latency_accuracy_retention"] = {
        "type": "relative",
        "retained_fraction": EXPECTED_RETENTION,
        "reference": "accuracy_winner",
    }
    projected["selection"]["multi_objective_mode"][
        "multi_objective_min_accuracy"
    ] = None
    projected["selection"]["latency_tolerance"]["value_ms"] = (
        EXPECTED_LATENCY_TOLERANCE_MS
    )
    projected["runtime_supersession"]["target_runtime_path"] = str(
        DEFAULT_RUNTIME.resolve()
    )
    projected["followup_contract"] = copy.deepcopy(dict(manifest))
    return projected


def validate_projected_manifest(projected: Mapping[str, Any]) -> None:
    contract = projected.get("followup_contract")
    if not isinstance(contract, dict):
        raise ContractError("execution projection lacks follow-up contract")
    validate_followup_manifest(contract)
    expanded.require_equal(
        projected["manifest_id"], EXPECTED_MANIFEST_ID, "projected manifest ID"
    )
    expanded.require_equal(
        tuple(projected["search_design"]["search_seeds"]),
        EXPECTED_SEEDS,
        "projected search seeds",
    )
    expanded.require_equal(
        projected["search_design"]["total_candidate_budget"],
        EXPECTED_NEW_CANDIDATE_BUDGET,
        "projected budget",
    )
    expanded.require_equal(
        projected["selection"]["latency_mode"]["latency_accuracy_retention"][
            "retained_fraction"
        ],
        EXPECTED_RETENTION,
        "projected retention",
    )
    expanded.require_equal(
        projected["selection"]["multi_objective_mode"][
            "multi_objective_min_accuracy"
        ],
        None,
        "projected multi-objective floor",
    )
    expanded.require_equal(
        projected["frozen_identity"]["training_controls"]["train_epochs"],
        10,
        "projected training epochs",
    )
    expanded.require_equal(
        projected["frozen_identity"]["runtime"]["gpu_count_per_node"],
        8,
        "projected GPUs",
    )


def selection_settings(projected: dict[str, Any], seed: int) -> dict[str, Any]:
    settings = ORIGINAL_SELECTION_SETTINGS(projected, seed)
    settings["session_id"] = f"dino_low_latency_followup_seed_{seed}"
    settings["experiment_id"] = f"dino_low_latency_followup_seed_{seed}"
    # Acquisition follows the latency-mode constrained objective.  Both raw
    # objectives are still measured, persisted, and later supplied to all
    # three production selectors over the complete old-plus-new union.
    settings["selection_mode"] = "latency"
    # Retain every produced artifact/checkpoint.  DINO writes only the frozen
    # terminal checkpoint because checkpoint_interval == train_epochs.
    settings["automl_delete_intermediate_ckpt"] = False
    settings["automl_checkpoint_retention_strategy"] = "terminal"
    return settings


def validate_selector_configuration(projected: dict[str, Any]) -> dict[str, Any]:
    settings = selection_settings(projected, EXPECTED_SEEDS[0])
    objective = expanded.parse_objective_config(settings)
    config = objective.selection_config
    if config is None:
        raise ContractError("production objective parser did not build a selector")
    expanded.require_equal(
        config.mode,
        "latency",
        "parsed optimizer selection mode",
    )
    expanded.require_equal(
        config.latency_accuracy_retention.kind,
        "relative",
        "parsed latency retention type",
    )
    expanded.require_equal(
        config.latency_accuracy_retention.value,
        EXPECTED_RETENTION,
        "parsed latency retention value",
    )
    expanded.require_equal(
        config.multi_objective_min_accuracy,
        None,
        "parsed multi-objective accuracy floor",
    )
    expanded.require_equal(
        config.latency_tolerance,
        EXPECTED_LATENCY_TOLERANCE_MS,
        "parsed latency tolerance",
    )
    expanded.require_equal(
        settings["automl_delete_intermediate_ckpt"],
        False,
        "trial artifact retention",
    )
    return {"settings": settings, "parsed_selection": config.to_dict()}


def _base_load_manifest(
    path: Path, *, supplied_file_sha256: str
) -> tuple[dict[str, Any], str]:
    manifest, actual = load_followup_manifest(
        path, supplied_file_sha256=supplied_file_sha256
    )
    projected = project_execution_manifest(manifest)
    validate_projected_manifest(projected)
    return projected, actual


def runtime_contract_payload(
    projected: dict[str, Any],
    manifest_path: Path,
    manifest_file_sha256: str,
    runtime_dir: Path,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "contract_id": "dino_low_latency_followup_runtime_20260728_v1",
        "manifest_id": projected["manifest_id"],
        "manifest_path": str(manifest_path.resolve()),
        "manifest_file_sha256": expanded.require_sha256(
            manifest_file_sha256, "runtime marker manifest SHA256"
        ),
        "manifest_internal_sha256": projected["manifest_sha256"],
        "target_runtime_path": str(runtime_dir.resolve()),
        "sealed_archive_candidate_count": EXPECTED_EXISTING_CANDIDATE_COUNT,
        "new_candidate_budget": EXPECTED_NEW_CANDIDATE_BUDGET,
        "optimizer_prior_candidate_count": 0,
        "final_union_candidate_budget": EXPECTED_UNION_CANDIDATE_COUNT,
        "valid_objective_observations_reused_by_optimizer": 0,
    }
    payload["contract_sha256"] = expanded.sha256_value(payload)
    return payload


def configure_expanded_execution() -> None:
    """Bind the generic sealed execution engine to this frozen contract."""

    expanded.EXPECTED_SEARCH_SEEDS = EXPECTED_SEEDS
    expanded.EXPECTED_RECOMMENDATIONS_PER_SEED = (
        EXPECTED_RECOMMENDATIONS_PER_SEED
    )
    expanded.EXPECTED_TOTAL_CANDIDATES = EXPECTED_NEW_CANDIDATE_BUDGET
    expanded.EXPECTED_ACKNOWLEDGEMENT = EXPECTED_ACKNOWLEDGEMENT
    expanded.load_manifest = _base_load_manifest
    expanded.validate_manifest_contract = validate_projected_manifest
    expanded.selection_settings = selection_settings
    expanded.validate_selector_configuration = validate_selector_configuration
    expanded.runtime_contract_payload = runtime_contract_payload


def wrapper_source_provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    expected = manifest["sealed_source"]["followup_runner"]
    path = Path(__file__).resolve()
    observed = expanded.sha256_file(path)
    expanded.require_equal(observed, expected["sha256"], "follow-up runner SHA256")
    repo = Path(
        manifest["sealed_source"]["source_repository"]["path"]
    ).resolve()
    provenance = expanded.git_path_provenance(repo, path)
    return {
        "path": str(path),
        "expected_sha256": expected["sha256"],
        "observed_sha256": observed,
        **provenance,
    }


def manifest_source_provenance(path: Path) -> dict[str, Any]:
    repo = Path("/localhome/local-rarunachalam/tao-automl")
    return expanded.git_path_provenance(repo, path.resolve())


def validate_local_provenance(
    manifest: Mapping[str, Any],
    projected: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    expanded.require_equal(
        expanded.sha256_file(PHASE2_DIR / "expanded_search_runner.py"),
        EXPECTED_BASE_RUNNER_SHA256,
        "sealed execution runner SHA256",
    )
    expanded.require_equal(
        expanded.sha256_file(SEALED_ARCHIVE_TABLE),
        EXPECTED_ARCHIVE_TABLE_SHA256,
        "sealed archive table SHA256",
    )
    expanded.require_equal(
        expanded.sha256_file(SEALED_COMBINED_SELECTION),
        EXPECTED_COMBINED_SELECTION_SHA256,
        "sealed combined selection SHA256",
    )
    checks = expanded.validate_local_provenance(projected, manifest_path)
    checks["followup_runner_source"] = wrapper_source_provenance(manifest)
    checks["followup_manifest_source"] = manifest_source_provenance(manifest_path)
    policy = manifest["sealed_source"]["latency_policy_profile"]
    policy_path = Path(policy["path"])
    expanded.require_equal(
        expanded.sha256_file(policy_path),
        policy["sha256"],
        "latency policy profile SHA256",
    )
    checks["latency_policy_profile"] = copy.deepcopy(policy)
    checks["sealed_archive"] = copy.deepcopy(
        manifest["sealed_source"]["existing_archive"]
    )
    return checks


def require_launch_sources_ready(local_checks: Mapping[str, Any]) -> None:
    expanded.require_launch_source_ready(local_checks["runner_source"])
    for name in ("followup_runner_source", "followup_manifest_source"):
        provenance = local_checks[name]
        if provenance.get("launch_source_ready") is not True:
            raise ContractError(
                f"launch requires committed clean {name}: "
                f"{provenance.get('blockers', [])}"
            )


def load_existing_archive_rows(
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source = manifest["sealed_source"]["existing_archive"]
    expanded.require_equal(
        expanded.sha256_file(SEALED_ARCHIVE_TABLE),
        source["candidate_table_sha256"],
        "sealed archive table SHA256",
    )
    payload = expanded.load_json(SEALED_ARCHIVE_TABLE)
    expanded.require_equal(
        payload["candidate_count"],
        EXPECTED_EXISTING_CANDIDATE_COUNT,
        "sealed archive candidate count",
    )
    expanded.require_equal(
        payload["manual_candidate_injection_used"],
        False,
        "sealed archive manual injection",
    )
    rows = copy.deepcopy(payload["rows"])
    expanded.require_equal(
        len(rows), EXPECTED_EXISTING_CANDIDATE_COUNT, "sealed archive row count"
    )
    return rows


def successful_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(dict(record))
        for record in records
        if record.get("status") == "success"
        and isinstance(record.get("objective_values"), dict)
    ]


def combine_union_results(
    manifest: Mapping[str, Any],
    projected: dict[str, Any],
    manifest_path: Path,
    manifest_file_sha256: str,
    runtime_dir: Path,
) -> dict[str, Any]:
    """Run production selection over sealed old60 + every valid new result."""

    expanded.validate_runtime_target(projected, runtime_dir)
    expanded.validate_runtime_contract_marker(
        projected, manifest_path, manifest_file_sha256, runtime_dir
    )
    expanded.validate_runtime_root_entries(projected, runtime_dir)
    new_records, new_successful = expanded.load_complete_archive(
        projected, manifest_file_sha256, runtime_dir
    )
    old_records = load_existing_archive_rows(manifest)
    old_ids = {record["candidate_id"] for record in old_records}
    new_ids = {record["candidate_id"] for record in new_records}
    if old_ids & new_ids:
        raise ContractError("new SHA-derived seeds collided with sealed candidate IDs")
    union_records = old_records + new_records
    expanded.require_equal(
        len(union_records), EXPECTED_UNION_CANDIDATE_COUNT, "union record count"
    )
    union_successful = successful_records(old_records) + new_successful
    analysis, order_audit = expanded.analyze_union_archive(
        projected, union_successful
    )
    analysis["search"] = {
        "algorithm": "bayesian",
        "new_search_seeds": list(EXPECTED_SEEDS),
        "new_recommendations_per_seed": EXPECTED_RECOMMENDATIONS_PER_SEED,
        "sealed_candidate_records": len(old_records),
        "new_candidate_records": len(new_records),
        "new_successful_candidates": len(new_successful),
        "union_candidate_records": len(union_records),
        "union_successful_candidates": len(union_successful),
        "optimizer_generation_population": "new_candidates_only",
        "production_selection_population": (
            "sealed_60_plus_every_valid_new_candidate"
        ),
        "all_modes_receive_identical_union_archive": True,
    }
    analysis["selection_authority"] = {
        "module": "tao_automl.selection",
        "function": "analyze_archive",
        "manual_override_used": False,
        "order_independence_audit": order_audit,
        "latency_accuracy_retention": EXPECTED_RETENTION,
        "multi_objective_min_accuracy": None,
    }
    analysis["manifest"] = {
        "path": str(manifest_path.resolve()),
        "whole_file_sha256": manifest_file_sha256,
        "internal_manifest_sha256": manifest["manifest_sha256"],
    }
    analysis["selection_isolation"] = {
        "selector_invoked_on_matched_measurements": False,
        "selection_time_objectives_replaced": False,
        "measurements_feed_selection": False,
        "measurements_feed_reselection": False,
        "algorithm_selected_candidate_overridden": False,
    }
    combined_path = runtime_dir / "expanded_combined_selection.json"
    expanded.atomic_json(combined_path, analysis)
    table_artifacts = expanded.write_candidate_artifacts(
        projected, union_records, analysis, runtime_dir
    )
    integrity = {
        "schema_version": 1,
        "created_at_utc": expanded.utc_timestamp(),
        "scope": copy.deepcopy(manifest["scope"]),
        "manifest": analysis["manifest"],
        "candidate_budget": {
            "sealed": EXPECTED_EXISTING_CANDIDATE_COUNT,
            "new_expected": EXPECTED_NEW_CANDIDATE_BUDGET,
            "new_observed": len(new_records),
            "new_successful": len(new_successful),
            "union_observed": len(union_records),
            "union_successful": len(union_successful),
        },
        "generation": {
            "algorithm_only": True,
            "optimizer_prior_candidate_count": 0,
            "manual_candidate_injection_used": False,
            "result_ordered_search_changes_used": False,
        },
        "selection": {
            "settings": selection_settings(projected, EXPECTED_SEEDS[0]),
            "selected_candidate_ids": {
                mode: value["winner_id"]
                for mode, value in analysis["selections"].items()
            },
            "manual_override_used": False,
            "candidate_reordering_used": False,
        },
        "checkpoint_retention": {
            "every_trial_artifact_retained": True,
            "every_produced_checkpoint_retained": True,
            "automl_delete_intermediate_ckpt": False,
        },
        "selection_isolation": copy.deepcopy(analysis["selection_isolation"]),
        "artifacts": {
            "combined_selection": str(combined_path),
            "combined_selection_sha256": expanded.sha256_file(combined_path),
            **table_artifacts,
        },
    }
    integrity_path = runtime_dir / "expanded_integrity_audit.json"
    expanded.atomic_json(integrity_path, integrity)
    result = {
        "status": "complete",
        "combined_selection": str(combined_path),
        "combined_selection_sha256": expanded.sha256_file(combined_path),
        "integrity_audit": str(integrity_path),
        "integrity_audit_sha256": expanded.sha256_file(integrity_path),
        "candidate_artifacts": table_artifacts,
        "selections": analysis["selections"],
    }
    expanded.atomic_json(runtime_dir / "expanded_completion.json", result)
    return result


def dry_run_report(
    manifest: Mapping[str, Any],
    projected: dict[str, Any],
    manifest_path: Path,
    manifest_file_sha256: str,
    local_checks: dict[str, Any],
    remote_checks: dict[str, Any] | None,
) -> dict[str, Any]:
    report = expanded.dry_run_report(
        projected,
        manifest_path,
        manifest_file_sha256,
        local_checks,
        remote_checks,
    )
    report["status"] = "followup_dry_run_validated_not_launched"
    report["manifest"] = {
        "path": str(manifest_path.resolve()),
        "whole_file_sha256": manifest_file_sha256,
        "internal_manifest_sha256": manifest["manifest_sha256"],
        "manifest_id": manifest["manifest_id"],
    }
    report["search"]["optimizer_generation_population"] = "new_candidates_only"
    report["search"]["sealed_prior_candidates_supplied_to_optimizer"] = 0
    report["search"]["final_selection_population"] = (
        "sealed_60_plus_every_valid_new_candidate"
    )
    report["search"]["sealed_archive_candidate_count"] = (
        EXPECTED_EXISTING_CANDIDATE_COUNT
    )
    report["search"]["total_candidate_budget"] = (
        EXPECTED_NEW_CANDIDATE_BUDGET
    )
    report["selector"]["fixed_existing_archive_opportunity_floor"] = (
        EXPECTED_OPPORTUNITY_FLOOR
    )
    report["selector"]["final_union_threshold_rule"] = (
        manifest["selection"]["final_union_threshold_rule"]
    )
    report["execution"]["retain_every_trial_artifact"] = True
    report["execution"]["retain_every_produced_checkpoint"] = True
    report["execution"]["old_archive_mutated"] = False
    report["execution"]["matched_measurements_feed_selection"] = False
    return report


def run_seed_entry(
    manifest_path: str,
    manifest_file_sha256: str,
    runtime_dir: str,
    seed: int,
    resume: bool,
) -> None:
    configure_expanded_execution()
    expanded.run_seed(
        manifest_path,
        manifest_file_sha256,
        runtime_dir,
        seed,
        resume,
    )


def launch_all_seeds(
    projected: dict[str, Any],
    manifest_path: Path,
    manifest_file_sha256: str,
    runtime_dir: Path,
    *,
    resume: bool,
) -> dict[int, int | None]:
    expanded.prepare_runtime_contract(
        projected,
        manifest_path,
        manifest_file_sha256,
        runtime_dir,
        resume=resume,
    )
    context = mp.get_context("spawn")
    processes = {
        seed: context.Process(
            target=run_seed_entry,
            args=(
                str(manifest_path),
                manifest_file_sha256,
                str(runtime_dir),
                seed,
                resume,
            ),
            name=f"dino-low-latency-followup-seed-{seed}",
        )
        for seed in EXPECTED_SEEDS
    }
    for process in processes.values():
        process.start()

    def forward_signal(signum: int, _frame: object) -> None:
        for process in processes.values():
            if process.is_alive() and process.pid:
                os.kill(process.pid, signum)

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)
    exit_codes: dict[int, int | None] = {}
    while processes:
        for seed, process in list(processes.items()):
            process.join(timeout=1)
            if not process.is_alive():
                exit_codes[seed] = process.exitcode
                processes.pop(seed)
                print(
                    f"SEED_PROCESS_EXIT seed={seed} exitcode={process.exitcode}",
                    flush=True,
                )
        if processes:
            time.sleep(1)
    expanded.atomic_json(
        runtime_dir / "seed_process_status.json", {"exit_codes": exit_codes}
    )
    return exit_codes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--combine-only", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest-file-sha256", required=True)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--verify-remote", action="store_true")
    parser.add_argument("--acknowledgement", default="")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    runtime_dir = args.runtime_dir.resolve()
    manifest, manifest_file_sha256 = load_followup_manifest(
        manifest_path, supplied_file_sha256=args.manifest_file_sha256
    )
    projected = project_execution_manifest(manifest)
    configure_expanded_execution()
    expanded.validate_runtime_target(projected, runtime_dir)
    report_path = args.report.resolve()
    expected_report = (runtime_dir / "dry_run.json").resolve()
    expanded.require_equal(report_path, expected_report, "dry-run report path")
    local_checks = validate_local_provenance(
        manifest, projected, manifest_path
    )
    if args.launch:
        require_launch_sources_ready(local_checks)

    remote_checks = None
    if args.verify_remote or args.launch:
        loaded_keys = expanded.load_env_file(
            Path(projected["frozen_identity"]["runtime"]["secrets_env_path"])
        )
        remote_checks = expanded.verify_remote_contract(projected)
        remote_checks["loaded_secret_keys"] = loaded_keys
        remote_checks["secret_values_recorded"] = False

    if args.combine_only:
        if args.resume:
            raise ContractError("--resume is valid only with --launch")
        result = combine_union_results(
            manifest,
            projected,
            manifest_path,
            manifest_file_sha256,
            runtime_dir,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    report = dry_run_report(
        manifest,
        projected,
        manifest_path,
        manifest_file_sha256,
        local_checks,
        remote_checks,
    )
    expanded.atomic_json(report_path, report)
    if not args.launch:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "manifest_file_sha256": manifest_file_sha256,
                    "search_seeds": list(EXPECTED_SEEDS),
                    "new_candidate_budget": EXPECTED_NEW_CANDIDATE_BUDGET,
                    "report": str(report_path),
                    "launched": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not args.verify_remote:
        raise ContractError("--launch requires --verify-remote")
    if args.acknowledgement != EXPECTED_ACKNOWLEDGEMENT:
        raise ContractError(
            "--launch requires the exact user-authorized acknowledgement"
        )
    exit_codes = launch_all_seeds(
        projected,
        manifest_path,
        manifest_file_sha256,
        runtime_dir,
        resume=args.resume,
    )
    if not all(code == 0 for code in exit_codes.values()):
        return 1
    result = combine_union_results(
        manifest,
        projected,
        manifest_path,
        manifest_file_sha256,
        runtime_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s",
    )
    try:
        raise SystemExit(main())
    except ContractError as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        raise SystemExit(2)
