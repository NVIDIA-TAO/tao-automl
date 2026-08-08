# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for immutable staged cross-mode campaign manifests."""

from __future__ import annotations

import copy

import pytest

from tao_automl.campaign_manifest import (
    AGENT_INTERVENTION_FLAGS,
    CAMPAIGN_MODES,
    SELECTION_ISOLATION_FLAGS,
    CampaignManifestValidationError,
    CampaignResumeMismatchError,
    create_campaign_manifest,
    load_campaign_manifest,
)
from tao_automl.ptm_registry import canonical_sha256


def _digest(character: str) -> str:
    return character * 64


def _objective(mode: str):
    accuracy = {"name": "mAP50", "role": "accuracy", "direction": "maximize"}
    latency = {
        "name": "median_latency_ms",
        "role": "latency",
        "direction": "minimize",
    }
    if mode == "accuracy":
        value = {
            "mode": mode,
            "primary_role": "accuracy",
            "metrics": [accuracy],
            "quality_constraint": None,
            "acquisition": "expected_improvement",
            "selection_policy": "highest_valid_accuracy",
        }
    elif mode == "latency":
        value = {
            "mode": mode,
            "primary_role": "latency",
            "metrics": [latency, accuracy],
            "quality_constraint": {
                "type": "relative_retention",
                "retained_fraction": 0.90,
                "reference": "best_observed_within_job",
                "reference_updates": "monotonic",
                "terminal_reference": "terminal_archive_accuracy_winner",
            },
            "acquisition": "constrained_expected_improvement",
            "selection_policy": "equivalent_fastest_accuracy_tiebreak",
        }
    else:
        value = {
            "mode": mode,
            "primary_role": "pareto",
            "metrics": [accuracy, latency],
            "quality_constraint": None,
            "acquisition": "parego_expected_improvement",
            "selection_policy": "normalized_augmented_chebyshev",
        }
    return value


def _manifest():
    parameters = {
        "model.dec_layers": {
            "type": "categorical",
            "values": [3, 4, 5, 6],
        },
        "model.enc_layers": {
            "type": "categorical",
            "values": [3, 4, 5, 6],
        },
        "train.optim.lr": {
            "type": "float",
            "minimum": 1e-5,
            "maximum": 1e-3,
            "scale": "log",
        },
    }
    ptms = [
        {
            "id": "dino.ptm-a",
            "artifact_sha256": _digest("1"),
            "registry_record_sha256": _digest("2"),
            "preflight_report_sha256": _digest("3"),
        },
        {
            "id": "dino.ptm-b",
            "artifact_sha256": _digest("4"),
            "registry_record_sha256": _digest("5"),
            "preflight_report_sha256": _digest("6"),
        },
    ]
    space_sha = canonical_sha256(parameters)
    modes = []
    for mode in CAMPAIGN_MODES:
        objective = _objective(mode)
        modes.append(
            {
                "mode": mode,
                "job_id": f"dino-pilot-{mode}",
                "seed": 314159,
                "observation_namespace": f"dino-pilot-{mode}-state",
                "observation_sharing": False,
                "initial_observation_ids": [],
                "ptm_policy": (
                    "registered_default"
                    if mode == "accuracy"
                    else "all_qualified"
                ),
                "allowed_ptm_ids": (
                    ["dino.ptm-a"]
                    if mode == "accuracy"
                    else [record["id"] for record in ptms]
                ),
                "search_space_sha256": space_sha,
                "objective": objective,
                "objective_sha256": canonical_sha256(objective),
            }
        )
    return {
        "schema_version": 1,
        "campaign_id": "dino-objective-aware-pilot-v1",
        "model": "dino",
        "task": "object_detection",
        "source": {
            "commit": "a" * 40,
            "dirty_tree_policy": "reject",
            "dirty": False,
            "diff_sha256": None,
        },
        "package": {
            "distribution": "nvidia-tao-automl",
            "version": "0.1.0",
            "wheel_sha256": _digest("7"),
        },
        "container": {
            "runtime": "enroot",
            "sqsh_uri": (
                "s3://immutable-images/"
                "tao-toolkit-7.0.1-pyt-cu128-20260729.sqsh"
            ),
            "sqsh_sha256": _digest("8"),
            "container_digest": "sha256:" + _digest("9"),
        },
        "runtime": {
            "tao_version": "7.0.1-pyt",
            "precision": "fp16",
            "train_batch_size_per_gpu": 2,
            "eval_batch_size_per_gpu": 2,
            "latency_batch_size_per_gpu": 1,
            "latency_protocol_sha256": _digest("0"),
            "latency_input_sha256": _digest("f"),
            "latency_timed_scope": "model_forward_only",
        },
        "ptms": ptms,
        "ptm_search": {
            "representation": "hierarchical_nonordinal_arms",
            "default_ptm_id": "dino.ptm-a",
            "arms": [
                {
                    "checkpoint_id": "dino.ptm-a",
                    "conditional_search_space_sha256": _digest("1"),
                    "preflight_provenance_sha256": _digest("2"),
                    "input_contract_sha256": _digest("3"),
                },
                {
                    "checkpoint_id": "dino.ptm-b",
                    "conditional_search_space_sha256": _digest("4"),
                    "preflight_provenance_sha256": _digest("5"),
                    "input_contract_sha256": _digest("6"),
                },
            ],
        },
        "dataset": {
            "id": "voc2007-detection-full-v1",
            "source": "https://host.robots.ox.ac.uk/pascal/VOC/voc2007",
            "manifest_sha256": _digest("a"),
            "conversion_sha256": _digest("b"),
            "splits": {
                "train": _digest("c"),
                "validation": _digest("d"),
                "test": _digest("e"),
            },
        },
        "algorithm": {
            "name": "bayesian",
            "implementation_version": "objective-aware-v1",
            "acquisition_version": "gp-parego-constrained-v1",
            "deterministic_replay": True,
        },
        "search_space": {
            "parameters": parameters,
            "sha256": space_sha,
        },
        "budget": {
            "max_candidates_per_mode": 6,
            "max_concurrent_candidates_per_mode": 3,
            "max_wallclock_minutes_per_mode": 1440,
            "max_terminal_failures_per_mode": 2,
        },
        "fidelity": {
            "unit": "epochs",
            "rungs": [1, 3, 6],
            "final_validation_budget": 12,
            "evaluation_interval": 1,
            "checkpoint_interval": 1,
            "policy": "frozen_successive_halving_then_equal_final_validation",
        },
        "resources": {
            "platform": "slurm",
            "nodes": 1,
            "gpus_per_node": 8,
            "tasks_per_node": 1,
            "distributed_workers_per_node": 8,
            "gpu_type": "NVIDIA-A100-SXM4-80GB",
            "cpus_per_task": 8,
            "memory_gib_per_node": 512,
            "time_limit_minutes": 480,
            "partition": "batch",
            "exclusive_node": True,
        },
        "workload": {
            "expected_candidate_jobs": 18,
            "expected_matched_validation_jobs": 18,
            "expected_total_jobs": 36,
            "estimated_storage_bytes": 2_000_000_000_000,
        },
        "retry_policy": {
            "max_retries_per_trial": 2,
            "retryable_failure_codes": [
                "node_failure",
                "scheduler_preemption",
            ],
            "preserve_failed_trials": True,
            "replacement_policy": "never_silent",
        },
        "stages": [
            {
                "name": "single_candidate_gate",
                "order": 1,
                "expected_jobs": 3,
                "entry_criteria": [
                    "local_model_preflight_passed",
                    "dataset_preflight_passed",
                    "ptm_preflight_passed",
                    "wheel_contents_verified",
                ],
                "exit_criteria": [
                    "one_candidate_per_mode_succeeded",
                    "artifact_contract_passed",
                ],
                "on_failure": "halt_before_next_stage",
            },
            {
                "name": "pilot_batch",
                "order": 2,
                "expected_jobs": 6,
                "entry_criteria": ["single_candidate_gate_passed"],
                "exit_criteria": [
                    "pilot_artifacts_passed",
                    "metric_sanity_passed",
                ],
                "on_failure": "halt_before_next_stage",
            },
            {
                "name": "full_search",
                "order": 3,
                "expected_jobs": 9,
                "entry_criteria": ["pilot_batch_passed"],
                "exit_criteria": [
                    "search_archives_sealed",
                    "selection_winners_frozen",
                ],
                "on_failure": "halt_before_next_stage",
            },
            {
                "name": "matched_validation",
                "order": 4,
                "expected_jobs": 18,
                "entry_criteria": ["selection_winners_frozen"],
                "exit_criteria": ["matched_validation_artifacts_sealed"],
                "on_failure": "halt_before_next_stage",
            },
        ],
        "cancellation": {
            "criteria": [
                "artifact_integrity_failure",
                "preflight_gate_failure",
                "metric_sanity_failure",
                "failure_budget_exceeded",
                "storage_budget_exceeded",
                "wallclock_budget_exceeded",
            ],
            "action": "cancel_pending_and_halt",
            "preserve_records": True,
        },
        "failed_trial_policy": {
            "preserve_records": True,
            "preserve_terminal_recommendation": True,
            "count_toward_candidate_budget": True,
            "silent_replacement": False,
            "terminal_status": "failed",
        },
        "agent_intervention_flags": {
            field: False for field in AGENT_INTERVENTION_FLAGS
        },
        "selection_isolation_flags": {
            field: False for field in SELECTION_ISOLATION_FLAGS
        },
        "modes": modes,
    }


def _error(document, match):
    with pytest.raises(CampaignManifestValidationError, match=match):
        create_campaign_manifest(document)


def test_valid_campaign_is_canonical_immutable_and_fair():
    source = _manifest()
    campaign = create_campaign_manifest(source)
    sealed = campaign.to_dict()

    assert sealed["manifest_sha256"] == canonical_sha256(campaign.stable_dict())
    assert campaign.fairness_audit.ok
    assert set(campaign.fairness_audit.passed_checks) >= {
        "exact_three_mode_jobs",
        "no_cross_mode_observation_sharing",
        "same_preregistered_seed",
        "latency_and_moo_full_ptm_inventory",
        "documented_accuracy_ptm_policy_exception",
        "same_search_space",
    }
    stable = campaign.stable_dict()
    moo = next(
        item for item in stable["modes"]
        if item["mode"] == "multi_objective"
    )
    assert moo["objective"]["acquisition"] == "parego_expected_improvement"
    assert "model.ptm_id" not in stable["search_space"]["parameters"]
    assert stable["dataset"]["id"] == "voc2007-detection-full-v1"
    assert load_campaign_manifest(sealed).manifest_sha256 == (
        campaign.manifest_sha256
    )

    source["source"]["commit"] = "f" * 40
    defensive = campaign.to_dict()
    defensive["source"]["commit"] = "0" * 40
    assert campaign.stable_dict()["source"]["commit"] == "a" * 40


def test_semantically_unordered_inventory_and_modes_hash_identically():
    first = _manifest()
    second = copy.deepcopy(first)
    second["ptms"].reverse()
    second["ptm_search"]["arms"].reverse()
    second["modes"].reverse()
    for mode in second["modes"]:
        mode["allowed_ptm_ids"].reverse()
    second["retry_policy"]["retryable_failure_codes"].reverse()
    second["cancellation"]["criteria"].reverse()
    for stage in second["stages"]:
        stage["entry_criteria"].reverse()
        stage["exit_criteria"].reverse()

    assert create_campaign_manifest(first).manifest_sha256 == (
        create_campaign_manifest(second).manifest_sha256
    )


def test_derived_mode_manifests_are_independent_and_bound_to_parent():
    campaign = create_campaign_manifest(_manifest())
    manifests = {
        mode: campaign.mode_manifest(mode) for mode in CAMPAIGN_MODES
    }
    assert len(
        {value["mode_manifest_sha256"] for value in manifests.values()}
    ) == 3
    for mode, value in manifests.items():
        assert value["parent_campaign_manifest_sha256"] == (
            campaign.manifest_sha256
        )
        assert value["mode_job"]["mode"] == mode
        assert value["mode_job"]["observation_sharing"] is False
        assert value["mode_job"]["initial_observation_ids"] == []
        assert value["source"]["commit"] == "a" * 40
        assert value["resources"]["nodes"] == 1
        assert value["resources"]["gpus_per_node"] == 8
        assert value["resources"]["tasks_per_node"] == 1
        assert value["resources"]["distributed_workers_per_node"] == 8
        assert value["runtime"]["precision"] == "fp16"
        assert value["runtime"]["latency_batch_size_per_gpu"] == 1
        campaign.assert_mode_resume_compatible(mode, value)


def test_resume_rejects_unsealed_tampered_or_changed_campaign():
    campaign = create_campaign_manifest(_manifest())
    campaign.assert_resume_compatible(campaign.to_dict())

    unsealed = campaign.stable_dict()
    with pytest.raises(CampaignResumeMismatchError, match="invalid or unsealed"):
        campaign.assert_resume_compatible(unsealed)

    tampered = campaign.to_dict()
    tampered["budget"]["max_wallclock_minutes_per_mode"] += 1
    with pytest.raises(CampaignResumeMismatchError, match="invalid or unsealed"):
        campaign.assert_resume_compatible(tampered)

    changed = _manifest()
    for mode in changed["modes"]:
        mode["seed"] = 271828
    changed_campaign = create_campaign_manifest(changed)
    with pytest.raises(
        CampaignResumeMismatchError,
        match="configuration identity changed",
    ):
        campaign.assert_resume_compatible(changed_campaign)


def test_mode_resume_rejects_any_shared_or_mode_config_drift():
    campaign = create_campaign_manifest(_manifest())
    persisted = campaign.mode_manifest("latency")
    persisted["mode_job"]["seed"] += 1
    with pytest.raises(CampaignResumeMismatchError, match="mode:latency"):
        campaign.assert_mode_resume_compatible("latency", persisted)

    persisted = campaign.mode_manifest("latency")
    persisted["source"]["commit"] = "f" * 40
    persisted["mode_manifest_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in persisted.items()
            if key != "mode_manifest_sha256"
        }
    )
    with pytest.raises(
        CampaignResumeMismatchError,
        match="shared configuration identity changed",
    ):
        campaign.assert_mode_resume_compatible("latency", persisted)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["source"].update(
                {"dirty": True, "diff_sha256": None}
            ),
            "source.dirty cannot be true",
        ),
        (
            lambda value: value["source"].update(
                {
                    "dirty_tree_policy": "allow_with_diff_hash",
                    "dirty": True,
                    "diff_sha256": _digest("f"),
                }
            ),
            None,
        ),
        (
            lambda value: value["source"].update(
                {"dirty": False, "diff_sha256": _digest("f")}
            ),
            "must be null for a clean tree",
        ),
        (
            lambda value: value["source"].update({"commit": "deadbee"}),
            "full lowercase Git object ID",
        ),
    ],
)
def test_dirty_tree_policy_and_full_commit_identity(mutation, match):
    document = _manifest()
    mutation(document)
    if match is None:
        campaign = create_campaign_manifest(document)
        assert campaign.stable_dict()["source"]["dirty"] is True
    else:
        _error(document, match)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["container"].update(
                {"sqsh_uri": "https://storage/image.sqsh?signature=secret"}
            ),
            "credential-free identity",
        ),
        (
            lambda value: value["container"].update(
                {"container_digest": "tao:latest"}
            ),
            "exact lowercase sha256",
        ),
        (
            lambda value: value["container"].update(
                {"sqsh_uri": "s3://immutable-images/tao-image.tar"}
            ),
            "exact .sqsh image",
        ),
        (
            lambda value: value["package"].update({"wheel_sha256": "unknown"}),
            "package.wheel_sha256",
        ),
        (
            lambda value: value["ptms"][0].update(
                {"preflight_report_sha256": "0" * 63}
            ),
            "preflight_report_sha256",
        ),
        (
            lambda value: value["dataset"]["splits"].update(
                {"validation": "not-a-hash"}
            ),
            "dataset.splits.validation",
        ),
        (
            lambda value: value["dataset"].update(
                {"source": "https://data/a?token=secret"}
            ),
            "credential-free identity",
        ),
    ],
)
def test_artifact_and_data_identities_are_exact_and_secret_free(mutation, match):
    document = _manifest()
    mutation(document)
    _error(document, match)


def test_search_space_and_objective_hashes_are_enforced():
    document = _manifest()
    document["search_space"]["parameters"]["model.enc_layers"]["values"].append(7)
    _error(document, "search_space.sha256 does not match")

    document = _manifest()
    document["modes"][0]["objective"]["selection_policy"] = "changed"
    _error(document, "objective_sha256 does not match")


def test_strict_campaign_rejects_scalarized_fallback_algorithm():
    document = _manifest()
    document["algorithm"]["name"] = "bfbo"
    _error(document, "not objective-aware.*latency")


@pytest.mark.parametrize(
    ("mode", "acquisition"),
    [
        ("accuracy", "raw_but_unknown"),
        ("latency", "expected_improvement"),
        ("multi_objective", "archive_only_post_selection"),
    ],
)
def test_strict_campaign_requires_exact_mode_acquisition(mode, acquisition):
    document = _manifest()
    record = next(item for item in document["modes"] if item["mode"] == mode)
    record["objective"]["acquisition"] = acquisition
    record["objective_sha256"] = canonical_sha256(record["objective"])

    _error(document, "objective-aware and equal")


def test_ptm_identity_is_a_hierarchical_arm_not_an_inner_parameter():
    document = _manifest()
    document["search_space"]["parameters"]["model.ptm_id"] = {
        "type": "categorical",
        "values": ["dino.ptm-a", "dino.ptm-b"],
    }
    document["search_space"]["sha256"] = canonical_sha256(
        document["search_space"]["parameters"]
    )
    _error(document, "must not encode PTM identity")

    document = _manifest()
    document["ptm_search"]["arms"].pop()
    _error(document, "every qualified PTM exactly once")


def test_accuracy_ptm_policy_exception_is_explicit_and_validated():
    document = _manifest()
    accuracy = document["modes"][0]
    accuracy["ptm_policy"] = "all_qualified_explicit"
    accuracy["allowed_ptm_ids"] = ["dino.ptm-a", "dino.ptm-b"]
    campaign = create_campaign_manifest(document)
    assert "same_preflight_ptm_inventory" in (
        campaign.fairness_audit.passed_checks
    )

    document = _manifest()
    document["modes"][0]["allowed_ptm_ids"] = ["dino.ptm-b"]
    _error(document, "registered default PTM")

    document = _manifest()
    accuracy = document["modes"][0]
    accuracy["ptm_policy"] = "user_provided"
    accuracy["allowed_ptm_ids"] = ["dino.ptm-b"]
    assert create_campaign_manifest(document).fairness_audit.ok

    document = _manifest()
    document["modes"][1]["ptm_policy"] = "registered_default"
    _error(document, "ptm_policy must be 'all_qualified'")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("nodes", 2, "resources.nodes must equal one"),
        ("gpus_per_node", 4, "gpus_per_node must equal eight"),
        ("tasks_per_node", 8, "tasks_per_node must equal one"),
        (
            "distributed_workers_per_node",
            1,
            "distributed_workers_per_node must equal eight",
        ),
        ("platform", "local", "resources.platform must be 'slurm'"),
        ("exclusive_node", False, "exclusive_node must be true"),
    ],
)
def test_slurm_contract_requires_one_exclusive_eight_gpu_node(
    field,
    value,
    match,
):
    document = _manifest()
    document["resources"][field] = value
    _error(document, match)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "latency_batch_size_per_gpu",
            0,
            "latency_batch_size_per_gpu must be an integer >= 1",
        ),
        (
            "latency_protocol_sha256",
            "unsealed",
            "latency_protocol_sha256",
        ),
        ("precision", "", "runtime.precision must be a non-empty string"),
    ],
)
def test_shared_runtime_and_latency_protocol_are_frozen(field, value, match):
    document = _manifest()
    document["runtime"][field] = value
    _error(document, match)


def test_staged_gates_and_workload_counts_are_frozen():
    document = _manifest()
    document["stages"][0]["entry_criteria"].remove("ptm_preflight_passed")
    _error(document, "missing required gates.*ptm_preflight_passed")

    document = _manifest()
    document["stages"][0]["expected_jobs"] = 6
    _error(document, "one job per mode")

    document = _manifest()
    document["stages"][2]["expected_jobs"] = 6
    _error(document, "three times.*max_candidates")

    document = _manifest()
    document["workload"]["expected_total_jobs"] = 35
    _error(document, "does not match the staged job count")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("max_retries_per_trial", 6, "exceeds the product bound"),
        ("preserve_failed_trials", False, "must be true"),
        ("replacement_policy", "replace", "never_silent"),
    ],
)
def test_retries_are_bounded_and_failed_trials_are_preserved(
    field,
    value,
    match,
):
    document = _manifest()
    document["retry_policy"][field] = value
    _error(document, match)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("preserve_records", False, "preserve_records must be true"),
        (
            "preserve_terminal_recommendation",
            False,
            "preserve_terminal_recommendation must be true",
        ),
        (
            "count_toward_candidate_budget",
            False,
            "count_toward_candidate_budget must be true",
        ),
        ("silent_replacement", True, "silent_replacement must be false"),
    ],
)
def test_terminal_failure_records_cannot_be_hidden(field, value, match):
    document = _manifest()
    document["failed_trial_policy"][field] = value
    _error(document, match)


@pytest.mark.parametrize(
    ("group", "field"),
    [
        ("agent_intervention_flags", field)
        for field in AGENT_INTERVENTION_FLAGS
    ]
    + [
        ("selection_isolation_flags", field)
        for field in SELECTION_ISOLATION_FLAGS
    ],
)
def test_all_intervention_and_selection_isolation_flags_are_immutable_false(
    group,
    field,
):
    document = _manifest()
    document[group][field] = True
    _error(document, f"{group}.{field} must be false")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["modes"][1].update(
                {"observation_sharing": True}
            ),
            "no_cross_mode_observation_sharing|disable sharing",
        ),
        (
            lambda value: value["modes"][1].update(
                {"initial_observation_ids": ["accuracy-rec-0"]}
            ),
            "must be empty",
        ),
        (
            lambda value: value["modes"][1].update(
                {"observation_namespace": value["modes"][0][
                    "observation_namespace"
                ]}
            ),
            "observation namespaces must be unique",
        ),
        (
            lambda value: value["modes"][2].update({"seed": 271828}),
            "same preregistered seed",
        ),
        (
            lambda value: value["modes"][1].update(
                {"allowed_ptm_ids": ["dino.ptm-a"]}
            ),
            "must equal the full PTM inventory",
        ),
        (
            lambda value: value["modes"][2].update(
                {"search_space_sha256": _digest("f")}
            ),
            "does not match campaign",
        ),
    ],
)
def test_three_mode_jobs_are_independent_and_fair(mutation, match):
    document = _manifest()
    mutation(document)
    _error(document, match)


@pytest.mark.parametrize(
    ("mode", "mutation", "match"),
    [
        (
            "accuracy",
            lambda objective: objective["metrics"][0].update(
                {"direction": "minimize"}
            ),
            "direction must be 'maximize'",
        ),
        (
            "accuracy",
            lambda objective: objective.update(
                {"acquisition": "archive_only_post_selection"}
            ),
            "must be objective-aware",
        ),
        (
            "latency",
            lambda objective: objective["quality_constraint"].update(
                {"reference": "external_baseline"}
            ),
            "best_observed_within_job",
        ),
        (
            "latency",
            lambda objective: objective["quality_constraint"].update(
                {"reference_updates": "oscillating"}
            ),
            "must be 'monotonic'",
        ),
        (
            "multi_objective",
            lambda objective: objective.update(
                {
                    "quality_constraint": {
                        "type": "relative_retention",
                        "value": 0.90,
                        "source": "latency_mode",
                    }
                }
            ),
            "absolute_minimum|multi_objective_explicit",
        ),
    ],
)
def test_mode_objective_contract_is_objective_aware_and_self_calibrating(
    mode,
    mutation,
    match,
):
    document = _manifest()
    mode_record = next(item for item in document["modes"] if item["mode"] == mode)
    mutation(mode_record["objective"])
    mode_record["objective_sha256"] = canonical_sha256(mode_record["objective"])
    _error(document, match)


def test_explicit_independent_multi_objective_floor_is_permitted():
    document = _manifest()
    mode = next(
        item for item in document["modes"]
        if item["mode"] == "multi_objective"
    )
    mode["objective"]["quality_constraint"] = {
        "type": "absolute_minimum",
        "value": 0.25,
        "source": "multi_objective_explicit",
    }
    mode["objective_sha256"] = canonical_sha256(mode["objective"])
    assert create_campaign_manifest(document).fairness_audit.ok


def test_unknown_fields_nan_and_bad_seals_fail_closed():
    document = _manifest()
    document["resources"]["gres"] = "gpu:8"
    _error(document, "not a recognized field")

    document = _manifest()
    document["search_space"]["parameters"]["bad"] = float("nan")
    _error(document, "finite canonical JSON")

    document = _manifest()
    document["schema_version"] = True
    _error(document, "schema_version must equal")

    campaign = create_campaign_manifest(_manifest())
    sealed = campaign.to_dict()
    sealed["manifest_sha256"] = _digest("f")
    with pytest.raises(CampaignManifestValidationError, match="does not match"):
        load_campaign_manifest(sealed)

    with pytest.raises(CampaignManifestValidationError, match="must not predeclare"):
        create_campaign_manifest(campaign.to_dict())

    with pytest.raises(CampaignManifestValidationError, match="root must be"):
        create_campaign_manifest(None)

    with pytest.raises(CampaignResumeMismatchError, match="not an object"):
        campaign.assert_mode_resume_compatible("accuracy", None)


def test_malformed_fairness_scalars_fail_closed_without_type_errors():
    document = _manifest()
    document["modes"][0]["job_id"] = {"unexpected": "object"}
    document["modes"][1]["seed"] = {"unexpected": "object"}
    document["modes"][2]["observation_namespace"] = ["unexpected"]
    with pytest.raises(CampaignManifestValidationError) as error:
        create_campaign_manifest(document)
    message = str(error.value)
    assert "job_id must be a non-empty string" in message
    assert "seed must be an integer" in message
    assert "observation_namespace must be a non-empty string" in message
