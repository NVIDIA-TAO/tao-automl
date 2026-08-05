from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import low_latency_followup_runner as followup  # noqa: E402


def load_contract() -> tuple[dict, dict, str]:
    path = followup.DEFAULT_MANIFEST
    whole = followup.expanded.sha256_file(path)
    manifest, observed = followup.load_followup_manifest(
        path, supplied_file_sha256=whole
    )
    assert observed == whole
    projected = followup.project_execution_manifest(manifest)
    return manifest, projected, whole


def test_manifest_is_canonical_and_pins_sealed_inputs():
    manifest, _projected, _whole = load_contract()
    source = manifest["sealed_source"]
    assert source["expanded_search_manifest"]["whole_file_sha256"] == (
        followup.EXPECTED_BASE_MANIFEST_SHA256
    )
    assert source["expanded_search_runner"]["sha256"] == (
        followup.EXPECTED_BASE_RUNNER_SHA256
    )
    assert source["existing_archive"]["candidate_table_sha256"] == (
        followup.EXPECTED_ARCHIVE_TABLE_SHA256
    )
    assert source["existing_archive"]["combined_selection_sha256"] == (
        followup.EXPECTED_COMBINED_SELECTION_SHA256
    )
    assert source["existing_archive"]["optimizer_prior_input"] is False
    assert source["existing_archive"]["final_selection_input"] is True


def test_search_seeds_are_exactly_sha_derived():
    manifest, _projected, _whole = load_contract()
    records = manifest["search_design"]["seed_derivation"]["records"]
    assert tuple(record["seed"] for record in records) == followup.EXPECTED_SEEDS
    for index, record in enumerate(records):
        seed, digest = followup.derive_seed(
            f"dino-low-latency-followup-v1:{index}"
        )
        assert digest == hashlib.sha256(
            record["material_utf8"].encode("utf-8")
        ).hexdigest()
        assert digest == record["sha256"]
        assert seed == record["seed"]


def test_budget_ranges_and_runtime_are_frozen_before_results():
    manifest, projected, _whole = load_contract()
    design = manifest["search_design"]
    assert design["recommendations_per_seed"] == 20
    assert design["new_candidate_budget"] == 60
    assert design["optimizer_generation_population"] == "new_candidates_only"
    assert design["optimizer_selection_mode"] == "latency"
    assert design["final_selection_population"] == (
        "sealed_60_plus_all_valid_new_candidates"
    )
    domains = manifest["search_space"]["search_domains"]
    assert domains["model.enc_layers"]["valid_options"] == [3, 4, 5, 6]
    assert domains["model.dec_layers"]["valid_options"] == [3, 4, 5, 6]
    assert domains["train.optim.lr"] == {
        "representation": "continuous",
        "valid_min": 1e-05,
        "valid_max": 0.0005,
    }
    assert domains["train.optim.weight_decay"] == {
        "representation": "continuous",
        "valid_min": 1e-05,
        "valid_max": 0.001,
    }
    assert projected["frozen_identity"]["training_controls"]["train_epochs"] == 10
    assert projected["frozen_identity"]["runtime"]["num_nodes"] == 1
    assert projected["frozen_identity"]["runtime"]["gpu_count_per_node"] == 8
    assert projected["frozen_identity"]["runtime"]["sqsh_path"].endswith(".sqsh")


def test_latency_policy_is_relative_90_percent_and_multiobjective_independent():
    manifest, projected, _whole = load_contract()
    policy = manifest["selection"]
    assert policy["latency_accuracy_retention"] == {
        "type": "relative",
        "retained_fraction": 0.90,
        "reference": "accuracy_winner",
    }
    assert policy["multi_objective_min_accuracy"] is None
    assert policy["latency_tolerance_ms"] == pytest.approx(0.73553775)
    opportunity = policy["fixed_opportunity_question"]
    assert opportunity["minimum_mAP50"] == pytest.approx(
        0.90 * opportunity["reference_accuracy"]
    )
    assert opportunity[
        "does_not_replace_relative_policy_for_final_union_selection"
    ] is True
    selector = followup.validate_selector_configuration(projected)
    parsed = selector["parsed_selection"]
    assert parsed["mode"] == "latency"
    assert parsed["latency_accuracy_retention"]["type"] == "relative"
    assert parsed["latency_accuracy_retention"]["value"] == pytest.approx(0.90)
    assert parsed["multi_objective_min_accuracy"] is None


def test_every_trial_artifact_and_produced_checkpoint_is_retained():
    manifest, projected, _whole = load_contract()
    assert manifest["runtime"]["retain_every_trial_artifact"] is True
    assert manifest["runtime"]["retain_every_produced_checkpoint"] is True
    settings = followup.selection_settings(projected, followup.EXPECTED_SEEDS[0])
    assert settings["selection_mode"] == "latency"
    assert settings["objectives"] == [
        {"metric": "mAP50", "direction": "maximize", "weight": 1.0},
        {"metric": "latency_ms", "direction": "minimize", "weight": 1.0},
    ]
    assert settings["automl_delete_intermediate_ckpt"] is False
    assert settings["automl_checkpoint_retention_strategy"] == "terminal"
    assert settings["automl_max_concurrent"] == 1
    assert settings["automl_max_recommendations"] == 20


def test_execution_projection_reuses_sealed_dino_protocol_without_mutation():
    manifest, projected, _whole = load_contract()
    sealed = followup.load_sealed_base_manifest(manifest)
    assert sealed["search_design"]["search_seeds"] == [314159, 271828, 161803]
    assert sealed["selection"]["latency_mode"]["latency_accuracy_retention"][
        "retained_fraction"
    ] == pytest.approx(0.98)
    assert projected["search_design"]["search_seeds"] == list(
        followup.EXPECTED_SEEDS
    )
    assert projected["selection"]["latency_mode"][
        "latency_accuracy_retention"
    ]["retained_fraction"] == pytest.approx(0.90)
    assert projected["selection"]["multi_objective_mode"][
        "multi_objective_min_accuracy"
    ] is None
    assert projected["frozen_identity"] == sealed["frozen_identity"]
    assert followup.expanded.sha256_file(followup.SEALED_BASE_MANIFEST) == (
        followup.EXPECTED_BASE_MANIFEST_SHA256
    )


def test_sealed_archive_is_loaded_but_never_used_as_optimizer_prior():
    manifest, _projected, _whole = load_contract()
    rows = followup.load_existing_archive_rows(manifest)
    assert len(rows) == 60
    assert len({row["candidate_id"] for row in rows}) == 60
    assert manifest["search_design"]["optimizer_generation_population"] == (
        "new_candidates_only"
    )
    prospective_new_ids = {
        followup.expanded.candidate_id(seed, rec_id)
        for seed in followup.EXPECTED_SEEDS
        for rec_id in range(followup.EXPECTED_RECOMMENDATIONS_PER_SEED)
    }
    assert not prospective_new_ids.intersection(
        {row["candidate_id"] for row in rows}
    )


def test_candidate_order_invariance_is_required_by_union_combiner():
    source = Path(followup.__file__).read_text(encoding="utf-8")
    assert "expanded.analyze_union_archive(" in source
    assert '"order_independence_audit": order_audit' in source
    assert "manual_override_used" in source
    assert "optimizer_prior_candidate_count" in source


def test_matched_measurements_are_isolated_from_search_and_selection():
    manifest, _projected, _whole = load_contract()
    prohibitions = manifest["frozen_prohibitions"]
    assert prohibitions["matched_measurements_feed_selection"] is False
    assert prohibitions["matched_measurements_feed_reselection"] is False
    assert prohibitions["winner_override"] is False
    assert prohibitions["post_result_threshold_change"] is False
    assert prohibitions["post_result_range_change"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("search_design", "new_candidate_budget"), 59),
        (("search_design", "search_seeds"), [1, 2, 3]),
        (
            ("selection", "latency_accuracy_retention", "retained_fraction"),
            0.98,
        ),
        (("selection", "multi_objective_min_accuracy"), 0.5),
        (("runtime", "gpus_per_node"), 1),
        (("frozen_prohibitions", "winner_override"), True),
    ],
)
def test_contract_rejects_post_freeze_changes(path, value):
    manifest, _projected, _whole = load_contract()
    changed = copy.deepcopy(manifest)
    cursor = changed
    for component in path[:-1]:
        cursor = cursor[component]
    cursor[path[-1]] = value
    with pytest.raises(followup.ContractError):
        followup.validate_followup_manifest(changed)


def test_wrong_manifest_hash_is_rejected():
    with pytest.raises(followup.ContractError):
        followup.load_followup_manifest(
            followup.DEFAULT_MANIFEST,
            supplied_file_sha256="0" * 64,
        )


def test_local_preflight_binds_sdk_sqsh_dataset_and_policy_profile():
    manifest, projected, _whole = load_contract()
    checks = followup.validate_local_provenance(
        manifest, projected, followup.DEFAULT_MANIFEST
    )
    assert checks["tao_sdk"]["branch"] == (
        "rarunachalam/pre-platform-sdk-removal-20260714"
    )
    assert checks["tao_skills"]["branch"] == (
        "rarunachalam/pre-platform-sdk-removal-20260714"
    )
    assert checks["latency_policy_profile"]["sha256"] == (
        "f6e56ff8d61c91654a13c9759d7cc63f371ed66a9e958bb90ae56cba5112739e"
    )
    assert projected["frozen_identity"]["dataset"]["source_uri"] == (
        followup.EXPECTED_SCOPE["dataset_uri"]
    )
    assert projected["frozen_identity"]["runtime"]["sqsh_sha256"] == (
        "88ba75e3a8eb9524fc0dbf026f2ea5da2c68696ae8d918b0afde5e0384ca641e"
    )
