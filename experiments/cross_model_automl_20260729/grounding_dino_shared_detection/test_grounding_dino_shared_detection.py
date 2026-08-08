"""Tests for fail-closed Grounding DINO shared-detection preparation."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tao_automl.ptm_registry import load_ptm_registry

from .contract import (
    AGENT_FLAGS,
    MODEL_ID,
    MODES,
    SELECTION_FLAGS,
    PreparationError,
    build_preparation,
    derive_category_contract,
    derive_official_ptms,
    grounding_annotation_contract,
    read_json,
    validate_preparation,
)
from .dataset_conversion import validate_conversion_manifest
from .dataset_stage import validate_stage_record
from .future_contract import validate_future_contract
from .qualification_campaign import (
    CampaignExecutionError,
    _gpu_guard,
    _remote_file,
)
from . import qualification_campaign
from .runtime_input_stage import (
    HF_REQUIRED_FILES,
    validate_runtime_input_stage,
)
from .successor_contract import validate_successor_contract
from .automatic_trigger import readiness
from . import automatic_trigger


HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def inputs():
    return read_json(HERE / "campaign.inputs.v1.json")


@pytest.fixture(scope="module")
def preparation(inputs):
    return build_preparation(experiment_dir=HERE, inputs=inputs)


def test_exact_repository_model_identifier_and_skill(preparation):
    assert preparation["model"]["id"] == "grounding_dino"
    assert preparation["model"]["skill"] == "tao-train-grounding-dino"
    assert preparation["automl"]["search_schema"]["model_id"] == MODEL_ID


def test_category_prompts_are_derived_verbatim_from_source(preparation):
    category = preparation["dataset"]["category_contract"]
    assert category["source_category_ids"] == [1, 2, 3, 4]
    assert category["model_category_ids"] == [0, 1, 2, 3]
    assert category["prompt_list"] == [
        "cone",
        "forklift",
        "cart",
        "fire_extinguisher",
    ]
    assert category["label_map"] == {
        "0": "cone",
        "1": "forklift",
        "2": "cart",
        "3": "fire_extinguisher",
    }
    assert category["manual_prompt_or_synonym_injection"] is False
    assert {
        item["derivation"] for item in category["prompt_mapping"]
    } == {"exact_source_coco_category_name"}


def test_category_mapping_is_input_order_invariant(inputs):
    manifest_path = (HERE / inputs["dataset"]["manifest_file"]).resolve()
    audit_path = (HERE / inputs["dataset"]["audit_file"]).resolve()
    manifest = read_json(manifest_path)
    audit = read_json(audit_path)
    expected = derive_category_contract(manifest, audit)

    changed_manifest = copy.deepcopy(manifest)
    changed_manifest["categories"].reverse()
    changed_audit = copy.deepcopy(audit)
    for split in ("train", "validation"):
        changed_audit["splits"][split]["categories"].reverse()
    assert derive_category_contract(changed_manifest, changed_audit) == expected


def test_duplicate_or_inconsistent_categories_fail_closed(inputs):
    manifest = read_json((HERE / inputs["dataset"]["manifest_file"]).resolve())
    audit = read_json((HERE / inputs["dataset"]["audit_file"]).resolve())
    duplicate = copy.deepcopy(manifest)
    duplicate["categories"][1]["id"] = duplicate["categories"][0]["id"]
    with pytest.raises(PreparationError, match="unique"):
        derive_category_contract(duplicate, audit)

    mismatched = copy.deepcopy(audit)
    mismatched["splits"]["validation"]["categories"][0]["name"] = "invented"
    with pytest.raises(PreparationError, match="identities differ"):
        derive_category_contract(manifest, mismatched)


def test_plain_coco_is_not_misrepresented_as_phrase_grounding(inputs):
    audit = read_json((HERE / inputs["dataset"]["audit_file"]).resolve())
    contract = grounding_annotation_contract(audit)
    assert contract["category_prompted_detection"]["supported_by_source"] is True
    phrase = contract["referring_expression_box_grounding"]
    assert phrase["supported_by_source"] is False
    assert phrase["images_with_caption"] == 0
    assert phrase["annotations_with_tokens_positive"] == 0
    assert "must not be represented as referring expressions" in phrase["blocker"]


def test_every_official_repository_ptm_is_derived_without_manual_filtering():
    records = derive_official_ptms()
    registry_records = load_ptm_registry().to_dict()["models"][MODEL_ID][
        "checkpoints"
    ]
    expected = sorted(
        item["id"]
        for item in registry_records
        if item["source"].get("official") is True
    )
    assert [item["id"] for item in records] == expected
    assert len(records) == 2
    assert all(item["status"] == "supported" for item in records)
    assert all(
        item["validation"]["status"] == "validated"
        and item["validation"]["tao_version"] == "7.1.0-rc-245"
        for item in records
    )
    for item in records:
        validation = item["validation"]
        assert "sqsh-sha256:e36640f9" in validation["container_identity"]
        assert (
            "completion_sha256="
            "172688d1af2479886c46f55fa148bf43a7487b517b2ea145c3359136100de698"
            in validation["evidence"]
        )
        assert (
            "pilot_handoff_sha256="
            "318d93fc04260650a67452eb00a710658744bf83fe3462115a9c320b12315ec8"
            in validation["evidence"]
        )


def test_schema_search_space_is_repository_derived(preparation):
    schema = preparation["automl"]["search_schema"]
    assert schema["source"] == "packaged_train_schema_automl_default_parameters"
    assert schema["parameter_names"] == sorted(schema["parameter_names"])
    assert "model.enc_layers" in schema["parameter_names"]
    assert "model.dec_layers" in schema["parameter_names"]
    assert "model.num_select" in schema["parameter_names"]
    assert all(
        node["automl_enabled"] is True
        for node in schema["parameters"].values()
    )


def test_three_independent_objective_aware_jobs_are_prepared(preparation):
    modes = preparation["automl"]["modes"]
    assert tuple(item["mode"] for item in modes) == MODES
    assert len({item["observation_namespace"] for item in modes}) == 3
    assert all(item["observation_sharing"] is False for item in modes)
    assert all(item["initial_observation_ids"] == [] for item in modes)
    assert [item["objective"]["acquisition"] for item in modes] == [
        "expected_improvement",
        "constrained_expected_improvement",
        "parego_expected_improvement",
    ]
    latency = modes[1]["objective"]["quality_constraint"]
    assert latency["retained_fraction"] == pytest.approx(0.90)
    assert latency["reference"] == "best_observed_within_job"
    assert latency["reference_updates"] == "monotonic"
    assert modes[2]["objective"]["quality_constraint"] is None


def test_all_jobs_are_direct_eight_gpu_sqsh_jobs(preparation):
    runtime = preparation["runtime"]
    assert runtime["platform"] == "slurm"
    assert runtime["nodes"] == 1
    assert runtime["gpus_per_node"] == 8
    assert runtime["distributed_workers_per_node"] == 8
    assert runtime["sqsh_direct_path"] is True
    assert runtime["slurm_use_sqsh_conversion"] is False
    assert runtime["sqsh_path"].endswith(".sqsh")
    assert preparation["direct_qualification"]["cpu_model_runs"] is False
    assert preparation["direct_qualification"]["smoke_or_ministep_runs"] is False


def test_qualification_covers_every_official_ptm(preparation):
    inventory = preparation["official_ptm_inventory"]
    jobs = preparation["direct_qualification"]["jobs"]
    assert inventory["manual_ptm_selection"] is False
    assert [job["ptm_id"] for job in jobs] == inventory["candidate_ids"]
    assert all(job["resource"]["gpus_per_node"] == 8 for job in jobs)
    assert all(
        job["workflow"][0] == "full_10_epoch_train_with_validation_each_epoch"
        for job in jobs
    )


def test_metric_contract_is_task_specific_and_ptm_gate_remains_closed(preparation):
    metric = preparation["metric_contract"]
    assert metric["val_mAP50"]["availability"] == "supported"
    assert metric["val_mAP50"]["task"] == "object_detection"
    assert metric["val_Pr@0.5"]["availability"] == "blocked"
    assert (
        metric["val_Pr@0.5"]["task"]
        == "referring_expression_box_grounding"
    )
    gate = preparation["automatic_gate"]
    assert gate["launch_authorized"] is False
    codes = {item["code"] for item in gate["blockers"]}
    assert "referring_expression_annotation_contract_missing" in codes
    assert "category_detection_metric_policy_not_supported" not in codes
    assert "official_ptms_not_production_qualified" not in codes
    assert "converted_dataset_artifacts_not_sealed" in codes


def test_no_agent_or_validation_measurement_can_change_selection(preparation):
    assert set(preparation["agent_intervention_flags"]) == set(AGENT_FLAGS)
    assert set(preparation["selection_isolation_flags"]) == set(SELECTION_FLAGS)
    assert not any(preparation["agent_intervention_flags"].values())
    assert not any(preparation["selection_isolation_flags"].values())


def test_preparation_performs_no_launch_or_model_execution(preparation):
    assert preparation["execution"] == {
        "jobs_submitted": 0,
        "scheduler_mutation_performed": False,
        "model_execution_performed": False,
    }
    validate_preparation(preparation)


def test_preparation_hash_detects_mutation(preparation):
    changed = copy.deepcopy(preparation)
    changed["runtime"]["gpus_per_node"] = 4
    with pytest.raises(PreparationError, match="eight-GPU"):
        validate_preparation(changed)

    changed = copy.deepcopy(preparation)
    changed["execution"]["jobs_submitted"] = 1
    with pytest.raises(PreparationError, match="model execution"):
        validate_preparation(changed)


def test_committed_preparation_artifact_is_valid_and_non_launching():
    document = read_json(HERE / "campaign.preparation.v1.json")
    validate_preparation(document)
    assert document["automatic_gate"]["launch_authorized"] is False
    assert document["execution"]["jobs_submitted"] == 0
    assert document["source"]["dirty"] is False
    assert document["metric_contract"]["val_mAP50"]["availability"] == "unregistered"


def test_committed_conversion_manifest_is_deterministic_and_annotation_lossless():
    document = read_json(HERE / "dataset_conversion.v1.json")

    validate_conversion_manifest(document)
    assert document["determinism"]["independent_runs"] == 2
    assert document["determinism"]["byte_identical"] is True
    assert document["semantic_validation"]["train"][
        "source_annotation_count"
    ] == 8395
    assert document["semantic_validation"]["train"][
        "output_instance_count"
    ] == 8395
    assert document["semantic_validation"]["train"][
        "excluded_empty_image_count"
    ] == 49
    assert len(
        document["semantic_validation"]["train"]["excluded_empty_image_ids"]
    ) == 49
    assert document["semantic_validation"]["validation"][
        "source_annotation_count"
    ] == 2186
    assert document["semantic_validation"]["validation"][
        "output_annotation_count"
    ] == 2186
    assert document["semantic_validation"]["validation"][
        "all_images_preserved"
    ] is True
    assert document["semantic_validation"]["annotation_lossless"] is True
    assert document["semantic_validation"]["image_count_lossless"] is False


def test_committed_stage_record_binds_nonwritable_lustre_files():
    document = read_json(HERE / "dataset_stage.v1.json")

    validate_stage_record(document)
    publication = document["publication"]
    assert publication["inside_existing_source_dataset_tree"] is True
    assert publication["all_hashes_and_sizes_match"] is True
    assert publication["published_files_nonwritable"] is True
    assert publication["lustre_root"].endswith(
        "tao_od_synthetic_full_dino_coco/grounding_dino_odvg_v1"
    )
    assert document["selection_or_execution"]["scheduler_jobs_submitted"] == 0


def test_committed_successor_contract_is_complete_and_fail_closed():
    document = read_json(HERE / "successor.contract.v1.json")

    validate_successor_contract(document)
    trigger = document["automatic_trigger"]
    assert trigger["launch_authorized"] is False
    assert document["execution"]["jobs_submitted"] == 0
    codes = {item["code"] for item in trigger["blockers"]}
    assert "rtdetr_first_candidate_gate_not_passed" not in codes
    assert "deformable_detr_first_candidate_gate_not_passed" in codes
    assert "official_ptm_checkpoints_not_staged" in codes
    assert "official_ptms_not_full_gpu_qualified" in codes
    assert document["predecessor_first_candidate_gates"]["rtdetr"][
        "passed"
    ] is True


def test_successor_qualifies_every_official_ptm_on_direct_eight_gpu_sqsh():
    document = read_json(HERE / "successor.contract.v1.json")
    inventory = document["ptm_inventory"]
    jobs = inventory["qualification_jobs"]

    assert inventory["manual_ptm_selection"] is False
    assert [item["ptm_id"] for item in jobs] == sorted(
        item["id"] for item in inventory["records"]
    )
    assert len(jobs) == 2
    for job in jobs:
        resources = job["resources"]
        assert resources["nodes"] == 1
        assert resources["gpus_per_node"] == 8
        assert resources["use_sqsh_conversion"] is False
        assert resources["sqsh_path"].endswith(".sqsh")
        assert job["production_preflight"]["cpu_load_smoke"] is False
        assert job["train"]["spec"]["train"]["num_gpus"] == 8
        assert job["train"]["spec"]["train"]["gpu_ids"] == list(range(8))
        assert job["train"]["spec"]["train"]["is_dry_run"] is False
        assert job["train"]["spec"]["dataset"]["eval_class_ids"] == [0, 1, 2, 3]
        assert job["evaluate"]["spec"]["evaluate"]["num_gpus"] == 8
        assert (
            job["evaluate"]["test_metric_may_not_feed_automl_selection"]
            is True
        )


def test_successor_mode_pilots_are_independent_and_algorithm_generated():
    document = read_json(HERE / "successor.contract.v1.json")
    jobs = document["automl_successor"]["mode_jobs"]

    assert [item["mode"] for item in jobs] == list(MODES)
    assert len(
        {item["independent_observation_namespace"] for item in jobs}
    ) == 3
    assert all(item["observation_sharing"] is False for item in jobs)
    assert all(item["candidate_generation"] == "algorithm_only" for item in jobs)
    assert [item["acquisition"] for item in jobs] == [
        "expected_improvement",
        "constrained_expected_improvement",
        "parego_expected_improvement",
    ]
    assert jobs[1]["objective"]["latency_accuracy_retention"][
        "retained_fraction"
    ] == pytest.approx(0.90)
    assert jobs[0]["objective"]["latency_accuracy_retention"] is None
    assert jobs[2]["objective"]["latency_accuracy_retention"] is None


def test_runtime_input_stage_seals_every_official_ptm_and_immutable_bert():
    inputs = read_json(HERE / "campaign.inputs.v3.json")
    document = read_json(HERE / "runtime_inputs.stage.v1.json")

    validate_runtime_input_stage(document, inputs=inputs)
    assert [item["id"] for item in document["official_ptms"]] == [
        "grounding_dino.commercial.swin_tiny.trainable.v1.0",
        "grounding_dino.commercial.swin_tiny.trainable.v1.1",
    ]
    assert document["official_ptms"][0]["verification_mode"] == (
        "immutable_identity_observed_sha256"
    )
    assert document["official_ptms"][1]["observed_sha256"] == (
        "8ea7e089e174e72a7fe57ff63cdba5e1e4994b159e41cf72122a7e0d841beaa6"
    )
    assert document["text_encoder"]["revision"] == (
        "86b5e0934494bd15c9632b12f734a8a67f723594"
    )
    assert [item["path"] for item in document["text_encoder"]["files"]] == list(
        HF_REQUIRED_FILES
    )
    assert document["execution"]["cpu_model_runs"] == 0
    assert document["execution"]["gpu_model_runs"] == 0
    assert document["execution"]["scheduler_jobs_submitted"] == 0


def test_future_contract_binds_only_fresh_ddetr_candidate_zero_gate():
    document = read_json(HERE / "successor.runtime.contract.v2.json")
    validate_future_contract(document)

    dependency = document["predecessor_release"]["deformable_detr"]
    assert dependency["artifact_path"].endswith(
        "deformable_detr_automl_synthetic_structured_config_fix_v1/"
        "first_candidate_gate/automatic_release.json"
    )
    assert dependency["static_campaign_manifest_sha256"] == (
        "d70063f3fc6c4ed7c44d8c7d979e2dc3ffc27f576ddd13cf000648a2c2a26e83"
    )
    assert dependency["source_head"] == (
        "8386f524502b1ae7e1a021a37ed8128e7a2fb719"
    )
    assert "candidates 1-19" in dependency["release_scope"]
    assert (
        document["automatic_trigger"]["predecessor_waits_for_full_budget"]
        is False
    )


def test_sealed_future_contract_remains_valid_after_registry_promotion():
    stage = read_json(HERE / "runtime_inputs.stage.v1.json")
    expected = read_json(HERE / "successor.runtime.contract.v2.json")
    validate_future_contract(expected)
    assert {
        job["registry_status_before_qualification"]
        for job in expected["qualification"]["jobs"]
    } == {"unverified"}
    text_root = stage["text_encoder"]["lustre_root"]
    for job in expected["qualification"]["jobs"]:
        assert job["train"]["spec"]["model"]["text_encoder_type"] == text_root
        assert job["evaluate"]["spec"]["model"]["text_encoder_type"] == text_root
        assert job["resources"]["gpus_per_node"] == 8
        assert job["train"]["spec"]["train"]["is_dry_run"] is False


def test_missing_fresh_ddetr_release_is_the_only_trigger_blocker(monkeypatch):
    contract = read_json(HERE / "successor.runtime.contract.v2.json")
    inputs = read_json(HERE / "campaign.inputs.v3.json")
    monkeypatch.setattr(
        automatic_trigger,
        "evaluate_fresh_ddetr_gate",
        lambda configuration: {
            "model": "deformable_detr",
            "artifact_path": configuration["artifact_path"],
            "expected_static_campaign_manifest_sha256": configuration[
                "static_campaign_manifest_sha256"
            ],
            "passed": False,
            "blockers": ["required automatic release artifact is absent"],
        },
    )

    result = readiness(contract=contract, inputs=inputs)
    assert result["ready"] is False
    assert result["waits_for_ddetr_full_budget"] is False
    assert result["rtdetr"]["passed"] is True
    assert result["deformable_detr"]["passed"] is False
    assert [item["code"] for item in result["blockers"]] == [
        "fresh_ddetr_three_mode_candidate_zero_gate_pending"
    ]


def test_gpu_qualification_guard_requires_eight_a100_or_h100_devices():
    command = _gpu_guard("grounding_dino train -e {config_path}")
    assert "wc -l)\" -eq 8" in command
    assert "NVIDIA (A100|H100)" in command
    assert "HF_HUB_OFFLINE=1" in command
    assert "TRANSFORMERS_OFFLINE=1" in command
    assert "grounding_dino train -e {config_path}" in command


def test_sqsh_content_identity_does_not_invent_a_read_only_mode(monkeypatch):
    digest = "a" * 64
    monkeypatch.setattr(
        qualification_campaign.workflow_support,
        "remote_output",
        lambda *_args, **_kwargs: f"123 644\n{digest}  /runtime.sqsh\n",
    )

    observed = _remote_file(
        "/runtime.sqsh",
        expected_size=123,
        expected_sha256=digest,
        require_nonwritable=False,
    )
    assert observed["mode"] == "644"
    with pytest.raises(CampaignExecutionError, match="identity changed"):
        _remote_file(
            "/staged-checkpoint.pth",
            expected_size=123,
            expected_sha256=digest,
            require_nonwritable=True,
        )
