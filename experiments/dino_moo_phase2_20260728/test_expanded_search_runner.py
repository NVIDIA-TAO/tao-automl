"""Unit tests for the manifest-driven expanded DINO search harness."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "expanded_search_runner.py"
POLICY_PATH = HERE / "expanded_search_derivation_policy.v1.json"
ONE_FACTOR_PATH = HERE / "one_factor_sensitivity_manifest.v1.json"
TRAIN_SCHEMA_PATH = Path(
    "/localhome/local-rarunachalam/tao-skills-external/"
    "skills/models/tao-train-dino/schemas/train.schema.json"
)
TRAIN_TEMPLATE_PATH = Path(
    "/localhome/local-rarunachalam/tao-skills-external/"
    "skills/models/tao-train-dino/references/spec_template_train.yaml"
)
EVALUATE_TEMPLATE_PATH = Path(
    "/localhome/local-rarunachalam/tao-skills-external/"
    "skills/models/tao-train-dino/references/spec_template_evaluate.yaml"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "expanded_search_runner_test_module",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_module()


def fixture_manifest() -> dict:
    policy = json.loads(POLICY_PATH.read_text())
    one = json.loads(ONE_FACTOR_PATH.read_text())
    axes = []
    for axis in policy["architecture_axis_policy"]["axes"]:
        axes.append(
            {
                "path": axis["path"],
                "reference": axis["reference"],
                "search_domain": copy.deepcopy(axis["search_domain"]),
                "compatibility_constraint": axis.get(
                    "compatibility_constraint"
                ),
                "qualified_non_reference_levels": [
                    next(
                        iter(
                            int(level)
                            for level in axis["non_reference_profile_ids"]
                        )
                    )
                ],
                "qualification_basis": (
                    "at_least_one_direction_agnostic_latency_effect_qualified"
                ),
                "preregistered_levels": copy.deepcopy(
                    axis["preregistered_levels"]
                ),
            }
        )
    training = copy.deepcopy(policy["always_included_training_parameters"])
    parameters = [axis["path"] for axis in axes] + [
        item["path"] for item in training
    ]
    domains = {
        axis["path"]: copy.deepcopy(axis["search_domain"]) for axis in axes
    }
    domains.update(
        {
            item["path"]: copy.deepcopy(item["search_domain"])
            for item in training
        }
    )
    selection = copy.deepcopy(policy["selection_contract"])
    selection["latency_tolerance"]["value_ms"] = 0.75
    manifest = {
        "schema_version": 1,
        "manifest_id": "dino_expanded_search_20260728_v1",
        "status": "preregistered_ready_to_launch",
        "scope": copy.deepcopy(policy["scope"]),
        "feeds_final_selection": True,
        "manual_override_permitted": False,
        "algorithm_only_selection_required": True,
        "derivation": {
            "sensitivity_result_path": str(HERE / "synthetic_result.json"),
            "sensitivity_result_sha256": (
                runner.EXPECTED_SENSITIVITY_RESULT_SHA256
            ),
            "sensitivity_report_sha256": (
                runner.EXPECTED_SENSITIVITY_REPORT_SHA256
            ),
            "runner_path": str(MODULE_PATH.resolve()),
            "runner_sha256": runner.sha256_file(MODULE_PATH),
            "analysis_erratum_contract_sha256": (
                runner.EXPECTED_ANALYSIS_ERRATUM_CONTRACT_SHA256
            ),
            "post_front_contract_sha256": (
                runner.EXPECTED_POST_FRONT_CONTRACT_SHA256
            ),
            "source_identity": {
                "sensitivity_manifest_path": str(
                    HERE / "sensitivity_latency_manifest.v2.json"
                ),
                "sensitivity_manifest_sha256": "2" * 64,
                "analysis_erratum": {
                    "contract_sha256": (
                        runner.EXPECTED_ANALYSIS_ERRATUM_CONTRACT_SHA256
                    ),
                    "sha256": (
                        "8e19287bf2ffd674f62b21cdaf11e000"
                        "b0eae1ed8af9d0ada1238491588993f2"
                    ),
                    "erratum_id": (
                        "dino_sensitivity_latency_analysis_erratum_"
                        "20260728_v1"
                    ),
                    "corrected_aggregator_sha256": (
                        "9209e748093e0555fe5cba339327a821"
                        "6744ec9ca6b9dae276c7041703a409c6"
                    ),
                    "submission_ledger_sha256": (
                        "b1c170c0d4697463d171cbeca3e4adcbd"
                        "34cc1cb7429c236f48b58c46c3b6d54"
                    ),
                },
            },
            "accuracy_retention_used_for_axis_derivation": False,
            "qualified_value_hull_used": False,
            "manual_override_used": False,
        },
        "search_space": {
            "architecture_axes": axes,
            "always_included_training_parameters": training,
            "search_parameters": parameters,
            "search_domains": domains,
            "compatibility_constraints": [
                "model.num_queries >= model.num_select"
            ],
            "reference_model_spec": copy.deepcopy(one["reference"]["model"]),
            "reference_optimizer": copy.deepcopy(one["reference"]["optimizer"]),
        },
        "search_design": copy.deepcopy(policy["search_design"]),
        "selection": selection,
        "post_front_matched_validation": copy.deepcopy(
            policy["post_front_matched_validation"]
        ),
        "frozen_identity": copy.deepcopy(policy["frozen_identity"]),
    }
    manifest["manifest_sha256"] = runner.sha256_value(manifest)
    return manifest


def valid_specs(manifest: dict) -> dict:
    return {
        "model.num_queries": 594,
        "model.enc_layers": 6,
        "model.dec_layers": 6,
        "model.num_select": 300,
        "train.optim.lr": 0.0002,
        "train.optim.weight_decay": 0.0001,
    }


def test_manifest_contract_separates_latency_and_multiobjective_constraints():
    manifest = fixture_manifest()
    runner.validate_manifest_contract(manifest)
    parsed = runner.validate_selector_configuration(manifest)

    settings = parsed["settings"]
    assert settings["latency_accuracy_retention"] == {
        "type": "relative",
        "retained_fraction": 0.98,
        "reference": "accuracy_winner",
    }
    assert settings["multi_objective_min_accuracy"] is None
    assert parsed["parsed_selection"]["latency_accuracy_retention"]["value"] == 0.98
    assert parsed["parsed_selection"]["multi_objective_min_accuracy"] is None


def test_manifest_rejects_erratum_or_post_front_contract_drift():
    manifest = fixture_manifest()
    manifest["derivation"]["sensitivity_result_sha256"] = "0" * 64
    with pytest.raises(
        runner.ContractError,
        match="approved sensitivity result whole-file SHA256",
    ):
        runner.validate_manifest_contract(manifest)

    manifest = fixture_manifest()
    manifest["derivation"]["source_identity"]["analysis_erratum"][
        "corrected_aggregator_sha256"
    ] = "0" * 64
    with pytest.raises(
        runner.ContractError,
        match="approved analysis erratum source identity",
    ):
        runner.validate_manifest_contract(manifest)

    manifest = fixture_manifest()
    manifest["post_front_matched_validation"]["allocation_design"][
        "allocation_count"
    ] = 5
    with pytest.raises(
        runner.ContractError,
        match="post-front matched-validation contract",
    ):
        runner.validate_manifest_contract(manifest)


def test_finite_integer_levels_use_canonical_numeric_schema_enum():
    from tao_automl.search_space.params import generate_hyperparams_to_search

    manifest = fixture_manifest()
    schema = json.loads(TRAIN_SCHEMA_PATH.read_text())
    derived, parameters, ranges = runner.build_search_contract(manifest, schema)
    base_specs = runner.training_spec(
        manifest,
        __import__("yaml").safe_load(TRAIN_TEMPLATE_PATH.read_text()),
    )
    records, names = generate_hyperparams_to_search(
        network="dino",
        action="train",
        train_specs=base_specs,
        automl_hyperparameters=parameters,
        schema=derived,
    )
    by_name = {record["parameter"]: record for record in records}

    assert set(names) == set(parameters)
    assert by_name["model.num_select"]["value_type"] == "ordered_int"
    assert by_name["model.num_select"]["valid_options"] == [50, 100, 200, 300]
    assert by_name["model.enc_layers"]["value_type"] == "ordered_int"
    assert by_name["model.dec_layers"]["value_type"] == "ordered_int"
    assert ranges["model.num_select"] == {
        "valid_options": [50, 100, 200, 300]
    }
    assert schema["properties"]["model"]["properties"]["num_select"].get(
        "enum"
    ) is None


@pytest.mark.parametrize(
    "bad_options",
    [
        [],
        [50, 50, 100],
        [50, "100", 200],
        [50, True, 200],
    ],
)
def test_invalid_finite_integer_options_fail_closed(bad_options):
    manifest = fixture_manifest()
    manifest["search_space"]["search_domains"]["model.num_select"][
        "valid_options"
    ] = bad_options
    schema = json.loads(TRAIN_SCHEMA_PATH.read_text())

    with pytest.raises(runner.ContractError):
        runner.build_search_contract(manifest, schema)


def test_full_model_mapping_propagates_to_train_and_evaluate_specs():
    import yaml

    manifest = fixture_manifest()
    specs = valid_specs(manifest)
    specs.update(
        {
            "model.num_queries": 450,
            "model.enc_layers": 3,
            "model.dec_layers": 4,
            "model.num_select": 200,
        }
    )
    resolved_model = runner.apply_candidate_to_reference_model(manifest, specs)
    train = runner.training_spec(
        manifest,
        yaml.safe_load(TRAIN_TEMPLATE_PATH.read_text()),
        specs,
    )
    evaluate = runner.evaluation_spec(
        manifest,
        yaml.safe_load(EVALUATE_TEMPLATE_PATH.read_text()),
        resolved_model,
        "/lustre/results/job/results_dir/train/model_epoch_009_step_1.pth",
        latency=False,
    )
    latency = runner.evaluation_spec(
        manifest,
        yaml.safe_load(EVALUATE_TEMPLATE_PATH.read_text()),
        resolved_model,
        "/lustre/results/job/results_dir/train/model_epoch_009_step_1.pth",
        latency=True,
    )

    assert train["model"] == resolved_model
    assert evaluate["model"] == resolved_model
    assert latency["model"] == resolved_model
    assert resolved_model["num_queries"] == 450
    assert resolved_model["enc_layers"] == 3
    assert resolved_model["dec_layers"] == 4
    assert resolved_model["num_select"] == 200
    # Non-searched architecture values are carried forward too.
    assert resolved_model["hidden_dim"] == 256
    assert resolved_model["return_interm_indices"] == [1, 2, 3, 4]
    assert train["train"]["optim"]["lr"] == pytest.approx(specs["train.optim.lr"])
    assert train["train"]["optim"]["weight_decay"] == pytest.approx(
        specs["train.optim.weight_decay"]
    )
    assert evaluate["dataset"]["batch_size"] == 4
    assert latency["dataset"]["batch_size"] == 1


def test_candidate_off_discrete_grid_is_rejected_without_quantization():
    manifest = fixture_manifest()
    specs = valid_specs(manifest)
    specs["model.num_select"] = 175

    with pytest.raises(
        runner.ContractError,
        match="not a preregistered finite option",
    ):
        runner.apply_candidate_to_reference_model(manifest, specs)


def test_candidate_constraint_is_checked_after_full_resolution():
    manifest = fixture_manifest()
    specs = valid_specs(manifest)
    specs["model.num_queries"] = 300
    specs["model.num_select"] = 300
    assert runner.apply_candidate_to_reference_model(manifest, specs)[
        "num_select"
    ] == 300

    manifest["search_space"]["search_domains"]["model.num_select"][
        "valid_options"
    ].append(400)
    specs["model.num_select"] = 400
    with pytest.raises(runner.ContractError, match="num_queries"):
        runner.apply_candidate_to_reference_model(manifest, specs)


def test_union_selector_is_order_independent_and_returns_nondominated_middle():
    manifest = fixture_manifest()
    base = valid_specs(manifest)
    records = [
        {
            "candidate_id": "accuracy",
            "specs": {**base, "train.optim.lr": 0.0001},
            "objective_values": {
                "mAP50": 0.92,
                "latency_ms": 20.0,
                "latency_ci95_low": 19.9,
                "latency_ci95_high": 20.1,
            },
        },
        {
            "candidate_id": "middle",
            "specs": {**base, "train.optim.lr": 0.0002},
            "objective_values": {
                "mAP50": 0.89,
                "latency_ms": 14.0,
                "latency_ci95_low": 13.9,
                "latency_ci95_high": 14.1,
            },
        },
        {
            "candidate_id": "latency",
            "specs": {**base, "train.optim.lr": 0.0003},
            "objective_values": {
                "mAP50": 0.86,
                "latency_ms": 10.0,
                "latency_ci95_low": 9.9,
                "latency_ci95_high": 10.1,
            },
        },
        {
            "candidate_id": "dominated",
            "specs": {**base, "train.optim.lr": 0.0004},
            "objective_values": {
                "mAP50": 0.80,
                "latency_ms": 18.0,
                "latency_ci95_low": 17.9,
                "latency_ci95_high": 18.1,
            },
        },
    ]
    analysis, audit = runner.analyze_union_archive(manifest, records)
    by_id = {item["candidate_id"]: item for item in analysis["candidates"]}

    assert audit["order_independent"] is True
    assert analysis["selections"]["accuracy"]["winner_id"] == "accuracy"
    # 98% of 0.92 excludes the two lower-accuracy points.
    assert analysis["selections"]["latency"]["winner_id"] == "accuracy"
    assert analysis["selections"]["multi_objective"]["winner_id"] == "middle"
    assert by_id["middle"]["multi_objective_pareto_rank"] == 0
    assert by_id["dominated"]["pareto_rank"] > 0


def test_manifest_requires_both_internal_and_whole_file_hashes(tmp_path):
    manifest = fixture_manifest()
    path = tmp_path / "expanded_search_manifest.v1.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    whole_file_sha = runner.sha256_file(path)

    loaded, observed = runner.load_manifest(
        path,
        supplied_file_sha256=whole_file_sha,
    )
    assert loaded["manifest_sha256"] == manifest["manifest_sha256"]
    assert observed == whole_file_sha

    path.write_text(path.read_text() + "\n")
    with pytest.raises(runner.ContractError, match="whole-file SHA256"):
        runner.load_manifest(path, supplied_file_sha256=whole_file_sha)


def test_runner_source_hash_is_manifest_pinned_and_tamper_evident():
    manifest = fixture_manifest()
    provenance = runner.validate_runner_source_provenance(manifest)
    assert provenance["observed_sha256"] == runner.sha256_file(MODULE_PATH)
    assert isinstance(provenance["launch_source_ready"], bool)
    assert isinstance(provenance["blockers"], list)

    tampered = copy.deepcopy(manifest)
    tampered["derivation"]["runner_sha256"] = "0" * 64
    with pytest.raises(runner.ContractError, match="runner source SHA256"):
        runner.validate_runner_source_provenance(tampered)


def test_launch_source_readiness_fails_closed_on_dirty_or_untracked_runner():
    with pytest.raises(runner.ContractError, match="tracked, committed, and clean"):
        runner.require_launch_source_ready(
            {
                "launch_source_ready": False,
                "blockers": ["runner_source_untracked"],
            }
        )
    runner.require_launch_source_ready(
        {"launch_source_ready": True, "blockers": []}
    )


def test_search_key_set_is_exact_and_cannot_receive_manual_extra_parameter():
    manifest = fixture_manifest()
    specs = valid_specs(manifest)
    specs["model.hidden_dim"] = 128

    with pytest.raises(runner.ContractError, match="candidate search keys drifted"):
        runner.apply_candidate_to_reference_model(manifest, specs)


def terminal_seed_records(manifest: dict, seed: int) -> dict:
    records = {}
    for rec_id in range(20):
        key = runner.candidate_id(seed, rec_id)
        specs = valid_specs(manifest)
        specs["train.optim.lr"] = 0.0001 + rec_id * 1.0e-6
        model = runner.apply_candidate_to_reference_model(manifest, specs)
        records[key] = {
            "candidate_id": key,
            "search_seed": seed,
            "training_seed": 1234,
            "rec_id": rec_id,
            "status": "success",
            "manual_candidate_injection_used": False,
            "specs": specs,
            "resolved_model_spec": model,
            "resolved_model_spec_sha256": runner.sha256_value(model),
            "objective_values": {
                "mAP50": 0.5 + rec_id * 0.001,
                "latency_ms": 20.0 - rec_id * 0.01,
                "latency_p95_ms": 20.2 - rec_id * 0.01,
                "latency_ci95_low": 19.9 - rec_id * 0.01,
                "latency_ci95_high": 20.1 - rec_id * 0.01,
            },
        }
    return records


def test_completed_seed_archive_is_immutable_and_requires_20_terminal_records(
    tmp_path,
):
    manifest = fixture_manifest()
    manifest_file_sha = "a" * 64
    records = terminal_seed_records(manifest, 314159)
    archive_path = runner.finalize_seed_archive(
        tmp_path,
        manifest_file_sha,
        314159,
        20,
        records,
        {"status": "success"},
    )
    original = archive_path.read_bytes()
    archive = json.loads(original)
    runner.validate_seed_archive(
        archive,
        manifest_file_sha256=manifest_file_sha,
        seed=314159,
        recommendations=20,
    )

    changed = copy.deepcopy(records)
    changed[runner.candidate_id(314159, 0)]["objective_values"][
        "mAP50"
    ] = 0.99
    with pytest.raises(runner.ContractError, match="already exists and differs"):
        runner.finalize_seed_archive(
            tmp_path,
            manifest_file_sha,
            314159,
            20,
            changed,
            {"status": "success"},
        )
    assert archive_path.read_bytes() == original

    incomplete = copy.deepcopy(archive)
    incomplete["records"].pop(runner.candidate_id(314159, 19))
    incomplete.pop("archive_sha256")
    incomplete["archive_sha256"] = runner.sha256_value(incomplete)
    with pytest.raises(runner.ContractError, match="candidate IDs"):
        runner.validate_seed_archive(
            incomplete,
            manifest_file_sha256=manifest_file_sha,
            seed=314159,
            recommendations=20,
        )


class _FakeSlurmHandler:
    def __init__(self):
        self.identities = {}

    def get_job_runtime_identity(self, job_id):
        return copy.deepcopy(self.identities[job_id])


class _FakeSlurmStore:
    def __init__(self, handler):
        self.handler = handler

    def get_job(self, job_id):
        return {
            "specs": {
                "_slurm_runtime": self.handler.get_job_runtime_identity(job_id)
            }
        }


class _FakeSlurmSDK:
    def __init__(self):
        self._handler = _FakeSlurmHandler()
        self._store = _FakeSlurmStore(self._handler)
        self.created = []

    def create_job(self, **kwargs):
        job = SimpleNamespace(id=f"job-{len(self.created)}")
        ordinal = len(self.created) + 100
        self._handler.identities[job.id] = {
            "slurm_job_id": str(ordinal),
            "retry_count": 0,
            "failed_slurm_job_ids": [],
            "launch_uncertain": False,
            "revision": 1,
        }
        self.created.append({"job": job, "kwargs": kwargs})
        return job

    @staticmethod
    def _runtime_from_entry(entry):
        return copy.deepcopy(entry["specs"]["_slurm_runtime"])


def _evaluate_action() -> dict:
    import yaml

    skill_info = yaml.safe_load(
        (
            TRAIN_TEMPLATE_PATH.parent / "skill_info.yaml"
        ).read_text()
    )
    return skill_info["actions"]["evaluate"]


@pytest.mark.parametrize("phase", ["accuracy", "latency"])
def test_submitted_measurement_child_is_reused_on_resume(
    tmp_path,
    monkeypatch,
    phase,
):
    import yaml

    manifest = fixture_manifest()
    sdk = _FakeSlurmSDK()
    submitted = []
    checkpoint = {
        "path": (
            "/lustre/fsw/tao/results/train-job/results_dir/train/"
            "model_epoch_009_step_1.pth"
        ),
        "sha256": "c" * 64,
    }
    resolved_model = runner.apply_candidate_to_reference_model(
        manifest,
        valid_specs(manifest),
    )
    template = yaml.safe_load(EVALUATE_TEMPLATE_PATH.read_text())
    action = _evaluate_action()

    def stop_after_submission(*_args, **_kwargs):
        raise RuntimeError("intentional stop after child submission")

    monkeypatch.setattr(runner, "wait_for_job", stop_after_submission)

    def invoke(existing_job=None):
        common = {
            "event_path": tmp_path / "events.jsonl",
            "seed": 314159,
            "rec_id": 0,
            "existing_job": existing_job,
            "on_submitted": submitted.append,
        }
        if phase == "accuracy":
            return runner.launch_accuracy_evaluation(
                sdk,
                manifest,
                action,
                template,
                resolved_model,
                checkpoint,
                **common,
            )
        sensitivity = {
            "latency_protocol": copy.deepcopy(
                manifest["frozen_identity"]["latency_protocol"]
            )
        }
        return runner.launch_latency_benchmark(
            sdk,
            manifest,
            sensitivity,
            action,
            template,
            HERE / "sensitivity_latency_block_runner.py",
            resolved_model,
            checkpoint,
            tmp_path,
            **common,
        )

    with pytest.raises(RuntimeError, match="intentional stop"):
        invoke()
    assert len(sdk.created) == 1
    assert len(submitted) == 1
    assert submitted[0]["tao_job_id"] == "job-0"
    assert submitted[0]["status"] == "submitted"

    with pytest.raises(RuntimeError, match="intentional stop"):
        invoke(submitted[0])
    assert len(sdk.created) == 1
    assert len(submitted) == 2
    assert submitted[-1]["slurm_job_id"] == "100"
    assert submitted[-1]["runtime_revision"] == 1


def _simulate_infrastructure_retry(sdk, job_id):
    previous = sdk._handler.identities[job_id]
    sdk._handler.identities[job_id] = {
        **previous,
        "slurm_job_id": "200",
        "retry_count": 1,
        "failed_slurm_job_ids": [previous["slurm_job_id"]],
        "launch_uncertain": False,
        "revision": previous["revision"] + 1,
    }
    return "Complete"


def test_accuracy_child_refreshes_terminal_identity_after_retry(
    tmp_path,
    monkeypatch,
):
    import yaml

    manifest = fixture_manifest()
    sdk = _FakeSlurmSDK()
    sdk.get_job_logs = lambda *_args, **_kwargs: "complete"
    sdk.get_job_results_dir = (
        lambda job_id: f"lustre:///lustre/results/{job_id}"
    )
    submitted = []
    monkeypatch.setattr(
        runner,
        "wait_for_job",
        lambda sdk_arg, job_id, **_kwargs: _simulate_infrastructure_retry(
            sdk_arg,
            job_id,
        ),
    )
    monkeypatch.setattr(runner, "read_status_map50", lambda *_args: 0.61)
    model = runner.apply_candidate_to_reference_model(
        manifest,
        valid_specs(manifest),
    )
    metric, evidence = runner.launch_accuracy_evaluation(
        sdk,
        manifest,
        _evaluate_action(),
        yaml.safe_load(EVALUATE_TEMPLATE_PATH.read_text()),
        model,
        {
            "path": "/lustre/results/train/model_epoch_009_step_1.pth",
            "sha256": "d" * 64,
        },
        event_path=tmp_path / "events.jsonl",
        seed=314159,
        rec_id=0,
        on_submitted=submitted.append,
    )

    assert metric == pytest.approx(0.61)
    assert evidence["slurm_job_id"] == "200"
    assert evidence["retry_count"] == 1
    assert evidence["failed_slurm_job_ids"] == ["100"]
    assert evidence["launch_uncertain"] is False
    assert evidence["runtime_revision"] == 2
    assert submitted[-1]["slurm_job_id"] == "200"


def test_latency_child_refreshes_terminal_identity_after_retry(
    tmp_path,
    monkeypatch,
):
    import yaml

    manifest = fixture_manifest()
    sdk = _FakeSlurmSDK()
    sdk.get_job_logs = (
        lambda *_args, **_kwargs: "TAO_AUTOML_LATENCY_COMPLETE"
    )
    sdk.get_job_results_dir = (
        lambda job_id: f"lustre:///lustre/results/{job_id}"
    )
    submitted = []
    monkeypatch.setattr(
        runner,
        "wait_for_job",
        lambda sdk_arg, job_id, **_kwargs: _simulate_infrastructure_retry(
            sdk_arg,
            job_id,
        ),
    )
    rounds = manifest["frozen_identity"]["latency_protocol"][
        "repeated_rounds"
    ]
    rank_records = [
        {
            "rank": rank,
            "samples_ms": [[4.0, 4.1] for _ in range(rounds)],
            "hardware": {"hostname": f"node-{rank}"},
            "runtime": {"torch": "test"},
        }
        for rank in range(8)
    ]
    monkeypatch.setattr(
        runner,
        "read_latency_rank_records",
        lambda *_args: rank_records,
    )
    monkeypatch.setattr(
        runner,
        "validate_latency_rank_contract",
        lambda *_args, **_kwargs: (
            {"gpu_name": "A100", "world_size": 8},
            {"identity_sha256": "e" * 64},
        ),
    )
    monkeypatch.setattr(runner, "enforce_shared_contract", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "latency_protocol_from_manifest",
        lambda *_args: object(),
    )
    statistics = SimpleNamespace(
        is_valid=True,
        invalid_reasons=(),
        median_ms=4.05,
        tail_latency_ms=4.1,
        mad_ms=0.05,
        iqr_ms=0.1,
        robust_cv=0.01,
        bootstrap_median_ci_ms=(4.0, 4.1),
        bootstrap_ci_width_ms=0.1,
        round_drift_fraction=0.0,
        device_median_range_fraction=0.0,
        synchronized_median_ms=4.05,
        synchronized_tail_latency_ms=4.1,
    )
    monkeypatch.setattr(
        runner,
        "aggregate_synchronized_latency",
        lambda *_args: statistics,
    )
    monkeypatch.setattr(runner, "asdict", lambda value: vars(value))
    model = runner.apply_candidate_to_reference_model(
        manifest,
        valid_specs(manifest),
    )
    metrics, evidence = runner.launch_latency_benchmark(
        sdk,
        manifest,
        {
            "latency_protocol": copy.deepcopy(
                manifest["frozen_identity"]["latency_protocol"]
            )
        },
        _evaluate_action(),
        yaml.safe_load(EVALUATE_TEMPLATE_PATH.read_text()),
        HERE / "sensitivity_latency_block_runner.py",
        model,
        {
            "path": "/lustre/results/train/model_epoch_009_step_1.pth",
            "sha256": "f" * 64,
        },
        tmp_path,
        event_path=tmp_path / "events.jsonl",
        seed=314159,
        rec_id=0,
        on_submitted=submitted.append,
    )

    assert metrics["latency_ms"] == pytest.approx(4.05)
    assert evidence["slurm_job_id"] == "200"
    assert evidence["retry_count"] == 1
    assert evidence["failed_slurm_job_ids"] == ["100"]
    assert evidence["launch_uncertain"] is False
    assert evidence["runtime_revision"] == 2
    assert submitted[-1]["slurm_job_id"] == "200"


def test_launch_uncertain_runtime_is_rejected():
    with pytest.raises(runner.ContractError, match="launch remains uncertain"):
        runner.validate_recorded_slurm_runtime(
            {
                "tao_job_id": "job",
                "slurm_job_id": "100",
                "retry_count": 0,
                "failed_slurm_job_ids": [],
                "launch_uncertain": True,
                "runtime_revision": 1,
            },
            label="test child",
        )


def test_frozen_slurm_child_times_override_invoking_environment(monkeypatch):
    manifest = fixture_manifest()
    monkeypatch.setenv("SLURM_TIME_HOURS", "99")
    monkeypatch.setenv("SLURM_TIMEOUT_HOURS", "98")

    runner.configure_slurm(manifest)

    assert __import__("os").environ["SLURM_TIME_HOURS"] == "4"
    assert __import__("os").environ["SLURM_TIMEOUT_HOURS"] == "3.8"
    assert __import__("os").environ["SLURM_USE_REQUEUE"] == "false"


def test_global_resume_starts_untouched_seed_and_reuses_partial_seed(tmp_path):
    untouched = tmp_path / "seed_314159"
    workspace, effective_resume = runner.resolve_seed_workspace(
        untouched,
        {},
        resume_requested=True,
    )
    assert workspace == untouched / "workspace"
    assert effective_resume is False

    partial = tmp_path / "seed_271828"
    run_workspace = partial / "workspace" / "run_20260728_000000"
    run_workspace.mkdir(parents=True)
    workspace, effective_resume = runner.resolve_seed_workspace(
        partial,
        {"seed_271828_rec_0": {"status": "recommended"}},
        resume_requested=True,
    )
    assert workspace == run_workspace
    assert effective_resume is True

    with pytest.raises(runner.ContractError, match="use --resume"):
        runner.resolve_seed_workspace(
            partial,
            {"seed_271828_rec_0": {"status": "recommended"}},
            resume_requested=False,
        )

    missing_workspace = tmp_path / "seed_161803"
    with pytest.raises(runner.ContractError, match="workspace is missing"):
        runner.resolve_seed_workspace(
            missing_workspace,
            {"seed_161803_rec_0": {"status": "recommended"}},
            resume_requested=True,
        )
