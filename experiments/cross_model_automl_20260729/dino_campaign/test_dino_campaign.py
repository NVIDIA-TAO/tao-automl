"""Contract tests for the direct three-mode DINO/VOC2007 campaign."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tao_automl.brain.bayesian import Bayesian
from tao_automl.objectives import parse_objective_config

from . import manifest_generator as generator
from . import run_campaign


HERE = Path(__file__).resolve().parent
INPUTS = HERE / "campaign.inputs.v1.json"
REPOSITORY = HERE.parents[2]


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _clean_git_repository(path: Path, files: dict[str, bytes]) -> str:
    for name, content in files.items():
        destination = path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Campaign Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-qm", "fixture"],
        check=True,
    )
    return _git(path, "rev-parse", "HEAD")


@pytest.fixture()
def sealed_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw = json.loads(INPUTS.read_text(encoding="utf-8"))

    source_repository = tmp_path / "source"
    source_commit = _clean_git_repository(
        source_repository,
        {
            "src/tao_automl/latency_benchmark.py": (
                REPOSITORY / "src/tao_automl/latency_benchmark.py"
            ).read_bytes(),
            "src/tao_automl/latency_stats.py": (
                REPOSITORY / "src/tao_automl/latency_stats.py"
            ).read_bytes(),
        },
    )
    wheel = tmp_path / "tao_automl.whl"
    wheel.write_bytes(b"sealed wheel fixture")
    raw["source"].update(
        {
            "repository": str(source_repository),
            "commit": "HEAD",
            "wheel_path": str(wheel),
            "wheel_sha256": generator.sha256_file(wheel),
            "wheel_source_commit": source_commit,
        }
    )

    skill_repository = tmp_path / "skills"
    skill_dir = skill_repository / "skills/models/tao-train-dino"
    skill_revision = _clean_git_repository(
        skill_repository,
        {
            "skills/models/tao-train-dino/references/skill_info.yaml": (
                b"actions:\n"
                b"  evaluate:\n"
                b"    command: dino evaluate -e {config_path}\n"
                b"    inputs: {}\n"
                b"    outputs: {}\n"
                b"    config_format: yaml\n"
                b"    upload_excludes: []\n"
            )
        },
    )
    sdk_repository = tmp_path / "sdk"
    sdk_revision = _clean_git_repository(
        sdk_repository,
        {"tao_sdk/__init__.py": b""},
    )
    raw["runtime"].update(
        {
            "skill_dir": str(skill_dir),
            "skill_revision": skill_revision,
            "sdk_dir": str(sdk_repository),
            "sdk_revision": sdk_revision,
        }
    )

    dataset_manifest = tmp_path / "dataset_manifest.json"
    dataset_manifest.write_text("{}\n", encoding="utf-8")
    integrity = tmp_path / "integrity.json"
    integrity.write_text("{}\n", encoding="utf-8")
    image_dir = tmp_path / "JPEGImages"
    image_dir.mkdir()
    raw["dataset"].update(
        {
            "manifest_path": str(dataset_manifest),
            "manifest_sha256": generator.sha256_file(dataset_manifest),
            "integrity_path": str(integrity),
            "integrity_sha256": generator.sha256_file(integrity),
            "local_image_dir": str(image_dir),
        }
    )
    expected_tree = {
        "algorithm": raw["dataset"]["image_tree_algorithm"],
        "sha256": raw["dataset"]["image_tree_sha256"],
        "file_count": raw["dataset"]["image_count"],
        "total_bytes": raw["dataset"]["image_total_bytes"],
    }
    monkeypatch.setattr(
        generator,
        "sha256sum_basename_tree",
        lambda _path: copy.deepcopy(expected_tree),
    )
    return generator.seal_manifest(generator.build_manifest(raw))


def test_manifest_freezes_direct_three_mode_contract(sealed_manifest):
    manifest = sealed_manifest
    assert manifest["execution"] == {
        "kind": "direct_full_search",
        "cpu_runs": 0,
        "smoke_runs": 0,
        "smoke_or_cpu_preflight_skipped_by_user": True,
        "shared_archive": False,
        "independent_mode_jobs": True,
        "submission_ready": True,
    }
    assert manifest["search"]["candidate_budget_per_mode"] == 20
    assert manifest["search"]["training_epochs"] == 10
    assert manifest["search"]["search_seed"] == 271828
    assert manifest["search"]["training_seed"] == 1234
    assert manifest["search"]["parameters"] == [
        "model.enc_layers",
        "model.dec_layers",
        "train.optim.lr",
        "train.optim.weight_decay",
    ]
    assert "model.num_queries" not in manifest["search"]["space"]
    assert run_campaign.custom_ranges(manifest) == {
        "model.enc_layers": {"valid_min": 3, "valid_max": 6},
        "model.dec_layers": {"valid_min": 3, "valid_max": 6},
        "train.optim.lr": {"valid_min": 1e-5, "valid_max": 5e-4},
        "train.optim.weight_decay": {
            "valid_min": 1e-5,
            "valid_max": 1e-3,
        },
    }
    assert [item["mode"] for item in manifest["modes"]] == [
        "accuracy",
        "latency",
        "multi_objective",
    ]
    assert len(
        {item["observation_namespace"] for item in manifest["modes"]}
    ) == 3
    assert all(not item["observation_sharing"] for item in manifest["modes"])
    assert all(item["initial_observation_ids"] == [] for item in manifest["modes"])


def test_modes_route_independent_objective_aware_policies(sealed_manifest):
    accuracy = run_campaign.mode_settings(sealed_manifest, "accuracy")
    latency = run_campaign.mode_settings(sealed_manifest, "latency")
    multi = run_campaign.mode_settings(sealed_manifest, "multi_objective")

    assert accuracy["selection_mode"] == "accuracy"
    assert "latency_accuracy_retention" not in accuracy
    assert latency["selection_mode"] == "latency"
    assert latency["latency_accuracy_retention"] == {
        "type": "relative",
        "retained_fraction": 0.90,
        "reference": "accuracy_winner",
    }
    assert multi["selection_mode"] == "multi_objective"
    assert "latency_accuracy_retention" not in multi
    assert multi["multi_objective_min_accuracy"] is None
    assert {
        item["experiment_id"] for item in (accuracy, latency, multi)
    } == {
        item["observation_namespace"] for item in sealed_manifest["modes"]
    }
    assert {
        item["random_seed"] for item in (accuracy, latency, multi)
    } == {271828}
    assert {
        item["objective_acquisition"]["calibration_points"]
        for item in (accuracy, latency, multi)
    } == {8}


def test_latency_protocol_uses_real_oriented_batches_and_a100_contract(
    sealed_manifest,
):
    protocol = sealed_manifest["latency_protocol"]
    descriptor = protocol["input_descriptor"]
    landscape = [1, 4, 800, 1333]
    portrait = [1, 4, 1333, 800]
    assert descriptor["validation_image_ids"] == [
        5,
        7,
        9,
        16,
        19,
        20,
        21,
        24,
        30,
        39,
        41,
        46,
        50,
        51,
        52,
        60,
    ]
    assert descriptor["shape_sequence"] == [
        landscape,
        landscape,
        landscape,
        portrait,
        landscape,
        portrait,
        portrait,
        landscape,
        landscape,
        landscape,
        landscape,
        portrait,
        landscape,
        landscape,
        landscape,
        landscape,
    ]
    assert protocol["timed_scope"] == (
        "model_forward_plus_dino_gpu_postprocess"
    )
    assert protocol["warmup_iterations"] == 50
    assert protocol["timed_iterations"] == 100
    assert protocol["repeated_rounds"] == 5
    assert protocol["raw_samples_per_candidate"] == 4000
    assert descriptor["required_hardware"] == {
        "gpu_name": "NVIDIA A100-SXM4-80GB",
        "compute_capability": "8.0",
        "total_memory_bytes": 85174583296,
    }


def test_manifest_tampering_and_extra_ptm_fail_closed(sealed_manifest):
    tampered = copy.deepcopy(sealed_manifest)
    tampered["search"]["candidate_budget_per_mode"] = 21
    with pytest.raises(generator.ManifestError):
        generator.validate_manifest(tampered)

    tampered = copy.deepcopy(sealed_manifest)
    tampered["ptms"].append(copy.deepcopy(tampered["ptms"][0]))
    tampered["ptms"][-1]["id"] = "unverified"
    tampered.pop("manifest_sha256")
    with pytest.raises(generator.ManifestError, match="status=supported"):
        generator.validate_manifest(tampered, require_seal=False)


def test_remote_contract_reads_every_launcher_field(
    sealed_manifest,
    monkeypatch: pytest.MonkeyPatch,
):
    expected_files = {}
    runtime = sealed_manifest["runtime"]
    dataset = sealed_manifest["dataset"]
    expected_files[runtime["sqsh_path"]] = (
        runtime["sqsh_size_bytes"],
        runtime["sqsh_sha256"],
    )
    expected_files[dataset["train_annotation"]] = (
        dataset["train_annotation_size_bytes"],
        dataset["train_annotation_sha256"],
    )
    expected_files[dataset["validation_annotation"]] = (
        dataset["validation_annotation_size_bytes"],
        dataset["validation_annotation_sha256"],
    )
    for ptm in sealed_manifest["ptms"]:
        expected_files[ptm["slurm_path"]] = (
            ptm["runtime_artifact"]["size_bytes"],
            ptm["runtime_artifact"]["sha256"],
        )

    def remote(command: str, *, timeout: int = 900) -> str:
        del timeout
        if "python3 -c" in command:
            return json.dumps(dataset["image_tree"])
        for path, (size, sha256) in expected_files.items():
            if path in command:
                return f"{size}\n{sha256}  {path}\n"
        raise AssertionError(f"unexpected remote command: {command}")

    monkeypatch.setattr(run_campaign, "remote_output", remote)
    result = run_campaign.verify_remote_contract(sealed_manifest)
    assert set(result) == {
        "sqsh",
        "train_annotation",
        "validation_annotation",
        "ptm:dino.coco.resnet50.trainable.v1.0",
        "voc_images",
    }


def test_clean_sdk_environment_is_explicit(
    sealed_manifest,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("PYTHONPATH", "/installed/sdk")
    run_campaign.configure_slurm_runtime(sealed_manifest)
    runtime = sealed_manifest["runtime"]
    assert os.environ["PYTHONPATH"].split(os.pathsep)[0] == runtime["sdk_dir"]
    assert os.environ["SLURM_USE_SQSH"] == "false"
    assert os.environ["SLURM_USE_REQUEUE"] == "true"
    assert os.environ["SLURM_BASE_RESULTS_DIR"] == runtime["base_results_dir"]
    assert os.environ["SLURM_CONTAINER_MOUNTS"] == "/lustre"
    assert os.environ["SLURM_TIME_HOURS"] == "4.0"
    assert os.environ["SLURM_TIMEOUT_HOURS"] == "3.8"


def test_missing_status_file_allows_log_fallback(monkeypatch):
    sdk = SimpleNamespace(
        get_job_results_dir=lambda _job_id: "lustre:///results/job-1"
    )
    monkeypatch.setattr(run_campaign, "remote_output", lambda _command: "")
    assert run_campaign._map50_from_status(sdk, "job-1") is None


class _BrainStore:
    def __init__(self, ranges):
        self.ranges = ranges

    def get_job_specs(self, _job_id):
        return {"train": {"num_epochs": 10}}

    def get_custom_param_ranges(self, _handler_id):
        return copy.deepcopy(self.ranges)


def test_controller_only_recommendations_stay_inside_frozen_4d_domain(
    sealed_manifest,
):
    settings = run_campaign.mode_settings(sealed_manifest, "multi_objective")
    objective_config = parse_objective_config(settings)
    parameters = [
        {
            "parameter": "model.enc_layers",
            "value_type": "int",
            "default_value": 6,
            "valid_min": 1,
            "valid_max": 100,
            "valid_options": [],
            "option_weights": None,
            "math_cond": None,
            "parent_param": None,
            "depends_on": None,
        },
        {
            "parameter": "model.dec_layers",
            "value_type": "int",
            "default_value": 6,
            "valid_min": 1,
            "valid_max": 100,
            "valid_options": [],
            "option_weights": None,
            "math_cond": None,
            "parent_param": None,
            "depends_on": None,
        },
        *[
            {
                "parameter": name,
                "value_type": "float",
                "default_value": default,
                "valid_min": 0.0,
                "valid_max": 1.0,
                "valid_options": [],
                "option_weights": None,
                "math_cond": None,
                "parent_param": None,
                "depends_on": None,
            }
            for name, default in (
                ("train.optim.lr", 2e-4),
                ("train.optim.weight_decay", 1e-4),
            )
        ],
    ]
    context = SimpleNamespace(
        id="dino-domain-contract",
        handler_id="dino-domain-contract",
        random_seed=271828,
        action="train",
    )
    brain = Bayesian(
        context,
        _BrainStore(run_campaign.custom_ranges(sealed_manifest)),
        "dino",
        parameters,
        metric="multi_objective_score",
        direction="maximize",
        objective_config=objective_config,
        acquisition_settings={"calibration_points": 8},
    )
    history = []
    for index in range(20):
        recommendation = brain.generate_recommendations(history)[0]
        assert recommendation["model.enc_layers"] in {3, 4, 5, 6}
        assert recommendation["model.dec_layers"] in {3, 4, 5, 6}
        assert 1e-5 <= recommendation["train.optim.lr"] <= 5e-4
        assert (
            1e-5
            <= recommendation["train.optim.weight_decay"]
            <= 1e-3
        )
        history.append(
            SimpleNamespace(
                id=index,
                status="success",
                result=0.0,
                objective_values={
                    "mAP50": 0.20 + index / 100.0,
                    "latency_ms": 70.0 - index / 10.0,
                },
            )
        )


def test_evaluation_child_job_is_persisted_and_reused(
    sealed_manifest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    script_runner = types.ModuleType("tao_sdk.script_runner")
    script_runner.build_entrypoint = lambda **_kwargs: {"command": "evaluate"}
    package = types.ModuleType("tao_sdk")
    package.script_runner = script_runner
    monkeypatch.setitem(sys.modules, "tao_sdk", package)
    monkeypatch.setitem(sys.modules, "tao_sdk.script_runner", script_runner)
    monkeypatch.setattr(run_campaign, "_map50_from_status", lambda *_args: None)

    class SDK:
        def __init__(self):
            self.created = 0

        def create_job(self, **_kwargs):
            self.created += 1
            return SimpleNamespace(id="evaluation-child")

        def get_job_status(self, _job_id):
            return SimpleNamespace(status="Complete")

        def get_job_logs(self, _job_id, tail=0):
            del tail
            return "Validation mAP50: 0.42"

        def get_job_results_dir(self, _job_id):
            return "lustre:///results/evaluation-child"

    sdk = SDK()
    ledger = []
    result = run_campaign._launch_evaluation(
        sdk,
        sealed_manifest,
        {"model": {}},
        events=tmp_path / "events.jsonl",
        mode="accuracy",
        candidate_id="accuracy_rec_0",
        on_submitted=ledger.append,
    )
    assert result[0] == 0.42
    assert ledger[0]["status"] == "submitted"
    assert ledger[0]["tao_job_id"] == "evaluation-child"
    assert ledger[-1]["status"] == "Complete"
    assert sdk.created == 1

    run_campaign._launch_evaluation(
        sdk,
        sealed_manifest,
        {"model": {}},
        events=tmp_path / "events.jsonl",
        mode="accuracy",
        candidate_id="accuracy_rec_0",
        existing_job=ledger[-1],
    )
    assert sdk.created == 1


def test_latency_install_payload_survives_sdk_config_path_formatting(
    sealed_manifest,
):
    payload = run_campaign._install_payload(sealed_manifest)
    command = f"{payload} && worker --config {{config_path}}"

    resolved = command.format(config_path="/tmp/spec.yaml")

    assert "tao_automl/__init__.py" in resolved
    assert "dino_latency_worker.py" in resolved
    assert "worker --config /tmp/spec.yaml" in resolved


def test_latency_child_job_is_persisted_and_reused(
    sealed_manifest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    script_runner = types.ModuleType("tao_sdk.script_runner")
    entrypoint_inputs = []

    def _build_entrypoint(**kwargs):
        entrypoint_inputs.append(copy.deepcopy(kwargs))
        return {"command": "latency"}

    script_runner.build_entrypoint = _build_entrypoint
    package = types.ModuleType("tao_sdk")
    package.script_runner = script_runner
    monkeypatch.setitem(sys.modules, "tao_sdk", package)
    monkeypatch.setitem(sys.modules, "tao_sdk.script_runner", script_runner)
    monkeypatch.setattr(run_campaign, "_install_payload", lambda _manifest: "x")

    import tao_automl.latency_benchmark as latency_benchmark

    aggregate = {
        "aggregate_sha256": "a" * 64,
        "statistics": {
            "is_valid": True,
            "raw_sample_count_total": 4000,
            "median_ms": 57.0,
            "p95_ms": 58.0,
            "bootstrap_median_ci_ms": [56.9, 57.1],
        },
    }
    monkeypatch.setattr(
        latency_benchmark,
        "combine_replica_records",
        lambda _records: copy.deepcopy(aggregate),
    )
    evidence = {
        "schema_version": 1,
        "descriptor_sha256": "b" * 64,
        "batches": [],
    }
    evidence["sha256"] = run_campaign.manifest_sha256(evidence)
    records = [
        {
            "tao_job_id": "latency-child",
            "input_evidence": copy.deepcopy(evidence),
            "rank_runtime_evidence": {
                **sealed_manifest["runtime"]["hardware_contract"],
                "local_rank": rank,
                "hostname": "node",
                "nvidia_smi": f"gpu-{rank}",
            },
        }
        for rank in range(8)
    ]
    remote_commands = []

    def _remote_output(command, timeout=900):
        del timeout
        remote_commands.append(command)
        return json.dumps(records)

    monkeypatch.setattr(run_campaign, "remote_output", _remote_output)

    class SDK:
        def __init__(self):
            self.created = 0

        def create_job(self, **_kwargs):
            self.created += 1
            return SimpleNamespace(id="latency-child")

        def get_job_status(self, _job_id):
            return SimpleNamespace(status="Complete")

        def get_job_logs(self, _job_id, tail=0):
            del tail
            return "TAO_AUTOML_DINO_LATENCY_COMPLETE"

        def get_job_results_dir(self, _job_id):
            return "lustre:///results/latency-child"

    sdk = SDK()
    ledger = []
    fingerprint = "c" * 64
    metrics, child = run_campaign._launch_latency(
        sdk,
        sealed_manifest,
        {"dataset": {}, "evaluate": {}},
        "/results/checkpoint.pth",
        fingerprint,
        events=tmp_path / "events.jsonl",
        mode="latency",
        candidate_id="latency_rec_0",
        on_submitted=ledger.append,
    )
    assert metrics["latency_ms"] == 57.0
    assert ledger[0]["status"] == "submitted"
    assert ledger[0]["tao_job_id"] == "latency-child"
    assert child["aggregate_evidence"]["aggregate"] == aggregate
    assert sdk.created == 1
    assert (
        '"$TAO_RESULTS_ROOT/$TAO_JOB_ID/latency"'
        in entrypoint_inputs[0]["command"]
    )
    assert "/results/latency-child/latency" in remote_commands[0]

    run_campaign._launch_latency(
        sdk,
        sealed_manifest,
        {"dataset": {}, "evaluate": {}},
        "/results/checkpoint.pth",
        fingerprint,
        events=tmp_path / "events.jsonl",
        mode="latency",
        candidate_id="latency_rec_0",
        existing_job=ledger[-1],
    )
    assert sdk.created == 1


def test_latency_source_enforces_cross_rank_input_identity():
    source = (HERE / "dino_latency_worker.py").read_text(encoding="utf-8")
    assert "dist.all_gather_object" in source
    assert "model_forward_plus_dino_gpu_postprocess" in source
    assert "postprocessor(outputs, original_sizes, image_names)" in source
    assert "torch.rand" not in source
