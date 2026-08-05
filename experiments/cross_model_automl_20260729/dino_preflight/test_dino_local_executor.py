# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import dino_local_executor as local
import dino_local_factories as factories
import dino_local_launch as launch
import test_dino_preflight as plan_fixtures
from tao_automl.ptm_preflight import CheckpointLoadSmokeRequest
from dino_preflight import (
    DINOPreflightCommandPlan,
    run_dino_local_preflight,
)
from dino_local_executor import (
    DINOLocalDockerExecutor,
    DINOLocalExecutionError,
    DINOLocalExecutorConfig,
    DINOLocalExecutorHooks,
    ContainerLatencyRuntime,
    DockerBind,
    LatencyRuntime,
)


def _sha(value: str | bytes) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _fixture_plan(tmp_path: Path) -> DINOPreflightCommandPlan:
    skill = plan_fixtures.skill_dir.__wrapped__(tmp_path)
    voc = plan_fixtures.voc_evidence.__wrapped__(tmp_path)
    inventory = plan_fixtures.resolved_inventory.__wrapped__(tmp_path)
    settings = plan_fixtures.settings.__wrapped__(skill)
    return plan_fixtures.plan.__wrapped__(
        voc,
        inventory,
        skill,
        settings,
    )


@pytest.fixture
def plan(tmp_path: Path) -> DINOPreflightCommandPlan:
    return _fixture_plan(tmp_path)


def _config(
    plan: DINOPreflightCommandPlan,
    tmp_path: Path,
    *,
    required_environment=(),
) -> DINOLocalExecutorConfig:
    results = (tmp_path / "local_executor_results").resolve()
    return DINOLocalExecutorConfig(
        plan_sha256=plan.plan_sha256,
        image=plan.settings.runtime_image_contract.runtime_image,
        results_root=results,
        mounts=(
            DockerBind(
                host_path=tmp_path.resolve(),
                container_path="/fixture",
                read_only=True,
            ),
            DockerBind(
                host_path=results,
                container_path="/results",
                read_only=False,
            ),
        ),
        required_environment=tuple(required_environment),
        poll_interval_seconds=0.001,
        max_polls=3,
        shm_size="8g",
        container_user="1000:1000",
    )


class FakeImageInspector:
    def __init__(
        self,
        config: DINOLocalExecutorConfig,
        *,
        returncode: int = 0,
        digest_matches: bool = True,
    ):
        self.calls = []
        image = config.image.split("@", 1)[0]
        repository = image.rsplit(":", 1)[0]
        digest = config.image.rsplit(":", 1)[1]
        if not digest_matches:
            digest = "f" * 64
        self.stdout = json.dumps(
            [
                {
                    "Id": f"sha256:{_sha('image-config')}",
                    "RepoDigests": [f"{repository}@sha256:{digest}"],
                }
            ]
        )
        self.returncode = returncode

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), copy.deepcopy(kwargs)))
        return subprocess.CompletedProcess(
            args=argv,
            returncode=self.returncode,
            stdout=self.stdout,
            stderr="not exposed",
        )


class FakeEntrypointBuilder:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        raw = kwargs["command"]
        if "dino_ptm_smoke_worker.py" in raw:
            action = "backbone_ptm_smoke"
        elif "dino_latency_worker.py" in raw:
            action = "dino_model_forward_latency"
        else:
            action = raw.split()[1]
        return {
            "command": json.dumps(
                {
                    "action": action,
                    "raw_command": raw,
                    "specs": kwargs["specs"],
                },
                sort_keys=True,
            ),
            "args_template": "",
        }


def _write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(item, sort_keys=True, allow_nan=False) + "\n"
            for item in records
        ),
        encoding="utf-8",
    )


class FakeDockerSDK:
    def __init__(
        self,
        results_root: Path,
        *,
        terminal_state: str = "Complete",
        omit_success_action: str | None = None,
        invalid_full_metric: bool = False,
        ambiguous_checkpoint: bool = False,
        invalid_backbone_evidence: bool = False,
    ):
        self.results_root = results_root
        self.terminal_state = terminal_state
        self.omit_success_action = omit_success_action
        self.invalid_full_metric = invalid_full_metric
        self.ambiguous_checkpoint = ambiguous_checkpoint
        self.invalid_backbone_evidence = invalid_backbone_evidence
        self.create_calls = []
        self._roots = {}

    def create_job(self, **kwargs):
        self.create_calls.append(copy.deepcopy(kwargs))
        job_id = f"fake-job-{len(self.create_calls):03d}"
        root = self.results_root / job_id
        root.mkdir(parents=True)
        self._roots[job_id] = root
        payload = json.loads(kwargs["command"])
        action = payload["action"]
        specs = payload["specs"]
        if action == "backbone_ptm_smoke":
            raw = payload["raw_command"]
            ptm_id = raw.split("--ptm-id ", 1)[1].split()[0].strip("'")
            digest = raw.split("--checkpoint-sha256 ", 1)[1].split()[0].strip(
                "'"
            )
            evidence = {
                "schema_version": 1,
                "ptm_id": ptm_id,
                "checkpoint_sha256": digest,
                "checkpoint_target": "model.pretrained_backbone_path",
                "device": "cuda:0",
                "loaded": True,
                "real_data": True,
                "train": {"batches": 1, "finite": True, "loss": 2.1},
                "validation": {
                    "batches": 1,
                    "finite": True,
                    "loss": 1.9,
                },
                "inference": {
                    "batches": 1,
                    "finite": True,
                    "output_tensor_count": 4,
                },
            }
            if self.invalid_backbone_evidence:
                evidence["inference"]["batches"] = 2
            path = root / "results_dir" / "ptm_smoke_evidence.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(evidence), encoding="utf-8")
        elif action == "dino_model_forward_latency":
            from tao_automl.latency_benchmark import (
                LatencyBenchmarkContract,
                ReplicaIdentity,
                run_replica_benchmark,
            )
            from tao_automl.latency_stats import LatencyValidityThresholds

            raw = payload["raw_command"]
            contract_container = raw.split("--contract ", 1)[1].split()[0].strip(
                "'"
            )
            contract_path = self.results_root / Path(
                contract_container
            ).relative_to("/results")
            contract_document = json.loads(
                contract_path.read_text(encoding="utf-8")
            )
            assert contract_document.pop("schema_version") == 1
            contract_document["validity_thresholds"] = (
                LatencyValidityThresholds(
                    **contract_document["validity_thresholds"]
                )
            )
            contract = LatencyBenchmarkContract(**contract_document)
            fingerprint = raw.split(
                "--candidate-fingerprint ", 1
            )[1].split()[0].strip("'")
            tick = 0

            def clock():
                nonlocal tick
                tick += 1_000_000
                return tick

            record = run_replica_benchmark(
                contract=contract,
                identity=ReplicaIdentity(
                    rank=0,
                    world_size=1,
                    device_id="cuda:0",
                    hardware_sha256=_sha("fake-container-gpu"),
                ),
                candidate_fingerprint=fingerprint,
                step=lambda _round, _iteration: None,
                synchronize=lambda: None,
                clock_ns=clock,
            )
            output_container = kwargs["env_vars"]["TAO_DINO_LATENCY_OUTPUT"]
            output_path = self.results_root / Path(
                output_container
            ).relative_to("/results")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(record, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            metric = 0.42
            dry = bool(specs.get("train", {}).get("is_dry_run"))
            if action == "train" and not dry and self.invalid_full_metric:
                metric = 1.5
            kpi_key = "val_mAP50" if action == "train" else "test_mAP50"
            records = [
                {
                    "status": "STARTED",
                    "message": f"Starting DINO {action}.",
                },
                {
                    "status": "RUNNING",
                    "message": "metric",
                    "kpi": {kpi_key: str(metric)},
                },
            ]
            if action == "train" and not dry:
                records.append(
                    {
                        "status": "RUNNING",
                        "message": "Training loop in progress",
                        "epoch": 0,
                        "step": 2,
                        "max_epoch": 0,
                        "max_step": 2,
                        "kpi": {"val_mAP50": str(metric)},
                    }
                )
                train_dir = root / "results_dir" / "train"
                train_dir.mkdir(parents=True, exist_ok=True)
                (train_dir / "model_epoch_000_step_00002.pth").write_bytes(
                    b"full-epoch-checkpoint"
                )
                if self.ambiguous_checkpoint:
                    (
                        train_dir / "model_epoch_000_step_00002.tlt"
                    ).write_bytes(b"ambiguous")
            if action == "inference":
                output = root / "results_dir" / "inference" / "prediction.txt"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("prediction", encoding="utf-8")
            if action != self.omit_success_action:
                records.append(
                    {
                        "status": "RUNNING",
                        "message": f"{action.capitalize()} finished successfully.",
                    }
                )
            _write_jsonl(
                root / "results_dir" / action / "status.json",
                records,
            )
        return SimpleNamespace(id=job_id, results_dir=str(root))

    def get_job_status(self, job_id):
        return SimpleNamespace(status=self.terminal_state, message="fake")

    def get_job_results_dir(self, job_id):
        return str(self._roots[job_id])

    def get_job_logs(self, job_id, tail=None):
        raise AssertionError("executor must not read or copy raw job logs")


class CountingLatencyFactory:
    def __init__(self):
        self.steps = 0
        self.synchronizations = 0
        self.calls = []
        self.tick = 0

    def __call__(self, **kwargs):
        self.calls.append(kwargs)

        def step(_round, _iteration):
            self.steps += 1

        def synchronize():
            self.synchronizations += 1

        def clock():
            self.tick += 1_000_000
            return self.tick

        return LatencyRuntime(
            step=step,
            synchronize=synchronize,
            clock_ns=clock,
            hardware_sha256=_sha("one-fake-gpu"),
        )


class ResumeRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        digest = hashlib.sha256(kwargs["state_path"].read_bytes()).hexdigest()
        next_request = _sha("next-request")
        return {
            "interrupted": True,
            "state_saved": True,
            "state_sha256": digest,
            "resumed": True,
            "replay_deterministic": True,
            "expected_next_request_sha256": next_request,
            "actual_next_request_sha256": next_request,
            "no_duplicate_trials": True,
            "no_lost_trials": True,
        }


def _executor(
    plan,
    config,
    *,
    sdk=None,
    builder=None,
    inspector=None,
    latency=None,
    resume=None,
):
    sdk = sdk or FakeDockerSDK(config.results_root)
    builder = builder or FakeEntrypointBuilder()
    inspector = inspector or FakeImageInspector(config)
    latency = latency or CountingLatencyFactory()
    resume = resume or ResumeRunner()
    executor = DINOLocalDockerExecutor(
        plan=plan,
        config=config,
        hooks=DINOLocalExecutorHooks(latency, resume),
        sdk=sdk,
        entrypoint_builder=builder,
        process_runner=inspector,
        sleeper=lambda _seconds: None,
    )
    return executor, sdk, builder, inspector, latency, resume


def test_complete_plan_uses_sdk_yaml_actions_backbone_worker_and_latency(
    plan,
    tmp_path,
    monkeypatch,
):
    secret = "must-never-appear"
    monkeypatch.setenv("NGC_KEY", secret)
    config = _config(plan, tmp_path, required_environment=("NGC_KEY",))
    executor, sdk, builder, inspector, latency, resume = _executor(
        plan, config
    )

    report = run_dino_local_preflight(plan=plan, executor=executor)

    assert report["completion_state"] == "completed"
    assert report["slurm_ready"] is True
    assert [json.loads(call["command"])["action"] for call in sdk.create_calls] == [
        "train",
        "train",
        "evaluate",
        "inference",
        "backbone_ptm_smoke",
        "train",
        "evaluate",
        "inference",
    ]
    assert all(call["gpu_count"] == 1 for call in sdk.create_calls)
    assert all(call["run_as_user"] is True for call in sdk.create_calls)
    assert all(call["image"] == config.image for call in sdk.create_calls)
    assert all(
        secret not in json.dumps(call, sort_keys=True)
        for call in sdk.create_calls
    )
    assert all(call["config_format"] == "yaml" for call in builder.calls)
    assert all(call["specs"] is not None for call in builder.calls)
    assert any(
        "dino_ptm_smoke_worker.py" in call["command"]
        for call in builder.calls
    )
    backbone_call = next(
        call
        for call in builder.calls
        if "dino_ptm_smoke_worker.py" in call["command"]
    )
    assert backbone_call["specs"]["model"][
        "pretrained_backbone_path"
    ].startswith("/fixture/")
    assert backbone_call["specs"]["train"]["pretrained_model_path"] == ""
    serialized_specs = json.dumps(
        [call["specs"] for call in builder.calls],
        sort_keys=True,
    )
    assert str(tmp_path) not in serialized_specs
    assert inspector.calls == [
        (
            ["docker", "image", "inspect", config.image],
            {"check": False, "capture_output": True, "text": True},
        )
    ]
    assert all("pull" not in item for item in inspector.calls[0][0])
    assert latency.steps == 50 + (5 * 100)
    assert latency.synchronizations == 2 * latency.steps
    assert len(latency.calls) == 1
    assert latency.calls[0]["checkpoint_path"].is_file()
    assert len(resume.calls) == 1


def test_config_is_external_non_secret_fixed_gpu_and_exact_digest(
    plan,
    tmp_path,
):
    config = _config(plan, tmp_path)
    path = (tmp_path / "executor.yaml").resolve()
    path.write_text(yaml.safe_dump(config.public_dict()), encoding="utf-8")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw.pop("gpu_count")
    raw.pop("docker_pull_policy")
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    loaded = DINOLocalExecutorConfig.from_file(path)

    assert loaded == config
    public = loaded.public_dict()
    assert public["gpu_count"] == 1
    assert public["docker_pull_policy"] == "never"
    assert loaded.image == plan.settings.runtime_image_contract.runtime_image


@pytest.mark.parametrize(
    "patch,code",
    [
        ({"image": "nvcr.io/nvidia/tao/tao-toolkit:7.0.1-pyt"}, "unpinned_image"),
        ({"ngc_key": "secret-value"}, "inline_secret_forbidden"),
        ({"password": "secret-value"}, "inline_secret_forbidden"),
    ],
)
def test_config_rejects_unpinned_images_and_inline_secrets(
    plan,
    tmp_path,
    patch,
    code,
):
    raw = _config(plan, tmp_path).public_dict()
    raw.pop("gpu_count")
    raw.pop("docker_pull_policy")
    raw.update(patch)
    with pytest.raises(DINOLocalExecutionError) as raised:
        DINOLocalExecutorConfig.from_mapping(raw)
    assert raised.value.code == code
    assert "secret-value" not in str(raised.value)


def test_missing_required_environment_does_not_name_or_expose_secret(
    plan,
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("NGC_KEY", raising=False)
    config = _config(plan, tmp_path, required_environment=("NGC_KEY",))
    with pytest.raises(DINOLocalExecutionError) as raised:
        DINOLocalDockerExecutor(
            plan=plan,
            config=config,
            hooks=DINOLocalExecutorHooks(
                CountingLatencyFactory(), ResumeRunner()
            ),
        )
    assert raised.value.code == "missing_environment"
    assert "NGC_KEY" not in str(raised.value)


def test_image_must_be_present_locally_and_match_repository_digest(
    plan,
    tmp_path,
):
    config = _config(plan, tmp_path)
    first = plan.commands[0]
    for inspector, expected in (
        (FakeImageInspector(config, returncode=1), "local_image_missing"),
        (
            FakeImageInspector(config, digest_matches=False),
            "image_digest_mismatch",
        ),
    ):
        executor, *_ = _executor(plan, config, inspector=inspector)
        result = executor(first)
        assert result.passed is False
        assert result.code == expected
        assert inspector.calls[0][0][:3] == ["docker", "image", "inspect"]
        assert "pull" not in inspector.calls[0][0]


def test_skill_image_or_plan_digest_mismatch_fails_before_sdk(
    plan,
    tmp_path,
):
    config = _config(plan, tmp_path)
    wrong = DINOLocalExecutorConfig(
        plan_sha256=config.plan_sha256,
        image=(
            f"{plan.settings.runtime_image_contract.runtime_repository}"
            f"@sha256:{'f' * 64}"
        ),
        results_root=config.results_root,
        mounts=config.mounts,
        poll_interval_seconds=config.poll_interval_seconds,
    )
    with pytest.raises(DINOLocalExecutionError) as raised:
        DINOLocalDockerExecutor(
            plan=plan,
            config=wrong,
            hooks=DINOLocalExecutorHooks(
                CountingLatencyFactory(), ResumeRunner()
            ),
        )
    assert raised.value.code == "image_contract_mismatch"


def test_backbone_worker_evidence_must_be_exact_finite_one_batch(
    plan,
    tmp_path,
):
    config = _config(plan, tmp_path)
    sdk = FakeDockerSDK(
        config.results_root,
        invalid_backbone_evidence=True,
    )
    executor, *_ = _executor(plan, config, sdk=sdk)
    report = run_dino_local_preflight(plan=plan, executor=executor)
    assert report["completion_state"] == "failed"
    assert report["failure"]["stage"] == "eligible_ptm_smoke"
    assert report["failure"]["code"] == "adapter_rejected"
    failed = [
        item
        for item in executor._results.values()
        if not item.passed
    ]
    assert [item.code for item in failed] == ["invalid_smoke_evidence"]


def test_full_model_smoke_uses_skill_actions_not_backbone_worker(
    plan,
    tmp_path,
):
    config = _config(plan, tmp_path)
    executor, sdk, builder, *_ = _executor(plan, config)
    dataset = executor(plan.commands[0])
    default_load = executor(plan.commands[1])
    full_smoke = next(
        item
        for item in plan.commands_for_stage("eligible_ptm_smoke")
        if item.metadata["checkpoint_target"] == "train.pretrained_model_path"
    )
    result = executor(full_smoke)
    assert dataset.passed and default_load.passed and result.passed
    actions = [json.loads(item["command"])["action"] for item in sdk.create_calls]
    assert actions == ["train", "train", "evaluate", "inference"]
    assert not any(
        "dino_ptm_smoke_worker.py" in item["command"]
        for item in builder.calls
    )
    assert builder.calls[0]["specs"]["train"]["is_dry_run"] is True
    assert builder.calls[1]["specs"]["train"]["is_dry_run"] is True
    assert len(
        builder.calls[2]["specs"]["dataset"]["test_data_sources"][
            "json_file"
        ]
    ) > 0
    assert builder.calls[3]["specs"]["dataset"][
        "infer_data_sources"
    ]["image_dir"][0].endswith("/smoke_inference_subset")


def test_action_status_success_is_exact_and_raw_logs_are_never_read(
    plan,
    tmp_path,
):
    config = _config(plan, tmp_path)
    sdk = FakeDockerSDK(
        config.results_root,
        omit_success_action="train",
    )
    executor, *_ = _executor(plan, config, sdk=sdk)
    report = run_dino_local_preflight(plan=plan, executor=executor)
    assert report["completion_state"] == "failed"
    assert report["failure"]["stage"] == "default_ptm_load"
    assert report["failure"]["code"] == "adapter_rejected"
    assert executor._results["default_ptm_load"].code == (
        "action_status_incomplete"
    )


def test_terminal_sdk_failure_is_classified_without_logs_or_retries(
    plan,
    tmp_path,
):
    config = _config(plan, tmp_path)
    sdk = FakeDockerSDK(config.results_root, terminal_state="Error")
    executor, *_ = _executor(plan, config, sdk=sdk)
    assert executor(plan.commands[0]).passed
    result = executor(plan.commands[1])
    assert result.passed is False
    assert result.code == "tao_action_failed"
    assert len(sdk.create_calls) == 1


@pytest.mark.parametrize(
    "sdk_kwargs,expected",
    [
        ({"invalid_full_metric": True}, "invalid_metric"),
        ({"ambiguous_checkpoint": True}, "ambiguous_checkpoint"),
    ],
)
def test_full_epoch_rejects_invalid_metric_or_ambiguous_checkpoint(
    plan,
    tmp_path,
    sdk_kwargs,
    expected,
):
    config = _config(plan, tmp_path)
    sdk = FakeDockerSDK(config.results_root, **sdk_kwargs)
    executor, *_ = _executor(plan, config, sdk=sdk)
    report = run_dino_local_preflight(plan=plan, executor=executor)
    assert report["completion_state"] == "failed"
    assert report["failure"]["stage"] == "default_model_full_epoch"
    assert report["failure"]["code"] == "adapter_rejected"
    assert executor._results["default_model_full_epoch"].code == expected


def test_undeclared_input_mount_fails_closed_before_action_submission(
    plan,
    tmp_path,
):
    config = _config(plan, tmp_path)
    results = config.results_root
    narrow = DINOLocalExecutorConfig(
        plan_sha256=config.plan_sha256,
        image=config.image,
        results_root=results,
        mounts=(
            DockerBind(
                plan.voc_integrity.dataset_root,
                "/dataset",
                True,
            ),
            DockerBind(results, "/results", False),
        ),
        poll_interval_seconds=0.001,
    )
    executor, sdk, *_ = _executor(plan, narrow)
    assert executor(plan.commands[0]).passed
    result = executor(plan.commands[1])
    assert result.passed is False
    assert result.code == "undeclared_mount"
    assert sdk.create_calls == []


def test_command_dependencies_and_cached_execution_prevent_reordering(
    plan,
    tmp_path,
):
    config = _config(plan, tmp_path)
    executor, sdk, *_ = _executor(plan, config)
    full_epoch = plan.commands_for_stage("default_model_full_epoch")[0]
    result = executor(full_epoch)
    assert not result.passed
    assert result.code == "dependency_not_complete"
    assert sdk.create_calls == []
    assert executor(full_epoch) is result


def test_materialized_artifacts_are_create_or_verify_and_detect_drift(
    plan,
    tmp_path,
):
    config = _config(plan, tmp_path)
    executor, *_ = _executor(plan, config)
    assert executor(plan.commands[0]).passed
    label = (
        config.results_root
        / ".dino-preflight"
        / plan.plan_sha256
        / "inputs"
        / "label_map.txt"
    )
    label.write_text("drift", encoding="utf-8")
    second, *_ = _executor(plan, config)
    result = second(plan.commands[0])
    assert not result.passed
    assert result.code == "artifact_drift"


def test_cli_loads_live_plan_and_hooks_from_external_factories(
    plan,
    tmp_path,
    monkeypatch,
    capsys,
):
    config = _config(plan, tmp_path)
    raw = config.public_dict()
    raw.pop("gpu_count")
    raw.pop("docker_pull_policy")
    config_path = (tmp_path / "cli_config.yaml").resolve()
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    report_path = (tmp_path / "cli_report.json").resolve()

    module = types.ModuleType("fixture_dino_factories")
    module.plan = lambda: plan
    module.hooks = lambda _plan, _config: DINOLocalExecutorHooks(
        CountingLatencyFactory(), ResumeRunner()
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    fixture_executor = plan_fixtures.FixtureExecutor(plan)
    monkeypatch.setattr(
        local,
        "DINOLocalDockerExecutor",
        lambda **_kwargs: fixture_executor,
    )

    rc = local.main(
        [
            "--config",
            str(config_path),
            "--plan-factory",
            "fixture_dino_factories:plan",
            "--hooks-factory",
            "fixture_dino_factories:hooks",
            "--report",
            str(report_path),
        ]
    )

    assert rc == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["completion_state"] == "completed"
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "completion_state": "completed",
        "plan_sha256": plan.plan_sha256,
        "report": str(report_path),
    }


def test_cli_failure_output_never_echoes_inline_secret(
    tmp_path,
    capsys,
):
    secret = "super-secret-value"
    path = (tmp_path / "bad.yaml").resolve()
    path.write_text(yaml.safe_dump({"password": secret}), encoding="utf-8")
    rc = local.main(
        [
            "--config",
            str(path),
            "--plan-factory",
            "bad:factory",
            "--hooks-factory",
            "bad:hooks",
            "--report",
            str((tmp_path / "report.json").resolve()),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert secret not in captured.out
    assert secret not in captured.err
    assert json.loads(captured.err)["code"] == "inline_secret_forbidden"


def test_concrete_hooks_run_container_latency_and_real_production_resume(
    plan,
    tmp_path,
):
    config = _config(plan, tmp_path)
    sdk = FakeDockerSDK(config.results_root)
    builder = FakeEntrypointBuilder()
    hooks = factories.build_default_hooks(plan, config)
    assert isinstance(
        hooks.latency_runtime_factory(
            plan=plan,
            command=plan.commands_for_stage("latency_instrumentation")[0],
            checkpoint_path=Path(__file__).resolve(),
            inference_spec={"inference": {}},
        ),
        ContainerLatencyRuntime,
    )
    executor = DINOLocalDockerExecutor(
        plan=plan,
        config=config,
        hooks=hooks,
        sdk=sdk,
        entrypoint_builder=builder,
        process_runner=FakeImageInspector(config),
        sleeper=lambda _seconds: None,
    )

    report = run_dino_local_preflight(plan=plan, executor=executor)

    assert report["completion_state"] == "completed"
    assert any(
        json.loads(item["command"])["action"]
        == "dino_model_forward_latency"
        for item in sdk.create_calls
    )
    replay = (
        config.results_root
        / ".dino-preflight"
        / plan.plan_sha256
        / "production_resume_replay"
        / "completion.json"
    )
    assert replay.is_file()
    replay_document = json.loads(replay.read_text(encoding="utf-8"))
    assert replay_document["implementation"] == {
        "brain": "tao_automl.brain.bayesian.Bayesian",
        "controller": "tao_automl.controller.controller.Controller",
        "state_store": "tao_automl.state.state_store.StateStore",
    }
    assert replay_document["evidence"]["replay_deterministic"] is True


def test_runtime_checkpoint_preflight_uses_real_standard_dry_run_contract(
    plan,
    tmp_path,
):
    results = (tmp_path / "runtime_load_smoke_results").resolve()
    sdk = FakeDockerSDK(results)
    builder = FakeEntrypointBuilder()
    smoke = launch.DINOStandardDryRunLoadSmoke(
        voc=plan.voc_integrity,
        cache_root=tmp_path.resolve(),
        results_root=results,
        seed=plan.settings.seed,
        container_user="1000:1000",
        poll_interval_seconds=0.001,
        max_polls=2,
        sdk=sdk,
        entrypoint_builder=builder,
        sleeper=lambda _seconds: None,
    )
    command = next(
        item
        for item in plan.commands_for_stage("eligible_ptm_smoke")
        if item.metadata["checkpoint_target"] == "train.pretrained_model_path"
    )
    checkpoint = Path(
        command.specs_by_action["train"]["train"][
            "pretrained_model_path"
        ]
    )
    request = CheckpointLoadSmokeRequest(
        checkpoint_id=command.ptm_id,
        model="dino",
        task="object_detection",
        tao_version=plan.settings.tao_version,
        checkpoint_path=checkpoint,
        checkpoint_spec_path=checkpoint,
        checkpoint_spec={},
        default_spec_overrides={},
        registry_record={
            "checkpoint_target": "train.pretrained_model_path",
        },
    )

    result = smoke(request)

    assert result.ok is True
    assert result.details["execution_backend"] == "docker"
    assert result.details["execution_driver"] == "docker_sdk"
    assert result.details["container_identity"] == (
        factories.TAO71_RUNTIME_IMAGE.rsplit("@", 1)[1]
    )
    assert result.details["checkpoint_loaded"] is True
    assert result.details["state_dict_compatible"] is True
    assert result.details["lightning_fast_dev_run"] is True
    identity = smoke.manifest_identity()
    assert identity["expected_train_batches"] == 1
    assert identity["expected_validation_batches"] == 1
    assert identity["dataset_manifest_sha256"] == (
        plan.voc_integrity.manifest_sha256
    )
    assert len(sdk.create_calls) == 1
    call = sdk.create_calls[0]
    assert call["image"] == factories.TAO71_RUNTIME_IMAGE
    assert call["gpu_count"] == 1
    assert builder.calls[0]["specs"]["train"]["is_dry_run"] is True
    assert builder.calls[0]["specs"]["train"][
        "pretrained_model_path"
    ].startswith("/ptm/")


def test_preparation_image_gate_inspects_exact_digest_without_pull():
    calls = []

    def inspect(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps([{
                "RepoDigests": [factories.TAO71_RUNTIME_IMAGE],
            }]),
            stderr="not exposed",
        )

    launch._verify_local_runtime_image(inspect)

    assert calls == [(
        ["docker", "image", "inspect", factories.TAO71_RUNTIME_IMAGE],
        {"check": False, "capture_output": True, "text": True},
    )]
    assert "pull" not in calls[0][0]


def test_reviewed_runtime_factory_binds_authoritative_skill_and_source():
    contract = factories.build_reviewed_runtime_image_contract()

    assert contract.runtime_image == factories.TAO71_RUNTIME_IMAGE
    assert contract.source_skill_revision == (
        factories.AUTHORITATIVE_DINO_SKILL_REVISION
    )
    assert contract.compatible_skill_revision == (
        factories.AUTHORITATIVE_DINO_SKILL_REVISION
    )
    assert contract.status == "verified"
