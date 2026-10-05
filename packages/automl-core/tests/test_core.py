# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from automl_core._worker import atomic
from automl_core.contracts import (
    Budget,
    Execution,
    Objective,
    Parameter,
    Result,
    Search,
    fingerprint,
)
from automl_core.execution import Executor
from automl_core.runner import Experiment, manage
from automl_core.search import ask, best, features, tell
from automl_core.workload import CommandWorkload


@pytest.fixture
def search():
    return Search(
        experiment_id="cpu-test",
        seed=7,
        objective=Objective(metric="validation.loss", direction="minimize"),
        budget=Budget(max_trials=4, trial_timeout_seconds=15, total_timeout_seconds=60),
        base_config={"model": {"x": 0.5}},
        parameters={"model.x": Parameter(kind="float", low=0, high=1)},
    )


@pytest.fixture
def execution():
    return Execution(kind="venv", python=sys.executable)


def workload(tmp_path, code=None):
    script = tmp_path / "train.py"
    script.write_text(
        code
        or """import json, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())
Path(sys.argv[2], 'metrics.json').write_text(json.dumps({'validation': {'loss': (config['model']['x']-.3)**2}}))
"""
    )
    return CommandWorkload(
        cwd=tmp_path,
        commands=[["{python}", str(script), "{config}", "{trial}"]],
        metrics_file="metrics.json",
    )


def test_cpu_run_resume_and_best(tmp_path, search, execution):
    experiment = Experiment(search, workload(tmp_path), execution, tmp_path / "runs")
    experiment.plan()
    assert not experiment.root.exists()
    report = experiment.run()
    assert [trial["state"] for trial in report["trials"]] == ["COMPLETE"] * 4
    assert report["best"]["result"]["metric"] == min(
        t["result"]["metric"] for t in report["trials"]
    )
    assert experiment.run(resume=True) == report
    with pytest.raises(ValueError, match="existing"):
        experiment.run()
    search.seed += 1
    changed = Experiment(search, experiment.workload, execution, tmp_path / "runs")
    with pytest.raises(ValueError, match="identity changed"):
        changed.run(resume=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_results_rejected(value):
    with pytest.raises(ValueError):
        Result(metric=value)


@pytest.mark.parametrize(
    "code", ["pass", "raise SystemExit(3)", "import time; time.sleep(20)"]
)
def test_failed_missing_metric_and_timeout(tmp_path, search, execution, code):
    search.budget.max_trials = 1
    search.budget.trial_timeout_seconds = 0.4
    experiment = Experiment(
        search, workload(tmp_path, code), execution, tmp_path / "runs"
    )
    report = experiment.run()
    assert report["state"] == "ERROR"
    assert report["best"] is None
    assert report["trials"][0]["result"] is None


def test_cancel_native_process(tmp_path, search, execution):
    experiment = Experiment(
        search,
        workload(tmp_path, "import time; time.sleep(20)"),
        execution,
        tmp_path / "runs",
    )
    results = []
    controller = threading.Thread(target=lambda: results.append(experiment.run()))
    controller.start()
    receipt = experiment.root / "trial-00000/_receipt.json"
    deadline = time.monotonic() + 10
    while not receipt.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    experiment.cancel()
    controller.join(timeout=10)
    assert not controller.is_alive()
    assert results[0]["state"] == "CANCELED"
    assert results[0]["trials"][0]["state"] == "CANCELED"


def test_reconcile_after_controller_crash(tmp_path, search, execution):
    search.budget.max_trials = 1
    work = workload(tmp_path)
    experiment = Experiment(search, work, execution, tmp_path / "runs")
    submit = experiment.executor.submit

    def interrupted(*args):
        submit(*args)
        raise RuntimeError("simulate controller crash after launch")

    experiment.executor.submit = interrupted
    with pytest.raises(RuntimeError):
        experiment.run()
    recovered = Experiment(search, work, execution, tmp_path / "runs")
    report = recovered.run(resume=True)
    assert len(report["trials"]) == 1
    assert report["state"] == "COMPLETE"


def test_unknown_launch_does_not_retry(tmp_path, search, execution):
    experiment = Experiment(search, workload(tmp_path), execution, tmp_path / "runs")
    experiment.executor.submit = lambda *args: None
    with pytest.raises(ValueError, match="UNKNOWN"):
        experiment.run()
    with pytest.raises(ValueError, match="UNKNOWN"):
        experiment.run(resume=True)
    assert len(experiment.status()["trials"]) == 1


def test_direction_failure_and_idempotent_tell(search):
    point, config = ask(search, [])
    assert ask(search, []) == (point, config)
    history = [{"point": point, "config": config, "state": "PENDING"}]
    tell(history[0], "ERROR")
    tell(history[0], "ERROR")
    with pytest.raises(ValueError, match="Conflicting"):
        tell(history[0], "COMPLETE", Result(metric=0))
    for metric in [4, 2, 9]:
        history.append({"point": [metric / 10], "state": "PENDING"})
        tell(history[-1], "COMPLETE", Result(metric=metric))
    assert best(search, history)["result"]["metric"] == 2
    search.objective.direction = "maximize"
    assert best(search, history)["result"]["metric"] == 9
    assert all(math.isfinite(x) for x in ask(search, history)[0])


@pytest.mark.parametrize(
    "change",
    [
        {"algorithm": "asha"},
        {"backend": "framework"},
        {"parameters": {"missing": {"kind": "float", "low": 0, "high": 1}}},
    ],
)
def test_contract_rejects_undeclared_features(search, change):
    with pytest.raises(ValueError):
        Search.model_validate({**search.model_dump(), **change})


def test_docker_plan_no_pull_and_env_names_only(tmp_path, monkeypatch):
    execution = Execution(
        kind="docker",
        python="python3",
        image="example@sha256:abc",
        environment=["TEST_SECRET"],
        gpu_devices="0,1",
    )
    work = workload(tmp_path)
    trial = tmp_path / "trial"
    plan = work.plan({"model": {"x": 0.5}}, trial, execution)
    args = Executor(execution).docker_create(trial, plan)
    monkeypatch.setenv("TEST_SECRET", "never-persist-me")
    assert "--pull=never" in args
    assert "TEST_SECRET" in args
    assert "never-persist-me" not in str(args)
    assert args[-3:] == [
        execution.image,
        str(trial / "_worker.py"),
        str(trial / "_execution.json"),
    ]


def test_artifact_escape(tmp_path):
    work = workload(tmp_path)
    (tmp_path / "outside.json").write_text('{"validation":{"loss":1}}')
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "metrics.json").symlink_to(tmp_path / "outside.json")
    with pytest.raises(ValueError, match="out-of-trial"):
        work.collect(trial, Objective(metric="validation.loss", direction="minimize"))


def test_nominal_and_quantized_features(search):
    search.parameters = {
        "model.a": Parameter(kind="categorical", choices=["red", "blue", "green"]),
        "model.b": Parameter(kind="int", low=1, high=3),
    }
    assert features(search, [0.5, 0.49]) == [0, 1, 0, 0.5]
    assert features(search, [0.6, 0.51]) == [0, 1, 0, 0.5]


def test_parameter_order_does_not_change_replay(search):
    search.base_config["model"]["y"] = 1
    search.parameters["model.y"] = Parameter(kind="int", low=1, high=3)
    before = ask(search, [])
    search.parameters = dict(reversed(list(search.parameters.items())))
    assert ask(search, []) == before


def test_environment_change_blocks_resume(tmp_path, search, execution):
    search.budget.max_trials = 1
    experiment = Experiment(search, workload(tmp_path), execution, tmp_path / "runs")
    experiment.run()
    experiment.executor.preflight = lambda cwd: {"different": "environment"}
    with pytest.raises(ValueError, match="environment changed"):
        experiment.run(resume=True)


def test_management_does_not_need_mutable_manifest(tmp_path, search, execution):
    search.budget.max_trials = 1
    experiment = Experiment(search, workload(tmp_path), execution, tmp_path / "runs")
    experiment.run()
    (tmp_path / "train.py").unlink()
    assert manage(tmp_path / "runs", search.experiment_id)["state"] == "COMPLETE"
    assert manage(tmp_path / "runs", search.experiment_id, cancel=True)[
        "cancel_requested"
    ]
    with pytest.raises(ValueError, match="Invalid experiment"):
        manage(tmp_path / "runs", "../outside", cancel=True)


def test_atomic_updates_from_concurrent_threads(tmp_path):
    path = tmp_path / "cancel.json"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda index: atomic(path, {"index": index}), range(100)))
    assert isinstance(json.loads(path.read_text())["index"], int)
    assert list(tmp_path.iterdir()) == [path]


def test_controller_lock(tmp_path, search, execution):
    experiment = Experiment(search, workload(tmp_path), execution, tmp_path / "runs")
    with experiment._lock(), pytest.raises(ValueError, match="Another controller"):
        experiment.run()


def test_docker_observed_state_and_owned_cancel(tmp_path, monkeypatch):
    execution = Execution(kind="docker", python="python", image="local:test")
    executor = Executor(execution)
    trial = tmp_path / "trial"
    trial.mkdir()
    calls = []
    native = {
        "State": {"Running": True},
        "Config": {"Labels": {"automl.trial": fingerprint(str(trial))}},
    }

    def docker(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, json.dumps([native]))

    monkeypatch.setattr(subprocess, "run", docker)
    assert executor.status(trial) == "RUNNING"
    executor.cancel(trial)
    assert (trial / "_cancel.json").exists()
    assert any(call[:2] == ["docker", "stop"] for call in calls)
    native["Config"]["Labels"] = {}
    with pytest.raises(ValueError, match="ownership"):
        executor.cancel(trial)


@pytest.mark.parametrize(
    "receipt,native,expected",
    [
        (
            {"state": "RUNNING"},
            {"Running": False, "Status": "exited", "ExitCode": 0},
            "UNKNOWN",
        ),
        (
            {"state": "COMPLETE"},
            {"Running": False, "Status": "exited", "ExitCode": 0},
            "COMPLETE",
        ),
        (
            {"state": "COMPLETE"},
            {"Running": False, "Status": "exited", "ExitCode": 2},
            "ERROR",
        ),
        ({}, {"Running": False, "Status": "created"}, "UNKNOWN"),
    ],
)
def test_docker_terminal_evidence(tmp_path, monkeypatch, receipt, native, expected):
    executor = Executor(Execution(kind="docker", python="python", image="local:test"))
    (tmp_path / "_receipt.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(executor, "_container", lambda trial: native)
    assert executor.status(tmp_path) == expected


def test_random_forest_cli_from_copied_assets(tmp_path):
    """With a wheel-installed test interpreter, this also verifies checkout independence."""
    import yaml

    example = Path(__file__).resolve().parents[1] / "examples"
    shutil.copytree(example, tmp_path / "example")
    manifest = tmp_path / "example/random_forest.yaml"
    data = yaml.safe_load(manifest.read_text())
    data["execution"]["python"] = sys.executable
    manifest.write_text(yaml.safe_dump(data))
    environment = {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }
    environment["TAO_SKILL_BANK_PATH"] = "/nonexistent"
    # Use the currently imported package location so this test also runs from source.
    # Wheel-isolation verification separately asserts this location is site-packages.
    import automl_core

    environment["PYTHONPATH"] = str(Path(automl_core.__file__).parents[1])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "automl_core.cli",
            "run",
            "--manifest",
            str(manifest),
            "--workspace",
            str(tmp_path / "runs"),
            "--approve",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    report = json.loads(result.stdout)
    assert len(report["trials"]) == 4
    assert all(trial["state"] == "COMPLETE" for trial in report["trials"])
    assert report["best"]["result"]["metric"] > 0.5
