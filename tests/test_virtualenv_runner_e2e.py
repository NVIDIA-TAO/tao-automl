# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end AutoML coverage for direct virtual-environment scripts."""

from __future__ import annotations

import json
import os
import subprocess
import venv
from pathlib import Path

import pytest


_EXAMPLE_SKILL = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "virtualenv_random_forest"
    / "skill"
)


def _write_stdlib_skill(root: Path) -> Path:
    skill_dir = root / "stdlib-skill"
    references = skill_dir / "references"
    schemas = skill_dir / "schemas"
    scripts = skill_dir / "scripts"
    references.mkdir(parents=True)
    schemas.mkdir()
    scripts.mkdir()
    (references / "skill_info.yaml").write_text(
        "network_arch: stdlib_model\n"
        "actions:\n"
        "  train:\n"
        "    config_format: json\n"
        "    execution:\n"
        "      type: python_script\n"
        "      script: scripts/train.py\n"
        "      args: [--config, '{config_path}']\n"
        "    outputs:\n"
        "      results_dir:\n"
        "        type: folder\n"
    )
    (references / "spec_template_train.yaml").write_text(
        "model:\n  width: 2\nresults_dir: ''\n"
    )
    (schemas / "train.schema.json").write_text(json.dumps({
        "type": "object",
        "default": {"model": {"width": 2}, "results_dir": ""},
        "properties": {
            "model": {
                "type": "object",
                "properties": {
                    "width": {
                        "type": "integer",
                        "default": 2,
                        "minimum": 1,
                        "maximum": 3,
                        "automl_enabled": True,
                    }
                },
            },
            "results_dir": {"type": "string", "default": ""},
        },
    }))
    (scripts / "train.py").write_text(
        "import argparse, json, sys\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--config', required=True)\n"
        "args = parser.parse_args()\n"
        "spec = json.loads(Path(args.config).read_text())\n"
        "accuracy = 0.8 + spec['model']['width'] / 100\n"
        "output = Path(spec['results_dir'])\n"
        "output.mkdir(parents=True, exist_ok=True)\n"
        "(output / 'metrics.json').write_text(json.dumps({\n"
        "    'accuracy': accuracy, 'python_prefix': sys.prefix\n"
        "}))\n"
        "print(f'accuracy: {accuracy}', flush=True)\n"
    )
    return skill_dir


def test_automl_runs_stdlib_script_in_isolated_virtualenv(tmp_path):
    from tao_automl.runner import AutoMLRunner
    from tao_sdk.platforms.virtualenv import VirtualEnvSDK

    venv_path = tmp_path / "stdlib-venv"
    venv.EnvBuilder(with_pip=False, system_site_packages=False).create(venv_path)
    sdk = VirtualEnvSDK(
        venv_path=venv_path,
        work_dir=tmp_path / "stdlib-jobs",
        state_file=tmp_path / "stdlib-jobs.db",
    )
    runner = AutoMLRunner(
        sdk=sdk,
        skill_dir=_write_stdlib_skill(tmp_path),
        action="train",
        poll_interval=0.01,
    )

    result = runner.run(
        automl_settings={
            "algorithm": "bayesian",
            "metric": "accuracy",
            "direction": "maximize",
            "automl_max_recommendations": 1,
            "run_baseline": False,
            "run_final_evaluation": False,
        },
        automl_hyperparameters=["model.width"],
        workspace_path=str(tmp_path / "stdlib-automl"),
        gpu_count=0,
    )

    assert result["progress"]["completed"] == 1
    assert result["best"]["metric_value"] > 0.8
    job = sdk.list_jobs()[0]
    metrics = json.loads(
        (Path(job["local_results_dir"]) / "results_dir/metrics.json").read_text()
    )
    assert Path(metrics["python_prefix"]).resolve() == venv_path.resolve()


@pytest.mark.skipif(
    os.environ.get("TAO_AUTOML_RUN_NETWORK_E2E") != "1",
    reason="set TAO_AUTOML_RUN_NETWORK_E2E=1 to install the public model dependency",
)
def test_automl_trains_public_model_in_isolated_virtualenv(tmp_path):
    from tao_automl.runner import AutoMLRunner
    from tao_sdk.platforms.virtualenv import VirtualEnvSDK

    venv_path = tmp_path / "model venv"
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(venv_path)
    subprocess.run(
        [
            str(venv_path / "bin" / "python"),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "scikit-learn==1.7.2",
        ],
        check=True,
    )
    jobs_dir = tmp_path / "jobs"

    sdk = VirtualEnvSDK(
        venv_path=venv_path,
        work_dir=jobs_dir,
        state_file=tmp_path / "jobs.db",
    )
    runner = AutoMLRunner(
        sdk=sdk,
        skill_dir=_EXAMPLE_SKILL,
        action="train",
        poll_interval=0.01,
    )
    result = runner.run(
        automl_settings={
            "algorithm": "bayesian",
            "metric": "accuracy",
            "direction": "maximize",
            "automl_max_recommendations": 2,
            "run_baseline": False,
            "run_final_evaluation": False,
        },
        automl_hyperparameters=["model.n_estimators", "model.max_depth"],
        workspace_path=str(tmp_path / "automl-state"),
        gpu_count=0,
    )

    assert result["progress"]["completed"] == 2
    assert result["best"]["metric_value"] > 0.8
    assert {item["status"] for item in result["history"]} == {"success"}

    jobs = sdk.list_jobs()
    assert len(jobs) == 2
    for job in jobs:
        metrics_path = (
            Path(job["local_results_dir"]) / "results_dir" / "metrics.json"
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert metrics["accuracy"] > 0.8
        assert Path(metrics["python_prefix"]).resolve() == venv_path.resolve()
        assert 4 <= metrics["model"]["n_estimators"] <= 12
        assert 1 <= metrics["model"]["max_depth"] <= 5
