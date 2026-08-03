# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the migrated AutoMLRunner.

Validates:
- SkillContext loads skill_info.yaml and spec_template_<action>.yaml.
- AutoMLRunner.__init__ takes (sdk, skill_dir, action) — no SkillBank.
- _make_sdk returns the right per-platform SDK class; rejects bad names.
- _run_one_job calls build_entrypoint with action_cfg fields and
  sdk.create_job with the resulting command + platform_kwargs.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def test_corrupt_active_job_ledger_fails_closed(tmp_path):
    from tao_automl.runner import _load_active_jobs

    (tmp_path / "active_jobs.json").write_text("{truncated")

    with pytest.raises(RuntimeError, match="refusing to launch additional jobs"):
        _load_active_jobs(str(tmp_path))


# ---------------------------------------------------------------------------
# SkillContext
# ---------------------------------------------------------------------------

def _write_fake_skill(tmp_path: Path, action: str = "train") -> Path:
    """Create a minimal skill bank layout for tests."""
    skill_dir = tmp_path / "models" / "fake-net"
    refs = skill_dir / "references"
    refs.mkdir(parents=True)
    (refs / "skill_info.yaml").write_text(
        "network_arch: fake-net\n"
        "container_image: nvcr.io/nvidia/tao/fake:0.1\n"
        "data_format: coco\n"
        "actions:\n"
        f"  {action}:\n"
        "    command: fake train -e {config_path}\n"
        "    config_format: yaml\n"
        "    inputs:\n"
        "      dataset.train_data_sources[0].image_dir:\n"
        "        type: file\n"
        "    outputs:\n"
        "      results_dir:\n"
        "        type: folder\n"
        "    upload_excludes: ['inputs/']\n"
    )
    (refs / f"spec_template_{action}.yaml").write_text(
        "train:\n"
        "  num_epochs: 12\n"
        "  optim:\n"
        "    lr: 2.0e-4\n"
        "dataset:\n"
        "  num_classes: 80\n"
    )
    return skill_dir


def _write_fake_action_skill(
    tmp_path: Path,
    *,
    network_arch: str = "dino",
    action: str = "quantize",
) -> Path:
    """Create a minimal non-train action skill that can run a shell command."""
    skill_dir = tmp_path / "models" / f"fake-{action}"
    refs = skill_dir / "references"
    refs.mkdir(parents=True)
    (refs / "skill_info.yaml").write_text(
        f"network_arch: {network_arch}\n"
        "container_image: nvcr.io/nvidia/tao/fake:0.1\n"
        "actions:\n"
        f"  {action}:\n"
        "    command: >-\n"
        "      python -c \"print('action_metric: 0.73')\"\n"
        "    config_format: yaml\n"
        "    inputs: {}\n"
        "    outputs: {}\n"
        "    upload_excludes: []\n"
    )
    results_dir = tmp_path / "action-results"
    (refs / f"spec_template_{action}.yaml").write_text(
        f"results_dir: {results_dir}\n"
        "train:\n"
        "  optim:\n"
        "    lr: 2.0e-4\n"
        "quantize:\n"
        f"  results_dir: {results_dir}\n"
    )
    return skill_dir


class _CompletedProcessSDK:
    """SDK shim that executes submitted jobs as real local subprocesses."""

    def __init__(self):
        self.jobs = {}

    def create_job(self, image, command, **kwargs):
        from tao_sdk.models import Job

        job_id = f"job-{len(self.jobs)}"
        env = dict(**kwargs.pop("env_vars", {}))
        proc_env = None
        if env:
            import os
            proc_env = os.environ.copy()
            proc_env.update(env)
        if proc_env is None:
            import os
            proc_env = os.environ.copy()
        proc_env["TAO_JOB_ID"] = job_id

        completed = subprocess.run(
            command,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=proc_env,
            check=False,
        )
        self.jobs[job_id] = completed
        return Job(
            id=job_id,
            network_arch="dino",
            action="quantize",
            workspace_id="local",
            backend_job_id=job_id,
            status="Complete" if completed.returncode == 0 else "Error",
        )

    def get_job_status(self, job_id):
        from tao_sdk.models import JobStatus

        completed = self.jobs[job_id]
        status = "Complete" if completed.returncode == 0 else "Error"
        return JobStatus(job_id=job_id, status=status)

    def get_job_logs(self, job_id, tail=None):
        logs = self.jobs[job_id].stdout or ""
        if tail is not None:
            return "\n".join(logs.splitlines()[-tail:])
        return logs


def _write_python_skill(tmp_path: Path) -> Path:
    """Create an external model skill backed by a direct Python script."""
    skill_dir = tmp_path / "models" / "public-random-forest"
    refs = skill_dir / "references"
    schemas = skill_dir / "schemas"
    scripts = skill_dir / "scripts"
    refs.mkdir(parents=True)
    schemas.mkdir()
    scripts.mkdir()
    (scripts / "train.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--config', required=True)\n"
        "parser.parse_args()\n"
        "print('accuracy: 0.875', flush=True)\n"
    )
    (refs / "skill_info.yaml").write_text(
        "network_arch: public_random_forest\n"
        "actions:\n"
        "  train:\n"
        "    config_format: json\n"
        "    execution:\n"
        "      type: python_script\n"
        "      script: scripts/train.py\n"
        "      args: [--config, '{config_path}']\n"
        "      cwd: .\n"
        "    outputs:\n"
        "      results_dir:\n"
        "        type: folder\n"
    )
    default_specs = {
        "model": {"n_estimators": 10, "max_depth": 3},
        "results_dir": "",
    }
    (refs / "spec_template_train.yaml").write_text(
        "model:\n"
        "  n_estimators: 10\n"
        "  max_depth: 3\n"
        "results_dir: ''\n"
    )
    (schemas / "train.schema.json").write_text(json.dumps({
        "type": "object",
        "default": default_specs,
        "properties": {
            "model": {
                "type": "object",
                "properties": {
                    "n_estimators": {
                        "type": "integer", "default": 10,
                        "minimum": 2, "maximum": 20,
                        "automl_enabled": True,
                    },
                    "max_depth": {
                        "type": "integer", "default": 3,
                        "minimum": 1, "maximum": 6,
                        "automl_enabled": True,
                    },
                },
            },
            "results_dir": {"type": "string", "default": ""},
        },
    }))
    return skill_dir


def test_skill_context_loads_skill_info_and_template(tmp_path):
    from tao_automl.runner import SkillContext
    skill_dir = _write_fake_skill(tmp_path)
    ctx = SkillContext(skill_dir=skill_dir, action="train")
    assert ctx.network_arch == "fake-net"
    assert ctx.action_cfg["command"] == "fake train -e {config_path}"
    assert ctx.action_cfg["config_format"] == "yaml"
    assert ctx.default_specs["train"]["num_epochs"] == 12
    assert ctx.default_specs["dataset"]["num_classes"] == 80


def test_skill_context_action_container_image_overrides_model_image(tmp_path):
    from tao_automl.runner import SkillContext
    skill_dir = _write_fake_skill(tmp_path, action="dataset_convert")
    info_path = skill_dir / "references/skill_info.yaml"
    info_path.write_text(
        info_path.read_text().replace(
            "    command: fake train -e {config_path}\n",
            "    container_image: nvcr.io/nvidia/tao/fake-ds:0.1\n"
            "    command: fake convert -e {config_path}\n",
        )
    )
    ctx = SkillContext(skill_dir=skill_dir, action="dataset_convert")
    assert ctx.container_image == "nvcr.io/nvidia/tao/fake-ds:0.1"


def test_skill_context_no_template_yields_empty_specs(tmp_path):
    """Models without a spec_template_<action>.yaml get default_specs={}.
    Caller is responsible for constructing the spec from skill SKILL.md."""
    from tao_automl.runner import SkillContext
    skill_dir = _write_fake_skill(tmp_path)
    (skill_dir / "references/spec_template_train.yaml").unlink()
    ctx = SkillContext(skill_dir=skill_dir, action="train")
    assert ctx.default_specs == {}


def test_skill_context_missing_action_raises(tmp_path):
    from tao_automl.runner import SkillContext
    skill_dir = _write_fake_skill(tmp_path, action="train")
    with pytest.raises(KeyError, match="evaluate"):
        SkillContext(skill_dir=skill_dir, action="evaluate")


def test_skill_context_missing_skill_info_raises(tmp_path):
    from tao_automl.runner import SkillContext
    with pytest.raises(FileNotFoundError, match="skill_info.yaml"):
        SkillContext(skill_dir=tmp_path / "nonexistent", action="train")


def test_generate_hyperparams_uses_selected_action_schema(monkeypatch):
    """Non-train AutoML must read the action schema instead of train."""
    from tao_automl.search_space import params

    seen = {}

    def fake_generate_schema(network, action):
        seen["network"] = network
        seen["action"] = action
        return {
            "default": {
                "quantize": {"calibration_batches": 4},
            },
            "properties": {
                "quantize": {
                    "type": "object",
                    "properties": {
                        "calibration_batches": {
                            "type": "integer",
                            "default": 4,
                            "minimum": 1,
                            "maximum": 8,
                            "automl_enabled": True,
                        }
                    },
                }
            },
        }

    monkeypatch.setattr(params, "generate_schema", fake_generate_schema)

    records, names = params.generate_hyperparams_to_search(
        network="fake-net",
        action="quantize",
        train_specs={"quantize": {"calibration_batches": 4}},
        automl_hyperparameters=None,
    )

    assert seen == {"network": "fake-net", "action": "quantize"}
    assert names == ["quantize.calibration_batches"]
    assert records[0]["parameter"] == "quantize.calibration_batches"


def test_quantize_mode_search_filters_static_ptq_for_fixed_torchao_backend(monkeypatch):
    """TorchAO validates only weight_only_ptq, so AutoML should not suggest static_ptq."""
    from tao_automl.search_space import params

    def fake_generate_schema(network, action):
        return {
            "default": {
                "quantize": {
                    "backend": "torchao",
                    "mode": "weight_only_ptq",
                },
            },
            "properties": {
                "quantize": {
                    "type": "object",
                    "properties": {
                        "backend": {
                            "type": "categorical",
                            "default": "torchao",
                            "enum": ["modelopt.pytorch", "torchao", "modelopt.onnx"],
                        },
                        "mode": {
                            "type": "categorical",
                            "default": "weight_only_ptq",
                            "enum": ["static_ptq", "weight_only_ptq"],
                        },
                    },
                },
            },
        }

    monkeypatch.setattr(params, "generate_schema", fake_generate_schema)

    records, names = params.generate_hyperparams_to_search(
        network="fake-net",
        action="quantize",
        train_specs={"quantize": {"backend": "torchao", "mode": "weight_only_ptq"}},
        automl_hyperparameters=["quantize.mode"],
    )

    assert names == ["quantize.mode"]
    assert records[0]["valid_options"] == ["weight_only_ptq"]


def test_quantize_algorithm_search_filters_invalid_modelopt_static_ptq_algorithms(monkeypatch):
    """ModelOpt PyTorch static PTQ should not suggest unregistered algorithms."""
    from tao_automl.search_space import params

    def fake_generate_schema(network, action):
        return {
            "default": {
                "quantize": {
                    "backend": "modelopt.pytorch",
                    "mode": "static_ptq",
                    "algorithm": "max",
                },
            },
            "properties": {
                "quantize": {
                    "type": "object",
                    "properties": {
                        "backend": {
                            "type": "categorical",
                            "default": "modelopt.pytorch",
                            "enum": ["modelopt.pytorch", "torchao", "modelopt.onnx"],
                        },
                        "mode": {
                            "type": "categorical",
                            "default": "static_ptq",
                            "enum": ["static_ptq", "weight_only_ptq"],
                        },
                        "algorithm": {
                            "type": "categorical",
                            "default": "max",
                            "enum": [
                                "minmax",
                                "max",
                                "entropy",
                                "awq_clip",
                                "awq_lite",
                                "awq_full",
                                "rtn_dq",
                            ],
                        },
                    },
                },
            },
        }

    monkeypatch.setattr(params, "generate_schema", fake_generate_schema)

    records, names = params.generate_hyperparams_to_search(
        network="fake-net",
        action="quantize",
        train_specs={
            "quantize": {
                "backend": "modelopt.pytorch",
                "mode": "static_ptq",
                "algorithm": "max",
            }
        },
        automl_hyperparameters=["quantize.algorithm"],
    )

    assert names == ["quantize.algorithm"]
    assert records[0]["valid_options"] == ["max", "awq_lite", "awq_full"]


def test_quantize_mode_and_algorithm_search_filters_for_fixed_torchao_backend(monkeypatch):
    """Searching mode and algorithm together should stay valid for TorchAO."""
    from tao_automl.search_space import params

    def fake_generate_schema(network, action):
        return {
            "default": {
                "quantize": {
                    "backend": "torchao",
                    "mode": "weight_only_ptq",
                    "algorithm": "minmax",
                },
            },
            "properties": {
                "quantize": {
                    "type": "object",
                    "properties": {
                        "backend": {
                            "type": "categorical",
                            "default": "torchao",
                            "enum": ["modelopt.pytorch", "torchao", "modelopt.onnx"],
                        },
                        "mode": {
                            "type": "categorical",
                            "default": "weight_only_ptq",
                            "enum": ["static_ptq", "weight_only_ptq"],
                        },
                        "algorithm": {
                            "type": "categorical",
                            "default": "minmax",
                            "enum": [
                                "minmax",
                                "max",
                                "entropy",
                                "awq_clip",
                                "awq_lite",
                                "awq_full",
                                "rtn_dq",
                            ],
                        },
                    },
                },
            },
        }

    monkeypatch.setattr(params, "generate_schema", fake_generate_schema)

    records, names = params.generate_hyperparams_to_search(
        network="fake-net",
        action="quantize",
        train_specs={
            "quantize": {
                "backend": "torchao",
                "mode": "weight_only_ptq",
                "algorithm": "minmax",
            }
        },
        automl_hyperparameters=["quantize.mode", "quantize.algorithm"],
    )

    ranges = {record["parameter"]: record["valid_options"] for record in records}
    assert names == ["quantize.mode", "quantize.algorithm"]
    assert ranges["quantize.mode"] == ["weight_only_ptq"]
    assert ranges["quantize.algorithm"] == ["minmax"]


def test_quantize_mode_and_algorithm_search_filters_for_fixed_modelopt_backend(monkeypatch):
    """Searching mode and algorithm together should stay valid for ModelOpt PyTorch."""
    from tao_automl.search_space import params

    def fake_generate_schema(network, action):
        return {
            "default": {
                "quantize": {
                    "backend": "modelopt.pytorch",
                    "mode": "static_ptq",
                    "algorithm": "max",
                },
            },
            "properties": {
                "quantize": {
                    "type": "object",
                    "properties": {
                        "backend": {
                            "type": "categorical",
                            "default": "modelopt.pytorch",
                            "enum": ["modelopt.pytorch", "torchao", "modelopt.onnx"],
                        },
                        "mode": {
                            "type": "categorical",
                            "default": "static_ptq",
                            "enum": ["static_ptq", "weight_only_ptq"],
                        },
                        "algorithm": {
                            "type": "categorical",
                            "default": "max",
                            "enum": [
                                "minmax",
                                "max",
                                "entropy",
                                "awq_clip",
                                "awq_lite",
                                "awq_full",
                                "rtn_dq",
                            ],
                        },
                    },
                },
            },
        }

    monkeypatch.setattr(params, "generate_schema", fake_generate_schema)

    records, names = params.generate_hyperparams_to_search(
        network="fake-net",
        action="quantize",
        train_specs={
            "quantize": {
                "backend": "modelopt.pytorch",
                "mode": "static_ptq",
                "algorithm": "max",
            }
        },
        automl_hyperparameters=["quantize.mode", "quantize.algorithm"],
    )

    ranges = {record["parameter"]: record["valid_options"] for record in records}
    assert names == ["quantize.mode", "quantize.algorithm"]
    assert ranges["quantize.mode"] == ["static_ptq"]
    assert ranges["quantize.algorithm"] == ["max", "awq_lite", "awq_full"]


def test_non_train_action_defaults_include_action_native_params():
    from tao_automl.schema.generate_schema import generate_schema
    from tao_automl.search_space import params

    distill_schema = generate_schema("classification_pyt", "distill")
    _distill_records, distill_names = params.generate_hyperparams_to_search(
        network="classification_pyt",
        action="distill",
        train_specs=distill_schema["default"],
        automl_hyperparameters=None,
    )
    assert {
        "distill.loss_type",
        "distill.loss_lambda",
        "distill.mode",
        "distill.use_mlp",
        "distill.mlp_hidden_size",
        "distill.mlp_num_inner",
    }.issubset(set(distill_names))

    prune_schema = generate_schema("ocrnet", "prune")
    _prune_records, prune_names = params.generate_hyperparams_to_search(
        network="ocrnet",
        action="prune",
        train_specs=prune_schema["default"],
        automl_hyperparameters=None,
    )
    assert {
        "prune.prune_setting.mode",
        "prune.prune_setting.amount",
        "prune.prune_setting.granularity",
        "prune.prune_setting.raw_prune_score",
    }.issubset(set(prune_names))

    quantize_schema = generate_schema("classification_pyt", "quantize")
    quantize_spec = quantize_schema["default"]
    quantize_spec["quantize"]["backend"] = "torchao"
    quantize_spec["quantize"]["mode"] = "weight_only_ptq"
    _quantize_records, quantize_names = params.generate_hyperparams_to_search(
        network="classification_pyt",
        action="quantize",
        train_specs=quantize_spec,
        automl_hyperparameters=None,
    )
    assert "quantize.mode" in quantize_names
    assert "quantize.algorithm" in quantize_names
    assert "quantize.backend" not in quantize_names


def test_custom_valid_options_cannot_reopen_schema_excluded_options():
    from tao_automl.utils.math_utils import get_valid_options

    options = get_valid_options(
        {
            "parameter": "quantize.mode",
            "valid_options": ["weight_only_ptq"],
        },
        {
            "quantize.mode": {
                "valid_options": ["weight_only_ptq", "static_ptq"],
            }
        },
    )

    assert options == ["weight_only_ptq"]


def test_custom_valid_options_cannot_reopen_invalid_modelopt_algorithms(monkeypatch, tmp_path):
    import json

    from tao_automl import AutoML
    from tao_automl.search_space import params

    def fake_generate_hyperparams_to_search(**kwargs):
        return [
            {
                "parameter": "quantize.algorithm",
                "value_type": "categorical",
                "default_value": "max",
                "valid_min": "",
                "valid_max": "",
                "valid_options": ["max", "awq_lite", "awq_full"],
                "option_weights": None,
                "math_cond": "",
                "parent_param": "",
                "depends_on": "",
            }
        ], ["quantize.algorithm"]

    monkeypatch.setattr(
        params,
        "generate_hyperparams_to_search",
        fake_generate_hyperparams_to_search,
    )

    AutoML(
        workspace=str(tmp_path),
        network="classification_pyt",
        train_specs={
            "quantize": {
                "backend": "modelopt.pytorch",
                "mode": "static_ptq",
                "algorithm": "max",
            }
        },
        settings={
            "algorithm": "bayesian",
            "metric": "val_acc_1",
            "direction": "maximize",
            "automl_max_recommendations": 1,
            "session_id": "fixedsession",
        },
        automl_hyperparameters=["quantize.algorithm"],
        custom_param_ranges={
            "quantize.algorithm": {
                "valid_options": ["minmax", "max", "entropy", "awq_lite"],
            }
        },
        action="quantize",
    )

    ranges = json.loads(
        (tmp_path / ".automl/custom_ranges/fixedsession.json").read_text()
    )
    assert ranges["quantize.algorithm"]["valid_options"] == ["max", "awq_lite"]


def test_automl_persists_sanitized_custom_valid_options(monkeypatch, tmp_path):
    import json

    from tao_automl import AutoML
    from tao_automl.search_space import params

    def fake_generate_hyperparams_to_search(**kwargs):
        return [
            {
                "parameter": "quantize.mode",
                "value_type": "categorical",
                "default_value": "weight_only_ptq",
                "valid_min": "",
                "valid_max": "",
                "valid_options": ["weight_only_ptq"],
                "option_weights": None,
                "math_cond": "",
                "parent_param": "",
                "depends_on": "",
            }
        ], ["quantize.mode"]

    monkeypatch.setattr(
        params,
        "generate_hyperparams_to_search",
        fake_generate_hyperparams_to_search,
    )

    AutoML(
        workspace=str(tmp_path),
        network="classification_pyt",
        train_specs={"quantize": {"backend": "torchao", "mode": "weight_only_ptq"}},
        settings={
            "algorithm": "bayesian",
            "metric": "val_acc_1",
            "direction": "maximize",
            "automl_max_recommendations": 1,
            "session_id": "fixedsession",
        },
        automl_hyperparameters=["quantize.mode"],
        custom_param_ranges={
            "quantize.mode": {
                "valid_options": ["weight_only_ptq", "static_ptq"],
            }
        },
        action="quantize",
    )

    ranges = json.loads(
        (tmp_path / ".automl/custom_ranges/fixedsession.json").read_text()
    )
    assert ranges["quantize.mode"]["valid_options"] == ["weight_only_ptq"]


def test_skill_context_resolves_python_script_execution_and_external_schema(tmp_path):
    from tao_automl.runner import SkillContext

    skill_dir = _write_python_skill(tmp_path)
    ctx = SkillContext(skill_dir=skill_dir, action="train")

    assert ctx.container_image == ""
    assert ctx.execution.script == (skill_dir / "scripts/train.py").resolve()
    assert ctx.execution.script_args == ("--config", "{config_path}")
    assert ctx.execution.config_format == "json"
    assert ctx.execution.cwd == skill_dir.resolve()
    assert ctx.schema["default"]["model"]["n_estimators"] == 10


def test_skill_context_rejects_missing_python_script(tmp_path):
    from tao_automl.runner import SkillContext

    skill_dir = _write_python_skill(tmp_path)
    (skill_dir / "scripts/train.py").unlink()

    with pytest.raises(FileNotFoundError, match="Python action script not found"):
        SkillContext(skill_dir=skill_dir, action="train")


def test_skill_context_requires_external_schema_for_python_script(tmp_path):
    from tao_automl.runner import SkillContext

    skill_dir = _write_python_skill(tmp_path)
    (skill_dir / "schemas/train.schema.json").unlink()

    with pytest.raises(FileNotFoundError, match="require an external AutoML schema"):
        SkillContext(skill_dir=skill_dir, action="train")


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ({}, "non-empty 'properties'"),
        ({"type": "object", "default": [], "properties": {"x": {"type": "integer"}}},
         "'default' must be a JSON object"),
        ({"type": "object", "properties": {"model": {"type": "object", "properties": {}}}},
         "requires non-empty nested properties"),
        ({"type": "object", "properties": {"x": {"anyOf": True}}},
         "'anyOf' must be a non-empty list"),
        ({"type": "object", "properties": {"x": {"type": ["integer", "null"]}}},
         "list-valued 'type'"),
        ({"type": "object", "properties": {"x": {
            "type": "integer", "default": True, "minimum": 1,
            "maximum": 2, "automl_enabled": True,
        }}}, "expected an integer"),
        ({"type": "object", "properties": {"x": {
            "type": "number", "default": 1.0, "minimum": 2.0,
            "maximum": 1.0, "automl_enabled": True,
        }}}, "minimum cannot exceed maximum"),
        ({"type": "object", "properties": {"x": {
            "type": "string", "default": "x", "automl_enabled": True,
        }}}, "requires an enum"),
        ({"type": "object", "properties": {"x": {
            "anyOf": [{"type": "integer"}, {"type": "number"}],
            "automl_enabled": True,
        }}}, "exactly one non-null anyOf type"),
    ],
)
def test_skill_context_rejects_malformed_python_search_schema(
    tmp_path, schema, message,
):
    from tao_automl.runner import SkillContext

    skill_dir = _write_python_skill(tmp_path)
    (skill_dir / "schemas/train.schema.json").write_text(json.dumps(schema))

    with pytest.raises((TypeError, ValueError), match=message):
        SkillContext(skill_dir=skill_dir, action="train")


def test_optional_anyof_schema_uses_non_null_search_type(tmp_path):
    from tao_automl.runner import SkillContext
    from tao_automl.search_space.params import generate_hyperparams_to_search

    skill_dir = _write_python_skill(tmp_path)
    schema_path = skill_dir / "schemas/train.schema.json"
    schema = json.loads(schema_path.read_text())
    n_estimators = schema["properties"]["model"]["properties"]["n_estimators"]
    n_estimators.pop("type")
    n_estimators["anyOf"] = [{"type": "null"}, {"type": "integer"}]
    n_estimators["properties"] = {}
    schema_path.write_text(json.dumps(schema))

    ctx = SkillContext(skill_dir=skill_dir, action="train")
    _, names = generate_hyperparams_to_search(
        ctx.network_arch,
        "train",
        ctx.default_specs,
        ["model.n_estimators"],
        schema=ctx.schema,
    )

    assert names == ["model.n_estimators"]


def test_optional_anyof_schema_uses_branch_search_metadata(tmp_path):
    from tao_automl.runner import SkillContext
    from tao_automl.search_space.params import generate_hyperparams_to_search

    skill_dir = _write_python_skill(tmp_path)
    schema_path = skill_dir / "schemas/train.schema.json"
    schema = json.loads(schema_path.read_text())
    parameter = schema["properties"]["model"]["properties"]["n_estimators"]
    parameter.clear()
    parameter["anyOf"] = [
        {"type": "null"},
        {
            "type": "integer",
            "default": 10,
            "minimum": 2,
            "maximum": 20,
            "automl_enabled": True,
        },
    ]
    schema_path.write_text(json.dumps(schema))

    ctx = SkillContext(skill_dir=skill_dir, action="train")
    records, names = generate_hyperparams_to_search(
        ctx.network_arch,
        "train",
        ctx.default_specs,
        [],
        schema=ctx.schema,
    )

    assert "model.n_estimators" in names
    record = next(r for r in records if r["parameter"] == "model.n_estimators")
    assert record["default_value"] == 10
    assert record["valid_min"] == 2
    assert record["valid_max"] == 20


def test_python_script_run_rejects_invalid_merged_spec_override(tmp_path):
    from tao_automl.runner import AutoMLRunner

    runner = AutoMLRunner(
        sdk=MagicMock(), skill_dir=_write_python_skill(tmp_path), action="train"
    )

    with pytest.raises(TypeError, match="expected an integer"):
        runner.run(
            automl_settings={"algorithm": "bayesian", "metric": "accuracy"},
            automl_hyperparameters=["model.n_estimators"],
            spec_overrides={"model.n_estimators": "many"},
            workspace_path=str(tmp_path / "workspace"),
        )


def test_runtime_python_execution_override_requires_external_schema(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    script = skill_dir / "train.py"
    script.write_text("print('accuracy: 1.0')\n")
    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")

    with pytest.raises(FileNotFoundError, match="external AutoML schema"):
        runner.run(
            execution={"type": "python_script", "script": str(script)},
            workspace_path=str(tmp_path / "workspace"),
        )


def test_external_schema_defaults_are_used_when_template_is_absent(tmp_path):
    from tao_automl.runner import SkillContext

    skill_dir = _write_python_skill(tmp_path)
    (skill_dir / "references/spec_template_train.yaml").unlink()

    ctx = SkillContext(skill_dir=skill_dir, action="train")
    assert ctx.default_specs["model"] == {"n_estimators": 10, "max_depth": 3}


def test_python_script_action_does_not_resolve_unused_container_image(tmp_path):
    from tao_automl.runner import SkillContext

    skill_dir = _write_python_skill(tmp_path)
    info_path = skill_dir / "references/skill_info.yaml"
    info_path.write_text(
        "container_image: key.that.does.not.exist\n" + info_path.read_text()
    )

    ctx = SkillContext(skill_dir=skill_dir, action="train")
    assert ctx.container_image == ""


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def test_extract_metric_allows_val_prefix_for_sparse4d_map():
    from tao_automl.runner import _extract_metric_from_logs
    logs = "Calculating metrics...\nmAP: 0.0000\nNDS: 0.0000\nExecution status: PASS\n"
    assert _extract_metric_from_logs(logs, "val_mAP") == 0.0


def test_extract_metric_supports_signed_values_and_exact_aliases():
    from tao_automl.runner import _extract_metric_from_logs

    logs = (
        "val_loss: 0.9\n"
        "loss_scale: 1024\n"
        "loss: -2.5e-1\n"
        "val_loss: 0.1\n"
    )

    assert _extract_metric_from_logs(logs, "loss") == pytest.approx(-0.25)
    assert _extract_metric_from_logs("val_loss: 0.2\n", "loss") is None
    assert _extract_metric_from_logs("loss: 0.2\n", "val_loss") is None


def test_cosmos_validation_loss_competes_by_log_position():
    from tao_automl.runner import _extract_metric_from_logs

    logs = "[SFT] Validation loss: 0.9\nval_loss: 0.2\n"

    assert _extract_metric_from_logs(logs, "val_loss") == pytest.approx(0.2)


def test_cosmos_validation_loss_does_not_satisfy_other_validation_metrics():
    from tao_automl.runner import _extract_metric_from_logs

    logs = "[SFT] Validation loss: 0.9\n"

    assert _extract_metric_from_logs(logs, "val_accuracy") is None


def test_extract_metric_uses_globally_latest_matching_format():
    from tao_automl.runner import _extract_metric_from_logs

    logs = "accuracy: 0.8\nnoise\nkpi: -1.25e-2\n"

    assert _extract_metric_from_logs(logs, "accuracy") == pytest.approx(-0.0125)


def test_extract_metric_reads_sparse4d_status_kpi_alias(tmp_path):
    from tao_automl.runner import _extract_metric_from_status_file

    status_path = tmp_path / "status.json"
    status_path.write_text(
        '{"status": "RUNNING", "kpi": {"img_bbox_NuScenes/mAP": 0.125}}\n'
    )

    assert _extract_metric_from_status_file(status_path, "val_mAP") == 0.125


def test_status_metric_ignores_non_objects_and_nonfinite_values(tmp_path):
    from tao_automl.runner import _extract_metric_from_status_file

    status_path = tmp_path / "status.json"
    status_path.write_text(
        '{"kpi": {"accuracy": 0.75}}\n'
        '[]\n'
        '{"kpi": {"accuracy": true}}\n'
        '{"kpi": {"accuracy": Infinity}}\n'
    )

    assert _extract_metric_from_status_file(status_path, "accuracy") == 0.75


@pytest.mark.parametrize("bad_metric", [True, float("nan"), float("inf"), -float("inf")])
def test_structured_metric_payloads_reject_boolean_and_nonfinite_values(bad_metric):
    from tao_automl.runner import (
        _extract_metric_from_best_score_payload,
        _extract_metric_from_metrics_payload,
        _merge_metric_payload,
    )

    assert _extract_metric_from_best_score_payload(
        {"best_score": bad_metric}, "accuracy"
    ) is None
    assert _extract_metric_from_metrics_payload(
        {"accuracy": bad_metric}, "accuracy"
    ) is None
    target = {}
    assert not _merge_metric_payload(target, {"metric_value": bad_metric})
    assert "metric_value" not in target


def test_multi_objective_callback_payload_requires_all_finite_values():
    from tao_automl.runner import _callback_metric_payload

    assert _callback_metric_payload(
        {"accuracy": 0.8, "latency": 12},
        "eval_fn",
    ) == {"accuracy": 0.8, "latency": 12.0}
    assert _callback_metric_payload(
        {"accuracy": 0.8, "latency": True},
        "eval_fn",
    ) is None
    assert _callback_metric_payload(
        {"accuracy": 0.8, "latency": float("nan")},
        "eval_fn",
    ) is None


def test_generic_log_metric_only_satisfies_primary_objective():
    from tao_automl.runner import _extract_metric_from_logs, _extract_metric_values

    assert _extract_metric_values(
        "kpi: 0.8\n",
        ["accuracy", "latency"],
        _extract_metric_from_logs,
    ) == {"accuracy": 0.8}


def test_extract_latency_aliases_from_logs_and_status(tmp_path):
    from tao_automl.runner import _extract_metric_from_logs, _extract_metric_from_status_file

    logs = "val_mAP: 0.812\ninference_latency_ms: 14.5\n"
    assert _extract_metric_from_logs(logs, "latency") == pytest.approx(14.5)

    status_path = tmp_path / "status.json"
    status_path.write_text(
        '{"status": "RUNNING", "kpi": {"avg_latency_ms": 12.25}}\n'
    )
    assert _extract_metric_from_status_file(status_path, "latency") == pytest.approx(12.25)


def test_latency_does_not_fall_back_to_primary_best_score():
    from tao_automl.runner import _extract_metric_from_best_score_payload

    payload = '{"best_score": 0.93, "metric": "val_mAP"}\n'
    assert _extract_metric_from_best_score_payload(payload, "latency") is None

    payload_with_latency = (
        '{"best_score": 0.93, "metric": "val_mAP", "latency_ms": 18.0}\n'
    )
    assert _extract_metric_from_best_score_payload(
        payload_with_latency, "latency"
    ) == pytest.approx(18.0)
    assert _extract_metric_from_best_score_payload(
        {"metric": "val_mAP", "val_mAP": 0.93},
        "latency",
    ) is None
    assert _extract_metric_from_best_score_payload(
        {"best_score": 0.93},
        "latency",
        allow_generic=False,
    ) is None


def test_execution_status_can_ignore_fatal_cleanup_patterns():
    from tao_automl.runner import _check_execution_status

    logs = "Saved best score to best_score.json\nRendezvousConnectionError\n"

    assert _check_execution_status(logs) == "FAIL"
    assert _check_execution_status(logs, include_fatal_patterns=False) is None


def test_execution_status_detects_nccl_watchdog_as_hard_failure():
    from tao_automl.runner import _check_execution_status, _has_hard_failure_pattern

    logs = (
        "Watchdog caught collective operation timeout: "
        "WorkNCCL(SeqNum=33271, OpType=ALLREDUCE)\n"
    )

    assert _check_execution_status(logs) == "FAIL"
    assert _has_hard_failure_pattern(logs)


@pytest.mark.parametrize(
    "raw_status,expected",
    (
        ("Completed", "Complete"),
        ("Succeeded", "Complete"),
        ("Failed", "Error"),
        ("Failure", "Error"),
        ("Cancelled", "Canceled"),
        ("Stopped", "Canceled"),
        ("NotFound", "Canceled"),
    ),
)
def test_confirmed_platform_status_returns_canonical_category(
    raw_status, expected,
):
    from tao_automl.runner import _confirmed_platform_status

    assert _confirmed_platform_status(SimpleNamespace(status=raw_status)) == expected


def test_unknown_job_status_is_not_proof_that_backend_writer_stopped():
    from tao_automl.runner import _confirmed_platform_status

    status = SimpleNamespace(status="Unknown", message="Job not found")
    assert _confirmed_platform_status(status) is None


def test_extract_metric_reads_cosmos_best_score_json(tmp_path):
    from tao_automl.runner import _extract_metric_from_local_results

    best_score = (
        tmp_path / "results" / "job-1" / "train_output_dir" / "best"
        / "best_score.json"
    )
    best_score.parent.mkdir(parents=True)
    best_score.write_text(
        '{"best_score": 0.8927091135965706, "metric": "val_loss"}\n'
    )

    metric = _extract_metric_from_local_results(
        "job-1",
        "val/avg_loss",
        {"mounts": [{"host_path": str(tmp_path / "results"),
                     "container_path": "/results"}]},
    )

    assert metric == pytest.approx(0.8927091135965706)


def test_extract_metric_reads_direct_script_metrics_json(tmp_path):
    from tao_automl.runner import _extract_metric_from_sdk_results

    results_dir = tmp_path / "job-results"
    metrics_path = results_dir / "results_dir" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text('{"accuracy": 0.9375, "score": 0.4}\n')

    class ResultsSDK:
        def get_job_results_dir(self, job_id):
            return str(results_dir)

    assert _extract_metric_from_sdk_results(
        ResultsSDK(), "job-1", "accuracy"
    ) == pytest.approx(0.9375)


def test_streamed_terminal_scan_uses_latest_explicit_status_marker():
    from tao_automl.runner import _extract_metric_from_logs, _scan_terminal_logs

    class StreamingSDK:
        def iter_job_log_chunks(self, job_id, chunk_size=256 * 1024):
            yield "Execution status: FAIL\n"
            yield "noise\n" * 2_000
            yield "accuracy: 0.9\nExecution status: PASS\n"

    metric, status, _ = _scan_terminal_logs(
        StreamingSDK(), "job-1", "accuracy", _extract_metric_from_logs, None, None
    )

    assert metric == pytest.approx(0.9)
    assert status == "PASS"


def test_non_streaming_sdk_retains_full_terminal_log_compatibility():
    from tao_automl.runner import _extract_metric_from_logs, _scan_terminal_logs

    full_logs = "accuracy: 0.875\n" + "noise\n" * 10_001

    class LegacySDK:
        def __init__(self):
            self.requested_tails = []

        def get_job_logs(self, job_id, tail=None):
            self.requested_tails.append(tail)
            if tail is None:
                return full_logs
            return "".join(full_logs.splitlines(keepends=True)[-tail:])

    sdk = LegacySDK()
    metric, status, _ = _scan_terminal_logs(
        sdk, "job-1", "accuracy", _extract_metric_from_logs, None, None
    )

    assert metric == pytest.approx(0.875)
    assert status is None
    assert sdk.requested_tails == [None]


def test_non_streaming_terminal_scan_finds_early_fail_with_cached_metric():
    from tao_automl.runner import _extract_metric_from_logs, _scan_terminal_logs

    full_logs = (
        "Execution status: FAIL\n"
        + "noise\n" * 10_001
        + "accuracy: 0.9\n"
    )

    class LegacySDK:
        def get_job_logs(self, job_id, tail=None):
            assert tail is None
            return full_logs

    metric, status, _ = _scan_terminal_logs(
        LegacySDK(),
        "job-1",
        "accuracy",
        _extract_metric_from_logs,
        0.9,
        None,
    )

    assert metric == pytest.approx(0.9)
    assert status == "FAIL"


def test_llm_config_accepts_provider_aliases():
    from tao_automl.brain.llm_client import LLMConfig

    config = LLMConfig.from_params({
        "base_url": "https://inference-api.nvidia.com",
        "model": "gcp/google/gemini-3.1-pro-preview",
        "api_key": "secret",
    })

    assert config.endpoint == "https://inference-api.nvidia.com"
    assert config.model == "gcp/google/gemini-3.1-pro-preview"
    assert config.api_key == "secret"


def test_algorithm_params_pass_provider_aliases_to_llm_client():
    from tao_automl.brain.factory import AlgorithmParams

    params = AlgorithmParams.from_dict({
        "base_url": "https://inference-api.nvidia.com",
        "model": "gcp/google/gemini-3.1-pro-preview",
        "api_key": "secret",
    })

    assert params.get_llm_params() == {
        "llm_endpoint": "https://inference-api.nvidia.com",
        "llm_model": "gcp/google/gemini-3.1-pro-preview",
        "llm_api_key": "secret",
    }


def test_algorithm_params_parse_hybrid_range_narrowing_flag():
    from tao_automl.brain.factory import AlgorithmParams

    assert AlgorithmParams.from_dict({
        "hybrid_enable_llm_range_narrowing": "true",
    }).hybrid_enable_llm_range_narrowing
    assert not AlgorithmParams.from_dict({}).hybrid_enable_llm_range_narrowing


def test_algorithm_params_parse_intermediate_checkpoint_cleanup_flag():
    from tao_automl.brain.factory import AlgorithmParams

    assert AlgorithmParams.from_dict({
        "automl_delete_intermediate_ckpt": "true",
    }).automl_delete_intermediate_ckpt
    assert not AlgorithmParams.from_dict({
        "automl_delete_intermediate_ckpt": "false",
    }).automl_delete_intermediate_ckpt
    assert AlgorithmParams.from_dict({}).automl_delete_intermediate_ckpt


def _install_sequential_fake_automl(monkeypatch, count=3):
    from tao_automl.types import Recommendation

    reports = []

    class FakeAutoML:
        def __init__(self, *args, **kwargs):
            self.recs = [
                Recommendation(i, {"train.num_epochs": i + 1}, "accuracy")
                for i in range(count)
            ]
            self.index = 0

        def is_complete(self):
            return self.index >= count

        def next_recommendation(self):
            return [self.recs[self.index]]

        def report_result(self, rec_id, metric_value, best_epoch=None, status="success"):
            rec = self.recs[self.index]
            rec.update_result(metric_value)
            rec.update_status(status)
            reports.append((rec_id, metric_value, status))
            self.index += 1

        def get_best(self):
            return None

        def get_progress(self):
            return {"completed": self.index, "best_metric": None}

        def get_history(self):
            return self.recs

    monkeypatch.setattr("tao_automl.AutoML", FakeAutoML)
    return reports


def test_job_creation_failures_do_not_count_as_missing_metrics(tmp_path, monkeypatch):
    from tao_automl.runner import AutoMLRunner

    reports = _install_sequential_fake_automl(monkeypatch)
    monkeypatch.setattr(
        AutoMLRunner, "_run_one_job", lambda self, *args, **kwargs: (None, "failure")
    )
    runner = AutoMLRunner(
        sdk=MagicMock(), skill_dir=_write_fake_skill(tmp_path), action="train"
    )

    with pytest.raises(RuntimeError, match="without a successful recommendation"):
        runner.run(
            image="nvcr.io/test:1",
            automl_settings={
                "algorithm": "bayesian",
                "metric": "accuracy",
                "run_baseline": False,
            },
            workspace_path=str(tmp_path / "workspace"),
        )

    assert len(reports) == 3
    assert {status for _, _, status in reports} == {"failure"}
    assert runner._consecutive_none_metrics == 0


def test_metric_missing_fail_fast_reports_third_failure_before_raising(
    tmp_path, monkeypatch,
):
    from tao_automl.runner import AutoMLRunner, MetricExtractorError

    reports = _install_sequential_fake_automl(monkeypatch)
    callbacks = []
    monkeypatch.setattr(
        AutoMLRunner,
        "_run_one_job",
        lambda self, *args, **kwargs: (None, "metric_missing"),
    )
    runner = AutoMLRunner(
        sdk=MagicMock(), skill_dir=_write_fake_skill(tmp_path), action="train"
    )

    with pytest.raises(MetricExtractorError, match="3 consecutive recs"):
        runner.run(
            image="nvcr.io/test:1",
            automl_settings={
                "algorithm": "bayesian",
                "metric": "accuracy",
                "run_baseline": False,
            },
            on_result=lambda rec, metric, status: callbacks.append(
                (rec.id, metric, status)
            ),
            workspace_path=str(tmp_path / "workspace"),
        )

    assert len(reports) == 3
    assert len(callbacks) == 3
    assert reports[-1] == (2, 0.0, "failure")
    assert callbacks[-1] == (2, None, "failure")


def test_promoted_metric_missing_checkpoint_carries_forward_prior_metric(
    tmp_path, monkeypatch
):
    from tao_automl.runner import AutoMLRunner
    from tao_automl.types import JobStates, Recommendation

    skill_dir = _write_fake_skill(tmp_path)
    results_root = tmp_path / "results"
    for job_id in ("parent-job", "child-job"):
        ckpt_dir = results_root / job_id / "results_dir" / "train"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "model_latest.pth").write_text("checkpoint")

    class FakeAutoML:
        def __init__(self, *args, **kwargs):
            self.rec = Recommendation(0, {"train.num_epochs": 2}, "val_mAP")
            self.rec.resume_from_job_id = "parent-job"
            self.rec.result = 0.42
            self.complete = False

        def is_complete(self):
            return self.complete

        def next_recommendation(self):
            return [self.rec]

        def report_result(self, rec_id, metric_value, best_epoch=None, status="success"):
            self.rec.update_result(metric_value)
            self.rec.update_status(status)
            self.complete = True

        def get_best(self):
            return self.rec if self.rec.status == JobStates.success else None

        def get_progress(self):
            return {"completed": int(self.complete), "best_metric": self.rec.result}

        def get_history(self):
            return [self.rec]

    def fake_run_one_job(self, *args, **kwargs):
        kwargs["rec"].assign_job_id("child-job")
        return None, "metric_missing"

    monkeypatch.setattr("tao_automl.AutoML", FakeAutoML)
    monkeypatch.setattr(AutoMLRunner, "_run_one_job", fake_run_one_job)

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    result = runner.run(
        image="nvcr.io/test:1",
        automl_settings={
            "algorithm": "dehb",
            "metric": "val_mAP",
            "direction": "maximize",
        },
        automl_hyperparameters=["train.optim.lr"],
        workspace_path=str(tmp_path / "workspace"),
        mounts=[{"host_path": str(results_root), "container_path": "/results"}],
    )

    assert result["best"]["metric_value"] == 0.42
    assert result["history"][0]["status"] == JobStates.success


def test_run_reports_baseline_metric_and_comparison(tmp_path, monkeypatch):
    from tao_automl.runner import AutoMLRunner
    from tao_automl.types import JobStates, Recommendation

    skill_dir = _write_fake_skill(tmp_path)

    class FakeAutoML:
        def __init__(self, *args, **kwargs):
            self.rec = Recommendation(0, {"train.num_epochs": 2}, "accuracy")
            self.complete = False

        def is_complete(self):
            return self.complete

        def next_recommendation(self):
            return [self.rec]

        def report_result(self, rec_id, metric_value, best_epoch=None, status="success"):
            self.rec.update_result(metric_value)
            self.rec.update_status(status)
            self.complete = True

        def get_best(self):
            return self.rec if self.rec.status == JobStates.success else None

        def get_progress(self):
            return {"completed": int(self.complete), "best_metric": self.rec.result}

        def get_history(self):
            return [self.rec]

    def fake_run_one_job(self, *args, **kwargs):
        return 0.62, "success"

    monkeypatch.setattr("tao_automl.AutoML", FakeAutoML)
    monkeypatch.setattr(AutoMLRunner, "_run_one_job", fake_run_one_job)

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    result = runner.run(
        image="nvcr.io/test:1",
        automl_settings={
            "algorithm": "bayesian",
            "metric": "accuracy",
            "direction": "maximize",
            "automl_max_recommendations": 1,
        },
        baseline_fn=lambda specs: 0.5,
        workspace_path=str(tmp_path / "workspace"),
    )

    assert result["baseline"]["status"] == "measured"
    assert result["baseline"]["metric_value"] == 0.5
    assert result["baseline"]["comparison_to_best"]["delta"] == pytest.approx(0.12)
    assert result["baseline"]["comparison_to_best"]["improved"] is True


def test_run_preserves_multi_objective_payload_and_explicit_primary_metric(
    tmp_path, monkeypatch,
):
    from tao_automl.runner import AutoMLRunner
    from tao_automl.types import JobStates, Recommendation

    captured = {}

    class FakeAutoML:
        def __init__(self, *args, **kwargs):
            self.rec = Recommendation(0, {"train.num_epochs": 2}, "accuracy")
            self.complete = False

        def is_complete(self):
            return self.complete

        def next_recommendation(self):
            return [self.rec]

        def report_result(self, rec_id, metric_value, best_epoch=None, status="success"):
            assert metric_value == {"accuracy": 0.8, "latency": 12.0}
            self.rec.update_objectives(metric_value, 0.68)
            self.rec.update_status(status)
            self.complete = True

        def get_best(self):
            return self.rec if self.rec.status == JobStates.success else None

        def get_progress(self):
            return {"completed": int(self.complete), "best_metric": 0.8}

        def get_history(self):
            return [self.rec]

        def get_status(self):
            return {"pareto_front": [{"rec_id": self.rec.id}]}

    def fake_run_one_job(self, *args, **kwargs):
        captured["metric_name"] = kwargs["metric_name"]
        captured["objective_names"] = kwargs["objective_names"]
        return {"accuracy": 0.8, "latency": 12}, "success"

    monkeypatch.setattr("tao_automl.AutoML", FakeAutoML)
    monkeypatch.setattr(AutoMLRunner, "_run_one_job", fake_run_one_job)

    runner = AutoMLRunner(
        sdk=MagicMock(),
        skill_dir=_write_fake_skill(tmp_path),
        action="train",
    )
    result = runner.run(
        image="nvcr.io/test:1",
        automl_settings={
            "algorithm": "bayesian",
            "objectives": [
                {"metric": "accuracy", "direction": "maximize"},
                {
                    "metric": "latency",
                    "direction": "minimize",
                    "scale": 100,
                },
            ],
            "run_baseline": False,
            "run_final_evaluation": False,
        },
        workspace_path=str(tmp_path / "workspace"),
    )

    assert captured == {
        "metric_name": "accuracy",
        "objective_names": ["accuracy", "latency"],
    }
    assert result["best"]["metric_value"] == pytest.approx(0.8)
    assert result["best"]["objective_score"] == pytest.approx(0.68)
    assert result["best"]["objective_values"] == {
        "accuracy": 0.8,
        "latency": 12.0,
    }
    assert result["pareto_front"] == [{"rec_id": 0}]


def test_run_reports_runner_owned_final_evaluation(tmp_path, monkeypatch):
    from tao_automl.runner import AutoMLRunner
    from tao_automl.types import JobStates, Recommendation

    skill_dir = _write_fake_skill(tmp_path)
    final_eval_calls = []
    final_record_path = tmp_path / "workspace" / "evaluations" / "best_automl.json"

    class FakeAutoML:
        def __init__(self, *args, **kwargs):
            self.rec = Recommendation(0, {"train.num_epochs": 2}, "accuracy")
            self.complete = False

        def is_complete(self):
            return self.complete

        def next_recommendation(self):
            return [self.rec]

        def report_result(self, rec_id, metric_value, best_epoch=None, status="success"):
            self.rec.update_result(metric_value)
            self.rec.update_status(status)
            self.complete = True

        def get_best(self):
            return self.rec if self.rec.status == JobStates.success else None

        def get_progress(self):
            return {"completed": int(self.complete), "best_metric": self.rec.result}

        def get_history(self):
            return [self.rec]

    def fake_run_one_job(self, *args, **kwargs):
        kwargs["rec"].assign_job_id("train-job-0")
        return 0.62, "success"

    def final_eval_fn(best_rec, train_job_id):
        final_eval_calls.append((best_rec.id, train_job_id))
        return {"metric_value": 0.64, "record_path": str(final_record_path)}

    monkeypatch.setattr("tao_automl.AutoML", FakeAutoML)
    monkeypatch.setattr(AutoMLRunner, "_run_one_job", fake_run_one_job)

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    result = runner.run(
        image="nvcr.io/test:1",
        automl_settings={
            "algorithm": "bayesian",
            "metric": "accuracy",
            "direction": "maximize",
            "automl_max_recommendations": 1,
            "run_final_evaluation": True,
        },
        baseline_fn=lambda specs: 0.5,
        final_eval_fn=final_eval_fn,
        workspace_path=str(tmp_path / "workspace"),
    )

    assert final_eval_calls == [(0, "train-job-0")]
    assert result["best"]["metric_value"] == 0.62
    assert result["final_evaluation"]["status"] == "measured"
    assert result["final_evaluation"]["source"] == "final_eval_fn"
    assert result["final_evaluation"]["metric_value"] == 0.64
    assert result["final_evaluation"]["record_path"] == str(final_record_path)
    assert result["final_evaluation"]["comparison_to_baseline"]["delta"] == pytest.approx(0.14)
    assert result["final_evaluation"]["comparison_to_baseline"]["improved"] is True


def test_effective_batch_is_capped_before_launch(tmp_path, monkeypatch):
    from tao_automl.runner import AutoMLRunner
    from tao_automl.types import JobStates, Recommendation

    skill_dir = _write_fake_skill(tmp_path)
    captured_specs = {}

    class FakeAutoML:
        def __init__(self, *args, **kwargs):
            self.rec = Recommendation(
                0,
                {
                    "train.train_batch_per_replica": 8,
                    "policy.parallelism.dp_shard_size": 8,
                },
                "accuracy",
            )
            self.complete = False

        def is_complete(self):
            return self.complete

        def next_recommendation(self):
            return [self.rec]

        def report_result(self, rec_id, metric_value, best_epoch=None, status="success"):
            self.rec.update_result(metric_value)
            self.rec.update_status(status)
            self.complete = True

        def get_best(self):
            return self.rec if self.rec.status == JobStates.success else None

        def get_progress(self):
            return {"completed": int(self.complete), "best_metric": self.rec.result}

        def get_history(self):
            return [self.rec]

    def fake_run_one_job(self, *args, **kwargs):
        captured_specs.update(kwargs["specs"])
        return 0.6, "success"

    monkeypatch.setattr("tao_automl.AutoML", FakeAutoML)
    monkeypatch.setattr(AutoMLRunner, "_run_one_job", fake_run_one_job)

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    result = runner.run(
        image="nvcr.io/test:1",
        automl_settings={
            "algorithm": "bayesian",
            "metric": "accuracy",
            "direction": "maximize",
            "automl_max_recommendations": 1,
            "train_sample_count": 31,
        },
        workspace_path=str(tmp_path / "workspace"),
    )

    assert captured_specs["train"]["train_batch_per_replica"] == 3
    assert result["history"][0]["adjustments"][0]["type"] == "effective_batch_cap"
    assert result["history"][0]["adjustments"][0]["from"] == 8
    assert result["history"][0]["adjustments"][0]["to"] == 3


def test_effective_batch_reports_invalid_when_no_rank_has_samples():
    from tao_automl.runner import _maybe_cap_effective_batch
    from tao_automl.types import Recommendation

    rec = Recommendation(
        0,
        {
            "train.train_batch_per_replica": 8,
            "policy.parallelism.dp_shard_size": 8,
        },
        "loss",
    )
    specs = {
        "train": {"train_batch_per_replica": 8},
        "policy": {"parallelism": {"dp_shard_size": 8}},
    }

    reason = _maybe_cap_effective_batch(
        specs, rec, {"train_sample_count": 2}, {}
    )

    assert "invalid_configuration" in reason
    assert rec.failure_reason == reason
    assert specs["train"]["train_batch_per_replica"] == 8


def test_validate_skill_runtime_probes_schema_import_path(tmp_path):
    from tao_automl.runner import validate_skill_runtime

    skill_dir = _write_fake_skill(tmp_path)
    info_path = skill_dir / "references/skill_info.yaml"
    info_path.write_text(
        info_path.read_text().replace("network_arch: fake-net", "network_arch: cosmos-rl")
    )

    result = validate_skill_runtime(skill_dir, action="train")

    assert result["network_arch"] == "cosmos-rl"
    assert result["action"] == "train"
    assert result["parameter_count"] > 0


# ---------------------------------------------------------------------------
# _make_sdk — platform selection
# ---------------------------------------------------------------------------

def test_make_sdk_rejects_unknown_platform():
    from tao_automl.runner import _make_sdk
    with pytest.raises(ValueError, match="Unknown platform"):
        _make_sdk("aws-batch")


def test_make_sdk_lists_all_platforms_in_error():
    from tao_automl.runner import _make_sdk, _PLATFORMS
    assert set(_PLATFORMS) == {
        "lepton", "slurm", "kubernetes", "docker", "brev", "virtualenv",
    }
    try:
        _make_sdk("nope")
    except ValueError as e:
        for p in _PLATFORMS:
            assert p in str(e)


def test_make_sdk_constructs_virtualenv_with_sdk_kwargs(tmp_path):
    from tao_automl.runner import _make_sdk

    with patch("tao_sdk.platforms.virtualenv.VirtualEnvSDK") as sdk_cls:
        instance = _make_sdk(
            "virtualenv",
            venv_path=str(tmp_path / "venv"),
            work_dir=str(tmp_path / "jobs"),
        )

    assert instance is sdk_cls.return_value
    sdk_cls.assert_called_once_with(
        venv_path=str(tmp_path / "venv"),
        work_dir=str(tmp_path / "jobs"),
    )


# ---------------------------------------------------------------------------
# AutoMLRunner — submission path uses build_entrypoint + new create_job shape
# ---------------------------------------------------------------------------

def test_run_one_job_calls_build_entrypoint_with_action_cfg(tmp_path):
    """_run_one_job should pass the action's command/inputs/outputs/config_format/
    upload_excludes to build_entrypoint, and the resulting command string to
    sdk.create_job. No old kwargs (specs=, script_runner=, network_arch=)."""
    from tao_automl.runner import AutoMLRunner, _POLL_LOG_TAIL_LINES

    skill_dir = _write_fake_skill(tmp_path)
    fake_sdk = MagicMock()
    fake_sdk.create_job.return_value = MagicMock(id="job-xyz", backend_job_id="be-xyz")

    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    rec = MagicMock(id=1)

    fake_ep = {"command": "BAKED_HEREDOC_COMMAND", "args_template": ""}

    # Patch build_entrypoint at its import site (inside _run_one_job).
    with patch("tao_sdk.script_runner.build_entrypoint", return_value=fake_ep) as build:
        # Avoid the polling loop — make get_job_status return Complete immediately.
        fake_sdk.get_job_status.return_value = MagicMock(status="Complete")
        fake_sdk.get_job_logs.return_value = "loss: 0.5\n"
        runner._poll_interval = 0  # tight loop
        runner._run_one_job(
            image="nvcr.io/test:1",
            action_cfg=runner.skill_ctx.action_cfg,
            specs={"train": {"num_epochs": 10}},
            rec=rec, metric_name="loss",
            workspace_path=str(tmp_path),
            platform_kwargs={"gpu_count": 4, "num_nodes": 1},
        )

    # build_entrypoint received the schema fields from skill_info
    build.assert_called_once()
    kwargs = build.call_args.kwargs
    assert kwargs["command"] == "fake train -e {config_path}"
    assert kwargs["config_format"] == "yaml"
    assert "dataset.train_data_sources[0].image_dir" in kwargs["inputs"]
    assert "results_dir" in kwargs["outputs"]
    assert kwargs["upload_excludes"] == ["inputs/"]
    assert kwargs["specs"]["train"]["num_epochs"] == 10

    # create_job got the baked command + platform kwargs (no old shape)
    create_kwargs = fake_sdk.create_job.call_args.kwargs
    assert create_kwargs["command"] == "BAKED_HEREDOC_COMMAND"
    assert create_kwargs["image"] == "nvcr.io/test:1"
    assert create_kwargs["gpu_count"] == 4
    assert create_kwargs["num_nodes"] == 1
    # Old kwargs should NOT be present
    for legacy in ("specs", "script_runner", "network_arch", "data_format",
                   "backend_details", "workspace_id", "train_dataset_uri"):
        assert legacy not in create_kwargs, f"legacy kwarg {legacy!r} leaked"
    assert fake_sdk.get_job_logs.call_count >= 1
    assert any(
        call.kwargs.get("tail") == _POLL_LOG_TAIL_LINES
        for call in fake_sdk.get_job_logs.call_args_list
    )
    assert any(not call.kwargs for call in fake_sdk.get_job_logs.call_args_list)


def test_run_one_job_ledgers_interrupted_ambiguous_submission(tmp_path):
    from tao_automl.runner import AutoMLRunner

    interrupted = SystemExit(1)
    interrupted.tao_job_id = "job-late-create"
    interrupted.tao_launch_uncertain = True
    fake_sdk = MagicMock()
    fake_sdk.create_job.side_effect = interrupted
    fake_sdk.cancel_job.side_effect = [False, True]
    fake_sdk.get_job_status.side_effect = [
        MagicMock(status="Pending"),
        MagicMock(status="Canceled"),
    ]
    runner = AutoMLRunner(
        sdk=fake_sdk,
        skill_dir=_write_fake_skill(tmp_path),
        action="train",
    )
    rec = MagicMock(id=7)
    workspace = tmp_path / "workspace"
    runner._cancel_confirmation_timeout = 0.1
    runner._cancel_confirmation_poll_interval = 0.001

    with patch(
        "tao_sdk.script_runner.build_entrypoint",
        return_value={"command": "true", "args_template": ""},
    ):
        with pytest.raises(SystemExit):
            runner._run_one_job(
                image="example.invalid/tao:test",
                action_cfg=runner.skill_ctx.action_cfg,
                specs={},
                rec=rec,
                metric_name="loss",
                workspace_path=str(workspace),
            )

    rec.assign_job_id.assert_called_once_with("job-late-create")
    assert runner._active_jobs == {}
    assert runner._cancel_requests == {}
    assert fake_sdk.cancel_job.call_args_list == [
        (("job-late-create",), {}),
        (("job-late-create",), {}),
    ]
    persisted = json.loads((workspace / "active_jobs.json").read_text())
    assert persisted == []
    artifacts = json.loads((workspace / "artifact_jobs.json").read_text())
    assert artifacts["job-late-create"] == "interrupted_canceled"


def test_run_one_job_prunes_confirmed_terminal_interrupted_launch(tmp_path):
    from tao_automl.runner import AutoMLRunner

    interrupted = KeyboardInterrupt("interrupted after backend create")
    interrupted.tao_job_id = "job-partial-output"
    interrupted.tao_launch_uncertain = False
    fake_sdk = MagicMock()
    fake_sdk.create_job.side_effect = interrupted
    fake_sdk.cancel_job.return_value = False
    fake_sdk.get_job_status.return_value = MagicMock(status="Canceled")
    fake_sdk.delete_job_artifacts.return_value = True
    runner = AutoMLRunner(
        sdk=fake_sdk,
        skill_dir=_write_fake_skill(tmp_path),
        action="train",
    )
    runner._delete_intermediate_ckpt = True
    rec = MagicMock(id=8)
    workspace = tmp_path / "workspace"

    with patch(
        "tao_sdk.script_runner.build_entrypoint",
        return_value={"command": "true", "args_template": ""},
    ):
        with pytest.raises(KeyboardInterrupt):
            runner._run_one_job(
                image="example.invalid/tao:test",
                action_cfg=runner.skill_ctx.action_cfg,
                specs={},
                rec=rec,
                metric_name="loss",
                workspace_path=str(workspace),
            )

    assert runner._active_jobs == {}
    assert runner._cancel_requests == {}
    fake_sdk.delete_job_artifacts.assert_called_once_with("job-partial-output")
    artifacts = json.loads((workspace / "artifact_jobs.json").read_text())
    assert artifacts["job-partial-output"] == "deleted"


def test_interrupt_immediately_after_sdk_return_cannot_orphan_writer(tmp_path):
    from tao_automl.runner import AutoMLRunner

    interrupt = KeyboardInterrupt("between return and registration")
    fake_sdk = MagicMock()
    fake_sdk.create_job.return_value = SimpleNamespace(
        id="job-returned-before-interrupt",
        backend_job_id="backend-1",
    )
    fake_sdk.cancel_job.return_value = True
    fake_sdk.get_job_status.return_value = SimpleNamespace(status="Canceled")
    runner = AutoMLRunner(
        sdk=fake_sdk,
        skill_dir=_write_fake_skill(tmp_path),
        action="train",
    )
    rec = MagicMock(id=10)
    rec.assign_job_id.side_effect = [interrupt, None]
    workspace = tmp_path / "workspace"

    with patch(
        "tao_sdk.script_runner.build_entrypoint",
        return_value={"command": "true", "args_template": ""},
    ):
        with pytest.raises(KeyboardInterrupt) as caught:
            runner._run_one_job(
                image="example.invalid/tao:test",
                action_cfg=runner.skill_ctx.action_cfg,
                specs={},
                rec=rec,
                metric_name="loss",
                workspace_path=str(workspace),
            )

    assert caught.value is interrupt
    assert rec.assign_job_id.call_args_list == [
        (("job-returned-before-interrupt",), {}),
        (("job-returned-before-interrupt",), {}),
    ]
    fake_sdk.cancel_job.assert_called_once_with("job-returned-before-interrupt")
    assert runner._active_jobs == {}
    assert json.loads((workspace / "active_jobs.json").read_text()) == []


def test_interrupted_guard_never_abandons_an_unledgered_writer(
    tmp_path, monkeypatch
):
    from tao_automl.runner import AutoMLRunner

    fake_sdk = MagicMock()
    fake_sdk.cancel_job.return_value = False
    fake_sdk.get_job_status.return_value = MagicMock(status="Pending")
    runner = AutoMLRunner(
        sdk=fake_sdk,
        skill_dir=_write_fake_skill(tmp_path),
        action="train",
    )
    persist = MagicMock(side_effect=[False, False, True])
    monkeypatch.setattr(runner, "_persist_active_jobs", persist)
    sleep = MagicMock()
    monkeypatch.setattr("tao_automl.runner.time.sleep", sleep)
    runner._cancel_confirmation_timeout = 0

    assert runner._guard_interrupted_launch(
        9, "job-unledgered", str(tmp_path / "workspace")
    ) is None
    assert persist.call_count == 3
    assert fake_sdk.cancel_job.call_count == 3
    assert runner._active_jobs == {9: "job-unledgered"}
    fake_sdk.delete_job_artifacts.assert_not_called()
    assert sleep.call_count == 2


def test_run_one_job_returns_multi_objective_values(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    fake_sdk = MagicMock()
    fake_sdk.create_job.return_value = MagicMock(id="job-mo", backend_job_id="be-mo")
    fake_sdk.get_job_status.return_value = MagicMock(status="Complete")
    fake_sdk.get_job_logs.return_value = "val_mAP: 0.75\nlatency_ms: 21.0\n"

    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    runner._poll_interval = 0
    rec = MagicMock(id=3)

    with patch(
        "tao_sdk.script_runner.build_entrypoint",
        return_value={"command": "BAKED_HEREDOC_COMMAND", "args_template": ""},
    ):
        metric, status = runner._run_one_job(
            image="nvcr.io/test:1",
            action_cfg=runner.skill_ctx.action_cfg,
            specs={"train": {"num_epochs": 1}},
            rec=rec,
            metric_name="val_mAP",
            objective_names=["val_mAP", "latency"],
            workspace_path=str(tmp_path),
            platform_kwargs={},
        )

    assert status == "success"
    assert metric == {"val_mAP": pytest.approx(0.75), "latency": pytest.approx(21.0)}


def test_run_one_job_accepts_multi_objective_eval_callback(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    fake_sdk = MagicMock()
    fake_sdk.create_job.return_value = MagicMock(
        id="job-mo-eval",
        backend_job_id="be-mo-eval",
    )
    fake_sdk.get_job_status.return_value = MagicMock(status="Complete")
    fake_sdk.get_job_logs.return_value = ""
    fake_sdk.get_job_results_dir.return_value = ""
    fake_sdk.read_job_result_file.return_value = ""

    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    runner._poll_interval = 0
    rec = MagicMock(id=4)

    with patch(
        "tao_sdk.script_runner.build_entrypoint",
        return_value={"command": "BAKED_HEREDOC_COMMAND", "args_template": ""},
    ):
        metric, status = runner._run_one_job(
            image="nvcr.io/test:1",
            action_cfg=runner.skill_ctx.action_cfg,
            specs={"train": {"num_epochs": 1}},
            rec=rec,
            metric_name="val_mAP",
            objective_names=["val_mAP", "latency"],
            eval_fn=lambda recommendation, job_id: {
                "val_mAP": 0.76,
                "latency": 20,
            },
            workspace_path=str(tmp_path),
            platform_kwargs={},
        )

    assert status == "success"
    assert metric == {"val_mAP": pytest.approx(0.76), "latency": 20.0}


def test_run_one_job_submits_nested_specs_to_python_script_sdk(tmp_path):
    """Python actions bypass the container entrypoint and image API."""
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_python_skill(tmp_path)
    fake_sdk = MagicMock()
    fake_sdk.create_python_job.return_value = MagicMock(
        id="job-python", backend_job_id="12345"
    )
    fake_sdk.get_job_status.return_value = MagicMock(status="Complete")
    fake_sdk.get_job_logs.return_value = "accuracy: 0.875\n"

    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    runner._poll_interval = 0
    rec = MagicMock(id=3)

    with patch(
        "tao_sdk.script_runner.build_entrypoint",
        side_effect=AssertionError("container entrypoint must not be built"),
    ) as build:
        metric, status = runner._run_one_job(
            image=None,
            action_cfg=runner.skill_ctx.action_cfg,
            specs={"model": {"n_estimators": 17, "max_depth": 4}},
            rec=rec,
            metric_name="accuracy",
            execution=runner.skill_ctx.execution,
            workspace_path=str(tmp_path / "workspace"),
            platform_kwargs={"gpu_count": 0},
        )

    build.assert_not_called()
    fake_sdk.create_job.assert_not_called()
    create_kwargs = fake_sdk.create_python_job.call_args.kwargs
    assert create_kwargs["script"] == str(skill_dir / "scripts/train.py")
    assert create_kwargs["specs"] == {
        "model": {"n_estimators": 17, "max_depth": 4}
    }
    assert create_kwargs["config_format"] == "json"
    assert create_kwargs["script_args"] == ["--config", "{config_path}"]
    assert create_kwargs["cwd"] == str(skill_dir)
    assert create_kwargs["network_arch"] == "public_random_forest"
    assert create_kwargs["action"] == "train"
    assert create_kwargs["gpu_count"] == 0
    assert metric == pytest.approx(0.875)
    assert status == "success"


def test_runner_runs_real_subprocess_job_for_non_train_action(tmp_path):
    """Exercise a real subprocess-backed SDK job for a non-train action."""
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_action_skill(tmp_path, action="quantize")
    runner = AutoMLRunner(
        sdk=_CompletedProcessSDK(),
        skill_dir=skill_dir,
        action="quantize",
        poll_interval=0,
    )

    result = runner.run(
        automl_settings={
            "algorithm": "bayesian",
            "metric": "action_metric",
            "direction": "maximize",
            "automl_max_recommendations": 1,
        },
        automl_hyperparameters=["train.optim.lr"],
        custom_param_ranges={
            "train.optim.lr": {"valid_min": 1e-5, "valid_max": 1e-3},
        },
        workspace_path=str(tmp_path / "workspace"),
        env_vars={"TAO_RESULTS_ROOT": str(tmp_path / "sdk-results")},
    )

    assert result["progress"]["completed"] == 1
    assert result["best"]["metric_value"] == pytest.approx(0.73)
    assert result["history"][0]["status"] == "success"


def test_run_one_job_allows_completed_metric_with_cleanup_rendezvous(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    fake_sdk = MagicMock()
    fake_sdk.create_job.return_value = MagicMock(id="job-xyz", backend_job_id="be-xyz")
    fake_sdk.get_job_status.return_value = MagicMock(status="Complete")
    fake_sdk.get_job_logs.return_value = (
        "[cosmos] Validation rank 0: avg_loss=0.951012, samples=6\n"
        "Saved best score to best_score.json: 0.9510115849549504\n"
        "torch.distributed.elastic.rendezvous.api.RendezvousConnectionError\n"
        "torch.distributed.DistNetworkError: Failed to recv, got 0 bytes.\n"
    )

    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    runner._poll_interval = 0
    rec = MagicMock(id=7)

    with patch(
        "tao_sdk.script_runner.build_entrypoint",
        return_value={"command": "BAKED_HEREDOC_COMMAND", "args_template": ""},
    ):
        metric, status = runner._run_one_job(
            image="nvcr.io/test:1",
            action_cfg=runner.skill_ctx.action_cfg,
            specs={"train": {"num_epochs": 1}},
            rec=rec,
            metric_name="val/avg_loss",
            workspace_path=str(tmp_path),
            platform_kwargs={},
        )

    assert metric == pytest.approx(0.951012)
    assert status == "success"
    fake_sdk.cancel_job.assert_not_called()


def test_terminal_streaming_recovers_metric_before_noisy_tail(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    full_logs = "accuracy: 0.875\n" + "noise\n" * 10_001

    class StreamingSDK:
        def __init__(self):
            self.create_job = MagicMock(
                return_value=MagicMock(id="job-noisy", backend_job_id="be")
            )
            self.log_tails = []

        def get_job_status(self, job_id):
            return MagicMock(status="Complete")

        def get_job_logs(self, job_id, tail=None):
            self.log_tails.append(tail)
            assert tail is not None
            return "".join(full_logs.splitlines(keepends=True)[-tail:])

        def iter_job_log_chunks(self, job_id, chunk_size=256 * 1024):
            for offset in range(0, len(full_logs), 31):
                yield full_logs[offset:offset + 31]

        def cancel_job(self, job_id):
            return False

    fake_sdk = StreamingSDK()
    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    runner._poll_interval = 0
    rec = MagicMock(id=10)

    with patch(
        "tao_sdk.script_runner.build_entrypoint",
        return_value={"command": "BAKED", "args_template": ""},
    ):
        metric, status = runner._run_one_job(
            image="nvcr.io/test:1",
            action_cfg=runner.skill_ctx.action_cfg,
            specs={"train": {"num_epochs": 1}},
            rec=rec,
            metric_name="accuracy",
            workspace_path=str(tmp_path),
            platform_kwargs={},
        )

    assert metric == pytest.approx(0.875)
    assert status == "success"
    assert fake_sdk.log_tails
    assert all(tail is not None for tail in fake_sdk.log_tails)


def test_run_one_job_cancels_hard_failure_and_recovers_remote_best_score(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    fake_sdk = MagicMock()
    fake_sdk.create_job.return_value = MagicMock(id="job-hard", backend_job_id="be-hard")
    fake_sdk.get_job_logs.return_value = (
        "Watchdog caught collective operation timeout: "
        "WorkNCCL(SeqNum=33271, OpType=ALLREDUCE)\n"
    )
    fake_sdk.get_job_status.return_value = MagicMock(status="Canceled")
    fake_sdk.read_job_result_file.return_value = (
        '{"best_score": 0.8927091135965706, "metric": "val_loss"}\n'
    )

    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    runner._poll_interval = 0
    rec = MagicMock(id=8)

    with patch(
        "tao_sdk.script_runner.build_entrypoint",
        return_value={"command": "BAKED_HEREDOC_COMMAND", "args_template": ""},
    ):
        metric, status = runner._run_one_job(
            image="nvcr.io/test:1",
            action_cfg=runner.skill_ctx.action_cfg,
            specs={"train": {"num_epochs": 10}},
            rec=rec,
            metric_name="val/avg_loss",
            workspace_path=str(tmp_path),
            platform_kwargs={},
        )

    assert metric == pytest.approx(0.8927091135965706)
    assert status == "failure"
    fake_sdk.cancel_job.assert_called_once_with("job-hard")


def test_run_one_job_preserves_metric_when_slurm_reports_canceled(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    fake_sdk = MagicMock()
    fake_sdk.create_job.return_value = MagicMock(id="job-canceled", backend_job_id="be")
    fake_sdk.get_job_status.return_value = MagicMock(status="Canceled")
    fake_sdk.get_job_logs.return_value = (
        "[SFT] Validation loss: 0.751 for train step 8/10, epoch 4\n"
    )

    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    runner._poll_interval = 0
    rec = MagicMock(id=9)

    with patch(
        "tao_sdk.script_runner.build_entrypoint",
        return_value={"command": "BAKED_HEREDOC_COMMAND", "args_template": ""},
    ):
        metric, status = runner._run_one_job(
            image="nvcr.io/test:1",
            action_cfg=runner.skill_ctx.action_cfg,
            specs={"train": {"num_epochs": 10}},
            rec=rec,
            metric_name="val_loss",
            workspace_path=str(tmp_path),
            platform_kwargs={},
        )

    assert metric == pytest.approx(0.751)
    assert status == "failure"


def test_new_job_is_canceled_if_first_active_ledger_write_fails(tmp_path, monkeypatch):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    fake_sdk = MagicMock()
    fake_sdk.create_job.return_value = MagicMock(
        id="job-unledgered", backend_job_id="backend-unledgered"
    )
    fake_sdk.cancel_job.return_value = True
    fake_sdk.get_job_status.return_value = SimpleNamespace(status="Canceled")
    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, poll_interval=0)
    runner._cancel_confirmation_timeout = 0
    monkeypatch.setattr(runner, "_persist_active_jobs", lambda _: False)
    rec = MagicMock(id=17)

    with patch(
        "tao_sdk.script_runner.build_entrypoint",
        return_value={"command": "BAKED", "args_template": ""},
    ):
        metric, status = runner._run_one_job(
            image="nvcr.io/test:1",
            action_cfg=runner.skill_ctx.action_cfg,
            specs={"train": {"num_epochs": 1}},
            rec=rec,
            metric_name="accuracy",
            workspace_path=str(tmp_path / "workspace"),
            platform_kwargs={},
        )

    assert metric is None
    assert status == "failure"
    assert runner._active_jobs == {}
    fake_sdk.cancel_job.assert_called_once_with("job-unledgered")
    fake_sdk.get_job_logs.assert_not_called()
    assert "active_job_registration_failed" in rec.failure_reason


def test_new_job_continues_when_active_ledger_retry_succeeds(tmp_path, monkeypatch):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    fake_sdk = MagicMock()
    fake_sdk.create_job.return_value = MagicMock(
        id="job-recovered", backend_job_id="backend-recovered"
    )
    fake_sdk.get_job_status.return_value = SimpleNamespace(status="Complete")
    fake_sdk.get_job_logs.return_value = "accuracy: 0.81\n"
    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, poll_interval=0)
    persist_results = iter((False, True))
    monkeypatch.setattr(
        runner,
        "_persist_active_jobs",
        lambda _: next(persist_results),
    )
    rec = MagicMock(id=18)

    with patch(
        "tao_sdk.script_runner.build_entrypoint",
        return_value={"command": "BAKED", "args_template": ""},
    ):
        metric, status = runner._run_one_job(
            image="nvcr.io/test:1",
            action_cfg=runner.skill_ctx.action_cfg,
            specs={"train": {"num_epochs": 1}},
            rec=rec,
            metric_name="accuracy",
            workspace_path=str(tmp_path / "workspace"),
            platform_kwargs={},
        )

    assert metric == pytest.approx(0.81)
    assert status == "success"
    assert runner._active_jobs == {18: "job-recovered"}
    fake_sdk.cancel_job.assert_not_called()


class _ArtifactSDK:
    def __init__(self):
        self.deleted = []
        self.canceled = []

    def delete_job_artifacts(self, job_id):
        self.deleted.append(job_id)
        return True

    def cancel_job(self, job_id):
        self.canceled.append(job_id)
        return None

    def get_job_status(self, job_id):
        return SimpleNamespace(status="Canceled", message="cancellation complete")


def test_retention_preflight_uses_concrete_sdk_capability(tmp_path):
    from tao_automl.runner import AutoMLRunner

    class SDK:
        def __init__(self):
            self.validated = None

        def validate_artifact_retention(self, **kwargs):
            self.validated = kwargs

    sdk = SDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._validate_artifact_retention_config({
        "mounts": [{"host_path": "/safe/results", "container_path": "/results"}],
        "gpu_count": 2,
    })

    assert sdk.validated == {
        "mounts": [{"host_path": "/safe/results", "container_path": "/results"}],
        "gpu_count": 2,
    }


def test_retention_preflight_propagates_unsupported_route_before_launch(tmp_path):
    from tao_automl.runner import AutoMLRunner

    class SDK:
        def validate_artifact_retention(self, **_kwargs):
            raise ValueError("named Docker volume cannot be reclaimed")

    runner = AutoMLRunner(sdk=SDK(), skill_dir=_write_fake_skill(tmp_path))
    workspace = tmp_path / "must-not-be-created"

    with pytest.raises(ValueError, match="cannot be reclaimed"):
        runner.run(
            automl_settings={
                "algorithm": "bayesian",
                "metric": "loss",
                "automl_delete_intermediate_ckpt": True,
            },
            workspace_path=str(workspace),
            mounts=[],
        )

    assert not workspace.exists()


def test_cleanup_requires_explicit_true_result(tmp_path):
    from tao_automl.runner import AutoMLRunner

    sdk = MagicMock()
    sdk.delete_job_artifacts.return_value = None
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._terminal_job_ids["job-1"] = "failure"

    assert not runner._delete_job_artifacts("job-1", "test")
    assert runner._terminal_job_ids["job-1"] == "failure"
    assert "job-1" not in runner._deleted_job_ids


def test_cleanup_retries_if_deletion_tombstone_is_not_durable(tmp_path, monkeypatch):
    from tao_automl.runner import AutoMLRunner

    sdk = MagicMock()
    sdk.delete_job_artifacts.return_value = True
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._workspace_path = str(tmp_path / "workspace")
    runner._terminal_job_ids["job-1"] = "failure"
    persist_results = iter((False, True))
    monkeypatch.setattr(
        runner,
        "_persist_artifact_jobs",
        lambda: next(persist_results),
    )

    assert not runner._delete_job_artifacts("job-1", "first attempt")
    assert runner._terminal_job_ids["job-1"] == "failure"
    assert "job-1" not in runner._deleted_job_ids

    assert runner._delete_job_artifacts("job-1", "retry")
    assert runner._terminal_job_ids["job-1"] == "deleted"
    assert sdk.delete_job_artifacts.call_count == 2


class _RetentionAutoML:
    def __init__(self, history, best=None):
        self.history = history
        self.best = best

    def get_history(self):
        return list(self.history)

    def get_best(self):
        return self.best


def _retention_rec(rec_id, job_id, status, metric=0.0, resume_from=None):
    from tao_automl.types import Recommendation

    rec = Recommendation(rec_id, {}, "accuracy")
    rec.assign_job_id(job_id)
    rec.update_result(metric)
    rec.update_status(status)
    rec.resume_from_job_id = resume_from
    return rec


def test_artifact_cleanup_preserves_best_active_and_resume_dependency(tmp_path):
    from tao_automl.runner import AutoMLRunner

    sdk = _ArtifactSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._algorithm = "bayesian"

    best = _retention_rec(0, "job-best", "success", 0.9)
    loser = _retention_rec(1, "job-loser", "success", 0.5)
    failed = _retention_rec(2, "job-failed", "failure")
    pending = _retention_rec(
        3, "job-active", "pending", resume_from="job-parent"
    )
    automl = _RetentionAutoML([best, loser, failed, pending], best=best)
    runner._active_jobs = {pending.id: pending.job_id}
    for rec in (best, loser, failed):
        runner._record_terminal_job(rec.job_id, rec.status)
    runner._record_terminal_job("job-parent", "success")

    runner._prune_intermediate_artifacts(automl, completed=False)

    assert sdk.deleted == ["job-failed", "job-loser"]
    assert "job-best" not in sdk.deleted
    assert "job-active" not in sdk.deleted
    assert "job-parent" not in sdk.deleted

    runner._active_jobs.clear()
    runner._prune_intermediate_artifacts(automl, completed=True)
    assert sdk.deleted == ["job-failed", "job-loser", "job-parent"]
    assert "job-best" not in sdk.deleted


def test_multifidelity_defers_successful_artifact_cleanup_until_complete(tmp_path):
    from tao_automl.runner import AutoMLRunner

    sdk = _ArtifactSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._algorithm = "hyperband"

    best = _retention_rec(0, "job-best", "success", 0.9)
    loser = _retention_rec(1, "job-loser", "success", 0.5)
    failed = _retention_rec(2, "job-failed", "failure")
    automl = _RetentionAutoML([best, loser, failed], best=best)
    for rec in (best, loser, failed):
        runner._record_terminal_job(rec.job_id, rec.status)

    runner._prune_intermediate_artifacts(automl, completed=False)
    assert sdk.deleted == ["job-failed"]

    runner._prune_intermediate_artifacts(automl, completed=True)
    assert sdk.deleted == ["job-failed", "job-loser"]
    assert "job-best" not in sdk.deleted


def test_incomplete_multifidelity_prunes_eliminated_successes(tmp_path):
    from tao_automl.runner import AutoMLRunner

    best = _retention_rec(0, "job-best", "success", 0.9)
    promotion_parent = _retention_rec(1, "job-promotion-parent", "success", 0.8)
    eliminated = _retention_rec(2, "job-eliminated", "success", 0.4)

    class RequiredCheckpointAutoML(_RetentionAutoML):
        def get_required_checkpoint_job_ids(self):
            return {promotion_parent.job_id}

    sdk = _ArtifactSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._algorithm = "asha"
    for rec in (best, promotion_parent, eliminated):
        runner._record_terminal_job(rec.job_id, rec.status)

    runner._prune_intermediate_artifacts(
        RequiredCheckpointAutoML(
            [best, promotion_parent, eliminated],
            best=best,
        ),
        completed=False,
    )

    assert sdk.deleted == ["job-eliminated"]
    assert "job-best" not in sdk.deleted
    assert "job-promotion-parent" not in sdk.deleted


def test_incomplete_multifidelity_releases_inactive_resume_parent(tmp_path):
    from tao_automl.runner import AutoMLRunner

    best = _retention_rec(0, "job-best", "success", 0.9)
    eliminated = _retention_rec(
        1,
        "job-eliminated-later-rung",
        "success",
        0.4,
        resume_from="job-eliminated-parent",
    )

    class RequiredCheckpointAutoML(_RetentionAutoML):
        def get_required_checkpoint_job_ids(self):
            return {best.job_id}

    sdk = _ArtifactSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._algorithm = "asha"
    for job_id in (
        best.job_id,
        eliminated.job_id,
        eliminated.resume_from_job_id,
    ):
        runner._record_terminal_job(job_id, "success")

    runner._prune_intermediate_artifacts(
        RequiredCheckpointAutoML([best, eliminated], best=best),
        completed=False,
    )

    assert sdk.deleted == [
        "job-eliminated-later-rung",
        "job-eliminated-parent",
    ]


def test_checkpoint_decision_window_supersedes_older_larger_budget():
    """A new Hyperband bracket can start below an older bracket's max budget."""
    from tao_automl.controller.controller import Controller

    old_high_budget = _retention_rec(
        0, "job-old-high-budget", "success", 0.9
    )
    old_high_budget.specs = {"train.num_epochs": 100}
    old_high_budget.checkpoint_window = 1
    current_a = _retention_rec(1, "job-current-a", "success", 0.8)
    current_a.specs = {"train.num_epochs": 10}
    current_a.checkpoint_window = 2
    current_b = _retention_rec(2, "job-current-b", "success", 0.7)
    current_b.specs = {"train.num_epochs": 10}
    current_b.checkpoint_window = 2

    controller = object.__new__(Controller)
    controller.history = [old_high_budget, current_a, current_b]
    controller.brain = SimpleNamespace(last_launched_count=2)

    required = controller.get_required_checkpoint_job_ids()

    assert required == {"job-current-a", "job-current-b"}
    assert "job-old-high-budget" not in required


def test_multi_objective_retention_preserves_every_pareto_checkpoint(tmp_path):
    from tao_automl.runner import AutoMLRunner

    best = _retention_rec(0, "job-scalarized-best", "success", 0.9)
    frontier_peer = _retention_rec(1, "job-frontier-peer", "success", 0.8)
    dominated = _retention_rec(2, "job-dominated", "success", 0.5)

    class ParetoAutoML(_RetentionAutoML):
        def get_pareto_front(self):
            return [best, frontier_peer]

    sdk = _ArtifactSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._algorithm = "bayesian"
    runner._retain_pareto_front = True
    for rec in (best, frontier_peer, dominated):
        runner._record_terminal_job(rec.job_id, rec.status)

    runner._prune_intermediate_artifacts(
        ParetoAutoML([best, frontier_peer, dominated], best=best),
        completed=True,
    )

    assert sdk.deleted == ["job-dominated"]
    assert "job-scalarized-best" not in sdk.deleted
    assert "job-frontier-peer" not in sdk.deleted


def test_public_automl_facade_exposes_pareto_front_for_retention():
    from tao_automl import AutoML

    expected = [SimpleNamespace(id=1, job_id="job-frontier")]
    automl = object.__new__(AutoML)
    automl._controller = MagicMock()
    automl._controller.get_pareto_front.return_value = expected

    assert automl.get_pareto_front() is expected
    automl._controller.get_pareto_front.assert_called_once_with()


def test_public_automl_facade_exposes_retention_proofs():
    from tao_automl import AutoML

    verified = SimpleNamespace(id=2, job_id="job-full-fidelity")
    automl = object.__new__(AutoML)
    automl._controller = MagicMock()
    automl._controller.get_required_checkpoint_job_ids.return_value = {
        "job-promotion-parent"
    }
    automl._controller.get_verified_full_fidelity_best.return_value = verified

    assert automl.get_required_checkpoint_job_ids() == {"job-promotion-parent"}
    assert automl.get_verified_full_fidelity_best() is verified


def test_hybrid_retains_all_successes_without_verified_full_fidelity_winner(tmp_path):
    from tao_automl.runner import AutoMLRunner

    sdk = _ArtifactSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._algorithm = "hybrid"

    outer_best = _retention_rec(0, "job-low-fidelity-best", "success", 0.95)
    possible_full_fidelity = _retention_rec(
        1, "job-possible-full-fidelity", "success", 0.9
    )
    failed = _retention_rec(2, "job-failed", "failure")
    automl = _RetentionAutoML(
        [outer_best, possible_full_fidelity, failed], best=outer_best
    )
    for rec in (outer_best, possible_full_fidelity, failed):
        runner._record_terminal_job(rec.job_id, rec.status)

    runner._prune_intermediate_artifacts(automl, completed=False)
    runner._prune_intermediate_artifacts(automl, completed=True)

    assert sdk.deleted == ["job-failed"]
    assert "job-low-fidelity-best" not in sdk.deleted
    assert "job-possible-full-fidelity" not in sdk.deleted


def test_hybrid_prunes_losers_with_verified_full_fidelity_winner(tmp_path):
    from tao_automl.runner import AutoMLRunner

    winner = _retention_rec(0, "job-full-fidelity-best", "success", 0.9)
    loser = _retention_rec(1, "job-hybrid-loser", "success", 0.5)

    class VerifiedHybridAutoML(_RetentionAutoML):
        def get_verified_full_fidelity_best(self):
            return winner

    sdk = _ArtifactSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._algorithm = "hybrid"
    for rec in (winner, loser):
        runner._record_terminal_job(rec.job_id, rec.status)

    runner._prune_intermediate_artifacts(
        VerifiedHybridAutoML([winner, loser], best=winner),
        completed=True,
    )

    assert sdk.deleted == ["job-hybrid-loser"]
    assert "job-full-fidelity-best" not in sdk.deleted


def test_hybrid_prunes_nonwinner_only_with_verified_full_fidelity_winner(tmp_path):
    from tao_automl.runner import AutoMLRunner

    winner = _retention_rec(0, "job-verified-winner", "success", 0.9)
    loser = _retention_rec(1, "job-other-success", "success", 0.8)

    class VerifiedHybridAutoML(_RetentionAutoML):
        def get_verified_full_fidelity_best(self):
            return winner

    sdk = _ArtifactSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._algorithm = "hybrid"
    for rec in (winner, loser):
        runner._record_terminal_job(rec.job_id, rec.status)

    runner._prune_intermediate_artifacts(
        VerifiedHybridAutoML([winner, loser], best=winner), completed=True
    )

    assert sdk.deleted == ["job-other-success"]
    assert "job-verified-winner" not in sdk.deleted


def test_cleanup_never_adopts_external_resume_parent_as_owned(tmp_path):
    from tao_automl.runner import AutoMLRunner

    best = _retention_rec(
        0,
        "job-best",
        "success",
        0.9,
        resume_from="external-restored-parent",
    )
    owned_loser = _retention_rec(1, "job-owned-loser", "success", 0.5)
    sdk = _ArtifactSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._algorithm = "bayesian"
    runner._record_terminal_job(best.job_id, best.status)
    runner._record_terminal_job(owned_loser.job_id, owned_loser.status)

    runner._prune_intermediate_artifacts(
        _RetentionAutoML([best, owned_loser], best=best), completed=True
    )

    assert sdk.deleted == ["job-owned-loser"]
    assert "external-restored-parent" not in sdk.deleted


@pytest.mark.parametrize(
    "cleanup_override, expected_deleted",
    [(None, ["job-1"]), (False, [])],
)
def test_run_prunes_non_best_by_default_and_supports_opt_out(
    tmp_path, monkeypatch, cleanup_override, expected_deleted,
):
    from tao_automl.runner import AutoMLRunner
    from tao_automl.types import Recommendation

    class TwoTrialAutoML:
        def __init__(self, *args, **kwargs):
            self.recs = [
                Recommendation(0, {"train.num_epochs": 1}, "accuracy"),
                Recommendation(1, {"train.num_epochs": 1}, "accuracy"),
            ]
            self.index = 0
            self._state_store = MagicMock()
            self._state_store.get_job_specs.return_value = None

        def is_complete(self):
            return self.index >= len(self.recs)

        def next_recommendation(self):
            return [self.recs[self.index]]

        def report_result(self, rec_id, metric_value, status):
            rec = self.recs[self.index]
            rec.update_result(metric_value)
            rec.update_status(status)
            self.index += 1

        def get_best(self):
            successful = [rec for rec in self.recs if rec.status == "success"]
            return max(successful, key=lambda rec: rec.result) if successful else None

        def get_progress(self):
            best = self.get_best()
            return {
                "completed": self.index,
                "best_metric": best.result if best else None,
            }

        def get_history(self):
            return list(self.recs)

    metrics = {0: 0.9, 1: 0.5}

    def complete_job(self, *args, **kwargs):
        rec = kwargs["rec"]
        rec.assign_job_id(f"job-{rec.id}")
        return metrics[rec.id], "success"

    sdk = _ArtifactSDK()
    monkeypatch.setattr("tao_automl.AutoML", TwoTrialAutoML)
    monkeypatch.setattr(AutoMLRunner, "_run_one_job", complete_job)
    settings = {
        "algorithm": "bayesian",
        "metric": "accuracy",
        "run_baseline": False,
    }
    if cleanup_override is not None:
        settings["automl_delete_intermediate_ckpt"] = cleanup_override

    result = AutoMLRunner(
        sdk=sdk, skill_dir=_write_fake_skill(tmp_path)
    ).run(automl_settings=settings, workspace_path=str(tmp_path / "workspace"))

    assert result["best"]["rec_id"] == 0
    assert sdk.deleted == expected_deleted
    assert "job-0" not in sdk.deleted


def test_runner_cap_is_checked_before_generation_and_never_final_prunes(
    tmp_path, monkeypatch,
):
    from tao_automl.runner import AutoMLRunner

    best = _retention_rec(
        0,
        "job-promotion-input",
        "success",
        0.7,
        resume_from="job-earlier-rung",
    )

    class IncompleteCappedAutoML:
        next_calls = 0

        def __init__(self, *args, **kwargs):
            pass

        def is_complete(self):
            return False

        def next_recommendation(self):
            type(self).next_calls += 1
            raise AssertionError("recommendations must not be generated past the cap")

        def get_progress(self):
            return {"completed": 1, "best_metric": best.result}

        def get_best(self):
            return best

        def get_history(self):
            return [best]

    monkeypatch.setattr("tao_automl.AutoML", IncompleteCappedAutoML)
    sdk = _ArtifactSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    prune_modes = []
    real_prune = runner._prune_intermediate_artifacts

    def track_prune(automl, *, completed):
        prune_modes.append(completed)
        return real_prune(automl, completed=completed)

    monkeypatch.setattr(runner, "_prune_intermediate_artifacts", track_prune)

    result = runner.run(
        automl_settings={
            "algorithm": "hyperband",
            "metric": "accuracy",
            "automl_max_recommendations": 1,
            "run_baseline": False,
            "run_final_evaluation": False,
        },
        workspace_path=str(tmp_path / "workspace"),
    )

    assert result["best"]["rec_id"] == best.id
    assert IncompleteCappedAutoML.next_calls == 0
    assert prune_modes and all(mode is False for mode in prune_modes)
    assert sdk.deleted == []


def test_runner_never_slices_generated_recommendation_batch(tmp_path, monkeypatch):
    from tao_automl.runner import AutoMLRunner
    from tao_automl.types import Recommendation

    class BatchedAutoML:
        def __init__(self, *args, **kwargs):
            self.recs = [
                Recommendation(0, {"train.num_epochs": 1}, "accuracy"),
                Recommendation(1, {"train.num_epochs": 1}, "accuracy"),
            ]
            self.completed = 0
            self.generated = False
            self._state_store = MagicMock()
            self._state_store.get_job_specs.return_value = None

        def is_complete(self):
            return self.completed == len(self.recs)

        def next_recommendation(self):
            assert not self.generated
            self.generated = True
            return list(self.recs)

        def report_result(self, rec_id, metric_value, status):
            rec = self.recs[rec_id]
            rec.update_result(metric_value)
            rec.update_status(status)
            self.completed += 1

        def get_progress(self):
            return {"completed": self.completed}

        def get_best(self):
            completed = [rec for rec in self.recs if rec.status == "success"]
            return max(completed, key=lambda rec: rec.result) if completed else None

        def get_history(self):
            return list(self.recs)

    launched = []

    def complete_job(self, *args, **kwargs):
        rec = kwargs["rec"]
        launched.append(rec.id)
        rec.assign_job_id(f"job-{rec.id}")
        return 0.8 - (rec.id * 0.1), "success"

    monkeypatch.setattr("tao_automl.AutoML", BatchedAutoML)
    monkeypatch.setattr(AutoMLRunner, "_run_one_job", complete_job)

    result = AutoMLRunner(
        sdk=_ArtifactSDK(), skill_dir=_write_fake_skill(tmp_path)
    ).run(
        automl_settings={
            "algorithm": "hyperband",
            "metric": "accuracy",
            "automl_max_recommendations": 1,
            "automl_delete_intermediate_ckpt": False,
            "run_baseline": False,
            "run_final_evaluation": False,
        },
        workspace_path=str(tmp_path / "workspace"),
    )

    assert launched == [0, 1]
    assert result["progress"]["completed"] == 2


def test_artifact_cleanup_warns_when_sdk_capability_is_absent(tmp_path, caplog):
    from tao_automl.runner import AutoMLRunner

    runner = AutoMLRunner(sdk=object(), skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    loser = _retention_rec(0, "job-loser", "failure")
    automl = _RetentionAutoML([loser])
    runner._record_terminal_job(loser.job_id, loser.status)

    runner._prune_intermediate_artifacts(automl, completed=False)
    runner._prune_intermediate_artifacts(automl, completed=False)

    messages = [record.message for record in caplog.records]
    assert sum("does not provide delete_job_artifacts" in msg for msg in messages) == 1


def test_resumed_cleanup_uses_persisted_parent_artifact_ledger(tmp_path):
    from tao_automl.runner import AutoMLRunner, _load_artifact_jobs

    workspace = str(tmp_path / "workspace")
    skill_dir = _write_fake_skill(tmp_path)
    first_runner = AutoMLRunner(
        sdk=_ArtifactSDK(), skill_dir=skill_dir
    )
    first_runner._workspace_path = workspace
    first_runner._record_terminal_job("job-parent", "success")

    sdk = _ArtifactSDK()
    resumed_runner = AutoMLRunner(sdk=sdk, skill_dir=skill_dir)
    resumed_runner._workspace_path = workspace
    resumed_runner._delete_intermediate_ckpt = True
    resumed_runner._algorithm = "hyperband"
    resumed_runner._terminal_job_ids.update(_load_artifact_jobs(workspace))
    child = _retention_rec(
        0, "job-final-best", "success", 0.9, resume_from="job-parent"
    )
    resumed_runner._record_terminal_job(child.job_id, child.status)

    resumed_runner._prune_intermediate_artifacts(
        _RetentionAutoML([child], best=child), completed=True
    )

    assert sdk.deleted == ["job-parent"]
    assert "job-final-best" not in sdk.deleted


def test_terminal_report_failure_keeps_normal_job_durably_active(tmp_path):
    from tao_automl.runner import AutoMLRunner

    rec = _retention_rec(7, "job-active", "pending")

    class FailingAutoML(_RetentionAutoML):
        def report_result(self, rec_id, metric_value, status):
            raise OSError("controller state write failed")

    sdk = _ArtifactSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._workspace_path = str(tmp_path / "workspace")
    runner._active_jobs = {rec.id: rec.job_id}
    assert runner._persist_active_jobs(runner._workspace_path)

    with pytest.raises(OSError, match="controller state write failed"):
        runner._finalize_terminal_job(
            automl=FailingAutoML([rec]),
            rec=rec,
            job_id=rec.job_id,
            metric_value=0.8,
            status="success",
            workspace_path=runner._workspace_path,
        )

    assert runner._active_jobs == {rec.id: rec.job_id}
    persisted = json.loads(
        (tmp_path / "workspace" / "active_jobs.json").read_text()
    )
    assert persisted[0]["job_id"] == rec.job_id
    assert runner._terminal_job_ids == {}
    assert sdk.deleted == []


def test_terminal_job_orders_report_ledger_clear_then_prune(tmp_path, monkeypatch):
    from tao_automl.runner import AutoMLRunner

    rec = _retention_rec(6, "job-ordered", "pending")
    events = []

    class OrderedAutoML(_RetentionAutoML):
        def report_result(self, rec_id, metric_value, status):
            assert runner._active_jobs == {rec.id: rec.job_id}
            events.append("report")
            rec.update_result(metric_value)
            rec.update_status(status)

    runner = AutoMLRunner(
        sdk=_ArtifactSDK(), skill_dir=_write_fake_skill(tmp_path)
    )
    runner._workspace_path = str(tmp_path / "workspace")
    runner._active_jobs = {rec.id: rec.job_id}
    assert runner._persist_active_jobs(runner._workspace_path)
    real_record = runner._record_terminal_job
    real_persist_active = runner._persist_active_jobs

    def record_terminal(job_id, status):
        assert runner._active_jobs == {rec.id: rec.job_id}
        events.append("ledger")
        return real_record(job_id, status)

    def persist_active(workspace_path):
        assert runner._active_jobs == {}
        events.append("clear")
        return real_persist_active(workspace_path)

    def prune(automl, *, completed):
        assert runner._active_jobs == {}
        events.append("prune")

    monkeypatch.setattr(runner, "_record_terminal_job", record_terminal)
    monkeypatch.setattr(runner, "_persist_active_jobs", persist_active)
    monkeypatch.setattr(runner, "_prune_intermediate_artifacts", prune)

    runner._finalize_terminal_job(
        automl=OrderedAutoML([rec], best=rec),
        rec=rec,
        job_id=rec.job_id,
        metric_value=0.8,
        status="success",
        workspace_path=runner._workspace_path,
    )

    assert events == ["report", "ledger", "clear", "prune"]


def test_terminal_ledger_failure_keeps_reported_job_durably_active(
    tmp_path, monkeypatch,
):
    from tao_automl.runner import AutoMLRunner

    rec = _retention_rec(8, "job-ledger-failure", "pending")

    class ReportingAutoML(_RetentionAutoML):
        def report_result(self, rec_id, metric_value, status):
            rec.update_result(metric_value)
            rec.update_status(status)

    runner = AutoMLRunner(
        sdk=_ArtifactSDK(), skill_dir=_write_fake_skill(tmp_path)
    )
    runner._workspace_path = str(tmp_path / "workspace")
    runner._active_jobs = {rec.id: rec.job_id}
    assert runner._persist_active_jobs(runner._workspace_path)
    monkeypatch.setattr(runner, "_persist_artifact_jobs", lambda: False)

    with pytest.raises(RuntimeError, match="terminal artifact ledger"):
        runner._finalize_terminal_job(
            automl=ReportingAutoML([rec], best=rec),
            rec=rec,
            job_id=rec.job_id,
            metric_value=0.8,
            status="success",
            workspace_path=runner._workspace_path,
        )

    assert rec.status == "success"
    assert runner._active_jobs == {rec.id: rec.job_id}
    persisted = json.loads(
        (tmp_path / "workspace" / "active_jobs.json").read_text()
    )
    assert persisted[0]["job_id"] == rec.job_id


def test_resume_report_failure_keeps_terminal_job_durably_active(tmp_path):
    from tao_automl.runner import AutoMLRunner

    class CompletedSDK(_ArtifactSDK):
        def get_job_logs(self, job_id, tail=None):
            return "accuracy: 0.7\n"

        def get_job_status(self, job_id):
            return SimpleNamespace(status="Complete")

    rec = _retention_rec(4, "job-recovered", "pending")

    class FailingResumeAutoML(_RetentionAutoML):
        def report_result(self, rec_id, metric_value, status):
            raise OSError("resume controller state write failed")

    runner = AutoMLRunner(
        sdk=CompletedSDK(),
        skill_dir=_write_fake_skill(tmp_path),
        poll_interval=0,
    )
    workspace = str(tmp_path / "workspace")
    runner._workspace_path = workspace
    runner._active_jobs = {rec.id: rec.job_id}
    assert runner._persist_active_jobs(workspace)

    with pytest.raises(OSError, match="resume controller state write failed"):
        runner._recover_pending_job(
            entry={"rec_id": rec.id, "job_id": rec.job_id},
            automl=FailingResumeAutoML([rec]),
            metric_name="accuracy",
            metric_extractor=None,
            eval_fn=None,
            workspace_path=workspace,
            on_result=None,
            platform_kwargs={},
        )

    assert runner._active_jobs == {rec.id: rec.job_id}
    persisted = json.loads(
        (tmp_path / "workspace" / "active_jobs.json").read_text()
    )
    assert persisted[0]["job_id"] == rec.job_id
    assert runner._terminal_job_ids == {}


def test_cancel_active_jobs_cleans_artifacts_and_persisted_state(tmp_path):
    from tao_automl.runner import AutoMLRunner

    sdk = _ArtifactSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._workspace_path = str(tmp_path / "workspace")
    rec = _retention_rec(7, "job-active", "pending")

    class CancelAutoML(_RetentionAutoML):
        def report_result(self, rec_id, metric_value, status):
            rec.update_result(metric_value)
            rec.update_status(status)

    runner._automl = CancelAutoML([rec])
    runner._active_jobs = {rec.id: rec.job_id}

    runner.cancel_active_jobs(reason="test cancellation")

    assert sdk.canceled == ["job-active"]
    assert sdk.deleted == ["job-active"]
    assert runner._active_jobs == {}
    assert json.loads((tmp_path / "workspace" / "active_jobs.json").read_text()) == []


def test_cancel_active_jobs_false_result_preserves_active_state(tmp_path):
    from tao_automl.runner import AutoMLRunner

    class RefusingSDK(_ArtifactSDK):
        def cancel_job(self, job_id):
            self.canceled.append(job_id)
            return False

        def get_job_status(self, job_id):
            return SimpleNamespace(status="Running")

    sdk = RefusingSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._workspace_path = str(tmp_path / "workspace")
    runner._active_jobs = {7: "job-still-running"}
    runner._cancel_confirmation_timeout = 0

    runner.cancel_active_jobs(reason="test cancellation refusal")

    assert sdk.canceled == ["job-still-running"]
    assert sdk.deleted == []
    assert runner._active_jobs == {7: "job-still-running"}
    persisted = json.loads(
        (tmp_path / "workspace" / "active_jobs.json").read_text()
    )
    assert persisted[0]["rec_id"] == 7
    assert persisted[0]["job_id"] == "job-still-running"


def test_cancel_false_reconciles_job_that_already_became_terminal(tmp_path):
    from tao_automl.runner import AutoMLRunner

    rec = _retention_rec(7, "job-finished-during-cancel", "pending")

    class AlreadyTerminalSDK(_ArtifactSDK):
        def cancel_job(self, job_id):
            self.canceled.append(job_id)
            return False

    class CancelAutoML(_RetentionAutoML):
        def report_result(self, rec_id, metric_value, status):
            rec.update_result(metric_value)
            rec.update_status(status)

    sdk = AlreadyTerminalSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._workspace_path = str(tmp_path / "workspace")
    runner._automl = CancelAutoML([rec])
    runner._active_jobs = {rec.id: rec.job_id}

    runner.cancel_active_jobs(reason="terminal race")

    assert sdk.canceled == [rec.job_id]
    assert sdk.deleted == [rec.job_id]
    assert runner._active_jobs == {}


def test_cancel_race_preserves_completed_job_for_result_recovery(tmp_path):
    from tao_automl.runner import AutoMLRunner

    rec = _retention_rec(7, "job-completed-before-cancel", "pending")

    class CompletedDuringCancelSDK(_ArtifactSDK):
        def cancel_job(self, job_id):
            self.canceled.append(job_id)
            return False

        def get_job_status(self, job_id):
            return SimpleNamespace(status="Complete")

    class TrackingAutoML(_RetentionAutoML):
        def __init__(self, history):
            super().__init__(history)
            self.reports = []

        def report_result(self, rec_id, metric_value, status):
            self.reports.append((rec_id, metric_value, status))

    sdk = CompletedDuringCancelSDK()
    automl = TrackingAutoML([rec])
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._workspace_path = str(tmp_path / "workspace")
    runner._automl = automl
    runner._active_jobs = {rec.id: rec.job_id}

    runner.cancel_active_jobs(reason="completion race")

    assert sdk.canceled == []
    assert sdk.deleted == []
    assert automl.reports == []
    assert runner._active_jobs == {rec.id: rec.job_id}
    persisted = json.loads(
        (tmp_path / "workspace" / "active_jobs.json").read_text()
    )
    assert persisted[0]["job_id"] == rec.job_id


def test_terminal_cancel_prunes_prior_multifidelity_losers(tmp_path):
    from tao_automl.runner import AutoMLRunner

    best = _retention_rec(0, "job-best", "success", 0.9)
    loser_a = _retention_rec(1, "job-loser-a", "success", 0.6)
    loser_b = _retention_rec(2, "job-loser-b", "success", 0.5)
    active = _retention_rec(3, "job-active", "pending")

    class CancelAutoML(_RetentionAutoML):
        def report_result(self, rec_id, metric_value, status):
            active.update_result(metric_value)
            active.update_status(status)

    sdk = _ArtifactSDK()
    automl = CancelAutoML(
        [best, loser_a, loser_b, active],
        best=best,
    )
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._algorithm = "hyperband"
    runner._delete_intermediate_ckpt = True
    runner._workspace_path = str(tmp_path / "workspace")
    runner._automl = automl
    runner._active_jobs = {active.id: active.job_id}
    for rec in (best, loser_a, loser_b):
        runner._record_terminal_job(rec.job_id, rec.status)

    runner.cancel_active_jobs(reason="terminal signal")

    assert runner._active_jobs == {}
    assert set(sdk.deleted) == {
        "job-active",
        "job-loser-a",
        "job-loser-b",
    }
    assert "job-best" not in sdk.deleted


def test_cancel_error_reconciles_job_that_already_became_terminal(tmp_path):
    from tao_automl.runner import AutoMLRunner

    rec = _retention_rec(7, "job-terminal-after-transport-error", "pending")

    class AmbiguousCancelSDK(_ArtifactSDK):
        def cancel_job(self, job_id):
            self.canceled.append(job_id)
            raise TimeoutError("cancel response lost")

    class CancelAutoML(_RetentionAutoML):
        def report_result(self, rec_id, metric_value, status):
            rec.update_result(metric_value)
            rec.update_status(status)

    sdk = AmbiguousCancelSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._workspace_path = str(tmp_path / "workspace")
    runner._automl = CancelAutoML([rec])
    runner._active_jobs = {rec.id: rec.job_id}

    runner.cancel_active_jobs(reason="terminal transport race")

    assert sdk.canceled == [rec.job_id]
    assert sdk.deleted == [rec.job_id]
    assert runner._active_jobs == {}


def test_cancel_history_failure_still_stops_writer_and_retains_identity(tmp_path):
    from tao_automl.runner import AutoMLRunner

    class UnreadableAutoML:
        def get_history(self):
            raise OSError("controller state unavailable")

    sdk = _ArtifactSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._workspace_path = str(tmp_path / "workspace")
    runner._automl = UnreadableAutoML()
    runner._active_jobs = {7: "job-history-unavailable"}

    runner.cancel_active_jobs(reason="signal")

    assert sdk.canceled == ["job-history-unavailable"]
    assert sdk.deleted == []
    assert runner._active_jobs == {7: "job-history-unavailable"}
    persisted = json.loads(
        (tmp_path / "workspace" / "active_jobs.json").read_text()
    )
    assert persisted[0]["cancel_requested"] is True


def test_cancel_intent_enospc_still_stops_writer(tmp_path, monkeypatch):
    from tao_automl.runner import AutoMLRunner

    rec = _retention_rec(7, "job-consuming-last-space", "pending")

    class CancelAutoML(_RetentionAutoML):
        def report_result(self, rec_id, metric_value, status):
            rec.update_result(metric_value)
            rec.update_status(status)

    sdk = _ArtifactSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._workspace_path = str(tmp_path / "workspace")
    runner._automl = CancelAutoML([rec])
    runner._active_jobs = {rec.id: rec.job_id}
    monkeypatch.setattr(runner, "_persist_active_jobs", lambda _path: False)

    runner.cancel_active_jobs(reason="disk full")

    assert sdk.canceled == [rec.job_id]
    assert sdk.deleted == [rec.job_id]
    assert runner._active_jobs == {rec.id: rec.job_id}


def test_artifact_ledger_enospc_reclaims_confirmed_canceled_job(
    tmp_path, monkeypatch,
):
    from tao_automl.runner import AutoMLRunner

    rec = _retention_rec(7, "job-partial-checkpoint", "pending")

    class CancelAutoML(_RetentionAutoML):
        def report_result(self, rec_id, metric_value, status):
            rec.update_result(metric_value)
            rec.update_status(status)

    sdk = _ArtifactSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._workspace_path = str(tmp_path / "workspace")
    runner._automl = CancelAutoML([rec])
    runner._active_jobs = {rec.id: rec.job_id}
    monkeypatch.setattr(runner, "_persist_artifact_jobs", lambda: False)

    runner.cancel_active_jobs(reason="disk full")

    assert sdk.canceled == [rec.job_id]
    assert sdk.deleted == [rec.job_id]
    assert runner._active_jobs == {rec.id: rec.job_id}


def test_unledgered_canceled_launch_is_reclaimed_when_ledgers_are_full(
    tmp_path, monkeypatch,
):
    from tao_automl.runner import AutoMLRunner

    sdk = _ArtifactSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._active_jobs = {9: "job-never-ledgered"}
    monkeypatch.setattr(runner, "_persist_active_jobs", lambda _path: False)
    monkeypatch.setattr(runner, "_persist_artifact_jobs", lambda: False)

    status = runner._cancel_unledgered_job(
        9,
        "job-never-ledgered",
        str(tmp_path / "workspace"),
    )

    assert status == "Canceled"
    assert sdk.canceled == ["job-never-ledgered"]
    assert sdk.deleted == ["job-never-ledgered"]
    assert runner._active_jobs == {}


def test_orphan_finalize_failure_does_not_skip_later_cancellations(
    tmp_path, monkeypatch,
):
    from tao_automl.runner import AutoMLRunner

    sdk = _ArtifactSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._workspace_path = str(tmp_path / "workspace")
    runner._automl = _RetentionAutoML([])
    runner._active_jobs = {1: "job-orphan-1", 2: "job-orphan-2"}
    persist_results = iter((False, True, True, True, True))
    monkeypatch.setattr(
        runner,
        "_persist_artifact_jobs",
        lambda: next(persist_results, True),
    )

    runner.cancel_active_jobs(reason="signal")

    assert sdk.canceled == ["job-orphan-1", "job-orphan-2"]


def test_cancel_timeout_preserves_active_state_and_skips_report_and_delete(tmp_path):
    from tao_automl.runner import AutoMLRunner

    rec = _retention_rec(7, "job-still-running", "pending")

    class RunningSDK(_ArtifactSDK):
        def __init__(self):
            super().__init__()
            self.status_checks = 0

        def get_job_status(self, job_id):
            self.status_checks += 1
            return SimpleNamespace(status="Running")

    class TrackingAutoML(_RetentionAutoML):
        def __init__(self, history):
            super().__init__(history)
            self.reports = []

        def report_result(self, rec_id, metric_value, status):
            self.reports.append((rec_id, metric_value, status))

    sdk = RunningSDK()
    automl = TrackingAutoML([rec])
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._workspace_path = str(tmp_path / "workspace")
    runner._automl = automl
    runner._active_jobs = {rec.id: rec.job_id}
    runner._cancel_confirmation_timeout = 0

    runner.cancel_active_jobs(reason="bounded timeout test")

    assert sdk.canceled == [rec.job_id]
    assert sdk.status_checks == 2
    assert automl.reports == []
    assert sdk.deleted == []
    assert runner._active_jobs == {rec.id: rec.job_id}
    persisted = json.loads(
        (tmp_path / "workspace" / "active_jobs.json").read_text()
    )
    assert persisted[0]["cancel_requested"] is True
    assert persisted[0]["job_id"] == rec.job_id


def test_cancel_report_failure_retains_confirmed_job_for_resume(tmp_path):
    from tao_automl.runner import AutoMLRunner

    rec = _retention_rec(9, "job-confirmed-canceled", "pending")

    class FailingCancelAutoML(_RetentionAutoML):
        def report_result(self, rec_id, metric_value, status):
            raise OSError("could not save canceled result")

    sdk = _ArtifactSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._workspace_path = str(tmp_path / "workspace")
    runner._automl = FailingCancelAutoML([rec])
    runner._active_jobs = {rec.id: rec.job_id}

    runner.cancel_active_jobs(reason="report persistence test")

    assert sdk.canceled == [rec.job_id]
    assert sdk.deleted == [rec.job_id]
    assert runner._active_jobs == {rec.id: rec.job_id}
    persisted = json.loads(
        (tmp_path / "workspace" / "active_jobs.json").read_text()
    )
    assert persisted[0]["cancel_requested"] is True


def test_cancel_active_jobs_reports_failure_to_automl_before_cleanup(tmp_path):
    from tao_automl.runner import AutoMLRunner

    rec = _retention_rec(7, "job-active", "pending")

    class CancelAutoML(_RetentionAutoML):
        def report_result(self, rec_id, metric_value, status):
            assert rec_id == rec.id
            rec.update_result(metric_value)
            rec.update_status(status)

        def is_complete(self):
            return rec.status == "failure"

    workspace = tmp_path / "workspace"

    class OrderingSDK(_ArtifactSDK):
        def cancel_job(self, job_id):
            persisted = json.loads((workspace / "active_jobs.json").read_text())
            assert persisted[0]["cancel_requested"] is True
            assert rec.status == "pending"
            return super().cancel_job(job_id)

        def delete_job_artifacts(self, job_id):
            assert rec.status == "failure"
            return super().delete_job_artifacts(job_id)

    sdk = OrderingSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._workspace_path = str(workspace)
    runner._automl = CancelAutoML([rec])
    runner._active_jobs = {rec.id: rec.job_id}

    runner.cancel_active_jobs(reason="test persisted cancellation")

    assert rec.status == "failure"
    assert rec.failure_reason == "job_canceled"
    assert runner._automl.is_complete()
    assert runner._active_jobs == {}
    assert sdk.deleted == ["job-active"]


def test_cancel_active_jobs_converges_confirmed_orphan_and_reports_corruption(
    tmp_path, caplog,
):
    from tao_automl.runner import AutoMLRunner

    class MissingRecAutoML(_RetentionAutoML):
        def report_result(self, rec_id, metric_value, status):
            raise KeyError(rec_id)

    sdk = _ArtifactSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    runner._workspace_path = str(tmp_path / "workspace")
    runner._automl = MissingRecAutoML([])
    runner._active_jobs = {7: "job-orphan"}

    runner.cancel_active_jobs(reason="test orphan cancellation")

    assert sdk.canceled == ["job-orphan"]
    assert sdk.deleted == ["job-orphan"]
    assert runner._active_jobs == {}
    assert json.loads(
        (tmp_path / "workspace" / "active_jobs.json").read_text()
    ) == []
    assert "AutoML state is corrupt" in caplog.text


def test_programmatic_run_registers_runner_and_cancels_on_exception(
    tmp_path, monkeypatch,
):
    import tao_automl.runner as runner_module
    from tao_automl.runner import AutoMLRunner
    from tao_automl.types import Recommendation

    class InterruptedAutoML:
        def __init__(self, *args, **kwargs):
            self.rec = Recommendation(0, {"train.num_epochs": 1}, "accuracy")
            self._state_store = MagicMock()
            self._state_store.get_job_specs.return_value = None

        def is_complete(self):
            return False

        def next_recommendation(self):
            return [self.rec]

        def get_progress(self):
            return {"completed": 0}

        def get_history(self):
            return [self.rec]

        def get_best(self):
            return None

        def report_result(self, rec_id, metric_value, status):
            assert rec_id == self.rec.id
            self.rec.update_result(metric_value)
            self.rec.update_status(status)

    def interrupt_job(self, *args, **kwargs):
        rec = kwargs["rec"]
        rec.assign_job_id("job-interrupted")
        self._active_jobs[rec.id] = rec.job_id
        self._persist_active_jobs(kwargs["workspace_path"])
        assert runner_module._runner is self
        assert handlers[runner_module.signal.SIGINT] is runner_module._signal_handler
        assert handlers[runner_module.signal.SIGTERM] is runner_module._signal_handler
        raise RuntimeError("orchestrator interrupted")

    host_sigint = object()
    host_sigterm = object()
    handlers = {
        runner_module.signal.SIGINT: host_sigint,
        runner_module.signal.SIGTERM: host_sigterm,
    }

    def fake_getsignal(signum):
        return handlers[signum]

    def fake_signal(signum, handler):
        previous = handlers[signum]
        handlers[signum] = handler
        return previous

    sdk = _ArtifactSDK()
    monkeypatch.setattr("tao_automl.AutoML", InterruptedAutoML)
    monkeypatch.setattr(AutoMLRunner, "_run_one_job", interrupt_job)
    monkeypatch.setattr(runner_module, "_runner", None)
    monkeypatch.setattr(runner_module.signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(runner_module.signal, "signal", fake_signal)
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))

    with pytest.raises(RuntimeError, match="orchestrator interrupted"):
        runner.run(
            automl_settings={
                "algorithm": "bayesian",
                "metric": "accuracy",
                "run_baseline": False,
            },
            workspace_path=str(tmp_path / "workspace"),
        )

    assert sdk.canceled == ["job-interrupted"]
    assert sdk.deleted == ["job-interrupted"]
    assert runner_module._runner is None
    assert handlers == {
        runner_module.signal.SIGINT: host_sigint,
        runner_module.signal.SIGTERM: host_sigterm,
    }


def test_signal_handler_defers_cleanup_until_interrupted_stack_unwinds(monkeypatch):
    import signal
    import tao_automl.runner as runner_module

    registered = MagicMock()
    monkeypatch.setattr(runner_module, "_runner", registered)

    with pytest.raises(SystemExit) as exc_info:
        runner_module._signal_handler(signal.SIGTERM, None)

    assert exc_info.value.code == 1
    assert registered._pending_signal == signal.SIGTERM
    registered.cancel_active_jobs.assert_not_called()


def test_import_and_construction_do_not_install_signal_handlers(tmp_path):
    import importlib
    import signal
    import tao_automl.runner as runner_module

    with patch.object(signal, "signal") as install_handler:
        runner_module = importlib.reload(runner_module)
        runner_module.AutoMLRunner(
            sdk=_ArtifactSDK(), skill_dir=_write_fake_skill(tmp_path)
        )

    install_handler.assert_not_called()


def test_recover_orphan_converges_then_raises_corrupt_state(tmp_path):
    from tao_automl.runner import AutoMLRunner

    sdk = _ArtifactSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=_write_fake_skill(tmp_path))
    runner._delete_intermediate_ckpt = True
    workspace = str(tmp_path / "workspace")
    runner._workspace_path = workspace
    runner._active_jobs = {17: "job-orphan"}
    assert runner._persist_active_jobs(workspace)

    with pytest.raises(RuntimeError, match="AutoML state is corrupt"):
        runner._recover_pending_job(
            entry={"rec_id": 17, "job_id": "job-orphan"},
            automl=_RetentionAutoML([]),
            metric_name="accuracy",
            metric_extractor=None,
            eval_fn=None,
            workspace_path=workspace,
            on_result=None,
            platform_kwargs={},
        )

    assert sdk.canceled == ["job-orphan"]
    assert sdk.deleted == ["job-orphan"]
    assert runner._active_jobs == {}
    assert json.loads(
        (tmp_path / "workspace" / "active_jobs.json").read_text()
    ) == []


def test_recover_pending_job_restores_job_id(tmp_path):
    from tao_automl.runner import AutoMLRunner
    from tao_automl.types import Recommendation

    class CompletedSDK:
        def get_job_logs(self, job_id, tail=None):
            return "accuracy: 0.7\n"

        def get_job_status(self, job_id):
            return SimpleNamespace(status="Complete")

    rec = Recommendation(4, {}, "accuracy")

    class RecoveringAutoML:
        def get_history(self):
            return [rec]

        def report_result(self, rec_id, metric_value, status):
            rec.update_result(metric_value)
            rec.update_status(status)

    runner = AutoMLRunner(
        sdk=CompletedSDK(), skill_dir=_write_fake_skill(tmp_path), poll_interval=0
    )
    runner._recover_pending_job(
        entry={"rec_id": 4, "job_id": "job-recovered"},
        automl=RecoveringAutoML(),
        metric_name="accuracy",
        metric_extractor=None,
        eval_fn=None,
        workspace_path=str(tmp_path / "workspace"),
        on_result=None,
        platform_kwargs={},
    )

    assert rec.job_id == "job-recovered"
    assert rec.status == "success"


def test_restored_cancel_intent_recovers_job_that_completed_first(tmp_path):
    from tao_automl.runner import AutoMLRunner
    from tao_automl.types import Recommendation

    class CompletedSDK:
        def __init__(self):
            self.cancel_calls = []
            self.deleted = []

        def get_job_logs(self, job_id, tail=None):
            return "accuracy: 0.75\n"

        def get_job_status(self, job_id):
            return SimpleNamespace(status="Complete")

        def cancel_job(self, job_id):
            self.cancel_calls.append(job_id)
            return False

        def delete_job_artifacts(self, job_id):
            self.deleted.append(job_id)
            return True

    rec = Recommendation(5, {}, "accuracy")

    class RecoveringAutoML:
        def get_history(self):
            return [rec]

        def report_result(self, rec_id, metric_value, status):
            rec.update_result(metric_value)
            rec.update_status(status)

    sdk = CompletedSDK()
    runner = AutoMLRunner(
        sdk=sdk,
        skill_dir=_write_fake_skill(tmp_path),
        poll_interval=0,
    )
    runner._recover_pending_job(
        entry={
            "rec_id": 5,
            "job_id": "job-completed-before-resume",
            "cancel_requested": True,
            "cancel_reason": "restored signal",
        },
        automl=RecoveringAutoML(),
        metric_name="accuracy",
        metric_extractor=None,
        eval_fn=None,
        workspace_path=str(tmp_path / "workspace"),
        on_result=None,
        platform_kwargs={},
    )

    assert rec.job_id == "job-completed-before-resume"
    assert rec.status == "success"
    assert rec.result == pytest.approx(0.75)
    assert sdk.cancel_calls == []
    assert sdk.deleted == []
    assert runner._active_jobs == {}


def test_runner_init_replaces_skillbank_with_skillcontext(tmp_path):
    """AutoMLRunner.__init__ no longer takes (sdk, poll_interval) only —
    skill_dir + action are now required."""
    from tao_automl.runner import AutoMLRunner, SkillContext
    skill_dir = _write_fake_skill(tmp_path)
    fake_sdk = MagicMock()
    runner = AutoMLRunner(sdk=fake_sdk, skill_dir=skill_dir, action="train")
    assert isinstance(runner.skill_ctx, SkillContext)
    assert runner.skill_ctx.network_arch == "fake-net"
    # Old API would've worked without skill_dir; new API is explicit.
    with pytest.raises(TypeError):
        AutoMLRunner(sdk=fake_sdk)  # missing skill_dir


def test_execution_parameter_is_appended_after_existing_positional_callbacks():
    """Adding Python execution must not rebind legacy positional arguments."""
    import inspect

    from tao_automl.runner import AutoMLRunner

    parameters = list(inspect.signature(AutoMLRunner.run).parameters)
    assert parameters.index("execution") > parameters.index("on_result")


def test_container_run_prefers_packaged_search_schema(tmp_path, monkeypatch):
    """A container skill's schema must drive search when the built-in cannot."""
    from types import SimpleNamespace

    from tao_automl.runner import AutoMLRunner
    from tao_automl.search_space.params import generate_hyperparams_to_search

    skill_dir = _write_fake_skill(tmp_path)
    schemas = skill_dir / "schemas"
    schemas.mkdir()
    packaged_schema = {
        "type": "object",
        "default": {"train": {"optim": {"lr": 2.0e-4}}},
        "properties": {
            "train": {
                "type": "object",
                "properties": {
                    "optim": {
                        "type": "object",
                        "properties": {
                            "lr": {
                                "type": "float",
                                "default": 2.0e-4,
                                "minimum": 1.0e-5,
                                "maximum": 1.0e-3,
                                "automl_enabled": True,
                            },
                        },
                    },
                },
            },
        },
    }
    (schemas / "train.schema.json").write_text(json.dumps(packaged_schema))

    # This represents a generated in-package schema that knows the parameter
    # but does not expose it to AutoML. Before the regression fix, container
    # execution passed search_schema=None and therefore selected this source.
    builtin_schema = {
        "type": "object",
        "default": {"train": {"optim": {"lr": 2.0e-4}}},
        "properties": {
            "train": {
                "type": "object",
                "properties": {
                    "optim": {
                        "type": "object",
                        "properties": {
                            "lr": {
                                "type": "float",
                                "default": 2.0e-4,
                                "minimum": 1.0e-5,
                                "maximum": 1.0e-3,
                                "automl_enabled": False,
                            },
                        },
                    },
                },
            },
        },
    }
    monkeypatch.setattr(
        "tao_automl.search_space.params.generate_schema",
        lambda *args, **kwargs: builtin_schema,
    )
    captured = {}
    best = SimpleNamespace(id=0, result=0.5, specs={})

    class CompleteAutoML:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            records, names = generate_hyperparams_to_search(
                network=kwargs["network"],
                action=kwargs["action"],
                train_specs=kwargs["train_specs"],
                automl_hyperparameters=kwargs["automl_hyperparameters"] or [],
                schema=kwargs["search_schema"],
            )
            captured["param_records"] = records
            captured["param_names"] = names

        def is_complete(self):
            return True

        def get_best(self):
            return best

        def get_progress(self):
            return {"completed": 0, "best_metric": 0.5}

        def get_history(self):
            return []

    monkeypatch.setattr("tao_automl.AutoML", CompleteAutoML)
    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    runner.run(
        automl_settings={"algorithm": "bayesian", "metric": "loss"},
        workspace_path=str(tmp_path / "workspace"),
    )

    assert captured["search_schema"] == packaged_schema
    assert captured["param_names"] == ["train.optim.lr"]
    assert captured["param_records"][0]["valid_min"] == 1.0e-5


# ---------------------------------------------------------------------------
# Within-trial checkpoint retention
# ---------------------------------------------------------------------------

def test_checkpoint_retention_disabled_leaves_specs_unchanged():
    from tao_automl.runner import _apply_checkpoint_retention_strategy

    specs = {
        "train": {
            "num_epochs": 4,
            "checkpoint_interval": 1,
            "checkpoint_interval_unit": "step",
            "checkpointer": {"enable_topk": False},
        }
    }
    before = json.loads(json.dumps(specs))

    effective = _apply_checkpoint_retention_strategy(
        specs,
        enabled=False,
        strategy="not-a-real-strategy",
        metric="val_loss",
        direction="minimize",
    )

    assert effective is None
    assert specs == before


@pytest.mark.parametrize("effective_epochs", [1, 2, 9])
def test_auto_terminal_retention_preserves_effective_epoch_budget(
    effective_epochs,
):
    from tao_automl.runner import _apply_checkpoint_retention_strategy

    specs = {
        "train": {
            "num_epochs": effective_epochs,
            "checkpoint_interval": 1,
            "checkpoint_interval_unit": "step",
        }
    }

    effective = _apply_checkpoint_retention_strategy(
        specs,
        enabled=True,
        strategy="auto",
        metric="val_loss",
        direction="minimize",
    )

    assert effective == "terminal"
    assert specs["train"]["num_epochs"] == effective_epochs
    assert specs["train"]["checkpoint_interval"] == effective_epochs
    assert specs["train"]["checkpoint_interval_unit"] == "epoch"


def test_auto_best_retention_injects_single_monitored_checkpoint():
    from tao_automl.runner import _apply_checkpoint_retention_strategy

    specs = {
        "train": {
            "num_epochs": 5,
            "checkpoint_interval": 1,
            "checkpointer": {"filename": "custom_best_{epoch:03d}"},
        }
    }

    effective = _apply_checkpoint_retention_strategy(
        specs,
        enabled=True,
        strategy="auto",
        metric="val_loss",
        direction="minimize",
    )

    assert effective == "best"
    assert specs["train"]["num_epochs"] == 5
    assert specs["train"]["checkpoint_interval"] == 1
    assert specs["train"]["checkpointer"] == {
        "filename": "custom_best_{epoch:03d}",
        "enable_topk": True,
        "replace_periodic": True,
        "monitor": "val_loss",
        "mode": "min",
        "save_top_k": 1,
    }


def test_explicit_best_retention_creates_checkpointer_config():
    from tao_automl.runner import _apply_checkpoint_retention_strategy

    specs = {"train": {"num_epochs": 3, "checkpoint_interval": 1}}

    effective = _apply_checkpoint_retention_strategy(
        specs,
        enabled=True,
        strategy="best",
        metric="val_acc",
        direction="maximize",
    )

    assert effective == "best"
    assert specs["train"]["checkpointer"] == {
        "enable_topk": True,
        "replace_periodic": True,
        "monitor": "val_acc",
        "mode": "max",
        "save_top_k": 1,
    }


def test_best_retention_preserves_trainer_declared_monitor():
    from tao_automl.runner import _apply_checkpoint_retention_strategy

    specs = {
        "train": {
            "num_epochs": 3,
            "checkpoint_interval": 1,
            "checkpointer": {"monitor": "val_loss", "mode": "min"},
        }
    }

    effective = _apply_checkpoint_retention_strategy(
        specs,
        enabled=True,
        strategy="best",
        metric="val_acc",
        direction="maximize",
    )

    assert effective == "best"
    assert specs["train"]["checkpointer"] == {
        "monitor": "val_loss",
        "mode": "min",
        "enable_topk": True,
        "replace_periodic": True,
        "save_top_k": 1,
    }


def test_checkpoint_retention_rejects_unknown_enabled_strategy():
    from tao_automl.runner import _apply_checkpoint_retention_strategy

    with pytest.raises(ValueError, match="must be one of"):
        _apply_checkpoint_retention_strategy(
            {"train": {"num_epochs": 3}},
            enabled=True,
            strategy="all",
            metric="val_loss",
            direction="minimize",
        )


@pytest.mark.parametrize(
    ("retention_enabled", "expected_interval", "expected_unit"),
    [(True, 2, "epoch"), (False, 1, "step")],
)
def test_runner_applies_retention_after_asha_budget_merge_and_honors_opt_out(
    tmp_path,
    monkeypatch,
    retention_enabled,
    expected_interval,
    expected_unit,
):
    from tao_automl.runner import AutoMLRunner
    from tao_automl.types import JobStates, Recommendation

    captured_specs = {}

    class OneRungAutoML:
        def __init__(self, *args, **kwargs):
            self.rec = Recommendation(
                0,
                {"train.num_epochs": 2, "train.optim.lr": 1.0e-5},
                "val_loss",
            )
            self.complete = False

        def is_complete(self):
            return self.complete

        def next_recommendation(self):
            return [self.rec]

        def report_result(self, rec_id, metric_value, status):
            self.rec.update_result(metric_value)
            self.rec.update_status(status)
            self.complete = True

        def get_best(self):
            return self.rec if self.rec.status == JobStates.success else None

        def get_progress(self):
            return {"completed": int(self.complete), "best_metric": self.rec.result}

        def get_history(self):
            return [self.rec]

        def get_required_checkpoint_job_ids(self):
            return set()

    def complete_job(self, *args, **kwargs):
        captured_specs.update(kwargs["specs"])
        return 0.25, "success"

    monkeypatch.setattr("tao_automl.AutoML", OneRungAutoML)
    monkeypatch.setattr(AutoMLRunner, "_run_one_job", complete_job)

    result = AutoMLRunner(
        sdk=MagicMock(), skill_dir=_write_fake_skill(tmp_path)
    ).run(
        image="nvcr.io/test:1",
        automl_settings={
            "algorithm": "asha",
            "metric": "val_loss",
            "direction": "minimize",
            "automl_delete_intermediate_ckpt": retention_enabled,
            "automl_checkpoint_retention_strategy": "terminal",
            "run_baseline": False,
            "run_final_evaluation": False,
        },
        spec_overrides={
            "train": {
                "num_epochs": 12,
                "checkpoint_interval": 1,
                "checkpoint_interval_unit": "step",
                "optim": {"lr": 2.0e-4},
            }
        },
        workspace_path=str(tmp_path / "workspace"),
    )

    assert result["best"]["metric_value"] == pytest.approx(0.25)
    assert captured_specs["train"]["num_epochs"] == 2
    assert captured_specs["train"]["checkpoint_interval"] == expected_interval
    assert captured_specs["train"]["checkpoint_interval_unit"] == expected_unit


# ---------------------------------------------------------------------------
# Output-directory collision guard
# ---------------------------------------------------------------------------

def test_auto_suffix_skips_nested_path_derived_from_declared_output():
    from tao_automl.runner import _auto_suffix_output_dirs

    specs = {
        "results_dir": "",
        "train": {"results_dir": "${results_dir}/train"},
        "export": {"output_dir": "/tmp/shared-export"},
        "logs": {"save_dir": "${shared_root}/logs"},
    }

    rewritten = _auto_suffix_output_dirs(
        specs,
        rec_id=7,
        declared_outputs={"results_dir"},
    )

    assert specs["train"]["results_dir"] == "${results_dir}/train"
    assert specs["export"]["output_dir"] == "/tmp/shared-export/rec_7"
    assert specs["logs"]["save_dir"] == "${shared_root}/logs/rec_7"
    assert rewritten == ["export.output_dir", "logs.save_dir"]


# ---------------------------------------------------------------------------
# Override merging
# ---------------------------------------------------------------------------

def test_merge_specs_deep_merges_dotted_keys():
    from tao_automl.runner import AutoMLRunner
    base = {"train": {"num_epochs": 12, "optim": {"lr": 2.0e-4}}}
    overrides = {"train.optim.lr": 5.0e-5, "train.num_gpus": 8}
    merged = AutoMLRunner._merge_specs(base, overrides)
    assert merged["train"]["num_epochs"] == 12       # untouched
    assert merged["train"]["optim"]["lr"] == 5.0e-5  # overridden
    assert merged["train"]["num_gpus"] == 8           # added


def test_merge_specs_deep_merges_nested_caller_spec():
    from tao_automl.runner import AutoMLRunner

    base = {
        "train": {
            "num_epochs": 12,
            "optim": {"lr": 2.0e-4, "monitor_name": "val_loss"},
            "checkpointer": {"monitor": "val_loss", "mode": "min"},
        }
    }
    overrides = {"train": {"num_epochs": 3, "optim": {"lr": 5.0e-5}}}

    merged = AutoMLRunner._merge_specs(base, overrides)

    assert merged["train"]["num_epochs"] == 3
    assert merged["train"]["optim"] == {
        "lr": 5.0e-5,
        "monitor_name": "val_loss",
    }
    assert merged["train"]["checkpointer"] == {
        "monitor": "val_loss",
        "mode": "min",
    }


def test_merge_specs_does_not_mutate_base():
    from tao_automl.runner import AutoMLRunner
    base = {"train": {"num_epochs": 12}}
    AutoMLRunner._merge_specs(base, {"train.num_epochs": 5})
    assert base["train"]["num_epochs"] == 12  # base stayed pristine


# ---------------------------------------------------------------------------
# Resume checkpoint handoff
# ---------------------------------------------------------------------------

def test_apply_resume_checkpoint_sets_training_checkpoint_path(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    template = skill_dir / "references/spec_template_train.yaml"
    template.write_text(
        "train:\n"
        "  num_epochs: 12\n"
        "  resume_training_checkpoint_path: ''\n"
        "dataset:\n"
        "  num_classes: 80\n"
    )

    results_root = tmp_path / "results"
    checkpoint = (
        results_root / "parent-job" / "results_dir" / "train" / "model_epoch_001.pth"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint")
    latest = checkpoint.parent / "classifier_model_latest.pth"
    latest.write_text("latest")

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    rec = MagicMock(id=2, resume_from_job_id="parent-job", resume_from_epoch=1)

    specs = {"train": {"resume_training_checkpoint_path": ""}}
    updated = runner._apply_resume_checkpoint(
        specs,
        rec,
        {"mounts": [{"host_path": str(results_root), "container_path": "/results"}]},
    )

    assert (
        updated["train"]["resume_training_checkpoint_path"]
        == "/results/parent-job/results_dir/train/model_epoch_001.pth"
    )
    assert rec.resume_checkpoint_path.endswith("model_epoch_001.pth")


def test_apply_resume_checkpoint_resolves_zero_indexed_bounded_best(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    (skill_dir / "references/spec_template_train.yaml").write_text(
        "train:\n"
        "  num_epochs: 12\n"
        "  resume_training_checkpoint_path: ''\n"
    )

    results_root = tmp_path / "results"
    checkpoint_dir = results_root / "parent-job" / "results_dir" / "train"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "model_best_000.pth"
    checkpoint.write_text("checkpoint")
    (checkpoint_dir / "changenet_model_classify_latest.pth").symlink_to(
        checkpoint.name
    )
    unrelated = results_root / "parent-job" / "inputs" / "model_epoch_000.pth"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("not a parent training output")

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    runner._terminal_job_ids["parent-job"] = "success"
    rec = MagicMock(
        id=2,
        resume_from_job_id="parent-job",
        resume_from_epoch=1,
        resume_from_step=None,
    )

    updated = runner._apply_resume_checkpoint(
        {"train": {"resume_training_checkpoint_path": ""}},
        rec,
        {"mounts": [{
            "host_path": str(results_root),
            "container_path": "/results",
        }]},
    )

    assert updated["train"]["resume_training_checkpoint_path"] == (
        "/results/parent-job/results_dir/train/model_best_000.pth"
    )
    assert rec.resume_checkpoint_path.endswith("model_best_000.pth")
    assert rec.resume_checkpoint_missing is False


def test_apply_resume_checkpoint_does_not_use_latest_for_requested_epoch(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    (skill_dir / "references/spec_template_train.yaml").write_text(
        "train:\n"
        "  resume_training_checkpoint_path: ''\n"
    )

    results_root = tmp_path / "results"
    checkpoint_dir = results_root / "parent-job" / "results_dir" / "train"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model_latest.pth").write_text("latest")

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    rec = MagicMock(id=2, resume_from_job_id="parent-job", resume_from_epoch=1)

    updated = runner._apply_resume_checkpoint(
        {"train": {"resume_training_checkpoint_path": ""}},
        rec,
        {"mounts": [{"host_path": str(results_root), "container_path": "/results"}]},
    )

    assert updated["train"]["resume_training_checkpoint_path"] == ""
    assert rec.resume_checkpoint_missing is True
    assert rec.resume_checkpoint_path is None


def test_apply_resume_checkpoint_sets_cosmos_resume_to_checkpoint_dir(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    template = skill_dir / "references/spec_template_train.yaml"
    template.write_text("train:\n  resume: false\n  epoch: 2\n")

    results_root = tmp_path / "results"
    checkpoint_dir = (
        results_root / "parent-job" / "train_output_dir" / "run1"
        / "checkpoints" / "epoch_1"
    )
    (checkpoint_dir / "policy").mkdir(parents=True)
    (checkpoint_dir / "policy" / "model_rank_0.pth").write_text("checkpoint")
    safetensor_dir = (
        results_root / "parent-job" / "train_output_dir" / "run1"
        / "safetensors" / "epoch_1"
    )
    safetensor_dir.mkdir(parents=True)
    (safetensor_dir / "adapter_model.safetensors").write_text("adapter")

    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    rec = MagicMock(id=3, resume_from_job_id="parent-job", resume_from_epoch=1)

    updated = runner._apply_resume_checkpoint(
        {"train": {"resume": False, "epoch": 2}},
        rec,
        {"mounts": [{"host_path": str(results_root), "container_path": "/results"}]},
    )

    assert (
        updated["train"]["resume"]
        == "/results/parent-job/train_output_dir/run1/checkpoints/epoch_1"
    )


def test_apply_resume_environment_enables_trusted_checkpoint_resume(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    rec = MagicMock(
        id=4,
        resume_from_job_id="parent-job",
        resume_checkpoint_path="/results/parent-job/train/checkpoint.pth",
    )

    updated = runner._apply_resume_environment(
        {"env_vars": {"WANDB_MODE": "disabled"}},
        rec,
    )

    assert updated["env_vars"]["WANDB_MODE"] == "disabled"
    assert updated["env_vars"]["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"


def test_apply_resume_environment_does_not_mutate_non_resume_kwargs(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    rec = MagicMock(id=5, resume_from_job_id=None)
    platform_kwargs = {"env_vars": {"WANDB_MODE": "disabled"}}

    assert runner._apply_resume_environment(platform_kwargs, rec) is platform_kwargs


def test_apply_resume_environment_ignores_missing_checkpoint_path(tmp_path):
    from tao_automl.runner import AutoMLRunner

    skill_dir = _write_fake_skill(tmp_path)
    runner = AutoMLRunner(sdk=MagicMock(), skill_dir=skill_dir, action="train")
    rec = MagicMock(
        id=6,
        resume_from_job_id="parent-job",
        resume_checkpoint_path=None,
    )
    platform_kwargs = {"env_vars": {"WANDB_MODE": "disabled"}}

    assert runner._apply_resume_environment(platform_kwargs, rec) is platform_kwargs
