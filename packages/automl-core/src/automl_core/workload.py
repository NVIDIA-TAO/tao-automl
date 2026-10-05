# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declarative adapter for programs that accept a config and emit metrics JSON."""

import json
from pathlib import Path
from typing import Literal

import tomli_w
import yaml
from pydantic import Field, model_validator

from automl_core.contracts import Contract, Result, TrialPlan, lookup


def read_manifest(path):
    return yaml.safe_load(Path(path).read_text())


def encode_config(config, format):
    if format == "toml":
        return tomli_w.dumps(config)
    if format == "yaml":
        return yaml.safe_dump(config, sort_keys=False)
    return json.dumps(config, indent=2, allow_nan=False)


def local_artifact(trial, name):
    path = (trial / name).resolve()
    if not path.is_relative_to(trial.resolve()) or not path.exists():
        raise ValueError(f"Missing or out-of-trial artifact: {name}")
    return path


class CommandWorkload(Contract):
    cwd: Path
    commands: list[list[str]] = Field(min_length=1)
    config_format: Literal["json", "yaml", "toml"] = "json"
    metrics_file: str
    artifacts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_artifacts(self):
        for name in [self.metrics_file, *self.artifacts]:
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Metric and artifact paths must be trial-relative")
        return self

    def identity(self):
        return {"adapter": "command-v1", **self.model_dump(mode="json")}

    def validate(self, config):
        encode_config(config, self.config_format)

    def plan(self, config, trial, execution):
        config_path = trial / f"config.{self.config_format}"
        substitutions = {
            "{python}": execution.python,
            "{config}": str(config_path),
            "{trial}": str(trial),
        }

        def expand(arg):
            for key, value in substitutions.items():
                arg = arg.replace(key, value)
            return arg

        return TrialPlan(
            commands=[[expand(arg) for arg in command] for command in self.commands],
            cwd=self.cwd,
            files={config_path.name: encode_config(config, self.config_format)},
        )

    def collect(self, trial, objective):
        metrics = json.loads(local_artifact(trial, self.metrics_file).read_text())
        value = lookup(metrics, objective.metric)
        if type(value) not in (int, float):
            raise ValueError("Objective must be a finite JSON number")
        return Result(
            metric=value,
            artifacts={
                name: str(local_artifact(trial, name)) for name in self.artifacts
            },
        )
