# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned, explicit contracts. Configuration at workload boundaries is nested."""

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def lookup(config, path):
    value = config
    for part in path.split("."):
        value = value[part]
    return value


def override(config, values):
    result = copy.deepcopy(config)
    for path, value in values.items():
        parts = path.split(".")
        parent = result
        for part in parts[:-1]:
            parent = parent[part]
        if parts[-1] not in parent:
            raise ValueError(f"Unknown configuration field: {path}")
        parent[parts[-1]] = value
    return result


class Parameter(Contract):
    kind: Literal["float", "int", "categorical"]
    low: float | None = Field(default=None, strict=True)
    high: float | None = Field(default=None, strict=True)
    log: bool = False
    choices: list[Any] | None = None

    @model_validator(mode="after")
    def valid_domain(self):
        if self.kind == "categorical":
            if (
                not self.choices
                or self.low is not None
                or self.high is not None
                or self.log
            ):
                raise ValueError("Categorical parameters need choices only")
            if any(type(x) not in (str, int, float, bool) for x in self.choices):
                raise ValueError("Choices must be scalar strings, numbers or booleans")
            if len({canonical(x) for x in self.choices}) != len(self.choices):
                raise ValueError("Choices must be unique")
        elif (
            self.choices is not None
            or self.low is None
            or self.high is None
            or self.low >= self.high
        ):
            raise ValueError("Numeric parameters need low < high, without choices")
        elif self.log and self.low <= 0:
            raise ValueError("Log-scale bounds must be positive")
        elif self.kind == "int" and (
            not self.low.is_integer() or not self.high.is_integer()
        ):
            raise ValueError("Integer bounds must be integral")
        canonical(self.model_dump())
        return self

    def decode(self, value):
        if self.kind == "categorical":
            return self.choices[
                min(int(value * len(self.choices)), len(self.choices) - 1)
            ]
        if self.log:
            decoded = math.exp(
                math.log(self.low) + value * (math.log(self.high) - math.log(self.low))
            )
        else:
            decoded = self.low + value * (self.high - self.low)
        return round(decoded) if self.kind == "int" else decoded


class Objective(Contract):
    metric: str = Field(min_length=1)
    direction: Literal["minimize", "maximize"]


class Budget(Contract):
    max_trials: int = Field(gt=0, strict=True)
    trial_timeout_seconds: float = Field(gt=0)
    total_timeout_seconds: float = Field(gt=0)


class Search(Contract):
    schema_version: Literal[1] = 1
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    algorithm: Literal["random", "bayesian"] = "bayesian"
    seed: int = Field(ge=0, le=2**32 - 1, strict=True)
    objective: Objective
    budget: Budget
    base_config: dict[str, Any]
    parameters: dict[str, Parameter] = Field(min_length=1)
    provenance: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_paths(self):
        canonical(self.base_config)
        for path in self.parameters:
            if not re.fullmatch(r"[A-Za-z_][\w-]*(\.[A-Za-z_][\w-]*)*", path):
                raise ValueError(f"Invalid parameter path: {path}")
            try:
                lookup(self.base_config, path)
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"Parameter is missing from base_config: {path}"
                ) from exc
            if any(other.startswith(path + ".") for other in self.parameters):
                raise ValueError(f"Overlapping parameter paths: {path}")
        return self


class Execution(Contract):
    kind: Literal["venv", "docker"]
    python: str = Field(min_length=1)
    image: str | None = None
    mounts: list[Path] = Field(default_factory=list)
    environment: list[str] = Field(default_factory=list)
    gpu_devices: str | None = None

    @model_validator(mode="after")
    def valid_execution(self):
        if (self.kind == "docker") != bool(self.image):
            raise ValueError(
                "Docker requires an explicit image; venv does not use an image"
            )
        if self.kind == "venv" and self.mounts:
            raise ValueError("Mounts apply to Docker execution")
        if any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in self.environment
        ):
            raise ValueError("Environment entries must be names, never NAME=value")
        return self


class TrialPlan(Contract):
    commands: list[list[str]] = Field(min_length=1)
    cwd: Path
    files: dict[str, str] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_plan(self):
        if any(
            not command or any(not isinstance(arg, str) or not arg for arg in command)
            for command in self.commands
        ):
            raise ValueError("Commands must be non-empty argv lists")
        for name in self.files:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Staged files must remain within the trial directory")
        return self


class Result(Contract):
    metric: float
    artifacts: dict[str, str] = Field(default_factory=dict)


class Workload(Protocol):
    def identity(self) -> dict: ...
    def validate(self, config: dict) -> None: ...
    def plan(self, config: dict, trial: Path, execution: Execution) -> TrialPlan: ...
    def collect(self, trial: Path, objective: Objective) -> Result: ...
