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
from urllib.parse import urlsplit

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
    absolute_deadline: float | None = Field(default=None, gt=0)

    def deadline(self, created_at):
        elapsed = created_at + self.total_timeout_seconds
        return (
            min(elapsed, self.absolute_deadline) if self.absolute_deadline else elapsed
        )


Algorithm = Literal[
    "random",
    "bayesian",
    "bfbo",
    "hyperband",
    "hyperband_es",
    "bohb",
    "asha",
    "dehb",
    "pbt",
    "llm",
    "hybrid",
    "autoresearch",
]
FAMILY = {"hyperband", "hyperband_es", "bohb", "asha", "dehb", "pbt"}


class Fidelity(Contract):
    """Cumulative workload units; the workload owns their native interpretation."""

    minimum: int = Field(gt=0, strict=True)
    maximum: int = Field(gt=0, strict=True)
    reduction_factor: int = Field(default=3, ge=2, strict=True)

    @model_validator(mode="after")
    def valid_range(self):
        if self.minimum >= self.maximum:
            raise ValueError("Fidelity minimum must be smaller than maximum")
        return self

    def rungs(self):
        values = [self.minimum]
        while values[-1] < self.maximum:
            values.append(min(self.maximum, values[-1] * self.reduction_factor))
        return values


class AlgorithmOptions(Contract):
    kappa: float = Field(default=2.0, gt=0)
    kappa_decay: float = Field(default=0.95, gt=0, le=1)
    kappa_min: float = Field(default=0.5, gt=0)
    penalization_radius: float = Field(default=0.1, gt=0)
    kde_samples: int = Field(default=64, ge=2, strict=True)
    min_points_in_model: int = Field(default=10, ge=2, strict=True)
    top_fraction: float = Field(default=0.15, gt=0, lt=1)
    mutation_factor: float = Field(default=0.5, gt=0, le=2)
    crossover_probability: float = Field(default=0.5, gt=0, le=1)
    population_size: int = Field(default=4, ge=2, strict=True)
    perturbation_factor: float = Field(default=1.2, gt=1)
    exploit_fraction: float = Field(default=0.25, gt=0, le=0.5)
    min_curve_points: int = Field(default=3, ge=3, strict=True)
    curve_confidence: float = Field(default=0.8, ge=0, le=1)
    early_stop_margin: float = Field(default=0.05, ge=0)


class ResearchPhase(Contract):
    name: str = Field(min_length=1)
    algorithm: Algorithm
    trials: int = Field(gt=0, strict=True)
    domains: dict[str, Parameter] = Field(default_factory=dict)
    carry_forward: Literal["best", "base"] = "best"

    @model_validator(mode="after")
    def numerical_phase(self):
        if self.algorithm in {"llm", "hybrid", "autoresearch"}:
            raise ValueError(
                "Research program phases must use non-recursive numerical algorithms"
            )
        return self


def narrow_domains(original, requested):
    """Validate phase domains at planning and again for untrusted LLM proposals."""
    if not isinstance(requested, dict) or set(requested) - original.keys():
        raise ValueError("Hybrid domains must use declared parameter names")
    domains = dict(original)
    for name, raw in requested.items():
        narrowed, outer = Parameter.model_validate(raw), original[name]
        if narrowed.kind != outer.kind or narrowed.log != outer.log:
            raise ValueError("Hybrid cannot change domain type or scale")
        if outer.kind == "categorical":
            if not {canonical(v) for v in narrowed.choices} <= {
                canonical(v) for v in outer.choices
            }:
                raise ValueError("Hybrid cannot add categorical choices")
        elif narrowed.low < outer.low or narrowed.high > outer.high:
            raise ValueError("Hybrid cannot expand numeric bounds")
        domains[name] = narrowed
    return domains


class LLMSettings(Contract):
    """Explicit controller-side API and spend envelope; never credential values."""

    endpoint: str
    model: str = Field(min_length=1)
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    max_calls: int = Field(gt=0, strict=True)
    max_output_tokens: int = Field(gt=0, strict=True)
    max_prompt_chars: int = Field(default=24000, ge=1000, strict=True)
    timeout_seconds: float = Field(default=60, gt=0, le=300)
    temperature: float = Field(default=0.2, ge=0, le=2)
    research_instructions: str = Field(default="", max_length=8000)
    phase_trials: int = Field(default=3, ge=1, strict=True)
    research_program: list[ResearchPhase] = Field(default_factory=list)
    phase_algorithms: list[Algorithm] = Field(
        default_factory=lambda: ["random", "bayesian", "bfbo"]
    )

    @model_validator(mode="after")
    def valid_endpoint(self):
        url = urlsplit(self.endpoint)
        if (
            url.scheme not in {"https", "http"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("LLM endpoint must be a credential-free HTTP(S) base URL")
        if url.scheme == "http" and url.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ValueError("Use HTTPS for non-loopback LLM endpoints")
        if not self.phase_algorithms or set(self.phase_algorithms) & {
            "llm",
            "hybrid",
            "autoresearch",
        }:
            raise ValueError(
                "Hybrid phases must select non-recursive numerical algorithms"
            )
        return self


class RecoveryPolicy(Contract):
    """Extra execution attempts, not extra algorithm observations or API calls."""

    max_retries_per_trial: int = Field(default=0, ge=0, le=10, strict=True)
    max_retries_total: int = Field(default=0, ge=0, strict=True)
    backoff_seconds: float = Field(default=30, ge=0, le=300)
    cancellation_timeout_seconds: float = Field(default=120, gt=0, le=600)
    heartbeat_timeout_seconds: float | None = Field(default=None, ge=15)
    progress_timeout_seconds: float | None = Field(default=None, ge=15)

    @model_validator(mode="after")
    def bounded_retries(self):
        if bool(self.max_retries_per_trial) != bool(self.max_retries_total):
            raise ValueError(
                "Enable retries with both per-trial and total attempt caps"
            )
        return self


class Search(Contract):
    schema_version: Literal[1] = 1
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    algorithm: Algorithm = "bayesian"
    seed: int = Field(ge=0, le=2**32 - 1, strict=True)
    objective: Objective
    budget: Budget
    base_config: dict[str, Any]
    parameters: dict[str, Parameter] = Field(min_length=1)
    provenance: dict[str, str] = Field(default_factory=dict)
    stop_on_failure: bool = False
    recovery: RecoveryPolicy = Field(default_factory=RecoveryPolicy)
    pause_after_trials: list[int] = Field(default_factory=list)
    fidelity: Fidelity | None = None
    options: AlgorithmOptions = Field(default_factory=AlgorithmOptions)
    llm: LLMSettings | None = None

    @model_validator(mode="after")
    def valid_paths(self):
        if self.llm and self.llm.research_program:
            if self.algorithm != "hybrid":
                raise ValueError(
                    "Explicit research programs use the hybrid phase controller"
                )
            if (
                sum(phase.trials for phase in self.llm.research_program)
                > self.budget.max_trials
            ):
                raise ValueError(
                    "Research program exceeds the experiment trial envelope"
                )
            if any(
                phase.algorithm not in self.llm.phase_algorithms
                or phase.trials > self.llm.phase_trials
                for phase in self.llm.research_program
            ):
                raise ValueError(
                    "Research program exceeds reviewed phase algorithms/budgets"
                )
            for phase in self.llm.research_program:
                narrow_domains(self.parameters, phase.domains)
                if (
                    phase.algorithm == "pbt"
                    and phase.trials < self.options.population_size
                ):
                    raise ValueError("A PBT phase must cover at least one population")
        needs_fidelity = self.algorithm in FAMILY or (
            self.algorithm == "hybrid"
            and self.llm is not None
            and bool(set(self.llm.phase_algorithms) & FAMILY)
        )
        if needs_fidelity and self.fidelity is None:
            raise ValueError("This algorithm requires an explicit fidelity contract")
        if self.fidelity is not None and not needs_fidelity:
            raise ValueError(
                "Fidelity applies to resource-allocation algorithms, not fixed-budget search"
            )
        if (self.algorithm in {"llm", "hybrid", "autoresearch"}) != (
            self.llm is not None
        ):
            raise ValueError(
                "LLM algorithms require explicit endpoint, model and call/token limits; numerical modes do not use llm settings"
            )
        if (
            self.algorithm == "pbt"
            and self.budget.max_trials < self.options.population_size
        ):
            raise ValueError("PBT budget must cover at least one population")
        if any(
            type(n) is not int or n < 1 or n >= self.budget.max_trials
            for n in self.pause_after_trials
        ):
            raise ValueError(
                "Pause boundaries must be positive trial counts below max_trials"
            )
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


class SlurmSettings(Contract):
    """Single-node, shared-filesystem SLURM. No site-specific defaults."""

    partition: str = Field(pattern=r"^[A-Za-z0-9_-]+(?:,[A-Za-z0-9_-]+)*$")
    account: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    qos: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]+$")
    time_limit_seconds: int = Field(gt=0, strict=True)
    cpus_per_task: int = Field(gt=0, strict=True)
    gpus: int = Field(ge=0, strict=True)
    memory_mb: int | None = Field(default=None, gt=0, strict=True)
    exclusive: bool = False
    startup_timeout_seconds: int = Field(gt=0, strict=True)
    cleanup_seconds: int = Field(gt=0, strict=True)
    poll_seconds: float = Field(default=30, ge=1, le=300)
    submission_reconcile_seconds: float = Field(default=120, ge=1, le=900)
    # A reviewed consumer template can provide its existing Pyxis envelope.
    template: Path | None = None
    bootstrap: Path | None = None
    supporting_files: dict[str, str] = Field(default_factory=dict)
    image_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    # Optional consumer-owned record writer. Values are argv templates, not shell.
    record_open: list[str] = Field(default_factory=list)
    # Optional retry-specific open hook; {retry_of} is the previous record ID.
    record_retry_open: list[str] = Field(default_factory=list)
    record_mark: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_slurm(self):
        if self.time_limit_seconds % 60:
            raise ValueError(
                "SLURM time_limit_seconds must be a whole number of minutes"
            )
        if (
            self.startup_timeout_seconds + self.cleanup_seconds
            >= self.time_limit_seconds
        ):
            raise ValueError(
                "Reserve positive workload time inside the SLURM allocation"
            )
        if bool(self.record_open) != bool(self.record_mark):
            raise ValueError("Supply both record_open and record_mark, or neither")
        if self.record_retry_open and not self.record_open:
            raise ValueError("record_retry_open requires the ordinary record hooks")
        for path, digest in self.supporting_files.items():
            if not Path(path).is_absolute() or not re.fullmatch(
                r"[a-f0-9]{64}", digest
            ):
                raise ValueError(
                    "SLURM supporting_files need absolute paths and SHA256 digests"
                )
        for path in (self.template, self.bootstrap):
            if path is not None and str(path) not in self.supporting_files:
                raise ValueError(
                    "Seal the template and bootstrap in slurm.supporting_files"
                )
        return self


class SharedVolume(Contract):
    """Same absolute path on controller and worker; no implicit data transfer."""

    path: Path
    source: str = Field(min_length=1)
    sub_path: str = ""
    read_only: bool = True

    @model_validator(mode="after")
    def valid_volume(self):
        if (
            not self.path.is_absolute()
            or self.path == Path("/")
            or ".." in self.path.parts
        ):
            raise ValueError(
                "Volume paths must be absolute, non-root paths without '..'"
            )
        if Path(self.sub_path).is_absolute() or ".." in Path(self.sub_path).parts:
            raise ValueError("Volume sub_path must stay within its source")
        return self


class ManagedSettings(Contract):
    volumes: list[SharedVolume] = Field(min_length=1)
    gpus: int = Field(ge=0)
    poll_seconds: float = Field(default=5, ge=0.1, le=60)
    submission_reconcile_seconds: float = Field(default=120, ge=1, le=600)

    @model_validator(mode="after")
    def distinct_volumes(self):
        paths = [v.path for v in self.volumes]
        if any(
            a.is_relative_to(b)
            for i, a in enumerate(paths)
            for j, b in enumerate(paths)
            if i != j
        ):
            raise ValueError("Shared volume paths must not overlap")
        if not any(not v.read_only for v in self.volumes):
            raise ValueError("Declare a persistent writable output volume")
        return self


class KubernetesSettings(ManagedSettings):
    context: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    cpus: int = Field(ge=1)
    memory_gib: int = Field(ge=1)
    shared_memory_gib: int = Field(default=8, ge=1)
    gpu_resource: str = "nvidia.com/gpu"
    service_account: str = Field(min_length=1)
    image_pull_secrets: list[str] = Field(default_factory=list)
    node_selector: dict[str, str] = Field(default_factory=dict)
    # Secret references, never values. Each is {name: <Secret>, key: <key>}.
    secrets: dict[str, dict[str, str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def resource_names(self):
        names = [
            self.namespace,
            self.service_account,
            *self.image_pull_secrets,
            *(v.source for v in self.volumes),
            *(ref.get("name", "") for ref in self.secrets.values()),
        ]
        if any(
            not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?", name)
            for name in names
        ):
            raise ValueError(
                "Kubernetes object names must be explicit DNS-style names, never CLI options"
            )
        return self


class BrevSettings(ManagedSettings):
    instance: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    # Explicit stable Docker daemon identity from the chosen existing instance.
    daemon_id: str = Field(min_length=1)
    gpu_devices: list[int] = Field(default_factory=list)
    cpus: int = Field(ge=1)
    memory_gib: int = Field(ge=1)
    shared_memory_gib: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def devices(self):
        if len(set(self.gpu_devices)) != self.gpus or any(
            n < 0 for n in self.gpu_devices
        ):
            raise ValueError(
                "Brev requires exactly gpus distinct nonnegative device indices"
            )
        if any(v.source != str(v.path) or v.sub_path for v in self.volumes):
            raise ValueError("Brev bind sources must equal the shared absolute paths")
        return self


class LeptonSettings(ManagedSettings):
    api_url: str
    token_env: str = "LEPTON_WORKSPACE_TOKEN"
    resource_shape: str = Field(min_length=1)
    node_groups: list[str] = Field(min_length=1)
    image_pull_secrets: list[str] = Field(default_factory=list)
    shared_memory_mib: int = Field(default=8192, ge=1)
    secrets: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def endpoint(self):
        url = urlsplit(self.api_url)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or not url.path.endswith("/api/v1")
        ):
            raise ValueError(
                "Lepton api_url must be an explicit HTTPS workspace /api/v1 URL without credentials"
            )
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.token_env):
            raise ValueError("Lepton token_env must be an environment variable name")
        if any(not v.source.startswith("node-nfs:") for v in self.volumes):
            raise ValueError("Lepton requires persistent shared node-nfs volumes")
        return self


class Execution(Contract):
    kind: Literal["venv", "docker", "slurm", "kubernetes", "brev", "lepton"]
    python: str = Field(min_length=1)
    image: str | None = None
    mounts: list[Path] = Field(default_factory=list)
    environment: list[str] = Field(default_factory=list)
    gpu_devices: str | None = None
    slurm: SlurmSettings | None = None
    kubernetes: KubernetesSettings | None = None
    brev: BrevSettings | None = None
    lepton: LeptonSettings | None = None

    @property
    def managed(self):
        return self.kind in {"slurm", "kubernetes", "brev", "lepton"}

    @property
    def poll_seconds(self):
        return getattr(self, self.kind).poll_seconds if self.managed else 0.2

    @model_validator(mode="after")
    def valid_execution(self):
        for name in ("slurm", "kubernetes", "brev", "lepton"):
            if getattr(self, name) is not None and self.kind != name:
                raise ValueError(f"{name} settings apply only to {name} execution")
        if self.kind in {"kubernetes", "brev", "lepton"}:
            settings = getattr(self, self.kind)
            if (
                settings is None
                or not self.image
                or not Path(self.python).is_absolute()
            ):
                raise ValueError(
                    "Managed containers require platform settings, an explicit image and absolute python"
                )
            if self.mounts or self.gpu_devices is not None:
                raise ValueError(
                    "Declare managed volumes and GPU resources in the platform settings"
                )
            if any(
                n in {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"}
                for n in self.environment
            ):
                raise ValueError("Managed GPU visibility belongs to the executor")
            if self.kind != "brev":
                if not re.fullmatch(r"[^\s@]+@sha256:[a-f0-9]{64}", self.image):
                    raise ValueError(
                        "Kubernetes and Lepton require an immutable image@sha256:digest"
                    )
                if set(self.environment) != set(settings.secrets):
                    raise ValueError(
                        "Each forwarded environment name needs a platform Secret reference"
                    )
            if self.kind == "kubernetes" and any(
                set(ref) != {"name", "key"} or not all(ref.values())
                for ref in settings.secrets.values()
            ):
                raise ValueError("Kubernetes Secret references require name and key")
        if self.kind == "slurm":
            if any(
                key == "CUDA_VISIBLE_DEVICES" or key.startswith(("SBATCH_", "SLURM_"))
                for key in self.environment
            ):
                raise ValueError(
                    "SLURM allocation variables must come from the scheduler, not environment forwarding"
                )
            if self.slurm is None or self.gpu_devices is not None:
                raise ValueError(
                    "SLURM requires slurm settings and scheduler-owned GPU visibility"
                )
            if self.image and (
                not Path(self.image).is_absolute()
                or not self.image.endswith(".sqsh")
                or not self.slurm.image_sha256
                or not self.slurm.template
            ):
                raise ValueError(
                    "SLURM containers require an absolute cached .sqsh, SHA256 and sealed template"
                )
            if not self.image and (
                self.mounts or self.slurm.template or self.slurm.image_sha256
            ):
                raise ValueError(
                    "Native SLURM execution does not use container mounts/template/image hash"
                )
            if not Path(self.python).is_absolute():
                raise ValueError(
                    "SLURM workload python must be an absolute execution-frame path"
                )
        elif self.slurm is not None:
            raise ValueError("slurm settings apply only to SLURM execution")
        elif self.kind in {"venv", "docker"} and (self.kind == "docker") != bool(
            self.image
        ):
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
    read_only_mounts: list[Path] = Field(default_factory=list)
    # Command-indexed structured progress files, relative to the attempt root.
    # The worker never mistakes its own heartbeat or arbitrary log output for progress.
    progress_files: dict[int, list[str]] = Field(default_factory=dict)

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
        for index, names in self.progress_files.items():
            if index < 0 or index >= len(self.commands) or not names:
                raise ValueError("Progress watches must name a valid command and files")
            for name in names:
                if not name or Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError("Progress files must remain within the attempt")
        return self


class Result(Contract):
    metric: float
    artifacts: dict[str, str] = Field(default_factory=dict)


class Workload(Protocol):
    def identity(self) -> dict: ...
    def validate(self, config: dict) -> None: ...
    def plan(self, config: dict, trial: Path, execution: Execution) -> TrialPlan: ...
    def collect(self, trial: Path, objective: Objective) -> Result: ...
