# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AutoML runner: wires the tao_automl brain to an execution SDK for HPO.

The runner is platform-agnostic: it accepts the container platform SDKs
(Lepton/Slurm/Kubernetes/Docker/Brev) or VirtualEnvSDK for direct Python
scripts. The caller picks the runtime; the runner doesn't choose for them.

Usage::

    from pathlib import Path
    from tao_sdk.platforms.docker import DockerSDK
    from tao_automl.runner import AutoMLRunner

    sdk = DockerSDK()
    runner = AutoMLRunner(
        sdk=sdk,
        skill_dir=(Path.home() / "tao-skills-external/skills/models/"
                   "tao-finetune-cosmos-reason"),
        action="train",
    )
    result = runner.run(
        train_dataset_uri="s3://bucket/data/subset",
        automl_settings={
            "algorithm": "bayesian",
            "metric": "loss",
            "automl_max_recommendations": 5,
        },
    )
    print(result)

Or execute a plan file::

    python -m tao_automl.runner automl_plan.json --platform lepton
"""

import argparse
import copy
import difflib
import functools
import json
import logging
import math
import os
import re
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tao_automl.objectives import is_latency_metric, parse_objective_config
from tao_sdk.checkpoints import (
    build_checkpoint_candidate,
    checkpoint_epoch as sdk_checkpoint_epoch,
    select_checkpoint_path,
)
from tao_automl.utils.spec_utils import resolve_schema_leaf

logger = logging.getLogger(__name__)


# TAO NV-Panoptix3D Lightning checkpoints store the completed epoch using a
# one-based internal counter while their filenames remain zero-based.  Passing
# the next Hyperband-family resource budget verbatim therefore makes Lightning
# stop immediately after restore (for example, epoch-0 checkpoint +
# ``num_epochs=2`` executes no epoch-1 batches).  The terminal budget must be
# advanced by one for resumed jobs so the requested additional epoch actually
# runs and emits a fresh metric/checkpoint.
_RESUME_EPOCH_BUDGET_OFFSETS = {"nvpanoptix3d": 1}


# ---------------------------------------------------------------------------
# SkillContext — replaces the deleted SkillBank. Reads skill_info.yaml and
# spec_template_<action>.yaml directly from the skill bank dir, the same way
# agent launch scripts do per platform/tao-sdk/SKILL.md's "Constructing the
# spec / args" guidance.
# ---------------------------------------------------------------------------


@dataclass
class SkillContext:
    """Resolved skill metadata for a single (skill, action) pair.

    The runner used to call ``SkillBank.get_model_config(network_arch)`` and
    ``SkillBank.get_default_specs(network_arch, action)`` — both methods are
    gone. This class replaces both by loading directly from the skill bank
    layout that's documented and validated by tao-skills-external/scripts/
    validate-skills.sh.
    """
    skill_dir: Path
    action: str
    skill_info: dict[str, Any] = field(init=False)
    action_cfg: dict[str, Any] = field(init=False)
    default_specs: dict[str, Any] = field(init=False)
    schema: dict[str, Any] | None = field(init=False)
    valid_spec_keys: set[str] = field(init=False)
    container_image: str = field(init=False)
    network_arch: str = field(init=False)
    execution: "PythonScriptExecution | None" = field(init=False)

    def __post_init__(self):
        self.skill_dir = Path(self.skill_dir)
        info_path = self.skill_dir / "references/skill_info.yaml"
        if not info_path.exists():
            raise FileNotFoundError(
                f"skill_info.yaml not found at {info_path}. "
                f"skill_dir must point at a model directory containing "
                f"references/skill_info.yaml."
            )
        self.skill_info = yaml.safe_load(info_path.read_text()) or {}

        actions = self.skill_info.get("actions") or {}
        if self.action not in actions:
            raise KeyError(
                f"Action {self.action!r} not declared in {info_path}. "
                f"Available: {sorted(actions.keys())}"
            )
        self.action_cfg = actions[self.action]
        self.network_arch = self.skill_info.get("network_arch", self.skill_dir.name)

        template_path = self.skill_dir / f"references/spec_template_{self.action}.yaml"
        self.default_specs = (
            yaml.safe_load(template_path.read_text()) if template_path.exists() else {}
        ) or {}
        schema_path = self.skill_dir / f"schemas/{self.action}.schema.json"
        if schema_path.exists():
            with open(schema_path) as f:
                self.schema = json.load(f) or {}
            if not isinstance(self.schema, dict):
                raise TypeError(f"{schema_path}: schema must be a JSON object")
            if not template_path.exists():
                schema_defaults = self.schema.get("default")
                if isinstance(schema_defaults, dict):
                    self.default_specs = copy.deepcopy(schema_defaults)
            self.valid_spec_keys = _schema_property_keys(self.schema) | _flatten_keys(
                self.schema.get("default", {})
            ) | _flatten_keys(self.default_specs)
        else:
            self.schema = None
            self.valid_spec_keys = _flatten_keys(self.default_specs)

        self.execution = PythonScriptExecution.from_config(
            self.action_cfg.get("execution"),
            skill_dir=self.skill_dir,
            default_config_format=self.action_cfg.get("config_format", "yaml"),
        )
        if self.execution is not None and self.schema is None:
            raise FileNotFoundError(
                "python_script actions require an external AutoML schema at "
                f"{schema_path}"
            )
        if self.execution is not None:
            _validate_external_automl_schema(self.schema, schema_path)
            _validate_specs_against_schema(
                self.default_specs,
                self.schema,
                schema_path,
                require_all=False,
                source=str(template_path if template_path.exists() else "schema.default"),
            )

        # Container image: action-level image overrides win, then model-level.
        # Python-script actions do not require an image, so an omitted image is
        # valid and must not trigger versions.yaml resolution.
        image_ref = (
            self.action_cfg.get("container_image")
            or self.skill_info.get("container_image", "")
        )
        if self.execution is None and image_ref:
            from tao_sdk.versions import resolve_container_image
            self.container_image = resolve_container_image(image_ref)
        else:
            self.container_image = ""

    def validate_runtime(self) -> dict[str, Any]:
        """Validate that the model/action can be loaded by AutoML runtime code.

        Skill JSON can exist even when the runtime import path is broken. This
        probes the generated schema path used by ``AutoML`` construction, which
        catches issues such as ``cosmos-rl`` versus ``cosmos_rl`` package names
        before a long-running launch starts.
        """
        if self.execution is not None and self.schema is not None:
            schema = self.schema
        else:
            from tao_automl.schema.generate_schema import generate_schema
            schema = generate_schema(self.network_arch, self.action)
        return {
            "network_arch": self.network_arch,
            "action": self.action,
            "schema_title": schema.get("title"),
            "parameter_count": len(_schema_property_keys(schema)),
        }


@dataclass(frozen=True)
class PythonScriptExecution:
    """Direct Python action metadata resolved relative to a model skill."""

    script: Path
    script_args: tuple[str, ...]
    config_format: str
    cwd: Path

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | None,
        *,
        skill_dir: Path,
        default_config_format: str,
    ) -> "PythonScriptExecution | None":
        if config is None:
            return None
        if not isinstance(config, dict):
            raise TypeError("action execution must be a mapping")
        execution_type = config.get("type")
        if execution_type != "python_script":
            raise ValueError(
                f"Unsupported action execution type {execution_type!r}; "
                "expected 'python_script'."
            )

        script_value = config.get("script")
        if not isinstance(script_value, str) or not script_value.strip():
            raise ValueError("python_script execution requires a non-empty 'script' path")
        script = Path(script_value).expanduser()
        if not script.is_absolute():
            script = skill_dir / script
        script = script.resolve()
        if not script.is_file():
            raise FileNotFoundError(f"Python action script not found: {script}")

        args = config.get(
            "args",
            config.get("script_args", ["--config", "{config_path}"]),
        )
        if not isinstance(args, (list, tuple)) or not all(
            isinstance(arg, str) for arg in args
        ):
            raise TypeError("python_script execution 'args' must be a list of strings")

        config_format = config.get("config_format", default_config_format)
        if config_format not in {"json", "yaml", "toml"}:
            raise ValueError(
                "python_script config_format must be one of: json, yaml, toml"
            )

        cwd_value = config.get("cwd", ".")
        if not isinstance(cwd_value, str) or not cwd_value.strip():
            raise ValueError("python_script execution 'cwd' must be a non-empty path")
        cwd = Path(cwd_value).expanduser()
        if not cwd.is_absolute():
            cwd = skill_dir / cwd
        cwd = cwd.resolve()
        if not cwd.is_dir():
            raise NotADirectoryError(f"Python action cwd not found: {cwd}")

        return cls(
            script=script,
            script_args=tuple(args),
            config_format=config_format,
            cwd=cwd,
        )


def validate_skill_runtime(skill_dir: str | Path, action: str = "train") -> dict[str, Any]:
    """Load a skill directory and validate its AutoML runtime schema path."""
    return SkillContext(skill_dir=Path(skill_dir), action=action).validate_runtime()

_DEFAULT_POLL_INTERVAL = 30
_POLL_LOG_TAIL_LINES = 10_000
_TERMINAL_LOG_OVERLAP_CHARS = 8 * 1024
_TERMINAL_LOG_CONTEXT_CHARS = 64 * 1024
_SUCCESS_PLATFORM_STATUSES = frozenset({
    "complete", "completed", "success", "succeeded",
})
_FAILURE_PLATFORM_STATUSES = frozenset({"error", "failed", "failure"})
_CANCELED_PLATFORM_STATUSES = frozenset({
    "canceled", "cancelled", "stopped", "terminated",
})
_QUIESCENT_PLATFORM_STATUSES = frozenset({"deleted", "notfound", "not_found"})
_CANCEL_CONFIRM_TIMEOUT_SECONDS = 30.0
_CANCEL_CONFIRM_POLL_SECONDS = 0.5
_TERMINAL_REC_STATUSES = {"success", "done", "failure", "error", "canceled"}
_SUCCESS_REC_STATUSES = {"success", "done"}
_DEFER_ARTIFACT_PRUNING_ALGORITHMS = frozenset({
    "hyperband", "h", "bohb", "asha", "dehb", "hyperband_es", "hes", "pbt",
    "hybrid",
})
_CHECKPOINT_RETENTION_STRATEGIES = frozenset({"auto", "best", "terminal"})


def _bool_setting(value: Any) -> bool:
    """Parse bool-like values accepted in AutoML settings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _apply_checkpoint_retention_strategy(
    specs: dict[str, Any],
    *,
    enabled: bool,
    strategy: str,
    metric: str,
    direction: str,
) -> str | None:
    """Bound checkpoints written inside one AutoML trial.

    Whole-job artifact cleanup cannot reclaim periodic checkpoints inside the
    surviving trial.  This launch-time policy complements that cleanup without
    changing the recommendation's effective epoch budget:

    * ``best`` asks compatible trainers to keep one monitored checkpoint and
      replace periodic saving.
    * ``terminal`` makes the periodic interval equal the recommendation's
      effective ``train.num_epochs`` so only its terminal epoch is saved.
    * ``auto`` uses ``best`` only when the merged spec already exposes a
      ``train.checkpointer`` mapping; otherwise it uses the portable terminal
      fallback.

    Returns the effective strategy, or ``None`` when retention is disabled.
    ``specs`` is intentionally untouched in the disabled case.
    """
    if not enabled:
        return None

    normalized = str(strategy or "auto").strip().lower()
    if normalized not in _CHECKPOINT_RETENTION_STRATEGIES:
        raise ValueError(
            "automl_checkpoint_retention_strategy must be one of: "
            + ", ".join(sorted(_CHECKPOINT_RETENTION_STRATEGIES))
        )
    if not isinstance(specs, dict):
        raise TypeError("AutoML trial specs must be a dictionary")
    train = specs.get("train")
    if not isinstance(train, dict):
        raise ValueError(
            "checkpoint retention requires a train configuration mapping"
        )

    existing_checkpointer = train.get("checkpointer")
    if existing_checkpointer is not None and not isinstance(
        existing_checkpointer, dict
    ):
        raise TypeError("train.checkpointer must be a mapping when provided")

    effective = normalized
    if effective == "auto":
        effective = "best" if isinstance(existing_checkpointer, dict) else "terminal"

    if effective == "best":
        normalized_direction = str(direction).strip().lower()
        if normalized_direction in {"min", "minimize"}:
            checkpoint_mode = "min"
        elif normalized_direction in {"max", "maximize"}:
            checkpoint_mode = "max"
        else:
            raise ValueError(
                "best checkpoint retention requires objective direction "
                "'minimize' or 'maximize'"
            )
        checkpointer = copy.deepcopy(existing_checkpointer or {})
        # A model can expose an AutoML objective through status artifacts
        # without registering that key in Lightning's callback metrics (Visual
        # ChangeNet classification reports val_acc this way). Preserve the
        # trainer-declared monitor/mode and use the objective only as fallback.
        checkpointer.setdefault("monitor", metric)
        checkpointer.setdefault("mode", checkpoint_mode)
        checkpointer.update({
            "enable_topk": True,
            "replace_periodic": True,
            "save_top_k": 1,
        })
        train["checkpointer"] = checkpointer
    else:
        num_epochs = train.get("num_epochs")
        if (
            not isinstance(num_epochs, int)
            or isinstance(num_epochs, bool)
            or num_epochs < 1
        ):
            raise ValueError(
                "terminal checkpoint retention requires a positive integer "
                "train.num_epochs"
            )
        train["checkpoint_interval_unit"] = "epoch"
        train["checkpoint_interval"] = num_epochs

    return effective


def _confirmed_platform_status(status_result: Any) -> str | None:
    """Return a canonical terminal status, or ``None`` if still active.

    ``Unknown`` is never proof of quiescence: it commonly means the SDK's
    local recovery row is missing while a remote writer may still exist.
    Explicit backend-authenticated deletion statuses are treated as canceled.
    """
    raw_status = getattr(status_result, "status", status_result)
    if not isinstance(raw_status, str) or not raw_status.strip():
        return None
    normalized = raw_status.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in _SUCCESS_PLATFORM_STATUSES:
        return "Complete"
    if normalized in _FAILURE_PLATFORM_STATUSES:
        return "Error"
    if normalized in _CANCELED_PLATFORM_STATUSES:
        return "Canceled"
    if normalized in _QUIESCENT_PLATFORM_STATUSES:
        return "Canceled"
    return None


def _finite_metric(value: Any) -> float | None:
    """Convert a metric to float only when it is non-boolean and finite."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        metric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return metric if math.isfinite(metric) else None


def _require_finite_metric(value: Any, source: str) -> float:
    metric = _finite_metric(value)
    if metric is None:
        raise ValueError(
            f"{source} must be a finite numeric metric; booleans, NaN, and "
            "infinite values are not valid metrics"
        )
    return metric


def _callback_metric(value: Any, source: str) -> float | None:
    metric = _finite_metric(value)
    if value is not None and metric is None:
        logger.warning("%s returned an invalid non-finite or boolean metric: %r", source, value)
    return metric


def _callback_metric_payload(value: Any, source: str):
    """Validate a callback result containing one or multiple metrics."""
    if not isinstance(value, dict):
        return _callback_metric(value, source)
    if not value:
        logger.warning("%s returned an empty metric dictionary", source)
        return None
    metrics = {}
    for key, item in value.items():
        if not isinstance(key, str):
            logger.warning("%s returned a non-string metric key: %r", source, key)
            return None
        metric = _finite_metric(item)
        if metric is None:
            logger.warning(
                "%s returned an invalid non-finite or boolean metric for %s: %r",
                source,
                key,
                item,
            )
            return None
        metrics[key] = metric
    return metrics


def _iter_terminal_log_chunks(sdk, job_id: str):
    """Yield bounded terminal log snapshots, streaming when the SDK can."""
    stream_method = getattr(type(sdk), "iter_job_log_chunks", None)
    if callable(stream_method):
        try:
            yield from stream_method(sdk, job_id)
        except Exception as exc:
            logger.warning("Failed to stream terminal logs for job %s: %s", job_id, exc)
        return
    try:
        # Preserve existing behavior for container SDKs that do not expose
        # paged logs. VirtualEnvSDK always takes the bounded streaming path.
        logs = sdk.get_job_logs(job_id)
    except Exception:
        return
    if logs:
        yield logs


def _scan_terminal_logs(
    sdk,
    job_id: str,
    metric_name: str,
    extract_fn,
    cached_metric: float | None,
    cached_exec_status: str | None,
) -> tuple[float | None, str | None, str]:
    """Extract terminal signals without materializing an unbounded log."""
    cached_metric = _finite_metric(cached_metric)
    overlap = ""
    context = ""
    latest_explicit_status = None
    cleanup_failure_seen = False
    hard_failure_seen = False
    for chunk in _iter_terminal_log_chunks(sdk, job_id):
        if not isinstance(chunk, str) or not chunk:
            continue
        scan_text = overlap + chunk
        try:
            metric = extract_fn(scan_text, metric_name)
        except Exception as exc:
            logger.warning("metric_extractor raised for job %s: %s", job_id, exc)
            metric = None
        metric = _callback_metric(metric, "metric_extractor")
        if metric is not None:
            cached_metric = metric

        explicit_status = _latest_explicit_execution_status(scan_text)
        if explicit_status is not None:
            latest_explicit_status = explicit_status
        cleanup_failure_seen = cleanup_failure_seen or any(
            pattern in scan_text for pattern in _CLEANUP_FATAL_PATTERNS
        )
        hard_failure_seen = hard_failure_seen or _has_hard_failure_pattern(
            scan_text
        )

        context = (context + chunk)[-_TERMINAL_LOG_CONTEXT_CHARS:]
        overlap = scan_text[-_TERMINAL_LOG_OVERLAP_CHARS:]
    if latest_explicit_status is not None:
        cached_exec_status = latest_explicit_status
    elif hard_failure_seen or (cleanup_failure_seen and cached_metric is None):
        cached_exec_status = "FAIL"
    return cached_metric, cached_exec_status, context


_METRIC_NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def _scan_terminal_metric_values(
    sdk,
    job_id: str,
    metric_names: list[str],
    extract_fn,
    cached_metrics: dict[str, float],
    cached_exec_status: str | None,
) -> tuple[dict[str, float], str | None, str]:
    """Extract terminal signals for one or more objective metrics."""
    overlap = ""
    context = ""
    latest_explicit_status = None
    cleanup_failure_seen = False
    hard_failure_seen = False
    for chunk in _iter_terminal_log_chunks(sdk, job_id):
        if not isinstance(chunk, str) or not chunk:
            continue
        scan_text = overlap + chunk
        try:
            values = _extract_metric_values(scan_text, metric_names, extract_fn)
        except Exception as exc:
            logger.warning("metric_extractor raised for job %s: %s", job_id, exc)
            values = {}
        cached_metrics.update(values)

        explicit_status = _latest_explicit_execution_status(scan_text)
        if explicit_status is not None:
            latest_explicit_status = explicit_status
        cleanup_failure_seen = cleanup_failure_seen or any(
            pattern in scan_text for pattern in _CLEANUP_FATAL_PATTERNS
        )
        hard_failure_seen = hard_failure_seen or _has_hard_failure_pattern(scan_text)

        context = (context + chunk)[-_TERMINAL_LOG_CONTEXT_CHARS:]
        overlap = scan_text[-_TERMINAL_LOG_OVERLAP_CHARS:]
    if latest_explicit_status is not None:
        cached_exec_status = latest_explicit_status
    elif hard_failure_seen or (
        cleanup_failure_seen and metric_names[0] not in cached_metrics
    ):
        cached_exec_status = "FAIL"
    return cached_metrics, cached_exec_status, context

_COSMOS_RL_SFT_VAL_RE = re.compile(
    rf'\[SFT\]\s+Validation loss:\s*(?P<value>{_METRIC_NUMBER_RE})',
    re.IGNORECASE,
)

_CLEANUP_FATAL_PATTERNS = (
    "RendezvousTimeoutError",
    "RendezvousConnectionError",
    "DistNetworkError",
    "C10d store has failed",
    "Connection was likely closed. Did the remote server shutdown or crash?",
)

_HARD_FATAL_PATTERNS = (
    "failed with return code",
    "Process group watchdog thread terminated",
    "Watchdog caught collective operation timeout",
    "torch.distributed.elastic.multiprocessing.errors.ChildFailedError",
    "Signal 6 (SIGABRT)",
    "Process 1 failed with return code",
)


def _extract_metric_from_logs(
    logs: str,
    metric_name: str,
    *,
    allow_generic: bool = True,
) -> float | None:
    """Extract the final metric value from TAO training logs.

    Returns the globally latest matching finite value. Handles:
    - Cosmos-RL validation: "[SFT] Validation loss: 0.12 for train step ..."
    - Generic: "loss: 0.123" or "best loss: 0.123"
    - Cosmos-RL: "Step: 107/107, Loss: 8.27675, Grad norm: ..."
    - KPI: "kpi: 0.123"
    - Epoch: "Epoch 10 loss: 0.123"

    Returns None if no pattern matches. Many TAO PyTorch-Lightning containers
    emit metrics via RichProgressBar (ANSI-styled, not regex-friendly) and the
    parseable values live in ``<results_dir>/train/status.json`` as JSONL —
    in that case ``_read_metric_from_status_json`` is the right reader and the
    caller falls back to it.
    """
    if not logs:
        return None

    candidates: list[tuple[int, float]] = []

    # Cosmos-RL's specialized marker represents validation loss only. Treat it
    # as a position-aware alias, so it neither satisfies val_accuracy nor
    # overrides a newer named validation-loss value.
    normalized_metric = metric_name.lower().replace("/", "_").replace(" ", "_")
    if normalized_metric in {"val_loss", "validation_loss"}:
        for match in _COSMOS_RL_SFT_VAL_RE.finditer(logs):
            value = _finite_metric(match.group("value"))
            if value is not None:
                candidates.append((match.start("value"), value))

    # Direct metric matches are boundary-delimited so requesting ``loss`` does
    # not accidentally read ``val_loss`` or ``loss_scale``. ``\s*`` around the
    # delimiter still handles terminal wrapping without losing source offsets.
    metric_aliases = _metric_aliases(metric_name)
    for suffix in ("_epoch", "_step"):
        if not metric_name.endswith(suffix):
            metric_aliases.append(f"{metric_name}{suffix}")
    if metric_name.lower().startswith("val_"):
        bare_metric = metric_name[4:]
        metric_aliases.append("Validation " + bare_metric.replace("_", " "))
    for alias in dict.fromkeys(metric_aliases):
        escaped_alias = re.escape(alias).replace(r"\ ", r"\s+")
        metric_pattern = re.compile(
            rf'(?<![A-Za-z0-9_/\-])(?:best\s+)?{escaped_alias}'
            rf'(?![A-Za-z0-9_/\-])\s*[:=]\s*(?P<value>{_METRIC_NUMBER_RE})',
            re.IGNORECASE,
        )
        for match in metric_pattern.finditer(logs):
            value = _finite_metric(match.group("value"))
            if value is not None:
                candidates.append((match.start("value"), value))

    # ``kpi:`` is the generic TAO metric contract when no metric label is
    # emitted. Treat it as another candidate instead of giving it fixed
    # priority over newer named values.
    if allow_generic:
        kpi_pattern = re.compile(
            rf'(?<![A-Za-z0-9_/\-])kpi(?![A-Za-z0-9_/\-])\s*[:=]\s*'
            rf'(?P<value>{_METRIC_NUMBER_RE})',
            re.IGNORECASE,
        )
        for match in kpi_pattern.finditer(logs):
            value = _finite_metric(match.group("value"))
            if value is not None:
                candidates.append((match.start("value"), value))

    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _metric_aliases(metric_name: str) -> list[str]:
    """Return common TAO spellings for a metric name.

    Different TAO entrypoints report the same KPI as ``val/loss`` in
    ``status.json`` or ``val_loss`` in Lightning monitor fields. AutoML callers
    should not have to know that spelling difference to get a valid metric.
    """
    aliases = [metric_name]
    if "/" in metric_name:
        aliases.append(metric_name.replace("/", "_"))
    if "_" in metric_name:
        aliases.append(metric_name.replace("_", "/"))
    normalized = metric_name.lower().replace("/", "_")
    if normalized in {"map", "val_map"}:
        aliases.append("mAP")
        aliases.append("img_bbox_NuScenes/mAP")
    if normalized == "train_loss_epoch":
        aliases.append("train_loss")
    if normalized == "train_loss":
        aliases.append("train_loss_epoch")
    if normalized in {"avg_loss", "val_avg_loss"}:
        aliases.append("avg_loss")
    if is_latency_metric(metric_name):
        aliases.extend([
            "latency",
            "latency_ms",
            "inference_latency",
            "inference_latency_ms",
            "avg_latency",
            "avg_latency_ms",
            "average_latency",
            "average_latency_ms",
            "runtime",
            "runtime_ms",
            "duration",
            "duration_ms",
        ])
    seen = set()
    return [alias for alias in aliases if not (alias in seen or seen.add(alias))]


def _extract_metric_from_status_file(status_path: Path, metric_name: str) -> float | None:
    """Read the latest finite KPI value from a TAO line-delimited status file."""
    if not status_path.exists():
        return None
    aliases = _metric_aliases(metric_name)
    try:
        lines = status_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        kpi = payload.get("kpi")
        if not isinstance(kpi, dict):
            continue
        for alias in aliases:
            if alias not in kpi:
                continue
            value = _finite_metric(kpi[alias])
            if value is not None:
                return value
    return None


def _extract_metric_from_best_score_payload(
    payload: str | dict[str, Any],
    metric_name: str,
    *,
    allow_generic: bool = True,
) -> float | None:
    """Read TAO/Cosmos best-score artifacts.

    Cosmos-RL writes ``train_output_dir/best/best_score.json`` as a compact
    JSON object after validation. It is more reliable than log scraping when a
    later distributed failure truncates the useful log tail.
    """
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    aliases = _metric_aliases(metric_name)
    aliases.extend(alias.replace("/", "_") for alias in list(aliases))
    aliases = list(dict.fromkeys(aliases))
    requested_aliases = set(aliases)

    for key in aliases:
        if key not in data:
            continue
        value = _finite_metric(data[key])
        if value is not None:
            return value

    metric_label = data.get("metric")
    if isinstance(metric_label, str):
        label_aliases = [metric_label, metric_label.replace("/", "_")]
        label_matches = any(alias in requested_aliases for alias in label_aliases)
        label_matches = label_matches or (
            is_latency_metric(metric_name) and is_latency_metric(metric_label)
        )
        requested_normalized = metric_name.lower().replace("/", "_")
        label_normalized = metric_label.lower().replace("/", "_")
        label_matches = label_matches or (
            "loss" in requested_normalized and "loss" in label_normalized
        )
        if not label_matches:
            return None
        for key in dict.fromkeys(label_aliases):
            if key not in data:
                continue
            value = _finite_metric(data[key])
            if value is not None:
                return value

    if not allow_generic:
        return None

    for key in ("best_score", "best_metric", "metric_value", "score", "value"):
        if key not in data:
            continue
        value = _finite_metric(data[key])
        if value is not None:
            return value
    return None


def _extract_metric_from_best_score_file(
    best_score_path: Path,
    metric_name: str,
    *,
    allow_generic: bool = True,
) -> float | None:
    if not best_score_path.exists():
        return None
    try:
        return _extract_metric_from_best_score_payload(
            best_score_path.read_text(encoding="utf-8"),
            metric_name,
            allow_generic=allow_generic,
        )
    except OSError:
        return None


def _extract_metric_from_metrics_payload(
    payload: str | dict[str, Any], metric_name: str,
) -> float | None:
    """Read a requested metric from a direct script's ``metrics.json``."""
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    aliases = [metric_name, *_metric_aliases(metric_name)]
    aliases.extend(alias.replace("/", "_") for alias in list(aliases))
    for key in dict.fromkeys(aliases):
        if key not in data:
            continue
        value = _finite_metric(data[key])
        if value is not None:
            return value
    return None


def _extract_metric_from_metrics_file(
    metrics_path: Path, metric_name: str,
) -> float | None:
    try:
        return _extract_metric_from_metrics_payload(
            metrics_path.read_text(encoding="utf-8"), metric_name
        )
    except OSError:
        return None


def _extract_metric_from_local_results(
    job_id: str,
    metric_name: str,
    platform_kwargs: dict | None,
    *,
    allow_generic: bool = True,
) -> float | None:
    """Fallback for local runs whose metrics are written to result artifacts.

    The platform SDK mounts a host results directory at ``/results``. When logs
    do not contain the metric, inspect the mounted job result folder and parse
    TAO's structured status/best-score artifacts.
    """
    for mount in (platform_kwargs or {}).get("mounts", []) or []:
        if not isinstance(mount, dict):
            continue
        if mount.get("container_path") != "/results":
            continue
        host_root = mount.get("host_path")
        if not host_root:
            continue
        job_root = Path(host_root) / job_id
        for best_score_path in sorted(job_root.rglob("best_score.json")):
            metric = _extract_metric_from_best_score_file(
                best_score_path,
                metric_name,
                allow_generic=allow_generic,
            )
            if metric is not None:
                return metric
        for metrics_path in sorted(job_root.rglob("metrics.json")):
            metric = _extract_metric_from_metrics_file(metrics_path, metric_name)
            if metric is not None:
                return metric
        for status_path in sorted(job_root.rglob("status.json")):
            metric = _extract_metric_from_status_file(status_path, metric_name)
            if metric is not None:
                return metric
    return None


def _extract_metric_from_sdk_results(
    sdk,
    job_id: str,
    metric_name: str,
    *,
    allow_generic: bool = True,
) -> float | None:
    """Recover metrics from SDK-managed result artifacts.

    Slurm jobs write results on Lustre, which may not be mounted on the local
    AutoML controller host. Newer SDKs expose ``read_job_result_file`` for that
    case; local platforms can still be handled by reading ``get_job_results_dir``.
    """
    candidates = (
        "train_output_dir/best/best_score.json",
        "results_dir/best/best_score.json",
        "best/best_score.json",
        "train_output_dir/metrics.json",
        "results_dir/metrics.json",
        "metrics.json",
    )

    read_remote = getattr(sdk, "read_job_result_file", None)
    if callable(read_remote):
        for relative_path in candidates:
            try:
                payload = read_remote(job_id, relative_path)
            except Exception:
                payload = ""
            if not payload:
                continue
            if relative_path.endswith("metrics.json"):
                metric = _extract_metric_from_metrics_payload(payload, metric_name)
            else:
                metric = _extract_metric_from_best_score_payload(
                    payload,
                    metric_name,
                    allow_generic=allow_generic,
                )
            if metric is not None:
                return metric

    try:
        results_dir = sdk.get_job_results_dir(job_id)
    except Exception:
        results_dir = ""
    results_path = _uri_to_local_path(results_dir)
    if results_path and results_path.exists():
        for best_score_path in sorted(results_path.rglob("best_score.json")):
            metric = _extract_metric_from_best_score_file(
                best_score_path,
                metric_name,
                allow_generic=allow_generic,
            )
            if metric is not None:
                return metric
        for metrics_path in sorted(results_path.rglob("metrics.json")):
            metric = _extract_metric_from_metrics_file(metrics_path, metric_name)
            if metric is not None:
                return metric
        for status_path in sorted(results_path.rglob("status.json")):
            metric = _extract_metric_from_status_file(status_path, metric_name)
            if metric is not None:
                return metric
    return None


def _recover_metric_from_artifacts(
    sdk,
    job_id: str,
    metric_name: str,
    platform_kwargs: dict | None,
    *,
    allow_generic: bool = True,
) -> float | None:
    local_metric = _extract_metric_from_local_results(
        job_id,
        metric_name,
        platform_kwargs,
        allow_generic=allow_generic,
    )
    if local_metric is not None:
        return local_metric
    return _extract_metric_from_sdk_results(
        sdk,
        job_id,
        metric_name,
        allow_generic=allow_generic,
    )


def _extract_metric_values(
    logs: str,
    metric_names: list[str],
    extract_fn,
) -> dict[str, float]:
    """Extract all requested objective metrics from a log snapshot."""
    values = {}
    for index, name in enumerate(metric_names):
        try:
            if extract_fn is _extract_metric_from_logs:
                value = extract_fn(
                    logs,
                    name,
                    allow_generic=index == 0,
                )
            else:
                value = extract_fn(logs, name)
        except Exception:
            raise
        value = _callback_metric(value, "metric_extractor")
        if value is not None:
            values[name] = value
    return values


def _recover_metric_values_from_artifacts(
    sdk,
    job_id: str,
    metric_names: list[str],
    platform_kwargs: dict | None,
) -> dict[str, float]:
    """Recover all requested objective metrics from result artifacts."""
    values = {}
    for index, name in enumerate(metric_names):
        metric = _recover_metric_from_artifacts(
            sdk,
            job_id,
            name,
            platform_kwargs,
            allow_generic=index == 0,
        )
        if metric is not None:
            values[name] = float(metric)
    return values


def _metric_payload_from_values(
    values: dict[str, float],
    metric_name: str,
    metric_names: list[str],
):
    """Return a legacy scalar for one metric or a dict for objectives."""
    if len(metric_names) == 1:
        return values.get(metric_name)
    missing = [name for name in metric_names if name not in values]
    if missing:
        return None
    return {name: values[name] for name in metric_names}


def _metric_payload_primary(payload, metric_name: str):
    if isinstance(payload, dict):
        return payload.get(metric_name)
    return payload


def _recommendation_primary_metric(rec, metric_name: str):
    if rec is None:
        return None
    getter = getattr(rec, "primary_metric_value", None)
    if callable(getter):
        return getter()
    objective_values = getattr(rec, "objective_values", None)
    if isinstance(objective_values, dict) and metric_name in objective_values:
        return objective_values[metric_name]
    return _metric_payload_primary(getattr(rec, "result", None), metric_name)


def _recommendation_objective_values(rec) -> dict:
    objective_values = getattr(rec, "objective_values", None)
    if isinstance(objective_values, dict):
        return dict(objective_values)
    return {}


def _format_metric_payload(payload) -> str:
    if payload is None:
        return "None"
    if isinstance(payload, dict):
        return json.dumps(payload, sort_keys=True)
    return f"{float(payload):.6f}"


def _uri_to_local_path(uri: str) -> Path | None:
    if not uri:
        return None
    if uri.startswith("lustre://"):
        uri = uri.removeprefix("lustre://")
        if not uri.startswith("/"):
            uri = "/" + uri
    elif uri.startswith("slurm://"):
        uri = uri.removeprefix("slurm://")
        if not uri.startswith("/"):
            uri = "/" + uri
    elif "://" in uri:
        return None
    return Path(uri)


_RESUME_FILE_EXTENSIONS = (".pth", ".pth.tar", ".pt", ".ckpt", ".hdf5", ".tlt")


def _local_results_mount(platform_kwargs: dict | None) -> tuple[Path, str] | None:
    """Return the host/container results mount pair when a /results bind exists."""
    for mount in (platform_kwargs or {}).get("mounts", []) or []:
        if not isinstance(mount, dict):
            continue
        container_path = str(mount.get("container_path", "")).rstrip("/")
        host_path = mount.get("host_path")
        if container_path == "/results" and host_path:
            return Path(host_path), container_path
    return None


def _as_container_path(path: Path, host_root: Path, container_root: str) -> str:
    """Map a host bind-mount path back to the path visible inside the job."""
    rel = path.relative_to(host_root)
    return f"{container_root.rstrip('/')}/{rel.as_posix()}"


def _optional_int(value) -> int | None:
    """Convert real numeric metadata to int; ignore mock/missing values."""
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _checkpoint_epoch(path: Path) -> int:
    """Best-effort epoch number used by older report/budget helpers."""
    return sdk_checkpoint_epoch(path) or 0


def _find_local_resume_artifact(
    job_id: str,
    platform_kwargs: dict | None,
    prefer_directory: bool,
    *,
    model_name: str,
    epoch: int | None = None,
    step: int | None = None,
    action: str = "resume",
) -> str | None:
    """Find a parent job checkpoint on a shared local /results mount.

    Multi-fidelity algorithms resume promoted trials from the checkpoint
    produced by a lower-budget job. For local Docker, all trials share the same
    host bind mount at /results, so the next container needs the container-side
    path rather than the host path.
    """
    mount = _local_results_mount(platform_kwargs)
    if not mount:
        return None
    host_root, container_root = mount
    job_root = host_root / job_id
    if not job_root.exists():
        return None

    candidates = []

    def usable(path: Path) -> bool:
        norm = path.as_posix()
        return "/inputs/" not in norm and "/ptm/" not in norm

    for root, dirs, files in os.walk(job_root):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if d not in {"inputs", "ptm", "__pycache__"}]
        if not usable(root_path):
            continue

        name = root_path.name.lower()
        if root_path != job_root and (name.startswith("epoch_") or name.startswith("step_")):
            try:
                if any(root_path.iterdir()):
                    candidates.append(
                        build_checkpoint_candidate(
                            root_path,
                            is_dir=True,
                            mtime=root_path.stat().st_mtime,
                        )
                    )
            except OSError:
                pass

        for filename in files:
            file_path = root_path / filename
            if not usable(file_path):
                continue
            lower = filename.lower()
            if not lower.endswith(_RESUME_FILE_EXTENSIONS):
                continue
            try:
                candidates.append(
                    build_checkpoint_candidate(
                        file_path,
                        is_dir=False,
                        mtime=file_path.stat().st_mtime,
                    )
                )
            except OSError:
                pass

    if not candidates:
        return None
    selected = select_checkpoint_path(
        candidates,
        model_name=model_name,
        epoch=epoch,
        step=step,
        action=action,
        prefer_directory=prefer_directory,
        allow_latest=epoch is None and step is None,
    )
    if selected is None:
        return None
    selected = Path(selected)
    # Cosmos-RL stores each epoch as ``checkpoints/epoch_N/policy``.  The
    # epoch directory is useful grouping metadata, but the trainer's resume
    # loader expects the policy directory that contains ``cosmos_config`` and
    # the rank-specific model/optimizer/scheduler state.  Passing epoch_N
    # makes Cosmos silently fall back to the base Hugging Face model and
    # restart at epoch 1, which defeats ASHA/Hyperband promotion semantics.
    if (
        model_name.replace("_", "-") == "cosmos-rl"
        and action in {"resume", "train"}
        and selected.is_dir()
    ):
        policy_dir = selected / "policy"
        if (policy_dir / "cosmos_config").is_file():
            selected = policy_dir
    return _as_container_path(selected, host_root, container_root)


def _find_sdk_resume_artifact(
    sdk,
    job_id: str,
    *,
    model_name: str,
    epoch: int | None = None,
    step: int | None = None,
    action: str = "resume",
    prefer_directory: bool = False,
) -> str | None:
    """Fallback checkpoint lookup for SDKs that expose result listings."""
    try:
        checkpoints = sdk.get_checkpoints(job_id)
    except Exception:
        checkpoints = []
    if not checkpoints:
        return None
    try:
        results_dir = sdk.get_job_results_dir(job_id).rstrip("/")
    except Exception:
        results_dir = ""

    def normalize(path: str) -> str:
        if "://" in path or path.startswith("/"):
            return path
        return f"{results_dir}/{path}" if results_dir else path

    candidates = [build_checkpoint_candidate(normalize(path)) for path in checkpoints]
    return select_checkpoint_path(
        candidates,
        model_name=model_name,
        epoch=epoch,
        step=step,
        action=action,
        prefer_directory=prefer_directory,
        allow_latest=epoch is None and step is None,
    )


def _job_has_checkpoint_artifact(sdk, job_id: str,
                                 platform_kwargs: dict | None) -> bool:
    """Return whether a completed job produced a usable checkpoint artifact."""
    return bool(
        _find_local_resume_artifact(
            job_id,
            platform_kwargs,
            prefer_directory=False,
            model_name="",
            action="best",
        )
        or _find_sdk_resume_artifact(sdk, job_id, model_name="", action="best")
    )


def _check_execution_status(
    logs: str,
    *,
    include_fatal_patterns: bool = True,
) -> str | None:
    """Check if logs contain Execution status: PASS or FAIL."""
    if not logs:
        return None
    explicit_status = _latest_explicit_execution_status(logs)
    if explicit_status is not None:
        return explicit_status
    if not include_fatal_patterns:
        return None
    fatal_patterns = _CLEANUP_FATAL_PATTERNS + _HARD_FATAL_PATTERNS
    if any(pattern in logs for pattern in fatal_patterns):
        return "FAIL"
    return None


def _latest_explicit_execution_status(logs: str) -> str | None:
    """Return the last explicit PASS/FAIL marker in a log snapshot."""
    for line in reversed((logs or "").strip().splitlines()):
        if "Execution status: PASS" in line:
            return "PASS"
        if "Execution status: FAIL" in line:
            return "FAIL"
    return None


def _has_hard_failure_pattern(logs: str) -> bool:
    return bool(logs and any(pattern in logs for pattern in _HARD_FATAL_PATTERNS))


class MetricExtractorError(RuntimeError):
    """Raised when the metric extractor returns None for N consecutive recs.

    Catches broken extractors fast instead of letting AutoML march on for
    hours producing useless data. Common cause: ``metric_name`` doesn't match
    what the container actually emits (e.g., user passes ``val_acc`` but the
    container writes metrics only to ``<results_dir>/train/status.json``,
    not to stdout — in which case the right fix is to pass ``eval_fn=`` with
    a status.json reader).
    """


# Spec keys we treat as per-job output directories. When a user hardcodes
# any of these to a local (non-URI) path, every rec would write to the same
# place and overwrite the previous one. ``_auto_suffix_output_dirs`` rewrites
# these to ``<value>/rec_<id>`` before submit; the SDK happy-path (env-var
# driven output routing) is unaffected because it kicks in for keys with
# remote URIs or empty strings.
_OUTPUT_DIR_KEY_SUFFIXES = ("results_dir", "output_dir", "save_dir")
_DECLARED_OUTPUT_INTERPOLATION_RE = re.compile(r"^\$\{([^{}]+)\}(?:/|$)")


def _iter_dotted_keys(d: dict, prefix: str = ""):
    """Yield (dotted_key, value) for every leaf in a nested dict."""
    if not isinstance(d, dict):
        return
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            yield from _iter_dotted_keys(v, full)
        else:
            yield full, v


def _auto_suffix_output_dirs(specs: dict, rec_id, declared_outputs: set) -> list[str]:
    """In-place: rewrite hardcoded local output-dir values to per-rec subdirs.

    Skips keys that are declared in the skill's ``script_runner["outputs"]``
    (those are SDK-routed at runtime via env vars). Skips values that are
    already remote URIs (treats `://` as the URI marker). A nested path derived
    from a declared output, such as ``${results_dir}/train``, is already rooted
    below the SDK's per-job destination and must not be suffixed. Returns the
    list of dotted keys we rewrote, for logging.
    """
    rewritten: list[str] = []
    # Snapshot first; mutating during traversal of a nested dict is fragile.
    leaves = list(_iter_dotted_keys(specs))
    for dotted, value in leaves:
        leaf = dotted.rsplit(".", 1)[-1]
        if not any(leaf == suffix for suffix in _OUTPUT_DIR_KEY_SUFFIXES):
            continue
        if dotted in declared_outputs:
            continue  # SDK will route this one via env vars
        if not isinstance(value, str) or not value:
            continue
        if "://" in value:
            continue  # remote URI — user opted into a specific destination
        interpolation = _DECLARED_OUTPUT_INTERPOLATION_RE.match(value)
        if interpolation and interpolation.group(1) in declared_outputs:
            continue  # Derived from an SDK-routed per-job output root.
        new_value = f"{value.rstrip('/')}/rec_{rec_id}"
        # Walk into specs and set
        cursor = specs
        parts = dotted.split(".")
        for p in parts[:-1]:
            cursor = cursor[p]
        cursor[parts[-1]] = new_value
        rewritten.append(dotted)
    return rewritten


# ---------------------------------------------------------------------------
# Key-validation helpers (fix #2: catch typos in spec_overrides /
# automl_hyperparameters at launch time instead of silently accepting them).
# ---------------------------------------------------------------------------

_PATH_PART_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)(?:\[(\d+)\])?$")


def _parse_path_part(part: str) -> tuple[str, int | None]:
    match = _PATH_PART_RE.match(str(part))
    if match:
        return match.group(1), int(match.group(2)) if match.group(2) is not None else None
    return str(part), None


def _flatten_keys(d: Any, prefix: str = "") -> set[str]:
    """Recursively flatten a nested spec dict into dotted keys."""
    keys: set[str] = set()
    if isinstance(d, dict):
        for k, v in d.items():
            full = f"{prefix}.{k}" if prefix else str(k)
            keys.add(full)
            keys |= _flatten_keys(v, full)
    elif isinstance(d, list):
        for idx, v in enumerate(d):
            full = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            keys.add(full)
            keys |= _flatten_keys(v, full)
    return keys


def _schema_property_keys(schema: Any, prefix: str = "") -> set[str]:
    """Flatten JSON-schema property names into dotted spec keys.

    The packaged spec template may omit optional fields that are still valid
    according to ``schemas/<action>.schema.json``. Validate against both so
    direct optional overrides such as ``custom.vision.fps`` do not require
    unsafe placeholder defaults in the template.
    """
    keys: set[str] = set()
    if not isinstance(schema, dict):
        return keys
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, child in properties.items():
            full = f"{prefix}.{name}" if prefix else str(name)
            keys.add(full)
            keys |= _schema_property_keys(child, full)
    items = schema.get("items")
    if isinstance(items, dict) and prefix:
        indexed = f"{prefix}[0]"
        keys.add(indexed)
        keys |= _schema_property_keys(items, indexed)
    return keys


_EXTERNAL_SEARCH_TYPES = {
    "integer", "int", "number", "float", "boolean", "bool", "string",
    "ordered_int", "ordered", "categorical",
}
_INTEGER_SCHEMA_TYPES = {"integer", "int", "ordered_int"}
_NUMBER_SCHEMA_TYPES = {"number", "float"}
_BOOLEAN_SCHEMA_TYPES = {"boolean", "bool"}


def _schema_search_enabled(metadata: dict[str, Any]) -> bool:
    enabled = metadata.get("automl_enabled", False)
    return enabled is True or (
        isinstance(enabled, str) and enabled.upper() == "TRUE"
    )


def _enum_contains(options: list[Any], value: Any) -> bool:
    for option in options:
        if isinstance(value, bool) or isinstance(option, bool):
            if type(value) is type(option) and value == option:
                return True
        elif value == option:
            return True
    return False


def _validate_schema_value(
    value: Any,
    value_type: str,
    metadata: dict[str, Any],
    nullable: bool,
    location: str,
) -> None:
    if value is None:
        if nullable:
            return
        raise TypeError(f"{location}: null is not allowed")

    if value_type in _INTEGER_SCHEMA_TYPES:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{location}: expected an integer, got {type(value).__name__}")
    elif value_type in _NUMBER_SCHEMA_TYPES:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{location}: expected a number, got {type(value).__name__}")
        if not math.isfinite(float(value)):
            raise ValueError(f"{location}: numeric values must be finite")
    elif value_type in _BOOLEAN_SCHEMA_TYPES:
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{location}: expected a boolean, got {type(value).__name__}")
    elif value_type == "string":
        if not isinstance(value, str):
            raise TypeError(f"{location}: expected a string, got {type(value).__name__}")
    elif value_type == "array":
        if not isinstance(value, list):
            raise TypeError(f"{location}: expected an array, got {type(value).__name__}")

    options = metadata.get("enum")
    if options is not None and not _enum_contains(options, value):
        raise ValueError(f"{location}: value {value!r} is not in enum {options!r}")

    for bound_name in ("minimum", "maximum"):
        if bound_name not in metadata:
            continue
        bound = metadata[bound_name]
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            raise TypeError(f"{location}: {bound_name} must be a finite number")
        if not math.isfinite(float(bound)):
            raise ValueError(f"{location}: {bound_name} must be finite")
        if value_type not in _INTEGER_SCHEMA_TYPES | _NUMBER_SCHEMA_TYPES:
            raise ValueError(
                f"{location}: {bound_name} is only valid for numeric properties"
            )
        if bound_name == "minimum" and value < bound:
            raise ValueError(f"{location}: value {value!r} is below minimum {bound!r}")
        if bound_name == "maximum" and value > bound:
            raise ValueError(f"{location}: value {value!r} exceeds maximum {bound!r}")


def _validate_specs_against_schema(
    specs: Any,
    schema: dict[str, Any],
    schema_path: Path,
    *,
    require_all: bool,
    source: str = "spec",
) -> None:
    """Validate concrete direct-script specs against their declared schema."""
    if not isinstance(specs, dict):
        raise TypeError(f"{source}: expected an object")

    def validate_object(values: dict, node: dict, prefix: str) -> None:
        properties = node.get("properties", {})
        unknown = sorted(set(values) - set(properties))
        if unknown:
            location = prefix or "<root>"
            raise ValueError(
                f"{source}: keys {unknown!r} at {location} are not declared in "
                f"{schema_path}"
            )

        required = node.get("required", [])
        if not isinstance(required, list) or not all(isinstance(k, str) for k in required):
            raise TypeError(f"{schema_path}: 'required' must be a list of property names")
        if require_all:
            missing = sorted(set(required) - set(values))
            if missing:
                location = prefix or "<root>"
                raise ValueError(f"{source}: missing required keys {missing!r} at {location}")

        for name, value in values.items():
            child = properties[name]
            dotted = f"{prefix}.{name}" if prefix else name
            value_type, metadata, nullable = resolve_schema_leaf(child)
            if value_type in {"object", "collection", "dict"}:
                if not isinstance(value, dict):
                    raise TypeError(
                        f"{source}.{dotted}: expected an object, got {type(value).__name__}"
                    )
                validate_object(value, child, dotted)
            else:
                _validate_schema_value(
                    value, value_type, metadata, nullable, f"{source}.{dotted}"
                )

    validate_object(specs, schema, "")


def _validate_external_automl_schema(schema: Any, schema_path: Path) -> None:
    """Fail early when a direct-script search schema is unusable or unsafe."""
    if not isinstance(schema, dict):
        raise TypeError(f"{schema_path}: schema must be a JSON object")
    if schema.get("type", "object") != "object":
        raise ValueError(f"{schema_path}: root schema type must be 'object'")
    if "default" in schema and not isinstance(schema["default"], dict):
        raise TypeError(f"{schema_path}: schema 'default' must be a JSON object")
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        raise ValueError(
            f"{schema_path}: schema requires a non-empty 'properties' object"
        )

    def validate_properties(nodes: dict, prefix: str = "") -> int:
        leaf_count = 0
        for name, node in nodes.items():
            dotted = f"{prefix}.{name}" if prefix else str(name)
            location = f"{schema_path}: property {dotted!r}"
            if not isinstance(name, str) or not name:
                raise TypeError(f"{schema_path}: property names must be non-empty strings")
            if not isinstance(node, dict):
                raise TypeError(f"{location} must be an object")

            raw_type = node.get("type")
            if isinstance(raw_type, list):
                raise ValueError(
                    f"{location} uses an unsupported list-valued 'type'; use "
                    "anyOf for optional values"
                )
            if raw_type is not None and not isinstance(raw_type, str):
                raise TypeError(f"{location} 'type' must be a string")

            any_of = node.get("anyOf")
            if raw_type and any_of is not None:
                raise ValueError(f"{location} cannot declare both 'type' and 'anyOf'")
            if any_of is not None:
                if (
                    not isinstance(any_of, list)
                    or not any_of
                    or not all(isinstance(option, dict) for option in any_of)
                ):
                    raise ValueError(
                        f"{location} 'anyOf' must be a non-empty list of schema objects"
                    )
                option_types = [option.get("type") for option in any_of]
                if not all(isinstance(option_type, str) for option_type in option_types):
                    raise ValueError(
                        f"{location} anyOf options must declare scalar string types"
                    )
                non_null = [option_type for option_type in option_types if option_type != "null"]
                if len(non_null) != 1:
                    raise ValueError(f"{location} supports exactly one non-null anyOf type")
            if not raw_type and any_of is None:
                raise ValueError(f"{location} requires 'type' or 'anyOf'")

            value_type, metadata, nullable = resolve_schema_leaf(node)
            object_like = value_type in {"object", "collection", "dict"}
            if object_like:
                nested = node.get("properties")
                if not isinstance(nested, dict) or not nested:
                    raise ValueError(f"{location} requires non-empty nested properties")
                leaf_count += validate_properties(nested, dotted)
                continue
            if node.get("properties"):
                raise ValueError(
                    f"{location} with nested properties must declare an object-like type"
                )

            options = metadata.get("enum")
            if options is not None and (not isinstance(options, list) or not options):
                raise ValueError(f"{location} 'enum' must be a non-empty list")
            minimum = metadata.get("minimum")
            maximum = metadata.get("maximum")
            for bound_name, bound in (("minimum", minimum), ("maximum", maximum)):
                if bound is not None and (
                    isinstance(bound, bool)
                    or not isinstance(bound, (int, float))
                    or not math.isfinite(float(bound))
                ):
                    raise ValueError(f"{location} {bound_name} must be a finite number")
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ValueError(f"{location} minimum cannot exceed maximum")

            if options is not None:
                for index, option in enumerate(options):
                    _validate_schema_value(
                        option,
                        value_type,
                        {key: value for key, value in metadata.items() if key != "enum"},
                        nullable,
                        f"{location} enum[{index}]",
                    )

            if _schema_search_enabled(metadata):
                if value_type not in _EXTERNAL_SEARCH_TYPES:
                    raise ValueError(
                        f"{location} uses unsupported searchable type {value_type!r}"
                    )
                numeric = value_type in _INTEGER_SCHEMA_TYPES | _NUMBER_SCHEMA_TYPES
                if numeric and options is None and (minimum is None or maximum is None):
                    raise ValueError(
                        f"{location} requires both minimum and maximum, or an enum"
                    )
                if value_type in {"string", "ordered_int", "ordered", "categorical"} \
                        and options is None:
                    raise ValueError(f"{location} requires an enum for discrete search")

            weights = metadata.get("option_weights")
            if weights is not None:
                if options is None or not isinstance(weights, list) or len(weights) != len(options):
                    raise ValueError(f"{location} option_weights must align with enum")
                if any(
                    isinstance(weight, bool)
                    or not isinstance(weight, (int, float))
                    or not math.isfinite(float(weight))
                    or weight < 0
                    for weight in weights
                ) or not any(weight > 0 for weight in weights):
                    raise ValueError(f"{location} option_weights must be finite and non-negative")

            if "default" in metadata:
                _validate_schema_value(
                    metadata["default"], value_type, metadata, nullable, f"{location} default"
                )
            leaf_count += 1
        return leaf_count

    if validate_properties(properties) == 0:
        raise ValueError(f"{schema_path}: schema does not declare any leaf parameters")
    if "default" in schema:
        _validate_specs_against_schema(
            schema["default"], schema, schema_path, require_all=False,
            source="schema.default",
        )


def _validate_keys_against_schema(
    provided_keys,
    base_specs,
    kind,
    schema_keys=None,
    *,
    allow_unknown=True,
):
    """Raise ValueError on provided keys that look like typos of existing
    schema keys. Accepts genuinely-new keys (logs a warning) so users who
    intentionally add a new spec field aren't blocked.
    """
    import difflib
    base_keys = set(schema_keys or ()) | _flatten_keys(base_specs)
    unknown = [k for k in provided_keys if k not in base_keys]
    for k in unknown:
        close = difflib.get_close_matches(k, base_keys, n=1, cutoff=0.85)
        if close:
            raise ValueError(
                f"{kind} key {k!r} is not in the skill's spec schema "
                f"but looks very close to existing key {close[0]!r} — "
                "did you mean that? (remove the typo; if you really "
                "intended a brand-new key, rename it so it doesn't collide.)"
            )
        if not allow_unknown:
            raise ValueError(
                f"{kind} key {k!r} is not declared in the external AutoML schema"
            )
        logger.warning(
            "%s key %r is not in the skill's spec schema. "
            "Accepting it, but double-check that it's intentional.", kind, k)


# ---------------------------------------------------------------------------
# Metric direction (fix #1: explicit minimize/maximize).
# ---------------------------------------------------------------------------

def _implicit_direction(metric_name: str) -> str:
    """The brain's existing rule: metric name containing 'loss' is minimized,
    everything else is maximized. Keep this in one place so we can layer an
    explicit override on top.
    """
    return "minimize" if "loss" in (metric_name or "").lower() else "maximize"


def _resolve_direction(metric_name: str, explicit) -> tuple[str, bool]:
    """Return (effective_direction, invert_needed).

    If the caller didn't pass a direction, follow the implicit rule.
    If they did, validate and report whether we need to invert reported
    values to make the brain (which uses the implicit rule internally)
    optimize in the requested direction.
    """
    implicit = _implicit_direction(metric_name)
    if explicit is None:
        return implicit, False
    if explicit not in ("minimize", "maximize"):
        raise ValueError(
            f"automl_settings['direction'] must be 'minimize' or 'maximize', "
            f"got {explicit!r}"
        )
    return explicit, explicit != implicit


# ---------------------------------------------------------------------------
# Recommendation safety helpers.
# ---------------------------------------------------------------------------

_SAMPLE_COUNT_KEYS = (
    "train_sample_count",
    "training_sample_count",
    "num_train_samples",
    "train_samples",
    "dataset_sample_count",
    "dataset_size",
    "samples_per_epoch",
    "custom.train_dataset.sample_count",
    "custom.train_dataset.num_samples",
    "custom.train_dataset.size",
    "dataset.train_sample_count",
    "dataset.num_train_samples",
    "data.train_sample_count",
    "train.num_samples",
)
_BATCH_SIZE_KEYS = (
    "train.train_batch_per_replica",
    "train.batch_size",
    "dataset.batch_size",
    "batch_size",
)
_MINI_BATCH_KEYS = (
    "train.train_policy.mini_batch",
    "train.mini_batch",
    "mini_batch",
)
_DP_SHARD_KEYS = (
    "policy.parallelism.dp_shard_size",
    "train.num_gpus",
    "num_gpus",
    "gpu_count",
)
_EFFECTIVE_BATCH_FAILURE_PATTERNS = (
    r"NoneType.*state_dict",
    r"scheduler.*None",
    r"0\s+training\s+steps",
    r"zero\s+training\s+steps",
    r"num_training_steps[^0-9]*0",
    r"train_batch_per_replica.*samples",
)


def _coerce_positive_int(value) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _get_dotted_value(source, dotted_key: str):
    if not isinstance(source, dict):
        return None
    if dotted_key in source:
        return source[dotted_key]
    cursor = source
    for part in dotted_key.split("."):
        key, idx = _parse_path_part(part)
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
        if idx is not None:
            if not isinstance(cursor, list) or idx >= len(cursor):
                return None
            cursor = cursor[idx]
    return cursor


def _set_dotted_value(target: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    cursor = target
    for part in parts[:-1]:
        key, idx = _parse_path_part(part)
        if key not in cursor:
            cursor[key] = [] if idx is not None else {}
        cursor = cursor[key]
        if idx is not None:
            while len(cursor) <= idx:
                cursor.append({})
            if cursor[idx] is None:
                cursor[idx] = {}
            cursor = cursor[idx]
    last_key, last_idx = _parse_path_part(parts[-1])
    if last_idx is None:
        cursor[last_key] = value
        return
    cursor.setdefault(last_key, [])
    while len(cursor[last_key]) <= last_idx:
        cursor[last_key].append(None)
    cursor[last_key][last_idx] = value


def _first_positive_int(sources, keys, default=None) -> int | None:
    for source in sources:
        for key in keys:
            value = _coerce_positive_int(_get_dotted_value(source, key))
            if value is not None:
                return value
    return default


def _first_present_key(source: dict, keys) -> tuple[str | None, int | None]:
    for key in keys:
        value = _coerce_positive_int(_get_dotted_value(source, key))
        if value is not None:
            return key, value
    return None, None


def _append_adjustment(rec, adjustment: dict[str, Any]) -> None:
    adjustments = getattr(rec, "adjustments", None)
    if adjustments is None:
        adjustments = []
        rec.adjustments = adjustments
    adjustments.append(adjustment)


def _maybe_cap_effective_batch(
    specs: dict,
    rec,
    automl_settings: dict,
    platform_kwargs: dict | None,
) -> str | None:
    """Cap impossible batch-size recommendations when sample count is known.

    Cosmos-RL FSDP sees roughly ``sample_count / dp_shard_size`` samples per
    rank. If a recommendation exceeds that, the trainer can produce zero steps
    and later crash while saving checkpoint state. When possible, cap the
    recommendation to the largest valid value and record the adjustment; if
    even one sample per rank is unavailable, return a failure reason so the
    caller can report an invalid recommendation without launching a job.
    """
    settings = automl_settings or {}
    kwargs = platform_kwargs or {}
    if settings.get("allow_unsafe_effective_batch") or kwargs.get(
        "allow_unsafe_effective_batch"
    ):
        return None

    sources = (settings, kwargs, specs)
    sample_count = _first_positive_int(sources, _SAMPLE_COUNT_KEYS)
    if sample_count is None:
        return None

    batch_key, batch = _first_present_key(specs, _BATCH_SIZE_KEYS)
    if batch_key is None or batch is None:
        return None

    dp_shard_size = _first_positive_int(sources, _DP_SHARD_KEYS, default=1) or 1
    samples_per_rank = sample_count / dp_shard_size
    if batch <= samples_per_rank:
        return None

    reason = (
        f"{batch_key}={batch} exceeds samples per rank "
        f"{samples_per_rank:.3g} (sample_count={sample_count}, "
        f"dp_shard_size={dp_shard_size})"
    )
    capped = int(samples_per_rank)
    mini_batch = _first_positive_int((specs,), _MINI_BATCH_KEYS, default=1) or 1
    if mini_batch > 1 and capped >= mini_batch:
        capped = (capped // mini_batch) * mini_batch

    if capped < 1:
        setattr(rec, "failure_reason", f"invalid_configuration: {reason}")
        return getattr(rec, "failure_reason")

    _set_dotted_value(specs, batch_key, capped)
    if isinstance(getattr(rec, "specs", None), dict):
        rec.specs[batch_key] = capped
    adjustment = {
        "type": "effective_batch_cap",
        "key": batch_key,
        "from": batch,
        "to": capped,
        "sample_count": sample_count,
        "dp_shard_size": dp_shard_size,
        "reason": reason,
    }
    _append_adjustment(rec, adjustment)
    logger.warning(
        "Rec %d: capped %s from %s to %s because %s",
        getattr(rec, "id", -1),
        batch_key,
        batch,
        capped,
        reason,
    )
    return None


def _classify_failure(logs: str) -> str | None:
    if not logs:
        return None
    for pattern in _EFFECTIVE_BATCH_FAILURE_PATTERNS:
        if re.search(pattern, logs, re.IGNORECASE | re.DOTALL):
            return (
                "invalid_configuration: effective batch size appears to have "
                "produced zero training steps; reduce train_batch_per_replica "
                "or increase dataset samples per data-parallel rank"
            )
    return None


def _compare_to_baseline(
    baseline_metric: float | None,
    best_metric: float | None,
    direction: str,
) -> dict[str, Any] | None:
    if baseline_metric is None or best_metric is None:
        return None
    if direction == "minimize":
        delta = baseline_metric - best_metric
    else:
        delta = best_metric - baseline_metric
    return {
        "delta": delta,
        "improved": delta > 0,
        "direction": direction,
    }


# Every key consumed from automl_settings anywhere in tao_automl: runner-level
# reads, controller/objective-level reads (the same dict flows into
# AutoML(settings=...)), and brain-level reads (AlgorithmParams.from_dict and
# per-brain factory lookups). tests/test_automl_settings_validation.py
# rederives this set from the source, so a new ``.get("key")`` consumer that
# isn't registered here fails CI. ``baseline_record_path`` and
# ``final_evaluation_record_path`` are read via _evaluation_record_path's
# key argument, which the source scan cannot see — keep them listed.
KNOWN_AUTOML_SETTINGS = frozenset({
    "algorithm",
    "allow_unsafe_effective_batch",
    "api_key",
    "automl_checkpoint_retention_strategy",
    "automl_crossover_prob",
    "automl_delete_intermediate_ckpt",
    "automl_early_stop_threshold",
    "automl_eval_interval",
    "automl_kde_samples",
    "automl_max_concurrent",
    "automl_max_epochs",
    "automl_max_experiments",
    "automl_max_generations",
    "automl_max_recommendations",
    "automl_max_trials",
    "automl_min_early_stop_epochs",
    "automl_min_points_in_model",
    "automl_min_top_configs",
    "automl_mutation_factor",
    "automl_perturbation_factor",
    "automl_population_size",
    "automl_range_override",
    "automl_reduction_factor",
    "automl_top_n_percent",
    "base_url",
    "baseline_metric",
    "baseline_record_path",
    "direction",
    "enable_llm_range_narrowing",
    "epoch_multiplier",
    "evaluation_records_dir",
    "experiment_id",
    "final_evaluation",
    "final_evaluation_metric",
    "final_evaluation_record_path",
    "hybrid_enable_llm_range_narrowing",
    "include_latency",
    "latency_direction",
    "latency_metric",
    "latency_objective",
    "latency_scale",
    "latency_weight",
    "llm_api_key",
    "llm_endpoint",
    "llm_max_tokens",
    "llm_model",
    "llm_temperature",
    "metric",
    "metric_scale",
    "metric_weight",
    "model",
    "multi_objective",
    "objectives",
    "research_program",
    "reuse_best_metric_for_final_evaluation",
    "run_baseline",
    "run_final_evaluation",
    "session_id",
    # The effective-batch safety check accepts any sample-count spelling in
    # automl_settings (variable-mediated lookup, invisible to the scan).
}) | frozenset(_SAMPLE_COUNT_KEYS)

# Accepted spellings for keys users reliably guess wrong. Aliases are
# normalized to their canonical key before any consumer reads the dict.
AUTOML_SETTING_ALIASES = {
    "num_recommendations": "automl_max_recommendations",
}


def _validate_automl_settings(settings: dict) -> dict:
    """Reject unknown automl_settings keys; normalize accepted aliases.

    Unknown keys used to be silently dropped, so a typo'd budget key ran the
    algorithm default instead of the caller's budget (observed live: 17 extra
    GPU allocations). Raising with a close-match suggestion is the only thing
    that stops the burn before submission.
    """
    normalized = dict(settings)
    for alias, canonical in AUTOML_SETTING_ALIASES.items():
        if alias not in normalized:
            continue
        alias_value = normalized.pop(alias)
        if canonical in normalized and normalized[canonical] != alias_value:
            raise ValueError(
                f"automl_settings has both {alias!r} and {canonical!r} with "
                f"different values ({alias_value!r} vs "
                f"{normalized[canonical]!r}); {alias!r} is an alias of "
                f"{canonical!r} — pass one."
            )
        normalized[canonical] = alias_value
        logger.info(
            "automl_settings: %r accepted as alias of %r", alias, canonical
        )

    unknown = sorted(set(normalized) - KNOWN_AUTOML_SETTINGS)
    if unknown:
        hints = []
        for key in unknown:
            close = difflib.get_close_matches(
                key,
                list(KNOWN_AUTOML_SETTINGS | set(AUTOML_SETTING_ALIASES)),
                n=1,
            )
            hints.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise ValueError(
            "Unrecognized automl_settings key(s): " + ", ".join(hints) + ". "
            "Unknown keys are not passed through — they would be silently "
            "ignored and the run would use defaults instead."
        )
    return normalized


def _evaluation_record_path(settings: dict, explicit_key: str, filename: str) -> str | None:
    if settings.get(explicit_key):
        return str(settings[explicit_key])
    records_dir = settings.get("evaluation_records_dir")
    if records_dir:
        return str(Path(records_dir) / filename)
    return None


def _merge_metric_payload(target: dict[str, Any], payload) -> bool:
    """Merge a metric callback payload into ``target``.

    Callback authors can return a bare float or a dict with ``metric_value`` and
    optional metadata such as ``record_path`` / ``job_id``. Returns True when a
    numeric metric was present.
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"metric", "metric_value"}:
                continue
            target[key] = value
        metric = payload.get("metric_value", payload.get("metric"))
    else:
        metric = payload
    metric = _finite_metric(metric)
    if metric is None:
        return False
    target["metric_value"] = metric
    return True


# ---------------------------------------------------------------------------
# Active-jobs persistence (fix #3: survive orchestrator crashes without
# leaking in-flight Lepton jobs).
# ---------------------------------------------------------------------------

def _active_jobs_path(workspace_path: str):
    from pathlib import Path
    return Path(workspace_path) / "active_jobs.json"


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically and durably replace a small runner recovery ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(directory_fd)
        except OSError:
            # Some filesystems/platforms do not support directory fsync. The
            # file itself was still flushed before the atomic replacement.
            pass
    finally:
        os.close(directory_fd)


def _save_active_jobs(workspace_path: str, active: dict) -> None:
    """Atomic write of {rec_id: {rec_id, job_id, submitted_at}} to disk."""
    p = _active_jobs_path(workspace_path)
    _atomic_write_json(p, list(active.values()))


def _load_active_jobs(workspace_path: str) -> list:
    p = _active_jobs_path(workspace_path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except Exception as e:
        raise RuntimeError(
            f"Couldn't read active_jobs.json; refusing to launch additional "
            f"jobs while a backend writer may be active: {e}"
        ) from e
    if not isinstance(data, list):
        raise RuntimeError(
            "active_jobs.json is not a list; refusing to launch additional jobs"
        )
    validated = []
    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise RuntimeError(f"active_jobs.json entry {index} is not an object")
        rec_id = entry.get("rec_id")
        job_id = entry.get("job_id")
        if not isinstance(rec_id, int) or not isinstance(job_id, str) or not job_id:
            raise RuntimeError(
                f"active_jobs.json entry {index} has an invalid rec_id/job_id"
            )
        validated.append(entry)
    return validated


def _artifact_jobs_path(workspace_path: str):
    return Path(workspace_path) / "artifact_jobs.json"


def _save_artifact_jobs(workspace_path: str, jobs: dict[str, str]) -> None:
    """Persist terminal artifact cleanup candidates and deletion tombstones."""
    p = _artifact_jobs_path(workspace_path)
    _atomic_write_json(p, jobs)


def _load_artifact_jobs(workspace_path: str) -> dict[str, str]:
    p = _artifact_jobs_path(workspace_path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except Exception as ex:
        logger.warning("Couldn't read artifact_jobs.json: %s; rebuilding from state", ex)
        return {}
    if not isinstance(data, dict):
        logger.warning("artifact_jobs.json is not an object; rebuilding from state")
        return {}
    return {
        str(job_id): str(status)
        for job_id, status in data.items()
        if job_id
    }


def _managed_runner_run(method):
    """Install scoped signal handling and cancel jobs while unwinding a run."""
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        global _runner
        previous_runner = _runner
        previous_signal_handlers = {}
        _runner = self
        self._signal_cleanup_performed = False
        self._pending_signal = None
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                previous_signal_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, _signal_handler)
            except ValueError:
                # Python permits signal installation only from the main
                # thread. Programmatic worker-thread runs still receive the
                # BaseException cleanup below.
                logger.debug(
                    "Skipping signal handler installation outside the main thread"
                )
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            if not self._signal_cleanup_performed:
                reason = (
                    f"received signal {self._pending_signal}"
                    if self._pending_signal is not None
                    else "AutoML runner interrupted"
                )
                try:
                    self.cancel_active_jobs(reason=reason)
                except BaseException:
                    logger.exception("AutoML cancellation sweep failed while unwinding")
                    logger.warning(
                        "The cancellation sweep was interrupted before it "
                        "finished; platform jobs from this run may still be "
                        "running. Check with squeue/scancel (SLURM) or the "
                        "platform's job list, or rerun cancel_active_jobs() "
                        "from a fresh session."
                    )
                finally:
                    self._signal_cleanup_performed = True
            raise
        finally:
            for signum, previous_handler in previous_signal_handlers.items():
                try:
                    signal.signal(signum, previous_handler)
                except ValueError:
                    logger.debug(
                        "Could not restore signal handler outside the main thread"
                    )
            if _runner is self:
                _runner = previous_runner

    return wrapped


class AutoMLRunner:
    """Wires AutoML brain to SDK execution for automated HPO loops.

    The runner accepts any container platform SDK (LeptonSDK / SlurmSDK /
    KubernetesSDK / DockerSDK / BrevSDK) or VirtualEnvSDK. It does NOT pick a
    runtime for the caller — instantiate the SDK you want and pass it in.

    ``skill_dir`` is the absolute path to a packaged or external model
    directory (for example a directory under ``tao-skills-external/skills/models``).
    The runner reads ``references/skill_info.yaml`` and
    ``references/spec_template_<action>.yaml`` from there.
    """

    # Raise MetricExtractorError if this many consecutive recs return None.
    # Catches broken extractors fast (e.g. wrong metric_name, container only
    # writes to status.json) instead of letting the brain see all-failure.
    _MAX_CONSECUTIVE_NONE_METRICS = 3

    def __init__(self, sdk, skill_dir, action: str = "train",
                 poll_interval: int = _DEFAULT_POLL_INTERVAL):
        self._sdk = sdk
        self.skill_ctx = SkillContext(skill_dir=Path(skill_dir), action=action)
        self._poll_interval = poll_interval
        self._active_jobs = {}
        self._consecutive_none_metrics = 0
        self._delete_intermediate_ckpt = False
        self._algorithm = ""
        self._retain_pareto_front = False
        self._terminal_job_ids = {}
        self._deleted_job_ids = set()
        self._cleanup_capability_warned = False
        self._workspace_path = None
        self._automl = None
        self._cancel_requests = {}
        self._cancel_confirmation_timeout = _CANCEL_CONFIRM_TIMEOUT_SECONDS
        self._cancel_confirmation_poll_interval = _CANCEL_CONFIRM_POLL_SECONDS
        self._signal_cleanup_performed = False
        self._pending_signal = None

    def _record_terminal_job(self, job_id, status: str) -> bool:
        """Remember a terminal job so later best-selection can prune it."""
        if not isinstance(job_id, str) or not job_id:
            return True
        if self._terminal_job_ids.get(job_id) == "deleted":
            return True
        self._terminal_job_ids[job_id] = str(status or "failure")
        return self._persist_artifact_jobs()

    def _persist_artifact_jobs(self) -> bool:
        if not self._workspace_path:
            return True
        try:
            _save_artifact_jobs(self._workspace_path, self._terminal_job_ids)
        except Exception as ex:
            logger.warning("Failed to persist artifact_jobs.json: %s", ex)
            return False
        return True

    def _collect_terminal_jobs(self, automl) -> tuple[set[str], list]:
        """Merge persisted controller jobs into this run's cleanup candidates."""
        try:
            history = list(automl.get_history())
        except Exception as ex:
            logger.warning("Could not inspect AutoML history for artifact cleanup: %s", ex)
            return {
                job_id for job_id, status in self._terminal_job_ids.items()
                if status != "deleted"
            }, []

        candidates = {
            job_id for job_id, status in self._terminal_job_ids.items()
            if status != "deleted"
        }
        for rec in history:
            job_id = getattr(rec, "job_id", None)
            status = str(getattr(rec, "status", ""))
            if isinstance(job_id, str) and job_id and status in _TERMINAL_REC_STATUSES:
                self._terminal_job_ids.setdefault(job_id, status)
                if self._terminal_job_ids[job_id] != "deleted":
                    candidates.add(job_id)
            # Do not adopt arbitrary resume parents as cleanup-owned. A
            # restored/custom recommendation may reference an external job;
            # only IDs already recorded by this runner (or represented as a
            # terminal recommendation above) are eligible candidates.
        if not self._persist_artifact_jobs():
            logger.warning(
                "Skipping artifact cleanup because its ownership ledger is not durable"
            )
            return set(), history
        return candidates, history

    def _delete_job_artifacts(self, job_id: str, reason: str) -> bool:
        """Delete one job's artifacts when the SDK exposes that capability."""
        if job_id in self._deleted_job_ids:
            return True
        delete_fn = getattr(self._sdk, "delete_job_artifacts", None)
        if not callable(delete_fn):
            if not self._cleanup_capability_warned:
                logger.warning(
                    "automl_delete_intermediate_ckpt is enabled, but %s does "
                    "not provide delete_job_artifacts(job_id); retaining artifacts",
                    type(self._sdk).__name__,
                )
                self._cleanup_capability_warned = True
            return False
        try:
            deleted = delete_fn(job_id)
        except Exception as ex:
            logger.warning("Failed to delete artifacts for job %s: %s", job_id, ex)
            return False
        if deleted is not True:
            logger.warning("Platform did not delete artifacts for job %s", job_id)
            return False
        previous_status = self._terminal_job_ids.get(job_id)
        self._terminal_job_ids[job_id] = "deleted"
        if not self._persist_artifact_jobs():
            if previous_status is None:
                self._terminal_job_ids.pop(job_id, None)
            else:
                self._terminal_job_ids[job_id] = previous_status
            logger.warning(
                "Artifacts for job %s were deleted, but the deletion tombstone "
                "was not durable; cleanup will retry idempotently",
                job_id,
            )
            return False
        self._deleted_job_ids.add(job_id)
        logger.info("Deleted artifacts for job %s (%s)", job_id, reason)
        return True

    def _validate_artifact_retention_config(self, platform_kwargs: dict) -> None:
        """Fail before launch when the selected SDK cannot reclaim its outputs.

        Resolve the capability on the concrete SDK class rather than through
        dynamic instance attribute lookup. This keeps legacy SDKs compatible
        and prevents mocks with arbitrary attributes from looking like a real
        retention implementation.
        """
        validator = getattr(
            type(self._sdk), "validate_artifact_retention", None
        )
        if callable(validator):
            validator(self._sdk, **platform_kwargs)

    @staticmethod
    def _verified_hybrid_best_job_id(automl, best) -> str | None:
        """Return an explicitly verified full-fidelity Hybrid winner, if exposed."""
        verifier = getattr(automl, "get_verified_full_fidelity_best", None)
        if not callable(verifier):
            return None
        try:
            verified = verifier()
        except Exception as ex:
            logger.warning(
                "Could not verify Hybrid full-fidelity winner for cleanup: %s", ex
            )
            return None
        verified_job_id = getattr(verified, "job_id", None)
        if isinstance(verified_job_id, str) and verified_job_id:
            return verified_job_id
        return None

    def _prune_intermediate_artifacts(self, automl, *, completed: bool) -> None:
        """Prune safe terminal artifacts while retaining best/resume inputs."""
        if not self._delete_intermediate_ckpt or automl is None:
            return

        candidates, history = self._collect_terminal_jobs(automl)
        protected = {
            job_id for job_id in self._active_jobs.values()
            if isinstance(job_id, str) and job_id
        }
        try:
            best = automl.get_best()
        except Exception as ex:
            logger.warning("Could not determine best job for artifact cleanup: %s", ex)
            return
        best_job_id = getattr(best, "job_id", None) if best is not None else None
        if isinstance(best_job_id, str) and best_job_id:
            protected.add(best_job_id)
        elif best is None:
            # Failed jobs remain safe to prune, but without a selected best we
            # cannot safely distinguish successful candidates from the model
            # artifact the caller may ultimately need.
            protected.update(
                job_id for job_id, status in self._terminal_job_ids.items()
                if status in _SUCCESS_REC_STATUSES
            )

        if self._retain_pareto_front:
            try:
                pareto_front = list(automl.get_pareto_front())
            except Exception as ex:
                # A multi-objective controller that cannot expose its frontier
                # cannot safely prove any successful checkpoint is dominated.
                logger.warning(
                    "Could not determine Pareto-front jobs for artifact cleanup: "
                    "%s; retaining successful artifacts",
                    ex,
                )
                protected.update(
                    job_id for job_id, status in self._terminal_job_ids.items()
                    if status in _SUCCESS_REC_STATUSES
                )
            else:
                protected.update(
                    job_id
                    for rec in pareto_front
                    for job_id in [getattr(rec, "job_id", None)]
                    if isinstance(job_id, str) and job_id
                )

        if not completed:
            # A parent may be selected only after its result is reported. Keep
            # explicit resume dependencies and the controller's current
            # promotion/population decision set, while releasing eliminated
            # successes instead of retaining the whole search indefinitely.
            for rec in history:
                if str(getattr(rec, "status", "")) not in {
                    "pending", "started", "running",
                }:
                    continue
                parent_job_id = getattr(rec, "resume_from_job_id", None)
                if isinstance(parent_job_id, str) and parent_job_id:
                    protected.add(parent_job_id)
            if self._algorithm in _DEFER_ARTIFACT_PRUNING_ALGORITHMS:
                required_fn = getattr(
                    automl, "get_required_checkpoint_job_ids", None
                )
                if callable(required_fn):
                    try:
                        protected.update(
                            job_id for job_id in required_fn()
                            if isinstance(job_id, str) and job_id
                        )
                    except Exception as ex:
                        logger.warning(
                            "Could not determine required promotion checkpoints: "
                            "%s; retaining successful artifacts",
                            ex,
                        )
                        protected.update(
                            job_id for job_id, status
                            in self._terminal_job_ids.items()
                            if status in _SUCCESS_REC_STATUSES
                        )
                else:
                    protected.update(
                        job_id for job_id, status
                        in self._terminal_job_ids.items()
                        if status in _SUCCESS_REC_STATUSES
                    )
        elif (
            self._algorithm == "hybrid"
            and self._verified_hybrid_best_job_id(automl, best) is None
        ):
            # Hybrid may have delegated different phases to multi-fidelity
            # brains. Its outer get_best() does not prove that the selected
            # record ran at full fidelity, so retain every successful artifact
            # unless the controller exposes an explicitly verified winner.
            protected.update(
                job_id for job_id, status in self._terminal_job_ids.items()
                if status in _SUCCESS_REC_STATUSES
            )
        elif self._algorithm == "hybrid":
            protected.add(self._verified_hybrid_best_job_id(automl, best))

        protected.discard(None)

        reason = "search complete; retaining final best" if completed else "terminal non-best job"
        for job_id in sorted(candidates - protected - self._deleted_job_ids):
            self._delete_job_artifacts(job_id, reason)

    def _finalize_terminal_job(
        self,
        *,
        automl,
        rec,
        job_id: str | None,
        metric_value,
        status: str,
        workspace_path: str | None,
        report_result: bool = True,
        require_failure: bool = False,
    ) -> None:
        """Durably report, ledger, clear, then prune one terminal job."""
        if report_result:
            automl.report_result(
                rec_id=rec.id,
                metric_value=metric_value if metric_value is not None else 0.0,
                status=status,
            )
            if require_failure and str(getattr(rec, "status", "")) != "failure":
                raise RuntimeError(
                    f"recommendation {rec.id} was not persisted as failure"
                )

        if isinstance(job_id, str) and job_id:
            if not self._record_terminal_job(job_id, status):
                raise RuntimeError(
                    f"terminal artifact ledger was not persisted for job {job_id}"
                )

            previous_active = self._active_jobs.pop(rec.id, None)
            previous_cancel_request = self._cancel_requests.pop(rec.id, None)
            state_changed = (
                previous_active is not None or previous_cancel_request is not None
            )
            active_path = workspace_path or self._workspace_path
            if state_changed and active_path and not self._persist_active_jobs(active_path):
                if previous_active is not None:
                    self._active_jobs[rec.id] = previous_active
                if previous_cancel_request is not None:
                    self._cancel_requests[rec.id] = previous_cancel_request
                raise RuntimeError(
                    f"active job ledger was not cleared for terminal job {job_id}"
                )

        self._prune_intermediate_artifacts(automl, completed=False)

    def _mark_cancel_requested(self, rec_id: int, reason: str) -> bool:
        """Persist cancellation intent before mutating the platform job."""
        self._cancel_requests.setdefault(
            rec_id,
            {
                "requested_at": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
            },
        )
        if not self._workspace_path:
            return True
        if self._persist_active_jobs(self._workspace_path):
            return True
        logger.warning(
            "Could not persist cancellation request for rec %s; retaining job", rec_id
        )
        return False

    def _wait_for_job_quiescence(self, job_id: str) -> str | None:
        """Poll the SDK until a job is terminal/quiescent, with a bounded wait."""
        status_fn = getattr(self._sdk, "get_job_status", None)
        if not callable(status_fn):
            logger.warning(
                "%s cannot confirm cancellation of job %s because it does not "
                "provide get_job_status(job_id)",
                type(self._sdk).__name__, job_id,
            )
            return None

        timeout = max(0.0, float(self._cancel_confirmation_timeout))
        deadline = time.monotonic() + timeout
        last_status = None
        while True:
            try:
                status_result = status_fn(job_id)
                last_status = getattr(status_result, "status", status_result)
                confirmed = _confirmed_platform_status(status_result)
                if confirmed is not None:
                    return confirmed
            except Exception as ex:
                logger.warning(
                    "Failed to confirm cancellation status for job %s: %s", job_id, ex
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "Timed out waiting for job %s to become quiescent "
                    "(last status=%s); retaining active state",
                    job_id, last_status,
                )
                return None
            time.sleep(min(self._cancel_confirmation_poll_interval, remaining))

    def _request_job_cancellation(
        self,
        rec_id: int,
        job_id: str,
        reason: str,
        *,
        allow_refused_terminal: bool = False,
    ) -> str | None:
        """Persist intent, request cancellation, and confirm backend quiescence."""
        intent_durable = self._mark_cancel_requested(rec_id, reason)
        if not intent_durable:
            # The existing active-job snapshot already contains the recovery
            # identity. In particular, ENOSPC must not prevent us from stopping
            # the writer that is consuming the remaining disk space.
            logger.critical(
                "Cancellation intent for job %s could not be rewritten; "
                "attempting backend cancellation while retaining active state",
                job_id,
            )
        status_fn = getattr(self._sdk, "get_job_status", None)
        if callable(status_fn):
            try:
                pre_cancel_status = _confirmed_platform_status(status_fn(job_id))
            except Exception as ex:
                logger.warning(
                    "Could not inspect job %s before cancellation: %s",
                    job_id,
                    ex,
                )
            else:
                if pre_cancel_status in ("Complete", "Error"):
                    return pre_cancel_status
        try:
            cancel_result = self._sdk.cancel_job(job_id)
        except Exception as ex:
            logger.warning(
                "Cancellation request for job %s (rec %s) failed: %s; "
                "checking whether the backend nevertheless became terminal",
                job_id, rec_id, ex,
            )
            return self._wait_for_job_quiescence(job_id)
        if cancel_result is False and not allow_refused_terminal:
            logger.warning(
                "Platform did not initiate cancellation for job %s (rec %s); "
                "checking whether it already reached a terminal state",
                job_id, rec_id,
            )
        return self._wait_for_job_quiescence(job_id)

    def _cancel_unledgered_job(
        self,
        rec_id: int,
        job_id: str,
        workspace_path: str,
    ) -> str | None:
        """Keep canceling until a newly launched, unledgered writer is quiescent.

        Returning to the normal run loop while the first active-job snapshot is
        not durable would make a process crash leak an untracked training job.
        This emergency path intentionally does not depend on that failed ledger.
        """
        while True:
            if self._persist_active_jobs(workspace_path):
                logger.warning(
                    "Recovered durable registration for job %s after an initial "
                    "active-job ledger failure",
                    job_id,
                )
                return None
            try:
                self._sdk.cancel_job(job_id)
            except BaseException as ex:
                logger.error(
                    "Emergency cancellation failed for unledgered job %s: %s",
                    job_id,
                    ex,
                )
            terminal_status = self._wait_for_job_quiescence(job_id)
            if terminal_status is not None:
                if self._delete_intermediate_ckpt:
                    self._terminal_job_ids.setdefault(
                        job_id, f"unledgered_{terminal_status.lower()}"
                    )
                    self._delete_job_artifacts(
                        job_id,
                        "quiescent launch after active-ledger failure",
                    )
                self._active_jobs.pop(rec_id, None)
                self._cancel_requests.pop(rec_id, None)
                return terminal_status
            logger.critical(
                "Job %s is not durably registered and is not yet quiescent; "
                "retrying cancellation instead of leaving an untracked writer",
                job_id,
            )
            time.sleep(max(self._cancel_confirmation_poll_interval, 0.1))

    def _guard_interrupted_launch(
        self,
        rec_id: int,
        job_id: str,
        workspace_path: str | None,
    ) -> str | None:
        """Retry an interrupted ambiguous launch through the reconciliation window.

        A create request can finish after its client was interrupted. One
        immediate ``cancel_job(False)`` is therefore not enough: retain the
        durable identity and keep checking until the late backend object is
        observed/canceled or the bounded guard window expires.
        """
        self._active_jobs[rec_id] = job_id
        self._cancel_requests.setdefault(
            rec_id,
            {
                "requested_at": datetime.now(timezone.utc).isoformat(),
                "reason": "job creation interrupted",
            },
        )
        deadline = time.monotonic() + max(
            0.0, float(self._cancel_confirmation_timeout)
        )
        registration_durable = False
        while True:
            if workspace_path:
                registration_durable = (
                    self._persist_active_jobs(workspace_path)
                    or registration_durable
                )
            try:
                self._sdk.cancel_job(job_id)
            except BaseException as ex:
                logger.warning(
                    "Interrupted-launch cancellation failed for job %s: %s",
                    job_id,
                    ex,
                )

            terminal_status = None
            try:
                status_result = self._sdk.get_job_status(job_id)
                terminal_status = _confirmed_platform_status(status_result)
            except BaseException as ex:
                logger.warning(
                    "Interrupted-launch status check failed for job %s: %s",
                    job_id,
                    ex,
                )
            if terminal_status is not None:
                if terminal_status in ("Complete", "Error"):
                    logger.warning(
                        "Interrupted launch %s reached %s before cancellation; "
                        "retaining its active identity for result recovery",
                        job_id,
                        terminal_status,
                    )
                    return None
                if not self._record_terminal_job(
                    job_id, f"interrupted_{terminal_status.lower()}"
                ):
                    logger.error(
                        "Could not persist terminal interrupted job %s; retaining "
                        "its active recovery identity",
                        job_id,
                    )
                else:
                    previous_active = self._active_jobs.pop(rec_id, None)
                    previous_cancel = self._cancel_requests.pop(rec_id, None)
                    if (
                        workspace_path
                        and not self._persist_active_jobs(workspace_path)
                    ):
                        if previous_active is not None:
                            self._active_jobs[rec_id] = previous_active
                        if previous_cancel is not None:
                            self._cancel_requests[rec_id] = previous_cancel
                    else:
                        if self._delete_intermediate_ckpt:
                            self._delete_job_artifacts(
                                job_id,
                                "terminal interrupted launch",
                            )
                        return terminal_status

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if not registration_durable:
                    logger.critical(
                        "Interrupted launch %s is neither quiescent nor durably "
                        "registered; continuing reconciliation past the %.1fs "
                        "guard window",
                        job_id,
                        self._cancel_confirmation_timeout,
                    )
                    time.sleep(max(
                        self._cancel_confirmation_poll_interval, 0.1
                    ))
                    continue
                logger.critical(
                    "Interrupted launch %s did not become quiescent within %.1fs; "
                    "its durable cancellation intent remains for resume",
                    job_id,
                    self._cancel_confirmation_timeout,
                )
                return None
            time.sleep(min(
                max(self._cancel_confirmation_poll_interval, 0.01), remaining
            ))

    def _finalize_orphan_terminal_job(
        self,
        rec_id: int,
        job_id: str,
        platform_status: str,
        *,
        raise_on_corruption: bool,
    ) -> None:
        """Converge a terminal active job whose controller record is missing."""
        orphan_status = f"orphan_{platform_status.lower()}"
        if not self._record_terminal_job(job_id, orphan_status):
            raise RuntimeError(
                f"terminal artifact ledger was not persisted for orphan job {job_id}"
            )

        previous_active = self._active_jobs.pop(rec_id, None)
        previous_cancel_request = self._cancel_requests.pop(rec_id, None)
        if self._workspace_path and not self._persist_active_jobs(self._workspace_path):
            if previous_active is not None:
                self._active_jobs[rec_id] = previous_active
            if previous_cancel_request is not None:
                self._cancel_requests[rec_id] = previous_cancel_request
            raise RuntimeError(
                f"active job ledger was not cleared for orphan job {job_id}"
            )

        if self._delete_intermediate_ckpt:
            self._delete_job_artifacts(job_id, "terminal orphan from corrupt AutoML state")
        message = (
            f"AutoML state is corrupt: active job {job_id} for recommendation "
            f"{rec_id} reached {platform_status}, but the recommendation is missing"
        )
        logger.error(message)
        if raise_on_corruption:
            raise RuntimeError(message)

    def cancel_active_jobs(self, *, reason: str = "AutoML run canceled") -> None:
        """Cancel active jobs only after durable intent and confirmed quiescence."""
        for rec_id, job_id in list(self._active_jobs.items()):
            rec = None
            history_available = True
            if self._automl is not None:
                try:
                    rec = next(
                        (item for item in self._automl.get_history() if item.id == rec_id),
                        None,
                    )
                except Exception as ex:
                    logger.warning(
                        "Could not inspect AutoML history while canceling rec %s: %s",
                        rec_id, ex,
                    )
                    history_available = False

            terminal_status = self._request_job_cancellation(
                rec_id,
                job_id,
                reason,
                allow_refused_terminal=(
                    history_available and self._automl is not None and rec is None
                ),
            )
            if terminal_status is None:
                continue
            if self._automl is None:
                logger.warning(
                    "Job %s is quiescent, but no AutoML controller is available; "
                    "retaining active state for reconciliation",
                    job_id,
                )
                continue
            if terminal_status in ("Complete", "Error"):
                logger.warning(
                    "Job %s reached %s before cancellation; retaining active "
                    "state so its result and checkpoint can be recovered",
                    job_id,
                    terminal_status,
                )
                continue
            if not history_available:
                logger.warning(
                    "Job %s is quiescent, but AutoML history could not be read; "
                    "retaining active state for controller reconciliation",
                    job_id,
                )
                continue
            if rec is None:
                try:
                    self._finalize_orphan_terminal_job(
                        rec_id,
                        job_id,
                        terminal_status,
                        raise_on_corruption=False,
                    )
                except Exception as ex:
                    logger.warning(
                        "Could not finalize quiescent orphan job %s: %s; "
                        "continuing cancellation of remaining jobs",
                        job_id,
                        ex,
                    )
                    if self._delete_intermediate_ckpt:
                        self._delete_job_artifacts(
                            job_id,
                            "quiescent orphan after ledger failure",
                        )
                continue

            try:
                rec.assign_job_id(job_id)
                rec.failure_reason = "job_canceled"
                self._finalize_terminal_job(
                    automl=self._automl,
                    rec=rec,
                    job_id=job_id,
                    metric_value=0.0,
                    status="failure",
                    workspace_path=self._workspace_path,
                    require_failure=True,
                )
            except Exception as ex:
                logger.warning(
                    "Job %s is quiescent but rec %s failure was not durably "
                    "finalized: %s; retaining active state for resume",
                    job_id, rec_id, ex,
                )
                if self._delete_intermediate_ckpt:
                    self._delete_job_artifacts(
                        job_id,
                        "confirmed cancellation after ledger failure",
                    )
                continue
            logger.info("Canceled job %s (rec %s): %s", job_id, rec_id, reason)

        # Terminal cancellation no longer needs promotion/resume parents.
        # Once every active writer is quiescent and durably finalized, apply
        # final-search retention so interrupted multi-fidelity runs do not
        # retain every earlier successful trial checkpoint.
        if (
            self._automl is not None
            and not self._active_jobs
            and self._delete_intermediate_ckpt
        ):
            self._prune_intermediate_artifacts(self._automl, completed=True)

    @_managed_runner_run
    def run(self, train_dataset_uri="", eval_dataset_uri="",
            base_checkpoint="", workspace_id=None, image=None,
            automl_settings=None,
            automl_hyperparameters=None, custom_param_ranges=None,
            workspace_path="./automl_workspace",
            spec_overrides=None, resume=False,
            metric_extractor=None,
            eval_fn=None,
            baseline_fn=None,
            final_eval_fn=None,
            on_recommendation=None, on_result=None, execution=None,
            **platform_kwargs) -> dict:
        """Run a full AutoML optimization loop.

        Args:
            train_dataset_uri: Training dataset URI (e.g. "s3://bucket/data").
            eval_dataset_uri: Eval dataset URI (optional).
            base_checkpoint: Pretrained checkpoint URI (optional).
            workspace_id: Workspace ID (default: from SDK).
            image: Docker image override. Default: from skill_info.yaml's
                ``container_image`` (resolved via tao_sdk.versions).
                Invalid for ``python_script`` execution, which runs through
                the virtual environment SDK without a container.
            automl_settings: Algorithm config (see AlgorithmParams).
            automl_hyperparameters: Param names to search, or None for schema defaults.
            custom_param_ranges: Per-param range overrides.
            workspace_path: Local path for AutoML state persistence.
            spec_overrides: Dict of spec overrides applied to base specs before
                AutoML starts. Dotted keys supported (e.g.
                {"train.epoch": 5, "policy.model_max_length": 40960}).
            resume: If True, resume from persisted state in workspace_path.
            execution: Optional ``python_script`` action mapping. This
                overrides ``actions.<action>.execution`` from skill metadata.
                Example: ``{"type": "python_script", "script":
                "scripts/train.py", "args": ["--config", "{config_path}"]}``.
            **platform_kwargs: Forwarded to ``sdk.create_job(...)``. Pass
                whichever kwargs your platform SDK accepts (Lepton:
                ``dedicated_node_group``, ``resource_shape``, ``num_nodes``;
                SLURM: ``partition``, ``account``, ``num_nodes``;
                Kubernetes: ``namespace``, ``node_selector``, ``num_nodes``;
                Docker: ``mounts``; Brev: ``instance_id``, ``gpu_type``).
                Virtualenv: ``env_vars``, ``gpu_count``, and ``gpu_ids``.
                Plus the platform-agnostic ``gpu_count`` (defaults to 1 if
                not specified by container SDKs).
            metric_extractor: Optional callable ``(logs: str, metric_name: str) -> float | None``
                invoked on each poll of a rec's training logs to pull the
                current/latest metric value. Return ``None`` if the metric
                isn't yet present in the log snapshot. When ``None`` (default),
                the built-in ``_extract_metric_from_logs`` is used, which
                recognises training-step loss, "Validation loss:" lines,
                generic ``<metric_name>: X`` patterns, and epoch summaries.
                Supply your own extractor when your container's log format
                differs or your metric lives outside the log (e.g. reading
                an accuracy value from a results.json on S3).
            eval_fn: Optional callable ``(rec, train_job_id: str) -> float | None``
                invoked once after a rec's training job reaches a terminal
                state. Intended for workflows where the real metric needs a
                separate pipeline (e.g., merge LoRA + run inference + parse
                results.json). Whatever this returns overrides any value
                captured by ``metric_extractor`` and is what the brain sees
                via ``report_result``. Return ``None`` to fall back to the
                extractor. Raised exceptions are caught and logged; the rec
                is reported with the extractor's value (or None + failure).
            baseline_fn: Optional callable ``(base_specs: dict) -> float | None``.
                When provided and ``automl_settings.run_baseline`` is not False,
                run a base/pretrained evaluation before tuning and include the
                metric plus best-vs-baseline comparison in the returned result.
                Callers may instead provide ``automl_settings.baseline_metric``
                when the baseline was measured by a separate workflow step.
            final_eval_fn: Optional callable ``(best_rec, train_job_id: str | None) -> float | dict | None``.
                When provided and ``automl_settings.run_final_evaluation`` is not
                False, run the final evaluation for the selected best
                recommendation before ``run`` returns. Return either a numeric
                metric or a dict containing ``metric_value`` plus optional
                metadata such as ``record_path``.
            on_recommendation: Callback(rec) called when a new rec is generated.
            on_result: Callback(rec, metric, status) called when a result is reported.

        `automl_settings` additions:
            direction: Optional ``"minimize" | "maximize"``. When set, this
                overrides the implicit "metric name contains 'loss' → minimize,
                else maximize" rule. Useful when your metric name doesn't hint
                at the direction (e.g. ``"bleu_score"``, ``"perplexity"``,
                ``"wer"``). The objective configuration handles direction and
                scalarization while reported values remain on their original
                scale.
            automl_delete_intermediate_ckpt: Defaults to true. Delete failed
                and non-best terminal job artifacts through the platform SDK.
                Hyperband-family/PBT searches retain only the current
                promotion-decision window, active resume parents, and current
                best, then collapse to the winner at completion. Hybrid
                successes require an explicitly verified full-fidelity winner
                before final pruning. Multi-objective searches retain every
                non-dominated Pareto-front checkpoint.
                SDKs that expose retention validation reject an output route
                that cannot be reclaimed before the first job is launched.
                Set false to retain every trial artifact for debugging.
            automl_checkpoint_retention_strategy: Bounds checkpoint files
                written inside each retained training job when
                ``automl_delete_intermediate_ckpt`` is true. ``"auto"``
                (default) uses ``"best"`` when the merged spec exposes
                ``train.checkpointer`` and otherwise uses ``"terminal"``.
                ``"best"`` configures a single checkpoint using the trainer's
                declared monitor/mode (or the objective as fallback) and asks
                compatible trainers to replace periodic saving. It
                requires a trainer whose ``train.checkpointer`` contract
                honors ``replace_periodic``; use ``"terminal"`` for older
                trainers whose monitored checkpointing is additive only.
                ``"terminal"`` sets the epoch checkpoint interval to that
                recommendation's effective ``train.num_epochs``; promotion
                budgets such as ASHA rung epochs are therefore preserved.

        Returns:
            Dict with the following schema. The keys below are a stable
            compatibility contract (rendered by ``tao_automl.format_result``);
            new keys may be added, but existing ones will not be renamed or
            removed.

            - ``best``: ``rec_id``, ``specs``, ``metric_value``,
              ``objective_score``, ``objective_values``, ``adjustments``
            - ``progress``: ``completed``, ``total``, ``best_metric``,
              ``best_rec_id``, ``algorithm``
            - ``baseline``: ``enabled``, ``metric_name``, ``metric_value``,
              ``status``, ``comparison_to_best``; plus ``failure_reason``
              when the baseline errored
            - ``final_evaluation``: ``enabled``, ``metric_name``,
              ``metric_value``, ``status``, ``comparison_to_baseline``; plus
              ``source`` / ``failure_reason`` / ``record_path`` when
              applicable. ``status`` is one of ``measured``, ``provided``,
              ``reused_best``, ``metric_missing``, ``callback_error``
              (final_eval_fn raised), ``unavailable``, ``skipped``,
              ``not_run``.
            - ``history``: list of per-recommendation dicts with ``rec_id``,
              ``metric``, ``objective_score``, ``objective_values``,
              ``status``, ``failure_reason``, ``adjustments``
            - ``pareto_front``: present for multi-objective sessions only
        """
        from tao_automl import AutoML

        automl_settings = automl_settings or {"algorithm": "bayesian", "metric": "loss"}
        automl_settings = _validate_automl_settings(automl_settings)
        self._delete_intermediate_ckpt = _bool_setting(
            automl_settings.get("automl_delete_intermediate_ckpt", True)
        )
        self._algorithm = str(automl_settings.get("algorithm", "")).lower()
        self._terminal_job_ids = {}
        self._deleted_job_ids = set()
        self._cleanup_capability_warned = False
        self._automl = None
        if self._delete_intermediate_ckpt:
            self._validate_artifact_retention_config(platform_kwargs)
        objective_config = parse_objective_config(automl_settings)
        self._retain_pareto_front = objective_config.is_multi_objective
        objective_names = objective_config.metric_names
        workspace_id = workspace_id or getattr(self._sdk, "_workspace_id", "")
        network_arch = self.skill_ctx.network_arch

        if not resume:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            workspace_path = os.path.join(workspace_path, f"run_{ts}")
        os.makedirs(workspace_path, exist_ok=True)
        self._workspace_path = workspace_path
        if resume:
            self._terminal_job_ids.update(_load_artifact_jobs(workspace_path))
            self._deleted_job_ids.update(
                job_id for job_id, status in self._terminal_job_ids.items()
                if status == "deleted"
            )
        logger.info("Workspace: %s", workspace_path)

        # Skill metadata is loaded once at __init__ via SkillContext (replaces
        # the deleted SkillBank). action_cfg carries command/inputs/outputs/
        # config_format/upload_excludes — exactly what build_entrypoint takes.
        base_specs = copy.deepcopy(self.skill_ctx.default_specs)
        action_cfg = self.skill_ctx.action_cfg
        resolved_execution = (
            PythonScriptExecution.from_config(
                execution,
                skill_dir=self.skill_ctx.skill_dir,
                default_config_format=action_cfg.get("config_format", "yaml"),
            )
            if execution is not None
            else self.skill_ctx.execution
        )
        if resolved_execution is not None:
            schema_path = (
                self.skill_ctx.skill_dir / f"schemas/{self.skill_ctx.action}.schema.json"
            )
            if self.skill_ctx.schema is None:
                raise FileNotFoundError(
                    "python_script actions require an external AutoML schema at "
                    f"{schema_path}"
                )
            _validate_external_automl_schema(self.skill_ctx.schema, schema_path)
            if image:
                raise ValueError(
                    "image cannot be used with python_script execution; "
                    "the virtual environment is the runtime"
                )
            resolved_image = None
        else:
            resolved_image = image or self.skill_ctx.container_image
            if not resolved_image:
                raise ValueError(
                    "Container execution requires an image in skill metadata "
                    "or the run(image=...) override"
                )
        data_format = self.skill_ctx.skill_info.get("data_format")

        # Tar-extraction note: the in-container script_runner now handles
        # tar/tar.gz extraction inline (commit 661040b on tao-sdk main:
        # "Port tar/tar.gz extraction into script_runner"). We no longer
        # need to prepend an extract_cmd to the action command — the runner
        # detects archive inputs and extracts them as part of input download.
        # The old AutoML extract_cmd is gone; if a future skill's media
        # layout breaks this assumption, fix it in script_runner, not here.

        # Inject dataset URIs declared by the skill's data_sources config.
        # Generic over any skill: maps spec keys to train/eval URIs using the
        # skill's own rules (source, path template, path_from_format).
        self._apply_data_sources(
            skill_info=self.skill_ctx.skill_info, specs=base_specs,
            action=self.skill_ctx.action,
            train_dataset_uri=train_dataset_uri,
            eval_dataset_uri=eval_dataset_uri,
            data_format=data_format,
        )

        # --- fix #2: validate spec_overrides + automl_hyperparameters
        #              against the schema before anything expensive runs.
        if spec_overrides:
            _validate_keys_against_schema(
                list(spec_overrides.keys()), base_specs, "spec_override",
                self.skill_ctx.valid_spec_keys,
                allow_unknown=resolved_execution is None,
            )
            base_specs = self._merge_specs(base_specs, spec_overrides)
        if automl_hyperparameters:
            _validate_keys_against_schema(
                list(automl_hyperparameters), base_specs, "automl_hyperparameter",
                self.skill_ctx.valid_spec_keys,
                allow_unknown=resolved_execution is None,
            )
        if resolved_execution is not None:
            _validate_specs_against_schema(
                base_specs,
                self.skill_ctx.schema,
                schema_path,
                require_all=True,
                source="merged train spec",
            )

        # Validate explicit direction. Direction is now honored by AutoML's
        # objective config, so the runner keeps reported values on their
        # original scale.
        metric_name = objective_config.primary_metric
        _effective_dir = objective_config.primary_direction

        baseline = {
            "enabled": bool(automl_settings.get("run_baseline", True)),
            "metric_name": metric_name,
            "metric_value": None,
            "status": "not_run",
        }
        if baseline["enabled"]:
            if automl_settings.get("baseline_metric") is not None:
                baseline["metric_value"] = _require_finite_metric(
                    automl_settings["baseline_metric"],
                    "automl_settings['baseline_metric']",
                )
                baseline["status"] = "provided"
            elif baseline_fn is not None:
                try:
                    baseline_metric = baseline_fn(copy.deepcopy(base_specs))
                except Exception as ex:
                    logger.warning("baseline_fn raised: %s", ex)
                    baseline["status"] = "failure"
                    baseline["failure_reason"] = str(ex)
                else:
                    if baseline_metric is not None:
                        validated_metric = _callback_metric(
                            baseline_metric, "baseline_fn"
                        )
                        if validated_metric is None:
                            baseline["status"] = "failure"
                            baseline["failure_reason"] = (
                                "baseline_fn returned an invalid non-finite or "
                                "boolean metric"
                            )
                        else:
                            baseline["metric_value"] = validated_metric
                            baseline["status"] = "measured"
                    else:
                        baseline["status"] = "metric_missing"
            else:
                baseline["status"] = "unavailable"
                baseline["failure_reason"] = (
                    "baseline_fn or automl_settings['baseline_metric'] was not provided"
                )
        else:
            baseline["status"] = "skipped"
        baseline_record_path = _evaluation_record_path(
            automl_settings, "baseline_record_path", "ptm_baseline.json"
        )
        if baseline_record_path:
            baseline.setdefault("record_path", baseline_record_path)

        automl = AutoML(
            workspace=workspace_path, network=network_arch,
            train_specs=base_specs, settings=automl_settings,
            automl_hyperparameters=automl_hyperparameters,
            custom_param_ranges=custom_param_ranges,
            action=self.skill_ctx.action,
            search_schema=self.skill_ctx.schema,
            resume=resume,
        )
        self._automl = automl
        logger.info("Starting AutoML loop: network=%s, algorithm=%s, "
                    "metric=%s, direction=%s",
                     network_arch, automl_settings.get("algorithm"),
                     metric_name, _effective_dir)

        # --- fix #3: if resuming, recover any jobs that were in flight when
        #              the previous orchestrator died. Poll each to terminal,
        #              report to the brain, then continue.
        if resume:
            pending = _load_active_jobs(workspace_path)
            if pending:
                logger.info("Resume: recovering %d in-flight job(s) from prior run",
                            len(pending))
                # Restore the entire durable snapshot before reconciling any
                # one entry so clearing the first job cannot erase later jobs
                # if this process is interrupted mid-recovery.
                for entry in pending:
                    rec_id = entry["rec_id"]
                    self._active_jobs[rec_id] = entry["job_id"]
                    if entry.get("cancel_requested"):
                        self._cancel_requests[rec_id] = {
                            "requested_at": entry.get("cancel_requested_at"),
                            "reason": (
                                entry.get("cancel_reason")
                                or "restored cancellation"
                            ),
                        }
                for entry in pending:
                    self._recover_pending_job(
                        entry=entry, automl=automl, metric_name=metric_name,
                        metric_extractor=metric_extractor, eval_fn=eval_fn,
                        workspace_path=workspace_path, on_result=on_result,
                        objective_names=objective_names,
                        platform_kwargs=platform_kwargs,
                    )

        while not automl.is_complete():
            progress = automl.get_progress()
            max_recommendations = automl_settings.get("automl_max_recommendations")
            if max_recommendations is not None:
                if int(progress.get("completed", 0)) >= int(max_recommendations):
                    logger.info(
                        "AutoML recommendation budget reached (%d/%d); stopping launch loop",
                        progress.get("completed", 0), int(max_recommendations),
                    )
                    break
            # Generate only after the runner-level cap check. Once a brain
            # emits a batch, run the whole batch: slicing can strand pending
            # promotion records inside multi-fidelity algorithms.
            recs = automl.next_recommendation()
            if not recs:
                logger.info("No recommendations available — waiting for results")
                time.sleep(5)
                continue
            # Pre-compute the set of declared-output spec keys so the
            # results_dir auto-suffix safety net doesn't fight the SDK's
            # env-var-driven output routing for keys the skill already owns.
            declared_outputs = set(action_cfg.get("outputs") or [])
            if isinstance(declared_outputs, dict):
                declared_outputs = set(declared_outputs.keys())

            for rec in recs:
                if on_recommendation:
                    on_recommendation(rec)
                logger.info("Recommendation %d: launching job with %d spec overrides",
                            rec.id, len(rec.specs))
                run_base_specs = base_specs
                try:
                    stored_specs = automl._state_store.get_job_specs(automl._context.id)
                    if stored_specs:
                        run_base_specs = stored_specs
                except Exception as ex:
                    logger.debug("Could not read AutoML-updated base specs: %s", ex)
                merged_specs = self._merge_specs(run_base_specs, rec.specs)
                effective_checkpoint_strategy = None
                if (
                    self._delete_intermediate_ckpt
                    and self.skill_ctx.action == "train"
                    and isinstance(merged_specs.get("train"), dict)
                ):
                    effective_checkpoint_strategy = (
                        _apply_checkpoint_retention_strategy(
                            merged_specs,
                            enabled=True,
                            strategy=automl_settings.get(
                                "automl_checkpoint_retention_strategy", "auto"
                            ),
                            metric=metric_name,
                            direction=_effective_dir,
                        )
                    )
                if effective_checkpoint_strategy is not None:
                    logger.info(
                        "Recommendation %d: checkpoint retention strategy=%s",
                        rec.id,
                        effective_checkpoint_strategy,
                    )
                merged_specs = self._apply_resume_checkpoint(
                    merged_specs, rec, platform_kwargs
                )
                if getattr(rec, "resume_checkpoint_missing", False):
                    logger.warning(
                        "Recommendation %d: skipping launch because promoted "
                        "parent checkpoint is missing",
                        rec.id,
                    )
                    automl.report_result(
                        rec_id=rec.id,
                        metric_value=0.0,
                        status="failure",
                    )
                    if on_result:
                        on_result(rec, None, "failure")
                    continue
                job_platform_kwargs = self._apply_resume_environment(
                    platform_kwargs, rec
                )
                # Output destination is resolved at runtime by script_runner
                # from TAO_RESULTS_ROOT (mount) / S3_BUCKET_NAME (cloud) env
                # vars the SDK injects. The agent doesn't pre-rewrite spec
                # output keys here — that lived in the deleted SDK contract.
                previous_metric = None
                if getattr(rec, "resume_from_job_id", None):
                    if objective_config.is_multi_objective and rec.objective_values:
                        previous_metric = _callback_metric_payload(
                            _metric_payload_from_values(
                                dict(rec.objective_values),
                                metric_name,
                                objective_names,
                            ),
                            "resumed recommendation",
                        )
                    else:
                        previous_metric = _finite_metric(rec.result)
                # Safety net: if a user hardcoded a local *.results_dir /
                # *.output_dir / *.save_dir in the spec, every rec would
                # write to the same path and overwrite the previous one.
                # Auto-suffix those with /rec_<id>. SDK-routed declared
                # outputs and remote URIs are left alone.
                rewritten = _auto_suffix_output_dirs(
                    merged_specs, rec.id, declared_outputs)
                if rewritten:
                    logger.warning(
                        "Rec %d: auto-suffixed %d hardcoded output dir(s) "
                        "with /rec_%d to prevent rec-to-rec overwrite: %s",
                        rec.id, len(rewritten), rec.id, rewritten)
                invalid_reason = _maybe_cap_effective_batch(
                    merged_specs, rec, automl_settings, job_platform_kwargs
                )
                if invalid_reason:
                    logger.warning(
                        "Rec %d: skipping invalid recommendation: %s",
                        rec.id, invalid_reason,
                    )
                    automl.report_result(
                        rec_id=rec.id,
                        metric_value=0.0,
                        status="failure",
                    )
                    if on_result:
                        on_result(rec, None, "failure")
                    continue
                if resolved_execution is not None:
                    _validate_specs_against_schema(
                        merged_specs,
                        self.skill_ctx.schema,
                        schema_path,
                        require_all=True,
                        source=f"recommendation {rec.id} merged spec",
                    )
                metric_value, status = self._run_one_job(
                    image=resolved_image, action_cfg=action_cfg,
                    specs=merged_specs, rec=rec, metric_name=metric_name,
                    execution=resolved_execution,
                    metric_extractor=metric_extractor,
                    eval_fn=eval_fn,
                    workspace_path=workspace_path,
                    objective_names=objective_names,
                    platform_kwargs=job_platform_kwargs,
                )
                metric_value = _callback_metric_payload(metric_value, "training metric")
                if status == "success" and metric_value is None:
                    status = "metric_missing"
                metric_missing = status == "metric_missing"
                if (
                    metric_missing
                    and previous_metric is not None
                    and getattr(rec, "job_id", None)
                    and _job_has_checkpoint_artifact(
                        self._sdk, rec.job_id, job_platform_kwargs
                    )
                ):
                    logger.warning(
                        "Rec %d: promoted job %s produced a checkpoint but no "
                        "fresh metric; carrying forward prior metric=%s",
                        rec.id, rec.job_id, _format_metric_payload(previous_metric),
                    )
                    metric_value = previous_metric
                    status = "success"
                    metric_missing = False
                elif metric_missing:
                    status = "failure"
                # Fail-loud on a broken extractor: if the configured metric
                # extractor (and eval_fn) both return None for N consecutive
                # recs, we're not measuring anything — raise instead of
                # letting the brain see all-failures for hours.
                if metric_missing:
                    self._consecutive_none_metrics += 1
                else:
                    self._consecutive_none_metrics = 0
                metric_error = self._consecutive_none_metrics >= (
                    self._MAX_CONSECUTIVE_NONE_METRICS
                )
                # Keep active_jobs durable until controller state and the
                # terminal artifact ledger have both been committed.
                self._finalize_terminal_job(
                    automl=automl,
                    rec=rec,
                    job_id=getattr(rec, "job_id", None),
                    metric_value=metric_value,
                    status=status,
                    workspace_path=workspace_path,
                )
                if on_result:
                    on_result(rec, metric_value, status)
                logger.info("Recommendation %d: metric=%.6f, status=%s",
                            rec.id,
                            _metric_payload_primary(metric_value, metric_name)
                            if metric_value is not None else 0.0,
                            status)
                if metric_error:
                    raise MetricExtractorError(
                        f"No metric extracted for "
                        f"{self._consecutive_none_metrics} consecutive "
                        f"recs (metric_name={metric_name!r}). Likely "
                        f"causes: (1) the container only emits this "
                        f"metric via <results_dir>/train/status.json, "
                        f"not stdout — pass eval_fn= with a status.json "
                        f"reader; (2) metric_name doesn't match what "
                        f"the container actually emits; (3) the regex "
                        f"in _extract_metric_from_logs needs a new "
                        f"pattern for this model. Inspect the last "
                        f"job's logs to confirm."
                    )

        search_complete = bool(automl.is_complete())
        best = automl.get_best()
        progress = automl.get_progress()
        history = automl.get_history()
        if best is None:
            failed = [r.id for r in history if r.status == "failure"]
            raise RuntimeError(
                "AutoML finished without a successful recommendation; "
                f"failed recommendation ids: {failed}"
            )

        best_metric = _recommendation_primary_metric(best, metric_name)
        final_evaluation = {
            "enabled": bool(automl_settings.get("run_final_evaluation", True)),
            "metric_name": metric_name,
            "metric_value": None,
            "status": "not_run",
        }
        final_record_path = _evaluation_record_path(
            automl_settings, "final_evaluation_record_path", "best_automl.json"
        )
        if final_record_path:
            final_evaluation["record_path"] = final_record_path

        if final_evaluation["enabled"]:
            provided_payload = automl_settings.get("final_evaluation")
            if isinstance(provided_payload, dict):
                if _merge_metric_payload(final_evaluation, provided_payload):
                    final_evaluation["status"] = provided_payload.get("status", "provided")
                else:
                    final_evaluation["status"] = "metric_missing"
                final_evaluation["source"] = provided_payload.get("source", "provided")
            elif automl_settings.get("final_evaluation_metric") is not None:
                final_evaluation["metric_value"] = _require_finite_metric(
                    automl_settings["final_evaluation_metric"],
                    "automl_settings['final_evaluation_metric']",
                )
                final_evaluation["status"] = "provided"
                final_evaluation["source"] = "automl_settings.final_evaluation_metric"
            elif final_eval_fn is not None:
                try:
                    payload = final_eval_fn(best, getattr(best, "job_id", None))
                except Exception as ex:
                    logger.warning("final_eval_fn raised for rec %s: %s", best.id, ex)
                    # A crashed user callback is not a failed evaluation:
                    # consumers must be able to tell "your code broke" from
                    # "the metric is bad or missing".
                    final_evaluation["status"] = "callback_error"
                    final_evaluation["failure_reason"] = str(ex)
                    final_evaluation["source"] = "final_eval_fn"
                else:
                    if _merge_metric_payload(final_evaluation, payload):
                        final_evaluation["status"] = "measured"
                    else:
                        final_evaluation["status"] = "metric_missing"
                    final_evaluation["source"] = "final_eval_fn"
            elif automl_settings.get("reuse_best_metric_for_final_evaluation"):
                final_evaluation["metric_value"] = best_metric
                final_evaluation["status"] = "reused_best"
                final_evaluation["source"] = "best_selection_metric"
            else:
                final_evaluation["status"] = "unavailable"
                final_evaluation["failure_reason"] = (
                    "final_eval_fn, automl_settings['final_evaluation_metric'], "
                    "or reuse_best_metric_for_final_evaluation=True was not provided"
                )
        else:
            final_evaluation["status"] = "skipped"
        final_evaluation["comparison_to_baseline"] = _compare_to_baseline(
            baseline.get("metric_value"),
            final_evaluation.get("metric_value"),
            _effective_dir,
        )
        # Only a controller-complete search may collapse retention to its final
        # winner. A runner-level budget stop can leave pending promotions whose
        # parent checkpoints must survive a later resume.
        self._prune_intermediate_artifacts(automl, completed=search_complete)

        result = {
            "best": {
                "rec_id": best.id if best else None,
                "job_id": getattr(best, "job_id", None) if best else None,
                "specs": best.specs if best else {},
                "metric_value": best_metric,
                "objective_score": getattr(best, "objective_score", None),
                "objective_values": _recommendation_objective_values(best),
                "adjustments": getattr(best, "adjustments", []) if best else [],
            },
            "progress": progress,
            "baseline": baseline,
            "final_evaluation": final_evaluation,
            "history": [
                {
                    "rec_id": r.id,
                    "job_id": getattr(r, "job_id", None),
                    "metric": _recommendation_primary_metric(r, metric_name),
                    "objective_score": getattr(r, "objective_score", None),
                    "objective_values": _recommendation_objective_values(r),
                    "status": r.status,
                    "failure_reason": getattr(r, "failure_reason", None),
                    "adjustments": getattr(r, "adjustments", []),
                }
                for r in history
            ],
            "algorithm_state": (
                automl.get_algorithm_state()
                if hasattr(automl, "get_algorithm_state")
                else {}
            ),
        }
        baseline["comparison_to_best"] = _compare_to_baseline(
            baseline.get("metric_value"),
            result["best"]["metric_value"],
            _effective_dir,
        )
        if objective_config.is_multi_objective:
            result["pareto_front"] = automl.get_status().get("pareto_front", [])
        logger.info(
            "AutoML %s: %d recommendations, best metric=%.6f (rec %s)",
            "complete" if search_complete else "stopped before controller completion",
            progress["completed"],
            best_metric if best_metric is not None else 0.0,
            best.id if best else "N/A",
        )
        return result

    def _run_one_job(self, image, action_cfg, specs, rec, metric_name,
                     execution=None,
                     metric_extractor=None,
                     eval_fn=None,
                     workspace_path=None,
                     objective_names=None,
                     platform_kwargs=None) -> tuple[float | dict[str, float] | None, str]:
        """Launch a single training job and wait for it to finish.

        Container actions use ``tao_sdk.script_runner.build_entrypoint`` and
        ``sdk.create_job(image, command, ...)``. Python actions submit the
        nested specs directly through ``sdk.create_python_job(...)``; the
        virtualenv SDK writes the config and launches the script with its
        environment's interpreter. Monitoring is shared by both paths.
        """
        if workspace_path and not self._workspace_path:
            self._workspace_path = workspace_path
        try:
            if execution is not None:
                if not hasattr(self._sdk, "create_python_job"):
                    raise TypeError(
                        f"{type(self._sdk).__name__} does not support "
                        "python_script execution; use VirtualEnvSDK"
                    )
                job = self._sdk.create_python_job(
                    script=str(execution.script),
                    specs=specs,
                    config_format=execution.config_format,
                    script_args=list(execution.script_args),
                    inputs=action_cfg.get("inputs"),
                    outputs=action_cfg.get("outputs"),
                    upload_excludes=action_cfg.get("upload_excludes", []),
                    cwd=str(execution.cwd),
                    network_arch=self.skill_ctx.network_arch,
                    action=self.skill_ctx.action,
                    **(platform_kwargs or {}),
                )
            else:
                from tao_sdk.script_runner import build_entrypoint

                ep = build_entrypoint(
                    command=action_cfg["command"],
                    specs=specs,
                    inputs=action_cfg.get("inputs"),
                    outputs=action_cfg.get("outputs"),
                    config_format=action_cfg.get("config_format", "toml"),
                    upload_excludes=action_cfg.get("upload_excludes", []),
                )
                job = self._sdk.create_job(
                    image=image,
                    command=ep["command"],
                    **(platform_kwargs or {}),
                )
            # Keep the first local copy of the SDK identity inside the launch
            # exception boundary. A signal can arrive immediately after the
            # SDK returns, before the next Python line registers the job.
            rec.assign_job_id(job.id)
            self._active_jobs[rec.id] = job.id
        except Exception as e:
            interrupted_job_id = getattr(e, "tao_job_id", None) or getattr(
                locals().get("job"), "id", None
            )
            if isinstance(interrupted_job_id, str) and interrupted_job_id:
                try:
                    rec.assign_job_id(interrupted_job_id)
                except BaseException as assign_ex:
                    logger.warning(
                        "Could not attach interrupted job %s to rec %s: %s",
                        interrupted_job_id,
                        rec.id,
                        assign_ex,
                    )
                terminal_status = self._guard_interrupted_launch(
                    rec.id, interrupted_job_id, workspace_path
                )
                if terminal_status is None:
                    raise
                rec.failure_reason = (
                    "job_creation_interrupted: backend writer was reconciled "
                    f"as {terminal_status}"
                )
                return None, "failure"
            logger.error("Failed to create job for rec %d: %s", rec.id, e)
            rec.failure_reason = f"job_creation_failed: {e}"
            return None, "failure"
        except BaseException as exc:
            # Remote SDKs attach a durable job ID when an interrupt lands
            # after submission may have started but before a Job object can be
            # returned. Register that identity before unwinding so the signal
            # path cannot leave a late-starting writer outside AutoML state.
            interrupted_job_id = getattr(exc, "tao_job_id", None) or getattr(
                locals().get("job"), "id", None
            )
            if isinstance(interrupted_job_id, str) and interrupted_job_id:
                try:
                    rec.assign_job_id(interrupted_job_id)
                except BaseException as assign_ex:
                    logger.warning(
                        "Could not attach interrupted job %s to rec %s: %s",
                        interrupted_job_id,
                        rec.id,
                        assign_ex,
                    )
                self._guard_interrupted_launch(
                    rec.id, interrupted_job_id, workspace_path
                )
                logger.critical(
                    "Job creation was interrupted after backend submission may "
                    "have started; retained durable job %s for cancellation/recovery",
                    interrupted_job_id,
                )
            raise

        # Persist in-flight state so a resume can recover it.
        if workspace_path and not self._persist_active_jobs(workspace_path):
            terminal_status = self._cancel_unledgered_job(
                rec.id, job.id, workspace_path
            )
            if terminal_status is None:
                logger.info(
                    "Rec %d: job %s registration recovered; continuing monitoring",
                    rec.id,
                    job.id,
                )
            else:
                self._record_terminal_job(
                    job.id, f"registration_{terminal_status.lower()}"
                )
                rec.failure_reason = (
                    "active_job_registration_failed: job was canceled because its "
                    "recovery ledger could not be persisted"
                )
                logger.error(
                    "Rec %d: canceled unledgered job %s after active-job persistence "
                    "failed",
                    rec.id,
                    job.id,
                )
                return None, "failure"
        logger.info("Rec %d: job %s submitted (backend: %s)",
                    rec.id, job.id, getattr(job, "backend_job_id", job.id))

        # Caller can plug in a custom extractor; fall back to the built-in.
        extract_fn = metric_extractor or _extract_metric_from_logs
        metric_names = list(objective_names or [metric_name])

        # Poll status AND logs simultaneously — Lepton clears logs fast after
        # completion, so we cache the best metric seen during polling.
        cached_metrics = {}
        cached_exec_status = None
        all_logs = ""
        job_status = None
        confirmed_job_status = None
        failure_cancel_requested = False

        while True:
            time.sleep(self._poll_interval)

            # Read logs every poll cycle to cache metrics before they expire
            try:
                logs = self._sdk.get_job_logs(
                    job.id, tail=_POLL_LOG_TAIL_LINES
                )
                if logs:
                    all_logs = logs  # Keep latest snapshot
                    try:
                        values = _extract_metric_values(logs, metric_names, extract_fn)
                    except Exception as ex:
                        logger.warning("metric_extractor raised for rec %d: %s",
                                       rec.id, ex)
                        values = {}
                    cached_metrics.update(values)
                    es = _check_execution_status(
                        logs, include_fatal_patterns=False
                    )
                    hard_failure = _has_hard_failure_pattern(logs)
                    if not es and (
                        metric_name not in cached_metrics or hard_failure
                    ):
                        es = _check_execution_status(logs)
                        if es == "FAIL":
                            artifact_metrics = _recover_metric_values_from_artifacts(
                                self._sdk, job.id, metric_names, platform_kwargs
                            )
                            if artifact_metrics:
                                cached_metrics.update(artifact_metrics)
                                primary_metric = cached_metrics.get(metric_name)
                                if not hard_failure:
                                    es = None
                                    logger.info(
                                        "Rec %d: ignoring cleanup failure text "
                                        "after recovering metric=%s from result "
                                        "artifacts",
                                        rec.id,
                                        _format_metric_payload(
                                            _metric_payload_from_values(
                                                cached_metrics, metric_name, metric_names
                                            )
                                        ),
                                    )
                                else:
                                    logger.warning(
                                        "Rec %d: recovered metric=%s from result "
                                        "artifacts before canceling hard failed job",
                                        rec.id,
                                        _format_metric_payload(primary_metric),
                                    )
                    if es:
                        cached_exec_status = es
                        if es == "FAIL":
                            logger.warning(
                                "Rec %d: job %s logs show execution failure; "
                                "canceling backend job",
                                rec.id, job.id,
                            )
                            if not failure_cancel_requested:
                                failure_cancel_requested = True
                                confirmed_job_status = self._request_job_cancellation(
                                    rec.id,
                                    job.id,
                                    "execution failure detected in job logs",
                                )
                            if confirmed_job_status is not None:
                                break
            except Exception:
                pass

            if confirmed_job_status is not None:
                break

            try:
                job_status = self._sdk.get_job_status(job.id)
            except Exception as e:
                logger.warning("Failed to get status for job %s: %s", job.id, e)
                continue
            terminal_status = _confirmed_platform_status(job_status)
            if terminal_status is not None:
                confirmed_job_status = terminal_status
                break

        cached_metrics, cached_exec_status, terminal_logs = _scan_terminal_metric_values(
            self._sdk,
            job.id,
            metric_names,
            extract_fn,
            cached_metrics,
            cached_exec_status,
        )
        if terminal_logs:
            all_logs = terminal_logs

        if metric_name not in cached_metrics or any(
            name not in cached_metrics for name in metric_names
        ):
            artifact_metrics = _recover_metric_values_from_artifacts(
                self._sdk, job.id, metric_names, platform_kwargs
            )
            if artifact_metrics:
                cached_metrics.update(artifact_metrics)
                logger.info(
                    "Rec %d: recovered metric=%s from result artifacts "
                    "before final status classification",
                    rec.id,
                    _format_metric_payload(
                        _metric_payload_from_values(
                            cached_metrics, metric_name, metric_names
                        )
                    ),
                )

        exec_status = cached_exec_status or _check_execution_status(
            all_logs,
            include_fatal_patterns=(
                metric_name not in cached_metrics or _has_hard_failure_pattern(all_logs)
            ),
        )
        status = (
            confirmed_job_status
            or (job_status.status if job_status is not None else "Error")
        )

        if status == "Error" or exec_status == "FAIL":
            reason = _classify_failure(all_logs)
            if reason:
                rec.failure_reason = reason
            logger.warning("Rec %d: job %s failed", rec.id, job.id)
            return _metric_payload_from_values(cached_metrics, metric_name, metric_names), "failure"
        if status == "Canceled":
            rec.failure_reason = "job_canceled"
            return _metric_payload_from_values(cached_metrics, metric_name, metric_names), "failure"

        # fix #4: if an eval_fn is provided, run it post-training and let its
        # return override the log-extracted metric. Errors are isolated.
        # An explicit evaluator is authoritative for checkpoint selection.
        # Never relabel a training KPI as an evaluation metric when evaluation
        # fails or returns no value.
        metric_values = {} if eval_fn is not None else dict(cached_metrics)
        eval_metric_used = False
        if eval_fn is not None:
            try:
                eval_metric = _callback_metric_payload(
                    eval_fn(rec, job.id), "eval_fn"
                )
            except Exception as ex:
                logger.warning("eval_fn raised for rec %d: %s; evaluation "
                               "metric is unavailable", rec.id, ex)
                eval_metric = None
            if eval_metric is not None:
                if isinstance(eval_metric, dict):
                    metric_values.update({
                        str(key): float(value)
                        for key, value in eval_metric.items()
                    })
                else:
                    metric_values[metric_name] = float(eval_metric)
                logger.info("Rec %d: eval_fn returned metric=%s "
                            "(overriding log-extracted %s)",
                            rec.id,
                            _format_metric_payload(
                                _metric_payload_from_values(
                                    metric_values, metric_name, metric_names
                                )
                            ),
                            _format_metric_payload(
                                _metric_payload_from_values(
                                    cached_metrics, metric_name, metric_names
                                )
                            ))
                eval_metric_used = True
        if eval_fn is not None and not eval_metric_used:
            logger.warning(
                "Rec %d: eval_fn produced no metric; refusing training-metric fallback",
                rec.id,
            )
            return None, "metric_missing"
        if not eval_metric_used:
            local_metrics = {}
            for index, name in enumerate(metric_names):
                local_metric = _extract_metric_from_local_results(
                    job.id,
                    name,
                    platform_kwargs,
                    allow_generic=index == 0,
                )
                if local_metric is not None:
                    local_metrics[name] = float(local_metric)
            if local_metrics:
                metric_values.update(local_metrics)
                logger.info(
                    "Rec %d: using local status metric(s)=%s",
                    rec.id, _format_metric_payload(local_metrics),
                )

        metric_value = _metric_payload_from_values(
            metric_values, metric_name, metric_names
        )
        if metric_value is None:
            logger.warning("Rec %d: job %s completed but no metric could be "
                           "extracted (neither metric_extractor nor eval_fn "
                           "produced all requested metrics: %s)",
                           rec.id, job.id, metric_names)
            return None, "metric_missing"

        logger.info(
            "Rec %d: job %s succeeded, metric=%s",
            rec.id, job.id, _format_metric_payload(metric_value)
        )
        return metric_value, "success"

    def _persist_active_jobs(self, workspace_path: str) -> bool:
        """Dump self._active_jobs to workspace/active_jobs.json atomically."""
        now_iso = datetime.now(timezone.utc).isoformat()
        snapshot = {}
        for rec_id, job_id in self._active_jobs.items():
            entry = {"rec_id": rec_id, "job_id": job_id, "updated_at": now_iso}
            cancel_request = self._cancel_requests.get(rec_id)
            if cancel_request is not None:
                entry.update({
                    "cancel_requested": True,
                    "cancel_requested_at": cancel_request.get("requested_at"),
                    "cancel_reason": cancel_request.get("reason"),
                })
            snapshot[rec_id] = entry
        try:
            _save_active_jobs(workspace_path, snapshot)
        except Exception as e:
            logger.warning("Failed to persist active_jobs.json: %s", e)
            return False
        return True

    def _recover_pending_job(self, entry, automl, metric_name,
                              metric_extractor, eval_fn, workspace_path,
                              on_result, objective_names=None,
                              platform_kwargs=None) -> None:
        """Poll an in-flight job (recovered on resume), extract its result,
        and report it to the brain. Mirrors the tail of _run_one_job.
        """
        rec_id = entry["rec_id"]
        job_id = entry["job_id"]
        self._workspace_path = self._workspace_path or workspace_path
        self._active_jobs[rec_id] = job_id
        if entry.get("cancel_requested"):
            self._cancel_requests.setdefault(
                rec_id,
                {
                    "requested_at": entry.get("cancel_requested_at"),
                    "reason": entry.get("cancel_reason") or "restored cancellation",
                },
            )

        # Find the matching Recommendation object in the brain's history so
        # we can pass it to on_result/eval_fn and update rec.assign_job_id.
        rec = next((r for r in automl.get_history() if r.id == rec_id), None)
        if rec is None:
            logger.warning("Resume: rec %d not in brain history; canceling orphan job %s",
                           rec_id, job_id)
            terminal_status = self._request_job_cancellation(
                rec_id,
                job_id,
                "orphaned AutoML resume job",
                allow_refused_terminal=True,
            )
            if terminal_status is None:
                raise RuntimeError(
                    f"Could not confirm orphan job {job_id} is quiescent; "
                    "active state was retained"
                )
            if terminal_status in ("Complete", "Error"):
                raise RuntimeError(
                    f"Orphan job {job_id} reached {terminal_status}; retaining "
                    "its checkpoint because the missing recommendation makes "
                    "winner selection unsafe"
                )
            self._finalize_orphan_terminal_job(
                rec_id,
                job_id,
                terminal_status,
                raise_on_corruption=True,
            )
            return

        rec.assign_job_id(job_id)
        if (
            rec_id in self._cancel_requests
            and str(getattr(rec, "status", "")) not in _TERMINAL_REC_STATUSES
        ):
            terminal_status = self._request_job_cancellation(
                rec_id,
                job_id,
                self._cancel_requests[rec_id].get("reason")
                or "restored cancellation",
                allow_refused_terminal=True,
            )
            if terminal_status is None:
                raise RuntimeError(
                    f"Cancellation of restored job {job_id} is still unconfirmed; "
                    "active state was retained"
                )
            if terminal_status == "Canceled":
                rec.failure_reason = "job_canceled"
                self._finalize_terminal_job(
                    automl=automl,
                    rec=rec,
                    job_id=job_id,
                    metric_value=0.0,
                    status="failure",
                    workspace_path=workspace_path,
                    require_failure=True,
                )
                return
            logger.warning(
                "Resume: job %s reached %s before its restored cancellation; "
                "recovering the terminal result instead of discarding it",
                job_id,
                terminal_status,
            )
        if str(getattr(rec, "status", "")) in _TERMINAL_REC_STATUSES:
            terminal_status = self._wait_for_job_quiescence(job_id)
            if terminal_status is None:
                raise RuntimeError(
                    f"Recommendation {rec_id} is terminal, but job {job_id} "
                    "was not confirmed quiescent; active state was retained"
                )
            self._finalize_terminal_job(
                automl=automl,
                rec=rec,
                job_id=job_id,
                metric_value=getattr(rec, "result", 0.0),
                status=str(rec.status),
                workspace_path=workspace_path,
                report_result=False,
            )
            return
        extract_fn = metric_extractor or _extract_metric_from_logs
        metric_names = list(objective_names or [metric_name])

        logger.info("Resume: polling rec %d job %s", rec_id, job_id)
        cached_metrics = {}
        cached_exec_status = None
        all_logs = ""
        job_status = None
        confirmed_job_status = None
        failure_cancel_requested = False

        # Poll until terminal.
        while True:
            time.sleep(self._poll_interval)
            try:
                logs = self._sdk.get_job_logs(
                    job_id, tail=_POLL_LOG_TAIL_LINES
                )
                if logs:
                    all_logs = logs
                    try:
                        values = _extract_metric_values(logs, metric_names, extract_fn)
                    except Exception as ex:
                        logger.warning("metric_extractor raised during resume "
                                        "for rec %d: %s", rec_id, ex)
                        values = {}
                    cached_metrics.update(values)
                    es = _check_execution_status(
                        logs, include_fatal_patterns=False
                    )
                    hard_failure = _has_hard_failure_pattern(logs)
                    if not es and (metric_name not in cached_metrics or hard_failure):
                        es = _check_execution_status(logs)
                        if es == "FAIL":
                            artifact_metrics = _recover_metric_values_from_artifacts(
                                self._sdk, job_id, metric_names, platform_kwargs
                            )
                            if artifact_metrics:
                                cached_metrics.update(artifact_metrics)
                                if not hard_failure:
                                    es = None
                                    logger.info(
                                        "Resume: rec %d ignoring cleanup "
                                        "failure text after recovering metric=%s "
                                        "from result artifacts",
                                        rec_id,
                                        _format_metric_payload(
                                            _metric_payload_from_values(
                                                cached_metrics, metric_name, metric_names
                                            )
                                        ),
                                    )
                                else:
                                    logger.warning(
                                        "Resume: rec %d recovered metric=%s from "
                                        "result artifacts before canceling hard "
                                        "failed job",
                                        rec_id,
                                        _format_metric_payload(
                                            _metric_payload_from_values(
                                                cached_metrics, metric_name, metric_names
                                            )
                                        ),
                                    )
                    if es:
                        cached_exec_status = es
                        if es == "FAIL":
                            logger.warning(
                                "Resume: rec %d job %s logs show execution "
                                "failure; canceling backend job",
                                rec_id, job_id,
                            )
                            if not failure_cancel_requested:
                                failure_cancel_requested = True
                                confirmed_job_status = self._request_job_cancellation(
                                    rec_id,
                                    job_id,
                                    "execution failure detected during resume",
                                )
                            if confirmed_job_status is not None:
                                break
            except Exception:
                pass
            if confirmed_job_status is not None:
                break
            try:
                job_status = self._sdk.get_job_status(job_id)
            except Exception as e:
                logger.warning("Resume: failed to get status for job %s: %s",
                               job_id, e)
                continue
            terminal_status = _confirmed_platform_status(job_status)
            if terminal_status is not None:
                confirmed_job_status = terminal_status
                break

        cached_metrics, cached_exec_status, terminal_logs = _scan_terminal_metric_values(
            self._sdk,
            job_id,
            metric_names,
            extract_fn,
            cached_metrics,
            cached_exec_status,
        )
        if terminal_logs:
            all_logs = terminal_logs

        if metric_name not in cached_metrics or any(
            name not in cached_metrics for name in metric_names
        ):
            artifact_metrics = _recover_metric_values_from_artifacts(
                self._sdk, job_id, metric_names, platform_kwargs
            )
            if artifact_metrics:
                cached_metrics.update(artifact_metrics)
                logger.info(
                    "Resume: rec %d recovered metric=%s from result "
                    "artifacts before final status classification",
                    rec_id,
                    _format_metric_payload(
                        _metric_payload_from_values(
                            cached_metrics, metric_name, metric_names
                        )
                    ),
                )

        exec_status = cached_exec_status or _check_execution_status(
            all_logs,
            include_fatal_patterns=(
                metric_name not in cached_metrics or _has_hard_failure_pattern(all_logs)
            ),
        )
        status = (
            confirmed_job_status
            or (job_status.status if job_status is not None else "Error")
        )

        if status == "Error" or exec_status == "FAIL":
            metric_value = _metric_payload_from_values(cached_metrics, metric_name, metric_names)
            report_status = "failure"
            reason = _classify_failure(all_logs)
            if reason:
                rec.failure_reason = reason
        elif status == "Canceled":
            metric_value = _metric_payload_from_values(cached_metrics, metric_name, metric_names)
            report_status = "failure"
            rec.failure_reason = "job_canceled"
        else:
            metric_values = {} if eval_fn is not None else dict(cached_metrics)
            eval_metric_used = False
            if eval_fn is not None:
                try:
                    em = _callback_metric_payload(eval_fn(rec, job_id), "eval_fn")
                except Exception as ex:
                    logger.warning("eval_fn raised during resume for rec %d: %s",
                                    rec_id, ex)
                    em = None
                if em is not None:
                    if isinstance(em, dict):
                        metric_values.update({
                            str(key): float(value)
                            for key, value in em.items()
                        })
                    else:
                        metric_values[metric_name] = float(em)
                    eval_metric_used = True
            if eval_fn is not None and not eval_metric_used:
                metric_value = None
                report_status = "failure"
                rec.failure_reason = "evaluation_metric_missing"
                automl.report_result(
                    rec_id=rec_id,
                    metric_value=0.0,
                    status=report_status,
                )
                if on_result:
                    try:
                        on_result(rec, metric_value, report_status)
                    except Exception as ex:
                        logger.warning("on_result callback failed for rec %d: %s", rec_id, ex)
                return
            if not eval_metric_used:
                local_metrics = {}
                for index, name in enumerate(metric_names):
                    local_metric = _extract_metric_from_local_results(
                        job_id,
                        name,
                        platform_kwargs,
                        allow_generic=index == 0,
                    )
                    if local_metric is not None:
                        local_metrics[name] = float(local_metric)
                if local_metrics:
                    metric_values.update(local_metrics)
                    logger.info(
                        "Resume: rec %d using local status metric(s)=%s",
                        rec_id, _format_metric_payload(local_metrics),
                    )
            metric_value = _metric_payload_from_values(
                metric_values, metric_name, metric_names
            )
            report_status = "success" if metric_value is not None else "failure"

        self._finalize_terminal_job(
            automl=automl,
            rec=rec,
            job_id=job_id,
            metric_value=metric_value,
            status=report_status,
            workspace_path=workspace_path,
        )
        if on_result:
            try:
                on_result(rec, metric_value, report_status)
            except Exception as ex:
                logger.warning("on_result callback raised during resume: %s", ex)
        logger.info("Resume: rec %d %s metric=%s",
                    rec_id, report_status,
                    _format_metric_payload(metric_value))

    # _skill_has_tarball_media was removed: tar/tar.gz extraction is now
    # handled by script_runner inline (tao-sdk commit 661040b "Port tar/tar.gz
    # extraction into script_runner"). The runner no longer prepends an
    # extract command, so the tarball-detection helper has no callers.

    @staticmethod
    def _set_nested(target: dict, dotted_key: str, value) -> None:
        """Mutate target in-place: set target[a][b][c] for dotted_key 'a.b.c'."""
        parts = dotted_key.split(".")
        cursor = target
        for part in parts[:-1]:
            key, idx = _parse_path_part(part)
            if key not in cursor:
                cursor[key] = [] if idx is not None else {}
            cursor = cursor[key]
            if idx is not None:
                if not isinstance(cursor, list):
                    raise TypeError(f"Spec path {dotted_key!r} expected list at {key!r}")
                while len(cursor) <= idx:
                    cursor.append({})
                if cursor[idx] is None:
                    cursor[idx] = {}
                cursor = cursor[idx]
        last_key, last_idx = _parse_path_part(parts[-1])
        if last_idx is None:
            cursor[last_key] = value
            return
        if last_key not in cursor or not isinstance(cursor[last_key], list):
            cursor[last_key] = []
        while len(cursor[last_key]) <= last_idx:
            cursor[last_key].append(None)
        cursor[last_key][last_idx] = value

    @staticmethod
    def _get_nested(source: dict, dotted_key: str):
        """Read target[a][b][c] for dotted_key 'a.b.c'; None if missing."""
        parts = dotted_key.split(".")
        cursor = source
        for part in parts:
            key, idx = _parse_path_part(part)
            if not isinstance(cursor, dict) or key not in cursor:
                return None
            cursor = cursor[key]
            if idx is not None:
                if not isinstance(cursor, list) or idx >= len(cursor):
                    return None
                cursor = cursor[idx]
        return cursor

    def _apply_resume_checkpoint(
        self, specs: dict, rec, platform_kwargs: dict | None
    ) -> dict:
        """Inject parent-checkpoint resume params for promoted recommendations.

        Hyperband-family algorithms and PBT return a recommendation with
        ``resume_from_job_id`` once they promote or exploit a prior trial. The
        brain knows which trial won, but the runner owns platform paths and
        skill specs, so the checkpoint handoff belongs here.
        """
        parent_job_id = getattr(rec, "resume_from_job_id", None)
        if not parent_job_id:
            return specs

        path_key = None
        for candidate in (
            "train.resume_training_checkpoint_path",
            "resume_training_checkpoint_path",
        ):
            if (
                candidate in self.skill_ctx.valid_spec_keys
                or self._get_nested(specs, candidate) is not None
            ):
                path_key = candidate
                break

        bool_or_path_key = None
        for candidate in ("train.resume", "resume"):
            if (
                candidate in self.skill_ctx.valid_spec_keys
                or self._get_nested(specs, candidate) is not None
            ):
                bool_or_path_key = candidate
                break

        if not path_key and not bool_or_path_key:
            logger.warning(
                "Rec %d requested resume from %s, but no resume spec key was "
                "found for %s",
                rec.id, parent_job_id, self.skill_ctx.network_arch,
            )
            return specs

        prefer_directory = bool_or_path_key is not None and path_key is None
        resume_epoch = _optional_int(getattr(rec, "resume_from_epoch", None))
        resume_step = _optional_int(getattr(rec, "resume_from_step", None))
        artifact = (
            _find_local_resume_artifact(
                parent_job_id,
                platform_kwargs,
                prefer_directory,
                model_name=self.skill_ctx.network_arch,
                epoch=resume_epoch,
                step=resume_step,
                action="resume",
            )
            or _find_sdk_resume_artifact(
                self._sdk,
                parent_job_id,
                model_name=self.skill_ctx.network_arch,
                epoch=resume_epoch,
                step=resume_step,
                action="resume",
                prefer_directory=prefer_directory,
            )
        )
        if not artifact:
            rec.resume_checkpoint_path = None
            rec.resume_checkpoint_missing = True
            logger.warning(
                "Rec %d requested resume from %s, but no checkpoint artifact "
                "could be resolved for epoch=%s step=%s",
                rec.id, parent_job_id, resume_epoch, resume_step,
            )
            return specs
        rec.resume_checkpoint_missing = False
        rec.resume_checkpoint_path = artifact

        if path_key:
            self._set_nested(specs, path_key, artifact)
            if bool_or_path_key:
                # Models such as BEVFusion expose both the checkpoint path and
                # an explicit resume switch. Supplying only the path makes
                # MMEngine load weights as initialization and restart epoch 1
                # instead of restoring optimizer/epoch state.
                self._set_nested(specs, bool_or_path_key, True)
            logger.info(
                "Rec %d will resume from parent job %s via %s=%s "
                "(epoch=%s step=%s)",
                rec.id, parent_job_id, path_key, artifact, resume_epoch, resume_step,
            )
        else:
            # Cosmos-RL's `train.resume` accepts either True or a concrete
            # checkpoint path. Use the path form so a new output directory can
            # still be used for the resumed trial.
            self._set_nested(specs, bool_or_path_key, artifact)
            logger.info(
                "Rec %d will resume from parent job %s via %s=%s "
                "(epoch=%s step=%s)",
                rec.id, parent_job_id, bool_or_path_key, artifact,
                resume_epoch, resume_step,
            )

        epoch_offset = _RESUME_EPOCH_BUDGET_OFFSETS.get(
            self.skill_ctx.network_arch, 0
        )
        requested_epochs = self._get_nested(specs, "train.num_epochs")
        if epoch_offset and isinstance(requested_epochs, int):
            effective_epochs = requested_epochs + epoch_offset
            self._set_nested(specs, "train.num_epochs", effective_epochs)
            logger.info(
                "Rec %d adjusted resumed %s epoch budget from %d to %d so "
                "the requested terminal epoch executes",
                rec.id,
                self.skill_ctx.network_arch,
                requested_epochs,
                effective_epochs,
            )
        return specs

    def _apply_resume_environment(
        self, platform_kwargs: dict | None, rec
    ) -> dict | None:
        """Add runtime env needed by model-specific checkpoint resume paths."""
        if not getattr(rec, "resume_from_job_id", None):
            return platform_kwargs
        if not getattr(rec, "resume_checkpoint_path", None):
            return platform_kwargs

        updated = copy.deepcopy(platform_kwargs or {})
        env_vars = dict(updated.get("env_vars") or {})
        if env_vars.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") != "1":
            env_vars["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
            logger.info(
                "Rec %d enabling PyTorch trusted-checkpoint resume for %s",
                rec.id, self.skill_ctx.network_arch,
            )
        updated["env_vars"] = env_vars
        return updated

    # _apply_output_destinations was removed: output destinations are
    # resolved at runtime by script_runner from TAO_RESULTS_ROOT (mount) /
    # S3_BUCKET_NAME (cloud) env vars the SDK injects in create_job. The
    # runner doesn't pre-rewrite output spec keys here anymore.

    @staticmethod
    def _apply_data_sources(skill_info, specs, action,
                            train_dataset_uri, eval_dataset_uri, data_format):
        """Resolve skill's data_sources[action] into concrete URIs on specs.

        The skill declares per-spec-key rules in ``skill_info.yaml``:
          source:     "train_datasets" | "eval_dataset"
          path:       template (e.g. "{train_dataset_annotation}") — substituted
                      from top-level scalar values in *specs*.
          path_from_format: {<format>: <str|list>} — appended when present;
                      a list collapses to the folder URI (script_runner then
                      downloads the full prefix).
        """
        data_sources = skill_info.get("data_sources", {}).get(action, {})
        if not data_sources:
            return

        source_to_uri = {
            "train_datasets": train_dataset_uri,
            "eval_dataset": eval_dataset_uri,
        }
        for spec_key, rule in data_sources.items():
            base_uri = source_to_uri.get(rule.get("source"))
            if not base_uri:
                continue
            # Normalize aws:// → s3://; the container's fsspec/s3fs understands
            # s3://, while aws:// is only used by the SDK for cloud_metadata keying.
            if base_uri.startswith("aws://"):
                base_uri = "s3://" + base_uri[len("aws://"):]
            base_uri = base_uri.rstrip("/") + "/"

            mapping = rule.get("mapping")
            if isinstance(mapping, dict):
                item = {}
                for field_name, field_cfg in mapping.items():
                    field_cfg = field_cfg or {}
                    path = field_cfg.get("path")
                    if path:
                        item[field_name] = base_uri + str(path).lstrip("/")
                    elif not field_cfg.get("optional"):
                        item[field_name] = base_uri.rstrip("/")
                current = AutoMLRunner._get_nested(specs, spec_key)
                if rule.get("multiple_sources") or isinstance(current, list):
                    AutoMLRunner._set_nested(specs, spec_key, [item])
                else:
                    AutoMLRunner._set_nested(specs, spec_key, item)
                continue

            path_template = rule.get("path")
            if path_template:
                resolved = path_template
                for k, v in specs.items():
                    if isinstance(v, (str, int, float)):
                        resolved = resolved.replace("{" + k + "}", str(v))
                AutoMLRunner._set_nested(specs, spec_key, base_uri + resolved)
                continue

            path_from_format = rule.get("path_from_format")
            if path_from_format is not None:
                suffix = path_from_format.get(data_format, path_from_format.get("*"))
                chosen = None
                if isinstance(suffix, str) and suffix:
                    chosen = suffix
                elif isinstance(suffix, list):
                    tarballs = [s for s in suffix if isinstance(s, str)
                                and s.endswith(".tar.gz")]
                    # Prefer videos.tar.gz for video-leaning formats (llava),
                    # otherwise take the last tarball (skills list less-preferred
                    # candidates first).
                    chosen = (next((s for s in tarballs if "video" in s), None)
                              or (tarballs[-1] if tarballs else None))
                if chosen:
                    AutoMLRunner._set_nested(specs, spec_key, base_uri + chosen)
                else:
                    AutoMLRunner._set_nested(specs, spec_key, base_uri)

    @staticmethod
    def _merge_specs(base_specs: dict, rec_specs: dict) -> dict:
        """Deep-merge nested caller specs and dotted recommendation keys."""
        merged = copy.deepcopy(base_specs)

        def merge_mapping(target: dict, overrides: dict) -> None:
            for key, value in overrides.items():
                # Brain recommendations use dotted/indexed paths. Caller spec
                # dictionaries are nested at the SDK boundary and must merge,
                # not replace an entire top-level train/model/dataset block.
                if "." in key or "[" in key:
                    AutoMLRunner._set_nested(target, key, copy.deepcopy(value))
                    continue
                current = target.get(key)
                if isinstance(current, dict) and isinstance(value, dict):
                    merge_mapping(current, value)
                else:
                    target[key] = copy.deepcopy(value)

        merge_mapping(merged, rec_specs)
        return merged


_PLATFORMS = (
    "lepton", "slurm", "kubernetes", "docker", "brev", "virtualenv",
)


def _make_sdk(platform: str, **sdk_kwargs):
    """Construct a platform SDK by name. No default — caller must pick.

    Matches platform/tao-sdk/SKILL.md's "It does not select platforms
    automatically" stance: none of the SDKs is a sensible default
    (Lepton biases DGX Cloud, SLURM biases on-prem clusters, etc.).
    """
    if platform == "lepton":
        from tao_sdk.platforms.lepton import LeptonSDK
        return LeptonSDK(**sdk_kwargs)
    if platform == "slurm":
        from tao_sdk.platforms.slurm import SlurmSDK
        return SlurmSDK(**sdk_kwargs)
    if platform == "kubernetes":
        from tao_sdk.platforms.kubernetes import KubernetesSDK
        return KubernetesSDK(**sdk_kwargs)
    if platform == "docker":
        from tao_sdk.platforms.docker import DockerSDK
        return DockerSDK(**sdk_kwargs)
    if platform == "brev":
        from tao_sdk.platforms.brev import BrevSDK
        return BrevSDK(**sdk_kwargs)
    if platform == "virtualenv":
        from tao_sdk.platforms.virtualenv import VirtualEnvSDK
        return VirtualEnvSDK(**sdk_kwargs)
    raise ValueError(
        f"Unknown platform {platform!r}. Choose one of: {', '.join(_PLATFORMS)}."
    )


def run_automl_plan(plan: dict, platform: str) -> dict:
    """Execute an AutoML plan file on the chosen platform.

    The plan JSON's ``params`` block must include ``skill_dir`` (absolute
    path to a packaged or external model metadata directory). Per-job kwargs
    go under ``params.platform_kwargs``. SDK constructor
    kwargs go under ``params.sdk_kwargs``; a virtualenv plan must provide at
    least ``sdk_kwargs.venv_path``. Direct-script metadata may live in the
    model action or in ``params.execution``.
    """
    if not plan.get("ready"):
        issues = plan.get("blocking_issues", ["Unknown issue"])
        print("Plan is not ready to execute:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)

    step = plan["steps"][0]
    params = step["params"]
    automl_settings = plan.get("automl_settings", {})

    skill_dir = params.get("skill_dir")
    if not skill_dir:
        raise ValueError("plan.steps[0].params.skill_dir is required "
                         "(absolute path to a model metadata directory).")
    action = params.get("action", "train")
    platform_kwargs = params.get("platform_kwargs") or {}
    sdk_kwargs = params.get("sdk_kwargs") or {}

    sdk = _make_sdk(platform, **sdk_kwargs)
    runner = AutoMLRunner(sdk=sdk, skill_dir=skill_dir, action=action)
    result = runner.run(
        train_dataset_uri=params["train_dataset_uri"],
        eval_dataset_uri=params.get("eval_dataset_uri", ""),
        base_checkpoint=params.get("base_checkpoint", ""),
        workspace_id=params.get("workspace_id"),
        image=params.get("image"),
        automl_settings=automl_settings,
        automl_hyperparameters=plan.get("automl_hyperparameters"),
        custom_param_ranges=plan.get("custom_param_ranges"),
        workspace_path=plan.get("automl_workspace_path", "./automl_workspace"),
        spec_overrides=params.get("spec_overrides"),
        execution=params.get("execution"),
        **platform_kwargs,
    )
    print(json.dumps(result, indent=2, default=str))
    return result


_runner = None

def _signal_handler(signum, frame):
    """Request unwind; cancellation runs after interrupted locks are released."""
    if _runner:
        _runner._pending_signal = signum
    raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Execute an AutoML plan against a chosen platform SDK.",
    )
    parser.add_argument("plan", help="Path to the AutoML plan JSON.")
    parser.add_argument(
        "--platform", required=True, choices=_PLATFORMS,
        help="Target platform SDK. Required — no default. "
             "Pick the backend you want to submit jobs to.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    with open(args.plan) as f:
        plan = json.load(f)
    run_automl_plan(plan, platform=args.platform)


if __name__ == "__main__":
    main()
