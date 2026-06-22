# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AutoML runner: wires the tao_automl brain to a platform SDK for HPO.

The runner is platform-agnostic: it accepts any of the 5 platform SDKs
(Lepton/Slurm/Kubernetes/Docker/Brev). The caller picks the platform; the
runner doesn't choose for them.

Usage::

    from pathlib import Path
    from tao_sdk.platforms.lepton import LeptonSDK   # or Slurm/K8s/Docker/Brev
    from tao_automl.runner import AutoMLRunner

    sdk = LeptonSDK()                                 # reads creds from env
    runner = AutoMLRunner(
        sdk=sdk,
        skill_dir=Path.home() / "tao-sdk/tao-skills-external/models/cosmos-rl",
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
import json
import logging
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

import yaml
from tao_sdk.checkpoints import (
    build_checkpoint_candidate,
    checkpoint_epoch as sdk_checkpoint_epoch,
    select_checkpoint_path,
)

logger = logging.getLogger(__name__)


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
    valid_spec_keys: set[str] = field(init=False)
    container_image: str = field(init=False)
    network_arch: str = field(init=False)

    def __post_init__(self):
        self.skill_dir = Path(self.skill_dir)
        info_path = self.skill_dir / "references/skill_info.yaml"
        if not info_path.exists():
            raise FileNotFoundError(
                f"skill_info.yaml not found at {info_path}. "
                f"skill_dir must point at a model directory inside "
                f"tao-skills-external/models/<name>/."
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
                schema = json.load(f) or {}
            self.valid_spec_keys = _schema_property_keys(schema) | _flatten_keys(
                schema.get("default", {})
            ) | _flatten_keys(self.default_specs)
        else:
            self.valid_spec_keys = _flatten_keys(self.default_specs)

        # Container image: action-level image overrides win, then model-level.
        # Values may be versions.yaml keys or absolute URIs.
        from tao_sdk.versions import resolve_container_image
        self.container_image = resolve_container_image(
            self.action_cfg.get("container_image")
            or self.skill_info.get("container_image", "")
        )

    def validate_runtime(self) -> dict[str, Any]:
        """Validate that the model/action can be loaded by AutoML runtime code.

        Skill JSON can exist even when the runtime import path is broken. This
        probes the generated schema path used by ``AutoML`` construction, which
        catches issues such as ``cosmos-rl`` versus ``cosmos_rl`` package names
        before a long-running launch starts.
        """
        from tao_automl.schema.generate_schema import generate_schema

        schema = generate_schema(self.network_arch, self.action)
        return {
            "network_arch": self.network_arch,
            "action": self.action,
            "schema_title": schema.get("title"),
            "parameter_count": len(_schema_property_keys(schema)),
        }


def validate_skill_runtime(skill_dir: str | Path, action: str = "train") -> dict[str, Any]:
    """Load a skill directory and validate its AutoML runtime schema path."""
    return SkillContext(skill_dir=Path(skill_dir), action=action).validate_runtime()

_DEFAULT_POLL_INTERVAL = 30
_TERMINAL_STATUSES = {"Complete", "Error", "Canceled"}


_COSMOS_RL_SFT_VAL_RE = re.compile(
    r'\[SFT\]\s+Validation loss:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)',
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


def _extract_metric_from_logs(logs: str, metric_name: str) -> float | None:
    """Extract the final metric value from TAO training logs.

    Searches logs in reverse (last occurrence = final value). Handles:
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
    lines = logs.strip().splitlines()

    # Cosmos-RL validation line: only triggers if the literal "[SFT] Validation
    # loss:" marker is present in the logs. We do NOT hijack on the substring
    # "val" in metric_name (that was a bug: it silently failed for any non-
    # cosmos model with metric_name like "val_loss" or "val_acc").
    if "val" in metric_name.lower():
        for line in reversed(lines):
            m = _COSMOS_RL_SFT_VAL_RE.search(line)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue
        # Fall through to the generic patterns instead of returning None — a
        # PyTorch-Lightning container emitting "val_loss: 0.12" on stdout
        # should still match Pattern 2 below.

    # Pattern 1: Cosmos-RL step format "Step: N/M, Loss: X.XXXX" (most specific)
    step_pattern = re.compile(
        r'Step:\s*\d+/\d+.*?Loss:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)',
        re.IGNORECASE,
    )
    for line in reversed(lines):
        match = step_pattern.search(line)
        if match:
            try:
                val = float(match.group(1))
                if val > 0:  # Skip 0.0 values (empty validation)
                    return val
            except ValueError:
                continue

    # Pattern 2: direct metric match (case-insensitive). Lightning progress
    # output may print metrics as ``train_loss_epoch: 18.901`` or split the
    # label and value across wrapped terminal lines, so also scan a
    # whitespace-normalized view of the full log.
    metric_aliases = _metric_aliases(metric_name)
    for suffix in ("_epoch", "_step"):
        if not metric_name.endswith(suffix):
            metric_aliases.append(f"{metric_name}{suffix}")
    if metric_name.lower().startswith("val_"):
        bare_metric = metric_name[4:]
        metric_aliases.extend([
            bare_metric,
            "Validation " + bare_metric.replace("_", " "),
        ])
    normalized_logs = re.sub(r"\s+", " ", logs)
    for alias in metric_aliases:
        metric_pattern = re.compile(
            rf'(?:best\s+)?{re.escape(alias)}\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)',
            re.IGNORECASE,
        )
        for line in reversed(lines):
            match = metric_pattern.search(line)
            if match:
                try:
                    val = float(match.group(1))
                    if val >= 0:
                        return val
                except ValueError:
                    continue
        matches = list(metric_pattern.finditer(normalized_logs))
        for match in reversed(matches):
            try:
                val = float(match.group(1))
                if val >= 0:
                    return val
            except ValueError:
                continue

    # Pattern 3: KPI
    kpi_pattern = re.compile(
        r'kpi\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)', re.IGNORECASE,
    )
    for line in reversed(lines):
        match = kpi_pattern.search(line)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    # Pattern 4: Epoch line
    epoch_pattern = re.compile(
        r'[Ee]poch\s+\d+.*?(?:loss|accuracy|mIoU)\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)',
    )
    for line in reversed(lines):
        match = epoch_pattern.search(line)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


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
        aliases.append("img_bbox_NuScenes/mAP")
    if normalized == "train_loss_epoch":
        aliases.append("train_loss")
    if normalized == "train_loss":
        aliases.append("train_loss_epoch")
    if normalized in {"avg_loss", "val_avg_loss"}:
        aliases.append("avg_loss")
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
        kpi = payload.get("kpi")
        if not isinstance(kpi, dict):
            continue
        for alias in aliases:
            if alias not in kpi:
                continue
            try:
                value = float(kpi[alias])
            except (TypeError, ValueError):
                continue
            if value == value:
                return value
    return None


def _extract_metric_from_best_score_payload(
    payload: str | dict[str, Any],
    metric_name: str,
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

    aliases = set(_metric_aliases(metric_name))
    aliases.update(alias.replace("/", "_") for alias in list(aliases))
    metric_label = data.get("metric")
    if isinstance(metric_label, str):
        aliases.add(metric_label)
        aliases.add(metric_label.replace("/", "_"))

    for key in ("best_score", "best_metric", "metric_value", "score", "value"):
        if key not in data:
            continue
        try:
            value = float(data[key])
        except (TypeError, ValueError):
            continue
        if value == value:
            return value

    for key in aliases:
        if key not in data:
            continue
        try:
            value = float(data[key])
        except (TypeError, ValueError):
            continue
        if value == value:
            return value
    return None


def _extract_metric_from_best_score_file(
    best_score_path: Path,
    metric_name: str,
) -> float | None:
    if not best_score_path.exists():
        return None
    try:
        return _extract_metric_from_best_score_payload(
            best_score_path.read_text(encoding="utf-8"),
            metric_name,
        )
    except OSError:
        return None


def _extract_metric_from_local_results(job_id: str, metric_name: str,
                                       platform_kwargs: dict | None) -> float | None:
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
                best_score_path, metric_name
            )
            if metric is not None:
                return metric
        for status_path in sorted(job_root.rglob("status.json")):
            metric = _extract_metric_from_status_file(status_path, metric_name)
            if metric is not None:
                return metric
    return None


def _extract_metric_from_sdk_results(sdk, job_id: str,
                                     metric_name: str) -> float | None:
    """Recover metrics from SDK-managed result artifacts.

    Slurm jobs write results on Lustre, which may not be mounted on the local
    AutoML controller host. Newer SDKs expose ``read_job_result_file`` for that
    case; local platforms can still be handled by reading ``get_job_results_dir``.
    """
    candidates = (
        "train_output_dir/best/best_score.json",
        "results_dir/best/best_score.json",
        "best/best_score.json",
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
            metric = _extract_metric_from_best_score_payload(payload, metric_name)
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
                best_score_path, metric_name
            )
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
) -> float | None:
    local_metric = _extract_metric_from_local_results(
        job_id, metric_name, platform_kwargs
    )
    if local_metric is not None:
        return local_metric
    return _extract_metric_from_sdk_results(sdk, job_id, metric_name)


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
    for line in reversed(logs.strip().splitlines()):
        if "Execution status: PASS" in line:
            return "PASS"
        if "Execution status: FAIL" in line:
            return "FAIL"
    if not include_fatal_patterns:
        return None
    fatal_patterns = _CLEANUP_FATAL_PATTERNS + _HARD_FATAL_PATTERNS
    if any(pattern in logs for pattern in fatal_patterns):
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
    already remote URIs (treats `://` as the URI marker). Returns the list
    of dotted keys we rewrote, for logging.
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


def _validate_keys_against_schema(provided_keys, base_specs, kind, schema_keys=None):
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
    if metric is None:
        return False
    target["metric_value"] = float(metric)
    return True


# ---------------------------------------------------------------------------
# Active-jobs persistence (fix #3: survive orchestrator crashes without
# leaking in-flight Lepton jobs).
# ---------------------------------------------------------------------------

def _active_jobs_path(workspace_path: str):
    from pathlib import Path
    return Path(workspace_path) / "active_jobs.json"


def _save_active_jobs(workspace_path: str, active: dict) -> None:
    """Atomic write of {rec_id: {rec_id, job_id, submitted_at}} to disk."""
    from pathlib import Path
    p = _active_jobs_path(workspace_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(list(active.values()), indent=2))
    tmp.replace(p)


def _load_active_jobs(workspace_path: str) -> list:
    p = _active_jobs_path(workspace_path)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception as e:
        logger.warning("Couldn't read active_jobs.json: %s; starting fresh", e)
        return []


class AutoMLRunner:
    """Wires AutoML brain to SDK execution for automated HPO loops.

    The runner accepts any of the 5 platform SDKs (LeptonSDK / SlurmSDK /
    KubernetesSDK / DockerSDK / BrevSDK). It does NOT pick a platform for
    the caller — instantiate the SDK you want and pass it in.

    ``skill_dir`` is the absolute path to a model directory inside the
    skill bank (e.g. ``Path.home() / 'tao-sdk/tao-skills-external/models/dino'``).
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
            on_recommendation=None, on_result=None,
            **platform_kwargs) -> dict:
        """Run a full AutoML optimization loop.

        Args:
            train_dataset_uri: Training dataset URI (e.g. "s3://bucket/data").
            eval_dataset_uri: Eval dataset URI (optional).
            base_checkpoint: Pretrained checkpoint URI (optional).
            workspace_id: Workspace ID (default: from SDK).
            image: Docker image override. Default: from skill_info.yaml's
                ``container_image`` (resolved via tao_sdk.versions).
            automl_settings: Algorithm config (see AlgorithmParams).
            automl_hyperparameters: Param names to search, or None for schema defaults.
            custom_param_ranges: Per-param range overrides.
            workspace_path: Local path for AutoML state persistence.
            spec_overrides: Dict of spec overrides applied to base specs before
                AutoML starts. Dotted keys supported (e.g.
                {"train.epoch": 5, "policy.model_max_length": 40960}).
            resume: If True, resume from persisted state in workspace_path.
            **platform_kwargs: Forwarded to ``sdk.create_job(...)``. Pass
                whichever kwargs your platform SDK accepts (Lepton:
                ``dedicated_node_group``, ``resource_shape``, ``num_nodes``;
                SLURM: ``partition``, ``account``, ``num_nodes``;
                Kubernetes: ``namespace``, ``node_selector``, ``num_nodes``;
                Docker: ``mounts``; Brev: ``instance_id``, ``gpu_type``).
                Plus the platform-agnostic ``gpu_count`` (defaults to 1 if
                not specified).
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
                ``"wer"``). Under the hood the runner negates reported values
                when the explicit direction disagrees with the implicit rule,
                then flips them back in the returned result — callers always
                see their original metric scale.

        Returns:
            Dict with keys: best, progress, baseline, final_evaluation, history.
        """
        from tao_automl import AutoML

        automl_settings = automl_settings or {"algorithm": "bayesian", "metric": "loss"}
        workspace_id = workspace_id or getattr(self._sdk, "_workspace_id", "")
        network_arch = self.skill_ctx.network_arch

        if not resume:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            workspace_path = os.path.join(workspace_path, f"run_{ts}")
        os.makedirs(workspace_path, exist_ok=True)
        logger.info("Workspace: %s", workspace_path)

        # Skill metadata is loaded once at __init__ via SkillContext (replaces
        # the deleted SkillBank). action_cfg carries command/inputs/outputs/
        # config_format/upload_excludes — exactly what build_entrypoint takes.
        base_specs = copy.deepcopy(self.skill_ctx.default_specs)
        resolved_image = image or self.skill_ctx.container_image
        action_cfg = self.skill_ctx.action_cfg
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
                self.skill_ctx.valid_spec_keys)
            base_specs = self._merge_specs(base_specs, spec_overrides)
        if automl_hyperparameters:
            _validate_keys_against_schema(
                list(automl_hyperparameters), base_specs, "automl_hyperparameter",
                self.skill_ctx.valid_spec_keys)

        # --- fix #1: resolve explicit direction. _invert_metric tells us
        #              whether to negate values before reporting to the brain
        #              (and flip them back in the returned result).
        metric_name = automl_settings.get("metric", "loss")
        _effective_dir, invert_metric = _resolve_direction(
            metric_name, automl_settings.get("direction"))

        baseline = {
            "enabled": bool(automl_settings.get("run_baseline", True)),
            "metric_name": metric_name,
            "metric_value": None,
            "status": "not_run",
        }
        if baseline["enabled"]:
            if automl_settings.get("baseline_metric") is not None:
                baseline["metric_value"] = float(automl_settings["baseline_metric"])
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
                        baseline["metric_value"] = float(baseline_metric)
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
            resume=resume,
        )
        logger.info("Starting AutoML loop: network=%s, algorithm=%s, "
                    "metric=%s, direction=%s%s",
                     network_arch, automl_settings.get("algorithm"),
                     metric_name, _effective_dir,
                     " (values will be inverted for the brain)" if invert_metric else "")

        # Unflip values if we inverted them for the brain, so callers see
        # metrics in their original scale regardless of `direction`.
        def _unflip(v):
            if v is None:
                return None
            return -v if invert_metric else v

        # --- fix #3: if resuming, recover any jobs that were in flight when
        #              the previous orchestrator died. Poll each to terminal,
        #              report to the brain, then continue.
        if resume:
            pending = _load_active_jobs(workspace_path)
            if pending:
                logger.info("Resume: recovering %d in-flight job(s) from prior run",
                            len(pending))
                for entry in pending:
                    self._recover_pending_job(
                        entry=entry, automl=automl, metric_name=metric_name,
                        metric_extractor=metric_extractor, eval_fn=eval_fn,
                        workspace_path=workspace_path, invert_metric=invert_metric,
                        on_result=on_result,
                        platform_kwargs=platform_kwargs,
                    )

        while not automl.is_complete():
            recs = automl.next_recommendation()
            progress = automl.get_progress()
            max_recommendations = automl_settings.get("automl_max_recommendations")
            if max_recommendations is not None:
                remaining = int(max_recommendations) - int(progress.get("completed", 0))
                if remaining <= 0:
                    logger.info(
                        "AutoML recommendation budget reached (%d/%d); stopping launch loop",
                        progress.get("completed", 0), int(max_recommendations),
                    )
                    break
                if len(recs) > remaining:
                    logger.info(
                        "Capping recommendations from %d to remaining budget %d",
                        len(recs), remaining,
                    )
                    recs = recs[:remaining]
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
                    try:
                        previous_metric = _unflip(float(rec.result))
                    except (TypeError, ValueError):
                        previous_metric = None
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
                metric_value, status = self._run_one_job(
                    image=resolved_image, action_cfg=action_cfg,
                    specs=merged_specs, rec=rec, metric_name=metric_name,
                    metric_extractor=metric_extractor,
                    eval_fn=eval_fn,
                    workspace_path=workspace_path,
                    platform_kwargs=job_platform_kwargs,
                )
                if (
                    status == "metric_missing"
                    and previous_metric is not None
                    and getattr(rec, "job_id", None)
                    and _job_has_checkpoint_artifact(
                        self._sdk, rec.job_id, job_platform_kwargs
                    )
                ):
                    logger.warning(
                        "Rec %d: promoted job %s produced a checkpoint but no "
                        "fresh metric; carrying forward prior metric=%f",
                        rec.id, rec.job_id, previous_metric,
                    )
                    metric_value = previous_metric
                    status = "success"
                elif status == "metric_missing":
                    status = "failure"
                # Fail-loud on a broken extractor: if the configured metric
                # extractor (and eval_fn) both return None for N consecutive
                # recs, we're not measuring anything — raise instead of
                # letting the brain see all-failures for hours.
                if metric_value is None:
                    self._consecutive_none_metrics += 1
                    if (self._consecutive_none_metrics
                            >= self._MAX_CONSECUTIVE_NONE_METRICS):
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
                            f"job's logs to confirm.")
                else:
                    self._consecutive_none_metrics = 0
                # Report to the brain, inverting if explicit direction disagrees
                # with the brain's implicit metric-name rule.
                report_value = metric_value
                if invert_metric and report_value is not None:
                    report_value = -report_value
                automl.report_result(
                    rec_id=rec.id,
                    metric_value=report_value if report_value is not None else 0.0,
                    status=status,
                )
                if on_result:
                    on_result(rec, metric_value, status)
                logger.info("Recommendation %d: metric=%.6f, status=%s",
                            rec.id, metric_value if metric_value is not None else 0.0, status)

        best = automl.get_best()
        progress = automl.get_progress()
        history = automl.get_history()
        if best is None:
            failed = [r.id for r in history if r.status == "failure"]
            raise RuntimeError(
                "AutoML finished without a successful recommendation; "
                f"failed recommendation ids: {failed}"
            )

        best_metric = _unflip(best.result) if best else None
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
                    final_evaluation["status"] = provided_payload.get(
                        "status", "metric_missing"
                    )
                final_evaluation["source"] = provided_payload.get("source", "provided")
            elif automl_settings.get("final_evaluation_metric") is not None:
                final_evaluation["metric_value"] = float(
                    automl_settings["final_evaluation_metric"]
                )
                final_evaluation["status"] = "provided"
                final_evaluation["source"] = "automl_settings.final_evaluation_metric"
            elif final_eval_fn is not None:
                try:
                    payload = final_eval_fn(best, getattr(best, "job_id", None))
                except Exception as ex:
                    logger.warning("final_eval_fn raised for rec %s: %s", best.id, ex)
                    final_evaluation["status"] = "failure"
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

        result = {
            "best": {
                "rec_id": best.id if best else None,
                "specs": best.specs if best else {},
                "metric_value": best_metric,
                "adjustments": getattr(best, "adjustments", []) if best else [],
            },
            "progress": progress,
            "baseline": baseline,
            "final_evaluation": final_evaluation,
            "history": [{"rec_id": r.id, "metric": _unflip(r.result),
                          "status": r.status,
                          "failure_reason": getattr(r, "failure_reason", None),
                          "adjustments": getattr(r, "adjustments", [])}
                         for r in history],
        }
        baseline["comparison_to_best"] = _compare_to_baseline(
            baseline.get("metric_value"),
            result["best"]["metric_value"],
            _effective_dir,
        )
        logger.info("AutoML complete: %d recommendations, best metric=%.6f (rec %s)",
                     progress["completed"],
                     _unflip(best.result) if best and best.result is not None else 0.0,
                     best.id if best else "N/A")
        return result

    def _run_one_job(self, image, action_cfg, specs, rec, metric_name,
                     metric_extractor=None,
                     eval_fn=None,
                     workspace_path=None,
                     platform_kwargs=None) -> tuple[float | None, str]:
        """Launch a single training job and wait for it to finish.

        Builds a container command via ``tao_sdk.script_runner.build_entrypoint``
        (inlines the in-container runner heredoc) and submits via the platform
        SDK's ``create_job(image, command, **platform_kwargs)``. Output
        destinations are resolved at runtime in the container from
        ``TAO_RESULTS_ROOT`` / ``S3_BUCKET_NAME`` env vars the SDK injects.
        """
        from tao_sdk.script_runner import build_entrypoint

        try:
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
        except Exception as e:
            logger.error("Failed to create job for rec %d: %s", rec.id, e)
            rec.failure_reason = f"job_creation_failed: {e}"
            return None, "failure"

        rec.assign_job_id(job.id)
        self._active_jobs[rec.id] = job.id
        # Persist in-flight state so a resume can recover it.
        if workspace_path:
            self._persist_active_jobs(workspace_path)
        logger.info("Rec %d: job %s submitted (backend: %s)",
                    rec.id, job.id, getattr(job, "backend_job_id", job.id))

        # Caller can plug in a custom extractor; fall back to the built-in.
        extract_fn = metric_extractor or _extract_metric_from_logs

        # Poll status AND logs simultaneously — Lepton clears logs fast after
        # completion, so we cache the best metric seen during polling.
        cached_metric = None
        cached_exec_status = None
        all_logs = ""
        job_status = None

        while True:
            time.sleep(self._poll_interval)

            # Read logs every poll cycle to cache metrics before they expire
            try:
                logs = self._sdk.get_job_logs(job.id)
                if logs:
                    all_logs = logs  # Keep latest snapshot
                    try:
                        m = extract_fn(logs, metric_name)
                    except Exception as ex:
                        logger.warning("metric_extractor raised for rec %d: %s",
                                       rec.id, ex)
                        m = None
                    if m is not None:
                        cached_metric = m
                    es = _check_execution_status(
                        logs, include_fatal_patterns=False
                    )
                    hard_failure = _has_hard_failure_pattern(logs)
                    if not es and (cached_metric is None or hard_failure):
                        es = _check_execution_status(logs)
                        if es == "FAIL":
                            artifact_metric = _recover_metric_from_artifacts(
                                self._sdk, job.id, metric_name, platform_kwargs
                            )
                            if artifact_metric is not None:
                                cached_metric = artifact_metric
                                if not hard_failure:
                                    es = None
                                    logger.info(
                                        "Rec %d: ignoring cleanup failure text "
                                        "after recovering metric=%f from result "
                                        "artifacts",
                                        rec.id, artifact_metric,
                                    )
                                else:
                                    logger.warning(
                                        "Rec %d: recovered metric=%f from result "
                                        "artifacts before canceling hard failed job",
                                        rec.id, artifact_metric,
                                    )
                    if es:
                        cached_exec_status = es
                        if es == "FAIL":
                            logger.warning(
                                "Rec %d: job %s logs show execution failure; "
                                "canceling backend job",
                                rec.id, job.id,
                            )
                            try:
                                self._sdk.cancel_job(job.id)
                            except Exception as ex:
                                logger.warning(
                                    "Failed to cancel failed job %s for rec %d: %s",
                                    job.id, rec.id, ex,
                                )
                            break
            except Exception:
                pass

            if cached_exec_status == "FAIL":
                break

            try:
                job_status = self._sdk.get_job_status(job.id)
            except Exception as e:
                logger.warning("Failed to get status for job %s: %s", job.id, e)
                continue
            if job_status.status in _TERMINAL_STATUSES:
                break

        # Final log read (may be empty if Lepton already cleaned up)
        try:
            final_logs = self._sdk.get_job_logs(job.id)
            if final_logs:
                all_logs = final_logs
                try:
                    m = extract_fn(final_logs, metric_name)
                except Exception as ex:
                    logger.warning("metric_extractor raised for rec %d: %s",
                                   rec.id, ex)
                    m = None
                if m is not None:
                    cached_metric = m
                es = _check_execution_status(
                    final_logs,
                    include_fatal_patterns=(
                        cached_metric is None
                        or _has_hard_failure_pattern(final_logs)
                    ),
                )
                if es:
                    cached_exec_status = es
        except Exception:
            pass

        if cached_metric is None:
            artifact_metric = _recover_metric_from_artifacts(
                self._sdk, job.id, metric_name, platform_kwargs
            )
            if artifact_metric is not None:
                cached_metric = artifact_metric
                logger.info(
                    "Rec %d: recovered metric=%f from result artifacts "
                    "before final status classification",
                    rec.id, artifact_metric,
                )

        exec_status = cached_exec_status or _check_execution_status(
            all_logs,
            include_fatal_patterns=(
                cached_metric is None or _has_hard_failure_pattern(all_logs)
            ),
        )
        status = job_status.status if job_status is not None else "Error"

        # fix #3: job has reached terminal state — clear it from active_jobs.json.
        self._active_jobs.pop(rec.id, None)
        if workspace_path:
            self._persist_active_jobs(workspace_path)

        if status == "Error" or exec_status == "FAIL":
            reason = _classify_failure(all_logs)
            if reason:
                rec.failure_reason = reason
            logger.warning("Rec %d: job %s failed", rec.id, job.id)
            return cached_metric, "failure"
        if status == "Canceled":
            rec.failure_reason = "job_canceled"
            return cached_metric, "failure"

        # fix #4: if an eval_fn is provided, run it post-training and let its
        # return override the log-extracted metric. Errors are isolated.
        metric_value = cached_metric
        eval_metric_used = False
        if eval_fn is not None:
            try:
                eval_metric = eval_fn(rec, job.id)
            except Exception as ex:
                logger.warning("eval_fn raised for rec %d: %s; falling back "
                                "to log-extracted metric", rec.id, ex)
                eval_metric = None
            if eval_metric is not None:
                logger.info("Rec %d: eval_fn returned metric=%f "
                            "(overriding log-extracted %s)",
                            rec.id, eval_metric,
                            f"{cached_metric:.6f}" if cached_metric is not None else "None")
                metric_value = eval_metric
                eval_metric_used = True
        local_metric = _extract_metric_from_local_results(
            job.id, metric_name, platform_kwargs
        )
        if local_metric is not None and not eval_metric_used:
            if metric_value is None:
                logger.info("Rec %d: recovered metric=%f from local status artifacts",
                            rec.id, local_metric)
            elif local_metric != metric_value:
                logger.info(
                    "Rec %d: using local status metric=%f instead of "
                    "log-extracted metric=%f",
                    rec.id, local_metric, metric_value,
                )
            metric_value = local_metric

        if metric_value is None:
            logger.warning("Rec %d: job %s completed but no metric could be "
                           "extracted (neither metric_extractor nor eval_fn "
                           "produced a value for '%s')",
                           rec.id, job.id, metric_name)
            return None, "metric_missing"

        logger.info("Rec %d: job %s succeeded, metric=%f", rec.id, job.id, metric_value)
        return metric_value, "success"

    def _persist_active_jobs(self, workspace_path: str) -> None:
        """Dump self._active_jobs to workspace/active_jobs.json atomically."""
        now_iso = datetime.now(timezone.utc).isoformat()
        snapshot = {
            rec_id: {"rec_id": rec_id, "job_id": job_id, "updated_at": now_iso}
            for rec_id, job_id in self._active_jobs.items()
        }
        try:
            _save_active_jobs(workspace_path, snapshot)
        except Exception as e:
            logger.warning("Failed to persist active_jobs.json: %s", e)

    def _recover_pending_job(self, entry, automl, metric_name,
                              metric_extractor, eval_fn, workspace_path,
                              invert_metric, on_result, platform_kwargs=None) -> None:
        """Poll an in-flight job (recovered on resume), extract its result,
        and report it to the brain. Mirrors the tail of _run_one_job.
        """
        rec_id = entry["rec_id"]
        job_id = entry["job_id"]

        # Find the matching Recommendation object in the brain's history so
        # we can pass it to on_result/eval_fn and update rec.assign_job_id.
        rec = next((r for r in automl.get_history() if r.id == rec_id), None)
        if rec is None:
            logger.warning("Resume: rec %d not in brain history; dropping pending job %s",
                           rec_id, job_id)
            return

        self._active_jobs[rec_id] = job_id
        extract_fn = metric_extractor or _extract_metric_from_logs

        logger.info("Resume: polling rec %d job %s", rec_id, job_id)
        cached_metric = None
        cached_exec_status = None
        all_logs = ""
        job_status = None

        # Poll until terminal.
        while True:
            time.sleep(self._poll_interval)
            try:
                logs = self._sdk.get_job_logs(job_id)
                if logs:
                    all_logs = logs
                    try:
                        m = extract_fn(logs, metric_name)
                    except Exception as ex:
                        logger.warning("metric_extractor raised during resume "
                                        "for rec %d: %s", rec_id, ex)
                        m = None
                    if m is not None:
                        cached_metric = m
                    es = _check_execution_status(
                        logs, include_fatal_patterns=False
                    )
                    hard_failure = _has_hard_failure_pattern(logs)
                    if not es and (cached_metric is None or hard_failure):
                        es = _check_execution_status(logs)
                        if es == "FAIL":
                            artifact_metric = _recover_metric_from_artifacts(
                                self._sdk, job_id, metric_name, platform_kwargs
                            )
                            if artifact_metric is not None:
                                cached_metric = artifact_metric
                                if not hard_failure:
                                    es = None
                                    logger.info(
                                        "Resume: rec %d ignoring cleanup "
                                        "failure text after recovering metric=%f "
                                        "from result artifacts",
                                        rec_id, artifact_metric,
                                    )
                                else:
                                    logger.warning(
                                        "Resume: rec %d recovered metric=%f from "
                                        "result artifacts before canceling hard "
                                        "failed job",
                                        rec_id, artifact_metric,
                                    )
                    if es:
                        cached_exec_status = es
                        if es == "FAIL":
                            logger.warning(
                                "Resume: rec %d job %s logs show execution "
                                "failure; canceling backend job",
                                rec_id, job_id,
                            )
                            try:
                                self._sdk.cancel_job(job_id)
                            except Exception as ex:
                                logger.warning(
                                    "Resume: failed to cancel failed job %s for "
                                    "rec %d: %s",
                                    job_id, rec_id, ex,
                                )
                            break
            except Exception:
                pass
            if cached_exec_status == "FAIL":
                break
            try:
                job_status = self._sdk.get_job_status(job_id)
            except Exception as e:
                logger.warning("Resume: failed to get status for job %s: %s",
                               job_id, e)
                continue
            if job_status.status in _TERMINAL_STATUSES:
                break

        # Final log read
        try:
            final_logs = self._sdk.get_job_logs(job_id)
            if final_logs:
                try:
                    m = extract_fn(final_logs, metric_name)
                except Exception:
                    m = None
                if m is not None:
                    cached_metric = m
                es = _check_execution_status(
                    final_logs,
                    include_fatal_patterns=(
                        cached_metric is None
                        or _has_hard_failure_pattern(final_logs)
                    ),
                )
                if es:
                    cached_exec_status = es
        except Exception:
            pass

        if cached_metric is None:
            artifact_metric = _recover_metric_from_artifacts(
                self._sdk, job_id, metric_name, platform_kwargs
            )
            if artifact_metric is not None:
                cached_metric = artifact_metric
                logger.info(
                    "Resume: rec %d recovered metric=%f from result "
                    "artifacts before final status classification",
                    rec_id, artifact_metric,
                )

        exec_status = cached_exec_status or _check_execution_status(
            all_logs,
            include_fatal_patterns=(
                cached_metric is None or _has_hard_failure_pattern(all_logs)
            ),
        )
        status = job_status.status if job_status is not None else "Error"
        self._active_jobs.pop(rec_id, None)
        self._persist_active_jobs(workspace_path)

        if status == "Error" or exec_status == "FAIL":
            metric_value = cached_metric
            report_status = "failure"
            reason = _classify_failure(all_logs)
            if reason:
                rec.failure_reason = reason
        elif status == "Canceled":
            metric_value = cached_metric
            report_status = "failure"
            rec.failure_reason = "job_canceled"
        else:
            metric_value = cached_metric
            eval_metric_used = False
            if eval_fn is not None:
                try:
                    em = eval_fn(rec, job_id)
                except Exception as ex:
                    logger.warning("eval_fn raised during resume for rec %d: %s",
                                    rec_id, ex)
                    em = None
                if em is not None:
                    metric_value = em
                    eval_metric_used = True
            local_metric = _extract_metric_from_local_results(
                job_id, metric_name, platform_kwargs
            )
            if local_metric is not None and not eval_metric_used:
                if metric_value is None:
                    logger.info(
                        "Resume: rec %d recovered metric=%f from local status "
                        "artifacts",
                        rec_id, local_metric,
                    )
                elif local_metric != metric_value:
                    logger.info(
                        "Resume: rec %d using local status metric=%f instead "
                        "of log-extracted metric=%f",
                        rec_id, local_metric, metric_value,
                    )
                metric_value = local_metric
            report_status = "success" if metric_value is not None else "failure"

        report_value = metric_value
        if invert_metric and report_value is not None:
            report_value = -report_value
        automl.report_result(
            rec_id=rec_id,
            metric_value=report_value if report_value is not None else 0.0,
            status=report_status,
        )
        if on_result:
            try:
                on_result(rec, metric_value, report_status)
            except Exception as ex:
                logger.warning("on_result callback raised during resume: %s", ex)
        logger.info("Resume: rec %d %s metric=%s",
                    rec_id, report_status,
                    f"{metric_value:.6f}" if metric_value is not None else "None")

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
        """Deep-merge recommendation specs into base specs."""
        import copy
        merged = copy.deepcopy(base_specs)
        for key, value in rec_specs.items():
            AutoMLRunner._set_nested(merged, key, value)
        return merged


_PLATFORMS = ("lepton", "slurm", "kubernetes", "docker", "brev")


def _make_sdk(platform: str):
    """Construct a platform SDK by name. No default — caller must pick.

    Matches platform/tao-sdk/SKILL.md's "It does not select platforms
    automatically" stance: none of the 5 SDKs is a sensible default
    (Lepton biases DGX Cloud, SLURM biases on-prem clusters, etc.).
    """
    if platform == "lepton":
        from tao_sdk.platforms.lepton import LeptonSDK
        return LeptonSDK()
    if platform == "slurm":
        from tao_sdk.platforms.slurm import SlurmSDK
        return SlurmSDK()
    if platform == "kubernetes":
        from tao_sdk.platforms.kubernetes import KubernetesSDK
        return KubernetesSDK()
    if platform == "docker":
        from tao_sdk.platforms.docker import DockerSDK
        return DockerSDK()
    if platform == "brev":
        from tao_sdk.platforms.brev import BrevSDK
        return BrevSDK()
    raise ValueError(
        f"Unknown platform {platform!r}. Choose one of: {', '.join(_PLATFORMS)}."
    )


def run_automl_plan(plan: dict, platform: str) -> dict:
    """Execute an AutoML plan file on the chosen platform.

    The plan JSON's ``params`` block must include ``skill_dir`` (absolute
    path to a model directory inside tao-skills-external). Per-platform
    create_job kwargs go under ``params.platform_kwargs``.
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
                         "(absolute path to a model dir in tao-skills-external).")
    action = params.get("action", "train")
    platform_kwargs = params.get("platform_kwargs") or {}

    sdk = _make_sdk(platform)
    runner = AutoMLRunner(sdk=sdk, skill_dir=skill_dir, action=action)
    global _runner
    _runner = runner
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
        **platform_kwargs,
    )
    print(json.dumps(result, indent=2, default=str))
    return result


_runner = None

def _signal_handler(signum, frame):
    if _runner and _runner._active_jobs:
        for rec_id, job_id in _runner._active_jobs.items():
            try:
                _runner._sdk.cancel_job(job_id)
                print(f"Canceled job {job_id} (rec {rec_id})")
            except Exception as e:
                print(f"Failed to cancel job {job_id}: {e}")
    sys.exit(1)

signal.signal(signal.SIGINT, _signal_handler)


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
