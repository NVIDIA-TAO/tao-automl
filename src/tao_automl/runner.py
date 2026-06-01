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
        self.network_arch = self.skill_info.get("network_arch", self.skill_dir.name).replace("-", "_")

        template_path = self.skill_dir / f"references/spec_template_{self.action}.yaml"
        self.default_specs = (
            yaml.safe_load(template_path.read_text()) if template_path.exists() else {}
        ) or {}

        # Container image: accept either a versions.yaml key or an absolute URI.
        from tao_sdk.versions import resolve_container_image
        self.container_image = resolve_container_image(
            self.skill_info.get("container_image", "")
        )

_DEFAULT_POLL_INTERVAL = 30
_TERMINAL_STATUSES = {"Complete", "Error", "Canceled"}


_COSMOS_RL_SFT_VAL_RE = re.compile(
    r'\[SFT\]\s+Validation loss:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)',
    re.IGNORECASE,
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

    # Pattern 2: direct metric match (case-insensitive)
    metric_pattern = re.compile(
        rf'(?:best\s+)?{re.escape(metric_name)}\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)',
        re.IGNORECASE,
    )
    for line in reversed(lines):
        match = metric_pattern.search(line)
        if match:
            try:
                val = float(match.group(1))
                if val > 0:  # Skip 0.0 values
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


def _check_execution_status(logs: str) -> str | None:
    """Check if logs contain Execution status: PASS or FAIL."""
    if not logs:
        return None
    for line in reversed(logs.strip().splitlines()):
        if "Execution status: PASS" in line:
            return "PASS"
        if "Execution status: FAIL" in line:
            return "FAIL"
    return None


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

def _flatten_keys(d: dict, prefix: str = "") -> set[str]:
    """Recursively flatten a nested spec dict into dotted keys."""
    keys: set[str] = set()
    if not isinstance(d, dict):
        return keys
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else str(k)
        keys.add(full)
        if isinstance(v, dict):
            keys |= _flatten_keys(v, full)
    return keys


def _validate_keys_against_schema(provided_keys, base_specs, kind):
    """Raise ValueError on provided keys that look like typos of existing
    schema keys. Accepts genuinely-new keys (logs a warning) so users who
    intentionally add a new spec field aren't blocked.
    """
    import difflib
    base_keys = _flatten_keys(base_specs)
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

    def run(self, train_dataset_uri, eval_dataset_uri="",
            base_checkpoint="", workspace_id=None, image=None,
            automl_settings=None,
            automl_hyperparameters=None, custom_param_ranges=None,
            workspace_path="./automl_workspace",
            spec_overrides=None, resume=False,
            metric_extractor=None,
            eval_fn=None,
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
            Dict with keys: best, progress, history.
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
                list(spec_overrides.keys()), base_specs, "spec_override")
            base_specs = self._merge_specs(base_specs, spec_overrides)
        if automl_hyperparameters:
            _validate_keys_against_schema(
                list(automl_hyperparameters), base_specs, "automl_hyperparameter")

        # --- fix #1: resolve explicit direction. _invert_metric tells us
        #              whether to negate values before reporting to the brain
        #              (and flip them back in the returned result).
        metric_name = automl_settings.get("metric", "loss")
        _effective_dir, invert_metric = _resolve_direction(
            metric_name, automl_settings.get("direction"))

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
                    )

        while not automl.is_complete():
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
                merged_specs = self._merge_specs(base_specs, rec.specs)
                # Output destination is resolved at runtime by script_runner
                # from TAO_RESULTS_ROOT (mount) / S3_BUCKET_NAME (cloud) env
                # vars the SDK injects. The agent doesn't pre-rewrite spec
                # output keys here — that lived in the deleted SDK contract.
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
                metric_value, status = self._run_one_job(
                    image=resolved_image, action_cfg=action_cfg,
                    specs=merged_specs, rec=rec, metric_name=metric_name,
                    metric_extractor=metric_extractor,
                    eval_fn=eval_fn,
                    workspace_path=workspace_path,
                    platform_kwargs=platform_kwargs,
                )
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

        # Unflip values if we inverted them for the brain, so callers see
        # metrics in their original scale regardless of `direction`.
        def _unflip(v):
            if v is None:
                return None
            return -v if invert_metric else v

        result = {
            "best": {
                "rec_id": best.id if best else None,
                "specs": best.specs if best else {},
                "metric_value": _unflip(best.result) if best else None,
            },
            "progress": progress,
            "history": [{"rec_id": r.id, "metric": _unflip(r.result),
                          "status": r.status} for r in history],
        }
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
                    es = _check_execution_status(logs)
                    if es:
                        cached_exec_status = es
            except Exception:
                pass

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
                es = _check_execution_status(final_logs)
                if es:
                    cached_exec_status = es
        except Exception:
            pass

        exec_status = cached_exec_status or _check_execution_status(all_logs)
        status = job_status.status

        # fix #3: job has reached terminal state — clear it from active_jobs.json.
        self._active_jobs.pop(rec.id, None)
        if workspace_path:
            self._persist_active_jobs(workspace_path)

        if status == "Error" or exec_status == "FAIL":
            logger.warning("Rec %d: job %s failed", rec.id, job.id)
            return cached_metric, "failure"
        if status == "Canceled":
            return None, "failure"

        # fix #4: if an eval_fn is provided, run it post-training and let its
        # return override the log-extracted metric. Errors are isolated.
        metric_value = cached_metric
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

        if metric_value is None:
            logger.warning("Rec %d: job %s completed but no metric could be "
                           "extracted (neither metric_extractor nor eval_fn "
                           "produced a value for '%s')",
                           rec.id, job.id, metric_name)
            return None, "failure"

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
                              invert_metric, on_result) -> None:
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
                    es = _check_execution_status(logs)
                    if es:
                        cached_exec_status = es
            except Exception:
                pass
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
                es = _check_execution_status(final_logs)
                if es:
                    cached_exec_status = es
        except Exception:
            pass

        exec_status = cached_exec_status or _check_execution_status(all_logs)
        status = job_status.status
        self._active_jobs.pop(rec_id, None)
        self._persist_active_jobs(workspace_path)

        if status == "Error" or exec_status == "FAIL":
            metric_value = cached_metric
            report_status = "failure"
        elif status == "Canceled":
            metric_value = None
            report_status = "failure"
        else:
            metric_value = cached_metric
            if eval_fn is not None:
                try:
                    em = eval_fn(rec, job_id)
                except Exception as ex:
                    logger.warning("eval_fn raised during resume for rec %d: %s",
                                    rec_id, ex)
                    em = None
                if em is not None:
                    metric_value = em
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
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value

    @staticmethod
    def _get_nested(source: dict, dotted_key: str):
        """Read target[a][b][c] for dotted_key 'a.b.c'; None if missing."""
        parts = dotted_key.split(".")
        cursor = source
        for part in parts:
            if not isinstance(cursor, dict) or part not in cursor:
                return None
            cursor = cursor[part]
        return cursor

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
            parts = key.split(".")
            target = merged
            for part in parts[:-1]:
                if part not in target or not isinstance(target[part], dict):
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value
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
