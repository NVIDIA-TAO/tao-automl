# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA TAO AutoML - Standalone hyperparameter search library.

Usage::

    from tao_automl import AutoML

    automl = AutoML(
        workspace="/path/to/workspace",
        network="dino",
        train_specs=specs,
        settings={"algorithm": "bayesian", "metric": "loss"},
        action="train",
    )

    while not automl.is_complete():
        rec = automl.next_recommendation()
        if not rec:
            continue
        # caller runs training with rec.specs (or rec[0].specs for parallel)
        metric_value = run_training(rec[0].specs)
        automl.report_result(rec[0].id, metric_value)

    best = automl.get_best()
"""

__version__ = "0.1.0"

import copy
import json
import logging
import os
import uuid
from collections.abc import Mapping

from tao_automl.formatting import format_result  # noqa: F401  (public API)
from tao_automl.objectives import parse_objective_config
from tao_automl.types import AutoMLContext, JobStates

logger = logging.getLogger(__name__)


def query_status(workspace_path: str) -> dict:
    """Query experiment status from a workspace without a live runner.

    Reads the persisted state files to reconstruct the current status.
    Safe to call from a separate process while the runner is active.

    Args:
        workspace_path: Path to the AutoML workspace directory.
            Since an ad-hoc ``AutoMLRunner.run()`` appends a timestamped,
            collision-safe suffix (e.g. ``run_20260423_183015_ab12cd``) to the
            base path, pass the
            full suffixed path here — or iterate over subdirectories
            of the base path.

    Returns:
        dict with keys:

        - ``progress``: ``{completed, failed, pending, total, best_metric, best_rec_id, algorithm}``
        - ``best``: ``{rec_id, specs, metric_value}``
        - ``recommendations``: list of per-rec dicts
        - ``active_jobs``: list of ``{rec_id, job_id, updated_at}`` for in-flight jobs
        - ``experiment_id``: the experiment session ID

        Returns a dict with ``error`` key if the workspace has no state.

    Example::

        from tao_automl import query_status

        status = query_status("./my_experiment")
        print(f"Progress: {status['progress']['completed']}/{status['progress']['total']}")
        print(f"Best mAP: {status['best']['metric_value']}")
        for rec in status['recommendations']:
            print(f"  Rec {rec['rec_id']}: {rec['status']} metric={rec['metric_value']}")
    """
    automl_dir = os.path.join(workspace_path, ".automl")
    if not os.path.isdir(automl_dir):
        return {"error": f"No AutoML state found at {workspace_path}"}

    controller_dir = os.path.join(automl_dir, "controller")
    if not os.path.isdir(controller_dir):
        return {"error": "No controller state found"}

    controller_files = [f for f in os.listdir(controller_dir) if f.endswith(".json")]
    if not controller_files:
        return {"error": "No experiment data found"}

    controller_files.sort(
        key=lambda f: os.path.getmtime(os.path.join(controller_dir, f)),
        reverse=True,
    )
    experiment_id = controller_files[0].replace(".json", "")

    controller_path = os.path.join(controller_dir, f"{experiment_id}.json")
    with open(controller_path) as f:
        recs = json.load(f)
    from tao_automl.recommendation_audit import validate_recommendation_audit
    for record in recs:
        audit = record.get("recommendation_audit", {})
        if audit:
            validate_recommendation_audit(audit)

    best_rec_path = os.path.join(automl_dir, "best_rec", f"{experiment_id}.json")
    best_info = None
    if os.path.exists(best_rec_path):
        with open(best_rec_path) as f:
            best_info = json.load(f)

    active_jobs_path = os.path.join(workspace_path, "active_jobs.json")
    active_jobs = []
    if os.path.exists(active_jobs_path):
        with open(active_jobs_path) as f:
            active_jobs = json.load(f)

    ptm_runtime_path = os.path.join(
        workspace_path,
        "ptm_runtime_manifest.json",
    )
    ptm_runtime = None
    if os.path.exists(ptm_runtime_path):
        with open(ptm_runtime_path, encoding="utf-8") as f:
            ptm_runtime = json.load(f)
        if (
            not isinstance(ptm_runtime, dict)
            or set(ptm_runtime) != {"manifest", "manifest_sha256"}
            or not isinstance(ptm_runtime.get("manifest"), dict)
            or not isinstance(ptm_runtime.get("manifest_sha256"), str)
        ):
            raise ValueError(
                "Persisted PTM runtime manifest has an invalid record shape"
            )
        from tao_automl.recommendation_audit import canonical_audit_sha256
        if (
            canonical_audit_sha256(ptm_runtime["manifest"])
            != ptm_runtime["manifest_sha256"]
        ):
            raise ValueError(
                "Persisted PTM runtime manifest failed integrity verification"
            )

    terminal = {JobStates.success, JobStates.done, JobStates.failure, JobStates.error}
    completed = [r for r in recs if r.get("status") in terminal]
    succeeded = [r for r in completed if r.get("status") in (JobStates.success, JobStates.done)]
    failed = [r for r in completed if r.get("status") in (JobStates.failure, JobStates.error)]
    pending = [r for r in recs if r.get("status") not in terminal]

    brain_path = os.path.join(automl_dir, "brain", f"{experiment_id}.json")
    algorithm = None
    total = len(recs)
    if os.path.exists(brain_path):
        with open(brain_path) as f:
            brain_info = json.load(f)
        algorithm = brain_info.get("algorithm")
        max_recs = brain_info.get("max_recommendations")
        if max_recs:
            total = max_recs

    best = {}
    if best_info:
        bd = best_info.get("rec_data", {})
        objective_values = bd.get("objective_values") or {}
        metric_name = bd.get("metric")
        best = {
            "rec_id": bd.get("id"),
            "specs": bd.get("specs", {}),
            "metric_value": objective_values.get(metric_name, bd.get("result")),
            "objective_score": bd.get("objective_score", bd.get("result")),
            "objective_values": objective_values,
        }

    status = {
        "experiment_id": experiment_id,
        "progress": {
            "completed": len(completed),
            "succeeded": len(succeeded),
            "failed": len(failed),
            "pending": len(pending),
            "total": total,
            "best_metric": best.get("metric_value"),
            "best_rec_id": best.get("rec_id"),
            "algorithm": algorithm,
        },
        "best": best,
        "recommendations": [
            {
                "rec_id": r.get("id"),
                "specs": r.get("specs", {}),
                "job_id": r.get("job_id"),
                "status": r.get("status"),
                "metric_value": (
                    (r.get("objective_values") or {}).get(r.get("metric"))
                    if r.get("objective_values") else r.get("result")
                ),
                "objective_score": r.get("objective_score", r.get("result")),
                "objective_values": r.get("objective_values", {}),
                "recommendation_audit": copy.deepcopy(
                    r.get("recommendation_audit", {})
                ),
                "created_on": r.get("created_on"),
                "last_modified": r.get("last_modified"),
            }
            for r in recs
        ],
        "active_jobs": active_jobs,
    }
    if ptm_runtime is not None:
        status["ptm_runtime"] = copy.deepcopy(ptm_runtime)
    return status


def _as_option_list(value):
    """Normalize an option field to a list."""
    if isinstance(value, str):
        return [item.strip() for item in value.split(",")]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _sanitize_custom_param_ranges(custom_param_ranges, param_records):
    """Constrain caller-provided categorical ranges to the generated schema."""
    param_options = {
        record.get("parameter"): record.get("valid_options")
        for record in param_records
        if isinstance(record, dict)
    }
    sanitized = copy.deepcopy(custom_param_ranges)
    for parameter, range_cfg in sanitized.items():
        if not isinstance(range_cfg, dict) or "valid_options" not in range_cfg:
            continue
        allowed_options = param_options.get(parameter)
        if not allowed_options:
            continue
        requested = _as_option_list(range_cfg["valid_options"])
        allowed = _as_option_list(allowed_options)
        filtered = [option for option in requested if option in allowed]
        dropped = [option for option in requested if option not in allowed]
        if dropped:
            logger.warning(
                "Dropped invalid custom options for %s: %s. Allowed options: %s",
                parameter,
                dropped,
                allowed,
            )
        range_cfg["valid_options"] = filtered or allowed
    return sanitized


def _resolve_session_id(state_store, settings, *, resume):
    """Resolve one persisted session for resume without guessing."""
    requested = settings.get("session_id")
    if requested is not None:
        if not isinstance(requested, str) or not requested.strip():
            raise ValueError("settings.session_id must be a non-empty string")
        return requested.strip()
    if not resume:
        return uuid.uuid4().hex[:12]

    persisted_ids = state_store.list_job_spec_ids()
    if not persisted_ids:
        raise ValueError(
            "Cannot resume AutoML: the workspace contains no persisted session; "
            "set resume=False to start a new run"
        )
    if len(persisted_ids) != 1:
        raise ValueError(
            "Cannot infer an AutoML resume session from a workspace containing "
            f"{len(persisted_ids)} sessions; set settings.session_id explicitly"
        )
    return persisted_ids[0]


def _resume_compatible_value(value):
    """Return a canonical strict-JSON representation for identity checks."""
    from tao_automl.recommendation_audit import audit_json_value

    return audit_json_value(value)


def _require_resume_match(*, label, persisted, requested):
    """Fail before construction when a persisted input would be overwritten."""
    if persisted is None:
        raise ValueError(
            f"Cannot resume AutoML: persisted {label} is missing or unreadable"
        )
    if _resume_compatible_value(persisted) != _resume_compatible_value(requested):
        raise ValueError(
            f"Cannot resume AutoML with different {label}; the persisted "
            "workspace was left unchanged"
        )


class AutoML:
    """Main entry point for TAO AutoML hyperparameter optimization.

    The caller is responsible for running training; this class only manages
    the search loop (generating recommendations, tracking results, deciding
    when to stop).

    Example::

        automl = AutoML(
            workspace="/tmp/my_experiment",
            network="dino",
            train_specs=my_train_spec_dict,
            settings={"algorithm": "bayesian", "metric": "loss",
                       "automl_max_recommendations": 20},
            action="train",
        )

        while not automl.is_complete():
            recs = automl.next_recommendation()
            for rec in recs:
                metric_value = train_model(rec.specs)
                automl.report_result(rec.id, metric_value)

        print("Best:", automl.get_best().specs)
    """

    def __init__(
        self,
        workspace,
        network,
        train_specs,
        settings,
        automl_hyperparameters=None,
        custom_param_ranges=None,
        action="train",
        resume=False,
        wandb_config=None,
        search_schema=None,
        resolved_ptm_inventory: "ResolvedPTMRuntimeInventory | None" = None,
    ):
        """
        Args:
            workspace: Path to workspace directory for state persistence.
            network: Network architecture name (e.g. ``"dino"``).
            train_specs: Action spec dict (the base configuration). The name
                is retained for compatibility with existing training callers.
            action: TAO action whose schema/search space should drive the
                optimization loop (for example ``"train"``, ``"distill"``,
                ``"prune"``, or ``"quantize"``).
            settings: Dict with keys ``algorithm``, ``metric``, and any
                algorithm-specific parameters accepted by
                :class:`~tao_automl.brain.factory.AlgorithmParams`.
            automl_hyperparameters: List of dotted parameter names to search.
                If *None*, every parameter marked ``automl_enabled`` in the
                network schema is included.
            custom_param_ranges: Optional dict mapping parameter names to
                custom range overrides (e.g.
                ``{"train.optim.lr": {"valid_min": 1e-5, "valid_max": 1e-2}}``).
            resume: Whether to resume from previously persisted state in
                *workspace*.
            wandb_config: Optional dict for WandB integration. Keys:
                ``enabled`` (bool), ``project``, ``entity``, ``api_key``,
                ``group``. Pass ``{"enabled": True}`` to activate; the
                API key can also come from ``WANDB_API_KEY`` env var.
            search_schema: Optional JSON schema describing the search space.
                When omitted, the schema is generated from the built-in TAO
                configuration module for ``network``. Supplying a schema lets
                external model scripts define searchable parameters without a
                corresponding ``tao_automl.config.<network>`` package.
            resolved_ptm_inventory: Optional live, typed
                :class:`tao_automl.ptm_runtime.ResolvedPTMRuntimeInventory`.
                When supplied, AutoML derives a conditional search space from
                every PTM-effective base spec and constructs the hierarchical
                native-Bayesian runtime. This argument never performs
                checkpoint resolution, download, or preflight.
        """
        # Lazy imports to avoid pulling in heavy deps (requests, omegaconf)
        # at package import time.
        from tao_automl.brain.factory import AlgorithmParams, BrainFactory
        from tao_automl.controller.controller import Controller
        from tao_automl.search_space.params import generate_hyperparams_to_search
        from tao_automl.state.state_store import StateStore

        if not settings or "algorithm" not in settings:
            raise ValueError("settings must include at least an 'algorithm' key")

        algorithm = settings["algorithm"]
        objective_config = parse_objective_config(settings)
        metric = objective_config.primary_metric
        brain_metric = objective_config.brain_metric

        # 1. State store
        self._state_store = StateStore(workspace)

        # 2. Context. A resume without an explicit identity is safe only when
        # the workspace contains exactly one persisted session.
        session_id = _resolve_session_id(
            self._state_store,
            settings,
            resume=resume,
        )
        self._context = AutoMLContext(
            id=session_id,
            network=network,
            action=action,
            workspace_path=workspace,
            metric=metric,
            handler_id=settings.get("experiment_id", session_id),
            random_seed=settings.get("random_seed", settings.get("seed")),
        )

        # 3. Persist the training spec so the brain can read it. Resume checks
        # compatibility before any write; otherwise a changed caller spec could
        # overwrite the only evidence needed to reject an incompatible state.
        if resume:
            _require_resume_match(
                label="training specification",
                persisted=self._state_store.get_job_specs(self._context.id),
                requested=train_specs,
            )
        else:
            self._state_store.save_job_specs(self._context.id, train_specs)

        # 4. Generate search space
        if automl_hyperparameters is None:
            # Caller did not specify; we will pass an empty list so that
            # generate_hyperparams_to_search enables only schema-default params.
            automl_hyperparameters = []

        self._ptm_runtime_manifest = None
        self._ptm_runtime_manifest_sha256 = None

        # 5. Algorithm params
        algo_params = AlgorithmParams.from_dict(settings)

        if resolved_ptm_inventory is None:
            # Existing direct BrainFactory path. Keep this path unchanged for
            # callers that did not request repository-owned PTM search.
            param_records, param_names = generate_hyperparams_to_search(
                network=network,
                action=action,
                train_specs=train_specs,
                automl_hyperparameters=automl_hyperparameters,
                schema=search_schema,
            )

            if custom_param_ranges:
                custom_param_ranges = _sanitize_custom_param_ranges(
                    custom_param_ranges,
                    param_records,
                )
                if resume:
                    _require_resume_match(
                        label="custom parameter ranges",
                        persisted=self._state_store.get_custom_param_ranges(
                            self._context.handler_id
                        ),
                        requested=custom_param_ranges,
                    )
                else:
                    self._state_store.save_custom_param_ranges(
                        self._context.handler_id, custom_param_ranges
                    )

            if not param_records or param_records == [{}]:
                requested = sorted(set(automl_hyperparameters))
                requested_detail = (
                    f" Requested parameters: {requested}." if requested else ""
                )
                message = (
                    f"No searchable parameters found for network {network!r}. "
                    "Check that the schema declares at least one supported "
                    "parameter with automl_enabled=true and that the parameter "
                    f"exists in train_specs.{requested_detail}"
                )
                if search_schema is not None:
                    raise ValueError(message)
                logger.warning(message)

            brain = BrainFactory.create_brain(
                algorithm=algorithm,
                context=self._context,
                state_store=self._state_store,
                network=network,
                parameters=param_records,
                params=algo_params,
                metric=brain_metric,
                resume=resume,
                objective_config=objective_config,
                acquisition_settings=settings.get("objective_acquisition"),
            )
        else:
            from tao_automl.brain.base import _stable_context_seed
            from tao_automl.ptm_runtime import (
                ResolvedPTMRuntimeInventory,
                build_hierarchical_ptm_runtime,
                canonical_ptm_algorithm,
            )
            from tao_automl.recommendation_audit import canonical_audit_sha256

            if not isinstance(
                resolved_ptm_inventory,
                ResolvedPTMRuntimeInventory,
            ):
                raise TypeError(
                    "resolved_ptm_inventory must be a live typed "
                    "ResolvedPTMRuntimeInventory"
                )
            resolved_ptm_inventory.validate()
            normalized_algorithm = canonical_ptm_algorithm(algorithm)
            if resolved_ptm_inventory.algorithm != normalized_algorithm:
                raise ValueError(
                    "Resolved PTM inventory algorithm does not match settings"
                )
            if resolved_ptm_inventory.model != network:
                raise ValueError(
                    f"Resolved PTM inventory model "
                    f"{resolved_ptm_inventory.model!r} does not match AutoML "
                    f"network {network!r}"
                )
            selection = objective_config.selection_config
            if (
                selection is None
                or selection.mode != resolved_ptm_inventory.mode
                or canonical_audit_sha256(objective_config.to_dict())
                != resolved_ptm_inventory.objective_config_sha256
            ):
                raise ValueError(
                    "Resolved PTM inventory objective configuration does not "
                    "match AutoML settings"
                )
            if not resolved_ptm_inventory.arms:
                raise ValueError(
                    "Resolved PTM inventory contains no conditional arms"
                )

            conditional_parameters = {}
            conditional_ranges = {}
            union_parameter_names = set()
            requested_ranges = custom_param_ranges or {}
            if not isinstance(requested_ranges, Mapping):
                raise TypeError("custom_param_ranges must be a mapping")
            applied_range_names = set()
            for arm in resolved_ptm_inventory.arms:
                arm_records, arm_names = generate_hyperparams_to_search(
                    network=network,
                    action=action,
                    train_specs=copy.deepcopy(arm.effective_base_spec),
                    automl_hyperparameters=automl_hyperparameters,
                    schema=copy.deepcopy(search_schema),
                )
                if not arm_records or arm_records == [{}] or not arm_names:
                    raise ValueError(
                        f"Resolved PTM arm {arm.checkpoint_id!r} has no "
                        "searchable conditional parameters"
                    )
                arm_ranges = _sanitize_custom_param_ranges(
                    requested_ranges,
                    arm_records,
                )
                arm_name_set = set(arm_names)
                # A global requested range may apply only to a subset of
                # conditional PTM spaces. Each arm receives the same request,
                # sanitized and restricted to parameters it actually owns.
                arm_ranges = {
                    name: copy.deepcopy(value)
                    for name, value in arm_ranges.items()
                    if name in arm_name_set
                }
                applied_range_names.update(arm_ranges)
                conditional_parameters[arm.checkpoint_id] = arm_records
                conditional_ranges[arm.checkpoint_id] = arm_ranges
                union_parameter_names.update(arm_names)
            unapplied_ranges = sorted(
                set(requested_ranges) - applied_range_names
            )
            if unapplied_ranges:
                raise ValueError(
                    "custom_param_ranges contains parameter(s) absent from "
                    "every resolved PTM conditional arm: "
                    + ", ".join(unapplied_ranges)
                )
            param_names = sorted(union_parameter_names)
            if not param_names:
                raise ValueError(
                    "Resolved PTM inventory produced an empty conditional "
                    "parameter-name union"
                )

            runtime = build_hierarchical_ptm_runtime(
                resolved_inventory=resolved_ptm_inventory,
                objective_config=objective_config,
                conditional_parameters=conditional_parameters,
                conditional_ranges=conditional_ranges,
                context=self._context,
                state_store=self._state_store,
                random_seed=_stable_context_seed(self._context),
                acquisition_settings=settings.get("objective_acquisition"),
                algorithm=normalized_algorithm,
                resume=resume,
            )
            brain = runtime.brain
            runtime_manifest = copy.deepcopy(dict(runtime.manifest))
            if canonical_audit_sha256(runtime_manifest) != (
                runtime.manifest_sha256
            ):
                raise ValueError(
                    "Built PTM runtime manifest integrity verification failed"
                )
            self._ptm_runtime_manifest = runtime_manifest
            self._ptm_runtime_manifest_sha256 = runtime.manifest_sha256

        # 6. Controller
        if resume:
            self._controller = Controller.load_state(
                brain=brain,
                context=self._context,
                state_store=self._state_store,
                settings=algo_params,
                metric=metric,
                algorithm=algorithm,
                parameter_names=param_names,
                wandb_config=wandb_config,
                objective_config=objective_config,
            )
        else:
            self._controller = Controller(
                brain=brain,
                context=self._context,
                state_store=self._state_store,
                settings=algo_params,
                metric=metric,
                algorithm=algorithm,
                parameter_names=param_names,
                wandb_config=wandb_config,
                objective_config=objective_config,
            )

        logger.info(
            "AutoML initialized: algorithm=%s, metric=%s, objectives=%s, params=%d, resume=%s",
            algorithm, metric, objective_config.metric_names, len(param_names), resume,
        )

    # ------------------------------------------------------------------
    # Public API (delegates to controller)
    # ------------------------------------------------------------------

    def next_recommendation(self):
        """Get next recommendation(s).

        Returns:
            list of :class:`~tao_automl.types.Recommendation` objects.
            May be empty if the brain is waiting for results.
        """
        return self._controller.next_recommendation()

    def report_result(self, rec_id, metric_value, best_epoch=None, status="success"):
        """Report a training result back.

        Args:
            rec_id: Recommendation ID (from ``rec.id``).
            metric_value: The metric value achieved, or a dict of objective
                metric values for multi-objective sessions.
            best_epoch: Best epoch number (optional).
            status: ``"success"`` or ``"failure"``.
        """
        self._controller.report_result(rec_id, metric_value, best_epoch, status)

    def get_best(self):
        """Get the best Recommendation so far, or None."""
        return self._controller.get_best()

    def get_progress(self):
        """Get a progress summary dict.

        Returns:
            dict with keys ``completed``, ``total``, ``best_metric``,
            ``best_rec_id``, ``algorithm``.
        """
        return self._controller.get_progress()

    def get_history(self):
        """Get all Recommendation objects generated so far."""
        return self._controller.get_history()

    def get_pareto_front(self):
        """Get non-dominated successful recommendations for all objectives."""
        return self._controller.get_pareto_front()

    def get_required_checkpoint_job_ids(self):
        """Get job IDs whose checkpoints are still needed by the search."""
        return self._controller.get_required_checkpoint_job_ids()

    def get_verified_full_fidelity_best(self):
        """Get a verified largest-budget winner when one can be proven."""
        return self._controller.get_verified_full_fidelity_best()

    def get_status(self):
        """Get a full status snapshot of the experiment.

        Returns:
            dict with keys ``progress``, ``best``, ``recommendations``,
            ``active_rec_ids``.
        """
        return self._controller.get_status()

    def is_complete(self):
        """Check if the optimization is done."""
        return self._controller.is_complete()

    def finish(self):
        """Finalize the AutoML session (close WandB, etc.)."""
        self._controller.finish_wandb()

    @property
    def wandb_group(self) -> str:
        """Return the WandB group name for child training runs to join."""
        return self._controller.wandb_group

    @property
    def ptm_runtime_manifest(self):
        """Return a defensive copy of the immutable PTM runtime manifest."""
        return copy.deepcopy(self._ptm_runtime_manifest)

    @property
    def ptm_runtime_manifest_sha256(self):
        """Return the immutable PTM runtime manifest SHA-256, when enabled."""
        return self._ptm_runtime_manifest_sha256
