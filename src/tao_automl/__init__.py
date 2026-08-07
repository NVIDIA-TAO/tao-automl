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
            Since ``AutoMLRunner.run()`` appends a timestamped suffix
            (e.g. ``run_20260423_183015``) to the base path, pass the
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

    return {
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
                "created_on": r.get("created_on"),
                "last_modified": r.get("last_modified"),
            }
            for r in recs
        ],
        "active_jobs": active_jobs,
    }


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

        # 2. Context
        session_id = settings.get("session_id", uuid.uuid4().hex[:12])
        self._context = AutoMLContext(
            id=session_id,
            network=network,
            action=action,
            workspace_path=workspace,
            metric=metric,
            handler_id=settings.get("experiment_id", session_id),
        )

        # 3. Persist the training spec so the brain can read it
        self._state_store.save_job_specs(self._context.id, train_specs)

        # 4. Generate search space
        if automl_hyperparameters is None:
            # Caller did not specify; we will pass an empty list so that
            # generate_hyperparams_to_search enables only schema-default params.
            automl_hyperparameters = []

        param_records, param_names = generate_hyperparams_to_search(
            network=network,
            action=action,
            train_specs=train_specs,
            automl_hyperparameters=automl_hyperparameters,
            schema=search_schema,
        )

        # 5. Custom parameter ranges
        if custom_param_ranges:
            custom_param_ranges = _sanitize_custom_param_ranges(
                custom_param_ranges,
                param_records,
            )
            self._state_store.save_custom_param_ranges(
                self._context.handler_id, custom_param_ranges
            )

        if not param_records or param_records == [{}]:
            requested = sorted(set(automl_hyperparameters))
            requested_detail = (
                f" Requested parameters: {requested}." if requested else ""
            )
            message = (
                f"No searchable parameters found for network {network!r}. Check "
                "that the schema declares at least one supported parameter with "
                "automl_enabled=true and that the parameter exists in train_specs."
                f"{requested_detail}"
            )
            if search_schema is not None:
                raise ValueError(message)
            logger.warning(message)

        # 6. Algorithm params
        algo_params = AlgorithmParams.from_dict(settings)

        # 7. Brain
        brain = BrainFactory.create_brain(
            algorithm=algorithm,
            context=self._context,
            state_store=self._state_store,
            network=network,
            parameters=param_records,
            params=algo_params,
            metric=brain_metric,
            resume=resume,
        )

        # 8. Controller
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

    def get_algorithm_state(self):
        """Return the persisted brain state for algorithm decision evidence."""
        return self._state_store.get_brain_info(self._context.id) or {}

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
