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

import json
import logging
import os
import uuid

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
        best = {
            "rec_id": bd.get("id"),
            "specs": bd.get("specs", {}),
            "metric_value": bd.get("result"),
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
                "metric_value": r.get("result"),
                "created_on": r.get("created_on"),
                "last_modified": r.get("last_modified"),
            }
            for r in recs
        ],
        "active_jobs": active_jobs,
    }


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
        resume=False,
        wandb_config=None,
    ):
        """
        Args:
            workspace: Path to workspace directory for state persistence.
            network: Network architecture name (e.g. ``"dino"``).
            train_specs: Training spec dict (the base configuration).
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
        metric = settings.get("metric", "loss")

        # 1. State store
        self._state_store = StateStore(workspace)

        # 2. Context
        session_id = settings.get("session_id", uuid.uuid4().hex[:12])
        self._context = AutoMLContext(
            id=session_id,
            network=network,
            action="train",
            workspace_path=workspace,
            metric=metric,
            handler_id=settings.get("experiment_id", session_id),
        )

        # 3. Persist the training spec so the brain can read it
        self._state_store.save_job_specs(self._context.id, train_specs)

        # 4. Custom parameter ranges
        if custom_param_ranges:
            self._state_store.save_custom_param_ranges(
                self._context.handler_id, custom_param_ranges
            )

        # 5. Generate search space
        if automl_hyperparameters is None:
            # Caller did not specify; we will pass an empty list so that
            # generate_hyperparams_to_search enables only schema-default params.
            automl_hyperparameters = []

        param_records, param_names = generate_hyperparams_to_search(
            network=network,
            action="train",
            train_specs=train_specs,
            automl_hyperparameters=automl_hyperparameters,
        )

        if not param_records or param_records == [{}]:
            logger.warning(
                "No searchable parameters found for network '%s'. "
                "Check that automl_hyperparameters match the schema.", network
            )

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
            metric=metric,
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
            )

        logger.info(
            "AutoML initialized: algorithm=%s, metric=%s, params=%d, resume=%s",
            algorithm, metric, len(param_names), resume,
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
            metric_value: The metric value achieved.
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
