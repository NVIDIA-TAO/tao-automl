# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AutoML optimization loop controller.

The controller manages the brain algorithm, generates recommendations,
and tracks results.  It does NOT launch jobs -- the caller does that.

Optionally integrates with Weights & Biases (wandb) for experiment tracking.
"""

import logging
import os

from tao_automl.types import Recommendation, JobStates

logger = logging.getLogger(__name__)

# Algorithms whose completion is determined by brain.done()
_BRAIN_DONE_ALGORITHMS = frozenset({
    "hyperband", "h", "bohb", "asha", "dehb", "hyperband_es", "hes", "pbt",
    "hybrid", "autoresearch",
})

# Algorithms whose completion is determined by max recommendations count
_MAX_REC_ALGORITHMS = frozenset({"bayesian", "b", "bfbo", "llm"})


class Controller:
    """AutoML optimization loop controller.

    The controller manages the brain algorithm, generates recommendations,
    and tracks results.  It does NOT launch jobs -- the caller does that.
    """

    def __init__(
        self,
        brain,
        context,
        state_store,
        settings,
        metric,
        algorithm,
        parameter_names=None,
        wandb_config=None,
    ):
        """
        Args:
            brain: Algorithm instance (Bayesian, Hyperband, etc.)
            context: AutoMLContext
            state_store: StateStore for persistence
            settings: AlgorithmParams
            metric: Optimization metric name
            algorithm: Algorithm name string
            parameter_names: List of parameter names being searched
            wandb_config: Optional dict with WandB settings.
                Keys: ``project``, ``entity``, ``api_key``, ``group``,
                ``enabled`` (bool, default False).
        """
        self.brain = brain
        self.context = context
        self.state_store = state_store
        self.settings = settings
        self.metric = metric
        self.algorithm = algorithm.lower()
        self.parameter_names = parameter_names or []
        self.history = []  # list of Recommendation objects
        self._next_id = 0

        # WandB integration (optional)
        self._wandb_config = wandb_config or {}
        self._wandb_initialized = False
        self._wandb_table = None
        self._wandb_group = f"automl_{self.context.id}"
        if self._wandb_config.get("enabled"):
            self._initialize_wandb()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next_recommendation(self):
        """Get next hyperparameter recommendation(s) from the brain.

        Returns:
            list of Recommendation objects.  May be empty if the brain is
            waiting for results (e.g. Bayesian waits for the previous run
            to finish), or contain multiple entries for parallel algorithms
            like Hyperband.
        """
        raw_recs = self.brain.generate_recommendations(self.history)

        if not raw_recs:
            return []

        recommendations = []
        for spec_dict in raw_recs:
            if not spec_dict:
                continue
            rec = Recommendation(
                identifier=self._next_id,
                specs=spec_dict,
                metric=self.metric,
            )
            self.history.append(rec)
            recommendations.append(rec)
            self._next_id += 1

        # Persist after generating new recommendations
        self.save_state()
        return recommendations

    def report_result(self, rec_id, metric_value, best_epoch=None, status="success"):
        """Feed back a training result.

        Thread/process-safe: acquires the state store's global lock so
        concurrent ``report_result`` calls serialize their state writes.

        Args:
            rec_id: Recommendation ID (int).
            metric_value: The metric value achieved (float).
            best_epoch: Best epoch number (optional).
            status: ``"success"`` or ``"failure"``.
        """
        with self.state_store.lock():
            rec = self._find_rec(rec_id)
            if rec is None:
                logger.warning("report_result: recommendation %s not found", rec_id)
                return

            rec.update_result(metric_value)
            rec.update_status(status if status else JobStates.success)
            if best_epoch is not None:
                rec.best_epoch_number = best_epoch

            # Persist brain and controller state (under lock)
            self.brain.save_state()
            self.save_state()

        logger.info(
            "Reported result for rec %d: metric=%.6f status=%s",
            rec_id, metric_value, status,
        )

        self._update_wandb_table()

    def get_best(self):
        """Return the best Recommendation so far, or None.

        Uses the convention that if the metric name contains ``"loss"`` then
        lower is better; otherwise higher is better.
        """
        completed = [
            r for r in self.history
            if r.status in (JobStates.success, JobStates.done)
        ]
        if not completed:
            return None

        lower_is_better = "loss" in self.metric.lower()
        if lower_is_better:
            return min(completed, key=lambda r: r.result)
        return max(completed, key=lambda r: r.result)

    def get_progress(self):
        """Return a progress summary dict.

        Keys: ``completed``, ``total``, ``best_metric``, ``best_rec_id``,
        ``algorithm``.
        """
        completed_recs = [
            r for r in self.history
            if r.status in (JobStates.success, JobStates.done, JobStates.failure, JobStates.error)
        ]
        best = self.get_best()

        total = self._estimate_total()

        return {
            "completed": len(completed_recs),
            "total": total,
            "best_metric": best.result if best else None,
            "best_rec_id": best.id if best else None,
            "algorithm": self.algorithm,
        }

    def get_history(self):
        """Return the full list of Recommendation objects."""
        return list(self.history)

    def get_status(self):
        """Return a structured status snapshot of the entire experiment.

        Returns:
            dict with keys ``progress``, ``best``, ``recommendations``,
            ``active_rec_id``.
        """
        progress = self.get_progress()
        best = self.get_best()

        recs = []
        for r in self.history:
            recs.append({
                "rec_id": r.id,
                "specs": r.specs,
                "job_id": r.job_id,
                "status": r.status,
                "metric_value": r.result,
                "created_on": r.created_on,
                "last_modified": r.last_modified,
            })

        active = [r.id for r in self.history if r.status in (JobStates.pending, JobStates.started, JobStates.running)]

        return {
            "progress": progress,
            "best": {
                "rec_id": best.id if best else None,
                "specs": best.specs if best else {},
                "metric_value": best.result if best else None,
            },
            "recommendations": recs,
            "active_rec_ids": active,
        }

    def is_complete(self):
        """Check if the optimization loop is done.

        * Bayesian / BFBO: completed count >= ``settings.automl_max_recommendations``.
        * Hyperband / BOHB / ASHA / DEHB / HyperBandES / PBT: delegates to
          ``brain.done()`` and verifies no outstanding pending experiments.
        """
        if self.algorithm in _MAX_REC_ALGORITHMS:
            completed = sum(
                1 for r in self.history
                if r.status in (JobStates.success, JobStates.done, JobStates.failure, JobStates.error)
            )
            return completed >= self.settings.automl_max_recommendations

        if self.algorithm in _BRAIN_DONE_ALGORITHMS:
            if hasattr(self.brain, "done"):
                return self.brain.done()
            # Fallback: treat as complete when all history entries are terminal
            if not self.history:
                return False
            return all(
                r.status in (JobStates.success, JobStates.done, JobStates.failure, JobStates.error)
                for r in self.history
            )

        # Unknown algorithm -- conservative default
        logger.warning("is_complete: unknown algorithm '%s', defaulting to False", self.algorithm)
        return False

    # ------------------------------------------------------------------
    # WandB integration
    # ------------------------------------------------------------------

    def _initialize_wandb(self):
        """Initialize WandB run for AutoML experiment tracking."""
        if self._wandb_initialized:
            return

        try:
            import wandb

            api_key = self._wandb_config.get(
                "api_key", os.getenv("WANDB_API_KEY", "")
            )
            if not api_key:
                logger.info("No WANDB_API_KEY found, skipping WandB initialization")
                return

            if not wandb.login(key=api_key):
                logger.warning("Failed to login to WandB, skipping")
                return

            group = self._wandb_config.get("group", self._wandb_group)
            wandb.init(
                project=self._wandb_config.get("project", "TAO AutoML"),
                entity=self._wandb_config.get("entity"),
                name="automl_brain",
                group=group,
                config={
                    "network": self.context.network,
                    "algorithm": self.algorithm,
                    "metric": self.metric,
                },
                dir=os.path.join(self.context.workspace_path, "wandb")
                if self.context.workspace_path else None,
                reinit=True,
            )

            self._wandb_initialized = True
            self._wandb_group = group
            logger.info("WandB initialized with group: %s", group)

            columns = ["experiment_id", "job_id", "status", self.metric, "best_epoch_number"]
            columns.extend(self.parameter_names)
            self._wandb_table = wandb.Table(columns=columns)

        except ImportError:
            logger.info("wandb package not installed, skipping WandB integration")
        except Exception as e:
            logger.warning("Failed to initialize WandB: %s", e)
            self._wandb_initialized = False

    def _update_wandb_table(self):
        """Rebuild and log the WandB table with current recommendation state."""
        if not self._wandb_initialized or self._wandb_table is None:
            return

        try:
            import wandb

            columns = ["experiment_id", "job_id", "status", self.metric, "best_epoch_number"]
            columns.extend(self.parameter_names)
            self._wandb_table = wandb.Table(columns=columns)

            for rec in self.history:
                result_value = rec.result
                if isinstance(result_value, float):
                    formatted = f"{result_value:.10f}".rstrip('0')
                    if formatted.endswith('.'):
                        formatted += '0'
                    result_value = formatted

                row_data = [
                    rec.id,
                    rec.job_id or "",
                    rec.status,
                    result_value,
                    rec.best_epoch_number,
                ]
                for param_name in self.parameter_names:
                    value = rec.specs.get(param_name, "N/A")
                    row_data.append(value)

                self._wandb_table.add_data(*row_data)

            wandb.log({"automl_experiments": self._wandb_table})
            logger.debug("Updated WandB table with %d recommendations", len(self.history))

        except Exception as e:
            logger.warning("Failed to update WandB table: %s", e)

    def finish_wandb(self):
        """Finalize WandB run. Call when the optimization loop ends."""
        if not self._wandb_initialized:
            return
        self._update_wandb_table()
        try:
            import wandb
            wandb.finish()
            logger.info("Closed WandB run")
        except Exception as e:
            logger.warning("Failed to close WandB run: %s", e)

    @property
    def wandb_group(self) -> str:
        """Return the WandB group name for child runs to join."""
        return self._wandb_group

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self):
        """Persist current controller state (history) to workspace."""
        serialized = [self._serialize_rec(r) for r in self.history]
        self.state_store.save_controller_info(self.context.id, serialized)

        # Also persist best recommendation
        best = self.get_best()
        if best is not None:
            self.state_store.save_best_rec_info(
                self.context.id,
                rec_number=best.id,
                rec_data=self._serialize_rec(best),
            )

    @classmethod
    def load_state(
        cls,
        brain,
        context,
        state_store,
        settings,
        metric,
        algorithm,
        parameter_names=None,
        wandb_config=None,
    ):
        """Load controller from persisted state.

        Returns a Controller instance with history restored from disk.
        """
        controller = cls(
            brain=brain,
            context=context,
            state_store=state_store,
            settings=settings,
            metric=metric,
            algorithm=algorithm,
            parameter_names=parameter_names,
            wandb_config=wandb_config,
        )

        saved = state_store.get_controller_info(context.id)
        if saved:
            for rec_dict in saved:
                rec = Recommendation(
                    identifier=int(rec_dict["id"]),
                    specs=rec_dict.get("specs", {}),
                    metric=metric,
                )
                rec.job_id = rec_dict.get("job_id")
                rec.status = rec_dict.get("status", JobStates.pending)
                rec.result = float(rec_dict.get("result", 0.0))
                rec.best_epoch_number = rec_dict.get("best_epoch_number", "")
                rec.resume_from_job_id = rec_dict.get("resume_from_job_id")
                rec.early_stop_epoch = rec_dict.get("early_stop_epoch")
                rec.created_on = rec_dict.get("created_on", "")
                rec.last_modified = rec_dict.get("last_modified", "")
                controller.history.append(rec)

            if controller.history:
                controller._next_id = max(r.id for r in controller.history) + 1

        logger.info(
            "Loaded controller state: %d recommendations, next_id=%d",
            len(controller.history), controller._next_id,
        )
        return controller

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_rec(self, rec_id):
        """Find a recommendation by ID in history."""
        for r in self.history:
            if r.id == rec_id:
                return r
        return None

    def _estimate_total(self):
        """Estimate total number of recommendations for progress reporting."""
        if self.algorithm in _MAX_REC_ALGORITHMS:
            return self.settings.automl_max_recommendations

        # For multi-fidelity algorithms the total is harder to know upfront.
        # Return the number of recommendations generated so far as a lower bound.
        return len(self.history)

    @staticmethod
    def _serialize_rec(rec):
        """Convert a Recommendation to a JSON-safe dict."""
        return {
            "id": rec.id,
            "specs": rec.specs,
            "job_id": rec.job_id,
            "status": rec.status,
            "result": rec.result,
            "best_epoch_number": rec.best_epoch_number,
            "metric": rec.metric,
            "resume_from_job_id": rec.resume_from_job_id,
            "early_stop_epoch": rec.early_stop_epoch,
            "created_on": rec.created_on,
            "last_modified": rec.last_modified,
        }
