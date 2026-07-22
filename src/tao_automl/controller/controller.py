# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AutoML optimization loop controller.

The controller manages the brain algorithm, generates recommendations,
and tracks results.  It does NOT launch jobs -- the caller does that.

Optionally integrates with Weights & Biases (wandb) for experiment tracking.
"""

import logging
import os

from tao_automl.objectives import parse_objective_config
from tao_automl.types import Recommendation, ResumeRecommendation, JobStates
from tao_automl.utils.value_utils import normalize_json_value

logger = logging.getLogger(__name__)

# Algorithms whose completion is determined by brain.done()
_BRAIN_DONE_ALGORITHMS = frozenset({
    "hyperband", "h", "bohb", "asha", "dehb", "hyperband_es", "hes", "pbt",
    "hybrid", "autoresearch",
})

# Multi-fidelity algorithms compare low-budget trials to decide promotion, but
# downstream model handoff should prefer the best trial at the largest observed
# budget once such trials exist.
_MULTI_FIDELITY_ALGORITHMS = frozenset({
    "hyperband", "h", "bohb", "asha", "dehb", "hyperband_es", "hes",
})

# Algorithms whose completion is determined by max recommendations count
_MAX_REC_ALGORITHMS = frozenset({"bayesian", "b", "bfbo", "llm"})

_BUDGET_KEY_NAMES = frozenset({
    "num_epochs",
    "epochs",
    "n_epochs",
    "max_iters",
    "epoch",
    "max_epochs",
})


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
        objective_config=None,
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
        self.objective_config = objective_config or parse_objective_config({"metric": metric})
        self.history = []  # list of Recommendation objects
        self._next_id = 0
        self._checkpoint_window = 0

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
            self.brain.save_state()
            self.save_state()
            return []

        recommendations = []
        for raw_rec in raw_recs:
            if not raw_rec:
                continue

            if isinstance(raw_rec, ResumeRecommendation):
                normalized_specs = normalize_json_value(
                    raw_rec.specs,
                    path=f"recommendation[{raw_rec.id}].specs",
                )
                rec = self._find_rec(raw_rec.id)
                if rec is None:
                    rec = Recommendation(
                        identifier=int(raw_rec.id),
                        specs=normalized_specs,
                        metric=self.metric,
                    )
                    self.history.append(rec)
                    self._next_id = max(self._next_id, rec.id + 1)
                else:
                    rec.specs = normalized_specs
                rec.status = JobStates.pending
                rec.resume_from_job_id = raw_rec.resume_from_job_id or raw_rec.job_id
                rec.resume_from_epoch = getattr(raw_rec, "resume_from_epoch", None)
                rec.resume_from_step = getattr(raw_rec, "resume_from_step", None)
                recommendations.append(rec)
                continue

            spec_dict = normalize_json_value(
                raw_rec,
                path=f"recommendation[{self._next_id}].specs",
            )
            rec = Recommendation(
                identifier=self._next_id,
                specs=spec_dict,
                metric=self.metric,
            )
            self.history.append(rec)
            recommendations.append(rec)
            self._next_id += 1

        # Keep the entire most recently issued decision window until the
        # brain is called again.  Looking only at the globally largest budget
        # is unsafe when Hyperband starts a later bracket at a smaller budget:
        # that later bracket still needs its just-finished checkpoints for its
        # next successive-halving decision.
        if recommendations:
            self._checkpoint_window += 1
            for rec in recommendations:
                rec.checkpoint_window = self._checkpoint_window

        # Persist after generating new recommendations
        self.save_state()
        return recommendations

    def report_result(self, rec_id, metric_value, best_epoch=None, status="success"):
        """Feed back a training result.

        Thread/process-safe: acquires the state store's global lock so
        concurrent ``report_result`` calls serialize their state writes.

        Args:
            rec_id: Recommendation ID (int).
            metric_value: The metric value achieved. For multi-objective
                sessions, pass a dict keyed by objective metric name.
            best_epoch: Best epoch number (optional).
            status: ``"success"`` or ``"failure"``.
        """
        with self.state_store.lock():
            rec = self._find_rec(rec_id)
            if rec is None:
                logger.warning("report_result: recommendation %s not found", rec_id)
                return

            rec_status = status if status else JobStates.success
            objective_values = self.objective_config.coerce_values(metric_value)
            try:
                objective_score = self.objective_config.scalarize(objective_values)
            except ValueError:
                if rec_status in (JobStates.success, JobStates.done):
                    raise
                objective_score = 0.0

            if self.objective_config.is_multi_objective and objective_values:
                if all(
                    name in objective_values
                    for name in self.objective_config.metric_names
                ):
                    rec.update_objectives(objective_values, objective_score)
                else:
                    rec.update_result(objective_score)
                    rec.objective_values = objective_values
                    rec.objective_score = objective_score
            else:
                rec.update_result(objective_score)
                rec.objective_values = objective_values
                rec.objective_score = objective_score
            rec.update_status(rec_status)
            if best_epoch is not None:
                rec.best_epoch_number = best_epoch

            # Persist brain and controller state (under lock)
            result_observer = getattr(self.brain, "on_recommendation_result", None)
            if result_observer is not None:
                result_observer(rec, self.history)
            self.brain.save_state()
            self.save_state()

        logger.info(
            "Reported result for rec %d: score=%.6f values=%s status=%s",
            rec_id, rec.result, rec.objective_values, status,
        )

        self._update_wandb_table()

    def get_best(self):
        """Return the best Recommendation so far, or None.

        Uses explicit objective configuration when present; otherwise preserves
        the legacy metric-name direction rule.
        """
        completed = [
            r for r in self.history
            if r.status in (JobStates.success, JobStates.done)
        ]

        # PBT reuses stable population member IDs across generations.  The
        # corresponding Recommendation objects are updated in place, so retain
        # the persisted best snapshot as an additional candidate instead of
        # forgetting a superior checkpoint from an earlier generation.
        if self.algorithm == "pbt":
            saved_best = self.state_store.get_best_rec_info(self.context.id)
            if saved_best and saved_best.get("rec_data"):
                rec_dict = saved_best["rec_data"]
                archived = Recommendation(
                    identifier=int(rec_dict["id"]),
                    specs=rec_dict.get("specs", {}),
                    metric=rec_dict.get("metric", self.metric),
                )
                archived.job_id = rec_dict.get("job_id")
                archived.status = rec_dict.get("status", JobStates.success)
                archived.result = float(rec_dict.get("result", 0.0))
                archived.objective_values = {
                    str(k): float(v)
                    for k, v in rec_dict.get("objective_values", {}).items()
                }
                if not archived.objective_values:
                    archived.objective_values = {self.metric: archived.result}
                archived.objective_score = float(
                    rec_dict.get("objective_score", archived.result)
                )
                archived.best_epoch_number = rec_dict.get("best_epoch_number", "")
                archived.resume_from_job_id = rec_dict.get("resume_from_job_id")
                archived.resume_from_epoch = rec_dict.get("resume_from_epoch")
                archived.resume_from_step = rec_dict.get("resume_from_step")
                archived.early_stop_epoch = rec_dict.get("early_stop_epoch")
                archived.created_on = rec_dict.get("created_on", "")
                archived.last_modified = rec_dict.get("last_modified", "")
                if archived.status in (JobStates.success, JobStates.done):
                    # Keep the archived snapshot eligible when it is strictly
                    # better, but prefer a terminal-generation checkpoint on
                    # an equal score.  PBT reuses member IDs, and selecting the
                    # older tie would hand downstream actions a checkpoint
                    # that predates the completed exploit/resume generation.
                    completed.append(archived)

        if not completed:
            return None

        if self.algorithm in _MULTI_FIDELITY_ALGORITHMS:
            completed = self._largest_budget_candidates(completed)

        if self.objective_config.score_direction == "minimize":
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
        pareto_front = self.get_pareto_front()

        total = self._estimate_total()

        return {
            "completed": len(completed_recs),
            "total": total,
            "best_metric": best.primary_metric_value() if best else None,
            "best_objective_score": best.objective_score if best else None,
            "best_rec_id": best.id if best else None,
            "algorithm": self.algorithm,
            "objectives": self.objective_config.to_dict(),
            "pareto_front_size": len(pareto_front),
        }

    def get_history(self):
        """Return the full list of Recommendation objects."""
        return list(self.history)

    def get_required_checkpoint_job_ids(self):
        """Return checkpoint jobs still required by an unfinished search.

        Multi-fidelity brains make promotion decisions in batches. This set is
        conservative for the current decision window, but releases eliminated
        trials as soon as the brain advances to the next rung or generation.
        """
        required = set()
        rec_by_id = {rec.id: rec for rec in self.history}
        active_states = {JobStates.pending, JobStates.started, JobStates.running}

        for rec in self.history:
            if rec.status in active_states and rec.job_id:
                required.add(rec.job_id)
            if rec.status in active_states and rec.resume_from_job_id:
                required.add(rec.resume_from_job_id)

        def add_rec_id(rec_id):
            try:
                rec = rec_by_id.get(int(rec_id))
            except (TypeError, ValueError):
                rec = None
            if rec is not None and rec.job_id:
                required.add(rec.job_id)

        def collect_brain(brain):
            if brain is None:
                return
            for rec_id in getattr(brain, "active_configs", set()) or set():
                add_rec_id(rec_id)
            for promotion in getattr(brain, "pending_promotions", []) or []:
                if isinstance(promotion, (list, tuple)) and promotion:
                    add_rec_id(promotion[0])
            population = getattr(brain, "population", None)
            if isinstance(population, dict):
                for rec_id in population:
                    add_rec_id(rec_id)
            considered = getattr(brain, "experiments_considered", []) or []
            for rec in considered:
                job_id = getattr(rec, "job_id", None)
                if job_id:
                    required.add(job_id)
            collect_brain(getattr(brain, "current_sub_brain", None))

        collect_brain(self.brain)

        checkpoint_windows = [
            getattr(rec, "checkpoint_window", 0) for rec in self.history
            if getattr(rec, "checkpoint_window", 0)
        ]
        if checkpoint_windows:
            latest_window = max(checkpoint_windows)
            required.update(
                rec.job_id for rec in self.history
                if getattr(rec, "checkpoint_window", 0) == latest_window
                and rec.job_id
            )

        # Fail-closed fallback for workspaces created before decision windows
        # were persisted.  Their current bracket cannot be reconstructed
        # reliably: the newest bracket may have a smaller budget than an older
        # one, so choosing the globally largest budget could delete a required
        # promotion parent.  New runs use the exact, bounded latest window.
        if (
            not checkpoint_windows
            and getattr(self.brain, "last_launched_count", 0)
        ):
            successful = [
                rec for rec in self.history
                if rec.status in (JobStates.success, JobStates.done) and rec.job_id
            ]
            required.update(rec.job_id for rec in successful)

        return required

    def get_verified_full_fidelity_best(self):
        """Return the best largest-budget result when it is provable."""
        completed = [
            rec for rec in self.history
            if rec.status in (JobStates.success, JobStates.done) and rec.job_id
        ]
        if not completed:
            return None
        known = [
            (self._recommendation_budget(rec), rec) for rec in completed
            if self._recommendation_budget(rec) is not None
        ]
        candidates = completed
        if known:
            largest = max(budget for budget, _rec in known)
            candidates = [rec for budget, rec in known if budget == largest]
        elif self.algorithm != "hybrid":
            return self.get_best()

        selector = (
            min if self.objective_config.score_direction == "minimize" else max
        )
        return selector(candidates, key=lambda rec: rec.result)

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
                "metric_value": r.primary_metric_value(),
                "objective_score": r.objective_score,
                "objective_values": dict(r.objective_values),
                "failure_reason": getattr(r, "failure_reason", None),
                "adjustments": getattr(r, "adjustments", []),
                "created_on": r.created_on,
                "last_modified": r.last_modified,
            })

        active = [r.id for r in self.history if r.status in (JobStates.pending, JobStates.started, JobStates.running)]

        return {
            "progress": progress,
            "best": {
                "rec_id": best.id if best else None,
                "specs": best.specs if best else {},
                "metric_value": best.primary_metric_value() if best else None,
                "objective_score": best.objective_score if best else None,
                "objective_values": dict(best.objective_values) if best else {},
            },
            "recommendations": recs,
            "pareto_front": self._serialize_pareto_front(),
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
            if self.objective_config.is_multi_objective:
                columns = [
                    "experiment_id", "job_id", "status", "objective_score",
                    *self.objective_config.metric_names, "best_epoch_number",
                ]
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
            if self.objective_config.is_multi_objective:
                columns = [
                    "experiment_id", "job_id", "status", "objective_score",
                    *self.objective_config.metric_names, "best_epoch_number",
                ]
            columns.extend(self.parameter_names)
            self._wandb_table = wandb.Table(columns=columns)

            for rec in self.history:
                result_value = rec.objective_score
                if isinstance(result_value, float):
                    formatted = f"{result_value:.10f}".rstrip('0')
                    if formatted.endswith('.'):
                        formatted += '0'
                    result_value = formatted

                row_data = [
                    rec.id,
                    rec.job_id or "",
                    rec.status,
                ]
                if self.objective_config.is_multi_objective:
                    row_data.append(result_value)
                    for name in self.objective_config.metric_names:
                        row_data.append(rec.objective_values.get(name, "N/A"))
                    row_data.append(rec.best_epoch_number)
                else:
                    row_data.extend([rec.primary_metric_value(), rec.best_epoch_number])
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
        objective_config=None,
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
            objective_config=objective_config,
        )

        saved = state_store.get_controller_info(context.id)
        if saved:
            for rec_dict in saved:
                rec = Recommendation(
                    identifier=int(rec_dict["id"]),
                    specs=normalize_json_value(
                        rec_dict.get("specs", {}),
                        path=f"recommendation[{rec_dict['id']}].specs",
                    ),
                    metric=metric,
                )
                rec.job_id = rec_dict.get("job_id")
                rec.status = rec_dict.get("status", JobStates.pending)
                rec.update_result(rec_dict.get("result", 0.0))
                objective_values = rec_dict.get("objective_values", {})
                if objective_values:
                    rec.update_objectives(
                        objective_values,
                        rec_dict.get("objective_score", rec.result),
                    )
                rec.best_epoch_number = rec_dict.get("best_epoch_number", "")
                rec.resume_from_job_id = rec_dict.get("resume_from_job_id")
                rec.resume_from_epoch = rec_dict.get("resume_from_epoch")
                rec.resume_from_step = rec_dict.get("resume_from_step")
                rec.checkpoint_window = int(rec_dict.get("checkpoint_window", 0) or 0)
                rec.early_stop_epoch = rec_dict.get("early_stop_epoch")
                rec.created_on = rec_dict.get("created_on", "")
                rec.last_modified = rec_dict.get("last_modified", "")
                controller.history.append(rec)

            if controller.history:
                controller._next_id = max(r.id for r in controller.history) + 1
                controller._checkpoint_window = max(
                    getattr(r, "checkpoint_window", 0)
                    for r in controller.history
                )

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

    @staticmethod
    def _recommendation_budget(rec):
        """Return the largest explicit training budget in a recommendation."""
        budgets = []
        if rec.early_stop_epoch is not None:
            budgets.append(rec.early_stop_epoch)
        for key, value in rec.specs.items():
            name = str(key).split(".")[-1]
            if name not in _BUDGET_KEY_NAMES:
                continue
            if isinstance(value, bool):
                continue
            try:
                budgets.append(float(value))
            except (TypeError, ValueError):
                continue
        return max(budgets) if budgets else None

    def _largest_budget_candidates(self, completed):
        """Filter completed recommendations to the largest observed budget."""
        budgeted = []
        for rec in completed:
            budget = self._recommendation_budget(rec)
            if budget is not None:
                budgeted.append((budget, rec))
        if not budgeted:
            return completed
        max_budget = max(budget for budget, _rec in budgeted)
        largest_budget_recs = [
            rec for budget, rec in budgeted
            if budget == max_budget
        ]
        return largest_budget_recs or completed

    def get_pareto_front(self):
        """Return non-dominated successful recommendations for configured objectives."""
        completed = [
            r for r in self.history
            if r.status in (JobStates.success, JobStates.done)
        ]
        if not self.objective_config.is_multi_objective:
            return completed
        return self.objective_config.pareto_front(completed)

    def _serialize_pareto_front(self):
        """Return JSON-safe Pareto-front records."""
        return [
            {
                "rec_id": rec.id,
                "specs": rec.specs,
                "metric_value": rec.primary_metric_value(),
                "objective_score": rec.objective_score,
                "objective_values": dict(rec.objective_values),
            }
            for rec in self.get_pareto_front()
        ]

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
            "objective_score": rec.objective_score,
            "objective_values": dict(rec.objective_values),
            "best_epoch_number": rec.best_epoch_number,
            "metric": rec.metric,
            "resume_from_job_id": rec.resume_from_job_id,
            "resume_from_epoch": rec.resume_from_epoch,
            "resume_from_step": rec.resume_from_step,
            "checkpoint_window": rec.checkpoint_window,
            "early_stop_epoch": rec.early_stop_epoch,
            "failure_reason": getattr(rec, "failure_reason", None),
            "adjustments": getattr(rec, "adjustments", []),
            "created_on": rec.created_on,
            "last_modified": rec.last_modified,
        }
