# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid Strategist.

LLM operates at a higher level than individual trials: it decides WHAT
dimension to explore and WHICH algorithm to use, then delegates to existing
TAO algorithms (Bayesian, ASHA, BOHB, etc.) for the actual sweep.

HybridBrain wraps the strategist as a Controller-compatible brain.
"""
import logging
from typing import Any, Dict, List, Optional

from tao_automl.brain.llm_client import LLMClient
from tao_automl.brain.prompts.autoresearch_prompts import (
    build_hybrid_strategy_prompt,
)
from tao_automl.types import JobStates

logger = logging.getLogger(__name__)

AVAILABLE_ALGORITHMS = ["bayesian", "asha", "bohb", "dehb", "pbt", "hyperband", "llm", "autoresearch"]


class HybridStrategist:
    """LLM-powered strategic planner for multi-phase AutoML.

    Instead of using a single algorithm for the entire run, the strategist:
    1. Analyzes which parameters are most promising to explore
    2. Picks the best algorithm for each exploration phase
    3. Sets phase budget (number of trials)
    4. Analyzes phase results and decides next direction
    5. Detects diminishing returns and recommends stopping
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        llm_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the HybridStrategist."""
        self.llm_client = llm_client or LLMClient(params=llm_params)
        self.completed_phases: List[Dict[str, Any]] = []
        self.full_history: List[Dict[str, Any]] = []

    def plan_next_phase(
        self,
        available_parameters: List[Dict[str, Any]],
        network: str,
        metric_name: str,
        metric_direction: str,
    ) -> Optional[Dict[str, Any]]:
        """Ask the LLM strategist to plan the next optimization phase."""
        messages = build_hybrid_strategy_prompt(
            full_history=self.full_history,
            available_parameters=available_parameters,
            available_algorithms=AVAILABLE_ALGORITHMS,
            network=network,
            metric_name=metric_name,
            metric_direction=metric_direction,
            completed_phases=self.completed_phases,
        )

        response = self.llm_client.chat(messages, json_mode=True, temperature=0.5)

        if not response.ok or response.json_content is None:
            logger.warning("Hybrid strategist LLM call failed: %s", response.error)
            return None

        plan = response.json_content
        plan = self._validate_plan(plan, available_parameters)

        logger.info(
            "Hybrid phase plan: action=%s, algorithm=%s, params=%s, trials=%d. Reasoning: %s",
            plan.get("action"),
            plan.get("algorithm"),
            plan.get("parameters"),
            plan.get("trials", 0),
            plan.get("reasoning", "")[:100],
        )

        return plan

    def record_phase_results(
        self,
        phase_plan: Dict[str, Any],
        results: List[Dict[str, Any]],
        best_config: Optional[Dict[str, Any]] = None,
        best_metric: Optional[float] = None,
    ):
        """Record results of a completed phase for history tracking."""
        phase_record = {
            "phase_number": len(self.completed_phases) + 1,
            "plan": phase_plan,
            "num_experiments": len(results),
            "best_config": best_config,
            "best_metric": best_metric,
        }
        self.completed_phases.append(phase_record)
        self.full_history.extend(results)

    def should_stop(self) -> bool:
        """Check if the strategist recommends stopping."""
        if not self.completed_phases:
            return False
        last_phase = self.completed_phases[-1]
        return last_phase.get("plan", {}).get("action") == "stop"

    def _validate_plan(
        self, plan: Dict[str, Any], available_parameters: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate and sanitize the strategist's plan."""
        available_names = {p["parameter"] for p in available_parameters}

        action = plan.get("action", "sweep")
        if action not in ("sweep", "single_trial", "stop"):
            action = "sweep"
        plan["action"] = action

        if action == "stop":
            return plan

        algorithm = plan.get("algorithm", "bayesian")
        if algorithm not in AVAILABLE_ALGORITHMS:
            algorithm = "bayesian"
        plan["algorithm"] = algorithm

        params = plan.get("parameters", [])
        if isinstance(params, str):
            params = [p.strip() for p in params.split(",")]
        valid_params = [p for p in params if p in available_names]
        if not valid_params:
            valid_params = list(available_names)[:5]
        plan["parameters"] = valid_params

        trials = plan.get("trials", 5)
        if not isinstance(trials, int) or trials < 1:
            trials = 5
        plan["trials"] = min(trials, 50)

        return plan

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state."""
        return {
            "completed_phases": self.completed_phases,
            "full_history": self.full_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], llm_params=None) -> "HybridStrategist":
        """Deserialize state."""
        strategist = cls(llm_params=llm_params)
        strategist.completed_phases = data.get("completed_phases", [])
        strategist.full_history = data.get("full_history", [])
        return strategist


class HybridBrain:
    """Controller-compatible brain that delegates to per-phase sub-brains.

    The Controller treats this like any other algorithm brain and calls
    generate_recommendations(history). Internally, the HybridBrain:
    1. Uses HybridStrategist to plan phases
    2. Creates a sub-brain for the current phase
    3. Delegates generate_recommendations to the sub-brain
    4. When the phase budget is reached, records phase results and plans the next phase
    5. Returns [] when the strategist says "stop"
    """

    def __init__(
        self,
        context,
        state_store,
        network: str,
        parameters: List[Dict[str, Any]],
        llm_params: Optional[Dict[str, Any]] = None,
        metric: str = "kpi",
    ):
        """Initialize the HybridBrain."""
        self.context = context
        self.state_store = state_store
        self.network = network
        self.all_parameters = parameters
        self.metric = metric
        self.llm_params = llm_params

        self.strategist = HybridStrategist(llm_params=llm_params)
        self.current_plan: Optional[Dict[str, Any]] = None
        self.current_sub_brain = None
        self.phase_experiment_count = 0
        self.reverse_sort = metric != "loss"
        self.num_epochs_per_experiment = 0
        self._stopped = False

    def generate_recommendations(self, history):
        """Generate recommendations by delegating to the current phase's sub-brain."""
        if self._stopped:
            return []

        if self.current_plan is None or self._phase_budget_exhausted():
            self._advance_phase(history)
            if self._stopped:
                return []

        if self.current_sub_brain is None:
            return []

        phase_start = len(history) - self.phase_experiment_count
        phase_history = history[max(0, phase_start):]

        return self.current_sub_brain.generate_recommendations(phase_history)

    def done(self):
        """Return True when the hybrid brain has stopped."""
        return self._stopped

    def _phase_budget_exhausted(self) -> bool:
        if self.current_plan is None:
            return True
        return self.phase_experiment_count >= self.current_plan.get("trials", 5)

    def _advance_phase(self, history):
        """Record current phase results (if any) and plan the next phase."""
        if self.current_plan and self.current_plan.get("action") != "stop":
            phase_results = []
            for rec in history[-self.phase_experiment_count:] if self.phase_experiment_count > 0 else []:
                if rec.status in [JobStates.success, JobStates.failure]:
                    phase_results.append({
                        "config": rec.specs if hasattr(rec, 'specs') else {},
                        "metric": rec.result,
                        "status": "success" if rec.status == JobStates.success else "failure",
                    })
            best_metric = None
            best_config = None
            if phase_results:
                successes = [r for r in phase_results if r["status"] == "success" and r["metric"] is not None]
                if successes:
                    selector = max if self.reverse_sort else min
                    best = selector(successes, key=lambda r: r["metric"])
                    best_metric = best["metric"]
                    best_config = best["config"]

            self.strategist.record_phase_results(
                phase_plan=self.current_plan,
                results=phase_results,
                best_config=best_config,
                best_metric=best_metric,
            )

        metric_direction = "maximize" if self.reverse_sort else "minimize"
        plan = self.strategist.plan_next_phase(
            available_parameters=self.all_parameters,
            network=self.network,
            metric_name=self.metric,
            metric_direction=metric_direction,
        )

        if plan is None or plan.get("action") == "stop":
            logger.info("Hybrid strategist says stop. Ending search.")
            self._stopped = True
            self.current_plan = plan
            return

        self.current_plan = plan
        self.phase_experiment_count = 0

        self._create_sub_brain(plan)
        logger.info(
            "Hybrid phase %d: algorithm=%s, params=%s, trials=%d",
            len(self.strategist.completed_phases) + 1,
            plan.get("algorithm"), plan.get("parameters"), plan.get("trials"),
        )

    def _create_sub_brain(self, plan: Dict[str, Any]):
        """Create the appropriate sub-brain for the current phase."""
        algorithm = plan.get("algorithm", "bayesian")
        phase_param_names = set(plan.get("parameters", []))

        if phase_param_names:
            phase_params = [p for p in self.all_parameters if p["parameter"] in phase_param_names]
        else:
            phase_params = self.all_parameters

        if not phase_params:
            phase_params = self.all_parameters

        try:
            from tao_automl.brain.factory import BrainFactory, AlgorithmParams
            algo_params = AlgorithmParams.from_dict(plan.get("algorithm_params", {}))
            self.current_sub_brain = BrainFactory.create_brain(
                algorithm=algorithm,
                context=self.context,
                state_store=self.state_store,
                network=self.network,
                parameters=phase_params,
                params=algo_params,
                metric=self.metric,
                resume=False,
            )
            self.num_epochs_per_experiment = getattr(
                self.current_sub_brain, 'num_epochs_per_experiment', 0
            )
        except Exception as e:
            logger.error("Failed to create sub-brain for algorithm '%s': %s", algorithm, e)
            self.current_sub_brain = None

    def save_state(self):
        """Save hybrid brain state."""
        state = {
            "strategist": self.strategist.to_dict(),
            "current_plan": self.current_plan,
            "phase_experiment_count": self.phase_experiment_count,
            "stopped": self._stopped,
        }
        self.state_store.save_brain_info(self.context.id, state)

    @staticmethod
    def load_state(context, state_store, network, parameters, llm_params=None, metric="kpi"):
        """Load hybrid brain state."""
        state = state_store.get_brain_info(context.id)
        brain = HybridBrain(context, state_store, network, parameters, llm_params, metric)

        if state:
            strategist_data = state.get("strategist")
            if strategist_data:
                brain.strategist = HybridStrategist.from_dict(strategist_data, llm_params)
            brain.current_plan = state.get("current_plan")
            brain.phase_experiment_count = state.get("phase_experiment_count", 0)
            brain._stopped = state.get("stopped", False)
            if brain.current_plan and not brain._stopped:
                brain._create_sub_brain(brain.current_plan)
            logger.info(
                "Loaded hybrid brain: %d phases completed, stopped=%s",
                len(brain.strategist.completed_phases), brain._stopped,
            )

        return brain
