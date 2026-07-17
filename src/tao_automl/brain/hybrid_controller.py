# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid Strategist.

LLM operates at a higher level than individual trials: it decides WHAT
dimension to explore and WHICH algorithm to use, then delegates to existing
TAO algorithms (Bayesian, ASHA, BOHB, etc.) for the actual sweep.

HybridBrain wraps the strategist as a Controller-compatible brain.
"""
import copy
import logging
import math
from typing import Any, Dict, List, Optional

from tao_automl.brain.llm_client import LLMClient
from tao_automl.brain.prompts.autoresearch_prompts import (
    build_hybrid_strategy_prompt,
)
from tao_automl.types import JobStates

logger = logging.getLogger(__name__)

AVAILABLE_ALGORITHMS = ["bayesian", "asha", "bohb", "dehb", "pbt", "hyperband", "llm", "autoresearch"]

CORE_TRAINING_PARAMETERS = (
    "train.epoch",
    "train.train_batch_per_replica",
    "train.optm_lr",
    "train.optm_weight_decay",
    "train.optm_warmup_epochs",
)

LORA_PARAMETERS = (
    "policy.lora.r",
    "policy.lora.lora_alpha",
    "policy.lora.lora_dropout",
)


def _metric_is_minimized(metric: str) -> bool:
    return "loss" in (metric or "").lower()


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
        enable_range_narrowing: bool = False,
    ):
        """Initialize the HybridStrategist."""
        self.llm_client = llm_client or LLMClient(params=llm_params)
        self.enable_range_narrowing = enable_range_narrowing
        self.completed_phases: List[Dict[str, Any]] = []
        self.full_history: List[Dict[str, Any]] = []
        self._restored_llm_usage: Dict[str, Any] = {}

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
            enable_range_narrowing=self.enable_range_narrowing,
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
        reverse_sort: bool = False,
    ):
        """Record results of a completed phase for history tracking."""
        annotated_results = []
        for result in results:
            annotated = copy.deepcopy(result)
            if annotated.get("status") == "success" and annotated.get("metric") is not None:
                annotated["decision"] = (
                    "keep" if best_metric is not None and annotated["metric"] == best_metric
                    else "discard"
                )
            else:
                annotated["decision"] = "discard"
            annotated_results.append(annotated)

        successes = [
            result for result in annotated_results
            if result.get("status") == "success" and result.get("metric") is not None
        ]
        failures = [
            result for result in annotated_results if result.get("status") == "failure"
        ]
        ordered_successes = sorted(
            successes,
            key=lambda result: result["metric"],
            reverse=reverse_sort,
        )
        phase_record = {
            "phase_number": len(self.completed_phases) + 1,
            "plan": phase_plan,
            "num_experiments": len(results),
            "num_success": len(successes),
            "num_failure": len(failures),
            "best_config": best_config,
            "best_metric": best_metric,
            "top_results": ordered_successes[:5],
            "bottom_results": ordered_successes[-5:],
            "failed_results": failures[:5],
        }
        self.completed_phases.append(phase_record)
        self.full_history.extend(annotated_results)

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
        available_lookup = {p["parameter"]: p for p in available_parameters}

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
            valid_params = [p["parameter"] for p in available_parameters[:5]]
        valid_params = self._expand_parameter_dependencies(
            valid_params, available_parameters, available_lookup, available_names
        )
        if not self.completed_phases:
            valid_params, added_params = self._expand_initial_coverage_guardrails(
                valid_params, available_parameters, available_names
            )
            if added_params:
                plan["guardrail_added_parameters"] = added_params
        plan["parameters"] = valid_params

        trials = plan.get("trials", 5)
        if not isinstance(trials, int) or trials < 1:
            trials = 5
        plan["trials"] = min(trials, 50)

        if self.enable_range_narrowing and self.completed_phases:
            overrides, rejected = self._validate_parameter_overrides(
                plan=plan,
                available_lookup=available_lookup,
                selected_params=set(valid_params),
            )
            if overrides:
                plan["parameter_overrides"] = overrides
            else:
                plan.pop("parameter_overrides", None)
            if rejected:
                plan["rejected_parameter_overrides"] = rejected
        else:
            for key in ("parameter_overrides", "parameter_ranges", "range_overrides", "suggested_ranges"):
                plan.pop(key, None)

        return plan

    @staticmethod
    def _extract_parameter_overrides(plan: Dict[str, Any]) -> Dict[str, Any]:
        """Accept a few common LLM field names for phase-local range overrides."""
        raw = None
        for key in ("parameter_overrides", "parameter_ranges", "range_overrides", "suggested_ranges"):
            if isinstance(plan.get(key), (dict, list)):
                raw = plan.get(key)
                break
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw

        overrides = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = item.get("parameter") or item.get("name")
            if not name:
                continue
            override = {k: v for k, v in item.items() if k not in ("parameter", "name")}
            overrides[name] = override
        return overrides

    def _validate_parameter_overrides(
        self,
        plan: Dict[str, Any],
        available_lookup: Dict[str, Dict[str, Any]],
        selected_params: set[str],
    ) -> tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        """Validate LLM range narrowing against the effective phase schema."""
        raw_overrides = self._extract_parameter_overrides(plan)
        if not raw_overrides:
            return {}, []

        accepted: Dict[str, Dict[str, Any]] = {}
        rejected: List[Dict[str, Any]] = []
        for name, override in raw_overrides.items():
            if name not in available_lookup:
                rejected.append({"parameter": name, "reason": "unknown parameter"})
                continue
            if name not in selected_params:
                rejected.append({"parameter": name, "reason": "parameter not selected for phase"})
                continue
            if not isinstance(override, dict):
                rejected.append({"parameter": name, "reason": "override must be an object"})
                continue

            param = available_lookup[name]
            option_override = self._validate_option_override(param, override)
            if option_override:
                accepted[name] = option_override
                continue

            range_override = self._validate_numeric_override(param, override)
            if range_override:
                accepted[name] = range_override
                continue

            rejected.append({"parameter": name, "reason": "not a valid narrowing"})

        return accepted, rejected

    @staticmethod
    def _as_option_list(value: Any) -> List[Any]:
        if value in (None, ""):
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @classmethod
    def _validate_option_override(
        cls, param: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        original_options = cls._as_option_list(param.get("valid_options"))
        if not original_options:
            return {}

        requested_options = cls._as_option_list(override.get("valid_options"))
        if not requested_options:
            lower_bound = cls._to_float(override.get("valid_min"))
            upper_bound = cls._to_float(override.get("valid_max"))
            if lower_bound is None and upper_bound is None:
                return {}
            numeric_options = [
                (option, cls._to_float(option)) for option in original_options
            ]
            if any(value is None for _, value in numeric_options):
                return {}
            requested_options = [
                option for option, value in numeric_options
                if (lower_bound is None or value >= lower_bound)
                and (upper_bound is None or value <= upper_bound)
            ]

        selected = []
        remaining = list(original_options)
        for requested in requested_options:
            for option in remaining:
                if cls._option_matches(option, requested):
                    selected.append(option)
                    remaining.remove(option)
                    break

        if not selected or len(selected) > len(original_options):
            return {}
        if len(selected) == len(original_options):
            return {}
        return {"valid_options": selected}

    @classmethod
    def _validate_numeric_override(
        cls, param: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        original_min = cls._to_float(param.get("valid_min"))
        original_max = cls._to_float(param.get("valid_max"))
        if original_min is None or original_max is None:
            return {}
        if math.isinf(original_min) or math.isinf(original_max):
            return {}

        suggested_min = cls._to_float(override.get("valid_min"))
        suggested_max = cls._to_float(override.get("valid_max"))
        if suggested_min is None and suggested_max is None:
            return {}

        new_min = original_min if suggested_min is None else max(original_min, suggested_min)
        new_max = original_max if suggested_max is None else min(original_max, suggested_max)
        if new_min > new_max:
            return {}
        if new_min == original_min and new_max == original_max:
            return {}

        value_type = param.get("value_type")
        if value_type in ("int", "integer"):
            return {"valid_min": int(round(new_min)), "valid_max": int(round(new_max))}
        return {"valid_min": new_min, "valid_max": new_max}

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(result):
            return None
        return result

    @staticmethod
    def _option_matches(option: Any, requested: Any) -> bool:
        if option == requested:
            return True
        if str(option) == str(requested):
            return True
        try:
            return float(option) == float(requested)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _expand_parameter_dependencies(
        params: List[str],
        available_parameters: List[Dict[str, Any]],
        available_lookup: Dict[str, Dict[str, Any]],
        available_names: set[str],
    ) -> List[str]:
        """Keep dependent parameters together inside a Hybrid phase.

        The strategist can choose a focused subset, but sub-brains only sample
        parameters present in that subset. If the plan includes a parent such as
        ``model.num_queries`` without its dependent ``model.num_select``, the
        train spec keeps the stale default for the dependent parameter and can
        become invalid. Expand the subset both ways while preserving schema order.
        """
        selected = set(params)
        changed = True
        while changed:
            changed = False
            for name in list(selected):
                depends_on = available_lookup.get(name, {}).get("depends_on")
                if depends_on in available_names and depends_on not in selected:
                    selected.add(depends_on)
                    changed = True
            for item in available_parameters:
                name = item.get("parameter")
                depends_on = item.get("depends_on")
                if depends_on in selected and name in available_names and name not in selected:
                    selected.add(name)
                    changed = True
        return [p["parameter"] for p in available_parameters if p["parameter"] in selected]

    @staticmethod
    def _expand_initial_coverage_guardrails(
        params: List[str],
        available_parameters: List[Dict[str, Any]],
        available_names: set[str],
    ) -> tuple[List[str], List[str]]:
        """Avoid harmful first-phase pruning of core training dimensions.

        The first phase is the only source of evidence for later LLM strategy.
        Keeping the core train knobs, and LoRA knobs when present, prevents a
        text prior from removing dimensions that traditional AutoML would have
        been able to explore.
        """
        selected = set(params)
        required = [p for p in CORE_TRAINING_PARAMETERS if p in available_names]
        if any(p.startswith("policy.lora.") for p in available_names):
            required.extend(p for p in LORA_PARAMETERS if p in available_names)

        added = [p for p in required if p not in selected]
        selected.update(added)
        ordered = [p["parameter"] for p in available_parameters if p["parameter"] in selected]
        return ordered, added

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state."""
        return {
            "enable_range_narrowing": self.enable_range_narrowing,
            "completed_phases": self.completed_phases,
            "full_history": self.full_history,
            "llm_usage": self._combined_llm_usage(),
        }

    def _combined_llm_usage(self) -> Dict[str, Any]:
        """Return cumulative usage, including calls made before a resume."""
        usage = getattr(getattr(self, "llm_client", None), "usage", None)
        current = usage.to_dict() if usage is not None and hasattr(usage, "to_dict") else {}
        keys = (
            "prompt_tokens", "completion_tokens", "total_tokens", "num_calls",
            "total_latency_ms", "errors",
        )
        combined = {
            key: self._restored_llm_usage.get(key, 0) + current.get(key, 0)
            for key in keys
        }
        combined["total_latency_ms"] = round(combined["total_latency_ms"], 1)
        return combined

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        llm_params=None,
        enable_range_narrowing: Optional[bool] = None,
    ) -> "HybridStrategist":
        """Deserialize state."""
        if enable_range_narrowing is None:
            enable_range_narrowing = bool(data.get("enable_range_narrowing", False))
        strategist = cls(
            llm_params=llm_params,
            enable_range_narrowing=enable_range_narrowing,
        )
        strategist.completed_phases = data.get("completed_phases", [])
        strategist.full_history = data.get("full_history", [])
        strategist._restored_llm_usage = data.get("llm_usage", {})
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
        max_experiments: int = 50,
        enable_llm_range_narrowing: bool = False,
    ):
        """Initialize the HybridBrain."""
        self.context = context
        self.state_store = state_store
        self.network = network
        self.all_parameters = parameters
        self.metric = metric
        self.llm_params = llm_params
        self.max_experiments = max_experiments
        self.enable_llm_range_narrowing = enable_llm_range_narrowing
        self.base_custom_ranges = copy.deepcopy(
            self.state_store.get_custom_param_ranges(self.context.handler_id) or {}
        )

        self.strategist = HybridStrategist(
            llm_params=llm_params,
            enable_range_narrowing=enable_llm_range_narrowing,
        )
        self.current_plan: Optional[Dict[str, Any]] = None
        self.current_sub_brain = None
        self.phase_experiment_count = 0
        self.current_phase_start = 0
        self.last_recorded_phase_start = -1
        self.total_experiment_count = 0
        self.reverse_sort = not _metric_is_minimized(metric)
        self.num_epochs_per_experiment = 0
        self._stopped = False

    def generate_recommendations(self, history):
        """Generate recommendations by delegating to the current phase's sub-brain."""
        self._sync_counts(history)
        if self.total_experiment_count >= self.max_experiments:
            self._record_current_phase(history)
            logger.info(
                "Hybrid experiment budget exhausted (%d/%d). Ending search.",
                self.total_experiment_count,
                self.max_experiments,
            )
            self._stopped = True
            return []

        if self._stopped:
            return []

        if self.current_plan is None or self._phase_budget_exhausted():
            self._advance_phase(history)
            if self._stopped:
                return []
            self._sync_counts(history)

        if self.current_sub_brain is None:
            return []

        phase_history = history[self.current_phase_start:]

        raw_recs = self.current_sub_brain.generate_recommendations(phase_history)
        return self._cap_recommendations_to_budget(raw_recs)

    def done(self):
        """Return True when the hybrid brain has stopped."""
        return self._stopped

    def _sync_counts(self, history):
        """Synchronize phase counters from the controller history."""
        self.total_experiment_count = len(history)
        self.phase_experiment_count = max(
            0, len(history) - max(0, self.current_phase_start)
        )

    def _phase_budget_exhausted(self) -> bool:
        if self.current_plan is None:
            return True
        return self.phase_experiment_count >= self.current_plan.get("trials", 5)

    def _cap_recommendations_to_budget(self, recommendations):
        """Prevent sub-brains from overshooting hybrid phase or total budget."""
        if not recommendations:
            return recommendations

        phase_remaining = self.current_plan.get("trials", 5) - self.phase_experiment_count
        total_remaining = self.max_experiments - self.total_experiment_count
        allowed = max(0, min(phase_remaining, total_remaining))
        if allowed <= 0:
            return []
        if len(recommendations) > allowed:
            logger.info(
                "Capping Hybrid recommendations from %d to %d to preserve phase/budget limits",
                len(recommendations),
                allowed,
            )
            return recommendations[:allowed]
        return recommendations

    def _record_current_phase(self, history):
        """Persist current phase results once, including terminal budget stops."""
        if not self.current_plan or self.current_plan.get("action") == "stop":
            return
        if self.last_recorded_phase_start == self.current_phase_start:
            return

        phase_results = []
        for rec in history[self.current_phase_start:]:
            if rec.status in [JobStates.success, JobStates.failure]:
                phase_results.append({
                    "rec_id": rec.id,
                    "config": rec.specs if hasattr(rec, 'specs') else {},
                    "metric": rec.result,
                    "status": "success" if rec.status == JobStates.success else "failure",
                })
        best_metric = None
        best_config = None
        successes = [
            result for result in phase_results
            if result["status"] == "success" and result["metric"] is not None
        ]
        if successes:
            selector = max if self.reverse_sort else min
            best = selector(successes, key=lambda result: result["metric"])
            best_metric = best["metric"]
            best_config = best["config"]

        self.strategist.record_phase_results(
            phase_plan=self.current_plan,
            results=phase_results,
            best_config=best_config,
            best_metric=best_metric,
            reverse_sort=self.reverse_sort,
        )
        self.last_recorded_phase_start = self.current_phase_start

    def _advance_phase(self, history):
        """Record current phase results (if any) and plan the next phase."""
        self._record_current_phase(history)

        metric_direction = "maximize" if self.reverse_sort else "minimize"
        available_parameters = self._parameters_for_planning()
        plan = self.strategist.plan_next_phase(
            available_parameters=available_parameters,
            network=self.network,
            metric_name=self.metric,
            metric_direction=metric_direction,
        )

        if plan is None or plan.get("action") == "stop":
            logger.info("Hybrid strategist says stop. Ending search.")
            self._stopped = True
            self.current_plan = plan
            return

        remaining_budget = max(0, self.max_experiments - len(history))
        if remaining_budget <= 0:
            logger.info(
                "Hybrid experiment budget exhausted before next phase (%d/%d).",
                len(history),
                self.max_experiments,
            )
            self._stopped = True
            self.current_plan = plan
            return

        requested_trials = plan.get("trials", remaining_budget)
        capped_trials = min(requested_trials, remaining_budget)
        reserved_refinement_budget = self._reserved_refinement_budget(remaining_budget)
        if reserved_refinement_budget:
            first_phase_cap = remaining_budget - reserved_refinement_budget
            capped_trials = min(capped_trials, first_phase_cap)
            plan["reserved_refinement_budget"] = reserved_refinement_budget
        if capped_trials != requested_trials:
            logger.info(
                "Capping Hybrid phase trials from %d to %d",
                requested_trials,
                capped_trials,
            )
        plan["trials"] = capped_trials

        self.current_plan = plan
        self.current_phase_start = len(history)
        self.last_recorded_phase_start = -1
        self.phase_experiment_count = 0

        self._create_sub_brain(plan)
        logger.info(
            "Hybrid phase %d: algorithm=%s, params=%s, trials=%d",
            len(self.strategist.completed_phases) + 1,
            plan.get("algorithm"), plan.get("parameters"), plan.get("trials"),
        )

    def _reserved_refinement_budget(self, remaining_budget: int) -> int:
        """Reserve enough budget for at least one evidence-based LLM phase."""
        if self.strategist.completed_phases:
            return 0
        if remaining_budget >= 8:
            return max(2, min(6, remaining_budget // 4))
        if remaining_budget >= 6:
            return 2
        if remaining_budget >= 4:
            return 1
        if remaining_budget >= 2:
            return 1
        return 0

    def _parameters_for_planning(self) -> List[Dict[str, Any]]:
        """Return the schema visible to the LLM strategist."""
        if not self.enable_llm_range_narrowing:
            return self.all_parameters
        return self._parameters_with_custom_ranges(self.all_parameters, self.base_custom_ranges)

    @staticmethod
    def _parameters_with_custom_ranges(
        parameters: List[Dict[str, Any]],
        custom_ranges: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply custom ranges to parameter copies for prompt/sub-brain consistency."""
        if not custom_ranges:
            return [copy.deepcopy(param) for param in parameters]

        merged = []
        for param in parameters:
            param_copy = copy.deepcopy(param)
            name = param_copy.get("parameter")
            if name in custom_ranges:
                for key, value in custom_ranges[name].items():
                    if value is not None:
                        param_copy[key] = value
            merged.append(param_copy)
        return merged

    def _phase_custom_ranges(self, plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Build the custom ranges the current sub-brain should see."""
        phase_ranges = copy.deepcopy(self.base_custom_ranges)
        if not self.enable_llm_range_narrowing:
            return phase_ranges

        plan.pop("applied_parameter_overrides", None)
        applied_overrides = {}
        for name, override in (plan.get("parameter_overrides") or {}).items():
            if not isinstance(override, dict):
                continue
            phase_ranges.setdefault(name, {})
            for key in ("valid_min", "valid_max", "valid_options", "option_weights"):
                if key in override and override[key] is not None:
                    phase_ranges[name][key] = override[key]
            applied_overrides[name] = {
                key: phase_ranges[name][key]
                for key in ("valid_min", "valid_max", "valid_options", "option_weights")
                if key in phase_ranges[name]
            }
        if applied_overrides:
            plan["applied_parameter_overrides"] = applied_overrides
        return phase_ranges

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
            if self.enable_llm_range_narrowing:
                phase_ranges = self._phase_custom_ranges(plan)
                self.state_store.save_custom_param_ranges(self.context.handler_id, phase_ranges)
                phase_params = self._parameters_with_custom_ranges(phase_params, phase_ranges)
                if plan.get("applied_parameter_overrides"):
                    logger.info(
                        "Hybrid phase applying LLM range overrides: %s",
                        plan["applied_parameter_overrides"],
                    )
            algo_params = AlgorithmParams.from_dict(plan.get("algorithm_params", {}))
            # BaseBrain derives its random seed from context.id. A fresh
            # sub-brain with the parent context would therefore replay the same
            # first proposal in every Hybrid phase. Keep the handler/workspace
            # scope, but give each phase a stable distinct seed identity.
            phase_context = copy.copy(self.context)
            phase_number = len(self.strategist.completed_phases) + 1
            phase_context.id = f"{self.context.id}-hybrid-phase-{phase_number}"
            self.current_sub_brain = BrainFactory.create_brain(
                algorithm=algorithm,
                context=phase_context,
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
            "current_phase_start": self.current_phase_start,
            "last_recorded_phase_start": self.last_recorded_phase_start,
            "total_experiment_count": self.total_experiment_count,
            "max_experiments": self.max_experiments,
            "stopped": self._stopped,
            "enable_llm_range_narrowing": self.enable_llm_range_narrowing,
            "base_custom_ranges": self.base_custom_ranges,
        }
        self.state_store.save_brain_info(self.context.id, state)

    @staticmethod
    def load_state(
        context, state_store, network, parameters,
        llm_params=None, metric="kpi", max_experiments=50,
        enable_llm_range_narrowing=False,
    ):
        """Load hybrid brain state."""
        state = state_store.get_brain_info(context.id)
        brain = HybridBrain(
            context, state_store, network, parameters,
            llm_params, metric, max_experiments, enable_llm_range_narrowing,
        )

        if state:
            brain.enable_llm_range_narrowing = state.get(
                "enable_llm_range_narrowing",
                enable_llm_range_narrowing,
            )
            brain.base_custom_ranges = copy.deepcopy(
                state.get("base_custom_ranges") or brain.base_custom_ranges
            )
            strategist_data = state.get("strategist")
            if strategist_data:
                brain.strategist = HybridStrategist.from_dict(
                    strategist_data,
                    llm_params,
                    enable_range_narrowing=brain.enable_llm_range_narrowing,
                )
            else:
                brain.strategist.enable_range_narrowing = brain.enable_llm_range_narrowing
            brain.current_plan = state.get("current_plan")
            brain.phase_experiment_count = state.get("phase_experiment_count", 0)
            brain.current_phase_start = state.get("current_phase_start", 0)
            brain.last_recorded_phase_start = state.get("last_recorded_phase_start", -1)
            brain.total_experiment_count = state.get("total_experiment_count", 0)
            brain.max_experiments = state.get("max_experiments", max_experiments)
            brain._stopped = state.get("stopped", False)
            if brain.current_plan and not brain._stopped:
                brain._create_sub_brain(brain.current_plan)
            logger.info(
                "Loaded hybrid brain: %d phases completed, stopped=%s",
                len(brain.strategist.completed_phases), brain._stopped,
            )

        return brain
