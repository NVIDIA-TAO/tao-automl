# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Autoresearch Controller -- Full Autonomous Agent Loop.

Combines:
- Autoresearch-style keep/discard loop with reasoning
- AutoML-Agent RAP (knowledge retrieval)
- AutoML-Agent training-free pre-screening
- AutoML-Agent multi-stage verification
- LLM-powered spec modification proposals

Operates as ``automl_algorithm = "autoresearch"`` within the TAO AutoML framework.
"""
import logging
import numpy as np
from copy import deepcopy
from typing import Any, Dict, List, Optional

from tao_automl.brain.llm_client import LLMClient
from tao_automl.brain.llm_analyzer import LLMAnalyzer
from tao_automl.brain.knowledge_retriever import KnowledgeRetriever
from tao_automl.brain.spec_prescreener import SpecPrescreener
from tao_automl.brain.verification import MultiStageVerifier
from tao_automl.brain.experiment_tracker import ExperimentTracker
from tao_automl.brain.prompts.autoresearch_prompts import (
    build_autoresearch_prompt,
    build_keep_discard_prompt,
)
from tao_automl.types import JobStates
from tao_automl.utils.math_utils import get_valid_options

logger = logging.getLogger(__name__)


def _metric_is_minimized(metric: str) -> bool:
    return "loss" in (metric or "").lower()


def _finite_numeric_bound(bound: Any) -> Optional[float]:
    if bound in (None, "", "inf", "-inf"):
        return None
    try:
        value = float(bound)
    except (ValueError, TypeError, OverflowError):
        return None
    if not np.isfinite(value):
        return None
    return value


class AutoresearchBrain:
    """Autonomous research brain for TAO AutoML.

    This brain operates differently from traditional algorithms:
    instead of generating hyperparameter recommendations within a fixed search space,
    it proposes full spec modifications, validates them, runs experiments,
    and makes keep/discard decisions with reasoning.
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
        research_program: Optional[str] = None,
        evolvable_text_parameters: Optional[List[str]] = None,
    ):
        """Initialize the AutoresearchBrain."""
        self.context = context
        self.state_store = state_store
        self.network = network
        self.parameters = parameters
        self.metric = metric
        self.max_experiments = max_experiments
        self.research_program = research_program
        self.llm_params = llm_params
        self.evolvable_text_parameters = set(evolvable_text_parameters or [])

        parameter_names = {p.get("parameter") for p in parameters}
        unknown_text_parameters = self.evolvable_text_parameters - parameter_names
        if unknown_text_parameters:
            raise ValueError(
                "evolvable_text_parameters must also be searchable parameters: "
                f"{sorted(unknown_text_parameters)}"
            )
        parameter_lookup = {p.get("parameter"): p for p in parameters}
        invalid_text_parameters = []
        for name in self.evolvable_text_parameters:
            parameter = parameter_lookup[name]
            dtype = parameter.get("value_type")
            options = parameter.get("valid_options") or []
            if dtype == "string":
                continue
            if dtype in ("categorical", "ordered") and options and all(
                isinstance(option, str) for option in options
            ):
                continue
            invalid_text_parameters.append(name)
        if invalid_text_parameters:
            raise TypeError(
                "evolvable_text_parameters must reference string-valued parameters: "
                f"{sorted(invalid_text_parameters)}"
            )

        self.custom_ranges = state_store.get_custom_param_ranges(context.handler_id) or {}
        if self.custom_ranges:
            logger.info(
                "Loaded %d custom parameter range(s) for autoresearch experiment %s",
                len(self.custom_ranges), context.handler_id,
            )

        self.llm_client = LLMClient(params=llm_params)

        self.tracker = ExperimentTracker(
            metric_direction="minimize" if _metric_is_minimized(metric) else "maximize"
        )
        self.knowledge_retriever = KnowledgeRetriever(llm_client=self.llm_client)
        self.prescreener = SpecPrescreener(llm_client=self.llm_client)
        self.verifier = MultiStageVerifier(llm_client=self.llm_client)
        self.analyzer = LLMAnalyzer(llm_client=self.llm_client, analysis_interval=5)

        self.external_knowledge: Optional[str] = None
        self._initialized = False
        self.reverse_sort = not _metric_is_minimized(metric)
        self.num_epochs_per_experiment = 0
        self.spec_schema: Dict[str, Any] = {}
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5

    def initialize(self, base_spec: Dict[str, Any]):
        """Initialize the autoresearch brain with a base spec and retrieve knowledge."""
        self.tracker.best_spec = deepcopy(base_spec)

        self.external_knowledge = self.knowledge_retriever.retrieve_knowledge(
            network=self.network,
            metric_name=self.metric,
            task_description=self.research_program or f"Optimizing {self.network}",
        )
        if self.external_knowledge:
            logger.info("Retrieved external knowledge for %s", self.network)
        self._initialized = True

    def _ensure_initialized(self):
        """Initialize from the persisted base train spec exactly once."""
        if self._initialized:
            return
        base_spec = self.state_store.get_job_specs(self.context.id) or {}
        self.initialize(base_spec)

    def generate_recommendations(self, history):
        """Generate next spec modification using LLM reasoning."""
        self._ensure_initialized()
        self._sync_from_controller(history)

        if history and history[-1].status not in [JobStates.success, JobStates.failure]:
            return []

        if len(self.tracker.history) >= self.max_experiments:
            logger.info("Autoresearch budget exhausted (%d experiments)", self.max_experiments)
            return []

        if self._consecutive_failures >= self._max_consecutive_failures:
            logger.warning(
                "Stopping autoresearch after %d consecutive failures",
                self._consecutive_failures,
            )
            return []

        completed = [e for e in self.tracker.history if e.status in ("success", "failure")]
        if self.analyzer.should_analyze(len(completed)):
            self.analyzer.analyze(
                experiments=self.tracker.get_history_for_llm(),
                parameters=self.parameters,
                network=self.network,
                metric_name=self.metric,
                metric_direction=self.tracker.metric_direction,
                best_metric=self.tracker.best_metric,
            )

        candidates = []
        num_candidates = 3
        for _ in range(num_candidates):
            proposal = self._propose_modification()
            if proposal and proposal.get("modifications"):
                candidates.append(proposal)

        if not candidates:
            logger.warning("LLM failed to propose any modifications, using random fallback")
            return self._random_fallback()

        valid_names = {p["parameter"] for p in self.parameters}
        for candidate in candidates:
            raw_mods = candidate.get("modifications", {})
            filtered = {k: v for k, v in raw_mods.items() if k in valid_names}
            dropped = set(raw_mods.keys()) - valid_names
            if dropped:
                logger.warning(
                    "Dropped invalid parameter names from LLM proposal: %s",
                    dropped,
                )
            candidate["modifications"] = filtered

        for candidate in candidates:
            candidate["modifications"] = self._validate_and_clamp(candidate["modifications"])

        candidates = [c for c in candidates if c.get("modifications")]
        if not candidates:
            logger.warning("All LLM proposals had invalid parameter names, using random fallback")
            return self._random_fallback()

        candidate_specs = [c.get("modifications", {}) for c in candidates]
        reference_results = [
            {"config": e.get("modifications", {}), "metric": e.get("metric")}
            for e in self.tracker.get_history_for_llm()
            if e.get("metric") is not None
        ][-5:]

        recommended = self.prescreener.prescreen(
            candidates=candidate_specs,
            network=self.network,
            metric_name=self.metric,
            metric_direction=self.tracker.metric_direction,
            reference_results=reference_results if reference_results else None,
            max_to_run=1,
            valid_parameter_names=list(valid_names),
        )

        best_candidate = recommended[0] if recommended else candidate_specs[0]
        best_idx = candidate_specs.index(best_candidate) if best_candidate in candidate_specs else 0
        modifications = best_candidate
        reasoning = candidates[best_idx].get("reasoning", "")

        if len(candidates) > 1:
            logger.info(
                "Pre-screener selected candidate %d/%d from %d proposals",
                best_idx + 1, len(recommended), len(candidates),
            )

        verification = self.verifier.verify_spec(
            proposed_changes=modifications,
            spec_schema_summary=self._get_schema_summary(),
            network=self.network,
        )

        if not verification.ok:
            logger.warning(
                "Spec verification failed: %s. Skipping this proposal.",
                verification.issues,
            )
            self._consecutive_failures += 1
            return []

        if verification.warnings:
            logger.info("Spec warnings: %s", verification.warnings)

        logger.info("Autoresearch proposal: %s (reasoning: %s)", modifications, reasoning)

        self._consecutive_failures = 0
        return [modifications]

    def done(self):
        """Return True when autoresearch is complete."""
        if len(self.tracker.history) >= self.max_experiments:
            return True
        if self._consecutive_failures >= self._max_consecutive_failures:
            return True
        return False

    def record_result(self, spec, metric_value, status, job_id=None, feedback=None):
        """Record an experiment result and make keep/discard decision."""
        modifications = self._extract_modifications(spec)

        entry = self.tracker.record_experiment(
            spec=spec,
            modifications=modifications,
            metric=metric_value,
            status="success" if status == "success" else "failure",
            reasoning="",
            job_id=job_id,
            feedback=feedback,
        )

        if status == "success" and metric_value is not None:
            reasoning = self._get_keep_discard_reasoning(
                current_result={
                    "metric": metric_value,
                    "status": status,
                    "feedback": feedback,
                },
                modifications=modifications,
            )
            entry.reasoning = reasoning

        result_verification = self.verifier.verify_result(
            result={"metric": metric_value, "status": status, "feedback": feedback},
            expected_range=None,
            metric_name=self.metric,
            network=self.network,
        )
        if not result_verification.should_count:
            logger.warning("Result verification: should not count this result")
            entry.decision = "discard"

        return entry

    def _parameters_with_custom_ranges(self):
        """Return a copy of self.parameters with custom range overrides applied."""
        if not self.custom_ranges and not self.evolvable_text_parameters:
            return self.parameters

        from copy import deepcopy
        params = deepcopy(self.parameters)
        for p in params:
            name = p["parameter"]
            if name in self.evolvable_text_parameters:
                p["evolvable_text"] = True
            if name in self.custom_ranges:
                custom = self.custom_ranges[name]
                for key in ("valid_min", "valid_max", "valid_options"):
                    if custom.get(key) is not None:
                        p[key] = custom[key]
        return params

    def _get_effective_bounds(self, param_dict):
        """Get effective min/max bounds considering custom range overrides."""
        name = param_dict["parameter"]
        v_min = param_dict.get("valid_min")
        v_max = param_dict.get("valid_max")

        if self.custom_ranges and name in self.custom_ranges:
            custom = self.custom_ranges[name]
            if custom.get("valid_min") is not None:
                v_min = custom["valid_min"]
            if custom.get("valid_max") is not None:
                v_max = custom["valid_max"]
        return v_min, v_max

    def _validate_and_clamp(self, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clamp LLM-proposed values against schema and custom ranges."""
        param_lookup = {p["parameter"]: p for p in self.parameters}
        validated = {}

        for name, value in modifications.items():
            param_dict = param_lookup.get(name)
            if param_dict is None:
                continue

            dtype = param_dict.get("value_type", "")
            v_min, v_max = self._get_effective_bounds(param_dict)

            if name in self.evolvable_text_parameters:
                if not isinstance(value, str) or not value.strip():
                    continue
                if len(value) > 16384:
                    logger.warning(
                        "Dropping oversized text proposal for %s (%d characters)",
                        name, len(value),
                    )
                    continue
                value = value.strip()

            elif dtype == "float":
                try:
                    value = float(value)
                    hard_min = _finite_numeric_bound(v_min)
                    if hard_min is not None and value < hard_min:
                        value = hard_min
                    hard_max = _finite_numeric_bound(v_max)
                    if hard_max is not None and value > hard_max:
                        value = hard_max
                except (ValueError, TypeError):
                    continue

            elif dtype in ("int", "integer"):
                try:
                    value = int(round(float(value)))
                    hard_min = _finite_numeric_bound(v_min)
                    if hard_min is not None:
                        value = max(int(np.ceil(hard_min)), value)
                    hard_max = _finite_numeric_bound(v_max)
                    if hard_max is not None:
                        hard_max_int = int(np.floor(hard_max))
                        if value > hard_max_int:
                            value = hard_max_int
                except (ValueError, TypeError, OverflowError):
                    continue

            elif dtype in ("categorical", "ordered"):
                valid_options = get_valid_options(param_dict, self.custom_ranges)
                if valid_options and value not in valid_options:
                    continue

            elif dtype == "bool":
                if not isinstance(value, bool):
                    value = str(value).lower() in ("true", "1", "yes")

            validated[name] = value

        return validated

    def _propose_modification(self) -> Optional[Dict[str, Any]]:
        """Use LLM to propose the next spec modification."""
        messages = build_autoresearch_prompt(
            spec_schema=self.spec_schema,
            current_best_spec=self.tracker.best_spec or {},
            experiment_history=self.tracker.get_history_for_llm(),
            network=self.network,
            metric_name=self.metric,
            metric_direction=self.tracker.metric_direction,
            research_program=self.research_program,
            external_knowledge=self.external_knowledge,
            parameters=self._parameters_with_custom_ranges(),
        )

        response = self.llm_client.chat(messages, json_mode=True, temperature=0.7)

        if not response.ok or response.json_content is None:
            return None

        return response.json_content

    def _get_keep_discard_reasoning(
        self, current_result: Dict[str, Any], modifications: Dict[str, Any]
    ) -> str:
        """Get LLM reasoning for keep/discard decision."""
        best_result = {"metric": self.tracker.best_metric or 0.0}
        messages = build_keep_discard_prompt(
            current_result=current_result,
            best_result=best_result,
            modifications_made=modifications,
            metric_name=self.metric,
            metric_direction=self.tracker.metric_direction,
        )

        response = self.llm_client.chat(messages, json_mode=True, temperature=0.2)
        if response.ok and response.json_content:
            return response.json_content.get("reasoning", "")
        return ""

    def _extract_modifications(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Extract modifications from a spec dict."""
        if not spec:
            return {}

        valid_names = {p["parameter"] for p in self.parameters}

        if any(k in valid_names for k in spec.keys()):
            return {k: v for k, v in spec.items() if k in valid_names}

        mods = {}
        for param in self.parameters:
            name = param["parameter"]
            parts = name.split(".")
            val = spec
            for part in parts:
                if isinstance(val, dict) and part in val:
                    val = val[part]
                else:
                    val = None
                    break
            if val is not None:
                mods[name] = val
        return mods

    def _sync_from_controller(self, recommendations):
        """Sync state from controller's recommendation objects."""
        for rec in recommendations:
            if rec.status in [JobStates.success, JobStates.failure]:
                if rec.id >= len(self.tracker.history):
                    self.record_result(
                        spec=rec.specs if hasattr(rec, 'specs') else {},
                        metric_value=rec.result,
                        status="success" if rec.status == JobStates.success else "failure",
                        job_id=rec.job_id if hasattr(rec, 'job_id') else None,
                        feedback=getattr(rec, "feedback", None),
                    )

    def _get_schema_summary(self) -> str:
        """Generate a brief schema summary for verification prompts."""
        lines = []
        for p in self._parameters_with_custom_ranges()[:20]:
            name = p.get("parameter", "")
            dtype = p.get("value_type", "")
            v_min = p.get("valid_min", "")
            v_max = p.get("valid_max", "")
            if name in self.evolvable_text_parameters:
                lines.append(f"{name} (free-form evolvable text)")
            else:
                lines.append(f"{name} ({dtype}): [{v_min}, {v_max}]")
        return "\n".join(lines)

    def _random_fallback(self) -> List[Dict[str, Any]]:
        """Generate random recommendation as fallback."""
        from tao_automl.brain.base import AutoMLAlgorithmBase
        base = AutoMLAlgorithmBase(self.context, self.state_store, self.network, self.parameters)
        rec = {}
        for param in self.parameters:
            rec[param["parameter"]] = base.generate_automl_param_rec_value(param)
        return [rec]

    def save_state(self):
        """Save autoresearch state to StateStore."""
        state = {
            "tracker": self.tracker.to_dict(),
            "external_knowledge": self.external_knowledge,
            "initialized": self._initialized,
            "consecutive_failures": self._consecutive_failures,
            "llm_usage": self.llm_client.get_usage_summary(),
            "analyses": self.analyzer.format_for_metadata(),
        }
        self.state_store.save_brain_info(self.context.id, state)

    @staticmethod
    def load_state(
        context, state_store, network, parameters,
        llm_params=None, metric="kpi", max_experiments=50, research_program=None,
        evolvable_text_parameters=None,
    ):
        """Load autoresearch state from StateStore."""
        state = state_store.get_brain_info(context.id)
        brain = AutoresearchBrain(
            context, state_store, network, parameters,
            llm_params, metric, max_experiments, research_program,
            evolvable_text_parameters,
        )

        if state:
            tracker_data = state.get("tracker")
            if tracker_data:
                brain.tracker = ExperimentTracker.from_dict(tracker_data)
            brain.external_knowledge = state.get("external_knowledge")
            brain._initialized = state.get("initialized", bool(brain.tracker.best_spec))
            brain._consecutive_failures = state.get("consecutive_failures", 0)
            logger.info(
                "Loaded autoresearch state: %d experiments, best_metric=%s",
                len(brain.tracker.history), brain.tracker.best_metric,
            )

        return brain
