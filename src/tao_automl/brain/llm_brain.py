# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM-powered AutoML algorithm.

Plugs into the existing BrainFactory and Controller as automl_algorithm = "llm".
Uses an LLM to generate hyperparameter recommendations instead of GP/Hyperband/DE math.
"""
import json
import logging
import numpy as np
from copy import deepcopy
from typing import Any, Dict, List, Optional

from tao_automl.brain.base import AutoMLAlgorithmBase
from tao_automl.brain.llm_client import LLMClient
from tao_automl.brain.prompts.llm_brain_prompts import (
    build_recommendation_with_reasoning_prompt,
)
from tao_automl.types import JobStates
from tao_automl.utils.math_utils import get_valid_options
from tao_automl.utils.spec_utils import get_flatten_specs

logger = logging.getLogger(__name__)


class LLMBrain(AutoMLAlgorithmBase):
    """LLM-powered AutoML algorithm that uses an LLM to generate hyperparameter recommendations."""

    def __init__(
        self,
        context,
        state_store,
        network,
        parameters,
        llm_params: Optional[Dict[str, Any]] = None,
        metric: str = "kpi",
    ):
        """Initialize the LLMBrain."""
        super().__init__(context, state_store, network, parameters)
        self.llm_client = LLMClient(params=llm_params)
        self.metric = metric
        self.experiment_history: List[Dict[str, Any]] = []
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_metric: Optional[float] = None
        self.external_knowledge: Optional[str] = None
        self.reverse_sort = True
        self.llm_params = llm_params

    def _parameters_with_custom_ranges(self):
        """Return a copy of self.parameters with custom range overrides applied."""
        if not self.custom_ranges:
            return self.parameters

        params = deepcopy(self.parameters)
        for p in params:
            name = p["parameter"]
            if name in self.custom_ranges:
                custom = self.custom_ranges[name]
                for key in ("valid_min", "valid_max", "valid_options"):
                    if custom.get(key) is not None:
                        p[key] = custom[key]
        return params

    def generate_recommendations(self, history):
        """Generate hyperparameter recommendations using an LLM."""
        get_flatten_specs(self.default_train_spec, self.default_train_spec_flattened)

        self._sync_history(history)

        if history and history[-1].status not in [JobStates.success, JobStates.failure]:
            return []

        metric_direction = "minimize" if not self.reverse_sort else "maximize"

        messages = build_recommendation_with_reasoning_prompt(
            parameters=self._parameters_with_custom_ranges(),
            history=self.experiment_history,
            best_config=self.best_config,
            best_metric=self.best_metric,
            network=self.network,
            metric_name=self.metric,
            metric_direction=metric_direction,
        )

        response = self.llm_client.chat(messages, json_mode=True, temperature=0.7)

        if not response.ok:
            logger.warning("LLM call failed: %s. Falling back to random.", response.error)
            return self._random_fallback()

        config = self._parse_llm_response(response)
        if config is None:
            logger.warning("Could not parse LLM response. Falling back to random.")
            return self._random_fallback()

        validated = self._validate_config(config)
        logger.info("LLM recommendation: %s", json.dumps(validated))

        return [validated]

    def _sync_history(self, recommendations):
        """Sync experiment history from controller's recommendation objects."""
        for rec in recommendations:
            if rec.status in [JobStates.success, JobStates.failure]:
                rec_id = rec.id
                if rec_id >= len(self.experiment_history):
                    entry = {
                        "config": rec.specs if hasattr(rec, 'specs') else {},
                        "metric": rec.result if rec.result is not None else 0.0,
                        "status": "success" if rec.status == JobStates.success else "failure",
                    }
                    self.experiment_history.append(entry)

                    if rec.status == JobStates.success:
                        is_better = (
                            self.best_metric is None
                            or (self.reverse_sort and rec.result > self.best_metric)
                            or (not self.reverse_sort and rec.result < self.best_metric)
                        )
                        if is_better:
                            self.best_metric = rec.result
                            self.best_config = rec.specs if hasattr(rec, 'specs') else {}

    def _parse_llm_response(self, response) -> Optional[Dict[str, Any]]:
        """Extract configuration from LLM response."""
        data = response.json_content
        if data is None:
            return None

        if isinstance(data, dict):
            if "config" in data:
                reasoning = data.get("reasoning", "")
                if reasoning:
                    logger.info("LLM reasoning: %s", reasoning)
                return data["config"]
            param_names = {p["parameter"] for p in self.parameters}
            if any(k in param_names for k in data.keys()):
                return data

        return None

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

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clamp LLM-proposed values against schema and custom ranges."""
        validated = {}
        for param_dict in self.parameters:
            name = param_dict["parameter"]
            if name not in config:
                validated[name] = self.generate_automl_param_rec_value(param_dict)
                continue

            value = config[name]
            dtype = param_dict.get("value_type", "")
            v_min, v_max = self._get_effective_bounds(param_dict)

            if dtype == "float":
                try:
                    value = float(value)
                    if v_min not in (None, '', "", "inf", "-inf"):
                        hard_min = float(v_min)
                        if not np.isinf(hard_min) and value < hard_min:
                            value = hard_min
                    if v_max not in (None, '', "", "inf", "-inf"):
                        hard_max = float(v_max)
                        if not np.isinf(hard_max) and value > hard_max:
                            value = hard_max
                except (ValueError, TypeError):
                    value = self.generate_automl_param_rec_value(param_dict)

            elif dtype in ("int", "integer"):
                try:
                    value = int(round(float(value)))
                    if v_min not in (None, '', ""):
                        fmin = float(v_min)
                        if not np.isinf(fmin):
                            value = max(int(fmin), value)
                    if v_max not in (None, '', ""):
                        fmax = float(v_max)
                        if not np.isinf(fmax):
                            value = min(int(fmax), value)
                except (ValueError, TypeError, OverflowError):
                    value = self.generate_automl_param_rec_value(param_dict)

            elif dtype in ("categorical", "ordered"):
                valid_options = get_valid_options(param_dict, self.custom_ranges)
                if valid_options and value not in valid_options:
                    value = self.generate_automl_param_rec_value(param_dict)

            elif dtype == "bool":
                if not isinstance(value, bool):
                    value = str(value).lower() in ("true", "1", "yes")

            validated[name] = value

        return validated

    def _random_fallback(self) -> List[Dict[str, Any]]:
        """Generate a random recommendation when LLM fails."""
        recommendations = []
        for param_dict in self.parameters:
            value = self.generate_automl_param_rec_value(param_dict)
            recommendations.append(value)
        return [dict(zip([p["parameter"] for p in self.parameters], recommendations))]

    def save_state(self):
        """Save LLM brain state to StateStore."""
        state = {
            "experiment_history": self.experiment_history,
            "best_config": self.best_config,
            "best_metric": self.best_metric,
            "llm_usage": self.llm_client.get_usage_summary(),
        }
        self.state_store.save_brain_info(self.context.id, state)

    @staticmethod
    def load_state(context, state_store, network, parameters, llm_params=None, metric="kpi"):
        """Load LLM brain state from StateStore."""
        state = state_store.get_brain_info(context.id)
        brain = LLMBrain(context, state_store, network, parameters, llm_params, metric)

        if state:
            brain.experiment_history = state.get("experiment_history", [])
            brain.best_config = state.get("best_config")
            brain.best_metric = state.get("best_metric")
            logger.info(
                "Loaded LLM brain state: %d experiments, best_metric=%s",
                len(brain.experiment_history), brain.best_metric,
            )

        return brain
