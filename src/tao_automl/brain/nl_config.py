# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Natural Language AutoML Configuration.

Translates user's natural language goals into AutoML configuration.
"""
import json
import logging
from typing import Any, Dict, List, Optional

from tao_automl.brain.llm_client import LLMClient, first_json_object
from tao_automl.brain.prompts.nl_config_prompts import build_nl_config_prompt

logger = logging.getLogger(__name__)

VALID_ALGORITHMS = {
    "bayesian", "hyperband", "bohb", "bfbo", "asha", "pbt", "dehb",
    "hyperband_es", "llm", "autoresearch",
}


class NLConfigGenerator:
    """Generates AutoML configuration from natural language descriptions."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        llm_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the NLConfigGenerator."""
        self.llm_client = llm_client or LLMClient(params=llm_params)

    def generate_config(
        self,
        user_prompt: str,
        network: str,
        available_parameters: List[Dict[str, Any]],
        dataset_info: Optional[str] = None,
        hardware_info: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate AutoML configuration from natural language.

        Returns:
            Dict with automl_algorithm, automl_hyperparameters,
            algorithm_specific_params, automl_range_override, metric, reasoning.

        Raises:
            ValueError: If LLM fails or returns invalid config.
        """
        messages = build_nl_config_prompt(
            user_prompt=user_prompt,
            network=network,
            available_parameters=available_parameters,
            dataset_info=dataset_info,
            hardware_info=hardware_info,
        )

        response = self.llm_client.chat(messages, json_mode=True, temperature=0.3)

        if not response.ok:
            raise ValueError(f"LLM config generation failed: {response.error}")

        config = first_json_object(response.json_content)
        if config is None:
            raise ValueError("Could not parse LLM response as JSON")

        validated = self._validate_config(config, available_parameters)
        logger.info(
            "NL Config generated: algorithm=%s, %d params, reasoning: %s",
            validated.get("automl_algorithm"),
            len(validated.get("automl_hyperparameters", [])),
            validated.get("reasoning", "")[:100],
        )
        return validated

    def _validate_config(
        self, config: Dict[str, Any], available_parameters: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate and sanitize the LLM-generated config."""
        available_names = {p["parameter"] for p in available_parameters}

        algorithm = str(config.get("automl_algorithm", "bayesian")).lower()
        if algorithm not in VALID_ALGORITHMS:
            logger.warning("Invalid algorithm '%s', defaulting to bayesian", algorithm)
            algorithm = "bayesian"

        raw_params = config.get("automl_hyperparameters", [])
        if isinstance(raw_params, str):
            try:
                raw_params = json.loads(raw_params)
            except json.JSONDecodeError:
                raw_params = [p.strip() for p in raw_params.split(",")]

        valid_params = [p for p in raw_params if p in available_names]
        if not valid_params and available_names:
            logger.warning("No valid params from LLM, using first 5 available")
            valid_params = list(available_names)[:5]

        range_override = config.get("automl_range_override", [])
        validated_overrides = []
        for override in (range_override or []):
            if isinstance(override, dict) and override.get("parameter") in available_names:
                validated_overrides.append(override)

        return {
            "automl_algorithm": algorithm,
            "automl_hyperparameters": valid_params,
            "algorithm_specific_params": config.get("algorithm_specific_params", {}),
            "automl_range_override": validated_overrides,
            "metric": config.get("metric", "kpi"),
            "reasoning": config.get("reasoning", ""),
        }
