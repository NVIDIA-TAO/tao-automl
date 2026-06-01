# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training-free Pre-screening (AutoML-Agent concept).

Uses an LLM to predict which candidate configurations are worth running
BEFORE spending GPU-hours on actual training.
"""
import logging
from typing import Any, Dict, List, Optional

from tao_automl.brain.llm_client import LLMClient
from tao_automl.brain.prompts.autoresearch_prompts import (
    build_prescreen_prompt,
)

logger = logging.getLogger(__name__)


class SpecPrescreener:
    """Pre-screens candidate configurations using LLM prediction."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        llm_params: Optional[Dict[str, Any]] = None,
        min_candidates_to_screen: int = 3,
    ):
        """Initialize the SpecPrescreener."""
        self.llm_client = llm_client or LLMClient(params=llm_params)
        self.min_candidates_to_screen = min_candidates_to_screen

    def prescreen(
        self,
        candidates: List[Dict[str, Any]],
        network: str,
        metric_name: str,
        metric_direction: str,
        reference_results: Optional[List[Dict[str, Any]]] = None,
        max_to_run: Optional[int] = None,
        valid_parameter_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Pre-screen candidates and return the recommended subset."""
        if len(candidates) < self.min_candidates_to_screen:
            logger.info(
                "Only %d candidates, below threshold %d -- skipping pre-screen",
                len(candidates), self.min_candidates_to_screen,
            )
            return candidates

        messages = build_prescreen_prompt(
            candidates=candidates,
            network=network,
            metric_name=metric_name,
            metric_direction=metric_direction,
            reference_results=reference_results,
            valid_parameter_names=valid_parameter_names,
        )

        response = self.llm_client.chat(messages, json_mode=True, temperature=0.2)

        if not response.ok or response.json_content is None:
            logger.warning("Pre-screening failed: %s. Returning all candidates.", response.error)
            return candidates

        data = response.json_content
        recommended_indices = data.get("recommended_to_run", [])
        confidence = data.get("confidence", "low")
        reasoning = data.get("reasoning", "")

        logger.info(
            "Pre-screen result: %d/%d candidates recommended (confidence=%s). %s",
            len(recommended_indices), len(candidates), confidence, reasoning,
        )

        if not recommended_indices:
            return candidates

        filtered = []
        for idx in recommended_indices:
            zero_idx = idx - 1
            if 0 <= zero_idx < len(candidates):
                filtered.append(candidates[zero_idx])

        if max_to_run and len(filtered) > max_to_run:
            filtered = filtered[:max_to_run]

        return filtered if filtered else candidates
