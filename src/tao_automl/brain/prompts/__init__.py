# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt templates for LLM-powered AutoML capabilities."""

from tao_automl.brain.prompts.llm_brain_prompts import (
    build_recommendation_prompt,
    build_recommendation_with_reasoning_prompt,
)
from tao_automl.brain.prompts.analyzer_prompts import (
    build_analysis_prompt,
)
from tao_automl.brain.prompts.nl_config_prompts import (
    build_nl_config_prompt,
)
from tao_automl.brain.prompts.autoresearch_prompts import (
    build_autoresearch_prompt,
    build_keep_discard_prompt,
    build_hybrid_strategy_prompt,
    build_prescreen_prompt,
    build_spec_verification_prompt,
    build_result_verification_prompt,
    build_knowledge_summary_prompt,
)

__all__ = [
    "build_recommendation_prompt",
    "build_recommendation_with_reasoning_prompt",
    "build_analysis_prompt",
    "build_nl_config_prompt",
    "build_autoresearch_prompt",
    "build_keep_discard_prompt",
    "build_hybrid_strategy_prompt",
    "build_prescreen_prompt",
    "build_spec_verification_prompt",
    "build_result_verification_prompt",
    "build_knowledge_summary_prompt",
]
