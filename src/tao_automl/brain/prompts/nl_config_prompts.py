# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt templates for Natural Language AutoML Configuration."""

from typing import Any, Dict, List, Optional


NL_CONFIG_SYSTEM = """\
You are an expert NVIDIA TAO Toolkit AutoML configuration assistant.
Users describe their optimization goals in natural language, and you translate
them into precise AutoML configurations.

You know all TAO AutoML algorithms and when to use each:
- **bayesian**: Best for small budgets (5-20 experiments), sequential, uses Gaussian Process.
- **bohb**: Good balance of speed and quality, combines Bayesian + Hyperband.
- **asha**: Best for large parallel runs, asynchronous successive halving.
- **hyperband**: Good for medium budgets, multi-fidelity with early stopping.
- **dehb**: Combines evolutionary search with multi-fidelity, good for complex spaces.
- **pbt**: Population-based training, good for long runs with checkpointing.
- **llm**: LLM-powered search, best when domain knowledge matters more than statistical rigor.
- **autoresearch**: Fully autonomous LLM agent loop with keep/discard reasoning.

You must return valid JSON that can be used directly as AutoML settings.
"""


def build_nl_config_prompt(
    user_prompt: str,
    network: str,
    available_parameters: List[Dict[str, Any]],
    dataset_info: Optional[str] = None,
    hardware_info: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build prompt to translate natural language goals into AutoML config."""
    param_summary = _format_available_params(available_parameters)

    user_content = f"""## User's Request
"{user_prompt}"

## Network
{network}

## Available Parameters for Tuning
{param_summary}
"""

    if dataset_info:
        user_content += f"""
## Dataset Info
{dataset_info}
"""

    if hardware_info:
        user_content += f"""
## Hardware Constraints
{hardware_info}
"""

    user_content += """
## Response Format
Return a JSON object with these keys:

1. **"automl_algorithm"**: Which algorithm to use
   (one of: bayesian, bohb, asha, hyperband, dehb, pbt, llm, autoresearch).
2. **"automl_hyperparameters"**: List of parameter names to tune
   (must be from the available parameters above).
3. **"algorithm_specific_params"**: Dict of algorithm-specific settings
   (e.g., automl_max_recommendations, automl_max_epochs).
4. **"automl_range_override"**: List of range override dicts, each with
   "parameter", "valid_min", "valid_max" (optional, only if you want to narrow ranges).
5. **"metric"**: Which metric to optimize (e.g., "kpi", "loss").
6. **"reasoning"**: Brief explanation of why you chose this configuration.

Return ONLY the JSON object.
"""

    return [
        {"role": "system", "content": NL_CONFIG_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def _format_available_params(parameters: List[Dict[str, Any]]) -> str:
    lines = []
    for p in parameters:
        name = p.get("parameter", "unknown")
        dtype = p.get("value_type", "unknown")
        default = p.get("default_value", "")
        v_min = p.get("valid_min", "")
        v_max = p.get("valid_max", "")
        options = p.get("valid_options", "")
        desc = f"- {name} ({dtype})"
        if v_min != "" and v_max != "":
            desc += f" [{v_min} .. {v_max}]"
        if options:
            desc += f" options={options}"
        if default != "":
            desc += f" (default={default})"
        lines.append(desc)
    return "\n".join(lines) if lines else "No parameters available."
