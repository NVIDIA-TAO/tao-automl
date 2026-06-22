# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt templates for the LLMBrain algorithm."""

import json
from typing import Any, Dict, List, Optional


SYSTEM_PROMPT = """\
You are an expert hyperparameter optimization agent for NVIDIA TAO Toolkit.
Your job is to propose the next hyperparameter configuration for a training experiment.

You have deep knowledge of computer vision model training, including:
- Learning rate schedules, optimizers, and their interactions with batch size
- Backbone architectures and their characteristics
- Augmentation strategies for different tasks (detection, segmentation, classification)
- Regularization techniques and when they help
- Network-specific tuning patterns for DINO, SegFormer, classification_pyt, etc.

RULES:
1. Return ONLY valid JSON matching the parameter schema provided.
2. Every value MUST be within the valid_min/valid_max range or valid_options for that parameter.
3. Reason about parameter correlations (e.g., LR should scale with batch size).
4. Learn from experiment history -- avoid configs similar to failed experiments.
5. Balance exploration (try new regions) with exploitation (refine near best config).
6. If a parameter has depends_on constraints, respect them.
"""


def build_recommendation_prompt(
    parameters: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
    best_config: Optional[Dict[str, Any]],
    best_metric: Optional[float],
    network: str,
    metric_name: str = "kpi",
    metric_direction: str = "maximize",
    external_knowledge: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build the chat messages for generating a hyperparameter recommendation."""
    param_schema = _format_parameters(parameters)
    history_table = _format_history(history, metric_name)

    history_fallback = (
        "No experiments completed yet. Propose a strong initial "
        "configuration based on your knowledge of the "
        "{network} network."
    )

    user_content = f"""## Task
Propose the next hyperparameter configuration for training a **{network}** model.
The goal is to **{metric_direction}** the metric **{metric_name}**.

## Parameter Schema
Each parameter has a name, type, valid range/options, and constraints.
You MUST return values within these bounds.

{param_schema}

## Experiment History ({len(history)} experiments completed)
{history_table if history_table else history_fallback}
"""

    if best_config and best_metric is not None:
        user_content += f"""
## Current Best
- Metric ({metric_name}): {best_metric}
- Config: {json.dumps(best_config, indent=2)}
"""

    if external_knowledge:
        user_content += f"""
## External Knowledge (from recent papers/benchmarks)
{external_knowledge}
"""

    user_content += """
## Response Format
Return a single JSON object where keys are parameter names and values are the proposed values.
Example: {"train.optim.lr": 0.001, "train.optim.weight_decay": 0.0005}

Return ONLY the JSON object, no explanation.
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_recommendation_with_reasoning_prompt(
    parameters: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
    best_config: Optional[Dict[str, Any]],
    best_metric: Optional[float],
    network: str,
    metric_name: str = "kpi",
    metric_direction: str = "maximize",
) -> List[Dict[str, str]]:
    """Same as build_recommendation_prompt but asks for reasoning alongside the config."""
    base_messages = build_recommendation_prompt(
        parameters, history, best_config, best_metric,
        network, metric_name, metric_direction,
    )
    base_messages[-1]["content"] = base_messages[-1]["content"].replace(
        "Return ONLY the JSON object, no explanation.",
        """Return a JSON object with two keys:
- "reasoning": A brief explanation of why you chose these values (2-3 sentences).
  If experiment history is present, explicitly cite the prior metric value(s)
  or current best metric you used to decide the change.
- "config": The proposed hyperparameter configuration.

Example:
{
  "reasoning": "The best prior val_loss was 0.73, so I am increasing LR because recent lower-LR trials converged too slowly...",
  "config": {"train.optim.lr": 0.003, "train.optim.weight_decay": 0.0001}
}"""
    )
    return base_messages


def _format_parameters(parameters: List[Dict[str, Any]]) -> str:
    lines = []
    for p in parameters:
        name = p.get("parameter", "unknown")
        dtype = p.get("value_type", "unknown")
        default = p.get("default_value", "")
        v_min = p.get("valid_min", "")
        v_max = p.get("valid_max", "")
        options = p.get("valid_options", "")
        math_cond = p.get("math_cond", "")
        depends_on = p.get("depends_on", "")

        desc = f"- **{name}** (type: {dtype})"
        if v_min != "" and v_max != "":
            desc += f" range: [{v_min}, {v_max}]"
        if options and options != "":
            desc += f" options: {options}"
        if default != "":
            desc += f" default: {default}"
        if math_cond and math_cond != "":
            desc += f" constraint: {math_cond}"
        if depends_on and depends_on != "":
            desc += f" depends_on: {depends_on}"
        lines.append(desc)
    return "\n".join(lines)


def _format_history(history: List[Dict[str, Any]], metric_name: str) -> str:
    if not history:
        return ""
    lines = [f"| # | {metric_name} | status | config summary |", "|---|---|---|---|"]
    for i, entry in enumerate(history):
        metric = entry.get("metric", "N/A")
        status = entry.get("status", "unknown")
        config = entry.get("config", {})
        config_str = ", ".join(f"{k}={v}" for k, v in list(config.items())[:5])
        if len(config) > 5:
            config_str += f", ... (+{len(config) - 5} more)"
        lines.append(f"| {i + 1} | {metric} | {status} | {config_str} |")
    return "\n".join(lines)
