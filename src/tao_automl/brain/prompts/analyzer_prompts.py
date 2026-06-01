# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt templates for the LLM Result Analyzer."""

import json
from typing import Any, Dict, List, Optional


ANALYZER_SYSTEM = """\
You are an expert ML experiment analyst for NVIDIA TAO Toolkit AutoML runs.
Your job is to analyze hyperparameter optimization results and provide actionable insights.

You can:
- Detect patterns in successful vs failed experiments
- Identify parameter sensitivities and correlations
- Predict whether the search is plateauing
- Suggest search space adjustments
- Triage failures (OOM, divergence, poor convergence)

Be concise and specific. Every insight should be actionable.
"""


def build_analysis_prompt(
    experiments: List[Dict[str, Any]],
    parameters: List[Dict[str, Any]],
    network: str,
    metric_name: str,
    metric_direction: str,
    best_metric: Optional[float] = None,
    analysis_type: str = "periodic",
    include_range_suggestions: bool = False,
) -> List[Dict[str, str]]:
    """Build analysis prompt for experiment results."""
    exp_table = _format_experiments(experiments, metric_name)
    param_schema = _format_parameter_schema(parameters)

    user_content = f"""## AutoML Run Analysis
- Network: **{network}**
- Metric: **{metric_name}** (goal: {metric_direction})
- Experiments completed: **{len(experiments)}**
- Best {metric_name}: **{best_metric}**

## Parameter Schema (current search ranges)
{param_schema}

## Experiment Results
{exp_table}

## Analysis Request ({analysis_type})
Provide a JSON response with these keys:

1. **"patterns"**: List of observed patterns (e.g., "LR > 0.01 always causes divergence").
2. **"parameter_sensitivity"**: Dict mapping parameter names to sensitivity levels ("high", "medium", "low", "unknown").
3. **"convergence_assessment"**: One of "improving", "plateauing", "diverging", or "insufficient_data".
4. **"failure_analysis"**: If any experiments failed, explain likely causes.
5. **"recommendations"**: List of specific, actionable suggestions for the next experiments.
6. **"summary"**: 2-3 sentence human-readable summary of the AutoML run state."""

    if include_range_suggestions:
        user_content += """
7. **"suggested_ranges"**: Dict mapping parameter names to narrowed search ranges
   based on your analysis. Only include parameters where you have enough evidence
   to confidently narrow the range. Each entry must have:
   - **"min"**: Suggested new minimum (number). Must be >= the current schema minimum.
   - **"max"**: Suggested new maximum (number). Must be <= the current schema maximum.
   - **"reason"**: One sentence explaining why this narrowing is justified.

   Only suggest ranges for float and int parameters. Only narrow -- never widen
   beyond the current schema bounds. If insufficient data, omit this key or return
   an empty dict.

   Example:
   ```json
   "suggested_ranges": {
     "train.optim.lr": {"min": 0.0005, "max": 0.005, "reason": "All top-3 results had LR in [0.0008, 0.004]"},
     "train.optim.weight_decay": {
       "min": 0.0001, "max": 0.001,
       "reason": "Weight decay > 0.001 consistently degraded mAP"
     }
   }
   ```"""

    user_content += "\n\nReturn ONLY the JSON object."

    return [
        {"role": "system", "content": ANALYZER_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def _format_parameter_schema(parameters: List[Dict[str, Any]]) -> str:
    """Format parameter definitions with current ranges for the prompt."""
    if not parameters:
        return "No parameter definitions available."
    lines = []
    for p in parameters:
        name = p.get("parameter", "")
        dtype = p.get("value_type", "")
        v_min = p.get("valid_min", "")
        v_max = p.get("valid_max", "")
        default = p.get("default_value", "")
        if dtype in ("float", "int", "integer"):
            lines.append(f"- **{name}** ({dtype}): range [{v_min}, {v_max}], default={default}")
        elif dtype in ("categorical", "ordered", "ordered_int"):
            options = p.get("valid_options", "")
            lines.append(f"- **{name}** ({dtype}): options={options}, default={default}")
        else:
            lines.append(f"- **{name}** ({dtype}): default={default}")
    return "\n".join(lines)


def _format_experiments(experiments: List[Dict[str, Any]], metric_name: str) -> str:
    if not experiments:
        return "No experiments completed yet."

    lines = [f"| # | {metric_name} | status | config |", "|---|---|---|---|"]
    for i, exp in enumerate(experiments):
        metric = exp.get("metric", "N/A")
        status = exp.get("status", "unknown")
        config = exp.get("config", {})
        config_str = json.dumps(config) if len(json.dumps(config)) < 120 else (
            json.dumps(dict(list(config.items())[:4])) + " ..."
        )
        lines.append(f"| {i + 1} | {metric} | {status} | {config_str} |")
    return "\n".join(lines)
