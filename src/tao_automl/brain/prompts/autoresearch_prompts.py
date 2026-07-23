# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt templates for the Autoresearch-style autonomous loop.

Covers: autonomous spec modification, keep/discard reasoning, hybrid strategy,
research programs, pre-screening, and verification.
"""

import json
from typing import Any, Dict, List, Optional


# --- Autoresearch Agent ---

AUTORESEARCH_SYSTEM = """\
You are an autonomous ML research agent for NVIDIA TAO Toolkit.
You operate in a loop: propose spec modifications -> run experiment -> evaluate -> keep/discard.

Your goal is to find the best training configuration for the given network and metric.

CRITICAL RULES:
1. You MUST ONLY use parameter names from the "Tunable Parameters" list provided.
   Do NOT invent parameter names or use alternative key formats.
   For example, if the parameter is "train.optim.lr", do NOT use "train_config.optimizer.lr".
2. Every proposed value MUST respect the type and range for that parameter.
3. Be conservative with expensive changes -- each experiment costs GPU-hours.
4. Prefer small, targeted changes over sweeping modifications.
5. Learn from failures -- if a direction doesn't work, try something different.
6. Explain your reasoning for every decision.
"""


def build_autoresearch_prompt(
    spec_schema: Dict[str, Any],
    current_best_spec: Dict[str, Any],
    experiment_history: List[Dict[str, Any]],
    network: str,
    metric_name: str,
    metric_direction: str,
    research_program: Optional[str] = None,
    external_knowledge: Optional[str] = None,
    parameters: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    """Build prompt for the autoresearch agent to propose next spec modification."""
    history_text = _format_autoresearch_history(experiment_history, metric_name)
    param_schema = _format_parameter_schema(parameters) if parameters else ""

    user_content = f"""## Network: {network}
## Metric: {metric_name} (goal: {metric_direction})
## Experiments completed: {len(experiment_history)}
"""

    if param_schema:
        user_content += f"""
## Tunable Parameters (use ONLY these exact parameter names)
{param_schema}
"""

    user_content += f"""
## Current Best Spec (abbreviated, for context only)
```json
{json.dumps(_abbreviate_spec(current_best_spec), indent=2)}
```

## Experiment History
{history_text}
"""

    if research_program:
        user_content += f"""
## Research Program (user directives)
{research_program}
"""

    if external_knowledge:
        user_content += f"""
## External Knowledge
{external_knowledge}
"""

    user_content += """
## Your Task
Propose a modification to the current best spec. You MUST use ONLY parameter names
from the "Tunable Parameters" list above. Do NOT invent new parameter names.

Return JSON with:

1. **"modifications"**: Dict of parameter names to new values. Keys MUST be from the Tunable Parameters list.
2. **"reasoning"**: Why you're making these changes (2-3 sentences).
3. **"expected_impact"**: "high", "medium", or "low" -- your confidence this will improve the metric.
4. **"exploration_vs_exploitation"**: "explore" if trying something new, "exploit" if refining near best.

Example:
```json
{
  "modifications": {"train.optim.lr": 0.003, "train.optim.weight_decay": 0.0001},
  "reasoning": "Last experiment showed LR 0.001 was too low. Tripling it while reducing weight decay.",
  "expected_impact": "medium",
  "exploration_vs_exploitation": "exploit"
}
```

Return ONLY the JSON object.
"""

    return [
        {"role": "system", "content": AUTORESEARCH_SYSTEM},
        {"role": "user", "content": user_content},
    ]


# --- Keep/Discard Decision ---

def build_keep_discard_prompt(
    current_result: Dict[str, Any],
    best_result: Dict[str, Any],
    modifications_made: Dict[str, Any],
    metric_name: str,
    metric_direction: str,
) -> List[Dict[str, str]]:
    """Build prompt for the keep/discard decision after an experiment."""
    current_metric = current_result.get("metric", "N/A")
    best_metric = best_result.get("metric", "N/A")
    has_previous_best = isinstance(best_metric, (int, float))
    improved = (
        (metric_direction == "maximize" and current_metric > best_metric) or
        (metric_direction == "minimize" and current_metric < best_metric)
    ) if isinstance(current_metric, (int, float)) and has_previous_best else False
    comparison = (
        "improved" if improved else "did NOT improve"
    ) if has_previous_best else "is the first measured result (no prior best)"

    user_content = f"""## Experiment Result
- Metric ({metric_name}): **{current_metric}**
- Previous best: **{best_metric}**
- Direction: {metric_direction}
- Comparison: {comparison}
- Modifications made: {json.dumps(modifications_made)}
- VRAM used: {current_result.get("vram_gb", "unknown")} GB
- Training time: {current_result.get("train_time", "unknown")}

## Decision
Return JSON with:
1. **"decision"**: "keep" or "discard"
2. **"reasoning"**: Why this decision (consider metric, complexity, VRAM, simplicity)
3. **"next_direction"**: What to try next based on this result
"""

    return [
        {"role": "system", "content": AUTORESEARCH_SYSTEM},
        {"role": "user", "content": user_content},
    ]


# --- Hybrid Strategist ---

HYBRID_SYSTEM = """\
You are a strategic ML research planner for NVIDIA TAO Toolkit.
You decide WHAT to explore and WHICH algorithm to use at each phase.
You delegate actual hyperparameter tuning to specialized algorithms
(Bayesian, ASHA, BOHB, etc.) but you control the strategy.
"""


def build_hybrid_strategy_prompt(
    full_history: List[Dict[str, Any]],
    available_parameters: List[Dict[str, Any]],
    available_algorithms: List[str],
    network: str,
    metric_name: str,
    metric_direction: str,
    completed_phases: List[Dict[str, Any]],
    enable_range_narrowing: bool = False,
) -> List[Dict[str, str]]:
    """Build prompt for the hybrid strategist to plan the next phase."""
    history_summary = _summarize_history(full_history, metric_name)
    param_names = [p.get("parameter", "") for p in available_parameters]
    parameter_summary = _format_parameter_schema(available_parameters)
    range_narrowing_instructions = ""
    if enable_range_narrowing:
        range_narrowing_instructions = """
7. **"parameter_overrides"**: Optional result-informed narrowing for the next phase only.
   - Use this only after at least one completed phase has successful results.
   - For numeric parameters, provide `{"valid_min": value, "valid_max": value}`.
     Either bound may be omitted; omitted bounds keep the original limit.
   - For option parameters, provide `{"valid_options": [subset...]}`.
     The subset must contain only values from the listed parameter constraints.
   - Do not expand ranges, invent options, or narrow a parameter that is not
     included in this phase's "parameters" list.
   - Example: `"parameter_overrides": {"train.train_batch_per_replica":
     {"valid_options": [8, 16]}}`.
"""

    user_content = f"""## Network: {network}
## Metric: {metric_name} ({metric_direction})
## Available algorithms: {', '.join(available_algorithms)}
## Available parameters: {', '.join(param_names)}
## Parameter constraints
{parameter_summary}
## Total experiments run: {len(full_history)}

## History Summary
{history_summary}

## Completed Phases
{json.dumps(completed_phases, indent=2) if completed_phases else "None yet -- this is the first phase."}

## Your Task
Plan the next optimization phase. Return JSON with:

1. **"action"**: "sweep" (run algorithm on parameter subset) or
   "single_trial" (test one specific config) or "stop" (search has converged)
2. **"algorithm"**: Which algorithm to use (from available list)
3. **"parameters"**: Which parameters to focus on (subset of available)
   - Keep dependent parameters together. If a parameter lists `depends_on`,
     include both that parameter and the dependency in the same phase.
   - If choosing a parent parameter, include any available child parameters
     that depend on it so the generated train spec remains valid.
   - In the first phase, do not prune core train controls such as epoch,
     batch size, learning rate, weight decay, or warmup when they are tunable.
   - If LoRA parameters are available in the first phase, include rank, alpha,
     and dropout along with the core train controls.
4. **"trials"**: How many experiments to run in this phase
   - Do not spend the full remaining experiment budget in the first phase;
     leave budget for at least one result-informed refinement phase.
5. **"algorithm_params"**: Algorithm-specific settings (e.g., automl_max_recommendations)
6. **"reasoning"**: Why this strategy
   - When experiment history is present, explicitly cite the prior
     `{metric_name}` value used to choose the next algorithm, parameters, or
     ranges. Do not describe a refinement without naming its measured metric.
{range_narrowing_instructions}
"""

    return [
        {"role": "system", "content": HYBRID_SYSTEM},
        {"role": "user", "content": user_content},
    ]


# --- Pre-Screener ---

PRESCREENER_SYSTEM = """\
You are an expert at predicting ML training outcomes for NVIDIA TAO Toolkit models.
Given a set of candidate configurations, you predict their likely performance
WITHOUT actually running training. Use your knowledge of the network architecture,
common hyperparameter ranges, and ML best practices.

Be calibrated -- acknowledge uncertainty. Rank by expected performance.
"""


def build_prescreen_prompt(
    candidates: List[Dict[str, Any]],
    network: str,
    metric_name: str,
    metric_direction: str,
    reference_results: Optional[List[Dict[str, Any]]] = None,
    valid_parameter_names: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Build prompt to pre-screen candidate configs without running them."""
    candidates_text = "\n".join(
        f"Config {i + 1}: {json.dumps(c)}" for i, c in enumerate(candidates)
    )

    user_content = f"""## Network: {network}
## Metric: {metric_name} ({metric_direction})

## Candidate Configurations
{candidates_text}
"""

    if valid_parameter_names:
        user_content += f"""
## Valid Parameter Names (only these keys are actually applied by the system)
{', '.join(valid_parameter_names)}

IMPORTANT: Configs using parameter names NOT in the list above will have those
parameters silently ignored, causing the experiment to run with defaults and likely fail.
Penalize configs that use incorrect or made-up parameter names.
"""

    if reference_results:
        ref_text = "\n".join(
            f"- Config: {json.dumps(r.get('config', {}))} -> {metric_name}: {r.get('metric', 'N/A')}"
            for r in reference_results[:5]
        )
        user_content += f"""
## Reference Results (from actual training)
{ref_text}
"""

    user_content += f"""
## Your Task
Rank all candidates by predicted {metric_name} (best first).
Return JSON with:

1. **"ranked_indices"**: List of candidate indices (1-based) from best to worst predicted.
2. **"predictions"**: Dict mapping candidate index to predicted {metric_name} range [low, high].
3. **"confidence"**: "high", "medium", or "low" overall confidence in the ranking.
4. **"reasoning"**: Brief explanation of the ranking logic.
5. **"recommended_to_run"**: List of candidate indices worth actually running (skip obvious bad ones).
"""

    return [
        {"role": "system", "content": PRESCREENER_SYSTEM},
        {"role": "user", "content": user_content},
    ]


# --- Verification ---

VERIFICATION_SYSTEM = """\
You are a quality assurance agent for NVIDIA TAO Toolkit AutoML experiments.
You validate configurations before they are run and results after they complete.
Flag any issues that could waste GPU resources.
"""


def build_spec_verification_prompt(
    proposed_spec: Dict[str, Any],
    spec_schema_summary: str,
    network: str,
) -> List[Dict[str, str]]:
    """Verify a proposed spec modification before launching a training job."""
    user_content = f"""## Network: {network}

## Proposed Spec Changes
{json.dumps(proposed_spec, indent=2)}

## Schema Constraints
{spec_schema_summary}

## Verification Checklist
Return JSON with:
1. **"valid"**: true/false -- can this spec be run without errors?
2. **"issues"**: List of specific problems found (empty list if valid).
3. **"warnings"**: List of concerns that won't cause errors but may waste GPU time.
4. **"risk_level"**: "safe", "moderate", or "risky" -- likelihood of crash/OOM/waste.
"""

    return [
        {"role": "system", "content": VERIFICATION_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def build_result_verification_prompt(
    result: Dict[str, Any],
    expected_range: Optional[Dict[str, float]],
    metric_name: str,
    network: str,
) -> List[Dict[str, str]]:
    """Verify training results are plausible and not corrupted."""
    user_content = f"""## Network: {network}
## Metric: {metric_name}

## Training Result
{json.dumps(result, indent=2)}

## Expected Range
{json.dumps(expected_range) if expected_range else "Not specified"}

## Verification
Return JSON with:
1. **"plausible"**: true/false -- do these results look legitimate?
2. **"issues"**: List of anomalies detected.
3. **"should_count"**: true/false -- should this result be used for optimization decisions?
"""

    return [
        {"role": "system", "content": VERIFICATION_SYSTEM},
        {"role": "user", "content": user_content},
    ]


# --- Knowledge Retrieval Summary ---

def build_knowledge_summary_prompt(
    raw_knowledge: str,
    network: str,
    metric_name: str,
    task_description: str,
) -> List[Dict[str, str]]:
    """Summarize retrieved external knowledge into actionable insights."""
    user_content = f"""## Task
Summarize the following retrieved knowledge into actionable hyperparameter tuning insights
for training a **{network}** model, optimizing **{metric_name}**.

## Task Description
{task_description}

## Raw Retrieved Knowledge
{raw_knowledge}

## Response Format
Return JSON with:
1. **"insights"**: List of specific, actionable insights for hyperparameter tuning.
2. **"recommended_ranges"**: Dict of parameter name -> suggested range based on the knowledge.
3. **"confidence"**: "high", "medium", or "low" -- how relevant the knowledge is to this task.
"""

    return [
        {"role": "system", "content": (
            "You are an expert at extracting ML training insights "
            "from research papers and benchmark results."
        )},
        {"role": "user", "content": user_content},
    ]


# --- Helpers ---

def _format_parameter_schema(parameters: List[Dict[str, Any]]) -> str:
    """Format parameter list into a schema description for the LLM."""
    if not parameters:
        return "No parameters defined."
    lines = []
    for p in parameters:
        name = p.get("parameter", "unknown")
        dtype = p.get("value_type", "unknown")
        default = p.get("default_value", "")
        v_min = p.get("valid_min", "")
        v_max = p.get("valid_max", "")
        options = p.get("valid_options", "")

        desc = f"- **{name}** (type: {dtype})"
        if v_min != "" and v_max != "":
            desc += f" range: [{v_min}, {v_max}]"
        if options and options != "":
            desc += f" options: {options}"
        if default != "":
            desc += f" default: {default}"
        lines.append(desc)
    return "\n".join(lines)


def _format_autoresearch_history(history: List[Dict[str, Any]], metric_name: str) -> str:
    if not history:
        return "No experiments yet."
    lines = []
    for i, entry in enumerate(history):
        status = entry.get("status", "unknown")
        metric = entry.get("metric", "N/A")
        reasoning = entry.get("reasoning", "")
        mods = entry.get("modifications", {})
        mods_str = ", ".join(f"{k}={v}" for k, v in list(mods.items())[:3])
        lines.append(
            f"[{i + 1}] [{status.upper()}] {metric_name}={metric} | {mods_str}" +
            (f" | {reasoning}" if reasoning else "")
        )
    return "\n".join(lines)


def _abbreviate_spec(spec: Dict[str, Any], max_depth: int = 2) -> Dict[str, Any]:
    """Truncate deeply nested spec for prompt brevity."""
    def _trunc(obj, depth):
        if depth >= max_depth:
            if isinstance(obj, dict):
                return f"{{...{len(obj)} keys}}"
            if isinstance(obj, list) and len(obj) > 3:
                return f"[...{len(obj)} items]"
        if isinstance(obj, dict):
            return {k: _trunc(v, depth + 1) for k, v in obj.items()}
        return obj
    return _trunc(spec, 0)


def _summarize_history(history: List[Dict[str, Any]], metric_name: str) -> str:
    if not history:
        return "No experiments run yet."

    metrics = [h.get("metric") for h in history if isinstance(h.get("metric"), (int, float))]
    if not metrics:
        return f"{len(history)} experiments, no valid metrics recorded."

    successes = sum(1 for h in history if h.get("status") == "success")
    failures = sum(1 for h in history if h.get("status") == "failure")
    return (
        f"Total: {len(history)} experiments ({successes} success, {failures} failure)\n"
        f"Best {metric_name}: {max(metrics)}\n"
        f"Worst {metric_name}: {min(metrics)}\n"
        f"Mean {metric_name}: {sum(metrics) / len(metrics):.4f}"
    )
