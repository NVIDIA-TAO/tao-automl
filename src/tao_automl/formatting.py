# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standard renderer for ``AutoMLRunner.run()`` result dicts.

Every generated runner used to hand-roll a summary against a guessed schema;
this is the one blessed renderer for the stable schema documented in
``AutoMLRunner.run``.
"""


def _fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _comparison_line(label, comparison):
    if not isinstance(comparison, dict):
        return f"{label}: n/a"
    arrow = "improved" if comparison.get("improved") else "regressed"
    return (
        f"{label}: {arrow} by {_fmt(comparison.get('delta'))} "
        f"({comparison.get('direction', '?')})"
    )


def format_result(result: dict) -> str:
    """Render an ``AutoMLRunner.run()`` result dict as a readable summary.

    Covers: best recommendation with its parameters, the per-recommendation
    metric/status table with failure reasons, the baseline comparison, and
    the final-evaluation status/metric/record path.
    """
    best = result.get("best") or {}
    progress = result.get("progress") or {}
    baseline = result.get("baseline") or {}
    final_evaluation = result.get("final_evaluation") or {}
    history = result.get("history") or []
    metric_name = (
        final_evaluation.get("metric_name")
        or baseline.get("metric_name")
        or "metric"
    )

    lines = ["AutoML result"]
    algorithm = progress.get("algorithm")
    completed = progress.get("completed")
    total = progress.get("total")
    if algorithm or completed is not None:
        lines.append(
            f"  algorithm={_fmt(algorithm)} "
            f"completed={_fmt(completed)}/{_fmt(total)}"
        )

    lines.append("")
    lines.append(
        f"Best: rec {_fmt(best.get('rec_id'))} — "
        f"{metric_name}={_fmt(best.get('metric_value'))}"
    )
    for key in sorted(best.get("specs") or {}):
        lines.append(f"  {key} = {_fmt((best.get('specs') or {}).get(key))}")

    if history:
        lines.append("")
        lines.append("History:")
        lines.append(f"  {'rec':>4}  {'status':<10} {metric_name}")
        for entry in history:
            row = (
                f"  {_fmt(entry.get('rec_id')):>4}  "
                f"{_fmt(entry.get('status')):<10} "
                f"{_fmt(entry.get('metric'))}"
            )
            reason = entry.get("failure_reason")
            if reason:
                row += f"  ({reason})"
            lines.append(row)

    lines.append("")
    if baseline.get("enabled") and baseline.get("metric_value") is not None:
        lines.append(
            f"Baseline: {metric_name}={_fmt(baseline.get('metric_value'))}"
        )
        lines.append(
            "  " + _comparison_line(
                "best vs baseline", baseline.get("comparison_to_best")
            )
        )
    else:
        lines.append(f"Baseline: {_fmt(baseline.get('status') or 'not_run')}")

    status = final_evaluation.get("status", "not_run")
    lines.append(f"Final evaluation: {status}")
    if status == "callback_error":
        lines.append(
            f"  final_eval_fn raised: {_fmt(final_evaluation.get('failure_reason'))}"
        )
    elif final_evaluation.get("failure_reason"):
        lines.append(f"  reason: {_fmt(final_evaluation.get('failure_reason'))}")
    if final_evaluation.get("metric_value") is not None:
        lines.append(
            f"  {metric_name}={_fmt(final_evaluation.get('metric_value'))}"
        )
        lines.append(
            "  " + _comparison_line(
                "vs baseline", final_evaluation.get("comparison_to_baseline")
            )
        )
    if final_evaluation.get("record_path"):
        lines.append(f"  record: {final_evaluation.get('record_path')}")

    if "pareto_front" in result:
        lines.append("")
        lines.append(
            f"Pareto front: {len(result.get('pareto_front') or [])} recommendation(s)"
        )

    return "\n".join(lines)
