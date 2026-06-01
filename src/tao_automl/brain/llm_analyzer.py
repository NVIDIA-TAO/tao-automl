# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM Result Analyzer.

Analyzes AutoML experiment results and provides actionable insights.
Works with ALL existing algorithms (Bayesian, BOHB, PBT, etc.).
"""
import logging
from typing import Any, Dict, List, Optional

from tao_automl.brain.llm_client import LLMClient
from tao_automl.brain.prompts.analyzer_prompts import build_analysis_prompt

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """Analyzes AutoML experiment results using an LLM.

    Provides pattern detection, convergence assessment, failure triage,
    and actionable recommendations.

    When narrow_ranges is enabled, also produces structured range suggestions
    that can be programmatically applied to narrow the search space.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        llm_params: Optional[Dict[str, Any]] = None,
        analysis_interval: int = 3,
        narrow_ranges: bool = False,
    ):
        """Initialize the LLMAnalyzer."""
        self.llm_client = llm_client or LLMClient(params=llm_params)
        self.analysis_interval = analysis_interval
        self.narrow_ranges = narrow_ranges
        self._last_analysis_count = 0
        self._analyses: List[Dict[str, Any]] = []
        self._applied_narrowings: List[Dict[str, Any]] = []

    def should_analyze(self, completed_count: int) -> bool:
        """Check if it's time to run analysis based on the interval."""
        if completed_count <= 0:
            return False
        if completed_count - self._last_analysis_count >= self.analysis_interval:
            return True
        return False

    def analyze(
        self,
        experiments: List[Dict[str, Any]],
        parameters: List[Dict[str, Any]],
        network: str,
        metric_name: str,
        metric_direction: str,
        best_metric: Optional[float] = None,
        analysis_type: str = "periodic",
    ) -> Optional[Dict[str, Any]]:
        """Run LLM analysis on experiment results."""
        messages = build_analysis_prompt(
            experiments=experiments,
            parameters=parameters,
            network=network,
            metric_name=metric_name,
            metric_direction=metric_direction,
            best_metric=best_metric,
            analysis_type=analysis_type,
            include_range_suggestions=self.narrow_ranges,
        )

        response = self.llm_client.chat(messages, json_mode=True, temperature=0.3)

        if not response.ok:
            logger.warning("LLM analysis failed: %s", response.error)
            return None

        analysis = response.json_content
        if analysis is None:
            logger.warning("Could not parse LLM analysis response")
            return None

        self._last_analysis_count = len(experiments)
        self._analyses.append(analysis)

        logger.info(
            "LLM Analysis [%s]: convergence=%s, %d patterns, %d recommendations",
            analysis_type,
            analysis.get("convergence_assessment", "unknown"),
            len(analysis.get("patterns", [])),
            len(analysis.get("recommendations", [])),
        )

        summary = analysis.get("summary", "")
        if summary:
            logger.info("LLM Analysis Summary: %s", summary)

        if self.narrow_ranges and analysis.get("suggested_ranges"):
            logger.info(
                "LLM suggested range narrowing for %d parameters",
                len(analysis["suggested_ranges"]),
            )

        return analysis

    def get_validated_range_narrowings(
        self, parameters: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Extract and validate range narrowing suggestions from the latest analysis.

        Ensures suggested ranges are strictly within the original schema bounds
        and applies a 10% safety margin to avoid over-aggressive narrowing.
        """
        latest = self.get_latest_analysis()
        if not latest:
            return {}

        suggested = latest.get("suggested_ranges", {})
        if not suggested or not isinstance(suggested, dict):
            return {}

        param_lookup = {p.get("parameter", ""): p for p in parameters}
        validated = {}

        for param_name, suggestion in suggested.items():
            if param_name not in param_lookup:
                logger.warning(
                    "Range narrowing: ignoring unknown parameter '%s'", param_name
                )
                continue

            schema = param_lookup[param_name]
            dtype = schema.get("value_type", "")
            if dtype not in ("float", "int", "integer"):
                logger.warning(
                    "Range narrowing: skipping non-numeric parameter '%s' (type=%s)",
                    param_name, dtype,
                )
                continue

            if not isinstance(suggestion, dict):
                continue

            try:
                new_min = float(suggestion.get("min", ""))
                new_max = float(suggestion.get("max", ""))
            except (ValueError, TypeError):
                logger.warning(
                    "Range narrowing: invalid min/max for '%s': %s", param_name, suggestion
                )
                continue

            if new_min >= new_max:
                logger.warning(
                    "Range narrowing: min >= max for '%s' (%.6g >= %.6g), skipping",
                    param_name, new_min, new_max,
                )
                continue

            orig_min = schema.get("valid_min", "")
            orig_max = schema.get("valid_max", "")
            if orig_min == "" or orig_max == "":
                logger.warning(
                    "Range narrowing: no original bounds for '%s', skipping", param_name
                )
                continue

            try:
                orig_min = float(orig_min)
                orig_max = float(orig_max)
            except (ValueError, TypeError):
                continue

            new_min = max(new_min, orig_min)
            new_max = min(new_max, orig_max)

            if new_min >= new_max:
                continue

            span = new_max - new_min
            margin = span * 0.1
            safe_min = max(new_min - margin, orig_min)
            safe_max = min(new_max + margin, orig_max)

            if dtype in ("int", "integer"):
                safe_min = int(safe_min)
                safe_max = int(safe_max)
                if safe_min >= safe_max:
                    continue

            reason = suggestion.get("reason", "LLM analysis")
            validated[param_name] = {
                "valid_min": safe_min,
                "valid_max": safe_max,
            }

            logger.info(
                "Range narrowing: %s [%.6g, %.6g] -> [%.6g, %.6g] (reason: %s)",
                param_name, orig_min, orig_max, safe_min, safe_max, reason,
            )

        if validated:
            self._applied_narrowings.append({
                "analysis_index": len(self._analyses) - 1,
                "narrowings": validated,
            })

        return validated

    def get_latest_analysis(self) -> Optional[Dict[str, Any]]:
        """Return the most recent analysis."""
        return self._analyses[-1] if self._analyses else None

    def get_all_analyses(self) -> List[Dict[str, Any]]:
        """Return all analyses."""
        return list(self._analyses)

    def format_for_metadata(self) -> Dict[str, Any]:
        """Format analyses for storage in job metadata."""
        latest = self.get_latest_analysis()
        return {
            "total_analyses": len(self._analyses),
            "latest_analysis": latest,
            "convergence_history": [
                a.get("convergence_assessment", "unknown") for a in self._analyses
            ],
            "applied_narrowings": self._applied_narrowings,
            "llm_usage": self.llm_client.get_usage_summary(),
        }
