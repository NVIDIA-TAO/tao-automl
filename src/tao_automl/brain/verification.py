# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-stage Verification (AutoML-Agent concept).

Validates configurations before launching GPU jobs and verifies results
after completion. Prevents wasting GPU-hours on invalid/risky configs.
"""
import logging
import math
from typing import Any, Dict, List, Optional

from tao_automl.brain.llm_client import LLMClient
from tao_automl.brain.prompts.autoresearch_prompts import (
    build_spec_verification_prompt,
    build_result_verification_prompt,
)

logger = logging.getLogger(__name__)


class SpecVerificationResult:
    """Result of spec verification."""

    def __init__(self, valid: bool, issues: List[str], warnings: List[str], risk_level: str):
        """Initialize SpecVerificationResult."""
        self.valid = valid
        self.issues = issues
        self.warnings = warnings
        self.risk_level = risk_level

    @property
    def ok(self) -> bool:
        """Return True if valid and not risky."""
        return self.valid and self.risk_level != "risky"

    def __repr__(self):
        """Return string representation."""
        return f"SpecVerification(valid={self.valid}, risk={self.risk_level}, issues={len(self.issues)})"


class ResultVerificationResult:
    """Result of training result verification."""

    def __init__(self, plausible: bool, issues: List[str], should_count: bool):
        """Initialize ResultVerificationResult."""
        self.plausible = plausible
        self.issues = issues
        self.should_count = should_count

    def __repr__(self):
        """Return string representation."""
        return f"ResultVerification(plausible={self.plausible}, count={self.should_count})"


class MultiStageVerifier:
    """Verifies specs before launch and results after completion.

    Stage 1 (Pre-launch): Validates proposed spec changes against schema and
    domain knowledge. Flags configs that will likely crash, OOM, or waste time.

    Stage 2 (Post-result): Verifies training results are plausible and not
    corrupted. Flags anomalies (NaN metrics, implausible accuracy, etc.).
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        llm_params: Optional[Dict[str, Any]] = None,
        enable_llm_verification: bool = True,
    ):
        """Initialize the MultiStageVerifier."""
        self.llm_client = llm_client or LLMClient(params=llm_params)
        self.enable_llm = enable_llm_verification

    def verify_spec(
        self,
        proposed_changes: Dict[str, Any],
        spec_schema_summary: str,
        network: str,
    ) -> SpecVerificationResult:
        """Verify a proposed spec modification before launching a training job."""
        issues = []
        warnings = []

        issues.extend(self._rule_based_spec_check(proposed_changes))

        if issues:
            return SpecVerificationResult(
                valid=False, issues=issues, warnings=warnings, risk_level="risky"
            )

        if self.enable_llm:
            llm_result = self._llm_verify_spec(proposed_changes, spec_schema_summary, network)
            if llm_result:
                if not llm_result.get("valid", True):
                    issues.extend(llm_result.get("issues", []))
                warnings.extend(llm_result.get("warnings", []))
                risk = llm_result.get("risk_level", "safe")
                return SpecVerificationResult(
                    valid=len(issues) == 0,
                    issues=issues,
                    warnings=warnings,
                    risk_level=risk,
                )

        return SpecVerificationResult(
            valid=True, issues=[], warnings=warnings, risk_level="safe"
        )

    def verify_result(
        self,
        result: Dict[str, Any],
        expected_range: Optional[Dict[str, float]],
        metric_name: str,
        network: str,
    ) -> ResultVerificationResult:
        """Verify training results are plausible."""
        issues = []

        metric_value = result.get("metric")
        if metric_value is not None:
            if isinstance(metric_value, float) and (math.isnan(metric_value) or math.isinf(metric_value)):
                issues.append(f"Metric value is {metric_value} (NaN or Inf)")
                return ResultVerificationResult(plausible=False, issues=issues, should_count=False)

        status = result.get("status", "")
        if status == "failure":
            issues.append("Experiment failed")
            return ResultVerificationResult(plausible=True, issues=issues, should_count=True)

        if self.enable_llm and not issues:
            llm_result = self._llm_verify_result(result, expected_range, metric_name, network)
            if llm_result:
                plausible = llm_result.get("plausible", True)
                should_count = llm_result.get("should_count", True)
                issues.extend(llm_result.get("issues", []))
                return ResultVerificationResult(
                    plausible=plausible, issues=issues, should_count=should_count
                )

        return ResultVerificationResult(plausible=True, issues=[], should_count=True)

    def _rule_based_spec_check(self, changes: Dict[str, Any]) -> List[str]:
        """Fast rule-based validation of spec changes."""
        issues = []
        for key, value in changes.items():
            if value is None:
                issues.append(f"Parameter '{key}' has None value")
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    issues.append(f"Parameter '{key}' has invalid value: {value}")
            if isinstance(value, (int, float)) and value < 0:
                if "lr" in key.lower() or "batch" in key.lower() or "epoch" in key.lower():
                    issues.append(f"Parameter '{key}' has negative value: {value}")
        return issues

    def _llm_verify_spec(
        self, changes: Dict[str, Any], schema_summary: str, network: str
    ) -> Optional[Dict[str, Any]]:
        """Use LLM for deeper spec verification."""
        messages = build_spec_verification_prompt(changes, schema_summary, network)
        response = self.llm_client.chat(messages, json_mode=True, temperature=0.1)
        if response.ok and response.json_content:
            return response.json_content
        return None

    def _llm_verify_result(
        self,
        result: Dict[str, Any],
        expected_range: Optional[Dict[str, float]],
        metric_name: str,
        network: str,
    ) -> Optional[Dict[str, Any]]:
        """Use LLM for deeper result verification."""
        messages = build_result_verification_prompt(result, expected_range, metric_name, network)
        response = self.llm_client.chat(messages, json_mode=True, temperature=0.1)
        if response.ok and response.json_content:
            return response.json_content
        return None
