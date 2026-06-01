# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Research Program System.

Defines and executes multi-phase research campaigns with different algorithms
per phase and results carrying forward between phases.
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ResearchPhase:
    """A single phase in a research program."""

    name: str
    algorithm: str = "bayesian"
    parameters: List[str] = field(default_factory=list)
    trials: int = 5
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    carry_forward: str = "best"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize phase to dict."""
        return {
            "name": self.name,
            "algorithm": self.algorithm,
            "parameters": self.parameters,
            "trials": self.trials,
            "algorithm_params": self.algorithm_params,
            "constraints": self.constraints,
            "carry_forward": self.carry_forward,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchPhase":
        """Deserialize phase from dict."""
        return cls(
            name=data.get("name", "unnamed"),
            algorithm=data.get("algorithm", "bayesian"),
            parameters=data.get("parameters", []),
            trials=data.get("trials", 5),
            algorithm_params=data.get("algorithm_params", {}),
            constraints=data.get("constraints", {}),
            carry_forward=data.get("carry_forward", "best"),
        )


@dataclass
class ResearchProgram:
    """A multi-phase research campaign definition."""

    objective: str = ""
    network: str = ""
    phases: List[ResearchPhase] = field(default_factory=list)
    global_constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize program to dict."""
        return {
            "objective": self.objective,
            "network": self.network,
            "phases": [p.to_dict() for p in self.phases],
            "global_constraints": self.global_constraints,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchProgram":
        """Deserialize program from dict."""
        return cls(
            objective=data.get("objective", ""),
            network=data.get("network", ""),
            phases=[ResearchPhase.from_dict(p) for p in data.get("phases", [])],
            global_constraints=data.get("global_constraints", {}),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ResearchProgram":
        """Deserialize program from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def validate(self, available_parameters: List[str], available_algorithms: List[str]) -> List[str]:
        """Validate the program against available parameters and algorithms."""
        issues = []
        if not self.phases:
            issues.append("Program has no phases defined")

        for i, phase in enumerate(self.phases):
            if phase.algorithm not in available_algorithms:
                issues.append(
                    f"Phase {i + 1} ({phase.name}): unknown algorithm '{phase.algorithm}'"
                )
            invalid_params = [p for p in phase.parameters if p not in available_parameters]
            if invalid_params:
                issues.append(
                    f"Phase {i + 1} ({phase.name}): unknown parameters {invalid_params}"
                )
            if phase.trials < 1:
                issues.append(f"Phase {i + 1} ({phase.name}): trials must be >= 1")

        return issues


class ResearchProgramExecutor:
    """Executes a research program phase by phase."""

    def __init__(self, program: ResearchProgram):
        """Initialize the ResearchProgramExecutor."""
        self.program = program
        self.current_phase_index = 0
        self.phase_results: List[Dict[str, Any]] = []
        self.carry_forward_config: Dict[str, Any] = {}

    def get_current_phase(self) -> Optional[ResearchPhase]:
        """Get the current phase to execute."""
        if self.current_phase_index >= len(self.program.phases):
            return None
        return self.program.phases[self.current_phase_index]

    def complete_phase(
        self,
        results: List[Dict[str, Any]],
        best_config: Optional[Dict[str, Any]] = None,
        best_metric: Optional[float] = None,
    ):
        """Record phase completion and advance to the next phase."""
        phase = self.get_current_phase()
        if phase is None:
            return

        phase_record = {
            "phase_name": phase.name,
            "phase_index": self.current_phase_index,
            "algorithm": phase.algorithm,
            "parameters": phase.parameters,
            "num_experiments": len(results),
            "best_config": best_config,
            "best_metric": best_metric,
        }
        self.phase_results.append(phase_record)

        if best_config and phase.carry_forward == "best":
            for param in phase.parameters:
                if param in best_config:
                    self.carry_forward_config[param] = best_config[param]

        logger.info(
            "Phase '%s' complete: %d experiments, best_metric=%s",
            phase.name, len(results), best_metric,
        )

        self.current_phase_index += 1

    def is_complete(self) -> bool:
        """Check if all phases have been executed."""
        return self.current_phase_index >= len(self.program.phases)

    def get_phase_parameters(self, all_parameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter parameters to those relevant to the current phase."""
        phase = self.get_current_phase()
        if phase is None:
            return all_parameters
        if not phase.parameters:
            return all_parameters
        return [p for p in all_parameters if p["parameter"] in phase.parameters]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the research program execution."""
        return {
            "program": self.program.to_dict(),
            "current_phase": self.current_phase_index,
            "total_phases": len(self.program.phases),
            "is_complete": self.is_complete(),
            "phase_results": self.phase_results,
            "carry_forward_config": self.carry_forward_config,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize executor state."""
        return {
            "program": self.program.to_dict(),
            "current_phase_index": self.current_phase_index,
            "phase_results": self.phase_results,
            "carry_forward_config": self.carry_forward_config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchProgramExecutor":
        """Deserialize executor state."""
        program = ResearchProgram.from_dict(data.get("program", {}))
        executor = cls(program)
        executor.current_phase_index = data.get("current_phase_index", 0)
        executor.phase_results = data.get("phase_results", [])
        executor.carry_forward_config = data.get("carry_forward_config", {})
        return executor
