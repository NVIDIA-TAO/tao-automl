# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experiment Tracker with keep/discard reasoning.

Tracks all experiments with autoresearch-style keep/discard decisions,
reasoning, and full history.
"""
import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExperimentEntry:
    """A single experiment entry in the tracker."""

    def __init__(
        self,
        experiment_id: int,
        spec: Dict[str, Any],
        modifications: Dict[str, Any],
        metric: Optional[float] = None,
        status: str = "pending",
        reasoning: str = "",
        decision: str = "",
        vram_gb: Optional[float] = None,
        train_time_seconds: Optional[float] = None,
        job_id: Optional[str] = None,
        feedback: Any = None,
    ):
        """Initialize an ExperimentEntry."""
        self.experiment_id = experiment_id
        self.spec = spec
        self.modifications = modifications
        self.metric = metric
        self.status = status
        self.reasoning = reasoning
        self.decision = decision
        self.vram_gb = vram_gb
        self.train_time_seconds = train_time_seconds
        self.job_id = job_id
        self.feedback = feedback
        self.timestamp = datetime.now(tz=timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entry to dict."""
        return {
            "experiment_id": self.experiment_id,
            "spec": self.spec,
            "modifications": self.modifications,
            "metric": self.metric,
            "status": self.status,
            "reasoning": self.reasoning,
            "decision": self.decision,
            "vram_gb": self.vram_gb,
            "train_time_seconds": self.train_time_seconds,
            "job_id": self.job_id,
            "feedback": self.feedback,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentEntry":
        """Deserialize entry from dict."""
        entry = cls(
            experiment_id=data.get("experiment_id", 0),
            spec=data.get("spec", {}),
            modifications=data.get("modifications", {}),
            metric=data.get("metric"),
            status=data.get("status", "pending"),
            reasoning=data.get("reasoning", ""),
            decision=data.get("decision", ""),
            vram_gb=data.get("vram_gb"),
            train_time_seconds=data.get("train_time_seconds"),
            job_id=data.get("job_id"),
            feedback=data.get("feedback"),
        )
        entry.timestamp = data.get("timestamp", entry.timestamp)
        return entry


class ExperimentTracker:
    """Tracks experiments with autoresearch-style keep/discard history."""

    def __init__(self, metric_direction: str = "maximize"):
        """Initialize the ExperimentTracker."""
        self.history: List[ExperimentEntry] = []
        self.best_spec: Optional[Dict[str, Any]] = None
        self.best_metric: Optional[float] = None
        self.best_experiment_id: Optional[int] = None
        self.metric_direction = metric_direction
        self._next_id = 0

    def set_baseline(self, spec: Dict[str, Any], metric: float):
        """Set the initial baseline spec and metric."""
        entry = ExperimentEntry(
            experiment_id=self._next_id,
            spec=deepcopy(spec),
            modifications={},
            metric=metric,
            status="success",
            reasoning="Baseline experiment",
            decision="keep",
        )
        self._next_id += 1
        self.history.append(entry)
        self.best_spec = deepcopy(spec)
        self.best_metric = metric
        self.best_experiment_id = entry.experiment_id
        logger.info("Baseline set: metric=%.4f", metric)

    def record_experiment(
        self,
        spec: Dict[str, Any],
        modifications: Dict[str, Any],
        metric: Optional[float],
        status: str,
        reasoning: str = "",
        vram_gb: Optional[float] = None,
        train_time_seconds: Optional[float] = None,
        job_id: Optional[str] = None,
        feedback: Any = None,
    ) -> ExperimentEntry:
        """Record a completed experiment."""
        entry = ExperimentEntry(
            experiment_id=self._next_id,
            spec=deepcopy(spec),
            modifications=modifications,
            metric=metric,
            status=status,
            reasoning=reasoning,
            vram_gb=vram_gb,
            train_time_seconds=train_time_seconds,
            job_id=job_id,
            feedback=feedback,
        )
        self._next_id += 1

        if status == "success" and metric is not None:
            is_better = self._is_improvement(metric)
            entry.decision = "keep" if is_better else "discard"
            if is_better:
                previous_best = self.best_metric
                self.best_spec = deepcopy(spec)
                self.best_metric = metric
                self.best_experiment_id = entry.experiment_id
                logger.info(
                    "Experiment %d: KEEP (metric=%.4f, improved from %.4f)",
                    entry.experiment_id, metric,
                    previous_best if previous_best is not None else 0.0,
                )
            else:
                logger.info(
                    "Experiment %d: DISCARD (metric=%.4f, best=%.4f)",
                    entry.experiment_id, metric,
                    self.best_metric if self.best_metric else 0.0,
                )
        elif status in ("failure", "crash"):
            entry.decision = "discard"
            logger.info("Experiment %d: DISCARD (status=%s)", entry.experiment_id, status)
        else:
            entry.decision = "pending"

        self.history.append(entry)
        return entry

    def _is_improvement(self, metric: float) -> bool:
        """Check if a metric value is an improvement over the current best."""
        if self.best_metric is None:
            return True
        if self.metric_direction == "maximize":
            return metric > self.best_metric
        return metric < self.best_metric

    def get_history_for_llm(self, max_entries: int = 50) -> List[Dict[str, Any]]:
        """Format history as context for LLM prompts."""
        entries = self.history[-max_entries:]
        return [
            {
                "experiment_id": e.experiment_id,
                "modifications": e.modifications,
                "metric": e.metric,
                "status": e.status,
                "decision": e.decision,
                "reasoning": e.reasoning,
                "feedback": e.feedback,
            }
            for e in entries
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics."""
        total = len(self.history)
        keeps = sum(1 for e in self.history if e.decision == "keep")
        discards = sum(1 for e in self.history if e.decision == "discard")
        crashes = sum(1 for e in self.history if e.status in ("failure", "crash"))
        metrics = [e.metric for e in self.history if e.metric is not None]

        return {
            "total_experiments": total,
            "keeps": keeps,
            "discards": discards,
            "crashes": crashes,
            "best_metric": self.best_metric,
            "best_experiment_id": self.best_experiment_id,
            "metric_range": [min(metrics), max(metrics)] if metrics else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tracker state for persistence."""
        return {
            "history": [e.to_dict() for e in self.history],
            "best_spec": self.best_spec,
            "best_metric": self.best_metric,
            "best_experiment_id": self.best_experiment_id,
            "metric_direction": self.metric_direction,
            "next_id": self._next_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentTracker":
        """Deserialize tracker state."""
        tracker = cls(metric_direction=data.get("metric_direction", "maximize"))
        tracker.best_spec = data.get("best_spec")
        tracker.best_metric = data.get("best_metric")
        tracker.best_experiment_id = data.get("best_experiment_id")
        tracker._next_id = data.get("next_id", 0)
        for entry_data in data.get("history", []):
            entry = ExperimentEntry.from_dict(entry_data)
            tracker.history.append(entry)
            if entry.decision == "keep" and entry.experiment_id == tracker.best_experiment_id:
                tracker.best_spec = entry.spec
        return tracker
