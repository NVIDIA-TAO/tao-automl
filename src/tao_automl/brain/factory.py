# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AutoML brain factory"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from tao_automl.brain.bayesian import Bayesian
from tao_automl.brain.hyperband import HyperBand
from tao_automl.brain.bohb import BOHB
from tao_automl.brain.bfbo import BFBO
from tao_automl.brain.asha import ASHA
from tao_automl.brain.pbt import PBT
from tao_automl.brain.dehb import DEHB
from tao_automl.brain.hyperband_es import HyperBandES
from tao_automl.brain.llm_brain import LLMBrain
from tao_automl.brain.hybrid_controller import HybridBrain
from tao_automl.brain.autoresearch_controller import AutoresearchBrain

logger = logging.getLogger(__name__)


def _as_bool(value: Any) -> bool:
    """Parse bool-like values from runner settings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


# Constants for algorithm names
class AlgorithmType:
    """Constants for AutoML algorithm types"""

    BAYESIAN = ("bayesian", "b")
    BFBO = ("bfbo",)
    HYPERBAND = ("hyperband", "h")
    BOHB = ("bohb",)
    ASHA = ("asha",)
    PBT = ("pbt",)
    DEHB = ("dehb",)
    HYPERBAND_ES = ("hyperband_es", "hes")
    LLM = ("llm",)
    HYBRID = ("hybrid",)
    AUTORESEARCH = ("autoresearch",)


@dataclass
class AlgorithmParams:
    """Dataclass to hold algorithm-specific parameters with defaults"""

    automl_max_recommendations: int = 20
    automl_max_epochs: int = 27
    automl_reduction_factor: int = 3
    epoch_multiplier: int = 1
    automl_max_concurrent: int = 4
    automl_population_size: int = 10
    automl_max_generations: int = 20
    automl_eval_interval: int = 10
    automl_perturbation_factor: float = 1.2
    automl_mutation_factor: float = 0.5
    automl_crossover_prob: float = 0.5
    automl_early_stop_threshold: float = 0.1
    automl_min_early_stop_epochs: int = 3
    automl_kde_samples: int = 64
    automl_top_n_percent: float = 15.0
    automl_min_points_in_model: int = 10
    automl_max_trials: int = None  # ASHA: max configs to try (None = unlimited)
    automl_min_top_configs: int = 5  # ASHA: min configs that must reach final rung before stopping

    # LLM/agentic algorithm params
    llm_endpoint: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096
    automl_max_experiments: int = 50  # autoresearch budget
    research_program: Optional[str] = None
    hybrid_enable_llm_range_narrowing: bool = False
    automl_delete_intermediate_ckpt: bool = True

    @classmethod
    def from_dict(cls, params_dict: Dict[str, Any]) -> 'AlgorithmParams':
        """Create AlgorithmParams from dictionary with defaults"""
        return cls(
            automl_max_recommendations=params_dict.get("automl_max_recommendations", 20),
            automl_max_epochs=params_dict.get("automl_max_epochs", 27),
            automl_reduction_factor=params_dict.get("automl_reduction_factor", 3),
            epoch_multiplier=params_dict.get("epoch_multiplier", 1),
            automl_max_concurrent=params_dict.get("automl_max_concurrent", 4),
            automl_population_size=params_dict.get("automl_population_size", 10),
            automl_max_generations=params_dict.get("automl_max_generations", 20),
            automl_eval_interval=params_dict.get("automl_eval_interval", 10),
            automl_perturbation_factor=params_dict.get("automl_perturbation_factor", 1.2),
            automl_mutation_factor=params_dict.get("automl_mutation_factor", 0.5),
            automl_crossover_prob=params_dict.get("automl_crossover_prob", 0.5),
            automl_early_stop_threshold=params_dict.get("automl_early_stop_threshold", 0.1),
            automl_min_early_stop_epochs=params_dict.get("automl_min_early_stop_epochs", 3),
            automl_kde_samples=params_dict.get("automl_kde_samples", 64),
            automl_top_n_percent=params_dict.get("automl_top_n_percent", 15.0),
            automl_min_points_in_model=params_dict.get("automl_min_points_in_model", 10),
            automl_max_trials=params_dict.get("automl_max_trials", None),
            automl_min_top_configs=params_dict.get("automl_min_top_configs", 5),
            automl_delete_intermediate_ckpt=_as_bool(
                params_dict.get("automl_delete_intermediate_ckpt", True)
            ),
            llm_endpoint=params_dict.get("llm_endpoint", params_dict.get("base_url", "")),
            llm_model=params_dict.get("llm_model", params_dict.get("model", "")),
            llm_api_key=params_dict.get("llm_api_key", params_dict.get("api_key", "")),
            llm_temperature=float(params_dict.get("llm_temperature", 0.7)),
            llm_max_tokens=int(params_dict.get("llm_max_tokens", 4096)),
            automl_max_experiments=int(params_dict.get("automl_max_experiments", 50)),
            research_program=params_dict.get("research_program"),
            hybrid_enable_llm_range_narrowing=_as_bool(
                params_dict.get(
                    "hybrid_enable_llm_range_narrowing",
                    params_dict.get("enable_llm_range_narrowing", False),
                )
            ),
        )

    def get_llm_params(self) -> Dict[str, Any]:
        """Extract LLM-related params as a dict for LLMClient."""
        d = {}
        if self.llm_endpoint:
            d["llm_endpoint"] = self.llm_endpoint
        if self.llm_model:
            d["llm_model"] = self.llm_model
        if self.llm_api_key:
            d["llm_api_key"] = self.llm_api_key
        if self.llm_temperature != 0.7:
            d["llm_temperature"] = str(self.llm_temperature)
        if self.llm_max_tokens != 4096:
            d["llm_max_tokens"] = str(self.llm_max_tokens)
        return d if d else None


class BrainFactory:
    """Factory class for creating AutoML brain instances"""

    @staticmethod
    def create_brain(
        algorithm: str,
        context,
        state_store,
        network: str,
        parameters: Any,
        params: AlgorithmParams,
        metric: str = "loss",
        resume: bool = False
    ):
        """Create brain instance based on algorithm type

        Args:
            algorithm: Algorithm name string
            context: AutoMLContext instance
            state_store: StateStore instance
            network: Network architecture name
            parameters: AutoML sweepable parameters
            params: AlgorithmParams with algorithm-specific settings
            metric: Metric to optimize (e.g., 'loss', 'val_accuracy', 'mIoU')
            resume: Whether to resume from previous state
        """
        algo_lower = algorithm.lower()

        if algo_lower in AlgorithmType.HYPERBAND:
            brain_class = HyperBand
            kwargs = {
                "context": context,
                "state_store": state_store,
                "network": network,
                "parameters": parameters,
                "max_epochs": int(params.automl_max_epochs),
                "reduction_factor": int(params.automl_reduction_factor),
                "epoch_multiplier": int(params.epoch_multiplier),
                "metric": metric
            }
        elif algo_lower in AlgorithmType.BAYESIAN:
            brain_class = Bayesian
            kwargs = {
                "context": context,
                "state_store": state_store,
                "network": network,
                "parameters": parameters,
                "metric": metric
            }
        elif algo_lower in AlgorithmType.BOHB:
            brain_class = BOHB
            kwargs = {
                "context": context,
                "state_store": state_store,
                "network": network,
                "parameters": parameters,
                "max_epochs": int(params.automl_max_epochs),
                "reduction_factor": int(params.automl_reduction_factor),
                "epoch_multiplier": int(params.epoch_multiplier),
                "kde_samples": int(params.automl_kde_samples),
                "top_n_percent": float(params.automl_top_n_percent),
                "min_points_in_model": int(params.automl_min_points_in_model),
                "metric": metric
            }
        elif algo_lower in AlgorithmType.BFBO:
            brain_class = BFBO
            kwargs = {
                "context": context,
                "state_store": state_store,
                "network": network,
                "parameters": parameters,
                "metric": metric
            }
        elif algo_lower in AlgorithmType.ASHA:
            brain_class = ASHA
            kwargs = {
                "context": context,
                "state_store": state_store,
                "network": network,
                "parameters": parameters,
                "max_epochs": int(params.automl_max_epochs),
                "reduction_factor": int(params.automl_reduction_factor),
                "epoch_multiplier": int(params.epoch_multiplier),
                "max_concurrent": int(params.automl_max_concurrent),
                "max_trials": params.automl_max_trials if params.automl_max_trials else None,
                "min_top_configs": int(params.automl_min_top_configs),
                "metric": metric
            }
        elif algo_lower in AlgorithmType.PBT:
            brain_class = PBT
            kwargs = {
                "context": context,
                "state_store": state_store,
                "network": network,
                "parameters": parameters,
                "population_size": int(params.automl_population_size),
                "max_generations": int(params.automl_max_generations),
                "eval_interval": int(params.automl_eval_interval),
                "perturbation_factor": float(params.automl_perturbation_factor),
                "metric": metric
            }
        elif algo_lower in AlgorithmType.DEHB:
            brain_class = DEHB
            kwargs = {
                "context": context,
                "state_store": state_store,
                "network": network,
                "parameters": parameters,
                "max_epochs": int(params.automl_max_epochs),
                "reduction_factor": int(params.automl_reduction_factor),
                "epoch_multiplier": int(params.epoch_multiplier),
                "mutation_factor": float(params.automl_mutation_factor),
                "crossover_prob": float(params.automl_crossover_prob),
                "metric": metric
            }
        elif algo_lower in AlgorithmType.HYPERBAND_ES:
            brain_class = HyperBandES
            kwargs = {
                "context": context,
                "state_store": state_store,
                "network": network,
                "parameters": parameters,
                "max_epochs": int(params.automl_max_epochs),
                "reduction_factor": int(params.automl_reduction_factor),
                "epoch_multiplier": int(params.epoch_multiplier),
                "early_stop_threshold": float(params.automl_early_stop_threshold),
                "min_early_stop_epochs": int(params.automl_min_early_stop_epochs),
                "metric": metric
            }
        elif algo_lower in AlgorithmType.LLM:
            brain_class = LLMBrain
            kwargs = {
                "context": context,
                "state_store": state_store,
                "network": network,
                "parameters": parameters,
                "llm_params": params.get_llm_params(),
                "metric": metric,
            }
        elif algo_lower in AlgorithmType.HYBRID:
            brain_class = HybridBrain
            kwargs = {
                "context": context,
                "state_store": state_store,
                "network": network,
                "parameters": parameters,
                "llm_params": params.get_llm_params(),
                "metric": metric,
                "max_experiments": int(params.automl_max_experiments),
                "enable_llm_range_narrowing": params.hybrid_enable_llm_range_narrowing,
            }
        elif algo_lower in AlgorithmType.AUTORESEARCH:
            brain_class = AutoresearchBrain
            kwargs = {
                "context": context,
                "state_store": state_store,
                "network": network,
                "parameters": parameters,
                "llm_params": params.get_llm_params(),
                "metric": metric,
                "max_experiments": int(params.automl_max_experiments),
                "research_program": params.research_program,
            }
        else:
            raise ValueError(f"AutoML Algorithm {algorithm} is not valid")

        # Create brain instance (load_state for resume, new instance otherwise)
        if resume:
            return brain_class.load_state(**kwargs)
        return brain_class(**kwargs)
