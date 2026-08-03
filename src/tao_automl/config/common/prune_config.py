# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default config file."""

from typing import Optional
from dataclasses import dataclass

from tao_automl.config.utils.types import (
    INT_FIELD,
    STR_FIELD,
    FLOAT_FIELD
)


@dataclass
class PruneConfig:
    """Prune config."""

    mode: str = STR_FIELD(
        value="amount",
        value_type="ordered",
        default_value="amount",
        automl_enabled="TRUE",
        valid_options="amount,threshold,experimental_hybrid",
        description="Pruning mode.",
        display="Pruning mode"
    )
    amount: float = FLOAT_FIELD(
        value=0.4,
        default_value=0.4,
        automl_enabled="TRUE",
        valid_min=0.0,
        valid_max=1.0,
        description="Pruning amount",
        display_name="Pruning amount"
    )
    threshold: Optional[float] = FLOAT_FIELD(
        value=None,
        default_value=None,
        valid_min=0.0,
        valid_max=1.0,
        description="Pruning threshold",
        display_name="Pruning threshold"
    )
    granularity: int = INT_FIELD(
        value=8,
        default_value=8,
        automl_enabled="TRUE",
        valid_min=1,
        valid_max=64,
        math_cond="/ 4",
        description="Pruning granularity",
        display="Pruning granularity"
    )
    raw_prune_score: str = STR_FIELD(
        value="L1",
        value_type="ordered",
        default_value="L1",
        automl_enabled="TRUE",
        valid_options="L1,L2",
        description="Learning rate monitor for AutoReduce learning rate scheduler.",
        display="lr monitor"
    )
