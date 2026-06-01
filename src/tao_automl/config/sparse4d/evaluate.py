# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration hyperparameter schema to run evaluation on model."""

from typing import List
from dataclasses import dataclass

from tao_automl.config.utils.types import (
    LIST_FIELD,
    DATACLASS_FIELD
)
from tao_automl.config.common.common_config import EvaluateConfig
from tao_automl.config.sparse4d.dataset import Sparse4DTrackingConfig


@dataclass
class Sparse4DEvaluateConfig(EvaluateConfig):
    """Evaluation configuration for Sparse4D."""

    metrics: List[str] = LIST_FIELD(
        arrList=["detection"],
        default_value=["detection"],
        description="Metrics to evaluate",
        display_name="Metrics to evaluate"
    )
    tracking: Sparse4DTrackingConfig = DATACLASS_FIELD(
        Sparse4DTrackingConfig(),
        description="Tracking config",
        display_name="Tracking config"
    )
