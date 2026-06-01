# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default config file."""

from dataclasses import dataclass
from omegaconf import MISSING
from typing import Optional, List

from tao_automl.config.utils.types import DATACLASS_FIELD


@dataclass
class DataConfig:
    """Dataset configuration template."""

    format: str = "COCO"
    annotations: List[str] = MISSING
    same_categories: bool = True


@dataclass
class MergeConfig:
    """Experiment configuration template."""

    data: DataConfig = DATACLASS_FIELD(DataConfig())
    results_dir: Optional[str] = None
