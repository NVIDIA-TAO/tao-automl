# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default config file for Sparse4D."""

from dataclasses import dataclass

from tao_automl.config.utils.types import (
    DATACLASS_FIELD
)
from tao_automl.config.common.common_config import (
    CommonExperimentConfig,
    ExportConfig,
)
from tao_automl.config.common.quantization import ModelQuantizationConfig
from tao_automl.config.sparse4d.model import Sparse4DModelConfig
from tao_automl.config.sparse4d.dataset import Omniverse3DDetTrackDatasetConfig
from tao_automl.config.sparse4d.train import Sparse4DTrainConfig
from tao_automl.config.sparse4d.inference import Sparse4DInferenceConfig
from tao_automl.config.sparse4d.evaluate import Sparse4DEvaluateConfig
from tao_automl.config.sparse4d.visualize import Sparse4DVisualizeConfig


@dataclass
class Sparse4DExportConfig(ExportConfig):
    """Export configuration for Sparse4D."""

    pass


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment configuration for Sparse4D."""

    train: Sparse4DTrainConfig = DATACLASS_FIELD(
        Sparse4DTrainConfig(),
        description="Train config",
        display_name="Train config"
    )
    model: Sparse4DModelConfig = DATACLASS_FIELD(
        Sparse4DModelConfig(),
        description="Model config",
        display_name="Model config"
    )
    dataset: Omniverse3DDetTrackDatasetConfig = DATACLASS_FIELD(
        Omniverse3DDetTrackDatasetConfig(),
        description="Dataset config",
        display_name="Dataset config"
    )
    inference: Sparse4DInferenceConfig = DATACLASS_FIELD(
        Sparse4DInferenceConfig(),
        description="Inference config",
        display_name="Inference config"
    )
    evaluate: Sparse4DEvaluateConfig = DATACLASS_FIELD(
        Sparse4DEvaluateConfig(),
        description="Evaluate config",
        display_name="Evaluate config"
    )
    export: Sparse4DExportConfig = DATACLASS_FIELD(
        Sparse4DExportConfig(),
        description="Export config",
        display_name="Export config"
    )
    visualize: Sparse4DVisualizeConfig = DATACLASS_FIELD(
        Sparse4DVisualizeConfig(),
        description="Visualize config",
        display_name="Visualize config"
    )
    quantize: ModelQuantizationConfig = DATACLASS_FIELD(
        ModelQuantizationConfig(),
        description="Configurable parameters for model quantization.",
    )

    def __post_init__(self):
        """Set default model name for Sparse4D."""
        if self.model_name is None:
            self.model_name = "sparse4d"
