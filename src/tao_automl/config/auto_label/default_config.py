# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default config file."""

from typing import List, Optional, Dict
from dataclasses import dataclass
from tao_automl.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    BOOL_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD,
)
from tao_automl.config.grounding_dino.dataset import GDINOAugmentationConfig
from tao_automl.config.grounding_dino.model import GDINOModelConfig
from tao_automl.config.grounding_dino.train import GDINOTrainExpConfig
from tao_automl.config.mal.default_config import (
    MALTrainExpConfig,
    MALEvalExpConfig,
    MALInferenceExpConfig,
    MALDatasetConfig,
    MALModelConfig,
)


@dataclass
class MALConfig:
    """MAL config."""

    dataset: MALDatasetConfig = DATACLASS_FIELD(
        MALDatasetConfig(),
        description="Configuration parameters for MAL dataset"
    )
    train: MALTrainExpConfig = DATACLASS_FIELD(
        MALTrainExpConfig(),
        description="Configuration parameters for MAL train"
    )
    model: MALModelConfig = DATACLASS_FIELD(
        MALModelConfig(),
        description="Configuration parameters for MAL model"
    )
    inference: MALInferenceExpConfig = DATACLASS_FIELD(
        MALInferenceExpConfig(),
        description="Configuration parameters for MAL inference"
    )
    evaluate: MALEvalExpConfig = DATACLASS_FIELD(
        MALEvalExpConfig(),
        description="Configuration parameters for MAL evaluation"
    )
    checkpoint: Optional[str] = STR_FIELD(
        None,
        default_value="",
        description="MAL model checkpoint path",
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Result directory",
    )


@dataclass
class GDINOConfig:
    """Grounding DINO config."""

    @dataclass
    class GDINODataConfig:
        """DINO dataset config used for auto-labeling."""

        image_dir: Optional[str] = STR_FIELD(
            None,
            default_value="",
            description="Image root directory",
        )
        noun_chunk_path: Optional[str] = STR_FIELD(
            value=None,
            default_value=""
        )
        class_names: Optional[List[str]] = LIST_FIELD(
            arrList=[],
            description="List of classes to run auto-labeling"
        )
        augmentation: GDINOAugmentationConfig = DATACLASS_FIELD(
            GDINOAugmentationConfig(),
            description="Configuration parameters for Grounding DINO augmenation"
        )

    train: GDINOTrainExpConfig = DATACLASS_FIELD(
        GDINOTrainExpConfig(),
        description="Configuration parameters for Grounding DINO train"
    )
    model: GDINOModelConfig = DATACLASS_FIELD(
        GDINOModelConfig(),
        description="Configuration parameters for Grounding DINO model"
    )
    dataset: GDINODataConfig = DATACLASS_FIELD(
        GDINODataConfig(),
        description="Configuration parameters for Grounding DINO dataset"
    )

    checkpoint: Optional[str] = STR_FIELD(
        None,
        default_value="",
        description="Grounding model checkpoint path",
    )

    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Result directory",
    )

    iteration_scheduler: List[Dict[str, float]] = LIST_FIELD(
        arrList=[{"conf_threshold": 0.5, "nms_threshold": 0.0}],
        default_values=[{"conf_threshold": 0.5, "nms_threshold": 0.0}],
        description="""The list of iteration schedule. Default is one iteration with confidence threshold of 0.5.
                    Next iteration eliminates classes/noun chunks that have been already detected."""
    )
    visualize: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to enable visualization of bounding boxes."
    )


@dataclass
class ExperimentConfig:
    """Experiment configuration template."""

    gpu_ids: List[int] = LIST_FIELD(
        arrList=[0],
        default_value=[0],
        description="Indices of GPUs to use"
    )
    num_gpus: int = INT_FIELD(value=1,
                              default_value=1,
                              description="Number of GPUs to use")
    batch_size: int = INT_FIELD(value=4,
                                default_value=4,
                                valid_min=1,
                                description="Batch size")
    num_workers: int = INT_FIELD(value=8,
                                 default_value=8,
                                 valid_min=1,
                                 description="Number of workers for dataloader")

    autolabel_type: str = STR_FIELD(
        value="mal",
        default_value="mal",
        description="Type of auto-labeling to run",
        valid_options="mal,grounding_dino"
    )

    mal: MALConfig = DATACLASS_FIELD(
        MALConfig(),
        description="Configuration parameters for MAL"
    )
    grounding_dino: GDINOConfig = DATACLASS_FIELD(
        GDINOConfig(),
        description="Configuration parameters for Grounding DINO"
    )

    results_dir: str = STR_FIELD(
        value="",
        default_value="",
        description="Result directory",
    )

    def __post_init__(self):
        """assertion check."""
        assert self.autolabel_type in ["mal", "grounding_dino"], f"Invalid option encountered. {self.autolabel_type}"
