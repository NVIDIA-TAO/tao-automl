# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration hyperparameter schema for the model."""

from dataclasses import dataclass
from typing import List, Optional

from tao_automl.config.utils.types import (
    BOOL_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
    DATACLASS_FIELD,
)


@dataclass
class MonoBackBone:
    """Define MonoBackBone dependency config"""

    pretrained_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="Pretrained path for mono backbone",
        description="""Path to load depth anything v2 as an encoder for Monocular DepthNet""",
    )
    use_bn: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Batch normalization in Monocular DepthNet",
        description="""A flag specifying whether to use batch normalization in Monocular DepthNet""",
    )
    use_clstoken: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Class token in Monocular DepthNet",
        description="""A flag specifying whether to use class token""",
    )


@dataclass
class StereoBackBone:
    """Define StereoBackBone dependency config"""

    depth_anything_v2_pretrained_path: Optional[str] = STR_FIELD(
        value="",
        default_value="",
        description="""Path to load depth anything v2 as an encoder for Stereo DepthNet (FoundationStereo)""",
    )
    edgenext_pretrained_path: Optional[str] = STR_FIELD(
        value="",
        default_value="",
        description="""Path to load edgenext encoder for Stereo DepthNet (FoundationStereo)""",
    )
    use_bn: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="batch normalization in DepthAnythingV2",
        description="""A flag specifying whether to use batch normalization in DepthAnythingV2""",
    )
    use_clstoken: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="class token in DepthAnythingV2",
        description="""A flag specifying whether to use class token""",
    )


@dataclass
class DepthNetModelConfig:
    """DepthNet model config."""

    model_type: str = STR_FIELD(
        value="MetricDepthAnything",
        default_value="MetricDepthAnything",
        description="Network name",
        valid_options=",".join([
            "FoundationStereo", "MetricDepthAnything", "RelativeDepthAnything"
        ])
    )
    mono_backbone: MonoBackBone = DATACLASS_FIELD(
        MonoBackBone(),
        value="",
        default_value="",
        display_name="Mono backbone configuration",
        description="Network defined paths for Monocular DepthNet Backbone",
    )
    stereo_backbone: StereoBackBone = DATACLASS_FIELD(
        StereoBackBone(),
        value="",
        default_value="",
        display_name="Stereo backbone configuration",
        description="Network defined paths for Edgenext and Depthanythingv2",
    )
    hidden_dims: List[int] = LIST_FIELD(
        arrList=[128, 128, 128],
        description="The hidden dimensions.",
        display_name="The hidden dimensions."
    )
    corr_radius: int = INT_FIELD(
        value=4,
        default_value=4,
        description="The width of the correlation pyramid",
        display_name="correlation pyramid width",
        valid_min=2,
        valid_max=8,
        automl_enabled="TRUE"
    )
    cv_group: int = INT_FIELD(
        value=8,
        default_value=8,
        description="cv group",
        display_name="cv group",
        valid_min=4,
        valid_max=16,
        automl_enabled="TRUE"
    )
    train_iters: int = INT_FIELD(
        value=22,
        default_value=22,
        description="Train Iteration",
        display_name="train iteration",
        valid_min=1,
    )
    valid_iters: int = INT_FIELD(
        value=22,
        default_value=22,
        description="Validation Iteration",
        display_name="Validation iteration",
        valid_min=1,
    )
    volume_dim: int = INT_FIELD(
        value=32,
        default_value=32,
        description="Volume dimension",
        display_name="volume dimension",
        valid_min=16,
        valid_max=64,
        automl_enabled="TRUE"
    )
    low_memory: int = INT_FIELD(
        value=0,
        default_value=0,
        description="reduce memory usage",
        display_name="reduce memory usage",
        valid_min=0,
        valid_max=4,
    )
    mixed_precision: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Mixed Precision Training",
        description="""A flag specifying whether to use mixed precision training""",
    )
    n_gru_layers: int = INT_FIELD(
        value=3,
        valid_min=1,
        valid_max=3,
        description="The number of hidden GRU levels",
        display_name="number of hidden GRU levels"
    )
    corr_levels: int = INT_FIELD(
        value=2,
        valid_min=1,
        valid_max=2,
        description="The number of levels in the correlation pyramid",
        display_name="number of correlation pyramid levels"
    )
    n_downsample: int = INT_FIELD(
        value=2,
        valid_min=1,
        valid_max=2,
        description="resolution of the disparity field (1/2^K)",
        display_name="disparity field resoultion"
    )
    encoder: str = STR_FIELD(
        value="vitl",
        default_value="vitl",
        description="DepthAnythingV2 Encoder options",
        valid_options=",".join([
            "vits", "vitb", "vitl", "vitg"
        ])
    )
    max_disparity: int = INT_FIELD(
        value=416,
        display_name="max disparity",
        description="""
        The maximum disparity of the model used in the training of a stereo model
        """
    )
