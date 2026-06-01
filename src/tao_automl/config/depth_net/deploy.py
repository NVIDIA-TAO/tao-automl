# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration hyperparameter schema to deploy the model."""

from dataclasses import dataclass

from tao_automl.config.utils.types import (
    DATACLASS_FIELD,
    STR_FIELD
)
from tao_automl.config.common.common_config import (
    GenTrtEngineConfig,
    TrtConfig
)


@dataclass
class DepthNetTrtConfig(TrtConfig):
    """Trt config."""

    data_type: str = STR_FIELD(
        value="FP32",
        default_value="FP32",
        description="The precision to be set for building the TensorRT engine.",
        display_name="data type",
        valid_options=",".join(["FP32", "FP16"])
    )


@dataclass
class DepthNetGenTrtEngineExpConfig(GenTrtEngineConfig):
    """Gen TRT Engine experiment config."""

    tensorrt: DepthNetTrtConfig = DATACLASS_FIELD(
        DepthNetTrtConfig(),
        description="Hyper parameters to configure the TensorRT Engine builder.",
        display_name="TensorRT hyper params."
    )
