# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AutoML search-space filtering."""

from tao_automl.schema.generate_schema import generate_schema
from tao_automl.search_space.params import generate_hyperparams_to_search


def test_cosmos_default_search_excludes_fps_when_nframes_is_active():
    specs = generate_schema("cosmos-rl", "train")["default"]

    _, param_names = generate_hyperparams_to_search(
        network="cosmos-rl",
        action="train",
        train_specs=specs,
        automl_hyperparameters=[],
    )

    assert "nframes" in specs["custom"]["vision"]
    assert "custom.vision.fps" not in param_names


def test_cosmos_explicit_fps_search_is_still_allowed():
    specs = generate_schema("cosmos-rl", "train")["default"]

    _, param_names = generate_hyperparams_to_search(
        network="cosmos-rl",
        action="train",
        train_specs=specs,
        automl_hyperparameters=["custom.vision.fps"],
    )

    assert "custom.vision.fps" in param_names
