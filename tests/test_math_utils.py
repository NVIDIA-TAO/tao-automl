# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AutoML math utilities."""

import pytest

from tao_automl.utils.math_utils import clamp_value


@pytest.mark.parametrize("suggestion", [0.0, 0.3, 1.0])
def test_clamp_value_preserves_fixed_range(suggestion):
    """Equal custom bounds must pin a parameter to their exact value."""
    assert clamp_value(suggestion, 0.3, 0.3) == pytest.approx(0.3)
