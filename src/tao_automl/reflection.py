# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared reflection safeguards, exposed through the existing TAO API."""

from tao_automl._shared.reflection import (
    REFLECTION_PRIVATE_FIELDS,
    sanitize_reflective_feedback,
)

__all__ = ["REFLECTION_PRIVATE_FIELDS", "sanitize_reflective_feedback"]
