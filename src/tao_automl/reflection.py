# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared safeguards for data sent to reflective optimization models."""

from __future__ import annotations

from typing import Any


REFLECTION_PRIVATE_FIELDS = frozenset({
    "id", "sample_id", "video_id", "dataset_id", "path", "video", "image",
    "gold", "gold_answer", "expected", "expected_answer", "reference",
    "reference_answer", "label", "target_label",
})


def sanitize_reflective_feedback(value: Any):
    """Remove common identifiers and ground-truth fields recursively.

    Callers remain responsible for not embedding answers in free-form text; a
    generic sanitizer cannot reliably distinguish an answer from ordinary prose.
    """
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
            if normalized in REFLECTION_PRIVATE_FIELDS:
                continue
            clean[key] = sanitize_reflective_feedback(item)
        return clean
    if isinstance(value, (list, tuple)):
        return [sanitize_reflective_feedback(item) for item in value]
    return value
