# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical value conversion for AutoML specs and persisted state."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from enum import Enum
from typing import Any

import numpy as np


def normalize_json_value(value: Any, *, path: str = "$") -> Any:
    """Return a finite, JSON-compatible copy of ``value``.

    AutoML algorithms frequently produce NumPy scalars and arrays. Converting
    them once at the recommendation boundary keeps the live recommendation,
    persisted state, and SDK submission types identical.
    """
    return _normalize_json_value(value, path=path, active_containers=set())


def normalize_finite_number(value: Any, *, path: str = "$") -> float:
    """Return one normalized finite, non-boolean numeric scalar."""
    normalized = normalize_json_value(value, path=path)
    if isinstance(normalized, bool) or not isinstance(normalized, (int, float)):
        raise TypeError(
            f"{path} must be a finite numeric scalar, "
            f"got {type(normalized).__name__}"
        )
    try:
        numeric_value = float(normalized)
    except OverflowError as exc:
        raise ValueError(
            f"{path} must be representable as a finite float"
        ) from exc
    if not math.isfinite(numeric_value):
        raise ValueError(f"{path} must be a finite float")
    return numeric_value


def _normalize_json_value(
    value: Any,
    *,
    path: str,
    active_containers: set[int],
) -> Any:
    if isinstance(value, Enum):
        return _normalize_json_value(
            value.value,
            path=path,
            active_containers=active_containers,
        )

    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, np.generic):
            if isinstance(value, np.bool_):
                item = bool(value)
            elif isinstance(value, np.integer):
                item = int(value)
            elif isinstance(value, np.floating):
                item = float(value)
            elif isinstance(value, np.str_):
                item = str(value)
            else:
                raise TypeError(
                    f"{path}: unsupported NumPy scalar type "
                    f"{type(value).__name__}"
                )
        return _normalize_json_value(
            item,
            path=path,
            active_containers=active_containers,
        )

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _normalize_json_value(
                value.item(),
                path=path,
                active_containers=active_containers,
            )
        return _normalize_container(
            value,
            value.tolist(),
            path=path,
            active_containers=active_containers,
        )

    if isinstance(value, os.PathLike):
        normalized_path = os.fspath(value)
        if not isinstance(normalized_path, str):
            raise TypeError(f"{path}: path values must resolve to strings")
        return normalized_path

    if value is None or isinstance(value, (bool, str, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: numeric values must be finite, got {value!r}")
        return value

    if isinstance(value, Mapping):
        return _normalize_mapping(
            value,
            path=path,
            active_containers=active_containers,
        )

    if isinstance(value, (list, tuple)):
        return _normalize_container(
            value,
            value,
            path=path,
            active_containers=active_containers,
        )

    raise TypeError(
        f"{path}: unsupported value type {type(value).__name__}; "
        "expected JSON-compatible data"
    )


def _normalize_mapping(
    value: Mapping,
    *,
    path: str,
    active_containers: set[int],
) -> dict[str, Any]:
    container_id = _enter_container(value, path, active_containers)
    try:
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{path}: mapping keys must be strings, got "
                    f"{type(key).__name__}"
                )
            normalized_key = str(key)
            normalized[normalized_key] = _normalize_json_value(
                item,
                path=f"{path}.{normalized_key}",
                active_containers=active_containers,
            )
        return normalized
    finally:
        active_containers.remove(container_id)


def _normalize_container(
    original: Any,
    items: Any,
    *,
    path: str,
    active_containers: set[int],
) -> list[Any]:
    container_id = _enter_container(original, path, active_containers)
    try:
        return [
            _normalize_json_value(
                item,
                path=f"{path}[{index}]",
                active_containers=active_containers,
            )
            for index, item in enumerate(items)
        ]
    finally:
        active_containers.remove(container_id)


def _enter_container(value: Any, path: str, active_containers: set[int]) -> int:
    container_id = id(value)
    if container_id in active_containers:
        raise ValueError(f"{path}: circular references are not supported")
    active_containers.add(container_id)
    return container_id
