# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic issuance-time audit records for AutoML recommendations.

The audit is captured before a recommendation is launched. It records the
search policy and all observations visible to the algorithm, while keeping
validation-only measurements outside the recommendation path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from tao_automl.selection import canonical_spec_fingerprint


RECOMMENDATION_AUDIT_SCHEMA_VERSION = 1

# Immutable campaign assertions. They are never inferred from results or
# changed after a winner is observed. Keep the schema separate from serialized
# records so validation cannot be weakened by mutating a public reference.
_ALGORITHMIC_CAMPAIGN_FLAG_NAMES = (
    "agent_selected_candidate",
    "agent_injected_candidate",
    "agent_modified_search_space_after_results",
    "agent_changed_seed_after_results",
    "agent_changed_budget_after_results",
    "agent_changed_threshold_after_results",
    "agent_changed_ptm_after_results",
    "agent_overrode_winner",
)
ALGORITHMIC_CAMPAIGN_FLAGS = MappingProxyType({
    name: False for name in _ALGORITHMIC_CAMPAIGN_FLAG_NAMES
})


def algorithmic_campaign_flags() -> dict[str, bool]:
    """Return a new strict-JSON record of the required campaign assertions."""
    return {name: False for name in _ALGORITHMIC_CAMPAIGN_FLAG_NAMES}


def validate_algorithmic_campaign_flags(value: Any) -> None:
    """Require the exact immutable flag schema with every assertion false."""
    if not isinstance(value, Mapping):
        raise ValueError(
            "Recommendation audit campaign intervention flags are missing"
        )
    if set(value) != set(_ALGORITHMIC_CAMPAIGN_FLAG_NAMES):
        raise ValueError(
            "Recommendation audit campaign intervention flag schema changed"
        )
    if any(value[name] is not False for name in _ALGORITHMIC_CAMPAIGN_FLAG_NAMES):
        raise ValueError(
            "Recommendation audit campaign intervention flags were modified"
        )


def _finite_or_tagged(value: float) -> float | dict[str, str]:
    value = float(value)
    if math.isfinite(value):
        return value
    if math.isnan(value):
        label = "nan"
    elif value > 0:
        label = "positive_infinity"
    else:
        label = "negative_infinity"
    return {"__automl_nonfinite__": label}


def audit_json_value(value: Any) -> Any:
    """Return a deterministic strict-JSON representation of audit metadata.

    Schema-derived search-space records commonly contain NumPy values and NaN
    sentinels. Unlike measured objectives, these sentinels are metadata, so the
    audit tags them explicitly instead of dropping a bound or emitting
    non-standard JSON.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _finite_or_tagged(value)
    if isinstance(value, Enum):
        return audit_json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result = {}
        for key in sorted(value, key=str):
            result[str(key)] = audit_json_value(value[key])
        return result
    if isinstance(value, (list, tuple)):
        return [audit_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [audit_json_value(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )

    # NumPy arrays/scalars and equivalent scalar containers.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if converted is not value:
            return audit_json_value(converted)
    item = getattr(value, "item", None)
    if callable(item):
        converted = item()
        if converted is not value:
            return audit_json_value(converted)

    # pandas.NA is a schema sentinel. Avoid importing pandas here.
    if type(value).__name__ == "NAType":
        return {"__automl_missing__": "pandas.NA"}

    raise TypeError(
        "Recommendation audit value is not deterministically JSON serializable: "
        f"{type(value).__name__}"
    )


def canonical_audit_sha256(value: Any) -> str:
    """Hash strict canonical JSON audit content."""
    encoded = json.dumps(
        audit_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def visible_history_snapshot(history: Iterable[Any]) -> list[dict[str, Any]]:
    """Capture every recommendation state visible to an algorithm call."""
    snapshot = []
    for recommendation in history:
        specs = audit_json_value(getattr(recommendation, "specs", {}) or {})
        snapshot.append(
            {
                "candidate_id": str(getattr(recommendation, "id", "")),
                "candidate_fingerprint": canonical_spec_fingerprint(specs),
                "status": str(getattr(recommendation, "status", "")),
                "result": audit_json_value(
                    getattr(recommendation, "result", None)
                ),
                "objective_score": audit_json_value(
                    getattr(recommendation, "objective_score", None)
                ),
                "objective_values": audit_json_value(
                    getattr(recommendation, "objective_values", {}) or {}
                ),
                "failure_reason": audit_json_value(
                    getattr(recommendation, "failure_reason", None)
                ),
            }
        )
    return snapshot


def _successful_observations(
    visible_history: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(item)
        for item in visible_history
        if item.get("status") in {"success", "done"}
    ]


def build_recommendation_audit(
    *,
    candidate_id: int | str,
    specs: Mapping[str, Any],
    algorithm: str,
    search_seed: int | None,
    search_space: Any,
    custom_ranges: Any,
    objective_config: Any,
    visible_history: Iterable[Mapping[str, Any]],
    acquisition: Any,
    is_resume_promotion: bool = False,
) -> dict[str, Any]:
    """Build a content-addressed, issuance-time recommendation audit."""
    normalized_specs = audit_json_value(specs)
    normalized_space = audit_json_value(search_space)
    normalized_ranges = audit_json_value(custom_ranges or {})
    normalized_history = audit_json_value(list(visible_history))
    normalized_objectives = audit_json_value(
        objective_config.to_dict()
        if objective_config is not None
        else {}
    )
    normalized_acquisition = audit_json_value(acquisition or {})

    record = {
        "schema_version": RECOMMENDATION_AUDIT_SCHEMA_VERSION,
        "candidate_id": str(candidate_id),
        "candidate_fingerprint": canonical_spec_fingerprint(normalized_specs),
        "generated_parameter_values": normalized_specs,
        "search_algorithm": str(algorithm),
        "search_seed": search_seed,
        "search_space": normalized_space,
        "search_space_sha256": canonical_audit_sha256(normalized_space),
        "custom_parameter_ranges": normalized_ranges,
        "custom_parameter_ranges_sha256": canonical_audit_sha256(
            normalized_ranges
        ),
        "objective_configuration": normalized_objectives,
        "objective_configuration_sha256": canonical_audit_sha256(
            normalized_objectives
        ),
        "history_visible_to_algorithm": normalized_history,
        "history_visible_sha256": canonical_audit_sha256(normalized_history),
        "previous_successful_observations": _successful_observations(
            normalized_history
        ),
        "acquisition": normalized_acquisition,
        "is_resume_promotion": bool(is_resume_promotion),
        "selection_time_measurements_only": True,
        "algorithmic_campaign_flags": algorithmic_campaign_flags(),
    }
    record["audit_sha256"] = canonical_audit_sha256(record)
    return record


def validate_recommendation_audit(record: Any) -> None:
    """Fail closed when persisted issuance evidence was changed in place."""
    if not isinstance(record, Mapping):
        raise ValueError("Recommendation audit must be a mapping")
    if record.get("schema_version") != RECOMMENDATION_AUDIT_SCHEMA_VERSION:
        raise ValueError("Recommendation audit schema version is unsupported")
    expected = record.get("audit_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Recommendation audit SHA-256 is missing or invalid")
    payload = copy.deepcopy(dict(record))
    payload.pop("audit_sha256", None)
    actual = canonical_audit_sha256(payload)
    if actual != expected:
        raise ValueError("Recommendation audit integrity verification failed")
    validate_algorithmic_campaign_flags(
        record.get("algorithmic_campaign_flags")
    )
    generated = record.get("generated_parameter_values")
    if not isinstance(generated, Mapping):
        raise ValueError(
            "Recommendation audit generated parameter values are missing"
        )
    if canonical_spec_fingerprint(generated) != record.get(
        "candidate_fingerprint"
    ):
        raise ValueError("Recommendation audit candidate fingerprint is invalid")
