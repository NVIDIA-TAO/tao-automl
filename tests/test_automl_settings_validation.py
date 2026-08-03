# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""automl_settings validation: unknown keys raise instead of silently running
algorithm defaults (a typo'd budget key cost 17 extra GPU allocations live)."""

import re
from pathlib import Path

import pytest

from tao_automl.runner import (
    AUTOML_SETTING_ALIASES,
    KNOWN_AUTOML_SETTINGS,
    _validate_automl_settings,
)


def test_typo_raises_with_suggestion():
    with pytest.raises(ValueError) as exc:
        _validate_automl_settings({
            "algorithm": "bayesian",
            "metric": "mAP50",
            "automl_max_recomendations": 3,
        })
    message = str(exc.value)
    assert "automl_max_recomendations" in message
    assert "automl_max_recommendations" in message  # the close-match hint


def test_num_recommendations_is_an_alias():
    normalized = _validate_automl_settings({
        "algorithm": "bayesian",
        "metric": "mAP50",
        "num_recommendations": 5,
    })
    assert normalized["automl_max_recommendations"] == 5
    assert "num_recommendations" not in normalized


def test_alias_conflicting_with_canonical_raises():
    with pytest.raises(ValueError, match="alias"):
        _validate_automl_settings({
            "num_recommendations": 5,
            "automl_max_recommendations": 20,
        })
    # Same value on both spellings is harmless.
    normalized = _validate_automl_settings({
        "num_recommendations": 5,
        "automl_max_recommendations": 5,
    })
    assert normalized["automl_max_recommendations"] == 5


def test_all_known_keys_are_accepted():
    settings = {key: "x" for key in KNOWN_AUTOML_SETTINGS}
    assert _validate_automl_settings(settings) == settings


def test_run_rejects_unknown_settings_before_any_submission(tmp_path):
    from unittest.mock import MagicMock
    from tao_automl.runner import AutoMLRunner
    from test_runner import _write_fake_skill

    runner = AutoMLRunner(
        sdk=MagicMock(), skill_dir=_write_fake_skill(tmp_path), action="train"
    )
    with pytest.raises(ValueError, match="num_recommendation"):
        runner.run(
            image="nvcr.io/test:1",
            automl_settings={
                "algorithm": "bayesian",
                "metric": "accuracy",
                "num_recommendation": 3,
            },
            workspace_path=str(tmp_path / "workspace"),
        )


_CONSUMER_PATTERN = re.compile(
    r'(?:automl_settings|\bsettings|params_dict)\s*(?:\.get\(\s*|\[\s*)"([a-zA-Z_0-9]+)"'
)


def test_known_settings_match_every_consumer_call_site():
    """Drift guard: every key any module reads from the settings dict must be
    registered in KNOWN_AUTOML_SETTINGS — otherwise the validator would reject
    a key the code actually consumes."""
    src = Path(__file__).resolve().parents[1] / "src" / "tao_automl"
    consumed = {}
    for path in src.rglob("*.py"):
        for match in _CONSUMER_PATTERN.finditer(path.read_text()):
            consumed.setdefault(match.group(1), set()).add(path.name)

    # _evaluation_record_path reads its key from an argument, and the
    # effective-batch check looks sample-count spellings up via
    # _SAMPLE_COUNT_KEYS — both invisible to the scan; the validator's
    # constant carries them explicitly.
    from tao_automl.runner import _SAMPLE_COUNT_KEYS
    unscannable = {
        "baseline_record_path", "final_evaluation_record_path",
    } | set(_SAMPLE_COUNT_KEYS)

    unregistered = set(consumed) - KNOWN_AUTOML_SETTINGS
    assert not unregistered, (
        f"settings keys consumed in source but missing from "
        f"KNOWN_AUTOML_SETTINGS: "
        f"{ {k: sorted(consumed[k]) for k in sorted(unregistered)} }"
    )
    assert unscannable <= KNOWN_AUTOML_SETTINGS
    for alias, canonical in AUTOML_SETTING_ALIASES.items():
        assert canonical in KNOWN_AUTOML_SETTINGS
        assert alias not in KNOWN_AUTOML_SETTINGS, (
            "an alias must not also be a known key, or normalization is dead"
        )
