#!/usr/bin/env python3

"""Focused self-tests for the frozen expanded-search derivation policy."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import random
import unittest


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "expanded_search_manifest_generator.py"
POLICY_PATH = HERE / "expanded_search_derivation_policy.v1.json"
SPEC = importlib.util.spec_from_file_location(
    "expanded_search_manifest_generator", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


def policy() -> dict:
    value = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    GENERATOR.validate_policy(value)
    return value


def decisions(value: dict) -> list[dict]:
    result = []
    for axis in value["architecture_axis_policy"]["axes"]:
        for level_text, profile_id in axis[
            "non_reference_profile_ids"
        ].items():
            result.append(
                {
                    "profile_id": profile_id,
                    "axis": axis["path"],
                    "level": int(level_text),
                    "support_validity_repeatability_gate": True,
                    "latency_effect_qualified": False,
                    "latency_mode_98pct_suitable": False,
                    "effect_direction": (
                        "uncertain_or_within_practical_band"
                    ),
                    "hierarchical_paired_effect_ci95_ms": [-0.5, 0.5],
                    "effective_noise_floor_ms": 0.75,
                    "feeds_final_selection": False,
                    "winner_selected": False,
                }
            )
    return result


class ExpandedSearchDerivationTests(unittest.TestCase):
    def test_single_qualified_level_includes_entire_axis_domain(self):
        frozen = policy()
        rows = decisions(frozen)
        row = next(
            item
            for item in rows
            if item["axis"] == "model.num_queries" and item["level"] == 450
        )
        row["latency_effect_qualified"] = True
        derived = GENERATOR.derive_architecture_axes(frozen, rows)
        self.assertEqual([item["path"] for item in derived], ["model.num_queries"])
        self.assertEqual(
            derived[0]["preregistered_levels"], [300, 450, 594, 750, 900]
        )
        self.assertEqual(
            derived[0]["search_domain"],
            {
                "representation": "integer_range",
                "valid_min": 300,
                "valid_max": 900,
            },
        )
        self.assertEqual(derived[0]["qualified_non_reference_levels"], [450])

    def test_discrete_axis_uses_all_levels_not_qualified_hull(self):
        frozen = policy()
        rows = decisions(frozen)
        row = next(
            item
            for item in rows
            if item["axis"] == "model.num_select" and item["level"] == 100
        )
        row["latency_effect_qualified"] = True
        derived = GENERATOR.derive_architecture_axes(frozen, rows)
        self.assertEqual(
            derived[0]["search_domain"]["valid_options"], [50, 100, 200, 300]
        )
        self.assertEqual(
            derived[0]["preregistered_levels"], [50, 100, 200, 300]
        )

    def test_no_qualified_axis_blocks(self):
        frozen = policy()
        with self.assertRaisesRegex(
            GENERATOR.ContractError, "expanded search is blocked"
        ):
            GENERATOR.derive_architecture_axes(frozen, decisions(frozen))

    def test_evidence_order_cannot_change_axis_derivation(self):
        frozen = policy()
        rows = decisions(frozen)
        for axis, level in (
            ("model.num_queries", 900),
            ("model.enc_layers", 4),
            ("model.num_select", 50),
        ):
            next(
                row
                for row in rows
                if row["axis"] == axis and row["level"] == level
            )["latency_effect_qualified"] = True
        expected = GENERATOR.derive_architecture_axes(frozen, rows)
        shuffled = copy.deepcopy(rows)
        random.Random(20260728).shuffle(shuffled)
        self.assertEqual(
            GENERATOR.derive_architecture_axes(frozen, shuffled), expected
        )

    def test_complete_manifest_is_order_invariant(self):
        frozen = policy()
        rows = decisions(frozen)
        next(
            row
            for row in rows
            if row["axis"] == "model.enc_layers" and row["level"] == 4
        )["latency_effect_qualified"] = True
        kwargs = {
            "sensitivity_result_path": Path("/evidence/result.json"),
            "sensitivity_result_sha256": "a" * 64,
            "source_identity": {
                "reference_model_spec": {"enc_layers": 6},
                "reference_optimizer": {
                    "lr": 0.00045,
                    "weight_decay": 0.00026967723799334445,
                },
            },
            "latency_tolerance": 0.75,
            "policy_path": Path("/policy/policy.json"),
            "policy_sha256": "b" * 64,
            "generator_path": Path("/generator/generator.py"),
            "generator_sha256": "c" * 64,
            "sensitivity_report_sha256": "d" * 64,
        }
        expected = GENERATOR.build_manifest(frozen, rows, **kwargs)
        shuffled = copy.deepcopy(rows)
        random.Random(314159).shuffle(shuffled)
        actual = GENERATOR.build_manifest(frozen, shuffled, **kwargs)
        self.assertEqual(actual, expected)
        self.assertEqual(actual["manifest_sha256"], expected["manifest_sha256"])

    def test_latency_retention_annotation_is_independent(self):
        frozen = policy()
        rows = decisions(frozen)
        next(
            row
            for row in rows
            if row["axis"] == "model.dec_layers" and row["level"] == 3
        )["latency_effect_qualified"] = True
        expected = GENERATOR.derive_architecture_axes(frozen, rows)
        inverted = copy.deepcopy(rows)
        for row in inverted:
            row["latency_mode_98pct_suitable"] = not row[
                "latency_mode_98pct_suitable"
            ]
        self.assertEqual(
            GENERATOR.derive_architecture_axes(frozen, inverted), expected
        )

    def test_known_axis_level_identity_is_complete_and_order_invariant(self):
        frozen = policy()
        rows = decisions(frozen)
        synthetic_result = {
            "noise_floor": {"effective_noise_floor_ms": 0.75},
            "latency_effect_decisions": rows,
        }
        expected = GENERATOR.normalize_decisions(frozen, synthetic_result)
        shuffled_result = copy.deepcopy(synthetic_result)
        random.Random(17).shuffle(
            shuffled_result["latency_effect_decisions"]
        )
        self.assertEqual(
            GENERATOR.normalize_decisions(frozen, shuffled_result), expected
        )
        missing = copy.deepcopy(synthetic_result)
        missing["latency_effect_decisions"].pop()
        with self.assertRaisesRegex(
            GENERATOR.ContractError, "missing sensitivity decisions"
        ):
            GENERATOR.normalize_decisions(frozen, missing)
        unknown = copy.deepcopy(synthetic_result)
        unknown["latency_effect_decisions"][0]["level"] = 999
        with self.assertRaisesRegex(
            GENERATOR.ContractError, "unknown sensitivity axis/level"
        ):
            GENERATOR.normalize_decisions(frozen, unknown)

    def test_selection_modes_keep_independent_accuracy_policies(self):
        frozen = policy()
        selection = frozen["selection_contract"]
        self.assertEqual(
            frozen["sensitivity_evidence_contract"]["source_manifest"][
                "sha256"
            ],
            (
                "c569f858f4513139292d7189ab5e57f8"
                "97b8794fdbe5b2dcafc45b0efcd663aa"
            ),
        )
        self.assertEqual(
            selection["latency_mode"]["latency_accuracy_retention"],
            {
                "type": "relative",
                "retained_fraction": 0.98,
                "reference": "accuracy_winner",
            },
        )
        self.assertIsNone(
            selection["multi_objective_mode"][
                "multi_objective_min_accuracy"
            ]
        )

    def test_manual_true_flag_is_rejected(self):
        with self.assertRaisesRegex(
            GENERATOR.ContractError, "forbidden true audit flag"
        ):
            GENERATOR.validate_false_audit_flags(
                {"nested": {"manual_override_used": True}}
            )

    def test_qualification_boolean_must_match_preregistered_ci_rule(self):
        frozen = policy()
        rows = decisions(frozen)
        rows[0]["latency_effect_qualified"] = True
        synthetic_result = {
            "noise_floor": {"effective_noise_floor_ms": 0.75},
            "latency_effect_decisions": rows,
        }
        with self.assertRaisesRegex(
            GENERATOR.ContractError,
            "direction-agnostic CI qualification",
        ):
            GENERATOR.normalize_decisions(frozen, synthetic_result)

    def test_reliably_slower_level_qualifies_direction_agnostically(self):
        frozen = policy()
        rows = decisions(frozen)
        target = next(
            row
            for row in rows
            if row["axis"] == "model.dec_layers" and row["level"] == 5
        )
        target.update(
            {
                "latency_effect_qualified": True,
                "effect_direction": "slower",
                "hierarchical_paired_effect_ci95_ms": [0.8, 1.1],
                "latency_reduction_qualified": False,
                "future_shared_multi_objective_eligible": True,
            }
        )
        synthetic_result = {
            "noise_floor": {"effective_noise_floor_ms": 0.75},
            "latency_effect_decisions": rows,
        }
        normalized = GENERATOR.normalize_decisions(
            frozen, synthetic_result
        )
        derived = GENERATOR.derive_architecture_axes(frozen, normalized)
        self.assertEqual(
            [axis["path"] for axis in derived], ["model.dec_layers"]
        )
        self.assertEqual(
            derived[0]["qualified_non_reference_levels"], [5]
        )
        self.assertEqual(derived[0]["preregistered_levels"], [3, 4, 5, 6])


if __name__ == "__main__":
    unittest.main(verbosity=2)
