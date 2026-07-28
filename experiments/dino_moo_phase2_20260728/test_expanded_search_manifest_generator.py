#!/usr/bin/env python3

"""Focused self-tests for the frozen expanded-search derivation policy."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import random
import subprocess
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


def corrected_runner_identity(
    runner_path: Path,
    runner_sha256: str,
) -> dict[str, str]:
    return {
        "repository": str(runner_path.parent),
        "relative_path": runner_path.name,
        "head_commit": "1" * 40,
        "git_blob": "2" * 40,
        "sha256": runner_sha256,
    }


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
            "runner_path": Path("/runner/expanded_search_runner.py"),
            "runner_sha256": "e" * 64,
            "corrected_runner_commit_identity": (
                corrected_runner_identity(
                    Path("/runner/expanded_search_runner.py"),
                    "e" * 64,
                )
            ),
            "sensitivity_report_sha256": "d" * 64,
        }
        expected = GENERATOR.build_manifest(frozen, rows, **kwargs)
        shuffled = copy.deepcopy(rows)
        random.Random(314159).shuffle(shuffled)
        actual = GENERATOR.build_manifest(frozen, shuffled, **kwargs)
        self.assertEqual(actual, expected)
        self.assertEqual(actual["manifest_sha256"], expected["manifest_sha256"])
        self.assertEqual(
            actual["derivation"]["runner_path"],
            "/runner/expanded_search_runner.py",
        )
        self.assertEqual(actual["derivation"]["runner_sha256"], "e" * 64)
        self.assertEqual(
            actual["derivation"]["analysis_erratum_contract_sha256"],
            GENERATOR.EXPECTED_ANALYSIS_ERRATUM_CONTRACT_SHA256,
        )
        self.assertEqual(
            actual["derivation"]["post_front_contract_sha256"],
            GENERATOR.EXPECTED_POST_FRONT_CONTRACT_SHA256,
        )
        self.assertEqual(
            actual["post_front_matched_validation"],
            frozen["post_front_matched_validation"],
        )

    def test_runner_source_identity_cannot_be_omitted_or_malformed(self):
        frozen = policy()
        rows = decisions(frozen)
        rows[0]["latency_effect_qualified"] = True
        kwargs = {
            "sensitivity_result_path": Path("/evidence/result.json"),
            "sensitivity_result_sha256": "a" * 64,
            "source_identity": {
                "reference_model_spec": {"num_queries": 594},
                "reference_optimizer": {"lr": 0.1, "weight_decay": 0.01},
            },
            "latency_tolerance": 0.75,
            "policy_path": Path("/policy/policy.json"),
            "policy_sha256": "b" * 64,
            "generator_path": Path("/generator/generator.py"),
            "generator_sha256": "c" * 64,
            "runner_path": Path("/runner/expanded_search_runner.py"),
            "runner_sha256": "not-a-digest",
            "corrected_runner_commit_identity": (
                corrected_runner_identity(
                    Path("/runner/expanded_search_runner.py"),
                    "e" * 64,
                )
            ),
            "sensitivity_report_sha256": "d" * 64,
        }
        with self.assertRaisesRegex(
            GENERATOR.ContractError,
            "expanded runner source SHA256",
        ):
            GENERATOR.build_manifest(frozen, rows, **kwargs)

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
            frozen["sensitivity_evidence_contract"]["manifest_id"],
            "dino_sensitivity_latency_20260728_v2",
        )
        source_manifest = frozen["sensitivity_evidence_contract"][
            "source_manifest"
        ]
        self.assertEqual(
            source_manifest["path"],
            "sensitivity_latency_manifest.v2.json",
        )
        self.assertEqual(
            source_manifest["sha256"],
            (
                "aedc117414b2691c1a70b73fa4e9e0ac"
                "123cb4d20dfd9d25dfe2d4aa490d7655"
            ),
        )
        self.assertEqual(
            source_manifest["supersedes"],
            {
                "manifest_id": "dino_sensitivity_latency_20260728_v1",
                "manifest_sha256": (
                    "c569f858f4513139292d7189ab5e57f8"
                    "97b8794fdbe5b2dcafc45b0efcd663aa"
                ),
                "disposition": "preflight_failed_no_latency_measurements",
            },
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
        analysis_erratum = frozen["sensitivity_evidence_contract"][
            "analysis_erratum"
        ]
        self.assertEqual(
            analysis_erratum["erratum_id"],
            "dino_sensitivity_latency_analysis_erratum_20260728_v1",
        )
        self.assertEqual(
            analysis_erratum["sha256"],
            (
                "8e19287bf2ffd674f62b21cdaf11e000"
                "b0eae1ed8af9d0ada1238491588993f2"
            ),
        )
        self.assertFalse(
            frozen["post_front_matched_validation"][
                "selection_isolation"
            ]["measurements_feed_reselection"]
        )

    def test_erratum_and_post_front_contract_tampering_is_rejected(self):
        frozen = policy()
        frozen["sensitivity_evidence_contract"]["result_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            GENERATOR.ContractError,
            "approved sensitivity analysis result",
        ):
            GENERATOR.validate_policy(frozen)

        frozen = policy()
        frozen["sensitivity_evidence_contract"]["analysis_erratum"][
            "sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(
            GENERATOR.ContractError,
            "analysis erratum preregistration contract",
        ):
            GENERATOR.validate_policy(frozen)

        frozen = policy()
        frozen["post_front_matched_validation"]["allocation_design"][
            "allocation_count"
        ] = 5
        with self.assertRaisesRegex(
            GENERATOR.ContractError,
            "post-front matched-validation contract",
        ):
            GENERATOR.validate_policy(frozen)

    def test_approved_erratum_source_and_result_binding_are_exact(self):
        frozen = policy()
        contract = frozen["sensitivity_evidence_contract"][
            "analysis_erratum"
        ]
        erratum = json.loads(
            (HERE / contract["path"]).read_text(encoding="utf-8")
        )
        measurement = contract["measurement_contract"]
        source_files = contract["source_files"]
        fingerprints = contract["contract_fingerprints"]
        result = {
            "analysis_erratum": {
                "erratum_id": contract["erratum_id"],
                "erratum_path": str((HERE / contract["path"]).resolve()),
                "erratum_sha256": contract["sha256"],
                "reason_code": contract["reason_code"],
                "measurement_manifest_id": measurement["manifest_id"],
                "measurement_manifest_path": str(
                    (HERE / measurement["manifest_path"]).resolve()
                ),
                "measurement_manifest_sha256": measurement[
                    "manifest_sha256"
                ],
                "submission_ledger_path": str(
                    (HERE / measurement["submission_ledger_path"]).resolve()
                ),
                "submission_ledger_sha256": measurement[
                    "submission_ledger_sha256"
                ],
                "original_aggregator_sha256": source_files[
                    "original_aggregator_sha256"
                ],
                "corrected_aggregator_sha256": source_files[
                    "corrected_aggregator_sha256"
                ],
                "measurement_policy_sha256": contract[
                    "unchanged_policy_pins"
                ]["measurement_policy_sha256"],
                "qualification_policy_sha256": contract[
                    "unchanged_policy_pins"
                ]["qualification_policy_sha256"],
                "evidence_acquisition_policy_sha256": fingerprints[
                    "evidence_acquisition_policy_sha256"
                ],
                "sdk_state_inspection_policy_sha256": fingerprints[
                    "sdk_state_inspection_policy_sha256"
                ],
                "measurement_generation_unchanged": True,
                "qualification_policy_unchanged": True,
                "objective_values_altered": False,
                "raw_runtime_string_preserved": True,
                "correction": copy.deepcopy(contract["correction"]),
                "analysis_commit_correction": copy.deepcopy(
                    erratum["analysis_commit_correction"]
                ),
            },
            "analysis_source_checks": {
                "original_aggregator": source_files[
                    "original_aggregator_sha256"
                ],
                "corrected_aggregator": source_files[
                    "corrected_aggregator_sha256"
                ],
                "analysis_erratum": contract["sha256"],
            },
        }
        identity = GENERATOR.validate_analysis_erratum_source(
            frozen,
            result,
            HERE,
            json.loads(
                (HERE / measurement["manifest_path"]).read_text(
                    encoding="utf-8"
                )
            ),
        )
        self.assertEqual(identity["sha256"], contract["sha256"])

        tampered = copy.deepcopy(result)
        tampered["analysis_erratum"]["objective_values_altered"] = True
        with self.assertRaisesRegex(
            GENERATOR.ContractError,
            "approved analysis_erratum identity",
        ):
            GENERATOR.validate_analysis_erratum_source(
                frozen,
                tampered,
                HERE,
                json.loads(
                    (HERE / measurement["manifest_path"]).read_text(
                        encoding="utf-8"
                    )
                ),
            )

    def test_final_approved_sensitivity_result_passes_exact_preflight(self):
        frozen = policy()
        result_path = HERE / frozen["sensitivity_evidence_contract"][
            "result_path"
        ]
        result = GENERATOR.load_json(result_path)
        rows, source, tolerance = GENERATOR.validate_sensitivity_result(
            frozen,
            result,
            result_path=result_path,
            supplied_sha256=GENERATOR.sha256_file(result_path),
            source_base=HERE,
        )
        self.assertEqual(tolerance, 0.73553775)
        self.assertEqual(
            [item["path"] for item in GENERATOR.derive_architecture_axes(
                frozen, rows
            )],
            ["model.enc_layers", "model.dec_layers"],
        )
        self.assertEqual(
            source["analysis_erratum"]["contract_sha256"],
            GENERATOR.EXPECTED_ANALYSIS_ERRATUM_CONTRACT_SHA256,
        )
        self.assertEqual(
            result["report_sha256"],
            frozen["sensitivity_evidence_contract"][
                "result_report_sha256"
            ],
        )

    def test_v2_is_behaviorally_identical_to_byte_pinned_v1(self):
        frozen = policy()
        result_path = HERE / frozen["sensitivity_evidence_contract"][
            "result_path"
        ]
        result = GENERATOR.load_json(result_path)
        rows, source, tolerance = GENERATOR.validate_sensitivity_result(
            frozen,
            result,
            result_path=result_path,
            supplied_sha256=GENERATOR.sha256_file(result_path),
            source_base=HERE,
        )
        runner_path = HERE / "expanded_search_runner.py"
        runner_sha256 = GENERATOR.sha256_file(runner_path)
        repository = Path(
            frozen["frozen_identity"]["source_repositories"]["tao_automl"][
                "path"
            ]
        ).resolve()
        identity = {
            "repository": str(repository),
            "relative_path": runner_path.resolve().relative_to(
                repository
            ).as_posix(),
            "head_commit": subprocess.run(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "git_blob": subprocess.run(
                ["git", "-C", str(repository), "hash-object", str(runner_path)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "sha256": runner_sha256,
        }
        manifest = GENERATOR.build_manifest(
            frozen,
            rows,
            sensitivity_result_path=result_path,
            sensitivity_result_sha256=GENERATOR.sha256_file(result_path),
            source_identity=source,
            latency_tolerance=tolerance,
            policy_path=POLICY_PATH,
            policy_sha256=GENERATOR.sha256_file(POLICY_PATH),
            generator_path=MODULE_PATH,
            generator_sha256=GENERATOR.sha256_file(MODULE_PATH),
            runner_path=runner_path,
            runner_sha256=runner_sha256,
            corrected_runner_commit_identity=identity,
            sensitivity_report_sha256=result["report_sha256"],
        )
        GENERATOR.validate_v2_behavioral_identity(manifest, frozen)
        self.assertEqual(
            manifest["manifest_id"],
            "dino_expanded_search_20260728_v2",
        )
        self.assertEqual(
            GENERATOR.sha256_file(HERE / "expanded_search_manifest.v1.json"),
            (
                "57e331686b8896989263a39f72edb6954"
                "3fc58833f20a1e6e698c31f34d2e8be"
            ),
        )

        tampered = copy.deepcopy(manifest)
        tampered["selection"]["multi_objective_mode"]["rho"] = 0.5
        with self.assertRaisesRegex(
            GENERATOR.ContractError,
            "unchanged behavioral contract selection",
        ):
            GENERATOR.validate_v2_behavioral_identity(tampered, frozen)

    def test_corrected_runner_generation_gate_requires_committed_clean_file(
        self,
    ):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            subprocess.run(
                ["git", "init", "-q", str(repository)],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "config", "user.email", "x@y"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "config", "user.name", "test"],
                check=True,
            )
            runner_path = repository / "expanded_search_runner.py"
            runner_path.write_text("print('v2')\n", encoding="utf-8")
            subprocess.run(
                ["git", "-C", str(repository), "add", runner_path.name],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "commit", "-qm", "runner"],
                check=True,
            )
            frozen = policy()
            frozen["frozen_identity"]["source_repositories"]["tao_automl"][
                "path"
            ] = str(repository)
            identity = GENERATOR.require_corrected_runner_committed(
                frozen,
                runner_path,
            )
            self.assertEqual(
                identity["sha256"],
                GENERATOR.sha256_file(runner_path),
            )
            runner_path.write_text("print('dirty')\n", encoding="utf-8")
            with self.assertRaisesRegex(
                GENERATOR.ContractError,
                "committed and clean",
            ):
                GENERATOR.require_corrected_runner_committed(
                    frozen,
                    runner_path,
                )

    def test_superseded_v1_sensitivity_manifest_cannot_be_reintroduced(self):
        frozen = policy()
        frozen["sensitivity_evidence_contract"]["manifest_id"] = (
            "dino_sensitivity_latency_20260728_v1"
        )
        source_manifest = frozen["sensitivity_evidence_contract"][
            "source_manifest"
        ]
        source_manifest["path"] = "sensitivity_latency_manifest.v1.json"
        source_manifest["sha256"] = (
            "c569f858f4513139292d7189ab5e57f8"
            "97b8794fdbe5b2dcafc45b0efcd663aa"
        )
        with self.assertRaisesRegex(
            GENERATOR.ContractError,
            "pinned sensitivity manifest ID mismatch",
        ):
            GENERATOR.validate_policy(frozen)

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
