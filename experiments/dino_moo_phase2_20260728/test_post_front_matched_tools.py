"""Focused contract tests for expanded-search post-front matched validation.

These tests intentionally use synthetic, in-memory evidence.  They never call
the manifest generator's ``generate`` entry point, reserve the real runtime,
or invoke either the TAO SDK or SLURM.
"""

from __future__ import annotations

import ast
import copy
import inspect
from itertools import combinations
import json
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace
from typing import Any

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import post_front_matched_aggregator as aggregator  # noqa: E402
import post_front_matched_block_runner as block_runner  # noqa: E402
import post_front_matched_launcher as launcher  # noqa: E402
import post_front_matched_manifest_generator as generator  # noqa: E402


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _candidate(candidate_id: str) -> dict[str, Any]:
    model = {
        "backbone": "resnet_50",
        "num_queries": 100,
        "candidate": candidate_id,
    }
    return {
        "candidate_id": candidate_id,
        "candidate_table_record_sha256": SHA_A,
        "selection_audit_sha256": SHA_B,
        "global_pareto_rank": 0,
        "global_dominated_by": [],
        "search_seed": 314159,
        "training_seed": 314159,
        "rec_id": 0,
        "train_job_id": "train_job",
        "specs": {"model.num_queries": 100},
        "selection_time_objective_values": {
            "mAP50": 0.5,
            "latency_ms": 5.0,
        },
        "resolved_model_spec": model,
        "resolved_model_spec_sha256": generator.sha256_value(model),
        "checkpoint": {
            "path": f"/lustre/checkpoints/{candidate_id}.pth",
            "sha256": SHA_C,
        },
    }


def _latency_protocol() -> dict[str, Any]:
    return {
        "warmup_iterations": 50,
        "timed_iterations": 100,
        "repeated_rounds": 5,
        "preloaded_batches": 16,
        "batch_size_per_gpu": 1,
        "fixed_preprocessed_shapes": {
            "model_input": [1, 4, 800, 1333],
            "image_tensor": [1, 3, 800, 1333],
            "padding_mask": [1, 1, 800, 1333],
        },
        "precision": "fp32",
        "tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "benchmark_seed": 20260727,
        "tail_percentile": 95.0,
        "bootstrap_resamples": 5000,
        "bootstrap_confidence_level": 0.95,
        "bootstrap_seed": 424242,
        "synchronization": "cuda_sync_each_sample_and_nccl_barrier",
        "timed_scope": "model_forward_plus_dino_gpu_postprocess",
        "validity_thresholds": {
            "max_robust_cv": 0.1,
            "max_round_median_range_fraction": 0.05,
            "max_absolute_round_drift_fraction": 0.05,
            "max_device_median_range_fraction": 0.05,
            "max_bootstrap_ci_width_fraction": 0.03,
        },
    }


def _minimal_manifest(
    candidate_ids: list[str],
    *,
    runtime_path: Path | None = None,
) -> dict[str, Any]:
    candidate_ids = sorted(candidate_ids)
    candidates = [_candidate(candidate_id) for candidate_id in candidate_ids]
    schedule = generator.build_schedule(candidate_ids)
    output_contract = {
        "root_expression": "$TAO_RESULTS_ROOT/$TAO_JOB_ID",
        "sdk_job_scoped": True,
        "relative_layout": (
            "dino_moo_phase2_20260728/post_front_matched/"
            "<manifest_id>/<allocation_id>"
        ),
    }
    runtime = {
        "sqsh_path": "/lustre/images/dino.sqsh",
        "sqsh_sha256": SHA_A,
        "partition": "polar3",
        "account": "tao",
        "num_nodes": 1,
        "gpu_count": 8,
        "required_gpu_name": "NVIDIA A100-SXM4-80GB",
        "required_compute_capability": "8.0",
        "required_total_memory_bytes": 85_052_784_640,
        "required_torch": "2.7.0",
        "torch_version_match": "major_minor_patch",
        "required_cuda": "12.8",
        "required_cudnn": 90_700,
        "precision": "fp32",
        "tf32": False,
        "sdk_path": "/localhome/local-rarunachalam/tao-sdk",
        "sdk_branch": "rarunachalam/test",
        "sdk_commit": "0" * 40,
        "secrets_env_path": "/localhome/local-rarunachalam/.tao/config.env",
        "local_runtime_path": str(
            (
                runtime_path
                if runtime_path is not None
                else launcher.DEFAULT_RUNTIME
            ).resolve()
        ),
        "image_is_prebuilt_sqsh": True,
        "sdk_sqsh_conversion_enabled": False,
        "slurm_use_requeue": False,
        "slurm_time_hours": 4.0,
        "slurm_timeout_hours": 3.8,
        "submission_api": "tao_sdk.platforms.slurm.SlurmSDK.create_job",
        "output_contract": output_contract,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_id": "dino_expanded_post_front_matched_20260728_v1",
        "status": "immutable_ready_to_launch",
        "scope": copy.deepcopy(generator.EXPECTED_SCOPE),
        "feeds_final_selection": False,
        "feeds_reselection": False,
        "manual_candidate_addition_or_removal_permitted": False,
        "manual_winner_override_permitted": False,
        "selection_time_objective_replacement_permitted": False,
        "candidate_derivation": {
            "source": (
                "independent pinned tao_automl selector replay over every "
                "successful expanded_candidate_table row"
            ),
            "cross_check_source": "expanded_combined_selection.candidates",
            "predicate": (
                "recomputed valid is true and recomputed global "
                "pareto_rank equals zero"
            ),
            "canonical_order": "ascending UTF-8 candidate_id",
            "candidate_count": len(candidate_ids),
            "candidate_ids": candidate_ids,
            "candidate_set_sha256": generator.sha256_value(candidate_ids),
            "records_source": "expanded_candidate_table rows",
            "manual_filtering_used": False,
            "winner_identity_used": False,
            "objective_values_used_for_schedule": False,
            "selector_replay_proof": {
                "global_rank_zero_candidate_ids": candidate_ids,
                "global_rank_zero_candidate_set_sha256": (
                    generator.sha256_value(candidate_ids)
                ),
                "order_independent": True,
                "all_candidate_audits_exact_match": True,
                "candidate_table_audits_exact_match": True,
                "combined_analysis_exact_match": True,
                "global_rank_zero_front_exact_match": True,
            },
        },
        "selection_snapshot": {
            "selections": {
                "accuracy": {"winner_id": candidate_ids[0]},
                "latency": {"winner_id": candidate_ids[-1]},
                "multi_objective": {"winner_id": candidate_ids[0]},
            },
            "selection_authority": {
                "module": "tao_automl.selection",
                "function": "analyze_archive",
                "manual_override_used": False,
            },
            "preserved_unchanged": True,
        },
        "candidates": candidates,
        "schedule": schedule,
        "runtime": runtime,
        "latency_protocol": _latency_protocol(),
        "paired_analysis": {
            "bootstrap_resamples": 10_000,
            "bootstrap_confidence_level": 0.95,
            "bootstrap_seed": 20_260_728,
            "practical_tolerance_ms": (
                generator.EXPECTED_PRACTICAL_TOLERANCE_MS
            ),
        },
        "selection_isolation": {
            "measurements_feed_reselection": False,
            "winner_reselection_permitted": False,
            "original_selection_time_measurements_replaced": False,
            "algorithm_selected_candidate_overridden": False,
            "allowed_use": "stability analysis and hypothesis verdict only",
        },
        "incomplete_allocation_policy": (
            "Exclude the entire allocation and rerun the complete front "
            "under a new TAO job ID; never combine a partial block."
        ),
        "source_artifacts": {
            "dino_latency_benchmark": {"sha256": SHA_A},
            "latency_stats": {"sha256": SHA_C},
            "post_front_tools": {
                launcher.BLOCK_RUNNER.name: {"sha256": SHA_B},
                launcher.AGGREGATOR.name: {
                    "path": str(launcher.AGGREGATOR),
                    "sha256": SHA_A,
                    "git_blob": "0" * 40,
                    "head_git_blob": "0" * 40,
                },
            },
        },
    }
    manifest["manifest_sha256"] = generator.sha256_value(manifest)
    return manifest


def _selection_audit(
    candidate_id: str,
    *,
    valid: bool,
    rank: int,
    dominated_by: list[str],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "valid": valid,
        "pareto_rank": rank,
        "dominated_by": dominated_by,
    }


def _candidate_table_row(
    audit: dict[str, Any],
    *,
    status: str = "success",
) -> dict[str, Any]:
    candidate_id = audit["candidate_id"]
    model = {"backbone": "resnet_50", "candidate": candidate_id}
    return {
        "candidate_id": candidate_id,
        "status": status,
        "selection_audit": copy.deepcopy(audit),
        "resolved_model_spec": model,
        "resolved_model_spec_sha256": generator.sha256_value(model),
        "checkpoint": {
            "path": f"/lustre/checkpoints/{candidate_id}.pth",
            "sha256": SHA_C,
        },
        "objective_values": {
            "mAP50": 0.5,
            "latency_ms": 5.0,
            "latency_p95_ms": 5.5,
            "latency_ci95_low": 4.9,
            "latency_ci95_high": 5.1,
        },
        "search_seed": 314159,
        "training_seed": 314159,
        "rec_id": 0,
        "train_job_id": "train_job",
        "specs": {"model.num_queries": 100},
    }


def _front_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    audits = [
        _selection_audit(
            "z_front", valid=True, rank=0, dominated_by=[]
        ),
        _selection_audit(
            "d_dominated",
            valid=True,
            rank=1,
            dominated_by=["a_front"],
        ),
        _selection_audit(
            "i_invalid", valid=False, rank=0, dominated_by=[]
        ),
        _selection_audit(
            "a_front", valid=True, rank=0, dominated_by=[]
        ),
    ]
    rows = [_candidate_table_row(audit) for audit in reversed(audits)]
    return {"candidates": audits}, {
        "candidate_count": len(rows),
        "rows": rows,
    }


def _measurements(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for allocation in manifest["schedule"]["allocations"]:
        allocation_index = allocation["allocation_index"]
        for candidate_index, candidate_id in enumerate(
            manifest["candidate_derivation"]["candidate_ids"]
        ):
            median = 4.0 + candidate_index * 0.9 + allocation_index * 0.05
            result.append(
                {
                    "allocation_id": allocation["allocation_id"],
                    "candidate_id": candidate_id,
                    "median_ms": median,
                    "p95_ms": median + 0.8,
                }
            )
    return result


def _rehash_plan(plan: dict[str, Any]) -> None:
    plan.pop("block_plan_sha256", None)
    plan["block_plan_sha256"] = generator.sha256_value(plan)


def _rehash_manifest(manifest: dict[str, Any]) -> None:
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = generator.sha256_value(manifest)


def _write_dry_run(
    runtime_dir: Path,
    manifest: dict[str, Any],
    manifest_file_sha256: str,
) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "dry_run.json").write_text(
        json.dumps(
            {
                "status": "dry_run_validated_not_launched",
                "manifest": {
                    "whole_file_sha256": manifest_file_sha256,
                },
                "schedule_sha256": manifest["schedule"]["schedule_sha256"],
                "candidate_ids": manifest["candidate_derivation"][
                    "candidate_ids"
                ],
                "submission_ready": True,
            }
        ),
        encoding="utf-8",
    )


def _synthetic_submissions(
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    submissions = []
    for index, allocation in enumerate(manifest["schedule"]["allocations"]):
        staged = {
            f"configs/{candidate_id}.yaml": SHA_A
            for candidate_id in allocation["candidate_order"]
        }
        submissions.append(
            {
                "allocation_id": allocation["allocation_id"],
                "allocation_index": allocation["allocation_index"],
                "design_row_index": allocation["design_row_index"],
                "candidate_order": allocation["candidate_order"],
                "candidate_count": len(manifest["candidates"]),
                "block_plan_sha256": SHA_A,
                "command_sha256": SHA_B,
                "staging_bundle_sha256": SHA_A,
                "staging_bundle_json_sha256": SHA_B,
                "staging_file_sha256": staged,
                "tao_job_id": f"tao-job-{index}",
                "slurm_job_id": str(1000 + index),
                "retry_count": 0,
                "failed_slurm_job_ids": [],
                "launch_uncertain": False,
                "sdk_results_uri": f"lustre:///results/tao-job-{index}",
                "feeds_final_selection": False,
                "feeds_reselection": False,
            }
        )
    return submissions


def test_expanded_manifest_validation_accepts_only_pinned_v2() -> None:
    v2_path = HERE / "expanded_search_manifest.v2.json"
    v2 = generator.load_json(v2_path)
    generator.validate_expanded_manifest(v2, generator.sha256_file(v2_path))

    v1_path = HERE / "expanded_search_manifest.v1.json"
    v1 = generator.load_json(v1_path)
    with pytest.raises(generator.ContractError, match="manifest ID"):
        generator.validate_expanded_manifest(
            v1,
            generator.sha256_file(v1_path),
        )


def test_generator_cannot_create_post_front_contract_before_archive(
    tmp_path: Path,
) -> None:
    expanded_path = HERE / "expanded_search_manifest.v2.json"
    missing = tmp_path / "not_yet_sealed.json"
    with pytest.raises(generator.ContractError, match="cannot load"):
        generator.generate(
            expanded_manifest_path=expanded_path,
            expanded_manifest_sha256=generator.sha256_file(expanded_path),
            combined_path=missing,
            combined_sha256=SHA_A,
            table_path=missing,
            table_sha256=SHA_A,
            integrity_path=missing,
            integrity_sha256=SHA_A,
        )
    assert not (tmp_path / "post_front_matched_manifest.v1.json").exists()


def test_rank_zero_derivation_is_complete_canonical_and_order_independent() -> None:
    combined, table = _front_evidence()
    replay = {"analysis": copy.deepcopy(combined)}
    expected = generator.derive_candidate_records(combined, table, replay)
    assert [item["candidate_id"] for item in expected] == [
        "a_front",
        "z_front",
    ]
    assert all(item["global_pareto_rank"] == 0 for item in expected)
    assert all(item["global_dominated_by"] == [] for item in expected)

    reordered_combined = copy.deepcopy(combined)
    reordered_combined["candidates"].reverse()
    reordered_table = copy.deepcopy(table)
    reordered_table["rows"].reverse()
    assert (
        generator.derive_candidate_records(
            reordered_combined,
            reordered_table,
            replay,
        )
        == expected
    )


def test_rank_zero_derivation_fails_closed_on_join_or_front_tampering() -> None:
    combined, table = _front_evidence()
    joined_tamper = copy.deepcopy(table)
    row = next(
        item
        for item in joined_tamper["rows"]
        if item["candidate_id"] == "a_front"
    )
    row["selection_audit"]["valid"] = False
    with pytest.raises(generator.ContractError, match="selection audit"):
        generator.derive_candidate_records(
            combined,
            joined_tamper,
            {"analysis": copy.deepcopy(combined)},
        )

    dominated_front = copy.deepcopy(combined)
    audit = next(
        item
        for item in dominated_front["candidates"]
        if item["candidate_id"] == "a_front"
    )
    audit["dominated_by"] = ["z_front"]
    with pytest.raises(generator.ContractError, match="dominated_by"):
        generator.derive_candidate_records(
            dominated_front,
            table,
            {"analysis": copy.deepcopy(dominated_front)},
        )


def test_production_selector_replay_recomputes_front_and_rejects_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expanded = generator.load_json(HERE / "expanded_search_manifest.v2.json")
    repository = HERE.parent.parent
    (
        parse_objective_config,
        _,
        _,
        _,
    ) = generator._load_selection_api(repository)
    settings = generator.selector_settings(expanded)
    objective = parse_objective_config(copy.deepcopy(settings))
    raw = [
        ("candidate_accuracy", 0.90, 6.0),
        ("candidate_middle", 0.86, 4.8),
        ("candidate_latency", 0.80, 3.5),
    ]
    selector_inputs = [
        SimpleNamespace(
            id=candidate_id,
            specs={"model.num_queries": 100 + index},
            status="success",
            objective_values={
                "mAP50": accuracy,
                "latency_ms": latency,
                "latency_p95_ms": latency + 0.5,
                "latency_ci95_low": latency - 0.05,
                "latency_ci95_high": latency + 0.05,
            },
        )
        for index, (candidate_id, accuracy, latency) in enumerate(raw)
    ]
    analysis = objective.analyze_archive(selector_inputs).to_dict()
    audits = {
        item["candidate_id"]: item for item in analysis["candidates"]
    }
    table = {
        "successful_count": len(selector_inputs),
        "rows": [
            {
                "candidate_id": candidate.id,
                "status": "success",
                "specs": copy.deepcopy(candidate.specs),
                "objective_values": copy.deepcopy(
                    candidate.objective_values
                ),
                "selection_audit": copy.deepcopy(audits[candidate.id]),
            }
            for candidate in selector_inputs
        ],
    }
    selection_path = repository / "src" / "tao_automl" / "selection.py"
    provenance = {
        "repository": str(repository),
        "branch": "rarunachalam/pre-platform-sdk-removal-20260714",
        "head_commit": "0" * 40,
        "commit_policy": "required_ancestor",
        "selection_core_commit": "1" * 40,
        "selection_core_is_ancestor": True,
        "source_files": {
            "src/tao_automl/selection.py": {
                "path": str(selection_path),
                "sha256": generator.sha256_file(selection_path),
            }
        },
        "callables": {},
    }
    monkeypatch.setattr(
        generator,
        "selection_stack_provenance",
        lambda _: copy.deepcopy(provenance),
    )
    authority = {
        "module": "tao_automl.selection",
        "function": "analyze_archive",
        "source_path": str(selection_path),
        "source_sha256": generator.sha256_file(selection_path),
        "manual_override_used": False,
    }
    combined = copy.deepcopy(analysis)
    combined["search"] = {"successful_candidates": len(selector_inputs)}
    combined["selection_authority"] = copy.deepcopy(authority)
    integrity = {
        "selection": {
            "settings": copy.deepcopy(settings),
            "authority": copy.deepcopy(authority),
        }
    }
    replay = generator.replay_and_validate_selector(
        expanded_manifest=expanded,
        combined=combined,
        table=table,
        integrity=integrity,
    )
    assert replay["proof"]["order_independent"] is True
    assert replay["proof"]["all_candidate_audits_exact_match"] is True
    assert replay["proof"]["global_rank_zero_candidate_ids"] == [
        item["candidate_id"]
        for item in sorted(
            (
                audit
                for audit in analysis["candidates"]
                if audit["valid"] is True and audit["pareto_rank"] == 0
            ),
            key=lambda item: item["candidate_id"],
        )
    ]

    tampered = copy.deepcopy(table)
    tampered["rows"][0]["objective_values"]["mAP50"] = 0.01
    with pytest.raises(generator.ContractError, match="recomputed/combined"):
        generator.replay_and_validate_selector(
            expanded_manifest=expanded,
            combined=combined,
            table=tampered,
            integrity=integrity,
        )


@pytest.mark.parametrize("candidate_count", range(1, 11))
def test_williams_rows_and_six_allocation_projection(
    candidate_count: int,
) -> None:
    rows = generator.williams_design_rows(candidate_count)
    expected_row_count = (
        candidate_count if candidate_count % 2 == 0 else 2 * candidate_count
    )
    assert len(rows) == expected_row_count
    assert all(sorted(row) == list(range(candidate_count)) for row in rows)

    base = generator.williams_base_row(candidate_count)
    assert rows[:candidate_count] == [
        [(index + shift) % candidate_count for index in base]
        for shift in range(candidate_count)
    ]
    if candidate_count % 2:
        assert rows[candidate_count:] == [
            list(reversed(row)) for row in rows[:candidate_count]
        ]

    candidate_ids = [
        f"candidate_{index:02d}" for index in range(candidate_count)
    ]
    schedule = generator.build_schedule(candidate_ids)
    assert len(schedule["allocations"]) == 6
    assert schedule["selected_design_row_indices"] == [
        (allocation_index * len(rows)) // 6
        for allocation_index in range(6)
    ]
    for allocation in schedule["allocations"]:
        assert sorted(allocation["candidate_order"]) == candidate_ids
    assert schedule["audit"][
        "every_allocation_is_complete_permutation"
    ] is True
    assert schedule["audit"]["allocation_count"] == 6
    assert schedule["audit"][
        "allocation_complete_permutation_flags"
    ] == [True] * 6
    measured_maximum = max(
        schedule["audit"]["per_candidate_position_count_imbalance"].values()
    )
    assert (
        schedule["audit"]["maximum_per_candidate_position_count_imbalance"]
        == measured_maximum
    )
    assert schedule["audit"]["position_count_imbalance_within_one"] is (
        measured_maximum <= 1
    )
    assert schedule["row_selection_rule"] == (
        "allocation k uses design row floor(k*R/6), where R is the "
        "complete Williams design-row count"
    )
    unhashed = copy.deepcopy(schedule)
    claimed = unhashed.pop("schedule_sha256")
    assert generator.sha256_value(unhashed) == claimed
    assert generator.build_schedule(candidate_ids) == schedule

    if candidate_count == 6:
        for counts in schedule["audit"]["position_counts"].values():
            assert counts == {str(position): 1 for position in range(6)}


def test_schedule_requires_canonical_unique_candidate_ids() -> None:
    with pytest.raises(generator.ContractError, match="canonical"):
        generator.build_schedule(["candidate_b", "candidate_a"])
    with pytest.raises(generator.ContractError, match="unique"):
        generator.build_schedule(["candidate_a", "candidate_a"])


def test_manifest_contract_is_bound_to_prebuilt_sqsh_and_no_reselection() -> None:
    manifest = _minimal_manifest(["candidate_a", "candidate_b"])
    launcher.validate_manifest_contract(manifest)

    wrong_runtime = copy.deepcopy(manifest)
    wrong_runtime["runtime"]["local_runtime_path"] = "/tmp/unbound"
    with pytest.raises(launcher.ContractError, match="runtime path"):
        launcher.validate_manifest_contract(wrong_runtime)

    reselection = copy.deepcopy(manifest)
    reselection["selection_isolation"]["measurements_feed_reselection"] = True
    with pytest.raises(launcher.ContractError, match="selection isolation"):
        launcher.validate_manifest_contract(reselection)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("runtime", "sqsh_path"), "/lustre/images/unpinned.sqsh"),
        (("runtime", "sqsh_sha256"), "d" * 64),
        (("runtime", "sdk_path"), "/localhome/unpinned/tao-sdk"),
        (("runtime", "sdk_branch"), "rarunachalam/unpinned"),
        (("runtime", "sdk_commit"), "1" * 40),
        (("runtime", "account"), "different-account"),
        (("runtime", "partition"), "different-partition"),
        (("runtime", "required_gpu_name"), "NVIDIA H100 80GB HBM3"),
        (("runtime", "required_compute_capability"), "9.0"),
        (("runtime", "required_total_memory_bytes"), 1),
        (("runtime", "required_torch"), "9.9.9"),
        (("runtime", "required_cuda"), "99.0"),
        (("runtime", "required_cudnn"), 1),
        (
            (
                "latency_protocol",
                "validity_thresholds",
                "max_robust_cv",
            ),
            999.0,
        ),
        (("latency_protocol", "benchmark_seed"), 1),
        (("latency_protocol", "synchronization"), "none"),
    ],
)
def test_exact_reconstruction_rejects_self_rehashed_launch_semantic_drift(
    path: tuple[str, ...],
    replacement: Any,
) -> None:
    reconstructed = _minimal_manifest(["candidate_a", "candidate_b"])
    launcher.require_exact_reconstructed_manifest(
        reconstructed,
        copy.deepcopy(reconstructed),
    )

    tampered = copy.deepcopy(reconstructed)
    target: dict[str, Any] = tampered
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    _rehash_manifest(tampered)

    # A self-consistent hash is not an authority for launch semantics.  These
    # fields deliberately exercise gaps in the fragmentary schema validator;
    # the pinned-source reconstruction is the fail-closed authority.
    generator.validate_internal_digest(
        tampered,
        "manifest_sha256",
        "self-rehashed synthetic manifest",
    )
    launcher.validate_manifest_contract(tampered)
    with pytest.raises(
        launcher.ContractError,
        match="deterministic pinned-source reconstruction",
    ):
        launcher.require_exact_reconstructed_manifest(
            tampered,
            reconstructed,
        )


def test_final_source_validation_and_all_main_modes_keep_exact_reconstruction(
) -> None:
    def named_calls(function: Any) -> dict[str, list[ast.Call]]:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        calls: dict[str, list[ast.Call]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                continue
            calls.setdefault(name, []).append(node)
        return calls

    validation_calls = named_calls(launcher.validate_final_source_evidence)
    assert len(validation_calls["build_manifest"]) == 1
    assert len(
        validation_calls["require_exact_reconstructed_manifest"]
    ) == 1
    assert validation_calls["build_manifest"][0].lineno < validation_calls[
        "require_exact_reconstructed_manifest"
    ][0].lineno

    main_calls = named_calls(launcher.main)
    assert len(main_calls["validate_final_source_evidence"]) == 1
    validation_line = main_calls["validate_final_source_evidence"][0].lineno
    assert validation_line < main_calls["generate_configs"][0].lineno
    for operation in (
        "resume_incomplete_submission",
        "replacement_submission",
        "submit_all",
    ):
        assert len(main_calls[operation]) == 1
        assert validation_line < main_calls[operation][0].lineno


def test_launch_reservation_binds_runtime_and_blocks_duplicate_launch(
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "bound_runtime"
    manifest = _minimal_manifest(
        ["candidate_a", "candidate_b"],
        runtime_path=runtime_dir,
    )
    _write_dry_run(runtime_dir, manifest, SHA_A)

    with pytest.raises(launcher.ContractError, match="runtime directory"):
        launcher.reserve_launch(
            manifest,
            SHA_A,
            tmp_path / "different_runtime",
            {"validated": True},
        )

    marker = launcher.reserve_launch(
        manifest,
        SHA_A,
        runtime_dir,
        {"validated": True},
    )
    assert marker == runtime_dir / launcher.LAUNCH_CONTRACT_NAME
    contract = generator.load_json(marker)
    generator.validate_internal_digest(
        contract,
        "contract_sha256",
        "launch contract",
    )
    assert contract["runtime_path"] == str(runtime_dir.resolve())
    assert contract["feeds_reselection"] is False

    with pytest.raises(launcher.ContractError, match="not fresh"):
        launcher.reserve_launch(
            manifest,
            SHA_A,
            runtime_dir,
            {"validated": True},
        )


def test_launch_reservation_rejects_dry_run_for_another_manifest(
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "runtime"
    manifest = _minimal_manifest(["candidate_a"], runtime_path=runtime_dir)
    _write_dry_run(runtime_dir, manifest, SHA_B)
    with pytest.raises(launcher.ContractError, match="manifest SHA256"):
        launcher.reserve_launch(
            manifest,
            SHA_A,
            runtime_dir,
            {"validated": True},
        )
    assert not (runtime_dir / launcher.LAUNCH_CONTRACT_NAME).exists()


def test_failed_allocation_supersession_is_audited_and_partial_data_rejected(
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "runtime"
    manifest = _minimal_manifest(
        ["candidate_a", "candidate_b"],
        runtime_path=runtime_dir,
    )
    runtime_dir.mkdir(parents=True)
    base_source_checks = {"validated": True}
    launch_contract = launcher.launch_contract_payload(
        manifest,
        SHA_A,
        runtime_dir,
        base_source_checks,
    )
    launch_path = runtime_dir / launcher.LAUNCH_CONTRACT_NAME
    launcher.atomic_json(launch_path, launch_contract)
    source_checks = {
        **base_source_checks,
        "launch_contract": {
            "path": str(launch_path),
            "whole_file_sha256": generator.sha256_file(launch_path),
            "internal_sha256": launch_contract["contract_sha256"],
        },
    }
    submissions = _synthetic_submissions(manifest)
    ledger_path = runtime_dir / "block_submissions.json"
    initial = launcher.submission_ledger_payload(
        manifest,
        SHA_A,
        submissions,
        status="complete",
        source_checks=source_checks,
    )
    launcher.atomic_json(ledger_path, initial)
    initial_whole = generator.sha256_file(ledger_path)
    aggregator.load_ledger(
        ledger_path,
        initial_whole,
        manifest,
        SHA_A,
        base_source_checks,
    )
    recovery_events = [
        {
            "event_index": 0,
            "allocation_id": submissions[0]["allocation_id"],
            "command_sha256": submissions[0]["command_sha256"],
            "reason": "durably_terminal_submission_not_reused",
            "tao_job_id": "recovered-terminal-tao-job",
            "slurm_job_id": "1999",
            "sdk_status": "Error",
            "submission_attempted": True,
            "launch_uncertain": False,
            "reconciliation": {
                "sdk_job_ids_before": [],
                "sdk_job_ids_observed": ["recovered-terminal-tao-job"],
                "sdk_job_id_delta": ["recovered-terminal-tao-job"],
                "delta_is_exactly_one": True,
                "command_sha256": submissions[0]["command_sha256"],
                "decision": (
                    "terminal_job_excluded_and_complete_block_resubmitted"
                ),
                "duplicate_submission_permitted": False,
            },
            "partial_measurements_reused": False,
            "feeds_final_selection": False,
            "feeds_reselection": False,
        }
    ]
    recovered = launcher.submission_ledger_payload(
        manifest,
        SHA_A,
        submissions,
        status="complete",
        source_checks=source_checks,
        submission_recovery_events=recovery_events,
    )
    launcher.atomic_json(ledger_path, recovered)
    loaded_recovered, _ = aggregator.load_ledger(
        ledger_path,
        generator.sha256_file(ledger_path),
        manifest,
        SHA_A,
        base_source_checks,
    )
    assert loaded_recovered["submission_recovery_events"] == recovery_events

    recovery_events[0]["partial_measurements_reused"] = True
    invalid_recovery = launcher.submission_ledger_payload(
        manifest,
        SHA_A,
        submissions,
        status="complete",
        source_checks=source_checks,
        submission_recovery_events=recovery_events,
    )
    launcher.atomic_json(ledger_path, invalid_recovery)
    with pytest.raises(
        aggregator.ContractError,
        match="partial_measurements_reused",
    ):
        aggregator.load_ledger(
            ledger_path,
            generator.sha256_file(ledger_path),
            manifest,
            SHA_A,
            base_source_checks,
        )
    launcher.atomic_json(ledger_path, initial)

    prior = copy.deepcopy(submissions[0])
    replacement = copy.deepcopy(prior)
    replacement["tao_job_id"] = "tao-job-replacement"
    replacement["slurm_job_id"] = "2000"
    effective = [replacement, *submissions[1:]]
    intent = {
        "schema_version": 1,
        "intent_id": "dino_post_front_complete_allocation_replacement_v1",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": SHA_A,
        "allocation_id": prior["allocation_id"],
        "command_sha256": prior["command_sha256"],
        "ledger_revision": 2,
        "parent_ledger_whole_file_sha256": initial_whole,
        "parent_ledger_internal_sha256": initial["ledger_sha256"],
        "prior_tao_job_id": prior["tao_job_id"],
        "prior_slurm_job_id": prior["slurm_job_id"],
        "prior_sdk_status": "Error",
        "replacement_basis": "sdk_terminal_failure",
        "invalidation_evidence": None,
        "partial_measurements_reused": False,
        "feeds_final_selection": False,
        "feeds_reselection": False,
    }
    intent["intent_sha256"] = generator.sha256_value(intent)
    intent_path = runtime_dir / "replacement_intent.r002.json"
    launcher.atomic_json(intent_path, intent)
    history = [
        {
            "allocation_id": prior["allocation_id"],
            "reason": "durable_terminal_incomplete_allocation",
            "incomplete_allocation_policy": manifest[
                "incomplete_allocation_policy"
            ],
            "prior_sdk_status": "Error",
            "prior_sdk_message": "terminal failure",
            "prior_submission": prior,
            "parent_ledger_whole_file_sha256": initial_whole,
            "parent_ledger_internal_sha256": initial["ledger_sha256"],
            "replacement_basis": "sdk_terminal_failure",
            "invalidation_evidence": None,
            "replacement_intent": {
                "path": str(intent_path),
                "whole_file_sha256": generator.sha256_file(intent_path),
                "internal_sha256": intent["intent_sha256"],
            },
            "partial_measurements_reused": False,
        }
    ]
    revised = launcher.submission_ledger_payload(
        manifest,
        SHA_A,
        effective,
        status="complete",
        source_checks=source_checks,
        ledger_revision=2,
        superseded_submissions=history,
        parent_ledger_sha256=initial_whole,
    )
    launcher.atomic_json(ledger_path, revised)
    revised_whole = generator.sha256_file(ledger_path)
    aggregator.load_ledger(
        ledger_path,
        revised_whole,
        manifest,
        SHA_A,
        base_source_checks,
    )

    history[0]["partial_measurements_reused"] = True
    invalid = launcher.submission_ledger_payload(
        manifest,
        SHA_A,
        effective,
        status="complete",
        source_checks=source_checks,
        ledger_revision=2,
        superseded_submissions=history,
        parent_ledger_sha256=initial_whole,
    )
    launcher.atomic_json(ledger_path, invalid)
    with pytest.raises(aggregator.ContractError, match="partial measurement"):
        aggregator.load_ledger(
            ledger_path,
            generator.sha256_file(ledger_path),
            manifest,
            SHA_A,
            base_source_checks,
        )


def test_block_plan_rejects_digest_and_rehashed_semantic_tampering() -> None:
    manifest = _minimal_manifest(["candidate_a", "candidate_b"])
    configs = {
        candidate_id: f"candidate: {candidate_id}\n".encode()
        for candidate_id in manifest["candidate_derivation"]["candidate_ids"]
    }
    allocation = manifest["schedule"]["allocations"][0]
    plan = launcher.build_block_plan(
        manifest,
        SHA_A,
        allocation,
        configs,
    )
    block_runner.validate_plan(plan)

    digest_tamper = copy.deepcopy(plan)
    digest_tamper["candidates"][0]["position"] = 99
    with pytest.raises(ValueError, match="canonical digest"):
        block_runner.validate_plan(digest_tamper)

    semantic_tamper = copy.deepcopy(plan)
    semantic_tamper["feeds_reselection"] = True
    _rehash_plan(semantic_tamper)
    with pytest.raises(ValueError, match="validation-only"):
        block_runner.validate_plan(semantic_tamper)

    run_label_tamper = copy.deepcopy(plan)
    run_label_tamper["candidates"][0]["run_label"] = "manual_candidate"
    _rehash_plan(run_label_tamper)
    with pytest.raises(ValueError, match="run label"):
        block_runner.validate_plan(run_label_tamper)


def test_aggregation_rejects_incomplete_jobs_before_reading_results() -> None:
    manifest = _minimal_manifest(["candidate_a", "candidate_b"])
    incomplete_jobs = [
        {"allocation_id": f"allocation_{index}", "complete": True}
        for index in range(5)
    ]
    with pytest.raises(aggregator.ContractError, match="six complete"):
        aggregator.aggregate_bundles(
            manifest,
            SHA_A,
            incomplete_jobs,
            {},
        )

    six_with_failure = [
        {"allocation_id": f"allocation_{index}", "complete": index != 5}
        for index in range(6)
    ]
    with pytest.raises(aggregator.ContractError, match="six complete"):
        aggregator.aggregate_bundles(
            manifest,
            SHA_A,
            six_with_failure,
            {},
        )


def test_comparison_rejects_incomplete_matched_matrix() -> None:
    manifest = _minimal_manifest(
        ["candidate_a", "candidate_b", "candidate_c"]
    )
    measurements = _measurements(manifest)
    measurements.pop()
    with pytest.raises(aggregator.ContractError, match="measurement count"):
        aggregator.comparative_analysis(manifest, measurements)

    duplicate = _measurements(manifest)
    duplicate.append(copy.deepcopy(duplicate[0]))
    with pytest.raises(
        aggregator.ContractError,
        match="measurement count",
    ):
        aggregator.comparative_analysis(manifest, duplicate)


def test_paired_bootstrap_is_deterministic_at_preregistered_10k() -> None:
    values = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4]
    first = aggregator.paired_bootstrap_ci(
        values,
        resamples=10_000,
        confidence=0.95,
        seed=20_260_728,
    )
    second = aggregator.paired_bootstrap_ci(
        values,
        resamples=10_000,
        confidence=0.95,
        seed=20_260_728,
    )
    assert first == second
    assert first[0] <= first[1]
    assert first[0] <= -0.65 <= first[1]


def test_completed_sacct_nodelist_binds_to_allocation_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_remote_output(command: str, *, timeout: int = 900) -> str:
        del timeout
        if command.startswith("sacct "):
            return (
                "12345|COMPLETED|0:0|dgx-node[01741]\n"
            )
        if command.startswith("scontrol show hostnames "):
            return "DGX-NODE01741.example.test\n"
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(aggregator, "remote_output", fake_remote_output)
    accounting = aggregator.slurm_accounting(["12345"])
    row = accounting["12345"]
    assert row["expanded_node_hostnames"] == [
        "DGX-NODE01741.example.test"
    ]
    assert row["normalized_node_hostname"] == "dgx-node01741"
    evidence = aggregator.validate_scheduler_hostname_binding(
        row,
        "dgx-node01741",
    )
    assert evidence["status"] == "pass"

    with pytest.raises(
        aggregator.ContractError,
        match="NodeList/allocation hostname",
    ):
        aggregator.validate_scheduler_hostname_binding(
            row,
            "dgx-node99999",
        )


def test_aggregation_revalidates_sdk_and_imported_latency_module_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _minimal_manifest(["candidate_a"])
    latency_path = Path(aggregator.latency_stats_module.__file__).resolve()
    manifest["source_artifacts"]["latency_stats"] = {
        "path": str(latency_path),
        "sha256": generator.sha256_file(latency_path),
    }
    sdk_path = tmp_path / "tao-sdk"
    sdk_path.mkdir()
    manifest["runtime"]["sdk_path"] = str(sdk_path)
    manifest["runtime"]["sdk_commit"] = "d" * 40
    manifest["runtime"]["sdk_branch"] = "rarunachalam/release-7.1.0"
    sdk_dirty = False

    def fake_git_value(_repo: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return "d" * 40
        if arguments == ("branch", "--show-current"):
            return "rarunachalam/release-7.1.0"
        if arguments == (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ):
            return " M tao_sdk/platforms/slurm.py" if sdk_dirty else ""
        raise AssertionError(f"unexpected git arguments: {arguments}")

    monkeypatch.setattr(launcher, "git_value", fake_git_value)
    evidence = aggregator.aggregation_runtime_provenance(manifest)
    assert evidence["latency_stats"] == {
        "module": "tao_automl.latency_stats",
        "imported_file": str(latency_path),
        "expected_file": str(latency_path),
        "sha256": generator.sha256_file(latency_path),
        "imported_file_matches_manifest": True,
        "imported_sha256_matches_manifest": True,
    }
    assert evidence["tao_sdk"]["worktree_clean"] is True

    sdk_dirty = True
    with pytest.raises(aggregator.ContractError, match="must remain clean"):
        aggregator.aggregation_runtime_provenance(manifest)


def test_aggregation_rejects_latency_module_path_or_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _minimal_manifest(["candidate_a"])
    latency_path = Path(aggregator.latency_stats_module.__file__).resolve()
    manifest["source_artifacts"]["latency_stats"] = {
        "path": str(latency_path),
        "sha256": "0" * 64,
    }
    with pytest.raises(
        aggregator.ContractError,
        match="latency_stats module SHA256",
    ):
        aggregator.aggregation_runtime_provenance(manifest)

    shadow = tmp_path / "latency_stats.py"
    shadow.write_bytes(latency_path.read_bytes())
    manifest["source_artifacts"]["latency_stats"] = {
        "path": str(shadow),
        "sha256": generator.sha256_file(shadow),
    }
    with pytest.raises(
        aggregator.ContractError,
        match="latency_stats module path",
    ):
        aggregator.aggregation_runtime_provenance(manifest)


def test_within_allocation_p95_cluster_bootstrap_is_deterministic() -> None:
    protocol = aggregator.LatencyProtocol(
        warmup_iterations=0,
        timed_iterations=3,
        repeated_rounds=2,
        tail_percentile=95.0,
        bootstrap_resamples=100,
        bootstrap_confidence_level=0.95,
        bootstrap_seed=17,
    )
    samples = {
        0: {"0": [4.0, 4.1, 4.2], "1": [4.2, 4.3, 4.4]},
        1: {"0": [4.1, 4.2, 4.3], "1": [4.3, 4.4, 4.5]},
    }
    first = aggregator.cluster_bootstrap_tail_ci(samples, protocol)
    second = aggregator.cluster_bootstrap_tail_ci(samples, protocol)
    assert first == second
    assert first[0] <= first[1]


@pytest.mark.parametrize(
    ("delta", "ci", "point", "interval"),
    [
        (
            -1.0,
            [-1.1, -0.8],
            "first_practically_faster",
            "entirely_below_negative_tolerance",
        ),
        (
            1.0,
            [0.8, 1.1],
            "second_practically_faster",
            "entirely_above_positive_tolerance",
        ),
        (
            0.0,
            [
                -generator.EXPECTED_PRACTICAL_TOLERANCE_MS,
                generator.EXPECTED_PRACTICAL_TOLERANCE_MS,
            ],
            "practically_equivalent",
            "entirely_within_practical_tolerance",
        ),
        (
            -generator.EXPECTED_PRACTICAL_TOLERANCE_MS,
            [-0.8, 0.0],
            "practically_equivalent",
            "crosses_a_practical_tolerance_boundary",
        ),
        (
            generator.EXPECTED_PRACTICAL_TOLERANCE_MS,
            [0.0, 0.8],
            "practically_equivalent",
            "crosses_a_practical_tolerance_boundary",
        ),
    ],
)
def test_practical_tolerance_boundaries(
    delta: float,
    ci: list[float],
    point: str,
    interval: str,
) -> None:
    classification = aggregator.classify_difference(
        delta,
        ci,
        generator.EXPECTED_PRACTICAL_TOLERANCE_MS,
    )
    assert classification == {
        "point_classification": point,
        "descriptive_bootstrap_interval_classification": interval,
    }


def test_directional_claim_requires_all_six_differences_beyond_tolerance() -> None:
    tolerance = generator.EXPECTED_PRACTICAL_TOLERANCE_MS
    counterexample = [-2.0, -1.9, -1.8, -1.7, -1.6, 0.0]
    evidence = aggregator.directional_pairwise_evidence(
        counterexample,
        tolerance=tolerance,
        confidence=0.95,
    )
    assert evidence["first_faster_exact_test_passes"] is True
    assert evidence["all_six_beyond_negative_tolerance"] is False
    assert evidence["directional_claim"] == "no_stable_directional_claim"
    assert evidence["scope"] == "pairwise_only"
    assert evidence["simultaneous_order_inference_permitted"] is False

    unanimous = aggregator.directional_pairwise_evidence(
        [-2.0, -1.9, -1.8, -1.7, -1.6, -1.5],
        tolerance=tolerance,
        confidence=0.95,
    )
    assert unanimous["all_six_beyond_negative_tolerance"] is True
    assert unanimous["first_faster_test"]["permutation_count"] == 64
    assert unanimous["first_faster_test"]["p_value_one_sided"] == 1 / 64
    assert unanimous["directional_claim"] == "first_stably_faster"


def test_comparative_analysis_covers_every_unordered_pair() -> None:
    candidate_ids = [
        "candidate_a",
        "candidate_b",
        "candidate_c",
        "candidate_d",
    ]
    manifest = _minimal_manifest(candidate_ids)
    analysis = aggregator.comparative_analysis(
        manifest,
        _measurements(manifest),
    )
    pairs = analysis["all_pairwise_comparisons"]
    assert len(pairs) == len(list(combinations(candidate_ids, 2)))
    assert {
        (item["first_candidate_id"], item["second_candidate_id"])
        for item in pairs
    } == set(combinations(candidate_ids, 2))
    assert all(
        len(item["paired_median_differences_ms"]) == 6 for item in pairs
    )
    assert all(
        "p95_descriptive_bootstrap_interval_classification" in item
        for item in pairs
    )
    assert all(
        item["median_bootstrap_ci_is_descriptive_only"] is True
        and item["p95_bootstrap_ci_is_descriptive_only"] is True
        and item["pairwise_directional_evidence"]["scope"]
        == "pairwise_only"
        for item in pairs
    )
    assert all(
        item["allocation_ids"]
        == [
            allocation["allocation_id"]
            for allocation in manifest["schedule"]["allocations"]
        ]
        for item in pairs
    )
    expected_bootstrap = {
        "unit": "allocation",
        "resamples": 10_000,
        "confidence_level": 0.95,
        "seed": 20_260_728,
        "statistic": "median of allocation-paired differences",
    }
    assert {
        key: analysis["paired_bootstrap"][key]
        for key in expected_bootstrap
    } == expected_bootstrap
    assert analysis["directional_inference"] == {
        "method": (
            "exact one-sided paired sign-flip permutation test after "
            "shifting by the practical-tolerance boundary"
        ),
        "allocation_count": 6,
        "permutation_count": 64,
        "alpha": pytest.approx(0.05),
        "additional_unanimity_requirement": (
            "all six allocation-paired differences must be strictly "
            "beyond the claimed +/- practical-tolerance boundary"
        ),
        "scope": "pairwise_only",
        "multiplicity_adjustment": "none",
        "simultaneous_total_order_inference_permitted": False,
    }
    assert analysis["stable_total_order_claim_applicable"] is False
    assert analysis["descriptive_order_is_a_stable_total_order"] is False
    assert analysis["stable_ordering_claims_scope"] == "pairwise_only"


def test_validation_artifacts_preserve_selection_and_never_reselect() -> None:
    manifest = _minimal_manifest(
        ["candidate_a", "candidate_b", "candidate_c"]
    )
    configs = {
        candidate_id: f"candidate: {candidate_id}\n".encode()
        for candidate_id in manifest["candidate_derivation"]["candidate_ids"]
    }
    plan = launcher.build_block_plan(
        manifest,
        SHA_A,
        manifest["schedule"]["allocations"][0],
        configs,
    )
    for key in (
        "feeds_final_selection",
        "feeds_reselection",
        "manual_candidate_addition_or_removal_permitted",
        "winner_override_permitted",
        "selection_time_objective_replacement_permitted",
    ):
        assert plan[key] is False

    launch_contract = launcher.launch_contract_payload(
        manifest,
        SHA_A,
        Path(manifest["runtime"]["local_runtime_path"]),
        {"validated": True},
    )
    assert launch_contract["feeds_final_selection"] is False
    assert launch_contract["feeds_reselection"] is False
    assert launch_contract["manual_winner_override_permitted"] is False

    ledger = launcher.submission_ledger_payload(
        manifest,
        SHA_A,
        [],
        status="complete",
        source_checks={"validated": True},
    )
    assert ledger["feeds_final_selection"] is False
    assert ledger["feeds_reselection"] is False
    assert ledger["selection_time_objective_replacement_permitted"] is False
    assert ledger["ledger_revision"] == 1
    assert ledger["superseded_submissions"] == []
    assert ledger["pending_submission"] is None
    generator.validate_internal_digest(
        ledger,
        "ledger_sha256",
        "synthetic submission ledger",
    )

    report = aggregator.build_final_report(
        manifest,
        SHA_A,
        ledger,
        SHA_B,
        [],
        _measurements(manifest),
        {"complete_block_contract": "pass"},
    )
    assert report["original_selection_snapshot"] == manifest[
        "selection_snapshot"
    ]
    assert report["selection_isolation"] == {
        "frozen_archive_selector_replay_performed_during_source_validation": (
            False
        ),
        "selector_replay_result_used_only_for_candidate_set_integrity": False,
        "selector_replay_proof_sha256": None,
        "postfront_measurements_loaded_after_selector_replay": False,
        "selector_invoked_on_postfront_measurements": False,
        "measurements_feed_selection": False,
        "measurements_feed_reselection": False,
        "selection_time_objectives_replaced": False,
        "algorithm_selected_candidate_overridden": False,
        "allowed_use": "stability analysis and hypothesis verdict only",
    }
    assert "winner" not in json.dumps(report["analysis"]).lower()
    assert report["submission_recovery_events"] == []


def test_postfront_analysis_does_not_invoke_selector_or_mutate_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _minimal_manifest(
        ["candidate_a", "candidate_b", "candidate_c"]
    )
    measurements = _measurements(manifest)
    frozen_selection = copy.deepcopy(manifest["selection_snapshot"])
    frozen_measurements = copy.deepcopy(measurements)

    def forbidden_selector(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("post-front measurements reached a selector")

    monkeypatch.setattr(
        generator,
        "validate_completed_archive",
        forbidden_selector,
    )
    analysis = aggregator.comparative_analysis(manifest, measurements)
    assert manifest["selection_snapshot"] == frozen_selection
    assert measurements == frozen_measurements
    assert "winner" not in json.dumps(analysis).lower()
    assert "selected_candidate" not in json.dumps(analysis).lower()


def test_report_distinguishes_archive_replay_from_postfront_reselection() -> None:
    manifest = _minimal_manifest(["candidate_a", "candidate_b"])
    ledger = launcher.submission_ledger_payload(
        manifest,
        SHA_A,
        [],
        status="complete",
        source_checks={"selector_replay_proof_sha256": SHA_C},
    )
    report = aggregator.build_final_report(
        manifest,
        SHA_A,
        ledger,
        SHA_B,
        [],
        _measurements(manifest),
        {"complete_block_contract": "pass"},
    )
    isolation = report["selection_isolation"]
    assert (
        isolation[
            "frozen_archive_selector_replay_performed_during_source_validation"
        ]
        is True
    )
    assert isolation["selector_replay_proof_sha256"] == SHA_C
    assert isolation["selector_invoked_on_postfront_measurements"] is False
    assert isolation["measurements_feed_reselection"] is False
