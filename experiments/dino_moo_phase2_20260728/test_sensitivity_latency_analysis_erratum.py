"""Focused tests for the analysis-only sensitivity-latency v2 erratum."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import sensitivity_latency_aggregate as original
import sensitivity_latency_aggregate_erratum as erratum


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "sensitivity_latency_manifest.v2.json"
ERRATUM = HERE / "sensitivity_latency_analysis_erratum.v1.json"
LEDGER = HERE / "runtime/sensitivity_latency_v2/block_submissions.json"
LEDGER_SHA256 = (
    "b1c170c0d4697463d171cbeca3e4adcbd34cc1cb7429c236f48b58c46c3b6d54"
)


def runtime_contract() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text())["runtime_contract"]


def hardware(torch_version: str) -> dict[str, Any]:
    expected = runtime_contract()
    return {
        "devices": [
            {
                "index": rank,
                "name": expected["required_gpu_name"],
                "compute_capability": expected[
                    "required_compute_capability"
                ],
                "total_memory_bytes": expected[
                    "required_total_memory_bytes"
                ],
            }
            for rank in range(original.EXPECTED_RANKS)
        ],
        "runtime": {
            "torch": torch_version,
            "cuda": expected["required_cuda"],
            "cudnn": expected["required_cudnn"],
        },
    }


def rank_record(torch_version: str) -> dict[str, Any]:
    expected = runtime_contract()
    return {
        "runtime": {
            "torch": torch_version,
            "cuda": expected["required_cuda"],
            "cudnn": expected["required_cudnn"],
        }
    }


@pytest.mark.parametrize(
    "version",
    [
        "2.11.0",
        "2.11.0a0+a6c236b9fd.nv26.03.46836102",
        "2.11.0+cu132",
    ],
)
def test_build_suffix_is_accepted_at_allocation_and_rank(version: str) -> None:
    expected = runtime_contract()
    assert erratum.validate_allocation_hardware(
        hardware(version), expected
    )[0] == version
    assert erratum.validate_rank_runtime(
        rank_record(version), {"runtime_contract": expected}
    )[0] == version


@pytest.mark.parametrize("version", ["2.10.0", "2.11.1", "3.11.0"])
def test_numeric_release_drift_is_rejected_at_allocation_and_rank(
    version: str,
) -> None:
    expected = runtime_contract()
    with pytest.raises(ValueError, match="allocation runtime mismatch"):
        erratum.validate_allocation_hardware(hardware(version), expected)
    with pytest.raises(ValueError, match="rank runtime mismatch"):
        erratum.validate_rank_runtime(
            rank_record(version), {"runtime_contract": expected}
        )


def validate_real_contract(
    *,
    erratum_path: Path = ERRATUM,
    erratum_sha256: str | None = None,
    manifest_path: Path = MANIFEST,
    ledger_path: Path = LEDGER,
    corrected_source_path: Path = erratum.CORRECTED_SOURCE_PATH,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return erratum.validate_analysis_erratum(
        erratum_path,
        erratum_sha256 or erratum.sha256_file(erratum_path),
        manifest_path,
        ledger_path,
        LEDGER_SHA256,
        corrected_source_path=corrected_source_path,
    )


def test_real_erratum_binds_manifest_ledger_sources_and_policies() -> None:
    payload, manifest, identity = validate_real_contract()
    assert payload["objective_values_altered"] is False
    assert manifest["manifest_id"] == "dino_sensitivity_latency_20260728_v2"
    assert identity["measurement_manifest_sha256"] == (
        "aedc117414b2691c1a70b73fa4e9e0ac123cb4d20dfd9d25dfe2d4aa490d7655"
    )
    assert identity["submission_ledger_sha256"] == LEDGER_SHA256
    assert identity["objective_values_altered"] is False


def test_erratum_file_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    payload = json.loads(ERRATUM.read_text())
    payload["reason"] = "tampered"
    tampered = tmp_path / ERRATUM.name
    tampered.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="erratum digest mismatch"):
        validate_real_contract(
            erratum_path=tampered,
            erratum_sha256=erratum.sha256_file(ERRATUM),
        )


def test_manifest_tampering_is_rejected(tmp_path: Path) -> None:
    manifest_payload = json.loads(MANIFEST.read_text())
    manifest_payload["latency_protocol"]["timed_iterations"] += 1
    tampered_manifest = tmp_path / "manifest.json"
    tampered_manifest.write_text(json.dumps(manifest_payload))
    copied_ledger = tmp_path / "ledger.json"
    copied_ledger.write_bytes(LEDGER.read_bytes())

    contract_payload = json.loads(ERRATUM.read_text())
    contract_payload["measurement_contract"]["manifest_path"] = "manifest.json"
    contract_payload["measurement_contract"][
        "submission_ledger_path"
    ] = "ledger.json"
    copied_contract = tmp_path / "erratum.json"
    copied_contract.write_text(json.dumps(contract_payload))
    with pytest.raises(ValueError, match="measurement manifest mismatch"):
        erratum.validate_analysis_erratum(
            copied_contract,
            erratum.sha256_file(copied_contract),
            tampered_manifest,
            copied_ledger,
            LEDGER_SHA256,
        )


def test_corrected_source_tampering_is_rejected(tmp_path: Path) -> None:
    tampered_source = tmp_path / "aggregate_erratum.py"
    tampered_source.write_bytes(
        erratum.CORRECTED_SOURCE_PATH.read_bytes() + b"\n# tampered\n"
    )
    with pytest.raises(
        ValueError, match="corrected aggregator source mismatch"
    ):
        validate_real_contract(corrected_source_path=tampered_source)


def test_erratum_source_pin_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    payload = json.loads(ERRATUM.read_text())
    payload["source_pins"]["corrected_aggregator_sha256"] = "0" * 64
    payload["measurement_contract"]["manifest_path"] = str(MANIFEST)
    payload["measurement_contract"]["submission_ledger_path"] = str(LEDGER)
    tampered = tmp_path / "erratum.json"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(
        ValueError, match="corrected aggregator source mismatch"
    ):
        erratum.validate_analysis_erratum(
            tampered,
            erratum.sha256_file(tampered),
            MANIFEST,
            LEDGER,
            LEDGER_SHA256,
        )


def test_expected_remote_inventory_is_exact_and_discovery_free() -> None:
    (
        contract,
        _profiles,
        schedules,
        _artifacts,
        _accuracies,
        jobs,
        plans,
        _results,
    ) = synthetic_inputs()
    anchor = erratum.EXPECTED_EVIDENCE_ACQUISITION_POLICY[
        "remote_results_anchor"
    ]
    for job in jobs:
        job["sdk_job_scoped_result_root"] = (
            f"{anchor}/{job['tao_job_id']}"
        )
    expected = erratum.build_expected_evidence_files(
        contract, schedules, jobs, plans
    )
    assert len(expected) == 1017
    assert len({item["remote_path"] for item in expected}) == 1017
    assert sum(item["kind"] == "allocation_result" for item in expected) == 9
    assert sum(item["kind"] == "rank_result" for item in expected) == 1008
    assert all(item["remote_path"].startswith(anchor + "/") for item in expected)


def small_snapshot_inputs() -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    anchor = erratum.EXPECTED_EVIDENCE_ACQUISITION_POLICY[
        "remote_results_anchor"
    ]
    payloads = {
        f"{anchor}/job-a/allocation_result.json": b'{"status":"success"}\n',
        f"{anchor}/job-a/profiles/p/tao/latency/rank_0.json": (
            b'{"samples_ms":[[1.0]]}\n'
        ),
    }
    expected = [
        {
            "remote_path": path,
            "relative_path": erratum.validate_remote_path(path, anchor),
            "kind": "allocation_result" if index == 0 else "rank_result",
            "allocation_id": "allocation",
            "tao_job_id": "tao",
            "profile_id": None if index == 0 else "p",
            "rank": None if index == 0 else 0,
        }
        for index, path in enumerate(payloads)
    ]
    return expected, payloads


def configure_local_snapshot_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payloads: dict[str, bytes],
) -> None:
    monkeypatch.setattr(erratum, "HERE", tmp_path)

    def remote_inventory(
        expected: list[dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        return {
            item["remote_path"]: {
                "path": item["remote_path"],
                "size_bytes": len(payloads[item["remote_path"]]),
                "sha256": hashlib.sha256(
                    payloads[item["remote_path"]]
                ).hexdigest(),
                "file_type": "regular_non_symlink",
                "stat_fingerprint": {
                    "device": 1,
                    "inode": index,
                    "mode": 0o100444,
                    "mtime_ns": 1,
                    "ctime_ns": 1,
                },
            }
            for index, item in enumerate(expected)
        }, {"transport": "test", "requested_file_count": len(expected)}

    def transfer(
        relative_paths: list[str],
        remote_anchor: str,
        destination: Path,
    ) -> dict[str, Any]:
        by_relative = {
            erratum.validate_remote_path(path, remote_anchor): value
            for path, value in payloads.items()
        }
        for relative in relative_paths:
            target = destination / Path(
                *erratum.PurePosixPath(relative).parts
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(by_relative[relative])
        return {
            "executed": bool(relative_paths),
            "requested_file_count": len(relative_paths),
            "stdout_sha256": "test",
            "stderr_sha256": "test",
        }

    monkeypatch.setattr(erratum, "fetch_remote_inventory", remote_inventory)
    monkeypatch.setattr(erratum, "rsync_missing_files", transfer)


def test_snapshot_fetch_and_byte_identical_resume_are_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected, payloads = small_snapshot_inputs()
    configure_local_snapshot_transport(monkeypatch, tmp_path, payloads)
    snapshot_root = (
        tmp_path / "runtime/sensitivity_latency_v2/evidence_snapshot"
    )
    first = erratum.acquire_evidence_snapshot(expected, snapshot_root)
    assert first.report["transferred_file_count"] == 2
    assert first.report["existing_identical_file_count"] == 0
    assert first.report["overwrite_performed"] is False
    assert all(
        item["remote_sha256"] == item["local_sha256"]
        for item in first.report["inventory"]
    )

    def no_transfer(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("byte-identical resume must not transfer")

    monkeypatch.setattr(erratum, "rsync_missing_files", no_transfer)
    resumed = erratum.acquire_evidence_snapshot(expected, snapshot_root)
    assert resumed.report["transferred_file_count"] == 0
    assert resumed.report["existing_identical_file_count"] == 2


def test_snapshot_refuses_overwrite_and_extra_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected, payloads = small_snapshot_inputs()
    configure_local_snapshot_transport(monkeypatch, tmp_path, payloads)
    snapshot_root = (
        tmp_path / "runtime/sensitivity_latency_v2/evidence_snapshot"
    )
    completed = erratum.acquire_evidence_snapshot(expected, snapshot_root)
    first_local = next(iter(completed.by_remote_path.values()))
    first_local.chmod(0o644)
    first_local.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        erratum.acquire_evidence_snapshot(expected, snapshot_root)

    other_root = tmp_path / "runtime/sensitivity_latency_v2/extra_snapshot"
    other_root.mkdir(parents=True)
    (other_root / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError, match="unexpected snapshot file"):
        erratum.acquire_evidence_snapshot(expected, other_root)


def test_snapshot_rejects_duplicate_and_missing_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected, payloads = small_snapshot_inputs()
    configure_local_snapshot_transport(monkeypatch, tmp_path, payloads)
    duplicate = [expected[0], {**expected[1], "relative_path": expected[0][
        "relative_path"
    ]}]
    with pytest.raises(ValueError, match="duplicate expected snapshot"):
        erratum.acquire_evidence_snapshot(
            duplicate,
            tmp_path / "runtime/sensitivity_latency_v2/duplicate",
        )

    def empty_transfer(
        relative_paths: list[str],
        _remote_anchor: str,
        _destination: Path,
    ) -> dict[str, Any]:
        return {
            "executed": True,
            "requested_file_count": len(relative_paths),
            "stdout_sha256": "test",
            "stderr_sha256": "test",
        }

    monkeypatch.setattr(erratum, "rsync_missing_files", empty_transfer)
    with pytest.raises(RuntimeError, match="did not produce every missing"):
        erratum.acquire_evidence_snapshot(
            expected,
            tmp_path / "runtime/sensitivity_latency_v2/missing",
        )


def synthetic_inputs() -> tuple[Any, ...]:
    expected = runtime_contract()
    profile_ids = [f"profile_{index:02d}" for index in range(14)]
    profiles = [
        {
            "profile_id": profile_id,
            "axis": "axis",
            "level": index,
            "resolved_model_spec_sha256": f"model-{index}",
        }
        for index, profile_id in enumerate(profile_ids)
    ]
    schedules = []
    jobs = []
    plans = {}
    artifacts = {}
    accuracies = {}
    results = {}
    for allocation_index in range(9):
        allocation_id = f"allocation_{allocation_index:02d}"
        seed = allocation_index
        result_root = f"/results/{allocation_id}"
        schedules.append(
            {
                "allocation_id": allocation_id,
                "seed": seed,
                "repeat_index": allocation_index % 3,
                "williams_row_index": allocation_index,
                "profile_order": profile_ids,
            }
        )
        jobs.append(
            {
                "allocation_id": allocation_id,
                "tao_job_id": f"tao-{allocation_index}",
                "slurm_job_id": str(1000 + allocation_index),
                "sdk_job_scoped_result_root": result_root,
                "scheduler": {
                    "expanded_nodes": [f"node-{allocation_index}"],
                    "node_list": f"node-{allocation_index}",
                },
            }
        )
        plan_profiles = []
        runs = []
        verified = {}
        for position, profile_id in enumerate(profile_ids):
            run_label = f"run-{profile_id}"
            config_path = f"/configs/{allocation_id}/{profile_id}.yaml"
            config_sha = f"config-{allocation_index}-{position}"
            checkpoint = f"/checkpoints/{seed}/{profile_id}.pth"
            checkpoint_sha = f"checkpoint-{seed}-{position}"
            artifacts[(seed, profile_id)] = {
                "checkpoint_path": checkpoint,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_source_profile_id": profile_id,
                "resolved_model_spec_sha256": f"model-{position}",
            }
            accuracies[(seed, profile_id)] = {
                "mAP50": 0.5 + position / 1000.0
            }
            plan_profiles.append(
                {
                    "profile_id": profile_id,
                    "run_label": run_label,
                    "config_path": config_path,
                    "config_sha256": config_sha,
                }
            )
            verified[profile_id] = config_sha
            runs.append(
                {
                    "profile_id": profile_id,
                    "position": position,
                    "status": "success",
                    "exit_code": 0,
                    "seed": seed,
                    "run_label": run_label,
                    "config_sha256": config_sha,
                    "checkpoint_path": checkpoint,
                    "checkpoint_sha256": checkpoint_sha,
                    "checkpoint_source_profile_id": profile_id,
                    "resolved_model_spec_sha256": f"model-{position}",
                    "raw_samples_dir": (
                        f"{result_root}/profiles/{run_label}/"
                        f"tao-{allocation_index}/latency"
                    ),
                }
            )
        plans[allocation_id] = {
            "block_plan_sha256": f"plan-{allocation_index}",
            "profiles": plan_profiles,
        }
        results[allocation_id] = {
            "schema_version": 1,
            "status": "success",
            "manifest_id": "manifest",
            "manifest_sha256": "manifest-sha",
            "checkpoint_artifact_sha256": "checkpoint-artifact-sha",
            "schedule_sha256": "schedule-sha",
            "allocation_id": allocation_id,
            "seed": seed,
            "repeat_index": allocation_index % 3,
            "williams_row_index": allocation_index,
            "block_plan_sha256": f"plan-{allocation_index}",
            "tao_job_id": f"tao-{allocation_index}",
            "sdk_job_scoped_result_root": result_root,
            "feeds_final_selection": False,
            "manual_promotion_permitted": False,
            "hostname": f"node-{allocation_index}",
            "output_contract": {
                "root_env": "TAO_RESULTS_ROOT",
                "job_scope_env": "TAO_JOB_ID",
                "root": result_root,
            },
            "hardware": hardware(expected["required_torch"]),
            "profile_runs": runs,
            "verified_config_sha256": verified,
        }
    contract = {
        "manifest_id": "manifest",
        "design": {"schedule_sha256": "schedule-sha"},
        "runtime_contract": expected,
    }
    return (
        contract,
        profiles,
        schedules,
        artifacts,
        accuracies,
        jobs,
        plans,
        results,
    )


def test_erratum_does_not_alter_any_objective_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        contract,
        profiles,
        schedules,
        artifacts,
        accuracies,
        jobs,
        plans,
        results,
    ) = synthetic_inputs()

    monkeypatch.setattr(
        original,
        "latency_protocol",
        lambda _contract: SimpleNamespace(repeated_rounds=1),
    )
    monkeypatch.setattr(
        original,
        "result_path_for_job",
        lambda job, _contract, _block: Path(
            f"/results/{job['allocation_id']}/allocation_result.json"
        ),
    )

    def read_hashed_json(_path: Path, label: str) -> tuple[Any, str]:
        if label.endswith(" result"):
            allocation_id = label.removesuffix(" result")
            return copy.deepcopy(results[allocation_id]), f"result-{allocation_id}"
        rank = int(label.rsplit("_", 1)[1])
        profile_id = label.split("/", 2)[1]
        return {
            "runtime": {
                "torch": runtime_contract()["required_torch"],
                "cuda": runtime_contract()["required_cuda"],
                "cudnn": runtime_contract()["required_cudnn"],
            },
            "samples_ms": [[float(rank + int(profile_id[-2:]))]],
        }, f"rank-{label}"

    monkeypatch.setattr(original, "read_hashed_json", read_hashed_json)
    monkeypatch.setattr(
        original,
        "validate_rank_record",
        lambda _record, *, rank, **_kwargs: (
            f"input-{rank}",
            ("python", "2.11.0", "13.2", 92000),
            f"gpu-{rank}",
        ),
    )

    def aggregate_samples(
        samples: dict[int, dict[str, list[float]]],
        _protocol: Any,
    ) -> SimpleNamespace:
        values = [
            value
            for ranks in samples.values()
            for rank_values in ranks.values()
            for value in rank_values
        ]
        point = sum(values) / len(values)
        return SimpleNamespace(
            median_ms=point,
            tail_latency_ms=point + 1.0,
            bootstrap_median_ci_ms=(point - 0.1, point + 0.1),
            mad_ms=0.01,
            iqr_ms=0.02,
            robust_cv=0.001,
            round_median_range_ms=0.0,
            round_drift_ms=0.0,
            device_median_range_ms=7.0,
            raw_sample_count_total=len(values),
            samples_per_device=1,
            is_valid=True,
            invalid_reasons=(),
            validity_reason="valid",
        )

    monkeypatch.setattr(
        original, "aggregate_synchronized_latency", aggregate_samples
    )

    old_measurements, _, _ = original.aggregate_job_results(
        contract,
        profiles,
        schedules,
        artifacts,
        accuracies,
        jobs,
        plans,
        "manifest-sha",
        "checkpoint-artifact-sha",
    )
    remote_to_local = {}
    for job, schedule in zip(jobs, schedules, strict=True):
        remote_result = Path(
            f"/results/{job['allocation_id']}/allocation_result.json"
        )
        remote_to_local[str(remote_result)] = Path(
            f"/snapshot/{job['allocation_id']}/allocation_result.json"
        )
        for run in results[schedule["allocation_id"]]["profile_runs"]:
            for rank in range(original.EXPECTED_RANKS):
                remote_rank = Path(run["raw_samples_dir"]) / (
                    f"rank_{rank}.json"
                )
                remote_to_local[str(remote_rank)] = Path(
                    f"/snapshot/{schedule['allocation_id']}/"
                    f"{run['profile_id']}/rank_{rank}.json"
                )
    snapshot = erratum.EvidenceSnapshot(
        root=Path("/snapshot"),
        remote_anchor="/results",
        by_remote_path=remote_to_local,
        report={"complete": True},
    )
    new_measurements, _, consistency = erratum.aggregate_job_results(
        contract,
        profiles,
        schedules,
        artifacts,
        accuracies,
        jobs,
        plans,
        "manifest-sha",
        "checkpoint-artifact-sha",
        snapshot,
    )
    assert new_measurements == old_measurements
    assert consistency["allocation_torch_version_comparison"] == (
        "major_minor_patch"
    )
    assert consistency["raw_allocation_runtime_string_preserved"] is True
