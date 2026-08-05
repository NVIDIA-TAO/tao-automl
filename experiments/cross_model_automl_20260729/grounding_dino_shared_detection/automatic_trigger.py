#!/usr/bin/env python3

"""Automatically release Grounding DINO qualification on exact predecessor gates."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tao_automl.ptm_registry import canonical_sha256

try:
    from .contract import MODES, PreparationError, read_json, sha256_file
    from .future_contract import DEFAULT_OUTPUT, validate_future_contract
    from .runtime_input_stage import validate_runtime_input_stage
    from .successor_contract import _evaluate_rtdetr_gate
except ImportError:  # pragma: no cover - direct script execution
    from contract import MODES, PreparationError, read_json, sha256_file
    from future_contract import DEFAULT_OUTPUT, validate_future_contract
    from runtime_input_stage import validate_runtime_input_stage
    from successor_contract import _evaluate_rtdetr_gate


HERE = Path(__file__).resolve().parent
DEFAULT_INPUTS = HERE / "campaign.inputs.v3.json"
DEFAULT_RUNTIME_ROOT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "grounding_dino_synthetic_structured_config_successor_v1"
)
DEFAULT_TIMEOUT_SECONDS = 7 * 24 * 60 * 60


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def evaluate_fresh_ddetr_gate(
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the static campaign and its derived runtime release separately."""
    release_path = Path(configuration["artifact_path"])
    static_path = Path(configuration["static_campaign_manifest_path"])
    runtime_path = Path(configuration["runtime_launch_manifest_path"])
    result: dict[str, Any] = {
        "model": "deformable_detr",
        "artifact_path": str(release_path),
        "static_campaign_manifest_path": str(static_path),
        "expected_static_campaign_manifest_sha256": configuration[
            "static_campaign_manifest_sha256"
        ],
        "runtime_launch_manifest_path": str(runtime_path),
        "expected_source_head": configuration["source_head"],
        "passed": False,
        "blockers": [],
    }
    for label, path in (
        ("static campaign manifest", static_path),
        ("runtime launch manifest", runtime_path),
        ("automatic release", release_path),
    ):
        if not path.is_file():
            result["blockers"].append(f"{label} is absent")
    if result["blockers"]:
        return result

    static = read_json(static_path)
    static_payload = copy.deepcopy(static)
    static_sha = static_payload.pop("manifest_sha256", None)
    if (
        static_sha != configuration["static_campaign_manifest_sha256"]
        or static_sha != canonical_sha256(static_payload)
    ):
        result["blockers"].append("static campaign manifest identity differs")

    runtime = read_json(runtime_path)
    runtime_payload = copy.deepcopy(runtime)
    runtime_sha = runtime_payload.pop("manifest_sha256", None)
    if not isinstance(runtime_sha, str) or runtime_sha != canonical_sha256(
        runtime_payload
    ):
        result["blockers"].append("runtime launch manifest identity is invalid")
    if runtime.get("source", {}).get("commit") != configuration["source_head"]:
        result["blockers"].append("runtime launch source head differs")

    release = read_json(release_path)
    for field, expected in configuration["required_release_fields"].items():
        if release.get(field) != expected:
            result["blockers"].append(f"release field {field} differs")
    if release.get("manifest_sha256") != runtime_sha:
        result["blockers"].append("release runtime manifest identity differs")

    root = release_path.parents[1]
    records = {}
    for mode in MODES:
        path = root / "first_candidate_gate" / f"{mode}.json"
        if not path.is_file():
            result["blockers"].append(f"{mode} gate record is absent")
            continue
        gate = read_json(path)
        records[mode] = gate
        if gate.get("manifest_sha256") != runtime_sha:
            result["blockers"].append(f"{mode} gate runtime manifest differs")
        if gate.get("candidate_index") != 0:
            result["blockers"].append(f"{mode} candidate index differs")
        if gate.get("candidate_id") != configuration[
            "required_candidate_id_template"
        ].format(mode=mode):
            result["blockers"].append(f"{mode} candidate ID differs")
        if gate.get("passed") is not True:
            result["blockers"].append(f"{mode} candidate gate did not pass")
        if gate.get("reason") != configuration["required_reason"]:
            result["blockers"].append(f"{mode} gate reason differs")
    if (
        set(records) == set(MODES)
        and release.get("gate_record_sha256") != canonical_sha256(records)
    ):
        result["blockers"].append("release gate-record identity differs")
    result.update(
        {
            "static_campaign_manifest_sha256": static_sha,
            "runtime_manifest_sha256": runtime_sha,
            "static_campaign_file_sha256": sha256_file(static_path),
            "runtime_launch_manifest_file_sha256": sha256_file(runtime_path),
            "artifact_sha256": sha256_file(release_path),
            "passed": not result["blockers"],
        }
    )
    return result


def readiness(
    *,
    contract: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate only frozen local/predecessor evidence; no scheduler mutation."""
    validate_future_contract(contract)
    stage_path = Path(contract["runtime_inputs"]["stage_record_path"])
    stage = read_json(stage_path)
    validate_runtime_input_stage(stage, inputs=inputs)
    blockers = []
    ddetr = evaluate_fresh_ddetr_gate(
        contract["predecessor_release"]["deformable_detr"]
    )
    rtdetr = _evaluate_rtdetr_gate(
        contract["predecessor_release"]["rtdetr"]
    )
    if ddetr["passed"] is not True:
        blockers.append(
            {
                "code": "fresh_ddetr_three_mode_candidate_zero_gate_pending",
                "details": ddetr["blockers"],
            }
        )
    if rtdetr["passed"] is not True:
        blockers.append(
            {
                "code": "rtdetr_release_no_longer_valid",
                "details": rtdetr["blockers"],
            }
        )
    expected_files = {
        Path(contract["dataset"]["source_manifest_path"]): contract["dataset"][
            "source_manifest_sha256"
        ],
        Path(contract["dataset"]["conversion_manifest_path"]): contract[
            "dataset"
        ]["conversion_manifest_sha256"],
        Path(contract["dataset"]["stage_record_path"]): contract["dataset"][
            "stage_record_sha256"
        ],
        stage_path: contract["runtime_inputs"]["stage_record_file_sha256"],
    }
    integrity = contract["integrity"]
    for label in (
        "inputs",
        "runtime_stage",
        "future_contract_generator",
        "runtime_input_stage",
        "qualification_launcher",
        "automatic_trigger",
    ):
        expected_files[Path(integrity[f"{label}_path"])] = integrity[
            f"{label}_sha256"
        ]
    changed = [
        str(path)
        for path, expected in expected_files.items()
        if not path.is_file() or sha256_file(path) != expected
    ]
    if changed:
        blockers.append(
            {"code": "sealed_local_input_changed", "paths": changed}
        )
    return {
        "ready": not blockers,
        "blockers": blockers,
        "deformable_detr": ddetr,
        "rtdetr": rtdetr,
        "runtime_input_stage_sha256": stage["stage_record_sha256"],
        "contract_sha256": contract["contract_sha256"],
        "waits_for_ddetr_full_budget": False,
        "scheduler_mutation_performed": False,
    }


def _launch_once(
    *,
    contract_path: Path,
    contract: Mapping[str, Any],
    runtime_root: Path,
) -> int:
    runtime_root.mkdir(parents=True, exist_ok=True)
    lock_path = runtime_root / "automatic_trigger.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        launch_path = runtime_root / "automatic_launch.json"
        if launch_path.is_file():
            existing = read_json(launch_path)
            if existing.get("contract_sha256") != contract["contract_sha256"]:
                raise PreparationError(
                    "existing automatic launch belongs to another contract"
                )
            return int(existing.get("process_returncode", 0))
        command = list(contract["automatic_trigger"]["launch_command"])
        contract_positions = [
            index for index, value in enumerate(command) if value == "--contract"
        ]
        if len(contract_positions) != 1:
            raise PreparationError("qualification launch command is ambiguous")
        command[contract_positions[0] + 1] = str(contract_path)
        command.extend(["--runtime-root", str(runtime_root)])
        started = {
            "schema_version": 1,
            "at_utc": _utc_timestamp(),
            "contract_sha256": contract["contract_sha256"],
            "command": command,
            "automatic": True,
            "manual_confirmation": False,
            "ddetr_full_budget_dependency": False,
            "process_returncode": None,
        }
        atomic_json(runtime_root / "automatic_launch.started.json", started)
        log_path = runtime_root / "qualification_controller.log"
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": os.pathsep.join(
                    [
                        str(Path(contract["source"]["repository"]) / "src"),
                        str(Path(contract["source"]["repository"])),
                    ]
                ),
            }
        )
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=contract["source"]["repository"],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        completed = {
            **started,
            "completed_at_utc": _utc_timestamp(),
            "process_returncode": result.returncode,
            "log_path": str(log_path),
        }
        completed["launch_record_sha256"] = canonical_sha256(completed)
        atomic_json(launch_path, completed)
        return result.returncode


def watch(
    *,
    contract_path: Path,
    inputs_path: Path,
    runtime_root: Path,
    poll_seconds: float,
    timeout_seconds: float,
) -> int:
    if poll_seconds <= 0 or timeout_seconds <= 0:
        raise ValueError("trigger timing values must be positive")
    contract = read_json(contract_path)
    inputs = read_json(inputs_path)
    validate_future_contract(contract)
    started = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        observed = readiness(contract=contract, inputs=inputs)
        status = {
            "schema_version": 1,
            "at_utc": _utc_timestamp(),
            "attempt": attempt,
            "status": "ready" if observed["ready"] else "waiting",
            "readiness": observed,
            "contract_sha256": contract["contract_sha256"],
            "sdk_constructed": False,
            "scheduler_jobs_submitted": 0,
        }
        atomic_json(runtime_root / "automatic_trigger_status.json", status)
        if observed["ready"]:
            return _launch_once(
                contract_path=contract_path,
                contract=contract,
                runtime_root=runtime_root,
            )
        if time.monotonic() - started >= timeout_seconds:
            status["status"] = "timed_out"
            atomic_json(runtime_root / "automatic_trigger_status.json", status)
            return 2
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument("--watch", action="store_true")
    arguments = parser.parse_args()
    contract = read_json(arguments.contract)
    inputs = read_json(arguments.inputs)
    if not arguments.watch:
        print(
            json.dumps(
                readiness(contract=contract, inputs=inputs),
                sort_keys=True,
            )
        )
        return 0
    return watch(
        contract_path=arguments.contract.resolve(),
        inputs_path=arguments.inputs.resolve(),
        runtime_root=arguments.runtime_root.resolve(),
        poll_seconds=arguments.poll_seconds,
        timeout_seconds=arguments.timeout_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
