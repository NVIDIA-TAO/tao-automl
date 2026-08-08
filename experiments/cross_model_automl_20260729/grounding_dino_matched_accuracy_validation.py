#!/usr/bin/env python3

"""Prospective matched retraining for the frozen Grounding DINO coverage gap.

The two specifications are derived mechanically from frozen production
archives: the persisted accuracy-mode winner and the highest-accuracy distinct
specification discovered by another acquisition.  The six-repeat experiment
is validation-only.  It never invokes a selector and never changes a frozen
objective or winner.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

try:
    from . import grounding_dino_acquisition_audit
    from .grounding_dino_shared_detection import pilot_campaign
except ImportError:  # pragma: no cover - direct script execution
    import grounding_dino_acquisition_audit  # type: ignore[no-redef]
    from grounding_dino_shared_detection import (  # type: ignore[no-redef]
        pilot_campaign,
    )


DEFAULT_GDINO_ROOT = grounding_dino_acquisition_audit.DEFAULT_ROOT
DEFAULT_MGDINO_ROOT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask_grounding_dino_coco2017_three_mode_v11"
)
REPEAT_COUNT = 6
VALIDATION_FLAGS = {
    "agent_selected_candidate": False,
    "agent_overrode_winner": False,
    "agent_injected_candidate": False,
    "agent_removed_candidate_to_change_winner": False,
    "agent_changed_objective_weights_after_results": False,
    "agent_changed_accuracy_retention_after_results": False,
    "agent_changed_multi_objective_policy_after_results": False,
    "agent_changed_search_space_after_results": False,
    "agent_changed_seed_after_results": False,
    "agent_replaced_measurement": False,
    "agent_modified_metric_to_favor_candidate": False,
    "agent_increased_budget_for_preferred_candidate": False,
    "agent_reordered_candidates_to_affect_ties": False,
    "measurements_feed_selection": False,
    "measurements_feed_reselection": False,
    "selection_time_objectives_replaced": False,
    "algorithm_selected_candidate_overridden": False,
    "posthoc_measurements_feed_selection": False,
    "posthoc_measurements_feed_reselection": False,
    "historical_winner_overridden": False,
}


class MatchedAccuracyError(RuntimeError):
    """The prospective validation contract or execution is invalid."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _candidate_by_fingerprint(
    root: Path,
    mode: str,
    fingerprint: str,
) -> tuple[str, dict[str, Any], Path]:
    path = root / mode / "candidate_evidence.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        (key, value)
        for key, value in evidence["candidates"].items()
        if value.get("candidate_fingerprint") == fingerprint
    ]
    if len(matches) != 1:
        raise MatchedAccuracyError(
            f"expected one {mode} candidate for fingerprint {fingerprint}"
        )
    key, value = matches[0]
    return key, value, path


def build_contract(
    gdino_root: Path,
    mgdino_root: Path,
    *,
    script_path: Path | None = None,
) -> dict[str, Any]:
    gdino_root = gdino_root.resolve()
    mgdino_root = mgdino_root.resolve()
    acquisition = grounding_dino_acquisition_audit.build_audit(gdino_root)
    observation = acquisition["accuracy_coverage_observation"]
    accuracy_winner = observation["accuracy_winner"]
    higher = observation["higher_distinct_candidate"]
    if higher is None:
        raise MatchedAccuracyError(
            "frozen archives contain no distinct higher-accuracy specification"
        )
    if not observation["higher_candidate_reachable_in_accuracy_search_space"]:
        raise MatchedAccuracyError(
            "comparison specification is outside the frozen accuracy space"
        )

    sources: dict[str, Any] = {}
    for label, summary in (
        ("accuracy_archive_winner", accuracy_winner),
        ("higher_distinct_external_candidate", higher),
    ):
        key, candidate, path = _candidate_by_fingerprint(
            gdino_root,
            summary["mode"] if "mode" in summary else "accuracy",
            summary["candidate_fingerprint"],
        )
        sources[label] = {
            "source_mode": summary.get("mode", "accuracy"),
            "candidate_key": key,
            "candidate_id": candidate["candidate_id"],
            "candidate_fingerprint": candidate["candidate_fingerprint"],
            "checkpoint_id": (
                candidate.get("checkpoint_id") or summary["checkpoint_id"]
            ),
            "selection_time_mAP50": candidate["objective_values"]["mAP50"],
            "spec_sha256": canonical_sha256(candidate["specs"]),
            "candidate_evidence_path": str(path),
            "candidate_evidence_sha256": file_sha256(path),
        }

    labels = tuple(sources)
    schedule = []
    for repeat in range(REPEAT_COUNT):
        order = labels if repeat % 2 == 0 else tuple(reversed(labels))
        for position, label in enumerate(order):
            schedule.append(
                {
                    "cell_id": f"repeat_{repeat}_{label}",
                    "repeat": repeat,
                    "position": position,
                    "specification_label": label,
                }
            )
    manifest_path = gdino_root / "pilot.campaign.v1.json"
    mgdino_contract_path = mgdino_root / "campaign.v11.json"
    if not manifest_path.is_file() or not mgdino_contract_path.is_file():
        raise MatchedAccuracyError("frozen campaign contract is unavailable")
    script_path = (script_path or Path(__file__)).resolve()
    repository = script_path.parents[2]
    source_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    contract = {
        "schema_version": 1,
        "kind": "grounding_dino_matched_accuracy_validation_only",
        "purpose": (
            "separate finite-budget acquisition undercoverage from training "
            "variation without reselection"
        ),
        "grounding_dino_campaign": {
            "root": str(gdino_root),
            "manifest_path": str(manifest_path),
            "manifest_file_sha256": file_sha256(manifest_path),
            "manifest_sha256": json.loads(
                manifest_path.read_text(encoding="utf-8")
            )["manifest_sha256"],
            "acquisition_audit_sha256": acquisition["audit_sha256"],
        },
        "prerequisite_gate": {
            "type": "mask_grounding_dino_v11_successful_terminal_completion",
            "root": str(mgdino_root),
            "campaign_contract_path": str(mgdino_contract_path),
            "campaign_contract_file_sha256": file_sha256(mgdino_contract_path),
            "required_candidates_per_mode": 24,
            "required_modes": ["accuracy", "latency", "multi_objective"],
        },
        "specifications": sources,
        "design": {
            "repeat_count": REPEAT_COUNT,
            "schedule": schedule,
            "balanced_first_position": True,
            "same_training_seed_as_frozen_campaign": True,
            "same_training_and_evaluation_path": True,
            "same_dataset_ptm_fidelity_hardware_and_sqsh": True,
            "primary_paired_difference": (
                "higher_distinct_external_candidate_mAP50 - "
                "accuracy_archive_winner_mAP50"
            ),
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 20260808,
            "confidence_level": 0.95,
        },
        "execution": {
            "gpus_per_job": 8,
            "nodes_per_job": 1,
            "training_jobs": REPEAT_COUNT * 2,
            "standalone_evaluation_jobs": REPEAT_COUNT * 2,
            "model_runs_on_cpu": False,
            "smoke_tests": False,
            "maximum_infrastructure_retries": 3,
        },
        "validation_isolation": dict(VALIDATION_FLAGS),
        "implementation": {
            "path": str(script_path),
            "sha256": file_sha256(script_path),
            "repository": str(repository),
            "source_commit": source_commit,
        },
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    unsigned = dict(contract)
    expected = unsigned.pop("contract_sha256", None)
    if expected != canonical_sha256(unsigned):
        raise MatchedAccuracyError("matched-accuracy contract digest changed")
    if contract.get("validation_isolation") != VALIDATION_FLAGS:
        raise MatchedAccuracyError("validation isolation flags changed")
    if contract.get("design", {}).get("repeat_count") != REPEAT_COUNT:
        raise MatchedAccuracyError("matched repeat count changed")
    implementation = contract.get("implementation")
    if isinstance(implementation, Mapping):
        path = Path(implementation["path"])
        if file_sha256(path) != implementation["sha256"]:
            raise MatchedAccuracyError("matched validation implementation changed")
    schedule = contract["design"]["schedule"]
    if len(schedule) != REPEAT_COUNT * 2:
        raise MatchedAccuracyError("matched schedule size changed")
    positions = {
        label: [
            cell["position"]
            for cell in schedule
            if cell["specification_label"] == label
        ]
        for label in contract["specifications"]
    }
    if any(sorted(values) != [0, 0, 0, 1, 1, 1] for values in positions.values()):
        raise MatchedAccuracyError("matched schedule is not position-balanced")
    for record in contract["specifications"].values():
        path = Path(record["candidate_evidence_path"])
        if file_sha256(path) != record["candidate_evidence_sha256"]:
            raise MatchedAccuracyError("frozen candidate evidence changed")
        _, candidate, _ = _candidate_by_fingerprint(
            Path(contract["grounding_dino_campaign"]["root"]),
            record["source_mode"],
            record["candidate_fingerprint"],
        )
        if canonical_sha256(candidate["specs"]) != record["spec_sha256"]:
            raise MatchedAccuracyError("frozen candidate specification changed")


def _mgdino_terminal(contract: Mapping[str, Any]) -> bool:
    gate = contract["prerequisite_gate"]
    root = Path(gate["root"])
    contract_path = Path(gate["campaign_contract_path"])
    if file_sha256(contract_path) != gate["campaign_contract_file_sha256"]:
        raise MatchedAccuracyError("MGDINO v11 frozen contract changed")
    completion_path = root / "completion.json"
    if not completion_path.is_file():
        return False
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "success" or completion.get("mode_failures"):
        raise MatchedAccuracyError("MGDINO v11 reached a non-success terminal state")
    for mode in gate["required_modes"]:
        evidence_path = root / mode / "candidate_evidence.json"
        result_path = root / mode / "result.json"
        if not evidence_path.is_file() or not result_path.is_file():
            return False
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if len(evidence.get("candidates", {})) != gate["required_candidates_per_mode"]:
            return False
    return True


def wait_for_mgdino(
    contract: Mapping[str, Any],
    *,
    poll_seconds: float,
    timeout_seconds: float,
) -> None:
    started = time.monotonic()
    while not _mgdino_terminal(contract):
        if time.monotonic() - started >= timeout_seconds:
            raise MatchedAccuracyError("MGDINO v11 prerequisite gate timed out")
        time.sleep(poll_seconds)


def _load_env(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _load_candidate(contract: Mapping[str, Any], label: str) -> dict[str, Any]:
    record = contract["specifications"][label]
    _, candidate, _ = _candidate_by_fingerprint(
        Path(contract["grounding_dino_campaign"]["root"]),
        record["source_mode"],
        record["candidate_fingerprint"],
    )
    return candidate


def _persist_state(path: Path, state: dict[str, Any]) -> None:
    state["state_sha256"] = canonical_sha256(
        {key: value for key, value in state.items() if key != "state_sha256"}
    )
    atomic_json(path, state)


def _training_entrypoint(
    manifest: Mapping[str, Any], specification: Mapping[str, Any]
) -> tuple[str, str, str]:
    from tao_sdk.script_runner import build_entrypoint

    action = yaml.safe_load(
        (
            Path(manifest["runtime"]["skill_dir"])
            / "references"
            / "skill_info.yaml"
        ).read_text(encoding="utf-8")
    )["actions"]["train"]
    entrypoint = build_entrypoint(
        command=action["command"],
        specs=specification,
        inputs=action.get("inputs"),
        outputs=action.get("outputs"),
        config_format=action.get("config_format", "yaml"),
        upload_excludes=action.get("upload_excludes", []),
    )
    command = entrypoint["command"]
    return command, canonical_sha256(specification), hashlib.sha256(
        command.encode("utf-8")
    ).hexdigest()


def _submit_training_jobs(
    sdk: Any,
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    state_path: Path,
) -> None:
    runtime = manifest["runtime"]
    for cell in contract["design"]["schedule"]:
        cell_state = state["cells"].setdefault(cell["cell_id"], copy.deepcopy(cell))
        candidate = _load_candidate(contract, cell["specification_label"])
        command, spec_hash, command_hash = _training_entrypoint(
            manifest, candidate["specs"]
        )
        expected = {
            "candidate_fingerprint": candidate["candidate_fingerprint"],
            "training_spec_sha256": spec_hash,
            "training_command_sha256": command_hash,
        }
        if cell_state.get("training_job"):
            if any(
                cell_state["training_job"].get(key) != value
                for key, value in expected.items()
            ):
                raise MatchedAccuracyError("persisted training job identity changed")
            continue
        job = sdk.create_job(
            image=runtime["sqsh_path"],
            command=command,
            gpu_count=8,
            num_nodes=1,
            partition=runtime["partition"],
            account=runtime["account"],
        )
        cell_state["training_job"] = {
            **expected,
            "tao_job_id": job.id,
            "status": "submitted",
        }
        _persist_state(state_path, state)


def _complete_cells(
    sdk: Any,
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    state_path: Path,
    runtime_root: Path,
) -> None:
    common = pilot_campaign._experiment_module("dino_campaign")
    events = runtime_root / "events.jsonl"
    for cell in contract["design"]["schedule"]:
        cell_state = state["cells"][cell["cell_id"]]
        training = cell_state["training_job"]
        if training.get("status") != "Complete":
            status = common._wait_for_job(
                sdk,
                training["tao_job_id"],
                events=events,
                phase="matched_accuracy_training",
                mode="validation_only",
                candidate_id=cell["cell_id"],
            )
            training["status"] = status
            _persist_state(state_path, state)
            if status != "Complete":
                raise MatchedAccuracyError(
                    f"training {cell['cell_id']} ended as {status}"
                )
        if "terminal_checkpoint" not in cell_state:
            cell_state["terminal_checkpoint"] = (
                pilot_campaign.terminal_checkpoint_identity(
                    sdk,
                    training["tao_job_id"],
                    training_epochs=manifest["search"]["training_epochs"],
                )
            )
            _persist_state(state_path, state)

        candidate = _load_candidate(contract, cell["specification_label"])
        evaluation_spec = pilot_campaign.build_evaluation_spec(
            manifest,
            candidate["specs"],
            cell_state["terminal_checkpoint"]["path"],
        )

        def on_evaluation(value: dict[str, Any]) -> None:
            cell_state["evaluation_job"] = value
            _persist_state(state_path, state)

        metric, evaluation = common._launch_evaluation(
            sdk,
            manifest,
            evaluation_spec,
            events=events,
            mode="validation_only",
            candidate_id=cell["cell_id"],
            existing_job=cell_state.get("evaluation_job"),
            on_submitted=on_evaluation,
        )
        cell_state["evaluation_job"] = evaluation
        cell_state["mAP50"] = metric
        cell_state["validation_isolation"] = dict(VALIDATION_FLAGS)
        _persist_state(state_path, state)


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def analyze_completed(
    contract: Mapping[str, Any], state: Mapping[str, Any]
) -> dict[str, Any]:
    by_repeat: dict[int, dict[str, float]] = {}
    for cell in contract["design"]["schedule"]:
        value = state["cells"][cell["cell_id"]]
        if not isinstance(value.get("mAP50"), (int, float)):
            raise MatchedAccuracyError("matched validation is incomplete")
        by_repeat.setdefault(cell["repeat"], {})[
            cell["specification_label"]
        ] = float(value["mAP50"])
    differences = []
    rows = []
    for repeat in sorted(by_repeat):
        values = by_repeat[repeat]
        difference = (
            values["higher_distinct_external_candidate"]
            - values["accuracy_archive_winner"]
        )
        differences.append(difference)
        rows.append({"repeat": repeat, **values, "paired_difference": difference})

    rng = random.Random(contract["design"]["bootstrap_seed"])
    bootstrap = []
    for _ in range(contract["design"]["bootstrap_resamples"]):
        bootstrap.append(
            statistics.median(rng.choice(differences) for _ in differences)
        )
    alpha = 1.0 - contract["design"]["confidence_level"]
    interval = [
        percentile(bootstrap, alpha / 2.0),
        percentile(bootstrap, 1.0 - alpha / 2.0),
    ]
    all_positive = all(value > 0.0 for value in differences)
    all_negative = all(value < 0.0 for value in differences)
    if all_positive and interval[0] > 0.0:
        classification = "HIGHER_FINGERPRINT_REPRODUCIBLY_BETTER"
    elif all_negative and interval[1] < 0.0:
        classification = "ACCURACY_WINNER_REPRODUCIBLY_BETTER"
    elif interval[0] <= 0.0 <= interval[1]:
        classification = "DIRECTION_UNRESOLVED_OR_TRAINING_NOISE"
    else:
        classification = "MIXED_TRAINING_VARIATION"
    result = {
        "schema_version": 1,
        "kind": "grounding_dino_matched_accuracy_validation_result",
        "contract_sha256": contract["contract_sha256"],
        "per_repeat": rows,
        "paired_differences": differences,
        "median_paired_difference": statistics.median(differences),
        "paired_difference_mad": statistics.median(
            abs(value - statistics.median(differences))
            for value in differences
        ),
        "paired_bootstrap_95_interval": interval,
        "classification": classification,
        "selector_invoked": False,
        "validation_isolation": dict(VALIDATION_FLAGS),
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def launch(
    contract_path: Path,
    runtime_root: Path,
    *,
    env_path: Path,
    wait_for_prerequisite: bool,
    poll_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_contract(contract)
    if wait_for_prerequisite:
        wait_for_mgdino(
            contract,
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
        )
    elif not _mgdino_terminal(contract):
        raise MatchedAccuracyError("MGDINO v11 prerequisite is not complete")

    manifest_path = Path(
        contract["grounding_dino_campaign"]["manifest_path"]
    )
    if file_sha256(manifest_path) != contract["grounding_dino_campaign"][
        "manifest_file_sha256"
    ]:
        raise MatchedAccuracyError("Grounding DINO campaign manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _load_env(env_path)
    pilot_campaign.configure_slurm_runtime(manifest)
    sdk_root = str(manifest["runtime"]["sdk_dir"])
    if sdk_root not in sys.path:
        sys.path.insert(0, sdk_root)
    from tao_sdk.platforms.slurm import SlurmSDK

    runtime_root.mkdir(parents=True, exist_ok=True)
    state_path = runtime_root / "execution_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        expected = state.pop("state_sha256", None)
        if expected != canonical_sha256(state):
            raise MatchedAccuracyError("matched execution state digest changed")
        state["state_sha256"] = expected
        if state.get("contract_sha256") != contract["contract_sha256"]:
            raise MatchedAccuracyError("execution state belongs to another contract")
    else:
        state = {
            "schema_version": 1,
            "contract_path": str(contract_path.resolve()),
            "contract_file_sha256": file_sha256(contract_path),
            "contract_sha256": contract["contract_sha256"],
            "cells": {},
            "validation_isolation": dict(VALIDATION_FLAGS),
        }
        _persist_state(state_path, state)
    sdk = SlurmSDK(
        poll_interval=10,
        state_file=runtime_root / "slurm_state.db",
    )
    _submit_training_jobs(sdk, contract, manifest, state, state_path)
    _complete_cells(
        sdk,
        contract,
        manifest,
        state,
        state_path,
        runtime_root,
    )
    result = analyze_completed(contract, state)
    atomic_json(runtime_root / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--gdino-root", type=Path, default=DEFAULT_GDINO_ROOT)
    prepare.add_argument("--mgdino-root", type=Path, default=DEFAULT_MGDINO_ROOT)
    prepare.add_argument("--output", type=Path, required=True)
    execute = subparsers.add_parser("launch")
    execute.add_argument("--contract", type=Path, required=True)
    execute.add_argument("--runtime-root", type=Path, required=True)
    execute.add_argument(
        "--env-file",
        type=Path,
        default=Path("/localhome/local-rarunachalam/.tao/config.env"),
    )
    execute.add_argument("--wait-for-mgdino", action="store_true")
    execute.add_argument("--poll-seconds", type=float, default=30.0)
    execute.add_argument("--timeout-seconds", type=float, default=7 * 24 * 3600)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        contract = build_contract(args.gdino_root, args.mgdino_root)
        atomic_json(args.output, contract)
        print(json.dumps({
            "contract_sha256": contract["contract_sha256"],
            "output": str(args.output.resolve()),
        }, sort_keys=True))
        return 0
    result = launch(
        args.contract,
        args.runtime_root,
        env_path=args.env_file,
        wait_for_prerequisite=args.wait_for_mgdino,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps({
        "classification": result["classification"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
