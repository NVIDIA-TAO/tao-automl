#!/usr/bin/env python3

"""Read-only aggregation of expanded-front matched-latency measurements.

The six immutable allocation results are validated as complete matched blocks.
This module computes descriptive and all-pairs allocation-paired statistics.
Frozen-archive source validation independently replays the production selector
before result loading, but post-front measurements are never passed to a
selector and cannot modify selection-time objectives or winner identities.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np

import post_front_matched_launcher as launcher
import post_front_matched_manifest_generator as manifest_generator


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "post_front_matched_manifest.v1.json"
DEFAULT_RUNTIME = HERE / "runtime" / "post_front_matched"
DEFAULT_LEDGER = DEFAULT_RUNTIME / "block_submissions.json"
DEFAULT_SDK_STATE = DEFAULT_RUNTIME / "slurm_state.json"
DEFAULT_OUTPUT = DEFAULT_RUNTIME / "post_front_matched_analysis.json"
EXPECTED_ALLOCATIONS = 6
EXPECTED_RANKS = 8

AUTOML_SRC = HERE.parent.parent / "src"
if str(AUTOML_SRC) not in sys.path:
    sys.path.insert(0, str(AUTOML_SRC))

import tao_automl.latency_stats as latency_stats_module  # noqa: E402
from tao_automl.latency_stats import (  # noqa: E402
    LatencyProtocol,
    LatencyValidityThresholds,
    aggregate_synchronized_latency,
)


class ContractError(ValueError):
    """Raised when immutable results violate the matched design."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest-file-sha256", required=True)
    parser.add_argument("--submission-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--submission-ledger-sha256", required=True)
    parser.add_argument("--sdk-state", type=Path, default=DEFAULT_SDK_STATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--secrets-env",
        type=Path,
        default=Path("/localhome/local-rarunachalam/.tao/config.env"),
    )
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ContractError(
            f"{label} mismatch: expected {expected!r}, got {actual!r}"
        )


def major_minor_patch(value: Any, label: str) -> str:
    match = re.match(r"^(\d+\.\d+\.\d+)", str(value))
    if match is None:
        raise ContractError(f"{label} has no major.minor.patch prefix: {value}")
    return match.group(1)


def normalized_hostname(value: Any, label: str) -> str:
    """Return a stable short hostname for scheduler/runtime identity checks."""

    if not isinstance(value, str):
        raise ContractError(f"{label} must be a hostname string")
    hostname = value.strip().rstrip(".").lower()
    if (
        not hostname
        or any(character.isspace() for character in hostname)
        or re.fullmatch(r"[a-z0-9][a-z0-9._-]*", hostname) is None
    ):
        raise ContractError(f"{label} is invalid: {value!r}")
    return hostname.split(".", 1)[0]


def expand_slurm_nodelist(node_list: Any) -> dict[str, Any]:
    """Expand a one-node SLURM hostlist and retain normalized identity."""

    if not isinstance(node_list, str) or not node_list.strip():
        raise ContractError("completed SLURM allocation lacks NodeList")
    output = remote_output(
        " ".join(
            [
                "scontrol",
                "show",
                "hostnames",
                shlex.quote(node_list.strip()),
            ]
        ),
        timeout=120,
    )
    hostnames = [line.strip() for line in output.splitlines() if line.strip()]
    if len(hostnames) != 1:
        raise ContractError(
            "matched allocation must expand to exactly one SLURM node; "
            f"NodeList {node_list!r} expanded to {hostnames!r}"
        )
    return {
        "node_list": node_list.strip(),
        "expanded_node_hostnames": hostnames,
        "normalized_node_hostname": normalized_hostname(
            hostnames[0],
            "expanded SLURM hostname",
        ),
    }


def validate_scheduler_hostname_binding(
    job: dict[str, Any],
    allocation_hostname: Any,
) -> dict[str, Any]:
    """Bind sacct NodeList, allocation hostname, and all rank records."""

    expanded = job.get("expanded_node_hostnames")
    if not isinstance(expanded, list) or len(expanded) != 1:
        raise ContractError("job lacks one-node scheduler hostname evidence")
    scheduler_hostname = expanded[0]
    scheduler_normalized = normalized_hostname(
        scheduler_hostname,
        "expanded SLURM hostname",
    )
    require_equal(
        job.get("normalized_node_hostname"),
        scheduler_normalized,
        "stored normalized SLURM hostname",
    )
    allocation_normalized = normalized_hostname(
        allocation_hostname,
        "allocation hostname",
    )
    require_equal(
        allocation_normalized,
        scheduler_normalized,
        "sacct NodeList/allocation hostname",
    )
    return {
        "node_list": job.get("node_list"),
        "expanded_node_hostname": scheduler_hostname,
        "normalized_node_hostname": scheduler_normalized,
        "allocation_hostname": allocation_hostname,
        "allocation_normalized_hostname": allocation_normalized,
        "rank_hostname_binding": (
            "rank records are required to equal allocation_hostname"
        ),
        "status": "pass",
    }


def aggregation_runtime_provenance(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Revalidate executable Python and SDK provenance at aggregation time."""

    sources = manifest.get("source_artifacts", {})
    latency_source = sources.get("latency_stats")
    if not isinstance(latency_source, dict):
        raise ContractError("pinned latency_stats provenance is missing")
    expected_module_path = Path(str(latency_source.get("path", ""))).resolve()
    imported_file = getattr(latency_stats_module, "__file__", None)
    if not isinstance(imported_file, str) or not imported_file:
        raise ContractError("imported latency_stats module lacks __file__")
    imported_module_path = Path(imported_file).resolve()
    require_equal(
        imported_module_path,
        expected_module_path,
        "imported latency_stats module path",
    )
    expected_module_sha256 = manifest_generator.require_sha256(
        latency_source.get("sha256"),
        "pinned latency_stats SHA256",
    )
    actual_module_sha256 = manifest_generator.sha256_file(
        imported_module_path
    )
    require_equal(
        actual_module_sha256,
        expected_module_sha256,
        "imported latency_stats module SHA256",
    )

    runtime = manifest.get("runtime", {})
    sdk_path = Path(str(runtime.get("sdk_path", ""))).resolve()
    if not sdk_path.is_dir():
        raise ContractError(f"pinned TAO SDK repository is missing: {sdk_path}")
    expected_sdk_commit = manifest_generator.require_git_oid(
        runtime.get("sdk_commit"),
        "pinned TAO SDK commit",
    )
    try:
        actual_sdk_commit = manifest_generator.require_git_oid(
            launcher.git_value(sdk_path, "rev-parse", "HEAD"),
            "aggregation-time TAO SDK commit",
        )
        actual_sdk_branch = launcher.git_value(
            sdk_path,
            "branch",
            "--show-current",
        )
        sdk_status = launcher.git_value(
            sdk_path,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ContractError(
            f"could not revalidate pinned TAO SDK repository: {sdk_path}"
        ) from error
    require_equal(
        actual_sdk_commit,
        expected_sdk_commit,
        "aggregation-time TAO SDK commit",
    )
    expected_sdk_branch = runtime.get("sdk_branch")
    if not isinstance(expected_sdk_branch, str) or not expected_sdk_branch:
        raise ContractError("pinned TAO SDK branch is missing")
    require_equal(
        actual_sdk_branch,
        expected_sdk_branch,
        "aggregation-time TAO SDK branch",
    )
    if sdk_status:
        raise ContractError(
            "TAO SDK worktree must remain clean at aggregation time"
        )
    return {
        "checked_at_utc": timestamp(),
        "latency_stats": {
            "module": latency_stats_module.__name__,
            "imported_file": str(imported_module_path),
            "expected_file": str(expected_module_path),
            "sha256": actual_module_sha256,
            "imported_file_matches_manifest": True,
            "imported_sha256_matches_manifest": True,
        },
        "tao_sdk": {
            "path": str(sdk_path),
            "branch": actual_sdk_branch,
            "commit": actual_sdk_commit,
            "worktree_clean": True,
        },
    }


def load_ledger(
    path: Path,
    supplied_sha256: str,
    manifest: dict[str, Any],
    manifest_file_sha256: str,
    expected_source_checks: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    require_equal(
        path.resolve(),
        Path(manifest["runtime"]["local_runtime_path"]).resolve()
        / "block_submissions.json",
        "submission-ledger path",
    )
    ledger, whole_file_sha256 = manifest_generator.load_exact_json(
        path,
        supplied_sha256,
        "post-front submission ledger",
    )
    require_equal(ledger.get("schema_version"), 1, "ledger schema")
    require_equal(ledger.get("status"), "complete", "ledger status")
    require_equal(
        ledger.get("phase"),
        "expanded_post_front_matched_latency",
        "ledger phase",
    )
    require_equal(
        ledger.get("manifest_id"),
        manifest["manifest_id"],
        "ledger manifest ID",
    )
    require_equal(
        ledger.get("manifest_sha256"),
        manifest_file_sha256,
        "ledger manifest SHA256",
    )
    require_equal(
        ledger.get("schedule_sha256"),
        manifest["schedule"]["schedule_sha256"],
        "ledger schedule SHA256",
    )
    require_equal(
        ledger.get("expected_allocation_count"),
        EXPECTED_ALLOCATIONS,
        "ledger expected allocation count",
    )
    require_equal(
        ledger.get("allocation_count"),
        EXPECTED_ALLOCATIONS,
        "ledger allocation count",
    )
    for key in (
        "feeds_final_selection",
        "feeds_reselection",
        "selection_time_objective_replacement_permitted",
    ):
        require_equal(ledger.get(key), False, f"ledger {key}")
    require_equal(
        ledger.get("pending_submission"),
        None,
        "ledger pending submission",
    )
    revision = ledger.get("ledger_revision")
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 1
    ):
        raise ContractError("ledger revision must be a positive integer")
    history = ledger.get("superseded_submissions")
    if not isinstance(history, list) or len(history) != revision - 1:
        raise ContractError(
            "ledger supersession history must match its revision"
        )
    if revision == 1:
        require_equal(
            ledger.get("parent_ledger_sha256"),
            None,
            "initial ledger parent",
        )
    else:
        parent_sha256 = manifest_generator.require_sha256(
            ledger.get("parent_ledger_sha256"),
            "ledger parent SHA256",
        )
        require_equal(
            history[-1].get("parent_ledger_whole_file_sha256"),
            parent_sha256,
            "latest supersession parent SHA256",
        )
    claimed = manifest_generator.require_sha256(
        ledger.get("ledger_sha256"),
        "ledger canonical SHA256",
    )
    unhashed = copy.deepcopy(ledger)
    del unhashed["ledger_sha256"]
    require_equal(
        manifest_generator.sha256_value(unhashed),
        claimed,
        "ledger canonical digest",
    )
    submissions = ledger.get("submissions")
    if not isinstance(submissions, list) or len(submissions) != 6:
        raise ContractError("ledger must contain exactly six submissions")
    expected = {
        item["allocation_id"]: item
        for item in manifest["schedule"]["allocations"]
    }
    actual_ids = [item.get("allocation_id") for item in submissions]
    if set(actual_ids) != set(expected) or len(set(actual_ids)) != 6:
        raise ContractError("ledger allocation IDs differ from manifest")
    tao_ids: set[str] = set()
    slurm_ids: set[str] = set()
    for submission in submissions:
        allocation_id = submission["allocation_id"]
        allocation = expected[allocation_id]
        require_equal(
            submission.get("allocation_index"),
            allocation["allocation_index"],
            f"{allocation_id} allocation index",
        )
        require_equal(
            submission.get("candidate_order"),
            allocation["candidate_order"],
            f"{allocation_id} submitted order",
        )
        require_equal(
            submission.get("design_row_index"),
            allocation["design_row_index"],
            f"{allocation_id} design row",
        )
        for key in ("feeds_final_selection", "feeds_reselection"):
            require_equal(
                submission.get(key),
                False,
                f"{allocation_id} {key}",
            )
        require_equal(
            submission.get("candidate_count"),
            len(manifest["candidates"]),
            f"{allocation_id} candidate count",
        )
        for key in (
            "block_plan_sha256",
            "command_sha256",
            "staging_bundle_sha256",
            "staging_bundle_json_sha256",
        ):
            manifest_generator.require_sha256(
                submission.get(key),
                f"{allocation_id} {key}",
            )
        staged = submission.get("staging_file_sha256")
        if not isinstance(staged, dict):
            raise ContractError(
                f"{allocation_id} staging-file evidence is missing"
            )
        expected_config_names = {
            f"configs/{candidate_id}.yaml"
            for candidate_id in allocation["candidate_order"]
        }
        if not expected_config_names.issubset(staged):
            raise ContractError(
                f"{allocation_id} staged config set is incomplete"
            )
        for filename, digest in staged.items():
            if not isinstance(filename, str) or not filename:
                raise ContractError(f"{allocation_id} staged filename is invalid")
            manifest_generator.require_sha256(
                digest,
                f"{allocation_id} staged {filename}",
            )
        require_equal(
            submission.get("launch_uncertain"),
            False,
            f"{allocation_id} launch uncertainty",
        )
        tao_job_id = submission.get("tao_job_id")
        slurm_job_id = str(submission.get("slurm_job_id", ""))
        if not isinstance(tao_job_id, str) or not tao_job_id:
            raise ContractError(f"{allocation_id} lacks TAO job ID")
        if not slurm_job_id.isdigit():
            raise ContractError(f"{allocation_id} has invalid SLURM job ID")
        if tao_job_id in tao_ids or slurm_job_id in slurm_ids:
            raise ContractError("TAO and SLURM job IDs must be unique")
        tao_ids.add(tao_job_id)
        slurm_ids.add(slurm_job_id)
    recovery_events = ledger.get("submission_recovery_events")
    if not isinstance(recovery_events, list):
        raise ContractError("ledger submission recovery events are missing")
    submissions_by_allocation = {
        item["allocation_id"]: item for item in submissions
    }
    for index, event in enumerate(recovery_events):
        if not isinstance(event, dict):
            raise ContractError("ledger submission recovery event is invalid")
        allocation_id = event.get("allocation_id")
        if allocation_id not in expected:
            raise ContractError(
                "submission recovery allocation is outside manifest"
            )
        for key, expected_value in (
            ("event_index", index),
            (
                "command_sha256",
                submissions_by_allocation[allocation_id]["command_sha256"],
            ),
            ("launch_uncertain", False),
            ("partial_measurements_reused", False),
            ("feeds_final_selection", False),
            ("feeds_reselection", False),
        ):
            require_equal(
                event.get(key),
                expected_value,
                f"submission recovery event {index} {key}",
            )
        reason = event.get("reason")
        if reason not in {
            "durably_terminal_submission_not_reused",
            "proven_pre_scheduler_submission_abandoned",
        }:
            raise ContractError(
                f"submission recovery event {index} reason is invalid"
            )
        sdk_status = event.get("sdk_status")
        if not isinstance(sdk_status, str) or not sdk_status:
            raise ContractError(
                f"submission recovery event {index} SDK status is invalid"
            )
        submission_attempted = event.get("submission_attempted")
        if not isinstance(submission_attempted, bool):
            raise ContractError(
                f"submission recovery event {index} attempt flag is invalid"
            )
        tao_job_id = event.get("tao_job_id")
        slurm_job_id = str(event.get("slurm_job_id", ""))
        if (
            not isinstance(tao_job_id, str)
            or not tao_job_id
            or tao_job_id in tao_ids
        ):
            raise ContractError(
                f"submission recovery event {index} TAO identity is invalid"
            )
        tao_ids.add(tao_job_id)
        if reason == "durably_terminal_submission_not_reused":
            if sdk_status not in {"Error", "Canceled"}:
                raise ContractError(
                    "terminal recovery event must have terminal SDK status"
                )
            if not slurm_job_id.isdigit() or slurm_job_id in slurm_ids:
                raise ContractError(
                    "terminal recovery event SLURM identity is invalid"
                )
            slurm_ids.add(slurm_job_id)
        else:
            require_equal(
                submission_attempted,
                False,
                "pre-scheduler recovery submission attempted",
            )
            require_equal(
                slurm_job_id,
                "",
                "pre-scheduler recovery SLURM identity",
            )
        launcher.validate_recovery_event_reconciliation(
            event,
            submissions_by_allocation[allocation_id]["command_sha256"],
        )
    for index, supersession in enumerate(history):
        if not isinstance(supersession, dict):
            raise ContractError("ledger supersession record is invalid")
        allocation_id = supersession.get("allocation_id")
        if allocation_id not in expected:
            raise ContractError("supersession allocation is outside manifest")
        prior_status = supersession.get("prior_sdk_status")
        if prior_status not in {"Error", "Canceled", "Complete"}:
            raise ContractError(
                f"supersession {index} prior SDK status is invalid"
            )
        expected_reason = (
            "complete_but_semantically_invalid_allocation"
            if prior_status == "Complete"
            else "durable_terminal_incomplete_allocation"
        )
        require_equal(
            supersession.get("reason"),
            expected_reason,
            f"supersession {index} reason",
        )
        require_equal(
            supersession.get("incomplete_allocation_policy"),
            manifest["incomplete_allocation_policy"],
            f"supersession {index} policy",
        )
        require_equal(
            supersession.get("partial_measurements_reused"),
            False,
            f"supersession {index} partial measurement reuse",
        )
        manifest_generator.require_sha256(
            supersession.get("parent_ledger_whole_file_sha256"),
            f"supersession {index} parent ledger SHA256",
        )
        manifest_generator.require_sha256(
            supersession.get("parent_ledger_internal_sha256"),
            f"supersession {index} parent ledger internal SHA256",
        )
        prior = supersession.get("prior_submission")
        if not isinstance(prior, dict):
            raise ContractError(
                f"supersession {index} prior submission is missing"
            )
        require_equal(
            prior.get("allocation_id"),
            allocation_id,
            f"supersession {index} allocation binding",
        )
        require_equal(
            prior.get("candidate_order"),
            expected[allocation_id]["candidate_order"],
            f"supersession {index} candidate order",
        )
        for key in ("feeds_final_selection", "feeds_reselection"):
            require_equal(
                prior.get(key),
                False,
                f"supersession {index} prior {key}",
            )
        prior_tao_id = prior.get("tao_job_id")
        prior_slurm_id = str(prior.get("slurm_job_id", ""))
        if (
            not isinstance(prior_tao_id, str)
            or not prior_tao_id
            or not prior_slurm_id.isdigit()
        ):
            raise ContractError(
                f"supersession {index} prior job identity is invalid"
            )
        if prior_tao_id in tao_ids or prior_slurm_id in slurm_ids:
            raise ContractError(
                "effective and superseded job identities must be unique"
            )
        tao_ids.add(prior_tao_id)
        slurm_ids.add(prior_slurm_id)
        launcher.validate_replacement_intent_evidence(
            supersession.get("replacement_intent"),
            manifest=manifest,
            manifest_file_sha256=manifest_file_sha256,
            supersession=supersession,
            expected_revision=index + 2,
        )
    source_checks = ledger.get("source_checks")
    if not isinstance(source_checks, dict):
        raise ContractError("ledger source checks are missing")
    for key, expected_value in expected_source_checks.items():
        require_equal(
            source_checks.get(key),
            expected_value,
            f"ledger source check {key}",
        )
    launch_source = source_checks.get("launch_contract")
    if not isinstance(launch_source, dict):
        raise ContractError("ledger launch-contract evidence is missing")
    launch_path = Path(str(launch_source.get("path", ""))).resolve()
    require_equal(
        launch_path,
        path.resolve().parent / launcher.LAUNCH_CONTRACT_NAME,
        "launch-contract path",
    )
    launch_contract, launch_file_sha256 = (
        manifest_generator.load_exact_json(
            launch_path,
            launch_source.get("whole_file_sha256"),
            "post-front launch contract",
        )
    )
    manifest_generator.validate_internal_digest(
        launch_contract,
        "contract_sha256",
        "post-front launch contract",
    )
    require_equal(
        launch_contract["contract_sha256"],
        launch_source.get("internal_sha256"),
        "launch-contract internal SHA256",
    )
    require_equal(
        launch_file_sha256,
        launch_source.get("whole_file_sha256"),
        "launch-contract whole-file SHA256",
    )
    for key, expected_value in (
        ("schema_version", 1),
        ("contract_id", "dino_post_front_matched_launch_20260728_v1"),
        ("status", "reserved_before_sdk_initialization"),
        ("manifest_id", manifest["manifest_id"]),
        ("manifest_sha256", manifest_file_sha256),
        ("manifest_internal_sha256", manifest["manifest_sha256"]),
        ("schedule_sha256", manifest["schedule"]["schedule_sha256"]),
        (
            "candidate_set_sha256",
            manifest["candidate_derivation"]["candidate_set_sha256"],
        ),
        (
            "runtime_path",
            str(Path(manifest["runtime"]["local_runtime_path"]).resolve()),
        ),
        ("allocation_count", EXPECTED_ALLOCATIONS),
        (
            "allocation_ids",
            [
                item["allocation_id"]
                for item in manifest["schedule"]["allocations"]
            ],
        ),
        ("feeds_final_selection", False),
        ("feeds_reselection", False),
        ("selection_time_objective_replacement_permitted", False),
        ("manual_winner_override_permitted", False),
    ):
        require_equal(
            launch_contract.get(key),
            expected_value,
            f"launch contract {key}",
        )
    ledger_without_launch = copy.deepcopy(source_checks)
    del ledger_without_launch["launch_contract"]
    require_equal(
        launch_contract.get("source_checks"),
        ledger_without_launch,
        "launch/ledger source checks",
    )
    by_id = {item["allocation_id"]: item for item in submissions}
    ledger["submissions"] = [
        by_id[item["allocation_id"]]
        for item in manifest["schedule"]["allocations"]
    ]
    return ledger, whole_file_sha256


def sdk_database_path(state_path: Path) -> Path:
    if state_path.name.endswith(".json"):
        return state_path.with_suffix(".db")
    return Path(str(state_path) + ".db")


def local_lustre_path(uri: str) -> str:
    if uri.startswith("lustre://"):
        path = uri.removeprefix("lustre://")
        return path if path.startswith("/") else f"/{path}"
    if uri.startswith("/"):
        return uri
    raise ContractError(f"expected Lustre result URI, got {uri!r}")


def remote_output(command: str, *, timeout: int = 900) -> str:
    return launcher.remote_output(command, timeout=timeout)


def slurm_accounting(slurm_ids: list[str]) -> dict[str, dict[str, Any]]:
    command = " ".join(
        [
            "sacct",
            "-X",
            "-j",
            shlex.quote(",".join(slurm_ids)),
            "--noheader",
            "--parsable2",
            "--format=JobIDRaw,State,ExitCode,NodeList",
        ]
    )
    rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for line in remote_output(command, timeout=120).splitlines():
        fields = line.strip().split("|")
        if len(fields) < 4 or fields[0] not in slurm_ids:
            continue
        state = fields[1].split("+", 1)[0].split(None, 1)[0]
        rows[fields[0]].append(
            {
                "slurm_job_id": fields[0],
                "state": state,
                "exit_code": fields[2],
                "node_list": fields[3],
            }
        )
    result = {}
    for slurm_id in slurm_ids:
        candidates = rows.get(slurm_id, [])
        if len(candidates) != 1:
            raise ContractError(
                f"expected one sacct row for {slurm_id}, got {len(candidates)}"
            )
        row = candidates[0]
        row["complete"] = (
            row["state"] == "COMPLETED" and row["exit_code"] == "0:0"
        )
        if row["complete"]:
            row.update(expand_slurm_nodelist(row["node_list"]))
        else:
            row["expanded_node_hostnames"] = []
            row["normalized_node_hostname"] = None
        result[slurm_id] = row
    return result


def inspect_jobs(
    manifest: dict[str, Any],
    ledger: dict[str, Any],
    state_path: Path,
) -> tuple[list[dict[str, Any]], Path]:
    database = sdk_database_path(state_path)
    if not database.is_file():
        raise FileNotFoundError(f"SDK durable state is missing: {database}")
    sdk_path = manifest["runtime"]["sdk_path"]
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)
    from tao_sdk.platforms.slurm import SlurmSDK

    runtime = manifest["runtime"]
    os.environ["SLURM_USE_SQSH"] = "false"
    os.environ["SLURM_USE_REQUEUE"] = "false"
    os.environ["SLURM_PARTITION"] = runtime["partition"]
    os.environ["SLURM_ACCOUNT"] = runtime["account"]
    sdk = SlurmSDK(poll_interval=10, state_file=state_path)
    jobs = []
    try:
        accounting = slurm_accounting(
            [
                str(item["slurm_job_id"])
                for item in ledger["submissions"]
            ]
        )
        for submission in ledger["submissions"]:
            allocation_id = submission["allocation_id"]
            observation = launcher.observe_job_status_no_retry(
                sdk,
                submission["tao_job_id"],
            )
            identity = observation["runtime_identity"]
            actual_slurm_id = str(identity.get("slurm_job_id", ""))
            require_equal(
                actual_slurm_id,
                str(submission["slurm_job_id"]),
                f"{allocation_id} SDK/ledger SLURM ID",
            )
            require_equal(
                identity.get("launch_uncertain", False),
                False,
                f"{allocation_id} active launch uncertainty",
            )
            require_equal(
                identity.get("retry_count", 0),
                submission.get("retry_count", 0),
                f"{allocation_id} retry count",
            )
            require_equal(
                identity.get("failed_slurm_job_ids", []),
                submission.get("failed_slurm_job_ids", []),
                f"{allocation_id} failed SLURM IDs",
            )
            sacct = accounting[actual_slurm_id]
            results_uri = sdk.get_job_results_dir(submission["tao_job_id"])
            require_equal(
                results_uri,
                submission.get("sdk_results_uri"),
                f"{allocation_id} SDK results URI",
            )
            jobs.append(
                {
                    "allocation_id": allocation_id,
                    "allocation_index": submission["allocation_index"],
                    "design_row_index": submission["design_row_index"],
                    "candidate_order": copy.deepcopy(
                        submission["candidate_order"]
                    ),
                    "block_plan_sha256": submission[
                        "block_plan_sha256"
                    ],
                    "staging_file_sha256": copy.deepcopy(
                        submission["staging_file_sha256"]
                    ),
                    "tao_job_id": submission["tao_job_id"],
                    "slurm_job_id": actual_slurm_id,
                    "sdk_status": observation["status"],
                    "sdk_message": observation["message"],
                    "sdk_status_allow_retry": False,
                    "status_inspection_scheduler_identity_unchanged": True,
                    "slurm_state": sacct["state"],
                    "slurm_exit_code": sacct["exit_code"],
                    "node_list": sacct["node_list"],
                    "expanded_node_hostnames": copy.deepcopy(
                        sacct["expanded_node_hostnames"]
                    ),
                    "normalized_node_hostname": sacct[
                        "normalized_node_hostname"
                    ],
                    "result_root": local_lustre_path(
                        results_uri
                    ),
                    "complete": observation["status"] == "Complete"
                    and sacct["complete"],
                    "feeds_final_selection": False,
                    "feeds_reselection": False,
                }
            )
    finally:
        sdk._monitor.stop()
        sdk._store.close()
    return jobs, database


def fetch_allocation_bundle(
    manifest: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    result_root = Path(job["result_root"])
    if result_root.name != job["tao_job_id"]:
        raise ContractError(
            f"{job['allocation_id']}: SDK result root is not job scoped"
        )
    result_path = allocation_result_path(manifest, job)
    reader = "\n".join(
        [
            "import hashlib,json,sys",
            "from pathlib import Path",
            "path=Path(sys.argv[1])",
            "result_bytes=path.read_bytes()",
            "result=json.loads(result_bytes)",
            "records={}",
            "for run in result.get('candidate_runs', []):",
            " root=Path(run['raw_samples_dir'])",
            " paths=sorted(root.glob('rank_*.json'))",
            " payloads=[item.read_bytes() for item in paths]",
            " records[run['candidate_id']]={",
            "  'paths':[str(item) for item in paths],",
            (
                "  'sha256':[hashlib.sha256(item).hexdigest() "
                "for item in payloads],"
            ),
            "  'records':[json.loads(item) for item in payloads],",
            " }",
            (
                "print(json.dumps({'result_path':str(path),"
                "'result_sha256':hashlib.sha256(result_bytes).hexdigest(),"
                "'result':result,"
                "'rank_records':records},sort_keys=True))"
            ),
        ]
    )
    output = remote_output(
        f"python3 -c {shlex.quote(reader)} {shlex.quote(str(result_path))}",
        timeout=900,
    )
    return json.loads(output)


def allocation_result_path(
    manifest: dict[str, Any],
    job: dict[str, Any],
) -> Path:
    """Derive the only result path permitted for one job-scoped allocation."""

    result_root = Path(job["result_root"])
    if result_root.name != job["tao_job_id"]:
        raise ContractError(
            f"{job['allocation_id']}: SDK result root is not job scoped"
        )
    return (
        result_root
        / "dino_moo_phase2_20260728"
        / "post_front_matched"
        / manifest["manifest_id"]
        / job["allocation_id"]
        / "allocation_result.json"
    )


def available_artifacts_from_bundle(
    allocation_id: str,
    bundle: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Extract only already-read hashes; never treat them as measurements."""

    if not isinstance(bundle, dict):
        return []
    artifacts: list[dict[str, str]] = []
    result_path = bundle.get("result_path")
    result_sha256 = bundle.get("result_sha256")
    if (
        isinstance(result_path, str)
        and result_path
        and isinstance(result_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", result_sha256)
    ):
        artifacts.append(
            {
                "kind": "allocation_result",
                "allocation_id": allocation_id,
                "candidate_id": "",
                "rank": "",
                "path": result_path,
                "sha256": result_sha256,
            }
        )
    rank_records = bundle.get("rank_records")
    if isinstance(rank_records, dict):
        for candidate_id, record in rank_records.items():
            if not isinstance(candidate_id, str) or not isinstance(
                record, dict
            ):
                continue
            paths = record.get("paths")
            digests = record.get("sha256")
            if not isinstance(paths, list) or not isinstance(digests, list):
                continue
            for rank, (path, digest) in enumerate(zip(paths, digests)):
                if (
                    isinstance(path, str)
                    and path
                    and isinstance(digest, str)
                    and re.fullmatch(r"[0-9a-f]{64}", digest)
                ):
                    artifacts.append(
                        {
                            "kind": "rank_record",
                            "allocation_id": allocation_id,
                            "candidate_id": candidate_id,
                            "rank": str(rank),
                            "path": path,
                            "sha256": digest,
                        }
                    )
    return sorted(
        artifacts,
        key=lambda item: (
            item["kind"],
            item["allocation_id"],
            item["candidate_id"],
            item["rank"],
            item["path"],
        ),
    )


def probe_available_allocation_artifacts(
    manifest: dict[str, Any],
    job: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Hash every existing expected file without parsing failed JSON."""

    result_path = allocation_result_path(manifest, job)
    targets = [
        {
            "kind": "allocation_result",
            "allocation_id": job["allocation_id"],
            "candidate_id": "",
            "rank": "",
            "path": str(result_path),
        }
    ]
    for position, candidate_id in enumerate(job["candidate_order"]):
        run_label = (
            f"{job['allocation_id']}_p{position:03d}_{candidate_id}"
        )
        raw_dir = (
            result_path.parent
            / "candidates"
            / run_label
            / job["tao_job_id"]
            / "latency"
        )
        for rank in range(EXPECTED_RANKS):
            targets.append(
                {
                    "kind": "rank_record",
                    "allocation_id": job["allocation_id"],
                    "candidate_id": candidate_id,
                    "rank": str(rank),
                    "path": str(raw_dir / f"rank_{rank}.json"),
                }
            )
    encoded = base64.b64encode(
        json.dumps(targets, sort_keys=True).encode("utf-8")
    ).decode("ascii")
    reader = "\n".join(
        [
            "import base64,hashlib,json,sys",
            "from pathlib import Path",
            "targets=json.loads(base64.b64decode(sys.argv[1]))",
            "available=[]",
            "errors=[]",
            "for item in targets:",
            " path=Path(item['path'])",
            " try:",
            "  if path.is_file():",
            "   row=dict(item)",
            "   row['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()",
            "   available.append(row)",
            " except OSError as error:",
            (
                "  errors.append({'path':str(path),"
                "'error_type':type(error).__name__})"
            ),
            (
                "print(json.dumps({'available':available,'errors':errors},"
                "sort_keys=True))"
            ),
        ]
    )
    output = remote_output(
        f"python3 -c {shlex.quote(reader)} {shlex.quote(encoded)}",
        timeout=900,
    )
    payload = json.loads(output)
    available = payload.get("available")
    errors = payload.get("errors")
    if not isinstance(available, list) or not isinstance(errors, list):
        raise ContractError("artifact probe returned an invalid payload")
    return sorted(
        available,
        key=lambda item: (
            str(item.get("kind", "")),
            str(item.get("allocation_id", "")),
            str(item.get("candidate_id", "")),
            str(item.get("rank", "")),
            str(item.get("path", "")),
        ),
    ), {
        "status": "complete",
        "expected_artifact_count": len(targets),
        "available_artifact_count": len(available),
        "filesystem_error_count": len(errors),
        "filesystem_errors": errors,
    }


def latency_protocol(manifest: dict[str, Any]) -> LatencyProtocol:
    source = manifest["latency_protocol"]
    thresholds = source["validity_thresholds"]
    return LatencyProtocol(
        warmup_iterations=source["warmup_iterations"],
        timed_iterations=source["timed_iterations"],
        repeated_rounds=source["repeated_rounds"],
        tail_percentile=source["tail_percentile"],
        bootstrap_resamples=source["bootstrap_resamples"],
        bootstrap_confidence_level=source["bootstrap_confidence_level"],
        bootstrap_seed=source["bootstrap_seed"],
        expected_devices=tuple(str(index) for index in range(8)),
        validity_thresholds=LatencyValidityThresholds(
            max_robust_cv=thresholds["max_robust_cv"],
            max_round_median_range_fraction=thresholds[
                "max_round_median_range_fraction"
            ],
            max_absolute_round_drift_fraction=thresholds[
                "max_absolute_round_drift_fraction"
            ],
            max_device_median_range_fraction=thresholds[
                "max_device_median_range_fraction"
            ],
            max_bootstrap_ci_width_fraction=thresholds[
                "max_bootstrap_ci_width_fraction"
            ],
        ),
    )


def expected_rank_protocol(manifest: dict[str, Any]) -> dict[str, Any]:
    source = manifest["latency_protocol"]
    return {
        "warmup_iterations": source["warmup_iterations"],
        "timed_iterations": source["timed_iterations"],
        "repeated_rounds": source["repeated_rounds"],
        "preloaded_batches": source["preloaded_batches"],
        "batch_size_per_gpu": source["batch_size_per_gpu"],
        "precision": source["precision"],
        "tf32": source["tf32"],
        "cudnn_benchmark": source["cudnn_benchmark"],
        "cudnn_deterministic": source["cudnn_deterministic"],
        "timed_scope": source["timed_scope"],
        "excluded_scope": [
            "checkpoint_load",
            "disk_io",
            "decode_resize_normalize",
            "host_to_device_transfer",
            "coco_accumulation",
            "distributed_gather",
        ],
        "synchronization": source["synchronization"],
        "seed": source["benchmark_seed"],
    }


def validate_input_identity(
    record: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    metadata = record.get("benchmark_inputs")
    if not isinstance(metadata, dict):
        raise ContractError("rank record lacks benchmark input identity")
    batches = metadata.get("batches")
    expected_count = manifest["latency_protocol"]["preloaded_batches"]
    if not isinstance(batches, list) or len(batches) != expected_count:
        raise ContractError("benchmark input preload count mismatch")
    require_equal(
        metadata.get("batch_count"),
        expected_count,
        "benchmark batch count",
    )
    require_equal(
        metadata.get("example_count"),
        expected_count,
        "benchmark example count",
    )
    shapes = manifest["latency_protocol"]["fixed_preprocessed_shapes"]
    for index, batch in enumerate(batches):
        require_equal(batch.get("batch_index"), index, "benchmark batch index")
        require_equal(batch.get("batch_size"), 1, "benchmark batch size")
        require_equal(
            batch.get("model_input", {}).get("shape"),
            shapes["model_input"],
            "model input shape",
        )
        require_equal(
            batch.get("image_tensor_shape"),
            shapes["image_tensor"],
            "image tensor shape",
        )
        require_equal(
            batch.get("padding_mask", {}).get("shape"),
            shapes["padding_mask"],
            "padding mask shape",
        )
    identity = manifest_generator.require_sha256(
        metadata.get("identity_sha256"),
        "benchmark input identity",
    )
    payload = {
        "schema_version": metadata.get("schema_version"),
        "batch_count": metadata.get("batch_count"),
        "example_count": metadata.get("example_count"),
        "batches": batches,
    }
    require_equal(
        sha256_bytes(manifest_generator.canonical_bytes(payload)),
        identity,
        "benchmark input identity digest",
    )
    return identity


def validate_rank_record(
    record: dict[str, Any],
    *,
    rank: int,
    candidate: dict[str, Any],
    expected_config_path: str,
    allocation_hostname: str,
    manifest: dict[str, Any],
) -> tuple[str, tuple[Any, ...], str]:
    require_equal(record.get("rank"), rank, "rank identity")
    require_equal(record.get("local_rank"), rank, "local rank identity")
    require_equal(record.get("world_size"), 8, "rank world size")
    require_equal(
        record.get("checkpoint"),
        candidate["checkpoint"]["path"],
        "rank checkpoint",
    )
    require_equal(
        record.get("config_path"),
        expected_config_path,
        "rank config path",
    )
    actual_protocol = record.get("protocol")
    if not isinstance(actual_protocol, dict):
        raise ContractError("rank protocol is missing")
    require_equal(
        actual_protocol,
        expected_rank_protocol(manifest),
        "rank latency protocol",
    )
    hardware = record.get("hardware", {})
    runtime_contract = manifest["runtime"]
    require_equal(
        hardware.get("hostname"),
        allocation_hostname,
        "rank allocation hostname",
    )
    require_equal(
        hardware.get("gpu_name"),
        runtime_contract["required_gpu_name"],
        "rank GPU name",
    )
    require_equal(
        hardware.get("compute_capability"),
        runtime_contract["required_compute_capability"],
        "rank compute capability",
    )
    require_equal(
        hardware.get("total_memory_bytes"),
        runtime_contract["required_total_memory_bytes"],
        "rank GPU memory",
    )
    nvidia_smi = hardware.get("nvidia_smi")
    if not isinstance(nvidia_smi, str) or nvidia_smi.startswith("unavailable:"):
        raise ContractError("rank nvidia-smi evidence is unavailable")
    gpu_uuid = nvidia_smi.split(",", 1)[0].strip()
    if not gpu_uuid:
        raise ContractError("rank GPU UUID is empty")
    runtime = record.get("runtime", {})
    require_equal(
        major_minor_patch(runtime.get("torch"), "rank torch"),
        runtime_contract["required_torch"],
        "rank torch release",
    )
    require_equal(
        runtime.get("cuda"),
        runtime_contract["required_cuda"],
        "rank CUDA",
    )
    require_equal(
        runtime.get("cudnn"),
        runtime_contract["required_cudnn"],
        "rank cuDNN",
    )
    signature = (
        runtime.get("python"),
        runtime.get("torch"),
        runtime.get("cuda"),
        runtime.get("cudnn"),
    )
    if not all(value is not None for value in signature):
        raise ContractError("rank runtime signature is incomplete")
    return validate_input_identity(record, manifest), signature, gpu_uuid


def cluster_bootstrap_tail_ci(
    samples: dict[int, dict[str, list[float]]],
    protocol: LatencyProtocol,
) -> list[float]:
    """Bootstrap p95 by resampling whole device-round clusters."""

    clusters = np.asarray(
        [
            samples[round_index][device_id]
            for round_index in sorted(samples)
            for device_id in sorted(samples[round_index])
        ],
        dtype=np.float64,
    )
    if (
        clusters.ndim != 2
        or clusters.shape[1] != protocol.timed_iterations
    ):
        raise ContractError("p95 cluster-bootstrap sample shape is invalid")
    rng = np.random.default_rng(protocol.bootstrap_seed)
    indices = rng.integers(
        0,
        len(clusters),
        size=(protocol.bootstrap_resamples, len(clusters)),
    )
    resampled = clusters[indices].reshape(
        protocol.bootstrap_resamples,
        -1,
    )
    estimates = np.quantile(
        resampled,
        protocol.tail_percentile / 100.0,
        axis=1,
        method="linear",
    )
    alpha = 1.0 - protocol.bootstrap_confidence_level
    low, high = np.quantile(
        estimates,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    return [float(low), float(high)]


def stats_record(
    stats: Any,
    p95_bootstrap_ci_ms: list[float],
) -> dict[str, Any]:
    return {
        "median_ms": stats.median_ms,
        "p95_ms": stats.tail_latency_ms,
        "mad_ms": stats.mad_ms,
        "iqr_ms": stats.iqr_ms,
        "robust_cv": stats.robust_cv,
        "round_median_range_ms": stats.round_median_range_ms,
        "round_drift_ms": stats.round_drift_ms,
        "device_median_range_ms": stats.device_median_range_ms,
        "bootstrap_median_ci95_ms": list(stats.bootstrap_median_ci_ms),
        "bootstrap_p95_ci95_ms": p95_bootstrap_ci_ms,
        "p95_bootstrap_cluster_unit": "device_round",
        "raw_sample_count_total": stats.raw_sample_count_total,
        "samples_per_device": stats.samples_per_device,
        "is_valid": stats.is_valid,
        "invalid_reasons": list(stats.invalid_reasons),
    }


def aggregate_bundles(
    manifest: dict[str, Any],
    manifest_file_sha256: str,
    jobs: list[dict[str, Any]],
    bundles: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(jobs) != 6 or any(not item["complete"] for item in jobs):
        raise ContractError("aggregation requires six complete matched jobs")
    expected_allocation_ids = {
        item["allocation_id"]
        for item in manifest["schedule"]["allocations"]
    }
    require_equal(
        {item.get("allocation_id") for item in jobs},
        expected_allocation_ids,
        "complete matched allocation set",
    )
    require_equal(
        set(bundles),
        expected_allocation_ids,
        "matched result-bundle set",
    )
    if len({item.get("tao_job_id") for item in jobs}) != 6 or len(
        {item.get("slurm_job_id") for item in jobs}
    ) != 6:
        raise ContractError("aggregation requires six distinct job allocations")
    protocol = latency_protocol(manifest)
    candidates = {
        item["candidate_id"]: item for item in manifest["candidates"]
    }
    allocations = {
        item["allocation_id"]: item
        for item in manifest["schedule"]["allocations"]
    }
    input_identities: dict[int, str] = {}
    runtime_signature: tuple[Any, ...] | None = None
    allocation_gpu_uuids: dict[str, dict[int, str]] = defaultdict(dict)
    scheduler_hostname_bindings: dict[str, dict[str, Any]] = {}
    raw_input_artifacts: list[dict[str, str]] = []
    measurements = []
    semantic_stage_allocation_id: str | None = None
    for job in jobs:
        allocation_id = job["allocation_id"]
        semantic_stage_allocation_id = allocation_id
        allocation = allocations[allocation_id]
        bundle = bundles[allocation_id]
        require_equal(
            job.get("allocation_index"),
            allocation["allocation_index"],
            f"{allocation_id} job allocation index",
        )
        require_equal(
            job.get("design_row_index"),
            allocation["design_row_index"],
            f"{allocation_id} job design row",
        )
        require_equal(
            job.get("candidate_order"),
            allocation["candidate_order"],
            f"{allocation_id} job candidate order",
        )
        result_root = Path(job["result_root"])
        expected_result_path = (
            result_root
            / "dino_moo_phase2_20260728"
            / "post_front_matched"
            / manifest["manifest_id"]
            / allocation_id
            / "allocation_result.json"
        )
        require_equal(
            Path(str(bundle.get("result_path", ""))),
            expected_result_path,
            f"{allocation_id} result path",
        )
        result_sha256 = manifest_generator.require_sha256(
            bundle.get("result_sha256"),
            f"{allocation_id} result SHA256",
        )
        raw_input_artifacts.append(
            {
                "kind": "allocation_result",
                "allocation_id": allocation_id,
                "candidate_id": "",
                "rank": "",
                "path": str(expected_result_path),
                "sha256": result_sha256,
            }
        )
        result = bundle.get("result", {})
        for key, expected in (
            ("schema_version", 1),
            ("status", "success"),
            ("manifest_id", manifest["manifest_id"]),
            ("manifest_sha256", manifest_file_sha256),
            ("schedule_sha256", manifest["schedule"]["schedule_sha256"]),
            ("allocation_id", allocation_id),
            ("allocation_index", allocation["allocation_index"]),
            ("design_row_index", allocation["design_row_index"]),
            ("block_plan_sha256", job["block_plan_sha256"]),
            ("tao_job_id", job["tao_job_id"]),
            ("sdk_job_scoped_result_root", str(result_root)),
            ("feeds_final_selection", False),
            ("feeds_reselection", False),
            ("selection_time_objective_replacement_permitted", False),
        ):
            require_equal(
                result.get(key),
                expected,
                f"{allocation_id} result {key}",
            )
        staged = job["staging_file_sha256"]
        expected_config_digests = {
            candidate_id: staged[f"configs/{candidate_id}.yaml"]
            for candidate_id in allocation["candidate_order"]
        }
        expected_checkpoint_digests = {
            candidate["checkpoint"]["path"]: candidate["checkpoint"]["sha256"]
            for candidate in candidates.values()
        }
        require_equal(
            result.get("verified_config_sha256"),
            expected_config_digests,
            f"{allocation_id} verified config digests",
        )
        require_equal(
            result.get("verified_checkpoint_sha256"),
            expected_checkpoint_digests,
            f"{allocation_id} verified checkpoint digests",
        )
        hardware_evidence = result.get("hardware", {})
        devices = hardware_evidence.get("devices")
        if not isinstance(devices, list) or len(devices) != EXPECTED_RANKS:
            raise ContractError(
                f"{allocation_id} top-level hardware evidence is incomplete"
            )
        for index, device in enumerate(devices):
            for key, expected in (
                ("index", index),
                ("name", manifest["runtime"]["required_gpu_name"]),
                (
                    "compute_capability",
                    manifest["runtime"]["required_compute_capability"],
                ),
                (
                    "total_memory_bytes",
                    manifest["runtime"]["required_total_memory_bytes"],
                ),
            ):
                require_equal(
                    device.get(key),
                    expected,
                    f"{allocation_id} device {index} {key}",
                )
        top_runtime = hardware_evidence.get("runtime", {})
        require_equal(
            major_minor_patch(top_runtime.get("torch"), "allocation torch"),
            manifest["runtime"]["required_torch"],
            f"{allocation_id} allocation torch",
        )
        require_equal(
            top_runtime.get("cuda"),
            manifest["runtime"]["required_cuda"],
            f"{allocation_id} allocation CUDA",
        )
        require_equal(
            top_runtime.get("cudnn"),
            manifest["runtime"]["required_cudnn"],
            f"{allocation_id} allocation cuDNN",
        )
        runs = result.get("candidate_runs")
        expected_count = len(candidates)
        if not isinstance(runs, list) or len(runs) != expected_count:
            raise ContractError(
                f"{allocation_id}: incomplete allocation candidate set"
            )
        require_equal(
            [item.get("candidate_id") for item in runs],
            allocation["candidate_order"],
            f"{allocation_id} candidate order",
        )
        require_equal(
            [item.get("position") for item in runs],
            list(range(expected_count)),
            f"{allocation_id} positions",
        )
        require_equal(
            set(bundle.get("rank_records", {})),
            set(candidates),
            f"{allocation_id} rank-record candidate set",
        )
        if any(
            item.get("status") != "success" or item.get("exit_code") != 0
            for item in runs
        ):
            raise ContractError(
                f"{allocation_id}: partial allocation results are forbidden"
            )
        hostname = result.get("hostname")
        if not isinstance(hostname, str) or not hostname:
            raise ContractError(f"{allocation_id}: missing hostname")
        scheduler_hostname_bindings[allocation_id] = (
            validate_scheduler_hostname_binding(job, hostname)
        )
        for run in runs:
            candidate_id = run["candidate_id"]
            candidate = candidates[candidate_id]
            position = run["position"]
            expected_run_label = (
                f"{allocation_id}_p{position:03d}_{candidate_id}"
            )
            require_equal(
                run.get("run_label"),
                expected_run_label,
                f"{allocation_id}/{candidate_id} run label",
            )
            require_equal(
                run.get("checkpoint_path"),
                candidate["checkpoint"]["path"],
                f"{allocation_id}/{candidate_id} checkpoint path",
            )
            require_equal(
                run.get("checkpoint_sha256"),
                candidate["checkpoint"]["sha256"],
                f"{allocation_id}/{candidate_id} checkpoint SHA256",
            )
            require_equal(
                run.get("resolved_model_spec_sha256"),
                candidate["resolved_model_spec_sha256"],
                f"{allocation_id}/{candidate_id} model SHA256",
            )
            require_equal(
                run.get("candidate_table_record_sha256"),
                candidate["candidate_table_record_sha256"],
                f"{allocation_id}/{candidate_id} table-record SHA256",
            )
            require_equal(
                run.get("config_sha256"),
                expected_config_digests[candidate_id],
                f"{allocation_id}/{candidate_id} config SHA256",
            )
            expected_raw_dir = (
                expected_result_path.parent
                / "candidates"
                / expected_run_label
                / job["tao_job_id"]
                / "latency"
            )
            require_equal(
                Path(str(run.get("raw_samples_dir", ""))),
                expected_raw_dir,
                f"{allocation_id}/{candidate_id} raw-sample path",
            )
            raw = bundle.get("rank_records", {}).get(candidate_id)
            if not isinstance(raw, dict):
                raise ContractError(
                    f"{allocation_id}/{candidate_id}: rank bundle is missing"
                )
            paths = raw.get("paths")
            file_sha256 = raw.get("sha256")
            records = raw.get("records")
            expected_names = [f"rank_{rank}.json" for rank in range(8)]
            expected_paths = [
                str(expected_raw_dir / filename)
                for filename in expected_names
            ]
            if (
                not isinstance(paths, list)
                or paths != expected_paths
                or not isinstance(file_sha256, list)
                or len(file_sha256) != 8
                or not isinstance(records, list)
                or len(records) != 8
            ):
                raise ContractError(
                    f"{allocation_id}/{candidate_id}: expected rank_0..7"
                )
            for rank, (path, digest) in enumerate(
                zip(paths, file_sha256)
            ):
                manifest_generator.require_sha256(
                    digest,
                    f"{allocation_id}/{candidate_id}/rank_{rank} SHA256",
                )
                raw_input_artifacts.append(
                    {
                        "kind": "rank_record",
                        "allocation_id": allocation_id,
                        "candidate_id": candidate_id,
                        "rank": str(rank),
                        "path": path,
                        "sha256": digest,
                    }
                )
            samples = {
                round_index: {}
                for round_index in range(protocol.repeated_rounds)
            }
            for rank, record in enumerate(records):
                identity, signature, gpu_uuid = validate_rank_record(
                    record,
                    rank=rank,
                    candidate=candidate,
                    expected_config_path=str(run.get("config_path", "")),
                    allocation_hostname=hostname,
                    manifest=manifest,
                )
                prior_input = input_identities.setdefault(rank, identity)
                semantic_stage_allocation_id = None
                require_equal(
                    identity,
                    prior_input,
                    f"rank {rank} benchmark input identity",
                )
                semantic_stage_allocation_id = allocation_id
                if runtime_signature is None:
                    runtime_signature = signature
                else:
                    semantic_stage_allocation_id = None
                    require_equal(
                        signature,
                        runtime_signature,
                        "runtime signature",
                    )
                    semantic_stage_allocation_id = allocation_id
                prior_uuid = allocation_gpu_uuids[allocation_id].setdefault(
                    rank,
                    gpu_uuid,
                )
                require_equal(
                    gpu_uuid,
                    prior_uuid,
                    f"{allocation_id} rank {rank} GPU identity",
                )
                rank_samples = record.get("samples_ms")
                if (
                    not isinstance(rank_samples, list)
                    or len(rank_samples) != protocol.repeated_rounds
                ):
                    raise ContractError("rank samples have wrong round count")
                for round_index, values in enumerate(rank_samples):
                    if (
                        not isinstance(values, list)
                        or len(values) != protocol.timed_iterations
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not math.isfinite(float(value))
                            or float(value) <= 0
                            for value in values
                        )
                    ):
                        raise ContractError("rank timed samples are invalid")
                    samples[round_index][str(rank)] = values
            stats = aggregate_synchronized_latency(samples, protocol)
            if not stats.is_valid:
                raise ContractError(
                    f"{allocation_id}/{candidate_id}: {stats.validity_reason}"
                )
            measurements.append(
                {
                    "allocation_id": allocation_id,
                    "allocation_index": allocation["allocation_index"],
                    "tao_job_id": job["tao_job_id"],
                    "slurm_job_id": job["slurm_job_id"],
                    "node_list": job["node_list"],
                    "expanded_node_hostnames": copy.deepcopy(
                        job["expanded_node_hostnames"]
                    ),
                    "normalized_node_hostname": job[
                        "normalized_node_hostname"
                    ],
                    "hostname": hostname,
                    "candidate_id": candidate_id,
                    "position": run["position"],
                    "checkpoint_sha256": candidate["checkpoint"]["sha256"],
                    "input_identity_sha256_by_rank": {
                        str(rank): input_identities[rank]
                        for rank in range(8)
                    },
                    **stats_record(
                        stats,
                        cluster_bootstrap_tail_ci(samples, protocol),
                    ),
                }
            )
        semantic_stage_allocation_id = None
    expected_measurements = 6 * len(candidates)
    require_equal(
        len(measurements),
        expected_measurements,
        "matched measurement count",
    )
    gpu_counts = {
        allocation_id: len(set(by_rank.values()))
        for allocation_id, by_rank in allocation_gpu_uuids.items()
    }
    if any(count != 8 for count in gpu_counts.values()):
        raise ContractError("each allocation must contain eight distinct GPUs")
    return measurements, {
        "hardware_contract": "pass",
        "runtime_contract": "pass",
        "protocol_contract": "pass",
        "benchmark_input_identity": "pass",
        "complete_block_contract": "pass",
        "rank_files_per_candidate": 8,
        "runtime_signature": list(runtime_signature or ()),
        "gpu_uuid_count_by_allocation": gpu_counts,
        "scheduler_hostname_binding": "pass",
        "scheduler_hostname_bindings": scheduler_hostname_bindings,
        "raw_input_artifact_count": len(raw_input_artifacts),
        "raw_input_artifacts": raw_input_artifacts,
        "raw_input_inventory_sha256": (
            manifest_generator.sha256_value(raw_input_artifacts)
        ),
    }


def semantic_failure_allocation_ids(
    error: BaseException,
    manifest: dict[str, Any],
) -> list[str]:
    """Return exact allocation attribution, or zero/many to force a block."""

    allocation_ids = {
        item["allocation_id"]
        for item in manifest["schedule"]["allocations"]
    }
    attributed = {
        allocation_id
        for allocation_id in allocation_ids
        if allocation_id in str(error)
    }
    traceback = error.__traceback__
    while traceback is not None:
        local_value = traceback.tb_frame.f_locals.get(
            "semantic_stage_allocation_id"
        )
        if local_value in allocation_ids:
            attributed.add(local_value)
        traceback = traceback.tb_next
    return sorted(attributed)


def deterministic_failure_record(
    *,
    stage: str,
    error: BaseException,
) -> dict[str, Any]:
    """Canonicalize the observable failure without timestamps or traceback."""

    record = {
        "stage": stage,
        "exception_type": type(error).__name__,
        "message": str(error) or type(error).__name__,
    }
    record["error_sha256"] = manifest_generator.sha256_value(record)
    return record


def build_invalidation_evidence(
    *,
    manifest: dict[str, Any],
    manifest_file_sha256: str,
    ledger: dict[str, Any],
    ledger_file_sha256: str,
    ledger_path: Path,
    jobs: list[dict[str, Any]],
    allocation_ids: list[str],
    failure: dict[str, Any],
    available_artifacts: list[dict[str, str]],
    artifact_probe: dict[str, Any],
) -> dict[str, Any]:
    """Build immutable proof that authorizes exactly one full-block rerun."""

    expected_ids = {
        item["allocation_id"]
        for item in manifest["schedule"]["allocations"]
    }
    if (
        allocation_ids != sorted(allocation_ids)
        or len(set(allocation_ids)) != len(allocation_ids)
        or any(item not in expected_ids for item in allocation_ids)
    ):
        raise ContractError("invalidation attribution set is invalid")
    if len(jobs) != EXPECTED_ALLOCATIONS or not all(
        item.get("complete") is True for item in jobs
    ):
        raise ContractError(
            "invalidation evidence requires six Complete SDK/SLURM jobs"
        )
    by_job = {item["allocation_id"]: item for item in jobs}
    by_submission = {
        item["allocation_id"]: item for item in ledger["submissions"]
    }
    exact = len(allocation_ids) == 1
    allocation_id = allocation_ids[0] if exact else None
    canonical_artifacts = sorted(
        available_artifacts,
        key=lambda item: (
            item["kind"],
            item["allocation_id"],
            item["candidate_id"],
            item["rank"],
            item["path"],
        ),
    )
    require_equal(
        available_artifacts,
        canonical_artifacts,
        "invalidation available-artifact order",
    )
    if exact and any(
        item.get("allocation_id") != allocation_id
        for item in canonical_artifacts
    ):
        raise ContractError(
            "single-allocation evidence contains another allocation"
        )
    prior_identity = None
    job_status = None
    if exact:
        submission = by_submission[allocation_id]
        job = by_job[allocation_id]
        prior_identity = {
            "allocation_id": allocation_id,
            "tao_job_id": submission["tao_job_id"],
            "slurm_job_id": str(submission["slurm_job_id"]),
            "command_sha256": submission["command_sha256"],
            "block_plan_sha256": submission["block_plan_sha256"],
        }
        job_status = {
            "sdk_status": job["sdk_status"],
            "slurm_state": job["slurm_state"],
            "slurm_exit_code": job["slurm_exit_code"],
            "complete": job["complete"],
        }
        for key, expected in (
            ("sdk_status", "Complete"),
            ("slurm_state", "COMPLETED"),
            ("slurm_exit_code", "0:0"),
            ("complete", True),
        ):
            require_equal(
                job_status[key],
                expected,
                f"invalidation Complete job {key}",
            )
        status = "single_allocation_invalid_replacement_authorized"
        block_reason = None
    else:
        status = "replacement_blocked_unattributed_failure"
        block_reason = (
            "failure_could_not_be_attributed_to_any_allocation"
            if not allocation_ids
            else "failure_implicates_multiple_allocations"
        )
    source = (
        manifest.get("source_artifacts", {})
        .get("post_front_tools", {})
        .get(launcher.AGGREGATOR.name)
    )
    if not isinstance(source, dict):
        raise ContractError(
            "manifest lacks pinned post-front aggregator provenance"
        )
    payload = {
        "schema_version": 1,
        "evidence_id": "dino_post_front_complete_invalid_allocation_v1",
        "status": status,
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_file_sha256,
        "manifest_internal_sha256": manifest["manifest_sha256"],
        "schedule_sha256": manifest["schedule"]["schedule_sha256"],
        "complete_ledger": {
            "path": str(ledger_path.resolve()),
            "whole_file_sha256": ledger_file_sha256,
            "internal_sha256": ledger["ledger_sha256"],
            "revision": ledger["ledger_revision"],
            "status": ledger["status"],
        },
        "all_jobs_complete": True,
        "allocation_id": allocation_id,
        "attribution": {
            "status": (
                "exactly_one_allocation"
                if exact
                else "not_exactly_one_allocation"
            ),
            "allocation_ids": allocation_ids,
            "allocation_count": len(allocation_ids),
            "replacement_blocked": not exact,
            "block_reason": block_reason,
        },
        "prior_submission_identity": prior_identity,
        "job_status": job_status,
        "implicated_job_identities": [
            {
                "allocation_id": item,
                "tao_job_id": by_submission[item]["tao_job_id"],
                "slurm_job_id": str(by_submission[item]["slurm_job_id"]),
            }
            for item in allocation_ids
        ],
        "failure": copy.deepcopy(failure),
        "available_artifacts": canonical_artifacts,
        "available_artifact_count": len(canonical_artifacts),
        "available_artifacts_sha256": (
            manifest_generator.sha256_value(canonical_artifacts)
        ),
        "artifact_probe": copy.deepcopy(artifact_probe),
        "aggregator_source": copy.deepcopy(source),
        "replacement_permitted": exact,
        "full_allocation_discarded": exact,
        "partial_measurements_reused": False,
        "partial_measurements_used_for_analysis": False,
        "feeds_final_selection": False,
        "feeds_reselection": False,
        "selection_time_objectives_replaced": False,
    }
    payload["invalidation_sha256"] = manifest_generator.sha256_value(payload)
    return payload


def write_invalidation_evidence(
    *,
    manifest: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, str]:
    """Create once, or accept only an exact immutable replay."""

    path = launcher.invalidation_evidence_path(
        Path(manifest["runtime"]["local_runtime_path"]),
        payload["complete_ledger"]["revision"],
        payload["allocation_id"],
    )
    if path.exists():
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o022
        ):
            raise ContractError("existing invalidation evidence file is unsafe")
        existing = manifest_generator.load_json(path)
        manifest_generator.validate_internal_digest(
            existing,
            "invalidation_sha256",
            "existing invalidation evidence",
        )
        require_equal(
            existing,
            payload,
            "immutable invalidation evidence replay",
        )
    else:
        launcher.atomic_create_json(path, payload)
    return {
        "path": str(path),
        "whole_file_sha256": manifest_generator.sha256_file(path),
        "internal_sha256": payload["invalidation_sha256"],
    }


def invalidation_status_report(
    *,
    manifest: dict[str, Any],
    manifest_file_sha256: str,
    ledger: dict[str, Any],
    ledger_file_sha256: str,
    jobs: list[dict[str, Any]],
    evidence: dict[str, Any],
    evidence_reference: dict[str, str],
    loaded_keys: list[str],
    database: Path,
    runtime_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Expose the blocked/authorized outcome without any partial analysis."""

    return {
        "schema_version": 1,
        "status": evidence["status"],
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_file_sha256,
        "submission_ledger_sha256": ledger_file_sha256,
        "submission_ledger_revision": ledger["ledger_revision"],
        "invalidation_evidence": evidence_reference,
        "allocation_id": evidence["allocation_id"],
        "attribution": copy.deepcopy(evidence["attribution"]),
        "failure": copy.deepcopy(evidence["failure"]),
        "available_artifact_count": evidence["available_artifact_count"],
        "artifact_probe": copy.deepcopy(evidence["artifact_probe"]),
        "loaded_secret_keys": loaded_keys,
        "secret_values_recorded": False,
        "sdk_database_path": str(database),
        "aggregation_runtime_provenance": runtime_provenance,
        "jobs": copy.deepcopy(jobs),
        "full_allocation_discarded": evidence[
            "full_allocation_discarded"
        ],
        "partial_measurements_used": False,
        "partial_measurements_reused": False,
        "feeds_final_selection": False,
        "feeds_reselection": False,
        "selection_time_objectives_replaced": False,
        "replacement_permitted": evidence["replacement_permitted"],
    }


def paired_bootstrap_ci(
    values: list[float],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> list[float]:
    if (
        not values
        or any(not math.isfinite(value) for value in values)
        or resamples <= 0
        or not 0.0 < confidence < 1.0
    ):
        raise ContractError("paired-bootstrap inputs are invalid")
    source = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(source), size=(resamples, len(source)))
    estimates = np.median(source[indices], axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(
        estimates,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    return [float(low), float(high)]


def distribution_summary(values: list[float]) -> dict[str, Any]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ContractError("distribution values must be finite and non-empty")
    array = np.asarray(values, dtype=np.float64)
    q1, q3 = np.quantile(array, [0.25, 0.75], method="linear")
    median = float(np.median(array))
    return {
        "allocation_count": len(values),
        "values_ms": values,
        "median_ms": median,
        "mean_ms": float(np.mean(array)),
        "sample_stdev_ms": (
            float(statistics.stdev(values)) if len(values) > 1 else 0.0
        ),
        "min_ms": float(np.min(array)),
        "max_ms": float(np.max(array)),
        "range_ms": float(np.max(array) - np.min(array)),
        "mad_ms": float(np.median(np.abs(array - median))),
        "iqr_ms": float(q3 - q1),
    }


def classify_difference(
    delta: float,
    ci: list[float],
    tolerance: float,
) -> dict[str, str]:
    if ci[1] < -tolerance:
        interval = "entirely_below_negative_tolerance"
    elif ci[0] > tolerance:
        interval = "entirely_above_positive_tolerance"
    elif ci[0] >= -tolerance and ci[1] <= tolerance:
        interval = "entirely_within_practical_tolerance"
    else:
        interval = "crosses_a_practical_tolerance_boundary"
    if delta < -tolerance:
        point = "first_practically_faster"
    elif delta > tolerance:
        point = "second_practically_faster"
    else:
        point = "practically_equivalent"
    return {
        "point_classification": point,
        "descriptive_bootstrap_interval_classification": interval,
    }


def exact_shifted_sign_flip_test(
    values: list[float],
    *,
    boundary_ms: float,
    alternative: str,
) -> dict[str, Any]:
    """Exact one-sided paired sign-flip test at a practical boundary.

    The null randomization distribution enumerates every sign assignment to
    the absolute boundary-shifted paired differences.  This is deliberately
    exact for the six-allocation matched design; the percentile bootstrap is
    retained separately as descriptive uncertainty only.
    """

    if (
        not values
        or len(values) > 20
        or any(not math.isfinite(value) for value in values)
        or not math.isfinite(boundary_ms)
        or alternative not in {"less", "greater"}
    ):
        raise ContractError("exact shifted sign-flip inputs are invalid")
    shifted = [float(value - boundary_ms) for value in values]
    magnitudes = [abs(value) for value in shifted]
    observed = float(math.fsum(shifted))
    permutation_count = 1 << len(shifted)
    scale = max(
        1.0,
        abs(observed),
        math.fsum(magnitudes),
    )
    epsilon = 1e-12 * scale
    extreme_count = 0
    for mask in range(permutation_count):
        statistic = math.fsum(
            magnitude if mask & (1 << index) else -magnitude
            for index, magnitude in enumerate(magnitudes)
        )
        if alternative == "less":
            extreme_count += statistic <= observed + epsilon
        else:
            extreme_count += statistic >= observed - epsilon
    return {
        "method": (
            "exact paired sign-flip permutation test on differences "
            "shifted by the practical-tolerance boundary"
        ),
        "alternative": alternative,
        "boundary_ms": boundary_ms,
        "shifted_differences_ms": shifted,
        "test_statistic": "sum of boundary-shifted paired differences",
        "observed_statistic_ms": observed,
        "allocation_count": len(values),
        "permutation_count": permutation_count,
        "extreme_permutation_count": extreme_count,
        "p_value_one_sided": extreme_count / permutation_count,
    }


def directional_pairwise_evidence(
    values: list[float],
    *,
    tolerance: float,
    confidence: float,
) -> dict[str, Any]:
    """Gate a pairwise direction on exact evidence and all-six margins."""

    if (
        len(values) != EXPECTED_ALLOCATIONS
        or any(not math.isfinite(value) for value in values)
        or not math.isfinite(tolerance)
        or tolerance < 0.0
        or not 0.0 < confidence < 1.0
    ):
        raise ContractError("pairwise directional-evidence inputs are invalid")
    alpha = 1.0 - confidence
    first_test = exact_shifted_sign_flip_test(
        values,
        boundary_ms=-tolerance,
        alternative="less",
    )
    second_test = exact_shifted_sign_flip_test(
        values,
        boundary_ms=tolerance,
        alternative="greater",
    )
    first_all = all(value < -tolerance for value in values)
    second_all = all(value > tolerance for value in values)
    first_exact = first_test["p_value_one_sided"] <= alpha
    second_exact = second_test["p_value_one_sided"] <= alpha
    if first_all and first_exact:
        claim = "first_stably_faster"
    elif second_all and second_exact:
        claim = "second_stably_faster"
    else:
        claim = "no_stable_directional_claim"
    return {
        "scope": "pairwise_only",
        "simultaneous_order_inference_permitted": False,
        "alpha": alpha,
        "practical_tolerance_ms": tolerance,
        "all_six_beyond_negative_tolerance": first_all,
        "all_six_beyond_positive_tolerance": second_all,
        "first_faster_exact_test_passes": first_exact,
        "second_faster_exact_test_passes": second_exact,
        "first_faster_test": first_test,
        "second_faster_test": second_test,
        "directional_claim": claim,
        "claim_rule": (
            "A direction requires its one-sided exact shifted sign-flip "
            "p-value <= alpha and every one of the six allocation-paired "
            "differences strictly beyond the corresponding practical "
            "tolerance boundary."
        ),
    }


def comparative_analysis(
    manifest: dict[str, Any],
    measurements: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_ids = manifest["candidate_derivation"]["candidate_ids"]
    allocation_ids = [
        item["allocation_id"] for item in manifest["schedule"]["allocations"]
    ]
    by_key = {
        (item["allocation_id"], item["candidate_id"]): item
        for item in measurements
    }
    expected_keys = {
        (allocation_id, candidate_id)
        for allocation_id in allocation_ids
        for candidate_id in candidate_ids
    }
    require_equal(
        len(measurements),
        len(expected_keys),
        "matched comparison measurement count",
    )
    require_equal(set(by_key), expected_keys, "matched comparison matrix")
    paired = manifest["paired_analysis"]
    tolerance = paired["practical_tolerance_ms"]
    resamples = paired["bootstrap_resamples"]
    confidence = paired["bootstrap_confidence_level"]
    seed = paired["bootstrap_seed"]
    between = []
    aggregate_medians = {}
    for candidate_id in candidate_ids:
        medians = [
            by_key[(allocation_id, candidate_id)]["median_ms"]
            for allocation_id in allocation_ids
        ]
        p95s = [
            by_key[(allocation_id, candidate_id)]["p95_ms"]
            for allocation_id in allocation_ids
        ]
        median_summary = distribution_summary(medians)
        aggregate_medians[candidate_id] = median_summary["median_ms"]
        between.append(
            {
                "candidate_id": candidate_id,
                "median_latency": median_summary,
                "p95_latency": distribution_summary(p95s),
            }
        )
    pairs = []
    stable_edges = []
    for first_index, first in enumerate(candidate_ids):
        for second in candidate_ids[first_index + 1 :]:
            median_differences = [
                by_key[(allocation_id, first)]["median_ms"]
                - by_key[(allocation_id, second)]["median_ms"]
                for allocation_id in allocation_ids
            ]
            p95_differences = [
                by_key[(allocation_id, first)]["p95_ms"]
                - by_key[(allocation_id, second)]["p95_ms"]
                for allocation_id in allocation_ids
            ]
            median_delta = float(np.median(median_differences))
            p95_delta = float(np.median(p95_differences))
            median_ci = paired_bootstrap_ci(
                median_differences,
                resamples=resamples,
                confidence=confidence,
                seed=seed,
            )
            p95_ci = paired_bootstrap_ci(
                p95_differences,
                resamples=resamples,
                confidence=confidence,
                seed=seed,
            )
            classification = classify_difference(
                median_delta,
                median_ci,
                tolerance,
            )
            p95_classification = classify_difference(
                p95_delta,
                p95_ci,
                tolerance,
            )
            directional_evidence = directional_pairwise_evidence(
                median_differences,
                tolerance=tolerance,
                confidence=confidence,
            )
            p95_directional_evidence = directional_pairwise_evidence(
                p95_differences,
                tolerance=tolerance,
                confidence=confidence,
            )
            pair = {
                "first_candidate_id": first,
                "second_candidate_id": second,
                "delta_convention": (
                    "first minus second; negative means first is faster"
                ),
                "allocation_ids": allocation_ids,
                "paired_median_differences_ms": median_differences,
                "median_paired_difference_ms": median_delta,
                "median_paired_bootstrap_ci95_ms": median_ci,
                "median_bootstrap_ci_is_descriptive_only": True,
                "paired_p95_differences_ms": p95_differences,
                "median_paired_p95_difference_ms": p95_delta,
                "p95_paired_bootstrap_ci95_ms": p95_ci,
                "p95_bootstrap_ci_is_descriptive_only": True,
                "p95_point_classification": p95_classification[
                    "point_classification"
                ],
                "p95_descriptive_bootstrap_interval_classification": (
                    p95_classification[
                        "descriptive_bootstrap_interval_classification"
                    ]
                ),
                "p95_pairwise_directional_evidence": (
                    p95_directional_evidence
                ),
                "pairwise_directional_evidence": directional_evidence,
                "pairwise_directional_claim": directional_evidence[
                    "directional_claim"
                ],
                "practical_tolerance_ms": tolerance,
                **classification,
            }
            pairs.append(pair)
            if (
                directional_evidence["directional_claim"]
                == "first_stably_faster"
            ):
                stable_edges.append(
                    {
                        "faster_candidate_id": first,
                        "slower_candidate_id": second,
                        "scope": "pairwise_only",
                        "simultaneous_order_inference_permitted": False,
                    }
                )
            elif (
                directional_evidence["directional_claim"]
                == "second_stably_faster"
            ):
                stable_edges.append(
                    {
                        "faster_candidate_id": second,
                        "slower_candidate_id": first,
                        "scope": "pairwise_only",
                        "simultaneous_order_inference_permitted": False,
                    }
                )
    descriptive_order = sorted(
        candidate_ids,
        key=lambda item: (aggregate_medians[item], item),
    )
    stable_edge_set = {
        (item["faster_candidate_id"], item["slower_candidate_id"])
        for item in stable_edges
    }
    adjacent = [
        {
            "faster_candidate_id": descriptive_order[index],
            "slower_candidate_id": descriptive_order[index + 1],
            "pairwise_directional_claim_established": (
                descriptive_order[index],
                descriptive_order[index + 1],
            )
            in stable_edge_set,
            "scope": "pairwise_only",
            "simultaneous_order_inference_permitted": False,
        }
        for index in range(len(descriptive_order) - 1)
    ]
    return {
        "practical_tolerance_ms": tolerance,
        "paired_bootstrap": {
            "unit": "allocation",
            "resamples": resamples,
            "confidence_level": confidence,
            "seed": seed,
            "statistic": "median of allocation-paired differences",
            "confidence_interval_scope": (
                "descriptive per-comparison percentile interval; no "
                "multiplicity adjustment and never used alone for a "
                "stable-direction claim"
            ),
        },
        "directional_inference": {
            "method": (
                "exact one-sided paired sign-flip permutation test after "
                "shifting by the practical-tolerance boundary"
            ),
            "allocation_count": EXPECTED_ALLOCATIONS,
            "permutation_count": 1 << EXPECTED_ALLOCATIONS,
            "alpha": 1.0 - confidence,
            "additional_unanimity_requirement": (
                "all six allocation-paired differences must be strictly "
                "beyond the claimed +/- practical-tolerance boundary"
            ),
            "scope": "pairwise_only",
            "multiplicity_adjustment": "none",
            "simultaneous_total_order_inference_permitted": False,
        },
        "between_allocation_statistics": between,
        "all_pairwise_comparisons": pairs,
        "descriptive_latency_order": descriptive_order,
        "adjacent_order_stability": adjacent,
        "stable_ordering_claims": stable_edges,
        "stable_ordering_claims_scope": "pairwise_only",
        "descriptive_order_is_a_stable_total_order": False,
        "stable_total_order_claim_applicable": False,
        "ordering_claim_policy": (
            "Pairwise direction is emitted only when the one-sided exact "
            "shifted sign-flip test passes and all six paired differences "
            f"are strictly beyond the preregistered +/-{tolerance} ms "
            "practical tolerance. Bootstrap intervals remain descriptive. "
            "Unadjusted pairwise evidence never implies a simultaneous "
            "total order."
        ),
    }


def build_final_report(
    manifest: dict[str, Any],
    manifest_file_sha256: str,
    ledger: dict[str, Any],
    ledger_file_sha256: str,
    jobs: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    consistency: dict[str, Any],
    runtime_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selector_replay_validated = (
        manifest_generator.require_sha256(
            ledger.get("source_checks", {}).get(
                "selector_replay_proof_sha256"
            ),
            "aggregation selector-replay proof SHA256",
        )
        if "selector_replay_proof_sha256"
        in ledger.get("source_checks", {})
        else None
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": timestamp(),
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_file_sha256,
        "submission_ledger_sha256": ledger_file_sha256,
        "submission_ledger_revision": ledger["ledger_revision"],
        "submission_supersession_history": copy.deepcopy(
            ledger["superseded_submissions"]
        ),
        "submission_recovery_events": copy.deepcopy(
            ledger.get("submission_recovery_events", [])
        ),
        "complete_invalid_allocation_replacements": [
            copy.deepcopy(item)
            for item in ledger["superseded_submissions"]
            if item.get("prior_sdk_status") == "Complete"
        ],
        "schedule_sha256": manifest["schedule"]["schedule_sha256"],
        "candidate_ids": manifest["candidate_derivation"]["candidate_ids"],
        "source_checks": copy.deepcopy(ledger["source_checks"]),
        "aggregation_runtime_provenance": copy.deepcopy(
            runtime_provenance or {}
        ),
        "jobs": copy.deepcopy(jobs),
        "artifact_consistency": consistency,
        "per_allocation_candidate_measurements": measurements,
        "analysis": comparative_analysis(manifest, measurements),
        "original_selection_snapshot": copy.deepcopy(
            manifest["selection_snapshot"]
        ),
        "selection_isolation": {
            "frozen_archive_selector_replay_performed_during_source_validation": (
                selector_replay_validated is not None
            ),
            "selector_replay_result_used_only_for_candidate_set_integrity": (
                selector_replay_validated is not None
            ),
            "selector_replay_proof_sha256": selector_replay_validated,
            "postfront_measurements_loaded_after_selector_replay": (
                selector_replay_validated is not None
            ),
            "selector_invoked_on_postfront_measurements": False,
            "measurements_feed_selection": False,
            "measurements_feed_reselection": False,
            "selection_time_objectives_replaced": False,
            "algorithm_selected_candidate_overridden": False,
            "allowed_use": "stability analysis and hypothesis verdict only",
        },
    }
    report["report_sha256"] = manifest_generator.sha256_value(report)
    return report


def write_new_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except FileExistsError as error:
        raise ContractError(
            f"refusing to overwrite immutable analysis: {path}"
        ) from error


def main() -> int:
    args = parse_args()
    manifest, manifest_file_sha256 = launcher.load_manifest(
        args.manifest.resolve(),
        args.manifest_file_sha256,
    )
    runtime_dir = Path(manifest["runtime"]["local_runtime_path"]).resolve()
    require_equal(
        args.submission_ledger.resolve(),
        runtime_dir / "block_submissions.json",
        "CLI submission-ledger path",
    )
    require_equal(
        args.sdk_state.resolve(),
        runtime_dir / "slurm_state.json",
        "CLI SDK-state path",
    )
    require_equal(
        args.output.resolve(),
        runtime_dir / "post_front_matched_analysis.json",
        "CLI analysis-output path",
    )
    require_equal(
        args.secrets_env.resolve(),
        Path(manifest["runtime"]["secrets_env_path"]).resolve(),
        "CLI secrets-env path",
    )
    source_checks = launcher.validate_final_source_evidence(manifest)
    runtime_provenance = aggregation_runtime_provenance(manifest)
    ledger, ledger_file_sha256 = load_ledger(
        args.submission_ledger.resolve(),
        args.submission_ledger_sha256,
        manifest,
        manifest_file_sha256,
        source_checks,
    )
    loaded_keys = launcher.load_env_file(args.secrets_env.resolve())
    jobs, database = inspect_jobs(
        manifest,
        ledger,
        args.sdk_state.resolve(),
    )
    if not all(item["complete"] for item in jobs):
        status = {
            "schema_version": 1,
            "status": "pending_or_failed_no_partial_aggregation",
            "checked_at_utc": timestamp(),
            "manifest_id": manifest["manifest_id"],
            "manifest_sha256": manifest_file_sha256,
            "submission_ledger_sha256": ledger_file_sha256,
            "submission_recovery_events": copy.deepcopy(
                ledger["submission_recovery_events"]
            ),
            "loaded_secret_keys": loaded_keys,
            "secret_values_recorded": False,
            "sdk_database_path": str(database),
            "aggregation_runtime_provenance": runtime_provenance,
            "jobs": jobs,
            "partial_measurements_used": False,
            "feeds_final_selection": False,
            "feeds_reselection": False,
        }
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
        return 2
    bundles: dict[str, dict[str, Any]] = {}
    for job in jobs:
        try:
            bundles[job["allocation_id"]] = fetch_allocation_bundle(
                manifest,
                job,
            )
        except Exception as error:
            try:
                available_artifacts, artifact_probe = (
                    probe_available_allocation_artifacts(manifest, job)
                )
            except Exception as probe_error:
                available_artifacts = []
                artifact_probe = {
                    "status": "failed",
                    "available_artifact_count": 0,
                    "error": deterministic_failure_record(
                        stage="artifact_hash_probe",
                        error=probe_error,
                    ),
                }
            evidence = build_invalidation_evidence(
                manifest=manifest,
                manifest_file_sha256=manifest_file_sha256,
                ledger=ledger,
                ledger_file_sha256=ledger_file_sha256,
                ledger_path=args.submission_ledger.resolve(),
                jobs=jobs,
                allocation_ids=[job["allocation_id"]],
                failure=deterministic_failure_record(
                    stage="allocation_result_fetch",
                    error=error,
                ),
                available_artifacts=available_artifacts,
                artifact_probe=artifact_probe,
            )
            reference = write_invalidation_evidence(
                manifest=manifest,
                payload=evidence,
            )
            status = invalidation_status_report(
                manifest=manifest,
                manifest_file_sha256=manifest_file_sha256,
                ledger=ledger,
                ledger_file_sha256=ledger_file_sha256,
                jobs=jobs,
                evidence=evidence,
                evidence_reference=reference,
                loaded_keys=loaded_keys,
                database=database,
                runtime_provenance=runtime_provenance,
            )
            print(json.dumps(status, indent=2, sort_keys=True), flush=True)
            return 3
    try:
        measurements, consistency = aggregate_bundles(
            manifest,
            manifest_file_sha256,
            jobs,
            bundles,
        )
    except Exception as error:
        allocation_ids = semantic_failure_allocation_ids(error, manifest)
        artifact_scope = (
            allocation_ids
            if len(allocation_ids) == 1
            else sorted(bundles)
        )
        available_artifacts = sorted(
            [
                artifact
                for allocation_id in artifact_scope
                for artifact in available_artifacts_from_bundle(
                    allocation_id,
                    bundles.get(allocation_id),
                )
            ],
            key=lambda item: (
                item["kind"],
                item["allocation_id"],
                item["candidate_id"],
                item["rank"],
                item["path"],
            ),
        )
        evidence = build_invalidation_evidence(
            manifest=manifest,
            manifest_file_sha256=manifest_file_sha256,
            ledger=ledger,
            ledger_file_sha256=ledger_file_sha256,
            ledger_path=args.submission_ledger.resolve(),
            jobs=jobs,
            allocation_ids=allocation_ids,
            failure=deterministic_failure_record(
                stage="semantic_aggregation",
                error=error,
            ),
            available_artifacts=available_artifacts,
            artifact_probe={
                "status": "not_required_bundle_hashes_already_fetched",
                "allocation_ids": artifact_scope,
                "available_artifact_count": len(available_artifacts),
            },
        )
        reference = write_invalidation_evidence(
            manifest=manifest,
            payload=evidence,
        )
        status = invalidation_status_report(
            manifest=manifest,
            manifest_file_sha256=manifest_file_sha256,
            ledger=ledger,
            ledger_file_sha256=ledger_file_sha256,
            jobs=jobs,
            evidence=evidence,
            evidence_reference=reference,
            loaded_keys=loaded_keys,
            database=database,
            runtime_provenance=runtime_provenance,
        )
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
        return 3
    report = build_final_report(
        manifest,
        manifest_file_sha256,
        ledger,
        ledger_file_sha256,
        jobs,
        measurements,
        consistency,
        runtime_provenance,
    )
    write_new_report(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
