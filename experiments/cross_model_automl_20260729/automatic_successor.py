#!/usr/bin/env python3

"""Fail-closed automatic handoff from DINO to a sealed successor campaign.

The already-running DINO controller cannot import code added after it started.
This watcher therefore attaches to its immutable runtime directory, waits for
all three mode controllers to terminate, validates the completed objective-aware
campaign evidence, and executes exactly one pre-registered successor command.

The watcher never submits a fallback workload.  Missing, incomplete, or
tampered predecessor/successor evidence produces a terminal blocked record.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from tao_automl.recommendation_audit import (
    canonical_audit_sha256,
    validate_recommendation_audit,
)
from tao_automl.selection import (
    AccuracyConstraint,
    SelectionConfig,
    analyze_archive,
    canonical_spec_fingerprint,
)


MODES = ("accuracy", "latency", "multi_objective")
SUCCESS_CANDIDATE_STATES = frozenset({"success", "done"})
FAILED_HISTORY_STATES = frozenset({"failure", "error"})
RECOVERED_CANCELLATION_RECORD_KEYS = frozenset(
    {
        "agent_intervention_flags",
        "candidate_id",
        "rec_id",
        "recommendation_audit",
        "specs",
        "status",
    }
)
EXPECTED_MODEL_BASED_METHOD = {
    "accuracy": "accuracy_expected_improvement",
    "latency": "constrained_latency_expected_improvement",
    "multi_objective": "parego_expected_improvement",
}
EXPECTED_OPTIMIZATION_DIRECTION = {
    "accuracy": {"accuracy": "maximize"},
    "latency": {
        "accuracy": "constraint_maximize",
        "latency": "minimize",
    },
    "multi_objective": {
        "accuracy": "maximize",
        "latency": "minimize",
    },
}
SUCCESSOR_EXECUTION_KIND = "direct_full_qualification"
REQUIRED_SUCCESSOR_ENVIRONMENT = frozenset(
    {
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPATH",
    }
)
SECRET_ENVIRONMENT_NAMES = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
        "NGC_API_KEY",
        "NGC_KEY",
    }
)
ROUTING_ENVIRONMENT_NAMES = frozenset(
    {"SLURM_HOSTNAME", "SLURM_USER", "SSH_KEY_PATH"}
)
SELECTION_TIME_ISOLATION = {
    "selector_invoked_on_matched_measurements": False,
    "selection_time_objectives_replaced": False,
    "measurements_feed_selection": True,
    "measurements_feed_reselection": False,
    "algorithm_selected_candidate_overridden": False,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AutomaticSuccessorError(RuntimeError):
    """The automatic handoff cannot safely continue."""


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AutomaticSuccessorError(
            f"required artifact is unavailable: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise AutomaticSuccessorError(
            f"required artifact is invalid JSON: {path}: {exc}"
        ) from exc


def routing_identity_from_environment_file(path: Path) -> dict[str, Any]:
    """Resolve only non-secret SLURM/SSH routing identity from an env file."""
    if path.is_symlink() or not path.is_file():
        raise AutomaticSuccessorError(
            "successor environment file is unavailable or is a symlink"
        )
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if name not in ROUTING_ENVIRONMENT_NAMES:
            continue
        if name in values:
            raise AutomaticSuccessorError(
                f"successor environment file duplicates {name} "
                f"at line {line_number}"
            )
        value = raw_value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        if not value or any(character.isspace() for character in value):
            raise AutomaticSuccessorError(
                f"successor routing value {name} is empty or contains whitespace"
            )
        values[name] = value
    missing = sorted(
        name
        for name in ("SLURM_HOSTNAME", "SLURM_USER")
        if not values.get(name)
    )
    if missing:
        raise AutomaticSuccessorError(
            "successor environment lacks required routing keys: "
            + ", ".join(missing)
        )
    key_path_value = values.get("SSH_KEY_PATH")
    key_path: Path | None = None
    key_sha256: str | None = None
    if key_path_value is not None:
        key_path = Path(key_path_value)
        if (
            not key_path.is_absolute()
            or key_path.is_symlink()
            or not key_path.is_file()
        ):
            raise AutomaticSuccessorError(
                "successor SSH key path must be an absolute regular "
                "non-symlink file"
            )
        key_sha256 = sha256_file(key_path)
    return {
        "slurm_hostname": values["SLURM_HOSTNAME"],
        "slurm_user": values["SLURM_USER"],
        "ssh_key_path": str(key_path) if key_path is not None else None,
        "ssh_key_sha256": key_sha256,
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _require_exact_false_flags(
    value: Any,
    expected_names: set[str],
    *,
    path: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != expected_names:
        raise AutomaticSuccessorError(f"{path} flag schema changed")
    if any(value[name] is not False for name in expected_names):
        raise AutomaticSuccessorError(f"{path} contains an intervention")


def _finite_metric(value: Any, *, path: str) -> float:
    if isinstance(value, bool):
        raise AutomaticSuccessorError(f"{path} must be a finite number")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AutomaticSuccessorError(
            f"{path} must be a finite number"
        ) from exc
    if not math.isfinite(normalized):
        raise AutomaticSuccessorError(f"{path} must be finite")
    return normalized


def _single_command_value(command: list[str], flag: str) -> str:
    positions = [index for index, value in enumerate(command) if value == flag]
    if len(positions) != 1:
        raise AutomaticSuccessorError(
            f"successor command must contain {flag} exactly once"
        )
    position = positions[0]
    if position + 1 >= len(command) or command[position + 1].startswith("--"):
        raise AutomaticSuccessorError(
            f"successor command has no value for {flag}"
        )
    return command[position + 1]


def _load_canonical_successor_manifest(
    path: Path,
) -> dict[str, Any]:
    manifest = load_json(path)
    if not isinstance(manifest, Mapping):
        raise AutomaticSuccessorError(
            "successor campaign manifest must be a mapping"
        )
    manifest = copy.deepcopy(dict(manifest))
    expected = manifest.pop("manifest_sha256", None)
    if not isinstance(expected, str) or expected != canonical_sha256(manifest):
        raise AutomaticSuccessorError(
            "successor campaign manifest canonical hash changed"
        )
    manifest["manifest_sha256"] = expected
    return manifest


def validate_successor_descriptor(path: Path) -> dict[str, Any]:
    """Validate and content-address one immutable successor command."""
    descriptor = load_json(path)
    if not isinstance(descriptor, Mapping):
        raise AutomaticSuccessorError("successor descriptor must be a mapping")
    descriptor = dict(descriptor)
    expected_digest = descriptor.pop("descriptor_sha256", None)
    if descriptor.get("schema_version") != 1:
        raise AutomaticSuccessorError(
            "successor descriptor schema version is unsupported"
        )
    actual_digest = canonical_sha256(descriptor)
    if expected_digest != actual_digest:
        raise AutomaticSuccessorError(
            "successor descriptor integrity verification failed"
        )
    descriptor["descriptor_sha256"] = expected_digest

    predecessor = descriptor.get("predecessor")
    successor = descriptor.get("successor")
    if not isinstance(predecessor, Mapping) or not isinstance(
        successor, Mapping
    ):
        raise AutomaticSuccessorError(
            "successor descriptor requires predecessor and successor mappings"
        )
    if tuple(predecessor.get("required_modes", ())) != MODES:
        raise AutomaticSuccessorError(
            f"predecessor required_modes must equal {list(MODES)}"
        )
    manifest_path = Path(str(predecessor.get("manifest_path", "")))
    predecessor_runtime_root = Path(
        str(predecessor.get("runtime_root", ""))
    )
    manifest_validator_path = Path(
        str(predecessor.get("manifest_validator_path", ""))
    )
    if (
        not manifest_path.is_absolute()
        or not predecessor_runtime_root.is_absolute()
        or not manifest_validator_path.is_absolute()
        or _SHA256_RE.fullmatch(
            str(predecessor.get("manifest_file_sha256", ""))
        )
        is None
        or _SHA256_RE.fullmatch(str(predecessor.get("manifest_sha256", "")))
        is None
        or _SHA256_RE.fullmatch(
            str(predecessor.get("manifest_validator_sha256", ""))
        )
        is None
    ):
        raise AutomaticSuccessorError(
            "predecessor paths and manifest identities are incomplete"
        )
    controller = predecessor.get("controller_process")
    if not isinstance(controller, Mapping):
        raise AutomaticSuccessorError(
            "predecessor controller_process is required"
        )
    if (
        isinstance(controller.get("pid"), bool)
        or not isinstance(controller.get("pid"), int)
        or controller["pid"] <= 1
        or isinstance(controller.get("start_time_ticks"), bool)
        or not isinstance(controller.get("start_time_ticks"), int)
        or controller["start_time_ticks"] <= 0
        or _SHA256_RE.fullmatch(
            str(controller.get("cmdline_sha256", ""))
        )
        is None
        or not isinstance(controller.get("boot_id"), str)
        or not controller["boot_id"]
    ):
        raise AutomaticSuccessorError(
            "predecessor controller process identity is incomplete"
        )

    command = successor.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise AutomaticSuccessorError(
            "successor command must be a non-empty argument array"
        )
    working_directory = Path(str(successor.get("working_directory", "")))
    if not working_directory.is_absolute():
        raise AutomaticSuccessorError(
            "successor working_directory must be absolute"
        )
    required_files = successor.get("required_files")
    if not isinstance(required_files, list) or not required_files:
        raise AutomaticSuccessorError(
            "successor required_files must be a non-empty list"
        )
    for index, record in enumerate(required_files):
        if not isinstance(record, Mapping):
            raise AutomaticSuccessorError(
                f"successor required_files[{index}] must be a mapping"
            )
        artifact = Path(str(record.get("path", "")))
        expected = record.get("sha256")
        if not artifact.is_absolute() or not isinstance(expected, str):
            raise AutomaticSuccessorError(
                f"successor required_files[{index}] is incomplete"
            )
    if successor.get("execution_kind") != SUCCESSOR_EXECUTION_KIND:
        raise AutomaticSuccessorError(
            "successor execution_kind must be "
            f"{SUCCESSOR_EXECUTION_KIND!r}"
        )
    if successor.get("cpu_runs") != 0 or successor.get("smoke_runs") != 0:
        raise AutomaticSuccessorError(
            "successor must seal zero CPU and zero smoke runs"
        )
    if successor.get("model") != "deformable_detr":
        raise AutomaticSuccessorError(
            "the immediate successor model must be deformable_detr"
        )
    if "--acknowledge-direct-full-dataset" not in command:
        raise AutomaticSuccessorError(
            "successor command lacks direct-full-dataset acknowledgement"
        )
    executable = Path(command[0])
    if not executable.is_absolute():
        raise AutomaticSuccessorError(
            "successor executable must be an absolute path"
        )
    required_by_path: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(required_files):
        record_path = str(Path(record["path"]))
        if record_path in required_by_path:
            raise AutomaticSuccessorError(
                f"successor required_files[{index}] duplicates {record_path}"
            )
        if _SHA256_RE.fullmatch(str(record.get("sha256", ""))) is None:
            raise AutomaticSuccessorError(
                f"successor required_files[{index}] has an invalid SHA-256"
            )
        required_by_path[record_path] = record
    if str(executable) not in required_by_path:
        raise AutomaticSuccessorError(
            "successor executable must be content-addressed in required_files"
        )
    successor_manifest = Path(str(successor.get("manifest_path", "")))
    successor_runtime_root = Path(
        str(successor.get("runtime_root", ""))
    )
    launcher_path = Path(str(successor.get("launcher_path", "")))
    manifest_generator_path = Path(
        str(successor.get("manifest_generator_path", ""))
    )
    if (
        not successor_manifest.is_absolute()
        or not successor_runtime_root.is_absolute()
        or not launcher_path.is_absolute()
        or not manifest_generator_path.is_absolute()
        or _SHA256_RE.fullmatch(
            str(successor.get("manifest_file_sha256", ""))
        )
        is None
        or str(successor_manifest) not in required_by_path
        or str(launcher_path) not in required_by_path
        or str(manifest_generator_path) not in required_by_path
    ):
        raise AutomaticSuccessorError(
            "successor launcher, generator, manifest, and runtime must be "
            "absolute and content-addressed"
        )
    if (
        required_by_path[str(successor_manifest)]["sha256"]
        != successor["manifest_file_sha256"]
    ):
        raise AutomaticSuccessorError(
            "successor manifest hashes disagree"
        )
    if len(command) < 2 or command[1] != str(launcher_path):
        raise AutomaticSuccessorError(
            "successor command does not execute its content-addressed launcher"
        )
    if command.count("--launch") != 1:
        raise AutomaticSuccessorError(
            "successor command must contain --launch exactly once"
        )
    if "--resume" in command:
        raise AutomaticSuccessorError(
            "automatic successor must start a fresh sealed runtime"
        )
    if _single_command_value(command, "--manifest") != str(
        successor_manifest
    ):
        raise AutomaticSuccessorError(
            "successor command does not consume its sealed manifest"
        )
    if _single_command_value(command, "--runtime-root") != str(
        successor_runtime_root
    ):
        raise AutomaticSuccessorError(
            "successor command runtime differs from its sealed runtime root"
        )
    environment_file = Path(
        str(successor.get("environment_file", ""))
    )
    if (
        not environment_file.is_absolute()
        or _single_command_value(command, "--env-file")
        != str(environment_file)
    ):
        raise AutomaticSuccessorError(
            "successor command must use its explicit absolute environment file"
        )
    routing = successor.get("routing")
    if (
        not isinstance(routing, Mapping)
        or set(routing)
        != {
            "slurm_hostname",
            "slurm_user",
            "ssh_key_path",
            "ssh_key_sha256",
        }
        or not isinstance(routing.get("slurm_hostname"), str)
        or not routing["slurm_hostname"]
        or not isinstance(routing.get("slurm_user"), str)
        or not routing["slurm_user"]
        or (
            routing.get("ssh_key_path") is None
            and routing.get("ssh_key_sha256") is not None
        )
        or (
            routing.get("ssh_key_path") is not None
            and (
                not isinstance(routing["ssh_key_path"], str)
                or not Path(routing["ssh_key_path"]).is_absolute()
                or _SHA256_RE.fullmatch(
                    str(routing.get("ssh_key_sha256", ""))
                )
                is None
            )
        )
    ):
        raise AutomaticSuccessorError(
            "successor routing identity is incomplete"
        )
    environment = successor.get("environment")
    if (
        not isinstance(environment, Mapping)
        or not environment
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            for key, value in environment.items()
        )
    ):
        raise AutomaticSuccessorError(
            "successor environment must be a non-empty explicit mapping"
        )
    if not REQUIRED_SUCCESSOR_ENVIRONMENT.issubset(environment):
        missing = sorted(REQUIRED_SUCCESSOR_ENVIRONMENT - set(environment))
        raise AutomaticSuccessorError(
            f"successor environment is missing required keys: {missing}"
        )
    forbidden = sorted(SECRET_ENVIRONMENT_NAMES & set(environment))
    if forbidden:
        raise AutomaticSuccessorError(
            "successor descriptor must not embed credential variables: "
            f"{forbidden}"
        )
    for key in ("HOME",):
        if not Path(environment[key]).is_absolute():
            raise AutomaticSuccessorError(
                f"successor environment {key} must be absolute"
            )
    for key in ("PATH", "PYTHONPATH"):
        entries = environment[key].split(os.pathsep)
        if not entries or any(not Path(entry).is_absolute() for entry in entries):
            raise AutomaticSuccessorError(
                f"successor environment {key} entries must be absolute"
            )
    completion_artifact = Path(
        str(successor.get("completion_artifact", ""))
    )
    try:
        completion_artifact.relative_to(successor_runtime_root)
    except ValueError as exc:
        raise AutomaticSuccessorError(
            "successor completion artifact must be inside its runtime root"
        ) from exc
    if (
        not completion_artifact.is_absolute()
        or completion_artifact
        != successor_runtime_root / "completion.json"
    ):
        raise AutomaticSuccessorError(
            "successor completion_artifact must be the sealed runtime "
            "completion.json"
        )
    if _single_command_value(
        command,
        "--completion-artifact",
    ) != str(completion_artifact):
        raise AutomaticSuccessorError(
            "successor command does not bind its terminal completion artifact"
        )

    manifest = _load_canonical_successor_manifest(successor_manifest)
    execution = manifest.get("execution")
    runtime = manifest.get("runtime")
    ptms = manifest.get("ptms")
    integrity = manifest.get("integrity")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("campaign_id") != successor.get("campaign_id")
        or manifest.get("model") != "deformable_detr"
        or not isinstance(execution, Mapping)
        or execution.get("kind") != SUCCESSOR_EXECUTION_KIND
        or execution.get("cpu_runs") != 0
        or execution.get("smoke_runs") != 0
        or execution.get("ministep_runs") != 0
        or execution.get("local_model_runs") != 0
        or execution.get("full_training") is not True
        or execution.get("standalone_evaluation") is not True
        or not isinstance(runtime, Mapping)
        or runtime.get("nodes") != 1
        or runtime.get("tasks_per_node") != 1
        or runtime.get("gpus_per_node") != 8
        or not str(runtime.get("sqsh_path", "")).endswith(".sqsh")
        or _SHA256_RE.fullmatch(str(runtime.get("sqsh_sha256", "")))
        is None
        or not isinstance(ptms, list)
        or len(ptms) != 2
        or len({item.get("id") for item in ptms if isinstance(item, Mapping)})
        != 2
        or {
            item.get("workflow_id")
            for item in ptms
            if isinstance(item, Mapping)
        }
        != {"gcvit_tiny", "resnet50"}
        or not isinstance(integrity, Mapping)
    ):
        raise AutomaticSuccessorError(
            "successor manifest violates the direct full Deformable DETR "
            "execution contract"
        )
    if (
        integrity.get("launcher_sha256")
        != required_by_path[str(launcher_path)]["sha256"]
        or integrity.get("manifest_generator_sha256")
        != required_by_path[str(manifest_generator_path)]["sha256"]
    ):
        raise AutomaticSuccessorError(
            "successor manifest source identities disagree with the descriptor"
        )
    return descriptor


def verify_successor_inputs(descriptor: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Verify every successor source before any trigger is recorded."""
    successor = descriptor["successor"]
    working_directory = Path(successor["working_directory"])
    if not working_directory.is_dir():
        raise AutomaticSuccessorError(
            f"successor working directory is unavailable: {working_directory}"
        )
    evidence = []
    for record in successor["required_files"]:
        artifact = Path(record["path"])
        if not artifact.is_file():
            raise AutomaticSuccessorError(
                f"successor input is unavailable: {artifact}"
            )
        observed = sha256_file(artifact)
        if observed != record["sha256"]:
            raise AutomaticSuccessorError(
                f"successor input identity changed: {artifact}"
            )
        evidence.append(
            {
                "path": str(artifact),
                "sha256": observed,
                "size_bytes": artifact.stat().st_size,
            }
        )
    executable = successor["command"][0]
    if "/" in executable:
        executable_path = Path(executable)
        if not executable_path.is_file():
            raise AutomaticSuccessorError(
                f"successor executable is unavailable: {executable_path}"
            )
    environment_file = Path(successor["environment_file"])
    observed_routing = routing_identity_from_environment_file(environment_file)
    if observed_routing != successor["routing"]:
        raise AutomaticSuccessorError(
            "successor SLURM/SSH routing identity changed after sealing"
        )
    if Path(successor["completion_artifact"]).exists():
        raise AutomaticSuccessorError(
            "fresh automatic successor refuses a pre-existing completion "
            "artifact"
        )
    return evidence


def load_predecessor_manifest(
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    predecessor = descriptor["predecessor"]
    path = Path(predecessor["manifest_path"])
    if not path.is_file():
        raise AutomaticSuccessorError(
            f"predecessor manifest is unavailable: {path}"
        )
    if sha256_file(path) != predecessor["manifest_file_sha256"]:
        raise AutomaticSuccessorError(
            "predecessor manifest file identity changed"
        )
    validator_path = Path(predecessor["manifest_validator_path"])
    if (
        not validator_path.is_file()
        or sha256_file(validator_path)
        != predecessor["manifest_validator_sha256"]
    ):
        raise AutomaticSuccessorError(
            "predecessor manifest validator identity changed"
        )
    module_name = (
        "_sealed_dino_manifest_generator_"
        + predecessor["manifest_validator_sha256"][:16]
    )
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            validator_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError("unable to construct sealed validator module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manifest = module.load_manifest(path)
    except Exception as exc:
        raise AutomaticSuccessorError(
            f"predecessor manifest validation failed: {type(exc).__name__}"
        ) from exc
    if manifest.get("manifest_sha256") != predecessor["manifest_sha256"]:
        raise AutomaticSuccessorError(
            "predecessor canonical manifest identity changed"
        )
    if manifest.get("campaign_id") != predecessor["campaign_id"]:
        raise AutomaticSuccessorError(
            "predecessor campaign identity changed"
        )
    if manifest["search"]["candidate_budget_per_mode"] <= manifest["search"][
        "calibration_points"
    ]:
        raise AutomaticSuccessorError(
            "predecessor budget does not reach model-based acquisition"
        )
    return manifest


def _process_identity(pid: int) -> dict[str, Any] | None:
    proc = Path("/proc") / str(pid)
    try:
        raw_stat = (proc / "stat").read_text(encoding="utf-8")
        closing = raw_stat.rfind(")")
        fields = raw_stat[closing + 2 :].split()
        start_time_ticks = int(fields[19])
        cmdline = (proc / "cmdline").read_bytes()
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None
    return {
        "pid": pid,
        "start_time_ticks": start_time_ticks,
        "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
        "boot_id": boot_id,
    }


def verify_controller_alive(descriptor: Mapping[str, Any]) -> None:
    expected = dict(descriptor["predecessor"]["controller_process"])
    observed = _process_identity(expected["pid"])
    if observed != expected:
        raise AutomaticSuccessorError(
            "DINO controller exited or its sealed process identity changed "
            "before producing terminal evidence"
        )


def validate_successor_completion(
    descriptor: Mapping[str, Any],
    *,
    not_before_ns: int | None = None,
) -> dict[str, Any]:
    """Validate the sealed terminal record emitted by the successor."""
    successor = descriptor["successor"]
    path = Path(successor["completion_artifact"])
    if path.is_symlink() or not path.is_file():
        raise AutomaticSuccessorError(
            "successor returned without a regular completion artifact"
        )
    if not_before_ns is not None and path.stat().st_mtime_ns < not_before_ns:
        raise AutomaticSuccessorError(
            "successor completion artifact predates this launch"
        )
    value = load_json(path)
    if not isinstance(value, Mapping):
        raise AutomaticSuccessorError(
            "successor completion artifact must be a mapping"
        )
    completion = copy.deepcopy(dict(value))
    expected = completion.pop("completion_sha256", None)
    if expected != canonical_sha256(completion):
        raise AutomaticSuccessorError(
            "successor completion artifact integrity verification failed"
        )
    outcomes = completion.get("outcomes")
    workflows = completion.get("workflows")
    manifest = _load_canonical_successor_manifest(
        Path(successor["manifest_path"])
    )
    expected_workflows = {
        str(item["workflow_id"]): str(item["id"])
        for item in manifest["ptms"]
    }
    if (
        completion.get("schema_version") != 1
        or completion.get("campaign_id") != successor["campaign_id"]
        or completion.get("model") != "deformable_detr"
        or completion.get("manifest_sha256")
        != manifest["manifest_sha256"]
        or completion.get("terminal") is not True
        or completion.get("status")
        not in {"success", "terminal_with_failures"}
        or completion.get("logical_workflows_submitted") != 2
        or completion.get("workflows_started_in_parallel") is not True
        or completion.get("cpu_runs") != 0
        or completion.get("smoke_runs") != 0
        or completion.get("ministep_runs") != 0
        or completion.get("local_model_runs") != 0
        or completion.get("failures_preserved") is not True
        or completion.get("replacement_workflows_submitted") is not False
        or not isinstance(outcomes, Mapping)
        or set(outcomes) != set(expected_workflows)
        or any(
            value not in {"success", "terminal_failure"}
            for value in outcomes.values()
        )
        or not isinstance(workflows, list)
        or len(workflows) != 2
        or any(
            not isinstance(record, Mapping)
            or record.get("terminal") is not True
            or record.get("status")
            not in {"success", "terminal_failure"}
            for record in workflows
        )
    ):
        raise AutomaticSuccessorError(
            "successor completion artifact violates its terminal contract"
        )
    workflow_by_id: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(workflows):
        workflow_id = str(record.get("workflow_id", ""))
        if not workflow_id or workflow_id in workflow_by_id:
            raise AutomaticSuccessorError(
                "successor completion has duplicate or missing workflow IDs"
            )
        workflow_by_id[workflow_id] = record
        status = record["status"]
        exit_code = record.get("process_exit_code")
        if (
            workflow_id not in expected_workflows
            or record.get("ptm_id") != expected_workflows[workflow_id]
            or outcomes.get(workflow_id) != status
            or isinstance(exit_code, bool)
            or not isinstance(exit_code, int)
            or (status == "success" and exit_code != 0)
            or (status == "terminal_failure" and exit_code == 0)
        ):
            raise AutomaticSuccessorError(
                f"successor workflow {index} is inconsistent with its "
                "identity, outcome, or process exit"
            )
        if status == "success":
            metrics = record.get("metrics")
            if (
                record.get("failure_preserved") is not False
                or not isinstance(metrics, Mapping)
                or set(metrics) != {"mAP", "mAP50"}
            ):
                raise AutomaticSuccessorError(
                    f"successful successor workflow {workflow_id} lacks "
                    "its exact metric evidence"
                )
            for metric_name, metric_value in metrics.items():
                normalized = _finite_metric(
                    metric_value,
                    path=(
                        "successor_completion.workflows."
                        f"{workflow_id}.metrics.{metric_name}"
                    ),
                )
                if not 0.0 <= normalized <= 1.0:
                    raise AutomaticSuccessorError(
                        f"successor workflow {workflow_id} metric "
                        f"{metric_name} is outside [0, 1]"
                    )
        elif record.get("failure_preserved") is not True:
            raise AutomaticSuccessorError(
                f"failed successor workflow {workflow_id} was not preserved"
            )
    if set(workflow_by_id) != set(expected_workflows):
        raise AutomaticSuccessorError(
            "successor completion workflow identities are incomplete"
        )
    successful = sum(value == "success" for value in outcomes.values())
    failed = len(expected_workflows) - successful
    expected_status = "success" if failed == 0 else "terminal_with_failures"
    if (
        completion.get("successful_workflows") != successful
        or completion.get("failed_workflows") != failed
        or completion.get("status") != expected_status
    ):
        raise AutomaticSuccessorError(
            "successor completion counts or aggregate status are inconsistent"
        )
    completion["completion_sha256"] = expected
    completion["completion_file_sha256"] = sha256_file(path)
    return completion


def _validate_success_candidate(
    record: Mapping[str, Any],
    *,
    candidate_path: str,
    manifest: Mapping[str, Any],
    fingerprint: str,
) -> None:
    objectives = record.get("objective_values")
    if not isinstance(objectives, Mapping):
        raise AutomaticSuccessorError(
            f"{candidate_path}.objective_values is missing"
        )
    accuracy = _finite_metric(
        objectives.get("mAP50"),
        path=f"{candidate_path}.mAP50",
    )
    latency_value = _finite_metric(
        objectives.get("latency_ms"),
        path=f"{candidate_path}.latency_ms",
    )
    if not 0.0 <= accuracy <= 1.0:
        raise AutomaticSuccessorError(
            f"{candidate_path}.mAP50 must be in [0, 1]"
        )
    if latency_value <= 0.0:
        raise AutomaticSuccessorError(
            f"{candidate_path}.latency_ms must be > 0"
        )
    latency = record.get("selection_time_latency")
    try:
        aggregate_evidence = latency["aggregate_evidence"]
        aggregate = aggregate_evidence["aggregate"]
        statistics = aggregate["statistics"]
    except (KeyError, TypeError) as exc:
        raise AutomaticSuccessorError(
            f"{candidate_path} lacks latency quality evidence"
        ) from exc
    outer_payload = copy.deepcopy(dict(aggregate_evidence))
    outer_digest = outer_payload.pop("evidence_sha256", None)
    if outer_digest != canonical_sha256(outer_payload):
        raise AutomaticSuccessorError(
            f"{candidate_path} latency outer evidence hash changed"
        )
    aggregate_payload = copy.deepcopy(dict(aggregate))
    aggregate_digest = aggregate_payload.pop("aggregate_sha256", None)
    if aggregate_digest != canonical_sha256(aggregate_payload):
        raise AutomaticSuccessorError(
            f"{candidate_path} latency aggregate hash changed"
        )
    protocol = manifest["latency_protocol"]
    expected_contract = {
        "schema_version": 1,
        "warmup_iterations": protocol["warmup_iterations"],
        "timed_iterations": protocol["timed_iterations"],
        "repeated_rounds": protocol["repeated_rounds"],
        "tail_percentile": protocol["tail_percentile"],
        "bootstrap_resamples": protocol["bootstrap_resamples"],
        "bootstrap_confidence_level": protocol[
            "bootstrap_confidence_level"
        ],
        "bootstrap_seed": protocol["bootstrap_seed"],
        "batch_size_per_replica": protocol["batch_size_per_replica"],
        "precision": protocol["precision"],
        "timed_scope": protocol["timed_scope"],
        "input_sha256": protocol["input_sha256"],
        "runtime_sha256": protocol["runtime_sha256"],
        "expected_replicas": protocol["expected_replicas"],
        "measurement_role": protocol["measurement_role"],
        "synchronization": protocol["synchronization"],
        "validity_thresholds": protocol["validity_thresholds"],
    }
    if (
        aggregate.get("contract") != expected_contract
        or aggregate.get("contract_sha256")
        != canonical_sha256(expected_contract)
        or aggregate.get("candidate_fingerprint") != fingerprint
        or aggregate.get("selection_isolation") != SELECTION_TIME_ISOLATION
    ):
        raise AutomaticSuccessorError(
            f"{candidate_path} latency contract or isolation changed"
        )
    if (
        statistics.get("is_valid") is not True
        or statistics.get("raw_sample_count_total") != 4000
        or statistics.get("invalid_reasons") != []
    ):
        raise AutomaticSuccessorError(
            f"{candidate_path} failed the frozen latency quality gate"
        )
    median = _finite_metric(
        statistics.get("median_ms"),
        path=f"{candidate_path}.latency.median_ms",
    )
    p95 = _finite_metric(
        statistics.get("p95_ms"),
        path=f"{candidate_path}.latency.p95_ms",
    )
    ci = statistics.get("bootstrap_median_ci_ms")
    if not isinstance(ci, list) or len(ci) != 2:
        raise AutomaticSuccessorError(
            f"{candidate_path} latency confidence interval is missing"
        )
    ci_low = _finite_metric(
        ci[0],
        path=f"{candidate_path}.latency.ci_low",
    )
    ci_high = _finite_metric(
        ci[1],
        path=f"{candidate_path}.latency.ci_high",
    )
    if (
        median <= 0.0
        or p95 < median
        or not ci_low <= median <= ci_high
        or latency_value != median
        or _finite_metric(
            objectives.get("latency_p95_ms"),
            path=f"{candidate_path}.latency_p95_ms",
        )
        != p95
        or _finite_metric(
            objectives.get("latency_ci95_low_ms"),
            path=f"{candidate_path}.latency_ci95_low_ms",
        )
        != ci_low
        or _finite_metric(
            objectives.get("latency_ci95_high_ms"),
            path=f"{candidate_path}.latency_ci95_high_ms",
        )
        != ci_high
    ):
        raise AutomaticSuccessorError(
            f"{candidate_path} latency objectives differ from aggregate evidence"
        )
    rank_runtime = aggregate_evidence.get("rank_runtime_evidence")
    if not isinstance(rank_runtime, list) or len(rank_runtime) != 8:
        raise AutomaticSuccessorError(
            f"{candidate_path} lacks eight-replica runtime provenance"
        )
    runtime_hashes = set()
    ranks = set()
    for item in rank_runtime:
        if not isinstance(item, Mapping):
            raise AutomaticSuccessorError(
                f"{candidate_path} runtime provenance is malformed"
            )
        for key, expected in manifest["runtime"]["hardware_contract"].items():
            if item.get(key) != expected:
                raise AutomaticSuccessorError(
                    f"{candidate_path} runtime hardware changed: {key}"
                )
        ranks.add(item.get("local_rank"))
        runtime_contract = {
            key: value
            for key, value in item.items()
            if key not in {"hostname", "local_rank", "nvidia_smi"}
        }
        runtime_hashes.add(canonical_sha256(runtime_contract))
    if ranks != set(range(8)) or runtime_hashes != {
        aggregate.get("hardware_sha256")
    }:
        raise AutomaticSuccessorError(
            f"{candidate_path} runtime replica identity changed"
        )
    if (
        not isinstance(
            aggregate_evidence.get("input_evidence_sha256"),
            str,
        )
        or _SHA256_RE.fullmatch(
            aggregate_evidence["input_evidence_sha256"]
        )
        is None
    ):
        raise AutomaticSuccessorError(
            f"{candidate_path} input evidence identity is missing"
        )
    _require_exact_false_flags(
        record.get("matched_validation_selection_isolation_flags"),
        set(manifest["selection_isolation_flags"]),
        path=(
            f"{candidate_path}."
            "matched_validation_selection_isolation_flags"
        ),
    )


def _selection_config(
    manifest: Mapping[str, Any],
    mode: str,
) -> SelectionConfig:
    search = manifest["search"]
    retention = (
        AccuracyConstraint(
            kind="relative",
            value=search["latency_accuracy_retention"],
            reference="accuracy_winner",
        )
        if mode == "latency"
        else AccuracyConstraint()
    )
    return SelectionConfig(
        mode=mode,
        accuracy_metric="mAP50",
        latency_metric="latency_ms",
        latency_accuracy_retention=retention,
        multi_objective_min_accuracy=None,
        accuracy_tolerance=1.0e-12,
        latency_tolerance=search["latency_practical_tolerance_ms"],
        score_tolerance=1.0e-12,
        augmentation_rho=1.0e-6,
        normalization="pareto_front",
        latency_ci_low_metric="latency_ci95_low_ms",
        latency_ci_high_metric="latency_ci95_high_ms",
    )


def _history_snapshot(history_item: Mapping[str, Any]) -> dict[str, Any]:
    """Return immutable observation fields exposed to later recommendations."""
    return {
        "candidate_id": str(history_item["rec_id"]),
        "candidate_fingerprint": canonical_spec_fingerprint(
            history_item["specs"]
        ),
        "status": history_item["status"],
        "objective_values": history_item.get("objective_values", {}),
        "failure_reason": history_item.get("failure_reason"),
    }


def _visible_history_snapshot(item: Mapping[str, Any]) -> dict[str, Any]:
    """Project an issuance-time observation onto immutable raw evidence."""
    return {
        key: item.get(key)
        for key in (
            "candidate_id",
            "candidate_fingerprint",
            "status",
            "objective_values",
            "failure_reason",
        )
    }


def _validate_candidate_history_alignment(
    record: Mapping[str, Any],
    history_item: Mapping[str, Any],
    *,
    candidate_path: str,
) -> bool:
    """Validate candidate evidence against terminal runner history.

    Returns ``True`` only for a cancellation whose candidate artifact was
    intentionally left at the pre-submission ``recommended`` state by an
    interrupted controller.  The terminal runner history remains the
    authoritative preserved failure record in that narrow case.
    """
    state = str(record.get("status", "")).lower()
    history_status = str(history_item.get("status", "")).lower()
    if history_item.get("specs") != record.get("specs"):
        raise AutomaticSuccessorError(
            f"{candidate_path} specifications differ from runner history"
        )
    if history_status in SUCCESS_CANDIDATE_STATES:
        if (
            state not in SUCCESS_CANDIDATE_STATES
            or history_item.get("job_id") != record.get("train_job_id")
            or history_item.get("objective_values", {})
            != record.get("objective_values", {})
        ):
            raise AutomaticSuccessorError(
                f"{candidate_path} differs from runner history "
                "(successful candidate)"
            )
        return False
    if history_status not in FAILED_HISTORY_STATES:
        raise AutomaticSuccessorError(
            f"{candidate_path} history status is not terminal"
        )
    failure_reason = history_item.get("failure_reason")
    job_id = history_item.get("job_id")
    if (
        not isinstance(failure_reason, str)
        or not failure_reason
        or not isinstance(job_id, str)
        or not job_id
        or history_item.get("metric") != 0
        or history_item.get("objective_score") != 0
        or history_item.get("objective_values") != {"mAP50": 0}
    ):
        raise AutomaticSuccessorError(
            f"{candidate_path} runner failure sentinel is malformed"
        )
    if state == "recommended":
        if (
            failure_reason != "job_canceled"
            or set(record) != RECOVERED_CANCELLATION_RECORD_KEYS
        ):
            raise AutomaticSuccessorError(
                f"{candidate_path} is not a narrowly recoverable cancellation"
            )
        return True
    if state not in {"terminal_failure", "failure", "error"}:
        raise AutomaticSuccessorError(
            f"{candidate_path} is not terminal: {state!r}"
        )
    if (
        record.get("train_job_id") != job_id
        or record.get("failure_reason") != failure_reason
        or record.get("automl_status") not in (None, history_status)
        or record.get("reported_metric") is not None
        or record.get("objective_values") not in (None, {"mAP50": 0})
    ):
        raise AutomaticSuccessorError(
            f"{candidate_path} differs from runner history "
            "(failed candidate)"
        )
    return False


def _selection_winner_id(
    result: Mapping[str, Any],
    mode: str,
    *,
    manifest: Mapping[str, Any],
    archive: list[dict[str, Any]],
) -> str:
    persisted = result.get("selection_analysis")
    recomputed = analyze_archive(
        archive,
        _selection_config(manifest, mode),
    ).to_dict()
    if canonical_sha256(persisted) != canonical_sha256(recomputed):
        raise AutomaticSuccessorError(
            f"{mode} persisted selector evidence differs from production replay"
        )
    try:
        selection = recomputed["selections"][mode]
        winner_id = str(selection["winner_id"])
        selected_status = selection["status"]
        configured_mode = recomputed["algorithm"]["configuration"]["mode"]
    except (KeyError, TypeError) as exc:
        raise AutomaticSuccessorError(
            f"{mode} result lacks final selection evidence"
        ) from exc
    if selected_status != "selected" or configured_mode != mode:
        raise AutomaticSuccessorError(
            f"{mode} result did not complete its configured selection"
        )
    if mode == "multi_objective":
        configuration = recomputed["algorithm"]["configuration"]
        if configuration.get("multi_objective_min_accuracy") is not None:
            raise AutomaticSuccessorError(
                "multi-objective mode unexpectedly inherited an accuracy floor"
            )
        candidate_audits = {
            item["candidate_id"]: item
            for item in recomputed["candidates"]
        }
        winner_audit = candidate_audits[winner_id]
        if (
            winner_audit["multi_objective_pareto_rank"] != 0
            or winner_audit["multi_objective_dominated_by"]
        ):
            raise AutomaticSuccessorError(
                "multi-objective winner is not Pareto rank zero"
            )
    return winner_id


def validate_completed_dino(
    descriptor: Mapping[str, Any],
    runtime_root: Path,
) -> dict[str, Any]:
    """Validate the terminal DINO evidence that releases the successor."""
    predecessor = descriptor["predecessor"]
    manifest = load_predecessor_manifest(descriptor)
    expected_manifest_sha = manifest["manifest_sha256"]
    budget = manifest["search"]["candidate_budget_per_mode"]
    calibration = manifest["search"]["calibration_points"]
    expected_flags = set(manifest["agent_intervention_flags"])
    if manifest["search"]["latency_accuracy_retention"] != 0.90:
        raise AutomaticSuccessorError(
            "DINO latency mode is not sealed to 90% retained accuracy"
        )

    process_status_path = runtime_root / "mode_process_status.json"
    process_status = load_json(process_status_path)
    expected_status = {mode: 0 for mode in MODES}
    if process_status != expected_status:
        raise AutomaticSuccessorError(
            f"DINO mode process status did not pass: {process_status!r}"
        )

    modes: dict[str, Any] = {}
    for mode in MODES:
        mode_root = runtime_root / mode
        result_path = mode_root / "result.json"
        evidence_path = mode_root / "candidate_evidence.json"
        wrapper = load_json(result_path)
        evidence_wrapper = load_json(evidence_path)
        if (
            wrapper.get("schema_version") != 1
            or wrapper.get("manifest_sha256") != expected_manifest_sha
            or wrapper.get("mode") != mode
            or wrapper.get("status") != "success"
        ):
            raise AutomaticSuccessorError(
                f"{mode} result wrapper does not match the sealed campaign"
            )
        if (
            evidence_wrapper.get("schema_version") != 1
            or evidence_wrapper.get("manifest_sha256") != expected_manifest_sha
            or evidence_wrapper.get("mode") != mode
        ):
            raise AutomaticSuccessorError(
                f"{mode} candidate evidence does not match the sealed campaign"
            )
        result = wrapper.get("result")
        candidates = evidence_wrapper.get("candidates")
        if not isinstance(result, Mapping) or not isinstance(candidates, Mapping):
            raise AutomaticSuccessorError(
                f"{mode} result or candidate evidence is malformed"
            )
        progress = result.get("progress")
        history = result.get("history")
        if (
            not isinstance(progress, Mapping)
            or progress.get("completed") != budget
            or progress.get("total") != budget
            or not isinstance(history, list)
            or len(history) != budget
            or len(candidates) != budget
        ):
            raise AutomaticSuccessorError(
                f"{mode} did not complete the frozen {budget}-candidate budget"
            )

        history_ids = [str(item.get("rec_id")) for item in history]
        expected_ids = {str(index) for index in range(budget)}
        if len(set(history_ids)) != budget or set(history_ids) != expected_ids:
            raise AutomaticSuccessorError(
                f"{mode} recommendation history is incomplete or duplicated"
            )
        candidate_ids = {
            str(record.get("rec_id")) for record in candidates.values()
        }
        if candidate_ids != expected_ids:
            raise AutomaticSuccessorError(
                f"{mode} candidate evidence is not aligned with history"
            )

        success_ids: set[str] = set()
        recovered_cancellation_ids: list[str] = []
        adaptive_ids: list[str] = []
        archive: list[dict[str, Any]] = []
        history_by_id = {
            str(item["rec_id"]): item
            for item in history
        }
        for candidate_id, record in candidates.items():
            candidate_path = f"{mode}.candidates.{candidate_id}"
            if not isinstance(record, Mapping):
                raise AutomaticSuccessorError(
                    f"{candidate_path} must be a mapping"
                )
            _require_exact_false_flags(
                record.get("agent_intervention_flags"),
                expected_flags,
                path=f"{candidate_path}.agent_intervention_flags",
            )
            audit = record.get("recommendation_audit")
            try:
                validate_recommendation_audit(audit)
            except (TypeError, ValueError) as exc:
                raise AutomaticSuccessorError(
                    f"{candidate_path} recommendation audit failed: {exc}"
                ) from exc
            rec_id = str(record["rec_id"])
            if audit.get("candidate_id") != rec_id:
                raise AutomaticSuccessorError(
                    f"{candidate_path} recommendation ID changed"
                )
            specs = record.get("specs")
            fingerprint = canonical_spec_fingerprint(specs)
            if audit.get("candidate_fingerprint") != fingerprint:
                raise AutomaticSuccessorError(
                    f"{candidate_path} specification fingerprint changed"
                )
            if record.get("candidate_fingerprint") not in (
                None,
                fingerprint,
            ):
                raise AutomaticSuccessorError(
                    f"{candidate_path} persisted candidate fingerprint changed"
                )
            history_item = history_by_id[rec_id]
            history_status = str(history_item.get("status", "")).lower()
            recovered_cancellation = _validate_candidate_history_alignment(
                record,
                history_item,
                candidate_path=candidate_path,
            )
            if recovered_cancellation:
                recovered_cancellation_ids.append(rec_id)
            if (
                audit.get("search_algorithm") != "bayesian"
                or audit.get("search_seed")
                != manifest["search"]["search_seed"]
                or audit.get("custom_parameter_ranges_sha256")
                != canonical_audit_sha256(
                    audit.get("custom_parameter_ranges")
                )
                or audit.get("history_visible_sha256")
                != canonical_audit_sha256(
                    audit.get("history_visible_to_algorithm")
                )
            ):
                raise AutomaticSuccessorError(
                    f"{candidate_path} algorithm identity changed"
                )
            visible = audit.get("history_visible_to_algorithm")
            if not isinstance(visible, list):
                raise AutomaticSuccessorError(
                    f"{candidate_path} visible history is missing"
                )
            expected_visible = [
                _history_snapshot(history_by_id[str(index)])
                for index in range(int(rec_id))
            ]
            if [
                _visible_history_snapshot(item) for item in visible
            ] != expected_visible:
                raise AutomaticSuccessorError(
                    f"{candidate_path} visible history differs from the "
                    "issuance-ordered runner history"
                )
            successful_visible = [
                item
                for item in visible
                if item.get("status") in SUCCESS_CANDIDATE_STATES
            ]
            if audit.get("previous_successful_observations") != (
                successful_visible
            ):
                raise AutomaticSuccessorError(
                    f"{candidate_path} successful observation snapshot changed"
                )
            proposal = audit.get("acquisition", {}).get("proposal", {})
            decision = proposal.get("decision_state", {})
            if (
                proposal.get("acquisition_mode") != mode
                or decision.get("mode") != mode
                or decision.get("uses_raw_objectives") is not True
                or decision.get("selector_score_used") is not False
                or decision.get("observation_count")
                != len(successful_visible)
            ):
                raise AutomaticSuccessorError(
                    f"{candidate_path} objective-aware acquisition evidence changed"
                )
            should_be_model_based = len(successful_visible) >= calibration
            is_expected_model_based = (
                proposal.get("stage") == "model_based"
                and decision.get("stage") == "model_based"
                and decision.get("active_method")
                == EXPECTED_MODEL_BASED_METHOD[mode]
                and decision.get("optimization_direction")
                == EXPECTED_OPTIMIZATION_DIRECTION[mode]
            )
            if should_be_model_based and not is_expected_model_based:
                raise AutomaticSuccessorError(
                    f"{candidate_path} reverted from objective-aware "
                    "model-based acquisition after calibration"
                )
            if not should_be_model_based and proposal.get("stage") == (
                "model_based"
            ):
                raise AutomaticSuccessorError(
                    f"{candidate_path} model-based acquisition began incorrectly"
                )
            if is_expected_model_based:
                adaptive_ids.append(rec_id)
            if history_status in SUCCESS_CANDIDATE_STATES:
                _validate_success_candidate(
                    record,
                    candidate_path=candidate_path,
                    manifest=manifest,
                    fingerprint=fingerprint,
                )
                success_ids.add(rec_id)
                archive.append(
                    {
                        "id": int(rec_id),
                        "specs": copy.deepcopy(specs),
                        "status": history_status,
                        "objective_values": copy.deepcopy(
                            history_item.get("objective_values", {})
                        ),
                    }
                )

        if len(success_ids) <= calibration or not adaptive_ids:
            raise AutomaticSuccessorError(
                f"{mode} never reached objective-aware model-based acquisition"
            )
        best = result.get("best")
        if not isinstance(best, Mapping):
            raise AutomaticSuccessorError(f"{mode} result has no best candidate")
        best_id = str(best.get("rec_id"))
        if best_id not in success_ids:
            raise AutomaticSuccessorError(
                f"{mode} selected a failed or missing candidate"
            )
        if (
            _selection_winner_id(
                result,
                mode,
                manifest=manifest,
                archive=archive,
            )
            != best_id
        ):
            raise AutomaticSuccessorError(
                f"{mode} runner winner differs from selector evidence"
            )
        modes[mode] = {
            "result_path": str(result_path),
            "result_sha256": sha256_file(result_path),
            "candidate_evidence_path": str(evidence_path),
            "candidate_evidence_sha256": sha256_file(evidence_path),
            "completed_candidates": budget,
            "successful_candidates": len(success_ids),
            "failed_candidates": budget - len(success_ids),
            "recovered_canceled_candidate_ids": sorted(
                recovered_cancellation_ids,
                key=lambda value: int(value),
            ),
            "model_based_candidate_ids": sorted(
                adaptive_ids,
                key=lambda value: int(value),
            ),
            "selected_rec_id": best_id,
        }

    report = {
        "schema_version": 1,
        "status": "passed",
        "predecessor_campaign_id": predecessor["campaign_id"],
        "predecessor_manifest_sha256": expected_manifest_sha,
        "predecessor_manifest_path": predecessor["manifest_path"],
        "predecessor_manifest_file_sha256": predecessor[
            "manifest_file_sha256"
        ],
        "runtime_root": str(runtime_root),
        "mode_process_status_path": str(process_status_path),
        "mode_process_status_sha256": sha256_file(process_status_path),
        "modes": modes,
    }
    report["gate_evidence_sha256"] = canonical_sha256(report)
    return report


def trigger_successor(
    descriptor: Mapping[str, Any],
    *,
    state_dir: Path,
    gate_report: Mapping[str, Any],
) -> int:
    """Execute one sealed successor command and never silently retry it."""
    successor = descriptor["successor"]
    state_path = state_dir / "automatic_successor_state.json"
    if state_path.exists():
        previous = load_json(state_path)
        raise AutomaticSuccessorError(
            "automatic successor already has terminal or running state: "
            f"{previous.get('status')!r}"
        )
    inputs = verify_successor_inputs(descriptor)
    completion_path = Path(successor["completion_artifact"])
    if completion_path.exists():
        raise AutomaticSuccessorError(
            "fresh automatic successor refuses a pre-existing completion "
            "artifact"
        )
    launch_not_before_ns = time.time_ns()
    pending = {
        "schema_version": 1,
        "status": "successor_spawn_pending",
        "started_at_utc": utc_timestamp(),
        "descriptor_sha256": descriptor["descriptor_sha256"],
        "gate_evidence_sha256": gate_report["gate_evidence_sha256"],
        "successor_name": successor["name"],
        "command": list(successor["command"]),
        "working_directory": successor["working_directory"],
        "required_file_evidence": inputs,
        "cpu_runs": 0,
        "smoke_runs": 0,
        "launch_not_before_ns": launch_not_before_ns,
    }
    atomic_json(state_path, pending)
    log_path = state_dir / "automatic_successor.log"
    with log_path.open("ab", buffering=0) as log:
        try:
            process = subprocess.Popen(
                successor["command"],
                cwd=successor["working_directory"],
                env=dict(successor["environment"]),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            failed = {
                **pending,
                "status": "successor_spawn_failed",
                "finished_at_utc": utc_timestamp(),
                "failure_type": type(exc).__name__,
                "log_path": str(log_path),
            }
            atomic_json(state_path, failed)
            raise AutomaticSuccessorError(
                f"successor process could not start: {type(exc).__name__}"
            ) from exc
        child_identity = None
        for _attempt in range(50):
            child_identity = _process_identity(process.pid)
            if child_identity is not None:
                break
            time.sleep(0.01)
        if child_identity is None:
            process.terminate()
            failed = {
                **pending,
                "status": "successor_identity_failed",
                "finished_at_utc": utc_timestamp(),
                "pid": process.pid,
                "log_path": str(log_path),
            }
            atomic_json(state_path, failed)
            raise AutomaticSuccessorError(
                "successor process identity could not be recorded"
            )
        started = {
            **pending,
            "status": "successor_running",
            "pid": process.pid,
            "process_identity": child_identity,
        }
        atomic_json(state_path, started)
        decision_path = state_dir / "gate_decision.json"
        decision = load_json(decision_path)
        decision.update(
            {
                "successor_submitted": True,
                "successor_pid": process.pid,
                "successor_process_identity": child_identity,
                "successor_started_at_utc": utc_timestamp(),
            }
        )
        atomic_json(decision_path, decision)
        return_code = process.wait()
    try:
        completion = validate_successor_completion(
            descriptor,
            not_before_ns=launch_not_before_ns,
        )
    except AutomaticSuccessorError as exc:
        invalid = {
            **started,
            "status": "successor_completion_invalid",
            "finished_at_utc": utc_timestamp(),
            "return_code": return_code,
            "log_path": str(log_path),
            "completion_artifact": successor["completion_artifact"],
            "completion_error": str(exc),
        }
        atomic_json(state_path, invalid)
        raise
    successful = (
        return_code == 0 and completion.get("status") == "success"
    )
    finished = {
        **started,
        "status": (
            "successor_completed"
            if successful
            else "successor_failed"
        ),
        "finished_at_utc": utc_timestamp(),
        "return_code": return_code,
        "log_path": str(log_path),
        "completion_artifact": successor["completion_artifact"],
        "completion_artifact_present": True,
        "completion_status": completion["status"],
        "completion_sha256": completion["completion_sha256"],
        "completion_file_sha256": completion["completion_file_sha256"],
    }
    atomic_json(state_path, finished)
    return 0 if successful else (return_code or 1)


def reconcile_existing_trigger(
    descriptor: Mapping[str, Any],
    state_dir: Path,
) -> int | None:
    state_path = state_dir / "automatic_successor_state.json"
    if not state_path.is_file():
        return None
    state = load_json(state_path)
    if state.get("descriptor_sha256") != descriptor["descriptor_sha256"]:
        raise AutomaticSuccessorError(
            "existing automatic successor state belongs to another descriptor"
        )
    status = state.get("status")
    if status == "successor_completed":
        completion = validate_successor_completion(descriptor)
        if completion["status"] != "success":
            raise AutomaticSuccessorError(
                "completed successor state has a non-success completion"
            )
        return 0
    if status == "successor_running":
        identity = state.get("process_identity")
        pid = state.get("pid")
        if (
            isinstance(identity, Mapping)
            and isinstance(pid, int)
            and _process_identity(pid) == dict(identity)
        ):
            return 4
        completion = Path(descriptor["successor"]["completion_artifact"])
        if completion.is_file():
            raise AutomaticSuccessorError(
                "successor exited and produced its completion artifact, but "
                "the detached watcher cannot reconstruct the exit code; "
                "submission will not be repeated"
            )
        raise AutomaticSuccessorError(
            "successor process is no longer running; submission will not be "
            "repeated without explicit audit"
        )
    raise AutomaticSuccessorError(
        f"existing automatic successor state is terminal: {status!r}"
    )


def watch_and_trigger(
    descriptor_path: Path,
    runtime_root: Path,
    *,
    poll_seconds: float,
    once: bool = False,
) -> int:
    descriptor = validate_successor_descriptor(descriptor_path)
    expected_runtime = Path(descriptor["predecessor"]["runtime_root"])
    if runtime_root.resolve() != expected_runtime.resolve():
        raise AutomaticSuccessorError(
            "runtime root differs from the sealed predecessor descriptor"
        )
    state_dir = runtime_root / "automatic_successor"
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "watcher.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AutomaticSuccessorError(
                "another automatic successor watcher is already active"
            ) from exc
        reconciled = reconcile_existing_trigger(descriptor, state_dir)
        if reconciled is not None:
            return reconciled
        load_predecessor_manifest(descriptor)
        successor_inputs = verify_successor_inputs(descriptor)
        atomic_json(
            state_dir / "successor_preflight.json",
            {
                "schema_version": 1,
                "status": "ready",
                "verified_at_utc": utc_timestamp(),
                "descriptor_sha256": descriptor["descriptor_sha256"],
                "successor_name": descriptor["successor"]["name"],
                "successor_input_evidence": successor_inputs,
                "cpu_runs": 0,
                "smoke_runs": 0,
            },
        )
        while not (runtime_root / "mode_process_status.json").is_file():
            try:
                verify_controller_alive(descriptor)
            except AutomaticSuccessorError as exc:
                atomic_json(
                    state_dir / "gate_decision.json",
                    {
                        "schema_version": 1,
                        "status": "blocked",
                        "decided_at_utc": utc_timestamp(),
                        "descriptor_sha256": descriptor[
                            "descriptor_sha256"
                        ],
                        "reason": str(exc),
                        "successor_submitted": False,
                    },
                )
                raise
            atomic_json(
                state_dir / "watcher_status.json",
                {
                    "schema_version": 1,
                    "status": "waiting_for_dino_completion",
                    "updated_at_utc": utc_timestamp(),
                    "descriptor_sha256": descriptor["descriptor_sha256"],
                    "runtime_root": str(runtime_root),
                },
            )
            if once:
                return 3
            time.sleep(poll_seconds)
        try:
            report = validate_completed_dino(descriptor, runtime_root)
            inputs = verify_successor_inputs(descriptor)
        except AutomaticSuccessorError as exc:
            atomic_json(
                state_dir / "gate_decision.json",
                {
                    "schema_version": 1,
                    "status": "blocked",
                    "decided_at_utc": utc_timestamp(),
                    "descriptor_sha256": descriptor["descriptor_sha256"],
                    "reason": str(exc),
                    "successor_submitted": False,
                },
            )
            raise
        gate_decision = {
            **report,
            "decided_at_utc": utc_timestamp(),
            "descriptor_sha256": descriptor["descriptor_sha256"],
            "successor_name": descriptor["successor"]["name"],
            "successor_input_evidence": inputs,
            "successor_submitted": False,
        }
        atomic_json(state_dir / "gate_decision.json", gate_decision)
        return trigger_successor(
            descriptor,
            state_dir=state_dir,
            gate_report=report,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        raise AutomaticSuccessorError("--poll-seconds must be finite and > 0")
    return watch_and_trigger(
        args.descriptor.resolve(),
        args.runtime_root.resolve(),
        poll_seconds=args.poll_seconds,
        once=args.once,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AutomaticSuccessorError as exc:
        print(f"automatic successor blocked: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
