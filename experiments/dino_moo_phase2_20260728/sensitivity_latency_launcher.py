#!/usr/bin/env python3

"""Fail-closed launcher for nine matched DINO sensitivity latency blocks."""

from __future__ import annotations

import argparse
import base64
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import yaml

from sensitivity_latency_common import (
    DEFAULT_MANIFEST,
    build_profiles,
    build_schedule,
    canonical_bytes,
    load_checkpoint_artifact,
    load_contract,
    resolve_relative,
    sha256_file,
    sha256_value,
)


HERE = Path(__file__).resolve().parent
BLOCK_RUNNER = HERE / "sensitivity_latency_block_runner.py"
DEFAULT_RUNTIME = HERE / "runtime" / "sensitivity_latency"
STAGING_ROOT = Path("/tmp/dino_sensitivity_latency_20260728")
# Linux limits every argv/env string to 32 pages (128 KiB on the target's
# 4-KiB-page nodes), independently of ARG_MAX.  Stay at half that per argument
# and half the observed 2-MiB ARG_MAX for the complete rendered command.
STAGING_CHUNK_BYTES = 48 * 1024
MAX_RUNTIME_ARGUMENT_BYTES = 64 * 1024
MAX_RENDERED_COMMAND_BYTES = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate or submit all nine frozen sensitivity blocks."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Render and validate all nine blocks without submission (default).",
    )
    mode.add_argument(
        "--submit-blocks",
        action="store_true",
        help="Submit the complete nine-block plan concurrently.",
    )
    mode.add_argument(
        "--retry-allocation",
        metavar="ALLOCATION_ID",
        help=(
            "Submit one fresh complete 14-profile replacement block and "
            "write a new immutable nine-entry ledger that supersedes the "
            "provided prior ledger."
        ),
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint-artifact", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-artifact-sha256",
        required=True,
        help="Independent immutable whole-file SHA256 of checkpoint artifact.",
    )
    parser.add_argument(
        "--verify-remote",
        action="store_true",
        help="Verify SQSH, dataset, checkpoint, and accuracy evidence over SSH.",
    )
    parser.add_argument(
        "--acknowledgement",
        default="",
        help="Exact user-authorized acknowledgement required for submission.",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--prior-submission-ledger", type=Path)
    parser.add_argument("--prior-submission-ledger-sha256")
    parser.add_argument("--retry-ledger", type=Path)
    parser.add_argument("--retry-evidence", type=Path)
    parser.add_argument("--retry-evidence-sha256")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def git_value(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_sources(
    manifest_path: Path,
    contract: dict[str, Any],
) -> tuple[Path, Path, dict[str, str]]:
    frozen = contract["frozen_inputs"]
    benchmark = resolve_relative(manifest_path, frozen["benchmark_path"])
    evaluate_template = Path(frozen["evaluate_template_path"])
    checks = {}
    for label, path, expected in (
        ("benchmark", benchmark, frozen["benchmark_sha256"]),
        (
            "evaluate_template",
            evaluate_template,
            frozen["evaluate_template_sha256"],
        ),
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{label} drift: {actual} != {expected}")
        checks[f"{label}_sha256"] = actual
    runtime = contract["runtime_contract"]
    sdk_path = Path(runtime["sdk_path"])
    sdk_commit = git_value(sdk_path, "rev-parse", "HEAD")
    sdk_branch = git_value(sdk_path, "branch", "--show-current")
    if sdk_commit != runtime["sdk_commit"]:
        raise RuntimeError("TAO SDK commit drift")
    if sdk_branch != runtime["sdk_branch"]:
        raise RuntimeError("TAO SDK branch drift")
    checks["sdk_commit"] = sdk_commit
    checks["sdk_branch"] = sdk_branch
    automl_path = Path(runtime["automl_path"])
    automl_commit = git_value(automl_path, "rev-parse", "HEAD")
    automl_branch = git_value(automl_path, "branch", "--show-current")
    ancestor = runtime["automl_required_ancestor_commit"]
    ancestor_check = subprocess.run(
        ["git", "-C", str(automl_path), "merge-base", "--is-ancestor",
         ancestor, automl_commit],
        check=False,
    )
    if ancestor_check.returncode != 0:
        raise RuntimeError(
            "TAO AutoML HEAD does not contain the required ancestor commit"
        )
    if automl_branch != runtime["automl_branch"]:
        raise RuntimeError("TAO AutoML branch drift")
    checks["automl_commit"] = automl_commit
    checks["automl_branch"] = automl_branch
    checks["automl_required_ancestor_commit"] = ancestor
    source_paths = (
        ("launcher", Path(__file__).resolve()),
        ("block_runner", BLOCK_RUNNER),
        ("common", HERE / "sensitivity_latency_common.py"),
        ("aggregator", HERE / "sensitivity_latency_aggregate.py"),
        (
            "latency_stats",
            automl_path / "src" / "tao_automl" / "latency_stats.py",
        ),
    )
    pinned_sources = runtime["source_code_sha256"]
    for label, path in source_paths:
        actual = sha256_file(path)
        if actual != pinned_sources[label]:
            raise RuntimeError(
                f"{label} source drift: {actual} != "
                f"{pinned_sources[label]}"
            )
        checks[f"{label}_sha256"] = actual
    return benchmark, evaluate_template, checks


def validate_submission_source_state(automl_path: Path) -> None:
    required = [
        Path(__file__).resolve(),
        BLOCK_RUNNER,
        HERE / "sensitivity_latency_common.py",
        HERE / "sensitivity_latency_aggregate.py",
        automl_path / "src" / "tao_automl" / "latency_stats.py",
    ]
    relative = [str(path.relative_to(automl_path)) for path in required]
    tracked = subprocess.run(
        ["git", "-C", str(automl_path), "ls-files", "--error-unmatch",
         "--", *relative],
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        raise RuntimeError(
            "submission requires every harness source to be committed/tracked"
        )
    for cached in (False, True):
        command = ["git", "-C", str(automl_path), "diff", "--quiet"]
        if cached:
            command.append("--cached")
        command.extend(["--", *relative])
        if subprocess.run(command, check=False).returncode != 0:
            raise RuntimeError(
                "submission requires clean committed harness source files"
            )


def evaluation_config(
    contract: dict[str, Any],
    template: dict[str, Any],
    profile: dict[str, Any],
    checkpoint: str,
    seed: int,
) -> dict[str, Any]:
    config = copy.deepcopy(template)
    evaluation = contract["evaluation_config_contract"]
    runtime = contract["runtime_contract"]
    config["model"] = copy.deepcopy(profile["model"])
    config["wandb"]["enable"] = False
    config["dataset"]["num_classes"] = evaluation["dataset_num_classes"]
    config["dataset"]["eval_class_ids"] = copy.deepcopy(
        evaluation["eval_class_ids"]
    )
    config["dataset"]["batch_size"] = 1
    config["dataset"]["workers"] = 0
    config["dataset"]["pin_memory"] = False
    config["dataset"]["test_data_sources"] = {
        "image_dir": evaluation["test_image_dir"],
        "json_file": evaluation["test_annotation"],
    }
    augmentation = config["dataset"]["augmentation"]
    augmentation["test_random_resize"] = evaluation["test_random_resize"]
    augmentation["random_resize_max_size"] = evaluation[
        "random_resize_max_size"
    ]
    augmentation["fixed_padding"] = evaluation["fixed_padding"]
    config["train"]["activation_checkpoint"] = False
    config["train"]["seed"] = seed
    config["train"]["precision"] = "fp32"
    config["train"]["cudnn"]["benchmark"] = False
    config["train"]["cudnn"]["deterministic"] = True
    config["evaluate"]["checkpoint"] = checkpoint
    config["evaluate"]["batch_size"] = 1
    config["evaluate"]["num_gpus"] = runtime["gpu_count"]
    config["evaluate"]["gpu_ids"] = list(range(runtime["gpu_count"]))
    config["evaluate"]["num_nodes"] = runtime["num_nodes"]
    return config


def yaml_payload(value: Any) -> bytes:
    return yaml.safe_dump(value, sort_keys=True).encode("utf-8")


def build_block_plan(
    contract: dict[str, Any],
    manifest_sha256: str,
    checkpoint_artifact_sha256: str,
    profiles: list[dict[str, Any]],
    artifact_entries: dict[tuple[int, str], dict[str, Any]],
    block: dict[str, Any],
    configs: dict[tuple[int, str], bytes],
) -> dict[str, Any]:
    profile_by_id = {
        profile["profile_id"]: profile for profile in profiles
    }
    planned = []
    for position, profile_id in enumerate(block["profile_order"]):
        profile = profile_by_id[profile_id]
        artifact = artifact_entries[(block["seed"], profile_id)]
        config_path = (
            STAGING_ROOT
            / "configs"
            / f"seed_{block['seed']:06d}"
            / f"{profile_id}.yaml"
        )
        planned.append(
            {
                "profile_id": profile_id,
                "axis": profile["axis"],
                "level": profile["level"],
                "seed": block["seed"],
                "position": position,
                "run_label": (
                    f"{block['allocation_id']}_p{position:02d}_{profile_id}"
                ),
                "checkpoint_path": artifact["checkpoint_path"],
                "checkpoint_sha256": artifact["checkpoint_sha256"],
                "checkpoint_source_profile_id": artifact[
                    "checkpoint_source_profile_id"
                ],
                "resolved_model_spec_sha256": profile[
                    "resolved_model_spec_sha256"
                ],
                "config_path": str(config_path),
                "config_sha256": hashlib.sha256(
                    configs[(block["seed"], profile_id)]
                ).hexdigest(),
                "feeds_final_selection": False,
            }
        )
    runtime = contract["runtime_contract"]
    plan = {
        "schema_version": 1,
        "manifest_id": contract["manifest_id"],
        "manifest_sha256": manifest_sha256,
        "checkpoint_artifact_sha256": checkpoint_artifact_sha256,
        "schedule_sha256": contract["design"]["schedule_sha256"],
        "allocation_id": block["allocation_id"],
        "seed": block["seed"],
        "repeat_index": block["repeat_index"],
        "williams_row_index": block["williams_row_index"],
        "gpu_count": 8,
        "feeds_final_selection": False,
        "manual_promotion_permitted": False,
        "benchmark_sha256": contract["frozen_inputs"]["benchmark_sha256"],
        "expected_hardware": {
            "gpu_name": runtime["required_gpu_name"],
            "compute_capability": runtime["required_compute_capability"],
            "total_memory_bytes": runtime["required_total_memory_bytes"],
            "torch": runtime["required_torch"],
            "cuda": runtime["required_cuda"],
            "cudnn": runtime["required_cudnn"],
        },
        "latency_protocol": copy.deepcopy(contract["latency_protocol"]),
        "output_contract": copy.deepcopy(
            contract["runtime_contract"]["output_contract"]
        ),
        "profiles": planned,
    }
    plan["block_plan_sha256"] = sha256_value(plan)
    return plan


def staged_command(
    benchmark: Path,
    block: dict[str, Any],
    plan: dict[str, Any],
    configs: dict[tuple[int, str], bytes],
) -> tuple[str, dict[str, Any]]:
    files = {
        "sensitivity_latency_block_runner.py": BLOCK_RUNNER.read_bytes(),
        "dino_latency_benchmark.py": benchmark.read_bytes(),
        f"plans/{block['allocation_id']}.json": (
            json.dumps(plan, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    }
    for profile in plan["profiles"]:
        profile_id = profile["profile_id"]
        files[
            f"configs/seed_{block['seed']:06d}/{profile_id}.yaml"
        ] = configs[(block["seed"], profile_id)]
    file_sha256 = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in sorted(files.items())
    }
    encoded_files = {
        name: base64.b64encode(payload).decode("ascii")
        for name, payload in files.items()
    }
    bundle_json = json.dumps(
        encoded_files,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    compressed_bundle = gzip.compress(bundle_json, compresslevel=9, mtime=0)
    encoded_payload = base64.b64encode(compressed_bundle).decode("ascii")
    chunks = [
        encoded_payload[offset : offset + STAGING_CHUNK_BYTES]
        for offset in range(0, len(encoded_payload), STAGING_CHUNK_BYTES)
    ]
    if not chunks:
        raise RuntimeError("staging bundle must not be empty")
    max_chunk_bytes = max(
        len(chunk.encode("ascii")) for chunk in chunks
    )
    if max_chunk_bytes > STAGING_CHUNK_BYTES:
        raise RuntimeError("staging chunk exceeds the configured chunk size")
    compressed_bundle_sha256 = hashlib.sha256(compressed_bundle).hexdigest()
    bundle_json_sha256 = hashlib.sha256(bundle_json).hexdigest()
    installer = "\n".join(
        [
            "import base64,gzip,hashlib,json,sys",
            "from pathlib import Path",
            f"root=Path({str(STAGING_ROOT)!r})",
            "encoded=''.join(sys.argv[1:])",
            "compressed=base64.b64decode(encoded,validate=True)",
            (
                "actual=hashlib.sha256(compressed).hexdigest();"
                f"expected={compressed_bundle_sha256!r}"
            ),
            (
                "if actual!=expected: raise RuntimeError("
                "f'compressed staging bundle digest mismatch: "
                "{actual} != {expected}')"
            ),
            "bundle=gzip.decompress(compressed)",
            (
                "actual=hashlib.sha256(bundle).hexdigest();"
                f"expected={bundle_json_sha256!r}"
            ),
            (
                "if actual!=expected: raise RuntimeError("
                "f'staging bundle digest mismatch: {actual} != {expected}')"
            ),
            "files=json.loads(bundle)",
            f"expected_files={file_sha256!r}",
            (
                "if set(files)!=set(expected_files): raise RuntimeError("
                "'staging bundle file set mismatch')"
            ),
            "for name in sorted(files):",
            " relative=Path(name)",
            (
                " if relative.is_absolute() or '..' in relative.parts: "
                "raise RuntimeError('unsafe staged path')"
            ),
            " payload=base64.b64decode(files[name],validate=True)",
            " actual=hashlib.sha256(payload).hexdigest()",
            (
                " if actual!=expected_files[name]: raise RuntimeError("
                "f'staged file digest mismatch: {name}')"
            ),
            " path=root/name",
            " path.parent.mkdir(parents=True,exist_ok=True)",
            " pending=path.with_suffix(path.suffix+'.tmp')",
            " pending.write_bytes(payload)",
            " pending.replace(path)",
        ]
    )
    plan_path = STAGING_ROOT / "plans" / f"{block['allocation_id']}.json"
    command_parts = ["python", "-c", shlex.quote(installer)]
    command_parts.extend(shlex.quote(chunk) for chunk in chunks)
    command_parts.extend(
        [
            "&&",
            "python",
            shlex.quote(str(STAGING_ROOT / BLOCK_RUNNER.name)),
            "--plan",
            shlex.quote(str(plan_path)),
            "--benchmark-script",
            shlex.quote(str(STAGING_ROOT / "dino_latency_benchmark.py")),
            "--output-root",
            '"$TAO_RESULTS_ROOT/$TAO_JOB_ID"',
        ]
    )
    command = " ".join(command_parts)
    installer_bytes = len(installer.encode("utf-8"))
    static_runtime_arguments = [
        installer,
        *chunks,
        str(STAGING_ROOT / BLOCK_RUNNER.name),
        "--plan",
        str(plan_path),
        "--benchmark-script",
        str(STAGING_ROOT / "dino_latency_benchmark.py"),
        "--output-root",
        "$TAO_RESULTS_ROOT/$TAO_JOB_ID",
    ]
    runtime_argument_bytes = max(
        len(argument.encode("utf-8"))
        for argument in static_runtime_arguments
    )
    command_bytes = len(command.encode("utf-8"))
    if runtime_argument_bytes > MAX_RUNTIME_ARGUMENT_BYTES:
        raise RuntimeError(
            "staged command contains an argument above the fail-closed "
            f"{MAX_RUNTIME_ARGUMENT_BYTES}-byte limit"
        )
    if command_bytes > MAX_RENDERED_COMMAND_BYTES:
        raise RuntimeError(
            "rendered command exceeds the fail-closed total command limit: "
            f"{command_bytes} > {MAX_RENDERED_COMMAND_BYTES}"
        )
    summary = {
        "allocation_id": block["allocation_id"],
        "seed": block["seed"],
        "repeat_index": block["repeat_index"],
        "williams_row_index": block["williams_row_index"],
        "profile_order": block["profile_order"],
        "block_plan_sha256": plan["block_plan_sha256"],
        "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
        "command_bytes": command_bytes,
        "staging_bundle_compression": "gzip_mtime_0_level_9",
        "staging_bundle_uncompressed_bytes": len(bundle_json),
        "staging_bundle_compressed_bytes": len(compressed_bundle),
        "staging_bundle_sha256": compressed_bundle_sha256,
        "staging_bundle_json_sha256": bundle_json_sha256,
        "staging_file_sha256": file_sha256,
        "staging_chunk_count": len(chunks),
        "max_staging_chunk_bytes": max_chunk_bytes,
        "installer_argument_bytes": installer_bytes,
        "max_runtime_argument_bytes": runtime_argument_bytes,
        "runtime_argument_safety_limit_bytes": MAX_RUNTIME_ARGUMENT_BYTES,
        "rendered_command_safety_limit_bytes": MAX_RENDERED_COMMAND_BYTES,
        "output_root_expression": "$TAO_RESULTS_ROOT/$TAO_JOB_ID",
    }
    return command, summary


def load_env_file(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"required secrets env file not found: {path}")
    loaded = []
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"unsupported env line {line_number}")
        key, encoded = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ValueError(f"invalid env key on line {line_number}")
        tokens = shlex.split(encoded, comments=True, posix=True)
        if len(tokens) > 1:
            raise ValueError(f"unsupported env syntax on line {line_number}")
        os.environ.setdefault(key, tokens[0] if tokens else "")
        loaded.append(key)
    return sorted(loaded)


def ssh_target() -> str:
    host = os.environ.get("SLURM_HOSTNAME", "").split(",", 1)[0].strip()
    user = os.environ.get("SLURM_USER", "").strip()
    if not host or not user:
        raise RuntimeError("SLURM_USER and SLURM_HOSTNAME are required")
    return f"{user}@{host}"


def remote_file_sha256(path: str) -> tuple[str, str | None]:
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    key = os.environ.get("SSH_KEY_PATH")
    if key:
        command.extend(["-i", key])
    remote = (
        f"if test -f {shlex.quote(path)}; then "
        f"sha256sum {shlex.quote(path)}; else echo MISSING; fi"
    )
    command.extend([ssh_target(), remote])
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = completed.stdout.strip()
    if output == "MISSING":
        return "missing", None
    return "present", output.split(None, 1)[0]


def remote_directory_status(path: str) -> str:
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    key = os.environ.get("SSH_KEY_PATH")
    if key:
        command.extend(["-i", key])
    command.extend(
        [
            ssh_target(),
            (
                f"if test -d {shlex.quote(path)}; then "
                "echo PRESENT; else echo MISSING; fi"
            ),
        ]
    )
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return completed.stdout.strip().lower()


def verify_remote(
    contract: dict[str, Any],
    artifact_entries: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, Any]:
    runtime = contract["runtime_contract"]
    evaluation = contract["evaluation_config_contract"]
    declared = [
        ("sqsh", runtime["sqsh_path"], runtime["sqsh_sha256"]),
        (
            "validation_annotation",
            evaluation["test_annotation"],
            evaluation["test_annotation_sha256"],
        ),
    ]
    seen = {path for _, path, _ in declared}
    for entry in artifact_entries.values():
        path = entry["checkpoint_path"]
        if path not in seen:
            declared.append(
                ("checkpoint", path, entry["checkpoint_sha256"])
            )
            seen.add(path)
    artifacts = []
    for kind, path, expected in declared:
        status, actual = remote_file_sha256(path)
        artifacts.append(
            {
                "kind": kind,
                "path": path,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": status,
                "verified": status == "present" and actual == expected,
            }
        )
    image_dir = evaluation["test_image_dir"]
    directory_status = remote_directory_status(image_dir)
    artifacts.append(
        {
            "kind": "validation_image_dir",
            "path": image_dir,
            "status": directory_status,
            "verified": directory_status == "present",
        }
    )
    return {
        "verified": all(item["verified"] for item in artifacts),
        "artifacts": artifacts,
    }


def submit_blocks(
    contract: dict[str, Any],
    commands: list[tuple[str, dict[str, Any]]],
    runtime_dir: Path,
    *,
    manifest_sha256: str,
    checkpoint_artifact_sha256: str,
    schedule_sha256: str,
    source_checks: dict[str, str],
) -> list[dict[str, Any]]:
    sdk_path = contract["runtime_contract"]["sdk_path"]
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)
    from tao_sdk.platforms.slurm import SlurmSDK

    ledger = runtime_dir / "block_submissions.json"
    if ledger.exists():
        raise RuntimeError(
            f"submission ledger already exists; refusing duplicate launch: {ledger}"
        )
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime = contract["runtime_contract"]
    os.environ["SLURM_USE_SQSH"] = "false"
    os.environ["SLURM_PARTITION"] = runtime["partition"]
    os.environ["SLURM_ACCOUNT"] = runtime["account"]
    sdk = SlurmSDK(
        poll_interval=10,
        state_file=runtime_dir / "slurm_state.json",
    )
    submissions = []
    try:
        for command, summary in commands:
            job = sdk.create_job(
                image=runtime["sqsh_path"],
                command=command,
                gpu_count=runtime["gpu_count"],
                num_nodes=runtime["num_nodes"],
                partition=runtime["partition"],
                account=runtime["account"],
                env_vars={"NVIDIA_TF32_OVERRIDE": "0"},
            )
            identity = sdk._handler.get_job_runtime_identity(job.id)
            submissions.append(
                {
                    **summary,
                    "tao_job_id": job.id,
                    "slurm_job_id": identity.get("slurm_job_id", ""),
                    "sdk_results_uri": sdk.get_job_results_dir(job.id),
                    "feeds_final_selection": False,
                }
            )
            atomic_json(
                ledger,
                submission_ledger_payload(
                    contract,
                    submissions,
                    manifest_sha256=manifest_sha256,
                    checkpoint_artifact_sha256=(
                        checkpoint_artifact_sha256
                    ),
                    schedule_sha256=schedule_sha256,
                    source_checks=source_checks,
                    status="submitting_incomplete",
                ),
            )
    finally:
        sdk._monitor.stop()
        sdk._store.close()
    atomic_json(
        ledger,
        submission_ledger_payload(
            contract,
            submissions,
            manifest_sha256=manifest_sha256,
            checkpoint_artifact_sha256=checkpoint_artifact_sha256,
            schedule_sha256=schedule_sha256,
            source_checks=source_checks,
            status="complete",
        ),
    )
    return submissions


def submission_ledger_payload(
    contract: dict[str, Any],
    submissions: list[dict[str, Any]],
    *,
    manifest_sha256: str,
    checkpoint_artifact_sha256: str,
    schedule_sha256: str,
    source_checks: dict[str, str],
    status: str,
    supersedes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "status": status,
        "phase": "sensitivity_latency_blocks",
        "manifest_id": contract["manifest_id"],
        "manifest_sha256": manifest_sha256,
        "checkpoint_artifact_sha256": checkpoint_artifact_sha256,
        "schedule_sha256": schedule_sha256,
        "allocation_count": len(submissions),
        "expected_allocation_count": contract["design"]["allocation_count"],
        "feeds_final_selection": False,
        "manual_promotion_permitted": False,
        "source_checks": copy.deepcopy(source_checks),
        "submissions": copy.deepcopy(submissions),
    }
    if supersedes is not None:
        payload["supersedes"] = copy.deepcopy(supersedes)
    return payload


def validate_complete_ledger_for_retry(
    path: Path,
    expected_sha256: str,
    contract: dict[str, Any],
    commands: list[tuple[str, dict[str, Any]]],
    *,
    manifest_sha256: str,
    checkpoint_artifact_sha256: str,
    schedule_sha256: str,
) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("prior immutable submission ledger digest mismatch")
    ledger = json.loads(path.read_text())
    expected_summaries = {
        summary["allocation_id"]: summary for _, summary in commands
    }
    if (
        ledger.get("schema_version") != 1
        or ledger.get("status") != "complete"
        or ledger.get("phase") != "sensitivity_latency_blocks"
        or ledger.get("manifest_id") != contract["manifest_id"]
        or ledger.get("manifest_sha256") != manifest_sha256
        or ledger.get("checkpoint_artifact_sha256")
        != checkpoint_artifact_sha256
        or ledger.get("schedule_sha256") != schedule_sha256
        or ledger.get("feeds_final_selection") is not False
        or ledger.get("manual_promotion_permitted") is not False
    ):
        raise ValueError("prior submission ledger identity/policy mismatch")
    submissions = ledger.get("submissions")
    if not isinstance(submissions, list) or len(submissions) != len(
        expected_summaries
    ):
        raise ValueError("prior ledger must contain exactly nine submissions")
    tao_ids: set[str] = set()
    slurm_ids: set[str] = set()
    by_allocation: dict[str, dict[str, Any]] = {}
    for item in submissions:
        allocation_id = item.get("allocation_id")
        expected = expected_summaries.get(allocation_id)
        if expected is None or allocation_id in by_allocation:
            raise ValueError("prior ledger allocation identities are invalid")
        for key, value in expected.items():
            if item.get(key) != value:
                raise ValueError(
                    f"{allocation_id}: prior submitted {key} drift"
                )
        tao_id = item.get("tao_job_id")
        slurm_id = str(item.get("slurm_job_id", ""))
        if (
            not isinstance(tao_id, str)
            or not tao_id
            or not slurm_id.isdigit()
            or tao_id in tao_ids
            or slurm_id in slurm_ids
        ):
            raise ValueError("prior ledger job identities are invalid")
        tao_ids.add(tao_id)
        slurm_ids.add(slurm_id)
        by_allocation[allocation_id] = item
    if set(by_allocation) != set(expected_summaries):
        raise ValueError("prior ledger does not cover the frozen schedule")
    ledger["submissions"] = [
        by_allocation[summary["allocation_id"]]
        for _, summary in commands
    ]
    return ledger


def retry_complete_block(
    contract: dict[str, Any],
    command: str,
    summary: dict[str, Any],
    prior_ledger: dict[str, Any],
    prior_ledger_path: Path,
    prior_ledger_sha256: str,
    retry_ledger_path: Path,
    runtime_dir: Path,
    *,
    manifest_sha256: str,
    checkpoint_artifact_sha256: str,
    schedule_sha256: str,
    source_checks: dict[str, str],
    retry_evidence: dict[str, Any],
    retry_evidence_path: Path,
    retry_evidence_sha256: str,
) -> list[dict[str, Any]]:
    if retry_ledger_path.exists():
        raise RuntimeError(
            f"retry ledger already exists: {retry_ledger_path}"
        )
    sdk_path = contract["runtime_contract"]["sdk_path"]
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)
    from tao_sdk.platforms.slurm import SlurmSDK

    runtime = contract["runtime_contract"]
    os.environ["SLURM_USE_SQSH"] = "false"
    os.environ["SLURM_PARTITION"] = runtime["partition"]
    os.environ["SLURM_ACCOUNT"] = runtime["account"]
    sdk = SlurmSDK(
        poll_interval=10,
        state_file=runtime_dir / "slurm_state.json",
    )
    try:
        job = sdk.create_job(
            image=runtime["sqsh_path"],
            command=command,
            gpu_count=runtime["gpu_count"],
            num_nodes=runtime["num_nodes"],
            partition=runtime["partition"],
            account=runtime["account"],
            env_vars={"NVIDIA_TF32_OVERRIDE": "0"},
        )
        identity = sdk._handler.get_job_runtime_identity(job.id)
        replacement = {
            **summary,
            "tao_job_id": job.id,
            "slurm_job_id": identity.get("slurm_job_id", ""),
            "sdk_results_uri": sdk.get_job_results_dir(job.id),
            "retry_of": {
                "tao_job_id": next(
                    item["tao_job_id"]
                    for item in prior_ledger["submissions"]
                    if item["allocation_id"] == summary["allocation_id"]
                ),
                "reason": "complete_14_profile_block_retry",
            },
            "feeds_final_selection": False,
        }
    finally:
        sdk._monitor.stop()
        sdk._store.close()
    submissions = []
    replaced = None
    for item in prior_ledger["submissions"]:
        if item["allocation_id"] == summary["allocation_id"]:
            replaced = copy.deepcopy(item)
            submissions.append(replacement)
        else:
            submissions.append(copy.deepcopy(item))
    if replaced is None:
        raise RuntimeError("retry target is absent from prior ledger")
    payload = submission_ledger_payload(
        contract,
        submissions,
        manifest_sha256=manifest_sha256,
        checkpoint_artifact_sha256=checkpoint_artifact_sha256,
        schedule_sha256=schedule_sha256,
        source_checks=source_checks,
        status="complete",
        supersedes={
            "ledger_path": str(prior_ledger_path),
            "ledger_sha256": prior_ledger_sha256,
            "replaced_allocation_id": summary["allocation_id"],
            "replaced_submission": replaced,
            "retry_evidence_path": str(retry_evidence_path),
            "retry_evidence_sha256": retry_evidence_sha256,
            "retry_reason_code": retry_evidence["reason_code"],
            "replacement_tao_job_id": replacement["tao_job_id"],
            "replacement_slurm_job_id": replacement["slurm_job_id"],
            "policy": (
                "The prior ledger remains immutable. The replacement reruns "
                "all 14 profiles in the frozen order under one fresh TAO and "
                "SLURM allocation; no partial measurements are reused."
            ),
        },
    )
    atomic_json(retry_ledger_path, payload)
    return [replacement]


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    contract, one, one_path = load_contract(manifest_path)
    submitting = bool(args.submit_blocks or args.retry_allocation)
    if submitting:
        if args.retry_allocation:
            required_ack = (
                contract["submission_policy"][
                    "required_retry_acknowledgement_prefix"
                ]
                + args.retry_allocation
            )
        else:
            required_ack = contract["submission_policy"][
                "required_acknowledgement"
            ]
        if args.acknowledgement != required_ack:
            raise RuntimeError(
                "submission refused: exact user-authorized submission "
                "acknowledgement is required"
            )
        if not args.verify_remote:
            raise RuntimeError(
                "submission refused: --verify-remote is mandatory"
            )
    manifest_sha256 = sha256_file(manifest_path)
    benchmark, evaluate_template_path, source_checks = validate_sources(
        manifest_path, contract
    )
    if submitting:
        validate_submission_source_state(
            Path(contract["runtime_contract"]["automl_path"])
        )
        source_checks["submission_source_state"] = "tracked_and_clean"
    profiles = build_profiles(one)
    schedule = build_schedule(contract, profiles)
    artifact, artifact_entries = load_checkpoint_artifact(
        args.checkpoint_artifact.resolve(),
        args.checkpoint_artifact_sha256,
        contract,
        one,
        profiles,
    )
    template = yaml.safe_load(evaluate_template_path.read_text())
    configs = {}
    for seed in one["design"]["seeds"]:
        for profile in profiles:
            entry = artifact_entries[(seed, profile["profile_id"])]
            config = evaluation_config(
                contract,
                template,
                profile,
                entry["checkpoint_path"],
                seed,
            )
            if sha256_value(config["model"]) != profile[
                "resolved_model_spec_sha256"
            ]:
                raise RuntimeError("resolved evaluation model mapping drift")
            configs[(seed, profile["profile_id"])] = yaml_payload(config)

    plans = [
        build_block_plan(
            contract,
            manifest_sha256,
            args.checkpoint_artifact_sha256,
            profiles,
            artifact_entries,
            block,
            configs,
        )
        for block in schedule
    ]
    commands = [
        staged_command(benchmark, block, plan, configs)
        for block, plan in zip(schedule, plans, strict=True)
    ]
    remote = None
    loaded_secret_keys: list[str] = []
    if args.verify_remote or submitting:
        loaded_secret_keys = load_env_file(
            Path(contract["runtime_contract"]["secrets_env_path"])
        )
    if args.verify_remote:
        remote = verify_remote(contract, artifact_entries)

    if submitting:
        if remote is None or not remote["verified"]:
            raise RuntimeError(
                "submission refused: complete remote verification must pass"
            )
        schedule_sha256 = sha256_value(schedule)
        if args.retry_allocation:
            if (
                args.prior_submission_ledger is None
                or args.prior_submission_ledger_sha256 is None
                or args.retry_ledger is None
                or args.retry_evidence is None
                or args.retry_evidence_sha256 is None
            ):
                raise RuntimeError(
                    "retry requires --prior-submission-ledger, "
                    "--prior-submission-ledger-sha256, and --retry-ledger"
                    ", --retry-evidence, and --retry-evidence-sha256"
                )
            command_by_allocation = {
                summary["allocation_id"]: (command, summary)
                for command, summary in commands
            }
            if args.retry_allocation not in command_by_allocation:
                raise ValueError(
                    f"unknown retry allocation: {args.retry_allocation}"
                )
            prior_path = args.prior_submission_ledger.resolve()
            prior = validate_complete_ledger_for_retry(
                prior_path,
                args.prior_submission_ledger_sha256,
                contract,
                commands,
                manifest_sha256=manifest_sha256,
                checkpoint_artifact_sha256=(
                    args.checkpoint_artifact_sha256
                ),
                schedule_sha256=schedule_sha256,
            )
            command, summary = command_by_allocation[
                args.retry_allocation
            ]
            evidence_path = args.retry_evidence.resolve()
            evidence_bytes = evidence_path.read_bytes()
            if (
                hashlib.sha256(evidence_bytes).hexdigest()
                != args.retry_evidence_sha256
            ):
                raise RuntimeError("immutable retry evidence digest mismatch")
            retry_evidence = json.loads(evidence_bytes)
            prior_submission = next(
                item
                for item in prior["submissions"]
                if item["allocation_id"] == args.retry_allocation
            )
            if (
                retry_evidence.get("schema_version") != 1
                or retry_evidence.get("allocation_id")
                != args.retry_allocation
                or retry_evidence.get("prior_tao_job_id")
                != prior_submission["tao_job_id"]
                or str(retry_evidence.get("prior_slurm_job_id", ""))
                != str(prior_submission["slurm_job_id"])
                or retry_evidence.get("reason_code")
                not in {
                    "sdk_terminal_error",
                    "slurm_terminal_failure",
                    "complete_block_artifact_invalid",
                }
                or retry_evidence.get("retry_permitted") is not True
                or retry_evidence.get("partial_measurements_reusable")
                is not False
            ):
                raise ValueError("retry evidence identity/policy mismatch")
            submissions = retry_complete_block(
                contract,
                command,
                summary,
                prior,
                prior_path,
                args.prior_submission_ledger_sha256,
                args.retry_ledger.resolve(),
                args.runtime_dir.resolve(),
                manifest_sha256=manifest_sha256,
                checkpoint_artifact_sha256=(
                    args.checkpoint_artifact_sha256
                ),
                schedule_sha256=schedule_sha256,
                source_checks=source_checks,
                retry_evidence=retry_evidence,
                retry_evidence_path=evidence_path,
                retry_evidence_sha256=args.retry_evidence_sha256,
            )
            status = "replacement_block_submitted"
        else:
            if (
                args.prior_submission_ledger is not None
                or args.prior_submission_ledger_sha256 is not None
                or args.retry_ledger is not None
                or args.retry_evidence is not None
                or args.retry_evidence_sha256 is not None
            ):
                raise RuntimeError(
                    "retry-ledger arguments are valid only with "
                    "--retry-allocation"
                )
            submissions = submit_blocks(
                contract,
                commands,
                args.runtime_dir.resolve(),
                manifest_sha256=manifest_sha256,
                checkpoint_artifact_sha256=(
                    args.checkpoint_artifact_sha256
                ),
                schedule_sha256=schedule_sha256,
                source_checks=source_checks,
            )
            status = "submitted"
    else:
        submissions = []
        status = "dry_run_validated_not_submitted"

    report = {
        "schema_version": 1,
        "status": status,
        "manifest_id": contract["manifest_id"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "one_factor_manifest_path": str(one_path),
        "checkpoint_artifact_path": str(args.checkpoint_artifact.resolve()),
        "checkpoint_artifact_sha256": args.checkpoint_artifact_sha256,
        "checkpoint_artifact_id": artifact["artifact_id"],
        "feeds_final_selection": False,
        "manual_promotion_permitted": False,
        "source_checks": source_checks,
        "schedule_sha256": sha256_value(schedule),
        "profile_count": len(profiles),
        "allocation_count": len(schedule),
        "measurements_total": len(schedule) * len(profiles),
        "blocks": [summary for _, summary in commands],
        "block_plans": plans,
        "remote_verification": remote or {"status": "not_requested"},
        "loaded_secret_keys": loaded_secret_keys,
        "secret_values_recorded": False,
        "submissions": submissions,
        "retry_allocation": args.retry_allocation,
        "retry_ledger": (
            str(args.retry_ledger.resolve()) if args.retry_ledger else None
        ),
    }
    report["report_sha256"] = sha256_value(report)
    if args.report:
        atomic_json(args.report.resolve(), report)
    print(
        json.dumps(
            {
                "status": status,
                "manifest_id": contract["manifest_id"],
                "schedule_sha256": report["schedule_sha256"],
                "profile_count": report["profile_count"],
                "allocation_count": report["allocation_count"],
                "measurements_total": report["measurements_total"],
                "feeds_final_selection": False,
                "report": str(args.report.resolve()) if args.report else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
