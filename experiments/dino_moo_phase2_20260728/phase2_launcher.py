#!/usr/bin/env python3

"""Fail-closed preflight and launcher for matched DINO latency blocks."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "manifest.v1.json"
ALLOCATION_RUNNER = HERE / "allocation_runner.py"
EXPECTED_DEFAULT_MANIFEST_SHA256 = (
    "bd46e9160566845c71226aedf5ff08032b4eae70d442f32d7085582dfba738ad"
)
STAGING_ROOT = Path("/tmp/dino_moo_phase2_20260728")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and render all blocks without creating SLURM jobs (default).",
    )
    mode.add_argument(
        "--submit-recovery",
        action="store_true",
        help="Submit the four exact-config checkpoint-recovery training jobs.",
    )
    mode.add_argument(
        "--submit-blocks",
        action="store_true",
        help="Submit all six matched blocks after recovered artifacts are pinned.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--verify-remote",
        action="store_true",
        help="Read-only SHA256 verification of the SQSH and checkpoints over SSH.",
    )
    parser.add_argument(
        "--acknowledge-validation-only",
        action="store_true",
        help="Required for submission; confirms feeds_selection=false.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON preflight report path (runtime paths are recommended).",
    )
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if path.resolve() == DEFAULT_MANIFEST.resolve():
        if digest != EXPECTED_DEFAULT_MANIFEST_SHA256:
            raise RuntimeError(
                "immutable default manifest digest mismatch: "
                f"{digest} != {EXPECTED_DEFAULT_MANIFEST_SHA256}"
            )
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    if manifest.get("feeds_selection") is not False:
        raise ValueError("phase-2 manifest must set feeds_selection=false")
    return manifest, digest


def resolve_source(manifest_path: Path, relative_path: str) -> Path:
    return (manifest_path.parent / relative_path).resolve()


def load_phase1_module(path: Path):
    name = "dino_moo_phase1_pinned_run_experiment"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pinned phase-1 harness: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_schedule(manifest: dict[str, Any]) -> dict[str, Any]:
    candidate_ids = [item["candidate_id"] for item in manifest["candidates"]]
    candidate_set = set(candidate_ids)
    if len(candidate_ids) != 6 or len(candidate_set) != 6:
        raise ValueError("manifest must contain six unique candidates")
    schedule = manifest["schedule"]
    if len(schedule) != 6:
        raise ValueError("manifest must contain six allocation blocks")

    position_counts = {
        candidate_id: Counter() for candidate_id in candidate_ids
    }
    allocation_counts = Counter()
    adjacency_counts = Counter()
    for block in schedule:
        order = block["candidate_order"]
        if len(order) != 6 or set(order) != candidate_set:
            raise ValueError(
                f"{block['allocation_id']} is not a permutation of candidates"
            )
        for position, candidate_id in enumerate(order):
            position_counts[candidate_id][position] += 1
            allocation_counts[candidate_id] += 1
        adjacency_counts.update(zip(order, order[1:]))

    expected_positions = Counter({position: 1 for position in range(6)})
    if any(counts != expected_positions for counts in position_counts.values()):
        raise ValueError("every candidate must occupy every position exactly once")
    if set(allocation_counts.values()) != {6}:
        raise ValueError("every candidate must occur in all six allocations")
    expected_pairs = {
        (first, second)
        for first in candidate_ids
        for second in candidate_ids
        if first != second
    }
    if set(adjacency_counts) != expected_pairs or set(adjacency_counts.values()) != {
        1
    }:
        raise ValueError(
            "Williams schedule must contain every ordered adjacency exactly once"
        )
    return {
        "position_balance": "pass",
        "allocation_balance": "pass",
        "ordered_adjacency_balance": "pass",
    }


def validate_sources(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> tuple[Any, dict[str, str]]:
    source = manifest["source_artifacts"]
    checks = {}
    path_and_digest_fields = (
        ("combined_selection_path", "combined_selection_sha256"),
        ("dino_latency_benchmark_path", "dino_latency_benchmark_sha256"),
        ("phase1_hardware_contract_path", "phase1_hardware_contract_sha256"),
        ("phase1_launch_manifest_path", "phase1_launch_manifest_sha256"),
        ("phase1_run_experiment_path", "phase1_run_experiment_sha256"),
    )
    for path_key, digest_key in path_and_digest_fields:
        path = resolve_source(manifest_path, source[path_key])
        actual = sha256_file(path)
        expected = source[digest_key]
        if actual != expected:
            raise RuntimeError(
                f"pinned source drift for {path}: {actual} != {expected}"
            )
        checks[path_key] = actual
    for path_key, digest_key in (
        ("tao_dino_skill_info_path", "tao_dino_skill_info_sha256"),
        ("tao_dino_train_template_path", "tao_dino_train_template_sha256"),
    ):
        path = Path(source[path_key]).resolve()
        actual = sha256_file(path)
        if actual != source[digest_key]:
            raise RuntimeError(
                f"pinned skill source drift for {path}: "
                f"{actual} != {source[digest_key]}"
            )
        checks[path_key] = actual

    phase1_path = resolve_source(
        manifest_path,
        source["phase1_run_experiment_path"],
    )
    phase1 = load_phase1_module(phase1_path)
    if normalized(asdict(phase1.LATENCY_PROTOCOL)) != normalized(
        manifest["latency_protocol"]
    ):
        raise RuntimeError("phase-2 latency protocol differs from phase 1")
    frozen_benchmark = {
        "batch_size_per_gpu": phase1.LATENCY_BATCH_SIZE_PER_GPU,
        "benchmark_seed": phase1.LATENCY_BENCHMARK_SEED,
        "preloaded_batches": phase1.LATENCY_PRELOADED_BATCHES,
    }
    for key, value in frozen_benchmark.items():
        if manifest["latency_benchmark"][key] != value:
            raise RuntimeError(f"latency benchmark setting drift: {key}")
    if phase1.GPU_COUNT != manifest["hardware_and_runtime"]["gpu_count"]:
        raise RuntimeError("GPU count differs from the phase-1 harness")
    if phase1.SQSH_PATH != manifest["hardware_and_runtime"]["sqsh_path"]:
        raise RuntimeError("SQSH path differs from the phase-1 harness")
    return phase1, checks


def generate_configs(
    phase1: Any,
    manifest: dict[str, Any],
) -> dict[str, bytes]:
    configs = {}
    for candidate in manifest["candidates"]:
        specs = phase1.evaluation_specs(
            candidate["checkpoint"],
            candidate["num_queries"],
        )
        specs["dataset"]["batch_size"] = phase1.LATENCY_BATCH_SIZE_PER_GPU
        specs["evaluate"]["batch_size"] = phase1.LATENCY_BATCH_SIZE_PER_GPU
        payload = yaml.safe_dump(specs, sort_keys=True).encode("utf-8")
        actual = sha256_bytes(payload)
        if actual != candidate["config_sha256"]:
            raise RuntimeError(
                f"generated config drift for {candidate['candidate_id']}: "
                f"{actual} != {candidate['config_sha256']}"
            )
        configs[candidate["candidate_id"]] = payload
    return configs


def set_dotted_value(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor: Any = target
    parts = dotted_key.split(".")
    for raw_part in parts[:-1]:
        if "[" in raw_part:
            key, raw_index = raw_part[:-1].split("[", 1)
            cursor = cursor[key][int(raw_index)]
        else:
            cursor = cursor[raw_part]
    final = parts[-1]
    if "[" in final:
        key, raw_index = final[:-1].split("[", 1)
        cursor[key][int(raw_index)] = copy.deepcopy(value)
    else:
        cursor[final] = copy.deepcopy(value)


def generate_recovery_configs(
    phase1: Any,
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    template_path = Path(
        manifest["source_artifacts"]["tao_dino_train_template_path"]
    )
    template = yaml.safe_load(template_path.read_text())
    recovery_ids = set(manifest["checkpoint_recovery"]["candidate_ids"])
    configs = {}
    for candidate in manifest["candidates"]:
        candidate_id = candidate["candidate_id"]
        if candidate_id not in recovery_ids:
            continue
        specs = copy.deepcopy(template)
        overrides = {**phase1.SPEC_OVERRIDES, **candidate["specs"]}
        for key, value in overrides.items():
            set_dotted_value(specs, key, value)
        # This reproduces the phase-1 terminal checkpoint-retention strategy.
        specs["train"]["checkpoint_interval"] = specs["train"]["num_epochs"]
        payload = yaml.safe_dump(specs, sort_keys=True).encode("utf-8")
        actual = sha256_bytes(payload)
        expected = candidate["recovery_train_config_sha256"]
        if actual != expected:
            raise RuntimeError(
                f"recovery config drift for {candidate_id}: "
                f"{actual} != {expected}"
            )
        configs[candidate_id] = specs
    if set(configs) != recovery_ids:
        raise RuntimeError("recovery candidate set is incomplete")
    return configs


def recovery_commands(
    manifest: dict[str, Any],
    configs: dict[str, dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    from tao_sdk.script_runner import build_entrypoint

    skill_info = yaml.safe_load(
        Path(
            manifest["source_artifacts"]["tao_dino_skill_info_path"]
        ).read_text()
    )
    action = skill_info["actions"]["train"]
    commands = []
    for candidate_id in manifest["checkpoint_recovery"]["candidate_ids"]:
        entrypoint = build_entrypoint(
            command=action["command"],
            specs=configs[candidate_id],
            inputs=action["inputs"],
            outputs=action["outputs"],
            config_format=action["config_format"],
            upload_excludes=action["upload_excludes"],
        )
        command = entrypoint["command"]
        commands.append(
            (
                command,
                {
                    "candidate_id": candidate_id,
                    "command_sha256": sha256_bytes(command.encode("utf-8")),
                    "command_bytes": len(command.encode("utf-8")),
                    "training_seed": manifest["checkpoint_recovery"][
                        "training_seed"
                    ],
                    "train_epochs": manifest["checkpoint_recovery"][
                        "train_epochs"
                    ],
                    "feeds_selection": False,
                },
            )
        )
    return commands


def validate_candidate_evidence(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    source_path = resolve_source(
        manifest_path,
        manifest["source_artifacts"]["combined_selection_path"],
    )
    source = json.loads(source_path.read_text())
    audits = {item["candidate_id"]: item for item in source["candidates"]}
    records = source["candidate_records"]
    expected_front = {
        item["candidate_id"]
        for item in source["candidates"]
        if item["valid"] and item["pareto_rank"] == 0
    }
    manifest_ids = {item["candidate_id"] for item in manifest["candidates"]}
    if manifest_ids != expected_front:
        raise RuntimeError(
            "manifest candidate set differs from historical global Pareto front"
        )
    for candidate in manifest["candidates"]:
        candidate_id = candidate["candidate_id"]
        audit = audits[candidate_id]
        record = records[candidate_id]
        if audit["dominated_by"] or audit["pareto_rank"] != 0:
            raise RuntimeError(f"{candidate_id} is not globally nondominated")
        expected = {
            "checkpoint": record["checkpoint"],
            "num_queries": record["num_queries"],
            "specs": record["specs"],
            "mAP50": record["objective_values"]["mAP50"],
            "original_latency_median_ms": record["objective_values"]["latency_ms"],
            "original_latency_p95_ms": record["objective_values"]["latency_p95_ms"],
            "original_latency_ci95_low_ms": record["objective_values"][
                "latency_ci95_low"
            ],
            "original_latency_ci95_high_ms": record["objective_values"][
                "latency_ci95_high"
            ],
            "fingerprint": audit["fingerprint"],
            "pareto_rank": audit["pareto_rank"],
            "train_job_id": record["train_job_id"],
        }
        for key, value in expected.items():
            if candidate[key] != value:
                raise RuntimeError(f"historical evidence drift: {candidate_id}.{key}")
    return {
        "candidate_count": len(manifest_ids),
        "global_pareto_front_match": "pass",
        "all_dominated_by_sets_empty": "pass",
    }


def allocation_plan(
    manifest: dict[str, Any],
    block: dict[str, Any],
) -> dict[str, Any]:
    candidates = {
        item["candidate_id"]: item for item in manifest["candidates"]
    }
    planned = []
    for position, candidate_id in enumerate(block["candidate_order"]):
        run_label = (
            f"{manifest['manifest_id']}_{block['allocation_id']}_"
            f"p{position:02d}_{candidate_id}"
        )
        candidate = candidates[candidate_id]
        planned.append(
            {
                "candidate_id": candidate_id,
                "checkpoint": candidate["checkpoint"],
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "config_path": str(
                    STAGING_ROOT / "configs" / f"{candidate_id}.yaml"
                ),
                "position": position,
                "run_label": run_label,
            }
        )
    return {
        "schema_version": 1,
        "manifest_id": manifest["manifest_id"],
        "allocation_id": block["allocation_id"],
        "gpu_count": 8,
        "feeds_selection": False,
        "latency_protocol": manifest["latency_protocol"],
        "latency_benchmark": manifest["latency_benchmark"],
        "candidates": planned,
    }


def staged_command(
    manifest_path: Path,
    manifest: dict[str, Any],
    configs: dict[str, bytes],
    block: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    benchmark_path = resolve_source(
        manifest_path,
        manifest["source_artifacts"]["dino_latency_benchmark_path"],
    )
    plan = allocation_plan(manifest, block)
    relative_files = {
        "allocation_runner.py": ALLOCATION_RUNNER.read_bytes(),
        "dino_latency_benchmark.py": benchmark_path.read_bytes(),
        f"plans/{block['allocation_id']}.json": (
            json.dumps(plan, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    }
    for candidate_id, payload in configs.items():
        relative_files[f"configs/{candidate_id}.yaml"] = payload
    encoded = {
        path: base64.b64encode(payload).decode("ascii")
        for path, payload in relative_files.items()
    }
    encoded_payload = base64.b64encode(
        json.dumps(encoded, sort_keys=True).encode("utf-8")
    ).decode("ascii")
    installer = "\n".join(
        [
            "import base64,json",
            "from pathlib import Path",
            f"root=Path({str(STAGING_ROOT)!r})",
            f"files=json.loads(base64.b64decode({encoded_payload!r}))",
            "for name,value in files.items():",
            " path=root/name",
            " path.parent.mkdir(parents=True,exist_ok=True)",
            " path.write_bytes(base64.b64decode(value))",
        ]
    )
    plan_path = STAGING_ROOT / "plans" / f"{block['allocation_id']}.json"
    command = " ".join(
        [
            "python",
            "-c",
            shlex.quote(installer),
            "&&",
            "python",
            shlex.quote(str(STAGING_ROOT / "allocation_runner.py")),
            "--plan",
            shlex.quote(str(plan_path)),
            "--benchmark-script",
            shlex.quote(str(STAGING_ROOT / "dino_latency_benchmark.py")),
            "--output-root",
            '"$TAO_RESULTS_ROOT"',
        ]
    )
    summary = {
        "allocation_id": block["allocation_id"],
        "candidate_order": block["candidate_order"],
        "run_labels": [
            candidate["run_label"] for candidate in plan["candidates"]
        ],
        "command_sha256": sha256_bytes(command.encode("utf-8")),
        "command_bytes": len(command.encode("utf-8")),
    }
    return command, summary


def ssh_target() -> str:
    host = os.environ.get("SLURM_HOSTNAME", "").split(",", 1)[0].strip()
    user = os.environ.get("SLURM_USER", "").strip()
    if not host or not user:
        raise RuntimeError("SLURM_USER and SLURM_HOSTNAME are required")
    return f"{user}@{host}"


def remote_sha256(path: str) -> tuple[str, str | None]:
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    key_path = os.environ.get("SSH_KEY_PATH")
    if key_path:
        command.extend(["-i", key_path])
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
    digest = output.split(None, 1)[0]
    return "present", digest


def verify_remote_artifacts(manifest: dict[str, Any]) -> dict[str, Any]:
    artifacts = [
        {
            "kind": "sqsh",
            "path": manifest["hardware_and_runtime"]["sqsh_path"],
            "expected_sha256": manifest["hardware_and_runtime"]["sqsh_sha256"],
        },
        {
            "kind": "pretrained_model",
            "path": manifest["checkpoint_recovery"]["pretrained_model_path"],
            "expected_sha256": manifest["checkpoint_recovery"][
                "pretrained_model_sha256"
            ],
        },
        {
            "kind": "train_annotation",
            "path": manifest["dataset"]["train_annotation_path"],
            "expected_sha256": manifest["dataset"]["train_annotation_sha256"],
        },
        {
            "kind": "validation_annotation",
            "path": manifest["dataset"]["validation_annotation_path"],
            "expected_sha256": manifest["dataset"][
                "validation_annotation_sha256"
            ],
        }
    ]
    artifacts.extend(
        {
            "kind": "checkpoint",
            "candidate_id": candidate["candidate_id"],
            "path": candidate["checkpoint"],
            "expected_sha256": candidate["checkpoint_sha256"],
        }
        for candidate in manifest["candidates"]
    )
    results = []
    for artifact in artifacts:
        status, digest = remote_sha256(artifact["path"])
        result = {
            **artifact,
            "status": status,
            "actual_sha256": digest,
            "verified": (
                status == "present"
                and artifact["expected_sha256"] is not None
                and digest == artifact["expected_sha256"]
            ),
        }
        results.append(result)
    return {"artifacts": results}


def block_submission_blockers(
    manifest: dict[str, Any],
    remote: dict[str, Any] | None,
) -> list[str]:
    blockers = []
    for candidate in manifest["candidates"]:
        if candidate["checkpoint_sha256"] is None:
            blockers.append(
                f"{candidate['candidate_id']}: checkpoint missing/unpinned"
            )
    if remote is None:
        blockers.append("remote artifact verification not requested")
    else:
        blockers.extend(
            f"{item.get('candidate_id', item['kind'])}: remote artifact "
            f"{item['status']} or digest mismatch"
            for item in remote["artifacts"]
            if not item["verified"]
        )
    return blockers


def recovery_submission_blockers(
    remote: dict[str, Any] | None,
) -> list[str]:
    if remote is None:
        return ["remote artifact verification not requested"]
    required_kinds = {
        "sqsh",
        "pretrained_model",
        "train_annotation",
        "validation_annotation",
    }
    by_kind = {
        item["kind"]: item
        for item in remote["artifacts"]
        if item["kind"] in required_kinds
    }
    blockers = []
    for kind in sorted(required_kinds):
        item = by_kind.get(kind)
        if item is None or not item["verified"]:
            blockers.append(f"{kind}: remote artifact missing or digest mismatch")
    return blockers


def submit_jobs(
    manifest: dict[str, Any],
    commands: list[tuple[str, dict[str, Any]]],
    *,
    phase: str,
) -> dict[str, Any]:
    from tao_sdk.platforms.slurm import SlurmSDK

    runtime_dir = HERE / "runtime"
    os.environ["SLURM_USE_SQSH"] = "false"
    os.environ["SLURM_PARTITION"] = manifest["hardware_and_runtime"]["partition"]
    os.environ["SLURM_ACCOUNT"] = manifest["hardware_and_runtime"]["account"]
    sdk = SlurmSDK(
        poll_interval=10,
        state_file=runtime_dir / "slurm_state.db",
    )
    submissions = []
    for command, summary in commands:
        job = sdk.create_job(
            image=manifest["hardware_and_runtime"]["sqsh_path"],
            command=command,
            gpu_count=8,
            num_nodes=1,
            partition=manifest["hardware_and_runtime"]["partition"],
            account=manifest["hardware_and_runtime"]["account"],
        )
        identity = sdk._handler.get_job_runtime_identity(job.id)
        submissions.append(
            {
                **summary,
                "tao_job_id": job.id,
                "slurm_job_id": identity.get("slurm_job_id", ""),
            }
        )
        atomic_json(
            runtime_dir / f"{phase}_submissions.json",
            {
                "manifest_id": manifest["manifest_id"],
                "phase": phase,
                "feeds_selection": False,
                "submissions": submissions,
            },
        )
    return {"submissions": submissions}


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest, manifest_digest = load_manifest(manifest_path)
    schedule_checks = validate_schedule(manifest)
    phase1, source_checks = validate_sources(manifest_path, manifest)
    evidence_checks = validate_candidate_evidence(manifest_path, manifest)
    configs = generate_configs(phase1, manifest)
    block_commands = [
        staged_command(manifest_path, manifest, configs, block)
        for block in manifest["schedule"]
    ]
    recovery_configs = generate_recovery_configs(phase1, manifest)
    recovery_job_commands = recovery_commands(manifest, recovery_configs)
    remote = verify_remote_artifacts(manifest) if args.verify_remote else None
    block_blockers = block_submission_blockers(manifest, remote)
    recovery_blockers = recovery_submission_blockers(remote)
    if args.submit_recovery:
        mode = "submit_recovery"
    elif args.submit_blocks:
        mode = "submit_blocks"
    else:
        mode = "dry_run"
    report = {
        "mode": mode,
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_digest,
        "feeds_selection": False,
        "status": "preflight_complete",
        "checkpoint_recovery": {
            "required": True,
            "submission_ready": not recovery_blockers,
            "blockers": recovery_blockers,
            "jobs": [summary for _, summary in recovery_job_commands],
            "postcondition": (
                "Hash recovered checkpoints and create immutable manifest.v2.json; "
                "do not mutate manifest.v1.json."
            ),
        },
        "matched_blocks": {
            "submission_ready": not block_blockers,
            "blockers": block_blockers,
            "allocations": [summary for _, summary in block_commands],
        },
        "schedule_checks": schedule_checks,
        "source_checks": source_checks,
        "candidate_evidence_checks": evidence_checks,
        "remote_artifact_checks": remote,
    }
    if args.report:
        atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)

    if not (args.submit_recovery or args.submit_blocks):
        return 0
    if not args.acknowledge_validation_only:
        raise RuntimeError("submission requires --acknowledge-validation-only")
    if not args.verify_remote:
        raise RuntimeError("submission requires --verify-remote")
    if args.submit_recovery:
        if recovery_blockers:
            raise RuntimeError(
                "recovery submission blocked by immutable preflight: "
                + "; ".join(recovery_blockers)
            )
        submission = submit_jobs(
            manifest,
            recovery_job_commands,
            phase="checkpoint_recovery",
        )
    else:
        if block_blockers:
            raise RuntimeError(
                "matched-block submission blocked by immutable preflight: "
                + "; ".join(block_blockers)
            )
        submission = submit_jobs(
            manifest,
            block_commands,
            phase="matched_latency_blocks",
        )
    print(json.dumps(submission, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
