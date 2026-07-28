#!/usr/bin/env python3

"""Revalidate mAP50 for the four exact-config checkpoint reconstructions.

This workflow is validation-only. It never mutates the frozen Phase-1 replay,
manifest v2, or any AutoML winner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.v2.json"
RUNTIME_DIR = HERE / "runtime"
LEDGER = RUNTIME_DIR / "recovered_accuracy_submissions.json"
STATE_FILE = RUNTIME_DIR / "recovered_accuracy_state.json"
STATUS_REPORT = RUNTIME_DIR / "recovered_accuracy_status.json"
RESULT = HERE / "recovered_accuracy_revalidation.json"
PHASE1_HARNESS = HERE.parent / "dino_moo_review_20260727" / "run_experiment.py"
EXPECTED_MANIFEST_SHA256 = (
    "ccf88ad1a8c95a808bb9e217de50dc296b700e5af6e1dca474d56b967186e0d2"
)
ACKNOWLEDGEMENT = "VALIDATION_ONLY_NO_SELECTION_FEEDBACK"
TERMINAL = {"Complete", "Error", "Canceled"}
MAP50_PATTERNS = (
    re.compile(
        r"(?:Validation|Test)\s+mAP50\s*[:=]\s*"
        r"([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)"
    ),
    re.compile(
        r"\btest_mAP50\b[^0-9+\-]*"
        r"([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)"
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--submit", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--aggregate", action="store_true")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--state-file", type=Path, default=STATE_FILE)
    parser.add_argument("--report", type=Path, default=STATUS_REPORT)
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--acknowledgement", default="")
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if path.resolve() == MANIFEST.resolve() and digest != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"manifest v2 drift: {digest} != {EXPECTED_MANIFEST_SHA256}"
        )
    manifest = json.loads(raw)
    if manifest.get("manifest_id") != "dino_moo_phase2_20260728_v2":
        raise ValueError("unexpected manifest identity")
    if manifest.get("feeds_selection") is not False:
        raise ValueError("manifest must be validation-only")
    return manifest, digest


def load_phase1() -> Any:
    spec = importlib.util.spec_from_file_location(
        "dino_moo_phase1_revalidation",
        PHASE1_HARNESS,
    )
    if spec is None or spec.loader is None:
        raise ImportError(PHASE1_HARNESS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def recovered_candidates(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    expected = manifest["checkpoint_recovery"]["candidate_ids"]
    by_id = {item["candidate_id"]: item for item in manifest["candidates"]}
    candidates = [by_id[candidate_id] for candidate_id in expected]
    if len(candidates) != 4:
        raise RuntimeError("exactly four recovered candidates are required")
    for candidate in candidates:
        if candidate.get("checkpoint_origin") != "exact_config_retrain":
            raise ValueError(
                f"{candidate['candidate_id']} is not an exact-config retrain"
            )
        if not candidate.get("checkpoint_sha256"):
            raise ValueError(f"{candidate['candidate_id']} is not hash-pinned")
    return candidates


def evaluation_commands(
    manifest: dict[str, Any],
    phase1: Any,
) -> list[tuple[str, dict[str, Any]]]:
    from tao_sdk.script_runner import build_entrypoint

    commands = []
    for candidate in recovered_candidates(manifest):
        specs = phase1.evaluation_specs(
            candidate["checkpoint"],
            candidate["num_queries"],
        )
        payload = phase1.yaml.safe_dump(specs, sort_keys=True).encode("utf-8")
        digest = sha256_bytes(payload)
        entrypoint = build_entrypoint(
            command="dino evaluate -e {config_path}",
            specs=specs,
            inputs=phase1.EVALUATE_INPUTS,
            outputs=phase1.EVALUATE_OUTPUTS,
            config_format="yaml",
            upload_excludes=["inputs/"],
        )
        command = entrypoint["command"]
        commands.append(
            (
                command,
                {
                    "candidate_id": candidate["candidate_id"],
                    "checkpoint": candidate["checkpoint"],
                    "checkpoint_sha256": candidate["checkpoint_sha256"],
                    "historical_mAP50": candidate["mAP50"],
                    "evaluation_config_sha256": digest,
                    "latency_config_sha256": candidate["config_sha256"],
                    "command_sha256": sha256_bytes(command.encode("utf-8")),
                    "feeds_selection": False,
                },
            )
        )
    return commands


def create_sdk(manifest: dict[str, Any], state_file: Path) -> Any:
    from tao_sdk.platforms.slurm import SlurmSDK

    runtime = manifest["hardware_and_runtime"]
    os.environ["SLURM_USE_SQSH"] = "false"
    os.environ["SLURM_PARTITION"] = runtime["partition"]
    os.environ["SLURM_ACCOUNT"] = runtime["account"]
    return SlurmSDK(poll_interval=10, state_file=state_file)


def submit(
    manifest: dict[str, Any],
    manifest_digest: str,
    commands: list[tuple[str, dict[str, Any]]],
    ledger_path: Path,
    state_file: Path,
) -> dict[str, Any]:
    if ledger_path.exists():
        raise FileExistsError(
            f"submission ledger already exists; refusing duplicates: {ledger_path}"
        )
    sdk = create_sdk(manifest, state_file)
    runtime = manifest["hardware_and_runtime"]
    submissions = []
    for command, evidence in commands:
        job = sdk.create_job(
            image=runtime["sqsh_path"],
            command=command,
            gpu_count=runtime["gpu_count"],
            num_nodes=runtime["num_nodes"],
            partition=runtime["partition"],
            account=runtime["account"],
        )
        identity = sdk._handler.get_job_runtime_identity(job.id)
        submissions.append(
            {
                **evidence,
                "tao_job_id": job.id,
                "slurm_job_id": str(identity.get("slurm_job_id", "")),
            }
        )
        atomic_json(
            ledger_path,
            {
                "schema_version": 1,
                "purpose": "Recovered checkpoint mAP50 revalidation",
                "manifest_id": manifest["manifest_id"],
                "manifest_sha256": manifest_digest,
                "feeds_selection": False,
                "submissions": submissions,
            },
        )
    return json.loads(ledger_path.read_text())


def load_ledger(
    path: Path,
    manifest: dict[str, Any],
    manifest_digest: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("manifest_sha256") != manifest_digest:
        raise ValueError("ledger references a different manifest")
    if payload.get("feeds_selection") is not False:
        raise ValueError("ledger must be validation-only")
    expected = manifest["checkpoint_recovery"]["candidate_ids"]
    actual = [item["candidate_id"] for item in payload.get("submissions", [])]
    if actual != expected or len(set(actual)) != 4:
        raise ValueError("ledger must contain the four candidates in manifest order")
    if any(
        not str(item.get("slurm_job_id", "")).isdigit()
        for item in payload["submissions"]
    ):
        raise ValueError("ledger contains a missing SLURM job ID")
    return payload


def ssh_target() -> str:
    user = os.environ["SLURM_USER"].strip()
    host = os.environ["SLURM_HOSTNAME"].split(",", 1)[0].strip()
    return f"{user}@{host}"


def slurm_rows(slurm_ids: list[str]) -> dict[str, dict[str, str]]:
    command = ["ssh", "-o", "BatchMode=yes"]
    key_path = os.environ.get("SSH_KEY_PATH")
    if key_path:
        command.extend(["-i", key_path])
    remote = (
        f"sacct -j {','.join(slurm_ids)} "
        "--format=JobIDRaw,State,ExitCode,Elapsed,NodeList "
        "--parsable2 -X -n"
    )
    command.extend([ssh_target(), remote])
    output = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    ).stdout
    rows = {}
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) < 5:
            continue
        job_id = fields[0]
        rows[job_id] = {
            "state": fields[1].split("+", 1)[0].split(None, 1)[0],
            "exit_code": fields[2],
            "elapsed": fields[3],
            "node_list": fields[4],
        }
    return rows


def status(
    manifest: dict[str, Any],
    manifest_digest: str,
    ledger_path: Path,
    state_file: Path,
) -> tuple[dict[str, Any], Any]:
    ledger = load_ledger(ledger_path, manifest, manifest_digest)
    sdk = create_sdk(manifest, state_file)
    rows = slurm_rows(
        [item["slurm_job_id"] for item in ledger["submissions"]]
    )
    jobs = []
    for item in ledger["submissions"]:
        sdk_status = sdk.get_job_status(item["tao_job_id"]).status
        runtime = sdk._handler.get_job_runtime_identity(item["tao_job_id"])
        if str(runtime.get("slurm_job_id", "")) != item["slurm_job_id"]:
            raise RuntimeError(
                f"SDK/ledger SLURM identity mismatch for {item['candidate_id']}"
            )
        accounting = rows.get(item["slurm_job_id"])
        if accounting is None:
            raise RuntimeError(
                f"missing sacct row for {item['candidate_id']}"
            )
        jobs.append(
            {
                **item,
                "sdk_status": sdk_status,
                "slurm_accounting": accounting,
                "complete": (
                    sdk_status == "Complete"
                    and accounting["state"] == "COMPLETED"
                    and accounting["exit_code"] == "0:0"
                ),
            }
        )
    all_complete = all(item["complete"] for item in jobs)
    report = {
        "schema_version": 1,
        "checked_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "feeds_selection": False,
        "all_complete": all_complete,
        "status": "ready_for_aggregation" if all_complete else "pending",
        "jobs": jobs,
    }
    return report, sdk


def read_map50(phase1: Any, sdk: Any, job: dict[str, Any]) -> float:
    value = phase1.read_status_map50(sdk, job["tao_job_id"])
    if value is not None:
        return float(value)
    logs = sdk.get_job_logs(job["tao_job_id"], tail=5000)
    matches = []
    for pattern in MAP50_PATTERNS:
        matches.extend(float(item) for item in pattern.findall(logs))
    if not matches:
        raise RuntimeError(f"no mAP50 emitted for {job['candidate_id']}")
    return matches[-1]


def aggregate(
    report: dict[str, Any],
    sdk: Any,
    phase1: Any,
    manifest_digest: str,
    result_path: Path,
) -> dict[str, Any]:
    if not report["all_complete"]:
        raise RuntimeError("all four evaluation jobs must complete first")
    candidates = []
    for job in report["jobs"]:
        measured = read_map50(phase1, sdk, job)
        historical = float(job["historical_mAP50"])
        candidates.append(
            {
                "candidate_id": job["candidate_id"],
                "historical_mAP50": historical,
                "recovered_checkpoint_mAP50": measured,
                "absolute_delta": measured - historical,
                "relative_delta": (
                    (measured - historical) / historical
                    if historical != 0.0
                    else None
                ),
                "tao_job_id": job["tao_job_id"],
                "slurm_job_id": job["slurm_job_id"],
                "checkpoint": job["checkpoint"],
                "checkpoint_sha256": job["checkpoint_sha256"],
                "evaluation_config_sha256": job[
                    "evaluation_config_sha256"
                ],
                "feeds_selection": False,
            }
        )
    payload = {
        "schema_version": 1,
        "purpose": (
            "Comparable DINO evaluate-action revalidation for exact-config "
            "reconstructions of four deleted historical checkpoints."
        ),
        "manifest_sha256": manifest_digest,
        "historical_selection_unchanged": True,
        "feeds_selection": False,
        "interpretation": (
            "Differences quantify reconstruction variability. These values "
            "do not replace frozen archive measurements or rerank candidates."
        ),
        "candidates": candidates,
    }
    if result_path.exists():
        observed = json.loads(result_path.read_text())
        if observed != payload:
            raise RuntimeError("refusing to overwrite different revalidation evidence")
    else:
        atomic_json(result_path, payload)
    return payload


def main() -> int:
    args = parse_args()
    manifest, manifest_digest = load_manifest(args.manifest.resolve())
    phase1 = load_phase1()
    commands = evaluation_commands(manifest, phase1)

    if args.submit:
        if args.acknowledgement != ACKNOWLEDGEMENT:
            raise RuntimeError(
                f"submission requires --acknowledgement {ACKNOWLEDGEMENT}"
            )
        payload = submit(
            manifest,
            manifest_digest,
            commands,
            args.ledger.resolve(),
            args.state_file.resolve(),
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    report, sdk = status(
        manifest,
        manifest_digest,
        args.ledger.resolve(),
        args.state_file.resolve(),
    )
    atomic_json(args.report.resolve(), report)
    if args.aggregate:
        payload = aggregate(
            report,
            sdk,
            phase1,
            manifest_digest,
            args.result.resolve(),
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
