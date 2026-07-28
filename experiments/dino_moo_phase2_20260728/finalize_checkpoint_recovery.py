#!/usr/bin/env python3

"""Monitor checkpoint recovery and fail-closedly materialize manifest v2.

The tool never mutates ``manifest.v1.json``. It reconciles the four recovery
submissions against the SDK's durable state and SLURM accounting, locates one
epoch-9 checkpoint per successful job, hashes each checkpoint over read-only
SSH, and creates ``manifest.v2.json`` only when every invariant passes.
"""

from __future__ import annotations

import argparse
import copy
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

import yaml


HERE = Path(__file__).resolve().parent
MANIFEST_V1 = HERE / "manifest.v1.json"
MANIFEST_V2 = HERE / "manifest.v2.json"
RUNTIME_DIR = HERE / "runtime"
SUBMISSIONS_PATH = RUNTIME_DIR / "checkpoint_recovery_submissions.json"
# The recovery launcher passes ``slurm_state.db`` to the SDK. The SDK appends
# its own ``.db`` suffix, so ``sdk_db_path`` resolves the durable store as
# ``slurm_state.db.db`` for this preregistered run.
SDK_STATE_PATH = RUNTIME_DIR / "slurm_state.db"
STATUS_REPORT_PATH = RUNTIME_DIR / "checkpoint_recovery_status.json"
EXPECTED_MANIFEST_V1_SHA256 = (
    "bd46e9160566845c71226aedf5ff08032b4eae70d442f32d7085582dfba738ad"
)
SDK_TERMINAL_STATUSES = {"Complete", "Error", "Canceled"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--status",
        action="store_true",
        help="Report all four SDK/SLURM states and available checkpoint hashes.",
    )
    mode.add_argument(
        "--finalize",
        action="store_true",
        help="Create immutable manifest.v2.json when every recovery job passes.",
    )
    parser.add_argument("--manifest-v1", type=Path, default=MANIFEST_V1)
    parser.add_argument("--manifest-v2", type=Path, default=MANIFEST_V2)
    parser.add_argument("--submissions", type=Path, default=SUBMISSIONS_PATH)
    parser.add_argument("--sdk-state", type=Path, default=SDK_STATE_PATH)
    parser.add_argument("--report", type=Path, default=STATUS_REPORT_PATH)
    return parser.parse_args()


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def load_manifest_v1(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != EXPECTED_MANIFEST_V1_SHA256:
        raise RuntimeError(
            "manifest v1 digest mismatch: "
            f"{digest} != {EXPECTED_MANIFEST_V1_SHA256}"
        )
    manifest = json.loads(raw)
    if manifest.get("manifest_id") != "dino_moo_phase2_20260728_v1":
        raise ValueError("unexpected manifest v1 identity")
    if manifest.get("feeds_selection") is not False:
        raise ValueError("manifest v1 must be validation-only")
    return manifest, digest


def load_submissions(
    path: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    payload = json.loads(raw)
    if payload.get("manifest_id") != manifest["manifest_id"]:
        raise ValueError("recovery submissions reference a different manifest")
    if payload.get("phase") != "checkpoint_recovery":
        raise ValueError("submissions file is not the checkpoint-recovery phase")
    if payload.get("feeds_selection") is not False:
        raise ValueError("recovery submissions must set feeds_selection=false")
    submissions = payload.get("submissions")
    if not isinstance(submissions, list):
        raise ValueError("submissions must be a list")

    expected_ids = manifest["checkpoint_recovery"]["candidate_ids"]
    actual_ids = [item.get("candidate_id") for item in submissions]
    if len(submissions) != 4 or set(actual_ids) != set(expected_ids):
        raise ValueError(
            "submissions must contain exactly the four registered recovery "
            f"candidates: expected={expected_ids}, actual={actual_ids}"
        )
    if len(set(actual_ids)) != 4:
        raise ValueError("recovery candidate IDs are duplicated")
    tao_ids = [item.get("tao_job_id") for item in submissions]
    slurm_ids = [str(item.get("slurm_job_id", "")) for item in submissions]
    if any(not isinstance(job_id, str) or not job_id for job_id in tao_ids):
        raise ValueError("every recovery submission needs a TAO job ID")
    if any(not job_id.isdigit() for job_id in slurm_ids):
        raise ValueError("every recovery submission needs a numeric SLURM job ID")
    if len(set(tao_ids)) != 4 or len(set(slurm_ids)) != 4:
        raise ValueError("recovery TAO and SLURM job IDs must be unique")
    if any(item.get("feeds_selection") is not False for item in submissions):
        raise ValueError("every recovery job must set feeds_selection=false")

    by_candidate = {item["candidate_id"]: item for item in submissions}
    payload["submissions"] = [by_candidate[item] for item in expected_ids]
    return payload, digest


def sdk_db_path(state_path: Path) -> Path:
    if state_path.name.endswith(".json"):
        return state_path.with_suffix(".db")
    return Path(str(state_path) + ".db")


def monitor_state_summary(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {"exists": False, "active_job_ids": []}
    payload = json.loads(state_path.read_text())
    active = payload.get("active_jobs", [])
    return {
        "exists": True,
        "active_job_ids": sorted(
            item["id"]
            for item in active
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ),
    }


def ssh_target() -> str:
    user = os.environ.get("SLURM_USER", "").strip()
    host = os.environ.get("SLURM_HOSTNAME", "").split(",", 1)[0].strip()
    if not user or not host:
        raise RuntimeError(
            "SLURM_USER and SLURM_HOSTNAME are required; source "
            "/localhome/local-rarunachalam/.tao/config.env first"
        )
    return f"{user}@{host}"


def remote_output(command: str, *, timeout: int = 900) -> str:
    ssh_command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
    ]
    key_path = os.environ.get("SSH_KEY_PATH")
    if key_path:
        ssh_command.extend(["-i", key_path])
    ssh_command.extend([ssh_target(), command])
    completed = subprocess.run(
        ssh_command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.stdout


def slurm_accounting(slurm_job_id: str) -> dict[str, Any]:
    command = " ".join(
        [
            "sacct",
            "-X",
            "-j",
            shlex.quote(slurm_job_id),
            "--noheader",
            "--parsable2",
            "--format=JobIDRaw,State,ExitCode,NodeList",
        ]
    )
    rows = []
    for line in remote_output(command, timeout=120).splitlines():
        fields = line.strip().split("|")
        if len(fields) < 4 or fields[0] != slurm_job_id:
            continue
        rows.append(
            {
                "job_id_raw": fields[0],
                "state": fields[1],
                "exit_code": fields[2],
                "node_list": fields[3],
            }
        )
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one SLURM allocation accounting row for "
            f"{slurm_job_id}, found {len(rows)}"
        )
    row = rows[0]
    canonical_state = row["state"].split("+", 1)[0].split(None, 1)[0]
    row["canonical_state"] = canonical_state
    row["complete"] = canonical_state == "COMPLETED" and row["exit_code"] == "0:0"
    return row


def local_lustre_path(uri: str) -> str:
    if uri.startswith("lustre://"):
        path = uri.removeprefix("lustre://")
        return path if path.startswith("/") else f"/{path}"
    if uri.startswith("/"):
        return uri
    raise ValueError(f"expected a Lustre result URI, got {uri!r}")


def locate_terminal_checkpoint(
    result_root: str,
    tao_job_id: str,
) -> dict[str, Any]:
    root = Path(result_root)
    if root.name != tao_job_id or root.parent.name != "results":
        raise ValueError(
            f"result root is not scoped to TAO job {tao_job_id}: {result_root}"
        )
    script = "\n".join(
        [
            "import json,re,sys",
            "from pathlib import Path",
            "root=Path(sys.argv[1])",
            "all_paths=sorted(str(p) for p in root.rglob('*') "
            "if p.is_file() and p.suffix.lower() in ('.pth','.ckpt'))",
            "pattern=re.compile(r'^model_epoch_0*9_step_[0-9]+"
            "\\.(?:pth|ckpt)$',re.I)",
            "matches=[p for p in all_paths if pattern.match(Path(p).name)]",
            "print(json.dumps({'all_checkpoints':all_paths,"
            "'terminal_epoch_9':matches}))",
        ]
    )
    output = remote_output(
        f"python3 -c {shlex.quote(script)} {shlex.quote(result_root)}",
        timeout=300,
    )
    payload = json.loads(output)
    matches = payload["terminal_epoch_9"]
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one terminal epoch-9 checkpoint under "
            f"{result_root}, found {len(matches)}; "
            f"all checkpoints={payload['all_checkpoints']}"
        )
    checkpoint = matches[0]
    if not checkpoint.startswith(result_root.rstrip("/") + "/"):
        raise RuntimeError("resolved checkpoint escaped its TAO result root")
    metadata = remote_output(
        " && ".join(
            [
                f"stat -c '%s' {shlex.quote(checkpoint)}",
                f"sha256sum {shlex.quote(checkpoint)}",
            ]
        ),
        timeout=900,
    ).splitlines()
    if len(metadata) != 2:
        raise RuntimeError(f"could not read checkpoint metadata for {checkpoint}")
    size_bytes = int(metadata[0].strip())
    digest, returned_path = metadata[1].split(None, 1)
    if returned_path.strip() != checkpoint:
        raise RuntimeError("sha256sum returned a different checkpoint path")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RuntimeError("checkpoint SHA256 is malformed")
    return {
        "path": checkpoint,
        "sha256": digest,
        "size_bytes": size_bytes,
        "epoch": 9,
    }


def inspect_recovery_jobs(
    manifest: dict[str, Any],
    submissions: dict[str, Any],
    state_path: Path,
) -> dict[str, Any]:
    database_path = sdk_db_path(state_path)
    if not database_path.is_file():
        raise FileNotFoundError(
            f"SDK durable state database does not exist: {database_path}"
        )

    from tao_sdk.platforms.slurm import SlurmSDK

    os.environ["SLURM_USE_SQSH"] = "false"
    os.environ["SLURM_PARTITION"] = manifest["hardware_and_runtime"]["partition"]
    os.environ["SLURM_ACCOUNT"] = manifest["hardware_and_runtime"]["account"]
    sdk = SlurmSDK(poll_interval=10, state_file=state_path)
    jobs = []
    try:
        for submission in submissions["submissions"]:
            record: dict[str, Any] = {
                "candidate_id": submission["candidate_id"],
                "tao_job_id": submission["tao_job_id"],
                "submitted_slurm_job_id": str(submission["slurm_job_id"]),
                "feeds_selection": False,
            }
            try:
                status = sdk.get_job_status(submission["tao_job_id"])
                record["sdk_status"] = status.status
                record["sdk_message"] = status.message
                identity = sdk._handler.get_job_runtime_identity(
                    submission["tao_job_id"]
                )
                actual_slurm_id = str(identity.get("slurm_job_id", ""))
                record["sdk_slurm_job_id"] = actual_slurm_id
                if actual_slurm_id != record["submitted_slurm_job_id"]:
                    raise RuntimeError(
                        "SDK runtime SLURM ID differs from the submission ledger"
                    )
                record["slurm_accounting"] = slurm_accounting(actual_slurm_id)
                results_uri = sdk.get_job_results_dir(submission["tao_job_id"])
                result_root = local_lustre_path(results_uri)
                record["result_root"] = result_root
                sdk_complete = status.status == "Complete"
                slurm_complete = record["slurm_accounting"]["complete"]
                record["complete"] = sdk_complete and slurm_complete
                record["terminal"] = status.status in SDK_TERMINAL_STATUSES
                if record["complete"]:
                    record["checkpoint"] = locate_terminal_checkpoint(
                        result_root,
                        submission["tao_job_id"],
                    )
            except Exception as exc:
                record["query_error"] = f"{type(exc).__name__}: {exc}"
                record["complete"] = False
            jobs.append(record)
    finally:
        sdk._monitor.stop()
        sdk._store.close()

    all_complete = len(jobs) == 4 and all(job.get("complete") for job in jobs)
    any_failed = any(
        job.get("sdk_status") in {"Error", "Canceled"} or "query_error" in job
        for job in jobs
    )
    if all_complete:
        overall_status = "ready_for_manifest_v2"
    elif any_failed:
        overall_status = "failed_or_unverifiable"
    else:
        overall_status = "pending"
    return {
        "overall_status": overall_status,
        "all_complete": all_complete,
        "jobs": jobs,
        "sdk_state": {
            "monitor_state_path": str(state_path),
            "database_path": str(database_path),
            **monitor_state_summary(state_path),
        },
    }


def resolve_source(manifest_path: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def load_phase1_module(manifest_path: Path, manifest: dict[str, Any]):
    source = manifest["source_artifacts"]
    path = resolve_source(manifest_path, source["phase1_run_experiment_path"])
    actual = sha256_file(path)
    expected = source["phase1_run_experiment_sha256"]
    if actual != expected:
        raise RuntimeError(
            f"pinned phase-1 harness drift: {actual} != {expected}"
        )
    name = "dino_moo_phase2_recovery_pinned_phase1"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pinned phase-1 harness: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def eval_config_sha256(
    phase1: Any,
    checkpoint: str,
    num_queries: int,
) -> str:
    specs = phase1.evaluation_specs(checkpoint, num_queries)
    specs["dataset"]["batch_size"] = phase1.LATENCY_BATCH_SIZE_PER_GPU
    specs["evaluate"]["batch_size"] = phase1.LATENCY_BATCH_SIZE_PER_GPU
    payload = yaml.safe_dump(specs, sort_keys=True).encode("utf-8")
    return sha256_bytes(payload)


def git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(HERE.parent.parent), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def build_manifest_v2(
    manifest_v1_path: Path,
    manifest_v1: dict[str, Any],
    manifest_v1_sha256: str,
    submissions_path: Path,
    submissions_sha256: str,
    inspection: dict[str, Any],
    *,
    finalized_at: str,
) -> dict[str, Any]:
    if not inspection["all_complete"]:
        raise RuntimeError("all four recovery jobs must be Complete")
    phase1 = load_phase1_module(manifest_v1_path, manifest_v1)
    v2 = copy.deepcopy(manifest_v1)
    v2["manifest_id"] = "dino_moo_phase2_20260728_v2"
    v2["created_at_utc"] = finalized_at
    v2["feeds_selection"] = False

    jobs = {job["candidate_id"]: job for job in inspection["jobs"]}
    recovered_ids = manifest_v1["checkpoint_recovery"]["candidate_ids"]
    for candidate in v2["candidates"]:
        candidate_id = candidate["candidate_id"]
        if candidate_id in jobs:
            job = jobs[candidate_id]
            checkpoint = job["checkpoint"]
            candidate["checkpoint"] = checkpoint["path"]
            candidate["checkpoint_sha256"] = checkpoint["sha256"]
            candidate["checkpoint_status_at_audit"] = "recovered"
            candidate["checkpoint_origin"] = "exact_config_retrain"
            candidate["recovery_tao_job_id"] = job["tao_job_id"]
            candidate["recovery_slurm_job_id"] = job["sdk_slurm_job_id"]
        candidate["config_sha256"] = eval_config_sha256(
            phase1,
            candidate["checkpoint"],
            candidate["num_queries"],
        )

    provenance_jobs = []
    for candidate_id in recovered_ids:
        job = jobs[candidate_id]
        provenance_jobs.append(
            {
                "candidate_id": candidate_id,
                "tao_job_id": job["tao_job_id"],
                "slurm_job_id": job["sdk_slurm_job_id"],
                "sdk_status": job["sdk_status"],
                "slurm_state": job["slurm_accounting"]["canonical_state"],
                "slurm_exit_code": job["slurm_accounting"]["exit_code"],
                "node_list": job["slurm_accounting"]["node_list"],
                "result_root": job["result_root"],
                "checkpoint": job["checkpoint"],
            }
        )
    v2["checkpoint_recovery"]["required"] = False
    v2["checkpoint_recovery"]["status"] = "complete"
    v2["checkpoint_recovery"]["completed_candidate_ids"] = list(recovered_ids)
    v2["checkpoint_recovery"]["accuracy_revalidation"] = {
        "status": "not_run",
        "metric": "mAP50",
        "feeds_selection": False,
        "note": (
            "Recovery reproduces the registered configurations but cannot prove "
            "byte identity with deleted historical weights."
        ),
    }
    v2["recovery_provenance"] = {
        "finalized_at_utc": finalized_at,
        "source_manifest_id": manifest_v1["manifest_id"],
        "source_manifest_sha256": manifest_v1_sha256,
        "submissions_path": str(submissions_path.resolve()),
        "submissions_sha256": submissions_sha256,
        "sdk_state_database_path": inspection["sdk_state"]["database_path"],
        "feeds_selection": False,
        "jobs": provenance_jobs,
    }
    v2["artifact_audit"] = {
        "checked_at_utc": finalized_at,
        "method": (
            "SDK durable status plus SLURM sacct, followed by read-only SSH "
            "checkpoint discovery, stat, and sha256sum."
        ),
        "result": "ready_for_matched_latency_blocks",
    }
    v2["source_artifacts"]["manifest_v1_path"] = str(
        manifest_v1_path.resolve()
    )
    v2["source_artifacts"]["manifest_v1_sha256"] = manifest_v1_sha256
    v2["source_artifacts"]["checkpoint_recovery_submissions_path"] = str(
        submissions_path.resolve()
    )
    v2["source_artifacts"][
        "checkpoint_recovery_submissions_sha256"
    ] = submissions_sha256
    finalizer_path = Path(__file__).resolve()
    v2["source_artifacts"]["checkpoint_recovery_finalizer_path"] = str(
        finalizer_path
    )
    v2["source_artifacts"][
        "checkpoint_recovery_finalizer_sha256"
    ] = sha256_file(finalizer_path)
    v2["source_commits"]["phase2_checkpoint_recovery_tao_automl"] = git_commit()
    return v2


def create_or_verify_manifest_v2(
    path: Path,
    build,
) -> tuple[str, str]:
    if path.exists():
        existing = json.loads(path.read_text())
        finalized_at = existing.get("created_at_utc")
        if not isinstance(finalized_at, str) or not finalized_at:
            raise RuntimeError("existing manifest v2 has no creation timestamp")
        expected = build(finalized_at)
        if existing != expected:
            raise FileExistsError(
                "manifest v2 already exists with different content; refusing "
                "to overwrite it"
            )
        return "already_exists_identical", sha256_file(path)

    finalized_at = utc_timestamp()
    expected = build(finalized_at)
    payload = (json.dumps(expected, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return "created", sha256_bytes(payload)


def main() -> int:
    args = parse_args()
    manifest_v1_path = args.manifest_v1.resolve()
    submissions_path = args.submissions.resolve()
    state_path = args.sdk_state.resolve()
    manifest_v1, manifest_v1_sha256 = load_manifest_v1(manifest_v1_path)
    submissions, submissions_sha256 = load_submissions(
        submissions_path,
        manifest_v1,
    )
    inspection = inspect_recovery_jobs(
        manifest_v1,
        submissions,
        state_path,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "checked_at_utc": utc_timestamp(),
        "mode": "finalize" if args.finalize else "status",
        "manifest_v1_path": str(manifest_v1_path),
        "manifest_v1_sha256": manifest_v1_sha256,
        "submissions_path": str(submissions_path),
        "submissions_sha256": submissions_sha256,
        "feeds_selection": False,
        **inspection,
    }

    if args.finalize:
        if not inspection["all_complete"]:
            atomic_json(args.report, report)
            print(json.dumps(report, indent=2, sort_keys=True), flush=True)
            raise RuntimeError(
                "manifest v2 creation blocked until all four SDK and SLURM "
                "statuses are Complete and every epoch-9 checkpoint is hashed"
            )

        def builder(finalized_at: str) -> dict[str, Any]:
            return build_manifest_v2(
                manifest_v1_path,
                manifest_v1,
                manifest_v1_sha256,
                submissions_path,
                submissions_sha256,
                inspection,
                finalized_at=finalized_at,
            )

        disposition, manifest_v2_sha256 = create_or_verify_manifest_v2(
            args.manifest_v2.resolve(),
            builder,
        )
        report["manifest_v2"] = {
            "path": str(args.manifest_v2.resolve()),
            "disposition": disposition,
            "sha256": manifest_v2_sha256,
        }
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
