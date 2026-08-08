#!/usr/bin/env python3

"""Fail-closed checkpoint and accuracy workflow for DINO sensitivity jobs.

The frozen one-factor manifest and its 33-job submission report are the only
study-definition inputs. Results can populate immutable evidence artifacts,
but can never change profiles, ranges, or select a winner.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable

import yaml


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RUNTIME_DIR = HERE / "runtime" / "one_factor_sensitivity"
MANIFEST_PATH = HERE / "one_factor_sensitivity_manifest.v1.json"
SUBMISSION_REPORT_PATH = RUNTIME_DIR / "submission_report.json"
TRAINING_SDK_STATE = RUNTIME_DIR / "slurm_state.json"
TRAINING_STATUS_PATH = RUNTIME_DIR / "sensitivity_training_status.json"
CHECKPOINT_ARTIFACT_PATH = HERE / "sensitivity_training_checkpoints.v1.json"
EVALUATION_PLAN_PATH = (
    RUNTIME_DIR / "sensitivity_training_evaluation_plan.json"
)
EVALUATION_SUBMISSIONS_PATH = (
    RUNTIME_DIR / "sensitivity_training_evaluation_submissions.json"
)
EVALUATION_SDK_STATE = (
    RUNTIME_DIR / "sensitivity_training_evaluation_slurm_state.json"
)
EVALUATION_STATUS_PATH = (
    RUNTIME_DIR / "sensitivity_training_evaluation_status.json"
)
ACCURACY_ARTIFACT_PATH = HERE / "sensitivity_training_accuracy.v1.json"
LAUNCHER_PATH = HERE / "one_factor_sensitivity_launcher.py"
SDK_ROOT = Path("/localhome/local-rarunachalam/tao-sdk")

EXPECTED_MANIFEST_SHA256 = (
    "ee65fd9a09d7cacc40f88a0b95b07af3fd0560d8496407447344b310bb5eaa44"
)
EXPECTED_EVALUATE_TEMPLATE_SHA256 = (
    "561227c8fc7380db2fe4ae4922e44a83f7ade16281df589b9d63993488f1130a"
)
EVALUATION_ACKNOWLEDGEMENT = (
    "USER_AUTHORIZED_CONCURRENT_8GPU_SLURM_DINO_SENSITIVITY_EVAL_20260728"
)
SDK_TERMINAL_STATUSES = {"Complete", "Error", "Canceled"}
MAP50_KEYS = ("test_mAP50", "val_mAP50")


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


def immutable_json(path: Path, payload: Any) -> tuple[str, str]:
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if path.exists():
        observed = path.read_bytes()
        if observed != encoded:
            raise FileExistsError(
                f"immutable artifact already exists with different content: {path}"
            )
        return "already_exists_identical", sha256_bytes(observed)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return "created", sha256_bytes(encoded)


def load_env_file(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"required secrets file not found: {path}")
    loaded: list[str] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(
                f"unsupported environment line {line_number}: missing '='"
            )
        key, encoded = line.split("=", 1)
        key = key.strip()
        if (
            not key
            or not key.replace("_", "").isalnum()
            or key[0].isdigit()
        ):
            raise ValueError(f"invalid environment key on line {line_number}")
        tokens = shlex.split(encoded, comments=True, posix=True)
        if len(tokens) > 1:
            raise ValueError(
                f"unsupported environment value syntax on line {line_number}"
            )
        os.environ.setdefault(key, tokens[0] if tokens else "")
        loaded.append(key)
    return sorted(loaded)


def load_launcher():
    name = "dino_sensitivity_frozen_launcher"
    spec = importlib.util.spec_from_file_location(name, LAUNCHER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import sensitivity launcher: {LAUNCHER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_frozen_inputs(
    manifest_path: Path,
    submission_report_path: Path,
) -> dict[str, Any]:
    manifest_digest = sha256_file(manifest_path)
    if manifest_digest != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "frozen sensitivity manifest digest mismatch: "
            f"{manifest_digest} != {EXPECTED_MANIFEST_SHA256}"
        )
    launcher = load_launcher()
    manifest = launcher.load_manifest(manifest_path)
    plan, _commands = launcher.build_plan(manifest)
    report = read_json(submission_report_path)
    report_digest = sha256_file(submission_report_path)

    if report.get("study_id") != manifest["study_id"]:
        raise ValueError("submission report references another study")
    if report.get("manifest_sha256") != manifest_digest:
        raise ValueError("submission report manifest digest mismatch")
    if report.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("submission report plan digest mismatch")
    if report.get("feeds_final_selection") is not False:
        raise ValueError("submission report must set feeds_final_selection=false")
    if report.get("manual_selection_permitted") is not False:
        raise ValueError("submission report cannot permit manual selection")
    if report.get("status") != "training_submitted":
        raise ValueError("submission report must record training_submitted")
    if report.get("entries") != plan["entries"]:
        raise ValueError(
            "submission report entries differ from the regenerated frozen plan"
        )

    entries = plan["entries"]
    training_entries = [item for item in entries if item["training_required"]]
    reuse_entries = [item for item in entries if not item["training_required"]]
    if len(entries) != 42 or len(training_entries) != 33 or len(reuse_entries) != 9:
        raise ValueError("frozen 42/33/9 profile counts drifted")

    submissions = report.get("submissions")
    if not isinstance(submissions, list) or len(submissions) != 33:
        raise ValueError("submission report must contain all 33 training jobs")
    expected_ids = [item["entry_id"] for item in training_entries]
    actual_ids = [item.get("entry_id") for item in submissions]
    if actual_ids != expected_ids or len(set(actual_ids)) != 33:
        raise ValueError(
            "training submissions must exactly preserve frozen plan order"
        )
    by_entry = {item["entry_id"]: item for item in entries}
    tao_ids: list[str] = []
    slurm_ids: list[str] = []
    for submission in submissions:
        entry = by_entry[submission["entry_id"]]
        for key in (
            "profile_id",
            "seed",
            "resolved_model_spec_sha256",
            "resolved_train_spec_sha256",
            "train_command_sha256",
        ):
            if submission.get(key) != entry.get(key):
                raise ValueError(
                    f"training submission {submission['entry_id']} {key} drift"
                )
        if submission.get("feeds_final_selection") is not False:
            raise ValueError("training submission feeds final selection")
        tao_id = submission.get("tao_job_id")
        slurm_id = str(submission.get("slurm_job_id", ""))
        if not isinstance(tao_id, str) or not tao_id:
            raise ValueError("training submission lacks TAO job ID")
        if not slurm_id.isdigit():
            raise ValueError("training submission lacks numeric SLURM job ID")
        tao_ids.append(tao_id)
        slurm_ids.append(slurm_id)
    if len(set(tao_ids)) != 33 or len(set(slurm_ids)) != 33:
        raise ValueError("training TAO or SLURM IDs are duplicated")

    return {
        "manifest": manifest,
        "manifest_path": manifest_path.resolve(),
        "manifest_sha256": manifest_digest,
        "submission_report": report,
        "submission_report_path": submission_report_path.resolve(),
        "submission_report_sha256": report_digest,
        "plan": plan,
        "entries": entries,
        "training_entries": training_entries,
        "reuse_entries": reuse_entries,
        "submissions": submissions,
    }


def ensure_sdk_importable() -> None:
    sdk_path = str(SDK_ROOT)
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)


def sdk_db_path(state_path: Path) -> Path:
    if state_path.name.endswith(".json"):
        return state_path.with_suffix(".db")
    return Path(str(state_path) + ".db")


def configure_slurm(manifest: dict[str, Any]) -> None:
    slurm = manifest["runtime_contract"]["slurm"]
    os.environ["SLURM_USE_SQSH"] = "false"
    os.environ["SLURM_PARTITION"] = slurm["partition"]
    os.environ["SLURM_ACCOUNT"] = slurm["account"]


def ssh_target() -> str:
    user = os.environ.get("SLURM_USER", "").strip()
    host = os.environ.get("SLURM_HOSTNAME", "").split(",", 1)[0].strip()
    if not user or not host:
        raise RuntimeError(
            "SLURM_USER and SLURM_HOSTNAME are required; load config.env"
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


def batch_slurm_accounting(
    slurm_job_ids: Iterable[str],
) -> dict[str, dict[str, Any]]:
    expected = [str(job_id) for job_id in slurm_job_ids]
    if not expected:
        return {}
    if any(not job_id.isdigit() for job_id in expected):
        raise ValueError("SLURM accounting IDs must be numeric")
    command = " ".join([
        "sacct",
        "-X",
        "-j",
        shlex.quote(",".join(expected)),
        "--noheader",
        "--parsable2",
        "--format=JobIDRaw,State,ExitCode,NodeList",
    ])
    rows: dict[str, dict[str, Any]] = {}
    for line in remote_output(command, timeout=300).splitlines():
        fields = line.strip().split("|")
        if len(fields) < 4 or fields[0] not in expected:
            continue
        job_id = fields[0]
        if job_id in rows:
            raise RuntimeError(f"duplicate SLURM allocation row for {job_id}")
        canonical_state = fields[1].split("+", 1)[0].split(None, 1)[0]
        rows[job_id] = {
            "job_id_raw": job_id,
            "state": fields[1],
            "canonical_state": canonical_state,
            "exit_code": fields[2],
            "node_list": fields[3],
            "complete": (
                canonical_state == "COMPLETED" and fields[2] == "0:0"
            ),
        }
    return rows


def local_lustre_path(uri: str) -> str:
    if uri.startswith("lustre://"):
        path = uri.removeprefix("lustre://")
        return path if path.startswith("/") else f"/{path}"
    if uri.startswith("/"):
        return uri
    raise ValueError(f"expected Lustre results URI, got {uri!r}")


def validate_result_root(result_root: str, tao_job_id: str) -> None:
    root = Path(result_root)
    if root.name != tao_job_id or root.parent.name != "results":
        raise ValueError(
            f"result root is not scoped to TAO job {tao_job_id}: {result_root}"
        )


def remote_checkpoint_metadata(
    jobs: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    for job in jobs:
        validate_result_root(job["result_root"], job["tao_job_id"])
    encoded = base64.b64encode(canonical_bytes(jobs)).decode("ascii")
    script = "\n".join([
        "import base64,hashlib,json,re,sys",
        "from pathlib import Path",
        "jobs=json.loads(base64.b64decode(sys.argv[1]).decode())",
        "pattern=re.compile(r'^model_epoch_0*9_step_[0-9]+"
        r"\.(?:pth|ckpt)$',re.I)",
        "out=[]",
        "for job in jobs:",
        " rec={'entry_id':job['entry_id']}",
        " try:",
        "  root=Path(job['result_root'])",
        "  all_paths=sorted(str(p) for p in root.rglob('*') "
        "if p.is_file() and p.suffix.lower() in ('.pth','.ckpt'))",
        "  matches=[p for p in all_paths if pattern.match(Path(p).name)]",
        "  rec['all_checkpoints']=all_paths",
        "  rec['terminal_epoch_9']=matches",
        "  if len(matches)==1:",
        "   p=Path(matches[0]); h=hashlib.sha256()",
        "   with p.open('rb') as stream:",
        "    for block in iter(lambda:stream.read(1024*1024),b''):"
        " h.update(block)",
        "   rec['checkpoint']={'path':str(p),'sha256':h.hexdigest(),"
        "'size_bytes':p.stat().st_size,'epoch':9}",
        " except Exception as exc:",
        "  rec['error']=type(exc).__name__+': '+str(exc)",
        " out.append(rec)",
        "print(json.dumps(out,sort_keys=True))",
    ])
    output = remote_output(
        f"python3 -c {shlex.quote(script)} {shlex.quote(encoded)}",
        timeout=3600,
    )
    records = json.loads(output)
    if not isinstance(records, list) or len(records) != len(jobs):
        raise RuntimeError("remote checkpoint inspection returned wrong count")
    return {record["entry_id"]: record for record in records}


def inspect_jobs(
    study: dict[str, Any],
    submissions: list[dict[str, Any]],
    state_path: Path,
    *,
    parse_accuracy: bool,
) -> dict[str, Any]:
    database_path = sdk_db_path(state_path)
    if not database_path.is_file():
        raise FileNotFoundError(
            f"SDK durable state database does not exist: {database_path}"
        )
    ensure_sdk_importable()
    configure_slurm(study["manifest"])
    from tao_sdk.platforms.slurm import SlurmSDK

    slurm_rows = batch_slurm_accounting(
        str(item["slurm_job_id"]) for item in submissions
    )
    sdk = SlurmSDK(poll_interval=10, state_file=state_path)
    jobs: list[dict[str, Any]] = []
    try:
        for submission in submissions:
            tao_job_id = submission["tao_job_id"]
            submitted_slurm_id = str(submission["slurm_job_id"])
            record: dict[str, Any] = {
                "entry_id": submission["entry_id"],
                "profile_id": submission["profile_id"],
                "seed": submission["seed"],
                "tao_job_id": tao_job_id,
                "submitted_slurm_job_id": submitted_slurm_id,
                "feeds_final_selection": False,
            }
            try:
                status = sdk.get_job_status(tao_job_id)
                record["sdk_status"] = status.status
                record["sdk_message"] = status.message
                identity = sdk._handler.get_job_runtime_identity(tao_job_id)
                sdk_slurm_id = str(identity.get("slurm_job_id", ""))
                record["sdk_slurm_job_id"] = sdk_slurm_id
                if sdk_slurm_id != submitted_slurm_id:
                    raise RuntimeError(
                        "SDK SLURM ID differs from submission ledger"
                    )
                accounting = slurm_rows.get(sdk_slurm_id)
                if accounting is None:
                    raise RuntimeError("SLURM allocation row is missing")
                record["slurm_accounting"] = accounting
                result_root = local_lustre_path(
                    sdk.get_job_results_dir(tao_job_id)
                )
                validate_result_root(result_root, tao_job_id)
                record["result_root"] = result_root
                record["terminal"] = status.status in SDK_TERMINAL_STATUSES
                record["complete"] = (
                    status.status == "Complete" and accounting["complete"]
                )
            except Exception as exc:
                record["query_error"] = f"{type(exc).__name__}: {exc}"
                record["complete"] = False
            jobs.append(record)
    finally:
        sdk._monitor.stop()
        sdk._store.close()

    complete_jobs = [item for item in jobs if item.get("complete")]
    if parse_accuracy:
        metrics = remote_terminal_map50([
            {
                "entry_id": item["entry_id"],
                "tao_job_id": item["tao_job_id"],
                "result_root": item["result_root"],
            }
            for item in complete_jobs
        ])
        for item in complete_jobs:
            item["accuracy"] = metrics[item["entry_id"]]
    else:
        checkpoints = remote_checkpoint_metadata([
            {
                "entry_id": item["entry_id"],
                "tao_job_id": item["tao_job_id"],
                "result_root": item["result_root"],
            }
            for item in complete_jobs
        ])
        for item in complete_jobs:
            metadata = checkpoints[item["entry_id"]]
            matches = metadata.get("terminal_epoch_9", [])
            item["checkpoint_discovery"] = {
                "all_checkpoint_count": len(
                    metadata.get("all_checkpoints", [])
                ),
                "terminal_epoch_9_count": len(matches),
            }
            if metadata.get("error"):
                item["checkpoint_error"] = metadata["error"]
            elif len(matches) != 1 or "checkpoint" not in metadata:
                item["checkpoint_error"] = (
                    "expected exactly one epoch-9 checkpoint; "
                    f"observed {len(matches)}"
                )
            else:
                checkpoint = metadata["checkpoint"]
                if not checkpoint["path"].startswith(
                    item["result_root"].rstrip("/") + "/"
                ):
                    item["checkpoint_error"] = (
                        "checkpoint escaped TAO result root"
                    )
                elif not re.fullmatch(
                    r"[0-9a-f]{64}",
                    checkpoint["sha256"],
                ):
                    item["checkpoint_error"] = "checkpoint SHA256 malformed"
                else:
                    item["checkpoint"] = checkpoint

    expected_count = len(submissions)
    all_complete = (
        len(jobs) == expected_count
        and all(item.get("complete") for item in jobs)
    )
    if parse_accuracy:
        evidence_complete = all_complete and all(
            item.get("accuracy", {}).get("valid") is True for item in jobs
        )
    else:
        evidence_complete = all_complete and all(
            "checkpoint" in item and "checkpoint_error" not in item
            for item in jobs
        )
    any_failed = any(
        item.get("sdk_status") in {"Error", "Canceled"}
        or "query_error" in item
        or "checkpoint_error" in item
        or (
            parse_accuracy
            and item.get("complete")
            and item.get("accuracy", {}).get("valid") is not True
        )
        for item in jobs
    )
    if evidence_complete:
        overall_status = (
            "ready_for_accuracy_artifact"
            if parse_accuracy
            else "ready_for_checkpoint_artifact"
        )
    elif any_failed:
        overall_status = "failed_or_unverifiable"
    else:
        overall_status = "pending"
    return {
        "overall_status": overall_status,
        "expected_job_count": expected_count,
        "observed_job_count": len(jobs),
        "sdk_slurm_all_complete": all_complete,
        "evidence_complete": evidence_complete,
        "complete_job_count": sum(
            bool(item.get("complete")) for item in jobs
        ),
        "jobs": jobs,
        "sdk_state": {
            "state_path": str(state_path.resolve()),
            "database_path": str(database_path.resolve()),
        },
    }


def checkpoint_artifact_payload(
    study: dict[str, Any],
    inspection: dict[str, Any],
    *,
    created_at: str,
    workflow_script_sha256: str | None = None,
) -> dict[str, Any]:
    if (
        len(inspection["jobs"]) != 33
        or not inspection["sdk_slurm_all_complete"]
        or not inspection["evidence_complete"]
    ):
        raise RuntimeError(
            "checkpoint artifact requires all 33 SDK Complete / SLURM "
            "COMPLETED 0:0 jobs and one hashed epoch-9 checkpoint each"
        )
    entry_by_id = {
        item["entry_id"]: item for item in study["training_entries"]
    }
    jobs_by_id = {item["entry_id"]: item for item in inspection["jobs"]}
    entries = []
    for entry_id in [item["entry_id"] for item in study["training_entries"]]:
        entry = entry_by_id[entry_id]
        job = jobs_by_id[entry_id]
        entries.append({
            "entry_id": entry_id,
            "profile_id": entry["profile_id"],
            "seed": entry["seed"],
            "resolved_model_spec_sha256": (
                entry["resolved_model_spec_sha256"]
            ),
            "resolved_train_spec_sha256": (
                entry["resolved_train_spec_sha256"]
            ),
            "train_command_sha256": entry["train_command_sha256"],
            "tao_job_id": job["tao_job_id"],
            "slurm_job_id": job["sdk_slurm_job_id"],
            "sdk_status": job["sdk_status"],
            "slurm_state": (
                job["slurm_accounting"]["canonical_state"]
            ),
            "slurm_exit_code": job["slurm_accounting"]["exit_code"],
            "node_list": job["slurm_accounting"]["node_list"],
            "result_root": job["result_root"],
            "checkpoint": job["checkpoint"],
            "feeds_final_selection": False,
        })
    return {
        "schema_version": 1,
        "artifact_id": "dino_sensitivity_training_checkpoints_20260728_v1",
        "created_at_utc": created_at,
        "status": "complete",
        "study_id": study["manifest"]["study_id"],
        "feeds_final_selection": False,
        "manual_selection_permitted": False,
        "profiles_or_ranges_changed": False,
        "winner_selected": False,
        "source": {
            "manifest_path": str(study["manifest_path"]),
            "manifest_sha256": study["manifest_sha256"],
            "submission_report_path": str(
                study["submission_report_path"]
            ),
            "submission_report_sha256": (
                study["submission_report_sha256"]
            ),
            "frozen_plan_sha256": study["plan"]["plan_sha256"],
            "workflow_script_sha256": (
                workflow_script_sha256
                or sha256_file(Path(__file__).resolve())
            ),
        },
        "completion_contract": {
            "expected_entries": 33,
            "sdk_status": "Complete",
            "slurm_state": "COMPLETED",
            "slurm_exit_code": "0:0",
            "checkpoint_rule": (
                "Exactly one model_epoch_009_step_*.pth or .ckpt under "
                "each TAO result root, hashed with SHA256."
            ),
        },
        "entries": entries,
    }


def load_checkpoint_artifact(
    path: Path,
    study: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    artifact = read_json(path)
    digest = sha256_file(path)
    if (
        artifact.get("artifact_id")
        != "dino_sensitivity_training_checkpoints_20260728_v1"
        or artifact.get("status") != "complete"
        or artifact.get("feeds_final_selection") is not False
        or artifact.get("winner_selected") is not False
    ):
        raise ValueError("checkpoint artifact identity or policy is invalid")
    source = artifact.get("source", {})
    if (
        source.get("manifest_sha256") != study["manifest_sha256"]
        or source.get("submission_report_sha256")
        != study["submission_report_sha256"]
        or source.get("frozen_plan_sha256")
        != study["plan"]["plan_sha256"]
    ):
        raise ValueError("checkpoint artifact source provenance drift")
    if not re.fullmatch(
        r"[0-9a-f]{64}",
        str(source.get("workflow_script_sha256", "")),
    ):
        raise ValueError("checkpoint artifact workflow digest malformed")
    expected_ids = {
        item["entry_id"] for item in study["training_entries"]
    }
    entries = artifact.get("entries", [])
    actual_ids = [item.get("entry_id") for item in entries]
    if (
        len(entries) != 33
        or set(actual_ids) != expected_ids
        or len(set(actual_ids)) != 33
    ):
        raise ValueError("checkpoint artifact must contain all 33 entries")
    for item in entries:
        checkpoint = item.get("checkpoint", {})
        if (
            item.get("feeds_final_selection") is not False
            or checkpoint.get("epoch") != 9
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(checkpoint.get("sha256", "")),
            )
        ):
            raise ValueError("checkpoint artifact entry is malformed")
    return artifact, digest


def evaluation_skill(
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], Path, str]:
    skill = manifest["source_contract"]["dino_train_skill"]
    skill_info_path = Path(skill["skill_info_path"])
    if sha256_file(skill_info_path) != skill["skill_info_sha256"]:
        raise RuntimeError("pinned DINO skill_info.yaml drift")
    skill_info = yaml.safe_load(skill_info_path.read_text(encoding="utf-8"))
    action = skill_info["actions"]["evaluate"]
    if (
        action["command"] != "dino evaluate -e {config_path}"
        or action["config_format"] != "yaml"
    ):
        raise RuntimeError("DINO evaluate action contract drift")
    template_path = skill_info_path.with_name("spec_template_evaluate.yaml")
    template_digest = sha256_file(template_path)
    if template_digest != EXPECTED_EVALUATE_TEMPLATE_SHA256:
        raise RuntimeError("pinned DINO evaluate template drift")
    return action, template_path, template_digest


def evaluation_spec(
    manifest: dict[str, Any],
    template: dict[str, Any],
    entry: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    data = manifest["dataset_contract"]
    constants = manifest["controlled_constants"]
    spec = copy.deepcopy(template)
    spec["model"] = copy.deepcopy(entry["resolved_model_spec"])
    spec["wandb"]["enable"] = False
    spec["dataset"]["test_data_sources"] = {
        "image_dir": data["validation_image_dir"],
        "json_file": data["validation_annotation"],
    }
    spec["dataset"]["num_classes"] = data["num_classes"]
    spec["dataset"]["eval_class_ids"] = copy.deepcopy(
        data["eval_class_ids"]
    )
    spec["dataset"]["batch_size"] = constants[
        "train_batch_size_per_gpu"
    ]
    spec["dataset"]["workers"] = 8
    spec["dataset"]["augmentation"]["test_random_resize"] = 800
    spec["dataset"]["augmentation"]["random_resize_max_size"] = 1333
    spec["dataset"]["augmentation"]["fixed_padding"] = True
    spec["evaluate"]["batch_size"] = constants[
        "train_batch_size_per_gpu"
    ]
    spec["evaluate"]["num_gpus"] = 8
    spec["evaluate"]["gpu_ids"] = list(range(8))
    spec["evaluate"]["num_nodes"] = 1
    spec["evaluate"]["checkpoint"] = checkpoint["path"]
    return spec


def build_evaluation_plan(
    study: dict[str, Any],
    checkpoint_artifact: dict[str, Any],
    checkpoint_artifact_sha256: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    action, template_path, template_digest = evaluation_skill(
        study["manifest"]
    )
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    checkpoint_by_id = {
        item["entry_id"]: item for item in checkpoint_artifact["entries"]
    }
    ensure_sdk_importable()
    from tao_sdk.script_runner import build_entrypoint

    plan_entries: list[dict[str, Any]] = []
    commands: dict[str, str] = {}
    for entry in study["entries"]:
        checkpoint_source_id = (
            entry["entry_id"]
            if entry["training_required"]
            else entry["checkpoint_source_entry_id"]
        )
        source = checkpoint_by_id.get(checkpoint_source_id)
        if source is None:
            raise ValueError(
                f"checkpoint source missing for {entry['entry_id']}: "
                f"{checkpoint_source_id}"
            )
        if int(source["seed"]) != int(entry["seed"]):
            raise ValueError(
                f"cross-seed checkpoint reuse prohibited for {entry['entry_id']}"
            )
        checkpoint = source["checkpoint"]
        if (
            entry["training_required"]
            and checkpoint_source_id != entry["entry_id"]
        ):
            raise ValueError("trained profile must use its own checkpoint")
        if (
            not entry["training_required"]
            and not checkpoint_source_id.startswith("reference__seed_")
        ):
            raise ValueError(
                "postprocess-only profile must use same-seed reference checkpoint"
            )
        if (
            sha256_value(entry["resolved_model_spec"])
            != entry["resolved_model_spec_sha256"]
        ):
            raise ValueError("resolved model spec digest drift")

        spec = evaluation_spec(
            study["manifest"],
            template,
            entry,
            checkpoint,
        )
        if spec["model"] != entry["resolved_model_spec"]:
            raise RuntimeError("evaluation did not preserve full model spec")
        entrypoint = build_entrypoint(
            command=action["command"],
            specs=spec,
            inputs=action["inputs"],
            outputs=action["outputs"],
            config_format=action["config_format"],
            upload_excludes=action["upload_excludes"],
        )
        command = entrypoint["command"]
        commands[entry["entry_id"]] = command
        plan_entries.append({
            "entry_id": entry["entry_id"],
            "profile_id": entry["profile_id"],
            "seed": entry["seed"],
            "axis": entry["axis"],
            "level": entry["level"],
            "execution": entry["execution"],
            "training_required": entry["training_required"],
            "checkpoint_source_entry_id": checkpoint_source_id,
            "checkpoint_source_kind": (
                "trained_profile"
                if entry["training_required"]
                else "same_seed_reference_reuse"
            ),
            "checkpoint": copy.deepcopy(checkpoint),
            "resolved_model_spec": copy.deepcopy(
                entry["resolved_model_spec"]
            ),
            "resolved_model_spec_sha256": (
                entry["resolved_model_spec_sha256"]
            ),
            "resolved_train_spec_sha256": (
                entry["resolved_train_spec_sha256"]
            ),
            "evaluation_spec": spec,
            "evaluation_spec_sha256": sha256_value(spec),
            "evaluation_command_sha256": sha256_bytes(
                command.encode("utf-8")
            ),
            "evaluation_command_bytes": len(command.encode("utf-8")),
            "metric": "mAP50",
            "feeds_final_selection": False,
        })
    if len(plan_entries) != 42:
        raise RuntimeError("evaluation plan must contain exactly 42 entries")
    plan = {
        "schema_version": 1,
        "study_id": study["manifest"]["study_id"],
        "status": "validated_not_submitted",
        "feeds_final_selection": False,
        "manual_selection_permitted": False,
        "profiles_or_ranges_changed": False,
        "winner_selected": False,
        "source": {
            "manifest_sha256": study["manifest_sha256"],
            "submission_report_sha256": (
                study["submission_report_sha256"]
            ),
            "frozen_training_plan_sha256": (
                study["plan"]["plan_sha256"]
            ),
            "checkpoint_artifact_sha256": checkpoint_artifact_sha256,
            "evaluate_template_path": str(template_path),
            "evaluate_template_sha256": template_digest,
            "skill_info_sha256": (
                study["manifest"]["source_contract"]["dino_train_skill"][
                    "skill_info_sha256"
                ]
            ),
        },
        "contract": {
            "entry_count": 42,
            "trained_checkpoint_count": 33,
            "same_seed_reference_reuse_count": 9,
            "model_spec": "full frozen resolved_model_spec",
            "dataset_split": "frozen validation",
            "metric": "mAP50",
            "gpu_count": 8,
            "num_nodes": 1,
            "sqsh_path": (
                study["manifest"]["runtime_contract"]["sqsh_path"]
            ),
        },
        "entries": plan_entries,
    }
    plan["plan_sha256"] = sha256_value(plan)
    return plan, commands


def synthetic_checkpoint_artifact(study: dict[str, Any]) -> dict[str, Any]:
    return {
        "entries": [
            {
                "entry_id": entry["entry_id"],
                "profile_id": entry["profile_id"],
                "seed": entry["seed"],
                "checkpoint": {
                    "path": (
                        "/synthetic/sensitivity/"
                        f"{entry['entry_id']}/model_epoch_009_step_00001.pth"
                    ),
                    "sha256": hashlib.sha256(
                        entry["entry_id"].encode("utf-8")
                    ).hexdigest(),
                    "size_bytes": 1,
                    "epoch": 9,
                },
                "feeds_final_selection": False,
            }
            for entry in study["training_entries"]
        ]
    }


def validate_workflow(study: dict[str, Any]) -> dict[str, Any]:
    synthetic = synthetic_checkpoint_artifact(study)
    synthetic_checkpoint_digest = sha256_value(synthetic)
    plan, commands = build_evaluation_plan(
        study,
        synthetic,
        synthetic_checkpoint_digest,
    )
    by_id = {item["entry_id"]: item for item in study["entries"]}
    if any(
        item["resolved_model_spec"]
        != by_id[item["entry_id"]]["resolved_model_spec"]
        for item in plan["entries"]
    ):
        raise RuntimeError("self-test detected model-spec truncation")
    if any(item["feeds_final_selection"] is not False for item in plan["entries"]):
        raise RuntimeError("self-test detected selection feed")
    reuse = [
        item for item in plan["entries"]
        if item["checkpoint_source_kind"] == "same_seed_reference_reuse"
    ]
    if len(commands) != 42 or len(reuse) != 9:
        raise RuntimeError("self-test evaluation counts drifted")
    reference_map50 = {
        1234: 0.60,
        271828: 0.61,
        314159: 0.62,
    }
    synthetic_jobs = []
    for index, entry in enumerate(plan["entries"]):
        reference_value = reference_map50[int(entry["seed"])]
        value = (
            reference_value
            if entry["profile_id"] == "reference"
            else reference_value * (0.99 if index % 2 == 0 else 0.97)
        )
        synthetic_jobs.append({
            "entry_id": entry["entry_id"],
            "profile_id": entry["profile_id"],
            "seed": entry["seed"],
            "tao_job_id": f"synthetic-tao-{index}",
            "sdk_slurm_job_id": str(900000 + index),
            "sdk_status": "Complete",
            "slurm_accounting": {
                "canonical_state": "COMPLETED",
                "exit_code": "0:0",
                "node_list": f"synthetic-node-{index % 3}",
            },
            "result_root": f"/synthetic/results/synthetic-tao-{index}",
            "accuracy": {
                "mAP50": value,
                "valid": True,
                "terminal_records": [{
                    "line_number": 1,
                    "mAP50": value,
                    "status": "RUNNING",
                    "message": "Evaluate finished successfully.",
                }],
            },
        })
    synthetic_inspection = {
        "jobs": synthetic_jobs,
        "sdk_slurm_all_complete": True,
        "evidence_complete": True,
    }
    accuracy = accuracy_artifact_payload(
        study,
        synthetic_checkpoint_digest,
        plan,
        "synthetic-evaluation-submissions-sha256",
        synthetic_inspection,
        created_at="synthetic-self-test",
    )
    if (
        len(accuracy["entries"]) != 42
        or len(accuracy["profile_accuracy_summary"]) != 14
        or accuracy["winner_selected"] is not False
        or accuracy["selection"]["performed"] is not False
        or any(
            item["feeds_final_selection"] is not False
            for item in accuracy["entries"]
        )
    ):
        raise RuntimeError("self-test accuracy aggregation contract failed")
    return {
        "status": "validated",
        "study_id": study["manifest"]["study_id"],
        "manifest_sha256": study["manifest_sha256"],
        "submission_report_sha256": study["submission_report_sha256"],
        "frozen_plan_sha256": study["plan"]["plan_sha256"],
        "training_submission_count": len(study["submissions"]),
        "evaluation_entry_count": len(plan["entries"]),
        "trained_checkpoint_evaluations": 33,
        "same_seed_reference_reuse_evaluations": len(reuse),
        "all_full_model_specs_preserved": True,
        "all_feeds_final_selection_false": True,
        "accuracy_aggregation_entry_count": len(accuracy["entries"]),
        "accuracy_profile_summary_count": len(
            accuracy["profile_accuracy_summary"]
        ),
        "accuracy_retention_flags_exercised": True,
        "accuracy_aggregation_selects_no_winner": True,
        "synthetic_evaluation_plan_sha256": plan["plan_sha256"],
    }


def remote_terminal_map50(
    jobs: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    for job in jobs:
        validate_result_root(job["result_root"], job["tao_job_id"])
    encoded = base64.b64encode(canonical_bytes(jobs)).decode("ascii")
    script = "\n".join([
        "import base64,json,math,sys",
        "from pathlib import Path",
        "jobs=json.loads(base64.b64decode(sys.argv[1]).decode())",
        "out=[]",
        "for job in jobs:",
        " rec={'entry_id':job['entry_id']}; "
        "p=Path(job['result_root'])/'results_dir/evaluate/status.json'",
        " rec['status_path']=str(p); terminal=[]",
        " try:",
        "  for line_number,line in enumerate(p.read_text().splitlines(),1):",
        "   try: row=json.loads(line)",
        "   except json.JSONDecodeError: continue",
        "   msg=str(row.get('message','')).lower(); kpi=row.get('kpi')",
        "   if 'evaluate finished successfully' not in msg "
        "or not isinstance(kpi,dict): continue",
        "   raw=kpi.get('test_mAP50',kpi.get('val_mAP50'))",
        "   if raw is None: continue",
        "   value=float(raw)",
        "   terminal.append({'line_number':line_number,'mAP50':value,"
        "'status':row.get('status'),'message':row.get('message')})",
        "  rec['terminal_records']=terminal",
        "  if len(terminal)==1 and math.isfinite(terminal[0]['mAP50']) "
        "and 0.0<=terminal[0]['mAP50']<=1.0:",
        "   rec['mAP50']=terminal[0]['mAP50']; rec['valid']=True",
        "  else: rec['valid']=False; rec['error']='expected exactly one "
        "finite terminal mAP50 in [0,1]'",
        " except Exception as exc:",
        "  rec['valid']=False; rec['error']=type(exc).__name__+': '+str(exc)",
        " out.append(rec)",
        "print(json.dumps(out,sort_keys=True))",
    ])
    output = remote_output(
        f"python3 -c {shlex.quote(script)} {shlex.quote(encoded)}",
        timeout=900,
    )
    records = json.loads(output)
    if not isinstance(records, list) or len(records) != len(jobs):
        raise RuntimeError("terminal mAP50 parser returned wrong record count")
    return {item["entry_id"]: item for item in records}


def verify_evaluation_remote_contract(
    study: dict[str, Any],
    checkpoint_artifact: dict[str, Any],
) -> dict[str, Any]:
    manifest = study["manifest"]
    runtime = manifest["runtime_contract"]
    data = manifest["dataset_contract"]
    files = [
        {
            "kind": "sqsh",
            "path": runtime["sqsh_path"],
            "sha256": runtime["sqsh_sha256"],
        },
        {
            "kind": "validation_annotation",
            "path": data["validation_annotation"],
            "sha256": data["validation_annotation_sha256"],
        },
    ]
    files.extend({
        "kind": "checkpoint",
        "entry_id": item["entry_id"],
        "path": item["checkpoint"]["path"],
        "sha256": item["checkpoint"]["sha256"],
    } for item in checkpoint_artifact["entries"])
    encoded = base64.b64encode(canonical_bytes(files)).decode("ascii")
    script = "\n".join([
        "import base64,hashlib,json,sys",
        "from pathlib import Path",
        "items=json.loads(base64.b64decode(sys.argv[1]).decode()); out=[]",
        "for item in items:",
        " rec=dict(item); p=Path(item['path'])",
        " if not p.is_file(): rec['verified']=False; rec['error']='missing'",
        " else:",
        "  h=hashlib.sha256()",
        "  with p.open('rb') as stream:",
        "   for block in iter(lambda:stream.read(1024*1024),b''):"
        " h.update(block)",
        "  rec['observed_sha256']=h.hexdigest(); "
        "rec['verified']=h.hexdigest()==item['sha256']",
        " out.append(rec)",
        "print(json.dumps(out,sort_keys=True))",
    ])
    checks = json.loads(remote_output(
        f"python3 -c {shlex.quote(script)} {shlex.quote(encoded)}",
        timeout=3600,
    ))
    image_dir = data["validation_image_dir"]
    directory_present = (
        remote_output(
            f"test -d {shlex.quote(image_dir)} && echo PRESENT || echo MISSING",
            timeout=120,
        ).strip()
        == "PRESENT"
    )
    if (
        len(checks) != len(files)
        or not all(item.get("verified") for item in checks)
        or not directory_present
    ):
        raise RuntimeError("evaluation remote artifact verification failed")
    return {
        "all_verified": True,
        "files": checks,
        "validation_image_dir": {
            "path": image_dir,
            "present": True,
        },
    }


def submit_evaluations(
    study: dict[str, Any],
    plan: dict[str, Any],
    commands: dict[str, str],
    submissions_path: Path,
    state_path: Path,
    loaded_secret_keys: list[str],
    remote_checks: dict[str, Any],
) -> dict[str, Any]:
    if submissions_path.exists():
        raise FileExistsError(
            "evaluation submission ledger already exists; refusing to create "
            "duplicates"
        )
    ensure_sdk_importable()
    configure_slurm(study["manifest"])
    from tao_sdk.platforms.slurm import SlurmSDK

    runtime = study["manifest"]["runtime_contract"]
    slurm = runtime["slurm"]
    sdk = SlurmSDK(poll_interval=10, state_file=state_path)
    submissions: list[dict[str, Any]] = []

    def ledger(status: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "study_id": study["manifest"]["study_id"],
            "status": status,
            "feeds_final_selection": False,
            "manual_selection_permitted": False,
            "profiles_or_ranges_changed": False,
            "winner_selected": False,
            "evaluation_plan_sha256": plan["plan_sha256"],
            "checkpoint_artifact_sha256": (
                plan["source"]["checkpoint_artifact_sha256"]
            ),
            "loaded_secret_keys": loaded_secret_keys,
            "secret_values_recorded": False,
            "remote_checks": remote_checks,
            "expected_submission_count": 42,
            "submissions": submissions,
        }

    try:
        atomic_json(submissions_path, ledger("submitting"))
        for entry in plan["entries"]:
            job = sdk.create_job(
                image=runtime["sqsh_path"],
                command=commands[entry["entry_id"]],
                gpu_count=slurm["gpu_count_per_node"],
                num_nodes=slurm["num_nodes"],
                partition=slurm["partition"],
                account=slurm["account"],
            )
            identity = sdk._handler.get_job_runtime_identity(job.id)
            submissions.append({
                "entry_id": entry["entry_id"],
                "profile_id": entry["profile_id"],
                "seed": entry["seed"],
                "checkpoint_source_entry_id": (
                    entry["checkpoint_source_entry_id"]
                ),
                "checkpoint_sha256": entry["checkpoint"]["sha256"],
                "resolved_model_spec_sha256": (
                    entry["resolved_model_spec_sha256"]
                ),
                "evaluation_spec_sha256": (
                    entry["evaluation_spec_sha256"]
                ),
                "evaluation_command_sha256": (
                    entry["evaluation_command_sha256"]
                ),
                "tao_job_id": job.id,
                "slurm_job_id": str(identity.get("slurm_job_id", "")),
                "feeds_final_selection": False,
            })
            atomic_json(submissions_path, ledger("submitting"))
        result = ledger("submitted")
        atomic_json(submissions_path, result)
        return result
    finally:
        sdk._monitor.stop()
        sdk._store.close()


def load_evaluation_submissions(
    path: Path,
    plan: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    ledger = read_json(path)
    digest = sha256_file(path)
    if (
        ledger.get("study_id") != plan["study_id"]
        or ledger.get("evaluation_plan_sha256") != plan["plan_sha256"]
        or ledger.get("feeds_final_selection") is not False
        or ledger.get("winner_selected") is not False
    ):
        raise ValueError("evaluation submission ledger provenance drift")
    submissions = ledger.get("submissions")
    if not isinstance(submissions, list):
        raise ValueError("evaluation submissions must be a list")
    expected = {item["entry_id"]: item for item in plan["entries"]}
    seen: set[str] = set()
    tao_ids: set[str] = set()
    slurm_ids: set[str] = set()
    for item in submissions:
        entry_id = item.get("entry_id")
        if entry_id not in expected or entry_id in seen:
            raise ValueError("evaluation submission entry ID is invalid")
        entry = expected[entry_id]
        for key in (
            "profile_id",
            "seed",
            "checkpoint_source_entry_id",
            "resolved_model_spec_sha256",
            "evaluation_spec_sha256",
            "evaluation_command_sha256",
        ):
            if item.get(key) != entry.get(key):
                raise ValueError(f"evaluation submission {entry_id} {key} drift")
        if item.get("feeds_final_selection") is not False:
            raise ValueError("evaluation submission feeds final selection")
        tao_id = item.get("tao_job_id")
        slurm_id = str(item.get("slurm_job_id", ""))
        if not isinstance(tao_id, str) or not tao_id or not slurm_id.isdigit():
            raise ValueError("evaluation submission job identity malformed")
        if tao_id in tao_ids or slurm_id in slurm_ids:
            raise ValueError("evaluation submission job identity duplicated")
        seen.add(entry_id)
        tao_ids.add(tao_id)
        slurm_ids.add(slurm_id)
    if ledger.get("status") == "submitted":
        if len(submissions) != 42 or seen != set(expected):
            raise ValueError(
                "submitted evaluation ledger must contain all 42 entries"
            )
    return ledger, digest


def accuracy_artifact_payload(
    study: dict[str, Any],
    checkpoint_artifact_sha256: str,
    evaluation_plan: dict[str, Any],
    evaluation_submissions_sha256: str,
    inspection: dict[str, Any],
    *,
    created_at: str,
    workflow_script_sha256: str | None = None,
) -> dict[str, Any]:
    if (
        len(inspection["jobs"]) != 42
        or not inspection["sdk_slurm_all_complete"]
        or not inspection["evidence_complete"]
    ):
        raise RuntimeError(
            "accuracy artifact requires all 42 SDK Complete / SLURM "
            "COMPLETED 0:0 jobs and one terminal finite mAP50 each"
        )
    plan_by_id = {
        item["entry_id"]: item for item in evaluation_plan["entries"]
    }
    jobs_by_id = {item["entry_id"]: item for item in inspection["jobs"]}
    reference_by_seed: dict[int, tuple[str, float]] = {}
    for entry in evaluation_plan["entries"]:
        if entry["profile_id"] != "reference":
            continue
        job = jobs_by_id[entry["entry_id"]]
        reference_by_seed[int(entry["seed"])] = (
            entry["entry_id"],
            float(job["accuracy"]["mAP50"]),
        )
    if set(reference_by_seed) != {1234, 271828, 314159}:
        raise RuntimeError("same-seed reference accuracy is incomplete")

    entries: list[dict[str, Any]] = []
    for entry in evaluation_plan["entries"]:
        job = jobs_by_id[entry["entry_id"]]
        map50 = float(job["accuracy"]["mAP50"])
        reference_entry_id, reference_map50 = reference_by_seed[
            int(entry["seed"])
        ]
        threshold = 0.98 * reference_map50
        retained = map50 >= threshold
        entries.append({
            "entry_id": entry["entry_id"],
            "profile_id": entry["profile_id"],
            "seed": entry["seed"],
            "axis": entry["axis"],
            "level": entry["level"],
            "execution": entry["execution"],
            "checkpoint_source_entry_id": (
                entry["checkpoint_source_entry_id"]
            ),
            "checkpoint_source_kind": entry["checkpoint_source_kind"],
            "checkpoint": entry["checkpoint"],
            "resolved_model_spec": entry["resolved_model_spec"],
            "resolved_model_spec_sha256": (
                entry["resolved_model_spec_sha256"]
            ),
            "evaluation_spec_sha256": entry["evaluation_spec_sha256"],
            "evaluation_command_sha256": (
                entry["evaluation_command_sha256"]
            ),
            "tao_job_id": job["tao_job_id"],
            "slurm_job_id": job["sdk_slurm_job_id"],
            "sdk_status": job["sdk_status"],
            "slurm_state": job["slurm_accounting"]["canonical_state"],
            "slurm_exit_code": job["slurm_accounting"]["exit_code"],
            "node_list": job["slurm_accounting"]["node_list"],
            "result_root": job["result_root"],
            "terminal_status_record": (
                job["accuracy"]["terminal_records"][0]
            ),
            "mAP50": map50,
            "same_seed_accuracy_retention": {
                "reference_entry_id": reference_entry_id,
                "reference_mAP50": reference_map50,
                "retention_fraction": 0.98,
                "required_mAP50": threshold,
                "absolute_delta": map50 - reference_map50,
                "retained_fraction": (
                    map50 / reference_map50
                    if reference_map50 != 0.0
                    else None
                ),
                "passes": retained,
            },
            "feeds_final_selection": False,
        })

    profile_summaries = []
    for profile_id in dict.fromkeys(
        entry["profile_id"] for entry in evaluation_plan["entries"]
    ):
        profile_entries = [
            item for item in entries if item["profile_id"] == profile_id
        ]
        if len(profile_entries) != 3:
            raise RuntimeError(f"profile {profile_id} lacks three seeds")
        profile_summaries.append({
            "profile_id": profile_id,
            "axis": profile_entries[0]["axis"],
            "level": profile_entries[0]["level"],
            "seed_entry_ids": [item["entry_id"] for item in profile_entries],
            "same_seed_accuracy_retention_passes": [
                item["same_seed_accuracy_retention"]["passes"]
                for item in profile_entries
            ],
            "all_three_seeds_pass_98_percent_retention": all(
                item["same_seed_accuracy_retention"]["passes"]
                for item in profile_entries
            ),
            "feeds_final_selection": False,
        })

    return {
        "schema_version": 1,
        "artifact_id": "dino_sensitivity_training_accuracy_20260728_v1",
        "created_at_utc": created_at,
        "status": "complete",
        "study_id": study["manifest"]["study_id"],
        "feeds_final_selection": False,
        "manual_selection_permitted": False,
        "profiles_or_ranges_changed": False,
        "winner_selected": False,
        "selection": {
            "performed": False,
            "selected_entry_id": None,
            "reason": (
                "This immutable artifact records preregistered sensitivity "
                "measurements and retention flags only."
            ),
        },
        "source": {
            "manifest_sha256": study["manifest_sha256"],
            "submission_report_sha256": (
                study["submission_report_sha256"]
            ),
            "frozen_training_plan_sha256": (
                study["plan"]["plan_sha256"]
            ),
            "checkpoint_artifact_sha256": checkpoint_artifact_sha256,
            "evaluation_plan_sha256": evaluation_plan["plan_sha256"],
            "evaluation_submissions_sha256": (
                evaluation_submissions_sha256
            ),
            "workflow_script_sha256": (
                workflow_script_sha256
                or sha256_file(Path(__file__).resolve())
            ),
        },
        "completion_contract": {
            "expected_entries": 42,
            "sdk_status": "Complete",
            "slurm_state": "COMPLETED",
            "slurm_exit_code": "0:0",
            "terminal_metric": "mAP50",
            "accuracy_retention_fraction": 0.98,
            "reference_policy": "same seed only",
        },
        "entries": entries,
        "profile_accuracy_summary": profile_summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--submission-report",
        type=Path,
        default=SUBMISSION_REPORT_PATH,
    )
    parser.add_argument(
        "--checkpoint-artifact",
        type=Path,
        default=CHECKPOINT_ARTIFACT_PATH,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "validate",
        help="Offline validation of the frozen inputs and all 42 eval specs.",
    )
    training_status = subparsers.add_parser("training-status")
    training_status.add_argument("--sdk-state", type=Path, default=TRAINING_SDK_STATE)
    training_status.add_argument("--report", type=Path, default=TRAINING_STATUS_PATH)
    finalize_checkpoints = subparsers.add_parser("finalize-checkpoints")
    finalize_checkpoints.add_argument(
        "--sdk-state",
        type=Path,
        default=TRAINING_SDK_STATE,
    )
    finalize_checkpoints.add_argument(
        "--report",
        type=Path,
        default=TRAINING_STATUS_PATH,
    )
    evaluation_plan = subparsers.add_parser("evaluation-plan")
    evaluation_plan.add_argument("--report", type=Path, default=EVALUATION_PLAN_PATH)
    submit = subparsers.add_parser("submit-evaluations")
    submit.add_argument("--acknowledgement", default="")
    submit.add_argument("--verify-remote", action="store_true")
    submit.add_argument(
        "--submissions",
        type=Path,
        default=EVALUATION_SUBMISSIONS_PATH,
    )
    submit.add_argument("--sdk-state", type=Path, default=EVALUATION_SDK_STATE)
    evaluation_status = subparsers.add_parser("evaluation-status")
    evaluation_status.add_argument(
        "--submissions",
        type=Path,
        default=EVALUATION_SUBMISSIONS_PATH,
    )
    evaluation_status.add_argument(
        "--sdk-state",
        type=Path,
        default=EVALUATION_SDK_STATE,
    )
    evaluation_status.add_argument(
        "--report",
        type=Path,
        default=EVALUATION_STATUS_PATH,
    )
    finalize_accuracy = subparsers.add_parser("finalize-accuracy")
    finalize_accuracy.add_argument(
        "--submissions",
        type=Path,
        default=EVALUATION_SUBMISSIONS_PATH,
    )
    finalize_accuracy.add_argument(
        "--sdk-state",
        type=Path,
        default=EVALUATION_SDK_STATE,
    )
    finalize_accuracy.add_argument(
        "--report",
        type=Path,
        default=EVALUATION_STATUS_PATH,
    )
    finalize_accuracy.add_argument(
        "--accuracy-artifact",
        type=Path,
        default=ACCURACY_ARTIFACT_PATH,
    )
    return parser.parse_args()


def status_report(
    study: dict[str, Any],
    inspection: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checked_at_utc": utc_timestamp(),
        "mode": mode,
        "study_id": study["manifest"]["study_id"],
        "manifest_sha256": study["manifest_sha256"],
        "submission_report_sha256": study["submission_report_sha256"],
        "frozen_plan_sha256": study["plan"]["plan_sha256"],
        "feeds_final_selection": False,
        "manual_selection_permitted": False,
        "profiles_or_ranges_changed": False,
        "winner_selected": False,
        **inspection,
    }


def main() -> int:
    args = parse_args()
    study = load_frozen_inputs(
        args.manifest.resolve(),
        args.submission_report.resolve(),
    )
    command = args.command
    if command == "validate":
        print(json.dumps(validate_workflow(study), indent=2, sort_keys=True))
        return 0

    if command in {"training-status", "finalize-checkpoints"}:
        load_env_file(Path(
            study["manifest"]["runtime_contract"]["secrets_env_path"]
        ))
        inspection = inspect_jobs(
            study,
            study["submissions"],
            args.sdk_state.resolve(),
            parse_accuracy=False,
        )
        report = status_report(study, inspection, mode=command)
        if command == "finalize-checkpoints":
            if not inspection["evidence_complete"]:
                atomic_json(args.report.resolve(), report)
                raise RuntimeError(
                    "checkpoint finalization blocked until all 33 jobs are "
                    "SDK Complete, SLURM COMPLETED 0:0, and expose exactly "
                    "one hashed epoch-9 checkpoint"
                )
            artifact_path = args.checkpoint_artifact.resolve()
            existing_created_at = None
            existing_workflow_sha256 = None
            if artifact_path.exists():
                existing_artifact, _existing_digest = (
                    load_checkpoint_artifact(artifact_path, study)
                )
                existing_created_at = existing_artifact["created_at_utc"]
                existing_workflow_sha256 = existing_artifact["source"][
                    "workflow_script_sha256"
                ]
            payload = checkpoint_artifact_payload(
                study,
                inspection,
                created_at=existing_created_at or utc_timestamp(),
                workflow_script_sha256=existing_workflow_sha256,
            )
            disposition, digest = immutable_json(artifact_path, payload)
            report["checkpoint_artifact"] = {
                "path": str(artifact_path),
                "disposition": disposition,
                "sha256": digest,
            }
        atomic_json(args.report.resolve(), report)
        print(json.dumps({
            "overall_status": report["overall_status"],
            "complete_job_count": report["complete_job_count"],
            "expected_job_count": report["expected_job_count"],
            "evidence_complete": report["evidence_complete"],
            "report": str(args.report.resolve()),
            "checkpoint_artifact": report.get("checkpoint_artifact"),
        }, indent=2, sort_keys=True))
        return 0

    checkpoint_artifact, checkpoint_digest = load_checkpoint_artifact(
        args.checkpoint_artifact.resolve(),
        study,
    )
    evaluation_plan, commands = build_evaluation_plan(
        study,
        checkpoint_artifact,
        checkpoint_digest,
    )
    if command == "evaluation-plan":
        atomic_json(args.report.resolve(), evaluation_plan)
        print(json.dumps({
            "status": evaluation_plan["status"],
            "entry_count": len(evaluation_plan["entries"]),
            "plan_sha256": evaluation_plan["plan_sha256"],
            "report": str(args.report.resolve()),
            "feeds_final_selection": False,
        }, indent=2, sort_keys=True))
        return 0

    if command == "submit-evaluations":
        if args.acknowledgement != EVALUATION_ACKNOWLEDGEMENT:
            raise RuntimeError(
                "evaluation submission refused: exact acknowledgement required"
            )
        if not args.verify_remote:
            raise RuntimeError(
                "evaluation submission requires --verify-remote"
            )
        loaded = load_env_file(Path(
            study["manifest"]["runtime_contract"]["secrets_env_path"]
        ))
        remote_checks = verify_evaluation_remote_contract(
            study,
            checkpoint_artifact,
        )
        ledger = submit_evaluations(
            study,
            evaluation_plan,
            commands,
            args.submissions.resolve(),
            args.sdk_state.resolve(),
            loaded,
            remote_checks,
        )
        print(json.dumps({
            "status": ledger["status"],
            "submission_count": len(ledger["submissions"]),
            "submissions": str(args.submissions.resolve()),
            "feeds_final_selection": False,
        }, indent=2, sort_keys=True))
        return 0

    ledger, ledger_digest = load_evaluation_submissions(
        args.submissions.resolve(),
        evaluation_plan,
    )
    load_env_file(Path(
        study["manifest"]["runtime_contract"]["secrets_env_path"]
    ))
    inspection = inspect_jobs(
        study,
        ledger["submissions"],
        args.sdk_state.resolve(),
        parse_accuracy=True,
    )
    if len(ledger["submissions"]) != 42:
        inspection["overall_status"] = "incomplete_submission_ledger"
        inspection["evidence_complete"] = False
    report = status_report(study, inspection, mode=command)
    report["evaluation_plan_sha256"] = evaluation_plan["plan_sha256"]
    report["evaluation_submissions_sha256"] = ledger_digest
    if command == "finalize-accuracy":
        if not inspection["evidence_complete"]:
            atomic_json(args.report.resolve(), report)
            raise RuntimeError(
                "accuracy finalization blocked until all 42 jobs are SDK "
                "Complete, SLURM COMPLETED 0:0, and expose one terminal mAP50"
            )
        artifact_path = args.accuracy_artifact.resolve()
        existing_created_at = None
        existing_workflow_sha256 = None
        if artifact_path.exists():
            existing_artifact = read_json(artifact_path)
            existing_created_at = existing_artifact.get("created_at_utc")
            existing_workflow_sha256 = existing_artifact.get(
                "source",
                {},
            ).get("workflow_script_sha256")
        payload = accuracy_artifact_payload(
            study,
            checkpoint_digest,
            evaluation_plan,
            ledger_digest,
            inspection,
            created_at=existing_created_at or utc_timestamp(),
            workflow_script_sha256=existing_workflow_sha256,
        )
        disposition, digest = immutable_json(artifact_path, payload)
        report["accuracy_artifact"] = {
            "path": str(artifact_path),
            "disposition": disposition,
            "sha256": digest,
        }
    atomic_json(args.report.resolve(), report)
    print(json.dumps({
        "overall_status": report["overall_status"],
        "complete_job_count": report["complete_job_count"],
        "expected_job_count": report["expected_job_count"],
        "evidence_complete": report["evidence_complete"],
        "report": str(args.report.resolve()),
        "accuracy_artifact": report.get("accuracy_artifact"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
