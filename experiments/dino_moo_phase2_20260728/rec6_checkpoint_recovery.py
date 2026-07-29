#!/usr/bin/env python3

"""Fail-closed validation-only checkpoint recovery for frozen DINO rec6.

The historical checkpoint for ``seed_271828_rec_6`` is missing from its
sealed result path.  This launcher reconstructs the candidate's exact
configuration from the immutable expanded-search manifest and seed archive,
then submits one SQSH-backed, one-node/eight-GPU training job through
``SlurmSDK``.  The recovered checkpoint is retained as a separately
provenanced validation artifact.  It never replaces the historical archive
entry and is never fed to selection or reselection.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
RUNNER_PATH = HERE / "expanded_search_runner.py"
MANIFEST_PATH = HERE / "expanded_search_manifest.v2.json"
ARCHIVE_PATH = (
    HERE
    / "runtime"
    / "expanded_search_v2"
    / "seed_271828"
    / "seed_archive.v1.json"
)
PREREGISTRATION_PATH = (
    HERE / "rec6_checkpoint_recovery_preregistration.v1.json"
)
RUNTIME_DIR = HERE / "runtime" / "rec6_checkpoint_recovery"
DEFAULT_STATE_PATH = RUNTIME_DIR / "slurm_state.db"
DEFAULT_REPORT_PATH = RUNTIME_DIR / "dry_run.json"
DEFAULT_SUBMISSION_PATH = RUNTIME_DIR / "submission.json"
DEFAULT_STATUS_PATH = RUNTIME_DIR / "status.json"
DEFAULT_EVIDENCE_PATH = (
    HERE
    / "latency_90_policy"
    / "matched"
    / "rec6_checkpoint_recovery_evidence.v1.json"
)
CANDIDATE_TABLE_PATH = (
    HERE
    / "runtime"
    / "expanded_search_v2"
    / "expanded_candidate_table.json"
)

CANDIDATE_ID = "seed_271828_rec_6"
CAMPAIGN_ID = "dino_latency_90pct_matched_20260728_v1"
RECOVERY_EVIDENCE_ID = "dino_latency_90_checkpoint_recovery_20260728_v1"
ACKNOWLEDGEMENT = (
    "USER_AUTHORIZED_VALIDATION_ONLY_REC6_CHECKPOINT_RECOVERY_20260728"
)
EXPECTED_MANIFEST_SHA256 = (
    "9ac29e1aa07167a040d217fdab2d3cfdea0baad690dc95a70f2fe6715908793a"
)
EXPECTED_RUNNER_SHA256 = (
    "0eb1948d4fb887b9c3fe938d60865ebb4ef86ae00d9ca80aa0d42b465a073073"
)
EXPECTED_ARCHIVE_SHA256 = (
    "a42a989ea27940ea9ae481212a75216c7f23f01602b0c260b6750c9fdb709c9e"
)
EXPECTED_ARCHIVE_INTERNAL_SHA256 = (
    "eedaa0a37e49cfa86e54be15a56352e4891044856ad5db782f6a4eed464dfb36"
)
EXPECTED_CANDIDATE_TABLE_SHA256 = (
    "5ba323d05d9ec8e3703e636f8b5e2975cc620eeec10df75ec6e792318dc2df03"
)
EXPECTED_CANDIDATE_TABLE_RECORD_SHA256 = (
    "caff9ec134ab0a2a6a7e12b4011e2da99ab466fb1467b7f4d0525895675198e2"
)
EXPECTED_SPECS_SHA256 = (
    "1366f23682c5c495b65ee6132cd883f2891a5c8b0e278605ec372301b11319df"
)
EXPECTED_RECORD_SHA256 = (
    "a0f2d4ce0927c07ef8119a5d62897fb11aadc6301431c90cf38ae65675252552"
)
EXPECTED_TRAIN_SPEC_SHA256 = (
    "0ce980ef6e6f793ab3a3aaac27957bc6daee4ab7b8269958036cd900d3dd9092"
)
EXPECTED_MODEL_SPEC_SHA256 = (
    "2891ae9dbb6097c1da53ce68201359f7da7992d30515ab815c89f906cdce21b1"
)
EXPECTED_COMMAND_SHA256 = (
    "f2ea97f16b473cf1aa094e7523c6b3445f7ae5444489ca19052be8c6ca66f65b"
)
EXPECTED_COMMAND_SIZE_BYTES = 80_063
EXPECTED_HISTORICAL_CHECKPOINT_SHA256 = (
    "0338c35be50bbad6189d38e8f9007856a60e87a0861c8a6ff5d0bf85cd6df6c5"
)
EXPECTED_HISTORICAL_CHECKPOINT_SIZE = 475_869_698
EXPECTED_HISTORICAL_CHECKPOINT_PATH = (
    "/lustre/fs11/portfolios/edgeai/projects/"
    "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/results/"
    "ce213919-8b61-4d86-b82f-c439ce0823d2/results_dir/train/"
    "model_epoch_009_step_00440.pth"
)
EXPECTED_ORIGINAL_TAO_JOB_ID = "ce213919-8b61-4d86-b82f-c439ce0823d2"
EXPECTED_ORIGINAL_SLURM_JOB_ID = "30959893"
EXPECTED_ORIGINAL_NODE = "batch-block7-01453"
EXPECTED_ORIGINAL_SCHEDULER_TIMES = {
    "submitted_at_utc": "2026-07-27T23:44:38Z",
    "started_at_utc": "2026-07-27T23:45:17Z",
    "ended_at_utc": "2026-07-27T23:51:16Z",
}
ORIGINAL_ENTRYPOINT = {
    "path": (
        "/lustre/fsw/portfolios/edgeai/users/rarunachalam/entrypoints/"
        "job_ce213919-8b61-4d86-b82f-c439ce0823d2.sh"
    ),
    "sha256": "6318078504570a86357acca55b77ff558e9eb3c36e02bd44e20be8e9908baef2",
    "size_bytes": 80_094,
}
ORIGINAL_SBATCH = {
    "path": (
        "/lustre/fsw/portfolios/edgeai/users/rarunachalam/sbatch/"
        "job_ce213919-8b61-4d86-b82f-c439ce0823d2.sbatch"
    ),
    "sha256": "bbf5027c9a6c3f75ff3866b15e92774f1b4a6030264d49e9f2cc85f3233e3ad2",
    "size_bytes": 2_410,
}
ORIGINAL_REMOTE_SPECS = {
    "path": (
        "/lustre/fsw/portfolios/edgeai/users/rarunachalam/specs/"
        "ce213919-8b61-4d86-b82f-c439ce0823d2.json"
    ),
    "sha256": "f603823798ed0c78f153bd8f5ae6292882bc4109af9c47b24b64d89c965dc01d",
    "size_bytes": 1_595,
}
EXPECTED_SPECS = {
    "model.dec_layers": 3,
    "model.enc_layers": 4,
    "train.optim.lr": 0.000487310659095131,
    "train.optim.weight_decay": 0.0009,
}
EXPECTED_SELECTION_OBJECTIVES = {
    "latency_ms": 57.17349525,
    "mAP50": 0.6000121414379619,
}
SELECTION_ISOLATION = {
    "selector_invoked_on_recovered_measurements": False,
    "selection_time_objectives_replaced": False,
    "measurements_feed_selection": False,
    "measurements_feed_reselection": False,
    "algorithm_selected_candidate_overridden": False,
    "frozen_archive_mutated": False,
}


class RecoveryError(RuntimeError):
    """Raised when immutable recovery provenance or launch safety drifts."""


def canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise RecoveryError(f"value is not canonical JSON: {error}") from error
    return encoded.encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    if pending.exists():
        raise RecoveryError(f"stale pending artifact: {pending}")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(
            value,
            stream,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise RecoveryError(
            f"refusing to overwrite immutable recovery evidence: {path}"
        ) from error


def load_module() -> Any:
    if sha256_file(RUNNER_PATH) != EXPECTED_RUNNER_SHA256:
        raise RecoveryError("expanded_search_runner.py digest drift")
    spec = importlib.util.spec_from_file_location(
        "dino_expanded_search_runner_for_rec6_recovery",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RecoveryError("cannot import frozen expanded-search runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_launch_source_ready() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        PREREGISTRATION_PATH.resolve(),
    )
    relative_paths = [
        path.relative_to(REPOSITORY).as_posix() for path in paths
    ]
    tracked = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY),
            "ls-files",
            "--error-unmatch",
            "--",
            *relative_paths,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    clean = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY),
            "status",
            "--porcelain",
            "--",
            *relative_paths,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        raise RecoveryError(
            "recovery launcher and preregistration must be tracked before launch"
        )
    if clean.returncode != 0 or clean.stdout.strip():
        raise RecoveryError(
            "recovery launcher and preregistration must be committed and clean"
        )
    return {
        "git_head": subprocess.run(
            ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "files": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
    }


def load_preregistration() -> tuple[dict[str, Any], str]:
    preregistration = json.loads(
        PREREGISTRATION_PATH.read_text(encoding="utf-8")
    )
    if preregistration.get("preregistration_id") != (
        "dino_rec6_validation_only_checkpoint_recovery_20260728_v1"
    ):
        raise RecoveryError("unexpected recovery preregistration identity")
    if preregistration.get("candidate_id") != CANDIDATE_ID:
        raise RecoveryError("recovery preregistration candidate drift")
    if preregistration.get("selection_isolation") != SELECTION_ISOLATION:
        raise RecoveryError("recovery preregistration selection-isolation drift")
    if preregistration.get("historical_checkpoint", {}).get("path") != (
        EXPECTED_HISTORICAL_CHECKPOINT_PATH
    ):
        raise RecoveryError("recovery preregistration checkpoint-path drift")
    return preregistration, sha256_file(PREREGISTRATION_PATH)


def build_contract() -> tuple[Any, dict[str, Any], str, dict[str, Any]]:
    runner = load_module()
    preregistration, preregistration_sha256 = load_preregistration()
    if sha256_file(MANIFEST_PATH) != EXPECTED_MANIFEST_SHA256:
        raise RecoveryError("expanded-search manifest digest drift")
    if sha256_file(ARCHIVE_PATH) != EXPECTED_ARCHIVE_SHA256:
        raise RecoveryError("sealed seed archive digest drift")
    if sha256_file(CANDIDATE_TABLE_PATH) != EXPECTED_CANDIDATE_TABLE_SHA256:
        raise RecoveryError("sealed expanded candidate-table digest drift")
    manifest, manifest_sha256 = runner.load_manifest(
        MANIFEST_PATH,
        supplied_file_sha256=EXPECTED_MANIFEST_SHA256,
    )
    local = runner.validate_local_provenance(manifest, MANIFEST_PATH)
    archive = json.loads(ARCHIVE_PATH.read_text(encoding="utf-8"))
    if archive.get("archive_sha256") != EXPECTED_ARCHIVE_INTERNAL_SHA256:
        raise RecoveryError("sealed seed archive internal identity drift")
    record = archive.get("records", {}).get(CANDIDATE_ID)
    if not isinstance(record, dict):
        raise RecoveryError(f"{CANDIDATE_ID} absent from sealed archive")
    if sha256_value(record) != EXPECTED_RECORD_SHA256:
        raise RecoveryError("sealed rec6 candidate-record digest drift")
    table = json.loads(CANDIDATE_TABLE_PATH.read_text(encoding="utf-8"))
    table_records = [
        row
        for row in table.get("rows", [])
        if row.get("candidate_id") == CANDIDATE_ID
    ]
    if len(table_records) != 1:
        raise RecoveryError(
            "sealed expanded candidate table must contain rec6 exactly once"
        )
    if sha256_value(table_records[0]) != (
        EXPECTED_CANDIDATE_TABLE_RECORD_SHA256
    ):
        raise RecoveryError("sealed rec6 candidate-table record digest drift")
    if sha256_value(table_records[0].get("specs")) != EXPECTED_SPECS_SHA256:
        raise RecoveryError("sealed rec6 candidate specs digest drift")
    expected_record = {
        "candidate_id": CANDIDATE_ID,
        "search_seed": 271828,
        "training_seed": 1234,
        "rec_id": 6,
        "specs": EXPECTED_SPECS,
        "resolved_train_spec_sha256": EXPECTED_TRAIN_SPEC_SHA256,
        "resolved_model_spec_sha256": EXPECTED_MODEL_SPEC_SHA256,
        "train_job_id": EXPECTED_ORIGINAL_TAO_JOB_ID,
    }
    for key, expected in expected_record.items():
        if record.get(key) != expected:
            raise RecoveryError(f"sealed record drift: {key}")
    if record.get("training_runtime", {}).get("slurm_job_id") != (
        EXPECTED_ORIGINAL_SLURM_JOB_ID
    ):
        raise RecoveryError("sealed original SLURM identity drift")
    if record.get("checkpoint") != {
        "epoch": 9,
        "path": EXPECTED_HISTORICAL_CHECKPOINT_PATH,
        "sha256": EXPECTED_HISTORICAL_CHECKPOINT_SHA256,
        "size_bytes": EXPECTED_HISTORICAL_CHECKPOINT_SIZE,
    }:
        raise RecoveryError("sealed historical checkpoint identity drift")
    objectives = record.get("objective_values", {})
    for name, expected in EXPECTED_SELECTION_OBJECTIVES.items():
        if objectives.get(name) != expected:
            raise RecoveryError(f"sealed selection objective drift: {name}")

    template = yaml.safe_load(
        Path(local["train_template"]["path"]).read_text(encoding="utf-8")
    )
    train_spec = runner.training_spec(manifest, template, record["specs"])
    if runner.sha256_value(train_spec) != EXPECTED_TRAIN_SPEC_SHA256:
        raise RecoveryError("reconstructed train-spec digest drift")
    if runner.sha256_value(train_spec["model"]) != EXPECTED_MODEL_SPEC_SHA256:
        raise RecoveryError("reconstructed model-spec digest drift")
    frozen_training = manifest["frozen_identity"]["training_controls"]
    expected_training = {
        "seed": 1234,
        "num_epochs": frozen_training["train_epochs"],
        "num_gpus": frozen_training["num_gpus"],
        "gpu_ids": list(range(frozen_training["num_gpus"])),
        "checkpoint_interval": frozen_training["checkpoint_interval_epochs"],
        "cudnn": {
            "benchmark": frozen_training["cudnn_benchmark"],
            "deterministic": frozen_training["cudnn_deterministic"],
        },
        "activation_checkpoint": frozen_training["activation_checkpoint"],
    }
    for key, expected in expected_training.items():
        if train_spec["train"].get(key) != expected:
            raise RecoveryError(f"reconstructed training control drift: {key}")
    if expected_training["num_epochs"] != 10:
        raise RecoveryError("frozen training budget is not 10 epochs")
    if expected_training["num_gpus"] != 8:
        raise RecoveryError("frozen training topology is not eight GPUs")

    runner.ensure_sdk_importable()
    from tao_sdk.script_runner import build_entrypoint

    skill_info = yaml.safe_load(
        (runner.SKILL_DIR / "references" / "skill_info.yaml").read_text(
            encoding="utf-8"
        )
    )
    action = skill_info["actions"]["train"]
    entrypoint = build_entrypoint(
        command=action["command"],
        specs=train_spec,
        inputs=action["inputs"],
        outputs=action["outputs"],
        config_format=action["config_format"],
        upload_excludes=action["upload_excludes"],
    )
    command = entrypoint["command"]
    command_bytes = command.encode("utf-8")
    command_sha256 = hashlib.sha256(command_bytes).hexdigest()
    if command_sha256 != EXPECTED_COMMAND_SHA256:
        raise RecoveryError("reconstructed training command digest drift")
    if len(command_bytes) != EXPECTED_COMMAND_SIZE_BYTES:
        raise RecoveryError("reconstructed training command size drift")

    runtime = manifest["frozen_identity"]["runtime"]
    contract = {
        "schema_version": 1,
        "experiment": (
            "dino_rec6_validation_only_exact_configuration_checkpoint_recovery"
        ),
        "candidate_id": CANDIDATE_ID,
        "preregistration": {
            "path": str(PREREGISTRATION_PATH),
            "sha256": preregistration_sha256,
            "identity": preregistration["preregistration_id"],
        },
        "source_manifest": {
            "path": str(MANIFEST_PATH),
            "sha256": manifest_sha256,
        },
        "source_seed_archive": {
            "path": str(ARCHIVE_PATH),
            "whole_file_sha256": EXPECTED_ARCHIVE_SHA256,
            "internal_archive_sha256": EXPECTED_ARCHIVE_INTERNAL_SHA256,
            "candidate_record_sha256": EXPECTED_RECORD_SHA256,
        },
        "source_candidate_table": {
            "path": str(CANDIDATE_TABLE_PATH),
            "whole_file_sha256": EXPECTED_CANDIDATE_TABLE_SHA256,
            "candidate_record_sha256": (
                EXPECTED_CANDIDATE_TABLE_RECORD_SHA256
            ),
            "candidate_specs_sha256": EXPECTED_SPECS_SHA256,
        },
        "historical_training": {
            "tao_job_id": EXPECTED_ORIGINAL_TAO_JOB_ID,
            "slurm_job_id": EXPECTED_ORIGINAL_SLURM_JOB_ID,
            "node": EXPECTED_ORIGINAL_NODE,
            "scheduler_times": EXPECTED_ORIGINAL_SCHEDULER_TIMES,
            "entrypoint": ORIGINAL_ENTRYPOINT,
            "sbatch": ORIGINAL_SBATCH,
            "remote_specs": ORIGINAL_REMOTE_SPECS,
            "train_spec_sha256": EXPECTED_TRAIN_SPEC_SHA256,
            "model_spec_sha256": EXPECTED_MODEL_SPEC_SHA256,
            "checkpoint": record["checkpoint"],
            "checkpoint_missing_precondition": (
                "must be verified over SSH immediately before submission"
            ),
        },
        "reconstruction": {
            "candidate_specs": record["specs"],
            "search_seed": record["search_seed"],
            "training_seed": record["training_seed"],
            "train_epochs": 10,
            "checkpoint_interval_epochs": 10,
            "num_nodes": 1,
            "gpus_per_node": 8,
            "gpu_ids": list(range(8)),
            "precision": runtime["precision"],
            "distributed_strategy": runtime["distributed_strategy"],
            "cudnn": train_spec["train"]["cudnn"],
            "activation_checkpoint": train_spec["train"][
                "activation_checkpoint"
            ],
            "sqsh_path": runtime["sqsh_path"],
            "sqsh_sha256": runtime["sqsh_sha256"],
            "pretrained_model_path": runtime["pretrained_model_path"],
            "pretrained_model_sha256": runtime["pretrained_model_sha256"],
            "partition": runtime["partition"],
            "account": runtime["account"],
            "command_sha256": command_sha256,
            "command_size_bytes": len(command_bytes),
        },
        "selection_time_evidence_preserved": {
            "mAP50": objectives["mAP50"],
            "latency_ms": objectives["latency_ms"],
            "checkpoint_sha256": EXPECTED_HISTORICAL_CHECKPOINT_SHA256,
        },
        "selection_isolation": dict(SELECTION_ISOLATION),
        "retention_policy": {
            "retain_recovered_checkpoint": True,
            "recovered_artifact_is_separately_provenanced": True,
            "historical_archive_replacement_permitted": False,
            "historical_checkpoint_substitution_permitted_only_if_sha256_matches": (
                EXPECTED_HISTORICAL_CHECKPOINT_SHA256
            ),
        },
        "acceptance": {
            "configuration_exact_reconstruction": True,
            "byte_identical_checkpoint_not_assumed": True,
            "accuracy_or_latency_equivalence_not_assumed": True,
            "recovery_is_validation_only": True,
        },
    }
    return runner, manifest, command, contract


def verify_historical_checkpoint_missing(runner: Any) -> dict[str, Any]:
    output = runner.remote_output(
        "test ! -e "
        + shlex.quote(EXPECTED_HISTORICAL_CHECKPOINT_PATH)
        + " && echo MISSING || echo PRESENT",
        timeout=120,
    ).strip()
    if output != "MISSING":
        raise RecoveryError(
            "historical rec6 checkpoint exists; refusing duplicate recovery"
        )
    return {
        "path": EXPECTED_HISTORICAL_CHECKPOINT_PATH,
        "observed": "missing",
        "verified": True,
    }


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
    raise RecoveryError(f"expected a Lustre results URI, got {uri!r}")


def scheduler_accounting(
    runner: Any,
    slurm_job_id: str,
) -> dict[str, Any]:
    if not slurm_job_id.isdigit():
        raise RecoveryError("recovery SLURM job ID must be numeric")
    output = runner.remote_output(
        " ".join(
            [
                "sacct",
                "-X",
                "-j",
                shlex.quote(slurm_job_id),
                "--noheader",
                "--parsable2",
                "--format=JobIDRaw,State,ExitCode,NodeList,Submit,Start,End",
            ]
        ),
        timeout=120,
    )
    records = []
    for line in output.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 7 or fields[0] != slurm_job_id:
            continue
        state = fields[1].split("+", 1)[0].split(None, 1)[0]
        records.append(
            {
                "slurm_job_id": fields[0],
                "state": state,
                "exit_code": fields[2],
                "node": fields[3],
                "submit_time_utc": normalize_slurm_time(fields[4]),
                "start_time_utc": normalize_slurm_time(fields[5]),
                "end_time_utc": normalize_slurm_time(fields[6]),
            }
        )
    if len(records) != 1:
        raise RecoveryError(
            "expected exactly one top-level SLURM accounting row for "
            f"{slurm_job_id}, found {len(records)}"
        )
    record = records[0]
    record["complete"] = (
        record["state"] == "COMPLETED" and record["exit_code"] == "0:0"
    )
    return record


def normalize_slurm_time(value: str) -> str | None:
    value = value.strip()
    if not value or value in {"Unknown", "N/A", "None"}:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", value):
        return value + "Z"
    if re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
        value,
    ):
        return value
    raise RecoveryError(f"unexpected SLURM timestamp format: {value!r}")


def remote_file_identity(
    runner: Any,
    path: str,
    *,
    timeout: int = 900,
) -> dict[str, Any]:
    output = runner.remote_output(
        " && ".join(
            [
                f"test -f {shlex.quote(path)}",
                f"stat -c %s {shlex.quote(path)}",
                f"sha256sum {shlex.quote(path)}",
            ]
        ),
        timeout=timeout,
    ).splitlines()
    if len(output) != 2:
        raise RecoveryError(f"could not identify remote regular file: {path}")
    try:
        size_bytes = int(output[0].strip())
    except ValueError as error:
        raise RecoveryError(f"invalid remote file size for {path}") from error
    fields = output[1].split(None, 1)
    if len(fields) != 2:
        raise RecoveryError(f"invalid sha256sum output for {path}")
    digest, returned_path = fields
    if returned_path.strip() != path:
        raise RecoveryError(f"sha256sum returned a different path for {path}")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RecoveryError(f"malformed remote SHA256 for {path}")
    return {
        "path": path,
        "sha256": digest,
        "size_bytes": size_bytes,
    }


def locate_terminal_checkpoint(
    runner: Any,
    result_root: str,
    tao_job_id: str,
) -> dict[str, Any]:
    root = Path(result_root)
    if root.name != tao_job_id or root.parent.name != "results":
        raise RecoveryError(
            "recovery result root is not scoped to its TAO job: "
            f"{result_root}"
        )
    script = "\n".join(
        [
            "import json,re,sys",
            "from pathlib import Path",
            "root=Path(sys.argv[1])",
            "paths=sorted(str(p) for p in root.rglob('*') "
            "if p.is_file() and p.suffix.lower() in ('.pth','.ckpt'))",
            "pattern=re.compile(r'^model_epoch_0*9_step_[0-9]+"
            "\\.(?:pth|ckpt)$',re.I)",
            "matches=[p for p in paths if pattern.match(Path(p).name)]",
            "print(json.dumps({'all_checkpoints':paths,"
            "'terminal_epoch_9':matches}))",
        ]
    )
    output = runner.remote_output(
        f"python3 -c {shlex.quote(script)} {shlex.quote(result_root)}",
        timeout=300,
    )
    payload = json.loads(output)
    matches = payload.get("terminal_epoch_9")
    if not isinstance(matches, list) or len(matches) != 1:
        raise RecoveryError(
            "expected exactly one terminal epoch-9 recovery checkpoint under "
            f"{result_root}, found {matches!r}"
        )
    identity = remote_file_identity(runner, matches[0], timeout=1800)
    identity["epoch"] = 9
    return identity


def load_submission() -> dict[str, Any]:
    if not DEFAULT_SUBMISSION_PATH.is_file():
        raise RecoveryError(
            f"recovery submission record is missing: {DEFAULT_SUBMISSION_PATH}"
        )
    submission = json.loads(
        DEFAULT_SUBMISSION_PATH.read_text(encoding="utf-8")
    )
    if submission.get("candidate_id") != CANDIDATE_ID:
        raise RecoveryError("recovery submission candidate drift")
    if submission.get("command_sha256") != EXPECTED_COMMAND_SHA256:
        raise RecoveryError("recovery submission command digest drift")
    if submission.get("selection_isolation") != SELECTION_ISOLATION:
        raise RecoveryError("recovery submission selection-isolation drift")
    tao_job_id = submission.get("tao_job_id")
    slurm_job_id = str(submission.get("slurm_job_id", ""))
    if not isinstance(tao_job_id, str) or not tao_job_id:
        raise RecoveryError("recovery submission TAO job ID is missing")
    if not slurm_job_id.isdigit():
        raise RecoveryError("recovery submission SLURM job ID is invalid")
    return submission


def recovery_remote_artifact_paths(tao_job_id: str) -> dict[str, str]:
    base = "/lustre/fsw/portfolios/edgeai/users/rarunachalam"
    return {
        "entrypoint": f"{base}/entrypoints/job_{tao_job_id}.sh",
        "sbatch": f"{base}/sbatch/job_{tao_job_id}.sbatch",
        "remote_specs": f"{base}/specs/{tao_job_id}.json",
    }


def inspect_recovery(
    runner: Any,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    submission = load_submission()
    database = sdk_database_path(DEFAULT_STATE_PATH)
    if not database.is_file():
        raise RecoveryError(f"SDK durable state database is missing: {database}")
    runner.configure_slurm(manifest)
    runner.ensure_sdk_importable()
    from tao_sdk.platforms.slurm import SlurmSDK

    sdk = SlurmSDK(poll_interval=10, state_file=DEFAULT_STATE_PATH)
    try:
        status = sdk.get_job_status(submission["tao_job_id"])
        identity = runner._runtime_identity_from_store(
            sdk,
            submission["tao_job_id"],
        )
        sdk_slurm_job_id = str(identity.get("slurm_job_id", ""))
        if sdk_slurm_job_id != str(submission["slurm_job_id"]):
            raise RecoveryError(
                "SDK durable SLURM identity differs from submission record"
            )
        accounting = scheduler_accounting(runner, sdk_slurm_job_id)
        result_root = local_lustre_path(
            sdk.get_job_results_dir(submission["tao_job_id"])
        )
        sdk_complete = status.status == "Complete"
        complete = sdk_complete and accounting["complete"]
        record: dict[str, Any] = {
            "schema_version": 1,
            "candidate_id": CANDIDATE_ID,
            "tao_job_id": submission["tao_job_id"],
            "slurm_job_id": sdk_slurm_job_id,
            "sdk_status": status.status,
            "sdk_message": status.message,
            "slurm_accounting": accounting,
            "result_root": result_root,
            "complete": complete,
            "terminal": status.status in {"Complete", "Error", "Canceled"},
            "selection_isolation": dict(SELECTION_ISOLATION),
            "inspected_at_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }
        if complete:
            record["checkpoint"] = locate_terminal_checkpoint(
                runner,
                result_root,
                submission["tao_job_id"],
            )
            paths = recovery_remote_artifact_paths(
                submission["tao_job_id"]
            )
            record["launch_artifacts"] = {
                label: remote_file_identity(runner, path)
                for label, path in paths.items()
            }
            record["historical_checkpoint_missing"] = (
                verify_historical_checkpoint_missing(runner)
            )
        return record
    finally:
        monitor = getattr(sdk, "_monitor", None)
        if monitor is not None and callable(getattr(monitor, "stop", None)):
            monitor.stop()
        store = getattr(sdk, "_store", None)
        if store is not None and callable(getattr(store, "close", None)):
            store.close()


def build_recovery_evidence(
    status: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    if status.get("complete") is not True:
        raise RecoveryError(
            "recovery is not SDK-and-SLURM complete; evidence cannot be finalized"
        )
    accounting = status.get("slurm_accounting", {})
    if (
        accounting.get("state") != "COMPLETED"
        or accounting.get("exit_code") != "0:0"
    ):
        raise RecoveryError("recovery SLURM completion evidence is invalid")
    checkpoint = status.get("checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("epoch") != 9
        or not Path(str(checkpoint.get("path", ""))).is_absolute()
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(checkpoint.get("sha256", "")),
        )
        or not isinstance(checkpoint.get("size_bytes"), int)
        or checkpoint["size_bytes"] <= 0
    ):
        raise RecoveryError("recovered checkpoint identity is invalid")
    artifacts = status.get("launch_artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "entrypoint",
        "sbatch",
        "remote_specs",
    }:
        raise RecoveryError("recovery launch-artifact provenance is incomplete")
    if status.get("historical_checkpoint_missing", {}).get(
        "verified"
    ) is not True:
        raise RecoveryError("historical checkpoint absence was not preserved")
    core = {
        "schema_version": 1,
        "evidence_id": RECOVERY_EVIDENCE_ID,
        "status": "identity_preserving_recovery_complete",
        "campaign_id": CAMPAIGN_ID,
        "candidate_id": CANDIDATE_ID,
        "historical_checkpoint": {
            "path": EXPECTED_HISTORICAL_CHECKPOINT_PATH,
            "sha256": EXPECTED_HISTORICAL_CHECKPOINT_SHA256,
        },
        "candidate_table_record_sha256": (
            EXPECTED_CANDIDATE_TABLE_RECORD_SHA256
        ),
        "resolved_model_spec_sha256": EXPECTED_MODEL_SPEC_SHA256,
        "specs_sha256": EXPECTED_SPECS_SHA256,
        "search_seed": 271828,
        "training_seed": 1234,
        "rec_id": 6,
        "recovered_checkpoint": {
            "path": checkpoint["path"],
            "sha256": checkpoint["sha256"],
        },
        "exact_candidate_configuration_preserved": True,
        "architecture_proxy_used": False,
        "manual_candidate_substitution_used": False,
        "result_driven_parameter_change_used": False,
        "measurements_feed_selection": False,
        "measurements_feed_reselection": False,
        "algorithm_selected_candidate_overridden": False,
        "recovery_provenance": {
            "tao_job_id": status["tao_job_id"],
            "slurm_job_id": status["slurm_job_id"],
            "node": accounting["node"],
            "submit_time_utc": accounting["submit_time_utc"],
            "start_time_utc": accounting["start_time_utc"],
            "end_time_utc": accounting["end_time_utc"],
            "sdk_status": status["sdk_status"],
            "slurm_state": accounting["state"],
            "slurm_exit_code": accounting["exit_code"],
            "result_root": status["result_root"],
            "entrypoint": artifacts["entrypoint"],
            "sbatch": artifacts["sbatch"],
            "remote_specs": artifacts["remote_specs"],
            "reconstructed_command_sha256": EXPECTED_COMMAND_SHA256,
            "reconstructed_train_spec_sha256": EXPECTED_TRAIN_SPEC_SHA256,
            "reconstructed_model_spec_sha256": EXPECTED_MODEL_SPEC_SHA256,
            "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "source_seed_archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "source_candidate_table_sha256": (
                EXPECTED_CANDIDATE_TABLE_SHA256
            ),
            "preregistration": contract["preregistration"],
        },
        "checkpoint_identity": {
            "epoch": checkpoint["epoch"],
            "size_bytes": checkpoint["size_bytes"],
            "historical_sha256_match": (
                checkpoint["sha256"]
                == EXPECTED_HISTORICAL_CHECKPOINT_SHA256
            ),
            "byte_identical_checkpoint_assumed": False,
            "configuration_exact_not_byte_identity": True,
            "retained_for_matched_validation": True,
        },
        "selection_isolation": dict(SELECTION_ISOLATION),
        "frozen_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }
    return {
        **core,
        "artifact_integrity": {
            "canonical_payload_sha256": sha256_value(core),
            "hash_algorithm": "sha256",
            "hash_excludes": ["artifact_integrity"],
        },
    }


def validate_existing_evidence(
    evidence: dict[str, Any],
    status: dict[str, Any],
) -> None:
    integrity = evidence.get("artifact_integrity", {})
    core = {
        key: value
        for key, value in evidence.items()
        if key != "artifact_integrity"
    }
    if integrity.get("canonical_payload_sha256") != sha256_value(core):
        raise RecoveryError("existing recovery evidence internal digest drift")
    expected = {
        "evidence_id": RECOVERY_EVIDENCE_ID,
        "status": "identity_preserving_recovery_complete",
        "campaign_id": CAMPAIGN_ID,
        "candidate_id": CANDIDATE_ID,
    }
    for key, value in expected.items():
        if evidence.get(key) != value:
            raise RecoveryError(f"existing recovery evidence drift: {key}")
    recovered = evidence.get("recovered_checkpoint", {})
    checkpoint = status.get("checkpoint", {})
    if recovered != {
        "path": checkpoint.get("path"),
        "sha256": checkpoint.get("sha256"),
    }:
        raise RecoveryError(
            "existing recovery evidence checkpoint differs from live provenance"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument("--acknowledgement", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner, manifest, command, contract = build_contract()
    if args.status or args.finalize:
        runner.load_env_file(
            Path(manifest["frozen_identity"]["runtime"]["secrets_env_path"])
        )
        status = inspect_recovery(runner, manifest)
        atomic_json(DEFAULT_STATUS_PATH, status)
        if not args.finalize:
            print(json.dumps(status, indent=2, sort_keys=True), flush=True)
            return 0
        if status.get("complete") is not True:
            print(json.dumps(status, indent=2, sort_keys=True), flush=True)
            raise RecoveryError(
                "recovery is not complete; refusing to finalize evidence"
            )
        if DEFAULT_EVIDENCE_PATH.exists():
            evidence = json.loads(
                DEFAULT_EVIDENCE_PATH.read_text(encoding="utf-8")
            )
            validate_existing_evidence(evidence, status)
        else:
            evidence = build_recovery_evidence(status, contract)
            write_new_json(DEFAULT_EVIDENCE_PATH, evidence)
        print(
            json.dumps(
                {
                    "status": "recovery_evidence_finalized",
                    "path": str(DEFAULT_EVIDENCE_PATH),
                    "whole_file_sha256": sha256_file(
                        DEFAULT_EVIDENCE_PATH
                    ),
                    "internal_sha256": evidence["artifact_integrity"][
                        "canonical_payload_sha256"
                    ],
                    "evidence": evidence,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    contract["mode"] = "launch" if args.launch else "dry_run"
    atomic_json(DEFAULT_REPORT_PATH, contract)
    print(json.dumps(contract, indent=2, sort_keys=True), flush=True)
    if not args.launch:
        return 0
    if args.acknowledgement != ACKNOWLEDGEMENT:
        raise RecoveryError(
            f"launch requires --acknowledgement {ACKNOWLEDGEMENT}"
        )
    if DEFAULT_SUBMISSION_PATH.exists():
        submission = json.loads(
            DEFAULT_SUBMISSION_PATH.read_text(encoding="utf-8")
        )
        print(json.dumps(submission, indent=2, sort_keys=True), flush=True)
        return 0
    source = require_launch_source_ready()
    runner.load_env_file(
        Path(manifest["frozen_identity"]["runtime"]["secrets_env_path"])
    )
    remote_contract = runner.verify_remote_contract(manifest)
    missing_checkpoint = verify_historical_checkpoint_missing(runner)
    runner.configure_slurm(manifest)
    os.environ["SLURM_BASE_RESULTS_DIR"] = (
        "/lustre/fsw/portfolios/edgeai/users/rarunachalam"
    )
    runner.ensure_sdk_importable()
    from tao_sdk.platforms.slurm import SlurmSDK

    runtime = manifest["frozen_identity"]["runtime"]
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    sdk = SlurmSDK(poll_interval=10, state_file=DEFAULT_STATE_PATH)
    job = sdk.create_job(
        image=runtime["sqsh_path"],
        command=command,
        gpu_count=8,
        num_nodes=1,
        partition=runtime["partition"],
        account=runtime["account"],
    )
    identity = runner._runtime_identity_from_store(sdk, job.id)
    submission = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "tao_job_id": job.id,
        "slurm_job_id": str(identity.get("slurm_job_id", "")),
        "result_root": identity.get("result_root"),
        "source": source,
        "remote_contract": remote_contract,
        "historical_checkpoint_missing": missing_checkpoint,
        "command_sha256": EXPECTED_COMMAND_SHA256,
        "expected_historical_checkpoint": {
            "path": EXPECTED_HISTORICAL_CHECKPOINT_PATH,
            "sha256": EXPECTED_HISTORICAL_CHECKPOINT_SHA256,
            "size_bytes": EXPECTED_HISTORICAL_CHECKPOINT_SIZE,
        },
        "retention": {
            "retain_recovered_checkpoint": True,
            "historical_archive_replacement_permitted": False,
            "byte_identity_assumed": False,
        },
        "selection_isolation": dict(SELECTION_ISOLATION),
    }
    atomic_json(DEFAULT_SUBMISSION_PATH, submission)
    print(json.dumps(submission, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
