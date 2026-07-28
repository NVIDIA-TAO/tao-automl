#!/usr/bin/env python3

"""Validation-only exact-configuration checkpoint recovery for rec16.

This launcher reconstructs the frozen training configuration through the
expanded-search production harness.  It never mutates the frozen archive or
feeds a recovered checkpoint into selection.  A recovered checkpoint is not
accepted as the historical artifact unless its bytes match the historical
SHA256.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
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
RUNTIME_DIR = HERE / "runtime" / "rec16_checkpoint_recovery"
STATE_PATH = RUNTIME_DIR / "slurm_state.db"
REPORT_PATH = RUNTIME_DIR / "dry_run.json"
SUBMISSION_PATH = RUNTIME_DIR / "submission.json"

CANDIDATE_ID = "seed_271828_rec_16"
ACKNOWLEDGEMENT = (
    "USER_AUTHORIZED_VALIDATION_ONLY_REC16_CHECKPOINT_RECOVERY_20260728"
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
EXPECTED_TRAIN_SPEC_SHA256 = (
    "6f04eab6794cbf8bd707a966ab85b149d7bc24ea4ae238025bb6f3193fca9bf1"
)
EXPECTED_MODEL_SPEC_SHA256 = (
    "bc18216f670d96963ab795be8d6b845f576f4eed17a2482516da91d27eb6248d"
)
EXPECTED_HISTORICAL_CHECKPOINT_SHA256 = (
    "4b5ff50181ff919a2796cdd54027fff92eb57c908701a34408d29136d5565b4d"
)
EXPECTED_HISTORICAL_CHECKPOINT_SIZE = 506_687_042
EXPECTED_ORIGINAL_TAO_JOB_ID = "92d8f699-a780-4229-94ba-3520806d75da"
EXPECTED_SPECS = {
    "model.dec_layers": 3,
    "model.enc_layers": 6,
    "train.optim.lr": 0.0003007572504594793,
    "train.optim.weight_decay": 1.1000000000000001e-05,
}
ORIGINAL_ENTRYPOINT = {
    "path": (
        "/lustre/fsw/portfolios/edgeai/users/rarunachalam/entrypoints/"
        "job_92d8f699-a780-4229-94ba-3520806d75da.sh"
    ),
    "size_bytes": 80111,
    "sha256": "051c0fa574a1f7ad2b50560a0ae49f25f0518f748252fdded8bfe01e540ec206",
}
ORIGINAL_SBATCH = {
    "size_bytes": 2410,
    "sha256": "cb4faa856fcbbf9b8e1a0d57a1e7117b2210e42af1c5bbcedb5cf6e9bef19e95",
}


class RecoveryError(RuntimeError):
    """Raised when frozen recovery provenance or launch safety drifts."""


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


def load_module():
    if sha256_file(RUNNER_PATH) != EXPECTED_RUNNER_SHA256:
        raise RecoveryError("expanded_search_runner.py digest drift")
    spec = importlib.util.spec_from_file_location(
        "dino_expanded_search_runner_for_rec16_recovery",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RecoveryError("cannot import frozen expanded-search runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_launch_source_ready() -> dict[str, str]:
    relative = Path(__file__).resolve().relative_to(REPOSITORY).as_posix()
    checks = {
        "tracked": subprocess.run(
            ["git", "-C", str(REPOSITORY), "ls-files", "--error-unmatch", "--", relative],
            check=False,
            capture_output=True,
            text=True,
        ),
        "clean": subprocess.run(
            ["git", "-C", str(REPOSITORY), "status", "--porcelain", "--", relative],
            check=False,
            capture_output=True,
            text=True,
        ),
    }
    if checks["tracked"].returncode != 0:
        raise RecoveryError("recovery launcher must be tracked before launch")
    if checks["clean"].returncode != 0 or checks["clean"].stdout.strip():
        raise RecoveryError("recovery launcher must be committed and clean")
    return {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__).resolve()),
        "git_head": subprocess.run(
            ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }


def build_contract() -> tuple[Any, dict[str, Any], str, dict[str, Any]]:
    runner = load_module()
    if sha256_file(MANIFEST_PATH) != EXPECTED_MANIFEST_SHA256:
        raise RecoveryError("expanded-search manifest digest drift")
    if sha256_file(ARCHIVE_PATH) != EXPECTED_ARCHIVE_SHA256:
        raise RecoveryError("sealed seed archive digest drift")
    manifest, manifest_sha256 = runner.load_manifest(
        MANIFEST_PATH,
        supplied_file_sha256=EXPECTED_MANIFEST_SHA256,
    )
    runner.validate_local_provenance(manifest, MANIFEST_PATH)
    archive = json.loads(ARCHIVE_PATH.read_text(encoding="utf-8"))
    record = archive["records"].get(CANDIDATE_ID)
    if not isinstance(record, dict):
        raise RecoveryError(f"{CANDIDATE_ID} absent from sealed archive")
    expected_record = {
        "candidate_id": CANDIDATE_ID,
        "search_seed": 271828,
        "training_seed": 1234,
        "rec_id": 16,
        "specs": EXPECTED_SPECS,
        "resolved_train_spec_sha256": EXPECTED_TRAIN_SPEC_SHA256,
        "resolved_model_spec_sha256": EXPECTED_MODEL_SPEC_SHA256,
        "train_job_id": EXPECTED_ORIGINAL_TAO_JOB_ID,
    }
    for key, expected in expected_record.items():
        if record.get(key) != expected:
            raise RecoveryError(f"sealed record drift: {key}")
    checkpoint = record.get("checkpoint")
    if checkpoint != {
        "epoch": 9,
        "path": (
            "/lustre/fs11/portfolios/edgeai/projects/"
            "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/"
            "results/92d8f699-a780-4229-94ba-3520806d75da/results_dir/"
            "train/model_epoch_009_step_00440.pth"
        ),
        "sha256": EXPECTED_HISTORICAL_CHECKPOINT_SHA256,
        "size_bytes": EXPECTED_HISTORICAL_CHECKPOINT_SIZE,
    }:
        raise RecoveryError("sealed historical checkpoint identity drift")

    local = runner.validate_local_provenance(manifest, MANIFEST_PATH)
    template = yaml.safe_load(
        Path(local["train_template"]["path"]).read_text(encoding="utf-8")
    )
    train_spec = runner.training_spec(manifest, template, record["specs"])
    if runner.sha256_value(train_spec) != EXPECTED_TRAIN_SPEC_SHA256:
        raise RecoveryError("reconstructed train-spec digest drift")
    if runner.sha256_value(train_spec["model"]) != EXPECTED_MODEL_SPEC_SHA256:
        raise RecoveryError("reconstructed model-spec digest drift")
    if train_spec["train"]["seed"] != 1234:
        raise RecoveryError("training seed drift")
    if train_spec["train"]["num_epochs"] != 10:
        raise RecoveryError("training budget drift")
    if train_spec["train"]["num_gpus"] != 8:
        raise RecoveryError("training GPU topology drift")
    if train_spec["train"]["checkpoint_interval"] != 10:
        raise RecoveryError("terminal-checkpoint policy drift")
    if train_spec["train"]["cudnn"] != {
        "benchmark": False,
        "deterministic": True,
    }:
        raise RecoveryError("deterministic cuDNN policy drift")
    if train_spec["train"]["activation_checkpoint"] is not False:
        raise RecoveryError("activation-checkpoint policy drift")

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
    command_sha256 = hashlib.sha256(command.encode("utf-8")).hexdigest()
    contract = {
        "schema_version": 1,
        "experiment": "dino_rec16_validation_only_checkpoint_recovery",
        "candidate_id": CANDIDATE_ID,
        "source_manifest": {
            "path": str(MANIFEST_PATH),
            "sha256": manifest_sha256,
        },
        "source_seed_archive": {
            "path": str(ARCHIVE_PATH),
            "sha256": EXPECTED_ARCHIVE_SHA256,
        },
        "original_training": {
            "tao_job_id": EXPECTED_ORIGINAL_TAO_JOB_ID,
            "entrypoint": ORIGINAL_ENTRYPOINT,
            "sbatch": ORIGINAL_SBATCH,
            "train_spec_sha256": EXPECTED_TRAIN_SPEC_SHA256,
            "model_spec_sha256": EXPECTED_MODEL_SPEC_SHA256,
            "checkpoint_sha256": EXPECTED_HISTORICAL_CHECKPOINT_SHA256,
            "checkpoint_size_bytes": EXPECTED_HISTORICAL_CHECKPOINT_SIZE,
        },
        "reconstruction": {
            "candidate_specs": record["specs"],
            "training_seed": 1234,
            "train_epochs": 10,
            "checkpoint_interval_epochs": 10,
            "num_nodes": 1,
            "gpus_per_node": 8,
            "precision": "fp32",
            "distributed_strategy": "ddp",
            "sqsh_path": manifest["frozen_identity"]["runtime"]["sqsh_path"],
            "pretrained_model_path": manifest["frozen_identity"]["runtime"][
                "pretrained_model_path"
            ],
            "command_sha256": command_sha256,
            "command_size_bytes": len(command.encode("utf-8")),
        },
        "selection_isolation": {
            "selector_invoked_on_recovered_measurements": False,
            "selection_time_objectives_replaced": False,
            "measurements_feed_selection": False,
            "measurements_feed_reselection": False,
            "algorithm_selected_candidate_overridden": False,
            "frozen_archive_mutated": False,
        },
        "acceptance": {
            "configuration_exact_reconstruction": True,
            "byte_identical_checkpoint_not_assumed": True,
            "historical_checkpoint_substitution_permitted_only_if_sha256_matches": (
                EXPECTED_HISTORICAL_CHECKPOINT_SHA256
            ),
        },
    }
    return runner, manifest, command, contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--launch", action="store_true")
    parser.add_argument("--acknowledgement", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner, manifest, command, contract = build_contract()
    contract["mode"] = "launch" if args.launch else "dry_run"
    atomic_json(REPORT_PATH, contract)
    print(json.dumps(contract, indent=2, sort_keys=True), flush=True)
    if not args.launch:
        return 0
    if args.acknowledgement != ACKNOWLEDGEMENT:
        raise RecoveryError(
            f"launch requires --acknowledgement {ACKNOWLEDGEMENT}"
        )
    if SUBMISSION_PATH.exists():
        submission = json.loads(SUBMISSION_PATH.read_text(encoding="utf-8"))
        print(json.dumps(submission, indent=2, sort_keys=True), flush=True)
        return 0
    source = require_launch_source_ready()
    runner.load_env_file(
        Path(manifest["frozen_identity"]["runtime"]["secrets_env_path"])
    )
    runner.verify_remote_contract(manifest)
    runner.configure_slurm(manifest)
    os.environ["SLURM_BASE_RESULTS_DIR"] = (
        "/lustre/fsw/portfolios/edgeai/users/rarunachalam"
    )
    runner.ensure_sdk_importable()
    from tao_sdk.platforms.slurm import SlurmSDK

    runtime = manifest["frozen_identity"]["runtime"]
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    sdk = SlurmSDK(poll_interval=10, state_file=STATE_PATH)
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
        "command_sha256": contract["reconstruction"]["command_sha256"],
        "expected_historical_checkpoint_sha256": (
            EXPECTED_HISTORICAL_CHECKPOINT_SHA256
        ),
        "expected_historical_checkpoint_size_bytes": (
            EXPECTED_HISTORICAL_CHECKPOINT_SIZE
        ),
        "measurements_feed_selection": False,
        "measurements_feed_reselection": False,
        "frozen_archive_mutated": False,
    }
    atomic_json(SUBMISSION_PATH, submission)
    print(json.dumps(submission, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
