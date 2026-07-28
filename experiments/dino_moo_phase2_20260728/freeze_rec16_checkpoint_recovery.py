#!/usr/bin/env python3

"""Freeze fail-closed validation-only checkpoint recovery evidence for rec16.

The two completed scheduler-assigned exact-configuration retrains are the only
eligible attempts.  Selection is value-independent: the earliest attempt by
``(submit_time_utc, numeric_slurm_job_id, tao_job_id)`` is used.  Checkpoint
bytes, hashes, accuracy, latency, and candidate desirability never participate
in that rule.  The original-node replay is recorded only as supplementary,
non-gating pending work.
"""

from __future__ import annotations

import argparse
import copy
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
import tempfile
from typing import Any, Callable, Mapping, Protocol


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
DEFAULT_OUTPUT = HERE / "rec16_checkpoint_recovery_evidence.v1.json"
DEFAULT_SECRETS_ENV = Path("/localhome/local-rarunachalam/.tao/config.env")
EXPANDED_MANIFEST = HERE / "expanded_search_manifest.v2.json"
SEED_ARCHIVE = (
    HERE
    / "runtime"
    / "expanded_search_v2"
    / "seed_271828"
    / "seed_archive.v1.json"
)
RECOVERY_LAUNCHER = HERE / "rec16_checkpoint_recovery.py"

EVIDENCE_ID = "dino_rec16_checkpoint_recovery_20260728_v1"
CANDIDATE_ID = "seed_271828_rec_16"
EXPECTED_EXPANDED_MANIFEST_SHA256 = (
    "9ac29e1aa07167a040d217fdab2d3cfdea0baad690dc95a70f2fe6715908793a"
)
EXPECTED_SEED_ARCHIVE_SHA256 = (
    "a42a989ea27940ea9ae481212a75216c7f23f01602b0c260b6750c9fdb709c9e"
)
EXPECTED_SEED_ARCHIVE_INTERNAL_SHA256 = (
    "eedaa0a37e49cfa86e54be15a56352e4891044856ad5db782f6a4eed464dfb36"
)
EXPECTED_CANDIDATE_RECORD_SHA256 = (
    "7b7b7f05beafe5fde34b86e0a3f3e21a48a04e38ddfb46a7ad435e25d2a0760c"
)
EXPECTED_TRAIN_SPEC_SHA256 = (
    "6f04eab6794cbf8bd707a966ab85b149d7bc24ea4ae238025bb6f3193fca9bf1"
)
EXPECTED_MODEL_SPEC_SHA256 = (
    "bc18216f670d96963ab795be8d6b845f576f4eed17a2482516da91d27eb6248d"
)
EXPECTED_COMMAND_SHA256 = (
    "78174949b50d9a4cf619725a04f844e5f190bf4565716ea1eff2770ec21dd257"
)
EXPECTED_CANDIDATE_SPECS = {
    "model.dec_layers": 3,
    "model.enc_layers": 6,
    "train.optim.lr": 0.0003007572504594793,
    "train.optim.weight_decay": 1.1000000000000001e-05,
}
COMPLETED_RECOVERY_SOURCE = {
    "path": (
        "experiments/dino_moo_phase2_20260728/"
        "rec16_checkpoint_recovery.py"
    ),
    "git_commit": "782821bac5214a45869ba60e7f3ea169e83fa95d",
    "sha256": "e28698627dec1b35a74094deb3e97a3c80cdb5ff47e818f002f5cf2deccc355f",
}
SUPPLEMENTARY_RECOVERY_SOURCE = {
    "path": COMPLETED_RECOVERY_SOURCE["path"],
    "git_commit": "b8dffa1490bbfed88723ef0a5158c94525b6aa60",
    "sha256": "86ecbe1dea1df3ae3a5dbd40a9c4a141b5cbca96ca0bc819d7549cd926ec0349",
}
HISTORICAL = {
    "tao_job_id": "92d8f699-a780-4229-94ba-3520806d75da",
    "slurm_job_id": "30972522",
    "node": "batch-block7-02877",
    "submit_time_utc": "2026-07-28T02:19:00Z",
    "start_time_utc": "2026-07-28T02:19:51Z",
    "end_time_utc": "2026-07-28T02:26:18Z",
    "checkpoint": {
        "path": (
            "/lustre/fs11/portfolios/edgeai/projects/"
            "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/"
            "results/92d8f699-a780-4229-94ba-3520806d75da/results_dir/"
            "train/model_epoch_009_step_00440.pth"
        ),
        "sha256": (
            "4b5ff50181ff919a2796cdd54027fff92eb57c908701a34408d29136d5565b4d"
        ),
        "size_bytes": 506_687_042,
        "epoch": 9,
    },
    "entrypoint": {
        "path": (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/entrypoints/"
            "job_92d8f699-a780-4229-94ba-3520806d75da.sh"
        ),
        "sha256": (
            "051c0fa574a1f7ad2b50560a0ae49f25f0518f748252fdded8bfe01e540ec206"
        ),
        "size_bytes": 80_111,
    },
    "sbatch": {
        "path": (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/sbatch/"
            "job_92d8f699-a780-4229-94ba-3520806d75da.sbatch"
        ),
        "sha256": (
            "cb4faa856fcbbf9b8e1a0d57a1e7117b2210e42af1c5bbcedb5cf6e9bef19e95"
        ),
        "size_bytes": 2_410,
    },
    "remote_specs": {
        "path": (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/specs/"
            "92d8f699-a780-4229-94ba-3520806d75da.json"
        ),
        "sha256": (
            "c3d57e77529b4e2eda38e05e808f5bf3a7add15720ca883ff7a1211cce261b78"
        ),
        "size_bytes": 1_595,
    },
}
ATTEMPT_CONTRACTS = (
    {
        "submission_index": 0,
        "tao_job_id": "7b585a7b-a291-4473-8bda-8e2b542e3982",
        "slurm_job_id": "31002892",
        "submit_time_utc": "2026-07-28T12:06:17Z",
        "start_time_utc": "2026-07-28T12:06:21Z",
        "end_time_utc": "2026-07-28T12:12:57Z",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "node": "batch-block7-00556",
        "checkpoint": {
            "path": (
                "/lustre/fs11/portfolios/edgeai/projects/"
                "edgeai_tao-ptm_image-foundation-model-clip/users/"
                "rarunachalam/results/"
                "7b585a7b-a291-4473-8bda-8e2b542e3982/results_dir/train/"
                "model_epoch_009_step_00440.pth"
            ),
            "sha256": (
                "931bc787eb7b9b1752bd7613558a2e0f1d26ae7cc5a983d8f4333ef59abbd304"
            ),
            "size_bytes": 506_687_042,
            "epoch": 9,
        },
        "entrypoint": {
            "path": (
                "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
                "entrypoints/job_7b585a7b-a291-4473-8bda-8e2b542e3982.sh"
            ),
            "sha256": (
                "a66517b2791ebb292eb159a18c70f74577b7c32e7ace31fbe4762e37aa608496"
            ),
            "size_bytes": 80_111,
        },
        "sbatch": {
            "path": (
                "/lustre/fsw/portfolios/edgeai/users/rarunachalam/sbatch/"
                "job_7b585a7b-a291-4473-8bda-8e2b542e3982.sbatch"
            ),
            "sha256": (
                "d9ce7b7b86f51eb6c99e6cbcc04de2baf5e0a3c306335dad8eab89ae21091c31"
            ),
            "size_bytes": 2_410,
        },
        "remote_specs": {
            "path": (
                "/lustre/fsw/portfolios/edgeai/users/rarunachalam/specs/"
                "7b585a7b-a291-4473-8bda-8e2b542e3982.json"
            ),
            "sha256": (
                "67625dc5d52e8866ac498f6867e8821bfd1d52078922dae88c0fd2f621fcf4fa"
            ),
            "size_bytes": 1_595,
        },
    },
    {
        "submission_index": 1,
        "tao_job_id": "1dd7f4ab-843d-4425-8e41-248386ac9a6b",
        "slurm_job_id": "31002901",
        "submit_time_utc": "2026-07-28T12:07:04Z",
        "start_time_utc": "2026-07-28T12:07:07Z",
        "end_time_utc": "2026-07-28T12:13:39Z",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "node": "batch-block7-02873",
        "checkpoint": {
            "path": (
                "/lustre/fs11/portfolios/edgeai/projects/"
                "edgeai_tao-ptm_image-foundation-model-clip/users/"
                "rarunachalam/results/"
                "1dd7f4ab-843d-4425-8e41-248386ac9a6b/results_dir/train/"
                "model_epoch_009_step_00440.pth"
            ),
            "sha256": (
                "eed8e5f05ec24dd1d62ee4e68ed9c44194f32c24180764c1e0285f76b0d53a35"
            ),
            "size_bytes": 506_687_042,
            "epoch": 9,
        },
        "entrypoint": {
            "path": (
                "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
                "entrypoints/job_1dd7f4ab-843d-4425-8e41-248386ac9a6b.sh"
            ),
            "sha256": (
                "a66517b2791ebb292eb159a18c70f74577b7c32e7ace31fbe4762e37aa608496"
            ),
            "size_bytes": 80_111,
        },
        "sbatch": {
            "path": (
                "/lustre/fsw/portfolios/edgeai/users/rarunachalam/sbatch/"
                "job_1dd7f4ab-843d-4425-8e41-248386ac9a6b.sbatch"
            ),
            "sha256": (
                "4314698b217717d9f61ac35f81ff7df2f6aa137b4aa8bc28156e7c575c5fb299"
            ),
            "size_bytes": 2_410,
        },
        "remote_specs": {
            "path": (
                "/lustre/fsw/portfolios/edgeai/users/rarunachalam/specs/"
                "1dd7f4ab-843d-4425-8e41-248386ac9a6b.json"
            ),
            "sha256": (
                "53d2f8894506ec7a2ad6538e1d03ffe48b7492ac6bd4fb438ce62ec93d603a2a"
            ),
            "size_bytes": 1_595,
        },
    },
)
SUPPLEMENTARY = {
    "tao_job_id": "bc087e0c-e006-4a31-aa7b-228cb7340dbe",
    "slurm_job_id": "31003516",
    "expected_node": "batch-block7-02877",
    "submit_time_utc": "2026-07-28T12:23:26Z",
    "state": "PENDING",
    "entrypoint": {
        "path": (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/entrypoints/"
            "job_bc087e0c-e006-4a31-aa7b-228cb7340dbe.sh"
        ),
        "sha256": (
            "a66517b2791ebb292eb159a18c70f74577b7c32e7ace31fbe4762e37aa608496"
        ),
        "size_bytes": 80_111,
    },
    "sbatch": {
        "path": (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/sbatch/"
            "job_bc087e0c-e006-4a31-aa7b-228cb7340dbe.sbatch"
        ),
        "sha256": (
            "316344ac58bf2c633caee5fc0bd20af5a7dc3b97538b50b4a2ce2a1c0b8bb59d"
        ),
        "size_bytes": 2_448,
    },
    "remote_specs": {
        "path": (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/specs/"
            "bc087e0c-e006-4a31-aa7b-228cb7340dbe.json"
        ),
        "sha256": (
            "32bc9f42c2a349b1e24362f0209ef8fe9be3d5dff182448950f5b770dea220b1"
        ),
        "size_bytes": 1_595,
    },
}
POLICY_KEY = "earliest_submitted_exact_config_recovery_v1"
SELECTION_ISOLATION = {
    "selector_invoked_on_matched_measurements": False,
    "selection_time_objectives_replaced": False,
    "measurements_feed_selection": False,
    "measurements_feed_reselection": False,
    "algorithm_selected_candidate_overridden": False,
}
HEX = frozenset("0123456789abcdef")
UTC_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")


class FreezeError(RuntimeError):
    """Raised when recovery evidence cannot be frozen exactly."""


class RemoteProbe(Protocol):
    """Read-only cluster evidence interface used by the freezer."""

    def scheduler_record(self, slurm_job_id: str) -> dict[str, str]:
        """Return the exact top-level sacct record."""

    def pending_record(self, slurm_job_id: str) -> dict[str, str]:
        """Return the pending scheduler snapshot including its reason."""

    def file_identity(self, path: str) -> dict[str, Any]:
        """Return path, SHA256, and byte size for one remote regular file."""

    def path_absent(self, path: str) -> bool:
        """Return true only when the remote path does not exist."""

    def file_contains_line(self, path: str, line: str) -> bool:
        """Return true only when the remote file contains the exact line."""


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
        raise FreezeError(f"value is not canonical JSON: {error}") from error
    return encoded.encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise FreezeError(
            f"{label} mismatch: expected {expected!r}, got {actual!r}"
        )


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX for character in value)
    ):
        raise FreezeError(f"{label} must be a lowercase SHA256 digest")
    return value


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FreezeError(f"cannot read JSON source {path}: {error}") from error
    if not isinstance(value, dict):
        raise FreezeError(f"JSON source must be an object: {path}")
    return value


def load_env_file(path: Path) -> None:
    if not path.is_file():
        raise FreezeError(f"secrets env file is missing: {path}")
    for number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise FreezeError(f"unsupported env syntax on line {number}")
        key, encoded = line.split("=", 1)
        key = key.strip()
        if (
            not key
            or key[0].isdigit()
            or not key.replace("_", "").isalnum()
        ):
            raise FreezeError(f"invalid env key on line {number}")
        tokens = shlex.split(encoded, comments=True, posix=True)
        if len(tokens) > 1:
            raise FreezeError(f"unsupported env value on line {number}")
        os.environ.setdefault(key, tokens[0] if tokens else "")


class SSHRemoteProbe:
    """Read-only SSH implementation of the remote evidence contract."""

    def __init__(self, target: str, key_path: str | None):
        self.target = target
        self.key_path = key_path

    @classmethod
    def from_environment(cls) -> "SSHRemoteProbe":
        user = os.environ.get("SLURM_USER", "").strip()
        host = os.environ.get("SLURM_HOSTNAME", "").split(",", 1)[0].strip()
        if not user or not host:
            raise FreezeError("SLURM_USER and SLURM_HOSTNAME are required")
        return cls(
            target=f"{user}@{host}",
            key_path=os.environ.get("SSH_KEY_PATH") or None,
        )

    def _run(self, command: str, *, timeout: int = 900) -> str:
        ssh = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
        ]
        if self.key_path:
            ssh.extend(["-i", self.key_path])
        ssh.extend([self.target, command])
        completed = subprocess.run(
            ssh,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise FreezeError(f"remote read-only probe failed: {detail}")
        return completed.stdout

    @staticmethod
    def _timestamp(value: str, label: str) -> str:
        if value in {"", "Unknown", "N/A"}:
            raise FreezeError(f"{label} is not a completed timestamp")
        normalized = value if value.endswith("Z") else f"{value}Z"
        if UTC_PATTERN.fullmatch(normalized) is None:
            raise FreezeError(f"{label} is not an ISO second timestamp")
        return normalized

    def scheduler_record(self, slurm_job_id: str) -> dict[str, str]:
        require_numeric_slurm_id(slurm_job_id)
        output = self._run(
            "sacct -X "
            f"-j {shlex.quote(slurm_job_id)} "
            "--starttime 2026-07-28 "
            "--noheader --parsable2 "
            "-o JobIDRaw,State,ExitCode,NodeList,Submit,Start,End"
        )
        records = []
        for line in output.splitlines():
            fields = line.strip().split("|")
            if len(fields) < 7 or fields[0] != slurm_job_id:
                continue
            records.append(fields[:7])
        if len(records) != 1:
            raise FreezeError(
                f"expected one top-level sacct row for {slurm_job_id}"
            )
        job_id, state, exit_code, node, submit, start, end = records[0]
        return {
            "slurm_job_id": job_id,
            "state": state,
            "exit_code": exit_code,
            "node": node,
            "submit_time_utc": self._timestamp(
                submit,
                f"{job_id} submit",
            ),
            "start_time_utc": self._timestamp(start, f"{job_id} start"),
            "end_time_utc": self._timestamp(end, f"{job_id} end"),
        }

    def pending_record(self, slurm_job_id: str) -> dict[str, str]:
        require_numeric_slurm_id(slurm_job_id)
        sacct = self._run(
            "sacct -X "
            f"-j {shlex.quote(slurm_job_id)} "
            "--starttime 2026-07-28 "
            "--noheader --parsable2 "
            "-o JobIDRaw,State,ExitCode,NodeList,Submit"
        )
        rows = [
            line.strip().split("|")[:5]
            for line in sacct.splitlines()
            if line.strip().split("|", 1)[0] == slurm_job_id
        ]
        if len(rows) != 1 or len(rows[0]) != 5:
            raise FreezeError(
                f"expected one pending sacct row for {slurm_job_id}"
            )
        job_id, state, exit_code, node, submit = rows[0]
        queue = self._run(
            "squeue -h "
            f"-j {shlex.quote(slurm_job_id)} "
            "-o '%i|%T|%R|%N'"
        ).strip()
        fields = queue.split("|")
        if len(fields) != 4 or fields[0] != slurm_job_id:
            raise FreezeError(
                f"pending job {slurm_job_id} is absent from squeue"
            )
        if fields[1] != state:
            raise FreezeError(
                f"sacct/squeue state drift for pending job {slurm_job_id}"
            )
        return {
            "slurm_job_id": job_id,
            "state": state,
            "exit_code": exit_code,
            "node": node,
            "submit_time_utc": self._timestamp(
                submit,
                f"{job_id} submit",
            ),
            "reason": fields[2],
            "squeue_node": fields[3],
        }

    def file_identity(self, path: str) -> dict[str, Any]:
        quoted = shlex.quote(path)
        output = self._run(
            f"test -f {quoted} && "
            f"stat -Lc '%s' {quoted} && "
            f"sha256sum {quoted}",
            timeout=1800,
        ).splitlines()
        if len(output) != 2:
            raise FreezeError(f"invalid remote file evidence for {path}")
        try:
            size_bytes = int(output[0].strip())
        except ValueError as error:
            raise FreezeError(f"invalid remote byte size for {path}") from error
        digest_fields = output[1].split()
        if len(digest_fields) < 2:
            raise FreezeError(f"invalid remote SHA256 evidence for {path}")
        digest = require_sha256(digest_fields[0], f"{path} remote SHA256")
        return {"path": path, "sha256": digest, "size_bytes": size_bytes}

    def path_absent(self, path: str) -> bool:
        output = self._run(
            f"if test -e {shlex.quote(path)}; "
            "then echo PRESENT; else echo ABSENT; fi"
        ).strip()
        if output not in {"PRESENT", "ABSENT"}:
            raise FreezeError(f"invalid path-presence result for {path}")
        return output == "ABSENT"

    def file_contains_line(self, path: str, line: str) -> bool:
        output = self._run(
            f"grep -Fqx -- {shlex.quote(line)} {shlex.quote(path)} "
            "&& echo PRESENT || echo ABSENT"
        ).strip()
        if output not in {"PRESENT", "ABSENT"}:
            raise FreezeError(f"invalid line-presence result for {path}")
        return output == "PRESENT"


def require_numeric_slurm_id(value: Any) -> int:
    if not isinstance(value, str) or not value.isdigit():
        raise FreezeError("SLURM job ID must be an unsigned decimal string")
    return int(value)


def validate_file_identity(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
) -> None:
    identity_keys = ("path", "sha256", "size_bytes")
    require_equal(
        {key: observed.get(key) for key in identity_keys},
        {key: expected.get(key) for key in identity_keys},
        f"{label} file identity",
    )


def git_blob_bytes(commit: str, relative_path: str) -> bytes:
    if (
        not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
    ):
        raise FreezeError("recovery source commit must be a 40-hex git ID")
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY),
            "show",
            f"{commit}:{relative_path}",
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise FreezeError(
            f"cannot read recovery source {relative_path} at {commit}"
        )
    return completed.stdout


def load_recovery_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "dino_rec16_checkpoint_recovery_for_freezer",
        RECOVERY_LAUNCHER,
    )
    if spec is None or spec.loader is None:
        raise FreezeError("cannot import rec16 checkpoint recovery launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_local_source_identity() -> dict[str, Any]:
    require_equal(
        sha256_file(EXPANDED_MANIFEST),
        EXPECTED_EXPANDED_MANIFEST_SHA256,
        "expanded manifest SHA256",
    )
    require_equal(
        sha256_file(SEED_ARCHIVE),
        EXPECTED_SEED_ARCHIVE_SHA256,
        "seed archive whole-file SHA256",
    )
    archive = load_json(SEED_ARCHIVE)
    archive_claim = require_sha256(
        archive.get("archive_sha256"),
        "seed archive internal SHA256",
    )
    unhashed_archive = copy.deepcopy(archive)
    del unhashed_archive["archive_sha256"]
    require_equal(
        sha256_value(unhashed_archive),
        EXPECTED_SEED_ARCHIVE_INTERNAL_SHA256,
        "seed archive canonical SHA256",
    )
    require_equal(
        archive_claim,
        EXPECTED_SEED_ARCHIVE_INTERNAL_SHA256,
        "seed archive claimed SHA256",
    )
    require_equal(
        archive.get("manifest_file_sha256"),
        EXPECTED_EXPANDED_MANIFEST_SHA256,
        "seed archive expanded-manifest binding",
    )
    require_equal(archive.get("status"), "complete", "seed archive status")
    require_equal(archive.get("search_seed"), 271828, "seed archive seed")
    require_equal(
        archive.get("manual_candidate_injection_used"),
        False,
        "seed archive manual injection",
    )
    record = archive.get("records", {}).get(CANDIDATE_ID)
    if not isinstance(record, dict):
        raise FreezeError("rec16 is missing from the sealed seed archive")
    require_equal(
        sha256_value(record),
        EXPECTED_CANDIDATE_RECORD_SHA256,
        "rec16 sealed candidate-record SHA256",
    )
    for key, expected in (
        ("candidate_id", CANDIDATE_ID),
        ("search_seed", 271828),
        ("training_seed", 1234),
        ("rec_id", 16),
        ("specs", EXPECTED_CANDIDATE_SPECS),
        ("resolved_train_spec_sha256", EXPECTED_TRAIN_SPEC_SHA256),
        ("resolved_model_spec_sha256", EXPECTED_MODEL_SPEC_SHA256),
        ("train_job_id", HISTORICAL["tao_job_id"]),
    ):
        require_equal(record.get(key), expected, f"sealed rec16 {key}")
    require_equal(
        record.get("checkpoint"),
        HISTORICAL["checkpoint"],
        "sealed historical checkpoint identity",
    )

    for label, source in (
        ("completed recovery launcher", COMPLETED_RECOVERY_SOURCE),
        ("supplementary recovery launcher", SUPPLEMENTARY_RECOVERY_SOURCE),
    ):
        blob = git_blob_bytes(source["git_commit"], source["path"])
        require_equal(
            hashlib.sha256(blob).hexdigest(),
            source["sha256"],
            f"{label} source SHA256",
        )
    require_equal(
        sha256_file(RECOVERY_LAUNCHER),
        SUPPLEMENTARY_RECOVERY_SOURCE["sha256"],
        "current supplementary recovery launcher SHA256",
    )

    recovery = load_recovery_module()
    _, _, command, reconstruction = recovery.build_contract(None)
    require_equal(
        hashlib.sha256(command.encode("utf-8")).hexdigest(),
        EXPECTED_COMMAND_SHA256,
        "reconstructed recovery command SHA256",
    )
    require_equal(
        reconstruction["original_training"]["train_spec_sha256"],
        EXPECTED_TRAIN_SPEC_SHA256,
        "reconstructed train-spec SHA256",
    )
    require_equal(
        reconstruction["original_training"]["model_spec_sha256"],
        EXPECTED_MODEL_SPEC_SHA256,
        "reconstructed model-spec SHA256",
    )
    require_equal(
        reconstruction["reconstruction"]["candidate_specs"],
        EXPECTED_CANDIDATE_SPECS,
        "reconstructed candidate specs",
    )
    return {
        "expanded_manifest": {
            "path": str(EXPANDED_MANIFEST),
            "sha256": EXPECTED_EXPANDED_MANIFEST_SHA256,
        },
        "seed_archive": {
            "path": str(SEED_ARCHIVE),
            "whole_file_sha256": EXPECTED_SEED_ARCHIVE_SHA256,
            "internal_sha256": EXPECTED_SEED_ARCHIVE_INTERNAL_SHA256,
        },
        "candidate_record_sha256": EXPECTED_CANDIDATE_RECORD_SHA256,
        "candidate_specs": copy.deepcopy(EXPECTED_CANDIDATE_SPECS),
        "training_seed": 1234,
        "train_spec_sha256": EXPECTED_TRAIN_SPEC_SHA256,
        "model_spec_sha256": EXPECTED_MODEL_SPEC_SHA256,
        "command_sha256": EXPECTED_COMMAND_SHA256,
        "completed_recovery_launcher": copy.deepcopy(
            COMPLETED_RECOVERY_SOURCE
        ),
        "supplementary_recovery_launcher": copy.deepcopy(
            SUPPLEMENTARY_RECOVERY_SOURCE
        ),
    }


def validate_completed_attempt(
    probe: RemoteProbe,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    scheduler = probe.scheduler_record(contract["slurm_job_id"])
    expected_scheduler = {
        key: contract[key]
        for key in (
            "slurm_job_id",
            "state",
            "exit_code",
            "node",
            "submit_time_utc",
            "start_time_utc",
            "end_time_utc",
        )
    }
    require_equal(
        scheduler,
        expected_scheduler,
        f"recovery attempt {contract['submission_index']} sacct",
    )
    for label in ("checkpoint", "entrypoint", "sbatch", "remote_specs"):
        validate_file_identity(
            probe.file_identity(contract[label]["path"]),
            contract[label],
            f"recovery attempt {contract['submission_index']} {label}",
        )
    checkpoint = copy.deepcopy(contract["checkpoint"])
    historical_match = (
        checkpoint["sha256"] == HISTORICAL["checkpoint"]["sha256"]
    )
    require_equal(
        historical_match,
        False,
        f"recovery attempt {contract['submission_index']} historical hash",
    )
    return {
        "submission_index": contract["submission_index"],
        "tao_job_id": contract["tao_job_id"],
        "slurm_job_id": contract["slurm_job_id"],
        "submit_time_utc": contract["submit_time_utc"],
        "start_time_utc": contract["start_time_utc"],
        "end_time_utc": contract["end_time_utc"],
        "state": contract["state"],
        "exit_code": contract["exit_code"],
        "node": contract["node"],
        "checkpoint": checkpoint,
        "train_spec_sha256": EXPECTED_TRAIN_SPEC_SHA256,
        "model_spec_sha256": EXPECTED_MODEL_SPEC_SHA256,
        "command_sha256": EXPECTED_COMMAND_SHA256,
        "exact_config": True,
        "checkpoint_origin": "exact_config_retrain",
        "historical_checkpoint_sha256_match": historical_match,
        "entrypoint": copy.deepcopy(contract["entrypoint"]),
        "sbatch": copy.deepcopy(contract["sbatch"]),
        "remote_specs": copy.deepcopy(contract["remote_specs"]),
    }


def selection_key(attempt: Mapping[str, Any]) -> tuple[str, int, str]:
    submit = attempt.get("submit_time_utc")
    if not isinstance(submit, str) or UTC_PATTERN.fullmatch(submit) is None:
        raise FreezeError("attempt submit time must be a UTC second timestamp")
    tao_job_id = attempt.get("tao_job_id")
    if not isinstance(tao_job_id, str) or not tao_job_id:
        raise FreezeError("attempt TAO job ID must be a non-empty string")
    return (
        submit,
        require_numeric_slurm_id(attempt.get("slurm_job_id")),
        tao_job_id,
    )


def choose_recovery(
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(attempts) != 2:
        raise FreezeError("exactly two completed recovery attempts are required")
    for attempt in attempts:
        require_equal(
            {
                "state": attempt.get("state"),
                "exit_code": attempt.get("exit_code"),
                "exact_config": attempt.get("exact_config"),
            },
            {
                "state": "COMPLETED",
                "exit_code": "0:0",
                "exact_config": True,
            },
            "recovery-attempt eligibility",
        )
    return min(attempts, key=selection_key)


def validate_historical_remote(probe: RemoteProbe) -> dict[str, Any]:
    scheduler = probe.scheduler_record(HISTORICAL["slurm_job_id"])
    require_equal(
        scheduler,
        {
            key: HISTORICAL[key]
            for key in (
                "slurm_job_id",
                "node",
                "submit_time_utc",
                "start_time_utc",
                "end_time_utc",
            )
        }
        | {"state": "COMPLETED", "exit_code": "0:0"},
        "historical training sacct",
    )
    for label in ("entrypoint", "sbatch", "remote_specs"):
        validate_file_identity(
            probe.file_identity(HISTORICAL[label]["path"]),
            HISTORICAL[label],
            f"historical {label}",
        )
    if not probe.path_absent(HISTORICAL["checkpoint"]["path"]):
        raise FreezeError(
            "historical checkpoint unexpectedly exists; refuse recovery overlay"
        )
    return {
        "tao_job_id": HISTORICAL["tao_job_id"],
        "slurm_job_id": HISTORICAL["slurm_job_id"],
        "node": HISTORICAL["node"],
        "checkpoint": copy.deepcopy(HISTORICAL["checkpoint"]),
        "remote_checkpoint_status": "missing",
        "historical_identity_preserved": True,
        "replacement_of_historical_bytes_permitted": False,
        "entrypoint": copy.deepcopy(HISTORICAL["entrypoint"]),
        "sbatch": copy.deepcopy(HISTORICAL["sbatch"]),
        "remote_specs": copy.deepcopy(HISTORICAL["remote_specs"]),
    }


def validate_supplementary_pending(probe: RemoteProbe) -> dict[str, Any]:
    observed = probe.pending_record(SUPPLEMENTARY["slurm_job_id"])
    require_equal(
        {
            key: observed.get(key)
            for key in ("slurm_job_id", "state", "submit_time_utc")
        },
        {
            "slurm_job_id": SUPPLEMENTARY["slurm_job_id"],
            "state": "PENDING",
            "submit_time_utc": SUPPLEMENTARY["submit_time_utc"],
        },
        "supplementary exact-node scheduler identity",
    )
    if observed.get("reason") != "(Resources)":
        raise FreezeError(
            "supplementary exact-node replay is not pending for resources"
        )
    for label in ("entrypoint", "sbatch", "remote_specs"):
        validate_file_identity(
            probe.file_identity(SUPPLEMENTARY[label]["path"]),
            SUPPLEMENTARY[label],
            f"supplementary exact-node {label}",
        )
    expected_line = f"#SBATCH --nodelist={SUPPLEMENTARY['expected_node']}"
    if not probe.file_contains_line(
        SUPPLEMENTARY["sbatch"]["path"],
        expected_line,
    ):
        raise FreezeError("supplementary replay lacks its exact node pin")
    return {
        "tao_job_id": SUPPLEMENTARY["tao_job_id"],
        "slurm_job_id": SUPPLEMENTARY["slurm_job_id"],
        "submit_time_utc": SUPPLEMENTARY["submit_time_utc"],
        "state": "PENDING",
        "pending_reason": observed["reason"],
        "expected_node": SUPPLEMENTARY["expected_node"],
        "exact_config": True,
        "non_gating": True,
        "included_in_selection_policy": False,
        "selected_recovery_can_change": False,
        "source": copy.deepcopy(SUPPLEMENTARY_RECOVERY_SOURCE),
        "entrypoint": copy.deepcopy(SUPPLEMENTARY["entrypoint"]),
        "sbatch": copy.deepcopy(SUPPLEMENTARY["sbatch"]),
        "remote_specs": copy.deepcopy(SUPPLEMENTARY["remote_specs"]),
    }


def build_evidence(
    probe: RemoteProbe,
    source_identity: dict[str, Any],
    *,
    frozen_at_utc: str,
) -> dict[str, Any]:
    if UTC_PATTERN.fullmatch(frozen_at_utc) is None:
        raise FreezeError("frozen_at_utc must be a UTC second timestamp")
    historical = validate_historical_remote(probe)
    attempts = [
        validate_completed_attempt(probe, contract)
        for contract in ATTEMPT_CONTRACTS
    ]
    require_equal(
        [item["submission_index"] for item in attempts],
        [0, 1],
        "recovery attempt submission order",
    )
    chosen = choose_recovery(attempts)
    require_equal(
        chosen["submission_index"],
        0,
        "value-independent selected recovery index",
    )
    supplementary = validate_supplementary_pending(probe)
    unique_hashes = {
        historical["checkpoint"]["sha256"],
        *(attempt["checkpoint"]["sha256"] for attempt in attempts),
    }
    require_equal(
        len(unique_hashes),
        3,
        "historical/recovery checkpoint byte identities",
    )
    selected_checkpoint = copy.deepcopy(chosen["checkpoint"])
    evidence = {
        "schema_version": 1,
        "evidence_id": EVIDENCE_ID,
        "status": "complete",
        "frozen_at_utc": frozen_at_utc,
        "candidate_id": CANDIDATE_ID,
        "source_identity": copy.deepcopy(source_identity),
        "historical_checkpoint": historical,
        "recovery_attempts": attempts,
        "selection_policy": {
            "policy_key": POLICY_KEY,
            "eligible_attempt_predicate": (
                "exact_config is true and state is COMPLETED and "
                "exit_code is 0:0"
            ),
            "sort_key": [
                "submit_time_utc",
                "numeric_slurm_job_id",
                "tao_job_id",
            ],
            "ascending": True,
            "value_independent": True,
            "checkpoint_hash_used": False,
            "checkpoint_size_used": False,
            "objective_value_used": False,
            "selected_submission_index": chosen["submission_index"],
        },
        "selected_recovery": {
            "policy_key": POLICY_KEY,
            "submission_index": chosen["submission_index"],
            "tao_job_id": chosen["tao_job_id"],
            "slurm_job_id": chosen["slurm_job_id"],
            "checkpoint": selected_checkpoint,
            "checkpoint_origin": "exact_config_retrain",
            "exact_config": True,
            "historical_checkpoint_sha256_match": False,
            "byte_identical_to_historical": False,
            "configuration_exact_not_byte_identical": True,
            "validation_only": True,
        },
        "supplementary_exact_node_replay": supplementary,
        "selection_isolation": copy.deepcopy(SELECTION_ISOLATION),
    }
    evidence["evidence_sha256"] = sha256_value(evidence)
    validate_evidence(evidence)
    return evidence


def validate_evidence(evidence: dict[str, Any]) -> None:
    require_equal(
        set(evidence),
        {
            "schema_version",
            "evidence_id",
            "status",
            "frozen_at_utc",
            "candidate_id",
            "source_identity",
            "historical_checkpoint",
            "recovery_attempts",
            "selection_policy",
            "selected_recovery",
            "supplementary_exact_node_replay",
            "selection_isolation",
            "evidence_sha256",
        },
        "recovery evidence top-level shape",
    )
    require_equal(evidence["schema_version"], 1, "evidence schema")
    require_equal(evidence["evidence_id"], EVIDENCE_ID, "evidence ID")
    require_equal(evidence["status"], "complete", "evidence status")
    require_equal(evidence["candidate_id"], CANDIDATE_ID, "candidate ID")
    require_equal(
        evidence["selection_isolation"],
        SELECTION_ISOLATION,
        "selection-isolation flags",
    )
    require_equal(
        len(evidence["selection_isolation"]),
        5,
        "selection-isolation flag count",
    )
    require_equal(
        set(evidence["selection_isolation"].values()),
        {False},
        "selection-isolation false values",
    )
    selected = choose_recovery(evidence["recovery_attempts"])
    require_equal(
        evidence["selected_recovery"]["tao_job_id"],
        selected["tao_job_id"],
        "selected recovery TAO ID",
    )
    require_equal(
        evidence["selected_recovery"]["checkpoint"],
        selected["checkpoint"],
        "selected recovery checkpoint",
    )
    require_equal(
        evidence["selected_recovery"][
            "historical_checkpoint_sha256_match"
        ],
        False,
        "selected historical checkpoint hash match",
    )
    require_equal(
        evidence["supplementary_exact_node_replay"]["non_gating"],
        True,
        "supplementary non-gating flag",
    )
    require_equal(
        evidence["supplementary_exact_node_replay"]["state"],
        "PENDING",
        "supplementary pending state",
    )
    claimed = require_sha256(
        evidence["evidence_sha256"],
        "recovery evidence internal SHA256",
    )
    unhashed = copy.deepcopy(evidence)
    del unhashed["evidence_sha256"]
    require_equal(
        sha256_value(unhashed),
        claimed,
        "recovery evidence canonical SHA256",
    )


def write_new_evidence(path: Path, evidence: dict[str, Any]) -> None:
    """Create one immutable evidence file and refuse every overwrite."""

    validate_evidence(evidence)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".pending",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                evidence,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FreezeError(
                f"refusing to overwrite immutable evidence: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--secrets-env",
        type=Path,
        default=DEFAULT_SECRETS_ENV,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify and print evidence without creating --output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_identity = validate_local_source_identity()
    load_env_file(args.secrets_env)
    probe = SSHRemoteProbe.from_environment()
    evidence = build_evidence(
        probe,
        source_identity,
        frozen_at_utc=utc_now(),
    )
    print(json.dumps(evidence, indent=2, sort_keys=True), flush=True)
    if not args.dry_run:
        write_new_evidence(args.output.resolve(), evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
