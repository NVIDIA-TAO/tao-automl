#!/usr/bin/env python3

"""Read-only verification for the published Grounding DINO dataset view."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tao_automl.ptm_registry import canonical_sha256

try:
    from .contract import PreparationError, read_json, sha256_file
except ImportError:  # pragma: no cover - direct script execution
    from contract import PreparationError, read_json, sha256_file


HERE = Path(__file__).resolve().parent
DEFAULT_CONVERSION_MANIFEST = HERE / "dataset_conversion.v1.json"
DEFAULT_OUTPUT = HERE / "dataset_stage.v1.json"


def _remote_file_records(
    *,
    remote: str,
    paths: list[str],
    ssh_key_path: str | None,
) -> dict[str, dict[str, Any]]:
    ssh = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
    ]
    if ssh_key_path:
        ssh.extend(["-i", ssh_key_path])
    command = " && ".join(
        [
            *(f"test -r {shlex.quote(path)}" for path in paths),
            "sha256sum " + " ".join(shlex.quote(path) for path in paths),
            "stat -c '%n|%s|%a' "
            + " ".join(shlex.quote(path) for path in paths),
        ]
    )
    output = subprocess.run(
        [*ssh, remote, command],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.splitlines()
    hash_lines = output[: len(paths)]
    stat_lines = output[len(paths) :]
    hashes = {}
    for line in hash_lines:
        sha256, path = line.split(maxsplit=1)
        hashes[path] = sha256
    stats = {}
    for line in stat_lines:
        path, size, mode = line.rsplit("|", 2)
        stats[path] = (int(size), mode)
    if set(hashes) != set(paths) or set(stats) != set(paths):
        raise PreparationError("remote dataset verification output is incomplete")
    return {
        path: {
            "path": path,
            "sha256": hashes[path],
            "size_bytes": stats[path][0],
            "mode": stats[path][1],
        }
        for path in paths
    }


def build_stage_record(
    *,
    conversion_manifest_path: str | Path,
    remote_user: str,
    remote_host: str,
    ssh_key_path: str | None,
) -> dict[str, Any]:
    conversion_manifest_path = Path(conversion_manifest_path).resolve()
    conversion = read_json(conversion_manifest_path)
    root = conversion["staging"]["lustre_root"]
    expected = {
        name: {
            "path": value["lustre_path"],
            "sha256": value["sha256"],
            "size_bytes": value["size_bytes"],
        }
        for name, value in conversion["canonical_outputs"].items()
    }
    expected["conversion_manifest"] = {
        "path": str(Path(root) / "conversion_manifest.v1.json"),
        "sha256": sha256_file(conversion_manifest_path),
        "size_bytes": conversion_manifest_path.stat().st_size,
    }
    path_to_name = {
        record["path"]: name for name, record in expected.items()
    }
    observed_by_path = _remote_file_records(
        remote=f"{remote_user}@{remote_host}",
        paths=list(path_to_name),
        ssh_key_path=ssh_key_path,
    )
    observed = {
        path_to_name[path]: record
        for path, record in observed_by_path.items()
    }
    for name, expected_record in expected.items():
        for field in ("path", "sha256", "size_bytes"):
            if observed[name][field] != expected_record[field]:
                raise PreparationError(
                    f"published {name} {field} differs from sealed conversion"
                )
        if int(observed[name]["mode"], 8) & 0o222:
            raise PreparationError(f"published {name} remains writable")

    record = {
        "schema_version": 1,
        "conversion_manifest": {
            "path": str(conversion_manifest_path),
            "sha256": sha256_file(conversion_manifest_path),
            "semantic_sha256": conversion["manifest_sha256"],
        },
        "publication": {
            "lustre_root": root,
            "remote_user": remote_user,
            "remote_host": remote_host,
            "method": "temporary_sibling_then_target_absent_atomic_rename",
            "inside_existing_source_dataset_tree": True,
            "all_expected_files_present": True,
            "all_hashes_and_sizes_match": True,
            "published_files_nonwritable": True,
            "expected": expected,
            "observed": observed,
        },
        "selection_or_execution": {
            "selector_invoked": False,
            "model_execution_performed": False,
            "scheduler_jobs_submitted": 0,
        },
    }
    record["stage_record_sha256"] = canonical_sha256(record)
    return record


def validate_stage_record(record: Mapping[str, Any]) -> None:
    publication = record.get("publication", {})
    if publication.get("all_expected_files_present") is not True:
        raise PreparationError("published dataset is incomplete")
    if publication.get("all_hashes_and_sizes_match") is not True:
        raise PreparationError("published dataset identity differs")
    if publication.get("published_files_nonwritable") is not True:
        raise PreparationError("published dataset is mutable")
    expected = publication.get("expected")
    observed = publication.get("observed")
    if not isinstance(expected, Mapping) or not isinstance(observed, Mapping):
        raise PreparationError("stage evidence is malformed")
    if set(expected) != set(observed):
        raise PreparationError("stage expected/observed keys differ")
    for name in expected:
        for field in ("path", "sha256", "size_bytes"):
            if expected[name][field] != observed[name][field]:
                raise PreparationError(f"stage record differs for {name}/{field}")
    if record.get("selection_or_execution") != {
        "selector_invoked": False,
        "model_execution_performed": False,
        "scheduler_jobs_submitted": 0,
    }:
        raise PreparationError("stage record may not claim model execution")
    expected_record = copy.deepcopy(dict(record))
    observed_sha = expected_record.pop("stage_record_sha256", None)
    if observed_sha != canonical_sha256(expected_record):
        raise PreparationError("stage_record_sha256 does not match content")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conversion-manifest",
        type=Path,
        default=DEFAULT_CONVERSION_MANIFEST,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--remote-user", default=os.environ.get("SLURM_USER"))
    parser.add_argument(
        "--remote-host",
        default=(os.environ.get("SLURM_HOSTNAME", "").split(",")[0] or None),
    )
    parser.add_argument(
        "--ssh-key-path",
        default=os.environ.get("SSH_KEY_PATH"),
    )
    arguments = parser.parse_args()
    if not arguments.remote_user or not arguments.remote_host:
        raise PreparationError("SLURM_USER and SLURM_HOSTNAME are required")

    record = build_stage_record(
        conversion_manifest_path=arguments.conversion_manifest,
        remote_user=arguments.remote_user,
        remote_host=arguments.remote_host,
        ssh_key_path=arguments.ssh_key_path,
    )
    validate_stage_record(record)
    arguments.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "all_hashes_and_sizes_match": True,
                "scheduler_jobs_submitted": 0,
                "stage_record_sha256": record["stage_record_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
