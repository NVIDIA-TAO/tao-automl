#!/usr/bin/env python3

"""Stage immutable Grounding DINO runtime inputs before GPU allocation.

This is intentionally a data-only operation.  It resolves and downloads exact
NGC members plus an immutable Hugging Face BERT snapshot, verifies every byte,
and publishes read-only files on Lustre.  It never imports a model framework,
loads a checkpoint, constructs an SDK job, or mutates the scheduler.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from tao_automl.ptm_preflight import (
    AtomicArtifactCache,
    NGCCredential,
    NGCHTTPSClient,
)
from tao_automl.ptm_registry import canonical_sha256

try:
    from .contract import (
        MODEL_ID,
        PreparationError,
        derive_official_ptms,
        read_json,
        sha256_file,
    )
except ImportError:  # pragma: no cover - direct script execution
    from contract import (
        MODEL_ID,
        PreparationError,
        derive_official_ptms,
        read_json,
        sha256_file,
    )


HERE = Path(__file__).resolve().parent
DEFAULT_INPUTS = HERE / "campaign.inputs.v3.json"
DEFAULT_OUTPUT = HERE / "runtime_inputs.stage.v1.json"
HF_REPOSITORY = "google-bert/bert-base-uncased"
HF_REQUIRED_FILES = (
    "config.json",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run(
    command: Sequence[str],
    *,
    timeout: float,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=True,
        capture_output=capture_output,
        text=True,
        timeout=timeout,
    )


def _ssh_target(inputs: Mapping[str, Any]) -> str:
    routing = inputs["runtime"]["ssh"]
    return f"{routing['user']}@{routing['hostname']}"


def _ssh_options(inputs: Mapping[str, Any]) -> list[str]:
    options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    key = inputs["runtime"]["ssh"].get("key_path")
    if key:
        options.extend(["-i", str(key)])
    return options


def _ssh(
    inputs: Mapping[str, Any],
    remote_command: str,
    *,
    timeout: float = 1800,
) -> str:
    result = _run(
        [
            "ssh",
            *_ssh_options(inputs),
            _ssh_target(inputs),
            remote_command,
        ],
        timeout=timeout,
    )
    return result.stdout


def _remote_file_identity(
    inputs: Mapping[str, Any],
    path: str,
    *,
    timeout: float = 1800,
) -> dict[str, Any] | None:
    quoted = shlex.quote(path)
    output = _ssh(
        inputs,
        (
            f"if test -f {quoted}; then "
            f"stat -c '%s %a' {quoted}; sha256sum {quoted}; "
            "else printf 'ABSENT\\n'; fi"
        ),
        timeout=timeout,
    ).strip().splitlines()
    if output == ["ABSENT"]:
        return None
    if len(output) != 2:
        raise PreparationError(f"remote identity probe was incomplete for {path}")
    size_text, mode = output[0].split()
    digest = output[1].split()[0]
    try:
        size = int(size_text)
    except ValueError as exc:
        raise PreparationError(f"remote size was invalid for {path}") from exc
    if size <= 0 or _SHA256_RE.fullmatch(digest) is None:
        raise PreparationError(f"remote identity was invalid for {path}")
    return {"path": path, "size_bytes": size, "sha256": digest, "mode": mode}


def _publish_file(
    inputs: Mapping[str, Any],
    source: Path,
    destination: str,
    *,
    expected_size: int,
    expected_sha256: str,
) -> dict[str, Any]:
    """Publish one verified file without exposing a partial final pathname."""
    if not source.is_file():
        raise PreparationError(f"local staged source is absent: {source}")
    if source.stat().st_size != expected_size:
        raise PreparationError(f"local staged source size differs: {source}")
    if sha256_file(source) != expected_sha256:
        raise PreparationError(f"local staged source hash differs: {source}")

    existing = _remote_file_identity(inputs, destination)
    if existing is not None:
        if (
            existing["size_bytes"] != expected_size
            or existing["sha256"] != expected_sha256
        ):
            raise PreparationError(
                f"existing immutable destination differs: {destination}"
            )
        if int(existing["mode"], 8) & 0o222:
            raise PreparationError(
                f"existing immutable destination is writable: {destination}"
            )
        existing["cache_hit"] = True
        return existing

    destination_path = Path(destination)
    temporary = (
        destination_path.parent
        / f".{destination_path.name}.partial-{uuid.uuid4().hex}"
    )
    _ssh(
        inputs,
        f"mkdir -p {shlex.quote(str(destination_path.parent))}",
        timeout=60,
    )
    _run(
        [
            "scp",
            *_ssh_options(inputs),
            str(source),
            f"{_ssh_target(inputs)}:{temporary}",
        ],
        timeout=7200,
        capture_output=True,
    )
    quoted_temporary = shlex.quote(str(temporary))
    quoted_destination = shlex.quote(destination)
    _ssh(
        inputs,
        " && ".join(
            [
                f"test \"$(stat -c '%s' {quoted_temporary})\" = "
                f"{shlex.quote(str(expected_size))}",
                f"test \"$(sha256sum {quoted_temporary} | "
                f"awk '{{print $1}}')\" = {shlex.quote(expected_sha256)}",
                f"chmod a-w {quoted_temporary}",
                f"test ! -e {quoted_destination}",
                f"mv {quoted_temporary} {quoted_destination}",
            ]
        ),
        timeout=1800,
    )
    identity = _remote_file_identity(inputs, destination)
    if (
        identity is None
        or identity["size_bytes"] != expected_size
        or identity["sha256"] != expected_sha256
        or int(identity["mode"], 8) & 0o222
    ):
        raise PreparationError(
            f"published destination verification failed: {destination}"
        )
    identity["cache_hit"] = False
    return identity


def _resolve_hf_contract(token: str | None) -> dict[str, Any]:
    api = HfApi(token=token)
    info = api.model_info(HF_REPOSITORY, files_metadata=True)
    if not isinstance(info.sha, str) or not re.fullmatch(r"[0-9a-f]{40}", info.sha):
        raise PreparationError("Hugging Face model did not resolve to a commit")
    siblings = {
        item.rfilename: item
        for item in info.siblings
        if item.rfilename in HF_REQUIRED_FILES
    }
    missing = sorted(set(HF_REQUIRED_FILES) - set(siblings))
    if missing:
        raise PreparationError(
            "BERT repository is missing required files: " + ", ".join(missing)
        )
    files = []
    for name in HF_REQUIRED_FILES:
        item = siblings[name]
        if not isinstance(item.size, int) or item.size <= 0:
            raise PreparationError(f"Hugging Face size is missing for {name}")
        files.append(
            {
                "path": name,
                "expected_size_bytes": item.size,
                "blob_id": item.blob_id,
            }
        )
    return {
        "provider": "huggingface",
        "repository": info.id,
        "requested_repository": HF_REPOSITORY,
        "revision": info.sha,
        "files": files,
    }


def _download_hf_contract(
    contract: Mapping[str, Any],
    *,
    token: str | None,
    local_root: Path,
) -> Path:
    destination = local_root / contract["revision"]
    downloaded = Path(
        snapshot_download(
            repo_id=contract["repository"],
            revision=contract["revision"],
            allow_patterns=[item["path"] for item in contract["files"]],
            local_dir=destination,
            token=token,
        )
    ).resolve()
    if downloaded != destination.resolve():
        raise PreparationError("Hugging Face snapshot escaped the local cache root")
    return downloaded


def stage_runtime_inputs(
    *,
    inputs: Mapping[str, Any],
    local_cache_root: str | Path,
) -> dict[str, Any]:
    """Download, verify, and publish all non-dataset runtime inputs."""
    local_root = Path(local_cache_root).expanduser().resolve()
    local_root.mkdir(parents=True, exist_ok=True)
    credential = NGCCredential.from_environment()
    ngc_client = NGCHTTPSClient(credential)
    cache = AtomicArtifactCache(local_root / "ngc")

    ptm_rows = []
    checkpoint_root = Path(inputs["ptm_staging"]["checkpoint_root"])
    for record in derive_official_ptms():
        reference = ngc_client.resolve_member(record["source"])
        probe = ngc_client.probe_member(reference)
        if (
            probe.ok is not True
            or probe.remote_size_bytes != record["expected_size_bytes"]
        ):
            raise PreparationError(
                f"exact NGC member preflight failed for {record['id']}: "
                f"{probe.code}"
            )
        artifact = cache.fetch_ngc_member(
            checkpoint_id=record["id"],
            reference=reference,
            expected_size_bytes=record["expected_size_bytes"],
            expected_sha256=record.get("sha256"),
            client=ngc_client,
        )
        destination = str(
            checkpoint_root / record["id"] / record["source"]["member"]
        )
        published = _publish_file(
            inputs,
            artifact.path,
            destination,
            expected_size=artifact.size_bytes,
            expected_sha256=artifact.sha256,
        )
        ptm_rows.append(
            {
                "id": record["id"],
                "source": copy.deepcopy(record["source"]),
                "source_identity_sha256": artifact.source_identity_sha256,
                "registry_expected_sha256": record.get("sha256"),
                "registry_expected_size_bytes": record["expected_size_bytes"],
                "observed_sha256": artifact.sha256,
                "verification_mode": artifact.verification_mode,
                "local_cache_hit": artifact.cache_hit,
                "lustre": published,
            }
        )

    hf_token = os.environ.get("HF_TOKEN")
    hf_contract = _resolve_hf_contract(hf_token)
    hf_local = _download_hf_contract(
        hf_contract,
        token=hf_token,
        local_root=local_root / "huggingface" / "bert-base-uncased",
    )
    hf_lustre_root = (
        Path(inputs["ptm_staging"]["bert_cache_root"])
        / hf_contract["revision"]
    )
    hf_files = []
    for item in hf_contract["files"]:
        local_file = hf_local / item["path"]
        size = local_file.stat().st_size
        if size != item["expected_size_bytes"]:
            raise PreparationError(
                f"Hugging Face file size differs for {item['path']}"
            )
        digest = sha256_file(local_file)
        remote = _publish_file(
            inputs,
            local_file,
            str(hf_lustre_root / item["path"]),
            expected_size=size,
            expected_sha256=digest,
        )
        hf_files.append(
            {
                **copy.deepcopy(item),
                "sha256": digest,
                "lustre": remote,
            }
        )
    hf_contract = {
        **hf_contract,
        "lustre_root": str(hf_lustre_root),
        "files": hf_files,
        "tree_sha256": canonical_sha256(
            [
                {
                    "path": item["path"],
                    "size_bytes": item["lustre"]["size_bytes"],
                    "sha256": item["sha256"],
                }
                for item in hf_files
            ]
        ),
        "offline_runtime": {
            "model_text_encoder_type": str(hf_lustre_root),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    }

    runtime = inputs["runtime"]
    document = {
        "schema_version": 1,
        "created_at_utc": _utc_timestamp(),
        "model": MODEL_ID,
        "source_registry_policy": (
            "all_repository_records_with_source.official_true"
        ),
        "official_ptms": ptm_rows,
        "text_encoder": hf_contract,
        "runtime": {
            "sqsh_path": runtime["sqsh_path"],
            "sqsh_sha256": runtime["sqsh_sha256"],
            "sqsh_size_bytes": runtime["sqsh_size_bytes"],
            "skill_revision": runtime["skill_revision"],
            "sdk_revision": runtime["sdk_revision"],
        },
        "execution": {
            "cpu_model_runs": 0,
            "gpu_model_runs": 0,
            "scheduler_jobs_submitted": 0,
            "checkpoint_loads": 0,
            "operation": "data_only_download_checksum_and_lustre_publication",
        },
    }
    document["stage_record_sha256"] = canonical_sha256(document)
    validate_runtime_input_stage(document, inputs=inputs)
    return document


def validate_runtime_input_stage(
    document: Mapping[str, Any],
    *,
    inputs: Mapping[str, Any],
) -> None:
    if document.get("schema_version") != 1 or document.get("model") != MODEL_ID:
        raise PreparationError("runtime input stage schema or model differs")
    expected_ids = [record["id"] for record in derive_official_ptms()]
    rows = document.get("official_ptms")
    if not isinstance(rows, list) or [item.get("id") for item in rows] != expected_ids:
        raise PreparationError("runtime input stage PTM inventory differs")
    for item, record in zip(rows, derive_official_ptms(), strict=True):
        lustre = item.get("lustre", {})
        if lustre.get("size_bytes") != record["expected_size_bytes"]:
            raise PreparationError(f"staged PTM size differs for {record['id']}")
        if not _SHA256_RE.fullmatch(str(lustre.get("sha256", ""))):
            raise PreparationError(f"staged PTM hash is invalid for {record['id']}")
        if item.get("observed_sha256") != lustre.get("sha256"):
            raise PreparationError(f"local/remote PTM hash differs for {record['id']}")
        if record.get("sha256") is not None and item["observed_sha256"] != record["sha256"]:
            raise PreparationError(f"registered PTM hash differs for {record['id']}")
        if int(str(lustre.get("mode", "0")), 8) & 0o222:
            raise PreparationError(f"staged PTM is writable for {record['id']}")
    text = document.get("text_encoder", {})
    if text.get("repository") != HF_REPOSITORY:
        raise PreparationError("BERT repository identity differs")
    if not re.fullmatch(r"[0-9a-f]{40}", str(text.get("revision", ""))):
        raise PreparationError("BERT revision is not immutable")
    files = text.get("files")
    if not isinstance(files, list) or [item.get("path") for item in files] != list(
        HF_REQUIRED_FILES
    ):
        raise PreparationError("BERT required-file contract differs")
    for item in files:
        if item["expected_size_bytes"] != item["lustre"]["size_bytes"]:
            raise PreparationError(f"BERT size differs for {item['path']}")
        if item["sha256"] != item["lustre"]["sha256"]:
            raise PreparationError(f"BERT hash differs for {item['path']}")
        if int(str(item["lustre"].get("mode", "0")), 8) & 0o222:
            raise PreparationError(f"staged BERT file is writable: {item['path']}")
    expected_text_root = (
        Path(inputs["ptm_staging"]["bert_cache_root"]) / text["revision"]
    )
    if text.get("lustre_root") != str(expected_text_root):
        raise PreparationError("BERT Lustre root differs")
    if document.get("execution") != {
        "cpu_model_runs": 0,
        "gpu_model_runs": 0,
        "scheduler_jobs_submitted": 0,
        "checkpoint_loads": 0,
        "operation": "data_only_download_checksum_and_lustre_publication",
    }:
        raise PreparationError("runtime staging may not claim model execution")
    payload = copy.deepcopy(dict(document))
    expected = payload.pop("stage_record_sha256", None)
    if expected != canonical_sha256(payload):
        raise PreparationError("runtime input stage hash differs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--local-cache-root",
        type=Path,
        default=Path(
            "/localhome/local-rarunachalam/.tao/cache/"
            "grounding_dino_runtime_inputs"
        ),
    )
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    inputs = read_json(arguments.inputs)
    if arguments.check_only:
        document = read_json(arguments.output)
        validate_runtime_input_stage(document, inputs=inputs)
    else:
        document = stage_runtime_inputs(
            inputs=inputs,
            local_cache_root=arguments.local_cache_root,
        )
        arguments.output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "bert_revision": document["text_encoder"]["revision"],
                "model_runs": 0,
                "ptm_ids": [item["id"] for item in document["official_ptms"]],
                "scheduler_jobs_submitted": 0,
                "stage_record_sha256": document["stage_record_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
