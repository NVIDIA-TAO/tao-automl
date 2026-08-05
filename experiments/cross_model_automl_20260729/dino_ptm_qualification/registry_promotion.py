# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministically derive a TAO 7.1 DINO registry from qualification evidence.

No checkpoint ID is accepted on the command line. Membership is the exact
intersection of the immutable CPU and real-data GPU qualification reports.
Candidate mode emits a provisional registry for the complete local preflight;
final mode additionally requires a validated, SLURM-ready local report over
exactly that population.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from tao_automl.model_preflight import validate_model_preflight_report
from tao_automl.ptm_registry import PTMRegistry, canonical_sha256

try:
    from .qualification_driver import (
        DINOQualificationError,
        load_verified_qualification_completion,
    )
except ImportError:  # Direct execution from this directory.
    from qualification_driver import (
        DINOQualificationError,
        load_verified_qualification_completion,
    )


TAO_VERSION = "7.1.0-rc-245"
TAO_COMPATIBILITY = "==7.1.0"
CONTAINER_IDENTITY = (
    "sha256:949c0ea8ace09ac91951be4169353cf214daaa3ede7db9eed94070b020361667"
)
INTERVENTION_FLAGS = {
    "agent_selected_candidate": False,
    "agent_injected_candidate": False,
    "agent_modified_search_space_after_results": False,
    "agent_changed_seed_after_results": False,
    "agent_changed_budget_after_results": False,
    "agent_changed_threshold_after_results": False,
    "agent_changed_ptm_after_results": False,
    "agent_overrode_winner": False,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _create_only(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _qualified_population(
    completion: Mapping[str, Any],
    *,
    label: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if (
        completion.get("qualification_only") is not True
        or completion.get("runtime_eligibility_mutated") is not False
        or completion.get("selection_invoked") is not False
        or completion.get("agent_selected_checkpoint") is not False
    ):
        raise DINOQualificationError(
            f"{label} qualification isolation flags are invalid"
        )
    accounting = completion.get("candidate_accounting")
    if not isinstance(accounting, Mapping) or accounting.get("complete") is not True:
        raise DINOQualificationError(
            f"{label} candidate accounting is incomplete"
        )
    evaluated = tuple(accounting.get("evaluated_checkpoint_ids", ()))
    prepared = tuple(accounting.get("prepared_checkpoint_ids", ()))
    excluded = tuple(accounting.get("excluded_checkpoint_ids", ()))
    for values in (evaluated, prepared, excluded):
        if values != tuple(sorted(set(values))) or any(
            not isinstance(item, str) or not item for item in values
        ):
            raise DINOQualificationError(
                f"{label} candidate accounting IDs are invalid"
            )
    if set(prepared) & set(excluded) or set(prepared) | set(excluded) != set(
        evaluated
    ):
        raise DINOQualificationError(
            f"{label} candidate accounting is not an exact partition"
        )
    return evaluated, prepared, excluded


def _local_ptm_population(report: Mapping[str, Any]) -> dict[str, str]:
    if (
        report.get("completion_state") != "completed"
        or report.get("slurm_ready") is not True
    ):
        raise DINOQualificationError(
            "Final promotion requires a completed SLURM-ready local preflight"
        )
    inputs = report.get("inputs")
    records = report.get("records")
    if not isinstance(inputs, Mapping) or not isinstance(records, list):
        raise DINOQualificationError("Local preflight report is incomplete")
    input_ptms = inputs.get("eligible_ptms", [])
    input_hashes = {
        item["id"]: item["registry_record_sha256"]
        for item in input_ptms
    }
    input_ids = tuple(sorted(input_hashes))
    smoke = next(
        (
            item
            for item in records
            if item.get("stage") == "eligible_ptm_smoke"
            and item.get("status") == "passed"
        ),
        None,
    )
    if not isinstance(smoke, Mapping):
        raise DINOQualificationError(
            "Local preflight has no passing all-PTM smoke stage"
        )
    smoke_ids = tuple(sorted(
        item["ptm_id"]
        for item in smoke.get("evidence", {}).get("ptms", [])
    ))
    if not input_ids or smoke_ids != input_ids:
        raise DINOQualificationError(
            "Local preflight PTM population is incomplete"
        )
    return dict(sorted(input_hashes.items()))


def build_promoted_registry(
    *,
    base_registry: PTMRegistry,
    cpu_completion: Mapping[str, Any],
    gpu_completion: Mapping[str, Any],
    registry_version: str,
    validation_evidence: str,
    local_report: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the generated registry and unhashed audit document."""
    if not isinstance(registry_version, str) or not registry_version.strip():
        raise ValueError("registry_version must be a non-empty string")
    if not isinstance(validation_evidence, str) or not validation_evidence.strip():
        raise ValueError("validation_evidence must be a non-empty string")
    cpu_evaluated, cpu_prepared, cpu_excluded = _qualified_population(
        cpu_completion,
        label="CPU",
    )
    gpu_evaluated, gpu_prepared, gpu_excluded = _qualified_population(
        gpu_completion,
        label="GPU",
    )
    if gpu_completion.get("upstream_completion_sha256") != cpu_completion.get(
        "completion_sha256"
    ):
        raise DINOQualificationError(
            "GPU qualification is not bound to the CPU completion"
        )
    if gpu_evaluated != cpu_prepared:
        raise DINOQualificationError(
            "GPU qualification population is not exactly the CPU-pass population"
        )
    promoted = gpu_prepared
    if not promoted:
        raise DINOQualificationError(
            "No DINO checkpoint passed both qualification stages"
        )
    base_document = base_registry.to_dict()
    default_ptm = base_document["models"]["dino"]["default_ptm"]
    if default_ptm not in promoted:
        raise DINOQualificationError(
            "The registered DINO default did not pass target-release "
            "qualification; choose a new default through a separate product "
            "decision before runtime promotion"
        )
    local_record_hashes = None
    if local_report is not None:
        local_record_hashes = _local_ptm_population(local_report)
        if tuple(local_record_hashes) != promoted:
            raise DINOQualificationError(
                "Local preflight population does not match the promotion set"
            )

    document = base_document
    document["registry_version"] = registry_version.strip()
    records = document["models"]["dino"]["checkpoints"]
    known_ids = {item["id"] for item in records}
    if set(cpu_evaluated) - known_ids:
        raise DINOQualificationError(
            "Qualification evidence contains unknown registry checkpoints"
        )
    for record in records:
        if record["id"] not in promoted:
            continue
        record["status"] = "supported"
        record.pop("status_reason", None)
        compatibility = set(record.get("compatible_tao_versions", ()))
        compatibility.add(TAO_COMPATIBILITY)
        record["compatible_tao_versions"] = sorted(compatibility)
        record["validation"] = {
            "status": "validated",
            "tao_version": TAO_VERSION,
            "container_identity": CONTAINER_IDENTITY,
            "evidence": validation_evidence.strip(),
        }
    if local_record_hashes is not None:
        promoted_hashes = {
            record["id"]: canonical_sha256(record)
            for record in records
            if record["id"] in promoted
        }
        if promoted_hashes != local_record_hashes:
            raise DINOQualificationError(
                "Local preflight registry-record hashes do not match the "
                "generated final registry; candidate and final promotion "
                "arguments must be identical"
            )
    promoted_registry = PTMRegistry(document)
    promoted_document = promoted_registry.to_dict()
    audit = {
        "schema_version": 1,
        "promotion_mode": (
            "final_after_local_preflight"
            if local_report is not None
            else "candidate_for_local_preflight"
        ),
        "promotion_algorithm": (
            "exact_cpu_gpu_pass_intersection_with_default_ptm_gate"
        ),
        "base_registry_sha256": base_registry.document_sha256,
        "cpu_completion_sha256": cpu_completion["completion_sha256"],
        "gpu_completion_sha256": gpu_completion["completion_sha256"],
        "local_preflight_report_sha256": (
            local_report.get("report_sha256")
            if local_report is not None
            else None
        ),
        "cpu_evaluated_checkpoint_ids": list(cpu_evaluated),
        "cpu_prepared_checkpoint_ids": list(cpu_prepared),
        "cpu_excluded_checkpoint_ids": list(cpu_excluded),
        "gpu_evaluated_checkpoint_ids": list(gpu_evaluated),
        "gpu_prepared_checkpoint_ids": list(gpu_prepared),
        "gpu_excluded_checkpoint_ids": list(gpu_excluded),
        "promoted_checkpoint_ids": list(promoted),
        "runtime_distribution_ready": local_report is not None,
        "registry_version": promoted_registry.registry_version,
        "promoted_registry_sha256": promoted_registry.document_sha256,
        "intervention_flags": dict(INTERVENTION_FLAGS),
    }
    return promoted_document, audit


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DINOQualificationError(f"{path.name} must contain an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a DINO PTM registry from immutable qualification evidence."
    )
    parser.add_argument("--base-registry", required=True)
    parser.add_argument("--cpu-output-dir", required=True)
    parser.add_argument("--cpu-cache-dir", required=True)
    parser.add_argument("--gpu-output-dir", required=True)
    parser.add_argument("--gpu-cache-dir", required=True)
    parser.add_argument("--output-registry", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--registry-version", required=True)
    parser.add_argument("--validation-evidence", required=True)
    parser.add_argument("--local-report")
    args = parser.parse_args(argv)

    base_path = Path(args.base_registry).expanduser().resolve()
    output_path = Path(args.output_registry).expanduser().resolve()
    audit_path = Path(args.audit).expanduser().resolve()
    if output_path.exists() or audit_path.exists():
        raise FileExistsError("promotion outputs are create-only")
    base_registry = PTMRegistry(_load_json(base_path))
    cpu = load_verified_qualification_completion(
        output_dir=args.cpu_output_dir,
        cache_dir=args.cpu_cache_dir,
    )
    gpu = load_verified_qualification_completion(
        output_dir=args.gpu_output_dir,
        cache_dir=args.gpu_cache_dir,
    )
    local = None
    if args.local_report:
        local_path = Path(args.local_report).expanduser().resolve()
        local = validate_model_preflight_report(_load_json(local_path))
    document, audit = build_promoted_registry(
        base_registry=base_registry,
        cpu_completion=cpu,
        gpu_completion=gpu,
        registry_version=args.registry_version,
        validation_evidence=args.validation_evidence,
        local_report=local,
    )
    registry_bytes = (
        json.dumps(
            document,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    audit = {
        **audit,
        "base_registry_file_sha256": _sha256_file(base_path),
        "output_registry_file_sha256": hashlib.sha256(
            registry_bytes
        ).hexdigest(),
    }
    audit = {**audit, "audit_sha256": canonical_sha256(audit)}
    _create_only(output_path, registry_bytes)
    _create_only(audit_path, _canonical_bytes(audit) + b"\n")
    print(_canonical_bytes({
        "audit_sha256": audit["audit_sha256"],
        "output_registry_file_sha256": audit[
            "output_registry_file_sha256"
        ],
        "promoted_checkpoint_ids": audit["promoted_checkpoint_ids"],
        "runtime_distribution_ready": audit[
            "runtime_distribution_ready"
        ],
    }).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
