#!/usr/bin/env python3

"""Validate or concurrently launch six expanded-front latency allocations."""

from __future__ import annotations

import argparse
import base64
import copy
import fcntl
from functools import wraps
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from typing import Any

import yaml

import post_front_matched_manifest_generator as manifest_generator


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "post_front_matched_manifest.v1.json"
DEFAULT_RUNTIME = HERE / "runtime" / "post_front_matched"
DEFAULT_REPORT = DEFAULT_RUNTIME / "dry_run.json"
LAUNCH_CONTRACT_NAME = "launch_contract.v1.json"
BLOCK_RUNNER = HERE / "post_front_matched_block_runner.py"
AGGREGATOR = HERE / "post_front_matched_aggregator.py"
STAGING_CHUNK_BYTES = 48 * 1024
MAX_RUNTIME_ARGUMENT_BYTES = 64 * 1024
MAX_RENDERED_COMMAND_BYTES = 1024 * 1024
EXPECTED_ACKNOWLEDGEMENT = (
    "USER_AUTHORIZED_DINO_POST_FRONT_6X8GPU_VALIDATION_20260728"
)


class ContractError(ValueError):
    """Raised when launch-time evidence violates the immutable manifest."""


def exclusive_submission_operation(function: Any) -> Any:
    """Hold one nonblocking process lock across every mutating launch path."""

    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        runtime_dir = kwargs.get("runtime_dir")
        if not isinstance(runtime_dir, Path):
            raise ContractError(
                "mutating launch functions require keyword runtime_dir"
            )
        lock_path = (
            runtime_dir.parent
            / f".{runtime_dir.name}.submission.lock"
        )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as error:
            raise ContractError(
                "submission lock file cannot be opened safely"
            ) from error
        with os.fdopen(descriptor, "a+b") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                raise ContractError("submission lock file is unsafe")
            try:
                fcntl.flock(
                    stream.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError as error:
                raise ContractError(
                    "another post-front submission operation holds the lock"
                ) from error
            try:
                return function(*args, **kwargs)
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    return guarded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and render all six jobs without submission (default).",
    )
    mode.add_argument(
        "--launch",
        action="store_true",
        help="Submit all six independent matched allocations without waiting.",
    )
    mode.add_argument(
        "--replace-incomplete-allocation",
        metavar="ALLOCATION_ID",
        help=(
            "Rerun one complete allocation whose prior TAO job is durably "
            "Error/Canceled, or Complete only with exact aggregator "
            "invalidation evidence; never reuse its partial measurements."
        ),
    )
    mode.add_argument(
        "--resume-incomplete-submission",
        action="store_true",
        help=(
            "Reconcile the exact incomplete ledger with private SDK state "
            "and submit only allocations proven not to exist."
        ),
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest-file-sha256", required=True)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--submission-ledger-sha256",
        help=(
            "Required exact current ledger SHA256 for replacement or "
            "incomplete-submission resume mode."
        ),
    )
    parser.add_argument(
        "--invalidation-evidence",
        type=Path,
        help=(
            "Immutable aggregator evidence required only when replacing a "
            "TAO job whose durable status is Complete."
        ),
    )
    parser.add_argument(
        "--invalidation-evidence-sha256",
        help="Exact whole-file SHA256 for --invalidation-evidence.",
    )
    parser.add_argument(
        "--invalidation-evidence-internal-sha256",
        help="Exact canonical internal SHA256 for --invalidation-evidence.",
    )
    parser.add_argument(
        "--verify-remote",
        action="store_true",
        help="Read-only verify SQSH, dataset, and retained checkpoints.",
    )
    parser.add_argument("--acknowledgement", default="")
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


def atomic_create_json(path: Path, payload: Any) -> None:
    """Create an immutable JSON marker without an overwrite race."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, pending_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    pending = Path(pending_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(pending, path)
        except FileExistsError as error:
            raise ContractError(
                f"immutable file already exists; refusing overwrite: {path}"
            ) from error
    finally:
        pending.unlink(missing_ok=True)


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ContractError(
            f"{label} mismatch: expected {expected!r}, got {actual!r}"
        )


def require_exact_reconstructed_manifest(
    manifest: dict[str, Any],
    reconstructed: dict[str, Any],
) -> None:
    """Reject even self-rehashed semantic drift from the pinned source graph."""

    if manifest == reconstructed:
        return
    differing_keys = sorted(
        key
        for key in set(manifest) | set(reconstructed)
        if manifest.get(key) != reconstructed.get(key)
    )
    raise ContractError(
        "post-front manifest differs from deterministic pinned-source "
        "reconstruction at top-level keys: "
        + ", ".join(differing_keys)
    )


def load_manifest(
    path: Path,
    supplied_sha256: str,
) -> tuple[dict[str, Any], str]:
    manifest, whole_file_sha256 = manifest_generator.load_exact_json(
        path,
        supplied_sha256,
        "post-front manifest",
    )
    manifest_generator.validate_internal_digest(
        manifest,
        "manifest_sha256",
        "post-front manifest",
    )
    validate_manifest_contract(manifest)
    return manifest, whole_file_sha256


def validate_schedule(manifest: dict[str, Any]) -> None:
    candidate_ids = [
        item["candidate_id"] for item in manifest.get("candidates", [])
    ]
    require_equal(
        candidate_ids,
        sorted(candidate_ids),
        "manifest canonical candidate order",
    )
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ContractError("manifest candidate IDs must be unique and non-empty")
    expected = manifest_generator.build_schedule(candidate_ids)
    require_equal(manifest.get("schedule"), expected, "manifest schedule")


def validate_manifest_contract(manifest: dict[str, Any]) -> None:
    require_equal(manifest.get("schema_version"), 1, "manifest schema")
    require_equal(
        manifest.get("manifest_id"),
        "dino_expanded_post_front_matched_20260728_v1",
        "manifest ID",
    )
    require_equal(
        manifest.get("status"),
        "immutable_ready_to_launch",
        "manifest status",
    )
    require_equal(manifest.get("scope"), manifest_generator.EXPECTED_SCOPE, "scope")
    for key in (
        "feeds_final_selection",
        "feeds_reselection",
        "manual_candidate_addition_or_removal_permitted",
        "manual_winner_override_permitted",
        "selection_time_objective_replacement_permitted",
    ):
        require_equal(manifest.get(key), False, f"validation-only flag {key}")
    derivation = manifest.get("candidate_derivation", {})
    require_equal(
        derivation.get("source"),
        (
            "independent pinned tao_automl selector replay over every "
            "successful expanded_candidate_table row"
        ),
        "candidate derivation source",
    )
    require_equal(
        derivation.get("cross_check_source"),
        "expanded_combined_selection.candidates",
        "candidate derivation cross-check source",
    )
    require_equal(
        derivation.get("predicate"),
        (
            "recomputed valid is true and recomputed global "
            "pareto_rank equals zero"
        ),
        "candidate predicate",
    )
    for key in (
        "manual_filtering_used",
        "winner_identity_used",
        "objective_values_used_for_schedule",
    ):
        require_equal(derivation.get(key), False, f"candidate derivation {key}")
    candidate_ids = [item["candidate_id"] for item in manifest["candidates"]]
    if any(
        re.fullmatch(r"[A-Za-z0-9_.-]+", candidate_id) is None
        for candidate_id in candidate_ids
    ):
        raise ContractError(
            "manifest candidate IDs must be safe path components"
        )
    require_equal(
        derivation.get("candidate_count"),
        len(candidate_ids),
        "derived candidate count",
    )
    replay_proof = derivation.get("selector_replay_proof")
    if not isinstance(replay_proof, dict):
        raise ContractError("manifest lacks independent selector replay proof")
    require_equal(
        replay_proof.get("global_rank_zero_candidate_ids"),
        candidate_ids,
        "replayed global front candidates",
    )
    require_equal(
        replay_proof.get("global_rank_zero_candidate_set_sha256"),
        manifest_generator.sha256_value(candidate_ids),
        "replayed global front digest",
    )
    for key in (
        "order_independent",
        "all_candidate_audits_exact_match",
        "candidate_table_audits_exact_match",
        "combined_analysis_exact_match",
        "global_rank_zero_front_exact_match",
    ):
        require_equal(replay_proof.get(key), True, f"selector replay {key}")
    require_equal(
        derivation.get("candidate_ids"),
        candidate_ids,
        "derived candidate identities",
    )
    require_equal(
        derivation.get("candidate_set_sha256"),
        manifest_generator.sha256_value(candidate_ids),
        "candidate-set digest",
    )
    for candidate in manifest["candidates"]:
        candidate_id = candidate["candidate_id"]
        require_equal(
            candidate.get("global_pareto_rank"),
            0,
            f"{candidate_id} global rank",
        )
        require_equal(
            candidate.get("global_dominated_by"),
            [],
            f"{candidate_id} dominated_by",
        )
        manifest_generator.require_sha256(
            candidate.get("candidate_table_record_sha256"),
            f"{candidate_id} retained-record digest",
        )
        manifest_generator.require_sha256(
            candidate.get("selection_audit_sha256"),
            f"{candidate_id} selection-audit digest",
        )
        require_equal(
            manifest_generator.sha256_value(candidate["resolved_model_spec"]),
            candidate["resolved_model_spec_sha256"],
            f"{candidate_id} full model digest",
        )
        manifest_generator.require_sha256(
            candidate["checkpoint"]["sha256"],
            f"{candidate_id} checkpoint digest",
        )
    validate_schedule(manifest)
    runtime = manifest.get("runtime", {})
    require_equal(runtime.get("num_nodes"), 1, "runtime node count")
    require_equal(runtime.get("gpu_count"), 8, "runtime GPU count")
    require_equal(
        runtime.get("local_runtime_path"),
        str(DEFAULT_RUNTIME.resolve()),
        "post-front runtime path",
    )
    require_equal(
        runtime.get("image_is_prebuilt_sqsh"),
        True,
        "prebuilt SQSH image",
    )
    require_equal(
        runtime.get("sdk_sqsh_conversion_enabled"),
        False,
        "SDK SQSH conversion",
    )
    require_equal(runtime.get("slurm_use_requeue"), False, "SLURM requeue")
    require_equal(runtime.get("slurm_time_hours"), 4.0, "SLURM time")
    require_equal(runtime.get("slurm_timeout_hours"), 3.8, "SDK timeout")
    protocol = manifest.get("latency_protocol", {})
    require_equal(protocol.get("warmup_iterations"), 50, "latency warmups")
    require_equal(protocol.get("timed_iterations"), 100, "timed iterations")
    require_equal(protocol.get("repeated_rounds"), 5, "latency rounds")
    require_equal(protocol.get("batch_size_per_gpu"), 1, "latency batch")
    require_equal(protocol.get("precision"), "fp32", "latency precision")
    paired = manifest.get("paired_analysis", {})
    require_equal(
        paired.get("bootstrap_resamples"),
        10000,
        "paired bootstrap resamples",
    )
    require_equal(
        paired.get("bootstrap_confidence_level"),
        0.95,
        "paired bootstrap confidence",
    )
    require_equal(
        paired.get("bootstrap_seed"),
        20260728,
        "paired bootstrap seed",
    )
    require_equal(
        paired.get("practical_tolerance_ms"),
        manifest_generator.EXPECTED_PRACTICAL_TOLERANCE_MS,
        "practical tolerance",
    )
    require_equal(
        manifest.get("selection_isolation"),
        {
            "measurements_feed_reselection": False,
            "winner_reselection_permitted": False,
            "original_selection_time_measurements_replaced": False,
            "algorithm_selected_candidate_overridden": False,
            "allowed_use": "stability analysis and hypothesis verdict only",
        },
        "selection isolation",
    )
    require_equal(
        manifest.get("incomplete_allocation_policy"),
        (
            "Exclude the entire allocation and rerun the complete front "
            "under a new TAO job ID; never combine a partial block."
        ),
        "incomplete allocation policy",
    )


def load_pinned_source(source: dict[str, Any], label: str) -> Path:
    path = Path(source["path"]).resolve()
    require_equal(
        manifest_generator.sha256_file(path),
        source["sha256"],
        f"{label} source SHA256",
    )
    return path


def validate_final_source_evidence(manifest: dict[str, Any]) -> dict[str, Any]:
    sources = manifest["source_artifacts"]
    expanded_path = load_pinned_source(
        sources["expanded_manifest"],
        "expanded manifest",
    )
    expanded = manifest_generator.load_json(expanded_path)
    manifest_generator.validate_expanded_manifest(
        expanded,
        sources["expanded_manifest"]["sha256"],
    )
    combined_path = load_pinned_source(
        sources["expanded_combined_selection"],
        "combined selection",
    )
    table_path = load_pinned_source(
        sources["expanded_candidate_table"],
        "candidate table",
    )
    integrity_path = load_pinned_source(
        sources["expanded_integrity_audit"],
        "integrity audit",
    )
    combined = manifest_generator.load_json(combined_path)
    table = manifest_generator.load_json(table_path)
    integrity = manifest_generator.load_json(integrity_path)
    manifest_generator.validate_integrity_bindings(
        expanded_manifest_path=expanded_path,
        expanded_manifest_sha256=sources["expanded_manifest"]["sha256"],
        combined_path=combined_path,
        combined_sha256=sources["expanded_combined_selection"]["sha256"],
        table_path=table_path,
        table_sha256=sources["expanded_candidate_table"]["sha256"],
        integrity=integrity,
        combined=combined,
        table=table,
    )
    selector_replay = manifest_generator.validate_completed_archive(
        combined,
        table,
        expanded_manifest=expanded,
        expanded_manifest_path=expanded_path,
        expanded_manifest_sha256=sources["expanded_manifest"]["sha256"],
        integrity=integrity,
    )
    require_equal(
        manifest.get("selection_snapshot"),
        {
            "selections": combined["selections"],
            "selection_authority": combined["selection_authority"],
            "preserved_unchanged": True,
        },
        "frozen original selection snapshot",
    )
    require_equal(
        manifest.get("candidate_derivation", {}).get(
            "selector_replay_proof"
        ),
        selector_replay["proof"],
        "independent selector replay proof",
    )
    stored_selection_stack = sources.get("tao_automl_selection_stack")
    if not isinstance(stored_selection_stack, dict):
        raise ContractError("pinned selection-stack provenance is missing")
    require_equal(
        sources.get("tao_automl_repository"),
        {
            "path": stored_selection_stack["repository"],
            "branch": stored_selection_stack["branch"],
            "head_commit": stored_selection_stack["head_commit"],
            "commit_policy": stored_selection_stack["commit_policy"],
            "selection_core_commit": stored_selection_stack[
                "selection_core_commit"
            ],
        },
        "tao-automl repository provenance",
    )
    require_equal(
        manifest_generator.stable_selection_stack_projection(
            stored_selection_stack
        ),
        manifest_generator.stable_selection_stack_projection(
            selector_replay["selection_stack"]
        ),
        "pinned selection-stack content",
    )
    repository = Path(stored_selection_stack["repository"]).resolve()
    generation_head = manifest_generator.require_git_oid(
        stored_selection_stack.get("head_commit"),
        "selection-stack generation HEAD",
    )
    current_head = manifest_generator.require_git_oid(
        git_value(repository, "rev-parse", "HEAD"),
        "selection-stack current HEAD",
    )
    ancestry = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "merge-base",
            "--is-ancestor",
            generation_head,
            current_head,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestry.returncode != 0:
        raise ContractError(
            "selection-stack generation HEAD is not an ancestor of current HEAD"
        )
    derived = manifest_generator.derive_candidate_records(
        combined,
        table,
        selector_replay,
    )
    require_equal(derived, manifest["candidates"], "re-derived front records")
    require_equal(
        manifest_generator.build_schedule(
            [item["candidate_id"] for item in derived]
        ),
        manifest["schedule"],
        "re-derived balanced schedule",
    )
    for label in (
        "sensitivity_manifest",
        "dino_latency_benchmark",
        "dino_evaluate_template",
        "expanded_runner",
        "latency_stats",
        "expanded_runtime_contract",
    ):
        load_pinned_source(sources[label], label)
    for relative_path, source in sources[
        "tao_automl_selection_stack"
    ]["source_files"].items():
        path = load_pinned_source(source, relative_path)
        require_equal(
            path,
            (
                Path(sources["tao_automl_repository"]["path"])
                / relative_path
            ).resolve(),
            f"{relative_path} repository path",
        )
    tool_checks = {}
    for filename, source in sources["post_front_tools"].items():
        require_equal(
            filename in manifest_generator.TOOL_FILENAMES,
            True,
            f"unexpected post-front source {filename}",
        )
        path = load_pinned_source(source, filename)
        require_equal(path.name, filename, f"{filename} basename")
        for key in ("tracked", "committed", "clean_against_head"):
            require_equal(
                source.get(key),
                True,
                f"{filename} provenance {key}",
            )
        manifest_generator.require_git_oid(
            source.get("git_blob"),
            f"{filename} git blob",
        )
        require_equal(
            source.get("git_blob"),
            source.get("head_git_blob"),
            f"{filename} working-tree/HEAD blob",
        )
        tool_checks[filename] = source["sha256"]
    require_equal(
        set(tool_checks),
        set(manifest_generator.TOOL_FILENAMES),
        "post-front tool source set",
    )
    reconstructed_sources, sensitivity = manifest_generator.source_artifacts(
        expanded_path,
        sources["expanded_manifest"]["sha256"],
        combined_path,
        sources["expanded_combined_selection"]["sha256"],
        table_path,
        sources["expanded_candidate_table"]["sha256"],
        integrity_path,
        sources["expanded_integrity_audit"]["sha256"],
        expanded,
        selector_replay,
    )
    # The manifest records the repository HEAD at generation time.  A later
    # commit containing that immutable manifest is permitted, but only after
    # the ancestry check above; no launch-affecting source content may drift.
    reconstructed_sources["tao_automl_repository"]["head_commit"] = (
        generation_head
    )
    reconstructed_sources["tao_automl_selection_stack"]["head_commit"] = (
        generation_head
    )
    require_equal(
        reconstructed_sources,
        sources,
        "reconstructed post-front source artifacts",
    )
    reconstructed_manifest = manifest_generator.build_manifest(
        expanded_manifest=expanded,
        candidates=derived,
        sources=reconstructed_sources,
        sensitivity_manifest=sensitivity,
        combined=combined,
        selector_replay=selector_replay,
    )
    require_exact_reconstructed_manifest(manifest, reconstructed_manifest)
    return {
        "expanded_manifest_sha256": sources["expanded_manifest"]["sha256"],
        "combined_selection_sha256": sources[
            "expanded_combined_selection"
        ]["sha256"],
        "candidate_table_sha256": sources["expanded_candidate_table"][
            "sha256"
        ],
        "integrity_audit_sha256": sources["expanded_integrity_audit"][
            "sha256"
        ],
        "expanded_runtime_contract_sha256": sources[
            "expanded_runtime_contract"
        ]["sha256"],
        "candidate_set_sha256": manifest["candidate_derivation"][
            "candidate_set_sha256"
        ],
        "selector_replay_proof_sha256": manifest_generator.sha256_value(
            selector_replay["proof"]
        ),
        "selection_stack_sha256": manifest_generator.sha256_value(
            manifest_generator.stable_selection_stack_projection(
                selector_replay["selection_stack"]
            )
        ),
        "tool_sources": tool_checks,
        "exact_manifest_reconstruction": True,
        "reconstructed_manifest_internal_sha256": reconstructed_manifest[
            "manifest_sha256"
        ],
    }


def load_expanded_runner(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "post_front_pinned_expanded_runner",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import expanded runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def generate_configs(manifest: dict[str, Any]) -> dict[str, bytes]:
    sources = manifest["source_artifacts"]
    expanded_path = Path(sources["expanded_manifest"]["path"])
    expanded = manifest_generator.load_json(expanded_path)
    runner_path = Path(sources["expanded_runner"]["path"])
    runner = load_expanded_runner(runner_path)
    template = yaml.safe_load(
        Path(sources["dino_evaluate_template"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    configs = {}
    for candidate in manifest["candidates"]:
        config = runner.evaluation_spec(
            expanded,
            template,
            candidate["resolved_model_spec"],
            candidate["checkpoint"]["path"],
            latency=True,
        )
        require_equal(
            config["model"],
            candidate["resolved_model_spec"],
            f"{candidate['candidate_id']} evaluation model",
        )
        require_equal(
            config["evaluate"]["checkpoint"],
            candidate["checkpoint"]["path"],
            f"{candidate['candidate_id']} evaluation checkpoint",
        )
        payload = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
        configs[candidate["candidate_id"]] = payload
    return configs


def build_block_plan(
    manifest: dict[str, Any],
    whole_file_sha256: str,
    allocation: dict[str, Any],
    configs: dict[str, bytes],
) -> dict[str, Any]:
    candidates = {
        item["candidate_id"]: item for item in manifest["candidates"]
    }
    planned = []
    for position, candidate_id in enumerate(allocation["candidate_order"]):
        candidate = candidates[candidate_id]
        config_relative_path = f"configs/{candidate_id}.yaml"
        planned.append(
            {
                "candidate_id": candidate_id,
                "position": position,
                "run_label": (
                    f"{allocation['allocation_id']}_p{position:03d}_"
                    f"{candidate_id}"
                ),
                "checkpoint_path": candidate["checkpoint"]["path"],
                "checkpoint_sha256": candidate["checkpoint"]["sha256"],
                "resolved_model_spec_sha256": candidate[
                    "resolved_model_spec_sha256"
                ],
                "candidate_table_record_sha256": candidate[
                    "candidate_table_record_sha256"
                ],
                "config_relative_path": config_relative_path,
                "config_sha256": sha256_bytes(configs[candidate_id]),
            }
        )
    runtime = manifest["runtime"]
    plan = {
        "schema_version": 1,
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": whole_file_sha256,
        "schedule_sha256": manifest["schedule"]["schedule_sha256"],
        "allocation_id": allocation["allocation_id"],
        "allocation_index": allocation["allocation_index"],
        "design_row_index": allocation["design_row_index"],
        "candidate_count": len(planned),
        "gpu_count": 8,
        "num_nodes": 1,
        "feeds_final_selection": False,
        "feeds_reselection": False,
        "manual_candidate_addition_or_removal_permitted": False,
        "winner_override_permitted": False,
        "selection_time_objective_replacement_permitted": False,
        "benchmark_sha256": manifest["source_artifacts"][
            "dino_latency_benchmark"
        ]["sha256"],
        "latency_stats_sha256": manifest["source_artifacts"][
            "latency_stats"
        ]["sha256"],
        "block_runner_sha256": manifest["source_artifacts"][
            "post_front_tools"
        ][BLOCK_RUNNER.name]["sha256"],
        "expected_hardware": {
            "gpu_name": runtime["required_gpu_name"],
            "compute_capability": runtime["required_compute_capability"],
            "total_memory_bytes": runtime["required_total_memory_bytes"],
            "torch": runtime["required_torch"],
            "torch_version_match": runtime["torch_version_match"],
            "cuda": runtime["required_cuda"],
            "cudnn": runtime["required_cudnn"],
        },
        "latency_protocol": copy.deepcopy(manifest["latency_protocol"]),
        "output_contract": copy.deepcopy(runtime["output_contract"]),
        "candidates": planned,
    }
    plan["block_plan_sha256"] = manifest_generator.sha256_value(plan)
    return plan


def staged_command(
    manifest: dict[str, Any],
    allocation: dict[str, Any],
    plan: dict[str, Any],
    configs: dict[str, bytes],
) -> tuple[str, dict[str, Any]]:
    benchmark_path = Path(
        manifest["source_artifacts"]["dino_latency_benchmark"]["path"]
    )
    latency_stats_path = Path(
        manifest["source_artifacts"]["latency_stats"]["path"]
    )
    files = {
        BLOCK_RUNNER.name: BLOCK_RUNNER.read_bytes(),
        "dino_latency_benchmark.py": benchmark_path.read_bytes(),
        "latency_stats.py": latency_stats_path.read_bytes(),
        f"plans/{allocation['allocation_id']}.json": (
            json.dumps(plan, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    }
    for candidate_id, payload in configs.items():
        files[f"configs/{candidate_id}.yaml"] = payload
    file_sha256 = {
        name: sha256_bytes(payload)
        for name, payload in sorted(files.items())
    }
    encoded_files = {
        name: base64.b64encode(payload).decode("ascii")
        for name, payload in files.items()
    }
    bundle = json.dumps(
        encoded_files,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    compressed = gzip.compress(bundle, compresslevel=9, mtime=0)
    encoded = base64.b64encode(compressed).decode("ascii")
    chunks = [
        encoded[offset : offset + STAGING_CHUNK_BYTES]
        for offset in range(0, len(encoded), STAGING_CHUNK_BYTES)
    ]
    if not chunks:
        raise ContractError("staging bundle must not be empty")
    compressed_sha256 = sha256_bytes(compressed)
    bundle_sha256 = sha256_bytes(bundle)
    installer = "\n".join(
        [
            "import base64,gzip,hashlib,json,os,stat,sys",
            "from pathlib import Path",
            "root=Path(sys.argv[1])",
            "info=root.lstat()",
            (
                "if not stat.S_ISDIR(info.st_mode) or "
                "stat.S_ISLNK(info.st_mode): "
                "raise RuntimeError('staging root is not a real directory')"
            ),
            (
                "if info.st_uid!=os.geteuid(): "
                "raise RuntimeError('staging root ownership mismatch')"
            ),
            "os.chmod(root,0o700)",
            (
                "if stat.S_IMODE(root.lstat().st_mode)!=0o700: "
                "raise RuntimeError('staging root mode mismatch')"
            ),
            "encoded=''.join(sys.argv[2:])",
            "compressed=base64.b64decode(encoded,validate=True)",
            (
                "actual=hashlib.sha256(compressed).hexdigest();"
                f"expected={compressed_sha256!r}"
            ),
            (
                "if actual!=expected: raise RuntimeError("
                "'compressed staging bundle digest mismatch')"
            ),
            "bundle=gzip.decompress(compressed)",
            (
                "actual=hashlib.sha256(bundle).hexdigest();"
                f"expected={bundle_sha256!r}"
            ),
            (
                "if actual!=expected: raise RuntimeError("
                "'staging bundle digest mismatch')"
            ),
            "files=json.loads(bundle)",
            f"expected_files={file_sha256!r}",
            (
                "if set(files)!=set(expected_files): raise RuntimeError("
                "'staging file set mismatch')"
            ),
            "for name in sorted(files):",
            " relative=Path(name)",
            (
                " if relative.is_absolute() or '..' in relative.parts: "
                "raise RuntimeError('unsafe staged path')"
            ),
            " payload=base64.b64decode(files[name],validate=True)",
            " actual=hashlib.sha256(payload).hexdigest()",
            (
                " if actual!=expected_files[name]: raise RuntimeError("
                "f'staged file digest mismatch: {name}')"
            ),
            " path=root/name",
            " if path.parent!=root:",
            "  path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)",
            "  parent_info=path.parent.lstat()",
            (
                "  if not stat.S_ISDIR(parent_info.st_mode) or "
                "stat.S_ISLNK(parent_info.st_mode) or "
                "parent_info.st_uid!=os.geteuid(): "
                "raise RuntimeError('unsafe staging parent')"
            ),
            "  os.chmod(path.parent,0o700)",
            " pending=path.with_suffix(path.suffix+'.tmp')",
            (
                " flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL"
                "+getattr(os,'O_NOFOLLOW',0)"
            ),
            " descriptor=os.open(pending,flags,0o600)",
            " with os.fdopen(descriptor,'wb') as stream:",
            "  stream.write(payload)",
            "  stream.flush()",
            "  os.fsync(stream.fileno())",
            " pending.replace(path)",
            " final_info=path.lstat()",
            (
                " if not stat.S_ISREG(final_info.st_mode) or "
                "stat.S_ISLNK(final_info.st_mode) or "
                "final_info.st_uid!=os.geteuid(): "
                "raise RuntimeError('unsafe staged file')"
            ),
            " os.chmod(path,0o600)",
            (
                " if hashlib.sha256(path.read_bytes()).hexdigest()"
                "!=expected_files[name]: "
                "raise RuntimeError(f'post-write digest mismatch: {name}')"
            ),
        ]
    )
    plan_relative = f"plans/{allocation['allocation_id']}.json"
    command_parts = [
        "umask",
        "077",
        "&&",
        'STAGING_ROOT="$(mktemp -d '
        '--tmpdir="${TMPDIR:-/tmp}" '
        '"tao_dino_post_front_${TAO_JOB_ID:?}.XXXXXXXX")"',
        "&&",
        "python",
        "-c",
        shlex.quote(installer),
        '"$STAGING_ROOT"',
    ]
    command_parts.extend(shlex.quote(chunk) for chunk in chunks)
    command_parts.extend(
        [
            "&&",
            "python",
            '"$STAGING_ROOT/' + BLOCK_RUNNER.name + '"',
            "--plan",
            '"$STAGING_ROOT/' + plan_relative + '"',
            "--benchmark-script",
            '"$STAGING_ROOT/dino_latency_benchmark.py"',
            "--latency-stats-module",
            '"$STAGING_ROOT/latency_stats.py"',
            "--output-root",
            '"$TAO_RESULTS_ROOT/$TAO_JOB_ID"',
        ]
    )
    command = " ".join(command_parts)
    runtime_arguments = [
        installer,
        *chunks,
        "$STAGING_ROOT",
        f"$STAGING_ROOT/{BLOCK_RUNNER.name}",
        f"$STAGING_ROOT/{plan_relative}",
        "$STAGING_ROOT/dino_latency_benchmark.py",
        "$STAGING_ROOT/latency_stats.py",
        "$TAO_RESULTS_ROOT/$TAO_JOB_ID",
    ]
    max_argument_bytes = max(
        len(item.encode("utf-8")) for item in runtime_arguments
    )
    command_bytes = len(command.encode("utf-8"))
    if max_argument_bytes > MAX_RUNTIME_ARGUMENT_BYTES:
        raise ContractError("staged command contains an oversized argument")
    if command_bytes > MAX_RENDERED_COMMAND_BYTES:
        raise ContractError("rendered command exceeds safety limit")
    return command, {
        "allocation_id": allocation["allocation_id"],
        "allocation_index": allocation["allocation_index"],
        "design_row_index": allocation["design_row_index"],
        "candidate_order": allocation["candidate_order"],
        "candidate_count": len(allocation["candidate_order"]),
        "block_plan_sha256": plan["block_plan_sha256"],
        "command_sha256": sha256_bytes(command.encode("utf-8")),
        "command_bytes": command_bytes,
        "staging_bundle_compression": "gzip_mtime_0_level_9",
        "staging_bundle_sha256": compressed_sha256,
        "staging_bundle_json_sha256": bundle_sha256,
        "staging_file_sha256": file_sha256,
        "staging_chunk_count": len(chunks),
        "max_staging_chunk_bytes": max(len(item) for item in chunks),
        "max_runtime_argument_bytes": max_argument_bytes,
        "staging_root_policy": (
            "mktemp job-scoped under TMPDIR with 0700 directory, "
            "0600 regular files, owner/symlink/digest validation"
        ),
        "output_root_expression": "$TAO_RESULTS_ROOT/$TAO_JOB_ID",
    }


def load_env_file(path: Path) -> list[str]:
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise PermissionError(
            "required secrets env file must be owner-owned, non-symlink, "
            "and mode 0600 or stricter"
        )
    if not path.is_file():
        raise FileNotFoundError(f"required secrets env file not found: {path}")
    parsed: dict[str, str] = {}
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
            raise ValueError(f"unsupported env line {line_number}")
        key, encoded = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ValueError(f"invalid env key on line {line_number}")
        tokens = shlex.split(encoded, comments=True, posix=True)
        if len(tokens) > 1:
            raise ValueError(f"unsupported env syntax on line {line_number}")
        if key in parsed:
            raise ValueError(f"duplicate env key on line {line_number}: {key}")
        parsed[key] = tokens[0] if tokens else ""
    for key, value in parsed.items():
        if key in os.environ and os.environ[key] != value:
            raise ValueError(
                f"ambient environment conflicts with pinned env file: {key}"
            )
        os.environ[key] = value
    return sorted(parsed)


def ssh_target() -> str:
    host = os.environ.get("SLURM_HOSTNAME", "").split(",", 1)[0].strip()
    user = os.environ.get("SLURM_USER", "").strip()
    if not host or not user:
        raise RuntimeError("SLURM_USER and SLURM_HOSTNAME are required")
    return f"{user}@{host}"


def remote_output(command: str, *, timeout: int = 900) -> str:
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    key = os.environ.get("SSH_KEY_PATH")
    if key:
        ssh.extend(["-i", key])
    ssh.extend([ssh_target(), command])
    return subprocess.run(
        ssh,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    ).stdout


def enforce_remote_sdk_containment() -> dict[str, Any]:
    """Create/harden and verify SDK state directories before each submission."""

    configured = os.environ.get("SLURM_BASE_RESULTS_DIR", "").strip()
    if not configured or not Path(configured).is_absolute():
        raise ContractError(
            "SLURM_BASE_RESULTS_DIR must be an absolute configured path"
        )
    script = "\n".join(
        [
            "import json,os,stat,sys",
            "from pathlib import Path",
            "configured=Path(sys.argv[1])",
            "base=configured.resolve(strict=True)",
            "info=base.lstat()",
            (
                "assert stat.S_ISDIR(info.st_mode) and "
                "not stat.S_ISLNK(info.st_mode), 'base is not a real directory'"
            ),
            "assert info.st_uid==os.geteuid(), 'base ownership mismatch'",
            "os.chmod(base,0o700)",
            (
                "names=('sbatch','env','meta','entrypoints','specs',"
                "'slurm-logs','results')"
            ),
            "records=[]",
            "for name in names:",
            " path=base/name",
            " try:",
            "  os.mkdir(path,0o700)",
            " except FileExistsError:",
            "  pass",
            " child=path.lstat()",
            (
                " assert stat.S_ISDIR(child.st_mode) and "
                "not stat.S_ISLNK(child.st_mode), f'{name} is unsafe'"
            ),
            (
                " assert child.st_uid==os.geteuid(), "
                "f'{name} ownership mismatch'"
            ),
            " os.chmod(path,0o700)",
            " final=path.lstat()",
            (
                " assert stat.S_IMODE(final.st_mode)==0o700, "
                "f'{name} mode mismatch'"
            ),
            (
                " records.append({'name':name,'mode':'0700',"
                "'owner_uid':final.st_uid,'real_directory':True})"
            ),
            (
                "print(json.dumps({'configured_path':str(configured),"
                "'canonical_path':str(base),'base_mode':'0700',"
                "'base_owner_uid':base.lstat().st_uid,"
                "'sensitive_directories':records,"
                "'secret_values_recorded':False},sort_keys=True))"
            ),
        ]
    )
    output = remote_output(
        " ".join(
            [
                "python3",
                "-c",
                shlex.quote(script),
                shlex.quote(configured),
            ]
        ),
        timeout=120,
    )
    evidence = json.loads(output)
    if (
        not isinstance(evidence, dict)
        or evidence.get("base_mode") != "0700"
        or len(evidence.get("sensitive_directories", [])) != 7
        or any(
            item.get("mode") != "0700"
            or item.get("real_directory") is not True
            for item in evidence.get("sensitive_directories", [])
        )
    ):
        raise ContractError("remote SDK containment evidence is invalid")
    evidence["evidence_sha256"] = manifest_generator.sha256_value(evidence)
    return evidence


def observe_job_status_no_retry(sdk: Any, job_id: str) -> dict[str, Any]:
    """Observe one SDK job while proving that inspection cannot relaunch it."""

    entry = sdk.get_job(job_id)
    if not isinstance(entry, dict) or entry.get("backend_type") != "slurm":
        raise ContractError(f"durable SLURM job is missing: {job_id}")
    job = sdk._load_job_from_store(job_id)
    if job is None:
        raise ContractError(f"could not restore durable SLURM job: {job_id}")
    before = sdk._handler.get_job_runtime_identity(job_id)
    protected = {
        key: copy.deepcopy(before.get(key))
        for key in (
            "slurm_job_id",
            "failed_slurm_job_ids",
            "retry_count",
            "submission_attempted",
            "launch_uncertain",
            "launch_token",
            "pre_launch_slurm_job_id",
        )
    }
    durable_status = str(entry.get("status", "Unknown"))
    if durable_status in {"Complete", "Error", "Canceled"}:
        status = durable_status
        message = f"Terminal status {durable_status} was durably recorded"
    else:
        raw = sdk._handler.get_tao_job_status(
            job_id,
            allow_retry=False,
        )
        status = sdk._handler.normalize_status(raw)
        message = sdk._handler.get_job_message(job_id)
    after = sdk._handler.get_job_runtime_identity(job_id)
    for key, expected in protected.items():
        require_equal(
            after.get(key),
            expected,
            f"{job_id} side-effect-free status field {key}",
        )
    return {
        "status": status,
        "message": message,
        "runtime_identity": after,
        "allow_retry": False,
        "scheduler_identity_unchanged": True,
    }


def verify_remote(manifest: dict[str, Any]) -> dict[str, Any]:
    declared = [
        (
            "sqsh",
            manifest["runtime"]["sqsh_path"],
            manifest["runtime"]["sqsh_sha256"],
        ),
        (
            "validation_annotation",
            manifest["dataset"]["validation_annotation"],
            manifest["dataset"]["validation_annotation_sha256"],
        ),
    ]
    seen = {path for _, path, _ in declared}
    for candidate in manifest["candidates"]:
        path = candidate["checkpoint"]["path"]
        if path not in seen:
            declared.append(
                (
                    "checkpoint",
                    path,
                    candidate["checkpoint"]["sha256"],
                )
            )
            seen.add(path)
    artifacts = []
    for kind, path, expected in declared:
        command = (
            f"if test -f {shlex.quote(path)}; then "
            f"sha256sum {shlex.quote(path)}; else echo MISSING; fi"
        )
        output = remote_output(command).strip()
        actual = None if output == "MISSING" else output.split(None, 1)[0]
        artifacts.append(
            {
                "kind": kind,
                "path": path,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "verified": actual == expected,
            }
        )
    image_dir = manifest["dataset"]["validation_image_dir"]
    present = (
        remote_output(
            f"test -d {shlex.quote(image_dir)} && echo PRESENT || echo MISSING",
            timeout=120,
        ).strip()
        == "PRESENT"
    )
    artifacts.append(
        {
            "kind": "validation_image_dir",
            "path": image_dir,
            "verified": present,
        }
    )
    return {
        "all_verified": all(item["verified"] for item in artifacts),
        "artifacts": artifacts,
    }


def git_value(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_launch_source_state(manifest: dict[str, Any]) -> dict[str, Any]:
    sources = manifest["source_artifacts"]
    paths = [
        Path(item["path"]).resolve()
        for item in sources["post_front_tools"].values()
    ]
    paths.extend(
        [
            Path(sources["expanded_runner"]["path"]).resolve(),
            Path(sources["latency_stats"]["path"]).resolve(),
            Path(sources["dino_latency_benchmark"]["path"]).resolve(),
        ]
    )
    repo = Path(
        manifest_generator.load_json(
            Path(sources["expanded_manifest"]["path"])
        )["frozen_identity"]["source_repositories"]["tao_automl"]["path"]
    ).resolve()
    relative = []
    for path in paths:
        try:
            relative.append(str(path.relative_to(repo)))
        except ValueError as error:
            raise ContractError(f"launch source escaped repository: {path}") from error
    tracked = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "ls-files",
            "--error-unmatch",
            "--",
            *relative,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        raise ContractError("launch requires every harness source to be tracked")
    for cached in (False, True):
        command = ["git", "-C", str(repo), "diff", "--quiet"]
        if cached:
            command.append("--cached")
        command.extend(["--", *relative])
        if subprocess.run(command, check=False).returncode != 0:
            raise ContractError("launch requires committed clean harness sources")
    sdk = Path(manifest["runtime"]["sdk_path"])
    require_equal(
        git_value(sdk, "rev-parse", "HEAD"),
        manifest["runtime"]["sdk_commit"],
        "TAO SDK commit",
    )
    require_equal(
        git_value(sdk, "branch", "--show-current"),
        manifest["runtime"]["sdk_branch"],
        "TAO SDK branch",
    )
    sdk_status = git_value(
        sdk,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if sdk_status:
        raise ContractError("TAO SDK worktree must be completely clean")
    return {
        "repository": str(repo),
        "source_count": len(relative),
        "tracked_committed_clean": True,
        "tao_sdk_commit": manifest["runtime"]["sdk_commit"],
        "tao_sdk_branch": manifest["runtime"]["sdk_branch"],
        "tao_sdk_worktree_clean": True,
    }


def submission_ledger_payload(
    manifest: dict[str, Any],
    manifest_file_sha256: str,
    submissions: list[dict[str, Any]],
    *,
    status: str,
    source_checks: dict[str, Any],
    ledger_revision: int = 1,
    superseded_submissions: list[dict[str, Any]] | None = None,
    submission_recovery_events: list[dict[str, Any]] | None = None,
    parent_ledger_sha256: str | None = None,
    pending_submission: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "status": status,
        "phase": "expanded_post_front_matched_latency",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_file_sha256,
        "schedule_sha256": manifest["schedule"]["schedule_sha256"],
        "expected_allocation_count": 6,
        "allocation_count": len(submissions),
        "ledger_revision": ledger_revision,
        "parent_ledger_sha256": parent_ledger_sha256,
        "superseded_submissions": copy.deepcopy(
            superseded_submissions or []
        ),
        "submission_recovery_events": copy.deepcopy(
            submission_recovery_events or []
        ),
        "pending_submission": copy.deepcopy(pending_submission),
        "feeds_final_selection": False,
        "feeds_reselection": False,
        "selection_time_objective_replacement_permitted": False,
        "source_checks": copy.deepcopy(source_checks),
        "submissions": copy.deepcopy(submissions),
    }
    payload["ledger_sha256"] = manifest_generator.sha256_value(payload)
    return payload


def launch_contract_payload(
    manifest: dict[str, Any],
    manifest_file_sha256: str,
    runtime_dir: Path,
    source_checks: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "contract_id": "dino_post_front_matched_launch_20260728_v1",
        "status": "reserved_before_sdk_initialization",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_file_sha256,
        "manifest_internal_sha256": manifest["manifest_sha256"],
        "schedule_sha256": manifest["schedule"]["schedule_sha256"],
        "candidate_set_sha256": manifest["candidate_derivation"][
            "candidate_set_sha256"
        ],
        "runtime_path": str(runtime_dir.resolve()),
        "allocation_count": 6,
        "allocation_ids": [
            item["allocation_id"]
            for item in manifest["schedule"]["allocations"]
        ],
        "source_checks": copy.deepcopy(source_checks),
        "feeds_final_selection": False,
        "feeds_reselection": False,
        "selection_time_objective_replacement_permitted": False,
        "manual_winner_override_permitted": False,
    }
    payload["contract_sha256"] = manifest_generator.sha256_value(payload)
    return payload


def reserve_launch(
    manifest: dict[str, Any],
    manifest_file_sha256: str,
    runtime_dir: Path,
    source_checks: dict[str, Any],
) -> Path:
    expected_runtime = Path(manifest["runtime"]["local_runtime_path"]).resolve()
    require_equal(
        runtime_dir.resolve(),
        expected_runtime,
        "launch runtime directory",
    )
    runtime_dir.mkdir(parents=True, exist_ok=True)
    permitted_existing = {"dry_run.json"}
    unexpected = sorted(
        item.name
        for item in runtime_dir.iterdir()
        if item.name not in permitted_existing
    )
    if unexpected:
        raise ContractError(
            "launch runtime is not fresh; unexpected entries: "
            + ", ".join(unexpected)
        )
    dry_run_path = runtime_dir / "dry_run.json"
    if not dry_run_path.is_file():
        raise ContractError("launch requires the current bound dry-run report")
    dry_run = manifest_generator.load_json(dry_run_path)
    require_equal(
        dry_run.get("status"),
        "dry_run_validated_not_launched",
        "launch dry-run status",
    )
    require_equal(
        dry_run.get("manifest", {}).get("whole_file_sha256"),
        manifest_file_sha256,
        "launch dry-run manifest SHA256",
    )
    require_equal(
        dry_run.get("schedule_sha256"),
        manifest["schedule"]["schedule_sha256"],
        "launch dry-run schedule SHA256",
    )
    require_equal(
        dry_run.get("candidate_ids"),
        manifest["candidate_derivation"]["candidate_ids"],
        "launch dry-run candidates",
    )
    require_equal(
        dry_run.get("submission_ready"),
        True,
        "launch dry-run readiness",
    )
    marker = runtime_dir / LAUNCH_CONTRACT_NAME
    atomic_create_json(
        marker,
        launch_contract_payload(
            manifest,
            manifest_file_sha256,
            runtime_dir,
            source_checks,
        ),
    )
    return marker


def validate_submission_commands(
    manifest: dict[str, Any],
    commands: list[tuple[str, dict[str, Any]]],
) -> None:
    allocations = manifest["schedule"]["allocations"]
    require_equal(len(commands), 6, "submission command count")
    require_equal(len(allocations), 6, "manifest allocation count")
    for (command, summary), allocation in zip(commands, allocations):
        if not isinstance(command, str) or not command:
            raise ContractError("rendered submission command must be non-empty")
        require_equal(
            summary.get("allocation_id"),
            allocation["allocation_id"],
            "submission allocation ID",
        )
        require_equal(
            summary.get("allocation_index"),
            allocation["allocation_index"],
            f"{allocation['allocation_id']} allocation index",
        )
        require_equal(
            summary.get("design_row_index"),
            allocation["design_row_index"],
            f"{allocation['allocation_id']} design row",
        )
        require_equal(
            summary.get("candidate_order"),
            allocation["candidate_order"],
            f"{allocation['allocation_id']} candidate order",
        )
        manifest_generator.require_sha256(
            summary.get("block_plan_sha256"),
            f"{allocation['allocation_id']} block-plan SHA256",
        )


def invalidation_evidence_path(
    runtime_dir: Path,
    ledger_revision: int,
    allocation_id: str | None,
) -> Path:
    """Return the one immutable path for a complete-ledger invalidation."""

    if (
        isinstance(ledger_revision, bool)
        or not isinstance(ledger_revision, int)
        or ledger_revision < 1
    ):
        raise ContractError("invalidation ledger revision must be positive")
    suffix = allocation_id if allocation_id is not None else "blocked"
    if re.fullmatch(r"[A-Za-z0-9_.-]+", suffix) is None:
        raise ContractError("invalidation allocation ID is unsafe")
    return (
        runtime_dir.resolve()
        / f"allocation_invalidation.r{ledger_revision:03d}.{suffix}.json"
    )


def validate_complete_invalidation_evidence(
    evidence: Any,
    *,
    manifest: dict[str, Any],
    manifest_file_sha256: str,
    allocation_id: str,
    prior_submission: dict[str, Any],
    parent_ledger_whole_file_sha256: str,
    parent_ledger_internal_sha256: str,
    parent_ledger_revision: int,
) -> dict[str, Any]:
    """Validate exact aggregator proof before replacing a Complete job."""

    if not isinstance(evidence, dict):
        raise ContractError(
            "Complete allocation replacement requires invalidation evidence"
        )
    expected_path = invalidation_evidence_path(
        Path(manifest["runtime"]["local_runtime_path"]),
        parent_ledger_revision,
        allocation_id,
    )
    path = Path(str(evidence.get("path", ""))).resolve()
    require_equal(path, expected_path, "invalidation-evidence path")
    payload, whole_file_sha256 = manifest_generator.load_exact_json(
        path,
        evidence.get("whole_file_sha256"),
        "Complete-allocation invalidation evidence",
    )
    manifest_generator.validate_internal_digest(
        payload,
        "invalidation_sha256",
        "Complete-allocation invalidation evidence",
    )
    require_equal(
        whole_file_sha256,
        evidence.get("whole_file_sha256"),
        "invalidation-evidence whole-file SHA256",
    )
    require_equal(
        payload["invalidation_sha256"],
        evidence.get("internal_sha256"),
        "invalidation-evidence internal SHA256",
    )
    for key, expected in (
        ("schema_version", 1),
        (
            "evidence_id",
            "dino_post_front_complete_invalid_allocation_v1",
        ),
        ("status", "single_allocation_invalid_replacement_authorized"),
        ("manifest_id", manifest["manifest_id"]),
        ("manifest_sha256", manifest_file_sha256),
        ("manifest_internal_sha256", manifest["manifest_sha256"]),
        ("schedule_sha256", manifest["schedule"]["schedule_sha256"]),
        ("allocation_id", allocation_id),
        ("all_jobs_complete", True),
        ("replacement_permitted", True),
        ("full_allocation_discarded", True),
        ("partial_measurements_reused", False),
        ("partial_measurements_used_for_analysis", False),
        ("feeds_final_selection", False),
        ("feeds_reselection", False),
        ("selection_time_objectives_replaced", False),
    ):
        require_equal(
            payload.get(key),
            expected,
            f"invalidation evidence {key}",
        )
    attribution = payload.get("attribution")
    if not isinstance(attribution, dict):
        raise ContractError("invalidation evidence attribution is missing")
    for key, expected in (
        ("status", "exactly_one_allocation"),
        ("allocation_ids", [allocation_id]),
        ("allocation_count", 1),
        ("replacement_blocked", False),
        ("block_reason", None),
    ):
        require_equal(
            attribution.get(key),
            expected,
            f"invalidation attribution {key}",
        )
    parent = payload.get("complete_ledger")
    if not isinstance(parent, dict):
        raise ContractError("invalidation evidence ledger binding is missing")
    for key, expected in (
        (
            "path",
            str(
                Path(manifest["runtime"]["local_runtime_path"]).resolve()
                / "block_submissions.json"
            ),
        ),
        ("whole_file_sha256", parent_ledger_whole_file_sha256),
        ("internal_sha256", parent_ledger_internal_sha256),
        ("revision", parent_ledger_revision),
        ("status", "complete"),
    ):
        require_equal(
            parent.get(key),
            expected,
            f"invalidation ledger binding {key}",
        )
    identity = payload.get("prior_submission_identity")
    if not isinstance(identity, dict):
        raise ContractError("invalidation prior-job identity is missing")
    for key, expected in (
        ("allocation_id", allocation_id),
        ("tao_job_id", prior_submission["tao_job_id"]),
        ("slurm_job_id", str(prior_submission["slurm_job_id"])),
        ("command_sha256", prior_submission["command_sha256"]),
        ("block_plan_sha256", prior_submission["block_plan_sha256"]),
    ):
        require_equal(
            identity.get(key),
            expected,
            f"invalidation prior identity {key}",
        )
    job_status = payload.get("job_status")
    if not isinstance(job_status, dict):
        raise ContractError("invalidation Complete-job status is missing")
    for key, expected in (
        ("sdk_status", "Complete"),
        ("slurm_state", "COMPLETED"),
        ("slurm_exit_code", "0:0"),
        ("complete", True),
    ):
        require_equal(
            job_status.get(key),
            expected,
            f"invalidation job status {key}",
        )
    failure = payload.get("failure")
    if not isinstance(failure, dict):
        raise ContractError("invalidation deterministic failure is missing")
    if failure.get("stage") not in {
        "allocation_result_fetch",
        "semantic_aggregation",
    }:
        raise ContractError("invalidation failure stage is invalid")
    if (
        not isinstance(failure.get("exception_type"), str)
        or not failure["exception_type"]
        or not isinstance(failure.get("message"), str)
        or not failure["message"]
    ):
        raise ContractError("invalidation failure description is invalid")
    manifest_generator.require_sha256(
        failure.get("error_sha256"),
        "invalidation deterministic error SHA256",
    )
    expected_error = copy.deepcopy(failure)
    del expected_error["error_sha256"]
    require_equal(
        manifest_generator.sha256_value(expected_error),
        failure["error_sha256"],
        "invalidation deterministic error digest",
    )
    artifacts = payload.get("available_artifacts")
    if not isinstance(artifacts, list):
        raise ContractError("invalidation artifact inventory is invalid")
    canonical_artifacts = sorted(
        artifacts,
        key=lambda item: (
            str(item.get("kind", "")),
            str(item.get("allocation_id", "")),
            str(item.get("candidate_id", "")),
            str(item.get("rank", "")),
            str(item.get("path", "")),
        ),
    )
    require_equal(
        artifacts,
        canonical_artifacts,
        "invalidation canonical artifact inventory",
    )
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ContractError("invalidation artifact entry is invalid")
        if artifact.get("kind") not in {
            "allocation_result",
            "rank_record",
        }:
            raise ContractError("invalidation artifact kind is invalid")
        require_equal(
            artifact.get("allocation_id"),
            allocation_id,
            "invalidation artifact allocation",
        )
        if (
            not isinstance(artifact.get("path"), str)
            or not artifact["path"]
        ):
            raise ContractError("invalidation artifact path is invalid")
        manifest_generator.require_sha256(
            artifact.get("sha256"),
            "invalidation artifact SHA256",
        )
    require_equal(
        payload.get("available_artifact_count"),
        len(artifacts),
        "invalidation artifact count",
    )
    require_equal(
        payload.get("available_artifacts_sha256"),
        manifest_generator.sha256_value(artifacts),
        "invalidation artifact-inventory SHA256",
    )
    if not isinstance(payload.get("artifact_probe"), dict):
        raise ContractError("invalidation artifact-probe evidence is missing")
    aggregator_source = payload.get("aggregator_source")
    expected_source = (
        manifest.get("source_artifacts", {})
        .get("post_front_tools", {})
        .get(AGGREGATOR.name)
    )
    if not isinstance(expected_source, dict):
        raise ContractError(
            "manifest lacks pinned post-front aggregator provenance"
        )
    if not isinstance(aggregator_source, dict):
        raise ContractError("invalidation aggregator provenance is missing")
    for key in ("path", "sha256", "git_blob", "head_git_blob"):
        require_equal(
            aggregator_source.get(key),
            expected_source.get(key),
            f"invalidation aggregator source {key}",
        )
    return payload


def reject_complete_replacement_after_analysis(
    *,
    manifest: dict[str, Any],
    manifest_file_sha256: str,
    parent_ledger_whole_file_sha256: str,
    parent_ledger_revision: int,
) -> None:
    """A successful immutable analysis proves the Complete block is valid."""

    path = (
        Path(manifest["runtime"]["local_runtime_path"]).resolve()
        / "post_front_matched_analysis.json"
    )
    if not path.exists():
        return
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o022
    ):
        raise ContractError(
            "Complete allocation analysis marker exists but is unsafe"
        )
    report = manifest_generator.load_json(path)
    manifest_generator.validate_internal_digest(
        report,
        "report_sha256",
        "existing post-front matched analysis",
    )
    for key, expected in (
        ("schema_version", 1),
        ("status", "complete"),
        ("manifest_id", manifest["manifest_id"]),
        ("manifest_sha256", manifest_file_sha256),
        (
            "submission_ledger_sha256",
            parent_ledger_whole_file_sha256,
        ),
        ("submission_ledger_revision", parent_ledger_revision),
    ):
        require_equal(
            report.get(key),
            expected,
            f"existing post-front analysis {key}",
        )
    raise ContractError(
        "Complete allocation has a valid immutable analysis; "
        "replacement is forbidden"
    )


def validate_replacement_intent_evidence(
    evidence: Any,
    *,
    manifest: dict[str, Any],
    manifest_file_sha256: str,
    supersession: dict[str, Any],
    expected_revision: int,
) -> dict[str, Any]:
    if not isinstance(evidence, dict):
        raise ContractError("replacement-intent evidence is missing")
    expected_path = (
        Path(manifest["runtime"]["local_runtime_path"]).resolve()
        / f"replacement_intent.r{expected_revision:03d}.json"
    )
    path = Path(str(evidence.get("path", ""))).resolve()
    require_equal(path, expected_path, "replacement-intent path")
    intent, whole_file_sha256 = manifest_generator.load_exact_json(
        path,
        evidence.get("whole_file_sha256"),
        "complete-allocation replacement intent",
    )
    manifest_generator.validate_internal_digest(
        intent,
        "intent_sha256",
        "complete-allocation replacement intent",
    )
    require_equal(
        whole_file_sha256,
        evidence.get("whole_file_sha256"),
        "replacement-intent whole-file SHA256",
    )
    require_equal(
        intent["intent_sha256"],
        evidence.get("internal_sha256"),
        "replacement-intent internal SHA256",
    )
    prior = supersession["prior_submission"]
    prior_status = supersession["prior_sdk_status"]
    replacement_basis = (
        "aggregator_single_allocation_invalidation"
        if prior_status == "Complete"
        else "sdk_terminal_failure"
    )
    require_equal(
        supersession.get("replacement_basis"),
        replacement_basis,
        "replacement supersession basis",
    )
    for key, expected in (
        ("schema_version", 1),
        (
            "intent_id",
            "dino_post_front_complete_allocation_replacement_v1",
        ),
        ("manifest_id", manifest["manifest_id"]),
        ("manifest_sha256", manifest_file_sha256),
        ("allocation_id", supersession["allocation_id"]),
        ("command_sha256", prior["command_sha256"]),
        ("ledger_revision", expected_revision),
        (
            "parent_ledger_whole_file_sha256",
            supersession["parent_ledger_whole_file_sha256"],
        ),
        (
            "parent_ledger_internal_sha256",
            supersession["parent_ledger_internal_sha256"],
        ),
        ("prior_tao_job_id", prior["tao_job_id"]),
        ("prior_slurm_job_id", str(prior["slurm_job_id"])),
        ("prior_sdk_status", prior_status),
        ("replacement_basis", replacement_basis),
        (
            "invalidation_evidence",
            supersession.get("invalidation_evidence"),
        ),
        ("partial_measurements_reused", False),
        ("feeds_final_selection", False),
        ("feeds_reselection", False),
    ):
        require_equal(
            intent.get(key),
            expected,
            f"replacement intent {key}",
        )
    if prior_status == "Complete":
        validate_complete_invalidation_evidence(
            supersession.get("invalidation_evidence"),
            manifest=manifest,
            manifest_file_sha256=manifest_file_sha256,
            allocation_id=supersession["allocation_id"],
            prior_submission=prior,
            parent_ledger_whole_file_sha256=supersession[
                "parent_ledger_whole_file_sha256"
            ],
            parent_ledger_internal_sha256=supersession[
                "parent_ledger_internal_sha256"
            ],
            parent_ledger_revision=expected_revision - 1,
        )
    else:
        require_equal(
            supersession.get("invalidation_evidence"),
            None,
            "terminal replacement invalidation evidence",
        )
    return intent


def validate_recovery_event_reconciliation(
    event: dict[str, Any],
    expected_command_sha256: str,
) -> dict[str, Any]:
    """Validate the write-ahead recovery proof without trusting SDK order."""

    reconciliation = event.get("reconciliation")
    if not isinstance(reconciliation, dict):
        raise ContractError("submission recovery reconciliation is missing")
    before = reconciliation.get("sdk_job_ids_before")
    observed = reconciliation.get("sdk_job_ids_observed")
    delta = reconciliation.get("sdk_job_id_delta")
    for value, label in (
        (before, "before"),
        (observed, "observed"),
        (delta, "delta"),
    ):
        if (
            not isinstance(value, list)
            or value != sorted(value)
            or len(value) != len(set(value))
            or any(not isinstance(item, str) or not item for item in value)
        ):
            raise ContractError(
                f"submission recovery SDK {label} set is invalid"
            )
    require_equal(
        sorted(set(observed) - set(before)),
        delta,
        "submission recovery exact SDK delta",
    )
    require_equal(
        delta,
        [event.get("tao_job_id")],
        "submission recovery TAO job delta",
    )
    for key, expected in (
        ("delta_is_exactly_one", True),
        ("command_sha256", expected_command_sha256),
        ("duplicate_submission_permitted", False),
    ):
        require_equal(
            reconciliation.get(key),
            expected,
            f"submission recovery reconciliation {key}",
        )
    expected_decision = (
        "terminal_job_excluded_and_complete_block_resubmitted"
        if event.get("reason")
        == "durably_terminal_submission_not_reused"
        else "pre_scheduler_job_terminalized_then_resubmit"
    )
    require_equal(
        reconciliation.get("decision"),
        expected_decision,
        "submission recovery reconciliation decision",
    )
    return reconciliation


def load_complete_ledger_for_replacement(
    path: Path,
    supplied_sha256: str,
    manifest: dict[str, Any],
    manifest_file_sha256: str,
    commands: list[tuple[str, dict[str, Any]]],
    expected_source_checks: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    expected_path = (
        Path(manifest["runtime"]["local_runtime_path"]).resolve()
        / "block_submissions.json"
    )
    require_equal(path.resolve(), expected_path, "replacement ledger path")
    ledger, whole_file_sha256 = manifest_generator.load_exact_json(
        path,
        supplied_sha256,
        "post-front replacement source ledger",
    )
    manifest_generator.validate_internal_digest(
        ledger,
        "ledger_sha256",
        "post-front replacement source ledger",
    )
    for key, expected in (
        ("schema_version", 1),
        ("status", "complete"),
        ("phase", "expanded_post_front_matched_latency"),
        ("manifest_id", manifest["manifest_id"]),
        ("manifest_sha256", manifest_file_sha256),
        ("schedule_sha256", manifest["schedule"]["schedule_sha256"]),
        ("expected_allocation_count", 6),
        ("allocation_count", 6),
        ("pending_submission", None),
        ("feeds_final_selection", False),
        ("feeds_reselection", False),
        ("selection_time_objective_replacement_permitted", False),
    ):
        require_equal(ledger.get(key), expected, f"replacement ledger {key}")
    revision = ledger.get("ledger_revision")
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 1
    ):
        raise ContractError("replacement ledger revision must be positive")
    history = ledger.get("superseded_submissions")
    if not isinstance(history, list) or len(history) != revision - 1:
        raise ContractError(
            "replacement ledger supersession history/revision is invalid"
        )
    recovery_events = ledger.get("submission_recovery_events")
    if not isinstance(recovery_events, list):
        raise ContractError(
            "replacement ledger initial recovery events are invalid"
        )
    if revision == 1:
        require_equal(
            ledger.get("parent_ledger_sha256"),
            None,
            "initial ledger parent",
        )
    else:
        parent_sha256 = manifest_generator.require_sha256(
            ledger.get("parent_ledger_sha256"),
            "replacement ledger parent SHA256",
        )
        require_equal(
            history[-1].get("parent_ledger_whole_file_sha256"),
            parent_sha256,
            "replacement ledger latest parent SHA256",
        )
    source_checks = ledger.get("source_checks")
    if not isinstance(source_checks, dict):
        raise ContractError("replacement ledger source checks are missing")
    for key, expected in expected_source_checks.items():
        require_equal(
            source_checks.get(key),
            expected,
            f"replacement ledger source check {key}",
        )
    launch_source = source_checks.get("launch_contract")
    if not isinstance(launch_source, dict):
        raise ContractError("replacement ledger launch contract is missing")
    launch_path = Path(str(launch_source.get("path", ""))).resolve()
    require_equal(
        launch_path,
        expected_path.parent / LAUNCH_CONTRACT_NAME,
        "replacement launch-contract path",
    )
    launch_contract, launch_file_sha256 = (
        manifest_generator.load_exact_json(
            launch_path,
            launch_source.get("whole_file_sha256"),
            "replacement launch contract",
        )
    )
    manifest_generator.validate_internal_digest(
        launch_contract,
        "contract_sha256",
        "replacement launch contract",
    )
    require_equal(
        launch_file_sha256,
        launch_source.get("whole_file_sha256"),
        "replacement launch-contract whole-file SHA256",
    )
    require_equal(
        launch_contract["contract_sha256"],
        launch_source.get("internal_sha256"),
        "replacement launch-contract internal SHA256",
    )
    launch_source_checks = copy.deepcopy(source_checks)
    del launch_source_checks["launch_contract"]
    require_equal(
        launch_contract.get("source_checks"),
        launch_source_checks,
        "replacement launch/ledger source checks",
    )

    validate_submission_commands(manifest, commands)
    expected_summaries = {
        summary["allocation_id"]: summary for _, summary in commands
    }
    submissions = ledger.get("submissions")
    if not isinstance(submissions, list) or len(submissions) != 6:
        raise ContractError("replacement ledger must have six submissions")
    actual_ids = [item.get("allocation_id") for item in submissions]
    require_equal(
        actual_ids,
        [
            item["allocation_id"]
            for item in manifest["schedule"]["allocations"]
        ],
        "replacement ledger allocation order",
    )
    tao_ids: set[str] = set()
    slurm_ids: set[str] = set()
    for submission in submissions:
        allocation_id = submission["allocation_id"]
        expected = expected_summaries[allocation_id]
        for key in (
            "allocation_index",
            "design_row_index",
            "candidate_order",
            "candidate_count",
            "block_plan_sha256",
            "command_sha256",
            "staging_file_sha256",
        ):
            require_equal(
                submission.get(key),
                expected.get(key),
                f"{allocation_id} replacement source {key}",
            )
        for key in ("feeds_final_selection", "feeds_reselection"):
            require_equal(
                submission.get(key),
                False,
                f"{allocation_id} replacement source {key}",
            )
        require_equal(
            submission.get("launch_uncertain"),
            False,
            f"{allocation_id} replacement source launch uncertainty",
        )
        tao_job_id = submission.get("tao_job_id")
        slurm_job_id = str(submission.get("slurm_job_id", ""))
        if not isinstance(tao_job_id, str) or not tao_job_id:
            raise ContractError(
                f"{allocation_id} replacement source TAO job ID is invalid"
            )
        if not slurm_job_id.isdigit():
            raise ContractError(
                f"{allocation_id} replacement source SLURM job ID is invalid"
            )
        if tao_job_id in tao_ids or slurm_job_id in slurm_ids:
            raise ContractError(
                "replacement source TAO and SLURM IDs must be unique"
            )
        tao_ids.add(tao_job_id)
        slurm_ids.add(slurm_job_id)
    for index, supersession in enumerate(history):
        if not isinstance(supersession, dict):
            raise ContractError("replacement supersession record is invalid")
        allocation_id = supersession.get("allocation_id")
        if allocation_id not in expected_summaries:
            raise ContractError(
                f"replacement supersession {index} allocation is invalid"
            )
        prior_status = supersession.get("prior_sdk_status")
        if prior_status not in {"Error", "Canceled", "Complete"}:
            raise ContractError(
                f"replacement supersession {index} status is invalid"
            )
        expected_reason = (
            "complete_but_semantically_invalid_allocation"
            if prior_status == "Complete"
            else "durable_terminal_incomplete_allocation"
        )
        require_equal(
            supersession.get("reason"),
            expected_reason,
            f"replacement supersession {index} reason",
        )
        require_equal(
            supersession.get("incomplete_allocation_policy"),
            manifest["incomplete_allocation_policy"],
            f"replacement supersession {index} policy",
        )
        require_equal(
            supersession.get("partial_measurements_reused"),
            False,
            f"replacement supersession {index} partial reuse",
        )
        manifest_generator.require_sha256(
            supersession.get("parent_ledger_whole_file_sha256"),
            f"replacement supersession {index} parent SHA256",
        )
        manifest_generator.require_sha256(
            supersession.get("parent_ledger_internal_sha256"),
            f"replacement supersession {index} parent internal SHA256",
        )
        prior = supersession.get("prior_submission")
        if not isinstance(prior, dict):
            raise ContractError(
                f"replacement supersession {index} prior submission is invalid"
            )
        require_equal(
            prior.get("allocation_id"),
            allocation_id,
            f"replacement supersession {index} allocation binding",
        )
        prior_tao_id = prior.get("tao_job_id")
        prior_slurm_id = str(prior.get("slurm_job_id", ""))
        if (
            not isinstance(prior_tao_id, str)
            or not prior_tao_id
            or not prior_slurm_id.isdigit()
            or prior_tao_id in tao_ids
            or prior_slurm_id in slurm_ids
        ):
            raise ContractError(
                "replacement supersession job identity is invalid or reused"
            )
        tao_ids.add(prior_tao_id)
        slurm_ids.add(prior_slurm_id)
        validate_replacement_intent_evidence(
            supersession.get("replacement_intent"),
            manifest=manifest,
            manifest_file_sha256=manifest_file_sha256,
            supersession=supersession,
            expected_revision=index + 2,
        )
    for index, event in enumerate(recovery_events):
        if not isinstance(event, dict):
            raise ContractError(
                "replacement ledger recovery event is invalid"
            )
        allocation_id = event.get("allocation_id")
        expected = expected_summaries.get(allocation_id)
        tao_id = event.get("tao_job_id")
        slurm_id = str(event.get("slurm_job_id", ""))
        if (
            event.get("event_index") != index
            or expected is None
            or event.get("command_sha256")
            != expected["command_sha256"]
            or event.get("reason")
            not in {
                "durably_terminal_submission_not_reused",
                "proven_pre_scheduler_submission_abandoned",
            }
            or event.get("sdk_status") not in {"Error", "Canceled"}
            or event.get("partial_measurements_reused") is not False
            or not isinstance(tao_id, str)
            or not tao_id
            or tao_id in tao_ids
            or (slurm_id and (not slurm_id.isdigit() or slurm_id in slurm_ids))
        ):
            raise ContractError(
                "replacement ledger recovery event is invalid"
            )
        validate_recovery_event_reconciliation(
            event,
            expected["command_sha256"],
        )
        tao_ids.add(tao_id)
        if slurm_id:
            slurm_ids.add(slurm_id)
    return ledger, whole_file_sha256


def configure_slurm_sdk_environment(runtime: dict[str, Any]) -> None:
    os.environ["SLURM_USE_SQSH"] = "false"
    os.environ["SLURM_USE_REQUEUE"] = "false"
    os.environ["SLURM_USE_TIMEOUT"] = "true"
    os.environ["SLURM_MAX_GPUS_PER_NODE"] = "8"
    os.environ["SLURM_PARTITION"] = runtime["partition"]
    os.environ["SLURM_ACCOUNT"] = runtime["account"]
    os.environ["SLURM_TIME_HOURS"] = str(runtime["slurm_time_hours"])
    os.environ["SLURM_TIMEOUT_HOURS"] = str(runtime["slurm_timeout_hours"])


def durable_sdk_job_ids(sdk: Any) -> set[str]:
    rows = sdk.list_jobs()
    if not isinstance(rows, list):
        raise ContractError("SDK durable job listing is invalid")
    identifiers: list[str] = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("backend_type") != "slurm"
            or not isinstance(row.get("job_id"), str)
            or not row["job_id"]
        ):
            raise ContractError("SDK durable state contains an invalid job")
        identifiers.append(row["job_id"])
    if len(set(identifiers)) != len(identifiers):
        raise ContractError("SDK durable state contains duplicate job IDs")
    return set(identifiers)


def verified_submission_record(
    *,
    sdk: Any,
    job_id: str,
    command: str,
    summary: dict[str, Any],
    containment: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(job_id, str) or not job_id:
        raise ContractError("SDK returned an invalid TAO job ID")
    sdk._load_job_from_store(job_id)
    identity = sdk._handler.get_job_runtime_identity(job_id)
    slurm_job_id = str(identity.get("slurm_job_id", ""))
    if not slurm_job_id.isdigit():
        raise ContractError(
            f"{summary['allocation_id']}: invalid SLURM job ID"
        )
    require_equal(
        identity.get("launch_uncertain", False),
        False,
        f"{summary['allocation_id']} launch uncertainty",
    )
    remote_entrypoint = str(identity.get("remote_entrypoint", ""))
    if not remote_entrypoint:
        raise ContractError(
            f"{summary['allocation_id']}: remote entrypoint is missing"
        )
    actual_entrypoint = sdk._handler._read_remote_log_file(
        remote_entrypoint
    )
    expected_entrypoint = sdk._handler._build_entrypoint_script(command)
    if actual_entrypoint is None or sha256_bytes(
        actual_entrypoint.encode("utf-8")
    ) != sha256_bytes(expected_entrypoint.encode("utf-8")):
        raise ContractError(
            f"{summary['allocation_id']}: remote entrypoint digest mismatch"
        )
    results_uri = sdk.get_job_results_dir(job_id)
    if not isinstance(results_uri, str) or not results_uri:
        raise ContractError(
            f"{summary['allocation_id']}: invalid SDK results URI"
        )
    return {
        **summary,
        "tao_job_id": job_id,
        "slurm_job_id": slurm_job_id,
        "retry_count": identity.get("retry_count", 0),
        "failed_slurm_job_ids": identity.get(
            "failed_slurm_job_ids",
            [],
        ),
        "launch_uncertain": False,
        "sdk_results_uri": results_uri,
        "remote_entrypoint_sha256": sha256_bytes(
            expected_entrypoint.encode("utf-8")
        ),
        "remote_sdk_containment": copy.deepcopy(containment),
        "feeds_final_selection": False,
        "feeds_reselection": False,
    }


def validate_launch_contract_binding(
    *,
    ledger: dict[str, Any],
    ledger_path: Path,
    manifest: dict[str, Any],
    manifest_file_sha256: str,
    expected_source_checks: dict[str, Any],
) -> None:
    source_checks = ledger.get("source_checks")
    if not isinstance(source_checks, dict):
        raise ContractError("incomplete ledger source checks are missing")
    for key, expected in expected_source_checks.items():
        require_equal(
            source_checks.get(key),
            expected,
            f"incomplete ledger source check {key}",
        )
    launch_source = source_checks.get("launch_contract")
    if not isinstance(launch_source, dict):
        raise ContractError("incomplete ledger launch contract is missing")
    launch_path = Path(str(launch_source.get("path", ""))).resolve()
    require_equal(
        launch_path,
        ledger_path.parent / LAUNCH_CONTRACT_NAME,
        "incomplete launch-contract path",
    )
    contract, whole_sha256 = manifest_generator.load_exact_json(
        launch_path,
        launch_source.get("whole_file_sha256"),
        "incomplete launch contract",
    )
    manifest_generator.validate_internal_digest(
        contract,
        "contract_sha256",
        "incomplete launch contract",
    )
    require_equal(
        whole_sha256,
        launch_source.get("whole_file_sha256"),
        "incomplete launch-contract whole SHA256",
    )
    require_equal(
        contract.get("contract_sha256"),
        launch_source.get("internal_sha256"),
        "incomplete launch-contract internal SHA256",
    )
    for key, expected in (
        ("manifest_id", manifest["manifest_id"]),
        ("manifest_sha256", manifest_file_sha256),
        ("schedule_sha256", manifest["schedule"]["schedule_sha256"]),
        ("runtime_path", str(ledger_path.parent.resolve())),
        ("allocation_count", 6),
        ("feeds_final_selection", False),
        ("feeds_reselection", False),
        ("manual_winner_override_permitted", False),
    ):
        require_equal(
            contract.get(key),
            expected,
            f"incomplete launch contract {key}",
        )
    contract_sources = copy.deepcopy(source_checks)
    del contract_sources["launch_contract"]
    require_equal(
        contract,
        launch_contract_payload(
            manifest,
            manifest_file_sha256,
            ledger_path.parent,
            contract_sources,
        ),
        "reconstructed incomplete launch contract",
    )
    require_equal(
        contract.get("source_checks"),
        contract_sources,
        "incomplete launch/ledger source checks",
    )


def load_incomplete_ledger_for_resume(
    *,
    path: Path,
    supplied_sha256: str,
    manifest: dict[str, Any],
    manifest_file_sha256: str,
    commands: list[tuple[str, dict[str, Any]]],
    expected_source_checks: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    expected_path = (
        Path(manifest["runtime"]["local_runtime_path"]).resolve()
        / "block_submissions.json"
    )
    require_equal(path.resolve(), expected_path, "resume ledger path")
    ledger, whole_sha256 = manifest_generator.load_exact_json(
        path,
        supplied_sha256,
        "post-front incomplete submission ledger",
    )
    manifest_generator.validate_internal_digest(
        ledger,
        "ledger_sha256",
        "post-front incomplete submission ledger",
    )
    for key, expected in (
        ("schema_version", 1),
        ("phase", "expanded_post_front_matched_latency"),
        ("manifest_id", manifest["manifest_id"]),
        ("manifest_sha256", manifest_file_sha256),
        ("schedule_sha256", manifest["schedule"]["schedule_sha256"]),
        ("expected_allocation_count", 6),
        ("feeds_final_selection", False),
        ("feeds_reselection", False),
        ("selection_time_objective_replacement_permitted", False),
    ):
        require_equal(ledger.get(key), expected, f"resume ledger {key}")
    if ledger.get("status") not in {
        "submitting_incomplete",
        "replacement_in_progress",
    }:
        raise ContractError("resume ledger is not an incomplete submission")
    validate_submission_commands(manifest, commands)
    expected_summaries = {
        summary["allocation_id"]: summary for _, summary in commands
    }
    submissions = ledger.get("submissions")
    if not isinstance(submissions, list) or len(submissions) > 6:
        raise ContractError("resume ledger submissions are invalid")
    require_equal(
        ledger.get("allocation_count"),
        len(submissions),
        "resume ledger allocation count",
    )
    expected_prefix = [
        allocation["allocation_id"]
        for allocation in manifest["schedule"]["allocations"]
    ][: len(submissions)]
    historical_tao_ids: set[str] = set()
    historical_slurm_ids: set[str] = set()
    if ledger["status"] == "submitting_incomplete":
        require_equal(
            [item.get("allocation_id") for item in submissions],
            expected_prefix,
            "resume ledger allocation prefix",
        )
        require_equal(ledger.get("ledger_revision"), 1, "initial revision")
        require_equal(
            ledger.get("superseded_submissions"),
            [],
            "initial supersession history",
        )
        require_equal(
            ledger.get("parent_ledger_sha256"),
            None,
            "initial parent ledger",
        )
    else:
        if len(submissions) != 6:
            raise ContractError(
                "replacement resume must retain six effective submissions"
            )
        revision = ledger.get("ledger_revision")
        history = ledger.get("superseded_submissions")
        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 2
            or not isinstance(history, list)
            or len(history) != revision - 1
        ):
            raise ContractError("replacement resume revision/history is invalid")
        parent_sha256 = manifest_generator.require_sha256(
            ledger.get("parent_ledger_sha256"),
            "replacement resume parent SHA256",
        )
        require_equal(
            history[-1].get("parent_ledger_whole_file_sha256"),
            parent_sha256,
            "replacement resume latest parent SHA256",
        )
        for index, supersession in enumerate(history):
            prior_status = (
                supersession.get("prior_sdk_status")
                if isinstance(supersession, dict)
                else None
            )
            expected_reason = (
                "complete_but_semantically_invalid_allocation"
                if prior_status == "Complete"
                else "durable_terminal_incomplete_allocation"
            )
            if (
                not isinstance(supersession, dict)
                or supersession.get("reason")
                != expected_reason
                or supersession.get("incomplete_allocation_policy")
                != manifest["incomplete_allocation_policy"]
                or prior_status not in {"Error", "Canceled", "Complete"}
                or supersession.get("partial_measurements_reused")
                is not False
            ):
                raise ContractError(
                    "replacement resume supersession is invalid"
                )
            manifest_generator.require_sha256(
                supersession.get("parent_ledger_internal_sha256"),
                "replacement resume parent internal SHA256",
            )
            prior = supersession.get("prior_submission")
            if not isinstance(prior, dict):
                raise ContractError(
                    "replacement resume prior submission is invalid"
                )
            prior_tao = prior.get("tao_job_id")
            prior_slurm = str(prior.get("slurm_job_id", ""))
            if (
                not isinstance(prior_tao, str)
                or not prior_tao
                or not prior_slurm.isdigit()
                or prior_tao in historical_tao_ids
                or prior_slurm in historical_slurm_ids
            ):
                raise ContractError(
                    "replacement resume historical identity is invalid"
                )
            historical_tao_ids.add(prior_tao)
            historical_slurm_ids.add(prior_slurm)
            validate_replacement_intent_evidence(
                supersession.get("replacement_intent"),
                manifest=manifest,
                manifest_file_sha256=manifest_file_sha256,
                supersession=supersession,
                expected_revision=index + 2,
            )
        latest_prior = history[-1]["prior_submission"]
        latest_effective = next(
            item
            for item in submissions
            if item["allocation_id"] == history[-1]["allocation_id"]
        )
        require_equal(
            latest_effective,
            latest_prior,
            "replacement resume effective prior binding",
        )
    tao_ids: set[str] = set()
    slurm_ids: set[str] = set()
    for submission in submissions:
        allocation_id = submission.get("allocation_id")
        if allocation_id not in expected_summaries:
            raise ContractError("resume ledger allocation is unknown")
        expected = expected_summaries[allocation_id]
        for key in (
            "allocation_index",
            "design_row_index",
            "candidate_order",
            "candidate_count",
            "block_plan_sha256",
            "command_sha256",
            "staging_file_sha256",
        ):
            require_equal(
                submission.get(key),
                expected.get(key),
                f"{allocation_id} resume submission {key}",
            )
        tao_id = submission.get("tao_job_id")
        slurm_id = str(submission.get("slurm_job_id", ""))
        if (
            not isinstance(tao_id, str)
            or not tao_id
            or not slurm_id.isdigit()
            or tao_id in tao_ids
            or slurm_id in slurm_ids
        ):
            raise ContractError("resume ledger job identity is invalid")
        tao_ids.add(tao_id)
        slurm_ids.add(slurm_id)
        reconciliation = submission.get("submission_reconciliation")
        if reconciliation is not None:
            if (
                not isinstance(reconciliation, dict)
                or reconciliation.get("command_sha256")
                != expected["command_sha256"]
                or reconciliation.get("duplicate_submission_permitted")
                is not False
                or reconciliation.get("decision")
                not in {
                    "adopt_exact_preexisting_submission",
                    "proven_absent_then_submit_exactly_once",
                }
            ):
                raise ContractError(
                    "resume adopted-submission reconciliation is invalid"
                )
    if ledger["status"] == "replacement_in_progress":
        latest_prior = history[-1]["prior_submission"]
        require_equal(
            tao_ids & historical_tao_ids,
            {latest_prior["tao_job_id"]},
            "replacement resume active/historical TAO overlap",
        )
        require_equal(
            slurm_ids & historical_slurm_ids,
            {str(latest_prior["slurm_job_id"])},
            "replacement resume active/historical SLURM overlap",
        )
    recovery_events = ledger.get("submission_recovery_events")
    if not isinstance(recovery_events, list):
        raise ContractError("resume recovery-event list is invalid")
    for index, event in enumerate(recovery_events):
        if (
            not isinstance(event, dict)
            or event.get("event_index") != index
            or event.get("allocation_id") not in expected_summaries
            or event.get("partial_measurements_reused") is not False
            or event.get("feeds_final_selection") is not False
            or event.get("feeds_reselection") is not False
        ):
            raise ContractError("resume recovery event is invalid")
        tao_id = event.get("tao_job_id")
        expected = expected_summaries[event["allocation_id"]]
        if (
            event.get("command_sha256") != expected["command_sha256"]
            or event.get("reason")
            not in {
                "durably_terminal_submission_not_reused",
                "proven_pre_scheduler_submission_abandoned",
            }
            or event.get("sdk_status") not in {"Error", "Canceled"}
            or event.get("launch_uncertain") is not False
        ):
            raise ContractError("resume recovery event semantics are invalid")
        validate_recovery_event_reconciliation(
            event,
            expected["command_sha256"],
        )
        if (
            not isinstance(tao_id, str)
            or not tao_id
            or tao_id in tao_ids
            or tao_id in historical_tao_ids
        ):
            raise ContractError("resume recovery TAO job identity is invalid")
        tao_ids.add(tao_id)
        slurm_id = str(event.get("slurm_job_id", ""))
        if slurm_id:
            if (
                not slurm_id.isdigit()
                or slurm_id in slurm_ids
                or slurm_id in historical_slurm_ids
            ):
                raise ContractError(
                    "resume recovery SLURM job identity is invalid"
                )
            slurm_ids.add(slurm_id)
        if (
            event["reason"] == "proven_pre_scheduler_submission_abandoned"
            and slurm_id
        ):
            raise ContractError(
                "pre-scheduler recovery event cannot have a SLURM ID"
            )
    pending = ledger.get("pending_submission")
    if pending is not None:
        if not isinstance(pending, dict):
            raise ContractError("resume pending submission is invalid")
        allocation_id = pending.get("allocation_id")
        if allocation_id not in expected_summaries:
            raise ContractError("resume pending allocation is invalid")
        require_equal(
            pending.get("command_sha256"),
            expected_summaries[allocation_id]["command_sha256"],
            "resume pending command SHA256",
        )
        expected_pending_id = (
            manifest["schedule"]["allocations"][len(submissions)][
                "allocation_id"
            ]
            if ledger["status"] == "submitting_incomplete"
            and len(submissions) < 6
            else ledger["superseded_submissions"][-1]["allocation_id"]
            if ledger["status"] == "replacement_in_progress"
            else None
        )
        require_equal(
            allocation_id,
            expected_pending_id,
            "resume pending allocation position",
        )
        snapshot = pending.get("sdk_job_ids_before")
        if (
            not isinstance(snapshot, list)
            or snapshot != sorted(snapshot)
            or len(set(snapshot)) != len(snapshot)
            or any(not isinstance(item, str) or not item for item in snapshot)
        ):
            raise ContractError("resume SDK job snapshot is invalid")
        manifest_generator.require_sha256(
            pending.get("remote_sdk_containment_sha256"),
            "resume pending remote containment SHA256",
        )
    validate_launch_contract_binding(
        ledger=ledger,
        ledger_path=path.resolve(),
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha256,
        expected_source_checks=expected_source_checks,
    )
    return ledger, whole_sha256


@exclusive_submission_operation
def replacement_submission(
    *,
    manifest: dict[str, Any],
    manifest_file_sha256: str,
    commands: list[tuple[str, dict[str, Any]]],
    runtime_dir: Path,
    source_checks: dict[str, Any],
    allocation_id: str,
    supplied_ledger_sha256: str,
    invalidation_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    command_by_id = {
        summary["allocation_id"]: (command, summary)
        for command, summary in commands
    }
    if allocation_id not in command_by_id:
        raise ContractError(f"unknown replacement allocation: {allocation_id}")
    ledger_path = runtime_dir / "block_submissions.json"
    ledger, parent_file_sha256 = load_complete_ledger_for_replacement(
        ledger_path,
        supplied_ledger_sha256,
        manifest,
        manifest_file_sha256,
        commands,
        source_checks,
    )
    submissions = copy.deepcopy(ledger["submissions"])
    target_index = next(
        index
        for index, item in enumerate(submissions)
        if item["allocation_id"] == allocation_id
    )
    prior = submissions[target_index]
    sdk_path = manifest["runtime"]["sdk_path"]
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)
    from tao_sdk.platforms.slurm import SlurmSDK

    runtime = manifest["runtime"]
    configure_slurm_sdk_environment(runtime)
    sdk = SlurmSDK(
        poll_interval=10,
        state_file=runtime_dir / "slurm_state.json",
    )
    try:
        prior_observation = observe_job_status_no_retry(
            sdk,
            prior["tao_job_id"],
        )
        prior_status = prior_observation["status"]
        if prior_status not in {"Error", "Canceled", "Complete"}:
            raise ContractError(
                f"{allocation_id} is neither durably failed nor Complete: "
                f"{prior_status}"
            )
        prior_identity = prior_observation["runtime_identity"]
        require_equal(
            str(prior_identity.get("slurm_job_id", "")),
            str(prior["slurm_job_id"]),
            f"{allocation_id} prior SLURM identity",
        )
        require_equal(
            prior_identity.get("launch_uncertain", False),
            False,
            f"{allocation_id} prior launch uncertainty",
        )
        parent_internal_sha256 = manifest_generator.require_sha256(
            ledger.get("ledger_sha256"),
            "replacement parent ledger internal SHA256",
        )
        if prior_status == "Complete":
            reject_complete_replacement_after_analysis(
                manifest=manifest,
                manifest_file_sha256=manifest_file_sha256,
                parent_ledger_whole_file_sha256=parent_file_sha256,
                parent_ledger_revision=ledger["ledger_revision"],
            )
            validate_complete_invalidation_evidence(
                invalidation_evidence,
                manifest=manifest,
                manifest_file_sha256=manifest_file_sha256,
                allocation_id=allocation_id,
                prior_submission=prior,
                parent_ledger_whole_file_sha256=parent_file_sha256,
                parent_ledger_internal_sha256=parent_internal_sha256,
                parent_ledger_revision=ledger["ledger_revision"],
            )
        elif invalidation_evidence is not None:
            raise ContractError(
                "Error/Canceled replacement must not supply invalidation evidence"
            )
        command, summary = command_by_id[allocation_id]
        next_revision = ledger["ledger_revision"] + 1
        evidence_reference = copy.deepcopy(invalidation_evidence)
        replacement_basis = (
            "aggregator_single_allocation_invalidation"
            if prior_status == "Complete"
            else "sdk_terminal_failure"
        )
        replacement_intent_path = (
            runtime_dir
            / f"replacement_intent.r{next_revision:03d}.json"
        )
        replacement_intent = {
            "schema_version": 1,
            "intent_id": (
                "dino_post_front_complete_allocation_replacement_v1"
            ),
            "manifest_id": manifest["manifest_id"],
            "manifest_sha256": manifest_file_sha256,
            "allocation_id": allocation_id,
            "command_sha256": summary["command_sha256"],
            "ledger_revision": next_revision,
            "parent_ledger_whole_file_sha256": parent_file_sha256,
            "parent_ledger_internal_sha256": parent_internal_sha256,
            "prior_tao_job_id": prior["tao_job_id"],
            "prior_slurm_job_id": str(prior["slurm_job_id"]),
            "prior_sdk_status": prior_status,
            "replacement_basis": replacement_basis,
            "invalidation_evidence": evidence_reference,
            "partial_measurements_reused": False,
            "feeds_final_selection": False,
            "feeds_reselection": False,
        }
        replacement_intent["intent_sha256"] = (
            manifest_generator.sha256_value(replacement_intent)
        )
        if replacement_intent_path.exists():
            intent_info = replacement_intent_path.lstat()
            if (
                not stat.S_ISREG(intent_info.st_mode)
                or stat.S_ISLNK(intent_info.st_mode)
                or intent_info.st_uid != os.geteuid()
                or stat.S_IMODE(intent_info.st_mode) & 0o022
            ):
                raise ContractError("orphan replacement intent file is unsafe")
            existing_intent = manifest_generator.load_json(
                replacement_intent_path
            )
            manifest_generator.validate_internal_digest(
                existing_intent,
                "intent_sha256",
                "orphan replacement intent",
            )
            require_equal(
                existing_intent,
                replacement_intent,
                "orphan replacement intent replay",
            )
        else:
            atomic_create_json(
                replacement_intent_path,
                replacement_intent,
            )
        history = copy.deepcopy(ledger["superseded_submissions"])
        supersession = {
            "allocation_id": allocation_id,
            "reason": (
                "complete_but_semantically_invalid_allocation"
                if prior_status == "Complete"
                else "durable_terminal_incomplete_allocation"
            ),
            "incomplete_allocation_policy": manifest[
                "incomplete_allocation_policy"
            ],
            "prior_sdk_status": prior_status,
            "prior_sdk_message": prior_observation["message"],
            "prior_status_observed_with_allow_retry": False,
            "prior_submission": copy.deepcopy(prior),
            "parent_ledger_whole_file_sha256": parent_file_sha256,
            "parent_ledger_internal_sha256": parent_internal_sha256,
            "replacement_basis": replacement_basis,
            "invalidation_evidence": evidence_reference,
            "replacement_intent": {
                "path": str(replacement_intent_path),
                "whole_file_sha256": manifest_generator.sha256_file(
                    replacement_intent_path
                ),
                "internal_sha256": replacement_intent["intent_sha256"],
            },
            "partial_measurements_reused": False,
        }
        history.append(supersession)
        containment = enforce_remote_sdk_containment()
        sdk_job_ids_before = sorted(durable_sdk_job_ids(sdk))
        expected_sdk_ids = {
            item["tao_job_id"] for item in submissions
        } | {
            item["prior_submission"]["tao_job_id"]
            for item in history
        } | {
            item["tao_job_id"]
            for item in ledger["submission_recovery_events"]
        }
        require_equal(
            sdk_job_ids_before,
            sorted(expected_sdk_ids),
            "replacement private SDK-state job set",
        )
        atomic_json(
            ledger_path,
            submission_ledger_payload(
                manifest,
                manifest_file_sha256,
                submissions,
                status="replacement_in_progress",
                source_checks=ledger["source_checks"],
                ledger_revision=next_revision,
                superseded_submissions=history,
                submission_recovery_events=ledger[
                    "submission_recovery_events"
                ],
                parent_ledger_sha256=parent_file_sha256,
                pending_submission={
                    "allocation_id": allocation_id,
                    "command_sha256": summary["command_sha256"],
                    "state": "replacement_intent_recorded_before_create_job",
                    "sdk_job_ids_before": sdk_job_ids_before,
                    "remote_sdk_containment_sha256": containment[
                        "evidence_sha256"
                    ],
                },
            ),
        )
        job = sdk.create_job(
            image=runtime["sqsh_path"],
            command=command,
            gpu_count=8,
            num_nodes=1,
            partition=runtime["partition"],
            account=runtime["account"],
            env_vars={"NVIDIA_TF32_OVERRIDE": "0"},
        )
        replacement = verified_submission_record(
            sdk=sdk,
            job_id=job.id,
            command=command,
            summary=summary,
            containment=containment,
        )
        tao_job_id = replacement["tao_job_id"]
        slurm_job_id = replacement["slurm_job_id"]
        prior_tao_ids = {
            item["tao_job_id"] for item in ledger["submissions"]
        }
        prior_slurm_ids = {
            str(item["slurm_job_id"]) for item in ledger["submissions"]
        }
        for item in history:
            historical = item["prior_submission"]
            prior_tao_ids.add(historical["tao_job_id"])
            prior_slurm_ids.add(str(historical["slurm_job_id"]))
        for event in ledger["submission_recovery_events"]:
            prior_tao_ids.add(event["tao_job_id"])
            prior_slurm_id = str(event.get("slurm_job_id", ""))
            if prior_slurm_id:
                prior_slurm_ids.add(prior_slurm_id)
        if tao_job_id in prior_tao_ids or slurm_job_id in prior_slurm_ids:
            raise ContractError("replacement job identity is not globally unique")
        submissions[target_index] = replacement
        atomic_json(
            ledger_path,
            submission_ledger_payload(
                manifest,
                manifest_file_sha256,
                submissions,
                status="complete",
                source_checks=ledger["source_checks"],
                ledger_revision=next_revision,
                superseded_submissions=history,
                submission_recovery_events=ledger[
                    "submission_recovery_events"
                ],
                parent_ledger_sha256=parent_file_sha256,
            ),
        )
        return replacement
    finally:
        sdk._monitor.stop()
        sdk._store.close()


def submission_recovery_event(
    *,
    events: list[dict[str, Any]],
    allocation_id: str,
    command_sha256: str,
    job_id: str,
    identity: dict[str, Any],
    status: str,
    reason: str,
    reconciliation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "event_index": len(events),
        "allocation_id": allocation_id,
        "command_sha256": command_sha256,
        "reason": reason,
        "tao_job_id": job_id,
        "slurm_job_id": str(identity.get("slurm_job_id", "")),
        "sdk_status": status,
        "submission_attempted": bool(
            identity.get("submission_attempted", False)
        ),
        "launch_uncertain": False,
        "reconciliation": copy.deepcopy(reconciliation),
        "partial_measurements_reused": False,
        "feeds_final_selection": False,
        "feeds_reselection": False,
    }


@exclusive_submission_operation
def resume_incomplete_submission(
    *,
    manifest: dict[str, Any],
    manifest_file_sha256: str,
    commands: list[tuple[str, dict[str, Any]]],
    runtime_dir: Path,
    source_checks: dict[str, Any],
    supplied_ledger_sha256: str,
) -> list[dict[str, Any]]:
    ledger_path = runtime_dir / "block_submissions.json"
    ledger, _ = load_incomplete_ledger_for_resume(
        path=ledger_path,
        supplied_sha256=supplied_ledger_sha256,
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha256,
        commands=commands,
        expected_source_checks=source_checks,
    )
    sdk_path = manifest["runtime"]["sdk_path"]
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)
    from tao_sdk.platforms.slurm import SlurmSDK

    runtime = manifest["runtime"]
    configure_slurm_sdk_environment(runtime)
    sdk = SlurmSDK(
        poll_interval=10,
        state_file=runtime_dir / "slurm_state.json",
    )
    command_by_id = {
        summary["allocation_id"]: (command, summary)
        for command, summary in commands
    }
    submissions = copy.deepcopy(ledger["submissions"])
    events = copy.deepcopy(ledger["submission_recovery_events"])
    status = ledger["status"]
    revision = ledger["ledger_revision"]
    history = copy.deepcopy(ledger["superseded_submissions"])
    parent_sha256 = ledger["parent_ledger_sha256"]
    resume_no_delta_proof: dict[str, Any] | None = None

    def write_ledger(
        *,
        ledger_status: str,
        pending: dict[str, Any] | None,
    ) -> None:
        atomic_json(
            ledger_path,
            submission_ledger_payload(
                manifest,
                manifest_file_sha256,
                submissions,
                status=ledger_status,
                source_checks=ledger["source_checks"],
                ledger_revision=revision,
                superseded_submissions=history,
                submission_recovery_events=events,
                parent_ledger_sha256=parent_sha256,
                pending_submission=pending,
            ),
        )

    def known_job_ids() -> set[str]:
        return {
            item["tao_job_id"] for item in submissions
        } | {
            item["tao_job_id"] for item in events
        } | {
            item["prior_submission"]["tao_job_id"]
            for item in history
        }

    def ensure_unique(record: dict[str, Any]) -> None:
        existing_tao = known_job_ids()
        existing_slurm = {
            str(item["slurm_job_id"])
            for item in [
                *submissions,
                *events,
                *[
                    history_item["prior_submission"]
                    for history_item in history
                ],
            ]
            if str(item.get("slurm_job_id", ""))
        }
        if (
            record["tao_job_id"] in existing_tao
            or str(record["slurm_job_id"]) in existing_slurm
        ):
            raise ContractError("recovered job identity is not globally unique")

    def apply_effective_submission(
        allocation_id: str,
        record: dict[str, Any],
    ) -> None:
        ensure_unique(record)
        if status == "submitting_incomplete":
            expected = manifest["schedule"]["allocations"][len(submissions)][
                "allocation_id"
            ]
            require_equal(
                allocation_id,
                expected,
                "recovered initial allocation order",
            )
            submissions.append(record)
            return
        target_index = next(
            index
            for index, item in enumerate(submissions)
            if item["allocation_id"] == allocation_id
        )
        prior_tao = submissions[target_index]["tao_job_id"]
        prior_slurm = str(submissions[target_index]["slurm_job_id"])
        if (
            record["tao_job_id"] == prior_tao
            or str(record["slurm_job_id"]) == prior_slurm
        ):
            raise ContractError("replacement recovery reused prior identity")
        submissions[target_index] = record

    try:
        current_ids = durable_sdk_job_ids(sdk)
        pending = ledger.get("pending_submission")
        if pending is None and current_ids != known_job_ids():
            raise ContractError(
                "SDK durable state has an unbound job before finalization"
            )
        if pending is not None and not known_job_ids().issubset(current_ids):
            raise ContractError(
                "SDK durable state is missing a ledger-bound job identity"
            )
        if pending is not None:
            before = set(pending["sdk_job_ids_before"])
            if before != known_job_ids():
                raise ContractError(
                    "pending SDK snapshot is not the exact bound job set"
                )
            if not before.issubset(current_ids):
                raise ContractError(
                    "SDK durable state regressed after pending intent"
                )
            created = sorted(current_ids - before)
            if len(created) > 1:
                raise ContractError(
                    "ambiguous pending submission created multiple SDK jobs"
                )
            allocation_id = pending["allocation_id"]
            command, summary = command_by_id[allocation_id]
            if created:
                job_id = created[0]
                entry = sdk.get_job(job_id)
                sdk._load_job_from_store(job_id)
                identity = sdk._handler.get_job_runtime_identity(job_id)
                if (
                    not isinstance(entry, dict)
                    or entry.get("backend_type") != "slurm"
                    or entry.get("image") != runtime["sqsh_path"]
                    or str(entry.get("results_dir", "")).rsplit("/", 1)[-1]
                    != job_id
                    or not sdk._backend_identity_matches(
                        job_id,
                        entry=entry,
                    )
                ):
                    raise ContractError(
                        "pending SDK job provenance does not match the launch"
                    )
                reconciliation = {
                    "sdk_job_ids_before": sorted(before),
                    "sdk_job_ids_observed": sorted(current_ids),
                    "sdk_job_id_delta": created,
                    "delta_is_exactly_one": True,
                    "tao_job_id": job_id,
                    "backend_type": entry["backend_type"],
                    "image_matches_manifest": True,
                    "results_uri_job_scoped": True,
                    "backend_identity_matches": True,
                    "command_sha256": summary["command_sha256"],
                    "status_observed_with_allow_retry": False,
                    "duplicate_submission_permitted": False,
                }
                if identity.get("launch_uncertain", False):
                    raise ContractError(
                        "pending SDK launch remains uncertain; refusing duplicate"
                    )
                slurm_id = str(identity.get("slurm_job_id", ""))
                if slurm_id:
                    if not slurm_id.isdigit():
                        raise ContractError(
                            "pending SDK job has invalid scheduler identity"
                        )
                    expected_entrypoint = (
                        sdk._handler._build_entrypoint_script(command)
                    )
                    actual_entrypoint = sdk._handler._read_remote_log_file(
                        str(identity.get("remote_entrypoint", ""))
                    )
                    if (
                        actual_entrypoint is None
                        or sha256_bytes(
                            actual_entrypoint.encode("utf-8")
                        )
                        != sha256_bytes(
                            expected_entrypoint.encode("utf-8")
                        )
                    ):
                        raise ContractError(
                            "pending SDK job remote command does not match intent"
                        )
                    reconciliation["remote_entrypoint_sha256"] = (
                        sha256_bytes(actual_entrypoint.encode("utf-8"))
                    )
                    if entry.get("status") not in {
                        "Pending",
                        "Running",
                        "Complete",
                        "Error",
                        "Canceled",
                    }:
                        raise ContractError(
                            "pending SDK job durable status is not adoptable"
                        )
                    observation = observe_job_status_no_retry(sdk, job_id)
                    reconciliation["observed_sdk_status"] = observation[
                        "status"
                    ]
                    if observation["status"] in {"Error", "Canceled"}:
                        reconciliation["decision"] = (
                            "terminal_job_excluded_and_complete_block_resubmitted"
                        )
                        events.append(
                            submission_recovery_event(
                                events=events,
                                allocation_id=allocation_id,
                                command_sha256=summary["command_sha256"],
                                job_id=job_id,
                                identity=observation["runtime_identity"],
                                status=observation["status"],
                                reason=(
                                    "durably_terminal_submission_not_reused"
                                ),
                                reconciliation=reconciliation,
                            )
                        )
                    else:
                        containment = enforce_remote_sdk_containment()
                        require_equal(
                            containment["evidence_sha256"],
                            pending[
                                "remote_sdk_containment_sha256"
                            ],
                            "recovered containment evidence",
                        )
                        record = verified_submission_record(
                            sdk=sdk,
                            job_id=job_id,
                            command=command,
                            summary=summary,
                            containment=containment,
                        )
                        reconciliation["decision"] = (
                            "adopt_exact_preexisting_submission"
                        )
                        record["submission_reconciliation"] = reconciliation
                        apply_effective_submission(allocation_id, record)
                elif (
                    not identity.get("submission_attempted", False)
                    and isinstance(entry, dict)
                ):
                    staged_entrypoint = sdk._handler._read_remote_log_file(
                        str(identity.get("remote_entrypoint", ""))
                    )
                    if (
                        staged_entrypoint is not None
                        and sha256_bytes(
                            staged_entrypoint.encode("utf-8")
                        )
                        != sha256_bytes(
                            sdk._handler._build_entrypoint_script(
                                command
                            ).encode("utf-8")
                        )
                    ):
                        raise ContractError(
                            "pre-scheduler staged command does not match intent"
                        )
                    if not sdk.cancel_job(job_id):
                        raise ContractError(
                            "could not terminalize abandoned pre-scheduler job"
                        )
                    terminal_entry = sdk.get_job(job_id)
                    terminal_identity = sdk._handler.get_job_runtime_identity(
                        job_id
                    )
                    if (
                        not isinstance(terminal_entry, dict)
                        or terminal_entry.get("status")
                        not in {"Error", "Canceled"}
                        or terminal_identity.get("launch_uncertain", False)
                        or str(terminal_identity.get("slurm_job_id", ""))
                    ):
                        raise ContractError(
                            "abandoned pre-scheduler job is not durably terminal"
                        )
                    reconciliation["remote_entrypoint_sha256"] = (
                        sha256_bytes(staged_entrypoint.encode("utf-8"))
                        if staged_entrypoint is not None
                        else None
                    )
                    reconciliation["observed_sdk_status"] = str(
                        terminal_entry["status"]
                    )
                    reconciliation["decision"] = (
                        "pre_scheduler_job_terminalized_then_resubmit"
                    )
                    events.append(
                        submission_recovery_event(
                            events=events,
                            allocation_id=allocation_id,
                            command_sha256=summary["command_sha256"],
                            job_id=job_id,
                            identity=terminal_identity,
                            status=str(terminal_entry["status"]),
                            reason="proven_pre_scheduler_submission_abandoned",
                            reconciliation=reconciliation,
                        )
                    )
                else:
                    raise ContractError(
                        "pending SDK job has no definitive scheduler identity"
                    )
            else:
                resume_no_delta_proof = {
                    "sdk_job_ids_before": sorted(before),
                    "sdk_job_ids_observed": sorted(current_ids),
                    "sdk_job_id_delta": [],
                    "delta_is_exactly_zero": True,
                    "command_sha256": summary["command_sha256"],
                    "decision": (
                        "proven_absent_then_submit_exactly_once"
                    ),
                    "duplicate_submission_permitted": False,
                }
            write_ledger(ledger_status=status, pending=None)

        while True:
            if status == "submitting_incomplete":
                if len(submissions) == 6:
                    break
                allocation_id = manifest["schedule"]["allocations"][
                    len(submissions)
                ]["allocation_id"]
            else:
                allocation_id = history[-1]["allocation_id"]
                prior_id = history[-1]["prior_submission"]["tao_job_id"]
                effective = next(
                    item
                    for item in submissions
                    if item["allocation_id"] == allocation_id
                )
                if effective["tao_job_id"] != prior_id:
                    break
            command, summary = command_by_id[allocation_id]
            current_ids = durable_sdk_job_ids(sdk)
            if current_ids != known_job_ids():
                raise ContractError(
                    "unbound SDK job exists before resumed submission"
                )
            containment = enforce_remote_sdk_containment()
            intent = {
                "allocation_id": allocation_id,
                "command_sha256": summary["command_sha256"],
                "state": (
                    "resume_intent_recorded_before_create_job"
                    if status == "submitting_incomplete"
                    else "replacement_resume_intent_before_create_job"
                ),
                "sdk_job_ids_before": sorted(current_ids),
                "remote_sdk_containment_sha256": containment[
                    "evidence_sha256"
                ],
            }
            write_ledger(ledger_status=status, pending=intent)
            job = sdk.create_job(
                image=runtime["sqsh_path"],
                command=command,
                gpu_count=8,
                num_nodes=1,
                partition=runtime["partition"],
                account=runtime["account"],
                env_vars={"NVIDIA_TF32_OVERRIDE": "0"},
            )
            record = verified_submission_record(
                sdk=sdk,
                job_id=job.id,
                command=command,
                summary=summary,
                containment=containment,
            )
            if resume_no_delta_proof is not None:
                resumed_proof = copy.deepcopy(resume_no_delta_proof)
                resumed_proof.update(
                    {
                        "new_tao_job_id": record["tao_job_id"],
                        "new_slurm_job_id": record["slurm_job_id"],
                        "remote_entrypoint_sha256": record[
                            "remote_entrypoint_sha256"
                        ],
                    }
                )
                record["submission_reconciliation"] = resumed_proof
                resume_no_delta_proof = None
            apply_effective_submission(allocation_id, record)
            write_ledger(ledger_status=status, pending=None)
        write_ledger(ledger_status="complete", pending=None)
        return submissions
    finally:
        sdk._monitor.stop()
        sdk._store.close()


@exclusive_submission_operation
def submit_all(
    manifest: dict[str, Any],
    manifest_file_sha256: str,
    commands: list[tuple[str, dict[str, Any]]],
    runtime_dir: Path,
    source_checks: dict[str, Any],
) -> list[dict[str, Any]]:
    validate_submission_commands(manifest, commands)
    launch_contract_path = reserve_launch(
        manifest,
        manifest_file_sha256,
        runtime_dir,
        source_checks,
    )
    launch_contract = manifest_generator.load_json(launch_contract_path)
    manifest_generator.validate_internal_digest(
        launch_contract,
        "contract_sha256",
        "post-front launch contract",
    )
    ledger_source_checks = {
        **copy.deepcopy(source_checks),
        "launch_contract": {
            "path": str(launch_contract_path),
            "whole_file_sha256": manifest_generator.sha256_file(
                launch_contract_path
            ),
            "internal_sha256": launch_contract["contract_sha256"],
        },
    }
    ledger_path = runtime_dir / "block_submissions.json"
    atomic_json(
        ledger_path,
        submission_ledger_payload(
            manifest,
            manifest_file_sha256,
            [],
            status="submitting_incomplete",
            source_checks=ledger_source_checks,
        ),
    )
    sdk_path = manifest["runtime"]["sdk_path"]
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)
    from tao_sdk.platforms.slurm import SlurmSDK

    runtime = manifest["runtime"]
    configure_slurm_sdk_environment(runtime)
    sdk = SlurmSDK(
        poll_interval=10,
        state_file=runtime_dir / "slurm_state.json",
    )
    submissions = []
    tao_job_ids: set[str] = set()
    slurm_job_ids: set[str] = set()
    try:
        for command, summary in commands:
            containment = enforce_remote_sdk_containment()
            sdk_job_ids_before = sorted(durable_sdk_job_ids(sdk))
            require_equal(
                sdk_job_ids_before,
                sorted(item["tao_job_id"] for item in submissions),
                "initial launch private SDK-state job set",
            )
            atomic_json(
                ledger_path,
                submission_ledger_payload(
                    manifest,
                    manifest_file_sha256,
                    submissions,
                    status="submitting_incomplete",
                    source_checks=ledger_source_checks,
                    pending_submission={
                        "allocation_id": summary["allocation_id"],
                        "command_sha256": summary["command_sha256"],
                        "state": "intent_recorded_before_create_job",
                        "sdk_job_ids_before": sdk_job_ids_before,
                        "remote_sdk_containment_sha256": containment[
                            "evidence_sha256"
                        ],
                    },
                ),
            )
            job = sdk.create_job(
                image=runtime["sqsh_path"],
                command=command,
                gpu_count=8,
                num_nodes=1,
                partition=runtime["partition"],
                account=runtime["account"],
                env_vars={"NVIDIA_TF32_OVERRIDE": "0"},
            )
            submission = verified_submission_record(
                sdk=sdk,
                job_id=job.id,
                command=command,
                summary=summary,
                containment=containment,
            )
            tao_job_id = submission["tao_job_id"]
            slurm_job_id = submission["slurm_job_id"]
            if (
                tao_job_id in tao_job_ids
                or slurm_job_id in slurm_job_ids
            ):
                raise ContractError("TAO and SLURM job IDs must be unique")
            tao_job_ids.add(tao_job_id)
            slurm_job_ids.add(slurm_job_id)
            submissions.append(submission)
            atomic_json(
                ledger_path,
                submission_ledger_payload(
                    manifest,
                    manifest_file_sha256,
                    submissions,
                    status="submitting_incomplete",
                    source_checks=ledger_source_checks,
                ),
            )
    finally:
        sdk._monitor.stop()
        sdk._store.close()
    if len(submissions) != 6:
        raise ContractError("all six jobs must be submitted into one ledger")
    atomic_json(
        ledger_path,
        submission_ledger_payload(
            manifest,
            manifest_file_sha256,
            submissions,
            status="complete",
            source_checks=ledger_source_checks,
        ),
    )
    return submissions


def main() -> int:
    args = parse_args()
    invalidation_arguments = (
        args.invalidation_evidence,
        args.invalidation_evidence_sha256,
        args.invalidation_evidence_internal_sha256,
    )
    if any(item is not None for item in invalidation_arguments) and not all(
        item is not None for item in invalidation_arguments
    ):
        raise ContractError(
            "all three invalidation-evidence arguments are required together"
        )
    if (
        any(item is not None for item in invalidation_arguments)
        and args.replace_incomplete_allocation is None
    ):
        raise ContractError(
            "invalidation evidence is valid only for allocation replacement"
        )
    invalidation_reference = (
        {
            "path": str(args.invalidation_evidence.resolve()),
            "whole_file_sha256": manifest_generator.require_sha256(
                args.invalidation_evidence_sha256,
                "CLI invalidation-evidence whole-file SHA256",
            ),
            "internal_sha256": manifest_generator.require_sha256(
                args.invalidation_evidence_internal_sha256,
                "CLI invalidation-evidence internal SHA256",
            ),
        }
        if args.invalidation_evidence is not None
        else None
    )
    manifest_path = args.manifest.resolve()
    manifest, manifest_file_sha256 = load_manifest(
        manifest_path,
        args.manifest_file_sha256,
    )
    runtime_dir = args.runtime_dir.resolve()
    require_equal(
        runtime_dir,
        Path(manifest["runtime"]["local_runtime_path"]).resolve(),
        "CLI runtime directory",
    )
    report_path = args.report.resolve()
    require_equal(
        report_path,
        runtime_dir / "dry_run.json",
        "CLI dry-run report path",
    )
    source_checks = validate_final_source_evidence(manifest)
    configs = generate_configs(manifest)
    rendered = []
    for allocation in manifest["schedule"]["allocations"]:
        plan = build_block_plan(
            manifest,
            manifest_file_sha256,
            allocation,
            configs,
        )
        rendered.append(
            staged_command(manifest, allocation, plan, configs)
        )
    loaded_keys = (
        load_env_file(Path(manifest["runtime"]["secrets_env_path"]))
        if (
            args.verify_remote
            or args.launch
            or args.resume_incomplete_submission
            or args.replace_incomplete_allocation is not None
        )
        else []
    )
    remote = verify_remote(manifest) if args.verify_remote else None
    blockers = []
    if remote is None:
        blockers.append("remote artifact verification not requested")
    elif not remote["all_verified"]:
        blockers.append("remote artifact verification failed")
    report = {
        "schema_version": 1,
        "status": "dry_run_validated_not_launched",
        "manifest": {
            "path": str(manifest_path),
            "whole_file_sha256": manifest_file_sha256,
            "internal_sha256": manifest["manifest_sha256"],
        },
        "candidate_ids": manifest["candidate_derivation"]["candidate_ids"],
        "candidate_count": len(manifest["candidates"]),
        "schedule_sha256": manifest["schedule"]["schedule_sha256"],
        "allocations": [summary for _, summary in rendered],
        "source_checks": source_checks,
        "remote_checks": remote,
        "loaded_secret_keys": loaded_keys,
        "secret_values_recorded": False,
        "submission_ready": not blockers,
        "blockers": blockers,
        "feeds_final_selection": False,
        "feeds_reselection": False,
        "selection_time_objectives_replaced": False,
        "requested_operation": (
            "replace_incomplete_allocation"
            if args.replace_incomplete_allocation is not None
            else "resume_incomplete_submission"
            if args.resume_incomplete_submission
            else "launch"
            if args.launch
            else "dry_run"
        ),
        "replacement_allocation_id": args.replace_incomplete_allocation,
    }
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if (
        not args.launch
        and not args.resume_incomplete_submission
        and args.replace_incomplete_allocation is None
    ):
        return 0
    if args.acknowledgement != EXPECTED_ACKNOWLEDGEMENT:
        raise ContractError("launch acknowledgement does not match")
    if not args.verify_remote or blockers:
        raise ContractError("launch requires successful remote verification")
    launch_checks = validate_launch_source_state(manifest)
    launch_source_checks = {
        **source_checks,
        "launch_source_state": launch_checks,
    }
    if args.resume_incomplete_submission:
        if args.submission_ledger_sha256 is None:
            raise ContractError(
                "resume requires --submission-ledger-sha256"
            )
        submissions = resume_incomplete_submission(
            manifest=manifest,
            manifest_file_sha256=manifest_file_sha256,
            commands=rendered,
            runtime_dir=runtime_dir,
            source_checks=launch_source_checks,
            supplied_ledger_sha256=args.submission_ledger_sha256,
        )
        print(
            json.dumps(
                {
                    "status": (
                        "incomplete_submission_reconciled_and_completed"
                    ),
                    "submission_count": len(submissions),
                    "submissions": submissions,
                    "duplicate_submission_permitted": False,
                    "feeds_final_selection": False,
                    "feeds_reselection": False,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    if args.replace_incomplete_allocation is not None:
        if args.submission_ledger_sha256 is None:
            raise ContractError(
                "replacement requires --submission-ledger-sha256"
            )
        replacement = replacement_submission(
            manifest=manifest,
            manifest_file_sha256=manifest_file_sha256,
            commands=rendered,
            runtime_dir=runtime_dir,
            source_checks=launch_source_checks,
            allocation_id=args.replace_incomplete_allocation,
            supplied_ledger_sha256=args.submission_ledger_sha256,
            invalidation_evidence=invalidation_reference,
        )
        print(
            json.dumps(
                {
                    "status": (
                        "complete_front_replacement_submitted_not_waited"
                    ),
                    "allocation_id": args.replace_incomplete_allocation,
                    "replacement_submission": replacement,
                    "partial_measurements_reused": False,
                    "feeds_final_selection": False,
                    "feeds_reselection": False,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    if args.submission_ledger_sha256 is not None:
        raise ContractError(
            "--submission-ledger-sha256 requires replacement or resume"
        )
    submissions = submit_all(
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha256,
        commands=rendered,
        runtime_dir=runtime_dir,
        source_checks=launch_source_checks,
    )
    print(
        json.dumps(
            {
                "status": "six_jobs_submitted_concurrently_not_waited",
                "submission_count": len(submissions),
                "submissions": submissions,
                "feeds_final_selection": False,
                "feeds_reselection": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
