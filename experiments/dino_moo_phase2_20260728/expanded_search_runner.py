#!/usr/bin/env python3

"""Run the preregistered expanded DINO search from an immutable manifest.

The launcher is deliberately manifest-driven:

* the sensitivity result determines which architecture axes exist;
* three independent, sequential Bayesian subarchives generate 20 candidates
  each;
* every candidate uses the same eight-GPU training, accuracy-evaluation, and
  stabilized selection-time latency protocols;
* the complete successful union is passed unchanged to the production
  ``tao_automl`` archive selector;
* no candidate ID or observed result can change the search space, selector
  configuration, or winner.

The default action is a read-only dry run.  Launching requires the exact
whole-file manifest SHA, remote artifact verification, and an explicit
acknowledgement.  This script does not perform the later matched-allocation
remeasurement of the final Pareto front.
"""

from __future__ import annotations

import argparse
import base64
import copy
import csv
from dataclasses import asdict
import fcntl
import hashlib
import inspect
import json
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping

import yaml

from tao_automl.latency_stats import (
    LatencyProtocol,
    LatencyValidityThresholds,
    aggregate_synchronized_latency,
)
from tao_automl.objectives import parse_objective_config
from tao_automl.selection import analyze_archive


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "expanded_search_manifest.v1.json"
DEFAULT_RUNTIME = HERE / "runtime" / "expanded_search"
DEFAULT_REPORT = DEFAULT_RUNTIME / "dry_run.json"
SKILL_DIR = Path(
    "/localhome/local-rarunachalam/tao-skills-external/"
    "skills/models/tao-train-dino"
)
SDK_ROOT = Path("/localhome/local-rarunachalam/tao-sdk")
EXPECTED_SCOPE = {
    "model_family": "DINO ResNet50",
    "dataset_uri": (
        "s3://nvcf-storage-handling/data/"
        "tao_od_synthetic_full_dino_coco/"
    ),
    "other_models_permitted": False,
    "other_datasets_permitted": False,
}
EXPECTED_SEARCH_SEEDS = (314159, 271828, 161803)
EXPECTED_RECOMMENDATIONS_PER_SEED = 20
EXPECTED_TOTAL_CANDIDATES = 60
EXPECTED_TRAINING_SEED = 1234
EXPECTED_ACKNOWLEDGEMENT = (
    "USER_AUTHORIZED_3X8GPU_SLURM_DINO_EXPANDED_SEARCH_20260728"
)
TERMINAL_JOB_STATUSES = frozenset({"Complete", "Error", "Canceled"})
SUCCESS_REC_STATUSES = frozenset({"success", "done"})
TERMINAL_CANDIDATE_STATUSES = frozenset(
    {"success", "training_or_measurement_failure"}
)
HEX = frozenset("0123456789abcdef")
EXPECTED_ANALYSIS_ERRATUM_CONTRACT_SHA256 = (
    "609bc9863a7e3289fe5f374b935f9da8422860eb00c62ea3d4bab00846d2fd7f"
)
EXPECTED_POST_FRONT_CONTRACT_SHA256 = (
    "aba3a961bf50caf15803f271b59d7ffbd091414816d14f3deb793452f75ec281"
)
EXPECTED_SENSITIVITY_RESULT_SHA256 = (
    "33aea1c13ece0ce632587abd16ed6020ecc88c63220f89891a5f30183322eaea"
)
EXPECTED_SENSITIVITY_REPORT_SHA256 = (
    "40a8bccb6e43b8238c2cf6b47eaf3253e735d82fd160212d12915b3137a3fa79"
)
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


class ContractError(ValueError):
    """Raised when a preregistered experiment contract is violated."""


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_bytes(value: Any) -> bytes:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ContractError(f"value is not canonical JSON: {error}") from error
    return payload.encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_schema_value(value: Any) -> str:
    """Hash generated TAO schema values, preserving their Infinity bounds."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX for character in value)
    ):
        raise ContractError(f"{label} must be a lowercase SHA256 digest")
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
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


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ContractError(f"non-finite JSON constant: {token}")
            ),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return value


def load_schema_json(path: Path) -> dict[str, Any]:
    """Load the pinned TAO schema, whose generated bounds may use Infinity."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read schema JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain a schema object")
    return value


def finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ContractError(f"{label} must be a finite number")
    return float(value)


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ContractError(
            f"{label} mismatch: expected {expected!r}, got {actual!r}"
        )


def git_value(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout.strip()


def git_is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode not in (0, 1):
        raise ContractError(
            f"cannot verify git ancestry in {repo}: {result.stderr.strip()}"
        )
    return result.returncode == 0


def nested_value(value: Mapping[str, Any], dotted_path: str) -> Any:
    cursor: Any = value
    for component in dotted_path.split("."):
        if not isinstance(cursor, Mapping) or component not in cursor:
            raise ContractError(f"missing field {dotted_path}")
        cursor = cursor[component]
    return cursor


def set_dotted(value: dict[str, Any], dotted_path: str, replacement: Any) -> None:
    cursor: Any = value
    components = dotted_path.split(".")
    for component in components[:-1]:
        if not isinstance(cursor, dict) or component not in cursor:
            raise ContractError(f"cannot set missing field {dotted_path}")
        cursor = cursor[component]
    if not isinstance(cursor, dict) or components[-1] not in cursor:
        raise ContractError(f"cannot set missing field {dotted_path}")
    cursor[components[-1]] = copy.deepcopy(replacement)


def schema_node(schema: dict[str, Any], dotted_path: str) -> dict[str, Any]:
    cursor: Any = schema
    for component in dotted_path.split("."):
        properties = cursor.get("properties") if isinstance(cursor, dict) else None
        if not isinstance(properties, dict) or component not in properties:
            raise ContractError(f"schema does not contain {dotted_path}")
        cursor = properties[component]
    if not isinstance(cursor, dict):
        raise ContractError(f"schema node {dotted_path} is not an object")
    return cursor


def load_manifest(
    path: Path,
    *,
    supplied_file_sha256: str,
) -> tuple[dict[str, Any], str]:
    supplied = require_sha256(
        supplied_file_sha256,
        "supplied expanded-search manifest whole-file SHA256",
    )
    actual_file_sha = sha256_file(path)
    require_equal(actual_file_sha, supplied, "expanded manifest whole-file SHA256")
    manifest = load_json(path)
    claimed = require_sha256(
        manifest.get("manifest_sha256"),
        "expanded manifest internal manifest_sha256",
    )
    unhashed = copy.deepcopy(manifest)
    del unhashed["manifest_sha256"]
    require_equal(
        sha256_value(unhashed),
        claimed,
        "expanded manifest canonical internal SHA256",
    )
    validate_manifest_contract(manifest)
    return manifest, actual_file_sha


def validate_manifest_contract(manifest: dict[str, Any]) -> None:
    require_equal(manifest.get("schema_version"), 1, "manifest schema_version")
    require_equal(
        manifest.get("manifest_id"),
        "dino_expanded_search_20260728_v1",
        "manifest_id",
    )
    require_equal(
        manifest.get("status"),
        "preregistered_ready_to_launch",
        "manifest status",
    )
    require_equal(manifest.get("scope"), EXPECTED_SCOPE, "DINO-only scope")
    require_equal(
        manifest.get("feeds_final_selection"),
        True,
        "feeds_final_selection",
    )
    require_equal(
        manifest.get("manual_override_permitted"),
        False,
        "manual_override_permitted",
    )
    require_equal(
        manifest.get("algorithm_only_selection_required"),
        True,
        "algorithm_only_selection_required",
    )

    derivation = manifest["derivation"]
    require_equal(
        derivation.get("accuracy_retention_used_for_axis_derivation"),
        False,
        "accuracy-retention axis derivation flag",
    )
    require_equal(
        derivation.get("qualified_value_hull_used"),
        False,
        "qualified-value hull flag",
    )
    require_equal(
        derivation.get("manual_override_used"),
        False,
        "derivation manual override flag",
    )
    require_equal(
        derivation.get("sensitivity_result_sha256"),
        EXPECTED_SENSITIVITY_RESULT_SHA256,
        "approved sensitivity result whole-file SHA256",
    )
    require_equal(
        derivation.get("sensitivity_report_sha256"),
        EXPECTED_SENSITIVITY_REPORT_SHA256,
        "approved sensitivity result report SHA256",
    )
    runner_path = derivation.get("runner_path")
    if (
        not isinstance(runner_path, str)
        or not Path(runner_path).is_absolute()
        or Path(runner_path).name != "expanded_search_runner.py"
    ):
        raise ContractError("expanded runner path must be absolute and canonical")
    require_sha256(
        derivation.get("runner_sha256"),
        "expanded runner source SHA256",
    )
    require_equal(
        derivation.get("analysis_erratum_contract_sha256"),
        EXPECTED_ANALYSIS_ERRATUM_CONTRACT_SHA256,
        "analysis erratum contract SHA256",
    )
    require_equal(
        derivation.get("post_front_contract_sha256"),
        EXPECTED_POST_FRONT_CONTRACT_SHA256,
        "post-front contract SHA256",
    )
    analysis_erratum = derivation.get("source_identity", {}).get(
        "analysis_erratum"
    )
    if not isinstance(analysis_erratum, dict):
        raise ContractError("manifest lacks approved analysis erratum provenance")
    require_equal(
        analysis_erratum.get("contract_sha256"),
        EXPECTED_ANALYSIS_ERRATUM_CONTRACT_SHA256,
        "approved analysis erratum provenance",
    )
    require_equal(
        {
            key: analysis_erratum.get(key)
            for key in (
                "sha256",
                "erratum_id",
                "corrected_aggregator_sha256",
                "submission_ledger_sha256",
            )
        },
        {
            "sha256": (
                "8e19287bf2ffd674f62b21cdaf11e000"
                "b0eae1ed8af9d0ada1238491588993f2"
            ),
            "erratum_id": (
                "dino_sensitivity_latency_analysis_erratum_20260728_v1"
            ),
            "corrected_aggregator_sha256": (
                "9209e748093e0555fe5cba339327a821"
                "6744ec9ca6b9dae276c7041703a409c6"
            ),
            "submission_ledger_sha256": (
                "b1c170c0d4697463d171cbeca3e4adcbd"
                "34cc1cb7429c236f48b58c46c3b6d54"
            ),
        },
        "approved analysis erratum source identity",
    )

    design = manifest["search_design"]
    require_equal(design.get("algorithm"), "bayesian", "search algorithm")
    require_equal(
        tuple(design.get("search_seeds", [])),
        EXPECTED_SEARCH_SEEDS,
        "search seeds",
    )
    require_equal(
        design.get("training_seed"),
        EXPECTED_TRAINING_SEED,
        "training seed",
    )
    require_equal(
        design.get("recommendations_per_seed"),
        EXPECTED_RECOMMENDATIONS_PER_SEED,
        "recommendations per seed",
    )
    require_equal(
        design.get("total_candidate_budget"),
        EXPECTED_TOTAL_CANDIDATES,
        "total candidate budget",
    )
    require_equal(
        design.get("candidate_generation_mode_independence"),
        True,
        "candidate-generation mode independence",
    )
    require_equal(
        design.get("manual_candidate_injection_permitted"),
        False,
        "manual candidate injection",
    )
    require_equal(
        design.get("result_ordered_search_changes_permitted"),
        False,
        "result-ordered search changes",
    )

    search = manifest["search_space"]
    axes = search.get("architecture_axes")
    if not isinstance(axes, list) or not axes:
        raise ContractError("expanded search must contain at least one derived axis")
    search_parameters = search.get("search_parameters")
    if not isinstance(search_parameters, list) or len(search_parameters) != len(
        set(search_parameters)
    ):
        raise ContractError("search parameters must be a unique ordered list")
    expected_parameters = [axis["path"] for axis in axes] + [
        item["path"] for item in search["always_included_training_parameters"]
    ]
    require_equal(
        search_parameters,
        expected_parameters,
        "search parameter order",
    )
    require_equal(
        set(search["search_domains"]),
        set(search_parameters),
        "search domain keys",
    )
    require_equal(
        search.get("compatibility_constraints"),
        ["model.num_queries >= model.num_select"],
        "compatibility constraint",
    )
    reference_model = search.get("reference_model_spec")
    if not isinstance(reference_model, dict) or len(reference_model) < 20:
        raise ContractError("full DINO reference model mapping is required")
    for axis in axes:
        path = axis["path"]
        if not path.startswith("model."):
            raise ContractError(f"architecture axis escaped model scope: {path}")
        nested_value({"model": reference_model}, path)
        if not axis.get("qualified_non_reference_levels"):
            raise ContractError(f"derived axis lacks qualification evidence: {path}")
        require_equal(
            axis.get("qualification_basis"),
            "at_least_one_direction_agnostic_latency_effect_qualified",
            f"{path} qualification basis",
        )
    for path, domain in search["search_domains"].items():
        representation = domain.get("representation")
        if representation not in {
            "continuous",
            "integer_range",
            "ordered_integer_levels",
        }:
            raise ContractError(
                f"{path} uses unsupported search representation {representation!r}"
            )
        if representation == "ordered_integer_levels":
            options = domain.get("valid_options")
            if (
                not isinstance(options, list)
                or not options
                or len(options) != len(set(options))
                or any(isinstance(item, bool) or not isinstance(item, int) for item in options)
            ):
                raise ContractError(
                    f"{path} ordered integer options must be unique non-empty ints"
                )
        else:
            lower = finite_number(domain.get("valid_min"), f"{path}.valid_min")
            upper = finite_number(domain.get("valid_max"), f"{path}.valid_max")
            if lower > upper:
                raise ContractError(f"{path} has an inverted search interval")

    selection = manifest["selection"]
    require_equal(selection.get("accuracy_metric"), "mAP50", "accuracy metric")
    require_equal(selection.get("latency_metric"), "latency_ms", "latency metric")
    require_equal(
        selection["latency_mode"]["latency_accuracy_retention"],
        {
            "type": "relative",
            "retained_fraction": 0.98,
            "reference": "accuracy_winner",
        },
        "latency accuracy-retention policy",
    )
    require_equal(
        selection["multi_objective_mode"].get("multi_objective_min_accuracy"),
        None,
        "multi-objective minimum accuracy",
    )
    require_equal(
        selection["multi_objective_mode"].get("selector"),
        "normalized_augmented_chebyshev",
        "multi-objective selector",
    )
    require_equal(
        selection["multi_objective_mode"].get("weights"),
        {"accuracy_regret": 1.0, "latency_regret": 1.0},
        "multi-objective weights",
    )
    finite_number(
        selection["latency_tolerance"].get("value_ms"),
        "selection latency tolerance",
    )
    require_equal(
        selection.get("dominated_multi_objective_winner_permitted"),
        False,
        "dominated-winner permission",
    )
    require_equal(
        selection.get("manual_winner_override_permitted"),
        False,
        "manual winner override",
    )
    post_front = manifest.get("post_front_matched_validation")
    if not isinstance(post_front, dict):
        raise ContractError("post-front matched validation is not preregistered")
    require_equal(
        sha256_value(post_front),
        EXPECTED_POST_FRONT_CONTRACT_SHA256,
        "post-front matched-validation contract",
    )
    require_equal(
        post_front["allocation_design"]["allocation_count"],
        6,
        "post-front allocation count",
    )
    require_equal(
        post_front["allocation_design"]["gpus_per_node"],
        8,
        "post-front GPU count",
    )
    require_equal(
        post_front["latency_protocol"]["warmup_iterations"],
        50,
        "post-front warmups",
    )
    require_equal(
        post_front["latency_protocol"]["timed_iterations_per_round"],
        100,
        "post-front timed iterations",
    )
    require_equal(
        post_front["latency_protocol"]["repeated_rounds"],
        5,
        "post-front rounds",
    )
    require_equal(
        post_front["selection_isolation"]["measurements_feed_reselection"],
        False,
        "post-front reselection feed",
    )

    frozen = manifest["frozen_identity"]
    runtime = frozen["runtime"]
    require_equal(runtime.get("num_nodes"), 1, "SLURM node count")
    require_equal(runtime.get("gpu_count_per_node"), 8, "GPUs per node")
    require_equal(runtime.get("precision"), "fp32", "runtime precision")
    require_equal(runtime.get("distributed_strategy"), "ddp", "strategy")
    require_equal(runtime.get("tf32"), False, "TF32")
    training = frozen["training_controls"]
    require_equal(training.get("train_epochs"), 10, "training epochs")
    require_equal(training.get("num_gpus"), 8, "training GPUs")
    require_equal(training.get("num_nodes"), 1, "training nodes")
    require_equal(training.get("global_batch_size"), 32, "global batch size")
    require_equal(training.get("activation_checkpoint"), False, "activation checkpoint")
    require_equal(training.get("cudnn_benchmark"), False, "cuDNN benchmark")
    require_equal(training.get("cudnn_deterministic"), True, "cuDNN deterministic")
    dataset = frozen["dataset"]
    require_equal(dataset.get("source_uri"), EXPECTED_SCOPE["dataset_uri"], "dataset URI")
    require_equal(dataset.get("num_classes"), 5, "dataset class count")
    require_equal(dataset.get("eval_class_ids"), [1, 2, 3, 4], "eval classes")


def git_path_provenance(repo: Path, source: Path) -> dict[str, Any]:
    try:
        relative = source.resolve().relative_to(repo.resolve())
    except ValueError as error:
        raise ContractError(
            f"runner source {source} escaped repository {repo}"
        ) from error
    relative_text = relative.as_posix()

    def git_status(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )

    tracked = (
        git_status("ls-files", "--error-unmatch", "--", relative_text).returncode
        == 0
    )
    committed = (
        tracked
        and git_status("cat-file", "-e", f"HEAD:{relative_text}").returncode
        == 0
    )
    clean_against_head = (
        tracked
        and git_status("diff", "--quiet", "HEAD", "--", relative_text).returncode
        == 0
    )
    current_blob = git_value(repo, "hash-object", str(source))
    head_blob = (
        git_value(repo, "rev-parse", f"HEAD:{relative_text}")
        if committed
        else None
    )
    head_matches = committed and current_blob == head_blob
    launch_source_ready = bool(
        tracked and committed and clean_against_head and head_matches
    )
    reasons = []
    if not tracked:
        reasons.append("runner_source_untracked")
    if not committed:
        reasons.append("runner_source_not_in_HEAD")
    if tracked and not clean_against_head:
        reasons.append("runner_source_dirty_against_HEAD")
    if committed and not head_matches:
        reasons.append("runner_source_blob_mismatch")
    return {
        "repository": str(repo.resolve()),
        "relative_path": relative_text,
        "tracked": tracked,
        "committed": committed,
        "clean_against_head": clean_against_head,
        "current_git_blob": current_blob,
        "head_git_blob": head_blob,
        "head_matches_current": head_matches,
        "launch_source_ready": launch_source_ready,
        "blockers": reasons,
    }


def validate_runner_source_provenance(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    derivation = manifest["derivation"]
    expected_path = Path(derivation["runner_path"]).resolve()
    actual_path = Path(__file__).resolve()
    require_equal(actual_path, expected_path, "expanded runner source path")
    observed_sha256 = sha256_file(actual_path)
    require_equal(
        observed_sha256,
        derivation["runner_sha256"],
        "expanded runner source SHA256",
    )
    repository = Path(
        manifest["frozen_identity"]["source_repositories"]["tao_automl"][
            "path"
        ]
    )
    git_provenance = git_path_provenance(repository, actual_path)
    return {
        "path": str(actual_path),
        "expected_sha256": derivation["runner_sha256"],
        "observed_sha256": observed_sha256,
        **git_provenance,
    }


def require_launch_source_ready(runner_source: Mapping[str, Any]) -> None:
    if runner_source.get("launch_source_ready") is not True:
        raise ContractError(
            "launch requires the manifest-pinned expanded runner source to "
            "be tracked, committed, and clean: "
            f"{runner_source.get('blockers', [])}"
        )


def validate_local_provenance(
    manifest: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    source_repositories = manifest["frozen_identity"]["source_repositories"]
    for key, expected in source_repositories.items():
        repo = Path(expected["path"])
        head = git_value(repo, "rev-parse", "HEAD")
        branch = git_value(repo, "branch", "--show-current")
        require_equal(branch, expected["branch"], f"{key} branch")
        if expected.get("commit_policy") == "required_ancestor":
            base = expected.get("selection_core_commit", expected.get("commit"))
            if not base or not git_is_ancestor(repo, base, head):
                raise ContractError(f"{key} required commit is not an ancestor of HEAD")
        else:
            require_equal(head, expected["commit"], f"{key} commit")
        checks[key] = {"path": str(repo), "branch": branch, "commit": head}
    checks["runner_source"] = validate_runner_source_provenance(manifest)

    derivation = manifest["derivation"]
    sensitivity_path = Path(derivation["sensitivity_result_path"]).resolve()
    require_equal(
        sha256_file(sensitivity_path),
        derivation["sensitivity_result_sha256"],
        "sensitivity result provenance",
    )
    sensitivity_manifest_path = Path(
        derivation["source_identity"]["sensitivity_manifest_path"]
    ).resolve()
    require_equal(
        sha256_file(sensitivity_manifest_path),
        derivation["source_identity"]["sensitivity_manifest_sha256"],
        "sensitivity manifest provenance",
    )
    sensitivity_manifest = load_json(sensitivity_manifest_path)
    benchmark_path = (
        sensitivity_manifest_path.parent
        / sensitivity_manifest["frozen_inputs"]["benchmark_path"]
    ).resolve()
    require_equal(
        sha256_file(benchmark_path),
        sensitivity_manifest["frozen_inputs"]["benchmark_sha256"],
        "latency benchmark source",
    )
    evaluate_template_path = Path(
        sensitivity_manifest["frozen_inputs"]["evaluate_template_path"]
    ).resolve()
    require_equal(
        sha256_file(evaluate_template_path),
        sensitivity_manifest["frozen_inputs"]["evaluate_template_sha256"],
        "DINO evaluate template",
    )
    skill_info_path = SKILL_DIR / "references" / "skill_info.yaml"
    train_template_path = SKILL_DIR / "references" / "spec_template_train.yaml"
    train_schema_path = SKILL_DIR / "schemas" / "train.schema.json"
    for path in (
        skill_info_path,
        train_template_path,
        train_schema_path,
        evaluate_template_path,
        benchmark_path,
    ):
        if not path.is_file():
            raise ContractError(f"required local artifact is missing: {path}")

    checks["manifest_path"] = str(manifest_path.resolve())
    checks["sensitivity_result"] = {
        "path": str(sensitivity_path),
        "sha256": sha256_file(sensitivity_path),
    }
    checks["sensitivity_manifest"] = {
        "path": str(sensitivity_manifest_path),
        "sha256": sha256_file(sensitivity_manifest_path),
    }
    checks["benchmark"] = {
        "path": str(benchmark_path),
        "sha256": sha256_file(benchmark_path),
    }
    checks["evaluate_template"] = {
        "path": str(evaluate_template_path),
        "sha256": sha256_file(evaluate_template_path),
    }
    checks["skill_info"] = {
        "path": str(skill_info_path),
        "sha256": sha256_file(skill_info_path),
    }
    checks["train_template"] = {
        "path": str(train_template_path),
        "sha256": sha256_file(train_template_path),
    }
    checks["train_schema"] = {
        "path": str(train_schema_path),
        "sha256": sha256_file(train_schema_path),
    }
    return checks


def build_search_contract(
    manifest: dict[str, Any],
    base_schema: dict[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, dict[str, Any]]]:
    """Encode the manifest domains in the packaged schema/search API.

    Numeric finite option sets use JSON Schema ``enum``.  That is the existing
    canonical path by which the external-schema loader emits ``ordered_int``;
    it avoids widening a preregistered discrete domain into an integer range.
    """

    derived_schema = copy.deepcopy(base_schema)
    parameters = list(manifest["search_space"]["search_parameters"])
    domains = manifest["search_space"]["search_domains"]
    custom_ranges: dict[str, dict[str, Any]] = {}
    for path in parameters:
        domain = domains[path]
        node = schema_node(derived_schema, path)
        representation = domain["representation"]
        if representation == "ordered_integer_levels":
            if node.get("type") not in {"int", "integer"}:
                raise ContractError(
                    f"{path} ordered integer domain targets {node.get('type')!r}"
                )
            options = list(domain["valid_options"])
            if not options:
                raise ContractError(f"{path} contains no finite options")
            try:
                unique_options = set(options)
            except TypeError as error:
                raise ContractError(
                    f"{path} contains non-scalar options"
                ) from error
            if len(options) != len(unique_options):
                raise ContractError(f"{path} contains duplicate options")
            if any(isinstance(item, bool) or not isinstance(item, int) for item in options):
                raise ContractError(f"{path} contains non-integer options")
            minimum = node.get("minimum")
            maximum = node.get("maximum")
            if minimum is not None and any(item < minimum for item in options):
                raise ContractError(f"{path} option is below schema minimum")
            if maximum not in (None, "Infinity") and not (
                isinstance(maximum, float) and math.isinf(maximum)
            ):
                if any(item > maximum for item in options):
                    raise ContractError(f"{path} option is above schema maximum")
            node["enum"] = options
            custom_ranges[path] = {"valid_options": options}
        elif representation == "integer_range":
            if node.get("type") not in {"int", "integer"}:
                raise ContractError(f"{path} integer range targets non-integer schema")
            custom_ranges[path] = {
                "valid_min": int(domain["valid_min"]),
                "valid_max": int(domain["valid_max"]),
            }
        elif representation == "continuous":
            if node.get("type") not in {"float", "number"}:
                raise ContractError(f"{path} continuous range targets non-float schema")
            custom_ranges[path] = {
                "valid_min": float(domain["valid_min"]),
                "valid_max": float(domain["valid_max"]),
            }
        else:  # validated earlier, retained as a fail-closed guard
            raise ContractError(f"unsupported search representation: {representation}")
    return derived_schema, parameters, custom_ranges


def apply_candidate_to_reference_model(
    manifest: dict[str, Any],
    candidate_specs: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = manifest["search_space"]["search_parameters"]
    if set(candidate_specs) != set(parameters):
        missing = sorted(set(parameters) - set(candidate_specs))
        extra = sorted(set(candidate_specs) - set(parameters))
        raise ContractError(
            f"candidate search keys drifted: missing={missing}, extra={extra}"
        )
    domains = manifest["search_space"]["search_domains"]
    model = copy.deepcopy(manifest["search_space"]["reference_model_spec"])
    for path in parameters:
        raw = candidate_specs[path]
        domain = domains[path]
        representation = domain["representation"]
        if representation == "continuous":
            value = finite_number(raw, path)
            if not domain["valid_min"] <= value <= domain["valid_max"]:
                raise ContractError(f"{path}={value} escaped frozen range")
        elif representation == "integer_range":
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ContractError(f"{path} must be an integer")
            value = raw
            if not domain["valid_min"] <= value <= domain["valid_max"]:
                raise ContractError(f"{path}={value} escaped frozen range")
        elif representation == "ordered_integer_levels":
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ContractError(f"{path} must be an integer option")
            value = raw
            if value not in domain["valid_options"]:
                raise ContractError(
                    f"{path}={value} is not a preregistered finite option"
                )
        else:
            raise ContractError(f"unsupported domain representation for {path}")
        if path.startswith("model."):
            set_dotted({"model": model}, path, value)
    if int(model["num_queries"]) < int(model["num_select"]):
        raise ContractError(
            "candidate violates model.num_queries >= model.num_select"
        )
    return model


def training_spec(
    manifest: dict[str, Any],
    template: dict[str, Any],
    candidate_specs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    frozen = manifest["frozen_identity"]
    runtime = frozen["runtime"]
    data = frozen["dataset"]
    controls = frozen["training_controls"]
    spec = copy.deepcopy(template)
    spec["model"] = copy.deepcopy(manifest["search_space"]["reference_model_spec"])
    spec["dataset"]["train_data_sources"][0] = {
        "image_dir": data["train_image_dir"],
        "json_file": data["train_annotation"],
    }
    spec["dataset"]["val_data_sources"][0] = {
        "image_dir": data["validation_image_dir"],
        "json_file": data["validation_annotation"],
    }
    spec["dataset"]["num_classes"] = data["num_classes"]
    spec["dataset"]["eval_class_ids"] = copy.deepcopy(data["eval_class_ids"])
    spec["dataset"]["batch_size"] = controls["batch_size_per_gpu"]
    spec["train"]["pretrained_model_path"] = runtime["pretrained_model_path"]
    spec["train"]["num_gpus"] = controls["num_gpus"]
    spec["train"]["gpu_ids"] = list(range(controls["num_gpus"]))
    spec["train"]["num_nodes"] = controls["num_nodes"]
    spec["train"]["num_epochs"] = controls["train_epochs"]
    spec["train"]["checkpoint_interval"] = controls["checkpoint_interval_epochs"]
    spec["train"]["validation_interval"] = controls["validation_interval_epochs"]
    spec["train"]["seed"] = manifest["search_design"]["training_seed"]
    spec["train"]["precision"] = runtime["precision"]
    spec["train"]["distributed_strategy"] = runtime["distributed_strategy"]
    spec["train"]["activation_checkpoint"] = controls["activation_checkpoint"]
    spec["train"]["cudnn"]["benchmark"] = controls["cudnn_benchmark"]
    spec["train"]["cudnn"]["deterministic"] = controls["cudnn_deterministic"]
    reference_optimizer = manifest["search_space"]["reference_optimizer"]
    spec["train"]["optim"]["lr"] = reference_optimizer["lr"]
    spec["train"]["optim"]["weight_decay"] = reference_optimizer["weight_decay"]
    spec["wandb"]["enable"] = controls["wandb_enabled"]
    if candidate_specs is not None:
        resolved_model = apply_candidate_to_reference_model(manifest, candidate_specs)
        spec["model"] = resolved_model
        for path, value in candidate_specs.items():
            if not path.startswith("model."):
                set_dotted(spec, path, value)
    return spec


def spec_overrides_from_base(spec: dict[str, Any]) -> dict[str, Any]:
    """Return nested caller overrides; the runner deep-merges this mapping."""

    return {
        "model": copy.deepcopy(spec["model"]),
        "dataset": copy.deepcopy(spec["dataset"]),
        "train": copy.deepcopy(spec["train"]),
        "wandb": copy.deepcopy(spec["wandb"]),
    }


def evaluation_spec(
    manifest: dict[str, Any],
    template: dict[str, Any],
    resolved_model: dict[str, Any],
    checkpoint: str,
    *,
    latency: bool,
) -> dict[str, Any]:
    frozen = manifest["frozen_identity"]
    data = frozen["dataset"]
    accuracy = frozen["accuracy_evaluation"]
    latency_protocol = frozen["latency_protocol"]
    spec = copy.deepcopy(template)
    spec["model"] = copy.deepcopy(resolved_model)
    spec["wandb"]["enable"] = False
    spec["dataset"]["test_data_sources"] = {
        "image_dir": data["validation_image_dir"],
        "json_file": data["validation_annotation"],
    }
    spec["dataset"]["num_classes"] = data["num_classes"]
    spec["dataset"]["eval_class_ids"] = copy.deepcopy(data["eval_class_ids"])
    batch_size = (
        latency_protocol["batch_size_per_gpu"]
        if latency
        else accuracy["batch_size_per_gpu"]
    )
    spec["dataset"]["batch_size"] = batch_size
    spec["dataset"]["workers"] = 0 if latency else 8
    spec["dataset"]["augmentation"]["test_random_resize"] = accuracy[
        "test_random_resize"
    ]
    spec["dataset"]["augmentation"]["random_resize_max_size"] = accuracy[
        "random_resize_max_size"
    ]
    spec["dataset"]["augmentation"]["fixed_padding"] = accuracy["fixed_padding"]
    spec["evaluate"]["batch_size"] = batch_size
    spec["evaluate"]["num_gpus"] = frozen["runtime"]["gpu_count_per_node"]
    spec["evaluate"]["gpu_ids"] = list(
        range(frozen["runtime"]["gpu_count_per_node"])
    )
    spec["evaluate"]["num_nodes"] = frozen["runtime"]["num_nodes"]
    spec["evaluate"]["checkpoint"] = checkpoint
    if spec["model"] != resolved_model:
        raise ContractError("evaluation did not preserve the full model mapping")
    return spec


def selection_settings(manifest: dict[str, Any], seed: int) -> dict[str, Any]:
    design = manifest["search_design"]
    selection = manifest["selection"]
    latency_retention = selection["latency_mode"]["latency_accuracy_retention"]
    multi = selection["multi_objective_mode"]
    return {
        "algorithm": design["algorithm"],
        "automl_max_recommendations": design["recommendations_per_seed"],
        "automl_max_concurrent": 1,
        "session_id": f"dino_expanded_search_seed_{seed}",
        "experiment_id": f"dino_expanded_search_seed_{seed}",
        "random_seed": seed,
        "selection_mode": "multi_objective",
        "objectives": [
            {"metric": "mAP50", "direction": "maximize", "weight": 1.0},
            {"metric": "latency_ms", "direction": "minimize", "weight": 1.0},
        ],
        "accuracy_metric": "mAP50",
        "latency_metric": "latency_ms",
        "latency_accuracy_retention": copy.deepcopy(latency_retention),
        "multi_objective_min_accuracy": multi["multi_objective_min_accuracy"],
        "objective_normalization": "pareto_front",
        "augmentation_rho": multi["augmentation_rho"],
        "accuracy_tolerance": selection["accuracy_mode"]["accuracy_tolerance"],
        "latency_tolerance": selection["latency_tolerance"]["value_ms"],
        "selection_score_tolerance": multi["selection_score_tolerance"],
        "latency_ci_low_metric": "latency_ci95_low",
        "latency_ci_high_metric": "latency_ci95_high",
        "require_eval_fn_success": True,
        # Every recommendation receives a held-out mAP50 evaluation.  The old
        # permissive PTM baseline is intentionally not used as a constraint.
        "run_baseline": False,
        "run_final_evaluation": False,
        "automl_delete_intermediate_ckpt": True,
        "automl_checkpoint_retention_strategy": "terminal",
    }


def validate_selector_configuration(manifest: dict[str, Any]) -> dict[str, Any]:
    settings = selection_settings(manifest, EXPECTED_SEARCH_SEEDS[0])
    objective = parse_objective_config(settings)
    config = objective.selection_config
    if config is None:
        raise ContractError("production objective parser did not build a selector")
    require_equal(
        config.latency_accuracy_retention.kind,
        "relative",
        "parsed latency retention type",
    )
    require_equal(
        config.latency_accuracy_retention.value,
        0.98,
        "parsed latency retention value",
    )
    require_equal(
        config.multi_objective_min_accuracy,
        None,
        "parsed multi-objective accuracy floor",
    )
    require_equal(config.latency_tolerance, settings["latency_tolerance"], "latency tolerance")
    return {
        "settings": settings,
        "parsed_selection": config.to_dict(),
    }


def load_env_file(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"required secrets env file not found: {path}")
    loaded: list[str] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ContractError(f"unsupported env line {number}: missing '='")
        key, encoded = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ContractError(f"invalid env key on line {number}")
        tokens = shlex.split(encoded, comments=True, posix=True)
        if len(tokens) > 1:
            raise ContractError(f"unsupported env syntax on line {number}")
        os.environ.setdefault(key, tokens[0] if tokens else "")
        loaded.append(key)
    return sorted(loaded)


def ssh_target() -> str:
    user = os.environ.get("SLURM_USER", "").strip()
    host = os.environ.get("SLURM_HOSTNAME", "").split(",", 1)[0].strip()
    if not user or not host:
        raise ContractError("SLURM_USER and SLURM_HOSTNAME are required")
    return f"{user}@{host}"


def remote_output(command: str, *, timeout: int = 900) -> str:
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    key_path = os.environ.get("SSH_KEY_PATH")
    if key_path:
        ssh.extend(["-i", key_path])
    ssh.extend([ssh_target(), command])
    result = subprocess.run(
        ssh,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout


def verify_remote_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    frozen = manifest["frozen_identity"]
    runtime = frozen["runtime"]
    data = frozen["dataset"]
    files = [
        ("sqsh", runtime["sqsh_path"], runtime["sqsh_sha256"]),
        (
            "pretrained_model",
            runtime["pretrained_model_path"],
            runtime["pretrained_model_sha256"],
        ),
        ("train_annotation", data["train_annotation"], data["train_annotation_sha256"]),
        (
            "validation_annotation",
            data["validation_annotation"],
            data["validation_annotation_sha256"],
        ),
    ]
    checks: list[dict[str, Any]] = []
    for kind, path, expected in files:
        output = remote_output(
            f"test -f {shlex.quote(path)} && sha256sum {shlex.quote(path)} "
            "|| echo MISSING",
            timeout=1800,
        ).strip()
        observed = output.split(None, 1)[0] if output and output != "MISSING" else None
        verified = observed == expected
        checks.append(
            {
                "kind": kind,
                "path": path,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "verified": verified,
            }
        )
    for kind, path in (
        ("train_image_dir", data["train_image_dir"]),
        ("validation_image_dir", data["validation_image_dir"]),
    ):
        present = remote_output(
            f"test -d {shlex.quote(path)} && echo PRESENT || echo MISSING"
        ).strip() == "PRESENT"
        checks.append(
            {"kind": kind, "path": path, "verified": present}
        )
    if not all(item["verified"] for item in checks):
        raise ContractError("remote expanded-search artifact verification failed")
    return {"all_verified": True, "artifacts": checks}


def ensure_sdk_importable() -> None:
    path = str(SDK_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def configure_slurm(manifest: dict[str, Any]) -> None:
    runtime = manifest["frozen_identity"]["runtime"]
    os.environ["SLURM_USE_SQSH"] = "false"
    os.environ["SLURM_USE_REQUEUE"] = "false"
    os.environ["SLURM_PARTITION"] = runtime["partition"]
    os.environ["SLURM_ACCOUNT"] = runtime["account"]
    # These are part of the preregistered per-child allocation contract.
    # Never inherit a longer value from the invoking shell or secrets file.
    os.environ["SLURM_TIME_HOURS"] = "4"
    os.environ["SLURM_TIMEOUT_HOURS"] = "3.8"


def _runtime_identity_from_store(sdk: Any, job_id: str) -> dict[str, Any]:
    store = getattr(sdk, "_store", None)
    if store is None or not callable(getattr(store, "get_job", None)):
        raise ContractError("SLURM SDK durable job store is unavailable")
    entry = store.get_job(job_id)
    extractor = getattr(sdk, "_runtime_from_entry", None)
    if callable(extractor):
        runtime = extractor(entry)
    else:
        specs = entry.get("specs") if isinstance(entry, dict) else None
        if isinstance(specs, str):
            specs = json.loads(specs)
        runtime = (
            specs.get("_slurm_runtime")
            if isinstance(specs, dict)
            else None
        )
    if not isinstance(runtime, dict):
        raise ContractError(
            f"durable SLURM runtime identity is missing for TAO job {job_id}"
        )
    return runtime


def validate_recorded_slurm_runtime(
    evidence: Mapping[str, Any],
    *,
    label: str,
    require_active_slurm_id: bool = True,
) -> dict[str, Any]:
    """Validate and normalize scheduler-generation evidence."""

    tao_job_id = evidence.get("tao_job_id")
    if not isinstance(tao_job_id, str) or not tao_job_id:
        raise ContractError(f"{label} has invalid TAO job ID")
    slurm_job_id = str(evidence.get("slurm_job_id", "") or "")
    if (
        (require_active_slurm_id and not slurm_job_id)
        or (slurm_job_id and not slurm_job_id.isdigit())
    ):
        raise ContractError(f"{label} has invalid active SLURM job ID")
    failed_ids_raw = evidence.get("failed_slurm_job_ids", [])
    if not isinstance(failed_ids_raw, list):
        raise ContractError(f"{label} failed SLURM job IDs must be a list")
    failed_ids = [str(value) for value in failed_ids_raw]
    if (
        any(not value.isdigit() for value in failed_ids)
        or len(failed_ids) != len(set(failed_ids))
        or slurm_job_id in failed_ids
    ):
        raise ContractError(f"{label} has invalid failed SLURM job lineage")
    try:
        retry_count = int(evidence.get("retry_count", 0))
        revision = int(
            evidence.get("runtime_revision", evidence.get("revision", 0))
        )
    except (TypeError, ValueError) as error:
        raise ContractError(f"{label} has invalid runtime counters") from error
    if retry_count < 0 or revision < 0:
        raise ContractError(f"{label} has negative runtime counters")
    launch_uncertain = evidence.get("launch_uncertain", False)
    if not isinstance(launch_uncertain, bool):
        raise ContractError(f"{label} launch_uncertain must be boolean")
    if launch_uncertain:
        raise ContractError(
            f"{label} launch remains uncertain; refusing measurement evidence"
        )
    return {
        "tao_job_id": tao_job_id,
        "slurm_job_id": slurm_job_id,
        "retry_count": retry_count,
        "failed_slurm_job_ids": failed_ids,
        "launch_uncertain": False,
        "runtime_revision": revision,
    }


def slurm_runtime_evidence(
    sdk: Any,
    job_id: str,
    *,
    label: str,
    require_active_slurm_id: bool = True,
) -> dict[str, Any]:
    """Return one quiescent, durable scheduler-generation identity."""

    handler_runtime = sdk._handler.get_job_runtime_identity(job_id)
    durable_runtime = _runtime_identity_from_store(sdk, job_id)
    generation_keys = (
        "slurm_job_id",
        "retry_count",
        "failed_slurm_job_ids",
        "launch_uncertain",
        "revision",
    )
    for key in generation_keys:
        require_equal(
            handler_runtime.get(key),
            durable_runtime.get(key),
            f"{label} durable runtime {key}",
        )
    return validate_recorded_slurm_runtime(
        {**durable_runtime, "tao_job_id": job_id},
        label=label,
        require_active_slurm_id=require_active_slurm_id,
    )


def reconcile_slurm_runtime_evidence(
    persisted: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Reject regressed/conflicting resume identity while allowing retries."""

    validate_recorded_slurm_runtime(persisted, label=f"{label} persisted")
    validate_recorded_slurm_runtime(current, label=f"{label} current")
    for key in (
        "tao_job_id",
        "slurm_job_id",
        "retry_count",
        "failed_slurm_job_ids",
        "launch_uncertain",
        "runtime_revision",
    ):
        if key not in persisted:
            raise ContractError(f"{label} persisted runtime lacks {key}")
    require_equal(
        persisted["tao_job_id"],
        current["tao_job_id"],
        f"{label} TAO job identity",
    )
    persisted_revision = int(persisted["runtime_revision"])
    current_revision = int(current["runtime_revision"])
    persisted_retry = int(persisted["retry_count"])
    current_retry = int(current["retry_count"])
    if current_revision < persisted_revision or current_retry < persisted_retry:
        raise ContractError(f"{label} durable runtime generation regressed")
    if not set(map(str, persisted["failed_slurm_job_ids"])).issubset(
        set(map(str, current["failed_slurm_job_ids"]))
    ):
        raise ContractError(f"{label} failed scheduler lineage regressed")
    if current_revision == persisted_revision:
        for key in (
            "slurm_job_id",
            "retry_count",
            "failed_slurm_job_ids",
            "launch_uncertain",
        ):
            require_equal(
                persisted[key],
                current[key],
                f"{label} same-revision runtime {key}",
            )


def local_lustre_path(uri: str) -> str:
    if uri.startswith("lustre://"):
        value = uri.removeprefix("lustre://")
        return value if value.startswith("/") else f"/{value}"
    if uri.startswith("/"):
        return uri
    raise ContractError(f"expected Lustre result URI, got {uri!r}")


def validate_result_root(result_root: str, tao_job_id: str) -> None:
    root = Path(result_root)
    if root.name != tao_job_id or root.parent.name != "results":
        raise ContractError(
            f"result root is not scoped to TAO job {tao_job_id}: {result_root}"
        )


def discover_terminal_checkpoint(sdk: Any, train_job_id: str) -> dict[str, Any]:
    root = local_lustre_path(sdk.get_job_results_dir(train_job_id))
    validate_result_root(root, train_job_id)
    script = "\n".join(
        [
            "import hashlib,json,re,sys",
            "from pathlib import Path",
            "root=Path(sys.argv[1])",
            "pat=re.compile(r'^model_epoch_0*9_step_[0-9]+\\.(?:pth|ckpt)$',re.I)",
            "paths=sorted(p for p in root.rglob('*') if p.is_file() and pat.match(p.name))",
            "if len(paths)!=1: raise RuntimeError(f'expected one epoch-9 checkpoint, got {len(paths)}')",
            "p=paths[0]; h=hashlib.sha256()",
            "with p.open('rb') as stream:",
            " for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)",
            "print(json.dumps({'path':str(p),'sha256':h.hexdigest(),'size_bytes':p.stat().st_size,'epoch':9},sort_keys=True))",
        ]
    )
    output = remote_output(
        f"python3 -c {shlex.quote(script)} {shlex.quote(root)}",
        timeout=3600,
    )
    checkpoint = json.loads(output)
    require_sha256(checkpoint["sha256"], "checkpoint SHA256")
    if not checkpoint["path"].startswith(f"{root.rstrip('/')}/"):
        raise ContractError("checkpoint escaped the training job result root")
    return checkpoint


def append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"timestamp_utc": utc_timestamp(), **payload}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def wait_for_job(
    sdk: Any,
    job_id: str,
    *,
    event_path: Path,
    phase: str,
    seed: int,
    rec_id: int,
) -> str:
    previous = None
    while True:
        status = sdk.get_job_status(job_id).status
        if status != previous:
            append_event(
                event_path,
                {
                    "event": "job_status",
                    "phase": phase,
                    "seed": seed,
                    "rec_id": rec_id,
                    "tao_job_id": job_id,
                    "status": status,
                },
            )
            print(
                f"JOB_STATUS phase={phase} seed={seed} rec={rec_id} "
                f"job={job_id} status={status}",
                flush=True,
            )
            previous = status
        if status in TERMINAL_JOB_STATUSES:
            return status
        time.sleep(10)


def dino_metric_extractor(logs: str, metric_name: str) -> float | None:
    if metric_name != "mAP50":
        return None
    values: list[float] = []
    for pattern in MAP50_PATTERNS:
        values.extend(float(match) for match in pattern.findall(logs))
    return values[-1] if values else None


def read_status_map50(sdk: Any, eval_job_id: str) -> float | None:
    result_root = local_lustre_path(sdk.get_job_results_dir(eval_job_id))
    validate_result_root(result_root, eval_job_id)
    status_path = f"{result_root}/results_dir/evaluate/status.json"
    output = remote_output(
        f"test -f {shlex.quote(status_path)} && tail -100 {shlex.quote(status_path)}"
    )
    values: list[float] = []
    for line in output.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        kpi = record.get("kpi")
        if isinstance(kpi, dict):
            value = kpi.get("test_mAP50", kpi.get("val_mAP50"))
            if value is not None:
                values.append(finite_number(value, "evaluation mAP50"))
    return values[-1] if values else None


def benchmark_command(
    benchmark_path: Path,
    checkpoint: str,
    protocol: dict[str, Any],
    gpu_count: int,
) -> str:
    source = base64.b64encode(benchmark_path.read_bytes()).decode("ascii")
    install = (
        "import base64;"
        "open('/tmp/dino_latency_benchmark.py','wb').write("
        f"base64.b64decode('{source}'))"
    )
    return " ".join(
        [
            "python -c",
            shlex.quote(install),
            "&&",
            "torchrun",
            "--standalone",
            f"--nproc_per_node={gpu_count}",
            "/tmp/dino_latency_benchmark.py",
            "--config",
            "{config_path}",
            "--checkpoint",
            shlex.quote(checkpoint),
            "--output-root",
            '"$TAO_RESULTS_ROOT"',
            "--warmup-iterations",
            str(protocol["warmup_iterations"]),
            "--timed-iterations",
            str(protocol["timed_iterations"]),
            "--rounds",
            str(protocol["repeated_rounds"]),
            "--preloaded-batches",
            str(protocol["preloaded_batches"]),
            "--seed",
            str(protocol["benchmark_seed"]),
        ]
    )


def read_latency_rank_records(sdk: Any, job_id: str, gpu_count: int) -> list[dict[str, Any]]:
    result_root = local_lustre_path(sdk.get_job_results_dir(job_id))
    validate_result_root(result_root, job_id)
    reader = (
        "import glob,json,sys;"
        "paths=sorted(glob.glob(sys.argv[1]+'/rank_*.json'));"
        "print(json.dumps([json.load(open(path)) for path in paths]))"
    )
    output = remote_output(
        f"python3 -c {shlex.quote(reader)} "
        f"{shlex.quote(result_root + '/latency')}",
        timeout=600,
    )
    records = json.loads(output)
    if not isinstance(records, list) or len(records) != gpu_count:
        raise ContractError(
            f"expected {gpu_count} latency rank records, got "
            f"{len(records) if isinstance(records, list) else type(records).__name__}"
        )
    require_equal(
        {int(record["rank"]) for record in records},
        set(range(gpu_count)),
        "latency rank identities",
    )
    return records


def major_minor_patch(version: Any, label: str) -> str:
    match = re.match(r"^(\d+\.\d+\.\d+)", str(version))
    if match is None:
        raise ContractError(f"{label} has no major.minor.patch prefix: {version}")
    return match.group(1)


def validate_latency_rank_contract(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    checkpoint: dict[str, Any],
    resolved_model_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = manifest["frozen_identity"]
    expected_runtime = frozen["runtime"]
    protocol = frozen["latency_protocol"]
    gpu_count = expected_runtime["gpu_count_per_node"]
    signatures = set()
    input_identities = set()
    for record in records:
        require_equal(record.get("world_size"), gpu_count, "latency world size")
        require_equal(record.get("checkpoint"), checkpoint["path"], "latency checkpoint")
        hardware = record["hardware"]
        runtime = record["runtime"]
        require_equal(hardware["gpu_name"], expected_runtime["required_gpu_model"], "GPU model")
        require_equal(
            hardware["compute_capability"],
            expected_runtime["required_compute_capability"],
            "GPU compute capability",
        )
        require_equal(
            hardware["total_memory_bytes"],
            expected_runtime["required_gpu_memory_bytes"],
            "GPU memory",
        )
        require_equal(
            major_minor_patch(runtime["torch"], "torch runtime"),
            expected_runtime["torch"],
            "torch major.minor.patch",
        )
        require_equal(runtime["cuda"], expected_runtime["cuda"], "CUDA runtime")
        require_equal(runtime["cudnn"], expected_runtime["cudnn"], "cuDNN runtime")
        signatures.add(
            (
                hardware["gpu_name"],
                hardware["compute_capability"],
                hardware["total_memory_bytes"],
                runtime["torch"],
                runtime["cuda"],
                runtime["cudnn"],
            )
        )
        observed_protocol = record["protocol"]
        for key, expected_key in (
            ("warmup_iterations", "warmup_iterations"),
            ("timed_iterations", "timed_iterations"),
            ("repeated_rounds", "repeated_rounds"),
            ("preloaded_batches", "preloaded_batches"),
            ("batch_size_per_gpu", "batch_size_per_gpu"),
            ("precision", "precision"),
            ("tf32", "tf32"),
            ("cudnn_benchmark", "cudnn_benchmark"),
            ("cudnn_deterministic", "cudnn_deterministic"),
        ):
            require_equal(
                observed_protocol[key],
                protocol[expected_key],
                f"latency protocol {key}",
            )
        require_equal(
            observed_protocol["seed"],
            protocol["benchmark_seed"],
            "latency benchmark seed",
        )
        metadata = record["benchmark_inputs"]
        input_identities.add(metadata["identity_sha256"])
        if metadata["batch_count"] != protocol["preloaded_batches"]:
            raise ContractError("latency preload batch count drift")
        for batch in metadata["batches"]:
            require_equal(
                batch["model_input"]["shape"],
                protocol["fixed_preprocessed_shapes"]["model_input"],
                "latency model input shape",
            )
            require_equal(
                batch["image_tensor_shape"],
                protocol["fixed_preprocessed_shapes"]["image_tensor"],
                "latency image shape",
            )
            require_equal(
                batch["padding_mask"]["shape"],
                protocol["fixed_preprocessed_shapes"]["padding_mask"],
                "latency padding-mask shape",
            )
    if len(signatures) != 1:
        raise ContractError("latency ranks do not share one hardware/runtime signature")
    if len(input_identities) != 1:
        raise ContractError("latency ranks do not share one exact input identity")
    signature = next(iter(signatures))
    input_identity = next(iter(input_identities))
    return (
        {
            "gpu_name": signature[0],
            "compute_capability": signature[1],
            "total_memory_bytes": signature[2],
            "torch": signature[3],
            "cuda": signature[4],
            "cudnn": signature[5],
            "world_size": gpu_count,
            "sqsh_path": expected_runtime["sqsh_path"],
        },
        {
            "identity_sha256": input_identity,
            "resolved_model_spec_sha256": resolved_model_sha256,
            "batch_count": protocol["preloaded_batches"],
            "fixed_preprocessed_shapes": copy.deepcopy(
                protocol["fixed_preprocessed_shapes"]
            ),
        },
    )


def enforce_shared_contract(
    runtime_dir: Path,
    hardware: dict[str, Any],
    inputs: dict[str, Any],
) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    lock_path = runtime_dir / ".selection_latency_contract.lock"
    hardware_path = runtime_dir / "selection_latency_hardware_contract.json"
    inputs_path = runtime_dir / "selection_latency_input_contract.json"
    # The model digest is candidate-specific and therefore excluded from the
    # shared input identity.  It remains in each candidate record.
    shared_inputs = {
        key: value
        for key, value in inputs.items()
        if key != "resolved_model_spec_sha256"
    }
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for path, actual in (
            (hardware_path, hardware),
            (inputs_path, shared_inputs),
        ):
            if path.exists():
                require_equal(load_json(path), actual, f"shared contract {path.name}")
            else:
                atomic_json(path, actual)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def latency_protocol_from_manifest(
    manifest: dict[str, Any],
    sensitivity_manifest: dict[str, Any],
) -> LatencyProtocol:
    frozen = manifest["frozen_identity"]["latency_protocol"]
    source = sensitivity_manifest["latency_protocol"]
    for key in (
        "warmup_iterations",
        "timed_iterations",
        "repeated_rounds",
        "preloaded_batches",
        "batch_size_per_gpu",
        "precision",
        "tf32",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "benchmark_seed",
        "bootstrap_resamples",
        "bootstrap_confidence_level",
        "bootstrap_seed",
    ):
        require_equal(source[key], frozen[key], f"latency source {key}")
    thresholds = LatencyValidityThresholds(**source["validity_thresholds"])
    gpu_count = manifest["frozen_identity"]["runtime"]["gpu_count_per_node"]
    return LatencyProtocol(
        warmup_iterations=source["warmup_iterations"],
        timed_iterations=source["timed_iterations"],
        repeated_rounds=source["repeated_rounds"],
        tail_percentile=source["tail_percentile"],
        bootstrap_resamples=source["bootstrap_resamples"],
        bootstrap_confidence_level=source["bootstrap_confidence_level"],
        bootstrap_seed=source["bootstrap_seed"],
        expected_devices=tuple(str(index) for index in range(gpu_count)),
        validity_thresholds=thresholds,
    )


def launch_accuracy_evaluation(
    sdk: Any,
    manifest: dict[str, Any],
    action: dict[str, Any],
    template: dict[str, Any],
    resolved_model: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    event_path: Path,
    seed: int,
    rec_id: int,
    existing_job: dict[str, Any] | None = None,
    on_submitted: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[float, dict[str, Any]]:
    ensure_sdk_importable()
    from tao_sdk.script_runner import build_entrypoint

    spec = evaluation_spec(
        manifest,
        template,
        resolved_model,
        checkpoint["path"],
        latency=False,
    )
    entrypoint = build_entrypoint(
        command=action["command"],
        specs=spec,
        inputs=action["inputs"],
        outputs=action["outputs"],
        config_format=action["config_format"],
        upload_excludes=action["upload_excludes"],
    )
    runtime = manifest["frozen_identity"]["runtime"]
    expected = {
        "evaluation_spec_sha256": sha256_value(spec),
        "evaluation_command_sha256": sha256_bytes(
            entrypoint["command"].encode("utf-8")
        ),
        "resolved_model_spec_sha256": sha256_value(spec["model"]),
        "checkpoint_sha256": checkpoint["sha256"],
    }
    if existing_job is not None:
        job_id = existing_job.get("tao_job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ContractError(
                "persisted accuracy child lacks a TAO job identifier"
            )
        for key, value in expected.items():
            require_equal(
                existing_job.get(key),
                value,
                f"persisted accuracy child {key}",
            )
        active_runtime = slurm_runtime_evidence(
            sdk,
            job_id,
            label="accuracy child resume",
        )
        reconcile_slurm_runtime_evidence(
            existing_job,
            active_runtime,
            label="accuracy child resume",
        )
        submitted = {
            **existing_job,
            **expected,
            **active_runtime,
        }
        if on_submitted is not None:
            on_submitted(copy.deepcopy(submitted))
    else:
        job = sdk.create_job(
            image=runtime["sqsh_path"],
            command=entrypoint["command"],
            gpu_count=runtime["gpu_count_per_node"],
            num_nodes=runtime["num_nodes"],
            partition=runtime["partition"],
            account=runtime["account"],
        )
        job_id = job.id
        active_runtime = slurm_runtime_evidence(
            sdk,
            job_id,
            label="accuracy child submission",
        )
        submitted = {
            "status": "submitted",
            "submitted_at_utc": utc_timestamp(),
            **expected,
            **active_runtime,
        }
        if on_submitted is not None:
            on_submitted(copy.deepcopy(submitted))
    status = wait_for_job(
        sdk,
        job_id,
        event_path=event_path,
        phase="accuracy_evaluation",
        seed=seed,
        rec_id=rec_id,
    )
    terminal_runtime = slurm_runtime_evidence(
        sdk,
        job_id,
        label="accuracy child terminal",
    )
    reconcile_slurm_runtime_evidence(
        submitted,
        terminal_runtime,
        label="accuracy child terminal",
    )
    final_evidence = {
        **submitted,
        **terminal_runtime,
        "status": status,
        "terminal_at_utc": utc_timestamp(),
    }
    if on_submitted is not None:
        on_submitted(copy.deepcopy(final_evidence))
    logs = sdk.get_job_logs(job_id, tail=5000)
    if status != "Complete":
        raise RuntimeError(
            f"accuracy evaluation {job_id} ended {status}: {logs[-4000:]}"
        )
    map50 = read_status_map50(sdk, job_id)
    if map50 is None:
        map50 = dino_metric_extractor(logs, "mAP50")
    if map50 is None:
        raise RuntimeError(f"accuracy evaluation {job_id} emitted no mAP50")
    result_root = local_lustre_path(sdk.get_job_results_dir(job_id))
    validate_result_root(result_root, job_id)
    return float(map50), {
        **final_evidence,
        "result_root": result_root,
    }


def launch_latency_benchmark(
    sdk: Any,
    manifest: dict[str, Any],
    sensitivity_manifest: dict[str, Any],
    action: dict[str, Any],
    template: dict[str, Any],
    benchmark_path: Path,
    resolved_model: dict[str, Any],
    checkpoint: dict[str, Any],
    runtime_dir: Path,
    *,
    event_path: Path,
    seed: int,
    rec_id: int,
    existing_job: dict[str, Any] | None = None,
    on_submitted: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    ensure_sdk_importable()
    from tao_sdk.script_runner import build_entrypoint

    protocol = manifest["frozen_identity"]["latency_protocol"]
    runtime = manifest["frozen_identity"]["runtime"]
    spec = evaluation_spec(
        manifest,
        template,
        resolved_model,
        checkpoint["path"],
        latency=True,
    )
    command = benchmark_command(
        benchmark_path,
        checkpoint["path"],
        protocol,
        runtime["gpu_count_per_node"],
    )
    entrypoint = build_entrypoint(
        command=command,
        specs=spec,
        inputs=action["inputs"],
        outputs={},
        config_format=action["config_format"],
        upload_excludes=action["upload_excludes"],
    )
    expected = {
        "benchmark_source_sha256": sha256_file(benchmark_path),
        "latency_spec_sha256": sha256_value(spec),
        "latency_command_sha256": sha256_bytes(
            entrypoint["command"].encode("utf-8")
        ),
        "resolved_model_spec_sha256": sha256_value(resolved_model),
        "checkpoint_sha256": checkpoint["sha256"],
    }
    if existing_job is not None:
        job_id = existing_job.get("tao_job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ContractError(
                "persisted latency child lacks a TAO job identifier"
            )
        for key, value in expected.items():
            require_equal(
                existing_job.get(key),
                value,
                f"persisted latency child {key}",
            )
        active_runtime = slurm_runtime_evidence(
            sdk,
            job_id,
            label="latency child resume",
        )
        reconcile_slurm_runtime_evidence(
            existing_job,
            active_runtime,
            label="latency child resume",
        )
        submitted = {
            **existing_job,
            **expected,
            **active_runtime,
        }
        if on_submitted is not None:
            on_submitted(copy.deepcopy(submitted))
    else:
        job = sdk.create_job(
            image=runtime["sqsh_path"],
            command=entrypoint["command"],
            gpu_count=runtime["gpu_count_per_node"],
            num_nodes=runtime["num_nodes"],
            partition=runtime["partition"],
            account=runtime["account"],
        )
        job_id = job.id
        active_runtime = slurm_runtime_evidence(
            sdk,
            job_id,
            label="latency child submission",
        )
        submitted = {
            "status": "submitted",
            "submitted_at_utc": utc_timestamp(),
            **expected,
            **active_runtime,
        }
        if on_submitted is not None:
            on_submitted(copy.deepcopy(submitted))
    status = wait_for_job(
        sdk,
        job_id,
        event_path=event_path,
        phase="selection_time_latency",
        seed=seed,
        rec_id=rec_id,
    )
    terminal_runtime = slurm_runtime_evidence(
        sdk,
        job_id,
        label="latency child terminal",
    )
    reconcile_slurm_runtime_evidence(
        submitted,
        terminal_runtime,
        label="latency child terminal",
    )
    final_evidence = {
        **submitted,
        **terminal_runtime,
        "status": status,
        "terminal_at_utc": utc_timestamp(),
    }
    if on_submitted is not None:
        on_submitted(copy.deepcopy(final_evidence))
    logs = sdk.get_job_logs(job_id, tail=5000)
    if status != "Complete" or "TAO_AUTOML_LATENCY_COMPLETE" not in logs:
        raise RuntimeError(
            f"latency benchmark {job_id} did not complete cleanly "
            f"(status={status}): {logs[-6000:]}"
        )
    rank_records = read_latency_rank_records(
        sdk,
        job_id,
        runtime["gpu_count_per_node"],
    )
    model_sha = expected["resolved_model_spec_sha256"]
    hardware, inputs = validate_latency_rank_contract(
        manifest,
        rank_records,
        checkpoint=checkpoint,
        resolved_model_sha256=model_sha,
    )
    enforce_shared_contract(runtime_dir, hardware, inputs)
    samples = {
        round_index: {
            str(record["rank"]): record["samples_ms"][round_index]
            for record in rank_records
        }
        for round_index in range(protocol["repeated_rounds"])
    }
    latency_protocol = latency_protocol_from_manifest(
        manifest,
        sensitivity_manifest,
    )
    statistics = aggregate_synchronized_latency(samples, latency_protocol)
    if not statistics.is_valid:
        raise RuntimeError(
            "selection-time latency failed frozen validity thresholds: "
            + ",".join(statistics.invalid_reasons)
        )
    statistics_dict = asdict(statistics)
    metrics = {
        "latency_ms": statistics.median_ms,
        "latency_p95_ms": statistics.tail_latency_ms,
        "latency_mad_ms": statistics.mad_ms,
        "latency_iqr_ms": statistics.iqr_ms,
        "latency_robust_cv": statistics.robust_cv,
        "latency_ci95_low": statistics.bootstrap_median_ci_ms[0],
        "latency_ci95_high": statistics.bootstrap_median_ci_ms[1],
        "latency_bootstrap_ci_width_ms": statistics.bootstrap_ci_width_ms,
        "latency_round_drift_fraction": statistics.round_drift_fraction,
        "latency_device_range_fraction": statistics.device_median_range_fraction,
        "latency_synchronized_median_ms": statistics.synchronized_median_ms,
        "latency_synchronized_p95_ms": statistics.synchronized_tail_latency_ms,
    }
    result_root = local_lustre_path(sdk.get_job_results_dir(job_id))
    validate_result_root(result_root, job_id)
    return metrics, {
        **final_evidence,
        "result_root": result_root,
        "raw_samples_dir": f"{result_root}/latency",
        "statistics": statistics_dict,
        "hardware_contract": hardware,
        "benchmark_inputs": inputs,
        "rank_hardware": [
            {
                "rank": record["rank"],
                "hostname": record["hardware"]["hostname"],
                "hardware": record["hardware"],
                "runtime": record["runtime"],
            }
            for record in rank_records
        ],
    }


def candidate_id(seed: int, rec_id: int) -> str:
    return f"seed_{seed}_rec_{rec_id}"


def read_candidate_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 1,
            "manual_candidate_injection_permitted": False,
            "records": {},
        }
    ledger = load_json(path)
    require_equal(ledger.get("schema_version"), 1, "candidate ledger schema")
    require_equal(
        ledger.get("manual_candidate_injection_permitted"),
        False,
        "candidate ledger manual injection",
    )
    if not isinstance(ledger.get("records"), dict):
        raise ContractError("candidate ledger records must be a mapping")
    return ledger


def write_candidate_ledger(
    path: Path,
    manifest_file_sha256: str,
    seed: int,
    records: dict[str, Any],
) -> None:
    atomic_json(
        path,
        {
            "schema_version": 1,
            "manifest_file_sha256": manifest_file_sha256,
            "search_seed": seed,
            "manual_candidate_injection_permitted": False,
            "records": records,
        },
    )


def seed_archive_path(runtime_dir: Path, seed: int) -> Path:
    return runtime_dir / f"seed_{seed}" / "seed_archive.v1.json"


def validate_seed_archive(
    archive: dict[str, Any],
    *,
    manifest_file_sha256: str,
    seed: int,
    recommendations: int,
) -> None:
    require_equal(archive.get("schema_version"), 1, "seed archive schema")
    require_equal(archive.get("status"), "complete", "seed archive status")
    require_equal(
        archive.get("manifest_file_sha256"),
        manifest_file_sha256,
        "seed archive manifest",
    )
    require_equal(archive.get("search_seed"), seed, "seed archive search seed")
    require_equal(
        archive.get("manual_candidate_injection_used"),
        False,
        "seed archive manual injection",
    )
    records = archive.get("records")
    if not isinstance(records, dict):
        raise ContractError("seed archive records must be a mapping")
    expected_ids = {
        candidate_id(seed, rec_id) for rec_id in range(recommendations)
    }
    require_equal(set(records), expected_ids, "seed archive candidate IDs")
    for key, record in records.items():
        if record.get("status") not in TERMINAL_CANDIDATE_STATUSES:
            raise ContractError(
                f"seed archive contains non-terminal candidate {key}: "
                f"{record.get('status')}"
            )
        require_equal(record.get("candidate_id"), key, f"{key} identity")
        require_equal(record.get("search_seed"), seed, f"{key} search seed")
        require_equal(
            record.get("manual_candidate_injection_used"),
            False,
            f"{key} manual candidate injection",
        )
    claimed = require_sha256(
        archive.get("archive_sha256"),
        "seed archive internal SHA256",
    )
    unhashed = copy.deepcopy(archive)
    del unhashed["archive_sha256"]
    require_equal(
        sha256_value(unhashed),
        claimed,
        "seed archive internal SHA256",
    )


def finalize_seed_archive(
    runtime_dir: Path,
    manifest_file_sha256: str,
    seed: int,
    recommendations: int,
    records: dict[str, Any],
    automl_result: dict[str, Any],
) -> Path:
    path = seed_archive_path(runtime_dir, seed)
    automl_result_sha256 = sha256_value(automl_result)
    if path.exists():
        existing = load_json(path)
        validate_seed_archive(
            existing,
            manifest_file_sha256=manifest_file_sha256,
            seed=seed,
            recommendations=recommendations,
        )
        if (
            existing["records"] != records
            or existing["automl_result_sha256"] != automl_result_sha256
        ):
            raise ContractError(
                f"immutable completed seed archive already exists and differs: {path}"
            )
        return path
    payload = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": utc_timestamp(),
        "manifest_file_sha256": manifest_file_sha256,
        "search_seed": seed,
        "recommendations": recommendations,
        "manual_candidate_injection_used": False,
        "records": copy.deepcopy(records),
        "automl_result_sha256": automl_result_sha256,
    }
    payload["archive_sha256"] = sha256_value(payload)
    validate_seed_archive(
        payload,
        manifest_file_sha256=manifest_file_sha256,
        seed=seed,
        recommendations=recommendations,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    if pending.exists():
        raise ContractError(f"stale pending immutable seed archive: {pending}")
    try:
        with pending.open("x", encoding="utf-8") as stream:
            json.dump(
                payload,
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
    finally:
        if pending.exists():
            pending.unlink()
    return path


def resolve_seed_workspace(
    seed_dir: Path,
    records: Mapping[str, Any],
    *,
    resume_requested: bool,
) -> tuple[Path, bool]:
    """Resolve a seed's exact runner workspace without duplicating work.

    ``--resume`` is global across the three controllers.  A seed that never
    started must therefore begin fresh while another seed resumes its one
    durable ``run_*`` workspace.
    """

    workspace_base = seed_dir / "workspace"
    workspace_exists = (
        workspace_base.exists() and any(workspace_base.iterdir())
    )
    has_partial_state = bool(records) or workspace_exists
    if not resume_requested and has_partial_state:
        raise ContractError(
            f"seed state already exists at {seed_dir}; use --resume rather "
            "than starting a duplicate controller"
        )
    if resume_requested and records and not workspace_exists:
        raise ContractError(
            f"candidate ledger exists at {seed_dir} but its AutoML workspace "
            "is missing; refusing an unsafe partial resume"
        )
    effective_resume = resume_requested and has_partial_state
    if not effective_resume:
        return workspace_base, False
    run_workspaces = sorted(
        path for path in workspace_base.glob("run_*") if path.is_dir()
    )
    if len(run_workspaces) != 1:
        raise ContractError(
            f"resume requires exactly one full run_* workspace under "
            f"{workspace_base}, found {len(run_workspaces)}"
        )
    return run_workspaces[0], True


def run_seed(
    manifest_path: str,
    manifest_file_sha256: str,
    runtime_dir_path: str,
    seed: int,
    resume: bool,
) -> None:
    manifest, _ = load_manifest(
        Path(manifest_path),
        supplied_file_sha256=manifest_file_sha256,
    )
    local_checks = validate_local_provenance(manifest, Path(manifest_path))
    sensitivity_manifest_path = Path(
        manifest["derivation"]["source_identity"]["sensitivity_manifest_path"]
    )
    sensitivity_manifest = load_json(sensitivity_manifest_path)
    benchmark_path = Path(local_checks["benchmark"]["path"])
    evaluate_template_path = Path(local_checks["evaluate_template"]["path"])
    evaluate_template = yaml.safe_load(evaluate_template_path.read_text())
    train_template = yaml.safe_load(Path(local_checks["train_template"]["path"]).read_text())
    base_schema = load_schema_json(Path(local_checks["train_schema"]["path"]))
    derived_schema, parameters, custom_ranges = build_search_contract(
        manifest,
        base_schema,
    )
    base_train_spec = training_spec(manifest, train_template)
    overrides = spec_overrides_from_base(base_train_spec)

    skill_info = yaml.safe_load(
        (SKILL_DIR / "references" / "skill_info.yaml").read_text()
    )
    evaluate_action = skill_info["actions"]["evaluate"]
    require_equal(
        evaluate_action["command"],
        "dino evaluate -e {config_path}",
        "DINO evaluate command",
    )
    require_equal(evaluate_action["config_format"], "yaml", "evaluate format")

    runtime_dir = Path(runtime_dir_path)
    seed_dir = runtime_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    event_path = seed_dir / "events.jsonl"
    ledger_path = seed_dir / "candidate_evaluations.json"
    result_path = seed_dir / "result.json"
    ledger = read_candidate_ledger(ledger_path)
    existing_manifest_sha = ledger.get("manifest_file_sha256")
    if existing_manifest_sha is not None:
        require_equal(
            existing_manifest_sha,
            manifest_file_sha256,
            "candidate ledger manifest",
        )
    records: dict[str, Any] = ledger["records"]
    final_archive_path = seed_archive_path(runtime_dir, seed)
    if final_archive_path.exists():
        final_archive = load_json(final_archive_path)
        validate_seed_archive(
            final_archive,
            manifest_file_sha256=manifest_file_sha256,
            seed=seed,
            recommendations=manifest["search_design"][
                "recommendations_per_seed"
            ],
        )
        # A completed immutable seed is idempotent.  Never reopen its
        # workspace, resubmit jobs, or rewrite its candidate records.
        print(
            f"SEED_ARCHIVE_ALREADY_COMPLETE seed={seed} "
            f"path={final_archive_path}",
            flush=True,
        )
        return
    automl_workspace_path, effective_resume = resolve_seed_workspace(
        seed_dir,
        records,
        resume_requested=resume,
    )

    configure_slurm(manifest)
    ensure_sdk_importable()
    from tao_automl.runner import AutoMLRunner
    from tao_sdk.platforms.slurm import SlurmSDK

    sdk = SlurmSDK(
        poll_interval=10,
        state_file=seed_dir / "slurm_state.json",
    )
    runner = AutoMLRunner(
        sdk=sdk,
        skill_dir=SKILL_DIR,
        action="train",
        poll_interval=10,
    )
    # JSON-Schema enum is the canonical numeric discrete-space mechanism.
    # The overlay changes only metadata used by the brain; all training/eval
    # configs still come from the pinned skill templates.
    runner.skill_ctx.schema = derived_schema

    def upsert(rec_id: int, changes: dict[str, Any]) -> dict[str, Any]:
        key = candidate_id(seed, rec_id)
        record = records.setdefault(
            key,
            {
                "candidate_id": key,
                "search_seed": seed,
                "training_seed": manifest["search_design"]["training_seed"],
                "rec_id": rec_id,
                "status": "created",
                "attempts": [],
            },
        )
        record.update(copy.deepcopy(changes))
        write_candidate_ledger(
            ledger_path,
            manifest_file_sha256,
            seed,
            records,
        )
        return record

    def on_recommendation(rec: Any) -> None:
        rec_specs = dict(rec.specs)
        resolved_model = apply_candidate_to_reference_model(manifest, rec_specs)
        resolved_train = training_spec(manifest, train_template, rec_specs)
        record = upsert(
            int(rec.id),
            {
                "status": "recommended",
                "specs": rec_specs,
                "resolved_model_spec": resolved_model,
                "resolved_model_spec_sha256": sha256_value(resolved_model),
                "resolved_train_spec_sha256": sha256_value(resolved_train),
                "search_manifest_file_sha256": manifest_file_sha256,
                "manual_candidate_injection_used": False,
            },
        )
        append_event(
            event_path,
            {
                "event": "algorithm_recommendation",
                "candidate_id": record["candidate_id"],
                "specs": rec_specs,
                "resolved_model_spec_sha256": record[
                    "resolved_model_spec_sha256"
                ],
                "resolved_train_spec_sha256": record[
                    "resolved_train_spec_sha256"
                ],
            },
        )

    def evaluate_candidate(rec: Any, train_job_id: str) -> dict[str, Any]:
        rec_id = int(rec.id)
        key = candidate_id(seed, rec_id)
        rec_specs = dict(rec.specs)
        resolved_model = apply_candidate_to_reference_model(manifest, rec_specs)
        existing = records.get(key, {})
        training_runtime = slurm_runtime_evidence(
            sdk,
            train_job_id,
            label=f"{key} training terminal",
        )
        if isinstance(existing.get("training_runtime"), dict):
            reconcile_slurm_runtime_evidence(
                existing["training_runtime"],
                training_runtime,
                label=f"{key} training resume",
            )
        if existing.get("status") == "success":
            require_equal(existing.get("specs"), rec_specs, "resumed candidate specs")
            require_equal(
                existing.get("train_job_id"),
                train_job_id,
                "resumed candidate training job",
            )
            upsert(rec_id, {"training_runtime": training_runtime})
            return copy.deepcopy(existing["objective_values"])

        record = upsert(
            rec_id,
            {
                "status": "evaluating",
                "specs": rec_specs,
                "resolved_model_spec": resolved_model,
                "resolved_model_spec_sha256": sha256_value(resolved_model),
                "resolved_train_spec_sha256": sha256_value(
                    training_spec(manifest, train_template, rec_specs)
                ),
                "train_job_id": train_job_id,
                "training_runtime": training_runtime,
            },
        )
        try:
            checkpoint = record.get("checkpoint")
            if not isinstance(checkpoint, dict):
                checkpoint = discover_terminal_checkpoint(sdk, train_job_id)
                record = upsert(rec_id, {"checkpoint": checkpoint})

            cached_accuracy = record.get("accuracy_evaluation")
            if (
                not isinstance(cached_accuracy, dict)
                or cached_accuracy.get("status") != "Complete"
                or "mAP50" not in record
            ):
                map50, accuracy_job = launch_accuracy_evaluation(
                    sdk,
                    manifest,
                    evaluate_action,
                    evaluate_template,
                    resolved_model,
                    checkpoint,
                    event_path=event_path,
                    seed=seed,
                    rec_id=rec_id,
                    existing_job=(
                        cached_accuracy
                        if isinstance(cached_accuracy, dict)
                        else None
                    ),
                    on_submitted=lambda job: upsert(
                        rec_id,
                        {"accuracy_evaluation": job},
                    ),
                )
                record = upsert(
                    rec_id,
                    {
                        "mAP50": map50,
                        "accuracy_evaluation": accuracy_job,
                    },
                )
            else:
                map50 = finite_number(record["mAP50"], "cached mAP50")

            cached_latency = record.get("selection_time_latency")
            latency_metrics, latency_job = launch_latency_benchmark(
                sdk,
                manifest,
                sensitivity_manifest,
                evaluate_action,
                evaluate_template,
                benchmark_path,
                resolved_model,
                checkpoint,
                runtime_dir,
                event_path=event_path,
                seed=seed,
                rec_id=rec_id,
                existing_job=(
                    cached_latency
                    if isinstance(cached_latency, dict)
                    else None
                ),
                on_submitted=lambda job: upsert(
                    rec_id,
                    {"selection_time_latency": job},
                ),
            )
            objective_values = {"mAP50": map50, **latency_metrics}
            record = upsert(
                rec_id,
                {
                    "status": "success",
                    "selection_time_latency": latency_job,
                    "objective_values": objective_values,
                    "completed_at_utc": utc_timestamp(),
                    "measurements_feed_selection": True,
                    "winner_selected_during_measurement": False,
                },
            )
            append_event(
                event_path,
                {
                    "event": "candidate_measurement_complete",
                    "candidate_id": key,
                    "objective_values": objective_values,
                    "train_job_id": train_job_id,
                    "accuracy_job_id": record["accuracy_evaluation"]["tao_job_id"],
                    "latency_job_id": latency_job["tao_job_id"],
                },
            )
            return copy.deepcopy(objective_values)
        except BaseException as error:
            attempts = list(record.get("attempts", []))
            attempts.append(
                {
                    "failed_at_utc": utc_timestamp(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            upsert(
                rec_id,
                {
                    "status": "measurement_failure",
                    "failure_reason": f"{type(error).__name__}: {error}",
                    "attempts": attempts,
                },
            )
            append_event(
                event_path,
                {
                    "event": "candidate_measurement_failure",
                    "candidate_id": key,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise

    def on_result(rec: Any, metric: Any, status: str) -> None:
        rec_id = int(rec.id)
        key = candidate_id(seed, rec_id)
        record = records.get(key)
        if record is None:
            # The only legitimate path here is a training failure before
            # eval_fn.  Its recommendation still came from the brain.
            on_recommendation(rec)
            record = records[key]
        changes: dict[str, Any] = {
            "automl_result_status": status,
            "train_job_id": getattr(rec, "job_id", None),
        }
        train_job_id = getattr(rec, "job_id", None)
        if isinstance(train_job_id, str) and train_job_id:
            training_runtime = slurm_runtime_evidence(
                sdk,
                train_job_id,
                label=f"{key} AutoML training result",
                require_active_slurm_id=(
                    str(status).lower() in SUCCESS_REC_STATUSES
                ),
            )
            if isinstance(record.get("training_runtime"), dict):
                reconcile_slurm_runtime_evidence(
                    record["training_runtime"],
                    training_runtime,
                    label=f"{key} AutoML training result",
                )
            changes["training_runtime"] = training_runtime
        if str(status).lower() not in SUCCESS_REC_STATUSES and record.get(
            "status"
        ) != "success":
            changes["status"] = "training_or_measurement_failure"
            changes["failure_reason"] = getattr(
                rec,
                "failure_reason",
                record.get("failure_reason", f"AutoML status {status}"),
            )
        upsert(rec_id, changes)
        append_event(
            event_path,
            {
                "event": "automl_result",
                "candidate_id": key,
                "status": status,
                "metric": metric,
                "train_job_id": getattr(rec, "job_id", None),
            },
        )

    settings = selection_settings(manifest, seed)
    try:
        result = runner.run(
            train_dataset_uri=manifest["scope"]["dataset_uri"],
            eval_dataset_uri=manifest["scope"]["dataset_uri"],
            base_checkpoint=manifest["frozen_identity"]["runtime"][
                "pretrained_model_path"
            ],
            workspace_id=f"dino-expanded-search-{seed}",
            image=manifest["frozen_identity"]["runtime"]["sqsh_path"],
            automl_settings=settings,
            automl_hyperparameters=parameters,
            custom_param_ranges=custom_ranges,
            workspace_path=str(automl_workspace_path),
            spec_overrides=overrides,
            metric_extractor=dino_metric_extractor,
            eval_fn=evaluate_candidate,
            on_recommendation=on_recommendation,
            on_result=on_result,
            resume=effective_resume,
            gpu_count=manifest["frozen_identity"]["runtime"]["gpu_count_per_node"],
            num_nodes=manifest["frozen_identity"]["runtime"]["num_nodes"],
            partition=manifest["frozen_identity"]["runtime"]["partition"],
            account=manifest["frozen_identity"]["runtime"]["account"],
        )
    except BaseException as error:
        atomic_json(
            result_path,
            {
                "schema_version": 1,
                "status": "failure",
                "search_seed": seed,
                "manifest_file_sha256": manifest_file_sha256,
                "error": f"{type(error).__name__}: {error}",
                "candidate_count": len(records),
            },
        )
        raise
    atomic_json(
        result_path,
        {
            "schema_version": 1,
            "status": "success",
            "search_seed": seed,
            "manifest_file_sha256": manifest_file_sha256,
            "candidate_count": len(records),
            "derived_search_schema_sha256": sha256_schema_value(derived_schema),
            "custom_param_ranges": custom_ranges,
            "result": result,
        },
    )
    archive_path = finalize_seed_archive(
        runtime_dir,
        manifest_file_sha256,
        seed,
        manifest["search_design"]["recommendations_per_seed"],
        records,
        result,
    )
    print(
        f"SEED_ARCHIVE_COMPLETE seed={seed} path={archive_path} "
        f"sha256={sha256_file(archive_path)}",
        flush=True,
    )


def load_complete_archive(
    manifest: dict[str, Any],
    manifest_file_sha256: str,
    runtime_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for seed in manifest["search_design"]["search_seeds"]:
        path = seed_archive_path(runtime_dir, seed)
        archive = load_json(path)
        validate_seed_archive(
            archive,
            manifest_file_sha256=manifest_file_sha256,
            seed=seed,
            recommendations=manifest["search_design"][
                "recommendations_per_seed"
            ],
        )
        by_id = archive["records"]
        records.extend(copy.deepcopy(by_id[key]) for key in sorted(by_id))
    if len(records) != EXPECTED_TOTAL_CANDIDATES:
        raise ContractError("expanded archive must contain exactly 60 candidate records")
    successful = [
        item
        for item in records
        if item.get("status") == "success"
        and isinstance(item.get("objective_values"), dict)
    ]
    for record in successful:
        require_equal(
            sha256_value(record["resolved_model_spec"]),
            record["resolved_model_spec_sha256"],
            f"{record['candidate_id']} full model digest",
        )
        apply_candidate_to_reference_model(manifest, record["specs"])
        validate_recorded_slurm_runtime(
            record.get("training_runtime", {}),
            label=f"{record['candidate_id']} training runtime",
        )
        validate_recorded_slurm_runtime(
            record.get("accuracy_evaluation", {}),
            label=f"{record['candidate_id']} accuracy runtime",
        )
        validate_recorded_slurm_runtime(
            record.get("selection_time_latency", {}),
            label=f"{record['candidate_id']} latency runtime",
        )
        for metric in (
            "mAP50",
            "latency_ms",
            "latency_p95_ms",
            "latency_ci95_low",
            "latency_ci95_high",
        ):
            finite_number(
                record["objective_values"].get(metric),
                f"{record['candidate_id']}.{metric}",
            )
    if not successful:
        raise ContractError("expanded archive contains no successful candidates")
    return records, successful


def selector_candidates(records: Iterable[dict[str, Any]]) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            id=record["candidate_id"],
            specs=copy.deepcopy(record["specs"]),
            status="success",
            objective_values=copy.deepcopy(record["objective_values"]),
        )
        for record in records
    ]


def analysis_signature(analysis: dict[str, Any]) -> dict[str, Any]:
    candidates = {
        item["candidate_id"]: {
            "valid": item["valid"],
            "pareto_rank": item["pareto_rank"],
            "dominated_by": item["dominated_by"],
            "multi_objective_pareto_rank": item[
                "multi_objective_pareto_rank"
            ],
            "multi_objective_dominated_by": item[
                "multi_objective_dominated_by"
            ],
            "normalized_accuracy_objective": item[
                "normalized_accuracy_objective"
            ],
            "normalized_latency_objective": item[
                "normalized_latency_objective"
            ],
            "multi_objective_compromise_score": item[
                "multi_objective_compromise_score"
            ],
            "winner": item["winner"],
        }
        for item in analysis["candidates"]
    }
    return {
        "selections": analysis["selections"],
        "normalization_bounds": analysis["algorithm"]["normalization_bounds"],
        "candidates": candidates,
    }


def analyze_union_archive(
    manifest: dict[str, Any],
    successful: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = selection_settings(manifest, EXPECTED_SEARCH_SEEDS[0])
    objective = parse_objective_config(settings)
    orderings = {
        "archive_order": successful,
        "reverse_order": list(reversed(successful)),
        "candidate_id_order": sorted(successful, key=lambda item: item["candidate_id"]),
    }
    analyses = {
        name: objective.analyze_archive(selector_candidates(records)).to_dict()
        for name, records in orderings.items()
    }
    signatures = {
        name: analysis_signature(value)
        for name, value in analyses.items()
    }
    reference = signatures["archive_order"]
    if any(value != reference for value in signatures.values()):
        raise ContractError("production selector changed under candidate reordering")
    analysis = analyses["archive_order"]
    audits = {
        item["candidate_id"]: item for item in analysis["candidates"]
    }
    multi_id = analysis["selections"]["multi_objective"]["winner_id"]
    if multi_id is not None:
        multi_audit = audits[multi_id]
        if multi_audit["multi_objective_pareto_rank"] != 0:
            raise ContractError("production multi-objective winner is dominated")
    return analysis, {
        "orderings_checked": list(orderings),
        "order_independent": True,
        "signature_sha256": sha256_value(reference),
    }


def write_candidate_artifacts(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    analysis: dict[str, Any],
    runtime_dir: Path,
) -> dict[str, str]:
    audits = {item["candidate_id"]: item for item in analysis["candidates"]}
    table = []
    for record in sorted(records, key=lambda item: item["candidate_id"]):
        row = {
            "candidate_id": record["candidate_id"],
            "search_seed": record["search_seed"],
            "training_seed": record["training_seed"],
            "rec_id": record["rec_id"],
            "status": record["status"],
            "specs": copy.deepcopy(record.get("specs")),
            "resolved_model_spec": copy.deepcopy(record.get("resolved_model_spec")),
            "resolved_model_spec_sha256": record.get(
                "resolved_model_spec_sha256"
            ),
            "checkpoint": copy.deepcopy(record.get("checkpoint")),
            "train_job_id": record.get("train_job_id"),
            "training_runtime": copy.deepcopy(
                record.get("training_runtime")
            ),
            "accuracy_evaluation": copy.deepcopy(
                record.get("accuracy_evaluation")
            ),
            "selection_time_latency": copy.deepcopy(
                record.get("selection_time_latency")
            ),
            "objective_values": copy.deepcopy(record.get("objective_values")),
            "selection_audit": copy.deepcopy(audits.get(record["candidate_id"])),
            "failure_reason": record.get("failure_reason"),
        }
        table.append(row)
    json_path = runtime_dir / "expanded_candidate_table.json"
    atomic_json(
        json_path,
        {
            "schema_version": 1,
            "candidate_count": len(table),
            "successful_count": sum(item["status"] == "success" for item in table),
            "manual_candidate_injection_used": False,
            "rows": table,
        },
    )

    csv_path = runtime_dir / "expanded_candidate_table.csv"
    search_parameters = manifest["search_space"]["search_parameters"]
    columns = [
        "candidate_id",
        "search_seed",
        "training_seed",
        "rec_id",
        "status",
        *search_parameters,
        "mAP50",
        "latency_ms",
        "latency_p95_ms",
        "latency_mad_ms",
        "latency_iqr_ms",
        "latency_robust_cv",
        "latency_ci95_low",
        "latency_ci95_high",
        "latency_accuracy_feasible",
        "multi_objective_accuracy_feasible",
        "pareto_rank",
        "dominated_by",
        "multi_objective_pareto_rank",
        "multi_objective_dominated_by",
        "normalized_accuracy_objective",
        "normalized_latency_objective",
        "multi_objective_compromise_score",
        "accuracy_winner",
        "latency_winner",
        "multi_objective_winner",
        "train_job_id",
        "train_slurm_job_id",
        "train_retry_count",
        "train_failed_slurm_job_ids",
        "train_launch_uncertain",
        "train_runtime_revision",
        "accuracy_job_id",
        "accuracy_slurm_job_id",
        "accuracy_retry_count",
        "accuracy_failed_slurm_job_ids",
        "accuracy_launch_uncertain",
        "accuracy_runtime_revision",
        "latency_job_id",
        "latency_slurm_job_id",
        "latency_retry_count",
        "latency_failed_slurm_job_ids",
        "latency_launch_uncertain",
        "latency_runtime_revision",
        "checkpoint_path",
        "checkpoint_sha256",
        "failure_reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in table:
            record = next(
                item for item in records if item["candidate_id"] == row["candidate_id"]
            )
            specs = record.get("specs", {})
            values = record.get("objective_values", {})
            audit = row["selection_audit"] or {}
            winner = audit.get("winner", {})
            checkpoint = record.get("checkpoint", {})
            training_runtime = record.get("training_runtime", {})
            accuracy_job = record.get("accuracy_evaluation", {})
            latency_job = record.get("selection_time_latency", {})
            output = {
                "candidate_id": record["candidate_id"],
                "search_seed": record["search_seed"],
                "training_seed": record["training_seed"],
                "rec_id": record["rec_id"],
                "status": record["status"],
                **{path: specs.get(path) for path in search_parameters},
                **{name: values.get(name) for name in (
                    "mAP50",
                    "latency_ms",
                    "latency_p95_ms",
                    "latency_mad_ms",
                    "latency_iqr_ms",
                    "latency_robust_cv",
                    "latency_ci95_low",
                    "latency_ci95_high",
                )},
                "latency_accuracy_feasible": audit.get(
                    "latency_accuracy_feasible"
                ),
                "multi_objective_accuracy_feasible": audit.get(
                    "multi_objective_accuracy_feasible"
                ),
                "pareto_rank": audit.get("pareto_rank"),
                "dominated_by": ";".join(audit.get("dominated_by", [])),
                "multi_objective_pareto_rank": audit.get(
                    "multi_objective_pareto_rank"
                ),
                "multi_objective_dominated_by": ";".join(
                    audit.get("multi_objective_dominated_by", [])
                ),
                "normalized_accuracy_objective": audit.get(
                    "normalized_accuracy_objective"
                ),
                "normalized_latency_objective": audit.get(
                    "normalized_latency_objective"
                ),
                "multi_objective_compromise_score": audit.get(
                    "multi_objective_compromise_score"
                ),
                "accuracy_winner": winner.get("accuracy"),
                "latency_winner": winner.get("latency"),
                "multi_objective_winner": winner.get("multi_objective"),
                "train_job_id": record.get("train_job_id"),
                "train_slurm_job_id": training_runtime.get("slurm_job_id"),
                "train_retry_count": training_runtime.get("retry_count"),
                "train_failed_slurm_job_ids": ";".join(
                    training_runtime.get("failed_slurm_job_ids", [])
                ),
                "train_launch_uncertain": training_runtime.get(
                    "launch_uncertain"
                ),
                "train_runtime_revision": training_runtime.get(
                    "runtime_revision"
                ),
                "accuracy_job_id": accuracy_job.get("tao_job_id"),
                "accuracy_slurm_job_id": accuracy_job.get("slurm_job_id"),
                "accuracy_retry_count": accuracy_job.get("retry_count"),
                "accuracy_failed_slurm_job_ids": ";".join(
                    accuracy_job.get("failed_slurm_job_ids", [])
                ),
                "accuracy_launch_uncertain": accuracy_job.get(
                    "launch_uncertain"
                ),
                "accuracy_runtime_revision": accuracy_job.get(
                    "runtime_revision"
                ),
                "latency_job_id": latency_job.get("tao_job_id"),
                "latency_slurm_job_id": latency_job.get("slurm_job_id"),
                "latency_retry_count": latency_job.get("retry_count"),
                "latency_failed_slurm_job_ids": ";".join(
                    latency_job.get("failed_slurm_job_ids", [])
                ),
                "latency_launch_uncertain": latency_job.get(
                    "launch_uncertain"
                ),
                "latency_runtime_revision": latency_job.get(
                    "runtime_revision"
                ),
                "checkpoint_path": checkpoint.get("path"),
                "checkpoint_sha256": checkpoint.get("sha256"),
                "failure_reason": record.get("failure_reason"),
            }
            writer.writerow(output)
    return {
        "candidate_table_json": str(json_path),
        "candidate_table_json_sha256": sha256_file(json_path),
        "candidate_table_csv": str(csv_path),
        "candidate_table_csv_sha256": sha256_file(csv_path),
    }


def combine_results(
    manifest: dict[str, Any],
    manifest_path: Path,
    manifest_file_sha256: str,
    runtime_dir: Path,
) -> dict[str, Any]:
    records, successful = load_complete_archive(
        manifest,
        manifest_file_sha256,
        runtime_dir,
    )
    analysis, order_audit = analyze_union_archive(manifest, successful)
    analysis["search"] = {
        "algorithm": manifest["search_design"]["algorithm"],
        "seeds": manifest["search_design"]["search_seeds"],
        "recommendations_per_seed": manifest["search_design"][
            "recommendations_per_seed"
        ],
        "total_candidate_records": len(records),
        "successful_candidates": len(successful),
        "candidate_generation": (
            "three preregistered sequential Bayesian subarchives"
        ),
        "selection_population": (
            "unchanged union of every successful finite measured candidate"
        ),
        "all_modes_receive_identical_archive": True,
    }
    analysis["selection_authority"] = {
        "module": "tao_automl.selection",
        "function": "analyze_archive",
        "source_path": str(
            Path(inspect.getsourcefile(analyze_archive) or "").resolve()
        ),
        "source_sha256": sha256_file(
            Path(inspect.getsourcefile(analyze_archive) or "").resolve()
        ),
        "manual_override_used": False,
        "candidate_reordering_used": False,
        "order_independence_audit": order_audit,
    }
    analysis["manifest"] = {
        "path": str(manifest_path.resolve()),
        "whole_file_sha256": manifest_file_sha256,
        "internal_manifest_sha256": manifest["manifest_sha256"],
    }
    analysis["post_front_matched_validation"] = {
        "contract": copy.deepcopy(
            manifest["post_front_matched_validation"]
        ),
        "contract_sha256": EXPECTED_POST_FRONT_CONTRACT_SHA256,
        "selection_time_measurements_replaced": False,
        "measurements_feed_reselection": False,
        "status": "preregistered_not_run_by_expanded_search_runner",
    }
    combined_path = runtime_dir / "expanded_combined_selection.json"
    atomic_json(combined_path, analysis)
    table_artifacts = write_candidate_artifacts(
        manifest,
        records,
        analysis,
        runtime_dir,
    )
    integrity = {
        "schema_version": 1,
        "created_at_utc": utc_timestamp(),
        "scope": copy.deepcopy(manifest["scope"]),
        "manifest": analysis["manifest"],
        "runner_source": validate_runner_source_provenance(manifest),
        "candidate_budget": {
            "expected": EXPECTED_TOTAL_CANDIDATES,
            "observed": len(records),
            "successful": len(successful),
            "failed": len(records) - len(successful),
        },
        "selection": {
            "authority": analysis["selection_authority"],
            "settings": selection_settings(
                manifest,
                EXPECTED_SEARCH_SEEDS[0],
            ),
            "selected_candidate_ids": {
                mode: value["winner_id"]
                for mode, value in analysis["selections"].items()
            },
            "manual_override_used": False,
            "algorithm_only": True,
            "dominated_multi_objective_winner": False,
        },
        "selection_time_measurements_preserved": True,
        "post_selection_measurements_feed_selection": False,
        "post_front_matched_validation": copy.deepcopy(
            analysis["post_front_matched_validation"]
        ),
        "artifacts": {
            "combined_selection": str(combined_path),
            "combined_selection_sha256": sha256_file(combined_path),
            **table_artifacts,
        },
    }
    integrity_path = runtime_dir / "expanded_integrity_audit.json"
    atomic_json(integrity_path, integrity)
    # Include the integrity file's own whole-file digest in stdout/result
    # rather than recursively embedding it in itself.
    result = {
        "status": "complete",
        "combined_selection": str(combined_path),
        "combined_selection_sha256": sha256_file(combined_path),
        "integrity_audit": str(integrity_path),
        "integrity_audit_sha256": sha256_file(integrity_path),
        "candidate_artifacts": table_artifacts,
        "selections": analysis["selections"],
    }
    atomic_json(runtime_dir / "expanded_completion.json", result)
    return result


def dry_run_report(
    manifest: dict[str, Any],
    manifest_path: Path,
    manifest_file_sha256: str,
    local_checks: dict[str, Any],
    remote_checks: dict[str, Any] | None,
) -> dict[str, Any]:
    train_schema = load_schema_json(Path(local_checks["train_schema"]["path"]))
    derived_schema, parameters, custom_ranges = build_search_contract(
        manifest,
        train_schema,
    )
    train_template = yaml.safe_load(
        Path(local_checks["train_template"]["path"]).read_text()
    )
    base_spec = training_spec(manifest, train_template)
    selector = validate_selector_configuration(manifest)
    recommendations = [
        {
            "search_seed": seed,
            "budget": manifest["search_design"]["recommendations_per_seed"],
            "candidate_ids": [
                candidate_id(seed, index)
                for index in range(
                    manifest["search_design"]["recommendations_per_seed"]
                )
            ],
            "generation": "sequential Bayesian; later recommendations depend on measured prior candidates",
        }
        for seed in manifest["search_design"]["search_seeds"]
    ]
    return {
        "schema_version": 1,
        "status": "dry_run_validated_not_launched",
        "manifest": {
            "path": str(manifest_path.resolve()),
            "whole_file_sha256": manifest_file_sha256,
            "internal_manifest_sha256": manifest["manifest_sha256"],
        },
        "scope": copy.deepcopy(manifest["scope"]),
        "source_checks": local_checks,
        "runner_source": copy.deepcopy(local_checks["runner_source"]),
        "remote_checks": remote_checks,
        "search": {
            "parameters": parameters,
            "domains": copy.deepcopy(manifest["search_space"]["search_domains"]),
            "custom_param_ranges": custom_ranges,
            "packaged_schema_sha256": local_checks["train_schema"]["sha256"],
            "derived_schema_sha256": sha256_schema_value(derived_schema),
            "numeric_finite_domains_encoded_as_json_schema_enum": [
                path
                for path in parameters
                if manifest["search_space"]["search_domains"][path][
                    "representation"
                ]
                == "ordered_integer_levels"
            ],
            "reference_model_spec_sha256": sha256_value(
                manifest["search_space"]["reference_model_spec"]
            ),
            "base_train_spec_sha256": sha256_value(base_spec),
            "subarchives": recommendations,
            "total_candidate_budget": EXPECTED_TOTAL_CANDIDATES,
            "manual_candidate_injection_permitted": False,
        },
        "selector": selector,
        "post_front_matched_validation": {
            "contract": copy.deepcopy(
                manifest["post_front_matched_validation"]
            ),
            "contract_sha256": EXPECTED_POST_FRONT_CONTRACT_SHA256,
            "launched": False,
            "measurements_feed_reselection": False,
        },
        "execution": {
            "platform": "SLURM",
            "image": manifest["frozen_identity"]["runtime"]["sqsh_path"],
            "nodes_per_job": 1,
            "gpus_per_job": 8,
            "concurrent_seed_controllers": 3,
            "one_training_job_per_candidate": True,
            "accuracy_evaluation_per_successful_candidate": True,
            "stabilized_latency_job_per_successful_candidate": True,
            "selection_time_latency_measurements_preserved": True,
            "post_front_remeasurement_is_separate": True,
            "launched": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and render the launch contract without creating jobs (default).",
    )
    mode.add_argument(
        "--launch",
        action="store_true",
        help="Launch or resume all three sequential Bayesian subarchives.",
    )
    mode.add_argument(
        "--combine-only",
        action="store_true",
        help="Run the production selectors over the complete persisted archive.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--manifest-file-sha256",
        required=True,
        help="Exact SHA256 of the immutable expanded-search JSON file.",
    )
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--verify-remote",
        action="store_true",
        help="Verify SQSH, PTM, annotation hashes, and image directories over SSH.",
    )
    parser.add_argument(
        "--acknowledgement",
        default="",
        help="Exact authorization string required with --launch.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the exact persisted seed workspaces; valid only with --launch.",
    )
    return parser.parse_args()


def launch_all_seeds(
    manifest: dict[str, Any],
    manifest_path: Path,
    manifest_file_sha256: str,
    runtime_dir: Path,
    *,
    resume: bool,
) -> dict[int, int | None]:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    processes = {
        seed: context.Process(
            target=run_seed,
            args=(
                str(manifest_path),
                manifest_file_sha256,
                str(runtime_dir),
                seed,
                resume,
            ),
            name=f"dino-expanded-seed-{seed}",
        )
        for seed in manifest["search_design"]["search_seeds"]
    }
    for process in processes.values():
        process.start()

    def forward_signal(signum: int, _frame: object) -> None:
        for process in processes.values():
            if process.is_alive() and process.pid:
                os.kill(process.pid, signum)

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)
    exit_codes: dict[int, int | None] = {}
    while processes:
        for seed, process in list(processes.items()):
            process.join(timeout=1)
            if not process.is_alive():
                exit_codes[seed] = process.exitcode
                processes.pop(seed)
                print(
                    f"SEED_PROCESS_EXIT seed={seed} exitcode={process.exitcode}",
                    flush=True,
                )
        if processes:
            time.sleep(1)
    atomic_json(runtime_dir / "seed_process_status.json", {"exit_codes": exit_codes})
    return exit_codes


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    runtime_dir = args.runtime_dir.resolve()
    manifest, manifest_file_sha256 = load_manifest(
        manifest_path,
        supplied_file_sha256=args.manifest_file_sha256,
    )
    local_checks = validate_local_provenance(manifest, manifest_path)
    if args.launch:
        require_launch_source_ready(local_checks["runner_source"])

    remote_checks = None
    if args.verify_remote or args.launch:
        loaded_keys = load_env_file(
            Path(manifest["frozen_identity"]["runtime"]["secrets_env_path"])
        )
        remote_checks = verify_remote_contract(manifest)
        remote_checks["loaded_secret_keys"] = loaded_keys
        remote_checks["secret_values_recorded"] = False

    if args.combine_only:
        if args.resume:
            raise ContractError("--resume is valid only with --launch")
        result = combine_results(
            manifest,
            manifest_path,
            manifest_file_sha256,
            runtime_dir,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    report = dry_run_report(
        manifest,
        manifest_path,
        manifest_file_sha256,
        local_checks,
        remote_checks,
    )
    atomic_json(args.report.resolve(), report)
    if not args.launch:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "manifest_file_sha256": manifest_file_sha256,
                    "search_parameters": report["search"]["parameters"],
                    "total_candidate_budget": EXPECTED_TOTAL_CANDIDATES,
                    "report": str(args.report.resolve()),
                    "launched": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not args.verify_remote:
        raise ContractError("--launch requires --verify-remote")
    if args.acknowledgement != EXPECTED_ACKNOWLEDGEMENT:
        raise ContractError(
            "--launch requires the exact user-authorized acknowledgement"
        )
    exit_codes = launch_all_seeds(
        manifest,
        manifest_path,
        manifest_file_sha256,
        runtime_dir,
        resume=args.resume,
    )
    if not all(code == 0 for code in exit_codes.values()):
        return 1
    result = combine_results(
        manifest,
        manifest_path,
        manifest_file_sha256,
        runtime_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s",
    )
    try:
        raise SystemExit(main())
    except ContractError as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        raise SystemExit(2)
