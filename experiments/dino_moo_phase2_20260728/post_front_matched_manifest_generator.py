#!/usr/bin/env python3

"""Generate the immutable expanded-search post-front latency manifest.

The global Pareto front is independently recomputed by the manifest-pinned
``tao_automl`` selector from every successful objective record in the sealed
candidate table.  Generation fails unless that complete replay exactly matches
the combined-selection audit and each table audit.  Exact checkpoints and full
model records are then joined from the same independently hashed table.  No
measured value, winner identity, or candidate desirability participates in
scheduling.

This program only validates evidence and creates a manifest.  It has no launch
path and refuses to overwrite an existing manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_EXPANDED_MANIFEST = HERE / "expanded_search_manifest.v2.json"
DEFAULT_EXPANDED_RUNTIME = HERE / "runtime" / "expanded_search_v2"
DEFAULT_COMBINED_SELECTION = (
    DEFAULT_EXPANDED_RUNTIME / "expanded_combined_selection.json"
)
DEFAULT_CANDIDATE_TABLE = (
    DEFAULT_EXPANDED_RUNTIME / "expanded_candidate_table.json"
)
DEFAULT_INTEGRITY_AUDIT = (
    DEFAULT_EXPANDED_RUNTIME / "expanded_integrity_audit.json"
)
DEFAULT_PROTOCOL_ERRATUM = HERE / "phase2_protocol_erratum.v1.json"
DEFAULT_OUTPUT = HERE / "post_front_matched_manifest.v1.json"
EXPECTED_PROTOCOL_ERRATUM_FILE_SHA256 = (
    "95bba65099027459a50b5e74e43a4ab32c56057e534e70aa7f85bdc9246a7d13"
)
EXPECTED_PROTOCOL_ERRATUM_CANONICAL_SHA256 = (
    "e77b2a4a00cc99c687b3afe832de2295d8ca55a08cb157c0d9e5da7581ebce71"
)
EXPECTED_PROTOCOL_ERRATUM_ID = "dino_phase2_protocol_erratum_20260728_v1"
EXPECTED_PROTOCOL_ERRATUM_ISSUED_AT_UTC = "2026-07-28T06:36:41Z"
EXPECTED_POST_FRONT_CONTRACT_SHA256 = (
    "aba3a961bf50caf15803f271b59d7ffbd091414816d14f3deb793452f75ec281"
)
EXPECTED_EXPANDED_MANIFEST_FILE_SHA256 = (
    "9ac29e1aa07167a040d217fdab2d3cfdea0baad690dc95a70f2fe6715908793a"
)
EXPECTED_EXPANDED_MANIFEST_INTERNAL_SHA256 = (
    "910744ae2fead7e4e2e9a53fc672baef1ac43307e3979671b2b876fff422de96"
)
EXPECTED_PRACTICAL_TOLERANCE_MS = 0.73553775
EXPECTED_ALLOCATION_COUNT = 6
EXPECTED_GPU_COUNT = 8
EXPECTED_SEARCH_SEEDS = (314159, 271828, 161803)
EXPECTED_RECOMMENDATIONS_PER_SEED = 20
EXPECTED_TOTAL_CANDIDATES = 60
EXPECTED_TRAINING_SEED = 1234
EXPECTED_SEARCH_PARAMETERS = (
    "model.enc_layers",
    "model.dec_layers",
    "train.optim.lr",
    "train.optim.weight_decay",
)
TERMINAL_CANDIDATE_STATUSES = frozenset(
    {"success", "training_or_measurement_failure"}
)
EXPECTED_SCOPE = {
    "model_family": "DINO ResNet50",
    "dataset_uri": (
        "s3://nvcf-storage-handling/data/"
        "tao_od_synthetic_full_dino_coco/"
    ),
    "other_models_permitted": False,
    "other_datasets_permitted": False,
}
HEX = frozenset("0123456789abcdef")
TOOL_FILENAMES = (
    "post_front_matched_manifest_generator.py",
    "post_front_matched_launcher.py",
    "post_front_matched_block_runner.py",
    "post_front_matched_aggregator.py",
)
SELECTION_STACK_RELATIVE_PATHS = (
    "src/tao_automl/selection.py",
    "src/tao_automl/objectives.py",
    "src/tao_automl/utils/value_utils.py",
)


class ContractError(ValueError):
    """Raised when frozen post-front evidence violates the contract."""


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ContractError(
            f"{label} mismatch: expected {expected!r}, got {actual!r}"
        )


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX for character in value)
    ):
        raise ContractError(f"{label} must be a lowercase SHA256 digest")
    return value


def require_git_oid(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in (40, 64)
        or any(character not in HEX for character in value)
    ):
        raise ContractError(f"{label} must be a lowercase git object ID")
    return value


def finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ContractError(f"{label} must be finite")
    return float(value)


def _reject_duplicate_pairs(
    pairs: Iterable[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON constant is forbidden: {value}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return value


def load_exact_json(
    path: Path,
    supplied_sha256: str,
    label: str,
) -> tuple[dict[str, Any], str]:
    supplied = require_sha256(supplied_sha256, f"{label} supplied SHA256")
    try:
        actual = sha256_file(path)
    except OSError as error:
        raise ContractError(f"cannot load {label} from {path}: {error}") from error
    require_equal(actual, supplied, f"{label} whole-file SHA256")
    return load_json(path), actual


def validate_internal_digest(
    value: dict[str, Any],
    field: str,
    label: str,
) -> str:
    claimed = require_sha256(value.get(field), f"{label} {field}")
    unhashed = copy.deepcopy(value)
    del unhashed[field]
    require_equal(sha256_value(unhashed), claimed, f"{label} canonical digest")
    return claimed


def validate_protocol_erratum(
    erratum: dict[str, Any],
    whole_file_sha256: str,
) -> dict[str, Any]:
    """Validate the exact pre-data correction that governs post-front claims."""

    require_equal(
        whole_file_sha256,
        EXPECTED_PROTOCOL_ERRATUM_FILE_SHA256,
        "phase-2 protocol erratum whole-file SHA256",
    )
    require_equal(
        sha256_value(erratum),
        EXPECTED_PROTOCOL_ERRATUM_CANONICAL_SHA256,
        "phase-2 protocol erratum canonical SHA256",
    )
    require_equal(
        set(erratum),
        {
            "schema_version",
            "erratum_id",
            "status",
            "issued_at_utc",
            "scope",
            "source_pins",
            "issuance_state",
            "corrections",
            "invariants",
            "post_front_enforcement",
        },
        "phase-2 protocol erratum top-level schema",
    )
    require_equal(erratum.get("schema_version"), 1, "protocol erratum schema")
    require_equal(
        erratum.get("erratum_id"),
        EXPECTED_PROTOCOL_ERRATUM_ID,
        "protocol erratum ID",
    )
    require_equal(
        erratum.get("status"),
        "issued_before_expanded_selection_and_post_front_measurement",
        "protocol erratum issuance status",
    )
    require_equal(
        erratum.get("issued_at_utc"),
        EXPECTED_PROTOCOL_ERRATUM_ISSUED_AT_UTC,
        "protocol erratum issuance time",
    )
    require_equal(
        erratum.get("scope"),
        {
            **EXPECTED_SCOPE,
            "expanded_manifest_interpretation_only": True,
            "post_front_analysis_policy_only": True,
        },
        "protocol erratum scope",
    )
    require_equal(
        erratum.get("source_pins"),
        {
            "expanded_manifest": {
                "internal_manifest_sha256": (
                    EXPECTED_EXPANDED_MANIFEST_INTERNAL_SHA256
                ),
                "path": "expanded_search_manifest.v2.json",
                "whole_file_sha256": (
                    EXPECTED_EXPANDED_MANIFEST_FILE_SHA256
                ),
            },
            "frozen_post_front_contract": {
                "allocation_count": EXPECTED_ALLOCATION_COUNT,
                "canonical_sha256": EXPECTED_POST_FRONT_CONTRACT_SHA256,
                "json_pointer": "/post_front_matched_validation",
                "paired_bootstrap_confidence_level": 0.95,
                "paired_bootstrap_resamples": 10000,
                "paired_bootstrap_seed": 20260728,
                "practical_tolerance_ms": (
                    EXPECTED_PRACTICAL_TOLERANCE_MS
                ),
            },
            "production_selector": {
                "authority": "executed production behavior",
                "commit": "83d9d7ecc783724f674cb954f9fbb6c91ea8b0eb",
                "git_blob": "3533fd3e1751f9ffdb03abe1cb58b8739ba4bd7f",
                "relative_path": "src/tao_automl/selection.py",
                "repository_path": (
                    "/localhome/local-rarunachalam/tao-automl"
                ),
                "sha256": (
                    "7e787a18bca05464e0043367aee4f2c8cff3d93aef7f9e92"
                    "aaf88c47d255a532"
                ),
            },
        },
        "protocol erratum source pins",
    )
    require_equal(
        erratum.get("issuance_state"),
        {
            "absent_expanded_outputs": [
                "runtime/expanded_search_v2/expanded_combined_selection.json",
                "runtime/expanded_search_v2/expanded_candidate_table.json",
                "runtime/expanded_search_v2/expanded_candidate_table.csv",
                "runtime/expanded_search_v2/expanded_integrity_audit.json",
                "runtime/expanded_search_v2/expanded_completion.json",
            ],
            "candidate_objective_values_already_existed": True,
            "completed_union_selection_existed": False,
            "expanded_archive_complete": False,
            "expanded_runtime_path": "runtime/expanded_search_v2",
            "expected_candidate_count": EXPECTED_TOTAL_CANDIDATES,
            "final_global_pareto_front_known_or_used": False,
            "post_front": {
                "candidate_measurement_count": 0,
                "manifest_existed": False,
                "manifest_path": "post_front_matched_manifest.v1.json",
                "pairwise_comparison_count": 0,
                "slurm_allocation_count": 0,
                "tao_job_count": 0,
            },
            "successful_candidate_count": 15,
            "successful_candidate_count_by_search_seed": {
                "161803": 5,
                "271828": 5,
                "314159": 5,
            },
        },
        "protocol erratum issuance-state evidence",
    )
    require_equal(
        erratum.get("invariants"),
        {
            "archive_objective_value_changed": False,
            "expanded_manifest_mutation_permitted": False,
            "expanded_search_runner_change_required": False,
            "manual_candidate_addition_or_removal_permitted": False,
            "manual_winner_override_permitted": False,
            "post_front_measurements_feed_reselection": False,
            "post_front_winner_reselection_permitted": False,
            "search_budget_changed": False,
            "search_space_changed": False,
            "selection_configuration_changed": False,
            "selection_implementation_changed": False,
            "selection_time_measurement_replacement_permitted": False,
            "training_or_evaluation_changed": False,
        },
        "protocol erratum invariants",
    )
    require_equal(
        erratum.get("post_front_enforcement"),
        {
            "aggregation_must_validate_exact_erratum": True,
            "effective_directional_classification_controls_stable_ordering_claims": (
                True
            ),
            "effective_directional_classification_must_be_reported": True,
            "erratum_absence_or_drift_blocks_execution": True,
            "erratum_whole_file_sha256_must_be_pinned": True,
            "launch_must_validate_exact_erratum": True,
            "manifest_generation_must_validate_exact_erratum": True,
            "original_bootstrap_classification_must_be_reported": True,
        },
        "protocol erratum enforcement",
    )
    corrections = erratum.get("corrections")
    if not isinstance(corrections, dict):
        raise ContractError("protocol erratum corrections must be an object")
    require_equal(
        set(corrections),
        {
            "post_front_paired_classification",
            "selection_tie_break_documentation",
        },
        "protocol erratum correction set",
    )
    paired = corrections.get("post_front_paired_classification")
    if not isinstance(paired, dict):
        raise ContractError("protocol erratum paired classification is missing")
    require_equal(
        paired.get("both_policy_branches_must_be_emitted"),
        True,
        "protocol erratum paired-branch emission",
    )
    require_equal(
        paired.get("endpoints"),
        ["median_ms", "p95_ms"],
        "protocol erratum paired endpoints",
    )
    original = paired.get("original_preregistered_bootstrap_classification")
    effective = paired.get("effective_erratum_directional_classification")
    if not isinstance(original, dict) or not isinstance(effective, dict):
        raise ContractError(
            "protocol erratum must contain both paired classification branches"
        )
    require_equal(
        original.get("status"),
        "preserved_and_reported",
        "original preregistered classification status",
    )
    require_equal(
        original.get("point_classification_preserved"),
        True,
        "original preregistered classification preservation",
    )
    require_equal(
        {
            "status": effective.get("status"),
            "allocation_count": effective.get("allocation_count"),
            "permutation_count": effective.get("permutation_count"),
            "alpha": effective.get("alpha"),
            "bootstrap_role": effective.get("bootstrap_role"),
            "practical_tolerance_ms": effective.get(
                "practical_tolerance_ms"
            ),
            "scope": effective.get("scope"),
            "simultaneous_total_order_inference_permitted": effective.get(
                "simultaneous_total_order_inference_permitted"
            ),
        },
        {
            "status": "controls_directional_and_ordering_claims",
            "allocation_count": EXPECTED_ALLOCATION_COUNT,
            "permutation_count": 64,
            "alpha": 0.05,
            "bootstrap_role": "descriptive_only",
            "practical_tolerance_ms": EXPECTED_PRACTICAL_TOLERANCE_MS,
            "scope": "pairwise_only",
            "simultaneous_total_order_inference_permitted": False,
        },
        "effective erratum directional policy",
    )
    return {
        "original_preregistered_bootstrap_classification": copy.deepcopy(
            original
        ),
        "effective_erratum_directional_classification": copy.deepcopy(
            effective
        ),
        "both_policy_branches_must_be_emitted": True,
    }


def load_and_validate_protocol_erratum(
    path: Path,
    supplied_sha256: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    erratum, whole_file_sha256 = load_exact_json(
        path,
        supplied_sha256,
        "phase-2 protocol erratum",
    )
    policy = validate_protocol_erratum(erratum, whole_file_sha256)
    return erratum, whole_file_sha256, policy


def validate_expanded_manifest(
    manifest: dict[str, Any],
    whole_file_sha256: str,
) -> None:
    validate_internal_digest(
        manifest,
        "manifest_sha256",
        "expanded-search manifest",
    )
    require_equal(
        manifest.get("manifest_id"),
        "dino_expanded_search_20260728_v2",
        "expanded-search manifest ID",
    )
    require_equal(
        whole_file_sha256,
        EXPECTED_EXPANDED_MANIFEST_FILE_SHA256,
        "expanded-search v2 whole-file SHA256",
    )
    require_equal(
        manifest.get("manifest_sha256"),
        EXPECTED_EXPANDED_MANIFEST_INTERNAL_SHA256,
        "expanded-search v2 internal SHA256",
    )
    require_equal(
        manifest.get("status"),
        "preregistered_ready_to_launch",
        "expanded-search manifest status",
    )
    require_equal(manifest.get("scope"), EXPECTED_SCOPE, "DINO-only scope")
    require_equal(
        manifest.get("algorithm_only_selection_required"),
        True,
        "algorithm-only selection",
    )
    require_equal(
        manifest.get("manual_override_permitted"),
        False,
        "expanded-search manual override",
    )
    require_equal(
        {
            "target_runtime_path": manifest.get(
                "runtime_supersession", {}
            ).get("target_runtime_path"),
            "valid_objective_observations_reused": manifest.get(
                "runtime_supersession", {}
            ).get("valid_objective_observations_reused"),
            "v1_runtime_reused": manifest.get(
                "runtime_supersession", {}
            ).get("v1_runtime_reused"),
        },
        {
            "target_runtime_path": str(DEFAULT_EXPANDED_RUNTIME.resolve()),
            "valid_objective_observations_reused": 0,
            "v1_runtime_reused": False,
        },
        "expanded-search v2 runtime supersession",
    )
    require_equal(
        manifest["derivation"].get("post_front_contract_sha256"),
        EXPECTED_POST_FRONT_CONTRACT_SHA256,
        "expanded-search post-front contract",
    )
    contract = manifest.get("post_front_matched_validation")
    if not isinstance(contract, dict):
        raise ContractError("expanded-search manifest lacks post-front contract")
    require_equal(
        sha256_value(contract),
        EXPECTED_POST_FRONT_CONTRACT_SHA256,
        "post-front contract canonical SHA256",
    )
    require_equal(
        contract["allocation_design"]["allocation_count"],
        EXPECTED_ALLOCATION_COUNT,
        "post-front allocation count",
    )
    require_equal(
        contract["allocation_design"]["gpus_per_node"],
        EXPECTED_GPU_COUNT,
        "post-front GPU count",
    )
    require_equal(
        contract["ordering"]["algorithm"],
        "balanced_williams_rows_v1",
        "post-front ordering algorithm",
    )
    require_equal(
        contract["selection_isolation"],
        {
            "measurements_feed_reselection": False,
            "winner_reselection_permitted": False,
            "original_selection_time_measurements_replaced": False,
            "algorithm_selected_candidate_overridden": False,
            "allowed_use": "stability analysis and hypothesis verdict only",
        },
        "post-front selection isolation",
    )
    tolerance = finite_number(
        manifest["selection"]["latency_tolerance"]["value_ms"],
        "expanded-search latency tolerance",
    )
    require_equal(
        tolerance,
        EXPECTED_PRACTICAL_TOLERANCE_MS,
        "imported practical tolerance",
    )
    require_sha256(whole_file_sha256, "expanded-search whole-file SHA256")


def _git_text(repository: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ContractError(
            f"cannot inspect git provenance with {arguments!r}: {error}"
        ) from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ContractError(
            f"git provenance command failed {arguments!r}: {detail}"
        )
    return result.stdout.strip()


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ContractError(
            f"cannot inspect git source with {arguments!r}: {error}"
        ) from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode(
            "utf-8",
            errors="replace",
        ).strip()
        raise ContractError(
            f"git source command failed {arguments!r}: {detail}"
        )
    return result.stdout


def _load_selection_api(
    repository: Path,
) -> tuple[Any, Any, Any, Any]:
    """Load and path-check the production parser, selector, and normalizer."""

    source_root = (repository / "src").resolve()
    source_root_text = str(source_root)
    if source_root_text not in sys.path:
        sys.path.insert(0, source_root_text)
    try:
        from tao_automl.objectives import parse_objective_config
        from tao_automl.selection import (
            analyze_archive,
            canonical_spec_fingerprint,
        )
        from tao_automl.utils.value_utils import normalize_finite_number
    except ImportError as error:
        raise ContractError(
            f"cannot import the pinned tao_automl selection stack: {error}"
        ) from error

    expected_paths = {
        "parse_objective_config": (
            repository / "src" / "tao_automl" / "objectives.py"
        ).resolve(),
        "analyze_archive": (
            repository / "src" / "tao_automl" / "selection.py"
        ).resolve(),
        "canonical_spec_fingerprint": (
            repository / "src" / "tao_automl" / "selection.py"
        ).resolve(),
        "normalize_finite_number": (
            repository / "src" / "tao_automl" / "utils" / "value_utils.py"
        ).resolve(),
    }
    callables = {
        "parse_objective_config": parse_objective_config,
        "analyze_archive": analyze_archive,
        "canonical_spec_fingerprint": canonical_spec_fingerprint,
        "normalize_finite_number": normalize_finite_number,
    }
    for name, function in callables.items():
        source = inspect.getsourcefile(function)
        if not source:
            raise ContractError(f"cannot resolve source for {name}")
        require_equal(
            Path(source).resolve(),
            expected_paths[name],
            f"{name} import source",
        )
    return (
        parse_objective_config,
        analyze_archive,
        canonical_spec_fingerprint,
        normalize_finite_number,
    )


def selection_stack_provenance(
    expanded_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Prove the replay uses the manifest-pinned, unmodified selector stack."""

    repository_contract = expanded_manifest["frozen_identity"][
        "source_repositories"
    ]["tao_automl"]
    repository = Path(repository_contract["path"]).resolve()
    require_equal(
        repository,
        HERE.parent.parent.resolve(),
        "tao_automl selection repository",
    )
    branch = _git_text(repository, "branch", "--show-current")
    require_equal(
        branch,
        repository_contract["branch"],
        "tao_automl selection branch",
    )
    head_commit = require_git_oid(
        _git_text(repository, "rev-parse", "HEAD"),
        "tao_automl HEAD commit",
    )
    selection_core_commit = require_git_oid(
        repository_contract.get("selection_core_commit"),
        "selection-core commit",
    )
    require_equal(
        _git_text(
            repository,
            "cat-file",
            "-t",
            f"{selection_core_commit}^{{commit}}",
        ),
        "commit",
        "selection-core git object",
    )
    _git_text(
        repository,
        "merge-base",
        "--is-ancestor",
        selection_core_commit,
        head_commit,
    )

    files: dict[str, dict[str, Any]] = {}
    for relative_path in SELECTION_STACK_RELATIVE_PATHS:
        path = (repository / relative_path).resolve()
        if not path.is_file():
            raise ContractError(f"selection source is missing: {path}")
        require_equal(
            _git_text(
                repository,
                "ls-files",
                "--error-unmatch",
                "--",
                relative_path,
            ),
            relative_path,
            f"{relative_path} tracked path",
        )
        pinned_bytes = _git_bytes(
            repository,
            "show",
            f"{selection_core_commit}:{relative_path}",
        )
        current_bytes = path.read_bytes()
        pinned_sha256 = hashlib.sha256(pinned_bytes).hexdigest()
        current_sha256 = hashlib.sha256(current_bytes).hexdigest()
        require_equal(
            current_sha256,
            pinned_sha256,
            f"{relative_path} selection-core content",
        )
        head_blob = _git_text(
            repository,
            "rev-parse",
            f"HEAD:{relative_path}",
        )
        pinned_blob = _git_text(
            repository,
            "rev-parse",
            f"{selection_core_commit}:{relative_path}",
        )
        current_blob = _git_text(
            repository,
            "hash-object",
            str(path),
        )
        require_equal(
            current_blob,
            pinned_blob,
            f"{relative_path} working-tree git blob",
        )
        require_equal(
            head_blob,
            pinned_blob,
            f"{relative_path} HEAD git blob",
        )
        files[relative_path] = {
            "path": str(path),
            "relative_path": relative_path,
            "sha256": current_sha256,
            "selection_core_sha256": pinned_sha256,
            "git_blob": current_blob,
            "head_git_blob": head_blob,
            "selection_core_git_blob": pinned_blob,
            "tracked": True,
            "clean_against_selection_core": True,
        }

    (
        parse_objective_config,
        analyze_archive,
        canonical_spec_fingerprint,
        normalize_finite_number,
    ) = _load_selection_api(repository)
    callable_sources = {
        "parse_objective_config": Path(
            inspect.getsourcefile(parse_objective_config) or ""
        ).resolve(),
        "analyze_archive": Path(
            inspect.getsourcefile(analyze_archive) or ""
        ).resolve(),
        "canonical_spec_fingerprint": Path(
            inspect.getsourcefile(canonical_spec_fingerprint) or ""
        ).resolve(),
        "normalize_finite_number": Path(
            inspect.getsourcefile(normalize_finite_number) or ""
        ).resolve(),
    }
    callables = {
        name: {
            "path": str(source),
            "relative_path": source.relative_to(repository).as_posix(),
            "sha256": sha256_file(source),
        }
        for name, source in sorted(callable_sources.items())
    }
    return {
        "repository": str(repository),
        "branch": branch,
        "head_commit": head_commit,
        "commit_policy": "required_ancestor",
        "selection_core_commit": selection_core_commit,
        "selection_core_is_ancestor": True,
        "source_files": files,
        "callables": callables,
    }


def stable_selection_stack_projection(
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Return selector identity that remains stable after manifest commits."""

    return {
        "repository": provenance.get("repository"),
        "branch": provenance.get("branch"),
        "commit_policy": provenance.get("commit_policy"),
        "selection_core_commit": provenance.get("selection_core_commit"),
        "selection_core_is_ancestor": provenance.get(
            "selection_core_is_ancestor"
        ),
        "source_files": copy.deepcopy(provenance.get("source_files")),
        "callables": copy.deepcopy(provenance.get("callables")),
    }


def clean_head_source_provenance(
    repository: Path,
    path: Path,
) -> dict[str, Any]:
    """Return fail-closed git evidence for a launch-affecting source file."""

    repository = repository.resolve()
    path = path.resolve()
    try:
        relative_path = path.relative_to(repository).as_posix()
    except ValueError as error:
        raise ContractError(
            f"source {path} escaped repository {repository}"
        ) from error
    if not path.is_file():
        raise ContractError(f"source file is missing: {path}")
    require_equal(
        _git_text(
            repository,
            "ls-files",
            "--error-unmatch",
            "--",
            relative_path,
        ),
        relative_path,
        f"{relative_path} tracked path",
    )
    head_blob = require_git_oid(
        _git_text(repository, "rev-parse", f"HEAD:{relative_path}"),
        f"{relative_path} HEAD blob",
    )
    current_blob = require_git_oid(
        _git_text(repository, "hash-object", str(path)),
        f"{relative_path} working-tree blob",
    )
    require_equal(
        current_blob,
        head_blob,
        f"{relative_path} clean HEAD content",
    )
    return {
        "path": str(path),
        "relative_path": relative_path,
        "sha256": sha256_file(path),
        "git_blob": current_blob,
        "head_git_blob": head_blob,
        "tracked": True,
        "committed": True,
        "clean_against_head": True,
    }


def selector_settings(
    expanded_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct the exact settings used by expanded_search_runner.py."""

    design = expanded_manifest["search_design"]
    selection = expanded_manifest["selection"]
    seeds = design["search_seeds"]
    require_equal(
        seeds,
        [314159, 271828, 161803],
        "expanded selector seeds",
    )
    multi = selection["multi_objective_mode"]
    settings = {
        "algorithm": design["algorithm"],
        "automl_max_recommendations": design["recommendations_per_seed"],
        "automl_max_concurrent": 1,
        "session_id": f"dino_expanded_search_seed_{seeds[0]}",
        "experiment_id": f"dino_expanded_search_seed_{seeds[0]}",
        "random_seed": seeds[0],
        "selection_mode": "multi_objective",
        "objectives": [
            {"metric": "mAP50", "direction": "maximize", "weight": 1.0},
            {"metric": "latency_ms", "direction": "minimize", "weight": 1.0},
        ],
        "accuracy_metric": "mAP50",
        "latency_metric": "latency_ms",
        "latency_accuracy_retention": copy.deepcopy(
            selection["latency_mode"]["latency_accuracy_retention"]
        ),
        "multi_objective_min_accuracy": multi[
            "multi_objective_min_accuracy"
        ],
        "objective_normalization": "pareto_front",
        "augmentation_rho": multi["augmentation_rho"],
        "accuracy_tolerance": selection["accuracy_mode"][
            "accuracy_tolerance"
        ],
        "latency_tolerance": selection["latency_tolerance"]["value_ms"],
        "selection_score_tolerance": multi[
            "selection_score_tolerance"
        ],
        "latency_ci_low_metric": "latency_ci95_low",
        "latency_ci_high_metric": "latency_ci95_high",
        "require_eval_fn_success": True,
        "run_baseline": False,
        "run_final_evaluation": False,
        "automl_delete_intermediate_ckpt": True,
        "automl_checkpoint_retention_strategy": "terminal",
    }
    require_equal(
        settings["latency_accuracy_retention"],
        {
            "type": "relative",
            "retained_fraction": 0.98,
            "reference": "accuracy_winner",
        },
        "replay latency retention",
    )
    require_equal(
        settings["multi_objective_min_accuracy"],
        None,
        "replay multi-objective floor",
    )
    require_equal(
        settings["latency_tolerance"],
        EXPECTED_PRACTICAL_TOLERANCE_MS,
        "replay latency tolerance",
    )
    return settings


def _analysis_projection(analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "algorithm": copy.deepcopy(analysis.get("algorithm")),
        "selections": copy.deepcopy(analysis.get("selections")),
        "candidates": copy.deepcopy(analysis.get("candidates")),
    }


def replay_and_validate_selector(
    *,
    expanded_manifest: dict[str, Any],
    combined: dict[str, Any],
    table: dict[str, Any],
    integrity: dict[str, Any],
) -> dict[str, Any]:
    """Independently replay the production selector from every successful row."""

    provenance = selection_stack_provenance(expanded_manifest)
    parse_objective_config, analyze_archive, _, _ = _load_selection_api(
        Path(provenance["repository"])
    )
    settings = selector_settings(expanded_manifest)
    require_equal(
        integrity.get("selection", {}).get("settings"),
        settings,
        "integrity/replay selector settings",
    )
    try:
        objective = parse_objective_config(copy.deepcopy(settings))
    except (TypeError, ValueError) as error:
        raise ContractError(
            f"pinned objective parser rejected replay settings: {error}"
        ) from error
    config = objective.selection_config
    if config is None:
        raise ContractError("pinned objective parser did not build a selector")
    require_equal(
        config.to_dict(),
        combined.get("algorithm", {}).get("configuration"),
        "parsed/combined selector configuration",
    )
    require_equal(
        objective.to_dict()["objectives"],
        [
            {
                "metric": "mAP50",
                "direction": "maximize",
                "weight": 1.0,
                "scale": 1.0,
            },
            {
                "metric": "latency_ms",
                "direction": "minimize",
                "weight": 1.0,
                "scale": 1.0,
            },
        ],
        "parsed replay objective definitions",
    )

    rows = table.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ContractError("candidate table has no rows for selector replay")
    row_ids: set[str] = set()
    successful_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ContractError("candidate-table replay row is invalid")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ContractError("candidate-table replay row lacks candidate_id")
        if candidate_id in row_ids:
            raise ContractError(
                f"duplicate candidate-table replay row: {candidate_id}"
            )
        row_ids.add(candidate_id)
        if row.get("status") == "success":
            if not isinstance(row.get("specs"), dict):
                raise ContractError(
                    f"{candidate_id} successful row lacks specs"
                )
            if not isinstance(row.get("objective_values"), dict):
                raise ContractError(
                    f"{candidate_id} successful row lacks objective values"
                )
            successful_rows.append(row)
        elif row.get("selection_audit") is not None:
            raise ContractError(
                f"{candidate_id} unsuccessful row has a selection audit"
            )
    require_equal(
        len(successful_rows),
        table.get("successful_count"),
        "selector replay successful count",
    )
    require_equal(
        combined.get("search", {}).get("successful_candidates"),
        len(successful_rows),
        "combined selector successful count",
    )
    if not successful_rows:
        raise ContractError("selector replay has no successful candidates")

    def candidates(
        ordered_rows: Iterable[dict[str, Any]],
    ) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                id=row["candidate_id"],
                specs=copy.deepcopy(row["specs"]),
                status="success",
                objective_values=copy.deepcopy(row["objective_values"]),
            )
            for row in ordered_rows
        ]

    orderings = {
        "candidate_table_order": successful_rows,
        "reverse_candidate_table_order": list(reversed(successful_rows)),
        "candidate_id_order": sorted(
            successful_rows,
            key=lambda row: row["candidate_id"],
        ),
    }
    replay_analyses: dict[str, dict[str, Any]] = {}
    for name, ordered_rows in orderings.items():
        try:
            replay_analyses[name] = objective.analyze_archive(
                candidates(ordered_rows)
            ).to_dict()
        except (TypeError, ValueError, RuntimeError) as error:
            raise ContractError(
                f"pinned selector replay failed for {name}: {error}"
            ) from error
    reference = replay_analyses["candidate_table_order"]
    reference_projection = _analysis_projection(reference)
    for name, analysis in replay_analyses.items():
        require_equal(
            _analysis_projection(analysis),
            reference_projection,
            f"selector replay order independence ({name})",
        )

    direct = analyze_archive(
        candidates(successful_rows),
        config,
        accuracy_weight=1.0,
        latency_weight=1.0,
    ).to_dict()
    require_equal(
        _analysis_projection(direct),
        reference_projection,
        "objective-wrapper/direct-selector replay",
    )
    combined_projection = _analysis_projection(combined)
    require_equal(
        combined_projection,
        reference_projection,
        "recomputed/combined selection analysis",
    )

    replay_audits = reference["candidates"]
    if len(replay_audits) != len(successful_rows):
        raise ContractError(
            "selector replay did not audit every successful candidate"
        )
    replay_by_id = {
        audit["candidate_id"]: audit for audit in replay_audits
    }
    if len(replay_by_id) != len(replay_audits):
        raise ContractError("selector replay produced duplicate candidate audits")
    successful_ids = {row["candidate_id"] for row in successful_rows}
    require_equal(
        set(replay_by_id),
        successful_ids,
        "selector replay candidate population",
    )
    for row in successful_rows:
        candidate_id = row["candidate_id"]
        require_equal(
            replay_by_id[candidate_id].get("valid"),
            True,
            f"{candidate_id} replay validity",
        )
        require_equal(
            row.get("selection_audit"),
            replay_by_id[candidate_id],
            f"{candidate_id} table/replay selection audit",
        )

    replay_front, _ = global_front_audits(reference)
    combined_front, _ = global_front_audits(combined)
    replay_front_ids = [item["candidate_id"] for item in replay_front]
    combined_front_ids = [item["candidate_id"] for item in combined_front]
    require_equal(
        combined_front_ids,
        replay_front_ids,
        "recomputed/combined global rank-zero front",
    )

    selection_source = provenance["source_files"][
        "src/tao_automl/selection.py"
    ]
    authority = combined.get("selection_authority", {})
    require_equal(
        {
            "module": authority.get("module"),
            "function": authority.get("function"),
            "source_path": authority.get("source_path"),
            "source_sha256": authority.get("source_sha256"),
        },
        {
            "module": "tao_automl.selection",
            "function": "analyze_archive",
            "source_path": selection_source["path"],
            "source_sha256": selection_source["sha256"],
        },
        "combined selector source authority",
    )
    require_equal(
        integrity.get("selection", {}).get("authority"),
        authority,
        "integrity/combined selector authority",
    )

    proof = {
        "method": (
            "independent pinned production-selector replay over every "
            "successful candidate-table objective record"
        ),
        "selector_settings": copy.deepcopy(settings),
        "selector_settings_sha256": sha256_value(settings),
        "parsed_objective_config": objective.to_dict(),
        "parsed_objective_config_sha256": sha256_value(
            objective.to_dict()
        ),
        "successful_candidate_count": len(successful_rows),
        "successful_candidate_ids": sorted(successful_ids),
        "successful_candidate_set_sha256": sha256_value(
            sorted(successful_ids)
        ),
        "orderings_checked": list(orderings),
        "order_independent": True,
        "all_candidate_audits_exact_match": True,
        "candidate_table_audits_exact_match": True,
        "combined_analysis_exact_match": True,
        "global_rank_zero_front_exact_match": True,
        "global_rank_zero_candidate_ids": replay_front_ids,
        "global_rank_zero_candidate_set_sha256": sha256_value(
            replay_front_ids
        ),
        "recomputed_analysis_sha256": sha256_value(reference_projection),
        "combined_analysis_sha256": sha256_value(combined_projection),
        "selection_stack_content_sha256": sha256_value(
            stable_selection_stack_projection(provenance)
        ),
    }
    return {
        "analysis": reference,
        "proof": proof,
        "selection_stack": provenance,
    }


def global_front_audits(
    combined: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    audits = combined.get("candidates")
    if not isinstance(audits, list) or not audits:
        raise ContractError("combined selection candidates must be non-empty")
    by_id: dict[str, dict[str, Any]] = {}
    for audit in audits:
        if not isinstance(audit, dict):
            raise ContractError("combined selection candidate audit is invalid")
        candidate_id = audit.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ContractError("candidate audit lacks candidate_id")
        if candidate_id in by_id:
            raise ContractError(f"duplicate candidate audit: {candidate_id}")
        by_id[candidate_id] = audit
    front = sorted(
        (
            audit
            for audit in audits
            if audit.get("valid") is True and audit.get("pareto_rank") == 0
        ),
        key=lambda item: item["candidate_id"],
    )
    if not front:
        raise ContractError("global Pareto front is empty")
    for audit in front:
        require_equal(
            audit.get("dominated_by"),
            [],
            f"{audit['candidate_id']} rank-zero dominated_by",
        )
    return front, by_id


def validate_integrity_bindings(
    *,
    expanded_manifest_path: Path,
    expanded_manifest_sha256: str,
    combined_path: Path,
    combined_sha256: str,
    table_path: Path,
    table_sha256: str,
    integrity: dict[str, Any],
    combined: dict[str, Any],
    table: dict[str, Any],
) -> None:
    require_equal(
        integrity.get("schema_version"),
        1,
        "expanded integrity schema",
    )
    require_equal(
        integrity.get("scope"),
        EXPECTED_SCOPE,
        "expanded integrity scope",
    )
    require_equal(
        integrity["manifest"].get("path"),
        str(expanded_manifest_path.resolve()),
        "integrity expanded manifest path",
    )
    require_equal(
        integrity["manifest"].get("whole_file_sha256"),
        expanded_manifest_sha256,
        "integrity expanded manifest SHA256",
    )
    require_equal(
        integrity["manifest"].get("internal_manifest_sha256"),
        EXPECTED_EXPANDED_MANIFEST_INTERNAL_SHA256,
        "integrity expanded manifest internal SHA256",
    )
    candidate_budget = integrity.get("candidate_budget", {})
    require_equal(candidate_budget.get("expected"), 60, "integrity budget")
    require_equal(candidate_budget.get("observed"), 60, "integrity observed")
    require_equal(
        candidate_budget.get("successful"),
        table.get("successful_count"),
        "integrity successful count",
    )
    require_equal(
        candidate_budget.get("failed"),
        60 - table.get("successful_count", -1),
        "integrity failed count",
    )
    selection = integrity.get("selection", {})
    require_equal(selection.get("manual_override_used"), False, "manual override")
    require_equal(selection.get("algorithm_only"), True, "algorithm-only audit")
    require_equal(
        integrity.get("selection_time_measurements_preserved"),
        True,
        "selection-time measurements preserved",
    )
    require_equal(
        integrity.get("post_selection_measurements_feed_selection"),
        False,
        "post-selection measurements feed selection",
    )
    post = integrity.get("post_front_matched_validation", {})
    require_equal(
        post.get("contract_sha256"),
        EXPECTED_POST_FRONT_CONTRACT_SHA256,
        "integrity post-front contract",
    )
    require_equal(
        post.get("measurements_feed_reselection"),
        False,
        "integrity post-front reselection",
    )
    require_equal(
        integrity.get("selection", {}).get("selected_candidate_ids"),
        {
            mode: selection["winner_id"]
            for mode, selection in combined["selections"].items()
        },
        "integrity selected candidate IDs",
    )
    artifacts = integrity.get("artifacts", {})
    require_equal(
        Path(artifacts.get("combined_selection", "")).resolve(),
        combined_path.resolve(),
        "integrity combined-selection path",
    )
    require_equal(
        artifacts.get("combined_selection_sha256"),
        combined_sha256,
        "integrity combined-selection SHA256",
    )
    require_equal(
        Path(artifacts.get("candidate_table_json", "")).resolve(),
        table_path.resolve(),
        "integrity candidate-table path",
    )
    require_equal(
        artifacts.get("candidate_table_json_sha256"),
        table_sha256,
        "integrity candidate-table SHA256",
    )


def candidate_table_projection(
    record: dict[str, Any],
    selection_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    """Reconstruct exactly the JSON-row projection emitted by the runner."""

    return {
        "candidate_id": record.get("candidate_id"),
        "search_seed": record.get("search_seed"),
        "training_seed": record.get("training_seed"),
        "rec_id": record.get("rec_id"),
        "status": record.get("status"),
        "specs": copy.deepcopy(record.get("specs")),
        "resolved_model_spec": copy.deepcopy(
            record.get("resolved_model_spec")
        ),
        "resolved_model_spec_sha256": record.get(
            "resolved_model_spec_sha256"
        ),
        "checkpoint": copy.deepcopy(record.get("checkpoint")),
        "train_job_id": record.get("train_job_id"),
        "training_runtime": copy.deepcopy(record.get("training_runtime")),
        "accuracy_evaluation": copy.deepcopy(
            record.get("accuracy_evaluation")
        ),
        "selection_time_latency": copy.deepcopy(
            record.get("selection_time_latency")
        ),
        "objective_values": copy.deepcopy(record.get("objective_values")),
        "selection_audit": copy.deepcopy(selection_audit),
        "failure_reason": record.get("failure_reason"),
    }


def candidate_table_csv_bytes(
    records_by_id: dict[str, dict[str, Any]],
    audits_by_id: dict[str, dict[str, Any]],
) -> bytes:
    """Reconstruct the runner's exact immutable candidate-table CSV bytes."""

    columns = [
        "candidate_id",
        "search_seed",
        "training_seed",
        "rec_id",
        "status",
        *EXPECTED_SEARCH_PARAMETERS,
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
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=columns,
        lineterminator="\n",
    )
    writer.writeheader()
    for candidate_id in sorted(records_by_id):
        record = records_by_id[candidate_id]
        specs = record.get("specs", {})
        values = record.get("objective_values", {})
        audit = audits_by_id.get(candidate_id) or {}
        winner = audit.get("winner", {})
        checkpoint = record.get("checkpoint", {})
        training_runtime = record.get("training_runtime", {})
        accuracy_job = record.get("accuracy_evaluation", {})
        latency_job = record.get("selection_time_latency", {})
        writer.writerow(
            {
                "candidate_id": candidate_id,
                "search_seed": record.get("search_seed"),
                "training_seed": record.get("training_seed"),
                "rec_id": record.get("rec_id"),
                "status": record.get("status"),
                **{
                    path: specs.get(path)
                    for path in EXPECTED_SEARCH_PARAMETERS
                },
                **{
                    name: values.get(name)
                    for name in (
                        "mAP50",
                        "latency_ms",
                        "latency_p95_ms",
                        "latency_mad_ms",
                        "latency_iqr_ms",
                        "latency_robust_cv",
                        "latency_ci95_low",
                        "latency_ci95_high",
                    )
                },
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
        )
    return stream.getvalue().encode("utf-8")


def _validate_seed_archive_record(
    record: dict[str, Any],
    *,
    candidate_id: str,
    search_seed: int,
    rec_id: int,
    expanded_manifest_sha256: str,
) -> None:
    if not isinstance(record, dict):
        raise ContractError(f"{candidate_id} seed-archive record is invalid")
    require_equal(
        record.get("candidate_id"),
        candidate_id,
        f"{candidate_id} archive identity",
    )
    require_equal(
        record.get("search_seed"),
        search_seed,
        f"{candidate_id} archive search seed",
    )
    require_equal(
        record.get("training_seed"),
        EXPECTED_TRAINING_SEED,
        f"{candidate_id} training seed",
    )
    require_equal(
        record.get("rec_id"),
        rec_id,
        f"{candidate_id} recommendation ID",
    )
    require_equal(
        record.get("search_manifest_file_sha256"),
        expanded_manifest_sha256,
        f"{candidate_id} expanded-manifest binding",
    )
    require_equal(
        record.get("manual_candidate_injection_used"),
        False,
        f"{candidate_id} manual candidate injection",
    )
    status = record.get("status")
    if status not in TERMINAL_CANDIDATE_STATUSES:
        raise ContractError(
            f"{candidate_id} is not terminal in seed archive: {status!r}"
        )
    if status == "success":
        if not isinstance(record.get("objective_values"), dict):
            raise ContractError(
                f"{candidate_id} successful archive record lacks objectives"
            )
        model = record.get("resolved_model_spec")
        if not isinstance(model, dict):
            raise ContractError(
                f"{candidate_id} successful archive record lacks model spec"
            )
        model_sha256 = require_sha256(
            record.get("resolved_model_spec_sha256"),
            f"{candidate_id} model-spec SHA256",
        )
        require_equal(
            sha256_value(model),
            model_sha256,
            f"{candidate_id} model-spec digest",
        )


def validate_expanded_archive_authority(
    *,
    seed_archive_paths: Iterable[Path],
    expanded_manifest_sha256: str,
    combined: dict[str, Any],
    combined_sha256: str,
    table: dict[str, Any],
    table_sha256: str,
    candidate_table_csv_path: Path,
    candidate_table_csv_sha256: str,
    integrity_sha256: str,
) -> dict[str, Any]:
    """Bind the final outputs to the exact immutable three-seed record union."""

    for digest, label in (
        (expanded_manifest_sha256, "expanded manifest"),
        (combined_sha256, "combined selection"),
        (table_sha256, "candidate table JSON"),
        (candidate_table_csv_sha256, "candidate table CSV"),
        (integrity_sha256, "integrity audit"),
    ):
        require_sha256(digest, f"{label} SHA256")
    csv_path = candidate_table_csv_path.resolve()
    try:
        csv_actual_bytes = csv_path.read_bytes()
    except OSError as error:
        raise ContractError(
            f"cannot load expanded candidate-table CSV from {csv_path}: "
            f"{error}"
        ) from error
    csv_actual_sha256 = hashlib.sha256(csv_actual_bytes).hexdigest()
    require_equal(
        csv_actual_sha256,
        candidate_table_csv_sha256,
        "expanded candidate-table CSV whole-file SHA256",
    )

    raw_paths = [Path(path).resolve() for path in seed_archive_paths]
    if len(raw_paths) != len(EXPECTED_SEARCH_SEEDS):
        raise ContractError("exactly three seed archives are required")
    if len(set(raw_paths)) != len(raw_paths):
        raise ContractError("seed archive paths must be unique")

    records_by_id: dict[str, dict[str, Any]] = {}
    seed_evidence_by_seed: dict[int, dict[str, Any]] = {}
    for path in raw_paths:
        if path.name != "seed_archive.v1.json":
            raise ContractError(
                f"unexpected seed archive basename: {path.name}"
            )
        archive = load_json(path)
        require_equal(
            set(archive),
            {
                "schema_version",
                "status",
                "created_at_utc",
                "manifest_file_sha256",
                "search_seed",
                "recommendations",
                "manual_candidate_injection_used",
                "records",
                "automl_result_sha256",
                "archive_sha256",
            },
            f"{path} seed archive schema",
        )
        require_equal(archive.get("schema_version"), 1, f"{path} schema")
        require_equal(archive.get("status"), "complete", f"{path} status")
        require_equal(
            archive.get("manifest_file_sha256"),
            expanded_manifest_sha256,
            f"{path} expanded-manifest SHA256",
        )
        seed = archive.get("search_seed")
        if seed not in EXPECTED_SEARCH_SEEDS:
            raise ContractError(f"{path} has unexpected search seed: {seed!r}")
        if seed in seed_evidence_by_seed:
            raise ContractError(f"duplicate seed archive for search seed {seed}")
        require_equal(
            path.parent.name,
            f"seed_{seed}",
            f"{path} seed directory",
        )
        require_equal(
            archive.get("recommendations"),
            EXPECTED_RECOMMENDATIONS_PER_SEED,
            f"seed {seed} recommendation count",
        )
        require_equal(
            archive.get("manual_candidate_injection_used"),
            False,
            f"seed {seed} manual candidate injection",
        )
        require_sha256(
            archive.get("automl_result_sha256"),
            f"seed {seed} AutoML-result SHA256",
        )
        internal_sha256 = validate_internal_digest(
            archive,
            "archive_sha256",
            f"seed {seed} archive",
        )
        whole_file_sha256 = sha256_file(path)
        records = archive.get("records")
        if not isinstance(records, dict):
            raise ContractError(f"seed {seed} records must be a mapping")
        expected_ids = {
            f"seed_{seed}_rec_{rec_id}"
            for rec_id in range(EXPECTED_RECOMMENDATIONS_PER_SEED)
        }
        require_equal(
            set(records),
            expected_ids,
            f"seed {seed} exact candidate IDs",
        )
        ordered_ids = sorted(records)
        ordered_records: list[dict[str, Any]] = []
        successful_count = 0
        for candidate_id in ordered_ids:
            record = records[candidate_id]
            rec_id = int(candidate_id.rsplit("_", 1)[1])
            _validate_seed_archive_record(
                record,
                candidate_id=candidate_id,
                search_seed=seed,
                rec_id=rec_id,
                expanded_manifest_sha256=expanded_manifest_sha256,
            )
            if candidate_id in records_by_id:
                raise ContractError(
                    f"duplicate candidate across seed archives: {candidate_id}"
                )
            records_by_id[candidate_id] = copy.deepcopy(record)
            ordered_records.append(copy.deepcopy(record))
            successful_count += record.get("status") == "success"
        seed_evidence_by_seed[seed] = {
            "path": str(path),
            "whole_file_sha256": whole_file_sha256,
            "internal_archive_sha256": internal_sha256,
            "search_seed": seed,
            "record_count": len(ordered_records),
            "terminal_record_count": len(ordered_records),
            "successful_record_count": successful_count,
            "failed_record_count": len(ordered_records) - successful_count,
            "candidate_ids_sha256": sha256_value(ordered_ids),
            "full_records_sha256": sha256_value(ordered_records),
        }

    require_equal(
        set(seed_evidence_by_seed),
        set(EXPECTED_SEARCH_SEEDS),
        "exact expanded search-seed archive population",
    )
    candidate_ids = sorted(records_by_id)
    require_equal(
        len(candidate_ids),
        EXPECTED_TOTAL_CANDIDATES,
        "expanded archive union candidate count",
    )
    expected_union_ids = {
        f"seed_{seed}_rec_{rec_id}"
        for seed in EXPECTED_SEARCH_SEEDS
        for rec_id in range(EXPECTED_RECOMMENDATIONS_PER_SEED)
    }
    require_equal(
        set(candidate_ids),
        expected_union_ids,
        "expanded archive exact union candidate IDs",
    )

    audits = combined.get("candidates")
    if not isinstance(audits, list):
        raise ContractError("combined candidate audits must be a list")
    audits_by_id: dict[str, dict[str, Any]] = {}
    for audit in audits:
        if not isinstance(audit, dict):
            raise ContractError("combined candidate audit is invalid")
        candidate_id = audit.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ContractError("combined candidate audit lacks candidate ID")
        if candidate_id in audits_by_id:
            raise ContractError(
                f"duplicate combined candidate audit: {candidate_id}"
            )
        audits_by_id[candidate_id] = audit

    rows = table.get("rows")
    if not isinstance(rows, list):
        raise ContractError("expanded candidate table rows must be a list")
    require_equal(
        table.get("schema_version"),
        1,
        "expanded candidate-table schema",
    )
    require_equal(
        table.get("candidate_count"),
        EXPECTED_TOTAL_CANDIDATES,
        "expanded candidate-table count",
    )
    require_equal(
        table.get("manual_candidate_injection_used"),
        False,
        "expanded candidate-table manual injection",
    )
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ContractError("expanded candidate-table row is invalid")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ContractError("expanded candidate-table row lacks candidate ID")
        if candidate_id in rows_by_id:
            raise ContractError(
                f"duplicate expanded candidate-table row: {candidate_id}"
            )
        rows_by_id[candidate_id] = row
    require_equal(
        set(rows_by_id),
        set(candidate_ids),
        "seed-archive/candidate-table candidate population",
    )
    successful_ids = {
        candidate_id
        for candidate_id, record in records_by_id.items()
        if record.get("status") == "success"
    }
    require_equal(
        set(audits_by_id),
        successful_ids,
        "seed-archive/combined successful candidate population",
    )
    require_equal(
        table.get("successful_count"),
        len(successful_ids),
        "expanded candidate-table successful count",
    )
    expected_rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        expected = candidate_table_projection(
            records_by_id[candidate_id],
            audits_by_id.get(candidate_id),
        )
        require_equal(
            rows_by_id[candidate_id],
            expected,
            f"{candidate_id} seed-archive/candidate-table projection",
        )
        expected_rows.append(expected)
    require_equal(
        csv_actual_bytes,
        candidate_table_csv_bytes(records_by_id, audits_by_id),
        "expanded candidate-table CSV exact archive projection",
    )

    full_record_union = [
        copy.deepcopy(records_by_id[candidate_id])
        for candidate_id in candidate_ids
    ]
    return {
        "search_seeds": list(EXPECTED_SEARCH_SEEDS),
        "recommendations_per_seed": EXPECTED_RECOMMENDATIONS_PER_SEED,
        "candidate_count": EXPECTED_TOTAL_CANDIDATES,
        "terminal_candidate_count": EXPECTED_TOTAL_CANDIDATES,
        "successful_candidate_count": len(successful_ids),
        "failed_candidate_count": (
            EXPECTED_TOTAL_CANDIDATES - len(successful_ids)
        ),
        "manual_candidate_injection_used": False,
        "canonical_order": "ascending UTF-8 candidate_id",
        "candidate_ids": candidate_ids,
        "candidate_ids_sha256": sha256_value(candidate_ids),
        "full_record_union_sha256": sha256_value(full_record_union),
        "candidate_table_projection_sha256": sha256_value(expected_rows),
        "expanded_combined_selection_sha256": combined_sha256,
        "expanded_candidate_table_sha256": table_sha256,
        "expanded_candidate_table_csv_sha256": (
            candidate_table_csv_sha256
        ),
        "expanded_integrity_audit_sha256": integrity_sha256,
        "seed_archives": [
            seed_evidence_by_seed[seed]
            for seed in EXPECTED_SEARCH_SEEDS
        ],
    }


def validate_completed_archive(
    combined: dict[str, Any],
    table: dict[str, Any],
    *,
    expanded_manifest: dict[str, Any],
    expanded_manifest_path: Path,
    expanded_manifest_sha256: str,
    integrity: dict[str, Any],
) -> dict[str, Any]:
    """Prove the selector and retained table represent the sealed 60-point run."""

    require_equal(
        combined.get("manifest"),
        {
            "path": str(expanded_manifest_path.resolve()),
            "whole_file_sha256": expanded_manifest_sha256,
            "internal_manifest_sha256": (
                EXPECTED_EXPANDED_MANIFEST_INTERNAL_SHA256
            ),
        },
        "combined expanded-manifest identity",
    )
    search = combined.get("search", {})
    require_equal(search.get("algorithm"), "bayesian", "combined algorithm")
    require_equal(
        search.get("seeds"),
        [314159, 271828, 161803],
        "combined search seeds",
    )
    require_equal(
        search.get("recommendations_per_seed"),
        20,
        "combined per-seed budget",
    )
    require_equal(
        search.get("total_candidate_records"),
        60,
        "combined total candidate records",
    )
    require_equal(
        search.get("all_modes_receive_identical_archive"),
        True,
        "combined shared archive",
    )
    authority = combined.get("selection_authority", {})
    require_equal(
        {
            "module": authority.get("module"),
            "function": authority.get("function"),
            "manual_override_used": authority.get("manual_override_used"),
            "candidate_reordering_used": authority.get(
                "candidate_reordering_used"
            ),
            "order_independent": authority.get(
                "order_independence_audit", {}
            ).get("order_independent"),
        },
        {
            "module": "tao_automl.selection",
            "function": "analyze_archive",
            "manual_override_used": False,
            "candidate_reordering_used": False,
            "order_independent": True,
        },
        "combined selection authority",
    )
    audits = combined.get("candidates")
    if not isinstance(audits, list) or not audits:
        raise ContractError("combined selector candidate set is empty")
    if any(not isinstance(item, dict) for item in audits):
        raise ContractError("combined selector candidate audit is invalid")
    audit_ids = [item.get("candidate_id") for item in audits]
    if (
        any(not isinstance(item, str) or not item for item in audit_ids)
        or len(set(audit_ids)) != len(audit_ids)
    ):
        raise ContractError("combined selector candidate IDs are invalid")

    require_equal(table.get("schema_version"), 1, "candidate-table schema")
    require_equal(
        table.get("manual_candidate_injection_used"),
        False,
        "candidate-table manual injection",
    )
    rows = table.get("rows")
    if not isinstance(rows, list) or len(rows) != 60:
        raise ContractError("candidate table must contain all 60 records")
    if any(not isinstance(row, dict) for row in rows):
        raise ContractError("candidate table contains a non-object row")
    require_equal(table.get("candidate_count"), 60, "candidate-table count")
    successful_ids = [
        row.get("candidate_id")
        for row in rows
        if row.get("status") == "success"
    ]
    require_equal(
        table.get("successful_count"),
        len(successful_ids),
        "candidate-table successful count",
    )
    require_equal(
        set(successful_ids),
        set(audit_ids),
        "selector/table successful candidate population",
    )
    if len(successful_ids) != len(set(successful_ids)):
        raise ContractError("candidate table has duplicate successful IDs")
    for mode in ("accuracy", "latency", "multi_objective"):
        winner_id = combined.get("selections", {}).get(mode, {}).get(
            "winner_id"
        )
        if winner_id not in set(audit_ids):
            raise ContractError(f"{mode} winner is outside the sealed archive")
    return replay_and_validate_selector(
        expanded_manifest=expanded_manifest,
        combined=combined,
        table=table,
        integrity=integrity,
    )


def derive_candidate_records(
    combined: dict[str, Any],
    table: dict[str, Any],
    selector_replay: dict[str, Any],
) -> list[dict[str, Any]]:
    replay_analysis = selector_replay.get("analysis")
    if not isinstance(replay_analysis, dict):
        raise ContractError("independent selector replay analysis is missing")
    front, _ = global_front_audits(replay_analysis)
    combined_front, _ = global_front_audits(combined)
    require_equal(
        front,
        combined_front,
        "derived replay/combined global front audits",
    )
    rows = table.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ContractError("expanded candidate table rows must be non-empty")
    require_equal(
        table.get("candidate_count"),
        len(rows),
        "expanded candidate-table row count",
    )
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ContractError("expanded candidate-table row is invalid")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ContractError("candidate-table row lacks candidate_id")
        if candidate_id in by_id:
            raise ContractError(f"duplicate candidate-table row: {candidate_id}")
        by_id[candidate_id] = row

    records: list[dict[str, Any]] = []
    for audit in front:
        candidate_id = audit["candidate_id"]
        row = by_id.get(candidate_id)
        if row is None:
            raise ContractError(
                f"global rank-zero candidate lacks retained record: {candidate_id}"
            )
        require_equal(
            row.get("status"),
            "success",
            f"{candidate_id} retained status",
        )
        require_equal(
            row.get("selection_audit"),
            audit,
            f"{candidate_id} exact selection audit",
        )
        model = row.get("resolved_model_spec")
        if not isinstance(model, dict) or not model:
            raise ContractError(f"{candidate_id} lacks full model mapping")
        model_sha256 = require_sha256(
            row.get("resolved_model_spec_sha256"),
            f"{candidate_id} resolved model SHA256",
        )
        require_equal(
            sha256_value(model),
            model_sha256,
            f"{candidate_id} full model mapping",
        )
        checkpoint = row.get("checkpoint")
        if not isinstance(checkpoint, dict):
            raise ContractError(f"{candidate_id} lacks retained checkpoint")
        checkpoint_path = checkpoint.get("path")
        if (
            not isinstance(checkpoint_path, str)
            or not checkpoint_path
            or not Path(checkpoint_path).is_absolute()
        ):
            raise ContractError(
                f"{candidate_id} checkpoint path must be non-empty absolute"
            )
        checkpoint_sha256 = require_sha256(
            checkpoint.get("sha256"),
            f"{candidate_id} checkpoint SHA256",
        )
        objectives = row.get("objective_values")
        if not isinstance(objectives, dict):
            raise ContractError(f"{candidate_id} lacks objective values")
        for metric in (
            "mAP50",
            "latency_ms",
            "latency_p95_ms",
            "latency_ci95_low",
            "latency_ci95_high",
        ):
            finite_number(
                objectives.get(metric),
                f"{candidate_id}.{metric}",
            )
        record = {
            "candidate_id": candidate_id,
            "candidate_table_record_sha256": sha256_value(row),
            "selection_audit_sha256": sha256_value(audit),
            "global_pareto_rank": 0,
            "global_dominated_by": [],
            "search_seed": row.get("search_seed"),
            "training_seed": row.get("training_seed"),
            "rec_id": row.get("rec_id"),
            "train_job_id": row.get("train_job_id"),
            "specs": copy.deepcopy(row.get("specs")),
            "selection_time_objective_values": copy.deepcopy(objectives),
            "resolved_model_spec": copy.deepcopy(model),
            "resolved_model_spec_sha256": model_sha256,
            "checkpoint": {
                "path": checkpoint_path,
                "sha256": checkpoint_sha256,
            },
        }
        records.append(record)
    require_equal(
        [item["candidate_id"] for item in records],
        sorted(item["candidate_id"] for item in records),
        "canonical candidate order",
    )
    return records


def williams_base_row(candidate_count: int) -> list[int]:
    if isinstance(candidate_count, bool) or candidate_count <= 0:
        raise ContractError("candidate count must be a positive integer")
    row = [0]
    low = 1
    high = candidate_count - 1
    while low <= high:
        row.append(low)
        low += 1
        if low <= high:
            row.append(high)
            high -= 1
    require_equal(
        sorted(row),
        list(range(candidate_count)),
        "Williams base-row permutation",
    )
    return row


def williams_design_rows(candidate_count: int) -> list[list[int]]:
    base = williams_base_row(candidate_count)
    rows = [
        [(index + shift) % candidate_count for index in base]
        for shift in range(candidate_count)
    ]
    if candidate_count % 2 == 1:
        rows.extend(list(reversed(row)) for row in rows[:candidate_count])
    return rows


def build_schedule(candidate_ids: list[str]) -> dict[str, Any]:
    canonical_ids = sorted(candidate_ids)
    if not canonical_ids or len(set(canonical_ids)) != len(canonical_ids):
        raise ContractError("schedule candidate IDs must be unique and non-empty")
    require_equal(candidate_ids, canonical_ids, "schedule canonical candidate IDs")
    design_rows = williams_design_rows(len(canonical_ids))
    design_count = len(design_rows)
    # This exact projection was frozen before the expanded search produced
    # results.  Do not optimize or adapt it after observing the front.
    selected_indices = [
        (allocation_index * design_count) // EXPECTED_ALLOCATION_COUNT
        for allocation_index in range(EXPECTED_ALLOCATION_COUNT)
    ]
    allocations = []
    position_counts = {
        candidate_id: Counter() for candidate_id in canonical_ids
    }
    adjacency_counts: Counter[tuple[str, str]] = Counter()
    complete_permutations: list[bool] = []
    for allocation_index, design_index in enumerate(selected_indices):
        order = [canonical_ids[index] for index in design_rows[design_index]]
        complete_permutations.append(
            len(order) == len(canonical_ids)
            and len(set(order)) == len(canonical_ids)
            and set(order) == set(canonical_ids)
        )
        allocation = {
            "allocation_id": f"post_front_allocation_{allocation_index:02d}",
            "allocation_index": allocation_index,
            "design_row_index": design_index,
            "candidate_order": order,
        }
        allocations.append(allocation)
        for position, candidate_id in enumerate(order):
            position_counts[candidate_id][position] += 1
        adjacency_counts.update(zip(order, order[1:]))
    every_allocation_is_complete_permutation = all(complete_permutations)
    require_equal(
        every_allocation_is_complete_permutation,
        True,
        "Williams allocation completeness",
    )
    position_count_imbalances = {
        candidate_id: (
            max(counts.get(position, 0) for position in range(len(canonical_ids)))
            - min(
                counts.get(position, 0)
                for position in range(len(canonical_ids))
            )
        )
        for candidate_id, counts in position_counts.items()
    }
    maximum_position_count_imbalance = max(
        position_count_imbalances.values(),
        default=0,
    )
    position_count_imbalance_within_one = (
        maximum_position_count_imbalance <= 1
    )
    schedule_payload = {
        "algorithm": "balanced_williams_rows_v1",
        "row_selection_rule": (
            "allocation k uses design row floor(k*R/6), where R is the "
            "complete Williams design-row count"
        ),
        "canonical_candidate_ids": canonical_ids,
        "candidate_count": len(canonical_ids),
        "base_row_indices": williams_base_row(len(canonical_ids)),
        "design_row_count": design_count,
        "selected_design_row_indices": selected_indices,
        "allocations": allocations,
        "audit": {
            "allocation_count": EXPECTED_ALLOCATION_COUNT,
            "allocation_complete_permutation_flags": complete_permutations,
            "every_allocation_is_complete_permutation": (
                every_allocation_is_complete_permutation
            ),
            "position_counts": {
                candidate_id: {
                    str(position): counts.get(position, 0)
                    for position in range(len(canonical_ids))
                }
                for candidate_id, counts in sorted(position_counts.items())
            },
            "per_candidate_position_count_imbalance": dict(
                sorted(position_count_imbalances.items())
            ),
            "maximum_per_candidate_position_count_imbalance": (
                maximum_position_count_imbalance
            ),
            "position_count_imbalance_within_one": (
                position_count_imbalance_within_one
            ),
            "ordered_immediate_adjacency_counts": [
                {
                    "first_candidate_id": first,
                    "second_candidate_id": second,
                    "count": count,
                }
                for (first, second), count in sorted(adjacency_counts.items())
            ],
        },
    }
    schedule_payload["schedule_sha256"] = sha256_value(schedule_payload)
    return schedule_payload


def reconcile_latency_protocol(
    expanded_manifest: dict[str, Any],
    sensitivity_manifest: dict[str, Any],
) -> dict[str, Any]:
    frozen = expanded_manifest["frozen_identity"]["latency_protocol"]
    source = sensitivity_manifest["latency_protocol"]
    shared = (
        "warmup_iterations",
        "timed_iterations",
        "repeated_rounds",
        "preloaded_batches",
        "batch_size_per_gpu",
        "fixed_preprocessed_shapes",
        "precision",
        "tf32",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "benchmark_seed",
        "tail_percentile",
        "bootstrap_resamples",
        "bootstrap_confidence_level",
        "bootstrap_seed",
        "synchronization",
    )
    for key in shared:
        require_equal(source.get(key), frozen.get(key), f"latency protocol {key}")
    require_equal(
        source.get("timed_scope"),
        frozen.get("warmup_and_timing_scope"),
        "latency timed scope",
    )
    require_equal(source["warmup_iterations"], 50, "latency warmups")
    require_equal(source["timed_iterations"], 100, "timed iterations")
    require_equal(source["repeated_rounds"], 5, "latency rounds")
    require_equal(
        source["timed_iterations"]
        * source["repeated_rounds"]
        * EXPECTED_GPU_COUNT,
        4000,
        "raw samples per candidate allocation",
    )
    return copy.deepcopy(source)


def source_artifacts(
    expanded_manifest_path: Path,
    expanded_manifest_sha256: str,
    combined_path: Path,
    combined_sha256: str,
    table_path: Path,
    table_sha256: str,
    candidate_table_csv_path: Path,
    candidate_table_csv_sha256: str,
    integrity_path: Path,
    integrity_sha256: str,
    protocol_erratum_path: Path,
    protocol_erratum_sha256: str,
    protocol_erratum: dict[str, Any],
    archive_snapshot: dict[str, Any],
    expanded_manifest: dict[str, Any],
    selector_replay: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection_stack = selector_replay.get("selection_stack")
    if not isinstance(selection_stack, dict):
        raise ContractError("selector replay lacks selection-stack provenance")
    require_equal(
        selection_stack_provenance(expanded_manifest),
        selection_stack,
        "selector replay/source-artifact selection stack",
    )
    repository = Path(selection_stack["repository"]).resolve()
    runtime_contract_path = (
        DEFAULT_EXPANDED_RUNTIME / "runtime_contract.v2.json"
    ).resolve()
    runtime_contract = load_json(runtime_contract_path)
    runtime_contract_internal = runtime_contract.get("contract_sha256")
    require_sha256(
        runtime_contract_internal,
        "expanded runtime contract internal SHA256",
    )
    runtime_unhashed = copy.deepcopy(runtime_contract)
    del runtime_unhashed["contract_sha256"]
    require_equal(
        sha256_value(runtime_unhashed),
        runtime_contract_internal,
        "expanded runtime contract canonical SHA256",
    )
    require_equal(
        {
            "contract_id": runtime_contract.get("contract_id"),
            "manifest_id": runtime_contract.get("manifest_id"),
            "manifest_path": runtime_contract.get("manifest_path"),
            "manifest_file_sha256": runtime_contract.get(
                "manifest_file_sha256"
            ),
            "manifest_internal_sha256": runtime_contract.get(
                "manifest_internal_sha256"
            ),
            "target_runtime_path": runtime_contract.get(
                "target_runtime_path"
            ),
            "valid_objective_observations_reused": runtime_contract.get(
                "valid_objective_observations_reused"
            ),
            "v1_runtime_reused": runtime_contract.get("v1_runtime_reused"),
        },
        {
            "contract_id": "dino_expanded_search_runtime_20260728_v2",
            "manifest_id": "dino_expanded_search_20260728_v2",
            "manifest_path": str(expanded_manifest_path.resolve()),
            "manifest_file_sha256": expanded_manifest_sha256,
            "manifest_internal_sha256": (
                EXPECTED_EXPANDED_MANIFEST_INTERNAL_SHA256
            ),
            "target_runtime_path": str(DEFAULT_EXPANDED_RUNTIME.resolve()),
            "valid_objective_observations_reused": 0,
            "v1_runtime_reused": False,
        },
        "expanded runtime contract identity",
    )
    sensitivity_path = Path(
        expanded_manifest["derivation"]["source_identity"][
            "sensitivity_manifest_path"
        ]
    ).resolve()
    sensitivity_sha256 = expanded_manifest["derivation"]["source_identity"][
        "sensitivity_manifest_sha256"
    ]
    require_equal(
        sha256_file(sensitivity_path),
        sensitivity_sha256,
        "pinned sensitivity manifest",
    )
    sensitivity = load_json(sensitivity_path)
    benchmark_path = (
        sensitivity_path.parent
        / sensitivity["frozen_inputs"]["benchmark_path"]
    ).resolve()
    evaluate_template_path = Path(
        sensitivity["frozen_inputs"]["evaluate_template_path"]
    ).resolve()
    for path, expected, label in (
        (
            benchmark_path,
            sensitivity["frozen_inputs"]["benchmark_sha256"],
            "DINO latency benchmark",
        ),
        (
            evaluate_template_path,
            sensitivity["frozen_inputs"]["evaluate_template_sha256"],
            "DINO evaluate template",
        ),
    ):
        require_equal(sha256_file(path), expected, label)

    expanded_runner_path = Path(
        expanded_manifest["derivation"]["runner_path"]
    ).resolve()
    require_equal(
        sha256_file(expanded_runner_path),
        expanded_manifest["derivation"]["runner_sha256"],
        "expanded runner source",
    )
    tool_sources = {}
    for filename in TOOL_FILENAMES:
        path = (HERE / filename).resolve()
        tool_sources[filename] = clean_head_source_provenance(
            repository,
            path,
        )
    latency_stats_path = (
        HERE.parent.parent / "src" / "tao_automl" / "latency_stats.py"
    ).resolve()
    if not latency_stats_path.is_file():
        raise ContractError(f"latency statistics source is missing: {latency_stats_path}")
    artifacts = {
        "tao_automl_repository": {
            "path": selection_stack["repository"],
            "branch": selection_stack["branch"],
            "head_commit": selection_stack["head_commit"],
            "commit_policy": selection_stack["commit_policy"],
            "selection_core_commit": selection_stack[
                "selection_core_commit"
            ],
        },
        "tao_automl_selection_stack": copy.deepcopy(selection_stack),
        "expanded_manifest": {
            "path": str(expanded_manifest_path.resolve()),
            "sha256": expanded_manifest_sha256,
            "internal_sha256": expanded_manifest["manifest_sha256"],
        },
        "expanded_combined_selection": {
            "path": str(combined_path.resolve()),
            "sha256": combined_sha256,
        },
        "expanded_candidate_table": {
            "path": str(table_path.resolve()),
            "sha256": table_sha256,
        },
        "expanded_candidate_table_csv": {
            "path": str(candidate_table_csv_path.resolve()),
            "sha256": candidate_table_csv_sha256,
        },
        "expanded_integrity_audit": {
            "path": str(integrity_path.resolve()),
            "sha256": integrity_sha256,
        },
        "phase2_protocol_erratum": {
            "path": str(protocol_erratum_path.resolve()),
            "sha256": protocol_erratum_sha256,
            "erratum_id": protocol_erratum["erratum_id"],
            "issued_at_utc": protocol_erratum["issued_at_utc"],
        },
        "expanded_seed_archives": [
            {
                "path": evidence["path"],
                "sha256": evidence["whole_file_sha256"],
                "internal_sha256": evidence["internal_archive_sha256"],
                "search_seed": evidence["search_seed"],
            }
            for evidence in archive_snapshot["seed_archives"]
        ],
        "expanded_runtime_contract": {
            "path": str(runtime_contract_path),
            "sha256": sha256_file(runtime_contract_path),
            "internal_sha256": runtime_contract_internal,
        },
        "sensitivity_manifest": {
            "path": str(sensitivity_path),
            "sha256": sensitivity_sha256,
        },
        "dino_latency_benchmark": {
            "path": str(benchmark_path),
            "sha256": sha256_file(benchmark_path),
        },
        "dino_evaluate_template": {
            "path": str(evaluate_template_path),
            "sha256": sha256_file(evaluate_template_path),
        },
        "expanded_runner": {
            "path": str(expanded_runner_path),
            "sha256": sha256_file(expanded_runner_path),
        },
        "latency_stats": {
            "path": str(latency_stats_path),
            "sha256": sha256_file(latency_stats_path),
        },
        "post_front_tools": tool_sources,
    }
    return artifacts, sensitivity


def build_manifest(
    *,
    expanded_manifest: dict[str, Any],
    candidates: list[dict[str, Any]],
    sources: dict[str, Any],
    sensitivity_manifest: dict[str, Any],
    combined: dict[str, Any],
    selector_replay: dict[str, Any],
    protocol_erratum: dict[str, Any],
    archive_snapshot: dict[str, Any],
) -> dict[str, Any]:
    replay_proof = selector_replay.get("proof")
    replay_analysis = selector_replay.get("analysis")
    if not isinstance(replay_proof, dict) or not isinstance(
        replay_analysis,
        dict,
    ):
        raise ContractError("selector replay evidence is incomplete")
    candidate_ids = [item["candidate_id"] for item in candidates]
    require_equal(
        replay_proof.get("global_rank_zero_candidate_ids"),
        candidate_ids,
        "manifest/replay rank-zero candidate IDs",
    )
    schedule = build_schedule(candidate_ids)
    protocol = reconcile_latency_protocol(
        expanded_manifest,
        sensitivity_manifest,
    )
    post = expanded_manifest["post_front_matched_validation"]
    runtime = expanded_manifest["frozen_identity"]["runtime"]
    manifest = {
        "schema_version": 1,
        "manifest_id": "dino_expanded_post_front_matched_20260728_v1",
        "status": "immutable_ready_to_launch",
        "scope": copy.deepcopy(expanded_manifest["scope"]),
        "feeds_final_selection": False,
        "feeds_reselection": False,
        "manual_candidate_addition_or_removal_permitted": False,
        "manual_winner_override_permitted": False,
        "selection_time_objective_replacement_permitted": False,
        "source_artifacts": copy.deepcopy(sources),
        "expanded_archive_snapshot": copy.deepcopy(archive_snapshot),
        "candidate_derivation": {
            "source": (
                "independent pinned tao_automl selector replay over every "
                "successful expanded_candidate_table row"
            ),
            "cross_check_source": "expanded_combined_selection.candidates",
            "predicate": (
                "recomputed valid is true and recomputed global "
                "pareto_rank equals zero"
            ),
            "canonical_order": "ascending UTF-8 candidate_id",
            "candidate_count": len(candidates),
            "candidate_ids": candidate_ids,
            "candidate_set_sha256": sha256_value(candidate_ids),
            "records_source": "expanded_candidate_table rows",
            "manual_filtering_used": False,
            "winner_identity_used": False,
            "objective_values_used_for_schedule": False,
            "selector_replay_proof": copy.deepcopy(replay_proof),
        },
        "selection_snapshot": {
            "selections": copy.deepcopy(combined["selections"]),
            "selection_authority": copy.deepcopy(
                combined["selection_authority"]
            ),
            "preserved_unchanged": True,
        },
        "candidates": copy.deepcopy(candidates),
        "schedule": schedule,
        "runtime": {
            "sqsh_path": runtime["sqsh_path"],
            "sqsh_sha256": runtime["sqsh_sha256"],
            "partition": runtime["partition"],
            "account": runtime["account"],
            "num_nodes": 1,
            "gpu_count": 8,
            "required_gpu_name": runtime["required_gpu_model"],
            "required_compute_capability": runtime[
                "required_compute_capability"
            ],
            "required_total_memory_bytes": runtime[
                "required_gpu_memory_bytes"
            ],
            "required_torch": runtime["torch"],
            "torch_version_match": "major_minor_patch",
            "required_cuda": runtime["cuda"],
            "required_cudnn": runtime["cudnn"],
            "precision": runtime["precision"],
            "tf32": runtime["tf32"],
            "sdk_path": expanded_manifest["frozen_identity"][
                "source_repositories"
            ]["tao_sdk"]["path"],
            "sdk_branch": expanded_manifest["frozen_identity"][
                "source_repositories"
            ]["tao_sdk"]["branch"],
            "sdk_commit": expanded_manifest["frozen_identity"][
                "source_repositories"
            ]["tao_sdk"]["commit"],
            "secrets_env_path": runtime["secrets_env_path"],
            "local_runtime_path": str(
                (HERE / "runtime" / "post_front_matched").resolve()
            ),
            "image_is_prebuilt_sqsh": True,
            "sdk_sqsh_conversion_enabled": False,
            "slurm_use_requeue": False,
            "slurm_time_hours": 4.0,
            "slurm_timeout_hours": 3.8,
            "submission_api": "tao_sdk.platforms.slurm.SlurmSDK.create_job",
            "output_contract": {
                "root_expression": "$TAO_RESULTS_ROOT/$TAO_JOB_ID",
                "sdk_job_scoped": True,
                "relative_layout": (
                    "dino_moo_phase2_20260728/post_front_matched/"
                    "<manifest_id>/<allocation_id>"
                ),
            },
        },
        "dataset": copy.deepcopy(expanded_manifest["frozen_identity"]["dataset"]),
        "latency_protocol": protocol,
        "paired_analysis": {
            **copy.deepcopy(post["paired_analysis"]),
            "practical_tolerance_ms": EXPECTED_PRACTICAL_TOLERANCE_MS,
            "policy_erratum_id": protocol_erratum["erratum_id"],
            "policy_erratum_sha256": (
                sources["phase2_protocol_erratum"]["sha256"]
            ),
            "both_policy_branches_must_be_emitted": True,
            "original_preregistered_bootstrap_classification": (
                copy.deepcopy(
                    protocol_erratum["corrections"][
                        "post_front_paired_classification"
                    ]["original_preregistered_bootstrap_classification"]
                )
            ),
            "effective_erratum_directional_classification": copy.deepcopy(
                protocol_erratum["corrections"][
                    "post_front_paired_classification"
                ]["effective_erratum_directional_classification"]
            ),
        },
        "allocation_contract": copy.deepcopy(post["allocation_design"]),
        "ordering_contract": copy.deepcopy(post["ordering"]),
        "selection_isolation": copy.deepcopy(post["selection_isolation"]),
        "incomplete_allocation_policy": post["allocation_design"][
            "incomplete_allocation_policy"
        ],
    }
    manifest["manifest_sha256"] = sha256_value(manifest)
    return manifest


def generate(
    *,
    expanded_manifest_path: Path,
    expanded_manifest_sha256: str,
    combined_path: Path,
    combined_sha256: str,
    table_path: Path,
    table_sha256: str,
    integrity_path: Path,
    integrity_sha256: str,
    protocol_erratum_path: Path | None = None,
    protocol_erratum_sha256: str | None = None,
    seed_archive_paths: Iterable[Path] | None = None,
) -> dict[str, Any]:
    resolved_erratum_path = (
        DEFAULT_PROTOCOL_ERRATUM.resolve()
        if protocol_erratum_path is None
        else protocol_erratum_path.resolve()
    )
    supplied_erratum_sha256 = (
        EXPECTED_PROTOCOL_ERRATUM_FILE_SHA256
        if protocol_erratum_sha256 is None
        else protocol_erratum_sha256
    )
    protocol_erratum, protocol_erratum_actual, _ = (
        load_and_validate_protocol_erratum(
            resolved_erratum_path,
            supplied_erratum_sha256,
        )
    )
    expanded, expanded_actual = load_exact_json(
        expanded_manifest_path,
        expanded_manifest_sha256,
        "expanded-search manifest",
    )
    validate_expanded_manifest(expanded, expanded_actual)
    combined, combined_actual = load_exact_json(
        combined_path,
        combined_sha256,
        "expanded combined selection",
    )
    table, table_actual = load_exact_json(
        table_path,
        table_sha256,
        "expanded candidate table",
    )
    integrity, integrity_actual = load_exact_json(
        integrity_path,
        integrity_sha256,
        "expanded integrity audit",
    )
    require_equal(
        combined["manifest"]["whole_file_sha256"],
        expanded_actual,
        "combined-selection expanded manifest",
    )
    require_equal(
        combined["post_front_matched_validation"]["contract_sha256"],
        EXPECTED_POST_FRONT_CONTRACT_SHA256,
        "combined-selection post-front contract",
    )
    require_equal(
        combined["post_front_matched_validation"][
            "measurements_feed_reselection"
        ],
        False,
        "combined-selection measurements feed reselection",
    )
    require_equal(
        combined["selection_authority"]["manual_override_used"],
        False,
        "combined-selection manual override",
    )
    validate_integrity_bindings(
        expanded_manifest_path=expanded_manifest_path,
        expanded_manifest_sha256=expanded_actual,
        combined_path=combined_path,
        combined_sha256=combined_actual,
        table_path=table_path,
        table_sha256=table_actual,
        integrity=integrity,
        combined=combined,
        table=table,
    )
    artifacts = integrity.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ContractError("expanded integrity artifacts are missing")
    candidate_table_csv_path = Path(
        artifacts.get("candidate_table_csv", "")
    ).resolve()
    candidate_table_csv_sha256 = require_sha256(
        artifacts.get("candidate_table_csv_sha256"),
        "integrity candidate-table CSV SHA256",
    )
    resolved_seed_archive_paths = (
        [
            table_path.parent / f"seed_{seed}" / "seed_archive.v1.json"
            for seed in EXPECTED_SEARCH_SEEDS
        ]
        if seed_archive_paths is None
        else [Path(path).resolve() for path in seed_archive_paths]
    )
    archive_snapshot = validate_expanded_archive_authority(
        seed_archive_paths=resolved_seed_archive_paths,
        expanded_manifest_sha256=expanded_actual,
        combined=combined,
        combined_sha256=combined_actual,
        table=table,
        table_sha256=table_actual,
        candidate_table_csv_path=candidate_table_csv_path,
        candidate_table_csv_sha256=candidate_table_csv_sha256,
        integrity_sha256=integrity_actual,
    )
    selector_replay = validate_completed_archive(
        combined,
        table,
        expanded_manifest=expanded,
        expanded_manifest_path=expanded_manifest_path,
        expanded_manifest_sha256=expanded_actual,
        integrity=integrity,
    )
    candidates = derive_candidate_records(
        combined,
        table,
        selector_replay,
    )
    sources, sensitivity = source_artifacts(
        expanded_manifest_path,
        expanded_actual,
        combined_path,
        combined_actual,
        table_path,
        table_actual,
        candidate_table_csv_path,
        candidate_table_csv_sha256,
        integrity_path,
        integrity_actual,
        resolved_erratum_path,
        protocol_erratum_actual,
        protocol_erratum,
        archive_snapshot,
        expanded,
        selector_replay,
    )
    return build_manifest(
        expanded_manifest=expanded,
        candidates=candidates,
        sources=sources,
        sensitivity_manifest=sensitivity,
        combined=combined,
        selector_replay=selector_replay,
        protocol_erratum=protocol_erratum,
        archive_snapshot=archive_snapshot,
    )


def write_new_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise ContractError(f"refusing to overwrite immutable manifest: {path}") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expanded-manifest",
        type=Path,
        default=DEFAULT_EXPANDED_MANIFEST,
    )
    parser.add_argument("--expanded-manifest-sha256", required=True)
    parser.add_argument(
        "--combined-selection",
        type=Path,
        default=DEFAULT_COMBINED_SELECTION,
    )
    parser.add_argument("--combined-selection-sha256", required=True)
    parser.add_argument(
        "--candidate-table",
        type=Path,
        default=DEFAULT_CANDIDATE_TABLE,
    )
    parser.add_argument("--candidate-table-sha256", required=True)
    parser.add_argument(
        "--integrity-audit",
        type=Path,
        default=DEFAULT_INTEGRITY_AUDIT,
    )
    parser.add_argument("--integrity-audit-sha256", required=True)
    parser.add_argument(
        "--protocol-erratum",
        type=Path,
        required=True,
        help="Exact pre-data phase-2 protocol erratum.",
    )
    parser.add_argument(
        "--protocol-erratum-sha256",
        required=True,
        help="Exact whole-file SHA256 of --protocol-erratum.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = generate(
        expanded_manifest_path=args.expanded_manifest.resolve(),
        expanded_manifest_sha256=args.expanded_manifest_sha256,
        combined_path=args.combined_selection.resolve(),
        combined_sha256=args.combined_selection_sha256,
        table_path=args.candidate_table.resolve(),
        table_sha256=args.candidate_table_sha256,
        integrity_path=args.integrity_audit.resolve(),
        integrity_sha256=args.integrity_audit_sha256,
        protocol_erratum_path=args.protocol_erratum.resolve(),
        protocol_erratum_sha256=args.protocol_erratum_sha256,
    )
    write_new_manifest(args.output.resolve(), manifest)
    print(
        json.dumps(
            {
                "status": "manifest_created_not_launched",
                "path": str(args.output.resolve()),
                "whole_file_sha256": sha256_file(args.output.resolve()),
                "internal_manifest_sha256": manifest["manifest_sha256"],
                "candidate_ids": manifest["candidate_derivation"][
                    "candidate_ids"
                ],
                "allocation_count": EXPECTED_ALLOCATION_COUNT,
                "feeds_final_selection": False,
                "feeds_reselection": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
