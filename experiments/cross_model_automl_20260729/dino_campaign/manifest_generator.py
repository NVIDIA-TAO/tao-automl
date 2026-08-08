#!/usr/bin/env python3

"""Generate and verify the direct full-VOC2007 DINO campaign manifest.

The manifest deliberately describes three full AutoML searches.  It contains
no CPU qualification, model smoke, single-candidate smoke, or shared-archive
stage.  Exact SLURM-visible dataset, checkpoint, and SQSH identities are
required before a manifest can be sealed as submission-ready.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tao_automl.ptm_registry import canonical_sha256, load_ptm_registry


SCHEMA_VERSION = 1
MODES = ("accuracy", "latency", "multi_objective")
AGENT_FLAGS = (
    "agent_selected_candidate",
    "agent_injected_candidate",
    "agent_modified_search_space_after_results",
    "agent_changed_seed_after_results",
    "agent_changed_budget_after_results",
    "agent_changed_threshold_after_results",
    "agent_changed_ptm_after_results",
    "agent_overrode_winner",
)
SELECTION_FLAGS = (
    "selector_invoked_on_matched_measurements",
    "selection_time_objectives_replaced",
    "measurements_feed_selection",
    "measurements_feed_reselection",
    "algorithm_selected_candidate_overridden",
)
SEARCH_PARAMETERS = (
    "model.enc_layers",
    "model.dec_layers",
    "train.optim.lr",
    "train.optim.weight_decay",
)
SEARCH_SPACE = {
    "model.enc_layers": {
        "type": "integer",
        "minimum": 3,
        "maximum": 6,
        "values": [3, 4, 5, 6],
    },
    "model.dec_layers": {
        "type": "integer",
        "minimum": 3,
        "maximum": 6,
        "values": [3, 4, 5, 6],
    },
    "train.optim.lr": {
        "type": "float",
        "minimum": 1.0e-5,
        "maximum": 5.0e-4,
    },
    "train.optim.weight_decay": {
        "type": "float",
        "minimum": 1.0e-5,
        "maximum": 1.0e-3,
    },
}
FROZEN_CANDIDATE_BUDGET = 20
FROZEN_TRAINING_EPOCHS = 10
FROZEN_SEARCH_SEED = 271828
FROZEN_TRAINING_SEED = 1234
FROZEN_CALIBRATION_POINTS = 8
FROZEN_LATENCY_RETENTION = 0.90
FROZEN_LATENCY_TOLERANCE_MS = 0.73553775
FROZEN_SLURM_RETRY_CAP = 10
FROZEN_HARDWARE_CONTRACT = {
    "gpu_name": "NVIDIA A100-SXM4-80GB",
    "compute_capability": "8.0",
    "total_memory_bytes": 85174583296,
}
DEFAULT_LATENCY_PROTOCOL = {
    "warmup_iterations": 50,
    "timed_iterations": 100,
    "repeated_rounds": 5,
    "preloaded_batches": 16,
    "benchmark_seed": 20260727,
    "tail_percentile": 95.0,
    "bootstrap_resamples": 5000,
    "bootstrap_confidence_level": 0.95,
    "bootstrap_seed": 424242,
    "batch_size_per_replica": 1,
    "expected_replicas": 8,
    "precision": "fp32",
    "timed_scope": "model_forward_plus_dino_gpu_postprocess",
    "excluded_scope": [
        "checkpoint_load",
        "disk_io",
        "decode_resize_normalize",
        "host_to_device_transfer",
        "coco_accumulation",
        "distributed_gather",
    ],
    "synchronization": "accelerator_sync_before_and_after_each_sample",
    "replica_alignment": "nccl_barrier_before_each_timed_sample",
    "measurement_role": "selection_time",
    "raw_samples_per_candidate": 4000,
    "validity_thresholds": {
        "max_robust_cv": 0.10,
        "max_round_median_range_fraction": 0.05,
        "max_absolute_round_drift_fraction": 0.05,
        "max_device_median_range_fraction": 0.05,
        "max_bootstrap_ci_width_fraction": 0.03,
    },
}


class ManifestError(ValueError):
    """The direct DINO campaign contract is incomplete or inconsistent."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256sum_basename_tree(path: str | Path) -> dict[str, Any]:
    """Hash sorted ``sha256sum`` basename lines for one flat file tree."""
    root = Path(path)
    if not root.is_dir():
        raise ManifestError(f"image tree is unavailable: {root}")
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for item in sorted(root.iterdir(), key=lambda candidate: candidate.name):
        if item.is_symlink() or not item.is_file():
            raise ManifestError(
                "image tree must contain regular files only: "
                f"{item}"
            )
        item_sha = sha256_file(item)
        digest.update(f"{item_sha}  {item.name}\n".encode("utf-8"))
        file_count += 1
        total_bytes += item.stat().st_size
    return {
        "algorithm": "sha256_of_sorted_sha256sum_basename_lines",
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def manifest_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ManifestError(f"{name} must be lowercase SHA-256 hex")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{name} must be a non-empty string")
    return value.strip()


def _absolute(value: Any, name: str, *, suffix: str | None = None) -> str:
    path = Path(_text(value, name))
    if not path.is_absolute():
        raise ManifestError(f"{name} must be an absolute path")
    if suffix is not None and not str(path).endswith(suffix):
        raise ManifestError(f"{name} must end in {suffix!r}")
    return str(path)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ManifestError(f"{name} must be an integer >= 1")
    return value


def _git_object_id(value: Any, name: str) -> str:
    object_id = _text(value, name)
    if (
        len(object_id) != 40
        or any(character not in "0123456789abcdef" for character in object_id)
    ):
        raise ManifestError(f"{name} must be a full lowercase Git object ID")
    return object_id


def _finite_fraction(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ManifestError(f"{name} must be a finite number in (0, 1]")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ManifestError(
            f"{name} must be a finite number in (0, 1]"
        ) from exc
    if not math.isfinite(number) or not 0.0 < number <= 1.0:
        raise ManifestError(f"{name} must be a finite number in (0, 1]")
    return number


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _registry_records() -> tuple[Any, dict[str, Mapping[str, Any]]]:
    registry = load_ptm_registry()
    model = registry.to_dict()["models"]["dino"]
    records = {
        record["id"]: copy.deepcopy(record)
        for record in model["checkpoints"]
    }
    return registry, records


def supported_dino_records() -> tuple[Mapping[str, Any], ...]:
    """Return every and only registry-supported DINO detection checkpoint."""
    _, records = _registry_records()
    return tuple(
        records[checkpoint_id]
        for checkpoint_id in sorted(records)
        if records[checkpoint_id].get("status") == "supported"
        and "object_detection"
        in records[checkpoint_id].get("task_compatibility", ())
    )


def _tao_release(version: str) -> str:
    """Normalize an RC/build identity to the registry compatibility release."""
    value = _text(version, "runtime.tao_version")
    head = value.split("-", 1)[0]
    components = head.split(".")
    if len(components) != 3 or any(not item.isdigit() for item in components):
        raise ManifestError(
            "runtime.tao_version must begin with a major.minor.patch release"
        )
    return head


def _compatible(specifiers: Sequence[str], tao_release: str) -> bool:
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.version import Version

    for raw in specifiers:
        try:
            if SpecifierSet(str(raw)).contains(
                Version(tao_release),
                prereleases=True,
            ):
                return True
        except InvalidSpecifier as exc:
            raise ManifestError(
                f"invalid registry TAO compatibility specifier {raw!r}"
            ) from exc
    return False


def _runtime_artifacts(
    record: Mapping[str, Any],
    tao_version: str,
) -> tuple[dict[str, Any], ...]:
    release = _tao_release(tao_version)
    artifacts = []
    if _compatible(record.get("compatible_tao_versions", ()), release):
        artifacts.append(
            {
                "kind": "registry_source",
                "sha256": record.get("sha256"),
                "size_bytes": record.get("expected_size_bytes"),
                "member": record.get("source", {}).get("member"),
                "adapter_id": None,
            }
        )
    for adapter in record.get("artifact_adapters", ()):
        if not _compatible(adapter.get("compatible_tao_versions", ()), release):
            continue
        output = adapter.get("output", {})
        artifacts.append(
            {
                "kind": "registered_adapter_output",
                "sha256": output.get("sha256"),
                "size_bytes": output.get("expected_size_bytes"),
                "member": output.get("member"),
                "adapter_id": adapter.get("id"),
            }
        )
    valid = []
    for artifact in artifacts:
        try:
            _sha(artifact["sha256"], "runtime PTM artifact sha256")
            _positive_int(
                artifact["size_bytes"],
                "runtime PTM artifact size_bytes",
            )
            _text(artifact["member"], "runtime PTM artifact member")
        except ManifestError:
            continue
        valid.append(artifact)
    return tuple(valid)


def _resolve_ptms(
    supplied: Mapping[str, Any],
    *,
    tao_version: str,
) -> list[dict[str, Any]]:
    registry, all_records = _registry_records()
    supported = supported_dino_records()
    supported_ids = {record["id"] for record in supported}
    if set(supplied) != supported_ids:
        raise ManifestError(
            "ptm_runtime must contain every and only registry status=supported "
            f"DINO detection PTM; expected={sorted(supported_ids)}, "
            f"observed={sorted(supplied)}"
        )
    result = []
    for record in supported:
        checkpoint_id = record["id"]
        runtime = supplied[checkpoint_id]
        if not isinstance(runtime, Mapping):
            raise ManifestError(f"ptm_runtime[{checkpoint_id!r}] must be an object")
        artifact_sha = _sha(
            runtime.get("artifact_sha256"),
            f"ptm_runtime[{checkpoint_id!r}].artifact_sha256",
        )
        choices = _runtime_artifacts(record, tao_version)
        matching = [item for item in choices if item["sha256"] == artifact_sha]
        if len(matching) != 1:
            raise ManifestError(
                f"PTM {checkpoint_id!r} artifact {artifact_sha} is not the "
                f"single registered runtime artifact for TAO {tao_version}"
            )
        artifact = matching[0]
        supplied_size = _positive_int(
            runtime.get("artifact_size_bytes"),
            f"ptm_runtime[{checkpoint_id!r}].artifact_size_bytes",
        )
        if supplied_size != artifact["size_bytes"]:
            raise ManifestError(
                f"PTM {checkpoint_id!r} artifact size does not match registry"
            )
        result.append(
            {
                "id": checkpoint_id,
                "registry_status": "supported",
                "registry_record_sha256": canonical_sha256(record),
                "registry_sha256": registry.document_sha256,
                "checkpoint_target": record["checkpoint_target"],
                "source_identity": record["source"]["immutable_identity"],
                "runtime_artifact": copy.deepcopy(artifact),
                "slurm_path": _absolute(
                    runtime.get("slurm_path"),
                    f"ptm_runtime[{checkpoint_id!r}].slurm_path",
                ),
                "default_spec_overrides": copy.deepcopy(
                    record["default_spec_overrides"]
                ),
                "input_contract": copy.deepcopy(record["input_contract"]),
            }
        )
    return result


def _mode_objective(mode: str, retention: float) -> dict[str, Any]:
    objectives = [
        {"metric": "mAP50", "direction": "maximize", "role": "accuracy"},
        {"metric": "latency_ms", "direction": "minimize", "role": "latency"},
    ]
    if mode == "accuracy":
        return {
            "selection_mode": mode,
            "objectives": objectives,
            "acquisition": "expected_improvement",
            "latency_accuracy_retention": None,
            "multi_objective_min_accuracy": None,
            "selection_policy": "highest_valid_accuracy",
        }
    if mode == "latency":
        return {
            "selection_mode": mode,
            "objectives": objectives,
            "acquisition": "constrained_expected_improvement",
            "latency_accuracy_retention": {
                "type": "relative",
                "retained_fraction": retention,
                "reference": "accuracy_winner",
            },
            "multi_objective_min_accuracy": None,
            "selection_policy": "equivalent_fastest_accuracy_tiebreak",
        }
    return {
        "selection_mode": mode,
        "objectives": objectives,
        "acquisition": "parego_expected_improvement",
        "latency_accuracy_retention": None,
        "multi_objective_min_accuracy": None,
        "selection_policy": "normalized_augmented_chebyshev",
    }


def build_manifest(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Build one strict, unsealed direct-campaign manifest."""
    if not isinstance(inputs, Mapping):
        raise ManifestError("campaign inputs must be an object")
    source = inputs.get("source")
    runtime = inputs.get("runtime")
    dataset = inputs.get("dataset")
    search = inputs.get("search")
    if not all(isinstance(value, Mapping) for value in (source, runtime, dataset, search)):
        raise ManifestError("source, runtime, dataset, and search must be objects")

    repository = Path(
        _absolute(source.get("repository"), "source.repository")
    )
    if not repository.is_dir():
        raise ManifestError("source.repository must exist when sealing")
    requested_commit = _text(source.get("commit"), "source.commit")
    commit = (
        _git(repository, "rev-parse", "HEAD")
        if requested_commit == "HEAD"
        else _git_object_id(requested_commit, "source.commit")
    )
    if source.get("dirty") is not False:
        raise ManifestError("direct campaign requires source.dirty=false")
    if _git(repository, "rev-parse", "HEAD") != commit:
        raise ManifestError("source.commit does not match source.repository HEAD")
    if _git(repository, "status", "--porcelain"):
        raise ManifestError("source.repository must be clean before sealing")

    wheel_path = Path(
        _absolute(source.get("wheel_path"), "source.wheel_path", suffix=".whl")
    )
    wheel_sha = _sha(source.get("wheel_sha256"), "source.wheel_sha256")
    if not wheel_path.is_file() or sha256_file(wheel_path) != wheel_sha:
        raise ManifestError("source wheel SHA-256 does not match the file")
    wheel_source_commit = _git_object_id(
        source.get("wheel_source_commit"),
        "source.wheel_source_commit",
    )
    if subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "merge-base",
            "--is-ancestor",
            wheel_source_commit,
            commit,
        ],
        check=False,
        capture_output=True,
        timeout=30,
    ).returncode != 0:
        raise ManifestError(
            "source.wheel_source_commit must be an ancestor of source.commit"
        )

    skill_dir = Path(
        _absolute(runtime.get("skill_dir"), "runtime.skill_dir")
    )
    skill_revision = _git_object_id(
        runtime.get("skill_revision"),
        "runtime.skill_revision",
    )
    if not skill_dir.is_dir():
        raise ManifestError("runtime.skill_dir must exist when sealing")
    if _git(skill_dir, "rev-parse", "HEAD") != skill_revision:
        raise ManifestError("runtime.skill_revision does not match skill_dir")
    if _git(skill_dir, "status", "--porcelain"):
        raise ManifestError("runtime.skill_dir repository must be clean")

    sdk_dir = Path(_absolute(runtime.get("sdk_dir"), "runtime.sdk_dir"))
    sdk_revision = _git_object_id(
        runtime.get("sdk_revision"),
        "runtime.sdk_revision",
    )
    if not sdk_dir.is_dir():
        raise ManifestError("runtime.sdk_dir must exist when sealing")
    if _git(sdk_dir, "rev-parse", "HEAD") != sdk_revision:
        raise ManifestError("runtime.sdk_revision does not match sdk_dir")
    if _git(sdk_dir, "status", "--porcelain"):
        raise ManifestError("runtime.sdk_dir repository must be clean")

    tao_version = _text(runtime.get("tao_version"), "runtime.tao_version")
    sqsh_path = _absolute(
        runtime.get("sqsh_path"),
        "runtime.sqsh_path",
        suffix=".sqsh",
    )
    sqsh_sha = _sha(runtime.get("sqsh_sha256"), "runtime.sqsh_sha256")
    sqsh_size = _positive_int(
        runtime.get("sqsh_size_bytes"),
        "runtime.sqsh_size_bytes",
    )
    image_reference = _text(
        runtime.get("image_reference"),
        "runtime.image_reference",
    )

    dataset_manifest = Path(
        _absolute(dataset.get("manifest_path"), "dataset.manifest_path")
    )
    dataset_manifest_sha = _sha(
        dataset.get("manifest_sha256"),
        "dataset.manifest_sha256",
    )
    if (
        dataset_manifest.is_file()
        and sha256_file(dataset_manifest) != dataset_manifest_sha
    ):
        raise ManifestError("dataset manifest SHA-256 does not match the file")
    integrity = Path(
        _absolute(dataset.get("integrity_path"), "dataset.integrity_path")
    )
    integrity_sha = _sha(
        dataset.get("integrity_sha256"),
        "dataset.integrity_sha256",
    )
    if integrity.is_file() and sha256_file(integrity) != integrity_sha:
        raise ManifestError("dataset integrity SHA-256 does not match the file")
    slurm_root = _absolute(dataset.get("slurm_root"), "dataset.slurm_root")
    train_annotation_sha = _sha(
        dataset.get("train_annotation_sha256"),
        "dataset.train_annotation_sha256",
    )
    train_annotation_size = _positive_int(
        dataset.get("train_annotation_size_bytes"),
        "dataset.train_annotation_size_bytes",
    )
    validation_annotation_sha = _sha(
        dataset.get("validation_annotation_sha256"),
        "dataset.validation_annotation_sha256",
    )
    validation_annotation_size = _positive_int(
        dataset.get("validation_annotation_size_bytes"),
        "dataset.validation_annotation_size_bytes",
    )
    image_count = _positive_int(
        dataset.get("image_count"),
        "dataset.image_count",
    )
    image_total_bytes = _positive_int(
        dataset.get("image_total_bytes"),
        "dataset.image_total_bytes",
    )
    image_tree_sha = _sha(
        dataset.get("image_tree_sha256"),
        "dataset.image_tree_sha256",
    )
    image_tree_algorithm = _text(
        dataset.get("image_tree_algorithm"),
        "dataset.image_tree_algorithm",
    )
    if (
        image_tree_algorithm
        != "sha256_of_sorted_sha256sum_basename_lines"
    ):
        raise ManifestError("dataset.image_tree_algorithm is unsupported")
    local_image_dir = Path(
        _absolute(dataset.get("local_image_dir"), "dataset.local_image_dir")
    )
    observed_image_tree = sha256sum_basename_tree(local_image_dir)
    expected_image_tree = {
        "algorithm": image_tree_algorithm,
        "sha256": image_tree_sha,
        "file_count": image_count,
        "total_bytes": image_total_bytes,
    }
    if observed_image_tree != expected_image_tree:
        raise ManifestError("local VOC JPEG tree does not match its identity")

    candidate_budget = _positive_int(
        search.get("candidate_budget_per_mode"),
        "search.candidate_budget_per_mode",
    )
    training_epochs = _positive_int(
        search.get("training_epochs"),
        "search.training_epochs",
    )
    search_seed = _positive_int(search.get("search_seed"), "search.search_seed")
    training_seed = _positive_int(
        search.get("training_seed"),
        "search.training_seed",
    )
    retention = _finite_fraction(
        search.get("latency_accuracy_retention"),
        "search.latency_accuracy_retention",
    )
    if (
        candidate_budget != FROZEN_CANDIDATE_BUDGET
        or training_epochs != FROZEN_TRAINING_EPOCHS
        or search_seed != FROZEN_SEARCH_SEED
        or training_seed != FROZEN_TRAINING_SEED
    ):
        raise ManifestError(
            "DINO campaign budget and seeds must remain 20 recommendations, "
            "10 epochs, search seed 271828, and training seed 1234"
        )
    if retention != FROZEN_LATENCY_RETENTION:
        raise ManifestError(
            "the direct DINO profile must explicitly freeze 0.90 retention"
        )
    latency_tolerance = float(
        search.get("latency_practical_tolerance_ms")
    )
    if (
        not math.isfinite(latency_tolerance)
        or latency_tolerance != FROZEN_LATENCY_TOLERANCE_MS
    ):
        raise ManifestError(
            "DINO campaign latency tolerance must remain 0.73553775 ms"
        )
    calibration_points = _positive_int(
        search.get("calibration_points"),
        "search.calibration_points",
    )
    if calibration_points != FROZEN_CALIBRATION_POINTS:
        raise ManifestError("DINO campaign calibration_points must remain 8")

    ptms = _resolve_ptms(
        inputs.get("ptm_runtime", {}),
        tao_version=tao_version,
    )
    input_contracts = {
        canonical_sha256(record["input_contract"]): record["input_contract"]
        for record in ptms
    }
    if len(input_contracts) != 1:
        raise ManifestError(
            "the narrow direct campaign requires one shared PTM input contract"
        )
    next(iter(input_contracts.values()))
    input_descriptor = {
        "schema_version": 1,
        "shape_sequence": [
            [1, 4, 800, 1333],
            [1, 4, 800, 1333],
            [1, 4, 800, 1333],
            [1, 4, 1333, 800],
            [1, 4, 800, 1333],
            [1, 4, 1333, 800],
            [1, 4, 1333, 800],
            [1, 4, 800, 1333],
            [1, 4, 800, 1333],
            [1, 4, 800, 1333],
            [1, 4, 800, 1333],
            [1, 4, 1333, 800],
            [1, 4, 800, 1333],
            [1, 4, 800, 1333],
            [1, 4, 800, 1333],
            [1, 4, 800, 1333],
        ],
        "validation_image_ids": [
            5,
            7,
            9,
            16,
            19,
            20,
            21,
            24,
            30,
            39,
            41,
            46,
            50,
            51,
            52,
            60,
        ],
        "dtype": "float32",
        "content": "first_16_deterministic_preprocessed_validation_batches",
        "padding_mask_channel": 3,
        "preloaded_batches": 16,
        "benchmark_seed": 20260727,
        "validation_annotation_sha256": validation_annotation_sha,
        "image_tree_sha256": image_tree_sha,
        "required_hardware": copy.deepcopy(FROZEN_HARDWARE_CONTRACT),
    }
    worker_path = Path(__file__).with_name("dino_latency_worker.py")
    runtime_identity = {
        "image_reference": image_reference,
        "sqsh_sha256": sqsh_sha,
        "sqsh_size_bytes": sqsh_size,
        "worker_sha256": sha256_file(worker_path),
        "latency_benchmark_sha256": sha256_file(
            repository / "src/tao_automl/latency_benchmark.py"
        ),
        "latency_stats_sha256": sha256_file(
            repository / "src/tao_automl/latency_stats.py"
        ),
        "precision": "fp32",
        "tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "required_hardware": copy.deepcopy(FROZEN_HARDWARE_CONTRACT),
    }
    mode_records = []
    campaign_id = _text(inputs.get("campaign_id"), "campaign_id")
    for mode in MODES:
        objective = _mode_objective(mode, retention)
        mode_records.append(
            {
                "mode": mode,
                "job_id": f"{campaign_id}-{mode}",
                "session_id": f"{campaign_id}-{mode}",
                "observation_namespace": f"{campaign_id}-{mode}-observations",
                "observation_sharing": False,
                "initial_observation_ids": [],
                "objective": objective,
                "objective_sha256": canonical_sha256(objective),
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "model": "dino",
        "task": "object_detection",
        "execution": {
            "kind": "direct_full_search",
            "cpu_runs": 0,
            "smoke_runs": 0,
            "smoke_or_cpu_preflight_skipped_by_user": True,
            "shared_archive": False,
            "independent_mode_jobs": True,
            "submission_ready": True,
        },
        "source": {
            "repository": str(repository),
            "commit": commit,
            "dirty": False,
            "wheel_path": str(wheel_path),
            "wheel_sha256": wheel_sha,
            "wheel_source_commit": wheel_source_commit,
        },
        "runtime": {
            "platform": "slurm",
            "skill_dir": str(skill_dir),
            "skill_revision": skill_revision,
            "sdk_dir": str(sdk_dir),
            "sdk_revision": sdk_revision,
            "tao_version": tao_version,
            "sqsh_path": sqsh_path,
            "sqsh_sha256": sqsh_sha,
            "sqsh_size_bytes": sqsh_size,
            "image_reference": image_reference,
            "partition": _text(runtime.get("partition"), "runtime.partition"),
            "account": _text(runtime.get("account"), "runtime.account"),
            "base_results_dir": _absolute(
                runtime.get("base_results_dir"),
                "runtime.base_results_dir",
            ),
            "container_mounts": _absolute(
                runtime.get("container_mounts"),
                "runtime.container_mounts",
            ),
            "nodes": 1,
            "gpus_per_node": 8,
            "tasks_per_node": 1,
            "distributed_workers_per_node": 8,
            "precision": "fp32",
            "train_batch_size_per_gpu": 4,
            "evaluation_batch_size_per_gpu": 4,
            "latency_batch_size_per_gpu": 1,
            "hardware_contract": copy.deepcopy(FROZEN_HARDWARE_CONTRACT),
            "time_hours": 4.0,
            "timeout_hours": 3.8,
            "slurm_use_requeue": True,
            "max_job_retries": FROZEN_SLURM_RETRY_CAP,
            "slurm_use_sqsh": False,
        },
        "dataset": {
            "id": "pascal_voc_2007_full_detection",
            "manifest_path": str(dataset_manifest),
            "manifest_sha256": dataset_manifest_sha,
            "integrity_path": str(integrity),
            "integrity_sha256": integrity_sha,
            "slurm_root": slurm_root,
            "local_image_dir": str(local_image_dir),
            "train_image_dir": f"{slurm_root}/VOCdevkit/VOC2007/JPEGImages",
            "train_annotation": (
                f"{slurm_root}/coco/annotations/instances_train2007.json"
            ),
            "train_annotation_sha256": train_annotation_sha,
            "train_annotation_size_bytes": train_annotation_size,
            "validation_image_dir": (
                f"{slurm_root}/VOCdevkit/VOC2007/JPEGImages"
            ),
            "validation_annotation": (
                f"{slurm_root}/coco/annotations/instances_val2007.json"
            ),
            "validation_annotation_sha256": validation_annotation_sha,
            "validation_annotation_size_bytes": validation_annotation_size,
            "image_tree": {
                "algorithm": image_tree_algorithm,
                "sha256": image_tree_sha,
                "file_count": image_count,
                "total_bytes": image_total_bytes,
            },
            "num_classes": 21,
            "eval_class_ids": list(range(1, 21)),
        },
        "ptms": ptms,
        "search": {
            "algorithm": "bayesian",
            "implementation": "objective_aware_bayesian_v1",
            "search_seed": search_seed,
            "training_seed": training_seed,
            "candidate_budget_per_mode": candidate_budget,
            "max_concurrent_candidates_per_mode": 1,
            "training_epochs": training_epochs,
            "calibration_points": calibration_points,
            "parameters": list(SEARCH_PARAMETERS),
            "space": copy.deepcopy(SEARCH_SPACE),
            "space_sha256": canonical_sha256(SEARCH_SPACE),
            "latency_accuracy_retention": retention,
            "latency_practical_tolerance_ms": latency_tolerance,
        },
        "latency_protocol": {
            **copy.deepcopy(DEFAULT_LATENCY_PROTOCOL),
            "input_descriptor": input_descriptor,
            "input_sha256": canonical_sha256(input_descriptor),
            "runtime_identity": runtime_identity,
            "runtime_sha256": canonical_sha256(runtime_identity),
        },
        "modes": mode_records,
        "agent_intervention_flags": {name: False for name in AGENT_FLAGS},
        "selection_isolation_flags": {
            name: False for name in SELECTION_FLAGS
        },
    }
    validate_manifest(manifest, require_seal=False)
    return manifest


def validate_manifest(
    document: Mapping[str, Any],
    *,
    require_seal: bool = True,
) -> dict[str, Any]:
    """Validate invariants that must hold before any SDK is constructed."""
    if not isinstance(document, Mapping):
        raise ManifestError("manifest must be an object")
    value = copy.deepcopy(dict(document))
    seal = value.pop("manifest_sha256", None)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("manifest schema_version is unsupported")
    if value.get("model") != "dino" or value.get("task") != "object_detection":
        raise ManifestError("manifest must target DINO object detection")
    source = value.get("source", {})
    _absolute(source.get("repository"), "source.repository")
    _git_object_id(source.get("commit"), "source.commit")
    if source.get("dirty") is not False:
        raise ManifestError("sealed source must be clean")
    _absolute(source.get("wheel_path"), "source.wheel_path", suffix=".whl")
    _sha(source.get("wheel_sha256"), "source.wheel_sha256")
    _git_object_id(
        source.get("wheel_source_commit"),
        "source.wheel_source_commit",
    )
    execution = value.get("execution", {})
    if execution != {
        "kind": "direct_full_search",
        "cpu_runs": 0,
        "smoke_runs": 0,
        "smoke_or_cpu_preflight_skipped_by_user": True,
        "shared_archive": False,
        "independent_mode_jobs": True,
        "submission_ready": True,
    }:
        raise ManifestError(
            "execution must be direct, independent, submission-ready, and "
            "contain zero CPU/smoke runs"
        )
    runtime = value.get("runtime", {})
    if (
        runtime.get("platform") != "slurm"
        or runtime.get("nodes") != 1
        or runtime.get("gpus_per_node") != 8
        or runtime.get("tasks_per_node") != 1
        or runtime.get("distributed_workers_per_node") != 8
        or runtime.get("time_hours") != 4.0
        or runtime.get("timeout_hours") != 3.8
        or runtime.get("slurm_use_requeue") is not True
        or runtime.get("max_job_retries") != FROZEN_SLURM_RETRY_CAP
        or runtime.get("slurm_use_sqsh") is not False
        or runtime.get("base_results_dir")
        != "/lustre/fsw/portfolios/edgeai/users/rarunachalam"
        or runtime.get("container_mounts") != "/lustre"
        or runtime.get("hardware_contract") != FROZEN_HARDWARE_CONTRACT
    ):
        raise ManifestError("runtime must be one SLURM node with eight GPU workers")
    _absolute(runtime.get("sqsh_path"), "runtime.sqsh_path", suffix=".sqsh")
    _sha(runtime.get("sqsh_sha256"), "runtime.sqsh_sha256")
    _positive_int(runtime.get("sqsh_size_bytes"), "runtime.sqsh_size_bytes")
    _absolute(runtime.get("sdk_dir"), "runtime.sdk_dir")
    _git_object_id(runtime.get("sdk_revision"), "runtime.sdk_revision")
    _absolute(runtime.get("skill_dir"), "runtime.skill_dir")
    _git_object_id(runtime.get("skill_revision"), "runtime.skill_revision")
    dataset = value.get("dataset", {})
    if (
        dataset.get("id") != "pascal_voc_2007_full_detection"
        or dataset.get("num_classes") != 21
        or dataset.get("eval_class_ids") != list(range(1, 21))
    ):
        raise ManifestError("manifest does not bind the complete VOC2007 contract")
    for name in (
        "local_image_dir",
        "train_image_dir",
        "train_annotation",
        "validation_image_dir",
        "validation_annotation",
    ):
        _absolute(dataset.get(name), f"dataset.{name}")
    _sha(
        dataset.get("train_annotation_sha256"),
        "dataset.train_annotation_sha256",
    )
    _positive_int(
        dataset.get("train_annotation_size_bytes"),
        "dataset.train_annotation_size_bytes",
    )
    _sha(
        dataset.get("validation_annotation_sha256"),
        "dataset.validation_annotation_sha256",
    )
    _positive_int(
        dataset.get("validation_annotation_size_bytes"),
        "dataset.validation_annotation_size_bytes",
    )
    image_tree = dataset.get("image_tree", {})
    if (
        image_tree.get("algorithm")
        != "sha256_of_sorted_sha256sum_basename_lines"
    ):
        raise ManifestError("VOC image-tree digest algorithm changed")
    _sha(image_tree.get("sha256"), "dataset.image_tree.sha256")
    if (
        image_tree.get("file_count") != 9963
        or image_tree.get("total_bytes") != 875453699
    ):
        raise ManifestError("VOC image-tree count or byte size changed")

    records = value.get("ptms")
    if not isinstance(records, list) or not records:
        raise ManifestError("manifest must contain at least one supported PTM")
    expected_supported = {record["id"] for record in supported_dino_records()}
    observed = {record.get("id") for record in records}
    if observed != expected_supported or any(
        record.get("registry_status") != "supported" for record in records
    ):
        raise ManifestError(
            "manifest PTMs must remain every and only status=supported DINO PTMs"
        )

    search = value.get("search", {})
    if search.get("algorithm") != "bayesian":
        raise ManifestError("direct campaign requires Bayesian acquisition")
    if search.get("parameters") != list(SEARCH_PARAMETERS):
        raise ManifestError("search parameter order or membership changed")
    if search.get("space") != SEARCH_SPACE:
        raise ManifestError("search space changed")
    if search.get("space_sha256") != canonical_sha256(SEARCH_SPACE):
        raise ManifestError("search-space hash changed")
    if _finite_fraction(
        search.get("latency_accuracy_retention"),
        "search.latency_accuracy_retention",
    ) != FROZEN_LATENCY_RETENTION:
        raise ManifestError("latency retention must remain explicit 0.90")
    if _positive_int(
        search.get("candidate_budget_per_mode"),
        "search.candidate_budget_per_mode",
    ) != FROZEN_CANDIDATE_BUDGET:
        raise ManifestError("candidate budget must remain 20 per mode")
    if (
        _positive_int(search.get("training_epochs"), "search.training_epochs")
        != FROZEN_TRAINING_EPOCHS
    ):
        raise ManifestError("training epochs must remain 10")
    if search.get("search_seed") != FROZEN_SEARCH_SEED:
        raise ManifestError("search seed must remain 271828")
    if search.get("training_seed") != FROZEN_TRAINING_SEED:
        raise ManifestError("training seed must remain 1234")
    if search.get("calibration_points") != FROZEN_CALIBRATION_POINTS:
        raise ManifestError("calibration points must remain 8")
    if (
        search.get("latency_practical_tolerance_ms")
        != FROZEN_LATENCY_TOLERANCE_MS
    ):
        raise ManifestError("latency tolerance must remain 0.73553775 ms")
    if search.get("max_concurrent_candidates_per_mode") != 1:
        raise ManifestError(
            "sequential Bayesian mode jobs require max concurrency one"
        )

    modes = value.get("modes")
    if (
        not isinstance(modes, list)
        or tuple(item.get("mode") for item in modes) != MODES
    ):
        raise ManifestError("manifest must contain accuracy, latency, and MOO")
    namespaces = set()
    for record in modes:
        mode = record["mode"]
        if record.get("observation_sharing") is not False:
            raise ManifestError("mode observations must never be shared")
        if record.get("initial_observation_ids") != []:
            raise ManifestError("every mode must begin with an empty archive")
        namespace = _text(
            record.get("observation_namespace"),
            f"modes[{mode}].observation_namespace",
        )
        if namespace in namespaces:
            raise ManifestError("mode observation namespaces must be unique")
        namespaces.add(namespace)
        objective = record.get("objective")
        if objective != _mode_objective(
            mode,
            search["latency_accuracy_retention"],
        ):
            raise ManifestError(f"{mode} objective contract changed")
        if record.get("objective_sha256") != canonical_sha256(objective):
            raise ManifestError(f"{mode} objective hash changed")
    moo = modes[2]["objective"]
    if (
        moo["latency_accuracy_retention"] is not None
        or moo["multi_objective_min_accuracy"] is not None
    ):
        raise ManifestError("multi-objective mode inherited an accuracy floor")
    protocol = value.get("latency_protocol", {})
    for key, expected in DEFAULT_LATENCY_PROTOCOL.items():
        if protocol.get(key) != expected:
            raise ManifestError(f"latency protocol changed: {key}")
    input_descriptor = protocol.get("input_descriptor")
    if (
        not isinstance(input_descriptor, Mapping)
        or input_descriptor.get("shape_sequence")
        != [
            [1, 4, 800, 1333],
            [1, 4, 800, 1333],
            [1, 4, 800, 1333],
            [1, 4, 1333, 800],
            [1, 4, 800, 1333],
            [1, 4, 1333, 800],
            [1, 4, 1333, 800],
            [1, 4, 800, 1333],
            [1, 4, 800, 1333],
            [1, 4, 800, 1333],
            [1, 4, 800, 1333],
            [1, 4, 1333, 800],
            [1, 4, 800, 1333],
            [1, 4, 800, 1333],
            [1, 4, 800, 1333],
            [1, 4, 800, 1333],
        ]
        or input_descriptor.get("validation_image_ids")
        != [5, 7, 9, 16, 19, 20, 21, 24, 30, 39, 41, 46, 50, 51, 52, 60]
        or input_descriptor.get("required_hardware")
        != FROZEN_HARDWARE_CONTRACT
        or input_descriptor.get("preloaded_batches") != 16
        or input_descriptor.get("benchmark_seed") != 20260727
        or protocol.get("input_sha256")
        != canonical_sha256(input_descriptor)
    ):
        raise ManifestError("latency input descriptor changed")
    runtime_identity = protocol.get("runtime_identity")
    if (
        not isinstance(runtime_identity, Mapping)
        or protocol.get("runtime_sha256")
        != canonical_sha256(runtime_identity)
    ):
        raise ManifestError("latency runtime identity changed")
    if value.get("agent_intervention_flags") != {
        name: False for name in AGENT_FLAGS
    }:
        raise ManifestError("agent intervention flags must all remain false")
    if value.get("selection_isolation_flags") != {
        name: False for name in SELECTION_FLAGS
    }:
        raise ManifestError("selection-isolation flags must all remain false")

    if require_seal:
        _sha(seal, "manifest_sha256")
        if manifest_sha256(value) != seal:
            raise ManifestError("manifest SHA-256 verification failed")
        value["manifest_sha256"] = seal
    elif seal is not None:
        raise ManifestError("unsealed manifest input must not predeclare a hash")
    return value


def seal_manifest(document: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_manifest(document, require_seal=False)
    value["manifest_sha256"] = manifest_sha256(value)
    return validate_manifest(value)


def load_manifest(path: str | Path) -> dict[str, Any]:
    return validate_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.inputs.read_text(encoding="utf-8"))
    sealed = seal_manifest(build_manifest(raw))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(args.output.resolve()),
                "manifest_sha256": sealed["manifest_sha256"],
                "supported_ptm_ids": [item["id"] for item in sealed["ptms"]],
                "mode_job_ids": [
                    item["job_id"] for item in sealed["modes"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
