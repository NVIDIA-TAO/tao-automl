"""Shared deterministic contracts for the DINO sensitivity latency study."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "sensitivity_latency_manifest.v1.json"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_relative(owner: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (owner.parent / path).resolve()


def load_contract(path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    contract = json.loads(path.read_text())
    if contract.get("schema_version") != 1:
        raise ValueError("sensitivity latency manifest schema_version must be 1")
    if contract.get("feeds_final_selection") is not False:
        raise ValueError("feeds_final_selection must be false")
    if contract.get("manual_promotion_permitted") is not False:
        raise ValueError("manual promotion must be disabled")
    frozen = contract["frozen_inputs"]
    one_path = resolve_relative(path, frozen["one_factor_manifest_path"])
    actual = sha256_file(one_path)
    if actual != frozen["one_factor_manifest_sha256"]:
        raise RuntimeError(
            f"one-factor manifest drift: {actual} != "
            f"{frozen['one_factor_manifest_sha256']}"
        )
    one = json.loads(one_path.read_text())
    if one.get("feeds_final_selection") is not False:
        raise ValueError("one-factor manifest must be validation-only")
    return contract, one, one_path


def _set_model_value(model: dict[str, Any], path: str, value: Any) -> None:
    prefix = "model."
    if not path.startswith(prefix) or "." in path[len(prefix) :]:
        raise ValueError(f"unsupported one-factor model path: {path}")
    model[path[len(prefix) :]] = copy.deepcopy(value)


def build_profiles(one: dict[str, Any]) -> list[dict[str, Any]]:
    reference = copy.deepcopy(one["reference"]["model"])
    profiles = [
        {
            "profile_id": one["reference"]["profile_id"],
            "axis": None,
            "level": None,
            "execution": "train_evaluate_benchmark",
            "model": reference,
        }
    ]
    seen = {sha256_value(reference)}
    for axis in one["design"]["axes"]:
        for level in sorted(axis["levels"]):
            if level == axis["reference"]:
                continue
            model = copy.deepcopy(reference)
            _set_model_value(model, axis["path"], level)
            digest = sha256_value(model)
            if digest in seen:
                raise ValueError("duplicate resolved model profile")
            seen.add(digest)
            short = axis["path"].removeprefix("model.")
            profiles.append(
                {
                    "profile_id": f"{short}_{str(level).replace('.', 'p')}",
                    "axis": axis["path"],
                    "level": level,
                    "execution": axis["execution"],
                    "model": model,
                }
            )
    if len(profiles) != one["design"]["expected_unique_profiles"]:
        raise ValueError("one-factor profile count drift")
    frozen = one["digest_contract"]["frozen_profile_digests"]
    for profile in profiles:
        expected = frozen[profile["profile_id"]][
            "resolved_model_spec_sha256"
        ]
        actual = sha256_value(profile["model"])
        if actual != expected:
            raise RuntimeError(
                f"resolved model digest drift for {profile['profile_id']}"
            )
        profile["resolved_model_spec_sha256"] = actual
    return profiles


def williams_base_indices(size: int) -> list[int]:
    if size <= 0 or size % 2:
        raise ValueError("Williams design requires a positive even size")
    base = [0]
    for low in range(1, size // 2 + 1):
        base.append(low)
        high = size - low
        if high != low:
            base.append(high)
    if len(base) != size or len(set(base)) != size:
        raise RuntimeError("invalid Williams base construction")
    return base


def build_schedule(
    contract: dict[str, Any],
    profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    profile_ids = [profile["profile_id"] for profile in profiles]
    size = len(profile_ids)
    base = williams_base_indices(size)
    schedule = []
    rows = contract["design"]["williams_row_indices_by_seed"]
    for seed in contract["design"]["training_seeds"]:
        seed_rows = rows[str(seed)]
        if len(seed_rows) != contract["design"]["allocations_per_seed"]:
            raise ValueError(f"wrong row count for seed {seed}")
        for repeat_index, row_index in enumerate(seed_rows):
            order = [
                profile_ids[(index + row_index) % size] for index in base
            ]
            schedule.append(
                {
                    "allocation_id": (
                        f"seed_{seed:06d}_allocation_{repeat_index}_"
                        f"row_{row_index:02d}"
                    ),
                    "seed": seed,
                    "repeat_index": repeat_index,
                    "williams_row_index": row_index,
                    "profile_order": order,
                }
            )
    validate_schedule(contract, profiles, schedule)
    return schedule


def validate_schedule(
    contract: dict[str, Any],
    profiles: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
) -> None:
    profile_ids = [profile["profile_id"] for profile in profiles]
    profile_set = set(profile_ids)
    if len(schedule) != contract["design"]["allocation_count"]:
        raise ValueError("allocation count drift")
    if len({block["allocation_id"] for block in schedule}) != len(schedule):
        raise ValueError("allocation IDs must be unique")
    global_positions = {profile_id: set() for profile_id in profile_ids}
    per_seed: dict[int, list[dict[str, Any]]] = {}
    for block in schedule:
        order = block["profile_order"]
        if len(order) != len(profile_ids) or set(order) != profile_set:
            raise ValueError(f"{block['allocation_id']} is not a full permutation")
        per_seed.setdefault(block["seed"], []).append(block)
        for position, profile_id in enumerate(order):
            global_positions[profile_id].add(position)
    for seed in contract["design"]["training_seeds"]:
        blocks = per_seed.get(seed, [])
        if len(blocks) != contract["design"]["allocations_per_seed"]:
            raise ValueError(f"seed {seed} does not have three blocks")
        for profile_id in profile_ids:
            positions = {
                block["profile_order"].index(profile_id) for block in blocks
            }
            if len(positions) != len(blocks):
                raise ValueError(
                    f"seed {seed} repeats a position for {profile_id}"
                )
    if any(len(positions) != len(schedule) for positions in global_positions.values()):
        raise ValueError("global partial-Williams rows repeat a profile position")
    expected_digest = contract["design"]["schedule_sha256"]
    actual_digest = sha256_value(schedule)
    if expected_digest != "TO_BE_FROZEN" and actual_digest != expected_digest:
        raise RuntimeError(
            f"schedule digest drift: {actual_digest} != {expected_digest}"
        )


def validate_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def load_checkpoint_artifact(
    path: Path,
    expected_file_sha256: str,
    contract: dict[str, Any],
    one: dict[str, Any],
    profiles: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[int, str], dict[str, Any]]]:
    validate_sha256(expected_file_sha256, "checkpoint artifact file digest")
    frozen_inputs = contract["frozen_inputs"]
    if expected_file_sha256 != frozen_inputs["checkpoint_artifact_sha256"]:
        raise RuntimeError(
            "checkpoint artifact digest does not match the preregistered digest"
        )
    actual_file_sha256 = sha256_file(path)
    if actual_file_sha256 != expected_file_sha256:
        raise RuntimeError(
            f"checkpoint artifact drift: {actual_file_sha256} != "
            f"{expected_file_sha256}"
        )
    artifact = json.loads(path.read_text())
    if (
        artifact.get("schema_version") != 1
        or artifact.get("artifact_id")
        != frozen_inputs["checkpoint_artifact_id"]
        or artifact.get("status") != "complete"
        or artifact.get("study_id") != one["study_id"]
        or artifact.get("feeds_final_selection") is not False
        or artifact.get("manual_selection_permitted") is not False
        or artifact.get("winner_selected") is not False
    ):
        raise ValueError("checkpoint artifact identity or policy mismatch")
    if not isinstance(artifact.get("entries"), list):
        raise ValueError("checkpoint artifact entries must be a list")
    source = artifact.get("source", {})
    if (
        source.get("manifest_sha256")
        != frozen_inputs["one_factor_manifest_sha256"]
        or source.get("frozen_plan_sha256")
        != frozen_inputs["one_factor_plan_sha256"]
    ):
        raise ValueError("checkpoint artifact source provenance mismatch")
    validate_sha256(
        source.get("workflow_script_sha256"),
        "checkpoint workflow script",
    )

    trained_profiles = [
        profile
        for profile in profiles
        if profile["execution"] == "train_evaluate_benchmark"
    ]
    expected_trained_keys = {
        (seed, profile["profile_id"])
        for seed in one["design"]["seeds"]
        for profile in trained_profiles
    }
    trained_entries: dict[tuple[int, str], dict[str, Any]] = {}
    profile_by_id = {
        profile["profile_id"]: profile for profile in profiles
    }
    frozen = one["digest_contract"]["frozen_profile_digests"]
    checkpoint_owners: dict[tuple[str, str], tuple[int, str]] = {}
    for entry in artifact["entries"]:
        seed = entry.get("seed")
        profile_id = entry.get("profile_id")
        key = (seed, profile_id)
        if key in trained_entries:
            raise ValueError(f"duplicate checkpoint artifact entry: {key}")
        if key not in expected_trained_keys:
            raise ValueError(f"unexpected checkpoint artifact entry: {key}")
        if entry.get("entry_id") != f"{profile_id}__seed_{seed}":
            raise ValueError(f"{key}: training entry ID mismatch")
        if (
            entry.get("feeds_final_selection") is not False
            or entry.get("sdk_status") != "Complete"
            or entry.get("slurm_state") != "COMPLETED"
            or entry.get("slurm_exit_code") != "0:0"
        ):
            raise ValueError(f"{key}: training completion evidence mismatch")
        checkpoint = entry.get("checkpoint", {})
        checkpoint_path = checkpoint.get("path")
        checkpoint_sha256 = checkpoint.get("sha256")
        if (
            not isinstance(checkpoint_path, str)
            or not checkpoint_path.startswith("/lustre/")
            or checkpoint.get("epoch") != 9
            or not isinstance(checkpoint.get("size_bytes"), int)
            or checkpoint["size_bytes"] <= 0
        ):
            raise ValueError(f"{key}: checkpoint must be an absolute Lustre path")
        validate_sha256(checkpoint_sha256, f"{key} checkpoint")
        model_digest = profile_by_id[profile_id]["resolved_model_spec_sha256"]
        if entry.get("resolved_model_spec_sha256") != model_digest:
            raise ValueError(f"{key}: resolved model digest mismatch")
        train_digest = frozen[profile_id][
            "resolved_train_spec_sha256_by_seed"
        ][str(seed)]
        if entry.get("resolved_train_spec_sha256") != train_digest:
            raise ValueError(f"{key}: resolved train digest mismatch")
        identity = (checkpoint_path, checkpoint_sha256)
        owner = checkpoint_owners.setdefault(identity, key)
        if owner != key:
            raise ValueError(
                f"{key}: trained checkpoint is already owned by {owner}"
            )
        trained_entries[key] = {
            "profile_id": profile_id,
            "seed": seed,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_source_profile_id": profile_id,
            "checkpoint_source_entry_id": entry["entry_id"],
            "resolved_model_spec_sha256": model_digest,
            "resolved_train_spec_sha256": train_digest,
            "training_artifact_entry": copy.deepcopy(entry),
        }
    if set(trained_entries) != expected_trained_keys:
        missing = sorted(expected_trained_keys - set(trained_entries))
        raise ValueError(f"checkpoint artifact incomplete: {missing}")

    expanded_entries: dict[tuple[int, str], dict[str, Any]] = {}
    reference_id = one["reference"]["profile_id"]
    for seed in one["design"]["seeds"]:
        reference = trained_entries[(seed, reference_id)]
        for profile in profiles:
            key = (seed, profile["profile_id"])
            if profile["execution"] == "train_evaluate_benchmark":
                expanded_entries[key] = copy.deepcopy(trained_entries[key])
                continue
            train_digest = frozen[profile["profile_id"]][
                "resolved_train_spec_sha256_by_seed"
            ][str(seed)]
            expanded_entries[key] = {
                "profile_id": profile["profile_id"],
                "seed": seed,
                "checkpoint_path": reference["checkpoint_path"],
                "checkpoint_sha256": reference["checkpoint_sha256"],
                "checkpoint_source_profile_id": reference_id,
                "checkpoint_source_entry_id": (
                    f"{reference_id}__seed_{seed}"
                ),
                "resolved_model_spec_sha256": profile[
                    "resolved_model_spec_sha256"
                ],
                "resolved_train_spec_sha256": train_digest,
                "training_artifact_entry": copy.deepcopy(
                    reference["training_artifact_entry"]
                ),
            }
    return artifact, expanded_entries


def load_accuracy_artifact(
    path: Path,
    expected_file_sha256: str,
    checkpoint_artifact_sha256: str,
    contract: dict[str, Any],
    one: dict[str, Any],
    profiles: list[dict[str, Any]],
    checkpoint_entries: dict[tuple[int, str], dict[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[int, str], dict[str, Any]]]:
    validate_sha256(expected_file_sha256, "accuracy artifact file digest")
    actual_file_sha256 = sha256_file(path)
    if actual_file_sha256 != expected_file_sha256:
        raise RuntimeError(
            f"accuracy artifact drift: {actual_file_sha256} != "
            f"{expected_file_sha256}"
        )
    artifact = json.loads(path.read_text())
    frozen_inputs = contract["frozen_inputs"]
    if (
        artifact.get("schema_version") != 1
        or artifact.get("artifact_id") != frozen_inputs["accuracy_artifact_id"]
        or artifact.get("status") != "complete"
        or artifact.get("study_id") != one["study_id"]
        or artifact.get("feeds_final_selection") is not False
        or artifact.get("manual_selection_permitted") is not False
        or artifact.get("winner_selected") is not False
        or artifact.get("selection", {}).get("performed") is not False
    ):
        raise ValueError("accuracy artifact identity or policy mismatch")
    source = artifact.get("source", {})
    if (
        source.get("manifest_sha256")
        != frozen_inputs["one_factor_manifest_sha256"]
        or source.get("frozen_training_plan_sha256")
        != frozen_inputs["one_factor_plan_sha256"]
        or source.get("checkpoint_artifact_sha256")
        != checkpoint_artifact_sha256
    ):
        raise ValueError("accuracy artifact source provenance mismatch")
    for key in (
        "evaluation_plan_sha256",
        "evaluation_submissions_sha256",
        "workflow_script_sha256",
    ):
        validate_sha256(source.get(key), f"accuracy artifact {key}")

    expected_keys = {
        (seed, profile["profile_id"])
        for seed in one["design"]["seeds"]
        for profile in profiles
    }
    entries: dict[tuple[int, str], dict[str, Any]] = {}
    profile_by_id = {
        profile["profile_id"]: profile for profile in profiles
    }
    for entry in artifact.get("entries", []):
        seed = entry.get("seed")
        profile_id = entry.get("profile_id")
        key = (seed, profile_id)
        if key in entries or key not in expected_keys:
            raise ValueError(f"duplicate or unexpected accuracy entry: {key}")
        expected_checkpoint = checkpoint_entries[key]
        checkpoint = entry.get("checkpoint", {})
        if (
            checkpoint.get("path")
            != expected_checkpoint["checkpoint_path"]
            or checkpoint.get("sha256")
            != expected_checkpoint["checkpoint_sha256"]
            or entry.get("checkpoint_source_entry_id")
            != expected_checkpoint["checkpoint_source_entry_id"]
        ):
            raise ValueError(f"{key}: accuracy checkpoint provenance mismatch")
        if (
            entry.get("resolved_model_spec_sha256")
            != profile_by_id[profile_id]["resolved_model_spec_sha256"]
        ):
            raise ValueError(f"{key}: accuracy model digest mismatch")
        if (
            entry.get("feeds_final_selection") is not False
            or entry.get("sdk_status") != "Complete"
            or entry.get("slurm_state") != "COMPLETED"
            or entry.get("slurm_exit_code") != "0:0"
        ):
            raise ValueError(f"{key}: accuracy completion evidence mismatch")
        map50 = entry.get("mAP50")
        if (
            isinstance(map50, bool)
            or not isinstance(map50, (int, float))
            or not math.isfinite(float(map50))
            or not 0.0 <= float(map50) <= 1.0
        ):
            raise ValueError(f"{key}: mAP50 must be finite and in [0,1]")
        entries[key] = copy.deepcopy(entry)
    if set(entries) != expected_keys:
        missing = sorted(expected_keys - set(entries))
        raise ValueError(f"accuracy artifact incomplete: {missing}")

    reference_id = one["reference"]["profile_id"]
    for seed in one["design"]["seeds"]:
        reference_map50 = float(entries[(seed, reference_id)]["mAP50"])
        for profile in profiles:
            entry = entries[(seed, profile["profile_id"])]
            threshold = 0.98 * reference_map50
            retention = entry.get("same_seed_accuracy_retention", {})
            expected_reference_entry = f"{reference_id}__seed_{seed}"
            if (
                retention.get("reference_entry_id")
                != expected_reference_entry
                or not math.isclose(
                    float(retention.get("reference_mAP50", math.nan)),
                    reference_map50,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    float(retention.get("required_mAP50", math.nan)),
                    threshold,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or retention.get("retention_fraction") != 0.98
                or retention.get("passes")
                != (float(entry["mAP50"]) >= threshold)
            ):
                raise ValueError(
                    f"{seed}/{profile['profile_id']}: same-seed retention drift"
                )
    return artifact, entries
