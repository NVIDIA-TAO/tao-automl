#!/usr/bin/env python3

"""Seal the Mask Grounding DINO/COCO2017 campaign after immutable prerequisites exist."""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tao_automl.ptm_registry import canonical_sha256
from tao_automl.selection import canonical_spec_fingerprint

from . import campaign_contract


HERE = Path(__file__).resolve().parent
DEFAULT_REPOSITORY = Path("/localhome/local-rarunachalam/tao-automl")
DEFAULT_WHEEL = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/wheels/35972c1/"
    "nvidia_tao_automl-0.1.0-py3-none-any.whl"
)
DEFAULT_SDK = Path(
    "/localhome/local-rarunachalam/.tao/worktrees/"
    "tao-sdk-bounded-self-requeue"
)
DEFAULT_SKILLS = Path(
    "/localhome/local-rarunachalam/.tao/worktrees/"
    "tao-skills-release-7.1.0"
)
DEFAULT_DATASET_MANIFEST = Path(
    "/localhome/local-rarunachalam/.tao/datasets/"
    "cross_model_automl_20260729/manifests/"
    "coco2017_instance_panoptic_v1.FILE_MANIFEST.sha256"
)
DEFAULT_STAGE_MANIFEST = (
    DEFAULT_REPOSITORY
    / "experiments/cross_model_automl_20260729/"
    "segmentation_datasets/dataset_stage_manifest.v1.json"
)
DEFAULT_CONTIGUOUS_VALIDATION_MANIFEST = (
    HERE / "coco2017_contiguous_validation.v1.json"
)
DEFAULT_TEXT_ENCODER_STAGE = (
    DEFAULT_REPOSITORY
    / "experiments/cross_model_automl_20260729/"
    "grounding_dino_shared_detection/runtime_inputs.stage.v1.json"
)
DEFAULT_QUALIFICATION = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask_grounding_dino_coco2017_ptm_qualification_v5/completion.json"
)
DEFAULT_QUALIFICATION_CONTRACT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask_grounding_dino_coco2017_ptm_qualification_v5/qualification.v5.json"
)
DEFAULT_PREDECESSOR_QUALIFICATION = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask_grounding_dino_coco2017_ptm_qualification_v2/completion.json"
)
DEFAULT_PTM_STAGE_MANIFEST = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask_grounding_dino_coco2017_ptm_qualification_v1/ptm_stage_manifest.json"
)
EXPECTED_DATASET_FILE_MANIFEST_SHA256 = (
    "10566a60498de9998154f44a34445a488c9f030e09f2a7346d20a4a1c55f804e"
)
EXPECTED_STAGE_MANIFEST_SHA256 = (
    "437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d"
)
EXPECTED_CONTIGUOUS_VALIDATION_MANIFEST_SHA256 = (
    "3c2d09d20211017575a2c51a6797ef91f1939340d978a5d11d1d1edab1a30b2d"
)
EXPECTED_TEXT_ENCODER_STAGE_SHA256 = (
    "ac5b6c12bc7d5abd06beaeb61c79426a6f917d4671551fce202fa63fe6dbe160"
)
EXPECTED_WHEEL_SHA256 = (
    "304824dc95ee0ef763ae72f8872e79e593613a36a17e20dbf50b3a561892b381"
)
EXPECTED_SDK_COMMIT = "98c1144fd57b28f38ab5b7b41c113fac6e5e670a"
EXPECTED_SKILLS_COMMIT = "2e9c1b25f3c7cb1ae444c75652e36c47eace8229"
EXPECTED_PREDECESSOR_QUALIFICATION_SHA256 = (
    "35e2d52317ab8458cbfa4efdf8bfa320f083aa19daaaf822b7cde9ed39dfe0eb"
)


class ManifestGenerationError(RuntimeError):
    """The campaign cannot be sealed from the supplied artifacts."""


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def dataset_record(
    manifest: str | Path,
    stage_manifest: str | Path = DEFAULT_STAGE_MANIFEST,
    contiguous_validation_manifest: str | Path = (
        DEFAULT_CONTIGUOUS_VALIDATION_MANIFEST
    ),
) -> dict[str, Any]:
    path = Path(manifest).resolve()
    if (
        not path.is_file()
        or campaign_contract.sha256_file(path)
        != EXPECTED_DATASET_FILE_MANIFEST_SHA256
    ):
        raise ManifestGenerationError(
            "canonical COCO2017 file manifest is unavailable or changed"
        )
    stage_path = Path(stage_manifest).resolve()
    if (
        not stage_path.is_file()
        or campaign_contract.sha256_file(stage_path)
        != EXPECTED_STAGE_MANIFEST_SHA256
    ):
        raise ManifestGenerationError(
            "final segmentation dataset stage manifest is unavailable or changed"
        )
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    contiguous_path = Path(contiguous_validation_manifest).resolve()
    if (
        not contiguous_path.is_file()
        or campaign_contract.sha256_file(contiguous_path)
        != EXPECTED_CONTIGUOUS_VALIDATION_MANIFEST_SHA256
    ):
        raise ManifestGenerationError(
            "contiguous COCO validation manifest is unavailable or changed"
        )
    contiguous = json.loads(contiguous_path.read_text(encoding="utf-8"))
    execution = stage.get("execution_contract", {})
    validation = stage.get("validation", {})
    coco = stage.get("datasets", {}).get("coco2017", {})
    file_manifest = coco.get("file_manifest", {})
    source_archives = coco.get("source_archives", {})
    odvg = coco.get("odvg_projection", {})
    contiguous_source = contiguous.get("source", {})
    contiguous_output = contiguous.get("output", {})
    contiguous_verification = contiguous.get("verification", {})
    if (
        stage.get("schema_version") != 1
        or execution.get("data_only") is not True
        or execution.get("model_invoked") is not False
        or execution.get("cpu_model_smoke_run") is not False
        or execution.get("gpu_model_smoke_run") is not False
        or execution.get("training_run") is not False
        or execution.get("evaluation_run") is not False
        or execution.get("latency_benchmark_run") is not False
        or execution.get("slurm_job_submitted") is not False
        or validation.get("coco2017_subreport_content_sha256")
        != "deced9d6766344fe6fc69cd9de3bcff2cba456a14b3391d07bcedb74c250909e"
        or coco.get("dataset_id")
        != "coco2017_instance_panoptic"
        or coco.get("lustre_root")
        != (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
            "cross_model_automl_20260729/"
            "coco2017_instance_panoptic_v1"
        )
        or coco.get("task_scope", {}).get("mask_grounding_dino")
        != "category_prompted_grounded_instance_segmentation"
        or coco.get("splits", {}).get("train_images") != 118287
        or coco.get("splits", {}).get("val_images") != 5000
        or coco.get("splits", {}).get("train_instance_annotations") != 860001
        or coco.get("splits", {}).get("val_instance_annotations") != 36781
        or coco.get("categories", {}).get("instance_things") != 80
        or coco.get("tao_assets", {}).get("instance_label_map", {}).get(
            "categories"
        )
        != 80
        or coco.get("tao_assets", {}).get("instance_label_map", {}).get(
            "sha256"
        )
        != "67f15c4dd7d52aa73025da8307dec17e907f13db6d5d82332a670f73da68c306"
        or odvg.get("converter_commit")
        != "dcea3a39bd3e4709e2325e4b61a4f179efebde4c"
        or odvg.get("official_function")
        != (
            "nvidia_tao_ds.annotations.conversion.coco_to_odvg."
            "convert_coco_to_odvg"
        )
        or odvg.get("projected_images") != 117266
        or odvg.get("projected_annotations") != 860001
        or odvg.get("masks_preserved_exactly") != 860001
        or odvg.get("jsonl_sha256")
        != "d5deb4f5cfe027786fb1ceb52632ad6d3ef027e95e434525ba715d6841fb2921"
        or odvg.get("label_map_sha256")
        != "02075d96f6bf06d061f9329b4775dc7c3bb5ac140c77bc5c0e465d305c46d6c1"
        or contiguous.get("schema_version") != 1
        or contiguous.get("scope")
        != "mask_grounding_dino_category_prompted_od_validation_annotations"
        or contiguous.get("execution_contract", {}).get(
            "model_invoked"
        )
        is not False
        or contiguous.get("execution_contract", {}).get(
            "slurm_job_submitted"
        )
        is not False
        or contiguous_source.get("dataset_stage_manifest_sha256")
        != EXPECTED_STAGE_MANIFEST_SHA256
        or contiguous_source.get("sha256")
        != "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
        or contiguous_output.get("lustre_path")
        != campaign_contract.FROZEN_CONTIGUOUS_VALIDATION_JSON
        or contiguous_output.get("sha256")
        != campaign_contract.FROZEN_CONTIGUOUS_VALIDATION_SHA256
        or contiguous_output.get("category_ids") != list(range(80))
        or contiguous_output.get("remote_read_only") is not True
        or contiguous_verification.get("annotations_preserved_except_category_id")
        != 36781
        or contiguous_verification.get("segmentations_preserved_exactly")
        != 36781
        or contiguous_verification.get("repeat_conversion_byte_identical")
        is not True
        or contiguous_verification.get("remote_sha256sum_check") != "passed"
        or source_archives.get("all_archive_integrity_checks_passed") is not True
        or source_archives.get("annotations_trainval2017.zip", {}).get("sha256")
        != "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
        or source_archives.get(
            "panoptic_annotations_trainval2017.zip", {}
        ).get("sha256")
        != "c05f76d2129b6b561eb70efe16e7006df62f73fb92889132d373b9d90e31a370"
        or source_archives.get("train2017.zip", {}).get("sha256")
        != "69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929"
        or source_archives.get("val2017.zip", {}).get("sha256")
        != "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05"
        or file_manifest.get("entries") != 246593
        or file_manifest.get("sha256")
        != EXPECTED_DATASET_FILE_MANIFEST_SHA256
        or file_manifest.get("remote_sha256sum_check") != "passed"
        or file_manifest.get("remote_file_set_check") != "passed"
        or coco.get("remote_read_only") is not True
        or coco.get("remote_writable_entries_after_lock") != 0
        or stage.get("transfer_provenance", {}).get(
            "remote_bytes_verified_against_local_manifest"
        )
        is not True
    ):
        raise ManifestGenerationError(
            "final COCO2017 stage provenance does not pass the frozen contract"
        )
    return {
        "id": "coco2017_full_category_prompted_grounded_instance_segmentation",
        "official_sources": {
            name: item["url"] for name, item in source_archives.items()
            if isinstance(item, dict) and "url" in item
        },
        "license": "Creative Commons Attribution 4.0 for COCO annotations",
        "prepared_root": (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
            "cross_model_automl_20260729/"
            "coco2017_instance_panoptic_v1"
        ),
        "train_image_count": 118287,
        "validation_image_count": 5000,
        "train_instance_annotations": 860001,
        "validation_instance_annotations": 36781,
        "num_classes": 80,
        "train_instance_json_sha256": (
            "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d"
        ),
        "validation_instance_json_sha256": (
            "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
        ),
        "label_map_sha256": (
            "67f15c4dd7d52aa73025da8307dec17e907f13db6d5d82332a670f73da68c306"
        ),
        "train_odvg_jsonl_sha256": odvg["jsonl_sha256"],
        "train_odvg_label_map_sha256": odvg["label_map_sha256"],
        "train_odvg_projected_images": odvg["projected_images"],
        "train_odvg_projected_annotations": odvg[
            "projected_annotations"
        ],
        "train_odvg_masks_preserved": odvg["masks_preserved_exactly"],
        "contiguous_validation_json_path": contiguous_output["lustre_path"],
        "contiguous_validation_json_sha256": contiguous_output["sha256"],
        "contiguous_validation_manifest_path": (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
            "cross_model_automl_20260729/"
            "mask_grounding_dino_coco2017_od_v1/"
            "coco2017_contiguous_validation.v1.json"
        ),
        "contiguous_validation_manifest_local_path": str(contiguous_path),
        "contiguous_validation_manifest_sha256": (
            EXPECTED_CONTIGUOUS_VALIDATION_MANIFEST_SHA256
        ),
        "contiguous_validation_image_count": contiguous_output["image_count"],
        "contiguous_validation_annotation_count": contiguous_output[
            "annotation_count"
        ],
        "contiguous_validation_category_ids": contiguous_output[
            "category_ids"
        ],
        "contiguous_validation_remote_read_only": True,
        "official_archive_sha256": {
            name: item["sha256"] for name, item in source_archives.items()
            if isinstance(item, dict) and "sha256" in item
        },
        "content_sha256": (
            "deced9d6766344fe6fc69cd9de3bcff2cba456a14b3391d07bcedb74c250909e"
        ),
        "manifest_path": str(path),
        "manifest_sha256": EXPECTED_DATASET_FILE_MANIFEST_SHA256,
        "file_manifest_entry_count": 246593,
        "remote_sha256sum_check": "passed_all_246593",
        "stage_manifest_path": str(stage_path),
        "stage_manifest_lustre_path": (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
            "cross_model_automl_20260729/coco2017_instance_panoptic_v1/"
            "dataset_stage_manifest.v1.json"
        ),
        "stage_manifest_sha256": EXPECTED_STAGE_MANIFEST_SHA256,
        "remote_file_manifest_path": file_manifest["lustre_path"],
        "remote_read_only": True,
        "remote_writable_entries_after_lock": 0,
    }


def ptm_stage_record(path: str | Path) -> dict[str, Any]:
    """Validate and bind the immutable official Mask Grounding DINO PTM stage."""
    stage_path = Path(path).resolve()
    if not stage_path.is_file():
        raise ManifestGenerationError(
            f"PTM stage manifest is unavailable: {stage_path}"
        )
    document = json.loads(stage_path.read_text(encoding="utf-8"))
    supplied = document.get("manifest_sha256")
    payload = copy.deepcopy(document)
    payload.pop("manifest_sha256", None)
    snapshot = campaign_contract.mask_grounding_dino_registry_snapshot()
    records = document.get("checkpoints")
    if (
        supplied != canonical_sha256(payload)
        or document.get("schema_version") != 1
        or document.get("model") != "mask_grounding_dino"
        or not isinstance(document.get("registry_sha256"), str)
        or len(document["registry_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in document["registry_sha256"]
        )
        or document.get("stage_complete") is not True
        or document.get("remote_read_only") is not True
        or document.get("cpu_model_runs") != 0
        or document.get("smoke_model_runs") != 0
        or document.get("mini_step_runs") != 0
        or not isinstance(records, list)
    ):
        raise ManifestGenerationError(
            "official Mask Grounding DINO PTM stage is incomplete or changed"
        )
    by_id = {
        item.get("id"): item
        for item in records
        if isinstance(item, dict)
    }
    expected = {
        item["id"]: item for item in snapshot["records"]
    }
    if set(by_id) != set(expected) or len(records) != len(expected):
        raise ManifestGenerationError(
            "PTM stage must contain every and only official Mask Grounding DINO arm"
        )
    for checkpoint_id, registry_record in expected.items():
        item = by_id[checkpoint_id]
        digest = item.get("sha256")
        if (
            not isinstance(item.get("path"), str)
            or not item["path"].startswith("/lustre/")
            or isinstance(item.get("size_bytes"), bool)
            or item.get("size_bytes")
            != registry_record["expected_size_bytes"]
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or item.get("immutable_source_identity")
            != registry_record["source"]["immutable_identity"]
            or (
                registry_record.get("sha256") is not None
                and digest != registry_record["sha256"]
            )
            or item.get("remote_read_only") is not True
        ):
            raise ManifestGenerationError(
                f"official PTM stage record changed: {checkpoint_id}"
            )
    return {
        "path": str(stage_path),
        "sha256": campaign_contract.sha256_file(stage_path),
        "manifest_sha256": supplied,
        "checkpoint_ids": sorted(by_id),
        "checkpoints": copy.deepcopy(records),
    }


def _lower_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ManifestGenerationError(f"{name} must be lowercase SHA-256")
    return value


def resume_predecessor_record(path: str | Path) -> dict[str, Any]:
    """Bind an interrupted campaign whose workspaces may be resumed exactly."""
    contract_path = Path(path).resolve()
    try:
        document = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestGenerationError(
            "resume predecessor contract is unavailable or invalid"
        ) from exc
    payload = copy.deepcopy(document)
    supplied = payload.pop("contract_sha256", None)
    if (
        supplied != canonical_sha256(payload)
        or document.get("model") != "mask_grounding_dino"
        or document.get("campaign_id")
        != "mask_grounding_dino-coco2017-objective-aware-three-mode-v5-20260803"
        or document.get("search", {}).get("space_sha256")
        != canonical_sha256(campaign_contract.SEARCH_SPACE)
        or document.get("search", {}).get("candidate_budget_per_mode")
        != campaign_contract.FROZEN_CANDIDATE_BUDGET
        or any(document.get("agent_intervention_flags", {}).values())
    ):
        raise ManifestGenerationError(
            "resume predecessor is not the frozen Mask Grounding DINO campaign"
        )
    return {
        "schema_version": 1,
        "kind": "evaluator_overlay_only_successor",
        "path": str(contract_path),
        "file_sha256": campaign_contract.sha256_file(contract_path),
        "contract_sha256": supplied,
        "campaign_id": document["campaign_id"],
        "source_commit": document["runtime"]["source_commit"],
        "workspace_reuse_allowed": True,
        "training_job_reuse_required": True,
        "recommendation_change_allowed": False,
        "training_relaunch_allowed": False,
        "objective_policy_change_allowed": False,
    }


def first_candidate_training_reuse_record(
    predecessor_contract: str | Path,
    predecessor_runtime_root: str | Path,
) -> dict[str, Any]:
    """Bind the exact completed rec-0 training jobs for a clean successor."""
    contract_path = Path(predecessor_contract).resolve()
    runtime_root = Path(predecessor_runtime_root).resolve()
    try:
        predecessor = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestGenerationError(
            "first-candidate predecessor contract is unavailable or invalid"
        ) from exc
    payload = copy.deepcopy(predecessor)
    predecessor_sha256 = payload.pop("contract_sha256", None)
    if (
        predecessor_sha256 != canonical_sha256(payload)
        or predecessor.get("model") != "mask_grounding_dino"
        or predecessor.get("campaign_id")
        != "mask_grounding_dino-coco2017-objective-aware-three-mode-v5-20260803"
        or any(predecessor.get("agent_intervention_flags", {}).values())
    ):
        raise ManifestGenerationError(
            "first-candidate predecessor is not the frozen MGD campaign"
        )

    modes: dict[str, Any] = {}
    for mode in campaign_contract.MODES:
        mode_root = runtime_root / mode
        evidence_path = mode_root / "candidate_evidence.json"
        state_db = mode_root / "slurm_state.db"
        try:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ManifestGenerationError(
                f"{mode} predecessor candidate evidence is unavailable"
            ) from exc
        candidates = evidence.get("candidates")
        candidate_id = f"{mode}_rec_0"
        candidate = (
            candidates.get(candidate_id)
            if isinstance(candidates, Mapping)
            else None
        )
        if (
            evidence.get("contract_sha256") != predecessor_sha256
            or evidence.get("mode") != mode
            or not isinstance(candidate, Mapping)
            or candidate.get("rec_id") != "0"
            or candidate.get("status") != "terminal_failure"
            or candidate.get("automl_status") != "failure"
            or not str(candidate.get("failure_reason", "")).startswith(
                "required_eval_fn_failed:latency job "
            )
            or not isinstance(candidate.get("terminal_checkpoint"), Mapping)
            or not isinstance(candidate.get("specs"), Mapping)
            or candidate.get("candidate_fingerprint")
            != canonical_spec_fingerprint(candidate["specs"])
            or not isinstance(candidate.get("train_job_id"), str)
        ):
            raise ManifestGenerationError(
                f"{mode} rec-0 is not the expected latency-only failure"
            )
        later = [
            value
            for key, value in candidates.items()
            if key != candidate_id
        ]
        if any(
            not isinstance(value, Mapping)
            or value.get("status") not in {"terminal_failure", "recommended"}
            or "objective_values" in value
            for value in later
        ):
            raise ManifestGenerationError(
                f"{mode} contains an observation that cannot be discarded"
            )
        if not state_db.is_file():
            raise ManifestGenerationError(
                f"{mode} predecessor SLURM state is unavailable"
            )
        with sqlite3.connect(state_db) as connection:
            row = connection.execute(
                "SELECT status, results_dir FROM jobs WHERE job_id = ?",
                (candidate["train_job_id"],),
            ).fetchone()
        if row is None or row[0] != "Complete":
            raise ManifestGenerationError(
                f"{mode} rec-0 training job is not durably complete"
            )
        results_dir = str(row[1])
        results_path = (
            results_dir.removeprefix("lustre://")
            if results_dir.startswith("lustre://")
            else results_dir
        )
        checkpoint = dict(candidate["terminal_checkpoint"])
        if (
            not checkpoint.get("path", "").startswith(
                results_path.rstrip("/") + "/"
            )
            or not isinstance(checkpoint.get("size_bytes"), int)
            or checkpoint["size_bytes"] < 1
            or not isinstance(checkpoint.get("sha256"), str)
            or len(checkpoint["sha256"]) != 64
        ):
            raise ManifestGenerationError(
                f"{mode} rec-0 checkpoint is outside its training job"
            )
        modes[mode] = {
            "candidate_id": candidate_id,
            "rec_id": "0",
            "candidate_fingerprint": candidate["candidate_fingerprint"],
            "checkpoint_id": candidate["checkpoint_id"],
            "specs_sha256": canonical_sha256(candidate["specs"]),
            "recommendation_audit_sha256": candidate[
                "recommendation_audit"
            ]["audit_sha256"],
            "source_train_job_id": candidate["train_job_id"],
            "source_results_dir": results_dir,
            "terminal_checkpoint": checkpoint,
            "source_state_file": str((mode_root / "slurm_state.json").resolve()),
            "source_state_db": str(state_db.resolve()),
            "source_state_db_sha256": campaign_contract.sha256_file(state_db),
            "source_candidate_evidence": str(evidence_path.resolve()),
            "source_candidate_evidence_sha256": campaign_contract.sha256_file(
                evidence_path
            ),
            "discarded_non_observations": len(later),
        }
    value = {
        "schema_version": 1,
        "kind": "first_candidate_completed_training_reuse",
        "source_contract": {
            "path": str(contract_path),
            "file_sha256": campaign_contract.sha256_file(contract_path),
            "contract_sha256": predecessor_sha256,
            "source_commit": predecessor["runtime"]["source_commit"],
        },
        "source_runtime_root": str(runtime_root),
        "modes": modes,
        "fresh_controller_state_required": True,
        "training_relaunch_allowed": False,
        "objective_reuse_allowed": False,
        "evaluation_reuse_allowed": False,
        "latency_reuse_allowed": False,
        "new_training_jobs_submitted": 0,
        "agent_selected_candidate": False,
        "agent_overrode_observation": False,
    }
    value["record_sha256"] = canonical_sha256(value)
    return value


def _successor_qualification_evidence_record(
    evidence_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    """Bind v5 evaluator recovery to the exact frozen v3 training evidence."""
    try:
        source = json.loads(contract_path.read_text(encoding="utf-8"))
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestGenerationError(
            "Mask Grounding DINO v5 qualification JSON is invalid"
        ) from exc
    source_payload = copy.deepcopy(source)
    source_internal = source_payload.pop("contract_sha256", None)
    evidence_payload = copy.deepcopy(evidence)
    evidence_internal = evidence_payload.pop("evidence_sha256", None)
    predecessor = source.get("predecessor", {})
    try:
        training_path = Path(predecessor["completion_path"]).resolve()
        training = json.loads(training_path.read_text(encoding="utf-8"))
    except (KeyError, OSError, ValueError) as exc:
        raise ManifestGenerationError(
            "frozen v3 training qualification is unavailable"
        ) from exc
    training_payload = copy.deepcopy(training)
    training_internal = training_payload.pop("evidence_sha256", None)
    snapshot = campaign_contract.mask_grounding_dino_registry_snapshot()
    expected_ids = {record["id"] for record in snapshot["records"]}
    source_workflows = training.get("workflows")
    recovery_workflows = evidence.get("workflows")
    if (
        source_internal != canonical_sha256(source_payload)
        or source.get("campaign_id")
        != "mask_grounding_dino-coco2017-coco-metric-recovery-v5-20260802"
        or source.get("model") != "mask_grounding_dino"
        or source.get("task")
        != "category_prompted_grounded_instance_segmentation"
        or source.get("primary_metric") != "segm_val_mAP50_95"
        or source.get("sqsh") != campaign_contract.FROZEN_SQSH
        or source.get("execution", {}).get("scope")
        != "standalone_full_validation_only"
        or source.get("execution", {}).get("training_jobs_submitted") != 0
        or source.get("execution", {}).get("evaluation_jobs_expected") != 4
        or source.get("execution", {}).get("gpus_per_job") != 8
        or any(source.get("agent_intervention_flags", {}).values())
        or evidence_internal != canonical_sha256(evidence_payload)
        or evidence.get("campaign_id") != source.get("campaign_id")
        or evidence.get("contract_sha256") != source_internal
        or evidence.get("model") != "mask_grounding_dino"
        or evidence.get("task")
        != "category_prompted_grounded_instance_segmentation"
        or evidence.get("primary_metric") != "segm_val_mAP50_95"
        or evidence.get("predecessor") != predecessor
        or evidence.get("overlay") != source.get("overlay")
        or evidence.get("training_jobs_submitted") != 0
        or evidence.get("evaluation_jobs_submitted") != 4
        or evidence.get("evaluations_submitted_concurrently") is not True
        or any(evidence.get(name) != 0 for name in (
            "cpu_model_runs", "smoke_model_runs", "mini_step_runs"
        ))
        or evidence.get("selection_invoked") is not False
        or evidence.get("validation_measurements_feed_selection") is not False
        or any(evidence.get("agent_intervention_flags", {}).values())
        or training_internal != canonical_sha256(training_payload)
        or predecessor.get("completion_path") != str(training_path)
        or predecessor.get("completion_file_sha256")
        != campaign_contract.sha256_file(training_path)
        or predecessor.get("evidence_sha256") != training_internal
        or training.get("campaign_id")
        != "mask_grounding_dino-coco2017-direct-full-qualification-v3-20260801"
        or training.get("model") != "mask_grounding_dino"
        or training.get("task")
        != "category_prompted_grounded_instance_segmentation"
        or training.get("primary_metric") != "segm_val_mAP50_95"
        or not isinstance(source_workflows, list)
        or not isinstance(recovery_workflows, list)
        or len(source_workflows) != 4
        or len(recovery_workflows) != 4
        or {item.get("checkpoint_id") for item in source_workflows}
        != expected_ids
        or {item.get("checkpoint_id") for item in recovery_workflows}
        != expected_ids
    ):
        raise ManifestGenerationError(
            "Mask Grounding DINO v5 qualification identity changed"
        )
    for workflow in recovery_workflows:
        payload = copy.deepcopy(workflow)
        workflow_sha = payload.pop("workflow_sha256", None)
        evaluation = workflow.get("evaluation_job", {})
        if (
            workflow_sha != canonical_sha256(payload)
            or workflow.get("status") != "success"
            or workflow.get("training_reused") is not True
            or workflow.get("training_jobs_submitted") != 0
            or workflow.get("metric_sanity_gate_passed") is not True
            or evaluation.get("status") != "Complete"
            or evaluation.get("nodes") != 1
            or evaluation.get("gpus") != 8
            or any(workflow.get("agent_intervention_flags", {}).values())
        ):
            raise ManifestGenerationError(
                "Mask Grounding DINO v5 recovery workflow is invalid"
            )
    for workflow in source_workflows:
        diagnostics = workflow.get("diagnostics", {})
        if (
            workflow.get("status") != "failure"
            or workflow.get("failure_code") != "task_correct_metric_missing"
            or diagnostics.get("train_job", {}).get("status") != "Complete"
            or diagnostics.get("evaluation_job", {}).get("status") != "Complete"
            or not isinstance(
                diagnostics.get("train_job", {}).get("terminal_checkpoint"),
                dict,
            )
        ):
            raise ManifestGenerationError(
                "frozen v3 training workflow is not the expected metric-only failure"
            )
    for name, value in (
        ("v5 contract SHA-256", source_internal),
        ("v5 evidence SHA-256", evidence_internal),
        ("v3 evidence SHA-256", training_internal),
    ):
        _lower_sha256(value, name)
    runtime = source["runtime"]
    return {
        "schema_version": 2,
        "kind": "direct_full_gpu_qualification_runtime_local_v2",
        "enabled": True,
        "scope": "campaign_local_in_memory_projection",
        "model": "mask_grounding_dino",
        "task": "category_prompted_grounded_instance_segmentation",
        "tao_version": "7.1.0",
        "container_sha256": campaign_contract.FROZEN_SQSH["sha256"],
        "base_registry_version": snapshot["registry_version"],
        "base_registry_sha256": snapshot["registry_sha256"],
        "qualification_file_sha256": campaign_contract.sha256_file(
            evidence_path
        ),
        "qualification_evidence_sha256": evidence_internal,
        "qualification_contract_path": str(contract_path),
        "qualification_contract_file_sha256": (
            campaign_contract.sha256_file(contract_path)
        ),
        "qualification_contract_sha256": source_internal,
        "qualification_campaign_id": source["campaign_id"],
        "qualification_campaign_sha256": training[
            "qualification_campaign_sha256"
        ],
        "qualification_successor_version": 5,
        "training_qualification_path": str(training_path),
        "training_qualification_file_sha256": (
            campaign_contract.sha256_file(training_path)
        ),
        "training_qualification_evidence_sha256": training_internal,
        "training_qualification_contract_path": predecessor["contract_path"],
        "training_qualification_contract_file_sha256": predecessor[
            "contract_file_sha256"
        ],
        "training_qualification_contract_sha256": predecessor[
            "contract_sha256"
        ],
        "evaluation_recovery_jobs_submitted": 4,
        "training_jobs_submitted": 0,
        "metric_recovery_overlay_sha256": source["overlay"][
            "archive_sha256"
        ],
        "metric_recovery_source_commit": source["overlay"][
            "source_commit"
        ],
        "replacement_workflows_submitted": True,
        "replacement_workflow_count": 4,
        "checkpoint_resume_policy": copy.deepcopy(
            campaign_contract.CHECKPOINT_RESUME_POLICY
        ),
        "predecessor_failure_evidence": copy.deepcopy(
            training["predecessor_failure_evidence"]
        ),
        "ptm_stage_manifest_path": training["ptm_stage_manifest_path"],
        "ptm_stage_manifest_sha256": training[
            "ptm_stage_manifest_sha256"
        ],
        "ptm_stage_content_sha256": runtime["ptm_stage_content_sha256"],
        "qualification_source_commit": runtime["source_commit"],
        "qualification_source_wheel_sha256": runtime["wheel_sha256"],
        "qualification_source_sdk_commit": runtime["sdk_commit"],
        "qualification_source_skills_commit": runtime["skills_commit"],
        "repository_registry_mutation_allowed": False,
        "failed_arm_promotion_allowed": False,
        "unsupported_arm_promotion_allowed": False,
        "agent_override_allowed": False,
    }


def qualification_evidence_record(
    path: str | Path,
    qualification_contract: str | Path | None = None,
) -> dict[str, Any]:
    """Bind immutable direct-run evidence, including v5 metric recovery."""
    evidence_path = Path(path).resolve()
    if not evidence_path.is_file():
        raise ManifestGenerationError(
            "completed Mask Grounding DINO qualification is unavailable"
        )
    document = json.loads(evidence_path.read_text(encoding="utf-8"))
    if document.get("campaign_id") == (
        "mask_grounding_dino-coco2017-coco-metric-recovery-v5-20260802"
    ):
        if qualification_contract is None:
            raise ManifestGenerationError(
                "v5 qualification contract is required"
            )
        contract_path = Path(qualification_contract).resolve()
        if not contract_path.is_file():
            raise ManifestGenerationError(
                "v5 qualification contract is unavailable"
            )
        return _successor_qualification_evidence_record(
            evidence_path, contract_path
        )
    supplied = document.get("evidence_sha256")
    payload = copy.deepcopy(document)
    payload.pop("evidence_sha256", None)
    workflows = document.get("workflows")
    if (
        supplied != canonical_sha256(payload)
        or document.get("schema_version") != 1
        or document.get("model") != "mask_grounding_dino"
        or document.get("task")
        != "category_prompted_grounded_instance_segmentation"
        or document.get("primary_metric") != "segm_val_mAP50_95"
        or document.get("sqsh_sha256")
        != campaign_contract.FROZEN_SQSH["sha256"]
        or document.get("cpu_model_runs") != 0
        or document.get("smoke_model_runs") != 0
        or document.get("mini_step_runs") != 0
        or document.get("replacement_workflows_submitted") is not True
        or document.get("replacement_workflow_count") != 4
        or document.get("checkpoint_resume_policy")
        != campaign_contract.CHECKPOINT_RESUME_POLICY
        or not isinstance(
            document.get("predecessor_failure_evidence"), dict
        )
        or document["predecessor_failure_evidence"].get(
            "all_terminal_failures_preserved"
        )
        is not True
        or not isinstance(workflows, list)
        or len(workflows) != 4
        or any(
            not isinstance(item, dict)
            or item.get("terminal") is not True
            or item.get("status") not in {"success", "failure"}
            for item in workflows
        )
    ):
        raise ManifestGenerationError(
            "completed Mask Grounding DINO qualification is invalid"
        )
    for name in (
        "qualification_contract_sha256",
        "qualification_campaign_sha256",
        "registry_sha256",
        "ptm_stage_manifest_sha256",
    ):
        digest = document.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ManifestGenerationError(
                f"qualification {name} is not lowercase SHA-256"
            )
    return {
        "schema_version": 2,
        "kind": "direct_full_gpu_qualification_runtime_local_v2",
        "enabled": True,
        "scope": "campaign_local_in_memory_projection",
        "model": "mask_grounding_dino",
        "task": "category_prompted_grounded_instance_segmentation",
        "tao_version": "7.1.0",
        "container_sha256": campaign_contract.FROZEN_SQSH["sha256"],
        "base_registry_version": (
            campaign_contract.mask_grounding_dino_registry_snapshot()[
                "registry_version"
            ]
        ),
        "base_registry_sha256": document["registry_sha256"],
        "qualification_file_sha256": campaign_contract.sha256_file(
            evidence_path
        ),
        "qualification_evidence_sha256": supplied,
        "qualification_contract_sha256": document[
            "qualification_contract_sha256"
        ],
        "qualification_campaign_sha256": document[
            "qualification_campaign_sha256"
        ],
        "replacement_workflows_submitted": True,
        "replacement_workflow_count": 4,
        "checkpoint_resume_policy": copy.deepcopy(
            campaign_contract.CHECKPOINT_RESUME_POLICY
        ),
        "predecessor_failure_evidence": copy.deepcopy(
            document["predecessor_failure_evidence"]
        ),
        "repository_registry_mutation_allowed": False,
        "failed_arm_promotion_allowed": False,
        "unsupported_arm_promotion_allowed": False,
        "agent_override_allowed": False,
    }


def _runtime(
    *,
    repository: Path,
    wheel: Path,
    sdk: Path,
    skills: Path,
    qualification: Path,
    qualification_contract: Path,
    ptm_stage_manifest: Path,
    text_encoder_stage: Path,
    predecessor_qualification: Path,
) -> dict[str, Any]:
    if not wheel.is_file() or (
        campaign_contract.sha256_file(wheel) != EXPECTED_WHEEL_SHA256
    ):
        raise ManifestGenerationError("production AutoML wheel changed")
    if _git(sdk, "rev-parse", "HEAD") != EXPECTED_SDK_COMMIT:
        raise ManifestGenerationError("TAO SDK commit changed")
    if _git(skills, "rev-parse", "HEAD") != EXPECTED_SKILLS_COMMIT:
        raise ManifestGenerationError("TAO skills commit changed")
    if _git(repository, "status", "--porcelain"):
        raise ManifestGenerationError(
            "AutoML source must be clean before campaign sealing"
        )
    ptm_stage = ptm_stage_record(ptm_stage_manifest)
    if (
        not text_encoder_stage.is_file()
        or campaign_contract.sha256_file(text_encoder_stage)
        != EXPECTED_TEXT_ENCODER_STAGE_SHA256
    ):
        raise ManifestGenerationError(
            "frozen Grounding DINO text-encoder stage changed"
        )
    text_stage = json.loads(
        text_encoder_stage.read_text(encoding="utf-8")
    )
    text_encoder = text_stage.get("text_encoder", {})
    if (
        text_stage.get("execution", {}).get("cpu_model_runs") != 0
        or text_stage.get("execution", {}).get("gpu_model_runs") != 0
        or text_stage.get("execution", {}).get("scheduler_jobs_submitted") != 0
        or text_encoder.get("lustre_root")
        != campaign_contract.FROZEN_TEXT_ENCODER_ROOT
        or text_encoder.get("tree_sha256")
        != campaign_contract.FROZEN_TEXT_ENCODER_TREE_SHA256
        or text_encoder.get("offline_runtime", {}).get("HF_HUB_OFFLINE")
        != "1"
        or text_encoder.get("offline_runtime", {}).get(
            "TRANSFORMERS_OFFLINE"
        )
        != "1"
    ):
        raise ManifestGenerationError(
            "frozen Grounding DINO text-encoder provenance changed"
        )
    predecessor = predecessor_qualification.resolve()
    if (
        not predecessor.is_file()
        or campaign_contract.sha256_file(predecessor)
        != EXPECTED_PREDECESSOR_QUALIFICATION_SHA256
    ):
        raise ManifestGenerationError(
            "preserved v1 qualification evidence is unavailable or changed"
        )
    predecessor_document = json.loads(
        predecessor.read_text(encoding="utf-8")
    )
    predecessor_workflows = predecessor_document.get("workflows")
    if (
        not isinstance(predecessor_workflows, list)
        or len(predecessor_workflows) != 4
        or any(
            not isinstance(item, dict)
            or item.get("status") != "failure"
            or item.get("failure_preserved") is not True
            for item in predecessor_workflows
        )
    ):
        raise ManifestGenerationError(
            "preserved v1 qualification is not the sealed four-arm failure"
        )
    source_commit = _git(repository, "rev-parse", "HEAD")
    eligibility = qualification_evidence_record(
        qualification, qualification_contract
    )
    qualification_source = json.loads(
        qualification_contract.read_text(encoding="utf-8")
    )
    evaluation_overlay = qualification_source.get("overlay")
    if (
        eligibility.get("qualification_successor_version") == 5
        and (
            not isinstance(evaluation_overlay, dict)
            or evaluation_overlay.get("archive_sha256")
            != eligibility.get("metric_recovery_overlay_sha256")
            or evaluation_overlay.get("source_commit")
            != eligibility.get("metric_recovery_source_commit")
        )
    ):
        raise ManifestGenerationError(
            "v5 qualification evaluator overlay identity changed"
        )
    eligibility.update(
        {
            "eligibility_source_commit": source_commit,
            "wheel_sha256": EXPECTED_WHEEL_SHA256,
            "sdk_commit": EXPECTED_SDK_COMMIT,
            "skills_commit": EXPECTED_SKILLS_COMMIT,
        }
    )
    return {
        "repository": str(repository.resolve()),
        "source_commit": source_commit,
        "source_dirty": False,
        "wheel_path": str(wheel.resolve()),
        "wheel_sha256": EXPECTED_WHEEL_SHA256,
        "sdk_dir": str(sdk.resolve()),
        "sdk_commit": EXPECTED_SDK_COMMIT,
        "skills_repository": str(skills.resolve()),
        "skills_commit": EXPECTED_SKILLS_COMMIT,
        "skill_dir": str(
            (
                skills
                / "skills/models/tao-train-mask-grounding-dino"
            ).resolve()
        ),
        "text_encoder_stage_path": str(text_encoder_stage.resolve()),
        "text_encoder_stage_sha256": EXPECTED_TEXT_ENCODER_STAGE_SHA256,
        "text_encoder_root": text_encoder["lustre_root"],
        "text_encoder_tree_sha256": text_encoder["tree_sha256"],
        "text_encoder_files": [
            {
                "path": item["lustre"]["path"],
                "size_bytes": item["lustre"]["size_bytes"],
                "sha256": item["lustre"]["sha256"],
                "mode": item["lustre"]["mode"],
            }
            for item in text_encoder["files"]
        ],
        "offline_environment": copy.deepcopy(
            text_encoder["offline_runtime"]
        ),
        "qualification_evidence_path": str(qualification.resolve()),
        "qualification_contract_path": str(
            qualification_contract.resolve()
        ),
        "qualification_contract_file_sha256": eligibility.get(
            "qualification_contract_file_sha256"
        ),
        "evaluation_overlay": copy.deepcopy(evaluation_overlay),
        "runtime_local_eligibility": eligibility,
        "predecessor_failure_evidence": copy.deepcopy(
            eligibility["predecessor_failure_evidence"]
        ),
        "ptm_stage_manifest_path": ptm_stage["path"],
        "ptm_stage_manifest_sha256": ptm_stage["sha256"],
        "ptm_stage_content_sha256": ptm_stage["manifest_sha256"],
        "partition": "polar3",
        "account": "edgeai_tao-ptm_image-foundation-model-clip",
        "base_results_dir": (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam"
        ),
        "container_mounts": "/lustre",
        "time_hours": 4.0,
        "timeout_hours": 3.8,
        "max_job_retries": campaign_contract.FROZEN_SLURM_RETRY_CAP,
        "hardware_contract": copy.deepcopy(
            campaign_contract.FROZEN_HARDWARE
        ),
    }


def build_contract(
    *,
    repository: str | Path = DEFAULT_REPOSITORY,
    wheel: str | Path = DEFAULT_WHEEL,
    sdk: str | Path = DEFAULT_SDK,
    skills: str | Path = DEFAULT_SKILLS,
    dataset_manifest: str | Path = DEFAULT_DATASET_MANIFEST,
    stage_manifest: str | Path = DEFAULT_STAGE_MANIFEST,
    contiguous_validation_manifest: str | Path = (
        DEFAULT_CONTIGUOUS_VALIDATION_MANIFEST
    ),
    qualification: str | Path = DEFAULT_QUALIFICATION,
    qualification_contract: str | Path = DEFAULT_QUALIFICATION_CONTRACT,
    ptm_stage_manifest: str | Path = DEFAULT_PTM_STAGE_MANIFEST,
    text_encoder_stage: str | Path = DEFAULT_TEXT_ENCODER_STAGE,
    predecessor_qualification: str | Path = (
        DEFAULT_PREDECESSOR_QUALIFICATION
    ),
    resume_predecessor_contract: str | Path | None = None,
    first_candidate_reuse_root: str | Path | None = None,
    first_candidate_reuse_contract: str | Path | None = None,
) -> dict[str, Any]:
    repository_path = Path(repository).resolve()
    runtime = _runtime(
        repository=repository_path,
        wheel=Path(wheel).resolve(),
        sdk=Path(sdk).resolve(),
        skills=Path(skills).resolve(),
        qualification=Path(qualification),
        qualification_contract=Path(qualification_contract),
        ptm_stage_manifest=Path(ptm_stage_manifest),
        text_encoder_stage=Path(text_encoder_stage),
        predecessor_qualification=Path(predecessor_qualification),
    )
    if resume_predecessor_contract is not None:
        predecessor_path = Path(resume_predecessor_contract).resolve()
        predecessor_document = json.loads(
            predecessor_path.read_text(encoding="utf-8")
        )
        runtime["resume_predecessor_contract"] = resume_predecessor_record(
            predecessor_path
        )
        # Resume must reconstruct the byte-identical PTM runtime manifest that
        # the hierarchical brain persisted before interruption.  The source
        # successor changes only evaluator execution; it must not manufacture
        # a new PTM-preflight identity or search configuration.
        runtime["runtime_local_eligibility"] = copy.deepcopy(
            predecessor_document["runtime"]["runtime_local_eligibility"]
        )
    if (first_candidate_reuse_root is None) != (
        first_candidate_reuse_contract is None
    ):
        raise ManifestGenerationError(
            "first-candidate reuse root and contract must be supplied together"
        )
    if first_candidate_reuse_root is not None:
        if resume_predecessor_contract is not None:
            raise ManifestGenerationError(
                "workspace resume and fresh first-candidate reuse conflict"
            )
        runtime["first_candidate_training_reuse"] = (
            first_candidate_training_reuse_record(
                first_candidate_reuse_contract,
                first_candidate_reuse_root,
            )
        )
    value = campaign_contract.build_preregistered_contract(
        campaign_id=(
            "mask_grounding_dino-coco2017-objective-aware-three-mode-v5-20260803"
        ),
        dataset=dataset_record(
            dataset_manifest,
            stage_manifest,
            contiguous_validation_manifest,
        ),
        skill_dir=(
            Path(skills).resolve()
            / "skills/models/tao-train-mask-grounding-dino"
        ),
        runtime=runtime,
    )
    value.pop("contract_sha256")
    value["launcher_integrity"] = {
        "ddp_strategy_audit_sha256": campaign_contract.sha256_file(
            HERE / "ddp_strategy_audit.v2.json"
        ),
        "campaign_contract_sha256": campaign_contract.sha256_file(
            HERE / "campaign_contract.py"
        ),
        "qualification_gate_sha256": campaign_contract.sha256_file(
            HERE / "qualification_gate.py"
        ),
        "qualification_campaign_sha256": campaign_contract.sha256_file(
            HERE / "qualification_campaign.py"
        ),
        "run_campaign_sha256": campaign_contract.sha256_file(
            HERE / "run_campaign.py"
        ),
        "mask_grounding_dino_latency_worker_sha256": (
            campaign_contract.sha256_file(
                HERE / "mask_grounding_dino_latency_worker.py"
            )
        ),
        "checkpoint_resume_sha256": campaign_contract.sha256_file(
            HERE.parent / "checkpoint_resume.py"
        ),
        "manifest_generator_sha256": campaign_contract.sha256_file(
            HERE / "manifest_generator.py"
        ),
    }
    value["contract_sha256"] = canonical_sha256(value)
    return campaign_contract.validate_contract(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument("--wheel", type=Path, default=DEFAULT_WHEEL)
    parser.add_argument("--sdk", type=Path, default=DEFAULT_SDK)
    parser.add_argument("--skills", type=Path, default=DEFAULT_SKILLS)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument(
        "--stage-manifest",
        type=Path,
        default=DEFAULT_STAGE_MANIFEST,
    )
    parser.add_argument(
        "--contiguous-validation-manifest",
        type=Path,
        default=DEFAULT_CONTIGUOUS_VALIDATION_MANIFEST,
    )
    parser.add_argument(
        "--text-encoder-stage",
        type=Path,
        default=DEFAULT_TEXT_ENCODER_STAGE,
    )
    parser.add_argument(
        "--qualification", type=Path, default=DEFAULT_QUALIFICATION
    )
    parser.add_argument(
        "--qualification-contract",
        type=Path,
        default=DEFAULT_QUALIFICATION_CONTRACT,
    )
    parser.add_argument(
        "--predecessor-qualification",
        type=Path,
        default=DEFAULT_PREDECESSOR_QUALIFICATION,
    )
    parser.add_argument(
        "--ptm-stage-manifest",
        type=Path,
        default=DEFAULT_PTM_STAGE_MANIFEST,
    )
    parser.add_argument(
        "--resume-predecessor-contract",
        type=Path,
        default=None,
        help=(
            "Seal an evaluator-overlay-only successor that resumes the exact "
            "predecessor AutoML workspaces and training jobs."
        ),
    )
    parser.add_argument(
        "--first-candidate-reuse-root",
        type=Path,
        default=None,
        help="Fresh-state successor source containing exact completed rec-0 jobs.",
    )
    parser.add_argument(
        "--first-candidate-reuse-contract",
        type=Path,
        default=None,
        help="Contract that produced the reusable rec-0 training jobs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    contract = build_contract(
        repository=args.repository,
        wheel=args.wheel,
        sdk=args.sdk,
        skills=args.skills,
        dataset_manifest=args.dataset_manifest,
        stage_manifest=args.stage_manifest,
        contiguous_validation_manifest=(
            args.contiguous_validation_manifest
        ),
        qualification=args.qualification,
        qualification_contract=args.qualification_contract,
        ptm_stage_manifest=args.ptm_stage_manifest,
        text_encoder_stage=args.text_encoder_stage,
        predecessor_qualification=args.predecessor_qualification,
        resume_predecessor_contract=args.resume_predecessor_contract,
        first_candidate_reuse_root=args.first_candidate_reuse_root,
        first_candidate_reuse_contract=args.first_candidate_reuse_contract,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "contract_sha256": contract["contract_sha256"],
                "launch_authorized": False,
                "reason": (
                    "dynamic direct-full-run PTM qualification and supported "
                    "registry gates are evaluated by the automatic trigger"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
