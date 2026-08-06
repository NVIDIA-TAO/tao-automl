#!/usr/bin/env python3

"""Seal the SegFormer/VOC2012 campaign after immutable prerequisites exist."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from tao_automl.ptm_registry import canonical_sha256

from . import campaign_contract


HERE = Path(__file__).resolve().parent
DEFAULT_REPOSITORY = Path("/localhome/local-rarunachalam/tao-automl")
DEFAULT_WHEEL = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/wheel/segformer-v5-selective-recovery/"
    "nvidia_tao_automl-0.1.0-py3-none-any.whl"
)
DEFAULT_SDK = Path(
    "/localhome/local-rarunachalam/.tao/worktrees/"
    "tao-sdk-slurm-a2e50d0"
)
DEFAULT_SKILLS = Path(
    "/localhome/local-rarunachalam/.tao/worktrees/"
    "tao-skills-release-7.1.0"
)
DEFAULT_DATASET_MANIFEST = Path(
    "/localhome/local-rarunachalam/.tao/datasets/"
    "cross_model_automl_20260729/manifests/"
    "voc2012_segmentation_v1.FILE_MANIFEST.sha256"
)
DEFAULT_STAGE_MANIFEST = (
    DEFAULT_REPOSITORY
    / "experiments/cross_model_automl_20260729/"
    "segmentation_datasets/dataset_stage_manifest.v1.json"
)
DEFAULT_QUALIFICATION = Path(
    campaign_contract.FROZEN_V5_QUALIFICATION_CONTRACT[
        "qualification_evidence_path"
    ]
)
DEFAULT_QUALIFICATION_CONTRACT = Path(
    campaign_contract.FROZEN_V5_QUALIFICATION_CONTRACT["path"]
)
DEFAULT_PTM_STAGE_MANIFEST = Path(
    campaign_contract.FROZEN_V5_QUALIFICATION_CONTRACT[
        "ptm_stage_manifest_path"
    ]
)
DEFAULT_SUCCESSOR_RUNTIME_ROOT = Path(
    campaign_contract.FROZEN_V6_SUCCESSOR_RUNTIME_ROOT
)
DEFAULT_SUCCESSOR_CONTRACT = Path(
    campaign_contract.FROZEN_V6_SUCCESSOR_CONTRACT_PATH
)
EXPECTED_DATASET_FILE_MANIFEST_SHA256 = (
    "051ab20215b8e6976763ac82a3db20a68264759edef3d62fd0c8553c501123ff"
)
EXPECTED_STAGE_MANIFEST_SHA256 = (
    "437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d"
)
EXPECTED_WHEEL_SHA256 = (
    "a5e78903aa7c540a7c13b9b413ed5daf64534df04cc91a21d9480875e7d16f3e"
)
EXPECTED_SDK_COMMIT = "acb3c206bc0e04517ab852914efa9b6c2f7c656c"
EXPECTED_SKILLS_COMMIT = "6d22b2efcd355454448f075945d318f0608ed480"


class ManifestGenerationError(RuntimeError):
    """The campaign cannot be sealed from the supplied artifacts."""


def _lower_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ManifestGenerationError(f"{name} must be lowercase SHA-256")
    return value


def _terminal_completion_sha(document: dict[str, Any]) -> str:
    payload = copy.deepcopy(document)
    supplied = payload.pop("evidence_sha256", None)
    if (
        supplied != canonical_sha256(payload)
        or document.get("terminal") is not True
        or not isinstance(document.get("successful_workflows"), int)
        or isinstance(document.get("successful_workflows"), bool)
        or document["successful_workflows"] < 1
    ):
        raise ManifestGenerationError(
            "SegFormer v5 completion is invalid or has zero successes"
        )
    return _lower_sha256(supplied, "qualification evidence SHA-256")


def qualification_evidence_record(
    path: str | Path,
    qualification_contract: str | Path,
) -> dict[str, Any]:
    """Validate and bind the exact terminal v5 qualification boundary."""
    from .qualification_gate import audit_qualification

    evidence_path = Path(path).resolve()
    contract_path = Path(qualification_contract).resolve()
    frozen = campaign_contract.FROZEN_V5_QUALIFICATION_CONTRACT
    stage_path = Path(frozen["ptm_stage_manifest_path"]).resolve()
    if (
        str(evidence_path) != frozen["qualification_evidence_path"]
        or not evidence_path.is_file()
    ):
        raise ManifestGenerationError(
            "terminal SegFormer v5 qualification evidence is unavailable"
        )
    if (
        str(contract_path) != frozen["path"]
        or not contract_path.is_file()
        or campaign_contract.sha256_file(contract_path)
        != frozen["whole_file_sha256"]
    ):
        raise ManifestGenerationError(
            "immutable SegFormer v5 qualification contract changed"
        )
    if (
        not stage_path.is_file()
        or campaign_contract.sha256_file(stage_path)
        != frozen["ptm_stage_manifest_whole_file_sha256"]
    ):
        raise ManifestGenerationError(
            "immutable SegFormer v5 PTM stage manifest changed"
        )
    try:
        source_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise ManifestGenerationError(
            "SegFormer v5 qualification boundary contains invalid JSON"
        ) from exc

    source_payload = copy.deepcopy(source_contract)
    source_internal = source_payload.pop("contract_sha256", None)
    stage_payload = copy.deepcopy(stage)
    stage_internal = stage_payload.pop("stage_manifest_sha256", None)
    evidence_internal = _terminal_completion_sha(evidence)
    runtime = source_contract.get("runtime", {})
    launchers = source_contract.get("launcher_integrity", {})
    policy = source_contract.get("qualification_policy", {})
    snapshot_records = {
        record["id"]: record["registry_record_sha256"]
        for record in source_contract.get("ptm_inventory", {}).get(
            "records", ()
        )
        if isinstance(record, dict) and "id" in record
    }
    expected_records = {
        record["id"]: record["registry_record_sha256"]
        for record in campaign_contract.segformer_registry_snapshot()[
            "records"
        ]
    }
    if (
        source_internal != canonical_sha256(source_payload)
        or source_internal != frozen["contract_sha256"]
        or source_contract.get("campaign_id") != frozen["campaign_id"]
        or runtime.get("source_commit") != frozen["source_commit"]
        or runtime.get("wheel_sha256") != frozen["wheel_sha256"]
        or runtime.get("sdk_commit") != frozen["sdk_commit"]
        or runtime.get("skills_commit") != frozen["skills_commit"]
        or source_contract.get("ptm_inventory", {}).get("registry_version")
        != frozen["registry_version"]
        or source_contract.get("ptm_inventory", {}).get("registry_sha256")
        != frozen["registry_sha256"]
        or snapshot_records != expected_records
        or launchers.get("qualification_campaign_sha256")
        != frozen["qualification_controller_sha256"]
        or launchers.get("qualification_gate_sha256")
        != frozen["qualification_gate_sha256"]
        or policy.get("campaign_id") != frozen["qualification_campaign_id"]
        or policy.get("qualification_evidence_path") != str(evidence_path)
        or policy.get("ptm_stage_manifest_path") != str(stage_path)
        or stage_internal != canonical_sha256(stage_payload)
        or stage_internal != frozen["ptm_stage_manifest_sha256"]
        or stage.get("automl_contract_sha256") != source_internal
        or stage.get("campaign_id") != frozen["qualification_campaign_id"]
        or stage.get("runtime", {}).get("source_commit")
        != frozen["source_commit"]
        or stage.get("runtime", {}).get("sdk_commit")
        != frozen["sdk_commit"]
        or stage.get("runtime", {}).get("skills_commit")
        != frozen["skills_commit"]
        or evidence.get("campaign_id")
        != frozen["qualification_campaign_id"]
        or evidence.get("automl_contract_sha256") != source_internal
        or evidence.get("qualification_controller_sha256")
        != frozen["qualification_controller_sha256"]
        or evidence.get("source_commit") != frozen["source_commit"]
        or evidence.get("ptm_stage_manifest_path") != str(stage_path)
        or evidence.get("ptm_stage_manifest_sha256") != stage_internal
    ):
        raise ManifestGenerationError(
            "terminal SegFormer v5 source, controller, contract, stage, or "
            "completion identity changed"
        )
    try:
        decision = audit_qualification(evidence_path)
    except Exception as exc:
        raise ManifestGenerationError(
            "SegFormer v5 qualification gate rejected terminal evidence"
        ) from exc
    allowed_without_projection = {
        "registry_not_supported",
        "no_runtime_qualified_ptm",
    }
    unexpected = [
        item
        for item in decision.blockers
        if item.get("code") not in allowed_without_projection
    ]
    if unexpected:
        raise ManifestGenerationError(
            "SegFormer v5 qualification gate found invalid workflow evidence: "
            + ", ".join(str(item.get("code")) for item in unexpected)
        )
    return {
        "schema_version": 1,
        "kind": campaign_contract.RUNTIME_LOCAL_ELIGIBILITY_KIND,
        "enabled": True,
        "scope": "campaign_local_in_memory_projection",
        "model": "segformer",
        "task": "semantic_segmentation",
        "tao_version": "7.1.0",
        "container_sha256": campaign_contract.FROZEN_SQSH["sha256"],
        "base_registry_version": frozen["registry_version"],
        "base_registry_sha256": frozen["registry_sha256"],
        "qualification_evidence_path": str(evidence_path),
        "qualification_file_sha256": campaign_contract.sha256_file(
            evidence_path
        ),
        "qualification_evidence_sha256": evidence_internal,
        "qualification_contract_sha256": frozen["contract_sha256"],
        "qualification_controller_sha256": frozen[
            "qualification_controller_sha256"
        ],
        "license_policy": "complete_existing_registry_metadata_only",
        "checkpoint_spec_file": copy.deepcopy(
            campaign_contract.FROZEN_RUNTIME_LOCAL_CHECKPOINT_SPEC_FILE
        ),
        "repository_registry_mutation_allowed": False,
        "missing_license_normalization_allowed": False,
        "failed_arm_promotion_allowed": False,
        "unsupported_arm_promotion_allowed": False,
        "agent_override_allowed": False,
    }


def wait_for_terminal_qualification(
    path: str | Path,
    qualification_contract: str | Path,
    *,
    status_path: str | Path,
    poll_seconds: float = 30.0,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Wait only for absence; a present invalid completion fails closed."""
    from . import run_campaign

    if poll_seconds < 0 or (
        timeout_seconds is not None and timeout_seconds < 0
    ):
        raise ValueError("automatic successor timing values cannot be negative")
    evidence_path = Path(path).resolve()
    status = Path(status_path).resolve()
    started = time.monotonic()
    attempt = 0
    while not evidence_path.is_file():
        attempt += 1
        run_campaign.atomic_json(
            status,
            {
                "schema_version": 1,
                "automatic_successor": True,
                "state": "waiting_for_terminal_v5_completion",
                "attempt": attempt,
                "qualification_path": str(evidence_path),
                "successor_contract_sealed": False,
                "model_jobs_launched": False,
                "checked_at_utc": run_campaign.utc_timestamp(),
            },
        )
        if (
            timeout_seconds is not None
            and time.monotonic() - started >= timeout_seconds
        ):
            raise TimeoutError(
                "automatic SegFormer successor timed out waiting for v5"
            )
        time.sleep(poll_seconds)
    try:
        record = qualification_evidence_record(
            evidence_path,
            qualification_contract,
        )
    except Exception as exc:
        run_campaign.atomic_json(
            status,
            {
                "schema_version": 1,
                "automatic_successor": True,
                "state": "terminal_v5_evidence_rejected",
                "qualification_path": str(evidence_path),
                "successor_contract_sealed": False,
                "model_jobs_launched": False,
                "checked_at_utc": run_campaign.utc_timestamp(),
                "reason": str(exc),
            },
        )
        raise
    run_campaign.atomic_json(
        status,
        {
            "schema_version": 1,
            "automatic_successor": True,
            "state": "terminal_v5_evidence_accepted",
            "qualification_path": str(evidence_path),
            "qualification_file_sha256": record[
                "qualification_file_sha256"
            ],
            "qualification_evidence_sha256": record[
                "qualification_evidence_sha256"
            ],
            "successor_contract_sealed": False,
            "model_jobs_launched": False,
            "checked_at_utc": run_campaign.utc_timestamp(),
        },
    )
    return record


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _source_seal_identity(repository: Path, wheel: Path) -> dict[str, Any]:
    repository = repository.resolve()
    if repository != HERE.parents[2].resolve():
        raise ManifestGenerationError(
            "automatic watcher must execute from the repository it seals"
        )
    if _git(repository, "status", "--porcelain"):
        raise ManifestGenerationError(
            "automatic watcher source must remain clean"
        )
    if (
        not wheel.is_file()
        or campaign_contract.sha256_file(wheel) != EXPECTED_WHEEL_SHA256
    ):
        raise ManifestGenerationError("automatic watcher wheel changed")
    return {
        "source_commit": _git(repository, "rev-parse", "HEAD"),
        "wheel_sha256": EXPECTED_WHEEL_SHA256,
        "campaign_contract_sha256": campaign_contract.sha256_file(
            HERE / "campaign_contract.py"
        ),
        "qualification_gate_sha256": campaign_contract.sha256_file(
            HERE / "qualification_gate.py"
        ),
        "manifest_generator_sha256": campaign_contract.sha256_file(
            HERE / "manifest_generator.py"
        ),
        "run_campaign_sha256": campaign_contract.sha256_file(
            HERE / "run_campaign.py"
        ),
    }


def dataset_record(
    manifest: str | Path,
    stage_manifest: str | Path = DEFAULT_STAGE_MANIFEST,
) -> dict[str, Any]:
    path = Path(manifest).resolve()
    if (
        not path.is_file()
        or campaign_contract.sha256_file(path)
        != EXPECTED_DATASET_FILE_MANIFEST_SHA256
    ):
        raise ManifestGenerationError(
            "canonical VOC2012 file manifest is unavailable or changed"
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
    execution = stage.get("execution_contract", {})
    validation = stage.get("validation", {})
    voc = stage.get("datasets", {}).get("voc2012", {})
    file_manifest = voc.get("file_manifest", {})
    source_archive = voc.get("source_archive", {})
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
        or validation.get("voc2012_subreport_content_sha256")
        != "815b5d01b625238b449c4bca828bf96107b367f0f4d5d8a31d2f97c6161a5de0"
        or voc.get("dataset_id")
        != "pascal_voc2012_segmentation_trainval"
        or voc.get("prepared_root")
        != (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
            "cross_model_automl_20260729/"
            "voc2012_segmentation_v1/prepared"
        )
        or voc.get("splits")
        != {
            "train_pairs": 1464,
            "val_pairs": 1449,
            "train_val_disjoint": True,
        }
        or voc.get("label_contract", {}).get("num_model_classes") != 21
        or voc.get("label_contract", {}).get("valid_class_ids")
        != list(range(21))
        or voc.get("label_contract", {}).get("ignore_id") != 255
        or voc.get("label_contract", {}).get("label_transform") != "None"
        or source_archive.get("url")
        != (
            "https://thor.robots.ox.ac.uk/pascal/VOC/"
            "voc2012/VOCtrainval_11-May-2012.tar"
        )
        or source_archive.get("sha256")
        != "e14f763270cf193d0b5f74b169f44157a4b0c6efa708f4dd0ff78ee691763bcb"
        or source_archive.get("archive_integrity_passed") is not True
        or file_manifest.get("entries") != 5827
        or file_manifest.get("sha256")
        != EXPECTED_DATASET_FILE_MANIFEST_SHA256
        or file_manifest.get("remote_sha256sum_check") != "passed"
        or file_manifest.get("remote_file_set_check") != "passed"
        or voc.get("remote_read_only") is not True
        or voc.get("remote_writable_entries_after_lock") != 0
        or stage.get("transfer_provenance", {}).get(
            "remote_bytes_verified_against_local_manifest"
        )
        is not True
    ):
        raise ManifestGenerationError(
            "final VOC2012 stage provenance does not pass the frozen contract"
        )
    return {
        "id": "pascal_voc_2012_full_semantic_segmentation",
        "official_source": (
            "https://thor.robots.ox.ac.uk/pascal/VOC/"
            "voc2012/VOCtrainval_11-May-2012.tar"
        ),
        "license": "PASCAL VOC terms of use",
        "prepared_root": (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
            "cross_model_automl_20260729/"
            "voc2012_segmentation_v1/prepared"
        ),
        "train_image_count": 1464,
        "train_mask_count": 1464,
        "validation_image_count": 1449,
        "validation_mask_count": 1449,
        "num_classes": 21,
        "ignore_label": 255,
        "official_archive_sha256": (
            "e14f763270cf193d0b5f74b169f44157a4b0c6efa708f4dd0ff78ee691763bcb"
        ),
        # This is the stable, VOC-only semantic-validation subreport hash.
        "content_sha256": (
            "815b5d01b625238b449c4bca828bf96107b367f0f4d5d8a31d2f97c6161a5de0"
        ),
        "manifest_path": str(path),
        "manifest_sha256": EXPECTED_DATASET_FILE_MANIFEST_SHA256,
        "file_manifest_entry_count": 5827,
        "remote_sha256sum_check": "passed_all_5827",
        "stage_manifest_path": str(stage_path),
        "stage_manifest_lustre_path": (
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
            "cross_model_automl_20260729/voc2012_segmentation_v1/"
            "dataset_stage_manifest.v1.json"
        ),
        "stage_manifest_sha256": EXPECTED_STAGE_MANIFEST_SHA256,
        "remote_file_manifest_path": file_manifest["lustre_path"],
        "remote_read_only": True,
        "remote_writable_entries_after_lock": 0,
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
    head = _git(repository, "rev-parse", "HEAD")
    if (
        ptm_stage_manifest.resolve()
        != Path(
            campaign_contract.FROZEN_V5_QUALIFICATION_CONTRACT[
                "ptm_stage_manifest_path"
            ]
        ).resolve()
    ):
        raise ManifestGenerationError(
            "successor must reuse the exact sealed v5 PTM stage manifest"
        )
    runtime_local_eligibility = qualification_evidence_record(
        qualification,
        qualification_contract,
    )
    runtime_resolver = repository / "src/tao_automl/ptm_runtime.py"
    if not runtime_resolver.is_file():
        raise ManifestGenerationError(
            "successor source lacks the runtime PTM resolver"
        )
    runtime_local_eligibility.update(
        {
            "eligibility_gate_sha256": campaign_contract.sha256_file(
                HERE / "qualification_gate.py"
            ),
            "runtime_resolver_sha256": campaign_contract.sha256_file(
                runtime_resolver
            ),
            "eligibility_source_commit": head,
            "wheel_sha256": EXPECTED_WHEEL_SHA256,
            "sdk_commit": EXPECTED_SDK_COMMIT,
            "skills_commit": EXPECTED_SKILLS_COMMIT,
        }
    )
    return {
        "repository": str(repository.resolve()),
        "source_commit": head,
        "source_dirty": False,
        "wheel_path": str(wheel.resolve()),
        "wheel_sha256": EXPECTED_WHEEL_SHA256,
        "sdk_dir": str(sdk.resolve()),
        "sdk_commit": EXPECTED_SDK_COMMIT,
        "skills_repository": str(skills.resolve()),
        "skills_commit": EXPECTED_SKILLS_COMMIT,
        "automatic_successor_contract_path": str(
            DEFAULT_SUCCESSOR_CONTRACT.resolve()
        ),
        "automatic_successor_runtime_root": str(
            DEFAULT_SUCCESSOR_RUNTIME_ROOT.resolve()
        ),
        "skill_dir": str(
            (
                skills
                / "skills/models/tao-train-segformer"
            ).resolve()
        ),
        "qualification_evidence_path": str(qualification.resolve()),
        "runtime_local_eligibility": runtime_local_eligibility,
        "ptm_stage_manifest_path": str(ptm_stage_manifest.resolve()),
        "partition": campaign_contract.FROZEN_SLURM_PARTITION,
        "account": "edgeai_tao-ptm_image-foundation-model-clip",
        "base_results_dir": (
            "/lustre/fsw/portfolios/edgeai/projects/"
            "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam"
        ),
        "container_mounts": "/lustre",
        "time_hours": campaign_contract.FROZEN_SLURM_TIME_HOURS,
        "timeout_hours": campaign_contract.FROZEN_SLURM_TIMEOUT_HOURS,
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
    qualification: str | Path = DEFAULT_QUALIFICATION,
    qualification_contract: str | Path = DEFAULT_QUALIFICATION_CONTRACT,
    ptm_stage_manifest: str | Path = DEFAULT_PTM_STAGE_MANIFEST,
) -> dict[str, Any]:
    repository_path = Path(repository).resolve()
    value = campaign_contract.build_preregistered_contract(
        campaign_id=(
            "segformer-voc2012-objective-aware-three-mode-20260806-v8"
        ),
        dataset=dataset_record(dataset_manifest, stage_manifest),
        skill_dir=(
            Path(skills).resolve()
            / "skills/models/tao-train-segformer"
        ),
        runtime=_runtime(
            repository=repository_path,
            wheel=Path(wheel).resolve(),
            sdk=Path(sdk).resolve(),
            skills=Path(skills).resolve(),
            qualification=Path(qualification),
            qualification_contract=Path(qualification_contract),
            ptm_stage_manifest=Path(ptm_stage_manifest),
        ),
    )
    value.pop("contract_sha256")
    value["launcher_integrity"] = {
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
        "segformer_latency_worker_sha256": (
            campaign_contract.sha256_file(
                HERE / "segformer_latency_worker.py"
            )
        ),
        "manifest_generator_sha256": campaign_contract.sha256_file(
            HERE / "manifest_generator.py"
        ),
    }
    value["contract_sha256"] = canonical_sha256(value)
    sealed = campaign_contract.validate_contract(value)
    from .qualification_gate import audit_qualification

    try:
        decision = audit_qualification(
            qualification,
            expected_contract=sealed,
        )
        decision.assert_runtime_ready()
    except Exception as exc:
        raise ManifestGenerationError(
            "sealed SegFormer successor does not authorize an exact "
            "runtime-local PTM projection"
        ) from exc
    if decision.runtime_eligibility.get("repository_registry_mutated") is not False:
        raise ManifestGenerationError(
            "runtime-local eligibility attempted to mutate the repository"
        )
    return sealed


def _seal_json_no_overwrite(path: Path, value: dict[str, Any]) -> bool:
    """Atomically publish exact read-only JSON without replacing a file."""
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != encoded
            or path.stat().st_mode & 0o222
        ):
            raise ManifestGenerationError(
                f"existing sealed JSON differs; refusing overwrite: {path}"
            )
        return False
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if (
                path.is_symlink()
                or not path.is_file()
                or path.read_bytes() != encoded
                or path.stat().st_mode & 0o222
            ):
                raise ManifestGenerationError(f"concurrent seal differs: {path}")
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def seal_contract_no_overwrite(
    path: str | Path,
    contract: dict[str, Any],
) -> bool:
    campaign_contract.validate_contract(contract)
    return _seal_json_no_overwrite(Path(path).resolve(), contract)


def _assert_successor_paths(
    contract: dict[str, Any],
    contract_path: Path,
    runtime_root: Path,
) -> None:
    runtime = contract.get("runtime", {})
    if (
        contract_path.resolve()
        != Path(str(runtime.get("automatic_successor_contract_path", ""))).resolve()
        or runtime_root.resolve()
        != Path(str(runtime.get("automatic_successor_runtime_root", ""))).resolve()
    ):
        raise ManifestGenerationError(
            "automatic successor contract or runtime root differs from its "
            "sealed path"
        )


def _completed_mode_status(path: Path) -> bool:
    try:
        statuses = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return False
    return (
        isinstance(statuses, dict)
        and set(statuses) == set(campaign_contract.MODES)
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value == 0
            for value in statuses.values()
        )
    )


def _successful_result(
    claim: dict[str, Any],
    result_path: Path,
    mode_status_path: Path,
) -> bool:
    if not result_path.exists() and not result_path.is_symlink():
        return False
    if (
        result_path.is_symlink()
        or not result_path.is_file()
        or result_path.stat().st_mode & 0o222
        or not _completed_mode_status(mode_status_path)
    ):
        raise ManifestGenerationError(
            "sealed automatic successor result or mode status is invalid"
        )
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise ManifestGenerationError(
            "sealed automatic successor result is invalid"
        ) from exc
    payload = copy.deepcopy(result)
    supplied = payload.pop("result_sha256", None)
    if (
        set(result)
        != {
            *claim,
            "returncode",
            "resume",
            "completed_at_utc",
            "mode_process_status_sha256",
            "result_sha256",
        }
        or any(result.get(name) != value for name, value in claim.items())
        or result.get("returncode") != 0
        or not isinstance(result.get("resume"), bool)
        or not isinstance(result.get("completed_at_utc"), str)
        or not result["completed_at_utc"]
        or result.get("mode_process_status_sha256")
        != campaign_contract.sha256_file(mode_status_path)
        or supplied != canonical_sha256(payload)
    ):
        raise ManifestGenerationError(
            "sealed automatic successor result integrity failed"
        )
    return True


def _seal_successful_result(
    *,
    claim: dict[str, Any],
    result_path: Path,
    mode_status_path: Path,
    resume: bool,
) -> None:
    from . import run_campaign

    if not _completed_mode_status(mode_status_path):
        raise ManifestGenerationError(
            "successful runner return lacks all-zero mode completion status"
        )
    result = {
        **claim,
        "returncode": 0,
        "resume": resume,
        "completed_at_utc": run_campaign.utc_timestamp(),
        "mode_process_status_sha256": campaign_contract.sha256_file(
            mode_status_path
        ),
    }
    result["result_sha256"] = canonical_sha256(result)
    _seal_json_no_overwrite(result_path, result)
    _successful_result(claim, result_path, mode_status_path)


def launch_successor_once(
    *,
    contract: dict[str, Any],
    contract_path: Path,
    runtime_root: Path,
    env_file: Path,
    poll_seconds: float,
    resume: bool,
) -> int:
    """Serialize launch attempts and allow only explicit supported resume."""
    from . import run_campaign

    _assert_successor_paths(contract, contract_path, runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    lock_path = runtime_root / "automatic_successor_launch.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ManifestGenerationError(
                "the automatic successor launch is already active"
            ) from exc
        claim = {
            "schema_version": 1,
            "kind": "segformer_v5_completion_to_three_mode_launch",
            "contract_path": str(contract_path.resolve()),
            "contract_sha256": contract["contract_sha256"],
            "qualification_file_sha256": contract[
                "qualification_policy"
            ]["runtime_local_eligibility"]["qualification_file_sha256"],
            "qualification_evidence_sha256": contract[
                "qualification_policy"
            ]["runtime_local_eligibility"]["qualification_evidence_sha256"],
            "runtime_root": str(runtime_root.resolve()),
            "automatic_trigger": True,
            "launch": True,
        }
        claim_path = runtime_root / "automatic_successor_launch_claim.json"
        claimed_now = _seal_json_no_overwrite(claim_path, claim)
        result_path = runtime_root / "automatic_successor_launch_result.json"
        mode_status = runtime_root / "mode_process_status.json"
        if not claimed_now and not resume:
            if _successful_result(claim, result_path, mode_status):
                return 0
            raise ManifestGenerationError(
                "automatic successor was already claimed; use --resume only "
                "to continue the existing supported runtime"
            )
        if claimed_now and resume:
            raise ManifestGenerationError(
                "--resume requires an existing automatic successor claim"
            )
        if not claimed_now and resume and _completed_mode_status(mode_status):
            if _successful_result(claim, result_path, mode_status):
                return 0
            _seal_successful_result(
                claim=claim,
                result_path=result_path,
                mode_status_path=mode_status,
                resume=True,
            )
            return 0
        runner_arguments = [
            "--contract",
            str(contract_path.resolve()),
            "--runtime-root",
            str(runtime_root.resolve()),
            "--env-file",
            str(env_file.resolve()),
            "--automatic-trigger",
            "--launch",
            "--poll-seconds",
            str(poll_seconds),
        ]
        if resume:
            runner_arguments.append("--resume")
        returncode = run_campaign.main(runner_arguments)
        if returncode == 0:
            _seal_successful_result(
                claim=claim,
                result_path=result_path,
                mode_status_path=mode_status,
                resume=resume,
            )
            return 0
        failure = {
            **claim,
            "returncode": returncode,
            "resume": resume,
            "completed_at_utc": run_campaign.utc_timestamp(),
        }
        failure["result_sha256"] = canonical_sha256(failure)
        attempts = runtime_root / "automatic_successor_launch_attempts"
        _seal_json_no_overwrite(
            attempts / f"{time.time_ns()}-{os.getpid()}.json",
            failure,
        )
        return returncode


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
        "--qualification", type=Path, default=DEFAULT_QUALIFICATION
    )
    parser.add_argument(
        "--qualification-contract",
        type=Path,
        default=DEFAULT_QUALIFICATION_CONTRACT,
    )
    parser.add_argument(
        "--ptm-stage-manifest",
        type=Path,
        default=DEFAULT_PTM_STAGE_MANIFEST,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--automatic-trigger",
        action="store_true",
        help=(
            "Wait for exact terminal v5 evidence, seal the successor, and "
            "continue without confirmation."
        ),
    )
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=DEFAULT_SUCCESSOR_RUNTIME_ROOT,
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("/localhome/local-rarunachalam/.tao/config.env"),
    )
    args = parser.parse_args(argv)
    if args.launch and not args.automatic_trigger:
        raise ManifestGenerationError(
            "automatic successor launch requires --automatic-trigger"
        )
    if args.resume and not args.launch:
        raise ManifestGenerationError(
            "--resume is only valid for an automatic successor launch"
        )
    if args.automatic_trigger and (
        args.output.resolve() != DEFAULT_SUCCESSOR_CONTRACT.resolve()
        or args.runtime_root.resolve()
        != DEFAULT_SUCCESSOR_RUNTIME_ROOT.resolve()
    ):
        raise ManifestGenerationError(
            "automatic successor requires the exact sealed v6 contract and "
            "fresh runtime paths"
        )
    source_seal = None
    if args.automatic_trigger:
        source_seal = _source_seal_identity(args.repository, args.wheel)
        wait_for_terminal_qualification(
            args.qualification,
            args.qualification_contract,
            status_path=(
                args.runtime_root / "automatic_successor_status.json"
            ),
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        if _source_seal_identity(args.repository, args.wheel) != source_seal:
            raise ManifestGenerationError(
                "automatic watcher source changed while waiting for v5"
            )
    contract = build_contract(
        repository=args.repository,
        wheel=args.wheel,
        sdk=args.sdk,
        skills=args.skills,
        dataset_manifest=args.dataset_manifest,
        stage_manifest=args.stage_manifest,
        qualification=args.qualification,
        qualification_contract=args.qualification_contract,
        ptm_stage_manifest=args.ptm_stage_manifest,
    )
    if source_seal is not None and (
        contract["runtime"]["source_commit"] != source_seal["source_commit"]
        or contract["runtime"]["wheel_sha256"] != source_seal["wheel_sha256"]
        or any(
            contract["launcher_integrity"][name] != source_seal[name]
            for name in (
                "campaign_contract_sha256",
                "qualification_gate_sha256",
                "manifest_generator_sha256",
                "run_campaign_sha256",
            )
        )
    ):
        raise ManifestGenerationError(
            "generated successor differs from the watcher source seal"
        )
    created = seal_contract_no_overwrite(args.output, contract)
    from . import run_campaign

    if args.automatic_trigger:
        run_campaign.atomic_json(
            args.runtime_root / "automatic_successor_status.json",
            {
                "schema_version": 1,
                "automatic_successor": True,
                "state": "successor_contract_sealed",
                "qualification_path": str(args.qualification.resolve()),
                "qualification_file_sha256": contract[
                    "qualification_policy"
                ]["runtime_local_eligibility"][
                    "qualification_file_sha256"
                ],
                "qualification_evidence_sha256": contract[
                    "qualification_policy"
                ]["runtime_local_eligibility"][
                    "qualification_evidence_sha256"
                ],
                "successor_contract_path": str(args.output.resolve()),
                "successor_contract_sha256": contract["contract_sha256"],
                "successor_contract_created": created,
                "model_jobs_launched": False,
                "checked_at_utc": run_campaign.utc_timestamp(),
            },
        )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "contract_sha256": contract["contract_sha256"],
                "launch_authorized": False,
                "reason": (
                    "the automatic trigger revalidates the exact v5 "
                    "completion and its campaign-local in-memory PTM "
                    "eligibility projection before launch"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not args.launch:
        return 0
    return launch_successor_once(
        contract=contract,
        contract_path=args.output.resolve(),
        runtime_root=args.runtime_root.resolve(),
        env_file=args.env_file.resolve(),
        poll_seconds=args.poll_seconds,
        resume=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
