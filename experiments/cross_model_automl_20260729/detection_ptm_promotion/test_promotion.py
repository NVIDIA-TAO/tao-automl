# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for deterministic detection PTM qualification promotion."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tao_automl.ptm_registry import PTMRegistry, canonical_sha256, load_ptm_registry

from .promotion import (
    DetectionPTMPromotionError,
    QualificationEvidence,
    build_candidate_registry,
    derive_qualification_decision,
    main,
)


ROOT = Path(__file__).resolve().parent.parent
DDETR_MANIFEST = (
    ROOT / "deformable_detr_synthetic_campaign" / "campaign.v1.json"
)
RTDETR_MANIFEST = ROOT / "rtdetr_campaign" / "campaign.v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rehash(value: dict, field: str) -> dict:
    value.pop(field, None)
    value[field] = canonical_sha256(value)
    return value


def _test_case(model: str) -> tuple[PTMRegistry, dict]:
    """Build an immutable pre-promotion fixture from the live registry."""
    document = load_ptm_registry().to_dict()
    document["registry_version"] = "test-pre-detection-promotion"
    for record in document["models"][model]["checkpoints"]:
        record["status"] = "unverified"
        record["status_reason"] = "synthetic pre-promotion test fixture"
        record.pop("validation", None)
        if model == "rtdetr":
            record["default_spec_overrides"].pop("dataset", None)
    base = PTMRegistry(document)
    manifest_path = (
        DDETR_MANIFEST if model == "deformable_detr" else RTDETR_MANIFEST
    )
    manifest = _load(manifest_path)
    manifest["integrity"]["ptm_registry_sha256"] = base.document_sha256
    records = {
        item["id"]: item
        for item in base.to_dict()["models"][model]["checkpoints"]
    }
    for ptm in manifest["ptms"]:
        record = records[ptm["id"]]
        ptm["registry_status_before_qualification"] = "unverified"
        ptm["registry_record_sha256"] = canonical_sha256(record)
        ptm["default_spec_overrides"] = copy.deepcopy(
            record["default_spec_overrides"]
        )
    _rehash(manifest, "manifest_sha256")
    return base, manifest


def _success_workflow(
    manifest: dict,
    ptm: dict,
    *,
    resume: bool,
) -> dict:
    checkpoint_id = ptm["id"]
    workflow_id = ptm["workflow_id"]
    checkpoint_sha = canonical_sha256(
        {"checkpoint_id": checkpoint_id, "kind": "trained"}
    )
    train_root = f"/lustre/results/train/{workflow_id}"
    eval_root = f"/lustre/results/evaluate/{workflow_id}"
    checkpoint_path = (
        f"{train_root}/results_dir/train/model_epoch_009.pth"
    )
    validation_metrics = [
        {"mAP": 0.2 + index / 100, "mAP50": 0.4 + index / 100}
        for index in range(10)
    ]
    train = {
        "tao_job_id": f"train-{workflow_id}",
        "status": "Complete",
        "result_root": train_root,
        "full_dataset": True,
        "gpus": 8,
        "nodes": 1,
        "training_epochs": 10,
        "validation_interval": 1,
        "status_evidence": {
            "path": f"{train_root}/results_dir/train/status.json",
            "sha256": canonical_sha256(
                {"checkpoint_id": checkpoint_id, "status": "train"}
            ),
            "size_bytes": 1000,
            "terminal_success": True,
            "terminal_success_message": "Train finished successfully.",
            "validation_record_count": 10,
            "validation_metrics": validation_metrics,
        },
        "terminal_checkpoint": {
            "path": checkpoint_path,
            "sha256": checkpoint_sha,
            "size_bytes": 100,
            "terminal_epoch_index": 9,
            "training_epochs": 10,
        },
    }
    if resume:
        train["reused_for_resume"] = True
    evaluation = {
        "tao_job_id": f"eval-{workflow_id}",
        "status": "Complete",
        "result_root": eval_root,
        "full_validation_split": True,
        "gpus": 8,
        "nodes": 1,
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_size_bytes": 100,
        "status_evidence": {
            "path": f"{eval_root}/results_dir/evaluate/status.json",
            "sha256": canonical_sha256(
                {"checkpoint_id": checkpoint_id, "status": "evaluate"}
            ),
            "size_bytes": 500,
            "terminal_success": True,
            "terminal_success_message": "Evaluate finished successfully.",
            "test_metric_record_count": 1,
            "metrics": {"mAP": 0.29, "mAP50": 0.49},
        },
    }
    if resume:
        evaluation["resume_only"] = True
    record = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "workflow_id": workflow_id,
        "ptm_id": checkpoint_id,
        "ptm_sha256": ptm["artifact"]["sha256"],
        "ptm_path": ptm["artifact"]["slurm_path"],
        "status": "success",
        "terminal": True,
        "failure_preserved": False,
        "process_exit_code": 0,
        "agent_intervention_flags": copy.deepcopy(
            ptm["agent_intervention_flags"]
        ),
        "jobs": {"train": train, "evaluation": evaluation},
        "metrics": {"mAP": 0.29, "mAP50": 0.49},
    }
    if resume:
        record["resume"] = {
            "completed_training_job_reused": True,
            "training_job_submitted": False,
            "selection_or_candidate_change": False,
            "prior_workflow_artifact_modified": False,
            "checkpoint_resolved_after_fix": True,
        }
    return record


def _failed_workflow(manifest: dict, ptm: dict) -> dict:
    return {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "workflow_id": ptm["workflow_id"],
        "ptm_id": ptm["id"],
        "ptm_sha256": ptm["artifact"]["sha256"],
        "ptm_path": ptm["artifact"]["slurm_path"],
        "status": "terminal_failure",
        "terminal": True,
        "failure_preserved": True,
        "process_exit_code": 1,
        "agent_intervention_flags": copy.deepcopy(
            ptm["agent_intervention_flags"]
        ),
        "failure": {
            "type": "CampaignExecutionError",
            "message": "preserved terminal failure",
            "replacement_submitted": False,
        },
    }


def _completion(
    manifest: dict,
    *,
    resume: bool,
    failed_ids: tuple[str, ...] = (),
) -> dict:
    workflows = [
        (
            _failed_workflow(manifest, ptm)
            if ptm["id"] in failed_ids
            else _success_workflow(manifest, ptm, resume=resume)
        )
        for ptm in manifest["ptms"]
    ]
    successes = sum(item["status"] == "success" for item in workflows)
    value = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "model": manifest["model"],
        "manifest_sha256": manifest["manifest_sha256"],
        "terminal": True,
        "status": (
            "success"
            if successes == len(workflows)
            else "terminal_with_failures"
        ),
        "logical_workflows_submitted": len(workflows),
        "successful_workflows": successes,
        "failed_workflows": len(workflows) - successes,
        "workflows_started_in_parallel": True,
        "cpu_runs": 0,
        "smoke_runs": 0,
        "ministep_runs": 0,
        "local_model_runs": 0,
        "failures_preserved": True,
        "replacement_workflows_submitted": False,
        "outcomes": {
            item["workflow_id"]: item["status"] for item in workflows
        },
        "workflows": workflows,
    }
    if resume:
        prior_manifest = manifest["resume_contract"]["prior_manifest"]
        value.update(
            {
                "completion_generated_automatically": True,
                "resume_completed_training": True,
                "completed_training_jobs_reused": len(workflows),
                "training_jobs_submitted": 0,
                "prior_completion_artifact_modified": False,
                "prior_completion": {
                    "path": "/immutable/completion.json",
                    "file_sha256": "a" * 64,
                    "completion_sha256": "b" * 64,
                    "manifest_sha256": prior_manifest["manifest_sha256"],
                },
            }
        )
    return _rehash(value, "completion_sha256")


def _evidence(
    model: str,
    manifest: dict,
    completion: dict,
) -> QualificationEvidence:
    return QualificationEvidence(
        model=model,
        manifest=manifest,
        completion=completion,
        manifest_path=f"/sealed/{model}/manifest.json",
        completion_path=(
            f"/sealed/{model}/"
            + ("completion.resume.json" if model == "rtdetr" else "completion.json")
        ),
        manifest_file_sha256="c" * 64,
        completion_file_sha256="d" * 64,
    )


def _records(document: dict, model: str) -> dict[str, dict]:
    return {
        item["id"]: item
        for item in document["models"][model]["checkpoints"]
    }


def test_ddetr_promotes_exact_full_success_population_and_preserves_default():
    base, manifest = _test_case("deformable_detr")
    evidence = _evidence(
        "deformable_detr",
        manifest,
        _completion(manifest, resume=False),
    )

    candidate, audit = build_candidate_registry(
        base_registry=base,
        qualifications=(evidence,),
        registry_version="candidate-ddetr-v1",
    )

    before = _records(base.to_dict(), "deformable_detr")
    after = _records(candidate, "deformable_detr")
    assert audit["promoted_checkpoint_ids"] == sorted(before)
    assert audit["failed_checkpoint_ids"] == []
    assert candidate["models"]["deformable_detr"]["default_ptm"] is None
    for checkpoint_id, record in after.items():
        assert record["status"] == "supported"
        assert record["compatible_tao_versions"] == ["==7.1.0"]
        assert record["sha256"] == next(
            item["artifact"]["sha256"]
            for item in manifest["ptms"]
            if item["id"] == checkpoint_id
        )
        assert record["validation"]["tao_version"] == "7.1.0-rc-245"
        assert "sqsh-sha256:" in record["validation"]["container_identity"]
    assert all(value is False for value in audit["intervention_flags"].values())


def test_partial_failure_is_untouched_and_preserved_in_audit():
    base, manifest = _test_case("deformable_detr")
    failed_id = manifest["ptms"][0]["id"]
    evidence = _evidence(
        "deformable_detr",
        manifest,
        _completion(
            manifest,
            resume=False,
            failed_ids=(failed_id,),
        ),
    )

    candidate, audit = build_candidate_registry(
        base_registry=base,
        qualifications=(evidence,),
        registry_version="candidate-partial-v1",
    )

    before = _records(base.to_dict(), "deformable_detr")
    after = _records(candidate, "deformable_detr")
    assert after[failed_id] == before[failed_id]
    assert audit["failed_checkpoint_ids"] == [failed_id]
    assert audit["models"][0]["failure_records"] == [
        {
            "checkpoint_id": failed_id,
            "workflow_id": manifest["ptms"][0]["workflow_id"],
            "failure_type": "CampaignExecutionError",
            "failure_message": "preserved terminal failure",
            "replacement_submitted": False,
        }
    ]


def test_rtdetr_resume_promotes_and_projects_frozen_input_contracts():
    base, manifest = _test_case("rtdetr")
    evidence = _evidence(
        "rtdetr",
        manifest,
        _completion(manifest, resume=True),
    )

    candidate, audit = build_candidate_registry(
        base_registry=base,
        qualifications=(evidence,),
        registry_version="candidate-rtdetr-v1",
    )

    records = _records(candidate, "rtdetr")
    for ptm in manifest["ptms"]:
        contract = ptm["input_contract"]
        expected = {
            "train_spatial_size": [
                contract["height"],
                contract["width"],
            ],
            "eval_spatial_size": [
                contract["height"],
                contract["width"],
            ],
            "preserve_aspect_ratio": contract["preprocessing"][
                "preserve_aspect_ratio"
            ],
        }
        assert (
            records[ptm["id"]]["default_spec_overrides"]["dataset"][
                "augmentation"
            ]
            == expected
        )
        assert audit["models"][0]["compatibility_projections"][
            ptm["id"]
        ] == {"dataset": {"augmentation": expected}}
    assert {
        tuple(
            record["default_spec_overrides"]["dataset"]["augmentation"][
                "train_spatial_size"
            ]
        )
        for record in records.values()
    } == {(544, 960), (640, 640)}
    assert audit["models"][0]["recovery_provenance"] == {
        "evaluation_only_resume": True,
        "completed_training_jobs_reused": 4,
        "training_jobs_submitted": 0,
        "prior_completion": {
            "path": "/immutable/completion.json",
            "file_sha256": "a" * 64,
            "completion_sha256": "b" * 64,
            "manifest_sha256": manifest["resume_contract"][
                "prior_manifest"
            ]["manifest_sha256"],
        },
        "prior_completion_artifact_modified": False,
        "initial_terminal_failures_preserved": True,
    }


def test_outcome_mapping_order_is_not_treated_as_identity():
    base, manifest = _test_case("rtdetr")
    completion = _completion(manifest, resume=True)
    completion["outcomes"] = dict(
        sorted(completion["outcomes"].items())
    )
    _rehash(completion, "completion_sha256")

    decision = derive_qualification_decision(
        base,
        _evidence("rtdetr", manifest, completion),
    )

    assert decision.promoted_checkpoint_ids == tuple(
        item["id"] for item in manifest["ptms"]
    )


def test_rtdetr_rejects_nonresume_completion():
    base, manifest = _test_case("rtdetr")
    completion = _completion(manifest, resume=True)
    completion.pop("resume_completed_training")
    _rehash(completion, "completion_sha256")
    with pytest.raises(
        DetectionPTMPromotionError,
        match="evaluation-only resume",
    ):
        derive_qualification_decision(
            base,
            _evidence("rtdetr", manifest, completion),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda manifest, completion: manifest["ptms"].pop(),
            "population drifted",
        ),
        (
            lambda manifest, completion: manifest["ptms"][0][
                "agent_intervention_flags"
            ].__setitem__("agent_selected_candidate", True),
            "agent intervention flags",
        ),
        (
            lambda manifest, completion: manifest["ptms"][0][
                "artifact"
            ].__setitem__("sha256", "f" * 64),
            "checkpoint hash drifted",
        ),
    ),
)
def test_manifest_population_identity_and_intervention_drift_rejected(
    mutation,
    message,
):
    base, manifest = _test_case("rtdetr")
    completion = _completion(manifest, resume=True)
    mutation(manifest, completion)
    _rehash(manifest, "manifest_sha256")
    for workflow in completion["workflows"]:
        workflow["manifest_sha256"] = manifest["manifest_sha256"]
    completion["manifest_sha256"] = manifest["manifest_sha256"]
    _rehash(completion, "completion_sha256")
    with pytest.raises(DetectionPTMPromotionError, match=message):
        derive_qualification_decision(
            base,
            _evidence("rtdetr", manifest, completion),
        )


def test_status_evidence_and_completion_hash_tampering_rejected():
    base, manifest = _test_case("deformable_detr")
    completion = _completion(manifest, resume=False)
    completion["workflows"][0]["jobs"]["train"]["status_evidence"][
        "validation_record_count"
    ] = 9
    _rehash(completion, "completion_sha256")
    with pytest.raises(
        DetectionPTMPromotionError,
        match="validation evidence is incomplete",
    ):
        derive_qualification_decision(
            base,
            _evidence("deformable_detr", manifest, completion),
        )

    completion = _completion(manifest, resume=False)
    completion["successful_workflows"] = 1
    with pytest.raises(
        DetectionPTMPromotionError,
        match="integrity failed",
    ):
        derive_qualification_decision(
            base,
            _evidence("deformable_detr", manifest, completion),
        )


def test_cli_is_create_only_and_never_modifies_base_registry(tmp_path):
    base, manifest = _test_case("deformable_detr")
    base_path = tmp_path / "base.json"
    manifest_path = tmp_path / "manifest.json"
    completion_path = tmp_path / "completion.json"
    output_path = tmp_path / "candidate.json"
    audit_path = tmp_path / "audit.json"
    base_path.write_text(
        json.dumps(base.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    completion_path.write_text(
        json.dumps(_completion(manifest, resume=False)),
        encoding="utf-8",
    )
    before = base_path.read_bytes()

    assert (
        main(
            [
                "--base-registry",
                str(base_path),
                "--qualification",
                "deformable_detr",
                str(manifest_path),
                str(completion_path),
                "--registry-version",
                "candidate-cli-v1",
                "--output-registry",
                str(output_path),
                "--audit",
                str(audit_path),
            ]
        )
        == 0
    )
    assert base_path.read_bytes() == before
    assert json.loads(audit_path.read_text())["live_registry_modified"] is False
    with pytest.raises(FileExistsError, match="create-only"):
        main(
            [
                "--base-registry",
                str(base_path),
                "--qualification",
                "deformable_detr",
                str(manifest_path),
                str(completion_path),
                "--registry-version",
                "candidate-cli-v1",
                "--output-registry",
                str(output_path),
                "--audit",
                str(audit_path),
            ]
        )

    with pytest.raises(
        DetectionPTMPromotionError,
        match="may not overwrite",
    ):
        main(
            [
                "--base-registry",
                str(base_path),
                "--qualification",
                "deformable_detr",
                str(manifest_path),
                str(completion_path),
                "--registry-version",
                "candidate-cli-v2",
                "--output-registry",
                str(base_path),
                "--audit",
                str(tmp_path / "other-audit.json"),
            ]
        )
