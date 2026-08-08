# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for deterministic evidence-only DINO registry promotion."""

from __future__ import annotations

import copy

import pytest

from tao_automl.ptm_registry import PTMRegistry, load_ptm_registry

from qualification_driver import DINOQualificationError
from registry_promotion import build_promoted_registry


DEFAULT = "dino.coco.resnet50.trainable.v1.0"
SECOND = "dino.coco.fan_small.trainable.v1.0"


def _completion(
    *,
    evaluated,
    prepared,
    digest,
    upstream=None,
):
    excluded = sorted(set(evaluated) - set(prepared))
    return {
        "completion_sha256": digest,
        "upstream_completion_sha256": upstream,
        "qualification_only": True,
        "runtime_eligibility_mutated": False,
        "selection_invoked": False,
        "agent_selected_checkpoint": False,
        "candidate_accounting": {
            "evaluated_checkpoint_ids": sorted(evaluated),
            "prepared_checkpoint_ids": sorted(prepared),
            "excluded_checkpoint_ids": excluded,
            "complete": True,
        },
    }


def _evidence():
    cpu = _completion(
        evaluated=(DEFAULT, SECOND),
        prepared=(DEFAULT, SECOND),
        digest="a" * 64,
    )
    gpu = _completion(
        evaluated=(DEFAULT, SECOND),
        prepared=(DEFAULT, SECOND),
        digest="b" * 64,
        upstream=cpu["completion_sha256"],
    )
    return cpu, gpu


def test_promotion_is_exact_intersection_and_preserves_nonmembers():
    base = load_ptm_registry()
    before = base.to_dict()
    cpu, gpu = _evidence()

    document, audit = build_promoted_registry(
        base_registry=base,
        cpu_completion=cpu,
        gpu_completion=gpu,
        registry_version="fixture-promoted-v1",
        validation_evidence="fixture/qualification.json",
    )

    validated = PTMRegistry(document)
    records = {
        item["id"]: item
        for item in validated.to_dict()["models"]["dino"]["checkpoints"]
    }
    assert audit["promoted_checkpoint_ids"] == sorted([DEFAULT, SECOND])
    assert audit["runtime_distribution_ready"] is False
    assert all(value is False for value in audit["intervention_flags"].values())
    assert records[SECOND]["status"] == "supported"
    assert records[SECOND]["compatible_tao_versions"] == ["==7.1.0"]
    assert records[SECOND]["validation"]["evidence"] == (
        "fixture/qualification.json"
    )
    untouched = next(
        item
        for item in before["models"]["dino"]["checkpoints"]
        if item["id"] == "dino.coco.gcvit_tiny.trainable.v1.0"
    )
    assert records[untouched["id"]] == untouched


def test_promotion_rejects_upstream_drift_and_missing_default():
    base = load_ptm_registry()
    cpu, gpu = _evidence()
    drifted = copy.deepcopy(gpu)
    drifted["upstream_completion_sha256"] = "f" * 64
    with pytest.raises(DINOQualificationError, match="not bound"):
        build_promoted_registry(
            base_registry=base,
            cpu_completion=cpu,
            gpu_completion=drifted,
            registry_version="v",
            validation_evidence="evidence",
        )

    without_default = _completion(
        evaluated=(DEFAULT, SECOND),
        prepared=(SECOND,),
        digest="b" * 64,
        upstream=cpu["completion_sha256"],
    )
    with pytest.raises(DINOQualificationError, match="default did not pass"):
        build_promoted_registry(
            base_registry=base,
            cpu_completion=cpu,
            gpu_completion=without_default,
            registry_version="v",
            validation_evidence="evidence",
        )


def test_final_promotion_requires_exact_slurm_ready_local_population():
    base = load_ptm_registry()
    cpu, gpu = _evidence()
    candidate_document, _ = build_promoted_registry(
        base_registry=base,
        cpu_completion=cpu,
        gpu_completion=gpu,
        registry_version="fixture-final-v1",
        validation_evidence="fixture/local_report.json",
    )
    records = {
        item["id"]: item
        for item in candidate_document["models"]["dino"]["checkpoints"]
    }
    from tao_automl.ptm_registry import canonical_sha256

    local = {
        "completion_state": "completed",
        "slurm_ready": True,
        "report_sha256": "c" * 64,
        "inputs": {
            "eligible_ptms": [
                {
                    "id": SECOND,
                    "registry_record_sha256": canonical_sha256(
                        records[SECOND]
                    ),
                },
                {
                    "id": DEFAULT,
                    "registry_record_sha256": canonical_sha256(
                        records[DEFAULT]
                    ),
                },
            ],
        },
        "records": [{
            "stage": "eligible_ptm_smoke",
            "status": "passed",
            "evidence": {
                "ptms": [{"ptm_id": DEFAULT}, {"ptm_id": SECOND}],
            },
        }],
    }

    final_document, audit = build_promoted_registry(
        base_registry=base,
        cpu_completion=cpu,
        gpu_completion=gpu,
        registry_version="fixture-final-v1",
        validation_evidence="fixture/local_report.json",
        local_report=local,
    )
    assert final_document == candidate_document
    assert audit["runtime_distribution_ready"] is True
    assert audit["local_preflight_report_sha256"] == "c" * 64

    local["inputs"]["eligible_ptms"].pop()
    with pytest.raises(DINOQualificationError, match="population"):
        build_promoted_registry(
            base_registry=base,
            cpu_completion=cpu,
            gpu_completion=gpu,
            registry_version="fixture-final-v2",
            validation_evidence="fixture/local_report.json",
            local_report=local,
        )
