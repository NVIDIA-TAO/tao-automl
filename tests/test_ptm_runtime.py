# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed tests for verified PTM runtime arm construction."""

from __future__ import annotations

import copy
import hashlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import tao_automl.ptm_runtime as runtime_module
from tao_automl.objectives import parse_objective_config
from tao_automl.ptm_preflight import (
    AccessProbeResult,
    ArtifactAdaptationEvidence,
    CheckpointLoadSmokeResult,
    CredentialProbeResult,
    PTMPreflightReport,
    PreparedPTM,
    TensorPreservationEvidence,
    ValidatedCheckpointSpec,
    VerifiedArtifact,
)
from tao_automl.ptm_registry import PTMRegistry, canonical_sha256
from tao_automl.ptm_runtime import (
    build_hierarchical_ptm_runtime,
    resolve_ptm_runtime_inventory,
)
from tao_automl.ptm_search import PTMArmObservation
from tao_automl.recommendation_audit import (
    build_recommendation_audit,
    canonical_audit_sha256,
    visible_history_snapshot,
)
from tao_automl.types import Recommendation


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _record(checkpoint_id, checkpoint_bytes, spec_bytes, *, default=False):
    suffix = checkpoint_id.rsplit(".", 1)[-1]
    return {
        "id": checkpoint_id,
        "status": "supported",
        "status_reason": "test-qualified",
        "source": {
            "provider": "ngc",
            "registry": "nvidia/tao",
            "resource": f"dino_{suffix}",
            "version": "1.0",
            "member": f"{suffix}.pth",
            "official": True,
            "immutable_identity": f"test:{suffix}:1",
        },
        "sha256": _sha(checkpoint_bytes),
        "expected_size_bytes": len(checkpoint_bytes),
        "compatible_tao_versions": [">=7.1,<7.2"],
        "model_family": "dino",
        "architecture": f"dino_{suffix}",
        "backbone": suffix,
        "checkpoint_target": "train.pretrained_model_path",
        "input_contract": {
            "channels": 3,
            "height": 640 if default else 512,
            "width": 640 if default else 512,
            "preprocessing": "rgb",
        },
        "default_spec_overrides": {
            "model": {
                "registry_only": suffix,
                "width": 20 if default else 30,
            }
        },
        "checkpoint_spec_file": {
            "source": "checkpoint_source",
            "member": f"{suffix}.yaml",
            "expected_size_bytes": len(spec_bytes),
            "sha256": _sha(spec_bytes),
        },
        "task_compatibility": ["object_detection"],
        "license": {
            "name": "test-license",
            "url": "https://example.invalid/license",
            "access_requirements": [],
        },
        "deprecation": {
            "is_deprecated": False,
            "reason": None,
            "replacement_id": None,
        },
        "validation": {
            "status": "validated",
            "tao_version": "7.1.0",
            "container_identity": "sha256:" + suffix * 64,
            "evidence": "unit-test",
        },
    }


def _artifact(path, data, checkpoint_id, kind):
    path.write_bytes(data)
    digest = _sha(data)
    return VerifiedArtifact(
        path=path,
        cache_relative_path=f"{checkpoint_id}/{path.name}",
        size_bytes=len(data),
        sha256=digest,
        expected_sha256=digest,
        verification_mode="sha256",
        cache_hit=False,
        source_identity_sha256=_sha(f"{checkpoint_id}:{kind}".encode()),
    )


def _report(tmp_path):
    checkpoint_data = {
        "dino.a": b"checkpoint-a",
        "dino.b": b"checkpoint-b",
    }
    spec_documents = {
        "dino.a": {
            "model": {"yaml_only": "a", "width": 10},
            "train": {"num_epochs": 4},
        },
        "dino.b": {
            "model": {"yaml_only": "b", "width": 15},
            "train": {"num_epochs": 5},
        },
    }
    spec_data = {
        checkpoint_id: (
            "model:\n"
            f"  yaml_only: {document['model']['yaml_only']}\n"
            f"  width: {document['model']['width']}\n"
            "train:\n"
            f"  num_epochs: {document['train']['num_epochs']}\n"
        ).encode()
        for checkpoint_id, document in spec_documents.items()
    }
    records = [
        _record(
            checkpoint_id,
            checkpoint_data[checkpoint_id],
            spec_data[checkpoint_id],
            default=checkpoint_id == "dino.a",
        )
        for checkpoint_id in ("dino.a", "dino.b")
    ]
    registry = PTMRegistry(
        {
            "schema_version": 1,
            "registry_version": "test-runtime-v1",
            "models": {
                "dino": {
                    "default_ptm": "dino.a",
                    "checkpoints": records,
                }
            },
        }
    )
    inventory = registry.compatibility(
        "dino",
        tao_version="7.1.0",
        task="object_detection",
    )
    prepared = []
    for checkpoint_id in ("dino.a", "dino.b"):
        record = registry.checkpoint(checkpoint_id)
        checkpoint = _artifact(
            tmp_path / f"{checkpoint_id}.pth",
            checkpoint_data[checkpoint_id],
            checkpoint_id,
            "checkpoint",
        )
        spec_artifact = _artifact(
            tmp_path / f"{checkpoint_id}.yaml",
            spec_data[checkpoint_id],
            checkpoint_id,
            "spec",
        )
        document = spec_documents[checkpoint_id]
        validated = ValidatedCheckpointSpec(
            path=spec_artifact.path,
            cache_relative_path=spec_artifact.cache_relative_path,
            artifact_sha256=spec_artifact.sha256,
            document_sha256=canonical_sha256(document),
            top_level_keys=("model", "train"),
            document=copy.deepcopy(document),
        )
        access = AccessProbeResult(
            ok=True,
            code="accessible",
            reason="test",
            status_code=200,
            remote_size_bytes=checkpoint.size_bytes,
            etag=f'"{checkpoint_id}"',
            exact_member_url=f"https://example.invalid/{checkpoint_id}",
        )
        smoke = CheckpointLoadSmokeResult(
            ok=True,
            code="load_passed",
            reason="test load passed",
            details={"checkpoint_id": checkpoint_id},
        )
        record_hash = canonical_sha256(record)
        provisional = PreparedPTM(
            checkpoint_id=checkpoint_id,
            registry_status="supported",
            runtime_eligible=True,
            checkpoint=checkpoint,
            checkpoint_spec_artifact=spec_artifact,
            checkpoint_spec=validated,
            access_probe=access,
            load_smoke=smoke,
            registry_record_sha256=record_hash,
            provenance_sha256="",
        )
        provenance = canonical_sha256(
            {
                "checkpoint_id": checkpoint_id,
                "purpose": "runtime",
                "registry_status": "supported",
                "runtime_eligible": True,
                "checkpoint": checkpoint.stable_dict(),
                "checkpoint_spec_artifact": spec_artifact.stable_dict(),
                "checkpoint_spec": validated.stable_dict(),
                "access_probe": access.to_dict(),
                "load_smoke": smoke.stable_dict(),
                "registry_record_sha256": record_hash,
            }
        )
        prepared.append(
            PreparedPTM(
                **{
                    **provisional.__dict__,
                    "provenance_sha256": provenance,
                }
            )
        )
    provisional_report = PTMPreflightReport(
        purpose="runtime",
        validation_statuses=("supported",),
        model="dino",
        task="object_detection",
        tao_version="7.1.0",
        registry_version=registry.registry_version,
        registry_sha256=registry.document_sha256,
        inventory=inventory,
        credential_probe=CredentialProbeResult(
            ok=True,
            code="credential_available",
            reason="test",
        ),
        prepared=tuple(prepared),
        exclusions=(),
        report_sha256="",
    )
    report = PTMPreflightReport(
        **{
            **provisional_report.__dict__,
            "report_sha256": canonical_sha256(
                provisional_report.stable_dict()
            ),
        }
    )
    return report, registry


def _objective(mode):
    return parse_objective_config(
        {
            "objectives": [
                {"metric": "mAP50", "direction": "maximize"},
                {"metric": "latency_ms", "direction": "minimize"},
            ],
            "selection_mode": mode,
            "accuracy_metric": "mAP50",
            "latency_metric": "latency_ms",
            "latency_accuracy_retention": 0.9,
        }
    )


def _resolve(report, mode, **kwargs):
    return resolve_ptm_runtime_inventory(
        report=report,
        objective_config=_objective(mode),
        base_model_defaults={
            "model": {"width": 1, "base_only": True},
            "train": {"num_epochs": 1},
        },
        profile_overrides={"model.width": 40},
        user_overrides={"model.width": 50},
        model="dino",
        **kwargs,
    )


class _StateStore:
    def __init__(self):
        self.brain_state = None

    def get_job_specs(self, _job_id):
        return {"wrong_global_base": True}

    def get_custom_param_ranges(self, _handler_id):
        return {"wrong.global.range": {"valid_min": -1}}

    def get_brain_info(self, _job_id):
        return copy.deepcopy(self.brain_state)

    def save_brain_info(self, _job_id, state):
        self.brain_state = copy.deepcopy(state)

    def lock(self):
        return nullcontext()


def _parameter(name, minimum, maximum):
    return {
        "parameter": name,
        "value_type": "float",
        "default_value": minimum,
        "valid_min": minimum,
        "valid_max": maximum,
        "valid_options": [],
        "option_weights": None,
        "math_cond": None,
        "parent_param": None,
        "depends_on": None,
    }


def _conditional_inputs(reverse=False):
    entries = [
        (
            "dino.a",
            [_parameter("model.width", 10.0, 20.0)],
            {"model.width": {"valid_min": 11.0, "valid_max": 19.0}},
        ),
        (
            "dino.b",
            [_parameter("model.depth", 2.0, 6.0)],
            {"model.depth": {"valid_min": 3.0, "valid_max": 5.0}},
        ),
    ]
    if reverse:
        entries.reverse()
    return (
        {checkpoint_id: parameters for checkpoint_id, parameters, _ in entries},
        {checkpoint_id: ranges for checkpoint_id, _, ranges in entries},
    )


def _context():
    return SimpleNamespace(
        id="runtime-session",
        handler_id="runtime-session",
        random_seed=999,
    )


def _signed_result(brain, config, history, identifier, accuracy, latency):
    raw = brain.generate_recommendations(history)
    proposal = brain.consume_last_recommendation_audits()
    assert len(raw) == len(proposal) == 1
    recommendation = Recommendation(identifier, raw[0], config.brain_metric)
    recommendation.recommendation_audit = build_recommendation_audit(
        candidate_id=identifier,
        specs=raw[0],
        algorithm="bayesian",
        search_seed=brain.random_seed,
        search_space=brain.parameters,
        custom_ranges=brain.custom_ranges,
        objective_config=config,
        visible_history=visible_history_snapshot(history),
        acquisition={"proposal": proposal[0]},
    )
    recommendation.objective_values = {
        "mAP50": accuracy,
        "latency_ms": latency,
    }
    recommendation.result = accuracy
    recommendation.objective_score = accuracy
    recommendation.status = "success"
    return recommendation, proposal[0]


@pytest.fixture
def verified_report(tmp_path, monkeypatch):
    report, registry = _report(tmp_path)
    monkeypatch.setattr(runtime_module, "load_ptm_registry", lambda: registry)
    return report


def test_resolve_full_latency_inventory_and_merge_all_precedence_layers(
    verified_report,
):
    resolved = _resolve(verified_report, "latency")

    assert resolved.checkpoint_ids == ("dino.a", "dino.b")
    assert resolved.ptm_policy == "all"
    assert resolved.inventory_sha256 == canonical_audit_sha256(
        resolved.stable_dict()
    )
    assert (
        "per_checkpoint_profile_overrides"
        not in resolved.base_layers_sha256
    )
    assert (
        "per_checkpoint_profile_overrides"
        not in resolved.stable_dict()["spec_merge_precedence"]
    )
    for arm in resolved.arms:
        spec = arm.effective_base_spec
        suffix = arm.checkpoint_id[-1]
        assert spec["model"] == {
            "base_only": True,
            "registry_only": suffix,
            "width": 50,
            "yaml_only": suffix,
        }
        assert spec["train"]["pretrained_model_path"] == arm.checkpoint_path
        assert arm.report_sha256 == verified_report.report_sha256
        assert len(arm.registry_record_sha256) == 64
        assert len(arm.input_contract_sha256) == 64


def test_remote_execution_checkpoint_projection_preserves_verified_identity(
    verified_report,
):
    projected = {
        item.checkpoint_id: {
            "path": f"/lustre/ptm/{item.checkpoint_id}.pth",
            "sha256": item.checkpoint.sha256,
            "size_bytes": item.checkpoint.size_bytes,
        }
        for item in verified_report.prepared
    }

    resolved = _resolve(
        verified_report,
        "latency",
        execution_checkpoint_artifacts=projected,
    )

    for arm in resolved.arms:
        assert arm.checkpoint_path == projected[arm.checkpoint_id]["path"]
        assert (
            arm.effective_base_spec["train"]["pretrained_model_path"]
            == projected[arm.checkpoint_id]["path"]
        )
        prepared = next(
            item
            for item in verified_report.prepared
            if item.checkpoint_id == arm.checkpoint_id
        )
        assert (
            arm.checkpoint_artifact_sha256
            == prepared.checkpoint.sha256
        )


def test_per_checkpoint_profile_overrides_preserve_ptm_input_contracts(
    verified_report,
):
    profiles = {
        "dino.a": {
            "dataset": {
                "augmentation": {
                    "eval_spatial_size": [544, 960],
                    "preserve_aspect_ratio": False,
                }
            }
        },
        "dino.b": {
            "dataset": {
                "augmentation": {
                    "eval_spatial_size": [640, 640],
                    "preserve_aspect_ratio": True,
                }
            }
        },
    }

    resolved = _resolve(
        verified_report,
        "latency",
        per_checkpoint_profile_overrides=profiles,
    )

    by_id = {
        arm.checkpoint_id: arm.effective_base_spec
        for arm in resolved.arms
    }
    assert by_id["dino.a"]["dataset"]["augmentation"] == {
        "eval_spatial_size": [544, 960],
        "preserve_aspect_ratio": False,
    }
    assert by_id["dino.b"]["dataset"]["augmentation"] == {
        "eval_spatial_size": [640, 640],
        "preserve_aspect_ratio": True,
    }
    assert (
        resolved.base_layers_sha256[
            "per_checkpoint_profile_overrides"
        ]
        == canonical_sha256(profiles)
    )
    precedence = resolved.stable_dict()["spec_merge_precedence"]
    assert precedence.index("automl_profile_overrides") < precedence.index(
        "per_checkpoint_profile_overrides"
    ) < precedence.index("user_overrides")


@pytest.mark.parametrize(
    "profiles,match",
    [
        ({"dino.a": {}}, "exactly the selected"),
        (
            {"dino.a": {}, "dino.b": []},
            "values must be mappings",
        ),
    ],
)
def test_per_checkpoint_profile_overrides_fail_closed(
    verified_report,
    profiles,
    match,
):
    with pytest.raises((ValueError, TypeError), match=match):
        _resolve(
            verified_report,
            "latency",
            per_checkpoint_profile_overrides=profiles,
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda values: values.pop("dino.b"), "exactly the selected"),
        (
            lambda values: values["dino.a"].update(
                {"path": "relative/a.pth"}
            ),
            "absolute shared-filesystem",
        ),
        (
            lambda values: values["dino.a"].update(
                {"sha256": "0" * 64}
            ),
            "content identity does not match",
        ),
        (
            lambda values: values["dino.a"].update(
                {"size_bytes": 999}
            ),
            "content identity does not match",
        ),
    ],
)
def test_remote_execution_checkpoint_projection_fails_closed(
    verified_report,
    mutation,
    match,
):
    projected = {
        item.checkpoint_id: {
            "path": f"/lustre/ptm/{item.checkpoint_id}.pth",
            "sha256": item.checkpoint.sha256,
            "size_bytes": item.checkpoint.size_bytes,
        }
        for item in verified_report.prepared
    }
    mutation(projected)

    with pytest.raises((ValueError, TypeError), match=match):
        _resolve(
            verified_report,
            "latency",
            execution_checkpoint_artifacts=projected,
        )


def test_bayesian_alias_is_canonical_across_build_and_resume(
    verified_report,
):
    config = _objective("multi_objective")
    resolved = _resolve(
        verified_report,
        "multi_objective",
        algorithm="b",
    )
    parameters, ranges = _conditional_inputs()
    store = _StateStore()
    built = build_hierarchical_ptm_runtime(
        resolved_inventory=resolved,
        objective_config=config,
        conditional_parameters=parameters,
        conditional_ranges=ranges,
        context=_context(),
        state_store=store,
        random_seed=314159,
        algorithm="b",
    )
    built.brain.save_state()
    resumed = build_hierarchical_ptm_runtime(
        resolved_inventory=resolved,
        objective_config=config,
        conditional_parameters=parameters,
        conditional_ranges=ranges,
        context=_context(),
        state_store=store,
        random_seed=314159,
        algorithm="b",
        resume=True,
    )

    assert resolved.algorithm == "bayesian"
    assert built.manifest["algorithm"] == "bayesian"
    assert resumed.brain.signature == built.brain.signature


def test_accuracy_ptm_policy_default_user_and_explicit_all(verified_report):
    default = _resolve(verified_report, "accuracy")
    user = _resolve(
        verified_report,
        "accuracy",
        ptm_policy="user",
        user_checkpoint_id="dino.b",
    )
    all_ptms = _resolve(
        verified_report,
        "accuracy",
        ptm_policy="all",
    )

    assert default.checkpoint_ids == ("dino.a",)
    assert user.checkpoint_ids == ("dino.b",)
    assert all_ptms.checkpoint_ids == ("dino.a", "dino.b")


@pytest.mark.parametrize("mode", ["latency", "multi_objective"])
def test_latency_and_moo_reject_narrow_ptm_policy(verified_report, mode):
    with pytest.raises(ValueError, match="complete prepared PTM inventory"):
        _resolve(verified_report, mode, ptm_policy="default")


@pytest.mark.parametrize(
    ("mode", "policy"),
    [("accuracy", None), ("multi_objective", None)],
)
def test_user_checkpoint_assignment_is_rejected_for_every_registry_arm(
    verified_report,
    mode,
    policy,
):
    with pytest.raises(ValueError, match="cannot assign registry-resolved"):
        resolve_ptm_runtime_inventory(
            report=verified_report,
            objective_config=_objective(mode),
            base_model_defaults={"model": {"width": 1}},
            user_overrides={
                "train.pretrained_model_path": "/manually/injected.pth"
            },
            ptm_policy=policy,
            model="dino",
        )


def test_only_native_bayesian_and_typed_valid_report_are_accepted(
    verified_report,
):
    with pytest.raises(ValueError, match="only native"):
        _resolve(verified_report, "accuracy", algorithm="random")
    with pytest.raises(TypeError, match="live typed"):
        resolve_ptm_runtime_inventory(
            report=verified_report.to_dict(),
            objective_config=_objective("accuracy"),
            base_model_defaults={},
            model="dino",
        )
    tampered = PTMPreflightReport(
        **{
            **verified_report.__dict__,
            "report_sha256": "0" * 64,
        }
    )
    with pytest.raises(ValueError, match="report integrity"):
        _resolve(tampered, "accuracy")


def test_runtime_eligibility_model_and_artifact_are_revalidated(
    verified_report,
):
    with pytest.raises(ValueError, match="does not match runtime model"):
        resolve_ptm_runtime_inventory(
            report=verified_report,
            objective_config=_objective("accuracy"),
            base_model_defaults={},
            model="rtdetr",
        )

    first = verified_report.prepared[0]
    first.checkpoint.path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size changed|checksum changed"):
        _resolve(verified_report, "accuracy")


def test_explicit_in_memory_registry_is_bound_without_mutating_packaged_registry(
    tmp_path,
    monkeypatch,
):
    report, explicit_registry = _report(tmp_path)
    unrelated_document = explicit_registry.to_dict()
    unrelated_document["registry_version"] = "unrelated-runtime-v1"
    unrelated_registry = PTMRegistry(unrelated_document)
    monkeypatch.setattr(
        runtime_module,
        "load_ptm_registry",
        lambda: unrelated_registry,
    )

    resolved = _resolve(
        report,
        "latency",
        registry=explicit_registry,
    )
    assert resolved.report.registry_sha256 == explicit_registry.document_sha256
    parameters, ranges = _conditional_inputs()
    built = build_hierarchical_ptm_runtime(
        resolved_inventory=resolved,
        objective_config=_objective("latency"),
        conditional_parameters=parameters,
        conditional_ranges=ranges,
        context=_context(),
        state_store=_StateStore(),
        random_seed=271828,
    )
    assert built.resolved_inventory.runtime_registry is explicit_registry

    with pytest.raises(ValueError, match="registry identity"):
        _resolve(report, "latency", registry=unrelated_registry)

    # The caller-owned projection and packaged-registry loader remain
    # independent objects; resolution performs no write or status mutation.
    assert explicit_registry.registry_version == "test-runtime-v1"
    assert unrelated_registry.registry_version == "unrelated-runtime-v1"


def test_adapted_checkpoint_provenance_is_recomputed_with_source_and_adapter(
    verified_report,
    tmp_path,
):
    """Runtime accepts the exact provenance shape emitted by PTM preflight."""
    original = verified_report.prepared[0]
    source = _artifact(
        tmp_path / "source-checkpoint.pth",
        b"source-checkpoint",
        original.checkpoint_id,
        "source-checkpoint",
    )
    tensor_digest = "a" * 64
    tensor_preservation = TensorPreservationEvidence(
        hash_algorithm="sha256_sorted_key_dtype_shape_raw_bytes_v1",
        input_tensor_count=1,
        output_tensor_count=1,
        input_tensor_keys_sha256=tensor_digest,
        output_tensor_keys_sha256=tensor_digest,
        input_tensor_values_sha256=tensor_digest,
        output_tensor_values_sha256=tensor_digest,
    )
    adaptation = ArtifactAdaptationEvidence(
        adapter_id="test-adapter",
        adapter_type="checkpoint_metadata_projection_v1",
        adapter_sha256="b" * 64,
        recipe_sha256="c" * 64,
        input_sha256=source.sha256,
        input_size_bytes=source.size_bytes,
        output_sha256=original.checkpoint.sha256,
        output_size_bytes=original.checkpoint.size_bytes,
        tensor_preservation=tensor_preservation,
        callback_details_sha256="d" * 64,
    )
    provenance_payload = {
        "checkpoint_id": original.checkpoint_id,
        "purpose": "runtime",
        "registry_status": original.registry_status,
        "runtime_eligible": True,
        "checkpoint": original.checkpoint.stable_dict(),
        "checkpoint_spec_artifact": (
            original.checkpoint_spec_artifact.stable_dict()
        ),
        "checkpoint_spec": original.checkpoint_spec.stable_dict(),
        "access_probe": original.access_probe.to_dict(),
        "load_smoke": original.load_smoke.stable_dict(),
        "registry_record_sha256": original.registry_record_sha256,
        "source_checkpoint": source.stable_dict(),
        "artifact_adaptation": adaptation.stable_dict(),
    }
    adapted = PreparedPTM(
        **{
            **original.__dict__,
            "source_checkpoint": source,
            "artifact_adaptation": adaptation,
            "provenance_sha256": canonical_sha256(provenance_payload),
        }
    )
    report = PTMPreflightReport(
        **{
            **verified_report.__dict__,
            "prepared": (adapted, *verified_report.prepared[1:]),
            "report_sha256": "",
        }
    )
    report = PTMPreflightReport(
        **{
            **report.__dict__,
            "report_sha256": canonical_sha256(report.stable_dict()),
        }
    )

    resolved = _resolve(report, "accuracy")

    assert resolved.checkpoint_ids == ("dino.a",)


def test_non_runtime_purpose_and_ineligible_prepared_arm_fail_closed(
    verified_report,
):
    qualification = PTMPreflightReport(
        **{
            **verified_report.__dict__,
            "purpose": "qualification",
        }
    )
    qualification = PTMPreflightReport(
        **{
            **qualification.__dict__,
            "report_sha256": canonical_sha256(qualification.stable_dict()),
        }
    )
    with pytest.raises(ValueError, match="non-runtime"):
        _resolve(qualification, "accuracy")

    original = verified_report.prepared[0]
    ineligible = PreparedPTM(
        **{
            **original.__dict__,
            "runtime_eligible": False,
            "provenance_sha256": "",
        }
    )
    provenance = canonical_sha256(
        {
            "checkpoint_id": ineligible.checkpoint_id,
            "purpose": "runtime",
            "registry_status": ineligible.registry_status,
            "runtime_eligible": False,
            "checkpoint": ineligible.checkpoint.stable_dict(),
            "checkpoint_spec_artifact": (
                ineligible.checkpoint_spec_artifact.stable_dict()
            ),
            "checkpoint_spec": ineligible.checkpoint_spec.stable_dict(),
            "access_probe": ineligible.access_probe.to_dict(),
            "load_smoke": ineligible.load_smoke.stable_dict(),
            "registry_record_sha256": ineligible.registry_record_sha256,
        }
    )
    ineligible = PreparedPTM(
        **{
            **ineligible.__dict__,
            "provenance_sha256": provenance,
        }
    )
    bad_report = PTMPreflightReport(
        **{
            **verified_report.__dict__,
            "prepared": (ineligible, *verified_report.prepared[1:]),
            "report_sha256": "",
        }
    )
    bad_report = PTMPreflightReport(
        **{
            **bad_report.__dict__,
            "report_sha256": canonical_sha256(bad_report.stable_dict()),
        }
    )
    with pytest.raises(ValueError, match="not runtime eligible"):
        _resolve(bad_report, "accuracy")


def test_builder_uses_separate_ptm_effective_specs_ranges_and_seeds(
    verified_report,
):
    config = _objective("multi_objective")
    resolved = _resolve(verified_report, "multi_objective")
    parameters, ranges = _conditional_inputs()
    built = build_hierarchical_ptm_runtime(
        resolved_inventory=resolved,
        objective_config=config,
        conditional_parameters=parameters,
        conditional_ranges=ranges,
        context=_context(),
        state_store=_StateStore(),
        random_seed=161803,
    )

    assert set(built.per_arm_seeds) == {"dino.a", "dino.b"}
    assert len(set(built.per_arm_seeds.values())) == 2
    assert built.manifest_sha256 == canonical_audit_sha256(built.manifest)
    assert built.manifest["required_inner_calibration_issues_per_arm"] == 4
    assert built.brain.scheduler.policy.initial_issues_per_arm == 4
    for arm in resolved.arms:
        inner = built.brain.inner_brains[arm.checkpoint_id]
        assert canonical_sha256(inner.default_train_spec) == (
            arm.effective_base_spec_sha256
        )
        assert inner.custom_ranges == ranges[arm.checkpoint_id]
        assert inner.random_seed == built.per_arm_seeds[arm.checkpoint_id]
        assert "wrong_global_base" not in inner.default_train_spec

    recommendation = built.brain.generate_recommendations([])[0]
    proposal = built.brain.consume_last_recommendation_audits()[0]
    selected = proposal["ptm"]["arm_id"]
    assert recommendation["train"]["pretrained_model_path"].endswith(
        f"{selected}.pth"
    )
    searched = parameters[selected][0]["parameter"].split(".")
    assert recommendation[searched[0]][searched[1]] != 50


def test_builder_default_scheduler_cannot_starve_inner_calibration(
    verified_report,
):
    config = _objective("accuracy")
    resolved = _resolve(verified_report, "accuracy", ptm_policy="all")
    parameters, ranges = _conditional_inputs()
    built = build_hierarchical_ptm_runtime(
        resolved_inventory=resolved,
        objective_config=config,
        conditional_parameters=parameters,
        conditional_ranges=ranges,
        context=_context(),
        state_store=_StateStore(),
        random_seed=271828,
    )
    scheduler = built.brain.scheduler
    quota = max(
        inner.calibration_points
        for inner in built.brain.inner_brains.values()
    )

    issued = [
        scheduler.choose_arm([]).checkpoint_id
        for _ in range(quota * len(resolved.arms))
    ]

    assert {
        checkpoint_id: issued.count(checkpoint_id)
        for checkpoint_id in resolved.checkpoint_ids
    } == {
        checkpoint_id: quota
        for checkpoint_id in resolved.checkpoint_ids
    }
    observations = [
        PTMArmObservation(
            candidate_id=f"{checkpoint_id}-valid-{index}",
            checkpoint_id=checkpoint_id,
            status="success",
            accuracy=0.5,
            latency=10.0,
        )
        for checkpoint_id in resolved.checkpoint_ids
        for index in range(quota)
    ]
    assert (
        scheduler.choose_arm(observations).stage
        == "mode_aware_outer_allocation"
    )


def test_builder_rejects_scheduler_quota_below_inner_calibration(
    verified_report,
):
    config = _objective("multi_objective")
    resolved = _resolve(verified_report, "multi_objective")
    parameters, ranges = _conditional_inputs()

    with pytest.raises(ValueError, match="inner Bayesian calibration"):
        build_hierarchical_ptm_runtime(
            resolved_inventory=resolved,
            objective_config=config,
            conditional_parameters=parameters,
            conditional_ranges=ranges,
            context=_context(),
            state_store=_StateStore(),
            random_seed=271828,
            acquisition_settings={"calibration_points": 5},
            scheduler_options={"initial_issues_per_arm": 4},
        )


def test_non_latency_scheduler_does_not_imply_a_90_percent_default(
    verified_report,
):
    config = parse_objective_config(
        {
            "objectives": [
                {"metric": "mAP50", "direction": "maximize"},
                {"metric": "latency_ms", "direction": "minimize"},
            ],
            "selection_mode": "multi_objective",
            "accuracy_metric": "mAP50",
            "latency_metric": "latency_ms",
        }
    )
    resolved = resolve_ptm_runtime_inventory(
        report=verified_report,
        objective_config=config,
        base_model_defaults={"model": {"width": 1}},
        model="dino",
    )
    parameters, ranges = _conditional_inputs()
    built = build_hierarchical_ptm_runtime(
        resolved_inventory=resolved,
        objective_config=config,
        conditional_parameters=parameters,
        conditional_ranges=ranges,
        context=_context(),
        state_store=_StateStore(),
        random_seed=7,
    )

    assert built.brain.scheduler.policy.latency_accuracy_retention == 0.98
    assert built.manifest["latency_retention_active"] is False


def test_builder_rejects_checkpoint_target_overlap(verified_report):
    resolved = _resolve(verified_report, "multi_objective")
    parameters, ranges = _conditional_inputs()
    parameters["dino.a"] = [
        _parameter("train.pretrained_model_path", 0.0, 1.0)
    ]
    ranges["dino.a"] = {}

    with pytest.raises(ValueError, match="overlaps searchable parameter"):
        build_hierarchical_ptm_runtime(
            resolved_inventory=resolved,
            objective_config=_objective("multi_objective"),
            conditional_parameters=parameters,
            conditional_ranges=ranges,
            context=_context(),
            state_store=_StateStore(),
            random_seed=1,
        )


def test_build_order_invariance_and_exact_resume(verified_report):
    config = _objective("multi_objective")
    resolved_a = _resolve(verified_report, "multi_objective")
    resolved_b = _resolve(verified_report, "multi_objective")
    params_a, ranges_a = _conditional_inputs()
    params_b, ranges_b = _conditional_inputs(reverse=True)
    store_a = _StateStore()
    store_b = _StateStore()
    built_a = build_hierarchical_ptm_runtime(
        resolved_inventory=resolved_a,
        objective_config=config,
        conditional_parameters=params_a,
        conditional_ranges=ranges_a,
        context=_context(),
        state_store=store_a,
        random_seed=314159,
    )
    built_b = build_hierarchical_ptm_runtime(
        resolved_inventory=resolved_b,
        objective_config=config,
        conditional_parameters=params_b,
        conditional_ranges=ranges_b,
        context=_context(),
        state_store=store_b,
        random_seed=314159,
    )

    assert built_a.to_dict() == built_b.to_dict()
    first_a, audit_a = _signed_result(
        built_a.brain,
        config,
        [],
        0,
        0.7,
        12.0,
    )
    first_b, audit_b = _signed_result(
        built_b.brain,
        config,
        [],
        0,
        0.7,
        12.0,
    )
    assert first_a.specs == first_b.specs
    assert audit_a == audit_b
    built_a.brain.save_state()

    resumed = build_hierarchical_ptm_runtime(
        resolved_inventory=resolved_a,
        objective_config=config,
        conditional_parameters=params_b,
        conditional_ranges=ranges_b,
        context=_context(),
        state_store=store_a,
        random_seed=314159,
        resume=True,
    )
    next_uninterrupted = built_a.brain.generate_recommendations([first_a])
    audit_uninterrupted = built_a.brain.consume_last_recommendation_audits()
    next_resumed = resumed.brain.generate_recommendations([first_a])
    audit_resumed = resumed.brain.consume_last_recommendation_audits()

    assert next_resumed == next_uninterrupted
    assert audit_resumed == audit_uninterrupted


def test_resolved_inventory_tamper_is_rejected_by_builder(verified_report):
    resolved = _resolve(verified_report, "multi_objective")
    resolved.arms[0].effective_base_spec["model"]["width"] = 999
    parameters, ranges = _conditional_inputs()

    with pytest.raises(ValueError, match="inventory integrity"):
        build_hierarchical_ptm_runtime(
            resolved_inventory=resolved,
            objective_config=_objective("multi_objective"),
            conditional_parameters=parameters,
            conditional_ranges=ranges,
            context=_context(),
            state_store=_StateStore(),
            random_seed=2,
        )
