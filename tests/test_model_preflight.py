# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import pytest

from tao_automl.model_preflight import (
    CANONICAL_MODEL_IDS,
    CANONICAL_MODEL_TASKS,
    MODEL_PREFLIGHT_STAGES,
    ModelPreflightInputs,
    ModelPreflightResumeError,
    ModelPreflightStepRequest,
    ModelPreflightStepResult,
    ModelPreflightValidationError,
    PreflightPTMIdentity,
    canonical_sha256,
    run_model_preflight,
    validate_model_preflight_report,
)


def _sha(label: str) -> str:
    return canonical_sha256({"label": label})


def _inputs(
    *,
    model_id: str = "dino",
    metric_name: str = "val_mAP50",
    ptm_ids: tuple[str, ...] = ("dino_r50", "dino_r101"),
) -> ModelPreflightInputs:
    ptms = tuple(
        PreflightPTMIdentity(
            id=ptm_id,
            checkpoint_sha256=_sha(f"{ptm_id}:checkpoint"),
            registry_record_sha256=_sha(f"{ptm_id}:registry"),
            ptm_preflight_sha256=_sha(f"{ptm_id}:preflight"),
        )
        for ptm_id in ptm_ids
    )
    return ModelPreflightInputs(
        preflight_id=f"{model_id}.local.v1",
        model_id=model_id,
        task=CANONICAL_MODEL_TASKS[model_id],
        tao_version="7.1.0",
        source_commit="a" * 40,
        package_sha256=_sha("wheel"),
        container_sha256=_sha("sqsh"),
        dataset_id="voc2007.full",
        dataset_manifest_sha256=_sha("dataset"),
        annotation_contract_sha256=_sha("annotations"),
        train_split_sha256=_sha("train"),
        validation_split_sha256=_sha("validation"),
        default_ptm_id=ptm_ids[0],
        eligible_ptms=ptms,
        merged_spec_sha256=_sha("merged-spec"),
        metric_name=metric_name,
        latency_protocol_sha256=_sha("latency-protocol"),
        latency_input_sha256=_sha("latency-input"),
        latency_timed_scope="model_forward_only",
        output_contract_sha256=_sha("output-contract"),
        seed=271828,
    )


def _success_evidence(
    stage: str,
    inputs: ModelPreflightInputs,
) -> dict:
    ptms = {item.id: item for item in inputs.eligible_ptms}
    default = ptms[inputs.default_ptm_id]
    if stage == "dataset_validation":
        return {
            "dataset_id": inputs.dataset_id,
            "manifest_sha256": inputs.dataset_manifest_sha256,
            "annotation_contract_sha256": inputs.annotation_contract_sha256,
            "annotations_valid": True,
            "train_split_sha256": inputs.train_split_sha256,
            "validation_split_sha256": inputs.validation_split_sha256,
            "train_samples": 5011,
            "validation_samples": 4952,
        }
    if stage == "default_ptm_load":
        return {
            "ptm_id": default.id,
            "checkpoint_sha256": default.checkpoint_sha256,
            "loaded": True,
            "input_contract_verified": True,
            "spec_merge_verified": True,
        }
    if stage == "eligible_ptm_smoke":
        return {
            "ptms": [
                {
                    "ptm_id": item.id,
                    "checkpoint_sha256": item.checkpoint_sha256,
                    "loaded": True,
                    "train_step_passed": True,
                    "validation_step_passed": True,
                    "inference_step_passed": True,
                }
                for item in reversed(inputs.eligible_ptms)
            ]
        }
    if stage == "default_model_full_epoch":
        return {
            "ptm_id": default.id,
            "single_gpu": True,
            "completed": True,
            "completed_epochs": 1,
            "training_batches": 157,
            "distinct_training_steps": 157,
            "final_checkpoint_sha256": _sha("epoch-checkpoint"),
        }
    if stage == "in_epoch_validation":
        return {
            "metric_name": inputs.metric_name,
            "metric_value": 0.42,
            "completed_evaluations": 1,
            "passed": True,
        }
    if stage == "standalone_evaluation":
        return {
            "metric_name": inputs.metric_name,
            "metric_value": 0.44,
            "completed_evaluations": 1,
            "passed": True,
            "runtime_metric_contract_verified": True,
        }
    if stage == "checkpoint_save_reload":
        return {
            "ptm_id": default.id,
            "saved": True,
            "reloaded": True,
            "saved_checkpoint_sha256": _sha("epoch-checkpoint"),
            "reloaded_checkpoint_sha256": _sha("epoch-checkpoint"),
        }
    if stage == "latency_instrumentation":
        return {
            "protocol_sha256": inputs.latency_protocol_sha256,
            "input_sha256": inputs.latency_input_sha256,
            "timed_scope": inputs.latency_timed_scope,
            "single_gpu": True,
            "warmup_iterations": 50,
            "timed_iterations": 100,
            "rounds": 5,
            "synchronized": True,
            "median_ms": 57.1,
            "p95_ms": 58.3,
            "mad_ms": 0.12,
            "iqr_ms": 0.31,
            "robust_cv": 0.004,
            "round_drift_ms": 0.18,
            "device_spread_ms": 0.0,
            "quality_gates_passed": True,
        }
    if stage == "output_artifact_validation":
        return {
            "contract_sha256": inputs.output_contract_sha256,
            "artifacts": [
                {
                    "artifact_id": "metrics",
                    "sha256": _sha("metrics"),
                    "size_bytes": 100,
                },
                {
                    "artifact_id": "checkpoint",
                    "sha256": _sha("checkpoint-output"),
                    "size_bytes": 1000,
                },
            ],
            "missing_artifact_ids": [],
            "valid": True,
        }
    if stage == "interrupted_resume_replay":
        next_hash = _sha("next-request")
        return {
            "interrupted": True,
            "state_saved": True,
            "state_sha256": _sha("resume-state"),
            "resumed": True,
            "replay_deterministic": True,
            "expected_next_request_sha256": next_hash,
            "actual_next_request_sha256": next_hash,
            "no_duplicate_trials": True,
            "no_lost_trials": True,
        }
    raise AssertionError(f"adapter must not be called for {stage}")


class SuccessfulAdapter:
    def __init__(self):
        self.requests: list[ModelPreflightStepRequest] = []

    def __call__(
        self,
        request: ModelPreflightStepRequest,
    ) -> ModelPreflightStepResult:
        self.requests.append(request)
        return ModelPreflightStepResult.success(
            request.stage,
            _success_evidence(request.stage, request.inputs),
        )


def _reseal_top_level(report: dict) -> dict:
    value = deepcopy(report)
    value.pop("report_sha256", None)
    value["report_sha256"] = canonical_sha256(value)
    return value


def _rechain_and_reseal(report: dict) -> dict:
    value = deepcopy(report)
    inputs = ModelPreflightInputs.from_dict(value["inputs"])
    prior: list[str] = []
    for index, record in enumerate(value["records"]):
        request = ModelPreflightStepRequest(
            inputs=inputs,
            stage=record["stage"],
            stage_index=index,
            prior_record_sha256s=tuple(prior),
        )
        record["stage_index"] = index
        record["request_sha256"] = request.canonical_sha256
        record["previous_record_sha256"] = prior[-1] if prior else None
        record["evidence_sha256"] = canonical_sha256(record["evidence"])
        record.pop("record_sha256", None)
        record["record_sha256"] = canonical_sha256(record)
        prior.append(record["record_sha256"])
    return _reseal_top_level(value)


def test_complete_preflight_is_content_addressed_and_slurm_ready():
    inputs = _inputs()
    adapter = SuccessfulAdapter()

    report = run_model_preflight(inputs, adapter)

    assert report["completion_state"] == "completed"
    assert report["slurm_ready"] is True
    assert all(report["readiness"].values())
    assert [item["stage"] for item in report["records"]] == list(
        MODEL_PREFLIGHT_STAGES
    )
    # Metric sanity is derived by the orchestrator, not delegated.
    assert [request.stage for request in adapter.requests] == [
        stage for stage in MODEL_PREFLIGHT_STAGES if stage != "metric_sanity"
    ]
    assert report["records"][6]["evidence"]["gate_type"] == (
        "validation_sanity_gate"
    )
    assert report["records"][6]["evidence"]["passed"] is True
    assert report["report_sha256"] == canonical_sha256({
        key: value
        for key, value in report.items()
        if key != "report_sha256"
    })
    validated = validate_model_preflight_report(
        report,
        expected_inputs=inputs,
    )
    assert validated == report


def test_default_epoch_and_all_ptm_smoke_are_separate_readiness_gates():
    inputs = _inputs()

    def adapter(request):
        if request.stage == "default_model_full_epoch":
            return ModelPreflightStepResult.failure(
                request.stage,
                "epoch_failed",
            )
        return ModelPreflightStepResult.success(
            request.stage,
            _success_evidence(request.stage, inputs),
        )

    report = run_model_preflight(inputs, adapter)

    assert report["completion_state"] == "failed"
    assert report["readiness"]["all_ptms_smoke_tested"] is True
    assert report["readiness"]["default_one_epoch_passed"] is False
    assert report["slurm_ready"] is False


@pytest.mark.parametrize("model_id", CANONICAL_MODEL_IDS)
def test_exactly_eight_canonical_model_identifiers_are_accepted(model_id):
    # Input construction is general even where a metric policy remains blocked.
    inputs = _inputs(model_id=model_id)
    assert inputs.model_id == model_id
    assert inputs.task == CANONICAL_MODEL_TASKS[model_id]


def test_unknown_model_and_task_alias_fail_closed():
    with pytest.raises(ModelPreflightValidationError, match="model_id must"):
        replace(_inputs(), model_id="rt-detr")
    with pytest.raises(ModelPreflightValidationError, match="task must"):
        replace(_inputs(), task="detection")


def test_inputs_are_order_normalized_and_immutable():
    left = _inputs()
    right = replace(left, eligible_ptms=tuple(reversed(left.eligible_ptms)))
    assert left.to_dict() == right.to_dict()
    assert left.canonical_sha256 == right.canonical_sha256
    with pytest.raises(FrozenInstanceError):
        left.seed = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "mutation,match",
    (
        pytest.param(
            lambda evidence: evidence["ptms"].pop(),
            "invalid_stage_evidence",
            id="missing",
        ),
        pytest.param(
            lambda evidence: evidence["ptms"].append(deepcopy(evidence["ptms"][0])),
            "invalid_stage_evidence",
            id="duplicate",
        ),
        pytest.param(
            lambda evidence: evidence["ptms"].append({
                **deepcopy(evidence["ptms"][0]),
                "ptm_id": "unexpected_ptm",
            }),
            "invalid_stage_evidence",
            id="unexpected",
        ),
    ),
)
def test_every_eligible_ptm_requires_exactly_one_smoke_record(mutation, match):
    inputs = _inputs()

    def adapter(request):
        evidence = _success_evidence(request.stage, inputs)
        if request.stage == "eligible_ptm_smoke":
            mutation(evidence)
        return ModelPreflightStepResult.success(request.stage, evidence)

    report = run_model_preflight(inputs, adapter)
    assert report["completion_state"] == "failed"
    assert report["failure"]["stage"] == "eligible_ptm_smoke"
    assert report["failure"]["code"] == match
    assert report["slurm_ready"] is False


def test_ptm_smoke_records_are_order_normalized():
    report = run_model_preflight(_inputs(), SuccessfulAdapter())
    evidence = report["records"][2]["evidence"]
    assert [item["ptm_id"] for item in evidence["ptms"]] == [
        "dino_r101",
        "dino_r50",
    ]


def test_dataset_identity_and_annotation_contract_must_match_inputs():
    inputs = _inputs()

    def adapter(request):
        evidence = _success_evidence(request.stage, inputs)
        if request.stage == "dataset_validation":
            evidence["manifest_sha256"] = _sha("different-dataset")
        return ModelPreflightStepResult.success(request.stage, evidence)

    report = run_model_preflight(inputs, adapter)
    assert report["failure"]["stage"] == "dataset_validation"
    assert report["failure"]["code"] == "invalid_stage_evidence"


def test_metric_must_be_finite_before_task_aware_gate():
    inputs = _inputs()

    def adapter(request):
        evidence = _success_evidence(request.stage, inputs)
        if request.stage == "standalone_evaluation":
            evidence["metric_value"] = float("nan")
        # Non-finite content is rejected at the result boundary.
        return ModelPreflightStepResult.success(request.stage, evidence)

    report = run_model_preflight(inputs, adapter)
    assert report["failure"]["stage"] == "standalone_evaluation"
    assert report["failure"]["code"] == "adapter_exception"
    assert report["slurm_ready"] is False


def test_repository_blocked_task_metric_policy_fails_closed():
    inputs = _inputs(
        model_id="grounding_dino",
        metric_name="val_Pr@0.5",
        ptm_ids=("grounding_dino_swin_t",),
    )
    report = run_model_preflight(inputs, SuccessfulAdapter())

    assert report["completion_state"] == "failed"
    assert report["failure"]["stage"] == "metric_sanity"
    assert report["failure"]["code"] == "metric_sanity_failed"
    assert report["records"][-1]["evidence"]["policy_availability"] == "blocked"
    assert report["slurm_ready"] is False


def test_stage_mismatch_is_rejected_without_copying_adapter_evidence():
    inputs = _inputs()

    def adapter(request):
        return ModelPreflightStepResult.success(
            "different_stage",
            {"secret": "must-not-enter-report"},
        )

    report = run_model_preflight(inputs, adapter)
    assert report["failure"]["code"] == "adapter_stage_mismatch"
    assert "must-not-enter-report" not in str(report)


def test_adapter_exception_records_only_safe_exception_type():
    inputs = _inputs()

    class PrivateCredentialError(RuntimeError):
        pass

    def adapter(_request):
        raise PrivateCredentialError("password=extremely-private")

    report = run_model_preflight(inputs, adapter)
    assert report["failure"]["code"] == "adapter_exception"
    assert report["records"][0]["evidence"] == {
        "exception_type": "PrivateCredentialError"
    }
    assert "extremely-private" not in str(report)
    assert "password=" not in str(report)


def test_checkpoint_save_and_reload_content_must_match():
    inputs = _inputs()

    def adapter(request):
        evidence = _success_evidence(request.stage, inputs)
        if request.stage == "checkpoint_save_reload":
            evidence["reloaded_checkpoint_sha256"] = _sha("corrupt")
        return ModelPreflightStepResult.success(request.stage, evidence)

    report = run_model_preflight(inputs, adapter)
    assert report["failure"]["stage"] == "checkpoint_save_reload"
    assert report["failure"]["code"] == "invalid_stage_evidence"


def test_reloaded_checkpoint_must_be_the_full_epoch_checkpoint():
    inputs = _inputs()

    def adapter(request):
        evidence = _success_evidence(request.stage, inputs)
        if request.stage == "checkpoint_save_reload":
            evidence["saved_checkpoint_sha256"] = _sha("other-checkpoint")
            evidence["reloaded_checkpoint_sha256"] = _sha("other-checkpoint")
        return ModelPreflightStepResult.success(request.stage, evidence)

    report = run_model_preflight(inputs, adapter)
    assert report["failure"]["stage"] == "checkpoint_save_reload"
    assert report["failure"]["code"] == "invalid_stage_evidence"


@pytest.mark.parametrize(
    "field,value",
    (
        ("rounds", 1),
        ("synchronized", False),
        ("quality_gates_passed", False),
        ("median_ms", -1.0),
        ("p95_ms", 10.0),
    ),
)
def test_latency_instrumentation_fails_closed(field, value):
    inputs = _inputs()

    def adapter(request):
        evidence = _success_evidence(request.stage, inputs)
        if request.stage == "latency_instrumentation":
            evidence[field] = value
        return ModelPreflightStepResult.success(request.stage, evidence)

    report = run_model_preflight(inputs, adapter)
    assert report["failure"]["stage"] == "latency_instrumentation"
    assert report["failure"]["code"] == "invalid_stage_evidence"
    assert report["readiness"]["latency_valid"] is False


def test_missing_output_artifact_fails_closed():
    inputs = _inputs()

    def adapter(request):
        evidence = _success_evidence(request.stage, inputs)
        if request.stage == "output_artifact_validation":
            evidence["missing_artifact_ids"] = ["metrics"]
            evidence["valid"] = False
        return ModelPreflightStepResult.success(request.stage, evidence)

    report = run_model_preflight(inputs, adapter)
    assert report["failure"]["stage"] == "output_artifact_validation"
    assert report["failure"]["code"] == "invalid_stage_evidence"


def test_interrupted_prefix_resumes_to_same_deterministic_report():
    inputs = _inputs()
    first_adapter = SuccessfulAdapter()
    interrupted = run_model_preflight(
        inputs,
        first_adapter,
        stop_after_stage="default_model_full_epoch",
    )

    assert interrupted["completion_state"] == "interrupted"
    assert interrupted["slurm_ready"] is False
    assert len(interrupted["records"]) == 4

    resumed_adapter = SuccessfulAdapter()
    resumed = run_model_preflight(
        inputs,
        resumed_adapter,
        resume_report=interrupted,
    )
    uninterrupted = run_model_preflight(inputs, SuccessfulAdapter())

    assert resumed == uninterrupted
    assert resumed["report_sha256"] == uninterrupted["report_sha256"]
    assert resumed_adapter.requests[0].stage == "in_epoch_validation"
    assert resumed_adapter.requests[0].stage_index == 4


def test_resume_rejects_changed_immutable_inputs():
    inputs = _inputs()
    interrupted = run_model_preflight(
        inputs,
        SuccessfulAdapter(),
        stop_after_stage="dataset_validation",
    )
    changed = replace(inputs, seed=inputs.seed + 1)
    with pytest.raises(ModelPreflightResumeError, match="do not match"):
        run_model_preflight(
            changed,
            SuccessfulAdapter(),
            resume_report=interrupted,
        )


def test_terminal_failure_cannot_be_resumed():
    inputs = _inputs()

    def failing(request):
        return ModelPreflightStepResult.failure(request.stage, "load_failed")

    failed = run_model_preflight(inputs, failing)
    with pytest.raises(ModelPreflightResumeError, match="terminal failed"):
        run_model_preflight(
            inputs,
            SuccessfulAdapter(),
            resume_report=failed,
        )


@pytest.mark.parametrize("tamper", ("duplicate", "missing", "unexpected"))
def test_report_rejects_duplicate_missing_or_unexpected_stage(tamper):
    report = run_model_preflight(_inputs(), SuccessfulAdapter())
    value = deepcopy(report)
    if tamper == "duplicate":
        value["records"].insert(3, deepcopy(value["records"][2]))
    elif tamper == "missing":
        value["records"].pop(3)
    else:
        value["records"][3]["stage"] = "not_a_stage"
    value = _reseal_top_level(value)

    with pytest.raises(ModelPreflightValidationError):
        validate_model_preflight_report(value)


def test_report_rejects_content_or_hash_chain_tampering():
    report = run_model_preflight(_inputs(), SuccessfulAdapter())
    value = deepcopy(report)
    value["records"][1]["evidence"]["loaded"] = False
    value = _reseal_top_level(value)
    with pytest.raises(ModelPreflightValidationError, match="evidence_sha256"):
        validate_model_preflight_report(value)


def test_rehashed_but_semantically_invalid_evidence_is_rejected():
    report = run_model_preflight(_inputs(), SuccessfulAdapter())
    value = deepcopy(report)
    value["records"][0]["evidence"]["annotations_valid"] = False
    value = _rechain_and_reseal(value)
    with pytest.raises(
        ModelPreflightValidationError,
        match="annotations_valid must be true",
    ):
        validate_model_preflight_report(value)


def test_adapter_request_has_exact_order_and_prior_hash_chain():
    adapter = SuccessfulAdapter()
    report = run_model_preflight(_inputs(), adapter)
    physical_records = [
        record
        for record in report["records"]
        if record["stage"] != "metric_sanity"
    ]
    for request, record in zip(adapter.requests, physical_records):
        expected_prior = tuple(
            item["record_sha256"]
            for item in report["records"][:request.stage_index]
        )
        assert request.prior_record_sha256s == expected_prior
        assert request.canonical_sha256 == record["request_sha256"]


def test_only_all_true_readiness_can_mark_slurm_ready():
    inputs = _inputs()
    complete = run_model_preflight(inputs, SuccessfulAdapter())
    assert complete["slurm_ready"] is True

    interrupted = run_model_preflight(
        inputs,
        SuccessfulAdapter(),
        stop_after_stage="output_artifact_validation",
    )
    assert interrupted["readiness"]["output_artifacts_valid"] is True
    assert interrupted["readiness"]["resume_replay_valid"] is False
    assert interrupted["slurm_ready"] is False


@pytest.mark.parametrize(
    "invalid",
    (
        {"local_gpu_count": 0},
        {"local_gpu_count": 2},
        {"seed": True},
        {"dataset_manifest_sha256": "not-a-hash"},
        {"default_ptm_id": "not_eligible"},
    ),
)
def test_invalid_frozen_inputs_fail_before_adapter_execution(invalid):
    with pytest.raises(ModelPreflightValidationError):
        replace(_inputs(), **invalid)
