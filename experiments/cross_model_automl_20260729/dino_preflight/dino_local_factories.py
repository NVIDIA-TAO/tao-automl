# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concrete runtime factories for the DINO local preflight executor.

These factories contain no credentials and do not select model parameters.
The latency factory requests the fixed, in-container DINO model-forward
worker.  The resume factory uses the production Bayesian brain, Controller,
and file StateStore to compare an uninterrupted recommendation with the next
recommendation produced after an actual state reload.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

try:
    from .dino_local_executor import (
        ContainerLatencyRuntime,
        DINOLocalExecutionError,
        DINOLocalExecutorConfig,
        DINOLocalExecutorHooks,
    )
    from .dino_preflight import (
        DINOPreflightCommand,
        DINOPreflightCommandPlan,
        DINOPreflightSettings,
        DINORuntimeImageContract,
        build_dino_preflight_plan,
        collect_voc_real_data_integrity,
        load_dino_skill_contract,
    )
except ImportError:  # pragma: no cover - direct module execution
    from dino_local_executor import (  # type: ignore[no-redef]
        ContainerLatencyRuntime,
        DINOLocalExecutionError,
        DINOLocalExecutorConfig,
        DINOLocalExecutorHooks,
    )
    from dino_preflight import (  # type: ignore[no-redef]
        DINOPreflightCommand,
        DINOPreflightCommandPlan,
        DINOPreflightSettings,
        DINORuntimeImageContract,
        build_dino_preflight_plan,
        collect_voc_real_data_integrity,
        load_dino_skill_contract,
    )


AUTHORITATIVE_DINO_SKILL_DIR = Path(
    "/localhome/local-rarunachalam/.tao/worktrees/"
    "tao-skills-release-7.1.0/skills/models/tao-train-dino"
)
AUTHORITATIVE_DINO_SKILL_REVISION = (
    "2e9c1b25f3c7cb1ae444c75652e36c47eace8229"
)
TAO71_RUNTIME_IMAGE = (
    "nvcr.io/nvstaging/tao/tao-toolkit-pyt"
    "@sha256:949c0ea8ace09ac91951be4169353cf214daaa3ede7db9eed94070b020361667"
)
TAO71_DINO_SOURCE_EVIDENCE = {
    "tao_pytorch_revision": "2fbd1f1246002e5212e99e864f2713abab060656",
    "files": {
        "nvidia_tao_pytorch/config/dino/default_config.py": (
            "458f9635f606941564e748cb7b0f07f69c0d770b5f40ba9d46609c513adeb2d7"
        ),
        "nvidia_tao_pytorch/cv/dino/scripts/evaluate.py": (
            "72002f17340851cf9a0ea360e56096bb37f418858af452997df3b0ebcaab628c"
        ),
        "nvidia_tao_pytorch/cv/dino/scripts/inference.py": (
            "f02cf8a512809af252885edf8d8a2feae8f4ab46352c31afb754b72b9d109bbc"
        ),
        "nvidia_tao_pytorch/cv/dino/scripts/train.py": (
            "1d265f62c76097ffd43d2fd5f9619797bc65ce7768a0fc5a31cc1d5d787b18e1"
        ),
    },
    "dry_run_contract": "Trainer.fast_dev_run=train.is_dry_run",
}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_thaw(item) for item in value]
    return value


def build_reviewed_runtime_image_contract() -> DINORuntimeImageContract:
    """Build the one reviewed release/7.1 skill-to-digest mapping."""
    skill_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=AUTHORITATIVE_DINO_SKILL_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if skill_revision != AUTHORITATIVE_DINO_SKILL_REVISION:
        raise DINOLocalExecutionError(
            "skill_contract_drift",
            "authoritative release/7.1 DINO skill revision changed",
        )
    skill = load_dino_skill_contract(AUTHORITATIVE_DINO_SKILL_DIR)
    expected_image = (
        "nvcr.io/nvstaging/tao/tao-toolkit-pyt:"
        "7.1.0-rc-245-multiarch"
    )
    if skill.container_image != expected_image:
        raise DINOLocalExecutionError(
            "skill_contract_drift",
            "authoritative release/7.1 DINO skill image changed",
        )
    source_root = Path("/localhome/local-rarunachalam/tao-pytorch")
    source_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_revision != TAO71_DINO_SOURCE_EVIDENCE[
        "tao_pytorch_revision"
    ]:
        raise DINOLocalExecutionError(
            "tao_source_drift",
            "reviewed TAO 7.1 DINO source revision changed",
        )
    for relative, digest in TAO71_DINO_SOURCE_EVIDENCE["files"].items():
        path = source_root / relative
        if not path.is_file() or _sha256_file(path) != digest:
            raise DINOLocalExecutionError(
                "tao_source_drift",
                "reviewed TAO 7.1 DINO source compatibility evidence changed",
            )
    return DINORuntimeImageContract(
        source_skill_revision=AUTHORITATIVE_DINO_SKILL_REVISION,
        compatible_skill_revision=AUTHORITATIVE_DINO_SKILL_REVISION,
        source_skill_image=skill.container_image,
        compatible_skill_image=skill.container_image,
        source_skill_contract_sha256=skill.sha256,
        compatible_skill_contract_sha256=skill.sha256,
        runtime_image=TAO71_RUNTIME_IMAGE,
        tao_schema_compatibility_sha256=_canonical_sha256(
            {
                "skill_revision": AUTHORITATIVE_DINO_SKILL_REVISION,
                "skill_contract_sha256": skill.sha256,
                "skill_file_sha256s": dict(skill.file_sha256s),
            }
        ),
        tao_source_compatibility_sha256=_canonical_sha256(
            TAO71_DINO_SOURCE_EVIDENCE
        ),
    )


class DINOContainerLatencyFactory:
    """Select the fixed one-GPU worker; never execute a host-side model."""

    def __call__(
        self,
        *,
        plan: DINOPreflightCommandPlan,
        command: DINOPreflightCommand,
        checkpoint_path: Path,
        inference_spec: Mapping[str, Any],
    ) -> ContainerLatencyRuntime:
        if (
            plan.latency_contract.expected_replicas != 1
            or plan.latency_contract.timed_scope != "model_forward"
            or plan.latency_contract.measurement_role != "validation_only"
            or plan.settings.precision != "fp32"
            or command.stage != "latency_instrumentation"
            or not checkpoint_path.is_file()
            or not isinstance(inference_spec, Mapping)
        ):
            raise DINOLocalExecutionError(
                "invalid_latency_runtime",
                "DINO container latency runtime violates the frozen contract",
            )
        return ContainerLatencyRuntime()


@dataclass(frozen=True, slots=True)
class DINOAuthoritativePlanFactory:
    """Zero-argument factory over live, already-qualified PTM prerequisites.

    The resolved inventory deliberately remains a live typed object; the
    audit-only serialized PTM report cannot replace its validated checkpoint
    specification documents.
    """

    voc_manifest_path: Path
    voc_dataset_root: Path
    resolved_ptm_inventory: Any
    settings: DINOPreflightSettings
    skill_dir: Path = AUTHORITATIVE_DINO_SKILL_DIR

    def __post_init__(self) -> None:
        for name in ("voc_manifest_path", "voc_dataset_root", "skill_dir"):
            path = Path(getattr(self, name))
            if not path.is_absolute():
                raise ValueError(f"{name} must be absolute")
            object.__setattr__(self, name, path.resolve(strict=False))
        if self.skill_dir != AUTHORITATIVE_DINO_SKILL_DIR:
            raise ValueError(
                "DINO local preflight must use the authoritative release/7.1.0 "
                "skill worktree"
            )
        if not isinstance(self.settings, DINOPreflightSettings):
            raise TypeError("settings must be DINOPreflightSettings")
        runtime = self.settings.runtime_image_contract
        if (
            runtime.source_skill_revision
            != AUTHORITATIVE_DINO_SKILL_REVISION
            or runtime.compatible_skill_revision
            != AUTHORITATIVE_DINO_SKILL_REVISION
        ):
            raise ValueError(
                "DINO runtime mapping must bind the reviewed 7.1 skill commit"
            )

    def __call__(self) -> DINOPreflightCommandPlan:
        voc = collect_voc_real_data_integrity(
            manifest_path=self.voc_manifest_path,
            dataset_root=self.voc_dataset_root,
        )
        return build_dino_preflight_plan(
            voc_integrity=voc,
            resolved_ptm_inventory=self.resolved_ptm_inventory,
            skill_dir=self.skill_dir,
            settings=self.settings,
        )


class DINOProductionResumeReplayRunner:
    """Prove deterministic interruption/resume with production AutoML state."""

    _PARAMETERS = (
        {
            "parameter": "model.num_queries",
            "value_type": "int",
            "default_value": 300,
            "valid_min": 100,
            "valid_max": 900,
            "valid_options": [],
            "option_weights": None,
            "math_cond": None,
            "parent_param": None,
            "depends_on": None,
        },
        {
            "parameter": "model.enc_layers",
            "value_type": "int",
            "default_value": 6,
            "valid_min": 3,
            "valid_max": 6,
            "valid_options": [],
            "option_weights": None,
            "math_cond": None,
            "parent_param": None,
            "depends_on": None,
        },
        {
            "parameter": "model.dec_layers",
            "value_type": "int",
            "default_value": 6,
            "valid_min": 3,
            "valid_max": 6,
            "valid_options": [],
            "option_weights": None,
            "math_cond": None,
            "parent_param": None,
            "depends_on": None,
        },
    )

    @staticmethod
    def _request_sha(recommendation: Any) -> str:
        return _canonical_sha256(
            {
                "candidate_id": str(recommendation.id),
                "specs": recommendation.specs,
                "recommendation_audit_sha256": (
                    recommendation.recommendation_audit["audit_sha256"]
                ),
            }
        )

    @staticmethod
    def _metric(state_path: Path) -> float:
        metric_path = state_path.parent / "in_epoch_validation_metrics.json"
        try:
            document = json.loads(metric_path.read_text(encoding="utf-8"))
            value = document["metric_value"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise DINOLocalExecutionError(
                "resume_metric_missing",
                "resume replay requires the completed physical validation metric",
            ) from exc
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise DINOLocalExecutionError(
                "resume_metric_invalid",
                "resume replay physical validation metric is invalid",
            )
        return float(value)

    @staticmethod
    def _make_controller(
        *,
        workspace: Path,
        context_id: str,
        seed: int,
        train_spec: Mapping[str, Any],
        load: bool,
    ) -> Any:
        from tao_automl.brain.bayesian import Bayesian
        from tao_automl.controller.controller import Controller
        from tao_automl.objectives import parse_objective_config
        from tao_automl.state.state_store import StateStore
        from tao_automl.types import AutoMLContext

        store = StateStore(str(workspace))
        context = AutoMLContext(
            id=context_id,
            network="dino",
            workspace_path=str(workspace),
            metric="mAP50",
            handler_id=context_id,
            num_gpu=1,
            random_seed=seed,
        )
        objective = parse_objective_config(
            {"metric": "mAP50", "direction": "maximize"}
        )
        parameters = [dict(item) for item in DINOProductionResumeReplayRunner._PARAMETERS]
        if not load:
            store.save_job_specs(context.id, _json_thaw(train_spec))
            brain = Bayesian(
                context,
                store,
                "dino",
                parameters,
                metric="mAP50",
                direction="maximize",
                objective_config=objective,
                acquisition_settings={"calibration_points": 2},
            )
            return Controller(
                brain=brain,
                context=context,
                state_store=store,
                settings=SimpleNamespace(automl_max_recommendations=3),
                metric="mAP50",
                algorithm="bayesian",
                objective_config=objective,
            )
        brain = Bayesian.load_state(
            context,
            store,
            "dino",
            parameters,
            metric="mAP50",
            direction="maximize",
            objective_config=objective,
            acquisition_settings={"calibration_points": 2},
        )
        return Controller.load_state(
            brain=brain,
            context=context,
            state_store=store,
            settings=SimpleNamespace(automl_max_recommendations=3),
            metric="mAP50",
            algorithm="bayesian",
            objective_config=objective,
        )

    def __call__(
        self,
        *,
        plan: DINOPreflightCommandPlan,
        command: DINOPreflightCommand,
        state_path: Path,
    ) -> Mapping[str, Any]:
        if command.stage != "interrupted_resume_replay":
            raise DINOLocalExecutionError(
                "invalid_resume_command",
                "resume runner received the wrong preflight stage",
            )
        state_sha256 = _sha256_file(state_path)
        root = state_path.parent / "production_resume_replay"
        completion = root / "completion.json"
        if completion.is_file():
            cached = json.loads(completion.read_text(encoding="utf-8"))
            evidence = cached.get("evidence")
            if (
                cached.get("plan_sha256") != plan.plan_sha256
                or cached.get("command_sha256") != command.sha256
                or not isinstance(evidence, dict)
                or evidence.get("state_sha256") != state_sha256
            ):
                raise DINOLocalExecutionError(
                    "resume_replay_drift",
                    "cached production resume replay does not match the plan",
                )
            return evidence
        if root.exists():
            raise DINOLocalExecutionError(
                "resume_replay_incomplete",
                "an incomplete production resume replay workspace exists",
            )
        root.mkdir(parents=True)
        metric = self._metric(state_path)
        train_spec = command.specs_by_action["train"]
        expected_context = (
            f"dino-preflight-expected-{command.metadata['workspace_identity_sha256'][:16]}"
        )
        resumed_context = (
            f"dino-preflight-resumed-{command.metadata['workspace_identity_sha256'][:16]}"
        )

        expected = self._make_controller(
            workspace=root / "expected",
            context_id=expected_context,
            seed=plan.settings.seed,
            train_spec=train_spec,
            load=False,
        )
        expected_first = expected.next_recommendation()[0]
        expected.report_result(expected_first.id, metric)
        expected_next = expected.next_recommendation()[0]

        interrupted = self._make_controller(
            workspace=root / "interrupted",
            context_id=resumed_context,
            seed=plan.settings.seed,
            train_spec=train_spec,
            load=False,
        )
        interrupted_first = interrupted.next_recommendation()[0]
        interrupted.report_result(interrupted_first.id, metric)
        resumed = self._make_controller(
            workspace=root / "interrupted",
            context_id=resumed_context,
            seed=plan.settings.seed,
            train_spec=train_spec,
            load=True,
        )
        actual_next = resumed.next_recommendation()[0]
        expected_sha = self._request_sha(expected_next)
        actual_sha = self._request_sha(actual_next)
        first_matches = (
            expected_first.specs == interrupted_first.specs
            and expected_first.recommendation_audit["audit_sha256"]
            == interrupted_first.recommendation_audit["audit_sha256"]
        )
        ids = [item.id for item in resumed.history]
        evidence = {
            "interrupted": True,
            "state_saved": (
                (root / "interrupted" / ".automl" / "brain" / f"{resumed_context}.json").is_file()
                and (
                    root
                    / "interrupted"
                    / ".automl"
                    / "controller"
                    / f"{resumed_context}.json"
                ).is_file()
            ),
            "state_sha256": state_sha256,
            "resumed": True,
            "replay_deterministic": first_matches and expected_sha == actual_sha,
            "expected_next_request_sha256": expected_sha,
            "actual_next_request_sha256": actual_sha,
            "no_duplicate_trials": len(ids) == len(set(ids)),
            "no_lost_trials": ids == [0, 1],
        }
        if any(value is not True for key, value in evidence.items() if key in {
            "interrupted",
            "state_saved",
            "resumed",
            "replay_deterministic",
            "no_duplicate_trials",
            "no_lost_trials",
        }):
            raise DINOLocalExecutionError(
                "resume_replay_failed",
                "production interrupted-resume replay did not reproduce state",
            )
        completion_document = {
            "schema_version": 1,
            "plan_sha256": plan.plan_sha256,
            "command_sha256": command.sha256,
            "physical_metric_sha256": _canonical_sha256(
                {"metric_name": "mAP50", "metric_value": metric}
            ),
            "implementation": {
                "brain": "tao_automl.brain.bayesian.Bayesian",
                "controller": "tao_automl.controller.controller.Controller",
                "state_store": "tao_automl.state.state_store.StateStore",
            },
            "evidence": evidence,
        }
        content = (
            json.dumps(
                completion_document,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        temporary = completion.with_name(f".{completion.name}.tmp")
        temporary.write_text(content, encoding="utf-8")
        try:
            os.link(temporary, completion)
        except FileExistsError:
            if completion.read_text(encoding="utf-8") != content:
                raise DINOLocalExecutionError(
                    "resume_replay_drift",
                    "production resume replay completion changed",
                )
        finally:
            temporary.unlink(missing_ok=True)
        return evidence


def build_default_hooks(
    plan: DINOPreflightCommandPlan,
    config: DINOLocalExecutorConfig,
) -> DINOLocalExecutorHooks:
    """Return the reviewed production hooks for the concrete local CLI."""
    if config.plan_sha256 != plan.plan_sha256:
        raise DINOLocalExecutionError(
            "plan_mismatch",
            "hook factory received a different plan/config pair",
        )
    return DINOLocalExecutorHooks(
        latency_runtime_factory=DINOContainerLatencyFactory(),
        resume_replay_runner=DINOProductionResumeReplayRunner(),
    )


__all__ = [
    "AUTHORITATIVE_DINO_SKILL_DIR",
    "AUTHORITATIVE_DINO_SKILL_REVISION",
    "TAO71_DINO_SOURCE_EVIDENCE",
    "TAO71_RUNTIME_IMAGE",
    "DINOAuthoritativePlanFactory",
    "DINOContainerLatencyFactory",
    "DINOProductionResumeReplayRunner",
    "build_reviewed_runtime_image_contract",
    "build_default_hooks",
]
