# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare and execute the real DINO local preflight in one process.

The live PTM report and ``ResolvedPTMRuntimeInventory`` never cross a JSON
boundary.  This launcher performs production checkpoint preflight, resolves
the typed all-PTM runtime inventory, freezes the DINO plan/executor
configuration, then invokes the concrete local executor.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from tao_automl.objectives import parse_objective_config
from tao_automl.ptm_preflight import (
    AtomicArtifactCache,
    CheckpointLoadSmokeRequest,
    CheckpointLoadSmokeResult,
    NGCCredential,
    NGCHTTPSClient,
    PTMCheckpointPreflight,
)
from tao_automl.ptm_registry import (
    load_ptm_registry,
    merge_ptm_spec_precedence,
)
from tao_automl.ptm_runtime import resolve_ptm_runtime_inventory

try:
    from .dino_local_executor import (
        DINOLocalDockerExecutor,
        DINOLocalExecutionError,
        DINOLocalExecutorConfig,
        DockerBind,
        _canonical_bytes,
        _read_status_jsonl,
        _require_absolute,
        _write_immutable,
    )
    from .dino_local_factories import (
        AUTHORITATIVE_DINO_SKILL_DIR,
        DINOAuthoritativePlanFactory,
        TAO71_RUNTIME_IMAGE,
        build_default_hooks,
        build_reviewed_runtime_image_contract,
    )
    from .dino_preflight import (
        DINOPreflightSettings,
        collect_voc_real_data_integrity,
        freeze_dino_preflight_plan,
        load_dino_skill_contract,
        run_dino_local_preflight,
    )
except ImportError:  # pragma: no cover - direct script execution
    from dino_local_executor import (  # type: ignore[no-redef]
        DINOLocalDockerExecutor,
        DINOLocalExecutionError,
        DINOLocalExecutorConfig,
        DockerBind,
        _canonical_bytes,
        _read_status_jsonl,
        _require_absolute,
        _write_immutable,
    )
    from dino_local_factories import (  # type: ignore[no-redef]
        AUTHORITATIVE_DINO_SKILL_DIR,
        DINOAuthoritativePlanFactory,
        TAO71_RUNTIME_IMAGE,
        build_default_hooks,
        build_reviewed_runtime_image_contract,
    )
    from dino_preflight import (  # type: ignore[no-redef]
        DINOPreflightSettings,
        collect_voc_real_data_integrity,
        freeze_dino_preflight_plan,
        load_dino_skill_contract,
        run_dino_local_preflight,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def _set_dotted(document: dict[str, Any], path: str, value: Any) -> None:
    cursor: Any = document
    parts = path.split(".")
    for token in parts[:-1]:
        if token not in cursor or cursor[token] is None:
            cursor[token] = {}
        cursor = cursor[token]
        if not isinstance(cursor, dict):
            raise ValueError(f"checkpoint target {path!r} crosses a non-object")
    cursor[parts[-1]] = value


class DINOStandardDryRunLoadSmoke:
    """Real one-train/one-validation-batch PTM load smoke in the pinned image."""

    def __init__(
        self,
        *,
        voc: Any,
        cache_root: Path,
        results_root: Path,
        seed: int,
        container_user: str | None,
        poll_interval_seconds: float,
        max_polls: int,
        sdk: Any | None = None,
        entrypoint_builder: Any | None = None,
        sleeper: Any = time.sleep,
    ):
        self.voc = voc
        self.cache_root = cache_root.resolve()
        self.results_root = results_root.resolve()
        self.seed = seed
        self.container_user = container_user
        self.poll_interval_seconds = poll_interval_seconds
        self.max_polls = max_polls
        self.sleeper = sleeper
        self.skill = load_dino_skill_contract(AUTHORITATIVE_DINO_SKILL_DIR)
        self.mounts = (
            DockerBind(voc.dataset_root, "/dataset", True),
            DockerBind(self.cache_root, "/ptm", True),
            DockerBind(self.results_root, "/results", False),
        )
        if sdk is None:
            previous = os.environ.get("DOCKER_PULL_POLICY")
            os.environ["DOCKER_PULL_POLICY"] = "never"
            try:
                from tao_sdk.platforms.docker import DockerSDK

                sdk = DockerSDK(
                    poll_interval=max(1, int(poll_interval_seconds)),
                    state_file=results_root / "ptm_load_smoke_sdk_state.json",
                )
            finally:
                if previous is None:
                    os.environ.pop("DOCKER_PULL_POLICY", None)
                else:
                    os.environ["DOCKER_PULL_POLICY"] = previous
        if entrypoint_builder is None:
            from tao_sdk.script_runner import build_entrypoint

            entrypoint_builder = build_entrypoint
        self.sdk = sdk
        self.entrypoint_builder = entrypoint_builder

    def manifest_identity(self) -> dict[str, Any]:
        """Freeze the real-data train/validation qualification contract."""
        source = Path(__file__).resolve()
        return {
            "callback": "tao71_dino_real_data_dry_run_load_smoke_v1",
            "worker_source_sha256": _sha256_file(source),
            "worker_source_size_bytes": source.stat().st_size,
            "execution_backend": "docker",
            "execution_driver": "tao_sdk.platforms.docker.DockerSDK",
            "container_image": TAO71_RUNTIME_IMAGE,
            "container_identity": TAO71_RUNTIME_IMAGE.rsplit("@", 1)[1],
            "tao_version": "7.1.0-rc-245",
            "pull_policy": "never",
            "gpu_count": 1,
            "train_is_dry_run": True,
            "lightning_fast_dev_run": True,
            "expected_train_batches": 1,
            "expected_validation_batches": 1,
            "dataset_evidence_sha256": _canonical_sha256(self.voc.to_dict()),
            "dataset_manifest_sha256": self.voc.manifest_sha256,
            "skill_contract_sha256": self.skill.sha256,
        }

    def _container_path(self, value: str | Path) -> str:
        host = Path(value).resolve(strict=False)
        matches = []
        for mount in self.mounts:
            try:
                relative = host.relative_to(mount.host_path)
            except ValueError:
                continue
            matches.append((len(mount.host_path.parts), mount, relative))
        if not matches:
            raise DINOLocalExecutionError(
                "undeclared_mount",
                "PTM load-smoke input is outside the declared binds",
            )
        _, mount, relative = max(matches, key=lambda item: item[0])
        return str(Path(mount.container_path) / relative)

    def _translate(self, value: Any) -> Any:
        if isinstance(value, str) and value.startswith("/"):
            return self._container_path(value)
        if isinstance(value, list):
            return [self._translate(item) for item in value]
        if isinstance(value, Mapping):
            return {str(key): self._translate(item) for key, item in value.items()}
        return value

    def _spec(self, request: CheckpointLoadSmokeRequest) -> dict[str, Any]:
        ptm_layer = merge_ptm_spec_precedence(
            model_defaults=request.checkpoint_spec,
            candidate_overrides=request.default_spec_overrides,
        ).spec
        target = request.registry_record.get("checkpoint_target")
        if target not in {
            "train.pretrained_model_path",
            "model.pretrained_backbone_path",
        }:
            raise ValueError("DINO registry checkpoint target is unsupported")
        profile = {
            "wandb": {"enable": False},
            "dataset": {
                "train_data_sources": [{
                    "image_dir": str(self.voc.image_root),
                    "json_file": str(self.voc.train_annotation_path),
                }],
                "val_data_sources": [{
                    "image_dir": str(self.voc.image_root),
                    "json_file": str(self.voc.validation_annotation_path),
                }],
                "num_classes": self.voc.dataset_num_classes,
                "eval_class_ids": list(self.voc.category_ids),
                "batch_size": 1,
            },
            "train": {
                "num_gpus": 1,
                "gpu_ids": [0],
                "num_nodes": 1,
                "seed": self.seed,
                "num_epochs": 1,
                "checkpoint_interval": 1,
                "validation_interval": 1,
                "precision": "fp32",
                "is_dry_run": True,
            },
        }
        candidate: dict[str, Any] = {}
        _set_dotted(candidate, str(target), str(request.checkpoint_path))
        if target == "train.pretrained_model_path":
            candidate["model"] = {
                **candidate.get("model", {}),
                "pretrained_backbone_path": None,
            }
        else:
            candidate["train"] = {
                **candidate.get("train", {}),
                "pretrained_model_path": "",
            }
        spec = merge_ptm_spec_precedence(
            model_defaults=_thaw(self.skill.templates["train"]),
            ptm_overrides=ptm_layer,
            automl_profile_overrides=profile,
            candidate_overrides=candidate,
        ).spec
        return self._translate(spec)

    def _poll(self, job_id: str) -> None:
        for index in range(self.max_polls):
            value = getattr(self.sdk.get_job_status(job_id), "status", None)
            if value in {"Complete", "Error", "Canceled"}:
                if value != "Complete":
                    raise RuntimeError(f"load smoke terminal state {value}")
                return
            if index + 1 < self.max_polls:
                self.sleeper(self.poll_interval_seconds)
        raise RuntimeError("load smoke polling budget exhausted")

    def __call__(
        self,
        request: CheckpointLoadSmokeRequest,
    ) -> CheckpointLoadSmokeResult:
        try:
            contract = _thaw(self.skill.actions["train"])
            merged_spec = self._spec(request)
            entrypoint = self.entrypoint_builder(
                command=contract["command"],
                specs=merged_spec,
                inputs=contract["inputs"],
                outputs=contract["outputs"],
                config_format="yaml",
                upload_excludes=contract.get("upload_excludes", []),
            )
            job = self.sdk.create_job(
                image=TAO71_RUNTIME_IMAGE,
                command=entrypoint["command"],
                gpu_count=1,
                env_vars={
                    "TAO_PTM_LOAD_SMOKE_ID": request.checkpoint_id,
                    "TAO_PREFLIGHT_ACTION": "train",
                },
                mounts=[item.to_sdk_dict() for item in self.mounts],
                shm_size="16g",
                run_as_user=True,
                container_user=self.container_user,
            )
            self._poll(job.id)
            root = Path(
                self.sdk.get_job_results_dir(job.id)
                or getattr(job, "results_dir", "")
            ).resolve()
            status_path = root / "results_dir" / "train" / "status.json"
            records = _read_status_jsonl(status_path, "train")
            if not records:
                raise RuntimeError("load smoke emitted no status")
            checkpoint_path = Path(request.checkpoint_path)
            return CheckpointLoadSmokeResult(
                True,
                "tao71_dino_dry_run_loaded",
                "Checkpoint completed the TAO 7.1 DINO dry-run load smoke",
                {
                    "contract_version": 1,
                    "execution_backend": "docker",
                    "execution_driver": "docker_sdk",
                    "container_image": TAO71_RUNTIME_IMAGE,
                    "container_identity": TAO71_RUNTIME_IMAGE.rsplit("@", 1)[1],
                    "tao_version": request.tao_version,
                    "gpu_count": 1,
                    "train_is_dry_run": True,
                    "lightning_fast_dev_run": True,
                    "checkpoint_sha256": _sha256_file(checkpoint_path),
                    "checkpoint_size_bytes": checkpoint_path.stat().st_size,
                    "checkpoint_loaded": True,
                    "state_dict_compatible": True,
                    "merged_spec_sha256": _canonical_sha256(merged_spec),
                    "registry_record_sha256": _canonical_sha256(
                        request.registry_record
                    ),
                    "dataset_manifest_sha256": self.voc.manifest_sha256,
                    "status_record_count": len(records),
                    "status_artifact_sha256": _sha256_file(status_path),
                    "status_artifact_size_bytes": status_path.stat().st_size,
                    "checkpoint_target": request.registry_record[
                        "checkpoint_target"
                    ],
                },
            )
        except Exception as exc:
            return CheckpointLoadSmokeResult(
                False,
                "tao71_dino_load_smoke_failed",
                "Checkpoint failed the TAO 7.1 DINO dry-run load smoke",
                {"exception_type": type(exc).__name__},
            )


def _git_identity(repository: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise DINOLocalExecutionError(
            "dirty_source_tree",
            "DINO preflight launch requires committed source",
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise DINOLocalExecutionError(
            "invalid_source_commit",
            "could not resolve the current source commit",
        )
    return commit


def _verify_wheel(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise DINOLocalExecutionError(
            "wheel_missing",
            "a regular committed-source wheel is required",
        )
    required = {
        "tao_automl/latency_benchmark.py",
        "tao_automl/latency_stats.py",
        "tao_automl/ptm_runtime.py",
    }
    with zipfile.ZipFile(path) as archive:
        missing = required - set(archive.namelist())
    if missing:
        raise DINOLocalExecutionError(
            "wheel_content_missing",
            "wheel does not contain the required production modules",
        )
    return _sha256_file(path)


def _verify_local_runtime_image(
    process_runner: Any = subprocess.run,
) -> None:
    completed = process_runner(
        ["docker", "image", "inspect", TAO71_RUNTIME_IMAGE],
        check=False,
        capture_output=True,
        text=True,
    )
    if getattr(completed, "returncode", 1) != 0:
        raise DINOLocalExecutionError(
            "local_image_missing",
            "exact TAO 7.1 runtime image is not available locally",
        )
    try:
        values = json.loads(completed.stdout)
    except (AttributeError, ValueError) as exc:
        raise DINOLocalExecutionError(
            "invalid_image_inspection",
            "Docker returned invalid image inspection data",
        ) from exc
    expected = TAO71_RUNTIME_IMAGE
    if (
        not isinstance(values, list)
        or len(values) != 1
        or expected not in values[0].get("RepoDigests", [])
    ):
        raise DINOLocalExecutionError(
            "image_digest_mismatch",
            "local TAO 7.1 image does not expose the reviewed repository digest",
        )


def _artifact_adapter() -> Any:
    source = (
        Path(__file__).resolve().parents[1]
        / "dino_ptm_qualification"
        / "dino_checkpoint_adapter.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "_dino_checkpoint_adapter_runtime",
        source,
    )
    if module_spec is None or module_spec.loader is None:
        raise DINOLocalExecutionError(
            "artifact_adapter_missing",
            "DINO checkpoint artifact adapter could not be loaded",
        )
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module.DINOCheckpointMetadataProjectionCallback()


def prepare_and_run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.tao_version != "7.1.0-rc-245":
        raise DINOLocalExecutionError(
            "tao_version_mismatch",
            "DINO local preflight is pinned to TAO 7.1.0-rc-245",
        )
    if re.fullmatch(r"[1-9]\d*:\d+", args.container_user) is None:
        raise DINOLocalExecutionError(
            "invalid_container_user",
            "container_user must be a non-root numeric UID:GID",
        )
    for name in ("seed", "input_height", "input_width", "max_polls"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise DINOLocalExecutionError(
                "invalid_launch_argument",
                f"{name} must be an integer >= 1",
            )
    source_repo = _require_absolute(args.source_repo, "source_repo")
    source_commit = _git_identity(source_repo)
    wheel_path = _require_absolute(args.wheel, "wheel")
    package_sha256 = _verify_wheel(wheel_path)
    voc_manifest = _require_absolute(args.voc_manifest, "voc_manifest")
    voc_root = _require_absolute(args.voc_root, "voc_root")
    cache_root = _require_absolute(args.ptm_cache, "ptm_cache")
    results_root = _require_absolute(args.results_root, "results_root")
    cache_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    runtime_contract = build_reviewed_runtime_image_contract()
    _verify_local_runtime_image()
    voc = collect_voc_real_data_integrity(
        manifest_path=voc_manifest,
        dataset_root=voc_root,
    )
    load_smoke = DINOStandardDryRunLoadSmoke(
        voc=voc,
        cache_root=cache_root,
        results_root=results_root,
        seed=args.seed,
        container_user=args.container_user,
        poll_interval_seconds=args.poll_interval_seconds,
        max_polls=args.max_polls,
    )
    registry_path = (
        _require_absolute(args.registry_path, "registry_path")
        if args.registry_path is not None
        else None
    )
    registry = load_ptm_registry(registry_path)
    credential = NGCCredential.from_environment()
    preflight = PTMCheckpointPreflight(
        registry=registry,
        cache=AtomicArtifactCache(cache_root),
        ngc_client=NGCHTTPSClient(credential),
        load_smoke=load_smoke,
        artifact_adapter=_artifact_adapter(),
    )
    report = preflight.run(
        model="dino",
        task="object_detection",
        tao_version=args.tao_version,
    )
    if not report.ok:
        raise DINOLocalExecutionError(
            "ptm_runtime_preflight_empty",
            "no supported DINO PTM passed runtime preflight",
        )
    objective = parse_objective_config({
        "objectives": [
            {"metric": "mAP50", "direction": "maximize"},
            {"metric": "latency_ms", "direction": "minimize"},
        ],
        "selection_mode": "multi_objective",
        "accuracy_metric": "mAP50",
        "latency_metric": "latency_ms",
        "latency_accuracy_retention": 0.90,
    })
    skill = load_dino_skill_contract(AUTHORITATIVE_DINO_SKILL_DIR)
    inventory = resolve_ptm_runtime_inventory(
        report=report,
        objective_config=objective,
        base_model_defaults=_thaw(skill.templates["train"]),
        model="dino",
        algorithm="bayesian",
        ptm_policy="all",
    )
    runtime_sha256 = _canonical_sha256({
        "source_commit": source_commit,
        "package_sha256": package_sha256,
        "runtime_image": runtime_contract.runtime_image,
        "skill_contract_sha256": skill.sha256,
        "ptm_report_sha256": report.report_sha256,
    })
    settings = DINOPreflightSettings(
        preflight_id=args.preflight_id,
        tao_version=args.tao_version,
        source_commit=source_commit,
        package_sha256=package_sha256,
        container_sha256=runtime_contract.runtime_digest,
        runtime_sha256=runtime_sha256,
        runtime_image_contract=runtime_contract,
        latency_input_descriptor={
            "shape": [1, 3, args.input_height, args.input_width],
            "dtype": "float32",
            "content": "seeded_preflight_tensor",
        },
        seed=args.seed,
        batch_size=1,
        precision="fp32",
    )
    plan = DINOAuthoritativePlanFactory(
        voc_manifest_path=voc_manifest,
        voc_dataset_root=voc_root,
        resolved_ptm_inventory=inventory,
        settings=settings,
    )()
    plan_path = _require_absolute(args.plan, "plan")
    freeze_dino_preflight_plan(plan_path, plan)
    executor_config = DINOLocalExecutorConfig(
        plan_sha256=plan.plan_sha256,
        image=runtime_contract.runtime_image,
        results_root=results_root,
        mounts=(
            DockerBind(voc_root, "/dataset", True),
            DockerBind(cache_root, "/ptm", True),
            DockerBind(results_root, "/results", False),
        ),
        required_environment=(),
        poll_interval_seconds=args.poll_interval_seconds,
        max_polls=args.max_polls,
        shm_size="16g",
        container_user=args.container_user,
    )
    config_path = _require_absolute(args.executor_config, "executor_config")
    config_document = executor_config.public_dict()
    config_document.pop("gpu_count")
    config_document.pop("docker_pull_policy")
    _write_immutable(
        config_path,
        yaml.safe_dump(config_document, sort_keys=True).encode("utf-8"),
    )
    executor = DINOLocalDockerExecutor(
        plan=plan,
        config=executor_config,
        hooks=build_default_hooks(plan, executor_config),
    )
    result = run_dino_local_preflight(plan=plan, executor=executor)
    report_path = _require_absolute(args.report, "report")
    _write_immutable(report_path, _canonical_bytes(result))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare and run the production DINO local preflight."
    )
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--registry-path", type=Path)
    parser.add_argument("--voc-manifest", type=Path, required=True)
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--ptm-cache", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--executor-config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--container-user", required=True)
    parser.add_argument("--preflight-id", default="dino.voc2007.local.v1")
    parser.add_argument("--tao-version", default="7.1.0-rc-245")
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--input-height", type=int, default=544)
    parser.add_argument("--input-width", type=int, default=960)
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--max-polls", type=int, default=720)
    args = parser.parse_args(argv)
    try:
        result = prepare_and_run(args)
        print(json.dumps({
            "completion_state": result["completion_state"],
            "report": str(Path(args.report).resolve()),
            "plan": str(Path(args.plan).resolve()),
            "executor_config": str(Path(args.executor_config).resolve()),
        }, sort_keys=True))
        return 0 if result["completion_state"] == "completed" else 1
    except Exception as exc:
        code = getattr(exc, "code", "dino_local_launch_failed")
        print(json.dumps({
            "completion_state": "failed",
            "code": code,
        }, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DINOStandardDryRunLoadSmoke",
    "main",
    "prepare_and_run",
]
