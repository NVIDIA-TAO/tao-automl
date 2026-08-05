#!/usr/bin/env python3

"""Run three independent, objective-aware Grounding DINO AutoML jobs.

No model operation occurs in the default plan path. Launch is fail-closed on
the sealed qualification handoff, supported-registry promotion, exact source
commit, pinned SQSH/SDK/skills, and a live typed PTM runtime preflight.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import copy
import hashlib
import importlib
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from tao_automl.ptm_registry import canonical_sha256

try:
    from .contract import AGENT_FLAGS, MODES, read_json
    from .pilot_manifest import (
        DEFAULT_INPUTS,
        DEFAULT_OUTPUT,
        SEARCH_PARAMETERS,
        build_manifest,
        load_manifest,
        seal_manifest,
        validate_manifest,
    )
    from .pilot_qualification import (
        PilotQualificationError,
        audit_pilot_qualification,
        evidence_load_callback,
    )
except ImportError:  # pragma: no cover - direct script execution
    from contract import AGENT_FLAGS, MODES, read_json
    from pilot_manifest import (  # type: ignore[no-redef]
        DEFAULT_INPUTS,
        DEFAULT_OUTPUT,
        SEARCH_PARAMETERS,
        build_manifest,
        load_manifest,
        seal_manifest,
        validate_manifest,
    )
    from pilot_qualification import (  # type: ignore[no-redef]
        PilotQualificationError,
        audit_pilot_qualification,
        evidence_load_callback,
    )


HERE = Path(__file__).resolve().parent
DEFAULT_RUNTIME_ROOT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "grounding_dino_three_mode_pilot_v1"
)
ENV_PATH = Path("/localhome/local-rarunachalam/.tao/config.env")
DEFAULT_TRIGGER_POLL_SECONDS = 30.0
DEFAULT_TRIGGER_TIMEOUT_SECONDS = 7 * 24 * 60 * 60


class PilotExecutionError(RuntimeError):
    """The sealed pilot cannot safely continue."""


def _experiment_module(package: str) -> Any:
    parent = str(HERE.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    return importlib.import_module(f"{package}.run_campaign")


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(
        destination.suffix
        + f".{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def configure_slurm_runtime(manifest: Mapping[str, Any]) -> None:
    """Configure the pinned SDK to consume the existing SQSH directly."""
    runtime = manifest["runtime"]
    sdk_dir = str(runtime["sdk_dir"])
    sys.path = [sdk_dir, *[item for item in sys.path if item != sdk_dir]]
    previous = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [
            sdk_dir,
            *[
                item
                for item in previous.split(os.pathsep)
                if item and item != sdk_dir
            ],
        ]
    )
    os.environ.update(
        {
            "SLURM_USE_SQSH": "false",
            "SLURM_USE_REQUEUE": "true",
            "SLURM_TIME_HOURS": str(runtime["time_hours"]),
            "SLURM_TIMEOUT_HOURS": str(runtime["timeout_hours"]),
            "SLURM_MAX_GPUS_PER_NODE": "8",
            "SLURM_PARTITION": str(runtime["partition"]),
            "SLURM_ACCOUNT": str(runtime["account"]),
            "SLURM_BASE_RESULTS_DIR": str(runtime["base_results_dir"]),
            "SLURM_CONTAINER_MOUNTS": str(runtime["container_mounts"]),
            "SLURM_MAX_JOB_RETRIES": str(
                runtime["max_infrastructure_retries"]
            ),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )


def _mode_record(
    manifest: Mapping[str, Any],
    mode: str,
) -> Mapping[str, Any]:
    matches = [item for item in manifest["modes"] if item["mode"] == mode]
    if len(matches) != 1:
        raise PilotExecutionError(f"manifest has no unique {mode!r} mode")
    return matches[0]


def mode_settings(
    manifest: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Translate one mode into production objective-aware Bayesian settings."""
    record = _mode_record(manifest, mode)
    search = manifest["search"]
    settings: dict[str, Any] = {
        "algorithm": "bayesian",
        "automl_max_recommendations": search["candidate_budget_per_mode"],
        "automl_max_concurrent": 1,
        "campaign_id": manifest["campaign_id"],
        "job_id": record["job_id"],
        "session_id": record["session_id"],
        "experiment_id": record["observation_namespace"],
        "random_seed": search["search_seed"],
        "objectives": [
            {"metric": "mAP50", "direction": "maximize"},
            {"metric": "latency_ms", "direction": "minimize"},
        ],
        "selection_mode": mode,
        "accuracy_metric": "mAP50",
        "latency_metric": "latency_ms",
        "multi_objective_min_accuracy": None,
        "objective_acquisition": {
            "calibration_points": search["calibration_points"],
            "augmentation_rho": 1.0e-6,
        },
        "objective_normalization": "pareto_front",
        "augmentation_rho": 1.0e-6,
        "accuracy_tolerance": 1.0e-12,
        "latency_tolerance": search[
            "latency_practical_tolerance_ms"
        ],
        "selection_score_tolerance": 1.0e-12,
        "latency_ci_low_metric": "latency_ci95_low_ms",
        "latency_ci_high_metric": "latency_ci95_high_ms",
        "run_baseline": False,
        "run_final_evaluation": False,
        "require_eval_fn_success": True,
        "automl_delete_intermediate_ckpt": False,
        "automl_checkpoint_retention_strategy": "terminal",
    }
    if mode == "latency":
        settings["latency_accuracy_retention"] = {
            "type": "relative",
            "retained_fraction": search["latency_accuracy_retention"],
            "reference": "accuracy_winner",
        }
    return settings


def custom_ranges(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return the exact preregistered discrete/log-bounded domain."""
    result: dict[str, dict[str, Any]] = {}
    for parameter in SEARCH_PARAMETERS:
        domain = manifest["search"]["space"][parameter]
        if "values" in domain:
            result[parameter] = {
                "valid_options": copy.deepcopy(domain["values"])
            }
        else:
            result[parameter] = {
                "valid_min": domain["minimum"],
                "valid_max": domain["maximum"],
            }
    return result


def spec_overrides(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze identical data, fidelity, text, and hardware for every mode."""
    dataset = manifest["dataset"]
    return {
        "dataset.train_data_sources[0].image_dir": dataset["train"][
            "image_dir"
        ],
        "dataset.train_data_sources[0].json_file": dataset["train"][
            "json_file"
        ],
        "dataset.train_data_sources[0].label_map": dataset["train"][
            "label_map"
        ],
        "dataset.val_data_sources.image_dir": dataset["validation"][
            "image_dir"
        ],
        "dataset.val_data_sources.json_file": dataset["validation"][
            "json_file"
        ],
        "dataset.eval_class_ids": dataset["eval_class_ids"],
        "dataset.batch_size": 4,
        "dataset.workers": 8,
        "model.text_encoder_type": manifest["runtime"][
            "text_encoder_root"
        ],
        "train.num_gpus": 8,
        "train.gpu_ids": list(range(8)),
        "train.num_nodes": 1,
        "train.num_epochs": manifest["search"]["training_epochs"],
        "train.validation_interval": 1,
        "train.checkpoint_interval": manifest["search"]["training_epochs"],
        "train.checkpoint_interval_unit": "epoch",
        "train.seed": manifest["search"]["training_seed"],
        "train.precision": "fp32",
        "train.distributed_strategy": "ddp",
        "train.is_dry_run": False,
        "train.cudnn.benchmark": False,
        "train.cudnn.deterministic": True,
        "wandb.enable": False,
    }


def _set_dotted(target: dict[str, Any], path: str, value: Any) -> None:
    cursor = target
    tokens = path.split(".")
    for token in tokens[:-1]:
        cursor = cursor.setdefault(token, {})
    cursor[tokens[-1]] = copy.deepcopy(value)


def nested_spec_overrides(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path, value in spec_overrides(manifest).items():
        if path.startswith("dataset.train_data_sources[0]."):
            field = path.rsplit(".", 1)[-1]
            values = result.setdefault("dataset", {}).setdefault(
                "train_data_sources",
                [{}],
            )
            values[0][field] = copy.deepcopy(value)
        else:
            _set_dotted(result, path, value)
    return result


def _deep_merge(
    target: dict[str, Any],
    update: Mapping[str, Any],
) -> None:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _load_skill_template(
    manifest: Mapping[str, Any],
    *,
    filename: str,
    digest_field: str,
) -> dict[str, Any]:
    path = (
        Path(manifest["runtime"]["skill_dir"])
        / "references"
        / filename
    )
    if not path.is_file():
        raise PilotExecutionError(f"skill template is unavailable: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[
        "runtime"
    ][digest_field]:
        raise PilotExecutionError(
            f"skill {filename} changed after pilot sealing"
        )
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PilotExecutionError(
            f"skill {filename} must contain a mapping"
        )
    return value


def build_evaluation_spec(
    manifest: Mapping[str, Any],
    recommendation_specs: Mapping[str, Any],
    checkpoint: str,
) -> dict[str, Any]:
    """Carry the complete PTM/candidate architecture into full evaluation."""
    specification = _load_skill_template(
        manifest,
        filename="spec_template_evaluate.yaml",
        digest_field="evaluate_template_sha256",
    )
    _deep_merge(specification, nested_spec_overrides(manifest))
    _deep_merge(specification, recommendation_specs)
    validation = manifest["dataset"]["validation"]
    specification["dataset"]["test_data_sources"] = {
        "image_dir": validation["image_dir"],
        "json_file": validation["json_file"],
    }
    specification["dataset"]["eval_class_ids"] = manifest["dataset"][
        "eval_class_ids"
    ]
    specification["dataset"]["batch_size"] = 4
    specification["evaluate"].update(
        {
            "batch_size": 4,
            "num_gpus": 8,
            "gpu_ids": list(range(8)),
            "num_nodes": 1,
            "checkpoint": checkpoint,
        }
    )
    specification["model"]["text_encoder_type"] = manifest["runtime"][
        "text_encoder_root"
    ]
    specification["wandb"]["enable"] = False
    queries = int(specification["model"]["num_queries"])
    specification["model"]["num_select"] = min(
        int(specification["model"]["num_select"]),
        queries,
    )
    return specification


def latency_worker_contract(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = manifest["latency_protocol"]
    return {
        "schema_version": 1,
        "warmup_iterations": protocol["warmup_iterations"],
        "timed_iterations": protocol["timed_iterations"],
        "repeated_rounds": protocol["repeated_rounds"],
        "tail_percentile": protocol["tail_percentile"],
        "bootstrap_resamples": protocol["bootstrap_resamples"],
        "bootstrap_confidence_level": protocol[
            "bootstrap_confidence_level"
        ],
        "bootstrap_seed": protocol["bootstrap_seed"],
        "batch_size_per_replica": protocol["batch_size_per_replica"],
        "precision": protocol["precision"],
        "timed_scope": protocol["timed_scope"],
        "input_sha256": protocol["input_descriptor_sha256"],
        "runtime_sha256": canonical_sha256(
            {
                "sqsh_sha256": manifest["runtime"]["sqsh_sha256"],
                "skill_revision": manifest["runtime"]["skill_revision"],
                "worker_sha256": hashlib.sha256(
                    (HERE / "grounding_dino_latency_worker.py").read_bytes()
                ).hexdigest(),
            }
        ),
        "expected_replicas": protocol["expected_replicas"],
        "measurement_role": protocol["measurement_role"],
        "synchronization": protocol["synchronization"],
        "validity_thresholds": protocol["validity_thresholds"],
    }


def latency_worker_command(
    *,
    checkpoint: str,
    candidate_fingerprint: str,
) -> str:
    return " ".join(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node=8",
            "/tmp/grounding_dino_automl/grounding_dino_latency_worker.py",
            "--config",
            "{config_path}",
            "--checkpoint",
            shlex.quote(checkpoint),
            "--contract",
            "/tmp/grounding_dino_automl/contract.json",
            "--input-descriptor",
            "/tmp/grounding_dino_automl/input_descriptor.json",
            "--candidate-fingerprint",
            shlex.quote(candidate_fingerprint),
            "--runtime-modules-root",
            "/tmp/grounding_dino_automl",
            "--output-root",
            '"$TAO_RESULTS_ROOT/$TAO_JOB_ID/latency"',
        ]
    )


def _install_latency_payload(manifest: Mapping[str, Any]) -> str:
    source_root = Path(manifest["source"]["repository"]) / "src/tao_automl"
    files = {
        "tao_automl/__init__.py": b"",
        "tao_automl/latency_stats.py": (
            source_root / "latency_stats.py"
        ).read_bytes(),
        "tao_automl/latency_benchmark.py": (
            source_root / "latency_benchmark.py"
        ).read_bytes(),
        "grounding_dino_latency_worker.py": (
            HERE / "grounding_dino_latency_worker.py"
        ).read_bytes(),
        "contract.json": json.dumps(
            latency_worker_contract(manifest),
            sort_keys=True,
        ).encode("utf-8"),
        "input_descriptor.json": json.dumps(
            manifest["latency_protocol"]["input_descriptor"],
            sort_keys=True,
        ).encode("utf-8"),
    }
    encoded = {
        name: base64.b64encode(content).decode("ascii")
        for name, content in files.items()
    }
    script = (
        "import base64,json,pathlib;"
        "root=pathlib.Path('/tmp/grounding_dino_automl');"
        f"files=json.loads({json.dumps(json.dumps(encoded))});"
        "[(root/name).parent.mkdir(parents=True,exist_ok=True) "
        "for name in files];"
        "[(root/name).write_bytes(base64.b64decode(data)) "
        "for name,data in files.items()]"
    )
    return (
        f"python -c {shlex.quote(script)}"
        .replace("{", "{{")
        .replace("}", "}}")
    )


def _launch_latency_child(
    sdk: Any,
    manifest: Mapping[str, Any],
    specification: Mapping[str, Any],
    checkpoint: str,
    candidate_fingerprint: str,
    *,
    events: Path,
    mode: str,
    candidate_id: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Launch and validate one eight-replica stabilized latency child."""
    from tao_automl.latency_benchmark import combine_replica_records
    from tao_sdk.script_runner import build_entrypoint

    common = _experiment_module("dino_campaign")
    benchmark_spec = copy.deepcopy(dict(specification))
    benchmark_spec["dataset"]["batch_size"] = 1
    benchmark_spec["dataset"]["workers"] = 0
    benchmark_spec["evaluate"]["batch_size"] = 1
    command = " ".join(
        [
            "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1",
            "&&",
            _install_latency_payload(manifest),
            "&&",
            latency_worker_command(
                checkpoint=checkpoint,
                candidate_fingerprint=candidate_fingerprint,
            ),
        ]
    )
    action = yaml.safe_load(
        (
            Path(manifest["runtime"]["skill_dir"])
            / "references"
            / "skill_info.yaml"
        ).read_text(encoding="utf-8")
    )["actions"]["evaluate"]
    entrypoint = build_entrypoint(
        command=command,
        specs=benchmark_spec,
        inputs=action["inputs"],
        outputs={},
        config_format=action.get("config_format", "yaml"),
        upload_excludes=action.get("upload_excludes", []),
    )
    runtime = manifest["runtime"]
    job = sdk.create_job(
        image=runtime["sqsh_path"],
        command=entrypoint["command"],
        gpu_count=8,
        num_nodes=1,
        partition=runtime["partition"],
        account=runtime["account"],
    )
    submitted = {
        "status": "submitted",
        "tao_job_id": job.id,
        "candidate_fingerprint": candidate_fingerprint,
        "spec_sha256": canonical_sha256(benchmark_spec),
        "command_sha256": hashlib.sha256(
            entrypoint["command"].encode("utf-8")
        ).hexdigest(),
        "contract_sha256": canonical_sha256(
            latency_worker_contract(manifest)
        ),
    }
    status = common._wait_for_job(
        sdk,
        job.id,
        events=events,
        phase="selection_time_latency",
        mode=mode,
        candidate_id=candidate_id,
    )
    logs = sdk.get_job_logs(job.id, tail=5000)
    if (
        status != "Complete"
        or "TAO_AUTOML_GROUNDING_DINO_LATENCY_COMPLETE" not in logs
    ):
        raise PilotExecutionError(
            f"latency job {job.id} ended as {status}: {logs[-3000:]}"
        )
    result_root = common._local_lustre_path(
        sdk.get_job_results_dir(job.id)
    )
    reader = (
        "import glob,json,sys;"
        "paths=sorted(glob.glob(sys.argv[1]+'/rank_*.json'));"
        "print(json.dumps([json.load(open(path)) for path in paths]))"
    )
    records = json.loads(
        common.remote_output(
            f"python3 -c {shlex.quote(reader)} "
            f"{shlex.quote(result_root + '/latency')}"
        )
    )
    expected_replicas = manifest["latency_protocol"]["expected_replicas"]
    if len(records) != expected_replicas:
        raise PilotExecutionError(
            f"latency job emitted {len(records)}/{expected_replicas} records"
        )
    if {
        item.get("tao_job_id") for item in records
        if isinstance(item, Mapping)
    } != {job.id}:
        raise PilotExecutionError(
            "latency records are not isolated to one TAO job"
        )
    input_hashes = set()
    for record in records:
        evidence = record.get("input_evidence")
        if not isinstance(evidence, Mapping):
            raise PilotExecutionError(
                "latency replica omitted image/text input evidence"
            )
        payload = dict(evidence)
        observed_hash = payload.pop("sha256", None)
        if observed_hash != canonical_sha256(payload):
            raise PilotExecutionError(
                "latency input evidence integrity failed"
            )
        input_hashes.add(observed_hash)
        runtime_evidence = record.get("rank_runtime_evidence")
        if not isinstance(runtime_evidence, Mapping):
            raise PilotExecutionError(
                "latency replica omitted runtime evidence"
            )
        for key, expected in runtime["hardware_contract"].items():
            if runtime_evidence.get(key) != expected:
                raise PilotExecutionError(
                    f"latency hardware contract changed: {key}"
                )
    if len(input_hashes) != 1 or None in input_hashes:
        raise PilotExecutionError(
            "latency replicas did not use identical inputs"
        )
    aggregate = combine_replica_records(records)
    statistics = aggregate["statistics"]
    if (
        statistics.get("is_valid") is not True
        or statistics.get("raw_sample_count_total")
        != manifest["latency_protocol"]["raw_samples_per_candidate"]
    ):
        raise PilotExecutionError(
            "selection-time latency failed the frozen quality gate"
        )
    low, high = statistics["bootstrap_median_ci_ms"]
    metrics = {
        "latency_ms": float(statistics["median_ms"]),
        "latency_p95_ms": float(statistics["p95_ms"]),
        "latency_ci95_low_ms": float(low),
        "latency_ci95_high_ms": float(high),
    }
    return metrics, {
        **submitted,
        "status": status,
        "result_root": result_root,
        "aggregate": aggregate,
        "input_evidence_sha256": next(iter(input_hashes)),
    }


def terminal_checkpoint_identity(
    sdk: Any,
    train_job_id: str,
    *,
    training_epochs: int,
) -> dict[str, Any]:
    """Resolve the one exact terminal-epoch Grounding DINO checkpoint."""
    qualification = _experiment_module("deformable_detr_campaign")
    value = qualification._terminal_checkpoint(
        sdk,
        train_job_id,
        training_epochs=training_epochs,
    )
    if (
        value.get("training_epochs") != training_epochs
        or value.get("terminal_epoch_index") != training_epochs - 1
        or not isinstance(value.get("path"), str)
        or not isinstance(value.get("sha256"), str)
        or value.get("size_bytes", 0) < 1
    ):
        raise PilotExecutionError(
            "exact terminal checkpoint evidence is incomplete"
        )
    return value


class AutomaticFirstCandidateGate:
    """Cross-mode barrier that releases candidates 1-19 without confirmation."""

    def __init__(
        self,
        root: str | Path,
        manifest: Mapping[str, Any],
        *,
        poll_seconds: float = 1.0,
        timeout_seconds: float = 4 * 60 * 60,
    ):
        self.root = Path(root)
        self.manifest_sha256 = manifest["manifest_sha256"]
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds

    def _path(self, mode: str) -> Path:
        return self.root / f"{mode}.json"

    def record(
        self,
        mode: str,
        *,
        candidate_id: str,
        passed: bool,
        evidence_sha256: str | None,
        reason: str,
    ) -> None:
        if mode not in MODES:
            raise PilotExecutionError(f"unsupported gate mode {mode!r}")
        value = {
            "schema_version": 1,
            "manifest_sha256": self.manifest_sha256,
            "mode": mode,
            "candidate_index": 0,
            "candidate_id": candidate_id,
            "passed": bool(passed),
            "evidence_sha256": evidence_sha256,
            "reason": reason,
            "automatic_release": True,
        }
        path = self._path(mode)
        if path.exists():
            if read_json(path) != value:
                raise PilotExecutionError(
                    f"first-candidate gate changed for {mode}"
                )
            return
        atomic_json(path, value)

    def wait_for_release(self) -> dict[str, Any]:
        started = time.monotonic()
        while True:
            records = {
                mode: read_json(self._path(mode))
                for mode in MODES
                if self._path(mode).is_file()
            }
            for mode, value in records.items():
                if value.get("manifest_sha256") != self.manifest_sha256:
                    raise PilotExecutionError(
                        f"{mode} first-candidate manifest changed"
                    )
            failed = [
                mode
                for mode, value in records.items()
                if value.get("passed") is False
            ]
            if failed:
                raise PilotExecutionError(
                    "first-candidate gate failed: "
                    + ", ".join(sorted(failed))
                )
            if set(records) == set(MODES) and all(
                value.get("passed") is True
                for value in records.values()
            ):
                release = {
                    "schema_version": 1,
                    "manifest_sha256": self.manifest_sha256,
                    "released": True,
                    "automatic": True,
                    "released_modes": list(MODES),
                    "remaining_candidates_per_mode": 19,
                    "gate_record_sha256": canonical_sha256(records),
                }
                atomic_json(self.root / "automatic_release.json", release)
                return release
            if time.monotonic() - started >= self.timeout_seconds:
                raise PilotExecutionError(
                    "first-candidate gate timed out before all modes passed"
                )
            time.sleep(self.poll_seconds)


class GroundingDINOCandidateEvaluator:
    """Standalone-eval, stabilized-latency, and automatic-gate adapter."""

    def __init__(
        self,
        *,
        sdk: Any,
        manifest: Mapping[str, Any],
        mode: str,
        runtime_root: str | Path,
        gate: AutomaticFirstCandidateGate,
    ):
        self.sdk = sdk
        self.manifest = manifest
        self.mode = mode
        self.runtime_root = Path(runtime_root)
        self.gate = gate
        self.events = self.runtime_root / mode / "child_events.jsonl"
        self.evidence_path = (
            self.runtime_root / mode / "candidate_evidence.json"
        )
        self.evidence: dict[str, Any] = {}
        if self.evidence_path.is_file():
            self.evidence = read_json(self.evidence_path).get(
                "candidates",
                {},
            )

    def _persist(self) -> None:
        atomic_json(
            self.evidence_path,
            {
                "schema_version": 1,
                "manifest_sha256": self.manifest["manifest_sha256"],
                "mode": self.mode,
                "candidates": self.evidence,
            },
        )

    def on_recommendation(self, recommendation: Any) -> None:
        from tao_automl.selection import canonical_spec_fingerprint

        if int(recommendation.id) > 0:
            self.gate.wait_for_release()
        candidate_id = f"{self.mode}_rec_{recommendation.id}"
        fingerprint = canonical_spec_fingerprint(recommendation.specs)
        audit = copy.deepcopy(recommendation.recommendation_audit)
        if audit.get("candidate_fingerprint") != fingerprint:
            raise PilotExecutionError(
                "recommendation audit fingerprint changed"
            )
        record = {
            "candidate_id": candidate_id,
            "recommendation_id": str(recommendation.id),
            "checkpoint_id": getattr(
                recommendation,
                "checkpoint_id",
                None,
            ),
            "specs": copy.deepcopy(recommendation.specs),
            "candidate_fingerprint": fingerprint,
            "recommendation_audit": audit,
            "agent_intervention_flags": {
                name: False for name in AGENT_FLAGS
            },
        }
        existing = self.evidence.get(candidate_id)
        if existing is not None and any(
            existing.get(key) != value for key, value in record.items()
        ):
            raise PilotExecutionError(
                f"resumed recommendation changed: {candidate_id}"
            )
        stored = self.evidence.setdefault(candidate_id, {})
        stored.update(record)
        stored.setdefault("status", "recommended")
        self._persist()

    def __call__(
        self,
        recommendation: Any,
        train_job_id: str,
    ) -> Mapping[str, float]:
        common = _experiment_module("dino_campaign")
        candidate_id = f"{self.mode}_rec_{recommendation.id}"
        record = self.evidence.setdefault(candidate_id, {})
        terminal = terminal_checkpoint_identity(
            self.sdk,
            train_job_id,
            training_epochs=self.manifest["search"]["training_epochs"],
        )
        specification = build_evaluation_spec(
            self.manifest,
            recommendation.specs,
            terminal["path"],
        )
        record.update(
            {
                "status": "standalone_evaluation",
                "train_job_id": train_job_id,
                "terminal_checkpoint": terminal,
            }
        )
        self._persist()
        map50, evaluation = common._launch_evaluation(
            self.sdk,
            self.manifest,
            specification,
            events=self.events,
            mode=self.mode,
            candidate_id=candidate_id,
        )
        record.update(
            {
                "mAP50": map50,
                "evaluation": evaluation,
                "standalone_test_metric_feeds_selection": True,
                "status": "selection_time_latency",
            }
        )
        self._persist()
        latency, latency_evidence = _launch_latency_child(
            self.sdk,
            self.manifest,
            specification,
            terminal["path"],
            record["candidate_fingerprint"],
            events=self.events,
            mode=self.mode,
            candidate_id=candidate_id,
        )
        objectives = {"mAP50": map50, **latency}
        record.update(
            {
                "status": "success",
                "selection_time_latency": latency_evidence,
                "objective_values": objectives,
                "measurement_role": "selection_time",
                "selection_time_measurements_feed_selection": True,
            }
        )
        self._persist()
        if int(recommendation.id) == 0:
            self.gate.record(
                self.mode,
                candidate_id=candidate_id,
                passed=True,
                evidence_sha256=canonical_sha256(record),
                reason="all frozen candidate gates passed",
            )
            self.gate.wait_for_release()
        return objectives

    def on_result(
        self,
        recommendation: Any,
        metric: Any,
        status: str,
    ) -> None:
        candidate_id = f"{self.mode}_rec_{recommendation.id}"
        record = self.evidence.setdefault(candidate_id, {})
        record["automl_status"] = status
        record["reported_metric"] = metric
        failed_first = False
        if str(status).lower() not in {"success", "done"}:
            record["status"] = "terminal_failure"
            record["failure_reason"] = getattr(
                recommendation,
                "failure_reason",
                None,
            )
            if int(recommendation.id) == 0:
                self.gate.record(
                    self.mode,
                    candidate_id=candidate_id,
                    passed=False,
                    evidence_sha256=None,
                    reason=record["failure_reason"]
                    or "first recommendation failed",
                )
                failed_first = True
        self._persist()
        if failed_first:
            raise PilotExecutionError(
                f"{candidate_id} failed the automatic first-candidate gate"
            )


def frozen_intent_signature(manifest: Mapping[str, Any]) -> str:
    """Hash preregistered intent while excluding live qualification outcome."""
    value = copy.deepcopy(dict(manifest))
    value.pop("manifest_sha256", None)
    qualification = value.pop("qualification_evidence", {})
    value["qualification_contract"] = {
        "contract_path": qualification.get("contract_path"),
        "contract_sha256": qualification.get("contract_sha256"),
    }
    # The input-profile digest includes the qualification runtime directory,
    # whose completion and handoff files are expected to appear after the
    # automatic trigger starts. Every static input is represented explicitly
    # elsewhere in this signature.
    value.pop("input_profile_sha256", None)
    value.pop("ptms", None)
    value["execution"]["submission_ready"] = False
    for mode in value["modes"]:
        mode["allowed_ptm_ids"] = []
    return canonical_sha256(value)


def wait_for_launch_authorization(
    preregistered_manifest: Mapping[str, Any],
    *,
    inputs: Mapping[str, Any],
    experiment_dir: str | Path,
    runtime_root: str | Path,
    poll_seconds: float = DEFAULT_TRIGGER_POLL_SECONDS,
    timeout_seconds: float = DEFAULT_TRIGGER_TIMEOUT_SECONDS,
    manifest_builder: Callable[[], Mapping[str, Any]] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Wait for typed qualification/registry readiness, then release exactly once."""
    if poll_seconds <= 0 or timeout_seconds <= 0:
        raise ValueError("automatic trigger timing values must be positive")
    frozen = frozen_intent_signature(preregistered_manifest)
    build_current = manifest_builder or (
        lambda: seal_manifest(
            build_manifest(inputs, experiment_dir=experiment_dir)
        )
    )
    root = Path(runtime_root)
    started = monotonic()
    attempt = 0
    while True:
        attempt += 1
        current = dict(build_current())
        validate_manifest(current)
        if frozen_intent_signature(current) != frozen:
            raise PilotExecutionError(
                "automatic trigger detected preregistered pilot drift"
            )
        ready = current["execution"]["submission_ready"] is True
        atomic_json(
            root / "automatic_trigger_status.json",
            {
                "schema_version": 1,
                "campaign_id": current["campaign_id"],
                "automatic_trigger": True,
                "attempt": attempt,
                "status": "ready" if ready else "waiting",
                "frozen_intent_sha256": frozen,
                "current_manifest_sha256": current["manifest_sha256"],
                "qualification_blockers": copy.deepcopy(
                    current["qualification_evidence"]["blockers"]
                ),
                "sdk_constructed": False,
                "slurm_jobs_submitted": False,
            },
        )
        if ready:
            assert_launchable(
                current,
                inputs=inputs,
                experiment_dir=experiment_dir,
            )
            atomic_json(root / "launch_manifest.json", current)
            return current
        if monotonic() - started >= timeout_seconds:
            raise PilotExecutionError(
                "automatic trigger timed out before pilot authorization"
            )
        sleeper(poll_seconds)


def launch_plan(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "submission_ready": manifest["execution"]["submission_ready"],
        "qualification_blockers": copy.deepcopy(
            manifest["qualification_evidence"]["blockers"]
        ),
        "qualified_ptm_ids": [item["id"] for item in manifest["ptms"]],
        "ptm_representation": manifest["ptm_representation"],
        "mode_jobs": [
            {
                "mode": mode,
                "job_id": _mode_record(manifest, mode)["job_id"],
                "objective": copy.deepcopy(
                    _mode_record(manifest, mode)["objective"]
                ),
                "settings": mode_settings(manifest, mode),
                "candidate_budget": 20,
                "initial_observation_ids": [],
            }
            for mode in MODES
        ],
        "first_candidate_gate": copy.deepcopy(
            manifest["first_candidate_gate"]
        ),
        "per_candidate_children": [
            "eight_gpu_ten_epoch_training",
            "eight_gpu_standalone_evaluation",
            "eight_replica_stabilized_latency",
        ],
        "slurm": {
            "nodes_per_child": 1,
            "gpus_per_child": 8,
            "sqsh_path": manifest["runtime"]["sqsh_path"],
            "sqsh_sha256": manifest["runtime"]["sqsh_sha256"],
        },
        "search_space": copy.deepcopy(manifest["search"]["space"]),
        "agent_intervention_flags": {
            name: False for name in AGENT_FLAGS
        },
    }


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def assert_launchable(
    manifest: Mapping[str, Any],
    *,
    inputs: Mapping[str, Any],
    experiment_dir: str | Path = HERE,
) -> None:
    """Rebuild every live input before any SDK construction."""
    validate_manifest(manifest)
    if manifest["execution"]["submission_ready"] is not True:
        raise PilotExecutionError("pilot manifest is not submission-ready")
    current = seal_manifest(
        build_manifest(inputs, experiment_dir=experiment_dir)
    )
    if current != manifest:
        raise PilotExecutionError(
            "pilot inputs, qualification, registry, source, or code changed "
            "after sealing; reseal instead of mutating the campaign"
        )
    decision = audit_pilot_qualification(
        inputs,
        experiment_dir=experiment_dir,
    )
    if decision.to_dict() != manifest["qualification_evidence"]:
        raise PilotExecutionError(
            "qualification decision changed after sealing"
        )
    try:
        decision.assert_runtime_ready()
    except PilotQualificationError as exc:
        raise PilotExecutionError(str(exc)) from exc
    repository = Path(manifest["source"]["repository"])
    if (
        _git(repository, "rev-parse", "HEAD")
        != manifest["source"]["commit"]
        or _git(repository, "status", "--porcelain")
    ):
        raise PilotExecutionError(
            "pilot launch requires its exact clean sealed source commit"
        )
    for label, path_key, revision_key in (
        ("SDK", "sdk_dir", "sdk_revision"),
        ("skills", "skill_dir", "skill_revision"),
    ):
        path = Path(manifest["runtime"][path_key])
        if (
            _git(path, "rev-parse", "HEAD")
            != manifest["runtime"][revision_key]
            or _git(path, "status", "--porcelain")
        ):
            raise PilotExecutionError(f"pinned {label} checkout changed")
    for filename, digest in (
        ("spec_template_train.yaml", "train_template_sha256"),
        ("spec_template_evaluate.yaml", "evaluate_template_sha256"),
    ):
        _load_skill_template(
            manifest,
            filename=filename,
            digest_field=digest,
        )


def build_live_runtime_preflight(
    manifest: Mapping[str, Any],
    *,
    inputs: Mapping[str, Any],
    experiment_dir: str | Path,
    cache_root: str | Path,
) -> Any:
    """Build the production typed report; JSON cannot restore this state."""
    assert_launchable(
        manifest,
        inputs=inputs,
        experiment_dir=experiment_dir,
    )
    from tao_automl.ptm_preflight import (
        AtomicArtifactCache,
        NGCCredential,
        NGCHTTPSClient,
        PTMCheckpointPreflight,
    )
    from tao_automl.ptm_registry import load_ptm_registry

    decision = audit_pilot_qualification(
        inputs,
        experiment_dir=experiment_dir,
    )
    credential = NGCCredential.from_environment()
    preflight = PTMCheckpointPreflight(
        registry=load_ptm_registry(),
        cache=AtomicArtifactCache(Path(cache_root)),
        ngc_client=NGCHTTPSClient(credential),
        load_smoke=evidence_load_callback(decision),
    )
    report = preflight.run(
        model="grounding_dino",
        task="grounded_object_detection",
        tao_version=manifest["runtime"]["tao_version"],
    )
    expected = tuple(sorted(item["id"] for item in manifest["ptms"]))
    observed = tuple(item.checkpoint_id for item in report.prepared)
    if observed != expected:
        raise PilotExecutionError(
            "live typed preflight did not prepare exactly the qualified PTMs"
        )
    return report


def execution_checkpoint_artifacts(
    manifest: Mapping[str, Any],
    report: Any,
) -> dict[str, dict[str, Any]]:
    sealed = {item["id"]: item["artifact"] for item in manifest["ptms"]}
    prepared = {
        item.checkpoint_id: item.checkpoint for item in report.prepared
    }
    if set(sealed) != set(prepared):
        raise PilotExecutionError(
            "sealed and live-preflight PTM inventories differ"
        )
    result = {}
    for checkpoint_id in sorted(sealed):
        artifact = sealed[checkpoint_id]
        verified = prepared[checkpoint_id]
        if (
            not str(artifact.get("slurm_path", "")).startswith("/lustre/")
            or artifact.get("sha256") != verified.sha256
            or artifact.get("size_bytes") != verified.size_bytes
        ):
            raise PilotExecutionError(
                "Lustre PTM projection changed content identity for "
                f"{checkpoint_id}"
            )
        result[checkpoint_id] = {
            "path": artifact["slurm_path"],
            "sha256": artifact["sha256"],
            "size_bytes": artifact["size_bytes"],
        }
    return result


def skill_base_model_defaults(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return _load_skill_template(
        manifest,
        filename="spec_template_train.yaml",
        digest_field="train_template_sha256",
    )


def build_runtime_inventory(
    manifest: Mapping[str, Any],
    *,
    mode: str,
    report: Any,
) -> Any:
    from tao_automl.objectives import parse_objective_config
    from tao_automl.ptm_runtime import resolve_ptm_runtime_inventory

    objective = parse_objective_config(mode_settings(manifest, mode))
    resolved = resolve_ptm_runtime_inventory(
        report=report,
        objective_config=objective,
        base_model_defaults=skill_base_model_defaults(manifest),
        profile_overrides=nested_spec_overrides(manifest),
        user_overrides={},
        ptm_policy="all",
        model="grounding_dino",
        algorithm="bayesian",
        execution_checkpoint_artifacts=execution_checkpoint_artifacts(
            manifest,
            report,
        ),
    )
    expected_paths = {
        item["id"]: item["artifact"]["slurm_path"]
        for item in manifest["ptms"]
    }
    if {
        arm.checkpoint_id: arm.checkpoint_path for arm in resolved.arms
    } != expected_paths:
        raise PilotExecutionError(
            "resolved runtime changed sealed PTM execution paths"
        )
    return resolved


def run_mode(
    *,
    manifest: Mapping[str, Any],
    mode: str,
    sdk: Any,
    resolved_ptm_inventory: Any,
    workspace_path: str | Path,
    eval_latency_fn: Callable[[Any, str], Mapping[str, float]],
    on_recommendation: Callable[[Any], None] | None = None,
    on_result: Callable[[Any, Any, str], None] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run one independent production mode with its own observation state."""
    from tao_automl.ptm_runtime import ResolvedPTMRuntimeInventory
    from tao_automl.runner import AutoMLRunner

    if not isinstance(
        resolved_ptm_inventory,
        ResolvedPTMRuntimeInventory,
    ):
        raise TypeError(
            "resolved_ptm_inventory must be the live typed production object"
        )
    expected = tuple(sorted(item["id"] for item in manifest["ptms"]))
    if (
        resolved_ptm_inventory.mode != mode
        or resolved_ptm_inventory.checkpoint_ids != expected
    ):
        raise PilotExecutionError(
            "resolved PTM inventory differs from the sealed mode/population"
        )
    runner = AutoMLRunner(
        sdk=sdk,
        skill_dir=Path(manifest["runtime"]["skill_dir"]),
        action="train",
        poll_interval=10,
    )
    return runner.run(
        train_dataset_uri="",
        eval_dataset_uri="",
        workspace_id=f"{manifest['campaign_id']}-{mode}",
        image=manifest["runtime"]["sqsh_path"],
        automl_settings=mode_settings(manifest, mode),
        automl_hyperparameters=list(SEARCH_PARAMETERS),
        custom_param_ranges=custom_ranges(manifest),
        workspace_path=str(workspace_path),
        spec_overrides=spec_overrides(manifest),
        metric_extractor=lambda _logs, _metric: None,
        eval_fn=eval_latency_fn,
        on_recommendation=on_recommendation,
        on_result=on_result,
        resume=resume,
        ptm_aware_runtime=True,
        resolved_ptm_inventory=resolved_ptm_inventory,
        gpu_count=8,
        num_nodes=1,
        partition=manifest["runtime"]["partition"],
        account=manifest["runtime"]["account"],
    )


def run_mode_with_candidate_children(
    *,
    manifest: Mapping[str, Any],
    mode: str,
    sdk: Any,
    resolved_ptm_inventory: Any,
    runtime_root: str | Path,
    resume: bool = False,
) -> dict[str, Any]:
    root = Path(runtime_root)
    gate = AutomaticFirstCandidateGate(
        root / "first_candidate_gate",
        manifest,
    )
    evaluator = GroundingDINOCandidateEvaluator(
        sdk=sdk,
        manifest=manifest,
        mode=mode,
        runtime_root=root,
        gate=gate,
    )
    return run_mode(
        manifest=manifest,
        mode=mode,
        sdk=sdk,
        resolved_ptm_inventory=resolved_ptm_inventory,
        workspace_path=root / mode / "workspace",
        eval_latency_fn=evaluator,
        on_recommendation=evaluator.on_recommendation,
        on_result=evaluator.on_result,
        resume=resume,
    )


def launch_mode_controllers(
    *,
    manifest: Mapping[str, Any],
    report: Any,
    runtime_root: str | Path,
    resume: bool = False,
    sdk_factory: Callable[[str, Path], Any] | None = None,
    inventory_builder: Callable[..., Any] = build_runtime_inventory,
    mode_runner: Callable[..., dict[str, Any]] = (
        run_mode_with_candidate_children
    ),
) -> dict[str, Any]:
    """Run the three independent objective-aware controllers concurrently."""
    root = Path(runtime_root)
    root.mkdir(parents=True, exist_ok=True)
    inventories = {
        mode: inventory_builder(manifest, mode=mode, report=report)
        for mode in MODES
    }
    for mode, inventory in inventories.items():
        atomic_json(
            root / mode / "resolved_ptm_runtime_inventory.json",
            inventory.to_dict(),
        )
    atomic_json(root / "live_ptm_preflight.json", report.to_dict())
    if sdk_factory is None:
        from tao_sdk.platforms.slurm import SlurmSDK

        expected_root = Path(manifest["runtime"]["sdk_dir"]).resolve()
        imported = Path(sys.modules[SlurmSDK.__module__].__file__).resolve()
        if expected_root not in imported.parents:
            raise PilotExecutionError(
                "SLURM SDK import did not resolve from the pinned checkout"
            )

        def sdk_factory(mode: str, state_file: Path) -> Any:
            del mode
            return SlurmSDK(poll_interval=10, state_file=state_file)

    sdks = {}
    for mode in MODES:
        state_file = root / mode / "slurm_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        sdks[mode] = sdk_factory(mode, state_file)
    controller_gate = AutomaticFirstCandidateGate(
        root / "first_candidate_gate",
        manifest,
        poll_seconds=1,
    )
    outcomes: dict[str, Any] = {}
    failures: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=3,
        thread_name_prefix="grounding-dino-automl",
    ) as pool:
        futures = {
            pool.submit(
                mode_runner,
                manifest=manifest,
                mode=mode,
                sdk=sdks[mode],
                resolved_ptm_inventory=inventories[mode],
                runtime_root=root,
                resume=resume,
            ): mode
            for mode in MODES
        }
        for future in concurrent.futures.as_completed(futures):
            mode = futures[future]
            try:
                outcomes[mode] = future.result()
            except BaseException as exc:
                failures[mode] = f"{type(exc).__name__}: {exc}"
                gate_path = root / "first_candidate_gate" / f"{mode}.json"
                if not gate_path.exists():
                    controller_gate.record(
                        mode,
                        candidate_id=f"{mode}_rec_0",
                        passed=False,
                        evidence_sha256=None,
                        reason=(
                            "mode controller failed before candidate-zero "
                            f"authorization: {type(exc).__name__}: {exc}"
                        ),
                    )
    completion = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "status": "success" if not failures else "terminal_with_failures",
        "mode_results": outcomes,
        "mode_failures": failures,
        "automatic_first_candidate_release": True,
        "agent_intervention_flags": {
            name: False for name in AGENT_FLAGS
        },
    }
    atomic_json(root / "completion.json", completion)
    if failures:
        raise PilotExecutionError(
            "one or more independent mode controllers failed: "
            + "; ".join(
                f"{mode}={reason}"
                for mode, reason in sorted(failures.items())
            )
        )
    return completion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=DEFAULT_RUNTIME_ROOT,
    )
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--automatic-trigger", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--env-file", type=Path, default=ENV_PATH)
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=DEFAULT_TRIGGER_POLL_SECONDS,
    )
    parser.add_argument(
        "--trigger-timeout-seconds",
        type=float,
        default=DEFAULT_TRIGGER_TIMEOUT_SECONDS,
    )
    args = parser.parse_args(argv)
    inputs = read_json(args.inputs)
    experiment_dir = args.inputs.resolve().parent
    if args.manifest.is_file():
        preregistered = load_manifest(args.manifest)
    else:
        preregistered = seal_manifest(
            build_manifest(inputs, experiment_dir=experiment_dir)
        )
    if not args.launch and not args.automatic_trigger:
        plan = launch_plan(preregistered)
        atomic_json(args.runtime_root / "launch_plan.json", plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.automatic_trigger and not args.manifest.is_file():
        raise PilotExecutionError(
            "automatic trigger requires a root-integrated sealed manifest"
        )
    if args.automatic_trigger:
        manifest = wait_for_launch_authorization(
            preregistered,
            inputs=inputs,
            experiment_dir=experiment_dir,
            runtime_root=args.runtime_root,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.trigger_timeout_seconds,
        )
    else:
        manifest = preregistered
        assert_launchable(
            manifest,
            inputs=inputs,
            experiment_dir=experiment_dir,
        )
        atomic_json(args.runtime_root / "launch_manifest.json", manifest)
    load_env_file = _experiment_module("dino_campaign").load_env_file
    loaded_names = load_env_file(args.env_file)
    configure_slurm_runtime(manifest)
    report = build_live_runtime_preflight(
        manifest,
        inputs=inputs,
        experiment_dir=experiment_dir,
        cache_root=args.runtime_root / "verified_ptm_cache",
    )
    atomic_json(
        args.runtime_root / "submission_provenance.json",
        {
            "schema_version": 1,
            "campaign_id": manifest["campaign_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "frozen_intent_sha256": frozen_intent_signature(
                preregistered
            ),
            "loaded_secret_keys": list(loaded_names),
            "secret_values_recorded": False,
            "ptm_preflight_report_sha256": report.report_sha256,
            "execution_checkpoint_artifacts": (
                execution_checkpoint_artifacts(manifest, report)
            ),
            "sqsh_path": manifest["runtime"]["sqsh_path"],
            "sqsh_sha256": manifest["runtime"]["sqsh_sha256"],
            "nodes_per_child": 1,
            "gpus_per_child": 8,
            "candidate_budget_per_mode": 20,
            "training_epochs": 10,
            "cpu_runs": 0,
            "smoke_runs": 0,
            "agent_intervention_flags": {
                name: False for name in AGENT_FLAGS
            },
        },
    )
    completion = launch_mode_controllers(
        manifest=manifest,
        report=report,
        runtime_root=args.runtime_root,
        resume=args.resume,
    )
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
