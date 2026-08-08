#!/usr/bin/env python3

"""Run three independent objective-aware Mask Grounding DINO AutoML jobs on SLURM.

There is deliberately no CPU/model-smoke or mini-step execution path.  The
automatic trigger waits for immutable direct-full-run PTM qualification and
an exact evidence-bound runtime eligibility decision, then starts all three
mode controllers.
Each candidate trains, runs standalone full validation, and measures stabilized
latency on one node/eight A100s in the pinned TAO SQSH.  The first successful
candidate from every mode must pass all gates before the remaining budget is
released automatically.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import multiprocessing as mp
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from tao_automl.ptm_registry import canonical_sha256
from tao_automl.recommendation_audit import validate_recommendation_audit
from tao_automl.selection import canonical_spec_fingerprint

try:
    from experiments.cross_model_automl_20260729 import checkpoint_resume
except ModuleNotFoundError:  # pragma: no cover - pytest direct-path import
    import checkpoint_resume
from . import campaign_contract
from .qualification_gate import (
    QualificationDecision,
    QualificationLoadEvidence,
    audit_qualification,
)

try:
    from experiments.cross_model_automl_20260729.dino_campaign import (
        run_campaign as workflow_support,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution
    repository = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repository))
    from experiments.cross_model_automl_20260729.dino_campaign import (
        run_campaign as workflow_support,
    )


HERE = Path(__file__).resolve().parent
ENV_PATH = Path("/localhome/local-rarunachalam/.tao/config.env")
DEFAULT_CONTRACT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask_grounding_dino_coco2017_three_mode_v4/campaign.v4.json"
)
DEFAULT_RUNTIME_ROOT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/mask_grounding_dino_coco2017_three_mode_v4"
)
TERMINAL_JOB_STATUSES = frozenset({"Complete", "Error", "Canceled"})
SUCCESS_RECOMMENDATION_STATUSES = frozenset({"success", "done"})

CampaignExecutionError = workflow_support.CampaignExecutionError
atomic_json = workflow_support.atomic_json
append_jsonl = workflow_support.append_jsonl
load_env_file = workflow_support.load_env_file
remote_output = workflow_support.remote_output
utc_timestamp = workflow_support.utc_timestamp
text_sha256 = workflow_support.text_sha256
_local_lustre_path = workflow_support._local_lustre_path
_merge_spec = workflow_support._merge_spec


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def load_contract(path: str | Path) -> dict[str, Any]:
    return campaign_contract.validate_contract(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def configure_slurm_runtime(contract: Mapping[str, Any]) -> None:
    runtime = contract["runtime"]
    sdk_dir = runtime["sdk_dir"]
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
            # A verified SQSH path is passed directly; no conversion job exists.
            "SLURM_USE_SQSH": "false",
            "SLURM_USE_REQUEUE": "true",
            "SLURM_TIME_HOURS": str(runtime["time_hours"]),
            "SLURM_TIMEOUT_HOURS": str(runtime["timeout_hours"]),
            "SLURM_MAX_GPUS_PER_NODE": "8",
            "SLURM_PARTITION": runtime["partition"],
            "SLURM_ACCOUNT": runtime["account"],
            "SLURM_BASE_RESULTS_DIR": runtime["base_results_dir"],
            "SLURM_CONTAINER_MOUNTS": runtime["container_mounts"],
            "SLURM_MAX_JOB_RETRIES": str(runtime["max_job_retries"]),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )


def _offline_text_environment(
    contract: Mapping[str, Any],
) -> dict[str, str]:
    environment = dict(contract["text_encoder"]["offline_environment"])
    if environment != {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }:
        raise CampaignExecutionError(
            "Mask Grounding DINO offline text environment changed"
        )
    return {**environment, "TOKENIZERS_PARALLELISM": "false"}


class FirstCandidateTrainingReuseSDK:
    """Reuse one exact completed rec-0 train job in a fresh controller."""

    def __init__(
        self,
        delegate: Any,
        source: Any,
        record: Mapping[str, Any],
    ):
        self._delegate = delegate
        self._source = source
        self._record = copy.deepcopy(dict(record))
        self._source_job_id = str(record["source_train_job_id"])
        self._armed = False
        self._reused = False
        source_job = source.get_job(self._source_job_id)
        if (
            not isinstance(source_job, Mapping)
            or source_job.get("status") != "Complete"
            or source_job.get("results_dir") != record["source_results_dir"]
        ):
            raise CampaignExecutionError(
                "first-candidate source training job is not durably reusable"
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def arm_first_candidate_training_reuse(self, recommendation: Any) -> None:
        rec_id = int(recommendation.id)
        if rec_id != 0:
            return
        if self._armed or self._reused:
            raise CampaignExecutionError(
                "first-candidate training reuse was armed more than once"
            )
        fingerprint = canonical_spec_fingerprint(recommendation.specs)
        if (
            fingerprint != self._record["candidate_fingerprint"]
            or canonical_sha256(recommendation.specs)
            != self._record["specs_sha256"]
            or _ptm_id(
                recommendation.recommendation_audit,
                (self._record["checkpoint_id"],),
            )
            != self._record["checkpoint_id"]
        ):
            raise CampaignExecutionError(
                "fresh rec-0 differs from the completed training candidate"
            )
        self._armed = True

    def create_job(self, *args: Any, **kwargs: Any) -> Any:
        if not self._armed:
            return self._delegate.create_job(*args, **kwargs)
        self._armed = False
        self._reused = True
        source_job = self._source.get_job(self._source_job_id)
        return SimpleNamespace(
            id=self._source_job_id,
            backend_job_id=source_job.get("backend_job_id", self._source_job_id),
        )

    def _for_source(self, job_id: str) -> Any | None:
        return self._source if job_id == self._source_job_id else None

    def get_job_status(self, job_id: str) -> Any:
        sdk = self._for_source(job_id) or self._delegate
        return sdk.get_job_status(job_id)

    def get_job_logs(self, job_id: str, tail: int | None = None) -> str:
        sdk = self._for_source(job_id) or self._delegate
        return sdk.get_job_logs(job_id, tail=tail)

    def get_job_results_dir(self, job_id: str) -> str:
        sdk = self._for_source(job_id) or self._delegate
        return sdk.get_job_results_dir(job_id)

    def get_failure_analysis(self, job_id: str) -> Any:
        sdk = self._for_source(job_id) or self._delegate
        return sdk.get_failure_analysis(job_id)

    def get_job(self, job_id: str) -> Any:
        sdk = self._for_source(job_id) or self._delegate
        return sdk.get_job(job_id)

    def get_checkpoints(self, job_id: str) -> list[str]:
        sdk = self._for_source(job_id) or self._delegate
        return sdk.get_checkpoints(job_id)

    def cancel_job(self, job_id: str) -> bool:
        sdk = self._for_source(job_id) or self._delegate
        return sdk.cancel_job(job_id)

    def assert_terminal_checkpoint(
        self,
        job_id: str,
        checkpoint: Mapping[str, Any],
    ) -> None:
        if job_id == self._source_job_id and dict(checkpoint) != self._record[
            "terminal_checkpoint"
        ]:
            raise CampaignExecutionError(
                "reused first-candidate terminal checkpoint identity changed"
            )

    def training_reuse_evidence(self, job_id: str) -> dict[str, Any] | None:
        if job_id != self._source_job_id:
            return None
        return {
            "schema_version": 1,
            "kind": "completed_training_job_reuse",
            "source_train_job_id": self._source_job_id,
            "source_results_dir": self._record["source_results_dir"],
            "source_state_db_sha256": self._record[
                "source_state_db_sha256"
            ],
            "source_candidate_evidence_sha256": self._record[
                "source_candidate_evidence_sha256"
            ],
            "training_job_reused": True,
            "new_training_job_submitted": False,
            "objective_values_reused": False,
            "standalone_evaluation_reused": False,
            "latency_measurement_reused": False,
        }


def _remote_file_identity(path: str) -> dict[str, Any]:
    script = (
        "import hashlib,json,pathlib,sys;"
        "p=pathlib.Path(sys.argv[1]);"
        "h=hashlib.sha256();"
        "f=p.open('rb');"
        "[(h.update(c)) for c in iter(lambda:f.read(1048576),b'')];"
        "f.close();"
        "print(json.dumps({'path':str(p),'size_bytes':p.stat().st_size,"
        "'sha256':h.hexdigest()}))"
    )
    try:
        return json.loads(
            remote_output(
                f"python3 -c {shlex.quote(script)} {shlex.quote(path)}"
            )
        )
    except Exception as exc:
        raise CampaignExecutionError(
            f"remote artifact unavailable or unreadable: {path}"
        ) from exc


def _verify_dataset_remote(contract: Mapping[str, Any]) -> dict[str, Any]:
    dataset = contract["dataset"]
    root = dataset["prepared_root"]
    expected = {
        "train_images": (f"{root}/images/train2017", 118287),
        "validation_images": (f"{root}/images/val2017", 5000),
    }
    count_script = (
        "import json,pathlib,sys;"
        "print(json.dumps({p:sum(1 for x in pathlib.Path(p).iterdir() "
        "if x.is_file()) for p in sys.argv[1:]}))"
    )
    counts = json.loads(
        remote_output(
            "python3 -c "
            f"{shlex.quote(count_script)} "
            + " ".join(shlex.quote(path) for path, _ in expected.values())
        )
    )
    for label, (path, count) in expected.items():
        if counts.get(path) != count:
            raise CampaignExecutionError(
                f"COCO2017 {label} count changed: {counts.get(path)} != {count}"
            )
    required_files = {
        "train_odvg_jsonl": (
            f"{root}/tao/mask_grounding_dino/train/"
            "instances_train2017_odvg.jsonl",
            dataset["train_odvg_jsonl_sha256"],
        ),
        "train_odvg_label_map": (
            f"{root}/tao/mask_grounding_dino/train/"
            "instances_train2017_odvg_labelmap.json",
            dataset["train_odvg_label_map_sha256"],
        ),
        "contiguous_validation_json": (
            dataset["contiguous_validation_json_path"],
            dataset["contiguous_validation_json_sha256"],
        ),
        "contiguous_validation_manifest": (
            dataset["contiguous_validation_manifest_path"],
            dataset["contiguous_validation_manifest_sha256"],
        ),
    }
    required_file_identities = {}
    for label, (path, expected_sha) in required_files.items():
        identity = _remote_file_identity(path)
        if identity["sha256"] != expected_sha:
            raise CampaignExecutionError(
                f"COCO2017 {label} identity changed"
            )
        required_file_identities[label] = identity
    stage_identity = _remote_file_identity(
        dataset["stage_manifest_lustre_path"]
    )
    if stage_identity["sha256"] != dataset["stage_manifest_sha256"]:
        raise CampaignExecutionError(
            "Lustre dataset stage manifest differs from the frozen record"
        )
    file_manifest_identity = _remote_file_identity(
        dataset["remote_file_manifest_path"]
    )
    if file_manifest_identity["sha256"] != dataset["manifest_sha256"]:
        raise CampaignExecutionError(
            "Lustre COCO2017 file manifest differs from the frozen record"
        )
    stage_reader = (
        "import json,sys;"
        "d=json.load(open(sys.argv[1]));v=d['datasets']['coco2017'];"
        "print(json.dumps({'remote_read_only':v['remote_read_only'],"
        "'remote_writable_entries_after_lock':"
        "v['remote_writable_entries_after_lock'],"
        "'remote_sha256sum_check':"
        "v['file_manifest']['remote_sha256sum_check'],"
        "'remote_file_set_check':"
        "v['file_manifest']['remote_file_set_check']}))"
    )
    stage = json.loads(
        remote_output(
            f"python3 -c {shlex.quote(stage_reader)} "
            f"{shlex.quote(dataset['stage_manifest_lustre_path'])}"
        )
    )
    if stage != {
        "remote_read_only": True,
        "remote_writable_entries_after_lock": 0,
        "remote_sha256sum_check": "passed",
        "remote_file_set_check": "passed",
    }:
        raise CampaignExecutionError(
            "Lustre COCO2017 stage record is not final and read-only"
        )
    writable = remote_output(
        "find "
        f"{shlex.quote(root)} "
        "-type f -perm /222 -print -quit"
    ).strip()
    if writable:
        raise CampaignExecutionError(
            f"Lustre COCO2017 staging contains a writable file: {writable}"
        )
    derived_root = str(
        Path(dataset["contiguous_validation_json_path"]).parent
    )
    derived_writable = remote_output(
        "find "
        f"{shlex.quote(derived_root)} "
        "-maxdepth 1 -type f -perm /222 -print -quit"
    ).strip()
    if derived_writable:
        raise CampaignExecutionError(
            "derived contiguous COCO validation stage contains a writable "
            f"file: {derived_writable}"
        )
    text_encoder_files = contract["runtime"].get("text_encoder_files")
    if (
        not isinstance(text_encoder_files, list)
        or len(text_encoder_files) != 5
    ):
        raise CampaignExecutionError(
            "frozen text-encoder file inventory is unavailable"
        )
    text_encoder_identities = {}
    for item in text_encoder_files:
        identity = _remote_file_identity(str(item["path"]))
        if (
            identity["size_bytes"] != item["size_bytes"]
            or identity["sha256"] != item["sha256"]
            or not identity["path"].startswith(
                campaign_contract.FROZEN_TEXT_ENCODER_ROOT + "/"
            )
        ):
            raise CampaignExecutionError(
                "frozen text-encoder artifact identity changed"
            )
        text_encoder_identities[Path(identity["path"]).name] = identity
    return {
        "prepared_root": root,
        "counts": {
            label: counts[path]
            for label, (path, _) in expected.items()
        },
        "required_files": required_file_identities,
        "text_encoder_files": text_encoder_identities,
        "content_sha256": dataset["content_sha256"],
        "stage_manifest_sha256": dataset["stage_manifest_sha256"],
        "remote_read_only": True,
        "remote_writable_entries_after_lock": 0,
    }


def verify_local_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Revalidate sealed code, wheel, skills, SDK, and dataset metadata."""
    runtime = contract["runtime"]
    repository = Path(runtime["repository"]).resolve()
    if (
        _git(repository, "rev-parse", "HEAD")
        != runtime["source_commit"]
        or _git(repository, "status", "--porcelain")
    ):
        raise CampaignExecutionError(
            "sealed AutoML source commit or clean state changed"
        )
    if (
        _git(Path(runtime["sdk_dir"]), "rev-parse", "HEAD")
        != runtime["sdk_commit"]
        or _git(
            Path(runtime["skills_repository"]), "rev-parse", "HEAD"
        )
        != runtime["skills_commit"]
    ):
        raise CampaignExecutionError("sealed SDK or skills commit changed")
    identities = {
        "ddp_strategy_audit": (
            HERE / "ddp_strategy_audit.v2.json",
            contract["launcher_integrity"]["ddp_strategy_audit_sha256"],
        ),
        "wheel": (
            runtime["wheel_path"],
            runtime["wheel_sha256"],
        ),
        "dataset_manifest": (
            contract["dataset"]["manifest_path"],
            contract["dataset"]["manifest_sha256"],
        ),
        "dataset_stage_manifest": (
            contract["dataset"]["stage_manifest_path"],
            contract["dataset"]["stage_manifest_sha256"],
        ),
        "contiguous_validation_manifest": (
            contract["dataset"][
                "contiguous_validation_manifest_local_path"
            ],
            contract["dataset"][
                "contiguous_validation_manifest_sha256"
            ],
        ),
        "text_encoder_stage": (
            runtime["text_encoder_stage_path"],
            runtime["text_encoder_stage_sha256"],
        ),
        "ptm_stage_manifest": (
            runtime["ptm_stage_manifest_path"],
            runtime["ptm_stage_manifest_sha256"],
        ),
        "predecessor_qualification": (
            runtime["predecessor_failure_evidence"]["path"],
            runtime["predecessor_failure_evidence"].get(
                "file_sha256",
                runtime["predecessor_failure_evidence"].get("sha256"),
            ),
        ),
        "qualification_evidence": (
            runtime["qualification_evidence_path"],
            runtime["runtime_local_eligibility"][
                "qualification_file_sha256"
            ],
        ),
        "campaign_contract": (
            HERE / "campaign_contract.py",
            contract["launcher_integrity"][
                "campaign_contract_sha256"
            ],
        ),
        "qualification_gate": (
            HERE / "qualification_gate.py",
            contract["launcher_integrity"][
                "qualification_gate_sha256"
            ],
        ),
        "qualification_campaign": (
            HERE / "qualification_campaign.py",
            contract["launcher_integrity"][
                "qualification_campaign_sha256"
            ],
        ),
        "run_campaign": (
            HERE / "run_campaign.py",
            contract["launcher_integrity"]["run_campaign_sha256"],
        ),
        "latency_worker": (
            HERE / "mask_grounding_dino_latency_worker.py",
            contract["launcher_integrity"][
                "mask_grounding_dino_latency_worker_sha256"
            ],
        ),
        "checkpoint_resume": (
            HERE.parent / "checkpoint_resume.py",
            contract["launcher_integrity"]["checkpoint_resume_sha256"],
        ),
    }
    eligibility = runtime["runtime_local_eligibility"]
    if eligibility.get("qualification_successor_version") == 5:
        identities.update(
            {
                "qualification_contract": (
                    runtime["qualification_contract_path"],
                    runtime["qualification_contract_file_sha256"],
                ),
                "training_qualification": (
                    eligibility["training_qualification_path"],
                    eligibility["training_qualification_file_sha256"],
                ),
                "training_qualification_contract": (
                    eligibility["training_qualification_contract_path"],
                    eligibility[
                        "training_qualification_contract_file_sha256"
                    ],
                ),
            }
        )
    resume_predecessor = runtime.get("resume_predecessor_contract")
    if isinstance(resume_predecessor, Mapping):
        identities["resume_predecessor_contract"] = (
            resume_predecessor["path"],
            resume_predecessor["file_sha256"],
        )
    first_candidate_reuse = runtime.get(
        "first_candidate_training_reuse"
    )
    if isinstance(first_candidate_reuse, Mapping):
        identities["first_candidate_source_contract"] = (
            first_candidate_reuse["source_contract"]["path"],
            first_candidate_reuse["source_contract"]["file_sha256"],
        )
        for mode, record in first_candidate_reuse["modes"].items():
            identities[f"{mode}_first_candidate_state"] = (
                record["source_state_db"],
                record["source_state_db_sha256"],
            )
            identities[f"{mode}_first_candidate_evidence"] = (
                record["source_candidate_evidence"],
                record["source_candidate_evidence_sha256"],
            )
    evidence = {}
    for name, (path_value, expected_sha) in identities.items():
        path = Path(path_value).resolve()
        if (
            not path.is_file()
            or campaign_contract.sha256_file(path) != expected_sha
        ):
            raise CampaignExecutionError(
                f"sealed local artifact changed: {name}"
            )
        evidence[name] = {
            "path": str(path),
            "sha256": expected_sha,
        }
    return {
        "source_commit": runtime["source_commit"],
        "sdk_commit": runtime["sdk_commit"],
        "skills_commit": runtime["skills_commit"],
        "artifacts": evidence,
    }


def launch_readiness(
    contract: Mapping[str, Any],
) -> tuple[bool, list[dict[str, Any]], QualificationDecision | None]:
    """Evaluate dynamic prerequisites without launching or mutating evidence."""
    blockers: list[dict[str, Any]] = []
    decision = None
    try:
        verify_local_contract(contract)
    except Exception as exc:
        blockers.append(
            {"code": "sealed_local_contract_not_ready", "reason": str(exc)}
        )
    try:
        _verify_dataset_remote(contract)
    except Exception as exc:
        blockers.append(
            {"code": "dataset_not_ready", "reason": str(exc)}
        )
    try:
        identity = _remote_file_identity(contract["sqsh"]["path"])
        if identity["sha256"] != contract["sqsh"]["sha256"]:
            raise CampaignExecutionError("pinned SQSH SHA-256 changed")
    except Exception as exc:
        blockers.append({"code": "sqsh_not_ready", "reason": str(exc)})
    try:
        overlay = contract["runtime"]["evaluation_overlay"]
        identity = _remote_file_identity(overlay["archive_path"])
        if (
            identity["sha256"] != overlay["archive_sha256"]
            or identity["size_bytes"] != overlay["archive_size_bytes"]
        ):
            raise CampaignExecutionError(
                "selection-time evaluator overlay identity changed"
            )
        readonly = remote_output(
            "test ! -w "
            + shlex.quote(overlay["archive_path"])
            + " && echo readonly"
        ).strip()
        if readonly != "readonly":
            raise CampaignExecutionError(
                "selection-time evaluator overlay is writable"
            )
    except Exception as exc:
        blockers.append(
            {"code": "evaluation_overlay_not_ready", "reason": str(exc)}
        )
    try:
        decision = audit_qualification(
            contract["qualification_policy"]["qualification_evidence_path"],
            expected_contract=contract,
        )
        decision.assert_runtime_ready()
    except Exception as exc:
        blockers.append(
            {"code": "ptm_qualification_not_ready", "reason": str(exc)}
        )
    if decision is not None and decision.runtime_ready:
        for record in decision.qualified:
            try:
                identity = _remote_file_identity(
                    record.source_checkpoint_path
                )
                if (
                    identity["sha256"] != record.source_checkpoint_sha256
                    or identity["size_bytes"]
                    != record.source_checkpoint_size_bytes
                ):
                    raise CampaignExecutionError(
                        "source checkpoint identity changed"
                    )
            except Exception as exc:
                blockers.append(
                    {
                        "code": "ptm_artifact_not_ready",
                        "checkpoint_id": record.checkpoint_id,
                        "reason": str(exc),
                    }
                )
    return not blockers, blockers, decision


def wait_for_launch_authorization(
    contract: Mapping[str, Any],
    *,
    runtime_root: Path,
    poll_seconds: float = 30.0,
    timeout_seconds: float | None = None,
) -> QualificationDecision:
    """Wait automatically until immutable campaign prerequisites are ready."""
    started = time.monotonic()
    while True:
        ready, blockers, decision = launch_readiness(contract)
        status = {
            "schema_version": 1,
            "contract_sha256": contract["contract_sha256"],
            "automatic_trigger": True,
            "launch_authorized": ready,
            "blockers": blockers,
            "checked_at_utc": utc_timestamp(),
            "model_jobs_launched": False,
        }
        atomic_json(runtime_root / "automatic_trigger_status.json", status)
        if ready and decision is not None:
            atomic_json(
                runtime_root / "qualification_decision.json",
                decision.to_dict(),
            )
            return decision
        if (
            timeout_seconds is not None
            and time.monotonic() - started >= timeout_seconds
        ):
            raise TimeoutError(
                "automatic Mask Grounding DINO trigger timed out: "
                + ", ".join(item["code"] for item in blockers)
            )
        time.sleep(poll_seconds)


def _execution_artifacts(
    decision: QualificationDecision,
) -> dict[str, dict[str, Any]]:
    return {
        item.checkpoint_id: {
            "path": item.source_checkpoint_path,
            "sha256": item.source_checkpoint_sha256,
            "size_bytes": item.source_checkpoint_size_bytes,
        }
        for item in decision.qualified
    }


def _per_checkpoint_profiles(
    decision: QualificationDecision,
) -> dict[str, dict[str, Any]]:
    registry = decision.runtime_registry
    return {
        checkpoint_id: {
            "model": {
                "backbone": copy.deepcopy(
                    registry.checkpoint(checkpoint_id)[
                        "default_spec_overrides"
                    ]["model"]["backbone"]
                )
            }
        }
        for checkpoint_id in decision.checkpoint_ids
    }


def build_live_runtime_inventory(
    *,
    contract: Mapping[str, Any],
    decision: QualificationDecision,
    mode: str,
    cache_root: str | Path,
) -> Any:
    """Construct the production typed hierarchical PTM inventory."""
    from tao_automl.objectives import parse_objective_config
    from tao_automl.ptm_preflight import (
        AtomicArtifactCache,
        NGCCredential,
        NGCHTTPSClient,
        PTMCheckpointPreflight,
    )
    from tao_automl.ptm_runtime import resolve_ptm_runtime_inventory

    objective = parse_objective_config(
        campaign_contract.mode_settings(str(contract["campaign_id"]), mode)
    )
    report = PTMCheckpointPreflight(
        registry=decision.runtime_registry,
        cache=AtomicArtifactCache(cache_root),
        ngc_client=NGCHTTPSClient(NGCCredential.from_environment()),
        load_smoke=QualificationLoadEvidence(decision),
    ).run(
        model="mask_grounding_dino",
        task="grounded_instance_segmentation",
        tao_version="7.1.0",
    )
    if (
        not report.ok
        or tuple(item.checkpoint_id for item in report.prepared)
        != decision.checkpoint_ids
        or report.exclusions
    ):
        raise CampaignExecutionError(
            "live PTM preflight did not preserve exactly the qualified arms"
        )
    template = (
        Path(contract["runtime"]["skill_dir"])
        / "references/spec_template_train.yaml"
    )
    base_defaults = yaml.safe_load(template.read_text(encoding="utf-8"))
    resolved = resolve_ptm_runtime_inventory(
        report=report,
        objective_config=objective,
        base_model_defaults=base_defaults,
        profile_overrides=campaign_contract.profile_overrides(
            contract["dataset"]["prepared_root"]
        ),
        user_overrides=None,
        ptm_policy="all",
        model="mask_grounding_dino",
        algorithm="bayesian",
        execution_checkpoint_artifacts=_execution_artifacts(decision),
        per_checkpoint_profile_overrides=_per_checkpoint_profiles(decision),
        registry=decision.runtime_registry,
    )
    if resolved.checkpoint_ids != decision.checkpoint_ids:
        raise CampaignExecutionError(
            "hierarchical runtime omitted a qualified PTM arm"
        )
    return resolved


def _wait_for_job(
    sdk: Any,
    job_id: str,
    *,
    events: Path,
    phase: str,
    mode: str,
    candidate_id: str,
) -> str:
    previous = None
    while True:
        status = sdk.get_job_status(job_id).status
        if status != previous:
            append_jsonl(
                events,
                {
                    "event": "slurm_job_status",
                    "phase": phase,
                    "mode": mode,
                    "candidate_id": candidate_id,
                    "tao_job_id": job_id,
                    "status": status,
                    "observed_at_utc": utc_timestamp(),
                },
            )
            previous = status
        if status in TERMINAL_JOB_STATUSES:
            return status
        time.sleep(10)


def _terminal_checkpoint(
    sdk: Any,
    job_id: str,
) -> dict[str, Any]:
    """Resolve exactly one terminal epoch-two Mask Grounding DINO checkpoint."""
    root = _local_lustre_path(sdk.get_job_results_dir(job_id))
    folder = f"{root}/results_dir/train"
    script = (
        "import glob,hashlib,json,pathlib,sys;"
        "paths=sorted(glob.glob(sys.argv[1]+'/model_epoch_002_step_*.pth'));"
        "assert len(paths)==1,paths;"
        "p=pathlib.Path(paths[0]);h=hashlib.sha256();f=p.open('rb');"
        "[(h.update(c)) for c in iter(lambda:f.read(1048576),b'')];f.close();"
        "print(json.dumps({'path':str(p),'filename':p.name,"
        "'size_bytes':p.stat().st_size,'sha256':h.hexdigest()}))"
    )
    try:
        evidence = json.loads(
            remote_output(
                f"python3 -c {shlex.quote(script)} {shlex.quote(folder)}"
            )
        )
    except Exception as exc:
        raise CampaignExecutionError(
            "exact terminal Mask Grounding DINO checkpoint is unavailable or ambiguous"
        ) from exc
    if (
        not re.fullmatch(r"model_epoch_002_step_[0-9]+[.]pth", evidence["filename"])
        or not evidence["path"].startswith("/lustre/")
        or evidence["size_bytes"] < 1
    ):
        raise CampaignExecutionError(
            "terminal Mask Grounding DINO checkpoint identity is invalid"
        )
    evidence.update(
        {
            "training_epochs": campaign_contract.FROZEN_TRAINING_EPOCHS,
            "terminal_epoch_index": 2,
            "naming_contract": "model_epoch_002_step_numeric",
            "ambiguity_policy": "fail_closed",
        }
    )
    return evidence


def _status_metric(
    sdk: Any,
    job_id: str,
    *,
    action: str,
    names: tuple[str, ...],
) -> float | None:
    root = _local_lustre_path(sdk.get_job_results_dir(job_id))
    status_path = f"{root}/results_dir/{action}/status.json"
    output = remote_output(
        f"(test -f {shlex.quote(status_path)} && "
        f"tail -200 {shlex.quote(status_path)}) || true"
    )
    values = []
    for line in output.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        kpi = record.get("kpi")
        if not isinstance(kpi, Mapping):
            continue
        for name in names:
            if name in kpi:
                try:
                    value = float(kpi[name])
                except (TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(value):
                    values.append(value)
                    break
    return values[-1] if values else None


def evaluation_spec(
    contract: Mapping[str, Any],
    recommendation_specs: Mapping[str, Any],
    checkpoint: str,
) -> dict[str, Any]:
    template = (
        Path(contract["runtime"]["skill_dir"])
        / "references/spec_template_evaluate.yaml"
    )
    spec = yaml.safe_load(template.read_text(encoding="utf-8"))
    spec = _merge_spec(
        spec,
        campaign_contract.profile_overrides(
            contract["dataset"]["prepared_root"]
        ),
    )
    spec = _merge_spec(spec, recommendation_specs)
    spec["results_dir"] = ""
    spec["wandb"]["enable"] = False
    spec["dataset"]["batch_size"] = 1
    spec["dataset"]["workers"] = 8
    spec["evaluate"].update(
        {
            "num_gpus": 8,
            "gpu_ids": list(range(8)),
            "num_nodes": 1,
            "checkpoint": checkpoint,
            "trt_engine": "",
            "results_dir": "",
            "batch_size": 1,
        }
    )
    return spec


def evaluator_overlay_install_command(
    contract: Mapping[str, Any],
) -> str:
    """Build the fail-closed evaluator overlay prefix sealed by the campaign.

    Mask Grounding DINO's COCO evaluator correction is intentionally installed
    into an ephemeral ``PYTHONPATH`` tree.  The SQSH and its installed package
    remain immutable.  Qualification proved the exact overlay; selection-time
    evaluation must consume that same artifact rather than merely citing its
    qualification evidence.
    """
    overlay = contract.get("runtime", {}).get("evaluation_overlay")
    if not isinstance(overlay, Mapping):
        raise CampaignExecutionError(
            "selection-time Mask Grounding DINO evaluation requires the "
            "sealed evaluator overlay"
        )
    archive = shlex.quote(str(overlay["archive_path"]))
    digest = shlex.quote(str(overlay["archive_sha256"]))
    base = shlex.quote(str(overlay["base_site_packages"]))
    installer = shlex.quote(
        f"{overlay['archive_root']}/install_overlay.py"
    )
    return " ".join(
        [
            "mgdino_overlay_tmp=$(mktemp -d",
            "/tmp/mgdino-coco-evaluator.XXXXXX)",
            "&& test \"$(sha256sum",
            archive,
            "| awk '{print $1}')\" =",
            digest,
            "&& tar --extract --file",
            archive,
            "--directory \"$mgdino_overlay_tmp\"",
            "&& mgdino_overlay_site=\"$mgdino_overlay_tmp/site-packages\"",
            "&& mkdir -p \"$mgdino_overlay_site/nvidia_tao_pytorch\"",
            "&& cp -as",
            f"{base}/nvidia_tao_pytorch/.",
            "\"$mgdino_overlay_site/nvidia_tao_pytorch/\"",
            f"&& python \"$mgdino_overlay_tmp\"/{installer}",
            "--base-site-packages",
            base,
            "--site-packages \"$mgdino_overlay_site\"",
            "--receipt \"${TAO_RESULTS_ROOT:?}/${TAO_JOB_ID:?}/"
            "runtime_overlay/receipt.json\"",
            "&& export PYTHONPATH=\"$mgdino_overlay_site"
            "${PYTHONPATH:+:$PYTHONPATH}\"",
        ]
    )


def _launch_evaluation(
    sdk: Any,
    contract: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    events: Path,
    mode: str,
    candidate_id: str,
) -> tuple[float, dict[str, Any]]:
    from tao_sdk.script_runner import build_entrypoint

    action = yaml.safe_load(
        (
            Path(contract["runtime"]["skill_dir"])
            / "references/skill_info.yaml"
        ).read_text(encoding="utf-8")
    )["actions"]["evaluate"]
    entrypoint = build_entrypoint(
        command=action["command"],
        specs=spec,
        inputs=action["inputs"],
        outputs=action["outputs"],
        config_format=action["config_format"],
        upload_excludes=action.get("upload_excludes", []),
    )
    base_command = entrypoint["command"]
    command = (
        f"{evaluator_overlay_install_command(contract)} && (\n"
        f"{base_command}\n)"
    )
    runtime = contract["runtime"]
    job = sdk.create_job(
        image=contract["sqsh"]["path"],
        command=command,
        gpu_count=8,
        num_nodes=1,
        partition=runtime["partition"],
        account=runtime["account"],
    )
    evidence = {
        "tao_job_id": job.id,
        "status": "submitted",
        "submitted_at_utc": utc_timestamp(),
        "spec_sha256": canonical_sha256(spec),
        "base_command_sha256": text_sha256(base_command),
        "command_sha256": text_sha256(command),
        "overlay_sha256": contract["runtime"]["evaluation_overlay"][
            "archive_sha256"
        ],
        "overlay_source_commit": contract["runtime"][
            "evaluation_overlay"
        ]["source_commit"],
        "installed_package_mutated": False,
    }
    status = _wait_for_job(
        sdk,
        job.id,
        events=events,
        phase="standalone_evaluation",
        mode=mode,
        candidate_id=candidate_id,
    )
    evidence.update({"status": status, "terminal_at_utc": utc_timestamp()})
    if status != "Complete":
        logs = sdk.get_job_logs(job.id, tail=1000)
        raise CampaignExecutionError(
            f"evaluation job {job.id} ended as {status}: {logs[-3000:]}"
        )
    metric = _status_metric(
        sdk,
        job.id,
        action="evaluate",
        names=("[segm] test_mAP@50-95",),
    )
    if metric is None or not 0.0 <= metric <= 1.0:
        raise CampaignExecutionError(
            f"evaluation job {job.id} emitted no valid segm_val_mAP50_95"
        )
    evidence["result_root"] = _local_lustre_path(
        sdk.get_job_results_dir(job.id)
    )
    evidence["segm_val_mAP50_95"] = metric
    return metric, evidence


def validation_input_descriptor(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind latency input to the first 16 COCO validation records."""
    validation_dir = (
        f"{contract['dataset']['prepared_root'].rstrip('/')}/images/val2017"
    )
    annotation_path = contract["dataset"][
        "contiguous_validation_json_path"
    ]
    reader = (
        "import hashlib,json,pathlib,sys;"
        "root=pathlib.Path(sys.argv[1]);"
        "doc=json.load(open(sys.argv[2]));records=doc['images'][:16];"
        "out=[];"
        "\nfor record in records:\n"
        " p=root/record['file_name'];"
        " h=hashlib.sha256();f=p.open('rb');"
        "\n for c in iter(lambda:f.read(1048576),b''):h.update(c)\n"
        " f.close();out.append({'id':record['id'],"
        "'file_name':record['file_name'],'width':record['width'],"
        "'height':record['height'],"
        "'size_bytes':p.stat().st_size,"
        "'sha256':h.hexdigest()})\n"
        "cats=[x['name'] for x in sorted(doc['categories'],"
        "key=lambda x:x['id'])];"
        "print(json.dumps({'images':out,'categories':cats}))"
    )
    observed = json.loads(
        remote_output(
            f"python3 -c {shlex.quote(reader)} "
            f"{shlex.quote(validation_dir)} "
            f"{shlex.quote(annotation_path)}"
        )
    )
    if (
        not isinstance(observed, Mapping)
        or not isinstance(observed.get("images"), list)
        or len(observed["images"]) != 16
        or not isinstance(observed.get("categories"), list)
        or len(observed["categories"]) != 80
        or len(set(observed["categories"])) != 80
        or any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("file_name"), str)
            or isinstance(item.get("id"), bool)
            or not isinstance(item.get("id"), int)
            or isinstance(item.get("width"), bool)
            or not isinstance(item.get("width"), int)
            or isinstance(item.get("height"), bool)
            or not isinstance(item.get("height"), int)
            or isinstance(item.get("size_bytes"), bool)
            or not isinstance(item.get("size_bytes"), int)
            or item["size_bytes"] < 1
            or re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256")))
            is None
            for item in observed["images"]
        )
    ):
        raise CampaignExecutionError(
            "COCO2017 validation split lacks 16 immutable latency inputs"
        )
    return {
        "schema_version": 1,
        "dataset_id": contract["dataset"]["id"],
        "dataset_content_sha256": contract["dataset"]["content_sha256"],
        "source_annotation_path": annotation_path,
        "source_annotation_sha256": contract["dataset"][
            "contiguous_validation_json_sha256"
        ],
        "selection_rule": (
            "first_16_images_in_frozen_contiguous_coco_annotation_order"
        ),
        "images": [dict(item) for item in observed["images"]],
        "category_prompts": list(observed["categories"]),
        "caption": " . ".join(observed["categories"]) + " .",
        "dtype": "float32",
        "channels": 3,
        "padding_mask_channel": 3,
        "preloaded_batches": 16,
        "text_tokenization_scope": "precomputed_outside_timed_scope",
        "benchmark_seed": campaign_contract.LATENCY_PROTOCOL[
            "benchmark_seed"
        ],
        "required_hardware": copy.deepcopy(
            campaign_contract.FROZEN_HARDWARE
        ),
    }


def _latency_contract_document(
    contract: Mapping[str, Any],
    input_sha256: str,
) -> dict[str, Any]:
    protocol = contract["latency_protocol"]
    runtime_identity = {
        "sqsh_sha256": contract["sqsh"]["sha256"],
        "latency_worker_sha256": contract["launcher_integrity"][
            "mask_grounding_dino_latency_worker_sha256"
        ],
        "precision": protocol["precision"],
        "hardware": campaign_contract.FROZEN_HARDWARE,
    }
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
        "input_sha256": input_sha256,
        "runtime_sha256": canonical_sha256(runtime_identity),
        "expected_replicas": protocol["expected_replicas"],
        "measurement_role": protocol["measurement_role"],
        "synchronization": protocol["synchronization"],
        "validity_thresholds": protocol["validity_thresholds"],
    }


def _payload_command(
    contract: Mapping[str, Any],
    descriptor: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    source_root = Path(contract["runtime"]["repository"]) / "src/tao_automl"
    latency_contract = _latency_contract_document(
        contract,
        canonical_sha256(descriptor),
    )
    files = {
        "tao_automl/__init__.py": b"",
        "tao_automl/latency_stats.py": (
            source_root / "latency_stats.py"
        ).read_bytes(),
        "tao_automl/latency_benchmark.py": (
            source_root / "latency_benchmark.py"
        ).read_bytes(),
        "mask_grounding_dino_latency_worker.py": (
            HERE / "mask_grounding_dino_latency_worker.py"
        ).read_bytes(),
        "contract.json": json.dumps(
            latency_contract, sort_keys=True
        ).encode("utf-8"),
        "input_descriptor.json": json.dumps(
            descriptor, sort_keys=True
        ).encode("utf-8"),
    }
    encoded = {
        name: base64.b64encode(content).decode("ascii")
        for name, content in files.items()
    }
    installer = (
        "import base64,json,pathlib;"
        "root=pathlib.Path('/tmp/mask_grounding_dino_campaign_runtime');"
        f"files=json.loads({json.dumps(json.dumps(encoded))});"
        "[(root/name).parent.mkdir(parents=True,exist_ok=True) "
        "for name in files];"
        "[(root/name).write_bytes(base64.b64decode(data)) "
        "for name,data in files.items()]"
    )
    command = f"python -c {shlex.quote(installer)}"
    return command.replace("{", "{{").replace("}", "}}"), latency_contract


def _launch_latency(
    sdk: Any,
    contract: Mapping[str, Any],
    spec: Mapping[str, Any],
    checkpoint: str,
    fingerprint: str,
    *,
    events: Path,
    mode: str,
    candidate_id: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    from tao_automl.latency_benchmark import combine_replica_records
    from tao_sdk.script_runner import build_entrypoint

    descriptor = validation_input_descriptor(contract)
    installer, latency_contract = _payload_command(contract, descriptor)
    benchmark_spec = copy.deepcopy(dict(spec))
    benchmark_spec["dataset"]["batch_size"] = 1
    benchmark_spec["dataset"]["workers"] = 0
    benchmark_spec["evaluate"]["batch_size"] = 1
    command = " ".join(
        [
            installer,
            "&& torchrun --standalone --nproc_per_node=8",
            "/tmp/mask_grounding_dino_campaign_runtime/mask_grounding_dino_latency_worker.py",
            "--config {config_path}",
            "--checkpoint",
            shlex.quote(checkpoint),
            "--contract /tmp/mask_grounding_dino_campaign_runtime/contract.json",
            "--input-descriptor "
            "/tmp/mask_grounding_dino_campaign_runtime/input_descriptor.json",
            "--candidate-fingerprint",
            shlex.quote(fingerprint),
            "--runtime-modules-root /tmp/mask_grounding_dino_campaign_runtime",
            "--output-root "
            '"$TAO_RESULTS_ROOT/$TAO_JOB_ID/latency"',
        ]
    )
    action = yaml.safe_load(
        (
            Path(contract["runtime"]["skill_dir"])
            / "references/skill_info.yaml"
        ).read_text(encoding="utf-8")
    )["actions"]["evaluate"]
    entrypoint = build_entrypoint(
        command=command,
        specs=benchmark_spec,
        inputs=action["inputs"],
        outputs={},
        config_format="yaml",
        upload_excludes=action.get("upload_excludes", []),
    )
    runtime = contract["runtime"]
    job = sdk.create_job(
        image=contract["sqsh"]["path"],
        command=entrypoint["command"],
        gpu_count=8,
        num_nodes=1,
        partition=runtime["partition"],
        account=runtime["account"],
        env_vars=_offline_text_environment(contract),
    )
    evidence = {
        "tao_job_id": job.id,
        "status": "submitted",
        "submitted_at_utc": utc_timestamp(),
        "spec_sha256": canonical_sha256(benchmark_spec),
        "command_sha256": text_sha256(entrypoint["command"]),
        "candidate_fingerprint": fingerprint,
        "input_descriptor": descriptor,
        "input_sha256": canonical_sha256(descriptor),
        "contract_sha256": canonical_sha256(latency_contract),
    }
    status = _wait_for_job(
        sdk,
        job.id,
        events=events,
        phase="selection_time_latency",
        mode=mode,
        candidate_id=candidate_id,
    )
    evidence.update(
        {"status": status, "terminal_at_utc": utc_timestamp()}
    )
    logs = sdk.get_job_logs(job.id, tail=5000)
    if (
        status != "Complete"
        or "TAO_AUTOML_MASK_GROUNDING_DINO_LATENCY_COMPLETE" not in logs
    ):
        raise CampaignExecutionError(
            f"latency job {job.id} ended as {status}: {logs[-3000:]}"
        )
    root = _local_lustre_path(sdk.get_job_results_dir(job.id))
    reader = (
        "import glob,json,sys;"
        "paths=sorted(glob.glob(sys.argv[1]+'/rank_*.json'));"
        "print(json.dumps([json.load(open(path)) for path in paths]))"
    )
    records = json.loads(
        remote_output(
            f"python3 -c {shlex.quote(reader)} "
            f"{shlex.quote(root + '/latency')}"
        )
    )
    if (
        len(records) != 8
        or {item.get("tao_job_id") for item in records} != {job.id}
    ):
        raise CampaignExecutionError(
            "latency evidence is not eight-replica job-isolated data"
        )
    input_hashes = set()
    for item in records:
        input_evidence = item.get("input_evidence")
        runtime_evidence = item.get("rank_runtime_evidence")
        if not isinstance(input_evidence, Mapping):
            raise CampaignExecutionError(
                "latency replica omitted immutable input evidence"
            )
        payload = dict(input_evidence)
        supplied = payload.pop("sha256", None)
        if supplied != canonical_sha256(payload):
            raise CampaignExecutionError(
                "latency input evidence integrity failed"
            )
        input_hashes.add(supplied)
        if not isinstance(runtime_evidence, Mapping):
            raise CampaignExecutionError(
                "latency replica omitted runtime evidence"
            )
        for key, expected in campaign_contract.FROZEN_HARDWARE.items():
            if runtime_evidence.get(key) != expected:
                raise CampaignExecutionError(
                    f"latency hardware contract changed: {key}"
                )
    if len(input_hashes) != 1:
        raise CampaignExecutionError(
            "latency replicas used different preprocessed inputs"
        )
    aggregate = combine_replica_records(records)
    statistics = aggregate["statistics"]
    if (
        not statistics["is_valid"]
        or statistics["raw_sample_count_total"] != 4000
    ):
        raise CampaignExecutionError(
            "selection-time latency failed its frozen quality gate"
        )
    low, high = statistics["bootstrap_median_ci_ms"]
    metrics = {
        "latency_ms": float(statistics["median_ms"]),
        "latency_p95_ms": float(statistics["p95_ms"]),
        "latency_ci95_low_ms": float(low),
        "latency_ci95_high_ms": float(high),
    }
    evidence.update(
        {
            "result_root": root,
            "aggregate": aggregate,
            "quality_gate_passed": True,
        }
    )
    return metrics, evidence


def _metric_extractor(logs: str, metric_name: str) -> float | None:
    if metric_name != "segm_val_mAP50_95":
        return None
    values = [
        float(value)
        for value in re.findall(
            r"\[segm\]\s+val_mAP@50-95[^0-9+\-]*"
            r"([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)",
            logs,
        )
    ]
    return values[-1] if values else None


def _ptm_id(
    recommendation_audit: Mapping[str, Any],
    allowed_ids: tuple[str, ...],
) -> str:
    try:
        checkpoint_id = recommendation_audit["acquisition"]["proposal"][
            "ptm"
        ]["arm_id"]
    except (KeyError, TypeError) as exc:
        raise CampaignExecutionError(
            "hierarchical recommendation omitted its signed PTM arm"
        ) from exc
    if checkpoint_id not in set(allowed_ids):
        raise CampaignExecutionError(
            f"recommendation emitted unqualified PTM arm {checkpoint_id!r}"
        )
    return str(checkpoint_id)


def _immutable_recommendation_record(
    recommendation: Any,
    mode: str,
    allowed_ids: tuple[str, ...],
) -> dict[str, Any]:
    candidate_id = f"{mode}_rec_{recommendation.id}"
    audit = copy.deepcopy(recommendation.recommendation_audit)
    try:
        validate_recommendation_audit(audit)
    except (TypeError, ValueError) as exc:
        raise CampaignExecutionError(
            f"{candidate_id} recommendation audit integrity failed"
        ) from exc
    fingerprint = canonical_spec_fingerprint(recommendation.specs)
    if (
        audit.get("candidate_id") != str(recommendation.id)
        or audit.get("candidate_fingerprint") != fingerprint
    ):
        raise CampaignExecutionError(
            f"{candidate_id} recommendation identity changed"
        )
    return {
        "candidate_id": candidate_id,
        "rec_id": str(recommendation.id),
        "status": "recommended",
        "checkpoint_id": _ptm_id(audit, allowed_ids),
        "specs": copy.deepcopy(recommendation.specs),
        "candidate_fingerprint": fingerprint,
        "recommendation_audit": audit,
        "agent_intervention_flags": {
            name: False for name in campaign_contract.PARETO_AGENT_FLAGS
        },
    }


def _preserve_or_add_recommendation(
    candidates: dict[str, Any],
    record: Mapping[str, Any],
) -> None:
    candidate_id = str(record["candidate_id"])
    existing = candidates.get(candidate_id)
    if existing is not None and any(
        existing.get(key) != value
        for key, value in record.items()
        if key != "status"
    ):
        raise CampaignExecutionError(
            f"resumed recommendation changed: {candidate_id}"
        )
    stored = candidates.setdefault(candidate_id, {})
    for key, value in record.items():
        if key == "status":
            stored.setdefault(key, copy.deepcopy(value))
        else:
            stored[key] = copy.deepcopy(value)


def _first_candidate_gate_dir(
    runtime_root: Path,
    contract_sha256: str,
) -> Path:
    """Preserve a predecessor gate while isolating a successor decision."""
    primary = runtime_root / "first_candidate_gate"
    release = primary / "release.json"
    if release.is_file():
        try:
            existing = json.loads(release.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
        if existing.get("contract_sha256") != contract_sha256:
            return runtime_root / (
                "first_candidate_gate_" + contract_sha256[:16]
            )
    return primary


def _await_first_candidate_release(
    *,
    runtime_root: Path,
    mode: str,
    evidence: Mapping[str, Any],
) -> None:
    contract_sha256 = str(evidence["contract_sha256"])
    gate_dir = _first_candidate_gate_dir(runtime_root, contract_sha256)
    atomic_json(gate_dir / f"{mode}.json", evidence)
    release = gate_dir / "release.json"
    while not release.is_file():
        time.sleep(2)
    decision = json.loads(release.read_text(encoding="utf-8"))
    if (
        decision.get("release_remaining_budget") is not True
        or decision.get("modes") != list(campaign_contract.MODES)
    ):
        raise CampaignExecutionError(
            f"first-candidate gate rejected {mode}: "
            f"{decision.get('reason', 'unspecified')}"
        )


def _require_remaining_budget_release(
    *,
    runtime_root: Path,
    contract_sha256: str,
    recommendation_id: Any,
) -> None:
    """Fail before launching any recommendation beyond the first candidate."""
    try:
        rec_id = int(recommendation_id)
    except (TypeError, ValueError) as exc:
        raise CampaignExecutionError(
            f"invalid recommendation id: {recommendation_id!r}"
        ) from exc
    if rec_id == 0:
        return
    release = _first_candidate_gate_dir(
        runtime_root, contract_sha256
    ) / "release.json"
    if not release.is_file():
        raise CampaignExecutionError(
            f"recommendation {rec_id} blocked: first-candidate release is missing"
        )
    decision = json.loads(release.read_text(encoding="utf-8"))
    if (
        decision.get("contract_sha256") != contract_sha256
        or decision.get("modes") != list(campaign_contract.MODES)
        or decision.get("release_remaining_budget") is not True
    ):
        raise CampaignExecutionError(
            f"recommendation {rec_id} blocked by first-candidate gate: "
            f"{decision.get('reason', 'unspecified')}"
        )


def _run_mode(
    contract_path: str,
    runtime_root: str,
    mode: str,
    resume: bool,
) -> None:
    contract = load_contract(contract_path)
    decision = audit_qualification(
        contract["qualification_policy"]["qualification_evidence_path"],
        expected_contract=contract,
    )
    decision.assert_runtime_ready()
    root = Path(runtime_root)
    mode_dir = root / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    events = mode_dir / "events.jsonl"
    evidence_path = mode_dir / "candidate_evidence.json"
    candidates: dict[str, Any] = {}
    evidence_migration: dict[str, Any] | None = None
    if resume and evidence_path.is_file():
        document = json.loads(evidence_path.read_text(encoding="utf-8"))
        source_contract_sha256 = document.get("contract_sha256")
        predecessor = contract["runtime"].get(
            "resume_predecessor_contract"
        )
        direct_resume = source_contract_sha256 == contract["contract_sha256"]
        successor_resume = (
            isinstance(predecessor, Mapping)
            and source_contract_sha256 == predecessor.get("contract_sha256")
        )
        if (
            not (direct_resume or successor_resume)
            or document.get("mode") != mode
            or not isinstance(document.get("candidates"), Mapping)
        ):
            raise CampaignExecutionError(
                f"{mode} resume evidence is incompatible"
            )
        candidates = copy.deepcopy(dict(document["candidates"]))
        if successor_resume:
            if (
                len(candidates) != 1
                or any(
                    record.get("status") != "recommended"
                    or "objective_values" in record
                    or any(
                        record.get("agent_intervention_flags", {}).get(name)
                        is not False
                        for name in campaign_contract.AGENT_FLAGS
                    )
                    for record in candidates.values()
                )
            ):
                raise CampaignExecutionError(
                    f"{mode} evaluator-overlay successor may resume only the "
                    "single untouched first recommendation"
                )
            evidence_migration = {
                "schema_version": 1,
                "kind": "evaluator_overlay_only_successor",
                "predecessor_contract_sha256": source_contract_sha256,
                "successor_contract_sha256": contract["contract_sha256"],
                "recommendations_changed": False,
                "training_jobs_relaunched": False,
                "objective_policy_changed": False,
                "selection_invoked": False,
                "migrated_at_utc": utc_timestamp(),
            }

    configure_slurm_runtime(contract)
    from tao_automl.runner import AutoMLRunner
    from tao_sdk.platforms.slurm import SlurmSDK

    import tao_sdk

    sdk_source = Path(tao_sdk.__file__).resolve()
    if not sdk_source.is_relative_to(
        Path(contract["runtime"]["sdk_dir"]).resolve()
    ):
        raise CampaignExecutionError(
            f"tao_sdk imported from unsealed source: {sdk_source}"
        )
    inventory = build_live_runtime_inventory(
        contract=contract,
        decision=decision,
        mode=mode,
        cache_root=root / "verified_ptm_cache",
    )
    delegate_sdk = SlurmSDK(
        poll_interval=10,
        state_file=mode_dir / "slurm_state.json",
    )
    reuse = contract["runtime"].get("first_candidate_training_reuse")
    if isinstance(reuse, Mapping):
        reuse_record = reuse["modes"][mode]
        source_sdk = SlurmSDK(
            poll_interval=10,
            state_file=reuse_record["source_state_file"],
        )
        sdk: Any = FirstCandidateTrainingReuseSDK(
            delegate_sdk,
            source_sdk,
            reuse_record,
        )
    else:
        sdk = delegate_sdk
    runner = AutoMLRunner(
        sdk=sdk,
        skill_dir=Path(contract["runtime"]["skill_dir"]),
        action="train",
        poll_interval=10,
    )
    train_action = copy.deepcopy(runner.skill_ctx.action_cfg)
    train_action["command"] = checkpoint_resume.wrap_train_command(
        train_action["command"],
        model_slug="mask_grounding_dino",
        decision_filename=(
            "mask_grounding_dino_checkpoint_resume_decision.json"
        ),
        history_directory=(
            "mask_grounding_dino_checkpoint_resume_decisions"
        ),
    )
    runner.skill_ctx.action_cfg = train_action

    def persist() -> None:
        value = {
            "schema_version": 1,
            "contract_sha256": contract["contract_sha256"],
            "mode": mode,
            "candidates": candidates,
        }
        if evidence_migration is not None:
            value["evidence_migration"] = evidence_migration
        atomic_json(evidence_path, value)

    def on_recommendation(rec: Any) -> None:
        _require_remaining_budget_release(
            runtime_root=root,
            contract_sha256=contract["contract_sha256"],
            recommendation_id=rec.id,
        )
        record = _immutable_recommendation_record(
            rec, mode, decision.checkpoint_ids
        )
        arm_reuse = getattr(
            sdk, "arm_first_candidate_training_reuse", None
        )
        if callable(arm_reuse):
            arm_reuse(rec)
        _preserve_or_add_recommendation(candidates, record)
        persist()

    def evaluate_candidate(
        rec: Any,
        train_job_id: str,
    ) -> dict[str, float]:
        candidate_id = f"{mode}_rec_{rec.id}"
        record = candidates[candidate_id]
        cached = record.get("objective_values")
        if record.get("status") == "success" and isinstance(
            cached, Mapping
        ):
            return {
                str(name): float(value)
                for name, value in cached.items()
            }
        terminal_checkpoint = _terminal_checkpoint(sdk, train_job_id)
        checkpoint_assertion = getattr(
            sdk, "assert_terminal_checkpoint", None
        )
        if callable(checkpoint_assertion):
            checkpoint_assertion(train_job_id, terminal_checkpoint)
        checkpoint = terminal_checkpoint["path"]
        specification = evaluation_spec(
            contract,
            rec.specs,
            checkpoint,
        )
        record.update(
            {
                "status": "evaluating",
                "train_job_id": train_job_id,
                "terminal_checkpoint": terminal_checkpoint,
            }
        )
        reuse_evidence = getattr(sdk, "training_reuse_evidence", None)
        if callable(reuse_evidence):
            training_reuse = reuse_evidence(train_job_id)
            if training_reuse is not None:
                record["training_reuse"] = training_reuse
        persist()
        validation_mask_ap, accuracy_job = _launch_evaluation(
            sdk,
            contract,
            specification,
            events=events,
            mode=mode,
            candidate_id=candidate_id,
        )
        # Persist the completed standalone evaluation before latency starts.
        # A latency failure must not erase already validated accuracy evidence.
        record.update(
            {
                "standalone_validation": accuracy_job,
                "partial_objective_values": {
                    "segm_val_mAP50_95": validation_mask_ap,
                },
            }
        )
        persist()
        latency, latency_job = _launch_latency(
            sdk,
            contract,
            specification,
            checkpoint,
            record["candidate_fingerprint"],
            events=events,
            mode=mode,
            candidate_id=candidate_id,
        )
        objectives = {"segm_val_mAP50_95": validation_mask_ap, **latency}
        record.update(
            {
                "status": "success",
                "objective_values": objectives,
                "standalone_validation": accuracy_job,
                "selection_time_latency": latency_job,
                "measurement_role": "selection_time",
                "selection_isolation_flags": {
                    name: False
                    for name in campaign_contract.SELECTION_FLAGS
                },
            }
        )
        persist()
        return objectives

    first_result_seen = (
        root / "first_candidate_gate" / f"{mode}.json"
    ).is_file()

    def on_result(rec: Any, metric: Any, status: str) -> None:
        nonlocal first_result_seen
        candidate_id = f"{mode}_rec_{rec.id}"
        record = candidates.setdefault(candidate_id, {})
        record["automl_status"] = status
        record["reported_metric"] = metric
        if str(status).lower() not in SUCCESS_RECOMMENDATION_STATUSES:
            record["status"] = "terminal_failure"
            record["failure_reason"] = getattr(rec, "failure_reason", None)
        persist()
        if first_result_seen:
            return
        first_result_seen = True
        objectives = record.get("objective_values")
        passed = (
            str(status).lower() in SUCCESS_RECOMMENDATION_STATUSES
            and isinstance(objectives, Mapping)
            and "segm_val_mAP50_95" in objectives
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in objectives.values()
            )
            and float(objectives["segm_val_mAP50_95"])
            >= campaign_contract.FROZEN_VALIDATION_SANITY_MIN_MASK_AP
            and record.get("standalone_validation", {}).get("status")
            == "Complete"
            and record.get("selection_time_latency", {}).get(
                "quality_gate_passed"
            )
            is True
        )
        _await_first_candidate_release(
            runtime_root=root,
            mode=mode,
            evidence={
                "schema_version": 1,
                "contract_sha256": contract["contract_sha256"],
                "mode": mode,
                "candidate_id": candidate_id,
                "passed": passed,
                "objective_values": copy.deepcopy(objectives),
                "standalone_validation_status": record.get(
                    "standalone_validation", {}
                ).get("status"),
                "latency_quality_gate_passed": record.get(
                    "selection_time_latency", {}
                ).get("quality_gate_passed"),
                "recommendation_audit_sha256": record.get(
                    "recommendation_audit", {}
                ).get("audit_sha256"),
                "agent_intervention_flags": {
                    name: False
                    for name in campaign_contract.PARETO_AGENT_FLAGS
                },
            },
        )

    result = runner.run(
        train_dataset_uri="",
        eval_dataset_uri="",
        workspace_id=f"{contract['campaign_id']}-{mode}",
        image=contract["sqsh"]["path"],
        automl_settings=campaign_contract.mode_settings(
            str(contract["campaign_id"]), mode
        ),
        automl_hyperparameters=list(
            campaign_contract.SEARCH_PARAMETERS
        ),
        custom_param_ranges=campaign_contract.custom_ranges(),
        workspace_path=str(mode_dir / "workspace"),
        spec_overrides=campaign_contract.profile_overrides(
            contract["dataset"]["prepared_root"]
        ),
        metric_extractor=_metric_extractor,
        eval_fn=evaluate_candidate,
        on_recommendation=on_recommendation,
        on_result=on_result,
        resume=resume,
        ptm_aware_runtime=True,
        resolved_ptm_inventory=inventory,
        gpu_count=8,
        num_nodes=1,
        partition=contract["runtime"]["partition"],
        account=contract["runtime"]["account"],
    )
    atomic_json(
        mode_dir / "result.json",
        {
            "schema_version": 1,
            "contract_sha256": contract["contract_sha256"],
            "mode": mode,
            "status": "success",
            "result": result,
        },
    )


def _release_first_candidate_gate(
    runtime_root: Path,
    processes: Mapping[str, mp.Process],
    contract_sha256: str,
) -> dict[str, Any] | None:
    gate_dir = _first_candidate_gate_dir(runtime_root, contract_sha256)
    release = gate_dir / "release.json"
    if release.is_file():
        return json.loads(release.read_text(encoding="utf-8"))
    cells = {}
    for mode in campaign_contract.MODES:
        path = gate_dir / f"{mode}.json"
        if path.is_file():
            cells[mode] = json.loads(path.read_text(encoding="utf-8"))
    dead_without_cell = [
        mode
        for mode, process in processes.items()
        if not process.is_alive() and mode not in cells
    ]
    if dead_without_cell:
        value = {
            "schema_version": 1,
            "contract_sha256": contract_sha256,
            "release_remaining_budget": False,
            "modes": list(campaign_contract.MODES),
            "reason": (
                "mode controller terminated before first-candidate evidence: "
                + ", ".join(sorted(dead_without_cell))
            ),
            "generated_automatically": True,
        }
        atomic_json(release, value)
        return value
    if set(cells) != set(campaign_contract.MODES):
        return None
    valid = all(
        cell.get("passed") is True
        and cell.get("contract_sha256") == contract_sha256
        and cell.get("mode") == mode
        for mode, cell in cells.items()
    )
    value = {
        "schema_version": 1,
        "contract_sha256": contract_sha256,
        "release_remaining_budget": valid,
        "modes": list(campaign_contract.MODES),
        "reason": (
            "all three real first candidates passed full train, standalone "
            "validation, stabilized latency, audit, and provenance gates"
            if valid
            else "one or more real first candidates failed a frozen gate"
        ),
        "first_candidates": cells,
        "generated_automatically": True,
        "generated_at_utc": utc_timestamp(),
    }
    atomic_json(release, value)
    return value


def launch_all_modes(
    contract_path: Path,
    runtime_root: Path,
    *,
    resume: bool,
) -> dict[str, int | None]:
    contract = load_contract(contract_path)
    context = mp.get_context("spawn")
    processes = {
        mode: context.Process(
            target=_run_mode,
            args=(
                str(contract_path),
                str(runtime_root),
                mode,
                resume,
            ),
            name=f"mask_grounding_dino-automl-{mode}",
        )
        for mode in campaign_contract.MODES
    }
    for process in processes.values():
        process.start()

    def forward(signum: int, _frame: object) -> None:
        for process in processes.values():
            if process.is_alive() and process.pid:
                os.kill(process.pid, signum)

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    remaining = dict(processes)
    exit_codes: dict[str, int | None] = {}
    while remaining:
        _release_first_candidate_gate(
            runtime_root,
            processes,
            contract["contract_sha256"],
        )
        for mode, process in list(remaining.items()):
            process.join(timeout=1)
            if not process.is_alive():
                exit_codes[mode] = process.exitcode
                remaining.pop(mode)
        if remaining:
            time.sleep(1)
    atomic_json(runtime_root / "mode_process_status.json", exit_codes)
    return exit_codes


def verify_live_runtime_preflight(
    *,
    contract: Mapping[str, Any],
    decision: QualificationDecision,
    cache_root: str | Path,
) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode in campaign_contract.MODES:
        inventory = build_live_runtime_inventory(
            contract=contract,
            decision=decision,
            mode=mode,
            cache_root=cache_root,
        )
        inventory.validate()
        if (
            inventory.mode != mode
            or inventory.checkpoint_ids != decision.checkpoint_ids
        ):
            raise CampaignExecutionError(
                f"{mode} live runtime inventory changed"
            )
        modes[mode] = {
            "inventory_sha256": inventory.inventory_sha256,
            "preflight_report_sha256": (
                inventory.report.report_sha256
            ),
            "checkpoint_ids": list(inventory.checkpoint_ids),
        }
    value = {
        "schema_version": 1,
        "contract_sha256": contract["contract_sha256"],
        "qualification_evidence_sha256": decision.evidence_sha256,
        "runtime_eligibility_sha256": decision.runtime_eligibility.get(
            "eligibility_sha256"
        ),
        "runtime_registry_sha256": (
            decision.runtime_registry.document_sha256
        ),
        "status": "success",
        "model_jobs_launched": False,
        "cpu_or_smoke_model_jobs_launched": False,
        "modes": modes,
    }
    value["record_sha256"] = canonical_sha256(value)
    return value


def launch_plan(
    contract: Mapping[str, Any],
    *,
    ready: bool,
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": contract["campaign_id"],
        "contract_sha256": contract["contract_sha256"],
        "model": "mask_grounding_dino",
        "dataset": contract["dataset"]["prepared_root"],
        "launch_authorized": ready,
        "blockers": copy.deepcopy(blockers),
        "automatic_trigger": True,
        "cpu_or_smoke_model_jobs": 0,
        "ptm_qualification": (
            "direct_full_dataset_one_node_eight_gpu_only"
        ),
        "three_independent_modes": list(campaign_contract.MODES),
        "first_candidate_gate": {
            "automatic_release": True,
            "required_modes": list(campaign_contract.MODES),
            "remaining_candidates_per_mode": (
                campaign_contract.FROZEN_CANDIDATE_BUDGET - 1
            ),
        },
        "resources_per_child": {
            "nodes": 1,
            "gpus": 8,
            "container": contract["sqsh"]["path"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT
    )
    parser.add_argument("--env-file", type=Path, default=ENV_PATH)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--automatic-trigger",
        action="store_true",
        help="Wait until immutable data/PTM/registry gates pass.",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    contract_path = args.contract.resolve()
    contract = load_contract(contract_path)
    runtime_root = args.runtime_root.resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    loaded_names = load_env_file(args.env_file)
    configure_slurm_runtime(contract)
    if args.automatic_trigger:
        decision = wait_for_launch_authorization(
            contract,
            runtime_root=runtime_root,
            poll_seconds=args.poll_seconds,
        )
        ready, blockers = True, []
    else:
        ready, blockers, decision = launch_readiness(contract)
    plan = launch_plan(contract, ready=ready, blockers=blockers)
    atomic_json(runtime_root / "launch_plan.json", plan)
    if not args.launch:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if not ready or decision is None:
        raise CampaignExecutionError(
            "Mask Grounding DINO launch is blocked: "
            + ", ".join(item["code"] for item in blockers)
        )
    live = verify_live_runtime_preflight(
        contract=contract,
        decision=decision,
        cache_root=runtime_root / "verified_ptm_cache",
    )
    atomic_json(runtime_root / "live_runtime_preflight.json", live)
    atomic_json(
        runtime_root / "submission_provenance.json",
        {
            "schema_version": 1,
            "contract_sha256": contract["contract_sha256"],
            "loaded_secret_keys": list(loaded_names),
            "secret_values_recorded": False,
            "qualification_evidence_sha256": decision.evidence_sha256,
            "runtime_eligibility_sha256": decision.runtime_eligibility.get(
                "eligibility_sha256"
            ),
            "runtime_registry_sha256": (
                decision.runtime_registry.document_sha256
            ),
            "live_runtime_preflight_sha256": live["record_sha256"],
            "sqsh_path": contract["sqsh"]["path"],
            "sqsh_sha256": contract["sqsh"]["sha256"],
            "nodes_per_child": 1,
            "gpus_per_child": 8,
            "cpu_runs": 0,
            "smoke_runs": 0,
            "launched_at_utc": utc_timestamp(),
        },
    )
    exit_codes = launch_all_modes(
        contract_path,
        runtime_root,
        resume=args.resume,
    )
    if set(exit_codes) != set(campaign_contract.MODES) or any(
        code != 0 for code in exit_codes.values()
    ):
        raise CampaignExecutionError(
            f"one or more Mask Grounding DINO mode controllers failed: {exit_codes}"
        )
    print(json.dumps(exit_codes, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
