#!/usr/bin/env python3

"""Seal the checkpoint-resumable Mask Grounding DINO v3 qualification."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

from tao_automl.ptm_registry import canonical_sha256

from . import campaign_contract, qualification_campaign, run_campaign


HERE = Path(__file__).resolve().parent
DEFAULT_PREDECESSOR_CONTRACT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask_grounding_dino_coco2017_three_mode_v2/campaign.v2.json"
)
DEFAULT_PREDECESSOR_COMPLETION = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask_grounding_dino_coco2017_ptm_qualification_v2/completion.json"
)
DEFAULT_OUTPUT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask_grounding_dino_coco2017_ptm_qualification_v3/"
    "qualification.v3.json"
)
DEFAULT_REPOSITORY = Path("/localhome/local-rarunachalam/tao-automl")
DEFAULT_SDK = Path(
    "/localhome/local-rarunachalam/.tao/worktrees/"
    "tao-sdk-bounded-self-requeue"
)
DEFAULT_SKILLS = Path(
    "/localhome/local-rarunachalam/.tao/worktrees/tao-skills-release-7.1.0"
)
DEFAULT_EVIDENCE = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/"
    "mask_grounding_dino_coco2017_ptm_qualification_v3/completion.json"
)
CAMPAIGN_ID = (
    "mask_grounding_dino-coco2017-direct-full-qualification-v3-20260801"
)


class QualificationSuccessorError(RuntimeError):
    """The v3 qualification successor cannot be sealed."""


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _validated_json(path: Path, digest_key: str) -> dict[str, Any]:
    if not path.is_file():
        raise QualificationSuccessorError(f"artifact is unavailable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = copy.deepcopy(value)
    supplied = payload.pop(digest_key, None)
    if supplied != canonical_sha256(payload):
        raise QualificationSuccessorError(f"artifact integrity failed: {path}")
    return value


def _predecessor_record(path: Path) -> dict[str, Any]:
    value = _validated_json(path, "evidence_sha256")
    workflows = value.get("workflows")
    if (
        value.get("model") != "mask_grounding_dino"
        or not isinstance(workflows, list)
        or len(workflows) != 4
        or any(
            not isinstance(item, dict)
            or item.get("status") != "failure"
            or item.get("terminal") is not True
            or item.get("failure_preserved") is not True
            for item in workflows
        )
    ):
        raise QualificationSuccessorError(
            "v2 must be a terminal preserved four-arm failure cohort"
        )
    return {
        "path": str(path.resolve()),
        "file_sha256": campaign_contract.sha256_file(path),
        "evidence_sha256": value["evidence_sha256"],
        "campaign_id": value.get("campaign_id"),
        "workflow_count": 4,
        "all_terminal_failures_preserved": True,
        "replacement_submitted": False,
    }


def build_contract(
    *,
    predecessor_contract: Path,
    predecessor_completion: Path,
    repository: Path,
    wheel: Path,
    sdk: Path,
    skills: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    predecessor = _validated_json(predecessor_contract, "contract_sha256")
    if (
        predecessor.get("campaign_id")
        != "mask_grounding_dino-coco2017-objective-aware-three-mode-v2-20260801"
        or predecessor.get("qualification_policy", {}).get("version") != 2
    ):
        raise QualificationSuccessorError("unexpected v2 predecessor contract")
    if _git(repository, "status", "--porcelain"):
        raise QualificationSuccessorError("AutoML repository must be clean")
    source_commit = _git(repository, "rev-parse", "HEAD")
    sdk_commit = _git(sdk, "rev-parse", "HEAD")
    skills_commit = _git(skills, "rev-parse", "HEAD")
    if sdk_commit != "1a981d79af40d156735f3d89b98495e7818d0891":
        raise QualificationSuccessorError("bounded-requeue SDK changed")
    if skills_commit != "2e9c1b25f3c7cb1ae444c75652e36c47eace8229":
        raise QualificationSuccessorError("TAO skills commit changed")
    if not wheel.is_file():
        raise QualificationSuccessorError("production wheel is unavailable")

    value = copy.deepcopy(predecessor)
    value["campaign_id"] = CAMPAIGN_ID
    runtime = value["runtime"]
    runtime.update(
        {
            "repository": str(repository.resolve()),
            "source_commit": source_commit,
            "source_dirty": False,
            "wheel_path": str(wheel.resolve()),
            "wheel_sha256": campaign_contract.sha256_file(wheel),
            "sdk_dir": str(sdk.resolve()),
            "sdk_commit": sdk_commit,
            "skills_repository": str(skills.resolve()),
            "skills_commit": skills_commit,
            "skill_dir": str(
                (skills / "skills/models/tao-train-mask-grounding-dino").resolve()
            ),
            "qualification_evidence_path": str(evidence_path.resolve()),
            "runtime_local_eligibility": None,
            "max_job_retries": campaign_contract.FROZEN_SLURM_RETRY_CAP,
            "qualification_predecessor": _predecessor_record(
                predecessor_completion
            ),
        }
    )
    policy = value["qualification_policy"]
    policy.update(
        {
            "version": 3,
            "qualification_campaign_id": CAMPAIGN_ID,
            "qualification_evidence_path": str(evidence_path.resolve()),
            "runtime_local_eligibility": None,
            "checkpoint_resume_policy": copy.deepcopy(
                campaign_contract.CHECKPOINT_RESUME_POLICY
            ),
            "predecessor_failure_evidence": copy.deepcopy(
                runtime["qualification_predecessor"]
            ),
            "replacement_scope": "all_four_v2_timeout_loops",
        }
    )
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
        "qualification_successor_sha256": campaign_contract.sha256_file(
            HERE / "qualification_successor.py"
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
        "ddp_strategy_audit_sha256": campaign_contract.sha256_file(
            HERE / "ddp_strategy_audit.v2.json"
        ),
    }
    value.pop("contract_sha256", None)
    value["contract_sha256"] = canonical_sha256(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predecessor-contract",
        type=Path,
        default=DEFAULT_PREDECESSOR_CONTRACT,
    )
    parser.add_argument(
        "--predecessor-completion",
        type=Path,
        default=DEFAULT_PREDECESSOR_COMPLETION,
    )
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, default=DEFAULT_SDK)
    parser.add_argument("--skills", type=Path, default=DEFAULT_SKILLS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    value = build_contract(
        predecessor_contract=args.predecessor_contract.resolve(),
        predecessor_completion=args.predecessor_completion.resolve(),
        repository=args.repository.resolve(),
        wheel=args.wheel.resolve(),
        sdk=args.sdk.resolve(),
        skills=args.skills.resolve(),
        evidence_path=args.evidence.resolve(),
    )
    if args.output.exists():
        if json.loads(args.output.read_text(encoding="utf-8")) != value:
            raise QualificationSuccessorError(
                "existing successor contract differs; refusing overwrite"
            )
    else:
        run_campaign.atomic_json(args.output.resolve(), value)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "contract_sha256": value["contract_sha256"],
                "checkpoint_resume_policy": value["qualification_policy"][
                    "checkpoint_resume_policy"
                ],
                "model_jobs_submitted": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
