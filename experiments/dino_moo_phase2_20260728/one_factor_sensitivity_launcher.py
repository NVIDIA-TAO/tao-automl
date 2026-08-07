#!/usr/bin/env python3

"""Fail-closed preregistered DINO one-factor sensitivity launcher.

The default action is a local dry run.  Submission builds every training job
from the pinned DINO train skill template and queues the complete deterministic
33-job plan through the TAO SDK's eight-GPU SLURM path.  Evaluation, matched
latency measurement, promotion, and final AutoML selection are intentionally
not performed by this launcher.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "one_factor_sensitivity_manifest.v1.json"
DEFAULT_RUNTIME = HERE / "runtime" / "one_factor_sensitivity"
SDK_ROOT = Path("/localhome/local-rarunachalam/tao-sdk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate or submit the preregistered DINO one-factor study."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and render the frozen plan without creating jobs (default).",
    )
    mode.add_argument(
        "--submit-training",
        action="store_true",
        help="Queue the complete frozen 33-job training plan.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--acknowledgement",
        default="",
        help="Exact user-authorized acknowledgement required for submission.",
    )
    parser.add_argument(
        "--verify-remote",
        action="store_true",
        help="Verify the remote SQSH, PTM, annotations, and image directories.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional path for the full resolved dry-run or submission report.",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=DEFAULT_RUNTIME,
        help="Dedicated state directory used only in submission mode.",
    )
    return parser.parse_args()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def git_value(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"cannot verify git ancestry in {repo}: {result.stderr.strip()}"
        )
    return result.returncode == 0


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    if manifest.get("feeds_final_selection") is not False:
        raise ValueError("feeds_final_selection must be false")
    if manifest.get("manual_selection_permitted") is not False:
        raise ValueError("manual_selection_permitted must be false")
    if manifest["scope"]["model_family"] != "DINO ResNet50":
        raise ValueError("only the preregistered DINO ResNet50 scope is allowed")
    if (
        manifest["scope"]["dataset_uri"]
        != "s3://nvcf-storage-handling/data/tao_od_synthetic_full_dino_coco/"
    ):
        raise ValueError("dataset scope drift")
    return manifest


def validate_source_contract(manifest: dict[str, Any]) -> dict[str, str]:
    source = manifest["source_contract"]
    checks: dict[str, str] = {}
    for key in ("tao_automl", "tao_sdk", "tao_skills"):
        expected = source[key]
        repo = Path(expected["path"])
        actual_commit = git_value(repo, "rev-parse", "HEAD")
        actual_branch = git_value(repo, "branch", "--show-current")
        if expected.get("commit_policy") == "required_ancestor":
            if not git_is_ancestor(repo, expected["commit"], actual_commit):
                raise RuntimeError(
                    f"{key} required base commit is not an ancestor of HEAD: "
                    f"{expected['commit']} !<= {actual_commit}"
                )
        elif actual_commit != expected["commit"]:
            raise RuntimeError(
                f"{key} commit drift: {actual_commit} != {expected['commit']}"
            )
        if actual_branch != expected["branch"]:
            raise RuntimeError(
                f"{key} branch drift: {actual_branch} != {expected['branch']}"
            )
        checks[f"{key}_commit"] = actual_commit
        checks[f"{key}_branch"] = actual_branch

    tao = source["tao_701_source"]
    tao_repo = Path(tao["path"])
    actual_tag_object = git_value(tao_repo, "rev-parse", tao["tag"])
    actual_commit = git_value(tao_repo, "rev-parse", f"{tao['tag']}^{{commit}}")
    if actual_tag_object != tao["annotated_tag_object"]:
        raise RuntimeError("TAO 7.0.1 annotated tag object drift")
    if actual_commit != tao["dereferenced_commit"]:
        raise RuntimeError("TAO 7.0.1 dereferenced commit drift")
    checks["tao_701_tag_object"] = actual_tag_object
    checks["tao_701_commit"] = actual_commit

    skill = source["dino_train_skill"]
    for path_key, digest_key in (
        ("skill_info_path", "skill_info_sha256"),
        ("train_template_path", "train_template_sha256"),
    ):
        actual = sha256_file(Path(skill[path_key]))
        if actual != skill[digest_key]:
            raise RuntimeError(
                f"pinned DINO skill artifact drift: {path_key} "
                f"{actual} != {skill[digest_key]}"
            )
        checks[digest_key] = actual

    skill_info = yaml.safe_load(Path(skill["skill_info_path"]).read_text())
    action = skill_info["actions"][skill["action"]]
    if action["command"] != skill["command"]:
        raise RuntimeError("DINO train action command drift")
    if action["config_format"] != skill["config_format"]:
        raise RuntimeError("DINO train config format drift")
    return checks


def set_dotted(target: dict[str, Any], path: str, value: Any) -> None:
    cursor: Any = target
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = copy.deepcopy(value)


def profile_id(path: str, level: Any) -> str:
    short = path.removeprefix("model.")
    return f"{short}_{str(level).replace('.', 'p')}"


def build_profiles(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    reference_model = copy.deepcopy(manifest["reference"]["model"])
    profiles: list[dict[str, Any]] = [
        {
            "profile_id": manifest["reference"]["profile_id"],
            "axis": None,
            "level": None,
            "execution": "train_evaluate_benchmark",
            "model": reference_model,
        }
    ]
    seen_model_digests = {sha256_value(reference_model)}
    for axis in manifest["design"]["axes"]:
        if axis["reference"] not in axis["levels"]:
            raise ValueError(f"axis reference missing from levels: {axis['path']}")
        if reference_model[axis["path"].removeprefix("model.")] != axis["reference"]:
            raise ValueError(f"reference model mismatch for {axis['path']}")
        for level in sorted(axis["levels"]):
            if level == axis["reference"]:
                continue
            model = copy.deepcopy(reference_model)
            set_dotted({"model": model}, axis["path"], level)
            if model["num_select"] > model["num_queries"]:
                raise ValueError("num_select must not exceed num_queries")
            digest = sha256_value(model)
            if digest in seen_model_digests:
                raise ValueError("duplicate resolved model profile")
            seen_model_digests.add(digest)
            profiles.append(
                {
                    "profile_id": profile_id(axis["path"], level),
                    "axis": axis["path"],
                    "level": level,
                    "execution": axis["execution"],
                    "model": model,
                }
            )
    expected = manifest["design"]["expected_unique_profiles"]
    if len(profiles) != expected:
        raise ValueError(f"generated {len(profiles)} profiles, expected {expected}")
    return profiles


def resolved_train_spec(
    manifest: dict[str, Any],
    template: dict[str, Any],
    profile: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    runtime = manifest["runtime_contract"]
    data = manifest["dataset_contract"]
    constants = manifest["controlled_constants"]
    spec = copy.deepcopy(template)
    spec["model"] = copy.deepcopy(profile["model"])
    spec["dataset"]["train_data_sources"][0] = {
        "image_dir": data["train_image_dir"],
        "json_file": data["train_annotation"],
    }
    spec["dataset"]["val_data_sources"][0] = {
        "image_dir": data["validation_image_dir"],
        "json_file": data["validation_annotation"],
    }
    spec["dataset"]["num_classes"] = data["num_classes"]
    spec["dataset"]["eval_class_ids"] = copy.deepcopy(data["eval_class_ids"])
    spec["dataset"]["batch_size"] = constants["train_batch_size_per_gpu"]
    spec["train"]["pretrained_model_path"] = runtime["pretrained_model_path"]
    spec["train"]["num_gpus"] = constants["train_num_gpus"]
    spec["train"]["gpu_ids"] = list(range(constants["train_num_gpus"]))
    spec["train"]["num_nodes"] = constants["train_num_nodes"]
    spec["train"]["num_epochs"] = constants["train_epochs"]
    spec["train"]["checkpoint_interval"] = constants[
        "checkpoint_interval_epochs"
    ]
    spec["train"]["validation_interval"] = constants[
        "validation_interval_epochs"
    ]
    spec["train"]["seed"] = seed
    spec["train"]["precision"] = constants["train_precision"]
    spec["train"]["distributed_strategy"] = constants[
        "train_distributed_strategy"
    ]
    spec["train"]["activation_checkpoint"] = constants[
        "activation_checkpoint"
    ]
    spec["train"]["cudnn"]["benchmark"] = constants["cudnn_benchmark"]
    spec["train"]["cudnn"]["deterministic"] = constants[
        "cudnn_deterministic"
    ]
    spec["train"]["optim"]["lr"] = constants["lr"]
    spec["train"]["optim"]["weight_decay"] = constants["weight_decay"]
    spec["wandb"]["enable"] = constants["wandb_enabled"]
    return spec


def load_skill_action(manifest: dict[str, Any]) -> dict[str, Any]:
    skill = manifest["source_contract"]["dino_train_skill"]
    skill_info = yaml.safe_load(Path(skill["skill_info_path"]).read_text())
    return skill_info["actions"][skill["action"]]


def ensure_sdk_importable() -> None:
    sdk_path = str(SDK_ROOT)
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)


def build_command(action: dict[str, Any], spec: dict[str, Any]) -> str:
    ensure_sdk_importable()
    from tao_sdk.script_runner import build_entrypoint

    entrypoint = build_entrypoint(
        command=action["command"],
        specs=spec,
        inputs=action["inputs"],
        outputs=action["outputs"],
        config_format=action["config_format"],
        upload_excludes=action["upload_excludes"],
    )
    return entrypoint["command"]


def build_plan(
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    template_path = Path(
        manifest["source_contract"]["dino_train_skill"]["train_template_path"]
    )
    template = yaml.safe_load(template_path.read_text())
    action = load_skill_action(manifest)
    profiles = build_profiles(manifest)
    frozen = manifest["digest_contract"]["frozen_profile_digests"]
    entries: list[dict[str, Any]] = []
    commands: dict[str, str] = {}
    computed_frozen: dict[str, Any] = {}

    for profile in profiles:
        pid = profile["profile_id"]
        model_digest = sha256_value(profile["model"])
        computed_frozen[pid] = {
            "resolved_model_spec_sha256": model_digest,
            "resolved_train_spec_sha256_by_seed": {},
        }
        for seed in manifest["design"]["seeds"]:
            spec = resolved_train_spec(manifest, template, profile, seed)
            spec_digest = sha256_value(spec)
            computed_frozen[pid]["resolved_train_spec_sha256_by_seed"][
                str(seed)
            ] = spec_digest
            entry_id = f"{pid}__seed_{seed}"
            training_required = (
                profile["execution"] == "train_evaluate_benchmark"
            )
            entry = {
                "entry_id": entry_id,
                "profile_id": pid,
                "seed": seed,
                "axis": profile["axis"],
                "level": profile["level"],
                "execution": profile["execution"],
                "training_required": training_required,
                "checkpoint_source_entry_id": (
                    None if training_required else f"reference__seed_{seed}"
                ),
                "resolved_model_spec": copy.deepcopy(profile["model"]),
                "resolved_model_spec_sha256": model_digest,
                "resolved_train_spec_sha256": spec_digest,
                "activation_checkpoint": spec["train"]["activation_checkpoint"],
                "feeds_final_selection": False,
            }
            if training_required:
                command = build_command(action, spec)
                command_digest = hashlib.sha256(command.encode("utf-8")).hexdigest()
                entry["train_command_sha256"] = command_digest
                entry["train_command_bytes"] = len(command.encode("utf-8"))
                commands[entry_id] = command
            entries.append(entry)

    if frozen != computed_frozen:
        raise RuntimeError(
            "resolved spec digest drift; manifest frozen_profile_digests "
            "does not match generated full specs"
        )
    train_count = sum(item["training_required"] for item in entries)
    if train_count != manifest["design"]["expected_training_jobs"]:
        raise RuntimeError("training job count drift")
    if len(entries) != manifest["design"]["expected_evaluation_profiles"]:
        raise RuntimeError("evaluation profile count drift")
    plan = {
        "study_id": manifest["study_id"],
        "status": "validated_not_submitted",
        "feeds_final_selection": False,
        "manual_selection_permitted": False,
        "profile_count": len(profiles),
        "training_job_count": train_count,
        "evaluation_profile_count": len(entries),
        "entries": entries,
    }
    plan["plan_sha256"] = sha256_value(plan)
    return plan, commands


def load_env_file(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"required secrets env file not found: {path}")
    loaded: list[str] = []
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"unsupported env line {number}: missing '='")
        key, encoded = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ValueError(f"invalid env key on line {number}")
        tokens = shlex.split(encoded, comments=True, posix=True)
        if len(tokens) > 1:
            raise ValueError(f"unsupported env value syntax on line {number}")
        value = tokens[0] if tokens else ""
        os.environ.setdefault(key, value)
        loaded.append(key)
    return sorted(loaded)


def ssh_target() -> str:
    user = os.environ.get("SLURM_USER", "").strip()
    host = os.environ.get("SLURM_HOSTNAME", "").split(",", 1)[0].strip()
    if not user or not host:
        raise RuntimeError("SLURM_USER and SLURM_HOSTNAME are required")
    return f"{user}@{host}"


def remote_probe(path: str, *, directory: bool = False) -> dict[str, Any]:
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    key_path = os.environ.get("SSH_KEY_PATH")
    if key_path:
        command.extend(["-i", key_path])
    quoted = shlex.quote(path)
    if directory:
        remote = f"test -d {quoted} && echo PRESENT || echo MISSING"
    else:
        remote = (
            f"if test -f {quoted}; then sha256sum {quoted}; "
            "else echo MISSING; fi"
        )
    command.extend([ssh_target(), remote])
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = completed.stdout.strip()
    if output == "MISSING":
        return {"path": path, "status": "missing", "sha256": None}
    if directory:
        return {"path": path, "status": "present", "sha256": None}
    return {
        "path": path,
        "status": "present",
        "sha256": output.split(None, 1)[0],
    }


def verify_remote_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    runtime = manifest["runtime_contract"]
    data = manifest["dataset_contract"]
    artifacts = [
        ("sqsh", runtime["sqsh_path"], runtime["sqsh_sha256"]),
        (
            "pretrained_model",
            runtime["pretrained_model_path"],
            runtime["pretrained_model_sha256"],
        ),
        (
            "train_annotation",
            data["train_annotation"],
            data["train_annotation_sha256"],
        ),
        (
            "validation_annotation",
            data["validation_annotation"],
            data["validation_annotation_sha256"],
        ),
    ]
    checks: list[dict[str, Any]] = []
    for kind, path, expected in artifacts:
        observed = remote_probe(path)
        verified = observed["status"] == "present" and observed["sha256"] == expected
        checks.append(
            {
                "kind": kind,
                **observed,
                "expected_sha256": expected,
                "verified": verified,
            }
        )
    for kind, path in (
        ("train_image_dir", data["train_image_dir"]),
        ("validation_image_dir", data["validation_image_dir"]),
    ):
        observed = remote_probe(path, directory=True)
        checks.append(
            {
                "kind": kind,
                **observed,
                "verified": observed["status"] == "present",
            }
        )
    if not all(item["verified"] for item in checks):
        raise RuntimeError("remote sensitivity-study artifact verification failed")
    return {"all_verified": True, "artifacts": checks}


def submit_training(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    commands: dict[str, str],
    runtime_dir: Path,
) -> list[dict[str, Any]]:
    ensure_sdk_importable()
    from tao_sdk.platforms.slurm import SlurmSDK

    runtime = manifest["runtime_contract"]
    slurm = runtime["slurm"]
    loaded_keys = load_env_file(Path(runtime["secrets_env_path"]))
    os.environ["SLURM_USE_SQSH"] = "false"
    os.environ["SLURM_PARTITION"] = slurm["partition"]
    os.environ["SLURM_ACCOUNT"] = slurm["account"]
    runtime_dir.mkdir(parents=True, exist_ok=True)
    sdk = SlurmSDK(
        poll_interval=10,
        state_file=runtime_dir / "slurm_state.json",
    )
    submissions: list[dict[str, Any]] = []
    for entry in plan["entries"]:
        if not entry["training_required"]:
            continue
        job = sdk.create_job(
            image=runtime["sqsh_path"],
            command=commands[entry["entry_id"]],
            gpu_count=slurm["gpu_count_per_node"],
            num_nodes=slurm["num_nodes"],
            partition=slurm["partition"],
            account=slurm["account"],
        )
        identity = sdk._handler.get_job_runtime_identity(job.id)
        submissions.append(
            {
                "entry_id": entry["entry_id"],
                "profile_id": entry["profile_id"],
                "seed": entry["seed"],
                "resolved_model_spec_sha256": entry[
                    "resolved_model_spec_sha256"
                ],
                "resolved_train_spec_sha256": entry[
                    "resolved_train_spec_sha256"
                ],
                "train_command_sha256": entry["train_command_sha256"],
                "tao_job_id": job.id,
                "slurm_job_id": identity.get("slurm_job_id", ""),
                "feeds_final_selection": False,
            }
        )
        atomic_json(
            runtime_dir / "training_submissions.json",
            {
                "study_id": manifest["study_id"],
                "plan_sha256": plan["plan_sha256"],
                "loaded_secret_keys": loaded_keys,
                "secret_values_recorded": False,
                "feeds_final_selection": False,
                "submissions": submissions,
            },
        )
    return submissions


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    source_checks = validate_source_contract(manifest)
    plan, commands = build_plan(manifest)
    submission = args.submit_training
    remote_checks = None
    if args.verify_remote:
        load_env_file(Path(manifest["runtime_contract"]["secrets_env_path"]))
        remote_checks = verify_remote_contract(manifest)

    if submission:
        if not args.verify_remote:
            raise RuntimeError("submission requires --verify-remote")
        expected_ack = manifest["submission_policy"]["required_acknowledgement"]
        if args.acknowledgement != expected_ack:
            raise RuntimeError(
                "submission refused: exact user-authorized concurrency "
                "acknowledgement is required"
            )
        submissions = submit_training(
            manifest,
            plan,
            commands,
            args.runtime_dir.resolve(),
        )
        plan["status"] = "training_submitted"
        plan["submissions"] = submissions
    else:
        plan["status"] = "dry_run_validated_not_submitted"

    report = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_checks": source_checks,
        "remote_checks": remote_checks,
        **plan,
    }
    if args.report:
        atomic_json(args.report.resolve(), report)
    print(
        json.dumps(
            {
                "study_id": manifest["study_id"],
                "status": report["status"],
                "plan_sha256": report["plan_sha256"],
                "profile_count": report["profile_count"],
                "training_job_count": report["training_job_count"],
                "evaluation_profile_count": report["evaluation_profile_count"],
                "feeds_final_selection": False,
                "report": str(args.report.resolve()) if args.report else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
