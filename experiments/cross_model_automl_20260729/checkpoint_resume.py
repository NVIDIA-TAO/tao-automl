#!/usr/bin/env python3

"""Deterministic same-job checkpoint continuation for sliced SLURM jobs.

The SLURM SDK re-executes the same generated entrypoint after its bounded
inner timeout.  This helper runs after ``script_runner`` has regenerated the
YAML and immediately before TAO training.  It considers only exact, regular,
non-symlink ``model_epoch_*_step_*.pth`` files in the current TAO job's own
``results_dir/train`` directory and selects the maximum
``(epoch, step, filename)`` identity.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shlex
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml


CHECKPOINT_PATTERN = re.compile(
    r"^model_epoch_(?P<epoch>[0-9]+)_step_(?P<step>[0-9]+)[.]pth$"
)
TRUSTED_CHECKPOINT_ENV = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"


class CheckpointResumeError(RuntimeError):
    """The generated TAO training spec cannot be resumed safely."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_component(value: str, name: str) -> str:
    if not value or re.fullmatch(r"[A-Za-z0-9._-]+", value) is None:
        raise CheckpointResumeError(f"{name} is invalid")
    return value


def _restart_count() -> int:
    raw = os.environ.get("SLURM_RESTART_COUNT", "0")
    if not raw or not raw.isdecimal():
        raise CheckpointResumeError(
            "SLURM_RESTART_COUNT must be a non-negative decimal integer"
        )
    return int(raw, 10)


def _checkpoint_identity(path: Path) -> dict[str, Any] | None:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None or path.is_symlink() or not path.is_file():
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size < 1:
        return None
    return {
        "path": str(path),
        "filename": path.name,
        "epoch": int(match.group("epoch")),
        "step": int(match.group("step")),
        "size_bytes": size,
    }


def select_latest_checkpoint(
    train_dir: str | Path,
    *,
    entries: Iterable[str | Path] | None = None,
) -> tuple[dict[str, Any] | None, int]:
    """Select max ``(epoch, step, filename)`` from one exact job directory."""
    directory = Path(train_dir)
    if entries is None:
        try:
            candidates = list(directory.iterdir())
        except FileNotFoundError:
            candidates = []
    else:
        candidates = [Path(item) for item in entries]
    eligible = []
    for path in candidates:
        if path.parent != directory:
            continue
        identity = _checkpoint_identity(path)
        if identity is not None:
            eligible.append(identity)
    if not eligible:
        return None, 0
    return (
        max(
            eligible,
            key=lambda item: (
                item["epoch"],
                item["step"],
                item["filename"],
            ),
        ),
        len(eligible),
    )


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise CheckpointResumeError(
                f"resume decision history would be overwritten: {path}"
            )
        return
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def inject_resume_checkpoint(
    spec_path: str | Path,
    *,
    model_slug: str,
    decision_filename: str,
    history_directory: str,
) -> dict[str, Any]:
    """Inject the latest same-job checkpoint into one generated TAO YAML."""
    slug = _safe_component(model_slug, "model_slug")
    decision_name = _safe_component(decision_filename, "decision_filename")
    history_name = _safe_component(history_directory, "history_directory")
    path = Path(spec_path)
    try:
        specification = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CheckpointResumeError(
            f"generated {slug} YAML is unreadable: {path}"
        ) from exc
    if not isinstance(specification, dict):
        raise CheckpointResumeError("generated training YAML must be a mapping")
    train = specification.get("train")
    results_dir = specification.get("results_dir")
    if not isinstance(train, dict):
        raise CheckpointResumeError("generated YAML has no train mapping")
    if (
        not isinstance(results_dir, str)
        or not results_dir
        or not Path(results_dir).is_absolute()
    ):
        raise CheckpointResumeError(
            "generated YAML results_dir must be an absolute same-job path"
        )

    runtime_root = os.environ.get("TAO_RESULTS_ROOT", "").rstrip("/")
    runtime_job_id = os.environ.get("TAO_JOB_ID", "")
    if runtime_root or runtime_job_id:
        if not runtime_root or not runtime_job_id or "/" in runtime_job_id:
            raise CheckpointResumeError(
                "TAO same-job result identity is incomplete or invalid"
            )
        expected = Path(runtime_root) / runtime_job_id / "results_dir"
        if Path(results_dir) != expected:
            raise CheckpointResumeError(
                "generated YAML results_dir is not this TAO job's output"
            )

    train_dir = Path(results_dir) / "train"
    selected, eligible_count = select_latest_checkpoint(train_dir)
    restart_count = _restart_count()
    if restart_count > 0 and selected is None:
        raise CheckpointResumeError(
            f"post-requeue {slug} slice has no eligible same-job checkpoint"
        )
    train["resume_training_checkpoint_path"] = (
        selected["path"] if selected is not None else ""
    )
    temporary = path.with_name(f".{path.name}.resume-{os.getpid()}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(specification, sort_keys=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

    slurm_job_id = _safe_component(
        os.environ.get("SLURM_JOB_ID", "local"), "SLURM_JOB_ID"
    )
    history_path = (
        Path(results_dir)
        / history_name
        / f"slurm_job_{slurm_job_id}_restart_{restart_count:04d}.json"
    )
    decision = {
        "schema_version": 1,
        "policy": "same_job_exact_epoch_step_max_with_history_v1",
        "model_slug": slug,
        "checkpoint_directory": str(train_dir),
        "checkpoint_pattern": CHECKPOINT_PATTERN.pattern,
        "tao_job_id": runtime_job_id or None,
        "slurm_job_id": None if slurm_job_id == "local" else slurm_job_id,
        "slurm_restart_count": restart_count,
        "eligible_checkpoint_count": eligible_count,
        "selected_checkpoint": selected,
        "resume_enabled": selected is not None,
        "selection_key": ["epoch", "step", "filename"],
        "resume_field": "train.resume_training_checkpoint_path",
        "same_job_only": True,
        "symlinks_eligible": False,
        "post_requeue_missing_checkpoint_behavior": "fail_closed",
        "history_path": str(history_path),
        "history_overwrite_allowed": False,
    }
    decision["decision_sha256"] = _canonical_sha256(decision)
    _write_immutable_json(history_path, decision)
    decision_path = path.parent / decision_name
    decision_tmp = decision_path.with_name(
        f".{decision_path.name}.{os.getpid()}.tmp"
    )
    try:
        decision_tmp.write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(decision_tmp, decision_path)
    finally:
        try:
            decision_tmp.unlink()
        except FileNotFoundError:
            pass
    return decision


def wrap_train_command(
    command: str,
    *,
    model_slug: str,
    decision_filename: str,
    history_directory: str,
    trust_checkpoint_on_fresh_start: bool = False,
) -> str:
    """Run the resume decision immediately before the TAO train command."""
    if not isinstance(command, str) or not command.strip():
        raise CheckpointResumeError("training command must be non-empty")
    for value, name in (
        (model_slug, "model_slug"),
        (decision_filename, "decision_filename"),
        (history_directory, "history_directory"),
    ):
        _safe_component(value, name)
    payload = base64.b64encode(Path(__file__).read_bytes()).decode("ascii")
    arguments = [
        model_slug,
        decision_filename,
        history_directory,
        "{config_path}",
    ]
    injector = " ".join(
        [
            "python3 -c",
            shlex.quote(
                "import base64;exec(base64.b64decode(" + repr(payload) + "))"
            ),
            *(shlex.quote(item) for item in arguments),
        ]
    )
    variable = re.sub(r"[^A-Za-z0-9]", "_", model_slug).upper()
    fresh_action = (
        ":"
        if trust_checkpoint_on_fresh_start
        else "unset " + TRUSTED_CHECKPOINT_ENV
    )
    return " ".join(
        [
            f'{variable}_RESUME_STATE="$({injector})"',
            f'&& case "${variable}_RESUME_STATE" in',
            "resume) export " + TRUSTED_CHECKPOINT_ENV + "=1 ;;",
            f"fresh) {fresh_action} ;;",
            "*) exit 86 ;;",
            "esac &&",
            command,
        ]
    )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 4:
        raise SystemExit(
            "usage: checkpoint_resume.py MODEL DECISION HISTORY SPEC.yaml"
        )
    decision = inject_resume_checkpoint(
        arguments[3],
        model_slug=arguments[0],
        decision_filename=arguments[1],
        history_directory=arguments[2],
    )
    print("resume" if decision["resume_enabled"] else "fresh")
    return 0


if __name__ == "__main__":  # pragma: no cover - embedded in job command
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_PATTERN",
    "CheckpointResumeError",
    "TRUSTED_CHECKPOINT_ENV",
    "inject_resume_checkpoint",
    "select_latest_checkpoint",
    "wrap_train_command",
]
