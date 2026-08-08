#!/usr/bin/env python3

"""Seal the live DINO-to-Deformable-DETR automatic handoff descriptor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .automatic_successor import (
    MODES,
    _process_identity,
    canonical_sha256,
    routing_identity_from_environment_file,
    sha256_file,
    validate_successor_descriptor,
)
from .deformable_detr_campaign.manifest_generator import (
    load_manifest as load_successor_manifest,
)
from .dino_campaign.manifest_generator import (
    load_manifest as load_predecessor_manifest,
)


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
DINO_MANIFEST = HERE / "dino_campaign" / "campaign.v1.json"
DINO_VALIDATOR = HERE / "dino_campaign" / "manifest_generator.py"
SUCCESSOR_ROOT = HERE / "deformable_detr_campaign"
SUCCESSOR_MANIFEST = SUCCESSOR_ROOT / "campaign.v1.json"
SUCCESSOR_GENERATOR = SUCCESSOR_ROOT / "manifest_generator.py"
SUCCESSOR_LAUNCHER = SUCCESSOR_ROOT / "run_campaign.py"
SUCCESSOR_INPUTS = SUCCESSOR_ROOT / "campaign.inputs.v1.json"
DEFAULT_ENV_FILE = Path("/localhome/local-rarunachalam/.tao/config.env")
DEFAULT_PYTHON = Path(
    "/localhome/local-rarunachalam/.tao/venvs/"
    "dino-multiobjective-py314/bin/python"
)


class DescriptorSealError(RuntimeError):
    """The live automatic handoff cannot be sealed safely."""


def _required_file(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise DescriptorSealError(f"required file is unavailable: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
    }


def build_descriptor(
    *,
    controller_pid: int,
    predecessor_runtime_root: Path,
    successor_runtime_root: Path,
    executable: Path,
    environment_file: Path,
) -> dict[str, Any]:
    predecessor_runtime_root = predecessor_runtime_root.resolve()
    successor_runtime_root = successor_runtime_root.resolve()
    executable = executable.resolve()
    environment_file = environment_file.resolve()
    if not predecessor_runtime_root.is_dir():
        raise DescriptorSealError(
            "predecessor runtime root is unavailable"
        )
    if successor_runtime_root.exists():
        raise DescriptorSealError(
            "fresh successor runtime root already exists"
        )
    if environment_file.is_symlink() or not environment_file.is_file():
        raise DescriptorSealError(
            "explicit environment file is unavailable or is a symlink"
        )
    try:
        routing = routing_identity_from_environment_file(environment_file)
    except Exception as exc:
        raise DescriptorSealError(
            f"successor routing identity cannot be sealed: {exc}"
        ) from exc
    controller = _process_identity(controller_pid)
    if controller is None:
        raise DescriptorSealError("DINO controller process is not alive")
    raw_cmdline = (
        Path("/proc") / str(controller_pid) / "cmdline"
    ).read_bytes()
    if (
        b"experiments.cross_model_automl_20260729.dino_campaign.run_campaign"
        not in raw_cmdline
        or str(DINO_MANIFEST.resolve()).encode() not in raw_cmdline
        or str(predecessor_runtime_root).encode() not in raw_cmdline
        or b"--launch" not in raw_cmdline
        or b"--resume" not in raw_cmdline
    ):
        raise DescriptorSealError(
            "controller command is not the resumed sealed DINO campaign"
        )

    predecessor = load_predecessor_manifest(DINO_MANIFEST)
    successor = load_successor_manifest(SUCCESSOR_MANIFEST)
    completion = successor_runtime_root / "completion.json"
    command = [
        str(executable),
        str(SUCCESSOR_LAUNCHER.resolve()),
        "--manifest",
        str(SUCCESSOR_MANIFEST.resolve()),
        "--runtime-root",
        str(successor_runtime_root),
        "--completion-artifact",
        str(completion),
        "--env-file",
        str(environment_file),
        "--launch",
        "--acknowledge-direct-full-dataset",
    ]
    required_paths = [
        executable,
        SUCCESSOR_LAUNCHER,
        SUCCESSOR_GENERATOR,
        SUCCESSOR_MANIFEST,
        SUCCESSOR_INPUTS,
        REPOSITORY / "src" / "tao_automl" / "data" / "ptm_registry.v1.json",
        REPOSITORY
        / "experiments"
        / "cross_model_automl_20260729"
        / "datasets"
        / "voc2007"
        / "manifest.v1.json",
        Path(successor["dataset"]["integrity"]["path"]),
    ]
    required_paths.extend(
        Path(item["checkpoint_spec"]["path"])
        for item in successor["ptms"]
    )
    payload = {
        "schema_version": 1,
        "predecessor": {
            "campaign_id": predecessor["campaign_id"],
            "manifest_path": str(DINO_MANIFEST.resolve()),
            "manifest_file_sha256": sha256_file(DINO_MANIFEST),
            "manifest_sha256": predecessor["manifest_sha256"],
            "manifest_validator_path": str(DINO_VALIDATOR.resolve()),
            "manifest_validator_sha256": sha256_file(DINO_VALIDATOR),
            "runtime_root": str(predecessor_runtime_root),
            "required_modes": list(MODES),
            "controller_process": controller,
        },
        "successor": {
            "name": "deformable-detr-voc2007-direct-full-qualification",
            "campaign_id": successor["campaign_id"],
            "model": "deformable_detr",
            "execution_kind": successor["execution"]["kind"],
            "cpu_runs": 0,
            "smoke_runs": 0,
            "manifest_path": str(SUCCESSOR_MANIFEST.resolve()),
            "manifest_file_sha256": sha256_file(SUCCESSOR_MANIFEST),
            "manifest_generator_path": str(SUCCESSOR_GENERATOR.resolve()),
            "launcher_path": str(SUCCESSOR_LAUNCHER.resolve()),
            "runtime_root": str(successor_runtime_root),
            "working_directory": str(REPOSITORY.resolve()),
            "command": command,
            "environment": {
                "HOME": "/localhome/local-rarunachalam",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": (
                    f"{executable.parent}:/usr/local/bin:/usr/bin:/bin"
                ),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": (
                    f"{REPOSITORY.resolve() / 'src'}:"
                    f"{REPOSITORY.resolve()}"
                ),
            },
            "environment_file": str(environment_file),
            "routing": routing,
            "completion_artifact": str(completion),
            "required_files": [
                _required_file(path)
                for path in dict.fromkeys(required_paths)
            ],
        },
    }
    return {
        **payload,
        "descriptor_sha256": canonical_sha256(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-pid", type=int, required=True)
    parser.add_argument(
        "--predecessor-runtime-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--successor-runtime-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executable", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--environment-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
    )
    args = parser.parse_args()
    descriptor = build_descriptor(
        controller_pid=args.controller_pid,
        predecessor_runtime_root=args.predecessor_runtime_root,
        successor_runtime_root=args.successor_runtime_root,
        executable=args.executable,
        environment_file=args.environment_file,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        descriptor,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != encoded:
            raise DescriptorSealError(
                "refusing to replace a different sealed descriptor"
            )
    else:
        pending = output.with_suffix(output.suffix + ".tmp")
        pending.write_text(encoded, encoding="utf-8")
        pending.replace(output)
    validate_successor_descriptor(output)
    print(descriptor["descriptor_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
