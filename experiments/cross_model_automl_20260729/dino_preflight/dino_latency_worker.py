# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned-image, single-GPU DINO model-forward latency worker.

The local executor copies this worker and the two production latency modules
into its declared results bind.  The worker loads the completed DINO
checkpoint in the exact TAO image, prepares one deterministic CUDA tensor
outside the timed scope, and delegates every warm-up/timed iteration and
synchronization to ``tao_automl.latency_benchmark.run_replica_benchmark``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


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


def _write_create_only(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError("immutable latency output already differs")
    finally:
        temporary.unlink(missing_ok=True)


def run_worker(
    *,
    config_path: Path,
    checkpoint_path: Path,
    contract_path: Path,
    input_descriptor_path: Path,
    candidate_fingerprint: str,
    runtime_modules_root: Path,
) -> dict[str, Any]:
    """Run the real DINO CUDA forward benchmark and return one raw record."""
    sys.path.insert(0, str(runtime_modules_root))
    import torch
    import yaml
    from omegaconf import OmegaConf
    from nvidia_tao_pytorch.cv.dino.model.pl_dino_model import DINOPlModel
    from tao_automl.latency_benchmark import (
        LatencyBenchmarkContract,
        ReplicaIdentity,
        run_replica_benchmark,
    )
    from tao_automl.latency_stats import LatencyValidityThresholds

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("DINO latency preflight requires one visible GPU")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    contract_document = json.loads(contract_path.read_text(encoding="utf-8"))
    descriptor = json.loads(input_descriptor_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("DINO latency spec must be a YAML mapping")
    if not checkpoint_path.is_file():
        raise ValueError("DINO latency checkpoint is missing")
    if not isinstance(contract_document, dict):
        raise ValueError("latency contract must be a JSON object")
    if not isinstance(descriptor, dict):
        raise ValueError("latency input descriptor must be a JSON object")
    if contract_document.pop("schema_version", None) != 1:
        raise ValueError("unsupported latency benchmark schema")
    thresholds = contract_document.get("validity_thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("latency validity thresholds are missing")
    contract_document["validity_thresholds"] = LatencyValidityThresholds(
        **thresholds
    )
    contract = LatencyBenchmarkContract(**contract_document)
    if contract.expected_replicas != 1 or contract.precision != "fp32":
        raise ValueError("DINO local latency contract must be one-replica fp32")
    if contract.timed_scope != "model_forward":
        raise ValueError("DINO local latency scope must be model_forward")
    if _canonical_sha256(descriptor) != contract.input_sha256:
        raise ValueError("latency input descriptor does not match the contract")
    shape = descriptor.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in shape
        )
        or shape[0] != contract.batch_size_per_replica
        or shape[1] != 3
        or descriptor.get("dtype") != "float32"
        or not isinstance(descriptor.get("content"), str)
    ):
        raise ValueError("unsupported DINO latency input descriptor")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    seed_material = _canonical_sha256(
        {"descriptor": descriptor, "runtime_sha256": contract.runtime_sha256}
    )
    seed = int(seed_material[:8], 16)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    model_input = torch.rand(
        tuple(shape),
        dtype=torch.float32,
        generator=generator,
        device="cpu",
    ).to(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    cfg = OmegaConf.create(document)
    lightning_model = DINOPlModel.load_from_checkpoint(
        str(checkpoint_path),
        map_location="cpu",
        experiment_spec=cfg,
        export=True,
    )
    lightning_model = lightning_model.to(device=device, dtype=torch.float32)
    lightning_model.eval()
    model = lightning_model.model
    last_output: Any = None

    @torch.inference_mode()
    def step(_round_index: int, _iteration: int) -> None:
        nonlocal last_output
        last_output = model(model_input)

    def synchronize() -> None:
        torch.cuda.synchronize(device)

    properties = torch.cuda.get_device_properties(device)
    hardware = {
        "device": "cuda:0",
        "gpu_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": int(properties.total_memory),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "cudnn": int(torch.backends.cudnn.version() or 0),
        "tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }
    record = run_replica_benchmark(
        contract=contract,
        identity=ReplicaIdentity(
            rank=0,
            world_size=1,
            device_id="cuda:0",
            hardware_sha256=_canonical_sha256(hardware),
        ),
        candidate_fingerprint=candidate_fingerprint,
        step=step,
        synchronize=synchronize,
    )
    if last_output is None:
        raise RuntimeError("DINO latency worker did not execute a model forward")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--input-descriptor", type=Path, required=True)
    parser.add_argument("--candidate-fingerprint", required=True)
    parser.add_argument("--runtime-modules-root", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.contract.read_text(encoding="utf-8"))
    output = Path(
        os.environ.get(
            "TAO_DINO_LATENCY_OUTPUT",
            args.contract.parent / "latency_replica_0.json",
        )
    )
    record = run_worker(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        contract_path=args.contract,
        input_descriptor_path=args.input_descriptor,
        candidate_fingerprint=args.candidate_fingerprint,
        runtime_modules_root=args.runtime_modules_root,
    )
    if document.get("measurement_role") != "validation_only":
        raise ValueError("DINO preflight latency must remain validation-only")
    _write_create_only(output, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
