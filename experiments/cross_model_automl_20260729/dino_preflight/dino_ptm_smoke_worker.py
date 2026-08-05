# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-container one-batch DINO smoke worker for backbone-only PTMs.

This file is copied into the executor's declared ``/results`` bind and invoked
inside the exact TAO image.  It intentionally does not expose a general command
runner: it accepts one SDK-written DINO YAML spec, performs one train batch,
one validation batch, and one eval-mode inference forward, then writes a small
finite evidence record under the action's results directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from omegaconf import OmegaConf

from nvidia_tao_pytorch.cv.deformable_detr.dataloader.pl_od_data_module import (
    ODDataModule,
)
from nvidia_tao_pytorch.cv.dino.model.pl_dino_model import DINOPlModel


SMOKE_SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _finite_tensor_summary(value: Any) -> tuple[int, bool]:
    count = 0
    finite = True
    if isinstance(value, torch.Tensor):
        return 1, bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        for item in value.values():
            child_count, child_finite = _finite_tensor_summary(item)
            count += child_count
            finite = finite and child_finite
    elif isinstance(value, (tuple, list)):
        for item in value:
            child_count, child_finite = _finite_tensor_summary(item)
            count += child_count
            finite = finite and child_finite
    return count, finite


def _loss(model: DINOPlModel, data: Any, targets: Any) -> torch.Tensor:
    outputs = model.model(
        data,
        targets=targets if model.model_config["use_dn"] else None,
    )
    losses = model.criterion(outputs, targets)
    selected = [
        losses[key] * model.weight_dict[key]
        for key in losses
        if key in model.weight_dict
    ]
    if not selected:
        raise RuntimeError("DINO criterion produced no weighted loss")
    return sum(selected)


def run_smoke(
    *,
    config_path: Path,
    ptm_id: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("backbone PTM smoke requires exactly one visible GPU")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("DINO config must be a YAML mapping")
    checkpoint = Path(document["model"]["pretrained_backbone_path"])
    if (
        not checkpoint.is_file()
        or _sha256_file(checkpoint) != checkpoint_sha256
    ):
        raise ValueError("backbone checkpoint does not match frozen identity")
    if document["train"].get("pretrained_model_path"):
        raise ValueError("backbone smoke must not bind a full detector checkpoint")
    if (
        document["train"].get("num_gpus") != 1
        or document["train"].get("gpu_ids") != [0]
        or document["train"].get("num_nodes") != 1
    ):
        raise ValueError("backbone smoke must remain single-GPU")

    torch.manual_seed(int(document["train"]["seed"]))
    torch.cuda.manual_seed_all(int(document["train"]["seed"]))
    device = torch.device("cuda:0")
    cfg = OmegaConf.create(document)
    data_module = ODDataModule(cfg.dataset)
    data_module.setup(stage="fit")
    train_batch = next(iter(data_module.train_dataloader()))
    validation_batch = next(iter(data_module.val_dataloader()))

    model = DINOPlModel(cfg).to(device)
    optimizer_config = model.configure_optimizers()
    optimizer = optimizer_config["optimizer"]

    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_data, train_targets, _ = _to_device(train_batch, device)
    train_loss = _loss(model, train_data, train_targets)
    if not bool(torch.isfinite(train_loss).item()):
        raise RuntimeError("DINO training loss is not finite")
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        validation_data, validation_targets, _ = _to_device(
            validation_batch, device
        )
        validation_loss = _loss(
            model, validation_data, validation_targets
        )
        inference_outputs = model.model(validation_data)
    if not bool(torch.isfinite(validation_loss).item()):
        raise RuntimeError("DINO validation loss is not finite")
    tensor_count, finite_outputs = _finite_tensor_summary(inference_outputs)
    if tensor_count < 1 or not finite_outputs:
        raise RuntimeError("DINO inference output is empty or non-finite")
    torch.cuda.synchronize(device)

    return {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "ptm_id": ptm_id,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_target": "model.pretrained_backbone_path",
        "device": "cuda:0",
        "loaded": True,
        "real_data": True,
        "train": {
            "batches": 1,
            "finite": True,
            "loss": float(train_loss.detach().cpu().item()),
        },
        "validation": {
            "batches": 1,
            "finite": True,
            "loss": float(validation_loss.detach().cpu().item()),
        },
        "inference": {
            "batches": 1,
            "finite": True,
            "output_tensor_count": tensor_count,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ptm-id", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    args = parser.parse_args()
    evidence = run_smoke(
        config_path=args.config,
        ptm_id=args.ptm_id,
        checkpoint_sha256=args.checkpoint_sha256,
    )
    document = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    destination = Path(document["results_dir"]) / "ptm_smoke_evidence.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}"
    )
    temporary.write_text(
        json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
