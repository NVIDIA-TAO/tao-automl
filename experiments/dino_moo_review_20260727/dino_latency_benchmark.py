#!/usr/bin/env python3

"""Eight-replica DINO request-latency benchmark for the TAO 7.0.1 SQSH.

This script runs inside the reviewed TAO container under ``torchrun``. Each GPU
loads one identical model replica and the same fixed, preprocessed validation
tensors. Timed scope is model forward plus DINO GPU postprocessing. Dataset I/O,
resize/normalization, host-to-device transfer, checkpoint loading, COCO
accumulation, and distributed result gathering are outside the timed region.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import time

from omegaconf import OmegaConf
import torch
import torch.distributed as dist

from nvidia_tao_pytorch.cv.deformable_detr.dataloader.pl_od_data_module import (
    ODDataModule,
)
from nvidia_tao_pytorch.cv.dino.model.pl_dino_model import DINOPlModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--timed-iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--preloaded-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def nvidia_smi_row(local_rank: int) -> str:
    fields = ",".join(
        [
            "uuid",
            "name",
            "driver_version",
            "pstate",
            "temperature.gpu",
            "clocks.sm",
            "clocks.mem",
            "power.draw",
            "power.limit",
        ]
    )
    command = [
        "nvidia-smi",
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
        "-i",
        str(local_rank),
    ]
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover - hardware diagnostic only
        return f"unavailable:{type(exc).__name__}:{exc}"


def move_preprocessed_batch(batch, device: torch.device):
    data, targets, image_names = batch
    data = data.to(device=device, dtype=torch.float32, non_blocking=False)
    original_sizes = torch.stack(
        [target["orig_size"] for target in targets],
        dim=0,
    ).to(device=device, non_blocking=False)
    return data, original_sizes, image_names


def main() -> int:
    args = parse_args()
    if min(
        args.warmup_iterations,
        args.timed_iterations,
        args.rounds,
        args.preloaded_batches,
    ) <= 0:
        raise ValueError("benchmark iteration and preload counts must be positive")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        raise RuntimeError(f"expected exactly 8 benchmark replicas, got {world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    cfg = OmegaConf.load(args.config)
    cfg.dataset.batch_size = 1
    cfg.dataset.workers = 0
    cfg.dataset.pin_memory = False
    cfg.evaluate.batch_size = 1
    # Native ``tao dino evaluate`` merges the YAML into DINOEvalExpConfig,
    # whose optional fixed-resize fields default to None.  This standalone
    # benchmark intentionally loads the same YAML directly, so restore those
    # schema defaults when the user config omits them.
    if "input_width" not in cfg.evaluate:
        cfg.evaluate.input_width = None
    if "input_height" not in cfg.evaluate:
        cfg.evaluate.input_height = None

    lightning_model = DINOPlModel.load_from_checkpoint(
        args.checkpoint,
        map_location="cpu",
        experiment_spec=cfg,
    )
    lightning_model = lightning_model.to(device=device, dtype=torch.float32)
    lightning_model.eval()
    model = lightning_model.model
    postprocessor = lightning_model.box_processors

    data_module = ODDataModule(cfg.dataset, subtask_config=cfg.evaluate)
    data_module.setup(stage="test")
    preloaded = []
    for batch_index, batch in enumerate(data_module.test_dataloader()):
        preloaded.append(move_preprocessed_batch(batch, device))
        if batch_index + 1 >= args.preloaded_batches:
            break
    if len(preloaded) != args.preloaded_batches:
        raise RuntimeError(
            f"requested {args.preloaded_batches} fixed batches, got {len(preloaded)}"
        )

    def invoke(iteration: int) -> None:
        data, original_sizes, image_names = preloaded[iteration % len(preloaded)]
        outputs = model(data)
        postprocessor(outputs, original_sizes, image_names)

    with torch.inference_mode():
        for iteration in range(args.warmup_iterations):
            invoke(iteration)
        torch.cuda.synchronize(device)

        samples_by_round = []
        for round_index in range(args.rounds):
            dist.barrier()
            round_samples = []
            for iteration in range(args.timed_iterations):
                # Align the eight independent replicas so index-wise
                # slowest-device diagnostics are meaningful. The barrier and
                # synchronization are outside the timed region.
                dist.barrier()
                torch.cuda.synchronize(device)
                started = time.perf_counter_ns()
                invoke(round_index * args.timed_iterations + iteration)
                torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
                round_samples.append(elapsed_ms)
            samples_by_round.append(round_samples)

    properties = torch.cuda.get_device_properties(device)
    record = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "protocol": {
            "warmup_iterations": args.warmup_iterations,
            "timed_iterations": args.timed_iterations,
            "repeated_rounds": args.rounds,
            "preloaded_batches": args.preloaded_batches,
            "batch_size_per_gpu": 1,
            "precision": "fp32",
            "tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "timed_scope": "model_forward_plus_dino_gpu_postprocess",
            "excluded_scope": [
                "checkpoint_load",
                "disk_io",
                "decode_resize_normalize",
                "host_to_device_transfer",
                "coco_accumulation",
                "distributed_gather",
            ],
            "synchronization": "cuda_sync_each_sample_and_nccl_barrier",
            "seed": args.seed,
        },
        "hardware": {
            "hostname": socket.gethostname(),
            "gpu_name": properties.name,
            "compute_capability": (
                f"{properties.major}.{properties.minor}"
            ),
            "total_memory_bytes": properties.total_memory,
            "nvidia_smi": nvidia_smi_row(local_rank),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "checkpoint": args.checkpoint,
        "config_path": args.config,
        "samples_ms": samples_by_round,
    }

    output_dir = (
        Path(args.output_root)
        / os.environ.get("TAO_JOB_ID", "unknown-job")
        / "latency"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rank_{rank}.json"
    pending_path = output_path.with_suffix(".json.tmp")
    pending_path.write_text(json.dumps(record, sort_keys=True) + "\n")
    pending_path.replace(output_path)
    dist.barrier()

    if rank == 0:
        print(
            "TAO_AUTOML_LATENCY_COMPLETE "
            f"world_size={world_size} rounds={args.rounds} "
            f"iterations={args.timed_iterations}",
            flush=True,
        )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
