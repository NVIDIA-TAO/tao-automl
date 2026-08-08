#!/usr/bin/env python3

"""Eight-replica stabilized OneFormer selection-time latency worker.

The worker runs only inside the pinned TAO SQSH on one eight-A100 allocation.
It preloads real validation images on every replica and excludes checkpoint
loading, data I/O, preprocessing, host-to-device transfer, and mask
postprocessing from the timed model-forward scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(
        contiguous.numpy().tobytes(order="C")
    ).hexdigest()


def _materialize_config(specification: dict[str, Any], omega_conf: Any) -> Any:
    """Materialize the same structured defaults Hydra supplies to TAO."""
    from nvidia_tao_pytorch.config.oneformer.default_config import (
        ExperimentConfig,
    )

    return omega_conf.merge(
        omega_conf.structured(ExperimentConfig()),
        omega_conf.create(specification),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--input-descriptor", type=Path, required=True)
    parser.add_argument("--candidate-fingerprint", required=True)
    parser.add_argument("--runtime-modules-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.runtime_modules_root))
    import torch
    import torch.distributed as dist
    import yaml
    from omegaconf import OmegaConf
    from nvidia_tao_pytorch.cv.oneformer.dataloader.pl_data_module import (
        SemSegmDataModule,
    )
    from nvidia_tao_pytorch.cv.oneformer.model.pl_oneformer import (
        OneformerPlModule,
    )
    from tao_automl.latency_benchmark import (
        LatencyBenchmarkContract,
        ReplicaIdentity,
        run_replica_benchmark,
        write_record_atomic,
    )
    from tao_automl.latency_stats import LatencyValidityThresholds

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8 or torch.cuda.device_count() != 8:
        raise RuntimeError(
            "OneFormer campaign latency requires exactly eight GPU replicas"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    specification = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    contract_document = json.loads(
        args.contract.read_text(encoding="utf-8")
    )
    descriptor = json.loads(
        args.input_descriptor.read_text(encoding="utf-8")
    )
    contract_document.pop("schema_version", None)
    thresholds = contract_document.get("validity_thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("latency validity thresholds are missing")
    contract_document["validity_thresholds"] = LatencyValidityThresholds(
        **thresholds
    )
    contract = LatencyBenchmarkContract(**contract_document)
    if (
        contract.expected_replicas != 8
        or contract.measurement_role != "selection_time"
        or contract.precision != "fp32"
        or contract.timed_scope != "oneformer_model_forward"
        or contract.synchronization
        != "accelerator_sync_before_and_after_each_sample"
    ):
        raise ValueError("OneFormer campaign latency contract changed")
    if _canonical_sha256(descriptor) != contract.input_sha256:
        raise ValueError("latency descriptor digest changed")
    if (
        descriptor.get("dtype") != "float32"
        or descriptor.get("channels") != 3
        or descriptor.get("preloaded_batches") != 16
        or descriptor.get("benchmark_seed") != 20260727
        or not isinstance(descriptor.get("validation_files"), list)
        or len(descriptor["validation_files"]) != 16
    ):
        raise ValueError("unsupported OneFormer latency input descriptor")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint unavailable: {args.checkpoint}")

    properties = torch.cuda.get_device_properties(device)
    hardware = {
        "gpu_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": int(properties.total_memory),
    }
    if hardware != descriptor["required_hardware"]:
        raise RuntimeError(f"OneFormer latency hardware changed: {hardware}")
    torch.manual_seed(descriptor["benchmark_seed"])
    torch.cuda.manual_seed_all(descriptor["benchmark_seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    config = _materialize_config(specification, OmegaConf)
    config.dataset.test.batch_size = 1
    config.dataset.test.num_workers = 0
    config.evaluate.batch_size = 1
    validation_dir = Path(str(config.dataset.test.images))
    observed_files = sorted(
        path.name for path in validation_dir.iterdir() if path.is_file()
    )[:16]
    expected_files = [
        item["name"] for item in descriptor["validation_files"]
    ]
    if observed_files != expected_files:
        raise ValueError("validation image order changed")
    for item in descriptor["validation_files"]:
        path = validation_dir / item["name"]
        if (
            not path.is_file()
            or path.stat().st_size != item["size_bytes"]
            or _file_sha256(path) != item["sha256"]
        ):
            raise ValueError(
                f"validation image identity changed: {item['name']}"
            )

    lightning_model = OneformerPlModule.load_from_checkpoint(
        str(args.checkpoint),
        map_location="cpu",
        cfg=config,
    ).to(device=device, dtype=torch.float32)
    lightning_model.eval()
    model = lightning_model.model

    data_module = SemSegmDataModule(config)
    data_module.setup(stage="test")
    preloaded: list[tuple[Any, list[str]]] = []
    input_batches = []
    for batch_index, batch in enumerate(data_module.test_dataloader()):
        data = batch["images"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=False,
        )
        if data.ndim != 4 or list(data.shape[:2]) != [1, 3]:
            raise RuntimeError(
                "preprocessed OneFormer input shape changed: "
                f"batch={batch_index}, observed={list(data.shape)}"
            )
        tasks = list(batch["tasks"])
        image_name = Path(str(batch["file_names"][0])).name
        if image_name != expected_files[batch_index]:
            raise RuntimeError(
                "OneFormer dataloader order differs from the frozen input "
                f"descriptor: {image_name} != {expected_files[batch_index]}"
            )
        preloaded.append((data, tasks))
        input_batches.append(
            {
                "batch_index": batch_index,
                "model_input_shape": list(data.shape),
                "model_input_dtype": str(data.dtype),
                "model_input_sha256": _tensor_sha256(data),
                "task_prompt": tasks[0],
                "image_name": image_name,
            }
        )
        if len(preloaded) == descriptor["preloaded_batches"]:
            break
    if len(preloaded) != descriptor["preloaded_batches"]:
        raise RuntimeError("could not preload the frozen 16 input batches")
    # Data was intentionally preloaded before process-group initialization so
    # every replica uses the same sequential validation inputs rather than a
    # DistributedSampler shard.
    dist.init_process_group(backend="nccl")
    input_evidence = {
        "schema_version": 1,
        "descriptor_sha256": contract.input_sha256,
        "batches": input_batches,
    }
    input_evidence["sha256"] = _canonical_sha256(input_evidence)
    gathered: list[str | None] = [None] * world_size
    dist.all_gather_object(gathered, input_evidence["sha256"])
    if set(gathered) != {input_evidence["sha256"]}:
        raise RuntimeError("latency replicas loaded different input batches")

    last_output: Any = None

    @torch.inference_mode()
    def step(round_index: int, iteration: int) -> None:
        nonlocal last_output
        linear_index = (
            iteration
            if round_index < 0
            else round_index * contract.timed_iterations + iteration
        )
        images, tasks = preloaded[linear_index % len(preloaded)]
        last_output = model(images, tasks=tasks, texts=None)

    synchronization_call = 0

    def synchronize() -> None:
        nonlocal synchronization_call
        if synchronization_call % 2 == 0:
            dist.barrier()
        torch.cuda.synchronize(device)
        synchronization_call += 1

    runtime_contract = {
        **hardware,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "cudnn": int(torch.backends.cudnn.version() or 0),
        "tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }
    try:
        nvidia_smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,pstate,"
                "temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit",
                "--format=csv,noheader,nounits",
                "-i",
                str(local_rank),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except Exception as exc:
        raise RuntimeError("nvidia-smi provenance failed") from exc

    record = run_replica_benchmark(
        contract=contract,
        identity=ReplicaIdentity(
            rank=rank,
            world_size=world_size,
            device_id=f"rank:{rank}",
            hardware_sha256=_canonical_sha256(runtime_contract),
        ),
        candidate_fingerprint=args.candidate_fingerprint,
        step=step,
        synchronize=synchronize,
    )
    if last_output is None:
        raise RuntimeError("OneFormer latency benchmark did not execute")
    record.pop("record_sha256")
    tao_job_id = os.environ.get("TAO_JOB_ID")
    if not tao_job_id:
        raise RuntimeError("TAO_JOB_ID is required")
    record["tao_job_id"] = tao_job_id
    record["input_evidence"] = input_evidence
    record["rank_runtime_evidence"] = {
        "hostname": socket.gethostname(),
        "local_rank": local_rank,
        "nvidia_smi": nvidia_smi,
        **runtime_contract,
    }
    record["record_sha256"] = _canonical_sha256(record)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_record_atomic(args.output_root / f"rank_{rank}.json", record)
    dist.barrier()
    if rank == 0:
        print("TAO_AUTOML_ONEFORMER_LATENCY_COMPLETE", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
