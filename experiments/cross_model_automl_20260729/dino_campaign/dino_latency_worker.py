#!/usr/bin/env python3

"""Eight-replica stabilized DINO latency worker for the direct VOC campaign.

The timed callable is the established DINO deployment scope: model forward
plus GPU box postprocessing.  Dataset I/O, preprocessing, transfers,
checkpoint loading, COCO accumulation, and distributed gathering are outside
the timed interval.  Sixteen deterministic preprocessed validation batches
are loaded before timing; no synthetic input or CPU benchmark is used.
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


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return _json_value(value.item())
    raise TypeError(
        "DINO latency input metadata is not JSON serializable: "
        f"{type(value).__name__}"
    )


def _tensor_sha256(tensor: Any) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes(order="C")).hexdigest()


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
    from nvidia_tao_pytorch.cv.deformable_detr.dataloader.pl_od_data_module import (
        ODDataModule,
    )
    from nvidia_tao_pytorch.cv.dino.model.pl_dino_model import DINOPlModel
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
            "DINO campaign latency requires exactly eight visible GPU replicas"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")

    specification = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    contract_document = json.loads(args.contract.read_text(encoding="utf-8"))
    input_descriptor = json.loads(
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
        or contract.timed_scope
        != "model_forward_plus_dino_gpu_postprocess"
        or contract.synchronization
        != "accelerator_sync_before_and_after_each_sample"
    ):
        raise ValueError("DINO campaign latency contract changed")
    if _canonical_sha256(input_descriptor) != contract.input_sha256:
        raise ValueError("latency input descriptor does not match its digest")
    if (
        input_descriptor.get("dtype") != "float32"
        or input_descriptor.get("padding_mask_channel") != 3
        or input_descriptor.get("preloaded_batches") != 16
        or input_descriptor.get("benchmark_seed") != 20260727
        or not isinstance(input_descriptor.get("shape_sequence"), list)
        or len(input_descriptor["shape_sequence"]) != 16
    ):
        raise ValueError("unsupported DINO latency input descriptor")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is unavailable: {args.checkpoint}")
    properties = torch.cuda.get_device_properties(device)
    observed_hardware_contract = {
        "gpu_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": int(properties.total_memory),
    }
    if observed_hardware_contract != input_descriptor["required_hardware"]:
        raise RuntimeError(
            "DINO latency hardware contract changed: "
            f"{observed_hardware_contract}"
        )

    torch.manual_seed(input_descriptor["benchmark_seed"])
    torch.cuda.manual_seed_all(input_descriptor["benchmark_seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    config = OmegaConf.create(specification)
    config.dataset.batch_size = 1
    config.dataset.workers = 0
    config.dataset.pin_memory = False
    config.evaluate.batch_size = 1
    if "input_width" not in config.evaluate:
        config.evaluate.input_width = None
    if "input_height" not in config.evaluate:
        config.evaluate.input_height = None
    annotation = Path(str(config.dataset.test_data_sources.json_file))
    if (
        not annotation.is_file()
        or _file_sha256(annotation)
        != input_descriptor["validation_annotation_sha256"]
    ):
        raise ValueError("validation annotation identity changed")
    annotation_document = json.loads(annotation.read_text(encoding="utf-8"))
    validation_image_ids = [
        item.get("id")
        for item in annotation_document.get("images", ())[:16]
    ]
    if validation_image_ids != input_descriptor["validation_image_ids"]:
        raise ValueError("validation image order changed")

    lightning_model = DINOPlModel.load_from_checkpoint(
        str(args.checkpoint),
        map_location="cpu",
        experiment_spec=config,
    )
    lightning_model = lightning_model.to(
        device=device,
        dtype=torch.float32,
    )
    lightning_model.eval()
    model = lightning_model.model
    postprocessor = lightning_model.box_processors

    data_module = ODDataModule(config.dataset, subtask_config=config.evaluate)
    data_module.setup(stage="test")
    preloaded = []
    input_batches = []
    for batch_index, batch in enumerate(data_module.test_dataloader()):
        data, targets, image_names = batch
        data = data.to(device=device, dtype=torch.float32, non_blocking=False)
        original_sizes = torch.stack(
            [target["orig_size"] for target in targets],
            dim=0,
        ).to(device=device, non_blocking=False)
        expected_shape = input_descriptor["shape_sequence"][batch_index]
        if list(data.shape) != expected_shape:
            raise RuntimeError(
                "preprocessed DINO input shape changed: "
                f"batch={batch_index}, expected={expected_shape}, "
                f"observed={list(data.shape)}"
            )
        preloaded.append((data, original_sizes, image_names))
        input_batches.append(
            {
                "batch_index": batch_index,
                "model_input_shape": list(data.shape),
                "model_input_dtype": str(data.dtype),
                "model_input_sha256": _tensor_sha256(data),
                "padding_mask_sha256": _tensor_sha256(data[:, 3:4]),
                "original_sizes": _json_value(original_sizes),
                "image_names": _json_value(image_names),
            }
        )
        if len(preloaded) == input_descriptor["preloaded_batches"]:
            break
    if len(preloaded) != input_descriptor["preloaded_batches"]:
        raise RuntimeError(
            "DINO latency worker could not preload the frozen 16 batches"
        )
    input_evidence = {
        "schema_version": 1,
        "descriptor_sha256": contract.input_sha256,
        "batches": input_batches,
    }
    input_evidence["sha256"] = _canonical_sha256(input_evidence)
    gathered_input_hashes: list[str | None] = [None] * world_size
    dist.all_gather_object(
        gathered_input_hashes,
        input_evidence["sha256"],
    )
    if set(gathered_input_hashes) != {input_evidence["sha256"]}:
        raise RuntimeError(
            "DINO latency replicas did not preload identical input batches"
        )

    last_output: Any = None

    @torch.inference_mode()
    def step(round_index: int, iteration: int) -> None:
        nonlocal last_output
        linear_index = (
            iteration
            if round_index < 0
            else round_index * contract.timed_iterations + iteration
        )
        data, original_sizes, image_names = preloaded[
            linear_index % len(preloaded)
        ]
        outputs = model(data)
        last_output = postprocessor(outputs, original_sizes, image_names)

    # ``run_replica_benchmark`` invokes this before and after each sample.
    # The pre-sample call aligns replicas; the post-sample call performs only
    # the CUDA completion synchronization and therefore does not include an
    # NCCL barrier in the timed interval.
    synchronization_call = 0

    def synchronize() -> None:
        nonlocal synchronization_call
        if synchronization_call % 2 == 0:
            dist.barrier()
        torch.cuda.synchronize(device)
        synchronization_call += 1

    runtime_contract = {
        **observed_hardware_contract,
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
        raise RuntimeError("nvidia-smi hardware provenance failed") from exc
    rank_runtime_evidence = {
        "hostname": socket.gethostname(),
        "local_rank": local_rank,
        "nvidia_smi": nvidia_smi,
        **runtime_contract,
    }
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
        raise RuntimeError("DINO latency benchmark did not execute")
    record.pop("record_sha256")
    tao_job_id = os.environ.get("TAO_JOB_ID")
    if not tao_job_id:
        raise RuntimeError("TAO_JOB_ID is required for job-isolated evidence")
    record["tao_job_id"] = tao_job_id
    record["input_evidence"] = input_evidence
    record["rank_runtime_evidence"] = rank_runtime_evidence
    record["record_sha256"] = _canonical_sha256(record)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_record_atomic(args.output_root / f"rank_{rank}.json", record)
    dist.barrier()
    if rank == 0:
        print("TAO_AUTOML_DINO_LATENCY_COMPLETE", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
