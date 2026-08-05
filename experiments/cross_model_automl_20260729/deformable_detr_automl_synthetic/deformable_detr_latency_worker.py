#!/usr/bin/env python3

"""Eight-replica stabilized Deformable DETR selection-time latency worker."""

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
            for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return _json_value(value.item())
    raise TypeError(
        "Deformable DETR latency metadata is not JSON serializable: "
        f"{type(value).__name__}"
    )


def _tensor_sha256(tensor: Any) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(
        contiguous.numpy().tobytes(order="C")
    ).hexdigest()


def _materialize_experiment_config(
    specification: Any,
    *,
    omega_conf: Any,
    experiment_config_type: Any,
) -> Any:
    """Merge raw worker YAML with the model's complete structured defaults."""
    if not isinstance(specification, dict):
        raise TypeError("Deformable DETR latency specification must be a mapping")
    defaults = omega_conf.structured(experiment_config_type())
    return omega_conf.merge(defaults, omega_conf.create(specification))


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
    from nvidia_tao_pytorch.config.deformable_detr.default_config import (
        ExperimentConfig,
    )
    from nvidia_tao_pytorch.cv.deformable_detr.dataloader.pl_od_data_module import (
        ODDataModule,
    )
    from nvidia_tao_pytorch.cv.deformable_detr.model.pl_dd_model import (
        DeformableDETRModel,
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
            "Deformable DETR latency requires exactly eight visible replicas"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
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
        or contract.timed_scope
        != "model_forward_plus_deformable_detr_gpu_postprocess"
        or contract.synchronization
        != "accelerator_sync_before_and_after_each_sample"
        or contract.warmup_iterations != 50
        or contract.timed_iterations != 100
        or contract.repeated_rounds != 5
    ):
        raise ValueError("Deformable DETR latency contract changed")
    if _canonical_sha256(descriptor) != contract.input_sha256:
        raise ValueError("latency input descriptor does not match its digest")
    images = descriptor.get("images")
    if (
        descriptor.get("dataset_id")
        != "tao_od_synthetic_full_dino_coco"
        or descriptor.get("selection_rule")
        != "first_16_images_in_frozen_coco_annotation_order"
        or descriptor.get("dtype") != "float32"
        or descriptor.get("padding_mask_channel") != 3
        or descriptor.get("preloaded_batches") != 16
        or descriptor.get("benchmark_seed") != 20260727
        or not isinstance(images, list)
        or len(images) != 16
    ):
        raise ValueError("unsupported Deformable DETR latency input descriptor")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is unavailable: {args.checkpoint}")

    properties = torch.cuda.get_device_properties(device)
    observed_hardware = {
        "gpu_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": int(properties.total_memory),
    }
    if observed_hardware != descriptor["required_hardware"]:
        raise RuntimeError(
            "Deformable DETR latency hardware changed: "
            f"{observed_hardware}"
        )

    torch.manual_seed(descriptor["benchmark_seed"])
    torch.cuda.manual_seed_all(descriptor["benchmark_seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # The TAO Hydra entrypoints merge raw YAML with ExperimentConfig before
    # constructing the model and datamodule.  This direct latency worker must
    # reproduce that contract so optional defaults such as export.format and
    # evaluate.input_width/input_height cannot disappear from raw OmegaConf.
    config = _materialize_experiment_config(
        specification,
        omega_conf=OmegaConf,
        experiment_config_type=ExperimentConfig,
    )
    config.dataset.batch_size = 1
    config.dataset.workers = 0
    config.dataset.pin_memory = False
    config.evaluate.batch_size = 1
    annotation = Path(str(config.dataset.test_data_sources.json_file))
    if (
        not annotation.is_file()
        or _file_sha256(annotation)
        != descriptor["source_annotation_sha256"]
    ):
        raise ValueError("validation annotation identity changed")
    annotation_document = json.loads(
        annotation.read_text(encoding="utf-8")
    )
    observed_images = [
        {
            "id": item["id"],
            "file_name": item["file_name"],
            "width": item["width"],
            "height": item["height"],
        }
        for item in annotation_document.get("images", ())[:16]
    ]
    if observed_images != images:
        raise ValueError(
            "the first 16 validation images changed from the frozen manifest"
        )

    lightning_model = DeformableDETRModel.load_from_checkpoint(
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
    batch_evidence = []
    for batch_index, batch in enumerate(data_module.test_dataloader()):
        data, targets, image_names = batch
        data = data.to(device=device, dtype=torch.float32, non_blocking=False)
        original_sizes = torch.stack(
            [target["orig_size"] for target in targets],
            dim=0,
        ).to(device=device, non_blocking=False)
        if int(data.shape[0]) != 1 or int(data.shape[1]) != 4:
            raise RuntimeError(
                "preprocessed Deformable DETR input must be NCHW with "
                "three RGB channels plus one padding-mask channel"
            )
        preloaded.append((data, original_sizes, image_names))
        batch_evidence.append(
            {
                "batch_index": batch_index,
                "validation_image_id": images[batch_index]["id"],
                "validation_file_name": images[batch_index]["file_name"],
                "model_input_shape": list(data.shape),
                "model_input_dtype": str(data.dtype),
                "model_input_sha256": _tensor_sha256(data),
                "padding_mask_sha256": _tensor_sha256(data[:, 3:4]),
                "original_sizes": _json_value(original_sizes),
                "image_names": _json_value(image_names),
            }
        )
        if len(preloaded) == descriptor["preloaded_batches"]:
            break
    if len(preloaded) != descriptor["preloaded_batches"]:
        raise RuntimeError("could not preload the frozen 16 validation batches")

    input_evidence = {
        "schema_version": 1,
        "descriptor_sha256": contract.input_sha256,
        "batches": batch_evidence,
    }
    input_evidence["sha256"] = _canonical_sha256(input_evidence)
    gathered_hashes: list[str | None] = [None] * world_size
    dist.all_gather_object(gathered_hashes, input_evidence["sha256"])
    if set(gathered_hashes) != {input_evidence["sha256"]}:
        raise RuntimeError(
            "latency replicas did not preload identical input batches"
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

    synchronization_call = 0

    def synchronize() -> None:
        nonlocal synchronization_call
        if synchronization_call % 2 == 0:
            dist.barrier()
        torch.cuda.synchronize(device)
        synchronization_call += 1

    runtime_contract = {
        **observed_hardware,
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
        raise RuntimeError("latency benchmark did not execute")
    record.pop("record_sha256")
    tao_job_id = os.environ.get("TAO_JOB_ID")
    if not tao_job_id:
        raise RuntimeError("TAO_JOB_ID is required for job-isolated evidence")
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
        print("TAO_AUTOML_DEFORMABLE_DETR_LATENCY_COMPLETE", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
