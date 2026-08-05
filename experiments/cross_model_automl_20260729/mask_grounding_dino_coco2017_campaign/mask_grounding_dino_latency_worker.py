#!/usr/bin/env python3

"""Eight-replica stabilized Mask Grounding DINO selection-time latency worker.

The worker runs only inside the pinned TAO SQSH on one eight-A100 allocation.
It preloads real validation images on every replica and excludes checkpoint
loading, data I/O, preprocessing, host-to-device transfer, and text
tokenization. The timed scope includes model forward and GPU mask
postprocessing.
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
    from nvidia_tao_pytorch.config.mask_grounding_dino.default_config import (
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
    from nvidia_tao_pytorch.cv.mask_grounding_dino.dataloader.od_data_module import (
        ODVGDataModule,
    )
    from nvidia_tao_pytorch.cv.mask_grounding_dino.model.pl_gdino_model import (
        MaskGDINOPlModel,
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
            "Mask Grounding DINO campaign latency requires exactly eight GPU replicas"
        )
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
    ):
        raise RuntimeError(
            "Mask Grounding DINO latency requires frozen offline text assets"
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
        != (
            "mask_grounding_dino_model_forward_plus_gpu_mask_postprocess;"
            "preprocessing_and_text_tokenization_excluded"
        )
        or contract.synchronization
        != "accelerator_sync_before_and_after_each_sample"
        or contract.warmup_iterations != 50
        or contract.timed_iterations != 100
        or contract.repeated_rounds != 5
    ):
        raise ValueError("Mask Grounding DINO campaign latency contract changed")
    if _canonical_sha256(descriptor) != contract.input_sha256:
        raise ValueError("latency descriptor digest changed")
    images = descriptor.get("images")
    prompts = descriptor.get("category_prompts")
    if (
        descriptor.get("dataset_id")
        != "coco2017_full_category_prompted_grounded_instance_segmentation"
        or descriptor.get("selection_rule")
        != "first_16_images_in_frozen_contiguous_coco_annotation_order"
        or descriptor.get("dtype") != "float32"
        or descriptor.get("channels") != 3
        or descriptor.get("padding_mask_channel") != 3
        or descriptor.get("preloaded_batches") != 16
        or descriptor.get("benchmark_seed") != 20260727
        or descriptor.get("text_tokenization_scope")
        != "precomputed_outside_timed_scope"
        or not isinstance(images, list)
        or len(images) != 16
        or not isinstance(prompts, list)
        or len(prompts) != 80
        or len(set(prompts)) != 80
        or descriptor.get("caption") != " . ".join(prompts) + " ."
    ):
        raise ValueError("unsupported Mask Grounding DINO latency input descriptor")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint unavailable: {args.checkpoint}")

    properties = torch.cuda.get_device_properties(device)
    hardware = {
        "gpu_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": int(properties.total_memory),
    }
    if hardware != descriptor["required_hardware"]:
        raise RuntimeError(f"Mask Grounding DINO latency hardware changed: {hardware}")
    torch.manual_seed(descriptor["benchmark_seed"])
    torch.cuda.manual_seed_all(descriptor["benchmark_seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    config = _materialize_config(specification, OmegaConf)
    config.dataset.batch_size = 1
    config.dataset.workers = 0
    config.dataset.pin_memory = False
    config.evaluate.batch_size = 1
    text_encoder_root = Path(str(config.model.text_encoder_type))
    if not text_encoder_root.is_dir():
        raise FileNotFoundError(
            f"offline text encoder is unavailable: {text_encoder_root}"
        )
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
    expected_images = [
        {
            "id": item["id"],
            "file_name": item["file_name"],
            "width": item["width"],
            "height": item["height"],
        }
        for item in images
    ]
    if observed_images != expected_images:
        raise ValueError(
            "the first 16 validation images changed from the frozen manifest"
        )
    categories = [
        item["name"]
        for item in sorted(
            annotation_document.get("categories", ()),
            key=lambda item: item["id"],
        )
    ]
    if categories != prompts:
        raise ValueError("category prompt contract changed")
    validation_dir = Path(
        str(config.dataset.test_data_sources.image_dir)
    )
    for item in images:
        path = validation_dir / item["file_name"]
        if (
            not path.is_file()
            or path.stat().st_size != item["size_bytes"]
            or _file_sha256(path) != item["sha256"]
        ):
            raise ValueError(
                f"validation image identity changed: {item['file_name']}"
            )

    data_module = ODVGDataModule(
        config.dataset,
        subtask_config=config.evaluate,
    )
    data_module.setup(stage="test")
    if data_module.test_dataset.cap_lists != prompts:
        raise ValueError("test dataset prompt derivation changed")
    lightning_model = MaskGDINOPlModel.load_from_checkpoint(
        str(args.checkpoint),
        map_location="cpu",
        experiment_spec=config,
        cap_lists=prompts,
        strict=False,
    )
    lightning_model = lightning_model.to(
        device=device,
        dtype=torch.float32,
    )
    lightning_model.eval()
    model = lightning_model.model
    postprocessor = lightning_model.box_processors
    (
        tokenized,
        _,
        position_ids,
        text_self_attention_masks,
    ) = lightning_model.tokenize_captions([descriptor["caption"]])
    token_inputs = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "position_ids": position_ids,
        "token_type_ids": tokenized["token_type_ids"],
        "text_self_attention_masks": text_self_attention_masks,
    }

    def move_to_device(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.to(device=device, non_blocking=False)
        if isinstance(value, dict):
            return {
                key: move_to_device(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [move_to_device(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move_to_device(item) for item in value)
        return value

    preloaded = []
    input_batches = []
    for batch_index, batch in enumerate(data_module.test_dataloader()):
        data, targets, image_names = batch
        data = data.to(
            device=device,
            dtype=torch.float32,
            non_blocking=False,
        )
        targets = move_to_device(targets)
        original_sizes = torch.stack(
            [target["orig_size"] for target in targets],
            dim=0,
        )
        target_sizes = torch.stack(
            [target["size"] for target in targets],
            dim=0,
        )
        if (
            data.ndim != 4
            or data.shape[0] != 1
            or data.shape[1] != 4
            or data.shape[2] < 32
            or data.shape[3] < 32
        ):
            raise RuntimeError(
                "preprocessed Mask Grounding DINO input must contain three "
                "RGB channels and one padding-mask channel"
            )
        expected = images[batch_index]
        preloaded.append(
            (
                data,
                targets,
                original_sizes,
                target_sizes,
                image_names,
            )
        )
        input_batches.append(
            {
                "batch_index": batch_index,
                "model_input_shape": list(data.shape),
                "model_input_dtype": str(data.dtype),
                "model_input_sha256": _tensor_sha256(data),
                "padding_mask_sha256": _tensor_sha256(data[:, 3:4]),
                "image_id": expected["id"],
                "image_name": expected["file_name"],
                "original_sizes": original_sizes.detach().cpu().tolist(),
                "target_sizes": target_sizes.detach().cpu().tolist(),
            }
        )
        if len(preloaded) == descriptor["preloaded_batches"]:
            break
    if len(preloaded) != descriptor["preloaded_batches"]:
        raise RuntimeError("could not preload the frozen 16 input batches")
    input_evidence = {
        "schema_version": 1,
        "descriptor_sha256": contract.input_sha256,
        "caption": descriptor["caption"],
        "token_inputs": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _tensor_sha256(value),
            }
            for name, value in token_inputs.items()
        },
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
        (
            data,
            targets,
            original_sizes,
            target_sizes,
            image_names,
        ) = preloaded[linear_index % len(preloaded)]
        outputs = model(
            data,
            input_ids=token_inputs["input_ids"],
            attention_mask=token_inputs["attention_mask"],
            position_ids=token_inputs["position_ids"],
            token_type_ids=token_inputs["token_type_ids"],
            text_self_attention_masks=token_inputs[
                "text_self_attention_masks"
            ],
            captions=[descriptor["caption"]],
            cat_list=None,
            is_training=False,
            one_hot_token=None,
            targets=targets,
        )
        last_output = postprocessor(
            outputs,
            original_sizes,
            image_names,
            input_sizes=target_sizes,
            label_positive_map=lightning_model.label_positive_map,
            text_threshold=config.evaluate.text_threshold,
            ioi_threshold=config.evaluate.ioi_threshold,
        )

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
        "hf_hub_offline": os.environ["HF_HUB_OFFLINE"],
        "transformers_offline": os.environ["TRANSFORMERS_OFFLINE"],
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
        raise RuntimeError("Mask Grounding DINO latency benchmark did not execute")
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
        print("TAO_AUTOML_MASK_GROUNDING_DINO_LATENCY_COMPLETE", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
