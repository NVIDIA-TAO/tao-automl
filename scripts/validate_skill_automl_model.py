#!/usr/bin/env python3
"""Run one TAO model through the skill-based AutoMLRunner workflow.

This is intentionally a single-model runner. Algorithm-level orchestration is
done outside this script so each algorithm run folder can be deleted and
reported independently.
"""

from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from tao_automl.runner import AutoMLRunner, _extract_metric_from_logs
from tao_sdk.platforms.docker import DockerSDK
from tao_sdk.script_runner import build_entrypoint


LOG = logging.getLogger("tao_automl_validation")
BUCKET_ROOT = "s3://nvcf-storage-handling/data"
VISUAL_CHANGENET_BACKBONE_CONTAINER_PATH = "/data/pretrained_models/C-RADIOv2_B.safetensors"
COSMOS_MODEL_CONTAINER_PATH = "/models/snapshots/cosmos-reason2-8b"
COSMOS_BLOBS_CONTAINER_PATH = "/models/blobs"


@dataclass(frozen=True)
class ModelProfile:
    train_uri: str
    eval_uri: str = ""
    inference_uri: str = ""
    calibration_uri: str = ""
    data_format: str | None = None
    num_classes: int | None = None
    data_source_dataset_name: str | None = None
    dataset_name: str | None = None
    model_type: str | None = None
    blocked: str | None = None
    captions: tuple[str, ...] = ()


MODEL_PROFILES: dict[str, ModelProfile] = {
    "action-recognition": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_action_recognition_train",
        f"{BUCKET_ROOT}/purpose_built_models_action_recognition_train",
    ),
    "bevfusion": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_bevfusion_train",
    ),
    "centerpose": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_centerpose_train",
        f"{BUCKET_ROOT}/purpose_built_models_centerpose_val",
    ),
    "classification-pyt": ModelProfile(
        "/data/image-classification-mini/train",
        "/data/image-classification-mini/val",
        num_classes=20,
    ),
    "clip": ModelProfile(f"{BUCKET_ROOT}/auto_label_train", f"{BUCKET_ROOT}/auto_label_val"),
    "cosmos-rl": ModelProfile(
        f"{BUCKET_ROOT}/cosmos_rl_its_subset",
        f"{BUCKET_ROOT}/cosmos_rl_its_eval",
        data_format="llava",
    ),
    "deformable-detr": ModelProfile(
        "/data/deformable-detr-mini/train",
        "/data/deformable-detr-mini/val",
        num_classes=6,
    ),
    "depth-net-mono": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_depth_net_train",
        f"{BUCKET_ROOT}/purpose_built_models_depth_net_val",
        data_source_dataset_name="RelativeMonoDataset",
        dataset_name="MonoDataset",
        model_type="RelativeDepthAnything",
    ),
    "depth-net-stereo": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_depth_net_train",
        f"{BUCKET_ROOT}/purpose_built_models_depth_net_val",
        data_source_dataset_name="Middlebury",
        dataset_name="StereoDataset",
        model_type="FoundationStereo",
    ),
    "dino": ModelProfile(
        f"{BUCKET_ROOT}/tao_od_synthetic_subset_train_no_convert",
        f"{BUCKET_ROOT}/tao_od_synthetic_subset_val_no_convert",
        num_classes=6,
    ),
    "grounding-dino": ModelProfile(
        "/data/grounding-dino-mini/train",
        "/data/grounding-dino-mini/val",
        num_classes=6,
        captions=("head", "helmet", "person"),
    ),
    "mae": ModelProfile(
        "/data/image-classification-mini/train",
        "/data/image-classification-mini/val",
        num_classes=20,
    ),
    "mal": ModelProfile("/data/mal-mini/train", "/data/mal-mini/val"),
    "mask-grounding-dino": ModelProfile(
        "/data/mask-grounding-dino-mini/train",
        "/data/mask-grounding-dino-mini/val",
        num_classes=6,
        captions=("person", "bicycle", "car"),
    ),
    "mask2former": ModelProfile(
        "/data/mask2former-mini/train",
        "/data/mask2former-mini/val",
        num_classes=201,
    ),
    "ml-recog": ModelProfile(
        "/data/ml-recog",
        "/data/ml-recog",
    ),
    "nvdinov2": ModelProfile(
        "/data/nvdinov2-mini",
        f"{BUCKET_ROOT}/nvdinov2_val_cats_dogs",
    ),
    "nvpanoptix3d": ModelProfile(
        "/data/nvpanoptix3d/train",
        "/data/nvpanoptix3d/val",
        "/data/nvpanoptix3d/val",
        num_classes=13,
    ),
    "ocdnet": ModelProfile(
        "/data/ocdnet/train/train",
        "/data/ocdnet/val/test",
        "/data/ocdnet/val/test/img",
    ),
    "ocrnet": ModelProfile(
        "/data/ocrnet/train/train",
        "/data/ocrnet/val/test",
        "/data/ocrnet/val/test",
    ),
    "oneformer": ModelProfile(
        "/data/oneformer/train",
        "/data/oneformer/val",
        "/data/oneformer/val/images",
        num_classes=133,
    ),
    "optical-inspection": ModelProfile(
        "/data/optical-inspection/train",
        "/data/optical-inspection/val",
        "/data/optical-inspection/val",
    ),
    "pointpillars": ModelProfile(
        "/data/pointpillars",
        "/data/pointpillars",
        "/data/pointpillars",
    ),
    "pose-classification": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_pose_classification_train/nvidia",
        f"{BUCKET_ROOT}/purpose_built_models_pose_classification_train/nvidia",
    ),
    "re-identification": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_re_identification_train",
        f"{BUCKET_ROOT}/purpose_built_models_re_identification_train",
        num_classes=100,
    ),
    "rtdetr": ModelProfile(
        "/data/rtdetr/train",
        "/data/rtdetr/val",
        "/data/rtdetr/val",
        num_classes=5,
    ),
    "segformer": ModelProfile(
        "/data/segformer",
        "/data/segformer",
        "/data/segformer",
        num_classes=2,
    ),
    "sparse4d": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_sparse4d_train",
    ),
    "vila": ModelProfile(
        f"{BUCKET_ROOT}/vila_lita_ft_youcook2_yaml",
        f"{BUCKET_ROOT}/vila_lita_ft_youcook2_val_yaml",
    ),
    "visual-changenet": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_visual_changenet_classify_train",
        f"{BUCKET_ROOT}/purpose_built_models_visual_changenet_classify_val",
    ),
}


CHECKPOINT_SUFFIXES = (
    ".pth",
    ".pth.tar",
    ".pt",
    ".ckpt",
    ".hdf5",
    ".tlt",
    ".safetensors",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _resolve_model_dir(skill_bank: Path, requested_model: str) -> Path:
    """Resolve either a canonical model-skill name or a network alias."""
    models_root = skill_bank / "skills" / "models"
    direct = models_root / requested_model
    if (direct / "references" / "skill_info.yaml").exists():
        return direct

    normalized = requested_model.replace("_", "-").lower()
    # Multiple packaged skills may intentionally share a TAO network_arch. Keep
    # the canonical validation alias stable while still allowing either skill to
    # be selected explicitly by its directory name above.
    canonical_skill_aliases = {
        "depth-net-stereo": "tao-train-foundation-stereo",
    }
    canonical_dir = models_root / canonical_skill_aliases.get(normalized, "")
    if canonical_dir.name and (
        canonical_dir / "references" / "skill_info.yaml"
    ).exists():
        return canonical_dir

    matches: list[Path] = []
    for candidate in sorted(models_root.iterdir()):
        info_path = candidate / "references" / "skill_info.yaml"
        if not candidate.is_dir() or not info_path.exists():
            continue
        info = _read_yaml(info_path)
        network_arch = str(info.get("network_arch", ""))
        aliases = {
            candidate.name.lower(),
            network_arch.lower(),
            network_arch.replace("_", "-").lower(),
        }
        if normalized in {alias.replace("_", "-") for alias in aliases}:
            matches.append(candidate)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"No packaged model skill resolves {requested_model!r}")
    raise KeyError(
        f"Model alias {requested_model!r} is ambiguous: "
        + ", ".join(path.name for path in matches)
    )


def _model_profile_key(model_dir: Path) -> str:
    info = _read_yaml(model_dir / "references" / "skill_info.yaml")
    network_arch = str(info.get("network_arch", ""))
    key = network_arch.replace("_", "-").lower()
    if key in MODEL_PROFILES:
        return key
    raise KeyError(
        f"No validation data profile for {model_dir.name} (network_arch={network_arch!r})"
    )


def _set_nested(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor: Any = target
    for part in parts[:-1]:
        name, idx = _parse_part(part)
        cursor = cursor.setdefault(name, [] if idx is not None else {})
        if idx is not None:
            while len(cursor) <= idx:
                cursor.append({})
            cursor = cursor[idx]
    name, idx = _parse_part(parts[-1])
    if idx is None:
        cursor[name] = value
        return
    cursor.setdefault(name, [])
    while len(cursor[name]) <= idx:
        cursor[name].append(None)
    cursor[name][idx] = value


def _get_nested(source: dict[str, Any], dotted_key: str) -> Any:
    cursor: Any = source
    for part in dotted_key.split("."):
        name, idx = _parse_part(part)
        if not isinstance(cursor, dict) or name not in cursor:
            return None
        cursor = cursor[name]
        if idx is not None:
            if not isinstance(cursor, list) or len(cursor) <= idx:
                return None
            cursor = cursor[idx]
    return cursor


def _parse_part(part: str) -> tuple[str, int | None]:
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*)(?:\[(\d+)])?$", part)
    if not match:
        return part, None
    return match.group(1), int(match.group(2)) if match.group(2) else None


def _flatten_keys(value: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            full = f"{prefix}.{key}" if prefix else str(key)
            keys.add(full)
            keys |= _flatten_keys(child, full)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            full = f"{prefix}[{idx}]"
            keys.add(full)
            keys |= _flatten_keys(child, full)
    return keys


def _schema_keys(skill_dir: Path, action: str = "train") -> set[str]:
    schema_path = skill_dir / "schemas" / f"{action}.schema.json"
    keys: set[str] = set()
    if not schema_path.exists():
        return keys

    def walk(schema: Any, prefix: str = "") -> None:
        if not isinstance(schema, dict):
            return
        props = schema.get("properties")
        if isinstance(props, dict):
            for key, child in props.items():
                full = f"{prefix}.{key}" if prefix else str(key)
                keys.add(full)
                walk(child, full)
        items = schema.get("items")
        if isinstance(items, dict) and prefix:
            walk(items, f"{prefix}[0]")

    schema = json.loads(schema_path.read_text())
    walk(schema)
    keys |= _flatten_keys(schema.get("default", {}))
    return keys


def _monitoring_metric(skill_text: str) -> str:
    match = re.search(
        r"\*\*(?:AutoML training metrics?|Pretraining monitoring metrics?|Training monitoring metrics?|Monitoring metric):\*\*\s*([^\n]+)",
        skill_text,
    )
    if not match:
        return "loss"
    value = match.group(1).strip()
    quoted = re.search(r"`([^`]+)`", value)
    metric = quoted.group(1) if quoted else value.replace("`", "")
    return metric.split(",", 1)[0].strip()


def _documented_automl_metric(skill_text: str) -> str | None:
    """Return the metric explicitly named by the skill's AutoML contract.

    Training-monitoring metrics and evaluation-backed AutoML objectives are
    intentionally distinct for some model skills.  Keep both in validation
    evidence instead of presenting the first training KPI as the documented
    AutoML objective.
    """
    match = re.search(
        r"\*\*AutoML metric contract:\*\*(.*?)(?=\n\s*(?:-\s+\*\*|#{2,})|\Z)",
        skill_text,
        flags=re.DOTALL,
    )
    if not match:
        return None
    contract = " ".join(line.strip() for line in match.group(1).splitlines())
    positive = re.search(
        r"(?<!not )\b(?:use|optimize)\s+`([^`]+)`",
        contract,
        flags=re.IGNORECASE,
    )
    return positive.group(1).strip() if positive else None


def _evaluation_metric(model: str, training_metric: str) -> str:
    """Return the task metric emitted by the packaged evaluate action."""
    return {
        "action-recognition": "accuracy",
        "centerpose": "test_3DIoU",
        "cosmos-rl": "BERTScore_F1",
        "deformable-detr": "test_mAP50",
        "dino": "test_mAP50",
        "depth-net-stereo": "val/epe",
        "grounding-dino": "test_mAP50",
        "mae": "ACC_all",
        "ocdnet": "train_loss",
        "ocrnet": "val_acc",
    }.get(model, training_metric)


def _checkpoint_evaluation_metric(model: str, automl_metric: str) -> str:
    """Return a standalone evaluator KPI when it differs from AutoML's train KPI."""
    return {
        "clip": "test/t2i_mAP",
        "mask-grounding-dino": "[segm] test_mAP50",
        "ml-recog": "test Precision at Rank 1",
        "nvpanoptix3d": "PRQ",
        "ocdnet": "hmean",
        "ocrnet": "test_acc",
        "oneformer": "test_mIoU",
        "optical-inspection": "test_acc",
        "pointpillars": "3d mAP",
        "pose-classification": "accuracy",
        "re-identification": "mAP",
        "rtdetr": "test_mAP50",
        "segformer": "test_miou",
        "visual-changenet": "test_acc",
    }.get(model, automl_metric)


def _direction(metric: str, model: str | None = None) -> str:
    lower = metric.lower()
    # Mono depth d1 is Delta-1 accuracy (higher is better), while stereo d1 is
    # the D1 outlier/error rate (lower is better).  The metric name alone is
    # therefore insufficient to choose the optimization direction.
    if model == "depth-net-mono" and lower in {"d1", "val/d1", "test/d1"}:
        return "maximize"
    error_metrics = ("loss", "epe", "rmse", "bp1", "bp2", "bp3", "d1")
    return "minimize" if any(name in lower for name in error_metrics) else "maximize"


def _gpu_device_ids(args: argparse.Namespace, gpu_count: int | None = None) -> list[str] | None:
    effective_count = args.num_gpus if gpu_count is None else gpu_count
    if effective_count == 0 or not args.gpu_device_id:
        return None
    ids = [item.strip() for item in str(args.gpu_device_id).split(",") if item.strip()]
    if len(ids) != effective_count:
        raise ValueError(
            f"Requested {effective_count} GPU(s), but --gpu-device-id resolved to {ids}"
        )
    return ids


def _parse_action_rows(skill_text: str, action: str) -> list[dict[str, str]]:
    rows = []
    in_table = False
    for raw in skill_text.splitlines():
        line = raw.strip()
        if line.startswith("| Action | Spec Key | Source | Files |"):
            in_table = True
            continue
        if in_table and (not line.startswith("|") or line.startswith("### ")):
            break
        if not in_table or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 5 or cells[0] != action:
            continue
        rows.append({
            "spec_key": cells[1],
            "source": cells[2],
            "files": cells[3],
            "list": cells[4].lower().startswith("y"),
        })
    return rows


def _parse_train_rows(skill_text: str) -> list[dict[str, str]]:
    return _parse_action_rows(skill_text, "train")


def _source_root(profile: ModelProfile, source: str) -> str:
    if source == "train_datasets":
        return profile.train_uri
    if source == "eval_dataset":
        return profile.eval_uri or profile.train_uri
    if source == "inference_dataset":
        return profile.inference_uri or profile.eval_uri or profile.train_uri
    if source == "calibration_dataset":
        return profile.calibration_uri or profile.train_uri
    return profile.train_uri


def _join_uri(root: str, suffix: str) -> str:
    suffix = _clean_file_spec(suffix)
    if not suffix:
        return root
    if suffix.startswith("/") or "://" in suffix:
        return suffix
    return f"{root.rstrip('/')}/{suffix.lstrip('/')}"


def _clean_file_spec(files: str) -> str:
    stripped = files.strip()
    if stripped.startswith("coco_panoptic:"):
        return stripped.split("coco_panoptic:", 1)[1].split(";", 1)[0].strip()
    if stripped.startswith("dataset root containing"):
        return ""
    if stripped.startswith("one image/video or a media folder/archive"):
        return ""
    if " extracted from " in files:
        return files.split(" extracted from ", 1)[1].strip()
    if " when using " in stripped:
        return stripped.split(" when using ", 1)[0].strip()
    if " or " in stripped:
        return stripped.split(" or ", 1)[0].strip()
    return stripped


def _files_mapping(files: str) -> dict[str, str] | None:
    if files.strip().startswith("coco_panoptic:"):
        return None
    if ":" not in files:
        return None
    mapping: dict[str, str] = {}
    for part in files.split(","):
        name, _, value = part.partition(":")
        name = name.strip()
        value = _clean_file_spec(value)
        if name and value and "{" not in value and "from convert" not in value:
            mapping[name] = value
    return mapping or None


def _add_data_source_overrides(
    overrides: dict[str, Any],
    profile: ModelProfile,
    train_rows: list[dict[str, str]],
) -> None:
    for row in train_rows:
        spec_key = row["spec_key"]
        files = row["files"].strip()
        if "{dataset_convert_job_id}" in files or "from convert" in files:
            continue
        if spec_key.endswith(".type") and files in {"ade", "coco", "coco_panoptic"}:
            overrides[spec_key] = files
            continue
        root = _source_root(profile, row["source"])
        if (
            row["source"] == "inference_dataset"
            and not profile.inference_uri
            and profile.eval_uri
            and files == "images_test.tar.gz"
        ):
            overrides[spec_key] = _join_uri(profile.eval_uri, "images_val.tar.gz")
            continue
        if "data_file:" in files and "+ dataset_name" in files:
            item = {
                "data_file": _join_uri(root, files.split("data_file:", 1)[1].split("+", 1)[0].strip()),
                "dataset_name": profile.data_source_dataset_name or "GenericDataset",
            }
            overrides[spec_key] = [item] if row["list"] else item
            continue
        if files == "prompt list":
            overrides[spec_key] = list(profile.captions or ("object",))
            continue
        mapping = _files_mapping(files)
        if mapping and "captions" in mapping:
            image_dir = mapping.get("image_dir")
            if image_dir:
                overrides[f"{spec_key}.image_dir"] = [_join_uri(root, image_dir)]
            overrides[f"{spec_key}.captions"] = list(profile.captions or ("object",))
            continue
        if mapping:
            item = {field: _join_uri(root, suffix) for field, suffix in mapping.items()}
            overrides[spec_key] = [item] if row["list"] else item
        else:
            value = _join_uri(root, _clean_file_spec(files))
            overrides[spec_key] = [value] if row["list"] else value


def _valid_set(overrides: dict[str, Any], specs: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    valid = dict(overrides)
    merged_keys = keys | _flatten_keys(specs)
    for key in list(valid):
        if key in merged_keys:
            continue
        # Object-valued overrides such as dataset.train_data_sources are valid
        # when a child key exists in the schema.
        prefix = f"{key}."
        list_prefix = f"{key}["
        if any(k.startswith(prefix) or k.startswith(list_prefix) for k in merged_keys):
            continue
        LOG.info("Dropping override not present in train schema: %s", key)
        valid.pop(key)
    return valid


def _minimal_train_overrides(
    specs: dict[str, Any],
    schema_keys: set[str],
    num_classes: int | None,
    model: str,
) -> dict[str, Any]:
    keys = schema_keys | _flatten_keys(specs)
    candidates: dict[str, Any] = {
        "train.num_epochs": 1,
        "train.epoch": 1,
        "train.max_epochs": 1,
        "train.checkpoint_interval": 1,
        "train.validation_interval": 1,
        "train.num_gpus": 1,
        "train.gpu_ids": [0],
        "train.optim.lr_step_size": 1,
        "validation.freq_in_epoch": 1,
        "train.ckpt.save_freq_in_epoch": 1,
        "train.ckpt.max_keep": 2,
        "train.ckpt.export_safetensors": True,
        "dataset.batch_size": 1,
        "dataset.workers": 0,
        "dataset.num_workers": 0,
    }
    if model == "cosmos-rl":
        candidates.update({
            "policy.model_name_or_path": COSMOS_MODEL_CONTAINER_PATH,
            "policy.parallelism.dp_shard_size": 1,
            "policy.parallelism.dp_replicate_size": 1,
            "train.train_batch_per_replica": 1,
            "train.train_policy.mini_batch": 1,
            "train.train_policy.dataset.test_size": 0,
            "validation.batch_size": 1,
            "validation.enable_dataset_cache": False,
            "custom.vision.nframes": 2,
            "logging.logger": ["console", "tao"],
        })
    if model == "clip":
        candidates.update({
            "dataset.train.batch_size": 1,
            "dataset.train.num_workers": 0,
            "dataset.val.batch_size": 1,
            "dataset.val.num_workers": 0,
        })
    if model == "deformable-detr":
        # The schema requires at least one worker; the generic zero-worker
        # baseline is intentionally overridden for this model.
        candidates["dataset.workers"] = 1
    if model == "mae":
        candidates.update({
            "dataset.batch_size": 2,
            "train.stage": "finetune",
            "model.arch": "convnextv2_atto",
        })
    if model == "ml-recog":
        candidates.update({
            "train.batch_size": 4,
            "train.val_batch_size": 4,
            "dataset.num_instance": 4,
            "dataset.workers": 0,
        })
    if model == "mal":
        candidates.update({
            "dataset.crop_size": 256,
            "dataset.num_workers_per_gpu": 1,
            "model.arch": "vit-deit-small/16",
            "train.warmup_epochs": 0,
        })
    if model == "bevfusion":
        candidates.update({
            # BEVFusion runs in its pinned TAO 5.5 container.  Keep this aligned
            # with the model skill's documented single-GPU workflow.
            "train.num_gpus": 1,
            "train.gpu_ids": [0],
            "dataset.train_dataset.batch_size": 1,
            "dataset.val_dataset.batch_size": 1,
            "dataset.test_dataset.batch_size": 1,
            "dataset.train_dataset.num_workers": 1,
            "dataset.val_dataset.num_workers": 1,
            "dataset.test_dataset.num_workers": 1,
            "wandb.enable": False,
        })
    if model == "optical-inspection":
        candidates["dataset.batch_size"] = 2
    if model == "ocdnet":
        candidates.update({
            "train.lr_scheduler.args.warmup_epoch": 0,
            "dataset.train_dataset.loader.batch_size": 1,
            "dataset.validate_dataset.loader.batch_size": 1,
            "dataset.train_dataset.loader.num_workers": 0,
            "dataset.validate_dataset.loader.num_workers": 0,
        })
    if model == "re-identification":
        candidates["dataset.batch_size"] = 16
        candidates["dataset.num_instances"] = 4
    if model == "segformer":
        candidates.update({
            "dataset.segment.num_classes": 2,
            "dataset.segment.batch_size": 1,
            "dataset.segment.workers": 0,
            "train.tensorboard.enabled": False,
        })
    if model == "pose-classification":
        candidates.update({
            "dataset.label_map": {f"class_{index}": index for index in range(6)},
            "model.graph_layout": "nvidia",
        })
    if model == "visual-changenet":
        candidates.update({
            "model.backbone.pretrained_backbone_path": VISUAL_CHANGENET_BACKBONE_CONTAINER_PATH,
            "dataset.classify.batch_size": 2,
            "dataset.classify.workers": 0,
        })
    if model in {"depth-net-mono", "depth-net-stereo"}:
        candidates.update({
            "train.precision": "fp32",
            "dataset.train_dataset.batch_size": 1,
            "dataset.train_dataset.workers": 0,
            "dataset.val_dataset.batch_size": 1,
            "dataset.val_dataset.workers": 0,
            "dataset.test_dataset.batch_size": 1,
            "dataset.test_dataset.workers": 0,
            "dataset.infer_dataset.batch_size": 1,
            "dataset.infer_dataset.workers": 0,
        })
    if model == "depth-net-mono":
        candidates.update({
            "model.model_type": "RelativeDepthAnything",
            "dataset.dataset_name": "MonoDataset",
            "dataset.min_depth": None,
            "dataset.max_depth": None,
        })
    if model == "depth-net-stereo":
        candidates.update({
            "model.model_type": "FoundationStereo",
            "model.encoder": "vits",
            "dataset.dataset_name": "StereoDataset",
            "dataset.train_dataset.augmentation.crop_size": [128, 128],
            "dataset.val_dataset.augmentation.crop_size": [128, 128],
            "dataset.test_dataset.augmentation.crop_size": [128, 128],
            "dataset.infer_dataset.augmentation.crop_size": [128, 128],
            "dataset.max_disparity": 128,
            "model.max_disparity": 128,
        })
    if num_classes:
        candidates.update({
            "dataset.num_classes": num_classes,
            "model.num_classes": num_classes,
            "model.sem_seg_head.num_classes": num_classes,
            "num_classes": num_classes,
        })
    if model == "mask2former":
        candidates.update({
            "dataset.train.type": "coco_panoptic",
            "dataset.val.type": "coco_panoptic",
            "dataset.test.type": "coco_panoptic",
            "dataset.contiguous_id": False,
            "dataset.augmentation.train_min_size": [128],
            "dataset.augmentation.train_crop_size": [128, 128],
        })
    if model == "oneformer":
        candidates.update({
            "train.num_gpus": 2,
            "train.gpu_ids": [0, 1],
            "train.precision": "32",
            "dataset.train.batch_size": 1,
            "dataset.val.batch_size": 1,
            "dataset.test.batch_size": 1,
            "dataset.augmentation.train_min_size": [128],
            "dataset.augmentation.train_crop_size": [128, 128],
            "dataset.augmentation.test_min_size": 128,
            "dataset.augmentation.test_max_size": 256,
        })
    if model == "nvdinov2":
        candidates.update({
            "wandb.enable": False,
            "model.backbone.teacher_type": "vit_s",
            "model.backbone.student_type": "vit_s",
            "model.backbone.img_size": 224,
            "dataset.workers": 2,
            "train.num_prototypes": 1024,
            "train.precision": "32-true",
            "train.use_custom_attention": False,
        })
    if model == "nvpanoptix3d":
        candidates.update({
            "train.num_gpus": 2,
            "train.gpu_ids": [0, 1],
            "train.precision": "fp32",
            "train.distributed_strategy": "ddp",
            "train.optim.monitor_name": "train_loss",
            "dataset.enable_3d": True,
            "dataset.contiguous_id": True,
            "dataset.train.batch_size": 1,
            "dataset.val.batch_size": 1,
            "dataset.test.batch_size": 1,
            "dataset.train.num_workers": 1,
            "dataset.val.num_workers": 1,
            "dataset.test.num_workers": 1,
        })
    if model == "sparse4d":
        candidates.update({
            "dataset.num_frames": 3,
            "dataset.sequences.split_num": 1,
            "dataset.train_dataset.sequences_split_num": 1,
            "model.head.instance_bank.num_anchor": 72,
            "model.head.instance_bank.num_temp_instances": 48,
            "model.head.num_output": 72,
            "train.precision": "fp32",
        })
    return {key: value for key, value in candidates.items() if key in keys}


def _automl_settings(
    algorithm: str,
    metric: str,
    model: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "algorithm": algorithm,
        "metric": metric,
        "direction": _direction(metric, model),
        "automl_max_recommendations": 2,
        "automl_max_epochs": 2,
        "automl_reduction_factor": 2,
        "automl_max_concurrent": 1,
        "automl_max_trials": 2,
        "automl_min_top_configs": 1,
        "automl_population_size": 2,
        "automl_max_generations": 1,
        "automl_eval_interval": 1,
        "automl_max_experiments": 2,
    }
    if algorithm == "pbt":
        # One generation only evaluates the initial population and exits before
        # PBT can exploit, perturb, or resume a member.  Two population members
        # over two generations is the smallest end-to-end PBT validation.
        settings.update({
            "automl_max_recommendations": 4,
            "automl_max_generations": 2,
        })
    if algorithm in {"llm", "hybrid", "autoresearch"}:
        key = os.environ.get("AUTOML_LLM_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise RuntimeError("LLM algorithm requested but no LLM API key is present in the environment")
        endpoint = (
            os.environ.get("AUTOML_LLM_ENDPOINT")
            or os.environ.get("base_url")
            or os.environ.get("BASE_URL")
            or os.environ.get("NVIDIA_INFERENCE_ENDPOINT")
        )
        model = (
            os.environ.get("AUTOML_LLM_MODEL")
            or os.environ.get("model")
            or os.environ.get("MODEL")
            or os.environ.get("NVIDIA_INFERENCE_MODEL")
        )
        if not endpoint or not model:
            raise RuntimeError(
                "LLM algorithm requested but no LLM endpoint/model is present in the environment"
            )
        settings.update({
            "llm_endpoint": endpoint,
            "llm_model": model,
            "llm_api_key": key,
        })
    return settings


def _metric_extractor_for(model: str):
    if model != "dino":
        return None

    def extract_dino_map50(logs: str, metric_name: str) -> float | None:
        patterns = [
            r"Validation\s+mAP50\s*[:=]\s*([0-9.]+)",
            r"mAP50\s*[:=]\s*([0-9.]+)",
        ]
        for pattern in patterns:
            matches = list(re.finditer(pattern, logs, flags=re.IGNORECASE))
            if matches:
                return float(matches[-1].group(1))
        return _extract_metric_from_logs(logs, metric_name)

    return extract_dino_map50


def _find_checkpoints(job_root: Path, model: str) -> list[str]:
    checkpoints: list[str] = []
    if not job_root.exists():
        return checkpoints
    for path in job_root.rglob("*"):
        rel = path.relative_to(job_root).as_posix()
        if rel.startswith("inputs/") or rel.startswith("ptm/") or "/inputs/" in rel or "/ptm/" in rel:
            continue
        if model == "cosmos-rl" and path.is_file() and path.name.endswith(".safetensors"):
            checkpoints.append(str(path))
        elif model != "cosmos-rl" and path.is_file() and path.name.lower().endswith(CHECKPOINT_SUFFIXES):
            checkpoints.append(str(path))
        elif path.is_dir() and path.name.lower().startswith(("epoch_", "step_")) and any(path.iterdir()):
            checkpoints.append(str(path))
    return sorted(set(checkpoints))


def _prefer_epoch_or_step_checkpoint(
    checkpoint_paths: list[str],
    model: str | None = None,
) -> str | None:
    exact = []
    fallback = []
    for path in checkpoint_paths:
        name = Path(path).name.lower()
        if "latest" in name:
            fallback.append(path)
            continue
        if re.search(r"(?:^|[_-])(epoch|step)[_-]?\d+", name) or re.search(r"/(?:epoch|step)_\d+", path):
            exact.append(path)
        else:
            fallback.append(path)

    if exact:
        if model == "nvdinov2":
            student = [path for path in exact if Path(path).name.lower().startswith("student_epoch_")]
            if student:
                exact = student

        def checkpoint_rank(path: str) -> tuple[int, int, str]:
            epochs = re.findall(r"(?:^|[_/-])epoch[_-]?(\d+)", path, flags=re.IGNORECASE)
            steps = re.findall(r"(?:^|[_/-])step[_-]?(\d+)", path, flags=re.IGNORECASE)
            return (
                max((int(value) for value in epochs), default=-1),
                max((int(value) for value in steps), default=-1),
                path,
            )

        return max(exact, key=checkpoint_rank)
    return (fallback or [None])[0]


def _checkpoint_progress(path: str) -> tuple[str, int] | None:
    """Return the explicit step or epoch encoded in a checkpoint filename."""
    step_match = re.search(r"(?:^|[_-])step[_-]?(\d+)", Path(path).name)
    if step_match:
        return "step", int(step_match.group(1))
    epoch_match = re.search(r"(?:^|[_-])epoch[_-]?(\d+)", Path(path).name)
    if epoch_match:
        return "epoch", int(epoch_match.group(1))
    return None


def _host_to_container_path(host_path: str, host_root: Path, container_root: str = "/results") -> str:
    path = Path(host_path)
    return f"{container_root.rstrip('/')}/{path.relative_to(host_root).as_posix()}"


def _checkpoint_action_container_path(
    checkpoint_path: str,
    host_root: Path,
    model: str,
) -> str:
    """Return the checkpoint argument expected by eval/inference actions."""
    action_path = Path(checkpoint_path)
    # The real Cosmos-RL checkpoint artifact is the LoRA safetensors file, but
    # its PEFT merge utility accepts the adapter directory so it can read both
    # adapter_config.json and adapter_model.safetensors.
    if model == "cosmos-rl" and action_path.name == "adapter_model.safetensors":
        action_path = action_path.parent
    return _host_to_container_path(str(action_path), host_root)


def _cosmos_inference_media_path(out_dir: Path) -> str:
    """Resolve one real staged video because Cosmos inference is non-recursive."""
    eval_root = out_dir.parent / "datasets" / "cosmos-rl" / "eval"
    annotation_path = eval_root / "annotations.json"
    records = json.loads(annotation_path.read_text())
    if not records or not records[0].get("video"):
        raise ValueError(f"No referenced Cosmos inference video in {annotation_path}")
    relative_video = Path(records[0]["video"])
    media_path = eval_root / relative_video
    if not media_path.is_file():
        raise FileNotFoundError(f"Referenced Cosmos inference video is missing: {media_path}")
    return f"/data/automl_datasets/cosmos-rl/eval/{relative_video.as_posix()}"


def _latest_kpi(job_root: Path) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for status_path in sorted(job_root.rglob("status.json")):
        try:
            lines = status_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            kpi = payload.get("kpi")
            if isinstance(kpi, dict):
                latest.update(kpi)
    return latest


def _metric_from_job(logs: str, kpi: dict[str, Any], metric_name: str) -> float | None:
    """Resolve the requested metric from TAO status KPI first, then logs."""
    candidates = (
        metric_name,
        metric_name.replace("/", "_"),
        metric_name.rsplit("/", 1)[-1],
    )
    for key in candidates:
        value = kpi.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
    return _extract_metric_from_logs(logs, metric_name)


def _minimal_action_overrides(
    specs: dict[str, Any],
    schema_keys: set[str],
    action: str,
    num_classes: int | None,
) -> dict[str, Any]:
    keys = schema_keys | _flatten_keys(specs)
    candidates: dict[str, Any] = {
        "dataset.batch_size": 1,
        "dataset.workers": 0,
        "dataset.num_workers": 0,
        f"{action}.num_gpus": 1,
        f"{action}.gpu_ids": [0],
        f"{action}.batch_size": 1,
        f"{action}.vis_after_n_batches": 1,
    }
    if num_classes:
        candidates.update({
            "dataset.num_classes": num_classes,
            "model.num_classes": num_classes,
            "model.sem_seg_head.num_classes": num_classes,
            "num_classes": num_classes,
        })
    return {key: value for key, value in candidates.items() if key in keys}


def _build_action_specs(
    model_dir: Path,
    skill_text: str,
    profile: ModelProfile,
    action: str,
    checkpoint_container_path: str,
    num_classes: int | None,
    trial_specs: dict[str, Any] | None = None,
    extra_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = _model_profile_key(model_dir)
    template = model_dir / "references" / f"spec_template_{action}.yaml"
    if not template.exists():
        template = model_dir / "references" / "spec_template.yaml"
    specs = _read_yaml(template)
    schema_keys = _schema_keys(model_dir, action)
    overrides: dict[str, Any] = {}
    _add_data_source_overrides(overrides, profile, _parse_action_rows(skill_text, action))
    if model == "grounding-dino":
        if action == "evaluate":
            overrides["dataset.test_data_sources.image_dir"] = f"{profile.eval_uri}/images"
        elif action == "inference":
            overrides["dataset.infer_data_sources.image_dir"] = [f"{profile.eval_uri}/images"]
    if model == "classification-pyt" and action in {"evaluate", "inference"}:
        overrides.update({
            "dataset.val_dataset.images_dir": f"{profile.eval_uri}/images_val",
            "dataset.test_dataset.images_dir": f"{profile.eval_uri}/images_val",
            "dataset.classes_file": f"{profile.eval_uri}/classes.txt",
        })
    if model == "rtdetr":
        overrides.update({
            "dataset.num_classes": 5,
            "dataset.eval_class_ids": [1, 2, 3, 4],
        })
        if action == "evaluate":
            overrides.update({
                "dataset.test_data_sources.image_dir": f"{profile.eval_uri}/images",
                "dataset.test_data_sources.json_file": f"{profile.eval_uri}/annotations.json",
            })
        elif action == "inference":
            overrides["dataset.infer_data_sources"] = {
                "image_dir": [f"{profile.inference_uri}/images"],
                "classmap": f"{profile.inference_uri}/label_map.txt",
            }
    if model == "segformer":
        overrides.update({
            "dataset.segment.root_dir": (
                profile.eval_uri if action == "evaluate" else profile.inference_uri
            ),
            "dataset.segment.num_classes": 2,
            "dataset.segment.batch_size": 1,
            "dataset.segment.workers": 0,
        })
    if model == "mae":
        split = "val" if action == "evaluate" else "test"
        overrides.update({
            "train.stage": "finetune",
            "model.arch": "convnextv2_atto",
            f"dataset.{split}_data_sources": (
                f"{profile.eval_uri}/images_val"
            ),
        })
    if model == "mal":
        overrides.update({
            "dataset.val_img_dir": f"{profile.eval_uri}/images",
            "dataset.val_ann_path": f"{profile.eval_uri}/annotations.json",
            "dataset.crop_size": 256,
            "dataset.num_workers_per_gpu": 1,
            "model.arch": "vit-deit-small/16",
        })
        if action == "inference":
            overrides.update({
                "inference.img_dir": f"{profile.eval_uri}/images",
                "inference.ann_path": f"{profile.eval_uri}/annotations.json",
                "inference.load_mask": False,
            })
    if model == "cosmos-rl":
        if action == "evaluate":
            overrides.update({
                "dataset.annotation_path": _join_uri(profile.eval_uri, "annotations.json"),
                "dataset.media_dir": profile.eval_uri,
                "evaluation.num_processes": 1,
                "evaluation.limit": 2,
                "evaluation.batch_size": 1,
                "generation.max_tokens": 32,
                "vision.nframes": 2,
                "model.model_name": COSMOS_MODEL_CONTAINER_PATH,
                "model.base_model_path": COSMOS_MODEL_CONTAINER_PATH,
            })
            if checkpoint_container_path:
                overrides.update({
                    "model.model_name": checkpoint_container_path,
                    "model.enable_lora": True,
                })
        elif action == "inference":
            overrides.update({
                "media": profile.eval_uri,
                "base_model_path": COSMOS_MODEL_CONTAINER_PATH,
            })
            if checkpoint_container_path:
                overrides.update({
                    "model_path": checkpoint_container_path,
                    "enable_lora": True,
                })
    if model == "clip" and action == "evaluate":
        overrides.update({
            "dataset.val.batch_size": 1,
            "dataset.val.num_workers": 0,
            "evaluate.batch_size": 1,
            "evaluate.num_workers": 0,
        })
    if model == "clip" and action == "inference":
        # The S3 validation source has images but no prompts file. Image-only
        # inference is a supported CLIP workflow; clear the optional path that
        # generic data-source mapping would otherwise synthesize.
        overrides["inference.text_file"] = None
    overrides.update(_minimal_action_overrides(specs, schema_keys, action, num_classes))
    if model == "rtdetr":
        overrides.update({
            "dataset.num_classes": 5,
            "dataset.eval_class_ids": [1, 2, 3, 4],
        })
    if model == "deformable-detr" and "dataset.workers" in schema_keys:
        overrides["dataset.workers"] = 1
    if profile.model_type and "model.model_type" in (schema_keys | _flatten_keys(specs)):
        overrides["model.model_type"] = profile.model_type
    if profile.dataset_name and "dataset.dataset_name" in (schema_keys | _flatten_keys(specs)):
        overrides["dataset.dataset_name"] = profile.dataset_name
    if model == "depth-net-stereo":
        source_root = profile.eval_uri if action in {"evaluate", "inference"} else profile.train_uri
        split = {"evaluate": "test", "inference": "infer"}.get(action, action)
        overrides.update({
            "model.encoder": "vits",
            "model.max_disparity": 128,
            "dataset.max_disparity": 128,
            f"dataset.{split}_dataset.augmentation.crop_size": [128, 128],
            f"dataset.{split}_dataset.data_sources": [{
                "data_file": _join_uri(source_root, "annotations.txt"),
                "dataset_name": "Middlebury",
            }],
        })
    if model == "action-recognition" and "dataset.label_map" in (schema_keys | _flatten_keys(specs)):
        overrides["dataset.label_map"] = {"catch": 0, "smile": 1}
    if model == "mask-grounding-dino":
        overrides.update({
            "dataset.val_data_sources.data_type": "OD",
            "dataset.test_data_sources.data_type": "OD",
            "dataset.infer_data_sources.data_type": "OD",
        })
        if action == "evaluate":
            overrides.update({
                "dataset.test_data_sources.image_dir": f"{profile.eval_uri}/images",
                "dataset.test_data_sources.json_file": f"{profile.eval_uri}/annotations.json",
            })
        elif action == "inference":
            overrides["dataset.infer_data_sources.image_dir"] = f"{profile.eval_uri}/images"
    if model == "mask2former":
        overrides.update({
            "dataset.train.type": "coco_panoptic",
            "dataset.val.type": "coco_panoptic",
            "dataset.test.type": "coco_panoptic",
            "dataset.contiguous_id": False,
            "dataset.train.img_dir": f"{profile.train_uri}/images",
            "dataset.label_map": f"{profile.train_uri}/label_map_panoptic.json",
            "dataset.train.instance_json": f"{profile.train_uri}/annotations.json",
            "dataset.train.panoptic_json": f"{profile.train_uri}/annotations_panoptic.json",
            "dataset.train.panoptic_dir": f"{profile.train_uri}/images_panoptic",
            "dataset.val.img_dir": f"{profile.eval_uri}/images",
            "dataset.val.instance_json": f"{profile.eval_uri}/annotations.json",
            "dataset.val.panoptic_json": f"{profile.eval_uri}/annotations_panoptic.json",
            "dataset.val.panoptic_dir": f"{profile.eval_uri}/images_panoptic",
            "dataset.test.img_dir": f"{profile.eval_uri}/images",
        })
    if model == "pose-classification" and action == "inference":
        overrides["inference.output_file"] = "/results/pose_classification_inference.txt"
    if model == "pose-classification":
        overrides.update({
            "dataset.label_map": {f"class_{index}": index for index in range(6)},
            "model.graph_layout": "nvidia",
        })
    if model == "re-identification":
        if action == "evaluate":
            overrides["evaluate.output_cmc_curve_plot"] = "/results/reid_cmc_curve.png"
            overrides["evaluate.output_sampled_matches_plot"] = "/results/reid_sampled_matches.png"
        if action == "inference":
            overrides["inference.output_file"] = "/results/reid_inference.json"
    if model == "ml-recog":
        if action == "evaluate":
            overrides.update({
                "dataset.val_dataset": {
                    "reference": "/data/ml-recog/unknown/reference/reference",
                    "query": "/data/ml-recog/unknown/test/test",
                },
            })
        elif action == "inference":
            overrides.update({
                "dataset.val_dataset": {
                    "reference": "/data/ml-recog/unknown/reference/reference",
                },
                "inference.input_path": "/data/ml-recog/unknown/test/test",
            })
    if model == "visual-changenet":
        overrides["model.backbone.pretrained_backbone_path"] = VISUAL_CHANGENET_BACKBONE_CONTAINER_PATH
    if model == "nvdinov2":
        overrides.update({
            "wandb.enable": False,
            "model.backbone.teacher_type": "vit_s",
            "model.backbone.student_type": "vit_s",
            "model.backbone.img_size": 224,
            "train.num_prototypes": 1024,
            "train.precision": "32-true",
            "train.use_custom_attention": False,
        })
        if action == "inference":
            overrides["dataset.workers"] = 2
            overrides["dataset.test_dataset.images_dir"] = "/data/nvdinov2-mini/images_train/images_train"
    if model == "nvpanoptix3d":
        overrides.update({
            "dataset.enable_3d": True,
            "dataset.contiguous_id": True,
            "dataset.frustum_mask_path": f"{profile.eval_uri}/meta/frustum_mask.npz",
            "dataset.label_map": f"{profile.eval_uri}/meta/colormap.json",
        })
        if action == "evaluate":
            overrides.update({
                "dataset.val.json_path": f"{profile.eval_uri}/meta/val.json",
                "dataset.val.base_dir": profile.eval_uri,
                "dataset.test.json_path": f"{profile.eval_uri}/meta/test.json",
                "dataset.test.base_dir": profile.eval_uri,
            })
        elif action == "inference":
            overrides["inference.images_dir"] = f"{profile.eval_uri}/inference_flat"
    if model == "ocdnet":
        overrides["dataset.validate_dataset.data_path"] = [profile.eval_uri]
        if action == "inference":
            overrides["inference.input_folder"] = profile.inference_uri
    if model == "ocrnet":
        overrides["dataset.character_list_file"] = "/data/ocrnet/character_list"
        if action == "evaluate":
            overrides.update({
                "evaluate.test_dataset_dir": profile.eval_uri,
                "evaluate.test_dataset_gt_file": f"{profile.eval_uri}/gt_new.txt",
            })
        elif action == "inference":
            overrides["inference.inference_dataset_dir"] = profile.inference_uri
    if model == "oneformer":
        overrides.update({
            "model.sem_seg_head.num_classes": 133,
            "dataset.contiguous_id": True,
            "dataset.train.images": f"{profile.train_uri}/images",
            "dataset.train.annotations": f"{profile.train_uri}/annotations.json",
            "dataset.label_map": f"{profile.train_uri}/label_map.json",
            "dataset.train.panoptic": f"{profile.train_uri}/images_panoptic",
            "dataset.val.images": f"{profile.eval_uri}/images",
            "dataset.val.annotations": f"{profile.eval_uri}/annotations.json",
            "dataset.val.panoptic": f"{profile.eval_uri}/images_panoptic",
            "dataset.test.images": f"{profile.eval_uri}/images",
        })
        if action == "evaluate":
            overrides.update({
                "dataset.test.annotations": f"{profile.eval_uri}/annotations.json",
                "dataset.test.panoptic": f"{profile.eval_uri}/images_panoptic",
            })
        elif action == "inference":
            overrides["inference.images_dir"] = profile.inference_uri
    if model == "optical-inspection":
        if action == "evaluate":
            overrides.update({
                "dataset.test_dataset.images_dir": f"{profile.eval_uri}/images",
                "dataset.test_dataset.csv_path": f"{profile.eval_uri}/dataset.csv",
            })
        elif action == "inference":
            overrides.update({
                "dataset.infer_dataset.images_dir": f"{profile.inference_uri}/images",
                "dataset.infer_dataset.csv_path": f"{profile.inference_uri}/dataset.csv",
            })
    if trial_specs:
        overrides.update(_valid_set(trial_specs, specs, schema_keys))
    if extra_overrides:
        overrides.update(_valid_set(extra_overrides, specs, schema_keys))
    if model == "mae" and action in {"evaluate", "inference"}:
        overrides["train.stage"] = "finetune"
    for key in (f"{action}.checkpoint", f"{action}.model_path", f"{action}.pretrained_model_path"):
        if key in (schema_keys | _flatten_keys(specs)):
            overrides[key] = checkpoint_container_path
            break
    overrides = _valid_set(overrides, specs, schema_keys)
    if model == "mask-grounding-dino" and action == "inference":
        overrides["dataset.infer_data_sources.captions"] = list(profile.captions or ("object",))
    for dotted_key, value in overrides.items():
        _set_nested(specs, dotted_key, value)
    return specs


def _resolve_action_image(skill_info: dict[str, Any], action_cfg: dict[str, Any]) -> str:
    from tao_sdk.versions import resolve_container_image

    return resolve_container_image(
        action_cfg.get("container_image") or skill_info.get("container_image", "")
    )


def _run_action_job(
    *,
    sdk: DockerSDK,
    image: str,
    action_cfg: dict[str, Any],
    specs: dict[str, Any],
    action: str,
    out_dir: Path,
    args: argparse.Namespace,
    mounts: list[dict[str, str]],
    env_vars: dict[str, str] | None = None,
    gpu_count: int | None = None,
    metric_name: str | None = None,
) -> dict[str, Any]:
    ep = build_entrypoint(
        command=action_cfg["command"],
        specs=specs,
        inputs=action_cfg.get("inputs"),
        outputs=action_cfg.get("outputs"),
        config_format=action_cfg.get("config_format", "toml"),
        upload_excludes=action_cfg.get("upload_excludes", []),
    )
    job = sdk.create_job(
        image=image,
        command=ep["command"],
        gpu_count=args.num_gpus if gpu_count is None else gpu_count,
        gpu_device_ids=_gpu_device_ids(args, gpu_count),
        env_vars=env_vars,
        mounts=mounts,
    )
    while True:
        time.sleep(args.poll_interval)
        status = sdk.get_job_status(job.id)
        LOG.info("%s job %s status=%s", action, job.id, status.status)
        if status.status in {"Complete", "Error", "Canceled"}:
            break
    logs = sdk.get_job_logs(job.id)
    job_root = out_dir / "results" / job.id
    success = status.status == "Complete" and "Execution status: FAIL" not in logs
    kpi = _latest_kpi(job_root)
    return {
        "action": action,
        "status": "success" if success else "failed",
        "job_id": job.id,
        "docker_status": status.status,
        "checkpoint_spec": _get_nested(specs, f"{action}.checkpoint"),
        "kpi": kpi,
        "metric_name": metric_name,
        "metric_value": _metric_from_job(logs, kpi, metric_name) if metric_name else None,
        "logs_tail": logs.splitlines()[-80:],
        "status_files": [str(path) for path in sorted(job_root.rglob("status.json"))],
    }


def _build_dataset_convert_specs(
    *,
    model_dir: Path,
    skill_text: str,
    profile: ModelProfile,
) -> dict[str, Any]:
    specs = _read_yaml(model_dir / "references" / "spec_template_dataset_convert.yaml")
    schema_keys = _schema_keys(model_dir, "dataset_convert")
    overrides: dict[str, Any] = {}
    _add_data_source_overrides(overrides, profile, _parse_action_rows(skill_text, "dataset_convert"))
    if _model_profile_key(model_dir) == "bevfusion":
        overrides.update({
            "root_dir": "/data/bevfusion",
            "results_dir": "/data/bevfusion",
            "mode": "training",
        })
    if _model_profile_key(model_dir) == "sparse4d":
        overrides.update({
            "aicity.num_frames": 3,
            "aicity.anchor_init_config.num_anchor": 72,
            # Sparse4D anchor initialization needs camera-group annotations.
            # The converter's supported default generates those annotations;
            # disabling grouping leaves no arrays for anchor concatenation.
            "aicity.camera_grouping_mode": "random",
        })
    overrides = _valid_set(overrides, specs, schema_keys)
    for dotted_key, value in overrides.items():
        _set_nested(specs, dotted_key, value)
    return specs


def _normalize_sparse4d_depth_paths(
    *,
    model_dir: Path,
    out_dir: Path,
    convert_root: Path,
) -> dict[str, Any] | None:
    script = model_dir / "scripts" / "normalize_depth_paths.py"
    train_root = out_dir / "aicity_root" / "train"
    train_ann_dir = convert_root / "train"
    if not script.exists() or not train_root.exists() or not train_ann_dir.exists():
        return None
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-root",
            str(train_root),
            str(train_ann_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "returncode": result.returncode,
        "stdout_tail": result.stdout.splitlines()[-10:],
        "stderr_tail": result.stderr.splitlines()[-10:],
    }


def _run_dataset_convert_preflight(
    *,
    args: argparse.Namespace,
    model_dir: Path,
    skill_text: str,
    skill_info: dict[str, Any],
    profile: ModelProfile,
    out_dir: Path,
    sdk: DockerSDK,
    mounts: list[dict[str, str]],
) -> dict[str, Any]:
    model = _model_profile_key(model_dir)
    actions = skill_info.get("actions") or {}
    action_cfg = actions.get("dataset_convert")
    template = model_dir / "references" / "spec_template_dataset_convert.yaml"
    if not action_cfg or not template.exists():
        return {
            "status": "failed",
            "reason": "dataset_convert action/template is not packaged by skill",
        }

    if model == "ocrnet":
        image = _resolve_action_image(skill_info, action_cfg)
        conversion_jobs: dict[str, dict[str, Any]] = {}
        lmdb_roots: dict[str, Path] = {}
        for split, image_dir, gt_file in (
            ("train", profile.train_uri, f"{profile.train_uri}/gt_new.txt"),
            ("val", profile.eval_uri, f"{profile.eval_uri}/gt_new.txt"),
        ):
            split_specs = _read_yaml(template)
            _set_nested(split_specs, "dataset_convert.input_img_dir", image_dir)
            _set_nested(split_specs, "dataset_convert.gt_file", gt_file)
            result = _run_action_job(
                sdk=sdk,
                image=image,
                action_cfg=action_cfg,
                specs=split_specs,
                action=f"dataset_convert_{split}",
                out_dir=out_dir,
                args=args,
                mounts=mounts,
                gpu_count=0,
            )
            conversion_jobs[split] = result
            if result["status"] != "success":
                return {
                    "status": "failed",
                    "reason": f"OCRNet {split} dataset_convert job failed",
                    "jobs": conversion_jobs,
                }
            job_root = out_dir / "results" / result["job_id"]
            data_file = next(iter(sorted(job_root.rglob("data.mdb"))), None)
            lock_file = next(iter(sorted(job_root.rglob("lock.mdb"))), None)
            if not data_file or not lock_file or data_file.parent != lock_file.parent:
                return {
                    "status": "failed",
                    "reason": f"OCRNet {split} dataset_convert completed without a usable LMDB folder",
                    "jobs": conversion_jobs,
                }
            lmdb_roots[split] = data_file.parent

        train_lmdb = _host_to_container_path(str(lmdb_roots["train"]), out_dir / "results")
        val_lmdb = _host_to_container_path(str(lmdb_roots["val"]), out_dir / "results")
        return {
            "status": "passed",
            "jobs": conversion_jobs,
            "artifacts": {
                "train_lmdb": str(lmdb_roots["train"]),
                "val_lmdb": str(lmdb_roots["val"]),
            },
            "train_overrides": {
                "dataset.train_dataset_dir": [train_lmdb],
                "dataset.val_dataset_dir": val_lmdb,
                "dataset.train_gt_file": "",
                "dataset.val_gt_file": "",
                "dataset.character_list_file": "/data/ocrnet/character_list",
            },
        }

    specs = _build_dataset_convert_specs(
        model_dir=model_dir,
        skill_text=skill_text,
        profile=profile,
    )
    if model == "bevfusion":
        # BEVFusion 5.5 requires results_dir == root_dir while reducing points.
        # Keep the mounted path instead of letting output materialization replace it.
        action_cfg = dict(action_cfg)
        action_cfg["outputs"] = {}
    image = _resolve_action_image(skill_info, action_cfg)
    job_result = _run_action_job(
        sdk=sdk,
        image=image,
        action_cfg=action_cfg,
        specs=specs,
        action="dataset_convert",
        out_dir=out_dir,
        args=args,
        mounts=mounts,
        gpu_count=args.num_gpus if model in {"pointpillars", "sparse4d"} else 0,
    )
    if job_result["status"] != "success":
        return {
            "status": "failed",
            "reason": "dataset_convert job failed",
            "job": job_result,
            "specs": specs,
        }

    convert_root = out_dir / "results" / job_result["job_id"] / "results_dir"
    train_overrides: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    missing: list[str] = []
    extra: dict[str, Any] = {}

    if model == "pointpillars":
        data_info = convert_root / "data_info"
        required = {
            "dbinfos_train": data_info / "dbinfos_train.pkl",
            "infos_train": data_info / "infos_train.pkl",
            "infos_val": data_info / "infos_val.pkl",
        }
        for name, path in required.items():
            if path.exists():
                artifacts[name] = str(path)
            else:
                missing.append(str(path))
        if not missing:
            train_overrides["dataset.data_info_path"] = _host_to_container_path(
                str(data_info),
                out_dir / "results",
            )

    elif model == "bevfusion":
        data_root = out_dir / "data_mount" / "bevfusion"
        train_ann = next(iter(sorted(data_root.rglob("kitti_person_infos_train.pkl"))), None)
        val_ann = next(iter(sorted(data_root.rglob("kitti_person_infos_val.pkl"))), None)
        reduced = data_root / "training" / "velodyne_reduced"
        required_bevfusion = {
            "train_ann": train_ann,
            "val_ann": val_ann,
            "velodyne_reduced": reduced if reduced.exists() else None,
        }
        for name, path in required_bevfusion.items():
            if path and path.exists():
                artifacts[name] = str(path)
            else:
                missing.append(name)
        if not missing:
            data_prefix = {
                "pts": "training/velodyne_reduced",
                "img": "training/image_2",
            }
            train_overrides.update({
                "dataset.root_dir": "/data/bevfusion",
                "dataset.train_dataset": {
                    "ann_file": f"/data/bevfusion/{train_ann.relative_to(data_root).as_posix()}",
                    "data_prefix": data_prefix,
                    "batch_size": 1,
                    "num_workers": 1,
                },
                "dataset.val_dataset": {
                    "ann_file": f"/data/bevfusion/{val_ann.relative_to(data_root).as_posix()}",
                    "data_prefix": data_prefix,
                    "batch_size": 1,
                    "num_workers": 1,
                },
                "dataset.test_dataset": {
                    "ann_file": f"/data/bevfusion/{val_ann.relative_to(data_root).as_posix()}",
                    "data_prefix": data_prefix,
                    "batch_size": 1,
                    "num_workers": 1,
                },
            })

    elif model == "sparse4d":
        extra["depth_path_normalization"] = _normalize_sparse4d_depth_paths(
            model_dir=model_dir,
            out_dir=out_dir,
            convert_root=convert_root,
        )
        anchor = next(iter(sorted(convert_root.rglob("anchor_init.npy"))), None)
        train_ann = sorted(convert_root.rglob("*_infos_train.pkl"))
        val_ann = sorted(convert_root.rglob("*_infos_val.pkl"))
        test_ann = sorted(convert_root.rglob("*_infos_test.pkl"))
        # The three-camera smoke fixture cannot use Data Services' random groups
        # (which require 5-10 cameras) and emits one current-job training file.
        # Reuse that freshly converted annotation for smoke val/test rather than
        # searching another job directory or accepting stale converted pickles.
        train_path = train_ann[0] if train_ann else None
        val_path = val_ann[0] if val_ann else train_path
        test_path = test_ann[0] if test_ann else train_path
        required_sparse = {
            "anchor": anchor,
            "train_ann": train_path,
            "val_ann": val_path,
            "test_ann": test_path,
        }
        for name, path in required_sparse.items():
            if path and path.exists():
                artifacts[name] = str(path)
            else:
                missing.append(name)
        if not missing:
            train_overrides.update({
                "dataset.data_root": "/data/aicity_root/train",
                "model.head.instance_bank.anchor": _host_to_container_path(
                    str(required_sparse["anchor"]),
                    out_dir / "results",
                ),
                "dataset.train_dataset.ann_file": _host_to_container_path(
                    str(required_sparse["train_ann"]),
                    out_dir / "results",
                ),
                "dataset.val_dataset.ann_file": _host_to_container_path(
                    str(required_sparse["val_ann"]),
                    out_dir / "results",
                ),
                "dataset.test_dataset.ann_file": _host_to_container_path(
                    str(required_sparse["test_ann"]),
                    out_dir / "results",
                ),
                "dataset.num_frames": 3,
                "dataset.sequences.split_num": 1,
                "dataset.train_dataset.sequences_split_num": 1,
                "model.head.instance_bank.num_anchor": 72,
                "model.head.instance_bank.num_temp_instances": 48,
                "model.head.num_output": 72,
                "train.precision": "fp32",
            })

    if missing:
        return {
            "status": "failed",
            "reason": "dataset_convert completed but required converted artifacts are missing",
            "missing_artifacts": missing,
            "artifacts": artifacts,
            "job": job_result,
            "specs": specs,
            **extra,
        }
    return {
        "status": "passed",
        "job": job_result,
        "convert_root": str(convert_root),
        "artifacts": artifacts,
        "train_overrides": train_overrides,
        "specs": specs,
        **extra,
    }


def _run_post_checks(
    *,
    args: argparse.Namespace,
    model_dir: Path,
    skill_text: str,
    skill_info: dict[str, Any],
    profile: ModelProfile,
    out_dir: Path,
    payload: dict[str, Any],
    sdk: DockerSDK,
    num_classes: int | None,
) -> dict[str, Any]:
    model = _model_profile_key(model_dir)
    checkpoints = payload.get("best_checkpoint_paths") or []
    checkpoint_path = _prefer_epoch_or_step_checkpoint(checkpoints, model=model)
    if not checkpoint_path:
        payload["checkpoint_validation"] = {
            "status": "failed",
            "reason": "no real checkpoint path found for best recommendation",
        }
        return payload

    best_rec_id = ((payload.get("result") or {}).get("best") or {}).get("rec_id")
    resume_record = next(
        (
            item for item in payload.get("resume_behavior", [])
            if item.get("rec_id") == best_rec_id
        ),
        None,
    )
    if resume_record:
        parent_path = str(resume_record.get("resume_checkpoint_path") or "")
        parent_progress = _checkpoint_progress(parent_path)
        promoted_progress = _checkpoint_progress(checkpoint_path)
        if (
            parent_progress
            and promoted_progress
            and parent_progress[0] == promoted_progress[0]
            and promoted_progress[1] <= parent_progress[1]
        ):
            payload["checkpoint_validation"] = {
                "status": "failed",
                "reason": (
                    "promoted checkpoint did not advance beyond its resume parent: "
                    f"{parent_progress[0]} {parent_progress[1]} -> "
                    f"{promoted_progress[1]}"
                ),
                "checkpoint_path": checkpoint_path,
                "resume_checkpoint_path": parent_path,
                "uses_latest": "latest" in Path(checkpoint_path).name.lower(),
            }
            payload["status"] = "failed"
            return payload

    host_root = out_dir / "results"
    checkpoint_container_path = _checkpoint_action_container_path(
        checkpoint_path, host_root, model
    )
    actions = skill_info.get("actions") or {}
    post_checks = []
    best_trial_specs = ((payload.get("result") or {}).get("best") or {}).get("specs") or {}
    dataset_convert_overrides = (
        (payload.get("dataset_convert") or {}).get("train_overrides") or {}
    )
    mounts = _mounts_for_model(out_dir, model, profile)
    action_env_vars = (
        {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}
        if model in {"clip", "ml-recog", "ocrnet", "oneformer", "optical-inspection", "re-identification"}
        else None
    )
    for action in ("evaluate", "inference"):
        if action == "evaluate":
            final_evaluation = next(
                (
                    item for item in payload.get("final_evaluation_jobs", [])
                    if item.get("action") == "final_evaluate"
                    and item.get("status") == "success"
                    and item.get("checkpoint_path") == checkpoint_path
                ),
                None,
            )
            if final_evaluation:
                reused = copy.deepcopy(final_evaluation)
                reused["action"] = "evaluate"
                reused["reused_from"] = "final_evaluate"
                post_checks.append(reused)
                continue
        action_cfg = actions.get(action)
        template = model_dir / "references" / f"spec_template_{action}.yaml"
        if not template.exists():
            template = model_dir / "references" / "spec_template.yaml"
        if not action_cfg or not template.exists():
            post_checks.append({
                "action": action,
                "status": "skipped",
                "reason": "action/template not packaged by skill",
            })
            continue
        image = _resolve_action_image(skill_info, action_cfg)
        specs = _build_action_specs(
            model_dir=model_dir,
            skill_text=skill_text,
            profile=profile,
            action=action,
            checkpoint_container_path=checkpoint_container_path,
            num_classes=num_classes,
            trial_specs=best_trial_specs,
            extra_overrides=dataset_convert_overrides,
        )
        if model == "cosmos-rl" and action == "inference":
            specs["media"] = _cosmos_inference_media_path(out_dir)
        post_checks.append(_run_action_job(
            sdk=sdk,
            image=image,
            action_cfg=action_cfg,
            specs=specs,
            action=action,
            out_dir=out_dir,
            args=args,
            mounts=mounts,
            env_vars=action_env_vars,
        ))

    payload["checkpoint_validation"] = {
        "status": (
            "success"
            if post_checks and all(item["status"] in {"success", "skipped"} for item in post_checks)
            else "failed"
        ),
        "checkpoint_path": checkpoint_path,
        "checkpoint_container_path": checkpoint_container_path,
        "uses_latest": "latest" in Path(checkpoint_path).name.lower(),
        "post_checks": post_checks,
    }
    train_passed = (
        bool(payload.get("jobs"))
        and all(data.get("status") == "success" for data in payload.get("jobs", {}).values())
        and ((payload.get("result") or {}).get("best") or {}).get("metric_value") is not None
        and bool(payload.get("best_checkpoint_paths"))
    )
    if train_passed and payload["checkpoint_validation"]["status"] == "success":
        payload["status"] = "passed"
    elif train_passed:
        payload["status"] = "failed"
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _select_best_job(
    job_runs: list[dict[str, Any]],
    best: dict[str, Any],
    latest_jobs: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Resolve the best run by job ID before falling back to member ID.

    PBT reuses recommendation/member IDs across generations, so the latest run
    for a member is not necessarily the globally best checkpoint.
    """
    best_job_id = best.get("job_id")
    if best_job_id:
        for run in reversed(job_runs):
            if run.get("job_id") == best_job_id:
                return run
    return latest_jobs.get(best.get("rec_id"), {})


def _supported_automl_parameters(skill_bank: Path, model: str) -> list[str] | None:
    schema_path = skill_bank / "skills" / "models" / model / "schemas" / "train.schema.json"

    def schema_defaults() -> list[str] | None:
        if not schema_path.exists():
            return None
        try:
            params = json.loads(schema_path.read_text()).get("automl_default_parameters")
        except json.JSONDecodeError:
            return None
        return params if isinstance(params, list) else None

    support_path = skill_bank / "skills" / "models" / "automl_support.json"
    if not support_path.exists():
        return schema_defaults()
    try:
        support = json.loads(support_path.read_text())
    except json.JSONDecodeError:
        return schema_defaults()
    for item in support.get("supported", []):
        if item.get("model") == model:
            params = item.get("automl_default_parameters", [])
            return params or schema_defaults() or []
    return schema_defaults()


def _pbt_resume_safe_parameters(params: list[str], model: str) -> list[str]:
    """Keep PBT perturbations compatible with checkpoints being resumed."""
    resume_safe = [
        param for param in params
        if param.startswith(("train.optim.", "train.optimizer."))
        or param in {"train.optm_lr", "train.lr", "train.learning_rate"}
    ]
    # NVDINOv2's generated search surface contains only data-loader controls.
    # Worker count is checkpoint-neutral and therefore safe across generations.
    if not resume_safe and model == "nvdinov2" and "dataset.workers" in params:
        return ["dataset.workers"]
    return resume_safe


def _minimal_custom_ranges(
    params: list[str] | None,
    model: str | None = None,
    schema_path: Path | None = None,
) -> dict[str, dict[str, Any]] | None:
    ranges: dict[str, dict[str, Any]] = {}
    for param in params or []:
        lower = param.lower()
        if "epoch" in lower:
            ranges[param] = {"valid_min": 1, "valid_max": 1}
        elif model == "rtdetr" and lower == "dataset.batch_size":
            # RT-DETR's AutoML search space includes batch size 2, which can
            # exhaust a single validation GPU for otherwise valid model
            # configurations. Keep the one-GPU validation workflow minimal.
            ranges[param] = {"valid_min": 1, "valid_max": 1}
        elif model == "ml-recog" and lower == "train.batch_size":
            ranges[param] = {"valid_min": 4, "valid_max": 4}
        elif "batch_size" in lower or lower.endswith("mini_batch") or ".mini_batch" in lower:
            ranges[param] = {"valid_min": 1, "valid_max": 2}
        elif model == "nvdinov2" and lower == "dataset.workers":
            ranges[param] = {"valid_min": 2, "valid_max": 2}
        elif lower.endswith("workers") or "num_workers" in lower:
            ranges[param] = {"valid_min": 0, "valid_max": 0}
        elif lower == "policy.lora.r":
            ranges[param] = {"valid_min": 2, "valid_max": 8}
        elif lower == "policy.lora.lora_alpha":
            ranges[param] = {"valid_min": 2, "valid_max": 16}
        elif lower == "policy.lora.lora_dropout":
            ranges[param] = {"valid_min": 0.0, "valid_max": 0.05}
        elif lower == "model.corr_radius":
            ranges[param] = {"valid_min": 4, "valid_max": 4}
        elif lower == "model.cv_group":
            ranges[param] = {"valid_min": 8, "valid_max": 8}
        elif lower == "model.volume_dim":
            ranges[param] = {"valid_min": 32, "valid_max": 32}
        elif model in {"dino", "grounding-dino"} and lower.endswith("num_queries"):
            ranges[param] = {"valid_min": 100, "valid_max": 100}
        elif lower.endswith("num_queries"):
            ranges[param] = {"valid_min": 20, "valid_max": 50}
        elif lower.endswith("num_select"):
            ranges[param] = {"valid_min": 1, "valid_max": 20}
        elif lower.endswith("enc_layers") or lower.endswith("dec_layers"):
            ranges[param] = {"valid_min": 1, "valid_max": 2}
        elif "random_crop" in lower:
            ranges[param] = {"valid_min": 128, "valid_max": 256}
        elif lower.endswith("hidden_dim"):
            ranges[param] = {"valid_min": 256, "valid_max": 256}
        elif lower.endswith("train_max_size") or lower.endswith("test_max_size"):
            ranges[param] = {"valid_min": 256, "valid_max": 256}
        elif lower.endswith("test_min_size"):
            ranges[param] = {"valid_min": 128, "valid_max": 128}
        elif lower in {"train.optim.lr", "train.lr"} or lower.endswith("_lr") or lower.endswith("learning_rate"):
            ranges[param] = {"valid_min": 0.00001, "valid_max": 0.001}
        elif lower in {"train.wd", "train.optim.weight_decay"} or lower.endswith("weight_decay"):
            ranges[param] = {"valid_min": 0.0, "valid_max": 0.0001}
        elif lower == "train.optim.lr_backbone":
            ranges[param] = {"valid_min": 0.000001, "valid_max": 0.0001}
        elif lower == "train.optim.lr_linear_proj_mult":
            ranges[param] = {"valid_min": 0.01, "valid_max": 0.1}

    # The built-in AutoML config can be older or more permissive than the
    # selected skill schema. Clamp every validation range to the real skill
    # bounds so the optimizer never proposes a value that the workflow itself
    # declares invalid.
    if schema_path and schema_path.exists():
        schema = json.loads(schema_path.read_text())
        for param, custom_range in ranges.items():
            node: Any = schema
            for part in param.replace("[0]", "").split("."):
                properties = node.get("properties", {}) if isinstance(node, dict) else {}
                node = properties.get(part)
                if not isinstance(node, dict):
                    break
            if not isinstance(node, dict):
                continue
            minimum = node.get("minimum")
            maximum = node.get("maximum")
            low = custom_range["valid_min"]
            high = custom_range["valid_max"]
            if isinstance(minimum, (int, float)) and math.isfinite(minimum):
                low = max(low, minimum)
                if high < minimum:
                    high = minimum
            if isinstance(maximum, (int, float)) and math.isfinite(maximum):
                high = min(high, maximum)
                if low > maximum:
                    low = maximum
            custom_range["valid_min"] = low
            custom_range["valid_max"] = high
    return ranges or None


def _read_s3_json(uri: str) -> Any:
    import boto3

    parsed = urlparse(uri)
    client = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("SECRET_KEY"),
        endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
    )
    body = client.get_object(
        Bucket=parsed.netloc,
        Key=parsed.path.lstrip("/"),
    )["Body"].read()
    return json.loads(body)


def _download_s3_file(uri: str, destination: Path) -> None:
    import boto3

    if destination.exists() and destination.stat().st_size > 0:
        return
    parsed = urlparse(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    client = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("SECRET_KEY"),
        endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
    )
    with destination.open("wb") as fh:
        client.download_fileobj(parsed.netloc, parsed.path.lstrip("/"), fh)


def _prepare_depth_data_mount(profile: ModelProfile, out_dir: Path, model: str) -> Path:
    data_root = out_dir / "data_mount"
    for uri in (profile.train_uri, profile.eval_uri):
        dataset_name = uri.rstrip("/").rsplit("/", 1)[-1]
        target = data_root / dataset_name
        archive = target / "images.tar.gz"
        _download_s3_file(_join_uri(uri, "images.tar.gz"), archive)
        if not (target / "left").exists():
            with tarfile.open(archive) as tar:
                tar.extractall(target, filter="data")
        if model == "depth-net-stereo":
            _download_s3_file(_join_uri(uri, "annotations.txt"), target / "annotations.txt")
        if model == "depth-net-mono":
            source_annotations = target / "annotations_stereo.txt"
            _download_s3_file(_join_uri(uri, "annotations.txt"), source_annotations)
            mono_lines = []
            for line in source_annotations.read_text().splitlines():
                fields = line.split()
                if len(fields) < 3:
                    raise ValueError(
                        f"Expected stereo depth annotation with at least 3 fields in {uri}: {line!r}"
                    )
                mono_lines.append(f"{fields[0]} {fields[-1]}")
            (target / "annotations.txt").write_text("\n".join(mono_lines) + "\n")
    return data_root


def _prepare_bevfusion_data_mount(profile: ModelProfile, out_dir: Path) -> Path:
    data_root = out_dir / "data_mount" / "bevfusion"
    for filename, extracted_dir in (
        ("ImageSets.tar.gz", "ImageSets"),
        ("training.tar.gz", "training"),
        ("testing.tar.gz", "testing"),
    ):
        archive = data_root / filename
        _download_s3_file(_join_uri(profile.train_uri, filename), archive)
        if not (data_root / extracted_dir).exists():
            with tarfile.open(archive) as tar:
                tar.extractall(data_root, filter="data")
    return data_root


def _clip_captions_from_coco(payload: Any) -> dict[str, str]:
    """Derive one deterministic retrieval caption per annotated COCO image."""
    categories = {
        item.get("id"): item.get("name")
        for item in payload.get("categories", [])
        if isinstance(item, dict) and item.get("id") is not None and item.get("name")
    }
    labels_by_image: dict[Any, set[str]] = {}
    for annotation in payload.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        label = categories.get(annotation.get("category_id"))
        if label:
            labels_by_image.setdefault(annotation.get("image_id"), set()).add(label)

    captions = {}
    for image in payload.get("images", []):
        if not isinstance(image, dict) or not image.get("file_name"):
            continue
        labels = sorted(labels_by_image.get(image.get("id"), ()))
        if labels:
            captions[image["file_name"]] = "a photo containing " + ", ".join(labels)
    return captions


def _prepare_clip_data_mount(profile: ModelProfile, out_dir: Path) -> Path:
    """Stage the real COCO inputs as the custom image-caption layout CLIP requires."""
    data_root = out_dir / "data_mount" / "clip"
    for split, uri in (("train", profile.train_uri), ("val", profile.eval_uri)):
        target = data_root / split
        image_dir = target / "images.tar.gz"
        caption_dir = target / "captions.tar.gz"
        annotation_path = target / "annotations.json"
        archive = target / "_archives" / "images.tar.gz"
        _download_s3_file(_join_uri(uri, "images.tar.gz"), archive)
        _download_s3_file(_join_uri(uri, "annotations.json"), annotation_path)

        if not image_dir.is_dir():
            extract_root = target / "_extracted"
            if extract_root.exists():
                shutil.rmtree(extract_root)
            extract_root.mkdir(parents=True)
            with tarfile.open(archive) as tar:
                tar.extractall(extract_root, filter="data")
            extracted_images = extract_root / "images"
            if not extracted_images.is_dir():
                raise FileNotFoundError(f"CLIP image archive did not contain images/: {archive}")
            image_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(extracted_images), str(image_dir))
            shutil.rmtree(extract_root)

        payload = json.loads(annotation_path.read_text())
        captions = _clip_captions_from_coco(payload)
        caption_dir.mkdir(parents=True, exist_ok=True)
        image_names = []
        for image_name, caption in captions.items():
            image_path = image_dir / image_name
            if not image_path.is_file():
                continue
            (caption_dir / Path(image_name).with_suffix(".txt")).write_text(caption + "\n")
            image_names.append(image_name)
        if not image_names:
            raise ValueError(f"No annotated CLIP image-caption pairs were staged from {uri}")
        (target / "image_list.txt").write_text("\n".join(sorted(image_names)) + "\n")
    return data_root


def _prepare_image_classification_mount(out_dir: Path) -> Path:
    """Stage a minimal real classification dataset from the configured S3 source.

    Algorithm roots are deliberately deleted before validation, so local bind
    mounts cannot rely on datasets left by an earlier algorithm.  Materialize
    one real image per class for train and validation directly from the TAO
    classification dataset using environment-only credentials.
    """
    import boto3

    data_root = out_dir.parent / "datasets" / "image-classification-mini"
    train_root = data_root / "train"
    val_root = data_root / "val"
    train_classes = train_root / "classes.txt"
    val_classes = val_root / "classes.txt"
    if train_classes.is_file() and val_classes.is_file():
        return data_root

    client = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("ACCESS_KEY"),
        aws_secret_access_key=(
            os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("SECRET_KEY")
        ),
        endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
    )
    bucket = "nvcf-storage-handling"
    staged_classes: dict[str, list[str]] = {}
    for source_split, target_split, image_dir_name in (
        ("images_train", train_root, "images_train"),
        ("images_test", val_root, "images_val"),
    ):
        prefix = f"data/classification_pyt/{source_split}/"
        response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
        class_prefixes = sorted(item["Prefix"] for item in response.get("CommonPrefixes", []))
        if not class_prefixes:
            raise RuntimeError(f"No classification classes found under s3://{bucket}/{prefix}")
        classes: list[str] = []
        for class_prefix in class_prefixes:
            class_name = class_prefix.rstrip("/").rsplit("/", 1)[-1]
            objects = client.list_objects_v2(Bucket=bucket, Prefix=class_prefix).get("Contents", [])
            candidates = sorted(
                item["Key"]
                for item in objects
                if item.get("Size", 0) > 0
                and item["Key"].lower().endswith((".jpg", ".jpeg", ".png"))
            )
            if not candidates:
                raise RuntimeError(
                    f"No classification image found under s3://{bucket}/{class_prefix}"
                )
            key = candidates[0]
            destination = target_split / image_dir_name / class_name / Path(key).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file() or destination.stat().st_size == 0:
                with destination.open("wb") as stream:
                    client.download_fileobj(bucket, key, stream)
            classes.append(class_name)
        staged_classes[str(target_split)] = classes

    train_names = staged_classes[str(train_root)]
    val_names = staged_classes[str(val_root)]
    if train_names != val_names:
        raise RuntimeError(
            "Classification train/evaluation class sets differ: "
            f"train={train_names}, val={val_names}"
        )
    class_text = "\n".join(train_names) + "\n"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)
    train_classes.write_text(class_text)
    val_classes.write_text(class_text)
    return data_root


def _prepare_deformable_detr_mount(out_dir: Path) -> Path:
    """Stage current-run detection inputs instead of depending on stale caches."""
    data_root = out_dir / "data_mount" / "deformable-detr"
    for split, dataset_name in (
        ("train", "tao_od_synthetic_subset_train_no_convert"),
        ("val", "tao_od_synthetic_subset_val_no_convert"),
    ):
        target = data_root / split
        source = f"{BUCKET_ROOT}/{dataset_name}"
        archive = target / "images.tar.gz"
        _download_s3_file(_join_uri(source, "images.tar.gz"), archive)
        _download_s3_file(_join_uri(source, "annotations.json"), target / "annotations.json")
        _download_s3_file(_join_uri(source, "label_map.txt"), target / "label_map.txt")
        if not (target / "images").is_dir():
            with tarfile.open(archive) as tar:
                tar.extractall(target, filter="data")
        if not (target / "images").is_dir():
            raise FileNotFoundError(
                f"Deformable-DETR image archive did not contain images/: {archive}"
            )
    return data_root


def _prepare_grounding_dino_mount(out_dir: Path) -> Path:
    """Stage real COCO inputs and derive the ODVG training contract."""
    data_root = out_dir / "data_mount" / "grounding-dino-mini"
    dataset_names = {
        "train": "tao_od_synthetic_subset_train_no_convert",
        "val": "tao_od_synthetic_subset_val_no_convert",
    }
    for split, dataset_name in dataset_names.items():
        target = data_root / split
        source = f"{BUCKET_ROOT}/{dataset_name}"
        archive = target / "images.tar.gz"
        annotations = target / "annotations.json"
        _download_s3_file(_join_uri(source, "images.tar.gz"), archive)
        _download_s3_file(_join_uri(source, "annotations.json"), annotations)
        if not (target / "images").is_dir():
            with tarfile.open(archive) as tar:
                tar.extractall(target, filter="data")
        if not (target / "images").is_dir():
            raise FileNotFoundError(
                f"Grounding-DINO image archive did not contain images/: {archive}"
            )

    val_annotations = data_root / "val" / "annotations.json"
    val_payload = json.loads(val_annotations.read_text())
    val_categories = [
        category for category in val_payload.get("categories", [])
        if isinstance(category, dict) and category.get("id") is not None
    ]
    category_id_map = {
        category["id"]: index
        for index, category in enumerate(sorted(val_categories, key=lambda item: item["id"]))
    }
    for category in val_categories:
        category["id"] = category_id_map[category["id"]]
    for annotation in val_payload.get("annotations", []):
        if annotation.get("category_id") in category_id_map:
            annotation["category_id"] = category_id_map[annotation["category_id"]]
    if sorted(category_id_map.values()) != list(range(len(category_id_map))):
        raise ValueError("Grounding-DINO validation categories are not contiguous")
    _write_json(val_annotations, val_payload)

    payload = json.loads((data_root / "train" / "annotations.json").read_text())
    label_map = {
        str(category["id"]): category["name"]
        for category in payload.get("categories", [])
        if category.get("id") is not None and category.get("name")
    }
    annotations_by_image: dict[Any, list[dict[str, Any]]] = {}
    for annotation in payload.get("annotations", []):
        bbox = annotation.get("bbox")
        label = annotation.get("category_id")
        if not isinstance(bbox, list) or len(bbox) != 4 or str(label) not in label_map:
            continue
        x, y, width, height = bbox
        if width <= 0 or height <= 0:
            continue
        annotations_by_image.setdefault(annotation.get("image_id"), []).append({
            "bbox": [x, y, x + width, y + height],
            "label": label,
            "category": label_map[str(label)],
        })

    records = []
    for image in payload.get("images", []):
        instances = annotations_by_image.get(image.get("id"), [])
        if image.get("file_name") and instances:
            records.append({
                "file_name": image["file_name"],
                "detection": {"instances": instances},
            })
    if not records or not label_map:
        raise ValueError("Grounding-DINO COCO source produced no ODVG records or labels")

    train_root = data_root / "train"
    (train_root / "annotations_odvg.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    _write_json(train_root / "annotations_odvg_labelmap.json", label_map)
    return data_root


def _prepare_mal_mount(out_dir: Path) -> Path:
    """Stage current-run COCO instance-segmentation data for MAL."""
    data_root = out_dir / "data_mount" / "mal-mini"
    for split, dataset_name in (
        ("train", "auto_label_train"),
        ("val", "auto_label_val"),
    ):
        target = data_root / split
        source = f"{BUCKET_ROOT}/{dataset_name}"
        archive = target / "images.tar.gz"
        _download_s3_file(_join_uri(source, "images.tar.gz"), archive)
        annotations_path = target / "annotations.json"
        _download_s3_file(_join_uri(source, "annotations.json"), annotations_path)
        annotations = json.loads(annotations_path.read_text())
        instances = annotations.get("annotations", [])
        if not instances or any(not item.get("segmentation") for item in instances):
            raise ValueError(
                "MAL AutoML validation requires non-empty segmentation ground truth "
                f"for a finite mIoU objective: {annotations_path}"
            )
        if not (target / "images").is_dir():
            with tarfile.open(archive) as tar:
                tar.extractall(target, filter="data")
        if not (target / "images").is_dir():
            raise FileNotFoundError(f"MAL image archive did not contain images/: {archive}")
    return data_root


def _prepare_mask_grounding_dino_mount(out_dir: Path) -> Path:
    """Stage the current run's real ODVG/COCO instance-segmentation data."""
    data_root = out_dir / "data_mount" / "mask-grounding-dino-mini"
    for split, dataset_name, required_files in (
        (
            "train",
            "segmentation_mask_grounding_dino_train",
            ("images.tar.gz", "annotations_odvg.jsonl", "annotations_odvg_labelmap.json"),
        ),
        (
            "val",
            "segmentation_mask_grounding_dino_val",
            ("images.tar.gz", "annotations.json"),
        ),
    ):
        target = data_root / split
        source = f"{BUCKET_ROOT}/{dataset_name}"
        for filename in required_files:
            _download_s3_file(_join_uri(source, filename), target / filename)
        archive = target / "images.tar.gz"
        if not (target / "images").is_dir():
            with tarfile.open(archive) as tar:
                tar.extractall(target, filter="data")
        if not (target / "images").is_dir():
            raise FileNotFoundError(
                f"Mask Grounding DINO image archive did not contain images/: {archive}"
            )
    return data_root


def _prepare_visual_changenet_backbone(out_dir: Path) -> Path:
    destination = out_dir / "ptm" / "c-radio-v2-b" / "C-RADIOv2_B.safetensors"
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    from huggingface_hub import hf_hub_download

    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(hf_hub_download(
        repo_id="nvidia/C-RADIOv2-B",
        filename="model.safetensors",
        token=os.environ.get("HF_TOKEN") or None,
    ))
    shutil.copy2(downloaded, destination)
    return destination


def _prepare_cosmos_model_mounts() -> tuple[Path, Path]:
    """Resolve the real gated Cosmos snapshot and its symlink target folder."""
    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": "nvidia/Cosmos-Reason2-8B",
        "token": os.environ.get("HF_TOKEN") or None,
    }
    try:
        snapshot = Path(snapshot_download(local_files_only=True, **kwargs))
    except FileNotFoundError:
        snapshot = Path(snapshot_download(**kwargs))
    blobs = snapshot.parents[1] / "blobs"
    if not (snapshot / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"Cosmos model index is missing from {snapshot}")
    if not blobs.is_dir():
        raise FileNotFoundError(f"Cosmos Hugging Face blob folder is missing from {blobs}")
    return snapshot, blobs


def _video_fps(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate", "-of", "default=nw=1:nk=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    fps = float(Fraction(value))
    if fps <= 0:
        raise ValueError(f"Invalid video frame rate {value!r} for {path}")
    return fps


def _video_codec(path: Path) -> str:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    codec = result.stdout.strip()
    if not codec:
        raise ValueError(f"Unable to determine video codec for {path}")
    return codec


def _ensure_cosmos_video_codec(path: Path) -> Path:
    """Transcode codecs absent from the Cosmos release image to bounded VP9."""
    if _video_codec(path) in {"vp8", "vp9", "av1", "mjpeg", "h264", "hevc"}:
        return path
    destination = path.with_suffix(".webm")
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(path), "-an",
            # The release image's torchvision backend decodes the entire clip
            # before sampling. Keep this validation subset bounded at the same
            # 2 FPS used by the evaluator and at a compact vision resolution.
            "-vf", "fps=2,scale=448:-2",
            "-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8",
            "-crf", "40", "-b:v", "0", str(destination),
        ],
        check=True,
    )
    path.unlink()
    return destination


def _stage_cosmos_split(
    annotation_path: Path,
    archive_path: Path,
    target: Path,
    limit: int = 2,
) -> int:
    """Extract referenced real videos and add measured FPS to a staged annotation."""
    payload = json.loads(annotation_path.read_text())
    if not isinstance(payload, list):
        raise TypeError(f"Cosmos validation annotations must be a list: {annotation_path}")
    records = [copy.deepcopy(record) for record in payload[:limit]]
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as tar:
        for record in records:
            relative_video = record.get("video")
            if not relative_video:
                raise ValueError(f"Cosmos record has no video path: {record.get('id')}")
            member = tar.getmember(f"videos/{relative_video}")
            source = tar.extractfile(member)
            if source is None:
                raise FileNotFoundError(f"Unable to extract {member.name} from {archive_path}")
            destination = target / relative_video
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            destination = _ensure_cosmos_video_codec(destination)
            record["video"] = destination.relative_to(target).as_posix()
            record["video_fps"] = _video_fps(destination)
    (target / "annotations.json").write_text(json.dumps(records, indent=2) + "\n")
    return len(records)


def _prepare_cosmos_data_mount(profile: ModelProfile, run_root: Path) -> Path:
    """Materialize a bounded real S3 subset required by Cosmos train/evaluate."""
    data_root = run_root / "datasets" / "cosmos-rl"
    cache_root = Path(
        os.environ.get("TAO_AUTOML_DATA_CACHE", str(Path.home() / "data"))
    )
    for split, uri in (("train", profile.train_uri), ("eval", profile.eval_uri)):
        source_name = uri.rstrip("/").rsplit("/", 1)[-1]
        source_root = cache_root / source_name
        annotation_path = source_root / "annotations.json"
        archive_path = source_root / "videos.tar.gz"
        _download_s3_file(_join_uri(uri, "annotations.json"), annotation_path)
        _download_s3_file(_join_uri(uri, "videos.tar.gz"), archive_path)
        target = data_root / split
        staged_annotation = target / "annotations.json"
        if not staged_annotation.is_file():
            _stage_cosmos_split(annotation_path, archive_path, target)
    return data_root


def _mounts_for_model(out_dir: Path, model: str, profile: ModelProfile) -> list[dict[str, str]]:
    mounts = [{"host_path": str(out_dir / "results"), "container_path": "/results"}]
    if model == "bevfusion":
        mounts.append({
            "host_path": str(_prepare_bevfusion_data_mount(profile, out_dir)),
            "container_path": "/data/bevfusion",
        })
    if model == "clip":
        mounts.append({
            "host_path": str(_prepare_clip_data_mount(MODEL_PROFILES[model], out_dir)),
            "container_path": "/data/clip",
            "read_only": True,
        })
    if model in {"depth-net-mono", "depth-net-stereo"}:
        mounts.append({
            "host_path": str(_prepare_depth_data_mount(MODEL_PROFILES[model], out_dir, model)),
            "container_path": "/data",
        })
    if model == "grounding-dino":
        dataset_root = _prepare_grounding_dino_mount(out_dir)
        mounts.append({
            "host_path": str(dataset_root),
            "container_path": "/data/grounding-dino-mini",
        })
    if model == "deformable-detr":
        dataset_root = _prepare_deformable_detr_mount(out_dir)
        mounts.extend([
            {
                # Local mounted inputs are not archive-materialized by the
                # runner. Bind the extracted folders at the paths produced by
                # the skill's images.tar.gz data-source contract.
                "host_path": str(dataset_root / "train" / "images"),
                "container_path": "/data/deformable-detr-mini/train/images.tar.gz",
                "read_only": True,
            },
            {
                "host_path": str(dataset_root / "val" / "images"),
                "container_path": "/data/deformable-detr-mini/val/images.tar.gz",
                "read_only": True,
            },
            *[
                {
                    "host_path": str(dataset_root / split / filename),
                    "container_path": f"/data/deformable-detr-mini/{split}/{filename}",
                    "read_only": True,
                }
                for split in ("train", "val")
                for filename in ("annotations.json", "label_map.txt")
            ],
        ])
    if model == "classification-pyt":
        dataset_root = _prepare_image_classification_mount(out_dir)
        mounts.append({
            "host_path": str(dataset_root),
            "container_path": "/data/image-classification-mini",
        })
    if model == "mae":
        dataset_root = _prepare_image_classification_mount(out_dir)
        mounts.append({
            "host_path": str(dataset_root),
            "container_path": "/data/image-classification-mini",
        })
    if model == "mal":
        dataset_root = _prepare_mal_mount(out_dir)
        mounts.append({
            "host_path": str(dataset_root),
            "container_path": "/data/mal-mini",
        })
    if model == "mask-grounding-dino":
        dataset_root = _prepare_mask_grounding_dino_mount(out_dir)
        mounts.append({
            "host_path": str(dataset_root),
            "container_path": "/data/mask-grounding-dino-mini",
        })
    if model == "mask2former":
        dataset_root = out_dir.parent / "datasets" / "mask2former-mini"
        mounts.append({
            "host_path": str(dataset_root),
            "container_path": "/data/mask2former-mini",
        })
    if model == "ml-recog":
        dataset_root = out_dir.parent / "datasets" / "ml-recog"
        mounts.append({
            "host_path": str(dataset_root),
            "container_path": "/data/ml-recog",
        })
    if model == "nvdinov2":
        dataset_root = out_dir.parent / "datasets" / "nvdinov2-mini"
        mounts.append({
            "host_path": str(dataset_root),
            "container_path": "/data/nvdinov2-mini",
        })
    if model == "nvpanoptix3d":
        dataset_root = out_dir.parent / "datasets" / "nvpanoptix3d"
        mounts.append({
            "host_path": str(dataset_root),
            "container_path": "/data/nvpanoptix3d",
        })
    if model == "ocdnet":
        dataset_root = out_dir.parent / "datasets" / "ocdnet"
        mounts.append({
            "host_path": str(dataset_root),
            "container_path": "/data/ocdnet",
        })
    if model == "ocrnet":
        dataset_root = out_dir.parent / "datasets" / "ocrnet"
        mounts.append({
            "host_path": str(dataset_root),
            "container_path": "/data/ocrnet",
        })
    if model == "oneformer":
        dataset_root = out_dir.parent / "datasets" / "oneformer"
        mounts.append({
            "host_path": str(dataset_root),
            "container_path": "/data/oneformer",
        })
    if model == "optical-inspection":
        dataset_root = out_dir.parent / "datasets" / "optical-inspection"
        mounts.append({
            "host_path": str(dataset_root),
            "container_path": "/data/optical-inspection",
        })
    if model == "rtdetr":
        dataset_root = out_dir.parent / "datasets" / "rtdetr"
        mounts.append({
            "host_path": str(dataset_root),
            "container_path": "/data/rtdetr",
        })
    if model == "segformer":
        dataset_root = out_dir.parent / "datasets" / "segformer" / "root"
        mounts.append({
            "host_path": str(dataset_root),
            "container_path": "/data/segformer",
        })
    if model == "pointpillars":
        dataset_root = out_dir.parent / "datasets" / "pointpillars"
        mounts.append({
            "host_path": str(dataset_root),
            "container_path": "/data/pointpillars",
        })
    if model == "visual-changenet":
        mounts.append({
            "host_path": str(_prepare_visual_changenet_backbone(out_dir)),
            "container_path": VISUAL_CHANGENET_BACKBONE_CONTAINER_PATH,
        })
    if model == "sparse4d":
        data_root = out_dir / "aicity_root"
        data_root.mkdir(parents=True, exist_ok=True)
        mounts.append({
            "host_path": str(data_root),
            "container_path": "/data/aicity_root",
        })
        source_root = os.environ.get("TAO_PYTORCH_SOURCE_ROOT")
        if source_root:
            train_entrypoint = (
                Path(source_root)
                / "nvidia_tao_pytorch/cv/sparse4d/scripts/train.py"
            )
            if not train_entrypoint.is_file():
                raise FileNotFoundError(
                    f"Sparse4D framework overlay is missing: {train_entrypoint}"
                )
            mounts.append({
                "host_path": str(train_entrypoint),
                "container_path": (
                    "/usr/local/lib/python3.12/dist-packages/"
                    "nvidia_tao_pytorch/cv/sparse4d/scripts/train.py"
                ),
                "read_only": True,
            })
    if model == "cosmos-rl":
        model_snapshot, model_blobs = _prepare_cosmos_model_mounts()
        mounts.extend([
            {
                "host_path": str(model_snapshot),
                "container_path": COSMOS_MODEL_CONTAINER_PATH,
                "read_only": True,
            },
            {
                # Snapshot weight symlinks use ../../blobs. Keep this mount
                # outside the read-only dataset tree so Docker can create both
                # mountpoints before the container starts.
                "host_path": str(model_blobs),
                "container_path": COSMOS_BLOBS_CONTAINER_PATH,
                "read_only": True,
            },
        ])
        dataset_root = out_dir.parent / "datasets" / "cosmos-rl"
        if dataset_root.exists():
            mounts.append({
                "host_path": str(dataset_root),
                "container_path": "/data/automl_datasets/cosmos-rl",
                "read_only": True,
            })
    return mounts


def _sample_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)][:8]
    if isinstance(payload, dict):
        for key in ("annotations", "data", "records", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)][:8]
        if payload:
            return [payload]
    return []


def _cosmos_video_fps_preflight(profile: ModelProfile) -> dict[str, Any]:
    checked = []
    missing = []
    for name, uri in {
        "train_annotation": _join_uri(profile.train_uri, "annotations.json"),
        "eval_annotation": _join_uri(profile.eval_uri, "annotations.json"),
    }.items():
        payload = _read_s3_json(uri)
        records = _sample_records(payload)
        has_video_fps = bool(records) and all("video_fps" in record for record in records)
        checked.append({
            "name": name,
            "uri": uri,
            "sampled_records": len(records),
            "has_video_fps": has_video_fps,
        })
        if not has_video_fps:
            missing.append(name)
    return {
        "status": "passed" if not missing else "failed",
        "checked": checked,
        "missing": missing,
    }


def run_model(args: argparse.Namespace) -> int:
    model_dir = _resolve_model_dir(args.skill_bank, args.model)
    model = _model_profile_key(model_dir)
    profile = MODEL_PROFILES[model]
    staged_data_root = args.run_root / "datasets" / model
    if model == "cosmos-rl":
        staged_data_root = _prepare_cosmos_data_mount(profile, args.run_root)
    if model == "cosmos-rl" and staged_data_root.exists():
        profile = replace(
            profile,
            train_uri="/data/automl_datasets/cosmos-rl/train",
            eval_uri="/data/automl_datasets/cosmos-rl/eval",
        )
    if model == "clip":
        # The available validation source is COCO detection data. The CLIP
        # skill explicitly permits a plumbing-only fallback that derives
        # same-basename captions from class labels when documented.
        profile = replace(
            profile,
            train_uri="/data/clip/train",
            eval_uri="/data/clip/val",
        )
    skill_text = (model_dir / "SKILL.md").read_text()
    if model == "dino":
        # DINO's compact skill delegates the mandatory per-action dataset
        # table to this reference. Parse both as the real skill workflow does.
        skill_text += "\n" + (model_dir / "references" / "dino-data-specs.md").read_text()
    skill_info = _read_yaml(model_dir / "references" / "skill_info.yaml")
    train_specs = _read_yaml(model_dir / "references" / "spec_template_train.yaml")
    if profile.data_format:
        skill_info["data_format"] = profile.data_format

    out_dir = args.run_root / model
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "result.json"
    if model == "depth-net-mono":
        # The available smoke dataset annotations contain left, right, and GT
        # fields. The mono skill explicitly requires a derived left+GT file.
        # The staged files are mounted at /data by _mounts_for_model().
        source_profile = MODEL_PROFILES[model]
        profile = replace(
            profile,
            train_uri=f"/data/{source_profile.train_uri.rstrip('/').rsplit('/', 1)[-1]}",
            eval_uri=f"/data/{source_profile.eval_uri.rstrip('/').rsplit('/', 1)[-1]}",
        )
    if model == "cosmos-rl" and staged_data_root.exists():
        staged_files = [path for path in staged_data_root.rglob("*") if path.is_file()]
        _write_json(out_dir / "evaluations" / "data_staging.json", {
            "source": {
                "train": MODEL_PROFILES[model].train_uri,
                "eval": MODEL_PROFILES[model].eval_uri,
            },
            "staged_path": str(staged_data_root),
            "container_path": "/data/automl_datasets/cosmos-rl",
            "file_count": len(staged_files),
            "bytes": sum(path.stat().st_size for path in staged_files),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    effective_num_classes = max(
        [value for value in (args.num_classes, profile.num_classes) if value is not None],
        default=None,
    )

    supported_params = _supported_automl_parameters(args.skill_bank, model_dir.name)
    if model == "depth-net-stereo" and supported_params:
        # The generated catalog includes non-train split augmentation fields
        # and model.volume_dim, which the current FoundationStereo train path
        # does not consume. Validate only effective train hyperparameters.
        supported_params = [
            param for param in supported_params
            if param in {"train.optim.lr", "train.optim.weight_decay"}
        ]
    if model == "classification-pyt" and supported_params:
        # Distillation teacher fields share the generated schema but are not
        # consumed by the ordinary classification_pyt train action.
        supported_params = [
            param for param in supported_params if not param.startswith("distill.")
        ]
    if model == "nvpanoptix3d" and supported_params:
        supported_params = [
            param for param in supported_params if param == "train.optim.lr"
        ]
    if model == "oneformer" and supported_params:
        # Keep the smoke search on effective optimizer parameters. The generated
        # catalog also exposes validation/test resize fields whose large sampled
        # values defeat the documented compact local workflow.
        supported_params = [
            param for param in supported_params
            if param in {"train.optim.lr", "train.optim.weight_decay"}
        ]
    if args.algorithm == "pbt" and supported_params:
        # PBT resumes checkpoints after exploit/explore.  Perturbing structural
        # model fields (for example decoder depth or hidden dimensions) makes
        # the source checkpoint incompatible with the resumed model.  Keep the
        # validation search on optimizer controls, which are checkpoint-safe
        # and still exercise PBT's copy/perturb/resume behavior.
        supported_params = _pbt_resume_safe_parameters(supported_params, model)
    if supported_params == [] and not args.post_check_only:
        payload = {
            "model": model,
            "algorithm": args.algorithm,
            "status": "blocked",
            "blocker": "skill is listed as AutoML-enabled but exposes no searchable AutoML train parameters",
        }
        _write_json(report_path, payload)
        print(json.dumps(payload))
        return 2

    if model == "cosmos-rl" and not args.post_check_only:
        if not os.environ.get("HF_TOKEN"):
            payload = {
                "model": model,
                "algorithm": args.algorithm,
                "status": "blocked",
                "blocker": "HF_TOKEN is required for the gated Cosmos-Reason2-8B model",
            }
            _write_json(report_path, payload)
            print(json.dumps(payload))
            return 2

    if args.post_check_only:
        if not report_path.exists():
            raise FileNotFoundError(f"{report_path} does not exist")
        payload = json.loads(report_path.read_text())
        sdk = DockerSDK(
            poll_interval=args.poll_interval,
            state_file=str(out_dir / "sdk_state_post_checks.json"),
        )
        payload = _run_post_checks(
            args=args,
            model_dir=model_dir,
            skill_text=skill_text,
            skill_info=skill_info,
            profile=profile,
            out_dir=out_dir,
            payload=payload,
            sdk=sdk,
            num_classes=effective_num_classes,
        )
        _write_json(report_path, payload)
        print(json.dumps({"model": model, "status": payload["status"], "report": str(report_path)}))
        return 0 if payload["status"] == "passed" else 1

    if profile.blocked and not args.allow_known_blockers:
        payload = {
            "model": model,
            "algorithm": args.algorithm,
            "status": "blocked",
            "blocker": profile.blocked,
        }
        _write_json(report_path, payload)
        print(json.dumps(payload))
        return 2

    schema_keys = _schema_keys(model_dir, "train")
    rows = _parse_train_rows(skill_text)
    overrides: dict[str, Any] = {}
    _add_data_source_overrides(overrides, profile, rows)
    if model == "grounding-dino":
        overrides.update({
            "dataset.train_data_sources[0].image_dir": f"{profile.train_uri}/images",
            "dataset.val_data_sources.image_dir": f"{profile.eval_uri}/images",
        })
    if model == "classification-pyt":
        overrides.update({
            "dataset.train_dataset.images_dir": f"{profile.train_uri}/images_train",
            "dataset.val_dataset.images_dir": f"{profile.eval_uri}/images_val",
            "dataset.classes_file": f"{profile.train_uri}/classes.txt",
        })
    if model == "mae":
        overrides.update({
            "dataset.train_data_sources": f"{profile.train_uri}/images_train",
            "dataset.val_data_sources": f"{profile.eval_uri}/images_val",
        })
    if model == "ml-recog":
        overrides.update({
            "dataset.train_dataset": "/data/ml-recog/known/train/train",
            "dataset.val_dataset": {
                "reference": "/data/ml-recog/known/reference/reference",
                "query": "/data/ml-recog/known/val/val",
            },
        })
    if model == "mal":
        overrides.update({
            "dataset.train_img_dir": f"{profile.train_uri}/images",
            "dataset.train_ann_path": f"{profile.train_uri}/annotations.json",
            "dataset.val_img_dir": f"{profile.eval_uri}/images",
            "dataset.val_ann_path": f"{profile.eval_uri}/annotations.json",
        })
    if model == "depth-net-stereo":
        overrides.update({
            "dataset.train_dataset.data_sources": [{
                "data_file": _join_uri(profile.train_uri, "annotations.txt"),
                "dataset_name": "Middlebury",
            }],
            "dataset.val_dataset.data_sources": [{
                "data_file": _join_uri(profile.eval_uri, "annotations.txt"),
                "dataset_name": "Middlebury",
            }],
        })
    if model == "cosmos-rl":
        overrides.update({
            "custom.train_dataset.annotation_path": _join_uri(profile.train_uri, "annotations.json"),
            "custom.train_dataset.media_path": profile.train_uri,
            "custom.val_dataset.annotation_path": _join_uri(profile.eval_uri, "annotations.json"),
            "custom.val_dataset.media_path": profile.eval_uri,
        })
    overrides.update(_minimal_train_overrides(train_specs, schema_keys, effective_num_classes, model))
    if model == "nvdinov2":
        overrides["dataset.train_dataset.images_dir"] = "/data/nvdinov2-mini/images_train/images_train"
    if model == "nvpanoptix3d":
        overrides.update({
            "dataset.frustum_mask_path": f"{profile.train_uri}/meta/frustum_mask.npz",
            "dataset.label_map": f"{profile.train_uri}/meta/colormap.json",
            "dataset.train.json_path": f"{profile.train_uri}/meta/train.json",
            "dataset.train.base_dir": profile.train_uri,
            "dataset.val.json_path": f"{profile.eval_uri}/meta/val.json",
            "dataset.val.base_dir": profile.eval_uri,
            "dataset.test.json_path": f"{profile.eval_uri}/meta/test.json",
            "dataset.test.base_dir": profile.eval_uri,
        })
    if model == "ocdnet":
        overrides.update({
            "dataset.train_dataset.data_path": [profile.train_uri],
            "dataset.validate_dataset.data_path": [profile.eval_uri],
        })
    if model == "oneformer":
        overrides.update({
            "model.sem_seg_head.num_classes": 133,
            "dataset.contiguous_id": True,
            "dataset.train.images": f"{profile.train_uri}/images",
            "dataset.train.annotations": f"{profile.train_uri}/annotations.json",
            "dataset.label_map": f"{profile.train_uri}/label_map.json",
            "dataset.train.panoptic": f"{profile.train_uri}/images_panoptic",
            "dataset.val.images": f"{profile.eval_uri}/images",
            "dataset.val.annotations": f"{profile.eval_uri}/annotations.json",
            "dataset.val.panoptic": f"{profile.eval_uri}/images_panoptic",
            "dataset.test.images": f"{profile.eval_uri}/images",
        })
    if model == "optical-inspection":
        overrides.update({
            "dataset.train_dataset.images_dir": f"{profile.train_uri}/images",
            "dataset.train_dataset.csv_path": f"{profile.train_uri}/dataset.csv",
            "dataset.validation_dataset.images_dir": f"{profile.eval_uri}/images",
            "dataset.validation_dataset.csv_path": f"{profile.eval_uri}/dataset.csv",
            "dataset.test_dataset.images_dir": f"{profile.eval_uri}/images",
            "dataset.test_dataset.csv_path": f"{profile.eval_uri}/dataset.csv",
        })
    if model == "rtdetr":
        overrides.update({
            "dataset.num_classes": 5,
            "dataset.eval_class_ids": [1, 2, 3, 4],
            "dataset.train_data_sources[0].image_dir": f"{profile.train_uri}/images",
            "dataset.train_data_sources[0].json_file": f"{profile.train_uri}/annotations.json",
            "dataset.val_data_sources.image_dir": f"{profile.eval_uri}/images",
            "dataset.val_data_sources.json_file": f"{profile.eval_uri}/annotations.json",
        })
    if model == "segformer":
        overrides.update({
            "dataset.segment.root_dir": profile.train_uri,
            "dataset.segment.num_classes": 2,
            "dataset.segment.batch_size": 1,
            "dataset.segment.workers": 0,
            "train.tensorboard.enabled": False,
        })
    if model == "mask-grounding-dino":
        overrides.update({
            "dataset.train_data_sources[0].image_dir": f"{profile.train_uri}/images",
            "dataset.train_data_sources[0].json_file": f"{profile.train_uri}/annotations_odvg.jsonl",
            "dataset.train_data_sources[0].label_map": f"{profile.train_uri}/annotations_odvg_labelmap.json",
            "dataset.val_data_sources.image_dir": f"{profile.eval_uri}/images",
            "dataset.val_data_sources.json_file": f"{profile.eval_uri}/annotations.json",
            "dataset.val_data_sources.data_type": "OD",
        })
    if model == "mask2former":
        overrides.update({
            "dataset.train.img_dir": f"{profile.train_uri}/images",
            "dataset.label_map": f"{profile.train_uri}/label_map_panoptic.json",
            "dataset.train.instance_json": f"{profile.train_uri}/annotations.json",
            "dataset.train.panoptic_json": f"{profile.train_uri}/annotations_panoptic.json",
            "dataset.train.panoptic_dir": f"{profile.train_uri}/images_panoptic",
            "dataset.val.img_dir": f"{profile.eval_uri}/images",
            "dataset.val.instance_json": f"{profile.eval_uri}/annotations.json",
            "dataset.val.panoptic_json": f"{profile.eval_uri}/annotations_panoptic.json",
            "dataset.val.panoptic_dir": f"{profile.eval_uri}/images_panoptic",
            "dataset.test.img_dir": f"{profile.eval_uri}/images",
        })
    overrides = _valid_set(overrides, train_specs, schema_keys)

    training_metric = _monitoring_metric(skill_text)
    metric = args.metric or _evaluation_metric(model, training_metric)
    checkpoint_evaluation_metric = _checkpoint_evaluation_metric(model, metric)
    selection_uses_training_kpi = checkpoint_evaluation_metric != metric or model == "nvdinov2"
    jobs: dict[int, dict[str, Any]] = {}
    job_runs: list[dict[str, Any]] = []

    def on_recommendation(rec) -> None:
        jobs.setdefault(rec.id, {})["specs"] = rec.specs
        LOG.info("model=%s rec=%s recommendation generated", model, rec.id)

    def on_result(rec, metric_value, status) -> None:
        run_data = {
            "rec_id": rec.id,
            "specs": copy.deepcopy(rec.specs),
            "job_id": getattr(rec, "job_id", None),
            "metric": metric_value,
            "status": status,
            "resume_from_job_id": getattr(rec, "resume_from_job_id", None),
            "resume_from_epoch": getattr(rec, "resume_from_epoch", None),
            "resume_from_step": getattr(rec, "resume_from_step", None),
            "resume_checkpoint_path": getattr(rec, "resume_checkpoint_path", None),
        }
        jobs[rec.id] = run_data
        job_runs.append(run_data)
        LOG.info("model=%s rec=%s status=%s metric=%s", model, rec.id, status, metric_value)

    sdk = DockerSDK(
        poll_interval=args.poll_interval,
        state_file=str(out_dir / "sdk_state.json"),
    )
    mounts = _mounts_for_model(out_dir, model, profile)
    dataset_convert_preflight = None
    if model in {"bevfusion", "ocrnet", "pointpillars", "sparse4d"}:
        dataset_convert_preflight = _run_dataset_convert_preflight(
            args=args,
            model_dir=model_dir,
            skill_text=skill_text,
            skill_info=skill_info,
            profile=profile,
            out_dir=out_dir,
            sdk=sdk,
            mounts=mounts,
        )
        if dataset_convert_preflight["status"] != "passed":
            payload = {
                "model": model,
                "algorithm": args.algorithm,
                "status": (
                    "blocked"
                    if "required converted artifacts are missing"
                    in dataset_convert_preflight.get("reason", "")
                    else "failed"
                ),
                "metric_documented": _monitoring_metric(skill_text),
                "metric_used_by_automl": metric,
                "train_dataset_uri": profile.train_uri,
                "eval_dataset_uri": profile.eval_uri,
                "dataset_convert": dataset_convert_preflight,
                "blocker": dataset_convert_preflight.get("reason"),
            }
            _write_json(report_path, payload)
            print(json.dumps(payload))
            return 2 if payload["status"] == "blocked" else 1
        overrides.update(dataset_convert_preflight.get("train_overrides") or {})
        overrides = _valid_set(overrides, train_specs, schema_keys)

    actions = skill_info.get("actions") or {}
    evaluate_cfg = actions.get("evaluate")
    evaluate_template = model_dir / "references" / "spec_template_evaluate.yaml"
    checkpoint_action_env = (
        {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}
        if model in {"clip", "ml-recog", "ocrnet", "oneformer", "optical-inspection", "re-identification"}
        else None
    )
    if (not evaluate_cfg or not evaluate_template.exists()) and model != "nvdinov2":
        payload = {
            "model": model,
            "algorithm": args.algorithm,
            "status": "blocked",
            "stage": "baseline_evaluation",
            "metric_documented": _monitoring_metric(skill_text),
            "metric_used_by_automl": metric,
            "blocker": "The model has no packaged runnable evaluate action/template for the required pre-AutoML baseline.",
        }
        _write_json(report_path, payload)
        print(json.dumps(payload))
        return 2

    baseline_train_result = None
    baseline_checkpoint_path = None
    if model != "cosmos-rl":
        baseline_train_specs = copy.deepcopy(train_specs)
        for dotted_key, value in overrides.items():
            _set_nested(baseline_train_specs, dotted_key, value)
        baseline_train_result = _run_action_job(
            sdk=sdk,
            image=_resolve_action_image(skill_info, actions["train"]),
            action_cfg=actions["train"],
            specs=baseline_train_specs,
            action="baseline_train",
            out_dir=out_dir,
            args=args,
            mounts=mounts,
            metric_name=training_metric,
        )
        baseline_checkpoints = _find_checkpoints(
            out_dir / "results" / baseline_train_result["job_id"], model
        )
        baseline_checkpoint_path = _prefer_epoch_or_step_checkpoint(
            baseline_checkpoints, model=model
        )
        if baseline_train_result["status"] != "success" or not baseline_checkpoint_path:
            payload = {
                "model": model,
                "algorithm": args.algorithm,
                "status": "blocked",
                "stage": "baseline_training",
                "metric_documented": training_metric,
                "metric_used_by_automl": metric,
                "baseline_training": baseline_train_result,
                "baseline_checkpoint_paths": baseline_checkpoints,
                "dataset_convert": dataset_convert_preflight,
                "blocker": (
                    "Scratch AutoML requires a successful minimal default train and a concrete "
                    "epoch/step checkpoint before the baseline evaluate action can run."
                ),
            }
            _write_json(report_path, payload)
            print(json.dumps({"model": model, "status": "blocked", "report": str(report_path)}))
            return 2

    baseline_checkpoint_container_path = (
        _host_to_container_path(baseline_checkpoint_path, out_dir / "results")
        if baseline_checkpoint_path
        else ""
    )
    if evaluate_cfg and evaluate_template.exists():
        baseline_specs = _build_action_specs(
            model_dir=model_dir,
            skill_text=skill_text,
            profile=profile,
            action="evaluate",
            checkpoint_container_path=baseline_checkpoint_container_path,
            num_classes=effective_num_classes,
            extra_overrides=(dataset_convert_preflight or {}).get("train_overrides") or {},
        )
        baseline_result = _run_action_job(
            sdk=sdk,
            image=_resolve_action_image(skill_info, evaluate_cfg),
            action_cfg=evaluate_cfg,
            specs=baseline_specs,
            action="baseline_evaluate",
            out_dir=out_dir,
            args=args,
            mounts=mounts,
            metric_name=checkpoint_evaluation_metric,
            env_vars=checkpoint_action_env,
        )
    else:
        baseline_result = {
            "action": "baseline_evaluate",
            "status": "skipped",
            "reason": "the packaged model skill has no evaluate action; selection uses the train KPI",
            "metric_name": metric,
            "metric_value": baseline_train_result.get("metric_value") if baseline_train_result else None,
        }
    if (
        baseline_result["status"] not in (
            {"success", "skipped"} if selection_uses_training_kpi else {"success"}
        )
        or (
            not selection_uses_training_kpi
            and baseline_result.get("metric_value") is None
        )
    ):
        payload = {
            "model": model,
            "algorithm": args.algorithm,
            "status": "blocked",
            "stage": "baseline_evaluation",
            "metric_documented": _monitoring_metric(skill_text),
            "metric_used_by_automl": metric,
            "baseline_training": baseline_train_result,
            "baseline_checkpoint_path": baseline_checkpoint_path,
            "baseline_evaluation": baseline_result,
            "blocker": (
                "The required baseline evaluate job failed or did not emit the selected AutoML metric; "
                "the skill workflow forbids falling back silently to a training-only proxy."
            ),
        }
        _write_json(report_path, payload)
        print(json.dumps({"model": model, "status": "blocked", "report": str(report_path)}))
        return 2

    baseline_selection_metric = (
        baseline_train_result.get("metric_value")
        if selection_uses_training_kpi and baseline_train_result
        else baseline_result["metric_value"]
    )
    if baseline_selection_metric is None:
        payload = {
            "model": model,
            "algorithm": args.algorithm,
            "status": "blocked",
            "stage": "baseline_training_metric",
            "metric_documented": training_metric,
            "metric_used_by_automl": metric,
            "metric_emitted_by_evaluate": checkpoint_evaluation_metric,
            "baseline_training": baseline_train_result,
            "baseline_evaluation": baseline_result,
            "blocker": "The baseline train job did not emit the documented AutoML selection metric.",
        }
        _write_json(report_path, payload)
        print(json.dumps({"model": model, "status": "blocked", "report": str(report_path)}))
        return 2

    settings = _automl_settings(args.algorithm, metric, model, args)
    settings.update({
        "baseline_metric": baseline_selection_metric,
        "baseline_training": baseline_train_result,
        "baseline_evaluation": baseline_result,
        "run_final_evaluation": True,
    })

    recommendation_eval_jobs: list[dict[str, Any]] = []
    final_eval_jobs: list[dict[str, Any]] = []

    def eval_fn(rec, train_job_id):
        checkpoint_paths = _find_checkpoints(out_dir / "results" / str(train_job_id), model)
        checkpoint_path = _prefer_epoch_or_step_checkpoint(checkpoint_paths, model=model)
        if not checkpoint_path:
            raise RuntimeError(f"No real checkpoint found for recommendation train job {train_job_id}")
        if not evaluate_cfg or not evaluate_template.exists():
            train_kpi = _latest_kpi(out_dir / "results" / str(train_job_id))
            selection_metric = _metric_from_job("", train_kpi, metric)
            recommendation_eval_jobs.append({
                "action": "recommendation_train_metric",
                "status": "success" if selection_metric is not None else "failed",
                "recommendation_id": getattr(rec, "id", None),
                "train_job_id": train_job_id,
                "checkpoint_path": checkpoint_path,
                "metric_name": metric,
                "metric_value": selection_metric,
                "reason": "no evaluate action is packaged; checkpoint usability is verified by inference",
            })
            return selection_metric
        specs = _build_action_specs(
            model_dir=model_dir,
            skill_text=skill_text,
            profile=profile,
            action="evaluate",
            checkpoint_container_path=_checkpoint_action_container_path(
                checkpoint_path, out_dir / "results", model
            ),
            num_classes=effective_num_classes,
            trial_specs=getattr(rec, "specs", None),
            extra_overrides=(dataset_convert_preflight or {}).get("train_overrides") or {},
        )
        evaluated = _run_action_job(
            sdk=sdk,
            image=_resolve_action_image(skill_info, evaluate_cfg),
            action_cfg=evaluate_cfg,
            specs=specs,
            action="recommendation_evaluate",
            out_dir=out_dir,
            args=args,
            mounts=mounts,
            metric_name=checkpoint_evaluation_metric,
            env_vars=checkpoint_action_env,
        )
        evaluated["recommendation_id"] = getattr(rec, "id", None)
        evaluated["train_job_id"] = train_job_id
        evaluated["checkpoint_path"] = checkpoint_path
        recommendation_eval_jobs.append(evaluated)
        if selection_uses_training_kpi:
            train_kpi = _latest_kpi(out_dir / "results" / str(train_job_id))
            selection_metric = _metric_from_job("", train_kpi, metric)
            evaluated["checkpoint_evaluation_metric"] = evaluated.get("metric_value")
            evaluated["selection_metric_from_training"] = selection_metric
            return selection_metric
        return evaluated.get("metric_value")

    def final_eval_fn(best_rec, train_job_id):
        checkpoint_paths = _find_checkpoints(out_dir / "results" / str(train_job_id), model)
        checkpoint_path = _prefer_epoch_or_step_checkpoint(checkpoint_paths, model=model)
        if not checkpoint_path:
            raise RuntimeError(f"No real checkpoint found for best train job {train_job_id}")
        if not evaluate_cfg or not evaluate_template.exists():
            train_kpi = _latest_kpi(out_dir / "results" / str(train_job_id))
            selection_metric = _metric_from_job("", train_kpi, metric)
            evaluated = {
                "action": "final_train_metric",
                "status": "success" if selection_metric is not None else "failed",
                "checkpoint_path": checkpoint_path,
                "metric_name": metric,
                "metric_value": selection_metric,
                "reason": "no evaluate action is packaged; checkpoint usability is verified by inference",
            }
            final_eval_jobs.append(evaluated)
            record_path = out_dir / "evaluations" / "best_automl.json"
            _write_json(record_path, evaluated)
            return {
                "metric_value": selection_metric,
                "record_path": str(record_path),
                "job_id": None,
                "status": evaluated["status"],
            }
        checkpoint_container_path = _checkpoint_action_container_path(
            checkpoint_path, out_dir / "results", model,
        )
        specs = _build_action_specs(
            model_dir=model_dir,
            skill_text=skill_text,
            profile=profile,
            action="evaluate",
            checkpoint_container_path=checkpoint_container_path,
            num_classes=effective_num_classes,
            trial_specs=getattr(best_rec, "specs", None),
            extra_overrides=(dataset_convert_preflight or {}).get("train_overrides") or {},
        )
        evaluated = _run_action_job(
            sdk=sdk,
            image=_resolve_action_image(skill_info, evaluate_cfg),
            action_cfg=evaluate_cfg,
            specs=specs,
            action="final_evaluate",
            out_dir=out_dir,
            args=args,
            mounts=mounts,
            metric_name=checkpoint_evaluation_metric,
            env_vars=checkpoint_action_env,
        )
        evaluated["checkpoint_path"] = checkpoint_path
        final_eval_jobs.append(evaluated)
        record_path = out_dir / "evaluations" / "best_automl.json"
        _write_json(record_path, evaluated)
        selection_metric = evaluated.get("metric_value")
        if selection_uses_training_kpi:
            train_kpi = _latest_kpi(out_dir / "results" / str(train_job_id))
            selection_metric = _metric_from_job("", train_kpi, metric)
            evaluated["checkpoint_evaluation_metric"] = evaluated.get("metric_value")
            evaluated["selection_metric_from_training"] = selection_metric
        return {
            "metric_value": selection_metric,
            "record_path": str(record_path),
            "job_id": evaluated.get("job_id"),
            "status": evaluated.get("status"),
        }

    runner = AutoMLRunner(
        sdk=sdk,
        skill_dir=model_dir,
        action="train",
        poll_interval=args.poll_interval,
    )
    run_error = None
    try:
        result = runner.run(
            train_dataset_uri=profile.train_uri,
            eval_dataset_uri=profile.eval_uri,
            image=_resolve_action_image(skill_info, actions["train"]),
            automl_settings=settings,
            automl_hyperparameters=supported_params,
            custom_param_ranges=_minimal_custom_ranges(
                supported_params,
                model=model,
                schema_path=(model_dir / "schemas" / "train.schema.json"),
            ),
            workspace_path=str(out_dir / "workspace"),
            spec_overrides=overrides,
            metric_extractor=_metric_extractor_for(model),
            eval_fn=eval_fn,
            final_eval_fn=final_eval_fn,
            on_recommendation=on_recommendation,
            on_result=on_result,
            gpu_count=args.num_gpus,
            gpu_device_ids=_gpu_device_ids(args),
            mounts=mounts,
        )
    except Exception as exc:
        run_error = str(exc)
        LOG.exception("AutoML run failed for model=%s algorithm=%s", model, args.algorithm)
        result = {
            "best": None,
            "progress": {},
            "history": [],
            "error": run_error,
        }

    for data in job_runs:
        job_id = data.get("job_id")
        job_root = out_dir / "results" / str(job_id) if job_id else Path("")
        data["checkpoint_paths"] = _find_checkpoints(job_root, model) if job_id else []
        data["checkpoint_count"] = len(data["checkpoint_paths"])

    best = result.get("best") or {}
    best_rec_id = best.get("rec_id")
    best_job = _select_best_job(job_runs, best, jobs)
    passed = (
        bool(job_runs)
        and all(data.get("status") == "success" for data in job_runs)
        and best.get("metric_value") is not None
        and bool(best_job.get("checkpoint_paths"))
    )
    payload = {
        "model": model,
        "algorithm": args.algorithm,
        "status": "passed" if passed else "failed",
        "metric_documented": _monitoring_metric(skill_text),
        "metric_documented_for_automl": _documented_automl_metric(skill_text),
        "metric_used_by_automl": metric,
        "metric_emitted_by_evaluate": checkpoint_evaluation_metric,
        "direction": _direction(metric, model),
        "train_dataset_uri": profile.train_uri,
        "eval_dataset_uri": profile.eval_uri,
        "spec_overrides": overrides,
        "result": result,
        "run_error": run_error,
        "jobs": jobs,
        "job_runs": job_runs,
        "best_checkpoint_paths": best_job.get("checkpoint_paths", []),
        "dataset_convert": dataset_convert_preflight,
        "baseline_training": baseline_train_result,
        "baseline_checkpoint_path": baseline_checkpoint_path,
        "baseline_evaluation": baseline_result,
        "recommendation_evaluation_jobs": recommendation_eval_jobs,
        "final_evaluation_jobs": final_eval_jobs,
        "resume_behavior": [
            {
                "rec_id": rid,
                "resume_from_job_id": data.get("resume_from_job_id"),
                "resume_from_epoch": data.get("resume_from_epoch"),
                "resume_from_step": data.get("resume_from_step"),
                "resume_checkpoint_path": data.get("resume_checkpoint_path"),
                "uses_latest": "latest" in str(data.get("resume_checkpoint_path", "")).lower(),
            }
            for rid, data in jobs.items()
            if data.get("resume_from_job_id")
        ],
    }
    if payload["status"] == "passed":
        payload = _run_post_checks(
            args=args,
            model_dir=model_dir,
            skill_text=skill_text,
            skill_info=skill_info,
            profile=profile,
            out_dir=out_dir,
            payload=payload,
            sdk=sdk,
            num_classes=effective_num_classes,
        )
    _write_json(report_path, payload)
    print(json.dumps({"model": model, "status": payload["status"], "report": str(report_path)}))
    return 0 if payload["status"] == "passed" else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--skill-bank", default="/localhome/local-rarunachalam/tao-skills-external", type=Path)
    parser.add_argument("--gpu-device-id", default="2")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--metric")
    parser.add_argument("--allow-known-blockers", action="store_true")
    parser.add_argument("--post-check-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        return run_model(parse_args(argv))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        LOG.exception("validation failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
