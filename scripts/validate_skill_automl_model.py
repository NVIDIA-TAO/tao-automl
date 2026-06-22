#!/usr/bin/env python3
"""Run one TAO model through the skill-based AutoMLRunner workflow.

This is intentionally a single-model runner. Algorithm-level orchestration is
done outside this script so each algorithm run folder can be deleted and
reported independently.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
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
        blocked="dataset_convert must be run before train; not yet implemented in this validator",
    ),
    "centerpose": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_centerpose_train",
        f"{BUCKET_ROOT}/purpose_built_models_centerpose_val",
    ),
    "classification-pyt": ModelProfile(
        f"{BUCKET_ROOT}/classification_train",
        f"{BUCKET_ROOT}/classification_val",
        num_classes=20,
    ),
    "clip": ModelProfile(f"{BUCKET_ROOT}/auto_label_train", f"{BUCKET_ROOT}/auto_label_val"),
    "cosmos-rl": ModelProfile(
        f"{BUCKET_ROOT}/cosmos_rl_its_subset",
        f"{BUCKET_ROOT}/cosmos_rl_its_eval",
        data_format="llava",
    ),
    "deformable-detr": ModelProfile(
        f"{BUCKET_ROOT}/object_detection_pyt_train",
        f"{BUCKET_ROOT}/object_detection_pyt_val",
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
        f"{BUCKET_ROOT}/classification_train",
        f"{BUCKET_ROOT}/classification_val",
        num_classes=20,
    ),
    "mal": ModelProfile(f"{BUCKET_ROOT}/auto_label_train", f"{BUCKET_ROOT}/auto_label_val"),
    "mask-grounding-dino": ModelProfile(
        f"{BUCKET_ROOT}/segmentation_mask_grounding_dino_train",
        f"{BUCKET_ROOT}/segmentation_mask_grounding_dino_val",
        num_classes=6,
        captions=("person", "bicycle", "car"),
    ),
    "mask2former": ModelProfile(
        f"{BUCKET_ROOT}/segmentation_mask2former_train",
        f"{BUCKET_ROOT}/segmentation_mask2former_val",
        num_classes=201,
    ),
    "ml-recog": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_ml_recog_train",
        f"{BUCKET_ROOT}/purpose_built_models_ml_recog_train",
    ),
    "nvdinov2": ModelProfile(
        "/data/nvdinov2-mini",
        f"{BUCKET_ROOT}/nvdinov2_val_cats_dogs",
    ),
    "nvpanoptix3d": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_nvpanoptix3d_train",
        f"{BUCKET_ROOT}/purpose_built_models_nvpanoptix3d_val",
        blocked="requires converted 3D assets/checkpoints not yet mapped by this validator",
    ),
    "ocdnet": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_ocdnet_train",
        f"{BUCKET_ROOT}/purpose_built_models_ocdnet_val",
    ),
    "ocrnet": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_ocrnet_train",
        f"{BUCKET_ROOT}/purpose_built_models_ocrnet_val",
    ),
    "oneformer": ModelProfile(
        f"{BUCKET_ROOT}/segmentation_oneformer_train",
        f"{BUCKET_ROOT}/segmentation_oneformer_val",
        num_classes=133,
    ),
    "optical-inspection": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_optical_inspection_train",
        f"{BUCKET_ROOT}/purpose_built_models_optical_inspection_val",
    ),
    "pointpillars": ModelProfile(
        f"{BUCKET_ROOT}/purpose_built_models_pointpillars_train",
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
        f"{BUCKET_ROOT}/object_detection_pyt_train",
        f"{BUCKET_ROOT}/object_detection_pyt_val",
        num_classes=6,
    ),
    "segformer": ModelProfile(
        f"{BUCKET_ROOT}/segmentation_segformer_train",
        f"{BUCKET_ROOT}/segmentation_segformer_val",
        num_classes=6,
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
    match = re.search(r"\*\*Monitoring metric:\*\*\s*([^\n]+)", skill_text)
    if not match:
        return "loss"
    metric = match.group(1).strip()
    return metric.split(",", 1)[0].strip()


def _direction(metric: str) -> str:
    return "minimize" if "loss" in metric.lower() else "maximize"


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
            "policy.model_name_or_path": "hf_model://nvidia/Cosmos-Reason2-8B",
            "policy.parallelism.dp_shard_size": 1,
            "policy.parallelism.dp_replicate_size": 1,
            "train.train_batch_per_replica": 1,
            "train.train_policy.mini_batch": 1,
            "train.train_policy.dataset.test_size": 0,
            "validation.batch_size": 1,
            "validation.enable_dataset_cache": False,
            "logging.logger": ["console", "tao"],
        })
    if model == "mae":
        candidates.update({
            "dataset.batch_size": 2,
            "train.stage": "finetune",
        })
    if model == "optical-inspection":
        candidates["dataset.batch_size"] = 2
    if model == "re-identification":
        candidates["dataset.batch_size"] = 16
        candidates["dataset.num_instances"] = 4
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


def _automl_settings(algorithm: str, metric: str, args: argparse.Namespace) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "algorithm": algorithm,
        "metric": metric,
        "direction": _direction(metric),
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
        if model == "nvdinov2" and name.startswith("student_epoch_"):
            exact.insert(0, path)
            continue
        if re.search(r"(?:^|[_-])(epoch|step)[_-]?\d+", name) or re.search(r"/(?:epoch|step)_\d+", path):
            exact.append(path)
        else:
            fallback.append(path)
    return (exact or fallback or [None])[0]


def _host_to_container_path(host_path: str, host_root: Path, container_root: str = "/results") -> str:
    path = Path(host_path)
    return f"{container_root.rstrip('/')}/{path.relative_to(host_root).as_posix()}"


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
    specs = _read_yaml(model_dir / "references" / f"spec_template_{action}.yaml")
    schema_keys = _schema_keys(model_dir, action)
    overrides: dict[str, Any] = {}
    _add_data_source_overrides(overrides, profile, _parse_action_rows(skill_text, action))
    overrides.update(_minimal_action_overrides(specs, schema_keys, action, num_classes))
    if profile.model_type and "model.model_type" in (schema_keys | _flatten_keys(specs)):
        overrides["model.model_type"] = profile.model_type
    if profile.dataset_name and "dataset.dataset_name" in (schema_keys | _flatten_keys(specs)):
        overrides["dataset.dataset_name"] = profile.dataset_name
    if model_dir.name == "action-recognition" and "dataset.label_map" in (schema_keys | _flatten_keys(specs)):
        overrides["dataset.label_map"] = {"catch": 0, "smile": 1}
    if model_dir.name == "mask-grounding-dino":
        overrides.update({
            "dataset.val_data_sources.data_type": "OD",
            "dataset.test_data_sources.data_type": "OD",
            "dataset.infer_data_sources.data_type": "OD",
        })
    if model_dir.name == "mask2former":
        overrides.update({
            "dataset.train.type": "coco_panoptic",
            "dataset.val.type": "coco_panoptic",
            "dataset.test.type": "coco_panoptic",
            "dataset.contiguous_id": False,
        })
    if model_dir.name == "pose-classification" and action == "inference":
        overrides["inference.output_file"] = "/results/pose_classification_inference.txt"
    if model_dir.name == "re-identification":
        if action == "evaluate":
            overrides["evaluate.output_cmc_curve_plot"] = "/results/reid_cmc_curve.png"
            overrides["evaluate.output_sampled_matches_plot"] = "/results/reid_sampled_matches.png"
        if action == "inference":
            overrides["inference.output_file"] = "/results/reid_inference.json"
    if model_dir.name == "visual-changenet":
        overrides["model.backbone.pretrained_backbone_path"] = VISUAL_CHANGENET_BACKBONE_CONTAINER_PATH
    if model_dir.name == "nvdinov2":
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
            overrides["dataset.test_dataset.images_dir"] = "/data/nvdinov2-mini/images_train"
    if trial_specs:
        overrides.update(_valid_set(trial_specs, specs, schema_keys))
    if extra_overrides:
        overrides.update(_valid_set(extra_overrides, specs, schema_keys))
    if model_dir.name == "mae" and action in {"evaluate", "inference"}:
        overrides["train.stage"] = "finetune"
    for key in (f"{action}.checkpoint", f"{action}.model_path", f"{action}.pretrained_model_path"):
        if key in (schema_keys | _flatten_keys(specs)):
            overrides[key] = checkpoint_container_path
            break
    overrides = _valid_set(overrides, specs, schema_keys)
    if model_dir.name == "mask-grounding-dino" and action == "inference":
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
        gpu_device_ids=(
            [args.gpu_device_id]
            if (gpu_count is None or gpu_count != 0) and args.gpu_device_id
            else None
        ),
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
    return {
        "action": action,
        "status": "success" if success else "failed",
        "job_id": job.id,
        "docker_status": status.status,
        "checkpoint_spec": _get_nested(specs, f"{action}.checkpoint"),
        "kpi": _latest_kpi(job_root),
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
    if model_dir.name == "sparse4d":
        overrides.update({
            "aicity.num_frames": 3,
            "aicity.anchor_init_config.num_anchor": 72,
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
    actions = skill_info.get("actions") or {}
    action_cfg = actions.get("dataset_convert")
    template = model_dir / "references" / "spec_template_dataset_convert.yaml"
    if not action_cfg or not template.exists():
        return {
            "status": "failed",
            "reason": "dataset_convert action/template is not packaged by skill",
        }

    specs = _build_dataset_convert_specs(
        model_dir=model_dir,
        skill_text=skill_text,
        profile=profile,
    )
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
        gpu_count=args.num_gpus if model_dir.name in {"pointpillars", "sparse4d"} else 0,
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

    if model_dir.name == "pointpillars":
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

    elif model_dir.name == "sparse4d":
        extra["depth_path_normalization"] = _normalize_sparse4d_depth_paths(
            model_dir=model_dir,
            out_dir=out_dir,
            convert_root=convert_root,
        )
        anchor = next(iter(sorted(convert_root.rglob("anchor_init.npy"))), None)
        train_ann = sorted(convert_root.rglob("*_infos_train.pkl"))
        val_ann = sorted(convert_root.rglob("*_infos_val.pkl"))
        test_ann = sorted(convert_root.rglob("*_infos_test.pkl"))
        required_sparse = {
            "anchor": anchor,
            "train_ann": train_ann[0] if train_ann else None,
            "val_ann": val_ann[0] if val_ann else None,
            "test_ann": test_ann[0] if test_ann else None,
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
    checkpoints = payload.get("best_checkpoint_paths") or []
    checkpoint_path = _prefer_epoch_or_step_checkpoint(checkpoints, model=model_dir.name)
    if not checkpoint_path:
        payload["checkpoint_validation"] = {
            "status": "failed",
            "reason": "no real checkpoint path found for best recommendation",
        }
        return payload

    host_root = out_dir / "results"
    checkpoint_container_path = _host_to_container_path(checkpoint_path, host_root)
    actions = skill_info.get("actions") or {}
    post_checks = []
    best_trial_specs = ((payload.get("result") or {}).get("best") or {}).get("specs") or {}
    dataset_convert_overrides = (
        (payload.get("dataset_convert") or {}).get("train_overrides") or {}
    )
    mounts = _mounts_for_model(out_dir, model_dir.name, profile)
    action_env_vars = (
        {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}
        if model_dir.name in {"ml-recog", "oneformer", "re-identification"}
        else None
    )
    for action in ("evaluate", "inference"):
        action_cfg = actions.get(action)
        template = model_dir / "references" / f"spec_template_{action}.yaml"
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


def _supported_automl_parameters(skill_bank: Path, model: str) -> list[str] | None:
    schema_path = skill_bank / "models" / model / "schemas" / "train.schema.json"

    def schema_defaults() -> list[str] | None:
        if not schema_path.exists():
            return None
        try:
            params = json.loads(schema_path.read_text()).get("automl_default_parameters")
        except json.JSONDecodeError:
            return None
        return params if isinstance(params, list) else None

    support_path = skill_bank / "models" / "automl_support.json"
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


def _minimal_custom_ranges(
    params: list[str] | None,
    model: str | None = None,
) -> dict[str, dict[str, Any]] | None:
    ranges: dict[str, dict[str, Any]] = {}
    for param in params or []:
        lower = param.lower()
        if "epoch" in lower:
            ranges[param] = {"valid_min": 1, "valid_max": 1}
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


def _prepare_depth_data_mount(profile: ModelProfile, out_dir: Path) -> Path:
    data_root = out_dir / "data_mount"
    for uri in (profile.train_uri, profile.eval_uri):
        dataset_name = uri.rstrip("/").rsplit("/", 1)[-1]
        target = data_root / dataset_name
        archive = target / "images.tar.gz"
        _download_s3_file(_join_uri(uri, "images.tar.gz"), archive)
        if not (target / "left").exists():
            with tarfile.open(archive) as tar:
                tar.extractall(target)
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


def _mounts_for_model(out_dir: Path, model: str, profile: ModelProfile) -> list[dict[str, str]]:
    mounts = [{"host_path": str(out_dir / "results"), "container_path": "/results"}]
    if model in {"depth-net-mono", "depth-net-stereo"}:
        mounts.append({
            "host_path": str(_prepare_depth_data_mount(profile, out_dir)),
            "container_path": "/data",
        })
    if model == "grounding-dino":
        dataset_root = out_dir.parent.parent / "datasets" / "grounding-dino-mini"
        mounts.append({
            "host_path": str(dataset_root),
            "container_path": "/data/grounding-dino-mini",
        })
    if model == "nvdinov2":
        dataset_root = out_dir.parent.parent / "datasets" / "nvdinov2-mini"
        mounts.append({
            "host_path": str(dataset_root),
            "container_path": "/data/nvdinov2-mini",
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
    model = args.model
    profile = MODEL_PROFILES.get(model)
    if profile is None:
        raise KeyError(f"No validation profile for {model}")

    model_dir = args.skill_bank / "models" / model
    skill_text = (model_dir / "SKILL.md").read_text()
    skill_info = _read_yaml(model_dir / "references" / "skill_info.yaml")
    train_specs = _read_yaml(model_dir / "references" / "spec_template_train.yaml")
    if profile.data_format:
        skill_info["data_format"] = profile.data_format

    out_dir = args.run_root / model
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "result.json"

    effective_num_classes = max(
        [value for value in (args.num_classes, profile.num_classes) if value is not None],
        default=None,
    )

    supported_params = _supported_automl_parameters(args.skill_bank, model)
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
        preflight = _cosmos_video_fps_preflight(profile)
        if preflight["status"] != "passed":
            payload = {
                "model": model,
                "algorithm": args.algorithm,
                "status": "blocked",
                "blocker": "Cosmos-RL annotations are missing video_fps in sampled records; SFT loader fails before checkpoint creation",
                "preflight": preflight,
                "attempted_training_evidence": (
                    "A real train attempt reached the Cosmos-RL process and failed with "
                    "Error processing sample: 'video_fps'."
                ),
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
    overrides.update(_minimal_train_overrides(train_specs, schema_keys, effective_num_classes, model))
    if model == "nvdinov2":
        overrides["dataset.train_dataset.images_dir"] = "/data/nvdinov2-mini/images_train"
    if model == "mask-grounding-dino":
        overrides["dataset.val_data_sources.data_type"] = "OD"
    overrides = _valid_set(overrides, train_specs, schema_keys)

    metric = args.metric or _monitoring_metric(skill_text)
    if model == "dino" and metric == "val_mAP50":
        metric = "mAP50"

    jobs: dict[int, dict[str, Any]] = {}

    def on_recommendation(rec) -> None:
        jobs.setdefault(rec.id, {})["specs"] = rec.specs
        LOG.info("model=%s rec=%s recommendation generated", model, rec.id)

    def on_result(rec, metric_value, status) -> None:
        jobs.setdefault(rec.id, {}).update({
            "job_id": getattr(rec, "job_id", None),
            "metric": metric_value,
            "status": status,
            "resume_from_job_id": getattr(rec, "resume_from_job_id", None),
            "resume_from_epoch": getattr(rec, "resume_from_epoch", None),
            "resume_from_step": getattr(rec, "resume_from_step", None),
            "resume_checkpoint_path": getattr(rec, "resume_checkpoint_path", None),
        })
        LOG.info("model=%s rec=%s status=%s metric=%s", model, rec.id, status, metric_value)

    sdk = DockerSDK(
        poll_interval=args.poll_interval,
        state_file=str(out_dir / "sdk_state.json"),
    )
    mounts = _mounts_for_model(out_dir, model, profile)
    dataset_convert_preflight = None
    if model in {"pointpillars", "sparse4d"}:
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
            automl_settings=_automl_settings(args.algorithm, metric, args),
            automl_hyperparameters=None,
            custom_param_ranges=_minimal_custom_ranges(supported_params, model=model),
            workspace_path=str(out_dir / "workspace"),
            spec_overrides=overrides,
            metric_extractor=_metric_extractor_for(model),
            on_recommendation=on_recommendation,
            on_result=on_result,
            gpu_count=args.num_gpus,
            gpu_device_ids=[args.gpu_device_id] if args.gpu_device_id else None,
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

    for rec_id, data in jobs.items():
        job_id = data.get("job_id")
        job_root = out_dir / "results" / str(job_id) if job_id else Path("")
        data["checkpoint_paths"] = _find_checkpoints(job_root, model) if job_id else []
        data["checkpoint_count"] = len(data["checkpoint_paths"])

    best = result.get("best") or {}
    best_rec_id = best.get("rec_id")
    best_job = jobs.get(best_rec_id, {})
    passed = (
        bool(jobs)
        and all(data.get("status") == "success" for data in jobs.values())
        and best.get("metric_value") is not None
        and bool(best_job.get("checkpoint_paths"))
    )
    payload = {
        "model": model,
        "algorithm": args.algorithm,
        "status": "passed" if passed else "failed",
        "metric_documented": _monitoring_metric(skill_text),
        "metric_used_by_automl": metric,
        "direction": _direction(metric),
        "train_dataset_uri": profile.train_uri,
        "eval_dataset_uri": profile.eval_uri,
        "spec_overrides": overrides,
        "result": result,
        "run_error": run_error,
        "jobs": jobs,
        "best_checkpoint_paths": best_job.get("checkpoint_paths", []),
        "dataset_convert": dataset_convert_preflight,
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
