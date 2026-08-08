# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preregistered, callback-executed local preflight for TAO DINO.

This module is deliberately an experiment adapter, not another orchestration
framework.  It translates verified VOC2007 and PTM runtime evidence into the
production :mod:`tao_automl.model_preflight` contract.  Physical TAO execution
is delegated to a caller-supplied executor so importing, planning, and testing
this module never downloads data, starts a container, reserves a GPU, or
submits a scheduler job.

The live :class:`~tao_automl.ptm_runtime.ResolvedPTMRuntimeInventory` is
required.  A serialized PTM report is insufficient because it omits the
validated checkpoint-spec document used to construct each effective spec.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import yaml

from tao_automl.latency_benchmark import (
    LatencyBenchmarkContract,
    combine_replica_records,
)
from tao_automl.model_preflight import (
    MODEL_PREFLIGHT_STAGES,
    ModelPreflightInputs,
    ModelPreflightStepRequest,
    ModelPreflightStepResult,
    PreflightPTMIdentity,
    canonical_sha256,
    run_model_preflight,
)
from tao_automl.ptm_preflight import PTMPreflightReport
from tao_automl.ptm_registry import (
    PTMCompatibilityResult,
    merge_ptm_spec_precedence,
    sha256_file,
)
from tao_automl.ptm_runtime import ResolvedPTMRuntimeInventory


DINO_PREFLIGHT_PLAN_SCHEMA_VERSION = 1
DINO_MODEL_ID = "dino"
DINO_TASK = "object_detection"
DINO_METRIC = "val_mAP50"
VOC_DATASET_ID = "pascal_voc_2007_full_detection"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_DIGEST_IMAGE_RE = re.compile(
    r"^(?P<repository>[^@:\s]+(?::\d+)?(?:/[^@:\s]+)*)"
    r"@sha256:(?P<digest>[0-9a-f]{64})$"
)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]*$")
_SAFE_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,79}$")
_PHYSICAL_STAGES = tuple(
    stage for stage in MODEL_PREFLIGHT_STAGES if stage != "metric_sanity"
)
_SKILL_FILES = {
    "skill_info": "references/skill_info.yaml",
    "train_template": "references/spec_template_train.yaml",
    "evaluate_template": "references/spec_template_evaluate.yaml",
    "inference_template": "references/spec_template_inference.yaml",
    "train_schema": "schemas/train.schema.json",
    "evaluate_schema": "schemas/evaluate.schema.json",
    "inference_schema": "schemas/inference.schema.json",
}
_TRAIN_INPUT_KEYS = frozenset(
    {
        "dataset.train_data_sources[0].image_dir",
        "dataset.train_data_sources[0].json_file",
        "dataset.val_data_sources[0].image_dir",
        "dataset.val_data_sources[0].json_file",
    }
)
_EVALUATE_INPUT_KEYS = frozenset(
    {
        "evaluate.checkpoint",
        "dataset.test_data_sources.image_dir",
        "dataset.test_data_sources.json_file",
    }
)
_INFERENCE_INPUT_KEYS = frozenset(
    {
        "inference.checkpoint",
        "dataset.infer_data_sources.image_dir",
        "dataset.infer_data_sources.classmap",
    }
)
_EXPECTED_ACTION_COMMANDS = {
    "train": "dino train -e {config_path}",
    "evaluate": "dino evaluate -e {config_path}",
    "inference": "dino inference -e {config_path}",
}
_DINO_CHECKPOINT_TARGETS = frozenset(
    {
        "train.pretrained_model_path",
        "model.pretrained_backbone_path",
    }
)
_OUTPUT_CONTRACT = {
    "required_input_artifact_ids": [
        "voc_label_map",
        "voc_inference_subset",
    ],
    "required_artifact_ids": [
        "full_epoch_checkpoint",
        "in_epoch_validation_metrics",
        "standalone_evaluation_metrics",
        "latency_aggregate",
        "resume_replay_state",
    ],
    "selection_isolation_required": True,
}
_ARTIFACT_CHECKPOINT_TOKEN = (
    "artifact://default_model_full_epoch/final_checkpoint"
)
_ARTIFACT_CLASSMAP_TOKEN = "artifact://voc2007/label_map.txt"
_ARTIFACT_INFERENCE_SUBSET_TOKEN = (
    "artifact://voc2007/inference_subset"
)
_INFERENCE_SUBSET_SIZE = 8


class DINOPreflightContractError(ValueError):
    """Raised before execution when a frozen DINO contract is incomplete."""


def _canonical_json(value: Any) -> Any:
    def thaw(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): thaw(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [thaw(child) for child in item]
        return copy.deepcopy(item)

    try:
        return json.loads(
            json.dumps(
                thaw(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise DINOPreflightContractError(
            "value must be finite canonical JSON"
        ) from exc


def _freeze_json(value: Any) -> Any:
    value = _canonical_json(value)
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return copy.deepcopy(value)


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DINOPreflightContractError(
            f"{name} must be lowercase SHA-256 hex"
        )
    return value


def _require_identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise DINOPreflightContractError(
            f"{name} must be a portable non-empty identifier"
        )
    return value


def _require_absolute_path(value: Path | str, name: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise DINOPreflightContractError(f"{name} must be an absolute path")
    return path


def _image_repository(image: str) -> str:
    """Strip an optional tag without confusing a registry port."""
    last_slash = image.rfind("/")
    last_colon = image.rfind(":")
    return image[:last_colon] if last_colon > last_slash else image


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DINOPreflightContractError(
            f"{name} must be a regular non-symlink JSON file"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DINOPreflightContractError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise DINOPreflightContractError(f"{name} must contain an object")
    return value


def _read_yaml_mapping(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DINOPreflightContractError(
            f"{name} must be a regular non-symlink YAML file"
        )
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise DINOPreflightContractError(f"{name} is not valid YAML") from exc
    if not isinstance(value, dict):
        raise DINOPreflightContractError(f"{name} must contain an object")
    return value


def _file_hash(path: Path, name: str) -> str:
    try:
        return sha256_file(path)
    except OSError as exc:
        raise DINOPreflightContractError(f"{name} could not be hashed") from exc


def _get_dotted_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for token in path.split("."):
        if not isinstance(current, Mapping) or token not in current:
            return None
        current = current[token]
    return current


def _smoke_model_token(checkpoint_id: str) -> str:
    return f"runtime://eligible_ptm_smoke/{checkpoint_id}/initialized_model"


@dataclass(frozen=True, slots=True)
class VOCRealDataIntegrityEvidence:
    """Typed result of the full VOC2007 real-data integrity verifier."""

    dataset_id: str
    dataset_root: Path
    manifest_path: Path
    integrity_path: Path
    image_root: Path
    train_annotation_path: Path
    validation_annotation_path: Path
    manifest_sha256: str
    integrity_sha256: str
    source_tree_sha256: str
    train_annotation_sha256: str
    validation_annotation_sha256: str
    annotation_contract_sha256: str
    validation_report_sha256: str
    train_samples: int
    validation_samples: int
    categories: tuple[tuple[int, str], ...]
    inference_subset: tuple[tuple[int, str], ...]
    invariants: Mapping[str, bool] = field(repr=False)
    status: str = "valid"
    dataset_terms_acknowledged: bool = True

    def __post_init__(self) -> None:
        if self.dataset_id != VOC_DATASET_ID:
            raise DINOPreflightContractError(
                f"dataset_id must be {VOC_DATASET_ID!r}"
            )
        for name in (
            "dataset_root",
            "manifest_path",
            "integrity_path",
            "image_root",
            "train_annotation_path",
            "validation_annotation_path",
        ):
            object.__setattr__(
                self,
                name,
                _require_absolute_path(getattr(self, name), name),
            )
        for name in (
            "manifest_sha256",
            "integrity_sha256",
            "source_tree_sha256",
            "train_annotation_sha256",
            "validation_annotation_sha256",
            "annotation_contract_sha256",
            "validation_report_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), name),
            )
        if self.status != "valid":
            raise DINOPreflightContractError(
                "VOC real-data integrity status must be 'valid'"
            )
        if self.dataset_terms_acknowledged is not True:
            raise DINOPreflightContractError(
                "VOC dataset terms acknowledgement is required"
            )
        for name in ("train_samples", "validation_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise DINOPreflightContractError(
                    f"{name} must be an integer >= 1"
                )
        categories = tuple(self.categories)
        if not categories:
            raise DINOPreflightContractError(
                "VOC evidence must contain categories"
            )
        ids = []
        for category_id, name in categories:
            if (
                isinstance(category_id, bool)
                or not isinstance(category_id, int)
                or category_id < 1
                or not isinstance(name, str)
                or not name.strip()
            ):
                raise DINOPreflightContractError(
                    "VOC categories must contain positive integer IDs and names"
                )
            ids.append(category_id)
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise DINOPreflightContractError(
                "VOC category IDs must be unique and sorted"
            )
        object.__setattr__(self, "categories", categories)
        if (
            len(categories) != 20
            or tuple(ids) != tuple(range(1, 21))
        ):
            raise DINOPreflightContractError(
                "full VOC2007 evidence must contain category IDs 1 through 20"
            )
        subset = tuple(self.inference_subset)
        if len(subset) != min(_INFERENCE_SUBSET_SIZE, self.validation_samples):
            raise DINOPreflightContractError(
                "VOC inference subset has the wrong deterministic size"
            )
        subset_ids = []
        for image_id, file_name in subset:
            if (
                isinstance(image_id, bool)
                or not isinstance(image_id, int)
                or image_id < 0
                or not isinstance(file_name, str)
                or not file_name
            ):
                raise DINOPreflightContractError(
                    "VOC inference subset entries are invalid"
                )
            relative = Path(file_name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in file_name
            ):
                raise DINOPreflightContractError(
                    "VOC inference subset paths must be safe relative paths"
                )
            subset_ids.append(image_id)
        if subset_ids != sorted(subset_ids) or len(subset_ids) != len(
            set(subset_ids)
        ):
            raise DINOPreflightContractError(
                "VOC inference subset must use unique ascending image IDs"
            )
        object.__setattr__(self, "inference_subset", subset)
        if not isinstance(self.invariants, Mapping) or not self.invariants:
            raise DINOPreflightContractError(
                "VOC integrity invariants must be a non-empty mapping"
            )
        normalized_invariants = {
            str(key): value for key, value in self.invariants.items()
        }
        if any(value is not True for value in normalized_invariants.values()):
            raise DINOPreflightContractError(
                "every VOC real-data integrity invariant must be true"
            )
        object.__setattr__(
            self,
            "invariants",
            MappingProxyType(dict(sorted(normalized_invariants.items()))),
        )
        expected_contract = canonical_sha256(
            {
                "dataset_id": self.dataset_id,
                "categories": [
                    {"id": category_id, "name": name}
                    for category_id, name in self.categories
                ],
                "invariants": dict(self.invariants),
            }
        )
        if self.annotation_contract_sha256 != expected_contract:
            raise DINOPreflightContractError(
                "annotation_contract_sha256 does not match VOC evidence"
            )

    @property
    def dataset_num_classes(self) -> int:
        """DINO class dimension: maximum COCO category ID plus one."""
        return max(category_id for category_id, _ in self.categories) + 1

    @property
    def category_ids(self) -> tuple[int, ...]:
        return tuple(category_id for category_id, _ in self.categories)

    @property
    def classmap_content(self) -> str:
        return "".join(f"{name}\n" for _, name in self.categories)

    @property
    def inference_subset_sha256(self) -> str:
        return canonical_sha256(
            [
                {"image_id": image_id, "file_name": file_name}
                for image_id, file_name in self.inference_subset
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "dataset_id": self.dataset_id,
            "dataset_root": str(self.dataset_root),
            "manifest_path": str(self.manifest_path),
            "integrity_path": str(self.integrity_path),
            "image_root": str(self.image_root),
            "train_annotation_path": str(self.train_annotation_path),
            "validation_annotation_path": str(
                self.validation_annotation_path
            ),
            "manifest_sha256": self.manifest_sha256,
            "integrity_sha256": self.integrity_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "train_annotation_sha256": self.train_annotation_sha256,
            "validation_annotation_sha256": (
                self.validation_annotation_sha256
            ),
            "annotation_contract_sha256": (
                self.annotation_contract_sha256
            ),
            "validation_report_sha256": self.validation_report_sha256,
            "train_samples": self.train_samples,
            "validation_samples": self.validation_samples,
            "categories": [
                {"id": category_id, "name": name}
                for category_id, name in self.categories
            ],
            "inference_subset": [
                {"image_id": image_id, "file_name": file_name}
                for image_id, file_name in self.inference_subset
            ],
            "inference_subset_sha256": self.inference_subset_sha256,
            "dataset_num_classes": self.dataset_num_classes,
            "invariants": dict(self.invariants),
            "dataset_terms_acknowledged": self.dataset_terms_acknowledged,
        }

    def validate_current_files(self) -> None:
        """Reject source, integrity, or annotation drift after validation."""
        expected = {
            self.manifest_path: self.manifest_sha256,
            self.integrity_path: self.integrity_sha256,
            self.train_annotation_path: self.train_annotation_sha256,
            self.validation_annotation_path: (
                self.validation_annotation_sha256
            ),
        }
        for path, digest in expected.items():
            if _file_hash(path, str(path)) != digest:
                raise DINOPreflightContractError(
                    f"VOC artifact changed after integrity validation: {path}"
                )
        if self.image_root.is_symlink() or not self.image_root.is_dir():
            raise DINOPreflightContractError(
                "VOC image root is missing or unsafe"
            )
        for _, file_name in self.inference_subset:
            image_path = self.image_root / file_name
            if image_path.is_symlink() or not image_path.is_file():
                raise DINOPreflightContractError(
                    "VOC deterministic inference subset is missing an image"
                )


def collect_voc_real_data_integrity(
    *,
    manifest_path: Path | str,
    dataset_root: Path | str,
    validator: Callable[..., Mapping[str, Any]] | None = None,
) -> VOCRealDataIntegrityEvidence:
    """Run an injected/read-only VOC verifier and return typed evidence.

    The default callback is the checked-in VOC2007 verifier.  Tests inject a
    fixture callback; neither path includes download functionality.
    """
    manifest_path = Path(manifest_path).resolve(strict=True)
    dataset_root = Path(dataset_root).resolve(strict=True)
    if validator is None:
        verifier_path = (
            Path(__file__).resolve().parents[1]
            / "datasets/voc2007/prepare_voc2007.py"
        )
        module_spec = importlib.util.spec_from_file_location(
            "_cross_model_voc2007_preparer",
            verifier_path,
        )
        if module_spec is None or module_spec.loader is None:
            raise DINOPreflightContractError(
                "checked-in VOC2007 verifier could not be loaded"
            )
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        validator = module.validate_prepared_dataset
    report = validator(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
    )
    if not isinstance(report, Mapping):
        raise DINOPreflightContractError(
            "VOC validator must return a mapping"
        )
    report = _canonical_json(dict(report))
    if report.get("status") != "valid":
        raise DINOPreflightContractError(
            "VOC validator did not return status='valid'"
        )
    try:
        reported_root = Path(report["dataset_root"]).resolve(strict=True)
    except (KeyError, OSError) as exc:
        raise DINOPreflightContractError(
            "VOC validator did not bind the prepared dataset root"
        ) from exc
    if reported_root != dataset_root:
        raise DINOPreflightContractError(
            "VOC validator result identifies a different dataset root"
        )

    manifest = _read_json(manifest_path, "VOC manifest")
    if manifest.get("dataset_id") != VOC_DATASET_ID:
        raise DINOPreflightContractError(
            "VOC manifest has the wrong dataset identity"
        )
    manifest_sha = _file_hash(manifest_path, "VOC manifest")
    if report.get("manifest_sha256") != manifest_sha:
        raise DINOPreflightContractError(
            "VOC validator manifest hash does not match the current file"
        )

    integrity_path = dataset_root / "integrity.v1.json"
    integrity = _read_json(integrity_path, "VOC integrity artifact")
    integrity_sha = _file_hash(integrity_path, "VOC integrity artifact")
    if report.get("integrity_sha256") != integrity_sha:
        raise DINOPreflightContractError(
            "VOC validator integrity hash does not match the current file"
        )
    if (
        integrity.get("manifest_sha256") != manifest_sha
        or integrity.get("dataset_terms_acknowledged") is not True
    ):
        raise DINOPreflightContractError(
            "VOC integrity artifact is not bound to the manifest and terms gate"
        )
    validation = report.get("validation")
    if (
        not isinstance(validation, Mapping)
        or integrity.get("validation") != validation
    ):
        raise DINOPreflightContractError(
            "VOC validator and integrity validation summaries differ"
        )
    invariants = validation.get("invariants")
    if not isinstance(invariants, Mapping) or any(
        value is not True for value in invariants.values()
    ):
        raise DINOPreflightContractError(
            "VOC validation invariants are incomplete"
        )

    conversion = manifest.get("conversion_contract")
    if not isinstance(conversion, Mapping):
        raise DINOPreflightContractError(
            "VOC manifest conversion contract is missing"
        )
    output_names = conversion.get("output_annotations")
    if not isinstance(output_names, Mapping):
        raise DINOPreflightContractError(
            "VOC output annotation contract is missing"
        )
    try:
        train_path = dataset_root / output_names["train"]
        validation_path = dataset_root / output_names["val"]
    except KeyError as exc:
        raise DINOPreflightContractError(
            "VOC train/validation output paths are missing"
        ) from exc
    train_document = _read_json(train_path, "VOC COCO train annotations")
    validation_document = _read_json(
        validation_path,
        "VOC COCO validation annotations",
    )
    train_sha = _file_hash(train_path, "VOC COCO train annotations")
    validation_sha = _file_hash(
        validation_path,
        "VOC COCO validation annotations",
    )
    output_records = {
        item.get("split"): item
        for item in integrity.get("outputs", [])
        if isinstance(item, Mapping)
    }
    for split, path, digest in (
        ("train", train_path, train_sha),
        ("val", validation_path, validation_sha),
    ):
        record = output_records.get(split)
        if (
            not isinstance(record, Mapping)
            or record.get("path")
            != path.relative_to(dataset_root).as_posix()
            or record.get("sha256") != digest
            or record.get("size_bytes") != path.stat().st_size
        ):
            raise DINOPreflightContractError(
                f"VOC {split} annotation is not bound by integrity evidence"
            )

    manifest_categories = manifest.get("categories")
    if (
        not isinstance(manifest_categories, list)
        or not manifest_categories
        or any(not isinstance(name, str) or not name for name in manifest_categories)
    ):
        raise DINOPreflightContractError(
            "VOC manifest categories are invalid"
        )
    expected_categories = tuple(
        (index, name)
        for index, name in enumerate(manifest_categories, start=1)
    )
    observed_categories = []
    validation_images: list[Mapping[str, Any]] | None = None
    for document_name, document in (
        ("train", train_document),
        ("validation", validation_document),
    ):
        categories = document.get("categories")
        images = document.get("images")
        annotations = document.get("annotations")
        if (
            not isinstance(categories, list)
            or not isinstance(images, list)
            or not isinstance(annotations, list)
        ):
            raise DINOPreflightContractError(
                f"VOC {document_name} COCO document is incomplete"
            )
        pairs = tuple(
            (item.get("id"), item.get("name"))
            for item in categories
            if isinstance(item, Mapping)
        )
        if pairs != expected_categories:
            raise DINOPreflightContractError(
                f"VOC {document_name} categories do not match the manifest"
            )
        observed_categories.append(pairs)
        if document_name == "validation":
            validation_images = [
                item for item in images if isinstance(item, Mapping)
            ]
            if len(validation_images) != len(images):
                raise DINOPreflightContractError(
                    "VOC validation image records are invalid"
                )
        allowed_ids = {category_id for category_id, _ in pairs}
        if any(
            not isinstance(item, Mapping)
            or item.get("category_id") not in allowed_ids
            for item in annotations
        ):
            raise DINOPreflightContractError(
                f"VOC {document_name} contains an invalid category ID"
            )
    if observed_categories[0] != observed_categories[1]:
        raise DINOPreflightContractError(
            "VOC train and validation category contracts differ"
        )

    splits = validation.get("splits")
    if not isinstance(splits, Mapping):
        raise DINOPreflightContractError(
            "VOC validation split summary is missing"
        )
    train_samples = len(train_document["images"])
    validation_samples = len(validation_document["images"])
    if (
        not isinstance(splits.get("train"), Mapping)
        or not isinstance(splits.get("val"), Mapping)
        or splits["train"].get("images") != train_samples
        or splits["val"].get("images") != validation_samples
    ):
        raise DINOPreflightContractError(
            "VOC COCO image counts differ from integrity evidence"
        )
    source_tree = integrity.get("source_tree")
    if not isinstance(source_tree, Mapping):
        raise DINOPreflightContractError(
            "VOC source-tree integrity evidence is missing"
        )
    if source_tree.get("algorithm") != "sha256":
        raise DINOPreflightContractError(
            "VOC source-tree evidence must use SHA-256"
        )
    source_tree_sha = source_tree.get("digest")
    _require_sha256(source_tree_sha, "VOC source_tree.sha256")
    if validation_images is None:
        raise DINOPreflightContractError(
            "VOC validation image inventory is missing"
        )
    for image in validation_images:
        image_id = image.get("id")
        file_name = image.get("file_name")
        if (
            isinstance(image_id, bool)
            or not isinstance(image_id, int)
            or image_id < 0
            or not isinstance(file_name, str)
            or not file_name
        ):
            raise DINOPreflightContractError(
                "VOC validation image identity is invalid"
            )
    ordered_validation_images = sorted(
        validation_images,
        key=lambda item: item["id"],
    )
    inference_subset = []
    for image in ordered_validation_images[:_INFERENCE_SUBSET_SIZE]:
        image_id = image.get("id")
        file_name = image.get("file_name")
        if (
            isinstance(image_id, bool)
            or not isinstance(image_id, int)
            or image_id < 0
            or not isinstance(file_name, str)
            or not file_name
        ):
            raise DINOPreflightContractError(
                "VOC validation image identity is invalid"
            )
        relative = Path(file_name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in file_name
        ):
            raise DINOPreflightContractError(
                "VOC validation image path is unsafe"
            )
        image_path = dataset_root / "VOCdevkit/VOC2007/JPEGImages" / relative
        if image_path.is_symlink() or not image_path.is_file():
            raise DINOPreflightContractError(
                "VOC deterministic inference image is missing"
            )
        inference_subset.append((image_id, file_name))

    annotation_contract_sha = canonical_sha256(
        {
            "dataset_id": VOC_DATASET_ID,
            "categories": [
                {"id": category_id, "name": name}
                for category_id, name in expected_categories
            ],
            "invariants": dict(invariants),
        }
    )
    evidence = VOCRealDataIntegrityEvidence(
        dataset_id=VOC_DATASET_ID,
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        integrity_path=integrity_path,
        image_root=dataset_root / "VOCdevkit/VOC2007/JPEGImages",
        train_annotation_path=train_path,
        validation_annotation_path=validation_path,
        manifest_sha256=manifest_sha,
        integrity_sha256=integrity_sha,
        source_tree_sha256=source_tree_sha,
        train_annotation_sha256=train_sha,
        validation_annotation_sha256=validation_sha,
        annotation_contract_sha256=annotation_contract_sha,
        validation_report_sha256=canonical_sha256(report),
        train_samples=train_samples,
        validation_samples=validation_samples,
        categories=expected_categories,
        inference_subset=tuple(inference_subset),
        invariants=dict(invariants),
    )
    evidence.validate_current_files()
    return evidence


@dataclass(frozen=True, slots=True)
class DINOSkillContract:
    """Content-addressed subset of the checked-in DINO skill."""

    skill_dir: Path
    container_image: str
    file_sha256s: Mapping[str, str] = field(repr=False)
    actions: Mapping[str, Mapping[str, Any]] = field(repr=False)
    templates: Mapping[str, Mapping[str, Any]] = field(repr=False)

    def __post_init__(self) -> None:
        skill_dir = _require_absolute_path(self.skill_dir, "skill_dir")
        object.__setattr__(self, "skill_dir", skill_dir)
        if not isinstance(self.container_image, str) or not self.container_image:
            raise DINOPreflightContractError(
                "DINO skill container image must be non-empty"
            )
        file_sha256s = dict(self.file_sha256s)
        if set(file_sha256s) != set(_SKILL_FILES):
            raise DINOPreflightContractError(
                "DINO skill file inventory is incomplete"
            )
        for name, digest in file_sha256s.items():
            _require_sha256(digest, f"skill file {name}")
        object.__setattr__(
            self,
            "file_sha256s",
            MappingProxyType(dict(sorted(file_sha256s.items()))),
        )
        actions = _freeze_json(self.actions)
        templates = _freeze_json(self.templates)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "templates", templates)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_dir": str(self.skill_dir),
            "container_image": self.container_image,
            "file_sha256s": dict(self.file_sha256s),
            "actions": _thaw_json(self.actions),
            "templates_sha256": {
                name: canonical_sha256(_thaw_json(template))
                for name, template in self.templates.items()
            },
        }


@dataclass(frozen=True, slots=True)
class DINORuntimeImageContract:
    """Reviewed mapping from the source skill to one exact TAO runtime.

    A DINO plan may be authored from an older checked-out skill while the
    physical preflight targets a reviewed TAO 7.1 skill/runtime pair.  The
    mapping is explicit and content addressed: no executor is permitted to
    infer that two image tags, schemas, or source trees are compatible.
    """

    source_skill_revision: str
    compatible_skill_revision: str
    source_skill_image: str
    compatible_skill_image: str
    source_skill_contract_sha256: str
    compatible_skill_contract_sha256: str
    runtime_image: str
    tao_schema_compatibility_sha256: str
    tao_source_compatibility_sha256: str
    status: str = "verified"

    def __post_init__(self) -> None:
        for name in (
            "source_skill_revision",
            "compatible_skill_revision",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
                raise DINOPreflightContractError(
                    f"{name} must be a full lowercase Git object ID"
                )
        for name in (
            "source_skill_contract_sha256",
            "compatible_skill_contract_sha256",
            "tao_schema_compatibility_sha256",
            "tao_source_compatibility_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), name),
            )
        for name in ("source_skill_image", "compatible_skill_image"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or "@" in value
                or any(character.isspace() for character in value)
            ):
                raise DINOPreflightContractError(
                    f"{name} must be a non-digest container image tag"
                )
        if (
            not isinstance(self.runtime_image, str)
            or _DIGEST_IMAGE_RE.fullmatch(self.runtime_image) is None
        ):
            raise DINOPreflightContractError(
                "runtime_image must be an exact repository@sha256 reference"
            )
        if self.status != "verified":
            raise DINOPreflightContractError(
                "runtime image compatibility status must be 'verified'"
            )

    @property
    def runtime_repository(self) -> str:
        match = _DIGEST_IMAGE_RE.fullmatch(self.runtime_image)
        assert match is not None
        return match.group("repository")

    @property
    def runtime_digest(self) -> str:
        match = _DIGEST_IMAGE_RE.fullmatch(self.runtime_image)
        assert match is not None
        return match.group("digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_skill_revision": self.source_skill_revision,
            "compatible_skill_revision": self.compatible_skill_revision,
            "source_skill_image": self.source_skill_image,
            "compatible_skill_image": self.compatible_skill_image,
            "source_skill_contract_sha256": (
                self.source_skill_contract_sha256
            ),
            "compatible_skill_contract_sha256": (
                self.compatible_skill_contract_sha256
            ),
            "runtime_image": self.runtime_image,
            "runtime_digest": self.runtime_digest,
            "tao_schema_compatibility_sha256": (
                self.tao_schema_compatibility_sha256
            ),
            "tao_source_compatibility_sha256": (
                self.tao_source_compatibility_sha256
            ),
            "status": self.status,
        }


def load_dino_skill_contract(
    skill_dir: Path | str,
) -> DINOSkillContract:
    """Load and fail closed on the exact DINO action/schema contract."""
    skill_dir = Path(skill_dir).resolve(strict=True)
    paths = {name: skill_dir / relative for name, relative in _SKILL_FILES.items()}
    skill_info = _read_yaml_mapping(paths["skill_info"], "DINO skill_info")
    if (
        skill_info.get("name") != "tao-train-dino"
        or skill_info.get("network_arch") != DINO_MODEL_ID
        or skill_info.get("data_format") not in {"coco", "coco_raw"}
        or skill_info.get("automl_enabled") is not True
    ):
        raise DINOPreflightContractError(
            "DINO skill identity or AutoML/data contract drifted"
        )
    actions_raw = skill_info.get("actions")
    if not isinstance(actions_raw, Mapping):
        raise DINOPreflightContractError("DINO skill actions are missing")
    actions = {}
    required_inputs = {
        "train": _TRAIN_INPUT_KEYS,
        "evaluate": _EVALUATE_INPUT_KEYS,
        "inference": _INFERENCE_INPUT_KEYS,
    }
    for action_name in ("train", "evaluate", "inference"):
        action = actions_raw.get(action_name)
        if not isinstance(action, Mapping):
            raise DINOPreflightContractError(
                f"DINO {action_name} action is missing"
            )
        if (
            action.get("command") != _EXPECTED_ACTION_COMMANDS[action_name]
            or action.get("mode") != "config"
            or action.get("config_format") != "yaml"
            or not isinstance(action.get("inputs"), Mapping)
            or not required_inputs[action_name].issubset(action["inputs"])
            or not isinstance(action.get("outputs"), Mapping)
        ):
            raise DINOPreflightContractError(
                f"DINO {action_name} action contract drifted"
            )
        actions[action_name] = _canonical_json(dict(action))

    templates = {
        "train": _read_yaml_mapping(
            paths["train_template"], "DINO train template"
        ),
        "evaluate": _read_yaml_mapping(
            paths["evaluate_template"], "DINO evaluate template"
        ),
        "inference": _read_yaml_mapping(
            paths["inference_template"], "DINO inference template"
        ),
    }
    schemas = {
        "train": _read_json(paths["train_schema"], "DINO train schema"),
        "evaluate": _read_json(
            paths["evaluate_schema"], "DINO evaluate schema"
        ),
        "inference": _read_json(
            paths["inference_schema"], "DINO inference schema"
        ),
    }
    try:
        num_classes = schemas["train"]["properties"]["dataset"]["properties"][
            "num_classes"
        ]
        num_gpus = schemas["train"]["properties"]["train"]["properties"][
            "num_gpus"
        ]
    except (KeyError, TypeError) as exc:
        raise DINOPreflightContractError(
            "DINO train schema lacks class/GPU fields"
        ) from exc
    if (
        num_classes.get("type") != "int"
        or num_classes.get("minimum") != 1
        or num_gpus.get("type") != "int"
        or num_gpus.get("minimum") != 1
    ):
        raise DINOPreflightContractError(
            "DINO train class/GPU schema contract drifted"
        )
    for action_name, template in templates.items():
        if (
            not isinstance(template.get("dataset"), Mapping)
            or not isinstance(template.get("model"), Mapping)
            or action_name not in template
            and action_name != "train"
        ):
            raise DINOPreflightContractError(
                f"DINO {action_name} template is incomplete"
            )
        if action_name == "train" and not isinstance(
            template.get("train"), Mapping
        ):
            raise DINOPreflightContractError(
                "DINO train template is incomplete"
            )

    return DINOSkillContract(
        skill_dir=skill_dir,
        container_image=str(skill_info.get("container_image", "")),
        file_sha256s={
            name: _file_hash(path, f"DINO skill file {name}")
            for name, path in paths.items()
        },
        actions=actions,
        templates=templates,
    )


@dataclass(frozen=True, slots=True)
class DINOPreflightSettings:
    """Frozen local resource, source, and latency settings."""

    preflight_id: str
    tao_version: str
    source_commit: str
    package_sha256: str
    container_sha256: str
    runtime_sha256: str
    runtime_image_contract: DINORuntimeImageContract
    latency_input_descriptor: Mapping[str, Any] = field(repr=False)
    seed: int = 271828
    batch_size: int = 1
    precision: str = "fp32"
    metric_name: str = DINO_METRIC
    latency_timed_scope: str = "model_forward"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "preflight_id",
            _require_identifier(self.preflight_id, "preflight_id"),
        )
        if not isinstance(self.tao_version, str) or not self.tao_version:
            raise DINOPreflightContractError(
                "tao_version must be non-empty"
            )
        if (
            not isinstance(self.source_commit, str)
            or _COMMIT_RE.fullmatch(self.source_commit) is None
        ):
            raise DINOPreflightContractError(
                "source_commit must be a full lowercase Git object ID"
            )
        for name in (
            "package_sha256",
            "container_sha256",
            "runtime_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), name),
            )
        if not isinstance(
            self.runtime_image_contract, DINORuntimeImageContract
        ):
            raise DINOPreflightContractError(
                "runtime_image_contract must be DINORuntimeImageContract"
            )
        if (
            self.runtime_image_contract.runtime_digest
            != self.container_sha256
        ):
            raise DINOPreflightContractError(
                "container_sha256 must match the exact runtime image digest"
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise DINOPreflightContractError(
                "seed must be an integer >= 0"
            )
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise DINOPreflightContractError(
                "batch_size must be an integer >= 1"
            )
        if not isinstance(self.precision, str) or not self.precision:
            raise DINOPreflightContractError("precision must be non-empty")
        if self.metric_name != DINO_METRIC:
            raise DINOPreflightContractError(
                f"DINO local preflight metric must be {DINO_METRIC!r}"
            )
        if (
            not isinstance(self.latency_timed_scope, str)
            or not self.latency_timed_scope
        ):
            raise DINOPreflightContractError(
                "latency_timed_scope must be non-empty"
            )
        object.__setattr__(
            self,
            "latency_input_descriptor",
            _freeze_json(self.latency_input_descriptor),
        )

    @property
    def latency_input_sha256(self) -> str:
        return canonical_sha256(_thaw_json(self.latency_input_descriptor))

    def latency_contract(self) -> LatencyBenchmarkContract:
        return LatencyBenchmarkContract(
            warmup_iterations=50,
            timed_iterations=100,
            repeated_rounds=5,
            tail_percentile=95.0,
            batch_size_per_replica=self.batch_size,
            precision=self.precision,
            timed_scope=self.latency_timed_scope,
            input_sha256=self.latency_input_sha256,
            runtime_sha256=self.runtime_sha256,
            expected_replicas=1,
            measurement_role="validation_only",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "preflight_id": self.preflight_id,
            "tao_version": self.tao_version,
            "source_commit": self.source_commit,
            "package_sha256": self.package_sha256,
            "container_sha256": self.container_sha256,
            "runtime_sha256": self.runtime_sha256,
            "runtime_image_contract": self.runtime_image_contract.to_dict(),
            "latency_input_descriptor": _thaw_json(
                self.latency_input_descriptor
            ),
            "latency_input_sha256": self.latency_input_sha256,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "precision": self.precision,
            "metric_name": self.metric_name,
            "latency_timed_scope": self.latency_timed_scope,
            "local_gpu_count": 1,
        }


@dataclass(frozen=True, slots=True)
class DINOPreflightCommand:
    """One immutable executor request in the preregistered plan."""

    command_id: str
    stage: str
    operation: str
    launcher: str
    specs_by_action: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict,
        repr=False,
    )
    ptm_id: str | None = None
    depends_on: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "command_id",
            _require_identifier(self.command_id, "command_id"),
        )
        if self.stage not in _PHYSICAL_STAGES:
            raise DINOPreflightContractError(
                f"command stage {self.stage!r} is not executable"
            )
        for name in ("operation", "launcher"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise DINOPreflightContractError(f"{name} must be non-empty")
        if self.ptm_id is not None:
            object.__setattr__(
                self,
                "ptm_id",
                _require_identifier(self.ptm_id, "ptm_id"),
            )
        dependencies = tuple(self.depends_on)
        for dependency in dependencies:
            _require_identifier(dependency, "depends_on")
        object.__setattr__(self, "depends_on", dependencies)
        object.__setattr__(
            self,
            "specs_by_action",
            _freeze_json(self.specs_by_action),
        )
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "stage": self.stage,
            "operation": self.operation,
            "launcher": self.launcher,
            "ptm_id": self.ptm_id,
            "depends_on": list(self.depends_on),
            "specs_by_action": _thaw_json(self.specs_by_action),
            "metadata": _thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class DINOPreflightCommandPlan:
    """Live typed plan plus its serializable, content-addressed artifact."""

    model_preflight_inputs: ModelPreflightInputs
    voc_integrity: VOCRealDataIntegrityEvidence
    skill_contract: DINOSkillContract
    settings: DINOPreflightSettings
    ptm_preflight_report_sha256: str
    resolved_ptm_inventory_sha256: str
    default_ptm_id: str
    eligible_ptm_ids: tuple[str, ...]
    latency_contract: LatencyBenchmarkContract
    commands: tuple[DINOPreflightCommand, ...]
    output_contract: Mapping[str, Any] = field(repr=False)
    inline_artifacts: Mapping[str, Any] = field(repr=False)
    plan_sha256: str = field(init=False)
    schema_version: int = DINO_PREFLIGHT_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DINO_PREFLIGHT_PLAN_SCHEMA_VERSION:
            raise DINOPreflightContractError(
                "unsupported DINOPreflight plan schema"
            )
        if not isinstance(self.model_preflight_inputs, ModelPreflightInputs):
            raise DINOPreflightContractError(
                "model_preflight_inputs must be typed production inputs"
            )
        if not isinstance(self.voc_integrity, VOCRealDataIntegrityEvidence):
            raise DINOPreflightContractError(
                "voc_integrity must be typed real-data evidence"
            )
        if not isinstance(self.skill_contract, DINOSkillContract):
            raise DINOPreflightContractError(
                "skill_contract must be DINOSkillContract"
            )
        if not isinstance(self.settings, DINOPreflightSettings):
            raise DINOPreflightContractError(
                "settings must be DINOPreflightSettings"
            )
        runtime = self.settings.runtime_image_contract
        if (
            runtime.source_skill_contract_sha256
            != self.skill_contract.sha256
            or runtime.compatible_skill_contract_sha256
            != self.skill_contract.sha256
            or runtime.source_skill_image
            != self.skill_contract.container_image
            or runtime.compatible_skill_image
            != self.skill_contract.container_image
            or _image_repository(runtime.compatible_skill_image)
            != runtime.runtime_repository
        ):
            raise DINOPreflightContractError(
                "runtime image mapping does not match the authoritative "
                "content-addressed DINO 7.1 skill contract"
            )
        for name in (
            "ptm_preflight_report_sha256",
            "resolved_ptm_inventory_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        object.__setattr__(
            self,
            "default_ptm_id",
            _require_identifier(self.default_ptm_id, "default_ptm_id"),
        )
        ids = tuple(self.eligible_ptm_ids)
        if not ids or ids != tuple(sorted(set(ids))):
            raise DINOPreflightContractError(
                "eligible PTM IDs must be non-empty, unique, and sorted"
            )
        if self.default_ptm_id not in ids:
            raise DINOPreflightContractError(
                "default PTM must be in the eligible inventory"
            )
        object.__setattr__(self, "eligible_ptm_ids", ids)
        if not isinstance(self.latency_contract, LatencyBenchmarkContract):
            raise DINOPreflightContractError(
                "latency_contract must be LatencyBenchmarkContract"
            )
        commands = tuple(self.commands)
        if not commands or not all(
            isinstance(command, DINOPreflightCommand)
            for command in commands
        ):
            raise DINOPreflightContractError(
                "commands must contain typed preflight commands"
            )
        object.__setattr__(self, "commands", commands)
        object.__setattr__(
            self,
            "output_contract",
            _freeze_json(self.output_contract),
        )
        object.__setattr__(
            self,
            "inline_artifacts",
            _freeze_json(self.inline_artifacts),
        )
        object.__setattr__(
            self,
            "plan_sha256",
            canonical_sha256(self._payload()),
        )
        self.validate()

    def commands_for_stage(
        self,
        stage: str,
    ) -> tuple[DINOPreflightCommand, ...]:
        return tuple(
            command for command in self.commands if command.stage == stage
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_preflight_inputs": self.model_preflight_inputs.to_dict(),
            "model_preflight_inputs_sha256": (
                self.model_preflight_inputs.canonical_sha256
            ),
            "voc_integrity": self.voc_integrity.to_dict(),
            "skill_contract": self.skill_contract.to_dict(),
            "skill_contract_sha256": self.skill_contract.sha256,
            "settings": self.settings.to_dict(),
            "ptm_preflight_report_sha256": (
                self.ptm_preflight_report_sha256
            ),
            "resolved_ptm_inventory_sha256": (
                self.resolved_ptm_inventory_sha256
            ),
            "default_ptm_id": self.default_ptm_id,
            "eligible_ptm_ids": list(self.eligible_ptm_ids),
            "ptm_checkpoint_targets": {
                command.ptm_id: _thaw_json(command.metadata)[
                    "checkpoint_target"
                ]
                for command in self.commands_for_stage(
                    "eligible_ptm_smoke"
                )
            },
            "latency_contract": self.latency_contract.to_dict(),
            "latency_contract_sha256": self.latency_contract.sha256,
            "output_contract": _thaw_json(self.output_contract),
            "inline_artifacts": _thaw_json(self.inline_artifacts),
            "commands": [
                {
                    **command.to_dict(),
                    "command_sha256": command.sha256,
                }
                for command in self.commands
            ],
            "execution_policy": {
                "executor_injected": True,
                "plan_build_invokes_executor": False,
                "plan_build_downloads": False,
                "plan_build_launches_container": False,
                "plan_build_uses_gpu": False,
                "execution_target": "local_single_gpu",
                "slurm_submission_supported": False,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        value = self._payload()
        value["plan_sha256"] = self.plan_sha256
        return value

    def validate(self) -> None:
        inputs = self.model_preflight_inputs
        if (
            inputs.model_id != DINO_MODEL_ID
            or inputs.task != DINO_TASK
            or inputs.local_gpu_count != 1
            or inputs.default_ptm_id != self.default_ptm_id
            or tuple(item.id for item in inputs.eligible_ptms)
            != self.eligible_ptm_ids
        ):
            raise DINOPreflightContractError(
                "production preflight inputs do not match the DINO plan"
            )
        if (
            inputs.dataset_manifest_sha256
            != self.voc_integrity.manifest_sha256
            or inputs.annotation_contract_sha256
            != self.voc_integrity.annotation_contract_sha256
            or inputs.train_split_sha256
            != self.voc_integrity.train_annotation_sha256
            or inputs.validation_split_sha256
            != self.voc_integrity.validation_annotation_sha256
        ):
            raise DINOPreflightContractError(
                "production preflight inputs do not match VOC evidence"
            )
        if (
            inputs.latency_protocol_sha256 != self.latency_contract.sha256
            or inputs.latency_input_sha256
            != self.latency_contract.input_sha256
            or inputs.latency_timed_scope
            != self.latency_contract.timed_scope
            or self.latency_contract.expected_replicas != 1
            or self.latency_contract.measurement_role != "validation_only"
        ):
            raise DINOPreflightContractError(
                "production latency inputs do not match the frozen contract"
            )
        ids = [command.command_id for command in self.commands]
        if len(ids) != len(set(ids)):
            raise DINOPreflightContractError(
                "preflight command IDs must be unique"
            )
        known = set(ids)
        for command in self.commands:
            if not set(command.depends_on).issubset(known):
                raise DINOPreflightContractError(
                    f"command {command.command_id!r} has an unknown dependency"
                )
        stage_counts = {
            stage: len(self.commands_for_stage(stage))
            for stage in _PHYSICAL_STAGES
        }
        if any(count == 0 for count in stage_counts.values()):
            raise DINOPreflightContractError(
                "every physical production preflight stage needs a command"
            )
        if stage_counts["eligible_ptm_smoke"] != len(
            self.eligible_ptm_ids
        ):
            raise DINOPreflightContractError(
                "eligible PTM smoke must contain exactly one command per PTM"
            )
        if any(
            count != 1
            for stage, count in stage_counts.items()
            if stage != "eligible_ptm_smoke"
        ):
            raise DINOPreflightContractError(
                "non-PTM-smoke stages must contain exactly one command"
            )
        smoke_ids = tuple(
            command.ptm_id
            for command in self.commands_for_stage("eligible_ptm_smoke")
        )
        if smoke_ids != self.eligible_ptm_ids:
            raise DINOPreflightContractError(
                "eligible PTM smoke commands are incomplete or unordered"
            )
        if canonical_sha256(_thaw_json(self.output_contract)) != (
            inputs.output_contract_sha256
        ):
            raise DINOPreflightContractError(
                "output contract hash does not match production inputs"
            )
        inline = _thaw_json(self.inline_artifacts)
        classmap = inline.get("voc_label_map")
        subset = inline.get("voc_inference_subset")
        if (
            not isinstance(classmap, Mapping)
            or classmap.get("line_count") != 20
            or classmap.get("content") != self.voc_integrity.classmap_content
            or classmap.get("sha256")
            != hashlib.sha256(
                self.voc_integrity.classmap_content.encode("utf-8")
            ).hexdigest()
            or not isinstance(subset, Mapping)
            or subset.get("entry_count")
            != len(self.voc_integrity.inference_subset)
            or subset.get("sha256")
            != self.voc_integrity.inference_subset_sha256
        ):
            raise DINOPreflightContractError(
                "inference subset or 20-line classmap artifact drifted"
            )
        latency_metadata = _thaw_json(
            self.commands_for_stage("latency_instrumentation")[0].metadata
        )
        if latency_metadata.get("single_gpu_cli") != {
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "device_id": "cuda:0",
            "gpu_ids": [0],
            "world_size_from_plan": True,
            "legacy_world_size_8_permitted": False,
        }:
            raise DINOPreflightContractError(
                "latency command must expose the single-GPU CLI seam"
            )
        if canonical_sha256(self._payload()) != self.plan_sha256:
            raise DINOPreflightContractError(
                "DINOPreflight plan integrity verification failed"
            )


def _validate_resolved_ptm_inventory(
    inventory: ResolvedPTMRuntimeInventory,
    *,
    tao_version: str,
) -> tuple[str, tuple[str, ...], dict[str, Any]]:
    if not isinstance(inventory, ResolvedPTMRuntimeInventory):
        raise DINOPreflightContractError(
            "resolved_ptm_inventory must be a live typed "
            "ResolvedPTMRuntimeInventory; serialized PTM evidence is rejected"
        )
    try:
        inventory.validate()
    except (TypeError, ValueError) as exc:
        raise DINOPreflightContractError(
            "resolved PTM inventory integrity verification failed"
        ) from exc
    report = inventory.report
    if not isinstance(report, PTMPreflightReport):
        raise DINOPreflightContractError(
            "resolved inventory does not retain typed PTM preflight evidence"
        )
    if (
        report.purpose != "runtime"
        or report.validation_statuses != ("supported",)
        or report.model != DINO_MODEL_ID
        or report.task != DINO_TASK
        or report.tao_version != tao_version
        or inventory.model != DINO_MODEL_ID
        or inventory.task != DINO_TASK
        or inventory.tao_version != tao_version
        or inventory.ptm_policy != "all"
        or not isinstance(report.inventory, PTMCompatibilityResult)
        or not report.credential_probe.ok
    ):
        raise DINOPreflightContractError(
            "PTM evidence is not the complete supported DINO runtime inventory"
        )
    if canonical_sha256(report.stable_dict()) != report.report_sha256:
        raise DINOPreflightContractError(
            "typed PTM preflight report integrity verification failed"
        )
    prepared = {item.checkpoint_id: item for item in report.prepared}
    prepared_ids = tuple(sorted(prepared))
    if not prepared_ids or len(prepared_ids) != len(report.prepared):
        raise DINOPreflightContractError(
            "typed PTM preflight prepared inventory is empty or duplicated"
        )
    if inventory.checkpoint_ids != prepared_ids:
        raise DINOPreflightContractError(
            "resolved PTM arms do not cover every prepared PTM"
        )
    default_ptm_id = report.inventory.default_checkpoint_id
    if default_ptm_id not in prepared:
        raise DINOPreflightContractError(
            "registered default DINO PTM did not pass typed preflight"
        )
    arms = {arm.checkpoint_id: arm for arm in inventory.arms}
    for checkpoint_id in prepared_ids:
        item = prepared[checkpoint_id]
        arm = arms[checkpoint_id]
        if (
            not item.runtime_eligible
            or item.registry_status != "supported"
            or not item.access_probe.ok
            or not item.load_smoke.ok
            or item.checkpoint.sha256 != arm.checkpoint_artifact_sha256
            or item.registry_record_sha256 != arm.registry_record_sha256
            or item.provenance_sha256 != arm.preflight_provenance_sha256
            or canonical_sha256(dict(arm.effective_base_spec))
            != arm.effective_base_spec_sha256
            or arm.checkpoint_target not in _DINO_CHECKPOINT_TARGETS
            or _get_dotted_path(
                arm.effective_base_spec,
                arm.checkpoint_target,
            )
            != arm.checkpoint_path
        ):
            raise DINOPreflightContractError(
                f"typed PTM evidence is inconsistent for {checkpoint_id!r}"
            )
        for artifact, expected, label in (
            (
                item.checkpoint.path,
                item.checkpoint.sha256,
                "checkpoint",
            ),
            (
                item.checkpoint_spec_artifact.path,
                item.checkpoint_spec_artifact.sha256,
                "checkpoint spec",
            ),
        ):
            if not artifact.is_file() or sha256_file(artifact) != expected:
                raise DINOPreflightContractError(
                    f"{checkpoint_id} {label} changed after PTM preflight"
                )
    return default_ptm_id, prepared_ids, prepared


def _dino_local_train_overrides(
    voc: VOCRealDataIntegrityEvidence,
    settings: DINOPreflightSettings,
) -> dict[str, Any]:
    return {
        "wandb": {"enable": False},
        "dataset": {
            "train_data_sources": [
                {
                    "image_dir": str(voc.image_root),
                    "json_file": str(voc.train_annotation_path),
                }
            ],
            "val_data_sources": [
                {
                    "image_dir": str(voc.image_root),
                    "json_file": str(voc.validation_annotation_path),
                }
            ],
            "test_data_sources": {
                "image_dir": str(voc.image_root),
                "json_file": str(voc.validation_annotation_path),
            },
            "infer_data_sources": {
                "image_dir": [_ARTIFACT_INFERENCE_SUBSET_TOKEN],
                "classmap": _ARTIFACT_CLASSMAP_TOKEN,
            },
            "num_classes": voc.dataset_num_classes,
            "eval_class_ids": list(voc.category_ids),
            "batch_size": settings.batch_size,
        },
        "train": {
            "num_gpus": 1,
            "gpu_ids": [0],
            "num_nodes": 1,
            "seed": settings.seed,
            "num_epochs": 1,
            "checkpoint_interval": 1,
            "validation_interval": 1,
            "precision": settings.precision,
        },
    }


def _assert_train_spec(
    spec: Mapping[str, Any],
    *,
    voc: VOCRealDataIntegrityEvidence,
) -> None:
    try:
        dataset = spec["dataset"]
        train = spec["train"]
    except (KeyError, TypeError) as exc:
        raise DINOPreflightContractError(
            "resolved DINO train spec is incomplete"
        ) from exc
    expected_train = [
        {
            "image_dir": str(voc.image_root),
            "json_file": str(voc.train_annotation_path),
        }
    ]
    expected_validation = [
        {
            "image_dir": str(voc.image_root),
            "json_file": str(voc.validation_annotation_path),
        }
    ]
    if (
        dataset.get("train_data_sources") != expected_train
        or dataset.get("val_data_sources") != expected_validation
        or dataset.get("num_classes") != max(voc.category_ids) + 1
        or train.get("num_gpus") != 1
        or train.get("gpu_ids") != [0]
        or train.get("num_nodes") != 1
        or train.get("num_epochs") != 1
        or train.get("checkpoint_interval") != 1
        or train.get("validation_interval") != 1
    ):
        raise DINOPreflightContractError(
            "DINO train spec violates the local one-epoch/single-GPU/data contract"
        )


def _action_spec(
    *,
    template: Mapping[str, Any],
    train_spec: Mapping[str, Any],
    action: str,
    checkpoint: str,
    voc: VOCRealDataIntegrityEvidence,
    settings: DINOPreflightSettings,
) -> dict[str, Any]:
    action_overrides: dict[str, Any] = {
        "wandb": {"enable": False},
        "model": copy.deepcopy(dict(train_spec["model"])),
        "dataset": {
            "train_data_sources": copy.deepcopy(
                train_spec["dataset"]["train_data_sources"]
            ),
            "val_data_sources": copy.deepcopy(
                train_spec["dataset"]["val_data_sources"]
            ),
            "test_data_sources": {
                "image_dir": str(voc.image_root),
                "json_file": str(voc.validation_annotation_path),
            },
            "infer_data_sources": {
                "image_dir": [_ARTIFACT_INFERENCE_SUBSET_TOKEN],
                "classmap": _ARTIFACT_CLASSMAP_TOKEN,
            },
            "num_classes": voc.dataset_num_classes,
            "eval_class_ids": list(voc.category_ids),
            "batch_size": settings.batch_size,
        },
    }
    if action == "evaluate":
        action_overrides["evaluate"] = {
            "checkpoint": checkpoint,
            "num_gpus": 1,
            "gpu_ids": [0],
            "num_nodes": 1,
            "batch_size": settings.batch_size,
        }
    elif action == "inference":
        action_overrides["inference"] = {
            "checkpoint": checkpoint,
            "num_gpus": 1,
            "gpu_ids": [0],
            "num_nodes": 1,
            "batch_size": settings.batch_size,
        }
    else:
        raise DINOPreflightContractError(
            f"unsupported DINO action spec {action!r}"
        )
    return merge_ptm_spec_precedence(
        model_defaults=template,
        candidate_overrides=action_overrides,
    ).spec


def build_dino_preflight_plan(
    *,
    voc_integrity: VOCRealDataIntegrityEvidence,
    resolved_ptm_inventory: ResolvedPTMRuntimeInventory,
    skill_dir: Path | str,
    settings: DINOPreflightSettings,
) -> DINOPreflightCommandPlan:
    """Freeze the complete local DINO plan without executing any command."""
    if not isinstance(voc_integrity, VOCRealDataIntegrityEvidence):
        raise DINOPreflightContractError(
            "typed VOC real-data integrity evidence is required"
        )
    if not isinstance(settings, DINOPreflightSettings):
        raise DINOPreflightContractError(
            "typed DINOPreflight settings are required"
        )
    voc_integrity.validate_current_files()
    skill = load_dino_skill_contract(skill_dir)
    (
        default_ptm_id,
        eligible_ptm_ids,
        prepared,
    ) = _validate_resolved_ptm_inventory(
        resolved_ptm_inventory,
        tao_version=settings.tao_version,
    )
    arms = {
        arm.checkpoint_id: arm for arm in resolved_ptm_inventory.arms
    }
    common_overrides = _dino_local_train_overrides(
        voc_integrity,
        settings,
    )
    train_specs = {}
    evaluate_specs = {}
    inference_specs = {}
    for checkpoint_id in eligible_ptm_ids:
        train_spec = merge_ptm_spec_precedence(
            model_defaults=arms[checkpoint_id].effective_base_spec,
            automl_profile_overrides=common_overrides,
        ).spec
        _assert_train_spec(train_spec, voc=voc_integrity)
        train_specs[checkpoint_id] = train_spec
        checkpoint = _smoke_model_token(checkpoint_id)
        evaluate_specs[checkpoint_id] = _action_spec(
            template=_thaw_json(skill.templates["evaluate"]),
            train_spec=train_spec,
            action="evaluate",
            checkpoint=checkpoint,
            voc=voc_integrity,
            settings=settings,
        )
        inference_specs[checkpoint_id] = _action_spec(
            template=_thaw_json(skill.templates["inference"]),
            train_spec=train_spec,
            action="inference",
            checkpoint=checkpoint,
            voc=voc_integrity,
            settings=settings,
        )

    default_evaluate_spec = _action_spec(
        template=_thaw_json(skill.templates["evaluate"]),
        train_spec=train_specs[default_ptm_id],
        action="evaluate",
        checkpoint=_ARTIFACT_CHECKPOINT_TOKEN,
        voc=voc_integrity,
        settings=settings,
    )
    default_inference_spec = _action_spec(
        template=_thaw_json(skill.templates["inference"]),
        train_spec=train_specs[default_ptm_id],
        action="inference",
        checkpoint=_ARTIFACT_CHECKPOINT_TOKEN,
        voc=voc_integrity,
        settings=settings,
    )
    latency_contract = settings.latency_contract()
    output_contract = copy.deepcopy(_OUTPUT_CONTRACT)
    merged_spec_sha = canonical_sha256(
        {
            checkpoint_id: canonical_sha256(train_specs[checkpoint_id])
            for checkpoint_id in eligible_ptm_ids
        }
    )
    ptm_identities = tuple(
        PreflightPTMIdentity(
            id=checkpoint_id,
            checkpoint_sha256=prepared[checkpoint_id].checkpoint.sha256,
            registry_record_sha256=(
                prepared[checkpoint_id].registry_record_sha256
            ),
            ptm_preflight_sha256=(
                prepared[checkpoint_id].provenance_sha256
            ),
        )
        for checkpoint_id in eligible_ptm_ids
    )
    model_inputs = ModelPreflightInputs(
        preflight_id=settings.preflight_id,
        model_id=DINO_MODEL_ID,
        task=DINO_TASK,
        tao_version=settings.tao_version,
        source_commit=settings.source_commit,
        package_sha256=settings.package_sha256,
        container_sha256=settings.container_sha256,
        dataset_id=voc_integrity.dataset_id,
        dataset_manifest_sha256=voc_integrity.manifest_sha256,
        annotation_contract_sha256=(
            voc_integrity.annotation_contract_sha256
        ),
        train_split_sha256=voc_integrity.train_annotation_sha256,
        validation_split_sha256=(
            voc_integrity.validation_annotation_sha256
        ),
        default_ptm_id=default_ptm_id,
        eligible_ptms=ptm_identities,
        merged_spec_sha256=merged_spec_sha,
        metric_name=settings.metric_name,
        latency_protocol_sha256=latency_contract.sha256,
        latency_input_sha256=latency_contract.input_sha256,
        latency_timed_scope=latency_contract.timed_scope,
        output_contract_sha256=canonical_sha256(output_contract),
        seed=settings.seed,
        local_gpu_count=1,
    )
    inline_artifacts = {
        "voc_label_map": {
            "path_token": _ARTIFACT_CLASSMAP_TOKEN,
            "media_type": "text/plain; charset=utf-8",
            "content": voc_integrity.classmap_content,
            "sha256": hashlib.sha256(
                voc_integrity.classmap_content.encode("utf-8")
            ).hexdigest(),
            "source": "verified COCO category order",
            "line_count": len(voc_integrity.categories),
        },
        "voc_inference_subset": {
            "path_token": _ARTIFACT_INFERENCE_SUBSET_TOKEN,
            "media_type": "application/json",
            "selection_rule": "lowest_coco_image_id_first",
            "source_image_root": str(voc_integrity.image_root),
            "entries": [
                {"image_id": image_id, "file_name": file_name}
                for image_id, file_name in voc_integrity.inference_subset
            ],
            "entry_count": len(voc_integrity.inference_subset),
            "sha256": voc_integrity.inference_subset_sha256,
        }
    }

    commands: list[DINOPreflightCommand] = [
        DINOPreflightCommand(
            command_id="dataset_validation",
            stage="dataset_validation",
            operation="verify_voc2007_real_data_integrity",
            launcher=(
                "experiments.cross_model_automl_20260729.datasets.voc2007."
                "prepare_voc2007.validate_prepared_dataset"
            ),
            metadata={
                "voc_integrity": voc_integrity.to_dict(),
                "network_access_permitted": False,
            },
        ),
        DINOPreflightCommand(
            command_id="default_ptm_load",
            stage="default_ptm_load",
            operation="load_default_ptm",
            launcher="model_executor_callback",
            ptm_id=default_ptm_id,
            specs_by_action={
                "train": train_specs[default_ptm_id],
            },
            depends_on=("dataset_validation",),
            metadata={
                "checkpoint_path": arms[default_ptm_id].checkpoint_path,
                "checkpoint_target": (
                    arms[default_ptm_id].checkpoint_target
                ),
                "checkpoint_sha256": (
                    prepared[default_ptm_id].checkpoint.sha256
                ),
                "input_contract_sha256": (
                    arms[default_ptm_id].input_contract_sha256
                ),
            },
        ),
    ]
    for checkpoint_id in eligible_ptm_ids:
        commands.append(
            DINOPreflightCommand(
                command_id=f"eligible_ptm_smoke/{checkpoint_id}",
                stage="eligible_ptm_smoke",
                operation="ptm_load_train_validation_inference_mini_step",
                launcher="model_executor_callback",
                ptm_id=checkpoint_id,
                specs_by_action={
                    "train": train_specs[checkpoint_id],
                    "evaluate": evaluate_specs[checkpoint_id],
                    "inference": inference_specs[checkpoint_id],
                },
                depends_on=("default_ptm_load",),
                metadata={
                    "checkpoint_path": arms[checkpoint_id].checkpoint_path,
                    "checkpoint_target": (
                        arms[checkpoint_id].checkpoint_target
                    ),
                    "initialized_model_binding": (
                        _smoke_model_token(checkpoint_id)
                    ),
                    "checkpoint_sha256": (
                        prepared[checkpoint_id].checkpoint.sha256
                    ),
                    "step_limits": {
                        "train_batches": 1,
                        "validation_batches": 1,
                        "inference_batches": 1,
                    },
                    "complete_epoch": False,
                },
            )
        )
    commands.extend(
        [
            DINOPreflightCommand(
                command_id="default_model_full_epoch",
                stage="default_model_full_epoch",
                operation="train_default_ptm_complete_one_epoch",
                launcher="tao_sdk.script_runner.build_entrypoint",
                ptm_id=default_ptm_id,
                specs_by_action={
                    "train": train_specs[default_ptm_id],
                },
                depends_on=tuple(
                    f"eligible_ptm_smoke/{checkpoint_id}"
                    for checkpoint_id in eligible_ptm_ids
                ),
                metadata={
                    "action_contract": _thaw_json(skill.actions["train"]),
                    "checkpoint_target": (
                        arms[default_ptm_id].checkpoint_target
                    ),
                    "complete_epochs": 1,
                    "single_gpu": True,
                    "automl_runner_compatible": True,
                },
            ),
            DINOPreflightCommand(
                command_id="in_epoch_validation",
                stage="in_epoch_validation",
                operation="read_default_epoch_validation",
                launcher="model_executor_callback",
                ptm_id=default_ptm_id,
                depends_on=("default_model_full_epoch",),
                metadata={
                    "metric_name": settings.metric_name,
                    "validation_interval": 1,
                },
            ),
            DINOPreflightCommand(
                command_id="standalone_evaluation",
                stage="standalone_evaluation",
                operation="standalone_dino_evaluate",
                launcher="tao_sdk.script_runner.build_entrypoint",
                ptm_id=default_ptm_id,
                specs_by_action={"evaluate": default_evaluate_spec},
                depends_on=("default_model_full_epoch",),
                metadata={
                    "action_contract": _thaw_json(
                        skill.actions["evaluate"]
                    ),
                    "checkpoint_binding": _ARTIFACT_CHECKPOINT_TOKEN,
                    "metric_name": settings.metric_name,
                },
            ),
            DINOPreflightCommand(
                command_id="checkpoint_save_reload",
                stage="checkpoint_save_reload",
                operation="reload_full_epoch_checkpoint",
                launcher="model_executor_callback",
                ptm_id=default_ptm_id,
                specs_by_action={
                    "train": train_specs[default_ptm_id],
                },
                depends_on=("default_model_full_epoch",),
                metadata={
                    "checkpoint_binding": _ARTIFACT_CHECKPOINT_TOKEN,
                    "byte_identity_required": True,
                },
            ),
            DINOPreflightCommand(
                command_id="latency_instrumentation",
                stage="latency_instrumentation",
                operation="stabilized_single_gpu_latency",
                launcher="tao_automl.latency_benchmark.run_replica_benchmark",
                ptm_id=default_ptm_id,
                specs_by_action={"inference": default_inference_spec},
                depends_on=("checkpoint_save_reload",),
                metadata={
                    "checkpoint_binding": _ARTIFACT_CHECKPOINT_TOKEN,
                    "latency_contract": latency_contract.to_dict(),
                    "latency_contract_sha256": latency_contract.sha256,
                    "candidate_fingerprint": canonical_sha256(
                        {
                            "ptm_id": default_ptm_id,
                            "checkpoint_binding": (
                                _ARTIFACT_CHECKPOINT_TOKEN
                            ),
                            "train_spec_sha256": canonical_sha256(
                                train_specs[default_ptm_id]
                            ),
                        }
                    ),
                    "selection_isolation": {
                        "selector_invoked_on_matched_measurements": False,
                        "selection_time_objectives_replaced": False,
                        "measurements_feed_selection": False,
                        "measurements_feed_reselection": False,
                        "algorithm_selected_candidate_overridden": False,
                    },
                    "single_gpu_cli": {
                        "world_size": 1,
                        "rank": 0,
                        "local_rank": 0,
                        "device_id": "cuda:0",
                        "gpu_ids": [0],
                        "world_size_from_plan": True,
                        "legacy_world_size_8_permitted": False,
                    },
                },
            ),
            DINOPreflightCommand(
                command_id="output_artifact_validation",
                stage="output_artifact_validation",
                operation="validate_preflight_output_artifacts",
                launcher="model_executor_callback",
                depends_on=(
                    "standalone_evaluation",
                    "latency_instrumentation",
                ),
                metadata={
                    "output_contract": output_contract,
                    "output_contract_sha256": canonical_sha256(
                        output_contract
                    ),
                },
            ),
            DINOPreflightCommand(
                command_id="interrupted_resume_replay",
                stage="interrupted_resume_replay",
                operation="automl_runner_interrupted_resume_replay",
                launcher="tao_automl.runner.AutoMLRunner.run",
                ptm_id=default_ptm_id,
                specs_by_action={
                    "train": train_specs[default_ptm_id],
                },
                depends_on=("output_artifact_validation",),
                metadata={
                    "resume": True,
                    "interrupt_after_completed_recommendations": 1,
                    "workspace_identity_sha256": canonical_sha256(
                        {
                            "preflight_id": settings.preflight_id,
                            "seed": settings.seed,
                            "merged_spec_sha256": merged_spec_sha,
                        }
                    ),
                    "same_seed_required": True,
                    "same_spec_required": True,
                    "same_next_request_required": True,
                    "no_duplicate_trials_required": True,
                    "no_lost_trials_required": True,
                },
            ),
        ]
    )
    provisional = {
        "schema_version": DINO_PREFLIGHT_PLAN_SCHEMA_VERSION,
        "model_preflight_inputs": model_inputs,
        "voc_integrity": voc_integrity,
        "skill_contract": skill,
        "settings": settings,
        "ptm_preflight_report_sha256": (
            resolved_ptm_inventory.report.report_sha256
        ),
        "resolved_ptm_inventory_sha256": (
            resolved_ptm_inventory.inventory_sha256
        ),
        "default_ptm_id": default_ptm_id,
        "eligible_ptm_ids": eligible_ptm_ids,
        "latency_contract": latency_contract,
        "commands": tuple(commands),
        "output_contract": output_contract,
        "inline_artifacts": inline_artifacts,
    }
    return DINOPreflightCommandPlan(**provisional)


@dataclass(frozen=True, slots=True)
class DINOPreflightExecutionResult:
    """Typed response from the injected physical executor."""

    command_id: str
    passed: bool
    evidence: Mapping[str, Any] = field(default_factory=dict)
    code: str = "ok"

    def __post_init__(self) -> None:
        _require_identifier(self.command_id, "execution command_id")
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be boolean")
        if not isinstance(self.code, str) or _SAFE_CODE_RE.fullmatch(
            self.code
        ) is None:
            raise ValueError("code must be a safe lowercase identifier")
        object.__setattr__(self, "evidence", _freeze_json(self.evidence))


class DINOPreflightExecutor(Protocol):
    """Physical execution boundary; implementations may use TAO SDK."""

    def __call__(
        self,
        command: DINOPreflightCommand,
    ) -> DINOPreflightExecutionResult:
        ...


class DINOModelPreflightAdapter:
    """Map DINO command results into the production stage evidence schema."""

    def __init__(
        self,
        plan: DINOPreflightCommandPlan,
        executor: DINOPreflightExecutor,
    ):
        if not isinstance(plan, DINOPreflightCommandPlan):
            raise TypeError("plan must be DINOPreflightCommandPlan")
        if not callable(executor):
            raise TypeError("executor must be callable")
        plan.validate()
        self.plan = plan
        self.executor = executor

    def _execute(
        self,
        command: DINOPreflightCommand,
    ) -> DINOPreflightExecutionResult:
        result = self.executor(command)
        if not isinstance(result, DINOPreflightExecutionResult):
            raise TypeError(
                "executor must return DINOPreflightExecutionResult"
            )
        if result.command_id != command.command_id:
            raise DINOPreflightContractError(
                "executor returned a result for a different command"
            )
        return result

    def __call__(
        self,
        request: ModelPreflightStepRequest,
    ) -> ModelPreflightStepResult:
        self.plan.validate()
        if (
            request.inputs.canonical_sha256
            != self.plan.model_preflight_inputs.canonical_sha256
        ):
            raise DINOPreflightContractError(
                "model preflight request does not match the frozen DINO plan"
            )
        commands = self.plan.commands_for_stage(request.stage)
        if not commands:
            raise DINOPreflightContractError(
                f"no DINO command is frozen for stage {request.stage!r}"
            )
        results = [self._execute(command) for command in commands]
        failed = next((result for result in results if not result.passed), None)
        if failed is not None:
            return ModelPreflightStepResult.failure(
                request.stage,
                failed.code,
            )

        if request.stage == "eligible_ptm_smoke":
            return ModelPreflightStepResult.success(
                request.stage,
                {
                    "ptms": [
                        _thaw_json(result.evidence)
                        for result in results
                    ]
                },
            )
        if request.stage == "latency_instrumentation":
            raw = _thaw_json(results[0].evidence)
            if set(raw) != {"replica_records"} or not isinstance(
                raw["replica_records"], list
            ):
                raise DINOPreflightContractError(
                    "latency executor must return replica_records only"
                )
            aggregate = combine_replica_records(raw["replica_records"])
            command = commands[0]
            metadata = _thaw_json(command.metadata)
            if (
                aggregate["contract_sha256"]
                != self.plan.latency_contract.sha256
                or aggregate["candidate_fingerprint"]
                != metadata["candidate_fingerprint"]
                or aggregate["selection_isolation"]
                != metadata["selection_isolation"]
            ):
                raise DINOPreflightContractError(
                    "latency aggregate does not match the frozen command"
                )
            statistics = aggregate["statistics"]
            return ModelPreflightStepResult.success(
                request.stage,
                {
                    "protocol_sha256": self.plan.latency_contract.sha256,
                    "input_sha256": (
                        self.plan.latency_contract.input_sha256
                    ),
                    "timed_scope": self.plan.latency_contract.timed_scope,
                    "single_gpu": True,
                    "warmup_iterations": (
                        self.plan.latency_contract.warmup_iterations
                    ),
                    "timed_iterations": (
                        self.plan.latency_contract.timed_iterations
                    ),
                    "rounds": (
                        self.plan.latency_contract.repeated_rounds
                    ),
                    "synchronized": True,
                    "median_ms": statistics["median_ms"],
                    "p95_ms": statistics["p95_ms"],
                    "mad_ms": statistics["mad_ms"],
                    "iqr_ms": statistics["iqr_ms"],
                    "robust_cv": statistics["robust_cv"],
                    "round_drift_ms": statistics["round_drift_ms"],
                    "device_spread_ms": (
                        statistics["device_median_range_ms"]
                    ),
                    "quality_gates_passed": statistics["is_valid"],
                },
            )
        if len(results) != 1:
            raise DINOPreflightContractError(
                f"stage {request.stage!r} unexpectedly returned multiple results"
            )
        return ModelPreflightStepResult.success(
            request.stage,
            _thaw_json(results[0].evidence),
        )


def run_dino_local_preflight(
    *,
    plan: DINOPreflightCommandPlan,
    executor: DINOPreflightExecutor,
    resume_report: Mapping[str, Any] | None = None,
    stop_after_stage: str | None = None,
) -> dict[str, Any]:
    """Execute or resume the production preflight using the frozen DINO plan."""
    adapter = DINOModelPreflightAdapter(plan, executor)
    return run_model_preflight(
        plan.model_preflight_inputs,
        adapter,
        resume_report=resume_report,
        stop_after_stage=stop_after_stage,
    )


def freeze_dino_preflight_plan(
    path: Path | str,
    plan: DINOPreflightCommandPlan,
    *,
    resume: bool = False,
) -> str:
    """Create the plan artifact once, or verify byte-identical resume."""
    if not isinstance(plan, DINOPreflightCommandPlan):
        raise TypeError("plan must be DINOPreflightCommandPlan")
    plan.validate()
    destination = Path(path)
    if not destination.is_absolute():
        raise DINOPreflightContractError(
            "plan artifact path must be absolute"
        )
    content = (
        json.dumps(
            plan.to_dict(),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if resume:
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_bytes() != content
        ):
            raise DINOPreflightContractError(
                "resume requires the byte-identical frozen plan artifact"
            )
        return plan.plan_sha256
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(destination, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise
    return plan.plan_sha256


__all__ = [
    "DINO_PREFLIGHT_PLAN_SCHEMA_VERSION",
    "DINOPreflightCommand",
    "DINOPreflightCommandPlan",
    "DINOPreflightContractError",
    "DINOPreflightExecutionResult",
    "DINOPreflightExecutor",
    "DINOPreflightSettings",
    "DINORuntimeImageContract",
    "DINOModelPreflightAdapter",
    "DINOSkillContract",
    "VOCRealDataIntegrityEvidence",
    "build_dino_preflight_plan",
    "collect_voc_real_data_integrity",
    "freeze_dino_preflight_plan",
    "load_dino_skill_contract",
    "run_dino_local_preflight",
]
