#!/usr/bin/env python3

"""Validate and seal the official COCO-to-Grounding-DINO conversion.

The actual conversion is performed by the pinned ``tao-dataservices``
implementation.  This module deliberately does not launch a model or submit a
scheduler job.  It verifies two independent converter outputs byte-for-byte
and semantically proves that:

* every source training annotation is represented exactly once in ODVG;
* the only excluded training images are source images with zero annotations;
* validation retains every image and annotation, changing only category IDs
  from the source IDs to contiguous ``0..N-1`` IDs; and
* category prompts are the source COCO category names without agent-authored
  synonyms or phrases.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tao_automl.ptm_registry import canonical_sha256

try:
    from .contract import PreparationError, read_json, sha256_file
except ImportError:  # pragma: no cover - direct script execution
    from contract import PreparationError, read_json, sha256_file


HERE = Path(__file__).resolve().parent
DEFAULT_DATASET_MANIFEST = (
    HERE.parent / "datasets" / "tao_od_synthetic_full_dino_coco" / "manifest.v1.json"
)
DEFAULT_ARTIFACT_ROOT = Path(
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/grounding_dino_dataset_conversion_v1"
)
DEFAULT_OUTPUT = HERE / "dataset_conversion.v1.json"
DEFAULT_LUSTRE_ROOT = Path(
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/"
    "tao_od_synthetic_full_dino_coco/grounding_dino_odvg_v1"
)


def _json_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _category_mapping(
    source: Mapping[str, Any],
) -> tuple[dict[int, int], dict[int, str]]:
    categories = sorted(source["categories"], key=lambda item: item["id"])
    source_to_model = {
        int(category["id"]): index
        for index, category in enumerate(categories)
    }
    source_to_name = {
        int(category["id"]): str(category["name"])
        for category in categories
    }
    return source_to_model, source_to_name


def _expected_odvg_instance(
    annotation: Mapping[str, Any],
    source_to_model: Mapping[int, int],
    source_to_name: Mapping[int, str],
) -> dict[str, Any]:
    x, y, width, height = annotation["bbox"]
    category_id = int(annotation["category_id"])
    result: dict[str, Any] = {
        "bbox": [
            round(x, 2),
            round(y, 2),
            round(x + width, 2),
            round(y + height, 2),
        ],
        "label": source_to_model[category_id],
        "category": source_to_name[category_id],
    }
    if annotation.get("segmentation"):
        result["mask"] = copy.deepcopy(annotation["segmentation"])
    return result


def _counter(values: Sequence[Mapping[str, Any]]) -> collections.Counter[str]:
    return collections.Counter(_json_value(value) for value in values)


def validate_train_odvg(
    source: Mapping[str, Any],
    odvg_path: str | Path,
    label_map_path: str | Path,
) -> dict[str, Any]:
    """Validate exact annotation preservation and audited empty-image removal."""
    source_to_model, source_to_name = _category_mapping(source)
    expected_label_map = {
        str(source_to_model[source_id]): source_to_name[source_id]
        for source_id in sorted(source_to_model)
    }
    label_map = read_json(label_map_path)
    if label_map != expected_label_map:
        raise PreparationError("ODVG label map differs from source categories")

    images = {int(image["id"]): image for image in source["images"]}
    annotations_by_image: dict[int, list[Mapping[str, Any]]] = {
        image_id: [] for image_id in images
    }
    for annotation in source["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    records = [
        json.loads(line)
        for line in Path(odvg_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    observed_by_image: dict[int, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        image_id = record.get("image_id")
        if isinstance(image_id, bool) or not isinstance(image_id, int):
            raise PreparationError(f"ODVG record {index} has invalid image_id")
        if image_id in observed_by_image:
            raise PreparationError(f"duplicate ODVG image_id {image_id}")
        observed_by_image[image_id] = record

    nonempty_ids = {
        image_id
        for image_id, annotations in annotations_by_image.items()
        if annotations
    }
    empty_ids = sorted(set(images) - nonempty_ids)
    if set(observed_by_image) != nonempty_ids:
        missing = sorted(nonempty_ids - set(observed_by_image))
        extra = sorted(set(observed_by_image) - nonempty_ids)
        raise PreparationError(
            f"ODVG image membership mismatch: missing={missing}, extra={extra}"
        )

    observed_instance_count = 0
    for image_id in sorted(nonempty_ids):
        source_image = images[image_id]
        observed = observed_by_image[image_id]
        expected_meta = {
            "file_name": source_image["file_name"],
            "height": source_image["height"],
            "width": source_image["width"],
            "image_id": image_id,
        }
        for key, value in expected_meta.items():
            if observed.get(key) != value:
                raise PreparationError(
                    f"ODVG image {image_id} metadata field {key!r} differs"
                )
        if set(observed) != set(expected_meta) | {"detection"}:
            raise PreparationError(f"ODVG image {image_id} has unexpected fields")
        detection = observed.get("detection")
        if not isinstance(detection, Mapping) or set(detection) != {"instances"}:
            raise PreparationError(
                f"ODVG image {image_id} has malformed detection record"
            )
        instances = detection["instances"]
        expected_instances = [
            _expected_odvg_instance(
                annotation,
                source_to_model,
                source_to_name,
            )
            for annotation in annotations_by_image[image_id]
        ]
        if _counter(instances) != _counter(expected_instances):
            raise PreparationError(
                f"ODVG instances differ for source image {image_id}"
            )
        observed_instance_count += len(instances)

    if observed_instance_count != len(source["annotations"]):
        raise PreparationError("ODVG annotation count differs from source")

    return {
        "source_image_count": len(images),
        "source_annotation_count": len(source["annotations"]),
        "output_record_count": len(records),
        "output_instance_count": observed_instance_count,
        "excluded_empty_image_count": len(empty_ids),
        "excluded_empty_image_ids": empty_ids,
        "all_source_annotations_preserved": True,
        "all_annotated_image_metadata_preserved": True,
        "manual_prompt_or_synonym_injection": False,
        "label_map": label_map,
        "label_map_sha256": canonical_sha256(label_map),
    }


def validate_validation_coco(
    source: Mapping[str, Any],
    converted_path: str | Path,
) -> dict[str, Any]:
    """Validate the contiguous validation COCO transformation."""
    converted = read_json(converted_path)
    if set(converted) != {"images", "annotations", "categories"}:
        raise PreparationError("converted validation COCO has unexpected fields")
    if converted["images"] != source["images"]:
        raise PreparationError("converted validation image records differ")

    source_to_model, _ = _category_mapping(source)
    expected_categories = []
    for category in source["categories"]:
        expected = copy.deepcopy(category)
        expected["id"] = source_to_model[int(category["id"])]
        expected_categories.append(expected)
    if converted["categories"] != expected_categories:
        raise PreparationError("converted validation categories differ")

    expected_annotations = []
    for annotation in source["annotations"]:
        expected = copy.deepcopy(annotation)
        expected["category_id"] = source_to_model[int(annotation["category_id"])]
        expected_annotations.append(expected)
    expected_by_id = {
        int(annotation["id"]): annotation
        for annotation in expected_annotations
    }
    observed_by_id = {
        int(annotation["id"]): annotation
        for annotation in converted["annotations"]
    }
    if len(observed_by_id) != len(converted["annotations"]):
        raise PreparationError("converted validation annotation IDs are not unique")
    if observed_by_id != expected_by_id:
        raise PreparationError("converted validation annotations differ")

    return {
        "source_image_count": len(source["images"]),
        "output_image_count": len(converted["images"]),
        "source_annotation_count": len(source["annotations"]),
        "output_annotation_count": len(converted["annotations"]),
        "source_category_ids": sorted(source_to_model),
        "output_category_ids": list(range(len(source_to_model))),
        "all_images_preserved": True,
        "all_annotations_preserved": True,
        "only_category_ids_remapped": True,
    }


def build_conversion_manifest(
    *,
    dataset_manifest_path: str | Path,
    artifact_root: str | Path,
    dataservices_root: str | Path,
    lustre_root: str | Path,
) -> dict[str, Any]:
    """Build a deterministic conversion proof from two independent runs."""
    dataset_manifest_path = Path(dataset_manifest_path).resolve()
    artifact_root = Path(artifact_root).resolve()
    dataservices_root = Path(dataservices_root).resolve()
    lustre_root = Path(lustre_root)
    source_manifest = read_json(dataset_manifest_path)

    source_paths = {
        "train": artifact_root / "source" / "train.annotations.json",
        "validation": artifact_root / "source" / "validation.annotations.json",
    }
    for split, path in source_paths.items():
        expected = source_manifest["splits"][split]["annotation"]
        if sha256_file(path) != expected["sha256"]:
            raise PreparationError(f"{split} source annotation hash differs")
        if path.stat().st_size != expected["size_bytes"]:
            raise PreparationError(f"{split} source annotation size differs")

    paths = {
        "run_a": {
            "train_odvg": (
                artifact_root / "run_a" / "train" / "train.annotations_odvg.jsonl"
            ),
            "train_label_map": (
                artifact_root
                / "run_a"
                / "train"
                / "train.annotations_odvg_labelmap.json"
            ),
            "validation_coco": (
                artifact_root
                / "run_a"
                / "validation"
                / "validation.annotations_remapped.json"
            ),
        },
        "run_b": {
            "train_odvg": (
                artifact_root / "run_b" / "train" / "train.annotations_odvg.jsonl"
            ),
            "train_label_map": (
                artifact_root
                / "run_b"
                / "train"
                / "train.annotations_odvg_labelmap.json"
            ),
            "validation_coco": (
                artifact_root
                / "run_b"
                / "validation"
                / "validation.annotations_remapped.json"
            ),
        },
    }
    identities = {
        run: {name: _file_record(path) for name, path in values.items()}
        for run, values in paths.items()
    }
    for name in paths["run_a"]:
        if identities["run_a"][name]["sha256"] != identities["run_b"][name]["sha256"]:
            raise PreparationError(f"converter output {name!r} is not deterministic")

    train_source = read_json(source_paths["train"])
    validation_source = read_json(source_paths["validation"])
    train_proof = validate_train_odvg(
        train_source,
        paths["run_a"]["train_odvg"],
        paths["run_a"]["train_label_map"],
    )
    validation_proof = validate_validation_coco(
        validation_source,
        paths["run_a"]["validation_coco"],
    )

    converter_files = {
        "coco_to_odvg": (
            dataservices_root
            / "nvidia_tao_ds"
            / "annotations"
            / "conversion"
            / "coco_to_odvg.py"
        ),
        "coco_to_contiguous": (
            dataservices_root
            / "nvidia_tao_ds"
            / "annotations"
            / "conversion"
            / "coco_to_contiguous.py"
        ),
    }
    import subprocess

    revision = subprocess.run(
        ["git", "-C", str(dataservices_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(dataservices_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    )

    canonical_outputs = {
        "train_odvg": {
            **identities["run_a"]["train_odvg"],
            "lustre_path": str(lustre_root / "train" / "annotations_odvg.jsonl"),
        },
        "train_label_map": {
            **identities["run_a"]["train_label_map"],
            "lustre_path": str(
                lustre_root / "train" / "annotations_odvg_labelmap.json"
            ),
        },
        "validation_coco": {
            **identities["run_a"]["validation_coco"],
            "lustre_path": str(
                lustre_root / "validation" / "annotations_remapped.json"
            ),
        },
    }
    for value in canonical_outputs.values():
        value.pop("path")

    document = {
        "schema_version": 1,
        "dataset_id": source_manifest["dataset_id"],
        "conversion_scope": "category_prompted_object_detection",
        "forbidden_claim": "referring_expression_box_grounding",
        "source_manifest": _file_record(dataset_manifest_path),
        "source_annotations": {
            split: {
                **_file_record(path),
                "lustre_path": source_manifest["splits"][split]["annotation"][
                    "path"
                ],
            }
            for split, path in source_paths.items()
        },
        "implementation": {
            "repository": str(dataservices_root),
            "revision": revision,
            "clean": not dirty,
            "use_all_categories": False,
            "converter_files": {
                name: _file_record(path)
                for name, path in converter_files.items()
            },
            "invocation": {
                "train": "convert_coco_to_odvg",
                "validation": "convert_coco_to_contiguous",
            },
        },
        "determinism": {
            "independent_runs": 2,
            "byte_identical": True,
            "run_output_identities": identities,
        },
        "semantic_validation": {
            "train": train_proof,
            "validation": validation_proof,
            "lossless_contract": (
                "all annotations and annotated-image metadata preserved; "
                "official ODVG converter intentionally excludes only audited "
                "zero-annotation training images"
            ),
            "annotation_lossless": True,
            "image_count_lossless": False,
        },
        "canonical_outputs": canonical_outputs,
        "staging": {
            "lustre_root": str(lustre_root),
            "inside_existing_source_dataset_tree": True,
            "atomic_publish_required": True,
            "published": False,
        },
        "execution": {
            "cpu_model_runs": 0,
            "gpu_model_runs": 0,
            "scheduler_jobs_submitted": 0,
        },
    }
    document["manifest_sha256"] = canonical_sha256(document)
    return document


def validate_conversion_manifest(document: Mapping[str, Any]) -> None:
    if document.get("determinism", {}).get("byte_identical") is not True:
        raise PreparationError("conversion is not byte deterministic")
    semantic = document.get("semantic_validation", {})
    if semantic.get("annotation_lossless") is not True:
        raise PreparationError("source annotations were not preserved")
    if semantic.get("image_count_lossless") is not False:
        raise PreparationError("official empty-image exclusion is obscured")
    if semantic.get("train", {}).get("excluded_empty_image_count") != 49:
        raise PreparationError("unexpected empty training image exclusion")
    if semantic.get("validation", {}).get("all_images_preserved") is not True:
        raise PreparationError("validation images were not preserved")
    if document.get("execution") != {
        "cpu_model_runs": 0,
        "gpu_model_runs": 0,
        "scheduler_jobs_submitted": 0,
    }:
        raise PreparationError("conversion manifest may not claim model execution")
    expected = copy.deepcopy(dict(document))
    observed = expected.pop("manifest_sha256", None)
    if observed != canonical_sha256(expected):
        raise PreparationError("manifest_sha256 does not match content")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--dataservices-root",
        type=Path,
        default=Path("/localhome/local-rarunachalam/tao-dataservices"),
    )
    parser.add_argument("--lustre-root", type=Path, default=DEFAULT_LUSTRE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()

    document = build_conversion_manifest(
        dataset_manifest_path=arguments.dataset_manifest,
        artifact_root=arguments.artifact_root,
        dataservices_root=arguments.dataservices_root,
        lustre_root=arguments.lustre_root,
    )
    validate_conversion_manifest(document)
    if not arguments.check_only:
        arguments.output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "annotation_lossless": document["semantic_validation"][
                    "annotation_lossless"
                ],
                "excluded_empty_training_images": document[
                    "semantic_validation"
                ]["train"]["excluded_empty_image_count"],
                "manifest_sha256": document["manifest_sha256"],
                "scheduler_jobs_submitted": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
