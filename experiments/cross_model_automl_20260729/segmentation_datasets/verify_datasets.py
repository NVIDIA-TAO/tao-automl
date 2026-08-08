#!/usr/bin/env python3
"""Verify the frozen VOC2012 and COCO2017 segmentation dataset projections.

This module performs data-only validation.  It does not import or execute TAO
models and it does not submit scheduler jobs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


VOC_CLASSES = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
VOC_VALID_LABELS = frozenset(range(21)) | {255}
COCO_EXPECTED = {
    "train": {"images": 118_287, "instance_annotations": 860_001},
    "val": {"images": 5_000, "instance_annotations": 36_781},
}


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash JSON using a deterministic serialization."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _split_ids(path: Path) -> tuple[str, ...]:
    values = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate split IDs in {path}")
    return values


def verify_voc2012(root: Path, source_archive: Path) -> dict[str, Any]:
    """Verify the lossless TAO folder projection of VOC2012 segmentation."""
    source = root / "source" / "VOCdevkit" / "VOC2012"
    prepared = root / "prepared"
    split_root = source / "ImageSets" / "Segmentation"
    split_ids = {
        split: _split_ids(split_root / f"{split}.txt") for split in ("train", "val")
    }
    expected_counts = {"train": 1464, "val": 1449}
    for split, expected in expected_counts.items():
        if len(split_ids[split]) != expected:
            raise ValueError(f"VOC2012 {split} has {len(split_ids[split])}, expected {expected}")
    if set(split_ids["train"]) & set(split_ids["val"]):
        raise ValueError("VOC2012 segmentation train/val identities overlap")
    trainval = set(_split_ids(split_root / "trainval.txt"))
    if trainval != set(split_ids["train"]) | set(split_ids["val"]):
        raise ValueError("VOC2012 trainval.txt is not the exact train/val union")

    split_reports: dict[str, Any] = {}
    global_pixels: Counter[int] = Counter()
    for split, identifiers in split_ids.items():
        image_dir = prepared / "images" / split
        mask_dir = prepared / "masks" / split
        image_stems = {path.stem for path in image_dir.glob("*.jpg")}
        mask_stems = {path.stem for path in mask_dir.glob("*.png")}
        expected_stems = set(identifiers)
        if image_stems != expected_stems or mask_stems != expected_stems:
            raise ValueError(f"VOC2012 {split} projection does not match official split")

        pixels: Counter[int] = Counter()
        for identifier in identifiers:
            source_image = source / "JPEGImages" / f"{identifier}.jpg"
            source_mask = source / "SegmentationClass" / f"{identifier}.png"
            image_path = image_dir / f"{identifier}.jpg"
            mask_path = mask_dir / f"{identifier}.png"
            if not source_image.samefile(image_path) or not source_mask.samefile(mask_path):
                raise ValueError(
                    "VOC2012 projection is not byte-identity preserving: "
                    f"{identifier}"
                )
            with Image.open(image_path) as image:
                image_size = image.size
                image.verify()
            with Image.open(mask_path) as mask:
                mask_size = mask.size
                labels, counts = np.unique(np.asarray(mask), return_counts=True)
            if image_size != mask_size:
                raise ValueError(f"VOC2012 image/mask dimensions differ: {identifier}")
            invalid = set(int(value) for value in labels) - VOC_VALID_LABELS
            if invalid:
                raise ValueError(f"VOC2012 invalid labels {sorted(invalid)} in {identifier}")
            pixels.update(
                {int(label): int(count) for label, count in zip(labels, counts)}
            )
        global_pixels.update(pixels)
        split_reports[split] = {
            "images": len(identifiers),
            "masks": len(identifiers),
            "pixel_counts_by_label": {str(key): pixels[key] for key in sorted(pixels)},
            "split_ids_sha256": hashlib.sha256(
                ("\n".join(identifiers) + "\n").encode("utf-8")
            ).hexdigest(),
        }

    return {
        "dataset_id": "pascal_voc2012_segmentation_trainval",
        "source_archive": {
            "path": str(source_archive),
            "size_bytes": source_archive.stat().st_size,
            "sha256": sha256_file(source_archive),
        },
        "official_split_counts": expected_counts,
        "train_val_disjoint": True,
        "trainval_images": len(trainval),
        "class_count_including_background": 21,
        "classes": list(VOC_CLASSES),
        "ignore_label": 255,
        "observed_labels": sorted(global_pixels),
        "pixel_counts_by_label": {
            str(key): global_pixels[key] for key in sorted(global_pixels)
        },
        "projection": {
            "root": str(prepared),
            "method": "hard-link projection from official split lists; no pixel conversion",
            "source_and_projection_same_inode": True,
        },
        "splits": split_reports,
    }


def _rounded_xyxy(bbox: list[float]) -> list[float]:
    x, y, width, height = bbox
    return [round(x, 2), round(y, 2), round(x + width, 2), round(y + height, 2)]


def verify_odvg_projection(
    instance_json: Path, odvg_jsonl: Path, label_map_path: Path
) -> dict[str, Any]:
    """Prove that the official COCO-to-ODVG projection preserves all masks."""
    source = _load_json(instance_json)
    categories = sorted(source["categories"], key=lambda category: category["id"])
    category_by_id = {category["id"]: category for category in categories}
    contiguous_by_id = {
        category["id"]: index for index, category in enumerate(categories)
    }
    expected_label_map = {
        str(index): category["name"] for index, category in enumerate(categories)
    }
    label_map = _load_json(label_map_path)
    if label_map != expected_label_map:
        raise ValueError("ODVG label map is not the canonical contiguous COCO mapping")

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    annotation_ids: set[int] = set()
    for annotation in source["annotations"]:
        annotation_id = annotation["id"]
        if annotation_id in annotation_ids:
            raise ValueError(f"duplicate COCO annotation ID {annotation_id}")
        annotation_ids.add(annotation_id)
        annotations_by_image[annotation["image_id"]].append(annotation)

    expected_images = [
        image for image in source["images"] if annotations_by_image[image["id"]]
    ]
    line_count = 0
    projected_annotations = 0
    mask_count = 0
    with odvg_jsonl.open("r", encoding="utf-8") as stream:
        for image, line in zip(expected_images, stream, strict=True):
            record = json.loads(line)
            expected_meta = {
                "file_name": image["file_name"],
                "height": image["height"],
                "width": image["width"],
                "image_id": image["id"],
            }
            for key, expected in expected_meta.items():
                if record.get(key) != expected:
                    raise ValueError(f"ODVG metadata mismatch for image {image['id']}: {key}")
            instances = record.get("detection", {}).get("instances")
            source_annotations = annotations_by_image[image["id"]]
            if not isinstance(instances, list) or len(instances) != len(source_annotations):
                raise ValueError(f"ODVG instance count mismatch for image {image['id']}")
            for annotation, projected in zip(source_annotations, instances, strict=True):
                category = category_by_id[annotation["category_id"]]
                expected = {
                    "bbox": _rounded_xyxy(annotation["bbox"]),
                    "label": contiguous_by_id[annotation["category_id"]],
                    "category": category["name"],
                }
                if annotation.get("segmentation"):
                    expected["mask"] = annotation["segmentation"]
                    mask_count += 1
                if projected != expected:
                    raise ValueError(
                        f"ODVG annotation mismatch at COCO annotation {annotation['id']}"
                    )
                projected_annotations += 1
            line_count += 1

        extra = stream.readline()
        if extra:
            raise ValueError("ODVG has records beyond the expected nonempty images")

    if projected_annotations != len(source["annotations"]):
        raise ValueError("ODVG did not preserve every COCO instance annotation")
    if mask_count != len(source["annotations"]):
        raise ValueError(
            f"{len(source['annotations']) - mask_count} COCO instances lack usable masks"
        )

    return {
        "converter_input": str(instance_json),
        "converter_output": str(odvg_jsonl),
        "label_map": str(label_map_path),
        "source_images": len(source["images"]),
        "projected_images": line_count,
        "empty_images_excluded_by_official_converter": len(source["images"]) - line_count,
        "source_annotations": len(source["annotations"]),
        "projected_annotations": projected_annotations,
        "mask_annotations_preserved_exactly": mask_count,
        "annotation_lossless": True,
        "empty_image_policy": (
            "The pinned official converter excludes images with zero instance "
            "annotations; no annotated image or annotation is excluded."
        ),
        "odvg_sha256": sha256_file(odvg_jsonl),
        "label_map_sha256": sha256_file(label_map_path),
    }


def _validate_instance_json(
    payload: dict[str, Any], image_dir: Path, split: str
) -> dict[str, Any]:
    expected = COCO_EXPECTED[split]
    images = payload["images"]
    annotations = payload["annotations"]
    categories = payload["categories"]
    if len(images) != expected["images"] or len(annotations) != expected["instance_annotations"]:
        raise ValueError(f"unexpected COCO {split} instance counts")
    image_by_id = {image["id"]: image for image in images}
    category_ids = {category["id"] for category in categories}
    if len(image_by_id) != len(images) or len(category_ids) != len(categories):
        raise ValueError(f"duplicate COCO {split} image or category IDs")
    annotation_ids: set[int] = set()
    category_counts: Counter[int] = Counter()
    masked = 0
    zero_area_annotation_ids: list[int] = []
    for annotation in annotations:
        annotation_id = annotation["id"]
        if annotation_id in annotation_ids:
            raise ValueError(f"duplicate COCO {split} annotation ID {annotation_id}")
        annotation_ids.add(annotation_id)
        image = image_by_id.get(annotation["image_id"])
        if image is None or annotation["category_id"] not in category_ids:
            raise ValueError(f"invalid COCO {split} annotation reference")
        bbox = annotation["bbox"]
        if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
            raise ValueError(f"invalid COCO {split} bbox at annotation {annotation_id}")
        if bbox[2] < 0 or bbox[3] < 0 or annotation["area"] < 0:
            raise ValueError(f"negative COCO {split} geometry at annotation {annotation_id}")
        if bbox[2] == 0 or bbox[3] == 0 or annotation["area"] == 0:
            zero_area_annotation_ids.append(annotation_id)
        if annotation.get("segmentation"):
            masked += 1
        category_counts[annotation["category_id"]] += 1
    actual_names = {path.name for path in image_dir.glob("*.jpg")}
    referenced_names = {image["file_name"] for image in images}
    if actual_names != referenced_names:
        raise ValueError(f"COCO {split} extracted image set differs from JSON")
    return {
        "images": len(images),
        "annotations": len(annotations),
        "annotations_with_masks": masked,
        "source_zero_area_annotation_ids": zero_area_annotation_ids,
        "source_zero_area_annotations_preserved": len(zero_area_annotation_ids),
        "categories": len(categories),
        "category_counts": {
            str(category_id): category_counts[category_id]
            for category_id in sorted(category_counts)
        },
    }


def _validate_panoptic_json(
    payload: dict[str, Any],
    image_dir: Path,
    panoptic_dir: Path,
    split: str,
    deep_pixel_check: bool,
) -> dict[str, Any]:
    images = payload["images"]
    annotations = payload["annotations"]
    categories = payload["categories"]
    if len(images) != COCO_EXPECTED[split]["images"] or len(annotations) != len(images):
        raise ValueError(f"unexpected COCO {split} panoptic counts")
    image_by_id = {image["id"]: image for image in images}
    annotation_by_image = {annotation["image_id"]: annotation for annotation in annotations}
    if len(image_by_id) != len(images) or len(annotation_by_image) != len(annotations):
        raise ValueError(f"duplicate COCO {split} panoptic image identity")
    category_ids = {category["id"] for category in categories}
    referenced_masks = {annotation["file_name"] for annotation in annotations}
    actual_masks = {path.name for path in panoptic_dir.glob("*.png")}
    if referenced_masks != actual_masks:
        raise ValueError(f"COCO {split} panoptic PNG set differs from JSON")

    segment_count = 0
    category_counts: Counter[int] = Counter()
    for image_id, annotation in annotation_by_image.items():
        image = image_by_id[image_id]
        image_path = image_dir / image["file_name"]
        mask_path = panoptic_dir / annotation["file_name"]
        with Image.open(image_path) as image_file:
            image_size = image_file.size
            image_file.verify()
        with Image.open(mask_path) as mask_file:
            mask_size = mask_file.size
            if deep_pixel_check:
                rgb = np.asarray(mask_file.convert("RGB"), dtype=np.uint32)
        if image_size != mask_size or image_size != (image["width"], image["height"]):
            raise ValueError(f"COCO {split} panoptic dimensions differ for image {image_id}")
        segment_ids: set[int] = set()
        area_by_id: dict[int, int] = {}
        for segment in annotation["segments_info"]:
            segment_id = segment["id"]
            if segment_id in segment_ids or segment["category_id"] not in category_ids:
                raise ValueError(f"invalid COCO {split} panoptic segment identity")
            segment_ids.add(segment_id)
            area_by_id[segment_id] = segment["area"]
            category_counts[segment["category_id"]] += 1
            segment_count += 1
        if deep_pixel_check:
            pixel_ids = rgb[:, :, 0] + 256 * rgb[:, :, 1] + 65536 * rgb[:, :, 2]
            ids, counts = np.unique(pixel_ids, return_counts=True)
            observed = {int(key): int(value) for key, value in zip(ids, counts)}
            if set(observed) - {0} != segment_ids:
                raise ValueError(f"COCO {split} panoptic mask IDs differ for image {image_id}")
            for segment_id, area in area_by_id.items():
                if observed[segment_id] != area:
                    raise ValueError(
                        f"COCO {split} panoptic area differs for segment {segment_id}"
                    )

    return {
        "images": len(images),
        "annotations": len(annotations),
        "segments": segment_count,
        "categories": len(categories),
        "category_counts": {
            str(category_id): category_counts[category_id]
            for category_id in sorted(category_counts)
        },
        "all_png_dimensions_checked": True,
        "all_segment_ids_and_areas_checked": deep_pixel_check,
    }


def _validate_unique_colors(label_map: list[dict[str, Any]], name: str) -> None:
    colors = [tuple(category["color"]) for category in label_map]
    if len(colors) != len(set(colors)) or any(
        len(color) != 3 or any(channel < 0 or channel > 255 for channel in color)
        for color in colors
    ):
        raise ValueError(f"{name} label map has invalid or duplicate colors")


def verify_coco2017(root: Path, deep_pixel_check: bool) -> dict[str, Any]:
    """Verify the complete COCO2017 instance/panoptic package and TAO assets."""
    annotations = root / "annotations"
    images = root / "images"
    report: dict[str, Any] = {"dataset_id": "coco2017_instance_panoptic"}
    split_reports: dict[str, Any] = {}
    for split in ("train", "val"):
        instance_path = annotations / f"instances_{split}2017.json"
        panoptic_path = annotations / f"panoptic_{split}2017.json"
        instance_payload = _load_json(instance_path)
        panoptic_payload = _load_json(panoptic_path)
        instance_categories = {
            (category["id"], category["name"]) for category in instance_payload["categories"]
        }
        panoptic_thing_categories = {
            (category["id"], category["name"])
            for category in panoptic_payload["categories"]
            if category["isthing"]
        }
        if instance_categories != panoptic_thing_categories:
            raise ValueError(f"COCO {split} instance/thing category mappings differ")
        split_reports[split] = {
            "instance": _validate_instance_json(
                instance_payload, images / f"{split}2017", split
            ),
            "panoptic": _validate_panoptic_json(
                panoptic_payload,
                images / f"{split}2017",
                annotations / f"panoptic_{split}2017",
                split,
                deep_pixel_check,
            ),
            "instance_json_sha256": sha256_file(instance_path),
            "panoptic_json_sha256": sha256_file(panoptic_path),
        }

    train_panoptic = _load_json(annotations / "panoptic_train2017.json")
    label_map_path = root / "tao" / "label_map_panoptic.json"
    label_map = _load_json(label_map_path)
    panoptic_categories = train_panoptic["categories"]
    comparable_keys = ("id", "name", "isthing")
    normalized_panoptic = [
        {key: category[key] for key in comparable_keys} for category in panoptic_categories
    ]
    normalized_label_map = [
        {key: category[key] for key in comparable_keys} for category in label_map
    ]
    if normalized_label_map != normalized_panoptic:
        raise ValueError("TAO panoptic label map differs from official COCO categories")
    _validate_unique_colors(label_map, "TAO panoptic")

    train_instances = _load_json(annotations / "instances_train2017.json")
    instance_label_map_path = root / "tao" / "label_map_instance.json"
    instance_label_map = _load_json(instance_label_map_path)
    expected_instance_identity = [
        {"id": category["id"], "name": category["name"], "isthing": 1}
        for category in train_instances["categories"]
    ]
    actual_instance_identity = [
        {
            "id": category["id"],
            "name": category["name"],
            "isthing": category["isthing"],
        }
        for category in instance_label_map
    ]
    if actual_instance_identity != expected_instance_identity:
        raise ValueError("TAO instance label map differs from official COCO categories")
    _validate_unique_colors(instance_label_map, "TAO instance")

    odvg = verify_odvg_projection(
        annotations / "instances_train2017.json",
        root
        / "tao"
        / "mask_grounding_dino"
        / "train"
        / "instances_train2017_odvg.jsonl",
        root
        / "tao"
        / "mask_grounding_dino"
        / "train"
        / "instances_train2017_odvg_labelmap.json",
    )
    archive_hashes = {}
    for archive in sorted((root / "source_archives").glob("*.zip")):
        archive_hashes[archive.name] = {
            "size_bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
        }
    report.update(
        {
            "splits": split_reports,
            "panoptic_label_map": {
                "path": str(label_map_path),
                "categories": len(label_map),
                "sha256": sha256_file(label_map_path),
                "matches_official_panoptic_categories": True,
            },
            "instance_label_map": {
                "path": str(instance_label_map_path),
                "categories": len(instance_label_map),
                "sha256": sha256_file(instance_label_map_path),
                "matches_official_instance_categories": True,
            },
            "mask_grounding_dino_odvg": odvg,
            "source_archives": archive_hashes,
            "deep_panoptic_pixel_check": deep_pixel_check,
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--voc-source-archive", type=Path, required=True)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deep-panoptic-pixel-check", action="store_true")
    args = parser.parse_args()

    result = {
        "schema_version": 1,
        "data_only_validation": True,
        "model_invoked": False,
        "slurm_job_submitted": False,
        "voc2012": verify_voc2012(args.voc_root, args.voc_source_archive),
        "coco2017": verify_coco2017(
            args.coco_root, args.deep_panoptic_pixel_check
        ),
    }
    result["content_sha256"] = canonical_json_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "content_sha256": result["content_sha256"]}))


if __name__ == "__main__":
    main()
