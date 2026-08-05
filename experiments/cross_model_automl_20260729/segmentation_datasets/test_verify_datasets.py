"""Data-only tests for the frozen segmentation dataset verifier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from .verify_datasets import (
    _rounded_xyxy,
    _split_ids,
    canonical_json_sha256,
    sha256_file,
    verify_odvg_projection,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _odvg_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = {
        "images": [
            {"id": 11, "file_name": "a.jpg", "height": 10, "width": 20},
            {"id": 12, "file_name": "empty.jpg", "height": 10, "width": 20},
        ],
        "categories": [
            {"id": 3, "name": "alpha"},
            {"id": 9, "name": "beta"},
        ],
        "annotations": [
            {
                "id": 101,
                "image_id": 11,
                "category_id": 9,
                "bbox": [1.234, 2.345, 4.567, 5.678],
                "segmentation": [[1, 2, 3, 4, 5, 6]],
            },
            {
                "id": 102,
                "image_id": 11,
                "category_id": 3,
                "bbox": [0, 0, 2, 3],
                "segmentation": {"size": [10, 20], "counts": "abc"},
            },
        ],
    }
    instance_path = tmp_path / "instances.json"
    _write_json(instance_path, source)
    label_map_path = tmp_path / "labelmap.json"
    _write_json(label_map_path, {"0": "alpha", "1": "beta"})
    odvg_path = tmp_path / "records.jsonl"
    record = {
        "file_name": "a.jpg",
        "height": 10,
        "width": 20,
        "image_id": 11,
        "detection": {
            "instances": [
                {
                    "bbox": [1.23, 2.35, 5.8, 8.02],
                    "label": 1,
                    "category": "beta",
                    "mask": [[1, 2, 3, 4, 5, 6]],
                },
                {
                    "bbox": [0, 0, 2, 3],
                    "label": 0,
                    "category": "alpha",
                    "mask": {"size": [10, 20], "counts": "abc"},
                },
            ]
        },
    }
    odvg_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return instance_path, odvg_path, label_map_path


def test_canonical_json_hash_is_mapping_order_invariant() -> None:
    assert canonical_json_sha256({"b": 2, "a": 1}) == canonical_json_sha256(
        {"a": 1, "b": 2}
    )


def test_converter_bbox_rounding_contract() -> None:
    assert _rounded_xyxy([1.234, 2.345, 4.567, 5.678]) == [
        1.23,
        2.35,
        5.8,
        8.02,
    ]


def test_duplicate_split_identity_is_rejected(tmp_path: Path) -> None:
    split = tmp_path / "train.txt"
    split.write_text("one\none\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate split IDs"):
        _split_ids(split)


def test_odvg_projection_preserves_every_mask_and_records_empty_images(
    tmp_path: Path,
) -> None:
    instance_path, odvg_path, label_map_path = _odvg_fixture(tmp_path)
    report = verify_odvg_projection(instance_path, odvg_path, label_map_path)
    assert report["source_images"] == 2
    assert report["projected_images"] == 1
    assert report["empty_images_excluded_by_official_converter"] == 1
    assert report["source_annotations"] == 2
    assert report["projected_annotations"] == 2
    assert report["mask_annotations_preserved_exactly"] == 2
    assert report["annotation_lossless"] is True


def test_odvg_projection_rejects_mask_mutation(tmp_path: Path) -> None:
    instance_path, odvg_path, label_map_path = _odvg_fixture(tmp_path)
    record = json.loads(odvg_path.read_text(encoding="utf-8"))
    record["detection"]["instances"][0]["mask"] = [[1, 2, 3, 4, 9, 9]]
    odvg_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ODVG annotation mismatch"):
        verify_odvg_projection(instance_path, odvg_path, label_map_path)


def test_committed_voc_profile_preserves_ignore_label() -> None:
    profile_path = Path(__file__).with_name("voc2012_segformer_dataset_profile.yaml")
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    segment = profile["dataset"]["segment"]
    label_ids = [entry["label_id"] for entry in segment["palette"]]
    colors = [entry["rgb"] for entry in segment["palette"]]
    assert segment["num_classes"] == 21
    assert segment["label_transform"] == "None"
    assert label_ids == [*range(21), 255]
    assert colors == [[label] for label in label_ids]


def test_committed_validation_report_is_self_consistent() -> None:
    report_path = Path(__file__).with_name("segmentation_dataset_validation.v1.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    content_hash = report.pop("content_sha256")
    assert canonical_json_sha256(report) == content_hash
    assert report["data_only_validation"] is True
    assert report["model_invoked"] is False
    assert report["slurm_job_submitted"] is False
    assert report["coco2017"]["deep_panoptic_pixel_check"] is True
    assert report["coco2017"]["instance_label_map"]["categories"] == 80
    assert report["coco2017"]["instance_label_map"][
        "matches_official_instance_categories"
    ] is True
    assert report["coco2017"]["mask_grounding_dino_odvg"][
        "annotation_lossless"
    ] is True


def test_committed_coco_bindings_keep_model_contracts_separate() -> None:
    bindings_path = Path(__file__).with_name("coco2017_tao_dataset_bindings.yaml")
    bindings = yaml.safe_load(bindings_path.read_text(encoding="utf-8"))
    oneformer = bindings["oneformer"]["dataset"]
    mask2former = bindings["mask2former"]["dataset"]
    mask_grounding_dino = bindings["mask_grounding_dino"]
    assert oneformer["contiguous_id"] is True
    assert oneformer["train"]["annotations"].endswith("panoptic_train2017.json")
    assert bindings["mask2former"]["task_scope"] == "instance_segmentation"
    assert mask2former["train"]["type"] == "coco"
    assert mask2former["train"]["instance_json"].endswith(
        "instances_train2017.json"
    )
    assert mask2former["label_map"].endswith("label_map_instance.json")
    assert (
        mask_grounding_dino["task_scope"]
        == "category_prompted_grounded_instance_segmentation"
    )
    assert (
        mask_grounding_dino["dataset"]["val_data_sources"]["data_type"] == "OD"
    )


def test_stage_manifest_hash_links_committed_evidence() -> None:
    evidence_root = Path(__file__).parent
    repository_root = evidence_root.parents[2]
    manifest = json.loads(
        (evidence_root / "dataset_stage_manifest.v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["schema_version"] == 1
    assert manifest["execution_contract"] == {
        "data_only": True,
        "model_invoked": False,
        "cpu_model_smoke_run": False,
        "gpu_model_smoke_run": False,
        "training_run": False,
        "evaluation_run": False,
        "latency_benchmark_run": False,
        "slurm_job_submitted": False,
    }
    validation = manifest["validation"]
    assert sha256_file(evidence_root / "segmentation_dataset_validation.v1.json") == (
        validation["report_file_sha256"]
    )
    assert sha256_file(evidence_root / "verify_datasets.py") == validation[
        "verifier_sha256"
    ]
    for artifact in manifest["data_contract_artifacts"].values():
        assert sha256_file(repository_root / artifact["path"]) == artifact["sha256"]
    for dataset in manifest["datasets"].values():
        assert dataset["file_manifest"]["remote_sha256sum_check"] == "passed"
        assert dataset["file_manifest"]["remote_file_set_check"] == "passed"
        assert dataset["remote_read_only"] is True
        assert dataset["remote_writable_entries_after_lock"] == 0
    assert manifest["transfer_provenance"][
        "remote_bytes_verified_against_local_manifest"
    ] is True
