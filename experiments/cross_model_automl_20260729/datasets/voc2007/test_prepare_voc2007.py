# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for deterministic, fail-closed VOC2007 preparation."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import tarfile

import pytest


DATASET_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATASET_DIR))
import prepare_voc2007 as voc  # noqa: E402


def _jpeg(width: int = 8, height: int = 6, components: int = 3) -> bytes:
    component_data = b"".join(
        bytes((component, 0x11, 0))
        for component in range(1, components + 1)
    )
    payload = (
        bytes((8,))
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + bytes((components,))
        + component_data
    )
    return (
        b"\xff\xd8"
        + b"\xff\xc0"
        + (len(payload) + 2).to_bytes(2, "big")
        + payload
        + b"\xff\xd9"
    )


def _object_xml(
    *,
    category: str,
    bbox: tuple[int, int, int, int],
    difficult: int,
    parts: tuple[
        tuple[str, tuple[int | float, int | float, int | float, int | float]],
        ...,
    ] = (),
) -> str:
    xmin, ymin, xmax, ymax = bbox
    part_xml = "".join(
        (
            f"<part><name>{name}</name><bndbox>"
            f"<xmin>{part_bbox[0]}</xmin><ymin>{part_bbox[1]}</ymin>"
            f"<xmax>{part_bbox[2]}</xmax><ymax>{part_bbox[3]}</ymax>"
            "</bndbox></part>"
        )
        for name, part_bbox in parts
    )
    return (
        "<object>"
        f"<name>{category}</name>"
        "<pose>Unspecified</pose>"
        "<truncated>0</truncated>"
        f"<difficult>{difficult}</difficult>"
        "<bndbox>"
        f"<xmin>{xmin}</xmin><ymin>{ymin}</ymin>"
        f"<xmax>{xmax}</xmax><ymax>{ymax}</ymax>"
        "</bndbox>"
        f"{part_xml}"
        "</object>"
    )


def _annotation_xml(
    identifier: str,
    objects: tuple[str, ...],
    *,
    width: int = 8,
    height: int = 6,
    depth: int = 3,
) -> bytes:
    return (
        "<annotation>"
        "<folder>VOC2007</folder>"
        f"<filename>{identifier}.jpg</filename>"
        "<source><database>The VOC2007 Database</database></source>"
        f"<size><width>{width}</width><height>{height}</height>"
        f"<depth>{depth}</depth></size>"
        "<segmented>0</segmented>"
        + "".join(objects)
        + "</annotation>"
    ).encode("ascii")


def _tar(path: Path, files: dict[str, bytes], *, link: str | None = None) -> None:
    with tarfile.open(path, "w") as archive:
        for name in sorted(files):
            content = files[name]
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            archive.addfile(info, BytesIO(content))
        if link is not None:
            info = tarfile.TarInfo(
                name="VOCdevkit/VOC2007/JPEGImages/unsafe.jpg"
            )
            info.type = tarfile.SYMTYPE
            info.linkname = link
            info.mtime = 0
            archive.addfile(info)


def _base_source_files() -> tuple[dict[str, bytes], dict[str, bytes]]:
    prefix = "VOCdevkit/VOC2007"
    first = _object_xml(
        category="aeroplane",
        bbox=(1, 2, 4, 5),
        difficult=1,
    )
    second = _object_xml(
        category="person",
        bbox=(2, 1, 8, 6),
        difficult=0,
        parts=(("head", (2, 1, 4, 3)),),
    )
    third = _object_xml(
        category="aeroplane",
        bbox=(1, 1, 8, 6),
        difficult=0,
    )
    trainval = {
        f"{prefix}/ImageSets/Main/train.txt": b"000001\n",
        f"{prefix}/ImageSets/Main/val.txt": b"000002\n",
        f"{prefix}/ImageSets/Main/trainval.txt": b"000001\n000002\n",
        f"{prefix}/JPEGImages/000001.jpg": _jpeg(),
        f"{prefix}/JPEGImages/000002.jpg": _jpeg(),
        f"{prefix}/Annotations/000001.xml": _annotation_xml(
            "000001",
            (first,),
        ),
        f"{prefix}/Annotations/000002.xml": _annotation_xml(
            "000002",
            (second,),
        ),
    }
    test = {
        f"{prefix}/ImageSets/Main/test.txt": b"000003\n",
        f"{prefix}/JPEGImages/000003.jpg": _jpeg(),
        f"{prefix}/Annotations/000003.xml": _annotation_xml(
            "000003",
            (third,),
        ),
    }
    return trainval, test


def _write_fixture(
    tmp_path: Path,
    *,
    mutate_trainval=None,
    mutate_test=None,
    trainval_link: str | None = None,
    test_link: str | None = None,
) -> tuple[Path, Path]:
    archives_dir = tmp_path / "archives"
    archives_dir.mkdir()
    trainval_files, test_files = _base_source_files()
    if mutate_trainval is not None:
        mutate_trainval(trainval_files)
    if mutate_test is not None:
        mutate_test(test_files)
    trainval_archive = archives_dir / "VOCtrainval_06-Nov-2007.tar"
    test_archive = archives_dir / "VOCtest_06-Nov-2007.tar"
    _tar(trainval_archive, trainval_files, link=trainval_link)
    _tar(test_archive, test_files, link=test_link)

    manifest = json.loads(
        (DATASET_DIR / "manifest.v1.json").read_text(encoding="utf-8")
    )
    manifest["expected_counts"] = {
        "train": {
            "images": 1,
            "objects_non_difficult": 0,
            "objects_total": None,
        },
        "val": {
            "images": 1,
            "objects_non_difficult": 1,
            "objects_total": None,
        },
        "trainval": {
            "images": 2,
            "objects_non_difficult": 1,
            "objects_total": None,
            "category_objects_non_difficult": {
                category: (
                    1 if category == "person" else 0
                )
                for category in manifest["categories"]
            },
        },
        "test": {
            "images": 1,
            "objects_non_difficult": 1,
            "objects_total": None,
            "count_derivation": "unit-test fixture",
        },
        "combined": {
            "images": 3,
            "objects_non_difficult": 2,
            "objects_total": None,
        },
    }
    for archive in manifest["archives"]:
        path = archives_dir / archive["filename"]
        archive["expected_size_bytes"] = path.stat().st_size
        archive["checksums"] = [{
            "algorithm": "md5",
            "value": hashlib.md5(
                path.read_bytes(),
                usedforsecurity=False,
            ).hexdigest(),
            "source": {
                "provider": "unit-test fixture",
                "url": "https://example.invalid/unit-test",
                "version": "1",
            },
        }]
    manifest["security_limits"] = {
        "maximum_archive_members": 100,
        "maximum_member_bytes": 1024 * 1024,
        "maximum_total_uncompressed_bytes": 10 * 1024 * 1024,
        "maximum_xml_bytes": 1024 * 1024,
    }
    manifest_path = tmp_path / "manifest.v1.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, archives_dir


def _error_code(exc_info) -> str:
    return exc_info.value.code


def test_frozen_repository_manifest_is_valid_and_explicit():
    manifest = voc.load_manifest(DATASET_DIR / "manifest.v1.json")

    assert manifest["models"] == ["dino", "deformable_detr", "rtdetr"]
    assert manifest["scope"]["task"] == "object_detection"
    assert manifest["expected_counts"]["combined"] == {
        "images": 9963,
        "objects_non_difficult": 24640,
        "objects_total": None,
    }
    assert manifest["license_and_terms"]["dataset_wide_spdx_license"] is None
    assert (
        manifest["license_and_terms"]["license_status"]
        == "source_image_rights_vary_manual_review_required"
    )
    assert all(
        archive["url"].startswith(
            "https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/"
        )
        for archive in manifest["archives"]
    )
    assert all(
        archive["sha256_status"]
        == "not_published_by_the_official_source_or_pinned_checksum_source"
        for archive in manifest["archives"]
    )


def test_successful_preparation_preserves_ids_bboxes_flags_and_parts(
    tmp_path: Path,
):
    manifest_path, archives_dir = _write_fixture(tmp_path)
    output = tmp_path / "prepared"

    result = voc.prepare_dataset(
        manifest_path=manifest_path,
        archives_dir=archives_dir,
        output=output,
        accept_terms=True,
    )

    assert result["status"] == "prepared"
    trainval = json.loads(
        (
            output
            / "coco/annotations/instances_trainval2007.json"
        ).read_text(encoding="utf-8")
    )
    assert [image["id"] for image in trainval["images"]] == [1, 2]
    assert [item["name"] for item in trainval["categories"]] == json.loads(
        manifest_path.read_text(encoding="utf-8")
    )["categories"]
    first, second = trainval["annotations"]
    assert first["image_id"] == 1
    assert first["category_id"] == 1
    assert first["voc_bbox"] == [1, 2, 4, 5]
    assert first["bbox"] == [0, 1, 4, 4]
    assert first["area"] == 16
    assert first["difficult"] == 1
    assert first["iscrowd"] == 0
    assert second["voc_metadata"]["parts"] == [{
        "name": "head",
        "voc_bbox": [2, 1, 4, 3],
    }]
    integrity = json.loads(
        (output / voc.INTEGRITY_FILENAME).read_text(encoding="utf-8")
    )
    assert integrity["network_access_by_preparer"] is False
    assert integrity["validation"]["invariants"] == {
        "all_bboxes_reversible": True,
        "all_categories_mapped": True,
        "all_difficult_flags_preserved": True,
        "all_images_preserved": True,
        "all_objects_preserved": True,
        "jpeg_dimensions_verified": True,
        "source_inventory_exact": True,
        "train_val_disjoint": True,
        "trainval_test_disjoint": True,
    }
    assert {
        item["role"] for item in integrity["archives"]
    } == {"trainval", "test"}
    assert all(
        len(item["computed_checksums"]["sha256"]) == 64
        for item in integrity["archives"]
    )


def test_decimal_voc_part_boxes_are_preserved_without_rounding(tmp_path: Path):
    def decimal_part(trainval_files):
        path = "VOCdevkit/VOC2007/Annotations/000002.xml"
        trainval_files[path] = _annotation_xml(
            "000002",
            (
                _object_xml(
                    category="person",
                    bbox=(2, 1, 8, 6),
                    difficult=0,
                    parts=(("head", (2.25, 1.5, 4.75, 3.125)),),
                ),
            ),
        )

    manifest_path, archives_dir = _write_fixture(
        tmp_path,
        mutate_trainval=decimal_part,
    )
    output = tmp_path / "prepared"
    voc.prepare_dataset(
        manifest_path=manifest_path,
        archives_dir=archives_dir,
        output=output,
        accept_terms=True,
    )

    trainval = json.loads(
        (
            output
            / "coco/annotations/instances_trainval2007.json"
        ).read_text(encoding="utf-8")
    )
    person = next(
        annotation
        for annotation in trainval["annotations"]
        if annotation["category_id"] == 15
    )
    assert person["voc_metadata"]["parts"] == [{
        "name": "head",
        "voc_bbox": [2.25, 1.5, 4.75, 3.125],
    }]
    validation = voc.validate_prepared_dataset(
        manifest_path=manifest_path,
        dataset_root=output,
    )
    assert validation["status"] == "valid"


def test_conversion_is_byte_deterministic_across_independent_outputs(
    tmp_path: Path,
):
    manifest_path, archives_dir = _write_fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    for output in (first, second):
        voc.prepare_dataset(
            manifest_path=manifest_path,
            archives_dir=archives_dir,
            output=output,
            accept_terms=True,
        )

    artifact_names = (
        "coco/annotations/instances_train2007.json",
        "coco/annotations/instances_val2007.json",
        "coco/annotations/instances_trainval2007.json",
        "coco/annotations/instances_test2007.json",
        voc.INTEGRITY_FILENAME,
        voc.INTEGRITY_DIGEST_FILENAME,
    )
    for relative in artifact_names:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_preparation_requires_explicit_terms_acknowledgement(tmp_path: Path):
    manifest_path, archives_dir = _write_fixture(tmp_path)
    output = tmp_path / "prepared"

    with pytest.raises(voc.DatasetPreparationError) as exc_info:
        voc.prepare_dataset(
            manifest_path=manifest_path,
            archives_dir=archives_dir,
            output=output,
            accept_terms=False,
        )

    assert _error_code(exc_info) == "dataset_terms_not_acknowledged"
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path: Path):
    manifest_path, archives_dir = _write_fixture(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    marker = output / "owned-by-user"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(voc.DatasetPreparationError) as exc_info:
        voc.prepare_dataset(
            manifest_path=manifest_path,
            archives_dir=archives_dir,
            output=output,
            accept_terms=True,
        )

    assert _error_code(exc_info) == "output_already_exists"
    assert marker.read_text(encoding="utf-8") == "keep"


def test_partial_archive_is_refused(tmp_path: Path):
    manifest_path, archives_dir = _write_fixture(tmp_path)
    test_archive = archives_dir / "VOCtest_06-Nov-2007.tar"
    test_archive.rename(test_archive.with_suffix(".tar.part"))
    manifest = voc.load_manifest(manifest_path)

    with pytest.raises(voc.DatasetPreparationError) as exc_info:
        voc.verify_archives(manifest, archives_dir)

    assert _error_code(exc_info) == "partial_archive_refused"


def test_archive_checksum_mismatch_is_refused(tmp_path: Path):
    manifest_path, archives_dir = _write_fixture(tmp_path)
    archive = archives_dir / "VOCtest_06-Nov-2007.tar"
    with archive.open("ab") as stream:
        stream.write(b"partial-or-tampered")
    manifest = voc.load_manifest(manifest_path)

    with pytest.raises(voc.DatasetPreparationError) as exc_info:
        voc.verify_archives(manifest, archives_dir)

    assert _error_code(exc_info) in {
        "archive_size_mismatch",
        "archive_checksum_mismatch",
    }


@pytest.mark.parametrize(
    ("malicious_name", "expected_code"),
    [
        ("../escaped", "archive_path_traversal_refused"),
        (
            "VOCdevkit/VOC2007/../../escaped",
            "archive_path_traversal_refused",
        ),
        ("/absolute/path", "archive_path_traversal_refused"),
    ],
)
def test_archive_path_traversal_is_refused_atomically(
    tmp_path: Path,
    malicious_name: str,
    expected_code: str,
):
    def mutate(files):
        files[malicious_name] = b"unsafe"

    manifest_path, archives_dir = _write_fixture(
        tmp_path,
        mutate_trainval=mutate,
    )
    output = tmp_path / "prepared"

    with pytest.raises(voc.DatasetPreparationError) as exc_info:
        voc.prepare_dataset(
            manifest_path=manifest_path,
            archives_dir=archives_dir,
            output=output,
            accept_terms=True,
        )

    assert _error_code(exc_info) == expected_code
    assert not output.exists()
    assert not (tmp_path / "escaped").exists()


def test_archive_symlink_member_is_refused_atomically(tmp_path: Path):
    manifest_path, archives_dir = _write_fixture(
        tmp_path,
        trainval_link="../../outside",
    )
    output = tmp_path / "prepared"

    with pytest.raises(voc.DatasetPreparationError) as exc_info:
        voc.prepare_dataset(
            manifest_path=manifest_path,
            archives_dir=archives_dir,
            output=output,
            accept_terms=True,
        )

    assert _error_code(exc_info) == "archive_unsafe_member_type"
    assert not output.exists()


def test_missing_source_image_is_detected_without_partial_publication(
    tmp_path: Path,
):
    def mutate(files):
        del files["VOCdevkit/VOC2007/JPEGImages/000002.jpg"]

    manifest_path, archives_dir = _write_fixture(
        tmp_path,
        mutate_trainval=mutate,
    )
    output = tmp_path / "prepared"

    with pytest.raises(voc.DatasetPreparationError) as exc_info:
        voc.prepare_dataset(
            manifest_path=manifest_path,
            archives_dir=archives_dir,
            output=output,
            accept_terms=True,
        )

    assert _error_code(exc_info) == "image_inventory_mismatch"
    assert not output.exists()


def test_unknown_category_is_detected_as_a_silent_loss_risk(tmp_path: Path):
    def mutate(files):
        path = "VOCdevkit/VOC2007/Annotations/000001.xml"
        files[path] = files[path].replace(b"aeroplane", b"spaceship")

    manifest_path, archives_dir = _write_fixture(
        tmp_path,
        mutate_trainval=mutate,
    )

    with pytest.raises(voc.DatasetPreparationError) as exc_info:
        voc.prepare_dataset(
            manifest_path=manifest_path,
            archives_dir=archives_dir,
            output=tmp_path / "prepared",
            accept_terms=True,
        )

    assert _error_code(exc_info) == "annotation_category_unknown"


def test_invalid_bbox_and_difficult_flag_are_rejected(tmp_path: Path):
    def mutate(files):
        path = "VOCdevkit/VOC2007/Annotations/000001.xml"
        files[path] = files[path].replace(
            b"<xmax>4</xmax>",
            b"<xmax>9</xmax>",
        ).replace(
            b"<difficult>1</difficult>",
            b"<difficult>2</difficult>",
        )

    manifest_path, archives_dir = _write_fixture(
        tmp_path,
        mutate_trainval=mutate,
    )

    with pytest.raises(voc.DatasetPreparationError) as exc_info:
        voc.prepare_dataset(
            manifest_path=manifest_path,
            archives_dir=archives_dir,
            output=tmp_path / "prepared",
            accept_terms=True,
        )

    assert _error_code(exc_info) in {
        "bbox_out_of_bounds",
        "xml_binary_flag_invalid",
    }


def test_trainval_test_overlap_is_rejected(tmp_path: Path):
    def mutate(files):
        files["VOCdevkit/VOC2007/ImageSets/Main/test.txt"] = b"000002\n"
        del files["VOCdevkit/VOC2007/JPEGImages/000003.jpg"]
        del files["VOCdevkit/VOC2007/Annotations/000003.xml"]

    manifest_path, archives_dir = _write_fixture(
        tmp_path,
        mutate_test=mutate,
    )

    with pytest.raises(voc.DatasetPreparationError) as exc_info:
        voc.prepare_dataset(
            manifest_path=manifest_path,
            archives_dir=archives_dir,
            output=tmp_path / "prepared",
            accept_terms=True,
        )

    assert _error_code(exc_info) == "trainval_test_overlap"


def test_extra_source_object_cannot_be_silently_dropped(tmp_path: Path):
    def mutate(files):
        path = "VOCdevkit/VOC2007/Annotations/000001.xml"
        extra = _object_xml(
            category="bird",
            bbox=(1, 1, 2, 2),
            difficult=0,
        ).encode("ascii")
        files[path] = files[path].replace(b"</annotation>", extra + b"</annotation>")

    manifest_path, archives_dir = _write_fixture(
        tmp_path,
        mutate_trainval=mutate,
    )

    with pytest.raises(voc.DatasetPreparationError) as exc_info:
        voc.prepare_dataset(
            manifest_path=manifest_path,
            archives_dir=archives_dir,
            output=tmp_path / "prepared",
            accept_terms=True,
        )

    assert (
        _error_code(exc_info)
        == "converted_non_difficult_object_count_mismatch"
    )


def test_prepared_output_tampering_is_detected(tmp_path: Path):
    manifest_path, archives_dir = _write_fixture(tmp_path)
    output = tmp_path / "prepared"
    voc.prepare_dataset(
        manifest_path=manifest_path,
        archives_dir=archives_dir,
        output=output,
        accept_terms=True,
    )
    annotation_path = (
        output / "coco/annotations/instances_test2007.json"
    )
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["annotations"][0]["difficult"] = 1
    annotation_path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(voc.DatasetPreparationError) as exc_info:
        voc.validate_prepared_dataset(
            manifest_path=manifest_path,
            dataset_root=output,
        )

    assert _error_code(exc_info) == "prepared_output_semantic_mismatch"


def test_manifest_refuses_invented_license_or_unfrozen_policy(tmp_path: Path):
    manifest = json.loads(
        (DATASET_DIR / "manifest.v1.json").read_text(encoding="utf-8")
    )
    for field, value, expected_code in (
        (
            ("frozen",),
            False,
            "manifest_not_frozen",
        ),
        (
            ("license_and_terms", "dataset_wide_spdx_license"),
            "CC-BY-4.0",
            "manifest_license_claim_invalid",
        ),
    ):
        changed = deepcopy(manifest)
        target = changed
        for key in field[:-1]:
            target = target[key]
        target[field[-1]] = value
        path = tmp_path / f"{expected_code}.json"
        path.write_text(json.dumps(changed), encoding="utf-8")

        with pytest.raises(voc.DatasetPreparationError) as exc_info:
            voc.load_manifest(path)

        assert _error_code(exc_info) == expected_code
