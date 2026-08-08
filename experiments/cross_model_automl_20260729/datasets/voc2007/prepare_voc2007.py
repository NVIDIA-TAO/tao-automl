# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Secure, deterministic VOC2007-to-COCO detection dataset preparation.

This tool intentionally has no download implementation. A user must review
the dataset card and terms, obtain both official archives, and then invoke the
checksum, preparation, and validation gates explicitly.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile
from typing import Any, BinaryIO, Iterable
import xml.etree.ElementTree as ET


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "manifest.v1.json"
INTEGRITY_FILENAME = "integrity.v1.json"
INTEGRITY_DIGEST_FILENAME = "integrity.v1.json.sha256"
_VOC_ID_RE = re.compile(r"^[0-9]{6}$")
_HEX_RE = re.compile(r"^[0-9a-f]+$")
_SUPPORTED_MODELS = ("dino", "deformable_detr", "rtdetr")
_ARCHIVE_ROLES = ("trainval", "test")
_COCO_FILENAMES = {
    "train": "coco/annotations/instances_train2007.json",
    "val": "coco/annotations/instances_val2007.json",
    "trainval": "coco/annotations/instances_trainval2007.json",
    "test": "coco/annotations/instances_test2007.json",
}
_JPEG_SOF_MARKERS = frozenset({
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
})


class DatasetPreparationError(RuntimeError):
    """Fail-closed dataset preparation or validation error."""

    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code = code
        self.details = dict(sorted(details.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "failed",
            "code": self.code,
            "message": str(self),
            "details": self.details,
        }


def _fail(code: str, message: str, **details: Any) -> None:
    raise DatasetPreparationError(code, message, **details)


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = (
        hashlib.md5(usedforsecurity=False)
        if algorithm == "md5"
        else hashlib.new(algorithm)
    )
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, content: bytes) -> None:
    """Write one file atomically without following an existing symlink."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        _fail(
            "output_symlink_refused",
            "Refusing to replace an output symlink",
            path=str(path),
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write(path, _canonical_json_bytes(value))


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(
            "manifest_field_invalid",
            "Manifest field must be an object",
            field=field,
        )
    return value


def _require_https_url(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.startswith("https://"):
        _fail(
            "manifest_url_invalid",
            "Manifest source URLs must use HTTPS",
            field=field,
            value=repr(value),
        )
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the frozen dataset contract."""
    if path.is_symlink() or not path.is_file():
        _fail(
            "manifest_not_regular_file",
            "Manifest must be a regular, non-symlink file",
            path=str(path),
        )
    path = path.resolve(strict=True)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(
            "manifest_unreadable",
            "Dataset manifest is not valid UTF-8 JSON",
            path=str(path),
            error=str(exc),
        )
    manifest = _require_mapping(manifest, "root")
    if manifest.get("schema_version") != 1:
        _fail(
            "manifest_schema_unsupported",
            "Only VOC2007 manifest schema version 1 is supported",
            actual=manifest.get("schema_version"),
        )
    if manifest.get("frozen") is not True:
        _fail(
            "manifest_not_frozen",
            "Dataset preparation requires a frozen manifest",
        )
    if manifest.get("scope", {}).get("task") != "object_detection":
        _fail(
            "manifest_scope_invalid",
            "This converter is detection-only",
            task=manifest.get("scope", {}).get("task"),
        )
    if tuple(manifest.get("models", ())) != _SUPPORTED_MODELS:
        _fail(
            "manifest_models_invalid",
            "Manifest model scope must remain DINO/DDETR/RT-DETR only",
            expected=list(_SUPPORTED_MODELS),
            actual=manifest.get("models"),
        )
    conversion = _require_mapping(
        manifest.get("conversion_contract"),
        "conversion_contract",
    )
    if conversion.get("output_annotations") != _COCO_FILENAMES:
        _fail(
            "manifest_conversion_outputs_invalid",
            "Manifest COCO output paths must match the converter contract",
            expected=_COCO_FILENAMES,
            actual=conversion.get("output_annotations"),
        )
    bbox_contract = _require_mapping(
        conversion.get("bbox"),
        "conversion_contract.bbox",
    )
    if bbox_contract.get("source_format") != (
        "one-based inclusive VOC xmin,ymin,xmax,ymax"
    ) or bbox_contract.get("formula") != {
        "height": "ymax - ymin + 1",
        "width": "xmax - xmin + 1",
        "x": "xmin - 1",
        "y": "ymin - 1",
    }:
        _fail(
            "manifest_bbox_contract_invalid",
            "Manifest bbox semantics must match the reversible converter",
        )
    difficult_contract = _require_mapping(
        conversion.get("difficult"),
        "conversion_contract.difficult",
    )
    if (
        difficult_contract.get("coco_iscrowd") != 0
        or difficult_contract.get("source_values") != [0, 1]
    ):
        _fail(
            "manifest_difficult_contract_invalid",
            "Manifest difficult/iscrowd semantics must match the converter",
        )

    categories = manifest.get("categories")
    if (
        not isinstance(categories, list)
        or not categories
        or any(not isinstance(item, str) or not item for item in categories)
        or len(categories) != len(set(categories))
    ):
        _fail(
            "manifest_categories_invalid",
            "Manifest categories must be unique non-empty strings",
        )

    counts = _require_mapping(manifest.get("expected_counts"), "expected_counts")
    for split in ("train", "val", "trainval", "test", "combined"):
        split_counts = _require_mapping(counts.get(split), f"expected_counts.{split}")
        for field in ("images", "objects_non_difficult"):
            if not _is_nonnegative_int(split_counts.get(field)):
                _fail(
                    "manifest_count_invalid",
                    "Expected counts must be non-negative integers",
                    split=split,
                    field=field,
                    actual=repr(split_counts.get(field)),
                )
        if split_counts.get("objects_total", "missing") is not None:
            _fail(
                "manifest_total_object_count_invented",
                "Total XML object counts must remain null until derived",
                split=split,
                actual=repr(split_counts.get("objects_total")),
            )
    if counts["train"]["images"] + counts["val"]["images"] != counts["trainval"]["images"]:
        _fail(
            "manifest_count_inconsistent",
            "Train and validation image totals must equal trainval",
        )
    if (
        counts["train"]["objects_non_difficult"]
        + counts["val"]["objects_non_difficult"]
        != counts["trainval"]["objects_non_difficult"]
    ):
        _fail(
            "manifest_count_inconsistent",
            "Train and validation object totals must equal trainval",
        )
    if counts["trainval"]["images"] + counts["test"]["images"] != counts["combined"]["images"]:
        _fail(
            "manifest_count_inconsistent",
            "Trainval and test image totals must equal combined",
        )
    if (
        counts["trainval"]["objects_non_difficult"]
        + counts["test"]["objects_non_difficult"]
        != counts["combined"]["objects_non_difficult"]
    ):
        _fail(
            "manifest_count_inconsistent",
            "Trainval and test object totals must equal combined",
        )
    category_counts = _require_mapping(
        counts["trainval"].get("category_objects_non_difficult"),
        "expected_counts.trainval.category_objects_non_difficult",
    )
    if set(category_counts) != set(categories):
        _fail(
            "manifest_category_counts_invalid",
            "Trainval category-count keys must match the category mapping",
        )
    if any(not _is_nonnegative_int(value) for value in category_counts.values()):
        _fail(
            "manifest_category_counts_invalid",
            "Trainval category counts must be non-negative integers",
        )
    if (
        sum(category_counts.values())
        != counts["trainval"]["objects_non_difficult"]
    ):
        _fail(
            "manifest_category_counts_invalid",
            "Trainval category counts must sum to the object total",
        )

    archives = manifest.get("archives")
    if not isinstance(archives, list) or len(archives) != 2:
        _fail(
            "manifest_archives_invalid",
            "Exactly two archive records are required",
        )
    roles = []
    filenames = []
    for index, archive_value in enumerate(archives):
        archive = _require_mapping(archive_value, f"archives[{index}]")
        role = archive.get("role")
        filename = archive.get("filename")
        if role not in _ARCHIVE_ROLES:
            _fail(
                "manifest_archive_role_invalid",
                "Archive role must be trainval or test",
                role=role,
            )
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            _fail(
                "manifest_archive_filename_invalid",
                "Archive filename must be a safe basename",
                filename=repr(filename),
            )
        _require_https_url(archive.get("url"), f"archives[{index}].url")
        _require_https_url(
            archive.get("official_download_page"),
            f"archives[{index}].official_download_page",
        )
        expected_size = archive.get("expected_size_bytes")
        if expected_size is not None and (
            not _is_nonnegative_int(expected_size) or expected_size == 0
        ):
            _fail(
                "manifest_archive_size_invalid",
                "Expected archive size must be null or a positive integer",
                role=role,
            )
        checksums = archive.get("checksums")
        if not isinstance(checksums, list) or not checksums:
            _fail(
                "manifest_checksum_missing",
                "Every archive requires a published checksum",
                role=role,
            )
        checksum_algorithms = set()
        for checksum_value in checksums:
            checksum = _require_mapping(
                checksum_value,
                f"archives[{index}].checksums",
            )
            algorithm = checksum.get("algorithm")
            expected_length = {"md5": 32, "sha256": 64}.get(algorithm)
            value = checksum.get("value")
            if (
                expected_length is None
                or not isinstance(value, str)
                or len(value) != expected_length
                or _HEX_RE.fullmatch(value) is None
            ):
                _fail(
                    "manifest_checksum_invalid",
                    "Checksum must be lowercase hexadecimal MD5 or SHA-256",
                    role=role,
                    algorithm=algorithm,
                    value=repr(value),
                )
            source = _require_mapping(
                checksum.get("source"),
                f"archives[{index}].checksums.source",
            )
            _require_https_url(
                source.get("url"),
                f"archives[{index}].checksums.source.url",
            )
            if not source.get("provider") or not source.get("version"):
                _fail(
                    "manifest_checksum_source_invalid",
                    "Checksum provenance must identify provider and version",
                    role=role,
                )
            if algorithm in checksum_algorithms:
                _fail(
                    "manifest_checksum_duplicate",
                    "Checksum algorithms cannot be repeated",
                    role=role,
                    algorithm=algorithm,
                )
            checksum_algorithms.add(algorithm)
        roles.append(role)
        filenames.append(filename)
    if tuple(sorted(roles)) != tuple(sorted(_ARCHIVE_ROLES)):
        _fail(
            "manifest_archive_roles_incomplete",
            "Trainval and test archives are both required",
        )
    if len(filenames) != len(set(filenames)):
        _fail(
            "manifest_archive_filename_duplicate",
            "Archive filenames must be unique",
        )

    limits = _require_mapping(manifest.get("security_limits"), "security_limits")
    for name in (
        "maximum_archive_members",
        "maximum_member_bytes",
        "maximum_total_uncompressed_bytes",
        "maximum_xml_bytes",
    ):
        if not _is_nonnegative_int(limits.get(name)) or limits[name] == 0:
            _fail(
                "manifest_security_limit_invalid",
                "Security limits must be positive integers",
                field=name,
            )
    terms = _require_mapping(
        manifest.get("license_and_terms"),
        "license_and_terms",
    )
    if terms.get("preparation_requires_explicit_terms_acknowledgement") is not True:
        _fail(
            "manifest_terms_gate_missing",
            "VOC preparation must retain an explicit terms gate",
        )
    if terms.get("dataset_wide_spdx_license", "missing") is not None:
        _fail(
            "manifest_license_claim_invalid",
            "This manifest must not assert a dataset-wide SPDX license",
        )
    return manifest


def _archive_by_role(
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {archive["role"]: archive for archive in manifest["archives"]}


def verify_archives(
    manifest: dict[str, Any],
    archives_dir: Path,
) -> list[dict[str, Any]]:
    """Verify both archive identities without extracting them."""
    if archives_dir.is_symlink() or not archives_dir.is_dir():
        _fail(
            "archives_directory_invalid",
            "Archive directory must be a non-symlink directory",
            path=str(archives_dir),
        )
    archives_dir = archives_dir.resolve(strict=True)
    records = []
    for role in _ARCHIVE_ROLES:
        archive = _archive_by_role(manifest)[role]
        filename = archive["filename"]
        path = archives_dir / filename
        if not path.exists():
            partials = sorted(
                item.name
                for item in archives_dir.iterdir()
                if item.name.startswith(filename)
                and (
                    item.name.endswith(".part")
                    or item.name.endswith(".partial")
                    or item.name.endswith(".tmp")
                )
            )
            if partials:
                _fail(
                    "partial_archive_refused",
                    "Only a partial archive is present; preparation is refused",
                    role=role,
                    expected=filename,
                    partials=partials,
                )
            _fail(
                "archive_missing",
                "Required official archive is missing",
                role=role,
                expected=filename,
            )
        if path.is_symlink() or not path.is_file():
            _fail(
                "archive_not_regular_file",
                "Archive must be a regular, non-symlink file",
                role=role,
                path=str(path),
            )
        expected_size = archive.get("expected_size_bytes")
        actual_size = path.stat().st_size
        if expected_size is not None and actual_size != expected_size:
            _fail(
                "archive_size_mismatch",
                "Archive size differs from the published identity",
                role=role,
                expected=expected_size,
                actual=actual_size,
            )
        algorithms = {
            checksum["algorithm"] for checksum in archive["checksums"]
        }
        algorithms.add("sha256")
        computed = {
            algorithm: _hash_file(path, algorithm)
            for algorithm in sorted(algorithms)
        }
        for checksum in archive["checksums"]:
            algorithm = checksum["algorithm"]
            if computed[algorithm] != checksum["value"]:
                _fail(
                    "archive_checksum_mismatch",
                    "Archive checksum differs from the published identity",
                    role=role,
                    algorithm=algorithm,
                    expected=checksum["value"],
                    actual=computed[algorithm],
                )
        records.append({
            "role": role,
            "filename": filename,
            "size_bytes": actual_size,
            "computed_checksums": computed,
            "published_checksums_verified": True,
            "source_url": archive["url"],
        })
    return records


def _validate_tar_member_name(member: tarfile.TarInfo) -> PurePosixPath:
    name = member.name.rstrip("/")
    if not name or "\x00" in name or "\\" in name:
        _fail(
            "archive_member_path_invalid",
            "Archive member has an invalid path",
            member=repr(member.name),
        )
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        _fail(
            "archive_path_traversal_refused",
            "Archive member attempts path traversal",
            member=member.name,
        )
    allowed_ancestors = {
        ("VOCdevkit",),
        ("VOCdevkit", "VOC2007"),
    }
    if path.parts in allowed_ancestors:
        if not member.isdir():
            _fail(
                "archive_member_outside_dataset",
                "Only directory ancestors may precede VOC2007 content",
                member=member.name,
            )
    elif path.parts[:2] != ("VOCdevkit", "VOC2007"):
        _fail(
            "archive_member_outside_dataset",
            "Archive member is outside VOCdevkit/VOC2007",
            member=member.name,
        )
    return path


def _copy_exact(
    source: BinaryIO,
    destination: BinaryIO,
    expected_bytes: int,
) -> None:
    remaining = expected_bytes
    while remaining:
        chunk = source.read(min(1024 * 1024, remaining))
        if not chunk:
            _fail(
                "archive_member_truncated",
                "Archive member ended before its declared size",
                expected_bytes=expected_bytes,
                missing_bytes=remaining,
            )
        destination.write(chunk)
        remaining -= len(chunk)
    if source.read(1):
        _fail(
            "archive_member_size_inconsistent",
            "Archive member contains bytes beyond its declared size",
        )


def _extract_one_archive(
    archive_path: Path,
    destination: Path,
    limits: dict[str, int],
) -> dict[str, int]:
    """Extract regular files/directories only, with no path following."""
    member_count = 0
    total_bytes = 0
    try:
        archive = tarfile.open(archive_path, mode="r:*")
    except (tarfile.TarError, OSError) as exc:
        _fail(
            "archive_unreadable",
            "Archive is not a readable tar file",
            path=str(archive_path),
            error=str(exc),
        )
    with archive:
        validated: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
        seen_paths: set[PurePosixPath] = set()
        for member_index, member in enumerate(archive, start=1):
            if member_index > limits["maximum_archive_members"]:
                _fail(
                    "archive_member_limit_exceeded",
                    "Archive has too many members",
                    path=str(archive_path),
                    actual=member_index,
                    maximum=limits["maximum_archive_members"],
                )
            path = _validate_tar_member_name(member)
            if path in seen_paths:
                _fail(
                    "archive_duplicate_member",
                    "Archive contains duplicate member paths",
                    path=str(archive_path),
                    member=member.name,
                )
            seen_paths.add(path)
            if not member.isdir() and not member.isreg():
                _fail(
                    "archive_unsafe_member_type",
                    "Links, devices, FIFOs, and special members are refused",
                    path=str(archive_path),
                    member=member.name,
                    tar_type=repr(member.type),
                )
            if member.size < 0 or member.size > limits["maximum_member_bytes"]:
                _fail(
                    "archive_member_size_refused",
                    "Archive member exceeds the configured size limit",
                    member=member.name,
                    size=member.size,
                    maximum=limits["maximum_member_bytes"],
                )
            total_bytes += member.size
            if total_bytes > limits["maximum_total_uncompressed_bytes"]:
                _fail(
                    "archive_uncompressed_size_refused",
                    "Archive exceeds the total uncompressed size limit",
                    path=str(archive_path),
                    actual=total_bytes,
                    maximum=limits["maximum_total_uncompressed_bytes"],
                )
            validated.append((member, path))

        for member, relative in validated:
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                if target.exists() and not target.is_dir():
                    _fail(
                        "archive_member_collision",
                        "Archive directory collides with an existing file",
                        member=member.name,
                    )
                target.mkdir(parents=True, exist_ok=True, mode=0o755)
                continue
            if target.exists() or target.is_symlink():
                _fail(
                    "archive_member_collision",
                    "Archive file would overwrite a previously extracted path",
                    member=member.name,
                )
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            source = archive.extractfile(member)
            if source is None:
                _fail(
                    "archive_member_unreadable",
                    "Regular archive member cannot be read",
                    member=member.name,
                )
            try:
                with target.open("xb") as destination_stream:
                    _copy_exact(source, destination_stream, member.size)
                    destination_stream.flush()
                    os.fsync(destination_stream.fileno())
            finally:
                source.close()
            os.chmod(target, 0o644)
            member_count += 1
    return {
        "regular_files_extracted": member_count,
        "uncompressed_bytes": total_bytes,
    }


def _read_split(path: Path, expected_count: int) -> tuple[str, ...]:
    if path.is_symlink() or not path.is_file():
        _fail(
            "split_file_missing",
            "Required split file is missing or not regular",
            path=str(path),
        )
    try:
        raw_lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        _fail(
            "split_file_unreadable",
            "Split file must contain ASCII VOC identifiers",
            path=str(path),
            error=str(exc),
        )
    identifiers = tuple(line.strip() for line in raw_lines)
    if any(_VOC_ID_RE.fullmatch(identifier) is None for identifier in identifiers):
        _fail(
            "split_identifier_invalid",
            "Every split entry must be one six-digit VOC identifier",
            path=str(path),
        )
    if len(identifiers) != len(set(identifiers)):
        _fail(
            "split_identifier_duplicate",
            "Split file contains duplicate image identifiers",
            path=str(path),
        )
    if len(identifiers) != expected_count:
        _fail(
            "split_count_mismatch",
            "Split image count differs from the frozen manifest",
            path=str(path),
            expected=expected_count,
            actual=len(identifiers),
        )
    return identifiers


def _read_exact(stream: BinaryIO, count: int, path: Path) -> bytes:
    value = stream.read(count)
    if len(value) != count:
        _fail(
            "jpeg_truncated",
            "JPEG ended before its declared marker payload",
            path=str(path),
        )
    return value


def _jpeg_dimensions(path: Path) -> tuple[int, int, int]:
    """Return JPEG width, height, component count without image libraries."""
    if path.is_symlink() or not path.is_file():
        _fail(
            "jpeg_missing",
            "Required JPEG image is missing or not regular",
            path=str(path),
        )
    with path.open("rb") as stream:
        if _read_exact(stream, 2, path) != b"\xff\xd8":
            _fail(
                "jpeg_signature_invalid",
                "Image does not have a JPEG SOI marker",
                path=str(path),
            )
        while True:
            prefix = stream.read(1)
            if not prefix:
                _fail(
                    "jpeg_dimensions_missing",
                    "JPEG has no supported start-of-frame marker",
                    path=str(path),
                )
            if prefix != b"\xff":
                continue
            marker_byte = _read_exact(stream, 1, path)
            while marker_byte == b"\xff":
                marker_byte = _read_exact(stream, 1, path)
            marker = marker_byte[0]
            if marker in {0x00, 0x01} or 0xD0 <= marker <= 0xD7:
                continue
            if marker in {0xD9, 0xDA}:
                _fail(
                    "jpeg_dimensions_missing",
                    "JPEG reached image data before a supported frame marker",
                    path=str(path),
                )
            segment_length = int.from_bytes(
                _read_exact(stream, 2, path),
                "big",
            )
            if segment_length < 2:
                _fail(
                    "jpeg_marker_length_invalid",
                    "JPEG marker has an invalid segment length",
                    path=str(path),
                    marker=marker,
                )
            payload_length = segment_length - 2
            if marker in _JPEG_SOF_MARKERS:
                header = _read_exact(stream, min(payload_length, 6), path)
                if payload_length < 6:
                    _fail(
                        "jpeg_frame_invalid",
                        "JPEG frame header is too short",
                        path=str(path),
                    )
                height = int.from_bytes(header[1:3], "big")
                width = int.from_bytes(header[3:5], "big")
                components = header[5]
                if width <= 0 or height <= 0 or components <= 0:
                    _fail(
                        "jpeg_dimensions_invalid",
                        "JPEG frame dimensions/components must be positive",
                        path=str(path),
                    )
                return width, height, components
            stream.seek(payload_length, os.SEEK_CUR)


def _required_text(parent: ET.Element, tag: str, path: Path) -> str:
    children = parent.findall(tag)
    if len(children) != 1 or children[0].text is None:
        _fail(
            "xml_field_invalid",
            "VOC XML field must occur exactly once with text",
            path=str(path),
            field=tag,
            occurrences=len(children),
        )
    value = children[0].text.strip()
    if not value:
        _fail(
            "xml_field_invalid",
            "VOC XML field cannot be empty",
            path=str(path),
            field=tag,
        )
    return value


def _parse_int(parent: ET.Element, tag: str, path: Path) -> int:
    value = _required_text(parent, tag, path)
    if re.fullmatch(r"-?[0-9]+", value) is None:
        _fail(
            "xml_integer_invalid",
            "VOC XML numeric fields must contain integers",
            path=str(path),
            field=tag,
            value=value,
        )
    return int(value)


def _parse_finite_number(
    parent: ET.Element,
    tag: str,
    path: Path,
) -> int | float:
    """Parse an exact integer or finite decimal used by VOC part metadata."""
    value = _required_text(parent, tag, path)
    if re.fullmatch(
        r"-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?",
        value,
    ) is None:
        _fail(
            "xml_number_invalid",
            "VOC XML numeric fields must contain finite decimal numbers",
            path=str(path),
            field=tag,
            value=value,
        )
    if re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    converted = float(value)
    if not math.isfinite(converted):
        _fail(
            "xml_number_invalid",
            "VOC XML numeric fields must be finite",
            path=str(path),
            field=tag,
            value=value,
        )
    return converted


def _parse_binary(parent: ET.Element, tag: str, path: Path) -> int:
    value = _parse_int(parent, tag, path)
    if value not in {0, 1}:
        _fail(
            "xml_binary_flag_invalid",
            "VOC flags must be zero or one",
            path=str(path),
            field=tag,
            value=value,
        )
    return value


def _parse_bbox(
    bbox: ET.Element,
    *,
    width: int,
    height: int,
    path: Path,
) -> tuple[int, int, int, int]:
    values = tuple(
        _parse_int(bbox, field, path)
        for field in ("xmin", "ymin", "xmax", "ymax")
    )
    xmin, ymin, xmax, ymax = values
    if not (1 <= xmin <= xmax <= width and 1 <= ymin <= ymax <= height):
        _fail(
            "bbox_out_of_bounds",
            "VOC one-based inclusive bounding box is invalid",
            path=str(path),
            bbox=list(values),
            image_width=width,
            image_height=height,
        )
    return values


def _parse_part_bbox(
    bbox: ET.Element,
    *,
    width: int,
    height: int,
    path: Path,
) -> tuple[int | float, int | float, int | float, int | float]:
    """Parse VOC part boxes, which legitimately contain decimal coordinates."""
    values = tuple(
        _parse_finite_number(bbox, field, path)
        for field in ("xmin", "ymin", "xmax", "ymax")
    )
    xmin, ymin, xmax, ymax = values
    if not (1 <= xmin <= xmax <= width and 1 <= ymin <= ymax <= height):
        _fail(
            "part_bbox_out_of_bounds",
            "VOC one-based part bounding box is invalid",
            path=str(path),
            bbox=list(values),
            image_width=width,
            image_height=height,
        )
    return values


def _parse_parts(
    object_element: ET.Element,
    *,
    width: int,
    height: int,
    path: Path,
) -> list[dict[str, Any]]:
    parts = []
    for part in object_element.findall("part"):
        name_elements = part.findall("name")
        class_elements = part.findall("class")
        if len(name_elements) + len(class_elements) != 1:
            _fail(
                "part_name_invalid",
                "VOC object parts require exactly one name/class field",
                path=str(path),
            )
        name = _required_text(
            part,
            "name" if name_elements else "class",
            path,
        )
        bboxes = part.findall("bndbox")
        if len(bboxes) != 1:
            _fail(
                "part_bbox_invalid",
                "VOC object parts require exactly one bounding box",
                path=str(path),
                part=name,
            )
        bbox = _parse_part_bbox(
            bboxes[0],
            width=width,
            height=height,
            path=path,
        )
        allowed_fields = {
            "name",
            "class",
            "pose",
            "truncated",
            "difficult",
            "bndbox",
        }
        unknown = sorted(
            child.tag
            for child in part
            if child.tag not in allowed_fields
        )
        if unknown:
            _fail(
                "part_field_unsupported",
                "Unknown VOC part fields would be lost",
                path=str(path),
                fields=unknown,
            )
        metadata: dict[str, Any] = {
            "name": name,
            "voc_bbox": list(bbox),
        }
        pose_elements = part.findall("pose")
        if pose_elements:
            metadata["pose"] = _required_text(part, "pose", path)
        truncated_elements = part.findall("truncated")
        if truncated_elements:
            metadata["truncated"] = _parse_binary(part, "truncated", path)
        difficult_elements = part.findall("difficult")
        if difficult_elements:
            metadata["difficult"] = _parse_binary(part, "difficult", path)
        parts.append(metadata)
    return parts


def _parse_annotation(
    xml_path: Path,
    jpeg_path: Path,
    *,
    identifier: str,
    image_id: int,
    annotation_id_start: int,
    category_ids: dict[str, int],
    maximum_xml_bytes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], Counter[str]]:
    if xml_path.is_symlink() or not xml_path.is_file():
        _fail(
            "annotation_missing",
            "Required VOC annotation is missing or not regular",
            path=str(xml_path),
        )
    xml_size = xml_path.stat().st_size
    if xml_size <= 0 or xml_size > maximum_xml_bytes:
        _fail(
            "annotation_size_refused",
            "VOC XML size is empty or exceeds the configured limit",
            path=str(xml_path),
            size=xml_size,
            maximum=maximum_xml_bytes,
        )
    xml_bytes = xml_path.read_bytes()
    upper_xml = xml_bytes.upper()
    if b"<!DOCTYPE" in upper_xml or b"<!ENTITY" in upper_xml:
        _fail(
            "xml_entity_declaration_refused",
            "DTD and entity declarations are not permitted",
            path=str(xml_path),
        )
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        _fail(
            "annotation_xml_invalid",
            "VOC annotation is not well-formed XML",
            path=str(xml_path),
            error=str(exc),
        )
    if root.tag != "annotation":
        _fail(
            "annotation_root_invalid",
            "VOC XML root must be annotation",
            path=str(xml_path),
            actual=root.tag,
        )
    filename = _required_text(root, "filename", xml_path)
    expected_filename = f"{identifier}.jpg"
    if filename != expected_filename:
        _fail(
            "annotation_filename_mismatch",
            "VOC XML filename does not match its split identifier",
            path=str(xml_path),
            expected=expected_filename,
            actual=filename,
        )
    sizes = root.findall("size")
    if len(sizes) != 1:
        _fail(
            "annotation_size_field_invalid",
            "VOC XML requires exactly one size element",
            path=str(xml_path),
        )
    width = _parse_int(sizes[0], "width", xml_path)
    height = _parse_int(sizes[0], "height", xml_path)
    depth = _parse_int(sizes[0], "depth", xml_path)
    if width <= 0 or height <= 0 or depth <= 0:
        _fail(
            "annotation_dimensions_invalid",
            "VOC XML dimensions must be positive",
            path=str(xml_path),
        )
    jpeg_width, jpeg_height, jpeg_components = _jpeg_dimensions(jpeg_path)
    if (width, height, depth) != (
        jpeg_width,
        jpeg_height,
        jpeg_components,
    ):
        _fail(
            "annotation_jpeg_dimensions_mismatch",
            "VOC XML dimensions do not match the JPEG frame",
            path=str(xml_path),
            xml=[width, height, depth],
            jpeg=[jpeg_width, jpeg_height, jpeg_components],
        )

    image = {
        "file_name": expected_filename,
        "height": height,
        "id": image_id,
        "license": 1,
        "source_annotation_sha256": hashlib.sha256(xml_bytes).hexdigest(),
        "source_image_sha256": _hash_file(jpeg_path),
        "voc_id": identifier,
        "width": width,
    }
    annotations = []
    category_counts: Counter[str] = Counter()
    next_annotation_id = annotation_id_start
    allowed_object_fields = {
        "name",
        "pose",
        "truncated",
        "occluded",
        "difficult",
        "bndbox",
        "part",
    }
    for object_index, object_element in enumerate(root.findall("object")):
        unknown = sorted(
            child.tag
            for child in object_element
            if child.tag not in allowed_object_fields
        )
        if unknown:
            _fail(
                "object_field_unsupported",
                "Unknown VOC object fields would be lost",
                path=str(xml_path),
                object_index=object_index,
                fields=unknown,
            )
        category = _required_text(object_element, "name", xml_path)
        if category not in category_ids:
            _fail(
                "annotation_category_unknown",
                "VOC object category is not in the frozen mapping",
                path=str(xml_path),
                object_index=object_index,
                category=category,
            )
        pose = _required_text(object_element, "pose", xml_path)
        truncated = _parse_binary(object_element, "truncated", xml_path)
        difficult = _parse_binary(object_element, "difficult", xml_path)
        occluded_elements = object_element.findall("occluded")
        occluded = None
        if occluded_elements:
            if len(occluded_elements) != 1:
                _fail(
                    "xml_field_invalid",
                    "VOC occluded flag may occur at most once",
                    path=str(xml_path),
                )
            occluded = _parse_binary(object_element, "occluded", xml_path)
        bboxes = object_element.findall("bndbox")
        if len(bboxes) != 1:
            _fail(
                "object_bbox_invalid",
                "Every VOC object requires exactly one bounding box",
                path=str(xml_path),
                object_index=object_index,
            )
        xmin, ymin, xmax, ymax = _parse_bbox(
            bboxes[0],
            width=width,
            height=height,
            path=xml_path,
        )
        coco_width = xmax - xmin + 1
        coco_height = ymax - ymin + 1
        voc_metadata: dict[str, Any] = {
            "object_index": object_index,
            "pose": pose,
            "truncated": truncated,
        }
        if occluded is not None:
            voc_metadata["occluded"] = occluded
        parts = _parse_parts(
            object_element,
            width=width,
            height=height,
            path=xml_path,
        )
        if parts:
            voc_metadata["parts"] = parts
        annotations.append({
            "area": coco_width * coco_height,
            "bbox": [
                xmin - 1,
                ymin - 1,
                coco_width,
                coco_height,
            ],
            "category_id": category_ids[category],
            "difficult": difficult,
            "id": next_annotation_id,
            "image_id": image_id,
            "iscrowd": 0,
            "voc_bbox": [xmin, ymin, xmax, ymax],
            "voc_metadata": voc_metadata,
        })
        category_counts[category] += 1
        next_annotation_id += 1
    return image, annotations, category_counts


def _coco_header(
    manifest: dict[str, Any],
    categories: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    terms = manifest["license_and_terms"]
    return {
        "info": {
            "description": (
                "PASCAL VOC2007 full detection data converted "
                f"deterministically to COCO ({split})"
            ),
            "source": manifest["official_metadata"]["dataset_page"],
            "version": manifest["dataset_version"],
        },
        "licenses": [{
            "id": 1,
            "name": (
                "No dataset-wide SPDX license asserted; review VOC database "
                "rights and source-image terms"
            ),
            "url": terms["database_rights_url"],
        }],
        "categories": categories,
        "images": [],
        "annotations": [],
    }


def _validate_source_inventory(
    voc_root: Path,
    all_ids: set[str],
) -> None:
    annotations_dir = voc_root / "Annotations"
    images_dir = voc_root / "JPEGImages"
    for directory, suffix, label in (
        (annotations_dir, ".xml", "annotation"),
        (images_dir, ".jpg", "image"),
    ):
        if directory.is_symlink() or not directory.is_dir():
            _fail(
                f"{label}_directory_missing",
                f"Required VOC {label} directory is missing",
                path=str(directory),
            )
        entries = tuple(directory.iterdir())
        if any(item.is_symlink() or not item.is_file() for item in entries):
            _fail(
                f"{label}_inventory_unsafe",
                f"VOC {label} directory may contain regular files only",
                path=str(directory),
            )
        unexpected_suffixes = sorted(
            item.name for item in entries if item.suffix != suffix
        )
        if unexpected_suffixes:
            _fail(
                f"{label}_inventory_unexpected_files",
                f"VOC {label} directory contains unexpected files",
                path=str(directory),
                sample=unexpected_suffixes[:10],
                count=len(unexpected_suffixes),
            )
        identifiers = {item.stem for item in entries}
        missing = sorted(all_ids - identifiers)
        extra = sorted(identifiers - all_ids)
        if missing or extra:
            _fail(
                f"{label}_inventory_mismatch",
                f"VOC {label} inventory does not exactly match split IDs",
                missing_count=len(missing),
                missing_sample=missing[:10],
                extra_count=len(extra),
                extra_sample=extra[:10],
            )


def build_coco_documents(
    manifest: dict[str, Any],
    voc_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Validate all source records and build deterministic COCO documents."""
    if voc_root.is_symlink() or not voc_root.is_dir():
        _fail(
            "voc_root_invalid",
            "VOC root must be a non-symlink directory",
            path=str(voc_root),
        )
    voc_root = voc_root.resolve(strict=True)
    counts = manifest["expected_counts"]
    main_dir = voc_root / "ImageSets" / "Main"
    split_ids = {
        split: _read_split(
            main_dir / f"{split}.txt",
            counts[split]["images"],
        )
        for split in ("train", "val", "trainval", "test")
    }
    train = set(split_ids["train"])
    val = set(split_ids["val"])
    trainval = set(split_ids["trainval"])
    test = set(split_ids["test"])
    if train & val:
        _fail(
            "train_val_overlap",
            "VOC train and validation splits must be disjoint",
            overlap_count=len(train & val),
        )
    if train | val != trainval:
        _fail(
            "trainval_membership_mismatch",
            "VOC trainval must equal the union of train and val",
            missing_count=len((train | val) - trainval),
            extra_count=len(trainval - (train | val)),
        )
    if trainval & test:
        _fail(
            "trainval_test_overlap",
            "VOC trainval and test splits must be disjoint",
            overlap_count=len(trainval & test),
        )
    all_ids = trainval | test
    if len(all_ids) != counts["combined"]["images"]:
        _fail(
            "combined_split_count_mismatch",
            "Combined unique image count differs from the manifest",
            expected=counts["combined"]["images"],
            actual=len(all_ids),
        )
    _validate_source_inventory(voc_root, all_ids)

    category_names = manifest["categories"]
    category_ids = {
        name: index for index, name in enumerate(category_names, start=1)
    }
    categories = [
        {"id": category_ids[name], "name": name}
        for name in category_names
    ]
    category_names_by_id = {
        category["id"]: category["name"] for category in categories
    }
    documents = {}
    split_summaries = {}
    for split in ("trainval", "test"):
        document = _coco_header(manifest, categories, split)
        category_counts: Counter[str] = Counter()
        category_non_difficult_counts: Counter[str] = Counter()
        annotation_id = 1
        for identifier in sorted(split_ids[split]):
            image, annotations, image_category_counts = _parse_annotation(
                voc_root / "Annotations" / f"{identifier}.xml",
                voc_root / "JPEGImages" / f"{identifier}.jpg",
                identifier=identifier,
                image_id=int(identifier),
                annotation_id_start=annotation_id,
                category_ids=category_ids,
                maximum_xml_bytes=manifest["security_limits"][
                    "maximum_xml_bytes"
                ],
            )
            document["images"].append(image)
            document["annotations"].extend(annotations)
            annotation_id += len(annotations)
            category_counts.update(image_category_counts)
            category_non_difficult_counts.update(
                category_names_by_id[annotation["category_id"]]
                for annotation in annotations
                if annotation["difficult"] == 0
            )
        expected = counts[split]
        if len(document["images"]) != expected["images"]:
            _fail(
                "converted_image_count_mismatch",
                "Converted image count differs from the manifest",
                split=split,
                expected=expected["images"],
                actual=len(document["images"]),
            )
        difficult_objects = sum(
            annotation["difficult"]
            for annotation in document["annotations"]
        )
        non_difficult_objects = (
            len(document["annotations"]) - difficult_objects
        )
        if non_difficult_objects != expected["objects_non_difficult"]:
            _fail(
                "converted_non_difficult_object_count_mismatch",
                "Converted non-difficult object count differs from official "
                "statistics",
                split=split,
                expected=expected["objects_non_difficult"],
                actual=non_difficult_objects,
                total_xml_objects=len(document["annotations"]),
            )
        actual_non_difficult_category_counts = {
            name: category_non_difficult_counts[name]
            for name in category_names
        }
        if (
            split == "trainval"
            and actual_non_difficult_category_counts
            != expected["category_objects_non_difficult"]
        ):
            differences = {
                name: {
                    "expected": (
                        expected["category_objects_non_difficult"][name]
                    ),
                    "actual": category_non_difficult_counts[name],
                }
                for name in category_names
                if category_non_difficult_counts[name]
                != expected["category_objects_non_difficult"][name]
            }
            _fail(
                "trainval_non_difficult_category_count_mismatch",
                "Converted non-difficult category counts differ from official "
                "trainval totals",
                differences=differences,
            )
        documents[split] = document
        split_summaries[split] = {
            "images": len(document["images"]),
            "objects_total": len(document["annotations"]),
            "objects_non_difficult": non_difficult_objects,
            "objects_difficult": difficult_objects,
            "category_objects_total": {
                name: category_counts[name] for name in category_names
            },
            "category_objects_non_difficult": {
                name: category_non_difficult_counts[name]
                for name in category_names
            },
        }

    for split in ("train", "val"):
        source_document = documents["trainval"]
        subset_ids = set(split_ids[split])
        document = _coco_header(manifest, categories, split)
        document["images"] = [
            image
            for image in source_document["images"]
            if image["voc_id"] in subset_ids
        ]
        subset_image_ids = {image["id"] for image in document["images"]}
        annotation_id = 1
        category_counts: Counter[str] = Counter()
        category_non_difficult_counts: Counter[str] = Counter()
        for source_annotation in source_document["annotations"]:
            if source_annotation["image_id"] not in subset_image_ids:
                continue
            annotation = dict(source_annotation)
            annotation["id"] = annotation_id
            document["annotations"].append(annotation)
            annotation_id += 1
            category_counts[
                category_names_by_id[annotation["category_id"]]
            ] += 1
            if annotation["difficult"] == 0:
                category_non_difficult_counts[
                    category_names_by_id[annotation["category_id"]]
                ] += 1
        difficult_objects = sum(
            annotation["difficult"]
            for annotation in document["annotations"]
        )
        non_difficult_objects = (
            len(document["annotations"]) - difficult_objects
        )
        if non_difficult_objects != counts[split]["objects_non_difficult"]:
            _fail(
                "subsplit_non_difficult_object_count_mismatch",
                "VOC train/val non-difficult object count differs from "
                "official statistics",
                split=split,
                expected=counts[split]["objects_non_difficult"],
                actual=non_difficult_objects,
                total_xml_objects=len(document["annotations"]),
            )
        documents[split] = document
        split_summaries[split] = {
            "images": len(document["images"]),
            "objects_total": len(document["annotations"]),
            "objects_non_difficult": non_difficult_objects,
            "objects_difficult": difficult_objects,
            "category_objects_total": {
                name: category_counts[name] for name in category_names
            },
            "category_objects_non_difficult": {
                name: category_non_difficult_counts[name]
                for name in category_names
            },
        }
    combined_total_objects = sum(
        summary["objects_total"]
        for name, summary in split_summaries.items()
        if name in {"trainval", "test"}
    )
    combined_non_difficult_objects = sum(
        summary["objects_non_difficult"]
        for name, summary in split_summaries.items()
        if name in {"trainval", "test"}
    )
    if (
        combined_non_difficult_objects
        != counts["combined"]["objects_non_difficult"]
    ):
        _fail(
            "combined_non_difficult_object_count_mismatch",
            "Combined non-difficult object count differs from official "
            "statistics",
            expected=counts["combined"]["objects_non_difficult"],
            actual=combined_non_difficult_objects,
            total_xml_objects=combined_total_objects,
        )
    summary = {
        "splits": split_summaries,
        "combined": {
            "images": len(all_ids),
            "objects_total": combined_total_objects,
            "objects_non_difficult": combined_non_difficult_objects,
            "objects_difficult": (
                combined_total_objects - combined_non_difficult_objects
            ),
        },
        "invariants": {
            "all_images_preserved": True,
            "all_objects_preserved": True,
            "all_categories_mapped": True,
            "all_bboxes_reversible": True,
            "all_difficult_flags_preserved": True,
            "jpeg_dimensions_verified": True,
            "source_inventory_exact": True,
            "train_val_disjoint": True,
            "trainval_test_disjoint": True,
        },
    }
    return documents, summary


def _tree_digest(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for current, directory_names, file_names in os.walk(root):
        directory_names.sort()
        file_names.sort()
        current_path = Path(current)
        for directory_name in directory_names:
            directory = current_path / directory_name
            if directory.is_symlink():
                _fail(
                    "source_tree_symlink_refused",
                    "Prepared source tree cannot contain symlinks",
                    path=str(directory),
                )
        for file_name in file_names:
            path = current_path / file_name
            if path.is_symlink() or not path.is_file():
                _fail(
                    "source_tree_file_invalid",
                    "Prepared source tree may contain regular files only",
                    path=str(path),
                )
            relative = path.relative_to(root).as_posix()
            size = path.stat().st_size
            file_sha256 = _hash_file(path)
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(size).encode("ascii"))
            digest.update(b"\0")
            digest.update(file_sha256.encode("ascii"))
            digest.update(b"\n")
            file_count += 1
            total_bytes += size
    return {
        "algorithm": "sha256",
        "digest": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def _acquire_lock(output: Path) -> tuple[int, Path]:
    lock_path = output.parent / f".{output.name}.prepare.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError:
        _fail(
            "preparation_lock_exists",
            "Another preparation may be active or requires recovery",
            path=str(lock_path),
        )
    return descriptor, lock_path


def prepare_dataset(
    *,
    manifest_path: Path,
    archives_dir: Path,
    output: Path,
    accept_terms: bool,
) -> dict[str, Any]:
    """Verify, extract, convert, audit, and atomically publish VOC2007."""
    manifest = load_manifest(manifest_path)
    manifest_path = manifest_path.resolve(strict=True)
    if not accept_terms:
        _fail(
            "dataset_terms_not_acknowledged",
            "Preparation requires explicit acknowledgement of dataset terms",
            database_rights_url=manifest["license_and_terms"][
                "database_rights_url"
            ],
            flickr_terms_url=manifest["license_and_terms"][
                "flickr_terms_url"
            ],
        )
    if output.is_symlink():
        _fail(
            "output_already_exists",
            "Refusing to overwrite an existing prepared dataset",
            path=str(output),
        )
    output = output.resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        _fail(
            "output_already_exists",
            "Refusing to overwrite an existing prepared dataset",
            path=str(output),
        )
    lock_descriptor, lock_path = _acquire_lock(output)
    staging: Path | None = None
    try:
        lock_content = _canonical_json_bytes({
            "output": str(output),
            "manifest_sha256": _hash_file(manifest_path),
        })
        owned_lock_descriptor = lock_descriptor
        lock_descriptor = -1
        with os.fdopen(owned_lock_descriptor, "wb") as lock_stream:
            lock_stream.write(lock_content)
            lock_stream.flush()
            os.fsync(lock_stream.fileno())
        archive_records = verify_archives(manifest, archives_dir)
        staging = Path(tempfile.mkdtemp(
            prefix=f".{output.name}.staging.",
            dir=output.parent,
        ))
        archive_stats = {}
        archives = _archive_by_role(manifest)
        for role in _ARCHIVE_ROLES:
            archive_stats[role] = _extract_one_archive(
                archives_dir.resolve(strict=True) / archives[role]["filename"],
                staging,
                manifest["security_limits"],
            )
        voc_root = staging / "VOCdevkit" / "VOC2007"
        documents, validation_summary = build_coco_documents(
            manifest,
            voc_root,
        )
        output_records = []
        for split in _COCO_FILENAMES:
            relative = Path(_COCO_FILENAMES[split])
            output_path = staging / relative
            content = _canonical_json_bytes(documents[split])
            _atomic_write(output_path, content)
            output_records.append({
                "split": split,
                "path": relative.as_posix(),
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            })
        source_tree = _tree_digest(staging / "VOCdevkit")
        integrity = {
            "schema_version": 1,
            "dataset_id": manifest["dataset_id"],
            "dataset_version": manifest["dataset_version"],
            "manifest_path": manifest_path.name,
            "manifest_sha256": _hash_file(manifest_path),
            "converter_sha256": _hash_file(Path(__file__).resolve()),
            "dataset_terms_acknowledged": True,
            "archives": archive_records,
            "archive_extraction": archive_stats,
            "source_tree": source_tree,
            "outputs": output_records,
            "validation": validation_summary,
            "network_access_by_preparer": False,
            "atomic_publication": True,
        }
        integrity_path = staging / INTEGRITY_FILENAME
        _atomic_json(integrity_path, integrity)
        integrity_sha256 = _hash_file(integrity_path)
        _atomic_write(
            staging / INTEGRITY_DIGEST_FILENAME,
            f"{integrity_sha256}  {INTEGRITY_FILENAME}\n".encode("ascii"),
        )
        if output.exists() or output.is_symlink():
            _fail(
                "output_created_during_preparation",
                "Output appeared while preparation was in progress",
                path=str(output),
            )
        os.rename(staging, output)
        _fsync_directory(output.parent)
        staging = None
        return {
            "status": "prepared",
            "output": str(output),
            "manifest_sha256": integrity["manifest_sha256"],
            "integrity_sha256": integrity_sha256,
            "validation": validation_summary,
        }
    finally:
        if lock_descriptor >= 0:
            os.close(lock_descriptor)
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        if lock_path.exists():
            lock_path.unlink()


def _load_json_regular(path: Path, code: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(code, "Required JSON artifact is missing or unsafe", path=str(path))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(code, "Required JSON artifact is invalid", path=str(path), error=str(exc))
    return _require_mapping(value, str(path))


def validate_prepared_dataset(
    *,
    manifest_path: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    """Recompute semantic and cryptographic gates for a prepared dataset."""
    manifest = load_manifest(manifest_path)
    manifest_path = manifest_path.resolve(strict=True)
    if dataset_root.is_symlink() or not dataset_root.is_dir():
        _fail(
            "prepared_dataset_root_invalid",
            "Prepared dataset root must be a non-symlink directory",
            path=str(dataset_root),
        )
    dataset_root = dataset_root.resolve(strict=True)
    integrity_path = dataset_root / INTEGRITY_FILENAME
    digest_path = dataset_root / INTEGRITY_DIGEST_FILENAME
    integrity = _load_json_regular(
        integrity_path,
        "integrity_artifact_invalid",
    )
    if digest_path.is_symlink() or not digest_path.is_file():
        _fail(
            "integrity_digest_missing",
            "Integrity digest sidecar is missing or unsafe",
            path=str(digest_path),
        )
    expected_digest_line = (
        f"{_hash_file(integrity_path)}  {INTEGRITY_FILENAME}\n"
    )
    try:
        actual_digest_line = digest_path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as exc:
        _fail(
            "integrity_digest_invalid",
            "Integrity digest sidecar is unreadable",
            path=str(digest_path),
            error=str(exc),
        )
    if actual_digest_line != expected_digest_line:
        _fail(
            "integrity_digest_mismatch",
            "Integrity artifact hash does not match its sidecar",
            expected=expected_digest_line.strip(),
            actual=actual_digest_line.strip(),
        )
    manifest_sha256 = _hash_file(manifest_path)
    if integrity.get("manifest_sha256") != manifest_sha256:
        _fail(
            "prepared_manifest_mismatch",
            "Prepared data was produced with a different manifest",
            expected=manifest_sha256,
            actual=integrity.get("manifest_sha256"),
        )
    if integrity.get("dataset_terms_acknowledged") is not True:
        _fail(
            "prepared_terms_audit_missing",
            "Prepared integrity record lacks explicit terms acknowledgement",
        )
    source_tree = _tree_digest(dataset_root / "VOCdevkit")
    if source_tree != integrity.get("source_tree"):
        _fail(
            "prepared_source_tree_mismatch",
            "Prepared VOC source tree hash/counts have changed",
            expected=integrity.get("source_tree"),
            actual=source_tree,
        )
    documents, validation_summary = build_coco_documents(
        manifest,
        dataset_root / "VOCdevkit" / "VOC2007",
    )
    expected_output_records = {
        record["split"]: record
        for record in integrity.get("outputs", [])
        if isinstance(record, dict) and "split" in record
    }
    if set(expected_output_records) != set(_COCO_FILENAMES):
        _fail(
            "prepared_output_inventory_invalid",
            "Integrity record must identify both COCO outputs",
        )
    for split, relative_name in _COCO_FILENAMES.items():
        output_path = dataset_root / relative_name
        if output_path.is_symlink() or not output_path.is_file():
            _fail(
                "prepared_output_missing",
                "Prepared COCO output is missing or unsafe",
                split=split,
                path=str(output_path),
            )
        content = output_path.read_bytes()
        expected_content = _canonical_json_bytes(documents[split])
        if content != expected_content:
            _fail(
                "prepared_output_semantic_mismatch",
                "COCO output is not the deterministic source conversion",
                split=split,
            )
        record = expected_output_records[split]
        if (
            record.get("path") != relative_name
            or record.get("size_bytes") != len(content)
            or record.get("sha256") != hashlib.sha256(content).hexdigest()
        ):
            _fail(
                "prepared_output_hash_mismatch",
                "COCO output hash/size differs from the integrity record",
                split=split,
            )
    if validation_summary != integrity.get("validation"):
        _fail(
            "prepared_validation_summary_mismatch",
            "Recomputed dataset summary differs from the integrity record",
        )
    return {
        "status": "valid",
        "dataset_root": str(dataset_root),
        "manifest_sha256": manifest_sha256,
        "integrity_sha256": _hash_file(integrity_path),
        "validation": validation_summary,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Frozen VOC2007 manifest (default: adjacent manifest.v1.json)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser(
        "verify-archives",
        help="Verify both downloaded archive identities without extraction",
    )
    verify.add_argument("--archives-dir", type=Path, required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Verify, safely extract, convert, and atomically publish",
    )
    prepare.add_argument("--archives-dir", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument(
        "--accept-dataset-terms",
        action="store_true",
        help="Confirm review of VOC database rights and source-image terms",
    )

    validate = subparsers.add_parser(
        "validate",
        help="Recompute all semantic and cryptographic prepared-data gates",
    )
    validate.add_argument("--dataset-root", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest_path = args.manifest
        manifest = load_manifest(manifest_path)
        manifest_path = manifest_path.resolve(strict=True)
        if args.command == "verify-archives":
            result = {
                "status": "verified",
                "manifest_sha256": _hash_file(manifest_path),
                "archives": verify_archives(manifest, args.archives_dir),
            }
        elif args.command == "prepare":
            result = prepare_dataset(
                manifest_path=manifest_path,
                archives_dir=args.archives_dir,
                output=args.output,
                accept_terms=args.accept_dataset_terms,
            )
        else:
            result = validate_prepared_dataset(
                manifest_path=manifest_path,
                dataset_root=args.dataset_root,
            )
    except DatasetPreparationError as exc:
        print(
            json.dumps(exc.to_dict(), sort_keys=True),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
