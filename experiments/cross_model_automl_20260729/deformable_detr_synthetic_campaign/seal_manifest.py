#!/usr/bin/env python3

"""Seal a direct Deformable-DETR qualification on the shared synthetic data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
BASE = (
    HERE.parent / "deformable_detr_campaign" / "campaign.v1.json"
)
DATASET_DIR = (
    HERE.parent
    / "datasets"
    / "tao_od_synthetic_full_dino_coco"
)
DATASET_MANIFEST = DATASET_DIR / "manifest.v1.json"
DATASET_INTEGRITY = DATASET_DIR / "integrity.v1.json"
DEFAULT_OUTPUT = HERE / "campaign.v1.json"
SDK_DIR = Path(
    "/localhome/local-rarunachalam/.tao/worktrees/"
    "tao-sdk-slurm-a2e50d0"
)
SDK_COMMIT = "a2e50d0930c3e3785b4b39fa8c3da88b39ff89e5"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def build(
    *,
    source_repository: Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads(BASE.read_text(encoding="utf-8"))
    manifest.pop("manifest_sha256")
    dataset = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    train = dataset["splits"]["train"]
    validation = dataset["splits"]["validation"]
    manifest["campaign_id"] = (
        "deformable-detr-synthetic-direct-qualification-20260730"
    )
    if source_repository is not None:
        manifest["source"]["repository"] = str(source_repository.resolve())
    manifest["runtime"].update(
        {
            "sdk_dir": str(SDK_DIR),
            "sdk_revision": SDK_COMMIT,
            "slurm_use_sqsh": True,
        }
    )
    manifest["dataset"] = {
        "id": dataset["dataset_id"],
        "manifest": {
            "path": str(DATASET_MANIFEST),
            "sha256": _sha256_file(DATASET_MANIFEST),
            "size_bytes": DATASET_MANIFEST.stat().st_size,
        },
        "integrity": {
            "path": str(DATASET_INTEGRITY),
            "sha256": _sha256_file(DATASET_INTEGRITY),
            "size_bytes": DATASET_INTEGRITY.stat().st_size,
        },
        "slurm_root": dataset["source"]["staged_lustre_root"],
        "train_image_dir": train["images"]["path"],
        "validation_image_dir": validation["images"]["path"],
        "train_annotation": train["annotation"]["path"],
        "train_annotation_sha256": train["annotation"]["sha256"],
        "train_annotation_size_bytes": train["annotation"]["size_bytes"],
        "validation_annotation": validation["annotation"]["path"],
        "validation_annotation_sha256": validation["annotation"]["sha256"],
        "validation_annotation_size_bytes": validation["annotation"][
            "size_bytes"
        ],
        "image_tree": copy.deepcopy(train["images"]["identity"]),
        "num_classes": dataset["num_classes_with_background"],
        "eval_class_ids": copy.deepcopy(dataset["eval_class_ids"]),
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--source-repository", type=Path)
    args = parser.parse_args()
    expected = build(source_repository=args.source_repository)
    if args.verify:
        observed = json.loads(args.output.read_text(encoding="utf-8"))
        if observed != expected:
            raise SystemExit("sealed synthetic manifest changed")
        print(expected["manifest_sha256"])
        return 0
    if args.output.exists():
        raise SystemExit(f"refusing to replace {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(expected["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
