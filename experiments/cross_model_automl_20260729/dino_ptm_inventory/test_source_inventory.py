"""Contracts for the frozen official DINO NGC source inventory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit


HERE = Path(__file__).resolve().parent
INVENTORY_PATH = HERE / "source_inventory.v1.json"

RESOURCE_ORDER = [
    "pretrained_dino_coco",
    "dino_with_fm_backbone",
    "pretrained_dino_imagenet",
    "pretrained_dino_nvimagenet",
]

EXPECTED = {
    "pretrained_dino_coco": [
        (
            "dino_fan_large_trainable_v1.0",
            "dino_fan_large_imagenet22k_36ep.pth",
            1197490926,
        ),
        (
            "dino_fan_small_deployable_v1.0",
            "dino_fan_small_ep12.onnx",
            215567293,
        ),
        (
            "dino_fan_small_trainable_v1.0",
            "dino_fan_small_ep12.pth",
            580862106,
        ),
        (
            "dino_gc_vit_tiny_deployable_v1.0",
            "dino_gcvit_tiny0_ep12.onnx",
            234840909,
        ),
        (
            "dino_gc_vit_tiny_trainable_v1.0",
            "dino_gcvit_tiny0_ep12.pth",
            577043347,
        ),
        (
            "dino_resnet_50_deployable_v1.0",
            "dino_resnet50_ep12.onnx",
            209270924,
        ),
        (
            "dino_resnet_50_trainable_v1.0",
            "dino_resnet50_ep12.pth",
            568767395,
        ),
    ],
    "dino_with_fm_backbone": [
        (
            "deployable_v1.0",
            "dino_nvdinov2_518_1536_coco_e36_op17.onnx",
            1473738476,
        ),
        (
            "trainable_v1.0",
            "dino_nvdinov2_518_1536_coco_e36.pth",
            4232490433,
        ),
    ],
    "pretrained_dino_imagenet": [
        (
            "gcvit_large_imagenet22k_384",
            "gcvit_large_imagenet22k_384.pth",
            864054197,
        ),
        (
            "gcvit_large_imagenet1k",
            "gcvit_large_imagenet1k.pth",
            814560029,
        ),
        ("gcvit_base_imagenet1k", "gcvit_base_imagenet1k.pth", 367540965),
        ("gcvit_small_imagenet1k", "gcvit_small_imagenet1k.pth", 210626653),
        ("gcvit_tiny_imagenet1k", "gcvit_tiny_imagenet1k.pth", 119124861),
        ("gcvit_xtiny_imagenet1k", "gcvit_xtiny_imagenet1k.pth", 82106937),
        ("gcvit_xxtiny_imagenet1k", "gcvit_xxtiny_imagenet1k.pth", 50020581),
        ("fan_hybrid_tiny", "fan_hybrid_tiny.pth", 30032227),
        ("fan_hybrid_small", "fan_hybrid_small.pth", 104729466),
        ("fan_hybrid_large_in22k", "fan_hybrid_large_in22k.pth", 342977114),
        (
            "fan_hybrid_large_in22k_384",
            "fan_hybrid_large_in22k_384.pth",
            308027006,
        ),
        (
            "fan_hybrid_large_in22k_1k",
            "fan_hybrid_large_in22k_1k.pth",
            308026197,
        ),
        (
            "fan_hybrid_large_in22k_1k_384",
            "fan_hybrid_large_in22k_1k_384.pth",
            308029433,
        ),
        ("fan_hybrid_base_in22k", "fan_hybrid_base_in22k.pth", 234955105),
        (
            "fan_hybrid_base_in22k_1k",
            "fan_hybrid_base_in22k_1k.pth",
            202329016,
        ),
        (
            "fan_hybrid_base_in22k_1k_384",
            "fan_hybrid_base_in22k_1k_384.pth",
            202331436,
        ),
    ],
    "pretrained_dino_nvimagenet": [
        ("resnet50", "resnet50_nvimagenetv2.pth.tar", 307117121),
        (
            "fan_large_hybrid_nvimagenet",
            "fan_large_hybrid_nvimagenet.pth",
            308027815,
        ),
        (
            "fan_small_hybrid_nvimagenet",
            "fan_small_hybrid_nvimagenet.pth",
            104734139,
        ),
        (
            "fan_base_hybrid_nvimagenet",
            "fan_base_hybrid_nvimagenet.pth",
            202330226,
        ),
        ("gcvit_base_nvimagenet", "gcvit_base_nvimagenet.pth", 367540965),
        ("gcvit_small_nvimagenet", "gcvit_small_nvimagenet.pth", 210626653),
        ("gcvit_tiny_nvimagenet", "gcvit_tiny_nvimagenet.pth", 119124861),
        ("gcvit_xtiny_nvimagenet", "gcvit_xtiny_nvimagenet.pth", 82106937),
        ("gcvit_xxtiny_nvimagenet", "gcvit_xxtiny_nvimagenet.pth", 50020581),
        (
            "fan_hybrid_tiny_nvimagenet",
            "fan_hybrid_tiny_nvimagenetv2.pth.tar",
            30046459,
        ),
    ],
}


def _load() -> dict:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def _flatten(inventory: dict) -> list[tuple[str, dict]]:
    return [
        (resource["resource"], record)
        for resource in inventory["resources"]
        for record in resource["records"]
    ]


def test_complete_official_version_member_size_projection() -> None:
    inventory = _load()
    resources = inventory["resources"]

    assert inventory["schema_version"] == 1
    assert [resource["resource"] for resource in resources] == RESOURCE_ORDER
    assert set(EXPECTED) == set(RESOURCE_ORDER)

    for resource in resources:
        expected = EXPECTED[resource["resource"]]
        actual = [
            (record["version"], record["member"], record["size_bytes"])
            for record in resource["records"]
        ]
        assert actual == expected
        assert resource["version_count"] == len(expected)
        assert resource["member_count"] == len(expected)
        assert resource["total_size_bytes"] == sum(row[2] for row in expected)

    assert sum(len(rows) for rows in EXPECTED.values()) == 35
    assert sum(row[2] for rows in EXPECTED.values() for row in rows) == 15621218789


def test_source_order_is_explicit_and_stable() -> None:
    for resource in _load()["resources"]:
        assert [record["source_order"] for record in resource["records"]] == list(
            range(resource["version_count"])
        )


def test_exact_identities_are_unique_and_have_no_signed_url_material() -> None:
    inventory = _load()
    rows = _flatten(inventory)
    identities = [record["immutable_identity"] for _, record in rows]
    tuples = [
        (resource, record["version"], record["member"]) for resource, record in rows
    ]

    assert len(identities) == len(set(identities)) == 35
    assert len(tuples) == len(set(tuples)) == 35

    for resource, record in rows:
        assert record["immutable_identity"] == (
            f"ngc://nvidia/tao/{resource}:{record['version']}#{record['member']}"
        )
        assert record["size_bytes"] > 0

    serialized = json.dumps(inventory).lower()
    forbidden = (
        "authorization",
        "bearer ",
        "ngc_key",
        "x-amz-",
        "x-goog-",
        "signature=",
        "token=",
        '"requestid":',
    )
    assert all(value not in serialized for value in forbidden)

    for resource in inventory["resources"]:
        endpoint = urlsplit(resource["metadata_endpoint"])
        assert endpoint.scheme == "https"
        assert endpoint.netloc == "api.ngc.nvidia.com"
        assert endpoint.query == ""
        assert endpoint.fragment == ""


def test_detector_backbone_and_deployable_roles_map_to_runtime_targets() -> None:
    inventory = _load()
    for resource, record in _flatten(inventory):
        resolution = record["resolution"]
        assert resolution["height"] > 0
        assert resolution["width"] > 0

        if resource in {"pretrained_dino_coco", "dino_with_fm_backbone"}:
            assert record["artifact_role"].startswith("full_detector_")
            assert resolution["scope"] == "detector_input"
            if record["artifact_role"] == "full_detector_trainable":
                assert record["checkpoint_target"] == "train.pretrained_model_path"
                assert record["automl_training_artifact"] is True
            else:
                assert record["checkpoint_target"] is None
                assert record["automl_training_artifact"] is False
        else:
            assert record["artifact_role"] == "backbone_only"
            assert record["checkpoint_target"] == "model.pretrained_backbone_path"
            assert record["automl_training_artifact"] is True
            assert resolution["scope"] == "backbone_pretraining"

    by_identity = {
        record["immutable_identity"]: record for _, record in _flatten(inventory)
    }
    fm = by_identity[
        "ngc://nvidia/tao/dino_with_fm_backbone:trainable_v1.0"
        "#dino_nvdinov2_518_1536_coco_e36.pth"
    ]
    assert fm["runtime_backbone"] == "vit_large_dinov2"
    assert fm["resolution"] == {
        "scope": "detector_input",
        "height": 1536,
        "width": 1536,
    }


def test_gc_vit_exclusion_is_static_and_bound_to_pinned_tao71_source() -> None:
    inventory = _load()
    rows = _flatten(inventory)
    gc_vit = [
        record for _, record in rows if record["runtime_backbone"].startswith("gc_vit")
    ]
    non_gc_vit = [
        record
        for _, record in rows
        if not record["runtime_backbone"].startswith("gc_vit")
    ]

    assert len(gc_vit) == 14
    assert all(
        record["runtime_status"] == "excluded_static_tao71_gcvit"
        for record in gc_vit
    )
    assert all(
        record["runtime_status"] != "excluded_static_tao71_gcvit"
        for record in non_gc_vit
    )

    evidence = inventory["pinned_tao71_static_runtime_evidence"]
    assert evidence["dynamic_model_execution_performed"] is False
    assert evidence["container_sha256"] == (
        "0dfaf824025b69e66c0c239ec6e11e64f6b2eecdcd42b41f9226e14f1348568e"
    )
    assert evidence["source_member_sha256"] == (
        "4be339eb5791d168cf295e8ebb4bb2c32439399577848e9d4c4731f799e012c7"
    )
    assert evidence["facts"] == {
        "gc_vit_import_present": False,
        "gc_vit_dictionary_present": False,
        "gc_vit_construction_branch_present": False,
        "unknown_backbone_falls_through_to_not_implemented": True,
    }


def test_canonical_records_are_hashable_and_hash_matches() -> None:
    inventory = _load()
    records = [record for _, record in _flatten(inventory)]
    canonical = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    assert hashlib.sha256(canonical).hexdigest() == (
        inventory["normalization"]["records_sha256"]
    )
    assert inventory["normalization"]["records_sha256"] == (
        "715942c92b7ee5523cc421621913e7715cdd1abe9713240ccfb70d117d133ea7"
    )


def test_catalog_labels_never_create_nonexistent_registry_versions() -> None:
    inventory = _load()
    discrepancies = inventory["catalog_discrepancies"]
    absent = [
        item
        for item in discrepancies
        if item.get("official_versions_api_present") is False
    ]
    assert absent == [
        {
            "resource": "pretrained_dino_coco",
            "model_card_label": "dino_fan_large_deployable_v1.0",
            "official_versions_api_present": False,
            "handling": "not_inventoried_without_an_exact_registry_version",
        }
    ]

    all_versions = {
        (resource, record["version"]) for resource, record in _flatten(inventory)
    }
    assert (
        "pretrained_dino_coco",
        "dino_fan_large_deployable_v1.0",
    ) not in all_versions
