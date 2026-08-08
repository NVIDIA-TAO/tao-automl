from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tao_automl.ptm_registry import canonical_sha256

from .qualification_load_audit import (
    QualificationLoadAuditError,
    _sealed_json,
    _write_new_read_only,
    classify_log_observations,
    extract_log_observations,
)


def test_legacy_positive_load_allows_only_identical_rank_duplicates():
    path = "/lustre/ptms/city.ptm"
    line = (
        "Loaded 1 compatible SegFormer pretrained tensors from "
        f"{path}: ['backbone.weight']\n"
    )
    observations = extract_log_observations(line * 8)

    result = classify_log_observations(
        observations,
        checkpoint_path=path,
        checkpoint_target="train.pretrained_model_path",
    )

    assert result["ptm_load_success"] is True
    assert result["classification"] == "positive_compatible_tensor_load"
    assert result["loaded_tensor_count"] == 1
    assert result["unique_positive_observations"][0][
        "loaded_keyset_sha256"
    ] == hashlib.sha256(b"backbone.weight").hexdigest()
    assert result["positive_observation_occurrences"] == 8
    assert len(result["unique_positive_observations"]) == 1
    assert result["finite_metric_override_allowed"] is False


def test_all_missing_all_unexpected_backbone_prefix_is_ptm_load_failure():
    path = "/lustre/ptms/backbone.ptm"
    line = (
        f"Loaded pretrained weights from {path}\n"
        "_IncompatibleKeys(missing_keys=['cls_token', "
        "'patch_embed.backbone.stem.weight'], unexpected_keys="
        "['backbone.cls_token', 'backbone.patch_embed.backbone.stem.weight'])\n"
    )
    observations = extract_log_observations(line * 8)

    result = classify_log_observations(
        observations,
        checkpoint_path=path,
        checkpoint_target="model.backbone.pretrained_backbone_path",
    )

    assert result["ptm_load_success"] is False
    assert result["classification"] == (
        "all_missing_all_unexpected_backbone_prefix"
    )
    assert result["loaded_tensor_count"] == 0
    assert result["incompatible_observation_occurrences"] == 8
    assert len(result["unique_incompatible_observations"]) == 1
    incompatible = result["unique_incompatible_observations"][0]
    assert incompatible["missing_tensor_count"] == 2
    assert incompatible["unexpected_tensor_count"] == 2
    assert incompatible["all_unexpected_backbone_prefix"] is True
    assert incompatible["all_loadable_unexpected_backbone_prefix"] is True
    assert result["finite_metric_override_allowed"] is False


def test_backbone_prefix_failure_allows_only_exact_classifier_head_extras():
    path = "/lustre/ptms/backbone.ptm"
    text = (
        f"Loaded pretrained weights from {path}\n"
        "_IncompatibleKeys(missing_keys=['cls_token'], unexpected_keys="
        "['backbone.cls_token', 'head.fc.weight', 'head.fc.bias'])\n"
    )
    result = classify_log_observations(
        extract_log_observations(text),
        checkpoint_path=path,
        checkpoint_target="model.backbone.pretrained_backbone_path",
    )
    incompatible = result["unique_incompatible_observations"][0]
    assert result["classification"] == (
        "all_missing_all_unexpected_backbone_prefix"
    )
    assert result["ptm_load_success"] is False
    assert incompatible["all_unexpected_backbone_prefix"] is False
    assert incompatible["all_loadable_unexpected_backbone_prefix"] is True
    assert incompatible["unexpected_backbone_prefix_count"] == 1
    assert incompatible["allowlisted_classifier_unexpected_count"] == 2


@pytest.mark.parametrize(
    "text",
    [
        (
            "Loaded 1 compatible SegFormer pretrained tensors from "
            "/lustre/ptms/other.ptm: ['x']\n"
        ),
        (
            "Loaded 1 compatible SegFormer pretrained tensors from "
            "/lustre/ptms/backbone.ptm: ['x']\n"
        ),
        (
            "Loaded pretrained weights from /lustre/ptms/backbone.ptm\n"
            "_IncompatibleKeys(missing_keys=['x'], "
            "unexpected_keys=['not_backbone.x'])\n"
        ),
        (
            "Loaded 1 compatible SegFormer backbone pretrained tensors from "
            "/lustre/ptms/backbone.ptm: ['x']\n"
            "_IncompatibleKeys(missing_keys=['x'], "
            "unexpected_keys=['backbone.x'])\n"
        ),
    ],
)
def test_missing_mismatched_or_conflicting_load_evidence_fails_closed(text):
    result = classify_log_observations(
        extract_log_observations(text),
        checkpoint_path="/lustre/ptms/backbone.ptm",
        checkpoint_target="model.backbone.pretrained_backbone_path",
    )
    assert result["ptm_load_success"] is False
    assert result["classification"] == (
        "load_evidence_missing_mismatched_or_ambiguous"
    )


def test_sealed_json_binds_whole_and_internal_hashes(tmp_path: Path):
    value = {"schema_version": 1, "payload": {"terminal": True}}
    value["evidence_sha256"] = canonical_sha256(value)
    path = tmp_path / "completion.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    whole = hashlib.sha256(path.read_bytes()).hexdigest()

    document, identity = _sealed_json(
        path,
        internal_key="evidence_sha256",
        expected_whole_sha256=whole,
    )
    assert document == value
    assert identity["whole_file_sha256"] == whole
    assert identity["internal_sha256"] == value["evidence_sha256"]

    with pytest.raises(QualificationLoadAuditError, match="whole-file SHA"):
        _sealed_json(
            path,
            internal_key="evidence_sha256",
            expected_whole_sha256="f" * 64,
        )


def test_audit_output_is_new_and_read_only(tmp_path: Path):
    output = tmp_path / "v4-load-audit.json"
    value = {"schema_version": 1, "audit_sha256": "a" * 64}
    _write_new_read_only(output, value)
    assert json.loads(output.read_text(encoding="utf-8")) == value
    assert output.stat().st_mode & 0o222 == 0

    with pytest.raises(QualificationLoadAuditError, match="refusing to overwrite"):
        _write_new_read_only(output, value)
