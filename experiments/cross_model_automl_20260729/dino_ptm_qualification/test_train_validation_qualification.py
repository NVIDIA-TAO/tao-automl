# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the second-stage DINO PTM qualification launcher."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import train_validation_qualification as qualification


def _arguments(tmp_path: Path, *, resume: bool = False) -> Namespace:
    return Namespace(
        output_dir=str(tmp_path / "evidence"),
        cache_dir=str(tmp_path / "cache"),
        runtime_results_dir=str(tmp_path / "runtime"),
        cpu_qualification_dir=str(tmp_path / "cpu-evidence"),
        cpu_cache_dir=str(tmp_path / "cpu-cache"),
        voc_manifest=str(tmp_path / "manifest.json"),
        voc_root=str(tmp_path / "voc"),
        container_user="1000:1000",
        registry_path=None,
        seed=271828,
        poll_interval_seconds=0.01,
        max_polls=3,
        resume=resume,
    )


def test_fresh_qualification_wires_real_data_smoke_without_registry_mutation(
    monkeypatch,
    tmp_path,
):
    calls = {}
    voc = SimpleNamespace()

    monkeypatch.setattr(
        qualification,
        "_verify_local_runtime_image",
        lambda: calls.setdefault("image_verified", True),
    )
    monkeypatch.setattr(
        qualification,
        "load_verified_qualification_completion",
        lambda **kwargs: (
            calls.setdefault("cpu_arguments", kwargs),
            {
                "completion_sha256": "c" * 64,
                "report": {
                    "prepared": [
                        {"checkpoint_id": "dino.fixture.resnet50"}
                    ]
                },
            },
        )[1],
    )
    monkeypatch.setattr(
        qualification,
        "collect_voc_real_data_integrity",
        lambda **kwargs: (
            calls.setdefault("voc_arguments", kwargs),
            voc,
        )[1],
    )

    class Smoke:
        def __init__(self, **kwargs):
            calls["smoke_arguments"] = kwargs

    monkeypatch.setattr(qualification, "DINOStandardDryRunLoadSmoke", Smoke)

    def run(**kwargs):
        calls["qualification_arguments"] = kwargs
        return {
            "completion_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "report": {"prepared": [], "exclusions": []},
        }

    monkeypatch.setattr(qualification, "run_dino_ptm_qualification", run)
    args = _arguments(tmp_path)

    completion = qualification.run_train_validation_qualification(args)

    assert completion["completion_sha256"] == "a" * 64
    assert calls["image_verified"] is True
    assert calls["smoke_arguments"]["voc"] is voc
    assert calls["smoke_arguments"]["sdk"] is None
    assert calls["qualification_arguments"]["resume"] is False
    assert calls["qualification_arguments"]["docker_load_smoke"].__class__ is Smoke
    config = calls["qualification_arguments"]["configuration"]
    assert config.checkpoint_ids == ("dino.fixture.resnet50",)
    assert config.upstream_completion_sha256 == "c" * 64


def test_resume_reconstructs_identity_without_image_or_docker_sdk(
    monkeypatch,
    tmp_path,
):
    calls = {}
    voc = SimpleNamespace()
    monkeypatch.setattr(
        qualification,
        "_verify_local_runtime_image",
        lambda: (_ for _ in ()).throw(AssertionError("image inspected")),
    )
    monkeypatch.setattr(
        qualification,
        "load_verified_qualification_completion",
        lambda **_kwargs: {
            "completion_sha256": "c" * 64,
            "report": {
                "prepared": [{"checkpoint_id": "dino.fixture.resnet50"}]
            },
        },
    )
    monkeypatch.setattr(
        qualification,
        "collect_voc_real_data_integrity",
        lambda **_kwargs: voc,
    )

    class Smoke:
        def __init__(self, **kwargs):
            calls["smoke_arguments"] = kwargs

    monkeypatch.setattr(qualification, "DINOStandardDryRunLoadSmoke", Smoke)
    monkeypatch.setattr(
        qualification,
        "run_dino_ptm_qualification",
        lambda **kwargs: (
            calls.setdefault("qualification_arguments", kwargs),
            {
                "completion_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
                "report": {"prepared": [], "exclusions": []},
            },
        )[1],
    )

    qualification.run_train_validation_qualification(
        _arguments(tmp_path, resume=True)
    )

    assert calls["smoke_arguments"]["sdk"].__class__ is object
    assert calls["smoke_arguments"]["entrypoint_builder"].__class__ is object
    assert calls["qualification_arguments"]["resume"] is True
