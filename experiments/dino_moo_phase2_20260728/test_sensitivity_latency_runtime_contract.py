"""Tests for the frozen sensitivity-latency runtime version contract."""

from __future__ import annotations

import pytest

import sensitivity_latency_aggregate
import sensitivity_latency_block_runner


@pytest.mark.parametrize(
    "version",
    [
        "2.11.0",
        "2.11.0a0+a6c236b9fd.nv26.03.46836102",
        "2.11.0+cu132",
    ],
)
def test_torch_build_suffixes_preserve_release_identity(version: str) -> None:
    assert (
        sensitivity_latency_block_runner.major_minor_patch(version, "torch")
        == "2.11.0"
    )
    assert (
        sensitivity_latency_aggregate.major_minor_patch(version, "torch")
        == "2.11.0"
    )


@pytest.mark.parametrize(
    ("module", "error"),
    [
        (sensitivity_latency_block_runner, RuntimeError),
        (sensitivity_latency_aggregate, ValueError),
    ],
)
def test_invalid_torch_version_is_rejected(module: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="major.minor.patch"):
        module.major_minor_patch("nightly", "torch")
