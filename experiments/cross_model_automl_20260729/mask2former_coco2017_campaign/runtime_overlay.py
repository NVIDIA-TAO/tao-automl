"""Sealed TAO PyTorch runtime overlay contract for Mask2Former.

The overlay is prepared inside each pinned SQSH job and injected through
``PYTHONPATH``.  It never mutates the package installed in the image.
"""

from __future__ import annotations

import copy
import re
import shlex
from collections.abc import Mapping
from typing import Any


OVERLAY_DIRECTORY = (
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
    "tao-pytorch-overlays/mask2former-instance-ap/"
    "c2e86fe1646ebe89fc280083797dcc544ce88322"
)
SUCCESSOR_OVERLAY_DIRECTORY = (
    "/lustre/fsw/portfolios/edgeai/projects/"
    "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/"
    "tao-pytorch-overlays/mask2former-instance-ap/"
    "c2e86fe1646ebe89fc280083797dcc544ce88322"
)
ARCHIVE_NAME = "tao-pytorch-mask2former-instance-ap-c2e86fe1646e.tar"
ARCHIVE_SHA256 = (
    "c395474592d557e0179066c1f99d5cb8f352e10e501621d57043782440dea8c2"
)
ARCHIVE_SIZE_BYTES = 92160
SOURCE_COMMIT = "c2e86fe1646ebe89fc280083797dcc544ce88322"
INSTALLER_NAME = "install_mask2former_source_overlay.py"
INSTALLER_SHA256 = (
    "8946340453c0a902e175208a82d382f63c81c5e2d858a09f5678fcc585666042"
)
INSTALLER_SIZE_BYTES = 7774
PYTHONPATH_ROOT = (
    "/tmp/tao-pytorch-mask2former-instance-ap-"
    "c2e86fe1646ebe89fc280083797dcc544ce88322"
)
RECEIPT_PATH = (
    "/tmp/mask2former-instance-ap-overlay-"
    "c2e86fe1646ebe89fc280083797dcc544ce88322.json"
)
RUNTIME_MEMBERS = {
    "nvidia_tao_pytorch/cv/mask2former/dataloader/datasets.py": (
        "3b60c5bee0d89156833abb6db854d121b9680d5ad4119b8f54e9e545921447b8"
    ),
    "nvidia_tao_pytorch/cv/mask2former/dataloader/pl_data_module.py": (
        "240a67b63d418ebf4d0047f680639d1db0f38d1eab070d9fd9ea5f375040e47b"
    ),
    "nvidia_tao_pytorch/cv/mask2former/model/pl_model.py": (
        "5d264ed878eebf67b4b914f59ca1482f9b64035bd6606af3eafc92c04d8f201f"
    ),
    "nvidia_tao_pytorch/cv/mask2former/utils/task_metrics.py": (
        "18539f50109027bb6d7d4d318ed41c3012fec6c2f3fbfd465539c0ab5ce57d2e"
    ),
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RuntimeOverlayError(ValueError):
    """The Mask2Former runtime overlay contract is invalid."""


def contract_record() -> dict[str, Any]:
    """Return the one immutable runtime overlay accepted by this campaign."""
    return {
        "schema_version": 1,
        "kind": "tao_pytorch_source_overlay",
        "model": "mask2former",
        "source_commit": SOURCE_COMMIT,
        "directory": OVERLAY_DIRECTORY,
        "archive": {
            "path": f"{OVERLAY_DIRECTORY}/{ARCHIVE_NAME}",
            "sha256": ARCHIVE_SHA256,
            "size_bytes": ARCHIVE_SIZE_BYTES,
        },
        "installer": {
            "path": f"{OVERLAY_DIRECTORY}/{INSTALLER_NAME}",
            "sha256": INSTALLER_SHA256,
            "size_bytes": INSTALLER_SIZE_BYTES,
        },
        "runtime_members": copy.deepcopy(RUNTIME_MEMBERS),
        "injection": {
            "mechanism": "PYTHONPATH",
            "pythonpath_root": PYTHONPATH_ROOT,
            "receipt_path": RECEIPT_PATH,
            "installed_package_mutated": False,
        },
        "remote_read_only_required": True,
    }


def successor_contract_record() -> dict[str, Any]:
    """Return the identical overlay staged in the user's project namespace."""
    value = contract_record()
    value["directory"] = SUCCESSOR_OVERLAY_DIRECTORY
    value["archive"]["path"] = (
        f"{SUCCESSOR_OVERLAY_DIRECTORY}/{ARCHIVE_NAME}"
    )
    value["installer"]["path"] = (
        f"{SUCCESSOR_OVERLAY_DIRECTORY}/{INSTALLER_NAME}"
    )
    return value


def validate_contract_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Reject any unsealed archive, installer, commit, or injection policy."""
    if not isinstance(record, Mapping):
        raise RuntimeOverlayError("runtime overlay must be a mapping")
    value = copy.deepcopy(dict(record))
    if value not in (contract_record(), successor_contract_record()):
        raise RuntimeOverlayError(
            "Mask2Former runtime overlay differs from the sealed contract"
        )
    for label in ("archive", "installer"):
        digest = value[label]["sha256"]
        if _SHA256_RE.fullmatch(digest) is None:
            raise RuntimeOverlayError(
                f"runtime overlay {label} digest is not lowercase SHA-256"
            )
    return value


def wrap_command(command: str, record: Mapping[str, Any]) -> str:
    """Prefix one in-container command with verified PYTHONPATH preparation."""
    if not isinstance(command, str) or not command.strip():
        raise RuntimeOverlayError("wrapped runtime command must be non-empty")
    value = validate_contract_record(record)
    archive = value["archive"]
    installer = value["installer"]
    injection = value["injection"]
    verify_installer = " ".join(
        [
            "test",
            f"\"$(stat -c %s {shlex.quote(installer['path'])})\"",
            "=",
            shlex.quote(str(installer["size_bytes"])),
            "&& test",
            "\"$(sha256sum "
            f"{shlex.quote(installer['path'])} | cut -d ' ' -f1)\"",
            "=",
            shlex.quote(installer["sha256"]),
        ]
    )
    prepare = " ".join(
        [
            "python",
            shlex.quote(installer["path"]),
            "--archive",
            shlex.quote(archive["path"]),
            "--expected-sha256",
            shlex.quote(archive["sha256"]),
            "--expected-source-commit",
            shlex.quote(value["source_commit"]),
            "--pythonpath-root",
            shlex.quote(injection["pythonpath_root"]),
            "--receipt",
            shlex.quote(injection["receipt_path"]),
        ]
    )
    return " ".join(
        [
            verify_installer,
            "&&",
            prepare,
            "&& export PYTHONPATH="
            f"{shlex.quote(injection['pythonpath_root'])}:"
            "\"$(printenv PYTHONPATH || true)\"",
            "&&",
            command,
        ]
    )
