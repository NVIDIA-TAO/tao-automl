# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AutoML contracts and execution independent of any model repository."""

import hashlib
from pathlib import Path

# Keep the v1 development version fixed until an explicit release.
__version__ = "1.0.0.dev0"


def build_fingerprint():
    """Bind resume to runtime source bytes, independently of the release version.

    Relative names and content hashes are stable across source checkouts and
    installed wheels. Bytecode, docs and installation paths are not identity.
    This is a compatibility guard, not a signature or a dependency inventory.
    """
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
