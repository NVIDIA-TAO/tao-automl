# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone, shared-volume worker guard for managed container jobs."""

import json
import sys
import time
from pathlib import Path

from _submission import gate
from _worker import atomic, main

if __name__ == "__main__":
    root = Path(sys.argv[1]).parent
    token = sys.argv[2]
    intent = json.loads((root / "_remote.json").read_text())
    if intent["token"] != token:
        raise SystemExit("Managed launch identity mismatch")
    # Claim exactly once, including a platform-created replacement pod. A lock
    # failure or a wrong/missing volume fails closed before any workload starts.
    claim = gate(root, job_id=token)
    if not claim["changed"]:
        raise SystemExit("Managed attempt is fenced or already claimed")
    if time.time() >= intent["deadline"] or (root / "_cancel.json").exists():
        atomic(
            root / "_receipt.json", {"state": "CANCELED", "finished_at": time.time()}
        )
        raise SystemExit(1)
    sys.argv = sys.argv[:2]
    main()
    receipt = json.loads((root / "_receipt.json").read_text())
    raise SystemExit(0 if receipt.get("state") == "COMPLETE" else 1)
