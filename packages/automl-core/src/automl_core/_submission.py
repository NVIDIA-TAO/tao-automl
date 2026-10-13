# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared-filesystem launch fence; staged with the standalone SLURM supervisor."""

import fcntl
import json
import time

if __package__:
    from automl_core._worker import atomic
else:
    from _worker import atomic


def gate(root, *, job_id=None, close=False):
    """Atomically claim or close an attempt, never waiting indefinitely on a lock.

    A claimed attempt cannot be closed/replaced. A closed attempt cannot run a
    workload, even if its original sbatch RPC completes after reconciliation.
    This requires shared POSIX flock semantics on controller and compute nodes.
    """
    with (root / "_submission.lock").open("a") as lock:
        until = time.monotonic() + 1
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= until:
                    raise
                time.sleep(min(0.05, max(0, until - time.monotonic())))
        path = root / "_submission_gate.json"
        saved = json.loads(path.read_text())
        if saved.get("state") not in {"open", "claimed", "closed"}:
            raise ValueError("Invalid submission gate")
        changed = saved["state"] == "open" and (job_id is not None or close)
        if changed:
            saved.update(state="closed" if close else "claimed", updated_at=time.time())
            if job_id is not None:
                saved["job_id"] = job_id
            atomic(path, saved)
        return {**saved, "changed": changed}
