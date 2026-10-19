# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conservative execution health and attempt bookkeeping, independent of samplers."""

import errno
import json
import time


def validate_policy(policy, execution, plan=None):
    if policy.max_retries_total:
        if execution.kind != "slurm":
            raise ValueError(
                "Automatic infrastructure classification currently requires SLURM"
            )
        settings = execution.slurm
        if settings.record_open and (
            not settings.record_retry_open
            or not any("{retry_of}" in arg for arg in settings.record_retry_open)
        ):
            raise ValueError(
                "Retry-enabled record hooks require record_retry_open with {retry_of}"
            )
    if (
        plan is not None
        and policy.progress_timeout_seconds
        and set(plan.progress_files) != set(range(len(plan.commands)))
    ):
        raise ValueError(
            "Progress timeout requires explicit watches for every workload command"
        )


def attempt_root(root, trial):
    path = root / trial.get("attempt_id", trial["id"])
    if path.resolve().parent != root.resolve():
        raise ValueError("Invalid recorded attempt directory")
    return path


def read_receipt(path, *, missing_ok=True):
    """Read fresh JSON, tolerating bounded transient filesystem read failures.

    Some network filesystems invalidate an open reader when an atomic writer
    replaces the pathname. Reopen on each retry; never reuse the failed handle,
    cached state, or an empty receipt as a substitute for unavailable evidence.
    Only reads repeat, with at most five attempts and 0.75 seconds of backoff.
    Persistent I/O failures, permissions and malformed JSON still fail closed.
    """
    for attempt in range(5):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            if missing_ok:
                return {}
            raise
        except OSError as exc:
            if (
                exc.errno not in {errno.EIO, errno.ESTALE, errno.ETIMEDOUT}
                or attempt == 4
            ):
                raise
            time.sleep(0.05 * 2**attempt)


def health(root, policy, job_id=None):
    """Suspicion requests cancellation/reconciliation, never proves job termination."""
    now = time.time()
    worker = read_receipt(root / "_receipt.json")
    batch = read_receipt(root / "_batch_receipt.json") if job_id else {}
    for receipt in (worker, batch):
        if receipt and job_id and receipt.get("slurm_job_id") != job_id:
            raise ValueError("Health receipt belongs to another SLURM job")
    issue = None
    if any(r.get("state") in {"ERROR", "CANCELED"} for r in (worker, batch)):
        issue = "terminal_worker_active_allocation"
    elif worker.get("workload_failed"):
        issue = "structured_workload_failure"
    elif policy.heartbeat_timeout_seconds and any(
        r.get("state") == "RUNNING"
        and now - r.get("heartbeat_at", r.get("started_at", now))
        >= policy.heartbeat_timeout_seconds
        for r in (worker, batch)
    ):
        issue = "heartbeat_stale"
    elif policy.heartbeat_timeout_seconds and any(
        r.get("state") == "COMPLETE"
        and now - r.get("finished_at", now) >= policy.heartbeat_timeout_seconds
        for r in (worker, batch)
    ):
        issue = "completed_worker_active_allocation"
    elif (
        policy.progress_timeout_seconds
        and worker.get("state") == "RUNNING"
        and worker.get("progress_watched")
        and now - worker.get("progress_at", now) >= policy.progress_timeout_seconds
    ):
        issue = "progress_stalled"
    if issue:
        evidence = {"reason": issue, "retryable": False, "observed_at": now}
        if issue == "heartbeat_stale":
            evidence["stale_heartbeats"] = [
                {
                    "publisher": name,
                    "heartbeat_at": receipt.get(
                        "heartbeat_at", receipt.get("started_at")
                    ),
                    "age_seconds": now
                    - receipt.get("heartbeat_at", receipt.get("started_at", now)),
                }
                for name, receipt in (("worker", worker), ("batch", batch))
                if receipt.get("state") == "RUNNING"
                and now - receipt.get("heartbeat_at", receipt.get("started_at", now))
                >= policy.heartbeat_timeout_seconds
            ]
        return evidence
    return None
