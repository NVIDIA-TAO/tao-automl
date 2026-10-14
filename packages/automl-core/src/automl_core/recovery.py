# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conservative execution health and attempt bookkeeping, independent of samplers."""

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


def read_receipt(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


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
