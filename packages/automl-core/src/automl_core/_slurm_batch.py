# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone batch supervisor; copied beside _worker.py on shared storage."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

if __package__:
    from automl_core._submission import gate
    from automl_core._worker import atomic, stop_child
else:
    from _submission import gate
    from _worker import atomic, stop_child


def main():
    root = Path(sys.argv[1]).resolve()
    intent = json.loads((root / "_slurm.json").read_text())
    job_id = os.environ["SLURM_JOB_ID"]
    if not job_id.isdecimal():
        raise ValueError("Invalid SLURM job identity")
    if intent.get("submission_guard"):
        # Claim before touching execution/worker receipts or starting the step.
        # Rejected late/duplicate jobs cannot overwrite an accepted job's files.
        try:
            accepted = (
                intent.get("owner_uid") == os.getuid()
                and gate(root, job_id=job_id)["changed"]
            )
        except (OSError, ValueError):
            accepted = False
        if not accepted:
            atomic(
                root / f"_submission_rejected-{job_id}.json",
                {
                    "slurm_job_id": job_id,
                    "state": "CANCELED",
                    "reason": "submission gate closed, claimed, or unavailable",
                    "finished_at": time.time(),
                },
            )
            return 1
    started = time.time()
    receipt = {"slurm_job_id": job_id, "started_at": started, "state": "ERROR"}
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    child = None
    try:
        for filename, expected in intent.get("supporting_files", {}).items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                raise ValueError("Supporting file changed after review")
        if intent["deadline"] - started < intent["time_limit_seconds"]:
            raise ValueError(
                "Insufficient remaining experiment time for this allocation"
            )
        plan_path = root / "_execution.json"
        plan = json.loads(plan_path.read_text())
        plan.update(
            slurm_job_id=job_id,
            deadline=min(
                intent["deadline"],
                started + intent["time_limit_seconds"] - intent["cleanup_seconds"],
            ),
        )
        atomic(plan_path, plan)

        def canceled():
            return (
                stop
                or (root / "_cancel.json").exists()
                or (root.parent / "cancel.json").exists()
            )

        if canceled():
            receipt.update(state="CANCELED", reason="cancel requested before step")
        else:
            receipt.update(state="RUNNING", heartbeat_at=time.time())
            atomic(root / "_batch_receipt.json", receipt)
            child = subprocess.Popen(
                ["/bin/bash", str(root / "_slurm_step.sh")], start_new_session=True
            )
            while child.poll() is None:
                if time.time() - receipt["heartbeat_at"] >= 5:
                    receipt["heartbeat_at"] = time.time()
                    atomic(root / "_batch_receipt.json", receipt)
                acknowledged = (root / "_receipt.json").exists()
                timeout = time.time() >= plan["deadline"] or (
                    not acknowledged
                    and time.time() - started >= intent["startup_timeout_seconds"]
                )
                if canceled() or timeout:
                    receipt["cleanup_confirmed"] = stop_child(child)
                    receipt.update(
                        state="CANCELED" if canceled() else "ERROR",
                        reason="cancel requested"
                        if canceled()
                        else "allocation/startup timeout",
                    )
                    break
                time.sleep(0.2)
            else:
                worker_path = root / "_receipt.json"
                worker = (
                    json.loads(worker_path.read_text()) if worker_path.exists() else {}
                )
                valid = worker.get("slurm_job_id") == job_id
                state = worker.get("state") if valid else "ERROR"
                receipt.update(
                    state=state
                    if state in {"COMPLETE", "CANCELED"} and child.returncode == 0
                    else "ERROR",
                    child_exit_code=child.returncode,
                    reason="step exited; require scheduler and worker evidence",
                )
    except BaseException as exc:  # noqa: BLE001 -- preserve terminal evidence on interruption
        if child and child.poll() is None:
            receipt["cleanup_confirmed"] = stop_child(child)
        receipt.update(state="ERROR", reason=type(exc).__name__)
    finally:
        receipt["finished_at"] = time.time()
        atomic(root / "_batch_receipt.json", receipt)
    return 0 if receipt["state"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
