# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone stdlib worker: survives controller restarts; also runs in containers."""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def atomic(path, value):
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def process_token(pid):
    try:
        # comm can contain spaces or parentheses; starttime is field 22.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def host_identity():
    return {
        "hostname": socket.gethostname(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def stop_child(child):
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            return False  # An uninterruptible child needs native reconciliation.
    except ProcessLookupError:
        pass
    return True


def progress(root, names):
    """Small structured files only; no raw log/error text enters receipts."""
    tokens, failed = [], False
    for name in names:
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Progress path escaped the attempt directory")
        try:
            stat = path.stat()
            tokens.append((name, stat.st_size, stat.st_mtime_ns))
            with path.open("rb") as stream:
                stream.seek(max(0, stat.st_size - 16384))
                lines = stream.read(16384).splitlines()
            # An incomplete last line must not hide the last complete event.
            for line in reversed(lines):
                try:
                    event = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                if isinstance(event, dict):
                    failed |= event.get("status") in {"FAILURE", "ERROR"}
                    break
        except FileNotFoundError:
            tokens.append((name, None, None))
    return tokens, failed


def main():
    plan_path = Path(sys.argv[1])
    root = plan_path.parent
    plan = json.loads(plan_path.read_text())
    if "runtime_timeout_seconds" in plan:
        # SLURM queue/startup time is separate from the workload runtime budget.
        plan["deadline"] = min(
            plan["deadline"], time.time() + plan["runtime_timeout_seconds"]
        )
    os.environ.update(plan.get("environment", {}))
    canceled = False

    def request_stop(signum, frame):
        nonlocal canceled
        canceled = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    receipt = {
        "host": host_identity(),
        "pid": os.getpid(),
        "token": process_token(os.getpid()),
        "state": "RUNNING",
        "started_at": time.time(),
        "heartbeat_at": time.time(),
    }
    if "slurm_job_id" in plan:
        receipt["slurm_job_id"] = plan["slurm_job_id"]
    atomic(root / "_receipt.json", receipt)
    child = None
    try:
        with (root / "trial.log").open("ab", buffering=0) as log:
            for index, command in enumerate(plan["commands"]):
                canceled = (
                    canceled
                    or (root / "_cancel.json").exists()
                    or (root.parent / "cancel.json").exists()
                )
                if canceled or time.time() >= plan["deadline"]:
                    receipt["state"] = "CANCELED" if canceled else "ERROR"
                    receipt["reason"] = "cancel requested" if canceled else "timeout"
                    break
                child = subprocess.Popen(
                    command,
                    cwd=plan["cwd"],
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
                receipt.update(
                    child_pid=child.pid,
                    child_token=process_token(child.pid),
                    command_index=index,
                    progress_at=time.time(),
                    heartbeat_at=time.time(),
                )
                names = plan.get("progress_files", {}).get(str(index), [])
                receipt["progress_watched"] = bool(names)
                previous = None
                atomic(root / "_receipt.json", receipt)
                while child.poll() is None:
                    if time.time() - receipt["heartbeat_at"] >= 5:
                        token, failed = progress(root, names)
                        if token != previous:
                            receipt["progress_at"] = time.time()
                            previous = token
                        receipt["heartbeat_at"] = time.time()
                        receipt["workload_failed"] = failed
                        atomic(root / "_receipt.json", receipt)
                    canceled = (
                        canceled
                        or (root / "_cancel.json").exists()
                        or (root.parent / "cancel.json").exists()
                    )
                    if canceled or time.time() >= plan["deadline"]:
                        receipt["cleanup_confirmed"] = stop_child(child)
                        receipt["state"] = "CANCELED" if canceled else "ERROR"
                        receipt["reason"] = (
                            "cancel requested" if canceled else "timeout"
                        )
                        break
                    time.sleep(0.1)
                if receipt["state"] != "RUNNING":
                    break
                if child.returncode:
                    receipt.update(
                        state="ERROR",
                        exit_code=child.returncode,
                        reason="command failed",
                    )
                    break
            else:
                receipt.update(state="COMPLETE", exit_code=0)
    except BaseException as exc:  # noqa: BLE001 -- persist terminal evidence even for worker interruption
        if child and child.poll() is None:
            receipt["cleanup_confirmed"] = stop_child(child)
        # Exception text could contain credentials supplied by a user command.
        receipt.update(state="ERROR", reason=type(exc).__name__)
        if isinstance(exc, OSError):
            receipt["error_errno"] = exc.errno  # Numeric only; never exception text.
    finally:
        receipt["finished_at"] = time.time()
        atomic(root / "_receipt.json", receipt)


if __name__ == "__main__":
    main()
